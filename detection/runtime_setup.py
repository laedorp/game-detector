"""Plan and, for source installs, fetch the matching ONNX Runtime provider.

The application bundles everything it needs for Intel CPUs, Intel graphics, and
Intel NPUs.  AMD and NVIDIA GPUs are different: ONNX Runtime publishes a
*separate* wheel per vendor, all four of which install the same ``onnxruntime``
module and therefore cannot be shipped together.  Exactly one can be present, so
the correct one has to be chosen after the hardware is known.

Release builds carry exactly one provider and are published as separate
hardware-specific bundles. A frozen PyInstaller executable cannot safely act as
``python -m pip``, so mutation is rejected there. For a source checkout, this
module can resolve the right distribution, install it beside the user's settings,
and activate that directory for the app and its child processes.

The vendor *drivers* (ROCm, CUDA, the Intel compute runtime) are deliberately
out of scope.  They are multi-gigabyte system components that require
administrator rights and can leave a machine unbootable when they go wrong, so
this module reports what is missing and links the vendor installer instead of
silently changing a system.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys


# Wheel names as published on PyPI.  These all provide the ``onnxruntime``
# module, which is why only one may be installed at a time.
DISTRIBUTION_CPU = "onnxruntime"
DISTRIBUTION_NVIDIA = "onnxruntime-gpu"
DISTRIBUTION_DIRECTML = "onnxruntime-directml"
DISTRIBUTION_ROCM = "onnxruntime-rocm"

RUNTIME_REQUIREMENTS = {
    DISTRIBUTION_CPU: "onnxruntime==1.28.0",
    DISTRIBUTION_NVIDIA: "onnxruntime-gpu==1.28.0",
    DISTRIBUTION_DIRECTML: "onnxruntime-directml==1.24.4",
    DISTRIBUTION_ROCM: "onnxruntime-rocm==1.22.2.post3",
}
RUNTIME_ROOT_ENV = "PROAIM_RUNTIME_ROOT"

CONFLICTING_DISTRIBUTIONS = (
    DISTRIBUTION_CPU,
    DISTRIBUTION_NVIDIA,
    DISTRIBUTION_DIRECTML,
    DISTRIBUTION_ROCM,
)


# ``None`` is a meaningful answer from :func:`installed_distribution` -- it
# means nothing is installed -- so a distinct sentinel is needed to express
# "caller did not say, go and look".
_PROBE = object()


class RuntimeSetupError(RuntimeError):
    """Raised when the required runtime cannot be installed."""


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    """What should be installed, and what the user must still do themselves."""

    distribution: str
    reason: str
    # Set when a vendor driver is also required.  A wheel alone does not make a
    # GPU usable, and saying so up front avoids an install that appears to
    # succeed and then falls back to the CPU.
    driver_note: str = ""

    @property
    def needs_driver(self) -> bool:
        return bool(self.driver_note)


def install_root(settings_directory: Path) -> Path:
    """Return the writable directory that holds fetched runtime packages."""

    return settings_directory / "runtime"


def default_settings_directory() -> Path:
    """Return ProAim's per-user configuration directory without importing a GUI."""

    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / "ProAim"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ProAim"
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "proaim"


def plan_for(vendor: str, system: str) -> RuntimePlan:
    """Choose the distribution for a GPU vendor on a given operating system."""

    normalized_vendor = str(vendor).strip().lower()
    normalized_system = str(system).strip().lower()

    if normalized_vendor == "nvidia":
        if normalized_system == "windows":
            return RuntimePlan(
                DISTRIBUTION_NVIDIA,
                "NVIDIA GPUs run best through CUDA/TensorRT on Windows",
                driver_note=(
                    "Use a recent NVIDIA driver. If CUDA providers are unavailable, "
                    "the DirectML build is a compatibility fallback."
                ),
            )
        return RuntimePlan(
            DISTRIBUTION_NVIDIA,
            "NVIDIA GPUs run through the CUDA execution provider on Linux",
            driver_note=(
                "Install the NVIDIA driver and a matching CUDA runtime; the "
                "wheel does not contain them."
            ),
        )

    if normalized_vendor == "amd":
        if normalized_system == "windows":
            return RuntimePlan(
                DISTRIBUTION_DIRECTML,
                "AMD GPUs run through DirectML on Windows",
            )
        return RuntimePlan(
            DISTRIBUTION_ROCM,
            "AMD GPUs run through the ROCm execution provider on Linux",
            driver_note=(
                "Install ROCm from AMD; RDNA2 cards such as the RX 6900/6950 "
                "series commonly also need HSA_OVERRIDE_GFX_VERSION=10.3.0."
            ),
        )

    return RuntimePlan(
        DISTRIBUTION_CPU,
        "no supported discrete GPU was detected; the CPU build is sufficient",
    )


def installed_distribution() -> str | None:
    """Return which ONNX Runtime distribution is already importable, if any."""

    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - importlib.metadata is standard
        return None
    for name in CONFLICTING_DISTRIBUTIONS:
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        return name
    return None


def _pip_command(
    target: Path,
    distribution: str,
    *,
    interpreter: str | Path | None = None,
) -> list[str]:
    """Build a source-environment installer command.

    A PyInstaller executable is not a general Python interpreter and cannot run
    ``-m pip``. Frozen builds therefore use prebuilt runtime-specific bundles;
    this helper is only for a real Python environment with a matching ABI.
    """

    executable = str(interpreter or sys.executable)
    return [
        executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(target),
        RUNTIME_REQUIREMENTS.get(distribution, distribution),
    ]


def ensure_runtime(
    plan: RuntimePlan,
    settings_directory: Path,
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess] | None = None,
    already_installed: str | None | object = _PROBE,
    interpreter: str | Path | None = None,
    frozen: bool | None = None,
) -> Path:
    """Install ``plan.distribution`` into the user's runtime directory.

    Returns the directory that was added to ``sys.path``.  An already-correct
    installation is left alone so that repeated runs are cheap and offline
    machines keep working.
    """

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if is_frozen:
        raise RuntimeSetupError(
            "Packaged ProAim cannot install Python execution providers in place. "
            "Download the ProAim bundle that matches this GPU, then restart the app."
        )

    current = installed_distribution() if already_installed is _PROBE else already_installed
    target = install_root(settings_directory)
    if current == plan.distribution:
        activate(target)
        return target

    if current is not None and current != plan.distribution:
        raise RuntimeSetupError(
            f"{current} is already installed but this machine needs "
            f"{plan.distribution}. They provide the same module and cannot be "
            f"installed together; remove {current} first."
        )

    target.mkdir(parents=True, exist_ok=True)
    command = _pip_command(target, plan.distribution, interpreter=interpreter)
    execute = runner or (
        lambda argv: subprocess.run(  # noqa: S603 - argv is built here, not user text
            list(argv), capture_output=True, text=True, check=False, timeout=1800
        )
    )
    try:
        completed = execute(command)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeSetupError(f"Could not run the installer: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        raise RuntimeSetupError(
            f"Installing {plan.distribution} failed: {tail}"
        )

    activate(target)
    return target


def activate(target: Path) -> bool:
    """Put a previously fetched runtime directory on the import path."""

    if not target.is_dir():
        return False
    text = str(target)
    if text not in sys.path:
        # Prepend so a fetched vendor build wins over any bundled CPU copy.
        sys.path.insert(0, text)
    os.environ[RUNTIME_ROOT_ENV] = text
    return True


def activate_configured_runtime(
    settings_directory: Path | None = None,
    *,
    frozen: bool | None = None,
) -> bool:
    """Activate a source-installed provider for this process and future children.

    Release bundles carry their provider inside the bundle. This path exists for
    source installations and deliberately does not download or mutate anything.
    """

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if is_frozen:
        # A release bundle carries one audited provider. Never let a stale
        # source-installed wheel override or mix with those frozen libraries.
        return False

    configured = os.environ.get(RUNTIME_ROOT_ENV, "").strip()
    target = (
        Path(configured).expanduser()
        if configured
        else install_root(settings_directory or default_settings_directory())
    )
    return activate(target)


def describe(plan: RuntimePlan, current: str | None) -> str:
    """Explain what will happen, for display before anything is downloaded."""

    lines = [f"Recommended package: {plan.distribution}", f"Reason: {plan.reason}"]
    if current == plan.distribution:
        lines.append("Status: already installed; nothing to download.")
    elif current is None:
        lines.append("Status: not installed; it will be downloaded on approval.")
    else:
        lines.append(
            f"Status: {current} is installed and conflicts with "
            f"{plan.distribution}; remove it first."
        )
    if plan.needs_driver:
        lines.append(f"Also required: {plan.driver_note}")
    return "\n".join(lines)
