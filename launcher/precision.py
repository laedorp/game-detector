"""Controller-precision launcher helpers with no detector dependencies.

The GUI uses this module to discover the supported physical controller, bind a
successful read-only mapping check to that device's identity, and construct an
internal worker command.  Nothing here reads video or target coordinates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys

from controller_precision.linux_evdev import (
    ControllerBackendError,
    ControllerCandidate,
    PXN_P5_8K_PRODUCT_ID,
    PXN_VENDOR_ID,
    describe_candidate,
    discover_controllers,
    require_evdev,
)
from controller_precision.protocol import is_controller_ready_line

from .settings import resource_root


@dataclass(frozen=True, slots=True)
class PrecisionPreset:
    """A named, user-facing fine-control curve preset."""

    key: str
    label: str
    strength: float
    description: str


PRECISION_PRESETS = (
    PrecisionPreset(
        "gentle",
        "Gentle",
        0.50,
        "Keeps half of your stick range while LT is held.",
    ),
    PrecisionPreset(
        "balanced",
        "Balanced",
        0.35,
        "A practical default for smaller, deliberate adjustments.",
    ),
    PrecisionPreset(
        "strong",
        "Strong",
        0.22,
        "Reduces movement more for very fine manual control.",
    ),
)
DEFAULT_PRECISION_PRESET = "balanced"
_PRESETS_BY_KEY = {preset.key: preset for preset in PRECISION_PRESETS}


@dataclass(frozen=True, slots=True)
class PrecisionReadiness:
    """Result of a non-mutating worker readiness check."""

    ready: bool
    summary: str
    action: str = ""


def precision_supported(platform: str | None = None) -> bool:
    current = sys.platform if platform is None else platform
    return current.startswith("linux")


def precision_preset(key: str) -> PrecisionPreset:
    return _PRESETS_BY_KEY.get(key, _PRESETS_BY_KEY[DEFAULT_PRECISION_PRESET])


def pxn_controllers(
    discovery: Callable[[], Sequence[ControllerCandidate]] = discover_controllers,
) -> tuple[ControllerCandidate, ...]:
    """Return physical PXN P5 8K candidates in deterministic path order."""

    matches = (
        candidate
        for candidate in discovery()
        if not candidate.is_virtual
        and candidate.vendor == PXN_VENDOR_ID
        and candidate.product == PXN_P5_8K_PRODUCT_ID
    )
    return tuple(sorted(matches, key=lambda candidate: str(candidate.path)))


def candidate_label(candidate: ControllerCandidate) -> str:
    return describe_candidate(candidate)


def candidate_identity(candidate: ControllerCandidate) -> str:
    """Return a stable fingerprint that binds verification to one device.

    A serial is preferred.  Devices without one also include their persistent
    path and physical port, preventing a bare boolean from authorizing an
    unrelated controller with the same USB product ID.
    """

    return candidate.identity.fingerprint()


def select_saved_candidate(
    candidates: Sequence[ControllerCandidate],
    saved_path: str,
) -> ControllerCandidate | None:
    """Prefer the saved path, falling back only when exactly one PXN exists."""

    requested = saved_path.strip()
    if requested:
        for candidate in candidates:
            if str(candidate.path) == requested or str(candidate.event_path) == requested:
                return candidate
    if len(candidates) == 1:
        return candidates[0]
    return None


def precision_readiness(
    candidate: ControllerCandidate,
    *,
    platform: str | None = None,
    uinput_path: Path = Path("/dev/uinput"),
    evdev_check: Callable[[], object] = require_evdev,
) -> PrecisionReadiness:
    """Check dependencies and permissions without changing system state."""

    if not precision_supported(platform):
        return PrecisionReadiness(
            False,
            "Controller precision is available only on Linux.",
        )
    try:
        evdev_check()
    except (ControllerBackendError, ImportError, OSError) as exc:
        return PrecisionReadiness(
            False,
            "The Linux controller support package is unavailable.",
            f"Install the app's evdev dependency, then reopen ProAim. Details: {exc}",
        )
    if not candidate.readable:
        return PrecisionReadiness(
            False,
            "The PXN controller was found, but this desktop session cannot read it.",
            "Reconnect the controller while signed in, then press Refresh. Do not run the whole app as root.",
        )
    if not uinput_path.exists():
        return PrecisionReadiness(
            False,
            f"The virtual-controller device {uinput_path} is missing.",
            (
                "In a terminal run: sudo modprobe uinput\n"
                "Then return here and press Refresh. This app never runs privileged commands automatically."
            ),
        )
    if not os.access(uinput_path, os.W_OK):
        return PrecisionReadiness(
            False,
            f"This desktop session cannot write to {uinput_path}.",
            (
                "Ask the system administrator to grant the active desktop user uaccess permission to "
                "/dev/uinput, then sign in again. Do not run the whole app as root."
            ),
        )
    return PrecisionReadiness(True, "Controller and virtual-controller access are ready.")


_VERIFIED_CALIBRATION = re.compile(r"\brest\s+(-?\d+),\s+pressed\s+(-?\d+),")


def verification_calibration(output: str) -> tuple[int, int] | None:
    """Extract the calibration emitted only after a successful mapping check."""

    matches = _VERIFIED_CALIBRATION.findall(output)
    if len(matches) != 1:
        return None
    rest, pressed = (int(value) for value in matches[0])
    if rest == pressed:
        return None
    return rest, pressed


def precision_command(
    candidate: ControllerCandidate,
    *,
    mode: str,
    preset_key: str = DEFAULT_PRECISION_PRESET,
    parent_pid: int | None = None,
    trigger_rest: int | None = None,
    trigger_pressed: int | None = None,
    verification_seconds: float = 4.0,
    executable: str | Path | None = None,
    app_script: str | Path | None = None,
    frozen: bool | None = None,
    platform: str | None = None,
) -> list[str]:
    """Build a shell-free command for mapping verification or the worker."""

    if not precision_supported(platform):
        raise ValueError("controller precision is available only on Linux")
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    prefix = [str(executable or sys.executable)]
    if not is_frozen:
        prefix.append(str(app_script or (resource_root() / "app.py")))
    command = [*prefix, "--controller-precision"]
    identity_arguments: list[str] = []
    if candidate.vendor is not None:
        identity_arguments.extend(("--vendor", hex(candidate.vendor)))
    if candidate.product is not None:
        identity_arguments.extend(("--product", hex(candidate.product)))
    if candidate.serial:
        identity_arguments.extend(("--serial", candidate.serial))
    # This is the fingerprint saved only after the GUI's read-only mapping
    # verification.  The worker validates it against its own discovery before
    # opening anything; no-serial devices bind it to both path and phys.
    identity_arguments.extend(("--expected-fingerprint", candidate_identity(candidate)))
    if mode == "verify":
        if verification_seconds <= 0.0:
            raise ValueError("verification duration must be greater than zero")
        command.extend(
            (
                "--verify-mapping",
                "--device",
                str(candidate.path),
                *identity_arguments,
                "--verification-seconds",
                f"{verification_seconds:g}",
            )
        )
        return command
    if mode != "run":
        raise ValueError("precision command mode must be 'verify' or 'run'")
    if parent_pid is None or parent_pid <= 0:
        raise ValueError("a positive launcher parent PID is required")
    if (trigger_rest is None) != (trigger_pressed is None):
        raise ValueError("trigger rest and pressed calibration must be provided together")
    if trigger_rest is not None and trigger_rest == trigger_pressed:
        raise ValueError("trigger rest and pressed calibration must differ")
    preset = precision_preset(preset_key)
    command.extend(
        (
            "--run",
            "--confirm-default-mapping",
            "--parent-pid",
            str(parent_pid),
            "--device",
            str(candidate.path),
            *identity_arguments,
            "--strength",
            f"{preset.strength:g}",
        )
    )
    if trigger_rest is not None and trigger_pressed is not None:
        command.extend(
            (
                "--trigger-rest",
                str(trigger_rest),
                "--trigger-pressed",
                str(trigger_pressed),
            )
        )
    return command
