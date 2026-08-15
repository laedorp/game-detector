"""Cross-vendor hardware discovery and tailored detector recommendations.

The detector must run unchanged on very different machines: an Intel laptop with
only a CPU, the same laptop once its integrated graphics driver is installed, a
box with a discrete AMD or NVIDIA card, or an Intel part with an NPU.  This
module answers two separate questions and keeps them separate on purpose:

1. *What hardware is physically present?*  That is read from the operating
   system and is true regardless of which Python packages happen to be
   installed.
2. *What can this installation actually use right now?*  A Radeon card is
   present whether or not a runtime that can drive it exists, so a recommendation
   reports ``ready`` plus an actionable ``setup_hint`` instead of silently
   pretending an unusable device is a choice.

Keeping those apart is what lets the launcher explain "you have a GPU, here is
the one package you are missing" rather than simply omitting the option.

This module deliberately imports no launcher or capture code. It reads the
small release-default pointer so a hardware scan cannot reset a rectangular
default back to a stale square size. It shells out only on Windows, where there
is no sysfs equivalent to read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import platform
import re
import subprocess
from typing import Any

from utils.inference_size import InferenceSizeLike
from utils.release_model_contract import load_release_default_contract


LINUX_PCI_ROOT = Path("/sys/bus/pci/devices")
LINUX_CPUINFO = Path("/proc/cpuinfo")

# Distributions ship the shared PCI database in one of these locations.  It is
# used only to turn raw identifiers into readable names, so every lookup falls
# back to the identifier itself when the file is absent.
PCI_ID_DATABASES = (
    Path("/usr/share/hwdata/pci.ids"),
    Path("/usr/share/misc/pci.ids"),
)

# PCI vendor identifiers, lowercase and without the ``0x`` prefix that sysfs
# writes, so both sysfs and Windows ``PNPDeviceID`` parsing normalize to these.
VENDOR_IDS = {
    "8086": "intel",
    "1002": "amd",
    "1022": "amd",
    "10de": "nvidia",
}

# PCI class prefixes.  ``03`` is a display controller; ``1200`` is the
# "processing accelerator" class Intel's NPU reports.
DISPLAY_CLASS_PREFIX = "03"
ACCELERATOR_CLASS_PREFIX = "1200"

# CPU flags that materially change which precision is worth recommending.
INT8_ACCELERATION_FLAGS = ("avx512_vnni", "avx_vnni", "amx_int8")
WIDE_VECTOR_FLAGS = ("avx2", "avx512f")

_RELEASE_DEFAULT = load_release_default_contract(Path(__file__).resolve().parents[1])
_RELEASE_DEFAULT_HW = tuple(_RELEASE_DEFAULT["input_shape_nchw"][2:4])
_RELEASE_DEFAULT_INFERENCE_SIZE: InferenceSizeLike = (
    _RELEASE_DEFAULT_HW[0]
    if _RELEASE_DEFAULT_HW[0] == _RELEASE_DEFAULT_HW[1]
    else _RELEASE_DEFAULT_HW
)


class Vendor(str, Enum):
    INTEL = "intel"
    AMD = "amd"
    NVIDIA = "nvidia"
    UNKNOWN = "unknown"


class AcceleratorKind(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"


class HardwareScanError(RuntimeError):
    """Raised when a scan cannot be completed at all."""


@dataclass(frozen=True, slots=True)
class Accelerator:
    """One physical compute device that could, in principle, run inference."""

    kind: AcceleratorKind
    vendor: Vendor
    name: str
    identifier: str = ""
    # ``None`` means "present but could not be classified".  Intel Arc is a
    # discrete part sharing a vendor with every integrated Intel GPU, and no
    # reliable vendor-neutral sysfs flag separates them, so this stays honest
    # rather than guessing from device-ID ranges that change every generation.
    discrete: bool | None = None

    @property
    def label(self) -> str:
        if self.kind is AcceleratorKind.GPU and self.discrete is not None:
            placement = "discrete" if self.discrete else "integrated"
            return f"{self.name} ({placement})"
        return self.name


@dataclass(frozen=True, slots=True)
class DirectMLAdapter:
    """One DXGI adapter in the exact order DirectML uses for ``device_id``."""

    index: int
    name: str
    vendor_id: str
    device_id: str
    dedicated_vram: int = 0
    adapter_luid: str = ""

    @property
    def identifier(self) -> str:
        return f"{self.vendor_id}:{self.device_id}"


@dataclass(frozen=True, slots=True)
class ProcessorInfo:
    name: str
    logical_cores: int | None = None
    flags: frozenset[str] = field(default_factory=frozenset)

    @property
    def has_int8_acceleration(self) -> bool:
        """True when the CPU has instructions that make INT8 clearly faster."""

        return any(flag in self.flags for flag in INT8_ACCELERATION_FLAGS)

    @property
    def has_wide_vectors(self) -> bool:
        return any(flag in self.flags for flag in WIDE_VECTOR_FLAGS)


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    system: str
    processor: ProcessorInfo
    accelerators: tuple[Accelerator, ...]
    runtime_devices: tuple[str, ...] = ()
    directml_adapters: tuple[DirectMLAdapter, ...] = ()

    def of_kind(self, kind: AcceleratorKind) -> tuple[Accelerator, ...]:
        return tuple(item for item in self.accelerators if item.kind is kind)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One tailored way to run the detector, ranked against the alternatives."""

    accelerator: Accelerator
    backend: str
    device: str
    precision: str
    inference_size: InferenceSizeLike
    ready: bool
    reason: str
    setup_hint: str = ""

    @property
    def summary(self) -> str:
        if self.ready and self.backend == "onnxruntime":
            if self.device.upper().startswith(("DIRECTML", "DML")):
                state = (
                    "provider verified at detector start; DXGI adapter binding "
                    "must still be confirmed in Task Manager"
                )
            else:
                state = "provider found; verified at detector start"
        else:
            state = "ready" if self.ready else "needs setup"
        return f"{self.accelerator.label} — {self.backend}/{self.device} ({state})"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _normalize_hex(value: str) -> str:
    return value.strip().lower().removeprefix("0x")


def _vendor_from_id(vendor_id: str) -> Vendor:
    name = VENDOR_IDS.get(_normalize_hex(vendor_id))
    return Vendor(name) if name else Vendor.UNKNOWN


def parse_linux_cpuinfo(text: str, logical_cores: int | None) -> ProcessorInfo:
    """Read the model name and feature flags from ``/proc/cpuinfo`` text."""

    name = ""
    flags: frozenset[str] = frozenset()
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip().lower()
        if not name and key == "model name":
            name = value.strip()
        elif not flags and key in ("flags", "features"):
            flags = frozenset(value.split())
        if name and flags:
            break
    return ProcessorInfo(
        name=name or platform.processor() or "Unknown CPU",
        logical_cores=logical_cores,
        flags=flags,
    )


def load_pci_names(
    databases: Sequence[Path] = PCI_ID_DATABASES,
) -> dict[tuple[str, str], str]:
    """Map ``(vendor_id, device_id)`` to a readable device name.

    ``pci.ids`` is a flat text file where a vendor line starts at column zero and
    each of its devices is indented by one tab.  Only display and accelerator
    lookups are ever performed, but parsing the whole file once is far simpler
    than trying to seek within it.
    """

    for database in databases:
        text = _read_text(database)
        if not text:
            continue
        names: dict[tuple[str, str], str] = {}
        vendor_id = ""
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            if not line.startswith("\t"):
                vendor_id = line[:4].strip().lower()
            elif not line.startswith("\t\t") and vendor_id:
                device_id = line[1:5].strip().lower()
                names[(vendor_id, device_id)] = line[5:].strip()
        if names:
            return names
    return {}


def scan_linux_accelerators(pci_root: Path = LINUX_PCI_ROOT) -> tuple[Accelerator, ...]:
    """Enumerate display controllers and accelerators from sysfs."""

    if not pci_root.is_dir():
        return ()
    found: list[Accelerator] = []
    try:
        entries = sorted(pci_root.iterdir())
    except OSError:
        return ()
    pci_names = load_pci_names()
    for entry in entries:
        pci_class = _normalize_hex(_read_text(entry / "class"))
        vendor_id = _normalize_hex(_read_text(entry / "vendor"))
        device_id = _normalize_hex(_read_text(entry / "device"))
        if not pci_class or not vendor_id:
            continue
        if pci_class.startswith(DISPLAY_CLASS_PREFIX):
            kind = AcceleratorKind.GPU
        elif pci_class.startswith(ACCELERATOR_CLASS_PREFIX):
            kind = AcceleratorKind.NPU
        else:
            continue
        vendor = _vendor_from_id(vendor_id)
        discrete: bool | None = None
        if kind is AcceleratorKind.GPU:
            # Dedicated video memory is the one vendor-neutral signal exposed
            # consistently for discrete cards; its absence is not proof of an
            # integrated part, so an unreadable value stays ``None``.
            if (entry / "mem_info_vram_total").is_file():
                discrete = True
            elif vendor is Vendor.INTEL and entry.name.endswith("00:02.0"):
                discrete = False
        readable = pci_names.get((vendor_id, device_id))
        found.append(
            Accelerator(
                kind=kind,
                vendor=vendor,
                name=readable or f"{vendor.value.upper()} device {device_id or 'unknown'}",
                identifier=f"{vendor_id}:{device_id}",
                discrete=discrete,
            )
        )
    return tuple(found)


def _default_powershell(query: str) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed, non-user-supplied query
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout


WINDOWS_GPU_QUERY = (
    "Get-CimInstance Win32_VideoController | "
    "ForEach-Object { \"$($_.PNPDeviceID)|$($_.Name)|$($_.AdapterRAM)\" }"
)


def scan_windows_directml_adapters() -> tuple[DirectMLAdapter, ...]:
    """Enumerate all DXGI adapters without guessing from WMI ordering.

    ONNX Runtime documents DirectML's ``device_id`` as the index returned by
    ``IDXGIFactory::EnumAdapters``. DXcam 0.3.0 uses ``EnumAdapters1`` in that
    same DXGI factory order and already ships in every Windows bundle, so reuse
    its audited COM declarations instead of maintaining a second ABI binding.
    """

    try:
        import ctypes

        from dxcam._libs.dxgi import DXGI_ADAPTER_DESC1
        from dxcam.util.io import enum_dxgi_adapters
    except Exception:
        return ()

    found: list[DirectMLAdapter] = []
    try:
        raw_adapters = enum_dxgi_adapters()
    except Exception:
        return ()
    for index, adapter in enumerate(raw_adapters):
        try:
            descriptor = DXGI_ADAPTER_DESC1()
            adapter.GetDesc1(ctypes.byref(descriptor))
            found.append(
                DirectMLAdapter(
                    index=index,
                    name=str(descriptor.Description).strip(),
                    vendor_id=f"{int(descriptor.VendorId):04x}",
                    device_id=f"{int(descriptor.DeviceId):04x}",
                    dedicated_vram=max(0, int(descriptor.DedicatedVideoMemory)),
                    adapter_luid=(
                        f"0x{int(descriptor.AdapterLuid.HighPart) & 0xffffffff:08x}_"
                        f"0x{int(descriptor.AdapterLuid.LowPart) & 0xffffffff:08x}"
                    ),
                )
            )
        except Exception:
            continue
    return tuple(found)


def parse_windows_video_controllers(text: str) -> tuple[Accelerator, ...]:
    """Parse ``PNPDeviceID|Name|AdapterRAM`` lines into accelerators."""

    found: list[Accelerator] = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or not parts[1].strip():
            continue
        pnp_id, name = parts[0].strip(), parts[1].strip()
        memory_text = parts[2].strip() if len(parts) > 2 else ""
        match = re.search(r"VEN_([0-9A-Fa-f]{4})", pnp_id)
        vendor = _vendor_from_id(match.group(1)) if match else Vendor.UNKNOWN
        device_match = re.search(r"DEV_([0-9A-Fa-f]{4})", pnp_id)
        discrete: bool | None = None
        try:
            # Windows reports adapter memory as a signed 32-bit value, so very
            # large discrete framebuffers can arrive negative or truncated.  Only
            # a confidently large positive reading is treated as discrete.
            if memory_text and int(memory_text) > 1_500_000_000:
                discrete = True
        except ValueError:
            discrete = None
        found.append(
            Accelerator(
                kind=AcceleratorKind.GPU,
                vendor=vendor,
                name=name,
                identifier=(
                    f"{match.group(1).lower()}:{device_match.group(1).lower()}"
                    if match and device_match
                    else pnp_id
                ),
                discrete=discrete,
            )
        )
    return tuple(found)


def scan_hardware(
    *,
    system: str | None = None,
    pci_root: Path = LINUX_PCI_ROOT,
    cpuinfo_path: Path = LINUX_CPUINFO,
    logical_cores: int | None = None,
    powershell_runner: Callable[[str], str] | None = None,
    directml_adapter_factory: Callable[[], Sequence[DirectMLAdapter]] | None = None,
    runtime_devices: Iterable[str] | None = None,
) -> HardwareProfile:
    """Describe the machine's processor and every accelerator attached to it."""

    resolved_system = (system or platform.system()).strip().lower()
    if logical_cores is None:
        import os

        logical_cores = os.cpu_count()

    if resolved_system == "linux":
        processor = parse_linux_cpuinfo(_read_text(cpuinfo_path), logical_cores)
        accelerators = scan_linux_accelerators(pci_root)
    elif resolved_system == "windows":
        processor = ProcessorInfo(
            name=platform.processor() or "Unknown CPU",
            logical_cores=logical_cores,
        )
        runner = powershell_runner or _default_powershell
        accelerators = parse_windows_video_controllers(runner(WINDOWS_GPU_QUERY))
        adapter_factory = directml_adapter_factory or scan_windows_directml_adapters
        try:
            directml_adapters = tuple(adapter_factory())
        except Exception:
            directml_adapters = ()
    else:
        processor = ProcessorInfo(
            name=platform.processor() or "Unknown CPU",
            logical_cores=logical_cores,
        )
        accelerators = ()
        directml_adapters = ()

    if resolved_system != "windows":
        directml_adapters = ()

    cpu = Accelerator(
        kind=AcceleratorKind.CPU,
        vendor=_vendor_from_cpu_name(processor.name),
        name=processor.name,
    )
    return HardwareProfile(
        system=resolved_system,
        processor=processor,
        accelerators=(cpu, *accelerators),
        runtime_devices=tuple(str(item).strip().upper() for item in (runtime_devices or ())),
        directml_adapters=directml_adapters,
    )


def _vendor_from_cpu_name(name: str) -> Vendor:
    folded = name.casefold()
    if "intel" in folded:
        return Vendor.INTEL
    if "amd" in folded or "ryzen" in folded or "epyc" in folded:
        return Vendor.AMD
    return Vendor.UNKNOWN


def _openvino_family_present(runtime_devices: Sequence[str], family: str) -> bool:
    return any(
        device == family or device.startswith(f"{family}.") for device in runtime_devices
    )


def _onnxruntime_providers(
    provider_factory: Callable[[], Sequence[str]] | None,
) -> tuple[str, ...]:
    if provider_factory is None:
        try:
            import onnxruntime  # type: ignore[import-not-found]
        except ImportError:
            return ()
        provider_factory = onnxruntime.get_available_providers
    try:
        return tuple(str(name) for name in provider_factory())
    except Exception:
        return ()


def recommend(
    profile: HardwareProfile,
    *,
    provider_factory: Callable[[], Sequence[str]] | None = None,
) -> tuple[Recommendation, ...]:
    """Rank tailored ways to run the detector on this machine, best first.

    Ordering is by expected throughput, but an entry that is not ``ready`` is
    always sorted below every ready one. For OpenVINO, ready means the device is
    exposed by Core. For ONNX Runtime it means the installed wheel exposes the
    provider; the driver and provider are still tested when the detector creates
    its real model session.
    """

    providers = _onnxruntime_providers(provider_factory)
    windows = profile.system == "windows"
    entries: list[Recommendation] = []

    for accelerator in profile.accelerators:
        if accelerator.kind is AcceleratorKind.CPU:
            entries.append(_cpu_recommendation(profile, accelerator))
        elif accelerator.kind is AcceleratorKind.NPU:
            entries.append(
                _intel_recommendation(profile, accelerator, "NPU", providers)
            )
        elif accelerator.vendor is Vendor.INTEL:
            entries.append(
                _intel_recommendation(profile, accelerator, "GPU", providers)
            )
        elif accelerator.vendor is Vendor.AMD:
            entries.append(
                _amd_recommendation(profile, accelerator, providers, windows)
            )
        elif accelerator.vendor is Vendor.NVIDIA:
            entries.append(
                _nvidia_recommendation(profile, accelerator, providers)
            )

    entries.sort(
        key=lambda recommendation: (
            not recommendation.ready,
            _recommendation_priority(recommendation),
        )
    )
    return tuple(entries)


def _recommendation_priority(recommendation: Recommendation) -> int:
    """Rank usable devices without putting a hybrid laptop's iGPU first.

    Product names and runtime-provider presence cannot predict exact latency,
    but placement and accelerator class provide a safe coarse ordering.  A
    discrete GPU should be the one-click choice ahead of an integrated GPU on
    the same machine; this is especially important for Intel-plus-GeForce
    laptops where DXGI adapter 0 is commonly the Intel display adapter.
    """

    accelerator = recommendation.accelerator
    if accelerator.kind is AcceleratorKind.GPU:
        if accelerator.discrete is True:
            return 0
        # NVIDIA and AMD display adapters are normally discrete even when WMI
        # reports an unusable/truncated AdapterRAM value.  Prefer them to an
        # explicitly integrated Intel GPU, while leaving the final provider and
        # physical-adapter checks to detector startup and the benchmark.
        if accelerator.vendor in {Vendor.NVIDIA, Vendor.AMD}:
            return 1
        if accelerator.discrete is False:
            return 10
        return 8
    if accelerator.kind is AcceleratorKind.NPU:
        return 5
    return 30


def _cpu_recommendation(
    profile: HardwareProfile, accelerator: Accelerator
) -> Recommendation:
    processor = profile.processor
    if processor.has_int8_acceleration:
        precision, reason = "int8", "CPU reports VNNI/AMX INT8 acceleration"
    elif processor.has_wide_vectors:
        precision, reason = "int8", "CPU has wide vector units; INT8 still wins on latency"
    else:
        precision, reason = "fp32", "no wide vector units detected; INT8 gains are uncertain"
    return Recommendation(
        accelerator=accelerator,
        backend="openvino",
        device="CPU",
        precision=precision,
        inference_size=(
            416 if precision == "int8" else _RELEASE_DEFAULT_INFERENCE_SIZE
        ),
        ready=True,
        reason=reason,
    )


def _intel_recommendation(
    profile: HardwareProfile,
    accelerator: Accelerator,
    family: str,
    providers: Sequence[str] = (),
) -> Recommendation:
    ready = _openvino_family_present(profile.runtime_devices, family)
    if ready:
        hint = ""
        reason = f"OpenVINO exposes {family} on this machine"
    elif (
        profile.system == "windows"
        and family == "GPU"
        and "DmlExecutionProvider" in providers
    ):
        return Recommendation(
            accelerator=accelerator,
            backend="onnxruntime",
            device=_directml_device(profile, accelerator),
            # Bundled ONNX graphs currently have float32 inputs and weights.
            # Do not advertise FP16 until a distinct validated FP16 artifact
            # is actually selected.
            precision="fp32",
            inference_size=_RELEASE_DEFAULT_INFERENCE_SIZE,
            ready=True,
            reason=(
                "OpenVINO does not expose this Intel GPU, but ONNX Runtime "
                "exposes DmlExecutionProvider"
            ),
        )
    elif profile.system == "windows":
        reason = f"Intel {family} is present but OpenVINO does not expose it"
        hint = "Install the latest Intel graphics driver, then re-scan."
    else:
        reason = f"Intel {family} is present but OpenVINO does not expose it"
        hint = (
            "Install the Intel compute runtime (Arch/CachyOS: "
            "sudo pacman -S intel-compute-runtime), then re-scan."
        )
    return Recommendation(
        accelerator=accelerator,
        backend="openvino",
        device=family,
        # Integrated Intel graphics run FP16 well and gain little from INT8.
        precision="fp16" if family == "GPU" else "int8",
        inference_size=_RELEASE_DEFAULT_INFERENCE_SIZE,
        ready=ready,
        reason=reason,
        setup_hint=hint,
    )


def _amd_recommendation(
    profile: HardwareProfile,
    accelerator: Accelerator,
    providers: Sequence[str],
    windows: bool,
) -> Recommendation:
    # OpenVINO has no AMD GPU plugin at all, so this path is ONNX Runtime only.
    if windows:
        provider = "DmlExecutionProvider"
        device = _directml_device(profile, accelerator)
        ready = provider in providers
    else:
        # ROCMExecutionProvider was removed after ORT 1.22. Prefer its supported
        # successor when a locally qualified AMD stack exposes it, while
        # retaining the pinned legacy provider for reproducible old setups.
        provider, device = next(
            (
                pair
                for pair in (
                    ("MIGraphXExecutionProvider", "MIGRAPHX"),
                    ("ROCMExecutionProvider", "ROCM"),
                )
                if pair[0] in providers
            ),
            ("MIGraphXExecutionProvider", "MIGRAPHX"),
        )
        ready = provider in providers
    if ready:
        hint = ""
    elif windows:
        hint = "Install onnxruntime-directml, then re-scan."
    else:
        hint = (
            "No qualified AMD provider is installed. Verify the exact GPU/OS in "
            "AMD's current support matrix and use MIGraphX on supported modern "
            "stacks; the legacy onnxruntime-rocm 1.22 path is experimental. The "
            "RX 6950 XT target should use DirectML on Windows."
        )
    return Recommendation(
        accelerator=accelerator,
        backend="onnxruntime",
        device=device,
        precision="fp32",
        inference_size=_RELEASE_DEFAULT_INFERENCE_SIZE,
        ready=ready,
        reason="AMD GPUs are unsupported by OpenVINO; ONNX Runtime is the usable path",
        setup_hint=hint,
    )


def _nvidia_recommendation(
    profile: HardwareProfile,
    accelerator: Accelerator,
    providers: Sequence[str],
) -> Recommendation:
    # CUDA is the dependable default shipped by the NVIDIA bundle. TensorRT is
    # optional and may be reported by ORT even when its separate libraries are
    # unavailable, so leave it as a manual advanced selection.
    for provider, device in (
        ("CUDAExecutionProvider", "CUDA"),
        ("TensorrtExecutionProvider", "TENSORRT"),
        # The Windows DirectML release is the broadly compatible NVIDIA/AMD
        # fallback and must remain selectable on GeForce systems too.
        ("DmlExecutionProvider", "DIRECTML"),
    ):
        if provider in providers:
            if device == "DIRECTML":
                device = _directml_device(profile, accelerator)
            return Recommendation(
                accelerator=accelerator,
                backend="onnxruntime",
                device=device,
                precision="fp32",
                inference_size=_RELEASE_DEFAULT_INFERENCE_SIZE,
                ready=True,
                reason="NVIDIA GPUs run through ONNX Runtime, not OpenVINO",
            )
    return Recommendation(
        accelerator=accelerator,
        backend="onnxruntime",
        device="CUDAExecutionProvider",
        precision="fp32",
        inference_size=_RELEASE_DEFAULT_INFERENCE_SIZE,
        ready=False,
        reason="NVIDIA GPUs are unsupported by OpenVINO; ONNX Runtime is the usable path",
        setup_hint="Install onnxruntime-gpu with a matching CUDA runtime, then re-scan.",
    )


def _directml_device(profile: HardwareProfile, accelerator: Accelerator) -> str:
    """Return a device token bound to one confidently matched DXGI adapter."""

    identifier = accelerator.identifier.casefold()
    exact = [
        adapter
        for adapter in profile.directml_adapters
        if adapter.identifier.casefold() == identifier
    ]
    if len(exact) != 1:
        name = accelerator.name.casefold().strip()
        exact = [
            adapter
            for adapter in profile.directml_adapters
            if adapter.name.casefold().strip() == name
        ]
    if len(exact) == 1:
        return f"DIRECTML:{exact[0].index}"
    # Never invent an adapter index from WMI order. DirectML's documented
    # default remains available, accompanied by the Task Manager warning.
    return "DIRECTML"


def describe(profile: HardwareProfile, plans: Sequence[Recommendation]) -> str:
    """Render a short human-readable report for the launcher and CLI."""

    lines = [
        f"System: {profile.system}",
        f"CPU: {profile.processor.name}"
        + (
            f" ({profile.processor.logical_cores} logical cores)"
            if profile.processor.logical_cores
            else ""
        ),
    ]
    accelerators = [
        item for item in profile.accelerators if item.kind is not AcceleratorKind.CPU
    ]
    if accelerators:
        lines.append("Accelerators:")
        lines.extend(f"  - {item.kind.value.upper()}: {item.label}" for item in accelerators)
    else:
        lines.append("Accelerators: none detected")
    if profile.directml_adapters:
        lines.append("DirectML/DXGI adapter order:")
        lines.extend(
            f"  - {adapter.index}: {adapter.name} [{adapter.identifier}]"
            for adapter in profile.directml_adapters
        )
    lines.append("Recommended order:")
    for index, plan in enumerate(plans, start=1):
        lines.append(f"  {index}. {plan.summary} — {plan.reason}")
        if plan.setup_hint:
            lines.append(f"     hint: {plan.setup_hint}")
    return "\n".join(lines)


def scan_and_recommend(
    *,
    core_factory: Callable[[], Any] | None = None,
    **scan_kwargs: Any,
) -> tuple[HardwareProfile, tuple[Recommendation, ...]]:
    """Convenience entry point used by the CLI and launcher."""

    from .devices import DeviceDiscoveryError, available_openvino_devices

    try:
        runtime_devices = available_openvino_devices(core_factory)
    except DeviceDiscoveryError:
        runtime_devices = ()
    profile = scan_hardware(runtime_devices=runtime_devices, **scan_kwargs)
    return profile, recommend(profile)
