"""OpenVINO device discovery for the desktop launcher."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


class DeviceDiscoveryError(RuntimeError):
    """Raised when OpenVINO devices cannot be enumerated."""


def available_openvino_devices(
    core_factory: Callable[[], Any] | None = None,
) -> tuple[str, ...]:
    """Return normalized physical devices in stable display order."""

    if core_factory is None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise DeviceDiscoveryError("OpenVINO is not installed") from exc
        core_type = getattr(ov, "Core", None)
        if core_type is None:
            try:
                from openvino.runtime import Core as core_type
            except ImportError as exc:
                raise DeviceDiscoveryError(
                    "OpenVINO does not expose the Runtime Core API"
                ) from exc
        core_factory = core_type
    try:
        core = core_factory()
        reported: Iterable[Any] = core.available_devices
    except Exception as exc:
        raise DeviceDiscoveryError(f"OpenVINO device discovery failed: {exc}") from exc

    normalized = {str(device).strip().upper() for device in reported if str(device).strip()}
    family_order = {"CPU": 0, "GPU": 1, "NPU": 2}

    def sort_key(device: str) -> tuple[int, str]:
        family = device.partition(".")[0]
        return family_order.get(family, 10), device

    return tuple(sorted(normalized, key=sort_key))


def selectable_openvino_devices(available: Iterable[str]) -> tuple[str, ...]:
    """Add AUTO and generic family aliases useful for indexed devices."""

    physical = tuple(dict.fromkeys(str(device).strip().upper() for device in available))
    aliases: list[str] = []
    for family in ("CPU", "GPU", "NPU"):
        if any(device == family or device.startswith(f"{family}.") for device in physical):
            aliases.append(family)
    indexed = [device for device in physical if device not in aliases]
    return tuple(dict.fromkeys(("AUTO", *aliases, *indexed)))