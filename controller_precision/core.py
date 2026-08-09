"""Pure precision-curve and controller-event mapping logic.

There are deliberately no operating-system, GUI, networking, capture, or
detection imports in this module.  The Linux backend supplies evdev event
objects and writes the returned instructions to a virtual controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Generic, Protocol, Sequence, TypeVar

from .codes import (
    ABS_BRAKE,
    ABS_RZ,
    ABS_Z,
    EV_ABS,
    EV_KEY,
    EV_SYN,
    SYN_DROPPED,
    SYN_REPORT,
)


class EventLike(Protocol):
    """Minimum shape accepted from evdev or a test double."""

    type: int
    code: int
    value: int


EventT = TypeVar("EventT", bound=EventLike)


@dataclass(frozen=True, slots=True)
class AxisRange:
    """Raw range and neutral point for a centered controller axis."""

    minimum: int
    maximum: int
    center: float | None = None

    def __post_init__(self) -> None:
        if self.maximum <= self.minimum:
            raise ValueError("axis maximum must be greater than minimum")
        center = self.midpoint if self.center is None else float(self.center)
        if not math.isfinite(center) or not self.minimum <= center <= self.maximum:
            raise ValueError("axis center must be finite and within its range")

    @property
    def midpoint(self) -> float:
        return (self.minimum + self.maximum) / 2.0

    @property
    def neutral(self) -> float:
        return self.midpoint if self.center is None else float(self.center)

    def normalize(self, value: int | float) -> float:
        """Map a raw centered-axis value to ``[-1, 1]``."""

        raw = min(max(float(value), self.minimum), self.maximum)
        center = self.neutral
        extent = self.maximum - center if raw >= center else center - self.minimum
        if extent <= 0.0:
            return 0.0
        return min(max((raw - center) / extent, -1.0), 1.0)

    def encode(self, normalized: float) -> int:
        """Map a normalized centered-axis value back to this raw range."""

        value = min(max(float(normalized), -1.0), 1.0)
        center = self.neutral
        extent = self.maximum - center if value >= 0.0 else center - self.minimum
        return int(min(max(round(center + value * extent), self.minimum), self.maximum))


@dataclass(frozen=True, slots=True)
class TriggerCalibration:
    """Rest and fully pressed values for an analog activation control."""

    rest: int = 0
    pressed: int = 255

    def __post_init__(self) -> None:
        if self.pressed == self.rest:
            raise ValueError("trigger pressed value must differ from rest")

    def pressure(self, value: int | float) -> float:
        fraction = (float(value) - self.rest) / (self.pressed - self.rest)
        return min(max(fraction, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    """Parameters for LT-held user-directed right-stick precision."""

    strength: float = 0.35
    exponent: float = 1.4
    deadzone: float = 0.04
    activate_at: float = 0.35
    release_at: float = 0.25
    right_x_code: int = ABS_Z
    right_y_code: int = ABS_RZ
    trigger_axis_code: int = ABS_BRAKE
    # The PXN advertises BTN_TL2, but generic HID button numbering does not
    # prove that it mirrors the analog LT axis.  Use ABS_BRAKE alone until a
    # calibration explicitly enables a digital companion code.
    trigger_button_code: int | None = None

    def __post_init__(self) -> None:
        numeric = (
            self.strength,
            self.exponent,
            self.deadzone,
            self.activate_at,
            self.release_at,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("precision parameters must be finite")
        if not 0.0 < self.strength <= 1.0:
            raise ValueError("strength must be greater than 0 and at most 1")
        if self.exponent <= 0.0:
            raise ValueError("exponent must be greater than 0")
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("deadzone must be at least 0 and less than 1")
        if not 0.0 <= self.release_at < self.activate_at <= 1.0:
            raise ValueError("trigger thresholds must satisfy 0 <= release < activate <= 1")
        if self.right_x_code == self.right_y_code:
            raise ValueError("right-stick axes must use different event codes")
        if self.trigger_axis_code in (self.right_x_code, self.right_y_code):
            raise ValueError("trigger axis must be different from both right-stick axes")


@dataclass(slots=True)
class TriggerHysteresis:
    """Stable pressed state for a noisy analog trigger."""

    calibration: TriggerCalibration
    activate_at: float = 0.35
    release_at: float = 0.25
    active: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.release_at < self.activate_at <= 1.0:
            raise ValueError("trigger thresholds must satisfy 0 <= release < activate <= 1")

    def update(self, raw_value: int) -> bool:
        pressure = self.calibration.pressure(raw_value)
        if self.active:
            if pressure <= self.release_at:
                self.active = False
        elif pressure >= self.activate_at:
            self.active = True
        return self.active


def apply_precision_curve(
    x: float,
    y: float,
    config: PrecisionConfig,
) -> tuple[float, float]:
    """Apply one radial precision curve without changing stick direction."""

    x_value = min(max(float(x), -1.0), 1.0)
    y_value = min(max(float(y), -1.0), 1.0)
    magnitude = math.hypot(x_value, y_value)
    if magnitude <= config.deadzone or magnitude == 0.0:
        return 0.0, 0.0

    clamped_magnitude = min(magnitude, 1.0)
    usable = (clamped_magnitude - config.deadzone) / (1.0 - config.deadzone)
    adjusted = config.strength * (usable**config.exponent)
    scale = adjusted / magnitude
    return x_value * scale, y_value * scale


@dataclass(frozen=True, slots=True)
class MappedEvent(Generic[EventT]):
    """One virtual-device write, optionally preserving the source object."""

    type: int
    code: int
    value: int
    source: EventT | None = None

    @classmethod
    def passthrough(cls, event: EventT) -> "MappedEvent[EventT]":
        return cls(event.type, event.code, event.value, event)

    @classmethod
    def replacement(cls, event_type: int, code: int, value: int) -> "MappedEvent[EventT]":
        return cls(event_type, code, value, None)

    @property
    def is_passthrough(self) -> bool:
        return self.source is not None


class DroppedEventsError(RuntimeError):
    """Raised when the kernel reports an unreliable event stream."""


class EventMapper(Generic[EventT]):
    """Turn complete physical input reports into virtual-device writes.

    While precision is inactive, a steady-state report is returned entirely as
    passthrough objects.  This prevents an enabled worker from altering normal
    controller input.  Entering or leaving precision writes both right-stick
    axes once so the virtual controller changes state immediately.
    """

    def __init__(
        self,
        config: PrecisionConfig | None = None,
        *,
        x_range: AxisRange = AxisRange(0, 255),
        y_range: AxisRange = AxisRange(0, 255),
        trigger_calibration: TriggerCalibration = TriggerCalibration(),
        initial_x: int | None = None,
        initial_y: int | None = None,
        initial_trigger: int | None = None,
        trigger_button_pressed: bool = False,
    ) -> None:
        self.config = config or PrecisionConfig()
        self.x_range = x_range
        self.y_range = y_range
        self.trigger = TriggerHysteresis(
            trigger_calibration,
            activate_at=self.config.activate_at,
            release_at=self.config.release_at,
        )
        self.raw_x = x_range.encode(0.0) if initial_x is None else int(initial_x)
        self.raw_y = y_range.encode(0.0) if initial_y is None else int(initial_y)
        self.raw_trigger = (
            trigger_calibration.rest if initial_trigger is None else int(initial_trigger)
        )
        self.trigger.update(self.raw_trigger)
        self.trigger_button_pressed = bool(trigger_button_pressed)
        self.active = self.trigger.active or self.trigger_button_pressed
        self._report: list[EventT] = []

    def feed(self, event: EventT) -> tuple[MappedEvent[EventT], ...]:
        """Consume an event and return writes when its report is complete."""

        if event.type == EV_SYN and event.code == SYN_DROPPED:
            self._report.clear()
            raise DroppedEventsError(
                "controller events were dropped; stop and restart precision control"
            )

        self._report.append(event)
        if event.type != EV_SYN or event.code != SYN_REPORT:
            return ()

        report = tuple(self._report)
        self._report.clear()
        return self.map_report(report)

    def map_report(self, report: Sequence[EventT]) -> tuple[MappedEvent[EventT], ...]:
        if not report:
            return ()
        if any(event.type == EV_SYN and event.code == SYN_DROPPED for event in report):
            raise DroppedEventsError(
                "controller events were dropped; stop and restart precision control"
            )

        previous_active = self.active
        right_changed = False
        for event in report:
            if event.type == EV_ABS:
                if event.code == self.config.right_x_code:
                    self.raw_x = event.value
                    right_changed = True
                elif event.code == self.config.right_y_code:
                    self.raw_y = event.value
                    right_changed = True
                elif event.code == self.config.trigger_axis_code:
                    self.raw_trigger = event.value
                    self.trigger.update(event.value)
            elif (
                event.type == EV_KEY
                and self.config.trigger_button_code is not None
                and event.code == self.config.trigger_button_code
            ):
                self.trigger_button_pressed = event.value != 0

        self.active = self.trigger.active or self.trigger_button_pressed
        transition = self.active != previous_active

        # Exact steady-state passthrough is the most important invariant: when
        # LT is not active, no event is synthesized, removed, or reordered.
        if not self.active and not previous_active:
            return tuple(MappedEvent.passthrough(event) for event in report)

        replace_axes = transition or right_changed
        output: list[MappedEvent[EventT]] = []
        sync_event: EventT | None = None
        for event in report:
            if event.type == EV_SYN and event.code == SYN_REPORT:
                sync_event = event
                continue
            if replace_axes and event.type == EV_ABS and event.code in (
                self.config.right_x_code,
                self.config.right_y_code,
            ):
                continue
            output.append(MappedEvent.passthrough(event))

        if replace_axes:
            raw_x, raw_y = self.current_output()
            output.extend(
                (
                    MappedEvent.replacement(EV_ABS, self.config.right_x_code, raw_x),
                    MappedEvent.replacement(EV_ABS, self.config.right_y_code, raw_y),
                )
            )

        if sync_event is not None:
            output.append(MappedEvent.passthrough(sync_event))
        return tuple(output)

    def current_output(self) -> tuple[int, int]:
        """Return the right-stick values that the virtual device should hold."""

        if not self.active:
            return self.raw_x, self.raw_y
        precise_x, precise_y = apply_precision_curve(
            self.x_range.normalize(self.raw_x),
            self.y_range.normalize(self.raw_y),
            self.config,
        )
        return self.x_range.encode(precise_x), self.y_range.encode(precise_y)
