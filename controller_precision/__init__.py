"""User-driven controller precision support.

This package is intentionally independent from video capture and object
detection.  It only transforms events produced by a physical controller while
the user holds a configured controller control.
"""

from .core import (
    AxisRange,
    DroppedEventsError,
    EventMapper,
    MappedEvent,
    PrecisionConfig,
    TriggerCalibration,
    TriggerHysteresis,
    apply_precision_curve,
)

__all__ = (
    "AxisRange",
    "DroppedEventsError",
    "EventMapper",
    "MappedEvent",
    "PrecisionConfig",
    "TriggerCalibration",
    "TriggerHysteresis",
    "apply_precision_curve",
)
