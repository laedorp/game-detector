"""Pure, identity-bound propagation of an observed direct-head location.

The direct-head detector remains the only source which may establish or move
the head's normalized location inside a player box.  Once established, a
current primary-player box may map that observed location into current screen
coordinates.  Mapping never refreshes the direct observation's identity
deadline, and a predicted primary box is explicitly distinguished from a
measured one.

This module deliberately owns no detector, controller, clock, or thread.  Its
explicit source timestamps and track generations make it suitable for unit
testing before it is connected to the live runtime.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math
import statistics
from typing import TypeAlias


Box: TypeAlias = tuple[float, float, float, float]
Point: TypeAlias = tuple[float, float]

# Live direct-head inference can produce clustered no-decoded gaps even while
# the primary detector continues to measure the same tracked player. Retain
# the verified normalized head offset across the measured 0.63 s p95 / 0.73 s
# maximum gap. This remains an immutable identity lease: only fresh measured
# geometry in the same tracker generation may publish it, and prediction-only
# geometry remains non-authoritative.
DIRECT_HEAD_ANCHOR_MAX_AGE_SECONDS = 0.750
DIRECT_HEAD_ANCHOR_FILTER_SAMPLES = 5
_NS_PER_SECOND = 1_000_000_000
_NORMALIZED_SIDE_MARGIN = 0.12
_NORMALIZED_TOP_MARGIN = 0.12
_NORMALIZED_HEAD_REGION_BOTTOM = 0.48


class DirectHeadProvenance(str, Enum):
    """Finite provenance for one position returned by the anchor."""

    DIRECT = "direct"
    MEASURED_PRIMARY = "measured_primary"
    PREDICTED_PRIMARY = "predicted_primary"


@dataclass(frozen=True, slots=True)
class DirectHeadAnchorSample:
    """One head position with separate geometry and identity timestamps.

    ``source_timestamp_ns`` identifies the image geometry used for ``point``.
    ``direct_source_timestamp_ns`` and ``identity_deadline_ns`` retain the
    immutable direct-head evidence boundary.  A caller must not replace that
    deadline with the newer geometry timestamp of a body-derived sample.
    """

    point: Point
    source_timestamp_ns: int
    direct_source_timestamp_ns: int
    identity_deadline_ns: int
    track_generation: int
    provenance: DirectHeadProvenance
    confidence: float
    motion_corroboration_permitted: bool

    @property
    def body_derived(self) -> bool:
        return self.provenance is not DirectHeadProvenance.DIRECT

    @property
    def primary_observed(self) -> bool:
        return self.provenance is not DirectHeadProvenance.PREDICTED_PRIMARY


class DirectHeadAnchor:
    """Robustly retain one direct head location inside one logical track.

    Direct observations are normalized to their exact same-source primary
    player box.  An odd-sized rolling median rejects isolated localization
    excursions without synthesizing a fixed body/head ratio.  Later primary
    geometry can translate and scale the retained location, but cannot extend
    the maximum age of the latest direct observation.
    """

    def __init__(
        self,
        *,
        max_direct_age_seconds: float = DIRECT_HEAD_ANCHOR_MAX_AGE_SECONDS,
        filter_samples: int = DIRECT_HEAD_ANCHOR_FILTER_SAMPLES,
    ) -> None:
        maximum_age = float(max_direct_age_seconds)
        if not math.isfinite(maximum_age) or maximum_age <= 0.0:
            raise ValueError("max_direct_age_seconds must be finite and positive")
        if (
            isinstance(filter_samples, bool)
            or not isinstance(filter_samples, int)
            or filter_samples < 3
            or filter_samples % 2 == 0
        ):
            raise ValueError("filter_samples must be an odd integer of at least 3")
        self.max_direct_age_ns = max(
            1,
            round(maximum_age * _NS_PER_SECOND),
        )
        self.filter_samples = filter_samples
        self._normalized_x: deque[float] = deque(maxlen=filter_samples)
        self._normalized_y: deque[float] = deque(maxlen=filter_samples)
        self._track_generation: int | None = None
        self._last_direct_source_ns: int | None = None
        self._last_direct_confidence = 0.0

    @property
    def active(self) -> bool:
        return self._last_direct_source_ns is not None

    @property
    def track_generation(self) -> int | None:
        return self._track_generation

    @property
    def last_direct_source_timestamp_ns(self) -> int | None:
        return self._last_direct_source_ns

    @property
    def identity_deadline_ns(self) -> int | None:
        timestamp = self._last_direct_source_ns
        return None if timestamp is None else timestamp + self.max_direct_age_ns

    @property
    def normalized_point(self) -> Point | None:
        if not self._normalized_x or not self._normalized_y:
            return None
        return (
            float(statistics.median(self._normalized_x)),
            float(statistics.median(self._normalized_y)),
        )

    def reset(self) -> None:
        """Hard-reset the identity binding and every retained coordinate."""

        self._track_generation = None
        self._clear_anchor()

    def advance_generation(self, track_generation: int) -> bool:
        """Bind a newer identity generation and invalidate the old anchor.

        Returns true only when the generation advanced.  A stale generation
        cannot move the binding backwards.
        """

        generation = _generation(track_generation)
        current = self._track_generation
        if current is not None and generation < current:
            return False
        if current == generation:
            return False
        self._track_generation = generation
        self._clear_anchor()
        return True

    def observe_direct(
        self,
        point: Sequence[float],
        primary_box: Sequence[float],
        *,
        track_generation: int,
        source_timestamp_ns: int,
        confidence: float,
    ) -> DirectHeadAnchorSample | None:
        """Accept one direct head and exact same-source primary measurement.

        A newer generation may establish a new anchor immediately because the
        direct observation itself is identity-bound.  A stale generation or
        non-advancing timestamp is discarded without disturbing newer state.
        """

        generation = _generation(track_generation)
        timestamp = _timestamp(source_timestamp_ns)
        box = _box(primary_box)
        head = _point(point)
        direct_confidence = _confidence(confidence)

        current = self._track_generation
        if current is not None and generation < current:
            return None
        if current is None or generation > current:
            self._track_generation = generation
            self._clear_anchor()
        previous_timestamp = self._last_direct_source_ns
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            return None
        if (
            previous_timestamp is not None
            and timestamp - previous_timestamp >= self.max_direct_age_ns
        ):
            # Never blend a newly reacquired coordinate with an expired track.
            self._clear_anchor()

        normalized_x, normalized_y = _normalize(head, box)
        # Match the live runtime's already-established direct-head anatomy
        # boundary. A contained head box may place its center just beyond a
        # noisy primary edge, so requiring [0, 1] here would silently reject
        # evidence which the exact association layer intentionally accepts.
        if not (
            -_NORMALIZED_SIDE_MARGIN
            <= normalized_x
            <= 1.0 + _NORMALIZED_SIDE_MARGIN
            and -_NORMALIZED_TOP_MARGIN
            <= normalized_y
            <= _NORMALIZED_HEAD_REGION_BOTTOM
        ):
            return None
        self._normalized_x.append(normalized_x)
        self._normalized_y.append(normalized_y)
        self._last_direct_source_ns = timestamp
        self._last_direct_confidence = direct_confidence
        filtered = self.normalized_point
        assert filtered is not None
        return self._sample(
            point=_map(filtered, box),
            source_timestamp_ns=timestamp,
            provenance=DirectHeadProvenance.DIRECT,
            confidence=direct_confidence,
            motion_corroboration_permitted=True,
        )

    def map_primary(
        self,
        primary_box: Sequence[float],
        *,
        track_generation: int,
        source_timestamp_ns: int,
        primary_observed: bool,
    ) -> DirectHeadAnchorSample | None:
        """Map the retained direct location through current primary geometry.

        Measured and predicted primary boxes have distinct provenance, and
        neither changes the stored direct timestamp or its hard deadline.
        A newer generation clears the old anchor and therefore cannot inherit
        its head position.  A stale generation is ignored.
        """

        if not isinstance(primary_observed, bool):
            raise TypeError("primary_observed must be bool")
        generation = _generation(track_generation)
        timestamp = _timestamp(source_timestamp_ns)
        box = _box(primary_box)
        current = self._track_generation
        if current is None:
            self._track_generation = generation
            return None
        if generation < current:
            return None
        if generation > current:
            self._track_generation = generation
            self._clear_anchor()
            return None
        direct_timestamp = self._last_direct_source_ns
        normalized = self.normalized_point
        if direct_timestamp is None or normalized is None:
            return None
        age_ns = timestamp - direct_timestamp
        if age_ns < 0:
            return None
        if age_ns >= self.max_direct_age_ns:
            self._clear_anchor()
            return None
        remaining = 1.0 - age_ns / self.max_direct_age_ns
        provenance = (
            DirectHeadProvenance.MEASURED_PRIMARY
            if primary_observed
            else DirectHeadProvenance.PREDICTED_PRIMARY
        )
        return self._sample(
            point=_map(normalized, box),
            source_timestamp_ns=timestamp,
            provenance=provenance,
            confidence=self._last_direct_confidence * remaining,
            # A body-derived coordinate and the body center are the same
            # evidence channel.  They can never independently authorize
            # predictive feed-forward.
            motion_corroboration_permitted=False,
        )

    def _sample(
        self,
        *,
        point: Point,
        source_timestamp_ns: int,
        provenance: DirectHeadProvenance,
        confidence: float,
        motion_corroboration_permitted: bool,
    ) -> DirectHeadAnchorSample:
        direct_timestamp = self._last_direct_source_ns
        generation = self._track_generation
        assert direct_timestamp is not None
        assert generation is not None
        return DirectHeadAnchorSample(
            point=point,
            source_timestamp_ns=source_timestamp_ns,
            direct_source_timestamp_ns=direct_timestamp,
            identity_deadline_ns=direct_timestamp + self.max_direct_age_ns,
            track_generation=generation,
            provenance=provenance,
            confidence=min(max(float(confidence), 0.0), 1.0),
            motion_corroboration_permitted=motion_corroboration_permitted,
        )

    def _clear_anchor(self) -> None:
        self._normalized_x.clear()
        self._normalized_y.clear()
        self._last_direct_source_ns = None
        self._last_direct_confidence = 0.0


def _generation(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("track_generation must be a non-negative integer")
    return value


def _timestamp(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("source_timestamp_ns must be a non-negative integer")
    return value


def _confidence(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be finite and between zero and one")
    return result


def _point(value: Sequence[float]) -> Point:
    if len(value) != 2:
        raise ValueError("point must contain two coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("point must contain only finite coordinates")
    return result[0], result[1]


def _box(value: Sequence[float]) -> Box:
    if len(value) != 4:
        raise ValueError("primary_box must contain four coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("primary_box must contain only finite coordinates")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("primary_box must have positive width and height")
    return result[0], result[1], result[2], result[3]


def _normalize(point: Point, box: Box) -> Point:
    return (
        (point[0] - box[0]) / (box[2] - box[0]),
        (point[1] - box[1]) / (box[3] - box[1]),
    )


def _map(normalized: Point, box: Box) -> Point:
    return (
        box[0] + normalized[0] * (box[2] - box[0]),
        box[1] + normalized[1] * (box[3] - box[1]),
    )


__all__ = (
    "DIRECT_HEAD_ANCHOR_FILTER_SAMPLES",
    "DIRECT_HEAD_ANCHOR_MAX_AGE_SECONDS",
    "DirectHeadAnchor",
    "DirectHeadAnchorSample",
    "DirectHeadProvenance",
)
