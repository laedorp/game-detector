"""Conservative exclusion of a third-person player's on-screen avatar.

The filter is deliberately geometric and temporal rather than an identity
classifier.  It considers only player-like labels, acquires one stable box over
several frames, and suppresses nothing whenever acquisition or association is
ambiguous.  This reduces false suppression of a close opponent, but it cannot
prove ownership or team membership.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any


MIN_SELF_BOX_HEIGHT = 0.25
MIN_SELF_BOX_WIDTH = 0.06
MAX_SELF_BOX_ASPECT_RATIO = 0.6
DEFAULT_ACQUIRE_FRAMES = 3
DEFAULT_LOST_GRACE_FRAMES = 3
DEFAULT_HANDOFF_CONFIRM_FRAMES = 3
PLAYER_LABEL_WORDS = frozenset(
    {
        "avatar",
        "character",
        "human",
        "humanoid",
        "person",
        "player",
    }
)
OTHER_PLAYER_LABEL_WORDS = frozenset({"bot", "enemy", "npc", "opponent"})


@dataclass(frozen=True, slots=True)
class NormalizedBottomZone:
    """A normalized rectangle anchored to the bottom edge of a frame."""

    left: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.left, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("self-avatar zone values must be finite")
        if not 0.0 <= self.left <= 1.0:
            raise ValueError("self-avatar zone left edge must be between 0 and 1")
        if not 0.0 < self.width <= 1.0:
            raise ValueError("self-avatar zone width must be greater than 0 and at most 1")
        if not 0.0 < self.height <= 1.0:
            raise ValueError("self-avatar zone height must be greater than 0 and at most 1")
        if self.left + self.width > 1.0:
            raise ValueError("self-avatar zone left edge plus width must be at most 1")

    @property
    def top(self) -> float:
        """Normalized top edge, measured from the top of the frame."""

        return 1.0 - self.height

    def pixel_bounds(self, frame_shape: Sequence[int]) -> tuple[int, int, int, int]:
        """Return inclusive, display-safe ``(x1, y1, x2, y2)`` pixel bounds."""

        frame_height, frame_width = _frame_dimensions(frame_shape)
        x1 = round(self.left * frame_width)
        y1 = round(self.top * frame_height)
        x2 = round((self.left + self.width) * frame_width)
        return (
            min(frame_width - 1, x1),
            min(frame_height - 1, y1),
            min(frame_width - 1, x2),
            frame_height - 1,
        )

    def contains_box_bottom_center(
        self,
        box: Sequence[float],
        frame_shape: Sequence[int],
    ) -> bool:
        """Whether the clipped box's bottom-center anchor lies inside this zone."""

        normalized = _normalized_box(box, frame_shape)
        if normalized is None:
            return False
        x1, _y1, x2, y2 = normalized
        anchor_x = (x1 + x2) * 0.5
        return self.left <= anchor_x <= self.left + self.width and self.top <= y2 <= 1.0

    def candidate_score(
        self,
        box: Sequence[float],
        frame_shape: Sequence[int],
        *,
        minimum_box_height: float = MIN_SELF_BOX_HEIGHT,
        minimum_box_width: float = MIN_SELF_BOX_WIDTH,
        maximum_aspect_ratio: float = MAX_SELF_BOX_ASPECT_RATIO,
    ) -> tuple[float, float, float] | None:
        """Rank an eligible box by expected anchor proximity, lowness, then size."""

        for value, name in (
            (minimum_box_height, "minimum box height"),
            (minimum_box_width, "minimum box width"),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite, greater than 0, and at most 1")
        if not math.isfinite(maximum_aspect_ratio) or not 0.0 < maximum_aspect_ratio <= 1.0:
            raise ValueError("maximum aspect ratio must be finite, greater than 0, and at most 1")

        normalized = _normalized_box(box, frame_shape)
        if normalized is None:
            return None
        x1, y1, x2, y2 = normalized
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        if box_width < minimum_box_width or box_height < minimum_box_height:
            return None
        if box_height > 0.0 and box_width / box_height > maximum_aspect_ratio:
            return None

        anchor_x = (x1 + x2) * 0.5
        if not (self.left <= anchor_x <= self.left + self.width and self.top <= y2 <= 1.0):
            return None

        expected_x = self.left + self.width * 0.5
        horizontal_distance = abs(anchor_x - expected_x)
        # Height is an eligibility condition, not the primary identity signal.
        return -horizontal_distance, y2, box_height


@dataclass(frozen=True, slots=True)
class ExclusionResult:
    """Retained detections plus the candidate hidden on this frame, if any."""

    detections: tuple[Any, ...]
    ignored_count: int
    ignored_detection: Any | None = None
    aim_safe: bool = False


@dataclass(frozen=True, slots=True)
class _Candidate:
    index: int
    detection: Any
    class_key: str
    box: tuple[float, float, float, float]
    prior_score: tuple[float, float, float]


class SelfAvatarFilter:
    """Acquire and follow one persistent on-screen avatar conservatively."""

    def __init__(
        self,
        zone: NormalizedBottomZone,
        *,
        acquire_frames: int = DEFAULT_ACQUIRE_FRAMES,
        lost_grace_frames: int = DEFAULT_LOST_GRACE_FRAMES,
        handoff_confirm_frames: int = DEFAULT_HANDOFF_CONFIRM_FRAMES,
    ) -> None:
        if acquire_frames <= 0:
            raise ValueError("acquire frames must be greater than zero")
        if lost_grace_frames < 0:
            raise ValueError("lost grace frames cannot be negative")
        if handoff_confirm_frames <= 0:
            raise ValueError("handoff confirm frames must be greater than zero")
        self.zone = zone
        self.acquire_frames = int(acquire_frames)
        self.lost_grace_frames = int(lost_grace_frames)
        self.handoff_confirm_frames = int(handoff_confirm_frames)
        self.reset()

    def reset(self) -> None:
        """Forget both pending acquisition and the current lock."""

        self._frame_dimensions: tuple[int, int] | None = None
        self._pending_box: tuple[float, float, float, float] | None = None
        self._pending_class: str | None = None
        self._pending_hits = 0
        self._locked_box: tuple[float, float, float, float] | None = None
        self._locked_class: str | None = None
        self._lost_frames = 0
        self._handoff_box: tuple[float, float, float, float] | None = None
        self._handoff_class: str | None = None
        self._handoff_hits = 0

    @property
    def acquired(self) -> bool:
        return self._locked_box is not None

    def apply(
        self,
        detections: Iterable[Any],
        frame_shape: Sequence[int],
    ) -> ExclusionResult:
        """Filter one frame, retaining everything when the choice is uncertain."""

        dimensions = _frame_dimensions(frame_shape)
        if self._frame_dimensions is not None and dimensions != self._frame_dimensions:
            self.reset()
        self._frame_dimensions = dimensions

        materialized = tuple(detections)
        candidates = _candidates(materialized, frame_shape, self.zone)
        if self._locked_box is not None and self._locked_class is not None:
            return self._apply_locked(materialized, candidates, frame_shape)
        return self._apply_acquisition(materialized, candidates)

    def _apply_acquisition(
        self,
        detections: tuple[Any, ...],
        candidates: tuple[_Candidate, ...],
    ) -> ExclusionResult:
        if not candidates:
            self._clear_pending()
            # No plausible self-avatar candidate is visible in this frame, so
            # aiming can proceed normally while acquisition waits.
            return ExclusionResult(detections, 0, aim_safe=True)

        # Multiple plausible players at startup is ambiguous.  Keep every box
        # and wait for an unambiguous view instead of guessing which is self.
        if len(candidates) != 1:
            self._clear_pending()
            return ExclusionResult(detections, 0)

        candidate = candidates[0]
        if self._pending_box is None or self._pending_class != candidate.class_key:
            self._start_pending(candidate)
        elif _association_score(self._pending_box, candidate.box) is None:
            self._start_pending(candidate)
        else:
            self._pending_box = candidate.box
            self._pending_hits += 1

        if self._pending_hits < self.acquire_frames:
            return ExclusionResult(detections, 0)

        self._locked_box = candidate.box
        self._locked_class = candidate.class_key
        self._lost_frames = 0
        self._clear_pending()
        return _remove_candidate(detections, candidate)

    def _apply_locked(
        self,
        detections: tuple[Any, ...],
        candidates: tuple[_Candidate, ...],
        frame_shape: Sequence[int],
    ) -> ExclusionResult:
        assert self._locked_box is not None
        assert self._locked_class is not None
        tracking_candidates = _tracking_candidates(
            detections,
            frame_shape,
            self._locked_class,
        )
        matches = tuple(
            candidate
            for candidate in tracking_candidates
            if _association_score(self._locked_box, candidate.box) is not None
        )

        # Zero matches means the avatar was missed.  Multiple matches means an
        # opponent or a duplicate box overlaps the track.  Both cases keep all
        # detections and age the lock rather than risk hiding the wrong one.
        if len(matches) != 1:
            self._clear_handoff()
            self._lost_frames += 1
            aim_safe = len(matches) == 0 and not candidates
            if self._lost_frames > self.lost_grace_frames:
                self._locked_box = None
                self._locked_class = None
                self._lost_frames = 0
                self._clear_pending()
                aim_safe = False
            return ExclusionResult(detections, 0, aim_safe=aim_safe)

        candidate = matches[0]
        self._lost_frames = 0
        if _is_strong_continuation(self._locked_box, candidate.box):
            self._locked_box = candidate.box
            self._clear_handoff()
            return _remove_candidate(detections, candidate)

        # A material but still plausible jump can be an opponent entering the
        # old avatar box.  Require another short stable sequence before moving
        # the lock; all detections remain visible during that confirmation.
        if (
            self._handoff_box is None
            or self._handoff_class != candidate.class_key
            or _association_score(self._handoff_box, candidate.box) is None
        ):
            self._handoff_box = candidate.box
            self._handoff_class = candidate.class_key
            self._handoff_hits = 1
        else:
            self._handoff_box = candidate.box
            self._handoff_hits += 1
        if self._handoff_hits < self.handoff_confirm_frames:
            return ExclusionResult(detections, 0)

        self._locked_box = candidate.box
        self._clear_handoff()
        return _remove_candidate(detections, candidate)

    def _start_pending(self, candidate: _Candidate) -> None:
        self._pending_box = candidate.box
        self._pending_class = candidate.class_key
        self._pending_hits = 1

    def _clear_pending(self) -> None:
        self._pending_box = None
        self._pending_class = None
        self._pending_hits = 0

    def _clear_handoff(self) -> None:
        self._handoff_box = None
        self._handoff_class = None
        self._handoff_hits = 0


def exclude_self_avatar(
    detections: Iterable[Any],
    frame_shape: Sequence[int],
    zone: NormalizedBottomZone,
) -> ExclusionResult:
    """One-frame utility that removes one unambiguous player-like candidate.

    The live pipeline uses :class:`SelfAvatarFilter` so a transient detection
    cannot be removed immediately.  This helper remains useful for tests and
    consumers that explicitly want stateless behavior.
    """

    materialized = tuple(detections)
    candidates = _candidates(materialized, frame_shape, zone)
    if len(candidates) != 1:
        return ExclusionResult(materialized, 0)
    return _remove_candidate(materialized, candidates[0])


def is_player_like(detection: Any) -> bool:
    """Return whether a detection label names a person/player-style class."""

    label = getattr(detection, "class_name", None)
    if label is None:
        label = getattr(detection, "label", "")
    tokens = re.findall(r"[a-z0-9]+", str(label).casefold())
    if any(_token_matches_words(token, OTHER_PLAYER_LABEL_WORDS) for token in tokens):
        return False
    return any(_token_matches_words(token, PLAYER_LABEL_WORDS) for token in tokens)


def _token_matches_words(token: str, words: frozenset[str]) -> bool:
    return any(token == word or re.fullmatch(rf"{re.escape(word)}\d+", token) for word in words)


def _candidates(
    detections: tuple[Any, ...],
    frame_shape: Sequence[int],
    zone: NormalizedBottomZone,
) -> tuple[_Candidate, ...]:
    candidates: list[_Candidate] = []
    for index, detection in enumerate(detections):
        box = _detection_box(detection)
        if not is_player_like(detection):
            continue
        score = zone.candidate_score(
            box,
            frame_shape,
            maximum_aspect_ratio=MAX_SELF_BOX_ASPECT_RATIO,
        )
        normalized = _normalized_box(box, frame_shape)
        if score is None or normalized is None:
            continue
        candidates.append(
            _Candidate(
                index=index,
                detection=detection,
                class_key=_class_key(detection),
                box=normalized,
                prior_score=score,
            )
        )
    return tuple(candidates)


def _tracking_candidates(
    detections: tuple[Any, ...],
    frame_shape: Sequence[int],
    class_key: str,
) -> tuple[_Candidate, ...]:
    """Return same-class tracks after acquisition, independent of anchor zone."""

    candidates: list[_Candidate] = []
    for index, detection in enumerate(detections):
        if not is_player_like(detection) or _class_key(detection) != class_key:
            continue
        normalized = _normalized_box(_detection_box(detection), frame_shape)
        if normalized is None:
            continue
        candidates.append(
            _Candidate(
                index=index,
                detection=detection,
                class_key=class_key,
                box=normalized,
                prior_score=(0.0, 0.0, 0.0),
            )
        )
    return tuple(candidates)


def _remove_candidate(
    detections: tuple[Any, ...],
    candidate: _Candidate,
) -> ExclusionResult:
    retained = tuple(
        detection
        for index, detection in enumerate(detections)
        if index != candidate.index
    )
    return ExclusionResult(retained, 1, candidate.detection, aim_safe=True)


def _association_score(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    previous_area = _box_area(previous)
    current_area = _box_area(current)
    if previous_area <= 0.0 or current_area <= 0.0:
        return None
    area_ratio = current_area / previous_area
    if not 0.55 <= area_ratio <= 1.8:
        return None

    previous_anchor = ((previous[0] + previous[2]) * 0.5, previous[3])
    current_anchor = ((current[0] + current[2]) * 0.5, current[3])
    delta_x = abs(previous_anchor[0] - current_anchor[0])
    delta_y = abs(previous_anchor[1] - current_anchor[1])
    iou = _box_iou(previous, current)
    if iou < 0.20 and not (delta_x <= 0.07 and delta_y <= 0.055):
        return None
    return iou, -(delta_x + delta_y)


def _is_strong_continuation(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
) -> bool:
    previous_area = _box_area(previous)
    current_area = _box_area(current)
    if previous_area <= 0.0 or current_area <= 0.0:
        return False
    area_ratio = current_area / previous_area
    previous_anchor = ((previous[0] + previous[2]) * 0.5, previous[3])
    current_anchor = ((current[0] + current[2]) * 0.5, current[3])
    iou = _box_iou(previous, current)
    return (
        0.72 <= area_ratio <= 1.38
        and (
            iou >= 0.70
            or (
                abs(previous_anchor[0] - current_anchor[0]) <= 0.025
                and abs(previous_anchor[1] - current_anchor[1]) <= 0.025
                and iou >= 0.60
            )
        )
    )


def _class_key(detection: Any) -> str:
    label = getattr(detection, "class_name", None)
    if label is None:
        label = getattr(detection, "label", "")
    return " ".join(re.findall(r"[a-z0-9]+", str(label).casefold()))


def _detection_box(detection: Any) -> Sequence[float]:
    box = getattr(detection, "box", None)
    if box is None:
        box = getattr(detection, "xyxy", None)
    if box is None:
        raise TypeError("each detection must expose a box or xyxy attribute")
    return box


def _normalized_box(
    box: Sequence[float],
    frame_shape: Sequence[int],
) -> tuple[float, float, float, float] | None:
    if len(box) != 4:
        raise ValueError("a detection box must contain exactly four coordinates")
    frame_height, frame_width = _frame_dimensions(frame_shape)
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    clipped = (
        min(float(frame_width), max(0.0, x1)) / frame_width,
        min(float(frame_height), max(0.0, y1)) / frame_height,
        min(float(frame_width), max(0.0, x2)) / frame_width,
        min(float(frame_height), max(0.0, y2)) / frame_height,
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def _frame_dimensions(frame_shape: Sequence[int]) -> tuple[int, int]:
    if len(frame_shape) < 2:
        raise ValueError("frame shape must contain height and width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("frame height and width must be greater than zero")
    return height, width
