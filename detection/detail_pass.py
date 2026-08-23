"""Planning and conservative cross-pass merging for detail inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor, gcd, isfinite
from numbers import Integral
from typing import Sequence

from utils.inference_size import InferenceSizeLike, normalize_inference_size

from .types import Detection


# Cross-pass consolidation uses a measured same-player overlap floor. Unlike
# ordinary NMS, it compares only one full-frame result with one detail-pass
# result, requires the same class, and uses a one-to-one match. Results within
# either individual pass are never compared, so unmatched nearby detections
# are preserved.
CROSS_PASS_DUPLICATE_IOU = 0.50
# The detail branch exists to recover targets that the full-frame resize makes
# small. Unmatched large detail detections are mostly redundant and, on the
# audited FORT validation split, materially increased false positives. Keep
# the limit resolution-independent by expressing it at a 1080p reference
# height. A detail detection that overlaps a primary result can still replace
# it regardless of size; this limit applies only to new, unmatched evidence.
DETAIL_REFERENCE_HEIGHT = 1080.0
DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT = 96.0
DETAIL_CROP_POLICY = "centered_model_aspect_roi"
DETAIL_TARGET_CENTERED_CROP_POLICY = "target_centered_model_aspect_roi"


@dataclass(frozen=True, slots=True)
class DetailPassPlan:
    crop_policy: str
    requested_crop_size: int
    requested_crop_height: int
    applied_crop_width: int
    applied_crop_height: int
    source_width: int
    source_height: int
    model_width: int
    model_height: int
    crop_x: int
    crop_y: int
    coverage_fraction: float
    full_frame_scale: float
    detail_scale: float
    effective_linear_magnification: float
    clamped: bool
    redundant: bool

    def as_record(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


class DetailPassStats:
    """Aggregate non-sensitive geometry and branch counts for live reports."""

    def __init__(self, requested_crop_size: int | None) -> None:
        self.requested_crop_size = requested_crop_size
        self.frames_seen = 0
        self.frames_applied = 0
        self.frames_redundant = 0
        self.frames_clamped = 0
        self.last_plan: DetailPassPlan | None = None
        self.primary_detections = 0
        self.detail_detections = 0
        self.cross_pass_matches = 0
        self.detail_replacements = 0
        self.unmatched_detail_accepted = 0
        self.unmatched_detail_rejected_large = 0
        self.merged_detections = 0

    def record(self, plan: DetailPassPlan) -> None:
        self.frames_seen += 1
        self.frames_clamped += int(plan.clamped)
        self.frames_redundant += int(plan.redundant)
        self.frames_applied += int(not plan.redundant)
        self.last_plan = plan

    def record_merge(
        self,
        *,
        primary: int,
        detail: int,
        matches: int,
        replacements: int,
        unmatched_accepted: int,
        unmatched_rejected_large: int,
        merged: int,
    ) -> None:
        """Accumulate branch decisions without retaining boxes or frame data."""

        values = (
            primary,
            detail,
            matches,
            replacements,
            unmatched_accepted,
            unmatched_rejected_large,
            merged,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 0
            for value in values
        ):
            raise ValueError("detail merge counts must be non-negative integers")
        self.primary_detections += int(primary)
        self.detail_detections += int(detail)
        self.cross_pass_matches += int(matches)
        self.detail_replacements += int(replacements)
        self.unmatched_detail_accepted += int(unmatched_accepted)
        self.unmatched_detail_rejected_large += int(unmatched_rejected_large)
        self.merged_detections += int(merged)

    def snapshot(self) -> dict[str, object]:
        return {
            "enabled": self.requested_crop_size is not None,
            "crop_policy": DETAIL_CROP_POLICY,
            "requested_crop_size": self.requested_crop_size,
            "duplicate_iou_threshold": CROSS_PASS_DUPLICATE_IOU,
            "unmatched_detail_reference_height": DETAIL_REFERENCE_HEIGHT,
            "unmatched_detail_max_reference_height": (
                DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT
            ),
            "frames_seen": self.frames_seen,
            "frames_applied": self.frames_applied,
            "frames_redundant": self.frames_redundant,
            "frames_clamped": self.frames_clamped,
            "primary_detections": self.primary_detections,
            "detail_detections": self.detail_detections,
            "cross_pass_matches": self.cross_pass_matches,
            "detail_replacements": self.detail_replacements,
            "unmatched_detail_accepted": self.unmatched_detail_accepted,
            "unmatched_detail_rejected_large": (
                self.unmatched_detail_rejected_large
            ),
            "merged_detections": self.merged_detections,
            "last_plan": self.last_plan.as_record() if self.last_plan else None,
        }


def plan_detail_pass(
    frame_shape: Sequence[int],
    crop_size: int,
    model_shape_hw: InferenceSizeLike,
    *,
    center_point: Sequence[float] | None = None,
) -> DetailPassPlan:
    """Resolve one bounded model-aspect ROI without copying frame data.

    ``crop_size`` is the requested source-pixel ROI width.  The height is
    derived from the detector's exact static input aspect ratio, then both
    dimensions are reduced together if the source cannot contain that ROI.
    A square model therefore preserves the former square-crop behavior, while
    a rectangular model uses its whole tensor instead of letterboxing a square
    detail crop.  With no ``center_point`` the ROI keeps its original centered
    behavior.  A supplied source-space ``(x, y)`` center is clamped so the same
    ROI remains wholly inside the source frame.
    """

    if isinstance(crop_size, bool) or not isinstance(crop_size, Integral):
        raise TypeError("detail crop size must be a positive integer")
    requested = int(crop_size)
    if requested <= 0:
        raise ValueError("detail crop size must be a positive integer")
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    source_height = int(frame_shape[0])
    source_width = int(frame_shape[1])
    if source_height <= 0 or source_width <= 0:
        raise ValueError("frame dimensions must be positive")
    model_height, model_width = normalize_inference_size(model_shape_hw)
    target_center: tuple[float, float] | None = None
    if center_point is not None:
        if isinstance(center_point, (str, bytes)) or len(center_point) != 2:
            raise TypeError("detail center_point must be a finite (x, y) pair")
        try:
            center_x, center_y = (float(value) for value in center_point)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "detail center_point must be a finite (x, y) pair"
            ) from exc
        if not isfinite(center_x) or not isfinite(center_y):
            raise ValueError("detail center_point coordinates must be finite")
        target_center = (center_x, center_y)

    requested_height = max(
        1,
        int(round(requested * model_height / float(model_width))),
    )
    # Width is the public control because horizontal source coverage is the
    # most intuitive quantity on a widescreen capture.  A height-constrained
    # source reduces the width with the same model aspect rather than silently
    # returning to a padded square ROI.
    common = gcd(model_width, model_height)
    aspect_width = model_width // common
    aspect_height = model_height // common
    aspect_units = min(
        requested // aspect_width,
        source_width // aspect_width,
        source_height // aspect_height,
    )
    if aspect_units > 0:
        # Exact integer aspect units make the resize scale identical on both
        # axes, which keeps source-coordinate mapping exact rather than relying
        # on a rounded one-pixel letterbox dimension.
        applied_width = aspect_units * aspect_width
        applied_height = aspect_units * aspect_height
    else:
        # Tiny synthetic/source frames may be smaller than one reduced model
        # aspect unit. Keep the operation defined and in bounds; the recorded
        # scale still captures this necessarily approximate fallback.
        applied_width = min(requested, source_width)
        applied_height = min(
            source_height,
            max(1, int(round(applied_width * model_height / float(model_width)))),
        )
    if target_center is None:
        crop_x = (source_width - applied_width) // 2
        crop_y = (source_height - applied_height) // 2
        crop_policy = DETAIL_CROP_POLICY
    else:
        desired_x = floor(target_center[0] - applied_width * 0.5)
        desired_y = floor(target_center[1] - applied_height * 0.5)
        crop_x = min(max(desired_x, 0), source_width - applied_width)
        crop_y = min(max(desired_y, 0), source_height - applied_height)
        crop_policy = DETAIL_TARGET_CENTERED_CROP_POLICY
    # Running the exact source rectangle twice cannot add evidence.
    redundant = (
        applied_width == source_width and applied_height == source_height
    )
    coverage = (applied_width * applied_height) / float(
        source_width * source_height
    )
    # Compare exact full-source and detail-ROI letterbox scales.  Apart from a
    # possible one-pixel rounding difference, the model-aspect ROI consumes
    # the complete static tensor.
    full_frame_scale = min(
        model_width / float(source_width),
        model_height / float(source_height),
    )
    detail_scale = min(
        model_width / float(applied_width),
        model_height / float(applied_height),
    )
    magnification = detail_scale / full_frame_scale
    return DetailPassPlan(
        crop_policy=crop_policy,
        requested_crop_size=requested,
        requested_crop_height=requested_height,
        applied_crop_width=applied_width,
        applied_crop_height=applied_height,
        source_width=source_width,
        source_height=source_height,
        model_width=model_width,
        model_height=model_height,
        crop_x=crop_x,
        crop_y=crop_y,
        coverage_fraction=coverage,
        full_frame_scale=full_frame_scale,
        detail_scale=detail_scale,
        effective_linear_magnification=magnification,
        clamped=(
            applied_width != requested or applied_height != requested_height
        ),
        redundant=redundant,
    )


def _intersection_over_union(first: Detection, second: Detection) -> float:
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def merge_cross_pass_detections(
    primary: Sequence[Detection],
    detail: Sequence[Detection],
    *,
    duplicate_iou: float = CROSS_PASS_DUPLICATE_IOU,
    source_height: int | None = None,
    unmatched_detail_max_reference_height: float | None = None,
    stats: DetailPassStats | None = None,
) -> list[Detection]:
    """Merge overlapping, same-class, one-to-one cross-pass duplicates.

    Results from within the same pass are never compared. Candidate pairs are
    processed by descending IoU with stable index tie-breaks. The more
    confident detection wins a match; exact confidence ties retain the primary
    result. Primary order is preserved. When a source height and reference
    limit are supplied, unmatched detail results are appended only when their
    source-space height falls within that small-object limit; matched results
    remain eligible at every size.
    """

    threshold = float(duplicate_iou)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("duplicate_iou must be between 0 and 1")
    if (source_height is None) != (unmatched_detail_max_reference_height is None):
        raise ValueError(
            "source_height and unmatched_detail_max_reference_height must be "
            "provided together"
        )
    if source_height is not None:
        if isinstance(source_height, bool) or not isinstance(source_height, Integral):
            raise TypeError("source_height must be a positive integer")
        if source_height <= 0:
            raise ValueError("source_height must be a positive integer")
        assert unmatched_detail_max_reference_height is not None
        if isinstance(unmatched_detail_max_reference_height, bool):
            raise TypeError(
                "unmatched_detail_max_reference_height must be positive"
            )
        unmatched_limit = float(unmatched_detail_max_reference_height)
        if not isfinite(unmatched_limit) or unmatched_limit <= 0.0:
            raise ValueError(
                "unmatched_detail_max_reference_height must be positive"
            )
    else:
        unmatched_limit = None
    primary_items = list(primary)
    detail_items = list(detail)
    candidates: list[tuple[float, int, int]] = []
    for primary_index, primary_detection in enumerate(primary_items):
        for detail_index, detail_detection in enumerate(detail_items):
            if primary_detection.class_id != detail_detection.class_id:
                continue
            overlap = _intersection_over_union(primary_detection, detail_detection)
            if overlap >= threshold:
                candidates.append((-overlap, primary_index, detail_index))
    candidates.sort()

    matched_primary: set[int] = set()
    matched_detail: set[int] = set()
    replacements: dict[int, Detection] = {}
    detail_replacements = 0
    for _negative_overlap, primary_index, detail_index in candidates:
        if primary_index in matched_primary or detail_index in matched_detail:
            continue
        matched_primary.add(primary_index)
        matched_detail.add(detail_index)
        primary_detection = primary_items[primary_index]
        detail_detection = detail_items[detail_index]
        if detail_detection.confidence > primary_detection.confidence:
            replacements[primary_index] = detail_detection
            detail_replacements += 1
        else:
            replacements[primary_index] = primary_detection

    merged = [
        replacements.get(index, detection)
        for index, detection in enumerate(primary_items)
    ]
    unmatched_accepted = 0
    unmatched_rejected_large = 0
    for index, detection in enumerate(detail_items):
        if index in matched_detail:
            continue
        if unmatched_limit is not None:
            assert source_height is not None
            projected_height = (
                max(0.0, detection.y2 - detection.y1)
                / float(source_height)
                * DETAIL_REFERENCE_HEIGHT
            )
            if projected_height > unmatched_limit:
                unmatched_rejected_large += 1
                continue
        merged.append(detection)
        unmatched_accepted += 1
    if stats is not None:
        if not isinstance(stats, DetailPassStats):
            raise TypeError("stats must be a DetailPassStats instance or None")
        stats.record_merge(
            primary=len(primary_items),
            detail=len(detail_items),
            matches=len(matched_detail),
            replacements=detail_replacements,
            unmatched_accepted=unmatched_accepted,
            unmatched_rejected_large=unmatched_rejected_large,
            merged=len(merged),
        )
    return merged
