"""Bounded optical-flow phase correction for delayed direct-head results.

The direct-head model observes an exact captured frame, but its result becomes
available several frames later.  This module keeps a small grayscale history
and moves the *observed* head point through those intervening images.  It does
not predict beyond the newest image, choose an identity, or manufacture a head
when the pixel evidence is weak.

Sparse pyramidal Lucas--Kanade flow is accepted only when a spatially diverse
set of features inside the confirmed head box survives a forward/backward
check and agrees on one robust translation.  Every resource and time bound is
explicit so this helper can be attached to a high-rate capture source without
creating an unbounded frame queue.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, hypot, isfinite
from numbers import Integral
from threading import Lock
from typing import TypeAlias

import cv2
import numpy as np


Box: TypeAlias = tuple[float, float, float, float]
Point: TypeAlias = tuple[float, float]


def _finite_box(value: Sequence[float], name: str = "head_box") -> Box:
    if len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    result = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite coordinates")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{name} must have positive width and height")
    return result


def _finite_point(value: Sequence[float], name: str = "anchor_point") -> Point:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two coordinates")
    result = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite coordinates")
    return result


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def direct_head_box_center(head_box: Sequence[float]) -> Point:
    """Return the center used by the current direct-head localization path."""

    x1, y1, x2, y2 = _finite_box(head_box)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass(frozen=True, slots=True)
class HeadFlowConfig:
    """Hard resource and acceptance bounds for head-interior optical flow."""

    max_features: int = 48
    min_features: int = 6
    quality_level: float = 0.01
    min_feature_distance: float = 3.0
    feature_block_size: int = 5
    # One keeps features strictly inside the model box.  A future live-qualified
    # value may include a small ring of same-target hair/shoulder texture for
    # very small boxes, but the default does not broaden the trust region.
    feature_roi_scale: float = 1.0
    roi_inset_fraction: float = 0.10
    crosshair_exclusion_radius_pixels: float = 14.0
    lk_window_size: int = 21
    lk_pyramid_levels: int = 3
    max_lk_error: float = 40.0
    max_forward_backward_error: float = 0.8
    max_inlier_residual: float = 1.5
    min_inlier_fraction: float = 0.60
    min_feature_span_fraction: float = 0.18
    max_frame_displacement_pixels: float = 64.0
    # At the measured capture cadence, a 42--54 ms old result commonly needs
    # 9--12 image-to-image hops.  The former 40 ms / 8-hop prototype rejected
    # ordinary live results before examining their pixel evidence.
    max_hops: int = 20
    max_phase_advance_seconds: float = 0.075
    max_history_frames: int = 20
    # Sixteen 1920x1080 grayscale frames fit under this cap (about 73 ms of
    # history at the observed ~205 Hz capture rate).  Frame count remains a
    # second independent bound for smaller sources.
    max_history_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        integer_bounds = {
            "max_features": (self.max_features, 8, 128),
            "min_features": (self.min_features, 3, self.max_features),
            "feature_block_size": (self.feature_block_size, 3, 15),
            "lk_window_size": (self.lk_window_size, 5, 61),
            "lk_pyramid_levels": (self.lk_pyramid_levels, 0, 5),
            "max_hops": (self.max_hops, 1, 64),
            "max_history_frames": (self.max_history_frames, 2, 64),
            "max_history_bytes": (
                self.max_history_bytes,
                1024 * 1024,
                512 * 1024 * 1024,
            ),
        }
        for name, (value, lower, upper) in integer_bounds.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if not lower <= int(value) <= int(upper):
                raise ValueError(f"{name} must be between {lower} and {upper}")
        if self.feature_block_size % 2 == 0:
            raise ValueError("feature_block_size must be odd")
        if self.lk_window_size % 2 == 0:
            raise ValueError("lk_window_size must be odd")

        for name in (
            "quality_level",
            "roi_inset_fraction",
            "min_inlier_fraction",
            "min_feature_span_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if self.roi_inset_fraction >= 0.5:
            raise ValueError("roi_inset_fraction must be less than 0.5")
        for name in (
            "min_feature_distance",
            "max_lk_error",
            "max_forward_backward_error",
            "max_inlier_residual",
            "max_frame_displacement_pixels",
            "max_phase_advance_seconds",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        feature_roi_scale = float(self.feature_roi_scale)
        if not isfinite(feature_roi_scale) or not 1.0 <= feature_roi_scale <= 2.0:
            raise ValueError("feature_roi_scale must be finite and between 1 and 2")
        crosshair_radius = float(self.crosshair_exclusion_radius_pixels)
        if not isfinite(crosshair_radius) or crosshair_radius < 0.0:
            raise ValueError(
                "crosshair_exclusion_radius_pixels must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class HeadFlowMeasurement:
    displacement: Point
    features_detected: int
    features_consistent: int
    inliers: int
    inlier_fraction: float
    median_forward_backward_error: float
    median_translation_residual: float


@dataclass(frozen=True, slots=True)
class PhaseAdvancedHead:
    """One exact head point translated only through observed newer frames."""

    point: Point
    head_box: Box
    anchor_timestamp_ns: int
    source_timestamp_ns: int
    identity_generation: int
    hops: int
    frames_spanned: int
    flow_measurements: int
    strategy: str
    minimum_inlier_fraction: float
    maximum_forward_backward_error: float
    # The pixels used to measure translation may cover a larger, textured
    # same-target region than the small head localization box.  Keep that ROI
    # separate so translation never changes the anatomical aim geometry.
    feature_box: Box | None = None


def _gray_u8(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.dtype != np.uint8:
        raise ValueError("frame must use uint8 pixels")
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("frame must be grayscale HxW or BGR HxWx3")
    if gray.shape[0] <= 0 or gray.shape[1] <= 0:
        raise ValueError("frame dimensions must be positive")
    return np.ascontiguousarray(gray)


def _clipped_feature_roi(
    box: Box,
    shape: tuple[int, int],
    inset_fraction: float,
) -> tuple[int, int, int, int] | None:
    height, width = shape
    inset_x = (box[2] - box[0]) * inset_fraction
    inset_y = (box[3] - box[1]) * inset_fraction
    left = max(0, int(ceil(box[0] + inset_x)))
    top = max(0, int(ceil(box[1] + inset_y)))
    right = min(width, int(floor(box[2] - inset_x)))
    bottom = min(height, int(floor(box[3] - inset_y)))
    if right - left < 4 or bottom - top < 4:
        return None
    return left, top, right, bottom


def _flow_crop_bounds(
    box: Box,
    shape: tuple[int, int],
    settings: HeadFlowConfig,
) -> tuple[int, int, int, int]:
    """Bound LK pyramid construction to one head-sized search patch."""

    height, width = shape
    margin = int(
        ceil(
            settings.max_frame_displacement_pixels
            + settings.lk_window_size * 0.5
        )
    )
    return (
        max(0, int(floor(box[0])) - margin),
        max(0, int(floor(box[1])) - margin),
        min(width, int(ceil(box[2])) + margin),
        min(height, int(ceil(box[3])) + margin),
    )


def _scaled_box(box: Box, scale: float) -> Box:
    center_x = (box[0] + box[2]) * 0.5
    center_y = (box[1] + box[3]) * 0.5
    half_width = (box[2] - box[0]) * scale * 0.5
    half_height = (box[3] - box[1]) * scale * 0.5
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def measure_head_translation(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    head_box: Sequence[float],
    *,
    config: HeadFlowConfig | None = None,
) -> HeadFlowMeasurement | None:
    """Measure one fail-closed observed translation of a confirmed head ROI."""

    settings = HeadFlowConfig() if config is None else config
    if not isinstance(settings, HeadFlowConfig):
        raise TypeError("config must be HeadFlowConfig or None")
    box = _finite_box(head_box)
    previous = _gray_u8(previous_frame)
    current = _gray_u8(current_frame)
    if previous.shape != current.shape:
        raise ValueError("previous and current frames must have identical dimensions")

    feature_box = _scaled_box(box, settings.feature_roi_scale)
    crop_left, crop_top, crop_right, crop_bottom = _flow_crop_bounds(
        feature_box,
        previous.shape,
        settings,
    )
    previous_patch = previous[crop_top:crop_bottom, crop_left:crop_right]
    current_patch = current[crop_top:crop_bottom, crop_left:crop_right]
    local_box = (
        feature_box[0] - crop_left,
        feature_box[1] - crop_top,
        feature_box[2] - crop_left,
        feature_box[3] - crop_top,
    )
    bounds = _clipped_feature_roi(
        local_box,
        previous_patch.shape,
        settings.roi_inset_fraction,
    )
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    mask = np.zeros(previous_patch.shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 255

    # A game crosshair is fixed in screen coordinates and can otherwise become
    # the strongest apparent "head" texture once the target reaches center.
    # Removing it is deliberately fail-closed: if the confirmed head is too
    # small to leave enough independent texture, wait for the next model seed.
    exclusion_radius = settings.crosshair_exclusion_radius_pixels
    if exclusion_radius > 0.0:
        center = (
            previous.shape[1] // 2 - crop_left,
            previous.shape[0] // 2 - crop_top,
        )
        cv2.circle(mask, center, int(ceil(exclusion_radius)), 0, thickness=-1)

    previous_lk = cv2.goodFeaturesToTrack(
        previous_patch,
        maxCorners=settings.max_features,
        qualityLevel=settings.quality_level,
        minDistance=settings.min_feature_distance,
        blockSize=settings.feature_block_size,
        mask=mask,
        useHarrisDetector=False,
    )
    if previous_lk is None or len(previous_lk) < settings.min_features:
        return None

    lk_parameters = {
        "winSize": (settings.lk_window_size, settings.lk_window_size),
        "maxLevel": settings.lk_pyramid_levels,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
        "minEigThreshold": 0.0001,
    }
    current_lk, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_patch,
        current_patch,
        previous_lk,
        None,
        **lk_parameters,
    )
    if current_lk is None or forward_status is None or forward_error is None:
        return None
    returned_lk, backward_status, backward_error = cv2.calcOpticalFlowPyrLK(
        current_patch,
        previous_patch,
        current_lk,
        None,
        **lk_parameters,
    )
    if returned_lk is None or backward_status is None or backward_error is None:
        return None

    previous_points = np.asarray(previous_lk, dtype=np.float32).reshape(-1, 2)
    current_points = np.asarray(current_lk, dtype=np.float32).reshape(-1, 2)
    returned_points = np.asarray(returned_lk, dtype=np.float32).reshape(-1, 2)
    forward_errors = np.asarray(forward_error, dtype=np.float32).reshape(-1)
    backward_errors = np.asarray(backward_error, dtype=np.float32).reshape(-1)
    forward_backward = np.linalg.norm(returned_points - previous_points, axis=1)
    consistent = (
        np.asarray(forward_status).reshape(-1).astype(bool)
        & np.asarray(backward_status).reshape(-1).astype(bool)
        & np.isfinite(previous_points).all(axis=1)
        & np.isfinite(current_points).all(axis=1)
        & np.isfinite(returned_points).all(axis=1)
        & np.isfinite(forward_errors)
        & np.isfinite(backward_errors)
        & np.isfinite(forward_backward)
        & (forward_errors <= settings.max_lk_error)
        & (backward_errors <= settings.max_lk_error)
        & (forward_backward <= settings.max_forward_backward_error)
    )
    if int(np.count_nonzero(consistent)) < settings.min_features:
        return None

    source = previous_points[consistent]
    destination = current_points[consistent]
    displacement = destination - source
    initial = np.median(displacement, axis=0)
    residual = np.linalg.norm(displacement - initial, axis=1)
    inlier_mask = residual <= settings.max_inlier_residual
    inlier_count = int(np.count_nonzero(inlier_mask))
    consistent_count = len(displacement)
    if inlier_count < settings.min_features:
        return None
    inlier_fraction = inlier_count / consistent_count
    if inlier_fraction < settings.min_inlier_fraction:
        return None

    inlier_source = source[inlier_mask]
    required_x_span = (right - left) * settings.min_feature_span_fraction
    required_y_span = (bottom - top) * settings.min_feature_span_fraction
    if (
        float(np.ptp(inlier_source[:, 0])) < required_x_span
        or float(np.ptp(inlier_source[:, 1])) < required_y_span
    ):
        return None

    inlier_displacement = displacement[inlier_mask]
    final = np.median(inlier_displacement, axis=0)
    dx, dy = float(final[0]), float(final[1])
    if not isfinite(dx) or not isfinite(dy):
        return None
    if hypot(dx, dy) > settings.max_frame_displacement_pixels:
        return None

    inlier_residual = np.linalg.norm(inlier_displacement - final, axis=1)
    return HeadFlowMeasurement(
        displacement=(dx, dy),
        features_detected=len(previous_points),
        features_consistent=consistent_count,
        inliers=inlier_count,
        inlier_fraction=inlier_fraction,
        median_forward_backward_error=float(
            np.median(forward_backward[consistent])
        ),
        median_translation_residual=float(np.median(inlier_residual)),
    )


@dataclass(frozen=True, slots=True)
class _RememberedGrayFrame:
    image: np.ndarray
    source_timestamp_ns: int
    identity_generation: int


class HeadFlowPhaseAdvancer:
    """Keep bounded grayscale history and replay flow from an exact anchor."""

    def __init__(self, config: HeadFlowConfig | None = None) -> None:
        if config is not None and not isinstance(config, HeadFlowConfig):
            raise TypeError("config must be HeadFlowConfig or None")
        self.config = HeadFlowConfig() if config is None else config
        self._history: deque[_RememberedGrayFrame] = deque()
        self._history_bytes = 0
        self._lock = Lock()

    @property
    def history_size(self) -> int:
        with self._lock:
            return len(self._history)

    @property
    def history_bytes(self) -> int:
        with self._lock:
            return self._history_bytes

    @property
    def history_span_seconds(self) -> float:
        with self._lock:
            if len(self._history) < 2:
                return 0.0
            return (
                self._history[-1].source_timestamp_ns
                - self._history[0].source_timestamp_ns
            ) / 1_000_000_000.0

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._history_bytes = 0

    def remember(
        self,
        frame: np.ndarray,
        *,
        source_timestamp_ns: int,
        identity_generation: int,
    ) -> None:
        """Own one grayscale copy; source timestamps must be strictly monotonic."""

        timestamp_ns = _non_negative_integer(
            source_timestamp_ns,
            "source_timestamp_ns",
        )
        generation = _non_negative_integer(
            identity_generation,
            "identity_generation",
        )
        converted = _gray_u8(frame)
        # cvtColor already returns an owned array for BGR input.  A grayscale
        # input may alias the caller, so copy only in that case instead of
        # paying for a redundant 2 MiB copy on every 1080p capture.
        gray = (
            converted.copy()
            if np.shares_memory(converted, frame)
            else converted
        )
        remembered = _RememberedGrayFrame(gray, timestamp_ns, generation)

        with self._lock:
            if self._history:
                latest = self._history[-1]
                if timestamp_ns <= latest.source_timestamp_ns:
                    raise ValueError(
                        "source_timestamp_ns must be strictly increasing"
                    )
                if (
                    generation != latest.identity_generation
                    or gray.shape != latest.image.shape
                ):
                    self._history.clear()
                    self._history_bytes = 0
            self._history.append(remembered)
            self._history_bytes += gray.nbytes
            while self._history and (
                len(self._history) > self.config.max_history_frames
                or self._history_bytes > self.config.max_history_bytes
            ):
                removed = self._history.popleft()
                self._history_bytes -= removed.image.nbytes

    def advance(
        self,
        head_box: Sequence[float],
        *,
        feature_box: Sequence[float] | None = None,
        anchor_point: Sequence[float] | None = None,
        anchor_timestamp_ns: int,
        identity_generation: int,
    ) -> PhaseAdvancedHead | None:
        """Move an exact old-frame head point to the newest remembered frame.

        ``anchor_point`` must be the model's exact localization point.  It is
        intentionally separate from ``head_box`` so this helper preserves the
        current box-center semantics (and any future anatomical convention)
        instead of silently choosing a different vertical aim point.

        ``feature_box`` is a fail-closed fallback only.  Every LK step first
        tries the anatomical ``head_box`` and consults this wider same-target
        ROI only when the head contains insufficient reliable texture.
        """

        box = _finite_box(head_box)
        fallback_feature_box = (
            None
            if feature_box is None
            else _finite_box(feature_box, "feature_box")
        )
        point = (
            direct_head_box_center(box)
            if anchor_point is None
            else _finite_point(anchor_point)
        )
        anchor_ns = _non_negative_integer(
            anchor_timestamp_ns,
            "anchor_timestamp_ns",
        )
        generation = _non_negative_integer(
            identity_generation,
            "identity_generation",
        )
        with self._lock:
            history = tuple(self._history)

        anchor_index = next(
            (
                index
                for index, remembered in enumerate(history)
                if remembered.source_timestamp_ns == anchor_ns
                and remembered.identity_generation == generation
            ),
            None,
        )
        if anchor_index is None:
            return None
        replay = history[anchor_index:]
        if not replay or any(
            item.identity_generation != generation for item in replay
        ):
            return None
        hops = len(replay) - 1
        if hops > self.config.max_hops:
            return None
        source_ns = replay[-1].source_timestamp_ns
        elapsed_seconds = (source_ns - anchor_ns) / 1_000_000_000.0
        if (
            elapsed_seconds < 0.0
            or elapsed_seconds > self.config.max_phase_advance_seconds
        ):
            return None

        current_box = box
        current_feature_box = fallback_feature_box
        current_point = point
        minimum_inlier_fraction = 1.0
        maximum_forward_backward_error = 0.0
        strategy = "exact" if hops == 0 else "sequential"
        flow_measurements = 0

        def measure_with_optional_fallback(
            previous_image: np.ndarray,
            current_image: np.ndarray,
            anatomical_box: Box,
            fallback_box: Box | None,
        ) -> HeadFlowMeasurement | None:
            # Preserve the quiet, target-specific head track whenever it has
            # enough texture.  The wider verified-body ROI exists solely for
            # tiny/long-range cases where the head measurement fails closed.
            measurement = measure_head_translation(
                previous_image,
                current_image,
                anatomical_box,
                config=self.config,
            )
            if measurement is None and fallback_box is not None:
                measurement = measure_head_translation(
                    previous_image,
                    current_image,
                    fallback_box,
                    config=self.config,
                )
            return measurement

        # For a delayed result, try one bounded endpoint measurement first.
        # Rebuilding an LK pyramid for every intervening full-rate image would
        # add several milliseconds precisely when we are trying to remove
        # phase delay.  Sequential replay remains the fallback for motion too
        # large or nonlinear for the direct endpoint search.
        endpoint = None
        if hops > 1:
            endpoint = measure_with_optional_fallback(
                replay[0].image,
                replay[-1].image,
                box,
                fallback_feature_box,
            )
        if endpoint is not None:
            dx, dy = endpoint.displacement
            current_point = (point[0] + dx, point[1] + dy)
            current_box = (
                box[0] + dx,
                box[1] + dy,
                box[2] + dx,
                box[3] + dy,
            )
            current_feature_box = (
                None
                if fallback_feature_box is None
                else (
                    fallback_feature_box[0] + dx,
                    fallback_feature_box[1] + dy,
                    fallback_feature_box[2] + dx,
                    fallback_feature_box[3] + dy,
                )
            )
            minimum_inlier_fraction = endpoint.inlier_fraction
            maximum_forward_backward_error = (
                endpoint.median_forward_backward_error
            )
            strategy = "direct"
            flow_measurements = 1
        else:
            for previous, current in zip(
                replay[:-1],
                replay[1:],
                strict=True,
            ):
                measurement = measure_with_optional_fallback(
                    previous.image,
                    current.image,
                    current_box,
                    current_feature_box,
                )
                if measurement is None:
                    return None
                dx, dy = measurement.displacement
                current_point = (current_point[0] + dx, current_point[1] + dy)
                current_box = (
                    current_box[0] + dx,
                    current_box[1] + dy,
                    current_box[2] + dx,
                    current_box[3] + dy,
                )
                current_feature_box = (
                    None
                    if current_feature_box is None
                    else (
                        current_feature_box[0] + dx,
                        current_feature_box[1] + dy,
                        current_feature_box[2] + dx,
                        current_feature_box[3] + dy,
                    )
                )
                minimum_inlier_fraction = min(
                    minimum_inlier_fraction,
                    measurement.inlier_fraction,
                )
                maximum_forward_backward_error = max(
                    maximum_forward_backward_error,
                    measurement.median_forward_backward_error,
                )
                flow_measurements += 1

        return PhaseAdvancedHead(
            point=current_point,
            head_box=current_box,
            anchor_timestamp_ns=anchor_ns,
            source_timestamp_ns=source_ns,
            identity_generation=generation,
            hops=hops,
            frames_spanned=hops,
            flow_measurements=flow_measurements,
            strategy=strategy,
            minimum_inlier_fraction=minimum_inlier_fraction,
            maximum_forward_backward_error=maximum_forward_backward_error,
            feature_box=current_feature_box,
        )
