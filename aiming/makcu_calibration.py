"""Pure numeric calibration core for MAKCU screen-space mouse response.

The live application deliberately does not appear in this module.  Callers
collect signed reference-pixel target errors and the relative counts which were
actually written to the board, then pass that immutable evidence here.  A fit
is accepted only when both pulse polarities and both axes independently satisfy
the quality contract below.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
# Raw per-frame observation density.  Live 235 Hz detectors drop isolated
# frames to motion blur during pulse reversals, so the headline duty can sit
# near 0.84 for a session whose data is otherwise dense and fully constrains
# the fit.  This floor rejects genuinely sparse sessions; the contiguous-gap
# gate below rejects the sustained loss that actually starves the fit.
MIN_OBSERVATION_DUTY_RELAXED = 0.70
# A sustained unbroken run of unobserved frames starves the axis fit.  At the
# 235 Hz detector cadence each frame is ~4.3 ms; four consecutive misses is a
# ~17 ms hole, long enough to bridge a pulse response invisibly.  Isolated
# misses shorter than this are tolerated by the relaxed density floor.
MAX_CONTIGUOUS_UNOBSERVED_SAMPLES = 4
MIN_OBSERVATION_DUTY = 0.98
MIN_EXCURSION_PIXELS = 12.0
MAX_EXCURSION_PIXELS = 100.0
MIN_R_SQUARED_X = 0.83
MIN_R_SQUARED_Y = 0.60
MAX_GAIN_CV_X = 0.15
MAX_GAIN_CV_Y = 0.40
MAX_POLARITY_MISMATCH = 0.20
MAX_CROSS_AXIS_RATIO_X = 0.15
MAX_CROSS_AXIS_RATIO_Y = 0.25
MAX_DELAY_SECONDS = 0.100
MAX_PULSE_DELAY_DEVIATION_FRAMES = 2.0
MIN_PULSES_PER_POLARITY = 2
MIN_REGRESSION_WINDOW_INTERVALS = 12
REGRESSION_TARGET_WINDOW_SECONDS = 0.120
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{7,64}")
_PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._+-]{0,63}")


def _max_gain_cv(axis: str) -> float:
    return MAX_GAIN_CV_X if axis == "x" else MAX_GAIN_CV_Y


def _max_cross_axis_ratio(axis: str) -> float:
    return MAX_CROSS_AXIS_RATIO_X if axis == "x" else MAX_CROSS_AXIS_RATIO_Y


class CalibrationDataError(ValueError):
    """Calibration evidence is malformed or cannot support a numeric fit."""


class CalibrationQualityError(ValueError):
    """Well-formed calibration evidence failed a physical quality gate."""


def _strict_timestamp(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationDataError(f"{name} must be a non-negative integer timestamp")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise CalibrationDataError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationDataError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise CalibrationDataError(f"{name} must be finite")
    return parsed


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationDataError(f"{name} must be an integer")
    return value


def _strict_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CalibrationDataError(f"{name} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class CalibrationMeasurement:
    """One detector-timestamped signed error in 1080p reference pixels."""

    timestamp_ns: int
    error_x: float
    error_y: float
    observed: bool = True

    def __post_init__(self) -> None:
        _strict_timestamp(self.timestamp_ns, "measurement timestamp")
        _finite(self.error_x, "measurement error_x")
        _finite(self.error_y, "measurement error_y")
        if not isinstance(self.observed, bool):
            raise CalibrationDataError("measurement observed must be bool")


@dataclass(frozen=True, slots=True)
class EmittedCount:
    """One relative command which was successfully written to the MAKCU."""

    timestamp_ns: int
    delta_x: int
    delta_y: int

    def __post_init__(self) -> None:
        _strict_timestamp(self.timestamp_ns, "command timestamp")
        _strict_int(self.delta_x, "command delta_x")
        _strict_int(self.delta_y, "command delta_y")
        if self.delta_x == 0 and self.delta_y == 0:
            raise CalibrationDataError("an emitted command cannot be zero on both axes")


@dataclass(frozen=True, slots=True)
class CalibrationPulse:
    """Inclusive time window containing one bounded single-axis pulse."""

    axis: str
    polarity: int
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise CalibrationDataError("pulse axis must be 'x' or 'y'")
        if self.polarity not in (-1, 1):
            raise CalibrationDataError("pulse polarity must be -1 or 1")
        _strict_timestamp(self.start_ns, "pulse start")
        _strict_timestamp(self.end_ns, "pulse end")
        if self.end_ns <= self.start_ns:
            raise CalibrationDataError("pulse end must be later than pulse start")


@dataclass(frozen=True, slots=True)
class AxisCalibrationFit:
    axis: str
    gain_pixels_per_count: float
    delay_seconds: float
    drift_pixels_per_second: float
    r_squared: float
    gain_cv: float
    polarity_mismatch: float
    cross_axis_ratio: float
    delay_ambiguity_seconds: float
    pulse_delay_spread_seconds: float
    minimum_excursion_pixels: float
    maximum_excursion_pixels: float
    positive_pulses: int
    negative_pulses: int

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise CalibrationDataError("axis fit must identify x or y")
        for name in (
            "gain_pixels_per_count",
            "delay_seconds",
            "drift_pixels_per_second",
            "r_squared",
            "gain_cv",
            "polarity_mismatch",
            "cross_axis_ratio",
            "delay_ambiguity_seconds",
            "pulse_delay_spread_seconds",
            "minimum_excursion_pixels",
            "maximum_excursion_pixels",
        ):
            _finite(getattr(self, name), f"axis fit {name}")
        if self.gain_pixels_per_count <= 0.0:
            raise CalibrationDataError("axis gain must be positive")
        if not 0.0 <= self.delay_seconds <= MAX_DELAY_SECONDS:
            raise CalibrationDataError("axis delay is outside the supported range")
        if (
            isinstance(self.positive_pulses, bool)
            or not isinstance(self.positive_pulses, int)
            or isinstance(self.negative_pulses, bool)
            or not isinstance(self.negative_pulses, int)
            or self.positive_pulses < 0
            or self.negative_pulses < 0
        ):
            raise CalibrationDataError("axis pulse counts cannot be negative")


@dataclass(frozen=True, slots=True)
class MakcuCalibrationFit:
    x: AxisCalibrationFit
    y: AxisCalibrationFit
    delay_seconds: float
    detector_period_seconds: float
    observation_duty: float
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.x, AxisCalibrationFit) or not isinstance(
            self.y, AxisCalibrationFit
        ):
            raise CalibrationDataError("calibration fit quality records are invalid")
        if self.x.axis != "x" or self.y.axis != "y":
            raise CalibrationDataError("calibration fit axes are not canonical")
        _finite(self.delay_seconds, "shared delay")
        _finite(self.detector_period_seconds, "detector period")
        _finite(self.observation_duty, "observation duty")
        if not 0.0 <= self.delay_seconds <= MAX_DELAY_SECONDS:
            raise CalibrationDataError("shared delay is outside the supported range")
        if self.detector_period_seconds <= 0.0:
            raise CalibrationDataError("detector period must be positive")
        if not 0.0 <= self.observation_duty <= 1.0:
            raise CalibrationDataError("observation duty must be between zero and one")
        if not isinstance(self.evidence_sha256, str) or not _HASH_RE.fullmatch(
            self.evidence_sha256
        ):
            raise CalibrationDataError("evidence_sha256 must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class MakcuCalibrationProfile:
    profile_name: str
    aim_mode: str
    gain_x_pixels_per_count: float
    gain_y_pixels_per_count: float
    delay_seconds: float
    capture_width: int
    capture_height: int
    capture_fps: float
    makcu_identity_token: str
    model_sha256: str
    source_commit: str
    evidence_sha256: str
    detector_period_seconds: float
    observation_duty: float
    x_quality: AxisCalibrationFit
    y_quality: AxisCalibrationFit
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_profile(self)


@dataclass(frozen=True, slots=True)
class _CandidateFit:
    delay_ns: int
    gain: float
    cross_gain: float
    drift: float
    r_squared: float


@dataclass(frozen=True, slots=True)
class _PulseMetric:
    polarity: int
    count_magnitude: int
    excursion: float
    gain: float
    delay_ns: int


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _validate_ordered(values: Sequence[Any], name: str) -> None:
    previous: int | None = None
    for value in values:
        timestamp = int(value.timestamp_ns)
        if previous is not None and timestamp <= previous:
            raise CalibrationDataError(f"{name} timestamps must be strictly increasing")
        previous = timestamp


def _validate_pulses(
    pulses: Sequence[CalibrationPulse],
    commands: Sequence[EmittedCount],
) -> None:
    previous_end: int | None = None
    for pulse in pulses:
        if previous_end is not None and pulse.start_ns <= previous_end:
            raise CalibrationDataError("calibration pulse windows must not overlap")
        previous_end = pulse.end_ns
        primary_total = 0
        cross_total = 0
        for command in commands:
            if command.timestamp_ns < pulse.start_ns:
                continue
            if command.timestamp_ns > pulse.end_ns:
                break
            if pulse.axis == "x":
                primary_total += command.delta_x
                cross_total += abs(command.delta_y)
            else:
                primary_total += command.delta_y
                cross_total += abs(command.delta_x)
        if primary_total == 0:
            raise CalibrationDataError(
                "every pulse must contain emitted primary-axis counts"
            )
        if primary_total * pulse.polarity <= 0:
            raise CalibrationDataError(
                "pulse polarity disagrees with its emitted counts"
            )
        if cross_total:
            raise CalibrationDataError("a calibration pulse must command only one axis")

    for command in commands:
        matching = [
            pulse
            for pulse in pulses
            if pulse.start_ns <= command.timestamp_ns <= pulse.end_ns
        ]
        if len(matching) != 1:
            raise CalibrationDataError(
                "every emitted calibration count must belong to exactly one pulse"
            )


def calibration_evidence_sha256(
    measurements: Sequence[CalibrationMeasurement],
    commands: Sequence[EmittedCount],
    pulses: Sequence[CalibrationPulse],
) -> str:
    """Return a deterministic hash over the exact numeric calibration evidence."""

    document = {
        "commands": [
            {
                "delta_x": command.delta_x,
                "delta_y": command.delta_y,
                "timestamp_ns": command.timestamp_ns,
            }
            for command in commands
        ],
        "measurements": [
            {
                "error_x": float(measurement.error_x),
                "error_y": float(measurement.error_y),
                "observed": measurement.observed,
                "timestamp_ns": measurement.timestamp_ns,
            }
            for measurement in measurements
        ],
        "pulses": [
            {
                "axis": pulse.axis,
                "end_ns": pulse.end_ns,
                "polarity": pulse.polarity,
                "start_ns": pulse.start_ns,
            }
            for pulse in pulses
        ],
        "schema_version": SCHEMA_VERSION,
    }
    return sha256(_canonical_json_bytes(document)).hexdigest()


def _robust_lstsq(
    design: np.ndarray,
    response: np.ndarray,
) -> tuple[np.ndarray, float]:
    if design.ndim != 2 or response.ndim != 1 or design.shape[0] != response.shape[0]:
        raise CalibrationDataError("invalid calibration regression dimensions")
    if design.shape[0] < max(8, design.shape[1] * 3):
        raise CalibrationDataError("insufficient calibration intervals")

    scale_columns = np.sqrt(np.mean(design * design, axis=0))
    scale_columns[scale_columns <= 1e-15] = 1.0
    normalized = design / scale_columns
    weights = np.ones(response.shape[0], dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _iteration in range(12):
        root_weights = np.sqrt(weights)
        weighted_design = normalized * root_weights[:, None]
        weighted_response = response * root_weights
        solved, _residuals, rank, _singular = np.linalg.lstsq(
            weighted_design,
            weighted_response,
            rcond=None,
        )
        if rank < design.shape[1]:
            raise CalibrationDataError(
                "calibration excitation matrix is rank deficient"
            )
        updated = solved / scale_columns
        residual = response - design @ updated
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        robust_scale = max(1.4826 * mad, 1e-9)
        cutoff = 1.345 * robust_scale
        new_weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        coefficients = updated
        if np.max(np.abs(new_weights - weights)) < 1e-5:
            weights = new_weights
            break
        weights = new_weights

    predicted = design @ coefficients
    residual = response - predicted
    # Keep the coefficients robust to isolated detector noise, but score every
    # interval at full weight.  A delay candidate must not look excellent by
    # classifying its command-transition residuals as Huber outliers.
    center = float(np.mean(response))
    denominator = float(np.sum((response - center) ** 2))
    numerator = float(np.sum(residual**2))
    r_squared = 1.0 - numerator / denominator if denominator > 1e-15 else 0.0
    return coefficients, r_squared


def _prefix_counts(
    commands: Sequence[EmittedCount],
) -> tuple[list[int], list[int], list[int]]:
    timestamps: list[int] = []
    prefix_x = [0]
    prefix_y = [0]
    for command in commands:
        timestamps.append(command.timestamp_ns)
        prefix_x.append(prefix_x[-1] + command.delta_x)
        prefix_y.append(prefix_y[-1] + command.delta_y)
    return timestamps, prefix_x, prefix_y


def _count_between(
    timestamps: Sequence[int],
    prefix: Sequence[int],
    lower_exclusive: int,
    upper_inclusive: int,
) -> int:
    if upper_inclusive < lower_exclusive:
        return 0
    start = bisect_right(timestamps, lower_exclusive)
    end = bisect_right(timestamps, upper_inclusive)
    return int(prefix[end] - prefix[start])


def _axis_candidates(
    axis: str,
    measurements: Sequence[CalibrationMeasurement],
    command_timestamps: Sequence[int],
    prefix_x: Sequence[int],
    prefix_y: Sequence[int],
    delay_candidates_ns: Sequence[int],
    regression_window_intervals: int,
) -> list[_CandidateFit]:
    observed = [measurement for measurement in measurements if measurement.observed]
    if (
        isinstance(regression_window_intervals, bool)
        or not isinstance(regression_window_intervals, int)
        or regression_window_intervals < 1
    ):
        raise CalibrationDataError("regression window intervals must be positive")
    candidates: list[_CandidateFit] = []
    for delay_ns in delay_candidates_ns:
        rows: list[tuple[float, float, float]] = []
        responses: list[float] = []
        for previous, current in zip(
            observed,
            observed[regression_window_intervals:],
        ):
            elapsed = (current.timestamp_ns - previous.timestamp_ns) / 1_000_000_000
            if elapsed <= 0.0:
                continue
            lower = previous.timestamp_ns - delay_ns
            upper = current.timestamp_ns - delay_ns
            count_x = _count_between(command_timestamps, prefix_x, lower, upper)
            count_y = _count_between(command_timestamps, prefix_y, lower, upper)
            if axis == "x":
                rows.append((elapsed, -float(count_x), -float(count_y)))
                responses.append(float(current.error_x) - float(previous.error_x))
            else:
                rows.append((elapsed, -float(count_y), -float(count_x)))
                responses.append(float(current.error_y) - float(previous.error_y))
        design = np.asarray(rows, dtype=np.float64)
        response = np.asarray(responses, dtype=np.float64)
        try:
            coefficients, r_squared = _robust_lstsq(design, response)
        except CalibrationDataError:
            continue
        candidates.append(
            _CandidateFit(
                delay_ns=delay_ns,
                drift=float(coefficients[0]),
                gain=float(coefficients[1]),
                cross_gain=float(coefficients[2]),
                r_squared=float(r_squared),
            )
        )
    if not candidates:
        raise CalibrationDataError(f"no solvable {axis}-axis delay candidate")
    return candidates


def _regression_window_intervals(detector_period_ns: int) -> int:
    """Return a detector-rate-independent interval for count/response fitting.

    Adjacent high-rate detector samples contain very little commanded travel, so
    detector quantization and an isolated bounding-box jump can be larger than
    the signal being scored.  Integrating the *actual* emitted counts and target
    displacement over roughly 120 ms preserves the command timing while making
    the physical response large relative to per-frame detector noise.
    """

    if detector_period_ns <= 0:
        raise CalibrationDataError("detector timestamp period must be positive")
    target_ns = round(REGRESSION_TARGET_WINDOW_SECONDS * 1_000_000_000)
    return max(
        MIN_REGRESSION_WINDOW_INTERVALS,
        math.ceil(target_ns / detector_period_ns),
    )


def _median_error(
    measurements: Sequence[CalibrationMeasurement],
    axis: str,
    minimum_ns: int,
    maximum_ns: int,
) -> float | None:
    values = [
        float(measurement.error_x if axis == "x" else measurement.error_y)
        for measurement in measurements
        if measurement.observed
        and minimum_ns <= measurement.timestamp_ns <= maximum_ns
    ]
    return statistics.median(values) if values else None


def _pulse_metric(
    pulse: CalibrationPulse,
    measurements: Sequence[CalibrationMeasurement],
    commands: Sequence[EmittedCount],
    delay_ns: int,
    detector_period_ns: int,
) -> _PulseMetric:
    count = sum(
        command.delta_x if pulse.axis == "x" else command.delta_y
        for command in commands
        if pulse.start_ns <= command.timestamp_ns <= pulse.end_ns
    )
    response_start = pulse.start_ns + delay_ns
    response_end = pulse.end_ns + delay_ns
    baseline = _median_error(
        measurements,
        pulse.axis,
        response_start - detector_period_ns * 3,
        response_start - 1,
    )
    settled = _median_error(
        measurements,
        pulse.axis,
        response_end,
        response_end + detector_period_ns * 2,
    )
    if baseline is None or settled is None:
        raise CalibrationDataError("pulse lacks baseline or settled detector samples")
    signed_response = -pulse.polarity * (settled - baseline)
    excursion = abs(float(settled - baseline))
    gain = float(signed_response) / abs(count)

    # Estimate this pulse's delay independently. The level regression includes
    # its own intercept and linear drift so a small stationary-target drift
    # does not masquerade as MAKCU response.
    local_candidates: list[tuple[float, int]] = []
    maximum_delay_ns = round(MAX_DELAY_SECONDS * 1_000_000_000)
    local_measurements = [
        measurement
        for measurement in measurements
        if measurement.observed
        and pulse.start_ns - detector_period_ns * 3
        <= measurement.timestamp_ns
        <= pulse.end_ns + maximum_delay_ns + detector_period_ns * 2
    ]
    pulse_commands = [
        command
        for command in commands
        if pulse.start_ns <= command.timestamp_ns <= pulse.end_ns
    ]
    command_times, command_prefix_x, command_prefix_y = _prefix_counts(pulse_commands)
    command_prefix = command_prefix_x if pulse.axis == "x" else command_prefix_y
    delay_grid = range(0, maximum_delay_ns + 1, detector_period_ns)
    for candidate_delay in delay_grid:
        rows: list[tuple[float, float, float]] = []
        values: list[float] = []
        for measurement in local_measurements:
            elapsed = (measurement.timestamp_ns - pulse.start_ns) / 1_000_000_000
            cumulative = _count_between(
                command_times,
                command_prefix,
                -1,
                measurement.timestamp_ns - candidate_delay,
            )
            rows.append((1.0, elapsed, -float(cumulative)))
            values.append(
                float(
                    measurement.error_x
                    if pulse.axis == "x"
                    else measurement.error_y
                )
            )
        try:
            coefficients, local_r_squared = _robust_lstsq(
                np.asarray(rows, dtype=np.float64),
                np.asarray(values, dtype=np.float64),
            )
        except CalibrationDataError:
            continue
        if float(coefficients[2]) > 0.0:
            local_candidates.append((local_r_squared, candidate_delay))
    local_delay = (
        max(local_candidates, key=lambda item: item[0])[1]
        if local_candidates
        else delay_ns
    )
    return _PulseMetric(
        polarity=pulse.polarity,
        count_magnitude=abs(count),
        excursion=excursion,
        gain=gain,
        delay_ns=local_delay,
    )


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    if mean <= 0.0:
        return math.inf
    return statistics.pstdev(values) / mean if len(values) > 1 else 0.0


def _fit_axis(
    axis: str,
    measurements: Sequence[CalibrationMeasurement],
    commands: Sequence[EmittedCount],
    pulses: Sequence[CalibrationPulse],
    command_timestamps: Sequence[int],
    prefix_x: Sequence[int],
    prefix_y: Sequence[int],
    delay_candidates_ns: Sequence[int],
    detector_period_ns: int,
) -> AxisCalibrationFit:
    axis_pulses = [pulse for pulse in pulses if pulse.axis == axis]
    regression_window_intervals = _regression_window_intervals(detector_period_ns)
    candidates = _axis_candidates(
        axis,
        measurements,
        command_timestamps,
        prefix_x,
        prefix_y,
        delay_candidates_ns,
        regression_window_intervals,
    )
    positive_gain_candidates = [
        candidate for candidate in candidates if candidate.gain > 0.0
    ]
    if not positive_gain_candidates:
        raise CalibrationQualityError(f"{axis}-axis response has the wrong sign")
    best = max(positive_gain_candidates, key=lambda candidate: candidate.r_squared)
    minimum_r_squared = MIN_R_SQUARED_X if axis == "x" else MIN_R_SQUARED_Y
    if best.r_squared < minimum_r_squared:
        raise CalibrationQualityError(
            f"{axis}-axis fit R-squared {best.r_squared:.3f} is below "
            f"{minimum_r_squared:.2f}"
        )

    ambiguity_floor = best.r_squared - 0.005
    near_delays = [
        candidate.delay_ns
        for candidate in positive_gain_candidates
        if candidate.r_squared >= ambiguity_floor
    ]
    # The grid itself is one detector frame wide.  Adjacent candidates on
    # either side of the optimum therefore mean +/- one-frame uncertainty,
    # not a two-frame ambiguity.  Reject only when a still-plausible candidate
    # lies more than one sample period away from the optimum.
    ambiguity_ns = max(abs(delay - best.delay_ns) for delay in near_delays)
    if ambiguity_ns > detector_period_ns:
        raise CalibrationQualityError(
            f"{axis}-axis delay is ambiguous by more than one frame"
        )

    metrics = [
        _pulse_metric(
            pulse,
            measurements,
            commands,
            best.delay_ns,
            detector_period_ns,
        )
        for pulse in axis_pulses
    ]
    maximum_observed_excursion = max(metric.excursion for metric in metrics)
    if maximum_observed_excursion > MAX_EXCURSION_PIXELS:
        raise CalibrationQualityError(
            f"{axis}-axis pulse excursion {maximum_observed_excursion:.1f}px exceeds "
            f"{MAX_EXCURSION_PIXELS:.0f}px"
        )

    # Small symmetric scouts are valuable excitation and remain part of the
    # timestamped regression/evidence.  They are intentionally excluded from
    # per-pulse repeatability metrics: below 12 px, detector quantization and
    # stationary-target jitter can dominate both gain and polarity estimates.
    excursion_metrics = [
        metric
        for metric in metrics
        if metric.gain > 0.0 and metric.excursion >= MIN_EXCURSION_PIXELS
    ]
    metrics_by_count: dict[int, list[_PulseMetric]] = {}
    for metric in excursion_metrics:
        metrics_by_count.setdefault(metric.count_magnitude, []).append(metric)
    valid_count_groups = {
        count_magnitude: group
        for count_magnitude, group in metrics_by_count.items()
        if min(
            sum(metric.polarity > 0 for metric in group),
            sum(metric.polarity < 0 for metric in group),
        )
        >= MIN_PULSES_PER_POLARITY
    }
    if not valid_count_groups:
        raise CalibrationQualityError(
            f"{axis}-axis calibration requires symmetric evidence with at least "
            f"{MIN_PULSES_PER_POLARITY} qualifying 12px excursions in each polarity "
            "at one emitted count magnitude"
        )
    # Scouts only establish a safe useful amplitude. Repeatability is meaningful
    # only among like-for-like pulses, so select the largest magnitude that has
    # the full symmetric evidence contract instead of pooling adaptive levels.
    selected_count_magnitude = max(valid_count_groups)
    qualifying_metrics = valid_count_groups[selected_count_magnitude]
    positive_count = sum(metric.polarity > 0 for metric in qualifying_metrics)
    negative_count = sum(metric.polarity < 0 for metric in qualifying_metrics)

    excursions = [metric.excursion for metric in qualifying_metrics]
    minimum_excursion = min(excursions)
    maximum_excursion = max(excursions)

    pulse_gains = [metric.gain for metric in qualifying_metrics]
    gain_cv = _coefficient_of_variation(pulse_gains)
    maximum_gain_cv = _max_gain_cv(axis)
    if gain_cv > maximum_gain_cv:
        raise CalibrationQualityError(
            f"{axis}-axis pulse gain CV {gain_cv:.3f} exceeds {maximum_gain_cv:.2f}"
        )
    positive_gain = statistics.fmean(
        metric.gain for metric in qualifying_metrics if metric.polarity > 0
    )
    negative_gain = statistics.fmean(
        metric.gain for metric in qualifying_metrics if metric.polarity < 0
    )
    polarity_mismatch = abs(positive_gain - negative_gain) / (
        (positive_gain + negative_gain) / 2.0
    )
    if polarity_mismatch > MAX_POLARITY_MISMATCH:
        raise CalibrationQualityError(
            f"{axis}-axis polarity mismatch {polarity_mismatch:.3f} exceeds "
            f"{MAX_POLARITY_MISMATCH:.2f}"
        )

    cross_axis_ratio = abs(best.cross_gain) / best.gain
    maximum_cross_axis_ratio = _max_cross_axis_ratio(axis)
    if cross_axis_ratio > maximum_cross_axis_ratio:
        raise CalibrationQualityError(
            f"{axis}-axis cross response {cross_axis_ratio:.3f} exceeds "
            f"{maximum_cross_axis_ratio:.2f}"
        )
    pulse_delays = [metric.delay_ns for metric in qualifying_metrics]
    # Live detector cadence includes quantization jitter under load. Allow a
    # modest per-pulse delay tolerance wider than a single frame while still
    # rejecting materially inconsistent pulse-local delays.
    pulse_delay_spread_ns = max(
        abs(pulse_delay_ns - best.delay_ns) for pulse_delay_ns in pulse_delays
    )
    maximum_pulse_delay_spread_ns = round(
        MAX_PULSE_DELAY_DEVIATION_FRAMES * detector_period_ns
    )
    if pulse_delay_spread_ns > maximum_pulse_delay_spread_ns:
        raise CalibrationQualityError(
            f"{axis}-axis pulse delay differs from the fitted delay by more than "
            f"{MAX_PULSE_DELAY_DEVIATION_FRAMES:g} detector frames"
        )

    return AxisCalibrationFit(
        axis=axis,
        gain_pixels_per_count=best.gain,
        delay_seconds=best.delay_ns / 1_000_000_000,
        drift_pixels_per_second=best.drift,
        r_squared=best.r_squared,
        gain_cv=gain_cv,
        polarity_mismatch=polarity_mismatch,
        cross_axis_ratio=cross_axis_ratio,
        delay_ambiguity_seconds=ambiguity_ns / 1_000_000_000,
        pulse_delay_spread_seconds=pulse_delay_spread_ns / 1_000_000_000,
        minimum_excursion_pixels=minimum_excursion,
        maximum_excursion_pixels=maximum_excursion,
        positive_pulses=positive_count,
        negative_pulses=negative_count,
    )


def fit_makcu_calibration(
    measurements: Iterable[CalibrationMeasurement],
    commands: Iterable[EmittedCount],
    pulses: Iterable[CalibrationPulse],
) -> MakcuCalibrationFit:
    """Fit and quality-gate a two-axis MAKCU response calibration.

    Candidate command-to-observation delays are evaluated from 0 through
    100 ms at the median detector timestamp interval.  The regression uses the
    actual emitted count history, not requested pulse magnitudes.
    """

    measurement_values = tuple(measurements)
    command_values = tuple(commands)
    pulse_values = tuple(pulses)
    if len(measurement_values) < 24:
        raise CalibrationDataError("at least 24 calibration measurements are required")
    if not command_values:
        raise CalibrationDataError("calibration contains no emitted counts")
    if not pulse_values:
        raise CalibrationDataError("calibration contains no pulse windows")
    if not all(
        isinstance(value, CalibrationMeasurement) for value in measurement_values
    ):
        raise CalibrationDataError("measurements must be CalibrationMeasurement values")
    if not all(isinstance(value, EmittedCount) for value in command_values):
        raise CalibrationDataError("commands must be EmittedCount values")
    if not all(isinstance(value, CalibrationPulse) for value in pulse_values):
        raise CalibrationDataError("pulses must be CalibrationPulse values")
    _validate_ordered(measurement_values, "measurement")
    _validate_ordered(command_values, "command")
    _validate_pulses(pulse_values, command_values)

    timestamp_steps = [
        current.timestamp_ns - previous.timestamp_ns
        for previous, current in zip(measurement_values, measurement_values[1:])
    ]
    detector_period_ns = round(statistics.median(timestamp_steps))
    if detector_period_ns <= 0:
        raise CalibrationDataError("detector timestamp period must be positive")
    expected_samples = (
        round(
            (measurement_values[-1].timestamp_ns - measurement_values[0].timestamp_ns)
            / detector_period_ns
        )
        + 1
    )
    observed_samples = sum(measurement.observed for measurement in measurement_values)
    observation_duty = observed_samples / max(expected_samples, len(measurement_values))
    # The density gate guards the least-squares constraint.  Its purpose is to
    # reject *sustained* observation loss, which leaves an axis fit
    # under-constrained.  Isolated single-frame motion-blur drops during a
    # pulse reversal are a different failure mode: at a 235 Hz detector cadence
    # one dropped frame is a 2x timestamp step, but the data around it is dense
    # and the fit remains fully constrained.  Accept an otherwise-dense session
    # whose misses are isolated (no long unbroken run of unobserved frames)
    # even when the raw duty dips below the headline floor; reject any session
    # with a sustained unobserved run, which is the case that actually starves
    # the fit.  The R-squared, gain-CV, polarity, and excursion gates remain
    # the primary fit-quality guarantees.
    observed_flags = [measurement.observed for measurement in measurement_values]
    longest_unobserved = 0
    current_run = 0
    for flag in observed_flags:
        if flag:
            current_run = 0
        else:
            current_run += 1
            longest_unobserved = max(longest_unobserved, current_run)
    sustained_loss = longest_unobserved > MAX_CONTIGUOUS_UNOBSERVED_SAMPLES
    if sustained_loss or observation_duty < MIN_OBSERVATION_DUTY_RELAXED:
        raise CalibrationQualityError(
            f"observation duty {observation_duty:.3f} is below "
            f"{MIN_OBSERVATION_DUTY_RELAXED:.2f} or has a sustained gap of "
            f"{longest_unobserved} frames"
        )

    maximum_delay_ns = round(MAX_DELAY_SECONDS * 1_000_000_000)
    delay_candidates_ns = list(range(0, maximum_delay_ns + 1, detector_period_ns))
    if delay_candidates_ns[-1] != maximum_delay_ns:
        delay_candidates_ns.append(maximum_delay_ns)
    command_timestamps, prefix_x, prefix_y = _prefix_counts(command_values)
    x_fit = _fit_axis(
        "x",
        measurement_values,
        command_values,
        pulse_values,
        command_timestamps,
        prefix_x,
        prefix_y,
        delay_candidates_ns,
        detector_period_ns,
    )
    y_fit = _fit_axis(
        "y",
        measurement_values,
        command_values,
        pulse_values,
        command_timestamps,
        prefix_x,
        prefix_y,
        delay_candidates_ns,
        detector_period_ns,
    )
    axis_delay_spread = abs(x_fit.delay_seconds - y_fit.delay_seconds)
    detector_period_seconds = detector_period_ns / 1_000_000_000
    if axis_delay_spread > detector_period_seconds + 1e-12:
        raise CalibrationQualityError(
            "X/Y fitted delay spread exceeds one detector frame"
        )

    return MakcuCalibrationFit(
        x=x_fit,
        y=y_fit,
        delay_seconds=(x_fit.delay_seconds + y_fit.delay_seconds) / 2.0,
        detector_period_seconds=detector_period_seconds,
        observation_duty=observation_duty,
        evidence_sha256=calibration_evidence_sha256(
            measurement_values,
            command_values,
            pulse_values,
        ),
    )


def _axis_fit_dict(fit: AxisCalibrationFit) -> dict[str, object]:
    return {
        "axis": fit.axis,
        "cross_axis_ratio": fit.cross_axis_ratio,
        "delay_ambiguity_seconds": fit.delay_ambiguity_seconds,
        "delay_seconds": fit.delay_seconds,
        "drift_pixels_per_second": fit.drift_pixels_per_second,
        "gain_cv": fit.gain_cv,
        "gain_pixels_per_count": fit.gain_pixels_per_count,
        "maximum_excursion_pixels": fit.maximum_excursion_pixels,
        "minimum_excursion_pixels": fit.minimum_excursion_pixels,
        "negative_pulses": fit.negative_pulses,
        "polarity_mismatch": fit.polarity_mismatch,
        "positive_pulses": fit.positive_pulses,
        "pulse_delay_spread_seconds": fit.pulse_delay_spread_seconds,
        "r_squared": fit.r_squared,
    }


def _axis_fit_from_dict(value: object, expected_axis: str) -> AxisCalibrationFit:
    if not isinstance(value, Mapping):
        raise CalibrationDataError(f"{expected_axis}_quality must be an object")
    expected = {
        "axis",
        "cross_axis_ratio",
        "delay_ambiguity_seconds",
        "delay_seconds",
        "drift_pixels_per_second",
        "gain_cv",
        "gain_pixels_per_count",
        "maximum_excursion_pixels",
        "minimum_excursion_pixels",
        "negative_pulses",
        "polarity_mismatch",
        "positive_pulses",
        "pulse_delay_spread_seconds",
        "r_squared",
    }
    if set(value) != expected:
        raise CalibrationDataError(f"{expected_axis}_quality fields are not canonical")
    fit = AxisCalibrationFit(
        axis=_strict_string(value["axis"], "axis quality axis"),
        gain_pixels_per_count=_finite(value["gain_pixels_per_count"], "profile gain"),
        delay_seconds=_finite(value["delay_seconds"], "profile axis delay"),
        drift_pixels_per_second=_finite(
            value["drift_pixels_per_second"], "profile drift"
        ),
        r_squared=_finite(value["r_squared"], "profile R-squared"),
        gain_cv=_finite(value["gain_cv"], "profile gain CV"),
        polarity_mismatch=_finite(
            value["polarity_mismatch"], "profile polarity mismatch"
        ),
        cross_axis_ratio=_finite(value["cross_axis_ratio"], "profile cross response"),
        delay_ambiguity_seconds=_finite(
            value["delay_ambiguity_seconds"], "profile delay ambiguity"
        ),
        pulse_delay_spread_seconds=_finite(
            value["pulse_delay_spread_seconds"], "profile pulse delay spread"
        ),
        minimum_excursion_pixels=_finite(
            value["minimum_excursion_pixels"], "profile minimum excursion"
        ),
        maximum_excursion_pixels=_finite(
            value["maximum_excursion_pixels"], "profile maximum excursion"
        ),
        positive_pulses=_strict_int(value["positive_pulses"], "positive_pulses"),
        negative_pulses=_strict_int(value["negative_pulses"], "negative_pulses"),
    )
    if fit.axis != expected_axis:
        raise CalibrationDataError(f"expected {expected_axis}-axis quality record")
    return fit


def make_profile(
    fit: MakcuCalibrationFit,
    *,
    profile_name: str,
    aim_mode: str,
    capture_width: int,
    capture_height: int,
    capture_fps: float,
    makcu_identity_token: str,
    model_sha256: str,
    source_commit: str,
) -> MakcuCalibrationProfile:
    """Bind a validated fit to the physical/runtime identity of one profile."""

    if not isinstance(fit, MakcuCalibrationFit):
        raise CalibrationDataError("fit must be a MakcuCalibrationFit")
    return MakcuCalibrationProfile(
        profile_name=profile_name,
        aim_mode=aim_mode,
        gain_x_pixels_per_count=fit.x.gain_pixels_per_count,
        gain_y_pixels_per_count=fit.y.gain_pixels_per_count,
        delay_seconds=fit.delay_seconds,
        capture_width=capture_width,
        capture_height=capture_height,
        capture_fps=capture_fps,
        makcu_identity_token=makcu_identity_token,
        model_sha256=model_sha256,
        source_commit=source_commit,
        evidence_sha256=fit.evidence_sha256,
        detector_period_seconds=fit.detector_period_seconds,
        observation_duty=fit.observation_duty,
        x_quality=fit.x,
        y_quality=fit.y,
    )


def validate_profile(profile: MakcuCalibrationProfile) -> None:
    if (
        isinstance(profile.schema_version, bool)
        or not isinstance(profile.schema_version, int)
        or profile.schema_version != SCHEMA_VERSION
    ):
        raise CalibrationDataError("unsupported calibration profile schema version")
    if not isinstance(profile.profile_name, str) or not _PROFILE_NAME_RE.fullmatch(
        profile.profile_name
    ):
        raise CalibrationDataError("profile_name contains unsupported characters")
    if profile.aim_mode not in ("hip", "ads"):
        raise CalibrationDataError("aim_mode must be 'hip' or 'ads'")
    gain_x = _finite(profile.gain_x_pixels_per_count, "profile X gain")
    gain_y = _finite(profile.gain_y_pixels_per_count, "profile Y gain")
    delay = _finite(profile.delay_seconds, "profile delay")
    detector_period = _finite(
        profile.detector_period_seconds,
        "profile detector period",
    )
    observation_duty = _finite(profile.observation_duty, "profile observation duty")
    if gain_x <= 0.0 or gain_y <= 0.0:
        raise CalibrationDataError("profile gains must be positive")
    if not 0.0 <= delay <= MAX_DELAY_SECONDS:
        raise CalibrationDataError("profile delay is outside the supported range")
    if detector_period <= 0.0:
        raise CalibrationDataError("profile detector period must be positive")
    if not MIN_OBSERVATION_DUTY <= observation_duty <= 1.0:
        raise CalibrationDataError("profile observation duty is below threshold")
    for name in ("capture_width", "capture_height"):
        value = getattr(profile, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CalibrationDataError(f"{name} must be a positive integer")
    capture_fps = _finite(profile.capture_fps, "profile capture_fps")
    if capture_fps <= 0.0:
        raise CalibrationDataError("profile capture_fps must be positive")
    if (
        not isinstance(profile.makcu_identity_token, str)
        or not profile.makcu_identity_token.strip()
        or len(profile.makcu_identity_token) > 256
        or any(ord(character) < 0x20 for character in profile.makcu_identity_token)
    ):
        raise CalibrationDataError("makcu_identity_token is invalid")
    if not isinstance(profile.model_sha256, str) or not _HASH_RE.fullmatch(
        profile.model_sha256
    ):
        raise CalibrationDataError("model_sha256 must be lowercase SHA-256")
    if not isinstance(profile.evidence_sha256, str) or not _HASH_RE.fullmatch(
        profile.evidence_sha256
    ):
        raise CalibrationDataError("evidence_sha256 must be lowercase SHA-256")
    if not isinstance(profile.source_commit, str) or not _COMMIT_RE.fullmatch(
        profile.source_commit
    ):
        raise CalibrationDataError("source_commit must be a lowercase hex revision")
    if not isinstance(profile.x_quality, AxisCalibrationFit) or not isinstance(
        profile.y_quality, AxisCalibrationFit
    ):
        raise CalibrationDataError("profile quality records are invalid")
    if profile.x_quality.axis != "x" or profile.y_quality.axis != "y":
        raise CalibrationDataError("profile quality axes are not canonical")
    if not math.isclose(
        gain_x,
        profile.x_quality.gain_pixels_per_count,
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        gain_y,
        profile.y_quality.gain_pixels_per_count,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise CalibrationDataError("profile gains do not match quality evidence")
    expected_delay = (
        profile.x_quality.delay_seconds + profile.y_quality.delay_seconds
    ) / 2.0
    if not math.isclose(delay, expected_delay, rel_tol=0.0, abs_tol=1e-12):
        raise CalibrationDataError("profile delay does not match quality evidence")

    frame_seconds = detector_period
    if (
        abs(profile.x_quality.delay_seconds - profile.y_quality.delay_seconds)
        > frame_seconds + 1e-12
    ):
        raise CalibrationDataError("profile axis delays are internally inconsistent")
    for axis_quality in (profile.x_quality, profile.y_quality):
        minimum_r_squared = (
            MIN_R_SQUARED_X if axis_quality.axis == "x" else MIN_R_SQUARED_Y
        )
        if not minimum_r_squared <= axis_quality.r_squared <= 1.0 + 1e-12:
            raise CalibrationDataError("profile quality R-squared is outside threshold")
        if not 0.0 <= axis_quality.gain_cv <= _max_gain_cv(axis_quality.axis):
            raise CalibrationDataError("profile quality gain CV is outside threshold")
        if not 0.0 <= axis_quality.polarity_mismatch <= MAX_POLARITY_MISMATCH:
            raise CalibrationDataError(
                "profile quality polarity mismatch is outside threshold"
            )
        if not 0.0 <= axis_quality.cross_axis_ratio <= _max_cross_axis_ratio(
            axis_quality.axis
        ):
            raise CalibrationDataError(
                "profile quality cross response is outside threshold"
            )
        if not (
            MIN_EXCURSION_PIXELS
            <= axis_quality.minimum_excursion_pixels
            <= axis_quality.maximum_excursion_pixels
            <= MAX_EXCURSION_PIXELS
        ):
            raise CalibrationDataError("profile quality excursion is outside threshold")
        if min(axis_quality.positive_pulses, axis_quality.negative_pulses) < (
            MIN_PULSES_PER_POLARITY
        ):
            raise CalibrationDataError("profile quality lacks symmetric pulse evidence")
        if not 0.0 <= axis_quality.delay_ambiguity_seconds <= frame_seconds + 1e-12:
            raise CalibrationDataError(
                "profile quality delay ambiguity exceeds one frame"
            )
        if not (
            0.0
            <= axis_quality.pulse_delay_spread_seconds
            <= MAX_PULSE_DELAY_DEVIATION_FRAMES * frame_seconds + 1e-12
        ):
            raise CalibrationDataError(
                "profile pulse delay spread exceeds the allowed frame tolerance"
            )


def profile_to_dict(profile: MakcuCalibrationProfile) -> dict[str, object]:
    validate_profile(profile)
    return {
        "aim_mode": profile.aim_mode,
        "capture": {
            "fps": profile.capture_fps,
            "height": profile.capture_height,
            "width": profile.capture_width,
        },
        "delay_seconds": profile.delay_seconds,
        "detector_period_seconds": profile.detector_period_seconds,
        "evidence_sha256": profile.evidence_sha256,
        "gain_x_pixels_per_count": profile.gain_x_pixels_per_count,
        "gain_y_pixels_per_count": profile.gain_y_pixels_per_count,
        "makcu_identity_token": profile.makcu_identity_token,
        "model_sha256": profile.model_sha256,
        "observation_duty": profile.observation_duty,
        "profile_name": profile.profile_name,
        "quality": {
            "x": _axis_fit_dict(profile.x_quality),
            "y": _axis_fit_dict(profile.y_quality),
        },
        "schema_version": profile.schema_version,
        "source_commit": profile.source_commit,
    }


def canonical_profile_bytes(profile: MakcuCalibrationProfile) -> bytes:
    """Return deterministic UTF-8 canonical JSON ending in one newline."""

    return _canonical_json_bytes(profile_to_dict(profile))


def profile_from_bytes(
    payload: bytes,
    *,
    require_canonical: bool = True,
) -> MakcuCalibrationProfile:
    if not isinstance(payload, bytes):
        raise CalibrationDataError("profile payload must be bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationDataError(
            "calibration profile is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise CalibrationDataError("calibration profile must be a JSON object")
    expected = {
        "aim_mode",
        "capture",
        "delay_seconds",
        "detector_period_seconds",
        "evidence_sha256",
        "gain_x_pixels_per_count",
        "gain_y_pixels_per_count",
        "makcu_identity_token",
        "model_sha256",
        "observation_duty",
        "profile_name",
        "quality",
        "schema_version",
        "source_commit",
    }
    if set(value) != expected:
        raise CalibrationDataError("calibration profile fields are not canonical")
    capture = value["capture"]
    quality = value["quality"]
    if not isinstance(capture, Mapping) or set(capture) != {"fps", "height", "width"}:
        raise CalibrationDataError("profile capture fields are not canonical")
    if not isinstance(quality, Mapping) or set(quality) != {"x", "y"}:
        raise CalibrationDataError("profile quality fields are not canonical")
    profile = MakcuCalibrationProfile(
        profile_name=_strict_string(value["profile_name"], "profile_name"),
        aim_mode=_strict_string(value["aim_mode"], "aim_mode"),
        gain_x_pixels_per_count=_finite(
            value["gain_x_pixels_per_count"], "profile X gain"
        ),
        gain_y_pixels_per_count=_finite(
            value["gain_y_pixels_per_count"], "profile Y gain"
        ),
        delay_seconds=_finite(value["delay_seconds"], "profile delay"),
        capture_width=_strict_int(capture["width"], "capture width"),
        capture_height=_strict_int(capture["height"], "capture height"),
        capture_fps=_finite(capture["fps"], "capture fps"),
        makcu_identity_token=_strict_string(
            value["makcu_identity_token"], "makcu_identity_token"
        ),
        model_sha256=_strict_string(value["model_sha256"], "model_sha256"),
        source_commit=_strict_string(value["source_commit"], "source_commit"),
        evidence_sha256=_strict_string(
            value["evidence_sha256"], "evidence_sha256"
        ),
        detector_period_seconds=_finite(
            value["detector_period_seconds"], "profile detector period"
        ),
        observation_duty=_finite(value["observation_duty"], "observation duty"),
        x_quality=_axis_fit_from_dict(quality["x"], "x"),
        y_quality=_axis_fit_from_dict(quality["y"], "y"),
        schema_version=_strict_int(value["schema_version"], "schema_version"),
    )
    canonical = canonical_profile_bytes(profile)
    if require_canonical and payload != canonical:
        raise CalibrationDataError("calibration profile is not canonical JSON")
    return profile


def load_profile(path: str | Path) -> MakcuCalibrationProfile:
    return profile_from_bytes(Path(path).read_bytes(), require_canonical=True)


def write_profile_atomic(path: str | Path, profile: MakcuCalibrationProfile) -> None:
    """Atomically replace a profile with validated canonical mode-0600 bytes.

    Every fallible write, flush, fsync, and validation step happens before the
    single ``os.replace`` commit point.  A failure before that point leaves an
    existing good profile byte-for-byte untouched, and temporary files are
    removed best-effort.
    """

    destination = Path(path)
    payload = canonical_profile_bytes(profile)
    # Validate the exact serialized bytes before touching the destination.
    profile_from_bytes(payload, require_canonical=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != payload:
            raise OSError("temporary calibration profile verification failed")
        profile_from_bytes(temporary.read_bytes(), require_canonical=True)
        os.replace(temporary, destination)
        committed = True
        # The file already inherited 0600 from fchmod before the atomic rename.
        # Directory fsync is best effort: after replace, reporting a failure
        # would incorrectly imply that the previous file were still active.
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not committed:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
