"""Fail-closed runtime orchestration for bounded MAKCU calibration.

This module owns no camera, detector, GUI, or process lifecycle.  A caller
feeds it one already-filtered detector observation and one *known* physical
activation state at a time.  It exclusively drives the small calibration API
on :class:`aiming.makcu.MakcuAimingController`, records the commands which the
worker actually wrote, and hands the resulting numeric evidence to
``fit_makcu_calibration``.  It never activates or writes a calibration profile.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import tempfile
from typing import Any, Literal, Mapping, Protocol

from aiming.makcu import (
    CALIBRATION_MAX_EXCURSION_COUNTS,
    CALIBRATION_MAX_RATE_COUNTS_PER_SECOND,
    CALIBRATION_MAX_SESSION_ABS_COUNTS,
    CALIBRATION_MAX_STEP_COUNTS,
    MakcuCalibrationSnapshot,
    MakcuRawActivationSnapshot,
)
from aiming.makcu_calibration import (
    AxisCalibrationFit,
    CalibrationDataError,
    CalibrationMeasurement,
    CalibrationPulse,
    CalibrationQualityError,
    EmittedCount,
    MakcuCalibrationFit,
    fit_makcu_calibration,
)


SESSION_EVIDENCE_SCHEMA_VERSION = 1
MAX_SESSION_EVIDENCE_BYTES = 32 * 1024 * 1024
REFERENCE_WIDTH = 1920.0
REFERENCE_HEIGHT = 1080.0
_TARGET_CONTINUITY_MINIMUM_IOU = 0.15
_TARGET_CONTINUITY_MINIMUM_AREA_RATIO = 0.5
_TARGET_CONTINUITY_MAXIMUM_AREA_RATIO = 2.0
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{7,64}")
_TEXT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._+:/@-]{0,255}")
_CONTEXT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")
_TARGET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class CalibrationSessionError(RuntimeError):
    """The runtime calibration session cannot safely continue."""


class CalibrationEvidenceError(ValueError):
    """A persisted session evidence artifact is malformed or noncanonical."""


class CalibrationSessionState(str, Enum):
    WAIT_RELEASE = "wait_release"
    WAIT_HOLD = "wait_hold"
    BASELINE_SETTLE = "baseline_settle"
    PULSE = "pulse"
    RESPONSE_SETTLE = "response_settle"
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _timestamp(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _strict_text(value: object, name: str, *, pattern: re.Pattern[str] = _TEXT_RE) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _bounded_text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _sanitized_reason(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    return (text[:237] + "...") if len(text) > 240 else (text or "unspecified failure")


def normalize_calibration_context(context_name: str) -> Literal["hip", "ads"]:
    """Map a named CLI context to the profile's canonical hip/ADS mode."""

    context = _strict_text(context_name, "calibration context", pattern=_CONTEXT_RE)
    normalized = context.casefold()
    for mode in ("hip", "ads"):
        if normalized == mode or any(
            normalized.startswith(f"{mode}{separator}")
            for separator in ("-", "_", "+", ".")
        ):
            return mode  # type: ignore[return-value]
    raise ValueError("calibration context must begin with 'hip' or 'ads'")


@dataclass(frozen=True, slots=True)
class CalibrationRuntimeBinding:
    """Immutable identity required to interpret or accept calibration evidence."""

    model_sha256: str
    labels_sha256: str
    source_commit: str
    build_identity: str
    backend: str
    runtime_version: str
    requested_provider: str
    active_provider: str
    active_device: str
    provider_options_sha256: str
    physical_device_token: str
    inference_width: int
    inference_height: int
    detail_pass_enabled: bool
    capture_kind: Literal["screen", "camera"]
    capture_backend: str
    capture_buffer_size: int
    capture_index: str
    capture_width: int
    capture_height: int
    capture_fps: float
    pixel_format: str
    rotation_degrees: int
    makcu_identity_token: str
    activation_button: int
    aim_label: str
    head_ratio: float
    invert_x: bool
    invert_y: bool
    context_name: str
    aim_mode: Literal["hip", "ads"]

    def __post_init__(self) -> None:
        for name in ("model_sha256", "labels_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be lowercase SHA-256")
        if not isinstance(self.source_commit, str) or _COMMIT_RE.fullmatch(
            self.source_commit
        ) is None:
            raise ValueError("source_commit must be a lowercase hex revision")
        for name in (
            "build_identity",
            "backend",
            "runtime_version",
            "requested_provider",
            "active_provider",
            "active_device",
            "capture_backend",
            "capture_index",
            "pixel_format",
            "aim_label",
        ):
            _bounded_text(getattr(self, name), name)
        if self.runtime_version.casefold() in {
            "unknown",
            "unavailable",
            "n/a",
            "none",
        }:
            raise ValueError("runtime_version must identify an exact runtime")
        if not isinstance(self.makcu_identity_token, str) or _HASH_RE.fullmatch(
            self.makcu_identity_token
        ) is None:
            raise ValueError("makcu_identity_token must be an opaque SHA-256 token")
        for name in ("provider_options_sha256", "physical_device_token"):
            value = getattr(self, name)
            if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be an opaque SHA-256 token")
        if self.capture_kind not in ("screen", "camera"):
            raise ValueError("capture_kind must be screen or camera")
        if (
            isinstance(self.capture_buffer_size, bool)
            or not isinstance(self.capture_buffer_size, int)
            or self.capture_buffer_size < 0
            or (self.capture_kind == "camera" and self.capture_buffer_size == 0)
        ):
            raise ValueError(
                "capture_buffer_size must be positive for camera and non-negative otherwise"
            )
        for name in (
            "inference_width",
            "inference_height",
            "capture_width",
            "capture_height",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.activation_button, bool)
            or not isinstance(self.activation_button, int)
            or not 0 <= self.activation_button <= 4
        ):
            raise ValueError("activation_button must be an integer in [0,4]")
        if not isinstance(self.detail_pass_enabled, bool):
            raise ValueError("detail_pass_enabled must be bool")
        if self.detail_pass_enabled:
            raise ValueError("calibration evidence cannot use the detail pass")
        if (
            isinstance(self.rotation_degrees, bool)
            or not isinstance(self.rotation_degrees, int)
            or self.rotation_degrees not in (0, 90, 180, 270)
        ):
            raise ValueError("rotation_degrees must be 0, 90, 180, or 270")
        if _finite(self.capture_fps, "capture_fps") <= 0.0:
            raise ValueError("capture_fps must be positive")
        head_ratio = _finite(self.head_ratio, "head_ratio")
        if not 0.0 <= head_ratio <= 0.5:
            raise ValueError("head_ratio must be between zero and 0.5")
        if not isinstance(self.invert_x, bool) or not isinstance(self.invert_y, bool):
            raise ValueError("axis inversion flags must be bool")
        canonical_mode = normalize_calibration_context(self.context_name)
        if self.aim_mode != canonical_mode:
            raise ValueError("aim_mode does not match the named calibration context")


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    """One strong-target decision in the perf_counter_ns timestamp domain."""

    measurement_ns: int
    error_x: float
    error_y: float
    confidence: float
    exact_label: bool
    unique_candidates: int
    self_safe: bool
    is_prediction: bool
    target_identity: str
    normalized_bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _timestamp(self.measurement_ns, "measurement_ns")
        _finite(self.error_x, "error_x")
        _finite(self.error_y, "error_y")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        for name in ("exact_label", "self_safe", "is_prediction"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if (
            isinstance(self.unique_candidates, bool)
            or not isinstance(self.unique_candidates, int)
            or self.unique_candidates < 0
        ):
            raise ValueError("unique_candidates must be a non-negative integer")
        _strict_text(
            self.target_identity,
            "target_identity",
            pattern=_TARGET_ID_RE,
        )
        if not isinstance(self.normalized_bbox, tuple) or len(self.normalized_bbox) != 4:
            raise ValueError("normalized_bbox must be a four-value tuple")
        left, top, right, bottom = (
            _finite(value, "normalized_bbox") for value in self.normalized_bbox
        )
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ValueError("normalized_bbox must be ordered inside [0,1]")


def target_within_safe_roi(
    normalized_bbox: tuple[float, float, float, float],
    margin_ratio: float,
) -> bool:
    """Return whether the complete target box, not only its head, is central."""

    margin = _finite(margin_ratio, "margin_ratio")
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin_ratio must be in [0,0.5)")
    if not isinstance(normalized_bbox, tuple) or len(normalized_bbox) != 4:
        raise ValueError("normalized_bbox must be a four-value tuple")
    left, top, right, bottom = (
        _finite(value, "normalized_bbox") for value in normalized_bbox
    )
    return (
        margin <= left < right <= 1.0 - margin
        and margin <= top < bottom <= 1.0 - margin
    )


def _target_bbox_is_continuous(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
) -> bool:
    """Conservatively associate consecutive raw calibration detections."""

    intersection_width = max(
        0.0,
        min(previous[2], current[2]) - max(previous[0], current[0]),
    )
    intersection_height = max(
        0.0,
        min(previous[3], current[3]) - max(previous[1], current[1]),
    )
    intersection = intersection_width * intersection_height
    previous_area = (previous[2] - previous[0]) * (previous[3] - previous[1])
    current_area = (current[2] - current[0]) * (current[3] - current[1])
    union = previous_area + current_area - intersection
    iou = intersection / union if union > 0.0 else 0.0
    area_ratio = current_area / previous_area if previous_area > 0.0 else math.inf
    return (
        iou >= _TARGET_CONTINUITY_MINIMUM_IOU
        and _TARGET_CONTINUITY_MINIMUM_AREA_RATIO
        <= area_ratio
        <= _TARGET_CONTINUITY_MAXIMUM_AREA_RATIO
    )


@dataclass(frozen=True, slots=True)
class CalibrationSessionConfig:
    minimum_confidence: float = 0.70
    safe_roi_margin_ratio: float = 0.08
    maximum_reference_error: float = 360.0
    release_dwell_seconds: float = 0.08
    maximum_observation_age_seconds: float = 0.020
    stationary_samples: int = 6
    stationary_span_seconds: float = 0.040
    stationary_range_pixels: float = 3.0
    stationary_speed_pixels_per_second: float = 20.0
    post_hold_settle_seconds: float = 0.300
    target_acquire_timeout_seconds: float = 2.0
    initial_settle_timeout_seconds: float = 1.5
    response_delay_seconds: float = 0.110
    response_settle_timeout_seconds: float = 0.75
    pulse_timeout_slack_seconds: float = 0.25
    held_session_timeout_seconds: float = 8.0
    arm_timeout_seconds: float = 30.0
    maximum_prehold_records: int = 2048
    maximum_evidence_records: int = 16384
    pulse_rate_counts_per_second: float = CALIBRATION_MAX_RATE_COUNTS_PER_SECOND
    initial_scout_counts: int = 50
    target_excursion_pixels: float = 30.0
    preferred_minimum_excursion_pixels: float = 20.0
    preferred_maximum_excursion_pixels: float = 60.0
    qualifying_minimum_excursion_pixels: float = 12.0
    abort_excursion_pixels: float = 100.0

    def __post_init__(self) -> None:
        unit_names = ("minimum_confidence", "safe_roi_margin_ratio")
        for name in unit_names:
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.safe_roi_margin_ratio >= 0.5:
            raise ValueError("safe_roi_margin_ratio must be below 0.5")
        for name in (
            "maximum_reference_error",
            "release_dwell_seconds",
            "maximum_observation_age_seconds",
            "stationary_span_seconds",
            "stationary_range_pixels",
            "stationary_speed_pixels_per_second",
            "post_hold_settle_seconds",
            "target_acquire_timeout_seconds",
            "initial_settle_timeout_seconds",
            "response_delay_seconds",
            "response_settle_timeout_seconds",
            "pulse_timeout_slack_seconds",
            "held_session_timeout_seconds",
            "arm_timeout_seconds",
            "pulse_rate_counts_per_second",
            "target_excursion_pixels",
            "preferred_minimum_excursion_pixels",
            "preferred_maximum_excursion_pixels",
            "qualifying_minimum_excursion_pixels",
            "abort_excursion_pixels",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.stationary_samples, bool)
            or not isinstance(self.stationary_samples, int)
            or self.stationary_samples < 3
        ):
            raise ValueError("stationary_samples must be an integer of at least three")
        for name in ("maximum_prehold_records", "maximum_evidence_records"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_evidence_records < self.maximum_prehold_records:
            raise ValueError("maximum_evidence_records cannot be below prehold records")
        if (
            isinstance(self.initial_scout_counts, bool)
            or not isinstance(self.initial_scout_counts, int)
            or not 1 <= self.initial_scout_counts <= CALIBRATION_MAX_EXCURSION_COUNTS
        ):
            raise ValueError("initial_scout_counts is outside the controller envelope")
        if self.pulse_rate_counts_per_second > (
            CALIBRATION_MAX_RATE_COUNTS_PER_SECOND
        ):
            raise ValueError("pulse rate exceeds the controller envelope")
        if not (
            self.qualifying_minimum_excursion_pixels
            <= self.preferred_minimum_excursion_pixels
            <= self.target_excursion_pixels
            <= self.preferred_maximum_excursion_pixels
            < self.abort_excursion_pixels
        ):
            raise ValueError("excursion thresholds are not ordered")


class CalibrationController(Protocol):
    @property
    def raw_activation_state(self) -> tuple[bool, bool]: ...

    @property
    def raw_activation_snapshot(self) -> MakcuRawActivationSnapshot: ...

    def enter_calibration_mode(self) -> object: ...

    def publish_calibration_lease(
        self,
        valid: bool,
        measurement_ns: int,
        token: object,
        *,
        activation_transition_sequence: int | None = None,
    ) -> None: ...

    def request_calibration_pulse(
        self,
        axis: str,
        signed_counts: int,
        bounded_rate: float,
        token: object,
    ) -> None: ...

    def calibration_snapshot(self) -> MakcuCalibrationSnapshot: ...

    def exit_calibration_mode(self, token: object) -> None: ...


@dataclass(frozen=True, slots=True)
class SessionObservationRecord:
    now_ns: int
    activation_known: bool
    activation_pressed: bool
    state: str
    observation: CalibrationObservation | None


@dataclass(frozen=True, slots=True)
class SessionPulseRecord:
    axis: str
    polarity: int
    requested_counts: int
    requested_rate: float
    request_ns: int
    event_start_index: int
    event_end_index: int
    first_emitted_ns: int | None
    last_emitted_ns: int | None
    actual_counts: int
    baseline_x: float
    baseline_y: float
    settled_x: float | None
    settled_y: float | None
    signed_response_pixels: float | None
    cross_response_pixels: float | None
    qualifying: bool
    complete: bool


@dataclass(frozen=True, slots=True)
class CalibrationSessionEvidence:
    binding: CalibrationRuntimeBinding
    outcome: Literal["success", "aborted"]
    reason: str
    cleanup_error: str | None
    evidence_complete: bool
    started_ns: int
    held_ns: int | None
    terminal_ns: int
    observations: tuple[SessionObservationRecord, ...]
    pulses: tuple[SessionPulseRecord, ...]
    emitted_events: tuple[tuple[int, int, int], ...]
    fit: MakcuCalibrationFit | None
    core_evidence_sha256: str | None
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class CalibrationSessionResult:
    outcome: Literal["success", "aborted"]
    reason: str
    fit: MakcuCalibrationFit | None
    evidence: CalibrationSessionEvidence


@dataclass(frozen=True, slots=True)
class CalibrationSessionStatus:
    state: CalibrationSessionState
    message: str
    axis: str | None
    pulse_counts: int
    emitted_abs_counts: int
    qualifying_x_positive: int
    qualifying_x_negative: int
    qualifying_y_positive: int
    qualifying_y_negative: int
    terminal: bool


@dataclass(slots=True)
class _PulseWork:
    axis: str
    polarity: int
    counts: int
    request_ns: int
    event_start_index: int
    baseline_x: float
    baseline_y: float
    deadline_ns: int
    first_emitted_ns: int | None = None
    last_emitted_ns: int | None = None
    event_end_index: int = 0
    actual_counts: int = 0


class MakcuCalibrationSession:
    """Deterministic one-shot calibration state machine around a live controller."""

    def __init__(
        self,
        controller: CalibrationController,
        binding: CalibrationRuntimeBinding,
        *,
        config: CalibrationSessionConfig | None = None,
        started_ns: int = 0,
    ) -> None:
        if not isinstance(binding, CalibrationRuntimeBinding):
            raise TypeError("binding must be CalibrationRuntimeBinding")
        self.controller = controller
        self.binding = binding
        self.config = config or CalibrationSessionConfig()
        self.started_ns = _timestamp(started_ns, "started_ns")
        self.state = CalibrationSessionState.WAIT_RELEASE
        self.message = "Keep the activation button released while calibration starts."
        self.result: CalibrationSessionResult | None = None
        self._token: object | None = None
        self._release_started_ns: int | None = None
        self._hold_deadline_ns = 0
        self._hold_settle_deadline_ns = 0
        self._target_deadline_ns = 0
        self._hold_started_ns: int | None = None
        self._arm_deadline_ns = 0
        self._held_ns: int | None = None
        self._last_now_ns: int | None = None
        self._activation_epoch: int | None = None
        self._activation_entered_ns: int | None = None
        self._activation_entry_report_sequence: int | None = None
        self._activation_entry_framed_report_sequence: int | None = None
        self._activation_entry_transition_sequence: int | None = None
        self._last_activation_snapshot: MakcuRawActivationSnapshot | None = None
        self._accepted_release_started_ns: int | None = None
        self._accepted_hold_transition_sequence: int | None = None
        self._last_measurement_ns: int | None = None
        self._target_identity: str | None = None
        self._target_bbox: tuple[float, float, float, float] | None = None
        self._observations: list[SessionObservationRecord] = []
        self._fit_measurements: list[CalibrationMeasurement] = []
        stationary_capacity = max(
            self.config.stationary_samples,
            math.ceil(
                self.config.stationary_span_seconds * self.binding.capture_fps
            )
            + 2,
        )
        self._stable: deque[CalibrationObservation] = deque(
            maxlen=stationary_capacity
        )
        self._settle_deadline_ns = 0
        self._response_ready_ns = 0
        self._axis_index = 0
        self._amplitude = self.config.initial_scout_counts
        self._pair_number = 0
        self._pair_records: list[SessionPulseRecord] = []
        self._next_polarities: list[int] = [1, -1]
        self._qualifying = {
            "x": {1: 0, -1: 0},
            "y": {1: 0, -1: 0},
        }
        # Repeatability is meaningful only between equal-size pulses. Scouts
        # may choose the final amplitude, but their qualifying responses must
        # not be pooled with a later amplitude to satisfy the two-pair gate.
        self._qualifying_amplitude: dict[str, int] = {
            "x": self.config.initial_scout_counts,
            "y": self.config.initial_scout_counts,
        }
        self._pulses: list[SessionPulseRecord] = []
        self._current: _PulseWork | None = None
        self._last_events: tuple[tuple[int, int, int], ...] = ()
        self._last_snapshot = MakcuCalibrationSnapshot()

    @property
    def terminal(self) -> bool:
        return self.state in (
            CalibrationSessionState.SUCCEEDED,
            CalibrationSessionState.ABORTED,
        )

    def status(self) -> CalibrationSessionStatus:
        axis = ("x", "y")[self._axis_index] if self._axis_index < 2 else None
        return CalibrationSessionStatus(
            state=self.state,
            message=self.message,
            axis=axis,
            pulse_counts=self._current.counts if self._current else 0,
            emitted_abs_counts=self._last_snapshot.emitted_abs_counts,
            qualifying_x_positive=self._qualifying["x"][1],
            qualifying_x_negative=self._qualifying["x"][-1],
            qualifying_y_positive=self._qualifying["y"][1],
            qualifying_y_negative=self._qualifying["y"][-1],
            terminal=self.terminal,
        )

    def update(
        self,
        now_ns: int,
        *,
        activation_known: bool,
        activation_pressed: bool,
        observation: CalibrationObservation | None,
        activation_snapshot: MakcuRawActivationSnapshot | None = None,
    ) -> CalibrationSessionStatus:
        """Advance once; callers must invoke this for every detector decision."""

        now = _timestamp(now_ns, "now_ns")
        if not isinstance(activation_known, bool) or not isinstance(
            activation_pressed, bool
        ):
            raise TypeError("activation state flags must be bool")
        if activation_snapshot is not None:
            if not isinstance(activation_snapshot, MakcuRawActivationSnapshot):
                raise TypeError(
                    "activation_snapshot must be MakcuRawActivationSnapshot or None"
                )
            if (
                activation_snapshot.known != activation_known
                or activation_snapshot.pressed != activation_pressed
            ):
                raise ValueError("activation snapshot disagrees with raw state flags")
            # update_from_controller obtains the immutable snapshot immediately
            # after its caller takes now_ns. Keep one monotonic session clock.
            now = max(now, activation_snapshot.captured_ns)
        if self.terminal:
            return self.status()
        if self._last_now_ns is not None and now <= self._last_now_ns:
            return self._abort(now, "session timestamps stopped increasing")
        self._last_now_ns = now
        self._observations.append(
            SessionObservationRecord(
                now_ns=now,
                activation_known=activation_known,
                activation_pressed=activation_pressed,
                state=self.state.value,
                observation=observation,
            )
        )
        if self._held_ns is None and len(self._observations) > (
            self.config.maximum_prehold_records
        ):
            del self._observations[
                : len(self._observations) - self.config.maximum_prehold_records
            ]
        elif len(self._observations) > self.config.maximum_evidence_records:
            return self._abort(now, "calibration evidence record budget was exceeded")

        if self.state is CalibrationSessionState.WAIT_RELEASE:
            if self._token is None:
                try:
                    self._token = self.controller.enter_calibration_mode()
                    snapshot = self.controller.calibration_snapshot()
                    self._accept_snapshot(
                        snapshot,
                        snapshot.captured_ns,
                        require_active=True,
                    )
                    if snapshot.emitted_events or snapshot.pending_counts:
                        raise CalibrationSessionError(
                            "calibration controller did not start with empty evidence"
                        )
                except Exception as exc:
                    return self._abort(
                        now,
                        f"could not enter exclusive calibration: {exc}",
                    )
                self.message = (
                    "Exclusive mode armed. Keep activation released; after "
                    "Release confirmed, press and continuously hold it."
                )
                self._arm_deadline_ns = now + round(
                    self.config.arm_timeout_seconds * 1_000_000_000
                )
                # The sample supplied to this call was gathered before
                # exclusive entry. Do not let it satisfy the controller's
                # deliberate post-entry fresh-release latch.
                self._release_started_ns = None
                return self.status()
        if activation_snapshot is not None:
            activation_error = self._accept_activation_snapshot(
                activation_snapshot,
                now,
            )
            if activation_error is not None:
                return self._abort(now, activation_error)

        if self.state is CalibrationSessionState.WAIT_RELEASE:
            if now > self._arm_deadline_ns:
                return self._abort(now, "fresh release/hold arming timed out")
            if activation_snapshot is not None:
                dwell_ns = round(
                    self.config.release_dwell_seconds * 1_000_000_000
                )
                if not activation_snapshot.known:
                    self.message = (
                        "Waiting for a fresh MAKCU activation report; movement is "
                        "disarmed."
                    )
                    return self.status()
                if (
                    activation_snapshot.framed_report_sequence
                    <= activation_snapshot.calibration_entry_framed_report_sequence
                ):
                    self.message = (
                        "Waiting for a post-entry framed MAKCU button report; "
                        "movement is disarmed."
                    )
                    return self.status()
                if activation_snapshot.pressed:
                    release_ns = activation_snapshot.completed_release_started_ns
                    press_ns = activation_snapshot.completed_press_ns
                    if release_ns is not None and press_ns is not None:
                        if (
                            press_ns - release_ns >= dwell_ns
                            and activation_snapshot.completed_press_transition_sequence
                            == activation_snapshot.transition_sequence
                        ):
                            self._accepted_release_started_ns = release_ns
                            self._accepted_hold_transition_sequence = (
                                activation_snapshot.completed_press_transition_sequence
                            )
                            self.state = CalibrationSessionState.WAIT_HOLD
                            self.message = (
                                "Release confirmed. Press and continuously hold "
                                "activation."
                            )
                            self._hold_deadline_ns = now + 15_000_000_000
                            return self._begin_post_hold_settle(now)
                        self.message = (
                            "Release was too brief. Fully release activation and "
                            "wait for Release confirmed before pressing again."
                        )
                    else:
                        self.message = (
                            "Activation is pressed. Fully release it and keep it "
                            "released until Release confirmed."
                        )
                    return self.status()
                release_ns = activation_snapshot.release_started_ns
                if release_ns is None:
                    self.message = (
                        "Fully release activation; waiting for a fresh release "
                        "report."
                    )
                    return self.status()
                if activation_snapshot.captured_ns - release_ns >= dwell_ns:
                    self._accepted_release_started_ns = release_ns
                    self.state = CalibrationSessionState.WAIT_HOLD
                    self.message = self._released_target_readiness_message(
                        observation,
                        now,
                    )
                    self._hold_deadline_ns = now + 15_000_000_000
                else:
                    self.message = (
                        "Keep activation fully released until Release confirmed."
                    )
                return self.status()
            if activation_known and not activation_pressed:
                if self._release_started_ns is None:
                    self._release_started_ns = now
                dwell_ns = round(self.config.release_dwell_seconds * 1_000_000_000)
                if now - self._release_started_ns >= dwell_ns:
                    self.state = CalibrationSessionState.WAIT_HOLD
                    self.message = self._released_target_readiness_message(
                        observation,
                        now,
                    )
                    self._hold_deadline_ns = now + 15_000_000_000
            else:
                self._release_started_ns = None
            return self.status()

        if self.state is CalibrationSessionState.WAIT_HOLD:
            if self._hold_started_ns is None and now > self._hold_deadline_ns:
                return self._abort(now, "fresh post-entry activation timed out")
            if self._hold_started_ns is not None:
                if not activation_known or not activation_pressed:
                    return self._abort(
                        now,
                        "physical activation was released or became unknown during "
                        "post-hold settling or safe-target acquisition",
                    )
                if (
                    activation_snapshot is not None
                    and activation_snapshot.transition_sequence
                    != self._accepted_hold_transition_sequence
                ):
                    return self._abort(
                        now,
                        "physical activation changed during post-hold settling or "
                        "safe-target acquisition",
                    )
                if now < self._hold_settle_deadline_ns:
                    self.message = (
                        "Hold detected. Keep holding while the selected aim mode "
                        "settles. No movement is authorized yet."
                    )
                    return self.status()
                if now > self._target_deadline_ns:
                    reason = self._observation_rejection(observation, now)
                    detail = reason or "no continuously safe exact target was available"
                    return self._abort(
                        now,
                        f"safe target was not ready before the hold deadline: {detail}",
                    )
                return self._begin_held_calibration(now, observation)
            if not activation_known:
                if activation_snapshot is not None:
                    return self._abort(
                        now,
                        "physical activation became unknown after release confirmation",
                    )
                return self.status()
            if not activation_pressed:
                self.message = self._released_target_readiness_message(
                    observation,
                    now,
                )
                return self.status()
            if activation_snapshot is not None:
                release_ns = activation_snapshot.completed_release_started_ns
                press_ns = activation_snapshot.completed_press_ns
                dwell_ns = round(
                    self.config.release_dwell_seconds * 1_000_000_000
                )
                if (
                    release_ns != self._accepted_release_started_ns
                    or press_ns is None
                    or release_ns is None
                    or press_ns - release_ns < dwell_ns
                    or activation_snapshot.completed_press_transition_sequence
                    != activation_snapshot.transition_sequence
                ):
                    return self._abort(
                        now,
                        "fresh activation did not follow the confirmed release dwell",
                    )
                self._accepted_hold_transition_sequence = (
                    activation_snapshot.completed_press_transition_sequence
                )
            return self._begin_post_hold_settle(now)

        if not activation_known or not activation_pressed:
            return self._abort(now, "physical activation was released or became unknown")
        if self._held_ns is None or now - self._held_ns > round(
            self.config.held_session_timeout_seconds * 1_000_000_000
        ):
            return self._abort(now, "held calibration session exceeded its deadline")
        if observation is None and self.state in (
            CalibrationSessionState.PULSE,
            CalibrationSessionState.RESPONSE_SETTLE,
        ):
            if self._current is not None and self.state is CalibrationSessionState.PULSE:
                try:
                    snapshot = self.controller.calibration_snapshot()
                    self._accept_snapshot(
                        snapshot,
                        snapshot.captured_ns,
                        require_active=True,
                    )
                except Exception as exc:
                    return self._abort(now, f"invalid calibration snapshot: {exc}")
                if snapshot.abort_reason:
                    return self._abort(now, f"controller aborted: {snapshot.abort_reason}")
                if snapshot.pending_counts:
                    return self.status()
                return self._advance_pulse(now, snapshot)
            if now > self._settle_deadline_ns:
                return self._abort(now, "stationary response did not settle before timeout")
            return self.status()
        rejection = self._observation_rejection(
            observation,
            now,
            lenient_roi=self.state
            in (
                CalibrationSessionState.BASELINE_SETTLE,
                CalibrationSessionState.PULSE,
                CalibrationSessionState.RESPONSE_SETTLE,
            ),
        )
        if rejection is not None:
            return self._abort(now, rejection)
        assert observation is not None and self._token is not None
        if self._target_identity != observation.target_identity:
            return self._abort(now, "the unique target identity changed")
        if self.state not in (
            CalibrationSessionState.BASELINE_SETTLE,
            CalibrationSessionState.PULSE,
            CalibrationSessionState.RESPONSE_SETTLE,
        ):
            if self._target_bbox is None or not _target_bbox_is_continuous(
                self._target_bbox,
                observation.normalized_bbox,
            ):
                return self._abort(
                    now,
                    "the unique target bounding box changed discontinuously",
                )
        if (
            self._last_measurement_ns is not None
            and observation.measurement_ns <= self._last_measurement_ns
        ):
            return self._abort(now, "detector timestamps stopped increasing")
        try:
            lease_options = {}
            if self._accepted_hold_transition_sequence is not None:
                lease_options["activation_transition_sequence"] = (
                    self._accepted_hold_transition_sequence
                )
            self.controller.publish_calibration_lease(
                True,
                observation.measurement_ns,
                self._token,
                **lease_options,
            )
        except Exception as exc:
            return self._abort(now, f"calibration lease failed: {exc}")
        self._target_bbox = observation.normalized_bbox
        self._record_fit_observation(observation)

        try:
            snapshot = self.controller.calibration_snapshot()
            self._accept_snapshot(
                snapshot,
                snapshot.captured_ns,
                require_active=True,
            )
        except Exception as exc:
            return self._abort(now, f"invalid calibration snapshot: {exc}")
        if snapshot.abort_reason:
            return self._abort(now, f"controller aborted: {snapshot.abort_reason}")

        if self._current is not None:
            delta_x = observation.error_x - self._current.baseline_x
            delta_y = observation.error_y - self._current.baseline_y
            if max(abs(delta_x), abs(delta_y)) > self.config.abort_excursion_pixels:
                return self._abort(
                    now,
                    "observed primary/cross-axis pulse excursion exceeded 100px",
                )

        if self.state is CalibrationSessionState.PULSE:
            return self._advance_pulse(now, snapshot)
        if self.state in (
            CalibrationSessionState.BASELINE_SETTLE,
            CalibrationSessionState.RESPONSE_SETTLE,
        ):
            return self._advance_settle(now, observation)
        return self._abort(now, "calibration entered an invalid state")

    def update_from_controller(
        self,
        now_ns: int,
        *,
        observation: CalibrationObservation | None,
    ) -> CalibrationSessionStatus:
        """Advance using one immutable controller-owned raw activation snapshot."""

        try:
            activation = self.controller.raw_activation_snapshot
        except Exception as exc:
            return self._abort(
                _timestamp(now_ns, "now_ns"),
                f"could not read raw activation snapshot: {exc}",
            )
        return self.update(
            now_ns,
            activation_known=activation.known,
            activation_pressed=activation.pressed,
            observation=observation,
            activation_snapshot=activation,
        )

    def _accept_activation_snapshot(
        self,
        snapshot: MakcuRawActivationSnapshot,
        now_ns: int,
    ) -> str | None:
        """Bind one monotonic raw-button evidence stream to this session."""

        if not snapshot.active:
            return "raw activation snapshot reports inactive calibration"
        if snapshot.captured_ns > now_ns:
            return "raw activation snapshot is in the future"
        if snapshot.transition_sequence > snapshot.report_sequence:
            return "raw activation transition sequence exceeds reports"
        if snapshot.framed_report_sequence > snapshot.report_sequence:
            return "raw activation framed reports exceed all reports"
        if self._activation_epoch is None:
            if snapshot.calibration_epoch <= 0:
                return "raw activation snapshot has no calibration epoch"
            self._activation_epoch = snapshot.calibration_epoch
            self._activation_entered_ns = snapshot.calibration_entered_ns
            self._activation_entry_report_sequence = (
                snapshot.calibration_entry_report_sequence
            )
            self._activation_entry_framed_report_sequence = (
                snapshot.calibration_entry_framed_report_sequence
            )
            self._activation_entry_transition_sequence = (
                snapshot.calibration_entry_transition_sequence
            )
        elif snapshot.calibration_epoch != self._activation_epoch:
            return "raw activation calibration epoch changed"
        elif snapshot.calibration_entered_ns != self._activation_entered_ns:
            return "raw activation calibration entry changed"
        elif (
            snapshot.calibration_entry_report_sequence
            != self._activation_entry_report_sequence
            or snapshot.calibration_entry_framed_report_sequence
            != self._activation_entry_framed_report_sequence
            or snapshot.calibration_entry_transition_sequence
            != self._activation_entry_transition_sequence
        ):
            return "raw activation calibration entry baseline changed"

        previous = self._last_activation_snapshot
        if previous is not None:
            if snapshot.captured_ns < previous.captured_ns:
                return "raw activation snapshot clock moved backwards"
            if snapshot.transition_sequence < previous.transition_sequence:
                return "raw activation transition sequence moved backwards"
            if snapshot.report_sequence < previous.report_sequence:
                return "raw activation report sequence moved backwards"
            if snapshot.framed_report_sequence < previous.framed_report_sequence:
                return "raw activation framed-report sequence moved backwards"
            report_delta = snapshot.report_sequence - previous.report_sequence
            framed_report_delta = (
                snapshot.framed_report_sequence - previous.framed_report_sequence
            )
            transition_delta = (
                snapshot.transition_sequence - previous.transition_sequence
            )
            if transition_delta > report_delta:
                return "raw activation transitions exceeded new reports"
            if framed_report_delta > report_delta:
                return "raw activation framed reports exceeded new reports"
            if snapshot.report_sequence == previous.report_sequence and (
                snapshot.known != previous.known
                or snapshot.pressed != previous.pressed
                or snapshot.last_report_framed != previous.last_report_framed
                or snapshot.continuous_state_since_ns
                != previous.continuous_state_since_ns
                or snapshot.release_started_ns != previous.release_started_ns
                or snapshot.release_started_report_sequence
                != previous.release_started_report_sequence
                or snapshot.completed_release_started_ns
                != previous.completed_release_started_ns
                or snapshot.completed_release_report_sequence
                != previous.completed_release_report_sequence
                or snapshot.completed_press_ns != previous.completed_press_ns
                or snapshot.completed_press_report_sequence
                != previous.completed_press_report_sequence
                or snapshot.completed_press_transition_sequence
                != previous.completed_press_transition_sequence
            ):
                return "raw activation state changed without a report"
            if snapshot.framed_report_sequence == previous.framed_report_sequence and (
                snapshot.post_entry_press_seen
                != previous.post_entry_press_seen
                or snapshot.release_started_ns != previous.release_started_ns
                or snapshot.release_started_report_sequence
                != previous.release_started_report_sequence
                or snapshot.completed_release_started_ns
                != previous.completed_release_started_ns
                or snapshot.completed_release_report_sequence
                != previous.completed_release_report_sequence
                or snapshot.completed_press_ns != previous.completed_press_ns
                or snapshot.completed_press_report_sequence
                != previous.completed_press_report_sequence
                or snapshot.completed_press_transition_sequence
                != previous.completed_press_transition_sequence
            ):
                return "calibration proof changed without a framed report"
            if (
                snapshot.transition_sequence == previous.transition_sequence
                and snapshot.pressed != previous.pressed
            ):
                return "raw activation level changed without a transition"
        self._last_activation_snapshot = snapshot
        return None

    def _begin_held_calibration(
        self,
        now_ns: int,
        observation: CalibrationObservation | None,
    ) -> CalibrationSessionStatus:
        """Authorize the first hold only after its release proof is complete."""

        reason = self._observation_rejection(
            observation,
            now_ns,
            lenient_roi=True,
        )
        if reason is not None:
            self.message = (
                "Keep holding activation; waiting for one safe exact target "
                f"({reason}). No movement is authorized yet."
            )
            return self.status()
        assert observation is not None
        try:
            assert self._token is not None
            lease_options = {}
            if self._accepted_hold_transition_sequence is not None:
                lease_options["activation_transition_sequence"] = (
                    self._accepted_hold_transition_sequence
                )
            self.controller.publish_calibration_lease(
                True,
                observation.measurement_ns,
                self._token,
                **lease_options,
            )
        except Exception as exc:
            return self._abort(
                now_ns,
                f"could not authorize calibration hold: {exc}",
            )
        self._held_ns = now_ns
        self._target_identity = observation.target_identity
        self._target_bbox = observation.normalized_bbox
        self.state = CalibrationSessionState.BASELINE_SETTLE
        self.message = "Hold still while the stationary baseline settles."
        self._settle_deadline_ns = now_ns + round(
            self.config.initial_settle_timeout_seconds * 1_000_000_000
        )
        self._record_fit_observation(observation)
        self._stable.append(observation)
        return self.status()

    def _begin_post_hold_settle(self, now_ns: int) -> CalibrationSessionStatus:
        """Latch the fresh hold but ignore pre-transition video without a lease."""

        self._hold_started_ns = now_ns
        self._hold_settle_deadline_ns = now_ns + round(
            self.config.post_hold_settle_seconds * 1_000_000_000
        )
        self._target_deadline_ns = self._hold_settle_deadline_ns + round(
            self.config.target_acquire_timeout_seconds * 1_000_000_000
        )
        self.message = (
            "Hold detected. Keep holding while the selected aim mode settles. "
            "No movement is authorized yet."
        )
        return self.status()

    def _released_target_readiness_message(
        self,
        observation: CalibrationObservation | None,
        now_ns: int,
    ) -> str:
        """Describe target readiness without authorizing a calibration lease."""

        reason = self._observation_rejection(observation, now_ns)
        if reason is None:
            return (
                "Release confirmed and target ready. Press and continuously hold "
                "activation."
            )
        return (
            "Release confirmed. Keep activation released; waiting for a safe target "
            f"({reason})."
        )

    def abort(self, reason: str, *, now_ns: int) -> CalibrationSessionStatus:
        """Idempotently stop an unfinished session without corrective movement."""

        _bounded_text(reason, "abort reason")
        if self.terminal:
            return self.status()
        return self._abort(_timestamp(now_ns, "now_ns"), reason)

    def _record_fit_observation(self, observation: CalibrationObservation) -> None:
        self._last_measurement_ns = observation.measurement_ns
        self._fit_measurements.append(
            CalibrationMeasurement(
                observation.measurement_ns,
                observation.error_x,
                observation.error_y,
                True,
            )
        )

    def _observation_rejection(
        self,
        observation: CalibrationObservation | None,
        now_ns: int,
        *,
        lenient_roi: bool = False,
    ) -> str | None:
        if observation is None:
            return "no exact target observation was available"
        age_ns = now_ns - observation.measurement_ns
        if age_ns < 0:
            return "detector observation timestamp is in the future"
        if age_ns > round(
            self.config.maximum_observation_age_seconds * 1_000_000_000
        ):
            return "detector observation is stale"
        if observation.is_prediction:
            return "predicted tracker output cannot authorize calibration"
        if not observation.exact_label:
            return "target label is not an exact match"
        if observation.unique_candidates != 1:
            return "calibration requires exactly one safe target candidate"
        if not observation.self_safe:
            return "self-safety filtering did not authorize the target"
        if observation.confidence < self.config.minimum_confidence:
            return "target confidence is below the calibration threshold"
        if observation.target_identity == "":
            return "target identity is empty"
        if not lenient_roi:
            if not target_within_safe_roi(
                observation.normalized_bbox,
                self.config.safe_roi_margin_ratio,
            ):
                return "the complete target box is outside the central safe ROI"
            if max(abs(observation.error_x), abs(observation.error_y)) > (
                self.config.maximum_reference_error
            ):
                return "target aim point is outside the central error envelope"
        return None

    def _accept_snapshot(
        self,
        snapshot: MakcuCalibrationSnapshot,
        captured_ns: int,
        *,
        require_active: bool,
    ) -> None:
        if not isinstance(snapshot, MakcuCalibrationSnapshot):
            raise CalibrationSessionError("snapshot has the wrong type")
        _timestamp(captured_ns, "snapshot captured_ns")
        if snapshot.captured_ns != captured_ns:
            raise CalibrationSessionError("snapshot capture ceiling is inconsistent")
        events = snapshot.emitted_events
        if events[: len(self._last_events)] != self._last_events:
            raise CalibrationSessionError("emitted event history changed or shrank")
        previous_timestamp: int | None = None
        emitted_x = 0
        emitted_y = 0
        emitted_abs = 0
        for event in events:
            if not isinstance(event, tuple) or len(event) != 3:
                raise CalibrationSessionError("emitted event is malformed")
            timestamp_ns, delta_x, delta_y = event
            _timestamp(timestamp_ns, "emitted event timestamp")
            if previous_timestamp is not None and timestamp_ns <= previous_timestamp:
                raise CalibrationSessionError("emitted event timestamps are not ordered")
            if timestamp_ns > captured_ns:
                raise CalibrationSessionError("emitted event timestamp is in the future")
            previous_timestamp = timestamp_ns
            if (
                isinstance(delta_x, bool)
                or not isinstance(delta_x, int)
                or isinstance(delta_y, bool)
                or not isinstance(delta_y, int)
                or bool(delta_x) == bool(delta_y)
                or abs(delta_x) + abs(delta_y) > CALIBRATION_MAX_STEP_COUNTS
            ):
                raise CalibrationSessionError("emitted event is not single-axis movement")
            emitted_x += delta_x
            emitted_y += delta_y
            emitted_abs += abs(delta_x) + abs(delta_y)
        if snapshot.movement_commands != len(events):
            raise CalibrationSessionError("movement command aggregate is inconsistent")
        if snapshot.emitted_x != emitted_x or snapshot.emitted_y != emitted_y:
            raise CalibrationSessionError("signed emitted-count aggregate is inconsistent")
        if snapshot.emitted_abs_counts != emitted_abs:
            raise CalibrationSessionError("absolute emitted-count aggregate is inconsistent")
        if emitted_abs > CALIBRATION_MAX_SESSION_ABS_COUNTS:
            raise CalibrationSessionError("emitted counts exceed the session envelope")
        expected_first = events[0][0] if events else None
        expected_last = events[-1][0] if events else None
        if (
            snapshot.first_emitted_ns != expected_first
            or snapshot.last_emitted_ns != expected_last
        ):
            raise CalibrationSessionError("first/last emitted timestamps are inconsistent")
        if require_active and not snapshot.active:
            raise CalibrationSessionError("controller left exclusive calibration mode")
        if self._current is None and len(events) != len(self._last_events):
            raise CalibrationSessionError("movement appeared without an outstanding pulse")
        self._last_events = tuple(events)
        self._last_snapshot = snapshot

    def _advance_pulse(
        self,
        now_ns: int,
        snapshot: MakcuCalibrationSnapshot,
    ) -> CalibrationSessionStatus:
        assert self._current is not None
        current = self._current
        if now_ns > current.deadline_ns:
            return self._abort(now_ns, "calibration pulse did not finish before timeout")
        if snapshot.pending_axis not in (None, current.axis):
            return self._abort(now_ns, "controller reported the wrong pending pulse axis")
        if snapshot.pending_counts:
            if (
                isinstance(snapshot.pending_counts, bool)
                or not isinstance(snapshot.pending_counts, int)
                or snapshot.pending_counts * current.polarity <= 0
                or abs(snapshot.pending_counts) > current.counts
            ):
                return self._abort(
                    now_ns,
                    "controller reported invalid remaining pulse counts",
                )
            if not math.isclose(
                snapshot.pending_rate_counts_per_second,
                self.config.pulse_rate_counts_per_second,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return self._abort(now_ns, "controller changed the pending pulse rate")
            return self.status()
        new_events = snapshot.emitted_events[current.event_start_index :]
        if not new_events:
            return self._abort(now_ns, "controller completed a pulse without emitted events")
        actual_primary = sum(
            delta_x if current.axis == "x" else delta_y
            for _timestamp_ns, delta_x, delta_y in new_events
        )
        actual_cross = sum(
            delta_y if current.axis == "x" else delta_x
            for _timestamp_ns, delta_x, delta_y in new_events
        )
        if actual_cross or actual_primary != current.polarity * current.counts:
            return self._abort(now_ns, "actual emitted pulse differs from its request")
        if any(
            (delta_x if current.axis == "x" else delta_y) * current.polarity <= 0
            for _timestamp_ns, delta_x, delta_y in new_events
        ):
            return self._abort(now_ns, "actual pulse reversed direction while in flight")
        current.first_emitted_ns = new_events[0][0]
        current.last_emitted_ns = new_events[-1][0]
        current.event_end_index = len(snapshot.emitted_events)
        current.actual_counts = actual_primary
        self.state = CalibrationSessionState.RESPONSE_SETTLE
        self.message = "Waiting for the bounded pulse response to settle."
        self._stable.clear()
        self._response_ready_ns = current.last_emitted_ns + round(
            self.config.response_delay_seconds * 1_000_000_000
        )
        self._settle_deadline_ns = current.last_emitted_ns + round(
            self.config.response_settle_timeout_seconds * 1_000_000_000
        )
        return self.status()

    def _advance_settle(
        self,
        now_ns: int,
        observation: CalibrationObservation,
    ) -> CalibrationSessionStatus:
        if now_ns > self._settle_deadline_ns:
            return self._abort(now_ns, "stationary response did not settle before timeout")
        if (
            self.state is CalibrationSessionState.RESPONSE_SETTLE
            and now_ns < self._response_ready_ns
        ):
            return self.status()
        self._stable.append(observation)
        if not self._stationary_window_ready():
            return self.status()
        baseline_x = statistics.median(value.error_x for value in self._stable)
        baseline_y = statistics.median(value.error_y for value in self._stable)
        if self.state is CalibrationSessionState.BASELINE_SETTLE:
            return self._request_next_pulse(now_ns, baseline_x, baseline_y)

        assert self._current is not None
        current = self._current
        settled_primary = baseline_x if current.axis == "x" else baseline_y
        initial_primary = (
            current.baseline_x if current.axis == "x" else current.baseline_y
        )
        settled_cross = baseline_y if current.axis == "x" else baseline_x
        initial_cross = (
            current.baseline_y if current.axis == "x" else current.baseline_x
        )
        response_delta = settled_primary - initial_primary
        cross_delta = settled_cross - initial_cross
        signed_response = -current.polarity * response_delta
        excursion = abs(response_delta)
        if max(excursion, abs(cross_delta)) > self.config.abort_excursion_pixels:
            return self._abort(
                now_ns,
                "settled primary/cross-axis pulse excursion exceeded 100px",
            )
        qualifying = (
            signed_response > 0.0
            and excursion >= self.config.qualifying_minimum_excursion_pixels
        )
        record = SessionPulseRecord(
            axis=current.axis,
            polarity=current.polarity,
            requested_counts=current.counts,
            requested_rate=self.config.pulse_rate_counts_per_second,
            request_ns=current.request_ns,
            event_start_index=current.event_start_index,
            event_end_index=current.event_end_index,
            first_emitted_ns=current.first_emitted_ns,
            last_emitted_ns=current.last_emitted_ns,
            actual_counts=current.actual_counts,
            baseline_x=current.baseline_x,
            baseline_y=current.baseline_y,
            settled_x=baseline_x,
            settled_y=baseline_y,
            signed_response_pixels=signed_response,
            cross_response_pixels=abs(cross_delta),
            qualifying=qualifying,
            complete=True,
        )
        self._pulses.append(record)
        self._pair_records.append(record)
        if qualifying:
            if self._qualifying_amplitude[current.axis] != current.counts:
                self._qualifying[current.axis] = {1: 0, -1: 0}
                self._qualifying_amplitude[current.axis] = current.counts
            self._qualifying[current.axis][current.polarity] += 1
        self._current = None
        if self._next_polarities:
            return self._request_next_pulse(now_ns, baseline_x, baseline_y)
        return self._complete_pair(now_ns, baseline_x, baseline_y)

    def _stationary_window_ready(self) -> bool:
        if len(self._stable) < self.config.stationary_samples:
            return False
        first = self._stable[0]
        last = self._stable[-1]
        elapsed = (last.measurement_ns - first.measurement_ns) / 1_000_000_000
        if elapsed < self.config.stationary_span_seconds:
            return False
        x_values = [value.error_x for value in self._stable]
        y_values = [value.error_y for value in self._stable]
        if max(x_values) - min(x_values) > self.config.stationary_range_pixels:
            return False
        if max(y_values) - min(y_values) > self.config.stationary_range_pixels:
            return False
        if abs(last.error_x - first.error_x) / elapsed > (
            self.config.stationary_speed_pixels_per_second
        ):
            return False
        if abs(last.error_y - first.error_y) / elapsed > (
            self.config.stationary_speed_pixels_per_second
        ):
            return False
        return True

    def _request_next_pulse(
        self,
        now_ns: int,
        baseline_x: float,
        baseline_y: float,
    ) -> CalibrationSessionStatus:
        if self._axis_index >= 2:
            return self._finish_fit(now_ns)
        if not self._next_polarities:
            return self._abort(now_ns, "pulse planner exhausted an empty pair")
        axis = ("x", "y")[self._axis_index]
        snapshot = self._last_snapshot
        if (
            len(self._next_polarities) == 2
            and snapshot.emitted_abs_counts + 2 * self._amplitude
            > CALIBRATION_MAX_SESSION_ABS_COUNTS
        ):
            remaining_pair_budget = (
                CALIBRATION_MAX_SESSION_ABS_COUNTS - snapshot.emitted_abs_counts
            ) // 2
            if remaining_pair_budget <= 0:
                return self._abort(
                    now_ns,
                    "bounded calibration budget cannot fit another complete symmetric "
                    f"{axis.upper()} pair at {self._amplitude} counts",
                )
            if remaining_pair_budget < self._amplitude:
                self._amplitude = remaining_pair_budget
                self._qualifying[axis] = {1: 0, -1: 0}
                self._qualifying_amplitude[axis] = remaining_pair_budget
        polarity = self._next_polarities.pop(0)
        if snapshot.emitted_abs_counts + self._amplitude > (
            CALIBRATION_MAX_SESSION_ABS_COUNTS
        ):
            return self._abort(now_ns, "adaptive plan exhausted the 2400-count budget")
        assert self._token is not None
        try:
            self.controller.request_calibration_pulse(
                axis,
                polarity * self._amplitude,
                self.config.pulse_rate_counts_per_second,
                self._token,
            )
        except Exception as exc:
            return self._abort(now_ns, f"could not queue bounded calibration pulse: {exc}")
        duration_seconds = self._amplitude / self.config.pulse_rate_counts_per_second
        self._current = _PulseWork(
            axis=axis,
            polarity=polarity,
            counts=self._amplitude,
            request_ns=now_ns,
            event_start_index=len(snapshot.emitted_events),
            baseline_x=baseline_x,
            baseline_y=baseline_y,
            deadline_ns=now_ns
            + round(
                (duration_seconds + self.config.pulse_timeout_slack_seconds)
                * 1_000_000_000
            ),
        )
        self.state = CalibrationSessionState.PULSE
        self.message = (
            f"Running bounded {axis.upper()} pulse "
            f"{polarity * self._amplitude:+d} counts."
        )
        self._stable.clear()
        return self.status()

    def _complete_pair(
        self,
        now_ns: int,
        baseline_x: float,
        baseline_y: float,
    ) -> CalibrationSessionStatus:
        axis = ("x", "y")[self._axis_index]
        if len(self._pair_records) != 2:
            return self._abort(now_ns, "adaptive pulse pair is incomplete")
        responses = [
            max(float(record.signed_response_pixels or 0.0), 0.0)
            for record in self._pair_records
        ]
        mean_response = statistics.fmean(responses)
        enough = bool(
            self._qualifying_amplitude[axis] == self._amplitude
            and min(self._qualifying[axis].values()) >= 2
        )
        if enough:
            self._axis_index += 1
            self._amplitude = self.config.initial_scout_counts
            self._pair_number = 0
            self._pair_records.clear()
            self._next_polarities = [1, -1]
            if self._axis_index >= 2:
                return self._finish_fit(now_ns)
            next_axis = ("x", "y")[self._axis_index]
            self._qualifying[next_axis] = {1: 0, -1: 0}
            self._qualifying_amplitude[next_axis] = self._amplitude
            return self._request_next_pulse(now_ns, baseline_x, baseline_y)

        current_amplitude = self._amplitude
        if current_amplitude >= CALIBRATION_MAX_EXCURSION_COUNTS:
            # Stay at the bounded final amplitude until two qualifying pairs
            # agree or the hard session budget can no longer fit a full
            # net-zero pair. `_request_next_pulse` performs that budget gate
            # before either half is queued.
            next_amplitude = CALIBRATION_MAX_EXCURSION_COUNTS
        elif mean_response <= 1.0:
            next_amplitude = min(
                CALIBRATION_MAX_EXCURSION_COUNTS,
                current_amplitude * 2,
            )
        else:
            scaled = round(
                current_amplitude
                * self.config.target_excursion_pixels
                / mean_response
            )
            if mean_response < self.config.preferred_minimum_excursion_pixels:
                scaled = max(scaled, current_amplitude + 1)
            elif mean_response > self.config.preferred_maximum_excursion_pixels:
                scaled = min(scaled, current_amplitude - 1)
            next_amplitude = min(
                CALIBRATION_MAX_EXCURSION_COUNTS,
                max(1, scaled),
            )

        self._amplitude = next_amplitude
        if next_amplitude != current_amplitude:
            self._qualifying[axis] = {1: 0, -1: 0}
            self._qualifying_amplitude[axis] = next_amplitude
        self._pair_number += 1
        order = [-1, 1] if self._pair_number % 2 else [1, -1]
        self._next_polarities = order
        self._pair_records.clear()
        return self._request_next_pulse(now_ns, baseline_x, baseline_y)

    def _finish_fit(self, now_ns: int) -> CalibrationSessionStatus:
        if self._last_snapshot.pending_counts:
            return self._abort(now_ns, "fit was attempted while a pulse remained pending")
        commands = tuple(
            EmittedCount(timestamp_ns, delta_x, delta_y)
            for timestamp_ns, delta_x, delta_y in self._last_snapshot.emitted_events
        )
        pulses = tuple(
            CalibrationPulse(
                record.axis,
                record.polarity,
                int(record.first_emitted_ns),
                int(record.last_emitted_ns),
            )
            for record in self._pulses
            if record.complete
            and record.first_emitted_ns is not None
            and record.last_emitted_ns is not None
        )
        try:
            fit = fit_makcu_calibration(self._fit_measurements, commands, pulses)
        except (CalibrationDataError, CalibrationQualityError) as exc:
            return self._abort(now_ns, f"calibration fit rejected the evidence: {exc}")
        return self._terminate(now_ns, outcome="success", reason="calibration fit passed", fit=fit)

    def _abort(self, now_ns: int, reason: str) -> CalibrationSessionStatus:
        if self.terminal:
            return self.status()
        return self._terminate(now_ns, outcome="aborted", reason=reason, fit=None)

    def _terminate(
        self,
        now_ns: int,
        *,
        outcome: Literal["success", "aborted"],
        reason: str,
        fit: MakcuCalibrationFit | None,
    ) -> CalibrationSessionStatus:
        reason = _sanitized_reason(reason)
        cleanup_errors: list[str] = []
        evidence_complete = True
        if self._token is not None:
            if outcome == "aborted":
                try:
                    measurement_ns = self._last_measurement_ns or now_ns
                    self.controller.publish_calibration_lease(
                        False,
                        measurement_ns,
                        self._token,
                    )
                except Exception as exc:
                    cleanup_errors.append(
                        _sanitized_reason(f"lease revoke: {exc}")
                    )
            try:
                snapshot = self.controller.calibration_snapshot()
                self._accept_snapshot(
                    snapshot,
                    snapshot.captured_ns,
                    require_active=False,
                )
            except Exception as exc:
                evidence_complete = False
                cleanup_errors.append(_sanitized_reason(f"final snapshot: {exc}"))
            if outcome == "aborted" and self._current is not None and not any(
                record.request_ns == self._current.request_ns
                for record in self._pulses
            ):
                current = self._current
                events = self._last_events[current.event_start_index :]
                actual = sum(
                    delta_x if current.axis == "x" else delta_y
                    for _timestamp_ns, delta_x, delta_y in events
                )
                self._pulses.append(
                    SessionPulseRecord(
                        axis=current.axis,
                        polarity=current.polarity,
                        requested_counts=current.counts,
                        requested_rate=self.config.pulse_rate_counts_per_second,
                        request_ns=current.request_ns,
                        event_start_index=current.event_start_index,
                        event_end_index=len(self._last_events),
                        first_emitted_ns=events[0][0] if events else None,
                        last_emitted_ns=events[-1][0] if events else None,
                        actual_counts=actual,
                        baseline_x=current.baseline_x,
                        baseline_y=current.baseline_y,
                        settled_x=None,
                        settled_y=None,
                        signed_response_pixels=None,
                        cross_response_pixels=None,
                        qualifying=False,
                        complete=False,
                    )
                )
            try:
                self.controller.exit_calibration_mode(self._token)
            except Exception as exc:
                cleanup_errors.append(_sanitized_reason(f"exclusive exit: {exc}"))
            self._token = None
        if outcome == "success" and (cleanup_errors or not evidence_complete):
            outcome = "aborted"
            fit = None
            reason = "successful fit could not exit exclusive calibration cleanly"
        cleanup_error = "; ".join(cleanup_errors) or None
        artifact = _make_session_evidence(
            binding=self.binding,
            outcome=outcome,
            reason=reason,
            cleanup_error=cleanup_error,
            evidence_complete=evidence_complete,
            started_ns=self.started_ns,
            held_ns=self._held_ns,
            terminal_ns=now_ns,
            observations=tuple(self._observations),
            pulses=tuple(self._pulses),
            emitted_events=self._last_events,
            fit=fit,
        )
        self.result = CalibrationSessionResult(
            outcome=outcome,
            reason=reason,
            fit=fit,
            evidence=artifact,
        )
        self.state = (
            CalibrationSessionState.SUCCEEDED
            if outcome == "success"
            else CalibrationSessionState.ABORTED
        )
        self.message = reason
        return self.status()


def _binding_dict(binding: CalibrationRuntimeBinding) -> dict[str, object]:
    return {
        "activation_button": binding.activation_button,
        "active_device": binding.active_device,
        "active_provider": binding.active_provider,
        "aim_label": binding.aim_label,
        "aim_mode": binding.aim_mode,
        "backend": binding.backend,
        "runtime_version": binding.runtime_version,
        "build_identity": binding.build_identity,
        "capture_fps": binding.capture_fps,
        "capture_backend": binding.capture_backend,
        "capture_buffer_size": binding.capture_buffer_size,
        "capture_height": binding.capture_height,
        "capture_index": binding.capture_index,
        "capture_kind": binding.capture_kind,
        "capture_width": binding.capture_width,
        "context_name": binding.context_name,
        "detail_pass_enabled": binding.detail_pass_enabled,
        "head_ratio": binding.head_ratio,
        "inference_height": binding.inference_height,
        "inference_width": binding.inference_width,
        "invert_x": binding.invert_x,
        "invert_y": binding.invert_y,
        "labels_sha256": binding.labels_sha256,
        "makcu_identity_token": binding.makcu_identity_token,
        "model_sha256": binding.model_sha256,
        "pixel_format": binding.pixel_format,
        "physical_device_token": binding.physical_device_token,
        "provider_options_sha256": binding.provider_options_sha256,
        "requested_provider": binding.requested_provider,
        "rotation_degrees": binding.rotation_degrees,
        "source_commit": binding.source_commit,
    }


def _axis_fit_dict(value: AxisCalibrationFit) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
    }


def _fit_dict(fit: MakcuCalibrationFit | None) -> dict[str, object] | None:
    if fit is None:
        return None
    return {
        "delay_seconds": fit.delay_seconds,
        "detector_period_seconds": fit.detector_period_seconds,
        "evidence_sha256": fit.evidence_sha256,
        "observation_duty": fit.observation_duty,
        "x": _axis_fit_dict(fit.x),
        "y": _axis_fit_dict(fit.y),
    }


def _observation_dict(value: CalibrationObservation) -> dict[str, object]:
    return {
        "confidence": value.confidence,
        "error_x": value.error_x,
        "error_y": value.error_y,
        "exact_label": value.exact_label,
        "is_prediction": value.is_prediction,
        "measurement_ns": value.measurement_ns,
        "normalized_bbox": list(value.normalized_bbox),
        "self_safe": value.self_safe,
        "target_identity": value.target_identity,
        "unique_candidates": value.unique_candidates,
    }


def _pulse_dict(value: SessionPulseRecord) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
    }


def _evidence_document(
    evidence: CalibrationSessionEvidence,
    *,
    include_hash: bool,
) -> dict[str, object]:
    document: dict[str, object] = {
        "binding": _binding_dict(evidence.binding),
        "cleanup_error": evidence.cleanup_error,
        "core_evidence_sha256": evidence.core_evidence_sha256,
        "emitted_events": [list(event) for event in evidence.emitted_events],
        "evidence_complete": evidence.evidence_complete,
        "fit": _fit_dict(evidence.fit),
        "held_ns": evidence.held_ns,
        "observations": [
            {
                "activation_known": record.activation_known,
                "activation_pressed": record.activation_pressed,
                "now_ns": record.now_ns,
                "observation": (
                    _observation_dict(record.observation)
                    if record.observation is not None
                    else None
                ),
                "state": record.state,
            }
            for record in evidence.observations
        ],
        "outcome": evidence.outcome,
        "pulses": [_pulse_dict(record) for record in evidence.pulses],
        "reason": evidence.reason,
        "schema_version": SESSION_EVIDENCE_SCHEMA_VERSION,
        "started_ns": evidence.started_ns,
        "terminal_ns": evidence.terminal_ns,
    }
    if include_hash:
        document["artifact_sha256"] = evidence.artifact_sha256
    return document


def _canonical_bytes(value: object) -> bytes:
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


def _make_session_evidence(
    *,
    binding: CalibrationRuntimeBinding,
    outcome: Literal["success", "aborted"],
    reason: str,
    cleanup_error: str | None,
    evidence_complete: bool,
    started_ns: int,
    held_ns: int | None,
    terminal_ns: int,
    observations: tuple[SessionObservationRecord, ...],
    pulses: tuple[SessionPulseRecord, ...],
    emitted_events: tuple[tuple[int, int, int], ...],
    fit: MakcuCalibrationFit | None,
) -> CalibrationSessionEvidence:
    placeholder = CalibrationSessionEvidence(
        binding=binding,
        outcome=outcome,
        reason=reason,
        cleanup_error=cleanup_error,
        evidence_complete=evidence_complete,
        started_ns=started_ns,
        held_ns=held_ns,
        terminal_ns=terminal_ns,
        observations=observations,
        pulses=pulses,
        emitted_events=emitted_events,
        fit=fit,
        core_evidence_sha256=fit.evidence_sha256 if fit else None,
        artifact_sha256="0" * 64,
    )
    digest = sha256(
        _canonical_bytes(_evidence_document(placeholder, include_hash=False))
    ).hexdigest()
    return CalibrationSessionEvidence(
        binding=binding,
        outcome=outcome,
        reason=reason,
        cleanup_error=cleanup_error,
        evidence_complete=evidence_complete,
        started_ns=started_ns,
        held_ns=held_ns,
        terminal_ns=terminal_ns,
        observations=observations,
        pulses=pulses,
        emitted_events=emitted_events,
        fit=fit,
        core_evidence_sha256=fit.evidence_sha256 if fit else None,
        artifact_sha256=digest,
    )


def session_evidence_bytes(evidence: CalibrationSessionEvidence) -> bytes:
    """Return strict canonical evidence bytes after recomputing its raw hash."""

    _validate_session_evidence(evidence)
    expected = sha256(
        _canonical_bytes(_evidence_document(evidence, include_hash=False))
    ).hexdigest()
    if evidence.artifact_sha256 != expected:
        raise CalibrationEvidenceError("artifact_sha256 does not match the evidence")
    payload = _canonical_bytes(_evidence_document(evidence, include_hash=True))
    if len(payload) > MAX_SESSION_EVIDENCE_BYTES:
        raise CalibrationEvidenceError("session evidence payload is unexpectedly large")
    return payload


def _validate_session_evidence(evidence: CalibrationSessionEvidence) -> None:
    if not isinstance(evidence, CalibrationSessionEvidence):
        raise CalibrationEvidenceError("evidence has the wrong type")
    if not isinstance(evidence.binding, CalibrationRuntimeBinding):
        raise CalibrationEvidenceError("evidence runtime binding is invalid")
    if evidence.outcome not in ("success", "aborted"):
        raise CalibrationEvidenceError("evidence outcome is invalid")
    try:
        _bounded_text(evidence.reason, "evidence reason")
        _timestamp(evidence.started_ns, "evidence started_ns")
        _timestamp(evidence.terminal_ns, "evidence terminal_ns")
        if evidence.held_ns is not None:
            _timestamp(evidence.held_ns, "evidence held_ns")
    except ValueError as exc:
        raise CalibrationEvidenceError(str(exc)) from exc
    if evidence.terminal_ns < evidence.started_ns:
        raise CalibrationEvidenceError("evidence terminal time precedes its start")
    if evidence.held_ns is not None and not (
        evidence.started_ns <= evidence.held_ns <= evidence.terminal_ns
    ):
        raise CalibrationEvidenceError("evidence held time is outside the session")
    if not isinstance(evidence.evidence_complete, bool):
        raise CalibrationEvidenceError("evidence_complete must be bool")
    if evidence.cleanup_error is not None:
        try:
            _bounded_text(evidence.cleanup_error, "cleanup_error")
        except ValueError as exc:
            raise CalibrationEvidenceError(str(exc)) from exc
    if evidence.outcome == "success":
        if not evidence.evidence_complete or evidence.cleanup_error is not None:
            raise CalibrationEvidenceError("successful evidence is not complete and clean")
        if evidence.held_ns is None:
            raise CalibrationEvidenceError("successful evidence lacks a hold timestamp")
        if not isinstance(evidence.fit, MakcuCalibrationFit):
            raise CalibrationEvidenceError("successful evidence lacks a numeric fit")
        if evidence.core_evidence_sha256 != evidence.fit.evidence_sha256:
            raise CalibrationEvidenceError("core evidence hash does not match the fit")
    elif evidence.fit is not None or evidence.core_evidence_sha256 is not None:
        raise CalibrationEvidenceError("aborted evidence must not contain an accepted fit")
    if not isinstance(evidence.artifact_sha256, str) or _HASH_RE.fullmatch(
        evidence.artifact_sha256
    ) is None:
        raise CalibrationEvidenceError("artifact_sha256 must be lowercase SHA-256")

    previous_now: int | None = None
    for record in evidence.observations:
        if not isinstance(record, SessionObservationRecord):
            raise CalibrationEvidenceError("observation record has the wrong type")
        if previous_now is not None and record.now_ns <= previous_now:
            raise CalibrationEvidenceError("observation record times are not ordered")
        previous_now = record.now_ns
        if record.state not in {state.value for state in CalibrationSessionState}:
            raise CalibrationEvidenceError("observation record state is invalid")
        if not isinstance(record.activation_known, bool) or not isinstance(
            record.activation_pressed, bool
        ):
            raise CalibrationEvidenceError("observation activation flags are invalid")
        if record.observation is not None and not isinstance(
            record.observation, CalibrationObservation
        ):
            raise CalibrationEvidenceError("raw observation has the wrong type")

    previous_event_ns: int | None = None
    emitted_abs = 0
    for event in evidence.emitted_events:
        if not isinstance(event, tuple) or len(event) != 3:
            raise CalibrationEvidenceError("emitted event is malformed")
        timestamp_ns, delta_x, delta_y = event
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
            or previous_event_ns is not None
            and timestamp_ns <= previous_event_ns
        ):
            raise CalibrationEvidenceError("emitted event timestamps are not ordered")
        if (
            isinstance(delta_x, bool)
            or not isinstance(delta_x, int)
            or isinstance(delta_y, bool)
            or not isinstance(delta_y, int)
            or bool(delta_x) == bool(delta_y)
        ):
            raise CalibrationEvidenceError("emitted event is not single-axis movement")
        previous_event_ns = timestamp_ns
        emitted_abs += abs(delta_x) + abs(delta_y)
    if emitted_abs > CALIBRATION_MAX_SESSION_ABS_COUNTS:
        raise CalibrationEvidenceError("evidence exceeds the controller motion budget")

    covered_event_index = 0
    for pulse in evidence.pulses:
        if not isinstance(pulse, SessionPulseRecord):
            raise CalibrationEvidenceError("pulse record has the wrong type")
        if pulse.axis not in ("x", "y") or pulse.polarity not in (-1, 1):
            raise CalibrationEvidenceError("pulse axis or polarity is invalid")
        if (
            isinstance(pulse.requested_counts, bool)
            or not isinstance(pulse.requested_counts, int)
            or not 1 <= pulse.requested_counts <= CALIBRATION_MAX_EXCURSION_COUNTS
        ):
            raise CalibrationEvidenceError("requested pulse count is invalid")
        if not 0.0 < _finite(pulse.requested_rate, "pulse rate") <= (
            CALIBRATION_MAX_RATE_COUNTS_PER_SECOND
        ):
            raise CalibrationEvidenceError("requested pulse rate is invalid")
        if pulse.event_start_index != covered_event_index:
            raise CalibrationEvidenceError("pulse event ranges are not contiguous")
        if not (
            pulse.event_start_index
            <= pulse.event_end_index
            <= len(evidence.emitted_events)
        ):
            raise CalibrationEvidenceError("pulse event range is invalid")
        pulse_events = evidence.emitted_events[
            pulse.event_start_index : pulse.event_end_index
        ]
        covered_event_index = pulse.event_end_index
        actual = sum(
            delta_x if pulse.axis == "x" else delta_y
            for _timestamp_ns, delta_x, delta_y in pulse_events
        )
        if actual != pulse.actual_counts:
            raise CalibrationEvidenceError("pulse actual count does not match events")
        if pulse.complete:
            if not pulse_events:
                raise CalibrationEvidenceError("completed pulse has no emitted events")
            if (
                pulse.first_emitted_ns != pulse_events[0][0]
                or pulse.last_emitted_ns != pulse_events[-1][0]
                or pulse.actual_counts != pulse.polarity * pulse.requested_counts
            ):
                raise CalibrationEvidenceError("completed pulse evidence is inconsistent")
        if not isinstance(pulse.qualifying, bool) or not isinstance(pulse.complete, bool):
            raise CalibrationEvidenceError("pulse status flags must be bool")
    if covered_event_index != len(evidence.emitted_events):
        raise CalibrationEvidenceError("emitted events are not covered by pulse records")
    if evidence.outcome == "success":
        assert evidence.fit is not None and evidence.held_ns is not None
        measurements = tuple(
            CalibrationMeasurement(
                record.observation.measurement_ns,
                record.observation.error_x,
                record.observation.error_y,
                True,
            )
            for record in evidence.observations
            if record.now_ns >= evidence.held_ns and record.observation is not None
        )
        commands = tuple(
            EmittedCount(timestamp_ns, delta_x, delta_y)
            for timestamp_ns, delta_x, delta_y in evidence.emitted_events
        )
        calibration_pulses = tuple(
            CalibrationPulse(
                record.axis,
                record.polarity,
                int(record.first_emitted_ns),
                int(record.last_emitted_ns),
            )
            for record in evidence.pulses
            if record.complete
            and record.first_emitted_ns is not None
            and record.last_emitted_ns is not None
        )
        try:
            refit = fit_makcu_calibration(
                measurements,
                commands,
                calibration_pulses,
            )
        except (CalibrationDataError, CalibrationQualityError) as exc:
            raise CalibrationEvidenceError(
                f"persisted success evidence does not refit: {exc}"
            ) from exc
        if refit != evidence.fit:
            raise CalibrationEvidenceError(
                "persisted numeric fit does not match a fresh evidence refit"
            )


def _expect_fields(
    value: object,
    expected: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CalibrationEvidenceError(f"{name} fields are not canonical")
    return value


def _json_int(value: object, name: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationEvidenceError(f"{name} must be an integer")
    return value


def _json_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise CalibrationEvidenceError(f"{name} must be bool")
    return value


def _json_text(value: object, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str):
        raise CalibrationEvidenceError(f"{name} must be a string")
    return value


def _binding_from_dict(value: object) -> CalibrationRuntimeBinding:
    expected = set(CalibrationRuntimeBinding.__dataclass_fields__)
    data = _expect_fields(value, expected, "runtime binding")
    try:
        return CalibrationRuntimeBinding(
            model_sha256=str(_json_text(data["model_sha256"], "model_sha256")),
            labels_sha256=str(_json_text(data["labels_sha256"], "labels_sha256")),
            source_commit=str(_json_text(data["source_commit"], "source_commit")),
            build_identity=str(_json_text(data["build_identity"], "build_identity")),
            backend=str(_json_text(data["backend"], "backend")),
            runtime_version=str(
                _json_text(data["runtime_version"], "runtime_version")
            ),
            requested_provider=str(
                _json_text(data["requested_provider"], "requested_provider")
            ),
            active_provider=str(_json_text(data["active_provider"], "active_provider")),
            active_device=str(_json_text(data["active_device"], "active_device")),
            provider_options_sha256=str(
                _json_text(
                    data["provider_options_sha256"],
                    "provider_options_sha256",
                )
            ),
            physical_device_token=str(
                _json_text(data["physical_device_token"], "physical_device_token")
            ),
            inference_width=int(
                _json_int(data["inference_width"], "inference_width")
            ),
            inference_height=int(
                _json_int(data["inference_height"], "inference_height")
            ),
            detail_pass_enabled=_json_bool(
                data["detail_pass_enabled"], "detail_pass_enabled"
            ),
            capture_kind=str(_json_text(data["capture_kind"], "capture_kind")),  # type: ignore[arg-type]
            capture_backend=str(
                _json_text(data["capture_backend"], "capture_backend")
            ),
            capture_buffer_size=int(
                _json_int(data["capture_buffer_size"], "capture_buffer_size")
            ),
            capture_index=str(_json_text(data["capture_index"], "capture_index")),
            capture_width=int(_json_int(data["capture_width"], "capture_width")),
            capture_height=int(_json_int(data["capture_height"], "capture_height")),
            capture_fps=_finite(data["capture_fps"], "capture_fps"),
            pixel_format=str(_json_text(data["pixel_format"], "pixel_format")),
            rotation_degrees=int(
                _json_int(data["rotation_degrees"], "rotation_degrees")
            ),
            makcu_identity_token=str(
                _json_text(data["makcu_identity_token"], "makcu_identity_token")
            ),
            activation_button=int(
                _json_int(data["activation_button"], "activation_button")
            ),
            aim_label=str(_json_text(data["aim_label"], "aim_label")),
            head_ratio=_finite(data["head_ratio"], "head_ratio"),
            invert_x=_json_bool(data["invert_x"], "invert_x"),
            invert_y=_json_bool(data["invert_y"], "invert_y"),
            context_name=str(_json_text(data["context_name"], "context_name")),
            aim_mode=str(_json_text(data["aim_mode"], "aim_mode")),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise CalibrationEvidenceError(str(exc)) from exc


def _observation_from_dict(value: object) -> CalibrationObservation:
    expected = set(CalibrationObservation.__dataclass_fields__)
    data = _expect_fields(value, expected, "raw observation")
    bbox = data["normalized_bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise CalibrationEvidenceError("raw observation bbox is invalid")
    try:
        return CalibrationObservation(
            measurement_ns=int(_json_int(data["measurement_ns"], "measurement_ns")),
            error_x=_finite(data["error_x"], "error_x"),
            error_y=_finite(data["error_y"], "error_y"),
            confidence=_finite(data["confidence"], "confidence"),
            exact_label=_json_bool(data["exact_label"], "exact_label"),
            unique_candidates=int(
                _json_int(data["unique_candidates"], "unique_candidates")
            ),
            self_safe=_json_bool(data["self_safe"], "self_safe"),
            is_prediction=_json_bool(data["is_prediction"], "is_prediction"),
            target_identity=str(
                _json_text(data["target_identity"], "target_identity")
            ),
            normalized_bbox=tuple(_finite(item, "bbox") for item in bbox),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise CalibrationEvidenceError(str(exc)) from exc


def _axis_fit_from_dict(value: object, expected_axis: str) -> AxisCalibrationFit:
    fields = set(AxisCalibrationFit.__dataclass_fields__)
    data = _expect_fields(value, fields, f"{expected_axis}-axis fit")
    try:
        fit = AxisCalibrationFit(
            axis=str(_json_text(data["axis"], "fit axis")),
            gain_pixels_per_count=_finite(data["gain_pixels_per_count"], "gain"),
            delay_seconds=_finite(data["delay_seconds"], "delay"),
            drift_pixels_per_second=_finite(data["drift_pixels_per_second"], "drift"),
            r_squared=_finite(data["r_squared"], "R-squared"),
            gain_cv=_finite(data["gain_cv"], "gain CV"),
            polarity_mismatch=_finite(data["polarity_mismatch"], "polarity mismatch"),
            cross_axis_ratio=_finite(data["cross_axis_ratio"], "cross-axis ratio"),
            delay_ambiguity_seconds=_finite(
                data["delay_ambiguity_seconds"], "delay ambiguity"
            ),
            pulse_delay_spread_seconds=_finite(
                data["pulse_delay_spread_seconds"], "pulse delay spread"
            ),
            minimum_excursion_pixels=_finite(
                data["minimum_excursion_pixels"], "minimum excursion"
            ),
            maximum_excursion_pixels=_finite(
                data["maximum_excursion_pixels"], "maximum excursion"
            ),
            positive_pulses=int(_json_int(data["positive_pulses"], "positive pulses")),
            negative_pulses=int(_json_int(data["negative_pulses"], "negative pulses")),
        )
    except (CalibrationDataError, ValueError) as exc:
        raise CalibrationEvidenceError(str(exc)) from exc
    if fit.axis != expected_axis:
        raise CalibrationEvidenceError("fit axis is not canonical")
    return fit


def _fit_from_dict(value: object) -> MakcuCalibrationFit | None:
    if value is None:
        return None
    expected = {
        "delay_seconds",
        "detector_period_seconds",
        "evidence_sha256",
        "observation_duty",
        "x",
        "y",
    }
    data = _expect_fields(value, expected, "numeric fit")
    try:
        return MakcuCalibrationFit(
            x=_axis_fit_from_dict(data["x"], "x"),
            y=_axis_fit_from_dict(data["y"], "y"),
            delay_seconds=_finite(data["delay_seconds"], "shared delay"),
            detector_period_seconds=_finite(
                data["detector_period_seconds"], "detector period"
            ),
            observation_duty=_finite(data["observation_duty"], "observation duty"),
            evidence_sha256=str(
                _json_text(data["evidence_sha256"], "fit evidence_sha256")
            ),
        )
    except (CalibrationDataError, ValueError) as exc:
        raise CalibrationEvidenceError(str(exc)) from exc


def _pulse_from_dict(value: object) -> SessionPulseRecord:
    fields = set(SessionPulseRecord.__dataclass_fields__)
    data = _expect_fields(value, fields, "pulse record")

    def optional_finite(name: str) -> float | None:
        return None if data[name] is None else _finite(data[name], name)

    return SessionPulseRecord(
        axis=str(_json_text(data["axis"], "pulse axis")),
        polarity=int(_json_int(data["polarity"], "pulse polarity")),
        requested_counts=int(
            _json_int(data["requested_counts"], "requested counts")
        ),
        requested_rate=_finite(data["requested_rate"], "requested rate"),
        request_ns=int(_json_int(data["request_ns"], "request_ns")),
        event_start_index=int(
            _json_int(data["event_start_index"], "event_start_index")
        ),
        event_end_index=int(_json_int(data["event_end_index"], "event_end_index")),
        first_emitted_ns=_json_int(
            data["first_emitted_ns"], "first_emitted_ns", optional=True
        ),
        last_emitted_ns=_json_int(
            data["last_emitted_ns"], "last_emitted_ns", optional=True
        ),
        actual_counts=int(_json_int(data["actual_counts"], "actual_counts")),
        baseline_x=_finite(data["baseline_x"], "baseline_x"),
        baseline_y=_finite(data["baseline_y"], "baseline_y"),
        settled_x=optional_finite("settled_x"),
        settled_y=optional_finite("settled_y"),
        signed_response_pixels=optional_finite("signed_response_pixels"),
        cross_response_pixels=optional_finite("cross_response_pixels"),
        qualifying=_json_bool(data["qualifying"], "qualifying"),
        complete=_json_bool(data["complete"], "complete"),
    )


def session_evidence_from_bytes(payload: bytes) -> CalibrationSessionEvidence:
    """Parse only exact canonical UTF-8 evidence and verify both hashes."""

    if not isinstance(payload, bytes):
        raise CalibrationEvidenceError("evidence payload must be bytes")
    if len(payload) > MAX_SESSION_EVIDENCE_BYTES:
        raise CalibrationEvidenceError("session evidence payload is unexpectedly large")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationEvidenceError("evidence is not valid UTF-8 JSON") from exc
    expected = {
        "artifact_sha256",
        "binding",
        "cleanup_error",
        "core_evidence_sha256",
        "emitted_events",
        "evidence_complete",
        "fit",
        "held_ns",
        "observations",
        "outcome",
        "pulses",
        "reason",
        "schema_version",
        "started_ns",
        "terminal_ns",
    }
    data = _expect_fields(document, expected, "session evidence")
    if _json_int(data["schema_version"], "schema_version") != (
        SESSION_EVIDENCE_SCHEMA_VERSION
    ):
        raise CalibrationEvidenceError("unsupported session evidence schema")
    observations_value = data["observations"]
    if not isinstance(observations_value, list):
        raise CalibrationEvidenceError("observations must be a list")
    observations: list[SessionObservationRecord] = []
    observation_fields = set(SessionObservationRecord.__dataclass_fields__)
    for value in observations_value:
        record = _expect_fields(value, observation_fields, "observation record")
        raw_observation = record["observation"]
        observations.append(
            SessionObservationRecord(
                now_ns=int(_json_int(record["now_ns"], "observation now_ns")),
                activation_known=_json_bool(
                    record["activation_known"], "activation_known"
                ),
                activation_pressed=_json_bool(
                    record["activation_pressed"], "activation_pressed"
                ),
                state=str(_json_text(record["state"], "observation state")),
                observation=(
                    None
                    if raw_observation is None
                    else _observation_from_dict(raw_observation)
                ),
            )
        )
    events_value = data["emitted_events"]
    if not isinstance(events_value, list):
        raise CalibrationEvidenceError("emitted_events must be a list")
    events: list[tuple[int, int, int]] = []
    for event in events_value:
        if not isinstance(event, list) or len(event) != 3:
            raise CalibrationEvidenceError("emitted event JSON is malformed")
        events.append(
            (
                int(_json_int(event[0], "event timestamp")),
                int(_json_int(event[1], "event delta_x")),
                int(_json_int(event[2], "event delta_y")),
            )
        )
    pulses_value = data["pulses"]
    if not isinstance(pulses_value, list):
        raise CalibrationEvidenceError("pulses must be a list")
    fit = _fit_from_dict(data["fit"])
    evidence = CalibrationSessionEvidence(
        binding=_binding_from_dict(data["binding"]),
        outcome=str(_json_text(data["outcome"], "outcome")),  # type: ignore[arg-type]
        reason=str(_json_text(data["reason"], "reason")),
        cleanup_error=_json_text(data["cleanup_error"], "cleanup_error", optional=True),
        evidence_complete=_json_bool(data["evidence_complete"], "evidence_complete"),
        started_ns=int(_json_int(data["started_ns"], "started_ns")),
        held_ns=_json_int(data["held_ns"], "held_ns", optional=True),
        terminal_ns=int(_json_int(data["terminal_ns"], "terminal_ns")),
        observations=tuple(observations),
        pulses=tuple(_pulse_from_dict(value) for value in pulses_value),
        emitted_events=tuple(events),
        fit=fit,
        core_evidence_sha256=_json_text(
            data["core_evidence_sha256"],
            "core_evidence_sha256",
            optional=True,
        ),
        artifact_sha256=str(
            _json_text(data["artifact_sha256"], "artifact_sha256")
        ),
    )
    canonical = session_evidence_bytes(evidence)
    if canonical != payload:
        raise CalibrationEvidenceError("session evidence JSON is not canonical")
    return evidence


def _read_private_regular_evidence_file(path: Path) -> bytes:
    path_metadata = path.lstat()
    if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISREG(
        path_metadata.st_mode
    ):
        raise CalibrationEvidenceError("session evidence path is not a regular file")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        current_path_metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not stat.S_ISREG(current_path_metadata.st_mode)
            or not os.path.samestat(path_metadata, metadata)
            or not os.path.samestat(current_path_metadata, metadata)
        ):
            raise CalibrationEvidenceError(
                "session evidence path changed while it was opened"
            )
        if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise PermissionError("session evidence file must have mode 0600")
        if metadata.st_size > MAX_SESSION_EVIDENCE_BYTES:
            raise CalibrationEvidenceError(
                "session evidence file is unexpectedly large"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            payload = stream.read(MAX_SESSION_EVIDENCE_BYTES + 1)
        if len(payload) > MAX_SESSION_EVIDENCE_BYTES:
            raise CalibrationEvidenceError(
                "session evidence file is unexpectedly large"
            )
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_session_evidence(path: str | Path) -> CalibrationSessionEvidence:
    """Load one bounded canonical private evidence artifact without following links."""

    return session_evidence_from_bytes(
        _read_private_regular_evidence_file(Path(path))
    )


def evidence_matches_binding(
    evidence: CalibrationSessionEvidence,
    binding: CalibrationRuntimeBinding,
) -> bool:
    """Use exact immutable identity equality when deciding whether success is reusable."""

    if not isinstance(binding, CalibrationRuntimeBinding):
        return False
    try:
        session_evidence_bytes(evidence)
    except (CalibrationEvidenceError, CalibrationDataError, CalibrationQualityError):
        return False
    return (
        evidence.outcome == "success"
        and evidence.evidence_complete
        and evidence.fit is not None
        and evidence.cleanup_error is None
        and evidence.binding == binding
    )


def write_session_evidence_exclusive(
    path: str | Path,
    evidence: CalibrationSessionEvidence,
) -> None:
    """Atomically create canonical mode-0600 evidence without overwriting a path."""

    destination = Path(path)
    payload = session_evidence_bytes(evidence)
    parent_preexisted = destination.parent.exists()
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if os.name == "posix" and not parent_preexisted:
        os.chmod(destination.parent, 0o700)
    parent_mode = stat.S_IMODE(destination.parent.stat().st_mode)
    if os.name == "posix" and parent_mode & 0o077:
        raise PermissionError(
            "calibration evidence directory must not be group/world accessible"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != payload:
            raise OSError("temporary evidence verification failed")
        os.link(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The complete hard link is already the atomic commit point. Some
            # platforms cannot open/fsync directory descriptors.
            pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
