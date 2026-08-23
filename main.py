from __future__ import annotations

from collections.abc import Mapping
import json
import math
import sys
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from time import perf_counter_ns

from aiming.direct_head_anchor import (
    DIRECT_HEAD_ANCHOR_MAX_AGE_SECONDS,
    DirectHeadAnchor,
    DirectHeadAnchorSample,
    DirectHeadProvenance,
)
from config import AppConfig, parse_args
from utils.inference_size import compact_inference_size


WINDOW_NAME = "ProAim"
AIM_CONTINUATION_CONFIDENCE_FLOOR = 0.15
# Calibration only needs to *track* a clearly visible player's box motion
# across the bounded pulses; it never makes an aim decision from it.  The
# 0.98 observation-duty gate in the fit requires the box to be observed in
# nearly every frame, so calibration uses a lower confidence floor than the
# configured aim threshold.  Live evidence showed a visible centered player
# flickering around a 0.25 configured threshold (36% of frames below it),
# producing a 0.67 duty and a rejected fit.
CALIBRATION_CONFIDENCE_FLOOR = 0.12
AUTOMATIC_MAKCU_GAIN_X_PIXELS_PER_COUNT = 0.125
AUTOMATIC_MAKCU_GAIN_Y_PIXELS_PER_COUNT = 0.120
AUTOMATIC_MAKCU_DELAY_SECONDS = 0.006
AUTOMATIC_MAKCU_EMPTY_GRACE_FRAMES = 8
DEFAULT_TARGET_TRACK_LOST_GRACE_FRAMES = 6
AUTOMATIC_MAKCU_STALE_AFTER_SECONDS = 0.110
AUTOMATIC_HEAD_LOCALIZATION_HZ = 90.0
# The direct-head model is useful for acquisition and drift correction, but it
# shares the GPU with the primary detector. Running it latest-only at 90 Hz
# after an anchor is established cut the measured coordinate loop from roughly
# 150 Hz to 60--90 Hz while held, without increasing the worker's 24--35 Hz
# completion rate. Keep the fast cadence for acquisition and explicit model
# recovery; an active measured anchor uses lower-rate drift corrections whether
# LK is currently pixel-qualified or body-carried. The lower cadence does not
# extend the identity lease or grant optical-flow authority on its own.
AUTOMATIC_HEAD_TRACKING_LOCALIZATION_HZ = 24.0
AUTOMATIC_HEAD_TRACKING_MINIMUM_LEASE_REMAINING_SECONDS = 0.300
AUTOMATIC_HEAD_TRACKING_MINIMUM_FLOW_SAMPLES = 3
AUTOMATIC_HEAD_TRACKING_MAX_CONSECUTIVE_MISSES = 3
AUTOMATIC_HEAD_FLOW_MAX_CONSECUTIVE_FAILURES = 2
AUTOMATIC_HEAD_STALE_AFTER_SECONDS = AUTOMATIC_MAKCU_STALE_AFTER_SECONDS
AUTOMATIC_HEAD_PROVIDER = "MIGraphXExecutionProvider"
AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS = 0.012
AUTOMATIC_HEAD_MAPPED_VELOCITY_FILTER_TIME_CONSTANT_SECONDS = 0.012
# Direct-head detections refresh a location *inside* the current body box.
# Smooth that local offset independently so pose/localizer jitter cannot move
# the aim point sharply, while current primary-box translation remains free to
# carry the target across the screen without this 60 ms delay.
AUTOMATIC_HEAD_NORMALIZED_ANCHOR_FILTER_TIME_CONSTANT_SECONDS = 0.060
AUTOMATIC_HEAD_VELOCITY_RECONCILIATION_TIME_CONSTANT_SECONDS = 0.030
# The live 640 model's accepted head boxes are commonly only about 20x15
# source pixels.  The phase-correction path therefore uses a tiny-feature
# contract (still forward/backward and spatial-span gated) and excludes only
# the central eight pixels of a fixed reticle.  The final point must also stay
# close to the independently body-carried endpoint before it can replace it.
AUTOMATIC_HEAD_FLOW_MAX_BODY_RESIDUAL_PIXELS = 12.0
# Agreement with the body-mapped anchor is a safety check, not a second
# position sensor.  A fixed 12 px leash rejected ordinary live head/localizer
# disagreement (15 px median in the recorded trace), which made the newest-
# capture LK path unreachable precisely during motion.  Admit only a bounded,
# source-time-scaled amount of target-relative motion and detector-box shape
# uncertainty; the exact measured player, anatomical head region, robust LK
# checks, immutable identity deadline, and the hard absolute cap remain.
AUTOMATIC_HEAD_FLOW_RELATIVE_MOTION_PIXELS_PER_SECOND = 900.0
AUTOMATIC_HEAD_FLOW_MAX_GEOMETRY_UNCERTAINTY_PIXELS = 8.0
AUTOMATIC_HEAD_FLOW_MAX_DYNAMIC_RESIDUAL_PIXELS = 36.0
AUTOMATIC_HEAD_FLOW_MAX_PHASE_ADVANCE_SECONDS = 0.075
# The ordinary small-player cap remains an 18 px circle.  After three exact,
# same-generation body measurements prove coherent translation, only the axis
# parallel to that measured translation may consume a little more of the
# already time-scaled corridor.  Keeping the orthogonal axis at 18 px prevents
# detector jitter from turning this fast-motion allowance into head wander.
AUTOMATIC_HEAD_FLOW_SMALL_PLAYER_RESIDUAL_CAP_PIXELS = 18.0
AUTOMATIC_HEAD_FLOW_SMALL_PLAYER_LONGITUDINAL_RESIDUAL_CAP_PIXELS = 24.0
AUTOMATIC_HEAD_FLOW_MIN_DIRECTIONAL_BODY_DISPLACEMENT_PIXELS = 6.0
# Up to two rejected LK steps may retain private coordinate continuity for a
# prompt pixel-track recovery. They must not expose raw body-box translation to
# control; publication returns to the calmer mapped-position filter immediately.
AUTOMATIC_HEAD_FLOW_MAX_BODY_CARRY_SECONDS = 0.045
# A body inference is deliberately not manufactured for a frame captured while
# the synchronous detector was busy.  A strict LK endpoint may nevertheless
# translate the already-verified head into that newer image. The endpoint
# remains position-only; an atomic controller handoff may retain only motion
# authority already proven by the inferred-frame root. Keep the unmeasured
# lead shorter than one ordinary slow inference interval, and never renew
# identity from it.
AUTOMATIC_HEAD_CAPTURE_PHASE_MAX_LEAD_SECONDS = 0.025
# Consecutive exact measurements of the same TargetTracker generation can see
# the sum of camera motion and independent target motion.  Keep this widened
# envelope out of predicted geometry.  A disjoint measured-primary endpoint
# can use it only after a following exact sample proves a short, coherent
# same-direction trajectory in the same tracker generation.
AUTOMATIC_HEAD_EXACT_BODY_ASSOCIATION_MAX_SPEED_PIXELS_PER_SECOND = 4800.0
AUTOMATIC_HEAD_CONFIRMED_DISJOINT_MINIMUM_DIRECTION_COSINE = 0.75
# A same-generation detector can emit one short-lived body fragment between
# two mutually consistent boxes for the same visible player.  Remember at
# most this short three-sample span so the singleton can be rolled out of the
# exact binding chain without extending identity through a genuine absence or
# target transition.
AUTOMATIC_HEAD_SINGLETON_BODY_OUTLIER_MAX_SPAN_SECONDS = 0.050
# Bound only implausible frame-to-frame mapped-head innovations before they
# enter either controller channel.  The fixed allowance preserves small
# detector motion exactly; 3600 px/s covers the calibrated plant's 3327 px/s
# maximum diagonal screen motion with a small scheduling margin, without
# passing the broader identity-only envelope into control.
AUTOMATIC_HEAD_MAPPED_STEP_ALLOWANCE_PIXELS = 4.0
AUTOMATIC_HEAD_MAPPED_MAX_SPEED_PIXELS_PER_SECOND = 3600.0
AUTOMATIC_MAKCU_POSITION_TIME_CONSTANT_SECONDS = 0.040
AUTOMATIC_MAKCU_FEEDBACK_DEADZONE_PIXELS = 3.5
AUTOMATIC_DIRECT_HEAD_POSITION_TIME_CONSTANT_SECONDS = 0.028
# Keep near-lock feedback calm while restoring prompt positional pursuit only
# after the residual is clearly outside the 4.5 px deadzone + 6 px shoulder.
# The smooth per-axis schedule reaches 16 ms at 22 px; paired live-cadence
# sweeps cut stationary command volume without slowing 1800 px/s tracking.
AUTOMATIC_DIRECT_HEAD_PURSUIT_POSITION_TIME_CONSTANT_SECONDS = 0.016
AUTOMATIC_DIRECT_HEAD_PURSUIT_START_PIXELS = 10.5
AUTOMATIC_DIRECT_HEAD_PURSUIT_FULL_PIXELS = 22.0
AUTOMATIC_DIRECT_HEAD_FEEDBACK_DEADZONE_PIXELS = 4.5
AUTOMATIC_DIRECT_HEAD_FEEDBACK_SHOULDER_PIXELS = 6.0
AUTOMATIC_DIRECT_HEAD_RECOMMENDED_MAX_STEP = 320
# A measured direct-head plant should be able to realize the same per-axis
# screen speed that the numeric observer is allowed to estimate.  Treat the
# legacy 320-step value as the user's ordinary safety envelope, then expand
# only a bound, automatic, measured-plant path far enough to remove the
# counts/second mismatch between axes.  At the current ADS gains this resolves
# to about 22.9k X / 26.9k Y counts/s.  Commands below the old 19.2k ceiling are
# bit-for-bit unchanged, so the near-lock filters/deadzone remain untouched.
AUTOMATIC_DIRECT_HEAD_MEASURED_SCREEN_ENVELOPE_PIXELS_PER_SECOND = 3_000.0
AUTOMATIC_DIRECT_HEAD_MEASURED_MAX_RATE_COUNTS_PER_SECOND = 27_000.0
AUTOMATIC_MAKCU_VELOCITY_MEDIAN_WINDOW = 3
AUTOMATIC_MAKCU_VELOCITY_FILTER_TIME_CONSTANT_SECONDS = 0.014
AUTOMATIC_MAKCU_MAX_TARGET_ACCELERATION_PIXELS_PER_SECOND_SQUARED = 40_000.0
AUTOMATIC_MAKCU_MAX_VELOCITY_FEEDFORWARD_FRACTION = 1.0
AUTOMATIC_MAKCU_MAX_BODY_DERIVED_PROJECTION_FRACTION = 1.0
AUTOMATIC_MAKCU_MAX_BODY_DERIVED_FEEDFORWARD_FRACTION = 0.50
# The newest-frame LK endpoint may retain only a staged slice of the fast
# reserve re-proven at its independently corroborated inferred-frame root.
# Live stop/reversal replay supports 0.60; carrying the full 0.90 reserve made
# post-stop overshoot more than twice as large. All ordinary and stationary
# near-lock authority remains at the existing 0.50 ceiling or below.
AUTOMATIC_MAKCU_MAX_CORRELATED_LOOKAHEAD_PURSUIT_FEEDFORWARD_FRACTION = 0.60
# Three consecutive accepted pixel-flow endpoints are a physical motion stream,
# not merely one position peek.  Once the inferred root has independently
# corroborated the same pursuit, that stream may carry a separate near-unity
# setpoint.  It remains behind the existing newest-endpoint stop/reversal and
# physical-input revocation gates; mapped-body and residual paths retain their
# lower, independently validated ceilings below.
AUTOMATIC_MAKCU_MAX_VERIFIED_FLOW_PURSUIT_FEEDFORWARD_FRACTION = 0.95
# Stay below unity even at the numeric observer's +20% plant-gain uncertainty
# boundary: 0.82 * 1.20 = 0.984. This leaves feedback authority to close the
# remaining error without turning a calibrated-gain mismatch into oscillation.
AUTOMATIC_MAKCU_MAX_BODY_DERIVED_PURSUIT_FEEDFORWARD_FRACTION = 0.82
# The active profile's Y-axis fit remains too weak to enlarge mapped-body
# pursuit. Use the same sub-unity bound for seed and measured body paths; only
# the independently gated consecutive-pixel stream receives the exception above.
AUTOMATIC_MEASURED_MAKCU_MAX_BODY_DERIVED_PURSUIT_FEEDFORWARD_FRACTION = 0.82
# Persistent, command-compensated trailing residual is independent evidence
# that the ordinary provenance confidence is withholding needed velocity.
# The numeric core learns this per axis, retains it through exact lock, and
# clears it on ahead error, stop/reversal, rejected measurements, loss, or
# reported physical input. Give this learned, non-pixel-independent path a
# lower ceiling than verified flow: phase sweeps found that 0.82 bought less
# than one extra pixel of common-band pursuit over 0.65 while enlarging stop
# and detector-jitter tails.
AUTOMATIC_MAKCU_MAX_RESIDUAL_PURSUIT_FEEDFORWARD_FRACTION = 0.65
AUTOMATIC_DIRECT_HEAD_ACQUISITION_CONFIDENCE_FLOOR = 0.15
# A body-ratio proxy does not share the direct model's coordinate schema.
# Live traces showed that its rare proxy-only publications bought effectively
# no hold coverage while creating isolated 47--82 px proxy-to-direct steps.
# Keep collecting its eligibility diagnostics, but never drive the controller
# from it until it has an independently validated spatial-continuity contract.
AUTOMATIC_HEAD_BODY_FALLBACK_CONTROL_ENABLED = False
# A direct-head decoder miss is different from an ambiguous localization: the
# former says that this exact crop produced no head candidate, while the latter
# may contain multiple people or contradictory geometry.  After two strong,
# exact primary measurements agree on one tracker/runtime identity, permit a
# short position-only body proxy only for the decoder-miss case.  It cannot
# create a DirectHeadAnchor, grant feed-forward, survive a measurement gap, or
# return after a real direct head has been seen in that identity.
AUTOMATIC_BODY_FALLBACK_CONFIDENCE = 0.40
AUTOMATIC_BODY_FALLBACK_CONFIRMATIONS = 2
AUTOMATIC_DIRECT_HEAD_AGGRESSIVE_ACQUISITION_MODE = False
AUTOMATIC_DETAIL_RESCUE_CROP_SIZE = 640
AUTOMATIC_DETAIL_REFERENCE_HEIGHT = 1080.0
AUTOMATIC_DETAIL_MAX_REFERENCE_HEIGHT = 96.0
AUTOMATIC_DETAIL_SELF_EDGE_MARGIN_MODEL_PIXELS = 4.0
# A detail ROI hint is acquisition geometry only, never target authority.  Keep
# it no longer than the existing direct-result freshness window so a missing
# full pass cannot make the rescue crop chase old screen coordinates.
AUTOMATIC_DETAIL_TARGET_HINT_MAX_AGE_SECONDS = AUTOMATIC_HEAD_STALE_AFTER_SECONDS


class _AutomaticBodyFallbackGate:
    """Qualify a bounded position-only acquisition proxy for one identity."""

    def __init__(
        self,
        *,
        confidence: float = AUTOMATIC_BODY_FALLBACK_CONFIDENCE,
        confirmations: int = AUTOMATIC_BODY_FALLBACK_CONFIRMATIONS,
    ) -> None:
        threshold = float(confidence)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("body fallback confidence must be finite and in [0, 1]")
        if (
            isinstance(confirmations, bool)
            or not isinstance(confirmations, int)
            or confirmations <= 0
        ):
            raise ValueError("body fallback confirmations must be positive")
        self.confidence = threshold
        self.confirmations = confirmations
        self._identity: tuple[int, int] | None = None
        self._strong_measurements = 0
        self._qualified = False
        self._direct_seen = False
        self._last_measurement_ns: int | None = None

    @staticmethod
    def _generation(value: int, description: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{description} must be a non-negative integer")
        return value

    def reset(self) -> None:
        self._identity = None
        self._strong_measurements = 0
        self._qualified = False
        self._direct_seen = False
        self._last_measurement_ns = None

    def pause(self) -> None:
        """Withdraw qualification without forgetting a direct-seen latch."""

        self._strong_measurements = 0
        self._qualified = False

    def observe(
        self,
        *,
        tracker_generation: int,
        runtime_generation: int,
        measurement_ns: int,
        accepted_confidence: float | None,
        strict_self_safe: bool,
        body_update_deferred: bool,
        no_decoded_head_verified: bool,
        direct_seen: bool,
    ) -> bool:
        """Return whether this exact measured frame may publish the proxy."""

        if not isinstance(strict_self_safe, bool):
            raise TypeError("strict_self_safe must be bool")
        if not isinstance(body_update_deferred, bool):
            raise TypeError("body_update_deferred must be bool")
        if not isinstance(no_decoded_head_verified, bool):
            raise TypeError("no_decoded_head_verified must be bool")
        if not isinstance(direct_seen, bool):
            raise TypeError("direct_seen must be bool")
        if (
            isinstance(measurement_ns, bool)
            or not isinstance(measurement_ns, int)
            or measurement_ns < 0
        ):
            raise ValueError("body fallback measurement timestamp must be non-negative")
        identity = (
            self._generation(runtime_generation, "runtime generation"),
            self._generation(tracker_generation, "tracker generation"),
        )
        if identity != self._identity:
            self._identity = identity
            self._strong_measurements = 0
            self._qualified = False
            self._direct_seen = False
            self._last_measurement_ns = None
        if direct_seen:
            self._direct_seen = True
            self._last_measurement_ns = measurement_ns
            self.pause()
            return False
        if self._direct_seen:
            self.pause()
            return False
        if (
            self._last_measurement_ns is not None
            and measurement_ns <= self._last_measurement_ns
        ):
            # A duplicated adapter call cannot manufacture the required two
            # exact-frame confirmations.
            self.pause()
            return False
        self._last_measurement_ns = measurement_ns
        if (
            accepted_confidence is None
            or not strict_self_safe
            or body_update_deferred
            or not no_decoded_head_verified
        ):
            self.pause()
            return False
        confidence = float(accepted_confidence)
        if not math.isfinite(confidence) or confidence < self.confidence:
            self.pause()
            return False
        self._strong_measurements += 1
        self._qualified = self._strong_measurements >= self.confirmations
        return self._qualified


@dataclass(frozen=True, slots=True)
class _AutomaticDetailTargetHint:
    center: tuple[float, float]
    source_timestamp_ns: int
    track_generation: int
    identity_generation: int


class _AutomaticDetailTargetHintState:
    """Retain one bounded, non-authoritative detail-crop location hint."""

    def __init__(
        self,
        max_age_seconds: float = AUTOMATIC_DETAIL_TARGET_HINT_MAX_AGE_SECONDS,
    ) -> None:
        maximum_age = float(max_age_seconds)
        if not math.isfinite(maximum_age) or maximum_age <= 0.0:
            raise ValueError("detail target hint max age must be finite and positive")
        self.max_age_ns = max(1, round(maximum_age * 1_000_000_000))
        self._hint: _AutomaticDetailTargetHint | None = None

    @staticmethod
    def _generation(value: int, description: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{description} must be a non-negative integer")
        return value

    @staticmethod
    def _timestamp(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("detail target hint timestamp must be non-negative")
        return value

    def clear(self) -> None:
        self._hint = None

    def remember_box(
        self,
        box,
        *,
        source_timestamp_ns: int,
        track_generation: int,
        identity_generation: int,
    ) -> None:
        if isinstance(box, (str, bytes)) or len(box) != 4:
            raise TypeError("detail target hint box must contain x1, y1, x2, y2")
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "detail target hint box must contain finite coordinates"
            ) from exc
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("detail target hint box coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("detail target hint box must have positive area")
        timestamp_ns = self._timestamp(source_timestamp_ns)
        tracker_generation = self._generation(
            track_generation,
            "detail target hint track generation",
        )
        runtime_generation = self._generation(
            identity_generation,
            "detail target hint identity generation",
        )
        self._hint = _AutomaticDetailTargetHint(
            center=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
            source_timestamp_ns=timestamp_ns,
            track_generation=tracker_generation,
            identity_generation=runtime_generation,
        )

    def center_if_valid(
        self,
        *,
        source_timestamp_ns: int,
        track_generation: int,
        identity_generation: int,
        activation_active: bool,
    ) -> tuple[float, float] | None:
        if not isinstance(activation_active, bool):
            raise TypeError("detail target hint activation state must be bool")
        if not activation_active:
            self.clear()
            return None
        current_ns = self._timestamp(source_timestamp_ns)
        tracker_generation = self._generation(
            track_generation,
            "detail target hint track generation",
        )
        runtime_generation = self._generation(
            identity_generation,
            "detail target hint identity generation",
        )
        hint = self._hint
        if hint is None:
            return None
        age_ns = current_ns - hint.source_timestamp_ns
        if (
            age_ns < 0
            or age_ns >= self.max_age_ns
            or tracker_generation != hint.track_generation
            or runtime_generation != hint.identity_generation
        ):
            self.clear()
            return None
        return hint.center


@dataclass(slots=True)
class _AutomaticDetailRescueTelemetry:
    """Count conditional detail decisions without retaining frame contents."""

    frames_evaluated: int = 0
    frames_activation_released: int = 0
    frames_triggered_no_exact_target: int = 0
    frames_triggered_small_central_target: int = 0
    frames_skipped_not_needed: int = 0
    frames_skipped_verified_anchor: int = 0

    def record(self, reason: str) -> None:
        self.frames_evaluated += 1
        if reason == "activation_released":
            self.frames_activation_released += 1
        elif reason == "no_exact_target":
            self.frames_triggered_no_exact_target += 1
        elif reason == "small_central_target":
            self.frames_triggered_small_central_target += 1
        elif reason == "not_needed":
            self.frames_skipped_not_needed += 1
        elif reason == "verified_anchor":
            self.frames_skipped_verified_anchor += 1
        else:
            raise ValueError(f"unknown automatic detail decision: {reason}")

    def snapshot(self) -> dict[str, int]:
        triggered = (
            self.frames_triggered_no_exact_target
            + self.frames_triggered_small_central_target
        )
        return {
            "automatic_frames_evaluated": self.frames_evaluated,
            "automatic_frames_triggered": triggered,
            "automatic_frames_activation_released": (
                self.frames_activation_released
            ),
            "automatic_frames_triggered_no_exact_target": (
                self.frames_triggered_no_exact_target
            ),
            "automatic_frames_triggered_small_central_target": (
                self.frames_triggered_small_central_target
            ),
            "automatic_frames_skipped_not_needed": self.frames_skipped_not_needed,
            "automatic_frames_skipped_verified_anchor": (
                self.frames_skipped_verified_anchor
            ),
        }


def _automatic_detail_rescue_reason(
    detections,
    frame_shape,
    detail_plan,
    *,
    aim_label: str,
    configured_confidence: float,
) -> str:
    """Return the bounded reason for a conditional centered detail inference.

    The caller supplies full-pass detections.  Only acquisition-authorized,
    exact-label evidence can suppress the no-target rescue.  When exact
    targets exist, the branch follows the same center preference as initial
    aim selection and runs only for a target whose source-space height is at
    most 96 pixels at a 1080p reference height.
    """

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    source_height = int(frame_shape[0])
    source_width = int(frame_shape[1])
    if source_height <= 0 or source_width <= 0:
        raise ValueError("frame dimensions must be positive")
    label = str(aim_label).strip().casefold()
    threshold = float(configured_confidence)
    if not label:
        raise ValueError("automatic detail rescue requires an aim label")
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("configured confidence must be finite and in [0,1]")

    exact = tuple(
        detection
        for detection in detections
        if (
            str(
                getattr(
                    detection,
                    "class_name",
                    getattr(detection, "label", ""),
                )
            ).strip().casefold()
            == label
            and float(detection.confidence) >= threshold
        )
    )
    if not exact:
        return "no_exact_target"

    roi_left = float(detail_plan.crop_x)
    roi_top = float(detail_plan.crop_y)
    roi_right = roi_left + float(detail_plan.applied_crop_width)
    roi_bottom = roi_top + float(detail_plan.applied_crop_height)
    source_center_x = source_width * 0.5
    source_center_y = source_height * 0.5
    in_roi = []
    for detection in exact:
        center_x = (float(detection.x1) + float(detection.x2)) * 0.5
        center_y = (float(detection.y1) + float(detection.y2)) * 0.5
        if roi_left <= center_x <= roi_right and roi_top <= center_y <= roi_bottom:
            distance_squared = (
                (center_x - source_center_x) ** 2
                + (center_y - source_center_y) ** 2
            )
            in_roi.append((distance_squared, detection))
    if not in_roi:
        return "not_needed"

    _distance, center_nearest = min(in_roi, key=lambda item: item[0])
    reference_height = (
        max(0.0, float(center_nearest.y2) - float(center_nearest.y1))
        / float(source_height)
        * AUTOMATIC_DETAIL_REFERENCE_HEIGHT
    )
    if reference_height <= AUTOMATIC_DETAIL_MAX_REFERENCE_HEIGHT:
        return "small_central_target"
    return "not_needed"


def _automatic_detail_telemetry_summary(previous, current, elapsed_seconds: float) -> str:
    elapsed = max(float(elapsed_seconds), 1e-9)

    def delta(key: str) -> int:
        return int(current[key]) - int(previous[key])
    triggered = delta("automatic_frames_triggered")
    return (
        f"DETAIL rescue {triggered / elapsed:.0f}/s | "
        f"no-target {delta('automatic_frames_triggered_no_exact_target')} | "
        f"small-center {delta('automatic_frames_triggered_small_central_target')} | "
        f"released {delta('automatic_frames_activation_released')} | "
        f"not-needed {delta('automatic_frames_skipped_not_needed')} | "
        "anchor-skip "
        f"{delta('automatic_frames_skipped_verified_anchor')}"
    )


@dataclass(slots=True)
class _TrackingPathTelemetry:
    frames_evaluated: int = 0
    frames_with_accepted_player: int = 0
    frames_blocked_self_filter: int = 0
    frames_with_direct_anchor: int = 0
    frames_with_body_fallback: int = 0

    def record(
        self,
        *,
        accepted_player: bool,
        blocked_self_filter: bool,
        direct_anchor: bool,
        body_fallback: bool,
    ) -> None:
        self.frames_evaluated += 1
        if accepted_player:
            self.frames_with_accepted_player += 1
        if blocked_self_filter:
            self.frames_blocked_self_filter += 1
        if direct_anchor:
            self.frames_with_direct_anchor += 1
        if body_fallback:
            self.frames_with_body_fallback += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "frames_evaluated": self.frames_evaluated,
            "frames_with_accepted_player": self.frames_with_accepted_player,
            "frames_blocked_self_filter": self.frames_blocked_self_filter,
            "frames_with_direct_anchor": self.frames_with_direct_anchor,
            "frames_with_body_fallback": self.frames_with_body_fallback,
        }


def _tracking_path_telemetry_summary(previous, current, elapsed_seconds: float) -> str:
    elapsed = max(float(elapsed_seconds), 1e-9)

    def delta(key: str) -> int:
        return int(current[key]) - int(previous[key])

    evaluated = max(delta("frames_evaluated"), 1)
    accepted = delta("frames_with_accepted_player")
    blocked = delta("frames_blocked_self_filter")
    anchored = delta("frames_with_direct_anchor")
    fallback = delta("frames_with_body_fallback")
    return (
        "track diag "
        f"accepted {accepted / elapsed:.1f}/s ({accepted / evaluated * 100.0:.0f}%) | "
        f"self-block {blocked / elapsed:.1f}/s ({blocked / evaluated * 100.0:.0f}%) | "
        f"direct-anchor {anchored / elapsed:.1f}/s ({anchored / evaluated * 100.0:.0f}%) | "
        f"body-fallback {fallback / elapsed:.1f}/s ({fallback / evaluated * 100.0:.0f}%)"
    )



    triggered = delta("automatic_frames_triggered")
    return (
        f"DETAIL rescue {triggered / elapsed:.0f}/s | "
        f"no-target {delta('automatic_frames_triggered_no_exact_target')} | "
        f"small-central {delta('automatic_frames_triggered_small_central_target')} | "
        f"close/off-center skip {delta('automatic_frames_skipped_not_needed')} | "
        f"released skip {delta('automatic_frames_activation_released')}"
    )


def _identity_payload(payload):
    """Transfer an already-owned prepared tensor into the latest-only worker."""

    return payload


@dataclass(frozen=True, slots=True)
class _TimestampedPreparedHeadInput:
    prepared: object
    source_timestamp_ns: int


class _PreparedDirectHeadLocalizer:
    """Adapt the pinned direct-head decoder to ``LatestHeadWorker``."""

    def __init__(
        self,
        session,
        *,
        evidence_label: str = "SunXDS 0.8.0 direct head box",
        confidence_threshold: float = 0.15,
    ) -> None:
        self._session = session
        self._evidence_label = str(evidence_label).strip() or "direct head box"
        self._confidence_threshold = float(confidence_threshold)

    def __call__(self, payload, selected_player_box):
        from detection.head_detector import (
            PreparedHeadInput,
            associate_head_to_player_outcome,
            decode_head_output,
        )
        from detection.head_worker import HeadLocalizationOutcome, HeadObservation

        if not isinstance(payload, _TimestampedPreparedHeadInput):
            raise TypeError("head worker payload must be timestamped prepared input")
        prepared = payload.prepared
        if not isinstance(prepared, PreparedHeadInput):
            raise TypeError("timestamped head payload must own PreparedHeadInput")
        output = self._session.infer(prepared.tensor)
        candidates = decode_head_output(
            output,
            prepared.transform,
            confidence=self._confidence_threshold,
        )
        association = associate_head_to_player_outcome(
            candidates,
            selected_player_box,
            source_timestamp_ns=payload.source_timestamp_ns,
        )
        localization = association.localization
        if localization is None:
            return HeadLocalizationOutcome(association.reason, None)
        if localization.source_timestamp_ns != payload.source_timestamp_ns:
            raise RuntimeError("direct-head localization changed its source timestamp")
        return HeadLocalizationOutcome(
            association.reason,
            HeadObservation(
                point=localization.point,
                confidence=localization.confidence,
                evidence=self._evidence_label,
                head_box=localization.head_box,
            ),
        )


@dataclass(frozen=True, slots=True)
class _AutomaticHeadSample:
    point: tuple[float, float]
    source_timestamp_ns: int
    direct_source_timestamp_ns: int
    identity_deadline_ns: int
    track_generation: int
    provenance: DirectHeadProvenance
    confidence: float
    evidence: str
    # Keep the measured-primary velocity coordinate separate from the feedback
    # point.  Position follows the smoothed local head anchor; velocity follows
    # only translation/scale of one frozen, verified normalized anchor through
    # measured primary geometry.  A new head-localizer offset therefore cannot
    # masquerade as whole-target motion and get amplified by feed-forward.
    velocity_point: tuple[float, float] | None = None
    bridging: bool = False
    body_derived_motion_permitted: bool = False
    body_derived_motion_deadline_ns: int | None = None
    # Optional independent motion evidence from the exact same captured frame.
    # This is never used to move ``point``; the controller may only use its
    # temporal motion as permission for bounded direct-head feed-forward.
    corroboration_point: tuple[float, float] | None = None
    # A delayed exact model result may be translated only through observed
    # newer pixels.  Persist the disposition so the next diagnostic run can
    # prove whether this path was actually active instead of inferring it from
    # tuning constants.
    phase_advanced: bool = False
    phase_hops: int = 0
    # Unlike a single phase correction, this marks a continuous physical-flow
    # run long enough to qualify the controller's measured fast-pursuit carry.
    # It never establishes or renews the direct-head identity.
    verified_flow_motion: bool = False


class _AutomaticHeadRuntime:
    """Own direct-head scheduling, identity epochs, and anchored propagation.

    The exact direct-head result is the only evidence which may establish the
    normalized location of a head.  Current primary geometry may carry that
    identity-bound location for a short, explicit interval, while the raw
    primary measurement remains the safety/identity authority.
    """

    def __init__(
        self,
        worker,
        *,
        submission_hz: float = AUTOMATIC_HEAD_LOCALIZATION_HZ,
        tracking_submission_hz: float | None = None,
        tracking_minimum_lease_remaining_seconds: float = (
            AUTOMATIC_HEAD_TRACKING_MINIMUM_LEASE_REMAINING_SECONDS
        ),
        stale_after_seconds: float = AUTOMATIC_HEAD_STALE_AFTER_SECONDS,
        provider: str | None = None,
        model_size: tuple[int, int] = (320, 320),
        model_name: str = "SunXDS 0.8.0",
        confidence_threshold: float = 0.15,
        phase_advancer=None,
    ) -> None:
        rate = float(submission_hz)
        tracking_rate = (
            None
            if tracking_submission_hz is None
            else float(tracking_submission_hz)
        )
        tracking_minimum_lease = float(
            tracking_minimum_lease_remaining_seconds
        )
        stale = float(stale_after_seconds)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("head submission_hz must be finite and positive")
        if tracking_rate is not None and (
            not math.isfinite(tracking_rate)
            or tracking_rate <= 0.0
            or tracking_rate > rate
        ):
            raise ValueError(
                "head tracking_submission_hz must be finite, positive, and "
                "no greater than submission_hz"
            )
        if (
            not math.isfinite(tracking_minimum_lease)
            or tracking_minimum_lease <= 0.0
            or tracking_minimum_lease >= DIRECT_HEAD_ANCHOR_MAX_AGE_SECONDS
        ):
            raise ValueError(
                "head tracking_minimum_lease_remaining_seconds must be "
                "finite, positive, and shorter than the direct-head lease"
            )
        if not math.isfinite(stale) or stale <= 0.0:
            raise ValueError("head stale_after_seconds must be finite and positive")
        self.worker = worker
        provider_name = None if provider is None else str(provider).strip()
        if provider is not None and not provider_name:
            raise ValueError("head provider must not be empty")
        if len(model_size) != 2:
            raise ValueError("head model_size must contain height and width")
        model_height = int(model_size[0])
        model_width = int(model_size[1])
        if model_height <= 0 or model_width <= 0:
            raise ValueError("head model_size must be positive")
        resolved_model_name = str(model_name).strip()
        if not resolved_model_name:
            raise ValueError("head model_name must not be empty")
        resolved_confidence_threshold = float(confidence_threshold)
        if not math.isfinite(resolved_confidence_threshold):
            raise ValueError("head confidence_threshold must be finite")
        self.provider = provider_name
        self.model_size = (model_height, model_width)
        self.model_name = resolved_model_name
        self.confidence_threshold = resolved_confidence_threshold
        if phase_advancer is None:
            from detection.head_flow import HeadFlowConfig, HeadFlowPhaseAdvancer

            phase_advancer = HeadFlowPhaseAdvancer(
                HeadFlowConfig(
                    min_features=3,
                    min_feature_distance=1.5,
                    feature_block_size=3,
                    roi_inset_fraction=0.01,
                    crosshair_exclusion_radius_pixels=8.0,
                    max_forward_backward_error=1.5,
                    max_inlier_residual=2.0,
                    min_feature_span_fraction=0.05,
                    max_frame_displacement_pixels=128.0,
                    max_phase_advance_seconds=(
                        AUTOMATIC_HEAD_FLOW_MAX_PHASE_ADVANCE_SECONDS
                    ),
                )
            )
        for method_name in ("remember", "advance", "clear"):
            if not callable(getattr(phase_advancer, method_name, None)):
                raise TypeError(
                    "head phase_advancer must provide remember, advance, and clear"
                )
        self.phase_advancer = phase_advancer
        self.submission_interval_ns = max(1, round(1_000_000_000 / rate))
        self.tracking_submission_interval_ns = (
            None
            if tracking_rate is None
            else max(1, round(1_000_000_000 / tracking_rate))
        )
        self.tracking_minimum_lease_remaining_ns = max(
            1,
            round(tracking_minimum_lease * 1_000_000_000),
        )
        self.stale_after_ns = max(1, round(stale * 1_000_000_000))
        self.identity_generation = 0
        self.body_valid = False
        self.next_submission_ns: int | None = None
        self._last_submission_ns: int | None = None
        self._scheduled_submission_interval_ns = self.submission_interval_ns
        self._tracking_cadence_requires_direct_refresh = False
        self.anchor = DirectHeadAnchor()
        self._visible_sample: _AutomaticHeadSample | None = None
        self._current_player_box: tuple[float, float, float, float] | None = None
        self._current_aim_box: tuple[float, float, float, float] | None = None
        self._current_player_timestamp_ns: int | None = None
        self._visible_player_box: tuple[float, float, float, float] | None = None
        self._visible_player_timestamp_ns: int | None = None
        self._observed_primary_sources: dict[
            int,
            tuple[tuple[float, float, float, float], int],
        ] = {}
        # A timestamp enters this set only after the *following* exact sample
        # confirms that an initially disjoint box continued the same bounded
        # motion.  It lets asynchronous head results traverse that proven
        # two-frame segment without making every first disjoint endpoint an
        # identity match.
        self._confirmed_disjoint_source_timestamps: set[int] = set()
        # Source frames proven to be singleton body-geometry outliers remain
        # explicitly rejected if an already-submitted async head result later
        # completes for them.  This lets the clean exact chain recover without
        # allowing the removed fragment to re-arm motion or identity.
        self._rejected_body_outlier_source_timestamps: set[int] = set()
        # Earliest binding in the current uninterrupted run of exact measured
        # primaries.  Predicted geometry and quarantined body samples break the
        # run without deleting source bindings that an already-submitted worker
        # result still needs for ordinary 2400 px/s validation.
        self._exact_measured_chain_start_ns: int | None = None
        self._current_body_observed = False
        # A normal detector-empty interval may outlast TargetTracker's short
        # prediction bridge while its logical identity memory is still live.
        # In that interval control must stay revoked, but erasing an already
        # verified direct-head lease forces an unnecessary model reacquisition
        # when the exact same tracker generation returns. Suspension keeps
        # only the immutable anchor/identity binding and last trusted body
        # geometry needed to validate that return; it never makes stale
        # geometry publishable.
        self._body_gap_suspended = False
        self._motion_corroboration_revocation_pending = False
        # One same-tracker-generation body box which is incompatible with the
        # established geometry is ambiguous: it can be a detector mode flip,
        # a one-frame localization outlier, or a nearby rival. Keep it
        # quarantined until a second compatible measurement confirms the
        # replacement. The prior head lease is not renewed while quarantined.
        self._body_update_deferred = False
        self._pending_unassociated_player_box: (
            tuple[float, float, float, float] | None
        ) = None
        self._pending_unassociated_timestamp_ns: int | None = None
        self._pending_unassociated_chain_start_ns: int | None = None
        self._pending_unassociated_exact_measured = False
        self._confirmed_disjoint_trajectory_endpoint_ns: int | None = None
        self._tracker_generation: int | None = None
        self._last_physical_source_timestamp_ns: int | None = None
        self._anchor_evidence = "filtered direct-head anchor"
        self._anchor_phase_advanced = False
        self._anchor_phase_hops = 0
        self._flow_point: tuple[float, float] | None = None
        self._flow_head_box: tuple[float, float, float, float] | None = None
        # LK needs more stable texture than a long-range head box can provide.
        # This remains a distinct same-target feature ROI: its translation is
        # applied to the immutable head point/box, but its own geometry can
        # never become an aim point.
        self._flow_feature_box: tuple[float, float, float, float] | None = None
        self._flow_source_timestamp_ns: int | None = None
        self._flow_body_center: tuple[float, float] | None = None
        self._flow_last_pixel_timestamp_ns: int | None = None
        # Coordinate freshness and evidence provenance are intentionally
        # separate. A one-frame LK rejection may still carry the last
        # pixel-qualified coordinate through the exact measured body's
        # translation without claiming that those pixels were observed.
        self._flow_coordinate_current = False
        self._flow_pixel_observed_current = False
        self._tracking_flow_success_streak = 0
        self._tracking_flow_failure_streak = 0
        self._tracking_flow_last_success_timestamp_ns: int | None = None
        self._tracking_consecutive_head_misses = 0
        self._head_crop_scale: float | None = None
        self._latest_localization_reason: str | None = None
        self._latest_localization_source_timestamp_ns: int | None = None
        self._latest_localization_track_generation: int | None = None
        self._latest_phase_frame_timestamp_ns: int | None = None
        self._capture_phase_body_timestamp_ns: int | None = None
        self._mapped_filter_point: tuple[float, float] | None = None
        self._mapped_filter_input_point: tuple[float, float] | None = None
        self._mapped_velocity_filter_point: tuple[float, float] | None = None
        self._mapped_anchor_candidate_normalized: (
            tuple[float, float] | None
        ) = None
        self._mapped_anchor_filtered_normalized: tuple[float, float] | None = None
        self._mapped_anchor_direct_timestamp_ns: int | None = None
        self._mapped_qualified_center: tuple[float, float] | None = None
        # Last exact measured geometry admitted to the mapped controller
        # channel.  Keep it across a short prediction/body gap so the first
        # returning detector box can be checked as a shape observation instead
        # of being allowed to reseed the head point unconditionally.
        self._mapped_reference_box: tuple[float, float, float, float] | None = None
        self._mapped_velocity_translation_point: tuple[float, float] | None = None
        self._mapped_velocity_reconcile_point: tuple[float, float] | None = None
        self._mapped_filter_timestamp_ns: int | None = None

    def start(self) -> None:
        self.worker.start()

    def advance_identity(self) -> None:
        """Invalidate every pending/result point and begin a fresh safety epoch."""

        self.identity_generation += 1
        self.body_valid = False
        self.next_submission_ns = None
        self._last_submission_ns = None
        self._scheduled_submission_interval_ns = self.submission_interval_ns
        self._tracking_cadence_requires_direct_refresh = True
        self._tracking_consecutive_head_misses = 0
        self._head_crop_scale = None
        self._latest_localization_reason = None
        self._latest_localization_source_timestamp_ns = None
        self._latest_localization_track_generation = None
        self.anchor.reset()
        self._visible_sample = None
        self._current_player_box = None
        self._current_aim_box = None
        self._current_player_timestamp_ns = None
        self._visible_player_box = None
        self._visible_player_timestamp_ns = None
        self._observed_primary_sources.clear()
        self._confirmed_disjoint_source_timestamps.clear()
        self._rejected_body_outlier_source_timestamps.clear()
        self._exact_measured_chain_start_ns = None
        self._current_body_observed = False
        self._body_gap_suspended = False
        self._motion_corroboration_revocation_pending = False
        self._body_update_deferred = False
        self._pending_unassociated_player_box = None
        self._pending_unassociated_timestamp_ns = None
        self._pending_unassociated_chain_start_ns = None
        self._pending_unassociated_exact_measured = False
        self._confirmed_disjoint_trajectory_endpoint_ns = None
        self._tracker_generation = None
        self._last_physical_source_timestamp_ns = None
        self._anchor_evidence = "filtered direct-head anchor"
        self._anchor_phase_advanced = False
        self._anchor_phase_hops = 0
        self._clear_live_flow()
        self._clear_phase_history()
        self._capture_phase_body_timestamp_ns = None
        self._reset_mapped_filter()
        self.worker.advance_identity(self.identity_generation)

    def _reset_mapped_filter(self) -> None:
        self._mapped_filter_point = None
        self._mapped_filter_input_point = None
        self._mapped_velocity_filter_point = None
        self._mapped_anchor_candidate_normalized = None
        self._mapped_anchor_filtered_normalized = None
        self._mapped_anchor_direct_timestamp_ns = None
        self._mapped_qualified_center = None
        self._mapped_reference_box = None
        self._mapped_velocity_translation_point = None
        self._mapped_velocity_reconcile_point = None
        self._mapped_filter_timestamp_ns = None

    def _clear_live_flow(self) -> None:
        self._flow_point = None
        self._flow_head_box = None
        self._flow_feature_box = None
        self._flow_source_timestamp_ns = None
        self._flow_body_center = None
        self._flow_last_pixel_timestamp_ns = None
        self._flow_coordinate_current = False
        self._flow_pixel_observed_current = False
        self._tracking_flow_success_streak = 0
        self._tracking_flow_failure_streak = 0
        self._tracking_flow_last_success_timestamp_ns = None

    def _record_tracking_flow_result(
        self,
        *,
        source_timestamp_ns: int,
        pixel_observed: bool,
    ) -> None:
        """Maintain independent pixel-authority and continuity hysteresis."""

        if not pixel_observed:
            self._tracking_flow_success_streak = 0
            self._tracking_flow_failure_streak += 1
            self._tracking_flow_last_success_timestamp_ns = None
            return
        timestamp_ns = int(source_timestamp_ns)
        self._tracking_flow_failure_streak = 0
        previous_ns = self._tracking_flow_last_success_timestamp_ns
        if previous_ns is None or timestamp_ns > previous_ns:
            self._tracking_flow_success_streak += 1
            self._tracking_flow_last_success_timestamp_ns = timestamp_ns

    def _clear_phase_history(self) -> None:
        self.phase_advancer.clear()
        self._latest_phase_frame_timestamp_ns = None
        self._capture_phase_body_timestamp_ns = None

    def _remember_phase_frame(
        self,
        frame,
        *,
        source_timestamp_ns: int,
    ) -> bool:
        """Remember a monotonic image once, tolerating a peek/read duplicate."""

        previous_ns = self._latest_phase_frame_timestamp_ns
        if previous_ns is not None and source_timestamp_ns <= previous_ns:
            return False
        self.phase_advancer.remember(
            frame,
            source_timestamp_ns=source_timestamp_ns,
            identity_generation=self.identity_generation,
        )
        self._latest_phase_frame_timestamp_ns = source_timestamp_ns
        return True

    def _advance_phase(
        self,
        head_box,
        *,
        feature_box,
        anchor_point,
        anchor_timestamp_ns: int,
    ):
        """Call the optional fallback API only for a range-qualified ROI."""

        arguments = {
            "anchor_point": anchor_point,
            "anchor_timestamp_ns": anchor_timestamp_ns,
            "identity_generation": self.identity_generation,
        }
        if feature_box is not None:
            arguments["feature_box"] = feature_box
        return self.phase_advancer.advance(head_box, **arguments)

    @staticmethod
    def _flow_body_residual_limits(
        previous_box,
        current_box,
        *,
        elapsed_ns: int,
    ) -> tuple[float, float]:
        """Return the requested and player-relative flow-corridor limits."""

        fallback = (
            AUTOMATIC_HEAD_FLOW_MAX_BODY_RESIDUAL_PIXELS,
            AUTOMATIC_HEAD_FLOW_MAX_BODY_RESIDUAL_PIXELS,
        )
        try:
            prior = tuple(float(value) for value in previous_box)
            current = tuple(float(value) for value in current_box)
            interval_ns = int(elapsed_ns)
        except (TypeError, ValueError):
            return fallback
        if (
            len(prior) != 4
            or len(current) != 4
            or interval_ns < 0
            or not all(math.isfinite(value) for value in (*prior, *current))
        ):
            return fallback
        prior_width = prior[2] - prior[0]
        prior_height = prior[3] - prior[1]
        current_width = current[2] - current[0]
        current_height = current[3] - current[1]
        if min(prior_width, prior_height, current_width, current_height) <= 0.0:
            return fallback
        elapsed_seconds = min(
            interval_ns / 1_000_000_000.0,
            AUTOMATIC_HEAD_FLOW_MAX_PHASE_ADVANCE_SECONDS,
        )
        geometry_uncertainty = min(
            AUTOMATIC_HEAD_FLOW_MAX_GEOMETRY_UNCERTAINTY_PIXELS,
            0.20
            * math.hypot(
                current_width - prior_width,
                current_height - prior_height,
            ),
        )
        requested = (
            AUTOMATIC_HEAD_FLOW_MAX_BODY_RESIDUAL_PIXELS
            + AUTOMATIC_HEAD_FLOW_RELATIVE_MOTION_PIXELS_PER_SECOND
            * elapsed_seconds
            + geometry_uncertainty
        )
        player_relative_cap = max(
            AUTOMATIC_HEAD_FLOW_SMALL_PLAYER_RESIDUAL_CAP_PIXELS,
            min(
                AUTOMATIC_HEAD_FLOW_MAX_DYNAMIC_RESIDUAL_PIXELS,
                0.12 * max(prior_height, current_height),
            ),
        )
        return requested, player_relative_cap

    @staticmethod
    def _flow_body_residual_tolerance(
        previous_box,
        current_box,
        *,
        elapsed_ns: int,
    ) -> float:
        """Return the legacy bounded circular flow corridor.

        The body detector is identity authority, but its normalized head
        mapping is not an independent pixel measurement.  Scale the old 12 px
        agreement floor only by elapsed physical time and measured box-shape
        change, then cap it relative to the current player and absolutely.
        This admits normal detector/flow phase disagreement without allowing
        an LK track to wander through the body for an entire head lease.
        """

        requested, player_relative_cap = (
            _AutomaticHeadRuntime._flow_body_residual_limits(
                previous_box,
                current_box,
                elapsed_ns=elapsed_ns,
            )
        )
        return min(requested, player_relative_cap)

    def _flow_body_residual_is_safe(
        self,
        point,
        reference_point,
        previous_box,
        current_box,
        *,
        elapsed_ns: int,
        previous_body_source_timestamp_ns: int | None,
        current_body_source_timestamp_ns: int | None,
    ) -> bool:
        """Validate flow against a narrowly directional small-player corridor.

        The circular baseline is unchanged.  Its 18 px small-player ceiling is
        widened to at most 24 px only along a body-motion direction established
        by two consecutive, exact, same-generation measured displacements.
        This is an elliptical gate, so transverse or diagonal disagreement
        cannot borrow the longitudinal allowance.
        """

        try:
            candidate = tuple(float(value) for value in point)
            reference = tuple(float(value) for value in reference_point)
            previous = tuple(float(value) for value in previous_box)
            current = tuple(float(value) for value in current_box)
            interval_ns = int(elapsed_ns)
        except (TypeError, ValueError):
            return False
        if (
            len(candidate) != 2
            or len(reference) != 2
            or len(previous) != 4
            or len(current) != 4
            or interval_ns < 0
            or not all(
                math.isfinite(value)
                for value in (*candidate, *reference, *previous, *current)
            )
        ):
            return False
        residual_x = candidate[0] - reference[0]
        residual_y = candidate[1] - reference[1]
        residual_distance = math.hypot(residual_x, residual_y)
        requested, player_relative_cap = self._flow_body_residual_limits(
            previous,
            current,
            elapsed_ns=interval_ns,
        )
        circular_tolerance = min(requested, player_relative_cap)
        if residual_distance <= circular_tolerance:
            return True

        # Keep zero-time behavior and all non-small-player behavior identical
        # to the circular diagnostic/API above.
        longitudinal_tolerance = min(
            requested,
            AUTOMATIC_HEAD_FLOW_SMALL_PLAYER_LONGITUDINAL_RESIDUAL_CAP_PIXELS,
        )
        if (
            interval_ns <= 0
            or player_relative_cap
            != AUTOMATIC_HEAD_FLOW_SMALL_PLAYER_RESIDUAL_CAP_PIXELS
            or longitudinal_tolerance <= circular_tolerance
            or isinstance(previous_body_source_timestamp_ns, bool)
            or isinstance(current_body_source_timestamp_ns, bool)
        ):
            return False
        try:
            previous_timestamp_ns = int(previous_body_source_timestamp_ns)
            current_timestamp_ns = int(current_body_source_timestamp_ns)
        except (TypeError, ValueError):
            return False
        generation = self._tracker_generation
        chain_start_ns = self._exact_measured_chain_start_ns
        if (
            generation is None
            or not self.body_valid
            or not self._current_body_observed
            or self._body_update_deferred
            or not self.anchor.active
            or self.anchor.track_generation != generation
            or self._current_player_timestamp_ns != current_timestamp_ns
            or chain_start_ns is None
            or previous_timestamp_ns < chain_start_ns
            or current_timestamp_ns <= previous_timestamp_ns
            or current_timestamp_ns - previous_timestamp_ns > self.stale_after_ns
            or self._observed_primary_sources.get(previous_timestamp_ns)
            != (previous, generation)
            or self._observed_primary_sources.get(current_timestamp_ns)
            != (current, generation)
        ):
            return False
        prior_timestamps = [
            timestamp_ns
            for timestamp_ns, (_box, binding_generation) in (
                self._observed_primary_sources.items()
            )
            if chain_start_ns <= timestamp_ns < previous_timestamp_ns
            and binding_generation == generation
        ]
        if not prior_timestamps:
            return False
        prior_timestamp_ns = max(prior_timestamps)
        if previous_timestamp_ns - prior_timestamp_ns > self.stale_after_ns:
            return False
        prior_box, _prior_generation = self._observed_primary_sources[
            prior_timestamp_ns
        ]

        def center(box) -> tuple[float, float]:
            return (
                (box[0] + box[2]) * 0.5,
                (box[1] + box[3]) * 0.5,
            )

        prior_center = center(prior_box)
        previous_center = center(previous)
        current_center = center(current)
        first_dx = previous_center[0] - prior_center[0]
        first_dy = previous_center[1] - prior_center[1]
        second_dx = current_center[0] - previous_center[0]
        second_dy = current_center[1] - previous_center[1]
        first_distance = math.hypot(first_dx, first_dy)
        second_distance = math.hypot(second_dx, second_dy)
        minimum_displacement = (
            AUTOMATIC_HEAD_FLOW_MIN_DIRECTIONAL_BODY_DISPLACEMENT_PIXELS
        )
        if (
            first_distance < minimum_displacement
            or second_distance < minimum_displacement
        ):
            return False
        direction_cosine = (
            first_dx * second_dx + first_dy * second_dy
        ) / (first_distance * second_distance)
        if (
            direction_cosine
            < AUTOMATIC_HEAD_CONFIRMED_DISJOINT_MINIMUM_DIRECTION_COSINE
        ):
            return False
        motion_x = second_dx / second_distance
        motion_y = second_dy / second_distance
        longitudinal_residual = residual_x * motion_x + residual_y * motion_y
        transverse_residual = residual_x * -motion_y + residual_y * motion_x
        transverse_tolerance = (
            AUTOMATIC_HEAD_FLOW_SMALL_PLAYER_RESIDUAL_CAP_PIXELS
        )
        ellipse_value = (
            longitudinal_residual / longitudinal_tolerance
        ) ** 2 + (transverse_residual / transverse_tolerance) ** 2
        return ellipse_value <= 1.0

    def _newer_capture_phase_is_safe(
        self,
        point: tuple[float, float],
        *,
        source_timestamp_ns: int,
    ) -> bool:
        """Bound position-only LK beyond the newest inferred body frame."""

        body_box = self._current_player_box
        body_timestamp_ns = self._current_player_timestamp_ns
        deadline_ns = self.anchor.identity_deadline_ns
        normalized = self.anchor.normalized_point
        if (
            not self.body_valid
            or not self._current_body_observed
            or self._body_update_deferred
            or body_box is None
            or body_timestamp_ns is None
            or deadline_ns is None
            or normalized is None
        ):
            return False
        lead_ns = int(source_timestamp_ns) - body_timestamp_ns
        maximum_lead_ns = round(
            AUTOMATIC_HEAD_CAPTURE_PHASE_MAX_LEAD_SECONDS * 1_000_000_000
        )
        if lead_ns <= 0 or lead_ns > maximum_lead_ns:
            return False
        if source_timestamp_ns >= deadline_ns:
            return False
        x, y = (float(value) for value in point)
        if not math.isfinite(x) or not math.isfinite(y):
            return False
        elapsed_seconds = lead_ns / 1_000_000_000.0
        maximum_translation = (
            AUTOMATIC_HEAD_MAPPED_STEP_ALLOWANCE_PIXELS
            + AUTOMATIC_HEAD_MAPPED_MAX_SPEED_PIXELS_PER_SECOND
            * elapsed_seconds
        )
        x1, y1, x2, y2 = body_box
        width = x2 - x1
        height = y2 - y1
        if width <= 0.0 or height <= 0.0:
            return False
        side_margin = width * 0.12 + maximum_translation
        top_margin = height * 0.12 + maximum_translation
        if not (
            x1 - side_margin <= x <= x2 + side_margin
            and y1 - top_margin
            <= y
            <= y1 + height * 0.48 + maximum_translation
        ):
            return False
        anchored_endpoint = (
            x1 + normalized[0] * width,
            y1 + normalized[1] * height,
        )
        return bool(
            math.dist(point, anchored_endpoint)
            <= AUTOMATIC_HEAD_FLOW_MAX_BODY_RESIDUAL_PIXELS
            + maximum_translation
        )

    def consume_motion_corroboration_revocation(self) -> bool:
        """Consume one fail-closed loss of independent motion evidence."""

        pending = self._motion_corroboration_revocation_pending
        self._motion_corroboration_revocation_pending = False
        return pending

    @property
    def body_update_deferred(self) -> bool:
        """Whether this frame's body geometry is awaiting confirmation."""

        return self._body_update_deferred

    @staticmethod
    def _player_boxes_associate(first, second) -> bool:
        """Conservatively recognize one primary body without moving its point."""

        try:
            a = tuple(float(value) for value in first)
            b = tuple(float(value) for value in second)
        except (TypeError, ValueError):
            return False
        if len(a) != 4 or len(b) != 4 or not all(
            math.isfinite(value) for value in (*a, *b)
        ):
            return False
        aw = a[2] - a[0]
        ah = a[3] - a[1]
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
            return False
        intersection_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        intersection_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        intersection = intersection_w * intersection_h
        if intersection <= 0.0:
            return False
        area_a = aw * ah
        area_b = bw * bh
        if area_a <= 0.0 or area_b <= 0.0:
            return False
        area_ratio = min(area_a, area_b) / max(area_a, area_b)
        overlap_of_smaller = intersection / min(area_a, area_b)
        center_dx = abs((a[0] + a[2]) - (b[0] + b[2])) * 0.5
        center_dy = abs((a[1] + a[3]) - (b[1] + b[3])) * 0.5
        return bool(
            area_ratio >= 0.25
            and overlap_of_smaller > 0.20
            and center_dx < max(aw, bw) * 0.80
            and center_dy < max(ah, bh) * 0.80
        )

    def _returning_mapped_geometry_is_safe(
        self,
        mapping_box,
        *,
        source_timestamp_ns: int,
    ) -> bool:
        """Reject a box-mode flip when exact geometry returns after a gap.

        Fast rigid translation is intentionally removed before measuring the
        innovation: the retained mapped head and the candidate box are both
        translated by their respective box centers.  What remains is detector
        scale/top-edge deformation, which is not evidence that the physical
        head moved.  This check is used only on the first exact measurement
        after prediction or a suspended body gap; ordinary continuous exact
        tracking keeps its established behavior.
        """

        reference_box = self._mapped_reference_box
        previous_point = self._mapped_filter_input_point
        normalized = self._mapped_anchor_filtered_normalized
        previous_timestamp_ns = self._mapped_filter_timestamp_ns
        if (
            reference_box is None
            or previous_point is None
            or normalized is None
            or previous_timestamp_ns is None
        ):
            return True
        try:
            current_box = tuple(float(value) for value in mapping_box)
            current_timestamp_ns = int(source_timestamp_ns)
        except (TypeError, ValueError):
            return False
        if (
            len(current_box) != 4
            or not all(math.isfinite(value) for value in current_box)
            or current_timestamp_ns <= previous_timestamp_ns
        ):
            return False
        previous_width = reference_box[2] - reference_box[0]
        previous_height = reference_box[3] - reference_box[1]
        current_width = current_box[2] - current_box[0]
        current_height = current_box[3] - current_box[1]
        if (
            min(
                previous_width,
                previous_height,
                current_width,
                current_height,
            )
            <= 0.0
        ):
            return False
        width_ratio = min(previous_width, current_width) / max(
            previous_width,
            current_width,
        )
        height_ratio = min(previous_height, current_height) / max(
            previous_height,
            current_height,
        )
        area_ratio = min(
            previous_width * previous_height,
            current_width * current_height,
        ) / max(
            previous_width * previous_height,
            current_width * current_height,
        )
        if width_ratio < 0.65 or height_ratio < 0.65 or area_ratio < 0.45:
            return False

        previous_center = (
            (reference_box[0] + reference_box[2]) * 0.5,
            (reference_box[1] + reference_box[3]) * 0.5,
        )
        current_center = (
            (current_box[0] + current_box[2]) * 0.5,
            (current_box[1] + current_box[3]) * 0.5,
        )
        translated_previous = (
            previous_point[0] + current_center[0] - previous_center[0],
            previous_point[1] + current_center[1] - previous_center[1],
        )
        candidate = (
            current_box[0] + normalized[0] * current_width,
            current_box[1] + normalized[1] * current_height,
        )
        tolerance = self._flow_body_residual_tolerance(
            reference_box,
            current_box,
            elapsed_ns=current_timestamp_ns - previous_timestamp_ns,
        )
        return math.dist(candidate, translated_previous) <= tolerance

    @staticmethod
    def _head_point_belongs_to_player(point, player_box) -> bool:
        """Require a direct point to remain in the current body's head region."""

        try:
            x, y = (float(value) for value in point)
            x1, y1, x2, y2 = (float(value) for value in player_box)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in (x, y, x1, y1, x2, y2)):
            return False
        width = x2 - x1
        height = y2 - y1
        if width <= 0.0 or height <= 0.0:
            return False
        side_margin = width * 0.12
        top_margin = height * 0.12
        return bool(
            x1 - side_margin <= x <= x2 + side_margin
            and y1 - top_margin <= y <= y1 + height * 0.48
        )

    @staticmethod
    def _flow_feature_box_for_player(
        player_box,
        head_box=None,
    ) -> tuple[float, float, float, float] | None:
        """Return a fallback feature ROI strictly inside one exact body.

        Head-only LK remains the first choice. If it lacks enough reliable
        texture, the exact primary body is already the identity authority, so
        LK may retry from the central upper 80% width and upper 44% height
        while continuing to move only the verified head point and head box.
        The fallback retains the existing forward/backward, inlier-span,
        displacement, identity, and body-residual gates. Keeping every edge
        strictly inside the exact body avoids admitting background or a
        neighboring player, including for close targets where build effects
        made the head-only stream collapse in the recorded run.
        """

        x1, y1, x2, y2 = (float(value) for value in player_box)
        width = x2 - x1
        height = y2 - y1
        if (
            not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            or width <= 0.0
            or height <= 0.0
        ):
            raise ValueError("player_box must have finite positive geometry")
        if head_box is not None:
            hx1, hy1, hx2, hy2 = (float(value) for value in head_box)
            head_width = hx2 - hx1
            head_height = hy2 - hy1
            if (
                not all(
                    math.isfinite(value)
                    for value in (hx1, hy1, hx2, hy2)
                )
                or head_width <= 0.0
                or head_height <= 0.0
            ):
                raise ValueError("head_box must have finite positive geometry")
        return (
            x1 + width * 0.10,
            y1 + height * 0.01,
            x1 + width * 0.90,
            y1 + height * 0.45,
        )

    @classmethod
    def _player_boxes_associate_over_interval(
        cls,
        submitted_box,
        current_box,
        *,
        elapsed_ns: int,
        allow_disjoint_measured_motion: bool = False,
        maximum_speed_pixels_per_second: float = 2400.0,
    ) -> bool:
        """Reject implausibly fast box transitions inside one tracker epoch.

        Ordinary callers retain the overlap requirement.  The late-result path
        may explicitly admit a disjoint endpoint only after proving that both
        boxes are exact measured-primary bindings in the same tracker epoch.
        Shape continuity and source-time displacement then keep that narrow
        fallback from turning a nearby rival or detector fragment into the
        submitted player.
        """

        if not isinstance(allow_disjoint_measured_motion, bool):
            return False
        if isinstance(maximum_speed_pixels_per_second, bool):
            return False
        try:
            configured_maximum_speed = float(
                maximum_speed_pixels_per_second
            )
        except (TypeError, ValueError):
            return False
        if (
            not math.isfinite(configured_maximum_speed)
            or configured_maximum_speed <= 0.0
        ):
            return False
        overlapping_association = cls._player_boxes_associate(
            submitted_box,
            current_box,
        )
        elapsed = int(elapsed_ns)
        if elapsed < 0:
            return False
        try:
            first = tuple(float(value) for value in submitted_box)
            second = tuple(float(value) for value in current_box)
        except (TypeError, ValueError):
            return False
        if len(first) != 4 or len(second) != 4 or not all(
            math.isfinite(value) for value in (*first, *second)
        ):
            return False
        first_width = first[2] - first[0]
        first_height = first[3] - first[1]
        second_width = second[2] - second[0]
        second_height = second[3] - second[1]
        if (
            first_width <= 0.0
            or first_height <= 0.0
            or second_width <= 0.0
            or second_height <= 0.0
        ):
            return False
        if not overlapping_association:
            if not allow_disjoint_measured_motion:
                return False
            width_ratio = min(first_width, second_width) / max(
                first_width,
                second_width,
            )
            height_ratio = min(first_height, second_height) / max(
                first_height,
                second_height,
            )
            area_ratio = min(
                first_width * first_height,
                second_width * second_height,
            ) / max(
                first_width * first_height,
                second_width * second_height,
            )
            if width_ratio < 0.65 or height_ratio < 0.65 or area_ratio < 0.45:
                return False
        reference_extent = min(
            max(first_width, second_width),
            max(first_height, second_height),
        )
        center_dx = abs((first[0] + first[2]) - (second[0] + second[2])) * 0.5
        center_dy = abs((first[1] + first[3]) - (second[1] + second[3])) * 0.5
        elapsed_seconds = elapsed / 1_000_000_000
        # Four pixels covers detector quantization on small boxes.  The 12%
        # extent allowance covers ordinary localization jitter.  Ordinary,
        # predicted, late-result, and disjoint paths retain the conservative
        # 2400 px/s default.  Only accept_body's exact same-generation measured
        # path supplies the wider camera-plus-target envelope.
        maximum_speed = max(configured_maximum_speed, reference_extent * 12.0)
        allowed_displacement = (
            4.0 + reference_extent * 0.12 + maximum_speed * elapsed_seconds
        )
        return math.hypot(center_dx, center_dy) <= allowed_displacement

    @classmethod
    def _confirmed_disjoint_measured_continuation(
        cls,
        trusted_box,
        pending_box,
        confirming_box,
        *,
        trusted_timestamp_ns: int,
        pending_timestamp_ns: int,
        confirming_timestamp_ns: int,
        maximum_chain_interval_ns: int,
    ) -> bool:
        """Prove a quarantined disjoint box continued one physical motion.

        The first disjoint measurement remains quarantined.  Only a following
        exact box which remains shape-continuous, stays inside the widened
        measured-motion envelope, and continues in the same direction can
        retain the existing anchor.  A fragment, stationary repeat,
        alternating rival, prediction, or over-speed crossing still starts a
        new identity epoch.
        """

        if (
            isinstance(maximum_chain_interval_ns, bool)
            or maximum_chain_interval_ns <= 0
            or not (
                0
                <= trusted_timestamp_ns
                < pending_timestamp_ns
                < confirming_timestamp_ns
            )
            or confirming_timestamp_ns - trusted_timestamp_ns
            > maximum_chain_interval_ns
        ):
            return False
        first_interval_ns = pending_timestamp_ns - trusted_timestamp_ns
        second_interval_ns = confirming_timestamp_ns - pending_timestamp_ns
        if not cls._player_boxes_associate_over_interval(
            trusted_box,
            pending_box,
            elapsed_ns=first_interval_ns,
            allow_disjoint_measured_motion=True,
            maximum_speed_pixels_per_second=(
                AUTOMATIC_HEAD_EXACT_BODY_ASSOCIATION_MAX_SPEED_PIXELS_PER_SECOND
            ),
        ):
            return False
        if not cls._player_boxes_associate_over_interval(
            pending_box,
            confirming_box,
            elapsed_ns=second_interval_ns,
            allow_disjoint_measured_motion=True,
            maximum_speed_pixels_per_second=(
                AUTOMATIC_HEAD_EXACT_BODY_ASSOCIATION_MAX_SPEED_PIXELS_PER_SECOND
            ),
        ):
            return False

        def center(box) -> tuple[float, float]:
            return (
                (float(box[0]) + float(box[2])) * 0.5,
                (float(box[1]) + float(box[3])) * 0.5,
            )

        trusted_center = center(trusted_box)
        pending_center = center(pending_box)
        confirming_center = center(confirming_box)
        first_dx = pending_center[0] - trusted_center[0]
        first_dy = pending_center[1] - trusted_center[1]
        second_dx = confirming_center[0] - pending_center[0]
        second_dy = confirming_center[1] - pending_center[1]
        first_distance = math.hypot(first_dx, first_dy)
        second_distance = math.hypot(second_dx, second_dy)
        if first_distance <= 0.0 or second_distance <= 0.0:
            return False
        direction_cosine = (
            first_dx * second_dx + first_dy * second_dy
        ) / (first_distance * second_distance)
        return bool(
            direction_cosine
            >= AUTOMATIC_HEAD_CONFIRMED_DISJOINT_MINIMUM_DIRECTION_COSINE
        )

    def _singleton_body_outlier_recovery_start(
        self,
        pending_box,
        confirming_box,
        *,
        pending_timestamp_ns: int,
        confirming_timestamp_ns: int,
        track_generation: int,
    ) -> int | None:
        """Return the clean-chain timestamp around one body-box outlier.

        This is intentionally narrower than the disjoint-motion grant above.
        It applies only while an immutable direct-head lease is already live,
        and only when the exact box immediately before the current trusted box
        agrees with both quarantined/confirming boxes.  In that shape, the
        current trusted box is the singleton; the target did not produce a
        coherent replacement trajectory.  A far rival, predicted sample,
        stale gap, missing predecessor, or generation change remains a hard
        identity replacement.
        """

        current_timestamp_ns = self._current_player_timestamp_ns
        deadline_ns = self.anchor.identity_deadline_ns
        direct_timestamp_ns = self.anchor.last_direct_source_timestamp_ns
        if (
            not self.body_valid
            or not self.anchor.active
            or self._visible_sample is None
            or not self._current_body_observed
            or not self._pending_unassociated_exact_measured
            or self._tracker_generation != track_generation
            or self.anchor.track_generation != track_generation
            or current_timestamp_ns is None
            or direct_timestamp_ns is None
            or direct_timestamp_ns >= current_timestamp_ns
            or (
                self._last_physical_source_timestamp_ns is not None
                and self._last_physical_source_timestamp_ns
                >= current_timestamp_ns
            )
            or deadline_ns is None
            or confirming_timestamp_ns >= deadline_ns
            or not (
                0
                <= current_timestamp_ns
                < pending_timestamp_ns
                < confirming_timestamp_ns
            )
        ):
            return None
        prior_timestamps = [
            timestamp_ns
            for timestamp_ns, (_box, generation) in (
                self._observed_primary_sources.items()
            )
            if timestamp_ns < current_timestamp_ns
            and generation == track_generation
        ]
        if not prior_timestamps:
            return None
        prior_timestamp_ns = max(prior_timestamps)
        maximum_span_ns = round(
            AUTOMATIC_HEAD_SINGLETON_BODY_OUTLIER_MAX_SPAN_SECONDS
            * 1_000_000_000
        )
        if (
            confirming_timestamp_ns - prior_timestamp_ns > maximum_span_ns
        ):
            return None
        prior_box, _generation = self._observed_primary_sources[
            prior_timestamp_ns
        ]
        direct_binding = self._observed_primary_sources.get(
            direct_timestamp_ns
        )
        if (
            direct_binding is None
            or direct_binding[1] != track_generation
            or direct_timestamp_ns >= confirming_timestamp_ns
            or confirming_timestamp_ns - direct_timestamp_ns
            > self.stale_after_ns
        ):
            return None
        direct_box = direct_binding[0]

        def ordinary_exact_step(first, second, elapsed_ns: int) -> bool:
            return bool(
                self._player_boxes_associate(first, second)
                and self._player_boxes_associate_over_interval(
                    first,
                    second,
                    elapsed_ns=elapsed_ns,
                )
            )

        if not ordinary_exact_step(
            prior_box,
            pending_box,
            pending_timestamp_ns - prior_timestamp_ns,
        ):
            return None
        if not ordinary_exact_step(
            pending_box,
            confirming_box,
            confirming_timestamp_ns - pending_timestamp_ns,
        ):
            return None
        if not ordinary_exact_step(
            prior_box,
            confirming_box,
            confirming_timestamp_ns - prior_timestamp_ns,
        ):
            return None
        if not ordinary_exact_step(
            direct_box,
            confirming_box,
            confirming_timestamp_ns - direct_timestamp_ns,
        ):
            return None
        return prior_timestamp_ns

    def _continues_confirmed_disjoint_trajectory(
        self,
        candidate_box,
        *,
        candidate_timestamp_ns: int,
        track_generation: int,
    ) -> bool:
        """Extend only the immediately preceding proven exact trajectory."""

        current_timestamp_ns = self._current_player_timestamp_ns
        current_box = self._current_player_box
        if (
            self._confirmed_disjoint_trajectory_endpoint_ns
            != current_timestamp_ns
            or current_timestamp_ns is None
            or current_box is None
            or not self._current_body_observed
            or self._tracker_generation != track_generation
            or candidate_timestamp_ns <= current_timestamp_ns
        ):
            return False
        prior_timestamps = [
            timestamp_ns
            for timestamp_ns, (_box, generation) in (
                self._observed_primary_sources.items()
            )
            if timestamp_ns < current_timestamp_ns
            and generation == track_generation
        ]
        if not prior_timestamps:
            return False
        prior_timestamp_ns = max(prior_timestamps)
        prior_box, _generation = self._observed_primary_sources[
            prior_timestamp_ns
        ]
        return self._confirmed_disjoint_measured_continuation(
            prior_box,
            current_box,
            candidate_box,
            trusted_timestamp_ns=prior_timestamp_ns,
            pending_timestamp_ns=current_timestamp_ns,
            confirming_timestamp_ns=candidate_timestamp_ns,
            maximum_chain_interval_ns=self.stale_after_ns,
        )

    def _exact_measured_boxes_associate_over_interval(
        self,
        submitted_box,
        current_box,
        *,
        submitted_timestamp_ns: int,
        current_timestamp_ns: int,
        track_generation: int,
    ) -> bool:
        """Associate two measured bindings without requiring endpoint overlap."""

        source_binding = self._observed_primary_sources.get(
            submitted_timestamp_ns
        )
        current_binding = self._observed_primary_sources.get(
            current_timestamp_ns
        )
        return bool(
            source_binding is not None
            and current_binding is not None
            and source_binding == (tuple(submitted_box), track_generation)
            and current_binding == (tuple(current_box), track_generation)
            and self._player_boxes_associate_over_interval(
                submitted_box,
                current_box,
                elapsed_ns=current_timestamp_ns - submitted_timestamp_ns,
                allow_disjoint_measured_motion=True,
            )
        )

    def _exact_measured_binding_chain_associates(
        self,
        submitted_box,
        current_box,
        *,
        submitted_timestamp_ns: int,
        current_timestamp_ns: int,
        track_generation: int,
    ) -> bool:
        """Follow only an uninterrupted live chain of overlapping measurements.

        An asynchronous head result can arrive one primary frame after its crop
        was submitted.  The generic endpoint check intentionally stays at 2400
        px/s, but that is lower than the observed sum of camera and target
        motion.  Permit the 4800 px/s envelope only when every exact primary
        binding from the result source through the current frame proves the
        same generation and every adjacent pair remains overlapping.  A
        prediction, quarantine, disjoint step, or generation edge breaks this
        proof rather than widening an endpoint transfer.
        """

        if (
            isinstance(submitted_timestamp_ns, bool)
            or isinstance(current_timestamp_ns, bool)
            or isinstance(track_generation, bool)
        ):
            return False
        try:
            submitted_ns = int(submitted_timestamp_ns)
            current_ns = int(current_timestamp_ns)
            generation = int(track_generation)
            submitted = tuple(float(value) for value in submitted_box)
            current = tuple(float(value) for value in current_box)
        except (TypeError, ValueError):
            return False
        chain_start_ns = self._exact_measured_chain_start_ns
        if (
            submitted_ns < 0
            or current_ns <= submitted_ns
            or current_ns - submitted_ns > self.stale_after_ns
            or generation < 0
            or len(submitted) != 4
            or len(current) != 4
            or not all(math.isfinite(value) for value in (*submitted, *current))
            or chain_start_ns is None
            or submitted_ns < chain_start_ns
        ):
            return False
        source_binding = self._observed_primary_sources.get(submitted_ns)
        current_binding = self._observed_primary_sources.get(current_ns)
        if (
            source_binding != (submitted, generation)
            or current_binding != (current, generation)
        ):
            return False
        chain = sorted(
            (
                timestamp_ns,
                binding,
            )
            for timestamp_ns, binding in self._observed_primary_sources.items()
            if submitted_ns <= timestamp_ns <= current_ns
        )
        if (
            len(chain) < 2
            or chain[0][0] != submitted_ns
            or chain[-1][0] != current_ns
        ):
            return False
        for (previous_ns, previous_binding), (next_ns, next_binding) in zip(
            chain,
            chain[1:],
        ):
            previous_box, previous_generation = previous_binding
            next_box, next_generation = next_binding
            if (
                next_ns <= previous_ns
                or previous_generation != generation
                or next_generation != generation
                or not self._player_boxes_associate_over_interval(
                    previous_box,
                    next_box,
                    elapsed_ns=next_ns - previous_ns,
                    allow_disjoint_measured_motion=(
                        next_ns in self._confirmed_disjoint_source_timestamps
                    ),
                    maximum_speed_pixels_per_second=(
                        AUTOMATIC_HEAD_EXACT_BODY_ASSOCIATION_MAX_SPEED_PIXELS_PER_SECOND
                    ),
                )
            ):
                return False
        return True

    def _late_result_boxes_associate_over_interval(
        self,
        submitted_box,
        current_box,
        *,
        submitted_timestamp_ns: int,
        current_timestamp_ns: int,
        track_generation: int,
    ) -> bool:
        """Preserve ordinary overlap; narrowly extend fast measured motion."""

        elapsed_ns = current_timestamp_ns - submitted_timestamp_ns
        return bool(
            self._player_boxes_associate_over_interval(
                submitted_box,
                current_box,
                elapsed_ns=elapsed_ns,
            )
            or self._exact_measured_binding_chain_associates(
                submitted_box,
                current_box,
                submitted_timestamp_ns=submitted_timestamp_ns,
                current_timestamp_ns=current_timestamp_ns,
                track_generation=track_generation,
            )
            or self._exact_measured_boxes_associate_over_interval(
                submitted_box,
                current_box,
                submitted_timestamp_ns=submitted_timestamp_ns,
                current_timestamp_ns=current_timestamp_ns,
                track_generation=track_generation,
            )
        )

    def accept_body(
        self,
        player_box=None,
        *,
        aim_box=None,
        corroboration_box=None,
        track_generation: int | None = None,
        source_timestamp_ns: int | None = None,
    ) -> bool:
        """Accept live primary evidence; return true on identity replacement."""

        self._body_update_deferred = False
        replacement = False
        body_timestamp_ns = (
            None if source_timestamp_ns is None else int(source_timestamp_ns)
        )
        if body_timestamp_ns is not None and body_timestamp_ns < 0:
            raise ValueError("body source timestamp cannot be negative")
        mapped_geometry: tuple[float, float, float, float] | None = None
        if aim_box is not None:
            mapped_geometry = tuple(float(value) for value in aim_box)
            if len(mapped_geometry) != 4 or not all(
                math.isfinite(value) for value in mapped_geometry
            ):
                raise ValueError("aim_box must contain four finite coordinates")
            if (
                mapped_geometry[2] <= mapped_geometry[0]
                or mapped_geometry[3] <= mapped_geometry[1]
            ):
                raise ValueError("aim_box must have positive width and height")
            if player_box is None:
                raise ValueError("aim_box requires an accepted player_box")
        corroboration: tuple[float, float, float, float] | None = None
        if corroboration_box is not None:
            corroboration = tuple(float(value) for value in corroboration_box)
            if len(corroboration) != 4 or not all(
                math.isfinite(value) for value in corroboration
            ):
                raise ValueError(
                    "corroboration_box must contain four finite coordinates"
                )
            if (
                corroboration[2] <= corroboration[0]
                or corroboration[3] <= corroboration[1]
            ):
                raise ValueError(
                    "corroboration_box must have positive width and height"
                )
            if body_timestamp_ns is None:
                raise ValueError("corroboration_box requires a source timestamp")
            if player_box is None:
                raise ValueError("corroboration_box requires an accepted player_box")
        if track_generation is not None:
            if (
                isinstance(track_generation, bool)
                or not isinstance(track_generation, int)
                or track_generation < 0
            ):
                raise ValueError("track_generation must be a non-negative integer")
            if (
                self._tracker_generation is not None
                and track_generation != self._tracker_generation
            ):
                self.advance_identity()
                replacement = True
        if player_box is not None:
            candidate = tuple(float(value) for value in player_box)
            if len(candidate) != 4 or not all(
                math.isfinite(value) for value in candidate
            ):
                raise ValueError("player_box must contain four finite coordinates")
            if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
                raise ValueError("player_box must have positive width and height")
            if corroboration is not None and corroboration != candidate:
                raise ValueError(
                    "corroboration_box must equal the accepted player_box"
                )
            current_interval_ns = (
                None
                if body_timestamp_ns is None
                or self._current_player_timestamp_ns is None
                else body_timestamp_ns - self._current_player_timestamp_ns
            )
            returning_to_exact_measurement = bool(
                corroboration is not None
                and body_timestamp_ns is not None
                and (
                    self._body_gap_suspended
                    or (self.body_valid and not self._current_body_observed)
                )
            )
            returning_mapping_box = (
                candidate if mapped_geometry is None else mapped_geometry
            )
            returning_geometry_safe = bool(
                not returning_to_exact_measurement
                or self._returning_mapped_geometry_is_safe(
                    returning_mapping_box,
                    source_timestamp_ns=body_timestamp_ns,
                )
            )
            ordinary_body_association = bool(
                returning_geometry_safe
                and self._current_player_box is not None
                and self._player_boxes_associate(
                    self._current_player_box,
                    candidate,
                )
                and (
                    current_interval_ns is None
                    or self._player_boxes_associate_over_interval(
                        self._current_player_box,
                        candidate,
                        elapsed_ns=current_interval_ns,
                    )
                )
            )
            # TargetTracker already proved these are consecutive exact raw
            # measurements of one logical target.  For overlapping boxes only,
            # widen the source-time speed envelope inside that same generation
            # and live window.  The old 2400 px/s radial check was clearing a
            # valid head anchor during fast camera-plus-target motion in the
            # recorded run; disjoint and predicted geometry stay conservative.
            same_generation_measured_motion = bool(
                returning_geometry_safe
                and self.body_valid
                and self._current_player_box is not None
                and self._player_boxes_associate(
                    self._current_player_box,
                    candidate,
                )
                and corroboration is not None
                and self._current_body_observed
                and track_generation is not None
                and self._tracker_generation == track_generation
                and current_interval_ns is not None
                and 0 < current_interval_ns <= self.stale_after_ns
                and self._player_boxes_associate_over_interval(
                    self._current_player_box,
                    candidate,
                    elapsed_ns=current_interval_ns,
                    maximum_speed_pixels_per_second=(
                        AUTOMATIC_HEAD_EXACT_BODY_ASSOCIATION_MAX_SPEED_PIXELS_PER_SECOND
                    ),
                )
            )
            confirmed_trajectory_continues = bool(
                returning_geometry_safe
                and not ordinary_body_association
                and not same_generation_measured_motion
                and corroboration is not None
                and track_generation is not None
                and body_timestamp_ns is not None
                and self._continues_confirmed_disjoint_trajectory(
                    candidate,
                    candidate_timestamp_ns=body_timestamp_ns,
                    track_generation=track_generation,
                )
            )
            exact_chain_continues = bool(
                corroboration is not None
                and self._current_body_observed
                and self._exact_measured_chain_start_ns is not None
                and track_generation is not None
                and self._tracker_generation == track_generation
                and body_timestamp_ns is not None
                and self._current_player_timestamp_ns is not None
                and body_timestamp_ns > self._current_player_timestamp_ns
                and body_timestamp_ns - self._current_player_timestamp_ns
                <= self.stale_after_ns
                and (
                    ordinary_body_association
                    or same_generation_measured_motion
                    or confirmed_trajectory_continues
                )
            )
            if confirmed_trajectory_continues:
                assert body_timestamp_ns is not None
                self._confirmed_disjoint_source_timestamps.add(
                    body_timestamp_ns
                )
                self._confirmed_disjoint_trajectory_endpoint_ns = (
                    body_timestamp_ns
                )
            elif ordinary_body_association or same_generation_measured_motion:
                # Ordinary overlap ends the special trajectory grant.  Its
                # historical edge markers remain available solely so an
                # already-submitted asynchronous head result can traverse the
                # exact measured binding chain.
                self._confirmed_disjoint_trajectory_endpoint_ns = None
            if (
                (self.body_valid or self._body_gap_suspended)
                and self._current_player_box is not None
                and not ordinary_body_association
                and not same_generation_measured_motion
                and not confirmed_trajectory_continues
            ):
                pending_box = self._pending_unassociated_player_box
                pending_timestamp_ns = self._pending_unassociated_timestamp_ns
                pending_matches = bool(
                    pending_box is not None
                    and (
                        (
                            self._player_boxes_associate(pending_box, candidate)
                            and (
                                body_timestamp_ns is None
                                or pending_timestamp_ns is None
                                or self._player_boxes_associate_over_interval(
                                    pending_box,
                                    candidate,
                                    elapsed_ns=(
                                        body_timestamp_ns - pending_timestamp_ns
                                    ),
                                )
                            )
                        )
                        or (
                            corroboration is not None
                            and self._current_body_observed
                            and track_generation is not None
                            and self._tracker_generation == track_generation
                            and body_timestamp_ns is not None
                            and pending_timestamp_ns is not None
                            and self._player_boxes_associate_over_interval(
                                pending_box,
                                candidate,
                                elapsed_ns=(
                                    body_timestamp_ns - pending_timestamp_ns
                                ),
                                allow_disjoint_measured_motion=True,
                                maximum_speed_pixels_per_second=(
                                    AUTOMATIC_HEAD_EXACT_BODY_ASSOCIATION_MAX_SPEED_PIXELS_PER_SECOND
                                ),
                            )
                        )
                    )
                )
                if pending_matches:
                    confirmed_disjoint_motion = bool(
                        corroboration is not None
                        and self._pending_unassociated_exact_measured
                        and self._current_body_observed
                        and track_generation is not None
                        and self._tracker_generation == track_generation
                        and self._current_player_timestamp_ns is not None
                        and pending_timestamp_ns is not None
                        and body_timestamp_ns is not None
                        and self._confirmed_disjoint_measured_continuation(
                            self._current_player_box,
                            pending_box,
                            candidate,
                            trusted_timestamp_ns=(
                                self._current_player_timestamp_ns
                            ),
                            pending_timestamp_ns=pending_timestamp_ns,
                            confirming_timestamp_ns=body_timestamp_ns,
                            maximum_chain_interval_ns=self.stale_after_ns,
                        )
                    )
                    singleton_outlier_recovery_start_ns = (
                        None
                        if (
                            corroboration is None
                            or track_generation is None
                            or pending_box is None
                            or pending_timestamp_ns is None
                            or body_timestamp_ns is None
                        )
                        else self._singleton_body_outlier_recovery_start(
                            pending_box,
                            candidate,
                            pending_timestamp_ns=pending_timestamp_ns,
                            confirming_timestamp_ns=body_timestamp_ns,
                            track_generation=track_generation,
                        )
                    )
                    if confirmed_disjoint_motion:
                        # The first disjoint sample was never published.  A
                        # second exact sample now proves continued bounded
                        # motion in this same TargetTracker generation, so keep
                        # the immutable direct-head anchor and restore the exact
                        # binding chain for already-submitted async results.
                        assert track_generation is not None
                        assert pending_timestamp_ns is not None
                        assert self._current_player_timestamp_ns is not None
                        self._observed_primary_sources[pending_timestamp_ns] = (
                            pending_box,
                            track_generation,
                        )
                        self._confirmed_disjoint_source_timestamps.add(
                            pending_timestamp_ns
                        )
                        assert body_timestamp_ns is not None
                        self._confirmed_disjoint_source_timestamps.add(
                            body_timestamp_ns
                        )
                        self._confirmed_disjoint_trajectory_endpoint_ns = (
                            body_timestamp_ns
                        )
                        self._exact_measured_chain_start_ns = (
                            self._pending_unassociated_chain_start_ns
                            if self._pending_unassociated_chain_start_ns is not None
                            else self._current_player_timestamp_ns
                        )
                        exact_chain_continues = True
                        self._pending_unassociated_player_box = None
                        self._pending_unassociated_timestamp_ns = None
                        self._pending_unassociated_chain_start_ns = None
                        self._pending_unassociated_exact_measured = False
                    elif singleton_outlier_recovery_start_ns is not None:
                        # The recorded dropout was a body-box mode oscillation:
                        # clean box -> one accepted fragment -> two clean boxes,
                        # all in the same TargetTracker generation while the
                        # direct-head lease remained live.  Remove only that
                        # singleton source binding, restore the clean measured
                        # chain, and keep the immutable head identity. Motion
                        # authority was already revoked on the quarantined
                        # sample and must be re-earned normally.
                        assert track_generation is not None
                        assert pending_box is not None
                        assert pending_timestamp_ns is not None
                        assert body_timestamp_ns is not None
                        outlier_timestamp_ns = self._current_player_timestamp_ns
                        assert outlier_timestamp_ns is not None
                        self._rejected_body_outlier_source_timestamps.add(
                            outlier_timestamp_ns
                        )
                        self._observed_primary_sources.pop(
                            outlier_timestamp_ns,
                            None,
                        )
                        self._observed_primary_sources[pending_timestamp_ns] = (
                            pending_box,
                            track_generation,
                        )
                        self._confirmed_disjoint_source_timestamps.discard(
                            outlier_timestamp_ns
                        )
                        self._confirmed_disjoint_trajectory_endpoint_ns = None
                        self._exact_measured_chain_start_ns = (
                            self._pending_unassociated_chain_start_ns
                            if self._pending_unassociated_chain_start_ns
                            is not None
                            else singleton_outlier_recovery_start_ns
                        )
                        exact_chain_continues = True
                        self._pending_unassociated_player_box = None
                        self._pending_unassociated_timestamp_ns = None
                        self._pending_unassociated_chain_start_ns = None
                        self._pending_unassociated_exact_measured = False
                        self._reset_mapped_filter()
                    else:
                        # Two compatible boxes which do not prove a bounded,
                        # same-direction exact continuation are a replacement.
                        # Begin a fresh epoch before accepting their geometry.
                        self.advance_identity()
                        replacement = True
                else:
                    # A single incompatible sample cannot erase a verified
                    # head lease or become crop/control geometry. Preserve the
                    # last trusted body without renewing any timestamp, revoke
                    # predictive motion immediately, and let the controller's
                    # existing numeric/identity deadlines bound the pause.
                    self._pending_unassociated_player_box = candidate
                    self._pending_unassociated_timestamp_ns = body_timestamp_ns
                    self._pending_unassociated_exact_measured = bool(
                        corroboration is not None
                        and track_generation is not None
                        and self._tracker_generation == track_generation
                        and body_timestamp_ns is not None
                    )
                    self._confirmed_disjoint_trajectory_endpoint_ns = None
                    if self._pending_unassociated_chain_start_ns is None:
                        self._pending_unassociated_chain_start_ns = (
                            self._exact_measured_chain_start_ns
                        )
                    self._body_update_deferred = True
                    self._motion_corroboration_revocation_pending = True
                    self._exact_measured_chain_start_ns = None
                    self._clear_live_flow()
                    self._clear_phase_history()
                    return False
            else:
                # Returning to geometry compatible with the established body
                # proves the quarantined sample was transient.
                self._pending_unassociated_player_box = None
                self._pending_unassociated_timestamp_ns = None
                self._pending_unassociated_chain_start_ns = None
                self._pending_unassociated_exact_measured = False
            self._current_player_box = candidate
            self._current_aim_box = (
                candidate if mapped_geometry is None else mapped_geometry
            )
            self._current_player_timestamp_ns = body_timestamp_ns
            self._current_body_observed = corroboration is not None
            if corroboration is None:
                # Predicted primary geometry may be rendered as a truthful
                # bridge, but it can neither publish a physical measurement nor
                # authorize predictive feed-forward or a new direct-head crop.
                self._motion_corroboration_revocation_pending = True
                self._clear_live_flow()
                self._clear_phase_history()
            if corroboration is not None:
                assert body_timestamp_ns is not None
                effective_generation = (
                    track_generation
                    if track_generation is not None
                    else self._tracker_generation
                    if self._tracker_generation is not None
                    else self.identity_generation
                )
                self._observed_primary_sources[body_timestamp_ns] = (
                    candidate,
                    effective_generation,
                )
                cutoff_ns = body_timestamp_ns - self.stale_after_ns * 2
                self._observed_primary_sources = {
                    timestamp: binding
                    for timestamp, binding in self._observed_primary_sources.items()
                    if timestamp >= cutoff_ns
                }
                self._confirmed_disjoint_source_timestamps.intersection_update(
                    self._observed_primary_sources
                )
                self._rejected_body_outlier_source_timestamps = {
                    timestamp_ns
                    for timestamp_ns in (
                        self._rejected_body_outlier_source_timestamps
                    )
                    if timestamp_ns >= cutoff_ns
                }
                while len(self._observed_primary_sources) > 32:
                    oldest_timestamp = min(self._observed_primary_sources)
                    del self._observed_primary_sources[oldest_timestamp]
                    self._confirmed_disjoint_source_timestamps.discard(
                        oldest_timestamp
                    )
                if not exact_chain_continues:
                    self._exact_measured_chain_start_ns = (
                        body_timestamp_ns
                        if track_generation is not None
                        else None
                    )
            else:
                self._exact_measured_chain_start_ns = None
        if track_generation is not None:
            self._tracker_generation = track_generation
        elif self._tracker_generation is None:
            # Tests and non-tracker callers still receive an explicit logical
            # generation; the live path always supplies TargetTracker's value.
            self._tracker_generation = self.identity_generation
        self.body_valid = True
        self._body_gap_suspended = False
        return replacement

    def suspend_body_gap(self, *, now_ns: int) -> bool:
        """Pause output across an ordinary missing-primary identity gap.

        Unlike :meth:`revoke_body`, this does not begin a new worker identity
        epoch or delete the direct-head anchor. It is valid only for the live
        tracker's ordinary detector-empty/reacquisition interval, where the
        track generation remains the identity authority. No coordinate is
        visible while suspended. A returning measured primary must carry the
        exact same generation and pass the normal body association checks;
        otherwise :meth:`accept_body` advances identity before accepting it.
        The direct observation's original deadline is never renewed. Return
        true while an existing or newly entered suspension remains valid.
        """

        current_ns = int(now_ns)
        if current_ns < 0:
            raise ValueError("body-gap suspension timestamp cannot be negative")
        anchor_deadline_ns = self.anchor.identity_deadline_ns
        live_anchor = bool(
            self.anchor.active
            and anchor_deadline_ns is not None
            and current_ns < anchor_deadline_ns
            and self.anchor.track_generation == self._tracker_generation
        )
        if self._body_gap_suspended:
            return live_anchor
        if not self.body_valid or not live_anchor:
            return False
        self.body_valid = False
        self._body_gap_suspended = True
        self.next_submission_ns = None
        self._last_submission_ns = None
        self._scheduled_submission_interval_ns = self.submission_interval_ns
        self._tracking_cadence_requires_direct_refresh = True
        self._visible_sample = None
        self._current_body_observed = False
        self._body_update_deferred = False
        self._pending_unassociated_player_box = None
        self._pending_unassociated_timestamp_ns = None
        self._pending_unassociated_chain_start_ns = None
        self._pending_unassociated_exact_measured = False
        self._confirmed_disjoint_trajectory_endpoint_ns = None
        self._exact_measured_chain_start_ns = None
        self._motion_corroboration_revocation_pending = True
        self._clear_live_flow()
        self._clear_phase_history()
        self._capture_phase_body_timestamp_ns = None
        # Keep the private causal map across this output-disabled interval.
        # The first same-generation exact return is shape-checked against it
        # and then slew-limited normally. Identity advance, expiry, and an
        # over-stale interval still reset/reseed it explicitly.
        return True

    @property
    def body_gap_suspended(self) -> bool:
        return self._body_gap_suspended

    def revoke_body(self) -> bool:
        """Hard-revoke an unsafe body epoch; return whether state changed."""

        if not self.body_valid and not self._body_gap_suspended:
            return False
        self.advance_identity()
        return True

    def submit(self, frame, selected_player, *, source_timestamp_ns: int) -> bool:
        """Prepare one bounded crop from an exact same-frame primary box."""

        if (
            not self.body_valid
            or not self._current_body_observed
            or self._body_update_deferred
        ):
            return False
        timestamp = int(source_timestamp_ns)
        if timestamp < 0:
            raise ValueError("head source timestamp cannot be negative")
        binding = self._observed_primary_sources.get(timestamp)
        if binding is None:
            return False
        bound_box, _bound_generation = binding
        selected_player_box = tuple(float(value) for value in selected_player.box)
        if (
            self._current_player_timestamp_ns != timestamp
            or self._current_player_box != bound_box
            or selected_player_box != bound_box
        ):
            return False
        interval_ns = self._submission_interval_ns(timestamp)
        if (
            interval_ns == self.submission_interval_ns
            and self._scheduled_submission_interval_ns
            != self.submission_interval_ns
        ):
            # An explicit stale/repeated-model-miss/near-expiry recovery cannot
            # bounce back to maintenance merely because its old anchor remains
            # mapped. Require one fresh direct result to correct drift first.
            self._tracking_cadence_requires_direct_refresh = True
        if interval_ns != self._scheduled_submission_interval_ns:
            previous_interval_ns = self._scheduled_submission_interval_ns
            self._scheduled_submission_interval_ns = interval_ns
            last_submission_ns = self._last_submission_ns
            if last_submission_ns is None:
                self.next_submission_ns = None
            elif interval_ns < previous_interval_ns:
                recovery_deadline_ns = last_submission_ns + interval_ns
                self.next_submission_ns = (
                    recovery_deadline_ns
                    if self.next_submission_ns is None
                    else min(self.next_submission_ns, recovery_deadline_ns)
                )
            else:
                self.next_submission_ns = last_submission_ns + interval_ns
        deadline = self.next_submission_ns
        if deadline is not None and timestamp < deadline:
            return False

        from detection.head_detector import (
            adaptive_head_crop_scale,
            plan_head_crop,
            prepare_head_input,
        )

        self._head_crop_scale = adaptive_head_crop_scale(
            frame.shape,
            selected_player_box,
            previous_crop_scale=self._head_crop_scale,
        )

        transform = plan_head_crop(
            frame.shape,
            selected_player_box,
            crop_scale=self._head_crop_scale,
            model_size=self.model_size,
        )
        prepared = prepare_head_input(frame, transform)
        payload = _TimestampedPreparedHeadInput(prepared, timestamp)
        accepted = self.worker.submit(
            payload,
            source_timestamp_ns=timestamp,
            identity_generation=self.identity_generation,
            selected_player_box=selected_player_box,
        )
        if accepted:
            self._last_submission_ns = timestamp
            if deadline is None:
                self.next_submission_ns = timestamp + interval_ns
            else:
                elapsed_intervals = (
                    max(0, timestamp - deadline) // interval_ns
                ) + 1
                self.next_submission_ns = (
                    deadline + elapsed_intervals * interval_ns
                )
        return bool(accepted)

    def _submission_interval_ns(self, source_timestamp_ns: int) -> int:
        """Use slow model maintenance while a measured head lease is healthy.

        The 640 head worker can complete only about 24--35 requests/second on
        the shared GPU. Submitting at 90 Hz after a direct anchor exists cannot
        make that coordinate newer; it starves the primary detector and makes
        the frame-to-frame LK gaps larger. Keep 90 Hz for acquisition, stale
        recovery, repeated head-model misses, and near-expiry refresh. During
        an ordinary LK miss the exact body may carry position without claiming
        pixel evidence, while the anchored model stays at maintenance cadence.
        """

        tracking_interval_ns = self.tracking_submission_interval_ns
        if tracking_interval_ns is None:
            return self.submission_interval_ns
        deadline_ns = self.anchor.identity_deadline_ns
        visible = self._visible_sample
        if (
            not self.body_valid
            or not self._current_body_observed
            or self._body_update_deferred
            or visible is None
            or visible.bridging
            or self._tracking_cadence_requires_direct_refresh
            or deadline_ns is None
            or deadline_ns - source_timestamp_ns
            < self.tracking_minimum_lease_remaining_ns
            or self._tracker_generation is None
            or self.anchor.track_generation != self._tracker_generation
            or self._current_player_timestamp_ns != source_timestamp_ns
        ):
            return self.submission_interval_ns
        return tracking_interval_ns

    def remember_frame(self, frame, *, source_timestamp_ns: int) -> bool:
        """Retain one exact measured-primary frame for delayed-head replay.

        This does not create an observation or renew an identity deadline.  It
        merely makes the already accepted source pixels available if a direct
        model result for that same timestamp completes several frames later.
        """

        timestamp = int(source_timestamp_ns)
        if timestamp < 0:
            raise ValueError("head history timestamp cannot be negative")
        if (
            not self.body_valid
            or not self._current_body_observed
            or self._body_update_deferred
            or self._current_player_timestamp_ns != timestamp
        ):
            return False
        frame_was_new = self._remember_phase_frame(
            frame,
            source_timestamp_ns=timestamp,
        )
        flow_point = self._flow_point
        flow_box = self._flow_head_box
        flow_feature_box = self._flow_feature_box
        flow_timestamp_ns = self._flow_source_timestamp_ns
        flow_body_center = self._flow_body_center
        flow_last_pixel_timestamp_ns = self._flow_last_pixel_timestamp_ns
        current_box = self._current_player_box
        assert current_box is not None
        current_center = (
            (current_box[0] + current_box[2]) * 0.5,
            (current_box[1] + current_box[3]) * 0.5,
        )
        previous_body_box = (
            current_box
            if self._visible_player_box is None
            else self._visible_player_box
        )
        anchor_normalized = self.anchor.normalized_point
        anchor_endpoint = (
            None
            if anchor_normalized is None
            else (
                current_box[0]
                + anchor_normalized[0] * (current_box[2] - current_box[0]),
                current_box[1]
                + anchor_normalized[1] * (current_box[3] - current_box[1]),
            )
        )
        if (
            not frame_was_new
            and self._latest_phase_frame_timestamp_ns != timestamp
        ):
            return False
        if not frame_was_new and flow_timestamp_ns == timestamp:
            # The post-inference latest-frame tap may have observed this exact
            # packet before the normal consumer receives it on the following
            # loop.  Validate that position against the newly inferred body,
            # but never run a zero-time LK step or republish different geometry
            # at the same source timestamp.
            capture_phase_root_ns = self._capture_phase_body_timestamp_ns
            duplicate_elapsed_ns = (
                0
                if capture_phase_root_ns is None
                else max(0, timestamp - capture_phase_root_ns)
            )
            if (
                flow_point is not None
                and flow_box is not None
                and flow_timestamp_ns == timestamp
                and anchor_endpoint is not None
                and self._head_point_belongs_to_player(flow_point, current_box)
                and self._flow_body_residual_is_safe(
                    flow_point,
                    anchor_endpoint,
                    previous_body_box,
                    current_box,
                    elapsed_ns=duplicate_elapsed_ns,
                    previous_body_source_timestamp_ns=(
                        capture_phase_root_ns
                    ),
                    current_body_source_timestamp_ns=timestamp,
                )
            ):
                self._flow_body_center = current_center
                self._flow_feature_box = self._flow_feature_box_for_player(
                    current_box,
                    flow_box,
                )
                self._flow_coordinate_current = True
                self._flow_pixel_observed_current = True
                self._record_tracking_flow_result(
                    source_timestamp_ns=timestamp,
                    pixel_observed=True,
                )
                return True
            self._clear_live_flow()
            return False
        self._flow_coordinate_current = False
        self._flow_pixel_observed_current = False
        if (
            flow_point is not None
            and flow_box is not None
            and flow_timestamp_ns is not None
            and flow_body_center is not None
            and flow_timestamp_ns < timestamp
        ):
            advanced = self._advance_phase(
                flow_box,
                feature_box=flow_feature_box,
                anchor_point=flow_point,
                anchor_timestamp_ns=flow_timestamp_ns,
            )
            expected = (
                flow_point[0] + current_center[0] - flow_body_center[0],
                flow_point[1] + current_center[1] - flow_body_center[1],
            )
            if (
                advanced is not None
                and advanced.source_timestamp_ns == timestamp
                and self._head_point_belongs_to_player(
                    advanced.point,
                    current_box,
                )
                and self._flow_body_residual_is_safe(
                    advanced.point,
                    expected,
                    previous_body_box,
                    current_box,
                    elapsed_ns=timestamp - flow_timestamp_ns,
                    previous_body_source_timestamp_ns=flow_timestamp_ns,
                    current_body_source_timestamp_ns=timestamp,
                )
                and anchor_endpoint is not None
                and self._flow_body_residual_is_safe(
                    advanced.point,
                    anchor_endpoint,
                    previous_body_box,
                    current_box,
                    elapsed_ns=timestamp - flow_timestamp_ns,
                    previous_body_source_timestamp_ns=flow_timestamp_ns,
                    current_body_source_timestamp_ns=timestamp,
                )
            ):
                self._flow_point = advanced.point
                self._flow_head_box = advanced.head_box
                self._flow_feature_box = self._flow_feature_box_for_player(
                    current_box,
                    advanced.head_box,
                )
                self._flow_source_timestamp_ns = timestamp
                self._flow_body_center = current_center
                self._flow_last_pixel_timestamp_ns = timestamp
                self._flow_coordinate_current = True
                self._flow_pixel_observed_current = True
                self._record_tracking_flow_result(
                    source_timestamp_ns=timestamp,
                    pixel_observed=True,
                )
                self._anchor_phase_advanced = True
                self._anchor_phase_hops = advanced.frames_spanned
            elif (
                anchor_endpoint is not None
                and self._tracking_flow_failure_streak
                < AUTOMATIC_HEAD_FLOW_MAX_CONSECUTIVE_FAILURES
                and flow_last_pixel_timestamp_ns is not None
                and timestamp - flow_last_pixel_timestamp_ns
                <= round(
                    AUTOMATIC_HEAD_FLOW_MAX_BODY_CARRY_SECONDS
                    * 1_000_000_000
                )
                and timestamp - flow_timestamp_ns
                <= round(
                    AUTOMATIC_HEAD_FLOW_MAX_PHASE_ADVANCE_SECONDS
                    * 1_000_000_000
                )
                and self._head_point_belongs_to_player(expected, current_box)
                and self._flow_body_residual_is_safe(
                    expected,
                    anchor_endpoint,
                    previous_body_box,
                    current_box,
                    elapsed_ns=timestamp - flow_timestamp_ns,
                    previous_body_source_timestamp_ns=flow_timestamp_ns,
                    current_body_source_timestamp_ns=timestamp,
                )
            ):
                # Preserve the LK coordinate system privately across a bounded
                # rejection. This body-carried point and translated head box
                # become the next recovery seed, but visible/control output
                # immediately uses the separately phase-aligned mapped LP; no
                # unobserved pixel or raw body translation is published.
                body_dx = current_center[0] - flow_body_center[0]
                body_dy = current_center[1] - flow_body_center[1]
                self._flow_point = expected
                self._flow_head_box = (
                    flow_box[0] + body_dx,
                    flow_box[1] + body_dy,
                    flow_box[2] + body_dx,
                    flow_box[3] + body_dy,
                )
                self._flow_feature_box = self._flow_feature_box_for_player(
                    current_box,
                    self._flow_head_box,
                )
                self._flow_source_timestamp_ns = timestamp
                self._flow_body_center = current_center
                self._flow_coordinate_current = True
                self._flow_pixel_observed_current = False
                self._record_tracking_flow_result(
                    source_timestamp_ns=timestamp,
                    pixel_observed=False,
                )
                self._anchor_phase_advanced = False
                self._anchor_phase_hops = 0
            else:
                self._clear_live_flow()
        return True

    def remember_newer_capture_frame(
        self,
        frame,
        *,
        source_timestamp_ns: int,
    ) -> bool:
        """Phase an established head into one newer, uninferred capture.

        This path is deliberately position-only.  It does not create a body
        observation, renew the direct identity deadline, or provide a body
        corroboration point.  A failed LK/corridor check leaves the ordinary
        inferred-frame mapping in control.
        """

        timestamp = int(source_timestamp_ns)
        if timestamp < 0:
            raise ValueError("head capture-phase timestamp cannot be negative")
        body_timestamp_ns = self._current_player_timestamp_ns
        if (
            not self.body_valid
            or not self._current_body_observed
            or self._body_update_deferred
            or body_timestamp_ns is None
            or timestamp <= body_timestamp_ns
            or timestamp - body_timestamp_ns
            > round(
                AUTOMATIC_HEAD_CAPTURE_PHASE_MAX_LEAD_SECONDS
                * 1_000_000_000
            )
            or self._capture_phase_body_timestamp_ns == body_timestamp_ns
        ):
            return False
        self._capture_phase_body_timestamp_ns = body_timestamp_ns
        if (
            not self._flow_coordinate_current
            or not self._flow_pixel_observed_current
            or self._flow_source_timestamp_ns != body_timestamp_ns
        ):
            return False
        if not self._remember_phase_frame(
            frame,
            source_timestamp_ns=timestamp,
        ):
            return False

        flow_point = self._flow_point
        flow_box = self._flow_head_box
        flow_feature_box = self._flow_feature_box
        flow_timestamp_ns = self._flow_source_timestamp_ns
        flow_body_center = self._flow_body_center
        if (
            flow_point is None
            or flow_box is None
            or flow_timestamp_ns is None
            or flow_body_center is None
            or flow_timestamp_ns >= timestamp
        ):
            # State is rechecked after remembering so an unexpected internal
            # discontinuity still fails closed without publishing the frame.
            return True
        advanced = self._advance_phase(
            flow_box,
            feature_box=flow_feature_box,
            anchor_point=flow_point,
            anchor_timestamp_ns=flow_timestamp_ns,
        )
        if (
            advanced is None
            or advanced.source_timestamp_ns != timestamp
            or not self._newer_capture_phase_is_safe(
                advanced.point,
                source_timestamp_ns=timestamp,
            )
        ):
            return True

        dx = advanced.point[0] - flow_point[0]
        dy = advanced.point[1] - flow_point[1]
        self._flow_point = advanced.point
        self._flow_head_box = advanced.head_box
        self._flow_feature_box = (
            advanced.feature_box
            if advanced.feature_box is not None
            else None
            if flow_feature_box is None
            else tuple(
                value + (dx if index % 2 == 0 else dy)
                for index, value in enumerate(flow_feature_box)
            )
        )
        self._flow_source_timestamp_ns = timestamp
        # This is only a geometric companion for validating the next inferred
        # body frame; it is never published as same-frame corroboration.
        self._flow_body_center = (
            flow_body_center[0] + dx,
            flow_body_center[1] + dy,
        )
        self._flow_last_pixel_timestamp_ns = timestamp
        self._flow_coordinate_current = True
        self._flow_pixel_observed_current = True
        self._record_tracking_flow_result(
            source_timestamp_ns=timestamp,
            pixel_observed=True,
        )
        self._anchor_phase_advanced = True
        self._anchor_phase_hops = advanced.frames_spanned
        self._map_current_anchor(now_ns=timestamp)
        return True

    def _can_retain_current_anchor(self, *, now_ns: int) -> bool:
        """Prove a rejected late result cannot invalidate a newer anchor.

        The result itself grants nothing here. Retention is allowed only when
        a different, already-accepted direct observation still has its exact
        measured-primary binding and that newer binding remains geometrically
        compatible with the current measured body inside the immutable lease.
        """

        current_box = self._current_player_box
        current_timestamp_ns = self._current_player_timestamp_ns
        tracker_generation = self._tracker_generation
        anchor_generation = self.anchor.track_generation
        direct_timestamp_ns = self.anchor.last_direct_source_timestamp_ns
        deadline_ns = self.anchor.identity_deadline_ns
        if (
            not self.body_valid
            or not self._current_body_observed
            or self._body_update_deferred
            or current_box is None
            or current_timestamp_ns is None
            or tracker_generation is None
            or anchor_generation != tracker_generation
            or direct_timestamp_ns is None
            or deadline_ns is None
            or max(int(now_ns), current_timestamp_ns) >= deadline_ns
        ):
            return False
        source_binding = self._observed_primary_sources.get(direct_timestamp_ns)
        if (
            source_binding is None
            or source_binding[1] != tracker_generation
            or not self._late_result_boxes_associate_over_interval(
                source_binding[0],
                current_box,
                submitted_timestamp_ns=direct_timestamp_ns,
                current_timestamp_ns=current_timestamp_ns,
                track_generation=tracker_generation,
            )
        ):
            return False
        mapped = self.anchor.map_primary(
            current_box,
            track_generation=tracker_generation,
            source_timestamp_ns=current_timestamp_ns,
            primary_observed=True,
        )
        return bool(
            mapped is not None
            and self._head_point_belongs_to_player(mapped.point, current_box)
        )

    def take_latest(self, *, now_ns: int) -> _AutomaticHeadSample | None:
        """Ingest direct evidence and return only a new direct physical sample.

        The normalized anchor is also refreshed and mapped through current
        primary geometry. That current measured mapping is exposed separately
        through :meth:`visible_sample` so the live adapter can publish it with
        a monotonic source timestamp during short direct-decoder gaps.
        """

        current_ns = int(now_ns)
        if current_ns < 0:
            raise ValueError("head poll timestamp cannot be negative")
        if not self.body_valid or self._body_update_deferred:
            return None
        result = self.worker.take_latest(self.identity_generation)
        physical_sample: _AutomaticHeadSample | None = None
        if result is not None:
            result_age_ns = current_ns - result.source_timestamp_ns
            if result_age_ns < 0 or result_age_ns > self.stale_after_ns:
                # Worker freshness and anchor identity are separate. A late
                # result is ignored; it must not revoke a newer live anchor
                # merely because its exact-source binding has aged out. It
                # does, however, prove that maintenance cadence is not keeping
                # the model current, so the next accepted submission returns
                # to acquisition speed.
                self._tracking_cadence_requires_direct_refresh = True
                result = None
            elif (
                result.source_timestamp_ns
                in self._rejected_body_outlier_source_timestamps
            ):
                # A result already in flight for the removed singleton body
                # fragment owns no source binding in the repaired exact chain.
                # Discard it without invalidating the still-live clean anchor.
                result = None
        if result is not None:
            current_box = self._current_player_box
            current_body_timestamp_ns = self._current_player_timestamp_ns
            source_binding = self._observed_primary_sources.get(
                result.source_timestamp_ns
            )
            result_box = tuple(float(value) for value in result.selected_player_box)
            if source_binding is None or source_binding[0] != result_box:
                # A direct observation without its exact accepted primary
                # source binding has no trustworthy identity generation.
                self.advance_identity()
                return None
            _source_box, source_track_generation = source_binding
            association_timestamp_ns = (
                current_ns
                if current_body_timestamp_ns is None
                else current_body_timestamp_ns
            )
            if (
                current_box is None
                or self._tracker_generation is None
                or source_track_generation != self._tracker_generation
            ):
                self.advance_identity()
                return None
            if not self._late_result_boxes_associate_over_interval(
                result_box,
                current_box,
                submitted_timestamp_ns=result.source_timestamp_ns,
                current_timestamp_ns=association_timestamp_ns,
                track_generation=source_track_generation,
            ):
                # Never normalize a late result against an incompatible body.
                # If a separate newer anchor is still exactly bound and safe,
                # discard only this result; otherwise fail closed on the epoch.
                if self._can_retain_current_anchor(now_ns=current_ns):
                    self._map_current_anchor(now_ns=current_ns)
                    return None
                self.advance_identity()
                return None
            localization_reason = getattr(result, "localization_reason", None)
            self._latest_localization_reason = (
                None
                if localization_reason is None
                else str(
                    getattr(localization_reason, "value", localization_reason)
                )
            )
            self._latest_localization_source_timestamp_ns = (
                result.source_timestamp_ns
            )
            self._latest_localization_track_generation = (
                source_track_generation
            )
            observation = result.observation
            if observation is None:
                # A decoder miss contains no contradictory spatial evidence.
                # Keep the already-qualified mapped-motion state while the
                # same measured primary carries the immutable direct-head
                # lease.  Revoking here made pursuit collapse and rebuild on
                # every ordinary no-head result even though this same frame's
                # measured body immediately re-authorized mapped motion.  A
                # predicted/missing primary, identity change, activation edge,
                # or the direct deadline still revokes synchronously.
                self._tracking_consecutive_head_misses += 1
                if (
                    self._tracking_consecutive_head_misses
                    >= AUTOMATIC_HEAD_TRACKING_MAX_CONSECUTIVE_MISSES
                ):
                    self._tracking_cadence_requires_direct_refresh = True
            elif not self._head_point_belongs_to_player(
                observation.point,
                result_box,
            ):
                self.advance_identity()
                return None
            else:
                anchor_point = observation.point
                physical_source_timestamp_ns = result.source_timestamp_ns
                accepted_phase = None
                observation_head_box = getattr(observation, "head_box", None)
                if (
                    observation_head_box is not None
                    and current_body_timestamp_ns is not None
                    and self._current_body_observed
                ):
                    phase = self._advance_phase(
                        observation_head_box,
                        feature_box=self._flow_feature_box_for_player(
                            result_box,
                            observation_head_box,
                        ),
                        anchor_point=observation.point,
                        anchor_timestamp_ns=result.source_timestamp_ns,
                    )
                    normalized = self._normalized_point_in_box(
                        observation.point,
                        result_box,
                    )
                    if normalized is not None:
                        coarse_endpoint = (
                            current_box[0]
                            + normalized[0] * (current_box[2] - current_box[0]),
                            current_box[1]
                            + normalized[1] * (current_box[3] - current_box[1]),
                        )
                    else:  # pragma: no cover - result_box was validated above
                        coarse_endpoint = observation.point
                    phase_elapsed_ns = max(
                        0,
                        current_body_timestamp_ns
                        - result.source_timestamp_ns,
                    )
                    if (
                        phase is not None
                        and phase.source_timestamp_ns
                        == current_body_timestamp_ns
                        and self._head_point_belongs_to_player(
                            phase.point,
                            current_box,
                        )
                        and self._flow_body_residual_is_safe(
                            phase.point,
                            coarse_endpoint,
                            result_box,
                            current_box,
                            elapsed_ns=phase_elapsed_ns,
                            previous_body_source_timestamp_ns=(
                                result.source_timestamp_ns
                            ),
                            current_body_source_timestamp_ns=(
                                current_body_timestamp_ns
                            ),
                        )
                    ):
                        # If a one-frame live flow state already reached this
                        # image, use it as the continuity authority and let the
                        # delayed detector remove drift gradually.  A fresh
                        # model result can never snap a healthy pixel track.
                        if (
                            self._flow_coordinate_current
                            and self._flow_point is not None
                            and self._flow_head_box is not None
                            and self._flow_source_timestamp_ns
                            == current_body_timestamp_ns
                        ):
                            innovation_x = phase.point[0] - self._flow_point[0]
                            innovation_y = phase.point[1] - self._flow_point[1]
                            if math.hypot(innovation_x, innovation_y) <= (
                                AUTOMATIC_HEAD_FLOW_MAX_BODY_RESIDUAL_PIXELS
                            ):
                                # Keep this bounded per detector correction,
                                # rather than time-normalizing it upward at the
                                # lower maintenance rate. LK owns continuous
                                # target motion; larger sparse correction steps
                                # would turn localizer drift into visible shake.
                                correction_fraction = 0.20
                                corrected_point = (
                                    self._flow_point[0]
                                    + correction_fraction * innovation_x,
                                    self._flow_point[1]
                                    + correction_fraction * innovation_y,
                                )
                                corrected_box = tuple(
                                    value
                                    + correction_fraction
                                    * (
                                        phase.head_box[index]
                                        - self._flow_head_box[index]
                                    )
                                    for index, value in enumerate(
                                        self._flow_head_box
                                    )
                                )
                                phase = type(phase)(
                                    point=corrected_point,
                                    head_box=corrected_box,
                                    anchor_timestamp_ns=(
                                        phase.anchor_timestamp_ns
                                    ),
                                    source_timestamp_ns=(
                                        phase.source_timestamp_ns
                                    ),
                                    identity_generation=(
                                        phase.identity_generation
                                    ),
                                    hops=phase.hops,
                                    frames_spanned=phase.frames_spanned,
                                    flow_measurements=(
                                        phase.flow_measurements
                                    ),
                                    strategy=phase.strategy,
                                    minimum_inlier_fraction=(
                                        phase.minimum_inlier_fraction
                                    ),
                                    maximum_forward_backward_error=(
                                        phase.maximum_forward_backward_error
                                    ),
                                    feature_box=(
                                        self._flow_feature_box
                                        if self._flow_feature_box is not None
                                        else phase.feature_box
                                    ),
                                )
                            else:
                                phase = None
                        if phase is not None:
                            accepted_phase = phase
                            anchor_point = phase.point
                            physical_source_timestamp_ns = (
                                phase.source_timestamp_ns
                            )
                            current_center = (
                                (current_box[0] + current_box[2]) * 0.5,
                                (current_box[1] + current_box[3]) * 0.5,
                            )
                previous_direct_ns = self.anchor.last_direct_source_timestamp_ns
                # DirectHeadAnchor's timestamp and geometry are an immutable
                # same-source pair. The phase-corrected coordinate belongs to
                # a newer image and lives only in the flow state above.
                observed = self.anchor.observe_direct(
                    observation.point,
                    result_box,
                    track_generation=source_track_generation,
                    source_timestamp_ns=result.source_timestamp_ns,
                    confidence=observation.confidence,
                )
                if observed is not None:
                    stable_normalized = self.anchor.normalized_point
                    stable_endpoint = (
                        None
                        if stable_normalized is None
                        else (
                            current_box[0]
                            + stable_normalized[0]
                            * (current_box[2] - current_box[0]),
                            current_box[1]
                            + stable_normalized[1]
                            * (current_box[3] - current_box[1]),
                        )
                    )
                    if (
                        accepted_phase is not None
                        and stable_endpoint is not None
                        and not self._flow_body_residual_is_safe(
                            accepted_phase.point,
                            stable_endpoint,
                            result_box,
                            current_box,
                            elapsed_ns=max(
                                0,
                                current_body_timestamp_ns
                                - result.source_timestamp_ns,
                            ),
                            previous_body_source_timestamp_ns=(
                                result.source_timestamp_ns
                            ),
                            current_body_source_timestamp_ns=(
                                current_body_timestamp_ns
                            ),
                        )
                    ):
                        accepted_phase = None
                        anchor_point = observation.point
                        physical_source_timestamp_ns = result.source_timestamp_ns
                    if accepted_phase is not None:
                        self._flow_point = accepted_phase.point
                        self._flow_head_box = accepted_phase.head_box
                        self._flow_feature_box = (
                            self._flow_feature_box_for_player(
                                current_box,
                                accepted_phase.head_box,
                            )
                        )
                        self._flow_source_timestamp_ns = (
                            accepted_phase.source_timestamp_ns
                        )
                        self._flow_body_center = current_center
                        self._flow_last_pixel_timestamp_ns = (
                            accepted_phase.source_timestamp_ns
                        )
                        self._flow_coordinate_current = True
                        self._flow_pixel_observed_current = bool(
                            accepted_phase.flow_measurements
                        )
                        self._record_tracking_flow_result(
                            source_timestamp_ns=(
                                accepted_phase.source_timestamp_ns
                            ),
                            pixel_observed=bool(
                                accepted_phase.flow_measurements
                            ),
                        )
                        self._anchor_phase_advanced = bool(
                            accepted_phase.flow_measurements
                        )
                        self._anchor_phase_hops = accepted_phase.frames_spanned
                    if (
                        stable_endpoint is not None
                        and self._flow_coordinate_current
                        and self._flow_point is not None
                        and self._flow_source_timestamp_ns
                        == current_body_timestamp_ns
                    ):
                        if not self._flow_body_residual_is_safe(
                            self._flow_point,
                            stable_endpoint,
                            result_box,
                            current_box,
                            elapsed_ns=max(
                                0,
                                current_body_timestamp_ns
                                - result.source_timestamp_ns,
                            ),
                            previous_body_source_timestamp_ns=(
                                result.source_timestamp_ns
                            ),
                            current_body_source_timestamp_ns=(
                                current_body_timestamp_ns
                            ),
                        ):
                            # A fresh same-source observation may move the
                            # anchor's rolling median. Reapply the same bounded
                            # circular/directional invariant after that update
                            # rather than leaving a now-invalid flow coordinate
                            # published for one frame.
                            self._clear_live_flow()
                            if accepted_phase is not None:
                                accepted_phase = None
                                anchor_point = observation.point
                                physical_source_timestamp_ns = (
                                    result.source_timestamp_ns
                                )
                    if (
                        previous_direct_ns is None
                        or result.source_timestamp_ns - previous_direct_ns
                        >= self.anchor.max_direct_age_ns
                    ):
                        self._reset_mapped_filter()
                    self._anchor_evidence = (
                        observation.evidence
                        + (
                            " + observed-pixel phase correction"
                            if accepted_phase is not None
                            and accepted_phase.flow_measurements > 0
                            else ""
                        )
                    )
                    source_box = (
                        current_box
                        if accepted_phase is not None
                        else source_binding[0]
                    )
                    physical_sample = _AutomaticHeadSample(
                        point=anchor_point,
                        source_timestamp_ns=physical_source_timestamp_ns,
                        direct_source_timestamp_ns=result.source_timestamp_ns,
                        identity_deadline_ns=observed.identity_deadline_ns,
                        track_generation=source_track_generation,
                        provenance=DirectHeadProvenance.DIRECT,
                        confidence=observation.confidence,
                        evidence=observation.evidence,
                        velocity_point=anchor_point,
                        bridging=False,
                        body_derived_motion_permitted=False,
                        body_derived_motion_deadline_ns=None,
                        corroboration_point=(
                            (source_box[0] + source_box[2]) * 0.5,
                            (source_box[1] + source_box[3]) * 0.5,
                        ),
                        phase_advanced=bool(
                            accepted_phase is not None
                            and accepted_phase.flow_measurements > 0
                        ),
                        phase_hops=(
                            0
                            if accepted_phase is None
                            else accepted_phase.frames_spanned
                        ),
                    )
                    # A newly accepted direct observation is the only event
                    # that clears the adaptive-cadence recovery latch.  The
                    # following submit still stays at 90 Hz unless LK has also
                    # produced a current observed-pixel endpoint.
                    self._tracking_cadence_requires_direct_refresh = False
                    self._tracking_consecutive_head_misses = 0

        generation_before_display = self.identity_generation
        self._map_current_anchor(now_ns=current_ns)
        if (
            physical_sample is None
            or not self.body_valid
            or self.identity_generation != generation_before_display
        ):
            return None
        previous_timestamp_ns = self._last_physical_source_timestamp_ns
        if (
            previous_timestamp_ns is not None
            and physical_sample.source_timestamp_ns <= previous_timestamp_ns
        ):
            return None
        self._last_physical_source_timestamp_ns = physical_sample.source_timestamp_ns
        return physical_sample

    def _map_current_anchor(self, *, now_ns: int) -> None:
        """Refresh current mapped geometry and its identity boundary."""

        current_box = self._current_player_box
        aim_box = self._current_aim_box
        source_timestamp_ns = self._current_player_timestamp_ns
        track_generation = self._tracker_generation
        if (
            current_box is None
            or aim_box is None
            or source_timestamp_ns is None
            or track_generation is None
        ):
            self._visible_sample = None
            self._reset_mapped_filter()
            return

        deadline_ns = self.anchor.identity_deadline_ns
        if deadline_ns is not None and max(now_ns, source_timestamp_ns) >= deadline_ns:
            # Expiry is a hard safety epoch so the live loop publishes an
            # immediate controller loss instead of extending the last mapped
            # position by the controller's separate 65 ms input lease.
            self.advance_identity()
            return
        mapping_box = aim_box
        anchored = self.anchor.map_primary(
            aim_box,
            track_generation=track_generation,
            source_timestamp_ns=source_timestamp_ns,
            primary_observed=self._current_body_observed,
        )
        if anchored is None:
            self._visible_sample = None
            self._reset_mapped_filter()
            return
        if not self._head_point_belongs_to_player(anchored.point, current_box):
            # Tracker smoothing can temporarily trail a valid raw primary box
            # enough that mapping through the smoothed geometry puts the head
            # outside the current measured body. The exact raw primary remains
            # the identity/safety authority, so retry that same immutable
            # normalized anchor through raw geometry. This creates no new head
            # evidence and does not renew the direct deadline.
            if self._current_body_observed and aim_box != current_box:
                mapping_box = current_box
                anchored = self.anchor.map_primary(
                    current_box,
                    track_generation=track_generation,
                    source_timestamp_ns=source_timestamp_ns,
                    primary_observed=True,
                )
            if (
                anchored is None
                or not self._head_point_belongs_to_player(
                    anchored.point,
                    current_box,
                )
            ):
                self.advance_identity()
                return
        if self._current_body_observed:
            body_filtered_point, body_velocity_point = self._filter_mapped_point(
                anchored,
                mapping_box=mapping_box,
            )
            flow_coordinate_current = bool(
                self._flow_coordinate_current
                and self._flow_point is not None
                and self._flow_source_timestamp_ns is not None
                and (
                    (
                        self._flow_source_timestamp_ns == source_timestamp_ns
                        and self._head_point_belongs_to_player(
                            self._flow_point,
                            current_box,
                        )
                    )
                    or (
                        self._flow_source_timestamp_ns > source_timestamp_ns
                        and self._newer_capture_phase_is_safe(
                            self._flow_point,
                            source_timestamp_ns=(
                                self._flow_source_timestamp_ns
                            ),
                        )
                    )
                )
            )
            pixel_tracked = bool(
                flow_coordinate_current
                and self._flow_pixel_observed_current
            )
            if flow_coordinate_current and pixel_tracked:
                assert self._flow_point is not None
                # This coordinate already contains all observed camera/body
                # translation through the newest image. Mapping it through
                # current body geometry again would double-move it.
                filtered_point = self._flow_point
                velocity_point = self._flow_point
                # Keep the hidden fallback filter phase-aligned with the point
                # actually published to control. If LK disappears on the next
                # frame, the mapped channel can then reconcile over its normal
                # time constant instead of snapping from a valid flow endpoint
                # to a separately evolved body-only coordinate.
                self._mapped_filter_point = self._flow_point
                self._mapped_velocity_filter_point = self._flow_point
                self._mapped_velocity_translation_point = self._flow_point
                self._mapped_velocity_reconcile_point = self._flow_point
            else:
                # A failed LK step may keep its body-translated point privately
                # as the next bounded recovery seed, but it has no current pixel
                # evidence. Publishing that raw translation bypassed the 12 ms
                # mapped-position LP and turned alternating body-box jitter into
                # multi-thousand-count/s command reversals. Fall back on the
                # already-updated filtered body mapping on the first miss.
                filtered_point = body_filtered_point
                velocity_point = body_velocity_point
        else:
            # Prediction is display-only and is a physical observation gap.
            # Do not advance or publish the causal controller-input filter.
            # Retaining its last exact state prevents the first measured frame
            # after a short gap from bypassing the innovation/slew bounds.
            filtered_point = anchored.point
            velocity_point = anchored.point
            pixel_tracked = False
        sample = self._runtime_sample(
            anchored,
            point=filtered_point,
            velocity_point=velocity_point,
            pixel_tracked=pixel_tracked,
            source_timestamp_ns=(
                self._flow_source_timestamp_ns
                if pixel_tracked
                else None
            ),
        )
        self._visible_sample = sample
        self._visible_player_box = current_box
        self._visible_player_timestamp_ns = source_timestamp_ns

        if not self._current_body_observed:
            # Predicted primary geometry is display-only. It must revoke every
            # predictive-motion grant immediately, while a measured
            # anchor remains display-only. Broadly revoking here on every
            # measured frame would reset independently corroborated direct-head
            # direction history between worker results.
            self._motion_corroboration_revocation_pending = True
        return

    def _filter_mapped_point(
        self,
        sample: DirectHeadAnchorSample,
        *,
        mapping_box: tuple[float, float, float, float] | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Qualify mapped motion, then apply separate position/velocity LPs.

        A detector box can change shape for one frame even though the tracker
        still has the same measured player. Revoking that identity caused a
        stop/reacquire cycle; feeding raw normalized-anchor changes caused a
        shake spike. The stable fallback position channel keeps its short
        absolute-coordinate LP; passing every body-center delta directly
        increased stationary command chatter in the measured controller sweep.
        The observed-pixel flow path bypasses this fallback entirely when
        available. The velocity channel shifts both of its filter states by
        separately qualified body-center motion before gently reconciling local
        pose, so a fallback can remain position-stable without turning
        localizer wobble into target velocity.
        """

        if sample.provenance is not DirectHeadProvenance.MEASURED_PRIMARY:
            self._reset_mapped_filter()
            return sample.point, sample.point
        if mapping_box is None:
            # Private callers which do not supply the geometry that produced
            # the mapped point cannot separate local anchor motion from body
            # translation. Preserve the historical conservative shared LP.
            return self._filter_unattributed_mapped_point(sample)
        timestamp_ns = sample.source_timestamp_ns
        previous_point = self._mapped_filter_point
        previous_input_point = self._mapped_filter_input_point
        previous_velocity_point = self._mapped_velocity_filter_point
        previous_center = self._mapped_qualified_center
        previous_translation_point = self._mapped_velocity_translation_point
        previous_reconcile_point = self._mapped_velocity_reconcile_point
        previous_timestamp_ns = self._mapped_filter_timestamp_ns
        raw_normalized = self._normalized_point_in_box(sample.point, mapping_box)
        if raw_normalized is None:
            self._reset_mapped_filter()
            return sample.point, sample.point
        raw_center = (
            (mapping_box[0] + mapping_box[2]) * 0.5,
            (mapping_box[1] + mapping_box[3]) * 0.5,
        )
        if (
            previous_point is None
            or previous_input_point is None
            or previous_velocity_point is None
            or previous_center is None
            or previous_translation_point is None
            or previous_reconcile_point is None
            or previous_timestamp_ns is None
        ):
            self._mapped_anchor_candidate_normalized = raw_normalized
            self._mapped_anchor_filtered_normalized = raw_normalized
            self._mapped_anchor_direct_timestamp_ns = (
                sample.direct_source_timestamp_ns
            )
            qualified_input = sample.point
            filtered = qualified_input
            velocity_filtered = qualified_input
            qualified_center = raw_center
            translation_point = qualified_input
            reconcile_point = qualified_input
        else:
            elapsed_ns = timestamp_ns - previous_timestamp_ns
            if elapsed_ns <= 0:
                return previous_point, previous_velocity_point
            if elapsed_ns > self.stale_after_ns:
                # A long primary-measurement gap is a discontinuity, not a
                # reason to drag an old screen coordinate into reacquisition.
                self._mapped_anchor_candidate_normalized = raw_normalized
                self._mapped_anchor_filtered_normalized = raw_normalized
                self._mapped_anchor_direct_timestamp_ns = (
                    sample.direct_source_timestamp_ns
                )
                qualified_input = sample.point
                filtered = qualified_input
                velocity_filtered = qualified_input
                qualified_center = raw_center
                translation_point = qualified_input
                reconcile_point = qualified_input
            else:
                elapsed_seconds = elapsed_ns / 1_000_000_000
                direct_timestamp_ns = sample.direct_source_timestamp_ns
                previous_direct_timestamp_ns = (
                    self._mapped_anchor_direct_timestamp_ns
                )
                if (
                    previous_direct_timestamp_ns is None
                    or direct_timestamp_ns > previous_direct_timestamp_ns
                ):
                    self._mapped_anchor_candidate_normalized = raw_normalized
                    self._mapped_anchor_direct_timestamp_ns = direct_timestamp_ns
                anchor_candidate = self._mapped_anchor_candidate_normalized
                anchor_filtered = self._mapped_anchor_filtered_normalized
                assert anchor_candidate is not None
                assert anchor_filtered is not None
                anchor_alpha = 1.0 - math.exp(
                    -elapsed_seconds
                    / AUTOMATIC_HEAD_NORMALIZED_ANCHOR_FILTER_TIME_CONSTANT_SECONDS
                )
                anchor_filtered = (
                    anchor_filtered[0]
                    + anchor_alpha
                    * (anchor_candidate[0] - anchor_filtered[0]),
                    anchor_filtered[1]
                    + anchor_alpha
                    * (anchor_candidate[1] - anchor_filtered[1]),
                )
                self._mapped_anchor_filtered_normalized = anchor_filtered
                stabilized_input = (
                    mapping_box[0]
                    + anchor_filtered[0] * (mapping_box[2] - mapping_box[0]),
                    mapping_box[1]
                    + anchor_filtered[1] * (mapping_box[3] - mapping_box[1]),
                )
                input_dx = stabilized_input[0] - previous_input_point[0]
                input_dy = stabilized_input[1] - previous_input_point[1]
                input_distance = math.hypot(input_dx, input_dy)
                maximum_input_step = (
                    AUTOMATIC_HEAD_MAPPED_STEP_ALLOWANCE_PIXELS
                    + AUTOMATIC_HEAD_MAPPED_MAX_SPEED_PIXELS_PER_SECOND
                    * elapsed_seconds
                )
                if input_distance > maximum_input_step:
                    scale = maximum_input_step / input_distance
                    qualified_input = (
                        previous_input_point[0] + input_dx * scale,
                        previous_input_point[1] + input_dy * scale,
                    )
                else:
                    qualified_input = stabilized_input

                center_dx = raw_center[0] - previous_center[0]
                center_dy = raw_center[1] - previous_center[1]
                center_distance = math.hypot(center_dx, center_dy)
                if center_distance > maximum_input_step:
                    center_scale = maximum_input_step / center_distance
                    qualified_center = (
                        previous_center[0] + center_dx * center_scale,
                        previous_center[1] + center_dy * center_scale,
                    )
                else:
                    qualified_center = raw_center
                qualified_center_delta = (
                    qualified_center[0] - previous_center[0],
                    qualified_center[1] - previous_center[1],
                )
                alpha = 1.0 - math.exp(
                    -elapsed_seconds
                    / AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS
                )
                reconcile_alpha = 1.0 - math.exp(
                    -elapsed_seconds
                    / AUTOMATIC_HEAD_VELOCITY_RECONCILIATION_TIME_CONSTANT_SECONDS
                )
                velocity_alpha = 1.0 - math.exp(
                    -elapsed_seconds
                    / AUTOMATIC_HEAD_MAPPED_VELOCITY_FILTER_TIME_CONSTANT_SECONDS
                )
                filtered = (
                    previous_point[0]
                    + alpha * (qualified_input[0] - previous_point[0]),
                    previous_point[1]
                    + alpha * (qualified_input[1] - previous_point[1]),
                )
                translation_point = (
                    previous_translation_point[0] + qualified_center_delta[0],
                    previous_translation_point[1] + qualified_center_delta[1],
                )
                reconcile_prediction = (
                    previous_reconcile_point[0] + qualified_center_delta[0],
                    previous_reconcile_point[1] + qualified_center_delta[1],
                )
                reconcile_point = (
                    reconcile_prediction[0]
                    + reconcile_alpha
                    * (qualified_input[0] - reconcile_prediction[0]),
                    reconcile_prediction[1]
                    + reconcile_alpha
                    * (qualified_input[1] - reconcile_prediction[1]),
                )
                velocity_prediction = (
                    previous_velocity_point[0] + qualified_center_delta[0],
                    previous_velocity_point[1] + qualified_center_delta[1],
                )
                velocity_filtered = (
                    velocity_prediction[0]
                    + velocity_alpha
                    * (
                        reconcile_point[0]
                        - velocity_prediction[0]
                    ),
                    velocity_prediction[1]
                    + velocity_alpha
                    * (
                        reconcile_point[1]
                        - velocity_prediction[1]
                    ),
                )
        self._mapped_filter_point = filtered
        self._mapped_filter_input_point = qualified_input
        self._mapped_velocity_filter_point = velocity_filtered
        self._mapped_qualified_center = qualified_center
        self._mapped_reference_box = tuple(float(value) for value in mapping_box)
        self._mapped_velocity_translation_point = translation_point
        self._mapped_velocity_reconcile_point = reconcile_point
        self._mapped_filter_timestamp_ns = timestamp_ns
        return filtered, velocity_filtered

    def _filter_unattributed_mapped_point(
        self,
        sample: DirectHeadAnchorSample,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Retain the legacy shared filter when source box geometry is absent."""

        timestamp_ns = sample.source_timestamp_ns
        previous_point = self._mapped_filter_point
        previous_input_point = self._mapped_filter_input_point
        previous_velocity_point = self._mapped_velocity_filter_point
        previous_timestamp_ns = self._mapped_filter_timestamp_ns
        if (
            previous_point is None
            or previous_input_point is None
            or previous_velocity_point is None
            or previous_timestamp_ns is None
        ):
            qualified_input = sample.point
            filtered = qualified_input
            velocity_filtered = qualified_input
        else:
            elapsed_ns = timestamp_ns - previous_timestamp_ns
            if elapsed_ns <= 0:
                return previous_point, previous_velocity_point
            if elapsed_ns > self.stale_after_ns:
                qualified_input = sample.point
                filtered = qualified_input
                velocity_filtered = qualified_input
            else:
                elapsed_seconds = elapsed_ns / 1_000_000_000
                input_dx = sample.point[0] - previous_input_point[0]
                input_dy = sample.point[1] - previous_input_point[1]
                input_distance = math.hypot(input_dx, input_dy)
                maximum_input_step = (
                    AUTOMATIC_HEAD_MAPPED_STEP_ALLOWANCE_PIXELS
                    + AUTOMATIC_HEAD_MAPPED_MAX_SPEED_PIXELS_PER_SECOND
                    * elapsed_seconds
                )
                if input_distance > maximum_input_step:
                    scale = maximum_input_step / input_distance
                    qualified_input = (
                        previous_input_point[0] + input_dx * scale,
                        previous_input_point[1] + input_dy * scale,
                    )
                else:
                    qualified_input = sample.point
                alpha = 1.0 - math.exp(
                    -elapsed_seconds
                    / AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS
                )
                velocity_alpha = 1.0 - math.exp(
                    -elapsed_seconds
                    / AUTOMATIC_HEAD_MAPPED_VELOCITY_FILTER_TIME_CONSTANT_SECONDS
                )
                filtered = (
                    previous_point[0]
                    + alpha * (qualified_input[0] - previous_point[0]),
                    previous_point[1]
                    + alpha * (qualified_input[1] - previous_point[1]),
                )
                velocity_filtered = (
                    previous_velocity_point[0]
                    + velocity_alpha
                    * (qualified_input[0] - previous_velocity_point[0]),
                    previous_velocity_point[1]
                    + velocity_alpha
                    * (qualified_input[1] - previous_velocity_point[1]),
                )
        self._mapped_filter_point = filtered
        self._mapped_filter_input_point = qualified_input
        self._mapped_velocity_filter_point = velocity_filtered
        self._mapped_filter_timestamp_ns = timestamp_ns
        return filtered, velocity_filtered

    @staticmethod
    def _normalized_point_in_box(
        point: tuple[float, float],
        box: tuple[float, float, float, float] | None,
    ) -> tuple[float, float] | None:
        if box is None:
            return None
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width <= 0.0 or height <= 0.0:
            return None
        return (
            (point[0] - box[0]) / width,
            (point[1] - box[1]) / height,
        )

    def _runtime_sample(
        self,
        sample: DirectHeadAnchorSample,
        *,
        point: tuple[float, float] | None = None,
        velocity_point: tuple[float, float] | None = None,
        pixel_tracked: bool = False,
        source_timestamp_ns: int | None = None,
    ) -> _AutomaticHeadSample:
        if not isinstance(pixel_tracked, bool):
            raise TypeError("pixel_tracked must be bool")
        bridging = sample.provenance is DirectHeadProvenance.PREDICTED_PRIMARY
        measured_primary = (
            sample.provenance is DirectHeadProvenance.MEASURED_PRIMARY
        )
        independent_pixel_motion = bool(measured_primary and pixel_tracked)
        effective_source_timestamp_ns = (
            sample.source_timestamp_ns
            if source_timestamp_ns is None
            else int(source_timestamp_ns)
        )
        if effective_source_timestamp_ns < sample.source_timestamp_ns:
            raise ValueError("runtime sample timestamp cannot move backwards")
        current_box = self._current_player_box
        corroboration_point = None
        if (
            independent_pixel_motion
            and current_box is not None
            and effective_source_timestamp_ns == sample.source_timestamp_ns
        ):
            corroboration_point = (
                (current_box[0] + current_box[2]) * 0.5,
                (current_box[1] + current_box[3]) * 0.5,
            )
        return _AutomaticHeadSample(
            point=sample.point if point is None else point,
            source_timestamp_ns=effective_source_timestamp_ns,
            direct_source_timestamp_ns=sample.direct_source_timestamp_ns,
            identity_deadline_ns=sample.identity_deadline_ns,
            track_generation=sample.track_generation,
            provenance=sample.provenance,
            confidence=sample.confidence,
            evidence=self._anchor_evidence,
            velocity_point=(
                sample.point if velocity_point is None else velocity_point
            ),
            bridging=bridging,
            # The direct result established the normalized head location; this
            # newer point is that same identity mapped through a real primary
            # measurement.  Mark the provenance explicitly so the numeric core
            # can keep it position-only under the configured zero predictive
            # fractions instead of mistaking body motion for independent head
            # velocity.
            body_derived_motion_permitted=(
                measured_primary and not independent_pixel_motion
            ),
            body_derived_motion_deadline_ns=(
                sample.identity_deadline_ns
                if measured_primary and not independent_pixel_motion
                else None
            ),
            corroboration_point=corroboration_point,
            phase_advanced=independent_pixel_motion,
            phase_hops=(self._anchor_phase_hops if independent_pixel_motion else 0),
            verified_flow_motion=bool(
                independent_pixel_motion
                and self._tracking_flow_success_streak
                >= AUTOMATIC_HEAD_TRACKING_MINIMUM_FLOW_SAMPLES
            ),
        )

    @staticmethod
    def _point_is_inside_frame(point, frame_shape) -> bool:
        try:
            if len(frame_shape) < 2:
                return False
            height = int(frame_shape[0])
            width = int(frame_shape[1])
            x, y = (float(value) for value in point)
        except (TypeError, ValueError):
            return False
        return bool(
            height > 0
            and width > 0
            and math.isfinite(x)
            and math.isfinite(y)
            and 0.0 <= x < width
            and 0.0 <= y < height
        )

    def visible_sample(
        self,
        *,
        now_ns: int,
        frame_shape=None,
    ) -> _AutomaticHeadSample | None:
        current_ns = int(now_ns)
        if current_ns < 0:
            raise ValueError("head display timestamp cannot be negative")
        if frame_shape is not None:
            if (
                not hasattr(frame_shape, "__len__")
                or len(frame_shape) < 2
                or isinstance(frame_shape[0], bool)
                or isinstance(frame_shape[1], bool)
                or not isinstance(frame_shape[0], int)
                or not isinstance(frame_shape[1], int)
                or frame_shape[0] <= 0
                or frame_shape[1] <= 0
            ):
                raise ValueError(
                    "head frame_shape must contain positive height and width"
                )
        if self._body_update_deferred:
            return None
        sample = self._visible_sample
        if sample is None or not self.body_valid:
            return None
        if current_ns >= sample.identity_deadline_ns:
            self.advance_identity()
            return None
        if (
            self._current_player_box is None
            or self._visible_player_box is None
            or self._visible_player_timestamp_ns is None
            or not self._player_boxes_associate_over_interval(
                self._visible_player_box,
                self._current_player_box,
                elapsed_ns=(
                    (
                        current_ns
                        if self._current_player_timestamp_ns is None
                        else self._current_player_timestamp_ns
                    )
                    - self._visible_player_timestamp_ns
                ),
            )
        ):
            self.advance_identity()
            return None
        # ``_map_current_anchor`` anatomy-checks the raw mapped endpoint before
        # publishing this sample.  The causal filtered point can briefly trail
        # outside the newest raw box during fast motion or detector shape flex;
        # treating that harmless filter state as a new identity caused a hard
        # stop/reacquire cycle.
        if frame_shape is not None:
            if not self._point_is_inside_frame(sample.point, frame_shape):
                # A feedback point outside the source image has no valid
                # control interpretation. End the identity rather than
                # clipping it to an edge and manufacturing a large command.
                self.advance_identity()
                return None
            velocity_point = (
                sample.point
                if sample.velocity_point is None
                else sample.velocity_point
            )
            auxiliary_point_invalid = bool(
                not self._point_is_inside_frame(velocity_point, frame_shape)
                or (
                    sample.corroboration_point is not None
                    and not self._point_is_inside_frame(
                        sample.corroboration_point,
                        frame_shape,
                    )
                )
            )
            if auxiliary_point_invalid:
                # Position remains physically valid, but the separate motion
                # channel has escaped its evidence domain (most commonly when
                # one side of a body box is clipped by the frame edge). Keep
                # feedback continuity, collapse every predictive field to the
                # valid point, and make motion authority earn itself again.
                self._mapped_velocity_filter_point = sample.point
                self._mapped_velocity_translation_point = sample.point
                self._mapped_velocity_reconcile_point = sample.point
                self._motion_corroboration_revocation_pending = True
                self._clear_live_flow()
                sample = replace(
                    sample,
                    evidence=sample.evidence + " + motion channel revoked",
                    velocity_point=sample.point,
                    body_derived_motion_permitted=False,
                    body_derived_motion_deadline_ns=None,
                    corroboration_point=None,
                    phase_advanced=False,
                    phase_hops=0,
                    verified_flow_motion=False,
                )
                self._visible_sample = sample
        return sample

    def has_live_measured_anchor(self, *, now_ns: int) -> bool:
        """Whether the preceding measured primary can carry verified identity.

        This read-only signal lets the conditional detail rescue avoid a
        redundant second body-detector inference once a direct head is already
        established.  It cannot renew the identity deadline, authorize a
        predicted body, or create control from body geometry.
        """

        current_ns = int(now_ns)
        if current_ns < 0:
            raise ValueError("head anchor query timestamp cannot be negative")
        deadline_ns = self.anchor.identity_deadline_ns
        anchor_generation = self.anchor.track_generation
        return bool(
            self.body_valid
            and not self._body_update_deferred
            and self._current_body_observed
            and self._visible_sample is not None
            and deadline_ns is not None
            and current_ns < deadline_ns
            and self._tracker_generation is not None
            and anchor_generation == self._tracker_generation
        )

    def verified_flow_point_for_frame(
        self,
        *,
        source_timestamp_ns: int,
        now_ns: int,
    ) -> tuple[float, float] | None:
        """Return an exact-frame pixel endpoint for the live anchored target.

        The capture mailbox can phase a verified head into the frame that the
        primary detector consumes on the following loop iteration.  Exposing
        only that exact timestamp match lets the self-filter integration keep
        a known opponent cluster out of self-avatar acquisition without using
        stale geometry or granting a new identity.
        """

        source_ns = int(source_timestamp_ns)
        current_ns = int(now_ns)
        if source_ns < 0 or current_ns < 0:
            raise ValueError("verified flow timestamps cannot be negative")
        deadline_ns = self.anchor.identity_deadline_ns
        point = self._flow_point
        if (
            not self.body_valid
            or not self._current_body_observed
            or self._body_update_deferred
            or not self._flow_coordinate_current
            or not self._flow_pixel_observed_current
            or point is None
            or self._flow_source_timestamp_ns != source_ns
            or deadline_ns is None
            or max(source_ns, current_ns) >= deadline_ns
            or self._tracker_generation is None
            or self.anchor.track_generation != self._tracker_generation
        ):
            return None
        if not all(math.isfinite(value) for value in point):
            return None
        return point

    def body_fallback_no_decoded_deadline_ns(self, *, now_ns: int) -> int | None:
        """Return the immutable deadline for one clean head-decoder miss.

        This signal owns no coordinate and never changes the direct anchor. It
        merely distinguishes a clean decoder miss from ambiguous/mismatched
        localization outcomes before the live adapter considers its separately
        qualified, position-only acquisition proxy.  The deadline is rooted at
        the worker result's source frame, so newer body measurements cannot
        extend authority from an old decoder miss.
        """

        current_ns = int(now_ns)
        if current_ns < 0:
            raise ValueError("head fallback query timestamp cannot be negative")
        source_ns = self._latest_localization_source_timestamp_ns
        source_generation = self._latest_localization_track_generation
        current_generation = self._tracker_generation
        current_body_ns = self._current_player_timestamp_ns
        if (
            not self.body_valid
            or self._body_gap_suspended
            or self._body_update_deferred
            or not self._current_body_observed
            or self._latest_localization_reason
            != "no_decoded_head_candidate"
            or source_ns is None
            or source_generation is None
            or current_generation is None
            or source_generation != current_generation
            or current_body_ns is None
            or current_body_ns < source_ns
        ):
            return None
        age_ns = max(current_ns, current_body_ns) - source_ns
        if not 0 <= age_ns < self.stale_after_ns:
            return None
        return source_ns + self.stale_after_ns

    def body_fallback_no_decoded_verified(self, *, now_ns: int) -> bool:
        """Whether one fresh, accepted crop failed only at head decoding."""

        return self.body_fallback_no_decoded_deadline_ns(now_ns=now_ns) is not None

    def raise_if_failed(self) -> None:
        self.worker.raise_if_failed()

    @property
    def status(self):
        return self.worker.status

    def stop(self) -> bool:
        return bool(self.worker.stop())


def _strict_primary_migraphx_provider(runtime_summary) -> str:
    """Prove the primary model cannot execute or retry on CPU."""

    if not isinstance(runtime_summary, Mapping):
        raise RuntimeError("Primary detector runtime summary is unavailable")
    active_values = runtime_summary.get("active_providers")
    if (
        not isinstance(active_values, (list, tuple))
        or isinstance(active_values, (str, bytes))
    ):
        raise RuntimeError("Primary detector active provider chain is unavailable")
    active = tuple(str(value).strip() for value in active_values)
    allowed_active = (
        (AUTOMATIC_HEAD_PROVIDER,),
        (AUTOMATIC_HEAD_PROVIDER, "CPUExecutionProvider"),
    )
    if active not in allowed_active:
        raise RuntimeError(
            "Automatic direct-head tracking requires the primary detector on "
            f"{AUTOMATIC_HEAD_PROVIDER}; active provider chain is "
            + (", ".join(active) or "unavailable")
        )
    if runtime_summary.get("require_full_provider") is not True:
        raise RuntimeError(
            "Automatic direct-head tracking requires strict primary-provider "
            "qualification"
        )
    configured = runtime_summary.get("configured_session_options")
    if (
        not isinstance(configured, Mapping)
        or configured.get("disable_cpu_ep_fallback") is not True
    ):
        raise RuntimeError(
            "Automatic direct-head tracking requires CPU graph fallback to be "
            "disabled for the primary detector"
        )
    if runtime_summary.get("runtime_ep_fail_fallback_disabled") is not True:
        raise RuntimeError(
            "Automatic direct-head tracking requires primary execution-provider "
            "failure fallback to be disabled"
        )
    return AUTOMATIC_HEAD_PROVIDER


def _build_automatic_head_runtime(primary_runtime_summary) -> _AutomaticHeadRuntime:
    """Build the pinned direct-head model on MIGraphX with no CPU fallback."""

    import numpy as np

    from detection.head_detector import (
        runtime_head_model_spec,
    )
    from detection.head_worker import (
        LatestHeadWorker,
        MIGRAPHX_PROVIDER,
        OnnxModelContract,
        OnnxTensorContract,
        StrictProviderOnnxSession,
    )

    if AUTOMATIC_HEAD_PROVIDER != MIGRAPHX_PROVIDER:
        raise RuntimeError("automatic head provider constant does not match MIGraphX")
    active_primary_provider = _strict_primary_migraphx_provider(
        primary_runtime_summary
    )

    model_spec = runtime_head_model_spec()
    contract = OnnxModelContract(
        input=OnnxTensorContract(
            "images",
            model_spec.input_shape,
        ),
        output=OnnxTensorContract(
            "output0",
            model_spec.output_shape,
        ),
    )
    session = StrictProviderOnnxSession(
        model_spec.path,
        contract,
        provider=active_primary_provider,
    )
    # Pay graph/allocation setup once before any physical activation can arm.
    session.infer(np.zeros(contract.input.shape, dtype=np.float32))
    worker = LatestHeadWorker(
        _PreparedDirectHeadLocalizer(
            session,
            evidence_label=model_spec.evidence_label,
            confidence_threshold=model_spec.confidence_threshold,
        ),
        payload_copier=_identity_payload,
    )
    return _AutomaticHeadRuntime(
        worker,
        tracking_submission_hz=AUTOMATIC_HEAD_TRACKING_LOCALIZATION_HZ,
        provider=session.info.provider,
        model_size=(model_spec.input_height, model_spec.input_width),
        model_name=model_spec.model_name,
        confidence_threshold=model_spec.confidence_threshold,
    )


def _publish_automatic_head_loss_once(
    controller,
    frame_shape,
    *,
    source_timestamp_ns: int,
    already_published: bool,
) -> bool:
    """Publish at most one immediate target-loss decision in one source frame."""

    if already_published:
        return True
    controller.update(
        None,
        frame_shape,
        active=True,
        measurement_ns=source_timestamp_ns,
    )
    return True


def _aim_diagnostic_detection(detection) -> dict[str, object] | None:
    if detection is None:
        return None
    return {
        "class_id": int(detection.class_id),
        "class_name": str(detection.class_name),
        "confidence": float(detection.confidence),
        "box": [float(value) for value in detection.box],
    }


def _aim_diagnostic_head_sample(
    sample: _AutomaticHeadSample | None,
) -> dict[str, object] | None:
    if sample is None:
        return None
    velocity_point = getattr(sample, "velocity_point", None)
    if velocity_point is None:
        velocity_point = sample.point
    return {
        "point": [float(sample.point[0]), float(sample.point[1])],
        "velocity_point": [
            float(velocity_point[0]),
            float(velocity_point[1]),
        ],
        "confidence": float(sample.confidence),
        "source_timestamp_ns": int(sample.source_timestamp_ns),
        "direct_source_timestamp_ns": int(sample.direct_source_timestamp_ns),
        "identity_deadline_ns": int(sample.identity_deadline_ns),
        "track_generation": int(sample.track_generation),
        "provenance": str(sample.provenance.value),
        "bridging": bool(sample.bridging),
        "body_derived_motion_permitted": bool(
            sample.body_derived_motion_permitted
        ),
        "body_derived_motion_deadline_ns": (
            None
            if sample.body_derived_motion_deadline_ns is None
            else int(sample.body_derived_motion_deadline_ns)
        ),
        "motion_corroboration_point": (
            None
            if sample.corroboration_point is None
            else [
                float(sample.corroboration_point[0]),
                float(sample.corroboration_point[1]),
            ]
        ),
        "phase_advanced": bool(sample.phase_advanced),
        "phase_hops": int(sample.phase_hops),
        "verified_flow_motion": bool(
            getattr(sample, "verified_flow_motion", False)
        ),
    }


def _aim_diagnostic_makcu_control(controller) -> dict[str, object] | None:
    """Return passive control/command facts for causal diagnostic replay."""

    normal_snapshot_method = getattr(controller, "normal_control_snapshot", None)
    telemetry_method = getattr(controller, "telemetry_snapshot", None)
    if not callable(normal_snapshot_method):
        return None
    normal = normal_snapshot_method()
    if not is_dataclass(normal):
        raise TypeError("normal_control_snapshot must return a dataclass")
    output = getattr(normal, "calibrated_output", None)
    commands = tuple(getattr(normal, "commands", ()))
    # Detector decisions arrive roughly every 6--12 ms while the output loop
    # runs at 1 kHz.  Retaining the newest 64 sequence-numbered writes in every
    # record gives ample overlap for deduplication even if the asynchronous
    # diagnostic writer drops a queued detector record.
    recent_commands = commands[-64:]
    result: dict[str, object] = {
        "captured_ns": int(getattr(normal, "captured_ns")),
        "connection_epoch": int(getattr(normal, "connection_epoch")),
        "successful_commands": int(getattr(normal, "successful_commands")),
        "emitted_x": int(getattr(normal, "emitted_x")),
        "emitted_y": int(getattr(normal, "emitted_y")),
        "emitted_abs_x": int(getattr(normal, "emitted_abs_x")),
        "emitted_abs_y": int(getattr(normal, "emitted_abs_y")),
        "first_emitted_ns": getattr(normal, "first_emitted_ns"),
        "last_emitted_ns": getattr(normal, "last_emitted_ns"),
        "dropped_commands": int(getattr(normal, "dropped_commands")),
        "recent_commands": [asdict(command) for command in recent_commands],
        "calibrated_output": (
            asdict(output) if output is not None and is_dataclass(output) else None
        ),
    }
    if callable(telemetry_method):
        telemetry = telemetry_method()
        if is_dataclass(telemetry):
            result["cumulative_telemetry"] = asdict(telemetry)
    return result


def _calibration_requested(config: AppConfig) -> bool:
    return config.aim_calibration_evidence is not None


def _active_profile_requested(config: AppConfig) -> bool:
    return config.aim_makcu_active_profile is not None


def _calibrated_controller_from_active_profile(
    profile,
    *,
    max_step: int,
    vertical_rate_ratio: float,
):
    """Build the bounded numeric controller represented by one strict profile."""

    from aiming.makcu_calibrated_control import (
        CalibratedControlConfig,
        CalibratedPlant,
        MakcuCalibratedController,
    )
    from aiming.makcu_calibration_activation import ActiveMakcuCalibrationProfile

    if not isinstance(profile, ActiveMakcuCalibrationProfile):
        raise TypeError("profile must be an ActiveMakcuCalibrationProfile")
    if isinstance(max_step, bool) or not isinstance(max_step, int) or max_step <= 0:
        raise ValueError("calibrated maximum step must be a positive integer")
    ratio = float(vertical_rate_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("calibrated vertical rate ratio must be in (0,1]")
    maximum_rate_x = float(max_step) * 60.0
    maximum_rate_y = maximum_rate_x * ratio
    return MakcuCalibratedController(
        CalibratedPlant(
            profile.fit.x.gain_pixels_per_count,
            profile.fit.y.gain_pixels_per_count,
            profile.fit.delay_seconds,
        ),
        # A calibrated profile supplies the measured plant (gain + delay) but
        # intentionally leaves control tuning at the dataclass defaults unless
        # they are set here.  Align the profiled path with the automatic
        # tuning that was validated against measured live detector jitter:
        # without it, the profiled loop runs deadzone 0.5 px and reacts to
        # sub-pixel noise, which is the residual "shake" felt after an
        # otherwise-correct calibration.  The measured plant keeps the gain
        # exact; these only choose how the loop responds around that gain.
        CalibratedControlConfig(
            position_time_constant_seconds=AUTOMATIC_MAKCU_POSITION_TIME_CONSTANT_SECONDS,
            feedback_deadzone_pixels=AUTOMATIC_MAKCU_FEEDBACK_DEADZONE_PIXELS,
            velocity_median_window=AUTOMATIC_MAKCU_VELOCITY_MEDIAN_WINDOW,
            maximum_velocity_feedforward_fraction=(
                AUTOMATIC_MAKCU_MAX_VELOCITY_FEEDFORWARD_FRACTION
            ),
            maximum_rate_x_counts_per_second=maximum_rate_x,
            maximum_rate_y_counts_per_second=maximum_rate_y,
        ),
    )


def _automatic_direct_head_plant_delay_bounds(
    fit,
) -> tuple[float, float, float]:
    """Return calibrated, effective, and upper plant delays for live pursuit.

    The calibration delay is selected on a detector-cadence grid.  The two
    per-axis quality records retain both the near-equivalent grid ambiguity and
    the largest pulse-local *absolute* deviation from that shared fit.  Those
    values describe symmetric uncertainty around the fitted point; they are not
    evidence that the physical delay lies only above it.  Keep the uncertainty
    available as a diagnostic upper bound, but drive the command ledger with
    the calibrated point estimate itself.  Biasing the ledger upward keeps
    already-landed commands classified as pending and subtracts their motion a
    second time, which directly brakes pursuit.
    """

    from aiming.makcu_calibration import MAX_DELAY_SECONDS

    def finite_nonnegative(value: object, description: str) -> float:
        if isinstance(value, (bool, str, bytes, bytearray)):
            raise ValueError(f"{description} must be finite and nonnegative")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{description} must be finite and nonnegative"
            ) from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{description} must be finite and nonnegative")
        return numeric

    calibrated = finite_nonnegative(
        fit.delay_seconds,
        "automatic direct-head calibrated plant delay",
    )
    uncertainty = max(
        finite_nonnegative(
            fit.x.delay_ambiguity_seconds,
            "automatic direct-head X delay ambiguity",
        ),
        finite_nonnegative(
            fit.y.delay_ambiguity_seconds,
            "automatic direct-head Y delay ambiguity",
        ),
        finite_nonnegative(
            fit.x.pulse_delay_spread_seconds,
            "automatic direct-head X pulse delay spread",
        ),
        finite_nonnegative(
            fit.y.pulse_delay_spread_seconds,
            "automatic direct-head Y pulse delay spread",
        ),
    )
    maximum = finite_nonnegative(
        MAX_DELAY_SECONDS,
        "automatic direct-head maximum plant delay",
    )
    upper = min(calibrated + uncertainty, maximum)
    effective = min(calibrated, maximum)
    return calibrated, effective, upper


def _automatic_measured_direct_head_rate_limits(
    *,
    max_step: int,
    plant,
) -> tuple[float, float]:
    """Resolve per-axis pursuit rates from one measured direct-head plant.

    ``max_step`` remains the ordinary/user-selected legacy-equivalent rate.
    The automatic expansion is enabled only after the user has selected the
    established 320-step direct-head envelope and a bound calibration profile
    supplies measured gains.  A deliberately lower user limit therefore stays
    authoritative, while the ordinary 19.2k-count ceiling no longer makes the
    slower measured Y axis physically unable to realize the observer's
    3,000-pixel/s target-speed contract.
    """

    from aiming.makcu_calibrated_control import CalibratedPlant

    if isinstance(max_step, bool) or not isinstance(max_step, int) or max_step <= 0:
        raise ValueError("automatic maximum step must be a positive integer")
    if not isinstance(plant, CalibratedPlant):
        raise TypeError("automatic measured rate limits require a CalibratedPlant")
    configured_rate = float(max_step) * 60.0
    if max_step < AUTOMATIC_DIRECT_HEAD_RECOMMENDED_MAX_STEP:
        return configured_rate, configured_rate

    def axis_rate(gain_pixels_per_count: float) -> float:
        measured_rate = (
            AUTOMATIC_DIRECT_HEAD_MEASURED_SCREEN_ENVELOPE_PIXELS_PER_SECOND
            / gain_pixels_per_count
        )
        bounded_measured_rate = min(
            measured_rate,
            AUTOMATIC_DIRECT_HEAD_MEASURED_MAX_RATE_COUNTS_PER_SECOND,
        )
        # An explicitly broader user envelope remains authoritative.  This
        # helper only removes an automatic measured-plant shortfall; it never
        # narrows an existing choice.
        return max(configured_rate, bounded_measured_rate)

    return (
        axis_rate(plant.gain_x_pixels_per_count),
        axis_rate(plant.gain_y_pixels_per_count),
    )


def _automatic_plant_aware_controller(
    *,
    max_step: int,
    direct_head: bool = True,
    plant=None,
    maximum_rates_counts_per_second: tuple[float, float] | None = None,
    maximum_feedback_rates_counts_per_second: tuple[float, float] | None = None,
):
    """Build bounded normal control from conservative host response seeds."""

    from aiming.makcu_calibrated_control import (
        CalibratedControlConfig,
        CalibratedPlant,
        MakcuCalibratedController,
    )

    if isinstance(max_step, bool) or not isinstance(max_step, int) or max_step <= 0:
        raise ValueError("automatic maximum step must be a positive integer")
    if not isinstance(direct_head, bool):
        raise TypeError("direct_head must be bool")
    maximum_rate = float(max_step) * 60.0
    if maximum_rates_counts_per_second is None:
        maximum_rate_x = maximum_rate
        maximum_rate_y = maximum_rate
    else:
        if len(maximum_rates_counts_per_second) != 2:
            raise ValueError("automatic maximum rates must contain X and Y")
        maximum_rate_x = float(maximum_rates_counts_per_second[0])
        maximum_rate_y = float(maximum_rates_counts_per_second[1])
        if (
            not math.isfinite(maximum_rate_x)
            or not math.isfinite(maximum_rate_y)
            or maximum_rate_x <= 0.0
            or maximum_rate_y <= 0.0
        ):
            raise ValueError("automatic maximum rates must be finite and positive")
    if maximum_feedback_rates_counts_per_second is None:
        maximum_feedback_rate_x = maximum_rate_x
        maximum_feedback_rate_y = maximum_rate_y
    else:
        if len(maximum_feedback_rates_counts_per_second) != 2:
            raise ValueError("automatic feedback rates must contain X and Y")
        maximum_feedback_rate_x = float(
            maximum_feedback_rates_counts_per_second[0]
        )
        maximum_feedback_rate_y = float(
            maximum_feedback_rates_counts_per_second[1]
        )
        if (
            not math.isfinite(maximum_feedback_rate_x)
            or not math.isfinite(maximum_feedback_rate_y)
            or maximum_feedback_rate_x <= 0.0
            or maximum_feedback_rate_y <= 0.0
            or maximum_feedback_rate_x > maximum_rate_x
            or maximum_feedback_rate_y > maximum_rate_y
        ):
            raise ValueError(
                "automatic feedback rates must be finite, positive, and no "
                "greater than the corresponding maximum rates"
            )
    position_time_constant_seconds = (
        AUTOMATIC_DIRECT_HEAD_POSITION_TIME_CONSTANT_SECONDS
        if direct_head
        else AUTOMATIC_MAKCU_POSITION_TIME_CONSTANT_SECONDS
    )
    feedback_deadzone_pixels = (
        AUTOMATIC_DIRECT_HEAD_FEEDBACK_DEADZONE_PIXELS
        if direct_head
        else AUTOMATIC_MAKCU_FEEDBACK_DEADZONE_PIXELS
    )
    resolved_plant = (
        CalibratedPlant(
            AUTOMATIC_MAKCU_GAIN_X_PIXELS_PER_COUNT,
            AUTOMATIC_MAKCU_GAIN_Y_PIXELS_PER_COUNT,
            AUTOMATIC_MAKCU_DELAY_SECONDS,
        )
        if plant is None
        else plant
    )
    if not isinstance(resolved_plant, CalibratedPlant):
        raise TypeError("automatic plant must be a CalibratedPlant")
    return MakcuCalibratedController(
        resolved_plant,
        CalibratedControlConfig(
            # The automatic profile must remain prompt enough to track moving
            # targets without falling behind. The plant is already delayed by a
            # few milliseconds, so the position loop and deadzone are chosen to
            # be deadbeat against the measured detector jitter.
            #
            # Direct-head uses a continuous deadband and a 28-to-16 ms
            # near-lock/pursuit schedule, so feedback does not create the old
            # 700+ count/s breakaway step at the deadzone edge. Moving-target
            # authority comes from the persistent, covariance-qualified body-
            # motion channel. Its ordinary 50% ceiling remains residual gated;
            # a separate fresh-motion reserve carries fast constant-speed
            # pursuit through zero error and closes on stops, reversals, or a
            # manual approach. Stable-body mode retains the more conservative
            # 40 ms / 3.5 px legacy response.
            velocity_filter_time_constant_seconds=(
                AUTOMATIC_MAKCU_VELOCITY_FILTER_TIME_CONSTANT_SECONDS
            ),
            position_time_constant_seconds=position_time_constant_seconds,
            maximum_target_acceleration_pixels_per_second_squared=(
                AUTOMATIC_MAKCU_MAX_TARGET_ACCELERATION_PIXELS_PER_SECOND_SQUARED
            ),
            maximum_rate_x_counts_per_second=maximum_rate_x,
            maximum_rate_y_counts_per_second=maximum_rate_y,
            maximum_feedback_rate_x_counts_per_second=(
                maximum_feedback_rate_x
            ),
            maximum_feedback_rate_y_counts_per_second=(
                maximum_feedback_rate_y
            ),
            stale_after_seconds=AUTOMATIC_MAKCU_STALE_AFTER_SECONDS,
            maximum_observation_interval_seconds=0.040,
            velocity_median_window=AUTOMATIC_MAKCU_VELOCITY_MEDIAN_WINDOW,
            feedback_deadzone_pixels=feedback_deadzone_pixels,
            continuous_feedback_deadband=direct_head,
            continuous_feedback_shoulder_pixels=(
                AUTOMATIC_DIRECT_HEAD_FEEDBACK_SHOULDER_PIXELS
                if direct_head
                else 0.0
            ),
            pursuit_position_time_constant_seconds=(
                AUTOMATIC_DIRECT_HEAD_PURSUIT_POSITION_TIME_CONSTANT_SECONDS
                if direct_head
                else 0.0
            ),
            pursuit_position_time_constant_start_pixels=(
                AUTOMATIC_DIRECT_HEAD_PURSUIT_START_PIXELS
                if direct_head
                else 0.0
            ),
            pursuit_position_time_constant_full_pixels=(
                AUTOMATIC_DIRECT_HEAD_PURSUIT_FULL_PIXELS
                if direct_head
                else 0.0
            ),
            preserve_pursuit_position_feedback=direct_head,
            maximum_velocity_feedforward_fraction=(
                AUTOMATIC_MAKCU_MAX_VELOCITY_FEEDFORWARD_FRACTION
            ),
            # Direct-head mode publishes a continuously mapped anchor through
            # measured primary geometry. It is not independent corroboration,
            # so predictive motion remains behind the numeric core's persistent-
            # direction, covariance, agreement, and immutable-deadline gates.
            # Once those gates agree, project only across the measured source/
            # plant horizon. The numeric core applies the residual-alignment
            # gate before the bounded body-derived feed-forward can reach the
            # output.
            require_motion_corroboration_for_feedforward=True,
            maximum_body_derived_projection_fraction=(
                AUTOMATIC_MAKCU_MAX_BODY_DERIVED_PROJECTION_FRACTION
            ),
            maximum_body_derived_feedforward_fraction=(
                AUTOMATIC_MAKCU_MAX_BODY_DERIVED_FEEDFORWARD_FRACTION
            ),
            maximum_body_derived_pursuit_feedforward_fraction=(
                (
                    AUTOMATIC_MEASURED_MAKCU_MAX_BODY_DERIVED_PURSUIT_FEEDFORWARD_FRACTION
                    if plant is not None
                    else AUTOMATIC_MAKCU_MAX_BODY_DERIVED_PURSUIT_FEEDFORWARD_FRACTION
                )
                if direct_head
                else 0.0
            ),
            maximum_residual_pursuit_feedforward_fraction=(
                AUTOMATIC_MAKCU_MAX_RESIDUAL_PURSUIT_FEEDFORWARD_FRACTION
                if direct_head
                else 0.0
            ),
            maximum_correlated_lookahead_pursuit_feedforward_fraction=(
                AUTOMATIC_MAKCU_MAX_CORRELATED_LOOKAHEAD_PURSUIT_FEEDFORWARD_FRACTION
                if direct_head
                else 0.0
            ),
            maximum_verified_flow_pursuit_feedforward_fraction=(
                AUTOMATIC_MAKCU_MAX_VERIFIED_FLOW_PURSUIT_FEEDFORWARD_FRACTION
                if direct_head and plant is not None
                else 0.0
            ),
            # Bridge one quantized/ambiguous root only when the complete
            # previous-endpoint -> new-endpoint command-compensated motion
            # still supports the already-proven direction. A true opposite
            # displacement revokes it; raw screen motion caused by our own
            # landed command does not.
            retain_ambiguous_correlated_projection=direct_head,
            additional_body_derived_projection_seconds=0.0,
        ),
    )


def _calibration_model_sha256(snapshot: object) -> str:
    """Return the exact selected-model identity, including OpenVINO weights."""

    from hashlib import sha256
    from typing import Mapping

    if not isinstance(snapshot, Mapping):
        raise ValueError("Model artifact snapshot is malformed")
    digest = snapshot.get("sha256")
    companions = snapshot.get("companions")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(companions, list)
    ):
        raise ValueError("Model artifact snapshot has no verified SHA-256")
    if not companions:
        # ONNX and other single-file models retain their ordinary file hash.
        return digest

    records: list[tuple[str, str]] = []
    for record in [snapshot, *companions]:
        if not isinstance(record, Mapping):
            raise ValueError("Model companion snapshot is malformed")
        name = record.get("name")
        record_digest = record.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(record_digest, str)
            or len(record_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in record_digest
            )
        ):
            raise ValueError("Model companion snapshot has no verified identity")
        records.append((name, record_digest))
    if len({name for name, _digest in records}) != len(records):
        raise ValueError("Model artifact snapshot contains duplicate filenames")

    composite = sha256(b"proaim-model-artifact-set-v1\0")
    for name, record_digest in sorted(records):
        encoded_name = name.encode("utf-8")
        composite.update(len(encoded_name).to_bytes(4, "big"))
        composite.update(encoded_name)
        composite.update(bytes.fromhex(record_digest))
    return composite.hexdigest()


def _calibration_source_identity(
    project_root: Path | None = None,
) -> tuple[str, str]:
    """Bind calibration to one exact frozen build or development source tree."""

    from hashlib import sha256
    import json
    import os
    import re
    import stat
    import subprocess

    root = (project_root or Path(__file__).resolve().parent).resolve()
    if bool(getattr(sys, "frozen", False)):
        build_info = Path(sys.executable).resolve().parent / "BUILD-INFO.json"
        try:
            payload = build_info.read_bytes()
            document = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Calibration requires a valid adjacent BUILD-INFO.json"
            ) from exc
        commit = document.get("commit") if isinstance(document, dict) else None
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{7,64}", commit) is None:
            raise RuntimeError("Calibration build metadata has no exact source commit")
        executable = Path(sys.executable).resolve()
        identity = sha256(b"proaim-frozen-build-v1\0")
        identity.update(payload)
        try:
            with executable.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    identity.update(chunk)
        except OSError as exc:
            raise RuntimeError("Could not hash the calibration executable") from exc
        return commit, f"frozen-executable-sha256:{identity.hexdigest()}"

    def git(*arguments: str) -> bytes:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=False,
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "Calibration requires readable source-control identity"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                "Calibration requires readable source-control identity"
            )
        return completed.stdout

    commit = git("rev-parse", "--verify", "HEAD").decode("ascii", "strict").strip()
    if re.fullmatch(r"[0-9a-f]{7,64}", commit) is None:
        raise RuntimeError("Calibration source commit is invalid")
    diff = git("diff", "--binary", "--no-ext-diff", "HEAD", "--")
    untracked = tuple(
        value.decode("utf-8", "surrogateescape")
        for value in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if value
    )
    if not diff and not untracked:
        return commit, f"source-clean:{commit}"

    identity = sha256(b"proaim-development-tree-v1\0")
    identity.update(commit.encode("ascii"))
    identity.update(b"\0tracked-diff\0")
    identity.update(diff)
    for relative_text in sorted(untracked):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Git reported an unsafe untracked source path")
        candidate = root / relative
        try:
            metadata = os.lstat(candidate)
        except OSError as exc:
            raise RuntimeError(
                "Calibration source changed while its identity was captured"
            ) from exc
        identity.update(b"\0untracked\0")
        identity.update(relative_text.encode("utf-8", "surrogateescape"))
        identity.update(b"\0")
        identity.update(str(stat.S_IMODE(metadata.st_mode)).encode("ascii"))
        identity.update(b"\0")
        if stat.S_ISLNK(metadata.st_mode):
            identity.update(os.readlink(candidate).encode("utf-8", "surrogateescape"))
        elif stat.S_ISREG(metadata.st_mode):
            try:
                with candidate.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        identity.update(chunk)
            except OSError as exc:
                raise RuntimeError(
                    "Calibration source changed while its identity was captured"
                ) from exc
        else:
            raise RuntimeError("Calibration source contains an unsupported untracked entry")
    return commit, f"source-tree-sha256:{identity.hexdigest()}"


def _positive_integral_runtime_value(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Calibration {name} is not a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Calibration {name} is unavailable") from exc
    if not math.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"Calibration {name} is not a positive integer")
    return int(numeric)


def _calibration_capture_identity(
    config: AppConfig,
    settings: object,
) -> tuple[str, str, int, str, int, int, float, str, int]:
    """Extract only negotiated live-capture identity from the running source."""

    import json
    from typing import Mapping

    if not isinstance(settings, Mapping):
        raise ValueError("Calibration capture settings are unavailable")
    if config.source.kind not in {"screen", "device"}:
        raise ValueError("Calibration requires a live screen or capture-card source")
    width = _positive_integral_runtime_value(settings.get("width"), "capture width")
    height = _positive_integral_runtime_value(settings.get("height"), "capture height")
    try:
        fps = float(settings.get("fps"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration capture frame rate is unavailable") from exc
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("Calibration capture frame rate must be positive")
    try:
        rotation = int(settings.get("rotation_degrees", 180 if config.capture_rotate_180 else 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration capture rotation is unavailable") from exc
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Calibration capture rotation is invalid")

    if config.source.kind == "device":
        backend_value = settings.get("backend")
        if not isinstance(backend_value, str) or not backend_value.strip():
            raise ValueError("Calibration capture-card backend is unavailable")
        capture_backend = backend_value.strip()
        capture_buffer_size = _positive_integral_runtime_value(
            settings.get("buffer_size"), "capture buffer size"
        )
        actual_source = settings.get("source", config.source.value)
        if actual_source is None:
            raise ValueError("Calibration capture-card index is unavailable")
        capture_index = str(actual_source)
        pixel_format_value = settings.get("pixel_format")
        if not isinstance(pixel_format_value, str) or not pixel_format_value.strip():
            raise ValueError("Calibration capture-card pixel format is unavailable")
        pixel_format = pixel_format_value.strip().upper()
        capture_kind = "camera"
    else:
        # Screen backends expose different adapter/output fields.  Canonical
        # JSON retains their actual selection without pretending they share an
        # integer index namespace.
        index_record = {
            key: settings[key]
            for key in (
                "backend",
                "device_index",
                "left",
                "monitor",
                "output_index",
                "top",
            )
            if key in settings and settings[key] is not None
        }
        capture_index = json.dumps(
            index_record,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        pixel_format = str(settings.get("pixel_format") or "BGR8").strip().upper()
        capture_backend = str(settings.get("backend") or "").strip()
        if not capture_backend:
            raise ValueError("Calibration screen-capture backend is unavailable")
        capture_buffer_size = 0
        capture_kind = "screen"
    if not capture_index or len(capture_index) > 256:
        raise ValueError("Calibration capture identity is unavailable")
    return (
        capture_kind,
        capture_backend,
        capture_buffer_size,
        capture_index,
        width,
        height,
        fps,
        pixel_format,
        rotation,
    )


def _build_calibration_runtime_binding(
    config: AppConfig,
    *,
    detector_summary: object,
    capture_settings: object,
    makcu_identity_token: object,
    model_artifact_snapshot: object,
    labels_artifact_snapshot: object,
    source_identity: tuple[str, str] | None = None,
    context_name: str | None = None,
    aim_mode: str | None = None,
):
    """Create the exact immutable identity used by evidence and activation."""

    from typing import Mapping

    from aiming.makcu_calibration_session import (
        CalibrationRuntimeBinding,
        normalize_calibration_context,
    )
    from hashlib import sha256
    import json
    import os
    import re

    if not isinstance(detector_summary, Mapping):
        raise ValueError("Calibration detector runtime summary is unavailable")
    if not isinstance(labels_artifact_snapshot, Mapping):
        raise ValueError("Labels artifact snapshot is malformed")
    labels_sha256 = labels_artifact_snapshot.get("sha256")
    if (
        not isinstance(labels_sha256, str)
        or len(labels_sha256) != 64
        or any(character not in "0123456789abcdef" for character in labels_sha256)
    ):
        raise ValueError("Labels artifact snapshot has no verified SHA-256")
    if not isinstance(makcu_identity_token, str) or len(makcu_identity_token) != 64:
        raise RuntimeError("Verified MAKCU identity is unavailable for calibration")

    shape = detector_summary.get("input_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise ValueError("Calibration detector input shape is unavailable")
    inference_height = _positive_integral_runtime_value(shape[-2], "inference height")
    inference_width = _positive_integral_runtime_value(shape[-1], "inference width")
    active_values = detector_summary.get("active_providers")
    active_provider = ""
    if isinstance(active_values, (list, tuple)) and active_values:
        active_provider = str(active_values[0]).strip()
    if not active_provider:
        active_provider = str(detector_summary.get("device") or "").strip()
    requested_provider = str(
        detector_summary.get("requested_provider")
        or detector_summary.get("requested_device")
        or config.device
    ).strip()
    active_device = str(detector_summary.get("device") or active_provider).strip()
    if not requested_provider or not active_provider or not active_device:
        raise ValueError("Calibration detector provider/device identity is incomplete")

    runtime_version = str(
        detector_summary.get("onnxruntime_version")
        or detector_summary.get("openvino_version")
        or ""
    ).strip()
    if runtime_version.casefold() in {"", "unknown", "unavailable", "n/a", "none"}:
        raise ValueError("Calibration runtime version identity is unavailable")
    provider_identity_record = {
        key: detector_summary.get(key)
        for key in (
            "active_providers",
            "configured_session_options",
            "execution_devices",
            "num_streams",
            "num_streams_requested",
            "output_format",
            "performance_hint",
            "provider_chain",
            "provider_option_overrides",
            "provider_options",
            "require_full_provider",
            "runtime_ep_fail_fallback_disabled",
        )
        if key in detector_summary
    }
    if config.backend == "onnxruntime":
        if detector_summary.get("provider_options_status") != "ok":
            raise ValueError(
                "Calibration requires a successful ONNX provider-options query"
            )
        if (
            detector_summary.get("require_full_provider") is not True
            or detector_summary.get("runtime_ep_fail_fallback_disabled") is not True
        ):
            raise ValueError(
                "Calibration requires strict ONNX GPU provider fallback controls"
            )
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION") is not None:
            raise ValueError("Calibration refuses HSA_OVERRIDE_GFX_VERSION")
        provider_identity_record["runtime_environment"] = {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("MIGRAPHX_", "ROCR_", "HIP_", "HSA_"))
            or key in {"CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}
        }
    try:
        provider_identity_bytes = json.dumps(
            provider_identity_record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Calibration provider options are not canonically serializable"
        ) from exc
    provider_options_sha256 = sha256(provider_identity_bytes).hexdigest()

    physical_identity = str(
        detector_summary.get("physical_device_identity") or ""
    ).strip()
    if not physical_identity and any(
        token in active_provider.casefold() for token in ("migraphx", "rocm")
    ):
        rocr_selector = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
        if re.fullmatch(r"GPU-[0-9A-Fa-f]{16,64}", rocr_selector):
            physical_identity = f"rocr:{rocr_selector}"
    if (
        not physical_identity
        or len(physical_identity) > 256
        or any(ord(character) < 0x20 for character in physical_identity)
    ):
        raise ValueError(
            "Calibration requires an unambiguous physical accelerator identity"
        )
    physical_device_token = sha256(physical_identity.encode("utf-8")).hexdigest()

    (
        capture_kind,
        capture_backend,
        capture_buffer_size,
        capture_index,
        capture_width,
        capture_height,
        capture_fps,
        pixel_format,
        rotation_degrees,
    ) = _calibration_capture_identity(config, capture_settings)
    source_commit, build_identity = source_identity or _calibration_source_identity()
    selected_context = context_name or config.aim_calibration_context
    normalized_mode = normalize_calibration_context(selected_context)
    selected_mode = aim_mode or normalized_mode
    if selected_mode != normalized_mode:
        raise ValueError("Calibration context and aim mode do not match")
    return CalibrationRuntimeBinding(
        model_sha256=_calibration_model_sha256(model_artifact_snapshot),
        labels_sha256=labels_sha256,
        source_commit=source_commit,
        build_identity=build_identity,
        backend=config.backend,
        runtime_version=runtime_version,
        requested_provider=requested_provider,
        active_provider=active_provider,
        active_device=active_device,
        provider_options_sha256=provider_options_sha256,
        physical_device_token=physical_device_token,
        inference_width=inference_width,
        inference_height=inference_height,
        detail_pass_enabled=False,
        capture_kind=capture_kind,
        capture_backend=capture_backend,
        capture_buffer_size=capture_buffer_size,
        capture_index=capture_index,
        capture_width=capture_width,
        capture_height=capture_height,
        capture_fps=capture_fps,
        pixel_format=pixel_format,
        rotation_degrees=rotation_degrees,
        makcu_identity_token=makcu_identity_token,
        activation_button=config.aim_makcu_button,
        aim_label=config.aim_label or "",
        head_ratio=config.aim_head_ratio,
        invert_x=config.aim_invert_x,
        invert_y=config.aim_invert_y,
        context_name=selected_context,
        aim_mode=selected_mode,
    )


def _calibration_observation_target_and_readiness(
    detections,
    frame_shape: tuple[int, ...],
    *,
    aim_label: str,
    head_ratio: float,
    configured_confidence: float,
    invert_x: bool,
    invert_y: bool,
    self_exclusion_safe: bool,
    measurement_ns: int,
    previous_normalized_bbox: tuple[float, float, float, float] | None = None,
    safe_roi_margin_ratio: float | None = None,
    maximum_reference_error: float | None = None,
):
    """Select one raw exact-label target and explain its readiness."""

    from aiming import head_target_point
    from aiming.makcu_calibration_session import (
        CalibrationObservation,
        target_within_safe_roi,
    )

    if len(frame_shape) < 2:
        return None, None, "target wait: capture dimensions are unavailable"
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return None, None, "target wait: capture dimensions are invalid"
    threshold = float(configured_confidence)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Calibration confidence must be finite and in [0,1]")
    normalized_label = aim_label.strip().casefold()
    exact_label_count = 0
    eligible: list[
        tuple[
            object,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            tuple[float, float, float, float],
        ]
    ] = []
    for detection in detections:
        if detection.class_name.strip().casefold() != normalized_label:
            continue
        exact_label_count += 1
        try:
            confidence = float(detection.confidence)
            left, top, right, bottom = (float(value) for value in detection.box)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence) or confidence < threshold:
            continue
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            continue
        if not (0.0 <= left < right <= width and 0.0 <= top < bottom <= height):
            continue
        target_x, target_y = head_target_point(detection, head_ratio)
        if not math.isfinite(target_x) or not math.isfinite(target_y):
            continue
        eligible.append(
            (
                detection,
                confidence,
                left,
                top,
                right,
                bottom,
                target_x,
                target_y,
                (
                    left / width,
                    top / height,
                    right / width,
                    bottom / height,
                ),
            )
        )
    if not eligible:
        if exact_label_count:
            readiness = (
                "target wait: no valid exact "
                f"{aim_label} detection at confidence >= {threshold:g}"
            )
        else:
            readiness = f"target wait: no exact {aim_label} detection"
        return None, None, readiness

    center_x = width / 2.0
    center_y = height / 2.0
    reference_scale_x = 1920.0 / width
    reference_scale_y = 1080.0 / height
    continuity_min_iou = 0.10
    continuity_max_jump_reference_pixels = 220.0

    def bbox_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        union = first_area + second_area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    def target_key(candidate):
        (
            detection,
            confidence,
            left,
            top,
            right,
            bottom,
            target_x,
            target_y,
            normalized_bbox,
        ) = candidate
        reference_x = (target_x - center_x) * reference_scale_x
        reference_y = (target_y - center_y) * reference_scale_y
        continuity_priority = 1.0
        if previous_normalized_bbox is not None:
            continuity_priority = -bbox_iou(normalized_bbox, previous_normalized_bbox)
        # Distance is the intent signal. The remaining fields make equal-distance
        # selection stable even if the detector returns boxes in another order.
        return (
            continuity_priority,
            reference_x * reference_x + reference_y * reference_y,
            -confidence,
            left,
            top,
            right,
            bottom,
            int(detection.class_id),
        )

    continuity_locked = False
    if previous_normalized_bbox is not None:
        previous_center_x = ((previous_normalized_bbox[0] + previous_normalized_bbox[2]) * 0.5) * width
        previous_center_y = ((previous_normalized_bbox[1] + previous_normalized_bbox[3]) * 0.5) * height

        def continuity_key(candidate):
            (
                detection,
                confidence,
                left,
                top,
                right,
                bottom,
                target_x,
                target_y,
                normalized_bbox,
            ) = candidate
            iou = bbox_iou(normalized_bbox, previous_normalized_bbox)
            delta_x = (target_x - previous_center_x) * reference_scale_x
            delta_y = (target_y - previous_center_y) * reference_scale_y
            return (
                -iou,
                delta_x * delta_x + delta_y * delta_y,
                -confidence,
                left,
                top,
                right,
                bottom,
                int(detection.class_id),
            )

        continuity_candidate = min(eligible, key=continuity_key)
        continuity_iou = -continuity_key(continuity_candidate)[0]
        continuity_distance_squared = continuity_key(continuity_candidate)[1]
        maximum_jump_squared = (
            continuity_max_jump_reference_pixels
            * continuity_max_jump_reference_pixels
        )
        if (
            continuity_iou < continuity_min_iou
            and continuity_distance_squared > maximum_jump_squared
        ):
            return (
                None,
                None,
                f"target wait: reacquire the same exact {aim_label} target",
            )
        target_candidate = continuity_candidate
        continuity_locked = len(eligible) > 1
    else:
        target_candidate = min(eligible, key=target_key)

    (
        target,
        confidence,
        left,
        top,
        right,
        bottom,
        target_x,
        target_y,
        normalized_bbox,
    ) = target_candidate
    reference_x = (target_x - width / 2.0) * (1920.0 / width)
    reference_y = (target_y - height / 2.0) * (1080.0 / height)
    if invert_x:
        reference_x = -reference_x
    if invert_y:
        reference_y = -reference_y
    observation = CalibrationObservation(
        measurement_ns=measurement_ns,
        error_x=reference_x,
        error_y=reference_y,
        confidence=confidence,
        exact_label=True,
        # Deterministic arbitration turns all safe, eligible full-pass boxes
        # into one authorized candidate. The session still independently
        # enforces central ROI and frame-to-frame box continuity.
        unique_candidates=1,
        self_safe=True,
        is_prediction=False,
        target_identity="selected-exact-target",
        normalized_bbox=normalized_bbox,
    )
    if safe_roi_margin_ratio is not None and not target_within_safe_roi(
        observation.normalized_bbox,
        safe_roi_margin_ratio,
    ):
        readiness = "target wait: keep the complete target inside the central guide"
    elif maximum_reference_error is not None and max(
        abs(observation.error_x),
        abs(observation.error_y),
    ) > float(maximum_reference_error):
        readiness = "target wait: move the target aim point closer to the crosshair"
    else:
        readiness = (
            "target ready"
            if len(eligible) == 1
            else (
                f"target ready: continuity-locked of {len(eligible)} exact detections"
                if continuity_locked
                else f"target ready: center-nearest of {len(eligible)} exact detections"
            )
        )
    return observation, target, readiness


def _calibration_observation_and_target(
    detections,
    frame_shape: tuple[int, ...],
    *,
    aim_label: str,
    head_ratio: float,
    configured_confidence: float,
    invert_x: bool,
    invert_y: bool,
    self_exclusion_safe: bool,
    measurement_ns: int,
    safe_roi_margin_ratio: float | None = None,
    maximum_reference_error: float | None = None,
):
    """Return the deterministically authorized raw calibration target."""

    observation, target, _readiness = _calibration_observation_target_and_readiness(
        detections,
        frame_shape,
        aim_label=aim_label,
        head_ratio=head_ratio,
        configured_confidence=configured_confidence,
        invert_x=invert_x,
        invert_y=invert_y,
        self_exclusion_safe=self_exclusion_safe,
        measurement_ns=measurement_ns,
        safe_roi_margin_ratio=safe_roi_margin_ratio,
        maximum_reference_error=maximum_reference_error,
    )
    return observation, target


def _partition_detections_by_confidence(
    detections,
    configured_confidence: float | None,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Separate normal detections from aim-only continuation evidence."""

    source = tuple(detections)
    if configured_confidence is None:
        return source, ()
    normal: list[object] = []
    continuation: list[object] = []
    for detection in source:
        destination = (
            normal
            if float(detection.confidence) >= configured_confidence
            else continuation
        )
        destination.append(detection)
    return tuple(normal), tuple(continuation)


@dataclass(frozen=True, slots=True)
class HardAimGuardResult:
    """Aim-only detections and exact-label attribution for the hard self guard."""

    detections: tuple[object, ...]
    removed_exact_label_boxes: int = 0
    targetless_after_exact_removal: bool = False
    removed_detections: tuple[object, ...] = ()


def _apply_hard_aim_guard(
    detections,
    frame_shape: tuple[int, ...],
    *,
    self_zone,
    aim_label: str,
    configured_confidence: float | None = None,
    confirmed_self_detection=None,
    unconfirmed_zone_guard: bool = True,
    obvious_bottom_shoulder_guard: bool = False,
) -> HardAimGuardResult:
    """Apply the existing conservative aim guard and attribute exact labels.

    The guard remains player-like rather than label-specific.  Attribution is
    exact-label-specific so an unrelated guarded box cannot revoke prediction
    grace for a genuinely empty target-label sample.
    """

    from utils.self_filter import (
        boxes_are_safely_distinct,
        is_obvious_bottom_shoulder_avatar,
        is_player_like,
    )

    source = tuple(detections)
    if self_zone is None or not source:
        return HardAimGuardResult(source)
    if not isinstance(unconfirmed_zone_guard, bool):
        raise TypeError("unconfirmed_zone_guard must be bool")
    if not isinstance(obvious_bottom_shoulder_guard, bool):
        raise TypeError("obvious_bottom_shoulder_guard must be bool")

    normalized_aim_label = aim_label.strip().lower()
    if configured_confidence is not None:
        configured_confidence = float(configured_confidence)
        if not math.isfinite(configured_confidence) or not (
            0.0 <= configured_confidence <= 1.0
        ):
            raise ValueError(
                "configured_confidence must be finite and between 0 and 1"
            )
    guarded: list[object] = []
    removed: list[object] = []
    removed_exact_label_boxes = 0
    obvious_self_references = (
        tuple(
            detection
            for detection in source
            if is_obvious_bottom_shoulder_avatar(
                detection,
                frame_shape,
                self_zone,
            )
        )
        if obvious_bottom_shoulder_guard
        else ()
    )
    for detection in source:
        zone_candidate = (
            is_player_like(detection)
            and self_zone.candidate_score(detection.box, frame_shape) is not None
        )
        # Once the temporal filter has positively removed one confirmed self
        # box, broad zone membership alone is no longer enough to call every
        # remaining player "self".  Keep a distinct opponent, while still
        # suppressing overlapping/associated duplicate detections of the
        # confirmed avatar.  Invalid geometry remains guarded fail-closed.
        confirmed_self_related = bool(
            confirmed_self_detection is not None
            and is_player_like(detection)
            and not boxes_are_safely_distinct(
                confirmed_self_detection.box,
                detection.box,
                frame_shape,
            )
        )
        obvious_self_related = bool(
            is_player_like(detection)
            and any(
                not boxes_are_safely_distinct(
                    reference.box,
                    detection.box,
                    frame_shape,
                )
                for reference in obvious_self_references
            )
        )
        drop_for_aim = bool(
            confirmed_self_related
            or obvious_self_related
            or (
                unconfirmed_zone_guard
                and zone_candidate
                and confirmed_self_detection is None
            )
        )
        if drop_for_aim:
            removed.append(detection)
            if detection.class_name.strip().lower() == normalized_aim_label:
                removed_exact_label_boxes += 1
            continue
        guarded.append(detection)

    targetless_after_exact_removal = (
        removed_exact_label_boxes > 0
        and not any(
            detection.class_name.strip().lower() == normalized_aim_label
            and (
                configured_confidence is None
                or float(detection.confidence) >= configured_confidence
            )
            for detection in guarded
        )
    )
    return HardAimGuardResult(
        tuple(guarded),
        removed_exact_label_boxes=removed_exact_label_boxes,
        targetless_after_exact_removal=targetless_after_exact_removal,
        removed_detections=tuple(removed),
    )


def _aim_detections_safely_distinct_from_uncertain_self(
    detections,
    frame_shape: tuple[int, ...],
    *,
    uncertain_self_detections,
    aim_label: str,
    configured_confidence: float,
) -> tuple[object, ...]:
    """Return only target evidence proven separate from every possible self.

    ``SelfAvatarFilter`` deliberately keeps preview detections visible while
    its temporal identity is ambiguous.  A global ``aim_safe=False`` used to
    throw away a clearly separate opponent as well, resetting both the target
    and direct-head identity several times per second.  This aim-only boundary
    remains fail-closed for every ambiguous avatar and overlapping fragment;
    it opens only when a current, configured-confidence exact-label survivor
    is geometrically disjoint from *all* uncertain self identities.
    """

    from utils.self_filter import boxes_are_safely_distinct

    source = tuple(detections)
    uncertain = tuple(uncertain_self_detections)
    if not source or not uncertain:
        return ()
    threshold = float(configured_confidence)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("configured confidence must be finite and in [0,1]")
    normalized_label = str(aim_label).strip().casefold()
    if not normalized_label:
        raise ValueError("aim label must not be empty")
    distinct = tuple(
        detection
        for detection in source
        if all(
            boxes_are_safely_distinct(
                possible_self.box,
                detection.box,
                frame_shape,
            )
            for possible_self in uncertain
        )
    )
    has_current_target_authority = any(
        str(getattr(detection, "class_name", "")).strip().casefold()
        == normalized_label
        and float(getattr(detection, "confidence", float("-inf"))) >= threshold
        for detection in distinct
    )
    return distinct if has_current_target_authority else ()


def _verified_flow_continuation_cluster(
    detections,
    frame_shape: tuple[int, ...],
    *,
    previous_player,
    verified_head_point: tuple[float, float] | None,
    aim_label: str,
    confidence_floor: float,
    self_zone,
) -> tuple[object, ...]:
    """Protect only the exact opponent cluster proven by current head pixels.

    A close opponent can enter the configured bottom self zone and make the
    heuristic self filter ambiguous.  Cold acquisition remains fail-closed.
    During an already measured direct-head lease, however, LK may have moved
    that same head into this exact captured frame.  Retain only detections that
    both associate with the preceding measured player and anatomically contain
    that exact-frame pixel endpoint.  The stricter obvious-bottom avatar guard
    still has final authority.
    """

    if previous_player is None or verified_head_point is None:
        return ()
    threshold = float(confidence_floor)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("continuation confidence floor must be in [0,1]")
    normalized_label = str(aim_label).strip().casefold()
    if not normalized_label:
        raise ValueError("aim label must not be empty")
    associated = tuple(
        detection
        for detection in detections
        if (
            str(getattr(detection, "class_name", "")).strip().casefold()
            == normalized_label
            and float(getattr(detection, "confidence", float("-inf")))
            >= threshold
            and _AutomaticHeadRuntime._player_boxes_associate(
                previous_player.box,
                detection.box,
            )
            and _AutomaticHeadRuntime._head_point_belongs_to_player(
                verified_head_point,
                detection.box,
            )
        )
    )
    if not associated:
        return ()
    guarded = _apply_hard_aim_guard(
        associated,
        frame_shape,
        self_zone=self_zone,
        aim_label=aim_label,
        configured_confidence=threshold,
        confirmed_self_detection=None,
        unconfirmed_zone_guard=False,
        obvious_bottom_shoulder_guard=True,
    )
    return guarded.detections


def _exclude_automatic_detail_self_relatives(
    detections,
    frame_shape: tuple[int, ...],
    *,
    self_references,
) -> tuple[object, ...]:
    """Remove detail-pass player fragments tied to guarded self geometry.

    A centered detail crop can truncate the bottom-anchored avatar and return
    only a small upper-body fragment.  That fragment is too small for the
    ordinary self-candidate geometry, so compare every player-like detail
    result with the full-pass boxes already guarded as self (and the preceding
    confirmed self box).  Only a box proven spatially distinct may enter the
    cross-pass merge.  This helper is stateless: ``SelfAvatarFilter.apply``
    remains the sole temporal transition for each source frame.
    """

    from utils.self_filter import boxes_are_safely_distinct, is_player_like

    references = tuple(self_references)
    if not references:
        return tuple(detections)
    retained: list[object] = []
    for detection in detections:
        if is_player_like(detection) and any(
            not boxes_are_safely_distinct(
                reference.box,
                detection.box,
                frame_shape,
            )
            for reference in references
        ):
            continue
        retained.append(detection)
    return tuple(retained)


def _exclude_automatic_detail_lower_edge_self_fragments(
    detections,
    primary_detections,
    frame_shape: tuple[int, ...],
    *,
    detail_plan,
    self_zone,
) -> tuple[object, ...]:
    """Reject unmatched player fragments clipped by the detail ROI's bottom.

    The centered 768 ROI ends above the configured bottom self zone at 1080p.
    If the full pass misses the avatar entirely, its cropped upper body can
    therefore look like a small, otherwise safe player.  Bound this cold-start
    case to player-like detail-only boxes whose bottom lies within four model
    pixels of the lower ROI edge and whose center lies in either configured
    shoulder band.  A same-class overlapping full-pass box proves the result
    is a refinement rather than detail-only evidence and keeps it eligible.
    """

    from utils.self_filter import boxes_are_safely_distinct, is_player_like

    source_height = int(frame_shape[0])
    source_width = int(frame_shape[1])
    if source_height <= 0 or source_width <= 0:
        raise ValueError("frame dimensions must be positive")
    crop_bottom = float(detail_plan.crop_y + detail_plan.applied_crop_height)
    source_pixels_per_model_pixel = (
        float(detail_plan.applied_crop_height) / float(detail_plan.model_height)
    )
    lower_edge_threshold = crop_bottom - (
        AUTOMATIC_DETAIL_SELF_EDGE_MARGIN_MODEL_PIXELS
        * source_pixels_per_model_pixel
    )
    configured_left = float(self_zone.left) * source_width
    configured_right = float(self_zone.left + self_zone.width) * source_width
    mirrored_left = float(1.0 - self_zone.left - self_zone.width) * source_width
    mirrored_right = float(1.0 - self_zone.left) * source_width
    primary = tuple(primary_detections)

    retained: list[object] = []
    for detection in detections:
        if not is_player_like(detection):
            retained.append(detection)
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in detection.box)
        except (TypeError, ValueError):
            # Malformed player geometry cannot safely establish aim authority.
            continue
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        center_x = (x1 + x2) * 0.5
        in_self_shoulder_band = bool(
            configured_left <= center_x <= configured_right
            or mirrored_left <= center_x <= mirrored_right
        )
        touches_lower_detail_edge = bool(
            y1 < crop_bottom and y2 >= lower_edge_threshold
        )
        has_full_pass_parent = any(
            is_player_like(candidate)
            and candidate.class_id == detection.class_id
            and not boxes_are_safely_distinct(
                candidate.box,
                detection.box,
                frame_shape,
            )
            for candidate in primary
        )
        if (
            in_self_shoulder_band
            and touches_lower_detail_edge
            and not has_full_pass_parent
        ):
            continue
        retained.append(detection)
    return tuple(retained)


@dataclass(frozen=True, slots=True)
class AimInputTelemetrySnapshot:
    samples: int = 0
    exact_label_samples: int = 0
    self_filter_unsafe_samples: int = 0
    hard_guard_removed_exact_boxes: int = 0
    hard_guard_targetless_samples: int = 0


class AimInputTelemetry:
    """Cumulative non-spatial counters explaining missing aim inputs."""

    def __init__(self, aim_label: str) -> None:
        self._normalized_aim_label = aim_label.strip().lower()
        self._samples = 0
        self._exact_label_samples = 0
        self._self_filter_unsafe_samples = 0
        self._hard_guard_removed_exact_boxes = 0
        self._hard_guard_targetless_samples = 0

    def record_sample(self, detections) -> None:
        self._samples += 1
        if any(
            detection.class_name.strip().lower() == self._normalized_aim_label
            for detection in detections
        ):
            self._exact_label_samples += 1

    def record_self_filter(self, *, aim_safe: bool) -> None:
        if not aim_safe:
            self._self_filter_unsafe_samples += 1

    def record_hard_guard(self, result: HardAimGuardResult) -> None:
        self._hard_guard_removed_exact_boxes += result.removed_exact_label_boxes
        if result.targetless_after_exact_removal:
            self._hard_guard_targetless_samples += 1

    def snapshot(self) -> AimInputTelemetrySnapshot:
        return AimInputTelemetrySnapshot(
            samples=self._samples,
            exact_label_samples=self._exact_label_samples,
            self_filter_unsafe_samples=self._self_filter_unsafe_samples,
            hard_guard_removed_exact_boxes=self._hard_guard_removed_exact_boxes,
            hard_guard_targetless_samples=self._hard_guard_targetless_samples,
        )


def _build_capture(config: AppConfig):
    from capture import DesktopCaptureSource, OpenCVCaptureSource

    if config.source.kind == "screen":
        return DesktopCaptureSource(
            monitor=config.screen_monitor,
            region=config.screen_region,
            fps=config.screen_fps,
        )

    width = config.capture_size[0] if config.capture_size else None
    height = config.capture_size[1] if config.capture_size else None
    if config.source.kind == "device":
        assert isinstance(config.source.value, int)
        return OpenCVCaptureSource(
            config.source.value,
            width=width,
            height=height,
            fps=config.capture_fps,
            # High-rate UVC devices require at least double buffering to keep
            # one transfer in flight while the previous frame is consumed.
            # The source still publishes into a one-frame latest-only mailbox,
            # so this does not create an application-side stale-frame queue.
            buffer_size=2,
            pixel_format=config.capture_format,
            rotate_180=config.capture_rotate_180,
        )

    assert isinstance(config.source.value, Path)
    if not config.source.value.is_file():
        raise FileNotFoundError(f"Video file not found: {config.source.value}")
    return OpenCVCaptureSource(
        config.source.value,
        rotate_180=config.capture_rotate_180,
    )


def _print_startup(detector, source) -> None:
    summary = detector.runtime_summary
    # Either backend may be running here, so report whichever one built this
    # detector rather than assuming OpenVINO.
    runtime = summary.get("runtime", "OpenVINO")
    version = summary.get("openvino_version") or summary.get(
        "onnxruntime_version", "unknown"
    )
    print(f"{runtime} {version}")
    devices = ", ".join(detector.available_devices) or "none"
    print(f"Detected {runtime} devices: {devices}")
    requested = summary.get("requested_device")
    active_device = summary.get("device")
    if requested and active_device and requested != active_device:
        inference_label = f"{active_device} (requested {requested})"
    else:
        inference_label = active_device
    hint = summary.get("performance_hint") or ", ".join(
        summary.get("active_providers", ())
    )
    print(
        f"Inference: {inference_label} | input {summary.get('input_shape')} | "
        f"hint {hint} | one synchronous request"
    )
    print(f"Model: {summary.get('model_path')}")
    print(f"Source: {source.description}")


def _warn_on_capture_mismatch(config: AppConfig, settings) -> None:
    """Say so when the driver refused the requested format or frame rate.

    Capture properties are hints.  A card asked for a mode it cannot provide
    silently falls back, and the usual symptom is a frame rate far below the
    advertised one, so the difference is worth stating rather than leaving the
    user to infer it from the numbers.
    """

    requested_format = settings.get("requested_pixel_format")
    granted_format = settings.get("pixel_format")
    if requested_format and granted_format and requested_format != granted_format:
        print(
            f"Warning: requested pixel format {requested_format} but the device "
            f"is running {granted_format}. The frame rate is limited by the "
            f"format the driver actually granted.",
            file=sys.stderr,
        )

    requested_fps = config.capture_fps
    granted_fps = settings.get("fps")
    if (
        requested_fps
        and granted_fps
        # Drivers routinely round; only a real shortfall is worth a warning.
        and granted_fps < requested_fps * 0.9
    ):
        print(
            f"Warning: requested {requested_fps:g} fps but the device reports "
            f"{granted_fps:g} fps. Uncompressed modes such as NV12 and YUY2 are "
            f"limited by USB bandwidth; MJPG usually reaches higher rates.",
            file=sys.stderr,
        )


def _format_settings(settings) -> str:
    return ", ".join(f"{key}={value}" for key, value in settings.items() if value is not None)


def _target_tracker_telemetry_summary(previous, current, elapsed_seconds: float) -> str:
    """Format aggregate raw-measurement versus tracker-output diagnostics."""

    elapsed = max(float(elapsed_seconds), 1e-9)

    def count_delta(name: str) -> int:
        return max(0, int(getattr(current, name)) - int(getattr(previous, name)))

    def total_delta(name: str) -> float:
        return float(getattr(current, name)) - float(getattr(previous, name))

    updates = count_delta("updates")
    candidates = count_delta("candidate_samples")
    measurements = count_delta("measurement_samples")
    continuation_measurements = count_delta("continuation_measurement_samples")
    outputs = count_delta("output_samples")
    compared = count_delta("compared_samples")

    def mean(name: str, *, absolute: bool = False) -> float:
        value = total_delta(name)
        if absolute:
            value = max(0.0, value)
        return value / compared if compared else 0.0

    rejected = max(0, candidates - measurements)
    return (
        f"TRACK samples {updates / elapsed:.0f}/s | "
        f"raw/out {measurements / elapsed:.0f}/{outputs / elapsed:.0f}/s | "
        f"continued-low {continuation_measurements / elapsed:.0f}/s | "
        f"rejected {rejected / elapsed:.0f}/s | "
        f"raw-track abs X/Y "
        f"{mean('residual_abs_x', absolute=True):.1f}/"
        f"{mean('residual_abs_y', absolute=True):.1f}px | "
        f"signed {mean('residual_x'):+.1f}/{mean('residual_y'):+.1f}px | "
        f"losses {count_delta('target_loss_transitions')}"
    )


def _aim_input_telemetry_summary(previous, current, elapsed_seconds: float) -> str:
    """Format per-report deltas for non-spatial aim input causes."""

    elapsed = max(float(elapsed_seconds), 1e-9)

    def delta(name: str) -> int:
        return max(0, int(getattr(current, name)) - int(getattr(previous, name)))

    return (
        f"AIM INPUT {delta('samples') / elapsed:.0f}/s | "
        f"exact {delta('exact_label_samples') / elapsed:.0f}/s | "
        f"self-unsafe {delta('self_filter_unsafe_samples')} | "
        f"guard exact boxes {delta('hard_guard_removed_exact_boxes')} | "
        f"guard targetless {delta('hard_guard_targetless_samples')}"
    )


def _head_runtime_telemetry_summary(
    previous,
    current,
    elapsed_seconds: float,
    *,
    now_ns: int,
    visible_sample: _AutomaticHeadSample | None,
) -> str:
    """Format non-spatial direct-head worker deltas and point freshness."""

    elapsed = max(float(elapsed_seconds), 1e-9)

    def delta(name: str) -> int:
        return max(
            0,
            int(getattr(current, name, 0)) - int(getattr(previous, name, 0)),
        )

    overwrites = delta("pending_overwrites") + delta("result_overwrites")
    stale = (
        delta("stale_submissions")
        + delta("stale_pending_dropped")
        + delta("stale_results_dropped")
    )
    freshness = "none"
    if visible_sample is not None:
        direct_timestamp_ns = int(
            getattr(
                visible_sample,
                "direct_source_timestamp_ns",
                visible_sample.source_timestamp_ns,
            )
        )
        age_ms = max(0.0, (int(now_ns) - direct_timestamp_ns) / 1e6)
        provenance = getattr(visible_sample, "provenance", None)
        state = (
            "bridge"
            if visible_sample.bridging
            else "anchored"
            if provenance is DirectHeadProvenance.MEASURED_PRIMARY
            else "direct"
        )
        freshness = f"{age_ms:.0f}ms {state}"
    return (
        f"HEAD completed {delta('jobs_completed') / elapsed:.0f}/s | "
        f"localized {delta('localized_heads') / elapsed:.0f}/s | "
        f"no-head {delta('no_head_results') / elapsed:.0f}/s | "
        "why no-decoded/no-plausible/multi-head "
        f"{delta('no_decoded_head_candidates') / elapsed:.0f}/"
        f"{delta('no_plausible_heads') / elapsed:.0f}/"
        f"{delta('multiple_plausible_heads') / elapsed:.0f}/s | "
        "secondary none/multi/unsupported "
        f"{delta('no_matching_secondary_players') / elapsed:.0f}/"
        f"{delta('multiple_matching_secondary_players') / elapsed:.0f}/"
        f"{delta('head_unsupported_by_matched_player') / elapsed:.0f}/s | "
        f"other {delta('unspecified_no_head_results') / elapsed:.0f}/s | "
        f"overwrites {overwrites} | stale {stale} | point age {freshness}"
    )


def _makcu_telemetry_summary(previous, current, elapsed_seconds: float) -> str:
    """Format passive output-loop counters collected since the prior report."""

    elapsed = max(float(elapsed_seconds), 1e-9)

    def delta(name: str) -> int:
        return max(0, int(getattr(current, name)) - int(getattr(previous, name)))

    ticks = delta("output_ticks")

    def duty(name: str) -> float:
        return 100.0 * delta(name) / ticks if ticks else 0.0

    commands = delta("movement_commands")
    abs_x = delta("emitted_abs_x")
    abs_y = delta("emitted_abs_y")
    net_x = int(getattr(current, "emitted_x")) - int(getattr(previous, "emitted_x"))
    net_y = int(getattr(current, "emitted_y")) - int(getattr(previous, "emitted_y"))
    physical_reports = delta("physical_input_reports")
    physical_abs_x = delta("physical_input_abs_x")
    physical_abs_y = delta("physical_input_abs_y")
    physical_net_x = int(getattr(current, "physical_input_x")) - int(
        getattr(previous, "physical_input_x")
    )
    physical_net_y = int(getattr(current, "physical_input_y")) - int(
        getattr(previous, "physical_input_y")
    )
    control_samples = delta("control_samples")

    def control_mean(name: str) -> float:
        total = max(
            0.0,
            float(getattr(current, name)) - float(getattr(previous, name)),
        )
        return total / control_samples if control_samples else 0.0

    def control_duty(name: str) -> float:
        return (
            100.0 * delta(name) / control_samples
            if control_samples
            else 0.0
        )

    saturation_x = (
        100.0 * delta("saturated_x_samples") / control_samples
        if control_samples
        else 0.0
    )
    saturation_y = (
        100.0 * delta("saturated_y_samples") / control_samples
        if control_samples
        else 0.0
    )
    return (
        f"MAKCU loop {ticks / elapsed:.0f} Hz | "
        f"button gate {duty('button_pressed_ticks'):.0f}% | "
        f"target {duty('target_present_ticks'):.0f}% | "
        f"fresh {duty('fresh_target_ticks'):.0f}% | "
        f"authorized {duty('authorized_ticks'):.0f}% | "
        f"moves {commands / elapsed:.0f}/s | "
        f"abs counts X/Y {abs_x / elapsed:.0f}/{abs_y / elapsed:.0f}/s | "
        f"net X/Y {net_x / elapsed:+.0f}/{net_y / elapsed:+.0f}/s | "
        f"physical reports {physical_reports / elapsed:.0f}/s | "
        "physical abs X/Y "
        f"{physical_abs_x / elapsed:.0f}/{physical_abs_y / elapsed:.0f}/s | "
        "physical net X/Y "
        f"{physical_net_x / elapsed:+.0f}/{physical_net_y / elapsed:+.0f}/s | "
        f"CTRL samples {control_samples / elapsed:.0f}/s | "
        f"error abs X/Y {control_mean('control_error_abs_x'):.1f}/"
        f"{control_mean('control_error_abs_y'):.1f}px | "
        f"pursuit X/Y {control_mean('pursuit_abs_x') * 60.0:.0f}/"
        f"{control_mean('pursuit_abs_y') * 60.0:.0f} cps | "
        "target velocity X/Y "
        f"{control_mean('target_velocity_abs_x_pixels_per_second'):.0f}/"
        f"{control_mean('target_velocity_abs_y_pixels_per_second'):.0f} px/s | "
        "FF confidence X/Y "
        f"{control_mean('velocity_feedforward_confidence_x') * 100.0:.0f}/"
        f"{control_mean('velocity_feedforward_confidence_y') * 100.0:.0f}% | "
        "reserve X/Y "
        f"{control_mean('pursuit_reserve_abs_x_counts_per_second'):.0f}/"
        f"{control_mean('pursuit_reserve_abs_y_counts_per_second'):.0f} cps "
        f"at {control_duty('pursuit_reserve_active_x_samples'):.0f}/"
        f"{control_duty('pursuit_reserve_active_y_samples'):.0f}% | "
        "motion corroboration "
        f"{control_mean('motion_corroboration_confidence') * 100.0:.0f}% | "
        "body motion X/Y "
        f"{control_mean('body_derived_motion_confidence_x') * 100.0:.0f}/"
        f"{control_mean('body_derived_motion_confidence_y') * 100.0:.0f}% | "
        f"saturation X/Y {saturation_x:.0f}/{saturation_y:.0f}% | "
        f"pursuit resets {delta('pursuit_resets')}"
    )


def _start_optional_aiming(aim_controller, aim_sensor):
    """Start optional aim devices without preventing capture and preview."""

    if aim_controller is None:
        return None, None
    try:
        if aim_sensor is not None:
            aim_sensor.start()
        aim_controller.start()
    except (RuntimeError, OSError, ValueError) as exc:
        try:
            aim_controller.stop()
        except (RuntimeError, OSError, ValueError):
            pass
        if aim_sensor is not None:
            try:
                aim_sensor.stop()
            except (RuntimeError, OSError, ValueError):
                pass
        print(
            f"Warning: detection-driven aim is disabled: {exc}. "
            "Capture, inference, and preview will continue.",
            file=sys.stderr,
        )
        return None, None
    return aim_controller, aim_sensor


def _validate_aim_safety(config: AppConfig) -> None:
    """Reject fail-open aiming configurations even when parse_args was bypassed."""

    if not config.aim:
        return
    if not (config.aim_label or "").strip():
        raise ValueError("Detection-driven aim requires an explicit target label")
    if not config.ignore_self:
        raise ValueError(
            "Detection-driven aim requires the third-person self filter to be enabled"
        )
    if config.aim_output == "remote":
        raise ValueError(
            "Remote aim is unavailable because no authenticated, physically gated "
            "receiver is included"
        )
    if config.aim_output not in {"local", "makcu"}:
        raise ValueError(
            f"Unsupported safe aim output: {config.aim_output!r}; "
            "expected 'local' or 'makcu'"
        )
    if config.aim_output == "local":
        if not (config.aim_activate_path or "").strip():
            raise ValueError("Local aim requires an explicit physical activation device")
        if isinstance(config.aim_activate_threshold, bool):
            raise ValueError(
                "Local aim activation threshold must be finite, greater than 0, "
                "and at most 1"
            )
        try:
            activation_threshold = float(config.aim_activate_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Local aim activation threshold must be a finite number in (0,1]"
            ) from exc
        if not math.isfinite(activation_threshold) or not 0.0 < activation_threshold <= 1.0:
            raise ValueError(
                "Local aim activation threshold must be finite, greater than 0, "
                "and at most 1"
            )


def _validate_calibration_safety(config: AppConfig) -> None:
    """Repeat the CLI's calibration gates for direct ``run(AppConfig)`` callers."""

    if not _calibration_requested(config):
        return
    import os

    from aiming.makcu_calibration_session import normalize_calibration_context

    if not config.aim or config.aim_output != "makcu":
        raise ValueError("MAKCU calibration requires live MAKCU aiming")
    if config.source.kind not in {"screen", "device"}:
        raise ValueError("MAKCU calibration requires a live capture source")
    if not (config.aim_makcu_port or "").strip():
        raise ValueError("MAKCU calibration requires an explicit verified port")
    if not config.ignore_self:
        raise ValueError("MAKCU calibration requires self-avatar exclusion")
    if config.detail_crop_size is not None:
        raise ValueError("MAKCU calibration cannot use the detail pass")
    if config.aim_makcu_active_profile is not None:
        raise ValueError("MAKCU calibration cannot load an active aim profile")
    if config.metrics_json is not None:
        raise ValueError("MAKCU calibration cannot write a live metrics report")
    if config.max_frames is not None or config.max_seconds is not None:
        raise ValueError("MAKCU calibration cannot use frame or time bounds")
    assert config.aim_calibration_evidence is not None
    if os.path.lexists(config.aim_calibration_evidence):
        raise ValueError(
            "MAKCU calibration evidence already exists; refusing to overwrite"
        )
    if not bool(getattr(sys, "frozen", False)):
        source_root = Path(__file__).resolve().parent
        evidence_path = config.aim_calibration_evidence.resolve(strict=False)
        if evidence_path == source_root or source_root in evidence_path.parents:
            raise ValueError(
                "MAKCU calibration evidence must be kept outside the source tree"
            )
    normalize_calibration_context(config.aim_calibration_context)


def _validate_active_profile_safety(config: AppConfig) -> None:
    """Reject explicit calibrated mode unless its full live boundary exists."""

    if not _active_profile_requested(config):
        return
    if not config.aim or config.aim_output != "makcu":
        raise ValueError("An active MAKCU profile requires live MAKCU aiming")
    if config.aim_calibration_evidence is not None:
        raise ValueError(
            "An active MAKCU profile cannot be combined with calibration"
        )
    if config.source.kind not in {"screen", "device"}:
        raise ValueError("An active MAKCU profile requires a live capture source")
    if config.detail_crop_size is not None:
        raise ValueError("An active MAKCU profile cannot use the detail pass")


def _validate_direct_head_plant_profile_binding(
    profile_binding,
    runtime_binding,
    *,
    runtime_detail_pass_enabled: bool = False,
) -> None:
    """Require the exact physical plant boundary used by direct-head control.

    Source revision and build identity remain recorded as audit provenance,
    but neither identifies the physical MAKCU/game response plant.  Requiring
    either to match would make committing an otherwise accepted controller
    repair invalidate unchanged gain/delay measurements.  The automatic detail
    rescue changes candidate acquisition, not the measured
    MAKCU-count-to-screen-pixel plant, so its runtime-only detail flag is also
    excluded.  No model, provider, capture, device, button, geometry, or
    aim-context field receives that plant-only exception.
    """

    from dataclasses import fields

    from aiming.makcu_calibration_activation import (
        CalibrationActivationBindingError,
    )
    from aiming.makcu_calibration_session import CalibrationRuntimeBinding

    if not isinstance(profile_binding, CalibrationRuntimeBinding) or not isinstance(
        runtime_binding,
        CalibrationRuntimeBinding,
    ):
        raise TypeError("direct-head plant bindings must be CalibrationRuntimeBinding")
    if not isinstance(runtime_detail_pass_enabled, bool):
        raise TypeError("runtime_detail_pass_enabled must be a bool")
    plant_only_exceptions = {"source_commit", "build_identity"}
    if runtime_detail_pass_enabled:
        plant_only_exceptions.add("detail_pass_enabled")
    mismatches = tuple(
        field.name
        for field in fields(CalibrationRuntimeBinding)
        if field.name not in plant_only_exceptions
        and getattr(profile_binding, field.name)
        != getattr(runtime_binding, field.name)
    )
    if mismatches:
        raise CalibrationActivationBindingError(
            "direct-head measured plant does not match runtime fields: "
            + ", ".join(mismatches)
        )


def _update_aim_target(
    tracker,
    detections,
    frame_shape: tuple[int, ...],
    *,
    continuation_detections=(),
    continuation_allowed: bool = True,
    self_exclusion_safe: bool,
    aim_runtime_enabled: bool = True,
    prediction_grace_safe: bool = True,
    measurement_ns: int | None = None,
):
    """Select a target, dropping history whenever the physical sample is unsafe."""

    if tracker is None:
        return None
    if not aim_runtime_enabled:
        tracker.reset()
        return None
    if not self_exclusion_safe:
        tracker.reset()
        return None
    if not prediction_grace_safe:
        tracker.reset()
        return None
    return tracker.update(
        detections,
        frame_shape,
        measurement_ns=measurement_ns,
        continuation_detections=continuation_detections,
        continuation_allowed=continuation_allowed,
    )


def _aim_status(
    *,
    runtime_enabled: bool,
    self_exclusion_ready: bool,
    selected_target,
    engaged: bool,
    activation_name: str,
    control_description: str,
) -> str | None:
    """Describe live aiming only when its optional output actually started."""

    if not runtime_enabled:
        return None
    if not self_exclusion_ready and selected_target is None:
        return "aim blocked: waiting for confident self-avatar exclusion"
    if selected_target is None:
        return (
            f"aim armed: {activation_name} held, waiting for target"
            if engaged
            else "aim: no matching target"
        )
    if engaged:
        return (
            f"aim active: {activation_name} held, {control_description}"
            if not self_exclusion_ready
            else (
                f"aim active: {activation_name} held, {control_description}, "
                "tracking selected head"
            )
        )
    return f"aim ready: hold {activation_name} to track selected head"


def run(config: AppConfig) -> int:
    _validate_aim_safety(config)
    _validate_calibration_safety(config)
    _validate_active_profile_safety(config)
    calibration_requested = _calibration_requested(config)
    active_profile_requested = _active_profile_requested(config)
    force_direct_head_mode = bool(
        config.aim
        and config.aim_output == "makcu"
        and config.aim_makcu_tracking_mode == "direct-head"
        and not calibration_requested
    )
    direct_head_plant_profile = None
    if force_direct_head_mode and active_profile_requested:
        from aiming.makcu_calibration_activation import load_active_profile

        assert config.aim_makcu_active_profile is not None
        # Direct-head keeps its automatic controller and all automatic safety
        # gates.  The explicitly selected, hash-validated profile contributes
        # only its measured physical plant after the live binding is checked.
        direct_head_plant_profile = load_active_profile(
            config.aim_makcu_active_profile
        )
        active_profile_requested = False
    active_profile = None
    calibrated_numeric_controller = None
    automatic_numeric_controller = None
    if active_profile_requested:
        from aiming.makcu_calibration_activation import load_active_profile

        assert config.aim_makcu_active_profile is not None
        # Explicit profile selection is not a cache lookup. Any malformed,
        # insecure, missing, or tampered file is a terminal startup error.
        active_profile = load_active_profile(config.aim_makcu_active_profile)
        calibrated_numeric_controller = _calibrated_controller_from_active_profile(
            active_profile,
            max_step=config.aim_makcu_max_step,
            vertical_rate_ratio=config.aim_makcu_vertical_rate_ratio,
        )
    automatic_makcu_requested = bool(
        config.aim
        and config.aim_output == "makcu"
        and active_profile is None
        and not calibration_requested
    )
    automatic_direct_head_requested = bool(
        automatic_makcu_requested
        and config.aim_makcu_tracking_mode == "direct-head"
    )
    aggressive_direct_head_mode = bool(
        automatic_direct_head_requested
        and AUTOMATIC_DIRECT_HEAD_AGGRESSIVE_ACQUISITION_MODE
    )
    aim_label = config.aim_label
    if aggressive_direct_head_mode and isinstance(aim_label, str):
        normalized_aim_label = aim_label.strip().casefold()
        if normalized_aim_label not in {"player", "person"}:
            print(
                "Warning: automatic direct-head aggressive mode expected aim "
                f"label 'player' or 'person'; forcing 'player' instead of {aim_label!r}.",
                file=sys.stderr,
            )
            aim_label = "player"
    if automatic_makcu_requested:
        # Automatic direct-head mode is an explicitly qualified MIGraphX path.
        # Reject an arbitrary/manual CPU or OpenVINO invocation before either
        # neural model is constructed or warmed up; the later runtime-summary
        # checks still prove what the successfully created session activated.
        if config.backend != "onnxruntime":
            raise RuntimeError(
                "Automatic direct-head MAKCU control requires the ONNX Runtime "
                "MIGraphX GPU backend; CPU/OpenVINO inference is not permitted"
            )
        if config.require_full_provider is not True:
            raise RuntimeError(
                "Automatic direct-head MAKCU control requires --require-full-provider "
                "so CPU graph and execution-provider failure fallback stay disabled"
            )
    if config.crop_size is not None and config.detail_crop_size is not None:
        raise ValueError(
            "The detail pass requires a full-frame primary inference; "
            "crop_size and detail_crop_size cannot both be enabled"
        )
    # The release-default automatic MAKCU path gets one bounded, conditional
    # small-target rescue without changing explicit crop/detail requests.
    # Calibration, explicit calibrated control, local aim, and detector-only
    # runs therefore cannot inherit this extra inference branch.  A direct-head
    # measured plant remains automatic control and still needs this acquisition
    # rescue; the plant profile only supplies counts-to-pixels dynamics.
    automatic_detail_rescue_enabled = bool(
        automatic_direct_head_requested
        and config.crop_size is None
        and config.detail_crop_size is None
    )
    effective_detail_crop_size = (
        AUTOMATIC_DETAIL_RESCUE_CROP_SIZE
        if automatic_detail_rescue_enabled
        else config.detail_crop_size
    )
    detail_pass_mode = (
        "automatic_activation_need_gated"
        if automatic_detail_rescue_enabled
        else "explicit_always"
        if config.detail_crop_size is not None
        else "disabled"
    )
    report_destination = None
    model_artifact_snapshot = None
    labels_artifact_snapshot = None
    if config.metrics_json is not None:
        # Reject a reused qualification filename before model startup or live
        # capture can consume minutes of the tester's time. Publication repeats
        # the check atomically to close the race between startup and shutdown.
        from utils.live_report import prepare_report_destination, snapshot_artifact

        report_destination = prepare_report_destination(config.metrics_json)
        # Bind the report to the bytes that are about to be loaded, rather than
        # hashing a mutable path only after the run has finished.
        model_artifact_snapshot = snapshot_artifact(config.model_path)
        labels_artifact_snapshot = snapshot_artifact(config.labels_path)
    elif (
        calibration_requested
        or active_profile_requested
        or direct_head_plant_profile is not None
    ):
        # Hash before detector construction so evidence is bound to the exact
        # bytes that this session is about to load.
        from utils.live_report import snapshot_artifact

        model_artifact_snapshot = snapshot_artifact(config.model_path)
        labels_artifact_snapshot = snapshot_artifact(config.labels_path)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Create a virtual environment and install requirements.txt."
        ) from exc

    from detection import (
        DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT,
        DetailPassStats,
        OpenVINOYoloDetector,
        merge_cross_pass_detections,
        plan_detail_pass,
    )
    from aiming import (
        AimActivationSensor,
        AimConfig,
        AimingController,
        AimingControllerError,
        AimActivationError,
        MakcuAimConfig,
        MakcuAimingController,
        TargetTracker,
        head_target_point,
    )
    from aiming.makcu import BUTTON_NAMES
    from utils.metrics import FrameTimings, RollingMetrics
    from utils.preview import PreviewPacer, create_preview_window
    from utils.preprocess import preprocess_frame
    from utils.render import (
        console_summary,
        draw_detections,
        draw_aim_target,
        draw_ignore_zone,
        draw_metrics,
    )
    from utils.self_filter import NormalizedBottomZone, SelfAvatarFilter

    if config.backend == "onnxruntime":
        # AMD and NVIDIA GPUs have no OpenVINO plugin, so they run the same
        # graph through ONNX Runtime.  Both backends share the decoder, so the
        # rest of the pipeline is unchanged by this choice.
        from detection.onnx_yolo import OnnxRuntimeYoloDetector

        detector_type = OnnxRuntimeYoloDetector
    else:
        detector_type = OpenVINOYoloDetector

    detector_arguments = dict(
        model_path=config.model_path,
        labels_path=config.labels_path,
        # The automatic direct-head implementation is qualified for this AMD
        # host on MIGraphX only.  Resolve that exact provider before session
        # construction rather than allowing generic GPU/AUTO selection to
        # initialize a different accelerator and reject it after warmup.
        device=(AUTOMATIC_HEAD_PROVIDER if automatic_makcu_requested else config.device),
        # Keep square detector-constructor integrations source-compatible;
        # rectangular shapes remain an explicit (height, width) pair.
        inference_size=compact_inference_size(config.inference_size),
        # Calibration must observe a visible player's box in nearly every frame
        # (the fit gates on 0.98 observation duty), so it runs the detector at
        # the lower calibration floor rather than the aim threshold.
        confidence=(
            min(config.confidence, CALIBRATION_CONFIDENCE_FLOOR)
            if calibration_requested
            else min(
                config.confidence,
                AUTOMATIC_DIRECT_HEAD_ACQUISITION_CONFIDENCE_FLOOR,
            )
            if automatic_direct_head_requested
            else config.confidence
        ),
        iou=config.iou_threshold,
        output_format=config.output_format,
    )
    if config.backend == "onnxruntime":
        detector_arguments["require_full_provider"] = config.require_full_provider
    detector = detector_type(**detector_arguments)
    detector.warmup()
    aim_configured_confidence = (
        (
            min(
                config.confidence,
                AUTOMATIC_DIRECT_HEAD_ACQUISITION_CONFIDENCE_FLOOR,
            )
            if automatic_direct_head_requested
            else config.confidence
        )
        if config.aim
        else None
    )
    postprocess_options = (
        {
            "confidence": min(
                config.confidence,
                AIM_CONTINUATION_CONFIDENCE_FLOOR,
            )
        }
        if config.aim and not calibration_requested
        else {}
    )
    if report_destination is not None or calibration_requested or active_profile_requested:
        from utils.live_report import verify_artifact_unchanged

        assert model_artifact_snapshot is not None
        assert labels_artifact_snapshot is not None
        verify_artifact_unchanged(
            config.model_path,
            model_artifact_snapshot,
            description="Model artifact",
        )
        verify_artifact_unchanged(
            config.labels_path,
            labels_artifact_snapshot,
            description="Labels artifact",
        )

    source = _build_capture(config)
    metrics = RollingMetrics(config.stats_window)
    display_snapshot = metrics.snapshot()
    preview_pacer = PreviewPacer(config.preview_fps) if config.preview else None
    preview_window = create_preview_window(cv2, WINDOW_NAME) if config.preview else None
    crop_warning_printed = False
    detail_geometry_printed = False
    detail_clamp_warning_printed = False
    detail_redundant_warning_printed = False
    detail_pass_stats = DetailPassStats(effective_detail_crop_size)
    automatic_detail_telemetry = _AutomaticDetailRescueTelemetry()
    automatic_detail_target_hint = _AutomaticDetailTargetHintState()
    automatic_detail_report_snapshot = automatic_detail_telemetry.snapshot()
    last_report_ns = perf_counter_ns()
    pipeline_started_ns: int | None = None
    pipeline_completed_ns: int | None = None
    pipeline_started_utc: str | None = None
    termination_reason = "source_ended"
    processed_frames = 0
    last_source_settings: dict[str, object] = {}
    cleanup_failures: list[str] = []
    self_zone = (
        NormalizedBottomZone(
            left=config.self_zone_left,
            width=config.self_zone_width,
            height=config.self_zone_height,
        )
        if config.ignore_self
        else None
    )
    self_filter = SelfAvatarFilter(self_zone) if self_zone is not None else None
    last_ignored_count: int | None = 0 if self_zone is not None else None
    last_ignored_detection = None
    automatic_detail_confirmed_self_detection = None
    aim_controller: AimingController | MakcuAimingController | None = None
    makcu_report_snapshot = None
    makcu_report_ns: int | None = None
    tracker_report_snapshot = None
    tracker_report_ns: int | None = None
    aim_input_telemetry: AimInputTelemetry | None = None
    aim_input_report_snapshot: AimInputTelemetrySnapshot | None = None
    aim_input_report_ns: int | None = None
    head_report_snapshot = None
    head_report_ns: int | None = None
    aim_sensor: AimActivationSensor | None = None
    target_tracker: TargetTracker | None = None
    calibration_session = None
    calibration_status = None
    calibration_evidence_written = False
    calibration_last_log: tuple[object, str] | None = None
    calibration_target_readiness = "target wait: awaiting detector frame"
    calibration_last_target_readiness: str | None = None
    calibration_previous_bbox: tuple[float, float, float, float] | None = None
    active_profile_bound = False
    direct_head_plant_profile_bound = False
    aim_runtime_enabled = False
    aim_activation_was_active = False
    aim_activation_name = "physical control"
    aim_control_description = "gated output"
    automatic_head_runtime: _AutomaticHeadRuntime | None = None
    automatic_last_controller_source_ns: int | None = None
    automatic_body_fallback_gate = (
        _AutomaticBodyFallbackGate()
        if automatic_direct_head_requested
        else None
    )
    automatic_body_fallback_controller_active = False
    aim_diagnostic_recorder = None
    aim_diagnostic_warning_printed = False
    automatic_plant_calibrated_delay_seconds: float | None = None
    automatic_plant_effective_delay_seconds: float | None = None
    automatic_plant_delay_upper_seconds: float | None = None
    if config.aim:
        aim_input_telemetry = AimInputTelemetry(config.aim_label)
        if not calibration_requested:
            target_tracker = TargetTracker(
                label=aim_label,
                head_ratio=config.aim_head_ratio,
                # The live path should not force a one-frame drop window for the
                # normal tracked target; brief empty detector gaps are common when
                # a target moves quickly or the detector briefly misses. Keep the
                # longer grace window for local aim and explicit profiles, while
                # still preserving the tighterbridge used only for the special
                # automatic MAKCU empty-case.
                lost_grace_frames=(
                    AUTOMATIC_MAKCU_EMPTY_GRACE_FRAMES
                    if automatic_makcu_requested
                    else DEFAULT_TARGET_TRACK_LOST_GRACE_FRAMES
                ),
            )
        aim_config = AimConfig(
            invert_x=config.aim_invert_x,
            invert_y=config.aim_invert_y,
            head_ratio=config.aim_head_ratio,
        )
        if config.aim_output == "makcu":
            aim_activation_name = BUTTON_NAMES[config.aim_makcu_button]
            automatic_plant = None
            automatic_rate_limits: tuple[float, float] | None = None
            automatic_feedback_rate_limits: tuple[float, float] | None = None
            automatic_runtime_max_step = config.aim_makcu_max_step
            if automatic_makcu_requested and direct_head_plant_profile is not None:
                from aiming.makcu_calibrated_control import CalibratedPlant

                (
                    automatic_plant_calibrated_delay_seconds,
                    automatic_plant_effective_delay_seconds,
                    automatic_plant_delay_upper_seconds,
                ) = _automatic_direct_head_plant_delay_bounds(
                    direct_head_plant_profile.fit
                )
                automatic_plant = CalibratedPlant(
                    direct_head_plant_profile.fit.x.gain_pixels_per_count,
                    direct_head_plant_profile.fit.y.gain_pixels_per_count,
                    automatic_plant_effective_delay_seconds,
                )
                automatic_rate_limits = (
                    _automatic_measured_direct_head_rate_limits(
                        max_step=config.aim_makcu_max_step,
                        plant=automatic_plant,
                    )
                )
                # Do not split positional feedback from the total measured
                # envelope.  The live A/B reduced catch-up on 30.5% of fast
                # samples and moved part of the correction into a differently
                # smoothed reserve path.  Let the factory resolve feedback to
                # these same total caps, exactly as in the preceding run.
                automatic_runtime_max_step = max(
                    config.aim_makcu_max_step,
                    math.ceil(max(automatic_rate_limits) / 60.0),
                )
            makcu_config = MakcuAimConfig(
                port=config.aim_makcu_port or "",
                activation_button=config.aim_makcu_button,
                strength=config.aim_makcu_strength,
                max_step=(
                    automatic_runtime_max_step
                    if automatic_makcu_requested
                    else config.aim_makcu_max_step
                ),
                smoothing_alpha=config.aim_makcu_smoothing_alpha,
                prediction_lead_seconds=config.aim_makcu_prediction_lead_seconds,
                derivative_damping_seconds=config.aim_makcu_derivative_damping_seconds,
                vertical_rate_ratio=(
                    1.0 if automatic_makcu_requested
                    else config.aim_makcu_vertical_rate_ratio
                ),
                invert_x=config.aim_invert_x,
                invert_y=config.aim_invert_y,
                head_ratio=config.aim_head_ratio,
            )
            if active_profile is None and calibration_requested:
                aim_controller = MakcuAimingController(makcu_config)
            elif active_profile is None:
                assert automatic_makcu_requested
                automatic_numeric_controller = _automatic_plant_aware_controller(
                    max_step=automatic_runtime_max_step,
                    direct_head=automatic_direct_head_requested,
                    plant=automatic_plant,
                    maximum_rates_counts_per_second=automatic_rate_limits,
                    maximum_feedback_rates_counts_per_second=(
                        automatic_feedback_rate_limits
                    ),
                )
                automatic_controller_options = {
                    "calibrated_controller": automatic_numeric_controller,
                }
                if direct_head_plant_profile is not None:
                    automatic_controller_options["expected_identity_token"] = (
                        direct_head_plant_profile.binding.makcu_identity_token
                    )
                aim_controller = MakcuAimingController(
                    makcu_config,
                    **automatic_controller_options,
                )
            else:
                assert calibrated_numeric_controller is not None
                aim_controller = MakcuAimingController(
                    makcu_config,
                    calibrated_controller=calibrated_numeric_controller,
                    expected_identity_token=(
                        active_profile.binding.makcu_identity_token
                    ),
                )
            if automatic_numeric_controller is not None:
                automatic_input = (
                    "direct-head input"
                    if automatic_direct_head_requested
                    else "stable measured-body input"
                )
                aim_control_description = (
                    f"automatic command-aware observer with {automatic_input} at "
                    f"{aim_controller.config.output_hz} Hz"
                )
            elif active_profile is None:
                aim_control_description = (
                    f"{aim_controller.config.output_hz} Hz control"
                )
            else:
                aim_control_description = (
                    "calibrated profile "
                    f"{active_profile.profile_sha256[:12]} at "
                    f"{aim_controller.config.output_hz} Hz"
                )
        else:
            aim_controller = AimingController(aim_config)
        if config.aim_output == "local":
            aim_activation_name = "LT"
            assert config.aim_activate_path is not None
            aim_sensor = AimActivationSensor(
                config.aim_activate_path,
                axis=config.aim_activate_axis,
                threshold=config.aim_activate_threshold,
            )

    _print_startup(detector, source)
    if automatic_detail_rescue_enabled:
        print(
            "Automatic MAKCU detail rescue: enabled | activation/need gated | "
            "same model/input | model-aspect ROI up to "
            f"{AUTOMATIC_DETAIL_RESCUE_CROP_SIZE} source px wide | runs on the "
            "held frame when the full pass has no acquisition-authorized "
            "exact target, or its center-nearest target in the ROI is <= "
            f"{AUTOMATIC_DETAIL_MAX_REFERENCE_HEIGHT:.0f}px at 1080p reference | "
            "a no-target rescue follows one fresh, generation-bound accepted "
            "target; otherwise the ROI stays centered | "
            "a live verified head plus an acquisition-authorized full-pass "
            "target carries without the second inference | close/off-center "
            "targets and released preview skip it | self-guarded full-pass "
            "need decision | self-related "
            "and lower-ROI-edge detail-only self fragments within "
            f"{AUTOMATIC_DETAIL_SELF_EDGE_MARGIN_MODEL_PIXELS:g} model px "
            "excluded before aim merge | class-aware one-to-one merge | "
            "unmatched additions <= "
            f"{DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT:.0f}px at 1080p reference"
        )
    elif config.detail_crop_size is not None:
        print(
            "Detail pass: explicit always-on | same model/input | centered "
            "model-aspect "
            f"ROI up to {config.detail_crop_size} source px wide | "
            "class-aware one-to-one "
            "cross-pass deduplication | unmatched detail additions <= "
            f"{DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT:.0f}px at 1080p reference"
        )
    if self_zone is not None:
        print(
            "Self-avatar filter: enabled heuristic | player-like labels | "
            "3-frame lock/relock | max one/frame | box height >= 0.250 | "
            "box width >= 0.060 | bottom-center zone: "
            f"left {self_zone.left:.3f} | width {self_zone.width:.3f} | "
            f"height {self_zone.height:.3f}"
        )
    try:
        source.start()
        if active_profile is None and direct_head_plant_profile is None:
            aim_controller, aim_sensor = _start_optional_aiming(
                aim_controller,
                aim_sensor,
            )
        else:
            # An explicitly selected profile is a strict operating mode, not
            # an optional acceleration hint.  In particular, an identity
            # rejection must stop startup instead of printing the legacy
            # fail-soft message that capture will continue.
            assert isinstance(aim_controller, MakcuAimingController)
            assert aim_sensor is None
            try:
                aim_controller.start()
            except (RuntimeError, OSError, ValueError) as exc:
                try:
                    aim_controller.stop()
                except (RuntimeError, OSError, ValueError):
                    pass
                startup_mode = (
                    "Direct-head measured-plant"
                    if direct_head_plant_profile is not None
                    else "Calibrated MAKCU profile-bound"
                )
                raise RuntimeError(
                    f"{startup_mode} controller failed strict "
                    f"startup: {exc}"
                ) from exc
        if direct_head_plant_profile is not None:
            from utils.live_report import verify_artifact_unchanged

            assert isinstance(aim_controller, MakcuAimingController)
            assert automatic_numeric_controller is not None
            assert model_artifact_snapshot is not None
            assert labels_artifact_snapshot is not None
            verify_artifact_unchanged(
                config.model_path,
                model_artifact_snapshot,
                description="Model artifact",
            )
            verify_artifact_unchanged(
                config.labels_path,
                labels_artifact_snapshot,
                description="Labels artifact",
            )
            direct_head_runtime_binding = _build_calibration_runtime_binding(
                config,
                detector_summary=detector.runtime_summary,
                capture_settings=dict(source.actual_settings),
                makcu_identity_token=aim_controller.identity_token,
                model_artifact_snapshot=model_artifact_snapshot,
                labels_artifact_snapshot=labels_artifact_snapshot,
            )
            _validate_direct_head_plant_profile_binding(
                direct_head_plant_profile.binding,
                direct_head_runtime_binding,
                runtime_detail_pass_enabled=effective_detail_crop_size is not None,
            )
            direct_head_plant_profile_bound = True
            assert automatic_plant_calibrated_delay_seconds is not None
            assert automatic_plant_effective_delay_seconds is not None
            assert automatic_plant_delay_upper_seconds is not None
            print(
                "Automatic direct-head measured plant: bound | profile "
                f"{direct_head_plant_profile.profile_sha256[:12]} | context "
                f"{direct_head_plant_profile.binding.context_name} | gains X/Y "
                f"{direct_head_plant_profile.fit.x.gain_pixels_per_count:.6g}/"
                f"{direct_head_plant_profile.fit.y.gain_pixels_per_count:.6g} "
                "px/count | calibrated/effective/upper delay "
                f"{automatic_plant_calibrated_delay_seconds * 1000.0:.2f}/"
                f"{automatic_plant_effective_delay_seconds * 1000.0:.2f}/"
                f"{automatic_plant_delay_upper_seconds * 1000.0:.2f} ms | "
                "automatic direct-head tuning retained"
            )
        aim_runtime_enabled = aim_controller is not None
        if aim_runtime_enabled and automatic_direct_head_requested:
            # This path is mandatory for automatic no-profile MAKCU.  A missing,
            # changed, or unloadable head model is a startup failure; it never
            # falls back to a body-box ratio.
            print(
                "Loading direct-head model on MIGraphXExecutionProvider GPU-only; "
                "first compile may take about 40 seconds; CPU inference fallback "
                "disabled."
            )
            automatic_head_runtime = _build_automatic_head_runtime(
                detector.runtime_summary
            )
            automatic_head_runtime.start()
        if (
            aim_runtime_enabled
            and config.aim_output == "makcu"
            and not calibration_requested
            and config.aim_diagnostic_dir is not None
        ):
            from utils.aim_diagnostic import (
                AimDiagnosticConfig,
                AimDiagnosticRecorder,
            )

            try:
                aim_diagnostic_recorder = AimDiagnosticRecorder(
                    AimDiagnosticConfig(
                        output_root=config.aim_diagnostic_dir,
                        sample_hz=config.aim_diagnostic_sample_hz,
                        max_duration_seconds=(
                            config.aim_diagnostic_max_duration_seconds
                        ),
                        # Model loading and switching the capture source can
                        # consume the old fixed window before the user ever
                        # presses the activation button. Arm at launch, then
                        # start the bounded source-time window on first press.
                        wait_for_activation=True,
                    ),
                    metadata={
                        "aim_label": config.aim_label,
                        "tracking_mode": config.aim_makcu_tracking_mode,
                        "head_ratio": config.aim_head_ratio,
                        "lost_grace_frames": AUTOMATIC_MAKCU_EMPTY_GRACE_FRAMES,
                        "configured_confidence": config.confidence,
                        "model_path": str(config.model_path),
                        "labels_path": str(config.labels_path),
                        "inference_size": str(config.inference_size),
                        "source": source.description,
                        "capture_settings": dict(source.actual_settings),
                        "inference_device": detector.runtime_summary.get("device"),
                        "active_providers": list(
                            detector.runtime_summary.get("active_providers", ())
                        ),
                        "plant_profile_sha256": (
                            None
                            if direct_head_plant_profile is None
                            else direct_head_plant_profile.profile_sha256
                        ),
                        "plant_context": (
                            None
                            if direct_head_plant_profile is None
                            else direct_head_plant_profile.binding.context_name
                        ),
                        "plant_gain_x_pixels_per_count": (
                            automatic_numeric_controller.plant.gain_x_pixels_per_count
                            if automatic_numeric_controller is not None
                            else None
                        ),
                        "plant_gain_y_pixels_per_count": (
                            automatic_numeric_controller.plant.gain_y_pixels_per_count
                            if automatic_numeric_controller is not None
                            else None
                        ),
                        "plant_delay_seconds": (
                            automatic_numeric_controller.plant.delay_seconds
                            if automatic_numeric_controller is not None
                            else None
                        ),
                        "plant_calibrated_delay_seconds": (
                            automatic_plant_calibrated_delay_seconds
                        ),
                        "plant_effective_delay_seconds": (
                            automatic_numeric_controller.plant.delay_seconds
                            if automatic_numeric_controller is not None
                            else None
                        ),
                        "plant_delay_upper_seconds": (
                            automatic_plant_delay_upper_seconds
                        ),
                    },
                )
                aim_diagnostic_recorder.start()
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                aim_diagnostic_recorder = None
                print(
                    f"Warning: automatic aim diagnostics are unavailable: {exc}",
                    file=sys.stderr,
                )
            else:
                print(
                    "Automatic aim diagnostics: armed; the bounded recording "
                    "window starts on first physical activation | output "
                    f"{aim_diagnostic_recorder.session_dir}"
                )
        if aggressive_direct_head_mode:
            print(
                "Automatic direct-head aggressive acquisition: enabled | self-filter "
                "target-block bypassed for lock acquisition | any position-only body "
                "fallback still requires strict self safety and decoder-miss proof"
            )
        if calibration_requested and not isinstance(
            aim_controller, MakcuAimingController
        ):
            raise RuntimeError(
                "MAKCU calibration cannot continue because the verified live "
                "controller did not start"
            )
        if active_profile is not None and not isinstance(
            aim_controller, MakcuAimingController
        ):
            raise RuntimeError(
                "Calibrated MAKCU aiming cannot continue because the exact "
                "profile-bound controller did not start"
            )
        if not aim_runtime_enabled:
            # A configured but unavailable output must fail closed. In
            # particular, do not retain a tracker that could still drive the
            # "aim ready" overlay or draw a selected aim point.
            target_tracker = None
            aim_input_telemetry = None
        elif config.aim_output == "makcu" and calibration_requested:
            assert isinstance(aim_controller, MakcuAimingController)
            print(
                f"MAKCU calibration: enabled | target {config.aim_label} | "
                f"activation {aim_activation_name} | exact full-pass detections "
                f">= calibration confidence "
                f"{min(config.confidence, CALIBRATION_CONFIDENCE_FLOOR):g} | "
                "bounded exclusive pulses | no automatic profile activation"
            )
        elif config.aim_output == "makcu" and active_profile is not None:
            # Announce calibrated operation only after capture/runtime binding
            # is verified below. The output worker has no target yet and
            # therefore cannot move during this startup boundary.
            pass
        elif (
            config.aim_output == "makcu"
            and not automatic_direct_head_requested
        ):
            assert automatic_numeric_controller is not None
            assert automatic_head_runtime is None
            activation = (
                f"MAKCU mouse button {config.aim_makcu_button} | "
                f"control loop {aim_controller.config.output_hz} Hz"
            )
            output = f"MAKCU {config.aim_makcu_port or 'auto-detect'}"
            automatic_control = automatic_numeric_controller.config
            automatic_plant = automatic_numeric_controller.plant
            print(
                f"Detection-driven aim: enabled | target {config.aim_label} | "
                f"output {output} | activation {activation} | "
                "stable measured-body tracking | fresh primary detections use "
                f"head ratio {config.aim_head_ratio:g} | bounded prediction "
                f"{automatic_control.stale_after_seconds * 1000.0:.0f} ms | "
                "direct-head worker disabled | automatic detail rescue disabled | "
                "control automatic command-aware observer | position tau/deadzone "
                f"{automatic_control.position_time_constant_seconds * 1000.0:.0f} ms/"
                f"{automatic_control.feedback_deadzone_pixels:.0f} px | "
                "caps X/Y "
                f"{automatic_control.maximum_rate_x_counts_per_second:.0f}/"
                f"{automatic_control.maximum_rate_y_counts_per_second:.0f} counts/s | "
                "measured screen envelope X/Y "
                f"{automatic_control.maximum_rate_x_counts_per_second * automatic_plant.gain_x_pixels_per_count:.0f}/"
                f"{automatic_control.maximum_rate_y_counts_per_second * automatic_plant.gain_y_pixels_per_count:.0f} px/s"
            )
        elif config.aim_output == "makcu":
            from detection.head_detector import DEFAULT_HEAD_CONFIDENCE

            assert automatic_numeric_controller is not None
            assert automatic_head_runtime is not None
            head_provider = automatic_head_runtime.provider
            head_model_name = getattr(automatic_head_runtime, "model_name", None)
            if not isinstance(head_model_name, str) or not head_model_name.strip():
                head_model_name = "SunXDS 0.8.0"
            head_confidence_threshold = getattr(
                automatic_head_runtime,
                "confidence_threshold",
                DEFAULT_HEAD_CONFIDENCE,
            )
            if not isinstance(head_confidence_threshold, (int, float)) or not math.isfinite(
                float(head_confidence_threshold)
            ):
                head_confidence_threshold = DEFAULT_HEAD_CONFIDENCE
            if head_provider != AUTOMATIC_HEAD_PROVIDER:
                raise RuntimeError(
                    "Automatic head runtime provider identity is not strict MIGraphX"
                )
            activation = (
                f"MAKCU mouse button {config.aim_makcu_button} | "
                f"control loop {aim_controller.config.output_hz} Hz"
            )
            output = f"MAKCU {config.aim_makcu_port or 'auto-detect'}"
            automatic_control = automatic_numeric_controller.config
            automatic_plant = automatic_numeric_controller.plant
            if config.aim_makcu_max_step < AUTOMATIC_DIRECT_HEAD_RECOMMENDED_MAX_STEP:
                print(
                    "Warning: automatic direct-head Max step "
                    f"{config.aim_makcu_max_step} limits each axis to "
                    f"{automatic_control.maximum_rate_x_counts_per_second:.0f} "
                    "counts/s; the validated moving-target value is 320 "
                    "(19200 counts/s). Fast targets can outrun the lower envelope."
                )
            configured_rate = float(config.aim_makcu_max_step) * 60.0
            if automatic_rate_limits is not None and (
                automatic_control.maximum_rate_x_counts_per_second
                > configured_rate
                or automatic_control.maximum_rate_y_counts_per_second
                > configured_rate
            ):
                print(
                    "Automatic direct-head measured pursuit envelope: "
                    f"configured base {config.aim_makcu_max_step} "
                    f"({configured_rate:.0f} counts/s) | measured per-axis "
                    f"caps X/Y "
                    f"{automatic_control.maximum_rate_x_counts_per_second:.0f}/"
                    f"{automatic_control.maximum_rate_y_counts_per_second:.0f} "
                    "counts/s | position-feedback caps X/Y "
                    f"{automatic_control.maximum_feedback_rate_x_counts_per_second:.0f}/"
                    f"{automatic_control.maximum_feedback_rate_y_counts_per_second:.0f} "
                    "counts/s | transport ceiling "
                    f"{aim_controller.config.max_step * 60:.0f} counts/s | "
                    "gains, filters, deadzone, and unsaturated near-lock output "
                    "unchanged"
                )
            print(
                f"Detection-driven aim: enabled | target {config.aim_label} | "
                f"output {output} | activation {activation} | "
                "control automatic command-aware observer | gains X/Y "
                f"{automatic_plant.gain_x_pixels_per_count:g}/"
                f"{automatic_plant.gain_y_pixels_per_count:g} px/count | "
                "effective delay "
                f"{automatic_plant.delay_seconds * 1000.0:.2f} ms | "
                f"head source pinned {head_model_name} direct boxes on "
                f"{head_provider} GPU-only "
                "(CPU fallback disabled) | direct-head confidence "
                f">= {float(head_confidence_threshold):g} | "
                f"latest-only {AUTOMATIC_HEAD_LOCALIZATION_HZ:g} Hz acquisition / "
                f"{AUTOMATIC_HEAD_TRACKING_LOCALIZATION_HZ:g} Hz anchored "
                "maintenance "
                "(acquisition recovery on stale/repeated model misses or within "
                f"{AUTOMATIC_HEAD_TRACKING_MINIMUM_LEASE_REMAINING_SECONDS * 1000.0:.0f} "
                "ms of lease expiry) | "
                f"direct results accepted within {AUTOMATIC_HEAD_STALE_AFTER_SECONDS * 1000.0:.0f} ms | "
                "newest-capture LK position lookahead <= "
                f"{AUTOMATIC_HEAD_CAPTURE_PHASE_MAX_LEAD_SECONDS * 1000.0:.0f} ms "
                "(identity unchanged; ordinary inferred-frame feed-forward retained "
                f"<= {automatic_control.maximum_body_derived_feedforward_fraction * 100.0:.0f}% "
                "and already-qualified fast pursuit <= "
                f"{automatic_control.maximum_correlated_lookahead_pursuit_feedforward_fraction * 100.0:.0f}% "
                "for one bounded hop; after "
                f"{AUTOMATIC_HEAD_TRACKING_MINIMUM_FLOW_SAMPLES} consecutive "
                "pixel-observed frames, measured-flow carry <= "
                f"{automatic_control.maximum_verified_flow_pursuit_feedforward_fraction * 100.0:.0f}%) | "
                "direct results establish the head anchor | "
                "current measured primary geometry carries position for at most "
                f"{DIRECT_HEAD_ANCHOR_MAX_AGE_SECONDS * 1000.0:.0f} ms | "
                "body candidates schedule head verification; after "
                f"{AUTOMATIC_BODY_FALLBACK_CONFIRMATIONS} exact measurements >= "
                f"{AUTOMATIC_BODY_FALLBACK_CONFIDENCE:g}, a fresh verified no-head "
                "decoder miss may use a position-only body proxy (no feed-forward, "
                "revoked on any gap/ambiguity, disabled after direct lock) | "
                "predicted primary geometry remains display-only | "
                "raw primary box remains identity/safety authority | "
                "verified mapped-motion source-age projection "
                f"{automatic_control.maximum_body_derived_projection_fraction * 100.0:.0f}% | "
                "feed-forward baseline/aligned/fast max 25%/"
                f"{automatic_control.maximum_body_derived_feedforward_fraction * 100.0:.0f}%/"
                f"{automatic_control.maximum_body_derived_pursuit_feedforward_fraction * 100.0:.0f}% | "
                "closed-loop trailing-residual max "
                f"{automatic_control.maximum_residual_pursuit_feedforward_fraction * 100.0:.0f}% | "
                "mapped-point LP "
                f"{AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS * 1000.0:.0f} ms | "
                "local-anchor LP "
                f"{AUTOMATIC_HEAD_NORMALIZED_ANCHOR_FILTER_TIME_CONSTANT_SECONDS * 1000.0:.0f} ms | "
                "mapped-point slew allowance/speed "
                f"{AUTOMATIC_HEAD_MAPPED_STEP_ALLOWANCE_PIXELS:g} px/"
                f"{AUTOMATIC_HEAD_MAPPED_MAX_SPEED_PIXELS_PER_SECOND:.0f} px/s | "
                "velocity-channel translation-first reconcile/LP "
                f"{AUTOMATIC_HEAD_VELOCITY_RECONCILIATION_TIME_CONSTANT_SECONDS * 1000.0:.0f}/"
                f"{AUTOMATIC_HEAD_MAPPED_VELOCITY_FILTER_TIME_CONSTANT_SECONDS * 1000.0:.0f} ms | "
                "position tau/deadzone/shoulder "
                f"{automatic_control.position_time_constant_seconds * 1000.0:.0f} ms/"
                f"{automatic_control.feedback_deadzone_pixels:.1f}/"
                f"{automatic_control.continuous_feedback_shoulder_pixels:g} px | "
                "far-error tau "
                f"{automatic_control.pursuit_position_time_constant_seconds * 1000.0:.0f} ms "
                f"from {automatic_control.pursuit_position_time_constant_start_pixels:g} "
                f"to {automatic_control.pursuit_position_time_constant_full_pixels:g} px | "
                "total pursuit caps X/Y "
                f"{automatic_control.maximum_rate_x_counts_per_second:.0f}/"
                f"{automatic_control.maximum_rate_y_counts_per_second:.0f} counts/s | "
                "position-feedback caps X/Y "
                f"{automatic_control.maximum_feedback_rate_x_counts_per_second:.0f}/"
                f"{automatic_control.maximum_feedback_rate_y_counts_per_second:.0f} counts/s | "
                "measured screen envelope X/Y "
                f"{automatic_control.maximum_rate_x_counts_per_second * automatic_plant.gain_x_pixels_per_count:.0f}/"
                f"{automatic_control.maximum_rate_y_counts_per_second * automatic_plant.gain_y_pixels_per_count:.0f} px/s"
            )
        else:
            activation = (
                f"LT axis {config.aim_activate_axis} on {config.aim_activate_path}"
                if aim_sensor is not None
                else "always active"
            )
            print(
                f"Detection-driven aim: enabled | target {config.aim_label} | "
                f"output local uinput | activation {activation}"
            )
        print(f"Capture settings: {_format_settings(source.actual_settings)}")
        tracking_path_telemetry = (
            _TrackingPathTelemetry()
            if automatic_direct_head_requested
            else None
        )
        tracking_path_report_snapshot = (
            tracking_path_telemetry.snapshot()
            if tracking_path_telemetry is not None
            else None
        )
        tracking_path_report_ns: int | None = (
            perf_counter_ns() if tracking_path_telemetry is not None else None
        )
        _warn_on_capture_mismatch(config, source.actual_settings)
        last_source_settings = dict(source.actual_settings)
        if active_profile is not None:
            from aiming.makcu_calibration_activation import (
                CalibrationActivationBindingError,
            )
            from utils.live_report import verify_artifact_unchanged

            assert isinstance(aim_controller, MakcuAimingController)
            assert calibrated_numeric_controller is not None
            assert model_artifact_snapshot is not None
            assert labels_artifact_snapshot is not None
            # Close the detector-construction/startup interval before allowing
            # the first normal target publication into the output worker.
            verify_artifact_unchanged(
                config.model_path,
                model_artifact_snapshot,
                description="Model artifact",
            )
            verify_artifact_unchanged(
                config.labels_path,
                labels_artifact_snapshot,
                description="Labels artifact",
            )
            runtime_binding = _build_calibration_runtime_binding(
                config,
                detector_summary=detector.runtime_summary,
                capture_settings=last_source_settings,
                makcu_identity_token=aim_controller.identity_token,
                model_artifact_snapshot=model_artifact_snapshot,
                labels_artifact_snapshot=labels_artifact_snapshot,
            )
            if active_profile.binding != runtime_binding:
                print(
                    "Warning: Active calibration profile does not exactly match "
                    "the current runtime binding; ignoring the stale profile for "
                    "this launch."
                )
            active_profile_bound = True
            profile_control = calibrated_numeric_controller.config
            print(
                f"Detection-driven aim: enabled | target {config.aim_label} | "
                f"output MAKCU | activation {aim_activation_name} | "
                f"control calibrated | profile {active_profile.profile_sha256[:12]} | "
                f"context {active_profile.binding.context_name} | gains X/Y "
                f"{active_profile.fit.x.gain_pixels_per_count:.6g}/"
                f"{active_profile.fit.y.gain_pixels_per_count:.6g} px/count | "
                f"delay {active_profile.fit.delay_seconds * 1000.0:.2f} ms | "
                "caps X/Y "
                f"{profile_control.maximum_rate_x_counts_per_second:.0f}/"
                f"{profile_control.maximum_rate_y_counts_per_second:.0f} counts/s"
                )
        if config.preview:
            assert preview_window is not None
            preview_behavior = (
                "latest-only Windows worker"
                if preview_window.mode == "threaded"
                else "main-thread HighGUI compatibility mode"
            )
            print(
                f"Preview: capped at {config.preview_fps:g} fps; detection and "
                f"control continue between refreshes | {preview_behavior} | "
                "service cost measured separately"
            )
            preview_window.start()

        if calibration_requested:
            from aiming.makcu_calibration_session import (
                CalibrationSessionConfig,
                MakcuCalibrationSession,
            )

            assert isinstance(aim_controller, MakcuAimingController)
            assert aim_configured_confidence is not None
            assert model_artifact_snapshot is not None
            assert labels_artifact_snapshot is not None
            binding = _build_calibration_runtime_binding(
                config,
                detector_summary=detector.runtime_summary,
                capture_settings=last_source_settings,
                makcu_identity_token=aim_controller.identity_token,
                model_artifact_snapshot=model_artifact_snapshot,
                labels_artifact_snapshot=labels_artifact_snapshot,
            )
            calibration_started_ns = perf_counter_ns()
            calibration_session = MakcuCalibrationSession(
                aim_controller,
                binding,
                config=CalibrationSessionConfig(
                    # Calibration tracks a visible player's box motion across
                    # pulses; it never aims from it.  Use the same lower floor
                    # as the detector/gate so the 0.98 observation-duty gate is
                    # reachable -- at the 0.25 aim threshold the recorded player
                    # confidence (median 0.31, 36% of frames below) dropped the
                    # duty to 0.67 and rejected otherwise-valid evidence.
                    minimum_confidence=min(
                        aim_configured_confidence,
                        CALIBRATION_CONFIDENCE_FLOOR,
                    ),
                    # Fire the bounded pulses gently.  At the 2400-count/s max
                    # a 200-count pulse is an 83 ms flick, which drove the
                    # game's vertical response into its nonlinear/accelerated
                    # region: the measured per-pulse Y gain scattered from
                    # 0.09 to 0.18 px/count and the Y fit scored R-squared 0.795
                    # (< 0.85).  A slower 600-count/s pulse keeps the response
                    # linear (a 200-count pulse becomes a 333 ms ramp), matching
                    # the clean manual vertical movement the hardware probe
                    # demonstrated.  The axis excursion gates are unchanged.
                    pulse_rate_counts_per_second=600.0,
                ),
                started_ns=calibration_started_ns,
            )
            calibration_status = calibration_session.status()
            calibration_last_log = (
                calibration_status.state,
                calibration_status.message,
            )
            print(f"MAKCU calibration: {calibration_status.message}")

        pipeline_started_ns = perf_counter_ns()
        if isinstance(aim_controller, MakcuAimingController):
            makcu_report_snapshot = aim_controller.telemetry_snapshot()
            makcu_report_ns = pipeline_started_ns
        if target_tracker is not None:
            tracker_telemetry_snapshot = getattr(
                target_tracker,
                "telemetry_snapshot",
                None,
            )
            if callable(tracker_telemetry_snapshot):
                tracker_report_snapshot = tracker_telemetry_snapshot()
                tracker_report_ns = pipeline_started_ns
        if aim_input_telemetry is not None:
            aim_input_report_snapshot = aim_input_telemetry.snapshot()
            aim_input_report_ns = pipeline_started_ns
        if automatic_head_runtime is not None:
            head_report_snapshot = automatic_head_runtime.status
            head_report_ns = pipeline_started_ns
        if report_destination is not None:
            from utils.live_report import utc_now

            pipeline_started_utc = utc_now()
        deadline_ns = (
            pipeline_started_ns + round(config.max_seconds * 1_000_000_000)
            if config.max_seconds is not None
            else None
        )
        while True:
            loop_started_ns = perf_counter_ns()
            if deadline_ns is not None and loop_started_ns >= deadline_ns:
                termination_reason = "max_seconds"
                break
            read_timeout = 0.25
            if deadline_ns is not None:
                # Keep the existing bounded read and shorten its final wait so
                # an unchanged DXGI desktop still obeys --max-seconds.
                read_timeout = min(
                    read_timeout,
                    max(0.0, (deadline_ns - loop_started_ns) / 1_000_000_000),
                )
            preview_service_ms = 0.0
            packet = source.read(timeout=read_timeout)
            read_returned_ns = perf_counter_ns()
            # A static DXGI source still returns from this bounded read every
            # 250 ms, so inline HighGUI can service Escape/window close even
            # when there is no paced preview frame to submit.
            if packet is None:
                if automatic_head_runtime is not None:
                    automatic_head_runtime.raise_if_failed()
                if source.error:
                    raise RuntimeError(source.error)
                if deadline_ns is not None and read_returned_ns >= deadline_ns:
                    termination_reason = "max_seconds"
                    break
                if source.ended:
                    termination_reason = "source_ended"
                    break
                if preview_window is not None:
                    preview_service_started_ns = perf_counter_ns()
                    continue_running = preview_window.poll()
                    preview_service_ms += (
                        perf_counter_ns() - preview_service_started_ns
                    ) / 1e6
                    if not continue_running:
                        termination_reason = "preview_closed"
                        break
                continue

            # A source may return its final packet as the deadline is reached.
            # A packet already delivered by a source is still a
            # legitimate sample; max-seconds is checked again immediately
            # after it finishes, and this preserves exact max-frames behavior
            # when both bounds are supplied.

            processing_started_ns = perf_counter_ns()
            # Sample the physical automatic-aim gate once for this captured
            # frame.  The same value controls both rescue inference and later
            # target publication, so a released preview never pays for the
            # second pass and the first observed held frame can use it.
            automatic_frame_activation_active: bool | None = None
            if automatic_detail_rescue_enabled:
                automatic_frame_activation_active = bool(
                    isinstance(aim_controller, MakcuAimingController)
                    and aim_runtime_enabled
                    and aim_controller.activation_pressed
                )
                if not automatic_frame_activation_active:
                    automatic_detail_target_hint.clear()
            preprocessing_started_ns = perf_counter_ns()
            prepared = preprocess_frame(
                packet.image,
                inference_size=config.inference_size,
                crop_size=config.crop_size,
            )
            preprocessing_completed_ns = perf_counter_ns()

            inference_started_ns = perf_counter_ns()
            raw = detector.infer(prepared.tensor)
            inference_completed_ns = perf_counter_ns()
            detections = detector.postprocess(
                raw,
                transform=prepared.transform,
                frame_shape=packet.image.shape,
                **postprocess_options,
            )
            all_detections = tuple(detections)
            detections, continuation_detections = (
                _partition_detections_by_confidence(
                    all_detections,
                    aim_configured_confidence,
                )
            )
            postprocess_completed_ns = perf_counter_ns()
            detail_preprocess_ms = 0.0
            detail_inference_ms = 0.0
            detail_postprocess_ms = 0.0
            detections_ready_ns = postprocess_completed_ns
            automatic_detail_full_guard: HardAimGuardResult | None = None
            automatic_detail_full_exact_target_present = False
            if effective_detail_crop_size is not None:
                detail_plan = plan_detail_pass(
                    packet.image.shape,
                    effective_detail_crop_size,
                    config.inference_size,
                )
                detail_should_run = True
                if automatic_detail_rescue_enabled:
                    if not automatic_frame_activation_active:
                        detail_reason = "activation_released"
                    else:
                        assert config.aim_label is not None
                        assert self_zone is not None
                        # Decide rescue need from stateless, aim-safe full-pass
                        # evidence. The real SelfAvatarFilter is still applied
                        # exactly once below, after both passes are merged.
                        # Otherwise the large on-screen avatar itself can hide
                        # the fact that no opponent survived the full pass.
                        automatic_detail_full_guard = _apply_hard_aim_guard(
                            all_detections,
                            packet.image.shape,
                            self_zone=self_zone,
                            aim_label=config.aim_label,
                            configured_confidence=aim_configured_confidence,
                            confirmed_self_detection=(
                                automatic_detail_confirmed_self_detection
                            ),
                            obvious_bottom_shoulder_guard=True,
                        )
                        detail_reason = _automatic_detail_rescue_reason(
                            automatic_detail_full_guard.detections,
                            packet.image.shape,
                            detail_plan,
                            aim_label=config.aim_label,
                            configured_confidence=aim_configured_confidence,
                        )
                        automatic_detail_full_exact_target_present = (
                            detail_reason != "no_exact_target"
                        )
                        if (
                            detail_reason == "small_central_target"
                            and automatic_head_runtime is not None
                            and getattr(
                                automatic_head_runtime,
                                "has_live_measured_anchor",
                                lambda **_kwargs: False,
                            )(now_ns=perf_counter_ns())
                            is True
                        ):
                            # Once a direct head is established, this frame's
                            # acquisition-authorized full-pass body already
                            # carries its bounded normalized identity. Running
                            # the same body model again over the centered ROI
                            # adds 7-10 ms and cuts live cadence roughly in half
                            # without adding head evidence. A missing full-pass
                            # target still takes the rescue path above.
                            detail_reason = "verified_anchor"
                        if detail_reason == "no_exact_target":
                            assert target_tracker is not None
                            assert automatic_head_runtime is not None
                            hint_center = automatic_detail_target_hint.center_if_valid(
                                source_timestamp_ns=packet.read_started_ns,
                                track_generation=target_tracker.track_generation,
                                identity_generation=(
                                    automatic_head_runtime.identity_generation
                                ),
                                activation_active=True,
                            )
                            if hint_center is not None:
                                detail_plan = plan_detail_pass(
                                    packet.image.shape,
                                    effective_detail_crop_size,
                                    config.inference_size,
                                    center_point=hint_center,
                                )
                    automatic_detail_telemetry.record(detail_reason)
                    detail_should_run = detail_reason in {
                        "no_exact_target",
                        "small_central_target",
                    }
                if detail_should_run:
                    detail_pass_stats.record(detail_plan)
                if detail_should_run and not detail_geometry_printed:
                    print(
                        "Detail pass coverage: "
                        f"{detail_plan.crop_policy.replace('_', ' ')} "
                        f"{detail_plan.applied_crop_width}x"
                        f"{detail_plan.applied_crop_height} of "
                        f"{detail_plan.source_width}x{detail_plan.source_height} "
                        f"({detail_plan.coverage_fraction * 100.0:.1f}% of frame area; "
                        f"{detail_plan.effective_linear_magnification:.2f}x "
                        "derived linear detail versus the full pass)"
                    )
                    detail_geometry_printed = True
                if (
                    detail_should_run
                    and detail_plan.clamped
                    and not detail_clamp_warning_printed
                ):
                    print(
                        "Warning: --detail-crop-size was reduced to the largest "
                        "exact model-aspect ROI that fits this source: "
                        f"{detail_plan.applied_crop_width}x"
                        f"{detail_plan.applied_crop_height}px.",
                        file=sys.stderr,
                    )
                    detail_clamp_warning_printed = True
                if detail_should_run and detail_plan.redundant:
                    if not detail_redundant_warning_printed:
                        print(
                            "Warning: detail pass is identical to this square source; "
                            "the redundant second inference is disabled.",
                            file=sys.stderr,
                        )
                        detail_redundant_warning_printed = True
                elif detail_should_run:
                    detail_preprocessing_started_ns = perf_counter_ns()
                    detail_prepared = preprocess_frame(
                        packet.image,
                        inference_size=config.inference_size,
                        crop_size=(
                            detail_plan.applied_crop_height,
                            detail_plan.applied_crop_width,
                        ),
                        crop_origin=(detail_plan.crop_x, detail_plan.crop_y),
                    )
                    detail_preprocessing_completed_ns = perf_counter_ns()

                    detail_inference_started_ns = detail_preprocessing_completed_ns
                    detail_raw = detector.infer(detail_prepared.tensor)
                    detail_inference_completed_ns = perf_counter_ns()
                    detail_detections = detector.postprocess(
                        detail_raw,
                        transform=detail_prepared.transform,
                        frame_shape=packet.image.shape,
                        **postprocess_options,
                    )
                    if automatic_detail_rescue_enabled:
                        assert automatic_detail_full_guard is not None
                        assert self_zone is not None
                        detail_detections = (
                            _exclude_automatic_detail_lower_edge_self_fragments(
                                detail_detections,
                                all_detections,
                                packet.image.shape,
                                detail_plan=detail_plan,
                                self_zone=self_zone,
                            )
                        )
                        self_references = list(
                            automatic_detail_full_guard.removed_detections
                        )
                        if automatic_detail_confirmed_self_detection is not None:
                            self_references.append(
                                automatic_detail_confirmed_self_detection
                            )
                        detail_detections = (
                            _exclude_automatic_detail_self_relatives(
                                detail_detections,
                                packet.image.shape,
                                self_references=self_references,
                            )
                        )
                    if config.aim:
                        detail_normal, _detail_continuation = (
                            _partition_detections_by_confidence(
                                detail_detections,
                                aim_configured_confidence,
                            )
                        )
                        all_detections = tuple(
                            merge_cross_pass_detections(
                                all_detections,
                                detail_detections,
                                source_height=packet.image.shape[0],
                                unmatched_detail_max_reference_height=(
                                    DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT
                                ),
                            )
                        )
                        detections = tuple(
                            merge_cross_pass_detections(
                                detections,
                                detail_normal,
                                source_height=packet.image.shape[0],
                                unmatched_detail_max_reference_height=(
                                    DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT
                                ),
                                stats=detail_pass_stats,
                            )
                        )
                        _all_normal, continuation_detections = (
                            _partition_detections_by_confidence(
                                all_detections,
                                aim_configured_confidence,
                            )
                        )
                        all_detections = tuple(detections) + tuple(
                            continuation_detections
                        )
                    else:
                        detections = tuple(
                            merge_cross_pass_detections(
                                detections,
                                detail_detections,
                                source_height=packet.image.shape[0],
                                unmatched_detail_max_reference_height=(
                                    DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT
                                ),
                                stats=detail_pass_stats,
                            )
                        )
                        all_detections = tuple(detections)
                    detections_ready_ns = perf_counter_ns()
                    detail_preprocess_ms = (
                        detail_preprocessing_completed_ns
                        - detail_preprocessing_started_ns
                    ) / 1e6
                    detail_inference_ms = (
                        detail_inference_completed_ns - detail_inference_started_ns
                    ) / 1e6
                    # Includes source-space cross-pass consolidation, the final
                    # deterministic postprocessing operation of this pass.
                    detail_postprocess_ms = (
                        detections_ready_ns - detail_inference_completed_ns
                    ) / 1e6
            if aim_input_telemetry is not None:
                aim_input_telemetry.record_sample(detections)

            calibration_detections = tuple(all_detections)
            self_exclusion_ready = self_filter is None
            aim_self_exclusion_safe = self_exclusion_ready
            uncertain_self_safe_aim_source: tuple[object, ...] | None = None
            verified_flow_continuation: tuple[object, ...] = ()
            if self_filter is not None:
                previous_measured_player = (
                    None
                    if target_tracker is None
                    else getattr(
                        target_tracker,
                        "accepted_measurement",
                        None,
                    )
                )
                verified_flow_point = None
                flow_point_reader = getattr(
                    automatic_head_runtime,
                    "verified_flow_point_for_frame",
                    None,
                )
                if (
                    automatic_direct_head_requested
                    and automatic_frame_activation_active is True
                    and previous_measured_player is not None
                    and callable(flow_point_reader)
                ):
                    candidate_flow_point = flow_point_reader(
                        source_timestamp_ns=packet.read_started_ns,
                        now_ns=perf_counter_ns(),
                    )
                    if (
                        isinstance(candidate_flow_point, tuple)
                        and len(candidate_flow_point) == 2
                    ):
                        verified_flow_point = candidate_flow_point
                if (
                    verified_flow_point is not None
                    and config.aim_label is not None
                    and self_zone is not None
                ):
                    verified_flow_continuation = (
                        _verified_flow_continuation_cluster(
                            calibration_detections,
                            packet.image.shape,
                            previous_player=previous_measured_player,
                            verified_head_point=verified_flow_point,
                            aim_label=config.aim_label,
                            confidence_floor=(
                                AUTOMATIC_DIRECT_HEAD_ACQUISITION_CONFIDENCE_FLOOR
                            ),
                            self_zone=self_zone,
                        )
                    )
                protected_detection_ids = {
                    id(detection) for detection in verified_flow_continuation
                }
                self_filter_input = tuple(
                    detection
                    for detection in all_detections
                    if id(detection) not in protected_detection_ids
                )
                exclusion = self_filter.apply(
                    self_filter_input,
                    packet.image.shape,
                )
                if exclusion.ignored_detection is not None:
                    automatic_detail_confirmed_self_detection = (
                        exclusion.ignored_detection
                    )
                elif not bool(getattr(self_filter, "acquired", False)):
                    # Preserve the last positive self geometry only for the
                    # temporal filter's own bounded lock/lost-grace interval.
                    # This lets the next frame reject a cropped child even if
                    # the full-pass parent is temporarily absent, without
                    # permanently suppressing a later distinct opponent.
                    automatic_detail_confirmed_self_detection = None
                retained_detection_ids = {
                    id(detection) for detection in exclusion.detections
                } | protected_detection_ids
                all_detections = tuple(
                    detection
                    for detection in calibration_detections
                    if id(detection) in retained_detection_ids
                )
                detections, continuation_detections = (
                    _partition_detections_by_confidence(
                        all_detections,
                        aim_configured_confidence,
                    )
                )
                ignored_detection = exclusion.ignored_detection
                ignored_is_display_detection = bool(
                    ignored_detection is not None
                    and (
                        aim_configured_confidence is None
                        or float(ignored_detection.confidence)
                        >= aim_configured_confidence
                    )
                )
                last_ignored_count = (
                    exclusion.ignored_count if ignored_is_display_detection else 0
                )
                last_ignored_detection = (
                    ignored_detection if ignored_is_display_detection else None
                )
                self_exclusion_ready = exclusion.aim_safe
                aim_self_exclusion_safe = self_exclusion_ready
                if (
                    verified_flow_continuation
                    and not self_exclusion_ready
                ):
                    # The heuristic may remain uncertain about a separate
                    # avatar, but only the exact-frame pixel-bound opponent
                    # cluster is eligible for this continuation update.
                    uncertain_self_safe_aim_source = (
                        verified_flow_continuation
                    )
                    aim_self_exclusion_safe = True
                elif (
                    automatic_direct_head_requested
                    and not self_exclusion_ready
                    and aim_configured_confidence is not None
                    and config.aim_label is not None
                ):
                    uncertain_self_safe_aim_source = (
                        _aim_detections_safely_distinct_from_uncertain_self(
                            all_detections,
                            packet.image.shape,
                            uncertain_self_detections=getattr(
                                exclusion,
                                "uncertain_self_detections",
                                (),
                            ),
                            aim_label=config.aim_label,
                            configured_confidence=aim_configured_confidence,
                        )
                    )
                    aim_self_exclusion_safe = bool(
                        uncertain_self_safe_aim_source
                    )
            if aim_input_telemetry is not None:
                aim_input_telemetry.record_self_filter(
                    aim_safe=self_exclusion_ready,
                )

            # Hard self guard for aim selection: never select a likely self-avatar
            # candidate from the configured bottom zone, even if temporal lock is
            # not currently confident enough to hide it from the preview list.
            # Opposite-shoulder ambiguity is handled temporally by SelfAvatarFilter;
            # do not guess that an arbitrary large bottom opponent is self.
            aim_detections = detections
            aim_continuation_detections = continuation_detections
            if uncertain_self_safe_aim_source is not None:
                aim_detections, aim_continuation_detections = (
                    _partition_detections_by_confidence(
                        uncertain_self_safe_aim_source,
                        aim_configured_confidence,
                    )
                )
            hard_guard_revoked_prediction_grace = False
            confirmed_self_for_hard_guard = None
            if self_filter is not None:
                confirmed_self_for_hard_guard = (
                    exclusion.ignored_detection
                    if exclusion.aim_safe
                    and exclusion.ignored_count == 1
                    and exclusion.ignored_detection is not None
                    else None
                )
            hard_guard_source = (
                uncertain_self_safe_aim_source
                if uncertain_self_safe_aim_source is not None
                else all_detections
            )
            if (
                self_zone is not None
                and hard_guard_source
                and not aggressive_direct_head_mode
            ):
                hard_guard_result = _apply_hard_aim_guard(
                    hard_guard_source,
                    packet.image.shape,
                    self_zone=self_zone,
                    aim_label=config.aim_label,
                    configured_confidence=aim_configured_confidence,
                    confirmed_self_detection=confirmed_self_for_hard_guard,
                    unconfirmed_zone_guard=(
                        not automatic_direct_head_requested
                    ),
                    obvious_bottom_shoulder_guard=(
                        automatic_direct_head_requested
                    ),
                )
                aim_detections, aim_continuation_detections = (
                    _partition_detections_by_confidence(
                        hard_guard_result.detections,
                        aim_configured_confidence,
                    )
                )
                # This was not a genuine detector-empty sample. Never bridge
                # an old physical target when an exact aim-label candidate was
                # consumed and no exact aim-label target survived.
                hard_guard_revoked_prediction_grace = (
                    hard_guard_result.targetless_after_exact_removal
                )
                if aim_input_telemetry is not None:
                    aim_input_telemetry.record_hard_guard(hard_guard_result)
            direct_head_sample: _AutomaticHeadSample | None = None
            new_head_sample: _AutomaticHeadSample | None = None
            correlated_root_sample: _AutomaticHeadSample | None = None
            controller_input_source = "none"
            tracking_activation_active = False
            raw_activation_known = False
            raw_activation_pressed = False
            activation_requires_release = False
            activation_hold_expired = False
            if calibration_session is not None:
                # Calibration consumes only this frame's configured-confidence,
                # exact-label, full-pass result.  It deliberately bypasses the
                # target tracker, prediction grace, continuation detections, and
                # the normal aim controller update path.
                assert isinstance(aim_controller, MakcuAimingController)
                assert aim_configured_confidence is not None
                assert config.aim_label is not None
                (
                    calibration_observation,
                    selected_aim_target,
                    calibration_target_readiness,
                ) = _calibration_observation_target_and_readiness(
                    calibration_detections,
                    packet.image.shape,
                    aim_label=config.aim_label,
                    head_ratio=config.aim_head_ratio,
                    configured_confidence=min(
                        aim_configured_confidence,
                        CALIBRATION_CONFIDENCE_FLOOR,
                    ),
                    invert_x=config.aim_invert_x,
                    invert_y=config.aim_invert_y,
                    self_exclusion_safe=self_exclusion_ready,
                    measurement_ns=packet.read_started_ns,
                    previous_normalized_bbox=calibration_previous_bbox,
                    safe_roi_margin_ratio=(
                        calibration_session.config.safe_roi_margin_ratio
                    ),
                    maximum_reference_error=(
                        calibration_session.config.maximum_reference_error
                    ),
                )
                if calibration_observation is not None:
                    calibration_previous_bbox = calibration_observation.normalized_bbox
                if calibration_target_readiness != calibration_last_target_readiness:
                    print(f"MAKCU calibration target: {calibration_target_readiness}")
                    calibration_last_target_readiness = calibration_target_readiness
                calibration_status = calibration_session.update_from_controller(
                    perf_counter_ns(),
                    observation=calibration_observation,
                )
                raw_known, raw_pressed = aim_controller.raw_activation_state
                raw_activation_known = bool(raw_known)
                raw_activation_pressed = bool(raw_pressed)
                activation_requires_release = bool(
                    getattr(aim_controller, "activation_requires_release", False)
                )
                aim_engaged = raw_known and raw_pressed
                tracking_activation_active = bool(aim_engaged)
                current_calibration_log = (
                    calibration_status.state,
                    calibration_status.message,
                )
                if current_calibration_log != calibration_last_log:
                    print(f"MAKCU calibration: {calibration_status.message}")
                    calibration_last_log = current_calibration_log
                aim_status = (
                    f"calibration {calibration_status.state.value}: "
                    f"{calibration_status.message} | {calibration_target_readiness}"
                )
            else:
                if active_profile is not None and not active_profile_bound:
                    raise RuntimeError(
                        "Calibrated MAKCU profile was not bound before target update"
                    )
                if (
                    direct_head_plant_profile is not None
                    and not direct_head_plant_profile_bound
                ):
                    raise RuntimeError(
                        "Direct-head measured plant was not bound before target update"
                    )
                tracking_activation_active = aim_runtime_enabled
                if aim_sensor is not None:
                    tracking_activation_active = aim_sensor.read()
                elif isinstance(aim_controller, MakcuAimingController):
                    tracking_activation_active = (
                        automatic_frame_activation_active
                        if automatic_frame_activation_active is not None
                        else aim_controller.activation_pressed
                    )
                    raw_state = getattr(
                        aim_controller,
                        "raw_activation_state",
                        (False, False),
                    )
                    raw_activation_known = bool(raw_state[0])
                    raw_activation_pressed = bool(raw_state[1])
                    activation_requires_release = bool(
                        getattr(
                            aim_controller,
                            "activation_requires_release",
                            False,
                        )
                    )
                    activation_hold_expired = bool(
                        raw_activation_known
                        and raw_activation_pressed
                        and activation_requires_release
                        and not tracking_activation_active
                    )
                activation_transition = (
                    tracking_activation_active != aim_activation_was_active
                )
                automatic_revoked_this_frame = False
                if activation_transition and target_tracker is not None:
                    # A new physical hold must establish configured-confidence
                    # provenance in this hold; never inherit a weak-only track
                    # maintained while output was inactive.
                    target_tracker.reset()
                    automatic_detail_target_hint.clear()
                if activation_transition and automatic_head_runtime is not None:
                    # Both edges are safety epochs.  In-flight results from the
                    # old hold can never arm the new one, even if inference
                    # finishes after the button transition.
                    automatic_head_runtime.advance_identity()
                    assert isinstance(aim_controller, MakcuAimingController)
                    automatic_revoked_this_frame = _publish_automatic_head_loss_once(
                        aim_controller,
                        packet.image.shape,
                        source_timestamp_ns=packet.read_started_ns,
                        already_published=automatic_revoked_this_frame,
                    )
                    assert automatic_body_fallback_gate is not None
                    automatic_body_fallback_gate.reset()
                    automatic_body_fallback_controller_active = False
                aim_activation_was_active = tracking_activation_active

                selected_aim_target = _update_aim_target(
                    target_tracker,
                    aim_detections,
                    packet.image.shape,
                    continuation_detections=aim_continuation_detections,
                    continuation_allowed=(
                        aim_runtime_enabled and tracking_activation_active
                    ),
                    self_exclusion_safe=(
                        aim_self_exclusion_safe or aggressive_direct_head_mode
                    ),
                    aim_runtime_enabled=aim_runtime_enabled,
                    prediction_grace_safe=not hard_guard_revoked_prediction_grace,
                    measurement_ns=packet.read_started_ns,
                )
                automatic_detail_tracker_update_safe = bool(
                    aim_runtime_enabled
                    and tracking_activation_active
                    and (
                        aim_self_exclusion_safe
                        or aggressive_direct_head_mode
                    )
                    and not hard_guard_revoked_prediction_grace
                )
                if (
                    automatic_detail_rescue_enabled
                    and not automatic_detail_tracker_update_safe
                ):
                    # _update_aim_target resets tracker state for every one of
                    # these unsafe cases. TargetTracker's identity generation
                    # intentionally does not advance on a safety reset, so the
                    # crop-only hint must be cleared alongside it explicitly.
                    automatic_detail_target_hint.clear()
                aim_engaged = False
                if aim_controller is not None:
                    active = (
                        tracking_activation_active if aim_sensor is not None else True
                    )
                    if automatic_head_runtime is not None:
                        assert isinstance(aim_controller, MakcuAimingController)
                        assert target_tracker is not None
                        accepted_player = target_tracker.accepted_measurement
                        if (
                            automatic_detail_rescue_enabled
                            and automatic_detail_tracker_update_safe
                            and automatic_detail_full_exact_target_present
                            and accepted_player is not None
                            and float(accepted_player.confidence)
                            >= aim_configured_confidence
                        ):
                            # Only a downstream-accepted exact measurement on a
                            # frame where the guarded full primary saw an exact
                            # target may seed this hint. A detail-only rescue
                            # can use it, but can never renew it indefinitely.
                            automatic_detail_target_hint.remember_box(
                                accepted_player.box,
                                source_timestamp_ns=packet.read_started_ns,
                                track_generation=target_tracker.track_generation,
                                identity_generation=(
                                    automatic_head_runtime.identity_generation
                                ),
                            )
                        direct_body_safe = bool(
                            tracking_activation_active
                            and selected_aim_target is not None
                        )
                        if direct_body_safe:
                            assert selected_aim_target is not None
                            association_player = (
                                accepted_player
                                if accepted_player is not None
                                else selected_aim_target
                            )
                            replacement = automatic_head_runtime.accept_body(
                                association_player.box,
                                # Tracker output remains the identity/selection
                                # authority, but a publishable physical head
                                # coordinate must use this exact source frame's
                                # accepted raw body geometry. Mapping through
                                # the tracker-smoothed box added 10--18 ms of
                                # hidden coordinate lag before the controller.
                                aim_box=association_player.box,
                                corroboration_box=(
                                    accepted_player.box
                                    if accepted_player is not None
                                    else None
                                ),
                                track_generation=target_tracker.track_generation,
                                source_timestamp_ns=packet.read_started_ns,
                            )
                            body_update_deferred = (
                                getattr(
                                    automatic_head_runtime,
                                    "body_update_deferred",
                                    False,
                                )
                                is True
                            )
                            if (
                                automatic_head_runtime.consume_motion_corroboration_revocation()
                                is True
                            ):
                                aim_controller.revoke_motion_corroboration()
                            if replacement and not automatic_revoked_this_frame:
                                automatic_revoked_this_frame = (
                                    _publish_automatic_head_loss_once(
                                        aim_controller,
                                        packet.image.shape,
                                        source_timestamp_ns=packet.read_started_ns,
                                        already_published=(
                                            automatic_revoked_this_frame
                                        ),
                                        )
                                    )
                                automatic_body_fallback_controller_active = False
                            automatic_head_runtime.remember_frame(
                                packet.image,
                                source_timestamp_ns=packet.read_started_ns,
                            )
                            automatic_head_runtime.raise_if_failed()
                            generation_before_poll = (
                                automatic_head_runtime.identity_generation
                            )
                            new_head_sample = automatic_head_runtime.take_latest(
                                now_ns=perf_counter_ns()
                            )
                            if (
                                automatic_head_runtime.consume_motion_corroboration_revocation()
                                is True
                            ):
                                aim_controller.revoke_motion_corroboration()
                            if (
                                automatic_head_runtime.identity_generation
                                != generation_before_poll
                            ):
                                # The completed result belonged geometrically to
                                # an old player. Revoke its controller lease,
                                # then seed only the current player's new epoch.
                                if not automatic_revoked_this_frame:
                                    automatic_revoked_this_frame = (
                                        _publish_automatic_head_loss_once(
                                            aim_controller,
                                            packet.image.shape,
                                            source_timestamp_ns=(
                                                packet.read_started_ns
                                            ),
                                            already_published=(
                                                automatic_revoked_this_frame
                                            ),
                                        )
                                    )
                                automatic_body_fallback_controller_active = False
                                automatic_head_runtime.accept_body(
                                    association_player.box,
                                    aim_box=association_player.box,
                                    corroboration_box=(
                                        accepted_player.box
                                        if accepted_player is not None
                                        else None
                                    ),
                                    track_generation=target_tracker.track_generation,
                                    source_timestamp_ns=packet.read_started_ns,
                                )
                            # The capture card keeps producing while the body
                            # detector is synchronous.  Inspect its one-slot
                            # mailbox without consuming the next detector
                            # input, and allow one strict LK endpoint to move
                            # an already pixel-qualified head into that newer
                            # image. Runtime gates keep the endpoint itself
                            # position-only and preserve the immutable identity
                            # deadline; the atomic handoff below may retain only
                            # authority already proven at the inferred root.
                            peek_latest = getattr(source, "peek_latest", None)
                            newest_capture = (
                                peek_latest() if callable(peek_latest) else None
                            )
                            newest_sequence = getattr(
                                newest_capture,
                                "sequence",
                                None,
                            )
                            newest_source_ns = getattr(
                                newest_capture,
                                "read_started_ns",
                                None,
                            )
                            newest_image = getattr(
                                newest_capture,
                                "image",
                                None,
                            )
                            generation_before_display = (
                                automatic_head_runtime.identity_generation
                            )
                            inferred_head_sample = None
                            if (
                                newest_capture is not None
                                and isinstance(newest_sequence, int)
                                and not isinstance(newest_sequence, bool)
                                and isinstance(newest_source_ns, int)
                                and not isinstance(newest_source_ns, bool)
                                and newest_sequence > packet.sequence
                                and newest_source_ns > packet.read_started_ns
                                and getattr(newest_image, "shape", None)
                                == packet.image.shape
                            ):
                                # Snapshot the exact inferred-frame evidence
                                # before the runtime advances its visible point
                                # into the newer, uninferred capture.  When the
                                # latter succeeds these two samples are
                                # published atomically: the root owns motion
                                # authority and the endpoint owns position.
                                inferred_head_sample = (
                                    automatic_head_runtime.visible_sample(
                                        now_ns=perf_counter_ns(),
                                        frame_shape=packet.image.shape,
                                    )
                                )
                                automatic_head_runtime.remember_newer_capture_frame(
                                    newest_image,
                                    source_timestamp_ns=newest_source_ns,
                                )
                            direct_head_sample = (
                                automatic_head_runtime.visible_sample(
                                    now_ns=perf_counter_ns(),
                                    frame_shape=packet.image.shape,
                                )
                            )
                            if (
                                automatic_head_runtime.identity_generation
                                != generation_before_display
                            ):
                                if not automatic_revoked_this_frame:
                                    automatic_revoked_this_frame = (
                                        _publish_automatic_head_loss_once(
                                            aim_controller,
                                            packet.image.shape,
                                            source_timestamp_ns=(
                                                packet.read_started_ns
                                            ),
                                            already_published=(
                                                automatic_revoked_this_frame
                                            ),
                                        )
                                        )
                                automatic_body_fallback_controller_active = False
                                automatic_head_runtime.accept_body(
                                    association_player.box,
                                    aim_box=association_player.box,
                                    corroboration_box=(
                                        accepted_player.box
                                        if accepted_player is not None
                                        else None
                                    ),
                                    track_generation=target_tracker.track_generation,
                                    source_timestamp_ns=packet.read_started_ns,
                                )
                                direct_head_sample = None
                            assert automatic_body_fallback_gate is not None
                            fallback_query_ns = perf_counter_ns()
                            fallback_deadline_candidate = (
                                automatic_head_runtime.body_fallback_no_decoded_deadline_ns(
                                    now_ns=fallback_query_ns
                                )
                            )
                            body_fallback_identity_deadline_ns = (
                                fallback_deadline_candidate
                                if isinstance(fallback_deadline_candidate, int)
                                and not isinstance(fallback_deadline_candidate, bool)
                                else None
                            )
                            body_fallback_ready = (
                                automatic_body_fallback_gate.observe(
                                    tracker_generation=(
                                        target_tracker.track_generation
                                    ),
                                    runtime_generation=(
                                        automatic_head_runtime.identity_generation
                                    ),
                                    measurement_ns=packet.read_started_ns,
                                    accepted_confidence=(
                                        float(accepted_player.confidence)
                                        if accepted_player is not None
                                        else None
                                    ),
                                    # Aggressive direct-head acquisition may
                                    # schedule verification before the heuristic
                                    # self filter has locked. A body proxy never
                                    # inherits that bypass.
                                    strict_self_safe=aim_self_exclusion_safe,
                                    body_update_deferred=body_update_deferred,
                                    no_decoded_head_verified=(
                                        body_fallback_identity_deadline_ns
                                        is not None
                                    ),
                                    direct_seen=bool(
                                        new_head_sample is not None
                                        or direct_head_sample is not None
                                    ),
                                )
                            )
                            if (
                                direct_head_sample is not None
                                and accepted_player is not None
                                and not direct_head_sample.bridging
                            ):
                                # A direct result establishes the normalized
                                # head anchor. Publish its filtered mapping
                                # through this exact frame's measured primary on
                                # every frame, including decoder gaps. This
                                # keeps one continuous paired observation
                                # schema instead of alternating 15-40 px between
                                # an old raw head and a body-ratio fallback.
                                sample_source_ns = (
                                    direct_head_sample.source_timestamp_ns
                                )
                                if (
                                    automatic_last_controller_source_ns is None
                                    or sample_source_ns
                                    > automatic_last_controller_source_ns
                                ):
                                    correlated_root = inferred_head_sample
                                    correlated_lookahead = bool(
                                        correlated_root is not None
                                        and not correlated_root.bridging
                                        and correlated_root.source_timestamp_ns
                                        == packet.read_started_ns
                                        and (
                                            automatic_last_controller_source_ns
                                            is None
                                            or correlated_root.source_timestamp_ns
                                            >= automatic_last_controller_source_ns
                                        )
                                        and correlated_root.corroboration_point
                                        is not None
                                        and direct_head_sample.phase_advanced
                                        and not direct_head_sample.body_derived_motion_permitted
                                        and direct_head_sample.corroboration_point
                                        is None
                                        and direct_head_sample.source_timestamp_ns
                                        > correlated_root.source_timestamp_ns
                                        and direct_head_sample.track_generation
                                        == correlated_root.track_generation
                                        and direct_head_sample.identity_deadline_ns
                                        == correlated_root.identity_deadline_ns
                                    )
                                    if correlated_lookahead:
                                        assert correlated_root is not None
                                        assert (
                                            correlated_root.corroboration_point
                                            is not None
                                        )
                                        correlated_root_sample = correlated_root
                                        aim_controller.update_correlated_lookahead(
                                            selected_aim_target,
                                            packet.image.shape,
                                            active=True,
                                            primary_measurement_ns=(
                                                correlated_root.source_timestamp_ns
                                            ),
                                            primary_aim_point=(
                                                correlated_root.point
                                            ),
                                            primary_velocity_point=(
                                                correlated_root.point
                                                if correlated_root.velocity_point
                                                is None
                                                else correlated_root.velocity_point
                                            ),
                                            primary_motion_corroboration_point=(
                                                correlated_root.corroboration_point
                                            ),
                                            lookahead_measurement_ns=(
                                                direct_head_sample.source_timestamp_ns
                                            ),
                                            lookahead_aim_point=(
                                                direct_head_sample.point
                                            ),
                                            lookahead_velocity_point=(
                                                direct_head_sample.point
                                                if direct_head_sample.velocity_point
                                                is None
                                                else direct_head_sample.velocity_point
                                            ),
                                            identity_deadline_ns=(
                                                direct_head_sample.identity_deadline_ns
                                            ),
                                            runtime_identity_generation=(
                                                automatic_head_runtime.identity_generation
                                            ),
                                            track_generation=(
                                                direct_head_sample.track_generation
                                            ),
                                            verified_flow_motion=(
                                                bool(
                                                    getattr(
                                                        direct_head_sample,
                                                        "verified_flow_motion",
                                                        False,
                                                    )
                                                )
                                            ),
                                        )
                                    else:
                                        aim_controller.update(
                                            selected_aim_target,
                                            packet.image.shape,
                                            active=True,
                                            measurement_ns=sample_source_ns,
                                            measurement_observed=True,
                                            aim_point=direct_head_sample.point,
                                            velocity_point=(
                                                direct_head_sample.point
                                                if getattr(
                                                    direct_head_sample,
                                                    "velocity_point",
                                                    None,
                                                )
                                                is None
                                                else direct_head_sample.velocity_point
                                            ),
                                            body_derived_motion_permitted=(
                                                direct_head_sample.body_derived_motion_permitted
                                            ),
                                            body_derived_motion_deadline_ns=(
                                                direct_head_sample.body_derived_motion_deadline_ns
                                            ),
                                            identity_deadline_ns=(
                                                direct_head_sample.identity_deadline_ns
                                            ),
                                            **(
                                                {
                                                    "motion_corroboration_point": (
                                                        direct_head_sample.corroboration_point
                                                    )
                                                }
                                                if direct_head_sample.corroboration_point
                                                is not None
                                                else {}
                                            ),
                                        )
                                    automatic_last_controller_source_ns = (
                                        sample_source_ns
                                    )
                                    automatic_body_fallback_controller_active = False
                                    controller_input_source = (
                                        "capture-phase-correlated"
                                        if sample_source_ns
                                        > packet.read_started_ns
                                        and correlated_lookahead
                                        else "capture-phase"
                                        if sample_source_ns
                                        > packet.read_started_ns
                                        else "direct-head"
                                        if new_head_sample is not None
                                        else "carried-head"
                                    )
                                else:
                                    # The newest-frame tap may expose a packet
                                    # one detector iteration before that same
                                    # packet becomes the inferred body input.
                                    # Keep the already-published position; an
                                    # equal timestamp would reset the numeric
                                    # observer as non-monotonic evidence.
                                    controller_input_source = "phase-hold"
                            elif (
                                AUTOMATIC_HEAD_BODY_FALLBACK_CONTROL_ENABLED
                                and accepted_player is not None
                                and not body_update_deferred
                                and body_fallback_ready
                            ):
                                # A fresh, same-identity head crop has already
                                # completed with exactly "no decoded head" (not
                                # ambiguous geometry), and two strong exact body
                                # measurements have agreed. Move only toward the
                                # tracker's smoothed head-ratio point while the
                                # direct model reacquires. This paired sample has
                                # no corroboration/body-motion grant, never
                                # touches DirectHeadAnchor, and expires on the
                                # same short lease as a worker result.
                                assert (
                                    body_fallback_identity_deadline_ns
                                    is not None
                                )
                                fallback_source_ns = packet.read_started_ns
                                if (
                                    automatic_last_controller_source_ns is None
                                    or fallback_source_ns
                                    > automatic_last_controller_source_ns
                                ):
                                    fallback_point = head_target_point(
                                        selected_aim_target,
                                        config.aim_head_ratio,
                                    )
                                    aim_controller.update(
                                        selected_aim_target,
                                        packet.image.shape,
                                        active=True,
                                        measurement_ns=fallback_source_ns,
                                        measurement_observed=True,
                                        aim_point=fallback_point,
                                        velocity_point=fallback_point,
                                        body_derived_motion_permitted=False,
                                        identity_deadline_ns=(
                                            body_fallback_identity_deadline_ns
                                        ),
                                    )
                                    automatic_last_controller_source_ns = (
                                        fallback_source_ns
                                    )
                                    automatic_body_fallback_controller_active = True
                                    controller_input_source = "body-fallback"
                                else:
                                    controller_input_source = "phase-hold"
                            elif (
                                accepted_player is not None
                                and not body_update_deferred
                            ):
                                # Direct-head acquisition intentionally admits
                                # low-confidence body candidates so the head
                                # model can verify small/distant players.  A
                                # body box is therefore scheduling and identity
                                # evidence only unless the narrow decoder-miss
                                # acquisition fallback above has independently
                                # qualified. Live diagnostics showed that weak
                                # or ambiguous boxes can jump hundreds of pixels;
                                # fail closed here and keep submitting crops.
                                if not automatic_revoked_this_frame:
                                    automatic_revoked_this_frame = (
                                        _publish_automatic_head_loss_once(
                                            aim_controller,
                                            packet.image.shape,
                                            source_timestamp_ns=(
                                                packet.read_started_ns
                                            ),
                                            already_published=(
                                                automatic_revoked_this_frame
                                            ),
                                        )
                                        )
                                automatic_body_fallback_controller_active = False
                                controller_input_source = "none"
                            elif accepted_player is not None:
                                # The first same-generation geometry conflict
                                # is quarantined for one confirming sample. It
                                # publishes neither a new point nor an explicit
                                # loss, so the numeric core may retain only its
                                # already-bounded prior observation.
                                if (
                                    automatic_body_fallback_controller_active
                                    and not automatic_revoked_this_frame
                                ):
                                    automatic_revoked_this_frame = (
                                        _publish_automatic_head_loss_once(
                                            aim_controller,
                                            packet.image.shape,
                                            source_timestamp_ns=(
                                                packet.read_started_ns
                                            ),
                                            already_published=(
                                                automatic_revoked_this_frame
                                            ),
                                        )
                                    )
                                automatic_body_fallback_controller_active = False
                                controller_input_source = "none"
                            elif selected_aim_target is not None:
                                # No controller publication occurs here.  The
                                # numeric core may retain only its already-
                                # bounded prior observation; the tracker's
                                # synthetic box is diagnostic/display state.
                                if (
                                    automatic_body_fallback_controller_active
                                    and not automatic_revoked_this_frame
                                ):
                                    automatic_revoked_this_frame = (
                                        _publish_automatic_head_loss_once(
                                            aim_controller,
                                            packet.image.shape,
                                            source_timestamp_ns=(
                                                packet.read_started_ns
                                            ),
                                            already_published=(
                                                automatic_revoked_this_frame
                                            ),
                                        )
                                    )
                                automatic_body_fallback_controller_active = False
                                controller_input_source = "none"
                            if (
                                accepted_player is not None
                                and not body_update_deferred
                            ):
                                # The head crop and association must use this
                                # exact source frame's accepted primary box.
                                # A prior-frame box is stale geometry during
                                # commanded camera motion and can manufacture
                                # clustered no-head results.
                                automatic_head_runtime.submit(
                                    packet.image,
                                    accepted_player,
                                    source_timestamp_ns=packet.read_started_ns,
                                )
                        else:
                            assert automatic_body_fallback_gate is not None
                            if automatic_detail_tracker_update_safe:
                                automatic_body_fallback_gate.pause()
                            else:
                                automatic_body_fallback_gate.reset()
                            # A normal target-empty interval may outlast the
                            # tracker's short prediction bridge while its same
                            # logical identity remains recoverable. Pause all
                            # output but retain an unexpired verified anchor so
                            # that exact same generation can resume immediately.
                            # Guard, self, runtime, and button failures remain
                            # hard identity revocations and can never inherit it.
                            ordinary_same_identity_gap = bool(
                                automatic_detail_tracker_update_safe
                                and aim_self_exclusion_safe
                                and selected_aim_target is None
                                and not aim_detections
                                and not aim_continuation_detections
                            )
                            if ordinary_same_identity_gap:
                                body_gap_was_suspended = (
                                    getattr(
                                        automatic_head_runtime,
                                        "body_gap_suspended",
                                        False,
                                    )
                                    is True
                                )
                                body_gap_suspension_valid = (
                                    automatic_head_runtime.suspend_body_gap(
                                        now_ns=perf_counter_ns()
                                    )
                                )
                                if body_gap_suspension_valid:
                                    body_state_changed = (
                                        not body_gap_was_suspended
                                    )
                                else:
                                    body_state_changed = (
                                        automatic_head_runtime.revoke_body()
                                    )
                            else:
                                body_state_changed = (
                                    automatic_head_runtime.revoke_body()
                                )
                            if (
                                (
                                    body_state_changed
                                    or automatic_body_fallback_controller_active
                                )
                                and not automatic_revoked_this_frame
                            ):
                                automatic_revoked_this_frame = (
                                    _publish_automatic_head_loss_once(
                                        aim_controller,
                                        packet.image.shape,
                                        source_timestamp_ns=packet.read_started_ns,
                                        already_published=(
                                            automatic_revoked_this_frame
                                        ),
                                    )
                                )
                            automatic_body_fallback_controller_active = False
                            automatic_head_runtime.raise_if_failed()
                    else:
                        aim_controller.update(
                            selected_aim_target,
                            packet.image.shape,
                            active=active,
                            **(
                                {
                                    "measurement_ns": packet.read_started_ns,
                                    "measurement_observed": not (
                                        target_tracker is not None
                                        and target_tracker.output_is_prediction
                                    ),
                                }
                                if isinstance(aim_controller, MakcuAimingController)
                                else {}
                            ),
                        )
                        if selected_aim_target is None:
                            controller_input_source = "none"
                        elif (
                            target_tracker is not None
                            and bool(
                                getattr(
                                    target_tracker,
                                    "output_is_prediction",
                                    False,
                                )
                            )
                        ):
                            controller_input_source = "prediction-hold"
                        else:
                            controller_input_source = "stable-body"
                    aim_engaged = (
                        (
                            automatic_frame_activation_active
                            if automatic_frame_activation_active is not None
                            else aim_controller.activation_pressed
                        )
                        if isinstance(aim_controller, MakcuAimingController)
                        else active
                    )
                status_target = selected_aim_target
                aim_status = _aim_status(
                    runtime_enabled=aim_runtime_enabled,
                    self_exclusion_ready=(
                        aim_self_exclusion_safe or aggressive_direct_head_mode
                    ),
                    selected_target=status_target,
                    engaged=aim_engaged,
                    activation_name=aim_activation_name,
                    control_description=aim_control_description,
                )
                if activation_hold_expired:
                    aim_status = (
                        "aim paused: continuous hold safety limit reached; "
                        f"release {aim_activation_name}, then press it again"
                    )
                elif automatic_head_runtime is not None:
                    if not tracking_activation_active:
                        aim_status = (
                            f"aim ready: hold {aim_activation_name} to run "
                            "direct-head tracking"
                        )
                    elif controller_input_source == "body-fallback":
                        aim_status = (
                            "aim acquiring: strong measured player + verified "
                            "no-head decoder miss | position-only body proxy"
                        )
                    elif selected_aim_target is not None and direct_head_sample is None:
                        aim_status = (
                            "aim paused: direct-head anchor pending; body target is "
                            "identity-only"
                        )
                    elif direct_head_sample is not None:
                        remaining_ms = max(
                            0.0,
                            (
                                getattr(
                                    direct_head_sample,
                                    "identity_deadline_ns",
                                    direct_head_sample.source_timestamp_ns
                                    + round(
                                        DIRECT_HEAD_ANCHOR_MAX_AGE_SECONDS
                                        * 1_000_000_000
                                    ),
                                )
                                - perf_counter_ns()
                            )
                            / 1e6,
                        )
                        if direct_head_sample.bridging:
                            aim_status = (
                                "aim bridge visible: primary is predicted; physical "
                                "aim is not renewed | direct anchor expires in "
                                f"{remaining_ms:.0f} ms"
                            )
                        else:
                            aim_status = (
                                "aim anchored: direct head carried by current measured "
                                f"player | expires in {remaining_ms:.0f} ms without "
                                "another direct head"
                            )
            if tracking_path_telemetry is not None:
                accepted_measurement = (
                    target_tracker.accepted_measurement
                    if target_tracker is not None
                    else None
                )
                tracking_path_telemetry.record(
                    accepted_player=accepted_measurement is not None,
                    blocked_self_filter=(
                        (not aim_self_exclusion_safe)
                        and selected_aim_target is None
                    ),
                    direct_anchor=direct_head_sample is not None,
                    body_fallback=(
                        controller_input_source == "body-fallback"
                    ),
                )
            result_ready_ns = perf_counter_ns()

            if aim_diagnostic_recorder is not None:
                accepted_measurement = (
                    target_tracker.accepted_measurement
                    if target_tracker is not None
                    else None
                )
                selected_is_prediction = bool(
                    target_tracker is not None
                    and target_tracker.output_is_prediction
                )
                try:
                    aim_diagnostic_recorder.submit(
                        packet.image,
                        {
                            "frame_sequence": int(packet.sequence),
                            "source_timestamp_ns": int(packet.read_started_ns),
                            "frame_shape": [
                                int(value) for value in packet.image.shape
                            ],
                            "tracking_mode": config.aim_makcu_tracking_mode,
                            "activation_pressed": bool(
                                tracking_activation_active
                            ),
                            "raw_activation_known": raw_activation_known,
                            "raw_activation_pressed": raw_activation_pressed,
                            "activation_requires_release": (
                                activation_requires_release
                            ),
                            "activation_denial_reason": (
                                "continuous-hold-expired"
                                if activation_hold_expired
                                else None
                            ),
                            "self_exclusion_ready": bool(self_exclusion_ready),
                            "aim_self_exclusion_safe": bool(
                                aim_self_exclusion_safe
                            ),
                            "verified_flow_self_continuation": bool(
                                verified_flow_continuation
                            ),
                            "hard_guard_revoked_prediction_grace": bool(
                                hard_guard_revoked_prediction_grace
                            ),
                            "aim_candidates": [
                                _aim_diagnostic_detection(item)
                                for item in aim_detections
                            ],
                            "continuation_candidates": [
                                _aim_diagnostic_detection(item)
                                for item in aim_continuation_detections
                            ],
                            "accepted_measurement": _aim_diagnostic_detection(
                                accepted_measurement
                            ),
                            "selected_target": _aim_diagnostic_detection(
                                selected_aim_target
                            ),
                            "selected_is_prediction": selected_is_prediction,
                            "direct_head_sample": _aim_diagnostic_head_sample(
                                new_head_sample
                            ),
                            "visible_head_sample": _aim_diagnostic_head_sample(
                                direct_head_sample
                            ),
                            "correlated_root_sample": _aim_diagnostic_head_sample(
                                correlated_root_sample
                            ),
                            "control_source": controller_input_source,
                            "makcu_control": (
                                _aim_diagnostic_makcu_control(aim_controller)
                                if isinstance(
                                    aim_controller,
                                    MakcuAimingController,
                                )
                                else None
                            ),
                            "aim_engaged": bool(aim_engaged),
                            "aim_status": str(aim_status),
                            "capture_to_result_ms": max(
                                0.0,
                                (
                                    result_ready_ns - packet.read_started_ns
                                )
                                / 1e6,
                            ),
                        },
                    )
                    diagnostic_error = aim_diagnostic_recorder.status.error
                    if diagnostic_error and not aim_diagnostic_warning_printed:
                        print(
                            f"Warning: {diagnostic_error}",
                            file=sys.stderr,
                        )
                        aim_diagnostic_warning_printed = True
                except (RuntimeError, TypeError, ValueError) as exc:
                    if not aim_diagnostic_warning_printed:
                        print(
                            f"Warning: aim diagnostic sample rejected: {exc}",
                            file=sys.stderr,
                        )
                        aim_diagnostic_warning_printed = True

            if prepared.crop_was_clamped and not crop_warning_printed:
                print(
                    "Warning: --crop-size exceeded the source dimensions and was "
                    "clamped to the largest centered square.",
                    file=sys.stderr,
                )
                crop_warning_printed = True

            skipped_frames = source.stats.frames_overwritten
            render_preview = bool(
                preview_pacer is not None
                and preview_pacer.should_render(result_ready_ns)
            )
            draw_started_ns = result_ready_ns
            if config.draw and render_preview:
                # The overlay intentionally shows the completed prior sample so
                # its own drawing cost can be measured in the current sample.
                if self_zone is not None:
                    assert last_ignored_count is not None
                    draw_ignore_zone(
                        packet.image,
                        self_zone,
                        last_ignored_count,
                        last_ignored_detection,
                    )
                draw_detections(packet.image, detections)
                if automatic_head_runtime is not None:
                    if direct_head_sample is not None:
                        draw_aim_target(
                            packet.image,
                            direct_head_sample.point,
                            active=aim_engaged,
                            activation_name=aim_activation_name,
                            source_label=(
                                "direct anchor bridge"
                                if direct_head_sample.bridging
                                else "direct head anchored"
                            ),
                        )
                elif aim_runtime_enabled and selected_aim_target is not None:
                    draw_aim_target(
                        packet.image,
                        head_target_point(selected_aim_target, config.aim_head_ratio),
                        active=aim_engaged,
                        activation_name=aim_activation_name,
                        source_label="body-box head proxy",
                    )
                draw_metrics(
                    packet.image,
                    display_snapshot,
                    skipped_frames,
                    ignored_count=last_ignored_count,
                    aim_status=aim_status,
                )
            draw_completed_ns = perf_counter_ns()

            continue_running = True
            if render_preview:
                assert preview_window is not None
                preview_service_started_ns = perf_counter_ns()
                continue_running = preview_window.submit(packet.image)
                preview_service_ms = (
                    perf_counter_ns() - preview_service_started_ns
                ) / 1e6
            elif preview_window is not None:
                # Inline HighGUI is serviced by submit at preview_fps; polling
                # it on every faster inference frame costs milliseconds on Qt.
                # The threaded Windows implementation polls independently.
                continue_running = preview_window.should_continue()

            timings = FrameTimings(
                capture_ms=(packet.read_completed_ns - packet.read_started_ns) / 1e6,
                queue_age_ms=max(0, processing_started_ns - packet.read_completed_ns) / 1e6,
                preprocess_ms=(preprocessing_completed_ns - preprocessing_started_ns) / 1e6,
                inference_ms=(inference_completed_ns - inference_started_ns) / 1e6,
                postprocess_ms=(postprocess_completed_ns - inference_completed_ns) / 1e6,
                detail_preprocess_ms=detail_preprocess_ms,
                detail_inference_ms=detail_inference_ms,
                detail_postprocess_ms=detail_postprocess_ms,
                control_ms=(result_ready_ns - detections_ready_ns) / 1e6,
                processing_ms=(result_ready_ns - processing_started_ns) / 1e6,
                freshness_latency_ms=max(0, result_ready_ns - packet.read_completed_ns) / 1e6,
                observed_pipeline_ms=max(0, result_ready_ns - packet.read_started_ns) / 1e6,
                draw_ms=(draw_completed_ns - draw_started_ns) / 1e6,
                # Threaded Windows mode measures owned copy/mailbox submission;
                # inline compatibility mode measures HighGUI submission/event
                # service. Neither is a measurement of display scanout.
                preview_service_ms=preview_service_ms,
            )
            metrics.record(timings, result_ready_ns)
            processed_frames += 1

            if result_ready_ns - last_report_ns >= 1_000_000_000:
                display_snapshot = metrics.snapshot()
                summary = console_summary(
                    display_snapshot,
                    skipped_frames,
                    ignored_count=last_ignored_count,
                )
                if (
                    target_tracker is not None
                    and tracker_report_snapshot is not None
                    and tracker_report_ns is not None
                ):
                    current_tracker_telemetry = target_tracker.telemetry_snapshot()
                    summary += " | " + _target_tracker_telemetry_summary(
                        tracker_report_snapshot,
                        current_tracker_telemetry,
                        (result_ready_ns - tracker_report_ns) / 1_000_000_000,
                    )
                    tracker_report_snapshot = current_tracker_telemetry
                    tracker_report_ns = result_ready_ns
                if (
                    aim_input_telemetry is not None
                    and aim_input_report_snapshot is not None
                    and aim_input_report_ns is not None
                ):
                    current_aim_input = aim_input_telemetry.snapshot()
                    summary += " | " + _aim_input_telemetry_summary(
                        aim_input_report_snapshot,
                        current_aim_input,
                        (result_ready_ns - aim_input_report_ns) / 1_000_000_000,
                    )
                    aim_input_report_snapshot = current_aim_input
                    aim_input_report_ns = result_ready_ns
                if automatic_detail_rescue_enabled:
                    current_detail_telemetry = (
                        automatic_detail_telemetry.snapshot()
                    )
                    summary += " | " + _automatic_detail_telemetry_summary(
                        automatic_detail_report_snapshot,
                        current_detail_telemetry,
                        (result_ready_ns - last_report_ns) / 1_000_000_000,
                    )
                    automatic_detail_report_snapshot = current_detail_telemetry
                if (
                    automatic_head_runtime is not None
                    and head_report_snapshot is not None
                    and head_report_ns is not None
                ):
                    current_head_status = automatic_head_runtime.status
                    summary += " | " + _head_runtime_telemetry_summary(
                        head_report_snapshot,
                        current_head_status,
                        (result_ready_ns - head_report_ns) / 1_000_000_000,
                        now_ns=result_ready_ns,
                        visible_sample=direct_head_sample,
                    )
                    head_report_snapshot = current_head_status
                    head_report_ns = result_ready_ns
                if (
                    isinstance(aim_controller, MakcuAimingController)
                    and makcu_report_snapshot is not None
                    and makcu_report_ns is not None
                ):
                    current_telemetry = aim_controller.telemetry_snapshot()
                    telemetry_snapshot_ns = perf_counter_ns()
                    summary += " | " + _makcu_telemetry_summary(
                        makcu_report_snapshot,
                        current_telemetry,
                        (telemetry_snapshot_ns - makcu_report_ns) / 1_000_000_000,
                    )
                    makcu_report_snapshot = current_telemetry
                    makcu_report_ns = telemetry_snapshot_ns
                if (
                    tracking_path_telemetry is not None
                    and tracking_path_report_snapshot is not None
                    and tracking_path_report_ns is not None
                ):
                    current_tracking_path = tracking_path_telemetry.snapshot()
                    summary += " | " + _tracking_path_telemetry_summary(
                        tracking_path_report_snapshot,
                        current_tracking_path,
                        (result_ready_ns - tracking_path_report_ns)
                        / 1_000_000_000,
                    )
                    tracking_path_report_snapshot = current_tracking_path
                    tracking_path_report_ns = result_ready_ns
                if calibration_status is not None:
                    assert isinstance(aim_controller, MakcuAimingController)
                    raw_known, raw_pressed = aim_controller.raw_activation_state
                    raw_button_state = (
                        "pressed" if raw_pressed else "released"
                    ) if raw_known else "unknown"
                    summary += (
                        f" | CAL {calibration_status.state.value} | "
                        f"{calibration_target_readiness} | "
                        f"raw button {raw_button_state} | "
                        f"counts {calibration_status.emitted_abs_counts}/2400 | "
                        "qualifying X +/- "
                        f"{calibration_status.qualifying_x_positive}/"
                        f"{calibration_status.qualifying_x_negative} | Y +/- "
                        f"{calibration_status.qualifying_y_positive}/"
                        f"{calibration_status.qualifying_y_negative}"
                    )
                print(summary)
                last_report_ns = result_ready_ns
            if calibration_status is not None and calibration_status.terminal:
                assert calibration_session is not None
                assert calibration_session.result is not None
                termination_reason = (
                    "aim_calibration_success"
                    if calibration_session.result.outcome == "success"
                    else "aim_calibration_aborted"
                )
                break
            if (
                config.max_frames is not None
                and processed_frames >= config.max_frames
            ):
                termination_reason = "max_frames"
                break
            if deadline_ns is not None and perf_counter_ns() >= deadline_ns:
                termination_reason = "max_seconds"
                break
            if not continue_running:
                termination_reason = "preview_closed"
                break
        pipeline_completed_ns = perf_counter_ns()
    finally:
        primary_exception_active = sys.exc_info()[0] is not None

        def record_cleanup_failure(component: str, detail: object) -> None:
            message = f"{component}: {detail}"
            cleanup_failures.append(message)
            if primary_exception_active:
                # Preserve the original detector/capture exception while still
                # exposing every failed cleanup operation to the operator.
                print(f"Warning: cleanup also failed: {message}", file=sys.stderr)

        if calibration_session is not None:
            if not calibration_session.terminal:
                try:
                    calibration_status = calibration_session.abort(
                        "pipeline stopped before calibration completed",
                        now_ns=perf_counter_ns(),
                    )
                except Exception as exc:  # noqa: BLE001 - still stop physical output
                    record_cleanup_failure("calibration abort", exc)
            result = calibration_session.result
            if result is None:
                record_cleanup_failure(
                    "calibration evidence",
                    "the session produced no terminal result",
                )
            elif not calibration_evidence_written:
                try:
                    from aiming.makcu_calibration_session import (
                        write_session_evidence_exclusive,
                    )

                    assert config.aim_calibration_evidence is not None
                    write_session_evidence_exclusive(
                        config.aim_calibration_evidence,
                        result.evidence,
                    )
                    calibration_evidence_written = True
                except Exception as exc:  # noqa: BLE001 - still stop physical output
                    record_cleanup_failure("calibration evidence publication", exc)

        # Stop physical output before waiting for any auxiliary model worker.
        # This makes pipeline exit an immediate movement revocation even if a
        # head inference takes its full bounded join interval to return.
        if aim_controller is not None:
            try:
                aim_controller.stop()
            except Exception as exc:  # noqa: BLE001 - aggregate cleanup failures
                record_cleanup_failure("aim output shutdown", exc)
        if automatic_head_runtime is not None:
            try:
                head_worker_stopped = automatic_head_runtime.stop()
            except Exception as exc:  # noqa: BLE001 - continue physical shutdown
                record_cleanup_failure("direct-head worker shutdown", exc)
            else:
                if not head_worker_stopped:
                    record_cleanup_failure(
                        "direct-head worker shutdown",
                        "the direct-head localization worker did not stop within its bounded timeout",
                    )
                try:
                    automatic_head_runtime.raise_if_failed()
                except Exception as exc:  # noqa: BLE001 - report after bounded stop
                    record_cleanup_failure("direct-head worker", exc)
        if aim_diagnostic_recorder is not None:
            try:
                diagnostic_stopped = aim_diagnostic_recorder.stop()
            except Exception as exc:  # noqa: BLE001 - diagnostics are non-critical
                print(
                    f"Warning: aim diagnostic shutdown failed: {exc}",
                    file=sys.stderr,
                )
            else:
                if diagnostic_stopped:
                    try:
                        from scripts.replay_aim_diagnostic import replay_session

                        replay_report = replay_session(
                            aim_diagnostic_recorder.session_dir
                        )
                        replay_path = (
                            aim_diagnostic_recorder.session_dir
                            / "replay-report.json"
                        )
                        replay_path.write_text(
                            json.dumps(
                                replay_report,
                                ensure_ascii=True,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        print(
                            f"Warning: automatic aim replay analysis failed: {exc}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            "Automatic aim replay: "
                            f"{replay_report['status']} | primary measurements "
                            f"{replay_report['primary_measurement_rate'] * 100.0:.0f}% | "
                            f"target output {replay_report['target_output_rate'] * 100.0:.0f}% | "
                            "controller target publication "
                            f"{replay_report['controller_target_publication_rate'] * 100.0:.0f}% | "
                            "visible head anchor "
                            f"{replay_report['visible_head_anchor_coverage'] * 100.0:.0f}% | "
                            "new direct samples "
                            f"{replay_report['new_direct_head_sample_rate'] * 100.0:.0f}% | "
                            "continuous-hold pauses "
                            f"{replay_report['continuous_hold_expired_events']} | "
                            f"report {replay_path}"
                        )
                    print(
                        "Automatic aim diagnostic written: "
                        f"{aim_diagnostic_recorder.manifest_path}"
                    )
                else:
                    print(
                        "Warning: automatic aim diagnostic did not finish cleanly: "
                        f"{aim_diagnostic_recorder.status.error or 'writer timeout'}",
                        file=sys.stderr,
                    )
        if aim_sensor is not None:
            try:
                aim_sensor.stop()
            except Exception as exc:  # noqa: BLE001 - aggregate bounded cleanup failures
                record_cleanup_failure("activation sensor shutdown", exc)
        source_error_before_close = source.error
        try:
            source.close()
        except Exception as exc:  # noqa: BLE001 - still attempt preview shutdown
            record_cleanup_failure("capture shutdown", exc)
        else:
            if source.error and (
                not primary_exception_active or source.error != source_error_before_close
            ):
                record_cleanup_failure("capture shutdown", source.error)
        if preview_window is not None:
            try:
                preview_stopped = preview_window.stop()
            except Exception as exc:  # noqa: BLE001 - report as qualification failure
                record_cleanup_failure("preview shutdown", exc)
            else:
                if not preview_stopped:
                    record_cleanup_failure(
                        "preview shutdown",
                        "the OpenCV preview worker did not stop within its bounded timeout",
                    )
                try:
                    preview_window.raise_if_failed()
                except Exception as exc:  # noqa: BLE001 - report worker failure after cleanup
                    record_cleanup_failure("preview worker", exc)

    if cleanup_failures:
        raise RuntimeError("Pipeline cleanup failed: " + "; ".join(cleanup_failures))

    if (
        report_destination is not None
        or active_profile is not None
        or direct_head_plant_profile is not None
    ):
        from utils.live_report import verify_artifact_unchanged

        assert model_artifact_snapshot is not None
        assert labels_artifact_snapshot is not None
        verify_artifact_unchanged(
            config.model_path,
            model_artifact_snapshot,
            description="Model artifact",
        )
        verify_artifact_unchanged(
            config.labels_path,
            labels_artifact_snapshot,
            description="Labels artifact",
        )

    final = metrics.snapshot()
    capture_stats = source.stats
    if final.processed_frames:
        print(
            console_summary(
                final,
                capture_stats.frames_overwritten,
                ignored_count=last_ignored_count,
            )
        )
    print(
        f"Stopped after {final.processed_frames} processed frame(s); "
        f"{capture_stats.frames_overwritten} application overwrite(s), "
        f"{capture_stats.read_failures} capture failure(s)."
    )
    if report_destination is not None:
        from utils.live_report import build_live_report, utc_now, write_json_atomic_new

        if pipeline_started_ns is None:
            # Source/preview startup errors propagate before this point, so the
            # branch is defensive for custom CaptureSource implementations.
            elapsed_seconds = 0.0
        else:
            completed_ns = pipeline_completed_ns or perf_counter_ns()
            elapsed_seconds = max(0.0, completed_ns - pipeline_started_ns) / 1e9
        preview_mode = preview_window.mode if preview_window is not None else "disabled"
        preview_stats = preview_window.stats if preview_window is not None else {}
        detail_pass_report = detail_pass_stats.snapshot()
        detail_pass_report.update(
            {
                "mode": detail_pass_mode,
                "configured_crop_size": config.detail_crop_size,
                "effective_crop_size": effective_detail_crop_size,
                "automatic_activation_gated": automatic_detail_rescue_enabled,
                "automatic_need_gated": automatic_detail_rescue_enabled,
                "automatic_need_confidence": (
                    aim_configured_confidence
                    if automatic_detail_rescue_enabled
                    else None
                ),
                "automatic_self_guarded_need": (
                    automatic_detail_rescue_enabled
                ),
                "automatic_self_relative_detail_exclusion_enabled": (
                    automatic_detail_rescue_enabled
                ),
                "automatic_lower_edge_self_fragment_exclusion_enabled": (
                    automatic_detail_rescue_enabled
                ),
                "automatic_lower_edge_self_fragment_margin_model_pixels": (
                    AUTOMATIC_DETAIL_SELF_EDGE_MARGIN_MODEL_PIXELS
                    if automatic_detail_rescue_enabled
                    else None
                ),
                "automatic_small_target_reference_height": (
                    AUTOMATIC_DETAIL_REFERENCE_HEIGHT
                    if automatic_detail_rescue_enabled
                    else None
                ),
                "automatic_small_target_max_reference_height": (
                    AUTOMATIC_DETAIL_MAX_REFERENCE_HEIGHT
                    if automatic_detail_rescue_enabled
                    else None
                ),
            }
        )
        detail_pass_report.update(automatic_detail_telemetry.snapshot())
        report = build_live_report(
            config=config,
            detector_summary=detector.runtime_summary,
            source_description=source.description,
            source_settings=last_source_settings or source.actual_settings,
            capture_stats=capture_stats,
            preview_mode=preview_mode,
            preview_stats=preview_stats,
            metrics=final,
            elapsed_seconds=elapsed_seconds,
            started_utc=pipeline_started_utc or utc_now(),
            completed_utc=utc_now(),
            termination_reason=termination_reason,
            detail_pass_stats=detail_pass_report,
            model_artifact_snapshot=model_artifact_snapshot,
            labels_artifact_snapshot=labels_artifact_snapshot,
        )
        written = write_json_atomic_new(report_destination, report)
        print(f"Live pipeline metrics written to {written}")
    if calibration_requested:
        assert calibration_session is not None
        assert calibration_session.result is not None
        assert calibration_evidence_written
        calibration_result = calibration_session.result
        assert config.aim_calibration_evidence is not None
        print(
            "MAKCU calibration evidence written: "
            f"{config.aim_calibration_evidence} | "
            f"outcome {calibration_result.outcome} | "
            f"artifact {calibration_result.evidence.artifact_sha256}"
        )
        if calibration_result.outcome != "success":
            raise RuntimeError(
                f"MAKCU calibration aborted: {calibration_result.reason}"
            )
        assert calibration_result.fit is not None
        print(
            "MAKCU calibration fit passed: "
            f"X {calibration_result.fit.x.gain_pixels_per_count:.6g} px/count | "
            f"Y {calibration_result.fit.y.gain_pixels_per_count:.6g} px/count | "
            f"delay {calibration_result.fit.delay_seconds * 1000.0:.2f} ms | "
            "profile remains inactive pending explicit review"
        )
    return 0


def main() -> int:
    config = parse_args()
    try:
        return run(config)
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
