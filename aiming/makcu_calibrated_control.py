"""Pure calibrated screen-error control for a delayed MAKCU mouse plant.

The detector observes screen-space error much more slowly than the MAKCU can
emit relative mouse counts.  A derivative of raw error therefore mixes two
different motions: independent target motion and the delayed response to our
own commands.  This module keeps those terms separate using the calibrated
plant equation::

    delta(error) = target_motion - gain * delayed_emitted_counts

Only counts which were actually emitted belong in that equation.  Requested
or fractional counts are deliberately not accepted as evidence.

This is a numeric core: it owns no threads, clocks, serial ports, or physical
authorization.  Its caller supplies monotonic timestamps and the exact
successful command records, and must continue to enforce the physical gate.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
import math
import statistics


__all__ = (
    "CalibratedControlConfig",
    "CalibratedControlOutput",
    "CalibratedPlant",
    "CORRELATED_LOOKAHEAD_MAX_LEAD_SECONDS",
    "CorrelatedLookaheadObservation",
    "EmittedMouseCommand",
    "MakcuCalibratedController",
    "ScreenErrorObservation",
)

_NS_PER_SECOND = 1_000_000_000

# The paired raw-point path is deliberately a small, dependency-free Kalman
# observer.  These values describe detector uncertainty, rather than tuning a
# differentiator.  The 2.5-pixel floor grows by a conservative fraction of the
# measured raw-versus-track residual on each sample, covering the observed
# 6--11 px player-box excursions without hiding two coordinates which genuinely
# agree.  Acceleration noise is derived from the existing bounded target-
# acceleration setting below.
_PAIRED_MEASUREMENT_SIGMA_PIXELS = 2.5
_PAIRED_TRACK_RESIDUAL_VARIANCE_FRACTION = 0.10
_PAIRED_INITIAL_VELOCITY_SIGMA_PIXELS_PER_SECOND = 1500.0
_PAIRED_MINIMUM_ACCELERATION_SIGMA_PIXELS_PER_SECOND_SQUARED = 1000.0
_PAIRED_MAXIMUM_ACCELERATION_SIGMA_PIXELS_PER_SECOND_SQUARED = 8000.0
_PAIRED_ACCELERATION_SIGMA_FRACTION = 0.40
_PAIRED_PLANT_GAIN_UNCERTAINTY_FRACTION = 0.20
_PAIRED_INNOVATION_GATE_MAHALANOBIS_SQUARED = 16.0
_PAIRED_REJECTIONS_BEFORE_RESEED = 3
_PAIRED_FEEDBACK_ENTER_SIGMAS = 1.25
_PAIRED_FEEDBACK_EXIT_SIGMAS = 2.0
_PAIRED_HELD_FEEDBACK_FRACTION = 0.75
_PAIRED_FEEDFORWARD_FULL_VELOCITY_SIGMA = 200.0
_PAIRED_FEEDFORWARD_ZERO_VELOCITY_SIGMA = 600.0
_PAIRED_FEEDFORWARD_ZERO_SIGNAL_TO_NOISE = 0.5
_PAIRED_FEEDFORWARD_FULL_SIGNAL_TO_NOISE = 2.0
_PAIRED_FULL_POSITION_CHANNEL_AGREEMENT_PIXELS = 8.0
_PAIRED_ZERO_POSITION_CHANNEL_AGREEMENT_PIXELS = 16.0
# Automatic direct-head position intentionally has more low-pass filtering
# than its translation-first velocity point. During real fast motion the
# upstream point therefore leads the tracked position by a short, predictable
# phase offset. Forgive only a bounded component in the direction of observed
# tracked-position motion; perpendicular, opposite, and raw-only staircase
# disagreement still reaches the unchanged 8/16 px rejection ramp.
_BODY_DERIVED_POSITION_CHANNEL_PHASE_SECONDS = 0.012
_BODY_DERIVED_MAXIMUM_POSITION_PHASE_ALLOWANCE_PIXELS = 24.0
# The mapped-body path can publish at the primary detector cadence even though
# its noise is still one temporally correlated detector signal.  Treat 46 Hz as
# the measured information-rate reference: faster publication increases R
# instead of making the same noisy geometry look more certain merely because
# it arrived in more, smaller steps.  Slower callers retain the historical R;
# the normalization is deliberately damping-only.
_BODY_DERIVED_MEASUREMENT_REFERENCE_HZ = 46.0
_BODY_DERIVED_MAXIMUM_MEASUREMENT_VARIANCE_SCALE = 4.0
# An opposite measured step is a reversal only when its position innovation is
# also outside the cadence-scaled observer uncertainty.  This keeps one model-
# pixel staircase or ordinary difference noise from repeatedly disarming slow,
# coherent motion while still vetoing a genuine fast reversal before the
# smoothed velocity crosses zero.
_BODY_DERIVED_REVERSAL_MINIMUM_INNOVATION_SIGMAS = 2.5
_BODY_DERIVED_IMMEDIATE_REVERSAL_MINIMUM_SPEED_PIXELS_PER_SECOND = 500.0
_CORROBORATION_MEASUREMENT_SIGMA_PIXELS = 6.0
_CORROBORATION_INNOVATION_GATE_MAHALANOBIS_SQUARED = 25.0
_CORROBORATION_ZERO_DIRECTION_COSINE = 0.55
_CORROBORATION_FULL_DIRECTION_COSINE = 0.90
_CORROBORATION_ZERO_SPEED_RATIO = 0.30
_CORROBORATION_FULL_SPEED_RATIO = 0.70
_CORROBORATION_ZERO_SIGNAL_TO_NOISE = 0.50
_CORROBORATION_FULL_SIGNAL_TO_NOISE = 1.75
_CORROBORATION_MINIMUM_DIRECTION_COSINE = 0.90
_CORROBORATION_ZERO_DIRECTION_PERSISTENCE_SECONDS = 0.016
_CORROBORATION_FULL_DIRECTION_PERSISTENCE_SECONDS = 0.050
_BODY_DERIVED_MINIMUM_DIRECTION_COSINE = 0.90
# A mapped measured-primary velocity is still one temporally correlated body
# detector signal.  Require the same 16--50 ms coherent direction horizon as
# independent corroboration before it can add open-loop feed-forward.  The
# shorter 8--35 ms experiment visibly amplified live detector jitter.
_BODY_DERIVED_ZERO_DIRECTION_PERSISTENCE_SECONDS = 0.016
_BODY_DERIVED_FULL_DIRECTION_PERSISTENCE_SECONDS = 0.050
_CORROBORATION_RISE_TIME_CONSTANT_SECONDS = 0.050
_CORROBORATION_FALL_TIME_CONSTANT_SECONDS = 0.010
# Positional lead is a short, closed-loop horizon rather than an open-loop
# velocity command. Once independent evidence has reached half confidence it
# may use that bounded horizon fully; explicit feed-forward remains scaled by
# the unmodified, slower confidence. This preserves prompt pursuit without
# weakening the exact-zero revoke boundary.
_CORROBORATION_FULL_MOTION_PROJECTION_CONFIDENCE = 0.50
# A body-mapped head coordinate and the body box which translated it are one
# item of evidence, not two independent motion measurements.  The automatic
# path may use the ordinary 25-to-50% ceiling only after the existing
# covariance/direction-persistence and residual gates agree. The separate
# high-speed reserve defined below also requires newest-frame physical motion,
# so stationary/reversal noise cannot receive its enlarged authority.
_MAXIMUM_BODY_DERIVED_FEEDFORWARD_FRACTION = 0.50
_BODY_DERIVED_BASELINE_FEEDFORWARD_FRACTION = 0.25
# The ordinary body-derived ceiling deliberately remains conservative.  A
# separate reserve may approach full velocity only during fast, coherent
# pursuit.  Its gates use command-compensated motion from the newest source
# frame, so a stop, reversal, or manual approach closes the reserve without
# waiting for the Kalman velocity to decay.
_MAXIMUM_BODY_DERIVED_PURSUIT_FEEDFORWARD_FRACTION = 0.95
# The immediate reserve remains confined to the very-fast regime. A separate
# residual-trained pursuit state below learns the missing constant-velocity
# command only after coherent motion has produced real, same-direction lag.
# That state covers the much more common 900--1400 px/s per-axis range without
# granting medium-speed detector jitter open-loop authority.
_BODY_DERIVED_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND = 1400.0
_BODY_DERIVED_PURSUIT_FULL_SPEED_PIXELS_PER_SECOND = 1800.0
# Velocity feed-forward must remain available at zero positional error or a
# constant-speed target necessarily falls behind before authority returns.
# Withdraw the reserve only once either fresh or command-aware residual is
# materially ahead of the moving target, with a smooth band for detector noise.
_BODY_DERIVED_PURSUIT_ZERO_OPPOSED_ERROR_PIXELS = -8.0
_BODY_DERIVED_PURSUIT_FULL_OPPOSED_ERROR_PIXELS = 0.0
_BODY_DERIVED_PURSUIT_ZERO_FRESH_SPEED_RATIO = 0.35
_BODY_DERIVED_PURSUIT_FULL_FRESH_SPEED_RATIO = 0.70
# This is an integral disturbance estimate, not another proportional gain. It
# charges only while both causal residuals trail coherent mapped-body motion,
# then retains the learned velocity fraction through exact lock. The newest
# command-compensated frame clears it on a stop, reversal, or manual approach.
# Body-only evidence retains the previously validated conservative learning
# band.  The independent phase-flow path has a separate earlier ramp below:
# unlike mapped-body fallback it supplies a genuinely independent motion
# measurement, and the live trace verifies its velocity against 45 ms
# command-compensated head/body windows.
_BODY_DERIVED_ADAPTIVE_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND = 900.0
_BODY_DERIVED_ADAPTIVE_PURSUIT_FULL_SPEED_PIXELS_PER_SECOND = 1200.0
_BODY_DERIVED_ADAPTIVE_PURSUIT_RISE_TIME_CONSTANT_SECONDS = 0.080
_VERIFIED_FLOW_MINIMUM_EXPECTED_DISPLACEMENT_FRACTION = 0.20
_VERIFIED_FLOW_MINIMUM_ALONG_DISPLACEMENT_PIXELS = 0.50
_COMMON_PAIRED_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND = 250.0
_COMMON_PAIRED_PURSUIT_FULL_SPEED_PIXELS_PER_SECOND = 500.0
_COMMON_PAIRED_PURSUIT_RISE_TIME_CONSTANT_SECONDS = 0.030
# One zero quantized step is absence of evidence, not proof that a coherent
# moving target stopped. Retain an already-earned rate across at most one such
# accepted sample and at most 25 ms; a second low-motion sample or a material
# innovation clears it.
_COMMON_PAIRED_PURSUIT_AMBIGUOUS_FRESH_MAX_SECONDS = 0.025
# Closed-loop residual pursuit is deliberately a position-authorized channel,
# not another derivative-confidence multiplier.  Three causal measured samples
# must agree for at least the 16 ms evidence horizon before it can charge.  The
# extra sample rejects the observed low-SNR 16 Hz two-sample jitter lobe.
# The retained fraction then rises smoothly; detector jitter inside the
# configured feedback deadzone cannot create it.
_RESIDUAL_PURSUIT_MINIMUM_ALIGNED_SAMPLES = 3
_RESIDUAL_PURSUIT_MINIMUM_ALIGNMENT_SECONDS = 0.016
_RESIDUAL_PURSUIT_RISE_TIME_CONSTANT_SECONDS = 0.040
_RESIDUAL_PURSUIT_MINIMUM_AXIS_SPEED_PIXELS_PER_SECOND = 250.0
_RESIDUAL_PURSUIT_MINIMUM_POSITION_AGREEMENT = 0.50
_RESIDUAL_PURSUIT_POSITION_SIGMA_FRACTION = 0.50
# Preserve an independently earned pursuit fraction across the ordinary
# phase-flow on/off handoff, but do not let body fallback maintain that larger
# grant indefinitely.  Fifty milliseconds covers the trace's normal source
# flaps while matching the already required coherent-direction horizon.
_COMMON_PAIRED_PURSUIT_HANDOFF_SECONDS = 0.050
# A correlated lookahead may carry an already-proven independent motion grant
# only across the single newer capture endpoint admitted by the live runtime.
# Keep the numeric boundary independent of caller cadence so a malformed batch
# cannot turn this into an open-ended prediction lease.
CORRELATED_LOOKAHEAD_MAX_LEAD_SECONDS = 0.025
_BODY_DERIVED_ADAPTIVE_PURSUIT_MAXIMUM_VECTOR_SPEED_PIXELS_PER_SECOND = 2200.0
_BODY_DERIVED_ADAPTIVE_MAXIMUM_POSITION_RECONCILIATION_PIXELS = 24.0
# Extra lead is opt-in bounded phase compensation on the closed-loop position
# path.  Its separate duration and ratio limits keep it subordinate to the
# feedback response time even when explicit velocity feed-forward is capped.
_MAXIMUM_ADDITIONAL_BODY_DERIVED_PROJECTION_SECONDS = 0.050
_MAXIMUM_BODY_PROJECTION_TO_POSITION_TIME_CONSTANT_RATIO = 1.25
_MAXIMUM_OBSERVER_DIAGNOSTIC = 1_000_000.0


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _positive(value: object, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _timestamp(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer monotonic timestamp")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


@dataclass(frozen=True, slots=True)
class CalibratedPlant:
    """Two independent pixels-per-count gains and total visible delay."""

    gain_x_pixels_per_count: float
    gain_y_pixels_per_count: float
    delay_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gain_x_pixels_per_count",
            _positive(self.gain_x_pixels_per_count, "X plant gain"),
        )
        object.__setattr__(
            self,
            "gain_y_pixels_per_count",
            _positive(self.gain_y_pixels_per_count, "Y plant gain"),
        )
        delay = _finite(self.delay_seconds, "plant delay")
        if not 0.0 <= delay <= 0.25:
            raise ValueError("plant delay must be between zero and 0.25 seconds")
        object.__setattr__(self, "delay_seconds", delay)


@dataclass(frozen=True, slots=True)
class CalibratedControlConfig:
    """Bounded tuning for :class:`MakcuCalibratedController`.

    Rate limits remain independently configurable for explicit hardware or
    user constraints.  Calibrated control deliberately gives both axes the
    full legacy 320-count, 60 Hz envelope by default: the measured per-axis
    gains already describe unequal plant response, so an implicit vertical
    multiplier would distort that calibration.
    """

    position_time_constant_seconds: float = 0.060
    velocity_filter_time_constant_seconds: float = 0.018
    maximum_target_speed_pixels_per_second: float = 3000.0
    maximum_target_acceleration_pixels_per_second_squared: float = 90_000.0
    maximum_rate_x_counts_per_second: float = 19_200.0
    maximum_rate_y_counts_per_second: float = 19_200.0
    # Optional closed-loop feedback limits inside the complete output envelope.
    # Zero resolves to the corresponding maximum rate (historical behavior).
    # Automatic measured-plant control uses this boundary to keep raw position
    # stair-steps at the proven legacy rate while allowing only qualified
    # velocity authority to use the remaining measured pursuit headroom.
    maximum_feedback_rate_x_counts_per_second: float = 0.0
    maximum_feedback_rate_y_counts_per_second: float = 0.0
    # Optional large-error slew cap. When the radial projected error exceeds
    # ``acquisition_error_threshold_pixels`` (fresh acquisition or a target
    # jump), the emitted rate is limited to ``acquisition_rate_counts_per_second``
    # instead of the full maximum. This turns violent reacquisition whips into
    # controlled approaches while leaving near-target tracking authority intact.
    # Zero disables the acquisition cap entirely (historical behavior).
    acquisition_error_threshold_pixels: float = 0.0
    acquisition_rate_counts_per_second: float = 0.0
    stale_after_seconds: float = 0.040
    maximum_observation_interval_seconds: float = 0.040
    maximum_error_jump_pixels: float = 180.0
    feedback_deadzone_pixels: float = 0.50
    # Legacy/profiled control preserves the historical hard deadzone.  The
    # automatic direct-head path may instead subtract the deadzone from the
    # feedback error, making the transition from rest continuous instead of
    # jumping immediately to a full proportional command at the boundary.
    continuous_feedback_deadband: bool = False
    # Optional smooth onset for the continuous deadband.  Across this many
    # pixels immediately outside the deadzone, feedback rises with a smoothstep
    # curve before becoming the ordinary linear response.  This damps small
    # detector-coordinate wobble without weakening pursuit once the residual
    # is meaningfully outside the lock region.  Zero preserves the historical
    # linear continuous-deadband response.
    continuous_feedback_shoulder_pixels: float = 0.0
    wrong_way_guard_pixels: float = 2.0
    velocity_median_window: int = 3
    maximum_velocity_feedforward_fraction: float = 1.0
    require_motion_corroboration_for_feedforward: bool = False
    maximum_command_history: int = 4096
    maximum_body_derived_projection_fraction: float = 0.0
    maximum_body_derived_feedforward_fraction: float = 0.0
    # Optional fast-pursuit reserve above the ordinary body-derived cap. Zero
    # disables it. The numeric core independently requires high speed, fresh
    # motion agreement, and the existing persistent evidence. Unlike ordinary
    # feedback, the reserve remains available at zero residual so a constant-
    # speed target need not fall behind before feed-forward turns on again.
    maximum_body_derived_pursuit_feedforward_fraction: float = 0.0
    # Optional closed-loop remedy for persistent pursuit lag.  Unlike the
    # body/flow velocity grants above, this additive missing-authority reserve
    # can charge only from multiple current, paired, measured observations
    # whose causal position residuals remain outside the deadzone and aligned
    # with command-compensated target motion.  It retains smoothly through
    # exact zero error, but any ahead residual, stop/reversal, rejected
    # observation, loss, release, or pending physical mouse input clears it.
    # The complete feed-forward result remains bounded by the configured
    # body-derived pursuit ceiling.  Zero preserves historical behavior.
    maximum_residual_pursuit_feedforward_fraction: float = 0.0
    # Optional ceiling for an already-earned fast-pursuit reserve carried from
    # the independently corroborated root of one atomic capture-lookahead
    # batch into its newer position-only endpoint. Zero preserves the ordinary
    # body-derived ceiling. The endpoint may reduce or revoke this grant, but
    # cannot expose a reserve which a later independent root has not verified.
    maximum_correlated_lookahead_pursuit_feedforward_fraction: float = 0.0
    # A capture endpoint backed by a consecutive, fail-closed pixel-flow run is
    # stronger evidence than the ordinary one-hop position lookahead above.
    # This optional *carry* ceiling still cannot arm its ordinary pursuit path
    # on its own: the inferred root must already have independently
    # corroborated the same motion, and the endpoint's command-compensated
    # stop/reversal checks remain able to reduce or revoke the grant before
    # output.  Because that consecutive pixel evidence is stronger than mapped
    # body motion, this ceiling may exceed the generic body-pursuit ceiling,
    # while remaining bounded by the global feed-forward and hard 0.95 limits.
    # Separately configured residual pursuit may learn missing authority from
    # several numerically accepted verified endpoints.  Zero preserves the
    # ordinary correlated ceiling for callers without qualified pixel motion.
    maximum_verified_flow_pursuit_feedforward_fraction: float = 0.0
    # A single quantized/ambiguous inferred root may momentarily lose newest-
    # frame motion confidence even though its accepted observer velocity and
    # independent identity remain continuous.  This opt-in retains only the
    # prior source-age *position projection* for one such sample; it never
    # carries feed-forward, increases authority, survives a direction change,
    # or bypasses a material stop/reversal.
    retain_ambiguous_correlated_projection: bool = False
    # Optional per-axis large-error feedback schedule.  All three values at
    # zero preserve the historical fixed position time constant.  When
    # enabled, each axis independently transitions from the ordinary time
    # constant at ``start`` to the faster pursuit time constant at ``full``
    # using a smoothstep curve; near-lock feedback is therefore unchanged.
    pursuit_position_time_constant_seconds: float = 0.0
    pursuit_position_time_constant_start_pixels: float = 0.0
    pursuit_position_time_constant_full_pixels: float = 0.0
    # Optional automatic-direct-head safety boundary. The paired/raw motion
    # coordinate is allowed to suppress prediction whenever it disagrees with
    # the filtered aim coordinate, but it must not also erase a clearly
    # off-target closed-loop correction. When enabled, positional confidence
    # follows the existing pursuit smoothstep from channel agreement near lock
    # to full trust at the pursuit threshold. The default preserves historical
    # behavior for explicit profiles and other callers.
    preserve_pursuit_position_feedback: bool = False
    # Optional fixed extension of body-derived positional prediction.  It is
    # applied only under an active provenance grant and remains multiplied by
    # the existing per-axis body-motion projection confidence.
    additional_body_derived_projection_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "position_time_constant_seconds",
            "velocity_filter_time_constant_seconds",
            "maximum_target_speed_pixels_per_second",
            "maximum_target_acceleration_pixels_per_second_squared",
            "maximum_rate_x_counts_per_second",
            "maximum_rate_y_counts_per_second",
            "stale_after_seconds",
            "maximum_observation_interval_seconds",
            "maximum_error_jump_pixels",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in (
            "feedback_deadzone_pixels",
            "continuous_feedback_shoulder_pixels",
            "wrong_way_guard_pixels",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        for axis in ("x", "y"):
            name = f"maximum_feedback_rate_{axis}_counts_per_second"
            maximum_name = f"maximum_rate_{axis}_counts_per_second"
            value = _finite(getattr(self, name), name)
            maximum = float(getattr(self, maximum_name))
            if value < 0.0 or value > maximum:
                raise ValueError(
                    f"{name} must be zero or no greater than {maximum_name}"
                )
            object.__setattr__(self, name, maximum if value == 0.0 else value)
        threshold = _finite(
            self.acquisition_error_threshold_pixels,
            "acquisition_error_threshold_pixels",
        )
        slew = _finite(
            self.acquisition_rate_counts_per_second,
            "acquisition_rate_counts_per_second",
        )
        if threshold < 0.0 or slew < 0.0:
            raise ValueError("acquisition slew settings cannot be negative")
        if (threshold == 0.0) != (slew == 0.0):
            raise ValueError(
                "acquisition_error_threshold_pixels and "
                "acquisition_rate_counts_per_second must both be set or both be zero"
            )
        object.__setattr__(self, "acquisition_error_threshold_pixels", threshold)
        object.__setattr__(self, "acquisition_rate_counts_per_second", slew)
        feedforward_fraction = _finite(
            self.maximum_velocity_feedforward_fraction,
            "maximum_velocity_feedforward_fraction",
        )
        if not 0.0 <= feedforward_fraction <= 1.0:
            raise ValueError(
                "maximum_velocity_feedforward_fraction must be between zero and one"
            )
        object.__setattr__(
            self,
            "maximum_velocity_feedforward_fraction",
            feedforward_fraction,
        )
        if not isinstance(self.require_motion_corroboration_for_feedforward, bool):
            raise TypeError(
                "require_motion_corroboration_for_feedforward must be bool"
            )
        if not isinstance(self.continuous_feedback_deadband, bool):
            raise TypeError("continuous_feedback_deadband must be bool")
        if not isinstance(self.preserve_pursuit_position_feedback, bool):
            raise TypeError("preserve_pursuit_position_feedback must be bool")
        if not isinstance(self.retain_ambiguous_correlated_projection, bool):
            raise TypeError("retain_ambiguous_correlated_projection must be bool")
        body_derived_projection_fraction = _finite(
            self.maximum_body_derived_projection_fraction,
            "maximum_body_derived_projection_fraction",
        )
        if not (
            0.0
            <= body_derived_projection_fraction
            <= 1.0
        ):
            raise ValueError(
                "maximum_body_derived_projection_fraction must be between zero and one"
            )
        object.__setattr__(
            self,
            "maximum_body_derived_projection_fraction",
            body_derived_projection_fraction,
        )
        body_derived_feedforward_fraction = _finite(
            self.maximum_body_derived_feedforward_fraction,
            "maximum_body_derived_feedforward_fraction",
        )
        if not (
            0.0
            <= body_derived_feedforward_fraction
            <= _MAXIMUM_BODY_DERIVED_FEEDFORWARD_FRACTION
        ):
            raise ValueError(
                "maximum_body_derived_feedforward_fraction must be between zero and 0.50"
            )
        object.__setattr__(
            self,
            "maximum_body_derived_feedforward_fraction",
            body_derived_feedforward_fraction,
        )
        pursuit_feedforward_fraction = _finite(
            self.maximum_body_derived_pursuit_feedforward_fraction,
            "maximum_body_derived_pursuit_feedforward_fraction",
        )
        if pursuit_feedforward_fraction != 0.0 and not (
            body_derived_feedforward_fraction
            <= pursuit_feedforward_fraction
            <= _MAXIMUM_BODY_DERIVED_PURSUIT_FEEDFORWARD_FRACTION
        ):
            raise ValueError(
                "maximum_body_derived_pursuit_feedforward_fraction must be "
                "zero or between the ordinary body-derived feed-forward "
                "fraction and 0.95"
            )
        object.__setattr__(
            self,
            "maximum_body_derived_pursuit_feedforward_fraction",
            pursuit_feedforward_fraction,
        )
        residual_pursuit_feedforward_fraction = _finite(
            self.maximum_residual_pursuit_feedforward_fraction,
            "maximum_residual_pursuit_feedforward_fraction",
        )
        if residual_pursuit_feedforward_fraction != 0.0 and not (
            0.0
            < residual_pursuit_feedforward_fraction
            <= pursuit_feedforward_fraction
            and residual_pursuit_feedforward_fraction
            <= feedforward_fraction
        ):
            raise ValueError(
                "maximum_residual_pursuit_feedforward_fraction must be zero "
                "or no greater than both the fast-pursuit and global "
                "feed-forward fractions"
            )
        object.__setattr__(
            self,
            "maximum_residual_pursuit_feedforward_fraction",
            residual_pursuit_feedforward_fraction,
        )
        correlated_pursuit_feedforward_fraction = _finite(
            self.maximum_correlated_lookahead_pursuit_feedforward_fraction,
            "maximum_correlated_lookahead_pursuit_feedforward_fraction",
        )
        if correlated_pursuit_feedforward_fraction != 0.0 and not (
            body_derived_feedforward_fraction
            <= correlated_pursuit_feedforward_fraction
            <= pursuit_feedforward_fraction
        ):
            raise ValueError(
                "maximum_correlated_lookahead_pursuit_feedforward_fraction "
                "must be zero or between the ordinary and fast body-derived "
                "feed-forward fractions"
            )
        object.__setattr__(
            self,
            "maximum_correlated_lookahead_pursuit_feedforward_fraction",
            correlated_pursuit_feedforward_fraction,
        )
        verified_flow_pursuit_feedforward_fraction = _finite(
            self.maximum_verified_flow_pursuit_feedforward_fraction,
            "maximum_verified_flow_pursuit_feedforward_fraction",
        )
        if verified_flow_pursuit_feedforward_fraction != 0.0 and not (
            correlated_pursuit_feedforward_fraction
            <= verified_flow_pursuit_feedforward_fraction
            <= _MAXIMUM_BODY_DERIVED_PURSUIT_FEEDFORWARD_FRACTION
            and verified_flow_pursuit_feedforward_fraction
            <= feedforward_fraction
        ):
            raise ValueError(
                "maximum_verified_flow_pursuit_feedforward_fraction must be "
                "zero or between the correlated-lookahead fraction and both "
                "the global feed-forward and hard 0.95 limits"
            )
        object.__setattr__(
            self,
            "maximum_verified_flow_pursuit_feedforward_fraction",
            verified_flow_pursuit_feedforward_fraction,
        )
        pursuit_time_constant = _finite(
            self.pursuit_position_time_constant_seconds,
            "pursuit_position_time_constant_seconds",
        )
        pursuit_start = _finite(
            self.pursuit_position_time_constant_start_pixels,
            "pursuit_position_time_constant_start_pixels",
        )
        pursuit_full = _finite(
            self.pursuit_position_time_constant_full_pixels,
            "pursuit_position_time_constant_full_pixels",
        )
        if pursuit_time_constant == pursuit_start == pursuit_full == 0.0:
            pass
        elif pursuit_time_constant <= 0.0:
            raise ValueError(
                "pursuit_position_time_constant_seconds must be positive when "
                "the pursuit schedule is enabled"
            )
        elif pursuit_start < 0.0 or pursuit_full <= pursuit_start:
            raise ValueError(
                "pursuit position thresholds require zero-or-greater start and "
                "full greater than start"
            )
        elif pursuit_time_constant > self.position_time_constant_seconds:
            raise ValueError(
                "pursuit_position_time_constant_seconds cannot exceed "
                "position_time_constant_seconds"
            )
        object.__setattr__(
            self,
            "pursuit_position_time_constant_seconds",
            pursuit_time_constant,
        )
        object.__setattr__(
            self,
            "pursuit_position_time_constant_start_pixels",
            pursuit_start,
        )
        object.__setattr__(
            self,
            "pursuit_position_time_constant_full_pixels",
            pursuit_full,
        )
        if self.preserve_pursuit_position_feedback and pursuit_full == 0.0:
            raise ValueError(
                "preserve_pursuit_position_feedback requires the pursuit "
                "position schedule"
            )
        additional_body_projection = _finite(
            self.additional_body_derived_projection_seconds,
            "additional_body_derived_projection_seconds",
        )
        if not (
            0.0
            <= additional_body_projection
            <= _MAXIMUM_ADDITIONAL_BODY_DERIVED_PROJECTION_SECONDS
        ):
            raise ValueError(
                "additional_body_derived_projection_seconds must be between "
                "zero and 0.05 seconds"
            )
        minimum_position_time_constant = (
            pursuit_time_constant
            if pursuit_time_constant > 0.0
            else self.position_time_constant_seconds
        )
        if (
            additional_body_projection
            / minimum_position_time_constant
            > _MAXIMUM_BODY_PROJECTION_TO_POSITION_TIME_CONSTANT_RATIO
        ):
            raise ValueError(
                "additional_body_derived_projection_seconds cannot exceed "
                "1.25 times the minimum active position time constant"
            )
        object.__setattr__(
            self,
            "additional_body_derived_projection_seconds",
            additional_body_projection,
        )
        if (
            isinstance(self.velocity_median_window, bool)
            or not isinstance(self.velocity_median_window, int)
            or self.velocity_median_window < 1
            or self.velocity_median_window > 9
            or self.velocity_median_window % 2 == 0
        ):
            raise ValueError("velocity_median_window must be an odd integer from 1 to 9")
        if (
            isinstance(self.maximum_command_history, bool)
            or not isinstance(self.maximum_command_history, int)
            or self.maximum_command_history < 16
        ):
            raise ValueError("maximum_command_history must be an integer of at least 16")
        if self.maximum_observation_interval_seconds > self.stale_after_seconds:
            raise ValueError(
                "maximum_observation_interval_seconds cannot exceed stale_after_seconds"
            )


@dataclass(frozen=True, slots=True)
class ScreenErrorObservation:
    """One observed X/Y error in the calibration profile's pixel space.

    ``error_*`` is the historical position-control observation and may be a
    tracker-smoothed point.  Supplying the optional ``velocity_error_*`` pair
    selects the automatic raw-point observer: that accepted detector point is
    the single measurement used to estimate both image position and target
    velocity.  Exact landed commands are its known control input.  An optional
    ``corroboration_error_*`` pair is an independent object's point from the
    same source frame (for example, the primary detector's player-box center).
    Its separate observer can authorize coherent translation feed-forward, but
    can never move or replace the raw aim coordinate.  Omitting the optional
    fields preserves the historical single-channel controller exactly.

    ``body_derived_motion_permitted`` is a narrow, per-observation provenance
    assertion for an aim point translated from a measured primary/body box.
    It is not independent corroboration and therefore cannot accompany the
    corroboration pair.  When both it and the separately configured bounded
    fractions are present, automatic corroboration-required control may use
    the separately bounded source-age projection and explicit feed-forward.
    Neither grant changes independent corroboration confidence.  The caller
    must revoke it on a primary prediction or when its immutable direct-head
    identity lease expires.  A permitted sample must carry that immutable
    lease's exclusive ``body_derived_motion_deadline_ns``; permission is
    withdrawn when the control clock reaches the deadline even if the mapped
    position observation itself remains fresh.

    ``identity_deadline_ns`` is a separate optional absolute authority lease.
    At that exclusive deadline all tracking/output is invalidated, preventing
    a fresh-looking mapped position from outliving the identity anchor which
    authorized it.  Callers without identity-bound control omit it unchanged.
    """

    timestamp_ns: int
    error_x_pixels: float
    error_y_pixels: float
    velocity_error_x_pixels: float | None = None
    velocity_error_y_pixels: float | None = None
    corroboration_error_x_pixels: float | None = None
    corroboration_error_y_pixels: float | None = None
    body_derived_motion_permitted: bool = False
    body_derived_motion_deadline_ns: int | None = None
    identity_deadline_ns: int | None = None

    def __post_init__(self) -> None:
        _timestamp(self.timestamp_ns, "observation timestamp")
        object.__setattr__(
            self,
            "error_x_pixels",
            _finite(self.error_x_pixels, "observation X error"),
        )
        object.__setattr__(
            self,
            "error_y_pixels",
            _finite(self.error_y_pixels, "observation Y error"),
        )
        paired = (
            self.velocity_error_x_pixels is not None,
            self.velocity_error_y_pixels is not None,
        )
        if paired[0] != paired[1]:
            raise ValueError(
                "velocity X/Y errors must either both be supplied or both be omitted"
            )
        if paired[0]:
            object.__setattr__(
                self,
                "velocity_error_x_pixels",
                _finite(
                    self.velocity_error_x_pixels,
                    "observation velocity X error",
                ),
            )
            object.__setattr__(
                self,
                "velocity_error_y_pixels",
                _finite(
                    self.velocity_error_y_pixels,
                    "observation velocity Y error",
                ),
            )
        corroboration = (
            self.corroboration_error_x_pixels is not None,
            self.corroboration_error_y_pixels is not None,
        )
        if corroboration[0] != corroboration[1]:
            raise ValueError(
                "corroboration X/Y errors must either both be supplied or both be omitted"
            )
        if corroboration[0] and not paired[0]:
            raise ValueError(
                "corroboration errors require the paired velocity error channel"
            )
        if corroboration[0]:
            object.__setattr__(
                self,
                "corroboration_error_x_pixels",
                _finite(
                    self.corroboration_error_x_pixels,
                    "observation corroboration X error",
                ),
            )
            object.__setattr__(
                self,
                "corroboration_error_y_pixels",
                _finite(
                    self.corroboration_error_y_pixels,
                    "observation corroboration Y error",
                ),
            )
        if not isinstance(self.body_derived_motion_permitted, bool):
            raise TypeError("body_derived_motion_permitted must be bool")
        if self.body_derived_motion_permitted and not paired[0]:
            raise ValueError(
                "body-derived motion permission requires the paired velocity error channel"
            )
        if self.body_derived_motion_permitted and corroboration[0]:
            raise ValueError(
                "body-derived motion permission cannot accompany independent corroboration"
            )
        deadline = self.body_derived_motion_deadline_ns
        if self.body_derived_motion_permitted:
            if deadline is None:
                raise ValueError(
                    "body-derived motion permission requires an immutable deadline"
                )
            parsed_deadline = _timestamp(
                deadline,
                "body-derived motion deadline",
            )
            if parsed_deadline <= self.timestamp_ns:
                raise ValueError(
                    "body-derived motion deadline must be after the observation timestamp"
                )
            object.__setattr__(
                self,
                "body_derived_motion_deadline_ns",
                parsed_deadline,
            )
        elif deadline is not None:
            raise ValueError(
                "body-derived motion deadline requires motion permission"
            )
        identity_deadline = self.identity_deadline_ns
        if identity_deadline is not None:
            parsed_identity_deadline = _timestamp(
                identity_deadline,
                "identity deadline",
            )
            if parsed_identity_deadline <= self.timestamp_ns:
                raise ValueError(
                    "identity deadline must be after the observation timestamp"
                )
            object.__setattr__(
                self,
                "identity_deadline_ns",
                parsed_identity_deadline,
            )


@dataclass(frozen=True, slots=True)
class CorrelatedLookaheadObservation:
    """One measured motion root followed by one newer position endpoint.

    ``primary`` is an exact inferred-frame head observation with an independent
    body-center corroboration point. ``lookahead`` is the same identity moved
    into one newer captured frame by bounded pixel flow. The latter is
    deliberately position-only by default: it can retain authority earned by
    ``primary`` for this batch, but can never create or increase that authority
    itself. ``verified_flow_motion`` marks the narrower live contract where a
    consecutive, fail-closed pixel-flow run observed this endpoint. Even then,
    the independently corroborated root must already have armed the ordinary
    pursuit carry; the flag only selects a larger preconfigured carry ceiling
    and cannot create identity, readiness, or velocity by itself.  When
    residual pursuit is separately enabled, several endpoints which also pass
    the numeric command-compensated motion check may instead learn a bounded
    closed-loop missing-authority reserve.

    The explicit runtime and tracker generations prevent an equal-timestamp
    body confirmation from being attached to a phase endpoint from another
    target or activation epoch.
    """

    primary: ScreenErrorObservation
    lookahead: ScreenErrorObservation
    runtime_identity_generation: int
    track_generation: int
    verified_flow_motion: bool = False

    def __post_init__(self) -> None:
        for name in ("runtime_identity_generation", "track_generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.verified_flow_motion, bool):
            raise TypeError("verified_flow_motion must be bool")
        primary = self.primary
        lookahead = self.lookahead
        if not isinstance(primary, ScreenErrorObservation):
            raise TypeError("primary must be a ScreenErrorObservation")
        if not isinstance(lookahead, ScreenErrorObservation):
            raise TypeError("lookahead must be a ScreenErrorObservation")
        if primary.velocity_error_x_pixels is None:
            raise ValueError("correlated primary requires a paired velocity point")
        if primary.corroboration_error_x_pixels is None:
            raise ValueError("correlated primary requires independent corroboration")
        if primary.body_derived_motion_permitted:
            raise ValueError("correlated primary cannot use body-derived permission")
        if lookahead.velocity_error_x_pixels is None:
            raise ValueError("correlated lookahead requires a paired velocity point")
        if lookahead.corroboration_error_x_pixels is not None:
            raise ValueError("correlated lookahead must remain position-only")
        if lookahead.body_derived_motion_permitted:
            raise ValueError("correlated lookahead cannot use body-derived permission")
        lead_ns = lookahead.timestamp_ns - primary.timestamp_ns
        maximum_lead_ns = round(
            CORRELATED_LOOKAHEAD_MAX_LEAD_SECONDS * _NS_PER_SECOND
        )
        if lead_ns <= 0 or lead_ns > maximum_lead_ns:
            raise ValueError(
                "correlated lookahead must be newer by at most "
                f"{CORRELATED_LOOKAHEAD_MAX_LEAD_SECONDS * 1000.0:g} ms"
            )
        primary_deadline_ns = primary.identity_deadline_ns
        if primary_deadline_ns is None:
            raise ValueError("correlated observations require an identity deadline")
        if lookahead.identity_deadline_ns != primary_deadline_ns:
            raise ValueError("correlated observations must share one identity deadline")
        if lookahead.timestamp_ns >= primary_deadline_ns:
            raise ValueError("correlated lookahead must precede its identity deadline")


@dataclass(frozen=True, slots=True)
class _CorrelatedLookaheadAuthority:
    """Authority ceiling earned at the measured root of one atomic batch."""

    authorized: bool
    projection_x: float
    projection_y: float
    ordinary_x: float
    ordinary_y: float
    total_x: float
    total_y: float
    velocity_direction_x: int
    velocity_direction_y: int
    ambiguous_projection_retained_x: bool = False
    ambiguous_projection_retained_y: bool = False


@dataclass(frozen=True, slots=True)
class _ResidualPursuitAxisState:
    """One axis of closed-loop, residual-trained missing authority."""

    authority: float = 0.0
    direction: int = 0
    aligned_samples: int = 0
    aligned_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class EmittedMouseCommand:
    """One relative command successfully written to the physical MAKCU."""

    timestamp_ns: int
    delta_x_counts: int
    delta_y_counts: int

    def __post_init__(self) -> None:
        _timestamp(self.timestamp_ns, "emitted command timestamp")
        for name in ("delta_x_counts", "delta_y_counts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.delta_x_counts == 0 and self.delta_y_counts == 0:
            raise ValueError("an emitted command cannot be zero on both axes")


@dataclass(frozen=True, slots=True)
class _PhysicalMouseInput:
    """One raw physical-mouse report observed by the MAKCU passthrough.

    This is intentionally distinct from :class:`EmittedMouseCommand`.  Both
    inputs move the same calibrated screen plant, but only the latter was
    generated by this controller.  Keeping two ledgers prevents diagnostics
    and write accounting from ever claiming that a user's hand movement was
    an injected command.
    """

    timestamp_ns: int
    delta_x_counts: int
    delta_y_counts: int

    def __post_init__(self) -> None:
        _timestamp(self.timestamp_ns, "physical mouse input timestamp")
        for name in ("delta_x_counts", "delta_y_counts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not -32_768 <= value <= 32_767:
                raise ValueError(f"{name} must fit a signed 16-bit report")
        if self.delta_x_counts == 0 and self.delta_y_counts == 0:
            raise ValueError("a physical mouse input cannot be zero on both axes")


@dataclass(frozen=True, slots=True)
class CalibratedControlOutput:
    """One bounded count-rate decision and its non-sensitive diagnostics."""

    timestamp_ns: int
    rate_x_counts_per_second: float
    rate_y_counts_per_second: float
    target_velocity_x_pixels_per_second: float
    target_velocity_y_pixels_per_second: float
    projected_error_x_pixels: float
    projected_error_y_pixels: float
    valid: bool
    saturated_x: bool = False
    saturated_y: bool = False
    reset_reason: str | None = None
    observer_position_sigma_x_pixels: float = 0.0
    observer_position_sigma_y_pixels: float = 0.0
    observer_velocity_sigma_x_pixels_per_second: float = 0.0
    observer_velocity_sigma_y_pixels_per_second: float = 0.0
    velocity_feedforward_confidence_x: float = 0.0
    velocity_feedforward_confidence_y: float = 0.0
    # Expose the exact boundary between the trusted filtered position and its
    # auxiliary motion coordinate. These diagnostics make a future live trace
    # distinguish a real deadzone/hold from a prediction-agreement revoke.
    position_channel_agreement: float = 0.0
    position_feedback_confidence_x: float = 0.0
    position_feedback_confidence_y: float = 0.0
    position_feedback_held_x: bool = False
    position_feedback_held_y: bool = False
    innovation_mahalanobis_squared: float = 0.0
    innovation_rejected: bool = False
    motion_corroboration_confidence: float = 0.0
    body_derived_motion_confidence_x: float = 0.0
    body_derived_motion_confidence_y: float = 0.0
    correlated_lookahead_active: bool = False
    lookahead_retained_authority_x: float = 0.0
    lookahead_retained_authority_y: float = 0.0
    ambiguous_lookahead_projection_retained_x: bool = False
    ambiguous_lookahead_projection_retained_y: bool = False
    pursuit_reserve_rate_x_counts_per_second: float = 0.0
    pursuit_reserve_rate_y_counts_per_second: float = 0.0
    residual_pursuit_authority_x: float = 0.0
    residual_pursuit_authority_y: float = 0.0
    # A newest-frame physical stop/reversal masks all velocity authority on
    # this axis.  The output adapter uses the edge to bypass its ordinary EMA,
    # so a safe numeric revoke also becomes an immediate emitted revoke.
    material_motion_revoked_x: bool = False
    material_motion_revoked_y: bool = False
    predictive_authority_revoked_x: bool = False
    predictive_authority_revoked_y: bool = False
    # Raw passthrough motion which has not yet reached the newest captured
    # observation.  Output yields that axis while the observer anticipates the
    # known physical displacement.
    physical_input_pending_x: bool = False
    physical_input_pending_y: bool = False


class MakcuCalibratedController:
    """Estimate target motion and control a calibrated delayed mouse plant.

    ``step`` is intended to run at the output worker cadence.  ``observation``
    is supplied only when a new real detector measurement exists.  Set
    ``target_lost`` only when the detector/tracker has explicitly revoked the
    physical target.  A synthetic tracker prediction or an omitted callback
    supplies neither an observation nor a loss; it can bridge only until
    ``stale_after_seconds`` expires.

    Call :meth:`record_emitted` once, immediately after each successful
    physical write.  Keeping successful writes separate from requested rates
    prevents fractional, clamped-away, or failed commands from being
    misclassified as plant motion.  ``emitted_commands`` on ``step`` remains a
    backwards-compatible batched form for callers which already stage writes.
    """

    def __init__(
        self,
        plant: CalibratedPlant,
        config: CalibratedControlConfig | None = None,
    ) -> None:
        if not isinstance(plant, CalibratedPlant):
            raise TypeError("plant must be a CalibratedPlant")
        self.plant = plant
        self.config = config or CalibratedControlConfig()
        self._delay_ns = round(plant.delay_seconds * _NS_PER_SECOND)
        self._commands: deque[EmittedMouseCommand] = deque()
        self._last_command_ns = -1
        self._physical_inputs: deque[_PhysicalMouseInput] = deque()
        self._last_physical_input_ns = -1
        self._last_step_ns = -1
        self._last_observation: ScreenErrorObservation | None = None
        self._identity_deadline_ns: int | None = None
        self._ready = False
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        window = self.config.velocity_median_window
        self._raw_velocity_x: deque[float] = deque(maxlen=window)
        self._raw_velocity_y: deque[float] = deque(maxlen=window)
        # One block-diagonal four-state observer: [position X, position Y,
        # velocity X, velocity Y]. Each independent image axis stores the
        # symmetric 2x2 covariance as (position variance, cross covariance,
        # velocity variance). The observer is used only when the paired raw
        # point is present; legacy/no-raw callers retain the original path.
        self._paired_position_x = 0.0
        self._paired_position_y = 0.0
        self._paired_covariance_x = (0.0, 0.0, 0.0)
        self._paired_covariance_y = (0.0, 0.0, 0.0)
        self._paired_feedback_hold_x = False
        self._paired_feedback_hold_y = False
        self._paired_measurement_agreement_x = 1.0
        self._paired_measurement_agreement_y = 1.0
        self._paired_rejection_count = 0
        self._last_innovation_rejected = False
        self._last_innovation_mahalanobis_squared = 0.0
        # Optional independent motion evidence is intentionally a second
        # observer, not another measurement fused into the direct aim point.
        # The primary/body detector may corroborate translation, but its box
        # geometry can never pull the requested coordinate away from the head.
        self._corroboration_position_x = 0.0
        self._corroboration_position_y = 0.0
        self._corroboration_velocity_x = 0.0
        self._corroboration_velocity_y = 0.0
        self._corroboration_covariance_x = (0.0, 0.0, 0.0)
        self._corroboration_covariance_y = (0.0, 0.0, 0.0)
        # The independent body/phase reference is intentionally sparse.  Its
        # physical measurements commonly arrive on alternating primary frames,
        # so keep the timestamp of the last *real* corroboration separately
        # from the primary observer's timestamp.  Predict-only primary frames
        # advance this observer without granting output authority.
        self._last_corroboration_measurement_ns: int | None = None
        self._motion_corroboration_confidence = 0.0
        # Large-error pursuit authority is provenance, not velocity
        # confidence. Keep the two mutually exclusive evidence sources
        # separate so the narrow body revoke does not discard a valid
        # independent corroboration grant. Ordinary output ticks may retain a
        # grant between detector samples; only accepted physical evidence may
        # create one.
        self._independent_pursuit_authorized = False
        self._body_derived_pursuit_authorized = False
        self._last_correlated_identity: tuple[int, int] | None = None
        self._last_observation_was_correlated_lookahead = False
        self._correlated_lookahead_authority_state: (
            _CorrelatedLookaheadAuthority | None
        ) = None
        self._correlated_lookahead_authority_deadline_ns: int | None = None
        self._corroboration_direction_anchor_x = 0.0
        self._corroboration_direction_anchor_y = 0.0
        self._corroboration_direction_persistence_seconds = 0.0
        self._corroboration_reseed_required = False
        # Per-sample permission for a body-mapped coordinate is deliberately
        # separate from the independent corroboration observer above.  It is
        # cleared before every new measurement is evaluated, so a missing or
        # rejected authorization can never inherit predictive motion.
        self._body_derived_motion_permitted = False
        self._body_derived_motion_deadline_ns: int | None = None
        self._body_derived_motion_confidence_x = 0.0
        self._body_derived_motion_confidence_y = 0.0
        self._body_derived_fresh_motion_confidence_x = 0.0
        self._body_derived_fresh_motion_confidence_y = 0.0
        # Newest command-compensated primary motion is provenance-neutral.  It
        # lets one already-qualified pursuit state survive a brief switch
        # between pixel-flow and mapped-body observations, while an immediate
        # stop, reversal, or manual approach still clears that state.
        self._paired_fresh_motion_confidence_x = 0.0
        self._paired_fresh_motion_confidence_y = 0.0
        self._paired_material_stop_or_reversal_x = False
        self._paired_material_stop_or_reversal_y = False
        self._adaptive_pursuit_low_fresh_samples_x = 0
        self._adaptive_pursuit_low_fresh_samples_y = 0
        self._common_pursuit_direction_anchor_x = 0.0
        self._common_pursuit_direction_anchor_y = 0.0
        self._common_pursuit_direction_persistence_seconds = 0.0
        self._body_derived_adaptive_pursuit_confidence_x = 0.0
        self._body_derived_adaptive_pursuit_confidence_y = 0.0
        self._residual_pursuit_x = _ResidualPursuitAxisState()
        self._residual_pursuit_y = _ResidualPursuitAxisState()
        # A verified capture endpoint may earn a narrowly leased positional
        # projection after the residual learner's existing multi-sample proof.
        # Keep this permission separate per axis from both ordinary
        # independent/body provenance and the residual velocity fraction.
        self._verified_residual_projection_x = False
        self._verified_residual_projection_y = False
        self._common_pursuit_handoff_deadline_ns: int | None = None
        self._body_derived_direction_anchor_x = 0.0
        self._body_derived_direction_anchor_y = 0.0
        self._body_derived_direction_persistence_seconds = 0.0
        self._body_derived_axis_direction_anchor_x = 0.0
        self._body_derived_axis_direction_anchor_y = 0.0
        self._body_derived_direction_persistence_x_seconds = 0.0
        self._body_derived_direction_persistence_y_seconds = 0.0

    @property
    def ready(self) -> bool:
        """Whether two continuous fresh observations currently authorize output."""

        return self._ready

    def reset(self, *, clear_command_history: bool = True) -> None:
        """Erase all learned motion; optionally erase recorded physical facts."""

        self._reset_tracking()
        self._last_step_ns = -1
        if clear_command_history:
            self._commands.clear()
            self._last_command_ns = -1
            self._physical_inputs.clear()
            self._last_physical_input_ns = -1

    def revoke_motion_corroboration(self) -> None:
        """Withdraw feed-forward permission without discarding head position.

        This is the narrow bridge operation for an explicit no-head result or
        a current primary-player prediction.  The accepted direct-head state,
        its freshness lease, landed-command ledger, and readiness remain
        untouched.  A later corroboration channel must seed its independent
        observer once before it can begin earning motion confidence again.

        Like :meth:`step` and :meth:`record_emitted`, this numeric method is not
        internally synchronized; the runtime owner must serialize calls.
        """

        self._corroboration_position_x = 0.0
        self._corroboration_position_y = 0.0
        self._corroboration_velocity_x = 0.0
        self._corroboration_velocity_y = 0.0
        self._corroboration_covariance_x = (0.0, 0.0, 0.0)
        self._corroboration_covariance_y = (0.0, 0.0, 0.0)
        self._last_corroboration_measurement_ns = None
        self._motion_corroboration_confidence = 0.0
        self._independent_pursuit_authorized = False
        self._corroboration_direction_anchor_x = 0.0
        self._corroboration_direction_anchor_y = 0.0
        self._corroboration_direction_persistence_seconds = 0.0
        self._corroboration_reseed_required = True
        self._last_correlated_identity = None
        self._last_observation_was_correlated_lookahead = False
        self._correlated_lookahead_authority_state = None
        self._correlated_lookahead_authority_deadline_ns = None
        # This operation is the existing runtime's broad predictive-motion
        # revoke on no-head/current-primary-prediction events.  The evidence
        # models remain separate, but no body-derived lease may survive it.
        self._clear_body_derived_motion()

    def revoke_body_derived_motion(self) -> None:
        """Immediately withdraw body-derived predictive-motion permission.

        Call this when the current primary/body geometry is predicted rather
        than measured, or when the direct-head identity lease which authorized
        its mapping expires.  Position freshness remains untouched, allowing
        ordinary bounded static feedback without retaining velocity lead.
        This narrower operation leaves the independent corroboration observer
        untouched.  :meth:`revoke_motion_corroboration` remains the broad
        runtime revoke and clears this permission as well.
        """

        self._clear_body_derived_motion()

    def _clear_body_derived_motion(self, *, clear_adaptive: bool = True) -> None:
        if not isinstance(clear_adaptive, bool):
            raise TypeError("clear_adaptive must be bool")
        self._body_derived_pursuit_authorized = False
        self._body_derived_motion_permitted = False
        self._body_derived_motion_deadline_ns = None
        self._body_derived_motion_confidence_x = 0.0
        self._body_derived_motion_confidence_y = 0.0
        self._body_derived_fresh_motion_confidence_x = 0.0
        self._body_derived_fresh_motion_confidence_y = 0.0
        if clear_adaptive:
            self._common_pursuit_direction_anchor_x = 0.0
            self._common_pursuit_direction_anchor_y = 0.0
            self._common_pursuit_direction_persistence_seconds = 0.0
            self._body_derived_adaptive_pursuit_confidence_x = 0.0
            self._body_derived_adaptive_pursuit_confidence_y = 0.0
            self._clear_residual_pursuit()
            self._common_pursuit_handoff_deadline_ns = None
            self._adaptive_pursuit_low_fresh_samples_x = 0
            self._adaptive_pursuit_low_fresh_samples_y = 0
            self._paired_material_stop_or_reversal_x = False
            self._paired_material_stop_or_reversal_y = False
        self._body_derived_direction_anchor_x = 0.0
        self._body_derived_direction_anchor_y = 0.0
        self._body_derived_direction_persistence_seconds = 0.0
        self._body_derived_axis_direction_anchor_x = 0.0
        self._body_derived_axis_direction_anchor_y = 0.0
        self._body_derived_direction_persistence_x_seconds = 0.0
        self._body_derived_direction_persistence_y_seconds = 0.0

    def record_emitted(self, command: EmittedMouseCommand) -> None:
        """Record one successful physical write exactly once.

        The command timestamp must use the control tick's monotonic timestamp,
        and this method must be called only after that tick's write succeeds.
        Strictly increasing timestamps make a duplicate handoff fail closed
        instead of counting one physical movement twice.  A failed or zero
        write is represented by not calling this method at all.
        """

        if not isinstance(command, EmittedMouseCommand):
            self._reset_tracking()
            raise TypeError("command must be an EmittedMouseCommand")
        if self._last_step_ns < 0:
            self._reset_tracking()
            raise RuntimeError("cannot record an emitted command before a control step")
        error = self._ingest_commands((command,), self._last_step_ns)
        if error is not None:
            self._reset_tracking()
            if error == "command-history-overflow":
                raise RuntimeError(error)
            raise ValueError(error)

    def preflight_emitted(self, command: EmittedMouseCommand) -> None:
        """Validate predictable accounting failures before a physical write.

        The integration owns serialization around this call, the device write,
        and :meth:`record_emitted`.  Under that lock, a successful preflight
        guarantees that history order, timestamp, and capacity cannot reject
        the same immutable command only after it has reached the device.
        """

        if not isinstance(command, EmittedMouseCommand):
            raise TypeError("command must be an EmittedMouseCommand")
        if self._last_step_ns < 0:
            raise RuntimeError("cannot record an emitted command before a control step")
        error = self._validate_commands((command,), self._last_step_ns)
        if error == "command-history-overflow":
            raise RuntimeError(error)
        if error is not None:
            raise ValueError(error)

    def record_physical_input(
        self,
        timestamp_ns: int,
        delta_x_counts: int,
        delta_y_counts: int,
    ) -> None:
        """Record one raw passthrough-mouse report exactly once.

        The serial integration calls this as soon as a complete ``km.mouse``
        frame is decoded.  Physical input is a known plant input just like a
        successful injected command for observer compensation, but it remains
        in a separate ledger and may be recorded before the next control step.
        Strict ordering and the same hard history bound make duplicate,
        malformed, or unconsumed input fail closed.
        """

        report = _PhysicalMouseInput(
            timestamp_ns,
            delta_x_counts,
            delta_y_counts,
        )
        if report.timestamp_ns <= self._last_physical_input_ns:
            self._reset_tracking()
            raise ValueError("non-monotonic-physical-input-history")
        if (
            len(self._commands) + len(self._physical_inputs) + 1
            > self.config.maximum_command_history
        ):
            self._reset_tracking()
            raise RuntimeError("physical-input-history-overflow")
        self._physical_inputs.append(report)
        self._last_physical_input_ns = report.timestamp_ns

    def step(
        self,
        now_ns: int,
        *,
        engaged: bool,
        observation: ScreenErrorObservation | None = None,
        correlated_lookahead: CorrelatedLookaheadObservation | None = None,
        target_lost: bool = False,
        observation_expected: bool | None = None,
        emitted_commands: Iterable[EmittedMouseCommand] = (),
    ) -> CalibratedControlOutput:
        """Ingest new physical evidence and return a bounded X/Y count rate."""

        current_ns = _timestamp(now_ns, "control timestamp")
        adaptive_observation_elapsed = 0.0
        correlated_authority = self._correlated_lookahead_authority_state
        prior_correlated_authority = correlated_authority
        verified_residual_measurement_x = False
        verified_residual_measurement_y = False
        verified_projection_x = 0.0
        verified_projection_y = 0.0
        predictive_authority_revoked_x = False
        predictive_authority_revoked_y = False
        if not isinstance(engaged, bool):
            raise TypeError("engaged must be bool")
        if not isinstance(target_lost, bool):
            raise TypeError("target_lost must be bool")
        if observation_expected is not None:
            if not isinstance(observation_expected, bool):
                raise TypeError("observation_expected must be bool or None")
            # Backwards-compatible alias.  Historically this flag's name could
            # be read as "a detector callback occurred".  Its only safe meaning
            # is the new explicit-loss meaning: tracker prediction must pass
            # False and no observation.
            target_lost = target_lost or observation_expected
        if observation is not None and not isinstance(
            observation, ScreenErrorObservation
        ):
            raise TypeError("observation must be a ScreenErrorObservation or None")
        if correlated_lookahead is not None and not isinstance(
            correlated_lookahead,
            CorrelatedLookaheadObservation,
        ):
            raise TypeError(
                "correlated_lookahead must be a "
                "CorrelatedLookaheadObservation or None"
            )
        if (
            correlated_lookahead is not None
            and not self.config.require_motion_corroboration_for_feedforward
        ):
            raise ValueError(
                "correlated lookahead requires corroboration-gated control"
            )
        if observation is not None and correlated_lookahead is not None:
            raise ValueError(
                "observation and correlated_lookahead are mutually exclusive"
            )
        if target_lost and (
            observation is not None or correlated_lookahead is not None
        ):
            self._reset_tracking()
            raise ValueError("target_lost cannot accompany a real observation")
        if self._last_step_ns >= 0 and current_ns <= self._last_step_ns:
            self._reset_tracking()
            return self._zero(current_ns, "non-monotonic-clock")
        self._last_step_ns = current_ns

        command_error = self._ingest_commands(emitted_commands, current_ns)
        if command_error is not None:
            self._reset_tracking()
            self._prune_commands(current_ns)
            return self._zero(current_ns, command_error)

        if not engaged:
            self._reset_tracking()
            self._prune_commands(current_ns)
            return self._zero(current_ns, "released")
        if target_lost:
            self._reset_tracking()
            self._prune_commands(current_ns)
            return self._zero(current_ns, "target-lost")

        if correlated_lookahead is not None:
            if correlated_lookahead.lookahead.timestamp_ns > current_ns:
                self._reset_tracking()
                return self._zero(current_ns, "future-observation")
            incoming_identity_deadline_ns = (
                correlated_lookahead.lookahead.identity_deadline_ns
            )
            assert incoming_identity_deadline_ns is not None
            if current_ns >= incoming_identity_deadline_ns:
                self._reset_tracking()
                self._prune_commands(current_ns)
                return self._zero(current_ns, "identity-expired")
            (
                discontinuity,
                adaptive_observation_elapsed,
                correlated_authority,
                verified_residual_measurement_x,
                verified_residual_measurement_y,
            ) = self._accept_correlated_lookahead(correlated_lookahead)
            if discontinuity is not None:
                return self._zero(current_ns, discontinuity)
            predictive_authority_revoked_x = bool(
                prior_correlated_authority is not None
                and (
                    (
                        prior_correlated_authority.total_x > 0.0
                        and correlated_authority.total_x <= 0.0
                    )
                    or (
                        prior_correlated_authority.projection_x > 0.0
                        and correlated_authority.projection_x <= 0.0
                    )
                )
            )
            predictive_authority_revoked_y = bool(
                prior_correlated_authority is not None
                and (
                    (
                        prior_correlated_authority.total_y > 0.0
                        and correlated_authority.total_y <= 0.0
                    )
                    or (
                        prior_correlated_authority.projection_y > 0.0
                        and correlated_authority.projection_y <= 0.0
                    )
                )
            )
            self._correlated_lookahead_authority_state = correlated_authority
            self._correlated_lookahead_authority_deadline_ns = min(
                correlated_lookahead.lookahead.timestamp_ns
                + round(
                    CORRELATED_LOOKAHEAD_MAX_LEAD_SECONDS * _NS_PER_SECOND
                ),
                correlated_lookahead.lookahead.identity_deadline_ns,
            )
        elif observation is not None:
            self._correlated_lookahead_authority_state = None
            self._correlated_lookahead_authority_deadline_ns = None
            self._clear_verified_residual_projection()
            # This ordinary sample owns its own provenance. Do not let the
            # prior batch's local cap/diagnostic survive merely because the
            # immutable state was cleared above.
            correlated_authority = None
            if observation.timestamp_ns > current_ns:
                self._reset_tracking()
                return self._zero(current_ns, "future-observation")
            incoming_identity_deadline_ns = observation.identity_deadline_ns
            effective_identity_deadline_ns = (
                incoming_identity_deadline_ns
                if incoming_identity_deadline_ns is not None
                else self._identity_deadline_ns
            )
            if (
                effective_identity_deadline_ns is not None
                and current_ns >= effective_identity_deadline_ns
            ):
                self._reset_tracking()
                self._prune_commands(current_ns)
                return self._zero(current_ns, "identity-expired")
            previous_observation = self._last_observation
            discontinuity = self._accept_observation(observation)
            if discontinuity is not None:
                return self._zero(current_ns, discontinuity)
            self._last_correlated_identity = None
            self._last_observation_was_correlated_lookahead = False
            if (
                prior_correlated_authority is not None
                and not self._independent_pursuit_authorized
                and not self._body_derived_motion_permitted
            ):
                # A true predictive-to-static provenance transition must also
                # withdraw the adapter's ordinary EMA/fractional tail. Valid
                # independent or mapped-body handoffs retain their intentional
                # smoothing and are not treated as a revoke.
                predictive_authority_revoked_x = bool(
                    prior_correlated_authority.total_x > 0.0
                    or prior_correlated_authority.projection_x > 0.0
                )
                predictive_authority_revoked_y = bool(
                    prior_correlated_authority.total_y > 0.0
                    or prior_correlated_authority.projection_y > 0.0
                )
            if previous_observation is not None:
                adaptive_observation_elapsed = (
                    observation.timestamp_ns - previous_observation.timestamp_ns
                ) / _NS_PER_SECOND

        identity_deadline_ns = self._identity_deadline_ns
        if (
            identity_deadline_ns is not None
            and current_ns >= identity_deadline_ns
        ):
            # Identity authority is an absolute lease, not observation
            # freshness.  Once it expires neither static position nor
            # predictive motion may survive a capture/detector starvation.
            self._reset_tracking()
            self._prune_commands(current_ns)
            return self._zero(current_ns, "identity-expired")

        correlated_deadline_ns = (
            self._correlated_lookahead_authority_deadline_ns
        )
        if (
            correlated_deadline_ns is not None
            and current_ns >= correlated_deadline_ns
        ):
            # Expire only predictive authority. The newest causal P1 position
            # retains the ordinary observation freshness lease, but a
            # residual learned exclusively from that P1 cannot outlive the
            # endpoint's shorter verified-correlation lease. An ordinary
            # replacement observation clears the correlated deadline above,
            # so its separately authorized residual remains unaffected.
            expired_authority = correlated_authority
            expired_residual_x = self._residual_pursuit_x.authority > 0.0
            expired_residual_y = self._residual_pursuit_y.authority > 0.0
            predictive_authority_revoked_x = bool(
                predictive_authority_revoked_x
                or expired_residual_x
                or (
                    expired_authority is not None
                    and (
                        expired_authority.total_x > 0.0
                        or expired_authority.projection_x > 0.0
                    )
                )
            )
            predictive_authority_revoked_y = bool(
                predictive_authority_revoked_y
                or expired_residual_y
                or (
                    expired_authority is not None
                    and (
                        expired_authority.total_y > 0.0
                        or expired_authority.projection_y > 0.0
                    )
                )
            )
            self._independent_pursuit_authorized = False
            self._body_derived_adaptive_pursuit_confidence_x = 0.0
            self._body_derived_adaptive_pursuit_confidence_y = 0.0
            self._clear_residual_pursuit()
            self._correlated_lookahead_authority_state = None
            self._correlated_lookahead_authority_deadline_ns = None
            correlated_authority = None

        if self._body_derived_motion_permitted:
            deadline_ns = self._body_derived_motion_deadline_ns
            if deadline_ns is None or current_ns >= deadline_ns:
                # The direct-head identity lease is independent of the mapped
                # position freshness lease.  Expiry withdraws only predictive
                # motion; the accepted absolute point may continue its ordinary
                # static bridge until the existing observation lease expires.
                self._clear_body_derived_motion()

        latest = self._last_observation
        if latest is None:
            return self._zero(current_ns, "awaiting-observation")
        age_ns = current_ns - latest.timestamp_ns
        if age_ns < 0:
            self._reset_tracking()
            return self._zero(current_ns, "future-observation")
        if age_ns > round(self.config.stale_after_seconds * _NS_PER_SECOND):
            self._reset_tracking()
            self._prune_commands(current_ns)
            return self._zero(current_ns, "stale-observation")
        if not self._ready:
            return self._zero(current_ns, "awaiting-confirmation")

        horizon_ns = current_ns + self._delay_ns
        pending_x, pending_y = self._counts_landing_between(
            latest.timestamp_ns,
            horizon_ns,
        )
        physical_input_pending_x, physical_input_pending_y = (
            self._physical_axes_landing_between(
                latest.timestamp_ns,
                horizon_ns,
            )
        )
        horizon_seconds = (horizon_ns - latest.timestamp_ns) / _NS_PER_SECOND
        paired_observer = latest.velocity_error_x_pixels is not None
        active_motion_corroboration_confidence = (
            self._motion_corroboration_confidence
            if self._independent_pursuit_authorized
            else 0.0
        )
        if paired_observer:
            projected_x, projected_velocity_x, projected_covariance_x = (
                self._paired_predict_axis(
                    self._paired_position_x,
                    self._velocity_x,
                    self._paired_covariance_x,
                    horizon_seconds,
                    self.plant.gain_x_pixels_per_count * pending_x,
                )
            )
            projected_y, projected_velocity_y, projected_covariance_y = (
                self._paired_predict_axis(
                    self._paired_position_y,
                    self._velocity_y,
                    self._paired_covariance_y,
                    horizon_seconds,
                    self.plant.gain_y_pixels_per_count * pending_y,
                )
            )
            position_sigma_x = math.sqrt(projected_covariance_x[0])
            position_sigma_y = math.sqrt(projected_covariance_y[0])
            velocity_sigma_x = math.sqrt(projected_covariance_x[2])
            velocity_sigma_y = math.sqrt(projected_covariance_y[2])
            feedforward_confidence_x = self._paired_velocity_confidence(
                projected_velocity_x,
                velocity_sigma_x,
            ) * self._paired_measurement_agreement_x
            feedforward_confidence_y = self._paired_velocity_confidence(
                projected_velocity_y,
                velocity_sigma_y,
            ) * self._paired_measurement_agreement_y
            ordinary_feedforward_confidence_x = feedforward_confidence_x
            ordinary_feedforward_confidence_y = feedforward_confidence_y
            body_feedforward_observer_confidence_x = 0.0
            body_feedforward_observer_confidence_y = 0.0
            if self.config.require_motion_corroboration_for_feedforward:
                if self._body_derived_motion_permitted:
                    # These are explicit provenance grants, not synthesized
                    # corroboration.  A high-confidence paired observer cannot
                    # multiply or otherwise bypass either configured ceiling.
                    # Projection additionally retains per-axis paired-observer
                    # covariance, signal, and direction persistence.  A strong
                    # horizontal pursuit therefore cannot authorize vertical
                    # point jitter or keep that axis armed through a reversal.
                    motion_projection_confidence_x = (
                        self.config.maximum_body_derived_projection_fraction
                        * self._body_derived_motion_confidence_x
                    )
                    motion_projection_confidence_y = (
                        self.config.maximum_body_derived_projection_fraction
                        * self._body_derived_motion_confidence_y
                    )
                    body_feedforward_observer_confidence_x = (
                        feedforward_confidence_x
                    )
                    body_feedforward_observer_confidence_y = (
                        feedforward_confidence_y
                    )
                else:
                    feedforward_confidence_x *= (
                        active_motion_corroboration_confidence
                    )
                    feedforward_confidence_y *= (
                        active_motion_corroboration_confidence
                    )
                    motion_projection_confidence = min(
                        active_motion_corroboration_confidence
                        / _CORROBORATION_FULL_MOTION_PROJECTION_CONFIDENCE,
                        1.0,
                    )
                    motion_projection_confidence_x = (
                        motion_projection_confidence
                    )
                    motion_projection_confidence_y = (
                        motion_projection_confidence
                    )

                # A newest-frame statistically material stop/reversal is
                # stronger evidence than either velocity observer's retained
                # state.  Remove that axis from source-age projection now,
                # before body reconciliation can reuse an adaptive grant.  The
                # causal static position remains available to calm feedback.
                if self._paired_material_stop_or_reversal_x:
                    motion_projection_confidence_x = 0.0
                    self._body_derived_adaptive_pursuit_confidence_x = 0.0
                if self._paired_material_stop_or_reversal_y:
                    motion_projection_confidence_y = 0.0
                    self._body_derived_adaptive_pursuit_confidence_y = 0.0

                # A velocity estimate is predictive motion whether it enters
                # as the explicit feed-forward term or as ``v * horizon`` in
                # positional feedback, so the selected evidence path gates
                # both.  Body-derived control keeps the caller's causal mapped
                # point as its exact static position; the paired Kalman state
                # supplies velocity only.  That prevents observer overshoot
                # from carrying a filtered <=4 px orbit outside its deadzone.
                # Independent direct-head control retains its historical
                # paired position. Default/explicit profiles retain their
                # historical full paired projection because they do not enable
                # this automatic-only requirement.
                measured_feedforward_error_x = latest.error_x_pixels
                measured_feedforward_error_y = latest.error_y_pixels
                if self._body_derived_motion_permitted:
                    # A vector already at the calibrated output envelope is a
                    # saturation problem, not an integral-bias problem. Keep
                    # the adaptive medium-speed state out of that regime on
                    # both axes; otherwise a cap-bound X pursuit can teach Y a
                    # rate which survives into the next reversal.
                    adaptive_pursuit_vector_speed = math.hypot(
                        projected_velocity_x,
                        projected_velocity_y,
                    )
                    if (
                        adaptive_pursuit_vector_speed
                        >= (
                            _BODY_DERIVED_ADAPTIVE_PURSUIT_MAXIMUM_VECTOR_SPEED_PIXELS_PER_SECOND
                        )
                    ):
                        self._body_derived_adaptive_pursuit_confidence_x = 0.0
                        self._body_derived_adaptive_pursuit_confidence_y = 0.0
                    assert latest.velocity_error_x_pixels is not None
                    assert latest.velocity_error_y_pixels is not None
                    adaptive_position_x = (
                        self._adaptive_body_position_reconciliation(
                            measured_feedforward_error_x,
                            latest.velocity_error_x_pixels,
                            projected_velocity_x,
                            self._body_derived_adaptive_pursuit_confidence_x,
                            motion_confidence=(
                                self._body_derived_motion_confidence_x
                            ),
                            fresh_motion_confidence=(
                                self._body_derived_fresh_motion_confidence_x
                            ),
                        )
                    )
                    adaptive_position_y = (
                        self._adaptive_body_position_reconciliation(
                            measured_feedforward_error_y,
                            latest.velocity_error_y_pixels,
                            projected_velocity_y,
                            self._body_derived_adaptive_pursuit_confidence_y,
                            motion_confidence=(
                                self._body_derived_motion_confidence_y
                            ),
                            fresh_motion_confidence=(
                                self._body_derived_fresh_motion_confidence_y
                            ),
                        )
                    )
                    static_projected_x = (
                        measured_feedforward_error_x
                        + adaptive_position_x
                        - self.plant.gain_x_pixels_per_count * pending_x
                    )
                    static_projected_y = (
                        measured_feedforward_error_y
                        + adaptive_position_y
                        - self.plant.gain_y_pixels_per_count * pending_y
                    )
                    body_projection_horizon = (
                        horizon_seconds
                        + self.config.additional_body_derived_projection_seconds
                    )
                    full_projected_x = (
                        static_projected_x
                        + projected_velocity_x * body_projection_horizon
                    )
                    full_projected_y = (
                        static_projected_y
                        + projected_velocity_y * body_projection_horizon
                    )
                else:
                    static_projected_x = (
                        self._paired_position_x
                        - self.plant.gain_x_pixels_per_count * pending_x
                    )
                    static_projected_y = (
                        self._paired_position_y
                        - self.plant.gain_y_pixels_per_count * pending_y
                    )
                    full_projected_x = projected_x
                    full_projected_y = projected_y
                if correlated_authority is not None:
                    motion_projection_confidence_x = min(
                        motion_projection_confidence_x,
                        correlated_authority.projection_x,
                    )
                    motion_projection_confidence_y = min(
                        motion_projection_confidence_y,
                        correlated_authority.projection_y,
                    )
                projected_x = static_projected_x + (
                    motion_projection_confidence_x
                    * (full_projected_x - static_projected_x)
                )
                projected_y = static_projected_y + (
                    motion_projection_confidence_y
                    * (full_projected_y - static_projected_y)
                )
                if self._body_derived_motion_permitted:
                    # Body-derived velocity is correlated with the mapped
                    # position, so ordinary enlarged feed-forward remains
                    # closed-loop residual gated. The separate fast reserve
                    # can remain live at zero error for constant-speed pursuit,
                    # but requires fresh command-compensated motion and backs
                    # out smoothly once either residual is materially ahead.
                    # A target stop, reversal, or manual approach withdraws its
                    # fresh-motion gate before the filtered velocity decays.
                    feedforward_cap_x = self._body_derived_feedforward_cap(
                        measured_feedforward_error_x,
                        projected_x,
                        projected_velocity_x,
                        fresh_motion_confidence=(
                            self._body_derived_fresh_motion_confidence_x
                        ),
                    )
                    feedforward_cap_y = self._body_derived_feedforward_cap(
                        measured_feedforward_error_y,
                        projected_y,
                        projected_velocity_y,
                        fresh_motion_confidence=(
                            self._body_derived_fresh_motion_confidence_y
                        ),
                    )
                    ordinary_feedforward_cap_x = (
                        self._body_derived_feedforward_cap(
                            measured_feedforward_error_x,
                            projected_x,
                            projected_velocity_x,
                        )
                    )
                    ordinary_feedforward_cap_y = (
                        self._body_derived_feedforward_cap(
                            measured_feedforward_error_y,
                            projected_y,
                            projected_velocity_y,
                        )
                    )
                    ordinary_feedforward_confidence_x = min(
                        body_feedforward_observer_confidence_x,
                        ordinary_feedforward_cap_x,
                    ) * self._body_derived_motion_confidence_x
                    ordinary_feedforward_confidence_y = min(
                        body_feedforward_observer_confidence_y,
                        ordinary_feedforward_cap_y,
                    ) * self._body_derived_motion_confidence_y
                    feedforward_confidence_x = min(
                        body_feedforward_observer_confidence_x,
                        feedforward_cap_x,
                    ) * self._body_derived_motion_confidence_x
                    feedforward_confidence_y = min(
                        body_feedforward_observer_confidence_y,
                        feedforward_cap_y,
                    )
                    feedforward_confidence_y *= (
                        self._body_derived_motion_confidence_y
                    )
            if not self._body_derived_motion_permitted:
                ordinary_feedforward_confidence_x = feedforward_confidence_x
                ordinary_feedforward_confidence_y = feedforward_confidence_y
            # The primary command-compensated velocity is one continuous
            # physical hypothesis even when its optional authority alternates
            # between pixel/body corroboration and mapped-body provenance.  A
            # provenance switch can withdraw or rebuild the *current* grant,
            # but must not erase a pursuit rate already learned from coherent
            # motion.  Newest-frame motion and residual direction are evaluated
            # on every accepted sample, so a stop, reversal, or user's manual
            # approach still clears the retained rate immediately.
            independent_direction_fully_qualified = bool(
                self._corroboration_direction_persistence_seconds
                >= _CORROBORATION_FULL_DIRECTION_PERSISTENCE_SECONDS
            )
            if self._body_derived_motion_permitted:
                adaptive_motion_confidence_x = (
                    self._body_derived_motion_confidence_x
                )
                adaptive_motion_confidence_y = (
                    self._body_derived_motion_confidence_y
                )
                adaptive_fresh_motion_confidence_x = (
                    self._body_derived_fresh_motion_confidence_x
                )
                adaptive_fresh_motion_confidence_y = (
                    self._body_derived_fresh_motion_confidence_y
                )
                adaptive_direction_persistence_x = (
                    self._body_derived_direction_persistence_x_seconds
                )
                adaptive_direction_persistence_y = (
                    self._body_derived_direction_persistence_y_seconds
                )
                adaptive_zero_speed = (
                    _BODY_DERIVED_ADAPTIVE_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND
                )
                adaptive_full_speed = (
                    _BODY_DERIVED_ADAPTIVE_PURSUIT_FULL_SPEED_PIXELS_PER_SECOND
                )
                adaptive_rise_time_constant = (
                    _BODY_DERIVED_ADAPTIVE_PURSUIT_RISE_TIME_CONSTANT_SECONDS
                )
            else:
                adaptive_motion_confidence_x = (
                    active_motion_corroboration_confidence
                )
                adaptive_motion_confidence_y = (
                    active_motion_corroboration_confidence
                )
                adaptive_fresh_motion_confidence_x = (
                    self._paired_fresh_motion_confidence_x
                )
                adaptive_fresh_motion_confidence_y = (
                    self._paired_fresh_motion_confidence_y
                )
                adaptive_direction_persistence_x = (
                    self._common_pursuit_direction_persistence_seconds
                )
                adaptive_direction_persistence_y = (
                    self._common_pursuit_direction_persistence_seconds
                )
                adaptive_zero_speed = (
                    _COMMON_PAIRED_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND
                )
                adaptive_full_speed = (
                    _COMMON_PAIRED_PURSUIT_FULL_SPEED_PIXELS_PER_SECOND
                )
                adaptive_rise_time_constant = (
                    _COMMON_PAIRED_PURSUIT_RISE_TIME_CONSTANT_SECONDS
                )
            adaptive_pursuit_vector_speed = math.hypot(
                projected_velocity_x,
                projected_velocity_y,
            )
            handoff_deadline_ns = self._common_pursuit_handoff_deadline_ns
            handoff_live = (
                handoff_deadline_ns is not None
                and current_ns < handoff_deadline_ns
                and self._body_derived_motion_permitted
            )
            if not self._body_derived_motion_permitted:
                if max(
                    adaptive_motion_confidence_x,
                    adaptive_motion_confidence_y,
                ) > 0.0:
                    # Fresh independent evidence has requalified the retained
                    # state; it no longer needs the bounded handoff lease.
                    self._common_pursuit_handoff_deadline_ns = None
                elif not handoff_live:
                    # Zero independent confidence outside an explicit source
                    # handoff is a real authorization loss, not a reason to
                    # coast indefinitely on an old near-unity velocity grant.
                    self._body_derived_adaptive_pursuit_confidence_x = 0.0
                    self._body_derived_adaptive_pursuit_confidence_y = 0.0
            elif handoff_deadline_ns is not None and not handoff_live:
                # Body fallback may retain the previously earned fraction only
                # through the short source-transition lease.  Beyond it, revoke
                # that independent state completely.  The ordinary body path
                # remains separately observer/confidence gated, and coherent
                # 900--1200 px/s body motion can re-earn adaptive authority from
                # zero without inheriting an unverified feed-forward floor.
                self._body_derived_adaptive_pursuit_confidence_x = 0.0
                self._body_derived_adaptive_pursuit_confidence_y = 0.0
                self._common_pursuit_handoff_deadline_ns = None
            if (
                self.config.require_motion_corroboration_for_feedforward
                and adaptive_observation_elapsed > 0.0
            ):
                retain_ambiguous_fresh_x = bool(
                    self._body_derived_adaptive_pursuit_confidence_x > 0.0
                    and adaptive_fresh_motion_confidence_x <= 0.0
                    and not self._paired_material_stop_or_reversal_x
                    and (
                        not self._body_derived_motion_permitted
                        or handoff_live
                    )
                    and self._adaptive_pursuit_low_fresh_samples_x == 1
                    and adaptive_observation_elapsed
                    <= _COMMON_PAIRED_PURSUIT_AMBIGUOUS_FRESH_MAX_SECONDS
                )
                retain_ambiguous_fresh_y = bool(
                    self._body_derived_adaptive_pursuit_confidence_y > 0.0
                    and adaptive_fresh_motion_confidence_y <= 0.0
                    and not self._paired_material_stop_or_reversal_y
                    and (
                        not self._body_derived_motion_permitted
                        or handoff_live
                    )
                    and self._adaptive_pursuit_low_fresh_samples_y == 1
                    and adaptive_observation_elapsed
                    <= _COMMON_PAIRED_PURSUIT_AMBIGUOUS_FRESH_MAX_SECONDS
                )
                if self._paired_material_stop_or_reversal_x:
                    self._body_derived_adaptive_pursuit_confidence_x = 0.0
                if self._paired_material_stop_or_reversal_y:
                    self._body_derived_adaptive_pursuit_confidence_y = 0.0
                self._body_derived_adaptive_pursuit_confidence_x = (
                    self._adaptive_body_pursuit_confidence(
                        self._body_derived_adaptive_pursuit_confidence_x,
                        measured_error=measured_feedforward_error_x,
                        projected_error=projected_x,
                        projected_velocity=projected_velocity_x,
                        projected_vector_speed=adaptive_pursuit_vector_speed,
                        motion_confidence=(
                            adaptive_motion_confidence_x
                            if self._body_derived_motion_permitted
                            or independent_direction_fully_qualified
                            else 0.0
                        ),
                        fresh_motion_confidence=(
                            adaptive_fresh_motion_confidence_x
                        ),
                        direction_persistence_seconds=(
                            adaptive_direction_persistence_x
                        ),
                        elapsed=adaptive_observation_elapsed,
                        qualification_zero_speed=adaptive_zero_speed,
                        qualification_full_speed=adaptive_full_speed,
                        rise_time_constant=adaptive_rise_time_constant,
                        retain_through_ambiguous_fresh_motion=(
                            retain_ambiguous_fresh_x
                        ),
                        binary_promotion=(
                            not self._body_derived_motion_permitted
                            and self._independent_pursuit_authorized
                            and independent_direction_fully_qualified
                        ),
                    )
                )
                self._body_derived_adaptive_pursuit_confidence_y = (
                    self._adaptive_body_pursuit_confidence(
                        self._body_derived_adaptive_pursuit_confidence_y,
                        measured_error=measured_feedforward_error_y,
                        projected_error=projected_y,
                        projected_velocity=projected_velocity_y,
                        projected_vector_speed=adaptive_pursuit_vector_speed,
                        motion_confidence=(
                            adaptive_motion_confidence_y
                            if self._body_derived_motion_permitted
                            or independent_direction_fully_qualified
                            else 0.0
                        ),
                        fresh_motion_confidence=(
                            adaptive_fresh_motion_confidence_y
                        ),
                        direction_persistence_seconds=(
                            adaptive_direction_persistence_y
                        ),
                        elapsed=adaptive_observation_elapsed,
                        qualification_zero_speed=adaptive_zero_speed,
                        qualification_full_speed=adaptive_full_speed,
                        rise_time_constant=adaptive_rise_time_constant,
                        retain_through_ambiguous_fresh_motion=(
                            retain_ambiguous_fresh_y
                        ),
                        binary_promotion=(
                            not self._body_derived_motion_permitted
                            and self._independent_pursuit_authorized
                            and independent_direction_fully_qualified
                        ),
                    )
                )
                if (
                    self._body_derived_motion_permitted
                    and handoff_deadline_ns is not None
                    and not handoff_live
                    and max(
                        self._body_derived_adaptive_pursuit_confidence_x,
                        self._body_derived_adaptive_pursuit_confidence_y,
                    )
                    > self.config.maximum_body_derived_feedforward_fraction
                ):
                    # The expired independent bridge has now been replaced by
                    # a freshly body-qualified high-speed state.
                    self._common_pursuit_handoff_deadline_ns = None
            if correlated_authority is not None:
                # Current output is capped below by the authority frozen at
                # P0. Keep a stronger internally qualified P1 estimate hidden
                # so the *next* independent P0 can choose whether to expose a
                # bounded part of it. A revoked axis still clears immediately;
                # profiles which did not opt into correlated fast retention
                # preserve the historical destructive ordinary-cap behavior.
                correlated_pursuit_maximum = (
                    self.config
                    .maximum_correlated_lookahead_pursuit_feedforward_fraction
                )
                retain_hidden = bool(
                    correlated_pursuit_maximum
                    > self.config.maximum_body_derived_feedforward_fraction
                )
                if retain_hidden:
                    if correlated_authority.total_x <= 0.0:
                        self._body_derived_adaptive_pursuit_confidence_x = 0.0
                    if correlated_authority.total_y <= 0.0:
                        self._body_derived_adaptive_pursuit_confidence_y = 0.0
                else:
                    self._body_derived_adaptive_pursuit_confidence_x = min(
                        self._body_derived_adaptive_pursuit_confidence_x,
                        correlated_authority.total_x,
                    )
                    self._body_derived_adaptive_pursuit_confidence_y = min(
                        self._body_derived_adaptive_pursuit_confidence_y,
                        correlated_authority.total_y,
                    )
            residual_observation = (
                correlated_lookahead.lookahead
                if correlated_lookahead is not None
                else observation
            )
            residual_measurement_event = residual_observation is not None
            residual_common_measured = bool(
                residual_measurement_event
                and self._last_observation is residual_observation
                and residual_observation is not None
                and residual_observation.velocity_error_x_pixels is not None
                and residual_observation.identity_deadline_ns is not None
                and not self._last_innovation_rejected
            )
            residual_ordinary_authorized = bool(
                residual_common_measured
                and residual_observation is not None
                and (
                    residual_observation.body_derived_motion_permitted
                    or self._independent_pursuit_authorized
                )
            )
            self._update_residual_pursuit(
                measurement_event=residual_measurement_event,
                current_measured_x=(
                    residual_ordinary_authorized
                    or (
                        residual_common_measured
                        and verified_residual_measurement_x
                    )
                ),
                current_measured_y=(
                    residual_ordinary_authorized
                    or (
                        residual_common_measured
                        and verified_residual_measurement_y
                    )
                ),
                elapsed=adaptive_observation_elapsed,
                measured_error_x=latest.error_x_pixels,
                measured_error_y=latest.error_y_pixels,
                projected_error_x=projected_x,
                projected_error_y=projected_y,
                projected_velocity_x=projected_velocity_x,
                projected_velocity_y=projected_velocity_y,
                position_sigma_x=position_sigma_x,
                position_sigma_y=position_sigma_y,
                position_agreement_x=self._paired_measurement_agreement_x,
                position_agreement_y=self._paired_measurement_agreement_y,
                revoke_x=(
                    self._last_innovation_rejected
                    or self._paired_material_stop_or_reversal_x
                    or physical_input_pending_x
                ),
                revoke_y=(
                    self._last_innovation_rejected
                    or self._paired_material_stop_or_reversal_y
                    or physical_input_pending_y
                ),
            )
            if self._residual_pursuit_x.authority <= 0.0:
                self._verified_residual_projection_x = False
            elif residual_measurement_event:
                # Only a currently accepted, internally verified P1 endpoint
                # may create or renew this narrow projection permission.  An
                # unverified endpoint, per-axis stop/reversal, rejection, or
                # physical input has already cleared the corresponding
                # residual state above and therefore withdraws immediately.
                self._verified_residual_projection_x = bool(
                    verified_residual_measurement_x
                )
            if self._residual_pursuit_y.authority <= 0.0:
                self._verified_residual_projection_y = False
            elif residual_measurement_event:
                self._verified_residual_projection_y = bool(
                    verified_residual_measurement_y
                )

            # The verified residual path previously added velocity authority
            # while leaving its accepted endpoint at the static source-time
            # coordinate.  At 900--2400 px/s that omitted the already-known
            # source-age plus calibrated-plant horizon and guaranteed visible
            # trail even with correct feed-forward.  Once the existing
            # three-sample/16 ms residual proof has earned authority, reuse
            # only that bounded projection on the proven axis.  Normalize by
            # the unchanged residual ceiling so activation follows the
            # learner's existing smooth 40 ms rise instead of producing a
            # binary rate step. This known-horizon portion adds no unmeasured
            # lead or extra velocity.
            residual_projection_maximum = (
                self.config.maximum_residual_pursuit_feedforward_fraction
            )
            if residual_projection_maximum > 0.0:
                verified_projection_x = (
                    min(
                        self._residual_pursuit_x.authority
                        / residual_projection_maximum,
                        1.0,
                    )
                    if self._verified_residual_projection_x
                    else 0.0
                )
                verified_projection_y = (
                    min(
                        self._residual_pursuit_y.authority
                        / residual_projection_maximum,
                        1.0,
                    )
                    if self._verified_residual_projection_y
                    else 0.0
                )
                if verified_projection_x > motion_projection_confidence_x:
                    projected_x = static_projected_x + (
                        verified_projection_x
                        * (full_projected_x - static_projected_x)
                    )
                if verified_projection_y > motion_projection_confidence_y:
                    projected_y = static_projected_y + (
                        verified_projection_y
                        * (full_projected_y - static_projected_y)
                    )
            # Provenance-derived authority is selected and, for P1, capped by
            # the independently frozen P0 grant first.  Residual pursuit is a
            # separately learned missing-authority reserve: add it only after
            # that cap, and never beyond the existing measured-pursuit ceiling.
            # This ordering lets several numerically verified P1 measurements
            # repair demonstrated trail without allowing one endpoint to create
            # or enlarge the ordinary flow/body grant.
            if self.config.require_motion_corroboration_for_feedforward:
                feedforward_confidence_x = min(
                    max(
                        feedforward_confidence_x,
                        self._body_derived_adaptive_pursuit_confidence_x,
                    ),
                    self.config.maximum_velocity_feedforward_fraction,
                )
                feedforward_confidence_y = min(
                    max(
                        feedforward_confidence_y,
                        self._body_derived_adaptive_pursuit_confidence_y,
                    ),
                    self.config.maximum_velocity_feedforward_fraction,
                )
                if correlated_authority is not None:
                    ordinary_feedforward_confidence_x = min(
                        ordinary_feedforward_confidence_x,
                        correlated_authority.ordinary_x,
                    )
                    ordinary_feedforward_confidence_y = min(
                        ordinary_feedforward_confidence_y,
                        correlated_authority.ordinary_y,
                    )
                    # A currently verified flow endpoint owns its separately
                    # configured ceiling directly.  The shared adaptive state
                    # is intentionally re-clamped to the generic body ceiling,
                    # so use only the portion of this still-live correlated
                    # grant above that ceiling to restore the verified setpoint.
                    # When flow is absent/revoked ``total_*`` cannot cross the
                    # body ceiling, preserving the former path bit-for-bit.
                    body_pursuit_ceiling = (
                        self.config.maximum_body_derived_pursuit_feedforward_fraction
                    )
                    if correlated_authority.total_x > body_pursuit_ceiling:
                        feedforward_confidence_x = max(
                            feedforward_confidence_x,
                            correlated_authority.total_x,
                        )
                    if correlated_authority.total_y > body_pursuit_ceiling:
                        feedforward_confidence_y = max(
                            feedforward_confidence_y,
                            correlated_authority.total_y,
                        )
                    feedforward_confidence_x = min(
                        feedforward_confidence_x,
                        correlated_authority.total_x,
                    )
                    feedforward_confidence_y = min(
                        feedforward_confidence_y,
                        correlated_authority.total_y,
                    )
                residual_total_ceiling = min(
                    self.config.maximum_body_derived_pursuit_feedforward_fraction,
                    self.config.maximum_velocity_feedforward_fraction,
                )
                feedforward_confidence_x += min(
                    self._residual_pursuit_x.authority,
                    max(residual_total_ceiling - feedforward_confidence_x, 0.0),
                )
                feedforward_confidence_y += min(
                    self._residual_pursuit_y.authority,
                    max(residual_total_ceiling - feedforward_confidence_y, 0.0),
                )
                # Mask the complete velocity contribution on the material
                # axis, including the ordinary sparse-corroboration term.  The
                # adaptive reserve was already revoked above, but allowing the
                # older base observer to coast for this sample still produced
                # a visible stop/reversal kick in causal trace replay.
                if self._paired_material_stop_or_reversal_x:
                    feedforward_confidence_x = 0.0
                    ordinary_feedforward_confidence_x = 0.0
                if self._paired_material_stop_or_reversal_y:
                    feedforward_confidence_y = 0.0
                    ordinary_feedforward_confidence_y = 0.0
            # A raw physical report is already a known plant input, but the
            # corresponding pixels have not reached the newest observation
            # yet.  Anticipate its position through ``pending_*`` above while
            # withholding every velocity term on only that axis.  This stops
            # the output adapter from fighting a user's hand or learning the
            # hand movement as target velocity; feedback remains available in
            # the same direction when a real residual is still present.
            if physical_input_pending_x:
                projected_x = (
                    latest.error_x_pixels
                    - self.plant.gain_x_pixels_per_count * pending_x
                )
                feedforward_confidence_x = 0.0
                ordinary_feedforward_confidence_x = 0.0
                predictive_authority_revoked_x = True
            if physical_input_pending_y:
                projected_y = (
                    latest.error_y_pixels
                    - self.plant.gain_y_pixels_per_count * pending_y
                )
                feedforward_confidence_y = 0.0
                ordinary_feedforward_confidence_y = 0.0
                predictive_authority_revoked_y = True
            self._paired_feedback_hold_x = self._paired_feedback_hold(
                self._paired_feedback_hold_x,
                projected_x,
                position_sigma_x,
            )
            self._paired_feedback_hold_y = self._paired_feedback_hold(
                self._paired_feedback_hold_y,
                projected_y,
                position_sigma_y,
            )
        else:
            projected_velocity_x = self._velocity_x
            projected_velocity_y = self._velocity_y
            projected_x = (
                latest.error_x_pixels
                + projected_velocity_x * horizon_seconds
                - self.plant.gain_x_pixels_per_count * pending_x
            )
            projected_y = (
                latest.error_y_pixels
                + projected_velocity_y * horizon_seconds
                - self.plant.gain_y_pixels_per_count * pending_y
            )
            position_sigma_x = 0.0
            position_sigma_y = 0.0
            velocity_sigma_x = 0.0
            velocity_sigma_y = 0.0
            feedforward_confidence_x = 1.0
            feedforward_confidence_y = 1.0
            ordinary_feedforward_confidence_x = feedforward_confidence_x
            ordinary_feedforward_confidence_y = feedforward_confidence_y
        if not all(
            math.isfinite(value)
            for value in (
                projected_x,
                projected_y,
                projected_velocity_x,
                projected_velocity_y,
                position_sigma_x,
                position_sigma_y,
                velocity_sigma_x,
                velocity_sigma_y,
                feedforward_confidence_x,
                feedforward_confidence_y,
            )
        ):
            self._reset_tracking()
            return self._zero(current_ns, "non-finite-control-state")

        # Large-error acquisition slew cap: scale both axis rate limits down
        # proportionally when the radial error exceeds the threshold. Scaling
        # (rather than clamping each axis independently) preserves the approach
        # direction so the reticle glides straight at the target.
        limit_x = self.config.maximum_rate_x_counts_per_second
        limit_y = self.config.maximum_rate_y_counts_per_second
        slew_threshold = self.config.acquisition_error_threshold_pixels
        if slew_threshold > 0.0:
            radial_error = math.hypot(projected_x, projected_y)
            if radial_error > slew_threshold:
                slew = self.config.acquisition_rate_counts_per_second
                limit_x = min(limit_x, slew)
                limit_y = min(limit_y, slew)
        pursuit_reserve_rate_x = 0.0
        pursuit_reserve_rate_y = 0.0
        position_confidence_x = 1.0
        position_confidence_y = 1.0
        if paired_observer:
            feedback_limit_x = min(
                limit_x,
                self.config.maximum_feedback_rate_x_counts_per_second,
            )
            feedback_limit_y = min(
                limit_y,
                self.config.maximum_feedback_rate_y_counts_per_second,
            )
            ordinary_pursuit_authorized = (
                not self.config.require_motion_corroboration_for_feedforward
                or self._body_derived_pursuit_authorized
                or self._independent_pursuit_authorized
            )
            pursuit_authorized_x = bool(
                ordinary_pursuit_authorized
                or self._verified_residual_projection_x
            )
            pursuit_authorized_y = bool(
                ordinary_pursuit_authorized
                or self._verified_residual_projection_y
            )
            pursuit_confidence_x = (
                1.0
                if ordinary_pursuit_authorized
                else verified_projection_x
            )
            pursuit_confidence_y = (
                1.0
                if ordinary_pursuit_authorized
                else verified_projection_y
            )
            position_confidence_x = self._paired_position_feedback_confidence(
                projected_x,
                self._paired_measurement_agreement_x,
            )
            position_confidence_y = self._paired_position_feedback_confidence(
                projected_y,
                self._paired_measurement_agreement_y,
            )
            rate_x, saturated_x = self._paired_axis_rate(
                projected_x,
                projected_velocity_x,
                self.plant.gain_x_pixels_per_count,
                limit_x,
                feedforward_confidence_x,
                feedback_limit=feedback_limit_x,
                feedback_held=self._paired_feedback_hold_x,
                position_confidence=position_confidence_x,
                pursuit_authorized=pursuit_authorized_x,
                pursuit_confidence=pursuit_confidence_x,
            )
            rate_y, saturated_y = self._paired_axis_rate(
                projected_y,
                projected_velocity_y,
                self.plant.gain_y_pixels_per_count,
                limit_y,
                feedforward_confidence_y,
                feedback_limit=feedback_limit_y,
                feedback_held=self._paired_feedback_hold_y,
                position_confidence=position_confidence_y,
                pursuit_authorized=pursuit_authorized_y,
                pursuit_confidence=pursuit_confidence_y,
            )
            ordinary_rate_x, _ordinary_saturated_x = self._paired_axis_rate(
                projected_x,
                projected_velocity_x,
                self.plant.gain_x_pixels_per_count,
                limit_x,
                ordinary_feedforward_confidence_x,
                feedback_limit=feedback_limit_x,
                feedback_held=self._paired_feedback_hold_x,
                position_confidence=position_confidence_x,
                pursuit_authorized=pursuit_authorized_x,
                pursuit_confidence=pursuit_confidence_x,
            )
            ordinary_rate_y, _ordinary_saturated_y = self._paired_axis_rate(
                projected_y,
                projected_velocity_y,
                self.plant.gain_y_pixels_per_count,
                limit_y,
                ordinary_feedforward_confidence_y,
                feedback_limit=feedback_limit_y,
                feedback_held=self._paired_feedback_hold_y,
                position_confidence=position_confidence_y,
                pursuit_authorized=pursuit_authorized_y,
                pursuit_confidence=pursuit_confidence_y,
            )
            pursuit_reserve_rate_x = rate_x - ordinary_rate_x
            pursuit_reserve_rate_y = rate_y - ordinary_rate_y
            if physical_input_pending_x:
                # Stock passthrough firmware may serialize an injected report
                # over a simultaneous physical report.  Yield the axis for the
                # short command-to-capture interval instead of merely blocking
                # an opposite correction: this both prevents a software/hand
                # fight and gives subsequent physical reports an injection-free
                # USB frame in which to pass through.
                rate_x = 0.0
                pursuit_reserve_rate_x = 0.0
            if physical_input_pending_y:
                rate_y = 0.0
                pursuit_reserve_rate_y = 0.0
        else:
            rate_x, saturated_x = self._axis_rate(
                projected_x,
                projected_velocity_x,
                self.plant.gain_x_pixels_per_count,
                limit_x,
            )
            rate_y, saturated_y = self._axis_rate(
                projected_y,
                projected_velocity_y,
                self.plant.gain_y_pixels_per_count,
                limit_y,
            )
        if not math.isfinite(rate_x) or not math.isfinite(rate_y):
            self._reset_tracking()
            return self._zero(current_ns, "non-finite-control-output")
        return CalibratedControlOutput(
            timestamp_ns=current_ns,
            rate_x_counts_per_second=rate_x,
            rate_y_counts_per_second=rate_y,
            target_velocity_x_pixels_per_second=projected_velocity_x,
            target_velocity_y_pixels_per_second=projected_velocity_y,
            projected_error_x_pixels=projected_x,
            projected_error_y_pixels=projected_y,
            valid=True,
            saturated_x=saturated_x,
            saturated_y=saturated_y,
            observer_position_sigma_x_pixels=self._bounded_diagnostic(
                position_sigma_x
            ),
            observer_position_sigma_y_pixels=self._bounded_diagnostic(
                position_sigma_y
            ),
            observer_velocity_sigma_x_pixels_per_second=(
                self._bounded_diagnostic(velocity_sigma_x)
            ),
            observer_velocity_sigma_y_pixels_per_second=(
                self._bounded_diagnostic(velocity_sigma_y)
            ),
            velocity_feedforward_confidence_x=feedforward_confidence_x,
            velocity_feedforward_confidence_y=feedforward_confidence_y,
            position_channel_agreement=(
                min(
                    self._paired_measurement_agreement_x,
                    self._paired_measurement_agreement_y,
                )
                if paired_observer
                else 1.0
            ),
            position_feedback_confidence_x=position_confidence_x,
            position_feedback_confidence_y=position_confidence_y,
            position_feedback_held_x=(
                self._paired_feedback_hold_x if paired_observer else False
            ),
            position_feedback_held_y=(
                self._paired_feedback_hold_y if paired_observer else False
            ),
            innovation_mahalanobis_squared=self._bounded_diagnostic(
                self._last_innovation_mahalanobis_squared
            ),
            innovation_rejected=self._last_innovation_rejected,
            motion_corroboration_confidence=(
                active_motion_corroboration_confidence
            ),
            body_derived_motion_confidence_x=(
                self._body_derived_motion_confidence_x
            ),
            body_derived_motion_confidence_y=(
                self._body_derived_motion_confidence_y
            ),
            correlated_lookahead_active=(
                correlated_authority is not None
                and self._independent_pursuit_authorized
            ),
            lookahead_retained_authority_x=(
                correlated_authority.total_x
                if correlated_authority is not None
                else 0.0
            ),
            lookahead_retained_authority_y=(
                correlated_authority.total_y
                if correlated_authority is not None
                else 0.0
            ),
            ambiguous_lookahead_projection_retained_x=(
                correlated_authority.ambiguous_projection_retained_x
                if correlated_authority is not None
                else False
            ),
            ambiguous_lookahead_projection_retained_y=(
                correlated_authority.ambiguous_projection_retained_y
                if correlated_authority is not None
                else False
            ),
            pursuit_reserve_rate_x_counts_per_second=pursuit_reserve_rate_x,
            pursuit_reserve_rate_y_counts_per_second=pursuit_reserve_rate_y,
            residual_pursuit_authority_x=(
                self._residual_pursuit_x.authority
            ),
            residual_pursuit_authority_y=(
                self._residual_pursuit_y.authority
            ),
            material_motion_revoked_x=(
                self.config.require_motion_corroboration_for_feedforward
                and self._paired_material_stop_or_reversal_x
            ),
            material_motion_revoked_y=(
                self.config.require_motion_corroboration_for_feedforward
                and self._paired_material_stop_or_reversal_y
            ),
            predictive_authority_revoked_x=predictive_authority_revoked_x,
            predictive_authority_revoked_y=predictive_authority_revoked_y,
            physical_input_pending_x=physical_input_pending_x,
            physical_input_pending_y=physical_input_pending_y,
        )

    def _reset_tracking(self) -> None:
        self._last_observation = None
        self._identity_deadline_ns = None
        self._ready = False
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._raw_velocity_x.clear()
        self._raw_velocity_y.clear()
        self._paired_position_x = 0.0
        self._paired_position_y = 0.0
        self._paired_covariance_x = (0.0, 0.0, 0.0)
        self._paired_covariance_y = (0.0, 0.0, 0.0)
        self._paired_feedback_hold_x = False
        self._paired_feedback_hold_y = False
        self._paired_measurement_agreement_x = 1.0
        self._paired_measurement_agreement_y = 1.0
        self._paired_rejection_count = 0
        self._last_innovation_rejected = False
        self._last_innovation_mahalanobis_squared = 0.0
        self._corroboration_position_x = 0.0
        self._corroboration_position_y = 0.0
        self._corroboration_velocity_x = 0.0
        self._corroboration_velocity_y = 0.0
        self._corroboration_covariance_x = (0.0, 0.0, 0.0)
        self._corroboration_covariance_y = (0.0, 0.0, 0.0)
        self._last_corroboration_measurement_ns = None
        self._motion_corroboration_confidence = 0.0
        self._independent_pursuit_authorized = False
        self._body_derived_pursuit_authorized = False
        self._last_correlated_identity = None
        self._last_observation_was_correlated_lookahead = False
        self._correlated_lookahead_authority_state = None
        self._correlated_lookahead_authority_deadline_ns = None
        self._corroboration_direction_anchor_x = 0.0
        self._corroboration_direction_anchor_y = 0.0
        self._corroboration_direction_persistence_seconds = 0.0
        self._corroboration_reseed_required = False
        self._body_derived_motion_permitted = False
        self._body_derived_motion_deadline_ns = None
        self._body_derived_motion_confidence_x = 0.0
        self._body_derived_motion_confidence_y = 0.0
        self._body_derived_fresh_motion_confidence_x = 0.0
        self._body_derived_fresh_motion_confidence_y = 0.0
        self._paired_fresh_motion_confidence_x = 0.0
        self._paired_fresh_motion_confidence_y = 0.0
        self._paired_material_stop_or_reversal_x = False
        self._paired_material_stop_or_reversal_y = False
        self._adaptive_pursuit_low_fresh_samples_x = 0
        self._adaptive_pursuit_low_fresh_samples_y = 0
        self._common_pursuit_direction_anchor_x = 0.0
        self._common_pursuit_direction_anchor_y = 0.0
        self._common_pursuit_direction_persistence_seconds = 0.0
        self._body_derived_adaptive_pursuit_confidence_x = 0.0
        self._body_derived_adaptive_pursuit_confidence_y = 0.0
        self._clear_residual_pursuit()
        self._common_pursuit_handoff_deadline_ns = None
        self._body_derived_direction_anchor_x = 0.0
        self._body_derived_direction_anchor_y = 0.0
        self._body_derived_direction_persistence_seconds = 0.0
        self._body_derived_axis_direction_anchor_x = 0.0
        self._body_derived_axis_direction_anchor_y = 0.0
        self._body_derived_direction_persistence_x_seconds = 0.0
        self._body_derived_direction_persistence_y_seconds = 0.0

    def _zero(self, now_ns: int, reason: str) -> CalibratedControlOutput:
        return CalibratedControlOutput(
            timestamp_ns=now_ns,
            rate_x_counts_per_second=0.0,
            rate_y_counts_per_second=0.0,
            target_velocity_x_pixels_per_second=0.0,
            target_velocity_y_pixels_per_second=0.0,
            projected_error_x_pixels=0.0,
            projected_error_y_pixels=0.0,
            valid=False,
            reset_reason=reason,
            observer_position_sigma_x_pixels=self._bounded_diagnostic(
                math.sqrt(self._paired_covariance_x[0])
            ),
            observer_position_sigma_y_pixels=self._bounded_diagnostic(
                math.sqrt(self._paired_covariance_y[0])
            ),
            observer_velocity_sigma_x_pixels_per_second=(
                self._bounded_diagnostic(math.sqrt(self._paired_covariance_x[2]))
            ),
            observer_velocity_sigma_y_pixels_per_second=(
                self._bounded_diagnostic(math.sqrt(self._paired_covariance_y[2]))
            ),
            innovation_mahalanobis_squared=self._bounded_diagnostic(
                self._last_innovation_mahalanobis_squared
            ),
            innovation_rejected=self._last_innovation_rejected,
            motion_corroboration_confidence=(
                self._motion_corroboration_confidence
                if self._independent_pursuit_authorized
                else 0.0
            ),
            body_derived_motion_confidence_x=(
                self._body_derived_motion_confidence_x
            ),
            body_derived_motion_confidence_y=(
                self._body_derived_motion_confidence_y
            ),
        )

    def _ingest_commands(
        self,
        commands: Iterable[EmittedMouseCommand],
        now_ns: int,
    ) -> str | None:
        try:
            records = tuple(commands)
        except TypeError as exc:
            raise TypeError("emitted_commands must be iterable") from exc
        error = self._validate_commands(records, now_ns)
        if error is not None:
            return error
        self._commands.extend(records)
        if records:
            self._last_command_ns = records[-1].timestamp_ns
        return None

    def _validate_commands(
        self,
        records: tuple[EmittedMouseCommand, ...],
        now_ns: int,
    ) -> str | None:
        previous_ns = self._last_command_ns
        for command in records:
            if not isinstance(command, EmittedMouseCommand):
                raise TypeError(
                    "emitted_commands must contain EmittedMouseCommand records"
                )
            if command.timestamp_ns <= previous_ns:
                return "non-monotonic-command-history"
            if command.timestamp_ns > now_ns:
                return "future-command"
            previous_ns = command.timestamp_ns
        if (
            len(self._commands) + len(self._physical_inputs) + len(records)
            > self.config.maximum_command_history
        ):
            return "command-history-overflow"
        return None

    def _accept_correlated_lookahead(
        self,
        batch: CorrelatedLookaheadObservation,
    ) -> tuple[
        str | None,
        float,
        _CorrelatedLookaheadAuthority,
        bool,
        bool,
    ]:
        """Atomically ingest measured evidence and one newer position.

        The inferred root can be newer than the prior primary observation, or
        it can confirm a phase point which was already accepted at that exact
        timestamp on the preceding detector iteration.  The latter updates
        only the independently observed corroboration channel; accepting the
        head point twice would be a non-monotonic primary observation.
        """

        primary = batch.primary
        lookahead = batch.lookahead
        identity = (
            batch.runtime_identity_generation,
            batch.track_generation,
        )
        previous = self._last_observation
        prior_authority = self._correlated_lookahead_authority_state
        prior_identity = self._last_correlated_identity
        root_elapsed = (
            (primary.timestamp_ns - previous.timestamp_ns) / _NS_PER_SECOND
            if previous is not None
            and primary.timestamp_ns > previous.timestamp_ns
            else 0.0
        )
        if previous is not None and primary.timestamp_ns < previous.timestamp_ns:
            self.revoke_motion_corroboration()
            return (
                "stale-correlated-root",
                0.0,
                self._zero_correlated_lookahead_authority(),
                False,
                False,
            )

        # Both accepted points below prune the shared command ledger. Preserve
        # the complete previous-endpoint -> new-endpoint slice first so an
        # ambiguous one-sample projection carry can be checked against physical
        # target motion rather than raw screen displacement. The latter moves
        # opposite our own command even while a target continues coherently.
        batch_count_x = 0
        batch_count_y = 0
        endpoint_count_x, endpoint_count_y = self._counts_landing_between(
            primary.timestamp_ns,
            lookahead.timestamp_ns,
        )
        if previous is not None:
            batch_count_x, batch_count_y = self._counts_landing_between(
                previous.timestamp_ns,
                lookahead.timestamp_ns,
            )

        if previous is not None and primary.timestamp_ns == previous.timestamp_ns:
            if (
                not self._last_observation_was_correlated_lookahead
                or self._last_correlated_identity != identity
                or previous.velocity_error_x_pixels is None
            ):
                self.revoke_motion_corroboration()
            else:
                # The independent observer was already predict-advanced to
                # this timestamp when the prior position-only endpoint was
                # accepted.  Fuse only the newly inferred body coordinate;
                # no command slice or primary Kalman update is repeated.
                self._independent_pursuit_authorized = (
                    self._update_motion_corroboration(
                        primary,
                        elapsed=0.0,
                        landed_x_pixels=0.0,
                        landed_y_pixels=0.0,
                    )
                )
                self._body_derived_pursuit_authorized = False
                self._body_derived_motion_permitted = False
                self._body_derived_motion_deadline_ns = None
        else:
            # A discontinuity may reseed the primary observer.  The newer
            # endpoint can still provide the second physical confirmation for
            # static position, but no motion authority is retained unless the
            # root itself completed the independent proof below.
            self._accept_observation(primary)
            self._last_observation_was_correlated_lookahead = False

        verified_flow_motion = bool(
            batch.verified_flow_motion
            and self._verified_flow_endpoint_supports_velocity(
                primary,
                lookahead,
                endpoint_count_x=endpoint_count_x,
                endpoint_count_y=endpoint_count_y,
            )
        )
        verified_flow_fresh_motion_x = 0.0
        verified_flow_fresh_motion_y = 0.0
        if verified_flow_motion:
            endpoint_elapsed = (
                lookahead.timestamp_ns - primary.timestamp_ns
            ) / _NS_PER_SECOND
            assert primary.velocity_error_x_pixels is not None
            assert primary.velocity_error_y_pixels is not None
            assert lookahead.velocity_error_x_pixels is not None
            assert lookahead.velocity_error_y_pixels is not None
            verified_flow_fresh_motion_x = (
                self._body_derived_fresh_motion_confidence(
                    velocity=self._velocity_x,
                    measured_motion_delta=(
                        lookahead.velocity_error_x_pixels
                        - primary.velocity_error_x_pixels
                        + self.plant.gain_x_pixels_per_count * endpoint_count_x
                    ),
                    elapsed=endpoint_elapsed,
                )
            )
            verified_flow_fresh_motion_y = (
                self._body_derived_fresh_motion_confidence(
                    velocity=self._velocity_y,
                    measured_motion_delta=(
                        lookahead.velocity_error_y_pixels
                        - primary.velocity_error_y_pixels
                        + self.plant.gain_y_pixels_per_count * endpoint_count_y
                    ),
                    elapsed=endpoint_elapsed,
                )
            )
        authority = self._correlated_lookahead_authority(
            prior_authority=prior_authority,
            same_identity=prior_identity == identity,
            observation_elapsed=root_elapsed,
            verified_flow_motion=verified_flow_motion,
            verified_flow_fresh_motion_x=verified_flow_fresh_motion_x,
            verified_flow_fresh_motion_y=verified_flow_fresh_motion_y,
        )
        before_lookahead = self._last_observation
        direction_state = (
            self._common_pursuit_direction_anchor_x,
            self._common_pursuit_direction_anchor_y,
            self._common_pursuit_direction_persistence_seconds,
        )
        discontinuity = self._accept_observation(lookahead)
        if discontinuity is not None or self._last_observation is not lookahead:
            self._independent_pursuit_authorized = False
            self._last_correlated_identity = None
            self._last_observation_was_correlated_lookahead = False
            return (
                discontinuity,
                0.0,
                self._zero_correlated_lookahead_authority(),
                False,
                False,
            )

        # The lookahead's own head flow is fresh evidence for immediate
        # stop/reversal revocation, but it is not independent evidence and may
        # not extend the common-direction qualification window.
        (
            self._common_pursuit_direction_anchor_x,
            self._common_pursuit_direction_anchor_y,
            self._common_pursuit_direction_persistence_seconds,
        ) = direction_state
        ambiguous_direction_revoked_x = bool(
            authority.ambiguous_projection_retained_x
            and not self._command_compensated_displacement_supports_direction(
                (
                    previous.velocity_error_x_pixels
                    if previous is not None
                    else None
                ),
                lookahead.velocity_error_x_pixels,
                self.plant.gain_x_pixels_per_count * batch_count_x,
                authority.velocity_direction_x,
            )
        )
        ambiguous_direction_revoked_y = bool(
            authority.ambiguous_projection_retained_y
            and not self._command_compensated_displacement_supports_direction(
                (
                    previous.velocity_error_y_pixels
                    if previous is not None
                    else None
                ),
                lookahead.velocity_error_y_pixels,
                self.plant.gain_y_pixels_per_count * batch_count_y,
                authority.velocity_direction_y,
            )
        )
        revoke_x = bool(
            self._paired_material_stop_or_reversal_x
            or ambiguous_direction_revoked_x
        )
        revoke_y = bool(
            self._paired_material_stop_or_reversal_y
            or ambiguous_direction_revoked_y
        )
        retained_authority = _CorrelatedLookaheadAuthority(
            authority.authorized,
            (
                0.0
                if revoke_x
                else authority.projection_x
            ),
            (
                0.0
                if revoke_y
                else authority.projection_y
            ),
            (
                0.0
                if revoke_x
                else authority.ordinary_x
            ),
            (
                0.0
                if revoke_y
                else authority.ordinary_y
            ),
            (
                0.0
                if revoke_x
                else authority.total_x
            ),
            (
                0.0
                if revoke_y
                else authority.total_y
            ),
            authority.velocity_direction_x,
            authority.velocity_direction_y,
            (
                authority.ambiguous_projection_retained_x
                and not revoke_x
            ),
            (
                authority.ambiguous_projection_retained_y
                and not revoke_y
            ),
        )
        self._independent_pursuit_authorized = bool(
            retained_authority.authorized
            and not self._last_innovation_rejected
            and not (
                self._paired_material_stop_or_reversal_x
                and self._paired_material_stop_or_reversal_y
            )
        )
        if not self._independent_pursuit_authorized:
            retained_authority = self._zero_correlated_lookahead_authority()
        # This is deliberately narrower than the caller's verified-flow flag.
        # Residual pursuit may treat P1 as fresh measured evidence without the
        # independent/body grant only after the numeric core has recovered
        # command-compensated motion along the current velocity and the
        # accepted P1 update still supports that axis.  The booleans are local
        # evidence for this control step; they do not extend identity or the
        # correlated P0 authority lease.
        verified_residual_measurement_x = bool(
            verified_flow_motion
            and verified_flow_fresh_motion_x > 0.0
            and not self._paired_material_stop_or_reversal_x
        )
        verified_residual_measurement_y = bool(
            verified_flow_motion
            and verified_flow_fresh_motion_y > 0.0
            and not self._paired_material_stop_or_reversal_y
        )
        self._last_correlated_identity = identity
        self._last_observation_was_correlated_lookahead = True
        elapsed = (
            (lookahead.timestamp_ns - before_lookahead.timestamp_ns)
            / _NS_PER_SECOND
            if before_lookahead is not None
            else 0.0
        )
        return (
            None,
            elapsed,
            retained_authority,
            verified_residual_measurement_x,
            verified_residual_measurement_y,
        )

    @staticmethod
    def _zero_correlated_lookahead_authority(
    ) -> _CorrelatedLookaheadAuthority:
        return _CorrelatedLookaheadAuthority(
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0,
        )

    @staticmethod
    def _command_compensated_displacement_supports_direction(
        previous_error: float | None,
        current_error: float | None,
        landed_command_pixels: float,
        retained_direction: int,
    ) -> bool:
        """Whether whole-batch physical motion supports a frozen direction.

        Error on screen decreases when our command lands, so adding that known
        displacement recovers target motion. A strict opposite sign revokes
        the one-sample bridge; exact zero remains admissible because it is
        absence of reversal evidence, not evidence of a reversal.
        """

        if (
            previous_error is None
            or current_error is None
            or retained_direction not in (-1, 1)
        ):
            return False
        physical_displacement = (
            current_error - previous_error + landed_command_pixels
        )
        return bool(
            math.isfinite(physical_displacement)
            and physical_displacement * retained_direction >= 0.0
        )

    def _verified_flow_endpoint_supports_velocity(
        self,
        primary: ScreenErrorObservation,
        lookahead: ScreenErrorObservation,
        *,
        endpoint_count_x: int,
        endpoint_count_y: int,
    ) -> bool:
        """Require newest physical pixels to keep moving with the root.

        Consecutive LK proves that the endpoint came from image motion, but it
        does not prove that a previously learned high velocity is still
        current. Recover target displacement by adding back commands which
        land between root and endpoint, then require a modest amount of motion
        along the independently estimated velocity vector. A stop, reversal,
        sharp turn, or manual approach therefore withdraws only the extra
        verified-flow reserve on the next endpoint.
        """

        elapsed = (
            lookahead.timestamp_ns - primary.timestamp_ns
        ) / _NS_PER_SECOND
        velocity_x = self._velocity_x
        velocity_y = self._velocity_y
        speed = math.hypot(velocity_x, velocity_y)
        if (
            elapsed <= 0.0
            or speed
            < _BODY_DERIVED_ADAPTIVE_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND
            or primary.velocity_error_x_pixels is None
            or primary.velocity_error_y_pixels is None
            or lookahead.velocity_error_x_pixels is None
            or lookahead.velocity_error_y_pixels is None
        ):
            return False
        physical_dx = (
            lookahead.velocity_error_x_pixels
            - primary.velocity_error_x_pixels
            + self.plant.gain_x_pixels_per_count * endpoint_count_x
        )
        physical_dy = (
            lookahead.velocity_error_y_pixels
            - primary.velocity_error_y_pixels
            + self.plant.gain_y_pixels_per_count * endpoint_count_y
        )
        along_displacement = (
            physical_dx * velocity_x + physical_dy * velocity_y
        ) / speed
        minimum_along_displacement = max(
            _VERIFIED_FLOW_MINIMUM_ALONG_DISPLACEMENT_PIXELS,
            speed
            * elapsed
            * _VERIFIED_FLOW_MINIMUM_EXPECTED_DISPLACEMENT_FRACTION,
        )
        return bool(
            math.isfinite(along_displacement)
            and along_displacement >= minimum_along_displacement
        )

    def _correlated_lookahead_authority(
        self,
        *,
        prior_authority: _CorrelatedLookaheadAuthority | None,
        same_identity: bool,
        observation_elapsed: float,
        verified_flow_motion: bool = False,
        verified_flow_fresh_motion_x: float = 0.0,
        verified_flow_fresh_motion_y: float = 0.0,
    ) -> _CorrelatedLookaheadAuthority:
        """Freeze root authority and admit narrowly verified fast flow.

        Ordinary lookahead remains position-only.  A caller-verified endpoint
        is different evidence: the runtime has already observed a consecutive
        flow streak and :meth:`_verified_flow_endpoint_supports_velocity` has
        recovered same-direction target motion after both generated and raw
        physical mouse inputs.  It may therefore promote the existing root
        velocity promptly, without waiting for the sparse corroboration EMA or
        adaptive pursuit integrator to mature.  The promotion is still bounded
        by the configured verified-flow ceiling and independently gated on
        each axis by fresh root and endpoint motion, stop/reversal state, and
        the same closed-loop residual used by the learned reserve.
        """

        if not isinstance(verified_flow_motion, bool):
            raise TypeError("verified_flow_motion must be bool")
        for name, value in (
            ("verified_flow_fresh_motion_x", verified_flow_fresh_motion_x),
            ("verified_flow_fresh_motion_y", verified_flow_fresh_motion_y),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        if not self._independent_pursuit_authorized:
            return self._zero_correlated_lookahead_authority()
        confidence = min(max(self._motion_corroboration_confidence, 0.0), 1.0)
        projection = min(
            confidence / _CORROBORATION_FULL_MOTION_PROJECTION_CONFIDENCE,
            1.0,
        )
        ordinary_ceiling = min(
            self.config.maximum_body_derived_feedforward_fraction,
            self.config.maximum_velocity_feedforward_fraction,
        )
        ordinary_x = min(
            self._paired_velocity_confidence(
                self._velocity_x,
                math.sqrt(self._paired_covariance_x[2]),
            )
            * self._paired_measurement_agreement_x
            * confidence,
            ordinary_ceiling,
        )
        ordinary_y = min(
            self._paired_velocity_confidence(
                self._velocity_y,
                math.sqrt(self._paired_covariance_y[2]),
            )
            * self._paired_measurement_agreement_y
            * confidence,
            ordinary_ceiling,
        )
        projection_x = projection
        projection_y = projection
        direction_x = (
            1 if self._velocity_x > 0.0 else -1 if self._velocity_x < 0.0 else 0
        )
        direction_y = (
            1 if self._velocity_y > 0.0 else -1 if self._velocity_y < 0.0 else 0
        )

        def retain_ambiguous_projection(
            *,
            axis: str,
            fresh_motion_confidence: float,
            material_stop_or_reversal: bool,
            direction: int,
        ) -> float:
            if (
                not self.config.retain_ambiguous_correlated_projection
                or material_stop_or_reversal
                or fresh_motion_confidence > 0.0
                or prior_authority is None
                or not prior_authority.authorized
                or not same_identity
                or direction == 0
                or not 0.0 < observation_elapsed
                <= _COMMON_PAIRED_PURSUIT_AMBIGUOUS_FRESH_MAX_SECONDS
            ):
                return 0.0
            prior_direction = getattr(
                prior_authority,
                f"velocity_direction_{axis}",
            )
            prior_projection = getattr(prior_authority, f"projection_{axis}")
            low_samples = getattr(
                self,
                f"_adaptive_pursuit_low_fresh_samples_{axis}",
            )
            if (
                low_samples != 1
                or prior_direction != direction
                or prior_projection <= 0.0
            ):
                return 0.0
            # This is a frozen ceiling, never newly earned authority.  The
            # independently accepted root can reduce the prior projection but
            # cannot increase it during the ambiguous sample.
            return min(projection, prior_projection)

        ambiguous_projection_x = 0.0
        ambiguous_projection_y = 0.0
        if self._paired_material_stop_or_reversal_x:
            projection_x = 0.0
            ordinary_x = 0.0
        elif self._paired_fresh_motion_confidence_x <= 0.0:
            ambiguous_projection_x = retain_ambiguous_projection(
                axis="x",
                fresh_motion_confidence=self._paired_fresh_motion_confidence_x,
                material_stop_or_reversal=False,
                direction=direction_x,
            )
            projection_x = ambiguous_projection_x
            # Position projection is closed-loop; explicit velocity output is
            # still withdrawn on the ambiguous sample.
            ordinary_x = 0.0
        if self._paired_material_stop_or_reversal_y:
            projection_y = 0.0
            ordinary_y = 0.0
        elif self._paired_fresh_motion_confidence_y <= 0.0:
            ambiguous_projection_y = retain_ambiguous_projection(
                axis="y",
                fresh_motion_confidence=self._paired_fresh_motion_confidence_y,
                material_stop_or_reversal=False,
                direction=direction_y,
            )
            projection_y = ambiguous_projection_y
            ordinary_y = 0.0

        # The first live validation proved that P1's fresh coordinate is safe,
        # but also showed the ordinary 50% ceiling binding throughout coherent
        # 250--1500 px/s pursuit. Carry only a small, already-earned portion of
        # the adaptive reserve. P0 must independently re-authorize it after a
        # full common-direction horizon; the speed ramp, current residual, and
        # per-axis fresh-motion/reversal gates can only reduce that prior state.
        # P1 remains position-only and cannot expose reserve which was first
        # learned at P1 until a following independently corroborated P0.
        ordinary_retained_maximum = max(
            ordinary_ceiling,
            min(
                self.config.maximum_correlated_lookahead_pursuit_feedforward_fraction,
                self.config.maximum_body_derived_pursuit_feedforward_fraction,
                self.config.maximum_velocity_feedforward_fraction,
            ),
        )
        verified_retention = bool(
            verified_flow_motion
            and self.config.maximum_verified_flow_pursuit_feedforward_fraction
            > 0.0
        )
        if verified_retention:
            retained_maximum = min(
                self.config.maximum_verified_flow_pursuit_feedforward_fraction,
                self.config.maximum_velocity_feedforward_fraction,
            )
        else:
            retained_maximum = min(
                self.config.maximum_correlated_lookahead_pursuit_feedforward_fraction,
                self.config.maximum_body_derived_pursuit_feedforward_fraction,
                self.config.maximum_velocity_feedforward_fraction,
            )
        latest = self._last_observation
        direction_qualified = bool(
            self._common_pursuit_direction_persistence_seconds
            >= _CORROBORATION_FULL_DIRECTION_PERSISTENCE_SECONDS
        )
        verified_vector_speed = math.hypot(self._velocity_x, self._velocity_y)
        verified_prompt_enabled = bool(
            verified_flow_motion
            and self.config.maximum_verified_flow_pursuit_feedforward_fraction
            > ordinary_retained_maximum
            and verified_vector_speed
            > _BODY_DERIVED_ADAPTIVE_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND
        )

        def retained_total(
            ordinary: float,
            projection_axis: float,
            velocity: float,
            measured_error: float,
            projected_error: float,
            adaptive: float,
            *,
            root_fresh_motion: float,
            endpoint_fresh_motion: float,
            material_stop_or_reversal: bool,
        ) -> tuple[float, bool]:
            verified_axis_motion = bool(
                verified_prompt_enabled
                and root_fresh_motion > 0.0
                and endpoint_fresh_motion > 0.0
                and not material_stop_or_reversal
                and abs(velocity) > 1e-9
            )
            axis_retained_maximum = (
                retained_maximum
                if verified_axis_motion
                else ordinary_retained_maximum
            )
            if latest is None:
                return ordinary, False
            direction = 1.0 if velocity > 0.0 else -1.0
            least_aligned_error = min(
                measured_error * direction,
                projected_error * direction,
            )
            residual_gate = self._ramp(
                least_aligned_error,
                _BODY_DERIVED_PURSUIT_ZERO_OPPOSED_ERROR_PIXELS,
                _BODY_DERIVED_PURSUIT_FULL_OPPOSED_ERROR_PIXELS,
            )
            retained = ordinary
            if (
                projection_axis > 0.0
                and direction_qualified
                and axis_retained_maximum > ordinary_ceiling
                and adaptive > ordinary
            ):
                retained = adaptive * residual_gate
            speed_gate = self._ramp(
                abs(velocity),
                _COMMON_PAIRED_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND,
                _COMMON_PAIRED_PURSUIT_FULL_SPEED_PIXELS_PER_SECOND,
            )
            ordinary_carry_ceiling = ordinary_ceiling + speed_gate * (
                ordinary_retained_maximum - ordinary_ceiling
            )
            if axis_retained_maximum > ordinary_retained_maximum:
                # The extra measured-flow reserve is reserved for genuinely
                # fast pursuit.  At ordinary/stationary speeds the exact same
                # 50--60% ceiling remains in force, so qualifying LK cannot
                # amplify a low-speed reticle or detector wobble.  The ramp is
                # the already validated vector-speed adaptive-pursuit band
                # used by the command-aware observer; a coherent diagonal
                # target must not be treated as two artificially slow axes.
                verified_speed_gate = self._ramp(
                    verified_vector_speed,
                    _BODY_DERIVED_ADAPTIVE_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND,
                    _BODY_DERIVED_ADAPTIVE_PURSUIT_FULL_SPEED_PIXELS_PER_SECOND,
                )
                carry_ceiling = ordinary_carry_ceiling + verified_speed_gate * (
                    axis_retained_maximum - ordinary_retained_maximum
                )
            else:
                carry_ceiling = ordinary_carry_ceiling
            total = min(max(ordinary, retained), carry_ceiling)
            if not verified_axis_motion or residual_gate <= 0.0:
                return total, False

            # Verified newest-frame motion is an authorization decision, not a
            # second amplitude multiplier.  Promote directly to its bounded
            # speed-ramped ceiling, while the residual ramp withdraws only the
            # extra grant as the reticle moves ahead of the target.
            promoted = ordinary + residual_gate * (
                carry_ceiling - ordinary
            )
            return max(total, promoted), promoted > ordinary

        total_x, verified_promoted_x = retained_total(
            ordinary_x,
            projection_x,
            self._velocity_x,
            latest.error_x_pixels if latest is not None else 0.0,
            self._paired_position_x,
            self._body_derived_adaptive_pursuit_confidence_x,
            root_fresh_motion=self._paired_fresh_motion_confidence_x,
            endpoint_fresh_motion=verified_flow_fresh_motion_x,
            material_stop_or_reversal=(
                self._paired_material_stop_or_reversal_x
            ),
        )
        total_y, verified_promoted_y = retained_total(
            ordinary_y,
            projection_y,
            self._velocity_y,
            latest.error_y_pixels if latest is not None else 0.0,
            self._paired_position_y,
            self._body_derived_adaptive_pursuit_confidence_y,
            root_fresh_motion=self._paired_fresh_motion_confidence_y,
            endpoint_fresh_motion=verified_flow_fresh_motion_y,
            material_stop_or_reversal=(
                self._paired_material_stop_or_reversal_y
            ),
        )
        if verified_promoted_x:
            self._body_derived_adaptive_pursuit_confidence_x = max(
                self._body_derived_adaptive_pursuit_confidence_x,
                total_x,
            )
        if verified_promoted_y:
            self._body_derived_adaptive_pursuit_confidence_y = max(
                self._body_derived_adaptive_pursuit_confidence_y,
                total_y,
            )
        if ambiguous_projection_x > 0.0:
            total_x = 0.0
        if ambiguous_projection_y > 0.0:
            total_y = 0.0
        return _CorrelatedLookaheadAuthority(
            True,
            projection_x,
            projection_y,
            ordinary_x,
            ordinary_y,
            total_x,
            total_y,
            direction_x,
            direction_y,
            ambiguous_projection_x > 0.0,
            ambiguous_projection_y > 0.0,
        )

    def _accept_observation(self, observation: ScreenErrorObservation) -> str | None:
        # Predictive authority belongs to accepted physical evidence. Clear
        # the former sample before evaluating a new one so a prediction,
        # discontinuity, or innovation rejection fails closed immediately.
        self._body_derived_motion_permitted = False
        self._body_derived_motion_deadline_ns = None
        self._body_derived_motion_confidence_x = 0.0
        self._body_derived_motion_confidence_y = 0.0
        self._body_derived_fresh_motion_confidence_x = 0.0
        self._body_derived_fresh_motion_confidence_y = 0.0
        self._paired_fresh_motion_confidence_x = 0.0
        self._paired_fresh_motion_confidence_y = 0.0
        self._paired_material_stop_or_reversal_x = False
        self._paired_material_stop_or_reversal_y = False
        self._independent_pursuit_authorized = False
        self._body_derived_pursuit_authorized = False
        if not observation.body_derived_motion_permitted:
            # The per-sample body grant is withdrawn immediately, but the
            # adaptive pursuit estimate belongs to the continuous measured-head
            # motion hypothesis.  It is updated below from the newly selected
            # provenance and clears on contrary physical evidence.
            self._clear_body_derived_motion(clear_adaptive=False)
        previous = self._last_observation
        if previous is None:
            self._seed_observation(observation)
            return "awaiting-confirmation"
        if observation.timestamp_ns <= previous.timestamp_ns:
            self._reset_tracking()
            return "non-monotonic-observation"

        elapsed_ns = observation.timestamp_ns - previous.timestamp_ns
        elapsed = elapsed_ns / _NS_PER_SECOND
        if elapsed > self.config.maximum_observation_interval_seconds:
            self._seed_observation(observation)
            return "observation-gap"

        paired_velocity = observation.velocity_error_x_pixels is not None
        previous_paired_velocity = previous.velocity_error_x_pixels is not None
        if paired_velocity != previous_paired_velocity:
            # Never splice a differentiated raw point onto a smoothed-point
            # history (or vice versa). The next continuous pair can confirm a
            # fresh velocity estimate without manufacturing a jump here.
            self._seed_observation(observation)
            return "velocity-channel-change"

        corroborated = observation.corroboration_error_x_pixels is not None
        previous_corroborated = (
            previous.corroboration_error_x_pixels is not None
        )
        if corroborated != previous_corroborated:
            # Corroboration is optional authority for predictive motion; it is
            # not a different position-measurement schema.  The live direct-
            # head path can briefly fall back from pixel flow to a body-carried
            # point and then recover on the next image.  Reseeding the *primary*
            # paired observer here discarded a still-valid head position and
            # velocity, producing a zero-rate frame plus another confirmation
            # frame on every such transition.
            #
            # Withdraw the optional grant on this sample, but keep its sparse
            # observer alive.  The live 45 Hz independent reference is normally
            # interleaved with roughly 90 Hz body-carried primary publications;
            # treating each expected absent frame as a schema break made every
            # returning reference a first sample forever.  The observer below
            # is predict-only while the reference is absent, including all
            # landed commands, and therefore still validates the next real
            # reference without authorizing this frame.
            self._independent_pursuit_authorized = False
            if (
                previous_corroborated
                and not corroborated
                and observation.body_derived_motion_permitted
                and max(
                    self._body_derived_adaptive_pursuit_confidence_x,
                    self._body_derived_adaptive_pursuit_confidence_y,
                )
                > 0.0
            ):
                self._common_pursuit_handoff_deadline_ns = (
                    observation.timestamp_ns
                    + round(
                        _COMMON_PAIRED_PURSUIT_HANDOFF_SECONDS * _NS_PER_SECOND
                    )
                )

        if paired_velocity:
            return self._accept_paired_observation(
                observation,
                previous,
                elapsed,
            )

        self._last_innovation_rejected = False
        self._last_innovation_mahalanobis_squared = 0.0
        count_x, count_y = self._counts_landing_between(
            previous.timestamp_ns,
            observation.timestamp_ns,
        )
        target_delta_x = (
            observation.error_x_pixels
            - previous.error_x_pixels
            + self.plant.gain_x_pixels_per_count * count_x
        )
        target_delta_y = (
            observation.error_y_pixels
            - previous.error_y_pixels
            + self.plant.gain_y_pixels_per_count * count_y
        )
        jump = self.config.maximum_error_jump_pixels
        if abs(target_delta_x) > jump or abs(target_delta_y) > jump:
            self._seed_observation(observation)
            return "error-jump"

        speed_limit = self.config.maximum_target_speed_pixels_per_second
        # Preserve the original independent adjacent-frame estimator for
        # every existing caller which has no separate raw point channel.
        raw_x = min(max(target_delta_x / elapsed, -speed_limit), speed_limit)
        raw_y = min(max(target_delta_y / elapsed, -speed_limit), speed_limit)
        self._raw_velocity_x.append(raw_x)
        self._raw_velocity_y.append(raw_y)
        robust_x = float(statistics.median(self._raw_velocity_x))
        robust_y = float(statistics.median(self._raw_velocity_y))
        alpha = 1.0 - math.exp(
            -elapsed / self.config.velocity_filter_time_constant_seconds
        )
        acceleration_step = (
            self.config.maximum_target_acceleration_pixels_per_second_squared
            * elapsed
        )
        self._velocity_x = self._filtered_velocity(
            self._velocity_x,
            robust_x,
            alpha,
            acceleration_step,
            speed_limit,
        )
        self._velocity_y = self._filtered_velocity(
            self._velocity_y,
            robust_y,
            alpha,
            acceleration_step,
            speed_limit,
        )
        self._last_observation = observation
        self._identity_deadline_ns = observation.identity_deadline_ns
        self._ready = True
        self._prune_commands(observation.timestamp_ns)
        return None

    def _accept_paired_observation(
        self,
        observation: ScreenErrorObservation,
        previous: ScreenErrorObservation,
        elapsed: float,
    ) -> str | None:
        """Update the one raw-point position/velocity observer.

        The state transition includes every command which became visible
        between the two image timestamps.  Thus the Kalman innovation is
        target/detector motion, rather than a derivative contaminated by the
        controller's own delayed movement.
        """

        measured_x = observation.velocity_error_x_pixels
        measured_y = observation.velocity_error_y_pixels
        assert measured_x is not None
        assert measured_y is not None
        prior_velocity_x = self._velocity_x
        prior_velocity_y = self._velocity_y
        # The downstream tracker point is not differentiated and never owns a
        # second control state. It is only an independent plausibility signal:
        # a one-model-pixel box-edge staircase which moves the raw point while
        # the tracked position stays fixed must not earn feed-forward trust.
        tracked_velocity_x = 0.0
        tracked_velocity_y = 0.0
        if observation.body_derived_motion_permitted:
            # This is deliberately the observed tracked-position step with no
            # landed-command compensation. A raw velocity-point staircase
            # while the causal tracked point is fixed must earn no allowance.
            tracked_velocity_x = (
                observation.error_x_pixels - previous.error_x_pixels
            ) / elapsed
            tracked_velocity_y = (
                observation.error_y_pixels - previous.error_y_pixels
            ) / elapsed
        measurement_agreement = self._paired_channel_agreement(
            measured_x,
            measured_y,
            observation.error_x_pixels,
            observation.error_y_pixels,
            tracked_velocity_x=tracked_velocity_x,
            tracked_velocity_y=tracked_velocity_y,
        )
        disagreement_x = measured_x - observation.error_x_pixels
        disagreement_y = measured_y - observation.error_y_pixels
        count_x, count_y = self._counts_landing_between(
            previous.timestamp_ns,
            observation.timestamp_ns,
        )
        previous_measured_x = previous.velocity_error_x_pixels
        previous_measured_y = previous.velocity_error_y_pixels
        assert previous_measured_x is not None
        assert previous_measured_y is not None
        measured_motion_delta_x = (
            measured_x
            - previous_measured_x
            + self.plant.gain_x_pixels_per_count * count_x
        )
        measured_motion_delta_y = (
            measured_y
            - previous_measured_y
            + self.plant.gain_y_pixels_per_count * count_y
        )
        predicted_x, predicted_velocity_x, predicted_covariance_x = (
            self._paired_predict_axis(
                self._paired_position_x,
                self._velocity_x,
                self._paired_covariance_x,
                elapsed,
                self.plant.gain_x_pixels_per_count * count_x,
            )
        )
        predicted_y, predicted_velocity_y, predicted_covariance_y = (
            self._paired_predict_axis(
                self._paired_position_y,
                self._velocity_y,
                self._paired_covariance_y,
                elapsed,
                self.plant.gain_y_pixels_per_count * count_y,
            )
        )
        # The accepted raw box and tracker point are correlated, so the latter
        # is not fused as a second measurement.  Their residual is still a
        # useful per-frame noise estimate: grow R instead of giving a visibly
        # quantized box edge the same trust as two agreeing coordinates.
        base_measurement_variance = _PAIRED_MEASUREMENT_SIGMA_PIXELS**2
        measurement_variance_x = (
            base_measurement_variance
            + _PAIRED_TRACK_RESIDUAL_VARIANCE_FRACTION
            * disagreement_x
            * disagreement_x
        )
        measurement_variance_y = (
            base_measurement_variance
            + _PAIRED_TRACK_RESIDUAL_VARIANCE_FRACTION
            * disagreement_y
            * disagreement_y
        )
        if observation.body_derived_motion_permitted:
            # Preserve the measured 46 Hz information rate when the mapped
            # body geometry is published at a faster primary-detector cadence.
            # This path is optional prediction provenance only; ordinary paired
            # and independently corroborated callers retain their exact R.
            reference_interval = (
                1.0 / _BODY_DERIVED_MEASUREMENT_REFERENCE_HZ
            )
            variance_scale = min(
                max(reference_interval / elapsed, 1.0),
                _BODY_DERIVED_MAXIMUM_MEASUREMENT_VARIANCE_SCALE,
            )
            measurement_variance_x *= variance_scale
            measurement_variance_y *= variance_scale
        innovation_x = measured_x - predicted_x
        innovation_y = measured_y - predicted_y
        innovation_variance_x = predicted_covariance_x[0] + measurement_variance_x
        innovation_variance_y = predicted_covariance_y[0] + measurement_variance_y
        mahalanobis_squared = (
            innovation_x * innovation_x / innovation_variance_x
            + innovation_y * innovation_y / innovation_variance_y
        )
        if not math.isfinite(mahalanobis_squared):
            self._reset_tracking()
            return "non-finite-observer"
        self._last_innovation_mahalanobis_squared = self._bounded_diagnostic(
            mahalanobis_squared
        )
        if mahalanobis_squared > _PAIRED_INNOVATION_GATE_MAHALANOBIS_SQUARED:
            self._last_innovation_rejected = True
            self._paired_rejection_count += 1
            # A rejected head coordinate cannot renew independent permission
            # for open-loop motion.  Position may retain its ordinary short
            # bridge, but predictive feed-forward fails closed immediately.
            self._motion_corroboration_confidence = 0.0
            self._corroboration_direction_persistence_seconds = 0.0
            self._clear_body_derived_motion()
            if self._paired_rejection_count >= _PAIRED_REJECTIONS_BEFORE_RESEED:
                self._seed_observation(observation)
                return "innovation-reacquired"
            # Keep the last accepted timestamp and command ledger. The next
            # accepted update will predict across the entire interval exactly
            # once. Meanwhile the existing state may bridge only its ordinary
            # freshness lease.
            return None

        self._last_innovation_rejected = False
        # Commit auxiliary-channel confidence only with an accepted observer
        # transaction. A rejected raw/motion point is still allowed to revoke
        # every predictive grant above, but it cannot poison the safe position
        # bridge which deliberately remains on the last accepted state.
        self._paired_measurement_agreement_x = measurement_agreement
        self._paired_measurement_agreement_y = measurement_agreement
        self._paired_position_x, self._velocity_x, self._paired_covariance_x = (
            self._paired_update_axis(
                predicted_x,
                predicted_velocity_x,
                predicted_covariance_x,
                measured_x,
                measurement_variance_x,
            )
        )
        self._paired_position_y, self._velocity_y, self._paired_covariance_y = (
            self._paired_update_axis(
                predicted_y,
                predicted_velocity_y,
                predicted_covariance_y,
                measured_y,
                measurement_variance_y,
            )
        )
        speed_limit = self.config.maximum_target_speed_pixels_per_second
        self._velocity_x = min(max(self._velocity_x, -speed_limit), speed_limit)
        self._velocity_y = min(max(self._velocity_y, -speed_limit), speed_limit)
        self._paired_fresh_motion_confidence_x = (
            self._body_derived_fresh_motion_confidence(
                velocity=self._velocity_x,
                measured_motion_delta=measured_motion_delta_x,
                elapsed=elapsed,
            )
        )
        self._paired_fresh_motion_confidence_y = (
            self._body_derived_fresh_motion_confidence(
                velocity=self._velocity_y,
                measured_motion_delta=measured_motion_delta_y,
                elapsed=elapsed,
            )
        )
        self._paired_material_stop_or_reversal_x = (
            self._material_stop_or_reversal(
                prior_velocity=prior_velocity_x,
                measured_motion_delta=measured_motion_delta_x,
                motion_innovation=innovation_x,
                motion_innovation_sigma=math.sqrt(innovation_variance_x),
            )
        )
        self._paired_material_stop_or_reversal_y = (
            self._material_stop_or_reversal(
                prior_velocity=prior_velocity_y,
                measured_motion_delta=measured_motion_delta_y,
                motion_innovation=innovation_y,
                motion_innovation_sigma=math.sqrt(innovation_variance_y),
            )
        )
        if (
            self._paired_fresh_motion_confidence_x > 0.0
            or self._paired_material_stop_or_reversal_x
        ):
            self._adaptive_pursuit_low_fresh_samples_x = 0
        else:
            self._adaptive_pursuit_low_fresh_samples_x += 1
        if (
            self._paired_fresh_motion_confidence_y > 0.0
            or self._paired_material_stop_or_reversal_y
        ):
            self._adaptive_pursuit_low_fresh_samples_y = 0
        else:
            self._adaptive_pursuit_low_fresh_samples_y += 1
        self._update_common_pursuit_direction_persistence(elapsed=elapsed)
        self._independent_pursuit_authorized = (
            self._update_motion_corroboration(
                observation,
                elapsed=elapsed,
                landed_x_pixels=self.plant.gain_x_pixels_per_count * count_x,
                landed_y_pixels=self.plant.gain_y_pixels_per_count * count_y,
            )
        )
        self._body_derived_motion_permitted = (
            observation.body_derived_motion_permitted
        )
        if self._body_derived_motion_permitted:
            self._body_derived_pursuit_authorized = True
            deadline_ns = observation.body_derived_motion_deadline_ns
            assert deadline_ns is not None
            self._body_derived_motion_deadline_ns = deadline_ns
            self._update_body_derived_motion_confidence(
                elapsed,
                measured_motion_delta_x=measured_motion_delta_x,
                measured_motion_delta_y=measured_motion_delta_y,
                motion_innovation_x=innovation_x,
                motion_innovation_y=innovation_y,
                motion_innovation_sigma_x=math.sqrt(innovation_variance_x),
                motion_innovation_sigma_y=math.sqrt(innovation_variance_y),
                prior_velocity_x=prior_velocity_x,
                prior_velocity_y=prior_velocity_y,
            )
        self._paired_rejection_count = 0
        self._last_observation = observation
        self._identity_deadline_ns = observation.identity_deadline_ns
        self._ready = True
        self._prune_commands(observation.timestamp_ns)
        return None

    def _seed_observation(self, observation: ScreenErrorObservation) -> None:
        self._reset_tracking()
        self._last_observation = observation
        self._identity_deadline_ns = observation.identity_deadline_ns
        if observation.velocity_error_x_pixels is not None:
            assert observation.velocity_error_y_pixels is not None
            self._paired_position_x = observation.velocity_error_x_pixels
            self._paired_position_y = observation.velocity_error_y_pixels
            measurement_agreement = self._paired_channel_agreement(
                observation.velocity_error_x_pixels,
                observation.velocity_error_y_pixels,
                observation.error_x_pixels,
                observation.error_y_pixels,
            )
            self._paired_measurement_agreement_x = measurement_agreement
            self._paired_measurement_agreement_y = measurement_agreement
            measurement_variance = _PAIRED_MEASUREMENT_SIGMA_PIXELS**2
            initial_velocity_sigma = min(
                self.config.maximum_target_speed_pixels_per_second,
                _PAIRED_INITIAL_VELOCITY_SIGMA_PIXELS_PER_SECOND,
            )
            initial_covariance = (
                measurement_variance,
                0.0,
                initial_velocity_sigma * initial_velocity_sigma,
            )
            self._paired_covariance_x = initial_covariance
            self._paired_covariance_y = initial_covariance
            if observation.corroboration_error_x_pixels is not None:
                assert observation.corroboration_error_y_pixels is not None
                self._seed_corroboration(
                    observation.corroboration_error_x_pixels,
                    observation.corroboration_error_y_pixels,
                    timestamp_ns=observation.timestamp_ns,
                )
        self._prune_commands(observation.timestamp_ns)

    def _update_motion_corroboration(
        self,
        observation: ScreenErrorObservation,
        *,
        elapsed: float,
        landed_x_pixels: float,
        landed_y_pixels: float,
    ) -> bool:
        """Update independent translation evidence without moving the aim point."""

        measured_x = observation.corroboration_error_x_pixels
        measured_y = observation.corroboration_error_y_pixels
        last_measurement_ns = self._last_corroboration_measurement_ns
        if (
            measured_x is not None
            and measured_y is not None
            and (
                self._corroboration_reseed_required
                or last_measurement_ns is None
            )
        ):
            self._seed_corroboration(
                measured_x,
                measured_y,
                timestamp_ns=observation.timestamp_ns,
            )
            self._motion_corroboration_confidence = 0.0
            return False
        if last_measurement_ns is None:
            # No independent physical point has seeded this observer yet.
            return False

        # Advance the independent state across every accepted primary interval,
        # including an interval with no independent measurement.  This consumes
        # the same landed-command slice before the shared ledger is pruned, so a
        # later sparse measurement cannot mistake our own intervening motion for
        # target velocity.
        predicted_x, predicted_velocity_x, predicted_covariance_x = (
            self._paired_predict_axis(
                self._corroboration_position_x,
                self._corroboration_velocity_x,
                self._corroboration_covariance_x,
                elapsed,
                landed_x_pixels,
            )
        )
        predicted_y, predicted_velocity_y, predicted_covariance_y = (
            self._paired_predict_axis(
                self._corroboration_position_y,
                self._corroboration_velocity_y,
                self._corroboration_covariance_y,
                elapsed,
                landed_y_pixels,
            )
        )
        self._corroboration_position_x = predicted_x
        self._corroboration_velocity_x = predicted_velocity_x
        self._corroboration_covariance_x = predicted_covariance_x
        self._corroboration_position_y = predicted_y
        self._corroboration_velocity_y = predicted_velocity_y
        self._corroboration_covariance_y = predicted_covariance_y

        measurement_gap = (
            observation.timestamp_ns - last_measurement_ns
        ) / _NS_PER_SECOND
        if measurement_gap > self.config.stale_after_seconds:
            # A genuinely missing independent stream cannot retain latent
            # velocity proof until it happens to reappear much later.
            self._motion_corroboration_confidence = 0.0
            self._corroboration_direction_anchor_x = 0.0
            self._corroboration_direction_anchor_y = 0.0
            self._corroboration_direction_persistence_seconds = 0.0
            self._corroboration_reseed_required = True
            if measured_x is not None and measured_y is not None:
                self._seed_corroboration(
                    measured_x,
                    measured_y,
                    timestamp_ns=observation.timestamp_ns,
                )
            return False
        if measured_x is None or measured_y is None:
            # Absence withdraws current output authority at the caller.  It is
            # not contradictory evidence, so preserve the latent covariance,
            # direction history, and confidence for the next sparse sample.
            return False

        measurement_variance = _CORROBORATION_MEASUREMENT_SIGMA_PIXELS**2
        innovation_x = measured_x - predicted_x
        innovation_y = measured_y - predicted_y
        mahalanobis_squared = (
            innovation_x
            * innovation_x
            / (predicted_covariance_x[0] + measurement_variance)
            + innovation_y
            * innovation_y
            / (predicted_covariance_y[0] + measurement_variance)
        )
        if (
            not math.isfinite(mahalanobis_squared)
            or mahalanobis_squared
            > _CORROBORATION_INNOVATION_GATE_MAHALANOBIS_SQUARED
        ):
            # A body-box excursion is never allowed to invalidate an otherwise
            # sound direct-head sample.  It merely withdraws permission for
            # predictive feed-forward and reseeds its own independent state.
            self._seed_corroboration(
                measured_x,
                measured_y,
                timestamp_ns=observation.timestamp_ns,
            )
            self._motion_corroboration_confidence = 0.0
            return False

        (
            self._corroboration_position_x,
            self._corroboration_velocity_x,
            self._corroboration_covariance_x,
        ) = self._paired_update_axis(
            predicted_x,
            predicted_velocity_x,
            predicted_covariance_x,
            measured_x,
            measurement_variance,
        )
        (
            self._corroboration_position_y,
            self._corroboration_velocity_y,
            self._corroboration_covariance_y,
        ) = self._paired_update_axis(
            predicted_y,
            predicted_velocity_y,
            predicted_covariance_y,
            measured_y,
            measurement_variance,
        )
        speed_limit = self.config.maximum_target_speed_pixels_per_second
        self._corroboration_velocity_x = min(
            max(self._corroboration_velocity_x, -speed_limit),
            speed_limit,
        )
        self._corroboration_velocity_y = min(
            max(self._corroboration_velocity_y, -speed_limit),
            speed_limit,
        )
        instantaneous_confidence = self._motion_corroboration(
            self._velocity_x,
            self._velocity_y,
            math.sqrt(self._paired_covariance_x[2]),
            math.sqrt(self._paired_covariance_y[2]),
            self._corroboration_velocity_x,
            self._corroboration_velocity_y,
            math.sqrt(self._corroboration_covariance_x[2]),
            math.sqrt(self._corroboration_covariance_y[2]),
        )
        raw_confidence = instantaneous_confidence * (
            self._direction_persistence_confidence(
                self._velocity_x,
                self._velocity_y,
                self._corroboration_velocity_x,
                self._corroboration_velocity_y,
                measurement_gap,
                evidence_present=instantaneous_confidence > 0.0,
            )
        )
        time_constant = (
            _CORROBORATION_RISE_TIME_CONSTANT_SECONDS
            if raw_confidence > self._motion_corroboration_confidence
            else _CORROBORATION_FALL_TIME_CONSTANT_SECONDS
        )
        alpha = 1.0 - math.exp(-measurement_gap / time_constant)
        self._motion_corroboration_confidence += alpha * (
            raw_confidence - self._motion_corroboration_confidence
        )
        self._last_corroboration_measurement_ns = observation.timestamp_ns
        # An accepted coordinate is not by itself current agreement.  A fixed
        # or contradictory independent reference drives the instantaneous
        # two-observer proof to exact zero while the diagnostic EMA decays
        # asymptotically; do not let that tiny historical remainder authorize a
        # binary near-unity setpoint.  Direction persistence remains the arm
        # condition, but once armed a coherent curved trajectory may retain its
        # setpoint while the direction anchor rotates and rebuilds.
        return instantaneous_confidence > 0.0

    def _seed_corroboration(
        self,
        measured_x: float,
        measured_y: float,
        *,
        timestamp_ns: int,
    ) -> None:
        self._corroboration_position_x = measured_x
        self._corroboration_position_y = measured_y
        self._corroboration_velocity_x = 0.0
        self._corroboration_velocity_y = 0.0
        measurement_variance = _CORROBORATION_MEASUREMENT_SIGMA_PIXELS**2
        initial_velocity_sigma = min(
            self.config.maximum_target_speed_pixels_per_second,
            _PAIRED_INITIAL_VELOCITY_SIGMA_PIXELS_PER_SECOND,
        )
        covariance = (
            measurement_variance,
            0.0,
            initial_velocity_sigma * initial_velocity_sigma,
        )
        self._corroboration_covariance_x = covariance
        self._corroboration_covariance_y = covariance
        self._last_corroboration_measurement_ns = timestamp_ns
        self._corroboration_direction_anchor_x = 0.0
        self._corroboration_direction_anchor_y = 0.0
        self._corroboration_direction_persistence_seconds = 0.0
        self._corroboration_reseed_required = False

    def _direction_persistence_confidence(
        self,
        direct_x: float,
        direct_y: float,
        reference_x: float,
        reference_y: float,
        elapsed: float,
        *,
        evidence_present: bool,
    ) -> float:
        """Require one stable translation direction, rejecting curved orbits."""

        if not evidence_present:
            self._corroboration_direction_persistence_seconds = 0.0
            return 0.0
        common_x = direct_x + reference_x
        common_y = direct_y + reference_y
        magnitude = math.hypot(common_x, common_y)
        if magnitude <= 1e-9:
            self._corroboration_direction_persistence_seconds = 0.0
            return 0.0
        direction_x = common_x / magnitude
        direction_y = common_y / magnitude
        anchor_magnitude = math.hypot(
            self._corroboration_direction_anchor_x,
            self._corroboration_direction_anchor_y,
        )
        if anchor_magnitude <= 1e-9:
            self._corroboration_direction_anchor_x = direction_x
            self._corroboration_direction_anchor_y = direction_y
            self._corroboration_direction_persistence_seconds = 0.0
            return 0.0
        alignment = (
            direction_x * self._corroboration_direction_anchor_x
            + direction_y * self._corroboration_direction_anchor_y
        )
        if alignment < _CORROBORATION_MINIMUM_DIRECTION_COSINE:
            self._corroboration_direction_anchor_x = direction_x
            self._corroboration_direction_anchor_y = direction_y
            self._corroboration_direction_persistence_seconds = 0.0
            return 0.0
        self._corroboration_direction_persistence_seconds += elapsed
        return self._ramp(
            self._corroboration_direction_persistence_seconds,
            _CORROBORATION_ZERO_DIRECTION_PERSISTENCE_SECONDS,
            _CORROBORATION_FULL_DIRECTION_PERSISTENCE_SECONDS,
        )

    def _update_body_derived_motion_confidence(
        self,
        elapsed: float,
        *,
        measured_motion_delta_x: float,
        measured_motion_delta_y: float,
        motion_innovation_x: float,
        motion_innovation_y: float,
        motion_innovation_sigma_x: float,
        motion_innovation_sigma_y: float,
        prior_velocity_x: float,
        prior_velocity_y: float,
    ) -> None:
        """Authorize only statistically coherent, persistent mapped motion.

        This uses the mapped point's own paired observer and therefore is not
        independent corroboration.  It cannot raise the independent diagnostic
        or exceed either configured ceiling.  Each axis earns its own grant so
        coherent motion on one axis cannot authorize orthogonal jitter.  A
        material command-compensated reversal withdraws that axis immediately,
        before the smoothed velocity estimate can carry the old direction into
        another source-age projection.
        """

        source_confidence_x = (
            self._paired_velocity_evidence(
                self._velocity_x,
                math.sqrt(self._paired_covariance_x[2]),
            )
            * self._paired_measurement_agreement_x
        )
        source_confidence_y = (
            self._paired_velocity_evidence(
                self._velocity_y,
                math.sqrt(self._paired_covariance_y[2]),
            )
            * self._paired_measurement_agreement_y
        )
        vector_confidence = self._update_body_derived_vector_confidence(
            source_confidence_x,
            source_confidence_y,
            elapsed=elapsed,
        )
        (
            self._body_derived_motion_confidence_x,
            self._body_derived_axis_direction_anchor_x,
            self._body_derived_direction_persistence_x_seconds,
        ) = self._body_derived_axis_confidence(
            velocity=self._velocity_x,
            source_confidence=source_confidence_x,
            measured_motion_delta=measured_motion_delta_x,
            motion_innovation=motion_innovation_x,
            motion_innovation_sigma=motion_innovation_sigma_x,
            prior_velocity=prior_velocity_x,
            direction_anchor=self._body_derived_axis_direction_anchor_x,
            direction_persistence_seconds=(
                self._body_derived_direction_persistence_x_seconds
            ),
            elapsed=elapsed,
        )
        (
            self._body_derived_motion_confidence_y,
            self._body_derived_axis_direction_anchor_y,
            self._body_derived_direction_persistence_y_seconds,
        ) = self._body_derived_axis_confidence(
            velocity=self._velocity_y,
            source_confidence=source_confidence_y,
            measured_motion_delta=measured_motion_delta_y,
            motion_innovation=motion_innovation_y,
            motion_innovation_sigma=motion_innovation_sigma_y,
            prior_velocity=prior_velocity_y,
            direction_anchor=self._body_derived_axis_direction_anchor_y,
            direction_persistence_seconds=(
                self._body_derived_direction_persistence_y_seconds
            ),
            elapsed=elapsed,
        )
        self._body_derived_motion_confidence_x = min(
            self._body_derived_motion_confidence_x,
            vector_confidence,
        )
        self._body_derived_motion_confidence_y = min(
            self._body_derived_motion_confidence_y,
            vector_confidence,
        )
        self._body_derived_fresh_motion_confidence_x = (
            self._body_derived_fresh_motion_confidence(
                velocity=self._velocity_x,
                measured_motion_delta=measured_motion_delta_x,
                elapsed=elapsed,
            )
        )
        self._body_derived_fresh_motion_confidence_y = (
            self._body_derived_fresh_motion_confidence(
                velocity=self._velocity_y,
                measured_motion_delta=measured_motion_delta_y,
                elapsed=elapsed,
            )
        )

    def _update_common_pursuit_direction_persistence(
        self,
        *,
        elapsed: float,
    ) -> None:
        """Retain coherent primary direction across optional evidence handoffs.

        The primary paired velocity remains one command-compensated physical
        hypothesis when phase flow briefly drops to mapped-body fallback.  Its
        direction history is therefore independent of which optional authority
        channel is present.  This state cannot authorize feed-forward by itself:
        the independent corroboration/body provenance and bounded handoff lease
        in :meth:`step` still do that.  Newest-frame stop or reversal evidence
        clears the history before the smoothed velocity can coast through it.
        """

        velocity_x = self._velocity_x
        velocity_y = self._velocity_y
        magnitude = math.hypot(velocity_x, velocity_y)
        fresh_confidence = max(
            self._paired_fresh_motion_confidence_x,
            self._paired_fresh_motion_confidence_y,
        )
        if (
            elapsed <= 0.0
            or magnitude
            <= _COMMON_PAIRED_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND
            or fresh_confidence <= 0.0
        ):
            self._common_pursuit_direction_anchor_x = 0.0
            self._common_pursuit_direction_anchor_y = 0.0
            self._common_pursuit_direction_persistence_seconds = 0.0
            return

        direction_x = velocity_x / magnitude
        direction_y = velocity_y / magnitude
        anchor_magnitude = math.hypot(
            self._common_pursuit_direction_anchor_x,
            self._common_pursuit_direction_anchor_y,
        )
        if anchor_magnitude <= 1e-9:
            self._common_pursuit_direction_anchor_x = direction_x
            self._common_pursuit_direction_anchor_y = direction_y
            self._common_pursuit_direction_persistence_seconds = 0.0
            return
        alignment = (
            direction_x * self._common_pursuit_direction_anchor_x
            + direction_y * self._common_pursuit_direction_anchor_y
        )
        if alignment < _CORROBORATION_MINIMUM_DIRECTION_COSINE:
            self._common_pursuit_direction_anchor_x = direction_x
            self._common_pursuit_direction_anchor_y = direction_y
            self._common_pursuit_direction_persistence_seconds = 0.0
            return
        self._common_pursuit_direction_persistence_seconds += elapsed

    @classmethod
    def _body_derived_fresh_motion_confidence(
        cls,
        *,
        velocity: float,
        measured_motion_delta: float,
        elapsed: float,
    ) -> float:
        """Score newest-frame motion agreement for the fast-pursuit reserve."""

        if elapsed <= 0.0 or velocity * measured_motion_delta <= 0.0:
            return 0.0
        measured_velocity = measured_motion_delta / elapsed
        if not math.isfinite(measured_velocity) or measured_velocity * velocity <= 0.0:
            return 0.0
        speed_ratio = min(abs(measured_velocity), abs(velocity)) / max(
            abs(measured_velocity),
            abs(velocity),
            1e-9,
        )
        return cls._ramp(
            speed_ratio,
            _BODY_DERIVED_PURSUIT_ZERO_FRESH_SPEED_RATIO,
            _BODY_DERIVED_PURSUIT_FULL_FRESH_SPEED_RATIO,
        )

    @staticmethod
    def _material_stop_or_reversal(
        *,
        prior_velocity: float,
        measured_motion_delta: float,
        motion_innovation: float,
        motion_innovation_sigma: float,
    ) -> bool:
        """Identify newest-frame contrary motion beyond detector uncertainty."""

        if (
            abs(prior_velocity)
            < _BODY_DERIVED_IMMEDIATE_REVERSAL_MINIMUM_SPEED_PIXELS_PER_SECOND
            or not math.isfinite(motion_innovation_sigma)
            or motion_innovation_sigma <= 0.0
        ):
            return False
        direction = 1.0 if prior_velocity > 0.0 else -1.0
        return bool(
            measured_motion_delta * direction <= 0.0
            and motion_innovation * direction
            <= -(
                _BODY_DERIVED_REVERSAL_MINIMUM_INNOVATION_SIGMAS
                * motion_innovation_sigma
            )
        )

    def _update_body_derived_vector_confidence(
        self,
        source_confidence_x: float,
        source_confidence_y: float,
        *,
        elapsed: float,
    ) -> float:
        """Retain the existing 2-D coherence gate for circular point jitter."""

        signal_confidence = min(
            max(max(source_confidence_x, source_confidence_y), 0.0),
            1.0,
        )
        magnitude = math.hypot(self._velocity_x, self._velocity_y)
        if signal_confidence <= 0.0 or magnitude <= 1e-9:
            self._body_derived_direction_anchor_x = 0.0
            self._body_derived_direction_anchor_y = 0.0
            self._body_derived_direction_persistence_seconds = 0.0
            return 0.0
        direction_x = self._velocity_x / magnitude
        direction_y = self._velocity_y / magnitude
        anchor_magnitude = math.hypot(
            self._body_derived_direction_anchor_x,
            self._body_derived_direction_anchor_y,
        )
        if anchor_magnitude <= 1e-9:
            self._body_derived_direction_anchor_x = direction_x
            self._body_derived_direction_anchor_y = direction_y
            self._body_derived_direction_persistence_seconds = 0.0
            return 0.0
        alignment = (
            direction_x * self._body_derived_direction_anchor_x
            + direction_y * self._body_derived_direction_anchor_y
        )
        if alignment < _BODY_DERIVED_MINIMUM_DIRECTION_COSINE:
            self._body_derived_direction_anchor_x = direction_x
            self._body_derived_direction_anchor_y = direction_y
            self._body_derived_direction_persistence_seconds = 0.0
            return 0.0
        self._body_derived_direction_persistence_seconds += elapsed
        persistence_confidence = self._ramp(
            self._body_derived_direction_persistence_seconds,
            _BODY_DERIVED_ZERO_DIRECTION_PERSISTENCE_SECONDS,
            _BODY_DERIVED_FULL_DIRECTION_PERSISTENCE_SECONDS,
        )
        return signal_confidence * persistence_confidence

    @classmethod
    def _body_derived_axis_confidence(
        cls,
        *,
        velocity: float,
        source_confidence: float,
        measured_motion_delta: float,
        motion_innovation: float,
        motion_innovation_sigma: float,
        prior_velocity: float,
        direction_anchor: float,
        direction_persistence_seconds: float,
        elapsed: float,
    ) -> tuple[float, float, float]:
        evidence = min(max(source_confidence, 0.0), 1.0)
        if evidence <= 0.0 or abs(velocity) <= 1e-9:
            return 0.0, 0.0, 0.0
        direction = 1.0 if velocity > 0.0 else -1.0
        if (
            direction_anchor != 0.0
            and abs(prior_velocity)
            >= _BODY_DERIVED_IMMEDIATE_REVERSAL_MINIMUM_SPEED_PIXELS_PER_SECOND
            and measured_motion_delta * direction_anchor < 0.0
            and motion_innovation * direction_anchor
            <= -(
                _BODY_DERIVED_REVERSAL_MINIMUM_INNOVATION_SIGMAS
                * motion_innovation_sigma
            )
        ):
            # Do not seed from the still-smoothed velocity on the reversal
            # frame.  The next accepted observation must establish a fresh
            # direction and then pass the ordinary persistence ramp.
            return 0.0, 0.0, 0.0
        if direction_anchor == 0.0:
            return 0.0, direction, 0.0
        if direction != direction_anchor:
            return 0.0, direction, 0.0
        persistence = direction_persistence_seconds + elapsed
        persistence_confidence = cls._ramp(
            persistence,
            _BODY_DERIVED_ZERO_DIRECTION_PERSISTENCE_SECONDS,
            _BODY_DERIVED_FULL_DIRECTION_PERSISTENCE_SECONDS,
        )
        return evidence * persistence_confidence, direction, persistence

    @staticmethod
    def _ramp(value: float, zero: float, full: float) -> float:
        return min(max((value - zero) / (full - zero), 0.0), 1.0)

    @classmethod
    def _motion_corroboration(
        cls,
        direct_x: float,
        direct_y: float,
        direct_sigma_x: float,
        direct_sigma_y: float,
        reference_x: float,
        reference_y: float,
        reference_sigma_x: float,
        reference_sigma_y: float,
    ) -> float:
        """Score vector direction, magnitude, and statistical significance."""

        direct_speed = math.hypot(direct_x, direct_y)
        reference_speed = math.hypot(reference_x, reference_y)
        if direct_speed <= 1e-9 or reference_speed <= 1e-9:
            return 0.0
        direction_cosine = (
            direct_x * reference_x + direct_y * reference_y
        ) / (direct_speed * reference_speed)
        direction_confidence = cls._ramp(
            direction_cosine,
            _CORROBORATION_ZERO_DIRECTION_COSINE,
            _CORROBORATION_FULL_DIRECTION_COSINE,
        )
        speed_ratio = min(direct_speed, reference_speed) / max(
            direct_speed,
            reference_speed,
        )
        speed_confidence = cls._ramp(
            speed_ratio,
            _CORROBORATION_ZERO_SPEED_RATIO,
            _CORROBORATION_FULL_SPEED_RATIO,
        )
        direct_sigma = math.hypot(direct_sigma_x, direct_sigma_y)
        reference_sigma = math.hypot(reference_sigma_x, reference_sigma_y)
        signal_to_noise = min(
            direct_speed / max(direct_sigma, 1e-9),
            reference_speed / max(reference_sigma, 1e-9),
        )
        signal_confidence = cls._ramp(
            signal_to_noise,
            _CORROBORATION_ZERO_SIGNAL_TO_NOISE,
            _CORROBORATION_FULL_SIGNAL_TO_NOISE,
        )
        return direction_confidence * speed_confidence * signal_confidence

    def _paired_predict_axis(
        self,
        position: float,
        velocity: float,
        covariance: tuple[float, float, float],
        elapsed: float,
        landed_command_pixels: float,
    ) -> tuple[float, float, tuple[float, float, float]]:
        """Predict one axis with landed mouse motion as a known input."""

        predicted_position = position + velocity * elapsed - landed_command_pixels
        position_variance, cross_covariance, velocity_variance = covariance
        acceleration_sigma = min(
            max(
                self.config.maximum_target_acceleration_pixels_per_second_squared
                * _PAIRED_ACCELERATION_SIGMA_FRACTION,
                _PAIRED_MINIMUM_ACCELERATION_SIGMA_PIXELS_PER_SECOND_SQUARED,
            ),
            _PAIRED_MAXIMUM_ACCELERATION_SIGMA_PIXELS_PER_SECOND_SQUARED,
        )
        acceleration_variance = acceleration_sigma * acceleration_sigma
        elapsed_squared = elapsed * elapsed
        process_position_variance = (
            0.25 * elapsed_squared * elapsed_squared * acceleration_variance
        )
        process_cross_covariance = (
            0.5 * elapsed_squared * elapsed * acceleration_variance
        )
        process_velocity_variance = elapsed_squared * acceleration_variance
        # The raw count is exact, while a no-profile pixel/count seed can be
        # imperfect. Represent a bounded 20% gain mismatch as control-input
        # uncertainty so a large command does not make the observer falsely
        # overconfident in its projected pixel position.
        input_sigma = (
            abs(landed_command_pixels)
            * _PAIRED_PLANT_GAIN_UNCERTAINTY_FRACTION
        )
        predicted_position_variance = (
            position_variance
            + 2.0 * elapsed * cross_covariance
            + elapsed_squared * velocity_variance
            + process_position_variance
            + input_sigma * input_sigma
        )
        predicted_cross_covariance = (
            cross_covariance
            + elapsed * velocity_variance
            + process_cross_covariance
        )
        predicted_velocity_variance = (
            velocity_variance + process_velocity_variance
        )
        predicted_covariance = self._bounded_covariance(
            predicted_position_variance,
            predicted_cross_covariance,
            predicted_velocity_variance,
        )
        return predicted_position, velocity, predicted_covariance

    @staticmethod
    def _paired_update_axis(
        predicted_position: float,
        predicted_velocity: float,
        predicted_covariance: tuple[float, float, float],
        measurement: float,
        measurement_variance: float,
    ) -> tuple[float, float, tuple[float, float, float]]:
        """Apply one scalar position measurement to one Kalman axis."""

        position_variance, cross_covariance, velocity_variance = (
            predicted_covariance
        )
        innovation_variance = position_variance + measurement_variance
        position_gain = position_variance / innovation_variance
        velocity_gain = cross_covariance / innovation_variance
        innovation = measurement - predicted_position
        updated_position = predicted_position + position_gain * innovation
        updated_velocity = predicted_velocity + velocity_gain * innovation
        updated_position_variance = (
            position_variance - position_gain * position_variance
        )
        updated_cross_covariance = (
            cross_covariance - position_gain * cross_covariance
        )
        updated_velocity_variance = (
            velocity_variance - velocity_gain * cross_covariance
        )
        covariance = MakcuCalibratedController._bounded_covariance(
            updated_position_variance,
            updated_cross_covariance,
            updated_velocity_variance,
        )
        return updated_position, updated_velocity, covariance

    @staticmethod
    def _bounded_covariance(
        position_variance: float,
        cross_covariance: float,
        velocity_variance: float,
    ) -> tuple[float, float, float]:
        """Keep the tiny covariance finite, symmetric, and positive semidefinite."""

        minimum_variance = 1e-12
        position_variance = max(position_variance, minimum_variance)
        velocity_variance = max(velocity_variance, minimum_variance)
        maximum_cross = math.sqrt(position_variance * velocity_variance)
        cross_covariance = min(
            max(cross_covariance, -maximum_cross),
            maximum_cross,
        )
        return position_variance, cross_covariance, velocity_variance

    def _paired_velocity_confidence(
        self,
        velocity: float,
        velocity_sigma: float,
    ) -> float:
        """Return covariance- and signal-limited velocity feed-forward weight."""

        covariance_confidence, signal_confidence = (
            self._paired_velocity_confidence_components(
                velocity,
                velocity_sigma,
            )
        )
        # Retain the historical multiplication order for existing profiles.
        return (
            self.config.maximum_velocity_feedforward_fraction
            * covariance_confidence
            * signal_confidence
        )

    @classmethod
    def _paired_velocity_evidence(
        cls,
        velocity: float,
        velocity_sigma: float,
    ) -> float:
        """Return normalized paired evidence, independent of explicit FF cap."""

        covariance_confidence, signal_confidence = (
            cls._paired_velocity_confidence_components(
                velocity,
                velocity_sigma,
            )
        )
        return covariance_confidence * signal_confidence

    @staticmethod
    def _paired_velocity_confidence_components(
        velocity: float,
        velocity_sigma: float,
    ) -> tuple[float, float]:
        covariance_confidence = min(
            max(
                (
                    _PAIRED_FEEDFORWARD_ZERO_VELOCITY_SIGMA
                    - velocity_sigma
                )
                / (
                    _PAIRED_FEEDFORWARD_ZERO_VELOCITY_SIGMA
                    - _PAIRED_FEEDFORWARD_FULL_VELOCITY_SIGMA
                ),
                0.0,
            ),
            1.0,
        )
        signal_to_noise = abs(velocity) / max(velocity_sigma, 1e-9)
        signal_confidence = min(
            max(
                (
                    signal_to_noise
                    - _PAIRED_FEEDFORWARD_ZERO_SIGNAL_TO_NOISE
                )
                / (
                    _PAIRED_FEEDFORWARD_FULL_SIGNAL_TO_NOISE
                    - _PAIRED_FEEDFORWARD_ZERO_SIGNAL_TO_NOISE
                ),
                0.0,
            ),
            1.0,
        )
        return covariance_confidence, signal_confidence

    @staticmethod
    def _paired_channel_agreement(
        raw_x: float,
        raw_y: float,
        tracked_x: float,
        tracked_y: float,
        *,
        tracked_velocity_x: float = 0.0,
        tracked_velocity_y: float = 0.0,
    ) -> float:
        # Treat X/Y as one detector point.  Independent axis confidence lets a
        # quantized point walk around a square while alternately trusting the
        # quiet-looking axis; radial agreement closes that rotating loophole.
        disagreement_x = raw_x - tracked_x
        disagreement_y = raw_y - tracked_y
        phase_x = (
            tracked_velocity_x * _BODY_DERIVED_POSITION_CHANNEL_PHASE_SECONDS
        )
        phase_y = (
            tracked_velocity_y * _BODY_DERIVED_POSITION_CHANNEL_PHASE_SECONDS
        )
        phase_magnitude = math.hypot(phase_x, phase_y)
        if phase_magnitude > 0.0:
            allowed_magnitude = min(
                phase_magnitude,
                _BODY_DERIVED_MAXIMUM_POSITION_PHASE_ALLOWANCE_PIXELS,
            )
            unit_x = phase_x / phase_magnitude
            unit_y = phase_y / phase_magnitude
            collinear = disagreement_x * unit_x + disagreement_y * unit_y
            forgiven = min(max(collinear, 0.0), allowed_magnitude)
            disagreement_x -= forgiven * unit_x
            disagreement_y -= forgiven * unit_y
        disagreement = math.hypot(disagreement_x, disagreement_y)
        return min(
            max(
                (
                    _PAIRED_ZERO_POSITION_CHANNEL_AGREEMENT_PIXELS
                    - disagreement
                )
                / (
                    _PAIRED_ZERO_POSITION_CHANNEL_AGREEMENT_PIXELS
                    - _PAIRED_FULL_POSITION_CHANNEL_AGREEMENT_PIXELS
                ),
                0.0,
            ),
            1.0,
        )

    def _paired_feedback_hold(
        self,
        held: bool,
        projected_error: float,
        position_sigma: float,
    ) -> bool:
        """Apply a covariance-sized Schmitt band to positional feedback."""

        enter_threshold = max(
            self.config.feedback_deadzone_pixels,
            _PAIRED_FEEDBACK_ENTER_SIGMAS * position_sigma,
        )
        exit_threshold = max(
            self.config.feedback_deadzone_pixels,
            _PAIRED_FEEDBACK_EXIT_SIGMAS * position_sigma,
        )
        if held:
            return abs(projected_error) <= exit_threshold
        return abs(projected_error) <= enter_threshold

    def _paired_axis_rate(
        self,
        projected_error: float,
        velocity: float,
        gain: float,
        limit: float,
        feedforward_confidence: float,
        *,
        feedback_limit: float | None = None,
        feedback_held: bool,
        position_confidence: float,
        pursuit_authorized: bool = True,
        pursuit_confidence: float = 1.0,
    ) -> tuple[float, bool]:
        resolved_pursuit_confidence = min(
            max(float(pursuit_confidence), 0.0),
            1.0,
        )
        feedback = 0.0
        if abs(projected_error) > self.config.feedback_deadzone_pixels:
            feedback_fraction = (
                _PAIRED_HELD_FEEDBACK_FRACTION if feedback_held else 1.0
            )
            feedback_error = self._feedback_error(projected_error)
            feedback = (
                position_confidence
                * feedback_fraction
                * feedback_error
                / (
                    gain
                    * (
                        self.config.position_time_constant_seconds
                        + resolved_pursuit_confidence
                        * (
                            self._position_time_constant_seconds_for_error(
                                projected_error
                            )
                            - self.config.position_time_constant_seconds
                        )
                        if pursuit_authorized
                        else self.config.position_time_constant_seconds
                    )
                )
            )
        resolved_feedback_limit = (
            limit if feedback_limit is None else min(feedback_limit, limit)
        )
        bounded_feedback = min(
            max(feedback, -resolved_feedback_limit),
            resolved_feedback_limit,
        )
        feed_forward = feedforward_confidence * velocity / gain
        requested = bounded_feedback + feed_forward
        if (
            abs(projected_error) > self.config.wrong_way_guard_pixels
            and requested * projected_error < 0.0
        ):
            requested = bounded_feedback
        bounded = min(max(requested, -limit), limit)
        return bounded, bounded != requested or bounded_feedback != feedback

    def _paired_position_feedback_confidence(
        self,
        projected_error: float,
        measurement_agreement: float,
    ) -> float:
        """Keep auxiliary disagreement from closing the far-error P loop.

        Position and velocity points intentionally use different filters in
        automatic direct-head control. During fast translation their phase
        separation can legitimately exceed the raw-motion agreement gate.
        That gate remains authoritative for every predictive term. The
        filtered position itself, however, is still the causal closed-loop aim
        coordinate, so smoothly restore its feedback authority only across the
        already configured pursuit region. Near-lock behavior remains
        bit-for-bit unchanged through the pursuit start threshold.
        """

        agreement = min(max(float(measurement_agreement), 0.0), 1.0)
        if not self.config.preserve_pursuit_position_feedback:
            return agreement
        magnitude = abs(float(projected_error))
        start = self.config.pursuit_position_time_constant_start_pixels
        full = self.config.pursuit_position_time_constant_full_pixels
        if magnitude <= start:
            return agreement
        if magnitude >= full:
            return 1.0
        fraction = (magnitude - start) / (full - start)
        pursuit_confidence = fraction * fraction * (3.0 - 2.0 * fraction)
        return max(agreement, pursuit_confidence)

    def _position_time_constant_seconds_for_error(
        self,
        projected_error: float,
    ) -> float:
        pursuit_time_constant = (
            self.config.pursuit_position_time_constant_seconds
        )
        if pursuit_time_constant == 0.0:
            return self.config.position_time_constant_seconds
        magnitude = abs(projected_error)
        start = self.config.pursuit_position_time_constant_start_pixels
        full = self.config.pursuit_position_time_constant_full_pixels
        if magnitude <= start:
            return self.config.position_time_constant_seconds
        if magnitude >= full:
            return pursuit_time_constant
        fraction = (magnitude - start) / (full - start)
        smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        return self.config.position_time_constant_seconds + smooth_fraction * (
            pursuit_time_constant - self.config.position_time_constant_seconds
        )

    def _body_derived_feedforward_error_gate(
        self,
        measured_error: float,
        projected_error: float,
        projected_velocity: float,
    ) -> float:
        """Return closed-loop authority for correlated mapped-body velocity."""

        if (
            measured_error * projected_velocity <= 0.0
            or projected_error * projected_velocity <= 0.0
        ):
            return 0.0
        magnitude = min(abs(measured_error), abs(projected_error))
        start = self.config.feedback_deadzone_pixels
        shoulder = self.config.continuous_feedback_shoulder_pixels
        if magnitude <= start:
            return 0.0
        if shoulder <= 0.0 or magnitude >= start + shoulder:
            return 1.0
        fraction = (magnitude - start) / shoulder
        return fraction * fraction * (3.0 - 2.0 * fraction)

    def _adaptive_body_position_reconciliation(
        self,
        measured_error: float,
        velocity_error: float,
        projected_velocity: float,
        adaptive_confidence: float,
        *,
        motion_confidence: float,
        fresh_motion_confidence: float,
    ) -> float:
        """Recover verified body translation hidden by the position-only LP."""

        maximum = self.config.maximum_body_derived_pursuit_feedforward_fraction
        if (
            maximum <= 0.0
            or adaptive_confidence <= 0.0
            or motion_confidence <= 0.0
            or fresh_motion_confidence <= 0.0
            or abs(projected_velocity) <= 1e-9
        ):
            return 0.0
        phase_error = velocity_error - measured_error
        if phase_error * projected_velocity <= 0.0:
            return 0.0
        bounded_phase_error = min(
            max(
                phase_error,
                -_BODY_DERIVED_ADAPTIVE_MAXIMUM_POSITION_RECONCILIATION_PIXELS,
            ),
            _BODY_DERIVED_ADAPTIVE_MAXIMUM_POSITION_RECONCILIATION_PIXELS,
        )
        return bounded_phase_error * min(
            max(adaptive_confidence / maximum, 0.0),
            1.0,
        )

    def _clear_residual_pursuit(self) -> None:
        self._residual_pursuit_x = _ResidualPursuitAxisState()
        self._residual_pursuit_y = _ResidualPursuitAxisState()
        self._clear_verified_residual_projection()

    def _clear_verified_residual_projection(self) -> None:
        self._verified_residual_projection_x = False
        self._verified_residual_projection_y = False

    def _update_residual_pursuit(
        self,
        *,
        measurement_event: bool,
        current_measured_x: bool,
        current_measured_y: bool,
        elapsed: float,
        measured_error_x: float,
        measured_error_y: float,
        projected_error_x: float,
        projected_error_y: float,
        projected_velocity_x: float,
        projected_velocity_y: float,
        position_sigma_x: float,
        position_sigma_y: float,
        position_agreement_x: float,
        position_agreement_y: float,
        revoke_x: bool,
        revoke_y: bool,
    ) -> None:
        """Update opt-in authority from persistent closed-loop target trail.

        This path intentionally does not reuse the sparse body/flow confidence
        EMA: the live trace showed that EMA at zero while the paired observer's
        position and target-velocity direction remained coherent.  Ordinary
        measurements retain their existing independent/body authorization.
        The narrow exception is a correlated P1 endpoint whose LK flag and
        command-compensated displacement were both accepted by the numeric
        core; its per-axis fresh-motion result may supply the same evidence
        without broadening ordinary detector authority.  Velocity is still
        only the direction and bounded setpoint.  Multiple causal,
        covariance-sized position residuals authorize the state, which makes
        it an integral remedy for demonstrated lag rather than another noisy
        derivative gain.
        """

        if self.config.maximum_residual_pursuit_feedforward_fraction <= 0.0:
            # Preserve the default path bit-for-bit and make a runtime config
            # transition fail closed if a caller ever replaces the config.
            self._clear_residual_pursuit()
            return
        self._residual_pursuit_x = self._residual_pursuit_axis_state(
            self._residual_pursuit_x,
            measurement_event=measurement_event,
            current_measured=current_measured_x,
            elapsed=elapsed,
            measured_error=measured_error_x,
            projected_error=projected_error_x,
            projected_velocity=projected_velocity_x,
            position_sigma=position_sigma_x,
            position_agreement=position_agreement_x,
            revoke=revoke_x,
        )
        self._residual_pursuit_y = self._residual_pursuit_axis_state(
            self._residual_pursuit_y,
            measurement_event=measurement_event,
            current_measured=current_measured_y,
            elapsed=elapsed,
            measured_error=measured_error_y,
            projected_error=projected_error_y,
            projected_velocity=projected_velocity_y,
            position_sigma=position_sigma_y,
            position_agreement=position_agreement_y,
            revoke=revoke_y,
        )

    def _residual_pursuit_axis_state(
        self,
        state: _ResidualPursuitAxisState,
        *,
        measurement_event: bool,
        current_measured: bool,
        elapsed: float,
        measured_error: float,
        projected_error: float,
        projected_velocity: float,
        position_sigma: float,
        position_agreement: float,
        revoke: bool,
    ) -> _ResidualPursuitAxisState:
        zero = _ResidualPursuitAxisState()
        values = (
            elapsed,
            measured_error,
            projected_error,
            projected_velocity,
            position_sigma,
            position_agreement,
        )
        if revoke or not all(math.isfinite(value) for value in values):
            return zero
        minimum_speed = (
            _RESIDUAL_PURSUIT_MINIMUM_AXIS_SPEED_PIXELS_PER_SECOND
        )
        direction = (
            1
            if projected_velocity >= minimum_speed
            else -1
            if projected_velocity <= -minimum_speed
            else 0
        )

        # A retained command is allowed at exact lock, but never after the
        # physical residual has crossed ahead or the observer has reversed or
        # stopped.  Evaluate this on every output tick so pending commands can
        # revoke before the next detector callback.
        if state.direction != 0:
            if (
                direction != state.direction
                or measured_error * state.direction < 0.0
                or projected_error * state.direction < 0.0
            ):
                return zero
        if not measurement_event:
            return state
        if (
            not current_measured
            or direction == 0
            or position_sigma < 0.0
            or position_agreement
            < _RESIDUAL_PURSUIT_MINIMUM_POSITION_AGREEMENT
        ):
            return zero

        least_aligned_error = min(
            measured_error * direction,
            projected_error * direction,
        )
        charge_threshold = max(
            self.config.feedback_deadzone_pixels,
            position_sigma * _RESIDUAL_PURSUIT_POSITION_SIGMA_FRACTION,
        )
        if least_aligned_error <= charge_threshold:
            if state.authority > 0.0 and least_aligned_error >= 0.0:
                # Exact/near lock is the desired result of the retained
                # disturbance estimate.  Hold its already-earned value, but
                # require a fresh three-sample proof before charging it further.
                return _ResidualPursuitAxisState(
                    state.authority,
                    state.direction,
                    0,
                    0.0,
                )
            return zero

        if state.direction == direction and state.aligned_samples > 0:
            aligned_samples = state.aligned_samples + 1
            aligned_seconds = state.aligned_seconds + max(elapsed, 0.0)
        else:
            aligned_samples = 1
            # The accepted observer velocity at this first qualifying sample
            # was itself measured over ``elapsed`` from the preceding causal
            # point. Count that evidence interval while the independent sample
            # counter still requires three ordinary accepted measurements;
            # neither a long two-sample interval nor a low-SNR two-sample lobe
            # can satisfy the complete qualification gate.
            aligned_seconds = max(elapsed, 0.0)
        authority = state.authority
        if (
            aligned_samples >= _RESIDUAL_PURSUIT_MINIMUM_ALIGNED_SAMPLES
            and aligned_seconds >= _RESIDUAL_PURSUIT_MINIMUM_ALIGNMENT_SECONDS
        ):
            speed_gate = self._ramp(
                abs(projected_velocity),
                _RESIDUAL_PURSUIT_MINIMUM_AXIS_SPEED_PIXELS_PER_SECOND,
                _COMMON_PAIRED_PURSUIT_FULL_SPEED_PIXELS_PER_SECOND,
            )
            target = min(
                self.config.maximum_residual_pursuit_feedforward_fraction,
                self.config.maximum_body_derived_pursuit_feedforward_fraction,
                self.config.maximum_velocity_feedforward_fraction,
            ) * speed_gate
            alpha = 1.0 - math.exp(
                -max(elapsed, 0.0)
                / _RESIDUAL_PURSUIT_RISE_TIME_CONSTANT_SECONDS
            )
            authority += alpha * (target - authority)
            authority = min(max(authority, 0.0), target)
        return _ResidualPursuitAxisState(
            authority,
            direction,
            aligned_samples,
            aligned_seconds,
        )

    def _adaptive_body_pursuit_confidence(
        self,
        current: float,
        *,
        measured_error: float,
        projected_error: float,
        projected_velocity: float,
        projected_vector_speed: float,
        motion_confidence: float,
        fresh_motion_confidence: float,
        direction_persistence_seconds: float,
        elapsed: float,
        qualification_zero_speed: float = (
            _BODY_DERIVED_ADAPTIVE_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND
        ),
        qualification_full_speed: float = (
            _BODY_DERIVED_ADAPTIVE_PURSUIT_FULL_SPEED_PIXELS_PER_SECOND
        ),
        rise_time_constant: float = (
            _BODY_DERIVED_ADAPTIVE_PURSUIT_RISE_TIME_CONSTANT_SECONDS
        ),
        retain_through_ambiguous_fresh_motion: bool = False,
        binary_promotion: bool = False,
    ) -> float:
        """Learn a stop-safe velocity fraction from persistent pursuit lag.

        ``motion_confidence`` is authorization here, not an amplitude control.
        Multiplying the already covariance- and persistence-qualified evidence
        into the feed-forward fraction a second time is what left the live
        reserve at only 30--45% despite an 82% configured ceiling.
        """

        configured_maximum = (
            self.config.maximum_body_derived_pursuit_feedforward_fraction
        )
        ordinary_maximum = self.config.maximum_body_derived_feedforward_fraction
        if configured_maximum <= ordinary_maximum:
            return 0.0
        # The adaptive path deliberately avoids multiplying coherent evidence
        # twice, but it remains velocity feed-forward and must still honor the
        # caller's global feed-forward ceiling.
        maximum = min(
            configured_maximum,
            self.config.maximum_velocity_feedforward_fraction,
        )
        if maximum <= 0.0:
            return 0.0
        retained = min(max(float(current), 0.0), maximum)
        if (
            elapsed <= 0.0
            or abs(projected_velocity) <= 1e-9
        ):
            return 0.0
        if (
            fresh_motion_confidence <= 0.0
            and not retain_through_ambiguous_fresh_motion
        ):
            # A second low-motion sample or a statistically material first
            # stop/reversal is the stop/manual-input boundary.  The caller may
            # explicitly bridge one merely quantized sample, never create or
            # increase a grant from it.
            return 0.0

        direction = 1.0 if projected_velocity > 0.0 else -1.0
        least_aligned_error = min(
            measured_error * direction,
            projected_error * direction,
        )
        residual_gate = self._ramp(
            least_aligned_error,
            _BODY_DERIVED_PURSUIT_ZERO_OPPOSED_ERROR_PIXELS,
            _BODY_DERIVED_PURSUIT_FULL_OPPOSED_ERROR_PIXELS,
        )
        retained *= residual_gate
        if residual_gate <= 0.0:
            return 0.0
        if retain_through_ambiguous_fresh_motion:
            return retained

        # A brief pixel-flow/body-fallback transition withdraws fresh authority
        # but is not contrary motion.  Preserve only the already-earned,
        # residual-aligned fraction while the newly selected provenance rebuilds
        # its covariance/persistence gate; it cannot charge without current
        # evidence.  Stops, reversals, and manual approaches were handled above
        # and still clear it on the newest accepted frame.
        if motion_confidence <= 0.0:
            return retained

        # Hand the cap-bound regime back to the existing immediate reserve,
        # whose broad plant-gain matrix is already validated there. Avoid a
        # speed-proportional fade: noisy 1,200 px/s observations crossed that
        # band repeatedly and turned an otherwise stable learned rate into
        # command churn.
        if (
            projected_vector_speed
            >= (
                _BODY_DERIVED_ADAPTIVE_PURSUIT_MAXIMUM_VECTOR_SPEED_PIXELS_PER_SECOND
            )
            or abs(projected_velocity) >= 2_000.0
        ):
            return 0.0
        if (
            direction_persistence_seconds
            < _BODY_DERIVED_FULL_DIRECTION_PERSISTENCE_SECONDS
        ):
            return retained

        # Independent phase/body corroboration is an authorization decision,
        # not another amplitude multiplier.  Once its covariance, newest-frame
        # motion, and 50 ms direction proof agree, promote directly to the
        # bounded target-velocity setpoint.  Waiting for position lag to grow
        # (or adding another slow rise filter) recreates the pursuit error this
        # branch exists to prevent.
        speed_gate = self._ramp(
            abs(projected_velocity),
            qualification_zero_speed,
            qualification_full_speed,
        )
        if binary_promotion:
            if speed_gate <= 0.0:
                return retained
            promoted = ordinary_maximum + speed_gate * (
                maximum - ordinary_maximum
            )
            return max(retained, promoted)

        # Mapped-body fallback is only one temporally correlated detector
        # signal and is less certain under plant-gain mismatch.  Give it the
        # slower validated rise instead of the independent path's immediate
        # near-unity setpoint.  It still begins before positional lag appears:
        # making feed-forward wait for an error is the circular dependency that
        # caused the live target trail in the first place.
        charge_gate = speed_gate * min(
            max(fresh_motion_confidence, 0.0),
            1.0,
        )
        if charge_gate <= 0.0:
            return retained
        alpha = 1.0 - math.exp(
            -elapsed / max(rise_time_constant, 1e-9)
        )
        target = maximum * charge_gate
        return max(retained, retained + alpha * (target - retained))

    def _body_derived_feedforward_cap(
        self,
        measured_error: float,
        projected_error: float,
        projected_velocity: float,
        *,
        fresh_motion_confidence: float = 0.0,
    ) -> float:
        """Return ordinary authority plus a stop-safe fast-pursuit reserve."""

        # A physical passthrough-mouse correction is not present in the MAKCU
        # command ledger and can therefore look like target velocity. When
        # that retained velocity is closing or has crossed either the fresh or
        # command-aware residual, do not let even the historical 25% baseline
        # suppress the user's correction or carry a stop/reversal. Aligned
        # pursuit retains the existing 25-to-configured ramp bit-for-bit.
        configured = self.config.maximum_body_derived_feedforward_fraction
        baseline = min(
            configured,
            _BODY_DERIVED_BASELINE_FEEDFORWARD_FRACTION,
        )
        ordinary_cap = 0.0
        if (
            measured_error * projected_velocity > 0.0
            and projected_error * projected_velocity > 0.0
        ):
            ordinary_cap = baseline + (configured - baseline) * (
                self._body_derived_feedforward_error_gate(
                    measured_error,
                    projected_error,
                    projected_velocity,
                )
            )
        pursuit_maximum = (
            self.config.maximum_body_derived_pursuit_feedforward_fraction
        )
        if pursuit_maximum <= ordinary_cap or fresh_motion_confidence <= 0.0:
            return ordinary_cap
        direction = 1.0 if projected_velocity > 0.0 else -1.0
        least_aligned_error = min(
            measured_error * direction,
            projected_error * direction,
        )
        residual_gate = self._ramp(
            least_aligned_error,
            _BODY_DERIVED_PURSUIT_ZERO_OPPOSED_ERROR_PIXELS,
            _BODY_DERIVED_PURSUIT_FULL_OPPOSED_ERROR_PIXELS,
        )
        speed_gate = self._ramp(
            abs(projected_velocity),
            _BODY_DERIVED_PURSUIT_ZERO_SPEED_PIXELS_PER_SECOND,
            _BODY_DERIVED_PURSUIT_FULL_SPEED_PIXELS_PER_SECOND,
        )
        reserve_gate = min(max(fresh_motion_confidence, 0.0), 1.0)
        reserve_gate *= residual_gate * speed_gate
        return ordinary_cap + reserve_gate * (pursuit_maximum - ordinary_cap)

    @staticmethod
    def _bounded_diagnostic(value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            return 0.0
        return min(value, _MAXIMUM_OBSERVER_DIAGNOSTIC)

    @staticmethod
    def _filtered_velocity(
        previous: float,
        robust_sample: float,
        alpha: float,
        acceleration_step: float,
        speed_limit: float,
    ) -> float:
        candidate = previous + (robust_sample - previous) * alpha
        change = min(max(candidate - previous, -acceleration_step), acceleration_step)
        return min(max(previous + change, -speed_limit), speed_limit)

    def _axis_rate(
        self,
        projected_error: float,
        velocity: float,
        gain: float,
        limit: float,
    ) -> tuple[float, bool]:
        feedback = 0.0
        if abs(projected_error) > self.config.feedback_deadzone_pixels:
            feedback = self._feedback_error(projected_error) / (
                gain * self.config.position_time_constant_seconds
            )
        feed_forward = velocity / gain
        requested = feedback + feed_forward
        if (
            abs(projected_error) > self.config.wrong_way_guard_pixels
            and requested * projected_error < 0.0
        ):
            # A retained velocity estimate may briefly lag a stop/reversal, but
            # it may not override a meaningful fresh residual and run away in
            # the wrong direction.  The positional term remains available.
            requested = feedback
        bounded = min(max(requested, -limit), limit)
        return bounded, bounded != requested

    def _feedback_error(self, projected_error: float) -> float:
        """Return legacy hard or continuous deadband feedback error."""

        if not self.config.continuous_feedback_deadband:
            return projected_error
        magnitude = max(
            abs(projected_error) - self.config.feedback_deadzone_pixels,
            0.0,
        )
        if magnitude == 0.0:
            return 0.0
        shoulder = self.config.continuous_feedback_shoulder_pixels
        if shoulder > 0.0 and magnitude < shoulder:
            normalized = magnitude / shoulder
            magnitude *= normalized * normalized * (3.0 - 2.0 * normalized)
        return math.copysign(magnitude, projected_error)

    def _counts_landing_between(self, start_ns: int, end_ns: int) -> tuple[int, int]:
        count_x = 0
        count_y = 0
        for command in self._commands:
            impact_ns = command.timestamp_ns + self._delay_ns
            if impact_ns <= start_ns:
                continue
            if impact_ns > end_ns:
                break
            count_x += command.delta_x_counts
            count_y += command.delta_y_counts
        physical_x, physical_y = self._physical_counts_landing_between(
            start_ns,
            end_ns,
        )
        count_x += physical_x
        count_y += physical_y
        return count_x, count_y


    def _physical_counts_landing_between(
        self,
        start_ns: int,
        end_ns: int,
    ) -> tuple[int, int]:
        count_x = 0
        count_y = 0
        for report in self._physical_inputs:
            impact_ns = report.timestamp_ns + self._delay_ns
            if impact_ns <= start_ns:
                continue
            if impact_ns > end_ns:
                break
            count_x += report.delta_x_counts
            count_y += report.delta_y_counts
        return count_x, count_y

    def _physical_axes_landing_between(
        self,
        start_ns: int,
        end_ns: int,
    ) -> tuple[bool, bool]:
        active_x = False
        active_y = False
        for report in self._physical_inputs:
            impact_ns = report.timestamp_ns + self._delay_ns
            if impact_ns <= start_ns:
                continue
            if impact_ns > end_ns:
                break
            active_x = active_x or report.delta_x_counts != 0
            active_y = active_y or report.delta_y_counts != 0
        return active_x, active_y

    def _prune_commands(self, reflected_through_ns: int) -> None:
        while self._commands:
            impact_ns = self._commands[0].timestamp_ns + self._delay_ns
            if impact_ns > reflected_through_ns:
                break
            self._commands.popleft()
        while self._physical_inputs:
            impact_ns = self._physical_inputs[0].timestamp_ns + self._delay_ns
            if impact_ns > reflected_through_ns:
                break
            self._physical_inputs.popleft()
