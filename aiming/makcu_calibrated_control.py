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
_CORROBORATION_RISE_TIME_CONSTANT_SECONDS = 0.050
_CORROBORATION_FALL_TIME_CONSTANT_SECONDS = 0.010
# Positional lead is a short, closed-loop horizon rather than an open-loop
# velocity command. Once independent evidence has reached half confidence it
# may use that bounded horizon fully; explicit feed-forward remains scaled by
# the unmodified, slower confidence. This preserves prompt pursuit without
# weakening the exact-zero revoke boundary.
_CORROBORATION_FULL_MOTION_PROJECTION_CONFIDENCE = 0.50
# A body-mapped head coordinate and the body box which translated it are one
# item of evidence, not two independent motion measurements.  A recent direct-
# head identity lease may authorize source-age projection of that same mapped
# point, while its additional open-loop velocity command remains deliberately
# bounded by this hard ceiling.
_MAXIMUM_BODY_DERIVED_FEEDFORWARD_FRACTION = 0.25
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
    stale_after_seconds: float = 0.040
    maximum_observation_interval_seconds: float = 0.040
    maximum_error_jump_pixels: float = 180.0
    feedback_deadzone_pixels: float = 0.50
    wrong_way_guard_pixels: float = 2.0
    velocity_median_window: int = 3
    maximum_velocity_feedforward_fraction: float = 1.0
    require_motion_corroboration_for_feedforward: bool = False
    maximum_command_history: int = 4096
    maximum_body_derived_projection_fraction: float = 0.0
    maximum_body_derived_feedforward_fraction: float = 0.0

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
        for name in ("feedback_deadzone_pixels", "wrong_way_guard_pixels"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
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
                "maximum_body_derived_feedforward_fraction must be between zero and 0.25"
            )
        object.__setattr__(
            self,
            "maximum_body_derived_feedforward_fraction",
            body_derived_feedforward_fraction,
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
    innovation_mahalanobis_squared: float = 0.0
    innovation_rejected: bool = False
    motion_corroboration_confidence: float = 0.0


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
        self._motion_corroboration_confidence = 0.0
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
        self._body_derived_motion_confidence = 0.0
        self._body_derived_direction_anchor_x = 0.0
        self._body_derived_direction_anchor_y = 0.0
        self._body_derived_direction_persistence_seconds = 0.0

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
        self._motion_corroboration_confidence = 0.0
        self._corroboration_direction_anchor_x = 0.0
        self._corroboration_direction_anchor_y = 0.0
        self._corroboration_direction_persistence_seconds = 0.0
        self._corroboration_reseed_required = True
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

    def _clear_body_derived_motion(self) -> None:
        self._body_derived_motion_permitted = False
        self._body_derived_motion_deadline_ns = None
        self._body_derived_motion_confidence = 0.0
        self._body_derived_direction_anchor_x = 0.0
        self._body_derived_direction_anchor_y = 0.0
        self._body_derived_direction_persistence_seconds = 0.0

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

    def step(
        self,
        now_ns: int,
        *,
        engaged: bool,
        observation: ScreenErrorObservation | None = None,
        target_lost: bool = False,
        observation_expected: bool | None = None,
        emitted_commands: Iterable[EmittedMouseCommand] = (),
    ) -> CalibratedControlOutput:
        """Ingest new physical evidence and return a bounded X/Y count rate."""

        current_ns = _timestamp(now_ns, "control timestamp")
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
        if target_lost and observation is not None:
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

        if observation is not None:
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
            discontinuity = self._accept_observation(observation)
            if discontinuity is not None:
                return self._zero(current_ns, discontinuity)

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
        horizon_seconds = (horizon_ns - latest.timestamp_ns) / _NS_PER_SECOND
        paired_observer = latest.velocity_error_x_pixels is not None
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
            if self.config.require_motion_corroboration_for_feedforward:
                if self._body_derived_motion_permitted:
                    # These are explicit provenance grants, not synthesized
                    # corroboration.  A high-confidence paired observer cannot
                    # multiply or otherwise bypass either configured ceiling.
                    # Projection additionally retains the paired observer's
                    # covariance/signal confidence: a bounded circular point
                    # wobble does not become full source-age translation merely
                    # because its anatomical identity lease is still valid.
                    motion_projection_confidence = (
                        self.config.maximum_body_derived_projection_fraction
                        * self._body_derived_motion_confidence
                    )
                    motion_projection_confidence_x = (
                        motion_projection_confidence
                    )
                    motion_projection_confidence_y = (
                        motion_projection_confidence
                    )
                    feedforward_confidence_x = min(
                        feedforward_confidence_x,
                        self.config.maximum_body_derived_feedforward_fraction,
                    ) * self._body_derived_motion_confidence
                    feedforward_confidence_y = min(
                        feedforward_confidence_y,
                        self.config.maximum_body_derived_feedforward_fraction,
                    ) * self._body_derived_motion_confidence
                else:
                    feedforward_confidence_x *= (
                        self._motion_corroboration_confidence
                    )
                    feedforward_confidence_y *= (
                        self._motion_corroboration_confidence
                    )
                    motion_projection_confidence = min(
                        self._motion_corroboration_confidence
                        / _CORROBORATION_FULL_MOTION_PROJECTION_CONFIDENCE,
                        1.0,
                    )
                    motion_projection_confidence_x = (
                        motion_projection_confidence
                    )
                    motion_projection_confidence_y = (
                        motion_projection_confidence
                    )

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
                if self._body_derived_motion_permitted:
                    static_projected_x = (
                        latest.error_x_pixels
                        - self.plant.gain_x_pixels_per_count * pending_x
                    )
                    static_projected_y = (
                        latest.error_y_pixels
                        - self.plant.gain_y_pixels_per_count * pending_y
                    )
                    full_projected_x = (
                        static_projected_x
                        + projected_velocity_x * horizon_seconds
                    )
                    full_projected_y = (
                        static_projected_y
                        + projected_velocity_y * horizon_seconds
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
                projected_x = static_projected_x + (
                    motion_projection_confidence_x
                    * (full_projected_x - static_projected_x)
                )
                projected_y = static_projected_y + (
                    motion_projection_confidence_y
                    * (full_projected_y - static_projected_y)
                )
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

        if paired_observer:
            rate_x, saturated_x = self._paired_axis_rate(
                projected_x,
                projected_velocity_x,
                self.plant.gain_x_pixels_per_count,
                self.config.maximum_rate_x_counts_per_second,
                feedforward_confidence_x,
                feedback_held=self._paired_feedback_hold_x,
                position_confidence=self._paired_measurement_agreement_x,
            )
            rate_y, saturated_y = self._paired_axis_rate(
                projected_y,
                projected_velocity_y,
                self.plant.gain_y_pixels_per_count,
                self.config.maximum_rate_y_counts_per_second,
                feedforward_confidence_y,
                feedback_held=self._paired_feedback_hold_y,
                position_confidence=self._paired_measurement_agreement_y,
            )
        else:
            rate_x, saturated_x = self._axis_rate(
                projected_x,
                projected_velocity_x,
                self.plant.gain_x_pixels_per_count,
                self.config.maximum_rate_x_counts_per_second,
            )
            rate_y, saturated_y = self._axis_rate(
                projected_y,
                projected_velocity_y,
                self.plant.gain_y_pixels_per_count,
                self.config.maximum_rate_y_counts_per_second,
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
            innovation_mahalanobis_squared=self._bounded_diagnostic(
                self._last_innovation_mahalanobis_squared
            ),
            innovation_rejected=self._last_innovation_rejected,
            motion_corroboration_confidence=(
                self._motion_corroboration_confidence
            ),
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
        self._motion_corroboration_confidence = 0.0
        self._corroboration_direction_anchor_x = 0.0
        self._corroboration_direction_anchor_y = 0.0
        self._corroboration_direction_persistence_seconds = 0.0
        self._corroboration_reseed_required = False
        self._body_derived_motion_permitted = False
        self._body_derived_motion_deadline_ns = None
        self._body_derived_motion_confidence = 0.0
        self._body_derived_direction_anchor_x = 0.0
        self._body_derived_direction_anchor_y = 0.0
        self._body_derived_direction_persistence_seconds = 0.0

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
        if len(self._commands) + len(records) > self.config.maximum_command_history:
            return "command-history-overflow"
        return None

    def _accept_observation(self, observation: ScreenErrorObservation) -> str | None:
        # Authorization belongs to one accepted measured sample.  Clear the
        # former sample before evaluating this one so absence, a discontinuity,
        # or an innovation rejection all fail closed immediately.
        self._body_derived_motion_permitted = False
        self._body_derived_motion_deadline_ns = None
        self._body_derived_motion_confidence = 0.0
        if not observation.body_derived_motion_permitted:
            self._clear_body_derived_motion()
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
            self._seed_observation(observation)
            return "corroboration-channel-change"

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
        # The downstream tracker point is not differentiated and never owns a
        # second control state. It is only an independent plausibility signal:
        # a one-model-pixel box-edge staircase which moves the raw point while
        # the tracked position stays fixed must not earn feed-forward trust.
        measurement_agreement = self._paired_channel_agreement(
            measured_x,
            measured_y,
            observation.error_x_pixels,
            observation.error_y_pixels,
        )
        self._paired_measurement_agreement_x = measurement_agreement
        self._paired_measurement_agreement_y = measurement_agreement
        disagreement_x = measured_x - observation.error_x_pixels
        disagreement_y = measured_y - observation.error_y_pixels
        count_x, count_y = self._counts_landing_between(
            previous.timestamp_ns,
            observation.timestamp_ns,
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
        self._update_motion_corroboration(
            observation,
            elapsed=elapsed,
            landed_x_pixels=self.plant.gain_x_pixels_per_count * count_x,
            landed_y_pixels=self.plant.gain_y_pixels_per_count * count_y,
        )
        self._body_derived_motion_permitted = (
            observation.body_derived_motion_permitted
        )
        if self._body_derived_motion_permitted:
            deadline_ns = observation.body_derived_motion_deadline_ns
            assert deadline_ns is not None
            self._body_derived_motion_deadline_ns = deadline_ns
            self._update_body_derived_motion_confidence(elapsed)
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
                self._corroboration_position_x = (
                    observation.corroboration_error_x_pixels
                )
                self._corroboration_position_y = (
                    observation.corroboration_error_y_pixels
                )
                corroboration_variance = (
                    _CORROBORATION_MEASUREMENT_SIGMA_PIXELS**2
                )
                corroboration_covariance = (
                    corroboration_variance,
                    0.0,
                    initial_velocity_sigma * initial_velocity_sigma,
                )
                self._corroboration_covariance_x = corroboration_covariance
                self._corroboration_covariance_y = corroboration_covariance
        self._prune_commands(observation.timestamp_ns)

    def _update_motion_corroboration(
        self,
        observation: ScreenErrorObservation,
        *,
        elapsed: float,
        landed_x_pixels: float,
        landed_y_pixels: float,
    ) -> None:
        """Update independent translation evidence without moving the aim point."""

        measured_x = observation.corroboration_error_x_pixels
        measured_y = observation.corroboration_error_y_pixels
        if measured_x is None or measured_y is None:
            self._motion_corroboration_confidence = 0.0
            return
        if self._corroboration_reseed_required:
            self._seed_corroboration(measured_x, measured_y)
            self._motion_corroboration_confidence = 0.0
            return

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
            self._seed_corroboration(measured_x, measured_y)
            self._motion_corroboration_confidence = 0.0
            return

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
        raw_confidence = self._motion_corroboration(
            self._velocity_x,
            self._velocity_y,
            math.sqrt(self._paired_covariance_x[2]),
            math.sqrt(self._paired_covariance_y[2]),
            self._corroboration_velocity_x,
            self._corroboration_velocity_y,
            math.sqrt(self._corroboration_covariance_x[2]),
            math.sqrt(self._corroboration_covariance_y[2]),
        )
        raw_confidence *= self._direction_persistence_confidence(
            self._velocity_x,
            self._velocity_y,
            self._corroboration_velocity_x,
            self._corroboration_velocity_y,
            elapsed,
            evidence_present=raw_confidence > 0.0,
        )
        time_constant = (
            _CORROBORATION_RISE_TIME_CONSTANT_SECONDS
            if raw_confidence > self._motion_corroboration_confidence
            else _CORROBORATION_FALL_TIME_CONSTANT_SECONDS
        )
        alpha = 1.0 - math.exp(-elapsed / time_constant)
        self._motion_corroboration_confidence += alpha * (
            raw_confidence - self._motion_corroboration_confidence
        )

    def _seed_corroboration(self, measured_x: float, measured_y: float) -> None:
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

    def _update_body_derived_motion_confidence(self, elapsed: float) -> None:
        """Authorize only statistically coherent, persistent mapped motion.

        This uses the mapped point's own paired observer and therefore is not
        independent corroboration.  It cannot raise the independent diagnostic
        or exceed either configured ceiling.  It distinguishes sustained
        translation (where source-age projection and bounded FF are useful)
        from a bounded anatomical point orbit (where prediction would magnify
        a few pixels of jitter into a visible circle).
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
        signal_confidence = min(
            max(max(source_confidence_x, source_confidence_y), 0.0),
            1.0,
        )
        magnitude = math.hypot(self._velocity_x, self._velocity_y)
        if signal_confidence <= 0.0 or magnitude <= 1e-9:
            self._body_derived_motion_confidence = 0.0
            self._body_derived_direction_anchor_x = 0.0
            self._body_derived_direction_anchor_y = 0.0
            self._body_derived_direction_persistence_seconds = 0.0
            return
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
            self._body_derived_motion_confidence = 0.0
            return
        alignment = (
            direction_x * self._body_derived_direction_anchor_x
            + direction_y * self._body_derived_direction_anchor_y
        )
        if alignment < _BODY_DERIVED_MINIMUM_DIRECTION_COSINE:
            self._body_derived_direction_anchor_x = direction_x
            self._body_derived_direction_anchor_y = direction_y
            self._body_derived_direction_persistence_seconds = 0.0
            self._body_derived_motion_confidence = 0.0
            return
        self._body_derived_direction_persistence_seconds += elapsed
        persistence_confidence = self._ramp(
            self._body_derived_direction_persistence_seconds,
            _CORROBORATION_ZERO_DIRECTION_PERSISTENCE_SECONDS,
            _CORROBORATION_FULL_DIRECTION_PERSISTENCE_SECONDS,
        )
        self._body_derived_motion_confidence = (
            signal_confidence * persistence_confidence
        )

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
    ) -> float:
        # Treat X/Y as one detector point.  Independent axis confidence lets a
        # quantized point walk around a square while alternately trusting the
        # quiet-looking axis; radial agreement closes that rotating loophole.
        disagreement = math.hypot(raw_x - tracked_x, raw_y - tracked_y)
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
        feedback_held: bool,
        position_confidence: float,
    ) -> tuple[float, bool]:
        feedback = 0.0
        if abs(projected_error) > self.config.feedback_deadzone_pixels:
            feedback_fraction = (
                _PAIRED_HELD_FEEDBACK_FRACTION if feedback_held else 1.0
            )
            feedback = position_confidence * feedback_fraction * projected_error / (
                gain * self.config.position_time_constant_seconds
            )
        feed_forward = feedforward_confidence * velocity / gain
        requested = feedback + feed_forward
        if (
            abs(projected_error) > self.config.wrong_way_guard_pixels
            and requested * projected_error < 0.0
        ):
            requested = feedback
        bounded = min(max(requested, -limit), limit)
        return bounded, bounded != requested

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
            feedback = projected_error / (
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
        return count_x, count_y

    def _prune_commands(self, reflected_through_ns: int) -> None:
        while self._commands:
            impact_ns = self._commands[0].timestamp_ns + self._delay_ns
            if impact_ns > reflected_through_ns:
                break
            self._commands.popleft()
