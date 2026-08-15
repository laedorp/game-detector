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
    maximum_command_history: int = 4096

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

    ``error_*`` is the position-control observation and may therefore be a
    tracker-smoothed point.  The optional ``velocity_error_*`` pair supplies
    the accepted raw point from that same target and source timestamp.  A raw
    velocity channel keeps the exact command ledger in the same measurement
    domain as the plant calibration instead of differentiating a downstream
    smoothing filter.  Omitting both optional fields preserves the historical
    single-channel estimator exactly.
    """

    timestamp_ns: int
    error_x_pixels: float
    error_y_pixels: float
    velocity_error_x_pixels: float | None = None
    velocity_error_y_pixels: float | None = None

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
        self._ready = False
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        window = self.config.velocity_median_window
        self._raw_velocity_x: deque[float] = deque(maxlen=window)
        self._raw_velocity_y: deque[float] = deque(maxlen=window)
        # The paired raw channel estimates one 2-D trajectory over shared
        # timestamps. Keep two samples even when the legacy median window is
        # one so an adjacent secant remains defined.
        self._paired_velocity_history: deque[tuple[int, float, float]] = deque(
            maxlen=max(2, window)
        )
        self._paired_position_x = 0.0
        self._paired_position_y = 0.0

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
            discontinuity = self._accept_observation(observation)
            if discontinuity is not None:
                return self._zero(current_ns, discontinuity)

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
        projected_x = (
            latest.error_x_pixels
            + self._velocity_x * horizon_seconds
            - self.plant.gain_x_pixels_per_count * pending_x
        )
        projected_y = (
            latest.error_y_pixels
            + self._velocity_y * horizon_seconds
            - self.plant.gain_y_pixels_per_count * pending_y
        )
        if not all(
            math.isfinite(value)
            for value in (projected_x, projected_y, self._velocity_x, self._velocity_y)
        ):
            self._reset_tracking()
            return self._zero(current_ns, "non-finite-control-state")

        rate_x, saturated_x = self._axis_rate(
            projected_x,
            self._velocity_x,
            self.plant.gain_x_pixels_per_count,
            self.config.maximum_rate_x_counts_per_second,
        )
        rate_y, saturated_y = self._axis_rate(
            projected_y,
            self._velocity_y,
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
            target_velocity_x_pixels_per_second=self._velocity_x,
            target_velocity_y_pixels_per_second=self._velocity_y,
            projected_error_x_pixels=projected_x,
            projected_error_y_pixels=projected_y,
            valid=True,
            saturated_x=saturated_x,
            saturated_y=saturated_y,
        )

    def _reset_tracking(self) -> None:
        self._last_observation = None
        self._ready = False
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._raw_velocity_x.clear()
        self._raw_velocity_y.clear()
        self._paired_velocity_history.clear()
        self._paired_position_x = 0.0
        self._paired_position_y = 0.0

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

        count_x, count_y = self._counts_landing_between(
            previous.timestamp_ns,
            observation.timestamp_ns,
        )
        velocity_error_x = (
            observation.velocity_error_x_pixels
            if paired_velocity
            else observation.error_x_pixels
        )
        velocity_error_y = (
            observation.velocity_error_y_pixels
            if paired_velocity
            else observation.error_y_pixels
        )
        previous_velocity_error_x = (
            previous.velocity_error_x_pixels
            if previous_paired_velocity
            else previous.error_x_pixels
        )
        previous_velocity_error_y = (
            previous.velocity_error_y_pixels
            if previous_paired_velocity
            else previous.error_y_pixels
        )
        assert velocity_error_x is not None
        assert velocity_error_y is not None
        assert previous_velocity_error_x is not None
        assert previous_velocity_error_y is not None
        target_delta_x = (
            velocity_error_x
            - previous_velocity_error_x
            + self.plant.gain_x_pixels_per_count * count_x
        )
        target_delta_y = (
            velocity_error_y
            - previous_velocity_error_y
            + self.plant.gain_y_pixels_per_count * count_y
        )
        jump = self.config.maximum_error_jump_pixels
        if abs(target_delta_x) > jump or abs(target_delta_y) > jump:
            self._seed_observation(observation)
            return "error-jump"

        speed_limit = self.config.maximum_target_speed_pixels_per_second
        if paired_velocity:
            self._paired_position_x += target_delta_x
            self._paired_position_y += target_delta_y
            self._paired_velocity_history.append(
                (
                    observation.timestamp_ns,
                    self._paired_position_x,
                    self._paired_position_y,
                )
            )
            robust_x, robust_y = self._paired_velocity_regression(speed_limit)
        else:
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
        self._ready = True
        self._prune_commands(observation.timestamp_ns)
        return None

    def _seed_observation(self, observation: ScreenErrorObservation) -> None:
        self._reset_tracking()
        self._last_observation = observation
        if observation.velocity_error_x_pixels is not None:
            self._paired_velocity_history.append(
                (observation.timestamp_ns, 0.0, 0.0)
            )
        self._prune_commands(observation.timestamp_ns)

    def _paired_velocity_regression(
        self,
        speed_limit: float,
    ) -> tuple[float, float]:
        """Fit one coherent X/Y velocity to shared raw-point timestamps.

        Ordinary least squares over reconstructed target displacement rejects
        adjacent-frame derivative noise without selecting X from one frame and
        Y from another.  A shared two-dimensional explained-variance gate
        suppresses fits dominated by curved/rotating point noise while giving
        a sufficiently linear trajectory its full slope.  A steady physical
        motion therefore keeps its feed-forward even when position error is
        zero.
        """

        history = tuple(self._paired_velocity_history)
        if len(history) < 2:
            return 0.0, 0.0
        origin_ns = history[0][0]
        times = [
            (timestamp_ns - origin_ns) / _NS_PER_SECOND
            for timestamp_ns, _x, _y in history
        ]
        mean_time = sum(times) / len(times)
        mean_x = sum(item[1] for item in history) / len(history)
        mean_y = sum(item[2] for item in history) / len(history)
        denominator = sum((value - mean_time) ** 2 for value in times)
        if denominator <= 0.0:
            return 0.0, 0.0
        slope_x = sum(
            (time_value - mean_time) * (item[1] - mean_x)
            for time_value, item in zip(times, history)
        ) / denominator
        slope_y = sum(
            (time_value - mean_time) * (item[2] - mean_y)
            for time_value, item in zip(times, history)
        ) / denominator

        total_variation = sum(
            (item[1] - mean_x) ** 2 + (item[2] - mean_y) ** 2
            for item in history
        )
        if total_variation <= 1e-12:
            return 0.0, 0.0
        residual_variation = sum(
            (
                item[1]
                - (mean_x + slope_x * (time_value - mean_time))
            )
            ** 2
            + (
                item[2]
                - (mean_y + slope_y * (time_value - mean_time))
            )
            ** 2
            for time_value, item in zip(times, history)
        )
        explained_fraction = min(
            max(1.0 - residual_variation / total_variation, 0.0),
            1.0,
        )
        # R² itself is not a suitable gain: multiplying a real noisy pursuit
        # by (for example) 0.75 creates avoidable lag.  Below 0.15 the samples
        # do not establish one velocity vector; above 0.50 the slope is kept
        # whole.  The bounded transition prevents a hard chatter threshold.
        coherence = min(max((explained_fraction - 0.15) / 0.35, 0.0), 1.0)
        slope_x *= coherence
        slope_y *= coherence

        return (
            min(max(slope_x, -speed_limit), speed_limit),
            min(max(slope_y, -speed_limit), speed_limit),
        )

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
