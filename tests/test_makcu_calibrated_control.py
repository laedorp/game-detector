from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
import unittest
from unittest.mock import patch

from aiming.controller import TargetTracker, head_target_point
from aiming.makcu_calibrated_control import (
    CalibratedControlConfig,
    CalibratedControlOutput,
    CalibratedPlant,
    CorrelatedLookaheadObservation,
    EmittedMouseCommand,
    MakcuCalibratedController,
    ScreenErrorObservation,
)
from detection.types import Detection


NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000


def _test_config(**overrides: object) -> CalibratedControlConfig:
    values: dict[str, object] = {
        "position_time_constant_seconds": 0.060,
        "velocity_filter_time_constant_seconds": 0.018,
        "maximum_target_speed_pixels_per_second": 3000.0,
        "maximum_target_acceleration_pixels_per_second_squared": 90_000.0,
        "maximum_rate_x_counts_per_second": 16_000.0,
        "maximum_rate_y_counts_per_second": 10_000.0,
        "stale_after_seconds": 0.040,
        "maximum_observation_interval_seconds": 0.040,
        "maximum_error_jump_pixels": 180.0,
        "feedback_deadzone_pixels": 0.50,
        "wrong_way_guard_pixels": 2.0,
        "velocity_median_window": 3,
    }
    values.update(overrides)
    return CalibratedControlConfig(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class _RunResult:
    errors: tuple[tuple[float, float], ...]
    outputs: tuple[CalibratedControlOutput | tuple[float, float], ...]


class _CurrentProportionalPi:
    """Small numeric reproduction of the current proportional/pursuit PI law."""

    def __init__(self, config: CalibratedControlConfig) -> None:
        self.config = config
        self.error_x = 0.0
        self.error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_sample_ns: int | None = None
        self.last_observed_ns: int | None = None
        self.ready = False

    def step(
        self,
        now_ns: int,
        observation: ScreenErrorObservation | None,
    ) -> tuple[float, float]:
        if observation is not None:
            elapsed = 0.0
            if self.last_sample_ns is not None:
                elapsed = (observation.timestamp_ns - self.last_sample_ns) / NS_PER_SECOND
            self.last_sample_ns = observation.timestamp_ns
            self.last_observed_ns = observation.timestamp_ns
            self.error_x = observation.error_x_pixels
            self.error_y = observation.error_y_pixels
            self.ready = True
            self.integral_x = self._integrate(
                self.error_x,
                self.integral_x,
                elapsed,
                self.config.maximum_rate_x_counts_per_second,
            )
            self.integral_y = self._integrate(
                self.error_y,
                self.integral_y,
                elapsed,
                self.config.maximum_rate_y_counts_per_second,
            )
        if (
            not self.ready
            or self.last_observed_ns is None
            or now_ns - self.last_observed_ns
            > round(self.config.stale_after_seconds * NS_PER_SECOND)
        ):
            self.integral_x = 0.0
            self.integral_y = 0.0
            self.ready = False
            return 0.0, 0.0
        return (
            self._rate(
                self.error_x,
                self.integral_x,
                self.config.maximum_rate_x_counts_per_second,
            ),
            self._rate(
                self.error_y,
                self.integral_y,
                self.config.maximum_rate_y_counts_per_second,
            ),
        )

    @staticmethod
    def _integrate(error: float, accumulated: float, elapsed: float, limit: float) -> float:
        # Current production tuning: strength 1.28 at a 60 Hz reference,
        # pursuit buildup over 120 ms, with a 50% output-limit integral cap.
        increment = error * 1.28 * 60.0 * max(elapsed, 0.0) / 0.12
        base = error * 1.28 * 60.0
        if increment > 0.0 and base + accumulated < limit:
            accumulated += min(increment, limit - base - accumulated)
        elif increment < 0.0 and base + accumulated > -limit:
            accumulated += max(increment, -limit - base - accumulated)
        return min(max(accumulated, -limit * 0.50), limit * 0.50)

    @staticmethod
    def _rate(error: float, integral: float, limit: float) -> float:
        base = error * 1.28 * 60.0
        requested = base + integral
        if abs(error) > 2.0 and requested * error < 0.0:
            requested = base
        return min(max(requested, -limit), limit)


def _target_velocity(elapsed: float) -> tuple[float, float]:
    if elapsed < 0.65:
        return 760.0, -380.0
    if elapsed < 0.92:
        return 0.0, 0.0
    if elapsed < 1.58:
        return -690.0, 440.0
    if elapsed < 1.86:
        return 0.0, 0.0
    if elapsed < 2.45:
        return 420.0, -250.0
    return 0.0, 0.0


def _observation_noise(index: int) -> tuple[float, float]:
    return (
        0.55 * math.sin(index * 1.73) + 0.22 * math.sin(index * 0.37),
        0.48 * math.sin(index * 1.21 + 0.4) + 0.18 * math.sin(index * 0.29),
    )


def _run_fake_plant(
    delay_ms: int,
    *,
    calibrated: bool,
    duration_ms: int = 3000,
    loss_after_ms: int | None = None,
    physical_gain_x: float = 0.075,
    physical_gain_y: float = 0.14,
    controller_plant: CalibratedPlant | None = None,
    control_config: CalibratedControlConfig | None = None,
) -> _RunResult:
    """Run a deterministic unequal-axis, delayed, integer-command plant."""

    plant = CalibratedPlant(
        physical_gain_x,
        physical_gain_y,
        delay_ms / 1000.0,
    )
    config = control_config or _test_config()
    controller = (
        MakcuCalibratedController(controller_plant or plant, config)
        if calibrated
        else None
    )
    baseline = None if calibrated else _CurrentProportionalPi(config)
    error_x = 95.0
    error_y = -62.0
    fractional_x = 0.0
    fractional_y = 0.0
    # (visible timestamp, delta X, delta Y)
    delayed: list[tuple[int, int, int]] = []
    delayed_index = 0
    prior_emitted: tuple[EmittedMouseCommand, ...] = ()
    observation_period_ns = round(NS_PER_SECOND / 130.0)
    next_observation_ns = 0
    observation_index = 0
    skipped_observations = {57, 58, 173, 291, 292}
    errors: list[tuple[float, float]] = []
    outputs: list[CalibratedControlOutput | tuple[float, float]] = []

    for tick in range(duration_ms):
        now_ns = tick * NS_PER_MS
        if tick:
            velocity_x, velocity_y = _target_velocity(tick / 1000.0)
            error_x += velocity_x * 0.001
            error_y += velocity_y * 0.001
        while delayed_index < len(delayed) and delayed[delayed_index][0] <= now_ns:
            _impact_ns, delta_x, delta_y = delayed[delayed_index]
            delayed_index += 1
            error_x -= plant.gain_x_pixels_per_count * delta_x
            error_y -= plant.gain_y_pixels_per_count * delta_y

        observation: ScreenErrorObservation | None = None
        if now_ns >= next_observation_ns:
            noise_x, noise_y = _observation_noise(observation_index)
            if (
                observation_index not in skipped_observations
                and (loss_after_ms is None or tick < loss_after_ms)
            ):
                observation = ScreenErrorObservation(
                    now_ns,
                    error_x + noise_x,
                    error_y + noise_y,
                )
            observation_index += 1
            next_observation_ns += observation_period_ns

        if controller is not None:
            output = controller.step(
                now_ns,
                engaged=True,
                observation=observation,
                emitted_commands=prior_emitted,
            )
            rate_x = output.rate_x_counts_per_second
            rate_y = output.rate_y_counts_per_second
            outputs.append(output)
        else:
            assert baseline is not None
            rate_x, rate_y = baseline.step(now_ns, observation)
            outputs.append((rate_x, rate_y))

        fractional_x += rate_x * 0.001
        fractional_y += rate_y * 0.001
        delta_x = math.trunc(fractional_x)
        delta_y = math.trunc(fractional_y)
        fractional_x -= delta_x
        fractional_y -= delta_y
        if delta_x or delta_y:
            delayed.append((now_ns + delay_ms * NS_PER_MS, delta_x, delta_y))
            prior_emitted = (
                EmittedMouseCommand(now_ns, delta_x, delta_y),
            )
        else:
            prior_emitted = ()
        errors.append((error_x, error_y))
    return _RunResult(tuple(errors), tuple(outputs))


@dataclass(frozen=True, slots=True)
class _TrackedJitterRunResult:
    radial_errors: tuple[float, ...]
    estimated_speeds: tuple[float, ...]
    steady_abs_counts_per_second: float
    maximum_requested_axis_rate: float


@dataclass(frozen=True, slots=True)
class _SmoothedFeedbackRunResult:
    radial_errors: tuple[float, ...]
    estimated_speeds: tuple[float, ...]
    emitted_counts: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class _DirectPointRunResult:
    moving_rms_pixels: float
    moving_p95_pixels: float
    reversal_rms_pixels: float
    maximum_reversal_error_pixels: float
    stationary_rms_pixels: float
    stationary_p95_pixels: float
    steady_abs_counts_per_second: float
    maximum_requested_axis_rate: float
    saturated_output_fraction: float = 0.0
    moving_mean_lag_pixels: float = 0.0
    moving_p95_lag_pixels: float = 0.0
    moving_mean_lag_ms: float = 0.0
    moving_p95_lag_ms: float = 0.0
    maximum_post_stop_overshoot_pixels: float = 0.0
    post_stop_rms_pixels: float = 0.0
    post_stop_p95_pixels: float = 0.0
    maximum_moving_motion_corroboration_confidence: float = 0.0


def _run_tracked_jitter_plant(
    controller: MakcuCalibratedController,
    *,
    raw_velocity_channel: bool = False,
    physical_gain_scale: float = 1.0,
) -> _TrackedJitterRunResult:
    """Exercise the real target filter and numeric controller under box jitter."""

    tracker = TargetTracker(label="player", lost_grace_frames=1)
    frame_shape = (1080, 1920, 3)
    error_x = 95.0
    error_y = -40.0
    fractional_x = 0.0
    fractional_y = 0.0
    delayed: list[tuple[int, int, int]] = []
    delayed_index = 0
    observation_index = 0
    radial_errors: list[float] = []
    emitted: list[tuple[int, int, int]] = []
    estimated_speeds: list[float] = []
    maximum_requested_axis_rate = 0.0

    for tick in range(4000):
        now_ns = tick * NS_PER_MS
        elapsed = tick / 1000.0
        if elapsed < 1.2:
            velocity_x, velocity_y = 760.0, -380.0
        elif elapsed < 2.0:
            velocity_x, velocity_y = 0.0, 0.0
        elif elapsed < 3.2:
            velocity_x, velocity_y = -690.0, 440.0
        else:
            velocity_x, velocity_y = 0.0, 0.0
        error_x += velocity_x * 0.001
        error_y += velocity_y * 0.001

        while delayed_index < len(delayed) and delayed[delayed_index][0] <= now_ns:
            _impact_ns, delta_x, delta_y = delayed[delayed_index]
            delayed_index += 1
            error_x -= 0.125 * physical_gain_scale * delta_x
            error_y -= 0.120 * physical_gain_scale * delta_y

        observation = None
        if tick % 8 == 0:
            # Eight reference pixels is representative of the 2--15 px
            # raw-versus-track residuals in the physical 130 Hz run. Mixed
            # frequencies keep this deterministic without creating a single
            # easy-to-notch alternating pattern.
            noise_x = 8.0 * (
                0.70 * math.sin(observation_index * 1.91)
                + 0.30 * math.sin(observation_index * 0.47 + 0.3)
            )
            noise_y = 8.0 * (
                0.72 * math.sin(observation_index * 1.57 + 0.2)
                + 0.28 * math.sin(observation_index * 0.39)
            )
            width = 160.0 + 6.4 * math.sin(observation_index * 1.31)
            height = 390.0 + 12.0 * math.sin(observation_index * 1.73 + 0.5)
            measured_head_x = 960.0 + error_x + noise_x
            measured_head_y = 540.0 + error_y + noise_y
            measured = Detection(
                0,
                "player",
                0.90,
                (
                    measured_head_x - width / 2.0,
                    measured_head_y - 0.12 * height,
                    measured_head_x + width / 2.0,
                    measured_head_y + 0.88 * height,
                ),
            )
            tracked = tracker.update(
                (measured,),
                frame_shape,
                measurement_ns=now_ns,
            )
            assert tracked is not None
            tracked_x, tracked_y = head_target_point(tracked, 0.12)
            observation = ScreenErrorObservation(
                now_ns,
                tracked_x - 960.0,
                tracked_y - 540.0,
                velocity_error_x_pixels=(
                    measured_head_x - 960.0
                    if raw_velocity_channel
                    else None
                ),
                velocity_error_y_pixels=(
                    measured_head_y - 540.0
                    if raw_velocity_channel
                    else None
                ),
            )
            observation_index += 1

        output = controller.step(
            now_ns,
            engaged=True,
            observation=observation,
        )
        maximum_requested_axis_rate = max(
            maximum_requested_axis_rate,
            abs(output.rate_x_counts_per_second),
            abs(output.rate_y_counts_per_second),
        )
        estimated_speeds.append(
            math.hypot(
                output.target_velocity_x_pixels_per_second,
                output.target_velocity_y_pixels_per_second,
            )
        )
        fractional_x += output.rate_x_counts_per_second * 0.001
        fractional_y += output.rate_y_counts_per_second * 0.001
        delta_x = math.trunc(fractional_x)
        delta_y = math.trunc(fractional_y)
        fractional_x -= delta_x
        fractional_y -= delta_y
        if delta_x or delta_y:
            command = EmittedMouseCommand(now_ns, delta_x, delta_y)
            controller.preflight_emitted(command)
            controller.record_emitted(command)
            delayed.append((now_ns + 8 * NS_PER_MS, delta_x, delta_y))
            emitted.append((tick, delta_x, delta_y))
        radial_errors.append(math.hypot(error_x, error_y))

    steady_abs_counts = sum(
        abs(delta_x) + abs(delta_y)
        for tick, delta_x, delta_y in emitted
        if tick >= 3300
    )
    return _TrackedJitterRunResult(
        tuple(radial_errors),
        tuple(estimated_speeds),
        steady_abs_counts / 0.7,
        maximum_requested_axis_rate,
    )


def _run_smoothed_feedback_plant(
    *,
    raw_velocity_channel: bool,
    duration_ms: int = 4000,
    position_time_constant_seconds: float = 0.100,
) -> _SmoothedFeedbackRunResult:
    """Close the exact command ledger around a separately smoothed point.

    The physical target is stationary. Only emitted integer commands move its
    raw screen error. The position channel deliberately follows that raw point
    with a bounded downstream response lag. Combining this delayed coordinate
    with the raw 8 ms command ledger manufactures target motion; the optional
    raw point keeps both sides of the velocity equation in one measurement
    domain.
    """

    plant = CalibratedPlant(0.125, 0.120, 0.008)
    controller = MakcuCalibratedController(
        plant,
        CalibratedControlConfig(
            velocity_median_window=5,
            velocity_filter_time_constant_seconds=0.040,
            maximum_target_acceleration_pixels_per_second_squared=20_000.0,
            stale_after_seconds=0.065,
            maximum_observation_interval_seconds=0.040,
        ),
    )
    raw_error_x = 100.0
    raw_error_y = -60.0
    smoothed_error_x = raw_error_x
    smoothed_error_y = raw_error_y
    fractional_x = 0.0
    fractional_y = 0.0
    delayed: list[tuple[int, int, int]] = []
    delayed_index = 0
    emitted: list[tuple[int, int, int]] = []
    radial_errors: list[float] = []
    estimated_speeds: list[float] = []
    observation_alpha = 1.0 - math.exp(
        -0.008 / position_time_constant_seconds
    )

    for tick in range(duration_ms):
        now_ns = tick * NS_PER_MS
        while delayed_index < len(delayed) and delayed[delayed_index][0] <= now_ns:
            _impact_ns, delta_x, delta_y = delayed[delayed_index]
            delayed_index += 1
            raw_error_x -= plant.gain_x_pixels_per_count * delta_x
            raw_error_y -= plant.gain_y_pixels_per_count * delta_y

        observation = None
        if tick % 8 == 0:
            smoothed_error_x += observation_alpha * (
                raw_error_x - smoothed_error_x
            )
            smoothed_error_y += observation_alpha * (
                raw_error_y - smoothed_error_y
            )
            observation = ScreenErrorObservation(
                now_ns,
                smoothed_error_x,
                smoothed_error_y,
                velocity_error_x_pixels=(
                    raw_error_x if raw_velocity_channel else None
                ),
                velocity_error_y_pixels=(
                    raw_error_y if raw_velocity_channel else None
                ),
            )

        output = controller.step(
            now_ns,
            engaged=True,
            observation=observation,
        )
        estimated_speeds.append(
            math.hypot(
                output.target_velocity_x_pixels_per_second,
                output.target_velocity_y_pixels_per_second,
            )
        )
        fractional_x += output.rate_x_counts_per_second * 0.001
        fractional_y += output.rate_y_counts_per_second * 0.001
        delta_x = math.trunc(fractional_x)
        delta_y = math.trunc(fractional_y)
        fractional_x -= delta_x
        fractional_y -= delta_y
        if delta_x or delta_y:
            command = EmittedMouseCommand(now_ns, delta_x, delta_y)
            controller.preflight_emitted(command)
            controller.record_emitted(command)
            delayed.append((now_ns + 8 * NS_PER_MS, delta_x, delta_y))
            emitted.append((tick, delta_x, delta_y))
        radial_errors.append(math.hypot(raw_error_x, raw_error_y))

    return _SmoothedFeedbackRunResult(
        tuple(radial_errors),
        tuple(estimated_speeds),
        tuple(emitted),
    )


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def _run_direct_point_plant(
    *,
    position_time_constant_seconds: float,
    feedback_deadzone_pixels: float,
    noise_amplitude_pixels: float,
    physical_gain_scale: float = 1.0,
) -> _DirectPointRunResult:
    """Exercise the automatic one-point measurement path with no velocity FF."""

    controller = MakcuCalibratedController(
        CalibratedPlant(0.125, 0.120, 0.008),
        CalibratedControlConfig(
            position_time_constant_seconds=position_time_constant_seconds,
            velocity_filter_time_constant_seconds=0.018,
            maximum_target_speed_pixels_per_second=3000.0,
            maximum_target_acceleration_pixels_per_second_squared=20_000.0,
            maximum_rate_x_counts_per_second=19_200.0,
            maximum_rate_y_counts_per_second=19_200.0,
            stale_after_seconds=0.065,
            maximum_observation_interval_seconds=0.040,
            maximum_error_jump_pixels=180.0,
            feedback_deadzone_pixels=feedback_deadzone_pixels,
            wrong_way_guard_pixels=2.0,
            velocity_median_window=5,
            maximum_velocity_feedforward_fraction=0.0,
        ),
    )
    error_x = 70.0
    error_y = -35.0
    fractional_x = 0.0
    fractional_y = 0.0
    delayed: list[tuple[int, int, int]] = []
    delayed_index = 0
    observation_index = 0
    next_observation_ms = 0
    observation_intervals_ms = (8, 8, 7, 9, 8, 7, 9)
    radial_errors: list[float] = []
    requested_rates: list[tuple[float, float]] = []

    for tick in range(4000):
        now_ns = tick * NS_PER_MS
        if 300 <= tick < 1500:
            velocity_x, velocity_y = 575.0, -287.5
        elif 2000 <= tick < 3200:
            velocity_x, velocity_y = -575.0, 345.0
        else:
            velocity_x, velocity_y = 0.0, 0.0
        error_x += velocity_x * 0.001
        error_y += velocity_y * 0.001

        while delayed_index < len(delayed) and delayed[delayed_index][0] <= tick:
            _impact_ms, delta_x, delta_y = delayed[delayed_index]
            delayed_index += 1
            error_x -= 0.125 * physical_gain_scale * delta_x
            error_y -= 0.120 * physical_gain_scale * delta_y

        observation = None
        if tick >= next_observation_ms:
            noise_x = noise_amplitude_pixels * (
                0.70 * math.sin(observation_index * 1.91)
                + 0.30 * math.sin(observation_index * 0.47 + 0.3)
            )
            noise_y = noise_amplitude_pixels * (
                0.72 * math.sin(observation_index * 1.57 + 0.2)
                + 0.28 * math.sin(observation_index * 0.39)
            )
            measured_x = error_x + noise_x
            measured_y = error_y + noise_y
            # The direct head detector intentionally supplies this same exact
            # point to the paired position and observer channels.
            observation = ScreenErrorObservation(
                now_ns,
                measured_x,
                measured_y,
                velocity_error_x_pixels=measured_x,
                velocity_error_y_pixels=measured_y,
            )
            next_observation_ms += observation_intervals_ms[
                observation_index % len(observation_intervals_ms)
            ]
            observation_index += 1

        output = controller.step(
            now_ns,
            engaged=True,
            observation=observation,
        )
        requested_rates.append(
            (
                output.rate_x_counts_per_second,
                output.rate_y_counts_per_second,
            )
        )
        fractional_x += output.rate_x_counts_per_second * 0.001
        fractional_y += output.rate_y_counts_per_second * 0.001
        delta_x = math.trunc(fractional_x)
        delta_y = math.trunc(fractional_y)
        fractional_x -= delta_x
        fractional_y -= delta_y
        if delta_x or delta_y:
            command = EmittedMouseCommand(now_ns, delta_x, delta_y)
            controller.preflight_emitted(command)
            controller.record_emitted(command)
            delayed.append((tick + 8, delta_x, delta_y))
        radial_errors.append(math.hypot(error_x, error_y))

    moving_errors = radial_errors[400:1450]
    reversal_errors = radial_errors[2000:2300]
    stationary_errors = radial_errors[3400:4000]
    stationary_rates = requested_rates[3400:4000]
    return _DirectPointRunResult(
        moving_rms_pixels=_rms(moving_errors),
        moving_p95_pixels=_percentile_95(moving_errors),
        reversal_rms_pixels=_rms(reversal_errors),
        maximum_reversal_error_pixels=max(reversal_errors),
        stationary_rms_pixels=_rms(stationary_errors),
        stationary_p95_pixels=_percentile_95(stationary_errors),
        steady_abs_counts_per_second=sum(
            abs(rate_x) + abs(rate_y) for rate_x, rate_y in stationary_rates
        )
        / 600.0,
        maximum_requested_axis_rate=max(
            max(abs(rate_x), abs(rate_y)) for rate_x, rate_y in requested_rates
        ),
    )


def _run_aged_corroborated_direct_point_plant(
    *,
    observation_hz: float,
    processing_age_ms: int,
    feedforward_fraction: float,
    physical_gain_scale: float,
    body_noise_amplitude_pixels: float = 8.0,
    position_time_constant_seconds: float = 0.022,
) -> _DirectPointRunResult:
    """Close the plant around 60/120 Hz source-dated head/body evidence."""

    controller = MakcuCalibratedController(
        CalibratedPlant(0.125, 0.120, 0.008),
        CalibratedControlConfig(
            position_time_constant_seconds=position_time_constant_seconds,
            velocity_filter_time_constant_seconds=0.018,
            maximum_target_speed_pixels_per_second=3000.0,
            maximum_target_acceleration_pixels_per_second_squared=20_000.0,
            maximum_rate_x_counts_per_second=19_200.0,
            maximum_rate_y_counts_per_second=19_200.0,
            stale_after_seconds=0.065,
            maximum_observation_interval_seconds=0.040,
            maximum_error_jump_pixels=180.0,
            feedback_deadzone_pixels=3.0,
            wrong_way_guard_pixels=2.0,
            velocity_median_window=5,
            maximum_velocity_feedforward_fraction=feedforward_fraction,
            require_motion_corroboration_for_feedforward=True,
        ),
    )
    error_x = 70.0
    error_y = -35.0
    fractional_x = 0.0
    fractional_y = 0.0
    delayed: list[tuple[int, int, int]] = []
    delayed_index = 0
    queued_observations: list[tuple[int, ScreenErrorObservation]] = []
    observation_index = 0
    next_source_ms = 0.0
    radial_errors: list[float] = []
    requested_rates: list[tuple[float, float]] = []

    for tick in range(4500):
        if 400 <= tick < 1600:
            velocity_x, velocity_y = 575.0, -287.5
        elif 2200 <= tick < 3400:
            velocity_x, velocity_y = -575.0, 345.0
        else:
            velocity_x, velocity_y = 0.0, 0.0
        error_x += velocity_x * 0.001
        error_y += velocity_y * 0.001

        while delayed_index < len(delayed) and delayed[delayed_index][0] <= tick:
            _impact_ms, delta_x, delta_y = delayed[delayed_index]
            delayed_index += 1
            error_x -= 0.125 * physical_gain_scale * delta_x
            error_y -= 0.120 * physical_gain_scale * delta_y

        while tick + 1e-9 >= next_source_ms:
            head_noise_x = (
                0.70 * math.sin(observation_index * 1.91)
                + 0.30 * math.sin(observation_index * 0.47 + 0.3)
            )
            head_noise_y = (
                0.72 * math.sin(observation_index * 1.57 + 0.2)
                + 0.28 * math.sin(observation_index * 0.39)
            )
            body_noise_x = body_noise_amplitude_pixels * (
                0.70 * math.sin(observation_index * 1.31 + 0.4)
                + 0.30 * math.sin(observation_index * 0.29)
            )
            body_noise_y = body_noise_amplitude_pixels * (
                0.72 * math.sin(observation_index * 1.73 + 0.7)
                + 0.28 * math.sin(observation_index * 0.37)
            )
            source_ms = round(next_source_ms)
            measured_x = error_x + head_noise_x
            measured_y = error_y + head_noise_y
            queued_observations.append(
                (
                    source_ms + processing_age_ms,
                    ScreenErrorObservation(
                        source_ms * NS_PER_MS,
                        measured_x,
                        measured_y,
                        velocity_error_x_pixels=measured_x,
                        velocity_error_y_pixels=measured_y,
                        corroboration_error_x_pixels=(
                            error_x + 100.0 + body_noise_x
                        ),
                        corroboration_error_y_pixels=(
                            error_y + 200.0 + body_noise_y
                        ),
                    ),
                )
            )
            observation_index += 1
            next_source_ms += 1000.0 / observation_hz

        observation = None
        if queued_observations and queued_observations[0][0] <= tick:
            _arrival_ms, observation = queued_observations.pop(0)
        output = controller.step(
            tick * NS_PER_MS,
            engaged=True,
            observation=observation,
        )
        requested_rates.append(
            (output.rate_x_counts_per_second, output.rate_y_counts_per_second)
        )
        fractional_x += output.rate_x_counts_per_second * 0.001
        fractional_y += output.rate_y_counts_per_second * 0.001
        delta_x = math.trunc(fractional_x)
        delta_y = math.trunc(fractional_y)
        fractional_x -= delta_x
        fractional_y -= delta_y
        if delta_x or delta_y:
            command = EmittedMouseCommand(tick * NS_PER_MS, delta_x, delta_y)
            controller.preflight_emitted(command)
            controller.record_emitted(command)
            delayed.append((tick + 8, delta_x, delta_y))
        radial_errors.append(math.hypot(error_x, error_y))

    moving_errors = radial_errors[600:1500] + radial_errors[2400:3300]
    reversal_errors = radial_errors[2200:2500]
    stationary_errors = radial_errors[3700:4500]
    stationary_rates = requested_rates[3700:4500]
    return _DirectPointRunResult(
        moving_rms_pixels=_rms(moving_errors),
        moving_p95_pixels=_percentile_95(moving_errors),
        reversal_rms_pixels=_rms(reversal_errors),
        maximum_reversal_error_pixels=max(reversal_errors),
        stationary_rms_pixels=_rms(stationary_errors),
        stationary_p95_pixels=_percentile_95(stationary_errors),
        steady_abs_counts_per_second=sum(
            abs(rate_x) + abs(rate_y) for rate_x, rate_y in stationary_rates
        )
        / len(stationary_rates),
        maximum_requested_axis_rate=max(
            max(abs(rate_x), abs(rate_y)) for rate_x, rate_y in requested_rates
        ),
    )


def _run_aged_body_derived_point_plant(
    *,
    observation_hz: float,
    processing_age_ms: int,
    physical_gain_scale: float,
    circular_jitter_pixels: float,
    body_derived_projection_fraction: float,
    body_derived_feedforward_fraction: float,
    maximum_rate_counts_per_second: float,
    body_derived_pursuit_feedforward_fraction: float = 0.0,
    circular_jitter_hz: float = 8.0,
    moving_velocity_x_pixels_per_second: float = 575.0,
    mapped_filter_time_constant_seconds: float = 0.012,
    position_time_constant_seconds: float = 0.045,
    feedback_deadzone_pixels: float = 2.5,
    continuous_feedback_deadband: bool = False,
    continuous_feedback_shoulder_pixels: float = 0.0,
    pursuit_position_time_constant_seconds: float = 0.0,
    pursuit_position_time_constant_start_pixels: float = 0.0,
    pursuit_position_time_constant_full_pixels: float = 0.0,
    additional_body_derived_projection_seconds: float = 0.0,
    translation_first_velocity_channel: bool = False,
    translation_first_position_channel: bool = False,
) -> _DirectPointRunResult:
    """Close the plant around source-dated, body-mapped point evidence.

    The legacy mapped coordinate filters the whole screen-space point.  The
    translation-first variant passes current body translation 1:1 and applies
    that same causal position filter only to circular local jitter.  Both
    remain one item of evidence: no synthetic body corroboration is supplied.
    The per-sample provenance flag authorizes configured source-age projection
    while explicit velocity feed-forward remains hard-capped at 0.25.
    """

    controller = MakcuCalibratedController(
        CalibratedPlant(0.125, 0.120, 0.006),
        CalibratedControlConfig(
            position_time_constant_seconds=position_time_constant_seconds,
            velocity_filter_time_constant_seconds=0.014,
            maximum_target_speed_pixels_per_second=3000.0,
            maximum_target_acceleration_pixels_per_second_squared=40_000.0,
            maximum_rate_x_counts_per_second=maximum_rate_counts_per_second,
            maximum_rate_y_counts_per_second=maximum_rate_counts_per_second,
            stale_after_seconds=0.110,
            maximum_observation_interval_seconds=0.040,
            maximum_error_jump_pixels=180.0,
            feedback_deadzone_pixels=feedback_deadzone_pixels,
            continuous_feedback_deadband=continuous_feedback_deadband,
            continuous_feedback_shoulder_pixels=(
                continuous_feedback_shoulder_pixels
            ),
            wrong_way_guard_pixels=2.0,
            velocity_median_window=3,
            maximum_velocity_feedforward_fraction=1.0,
            require_motion_corroboration_for_feedforward=True,
            maximum_body_derived_projection_fraction=(
                body_derived_projection_fraction
            ),
            maximum_body_derived_feedforward_fraction=(
                body_derived_feedforward_fraction
            ),
            maximum_body_derived_pursuit_feedforward_fraction=(
                body_derived_pursuit_feedforward_fraction
            ),
            pursuit_position_time_constant_seconds=(
                pursuit_position_time_constant_seconds
            ),
            pursuit_position_time_constant_start_pixels=(
                pursuit_position_time_constant_start_pixels
            ),
            pursuit_position_time_constant_full_pixels=(
                pursuit_position_time_constant_full_pixels
            ),
            additional_body_derived_projection_seconds=(
                additional_body_derived_projection_seconds
            ),
        ),
    )
    error_x = 70.0
    error_y = -35.0
    fractional_x = 0.0
    fractional_y = 0.0
    delayed: list[tuple[int, int, int]] = []
    delayed_index = 0
    queued_observations: list[tuple[int, ScreenErrorObservation]] = []
    observation_index = 0
    next_source_ms = 0.0
    filtered_x: float | None = None
    filtered_y: float | None = None
    radial_errors: list[float] = []
    requested_rates: list[tuple[float, float]] = []
    saturated_output_ticks = 0
    moving_lags_pixels: list[float] = []
    moving_lags_ms: list[float] = []
    post_stop_overshoots: list[float] = []
    source_interval_seconds = 1.0 / observation_hz
    filter_alpha = (
        1.0
        if mapped_filter_time_constant_seconds <= 0.0
        else 1.0
        - math.exp(
            -source_interval_seconds / mapped_filter_time_constant_seconds
        )
    )

    for tick in range(4500):
        if 400 <= tick < 1600:
            velocity_x = moving_velocity_x_pixels_per_second
            velocity_y = -moving_velocity_x_pixels_per_second * 0.5
        elif 2200 <= tick < 3400:
            velocity_x = -moving_velocity_x_pixels_per_second
            velocity_y = moving_velocity_x_pixels_per_second * 0.6
        else:
            velocity_x, velocity_y = 0.0, 0.0
        error_x += velocity_x * 0.001
        error_y += velocity_y * 0.001

        while delayed_index < len(delayed) and delayed[delayed_index][0] <= tick:
            _impact_ms, delta_x, delta_y = delayed[delayed_index]
            delayed_index += 1
            error_x -= 0.125 * physical_gain_scale * delta_x
            error_y -= 0.120 * physical_gain_scale * delta_y

        while tick + 1e-9 >= next_source_ms:
            phase = 2.0 * math.pi * circular_jitter_hz * next_source_ms / 1000.0
            local_x = circular_jitter_pixels * math.cos(phase)
            local_y = circular_jitter_pixels * math.sin(phase)
            mapped_x = error_x + local_x
            mapped_y = error_y + local_y
            if filtered_x is None or filtered_y is None:
                filtered_x = (
                    local_x if translation_first_position_channel else mapped_x
                )
                filtered_y = (
                    local_y if translation_first_position_channel else mapped_y
                )
            else:
                filter_input_x = (
                    local_x if translation_first_position_channel else mapped_x
                )
                filter_input_y = (
                    local_y if translation_first_position_channel else mapped_y
                )
                filtered_x += filter_alpha * (filter_input_x - filtered_x)
                filtered_y += filter_alpha * (filter_input_y - filtered_y)
            position_x = (
                error_x + filtered_x
                if translation_first_position_channel
                else filtered_x
            )
            position_y = (
                error_y + filtered_y
                if translation_first_position_channel
                else filtered_y
            )
            source_ms = round(next_source_ms)
            # Model a renewed direct-head identity lease at 10 Hz. The mapped
            # primary samples remain source-dated at the full detector cadence;
            # neither predictive grant is allowed to extend its own deadline.
            direct_source_ms = (source_ms // 100) * 100
            identity_deadline_ns = (direct_source_ms + 200) * NS_PER_MS
            queued_observations.append(
                (
                    source_ms + processing_age_ms,
                    ScreenErrorObservation(
                        source_ms * NS_PER_MS,
                        position_x,
                        position_y,
                        velocity_error_x_pixels=(
                            mapped_x
                            if translation_first_velocity_channel
                            else position_x
                        ),
                        velocity_error_y_pixels=(
                            mapped_y
                            if translation_first_velocity_channel
                            else position_y
                        ),
                        body_derived_motion_permitted=True,
                        body_derived_motion_deadline_ns=identity_deadline_ns,
                        identity_deadline_ns=identity_deadline_ns,
                    ),
                )
            )
            observation_index += 1
            next_source_ms += 1000.0 / observation_hz

        observation = None
        if queued_observations and queued_observations[0][0] <= tick:
            _arrival_ms, observation = queued_observations.pop(0)
        output = controller.step(
            tick * NS_PER_MS,
            engaged=True,
            observation=observation,
        )
        requested_rates.append(
            (output.rate_x_counts_per_second, output.rate_y_counts_per_second)
        )
        if output.saturated_x or output.saturated_y:
            saturated_output_ticks += 1
        fractional_x += output.rate_x_counts_per_second * 0.001
        fractional_y += output.rate_y_counts_per_second * 0.001
        delta_x = math.trunc(fractional_x)
        delta_y = math.trunc(fractional_y)
        fractional_x -= delta_x
        fractional_y -= delta_y
        if delta_x or delta_y:
            command = EmittedMouseCommand(tick * NS_PER_MS, delta_x, delta_y)
            controller.preflight_emitted(command)
            controller.record_emitted(command)
            delayed.append((tick + 6, delta_x, delta_y))
        target_speed = math.hypot(velocity_x, velocity_y)
        if (
            target_speed > 0.0
            and (600 <= tick < 1500 or 2400 <= tick < 3300)
        ):
            along_motion_error = (
                error_x * velocity_x + error_y * velocity_y
            ) / target_speed
            moving_lags_pixels.append(along_motion_error)
            moving_lags_ms.append(
                along_motion_error / target_speed * 1000.0
            )
        if 1600 <= tick < 1900:
            former_velocity_x = moving_velocity_x_pixels_per_second
            former_velocity_y = -moving_velocity_x_pixels_per_second * 0.5
            former_speed = math.hypot(former_velocity_x, former_velocity_y)
            former_along_error = (
                error_x * former_velocity_x + error_y * former_velocity_y
            ) / former_speed
            post_stop_overshoots.append(max(-former_along_error, 0.0))
        elif 3400 <= tick < 3700:
            former_velocity_x = -moving_velocity_x_pixels_per_second
            former_velocity_y = moving_velocity_x_pixels_per_second * 0.6
            former_speed = math.hypot(former_velocity_x, former_velocity_y)
            former_along_error = (
                error_x * former_velocity_x + error_y * former_velocity_y
            ) / former_speed
            post_stop_overshoots.append(max(-former_along_error, 0.0))
        radial_errors.append(math.hypot(error_x, error_y))

    moving_errors = radial_errors[600:1500] + radial_errors[2400:3300]
    reversal_errors = radial_errors[2200:2500]
    stationary_errors = radial_errors[3700:4500]
    stationary_rates = requested_rates[3700:4500]
    return _DirectPointRunResult(
        moving_rms_pixels=_rms(moving_errors),
        moving_p95_pixels=_percentile_95(moving_errors),
        reversal_rms_pixels=_rms(reversal_errors),
        maximum_reversal_error_pixels=max(reversal_errors),
        stationary_rms_pixels=_rms(stationary_errors),
        stationary_p95_pixels=_percentile_95(stationary_errors),
        steady_abs_counts_per_second=sum(
            abs(rate_x) + abs(rate_y) for rate_x, rate_y in stationary_rates
        )
        / len(stationary_rates),
        maximum_requested_axis_rate=max(
            max(abs(rate_x), abs(rate_y)) for rate_x, rate_y in requested_rates
        ),
        saturated_output_fraction=saturated_output_ticks / len(requested_rates),
        moving_mean_lag_pixels=sum(moving_lags_pixels)
        / len(moving_lags_pixels),
        moving_p95_lag_pixels=_percentile_95(moving_lags_pixels),
        moving_mean_lag_ms=sum(moving_lags_ms) / len(moving_lags_ms),
        moving_p95_lag_ms=_percentile_95(moving_lags_ms),
        maximum_post_stop_overshoot_pixels=max(post_stop_overshoots),
    )


def _run_capture_phase_lookahead_plant(
    *,
    correlated: bool,
    correlated_pursuit_feedforward_fraction: float = 0.0,
    verified_flow_pursuit_feedforward_fraction: float = 0.0,
    residual_pursuit_feedforward_fraction: float = 0.0,
    body_derived_pursuit_feedforward_fraction: float = 0.90,
    verified_flow_motion: bool = False,
    corroboration_motion_scale: float = 1.0,
    moving_velocity_x_pixels_per_second: float = 1_000.0,
    physical_gain_scale: float = 1.0,
    noise_scale: float = 1.0,
    maximum_screen_rate_pixels_per_second: float | None = None,
) -> _DirectPointRunResult:
    """Compare P1-only control with the same P1 plus its proven P0 motion.

    The capture endpoint is eight milliseconds newer than each inferred-frame
    root. Both runs receive the same endpoint schedule, target trajectory,
    deterministic detector-noise function, and delayed integer-count plant.
    Their coordinates may then diverge only because this is a closed loop. By
    default the correlated run atomically supplies the root's independent
    motion evidence, matching the live capture-phase handoff. Tests may set
    the corroboration motion scale to stationary or opposed while retaining
    the same command-compensated screen response, proving the verified-P1
    fallback without accidentally authorizing the ordinary root grant.
    """

    plant = CalibratedPlant(0.125, 0.120, 0.006)
    maximum_rate_x = (
        19_200.0
        if maximum_screen_rate_pixels_per_second is None
        else maximum_screen_rate_pixels_per_second
        / plant.gain_x_pixels_per_count
    )
    maximum_rate_y = (
        19_200.0
        if maximum_screen_rate_pixels_per_second is None
        else maximum_screen_rate_pixels_per_second
        / plant.gain_y_pixels_per_count
    )
    controller = MakcuCalibratedController(
        plant,
        _test_config(
            position_time_constant_seconds=0.028,
            velocity_filter_time_constant_seconds=0.014,
            maximum_target_acceleration_pixels_per_second_squared=40_000.0,
            maximum_rate_x_counts_per_second=maximum_rate_x,
            maximum_rate_y_counts_per_second=maximum_rate_y,
            maximum_feedback_rate_x_counts_per_second=maximum_rate_x,
            maximum_feedback_rate_y_counts_per_second=maximum_rate_y,
            stale_after_seconds=0.110,
            maximum_observation_interval_seconds=0.040,
            feedback_deadzone_pixels=4.5,
            continuous_feedback_deadband=True,
            continuous_feedback_shoulder_pixels=6.0,
            pursuit_position_time_constant_seconds=0.016,
            pursuit_position_time_constant_start_pixels=10.5,
            pursuit_position_time_constant_full_pixels=22.0,
            maximum_velocity_feedforward_fraction=1.0,
            require_motion_corroboration_for_feedforward=True,
            maximum_body_derived_projection_fraction=1.0,
            maximum_body_derived_feedforward_fraction=0.50,
            maximum_body_derived_pursuit_feedforward_fraction=(
                body_derived_pursuit_feedforward_fraction
            ),
            maximum_residual_pursuit_feedforward_fraction=(
                residual_pursuit_feedforward_fraction
            ),
            maximum_correlated_lookahead_pursuit_feedforward_fraction=(
                correlated_pursuit_feedforward_fraction
            ),
            maximum_verified_flow_pursuit_feedforward_fraction=(
                verified_flow_pursuit_feedforward_fraction
            ),
        ),
    )
    duration_ms = 3600
    identity_deadline_ns = (duration_ms + 1000) * NS_PER_MS
    error_x = 70.0
    error_y = -35.0
    corroboration_x = error_x + 100.0
    corroboration_y = error_y + 180.0
    fractional_x = 0.0
    fractional_y = 0.0
    delayed: list[tuple[int, int, int]] = []
    delayed_index = 0
    root: ScreenErrorObservation | None = None
    radial_errors: list[float] = []
    requested_rates: list[tuple[float, float]] = []
    moving_lags_pixels: list[float] = []
    moving_lags_ms: list[float] = []
    post_stop_overshoots: list[float] = []
    saturated_output_ticks = 0
    maximum_moving_motion_corroboration_confidence = 0.0
    def measured_point(timestamp_ms: int) -> tuple[float, float]:
        # Small continuous, non-commensurate components exercise stationary
        # lock without giving either run a privileged noise realization.
        elapsed = timestamp_ms / 1000.0
        return (
            error_x
            + noise_scale
            * (
                0.85 * math.sin(2.0 * math.pi * 7.0 * elapsed)
                + 0.35 * math.sin(2.0 * math.pi * 13.0 * elapsed + 0.2)
            ),
            error_y
            + noise_scale
            * (
                0.75 * math.sin(2.0 * math.pi * 9.0 * elapsed + 0.4)
                + 0.30 * math.sin(2.0 * math.pi * 17.0 * elapsed)
            ),
        )

    for tick in range(duration_ms):
        if 400 <= tick < 1600:
            velocity_x = moving_velocity_x_pixels_per_second
            velocity_y = -moving_velocity_x_pixels_per_second * 0.5
        elif 1600 <= tick < 2400:
            # Exercise an immediate material reversal, not a gentle ramp.
            velocity_x = -moving_velocity_x_pixels_per_second
            velocity_y = moving_velocity_x_pixels_per_second * 0.5
        else:
            velocity_x, velocity_y = 0.0, 0.0
        error_x += velocity_x * 0.001
        error_y += velocity_y * 0.001
        corroboration_x += velocity_x * corroboration_motion_scale * 0.001
        corroboration_y += velocity_y * corroboration_motion_scale * 0.001

        while delayed_index < len(delayed) and delayed[delayed_index][0] <= tick:
            _impact_ms, delta_x, delta_y = delayed[delayed_index]
            delayed_index += 1
            error_x -= (
                plant.gain_x_pixels_per_count
                * physical_gain_scale
                * delta_x
            )
            error_y -= (
                plant.gain_y_pixels_per_count
                * physical_gain_scale
                * delta_y
            )
            corroboration_x -= (
                plant.gain_x_pixels_per_count
                * physical_gain_scale
                * delta_x
            )
            corroboration_y -= (
                plant.gain_y_pixels_per_count
                * physical_gain_scale
                * delta_y
            )

        observation = None
        correlated_lookahead = None
        phase_ms = tick % 10
        if phase_ms == 0:
            measured_x, measured_y = measured_point(tick)
            root = ScreenErrorObservation(
                tick * NS_PER_MS,
                measured_x,
                measured_y,
                velocity_error_x_pixels=measured_x,
                velocity_error_y_pixels=measured_y,
                corroboration_error_x_pixels=(
                    corroboration_x
                    + 0.30 * math.sin(2.0 * math.pi * 5.0 * tick / 1000.0)
                ),
                corroboration_error_y_pixels=(
                    corroboration_y
                    + 0.25
                    * math.sin(2.0 * math.pi * 11.0 * tick / 1000.0 + 0.3)
                ),
                identity_deadline_ns=identity_deadline_ns,
            )
        elif phase_ms == 8:
            assert root is not None
            measured_x, measured_y = measured_point(tick)
            endpoint = ScreenErrorObservation(
                tick * NS_PER_MS,
                measured_x,
                measured_y,
                velocity_error_x_pixels=measured_x,
                velocity_error_y_pixels=measured_y,
                identity_deadline_ns=identity_deadline_ns,
            )
            if correlated:
                correlated_lookahead = CorrelatedLookaheadObservation(
                    root,
                    endpoint,
                    runtime_identity_generation=1,
                    track_generation=1,
                    verified_flow_motion=verified_flow_motion,
                )
            else:
                observation = endpoint

        output = controller.step(
            tick * NS_PER_MS,
            engaged=True,
            observation=observation,
            correlated_lookahead=correlated_lookahead,
        )
        requested_rates.append(
            (output.rate_x_counts_per_second, output.rate_y_counts_per_second)
        )
        if output.saturated_x or output.saturated_y:
            saturated_output_ticks += 1
        fractional_x += output.rate_x_counts_per_second * 0.001
        fractional_y += output.rate_y_counts_per_second * 0.001
        delta_x = math.trunc(fractional_x)
        delta_y = math.trunc(fractional_y)
        fractional_x -= delta_x
        fractional_y -= delta_y
        if delta_x or delta_y:
            command = EmittedMouseCommand(tick * NS_PER_MS, delta_x, delta_y)
            controller.preflight_emitted(command)
            controller.record_emitted(command)
            delayed.append((tick + 6, delta_x, delta_y))

        target_speed = math.hypot(velocity_x, velocity_y)
        if target_speed > 0.0 and (650 <= tick < 1500 or 1800 <= tick < 2300):
            maximum_moving_motion_corroboration_confidence = max(
                maximum_moving_motion_corroboration_confidence,
                output.motion_corroboration_confidence,
            )
            along_motion_error = (
                error_x * velocity_x + error_y * velocity_y
            ) / target_speed
            moving_lags_pixels.append(along_motion_error)
            moving_lags_ms.append(along_motion_error / target_speed * 1000.0)
        if 2400 <= tick < 2700:
            former_velocity_x = -moving_velocity_x_pixels_per_second
            former_velocity_y = moving_velocity_x_pixels_per_second * 0.5
            former_speed = math.hypot(former_velocity_x, former_velocity_y)
            former_along_error = (
                error_x * former_velocity_x + error_y * former_velocity_y
            ) / former_speed
            post_stop_overshoots.append(max(-former_along_error, 0.0))
        radial_errors.append(math.hypot(error_x, error_y))

    moving_errors = radial_errors[650:1500] + radial_errors[1800:2300]
    reversal_errors = radial_errors[1600:1850]
    stationary_errors = radial_errors[3000:3600]
    post_stop_errors = radial_errors[2400:2700]
    stationary_rates = requested_rates[3000:3600]
    return _DirectPointRunResult(
        moving_rms_pixels=_rms(moving_errors),
        moving_p95_pixels=_percentile_95(moving_errors),
        reversal_rms_pixels=_rms(reversal_errors),
        maximum_reversal_error_pixels=max(reversal_errors),
        stationary_rms_pixels=_rms(stationary_errors),
        stationary_p95_pixels=_percentile_95(stationary_errors),
        steady_abs_counts_per_second=sum(
            abs(rate_x) + abs(rate_y) for rate_x, rate_y in stationary_rates
        )
        / len(stationary_rates),
        maximum_requested_axis_rate=max(
            max(abs(rate_x), abs(rate_y)) for rate_x, rate_y in requested_rates
        ),
        saturated_output_fraction=saturated_output_ticks / duration_ms,
        moving_mean_lag_pixels=sum(moving_lags_pixels) / len(moving_lags_pixels),
        moving_p95_lag_pixels=_percentile_95(moving_lags_pixels),
        moving_mean_lag_ms=sum(moving_lags_ms) / len(moving_lags_ms),
        moving_p95_lag_ms=_percentile_95(moving_lags_ms),
        maximum_post_stop_overshoot_pixels=max(post_stop_overshoots),
        post_stop_rms_pixels=_rms(post_stop_errors),
        post_stop_p95_pixels=_percentile_95(post_stop_errors),
        maximum_moving_motion_corroboration_confidence=(
            maximum_moving_motion_corroboration_confidence
        ),
    )


class CalibratedControlUnitTests(unittest.TestCase):
    def test_body_channel_agreement_forgives_only_bounded_collinear_phase(
        self,
    ) -> None:
        agreement = MakcuCalibratedController._paired_channel_agreement

        # A 2,000 px/s tracked-position step predicts the full bounded 24 px
        # upstream LP lead. The collinear lead is expected rather than noise.
        self.assertEqual(
            agreement(
                24.0,
                0.0,
                0.0,
                0.0,
                tracked_velocity_x=2_000.0,
                tracked_velocity_y=0.0,
            ),
            1.0,
        )
        # Perpendicular and opposite disagreement receives no forgiveness.
        for raw_x, raw_y in ((0.0, 24.0), (-24.0, 0.0)):
            with self.subTest(raw_x=raw_x, raw_y=raw_y):
                self.assertEqual(
                    agreement(
                        raw_x,
                        raw_y,
                        0.0,
                        0.0,
                        tracked_velocity_x=2_000.0,
                        tracked_velocity_y=0.0,
                    ),
                    0.0,
                )
        # Only 24 px can be removed. Twelve residual pixels remain halfway
        # through the unchanged 8-to-16 px agreement ramp.
        self.assertEqual(
            agreement(
                36.0,
                0.0,
                0.0,
                0.0,
                tracked_velocity_x=2_000.0,
                tracked_velocity_y=0.0,
            ),
            0.5,
        )
        # Omitting the body-only phase input preserves the historical radial
        # gate exactly for every other caller.
        self.assertEqual(agreement(24.0, 0.0, 0.0, 0.0), 0.0)

    def test_body_phase_agreement_restores_translation_but_not_raw_staircase(
        self,
    ) -> None:
        config = _test_config(
            position_time_constant_seconds=0.028,
            velocity_filter_time_constant_seconds=0.014,
            maximum_target_acceleration_pixels_per_second_squared=40_000.0,
            stale_after_seconds=0.110,
            maximum_observation_interval_seconds=0.040,
            feedback_deadzone_pixels=4.5,
            continuous_feedback_deadband=True,
            continuous_feedback_shoulder_pixels=6.0,
            maximum_velocity_feedforward_fraction=1.0,
            require_motion_corroboration_for_feedforward=True,
            maximum_body_derived_projection_fraction=1.0,
            maximum_body_derived_feedforward_fraction=0.50,
        )
        plant = CalibratedPlant(0.125, 0.120, 0.008)
        translated = MakcuCalibratedController(plant, config)
        for index in range(40):
            timestamp_ns = index * 8 * NS_PER_MS
            tracked_x = 100.0 + index * 16.0
            translated_output = translated.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    tracked_x,
                    0.0,
                    velocity_error_x_pixels=tracked_x + 24.0,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=(
                        timestamp_ns + 100 * NS_PER_MS
                    ),
                ),
            )

        self.assertTrue(translated_output.valid)
        self.assertEqual(
            translated._paired_measurement_agreement_x,  # noqa: SLF001
            1.0,
        )
        self.assertGreater(translated_output.body_derived_motion_confidence_x, 0.99)
        self.assertAlmostEqual(
            translated_output.target_velocity_x_pixels_per_second,
            2_000.0,
            delta=1.0,
        )
        self.assertEqual(
            translated_output.velocity_feedforward_confidence_x,
            0.50,
        )

        staircase = MakcuCalibratedController(plant, config)
        for index in range(100):
            timestamp_ns = index * 8 * NS_PER_MS
            staircase_output = staircase.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    0.0,
                    0.0,
                    velocity_error_x_pixels=index * 4.6,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=(
                        timestamp_ns + 100 * NS_PER_MS
                    ),
                ),
            )

        self.assertEqual(
            staircase._paired_measurement_agreement_x,  # noqa: SLF001
            0.0,
        )
        self.assertEqual(staircase_output.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(staircase_output.rate_x_counts_per_second, 0.0)

    def test_pursuit_position_feedback_survives_auxiliary_channel_disagreement(
        self,
    ) -> None:
        """A motion-only disagreement cannot close a far-error position loop."""

        def controller(*, preserve: bool) -> MakcuCalibratedController:
            return MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.006),
                _test_config(
                    position_time_constant_seconds=0.028,
                    pursuit_position_time_constant_seconds=0.016,
                    pursuit_position_time_constant_start_pixels=10.5,
                    pursuit_position_time_constant_full_pixels=22.0,
                    preserve_pursuit_position_feedback=preserve,
                    stale_after_seconds=0.110,
                    feedback_deadzone_pixels=4.5,
                    continuous_feedback_deadband=True,
                    continuous_feedback_shoulder_pixels=6.0,
                    require_motion_corroboration_for_feedforward=True,
                    maximum_body_derived_projection_fraction=1.0,
                    maximum_body_derived_feedforward_fraction=0.50,
                ),
            )

        deadline_ns = 200 * NS_PER_MS

        def run(subject: MakcuCalibratedController, error_x: float):
            subject.step(
                0,
                engaged=True,
                observation=ScreenErrorObservation(
                    0,
                    error_x,
                    0.0,
                    velocity_error_x_pixels=error_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=deadline_ns,
                ),
            )
            return subject.step(
                8 * NS_PER_MS,
                engaged=True,
                observation=ScreenErrorObservation(
                    8 * NS_PER_MS,
                    error_x,
                    0.0,
                    velocity_error_x_pixels=error_x,
                    # Perpendicular auxiliary motion is untrusted, but the
                    # independently filtered X position remains causal.
                    velocity_error_y_pixels=20.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=deadline_ns,
                ),
            )

        historical = controller(preserve=False)
        historical_output = run(historical, 100.0)
        protected = controller(preserve=True)
        protected_output = run(protected, 100.0)

        self.assertEqual(
            protected._paired_measurement_agreement_x,  # noqa: SLF001
            0.0,
        )
        self.assertEqual(historical_output.rate_x_counts_per_second, 0.0)
        self.assertGreater(protected_output.rate_x_counts_per_second, 0.0)
        self.assertEqual(protected_output.position_channel_agreement, 0.0)
        self.assertEqual(protected_output.position_feedback_confidence_x, 1.0)
        self.assertEqual(
            protected_output.velocity_feedforward_confidence_x,
            0.0,
        )
        self.assertEqual(
            protected_output.velocity_feedforward_confidence_y,
            0.0,
        )

        # The opt-in is deliberately pursuit-only. The same disagreement at
        # or below the existing 10.5 px pursuit threshold retains the exact
        # historical near-lock suppression.
        near_lock = controller(preserve=True)
        near_lock_output = run(near_lock, 10.5)
        self.assertEqual(near_lock_output.rate_x_counts_per_second, 0.0)
        self.assertEqual(near_lock_output.position_feedback_confidence_x, 0.0)

        # Smooth onset immediately outside the lock region stays small rather
        # than replacing the former zero with a new breakaway step.
        shoulder_outputs = [
            run(controller(preserve=True), error_x)
            for error_x in (11.0, 14.0, 16.0)
        ]
        shoulder_rates = [
            output.rate_x_counts_per_second for output in shoulder_outputs
        ]
        self.assertEqual(shoulder_rates, sorted(shoulder_rates))
        self.assertLess(shoulder_rates[0], 25.0)
        self.assertLess(shoulder_rates[-1], 2_500.0)

    def test_rejected_auxiliary_point_does_not_poison_position_bridge(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.006),
            _test_config(
                position_time_constant_seconds=0.028,
                stale_after_seconds=0.110,
            ),
        )
        for timestamp_ns in (0, 8 * NS_PER_MS):
            output = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    100.0,
                    0.0,
                    velocity_error_x_pixels=100.0,
                    velocity_error_y_pixels=0.0,
                ),
            )
        self.assertTrue(output.valid)
        self.assertGreater(output.rate_x_counts_per_second, 0.0)
        self.assertEqual(
            controller._paired_measurement_agreement_x,  # noqa: SLF001
            1.0,
        )

        rejected = controller.step(
            16 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(
                16 * NS_PER_MS,
                900.0,
                0.0,
                velocity_error_x_pixels=1_000.0,
                velocity_error_y_pixels=0.0,
            ),
        )

        self.assertTrue(rejected.valid)
        self.assertTrue(rejected.innovation_rejected)
        self.assertAlmostEqual(rejected.projected_error_x_pixels, 100.0)
        self.assertGreater(rejected.rate_x_counts_per_second, 0.0)
        self.assertEqual(rejected.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(rejected.position_channel_agreement, 1.0)
        self.assertEqual(
            controller._paired_measurement_agreement_x,  # noqa: SLF001
            1.0,
        )

        bridged = controller.step(17 * NS_PER_MS, engaged=True)
        self.assertTrue(bridged.valid)
        self.assertGreater(bridged.rate_x_counts_per_second, 0.0)

    def test_pursuit_position_feedback_option_requires_schedule(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "preserve_pursuit_position_feedback",
        ):
            _test_config(
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                preserve_pursuit_position_feedback=1,
            )
        with self.assertRaisesRegex(ValueError, "requires the pursuit"):
            _test_config(preserve_pursuit_position_feedback=True)

    def test_continuous_feedback_deadband_has_no_breakaway_step(self) -> None:
        legacy = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.006),
            _test_config(
                position_time_constant_seconds=0.018,
                feedback_deadzone_pixels=4.5,
            ),
        )
        continuous = MakcuCalibratedController(
            legacy.plant,
            _test_config(
                position_time_constant_seconds=0.018,
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
            ),
        )

        for sign in (-1.0, 1.0):
            just_outside = sign * (4.5 + 1e-6)
            legacy_rate, _ = legacy._paired_axis_rate(
                just_outside,
                0.0,
                0.125,
                9_600.0,
                0.0,
                feedback_held=False,
                position_confidence=1.0,
            )
            continuous_rate, _ = continuous._paired_axis_rate(
                just_outside,
                0.0,
                0.125,
                9_600.0,
                0.0,
                feedback_held=False,
                position_confidence=1.0,
            )
            self.assertGreater(abs(legacy_rate), 1_900.0)
            self.assertLess(abs(continuous_rate), 0.01)
            self.assertEqual(math.copysign(1.0, continuous_rate), sign)

        with self.assertRaisesRegex(TypeError, "continuous_feedback_deadband"):
            _test_config(continuous_feedback_deadband=1)  # type: ignore[arg-type]

    def test_continuous_feedback_shoulder_smooths_only_near_lock(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.006),
            _test_config(
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
            ),
        )

        self.assertEqual(controller._feedback_error(4.5), 0.0)
        self.assertAlmostEqual(controller._feedback_error(5.5), 2.0 / 27.0)
        self.assertAlmostEqual(controller._feedback_error(7.5), 1.5)
        self.assertAlmostEqual(controller._feedback_error(10.5), 6.0)
        self.assertAlmostEqual(controller._feedback_error(20.0), 15.5)
        self.assertAlmostEqual(controller._feedback_error(-7.5), -1.5)

        for invalid in (-1.0, float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "continuous_feedback_shoulder_pixels",
            ):
                _test_config(continuous_feedback_shoulder_pixels=invalid)

    def test_pursuit_position_time_constant_is_opt_in_and_smooth(self) -> None:
        plant = CalibratedPlant(0.125, 0.120, 0.006)
        historical = MakcuCalibratedController(
            plant,
            _test_config(position_time_constant_seconds=0.018),
        )
        scheduled = MakcuCalibratedController(
            plant,
            _test_config(
                position_time_constant_seconds=0.018,
                pursuit_position_time_constant_seconds=0.010,
                pursuit_position_time_constant_start_pixels=14.0,
                pursuit_position_time_constant_full_pixels=30.0,
            ),
        )

        # An omitted/all-zero schedule is exactly the old fixed-tau law.
        self.assertEqual(
            historical._position_time_constant_seconds_for_error(1_000.0),
            0.018,
        )
        historical_rate, _ = historical._paired_axis_rate(
            36.0,
            0.0,
            0.125,
            100_000.0,
            0.0,
            feedback_held=False,
            position_confidence=1.0,
        )
        self.assertEqual(historical_rate, 36.0 / (0.125 * 0.018))

        # Each axis uses its own absolute residual. Smoothstep is exactly one
        # half at the interval midpoint, so 18 -> 10 ms becomes exactly 14 ms.
        for error in (-14.0, 0.0, 14.0):
            with self.subTest(endpoint="start", error=error):
                self.assertEqual(
                    scheduled._position_time_constant_seconds_for_error(error),
                    0.018,
                )
        for error in (-30.0, 30.0, 300.0):
            with self.subTest(endpoint="full", error=error):
                self.assertEqual(
                    scheduled._position_time_constant_seconds_for_error(error),
                    0.010,
                )
        self.assertAlmostEqual(
            scheduled._position_time_constant_seconds_for_error(22.0),
            0.014,
            delta=1e-15,
        )

        # Lock-region behavior is unchanged, while a large residual gets more
        # closed-loop pursuit authority without changing any rate ceiling.
        for error in (-12.0, 12.0):
            historical_near, _ = historical._paired_axis_rate(
                error,
                0.0,
                0.125,
                100_000.0,
                0.0,
                feedback_held=False,
                position_confidence=1.0,
            )
            scheduled_near, _ = scheduled._paired_axis_rate(
                error,
                0.0,
                0.125,
                100_000.0,
                0.0,
                feedback_held=False,
                position_confidence=1.0,
            )
            self.assertEqual(scheduled_near, historical_near)
        scheduled_far, _ = scheduled._paired_axis_rate(
            36.0,
            0.0,
            0.125,
            100_000.0,
            0.0,
            feedback_held=False,
            position_confidence=1.0,
        )
        self.assertAlmostEqual(scheduled_far / historical_rate, 1.8)

    def test_pursuit_and_extra_body_projection_config_is_validated(self) -> None:
        invalid_schedules = (
            {
                "pursuit_position_time_constant_seconds": 0.010,
            },
            {
                "pursuit_position_time_constant_seconds": 0.010,
                "pursuit_position_time_constant_start_pixels": 14.0,
                "pursuit_position_time_constant_full_pixels": 14.0,
            },
            {
                "position_time_constant_seconds": 0.018,
                "pursuit_position_time_constant_seconds": 0.020,
                "pursuit_position_time_constant_start_pixels": 14.0,
                "pursuit_position_time_constant_full_pixels": 30.0,
            },
        )
        for overrides in invalid_schedules:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                _test_config(**overrides)
        for invalid in (-0.001, 0.051, math.nan, True):
            with self.subTest(additional_projection=invalid), self.assertRaises(
                ValueError
            ):
                _test_config(
                    additional_body_derived_projection_seconds=invalid,
                )
        with self.assertRaisesRegex(
            ValueError,
            "1.25 times the minimum active position time constant",
        ):
            _test_config(
                position_time_constant_seconds=0.018,
                pursuit_position_time_constant_seconds=0.010,
                pursuit_position_time_constant_start_pixels=14.0,
                pursuit_position_time_constant_full_pixels=30.0,
                additional_body_derived_projection_seconds=0.0125001,
            )

    def test_automatic_pursuit_tau_requires_current_motion_authority(self) -> None:
        plant = CalibratedPlant(0.125, 0.120, 0.006)

        def make_controller(*, scheduled: bool) -> MakcuCalibratedController:
            pursuit = (
                {
                    "pursuit_position_time_constant_seconds": 0.010,
                    "pursuit_position_time_constant_start_pixels": 14.0,
                    "pursuit_position_time_constant_full_pixels": 30.0,
                }
                if scheduled
                else {}
            )
            return MakcuCalibratedController(
                plant,
                _test_config(
                    position_time_constant_seconds=0.018,
                    maximum_rate_x_counts_per_second=100_000.0,
                    maximum_rate_y_counts_per_second=100_000.0,
                    maximum_velocity_feedforward_fraction=0.0,
                    require_motion_corroboration_for_feedforward=True,
                    maximum_body_derived_projection_fraction=0.0,
                    maximum_body_derived_feedforward_fraction=0.0,
                    **pursuit,
                ),
            )

        baseline = make_controller(scheduled=False)
        scheduled = make_controller(scheduled=True)
        timestamp_ns = 0
        baseline_live = None
        scheduled_live = None
        for index in range(24):
            timestamp_ns = index * 8 * NS_PER_MS
            point_x = 40.0 + index * 4.6
            observation = ScreenErrorObservation(
                timestamp_ns,
                point_x,
                0.0,
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=NS_PER_SECOND,
            )
            baseline_live = baseline.step(
                timestamp_ns,
                engaged=True,
                observation=observation,
            )
            scheduled_live = scheduled.step(
                timestamp_ns,
                engaged=True,
                observation=observation,
            )
        assert baseline_live is not None
        assert scheduled_live is not None
        self.assertGreater(
            scheduled_live.rate_x_counts_per_second,
            baseline_live.rate_x_counts_per_second,
        )

        # Ordinary 1 kHz output ticks retain the last accepted physical grant
        # between detector frames, so the faster pursuit law remains useful at
        # 46/90 Hz instead of appearing for only one tick per measurement.
        hold_ns = timestamp_ns + NS_PER_MS
        baseline_hold = baseline.step(hold_ns, engaged=True)
        scheduled_hold = scheduled.step(hold_ns, engaged=True)
        self.assertTrue(scheduled_hold.valid)
        self.assertGreater(
            scheduled_hold.rate_x_counts_per_second,
            baseline_hold.rate_x_counts_per_second,
        )

        # A runtime prediction revokes body provenance without throwing away
        # the safe static point. Its following bridge must use the base 18 ms.
        baseline.revoke_body_derived_motion()
        scheduled.revoke_body_derived_motion()
        bridge_ns = timestamp_ns + 2 * NS_PER_MS
        baseline_bridge = baseline.step(bridge_ns, engaged=True)
        scheduled_bridge = scheduled.step(bridge_ns, engaged=True)
        self.assertTrue(scheduled_bridge.valid)
        self.assertEqual(
            scheduled_bridge.rate_x_counts_per_second,
            baseline_bridge.rate_x_counts_per_second,
        )

        # A new paired point without body provenance is an explicit predictive
        # revoke and must likewise use the base 18 ms feedback response.
        revoked_ns = timestamp_ns + 8 * NS_PER_MS
        revoked_x = 40.0 + 24 * 4.6
        revoked = ScreenErrorObservation(
            revoked_ns,
            revoked_x,
            0.0,
            velocity_error_x_pixels=revoked_x,
            velocity_error_y_pixels=0.0,
        )
        baseline_revoked = baseline.step(
            revoked_ns,
            engaged=True,
            observation=revoked,
        )
        scheduled_revoked = scheduled.step(
            revoked_ns,
            engaged=True,
            observation=revoked,
        )
        self.assertTrue(scheduled_revoked.valid)
        self.assertEqual(
            scheduled_revoked.rate_x_counts_per_second,
            baseline_revoked.rate_x_counts_per_second,
        )

        # A later physical body sample can restore pursuit immediately; the
        # broad no-head/current-primary-prediction revoke clears it as well.
        restored_ns = timestamp_ns + 16 * NS_PER_MS
        restored_x = 40.0 + 25 * 4.6
        restored = ScreenErrorObservation(
            restored_ns,
            restored_x,
            0.0,
            velocity_error_x_pixels=restored_x,
            velocity_error_y_pixels=0.0,
            body_derived_motion_permitted=True,
            body_derived_motion_deadline_ns=NS_PER_SECOND,
        )
        baseline_restored = baseline.step(
            restored_ns,
            engaged=True,
            observation=restored,
        )
        scheduled_restored = scheduled.step(
            restored_ns,
            engaged=True,
            observation=restored,
        )
        self.assertGreater(
            scheduled_restored.rate_x_counts_per_second,
            baseline_restored.rate_x_counts_per_second,
        )
        baseline.revoke_motion_corroboration()
        scheduled.revoke_motion_corroboration()
        broad_bridge_ns = timestamp_ns + 17 * NS_PER_MS
        baseline_broad_bridge = baseline.step(broad_bridge_ns, engaged=True)
        scheduled_broad_bridge = scheduled.step(
            broad_bridge_ns,
            engaged=True,
        )
        self.assertEqual(
            scheduled_broad_bridge.rate_x_counts_per_second,
            baseline_broad_bridge.rate_x_counts_per_second,
        )

    def test_velocity_error_pair_is_both_or_neither_and_finite(self) -> None:
        for values in ((1.0, None), (None, 1.0)):
            with self.subTest(values=values), self.assertRaisesRegex(
                ValueError,
                "both be supplied or both be omitted",
            ):
                ScreenErrorObservation(
                    0,
                    0.0,
                    0.0,
                    velocity_error_x_pixels=values[0],
                    velocity_error_y_pixels=values[1],
                )
        with self.assertRaisesRegex(ValueError, "velocity X error"):
            ScreenErrorObservation(
                0,
                0.0,
                0.0,
                velocity_error_x_pixels=math.nan,
                velocity_error_y_pixels=0.0,
            )

        with self.assertRaisesRegex(ValueError, "corroboration X/Y"):
            ScreenErrorObservation(
                0,
                0.0,
                0.0,
                velocity_error_x_pixels=0.0,
                velocity_error_y_pixels=0.0,
                corroboration_error_x_pixels=1.0,
            )
        with self.assertRaisesRegex(ValueError, "require the paired"):
            ScreenErrorObservation(
                0,
                0.0,
                0.0,
                corroboration_error_x_pixels=1.0,
                corroboration_error_y_pixels=1.0,
            )
        with self.assertRaisesRegex(ValueError, "corroboration X error"):
            ScreenErrorObservation(
                0,
                0.0,
                0.0,
                velocity_error_x_pixels=0.0,
                velocity_error_y_pixels=0.0,
                corroboration_error_x_pixels=math.inf,
                corroboration_error_y_pixels=0.0,
            )
        with self.assertRaisesRegex(TypeError, "body_derived_motion_permitted"):
            ScreenErrorObservation(
                0,
                0.0,
                0.0,
                body_derived_motion_permitted=1,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "requires the paired"):
            ScreenErrorObservation(
                0,
                0.0,
                0.0,
                body_derived_motion_permitted=True,
            )
        with self.assertRaisesRegex(ValueError, "cannot accompany"):
            ScreenErrorObservation(
                0,
                0.0,
                0.0,
                velocity_error_x_pixels=0.0,
                velocity_error_y_pixels=0.0,
                corroboration_error_x_pixels=100.0,
                corroboration_error_y_pixels=200.0,
                body_derived_motion_permitted=True,
            )
        with self.assertRaisesRegex(ValueError, "requires an immutable deadline"):
            ScreenErrorObservation(
                0,
                0.0,
                0.0,
                velocity_error_x_pixels=0.0,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
            )
        with self.assertRaisesRegex(ValueError, "requires motion permission"):
            ScreenErrorObservation(
                0,
                0.0,
                0.0,
                body_derived_motion_deadline_ns=1,
            )
        for invalid in (True, 1.5, -1):
            with self.subTest(motion_deadline=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                ScreenErrorObservation(
                    0,
                    0.0,
                    0.0,
                    velocity_error_x_pixels=0.0,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=invalid,  # type: ignore[arg-type]
                )
        with self.assertRaisesRegex(ValueError, "must be after"):
            ScreenErrorObservation(
                5,
                0.0,
                0.0,
                velocity_error_x_pixels=0.0,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=5,
            )
        for invalid in (True, 1.5, -1):
            with self.subTest(identity_deadline=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                ScreenErrorObservation(
                    0,
                    0.0,
                    0.0,
                    identity_deadline_ns=invalid,  # type: ignore[arg-type]
                )
        with self.assertRaisesRegex(ValueError, "identity deadline must be after"):
            ScreenErrorObservation(
                5,
                0.0,
                0.0,
                identity_deadline_ns=5,
            )

    def test_default_calibrated_envelope_has_no_hidden_vertical_ratio(self) -> None:
        config = CalibratedControlConfig()
        self.assertEqual(
            config.maximum_rate_y_counts_per_second,
            config.maximum_rate_x_counts_per_second,
        )
        self.assertEqual(config.maximum_velocity_feedforward_fraction, 1.0)
        self.assertEqual(config.maximum_body_derived_projection_fraction, 0.0)
        self.assertEqual(config.maximum_body_derived_feedforward_fraction, 0.0)
        self.assertEqual(
            config.maximum_body_derived_pursuit_feedforward_fraction,
            0.0,
        )
        self.assertEqual(
            config.maximum_residual_pursuit_feedforward_fraction,
            0.0,
        )
        self.assertEqual(
            config.maximum_correlated_lookahead_pursuit_feedforward_fraction,
            0.0,
        )
        self.assertEqual(
            config.maximum_verified_flow_pursuit_feedforward_fraction,
            0.0,
        )
        self.assertEqual(
            config.maximum_feedback_rate_x_counts_per_second,
            config.maximum_rate_x_counts_per_second,
        )
        self.assertEqual(
            config.maximum_feedback_rate_y_counts_per_second,
            config.maximum_rate_y_counts_per_second,
        )
        self.assertFalse(config.retain_ambiguous_correlated_projection)
        for invalid in (-0.01, 1.01, math.nan):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CalibratedControlConfig(
                    maximum_velocity_feedforward_fraction=invalid,
                )
        with self.assertRaises(TypeError):
            CalibratedControlConfig(
                require_motion_corroboration_for_feedforward=1,  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            CalibratedControlConfig(
                retain_ambiguous_correlated_projection=1,  # type: ignore[arg-type]
            )
        for invalid in (-1.0, 19_201.0, math.nan, True):
            with self.subTest(feedback_rate=invalid), self.assertRaises(ValueError):
                CalibratedControlConfig(
                    maximum_rate_x_counts_per_second=19_200.0,
                    maximum_feedback_rate_x_counts_per_second=invalid,
                )
        for invalid in (-0.01, 1.000001, math.nan, True):
            with self.subTest(body_derived_projection=invalid), self.assertRaises(
                ValueError
            ):
                CalibratedControlConfig(
                    maximum_body_derived_projection_fraction=invalid,
                )
        for invalid in (-0.01, 0.500001, math.nan, True):
            with self.subTest(body_derived_feedforward=invalid), self.assertRaises(
                ValueError
            ):
                CalibratedControlConfig(
                    maximum_body_derived_feedforward_fraction=invalid,
                )
        for invalid in (-0.01, 0.950001, math.nan, True):
            with self.subTest(body_derived_pursuit=invalid), self.assertRaises(
                ValueError
            ):
                CalibratedControlConfig(
                    maximum_body_derived_pursuit_feedforward_fraction=invalid,
                )
        with self.assertRaises(ValueError):
            CalibratedControlConfig(
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.49,
            )
        for invalid in (-0.01, 0.91, math.nan, True):
            with self.subTest(residual_pursuit=invalid), self.assertRaises(
                ValueError
            ):
                CalibratedControlConfig(
                    maximum_body_derived_feedforward_fraction=0.50,
                    maximum_body_derived_pursuit_feedforward_fraction=0.90,
                    maximum_residual_pursuit_feedforward_fraction=invalid,
                )
        for invalid in (-0.01, 0.49, 0.91, math.nan, True):
            with self.subTest(correlated_pursuit=invalid), self.assertRaises(
                ValueError
            ):
                CalibratedControlConfig(
                    maximum_body_derived_feedforward_fraction=0.50,
                    maximum_body_derived_pursuit_feedforward_fraction=0.90,
                    maximum_correlated_lookahead_pursuit_feedforward_fraction=(
                        invalid
                    ),
                )
        for invalid in (-0.01, 0.59, 0.950001, math.nan, True):
            with self.subTest(verified_flow_pursuit=invalid), self.assertRaises(
                ValueError
            ):
                CalibratedControlConfig(
                    maximum_body_derived_feedforward_fraction=0.50,
                    maximum_body_derived_pursuit_feedforward_fraction=0.90,
                    maximum_correlated_lookahead_pursuit_feedforward_fraction=(
                        0.60
                    ),
                    maximum_verified_flow_pursuit_feedforward_fraction=invalid,
                )
        decoupled_verified_flow = CalibratedControlConfig(
            maximum_velocity_feedforward_fraction=1.0,
            maximum_body_derived_feedforward_fraction=0.50,
            maximum_body_derived_pursuit_feedforward_fraction=0.82,
            maximum_correlated_lookahead_pursuit_feedforward_fraction=0.60,
            maximum_verified_flow_pursuit_feedforward_fraction=0.95,
        )
        self.assertEqual(
            decoupled_verified_flow.maximum_body_derived_pursuit_feedforward_fraction,
            0.82,
        )
        self.assertEqual(
            decoupled_verified_flow.maximum_verified_flow_pursuit_feedforward_fraction,
            0.95,
        )
        with self.assertRaises(ValueError):
            CalibratedControlConfig(
                maximum_velocity_feedforward_fraction=0.90,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.82,
                maximum_correlated_lookahead_pursuit_feedforward_fraction=0.60,
                maximum_verified_flow_pursuit_feedforward_fraction=0.95,
            )
        self.assertEqual(config.maximum_rate_y_counts_per_second, 19_200.0)

    def test_feedback_cap_preserves_motion_qualified_total_headroom(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.008),
            _test_config(
                position_time_constant_seconds=0.020,
                maximum_rate_x_counts_per_second=30_000.0,
                maximum_rate_y_counts_per_second=30_000.0,
                maximum_feedback_rate_x_counts_per_second=19_200.0,
                maximum_feedback_rate_y_counts_per_second=19_200.0,
            ),
        )

        feedback_only, feedback_saturated = controller._paired_axis_rate(
            100.0,
            0.0,
            0.10,
            30_000.0,
            0.0,
            feedback_held=False,
            feedback_limit=19_200.0,
            position_confidence=1.0,
        )
        with_velocity, total_saturated = controller._paired_axis_rate(
            100.0,
            1_000.0,
            0.10,
            30_000.0,
            1.0,
            feedback_held=False,
            feedback_limit=19_200.0,
            position_confidence=1.0,
        )

        self.assertEqual(feedback_only, 19_200.0)
        self.assertTrue(feedback_saturated)
        self.assertEqual(with_velocity, 29_200.0)
        self.assertTrue(total_saturated)
        self.assertLessEqual(with_velocity, 30_000.0)

    def test_body_feedforward_boost_requires_two_aligned_residuals(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                maximum_body_derived_feedforward_fraction=0.50,
            ),
        )

        self.assertEqual(
            controller._body_derived_feedforward_cap(4.5, 20.0, 1000.0),
            0.25,
        )
        self.assertAlmostEqual(
            controller._body_derived_feedforward_cap(7.5, 8.0, 1000.0),
            0.375,
        )
        self.assertEqual(
            controller._body_derived_feedforward_cap(10.5, 20.0, 1000.0),
            0.50,
        )
        for measured_error, projected_error, velocity in (
            (-20.0, 20.0, 1000.0),
            (20.0, -20.0, 1000.0),
            (20.0, 20.0, -1000.0),
            (0.0, 20.0, 1000.0),
            (20.0, 0.0, 1000.0),
            (20.0, 20.0, 0.0),
        ):
            with self.subTest(
                measured_error=measured_error,
                projected_error=projected_error,
                velocity=velocity,
            ):
                self.assertEqual(
                    controller._body_derived_feedforward_cap(
                        measured_error,
                        projected_error,
                        velocity,
                    ),
                    0.0,
                )

        # Direction is still aligned in these cases; being inside either side
        # of the deadband alone retains the established baseline authority.
        for measured_error, projected_error, velocity in (
            (3.0, 20.0, 1000.0),
            (20.0, 3.0, 1000.0),
        ):
            with self.subTest(
                measured_error=measured_error,
                projected_error=projected_error,
                velocity=velocity,
            ):
                self.assertEqual(
                    controller._body_derived_feedforward_cap(
                        measured_error,
                        projected_error,
                        velocity,
                    ),
                    0.25,
                )

    def test_fast_body_feedforward_reserve_carries_zero_error_and_closes_ahead(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.82,
            ),
        )
        cap = controller._body_derived_feedforward_cap

        # Constant-speed pursuit must retain velocity authority at exact lock;
        # otherwise each frame has to fall behind before feed-forward restarts.
        self.assertEqual(
            cap(0.0, 0.0, 1800.0, fresh_motion_confidence=1.0),
            0.82,
        )
        self.assertEqual(
            cap(20.0, 20.0, 1800.0, fresh_motion_confidence=0.0),
            0.50,
        )

        # It is an axis-local high-speed reserve, with smooth speed and
        # command-aware ahead-of-target withdrawal boundaries.
        self.assertEqual(
            cap(0.0, 0.0, 1400.0, fresh_motion_confidence=1.0),
            0.0,
        )
        self.assertAlmostEqual(
            cap(0.0, 0.0, 1600.0, fresh_motion_confidence=1.0),
            0.41,
        )
        self.assertEqual(
            cap(-8.0, -8.0, 1800.0, fresh_motion_confidence=1.0),
            0.0,
        )
        self.assertAlmostEqual(
            cap(-4.0, -4.0, 1800.0, fresh_motion_confidence=1.0),
            0.41,
        )

    def test_fast_body_feedforward_requires_fresh_command_compensated_motion(
        self,
    ) -> None:
        confidence = MakcuCalibratedController._body_derived_fresh_motion_confidence

        self.assertEqual(
            confidence(velocity=1000.0, measured_motion_delta=10.0, elapsed=0.010),
            1.0,
        )
        self.assertEqual(
            confidence(velocity=1000.0, measured_motion_delta=3.5, elapsed=0.010),
            0.0,
        )
        self.assertAlmostEqual(
            confidence(velocity=1000.0, measured_motion_delta=5.25, elapsed=0.010),
            0.5,
        )
        for measured_delta in (0.0, -10.0):
            with self.subTest(measured_delta=measured_delta):
                self.assertEqual(
                    confidence(
                        velocity=1000.0,
                        measured_motion_delta=measured_delta,
                        elapsed=0.010,
                    ),
                    0.0,
                )

    def test_adaptive_pursuit_learns_lag_holds_lock_and_revokes_fresh(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.82,
            ),
        )
        update = controller._adaptive_body_pursuit_confidence
        learned = update(
            0.0,
            measured_error=22.0,
            projected_error=22.0,
            projected_velocity=1200.0,
            projected_vector_speed=1200.0,
            motion_confidence=0.40,
            fresh_motion_confidence=1.0,
            direction_persistence_seconds=0.050,
            elapsed=0.080,
        )
        # Motion confidence authorizes the estimate; it does not attenuate the
        # already-bounded learned fraction a second time.
        self.assertAlmostEqual(learned, 0.82 * (1.0 - math.exp(-1.0)))
        held_lock = update(
            learned,
            measured_error=0.0,
            projected_error=0.0,
            projected_velocity=1200.0,
            projected_vector_speed=1200.0,
            motion_confidence=0.40,
            fresh_motion_confidence=1.0,
            direction_persistence_seconds=0.100,
            elapsed=0.010,
        )
        # Verified constant-speed motion keeps completing the conservative
        # body-only rise even at exact lock. Requiring lag to exist before the
        # setpoint can charge would make the target trail cyclically.
        self.assertAlmostEqual(
            held_lock,
            learned
            + (1.0 - math.exp(-0.010 / 0.080)) * (0.82 - learned),
        )
        for fresh, measured_error, projected_error in (
            (0.0, 0.0, 0.0),
            (1.0, -8.0, -8.0),
        ):
            with self.subTest(
                fresh=fresh,
                measured_error=measured_error,
            ):
                self.assertEqual(
                    update(
                        learned,
                        measured_error=measured_error,
                        projected_error=projected_error,
                        projected_velocity=1200.0,
                        projected_vector_speed=1200.0,
                        motion_confidence=0.40,
                        fresh_motion_confidence=fresh,
                        direction_persistence_seconds=0.100,
                        elapsed=0.010,
                    ),
                    0.0,
                )

    def test_adaptive_pursuit_honors_vector_and_global_feedforward_caps(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
            ),
        )
        update = controller._adaptive_body_pursuit_confidence
        common = dict(
            measured_error=22.0,
            projected_error=22.0,
            projected_velocity=1600.0,
            motion_confidence=0.50,
            fresh_motion_confidence=1.0,
            direction_persistence_seconds=0.100,
            elapsed=0.080,
        )
        # A diagonal 1600/1600 px/s observation is already in the cap-bound
        # vector regime even though neither individual axis reaches 2000 px/s.
        self.assertEqual(
            update(
                0.75,
                projected_vector_speed=math.hypot(1600.0, 1600.0),
                **common,
            ),
            0.0,
        )

        globally_disabled = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_velocity_feedforward_fraction=0.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
            ),
        )
        self.assertEqual(
            globally_disabled._adaptive_body_pursuit_confidence(
                0.75,
                projected_vector_speed=1600.0,
                **common,
            ),
            0.0,
        )

    def test_residual_pursuit_default_disabled_is_output_bit_identical(self) -> None:
        config = _test_config(
            position_time_constant_seconds=0.028,
            velocity_filter_time_constant_seconds=0.014,
            maximum_target_acceleration_pixels_per_second_squared=40_000.0,
            feedback_deadzone_pixels=4.5,
            continuous_feedback_deadband=True,
            maximum_velocity_feedforward_fraction=1.0,
            require_motion_corroboration_for_feedforward=True,
            maximum_body_derived_projection_fraction=1.0,
            maximum_body_derived_feedforward_fraction=0.50,
            maximum_body_derived_pursuit_feedforward_fraction=0.90,
        )
        self.assertEqual(
            config.maximum_residual_pursuit_feedforward_fraction,
            0.0,
        )
        deadline_ns = NS_PER_SECOND
        observations = tuple(
            ScreenErrorObservation(
                index * 8 * NS_PER_MS,
                20.0 + index * 8.0,
                0.0,
                velocity_error_x_pixels=20.0 + index * 8.0,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=deadline_ns,
                identity_deadline_ns=deadline_ns,
            )
            for index in range(10)
        )

        def run() -> tuple[CalibratedControlOutput, ...]:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.008),
                config,
            )
            return tuple(
                controller.step(
                    observation.timestamp_ns,
                    engaged=True,
                    observation=observation,
                )
                for observation in observations
            )

        disabled_outputs = run()
        # Simulate the pre-feature implementation by removing the update call
        # altogether. With the opt-in left at zero, every float, flag, and
        # reset reason must remain exactly equal (not merely approximately).
        with patch.object(
            MakcuCalibratedController,
            "_update_residual_pursuit",
            return_value=None,
        ):
            pre_feature_outputs = run()
        self.assertEqual(disabled_outputs, pre_feature_outputs)
        self.assertTrue(
            all(
                output.residual_pursuit_authority_x == 0.0
                and output.residual_pursuit_authority_y == 0.0
                for output in disabled_outputs
            )
        )

    def test_residual_pursuit_requires_current_paired_covariant_residuals(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                feedback_deadzone_pixels=4.5,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
                maximum_residual_pursuit_feedforward_fraction=0.80,
            ),
        )
        update = controller._residual_pursuit_axis_state

        def sample(
            state: object,
            *,
            current_measured: bool = True,
            measured_error: float = 20.0,
            projected_error: float = 20.0,
            projected_velocity: float = 500.0,
            position_sigma: float = 2.0,
            position_agreement: float = 1.0,
        ) -> object:
            return update(
                state,  # type: ignore[arg-type]
                measurement_event=True,
                current_measured=current_measured,
                elapsed=0.008,
                measured_error=measured_error,
                projected_error=projected_error,
                projected_velocity=projected_velocity,
                position_sigma=position_sigma,
                position_agreement=position_agreement,
                revoke=False,
            )

        zero = controller._residual_pursuit_x
        first = sample(zero)
        self.assertEqual(first.authority, 0.0)
        self.assertEqual(first.aligned_samples, 1)
        self.assertEqual(first.aligned_seconds, 0.008)
        second = sample(first)
        self.assertEqual(second.authority, 0.0)
        self.assertEqual(second.aligned_samples, 2)
        self.assertEqual(second.aligned_seconds, 0.016)
        qualified = sample(second)
        self.assertGreater(qualified.authority, 0.0)
        self.assertLess(qualified.authority, 0.80)
        self.assertLessEqual(qualified.authority, 0.80)

        # A prediction/non-paired callback, position-channel disagreement,
        # deadzone-sized jitter, or a residual below covariance uncertainty
        # cannot charge even when repeated for the full time horizon.
        for name, overrides in (
            ("not-current-measured", {"current_measured": False}),
            ("position-disagreement", {"position_agreement": 0.49}),
            (
                "inside-deadzone",
                {"measured_error": 4.5, "projected_error": 4.5},
            ),
            ("below-covariance", {"position_sigma": 50.0}),
        ):
            with self.subTest(name=name):
                state = zero
                for _ in range(4):
                    state = sample(state, **overrides)
                self.assertEqual(state.authority, 0.0)

        # At 126 Hz, a noisy 16 Hz lobe can expose two outside-deadzone points
        # before its measured direction turns. Repeated 12 px two-sample lobes
        # must never create retained velocity authority.
        state = zero
        for cycle in range(12):
            direction = 1.0 if cycle % 2 == 0 else -1.0
            for _ in range(2):
                state = sample(
                    state,
                    measured_error=12.0 * direction,
                    projected_error=12.0 * direction,
                    projected_velocity=500.0 * direction,
                )
                self.assertEqual(state.authority, 0.0)

    def test_residual_pursuit_holds_exact_lock_and_revokes_manual_motion(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                feedback_deadzone_pixels=4.5,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
                maximum_residual_pursuit_feedforward_fraction=0.80,
            ),
        )
        update = controller._residual_pursuit_axis_state

        def apply(
            state: object,
            *,
            measured_error: float,
            projected_error: float,
            velocity: float,
            measurement_event: bool = True,
            revoke: bool = False,
        ) -> object:
            return update(
                state,  # type: ignore[arg-type]
                measurement_event=measurement_event,
                current_measured=True,
                elapsed=0.008,
                measured_error=measured_error,
                projected_error=projected_error,
                projected_velocity=velocity,
                position_sigma=2.0,
                position_agreement=1.0,
                revoke=revoke,
            )

        def earned() -> object:
            state = controller._residual_pursuit_x
            for _ in range(3):
                state = apply(
                    state,
                    measured_error=20.0,
                    projected_error=20.0,
                    velocity=500.0,
                )
            return state

        learned = earned()
        self.assertGreater(learned.authority, 0.0)
        exact_lock = apply(
            learned,
            measured_error=0.0,
            projected_error=0.0,
            velocity=500.0,
        )
        self.assertEqual(exact_lock.authority, learned.authority)
        self.assertEqual(exact_lock.aligned_samples, 0)
        output_tick = apply(
            exact_lock,
            measured_error=0.0,
            projected_error=0.0,
            velocity=500.0,
            measurement_event=False,
        )
        self.assertEqual(output_tick, exact_lock)

        # These are the two fail-closed signatures available even when a
        # user's physical approach was not reported by the passthrough path:
        # the residual crosses ahead, or newest measured motion reverses/stops.
        for name, values in (
            ("manual-ahead", (-0.001, 0.0, 500.0, False)),
            ("manual-reversal", (20.0, 20.0, -500.0, False)),
            ("material-stop", (20.0, 20.0, 0.0, False)),
            ("innovation-or-physical-revoke", (20.0, 20.0, 500.0, True)),
        ):
            with self.subTest(name=name):
                measured, projected, velocity, revoke = values
                cleared = apply(
                    learned,
                    measured_error=measured,
                    projected_error=projected,
                    velocity=velocity,
                    revoke=revoke,
                )
                self.assertEqual(cleared.authority, 0.0)
                self.assertEqual(cleared.direction, 0)

    def test_residual_pursuit_requires_authorized_independent_corroboration(
        self,
    ) -> None:
        config = _test_config(
            feedback_deadzone_pixels=4.5,
            maximum_velocity_feedforward_fraction=1.0,
            require_motion_corroboration_for_feedforward=True,
            maximum_body_derived_feedforward_fraction=0.50,
            maximum_body_derived_pursuit_feedforward_fraction=0.80,
            maximum_residual_pursuit_feedforward_fraction=0.80,
        )
        deadline_ns = 2 * NS_PER_SECOND

        def run(corroboration: str) -> tuple[
            MakcuCalibratedController,
            CalibratedControlOutput,
        ]:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.008),
                config,
            )
            output = None
            for index in range(64):
                timestamp_ns = index * 8 * NS_PER_MS
                point_x = 40.0 + index * 4.6
                if corroboration == "coherent":
                    body_x = point_x + 100.0
                elif corroboration == "opposite":
                    body_x = 140.0 - index * 4.6
                else:
                    body_x = 140.0
                output = controller.step(
                    timestamp_ns,
                    engaged=True,
                    observation=ScreenErrorObservation(
                        timestamp_ns,
                        point_x,
                        0.0,
                        velocity_error_x_pixels=point_x,
                        velocity_error_y_pixels=0.0,
                        corroboration_error_x_pixels=body_x,
                        corroboration_error_y_pixels=200.0,
                        identity_deadline_ns=deadline_ns,
                    ),
                )
            assert output is not None
            return controller, output

        # Supplying a same-frame coordinate is not itself authorization. A
        # stationary or opposite-moving independent observer must keep the
        # closed-loop residual channel at exact zero despite persistent lag in
        # the primary point.
        for corroboration in ("stationary", "opposite"):
            with self.subTest(corroboration=corroboration):
                controller, output = run(corroboration)
                self.assertFalse(controller._independent_pursuit_authorized)
                self.assertEqual(output.motion_corroboration_confidence, 0.0)
                self.assertEqual(output.residual_pursuit_authority_x, 0.0)

        controller, coherent = run("coherent")
        self.assertTrue(controller._independent_pursuit_authorized)
        self.assertGreater(coherent.motion_corroboration_confidence, 0.90)
        self.assertGreater(coherent.residual_pursuit_authority_x, 0.0)
        self.assertLessEqual(coherent.residual_pursuit_authority_x, 0.80)

    def test_verified_correlated_p1_supplies_narrow_additive_residual(
        self,
    ) -> None:
        config = _test_config(
            position_time_constant_seconds=0.028,
            velocity_filter_time_constant_seconds=0.014,
            maximum_target_acceleration_pixels_per_second_squared=40_000.0,
            stale_after_seconds=0.110,
            feedback_deadzone_pixels=4.5,
            maximum_velocity_feedforward_fraction=1.0,
            require_motion_corroboration_for_feedforward=True,
            maximum_body_derived_projection_fraction=1.0,
            maximum_body_derived_feedforward_fraction=0.50,
            maximum_body_derived_pursuit_feedforward_fraction=0.82,
            maximum_residual_pursuit_feedforward_fraction=0.65,
            maximum_correlated_lookahead_pursuit_feedforward_fraction=0.60,
            maximum_verified_flow_pursuit_feedforward_fraction=0.60,
        )
        base_ns = 10 * NS_PER_SECOND
        deadline_ns = base_ns + 2 * NS_PER_SECOND

        def batch(
            index: int,
            *,
            coherent_root: bool,
            verified_flow: bool,
            phase_pixels: float = 12.0,
        ) -> CorrelatedLookaheadObservation:
            root_ns = base_ns + index * 10 * NS_PER_MS
            root_x = 40.0 + index * 15.0
            return CorrelatedLookaheadObservation(
                ScreenErrorObservation(
                    root_ns,
                    root_x,
                    0.0,
                    velocity_error_x_pixels=root_x,
                    velocity_error_y_pixels=0.0,
                    corroboration_error_x_pixels=(
                        root_x + 100.0 if coherent_root else 140.0
                    ),
                    corroboration_error_y_pixels=100.0,
                    identity_deadline_ns=deadline_ns,
                ),
                ScreenErrorObservation(
                    root_ns + 8 * NS_PER_MS,
                    root_x + phase_pixels,
                    0.0,
                    velocity_error_x_pixels=root_x + phase_pixels,
                    velocity_error_y_pixels=0.0,
                    identity_deadline_ns=deadline_ns,
                ),
                runtime_identity_generation=1,
                track_generation=1,
                verified_flow_motion=verified_flow,
            )

        def run(
            *,
            coherent_root: bool,
            verified_flow: bool,
            phase_pixels: float = 12.0,
        ) -> tuple[MakcuCalibratedController, CalibratedControlOutput]:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.008),
                config,
            )
            output = None
            for index in range(40):
                correlated = batch(
                    index,
                    coherent_root=coherent_root,
                    verified_flow=verified_flow,
                    phase_pixels=phase_pixels,
                )
                output = controller.step(
                    correlated.lookahead.timestamp_ns + NS_PER_MS,
                    engaged=True,
                    correlated_lookahead=correlated,
                )
            assert output is not None
            return controller, output

        # A runtime flag alone is insufficient.  With no accepted endpoint
        # displacement (or no verified-flow flag), the stationary independent
        # root cannot charge residual pursuit despite persistent primary trail.
        for name, verified, phase in (
            ("unverified", False, 12.0),
            ("unsupported-flow", True, 0.0),
            ("opposed-flow", True, -4.0),
        ):
            with self.subTest(name=name):
                controller, output = run(
                    coherent_root=False,
                    verified_flow=verified,
                    phase_pixels=phase,
                )
                self.assertFalse(controller._independent_pursuit_authorized)
                self.assertEqual(output.motion_corroboration_confidence, 0.0)
                self.assertEqual(output.residual_pursuit_authority_x, 0.0)
                self.assertEqual(output.velocity_feedforward_confidence_x, 0.0)

        controller, learned = run(
            coherent_root=False,
            verified_flow=True,
        )
        self.assertFalse(controller._independent_pursuit_authorized)
        self.assertEqual(learned.lookahead_retained_authority_x, 0.0)
        self.assertGreater(learned.residual_pursuit_authority_x, 0.60)
        self.assertAlmostEqual(
            learned.velocity_feedforward_confidence_x,
            learned.residual_pursuit_authority_x,
        )
        self.assertTrue(controller._verified_residual_projection_x)
        self.assertFalse(controller._verified_residual_projection_y)
        self.assertGreater(
            learned.projected_error_x_pixels,
            controller._paired_position_x,
        )
        self.assertEqual(learned.residual_pursuit_authority_y, 0.0)
        self.assertEqual(learned.velocity_feedforward_confidence_y, 0.0)

        # P1-only residual authority shares the endpoint's short correlated
        # lease, not the accepted position's longer freshness lease.  An
        # output-only tick at that deadline must visibly revoke and erase it.
        correlated_deadline_ns = (
            controller._correlated_lookahead_authority_deadline_ns
        )
        self.assertIsNotNone(correlated_deadline_ns)
        assert correlated_deadline_ns is not None
        expired = controller.step(correlated_deadline_ns, engaged=True)
        self.assertTrue(expired.valid)
        self.assertTrue(expired.predictive_authority_revoked_x)
        self.assertFalse(expired.predictive_authority_revoked_y)
        self.assertEqual(expired.residual_pursuit_authority_x, 0.0)
        self.assertEqual(expired.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(controller._residual_pursuit_x.authority, 0.0)
        self.assertFalse(controller._verified_residual_projection_x)

        # The next unverified P1 is still a measurement event and fails closed;
        # it cannot coast on authority learned by the verified run.
        controller, _learned_again = run(
            coherent_root=False,
            verified_flow=True,
        )
        unverified = batch(
            40,
            coherent_root=False,
            verified_flow=False,
        )
        cleared = controller.step(
            unverified.lookahead.timestamp_ns + NS_PER_MS,
            engaged=True,
            correlated_lookahead=unverified,
        )
        self.assertEqual(cleared.residual_pursuit_authority_x, 0.0)
        self.assertEqual(cleared.velocity_feedforward_confidence_x, 0.0)
        self.assertFalse(controller._verified_residual_projection_x)

        # A raw physical report withdraws the same axis on the next output
        # tick even without waiting for another detector measurement.
        controller, _learned_for_manual = run(
            coherent_root=False,
            verified_flow=True,
        )
        final_endpoint_ns = batch(
            39,
            coherent_root=False,
            verified_flow=True,
        ).lookahead.timestamp_ns
        controller.record_physical_input(final_endpoint_ns + 2 * NS_PER_MS, 1, 0)
        manual = controller.step(
            final_endpoint_ns + 3 * NS_PER_MS,
            engaged=True,
        )
        self.assertTrue(manual.physical_input_pending_x)
        self.assertFalse(controller._verified_residual_projection_x)
        self.assertEqual(manual.residual_pursuit_authority_x, 0.0)

        # Residual is missing authority, not an alternate total fraction.  It
        # is added after P0's 60% correlated cap, then bounded by the unchanged
        # 82% measured-pursuit ceiling.
        _coherent_controller, additive = run(
            coherent_root=True,
            verified_flow=True,
        )
        self.assertAlmostEqual(additive.lookahead_retained_authority_x, 0.60)
        self.assertGreater(additive.residual_pursuit_authority_x, 0.60)
        self.assertGreater(
            additive.velocity_feedforward_confidence_x,
            max(
                additive.lookahead_retained_authority_x,
                additive.residual_pursuit_authority_x,
            ),
        )
        self.assertAlmostEqual(additive.velocity_feedforward_confidence_x, 0.82)

    def test_residual_pursuit_external_gap_loss_release_and_mouse_clear_state(
        self,
    ) -> None:
        config = _test_config(
            maximum_observation_interval_seconds=0.040,
            stale_after_seconds=0.110,
            feedback_deadzone_pixels=4.5,
            maximum_velocity_feedforward_fraction=1.0,
            require_motion_corroboration_for_feedforward=True,
            maximum_body_derived_feedforward_fraction=0.50,
            maximum_body_derived_pursuit_feedforward_fraction=0.90,
            maximum_residual_pursuit_feedforward_fraction=0.80,
        )

        def ready_controller() -> MakcuCalibratedController:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.008),
                config,
            )
            deadline_ns = NS_PER_SECOND
            for index in range(3):
                timestamp_ns = index * 8 * NS_PER_MS
                point_x = 20.0 + index * 8.0
                controller.step(
                    timestamp_ns,
                    engaged=True,
                    observation=ScreenErrorObservation(
                        timestamp_ns,
                        point_x,
                        0.0,
                        velocity_error_x_pixels=point_x,
                        velocity_error_y_pixels=0.0,
                        body_derived_motion_permitted=True,
                        body_derived_motion_deadline_ns=deadline_ns,
                        identity_deadline_ns=deadline_ns,
                    ),
                )
            # Make each external lifecycle assertion independent of observer
            # convergence; qualification math is covered directly above.
            state = controller._residual_pursuit_x
            for _ in range(3):
                state = controller._residual_pursuit_axis_state(
                    state,
                    measurement_event=True,
                    current_measured=True,
                    elapsed=0.008,
                    measured_error=20.0,
                    projected_error=20.0,
                    projected_velocity=500.0,
                    position_sigma=2.0,
                    position_agreement=1.0,
                    revoke=False,
                )
            controller._residual_pursuit_x = state
            self.assertGreater(state.authority, 0.0)
            return controller

        released = ready_controller()
        released.step(17 * NS_PER_MS, engaged=False)
        self.assertEqual(released._residual_pursuit_x.authority, 0.0)

        lost = ready_controller()
        lost.step(17 * NS_PER_MS, engaged=True, target_lost=True)
        self.assertEqual(lost._residual_pursuit_x.authority, 0.0)

        gapped = ready_controller()
        gap_ns = 60 * NS_PER_MS
        gap = gapped.step(
            gap_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                gap_ns,
                80.0,
                0.0,
                velocity_error_x_pixels=80.0,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=NS_PER_SECOND,
                identity_deadline_ns=NS_PER_SECOND,
            ),
        )
        self.assertEqual(gap.reset_reason, "observation-gap")
        self.assertEqual(gapped._residual_pursuit_x.authority, 0.0)

        manual = ready_controller()
        manual.record_physical_input(17 * NS_PER_MS, 1, 0)
        pending = manual.step(18 * NS_PER_MS, engaged=True)
        self.assertTrue(pending.physical_input_pending_x)
        self.assertEqual(pending.residual_pursuit_authority_x, 0.0)
        self.assertEqual(manual._residual_pursuit_x.authority, 0.0)

    def test_independent_pursuit_promotes_verified_velocity_without_lag(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
            ),
        )
        update = controller._adaptive_body_pursuit_confidence
        common = dict(
            measured_error=0.0,
            projected_error=0.0,
            projected_vector_speed=500.0,
            motion_confidence=0.10,
            fresh_motion_confidence=1.0,
            direction_persistence_seconds=0.050,
            elapsed=0.008,
            qualification_zero_speed=250.0,
            qualification_full_speed=500.0,
            binary_promotion=True,
        )

        # Confidence is the authorization proof, not a second amplitude
        # control. Once qualified, full target-velocity authority is available
        # before positional lag has to build up.
        self.assertEqual(
            update(0.0, projected_velocity=500.0, **common),
            0.90,
        )
        self.assertAlmostEqual(
            update(
                0.0,
                projected_velocity=375.0,
                **{**common, "projected_vector_speed": 375.0},
            ),
            0.70,
        )

    def test_adaptive_position_reconciliation_is_motion_only_and_bounded(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.82,
            ),
        )
        reconcile = controller._adaptive_body_position_reconciliation
        # Coherent translation advances both source channels equally, so
        # reconciliation is exactly neutral even at full confidence.
        self.assertEqual(
            reconcile(
                18.0,
                18.0,
                1000.0,
                0.82,
                motion_confidence=0.40,
                fresh_motion_confidence=1.0,
            ),
            0.0,
        )
        local_phase_correction = reconcile(
            10.0,
            18.0,
            1000.0,
            0.82,
            motion_confidence=0.40,
            fresh_motion_confidence=1.0,
        )
        self.assertEqual(
            local_phase_correction,
            8.0,
        )
        # Full reconciliation reaches the other position channel; it cannot
        # double that local phase difference or lead beyond it.
        self.assertEqual(10.0 + local_phase_correction, 18.0)
        partial_local_phase_correction = reconcile(
            10.0,
            18.0,
            1000.0,
            0.41,
            motion_confidence=0.40,
            fresh_motion_confidence=1.0,
        )
        self.assertEqual(partial_local_phase_correction, 4.0)
        self.assertLess(
            10.0 + partial_local_phase_correction,
            18.0,
        )
        self.assertEqual(
            reconcile(
                10.0,
                50.0,
                1000.0,
                0.82,
                motion_confidence=0.40,
                fresh_motion_confidence=1.0,
            ),
            24.0,
        )
        for velocity_error, fresh in ((2.0, 1.0), (18.0, 0.0)):
            with self.subTest(velocity_error=velocity_error, fresh=fresh):
                self.assertEqual(
                    reconcile(
                        10.0,
                        velocity_error,
                        1000.0,
                        0.82,
                        motion_confidence=0.40,
                        fresh_motion_confidence=fresh,
                    ),
                    0.0,
                )

    def test_fast_body_feedforward_reserve_reaches_output_and_stops_fresh(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.006),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=(
                    40_000.0
                ),
                maximum_rate_x_counts_per_second=19_200.0,
                maximum_rate_y_counts_per_second=19_200.0,
                stale_after_seconds=0.110,
                maximum_observation_interval_seconds=0.040,
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.82,
            ),
        )
        deadline_ns = 2 * NS_PER_SECOND
        output = None
        # End at exact lock during coherent 1800 px/s translation. Ordinary
        # residual-gated feed-forward is zero there; only the new reserve can
        # prevent the controller from first falling behind again.
        for index in range(60):
            timestamp_ns = index * 8 * NS_PER_MS
            point_x = (index - 59) * 14.4
            output = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=deadline_ns,
                    identity_deadline_ns=deadline_ns,
                ),
            )

        assert output is not None
        self.assertTrue(output.valid)
        self.assertAlmostEqual(
            output.target_velocity_x_pixels_per_second,
            1800.0,
            delta=1e-4,
        )
        self.assertGreater(output.velocity_feedforward_confidence_x, 0.50)
        self.assertGreater(
            output.pursuit_reserve_rate_x_counts_per_second,
            0.0,
        )
        self.assertEqual(
            output.pursuit_reserve_rate_y_counts_per_second,
            0.0,
        )

        stop_ns = 60 * 8 * NS_PER_MS
        stopped = controller.step(
            stop_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                stop_ns,
                0.0,
                0.0,
                velocity_error_x_pixels=0.0,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=deadline_ns,
                identity_deadline_ns=deadline_ns,
            ),
        )
        # The observer intentionally still has velocity, proving newest-frame
        # physical motion—not slow velocity decay—closed the reserve.
        self.assertGreater(
            stopped.target_velocity_x_pixels_per_second,
            1000.0,
        )
        self.assertEqual(
            controller._body_derived_fresh_motion_confidence_x,
            0.0,
        )
        self.assertEqual(stopped.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(
            stopped.pursuit_reserve_rate_x_counts_per_second,
            0.0,
        )

    def test_body_derived_motion_is_separate_and_each_path_obeys_its_cap(
        self,
    ) -> None:
        """A mapped body point gets explicit bounded lead, never fake corr."""

        plant = CalibratedPlant(0.125, 0.120, 0.008)

        def controller(
            *,
            require_corroboration: bool,
            body_derived_projection: float = 0.0,
            body_derived_feedforward: float = 0.0,
            velocity_feedforward: float = 0.95,
        ) -> MakcuCalibratedController:
            return MakcuCalibratedController(
                plant,
                _test_config(
                    position_time_constant_seconds=0.012,
                    maximum_target_acceleration_pixels_per_second_squared=(
                        20_000.0
                    ),
                    stale_after_seconds=0.065,
                    maximum_observation_interval_seconds=0.065,
                    feedback_deadzone_pixels=3.0,
                    maximum_velocity_feedforward_fraction=velocity_feedforward,
                    require_motion_corroboration_for_feedforward=(
                        require_corroboration
                    ),
                    maximum_body_derived_projection_fraction=(
                        body_derived_projection
                    ),
                    maximum_body_derived_feedforward_fraction=(
                        body_derived_feedforward
                    ),
                ),
            )

        # Even an explicitly tagged/configured sample retains the historical
        # full paired behavior when the profile does not opt into automatic
        # corroboration-required control.
        full = controller(
            require_corroboration=False,
            body_derived_projection=0.25,
            body_derived_feedforward=0.25,
        )
        static = controller(require_corroboration=True)
        body_static = controller(require_corroboration=True)
        derived = controller(
            require_corroboration=True,
            body_derived_projection=0.25,
            body_derived_feedforward=0.25,
        )
        automatic = controller(
            require_corroboration=True,
            body_derived_projection=1.0,
            body_derived_feedforward=0.25,
        )
        projection_only = controller(
            require_corroboration=True,
            body_derived_projection=1.0,
            body_derived_feedforward=0.25,
            velocity_feedforward=0.0,
        )
        for index in range(41):
            timestamp_ns = index * 8 * NS_PER_MS
            point_x = (index - 40) * 4.6
            ordinary = ScreenErrorObservation(
                timestamp_ns,
                point_x,
                0.0,
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
            )
            full_output = full.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=(
                        timestamp_ns + 65 * NS_PER_MS
                    ),
                ),
            )
            static_output = static.step(
                timestamp_ns,
                engaged=True,
                observation=ordinary,
            )
            body_static_output = body_static.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=(
                        timestamp_ns + 65 * NS_PER_MS
                    ),
                ),
            )
            derived_output = derived.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=(
                        timestamp_ns + 65 * NS_PER_MS
                    ),
                ),
            )
            automatic_output = automatic.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=(
                        timestamp_ns + 65 * NS_PER_MS
                    ),
                ),
            )
            projection_only_output = projection_only.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=(
                        timestamp_ns + 65 * NS_PER_MS
                    ),
                ),
            )

        self.assertTrue(derived_output.valid)
        self.assertGreater(full_output.velocity_feedforward_confidence_x, 0.90)
        self.assertEqual(static_output.velocity_feedforward_confidence_x, 0.0)
        # This trajectory reaches the crosshair from the negative side. The
        # new cooperative-input guard removes explicit feed-forward at the
        # zero residual instead of retaining the old 25% baseline.
        self.assertEqual(derived_output.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(derived_output.motion_corroboration_confidence, 0.0)
        self.assertEqual(automatic_output.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(automatic_output.motion_corroboration_confidence, 0.0)
        self.assertEqual(
            projection_only_output.velocity_feedforward_confidence_x,
            0.0,
        )

        # The same observer state makes the positional-projection fraction
        # directly observable.  This closes the less obvious bypass where
        # v*horizon could retain full lead even though explicit FF was capped.
        full_motion_projection = (
            full_output.projected_error_x_pixels
            - static_output.projected_error_x_pixels
        )
        derived_motion_projection = (
            derived_output.projected_error_x_pixels
            - body_static_output.projected_error_x_pixels
        )
        automatic_motion_projection = (
            automatic_output.projected_error_x_pixels
            - body_static_output.projected_error_x_pixels
        )
        projection_only_motion = (
            projection_only_output.projected_error_x_pixels
            - body_static_output.projected_error_x_pixels
        )
        self.assertGreater(full_motion_projection, 4.0)
        self.assertGreater(automatic_motion_projection, 4.0)
        self.assertAlmostEqual(
            derived_motion_projection / automatic_motion_projection,
            0.25,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            projection_only_motion / automatic_motion_projection,
            1.0,
            delta=1e-12,
        )

        # Source age grows the full v*horizon term.  The hard ratio must stay
        # 0.25 rather than being promoted by observer confidence or age.
        full_bridge = full.step(352 * NS_PER_MS, engaged=True)
        static_bridge = static.step(352 * NS_PER_MS, engaged=True)
        body_static_bridge = body_static.step(352 * NS_PER_MS, engaged=True)
        derived_bridge = derived.step(352 * NS_PER_MS, engaged=True)
        automatic_bridge = automatic.step(352 * NS_PER_MS, engaged=True)
        full_bridge_projection = (
            full_bridge.projected_error_x_pixels
            - static_bridge.projected_error_x_pixels
        )
        derived_bridge_projection = (
            derived_bridge.projected_error_x_pixels
            - body_static_bridge.projected_error_x_pixels
        )
        automatic_bridge_projection = (
            automatic_bridge.projected_error_x_pixels
            - body_static_bridge.projected_error_x_pixels
        )
        self.assertGreater(full_bridge_projection, 22.0)
        self.assertLessEqual(
            derived_bridge_projection / automatic_bridge_projection,
            0.25,
        )
        self.assertAlmostEqual(
            automatic_bridge_projection / derived_bridge_projection,
            4.0,
            delta=1e-12,
        )
        self.assertEqual(derived_bridge.motion_corroboration_confidence, 0.0)
        self.assertLessEqual(
            derived_bridge.velocity_feedforward_confidence_x,
            0.25,
        )

    def test_extra_body_projection_requires_grant_and_axis_confidence(self) -> None:
        plant = CalibratedPlant(0.125, 0.120, 0.006)

        def make_controller(
            additional_projection_seconds: float,
        ) -> MakcuCalibratedController:
            return MakcuCalibratedController(
                plant,
                _test_config(
                    position_time_constant_seconds=0.018,
                    velocity_filter_time_constant_seconds=0.012,
                    maximum_target_acceleration_pixels_per_second_squared=(
                        40_000.0
                    ),
                    stale_after_seconds=0.110,
                    maximum_observation_interval_seconds=0.040,
                    require_motion_corroboration_for_feedforward=True,
                    maximum_body_derived_projection_fraction=1.0,
                    maximum_body_derived_feedforward_fraction=0.0,
                    additional_body_derived_projection_seconds=(
                        additional_projection_seconds
                    ),
                ),
            )

        # The option cannot affect an ordinary paired sample: body provenance
        # is a hard boundary even when the observer has learned clear motion.
        ordinary = make_controller(0.0)
        ordinary_extra = make_controller(0.012)
        for index in range(25):
            timestamp_ns = index * 8 * NS_PER_MS
            point_x = (index - 24) * 4.6
            observation = ScreenErrorObservation(
                timestamp_ns,
                point_x,
                0.0,
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
            )
            ordinary_output = ordinary.step(
                timestamp_ns,
                engaged=True,
                observation=observation,
            )
            ordinary_extra_output = ordinary_extra.step(
                timestamp_ns,
                engaged=True,
                observation=observation,
            )
            self.assertEqual(
                ordinary_extra_output.projected_error_x_pixels,
                ordinary_output.projected_error_x_pixels,
            )
            self.assertEqual(
                ordinary_extra_output.projected_error_y_pixels,
                ordinary_output.projected_error_y_pixels,
            )

        baseline = make_controller(0.0)
        extra = make_controller(0.012)
        observed_zero_confidence_with_motion = False
        observed_positive_confidence = False
        for index in range(41):
            timestamp_ns = index * 8 * NS_PER_MS
            point_x = (index - 40) * 4.6
            observation = ScreenErrorObservation(
                timestamp_ns,
                point_x,
                0.0,
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=(
                    timestamp_ns + 110 * NS_PER_MS
                ),
            )
            baseline_output = baseline.step(
                timestamp_ns,
                engaged=True,
                observation=observation,
            )
            extra_output = extra.step(
                timestamp_ns,
                engaged=True,
                observation=observation,
            )
            confidence = extra_output.body_derived_motion_confidence_x
            if (
                confidence == 0.0
                and abs(extra_output.target_velocity_x_pixels_per_second) > 1.0
            ):
                observed_zero_confidence_with_motion = True
                self.assertEqual(
                    extra_output.projected_error_x_pixels,
                    baseline_output.projected_error_x_pixels,
                )
            if confidence > 0.0:
                observed_positive_confidence = True
                # The only delta is the configured 12 ms positional horizon,
                # multiplied by the pre-existing per-axis confidence.
                self.assertAlmostEqual(
                    extra_output.projected_error_x_pixels
                    - baseline_output.projected_error_x_pixels,
                    confidence
                    * extra_output.target_velocity_x_pixels_per_second
                    * 0.012,
                    delta=1e-11,
                )

        self.assertTrue(observed_zero_confidence_with_motion)
        self.assertTrue(observed_positive_confidence)

    def test_body_derived_permission_fails_closed_on_absence_revoke_and_reject(
        self,
    ) -> None:
        def make_controller() -> MakcuCalibratedController:
            return MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.008),
                _test_config(
                    position_time_constant_seconds=0.012,
                    maximum_target_acceleration_pixels_per_second_squared=(
                        20_000.0
                    ),
                    stale_after_seconds=0.065,
                    maximum_observation_interval_seconds=0.065,
                    feedback_deadzone_pixels=3.0,
                    maximum_velocity_feedforward_fraction=0.95,
                    require_motion_corroboration_for_feedforward=True,
                    maximum_body_derived_projection_fraction=1.0,
                    maximum_body_derived_feedforward_fraction=0.25,
                ),
            )

        def observe(
            controller: MakcuCalibratedController,
            index: int,
            *,
            permitted: bool,
            point_x: float | None = None,
        ) -> CalibratedControlOutput:
            timestamp_ns = index * 8 * NS_PER_MS
            # Move away on the positive side so the ordinary grant first earns
            # aligned predictive authority; each following assertion can then
            # prove that its specific boundary withdraws that authority.
            measured_x = (index + 1) * 4.6 if point_x is None else point_x
            return controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    measured_x,
                    0.0,
                    velocity_error_x_pixels=measured_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=permitted,
                    body_derived_motion_deadline_ns=(
                        timestamp_ns + 65 * NS_PER_MS
                        if permitted
                        else None
                    ),
                ),
            )

        absent = make_controller()
        revoked = make_controller()
        rejected = make_controller()
        for index in range(41):
            absent_output = observe(absent, index, permitted=True)
            revoked_output = observe(revoked, index, permitted=True)
            rejected_output = observe(rejected, index, permitted=True)
        for output in (absent_output, revoked_output, rejected_output):
            self.assertAlmostEqual(
                output.velocity_feedforward_confidence_x,
                0.25,
                delta=1e-12,
            )

        # A new measured point with the provenance flag absent revokes before
        # that point can produce either kind of predictive motion.
        absent_output = observe(absent, 41, permitted=False)
        self.assertTrue(absent_output.valid)
        self.assertEqual(absent_output.velocity_feedforward_confidence_x, 0.0)
        self.assertAlmostEqual(
            absent_output.projected_error_x_pixels,
            absent._paired_position_x,  # noqa: SLF001 - numeric invariant test
            delta=1e-9,
        )

        # A primary prediction has no real observation to ingest.  The runtime
        # calls this narrow revoke before bridging the last measured position.
        revoked.revoke_body_derived_motion()
        revoked.revoke_body_derived_motion()  # Idempotent across repeated predictions.
        revoked_output = revoked.step(321 * NS_PER_MS, engaged=True)
        self.assertTrue(revoked_output.valid)
        self.assertEqual(revoked_output.velocity_feedforward_confidence_x, 0.0)
        self.assertAlmostEqual(
            revoked_output.projected_error_x_pixels,
            revoked._paired_position_x,  # noqa: SLF001 - numeric invariant test
            delta=1e-9,
        )

        # Even a sample which asks for permission cannot renew it when its aim
        # coordinate fails the paired observer's innovation gate.
        rejected_output = observe(
            rejected,
            41,
            permitted=True,
            point_x=1_000.0,
        )
        self.assertTrue(rejected_output.valid)
        self.assertTrue(rejected_output.innovation_rejected)
        self.assertEqual(rejected_output.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(rejected_output.motion_corroboration_confidence, 0.0)
        self.assertAlmostEqual(
            rejected_output.projected_error_x_pixels,
            rejected._paired_position_x,  # noqa: SLF001 - numeric invariant test
            delta=1e-9,
        )

        # The existing broad revoke used by no-head/prediction paths also
        # withdraws this mutually exclusive permission immediately.
        broad = make_controller()
        for index in range(41):
            broad_output = observe(broad, index, permitted=True)
        self.assertGreater(broad_output.velocity_feedforward_confidence_x, 0.0)
        broad.revoke_motion_corroboration()
        broad_output = broad.step(321 * NS_PER_MS, engaged=True)
        self.assertEqual(broad_output.velocity_feedforward_confidence_x, 0.0)
        self.assertAlmostEqual(
            broad_output.projected_error_x_pixels,
            broad._paired_position_x,  # noqa: SLF001 - numeric invariant test
            delta=1e-9,
        )

    def test_body_derived_motion_deadline_is_exact_and_keeps_static_lease(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.012,
                maximum_target_acceleration_pixels_per_second_squared=(
                    20_000.0
                ),
                stale_after_seconds=0.065,
                maximum_observation_interval_seconds=0.065,
                feedback_deadzone_pixels=4.0,
                maximum_velocity_feedforward_fraction=0.95,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.25,
            ),
        )
        motion_deadline_ns = 360 * NS_PER_MS
        identity_deadline_ns = 500 * NS_PER_MS
        for index in range(41):
            timestamp_ns = index * 8 * NS_PER_MS
            point_x = (index - 40) * 4.6
            output = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=motion_deadline_ns,
                    identity_deadline_ns=identity_deadline_ns,
                ),
            )
        self.assertTrue(output.valid)

        before = controller.step(motion_deadline_ns - 1, engaged=True)
        self.assertTrue(before.valid)
        self.assertGreater(before.projected_error_x_pixels, 20.0)
        self.assertGreater(before.rate_x_counts_per_second, 0.0)
        # The latest measured residual is exactly zero, so explicit feed-
        # forward is already withdrawn by the cooperative-input sign gate;
        # the still-live positional projection is what produces this command.
        self.assertEqual(before.velocity_feedforward_confidence_x, 0.0)

        at_deadline = controller.step(motion_deadline_ns, engaged=True)
        self.assertTrue(at_deadline.valid)
        self.assertTrue(controller.ready)
        self.assertIsNone(at_deadline.reset_reason)
        self.assertGreater(
            at_deadline.target_velocity_x_pixels_per_second,
            570.0,
        )
        self.assertEqual(at_deadline.motion_corroboration_confidence, 0.0)
        self.assertEqual(at_deadline.velocity_feedforward_confidence_x, 0.0)
        self.assertAlmostEqual(
            at_deadline.projected_error_x_pixels,
            0.0,
            delta=0.05,
        )
        self.assertEqual(at_deadline.rate_x_counts_per_second, 0.0)

        # The absolute mapped position remains usable through its ordinary
        # observation lease, but the expired predictive grant never returns.
        position_bridge = controller.step(384 * NS_PER_MS, engaged=True)
        self.assertTrue(position_bridge.valid)
        self.assertTrue(controller.ready)
        self.assertEqual(position_bridge.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(position_bridge.rate_x_counts_per_second, 0.0)

    def test_identity_deadline_zeros_all_output_at_exact_boundary(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                stale_after_seconds=0.065,
                maximum_observation_interval_seconds=0.065,
            ),
        )
        identity_deadline_ns = 50 * NS_PER_MS
        for timestamp_ns in (0, 8 * NS_PER_MS):
            output = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    20.0,
                    0.0,
                    identity_deadline_ns=identity_deadline_ns,
                ),
            )
        self.assertTrue(output.valid)

        before = controller.step(identity_deadline_ns - 1, engaged=True)
        self.assertTrue(before.valid)
        self.assertGreater(before.rate_x_counts_per_second, 0.0)

        expired = controller.step(identity_deadline_ns, engaged=True)
        self.assertFalse(expired.valid)
        self.assertEqual(expired.reset_reason, "identity-expired")
        self.assertEqual(expired.rate_x_counts_per_second, 0.0)
        self.assertEqual(expired.rate_y_counts_per_second, 0.0)
        self.assertFalse(controller.ready)

        starved = controller.step(identity_deadline_ns + 1, engaged=True)
        self.assertFalse(starved.valid)
        self.assertEqual(starved.reset_reason, "awaiting-observation")
        self.assertEqual(starved.rate_x_counts_per_second, 0.0)

        late_controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(),
        )
        expired_on_arrival = late_controller.step(
            identity_deadline_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                0,
                20.0,
                0.0,
                identity_deadline_ns=identity_deadline_ns,
            ),
        )
        self.assertFalse(expired_on_arrival.valid)
        self.assertEqual(expired_on_arrival.reset_reason, "identity-expired")
        self.assertEqual(expired_on_arrival.rate_x_counts_per_second, 0.0)

    def test_body_derived_projection_rejects_filtered_circular_point_jitter(
        self,
    ) -> None:
        """A live identity lease must not turn a small orbit into pursuit."""

        for amplitude in (1.0, 2.0, 3.0, 4.0):
            for jitter_hz in (4.0, 8.0, 12.0):
                with self.subTest(amplitude=amplitude, jitter_hz=jitter_hz):
                    controller = MakcuCalibratedController(
                        CalibratedPlant(0.125, 0.120, 0.008),
                        _test_config(
                            position_time_constant_seconds=0.012,
                            maximum_target_acceleration_pixels_per_second_squared=(
                                20_000.0
                            ),
                            maximum_rate_x_counts_per_second=19_200.0,
                            maximum_rate_y_counts_per_second=19_200.0,
                            stale_after_seconds=0.065,
                            maximum_observation_interval_seconds=0.040,
                            feedback_deadzone_pixels=4.0,
                            maximum_velocity_feedforward_fraction=0.95,
                            require_motion_corroboration_for_feedforward=True,
                            maximum_body_derived_projection_fraction=1.0,
                            maximum_body_derived_feedforward_fraction=0.25,
                        ),
                    )
                    filtered_x: float | None = None
                    filtered_y: float | None = None
                    alpha = 1.0 - math.exp(-(1.0 / 90.0) / 0.012)
                    outputs: list[CalibratedControlOutput] = []
                    for index in range(300):
                        source_ns = round(index * NS_PER_SECOND / 90.0)
                        phase = 2.0 * math.pi * jitter_hz * index / 90.0
                        mapped_x = amplitude * math.cos(phase)
                        mapped_y = amplitude * math.sin(phase)
                        if filtered_x is None or filtered_y is None:
                            filtered_x = mapped_x
                            filtered_y = mapped_y
                        else:
                            filtered_x += alpha * (mapped_x - filtered_x)
                            filtered_y += alpha * (mapped_y - filtered_y)
                        outputs.append(
                            controller.step(
                                source_ns + 54 * NS_PER_MS,
                                engaged=True,
                                observation=ScreenErrorObservation(
                                    source_ns,
                                    filtered_x,
                                    filtered_y,
                                    velocity_error_x_pixels=filtered_x,
                                    velocity_error_y_pixels=filtered_y,
                                    body_derived_motion_permitted=True,
                                    body_derived_motion_deadline_ns=(
                                        source_ns + 65 * NS_PER_MS
                                    ),
                                ),
                            )
                        )

                    steady = outputs[-180:]
                    self.assertEqual(
                        max(
                            output.motion_corroboration_confidence
                            for output in steady
                        ),
                        0.0,
                    )
                    self.assertEqual(
                        max(
                            output.velocity_feedforward_confidence_x
                            for output in steady
                        ),
                        0.0,
                    )
                    self.assertEqual(
                        max(
                            output.velocity_feedforward_confidence_y
                            for output in steady
                        ),
                        0.0,
                    )
                    self.assertEqual(
                        max(
                            abs(output.rate_x_counts_per_second)
                            + abs(output.rate_y_counts_per_second)
                            for output in steady
                        ),
                        0.0,
                    )

    def test_body_derived_dominant_axis_cannot_carry_other_axis_reversal(
        self,
    ) -> None:
        """A strong X pursuit must not authorize stale Y prediction."""

        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.006),
            _test_config(
                position_time_constant_seconds=0.045,
                maximum_rate_x_counts_per_second=12_000.0,
                maximum_rate_y_counts_per_second=12_000.0,
                stale_after_seconds=0.110,
                maximum_observation_interval_seconds=0.040,
                feedback_deadzone_pixels=2.5,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.25,
            ),
        )
        point_x = 0.0
        point_y = 0.0
        interval_seconds = 1.0 / 90.0
        for index in range(30):
            point_x += 1_400.0 * interval_seconds
            point_y += 575.0 * interval_seconds
            timestamp_ns = round(index * NS_PER_SECOND / 90.0)
            before = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    point_y,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=point_y,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
                ),
            )

        self.assertAlmostEqual(
            before.velocity_feedforward_confidence_x,
            0.25,
            delta=1e-9,
        )
        self.assertGreater(before.velocity_feedforward_confidence_y, 0.0)

        # On the very first opposing Y measurement the Kalman velocity still
        # points in the old positive direction.  The command-compensated
        # measured reversal must nevertheless revoke only Y prediction.
        point_x += 1_400.0 * interval_seconds
        point_y -= 575.0 * interval_seconds
        timestamp_ns = round(30 * NS_PER_SECOND / 90.0)
        reversed_y = controller.step(
            timestamp_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                timestamp_ns,
                point_x,
                point_y,
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=point_y,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
            ),
        )

        self.assertGreater(
            reversed_y.target_velocity_y_pixels_per_second,
            0.0,
        )
        self.assertEqual(reversed_y.velocity_feedforward_confidence_y, 0.0)
        self.assertAlmostEqual(
            reversed_y.projected_error_y_pixels,
            point_y,
            delta=1e-9,
        )
        self.assertAlmostEqual(
            reversed_y.velocity_feedforward_confidence_x,
            0.25,
            delta=1e-9,
        )
        self.assertGreater(reversed_y.projected_error_x_pixels, point_x)

    def test_body_derived_first_same_axis_reversal_has_no_stale_lead(self) -> None:
        """The smoothed old velocity cannot survive one reversed sample."""

        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.006),
            _test_config(
                position_time_constant_seconds=0.045,
                maximum_rate_x_counts_per_second=12_000.0,
                maximum_rate_y_counts_per_second=12_000.0,
                stale_after_seconds=0.110,
                maximum_observation_interval_seconds=0.040,
                feedback_deadzone_pixels=2.5,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.25,
            ),
        )
        point_x = 0.0
        interval_seconds = 1.0 / 90.0
        for index in range(30):
            point_x += 575.0 * interval_seconds
            timestamp_ns = round(index * NS_PER_SECOND / 90.0)
            before = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
                ),
            )
        self.assertAlmostEqual(
            before.velocity_feedforward_confidence_x,
            0.25,
            delta=1e-9,
        )

        point_x -= 575.0 * interval_seconds
        timestamp_ns = round(30 * NS_PER_SECOND / 90.0)
        reversed_x = controller.step(
            timestamp_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                timestamp_ns,
                point_x,
                0.0,
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
            ),
        )

        self.assertGreater(
            reversed_x.target_velocity_x_pixels_per_second,
            0.0,
        )
        self.assertEqual(reversed_x.velocity_feedforward_confidence_x, 0.0)
        self.assertAlmostEqual(
            reversed_x.projected_error_x_pixels,
            point_x,
            delta=1e-9,
        )

    def test_body_derived_direction_rejects_short_filtered_jitter(self) -> None:
        """Qualified mapped pursuit arms after 16 ms and is full at 50 ms."""

        direction_anchor = 0.0
        persistence_seconds = 0.0

        def advance(elapsed: float) -> float:
            nonlocal direction_anchor, persistence_seconds
            confidence, direction_anchor, persistence_seconds = (
                MakcuCalibratedController._body_derived_axis_confidence(
                    velocity=575.0,
                    source_confidence=1.0,
                    measured_motion_delta=5.0,
                    motion_innovation=0.0,
                    motion_innovation_sigma=1.0,
                    prior_velocity=575.0,
                    direction_anchor=direction_anchor,
                    direction_persistence_seconds=persistence_seconds,
                    elapsed=elapsed,
                )
            )
            return confidence

        self.assertEqual(advance(0.0), 0.0)  # Seed the direction.
        self.assertEqual(advance(0.016), 0.0)
        self.assertGreater(advance(0.001), 0.0)
        self.assertAlmostEqual(advance(0.033), 1.0, delta=1e-9)

    def test_body_derived_noisy_quantized_slow_motion_stays_armed(self) -> None:
        """Difference noise and one model-pixel stairs are not reversals."""

        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.006),
            _test_config(
                position_time_constant_seconds=0.045,
                maximum_rate_x_counts_per_second=12_000.0,
                maximum_rate_y_counts_per_second=12_000.0,
                stale_after_seconds=0.110,
                maximum_observation_interval_seconds=0.040,
                feedback_deadzone_pixels=2.5,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.25,
            ),
        )
        random_source = random.Random(3)
        model_pixel = 1920.0 / 416.0
        outputs: list[CalibratedControlOutput] = []
        for index in range(400):
            timestamp_ns = round(index * NS_PER_SECOND / 90.0)
            coherent_x = index * 300.0 / 90.0
            quantized_x = round(coherent_x / model_pixel) * model_pixel
            measured_x = quantized_x + random_source.gauss(0.0, 2.5)
            outputs.append(
                controller.step(
                    timestamp_ns,
                    engaged=True,
                    observation=ScreenErrorObservation(
                        timestamp_ns,
                        measured_x,
                        0.0,
                        velocity_error_x_pixels=measured_x,
                        velocity_error_y_pixels=0.0,
                        body_derived_motion_permitted=True,
                        body_derived_motion_deadline_ns=10 * NS_PER_SECOND,
                    ),
                )
            )

        steady = outputs[100:]
        self.assertGreater(
            min(output.target_velocity_x_pixels_per_second for output in steady),
            100.0,
        )
        self.assertGreater(
            min(output.body_derived_motion_confidence_x for output in steady),
            0.0,
        )
        self.assertGreater(
            min(output.velocity_feedforward_confidence_x for output in steady),
            0.0,
        )

    def test_body_derived_paired_noise_rate_is_normalized_above_46_hz(
        self,
    ) -> None:
        """Faster mapped publication must not manufacture observer certainty."""

        def run(
            rate_hz: float,
            *,
            body_derived: bool,
        ) -> CalibratedControlOutput:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.006),
                _test_config(
                    position_time_constant_seconds=0.045,
                    maximum_rate_x_counts_per_second=12_000.0,
                    maximum_rate_y_counts_per_second=12_000.0,
                    stale_after_seconds=0.110,
                    maximum_observation_interval_seconds=0.040,
                    feedback_deadzone_pixels=2.5,
                    require_motion_corroboration_for_feedforward=True,
                    maximum_body_derived_projection_fraction=0.0,
                    maximum_body_derived_feedforward_fraction=0.0,
                ),
            )
            output = None
            for index in range(120):
                timestamp_ns = round(index * NS_PER_SECOND / rate_hz)
                point_x = index * 575.0 / rate_hz
                output = controller.step(
                    timestamp_ns,
                    engaged=True,
                    observation=ScreenErrorObservation(
                        timestamp_ns,
                        point_x,
                        0.0,
                        velocity_error_x_pixels=point_x,
                        velocity_error_y_pixels=0.0,
                        body_derived_motion_permitted=body_derived,
                        body_derived_motion_deadline_ns=(
                            timestamp_ns + NS_PER_SECOND
                            if body_derived
                            else None
                        ),
                    ),
                )
            assert output is not None
            return output

        reference_body = run(46.0, body_derived=True)
        reference_plain = run(46.0, body_derived=False)
        fast_body = run(92.0, body_derived=True)
        fast_plain = run(92.0, body_derived=False)

        self.assertAlmostEqual(
            reference_body.observer_velocity_sigma_x_pixels_per_second,
            reference_plain.observer_velocity_sigma_x_pixels_per_second,
            delta=1e-5,
        )
        self.assertGreater(
            fast_body.observer_velocity_sigma_x_pixels_per_second,
            fast_plain.observer_velocity_sigma_x_pixels_per_second * 1.05,
        )
        self.assertGreater(
            fast_body.observer_position_sigma_x_pixels,
            fast_plain.observer_position_sigma_x_pixels * 1.20,
        )

    def test_independent_motion_corroboration_never_moves_direct_aim_point(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                maximum_velocity_feedforward_fraction=0.95,
                require_motion_corroboration_for_feedforward=True,
            ),
        )
        for index in range(12):
            timestamp_ns = index * 8 * NS_PER_MS
            output = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    10.0,
                    -5.0,
                    velocity_error_x_pixels=10.0,
                    velocity_error_y_pixels=-5.0,
                    # An arbitrary body-center offset is valid because this
                    # channel is evidence only, never a second aim coordinate.
                    corroboration_error_x_pixels=510.0,
                    corroboration_error_y_pixels=395.0,
                ),
            )
        self.assertTrue(output.valid)
        self.assertAlmostEqual(output.projected_error_x_pixels, 10.0, delta=0.01)
        self.assertAlmostEqual(output.projected_error_y_pixels, -5.0, delta=0.01)
        self.assertEqual(output.motion_corroboration_confidence, 0.0)
        self.assertEqual(output.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(output.velocity_feedforward_confidence_y, 0.0)

    def test_feedforward_requires_persistent_independently_corroborated_motion(
        self,
    ) -> None:
        def run(
            hz: float,
            point_at: object,
            body_at: object,
            *,
            samples: int = 180,
        ) -> list[CalibratedControlOutput]:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.008),
                _test_config(
                    maximum_target_acceleration_pixels_per_second_squared=(
                        20_000.0
                    ),
                    maximum_rate_x_counts_per_second=19_200.0,
                    maximum_rate_y_counts_per_second=19_200.0,
                    stale_after_seconds=0.065,
                    maximum_velocity_feedforward_fraction=0.95,
                    require_motion_corroboration_for_feedforward=True,
                ),
            )
            outputs: list[CalibratedControlOutput] = []
            for index in range(samples):
                timestamp_ns = round(index * NS_PER_SECOND / hz)
                point_x, point_y = point_at(index, hz)  # type: ignore[operator]
                body_x, body_y = body_at(index, hz)  # type: ignore[operator]
                outputs.append(
                    controller.step(
                        timestamp_ns,
                        engaged=True,
                        observation=ScreenErrorObservation(
                            timestamp_ns,
                            point_x,
                            point_y,
                            velocity_error_x_pixels=point_x,
                            velocity_error_y_pixels=point_y,
                            corroboration_error_x_pixels=body_x,
                            corroboration_error_y_pixels=body_y,
                        ),
                    )
                )
            return outputs

        for hz in (60.0, 120.0):
            with self.subTest(hz=hz, motion="linear"):
                linear = run(
                    hz,
                    lambda index, rate: (index * 575.0 / rate, 0.0),
                    lambda index, rate: (100.0 + index * 575.0 / rate, 200.0),
                )
                self.assertGreater(
                    min(
                        output.motion_corroboration_confidence
                        for output in linear[-30:]
                    ),
                    0.99,
                )
                self.assertGreater(
                    min(
                        output.velocity_feedforward_confidence_x
                        for output in linear[-30:]
                    ),
                    0.94,
                )

            with self.subTest(hz=hz, motion="opposed"):
                opposed = run(
                    hz,
                    lambda index, rate: (index * 575.0 / rate, 0.0),
                    lambda index, rate: (100.0 - index * 575.0 / rate, 200.0),
                )
                self.assertEqual(
                    max(
                        output.velocity_feedforward_confidence_x
                        for output in opposed[-60:]
                    ),
                    0.0,
                )

            with self.subTest(hz=hz, motion="correlated-circle"):
                circle = tuple(
                    (
                        8.0 * math.cos(index * math.pi / 8.0),
                        8.0 * math.sin(index * math.pi / 8.0),
                    )
                    for index in range(16)
                )
                circular = run(
                    hz,
                    lambda index, _rate: circle[index % len(circle)],
                    lambda index, _rate: (
                        circle[index % len(circle)][0] + 100.0,
                        circle[index % len(circle)][1] + 200.0,
                    ),
                    samples=480,
                )
                self.assertLess(
                    max(
                        output.motion_corroboration_confidence
                        for output in circular[-240:]
                    ),
                    0.002,
                )
                self.assertLess(
                    max(
                        output.velocity_feedforward_confidence_x
                        for output in circular[-240:]
                    ),
                    0.001,
                )

    def test_rejected_direct_head_immediately_revokes_corroborated_feedforward(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                maximum_target_acceleration_pixels_per_second_squared=(
                    20_000.0
                ),
                stale_after_seconds=0.065,
                maximum_velocity_feedforward_fraction=0.95,
                require_motion_corroboration_for_feedforward=True,
            ),
        )
        for index in range(40):
            timestamp_ns = index * 8 * NS_PER_MS
            point_x = index * 4.6
            accepted = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    corroboration_error_x_pixels=point_x + 100.0,
                    corroboration_error_y_pixels=200.0,
                ),
            )
        self.assertGreater(accepted.velocity_feedforward_confidence_x, 0.90)

        timestamp_ns = 40 * 8 * NS_PER_MS
        rejected = controller.step(
            timestamp_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                timestamp_ns,
                1_000.0,
                0.0,
                velocity_error_x_pixels=1_000.0,
                velocity_error_y_pixels=0.0,
                corroboration_error_x_pixels=40 * 4.6 + 100.0,
                corroboration_error_y_pixels=200.0,
            ),
        )
        self.assertTrue(rejected.valid)
        self.assertTrue(rejected.innovation_rejected)
        self.assertEqual(rejected.motion_corroboration_confidence, 0.0)
        self.assertEqual(rejected.velocity_feedforward_confidence_x, 0.0)

    def test_corroboration_handoff_keeps_primary_live_and_bounds_retained_pursuit(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.82,
            ),
        )

        def observe(index: int, provenance: str) -> CalibratedControlOutput:
            timestamp_ns = index * 8 * NS_PER_MS
            point_x = 40.0 + index * 4.6
            common = dict(
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
            )
            if provenance == "independent":
                observation = ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    corroboration_error_x_pixels=point_x + 100.0,
                    corroboration_error_y_pixels=200.0,
                    **common,
                )
            else:
                observation = ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
                    **common,
                )
            return controller.step(
                timestamp_ns,
                engaged=True,
                observation=observation,
            )

        for index in range(64):
            qualified = observe(index, "independent")
        self.assertTrue(qualified.valid)
        self.assertGreater(qualified.velocity_feedforward_confidence_x, 0.70)
        qualified_persistence = (
            controller._common_pursuit_direction_persistence_seconds
        )
        self.assertGreater(qualified_persistence, 0.050)

        # A phase-flow dropout and return are authority changes, not primary
        # position schema changes.  Neither transition may manufacture the
        # zero-rate confirmation frames seen in the live trace, and the already
        # qualified pursuit fraction may bridge the short handoff.
        for index in (64, 65):
            fallback = observe(index, "body")
            self.assertTrue(fallback.valid)
            self.assertIsNone(fallback.reset_reason)
        returned = observe(66, "independent")
        self.assertTrue(returned.valid)
        self.assertIsNone(returned.reset_reason)
        self.assertGreater(returned.motion_corroboration_confidence, 0.90)
        self.assertGreater(returned.velocity_feedforward_confidence_x, 0.50)
        self.assertGreaterEqual(
            controller._common_pursuit_direction_persistence_seconds,
            qualified_persistence,
        )

        # A longer body-only interval cannot inherit that independent near-unity
        # floor forever.  Once the bounded handoff expires, only the ordinary
        # independently gated body path remains at this sub-900 px/s speed.
        for index in range(67, 75):
            expired = observe(index, "body")
        self.assertTrue(expired.valid)
        self.assertIsNone(expired.reset_reason)
        self.assertLessEqual(expired.velocity_feedforward_confidence_x, 0.50)
        self.assertEqual(
            controller._body_derived_adaptive_pursuit_confidence_x,
            0.0,
        )

        # Returning after expiry still keeps position/control continuous, but
        # independent velocity authority must requalify from exact zero.
        requalifying = observe(75, "independent")
        self.assertTrue(requalifying.valid)
        self.assertIsNone(requalifying.reset_reason)
        self.assertEqual(requalifying.motion_corroboration_confidence, 0.0)
        self.assertEqual(requalifying.velocity_feedforward_confidence_x, 0.0)

        for index in range(76, 91):
            requalified = observe(index, "independent")
        self.assertGreater(requalified.velocity_feedforward_confidence_x, 0.50)
        observe(91, "body")

        # One repeated coordinate is ambiguous at detector precision: it may be
        # a true stop or one quantized no-motion frame during continued travel.
        # Drop the direction proof immediately, but bridge the already-earned
        # rate for this one sample so a single staircase frame cannot recreate
        # the live on/off pursuit. A second repeated sample confirms the stop
        # and must revoke the reserve completely.
        stop_ns = 92 * 8 * NS_PER_MS
        stopped_x = 40.0 + 91 * 4.6
        stopped = controller.step(
            stop_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                stop_ns,
                stopped_x,
                0.0,
                velocity_error_x_pixels=stopped_x,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
            ),
        )
        self.assertTrue(stopped.valid)
        self.assertIsNone(stopped.reset_reason)
        self.assertEqual(
            controller._common_pursuit_direction_persistence_seconds,
            0.0,
        )
        self.assertGreater(
            controller._body_derived_adaptive_pursuit_confidence_x,
            0.50,
        )
        confirmed_stop_ns = 93 * 8 * NS_PER_MS
        confirmed_stop = controller.step(
            confirmed_stop_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                confirmed_stop_ns,
                stopped_x,
                0.0,
                velocity_error_x_pixels=stopped_x,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
            ),
        )
        self.assertTrue(confirmed_stop.valid)
        self.assertIsNone(confirmed_stop.reset_reason)
        self.assertEqual(
            controller._body_derived_adaptive_pursuit_confidence_x,
            0.0,
        )
        self.assertEqual(
            confirmed_stop.pursuit_reserve_rate_x_counts_per_second,
            0.0,
        )

    def test_sparse_corroboration_survives_interleaved_primary_publications(
        self,
    ) -> None:
        """A 45 Hz reference must remain observable inside a 90 Hz stream."""

        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
            ),
        )

        outputs: list[CalibratedControlOutput] = []
        for index in range(96):
            timestamp_ns = index * 8 * NS_PER_MS
            point_x = 40.0 + index * 4.8
            common = dict(
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
            )
            if index % 2 == 0:
                observation = ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    corroboration_error_x_pixels=point_x + 100.0,
                    corroboration_error_y_pixels=200.0,
                    **common,
                )
            else:
                observation = ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
                    **common,
                )
            outputs.append(
                controller.step(
                    timestamp_ns,
                    engaged=True,
                    observation=observation,
                )
            )

        tail = outputs[-24:]
        independent_tail = tail[::2]
        self.assertTrue(all(output.valid for output in tail))
        self.assertGreater(
            min(
                output.motion_corroboration_confidence
                for output in independent_tail
            ),
            0.90,
        )
        self.assertGreater(
            min(
                output.velocity_feedforward_confidence_x
                for output in tail
            ),
            0.75,
        )

        # Missing evidence still withdraws its current-frame grant.  A long
        # absence expires the latent observer and the first returning physical
        # point only seeds it; sparse continuity is not an indefinite lease.
        last_point_x = 40.0 + 95 * 4.8
        for index in range(96, 111):
            timestamp_ns = index * 8 * NS_PER_MS
            last_point_x += 4.8
            expired = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    last_point_x,
                    0.0,
                    velocity_error_x_pixels=last_point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
                ),
            )
        self.assertTrue(expired.valid)
        returned_ns = 111 * 8 * NS_PER_MS
        last_point_x += 4.8
        returned = controller.step(
            returned_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                returned_ns,
                last_point_x,
                0.0,
                velocity_error_x_pixels=last_point_x,
                velocity_error_y_pixels=0.0,
                corroboration_error_x_pixels=last_point_x + 100.0,
                corroboration_error_y_pixels=200.0,
            ),
        )
        self.assertTrue(returned.valid)
        self.assertEqual(returned.motion_corroboration_confidence, 0.0)

    def test_independent_contradiction_revokes_binary_pursuit_exactly(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
            ),
        )
        point_x = 40.0
        for index in range(64):
            if index:
                point_x += 4.8
            timestamp_ns = index * 8 * NS_PER_MS
            qualified = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    corroboration_error_x_pixels=point_x + 100.0,
                    corroboration_error_y_pixels=200.0,
                ),
            )
        self.assertGreater(qualified.velocity_feedforward_confidence_x, 0.85)

        # The primary point keeps moving while the independent reference is
        # physically fixed. Its diagnostic EMA decays asymptotically, but once
        # current agreement reaches exact zero no historical epsilon may keep
        # the binary 90% setpoint alive.
        fixed_corroboration_x = point_x + 100.0
        for index in range(64, 71):
            point_x += 4.8
            timestamp_ns = index * 8 * NS_PER_MS
            contradicted = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    corroboration_error_x_pixels=fixed_corroboration_x,
                    corroboration_error_y_pixels=200.0,
                ),
            )
        self.assertTrue(contradicted.valid)
        self.assertFalse(controller._independent_pursuit_authorized)
        self.assertEqual(contradicted.motion_corroboration_confidence, 0.0)
        self.assertEqual(contradicted.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(
            controller._body_derived_adaptive_pursuit_confidence_x,
            0.0,
        )

    def test_corroboration_dropout_without_body_has_no_handoff_lease(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
            ),
        )
        point_x = 40.0
        for index in range(64):
            if index:
                point_x += 4.8
            timestamp_ns = index * 8 * NS_PER_MS
            qualified = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    corroboration_error_x_pixels=point_x + 100.0,
                    corroboration_error_y_pixels=200.0,
                ),
            )
        self.assertGreater(qualified.velocity_feedforward_confidence_x, 0.85)

        point_x += 4.8
        dropout_ns = 64 * 8 * NS_PER_MS
        dropout = controller.step(
            dropout_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                dropout_ns,
                point_x,
                0.0,
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
            ),
        )
        self.assertTrue(dropout.valid)
        self.assertIsNone(controller._common_pursuit_handoff_deadline_ns)
        self.assertEqual(dropout.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(
            controller._body_derived_adaptive_pursuit_confidence_x,
            0.0,
        )

    def test_fresh_independent_source_must_earn_its_own_direction_horizon(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
            ),
        )
        point_x = 40.0
        for index in range(64):
            if index:
                point_x += 4.8
            timestamp_ns = index * 8 * NS_PER_MS
            controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
                ),
            )
        self.assertGreater(
            controller._common_pursuit_direction_persistence_seconds,
            0.40,
        )
        self.assertEqual(
            controller._body_derived_adaptive_pursuit_confidence_x,
            0.0,
        )

        saw_partial_independent_proof = False
        fully_qualified = None
        for index in range(64, 80):
            point_x += 4.8
            timestamp_ns = index * 8 * NS_PER_MS
            output = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    corroboration_error_x_pixels=point_x + 100.0,
                    corroboration_error_y_pixels=200.0,
                ),
            )
            independent_persistence = (
                controller._corroboration_direction_persistence_seconds
            )
            if 0.0 < independent_persistence < 0.050:
                saw_partial_independent_proof = True
                self.assertEqual(
                    controller._body_derived_adaptive_pursuit_confidence_x,
                    0.0,
                )
            elif independent_persistence >= 0.050:
                fully_qualified = output
                break

        self.assertTrue(saw_partial_independent_proof)
        self.assertIsNotNone(fully_qualified)
        assert fully_qualified is not None
        self.assertGreater(
            controller._body_derived_adaptive_pursuit_confidence_x,
            0.85,
        )
        self.assertGreater(
            fully_qualified.velocity_feedforward_confidence_x,
            0.85,
        )

    def test_material_reversal_masks_all_velocity_during_body_handoff(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
            ),
        )
        point_x = 40.0
        for index in range(64):
            if index:
                point_x += 4.8
            timestamp_ns = index * 8 * NS_PER_MS
            qualified = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    corroboration_error_x_pixels=point_x + 100.0,
                    corroboration_error_y_pixels=200.0,
                ),
            )
        self.assertGreater(qualified.velocity_feedforward_confidence_x, 0.85)

        # Enter the valid body handoff, then provide a statistically material
        # opposite step. This must mask the sparse ordinary term as well as the
        # adaptive reserve and source-age projection on that same sample.
        point_x += 4.8
        handoff_ns = 64 * 8 * NS_PER_MS
        controller.step(
            handoff_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                handoff_ns,
                point_x,
                0.0,
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
            ),
        )
        point_x -= 8.0
        reversal_ns = 65 * 8 * NS_PER_MS
        reversed_output = controller.step(
            reversal_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                reversal_ns,
                point_x,
                0.0,
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=2 * NS_PER_SECOND,
            ),
        )
        self.assertTrue(reversed_output.valid)
        self.assertTrue(controller._paired_material_stop_or_reversal_x)
        self.assertEqual(reversed_output.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(
            reversed_output.pursuit_reserve_rate_x_counts_per_second,
            0.0,
        )
        self.assertEqual(
            controller._body_derived_adaptive_pursuit_confidence_x,
            0.0,
        )
        self.assertAlmostEqual(
            reversed_output.projected_error_x_pixels,
            point_x,
            delta=1e-6,
        )

    def test_legacy_paired_reversal_does_not_publish_automatic_revoke_edge(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                require_motion_corroboration_for_feedforward=False,
            ),
        )
        point_x = 40.0
        for index in range(64):
            if index:
                point_x += 4.8
            timestamp_ns = index * 8 * NS_PER_MS
            controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                ),
            )

        point_x -= 8.0
        reversal_ns = 64 * 8 * NS_PER_MS
        legacy_output = controller.step(
            reversal_ns,
            engaged=True,
            observation=ScreenErrorObservation(
                reversal_ns,
                point_x,
                0.0,
                velocity_error_x_pixels=point_x,
                velocity_error_y_pixels=0.0,
            ),
        )
        self.assertTrue(legacy_output.valid)
        self.assertTrue(controller._paired_material_stop_or_reversal_x)
        self.assertFalse(legacy_output.material_motion_revoked_x)
        self.assertGreater(
            legacy_output.velocity_feedforward_confidence_x,
            0.0,
        )

    def test_explicit_corroboration_revoke_preserves_position_lease_and_reseeds(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.022,
                maximum_target_acceleration_pixels_per_second_squared=(
                    20_000.0
                ),
                stale_after_seconds=0.065,
                maximum_observation_interval_seconds=0.065,
                feedback_deadzone_pixels=3.0,
                maximum_velocity_feedforward_fraction=0.95,
                require_motion_corroboration_for_feedforward=True,
            ),
        )

        def observe(index: int) -> CalibratedControlOutput:
            timestamp_ns = index * 8 * NS_PER_MS
            # End the accepted run exactly centered while velocity remains
            # about 575 px/s. This catches the subtle bypass where retained
            # velocity entered projected positional feedback even with the
            # explicit feed-forward confidence already at zero.
            point_x = (index - 40) * 4.6
            return controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    corroboration_error_x_pixels=point_x + 100.0,
                    corroboration_error_y_pixels=200.0,
                ),
            )

        for index in range(41):
            before_revoke = observe(index)
        self.assertTrue(controller.ready)
        self.assertGreater(before_revoke.velocity_feedforward_confidence_x, 0.90)
        self.assertGreater(
            before_revoke.target_velocity_x_pixels_per_second,
            570.0,
        )

        controller.revoke_motion_corroboration()
        controller.revoke_motion_corroboration()  # Explicitly idempotent.
        # The last direct-head point is exactly centered while its retained
        # observer velocity is about 575 px/s. Across the entire valid lease,
        # neither processing age nor the calibrated 8 ms delay may leak that
        # uncorroborated velocity through projected positional feedback.
        for age_ms in (1, 8, 32, 60):
            bridged = controller.step(
                (320 + age_ms) * NS_PER_MS,
                engaged=True,
            )
            with self.subTest(age_ms=age_ms):
                self.assertTrue(controller.ready)
                self.assertTrue(bridged.valid)
                self.assertIsNone(bridged.reset_reason)
                self.assertGreater(
                    bridged.target_velocity_x_pixels_per_second,
                    570.0,
                )
                self.assertEqual(bridged.motion_corroboration_confidence, 0.0)
                self.assertEqual(bridged.velocity_feedforward_confidence_x, 0.0)
                self.assertAlmostEqual(
                    bridged.projected_error_x_pixels,
                    0.0,
                    delta=1e-6,
                )
                self.assertEqual(bridged.rate_x_counts_per_second, 0.0)

        # The first returning body point seeds only its independent observer;
        # it cannot immediately restore held confidence. Direct-head position
        # remains valid throughout rather than requiring a new two-frame lease.
        first_return = observe(48)
        self.assertTrue(first_return.valid)
        self.assertTrue(controller.ready)
        self.assertEqual(first_return.motion_corroboration_confidence, 0.0)
        self.assertEqual(first_return.velocity_feedforward_confidence_x, 0.0)

        for index in range(49, 79):
            reacquired = observe(index)
        self.assertGreater(reacquired.motion_corroboration_confidence, 0.90)
        self.assertGreater(reacquired.velocity_feedforward_confidence_x, 0.85)

    def test_profiles_without_corroboration_requirement_keep_velocity_projection(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.022,
                maximum_target_acceleration_pixels_per_second_squared=(
                    20_000.0
                ),
                stale_after_seconds=0.065,
                maximum_velocity_feedforward_fraction=0.95,
                require_motion_corroboration_for_feedforward=False,
            ),
        )
        for index in range(41):
            timestamp_ns = index * 8 * NS_PER_MS
            point_x = (index - 40) * 4.6
            output = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                ),
            )
        self.assertTrue(output.valid)
        self.assertAlmostEqual(output.projected_error_x_pixels, 4.6, delta=0.2)

        bridged = controller.step(352 * NS_PER_MS, engaged=True)
        self.assertTrue(bridged.valid)
        self.assertGreater(bridged.target_velocity_x_pixels_per_second, 570.0)
        self.assertGreater(bridged.projected_error_x_pixels, 22.0)
        self.assertGreater(bridged.velocity_feedforward_confidence_x, 0.0)
        self.assertGreater(bridged.rate_x_counts_per_second, 0.0)

    def test_automatic_empty_only_bridge_remains_valid_with_processing_age(self) -> None:
        from main import _automatic_plant_aware_controller

        tracker = TargetTracker(label="player", lost_grace_frames=3)
        controller = _automatic_plant_aware_controller(max_step=320)
        frame_shape = (1080, 1920, 3)
        target = Detection(0, "player", 0.9, (900.0, 280.0, 1100.0, 880.0))
        base_ns = NS_PER_SECOND

        def publish(
            source_ms: int,
            detections: tuple[Detection, ...],
        ) -> tuple[Detection | None, CalibratedControlOutput]:
            source_ns = base_ns + source_ms * NS_PER_MS
            tracked = tracker.update(
                detections,
                frame_shape,
                measurement_ns=source_ns,
            )
            observation = None
            if tracked is not None and not tracker.output_is_prediction:
                point_x, point_y = head_target_point(tracked, 0.12)
                observation = ScreenErrorObservation(
                    source_ns,
                    point_x - frame_shape[1] / 2.0,
                    point_y - frame_shape[0] / 2.0,
                )
            output = controller.step(
                source_ns + 12 * NS_PER_MS,
                engaged=True,
                observation=observation,
                target_lost=tracked is None,
            )
            return tracked, output

        _first_target, first = publish(0, (target,))
        self.assertFalse(first.valid)
        _second_target, second = publish(8, (target,))
        self.assertTrue(second.valid)

        # The physical run's common empty intervals were around 50 ms, while
        # completed detector results arrived about 12 ms after their source
        # timestamp. The numeric lease must cover both without treating a
        # synthetic tracker prediction as a new observation.
        for source_ms in (16, 32, 48, 58):
            with self.subTest(source_ms=source_ms):
                predicted, output = publish(source_ms, ())
                self.assertIsNotNone(predicted)
                self.assertTrue(tracker.output_is_prediction)
                self.assertTrue(output.valid)

        expired, output = publish(59, ())
        self.assertIsNone(expired)
        self.assertFalse(output.valid)
        self.assertEqual(output.reset_reason, "target-lost")

    def test_automatic_numeric_lease_expires_after_one_hundred_ten_ms(self) -> None:
        from main import _automatic_plant_aware_controller

        controller = _automatic_plant_aware_controller(max_step=320)
        base_ns = NS_PER_SECOND
        controller.step(
            base_ns + 12 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(base_ns, 40.0, -20.0),
        )
        ready = controller.step(
            base_ns + 20 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(
                base_ns + 8 * NS_PER_MS,
                40.0,
                -20.0,
            ),
        )
        self.assertTrue(ready.valid)

        at_boundary = controller.step(
            base_ns + 118 * NS_PER_MS,
            engaged=True,
        )
        self.assertTrue(at_boundary.valid)
        expired = controller.step(
            base_ns + 118 * NS_PER_MS + 1,
            engaged=True,
        )
        self.assertFalse(expired.valid)
        self.assertEqual(expired.reset_reason, "stale-observation")

    def test_delay_corrected_velocity_uses_only_actual_emitted_counts(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.20, 0.012),
            _test_config(
                velocity_filter_time_constant_seconds=0.0001,
                maximum_target_acceleration_pixels_per_second_squared=1e9,
                velocity_median_window=1,
            ),
        )
        first = controller.step(
            100 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(100 * NS_PER_MS, 10.0, -4.0),
        )
        self.assertFalse(first.valid)
        # Twenty X and -10 Y counts become visible at 117 ms.  During the 20 ms
        # interval the target moves (+8, -6) px, so observed error moves only
        # (+6, -4): target motion minus each unequal calibrated response.
        second = controller.step(
            120 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(120 * NS_PER_MS, 16.0, -8.0),
            emitted_commands=(
                EmittedMouseCommand(105 * NS_PER_MS, 20, -10),
            ),
        )
        self.assertTrue(second.valid)
        self.assertAlmostEqual(
            second.target_velocity_x_pixels_per_second,
            400.0,
            delta=0.1,
        )
        self.assertAlmostEqual(
            second.target_velocity_y_pixels_per_second,
            -300.0,
            delta=0.1,
        )

    def test_raw_velocity_channel_does_not_differentiate_smoothed_plant_lag(
        self,
    ) -> None:
        config = _test_config(
            velocity_filter_time_constant_seconds=0.0001,
            maximum_target_acceleration_pixels_per_second_squared=1e9,
            velocity_median_window=1,
        )

        def controller() -> MakcuCalibratedController:
            return MakcuCalibratedController(
                CalibratedPlant(0.10, 0.20, 0.0),
                config,
            )

        raw = controller()
        raw.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(
                0,
                10.0,
                0.0,
                velocity_error_x_pixels=10.0,
                velocity_error_y_pixels=0.0,
            ),
        )
        # Twenty physical X counts have already moved the raw point from 10
        # to 8 px. The downstream position filter has not reflected them yet
        # and still reports 10 px. Raw plant correction must reconstruct zero
        # independent target motion instead of calling those counts velocity.
        raw_output = raw.step(
            20 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(
                20 * NS_PER_MS,
                10.0,
                0.0,
                velocity_error_x_pixels=8.0,
                velocity_error_y_pixels=0.0,
            ),
            emitted_commands=(EmittedMouseCommand(1 * NS_PER_MS, 20, 0),),
        )
        self.assertTrue(raw_output.valid)
        self.assertAlmostEqual(
            raw_output.target_velocity_x_pixels_per_second,
            0.0,
            delta=0.01,
        )
        self.assertGreater(raw_output.rate_x_counts_per_second, 0.0)

        legacy = controller()
        legacy.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(0, 10.0, 0.0),
        )
        legacy_output = legacy.step(
            20 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(20 * NS_PER_MS, 10.0, 0.0),
            emitted_commands=(EmittedMouseCommand(1 * NS_PER_MS, 20, 0),),
        )
        self.assertGreater(
            legacy_output.target_velocity_x_pixels_per_second,
            99.0,
        )

    def test_velocity_channel_change_reseeds_before_emitting(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.20, 0.0),
            _test_config(velocity_median_window=3),
        )
        controller.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(0, 10.0, 0.0),
        )

        changed = controller.step(
            10 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(
                10 * NS_PER_MS,
                10.0,
                0.0,
                velocity_error_x_pixels=10.0,
                velocity_error_y_pixels=0.0,
            ),
        )

        self.assertFalse(changed.valid)
        self.assertEqual(changed.reset_reason, "velocity-channel-change")
        self.assertFalse(controller.ready)

    def test_paired_observer_uses_one_state_for_linear_position_and_velocity(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.20, 0.0),
            _test_config(
                velocity_filter_time_constant_seconds=0.0001,
                maximum_target_acceleration_pixels_per_second_squared=1e9,
                velocity_median_window=5,
            ),
        )
        for index in range(50):
            timestamp_ns = index * 10 * NS_PER_MS
            output = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    index * 8.0,
                    index * -3.0,
                    velocity_error_x_pixels=index * 8.0,
                    velocity_error_y_pixels=index * -3.0,
                ),
            )

        self.assertTrue(output.valid)
        self.assertAlmostEqual(
            output.target_velocity_x_pixels_per_second,
            800.0,
            delta=0.1,
        )
        self.assertAlmostEqual(
            output.target_velocity_y_pixels_per_second,
            -300.0,
            delta=0.1,
        )
        # Position and velocity are projections of the same raw-point state,
        # not the old smoothed-position plus independent OLS derivative split.
        self.assertAlmostEqual(output.projected_error_x_pixels, 392.0, delta=0.01)
        self.assertAlmostEqual(output.projected_error_y_pixels, -147.0, delta=0.01)
        self.assertGreater(output.observer_position_sigma_x_pixels, 0.0)
        self.assertGreater(
            output.observer_velocity_sigma_x_pixels_per_second,
            0.0,
        )
        self.assertGreater(output.velocity_feedforward_confidence_x, 0.99)
        self.assertGreater(output.velocity_feedforward_confidence_y, 0.99)

    def test_paired_observer_attenuates_circular_frame_noise(self) -> None:
        config = _test_config(
            velocity_filter_time_constant_seconds=0.0001,
            maximum_target_speed_pixels_per_second=10_000.0,
            maximum_target_acceleration_pixels_per_second_squared=1e9,
            feedback_deadzone_pixels=0.0,
            wrong_way_guard_pixels=0.0,
            velocity_median_window=5,
        )
        plant = CalibratedPlant(0.10, 0.10, 0.0)
        legacy = MakcuCalibratedController(plant, config)
        paired = MakcuCalibratedController(plant, config)
        circular_points = ((10.0, 0.0), (0.0, 10.0), (-10.0, 0.0), (0.0, -10.0))
        legacy_speeds: list[float] = []
        paired_speeds: list[float] = []

        for index in range(16):
            timestamp_ns = index * 8 * NS_PER_MS
            raw_x, raw_y = circular_points[index % len(circular_points)]
            legacy_output = legacy.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    raw_x,
                    raw_y,
                ),
            )
            paired_output = paired.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    0.0,
                    0.0,
                    velocity_error_x_pixels=raw_x,
                    velocity_error_y_pixels=raw_y,
                ),
            )
            if index >= 8:
                legacy_speeds.append(
                    math.hypot(
                        legacy_output.target_velocity_x_pixels_per_second,
                        legacy_output.target_velocity_y_pixels_per_second,
                    )
                )
                paired_speeds.append(
                    math.hypot(
                        paired_output.target_velocity_x_pixels_per_second,
                        paired_output.target_velocity_y_pixels_per_second,
                    )
                )

        # Independent adjacent-frame medians follow this closed orbit as a
        # persistent 1,768 px/s rotating vector. The covariance-aware paired
        # observer sees the bounded repeated displacement and attenuates it by
        # more than 80% without a five-frame OLS derivative.
        self.assertGreater(min(legacy_speeds), 1_760.0)
        self.assertLess(max(paired_speeds), 251.0)
        self.assertLess(
            sum(paired_speeds) / len(paired_speeds),
            (sum(legacy_speeds) / len(legacy_speeds)) * 0.20,
        )

    def test_paired_observer_suppresses_one_model_pixel_staircases(self) -> None:
        q = 1920.0 / 416.0
        plant = CalibratedPlant(0.125, 0.120, 0.008)
        config = _test_config(
            maximum_target_acceleration_pixels_per_second_squared=20_000.0,
            maximum_rate_x_counts_per_second=19_200.0,
            maximum_rate_y_counts_per_second=19_200.0,
            stale_after_seconds=0.065,
        )

        def run(
            points: tuple[tuple[float, float], ...],
            *,
            samples: int,
        ) -> list[CalibratedControlOutput]:
            controller = MakcuCalibratedController(plant, config)
            outputs: list[CalibratedControlOutput] = []
            for index in range(samples):
                timestamp_ns = round(index * NS_PER_SECOND / 126.0)
                raw_x, raw_y = points[index % len(points)]
                outputs.append(
                    controller.step(
                        timestamp_ns,
                        engaged=True,
                        observation=ScreenErrorObservation(
                            timestamp_ns,
                            0.0,
                            0.0,
                            velocity_error_x_pixels=raw_x,
                            velocity_error_y_pixels=raw_y,
                        ),
                    )
                )
            return outputs

        # Exact logged false-motion reproduction: the tracked position stays
        # centered while the raw box edge walks one 416-model-pixel quantum on
        # every accepted 126 Hz sample. The old paired OLS reported 581.39 px/s
        # and requested about 5,271 counts/s indefinitely.
        monotonic = MakcuCalibratedController(plant, config)
        monotonic_outputs: list[CalibratedControlOutput] = []
        for index in range(192):
            timestamp_ns = round(index * NS_PER_SECOND / 126.0)
            monotonic_outputs.append(
                monotonic.step(
                    timestamp_ns,
                    engaged=True,
                    observation=ScreenErrorObservation(
                        timestamp_ns,
                        0.0,
                        0.0,
                        velocity_error_x_pixels=index * q,
                        velocity_error_y_pixels=0.0,
                    ),
                )
            )
        monotonic_tail = monotonic_outputs[-64:]
        self.assertGreater(
            monotonic_tail[-1].target_velocity_x_pixels_per_second,
            575.0,
        )
        self.assertLess(
            monotonic_tail[-1].target_velocity_x_pixels_per_second,
            590.0,
        )
        self.assertGreater(
            monotonic_tail[-1].observer_velocity_sigma_x_pixels_per_second,
            100.0,
        )
        self.assertEqual(monotonic_tail[-1].velocity_feedforward_confidence_x, 0.0)
        self.assertLess(
            max(abs(output.rate_x_counts_per_second) for output in monotonic_tail),
            1.0,
        )

        # Also bind the smaller four-sample q-by-q loop. It never earns
        # velocity feed-forward confidence and remains under ten percent of
        # the measured 4,412 counts/s rotating OLS failure.
        tiny_square = ((0.0, 0.0), (q, 0.0), (q, q), (0.0, q))
        tiny_outputs = run(tiny_square, samples=400)[-200:]
        tiny_rms_rate = math.sqrt(
            sum(
                output.rate_x_counts_per_second**2
                + output.rate_y_counts_per_second**2
                for output in tiny_outputs
            )
            / len(tiny_outputs)
        )
        self.assertLess(tiny_rms_rate, 4_412.0 * 0.10)
        self.assertEqual(
            max(output.velocity_feedforward_confidence_x for output in tiny_outputs),
            0.0,
        )
        self.assertEqual(
            max(output.velocity_feedforward_confidence_y for output in tiny_outputs),
            0.0,
        )

        # One-q axis steps around an eight-model-pixel square were the full
        # live orbit reproduction. Radial raw/tracker disagreement prevents an
        # alternating quiet-looking axis from reopening that loop.
        perimeter: list[tuple[float, float]] = []
        perimeter.extend((index * q, -4.0 * q) for index in range(-4, 5))
        perimeter.extend((4.0 * q, index * q) for index in range(-3, 5))
        perimeter.extend((index * q, 4.0 * q) for index in range(3, -5, -1))
        perimeter.extend((-4.0 * q, index * q) for index in range(3, -4, -1))
        perimeter_outputs = run(tuple(perimeter), samples=800)[-400:]
        self.assertLess(
            max(
                math.hypot(
                    output.rate_x_counts_per_second,
                    output.rate_y_counts_per_second,
                )
                for output in perimeter_outputs
            ),
            1.0,
        )

    def test_same_channel_quantization_is_bounded_to_position_feedback(self) -> None:
        """Bind the direct-head limitation when no independent anchor exists."""

        q = 1920.0 / 416.0
        config = _test_config(
            position_time_constant_seconds=0.022,
            maximum_target_acceleration_pixels_per_second_squared=20_000.0,
            maximum_rate_x_counts_per_second=19_200.0,
            maximum_rate_y_counts_per_second=19_200.0,
            stale_after_seconds=0.065,
            feedback_deadzone_pixels=3.0,
            maximum_velocity_feedforward_fraction=0.0,
        )
        plant = CalibratedPlant(0.125, 0.120, 0.008)

        def run(
            points: tuple[tuple[float, float], ...],
            samples: int,
        ) -> list[CalibratedControlOutput]:
            controller = MakcuCalibratedController(plant, config)
            outputs: list[CalibratedControlOutput] = []
            for index in range(samples):
                timestamp_ns = round(index * NS_PER_SECOND / 126.0)
                point_x, point_y = points[index % len(points)]
                outputs.append(
                    controller.step(
                        timestamp_ns,
                        engaged=True,
                        observation=ScreenErrorObservation(
                            timestamp_ns,
                            point_x,
                            point_y,
                            velocity_error_x_pixels=point_x,
                            velocity_error_y_pixels=point_y,
                        ),
                    )
                )
            return outputs

        # With identical channels this sample stream is observationally equal
        # to a real 581 px/s target. FF=0 removes the false 4,651 counts/s
        # derivative term, but no numeric observer can also reject its growing
        # position error without external evidence that the point is false.
        monotonic_points = tuple((index * q, 0.0) for index in range(192))
        monotonic = run(monotonic_points, len(monotonic_points))
        monotonic_tail = monotonic[-64:]
        self.assertEqual(
            max(
                output.velocity_feedforward_confidence_x
                for output in monotonic_tail
            ),
            0.0,
        )
        self.assertEqual(
            min(output.rate_x_counts_per_second for output in monotonic_tail),
            19_200.0,
        )
        self.assertEqual(
            max(output.rate_x_counts_per_second for output in monotonic_tail),
            19_200.0,
        )

        perimeter: list[tuple[float, float]] = []
        perimeter.extend((index * q, -4.0 * q) for index in range(-4, 5))
        perimeter.extend((4.0 * q, index * q) for index in range(-3, 5))
        perimeter.extend((index * q, 4.0 * q) for index in range(3, -5, -1))
        perimeter.extend((-4.0 * q, index * q) for index in range(3, -4, -1))
        perimeter_tail = run(tuple(perimeter), 800)[-400:]
        perimeter_rms_rate = math.sqrt(
            sum(
                output.rate_x_counts_per_second**2
                + output.rate_y_counts_per_second**2
                for output in perimeter_tail
            )
            / len(perimeter_tail)
        )
        self.assertEqual(
            max(
                output.velocity_feedforward_confidence_x
                for output in perimeter_tail
            ),
            0.0,
        )
        self.assertEqual(
            max(
                output.velocity_feedforward_confidence_y
                for output in perimeter_tail
            ),
            0.0,
        )
        # This upper bound protects the configured feedback-only envelope. It
        # deliberately does not claim the same-channel orbit is distinguishable.
        self.assertLess(perimeter_rms_rate, 9_700.0)
        self.assertLess(
            max(
                math.hypot(
                    output.rate_x_counts_per_second,
                    output.rate_y_counts_per_second,
                )
                for output in perimeter_tail
            ),
            11_100.0,
        )

    def test_paired_observer_tracks_variable_dt_motion_and_reversal(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                maximum_target_acceleration_pixels_per_second_squared=20_000.0,
                maximum_rate_x_counts_per_second=19_200.0,
                maximum_rate_y_counts_per_second=19_200.0,
                stale_after_seconds=0.065,
            ),
        )
        intervals_ms = (7, 9, 8, 8, 7, 9)
        timestamp_ns = 0
        position_x = 0.0
        outputs: list[CalibratedControlOutput] = []
        for index in range(240):
            if index:
                interval_ms = intervals_ms[index % len(intervals_ms)]
                timestamp_ns += interval_ms * NS_PER_MS
                velocity = 575.0 if index < 120 else -575.0
                position_x += velocity * interval_ms / 1000.0
            outputs.append(
                controller.step(
                    timestamp_ns,
                    engaged=True,
                    observation=ScreenErrorObservation(
                        timestamp_ns,
                        position_x,
                        0.0,
                        velocity_error_x_pixels=position_x,
                        velocity_error_y_pixels=0.0,
                    ),
                )
            )

        self.assertLess(
            max(
                abs(output.target_velocity_x_pixels_per_second - 575.0)
                for output in outputs[80:120]
            ),
            0.1,
        )
        first_negative = next(
            index
            for index, output in enumerate(outputs[120:])
            if output.target_velocity_x_pixels_per_second < 0.0
        )
        self.assertLessEqual(first_negative, 3)
        self.assertLess(
            max(
                abs(output.target_velocity_x_pixels_per_second + 575.0)
                for output in outputs[-40:]
            ),
            0.1,
        )
        self.assertGreater(outputs[-1].velocity_feedforward_confidence_x, 0.99)
        self.assertLess(
            outputs[-1].observer_velocity_sigma_x_pixels_per_second,
            200.0,
        )

    def test_paired_observer_gates_outlier_and_reacquires_fail_closed(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.0),
            _test_config(stale_after_seconds=0.065),
        )

        def observe(index: int, position: float) -> CalibratedControlOutput:
            timestamp_ns = index * 8 * NS_PER_MS
            return controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    position,
                    0.0,
                    velocity_error_x_pixels=position,
                    velocity_error_y_pixels=0.0,
                ),
            )

        observe(0, 0.0)
        observe(1, 0.0)
        accepted = observe(2, 0.0)
        rejected = observe(3, 1_000.0)
        self.assertTrue(rejected.valid)
        self.assertTrue(rejected.innovation_rejected)
        self.assertGreater(rejected.innovation_mahalanobis_squared, 16.0)
        self.assertAlmostEqual(rejected.projected_error_x_pixels, 0.0, delta=0.01)
        bridged = controller.step(25 * NS_PER_MS, engaged=True)
        self.assertTrue(bridged.innovation_rejected)
        self.assertEqual(
            bridged.innovation_mahalanobis_squared,
            rejected.innovation_mahalanobis_squared,
        )
        recovered = observe(4, 0.0)
        self.assertTrue(recovered.valid)
        self.assertFalse(recovered.innovation_rejected)
        self.assertAlmostEqual(
            recovered.target_velocity_x_pixels_per_second,
            accepted.target_velocity_x_pixels_per_second,
            delta=0.01,
        )

        observe(5, 1_000.0)
        observe(6, 1_000.0)
        reacquired = observe(7, 1_000.0)
        self.assertFalse(reacquired.valid)
        self.assertEqual(reacquired.reset_reason, "innovation-reacquired")
        self.assertFalse(controller.ready)

    def test_paired_release_and_loss_clear_observer_and_require_confirmation(
        self,
    ) -> None:
        for action, reason in (
            ({"engaged": False}, "released"),
            ({"engaged": True, "target_lost": True}, "target-lost"),
        ):
            with self.subTest(reason=reason):
                controller = MakcuCalibratedController(
                    CalibratedPlant(0.10, 0.10, 0.0),
                    _test_config(stale_after_seconds=0.065),
                )
                for index in range(2):
                    timestamp_ns = index * 8 * NS_PER_MS
                    controller.step(
                        timestamp_ns,
                        engaged=True,
                        observation=ScreenErrorObservation(
                            timestamp_ns,
                            10.0,
                            0.0,
                            velocity_error_x_pixels=10.0,
                            velocity_error_y_pixels=0.0,
                        ),
                    )
                stopped = controller.step(16 * NS_PER_MS, **action)
                self.assertFalse(stopped.valid)
                self.assertEqual(stopped.reset_reason, reason)
                self.assertEqual(
                    stopped.observer_velocity_sigma_x_pixels_per_second,
                    0.0,
                )
                first = controller.step(
                    24 * NS_PER_MS,
                    engaged=True,
                    observation=ScreenErrorObservation(
                        24 * NS_PER_MS,
                        10.0,
                        0.0,
                        velocity_error_x_pixels=10.0,
                        velocity_error_y_pixels=0.0,
                    ),
                )
                self.assertFalse(first.valid)
                self.assertEqual(first.reset_reason, "awaiting-confirmation")
                second = controller.step(
                    32 * NS_PER_MS,
                    engaged=True,
                    observation=ScreenErrorObservation(
                        32 * NS_PER_MS,
                        10.0,
                        0.0,
                        velocity_error_x_pixels=10.0,
                        velocity_error_y_pixels=0.0,
                    ),
                )
                self.assertTrue(second.valid)

    def test_velocity_feedforward_cap_does_not_change_legacy_callers(self) -> None:
        controllers = (
            MakcuCalibratedController(
                CalibratedPlant(0.10, 0.20, 0.012),
                _test_config(
                    maximum_velocity_feedforward_fraction=fraction,
                    velocity_filter_time_constant_seconds=0.0001,
                    maximum_target_acceleration_pixels_per_second_squared=1e9,
                    velocity_median_window=1,
                ),
            )
            for fraction in (0.0, 1.0)
        )
        outputs: list[list[CalibratedControlOutput]] = []
        for controller in controllers:
            outputs.append(
                [
                    controller.step(
                        100 * NS_PER_MS,
                        engaged=True,
                        observation=ScreenErrorObservation(
                            100 * NS_PER_MS,
                            10.0,
                            -4.0,
                        ),
                    ),
                    controller.step(
                        120 * NS_PER_MS,
                        engaged=True,
                        observation=ScreenErrorObservation(
                            120 * NS_PER_MS,
                            16.0,
                            -8.0,
                        ),
                        emitted_commands=(
                            EmittedMouseCommand(105 * NS_PER_MS, 20, -10),
                        ),
                    ),
                ]
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_successful_recorded_write_reduces_projected_error_exactly_once(self) -> None:
        def established_controller() -> tuple[
            MakcuCalibratedController,
            CalibratedControlOutput,
        ]:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.10, 0.20, 0.020),
                _test_config(velocity_median_window=1),
            )
            controller.step(
                0,
                engaged=True,
                observation=ScreenErrorObservation(0, 100.0, 0.0),
            )
            ready = controller.step(
                10 * NS_PER_MS,
                engaged=True,
                observation=ScreenErrorObservation(10 * NS_PER_MS, 100.0, 0.0),
            )
            return controller, ready

        controller, ready = established_controller()
        self.assertAlmostEqual(ready.projected_error_x_pixels, 100.0)
        controller.step(11 * NS_PER_MS, engaged=True)
        controller.record_emitted(
            EmittedMouseCommand(11 * NS_PER_MS, 100, 0)
        )
        accounted = controller.step(15 * NS_PER_MS, engaged=True)
        self.assertTrue(accounted.valid)
        self.assertAlmostEqual(accounted.projected_error_x_pixels, 90.0)
        self.assertLess(
            accounted.rate_x_counts_per_second,
            ready.rate_x_counts_per_second,
        )

        # A failed write or a tick that rounds to zero is represented by no
        # record_emitted call, so it cannot be mistaken for physical motion.
        absent, absent_ready = established_controller()
        absent.step(11 * NS_PER_MS, engaged=True)
        unchanged = absent.step(15 * NS_PER_MS, engaged=True)
        self.assertAlmostEqual(unchanged.projected_error_x_pixels, 100.0)
        self.assertAlmostEqual(
            unchanged.rate_x_counts_per_second,
            absent_ready.rate_x_counts_per_second,
        )

    def test_record_emitted_rejects_duplicate_nonmonotonic_and_future_events(self) -> None:
        def controller_at(now_ms: int) -> MakcuCalibratedController:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.10, 0.20, 0.020),
                _test_config(velocity_median_window=1),
            )
            controller.step(
                0,
                engaged=True,
                observation=ScreenErrorObservation(0, 20.0, 0.0),
            )
            controller.step(
                now_ms * NS_PER_MS,
                engaged=True,
                observation=ScreenErrorObservation(
                    now_ms * NS_PER_MS,
                    20.0,
                    0.0,
                ),
            )
            return controller

        duplicate = controller_at(10)
        command = EmittedMouseCommand(10 * NS_PER_MS, 4, -2)
        duplicate.record_emitted(command)
        with self.assertRaisesRegex(ValueError, "non-monotonic-command-history"):
            duplicate.record_emitted(command)
        self.assertFalse(duplicate.ready)

        nonmonotonic = controller_at(10)
        nonmonotonic.record_emitted(EmittedMouseCommand(10 * NS_PER_MS, 4, -2))
        nonmonotonic.step(11 * NS_PER_MS, engaged=True)
        with self.assertRaisesRegex(ValueError, "non-monotonic-command-history"):
            nonmonotonic.record_emitted(
                EmittedMouseCommand(9 * NS_PER_MS, 1, 0)
            )
        self.assertFalse(nonmonotonic.ready)

        future = controller_at(10)
        with self.assertRaisesRegex(ValueError, "future-command"):
            future.record_emitted(EmittedMouseCommand(11 * NS_PER_MS, 1, 0))
        self.assertFalse(future.ready)

        before_step = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.20, 0.020),
            _test_config(velocity_median_window=1),
        )
        with self.assertRaisesRegex(RuntimeError, "before a control step"):
            before_step.record_emitted(EmittedMouseCommand(0, 1, 0))

    def test_duplicate_control_timestamp_is_invalid_before_command_decision(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.20, 0.020),
            _test_config(velocity_median_window=1),
        )
        controller.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(0, 20.0, 0.0),
        )
        ready = controller.step(
            10 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(10 * NS_PER_MS, 20.0, 0.0),
        )
        self.assertTrue(ready.valid)

        duplicate = controller.step(10 * NS_PER_MS, engaged=True)

        self.assertFalse(duplicate.valid)
        self.assertEqual(duplicate.reset_reason, "non-monotonic-clock")
        self.assertEqual(duplicate.rate_x_counts_per_second, 0.0)
        self.assertFalse(controller.ready)

    def test_zero_current_error_retains_target_velocity_feed_forward(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.08, 0.15, 0.024),
            _test_config(
                velocity_filter_time_constant_seconds=0.0001,
                maximum_target_acceleration_pixels_per_second_squared=1e9,
                velocity_median_window=1,
            ),
        )
        controller.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(0, -8.0, 0.0),
        )
        output = controller.step(
            10 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(10 * NS_PER_MS, 0.0, 0.0),
        )
        self.assertTrue(output.valid)
        self.assertGreater(output.target_velocity_x_pixels_per_second, 790.0)
        self.assertGreater(output.rate_x_counts_per_second, 0.0)

    def test_release_loss_stale_and_jump_fail_closed_and_require_reconfirmation(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.12, 0.024),
            _test_config(velocity_median_window=1),
        )

        def establish(base_ms: int, error: float = 30.0) -> None:
            controller.step(
                base_ms * NS_PER_MS,
                engaged=True,
                observation=ScreenErrorObservation(base_ms * NS_PER_MS, error, 0.0),
            )
            output = controller.step(
                (base_ms + 8) * NS_PER_MS,
                engaged=True,
                observation=ScreenErrorObservation(
                    (base_ms + 8) * NS_PER_MS,
                    error + 1.0,
                    0.0,
                ),
            )
            self.assertTrue(output.valid)

        establish(0)
        released = controller.step(9 * NS_PER_MS, engaged=False)
        self.assertEqual(released.rate_x_counts_per_second, 0.0)
        self.assertEqual(released.reset_reason, "released")
        self.assertFalse(
            controller.step(10 * NS_PER_MS, engaged=True).valid
        )

        establish(20)
        lost = controller.step(
            29 * NS_PER_MS,
            engaged=True,
            target_lost=True,
        )
        self.assertEqual(lost.reset_reason, "target-lost")
        self.assertEqual(lost.rate_x_counts_per_second, 0.0)

        establish(40)
        stale = controller.step(89 * NS_PER_MS, engaged=True)
        self.assertEqual(stale.reset_reason, "stale-observation")
        self.assertEqual(stale.rate_y_counts_per_second, 0.0)

        establish(100)
        jumped = controller.step(
            116 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(116 * NS_PER_MS, 500.0, 0.0),
        )
        self.assertEqual(jumped.reset_reason, "error-jump")
        self.assertFalse(controller.ready)
        held = controller.step(117 * NS_PER_MS, engaged=True)
        self.assertEqual(held.reset_reason, "awaiting-confirmation")
        self.assertEqual(held.rate_x_counts_per_second, 0.0)

    def test_prediction_or_missing_callback_does_not_invent_target_loss(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.12, 0.024),
            _test_config(velocity_median_window=1),
        )
        controller.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(0, 20.0, 0.0),
        )
        established = controller.step(
            8 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(8 * NS_PER_MS, 22.0, 0.0),
        )
        self.assertTrue(established.valid)
        for gap_ms in (16, 24):
            with self.subTest(gap_ms=gap_ms):
                # A tracker prediction has no real observation, but is not an
                # explicit detector/tracker revocation either.
                output = controller.step(
                    gap_ms * NS_PER_MS,
                    engaged=True,
                    target_lost=False,
                )
                self.assertTrue(output.valid)
                self.assertIsNone(output.reset_reason)

    def test_legacy_observation_expected_alias_means_explicit_target_loss(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.12, 0.024),
            _test_config(velocity_median_window=1),
        )
        controller.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(0, 20.0, 0.0),
        )
        lost = controller.step(
            8 * NS_PER_MS,
            engaged=True,
            observation_expected=True,
        )
        self.assertEqual(lost.reset_reason, "target-lost")
        self.assertFalse(controller.ready)

    def test_release_preserves_pending_physical_command_correction(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.20, 0.020),
            _test_config(
                velocity_filter_time_constant_seconds=0.0001,
                maximum_target_acceleration_pixels_per_second_squared=1e9,
                velocity_median_window=1,
            ),
        )
        controller.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(0, 0.0, 0.0),
        )
        controller.step(
            10 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(10 * NS_PER_MS, 0.0, 0.0),
        )
        controller.step(11 * NS_PER_MS, engaged=True)
        controller.record_emitted(
            EmittedMouseCommand(11 * NS_PER_MS, 100, 0)
        )

        released = controller.step(12 * NS_PER_MS, engaged=False)
        self.assertEqual(released.reset_reason, "released")
        controller.step(
            20 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(20 * NS_PER_MS, 0.0, 0.0),
        )
        after_impact = controller.step(
            35 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(35 * NS_PER_MS, -10.0, 0.0),
        )
        self.assertTrue(after_impact.valid)
        self.assertAlmostEqual(
            after_impact.target_velocity_x_pixels_per_second,
            0.0,
            delta=0.1,
        )

    def test_limits_and_wrong_way_guard_are_hard_invariants(self) -> None:
        config = _test_config(
            maximum_rate_x_counts_per_second=700.0,
            maximum_rate_y_counts_per_second=350.0,
            maximum_error_jump_pixels=1000.0,
            velocity_median_window=1,
        )
        controller = MakcuCalibratedController(
            CalibratedPlant(0.05, 0.21, 0.050),
            config,
        )
        controller.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(0, 0.0, 0.0),
        )
        output = controller.step(
            10 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(10 * NS_PER_MS, 100.0, -80.0),
        )
        self.assertTrue(output.valid)
        self.assertLessEqual(abs(output.rate_x_counts_per_second), 700.0)
        self.assertLessEqual(abs(output.rate_y_counts_per_second), 350.0)
        self.assertTrue(math.isfinite(output.rate_x_counts_per_second))
        self.assertTrue(math.isfinite(output.rate_y_counts_per_second))
        for rate, error in (
            (output.rate_x_counts_per_second, output.projected_error_x_pixels),
            (output.rate_y_counts_per_second, output.projected_error_y_pixels),
        ):
            if abs(error) > config.wrong_way_guard_pixels:
                self.assertGreaterEqual(rate * error, 0.0)

    def test_invalid_numeric_inputs_never_enter_control_state(self) -> None:
        for plant_args in ((0.0, 0.1, 0.02), (0.1, math.nan, 0.02), (0.1, 0.1, -0.1)):
            with self.subTest(plant_args=plant_args), self.assertRaises(ValueError):
                CalibratedPlant(*plant_args)
        with self.assertRaises(ValueError):
            ScreenErrorObservation(0, math.inf, 0.0)
        with self.assertRaises(ValueError):
            EmittedMouseCommand(0, 0, 0)

    def test_correlated_lookahead_retains_only_root_authority_same_timestamp(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                stale_after_seconds=0.110,
                maximum_observation_interval_seconds=0.040,
                feedback_deadzone_pixels=4.5,
                continuous_feedback_deadband=True,
                continuous_feedback_shoulder_pixels=6.0,
                pursuit_position_time_constant_seconds=0.016,
                pursuit_position_time_constant_start_pixels=10.5,
                pursuit_position_time_constant_full_pixels=22.0,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
            ),
        )
        base_ns = 10 * NS_PER_SECOND
        identity_deadline_ns = base_ns + 2 * NS_PER_SECOND
        output = None
        for index in range(12):
            root_ns = base_ns + index * 10 * NS_PER_MS
            root_x = index * 6.0
            root = ScreenErrorObservation(
                root_ns,
                root_x,
                0.0,
                velocity_error_x_pixels=root_x,
                velocity_error_y_pixels=0.0,
                corroboration_error_x_pixels=root_x + 100.0,
                corroboration_error_y_pixels=100.0,
                identity_deadline_ns=identity_deadline_ns,
            )
            # This endpoint becomes the next iteration's equal-timestamp root.
            lookahead = ScreenErrorObservation(
                root_ns + 10 * NS_PER_MS,
                root_x + 6.0,
                0.0,
                velocity_error_x_pixels=root_x + 6.0,
                velocity_error_y_pixels=0.0,
                identity_deadline_ns=identity_deadline_ns,
            )
            output = controller.step(
                root_ns + 11 * NS_PER_MS,
                engaged=True,
                correlated_lookahead=CorrelatedLookaheadObservation(
                    root,
                    lookahead,
                    runtime_identity_generation=7,
                    track_generation=3,
                ),
            )
            self.assertEqual(
                controller._last_corroboration_measurement_ns,
                root_ns,
            )

        assert output is not None
        self.assertTrue(output.valid)
        self.assertTrue(output.correlated_lookahead_active)
        self.assertGreater(output.velocity_feedforward_confidence_x, 0.40)
        self.assertLessEqual(output.velocity_feedforward_confidence_x, 0.50)
        self.assertLessEqual(output.lookahead_retained_authority_x, 0.50)
        # A stationary root Y channel cannot authorize a newer Y phase step.
        self.assertEqual(output.velocity_feedforward_confidence_y, 0.0)
        self.assertEqual(output.lookahead_retained_authority_y, 0.0)

    def test_correlated_fast_retention_is_root_earned_and_speed_bounded(
        self,
    ) -> None:
        def run(
            speed_pixels_per_second: float,
            *,
            verified_flow_motion: bool = False,
        ) -> list[CalibratedControlOutput]:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.008),
                _test_config(
                    position_time_constant_seconds=0.028,
                    velocity_filter_time_constant_seconds=0.014,
                    maximum_target_acceleration_pixels_per_second_squared=(
                        40_000.0
                    ),
                    stale_after_seconds=0.110,
                    maximum_observation_interval_seconds=0.040,
                    feedback_deadzone_pixels=4.5,
                    continuous_feedback_deadband=True,
                    continuous_feedback_shoulder_pixels=6.0,
                    pursuit_position_time_constant_seconds=0.016,
                    pursuit_position_time_constant_start_pixels=10.5,
                    pursuit_position_time_constant_full_pixels=22.0,
                    maximum_velocity_feedforward_fraction=1.0,
                    require_motion_corroboration_for_feedforward=True,
                    maximum_body_derived_projection_fraction=1.0,
                    maximum_body_derived_feedforward_fraction=0.50,
                    maximum_body_derived_pursuit_feedforward_fraction=0.82,
                    maximum_correlated_lookahead_pursuit_feedforward_fraction=(
                        0.60
                    ),
                    maximum_verified_flow_pursuit_feedforward_fraction=0.95,
                ),
            )
            base_ns = 12 * NS_PER_SECOND
            identity_deadline_ns = base_ns + 2 * NS_PER_SECOND
            step_pixels = speed_pixels_per_second * 0.010
            phase_pixels = speed_pixels_per_second * 0.008
            outputs = []
            for index in range(40):
                root_ns = base_ns + index * 10 * NS_PER_MS
                root_x = 100.0 + index * step_pixels
                outputs.append(
                    controller.step(
                        root_ns + 9 * NS_PER_MS,
                        engaged=True,
                        correlated_lookahead=CorrelatedLookaheadObservation(
                            ScreenErrorObservation(
                                root_ns,
                                root_x,
                                0.0,
                                velocity_error_x_pixels=root_x,
                                velocity_error_y_pixels=0.0,
                                corroboration_error_x_pixels=root_x + 100.0,
                                corroboration_error_y_pixels=100.0,
                                identity_deadline_ns=identity_deadline_ns,
                            ),
                            ScreenErrorObservation(
                                root_ns + 8 * NS_PER_MS,
                                root_x + phase_pixels,
                                0.0,
                                velocity_error_x_pixels=(
                                    root_x + phase_pixels
                                ),
                                velocity_error_y_pixels=0.0,
                                identity_deadline_ns=identity_deadline_ns,
                            ),
                            1,
                            1,
                            verified_flow_motion=verified_flow_motion,
                        ),
                    )
                )
            return outputs

        below_ramp = run(200.0)
        midpoint = run(375.0)
        full = run(600.0)
        unverified_fast = run(1_500.0)
        verified_ordinary = run(600.0, verified_flow_motion=True)
        verified_fast = run(1_500.0, verified_flow_motion=True)

        self.assertLessEqual(
            below_ramp[-1].lookahead_retained_authority_x,
            0.50,
        )
        self.assertAlmostEqual(
            midpoint[-1].lookahead_retained_authority_x,
            0.55,
            places=6,
        )
        self.assertLessEqual(
            max(output.lookahead_retained_authority_x for output in full[:27]),
            0.50,
        )
        self.assertAlmostEqual(
            full[27].lookahead_retained_authority_x,
            0.60,
            places=6,
        )
        self.assertAlmostEqual(
            full[27].velocity_feedforward_confidence_x,
            0.60,
            places=6,
        )
        self.assertLessEqual(
            max(output.lookahead_retained_authority_x for output in full),
            0.60,
        )
        self.assertLessEqual(
            max(
                output.lookahead_retained_authority_x
                for output in unverified_fast
            ),
            0.60,
        )
        self.assertLessEqual(
            max(
                output.lookahead_retained_authority_x
                for output in verified_ordinary
            ),
            0.60,
        )
        self.assertAlmostEqual(
            verified_fast[-1].lookahead_retained_authority_x,
            0.95,
            places=6,
        )
        for output in (
            below_ramp
            + midpoint
            + full
            + unverified_fast
            + verified_ordinary
            + verified_fast
        ):
            self.assertEqual(output.lookahead_retained_authority_y, 0.0)
            self.assertEqual(output.velocity_feedforward_confidence_y, 0.0)

    def test_verified_flow_promotes_first_qualified_fast_batch(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                stale_after_seconds=0.110,
                maximum_observation_interval_seconds=0.040,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.82,
                maximum_correlated_lookahead_pursuit_feedforward_fraction=0.60,
                maximum_verified_flow_pursuit_feedforward_fraction=0.95,
            ),
        )
        base_ns = 14 * NS_PER_SECOND
        deadline_ns = base_ns + NS_PER_SECOND
        outputs: list[CalibratedControlOutput] = []
        for index in range(4):
            root_ns = base_ns + index * 10 * NS_PER_MS
            root_x = 100.0 + index * 15.0
            outputs.append(
                controller.step(
                    root_ns + 9 * NS_PER_MS,
                    engaged=True,
                    correlated_lookahead=CorrelatedLookaheadObservation(
                        ScreenErrorObservation(
                            root_ns,
                            root_x,
                            0.0,
                            velocity_error_x_pixels=root_x,
                            velocity_error_y_pixels=0.0,
                            corroboration_error_x_pixels=root_x + 100.0,
                            corroboration_error_y_pixels=100.0,
                            identity_deadline_ns=deadline_ns,
                        ),
                        ScreenErrorObservation(
                            root_ns + 8 * NS_PER_MS,
                            root_x + 12.0,
                            0.0,
                            velocity_error_x_pixels=root_x + 12.0,
                            velocity_error_y_pixels=0.0,
                            identity_deadline_ns=deadline_ns,
                        ),
                        1,
                        1,
                        # The runtime supplies this only after its own
                        # three-frame verified-flow streak.
                        verified_flow_motion=index == 3,
                    ),
                )
            )

        qualified = outputs[-1]
        self.assertLess(qualified.motion_corroboration_confidence, 0.05)
        self.assertLessEqual(
            max(output.lookahead_retained_authority_x for output in outputs[:-1]),
            0.60,
        )
        self.assertAlmostEqual(
            qualified.lookahead_retained_authority_x,
            0.95,
            places=6,
        )
        self.assertAlmostEqual(
            qualified.velocity_feedforward_confidence_x,
            0.95,
            places=6,
        )

        controller.record_physical_input(
            base_ns + 40 * NS_PER_MS,
            10,
            0,
        )
        manual_pending = controller.step(
            base_ns + 41 * NS_PER_MS,
            engaged=True,
        )
        self.assertTrue(manual_pending.physical_input_pending_x)
        self.assertTrue(manual_pending.predictive_authority_revoked_x)
        self.assertEqual(manual_pending.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(manual_pending.rate_x_counts_per_second, 0.0)

    def test_verified_flow_prompt_is_per_axis_and_residual_gated(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.82,
                maximum_correlated_lookahead_pursuit_feedforward_fraction=0.60,
                maximum_verified_flow_pursuit_feedforward_fraction=0.95,
            ),
        )
        controller._last_observation = ScreenErrorObservation(  # noqa: SLF001
            10 * NS_PER_MS,
            20.0,
            -20.0,
            velocity_error_x_pixels=20.0,
            velocity_error_y_pixels=-20.0,
        )
        controller._paired_position_x = 20.0  # noqa: SLF001
        controller._paired_position_y = -20.0  # noqa: SLF001
        controller._velocity_x = 1_500.0  # noqa: SLF001
        controller._velocity_y = -1_000.0  # noqa: SLF001
        controller._paired_fresh_motion_confidence_x = 1.0  # noqa: SLF001
        controller._paired_fresh_motion_confidence_y = 1.0  # noqa: SLF001
        controller._independent_pursuit_authorized = True  # noqa: SLF001
        controller._motion_corroboration_confidence = 0.02  # noqa: SLF001

        promoted = controller._correlated_lookahead_authority(  # noqa: SLF001
            prior_authority=None,
            same_identity=False,
            observation_elapsed=0.010,
            verified_flow_motion=True,
            verified_flow_fresh_motion_x=1.0,
            verified_flow_fresh_motion_y=1.0,
        )
        self.assertAlmostEqual(promoted.total_x, 0.95, places=6)
        self.assertAlmostEqual(promoted.total_y, 0.95, places=6)

        controller._paired_material_stop_or_reversal_x = True  # noqa: SLF001
        stopped_x = controller._correlated_lookahead_authority(  # noqa: SLF001
            prior_authority=promoted,
            same_identity=True,
            observation_elapsed=0.010,
            verified_flow_motion=True,
            verified_flow_fresh_motion_x=1.0,
            verified_flow_fresh_motion_y=1.0,
        )
        self.assertEqual(stopped_x.total_x, 0.0)
        self.assertAlmostEqual(stopped_x.total_y, 0.95, places=6)

        controller._paired_material_stop_or_reversal_x = False  # noqa: SLF001
        no_fresh_x = controller._correlated_lookahead_authority(  # noqa: SLF001
            prior_authority=promoted,
            same_identity=True,
            observation_elapsed=0.010,
            verified_flow_motion=True,
            verified_flow_fresh_motion_x=0.0,
            verified_flow_fresh_motion_y=1.0,
        )
        self.assertLessEqual(no_fresh_x.total_x, 0.60)
        self.assertAlmostEqual(no_fresh_x.total_y, 0.95, places=6)

        controller._last_observation = replace(  # noqa: SLF001
            controller._last_observation,  # type: ignore[arg-type]
            error_x_pixels=-9.0,
        )
        controller._paired_position_x = -9.0  # noqa: SLF001
        opposed_residual = controller._correlated_lookahead_authority(  # noqa: SLF001
            prior_authority=promoted,
            same_identity=True,
            observation_elapsed=0.010,
            verified_flow_motion=True,
            verified_flow_fresh_motion_x=1.0,
            verified_flow_fresh_motion_y=1.0,
        )
        self.assertLessEqual(opposed_residual.total_x, 0.60)
        self.assertAlmostEqual(opposed_residual.total_y, 0.95, places=6)

        # The stronger grant belongs only to the currently verified endpoint.
        # Removing that evidence must return the shared adaptive state to the
        # generic body ceiling rather than leaking 95% into body fallback.
        unverified = controller._correlated_lookahead_authority(  # noqa: SLF001
            prior_authority=promoted,
            same_identity=True,
            observation_elapsed=0.010,
            verified_flow_motion=False,
            verified_flow_fresh_motion_x=1.0,
            verified_flow_fresh_motion_y=1.0,
        )
        self.assertLessEqual(unverified.total_x, 0.60)
        self.assertLessEqual(unverified.total_y, 0.60)
        body_fallback = controller._adaptive_body_pursuit_confidence(  # noqa: SLF001
            0.95,
            measured_error=20.0,
            projected_error=20.0,
            projected_velocity=1_500.0,
            projected_vector_speed=1_500.0,
            motion_confidence=1.0,
            fresh_motion_confidence=1.0,
            direction_persistence_seconds=0.060,
            elapsed=0.010,
            binary_promotion=True,
        )
        self.assertLessEqual(body_fallback, 0.82)

    def test_zero_verified_flow_maximum_is_bit_identical(self) -> None:
        def run(verified_flow_motion: bool) -> list[CalibratedControlOutput]:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.008),
                _test_config(
                    maximum_velocity_feedforward_fraction=1.0,
                    require_motion_corroboration_for_feedforward=True,
                    maximum_body_derived_projection_fraction=1.0,
                    maximum_body_derived_feedforward_fraction=0.50,
                    maximum_body_derived_pursuit_feedforward_fraction=0.90,
                    maximum_correlated_lookahead_pursuit_feedforward_fraction=0.60,
                    maximum_verified_flow_pursuit_feedforward_fraction=0.0,
                ),
            )
            base_ns = 16 * NS_PER_SECOND
            deadline_ns = base_ns + NS_PER_SECOND
            outputs = []
            for index in range(10):
                root_ns = base_ns + index * 10 * NS_PER_MS
                root_x = 50.0 + index * 15.0
                outputs.append(
                    controller.step(
                        root_ns + 9 * NS_PER_MS,
                        engaged=True,
                        correlated_lookahead=CorrelatedLookaheadObservation(
                            ScreenErrorObservation(
                                root_ns,
                                root_x,
                                0.0,
                                velocity_error_x_pixels=root_x,
                                velocity_error_y_pixels=0.0,
                                corroboration_error_x_pixels=root_x + 100.0,
                                corroboration_error_y_pixels=100.0,
                                identity_deadline_ns=deadline_ns,
                            ),
                            ScreenErrorObservation(
                                root_ns + 8 * NS_PER_MS,
                                root_x + 12.0,
                                0.0,
                                velocity_error_x_pixels=root_x + 12.0,
                                velocity_error_y_pixels=0.0,
                                identity_deadline_ns=deadline_ns,
                            ),
                            1,
                            1,
                            verified_flow_motion=verified_flow_motion,
                        ),
                    )
                )
            return outputs

        self.assertEqual(run(False), run(True))

    def test_verified_flow_extra_requires_current_endpoint_motion(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(),
        )
        controller._velocity_x = 1_500.0  # noqa: SLF001
        controller._velocity_y = -750.0  # noqa: SLF001
        primary = ScreenErrorObservation(
            0,
            0.0,
            0.0,
            velocity_error_x_pixels=0.0,
            velocity_error_y_pixels=0.0,
        )

        def supported(endpoint_x: float, endpoint_y: float) -> bool:
            return controller._verified_flow_endpoint_supports_velocity(  # noqa: SLF001
                primary,
                ScreenErrorObservation(
                    8 * NS_PER_MS,
                    endpoint_x,
                    endpoint_y,
                    velocity_error_x_pixels=endpoint_x,
                    velocity_error_y_pixels=endpoint_y,
                ),
                endpoint_count_x=0,
                endpoint_count_y=0,
            )

        self.assertTrue(supported(12.0, -6.0))
        self.assertFalse(supported(0.0, 0.0))
        self.assertFalse(supported(-12.0, 6.0))

    def test_ambiguous_correlated_sample_retains_projection_only_once(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                position_time_constant_seconds=0.028,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                stale_after_seconds=0.110,
                maximum_observation_interval_seconds=0.040,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
                maximum_correlated_lookahead_pursuit_feedforward_fraction=0.60,
                retain_ambiguous_correlated_projection=True,
            ),
        )
        base_ns = 18 * NS_PER_SECOND
        deadline_ns = base_ns + 2 * NS_PER_SECOND
        for index in range(20):
            root_ns = base_ns + index * 10 * NS_PER_MS
            root_x = index * 6.0
            output = controller.step(
                root_ns + 9 * NS_PER_MS,
                engaged=True,
                correlated_lookahead=CorrelatedLookaheadObservation(
                    ScreenErrorObservation(
                        root_ns,
                        root_x,
                        0.0,
                        velocity_error_x_pixels=root_x,
                        velocity_error_y_pixels=0.0,
                        corroboration_error_x_pixels=root_x + 100.0,
                        corroboration_error_y_pixels=100.0,
                        identity_deadline_ns=deadline_ns,
                    ),
                    ScreenErrorObservation(
                        root_ns + 8 * NS_PER_MS,
                        root_x + 4.8,
                        0.0,
                        velocity_error_x_pixels=root_x + 4.8,
                        velocity_error_y_pixels=0.0,
                        identity_deadline_ns=deadline_ns,
                    ),
                    1,
                    1,
                ),
            )
        self.assertTrue(output.correlated_lookahead_active)
        prior = controller._correlated_lookahead_authority_state  # noqa: SLF001
        assert prior is not None
        self.assertGreater(prior.projection_x, 0.0)
        self.assertGreater(prior.total_x, 0.0)

        controller._independent_pursuit_authorized = True  # noqa: SLF001
        controller._paired_fresh_motion_confidence_x = 0.0  # noqa: SLF001
        controller._paired_material_stop_or_reversal_x = False  # noqa: SLF001
        controller._adaptive_pursuit_low_fresh_samples_x = 1  # noqa: SLF001
        retained = controller._correlated_lookahead_authority(  # noqa: SLF001
            prior_authority=prior,
            same_identity=True,
            observation_elapsed=10 * NS_PER_MS / NS_PER_SECOND,
        )
        self.assertTrue(retained.ambiguous_projection_retained_x)
        self.assertEqual(retained.projection_x, prior.projection_x)
        self.assertEqual(retained.ordinary_x, 0.0)
        self.assertEqual(retained.total_x, 0.0)

        controller._adaptive_pursuit_low_fresh_samples_x = 2  # noqa: SLF001
        repeated = controller._correlated_lookahead_authority(  # noqa: SLF001
            prior_authority=prior,
            same_identity=True,
            observation_elapsed=10 * NS_PER_MS / NS_PER_SECOND,
        )
        self.assertFalse(repeated.ambiguous_projection_retained_x)
        self.assertEqual(repeated.projection_x, 0.0)

        controller._adaptive_pursuit_low_fresh_samples_x = 1  # noqa: SLF001
        controller._velocity_x = -abs(controller._velocity_x)  # noqa: SLF001
        reversed_direction = controller._correlated_lookahead_authority(  # noqa: SLF001
            prior_authority=prior,
            same_identity=True,
            observation_elapsed=10 * NS_PER_MS / NS_PER_SECOND,
        )
        self.assertEqual(reversed_direction.projection_x, 0.0)

        controller._velocity_x = abs(controller._velocity_x)  # noqa: SLF001
        controller._paired_material_stop_or_reversal_x = True  # noqa: SLF001
        stopped = controller._correlated_lookahead_authority(  # noqa: SLF001
            prior_authority=prior,
            same_identity=True,
            observation_elapsed=10 * NS_PER_MS / NS_PER_SECOND,
        )
        self.assertEqual(stopped.projection_x, 0.0)
        self.assertFalse(stopped.ambiguous_projection_retained_x)

    def test_ambiguous_projection_direction_gate_uses_command_compensated_batch_motion(
        self,
    ) -> None:
        supports = (
            MakcuCalibratedController
            ._command_compensated_displacement_supports_direction
        )

        # Raw endpoint displacement is -6 px, but the 8 px landed command
        # reveals +2 px of continuing physical target motion.
        self.assertTrue(supports(10.0, 4.0, 8.0, 1))
        # With no compensating command, the same endpoint motion is a real
        # reversal and must revoke the frozen positive-direction projection.
        self.assertFalse(supports(10.0, 4.0, 0.0, 1))
        # A zero physical displacement is ambiguous rather than opposed.
        self.assertTrue(supports(10.0, 4.0, 6.0, 1))

    def test_correlated_lookahead_revokes_opposed_ambiguous_axis(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                stale_after_seconds=0.110,
                maximum_observation_interval_seconds=0.040,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
                maximum_correlated_lookahead_pursuit_feedforward_fraction=0.60,
                retain_ambiguous_correlated_projection=True,
            ),
        )
        base_ns = 24 * NS_PER_SECOND
        deadline_ns = base_ns + 2 * NS_PER_SECOND
        for index in range(20):
            root_ns = base_ns + index * 10 * NS_PER_MS
            root_x = index * 6.0
            controller.step(
                root_ns + 9 * NS_PER_MS,
                engaged=True,
                correlated_lookahead=CorrelatedLookaheadObservation(
                    ScreenErrorObservation(
                        root_ns,
                        root_x,
                        0.0,
                        velocity_error_x_pixels=root_x,
                        velocity_error_y_pixels=0.0,
                        corroboration_error_x_pixels=root_x + 100.0,
                        corroboration_error_y_pixels=100.0,
                        identity_deadline_ns=deadline_ns,
                    ),
                    ScreenErrorObservation(
                        root_ns + 8 * NS_PER_MS,
                        root_x + 4.8,
                        0.0,
                        velocity_error_x_pixels=root_x + 4.8,
                        velocity_error_y_pixels=0.0,
                        identity_deadline_ns=deadline_ns,
                    ),
                    1,
                    1,
                ),
            )

        prior = controller._correlated_lookahead_authority_state  # noqa: SLF001
        assert prior is not None
        self.assertGreater(prior.projection_x, 0.0)
        ambiguous = replace(
            prior,
            ordinary_x=0.0,
            total_x=0.0,
            velocity_direction_x=1,
            ambiguous_projection_retained_x=True,
        )
        previous = controller._last_observation  # noqa: SLF001
        assert previous is not None
        previous_x = previous.velocity_error_x_pixels
        assert previous_x is not None
        # Keep the observer below its separate material-reversal speed gate so
        # this specifically exercises the whole-batch bridge veto.
        controller._velocity_x = 100.0  # noqa: SLF001
        root_ns = previous.timestamp_ns
        batch = CorrelatedLookaheadObservation(
            ScreenErrorObservation(
                root_ns,
                previous_x,
                0.0,
                velocity_error_x_pixels=previous_x,
                velocity_error_y_pixels=0.0,
                corroboration_error_x_pixels=previous_x + 100.0,
                corroboration_error_y_pixels=100.0,
                identity_deadline_ns=deadline_ns,
            ),
            ScreenErrorObservation(
                root_ns + 8 * NS_PER_MS,
                previous_x - 0.1,
                0.0,
                velocity_error_x_pixels=previous_x - 0.1,
                velocity_error_y_pixels=0.0,
                identity_deadline_ns=deadline_ns,
            ),
            1,
            1,
        )
        with patch.object(
            MakcuCalibratedController,
            "_correlated_lookahead_authority",
            return_value=ambiguous,
        ):
            discontinuity, _, retained, verified_x, verified_y = (
                controller._accept_correlated_lookahead(batch)  # noqa: SLF001
            )

        self.assertIsNone(discontinuity)
        self.assertFalse(verified_x)
        self.assertFalse(verified_y)
        self.assertFalse(
            controller._paired_material_stop_or_reversal_x  # noqa: SLF001
        )
        self.assertEqual(retained.projection_x, 0.0)
        self.assertFalse(retained.ambiguous_projection_retained_x)

    def test_correlated_lookahead_authority_expires_at_endpoint_lease(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                stale_after_seconds=0.110,
                maximum_observation_interval_seconds=0.040,
                velocity_filter_time_constant_seconds=0.014,
                maximum_target_acceleration_pixels_per_second_squared=40_000.0,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
                maximum_correlated_lookahead_pursuit_feedforward_fraction=0.60,
            ),
        )
        base_ns = 20 * NS_PER_SECOND
        identity_deadline_ns = base_ns + 2 * NS_PER_SECOND
        final_endpoint_ns = 0
        for index in range(30):
            root_ns = base_ns + index * 10 * NS_PER_MS
            root_x = index * 6.0
            final_endpoint_ns = root_ns + 8 * NS_PER_MS
            output = controller.step(
                final_endpoint_ns + NS_PER_MS,
                engaged=True,
                correlated_lookahead=CorrelatedLookaheadObservation(
                    ScreenErrorObservation(
                        root_ns,
                        root_x,
                        0.0,
                        velocity_error_x_pixels=root_x,
                        velocity_error_y_pixels=0.0,
                        corroboration_error_x_pixels=root_x + 100.0,
                        corroboration_error_y_pixels=100.0,
                        identity_deadline_ns=identity_deadline_ns,
                    ),
                    ScreenErrorObservation(
                        final_endpoint_ns,
                        root_x + 4.8,
                        0.0,
                        velocity_error_x_pixels=root_x + 4.8,
                        velocity_error_y_pixels=0.0,
                        identity_deadline_ns=identity_deadline_ns,
                    ),
                    1,
                    1,
                ),
            )
        self.assertGreater(output.velocity_feedforward_confidence_x, 0.50)

        before = controller.step(
            final_endpoint_ns + 25 * NS_PER_MS - 1,
            engaged=True,
        )
        self.assertGreater(before.velocity_feedforward_confidence_x, 0.50)
        at_deadline = controller.step(
            final_endpoint_ns + 25 * NS_PER_MS,
            engaged=True,
        )
        self.assertTrue(at_deadline.valid)
        self.assertEqual(at_deadline.velocity_feedforward_confidence_x, 0.0)
        self.assertTrue(at_deadline.predictive_authority_revoked_x)
        self.assertEqual(
            controller._body_derived_adaptive_pursuit_confidence_x,
            0.0,
        )
        after = controller.step(
            final_endpoint_ns + 25 * NS_PER_MS + 1,
            engaged=True,
        )
        self.assertEqual(after.velocity_feedforward_confidence_x, 0.0)
        self.assertFalse(after.predictive_authority_revoked_x)

    def test_ordinary_handoff_clears_correlated_metadata_and_only_snaps_static(
        self,
    ) -> None:
        def established_controller() -> tuple[MakcuCalibratedController, int, int]:
            controller = MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.008),
                _test_config(
                    stale_after_seconds=0.110,
                    maximum_observation_interval_seconds=0.040,
                    velocity_filter_time_constant_seconds=0.014,
                    maximum_target_acceleration_pixels_per_second_squared=(
                        40_000.0
                    ),
                    maximum_velocity_feedforward_fraction=1.0,
                    require_motion_corroboration_for_feedforward=True,
                    maximum_body_derived_projection_fraction=1.0,
                    maximum_body_derived_feedforward_fraction=0.50,
                    maximum_body_derived_pursuit_feedforward_fraction=0.90,
                ),
            )
            base_ns = 25 * NS_PER_SECOND
            identity_deadline_ns = base_ns + 2 * NS_PER_SECOND
            for index in range(12):
                root_ns = base_ns + index * 10 * NS_PER_MS
                root_x = index * 6.0
                output = controller.step(
                    root_ns + 11 * NS_PER_MS,
                    engaged=True,
                    correlated_lookahead=CorrelatedLookaheadObservation(
                        ScreenErrorObservation(
                            root_ns,
                            root_x,
                            0.0,
                            velocity_error_x_pixels=root_x,
                            velocity_error_y_pixels=0.0,
                            corroboration_error_x_pixels=root_x + 100.0,
                            corroboration_error_y_pixels=100.0,
                            identity_deadline_ns=identity_deadline_ns,
                        ),
                        ScreenErrorObservation(
                            root_ns + 10 * NS_PER_MS,
                            root_x + 6.0,
                            0.0,
                            velocity_error_x_pixels=root_x + 6.0,
                            velocity_error_y_pixels=0.0,
                            identity_deadline_ns=identity_deadline_ns,
                        ),
                        5,
                        8,
                    ),
                )
            self.assertTrue(output.correlated_lookahead_active)
            self.assertGreater(output.lookahead_retained_authority_x, 0.0)
            return controller, base_ns, identity_deadline_ns

        for replacement in ("static", "independent"):
            with self.subTest(replacement=replacement):
                controller, base_ns, identity_deadline_ns = (
                    established_controller()
                )
                ordinary_ns = base_ns + 130 * NS_PER_MS
                observation_kwargs = {}
                if replacement == "independent":
                    observation_kwargs = {
                        "corroboration_error_x_pixels": 178.0,
                        "corroboration_error_y_pixels": 100.0,
                    }
                output = controller.step(
                    ordinary_ns + NS_PER_MS,
                    engaged=True,
                    observation=ScreenErrorObservation(
                        ordinary_ns,
                        78.0,
                        0.0,
                        velocity_error_x_pixels=78.0,
                        velocity_error_y_pixels=0.0,
                        identity_deadline_ns=identity_deadline_ns,
                        **observation_kwargs,
                    ),
                )
                self.assertTrue(output.valid)
                self.assertFalse(output.correlated_lookahead_active)
                self.assertEqual(output.lookahead_retained_authority_x, 0.0)
                self.assertEqual(output.lookahead_retained_authority_y, 0.0)
                if replacement == "static":
                    self.assertEqual(
                        output.velocity_feedforward_confidence_x,
                        0.0,
                    )
                    self.assertTrue(output.predictive_authority_revoked_x)
                else:
                    self.assertGreater(
                        output.velocity_feedforward_confidence_x,
                        0.0,
                    )
                    self.assertFalse(output.predictive_authority_revoked_x)

    def test_correlated_lookahead_cannot_arm_from_lookahead_motion_alone(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
                maximum_correlated_lookahead_pursuit_feedforward_fraction=0.60,
                maximum_verified_flow_pursuit_feedforward_fraction=0.90,
            ),
        )
        deadline_ns = NS_PER_SECOND
        output = controller.step(
            11 * NS_PER_MS,
            engaged=True,
            correlated_lookahead=CorrelatedLookaheadObservation(
                ScreenErrorObservation(
                    0,
                    0.0,
                    0.0,
                    velocity_error_x_pixels=0.0,
                    velocity_error_y_pixels=0.0,
                    corroboration_error_x_pixels=100.0,
                    corroboration_error_y_pixels=100.0,
                    identity_deadline_ns=deadline_ns,
                ),
                ScreenErrorObservation(
                    10 * NS_PER_MS,
                    8.0,
                    4.0,
                    velocity_error_x_pixels=8.0,
                    velocity_error_y_pixels=4.0,
                    identity_deadline_ns=deadline_ns,
                ),
                1,
                1,
                verified_flow_motion=True,
            ),
        )
        self.assertTrue(output.valid)
        self.assertFalse(output.correlated_lookahead_active)
        self.assertEqual(output.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(output.velocity_feedforward_confidence_y, 0.0)
        self.assertEqual(output.lookahead_retained_authority_x, 0.0)
        self.assertEqual(output.lookahead_retained_authority_y, 0.0)

    def test_correlated_equal_timestamp_identity_change_reseeds_evidence(
        self,
    ) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            _test_config(
                stale_after_seconds=0.110,
                maximum_observation_interval_seconds=0.040,
                maximum_velocity_feedforward_fraction=1.0,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=1.0,
                maximum_body_derived_feedforward_fraction=0.50,
                maximum_body_derived_pursuit_feedforward_fraction=0.90,
            ),
        )
        base_ns = 30 * NS_PER_SECOND
        deadline_ns = base_ns + 2 * NS_PER_SECOND
        final_endpoint_ns = 0
        final_x = 0.0
        for index in range(12):
            root_ns = base_ns + index * 10 * NS_PER_MS
            root_x = index * 6.0
            final_endpoint_ns = root_ns + 10 * NS_PER_MS
            final_x = root_x + 6.0
            qualified = controller.step(
                final_endpoint_ns + NS_PER_MS,
                engaged=True,
                correlated_lookahead=CorrelatedLookaheadObservation(
                    ScreenErrorObservation(
                        root_ns,
                        root_x,
                        0.0,
                        velocity_error_x_pixels=root_x,
                        velocity_error_y_pixels=0.0,
                        corroboration_error_x_pixels=root_x + 100.0,
                        corroboration_error_y_pixels=100.0,
                        identity_deadline_ns=deadline_ns,
                    ),
                    ScreenErrorObservation(
                        final_endpoint_ns,
                        final_x,
                        0.0,
                        velocity_error_x_pixels=final_x,
                        velocity_error_y_pixels=0.0,
                        identity_deadline_ns=deadline_ns,
                    ),
                    1,
                    1,
                ),
            )
        self.assertGreater(qualified.velocity_feedforward_confidence_x, 0.40)

        # A nearby replacement shares the already-admitted primary timestamp,
        # but its explicit identity tuple differs. It must not inherit the old
        # corroboration covariance, direction horizon, or retained cap.
        replacement = controller.step(
            final_endpoint_ns + 11 * NS_PER_MS,
            engaged=True,
            correlated_lookahead=CorrelatedLookaheadObservation(
                ScreenErrorObservation(
                    final_endpoint_ns,
                    final_x,
                    0.0,
                    velocity_error_x_pixels=final_x,
                    velocity_error_y_pixels=0.0,
                    corroboration_error_x_pixels=final_x + 101.0,
                    corroboration_error_y_pixels=100.0,
                    identity_deadline_ns=deadline_ns,
                ),
                ScreenErrorObservation(
                    final_endpoint_ns + 10 * NS_PER_MS,
                    final_x + 6.0,
                    0.0,
                    velocity_error_x_pixels=final_x + 6.0,
                    velocity_error_y_pixels=0.0,
                    identity_deadline_ns=deadline_ns,
                ),
                2,
                2,
            ),
        )
        self.assertEqual(replacement.motion_corroboration_confidence, 0.0)
        self.assertEqual(replacement.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(replacement.lookahead_retained_authority_x, 0.0)

    def test_correlated_lookahead_rejects_ungated_numeric_profile(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.1, 0.1, 0.0),
            _test_config(require_motion_corroboration_for_feedforward=False),
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires corroboration-gated control",
        ):
            controller.step(
                11 * NS_PER_MS,
                engaged=True,
                correlated_lookahead=CorrelatedLookaheadObservation(
                    ScreenErrorObservation(
                        0,
                        0.0,
                        0.0,
                        velocity_error_x_pixels=0.0,
                        velocity_error_y_pixels=0.0,
                        corroboration_error_x_pixels=100.0,
                        corroboration_error_y_pixels=100.0,
                        identity_deadline_ns=NS_PER_SECOND,
                    ),
                    ScreenErrorObservation(
                        10 * NS_PER_MS,
                        5.0,
                        0.0,
                        velocity_error_x_pixels=5.0,
                        velocity_error_y_pixels=0.0,
                        identity_deadline_ns=NS_PER_SECOND,
                    ),
                    1,
                    1,
                ),
            )

    def test_correlated_capture_phase_reduces_lag_without_destabilizing_lock(
        self,
    ) -> None:
        position_only = _run_capture_phase_lookahead_plant(correlated=False)
        correlated = _run_capture_phase_lookahead_plant(correlated=True)

        # The target/noise timeline, endpoint schedule, feedback tuning, rate
        # cap, and physical plant are identical. Retaining only P0's already-
        # earned authority must materially reduce pursuit lag at the newer P1.
        self.assertLess(
            correlated.moving_mean_lag_pixels,
            position_only.moving_mean_lag_pixels * 0.75,
            (position_only, correlated),
        )
        self.assertLess(
            correlated.moving_p95_lag_pixels,
            position_only.moving_p95_lag_pixels * 0.75,
            (position_only, correlated),
        )

        # That improvement cannot come from leaving predictive motion latched
        # across a reversal or stop, nor from trading pursuit lag for shake.
        self.assertLessEqual(
            correlated.maximum_reversal_error_pixels,
            position_only.maximum_reversal_error_pixels,
            (position_only, correlated),
        )
        self.assertLessEqual(
            correlated.post_stop_rms_pixels,
            position_only.post_stop_rms_pixels,
            (position_only, correlated),
        )
        self.assertLessEqual(
            correlated.post_stop_p95_pixels,
            position_only.post_stop_p95_pixels,
            (position_only, correlated),
        )
        # A zero-inertia plant can cross exact zero while its final six
        # milliseconds of already-emitted commands land. Keep that bounded
        # inside the existing continuous-feedback shoulder rather than hiding
        # it behind the position-only run's much larger trailing error.
        self.assertLessEqual(
            correlated.maximum_post_stop_overshoot_pixels,
            6.0,
            (position_only, correlated),
        )
        self.assertLessEqual(
            correlated.stationary_rms_pixels,
            position_only.stationary_rms_pixels,
            (position_only, correlated),
        )
        self.assertLessEqual(
            correlated.stationary_p95_pixels,
            position_only.stationary_p95_pixels,
            (position_only, correlated),
        )
        self.assertLessEqual(
            correlated.steady_abs_counts_per_second,
            position_only.steady_abs_counts_per_second,
            (position_only, correlated),
        )

    def test_correlated_capture_phase_carries_only_staged_earned_pursuit(
        self,
    ) -> None:
        ordinary = _run_capture_phase_lookahead_plant(correlated=True)
        retained = _run_capture_phase_lookahead_plant(
            correlated=True,
            correlated_pursuit_feedforward_fraction=0.60,
        )

        # The position stream, plant, feedback, ordinary 50% authority, and
        # target/noise timeline are identical. Only a reserve learned at one
        # P1 and re-proven by the following independent P0 may reach 60%.
        self.assertLess(
            retained.moving_mean_lag_pixels,
            ordinary.moving_mean_lag_pixels * 0.95,
            (ordinary, retained),
        )
        self.assertLess(
            retained.moving_p95_lag_pixels,
            ordinary.moving_p95_lag_pixels * 0.98,
            (ordinary, retained),
        )

        # The staged ceiling must not buy pursuit by disturbing established
        # lock or materially weakening the existing stop/reversal boundary.
        self.assertLessEqual(
            retained.stationary_rms_pixels,
            ordinary.stationary_rms_pixels + 0.05,
            (ordinary, retained),
        )
        self.assertLessEqual(
            retained.stationary_p95_pixels,
            ordinary.stationary_p95_pixels + 0.05,
            (ordinary, retained),
        )
        self.assertLessEqual(
            retained.reversal_rms_pixels,
            ordinary.reversal_rms_pixels * 1.02,
            (ordinary, retained),
        )
        self.assertLessEqual(
            retained.maximum_reversal_error_pixels,
            ordinary.maximum_reversal_error_pixels * 1.03,
            (ordinary, retained),
        )
        self.assertLessEqual(
            retained.maximum_post_stop_overshoot_pixels,
            ordinary.maximum_post_stop_overshoot_pixels + 2.0,
            (ordinary, retained),
        )
        self.assertLessEqual(
            retained.post_stop_rms_pixels,
            ordinary.post_stop_rms_pixels + 0.40,
            (ordinary, retained),
        )

    def test_verified_flow_reserve_reduces_fast_lag_without_changing_lock(
        self,
    ) -> None:
        ordinary = _run_capture_phase_lookahead_plant(
            correlated=True,
            correlated_pursuit_feedforward_fraction=0.60,
            moving_velocity_x_pixels_per_second=1_500.0,
        )
        verified = _run_capture_phase_lookahead_plant(
            correlated=True,
            correlated_pursuit_feedforward_fraction=0.60,
            verified_flow_pursuit_feedforward_fraction=0.90,
            verified_flow_motion=True,
            moving_velocity_x_pixels_per_second=1_500.0,
        )

        # The only changed input is the explicit consecutive-flow contract.
        # At this speed it may carry more of the reserve already earned by the
        # independent root, rather than waiting for a fresh trailing error.
        self.assertLess(
            verified.moving_mean_lag_pixels,
            ordinary.moving_mean_lag_pixels * 0.75,
            (ordinary, verified),
        )
        self.assertLess(
            verified.moving_rms_pixels,
            ordinary.moving_rms_pixels * 0.80,
            (ordinary, verified),
        )
        self.assertLessEqual(
            verified.moving_p95_lag_pixels,
            ordinary.moving_p95_lag_pixels,
            (ordinary, verified),
        )

        # The endpoint remains unable to retain authority through a material
        # reversal or stop, and the low-speed ceiling is identical. The brief
        # settling history may differ, but stationary noise/output must stay
        # inside the same tight envelope.
        self.assertLessEqual(
            verified.reversal_rms_pixels,
            ordinary.reversal_rms_pixels * 1.06,
            (ordinary, verified),
        )
        self.assertLessEqual(
            verified.maximum_post_stop_overshoot_pixels,
            15.0,
            (ordinary, verified),
        )
        self.assertLessEqual(
            verified.stationary_rms_pixels,
            ordinary.stationary_rms_pixels + 1.0,
            (ordinary, verified),
        )
        self.assertLessEqual(
            verified.steady_abs_counts_per_second,
            ordinary.steady_abs_counts_per_second + 4.0,
            (ordinary, verified),
        )

    def test_decoupled_verified_flow_ceiling_exact_gain_speed_sweep(self) -> None:
        """Lock the measured-flow-only rollout envelope across gain error."""

        expected = {
            # speed, physical gain: lag reduction, added reversal, added stop,
            # added stationary RMS.  The 2400/0.8 corner is output-capacity
            # bound (more than 53% saturation), so its smaller gain is expected.
            (900.0, 0.8): (0.321374, 1.710639, 1.241465, 0.051932),
            (900.0, 1.0): (0.000154, 0.0, 0.0, 0.000020),
            (900.0, 1.2): (0.0, 0.0, 0.0, -0.000234),
            (1_500.0, 0.8): (0.410964, 3.402709, 5.130434, 0.068675),
            (1_500.0, 1.0): (0.367736, 2.363131, 3.689512, 0.048893),
            (1_500.0, 1.2): (0.335574, 2.176598, 3.230671, 0.031671),
            (2_400.0, 0.8): (0.077503, -0.305323, 0.0, 0.105758),
            (2_400.0, 1.0): (0.444864, 4.106974, 7.142001, 0.003057),
            (2_400.0, 1.2): (0.419557, 3.981437, 6.439876, 0.001069),
        }
        for (speed, gain_scale), expected_metrics in expected.items():
            with self.subTest(speed=speed, gain_scale=gain_scale):
                common = dict(
                    correlated=True,
                    correlated_pursuit_feedforward_fraction=0.60,
                    residual_pursuit_feedforward_fraction=0.65,
                    body_derived_pursuit_feedforward_fraction=0.82,
                    verified_flow_motion=True,
                    corroboration_motion_scale=1.0,
                    moving_velocity_x_pixels_per_second=speed,
                    physical_gain_scale=gain_scale,
                    noise_scale=2.0,
                    maximum_screen_rate_pixels_per_second=3_000.0,
                )
                baseline = _run_capture_phase_lookahead_plant(
                    **common,
                    verified_flow_pursuit_feedforward_fraction=0.82,
                )
                decoupled = _run_capture_phase_lookahead_plant(
                    **common,
                    verified_flow_pursuit_feedforward_fraction=0.95,
                )
                measured = (
                    (
                        baseline.moving_mean_lag_pixels
                        - decoupled.moving_mean_lag_pixels
                    )
                    / baseline.moving_mean_lag_pixels,
                    decoupled.maximum_reversal_error_pixels
                    - baseline.maximum_reversal_error_pixels,
                    decoupled.maximum_post_stop_overshoot_pixels
                    - baseline.maximum_post_stop_overshoot_pixels,
                    decoupled.stationary_rms_pixels
                    - baseline.stationary_rms_pixels,
                )
                for actual, reference in zip(
                    measured,
                    expected_metrics,
                    strict=True,
                ):
                    self.assertAlmostEqual(actual, reference, delta=0.002)
                self.assertLessEqual(
                    decoupled.saturated_output_fraction,
                    baseline.saturated_output_fraction + 0.006,
                )

    def test_verified_flow_residual_closes_lag_without_independent_motion(
        self,
    ) -> None:
        for name, corroboration_motion_scale in (
            ("stationary", 0.0),
            ("opposed", -1.0),
        ):
            with self.subTest(corroboration=name):
                common = dict(
                    correlated=True,
                    correlated_pursuit_feedforward_fraction=0.60,
                    verified_flow_pursuit_feedforward_fraction=0.60,
                    body_derived_pursuit_feedforward_fraction=0.82,
                    verified_flow_motion=True,
                    corroboration_motion_scale=corroboration_motion_scale,
                    moving_velocity_x_pixels_per_second=1_500.0,
                )
                disabled = _run_capture_phase_lookahead_plant(**common)
                enabled = _run_capture_phase_lookahead_plant(
                    **common,
                    residual_pursuit_feedforward_fraction=0.65,
                )

                # Command-compensated stationary or opposed corroboration
                # never authorizes the ordinary P0/body grant. The only A/B
                # change is therefore the residual learned from sustained,
                # numerically accepted verified P1 displacement.
                self.assertEqual(
                    disabled.maximum_moving_motion_corroboration_confidence,
                    0.0,
                )
                self.assertEqual(
                    enabled.maximum_moving_motion_corroboration_confidence,
                    0.0,
                )
                self.assertLess(
                    enabled.moving_mean_lag_pixels,
                    disabled.moving_mean_lag_pixels * 0.65,
                    (disabled, enabled),
                )
                self.assertLess(
                    enabled.moving_p95_lag_pixels,
                    disabled.moving_p95_lag_pixels * 0.65,
                    (disabled, enabled),
                )

                # The additive reserve must still revoke at the material
                # reversal and stop. It may not exchange lower pursuit lag
                # for stationary shake or additional command activity.
                self.assertLessEqual(
                    enabled.maximum_reversal_error_pixels,
                    disabled.maximum_reversal_error_pixels,
                    (disabled, enabled),
                )
                self.assertLessEqual(
                    enabled.maximum_post_stop_overshoot_pixels,
                    6.0,
                    (disabled, enabled),
                )
                self.assertLessEqual(
                    enabled.post_stop_p95_pixels,
                    disabled.post_stop_p95_pixels,
                    (disabled, enabled),
                )
                self.assertLessEqual(
                    enabled.stationary_rms_pixels,
                    disabled.stationary_rms_pixels + 0.25,
                    (disabled, enabled),
                )
                self.assertLessEqual(
                    enabled.stationary_p95_pixels,
                    disabled.stationary_p95_pixels + 0.25,
                    (disabled, enabled),
                )
                self.assertLessEqual(
                    enabled.steady_abs_counts_per_second,
                    disabled.steady_abs_counts_per_second + 1.0,
                    (disabled, enabled),
                )

    def test_verified_residual_projection_sweep_reduces_known_horizon_lag(
        self,
    ) -> None:
        """Exercise the live envelope across speed, gain error, and jitter."""

        for speed in (900.0, 1_500.0, 2_400.0):
            for gain_scale in (0.80, 1.0, 1.20):
                with self.subTest(speed=speed, gain_scale=gain_scale):
                    common = dict(
                        correlated=True,
                        correlated_pursuit_feedforward_fraction=0.60,
                        verified_flow_pursuit_feedforward_fraction=0.60,
                        body_derived_pursuit_feedforward_fraction=0.82,
                        corroboration_motion_scale=0.0,
                        moving_velocity_x_pixels_per_second=speed,
                        physical_gain_scale=gain_scale,
                        noise_scale=2.0,
                        maximum_screen_rate_pixels_per_second=3_000.0,
                    )
                    disabled = _run_capture_phase_lookahead_plant(
                        **common,
                        verified_flow_motion=True,
                        residual_pursuit_feedforward_fraction=0.0,
                    )
                    unverified = _run_capture_phase_lookahead_plant(
                        **common,
                        verified_flow_motion=False,
                        residual_pursuit_feedforward_fraction=0.65,
                    )
                    enabled = _run_capture_phase_lookahead_plant(
                        **common,
                        verified_flow_motion=True,
                        residual_pursuit_feedforward_fraction=0.65,
                    )
                    # Merely enabling the profile is bit-for-bit inert without
                    # the runtime and numeric verified-flow contract.
                    self.assertEqual(
                        unverified.moving_mean_lag_pixels,
                        disabled.moving_mean_lag_pixels,
                    )
                    self.assertEqual(
                        unverified.moving_p95_lag_pixels,
                        disabled.moving_p95_lag_pixels,
                    )
                    self.assertEqual(
                        unverified.maximum_reversal_error_pixels,
                        disabled.maximum_reversal_error_pixels,
                    )
                    self.assertEqual(
                        unverified.post_stop_p95_pixels,
                        disabled.post_stop_p95_pixels,
                    )
                    self.assertLessEqual(
                        abs(
                            unverified.stationary_rms_pixels
                            - disabled.stationary_rms_pixels
                        ),
                        0.02,
                    )
                    self.assertLessEqual(
                        enabled.moving_mean_lag_pixels,
                        disabled.moving_mean_lag_pixels * 1.001,
                    )
                    self.assertLessEqual(
                        enabled.moving_p95_lag_pixels,
                        disabled.moving_p95_lag_pixels * 1.001,
                    )
                    if speed >= 1_500.0 or gain_scale <= 1.0:
                        self.assertLess(
                            enabled.moving_mean_lag_pixels,
                            disabled.moving_mean_lag_pixels * 0.72,
                        )
                        self.assertLess(
                            enabled.moving_p95_lag_pixels,
                            disabled.moving_p95_lag_pixels * 0.80,
                        )

                    # The existing per-axis stop/reversal gates must buy the
                    # pursuit improvement without a stationary-output trade.
                    self.assertLessEqual(
                        enabled.maximum_reversal_error_pixels,
                        disabled.maximum_reversal_error_pixels + 1.0,
                    )
                    self.assertLessEqual(
                        enabled.maximum_post_stop_overshoot_pixels,
                        disabled.maximum_post_stop_overshoot_pixels + 1.0,
                    )
                    self.assertLessEqual(
                        enabled.post_stop_p95_pixels,
                        disabled.post_stop_p95_pixels + 0.5,
                    )
                    self.assertLessEqual(
                        enabled.stationary_rms_pixels,
                        disabled.stationary_rms_pixels + 0.15,
                    )
                    self.assertLessEqual(
                        enabled.stationary_p95_pixels,
                        disabled.stationary_p95_pixels + 0.25,
                    )
                    self.assertLessEqual(
                        enabled.steady_abs_counts_per_second,
                        disabled.steady_abs_counts_per_second + 1.0,
                    )


class CalibratedControlPlantTests(unittest.TestCase):
    def test_direct_point_automatic_tuning_bounds_noise_motion_and_gain_error(
        self,
    ) -> None:
        candidates = tuple(
            (time_constant, deadzone)
            for time_constant in (0.022, 0.035, 0.045, 0.060)
            for deadzone in (2.5, 3.0)
        )
        nominal = {
            (time_constant, deadzone, noise): _run_direct_point_plant(
                position_time_constant_seconds=time_constant,
                feedback_deadzone_pixels=deadzone,
                noise_amplitude_pixels=noise,
            )
            for time_constant, deadzone in candidates
            for noise in (0.4, 0.7, 1.0)
        }
        mismatched = {
            (time_constant, deadzone, scale): _run_direct_point_plant(
                position_time_constant_seconds=time_constant,
                feedback_deadzone_pixels=deadzone,
                noise_amplitude_pixels=1.0,
                physical_gain_scale=scale,
            )
            for time_constant, deadzone in candidates
            for scale in (0.80, 1.20)
        }

        # With FF intentionally disabled, constant-motion lag is dominated by
        # the position time constant. The 22 ms candidate extends the requested
        # 35/45/60 ms sweep and wins at each tested deadzone.
        for deadzone in (2.5, 3.0):
            moving_rms = [
                nominal[(time_constant, deadzone, 1.0)].moving_rms_pixels
                for time_constant in (0.022, 0.035, 0.045, 0.060)
            ]
            self.assertLess(moving_rms[0], moving_rms[1] * 0.65)
            self.assertLess(moving_rms[1], moving_rms[2] * 0.80)
            self.assertLess(moving_rms[2], moving_rms[3] * 0.80)

        chosen_nominal = [
            nominal[(0.022, 3.0, noise)] for noise in (0.4, 0.7, 1.0)
        ]
        chosen_mismatch = [
            mismatched[(0.022, 3.0, scale)] for scale in (0.80, 1.20)
        ]
        self.assertLess(
            max(result.moving_rms_pixels for result in chosen_nominal),
            13.6,
        )
        self.assertLess(
            max(result.moving_p95_pixels for result in chosen_nominal),
            13.9,
        )
        self.assertLess(
            max(result.reversal_rms_pixels for result in chosen_nominal),
            15.0,
        )
        self.assertLess(
            max(result.stationary_rms_pixels for result in chosen_nominal),
            3.1,
        )
        self.assertLess(
            max(
                result.steady_abs_counts_per_second
                for result in chosen_nominal
            ),
            2.0,
        )

        # A physical gain 20% below the nominal calibration is the limiting
        # pursuit case. It remains bounded without FF or a stationary orbit.
        self.assertLess(
            max(result.moving_rms_pixels for result in chosen_mismatch),
            17.0,
        )
        self.assertLess(
            max(result.moving_p95_pixels for result in chosen_mismatch),
            17.3,
        )
        self.assertLess(
            max(result.reversal_rms_pixels for result in chosen_mismatch),
            18.4,
        )
        self.assertLess(
            max(
                result.maximum_reversal_error_pixels
                for result in chosen_mismatch
            ),
            22.6,
        )
        self.assertLess(
            max(result.stationary_rms_pixels for result in chosen_mismatch),
            2.9,
        )
        self.assertLess(
            max(
                result.steady_abs_counts_per_second
                for result in chosen_mismatch
            ),
            1.5,
        )
        self.assertLessEqual(
            max(
                result.maximum_requested_axis_rate
                for result in chosen_nominal + chosen_mismatch
            ),
            19_200.0,
        )

    def test_automatic_factory_enables_bounded_verified_mapped_motion(
        self,
    ) -> None:
        from main import _automatic_plant_aware_controller

        controller = _automatic_plant_aware_controller(max_step=320)
        self.assertEqual(
            controller.config.velocity_filter_time_constant_seconds,
            0.014,
        )
        self.assertEqual(controller.config.velocity_median_window, 3)
        self.assertEqual(
            controller.config.maximum_target_acceleration_pixels_per_second_squared,
            40_000.0,
        )
        self.assertEqual(controller.config.position_time_constant_seconds, 0.028)
        self.assertEqual(controller.config.feedback_deadzone_pixels, 4.5)
        self.assertTrue(controller.config.continuous_feedback_deadband)
        self.assertEqual(
            controller.config.continuous_feedback_shoulder_pixels,
            6.0,
        )
        self.assertEqual(
            controller.config.pursuit_position_time_constant_seconds,
            0.016,
        )
        self.assertEqual(
            controller.config.pursuit_position_time_constant_start_pixels,
            10.5,
        )
        self.assertEqual(
            controller.config.pursuit_position_time_constant_full_pixels,
            22.0,
        )
        self.assertTrue(controller.config.preserve_pursuit_position_feedback)
        self.assertTrue(controller.config.retain_ambiguous_correlated_projection)
        self.assertEqual(
            controller.config.additional_body_derived_projection_seconds,
            0.0,
        )
        self.assertEqual(controller.config.maximum_velocity_feedforward_fraction, 1.0)
        self.assertTrue(
            controller.config.require_motion_corroboration_for_feedforward
        )
        self.assertEqual(
            controller.config.maximum_body_derived_projection_fraction,
            1.0,
        )
        self.assertEqual(
            controller.config.maximum_body_derived_feedforward_fraction,
            0.50,
        )
        self.assertEqual(
            controller.config.maximum_body_derived_pursuit_feedforward_fraction,
            0.82,
        )
        self.assertEqual(
            controller.config.maximum_residual_pursuit_feedforward_fraction,
            0.65,
        )
        self.assertEqual(
            controller.config.maximum_correlated_lookahead_pursuit_feedforward_fraction,
            0.60,
        )
        self.assertEqual(
            controller.config.maximum_verified_flow_pursuit_feedforward_fraction,
            0.0,
        )
        measured_controller = _automatic_plant_aware_controller(
            max_step=320,
            plant=CalibratedPlant(0.1309, 0.1115, 0.0082),
        )
        self.assertEqual(
            measured_controller.config.maximum_body_derived_pursuit_feedforward_fraction,
            0.82,
        )
        self.assertEqual(
            measured_controller.config.maximum_residual_pursuit_feedforward_fraction,
            0.65,
        )
        self.assertEqual(
            measured_controller.config.maximum_correlated_lookahead_pursuit_feedforward_fraction,
            0.60,
        )
        self.assertEqual(
            measured_controller.config.maximum_verified_flow_pursuit_feedforward_fraction,
            0.95,
        )
        stable_body_controller = _automatic_plant_aware_controller(
            max_step=320,
            direct_head=False,
        )
        self.assertFalse(
            stable_body_controller.config.preserve_pursuit_position_feedback
        )
        self.assertFalse(
            stable_body_controller.config.retain_ambiguous_correlated_projection
        )
        output = None
        q = 1920.0 / 416.0
        for index in range(24):
            timestamp_ns = round(index * NS_PER_SECOND / 126.0)
            point_x = index * q
            output = controller.step(
                timestamp_ns,
                engaged=True,
                observation=ScreenErrorObservation(
                    timestamp_ns,
                    point_x,
                    0.0,
                    velocity_error_x_pixels=point_x,
                    velocity_error_y_pixels=0.0,
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=NS_PER_SECOND,
                    identity_deadline_ns=NS_PER_SECOND,
                ),
            )
        assert output is not None
        self.assertTrue(output.valid)
        self.assertGreater(output.target_velocity_x_pixels_per_second, 575.0)
        self.assertGreater(output.velocity_feedforward_confidence_x, 0.0)
        self.assertGreater(output.velocity_feedforward_confidence_x, 0.25)
        self.assertGreater(output.residual_pursuit_authority_x, 0.0)
        self.assertLessEqual(output.residual_pursuit_authority_x, 0.65)
        self.assertLessEqual(output.velocity_feedforward_confidence_x, 0.82)
        self.assertGreater(output.rate_x_counts_per_second, 0.0)

    def test_measured_direct_head_envelope_is_per_axis_and_near_lock_identical(
        self,
    ) -> None:
        from main import (
            _automatic_measured_direct_head_rate_limits,
            _automatic_plant_aware_controller,
        )

        plant = CalibratedPlant(
            0.1309219261247048,
            0.11146827699765323,
            0.008222712,
        )
        expanded_rates = _automatic_measured_direct_head_rate_limits(
            max_step=320,
            plant=plant,
        )
        self.assertAlmostEqual(expanded_rates[0], 3000.0 / plant.gain_x_pixels_per_count)
        self.assertAlmostEqual(expanded_rates[1], 3000.0 / plant.gain_y_pixels_per_count)
        self.assertGreater(expanded_rates[0], 19_200.0)
        self.assertGreater(expanded_rates[1], expanded_rates[0])

        conservative_rates = _automatic_measured_direct_head_rate_limits(
            max_step=200,
            plant=plant,
        )
        self.assertEqual(conservative_rates, (12_000.0, 12_000.0))

        baseline = _automatic_plant_aware_controller(
            max_step=320,
            plant=plant,
        )
        expanded = _automatic_plant_aware_controller(
            max_step=math.ceil(max(expanded_rates) / 60.0),
            plant=plant,
            maximum_rates_counts_per_second=expanded_rates,
        )
        # The two controllers see identical, unsaturated near-lock evidence.
        # Raising a ceiling must not change gain, filtering, or requested rate
        # anywhere below the old 19.2k-count boundary.
        for index, error_x in enumerate((8.0, 9.0, 10.0, 10.0, 10.0)):
            timestamp_ns = index * 10 * NS_PER_MS
            observation = ScreenErrorObservation(
                timestamp_ns,
                error_x,
                -error_x * 0.5,
            )
            baseline_output = baseline.step(
                timestamp_ns,
                engaged=True,
                observation=observation,
            )
            expanded_output = expanded.step(
                timestamp_ns,
                engaged=True,
                observation=observation,
            )
            self.assertEqual(expanded_output, baseline_output)

        # A far correction is the only changed regime: it can exceed the old
        # equal-axis ceiling but remains exactly bounded by the measured caps.
        far = _automatic_plant_aware_controller(
            max_step=math.ceil(max(expanded_rates) / 60.0),
            plant=plant,
            maximum_rates_counts_per_second=expanded_rates,
        )
        far.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(0, 150.0, 150.0),
        )
        far_output = far.step(
            10 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(
                10 * NS_PER_MS,
                150.0,
                150.0,
            ),
        )
        self.assertEqual(far_output.rate_x_counts_per_second, expanded_rates[0])
        self.assertEqual(far_output.rate_y_counts_per_second, expanded_rates[1])
        self.assertTrue(far_output.saturated_x)
        self.assertTrue(far_output.saturated_y)

        split = _automatic_plant_aware_controller(
            max_step=math.ceil(max(expanded_rates) / 60.0),
            plant=plant,
            maximum_rates_counts_per_second=expanded_rates,
            maximum_feedback_rates_counts_per_second=(19_200.0, 19_200.0),
        )
        self.assertEqual(
            split.config.maximum_feedback_rate_x_counts_per_second,
            19_200.0,
        )
        self.assertEqual(
            split.config.maximum_feedback_rate_y_counts_per_second,
            19_200.0,
        )
        split.step(
            0,
            engaged=True,
            observation=ScreenErrorObservation(
                0,
                150.0,
                150.0,
                velocity_error_x_pixels=150.0,
                velocity_error_y_pixels=150.0,
            ),
        )
        split_output = split.step(
            10 * NS_PER_MS,
            engaged=True,
            observation=ScreenErrorObservation(
                10 * NS_PER_MS,
                150.0,
                150.0,
                velocity_error_x_pixels=150.0,
                velocity_error_y_pixels=150.0,
            ),
        )
        self.assertEqual(split_output.rate_x_counts_per_second, 19_200.0)
        self.assertEqual(split_output.rate_y_counts_per_second, 19_200.0)
        self.assertTrue(split_output.saturated_x)
        self.assertTrue(split_output.saturated_y)

    def test_verified_mapped_motion_reduces_measured_live_cadence_lag(
        self,
    ) -> None:
        """Bounded lead fixes pursuit lag without changing plant feedback."""

        for physical_gain_scale in (0.80, 1.20):
            with self.subTest(physical_gain_scale=physical_gain_scale):
                feedback_only = _run_aged_body_derived_point_plant(
                    observation_hz=46.0,
                    processing_age_ms=26,
                    physical_gain_scale=physical_gain_scale,
                    circular_jitter_pixels=4.0,
                    body_derived_projection_fraction=0.0,
                    body_derived_feedforward_fraction=0.0,
                    maximum_rate_counts_per_second=12_000.0,
                )
                bounded_motion = _run_aged_body_derived_point_plant(
                    observation_hz=46.0,
                    processing_age_ms=26,
                    physical_gain_scale=physical_gain_scale,
                    circular_jitter_pixels=4.0,
                    body_derived_projection_fraction=1.0,
                    body_derived_feedforward_fraction=0.25,
                    maximum_rate_counts_per_second=12_000.0,
                )

                # 46 Hz and 26 ms are the newest active good-lock controller
                # cadence and point age. Keep the real 12 ms mapped-point LP,
                # 12,000 counts/s cap, and the former 45 ms feedback tuning in
                # the closed plant above.
                self.assertLess(
                    bounded_motion.moving_rms_pixels,
                    feedback_only.moving_rms_pixels * 0.70,
                )
                self.assertLess(
                    bounded_motion.moving_p95_pixels,
                    feedback_only.moving_p95_pixels * 0.90,
                )
                self.assertLess(
                    bounded_motion.reversal_rms_pixels,
                    feedback_only.reversal_rms_pixels * 0.75,
                )
                self.assertLessEqual(
                    bounded_motion.stationary_rms_pixels,
                    feedback_only.stationary_rms_pixels + 0.20,
                )
                self.assertLessEqual(
                    bounded_motion.steady_abs_counts_per_second,
                    feedback_only.steady_abs_counts_per_second * 1.05 + 1.0,
                )
                self.assertLessEqual(
                    bounded_motion.maximum_requested_axis_rate,
                    12_000.0,
                )

    def test_fast_pursuit_reserve_reduces_lag_without_lock_or_reversal_churn(
        self,
    ) -> None:
        baseline_results: list[_DirectPointRunResult] = []
        reserve_results: list[_DirectPointRunResult] = []
        nominal_fast_baseline: list[_DirectPointRunResult] = []
        nominal_fast_reserve: list[_DirectPointRunResult] = []

        for observation_hz, processing_age_ms in ((46.0, 26), (90.0, 19)):
            for physical_gain_scale in (0.80, 1.0, 1.20):
                for jitter_pixels in (4.0, 8.0):
                    for target_speed in (1200.0, 1600.0, 1800.0, 2400.0):
                        common = dict(
                            observation_hz=observation_hz,
                            processing_age_ms=processing_age_ms,
                            physical_gain_scale=physical_gain_scale,
                            circular_jitter_pixels=jitter_pixels,
                            body_derived_projection_fraction=1.0,
                            body_derived_feedforward_fraction=0.50,
                            maximum_rate_counts_per_second=19_200.0,
                            moving_velocity_x_pixels_per_second=target_speed,
                            position_time_constant_seconds=0.028,
                            feedback_deadzone_pixels=4.5,
                            continuous_feedback_deadband=True,
                            continuous_feedback_shoulder_pixels=6.0,
                            pursuit_position_time_constant_seconds=0.016,
                            pursuit_position_time_constant_start_pixels=10.5,
                            pursuit_position_time_constant_full_pixels=22.0,
                        )
                        baseline = _run_aged_body_derived_point_plant(
                            **common,
                            body_derived_pursuit_feedforward_fraction=0.0,
                        )
                        reserve = _run_aged_body_derived_point_plant(
                            **common,
                            body_derived_pursuit_feedforward_fraction=0.82,
                        )
                        baseline_results.append(baseline)
                        reserve_results.append(reserve)

                        with self.subTest(
                            observation_hz=observation_hz,
                            physical_gain_scale=physical_gain_scale,
                            jitter_pixels=jitter_pixels,
                            target_speed=target_speed,
                        ):
                            # The robust reserve is deliberately neutral below
                            # its fast-motion gate and at the cap-bound extreme.
                            # Across the useful middle it may trade a tiny tail
                            # increase for lower sustained RMS, but it cannot
                            # destabilize motion, reversal, or post-stop lock.
                            self.assertLessEqual(
                                reserve.moving_rms_pixels,
                                baseline.moving_rms_pixels * 1.02 + 0.05,
                            )
                            self.assertLessEqual(
                                reserve.moving_p95_pixels,
                                baseline.moving_p95_pixels * 1.04 + 0.05,
                            )
                            self.assertLessEqual(
                                reserve.reversal_rms_pixels,
                                baseline.reversal_rms_pixels * 1.003 + 0.05,
                            )
                            self.assertLessEqual(
                                reserve.maximum_reversal_error_pixels,
                                baseline.maximum_reversal_error_pixels
                                * 1.003
                                + 0.05,
                            )
                            self.assertLessEqual(
                                reserve.stationary_rms_pixels,
                                baseline.stationary_rms_pixels + 1.05,
                            )
                            self.assertLessEqual(
                                reserve.stationary_p95_pixels,
                                baseline.stationary_p95_pixels + 1.25,
                            )
                            self.assertLessEqual(
                                reserve.steady_abs_counts_per_second,
                                baseline.steady_abs_counts_per_second + 8.1,
                            )
                            self.assertLessEqual(
                                reserve.saturated_output_fraction,
                                baseline.saturated_output_fraction + 0.051,
                            )
                            self.assertLessEqual(
                                reserve.maximum_requested_axis_rate,
                                19_200.0,
                            )

                        if (
                            observation_hz == 90.0
                            and physical_gain_scale == 1.0
                            and target_speed in (1600.0, 1800.0)
                        ):
                            nominal_fast_baseline.append(baseline)
                            nominal_fast_reserve.append(reserve)

        def total(attribute: str, results: list[_DirectPointRunResult]) -> float:
            return sum(getattr(result, attribute) for result in results)

        # With the measured plant and current 90 Hz cadence, the combined
        # 1600/1800 px/s regime improves materially. Aggregate post-stop shake
        # metrics across all gain/jitter/cadence corners remain unchanged.
        self.assertLess(
            total("moving_rms_pixels", nominal_fast_reserve),
            total("moving_rms_pixels", nominal_fast_baseline) * 0.95,
        )
        for attribute in (
            "stationary_rms_pixels",
            "stationary_p95_pixels",
            "steady_abs_counts_per_second",
        ):
            self.assertLessEqual(
                total(attribute, reserve_results),
                total(attribute, baseline_results) * 1.01 + 0.05,
            )

    def test_measured_profile_adaptive_pursuit_handles_translation_first_source(
        self,
    ) -> None:
        baseline_results: list[_DirectPointRunResult] = []
        adaptive_results: list[_DirectPointRunResult] = []
        nominal_low_jitter_baseline: list[_DirectPointRunResult] = []
        nominal_low_jitter_adaptive: list[_DirectPointRunResult] = []

        for physical_gain_scale in (0.90, 1.0, 1.10):
            for jitter_pixels in (4.0, 8.0, 12.0):
                for target_speed in (1000.0, 1200.0, 1600.0, 1800.0, 2400.0):
                    common = dict(
                        observation_hz=90.0,
                        processing_age_ms=19,
                        physical_gain_scale=physical_gain_scale,
                        circular_jitter_pixels=jitter_pixels,
                        body_derived_projection_fraction=1.0,
                        body_derived_feedforward_fraction=0.50,
                        maximum_rate_counts_per_second=19_200.0,
                        moving_velocity_x_pixels_per_second=target_speed,
                        mapped_filter_time_constant_seconds=0.012,
                        position_time_constant_seconds=0.028,
                        feedback_deadzone_pixels=4.5,
                        continuous_feedback_deadband=True,
                        continuous_feedback_shoulder_pixels=6.0,
                        pursuit_position_time_constant_seconds=0.016,
                        pursuit_position_time_constant_start_pixels=10.5,
                        pursuit_position_time_constant_full_pixels=22.0,
                        translation_first_velocity_channel=True,
                        translation_first_position_channel=True,
                    )
                    baseline = _run_aged_body_derived_point_plant(
                        **common,
                        body_derived_pursuit_feedforward_fraction=0.0,
                    )
                    adaptive = _run_aged_body_derived_point_plant(
                        **common,
                        body_derived_pursuit_feedforward_fraction=0.90,
                    )
                    baseline_results.append(baseline)
                    adaptive_results.append(adaptive)
                    with self.subTest(
                        physical_gain_scale=physical_gain_scale,
                        jitter_pixels=jitter_pixels,
                        target_speed=target_speed,
                    ):
                        self.assertLessEqual(
                            adaptive.moving_rms_pixels,
                            baseline.moving_rms_pixels * 1.01 + 0.05,
                        )
                        self.assertLessEqual(
                            adaptive.moving_p95_pixels,
                            baseline.moving_p95_pixels * 1.02 + 0.05,
                        )
                        self.assertLessEqual(
                            adaptive.reversal_rms_pixels,
                            baseline.reversal_rms_pixels * 1.002 + 0.05,
                        )
                        self.assertLessEqual(
                            adaptive.stationary_rms_pixels,
                            baseline.stationary_rms_pixels + 0.80,
                        )
                        self.assertLessEqual(
                            adaptive.steady_abs_counts_per_second,
                            baseline.steady_abs_counts_per_second + 12.0,
                        )
                        self.assertLessEqual(
                            adaptive.maximum_requested_axis_rate,
                            19_200.0,
                        )
                    if (
                        physical_gain_scale == 1.0
                        and jitter_pixels == 4.0
                        and target_speed in (1200.0, 1600.0)
                    ):
                        nominal_low_jitter_baseline.append(baseline)
                        nominal_low_jitter_adaptive.append(adaptive)

        # With whole-screen position phase removed, the adaptive path now
        # supplies residual pursuit authority rather than repaying that source
        # lag. It still cuts nominal 1200/1600 px/s RMS by at least 17%.
        self.assertLess(
            sum(
                result.moving_rms_pixels
                for result in nominal_low_jitter_adaptive
            ),
            sum(
                result.moving_rms_pixels
                for result in nominal_low_jitter_baseline
            )
            * 0.83,
        )

    def test_continuous_direct_head_feedback_reduces_motion_error_and_churn(
        self,
    ) -> None:
        """A soft deadband permits faster pursuit without a breakaway kick."""

        former: list[_DirectPointRunResult] = []
        retuned: list[_DirectPointRunResult] = []
        for observation_hz, processing_age_ms in ((46.0, 26), (90.0, 17)):
            for physical_gain_scale in (0.80, 1.20):
                for jitter_pixels in (4.0, 8.0):
                    common = dict(
                        observation_hz=observation_hz,
                        processing_age_ms=processing_age_ms,
                        physical_gain_scale=physical_gain_scale,
                        circular_jitter_pixels=jitter_pixels,
                        body_derived_projection_fraction=1.0,
                        body_derived_feedforward_fraction=0.25,
                        maximum_rate_counts_per_second=9_600.0,
                    )
                    former.append(
                        _run_aged_body_derived_point_plant(
                            **common,
                            position_time_constant_seconds=0.040,
                            feedback_deadzone_pixels=3.5,
                        )
                    )
                    retuned.append(
                        _run_aged_body_derived_point_plant(
                            **common,
                            position_time_constant_seconds=0.018,
                            feedback_deadzone_pixels=4.5,
                            continuous_feedback_deadband=True,
                            continuous_feedback_shoulder_pixels=6.0,
                        )
                    )

        def mean(attribute: str, results: list[_DirectPointRunResult]) -> float:
            return sum(getattr(result, attribute) for result in results) / len(results)

        self.assertLess(
            mean("moving_rms_pixels", retuned),
            mean("moving_rms_pixels", former) * 0.85,
        )
        self.assertLess(
            mean("moving_p95_pixels", retuned),
            mean("moving_p95_pixels", former) * 0.90,
        )
        self.assertLess(
            mean("reversal_rms_pixels", retuned),
            mean("reversal_rms_pixels", former) * 0.90,
        )
        self.assertLess(
            mean("steady_abs_counts_per_second", retuned),
            mean("steady_abs_counts_per_second", former) * 0.75,
        )
        self.assertLessEqual(
            max(result.maximum_requested_axis_rate for result in retuned),
            9_600.0,
        )

    def test_shipped_residual_aligned_feedforward_improves_pursuit_without_lock_churn(
        self,
    ) -> None:
        """Fast-target authority improves pursuit without disturbing lock."""

        baseline: list[_DirectPointRunResult] = []
        shipped: list[_DirectPointRunResult] = []
        measured_plant_baseline: list[_DirectPointRunResult] = []
        measured_plant_shipped: list[_DirectPointRunResult] = []
        for observation_hz, processing_age_ms in ((46.0, 26), (90.0, 17)):
            for physical_gain_scale in (0.80, 1.0, 1.20):
                for jitter_pixels in (4.0, 8.0):
                    common = dict(
                        observation_hz=observation_hz,
                        processing_age_ms=processing_age_ms,
                        physical_gain_scale=physical_gain_scale,
                        circular_jitter_pixels=jitter_pixels,
                        moving_velocity_x_pixels_per_second=1800.0,
                        body_derived_projection_fraction=1.0,
                        maximum_rate_counts_per_second=19_200.0,
                        position_time_constant_seconds=0.018,
                        feedback_deadzone_pixels=4.5,
                        continuous_feedback_deadband=True,
                        continuous_feedback_shoulder_pixels=6.0,
                    )
                    former = _run_aged_body_derived_point_plant(
                        **common,
                        body_derived_feedforward_fraction=0.25,
                    )
                    combined = _run_aged_body_derived_point_plant(
                        **common,
                        body_derived_feedforward_fraction=0.50,
                    )
                    baseline.append(former)
                    shipped.append(combined)
                    if physical_gain_scale == 1.0:
                        measured_plant_baseline.append(former)
                        measured_plant_shipped.append(combined)

        def mean(attribute: str, results: list[_DirectPointRunResult]) -> float:
            return sum(getattr(result, attribute) for result in results) / len(
                results
            )

        # At the measured plant, across both live cadence/age pairs and two
        # mapped-point jitter amplitudes, residual-aligned feed-forward cuts
        # mean 1800 px/s pursuit RMS by more than 10%. Unlike time projection,
        # extra authority cannot authorize itself: both the fresh measured
        # residual and final command-aware residual must point with velocity.
        self.assertLess(
            mean("moving_rms_pixels", measured_plant_shipped),
            mean("moving_rms_pixels", measured_plant_baseline) * 0.92,
        )
        self.assertLess(
            mean("moving_p95_pixels", measured_plant_shipped),
            mean("moving_p95_pixels", measured_plant_baseline) * 0.995,
        )
        self.assertLess(
            mean("reversal_rms_pixels", measured_plant_shipped),
            mean("reversal_rms_pixels", measured_plant_baseline),
        )

        # A deliberately broad +/-20% physical-gain mismatch remains bounded;
        # each corner is no worse than the old 25% cap even when a weak plant
        # spends much of the run at the unchanged hard output ceiling.
        for former, combined in zip(baseline, shipped, strict=True):
            self.assertLessEqual(
                combined.moving_rms_pixels,
                former.moving_rms_pixels * 1.01,
            )
            self.assertLessEqual(
                combined.reversal_rms_pixels,
                former.reversal_rms_pixels * 1.01,
            )
            self.assertLessEqual(
                combined.saturated_output_fraction,
                former.saturated_output_fraction + 0.07,
            )

        # Stationary output remains inside the pre-existing envelope because
        # the new authority is unavailable when velocity and error disagree.
        self.assertLessEqual(
            max(result.maximum_reversal_error_pixels for result in shipped),
            max(result.maximum_reversal_error_pixels for result in baseline)
            * 1.01,
        )
        self.assertLess(
            max(result.stationary_rms_pixels for result in shipped),
            4.2,
        )
        self.assertLess(
            max(result.stationary_p95_pixels for result in shipped),
            6.3,
        )
        self.assertLess(
            mean("stationary_rms_pixels", shipped),
            mean("stationary_rms_pixels", baseline) * 1.15,
        )
        self.assertLess(
            max(result.steady_abs_counts_per_second for result in shipped),
            215.0,
        )

        # The hard output limit is invariant.  At the measured plant, hidden
        # pre-clamp saturation stays well below the removed projected-feedback
        # configuration while preserving the faster feed-forward pursuit.
        self.assertLessEqual(
            max(result.maximum_requested_axis_rate for result in shipped),
            19_200.0,
        )
        self.assertLess(
            max(
                result.saturated_output_fraction
                for result in measured_plant_shipped
            ),
            0.21,
        )
        self.assertLess(
            mean("saturated_output_fraction", measured_plant_shipped),
            0.16,
        )

    def test_direct_head_feedback_shoulder_reduces_lock_churn_without_slowing_pursuit(
        self,
    ) -> None:
        linear: list[_DirectPointRunResult] = []
        shouldered: list[_DirectPointRunResult] = []
        for observation_hz, processing_age_ms in ((46.0, 26), (90.0, 17)):
            for physical_gain_scale in (0.80, 1.20):
                for jitter_pixels in (4.0, 8.0, 12.0):
                    common = dict(
                        observation_hz=observation_hz,
                        processing_age_ms=processing_age_ms,
                        physical_gain_scale=physical_gain_scale,
                        circular_jitter_pixels=jitter_pixels,
                        body_derived_projection_fraction=1.0,
                        body_derived_feedforward_fraction=0.25,
                        maximum_rate_counts_per_second=19_200.0,
                        position_time_constant_seconds=0.018,
                        feedback_deadzone_pixels=4.5,
                        continuous_feedback_deadband=True,
                    )
                    linear.append(_run_aged_body_derived_point_plant(**common))
                    shouldered.append(
                        _run_aged_body_derived_point_plant(
                            **common,
                            continuous_feedback_shoulder_pixels=6.0,
                        )
                    )

        def mean(attribute: str, results: list[_DirectPointRunResult]) -> float:
            return sum(getattr(result, attribute) for result in results) / len(results)

        # Full linear response resumes at 10.5 px. The shoulder can spend at
        # most a small pursuit margin while materially removing back-and-forth
        # stationary command volume in the measured 4--12 px jitter range.
        self.assertLessEqual(
            mean("moving_rms_pixels", shouldered),
            mean("moving_rms_pixels", linear) * 1.04,
        )
        self.assertLessEqual(
            mean("moving_p95_pixels", shouldered),
            mean("moving_p95_pixels", linear) * 1.04,
        )
        self.assertLess(
            mean("steady_abs_counts_per_second", shouldered),
            mean("steady_abs_counts_per_second", linear) * 0.75,
        )
        self.assertLessEqual(
            max(result.maximum_requested_axis_rate for result in shouldered),
            19_200.0,
        )

    def test_direct_point_feedback_only_tuning_rejects_subpixel_noise(self) -> None:
        results = [
            _run_direct_point_plant(
                position_time_constant_seconds=0.022,
                feedback_deadzone_pixels=3.0,
                noise_amplitude_pixels=noise,
            )
            for noise in (0.4, 0.7, 1.0)
        ]
        self.assertLess(
            max(result.stationary_rms_pixels for result in results),
            3.1,
        )
        self.assertLess(
            max(result.steady_abs_counts_per_second for result in results),
            2.0,
        )

    def test_corroborated_feedforward_removes_60_120_hz_processing_lag(
        self,
    ) -> None:
        cases = (
            (60.0, 20),
            (120.0, 12),
        )
        for observation_hz, processing_age_ms in cases:
            for physical_gain_scale in (0.80, 1.20):
                with self.subTest(
                    observation_hz=observation_hz,
                    processing_age_ms=processing_age_ms,
                    physical_gain_scale=physical_gain_scale,
                ):
                    feedback_only = _run_aged_corroborated_direct_point_plant(
                        observation_hz=observation_hz,
                        processing_age_ms=processing_age_ms,
                        feedforward_fraction=0.0,
                        physical_gain_scale=physical_gain_scale,
                    )
                    corroborated = _run_aged_corroborated_direct_point_plant(
                        observation_hz=observation_hz,
                        processing_age_ms=processing_age_ms,
                        feedforward_fraction=0.95,
                        physical_gain_scale=physical_gain_scale,
                    )
                    # This uses source timestamps and real arrival age.  A
                    # separately observed body center confirms translation;
                    # the direct head remains the only position coordinate.
                    self.assertLess(
                        corroborated.moving_rms_pixels,
                        feedback_only.moving_rms_pixels * 0.70,
                    )
                    self.assertLess(
                        corroborated.moving_p95_pixels,
                        feedback_only.moving_p95_pixels * 0.72,
                    )
                    self.assertLess(
                        corroborated.reversal_rms_pixels,
                        feedback_only.reversal_rms_pixels,
                    )
                    self.assertLessEqual(
                        corroborated.stationary_rms_pixels,
                        feedback_only.stationary_rms_pixels + 0.15,
                    )
                    # With motion evidence gone, the automatic profile now
                    # freezes at the last absolute point rather than using
                    # residual observer velocity to creep inside its 3 px
                    # deadzone. Detector noise and integer count landing may
                    # therefore leave a bounded sub-3.5 px static residual.
                    self.assertLess(corroborated.stationary_rms_pixels, 3.5)
                    self.assertLess(
                        corroborated.steady_abs_counts_per_second,
                        4.0,
                    )
                    self.assertLessEqual(
                        corroborated.maximum_requested_axis_rate,
                        19_200.0,
                    )

    def test_direct_head_twelve_ms_position_response_cuts_live_cadence_lag(
        self,
    ) -> None:
        for physical_gain_scale in (0.80, 1.0, 1.20):
            with self.subTest(physical_gain_scale=physical_gain_scale):
                former = _run_aged_corroborated_direct_point_plant(
                    observation_hz=45.0,
                    processing_age_ms=29,
                    feedforward_fraction=0.95,
                    physical_gain_scale=physical_gain_scale,
                    position_time_constant_seconds=0.022,
                )
                faster = _run_aged_corroborated_direct_point_plant(
                    observation_hz=45.0,
                    processing_age_ms=29,
                    feedforward_fraction=0.95,
                    physical_gain_scale=physical_gain_scale,
                    position_time_constant_seconds=0.012,
                )

                # These are the measured controller-ingestion cadence and
                # median direct-point age from the 14:32 run. Faster position
                # feedback materially reduces pursuit and reversal error even
                # across the existing +/-20% uncalibrated plant envelope.
                self.assertLess(
                    faster.moving_rms_pixels,
                    former.moving_rms_pixels * 0.60,
                )
                self.assertLess(
                    faster.moving_p95_pixels,
                    former.moving_p95_pixels * 0.70,
                )
                self.assertLess(
                    faster.reversal_rms_pixels,
                    former.reversal_rms_pixels * 0.86,
                )
                self.assertLessEqual(
                    faster.stationary_rms_pixels,
                    former.stationary_rms_pixels + 0.15,
                )
                self.assertLess(faster.stationary_rms_pixels, 3.5)
                self.assertLess(faster.steady_abs_counts_per_second, 6.0)
                self.assertLessEqual(
                    faster.maximum_requested_axis_rate,
                    19_200.0,
                )

    def test_bounded_automatic_feedforward_handles_twenty_percent_gain_error(
        self,
    ) -> None:
        config = CalibratedControlConfig(
            position_time_constant_seconds=0.060,
            velocity_filter_time_constant_seconds=0.018,
            maximum_target_speed_pixels_per_second=3000.0,
            maximum_target_acceleration_pixels_per_second_squared=20_000.0,
            maximum_rate_x_counts_per_second=19_200.0,
            maximum_rate_y_counts_per_second=19_200.0,
            stale_after_seconds=0.065,
            maximum_observation_interval_seconds=0.040,
            velocity_median_window=5,
            maximum_velocity_feedforward_fraction=0.95,
        )
        results = [
            _run_tracked_jitter_plant(
                MakcuCalibratedController(
                    CalibratedPlant(0.125, 0.120, 0.008),
                    config,
                ),
                raw_velocity_channel=True,
                physical_gain_scale=scale,
            )
            for scale in (0.80, 1.20)
        ]

        moving_rms = [
            _rms(list(result.radial_errors[300:3200])) for result in results
        ]
        moving_p95 = [
            _percentile_95(list(result.radial_errors[300:3200]))
            for result in results
        ]
        reversal_rms = [
            _rms(list(result.radial_errors[2000:2300])) for result in results
        ]
        stationary_rms = [
            _rms(list(result.radial_errors[3300:])) for result in results
        ]

        # A 0.95 cap was the best candidate at or below 0.95 in the
        # deterministic +/-20% sweep: lower caps accumulated pursuit error,
        # while this setting retains sub-8 px moving RMS without instability.
        self.assertLess(max(moving_rms), 7.5)
        self.assertLess(max(moving_p95), 18.0)
        self.assertLess(max(reversal_rms), 17.0)
        self.assertLess(max(stationary_rms), 2.6)
        self.assertLess(
            max(result.steady_abs_counts_per_second for result in results),
            750.0,
        )
        self.assertLessEqual(
            max(result.maximum_requested_axis_rate for result in results),
            19_200.0,
        )

    def test_raw_velocity_channel_breaks_smoothed_feedback_limit_cycle(self) -> None:
        legacy = _run_smoothed_feedback_plant(raw_velocity_channel=False)
        paired = _run_smoothed_feedback_plant(raw_velocity_channel=True)
        legacy_steady_error = list(legacy.radial_errors[-1000:])
        paired_steady_error = list(paired.radial_errors[-1000:])
        legacy_steady_speed = list(legacy.estimated_speeds[-1000:])
        paired_steady_speed = list(paired.estimated_speeds[-1000:])
        legacy_counts = sum(
            abs(delta_x) + abs(delta_y)
            for tick, delta_x, delta_y in legacy.emitted_counts
            if tick >= 3000
        )
        paired_counts = sum(
            abs(delta_x) + abs(delta_y)
            for tick, delta_x, delta_y in paired.emitted_counts
            if tick >= 3000
        )

        # The legacy mixed-domain estimator interprets its own physical
        # response as target velocity and settles into a saturated orbit.
        self.assertGreater(_rms(legacy_steady_error), 100.0)
        self.assertGreater(sum(legacy_steady_speed) / 1000.0, 1_000.0)
        self.assertGreater(legacy_counts, 15_000)

        # Raw accepted points and landed commands share one coordinate domain:
        # a stationary target estimates zero velocity and emits no steady work.
        self.assertLess(_rms(paired_steady_error), 1.0)
        self.assertLess(max(paired_steady_speed), 0.01)
        self.assertEqual(paired_counts, 0)

    def test_automatic_seed_tracks_through_duplicate_box_mode_noise(self) -> None:
        """Exercise tracker arbitration and delayed control as one closed loop."""

        control_config = _test_config(
            maximum_rate_x_counts_per_second=19_200.0,
            maximum_rate_y_counts_per_second=19_200.0,
        )
        frame_shape = (1080, 1920, 3)
        center_x = frame_shape[1] / 2.0
        center_y = frame_shape[0] / 2.0
        head_ratio = 0.12
        narrow_width = frame_shape[1] * 0.078
        narrow_height = frame_shape[0] * 0.355
        wide_width = frame_shape[1] * 0.092
        wide_height = frame_shape[0] * 0.371

        def run(*, automatic: bool) -> dict[str, object]:
            controller = (
                MakcuCalibratedController(
                    CalibratedPlant(0.125, 0.120, 0.008),
                    control_config,
                )
                if automatic
                else None
            )
            legacy = None if automatic else _CurrentProportionalPi(control_config)
            tracker = TargetTracker(label="player", lost_grace_frames=1)
            error_x = 95.0
            error_y = 0.0
            fractional_x = 0.0
            fractional_y = 0.0
            delayed: list[tuple[int, int, int]] = []
            delayed_index = 0
            observation_index = 0
            radial_errors: list[float] = []
            tracked_head_y_offsets: list[float] = []
            maximum_rate = 0.0
            emitted_y = 0

            for tick in range(1, 3001):
                now_ns = tick * NS_PER_MS
                target_velocity_x, _target_velocity_y = _target_velocity(
                    tick / 1000.0
                )
                error_x += target_velocity_x * 0.001
                while (
                    delayed_index < len(delayed)
                    and delayed[delayed_index][0] <= now_ns
                ):
                    _impact_ns, delta_x, delta_y = delayed[delayed_index]
                    delayed_index += 1
                    error_x -= 0.125 * delta_x
                    error_y -= 0.112 * delta_y

                observation = None
                if tick == 1 or (tick - 1) % 8 == 0:
                    true_head_x = center_x + error_x
                    true_head_y = center_y + error_y

                    def detection(
                        width: float,
                        height: float,
                        target_y: float,
                        confidence: float,
                    ) -> Detection:
                        return Detection(
                            0,
                            "player",
                            confidence,
                            (
                                true_head_x - width / 2.0,
                                target_y - head_ratio * height,
                                true_head_x + width / 2.0,
                                target_y + (1.0 - head_ratio) * height,
                            ),
                        )

                    narrow_confidence = (
                        0.78 if observation_index % 2 else 0.91
                    )
                    wide_confidence = (
                        0.91 if observation_index % 2 else 0.78
                    )
                    narrow = detection(
                        narrow_width,
                        narrow_height,
                        true_head_y,
                        narrow_confidence,
                    )
                    # The alternative detector mode is wider and places its
                    # derived head 18 px higher. Its confidence wins every
                    # other frame, which would create vertical aim wiggle if
                    # association alternated box modes.
                    wide = detection(
                        wide_width,
                        wide_height,
                        true_head_y - 18.0,
                        wide_confidence,
                    )
                    candidates = (narrow,) if observation_index == 0 else (wide, narrow)
                    tracked = tracker.update(
                        candidates,
                        frame_shape,
                        measurement_ns=now_ns,
                    )
                    self.assertIsNotNone(tracked)
                    assert tracked is not None
                    self.assertEqual(tracked.confidence, narrow_confidence)
                    tracked_head_x, tracked_head_y = head_target_point(
                        tracked,
                        head_ratio,
                    )
                    tracked_head_y_offsets.append(tracked_head_y - true_head_y)
                    observation = ScreenErrorObservation(
                        now_ns,
                        tracked_head_x - center_x,
                        tracked_head_y - center_y,
                    )
                    observation_index += 1

                if controller is not None:
                    output = controller.step(
                        now_ns,
                        engaged=True,
                        observation=observation,
                    )
                    rate_x = output.rate_x_counts_per_second
                    rate_y = output.rate_y_counts_per_second
                else:
                    assert legacy is not None
                    rate_x, rate_y = legacy.step(now_ns, observation)
                maximum_rate = max(maximum_rate, abs(rate_x), abs(rate_y))

                fractional_x += rate_x * 0.001
                fractional_y += rate_y * 0.001
                delta_x = math.trunc(fractional_x)
                delta_y = math.trunc(fractional_y)
                fractional_x -= delta_x
                fractional_y -= delta_y
                if delta_x or delta_y:
                    if controller is not None:
                        emitted = EmittedMouseCommand(now_ns, delta_x, delta_y)
                        controller.preflight_emitted(emitted)
                        controller.record_emitted(emitted)
                    delayed.append(
                        (now_ns + 8 * NS_PER_MS, delta_x, delta_y)
                    )
                    emitted_y += abs(delta_y)
                radial_errors.append(math.hypot(error_x, error_y))

            return {
                "errors": radial_errors[300:],
                "head_y_offsets": tracked_head_y_offsets,
                "maximum_rate": maximum_rate,
                "emitted_y": emitted_y,
            }

        automatic = run(automatic=True)
        legacy = run(automatic=False)
        automatic_errors = automatic["errors"]
        legacy_errors = legacy["errors"]
        assert isinstance(automatic_errors, list)
        assert isinstance(legacy_errors, list)
        self.assertLess(_rms(automatic_errors), _rms(legacy_errors) * 0.30)
        self.assertLess(
            _percentile_95(automatic_errors),
            _percentile_95(legacy_errors) * 0.35,
        )
        self.assertLess(max(automatic_errors), 25.0)
        self.assertLessEqual(automatic["maximum_rate"], 19_200.0)
        self.assertEqual(automatic["emitted_y"], 0)
        head_y_offsets = automatic["head_y_offsets"]
        assert isinstance(head_y_offsets, list)
        self.assertLess(max(head_y_offsets) - min(head_y_offsets), 1e-9)

    def test_automatic_host_seed_beats_legacy_across_observed_response_modes(self) -> None:
        seed = CalibratedPlant(0.125, 0.120, 0.008)
        config = _test_config(
            maximum_rate_x_counts_per_second=19_200.0,
            maximum_rate_y_counts_per_second=19_200.0,
        )
        for gain_x in (0.118, 0.125, 0.127):
            for gain_y in (0.104, 0.120, 0.129):
                with self.subTest(gain_x=gain_x, gain_y=gain_y):
                    automatic = _run_fake_plant(
                        8,
                        calibrated=True,
                        physical_gain_x=gain_x,
                        physical_gain_y=gain_y,
                        controller_plant=seed,
                        control_config=config,
                    )
                    legacy = _run_fake_plant(
                        8,
                        calibrated=False,
                        physical_gain_x=gain_x,
                        physical_gain_y=gain_y,
                        control_config=config,
                    )
                    start = 300
                    automatic_error = [
                        math.hypot(x, y) for x, y in automatic.errors[start:]
                    ]
                    legacy_error = [
                        math.hypot(x, y) for x, y in legacy.errors[start:]
                    ]
                    self.assertLess(
                        _rms(automatic_error),
                        _rms(legacy_error) * 0.25,
                    )
                    self.assertLess(
                        _percentile_95(automatic_error),
                        _percentile_95(legacy_error) * 0.30,
                    )
                    self.assertLess(max(automatic_error), 25.0)

    def test_calibrated_control_beats_current_pi_across_gains_and_delays(self) -> None:
        for delay_ms in (12, 24, 50):
            with self.subTest(delay_ms=delay_ms):
                calibrated = _run_fake_plant(delay_ms, calibrated=True)
                baseline = _run_fake_plant(delay_ms, calibrated=False)
                # Exclude acquisition but retain every stop, reversal, noisy
                # sample, and one/two-frame detector callback gap.
                start = 180
                calibrated_radial = [
                    math.hypot(x, y) for x, y in calibrated.errors[start:]
                ]
                baseline_radial = [
                    math.hypot(x, y) for x, y in baseline.errors[start:]
                ]
                calibrated_rms = _rms(calibrated_radial)
                baseline_rms = _rms(baseline_radial)
                calibrated_p95 = _percentile_95(calibrated_radial)
                baseline_p95 = _percentile_95(baseline_radial)
                self.assertLess(calibrated_rms, baseline_rms * 0.55)
                self.assertLess(calibrated_p95, baseline_p95 * 0.60)

                config = _test_config()
                maximum_true_error = max(calibrated_radial)
                self.assertLess(maximum_true_error, 260.0)
                for output in calibrated.outputs:
                    self.assertIsInstance(output, CalibratedControlOutput)
                    assert isinstance(output, CalibratedControlOutput)
                    self.assertTrue(
                        math.isfinite(output.rate_x_counts_per_second)
                    )
                    self.assertTrue(
                        math.isfinite(output.rate_y_counts_per_second)
                    )
                    self.assertLessEqual(
                        abs(output.rate_x_counts_per_second),
                        config.maximum_rate_x_counts_per_second,
                    )
                    self.assertLessEqual(
                        abs(output.rate_y_counts_per_second),
                        config.maximum_rate_y_counts_per_second,
                    )
                    if output.valid:
                        for rate, error in (
                            (
                                output.rate_x_counts_per_second,
                                output.projected_error_x_pixels,
                            ),
                            (
                                output.rate_y_counts_per_second,
                                output.projected_error_y_pixels,
                            ),
                        ):
                            if abs(error) > config.wrong_way_guard_pixels:
                                self.assertGreaterEqual(rate * error, 0.0)

    def test_sustained_measurement_loss_stops_without_stored_motion(self) -> None:
        loss_after_ms = 300
        result = _run_fake_plant(
            50,
            calibrated=True,
            duration_ms=650,
            loss_after_ms=loss_after_ms,
        )
        before_loss = result.outputs[loss_after_ms - 1]
        self.assertIsInstance(before_loss, CalibratedControlOutput)
        assert isinstance(before_loss, CalibratedControlOutput)
        self.assertNotEqual(before_loss.rate_x_counts_per_second, 0.0)

        # This includes delayed plant effects from commands issued before
        # freshness expired.  They may still land, but no stored controller
        # motion survives the bounded observation-loss interval.
        for output in result.outputs[loss_after_ms + 45 :]:
            self.assertIsInstance(output, CalibratedControlOutput)
            assert isinstance(output, CalibratedControlOutput)
            self.assertFalse(output.valid)
            self.assertEqual(output.rate_x_counts_per_second, 0.0)
            self.assertEqual(output.rate_y_counts_per_second, 0.0)

    def test_physical_mouse_input_is_not_learned_as_target_velocity(self) -> None:
        plant = CalibratedPlant(0.1, 0.1, 0.010)
        accounted = MakcuCalibratedController(plant)
        unaccounted = MakcuCalibratedController(plant)

        def observation(timestamp_ns: int, error_x: float) -> ScreenErrorObservation:
            return ScreenErrorObservation(
                timestamp_ns,
                error_x,
                0.0,
                error_x,
                0.0,
            )

        first = observation(NS_PER_MS, 20.0)
        accounted.step(NS_PER_MS, engaged=True, observation=first)
        unaccounted.step(NS_PER_MS, engaged=True, observation=first)
        # Ten physical counts land after the calibrated 10 ms delay and move a
        # static target's screen error by exactly one pixel.
        accounted.record_physical_input(2 * NS_PER_MS, 10, 0)
        reflected = observation(13 * NS_PER_MS, 19.0)
        compensated = accounted.step(
            13 * NS_PER_MS,
            engaged=True,
            observation=reflected,
        )
        misclassified = unaccounted.step(
            13 * NS_PER_MS,
            engaged=True,
            observation=reflected,
        )

        self.assertTrue(compensated.valid)
        self.assertAlmostEqual(
            compensated.target_velocity_x_pixels_per_second,
            0.0,
            places=9,
        )
        self.assertLess(
            misclassified.target_velocity_x_pixels_per_second,
            -50.0,
        )

    def test_pending_physical_input_withdraws_prediction_without_fighting(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.1, 0.1, 0.010)
        )

        def observation(timestamp_ns: int, error_x: float) -> ScreenErrorObservation:
            return ScreenErrorObservation(
                timestamp_ns,
                error_x,
                0.0,
                error_x,
                0.0,
            )

        controller.step(
            NS_PER_MS,
            engaged=True,
            observation=observation(NS_PER_MS, 20.0),
        )
        controller.step(
            11 * NS_PER_MS,
            engaged=True,
            observation=observation(11 * NS_PER_MS, 20.0),
        )
        controller.record_physical_input(12 * NS_PER_MS, 10, 0)
        pending = controller.step(13 * NS_PER_MS, engaged=True)

        self.assertTrue(pending.valid)
        self.assertTrue(pending.physical_input_pending_x)
        self.assertFalse(pending.physical_input_pending_y)
        self.assertTrue(pending.predictive_authority_revoked_x)
        self.assertEqual(pending.velocity_feedforward_confidence_x, 0.0)
        self.assertEqual(pending.rate_x_counts_per_second, 0.0)
        self.assertAlmostEqual(pending.projected_error_x_pixels, 19.0)

        reflected = controller.step(
            23 * NS_PER_MS,
            engaged=True,
            observation=observation(23 * NS_PER_MS, 19.0),
        )
        self.assertFalse(reflected.physical_input_pending_x)
        self.assertAlmostEqual(
            reflected.target_velocity_x_pixels_per_second,
            0.0,
            places=9,
        )

    def test_physical_input_history_is_strict_and_cleared_by_full_reset(self) -> None:
        controller = MakcuCalibratedController(
            CalibratedPlant(0.1, 0.1, 0.010)
        )
        controller.record_physical_input(NS_PER_MS, 1, -1)
        with self.assertRaisesRegex(
            ValueError,
            "non-monotonic-physical-input-history",
        ):
            controller.record_physical_input(NS_PER_MS, 1, 0)

        controller.reset()
        controller.record_physical_input(NS_PER_MS, -1, 1)


if __name__ == "__main__":
    unittest.main()
