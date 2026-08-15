from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

from aiming.controller import TargetTracker, head_target_point
from aiming.makcu_calibrated_control import (
    CalibratedControlConfig,
    CalibratedControlOutput,
    CalibratedPlant,
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
    steady_abs_counts_per_second: float
    maximum_requested_axis_rate: float


def _run_tracked_jitter_plant(
    controller: MakcuCalibratedController,
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
            error_x -= 0.125 * delta_x
            error_y -= 0.120 * delta_y

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
        steady_abs_counts / 0.7,
        maximum_requested_axis_rate,
    )


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


class CalibratedControlUnitTests(unittest.TestCase):
    def test_default_calibrated_envelope_has_no_hidden_vertical_ratio(self) -> None:
        config = CalibratedControlConfig()
        self.assertEqual(
            config.maximum_rate_y_counts_per_second,
            config.maximum_rate_x_counts_per_second,
        )
        self.assertEqual(config.maximum_rate_y_counts_per_second, 19_200.0)

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

    def test_automatic_numeric_lease_expires_after_sixty_five_ms(self) -> None:
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
            base_ns + 73 * NS_PER_MS,
            engaged=True,
        )
        self.assertTrue(at_boundary.valid)
        expired = controller.step(
            base_ns + 73 * NS_PER_MS + 1,
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


class CalibratedControlPlantTests(unittest.TestCase):
    def test_automatic_velocity_damping_rejects_live_scale_box_jitter(self) -> None:
        from main import _automatic_plant_aware_controller

        automatic = _run_tracked_jitter_plant(
            _automatic_plant_aware_controller(max_step=320),
        )
        short_filter = _run_tracked_jitter_plant(
            MakcuCalibratedController(
                CalibratedPlant(0.125, 0.120, 0.008),
                CalibratedControlConfig(
                    maximum_rate_x_counts_per_second=19_200.0,
                    maximum_rate_y_counts_per_second=19_200.0,
                ),
            ),
        )

        # The final 700 ms are a stationary target. Do not turn small box
        # geometry changes into a near-continuous stream of opposing counts.
        self.assertLess(
            automatic.steady_abs_counts_per_second,
            short_filter.steady_abs_counts_per_second * 0.45,
        )
        self.assertLess(automatic.steady_abs_counts_per_second, 850.0)
        automatic_stationary = automatic.radial_errors[3300:]
        short_filter_stationary = short_filter.radial_errors[3300:]
        self.assertLess(
            _rms(list(automatic_stationary)),
            _rms(list(short_filter_stationary)),
        )
        self.assertLess(_rms(list(automatic_stationary)), 4.0)

        # Damping still has to pursue both high-speed directions rather than
        # buying quiet output by falling materially behind the moving target.
        automatic_moving = automatic.radial_errors[300:3200]
        short_filter_moving = short_filter.radial_errors[300:3200]
        self.assertLess(
            _rms(list(automatic_moving)),
            _rms(list(short_filter_moving)) * 1.15,
        )
        self.assertLess(_rms(list(automatic_moving)), 10.0)

        # Aggregate pursuit metrics can hide a sluggish response immediately
        # after the target reverses. Keep the first 300 ms of the second
        # pursuit bounded as well as the longer moving interval above.
        automatic_reversal = automatic.radial_errors[2000:2300]
        short_filter_reversal = short_filter.radial_errors[2000:2300]
        self.assertLess(
            _rms(list(automatic_reversal)),
            _rms(list(short_filter_reversal)) * 1.30,
        )
        self.assertLess(_rms(list(automatic_reversal)), 18.0)
        self.assertLess(max(automatic_reversal), 35.0)
        self.assertLess(
            automatic.maximum_requested_axis_rate,
            short_filter.maximum_requested_axis_rate * 0.85,
        )
        self.assertLessEqual(automatic.maximum_requested_axis_rate, 19_200.0)

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


if __name__ == "__main__":
    unittest.main()
