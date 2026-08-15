from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from aiming.controller import TargetTracker
from aiming.makcu import (
    CALIBRATION_LEASE_MAX_AGE_SECONDS,
    CALIBRATION_MAX_EXCURSION_COUNTS,
    CALIBRATION_MAX_RATE_COUNTS_PER_SECOND,
    CALIBRATION_MAX_SESSION_ABS_COUNTS,
    MAX_CONTINUOUS_ACTIVATION_SECONDS,
    MAX_VERTICAL_RATE_RATIO,
    MAKCU_BAUD_CHANGE,
    MAKCU_FAST_BAUD,
    MAKCU_PRODUCT_ID,
    MAKCU_VENDOR_ID,
    PURSUIT_DEADZONE_LEAK_TIME_SECONDS,
    TARGET_STALE_SECONDS,
    MakcuError,
    MakcuAimConfig,
    MakcuAimingController,
    MakcuCalibrationSnapshot,
    MakcuTelemetrySnapshot,
    _ButtonStreamParser,
    detect_makcu_port,
    makcu_target_delta,
)
from aiming.makcu_calibration import MIN_EXCURSION_PIXELS, MIN_PULSES_PER_POLARITY
from aiming.makcu_calibration_session import (
    CalibrationObservation,
    CalibrationSessionState,
    MakcuCalibrationSession,
)
from aiming.makcu_calibrated_control import (
    CalibratedControlConfig,
    CalibratedControlOutput,
    CalibratedPlant,
    MakcuCalibratedController,
    ScreenErrorObservation,
)
from detection.types import Detection


@dataclass(frozen=True)
class FakePort:
    device: str
    vid: int | None
    pid: int | None


class FakeSerial:
    def __init__(self, state: dict[str, int], **options) -> None:
        self.state = state
        self.port = options["port"]
        self._baudrate = options["baudrate"]
        self.responses = bytearray()
        self.writes: list[bytes] = []
        self.closed = False

    @property
    def baudrate(self) -> int:
        return self._baudrate

    @baudrate.setter
    def baudrate(self, value: int) -> None:
        self._baudrate = value

    @property
    def in_waiting(self) -> int:
        return len(self.responses)

    def reset_input_buffer(self) -> None:
        self.responses.clear()

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        if payload == MAKCU_BAUD_CHANGE and self._baudrate == 115_200:
            self.state["device_baud"] = MAKCU_FAST_BAUD
        elif payload == b"km.version()\r" and self._baudrate == self.state["device_baud"]:
            self.responses.extend(b"km.MAKCU-v1\r\n")
        return len(payload)

    def flush(self) -> None:
        pass

    def read(self, count: int) -> bytes:
        result = bytes(self.responses[:count])
        del self.responses[:count]
        return result

    def close(self) -> None:
        self.closed = True


class FakeSerialFactory:
    def __init__(self) -> None:
        self.state = {"device_baud": 115_200}
        self.connections: list[FakeSerial] = []

    def __call__(self, **options) -> FakeSerial:
        connection = FakeSerial(self.state, **options)
        self.connections.append(connection)
        return connection


class StalledReadSerial(FakeSerial):
    """Model a driver blocked in ``in_waiting`` while holding _serial_lock."""

    def __init__(
        self,
        state: dict[str, int],
        *,
        unblock_on_cancel: bool,
        **options,
    ) -> None:
        super().__init__(state, **options)
        self.unblock_on_cancel = unblock_on_cancel
        self.read_entered = threading.Event()
        self.release_read = threading.Event()
        self.cancel_read_calls = 0
        self.cancel_write_calls = 0

    @property
    def in_waiting(self) -> int:
        if b"km.buttons(1)\r" in self.writes:
            self.read_entered.set()
            self.release_read.wait(2.0)
        return len(self.responses)

    def cancel_read(self) -> None:
        self.cancel_read_calls += 1
        if self.unblock_on_cancel:
            self.release_read.set()

    def cancel_write(self) -> None:
        self.cancel_write_calls += 1

    def close(self) -> None:
        if self.unblock_on_cancel:
            self.release_read.set()
        super().close()


class StalledReadSerialFactory(FakeSerialFactory):
    def __init__(self, *, unblock_on_cancel: bool) -> None:
        super().__init__()
        self.unblock_on_cancel = unblock_on_cancel
        self.connections: list[StalledReadSerial] = []

    def __call__(self, **options) -> StalledReadSerial:
        connection = StalledReadSerial(
            self.state,
            unblock_on_cancel=self.unblock_on_cancel,
            **options,
        )
        self.connections.append(connection)
        return connection


class StalledCloseSerial(FakeSerial):
    """Model a native close that does not honor the configured I/O timeout."""

    def __init__(self, state: dict[str, int], **options) -> None:
        super().__init__(state, **options)
        self.close_entered = threading.Event()
        self.release_close = threading.Event()

    def close(self) -> None:
        if b"km.buttons(1)\r" in self.writes:
            self.close_entered.set()
            self.release_close.wait(2.0)
        super().close()


class StalledCloseSerialFactory(FakeSerialFactory):
    def __init__(self) -> None:
        super().__init__()
        self.connections: list[StalledCloseSerial] = []

    def __call__(self, **options) -> StalledCloseSerial:
        connection = StalledCloseSerial(self.state, **options)
        self.connections.append(connection)
        return connection


class MakcuAimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = FakeSerialFactory()
        self.port = FakePort("/dev/ttyACM0", MAKCU_VENDOR_ID, MAKCU_PRODUCT_ID)

    def controller(self, config: MakcuAimConfig | None = None) -> MakcuAimingController:
        return MakcuAimingController(
            config,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )

    def _start_test_calibration(
        self,
        config: MakcuAimConfig | None = None,
    ) -> tuple[MakcuAimingController, FakeSerial, object]:
        """Install a deterministic stand-in for the live worker used by a tick test."""

        controller = self.controller(config or MakcuAimConfig(output_hz=1000))
        controller.start(output_loop=False)
        worker = mock.Mock()
        worker.is_alive.return_value = True
        controller._output_thread = worker
        token = controller.enter_calibration_mode()
        return controller, self.factory.connections[-1], token

    @staticmethod
    def _movement_writes(connection: FakeSerial) -> tuple[bytes, ...]:
        return tuple(
            payload for payload in connection.writes if payload.startswith(b"km.move(")
        )

    @staticmethod
    def _queue_button_event(connection: FakeSerial, mask: int) -> None:
        connection.responses.extend(
            b"km.buttons" + bytes((mask,)) + b"\r\n"
        )

    def _queue_calibration_repress(self, connection: FakeSerial) -> None:
        self._queue_button_event(connection, 0)
        self._queue_button_event(connection, 0b00010)

    def _establish_calibration_hold(
        self,
        controller: MakcuAimingController,
        connection: FakeSerial,
        *,
        press_ns: int,
    ) -> None:
        """Deliver the fresh release and press as two real parser reads."""

        entered_ns = controller.raw_activation_snapshot.calibration_entered_ns
        initial_press_ns = max(press_ns - 2_000_000, entered_ns + 1)
        release_event_ns = max(press_ns - 1_000_000, initial_press_ns + 1)
        press_event_ns = max(press_ns, release_event_ns + 1)
        self._queue_button_event(connection, 0b00010)
        controller.poll_button_mask(now_ns=initial_press_ns)
        self._queue_button_event(connection, 0)
        controller.poll_button_mask(now_ns=release_event_ns)
        self._queue_button_event(connection, 0b00010)
        controller.poll_button_mask(now_ns=press_event_ns)

    @staticmethod
    def _calibration_observation(measurement_ns: int) -> CalibrationObservation:
        return CalibrationObservation(
            measurement_ns=measurement_ns,
            error_x=20.0,
            error_y=-16.0,
            confidence=0.95,
            exact_label=True,
            unique_candidates=1,
            self_safe=True,
            is_prediction=False,
            target_identity="stationary-target-1",
            normalized_bbox=(0.40, 0.20, 0.60, 0.86),
        )

    def _ready_calibrated_adapter(
        self,
        *,
        maximum_command_history: int = 4096,
        plant_delay_seconds: float = 0.0,
    ) -> tuple[
        MakcuAimingController,
        MakcuCalibratedController,
        FakeSerial,
        Detection,
        int,
    ]:
        """Return a deterministic adapter with two confirmed real samples."""

        factory = FakeSerialFactory()
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, plant_delay_seconds),
            CalibratedControlConfig(
                velocity_median_window=1,
                maximum_command_history=maximum_command_history,
            ),
        )
        controller = MakcuAimingController(
            MakcuAimConfig(
                max_step=320,
                output_hz=1000,
                vertical_rate_ratio=1.0,
            ),
            calibrated_controller=calibrated,
            serial_factory=factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        active = factory.connections[-1]
        target = Detection(
            0,
            "player",
            0.95,
            (1400.0, 528.0, 1420.0, 628.0),
        )
        base_ns = 60_000_000_000
        active.responses.extend(bytes((0b00010,)))
        for timestamp_ns in (base_ns, base_ns + 8_000_000):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=timestamp_ns,
            )
            controller._output_tick(0.001, now_ns=timestamp_ns)
        if not calibrated.ready:  # pragma: no cover - test helper invariant
            raise AssertionError("calibrated adapter did not become ready")
        return controller, calibrated, active, target, base_ns + 8_000_000

    def _body_derived_calibrated_adapter(
        self,
    ) -> tuple[
        MakcuAimingController,
        MakcuCalibratedController,
        FakeSerial,
        Detection,
    ]:
        """Return an armed adapter with the bounded mapped-body path enabled."""

        factory = FakeSerialFactory()
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.125, 0.120, 0.008),
            CalibratedControlConfig(
                position_time_constant_seconds=0.012,
                velocity_median_window=1,
                maximum_velocity_feedforward_fraction=0.25,
                require_motion_corroboration_for_feedforward=True,
                maximum_body_derived_projection_fraction=0.25,
                maximum_body_derived_feedforward_fraction=0.25,
                stale_after_seconds=0.065,
                maximum_observation_interval_seconds=0.040,
                feedback_deadzone_pixels=4.0,
            ),
        )
        controller = MakcuAimingController(
            MakcuAimConfig(
                max_step=320,
                output_hz=1000,
                vertical_rate_ratio=1.0,
            ),
            calibrated_controller=calibrated,
            serial_factory=factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        worker = mock.Mock()
        worker.is_alive.return_value = True
        controller._output_thread = worker
        active = factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        target = Detection(
            0,
            "player",
            0.95,
            (850.0, 200.0, 1070.0, 800.0),
        )
        return controller, calibrated, active, target

    def _run_horizontal_fake_plant(
        self,
        velocity_for_tick: Callable[[int], float],
        *,
        ticks: int,
        integral_time_seconds: float = 0.12,
    ) -> tuple[list[float], list[float], list[int], list[float]]:
        """Run the production controller against a deterministic 1-D plant."""

        controller = self.controller(
            MakcuAimConfig(
                strength=1.28,
                max_step=320,
                output_hz=1000,
                deadzone_pixels=2.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        base_ns = 20_000_000_000
        visual_error = 0.0
        plant_gain_pixels_per_count = 0.10
        write_index = len(active.writes)
        errors_before_output: list[float] = []
        errors_after_output: list[float] = []
        emitted_commands: list[int] = []
        measured_errors: list[float] = []
        try:
            with mock.patch(
                "aiming.makcu.PURSUIT_INTEGRAL_TIME_SECONDS",
                integral_time_seconds,
            ):
                for tick in range(ticks):
                    now_ns = base_ns + tick * 1_000_000
                    if tick % 8 == 0:
                        target = Detection(
                            0,
                            "player",
                            0.9,
                            (
                                950.0 + visual_error,
                                540.0,
                                970.0 + visual_error,
                                640.0,
                            ),
                        )
                        controller.update(
                            target,
                            (1080, 1920, 3),
                            measurement_ns=now_ns,
                        )
                    controller._output_tick(0.001, now_ns=now_ns)
                    emitted_x = 0
                    for payload in active.writes[write_index:]:
                        if payload.startswith(b"km.move("):
                            emitted_x += int(
                                payload.decode("ascii")
                                .strip()[8:-1]
                                .split(",")[0]
                            )
                    write_index = len(active.writes)
                    errors_before_output.append(visual_error)
                    emitted_commands.append(emitted_x)
                    measured_errors.append(controller._control_error_x)
                    visual_error += velocity_for_tick(tick) * 0.001
                    visual_error -= plant_gain_pixels_per_count * emitted_x
                    errors_after_output.append(visual_error)
            return (
                errors_before_output,
                errors_after_output,
                emitted_commands,
                measured_errors,
            )
        finally:
            controller._output_thread = None
            controller.stop()

    @staticmethod
    def _legacy_opposing_pursuit_stall(
        error: float,
        base: float,
        accumulated: float,
        *,
        in_deadzone: bool,
    ) -> float:
        """Reproduce the prior zero-output fallback for plant comparisons."""

        correction = base + accumulated
        if not in_deadzone and correction * error < 0.0:
            return 0.0
        return correction

    @staticmethod
    def _maximum_fresh_zero_run(
        commands: list[int],
        measured_errors: list[float],
        *,
        start: int,
        deadzone: float = 2.0,
    ) -> int:
        current = 0
        maximum = 0
        for command, error in zip(commands[start:], measured_errors[start:]):
            if abs(error) > deadzone and command == 0:
                current += 1
                maximum = max(maximum, current)
            else:
                current = 0
        return maximum

    def test_start_discovers_board_switches_to_fast_baud_and_enables_buttons(self) -> None:
        controller = self.controller()

        controller.start()

        self.assertEqual(controller.connected_port, "/dev/ttyACM0")
        self.assertEqual(self.factory.state["device_baud"], MAKCU_FAST_BAUD)
        active = self.factory.connections[-1]
        self.assertEqual(active.baudrate, MAKCU_FAST_BAUD)
        self.assertEqual(active.writes[-1], b"km.buttons(1)\r")
        controller.stop()
        self.assertIn(b"km.buttons(0)\r", active.writes)
        self.assertTrue(active.closed)

    def test_normal_threaded_stop_disables_stream_closes_and_can_restart(self) -> None:
        controller = MakcuAimingController(
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=True,
        )
        controller.start()
        first = self.factory.connections[-1]

        controller.stop()

        self.assertIn(b"km.buttons(0)\r", first.writes)
        self.assertTrue(first.closed)
        self.assertIsNone(controller._output_thread)
        self.assertIsNone(controller._serial)

        controller.start()
        second = self.factory.connections[-1]
        self.assertIsNot(second, first)
        controller.stop()
        self.assertTrue(second.closed)

    def test_stop_cancels_stalled_serial_read_and_joins_worker(self) -> None:
        factory = StalledReadSerialFactory(unblock_on_cancel=True)
        controller = MakcuAimingController(
            serial_factory=factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=True,
            stop_timeout=0.25,
        )
        controller.start()
        active = factory.connections[-1]
        self.assertTrue(active.read_entered.wait(1.0))

        started = time.monotonic()
        controller.stop()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertGreaterEqual(active.cancel_read_calls, 1)
        self.assertGreaterEqual(active.cancel_write_calls, 1)
        self.assertTrue(active.closed)
        self.assertIsNone(controller._output_thread)
        self.assertIsNone(controller._serial)

    def test_stop_times_out_disarmed_and_retains_uncooperative_worker(self) -> None:
        factory = StalledReadSerialFactory(unblock_on_cancel=False)
        controller = MakcuAimingController(
            serial_factory=factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=True,
            stop_timeout=0.12,
        )
        controller.start()
        active = factory.connections[-1]
        self.assertTrue(active.read_entered.wait(1.0))
        # If the blocked read eventually delivers a physical press after stop
        # has begun, the disarmed target state must still prevent movement.
        active.responses.extend(bytes((0b00010,)))
        worker = controller._output_thread
        assert worker is not None
        with controller._state_lock:
            controller._latest_target = Detection(
                0,
                "player",
                0.9,
                (1000, 400, 1100, 700),
            )
            controller._latest_active = True

        started = time.monotonic()
        try:
            with self.assertRaisesRegex(
                MakcuError,
                "shutdown did not finish.*output worker is still running",
            ):
                controller.stop()
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 0.4)
            self.assertIs(controller._output_thread, worker)
            self.assertTrue(worker.is_alive())
            self.assertIs(controller._serial, active)
            self.assertIsNone(controller._latest_target)
            self.assertFalse(controller._latest_active)
            with self.assertRaisesRegex(MakcuError, "shutdown is incomplete"):
                controller.start()
        finally:
            active.release_read.set()
            worker.join(1.0)
            self.assertFalse(
                any(write.startswith(b"km.move(") for write in active.writes)
            )
            controller.stop()

    def test_stop_bounds_stalled_native_close_and_keeps_shutdown_threads(self) -> None:
        factory = StalledCloseSerialFactory()
        controller = MakcuAimingController(
            serial_factory=factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
            stop_timeout=0.12,
        )
        controller.start()
        active = factory.connections[-1]

        started = time.monotonic()
        try:
            with self.assertRaisesRegex(
                MakcuError,
                "shutdown did not finish.*serial cancellation/close is still running",
            ):
                controller.stop()
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 0.4)
            self.assertTrue(active.close_entered.is_set())
            self.assertTrue(any(thread.is_alive() for thread in controller._shutdown_threads))
            self.assertIs(controller._serial, active)
        finally:
            active.release_close.set()
            for thread in controller._shutdown_threads:
                thread.join(1.0)
            controller.stop()

    def test_right_button_gates_bounded_detection_movement(self) -> None:
        controller = self.controller(MakcuAimConfig(strength=0.25, max_step=80))
        target = Detection(0, "player", 0.9, (700, 200, 900, 800))
        controller.start()
        active = self.factory.connections[-1]
        normalized_port = os.path.normcase(os.path.normpath("/dev/ttyACM0"))
        expected_identity = sha256(
            f"proaim-makcu-identity-v1\0{normalized_port}".encode("utf-8")
        ).hexdigest()
        self.assertEqual(controller.identity_token, expected_identity)
        self.assertNotIn("ttyACM0", controller.identity_token or "")

        controller.update(target, (1080, 1920, 3))
        self.assertNotIn(b"km.move(-40,-67)\r", active.writes)
        active.responses.extend(bytes((0b00010,)))
        controller.update(target, (1080, 1920, 3))
        self.assertEqual(active.writes[-1], b"km.move(-40,-67)\r")
        active.responses.extend(bytes((0,)))
        before_release = len(active.writes)
        controller.update(target, (1080, 1920, 3))
        self.assertEqual(len(active.writes), before_release)

    def test_calibrated_identity_mismatch_fails_before_button_stream_or_output(self) -> None:
        calibrated = MakcuCalibratedController(CalibratedPlant(0.10, 0.10, 0.0))
        controller = MakcuAimingController(
            MakcuAimConfig(),
            calibrated_controller=calibrated,
            expected_identity_token="f" * 64,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        with self.assertRaisesRegex(MakcuError, "identity does not match"):
            controller.start()
        self.assertIsNone(controller.identity_token)
        self.assertIsNone(controller.connected_port)
        self.assertIsNone(controller._serial)
        self.assertTrue(all(connection.closed for connection in self.factory.connections))
        self.assertFalse(
            any(
                payload == b"km.buttons(1)\r" or payload.startswith(b"km.move(")
                for connection in self.factory.connections
                for payload in connection.writes
            )
        )

    def test_calibrated_control_uses_real_samples_and_never_falls_back_to_pi(self) -> None:
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.0),
            CalibratedControlConfig(
                velocity_median_window=1,
                maximum_rate_x_counts_per_second=19_200.0,
                maximum_rate_y_counts_per_second=19_200.0,
            ),
        )
        controller = MakcuAimingController(
            MakcuAimConfig(
                strength=4.0,
                max_step=320,
                smoothing_alpha=0.01,
                output_hz=1000,
            ),
            calibrated_controller=calibrated,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        self.assertEqual(controller.control_mode, "calibrated")
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        base_ns = 50_000_000_000
        target = Detection(0, "player", 0.95, (1000.0, 528.0, 1020.0, 628.0))
        try:
            active.responses.extend(bytes((0b00010,)))
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=base_ns,
            )
            controller._output_tick(0.001, now_ns=base_ns)
            self.assertEqual(self._movement_writes(active), ())

            second_ns = base_ns + 8_000_000
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=second_ns,
            )
            controller._output_tick(0.001, now_ns=second_ns)
            writes_after_confirmation = self._movement_writes(active)
            self.assertEqual(len(writes_after_confirmation), 1)
            self.assertTrue(controller.calibrated_control_output.valid)
            self.assertTrue(calibrated.ready)
            observation = calibrated._last_observation
            assert observation is not None
            self.assertIsNone(observation.velocity_error_x_pixels)
            self.assertIsNone(observation.velocity_error_y_pixels)
            telemetry = controller.telemetry_snapshot()
            self.assertEqual(telemetry.control_samples, 1)
            self.assertGreater(telemetry.control_error_abs_x, 0.0)
            self.assertEqual(telemetry.control_error_abs_y, 0.0)
            self.assertEqual(telemetry.pursuit_abs_x, 0.0)
            self.assertEqual(telemetry.pursuit_abs_y, 0.0)
            self.assertEqual(controller._pursuit_correction_x, 0.0)
            self.assertEqual(controller._pursuit_correction_y, 0.0)
            self.assertEqual(controller._smoothed_rate_x, 0.0)
            self.assertEqual(controller._smoothed_rate_y, 0.0)

            predicted_ns = second_ns + 8_000_000
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=predicted_ns,
                measurement_observed=False,
            )
            controller._output_tick(0.001, now_ns=predicted_ns)
            self.assertTrue(calibrated.ready)
            self.assertEqual(
                controller.telemetry_snapshot().control_samples,
                telemetry.control_samples,
            )

            lost_ns = predicted_ns + 8_000_000
            controller.update(
                None,
                (1080, 1920, 3),
                measurement_ns=lost_ns,
                measurement_observed=True,
            )
            movement_count = len(self._movement_writes(active))
            controller._output_tick(0.001, now_ns=lost_ns)
            self.assertFalse(calibrated.ready)
            self.assertEqual(len(self._movement_writes(active)), movement_count)
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(controller._fractional_y, 0.0)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_calibrated_control_keeps_raw_velocity_point_separate_from_position(
        self,
    ) -> None:
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.0),
            CalibratedControlConfig(velocity_median_window=1),
        )
        controller = MakcuAimingController(
            MakcuAimConfig(
                head_ratio=0.0,
                invert_x=True,
                invert_y=True,
                output_hz=1000,
            ),
            calibrated_controller=calibrated,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        smoothed = Detection(
            0,
            "player",
            0.95,
            (950.0, 400.0, 970.0, 500.0),
        )
        raw = Detection(
            0,
            "player",
            0.95,
            (990.0, 420.0, 1010.0, 520.0),
        )
        measurement_ns = 51_000_000_000
        try:
            active.responses.extend(bytes((0b00010,)))
            controller.update(
                smoothed,
                (1080, 1920, 3),
                measurement_ns=measurement_ns,
                velocity_target=raw,
            )
            controller._output_tick(0.001, now_ns=measurement_ns)

            observation = calibrated._last_observation
            assert observation is not None
            # Position/safety retains the tracker-smoothed target while only
            # the velocity estimator receives the exact accepted point.
            self.assertEqual(observation.error_x_pixels, 0.0)
            self.assertEqual(observation.error_y_pixels, 140.0)
            self.assertEqual(observation.velocity_error_x_pixels, -40.0)
            self.assertEqual(observation.velocity_error_y_pixels, 120.0)
            self.assertIs(controller._latest_target, smoothed)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_calibrated_explicit_aim_point_never_falls_back_to_body_box(self) -> None:
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.0),
            CalibratedControlConfig(velocity_median_window=1),
        )
        controller = MakcuAimingController(
            MakcuAimConfig(
                head_ratio=0.0,
                invert_x=True,
                invert_y=True,
                output_hz=1000,
            ),
            calibrated_controller=calibrated,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        # The safety identity's historical box aim point is far down-right.
        # The face landmark is centered, so any nonzero observation would be a
        # body-box fallback rather than the explicitly published evidence.
        safety_target = Detection(
            0,
            "player",
            0.95,
            (1050.0, 600.0, 1250.0, 1000.0),
        )
        measurement_ns = 51_500_000_000
        try:
            controller.update(
                safety_target,
                (720, 1280, 3),
                measurement_ns=measurement_ns,
                aim_point=(640.0, 360.0),
            )
            controller._output_tick(0.001, now_ns=measurement_ns)

            observation = calibrated._last_observation
            assert observation is not None
            self.assertEqual(observation.error_x_pixels, 0.0)
            self.assertEqual(observation.error_y_pixels, 0.0)
            # One exact point feeds both channels, selecting the command-aware
            # paired observer without deriving anything from the body box.
            self.assertEqual(observation.velocity_error_x_pixels, 0.0)
            self.assertEqual(observation.velocity_error_y_pixels, 0.0)
            self.assertIs(controller._latest_target, safety_target)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_calibrated_explicit_position_and_velocity_points_share_mapping(self) -> None:
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.0),
            CalibratedControlConfig(velocity_median_window=1),
        )
        controller = MakcuAimingController(
            MakcuAimConfig(invert_x=True, invert_y=True, output_hz=1000),
            calibrated_controller=calibrated,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        safety_target = Detection(0, "player", 0.95, (50.0, 50.0, 90.0, 300.0))
        measurement_ns = 51_600_000_000
        try:
            controller.update(
                safety_target,
                (720, 1280, 3),
                measurement_ns=measurement_ns,
                aim_point=(600.0, 340.0),
                velocity_point=(800.0, 400.0),
                motion_corroboration_point=(700.0, 300.0),
            )
            controller._output_tick(0.001, now_ns=measurement_ns)

            observation = calibrated._last_observation
            assert observation is not None
            self.assertEqual(observation.error_x_pixels, 60.0)
            self.assertEqual(observation.error_y_pixels, 30.0)
            self.assertEqual(observation.velocity_error_x_pixels, -240.0)
            self.assertEqual(observation.velocity_error_y_pixels, -60.0)
            self.assertEqual(observation.corroboration_error_x_pixels, -90.0)
            self.assertEqual(observation.corroboration_error_y_pixels, 90.0)

            controller.update(
                safety_target,
                (720, 1280, 3),
                measurement_ns=measurement_ns + 8_000_000,
                aim_point=(600.0, 340.0),
                velocity_point=(800.0, 400.0),
            )
            controller._output_tick(
                0.008,
                now_ns=measurement_ns + 8_000_000,
            )
            next_observation = calibrated._last_observation
            assert next_observation is not None
            self.assertIsNone(next_observation.corroboration_error_x_pixels)
            self.assertIsNone(next_observation.corroboration_error_y_pixels)

            controller.update(
                None,
                (720, 1280, 3),
                measurement_ns=measurement_ns + 16_000_000,
            )
            self.assertIsNone(controller._latest_motion_corroboration_error)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_body_derived_permission_validates_and_is_per_sample(self) -> None:
        controller, calibrated, _active, target = (
            self._body_derived_calibrated_adapter()
        )
        frame_shape = (1080, 1920, 3)
        base_ns = 51_625_000_000
        try:
            original_sample_id = controller._latest_sample_id
            invalid_calls = (
                (
                    {"body_derived_motion_permitted": 1},
                    TypeError,
                    "must be bool",
                ),
                (
                    {"body_derived_motion_permitted": True},
                    ValueError,
                    "requires an aim point with a real observed target",
                ),
                (
                    {
                        "aim_point": (960.0, 540.0),
                        "motion_corroboration_point": (1060.0, 740.0),
                        "body_derived_motion_permitted": True,
                    },
                    ValueError,
                    "cannot accompany independent motion corroboration",
                ),
                (
                    {
                        "measurement_observed": False,
                        "aim_point": (960.0, 540.0),
                        "body_derived_motion_permitted": True,
                    },
                    ValueError,
                    "requires an aim point with a real observed target",
                ),
                (
                    {
                        "aim_point": (960.0, 540.0),
                        "body_derived_motion_permitted": True,
                    },
                    ValueError,
                    "requires an immutable deadline",
                ),
                (
                    {"body_derived_motion_deadline_ns": base_ns + 1},
                    ValueError,
                    "requires motion permission",
                ),
                (
                    {
                        "aim_point": (960.0, 540.0),
                        "body_derived_motion_permitted": True,
                        "body_derived_motion_deadline_ns": True,
                    },
                    TypeError,
                    "must be an integer",
                ),
                (
                    {
                        "aim_point": (960.0, 540.0),
                        "body_derived_motion_permitted": True,
                        "body_derived_motion_deadline_ns": base_ns,
                    },
                    ValueError,
                    "must be after measurement_ns",
                ),
                (
                    {"identity_deadline_ns": True},
                    TypeError,
                    "must be an integer",
                ),
                (
                    {"identity_deadline_ns": base_ns},
                    ValueError,
                    "must be after measurement_ns",
                ),
                (
                    {
                        "aim_point": (960.0, 540.0),
                        "body_derived_motion_permitted": True,
                        "body_derived_motion_deadline_ns": base_ns + 10,
                        "identity_deadline_ns": base_ns + 9,
                    },
                    ValueError,
                    "cannot precede",
                ),
            )
            for kwargs, exception, message in invalid_calls:
                with (
                    self.subTest(kwargs=kwargs),
                    self.assertRaisesRegex(exception, message),
                ):
                    controller.update(
                        target,
                        frame_shape,
                        measurement_ns=base_ns,
                        **kwargs,
                    )
            self.assertEqual(controller._latest_sample_id, original_sample_id)
            self.assertFalse(controller._latest_body_derived_motion_permitted)

            motion_deadline_ns = base_ns + 64_000_000
            identity_deadline_ns = base_ns + 160_000_000
            for index, point_x in enumerate((956.0, 960.0)):
                timestamp_ns = base_ns + index * 8_000_000
                controller.update(
                    target,
                    frame_shape,
                    measurement_ns=timestamp_ns,
                    aim_point=(point_x, 540.0),
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=motion_deadline_ns,
                    identity_deadline_ns=identity_deadline_ns,
                )
                self.assertTrue(
                    controller._latest_body_derived_motion_permitted
                )
                controller._output_tick(0.0, now_ns=timestamp_ns)

            observation = calibrated._last_observation
            assert observation is not None
            self.assertTrue(observation.body_derived_motion_permitted)
            self.assertEqual(
                observation.body_derived_motion_deadline_ns,
                motion_deadline_ns,
            )
            self.assertEqual(
                observation.identity_deadline_ns,
                identity_deadline_ns,
            )
            self.assertTrue(calibrated._body_derived_motion_permitted)
            self.assertIsNone(observation.corroboration_error_x_pixels)
            self.assertIsNone(observation.corroboration_error_y_pixels)

            # The default on the immediately following real sample is an
            # explicit revoke, not inheritance from the prior publication.
            next_ns = base_ns + 16_000_000
            controller.update(
                target,
                frame_shape,
                measurement_ns=next_ns,
                aim_point=(960.0, 540.0),
            )
            self.assertFalse(controller._latest_body_derived_motion_permitted)
            self.assertIsNone(
                controller._latest_body_derived_motion_deadline_ns
            )
            self.assertIsNone(controller._latest_identity_deadline_ns)
            controller._output_tick(0.0, now_ns=next_ns)
            next_observation = calibrated._last_observation
            assert next_observation is not None
            self.assertFalse(next_observation.body_derived_motion_permitted)
            self.assertFalse(calibrated._body_derived_motion_permitted)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_body_derived_revoke_discards_queued_permission_and_fraction(self) -> None:
        controller, calibrated, active, target = (
            self._body_derived_calibrated_adapter()
        )
        base_ns = 51_640_000_000
        motion_deadline_ns = base_ns + 400_000_000
        established_identity_deadline_ns = base_ns + 700_000_000
        queued_identity_deadline_ns = base_ns + 500_000_000
        try:
            for index in range(40):
                timestamp_ns = base_ns + index * 8_000_000
                point_x = 960.0 + (index - 39) * 4.6
                controller.update(
                    target,
                    (1080, 1920, 3),
                    measurement_ns=timestamp_ns,
                    aim_point=(point_x, 540.0),
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=motion_deadline_ns,
                    identity_deadline_ns=established_identity_deadline_ns,
                )
                controller._output_tick(0.0, now_ns=timestamp_ns)

            self.assertTrue(calibrated.ready)
            self.assertTrue(calibrated._body_derived_motion_permitted)
            accepted_timestamp_ns = calibrated._last_observation.timestamp_ns
            queued_ns = base_ns + 313_000_000
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=queued_ns,
                aim_point=(960.575, 540.0),
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=motion_deadline_ns,
                identity_deadline_ns=queued_identity_deadline_ns,
            )
            self.assertNotEqual(
                controller._calibrated_processed_sample_id,
                controller._latest_sample_id,
            )
            controller._fractional_x = 0.75
            controller._fractional_y = -0.75

            controller.revoke_body_derived_motion()

            self.assertFalse(controller._latest_body_derived_motion_permitted)
            self.assertIsNone(
                controller._latest_body_derived_motion_deadline_ns
            )
            self.assertFalse(calibrated._body_derived_motion_permitted)
            self.assertEqual(
                controller._calibrated_processed_sample_id,
                controller._latest_sample_id,
            )
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(controller._fractional_y, 0.0)
            controller._output_tick(0.0, now_ns=queued_ns)
            self.assertEqual(
                calibrated._last_observation.timestamp_ns,
                accepted_timestamp_ns,
            )
            output = controller.calibrated_control_output
            assert output is not None
            self.assertEqual(output.velocity_feedforward_confidence_x, 0.0)
            self.assertEqual(output.velocity_feedforward_confidence_y, 0.0)
            self.assertEqual(self._movement_writes(active), ())

            # The queued observation was deliberately consumed by the narrow
            # revoke, so the core still owns the older, later deadline. The
            # wrapper's atomic sample deadline must nevertheless zero output
            # at the queued identity boundary.
            controller._fractional_x = 0.75
            controller._fractional_y = -0.75
            movement_before = self._movement_writes(active)
            controller._output_tick(
                0.0,
                now_ns=queued_identity_deadline_ns,
            )
            expired = controller.calibrated_control_output
            assert expired is not None
            self.assertFalse(expired.valid)
            self.assertEqual(expired.reset_reason, "identity-expired")
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(controller._fractional_y, 0.0)
            self.assertEqual(self._movement_writes(active), movement_before)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_body_derived_permission_revokes_on_release_loss_stale_and_reset(
        self,
    ) -> None:
        for boundary in ("release-repress", "loss", "stale", "reset"):
            with self.subTest(boundary=boundary):
                controller, calibrated, active, target = (
                    self._body_derived_calibrated_adapter()
                )
                base_ns = 51_645_000_000
                motion_deadline_ns = base_ns + 64_000_000
                identity_deadline_ns = base_ns + 128_000_000
                try:
                    for index in range(2):
                        timestamp_ns = base_ns + index * 8_000_000
                        controller.update(
                            target,
                            (1080, 1920, 3),
                            measurement_ns=timestamp_ns,
                            aim_point=(960.0, 540.0),
                            body_derived_motion_permitted=True,
                            body_derived_motion_deadline_ns=(
                                motion_deadline_ns
                            ),
                            identity_deadline_ns=identity_deadline_ns,
                        )
                        controller._output_tick(0.0, now_ns=timestamp_ns)
                    self.assertTrue(calibrated._body_derived_motion_permitted)
                    controller._fractional_x = 0.75
                    controller._fractional_y = -0.75

                    if boundary == "release-repress":
                        # The final mask is pressed; the intermediate release
                        # must nevertheless revoke the old sample.
                        self._queue_button_event(active, 0)
                        self._queue_button_event(active, 0b00010)
                        controller._output_tick(
                            0.0,
                            now_ns=base_ns + 9_000_000,
                        )
                    elif boundary == "loss":
                        loss_ns = base_ns + 16_000_000
                        controller.update(
                            None,
                            (1080, 1920, 3),
                            measurement_ns=loss_ns,
                        )
                        controller._output_tick(0.0, now_ns=loss_ns)
                    elif boundary == "stale":
                        controller._output_tick(
                            0.0,
                            now_ns=base_ns + 74_000_000,
                        )
                    else:
                        token = controller.enter_calibration_mode()
                        self.assertIsNotNone(token)

                    self.assertFalse(
                        controller._latest_body_derived_motion_permitted
                    )
                    self.assertFalse(calibrated._body_derived_motion_permitted)
                    self.assertIsNone(
                        controller._latest_body_derived_motion_deadline_ns
                    )
                    self.assertIsNone(controller._latest_identity_deadline_ns)
                    self.assertEqual(controller._fractional_x, 0.0)
                    self.assertEqual(controller._fractional_y, 0.0)
                finally:
                    controller._output_thread = None
                    controller.stop()

    def test_body_and_identity_deadlines_bound_capture_starvation(self) -> None:
        controller, calibrated, active, target = (
            self._body_derived_calibrated_adapter()
        )
        base_ns = 51_647_000_000
        motion_deadline_ns = base_ns + 65_000_000
        identity_deadline_ns = base_ns + 200_000_000
        try:
            # Model a newly mapped body sample captured when its immutable
            # direct-motion lease has only one millisecond remaining. Its
            # ordinary 65 ms observation freshness would otherwise survive to
            # direct age 129 ms.
            for source_offset_ns in (56_000_000, 64_000_000):
                source_ns = base_ns + source_offset_ns
                controller.update(
                    target,
                    (1080, 1920, 3),
                    measurement_ns=source_ns,
                    aim_point=(960.0, 540.0),
                    body_derived_motion_permitted=True,
                    body_derived_motion_deadline_ns=motion_deadline_ns,
                    identity_deadline_ns=identity_deadline_ns,
                )
                with mock.patch(
                    "aiming.makcu.time.perf_counter_ns",
                    side_effect=(100, 100),
                ):
                    controller._output_tick(0.0, now_ns=source_ns)
            self.assertTrue(calibrated.ready)
            self.assertTrue(calibrated._body_derived_motion_permitted)

            controller._fractional_x = 0.75
            controller._fractional_y = -0.75
            just_before_ns = motion_deadline_ns - 1
            with mock.patch(
                "aiming.makcu.time.perf_counter_ns",
                side_effect=(100, 100),
            ):
                controller._output_tick(0.0, now_ns=just_before_ns)
            self.assertTrue(calibrated._body_derived_motion_permitted)
            self.assertEqual(controller._fractional_x, 0.75)

            movement_before = self._movement_writes(active)
            controller._output_tick(0.0, now_ns=motion_deadline_ns)
            self.assertTrue(calibrated.ready)
            self.assertFalse(calibrated._body_derived_motion_permitted)
            self.assertFalse(controller._latest_body_derived_motion_permitted)
            self.assertIsNone(
                controller._latest_body_derived_motion_deadline_ns
            )
            self.assertEqual(
                controller._latest_identity_deadline_ns,
                identity_deadline_ns,
            )
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(controller._fractional_y, 0.0)
            self.assertEqual(self._movement_writes(active), movement_before)

            # Static position remains usable inside the independent freshness
            # and identity leases after predictive permission expires.
            controller._output_tick(0.0, now_ns=base_ns + 100_000_000)
            output = controller.calibrated_control_output
            assert output is not None
            self.assertTrue(output.valid)
            self.assertTrue(calibrated.ready)

            # Refresh only static identity-bound position immediately before
            # the overall identity deadline, then starve capture again.
            for source_offset_ns in (191_000_000, 199_000_000):
                source_ns = base_ns + source_offset_ns
                controller.update(
                    target,
                    (1080, 1920, 3),
                    measurement_ns=source_ns,
                    aim_point=(960.0, 540.0),
                    identity_deadline_ns=identity_deadline_ns,
                )
                with mock.patch(
                    "aiming.makcu.time.perf_counter_ns",
                    side_effect=(100, 100),
                ):
                    controller._output_tick(0.0, now_ns=source_ns)
            controller._fractional_x = 0.75
            controller._fractional_y = -0.75
            with mock.patch(
                "aiming.makcu.time.perf_counter_ns",
                side_effect=(100, 100),
            ):
                controller._output_tick(0.0, now_ns=identity_deadline_ns - 1)
            self.assertTrue(controller.calibrated_control_output.valid)

            movement_before = self._movement_writes(active)
            controller._output_tick(0.0, now_ns=identity_deadline_ns)
            expired = controller.calibrated_control_output
            assert expired is not None
            self.assertFalse(expired.valid)
            self.assertEqual(expired.reset_reason, "identity-expired")
            self.assertFalse(calibrated.ready)
            self.assertIsNone(controller._latest_identity_deadline_ns)
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(controller._fractional_y, 0.0)
            self.assertEqual(self._movement_writes(active), movement_before)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_queued_body_sample_cannot_replay_permission_after_deadline(self) -> None:
        controller, calibrated, active, target = (
            self._body_derived_calibrated_adapter()
        )
        base_ns = 51_648_000_000
        motion_deadline_ns = base_ns + 65_000_000
        identity_deadline_ns = base_ns + 200_000_000
        try:
            first_ns = base_ns + 56_000_000
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=first_ns,
                aim_point=(960.0, 540.0),
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=motion_deadline_ns,
                identity_deadline_ns=identity_deadline_ns,
            )
            controller._output_tick(0.0, now_ns=first_ns)

            queued_ns = base_ns + 64_000_000
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=queued_ns,
                aim_point=(960.0, 540.0),
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=motion_deadline_ns,
                identity_deadline_ns=identity_deadline_ns,
            )
            self.assertNotEqual(
                controller._calibrated_processed_sample_id,
                controller._latest_sample_id,
            )
            controller._fractional_x = 0.75
            movement_before = self._movement_writes(active)

            controller._output_tick(0.0, now_ns=motion_deadline_ns)

            observation = calibrated._last_observation
            assert observation is not None
            self.assertEqual(observation.timestamp_ns, queued_ns)
            self.assertFalse(observation.body_derived_motion_permitted)
            self.assertIsNone(observation.body_derived_motion_deadline_ns)
            self.assertFalse(calibrated._body_derived_motion_permitted)
            self.assertFalse(controller._latest_body_derived_motion_permitted)
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(self._movement_writes(active), movement_before)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_deadlines_are_rechecked_at_physical_commit(self) -> None:
        for boundary in ("motion", "identity", "base"):
            with self.subTest(boundary=boundary):
                controller, calibrated, active, target = (
                    self._body_derived_calibrated_adapter()
                )
                base_ns = 51_649_000_000
                motion_deadline_ns = base_ns + 10_000_000
                identity_deadline_ns = base_ns + 100_000_000
                try:
                    for source_offset_ns in (0, 8_000_000):
                        source_ns = base_ns + source_offset_ns
                        kwargs = {
                            "identity_deadline_ns": identity_deadline_ns,
                        }
                        if boundary == "motion":
                            kwargs.update(
                                body_derived_motion_permitted=True,
                                body_derived_motion_deadline_ns=(
                                    motion_deadline_ns
                                ),
                            )
                        controller.update(
                            target,
                            (1080, 1920, 3),
                            measurement_ns=source_ns,
                            aim_point=(1000.0, 540.0),
                            **kwargs,
                        )
                        controller._output_tick(0.0, now_ns=source_ns)
                    self.assertTrue(calibrated.ready)
                    controller._fractional_x = 0.75
                    controller._fractional_y = -0.75
                    movement_before = self._movement_writes(active)

                    if boundary == "motion":
                        decision_ns = motion_deadline_ns - 500_000
                    elif boundary == "identity":
                        # Install a nearer overall deadline on a fresh pair.
                        identity_deadline_ns = base_ns + 12_000_000
                        for source_offset_ns in (10_000_000, 11_000_000):
                            source_ns = base_ns + source_offset_ns
                            controller.update(
                                target,
                                (1080, 1920, 3),
                                measurement_ns=source_ns,
                                aim_point=(1000.0, 540.0),
                                identity_deadline_ns=identity_deadline_ns,
                            )
                            controller._output_tick(0.0, now_ns=source_ns)
                        decision_ns = identity_deadline_ns - 500_000
                    else:
                        decision_ns = base_ns + 9_000_000
                    if boundary == "base":
                        clock_calls = 0

                        def cross_generation_at_commit() -> int:
                            nonlocal clock_calls
                            clock_calls += 1
                            if clock_calls == 2:
                                # Model a safety epoch changing after numeric
                                # computation but before physical commit while
                                # the observation itself remains fresh.
                                controller._normal_motion_generation += 1
                            return 100

                        clock_side_effect = cross_generation_at_commit
                    else:
                        clock_side_effect = (100, 1_000_100)
                    with mock.patch(
                        "aiming.makcu.time.perf_counter_ns",
                        side_effect=clock_side_effect,
                    ):
                        controller._output_tick(0.001, now_ns=decision_ns)

                    self.assertEqual(self._movement_writes(active), movement_before)
                    self.assertEqual(controller._fractional_x, 0.0)
                    self.assertEqual(controller._fractional_y, 0.0)
                    if boundary == "motion":
                        self.assertTrue(calibrated.ready)
                        self.assertFalse(
                            calibrated._body_derived_motion_permitted
                        )
                        self.assertIsNone(
                            controller._latest_body_derived_motion_deadline_ns
                        )
                        self.assertEqual(
                            controller._latest_identity_deadline_ns,
                            identity_deadline_ns,
                        )
                    else:
                        self.assertFalse(calibrated.ready)
                        self.assertIsNone(controller._latest_identity_deadline_ns)
                        self.assertEqual(
                            controller._calibrated_processed_sample_id,
                            controller._latest_sample_id,
                        )
                        # Deadline removal is not a new undated publication.
                        # Repeated worker ticks must not reconstruct the same
                        # observation after the commit-time identity reset.
                        replay_base_ns = (
                            identity_deadline_ns
                            if boundary == "identity"
                            else decision_ns
                        )
                        for offset_ns in (1, 1_000_000, 2_000_000):
                            controller._output_tick(
                                0.0,
                                now_ns=replay_base_ns + offset_ns,
                            )
                            replay = controller.calibrated_control_output
                            assert replay is not None
                            self.assertFalse(replay.valid)
                            self.assertEqual(
                                replay.reset_reason,
                                "awaiting-observation",
                            )
                            self.assertIsNone(calibrated._last_observation)
                            self.assertFalse(calibrated.ready)
                            self.assertEqual(
                                controller._calibrated_processed_sample_id,
                                controller._latest_sample_id,
                            )
                        self.assertEqual(
                            self._movement_writes(active),
                            movement_before,
                        )
                finally:
                    controller._output_thread = None
                    controller.stop()

    def test_corroboration_revoke_cannot_replay_or_leak_prior_feedforward(
        self,
    ) -> None:
        from main import _automatic_plant_aware_controller

        calibrated = _automatic_plant_aware_controller(max_step=200)
        controller = MakcuAimingController(
            MakcuAimConfig(
                max_step=200,
                output_hz=1000,
                vertical_rate_ratio=1.0,
            ),
            calibrated_controller=calibrated,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        safety_target = Detection(
            0,
            "player",
            0.95,
            (850.0, 200.0, 1070.0, 800.0),
        )
        base_ns = 51_650_000_000
        try:
            for index in range(40):
                timestamp_ns = base_ns + index * 8_000_000
                # The final accepted head point is exactly centered while its
                # retained observer velocity is still about 575 px/s. After
                # corroboration loss, that velocity must not sneak through
                # the projected P term as an implicit feed-forward command.
                point_x = 960.0 + (index - 39) * 4.6
                controller.update(
                    safety_target,
                    (1080, 1920, 3),
                    measurement_ns=timestamp_ns,
                    aim_point=(point_x, 540.0),
                    motion_corroboration_point=(point_x + 100.0, 740.0),
                )
                # Zero elapsed prevents the test harness from moving the fake
                # screen while still exercising each real numeric decision.
                controller._output_tick(0.0, now_ns=timestamp_ns)

            before = controller.calibrated_control_output
            assert before is not None
            self.assertTrue(before.valid)
            self.assertGreater(before.velocity_feedforward_confidence_x, 0.90)

            # Queue a newer corroborated sample but revoke it before the 1 kHz
            # owner consumes it. The wrapper must mark that exact sample
            # processed rather than replaying stale permission after unlock.
            queued_ns = base_ns + 313_000_000
            controller.update(
                safety_target,
                (1080, 1920, 3),
                measurement_ns=queued_ns,
                aim_point=(960.575, 540.0),
                motion_corroboration_point=(1060.575, 740.0),
            )
            self.assertNotEqual(
                controller._calibrated_processed_sample_id,
                controller._latest_sample_id,
            )
            controller._fractional_x = 0.75
            controller._fractional_y = -0.75
            # Broad corroboration loss must also revoke the mutually exclusive
            # body-derived permission if adversarial state reaches this
            # boundary.
            controller._latest_body_derived_motion_permitted = True
            controller._latest_body_derived_motion_deadline_ns = queued_ns + 1
            calibrated._body_derived_motion_permitted = True

            controller.revoke_motion_corroboration()

            self.assertIsNone(controller._latest_motion_corroboration_error)
            self.assertFalse(controller._latest_body_derived_motion_permitted)
            self.assertIsNone(
                controller._latest_body_derived_motion_deadline_ns
            )
            self.assertFalse(calibrated._body_derived_motion_permitted)
            self.assertEqual(
                controller._calibrated_processed_sample_id,
                controller._latest_sample_id,
            )
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(controller._fractional_y, 0.0)
            movement_before = self._movement_writes(active)
            controller._output_tick(0.001, now_ns=queued_ns)
            after = controller.calibrated_control_output
            assert after is not None
            self.assertTrue(after.valid)
            self.assertTrue(calibrated.ready)
            self.assertGreater(
                after.target_velocity_x_pixels_per_second,
                570.0,
            )
            self.assertEqual(after.motion_corroboration_confidence, 0.0)
            self.assertEqual(after.velocity_feedforward_confidence_x, 0.0)
            self.assertEqual(after.velocity_feedforward_confidence_y, 0.0)
            self.assertEqual(after.rate_x_counts_per_second, 0.0)
            self.assertEqual(self._movement_writes(active), movement_before)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_corroboration_revoke_is_noop_for_explicit_profile(self) -> None:
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.0),
            CalibratedControlConfig(
                velocity_median_window=1,
                require_motion_corroboration_for_feedforward=False,
            ),
        )
        controller = MakcuAimingController(
            calibrated_controller=calibrated,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller._latest_motion_corroboration_error = (1.0, 2.0)
        controller._latest_body_derived_motion_permitted = True
        controller._latest_body_derived_motion_deadline_ns = 10
        controller._fractional_x = 0.5
        with mock.patch.object(
            calibrated,
            "revoke_motion_corroboration",
        ) as revoke:
            controller.revoke_motion_corroboration()

        revoke.assert_not_called()
        self.assertEqual(
            controller._latest_motion_corroboration_error,
            (1.0, 2.0),
        )
        self.assertEqual(controller._fractional_x, 0.5)
        with mock.patch.object(
            calibrated,
            "revoke_body_derived_motion",
        ) as revoke_body:
            controller.revoke_body_derived_motion()
        revoke_body.assert_not_called()
        self.assertTrue(controller._latest_body_derived_motion_permitted)
        self.assertEqual(controller._latest_body_derived_motion_deadline_ns, 10)
        self.assertEqual(controller._fractional_x, 0.5)

    def test_body_derived_revoke_enabled_by_either_split_cap(self) -> None:
        cap_cases = (
            {
                "maximum_body_derived_projection_fraction": 0.25,
                "maximum_body_derived_feedforward_fraction": 0.0,
            },
            {
                "maximum_body_derived_projection_fraction": 0.0,
                "maximum_body_derived_feedforward_fraction": 0.25,
            },
        )
        for caps in cap_cases:
            with self.subTest(caps=caps):
                calibrated = MakcuCalibratedController(
                    CalibratedPlant(0.10, 0.10, 0.0),
                    CalibratedControlConfig(
                        require_motion_corroboration_for_feedforward=True,
                        **caps,
                    ),
                )
                controller = MakcuAimingController(
                    calibrated_controller=calibrated,
                    serial_factory=self.factory,
                    ports_provider=lambda: (self.port,),
                    sleep=lambda _seconds: None,
                    threaded_output=False,
                )
                controller._latest_body_derived_motion_permitted = True
                controller._latest_body_derived_motion_deadline_ns = 10
                controller._latest_sample_id = 3
                controller._calibrated_processed_sample_id = 2
                controller._fractional_x = 0.5
                with mock.patch.object(
                    calibrated,
                    "revoke_body_derived_motion",
                ) as revoke:
                    controller.revoke_body_derived_motion()

                revoke.assert_called_once_with()
                self.assertFalse(
                    controller._latest_body_derived_motion_permitted
                )
                self.assertIsNone(
                    controller._latest_body_derived_motion_deadline_ns
                )
                self.assertEqual(controller._calibrated_processed_sample_id, 3)
                self.assertEqual(controller._fractional_x, 0.0)

    def test_explicit_points_fail_closed_before_mutating_publication(self) -> None:
        controller = self.controller()
        controller.start(output_loop=False)
        target = Detection(0, "player", 0.9, (950.0, 540.0, 970.0, 640.0))
        controller.update(
            target,
            (1080, 1920, 3),
            measurement_ns=51_700_000_000,
            aim_point=(960.0, 540.0),
        )
        original_sample_id = controller._latest_sample_id
        original_source_ns = controller._latest_source_ns
        original_error = (
            controller._measurement_error_x,
            controller._measurement_error_y,
        )

        invalid_points = (
            ([960.0, 540.0], TypeError, "tuple"),
            ((math.nan, 540.0), ValueError, "finite"),
            ((960.0, math.inf), ValueError, "finite"),
            ((True, 540.0), ValueError, "finite"),
            ((-0.01, 540.0), ValueError, "inside"),
            ((1920.0, 540.0), ValueError, "inside"),
            ((960.0, 1080.0), ValueError, "inside"),
        )
        for point, exception, message in invalid_points:
            with (
                self.subTest(point=point),
                self.assertRaisesRegex(exception, message),
            ):
                controller.update(
                    target,
                    (1080, 1920, 3),
                    measurement_ns=51_800_000_000,
                    aim_point=point,  # type: ignore[arg-type]
                )

        self.assertEqual(controller._latest_sample_id, original_sample_id)
        self.assertEqual(controller._latest_source_ns, original_source_ns)
        self.assertEqual(
            (controller._measurement_error_x, controller._measurement_error_y),
            original_error,
        )

        with self.assertRaisesRegex(ValueError, "requires an aim point"):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=51_800_000_000,
                velocity_point=(960.0, 540.0),
            )
        with self.assertRaisesRegex(ValueError, "requires an aim point"):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=51_800_000_000,
                motion_corroboration_point=(960.0, 540.0),
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=51_800_000_000,
                velocity_target=target,
                aim_point=(960.0, 540.0),
            )
        with self.assertRaisesRegex(ValueError, "inside"):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=51_800_000_000,
                aim_point=(960.0, 540.0),
                velocity_point=(1920.0, 540.0),
            )
        with self.assertRaisesRegex(ValueError, "inside"):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=51_800_000_000,
                aim_point=(960.0, 540.0),
                motion_corroboration_point=(1920.0, 540.0),
            )
        with self.assertRaisesRegex(ValueError, "aim point requires"):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=51_800_000_000,
                measurement_observed=False,
                aim_point=(960.0, 540.0),
                motion_corroboration_point=(960.0, 540.0),
            )

    def test_explicit_point_timestamps_reject_late_frames_and_same_time_motion(
        self,
    ) -> None:
        controller = self.controller(MakcuAimConfig(head_ratio=0.0))
        controller.start(output_loop=False)
        first_target = Detection(0, "player", 0.9, (100.0, 100.0, 200.0, 600.0))
        current_target = Detection(
            0,
            "player",
            0.9,
            (1500.0, 100.0, 1700.0, 600.0),
        )
        late_target = Detection(0, "player", 0.9, (300.0, 100.0, 500.0, 600.0))
        first_ns = 51_850_000_000
        current_ns = first_ns + 100_000_000
        controller.update(
            first_target,
            (1080, 1920, 3),
            measurement_ns=first_ns,
            aim_point=(960.0, 540.0),
        )
        controller.update(
            current_target,
            (1080, 1920, 3),
            measurement_ns=current_ns,
            aim_point=(1060.0, 540.0),
        )
        self.assertEqual(controller._latest_velocity_x, 1000.0)

        with self.assertRaisesRegex(ValueError, "must not move backwards"):
            controller.update(
                late_target,
                (1080, 1920, 3),
                measurement_ns=current_ns - 1,
                aim_point=(980.0, 540.0),
            )
        self.assertIs(controller._latest_target, current_target)
        self.assertEqual(controller._measurement_error_x, 100.0)

        # A conflicting localization for the same source frame is accepted as
        # the current evidence but cannot imply infinite/undefined velocity.
        controller.update(
            current_target,
            (1080, 1920, 3),
            measurement_ns=current_ns,
            aim_point=(1080.0, 540.0),
        )
        self.assertEqual(controller._latest_velocity_x, 0.0)
        self.assertEqual(controller._latest_velocity_y, 0.0)
        self.assertEqual(controller._measurement_error_x, 120.0)

    def test_unobserved_publication_preserves_last_exact_point_without_observation(
        self,
    ) -> None:
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.0),
            CalibratedControlConfig(velocity_median_window=1),
        )
        controller = MakcuAimingController(
            MakcuAimConfig(output_hz=1000),
            calibrated_controller=calibrated,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        first_target = Detection(0, "player", 0.95, (100.0, 100.0, 200.0, 500.0))
        current_target = Detection(
            0,
            "player",
            0.95,
            (1500.0, 400.0, 1800.0, 1000.0),
        )
        source_ns = 51_900_000_000
        try:
            controller.update(
                first_target,
                (1080, 1920, 3),
                measurement_ns=source_ns,
                aim_point=(1000.0, 500.0),
            )
            controller._output_tick(0.001, now_ns=source_ns)
            exact_observation = calibrated._last_observation
            exact_error = (
                controller._measurement_error_x,
                controller._measurement_error_y,
            )

            # A current primary box can refresh safety presence at the exact
            # head frame's timestamp, but it is not a new head observation.
            controller.update(
                current_target,
                (1080, 1920, 3),
                measurement_ns=source_ns,
                measurement_observed=False,
            )
            controller._output_tick(0.001, now_ns=source_ns + 1_000_000)

            self.assertIs(calibrated._last_observation, exact_observation)
            self.assertEqual(
                (controller._measurement_error_x, controller._measurement_error_y),
                exact_error,
            )
            self.assertIs(controller._latest_target, current_target)
            with self.assertRaisesRegex(
                ValueError,
                "requires an observed safety target",
            ):
                controller.update(
                    current_target,
                    (1080, 1920, 3),
                    measurement_ns=source_ns,
                    measurement_observed=False,
                    aim_point=(1700.0, 500.0),
                )
        finally:
            controller._output_thread = None
            controller.stop()

    def test_explicit_head_loss_revokes_a_blocked_calibrated_commit(self) -> None:
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.0),
            CalibratedControlConfig(velocity_median_window=1),
        )
        controller = MakcuAimingController(
            MakcuAimConfig(max_step=320, output_hz=1000),
            calibrated_controller=calibrated,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        safety_target = Detection(
            0,
            "player",
            0.95,
            (1200.0, 400.0, 1500.0, 1000.0),
        )
        source_ns = 52_000_000_000
        controller.update(
            safety_target,
            (1080, 1920, 3),
            measurement_ns=source_ns,
            aim_point=(1200.0, 540.0),
        )

        entered = threading.Event()
        release_step = threading.Event()
        output_errors: list[BaseException] = []

        def blocked_step(*_args, **_kwargs) -> CalibratedControlOutput:
            entered.set()
            if not release_step.wait(1.0):
                raise AssertionError("timed out waiting to revoke the head")
            return CalibratedControlOutput(
                timestamp_ns=source_ns,
                rate_x_counts_per_second=1000.0,
                rate_y_counts_per_second=0.0,
                target_velocity_x_pixels_per_second=0.0,
                target_velocity_y_pixels_per_second=0.0,
                projected_error_x_pixels=240.0,
                projected_error_y_pixels=0.0,
                valid=True,
            )

        def output_tick() -> None:
            try:
                controller._output_tick(0.001, now_ns=source_ns)
            except BaseException as exc:  # noqa: BLE001 - assert thread outcome
                output_errors.append(exc)

        movement_before = self._movement_writes(active)
        try:
            with mock.patch.object(calibrated, "step", side_effect=blocked_step):
                worker = threading.Thread(target=output_tick)
                worker.start()
                self.assertTrue(entered.wait(1.0))

                controller.update(
                    None,
                    (1080, 1920, 3),
                    measurement_ns=source_ns + 1_000_000,
                    measurement_observed=True,
                )
                self.assertFalse(controller._measurement_target_present)
                self.assertIsNone(controller._latest_target)
                release_step.set()
                worker.join(1.0)
                self.assertFalse(worker.is_alive())

            self.assertEqual(output_errors, [])
            self.assertEqual(self._movement_writes(active), movement_before)
            self.assertFalse(calibrated.ready)
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(controller._fractional_y, 0.0)
        finally:
            release_step.set()
            controller._output_thread = None
            controller.stop()

    def test_calibrated_control_honors_visible_max_step_and_vertical_cap(self) -> None:
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.0),
            CalibratedControlConfig(
                velocity_median_window=1,
                maximum_rate_x_counts_per_second=19_200.0,
                maximum_rate_y_counts_per_second=19_200.0,
            ),
        )
        controller = MakcuAimingController(
            MakcuAimConfig(
                max_step=80,
                output_hz=1000,
                vertical_rate_ratio=0.25,
            ),
            calibrated_controller=calibrated,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        target = Detection(0, "player", 0.95, (1050.0, 628.0, 1070.0, 728.0))
        base_ns = 52_000_000_000
        original_step = calibrated.step

        def diagnostic_step(*args, **kwargs) -> CalibratedControlOutput:
            return replace(
                original_step(*args, **kwargs),
                observer_position_sigma_x_pixels=1.25,
                observer_position_sigma_y_pixels=2.50,
                observer_velocity_sigma_x_pixels_per_second=125.0,
                observer_velocity_sigma_y_pixels_per_second=250.0,
                target_velocity_x_pixels_per_second=600.0,
                target_velocity_y_pixels_per_second=-400.0,
                velocity_feedforward_confidence_x=0.75,
                velocity_feedforward_confidence_y=0.50,
                motion_corroboration_confidence=0.625,
                innovation_mahalanobis_squared=3.25,
                innovation_rejected=True,
            )

        try:
            active.responses.extend(bytes((0b00010,)))
            with mock.patch.object(
                calibrated,
                "step",
                side_effect=diagnostic_step,
            ):
                for timestamp_ns in (base_ns, base_ns + 8_000_000):
                    controller.update(
                        target,
                        (1080, 1920, 3),
                        measurement_ns=timestamp_ns,
                    )
                    controller._output_tick(0.001, now_ns=timestamp_ns)
            movement = self._movement_writes(active)
            self.assertEqual(len(movement), 1)
            delta_x, delta_y = (
                int(value)
                for value in movement[0].decode("ascii").strip()[8:-1].split(",")
            )
            self.assertGreater(delta_x, 0)
            self.assertGreater(delta_y, 0)
            self.assertLess(delta_y, delta_x)
            self.assertLessEqual(delta_x, math.ceil(80 * 60 / 1000))
            self.assertLessEqual(delta_y, math.ceil(80 * 60 * 0.25 / 1000))
            output = controller.calibrated_control_output
            assert output is not None
            self.assertLessEqual(abs(output.rate_x_counts_per_second), 80 * 60)
            self.assertLessEqual(
                abs(output.rate_y_counts_per_second),
                80 * 60 * 0.25,
            )
            self.assertTrue(output.saturated_x)
            self.assertTrue(output.saturated_y)
            self.assertEqual(output.observer_position_sigma_x_pixels, 1.25)
            self.assertEqual(output.observer_position_sigma_y_pixels, 2.50)
            self.assertEqual(
                output.observer_velocity_sigma_x_pixels_per_second,
                125.0,
            )
            self.assertEqual(
                output.observer_velocity_sigma_y_pixels_per_second,
                250.0,
            )
            self.assertEqual(output.velocity_feedforward_confidence_x, 0.75)
            self.assertEqual(output.velocity_feedforward_confidence_y, 0.50)
            self.assertEqual(output.motion_corroboration_confidence, 0.625)
            self.assertEqual(output.innovation_mahalanobis_squared, 3.25)
            self.assertTrue(output.innovation_rejected)
            telemetry = controller.telemetry_snapshot()
            self.assertEqual(telemetry.control_samples, 1)
            self.assertEqual(telemetry.saturated_x_samples, 1)
            self.assertEqual(telemetry.saturated_y_samples, 1)
            self.assertAlmostEqual(telemetry.pursuit_x, 75.0)
            self.assertAlmostEqual(telemetry.pursuit_y, -100.0 / 3.0)
            self.assertEqual(
                telemetry.motion_corroboration_confidence,
                0.625,
            )
        finally:
            controller._output_thread = None
            controller.stop()

    def test_calibrated_failed_write_is_never_recorded_as_physical_history(self) -> None:
        calibrated = MakcuCalibratedController(
            CalibratedPlant(0.10, 0.10, 0.0),
            CalibratedControlConfig(velocity_median_window=1),
        )
        controller = MakcuAimingController(
            MakcuAimConfig(max_step=320, output_hz=1000),
            calibrated_controller=calibrated,
            serial_factory=self.factory,
            ports_provider=lambda: (self.port,),
            sleep=lambda _seconds: None,
            threaded_output=False,
        )
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        target = Detection(0, "player", 0.95, (1000.0, 528.0, 1020.0, 628.0))
        base_ns = 54_000_000_000
        try:
            active.responses.extend(bytes((0b00010,)))
            controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)
            controller._output_tick(0.001, now_ns=base_ns)
            second_ns = base_ns + 8_000_000
            controller.update(target, (1080, 1920, 3), measurement_ns=second_ns)
            with (
                mock.patch.object(
                    controller,
                    "_command",
                    side_effect=MakcuError("simulated calibrated serial failure"),
                ),
                self.assertRaisesRegex(MakcuError, "simulated calibrated serial failure"),
            ):
                controller._output_tick(0.001, now_ns=second_ns)
            self.assertEqual(tuple(calibrated._commands), ())
            self.assertEqual(controller.telemetry_snapshot().movement_commands, 0)
            self.assertEqual(self._movement_writes(active), ())
        finally:
            controller._output_thread = None
            controller.stop()

    def test_duplicate_calibrated_tick_is_rejected_before_a_second_write(self) -> None:
        controller, calibrated, active, _target, ready_ns = (
            self._ready_calibrated_adapter()
        )
        try:
            movement_before = self._movement_writes(active)
            history_before = tuple(calibrated._commands)

            controller._output_tick(0.001, now_ns=ready_ns)

            self.assertEqual(self._movement_writes(active), movement_before)
            self.assertEqual(tuple(calibrated._commands), history_before)
            output = controller.calibrated_control_output
            assert output is not None
            self.assertFalse(output.valid)
            self.assertEqual(output.reset_reason, "non-monotonic-clock")
        finally:
            controller._output_thread = None
            controller.stop()

    def test_calibrated_commit_revalidates_loss_release_and_stop(self) -> None:
        for revocation in ("target-loss", "release", "stop"):
            with self.subTest(revocation=revocation):
                controller, calibrated, active, _target, ready_ns = (
                    self._ready_calibrated_adapter()
                )
                entered = threading.Event()
                release_step = threading.Event()
                output_errors: list[BaseException] = []
                disarm_thread: threading.Thread | None = None
                original_step = calibrated.step

                def blocked_step(*args, **kwargs):
                    entered.set()
                    if not release_step.wait(1.0):
                        raise AssertionError("timed out waiting to release control step")
                    return original_step(*args, **kwargs)

                next_ns = ready_ns + 1_000_000

                def output_tick() -> None:
                    try:
                        controller._output_tick(0.001, now_ns=next_ns)
                    except BaseException as exc:  # noqa: BLE001 - assert thread outcome
                        output_errors.append(exc)

                movement_before = self._movement_writes(active)
                try:
                    with mock.patch.object(
                        calibrated,
                        "step",
                        side_effect=blocked_step,
                    ):
                        output_thread = threading.Thread(target=output_tick)
                        output_thread.start()
                        self.assertTrue(entered.wait(1.0))

                        if revocation == "target-loss":
                            controller.update(
                                None,
                                (1080, 1920, 3),
                                measurement_ns=next_ns,
                                measurement_observed=True,
                            )
                        elif revocation == "release":
                            self._queue_button_event(active, 0)
                            self.assertFalse(
                                controller.poll_activation(now_ns=next_ns)
                            )
                        else:
                            controller._stop_event.set()
                            disarm_thread = threading.Thread(
                                target=controller._disarm_for_shutdown
                            )
                            disarm_thread.start()

                        release_step.set()
                        output_thread.join(1.0)
                        self.assertFalse(output_thread.is_alive())
                        if disarm_thread is not None:
                            disarm_thread.join(1.0)
                            self.assertFalse(disarm_thread.is_alive())

                    self.assertEqual(output_errors, [])
                    self.assertEqual(self._movement_writes(active), movement_before)
                    self.assertFalse(calibrated.ready)
                    self.assertEqual(controller._fractional_x, 0.0)
                    self.assertEqual(controller._fractional_y, 0.0)
                finally:
                    release_step.set()
                    controller._output_thread = None
                    controller.stop()

    def test_zero_rounded_calibrated_tick_discards_superseded_state(self) -> None:
        controller, calibrated, active, _target, ready_ns = (
            self._ready_calibrated_adapter()
        )
        entered = threading.Event()
        release_step = threading.Event()
        original_step = calibrated.step

        def zero_rate_step(*args, **kwargs) -> CalibratedControlOutput:
            entered.set()
            if not release_step.wait(1.0):
                raise AssertionError("timed out waiting to release control step")
            original = original_step(*args, **kwargs)
            return CalibratedControlOutput(
                timestamp_ns=original.timestamp_ns,
                rate_x_counts_per_second=0.25,
                rate_y_counts_per_second=0.0,
                target_velocity_x_pixels_per_second=(
                    original.target_velocity_x_pixels_per_second
                ),
                target_velocity_y_pixels_per_second=(
                    original.target_velocity_y_pixels_per_second
                ),
                projected_error_x_pixels=original.projected_error_x_pixels,
                projected_error_y_pixels=original.projected_error_y_pixels,
                valid=True,
            )

        next_ns = ready_ns + 1_000_000
        movement_before = self._movement_writes(active)
        output_errors: list[BaseException] = []

        def output_tick() -> None:
            try:
                controller._output_tick(0.001, now_ns=next_ns)
            except BaseException as exc:  # noqa: BLE001 - assert thread outcome
                output_errors.append(exc)

        try:
            with mock.patch.object(calibrated, "step", side_effect=zero_rate_step):
                worker = threading.Thread(target=output_tick)
                worker.start()
                self.assertTrue(entered.wait(1.0))
                controller.update(
                    None,
                    (1080, 1920, 3),
                    measurement_ns=next_ns,
                    measurement_observed=True,
                )
                release_step.set()
                worker.join(1.0)
                self.assertFalse(worker.is_alive())
            self.assertEqual(output_errors, [])
            self.assertEqual(self._movement_writes(active), movement_before)
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(controller._fractional_y, 0.0)
            self.assertFalse(calibrated.ready)
        finally:
            release_step.set()
            controller._output_thread = None
            controller.stop()

    def test_calibrated_commit_rechecks_measurement_age_and_hold_deadline(self) -> None:
        for boundary in ("measurement-age", "maximum-hold"):
            with self.subTest(boundary=boundary):
                controller, calibrated, active, _target, ready_ns = (
                    self._ready_calibrated_adapter()
                )
                movement_before = self._movement_writes(active)
                try:
                    if boundary == "measurement-age":
                        next_ns = ready_ns + 39_000_000
                    else:
                        next_ns = ready_ns + 1_000_000
                        with controller._state_lock:
                            controller._activation_started_ns = (
                                next_ns
                                - round(
                                    MAX_CONTINUOUS_ACTIVATION_SECONDS
                                    * 1_000_000_000
                                )
                                + 1_000_000
                            )
                    with mock.patch(
                        "aiming.makcu.time.perf_counter_ns",
                        side_effect=(100, 2_000_100),
                    ):
                        controller._output_tick(0.001, now_ns=next_ns)
                    self.assertEqual(
                        self._movement_writes(active),
                        movement_before,
                    )
                    self.assertFalse(calibrated.ready)
                    self.assertEqual(controller._fractional_x, 0.0)
                    self.assertEqual(controller._fractional_y, 0.0)
                    if boundary == "maximum-hold":
                        self.assertTrue(controller._activation_requires_release)
                finally:
                    controller._output_thread = None
                    controller.stop()

    def test_calibration_enter_and_exit_synchronously_reset_calibrated_core(self) -> None:
        controller, calibrated, active, target, ready_ns = (
            self._ready_calibrated_adapter()
        )
        try:
            self.assertTrue(calibrated.ready)
            token = controller.enter_calibration_mode()
            self.assertFalse(calibrated.ready)
            self.assertIsNone(controller.calibrated_control_output)

            calibrated.step(
                ready_ns + 1,
                engaged=True,
                observation=ScreenErrorObservation(ready_ns + 1, 40.0, 0.0),
            )
            calibrated.step(
                ready_ns + 2,
                engaged=True,
                observation=ScreenErrorObservation(ready_ns + 2, 40.0, 0.0),
            )
            self.assertTrue(calibrated.ready)

            controller.exit_calibration_mode(token)
            self.assertFalse(calibrated.ready)
            self.assertIsNone(controller.calibrated_control_output)

            movement_before = self._movement_writes(active)
            self._queue_calibration_repress(active)
            next_ns = ready_ns + 8_000_000
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=next_ns,
            )
            controller._output_tick(0.001, now_ns=next_ns)
            self.assertEqual(self._movement_writes(active), movement_before)
            self.assertFalse(calibrated.ready)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_calibrated_history_capacity_fails_before_physical_write(self) -> None:
        controller, _calibrated, active, target, ready_ns = (
            self._ready_calibrated_adapter(
                maximum_command_history=16,
                plant_delay_seconds=0.25,
            )
        )
        try:
            timestamp_ns = ready_ns
            for _attempt in range(32):
                if len(self._movement_writes(active)) >= 16:
                    break
                timestamp_ns += 1_000_000
                controller.update(
                    target,
                    (1080, 1920, 3),
                    measurement_ns=timestamp_ns,
                )
                controller._output_tick(0.001, now_ns=timestamp_ns)
            movement_before = self._movement_writes(active)
            self.assertEqual(len(movement_before), 16)

            timestamp_ns += 1_000_000
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=timestamp_ns,
            )
            with self.assertRaisesRegex(
                MakcuError,
                "command preflight.*command-history-overflow",
            ):
                controller._output_tick(0.001, now_ns=timestamp_ns)
            self.assertEqual(self._movement_writes(active), movement_before)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_activation_poll_is_read_only(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        pressed_ns = 1_000_000_000
        self.assertEqual(controller.raw_activation_state, (False, False))
        active.responses.extend(bytes((0b00010,)))

        self.assertTrue(controller.poll_activation(now_ns=pressed_ns))
        self.assertEqual(controller.raw_activation_state, (True, True))
        self.assertFalse(any(write.startswith(b"km.move(") for write in active.writes))
        controller._activation_requires_release = True
        self.assertFalse(controller.activation_pressed)
        self.assertEqual(controller.raw_activation_state, (True, True))
        active.responses.extend(bytes((0,)))
        self.assertFalse(controller.poll_activation(now_ns=pressed_ns + 1))
        self.assertEqual(controller.raw_activation_state, (True, False))

    def test_poll_button_mask_reports_latest_button_bits(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        now_ns = 2_000_000_000

        active.responses.extend(bytes((0b00010,)))
        self.assertEqual(controller.poll_button_mask(now_ns=now_ns), 0b00010)
        active.responses.extend(bytes((0,)))
        self.assertEqual(controller.poll_button_mask(now_ns=now_ns + 10_000_000), 0)

    def test_button_parser_reports_framed_provenance(self) -> None:
        parser = _ButtonStreamParser()
        self.assertEqual(parser.feed(bytes((0b00010,))), ((0b00010, False),))
        parser.reset()
        self.assertEqual(
            parser.feed(b"km.buttons" + bytes((0b00010,)) + b"\r\n"),
            ((0b00010, True),),
        )

    def test_framed_button_masks_survive_split_reads_and_ignore_prompt_noise(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        now_ns = 2_000_000_000

        # Current MAKCU firmware frames a raw five-bit mask after this prefix.
        for chunk in (b"km.but", b"tons", bytes((0b00010,)), b"\r", b"\n>>> "):
            active.responses.extend(chunk)
            controller.poll_button_mask(now_ns=now_ns)
            now_ns += 1_000_000
        self.assertEqual(controller.poll_button_mask(now_ns=now_ns), 0b00010)

        # Command acknowledgements, prompts, and terminal control sequences are
        # not mouse reports even though they can contain values below 0x20.
        active.responses.extend(b"\x06km.buttons(1)\r\n>>> \x1b[2K")
        self.assertEqual(controller.poll_button_mask(now_ns=now_ns + 1), 0b00010)

        for chunk in (b"km.buttons", bytes((0,)), b"\r\n", b">>> "):
            active.responses.extend(chunk)
            controller.poll_button_mask(now_ns=now_ns)
            now_ns += 1_000_000
        self.assertEqual(controller.poll_button_mask(now_ns=now_ns), 0)

    def test_every_frame_split_preserves_press_and_zero_release(self) -> None:
        for prefix in (b"km.buttons", b"km."):
            press = prefix + bytes((0b00010,)) + b"\r\n>>> "
            release = prefix + bytes((0,)) + b"\r\n>>> "

            for boundary in range(len(press) + 1):
                with self.subTest(prefix=prefix, event="press", boundary=boundary):
                    controller = self.controller()
                    controller.start()
                    active = self.factory.connections[-1]
                    try:
                        active.responses.extend(press[:boundary])
                        controller.poll_button_mask(now_ns=1_000_000_000)
                        active.responses.extend(press[boundary:])
                        self.assertEqual(
                            controller.poll_button_mask(now_ns=1_000_000_001),
                            0b00010,
                        )
                    finally:
                        controller.stop()

            for boundary in range(len(release) + 1):
                with self.subTest(prefix=prefix, event="release", boundary=boundary):
                    controller = self.controller()
                    controller.start()
                    active = self.factory.connections[-1]
                    try:
                        active.responses.extend(press)
                        self.assertEqual(
                            controller.poll_button_mask(now_ns=2_000_000_000),
                            0b00010,
                        )
                        active.responses.extend(release[:boundary])
                        controller.poll_button_mask(now_ns=2_000_000_001)
                        active.responses.extend(release[boundary:])
                        self.assertEqual(
                            controller.poll_button_mask(now_ns=2_000_000_002),
                            0,
                        )
                    finally:
                        controller.stop()

    def test_every_coalesced_frame_pair_split_preserves_zero_release(self) -> None:
        for prefix in (b"km.buttons", b"km."):
            press = prefix + bytes((0b00010,)) + b"\r\n>>> "
            release = prefix + bytes((0,)) + b"\r\n>>> "
            events = press + release

            # This includes boundaries inside both prefixes and the case where
            # the first terminator and following release arrive in one read.
            for boundary in range(len(events) + 1):
                with self.subTest(prefix=prefix, boundary=boundary):
                    controller = self.controller()
                    controller.start()
                    active = self.factory.connections[-1]
                    try:
                        active.responses.extend(events[:boundary])
                        controller.poll_button_mask(now_ns=3_000_000_000)
                        active.responses.extend(events[boundary:])
                        self.assertEqual(
                            controller.poll_button_mask(now_ns=3_000_000_001),
                            0,
                        )
                    finally:
                        controller.stop()

    def test_legacy_raw_button_masks_remain_supported_across_reads(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        now_ns = 3_000_000_000

        active.responses.extend(bytes((0b00010,)))
        self.assertEqual(controller.poll_button_mask(now_ns=now_ns), 0b00010)
        active.responses.extend(bytes((0,)))
        self.assertEqual(controller.poll_button_mask(now_ns=now_ns + 1), 0)

        # Native-compatible firmware has also used the shorter km.<mask>
        # framing. Its prefix may be divided by arbitrary serial read bounds.
        for chunk in (b"k", b"m.", bytes((0b00010,)), b"\r\n>>> "):
            active.responses.extend(chunk)
            controller.poll_button_mask(now_ns=now_ns)
            now_ns += 1_000_000
        self.assertEqual(controller.poll_button_mask(now_ns=now_ns), 0b00010)

    def test_valid_release_report_stops_activation_immediately(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        now_ns = 1_000_000_000
        active.responses.extend(bytes((0b00010,)))
        self.assertTrue(controller.poll_activation(now_ns=now_ns))

        active.responses.extend(bytes((0,)))
        self.assertFalse(controller.poll_activation(now_ns=now_ns + 1))

    def test_event_driven_press_remains_active_until_authoritative_release(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        pressed_ns = 1_000_000_000
        active.responses.extend(b"km.buttons" + bytes((0b00010,)) + b"\r\n>>> ")
        self.assertTrue(controller.poll_activation(now_ns=pressed_ns))

        # Button callbacks are event-driven: silence is a held button, not a
        # stale sample. This intentionally exceeds the v1.0.9 150 ms timeout.
        held_ns = pressed_ns + 1_000_000_000
        self.assertTrue(controller.poll_activation(now_ns=held_ns))
        self.assertEqual(controller.poll_button_mask(now_ns=held_ns + 1), 0b00010)

        active.responses.extend(b"km.buttons" + bytes((0,)) + b"\r\n>>> ")
        self.assertFalse(controller.poll_activation(now_ns=held_ns + 2))
        self.assertEqual(controller.poll_button_mask(now_ns=held_ns + 3), 0)

    def test_continuous_activation_is_bounded_and_requires_release(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        pressed_ns = 1_000_000_000
        active.responses.extend(b"km.buttons" + bytes((0b00010,)) + b"\r\n>>> ")
        self.assertTrue(controller.poll_activation(now_ns=pressed_ns))

        expired_ns = pressed_ns + int(
            (MAX_CONTINUOUS_ACTIVATION_SECONDS + 0.01) * 1_000_000_000
        )
        self.assertFalse(controller.poll_activation(now_ns=expired_ns))

        # Another press without an observed release cannot re-authorize output.
        active.responses.extend(b"km.buttons" + bytes((0b00010,)) + b"\r\n>>> ")
        self.assertFalse(controller.poll_activation(now_ns=expired_ns + 1))

        active.responses.extend(b"km.buttons" + bytes((0,)) + b"\r\n>>> ")
        self.assertFalse(controller.poll_activation(now_ns=expired_ns + 2))
        active.responses.extend(b"km.buttons" + bytes((0b00010,)) + b"\r\n>>> ")
        self.assertTrue(controller.poll_activation(now_ns=expired_ns + 3))

    def test_1000hz_ticks_preserve_fractional_motion_and_stop_when_stale(self) -> None:
        controller = self.controller(
            MakcuAimConfig(strength=0.5, max_step=160, output_hz=1000)
        )
        target = Detection(0, "player", 0.9, (1060, 480, 1260, 980))
        controller.start()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        controller.update(target, (1080, 1920, 3), active=True)
        movement_start = len(active.writes)

        for _ in range(10):
            controller._output_tick(0.001)

        movements = [
            write.decode("ascii")
            for write in active.writes[movement_start:]
            if write.startswith(b"km.move(")
        ]
        self.assertGreaterEqual(len(movements), 8)
        stale_now = controller._latest_update_ns + int((TARGET_STALE_SECONDS + 0.01) * 1e9)
        before_stale = len(active.writes)
        controller._output_tick(0.001, now_ns=stale_now)
        self.assertEqual(len(active.writes), before_stale)

    def test_telemetry_observes_gate_duty_and_successful_movement_only(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=0.5,
                max_step=1000,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        base_ns = time.perf_counter_ns()
        target = Detection(0, "player", 0.9, (1150, 540, 1170, 560))
        controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)

        try:
            self.assertEqual(controller.telemetry_snapshot(), MakcuTelemetrySnapshot())
            writes_before_snapshot = len(active.writes)
            controller.telemetry_snapshot()
            self.assertEqual(len(active.writes), writes_before_snapshot)

            active.responses.extend(bytes((0b00010,)))
            controller._output_tick(0.001, now_ns=base_ns + 1_000_000)
            controller._output_tick(0.001, now_ns=base_ns + 2_000_000)
            active.responses.extend(bytes((0,)))
            controller._output_tick(0.001, now_ns=base_ns + 3_000_000)

            self.assertEqual(
                controller.telemetry_snapshot(),
                MakcuTelemetrySnapshot(
                    output_ticks=3,
                    active_input_ticks=3,
                    button_pressed_ticks=2,
                    target_present_ticks=3,
                    fresh_target_ticks=3,
                    authorized_ticks=2,
                    movement_commands=2,
                    emitted_x=12,
                    emitted_y=0,
                    emitted_abs_x=12,
                    emitted_abs_y=0,
                    control_samples=1,
                    control_error_x=200.0,
                    control_error_abs_x=200.0,
                ),
            )
        finally:
            controller._output_thread = None
            controller.stop()

    def test_telemetry_distinguishes_missing_and_stale_targets_and_resets_on_start(self) -> None:
        controller = self.controller(MakcuAimConfig(output_hz=1000))
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        base_ns = time.perf_counter_ns()
        target = Detection(0, "player", 0.9, (1060, 480, 1260, 980))
        controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)

        try:
            controller._output_tick(0.001, now_ns=base_ns + 1_000_000)
            stale_ns = base_ns + round((TARGET_STALE_SECONDS + 0.01) * 1e9)
            controller._output_tick(0.001, now_ns=stale_ns)
            self.assertEqual(controller._pursuit_correction_x, 0.0)
            self.assertEqual(controller._pursuit_correction_y, 0.0)
            controller.update(None, (1080, 1920, 3), measurement_ns=stale_ns + 1)
            controller._output_tick(0.001, now_ns=stale_ns + 2)

            snapshot = controller.telemetry_snapshot()
            self.assertEqual(snapshot.output_ticks, 3)
            self.assertEqual(snapshot.active_input_ticks, 3)
            self.assertEqual(snapshot.button_pressed_ticks, 3)
            self.assertEqual(snapshot.target_present_ticks, 2)
            self.assertEqual(snapshot.fresh_target_ticks, 1)
            self.assertEqual(snapshot.authorized_ticks, 1)
        finally:
            controller._output_thread = None
            controller.stop()

        controller.start(output_loop=False)
        try:
            self.assertEqual(controller.telemetry_snapshot(), MakcuTelemetrySnapshot())
        finally:
            controller.stop()

    def test_control_telemetry_is_detector_sample_based_and_resets_once_per_gap(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=1.0,
                max_step=110,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        base_ns = time.perf_counter_ns()
        target = Detection(0, "player", 0.9, (1050, 540, 1070, 640))

        try:
            controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)
            controller._output_tick(0.0, now_ns=base_ns)
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=base_ns + 12_000_000,
            )
            controller._output_tick(0.012, now_ns=base_ns + 12_000_000)

            # Replaying the held detector sample at 1 kHz must not multiply
            # either its visual-error or pursuit observation.
            for offset_ms in range(13, 63):
                controller._output_tick(
                    0.001,
                    now_ns=base_ns + offset_ms * 1_000_000,
                )

            snapshot = controller.telemetry_snapshot()
            self.assertEqual(snapshot.control_samples, 2)
            self.assertAlmostEqual(snapshot.control_error_x, 200.0)
            self.assertAlmostEqual(snapshot.control_error_abs_x, 200.0)
            self.assertAlmostEqual(snapshot.pursuit_x, 10.0)
            self.assertAlmostEqual(snapshot.pursuit_abs_x, 10.0)
            self.assertEqual(snapshot.saturated_x_samples, 1)
            self.assertEqual(snapshot.pursuit_resets, 0)

            missing_ns = base_ns + 63_000_000
            controller.update(None, (1080, 1920, 3), measurement_ns=missing_ns)
            controller._output_tick(0.001, now_ns=missing_ns)
            for offset_ms in range(64, 74):
                controller._output_tick(
                    0.001,
                    now_ns=base_ns + offset_ms * 1_000_000,
                )

            self.assertEqual(controller.telemetry_snapshot().pursuit_resets, 1)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_telemetry_keeps_absolute_motion_and_ignores_failed_commands(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=0.5,
                max_step=1000,
                deadzone_pixels=0.0,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        right = Detection(0, "player", 0.9, (1150, 540, 1170, 560))
        left = Detection(0, "player", 0.9, (750, 540, 770, 560))

        try:
            controller.update(right, (1080, 1920, 3))
            controller.update(left, (1080, 1920, 3))
            snapshot = controller.telemetry_snapshot()
            self.assertEqual(snapshot.movement_commands, 2)
            self.assertEqual(snapshot.emitted_x, 0)
            self.assertEqual(snapshot.emitted_abs_x, 200)

            with (
                mock.patch.object(
                    controller,
                    "_command",
                    side_effect=MakcuError("simulated serial failure"),
                ),
                self.assertRaisesRegex(MakcuError, "simulated serial failure"),
            ):
                controller.update(right, (1080, 1920, 3))
            failed_snapshot = controller.telemetry_snapshot()
            self.assertEqual(failed_snapshot.output_ticks, 3)
            self.assertEqual(failed_snapshot.authorized_ticks, 3)
            self.assertEqual(failed_snapshot.movement_commands, 2)
            self.assertEqual(failed_snapshot.emitted_x, 0)
            self.assertEqual(failed_snapshot.emitted_abs_x, 200)
        finally:
            controller.stop()

    def test_calibration_requires_a_live_exact_1000hz_worker(self) -> None:
        controller = self.controller(MakcuAimConfig(output_hz=1000))
        controller.start(output_loop=False)
        try:
            with self.assertRaisesRegex(MakcuError, "live 1 kHz output worker"):
                controller.enter_calibration_mode()

            stopped_worker = mock.Mock()
            stopped_worker.is_alive.return_value = False
            controller._output_thread = stopped_worker
            with self.assertRaisesRegex(MakcuError, "live 1 kHz output worker"):
                controller.enter_calibration_mode()
        finally:
            controller._output_thread = None
            controller.stop()

        controller = self.controller(MakcuAimConfig(output_hz=500))
        controller.start(output_loop=False)
        try:
            live_worker = mock.Mock()
            live_worker.is_alive.return_value = True
            controller._output_thread = live_worker
            with self.assertRaisesRegex(MakcuError, "live 1 kHz output worker"):
                controller.enter_calibration_mode()
        finally:
            controller._output_thread = None
            controller.stop()

    def test_calibration_held_at_entry_requires_release_and_new_press(self) -> None:
        controller = self.controller(MakcuAimConfig(output_hz=1000))
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        base_ns = 15_000_000_000
        active.responses.extend(bytes((0b00010,)))
        self.assertTrue(controller.poll_activation(now_ns=base_ns))
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        token = controller.enter_calibration_mode()
        try:
            self.assertFalse(
                controller._activation_pressed_at(base_ns + 1_000_000)
            )
            self.assertEqual(self._movement_writes(active), ())

            self._establish_calibration_hold(
                controller,
                active,
                press_ns=base_ns + 2_000_000,
            )
            controller.publish_calibration_lease(
                True,
                base_ns + 2_000_000,
                token,
            )
            controller.request_calibration_pulse("x", 1, 1000.0, token)
            controller._output_tick(0.001, now_ns=base_ns + 3_000_000)

            self.assertEqual(self._movement_writes(active), (b"km.move(1,0)\r",))
            self.assertEqual(
                controller.calibration_snapshot().emitted_events,
                ((base_ns + 3_000_000, 1, 0),),
            )
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_session_consumes_release_dwell_between_updates(self) -> None:
        from tests.test_makcu_calibration_session import _binding

        controller = self.controller(MakcuAimConfig(output_hz=1000))
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        started_ns = time.perf_counter_ns()
        session = MakcuCalibrationSession(
            controller,
            _binding(),
            started_ns=started_ns,
        )
        decision_ns = started_ns
        try:
            status = session.update_from_controller(
                started_ns,
                observation=self._calibration_observation(started_ns),
            )
            self.assertEqual(status.state, CalibrationSessionState.WAIT_RELEASE)
            entered_ns = controller.raw_activation_snapshot.calibration_entered_ns

            release_ns = entered_ns + 1_000_000
            press_ns = entered_ns + 101_000_000
            self._queue_button_event(active, 0b00010)
            controller._output_tick(0.0, now_ns=entered_ns + 500_000)
            self._queue_button_event(active, 0)
            controller._output_tick(0.0, now_ns=release_ns)
            self._queue_button_event(active, 0b00010)
            controller._output_tick(0.0, now_ns=press_ns)

            decision_ns = press_ns + 1_000_000
            status = session.update_from_controller(
                decision_ns,
                observation=self._calibration_observation(decision_ns),
            )
            raw = controller.raw_activation_snapshot
            self.assertEqual(status.state, CalibrationSessionState.WAIT_HOLD)
            self.assertIn("aim mode settles", status.message)
            self.assertTrue(raw.pressed)
            self.assertEqual(raw.completed_release_started_ns, release_ns)
            self.assertEqual(raw.completed_press_ns, press_ns)
            self.assertEqual(controller.calibration_snapshot().emitted_events, ())
            self.assertEqual(self._movement_writes(active), ())

            decision_ns += 300_000_000
            status = session.update_from_controller(
                decision_ns,
                observation=self._calibration_observation(decision_ns),
            )
            self.assertEqual(status.state, CalibrationSessionState.BASELINE_SETTLE)
            self.assertEqual(controller.calibration_snapshot().emitted_events, ())
            self.assertEqual(self._movement_writes(active), ())
        finally:
            if not session.terminal:
                session.abort("test cleanup", now_ns=decision_ns + 1_000_000)
            controller._output_thread = None
            controller.stop()

    def test_calibration_session_rejects_too_brief_dwell_between_updates(self) -> None:
        from tests.test_makcu_calibration_session import _binding

        controller = self.controller(MakcuAimConfig(output_hz=1000))
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        started_ns = time.perf_counter_ns()
        session = MakcuCalibrationSession(
            controller,
            _binding(),
            started_ns=started_ns,
        )
        decision_ns = started_ns
        try:
            session.update_from_controller(
                started_ns,
                observation=self._calibration_observation(started_ns),
            )
            entered_ns = controller.raw_activation_snapshot.calibration_entered_ns

            release_ns = entered_ns + 1_000_000
            press_ns = release_ns + 40_000_000
            self._queue_button_event(active, 0b00010)
            controller._output_tick(0.0, now_ns=entered_ns + 500_000)
            self._queue_button_event(active, 0)
            controller._output_tick(0.0, now_ns=release_ns)
            self._queue_button_event(active, 0b00010)
            controller._output_tick(0.0, now_ns=press_ns)

            decision_ns = press_ns + 1_000_000
            status = session.update_from_controller(
                decision_ns,
                observation=self._calibration_observation(decision_ns),
            )
            self.assertEqual(status.state, CalibrationSessionState.WAIT_RELEASE)
            self.assertIn("too brief", status.message.lower())
            self.assertEqual(controller.calibration_snapshot().emitted_events, ())
            self.assertEqual(self._movement_writes(active), ())
        finally:
            if not session.terminal:
                session.abort("test cleanup", now_ns=decision_ns + 1_000_000)
            controller._output_thread = None
            controller.stop()

    def test_calibration_fresh_released_report_arms_without_priming_press(self) -> None:
        from tests.test_makcu_calibration_session import _binding

        controller = self.controller(MakcuAimConfig(output_hz=1000))
        controller.start(output_loop=False)
        active = self.factory.connections[-1]
        preentry_ns = time.perf_counter_ns()
        self._queue_button_event(active, 0)
        controller.poll_button_mask(now_ns=preentry_ns)
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        started_ns = time.perf_counter_ns()
        session = MakcuCalibrationSession(
            controller,
            _binding(),
            started_ns=started_ns,
        )
        decision_ns = started_ns
        try:
            session.update_from_controller(
                started_ns,
                observation=self._calibration_observation(started_ns),
            )
            entered_ns = controller.raw_activation_snapshot.calibration_entered_ns

            # An event-driven board can be silent while an already-released
            # button remains released. Cached pre-entry zero must not start the
            # dwell, because calibration has not yet received fresh evidence.
            decision_ns = entered_ns + 500_000_000
            status = session.update_from_controller(
                decision_ns,
                observation=self._calibration_observation(decision_ns),
            )
            raw = controller.raw_activation_snapshot
            self.assertEqual(status.state, CalibrationSessionState.WAIT_RELEASE)
            self.assertIn("post-entry framed", status.message.lower())
            self.assertFalse(raw.post_entry_press_seen)
            self.assertIsNone(raw.release_started_ns)
            self.assertEqual(self._movement_writes(active), ())

            # One fresh framed zero is direct physical release proof. No
            # artificial press/release cycle is required before the real hold.
            release_ns = entered_ns + 501_000_000
            self._queue_button_event(active, 0)
            controller._output_tick(0.0, now_ns=release_ns)

            decision_ns = release_ns + 81_000_000
            with mock.patch(
                "aiming.makcu.time.perf_counter_ns",
                return_value=decision_ns,
            ):
                status = session.update_from_controller(
                    decision_ns,
                    observation=self._calibration_observation(decision_ns),
                )
            self.assertEqual(status.state, CalibrationSessionState.WAIT_HOLD)
            self.assertIn("release confirmed", status.message.lower())
            self.assertFalse(controller.raw_activation_snapshot.post_entry_press_seen)
            self.assertEqual(self._movement_writes(active), ())

            final_press_ns = decision_ns + 1_000_000
            self._queue_button_event(active, 0b00010)
            controller._output_tick(0.0, now_ns=final_press_ns)
            decision_ns = final_press_ns + 1_000_000
            status = session.update_from_controller(
                decision_ns,
                observation=None,
            )
            self.assertEqual(status.state, CalibrationSessionState.WAIT_HOLD)
            self.assertIn("aim mode settles", status.message)
            self.assertEqual(controller.calibration_snapshot().emitted_events, ())
            self.assertEqual(self._movement_writes(active), ())

            decision_ns += 300_000_000
            status = session.update_from_controller(
                decision_ns,
                observation=self._calibration_observation(decision_ns),
            )
            self.assertEqual(status.state, CalibrationSessionState.BASELINE_SETTLE)
            self.assertEqual(controller.calibration_snapshot().emitted_events, ())
            self.assertEqual(self._movement_writes(active), ())
        finally:
            if not session.terminal:
                session.abort("test cleanup", now_ns=decision_ns + 1_000_000)
            controller._output_thread = None
            controller.stop()

    def test_calibration_rejects_standalone_control_bytes_as_proof(self) -> None:
        from tests.test_makcu_calibration_session import _binding

        controls = (
            ("nul", b"\x00"),
            ("ack", b"\x06"),
            ("lf", b"\x0a"),
            ("cr", b"\x0d"),
            ("esc", b"\x1b"),
            ("unit-separator", b"\x1f"),
        )
        for name, control in controls:
            with self.subTest(control=name):
                controller = self.controller(MakcuAimConfig(output_hz=1000))
                controller.start(output_loop=False)
                active = self.factory.connections[-1]
                live_worker = mock.Mock()
                live_worker.is_alive.return_value = True
                controller._output_thread = live_worker
                started_ns = time.perf_counter_ns()
                session = MakcuCalibrationSession(
                    controller,
                    _binding(),
                    started_ns=started_ns,
                )
                decision_ns = started_ns
                try:
                    session.update_from_controller(
                        started_ns,
                        observation=self._calibration_observation(started_ns),
                    )
                    entered_ns = (
                        controller.raw_activation_snapshot.calibration_entered_ns
                    )

                    for offset_ms, payload in (
                        (1, control),
                        (2, b"\x02"),
                        (3, b"\x00"),
                        (103, b"\x02"),
                    ):
                        active.responses.extend(payload)
                        controller._output_tick(
                            0.0,
                            now_ns=entered_ns + offset_ms * 1_000_000,
                        )

                    decision_ns = entered_ns + 104_000_000
                    status = session.update_from_controller(
                        decision_ns,
                        observation=self._calibration_observation(decision_ns),
                    )
                    raw = controller.raw_activation_snapshot
                    self.assertEqual(
                        status.state,
                        CalibrationSessionState.WAIT_RELEASE,
                    )
                    self.assertIn("post-entry framed", status.message.lower())
                    self.assertFalse(raw.post_entry_press_seen)
                    self.assertEqual(
                        raw.framed_report_sequence,
                        raw.calibration_entry_framed_report_sequence,
                    )
                    self.assertFalse(raw.last_report_framed)
                    self.assertIsNone(raw.release_started_ns)
                    self.assertIsNone(raw.completed_press_ns)
                    self.assertEqual(controller.calibration_snapshot().emitted_events, ())
                    self.assertEqual(self._movement_writes(active), ())
                finally:
                    if not session.terminal:
                        session.abort(
                            "test cleanup",
                            now_ns=decision_ns + 1_000_000,
                        )
                    controller._output_thread = None
                    controller.stop()

    def test_calibration_excludes_normal_output_and_rejects_token_misuse(self) -> None:
        controller = self.controller(
            MakcuAimConfig(output_hz=1000, head_ratio=0.0)
        )
        controller.start(output_loop=False)
        target = Detection(0, "player", 0.9, (1050, 540, 1070, 640))
        controller.update(target, (1080, 1920, 3), measurement_ns=1_000_000_000)
        controller._fractional_x = 0.75
        controller._smoothed_rate_x = 500.0
        controller._pursuit_correction_x = 25.0
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        token = controller.enter_calibration_mode()
        wrong_token = object()
        try:
            self.assertIsNone(controller._latest_target)
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(controller._smoothed_rate_x, 0.0)
            self.assertEqual(controller._pursuit_correction_x, 0.0)
            with self.assertRaisesRegex(MakcuError, "already active"):
                controller.enter_calibration_mode()
            with self.assertRaisesRegex(MakcuError, "invalid or inactive"):
                controller.publish_calibration_lease(True, 1_000_000_000, wrong_token)
            with self.assertRaisesRegex(MakcuError, "invalid or inactive"):
                controller.request_calibration_pulse("x", 1, 1000.0, wrong_token)
            with self.assertRaisesRegex(MakcuError, "invalid or inactive"):
                controller.exit_calibration_mode(wrong_token)
            with self.assertRaisesRegex(MakcuError, "unavailable during calibration"):
                controller.update(
                    target,
                    (1080, 1920, 3),
                    measurement_ns=1_000_000_001,
                )

            controller.exit_calibration_mode(token)
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=1_000_000_002,
            )
            self.assertIs(controller._latest_target, target)
        finally:
            if controller.calibration_snapshot().active:
                controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_rejects_pulse_bounds_and_parallel_axes(self) -> None:
        controller, active, token = self._start_test_calibration()
        base_ns = 20_000_000_000
        try:
            self._establish_calibration_hold(
                controller,
                active,
                press_ns=base_ns,
            )
            controller.publish_calibration_lease(True, base_ns, token)
            with self.assertRaisesRegex(ValueError, "axis"):
                controller.request_calibration_pulse("z", 1, 1000.0, token)
            for invalid_counts in (True, 1.5):
                with self.subTest(counts=invalid_counts):
                    with self.assertRaises(TypeError):
                        controller.request_calibration_pulse(
                            "x", invalid_counts, 1000.0, token
                        )
            with self.assertRaisesRegex(ValueError, "cannot be zero"):
                controller.request_calibration_pulse("x", 0, 1000.0, token)
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                controller.request_calibration_pulse(
                    "x", CALIBRATION_MAX_EXCURSION_COUNTS + 1, 1000.0, token
                )
            for invalid_rate in (
                True,
                0.0,
                math.nan,
                CALIBRATION_MAX_RATE_COUNTS_PER_SECOND + 1.0,
            ):
                with self.subTest(rate=invalid_rate):
                    with self.assertRaisesRegex(ValueError, "pulse rate"):
                        controller.request_calibration_pulse(
                            "x", 1, invalid_rate, token
                        )

            controller.request_calibration_pulse("x", 10, 1000.0, token)
            with self.assertRaisesRegex(MakcuError, "already pending"):
                controller.request_calibration_pulse("y", 10, 1000.0, token)
            snapshot = controller.calibration_snapshot()
            self.assertEqual(snapshot.pending_axis, "x")
            self.assertEqual(snapshot.pending_counts, 10)
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_session_budget_uses_actual_absolute_counts(self) -> None:
        controller, active, token = self._start_test_calibration()
        base_ns = 30_000_000_000
        now_ns = base_ns
        # At the validated low-end response of 0.10 px/count, a 120-count
        # excursion reaches the fitter's 12px quality floor. Two symmetric
        # pulses per polarity and axis consume 960 counts, leaving 1,440 for
        # bounded adaptive scouts/returns without weakening the core gates.
        minimum_qualifying_counts = math.ceil(
            MIN_EXCURSION_PIXELS / 0.10
        )
        minimum_evidence = (
            ("x", minimum_qualifying_counts),
            ("x", -minimum_qualifying_counts),
            ("x", minimum_qualifying_counts),
            ("x", -minimum_qualifying_counts),
            ("y", minimum_qualifying_counts),
            ("y", -minimum_qualifying_counts),
            ("y", minimum_qualifying_counts),
            ("y", -minimum_qualifying_counts),
        )
        self.assertEqual(MIN_PULSES_PER_POLARITY, 2)
        self.assertLessEqual(
            minimum_qualifying_counts,
            CALIBRATION_MAX_EXCURSION_COUNTS,
        )
        self.assertEqual(
            sum(abs(counts) for _axis, counts in minimum_evidence),
            960,
        )
        reserve = (
            ("x", 200),
            ("x", -200),
            ("y", 200),
            ("y", -200),
            ("x", 200),
            ("x", -200),
            ("y", 200),
            ("y", -40),
        )
        excursions = minimum_evidence + reserve
        self.assertEqual(
            sum(abs(counts) for _axis, counts in excursions),
            CALIBRATION_MAX_SESSION_ABS_COUNTS,
        )
        try:
            self._establish_calibration_hold(
                controller,
                active,
                press_ns=base_ns,
            )
            for axis, counts in excursions:
                controller.publish_calibration_lease(True, now_ns, token)
                controller.request_calibration_pulse(
                    axis,
                    counts,
                    CALIBRATION_MAX_RATE_COUNTS_PER_SECOND,
                    token,
                )
                while controller.calibration_snapshot().pending_counts:
                    now_ns += 1_000_000
                    if (now_ns - base_ns) % 20_000_000 == 0:
                        controller.publish_calibration_lease(True, now_ns, token)
                    controller._output_tick(0.001, now_ns=now_ns)

            snapshot = controller.calibration_snapshot()
            self.assertEqual(
                snapshot.emitted_abs_counts,
                CALIBRATION_MAX_SESSION_ABS_COUNTS,
            )
            self.assertEqual(snapshot.movement_commands, len(snapshot.emitted_events))
            self.assertLessEqual(
                len(snapshot.emitted_events),
                CALIBRATION_MAX_SESSION_ABS_COUNTS,
            )
            self.assertTrue(
                all(
                    bool(delta_x) != bool(delta_y)
                    for _, delta_x, delta_y in snapshot.emitted_events
                )
            )

            controller.publish_calibration_lease(True, now_ns, token)
            with self.assertRaisesRegex(ValueError, "session cannot exceed"):
                controller.request_calibration_pulse("x", 1, 1.0, token)
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_snapshot_records_exact_successful_emission_order(self) -> None:
        controller, active, token = self._start_test_calibration()
        base_ns = 40_000_000_000
        try:
            self._establish_calibration_hold(
                controller,
                active,
                press_ns=base_ns,
            )
            controller.publish_calibration_lease(True, base_ns, token)
            controller.request_calibration_pulse("x", 5, 2400.0, token)
            for offset_ms in (1, 2, 3):
                controller._output_tick(
                    0.001,
                    now_ns=base_ns + offset_ms * 1_000_000,
                )
            controller.publish_calibration_lease(
                True,
                base_ns + 3_000_000,
                token,
            )
            controller.request_calibration_pulse("y", -4, 2400.0, token)
            # Even a delayed tick cannot collapse 10 ms of backlog into a
            # command above the hard three-count per-tick ceiling.
            controller._output_tick(0.010, now_ns=base_ns + 13_000_000)
            controller._output_tick(0.001, now_ns=base_ns + 14_000_000)

            expected_events = (
                (base_ns + 1_000_000, 2, 0),
                (base_ns + 2_000_000, 2, 0),
                (base_ns + 3_000_000, 1, 0),
                (base_ns + 13_000_000, 0, -3),
                (base_ns + 14_000_000, 0, -1),
            )
            self.assertEqual(
                controller.calibration_snapshot(),
                MakcuCalibrationSnapshot(
                    active=True,
                    emitted_x=5,
                    emitted_y=-4,
                    emitted_abs_counts=9,
                    movement_commands=5,
                    first_emitted_ns=expected_events[0][0],
                    last_emitted_ns=expected_events[-1][0],
                    emitted_events=expected_events,
                ),
            )
            self.assertGreater(controller.calibration_snapshot().captured_ns, 0)
            self.assertEqual(
                self._movement_writes(active),
                (
                    b"km.move(2,0)\r",
                    b"km.move(2,0)\r",
                    b"km.move(1,0)\r",
                    b"km.move(0,-3)\r",
                    b"km.move(0,-1)\r",
                ),
            )
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_release_aborts_partial_pulse_without_return_motion(self) -> None:
        controller, active, token = self._start_test_calibration()
        base_ns = 50_000_000_000
        try:
            self._establish_calibration_hold(
                controller,
                active,
                press_ns=base_ns,
            )
            controller.publish_calibration_lease(True, base_ns, token)
            controller.request_calibration_pulse("x", 10, 1000.0, token)
            controller._output_tick(0.001, now_ns=base_ns + 1_000_000)
            with controller._state_lock:
                controller._fractional_x = 0.75
                controller._smoothed_rate_x = 500.0
                controller._pursuit_correction_x = 25.0
                controller._latest_active = True
            writes_after_first_count = self._movement_writes(active)

            self._queue_button_event(active, 0)
            controller._output_tick(0.001, now_ns=base_ns + 2_000_000)
            snapshot = controller.calibration_snapshot()
            self.assertEqual(snapshot.pending_counts, 0)
            self.assertEqual(snapshot.emitted_abs_counts, 1)
            self.assertEqual(snapshot.emitted_events, ((base_ns + 1_000_000, 1, 0),))
            self.assertIn("physical activation was released", snapshot.abort_reason or "")
            self.assertEqual(controller._fractional_x, 0.0)
            self.assertEqual(controller._smoothed_rate_x, 0.0)
            self.assertEqual(controller._pursuit_correction_x, 0.0)
            self.assertFalse(controller._latest_active)

            self._queue_button_event(active, 0b00010)
            controller._output_tick(0.010, now_ns=base_ns + 12_000_000)
            self.assertEqual(self._movement_writes(active), writes_after_first_count)
            with self.assertRaisesRegex(MakcuError, "session is aborted"):
                controller.publish_calibration_lease(
                    True, base_ns + 12_000_000, token
                )
            with self.assertRaisesRegex(MakcuError, "session is aborted"):
                controller.request_calibration_pulse("x", 1, 1000.0, token)
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_coalesced_release_repress_aborts_before_write(self) -> None:
        controller, active, token = self._start_test_calibration()
        entered_ns = controller.raw_activation_snapshot.calibration_entered_ns
        press_ns = entered_ns + 2_000_000
        try:
            self._establish_calibration_hold(
                controller,
                active,
                press_ns=press_ns,
            )
            held_transition = (
                controller.raw_activation_snapshot.completed_press_transition_sequence
            )
            assert held_transition is not None
            controller.publish_calibration_lease(
                True,
                press_ns,
                token,
                activation_transition_sequence=held_transition,
            )
            controller.request_calibration_pulse("x", 10, 1000.0, token)
            before = controller.raw_activation_snapshot

            # Both reports are consumed in one serial read. The final level is
            # pressed, but the intervening release must still revoke the lease.
            self._queue_calibration_repress(active)
            controller._output_tick(0.001, now_ns=press_ns + 1_000_000)

            raw = controller.raw_activation_snapshot
            snapshot = controller.calibration_snapshot()
            self.assertTrue(raw.pressed)
            self.assertEqual(raw.report_sequence, before.report_sequence + 2)
            self.assertEqual(
                raw.transition_sequence,
                before.transition_sequence + 2,
            )
            self.assertEqual(snapshot.pending_counts, 0)
            self.assertEqual(snapshot.emitted_events, ())
            self.assertIn(
                "physical activation was released",
                snapshot.abort_reason or "",
            )
            self.assertEqual(self._movement_writes(active), ())
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_lease_rejects_changed_hold_transition(self) -> None:
        controller, active, token = self._start_test_calibration()
        entered_ns = controller.raw_activation_snapshot.calibration_entered_ns
        press_ns = entered_ns + 2_000_000
        try:
            self._establish_calibration_hold(
                controller,
                active,
                press_ns=press_ns,
            )
            expected = (
                controller.raw_activation_snapshot.completed_press_transition_sequence
            )
            assert expected is not None

            self._queue_calibration_repress(active)
            controller.poll_button_mask(now_ns=press_ns + 1_000_000)
            with self.assertRaisesRegex(MakcuError, "hold transition changed"):
                controller.publish_calibration_lease(
                    True,
                    press_ns + 1_000_000,
                    token,
                    activation_transition_sequence=expected,
                )

            snapshot = controller.calibration_snapshot()
            self.assertEqual(snapshot.emitted_events, ())
            self.assertEqual(snapshot.pending_counts, 0)
            self.assertIn("hold transition changed", snapshot.abort_reason or "")
            self.assertEqual(self._movement_writes(active), ())
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_lease_rejects_legacy_naked_button_proof(self) -> None:
        controller, active, token = self._start_test_calibration()
        entered_ns = controller.raw_activation_snapshot.calibration_entered_ns
        try:
            for offset_ms, payload in (
                (1, b"\x02"),
                (2, b"\x00"),
                (102, b"\x02"),
            ):
                active.responses.extend(payload)
                controller._output_tick(
                    0.0,
                    now_ns=entered_ns + offset_ms * 1_000_000,
                )

            raw = controller.raw_activation_snapshot
            self.assertTrue(raw.pressed)
            self.assertFalse(raw.post_entry_press_seen)
            self.assertEqual(
                raw.framed_report_sequence,
                raw.calibration_entry_framed_report_sequence,
            )
            with self.assertRaisesRegex(MakcuError, "fresh release/hold transition"):
                controller.publish_calibration_lease(
                    True,
                    entered_ns + 102_000_000,
                    token,
                )
            self.assertEqual(controller.calibration_snapshot().emitted_events, ())
            self.assertEqual(self._movement_writes(active), ())
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_stale_lease_aborts_before_any_motion(self) -> None:
        controller, active, token = self._start_test_calibration()
        base_ns = 60_000_000_000
        try:
            self._establish_calibration_hold(
                controller,
                active,
                press_ns=base_ns,
            )
            controller.publish_calibration_lease(True, base_ns, token)
            controller.request_calibration_pulse("x", 10, 1000.0, token)
            stale_ns = base_ns + round(
                CALIBRATION_LEASE_MAX_AGE_SECONDS * 1_000_000_000
            ) + 1
            controller._output_tick(0.001, now_ns=stale_ns)

            snapshot = controller.calibration_snapshot()
            self.assertEqual(snapshot.pending_counts, 0)
            self.assertEqual(snapshot.emitted_events, ())
            self.assertIn("lease expired", snapshot.abort_reason or "")
            self.assertEqual(self._movement_writes(active), ())
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_invalid_lease_atomically_cancels_pending_motion(self) -> None:
        controller, active, token = self._start_test_calibration()
        base_ns = 70_000_000_000
        try:
            self._establish_calibration_hold(
                controller,
                active,
                press_ns=base_ns,
            )
            controller.publish_calibration_lease(True, base_ns, token)
            controller.request_calibration_pulse("y", -10, 1000.0, token)
            controller.publish_calibration_lease(False, base_ns + 1, token)

            snapshot = controller.calibration_snapshot()
            self.assertEqual(snapshot.pending_counts, 0)
            self.assertEqual(snapshot.emitted_events, ())
            self.assertIn("lease was invalidated", snapshot.abort_reason or "")
            self._queue_calibration_repress(active)
            controller._output_tick(0.010, now_ns=base_ns + 10_000_000)
            self.assertEqual(self._movement_writes(active), ())
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_write_failure_is_excluded_from_evidence_and_aborts(self) -> None:
        controller, active, token = self._start_test_calibration()
        base_ns = 80_000_000_000
        try:
            self._establish_calibration_hold(
                controller,
                active,
                press_ns=base_ns,
            )
            controller.publish_calibration_lease(True, base_ns, token)
            controller.request_calibration_pulse("x", 10, 1000.0, token)
            with (
                mock.patch.object(
                    controller,
                    "_command",
                    side_effect=MakcuError("simulated serial failure"),
                ),
                self.assertRaisesRegex(MakcuError, "simulated serial failure"),
            ):
                controller._output_tick(0.001, now_ns=base_ns + 1_000_000)

            failed = controller.calibration_snapshot()
            self.assertEqual(failed.emitted_events, ())
            self.assertEqual(failed.emitted_abs_counts, 0)
            self.assertEqual(failed.movement_commands, 0)
            self.assertEqual(failed.pending_counts, 0)
            self.assertIn("movement write failed", failed.abort_reason or "")
            self.assertEqual(controller.telemetry_snapshot().movement_commands, 0)

            controller._output_tick(0.010, now_ns=base_ns + 11_000_000)
            self.assertEqual(self._movement_writes(active), ())
        finally:
            controller.exit_calibration_mode(token)
            controller._output_thread = None
            controller.stop()

    def test_calibration_stop_invalidates_token_and_restart_erases_evidence(self) -> None:
        controller, active, old_token = self._start_test_calibration()
        base_ns = 90_000_000_000
        self._establish_calibration_hold(
            controller,
            active,
            press_ns=base_ns,
        )
        controller.publish_calibration_lease(True, base_ns, old_token)
        controller.request_calibration_pulse("x", 10, 1000.0, old_token)
        controller._output_tick(0.001, now_ns=base_ns + 1_000_000)
        controller._output_thread = None
        controller.stop()

        stopped = controller.calibration_snapshot()
        self.assertFalse(stopped.active)
        self.assertEqual(stopped.pending_counts, 0)
        self.assertEqual(stopped.emitted_events, ((base_ns + 1_000_000, 1, 0),))
        self.assertIn("stopped during calibration", stopped.abort_reason or "")
        with self.assertRaisesRegex(MakcuError, "invalid or inactive"):
            controller.publish_calibration_lease(
                True, base_ns + 2_000_000, old_token
            )

        controller.start(output_loop=False)
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        controller._output_thread = live_worker
        new_token = controller.enter_calibration_mode()
        try:
            self.assertIsNot(new_token, old_token)
            self.assertEqual(
                controller.calibration_snapshot(),
                MakcuCalibrationSnapshot(active=True),
            )
            with self.assertRaisesRegex(MakcuError, "invalid or inactive"):
                controller.exit_calibration_mode(old_token)
        finally:
            controller.exit_calibration_mode(new_token)
            controller._output_thread = None
            controller.stop()

    def test_velocity_uses_detector_sample_times_and_persists_between_ticks(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=0.5,
                max_step=1000,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.03,
                derivative_damping_seconds=0.008,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        first_ns = 1_000_000_000
        first = Detection(0, "player", 0.9, (950, 540, 970, 640))
        second = Detection(0, "player", 0.9, (956.6, 540, 976.6, 640))
        controller.update(first, (1080, 1920, 3), measurement_ns=first_ns)
        controller._output_tick(0.001, now_ns=first_ns + 1_000_000)
        controller.update(
            second,
            (1080, 1920, 3),
            measurement_ns=first_ns + 33_000_000,
        )

        controller._output_tick(0.001, now_ns=first_ns + 34_000_000)
        first_tick_rate = controller._smoothed_rate_x
        controller._output_tick(0.001, now_ns=first_ns + 35_000_000)
        second_tick_rate = controller._smoothed_rate_x

        self.assertAlmostEqual(controller._latest_velocity_x, 200.0, delta=0.01)
        self.assertGreater(first_tick_rate, 0.0)
        self.assertLess(
            abs(second_tick_rate - first_tick_rate),
            abs(first_tick_rate) * 0.05,
        )

    def test_approaching_velocity_cannot_reduce_current_error_correction(self) -> None:
        cases = (
            (
                "horizontal",
                Detection(0, "player", 0.9, (1076, 540, 1096, 640)),
                Detection(0, "player", 0.9, (1050, 540, 1070, 640)),
                "_smoothed_rate_x",
            ),
            (
                "vertical",
                Detection(0, "player", 0.9, (950, 666, 970, 766)),
                Detection(0, "player", 0.9, (950, 640, 970, 740)),
                "_smoothed_rate_y",
            ),
        )
        for axis, earlier, current, rate_attribute in cases:
            with self.subTest(axis=axis):
                controller = self.controller(
                    MakcuAimConfig(
                        strength=0.5,
                        max_step=1000,
                        output_hz=1000,
                        deadzone_pixels=0.0,
                        smoothing_alpha=1.0,
                        prediction_lead_seconds=0.03,
                        derivative_damping_seconds=0.008,
                        head_ratio=0.0,
                    )
                )
                controller.start(output_loop=False)
                controller._output_thread = object()
                active = self.factory.connections[-1]
                active.responses.extend(bytes((0b00010,)))
                base_ns = 9_000_000_000
                try:
                    controller.update(
                        earlier,
                        (1080, 1920, 3),
                        measurement_ns=base_ns,
                    )
                    controller.update(
                        current,
                        (1080, 1920, 3),
                        measurement_ns=base_ns + 10_000_000,
                    )

                    controller._output_tick(
                        0.001,
                        now_ns=base_ns + 22_000_000,
                    )

                    self.assertAlmostEqual(
                        getattr(controller, "_latest_velocity_" + rate_attribute[-1]),
                        -2600.0,
                    )
                    # Pure proportional correction is 100px * .5 * 60Hz.
                    # Closing velocity may not brake that measured command;
                    # the small pursuit term can only increase it.
                    self.assertGreaterEqual(getattr(controller, rate_attribute), 3000.0)
                finally:
                    controller._output_thread = None
                    controller.stop()

    def test_pursuit_integral_is_time_based_persists_at_crossing_and_unwinds(self) -> None:
        def response_after(
            duration_seconds: float,
            rate_hz: int,
        ) -> tuple[float, float]:
            controller = self.controller(
                MakcuAimConfig(
                    strength=0.5,
                    max_step=1000,
                    output_hz=rate_hz,
                    deadzone_pixels=0.0,
                    smoothing_alpha=1.0,
                    prediction_lead_seconds=0.0,
                    derivative_damping_seconds=0.0,
                    head_ratio=0.0,
                )
            )
            controller.start(output_loop=False)
            controller._output_thread = object()
            active = self.factory.connections[-1]
            active.responses.extend(bytes((0b00010,)))
            base_ns = 10_000_000_000
            right = Detection(0, "player", 0.9, (1150, 540, 1170, 640))
            controller.update(right, (1080, 1920, 3), measurement_ns=base_ns)
            period = 1.0 / rate_hz
            try:
                controller._output_tick(0.0, now_ns=base_ns)
                detector_stride = rate_hz // 100
                for index in range(round(duration_seconds * rate_hz)):
                    measurement_ns = base_ns + round((index + 1) * period * 1e9)
                    if (index + 1) % detector_stride == 0:
                        controller.update(
                            right,
                            (1080, 1920, 3),
                            measurement_ns=measurement_ns,
                        )
                    controller._output_tick(
                        period,
                        now_ns=measurement_ns,
                    )
                return controller._pursuit_correction_x, controller._smoothed_rate_x
            finally:
                controller._output_thread = None
                controller.stop()

        responses: dict[float, list[tuple[float, float]]] = {}
        for duration_seconds in (0.05, 0.12):
            responses[duration_seconds] = [
                response_after(duration_seconds, rate_hz)
                for rate_hz in (100, 1000, 2000)
            ]

        # A 200px error at strength 0.5 contributes 100 correction units.
        # The pursuit term reaches 50/120 of that after 50ms and the full
        # proportional amount after the configured 120ms buildup.
        expected_integrals = {0.05: 100.0 * 0.05 / 0.12, 0.12: 100.0}
        for duration_seconds, duration_responses in responses.items():
            expected_integral = expected_integrals[duration_seconds]
            reference_integral, reference_rate = duration_responses[0]
            self.assertAlmostEqual(reference_integral, expected_integral)
            for integral, rate in duration_responses[1:]:
                self.assertAlmostEqual(integral, reference_integral)
                self.assertAlmostEqual(rate, reference_rate, delta=1.0)
        self.assertGreater(responses[0.05][-1][1], 200.0 * 0.5 * 60.0)

        controller = self.controller(
            MakcuAimConfig(
                strength=1.0,
                max_step=1000,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        base_ns = 11_000_000_000
        right = Detection(0, "player", 0.9, (1050, 540, 1070, 640))
        left = Detection(0, "player", 0.9, (940, 540, 960, 640))
        far_left = Detection(0, "player", 0.9, (850, 540, 870, 640))
        try:
            controller.update(right, (1080, 1920, 3), measurement_ns=base_ns)
            controller._output_tick(0.0, now_ns=base_ns)
            for index in range(100):
                measurement_ns = base_ns + (index + 1) * 1_000_000
                if (index + 1) % 10 == 0:
                    controller.update(
                        right,
                        (1080, 1920, 3),
                        measurement_ns=measurement_ns,
                    )
                controller._output_tick(
                    0.001,
                    now_ns=measurement_ns,
                )
            retained = controller._pursuit_correction_x
            resets_before_crossing = controller.telemetry_snapshot().pursuit_resets
            self.assertGreater(retained, 0.0)
            controller.update(
                left,
                (1080, 1920, 3),
                measurement_ns=base_ns + 101_000_000,
            )
            controller._output_tick(0.001, now_ns=base_ns + 102_000_000)
            # One ordinary center crossing is evidence that the learned rate
            # should start unwinding, not evidence that it should disappear.
            # The retained internal rate must neither produce wrong-way output
            # nor suppress the fresh positional term on the new side. The
            # exact fallback includes the allowed 1 ms same-direction sample-
            # age assist: (-10px + -2600px/s * .001s) * 60Hz.
            self.assertGreater(controller._pursuit_correction_x, 0.0)
            self.assertLess(controller._pursuit_correction_x, retained)
            expected_reversal_rate = (
                controller._control_error_x
                + controller._latest_velocity_x * 0.001
            ) * controller.config.strength * 60.0
            self.assertAlmostEqual(
                controller._smoothed_rate_x,
                expected_reversal_rate,
            )
            self.assertLess(controller._smoothed_rate_x, 0.0)
            self.assertEqual(
                controller.telemetry_snapshot().pursuit_resets,
                resets_before_crossing,
            )

            # Sustained fresh error on the other side naturally unwinds and
            # then reverses the learned rate without an explicit reset.
            for index in range(11):
                measurement_ns = base_ns + (111 + index * 10) * 1_000_000
                controller.update(
                    far_left,
                    (1080, 1920, 3),
                    measurement_ns=measurement_ns,
                )
                controller._output_tick(0.01, now_ns=measurement_ns)
            self.assertLess(controller._pursuit_correction_x, 0.0)
            self.assertLess(controller._smoothed_rate_x, 0.0)
            controller.update(
                None,
                (1080, 1920, 3),
                measurement_ns=base_ns + 222_000_000,
            )
            controller._output_tick(0.001, now_ns=base_ns + 222_000_000)
            self.assertEqual(controller._pursuit_correction_x, 0.0)
            self.assertEqual(controller._pursuit_correction_y, 0.0)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_pursuit_integral_respects_ratio_and_remaining_headroom(self) -> None:
        cases = (
            (
                "half-limit cap",
                Detection(0, "player", 0.9, (970, 540, 990, 560)),
                0.40,
                50.0,
                70.0 * 60.0,
            ),
            (
                "remaining headroom",
                Detection(0, "player", 0.9, (1040, 540, 1060, 560)),
                0.02,
                10.0,
                100.0 * 60.0,
            ),
        )
        for name, target, duration_seconds, expected_integral, expected_rate in cases:
            with self.subTest(name=name):
                controller = self.controller(
                    MakcuAimConfig(
                        strength=1.0,
                        max_step=100,
                        output_hz=100,
                        deadzone_pixels=0.0,
                        smoothing_alpha=1.0,
                        prediction_lead_seconds=0.0,
                        derivative_damping_seconds=0.0,
                        head_ratio=0.0,
                    )
                )
                controller.start(output_loop=False)
                controller._output_thread = object()
                active = self.factory.connections[-1]
                active.responses.extend(bytes((0b00010,)))
                base_ns = 10_500_000_000
                controller.update(
                    target,
                    (1080, 1920, 3),
                    measurement_ns=base_ns,
                )
                try:
                    controller._output_tick(0.0, now_ns=base_ns)
                    steps = round(duration_seconds * 100)
                    for index in range(steps):
                        measurement_ns = base_ns + (index + 1) * 10_000_000
                        controller.update(
                            target,
                            (1080, 1920, 3),
                            measurement_ns=measurement_ns,
                        )
                        controller._output_tick(
                            0.01,
                            now_ns=measurement_ns,
                        )

                    self.assertAlmostEqual(
                        controller._pursuit_correction_x,
                        expected_integral,
                    )
                    self.assertAlmostEqual(controller._smoothed_rate_x, expected_rate)
                finally:
                    controller._output_thread = None
                    controller.stop()

    def test_centered_measurement_retains_learned_rate_without_reset(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=1.0,
                max_step=1000,
                output_hz=1000,
                deadzone_pixels=2.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        base_ns = 10_800_000_000
        right = Detection(0, "player", 0.9, (1050, 540, 1070, 640))
        centered = Detection(0, "player", 0.9, (950, 540, 970, 640))
        try:
            controller.update(right, (1080, 1920, 3), measurement_ns=base_ns)
            controller._output_tick(0.0, now_ns=base_ns)
            for index in range(1, 7):
                measurement_ns = base_ns + index * 10_000_000
                controller.update(
                    right,
                    (1080, 1920, 3),
                    measurement_ns=measurement_ns,
                )
                controller._output_tick(0.01, now_ns=measurement_ns)

            retained = controller._pursuit_correction_x
            resets_before_center = controller.telemetry_snapshot().pursuit_resets
            self.assertGreater(retained, 0.0)
            centered_ns = base_ns + 70_000_000
            controller.update(
                centered,
                (1080, 1920, 3),
                measurement_ns=centered_ns,
            )
            controller._output_tick(0.01, now_ns=centered_ns)

            expected = retained * math.exp(
                -0.01 / PURSUIT_DEADZONE_LEAK_TIME_SECONDS
            )
            self.assertAlmostEqual(controller._pursuit_correction_x, expected)
            self.assertAlmostEqual(controller._smoothed_rate_x, expected * 60.0)
            self.assertGreater(controller._smoothed_rate_x, 0.0)
            self.assertEqual(
                controller.telemetry_snapshot().pursuit_resets,
                resets_before_center,
            )
        finally:
            controller._output_thread = None
            controller.stop()

    def test_deadzone_pursuit_leak_is_detector_rate_invariant(self) -> None:
        def retained_after(rate_hz: int, duration_seconds: float) -> float:
            controller = self.controller(
                MakcuAimConfig(
                    strength=1.0,
                    max_step=1000,
                    output_hz=1000,
                    deadzone_pixels=2.0,
                    smoothing_alpha=1.0,
                    prediction_lead_seconds=0.0,
                    derivative_damping_seconds=0.0,
                    head_ratio=0.0,
                )
            )
            controller.start(output_loop=False)
            controller._output_thread = object()
            active = self.factory.connections[-1]
            active.responses.extend(bytes((0b00010,)))
            centered = Detection(0, "player", 0.9, (950, 540, 970, 640))
            base_ns = 10_900_000_000
            try:
                controller.update(
                    centered,
                    (1080, 1920, 3),
                    measurement_ns=base_ns,
                )
                controller._output_tick(0.0, now_ns=base_ns)
                controller._pursuit_correction_x = 70.0
                period = 1.0 / rate_hz
                steps = round(duration_seconds * rate_hz)
                for index in range(1, steps + 1):
                    measurement_ns = base_ns + round(index * period * 1e9)
                    controller.update(
                        centered,
                        (1080, 1920, 3),
                        measurement_ns=measurement_ns,
                    )
                    controller._output_tick(period, now_ns=measurement_ns)
                return controller._pursuit_correction_x
            finally:
                controller._output_thread = None
                controller.stop()

        duration_seconds = 0.40
        expected = 70.0 * math.exp(
            -duration_seconds / PURSUIT_DEADZONE_LEAK_TIME_SECONDS
        )
        results = [retained_after(rate, duration_seconds) for rate in (50, 100, 200)]
        for result in results:
            self.assertAlmostEqual(result, expected, places=10)
        self.assertAlmostEqual(max(results), min(results), places=10)

    def test_persistent_pursuit_reduces_constant_velocity_fake_plant_rms(self) -> None:
        def rms_error(integral_time_seconds: float) -> float:
            _before, after, _commands, _measurements = (
                self._run_horizontal_fake_plant(
                    lambda _tick: 800.0,
                    ticks=1500,
                    integral_time_seconds=integral_time_seconds,
                )
            )
            steady_errors = after[500:]
            return math.sqrt(
                sum(error * error for error in steady_errors)
                / len(steady_errors)
            )

        persistent_rms = rms_error(0.12)
        proportional_only_rms = rms_error(1e12)

        self.assertLess(persistent_rms, 10.0)
        self.assertGreater(proportional_only_rms, 90.0)
        self.assertLess(persistent_rms, proportional_only_rms * 0.10)

    def test_abrupt_stop_has_bounded_wrong_way_fake_plant_displacement(self) -> None:
        before, after, commands, measured = self._run_horizontal_fake_plant(
            lambda tick: 800.0 if tick < 750 else 0.0,
            ticks=2500,
        )
        with mock.patch(
            "aiming.makcu._combine_pursuit_correction",
            side_effect=self._legacy_opposing_pursuit_stall,
        ):
            _legacy_before, legacy_after, legacy_commands, legacy_measured = (
                self._run_horizontal_fake_plant(
                    lambda tick: 800.0 if tick < 750 else 0.0,
                    ticks=2500,
                )
            )

        # No command may oppose the latest fresh measurement once that
        # measurement is outside the deadzone.
        for command, error in zip(commands, measured):
            if abs(error) > 2.0:
                self.assertGreaterEqual(command * error, 0.0)

        wrong_way_burst = 0.0
        maximum_wrong_way_burst = 0.0
        for command, actual_error in zip(commands[750:], before[750:]):
            if abs(actual_error) > 2.0 and command * actual_error < 0.0:
                wrong_way_burst += abs(command) * 0.10
            else:
                maximum_wrong_way_burst = max(
                    maximum_wrong_way_burst,
                    wrong_way_burst,
                )
                wrong_way_burst = 0.0
        maximum_wrong_way_burst = max(maximum_wrong_way_burst, wrong_way_burst)

        # Only commands already authorized by the preceding 8 ms detector
        # sample can carry across the true zero crossing.
        self.assertLessEqual(maximum_wrong_way_burst, 5.0)
        self.assertGreaterEqual(min(after[750:]), -8.0)
        self.assertLess(abs(after[-1]), 4.0)

        # The old zero fallback could remain motionless against a fresh error
        # for the rest of the run. The base fallback never stalls longer than
        # one detector interval and roughly halves post-stop integrated error.
        current_zero_run = self._maximum_fresh_zero_run(
            commands,
            measured,
            start=750,
        )
        legacy_zero_run = self._maximum_fresh_zero_run(
            legacy_commands,
            legacy_measured,
            start=750,
        )
        self.assertLessEqual(current_zero_run, 8)
        self.assertGreater(legacy_zero_run, 1000)
        current_integrated_error = sum(abs(error) for error in after[750:])
        legacy_integrated_error = sum(
            abs(error) for error in legacy_after[750:]
        )
        self.assertLess(
            current_integrated_error,
            legacy_integrated_error * 0.60,
        )

    def test_target_reversal_has_bounded_wrong_way_fake_plant_displacement(self) -> None:
        before, after, commands, measured = self._run_horizontal_fake_plant(
            lambda tick: 800.0 if tick < 750 else -800.0,
            ticks=2500,
        )
        with mock.patch(
            "aiming.makcu._combine_pursuit_correction",
            side_effect=self._legacy_opposing_pursuit_stall,
        ):
            _legacy_before, legacy_after, legacy_commands, legacy_measured = (
                self._run_horizontal_fake_plant(
                    lambda tick: 800.0 if tick < 750 else -800.0,
                    ticks=2500,
                )
            )

        for command, error in zip(commands, measured):
            if abs(error) > 2.0:
                self.assertGreaterEqual(command * error, 0.0)

        wrong_way_burst = 0.0
        maximum_wrong_way_burst = 0.0
        for command, actual_error in zip(commands[750:], before[750:]):
            if abs(actual_error) > 2.0 and command * actual_error < 0.0:
                wrong_way_burst += abs(command) * 0.10
            else:
                maximum_wrong_way_burst = max(
                    maximum_wrong_way_burst,
                    wrong_way_burst,
                )
                wrong_way_burst = 0.0
        maximum_wrong_way_burst = max(maximum_wrong_way_burst, wrong_way_burst)

        self.assertLessEqual(maximum_wrong_way_burst, 6.0)
        self.assertGreaterEqual(min(after[750:]), -101.0)
        self.assertLess(abs(after[-1]), 3.0)

        current_zero_run = self._maximum_fresh_zero_run(
            commands,
            measured,
            start=750,
        )
        legacy_zero_run = self._maximum_fresh_zero_run(
            legacy_commands,
            legacy_measured,
            start=750,
        )
        self.assertLessEqual(current_zero_run, 8)
        self.assertGreaterEqual(legacy_zero_run, 80)
        current_initial_error = sum(abs(error) for error in after[750:900])
        legacy_initial_error = sum(
            abs(error) for error in legacy_after[750:900]
        )
        self.assertLess(current_initial_error, legacy_initial_error * 0.80)
        self.assertGreater(
            min(after[750:]),
            min(legacy_after[750:]) + 5.0,
        )

    def test_gate_loss_resets_pursuit_and_emits_no_movement(self) -> None:
        config = MakcuAimConfig(
            strength=1.0,
            max_step=1000,
            output_hz=1000,
            deadzone_pixels=0.0,
            smoothing_alpha=1.0,
            prediction_lead_seconds=0.0,
            derivative_damping_seconds=0.0,
            head_ratio=0.0,
        )
        target = Detection(0, "player", 0.9, (1050, 540, 1070, 640))

        for gate_loss in ("physical release", "inactive input"):
            with self.subTest(gate_loss=gate_loss):
                controller = self.controller(config)
                controller.start(output_loop=False)
                controller._output_thread = object()
                active = self.factory.connections[-1]
                active.responses.extend(bytes((0b00010,)))
                base_ns = time.perf_counter_ns()
                try:
                    controller.update(
                        target,
                        (1080, 1920, 3),
                        measurement_ns=base_ns,
                    )
                    controller._output_tick(0.0, now_ns=base_ns)
                    controller.update(
                        target,
                        (1080, 1920, 3),
                        measurement_ns=base_ns + 20_000_000,
                    )
                    controller._output_tick(
                        0.02,
                        now_ns=base_ns + 20_000_000,
                    )
                    self.assertGreater(controller._pursuit_correction_x, 0.0)
                    movement_count = sum(
                        write.startswith(b"km.move(") for write in active.writes
                    )

                    if gate_loss == "physical release":
                        active.responses.extend(bytes((0,)))
                    else:
                        controller.update(
                            target,
                            (1080, 1920, 3),
                            active=False,
                            measurement_ns=base_ns + 21_000_000,
                        )
                    controller._output_tick(
                        0.001,
                        now_ns=base_ns + 21_000_000,
                    )

                    self.assertEqual(controller._pursuit_correction_x, 0.0)
                    self.assertEqual(controller._pursuit_correction_y, 0.0)
                    self.assertEqual(
                        sum(write.startswith(b"km.move(") for write in active.writes),
                        movement_count,
                    )
                finally:
                    controller._output_thread = None
                    controller.stop()

    def test_stop_and_restart_clear_pursuit_memory(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=1.0,
                max_step=1000,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        target = Detection(0, "player", 0.9, (1050, 540, 1070, 640))
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        base_ns = 11_500_000_000
        controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)
        controller._output_tick(0.0, now_ns=base_ns)
        controller.update(
            target,
            (1080, 1920, 3),
            measurement_ns=base_ns + 20_000_000,
        )
        controller._output_tick(0.02, now_ns=base_ns + 20_000_000)
        self.assertGreater(controller._pursuit_correction_x, 0.0)

        controller._output_thread = None
        controller.stop()
        self.assertEqual(controller._pursuit_correction_x, 0.0)
        self.assertEqual(controller._pursuit_correction_y, 0.0)
        self.assertEqual(controller._pursuit_measurement_ns, 0)

        controller.start(output_loop=False)
        try:
            self.assertEqual(controller._pursuit_correction_x, 0.0)
            self.assertEqual(controller._pursuit_correction_y, 0.0)
            self.assertEqual(controller._pursuit_measurement_ns, 0)
        finally:
            controller.stop()

    def test_prediction_gap_preserves_pursuit_until_tracker_grace_expires(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=1.0,
                max_step=1000,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        tracker = TargetTracker(
            label="player",
            head_ratio=0.0,
            lost_grace_frames=1,
        )
        target = Detection(0, "player", 0.9, (1050, 540, 1070, 640))
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        base_ns = time.perf_counter_ns()
        try:
            first = tracker.update(
                [target],
                (1080, 1920, 3),
                measurement_ns=base_ns,
            )
            controller.update(first, (1080, 1920, 3), measurement_ns=base_ns)
            controller._output_tick(0.0, now_ns=base_ns)

            measured_ns = base_ns + 8_000_000
            measured = tracker.update(
                [target],
                (1080, 1920, 3),
                measurement_ns=measured_ns,
            )
            controller.update(
                measured,
                (1080, 1920, 3),
                measurement_ns=measured_ns,
            )
            controller._output_tick(0.008, now_ns=measured_ns)
            accumulated = controller._pursuit_correction_x
            self.assertGreater(accumulated, 0.0)
            resets_before_gap = controller.telemetry_snapshot().pursuit_resets
            control_samples_before_gap = (
                controller.telemetry_snapshot().control_samples
            )

            bridged_ns = base_ns + 16_000_000
            bridged = tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=bridged_ns,
            )
            self.assertIsNotNone(bridged)
            controller.update(
                bridged,
                (1080, 1920, 3),
                measurement_ns=bridged_ns,
                measurement_observed=False,
            )
            controller._output_tick(0.008, now_ns=bridged_ns)
            self.assertEqual(controller._pursuit_correction_x, accumulated)
            self.assertEqual(
                controller.telemetry_snapshot().pursuit_resets,
                resets_before_gap,
            )
            self.assertEqual(
                controller.telemetry_snapshot().control_samples,
                control_samples_before_gap,
            )

            expired_ns = measured_ns + 16_666_668
            expired = tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=expired_ns,
            )
            self.assertIsNone(expired)
            controller.update(
                expired,
                (1080, 1920, 3),
                measurement_ns=expired_ns,
            )
            controller._output_tick(0.001, now_ns=expired_ns)
            self.assertEqual(controller._pursuit_correction_x, 0.0)
            self.assertEqual(
                controller.telemetry_snapshot().pursuit_resets,
                resets_before_gap + 1,
            )
        finally:
            controller._output_thread = None
            controller.stop()

    def test_physical_release_during_predicted_gap_emits_no_movement(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=1.0,
                max_step=1000,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        tracker = TargetTracker(
            label="player",
            head_ratio=0.0,
            lost_grace_frames=1,
        )
        target = Detection(0, "player", 0.9, (1050, 540, 1070, 640))
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        base_ns = time.perf_counter_ns()
        try:
            measured = tracker.update(
                [target],
                (1080, 1920, 3),
                measurement_ns=base_ns,
            )
            controller.update(
                measured,
                (1080, 1920, 3),
                measurement_ns=base_ns,
            )
            controller._output_tick(0.008, now_ns=base_ns)
            movement_count = sum(
                write.startswith(b"km.move(") for write in active.writes
            )

            predicted_ns = base_ns + 8_000_000
            predicted = tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=predicted_ns,
            )
            self.assertIsNotNone(predicted)
            controller.update(
                predicted,
                (1080, 1920, 3),
                measurement_ns=predicted_ns,
                measurement_observed=False,
            )
            active.responses.extend(bytes((0,)))
            controller._output_tick(0.008, now_ns=predicted_ns)

            self.assertFalse(controller.activation_pressed)
            self.assertEqual(
                sum(write.startswith(b"km.move(") for write in active.writes),
                movement_count,
            )
            self.assertEqual(controller._pursuit_correction_x, 0.0)
            self.assertEqual(controller._pursuit_correction_y, 0.0)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_held_detector_sample_does_not_grow_pursuit_integral(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=1.0,
                max_step=1000,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        target = Detection(0, "player", 0.9, (1050, 540, 1070, 640))
        base_ns = time.perf_counter_ns()
        try:
            controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)
            controller._output_tick(0.0, now_ns=base_ns)
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=base_ns + 10_000_000,
            )
            controller._output_tick(0.01, now_ns=base_ns + 10_000_000)
            accumulated = controller._pursuit_correction_x
            self.assertGreater(accumulated, 0.0)

            for offset_ms in range(11, 141):
                controller._output_tick(
                    0.001,
                    now_ns=base_ns + offset_ms * 1_000_000,
                )

            self.assertEqual(controller._pursuit_correction_x, accumulated)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_confidence_does_not_scale_pure_proportional_rate(self) -> None:
        def horizontal_rate(confidence: float) -> float:
            controller = self.controller(
                MakcuAimConfig(
                    strength=0.5,
                    max_step=1000,
                    output_hz=1000,
                    deadzone_pixels=0.0,
                    smoothing_alpha=1.0,
                    prediction_lead_seconds=0.0,
                    derivative_damping_seconds=0.0,
                    head_ratio=0.0,
                )
            )
            controller.start(output_loop=False)
            controller._output_thread = object()
            active = self.factory.connections[-1]
            active.responses.extend(bytes((0b00010,)))
            target = Detection(0, "player", confidence, (1150, 540, 1170, 640))
            base_ns = time.perf_counter_ns()
            try:
                controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)
                controller._output_tick(0.001, now_ns=base_ns + 1_000_000)
                return controller._smoothed_rate_x
            finally:
                controller._output_thread = None
                controller.stop()

        low_confidence_rate = horizontal_rate(0.05)
        high_confidence_rate = horizontal_rate(0.99)
        self.assertEqual(low_confidence_rate, high_confidence_rate)
        self.assertEqual(low_confidence_rate, 200.0 * 0.5 * 60.0)

    def test_small_static_error_receives_full_proportional_rate(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=1.0,
                max_step=1000,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        # A 64px error is below the removed 130px close-range slowdown band.
        target = Detection(0, "player", 0.9, (1014, 540, 1034, 640))
        base_ns = time.perf_counter_ns()
        try:
            controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)
            controller._output_tick(0.001, now_ns=base_ns + 1_000_000)

            self.assertEqual(controller._smoothed_rate_x, 64.0 * 1.0 * 60.0)
        finally:
            controller._output_thread = None
            controller.stop()

    def test_old_source_measurement_expires_even_if_just_published(self) -> None:
        controller = self.controller(MakcuAimConfig(output_hz=1000))
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        target = Detection(0, "player", 0.9, (1060, 480, 1260, 980))
        now_ns = 3_000_000_000
        controller.update(
            target,
            (1080, 1920, 3),
            measurement_ns=now_ns - round((TARGET_STALE_SECONDS + 0.01) * 1e9),
        )
        before = len(active.writes)

        controller._output_tick(0.001, now_ns=now_ns)

        self.assertEqual(len(active.writes), before)

    def test_smoothing_response_is_invariant_to_output_rate(self) -> None:
        alpha = 0.78

        def response_after_10ms(rate_hz: int) -> float:
            controller = self.controller(
                MakcuAimConfig(
                    strength=0.5,
                    max_step=1000,
                    output_hz=rate_hz,
                    smoothing_alpha=alpha,
                    prediction_lead_seconds=0.0,
                    derivative_damping_seconds=0.0,
                    head_ratio=0.0,
                )
            )
            controller.start(output_loop=False)
            controller._output_thread = object()
            active = self.factory.connections[-1]
            active.responses.extend(bytes((0b00010,)))
            target = Detection(0, "player", 0.9, (1150, 540, 1170, 640))
            base_ns = 2_000_000_000
            controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)
            period = 1.0 / rate_hz
            for index in range(round(0.01 * rate_hz)):
                controller._output_tick(
                    period,
                    now_ns=base_ns + round((index + 1) * period * 1e9),
                )
            return controller._smoothed_rate_x

        # Isolate the smoothing filter from the independently tested pursuit
        # ramp so this remains a strict output-rate-invariance check.
        with mock.patch(
            "aiming.makcu.PURSUIT_INTEGRAL_TIME_SECONDS",
            1e12,
        ):
            self.assertAlmostEqual(
                response_after_10ms(100),
                response_after_10ms(1000),
                delta=1.0,
            )

    def test_vertical_motion_is_time_based_across_output_rates(self) -> None:
        def movement_after_50ms(
            rate_hz: int,
            vertical_rate_ratio: float | None = None,
        ) -> int:
            ratio_options = (
                {}
                if vertical_rate_ratio is None
                else {"vertical_rate_ratio": vertical_rate_ratio}
            )
            controller = self.controller(
                MakcuAimConfig(
                    strength=1.0,
                    max_step=160,
                    output_hz=rate_hz,
                    deadzone_pixels=0.0,
                    smoothing_alpha=1.0,
                    prediction_lead_seconds=0.0,
                    derivative_damping_seconds=0.0,
                    head_ratio=0.0,
                    **ratio_options,
                )
            )
            controller.start(output_loop=False)
            controller._output_thread = object()
            active = self.factory.connections[-1]
            active.responses.extend(bytes((0b00010,)))
            target = Detection(0, "player", 0.9, (955, 900, 965, 920))
            base_ns = 4_000_000_000
            controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)
            period = 1.0 / rate_hz
            for index in range(round(0.05 * rate_hz)):
                controller._output_tick(
                    period,
                    now_ns=base_ns + round((index + 1) * period * 1e9),
                )
            movement = 0
            for write in active.writes:
                if not write.startswith(b"km.move("):
                    continue
                _x, y = write.decode("ascii").strip()[8:-1].split(",")
                movement += int(y)
            controller._output_thread = None
            controller.stop()
            return movement

        self.assertAlmostEqual(
            movement_after_50ms(100),
            movement_after_50ms(1000),
            delta=1,
        )
        default_movement = movement_after_50ms(1000)
        self.assertEqual(
            default_movement,
            movement_after_50ms(1000, MAX_VERTICAL_RATE_RATIO),
        )
        self.assertAlmostEqual(
            movement_after_50ms(1000, 1.0),
            default_movement / MAX_VERTICAL_RATE_RATIO,
            delta=2,
        )
        self.assertAlmostEqual(
            movement_after_50ms(1000, MAX_VERTICAL_RATE_RATIO / 2.0),
            default_movement / 2.0,
            delta=2,
        )

    def test_saturated_horizontal_motion_is_time_based_across_output_rates(self) -> None:
        def movement_after_100ms(rate_hz: int) -> int:
            controller = self.controller(
                MakcuAimConfig(
                    strength=4.0,
                    max_step=160,
                    output_hz=rate_hz,
                    deadzone_pixels=0.0,
                    smoothing_alpha=1.0,
                    prediction_lead_seconds=0.0,
                    derivative_damping_seconds=0.0,
                    head_ratio=0.0,
                )
            )
            controller.start(output_loop=False)
            controller._output_thread = object()
            active = self.factory.connections[-1]
            active.responses.extend(bytes((0b00010,)))
            target = Detection(0, "player", 0.9, (1800, 540, 1820, 560))
            base_ns = 6_000_000_000
            controller.update(target, (1080, 1920, 3), measurement_ns=base_ns)
            period = 1.0 / rate_hz
            for index in range(round(0.1 * rate_hz)):
                controller._output_tick(
                    period,
                    now_ns=base_ns + round((index + 1) * period * 1e9),
                )
            movement = 0
            for write in active.writes:
                if not write.startswith(b"km.move("):
                    continue
                x, _y = write.decode("ascii").strip()[8:-1].split(",")
                movement += int(x)
            controller._output_thread = None
            controller.stop()
            return movement

        movements = [movement_after_100ms(rate) for rate in (100, 1000, 2000)]
        self.assertLessEqual(max(movements) - min(movements), 1)

    def test_positive_target_velocity_never_produces_reverse_center_correction(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=0.5,
                max_step=1000,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.028,
                derivative_damping_seconds=0.016,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        first = Detection(0, "player", 0.9, (943.4, 540, 963.4, 560))
        centered = Detection(0, "player", 0.9, (950, 540, 970, 560))
        base_ns = 7_000_000_000
        controller.update(first, (1080, 1920, 3), measurement_ns=base_ns)
        controller._output_tick(0.001, now_ns=base_ns + 1_000_000)
        controller.update(
            centered,
            (1080, 1920, 3),
            measurement_ns=base_ns + 33_000_000,
        )

        controller._output_tick(0.001, now_ns=base_ns + 34_000_000)

        self.assertGreater(controller._latest_velocity_x, 0.0)
        self.assertGreaterEqual(controller._smoothed_rate_x, 0.0)

    def test_clamped_delayed_tick_does_not_replay_integer_backlog(self) -> None:
        controller = self.controller(
            MakcuAimConfig(
                strength=4.0,
                max_step=10,
                output_hz=1000,
                deadzone_pixels=0.0,
                smoothing_alpha=1.0,
                prediction_lead_seconds=0.0,
                derivative_damping_seconds=0.0,
                head_ratio=0.0,
            )
        )
        controller.start(output_loop=False)
        controller._output_thread = object()
        active = self.factory.connections[-1]
        active.responses.extend(bytes((0b00010,)))
        far = Detection(0, "player", 0.9, (1800, 540, 1820, 560))
        centered = Detection(0, "player", 0.9, (950, 540, 970, 560))
        base_ns = 8_000_000_000
        controller.update(far, (1080, 1920, 3), measurement_ns=base_ns)
        controller._output_tick(0.01, now_ns=base_ns + 10_000_000)
        first_moves = [write for write in active.writes if write.startswith(b"km.move(")]
        controller.update(
            centered,
            (1080, 1920, 3),
            measurement_ns=base_ns + 11_000_000,
        )
        controller._output_tick(0.001, now_ns=base_ns + 12_000_000)
        all_moves = [write for write in active.writes if write.startswith(b"km.move(")]

        self.assertEqual(len(all_moves), len(first_moves))
        self.assertLess(abs(controller._fractional_x), 1.0)

    def test_large_target_jump_resets_sample_velocity(self) -> None:
        controller = self.controller(MakcuAimConfig(head_ratio=0.0))
        controller.start(output_loop=False)
        first = Detection(0, "player", 0.9, (950, 540, 970, 640))
        switched = Detection(0, "player", 0.9, (1290, 540, 1310, 640))

        controller.update(first, (1080, 1920, 3), measurement_ns=1_000_000_000)
        controller.update(
            switched,
            (1080, 1920, 3),
            measurement_ns=1_033_000_000,
        )

        self.assertEqual(controller._latest_velocity_x, 0.0)
        self.assertEqual(controller._latest_velocity_y, 0.0)

    def test_measurement_timestamp_requires_strict_integer_type(self) -> None:
        controller = self.controller()
        controller.start(output_loop=False)
        target = Detection(0, "player", 0.9, (950, 540, 970, 640))

        for invalid in (True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                TypeError,
                "integer monotonic timestamp",
            ):
                controller.update(
                    target,
                    (1080, 1920, 3),
                    measurement_ns=invalid,
                )

    def test_measurement_observed_requires_bool_and_a_predicted_target(self) -> None:
        controller = self.controller()
        controller.start(output_loop=False)
        target = Detection(0, "player", 0.9, (950, 540, 970, 640))

        with self.assertRaisesRegex(TypeError, "measurement_observed must be bool"):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_observed=1,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "requires a predicted target"):
            controller.update(
                None,
                (1080, 1920, 3),
                measurement_observed=False,
            )
        with self.assertRaisesRegex(ValueError, "requires a prior observed target"):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_observed=False,
            )
        with self.assertRaisesRegex(
            ValueError,
            "velocity target requires an observed position target",
        ):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_observed=False,
                velocity_target=target,
            )
        self.assertIsNone(controller._latest_target)
        self.assertFalse(controller._measurement_target_present)

        controller.update(target, (1080, 1920, 3), measurement_ns=1_000_000_000)
        controller.update(None, (1080, 1920, 3), measurement_ns=1_010_000_000)
        with self.assertRaisesRegex(ValueError, "requires a prior observed target"):
            controller.update(
                target,
                (1080, 1920, 3),
                measurement_ns=1_020_000_000,
                measurement_observed=False,
            )
        self.assertIsNone(controller._latest_target)
        self.assertFalse(controller._measurement_target_present)
        self.assertEqual(controller._latest_source_ns, 1_010_000_000)

    def test_backwards_measurement_timestamp_cannot_overwrite_newer_target(self) -> None:
        controller = self.controller(MakcuAimConfig(head_ratio=0.0))
        controller.start(output_loop=False)
        first = Detection(0, "player", 0.9, (950, 540, 970, 640))
        newer = Detection(0, "player", 0.9, (1050, 540, 1070, 640))
        late_old = Detection(0, "player", 0.9, (970, 540, 990, 640))
        controller.update(first, (1080, 1920, 3), measurement_ns=5_000_000_000)
        controller.update(newer, (1080, 1920, 3), measurement_ns=5_100_000_000)

        with self.assertRaisesRegex(ValueError, "must not move backwards"):
            controller.update(
                late_old,
                (1080, 1920, 3),
                measurement_ns=5_050_000_000,
            )

        self.assertIs(controller._latest_target, newer)
        self.assertEqual(controller._latest_measurement_ns, 5_100_000_000)

    def test_changed_geometry_at_same_timestamp_resets_velocity(self) -> None:
        controller = self.controller(MakcuAimConfig(head_ratio=0.0))
        controller.start(output_loop=False)
        first = Detection(0, "player", 0.9, (950, 540, 970, 640))
        moving = Detection(0, "player", 0.9, (970, 540, 990, 640))
        duplicate_time_change = Detection(0, "player", 0.9, (990, 540, 1010, 640))
        controller.update(first, (1080, 1920, 3), measurement_ns=5_000_000_000)
        controller.update(moving, (1080, 1920, 3), measurement_ns=5_100_000_000)
        self.assertGreater(controller._latest_velocity_x, 0.0)

        controller.update(
            duplicate_time_change,
            (1080, 1920, 3),
            measurement_ns=5_100_000_000,
        )

        self.assertEqual(controller._latest_velocity_x, 0.0)
        self.assertEqual(controller._latest_velocity_y, 0.0)

    def test_target_delta_uses_head_point_deadzone_and_maximum_step(self) -> None:
        config = MakcuAimConfig(strength=1.0, max_step=20, deadzone_pixels=2.0)
        far = Detection(0, "player", 0.9, (1800, 900, 1920, 1080))
        centered = Detection(0, "player", 0.9, (958, 538, 962, 558))
        self.assertEqual(makcu_target_delta(far, (1080, 1920, 3), config), (20, 20))
        self.assertEqual(makcu_target_delta(centered, (1080, 1920, 3), config), (0, 0))

    def test_target_delta_is_resolution_invariant_for_normalized_offset(self) -> None:
        config = MakcuAimConfig(
            strength=0.5,
            max_step=1000,
            deadzone_pixels=0.0,
            head_ratio=0.0,
        )
        deltas = []
        for height, width in ((720, 1280), (1080, 1920), (2160, 3840)):
            target_x = width * 0.60
            target_y = height * 0.40
            target = Detection(
                0,
                "player",
                0.9,
                (target_x - 5, target_y, target_x + 5, target_y + 20),
            )
            deltas.append(
                makcu_target_delta(target, (height, width, 3), config)
            )

        self.assertEqual(deltas, [(96, -54), (96, -54), (96, -54)])

    def test_gain_above_one_is_supported_but_bounded(self) -> None:
        config = MakcuAimConfig(strength=2.0, max_step=640)
        target = Detection(0, "player", 0.9, (1060, 480, 1260, 980))
        delta_x, delta_y = makcu_target_delta(target, (1080, 1920, 3), config)
        self.assertGreater(delta_x, 200)
        self.assertLessEqual(abs(delta_x), 640)
        self.assertLessEqual(abs(delta_y), 640)
        with self.assertRaises(ValueError):
            MakcuAimConfig(strength=4.01)

    def test_vertical_rate_ratio_default_and_validation(self) -> None:
        self.assertEqual(
            MakcuAimConfig().vertical_rate_ratio,
            MAX_VERTICAL_RATE_RATIO,
        )
        self.assertEqual(MakcuAimConfig(vertical_rate_ratio=1.0).vertical_rate_ratio, 1.0)
        for value in (True, "0.48", 0.0, -0.01, 1.01, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "vertical rate ratio",
            ):
                MakcuAimConfig(vertical_rate_ratio=value)

    def test_detect_makcu_port_returns_single_matching_device(self) -> None:
        detected = detect_makcu_port(ports_provider=lambda: (self.port,))

        self.assertEqual(detected, self.port.device)

    def test_detect_makcu_port_rejects_missing_or_multiple_boards(self) -> None:
        with self.assertRaisesRegex(MakcuError, "was not found"):
            detect_makcu_port(ports_provider=lambda: ())
        second = FakePort("/dev/ttyACM1", MAKCU_VENDOR_ID, MAKCU_PRODUCT_ID)
        with self.assertRaisesRegex(MakcuError, "More than one MAKCU"):
            detect_makcu_port(ports_provider=lambda: (self.port, second))

    def test_requested_port_accepts_missing_metadata_but_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "ttyUSB0"
            selected.touch()
            correct = FakePort(str(selected), MAKCU_VENDOR_ID, MAKCU_PRODUCT_ID)
            metadata_less = FakePort(str(selected), None, None)
            wrong = FakePort(str(selected), 0x1234, 0x5678)

            self.assertEqual(
                detect_makcu_port(
                    requested=str(selected), ports_provider=lambda: (correct,)
                ),
                str(selected),
            )
            self.assertEqual(
                detect_makcu_port(
                    requested=str(selected), ports_provider=lambda: (metadata_less,)
                ),
                str(selected),
            )
            with self.assertRaisesRegex(MakcuError, "not the expected MAKCU"):
                detect_makcu_port(
                    requested=str(selected), ports_provider=lambda: (wrong,)
                )
            # An explicit device path can be firmware-probed even when pyserial
            # supplies no USB metadata or omits it from enumeration entirely.
            self.assertEqual(
                detect_makcu_port(requested=str(selected), ports_provider=lambda: ()),
                str(selected),
            )

    @unittest.skipIf(os.name == "nt", "ttyCH343USB paths are Linux-specific")
    def test_detects_linux_tty_ch343_path_without_usb_metadata(self) -> None:
        port = FakePort("/dev/ttyCH343USB0", None, None)

        self.assertEqual(
            detect_makcu_port(ports_provider=lambda: (port,)),
            "/dev/ttyCH343USB0",
        )

    @unittest.skipUnless(os.name == "posix", "system tty discovery is Linux-only")
    def test_controller_auto_discovery_includes_system_tty_ch343_candidates(self) -> None:
        candidate = Path("/dev/ttyCH343USB7")
        list_ports = mock.Mock()
        list_ports.comports.return_value = ()
        controller = MakcuAimingController(
            serial_factory=self.factory,
            sleep=lambda _seconds: None,
            threaded_output=False,
        )

        with (
            mock.patch("aiming.makcu._list_ports", list_ports),
            mock.patch.object(Path, "glob", return_value=(candidate,)),
            mock.patch.object(Path, "is_dir", return_value=False),
        ):
            controller.start()

        try:
            self.assertEqual(controller.connected_port, str(candidate))
            self.assertEqual(self.factory.connections[-1].port, str(candidate))
        finally:
            controller.stop()


if __name__ == "__main__":
    unittest.main()
