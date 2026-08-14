from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from aiming.makcu import (
    MAX_CONTINUOUS_ACTIVATION_SECONDS,
    MAKCU_BAUD_CHANGE,
    MAKCU_FAST_BAUD,
    MAKCU_PRODUCT_ID,
    MAKCU_VENDOR_ID,
    TARGET_STALE_SECONDS,
    MakcuError,
    MakcuAimConfig,
    MakcuAimingController,
    detect_makcu_port,
    makcu_target_delta,
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

        controller.update(target, (1080, 1920, 3))
        self.assertNotIn(b"km.move(-40,-67)\r", active.writes)
        active.responses.extend(bytes((0b00010,)))
        controller.update(target, (1080, 1920, 3))
        self.assertEqual(active.writes[-1], b"km.move(-40,-67)\r")
        active.responses.extend(bytes((0,)))
        before_release = len(active.writes)
        controller.update(target, (1080, 1920, 3))
        self.assertEqual(len(active.writes), before_release)

    def test_activation_poll_is_read_only(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        pressed_ns = 1_000_000_000
        active.responses.extend(bytes((0b00010,)))

        self.assertTrue(controller.poll_activation(now_ns=pressed_ns))
        self.assertFalse(any(write.startswith(b"km.move(") for write in active.writes))
        active.responses.extend(bytes((0,)))
        self.assertFalse(controller.poll_activation(now_ns=pressed_ns + 1))

    def test_poll_button_mask_reports_latest_button_bits(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        now_ns = 2_000_000_000

        active.responses.extend(bytes((0b00010,)))
        self.assertEqual(controller.poll_button_mask(now_ns=now_ns), 0b00010)
        active.responses.extend(bytes((0,)))
        self.assertEqual(controller.poll_button_mask(now_ns=now_ns + 10_000_000), 0)

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

        self.assertAlmostEqual(response_after_10ms(100), response_after_10ms(1000), delta=1.0)

    def test_vertical_motion_is_time_based_across_output_rates(self) -> None:
        def movement_after_50ms(rate_hz: int) -> int:
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
