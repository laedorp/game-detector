from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

from aiming.makcu import (
    BUTTON_REPORT_STALE_SECONDS,
    BUTTON_STREAM_PERIOD_MS,
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
    vid: int
    pid: int


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
        self.assertIn(f"km.buttons(1,{BUTTON_STREAM_PERIOD_MS})\r".encode(), active.writes)
        controller.stop()
        self.assertIn(b"km.buttons(0)\r", active.writes)
        self.assertTrue(active.closed)

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

    def test_valid_release_report_stops_activation_immediately(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        now_ns = 1_000_000_000
        active.responses.extend(bytes((0b00010,)))
        self.assertTrue(controller.poll_activation(now_ns=now_ns))

        active.responses.extend(bytes((0,)))
        self.assertFalse(controller.poll_activation(now_ns=now_ns + 1))

    def test_stale_button_report_fails_closed_if_release_is_lost(self) -> None:
        controller = self.controller()
        controller.start()
        active = self.factory.connections[-1]
        pressed_ns = 1_000_000_000
        active.responses.extend(bytes((0b00010,)))
        self.assertTrue(controller.poll_activation(now_ns=pressed_ns))

        stale_ns = pressed_ns + int((BUTTON_REPORT_STALE_SECONDS + 0.01) * 1e9)
        self.assertFalse(controller.poll_activation(now_ns=stale_ns))
        self.assertEqual(controller.poll_button_mask(now_ns=stale_ns + 1), 0)

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

    def test_target_delta_uses_head_point_deadzone_and_maximum_step(self) -> None:
        config = MakcuAimConfig(strength=1.0, max_step=20, deadzone_pixels=2.0)
        far = Detection(0, "player", 0.9, (1800, 900, 1920, 1080))
        centered = Detection(0, "player", 0.9, (958, 538, 962, 558))
        self.assertEqual(makcu_target_delta(far, (1080, 1920, 3), config), (20, 20))
        self.assertEqual(makcu_target_delta(centered, (1080, 1920, 3), config), (0, 0))

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
        with self.assertRaisesRegex(MakcuError, "More than one MAKCU"):
            detect_makcu_port(ports_provider=lambda: (self.port, self.port))

    def test_requested_port_must_be_the_enumerated_makcu_usb_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "ttyUSB0"
            selected.touch()
            correct = FakePort(str(selected), MAKCU_VENDOR_ID, MAKCU_PRODUCT_ID)
            wrong = FakePort(str(selected), 0x1234, 0x5678)

            self.assertEqual(
                detect_makcu_port(
                    requested=str(selected), ports_provider=lambda: (correct,)
                ),
                str(selected),
            )
            with self.assertRaisesRegex(MakcuError, "not the expected MAKCU"):
                detect_makcu_port(
                    requested=str(selected), ports_provider=lambda: (wrong,)
                )
            with self.assertRaisesRegex(MakcuError, "not one uniquely enumerated"):
                detect_makcu_port(
                    requested=str(selected), ports_provider=lambda: ()
                )


if __name__ == "__main__":
    unittest.main()
