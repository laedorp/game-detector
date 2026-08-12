from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
import unittest

from aiming.protocol import decode_aim_command
from aiming.controller import (
    AimingController,
    AimConfig,
    TargetTracker,
    UdpAimingController,
    choose_target,
    head_target_point,
)
from controller_precision.codes import EV_ABS
from config import parse_args
from detection.types import Detection
from main import _start_optional_aiming


@dataclass(frozen=True)
class FakeAbsInfo:
    value: int
    min: int
    max: int
    fuzz: int
    flat: int
    resolution: int


class FakeEvdev:
    AbsInfo = FakeAbsInfo
    created: tuple[dict[int, list[tuple[int, FakeAbsInfo]]], dict[str, object]] | None = None

    @classmethod
    def UInput(cls, events, **options):
        cls.created = (events, options)
        return object()


class FakeSocket:
    def __init__(self) -> None:
        self.address: tuple[str, int] | None = None
        self.packets: list[bytes] = []
        self.closed = False

    def connect(self, address: tuple[str, int]) -> None:
        self.address = address

    def send(self, packet: bytes) -> int:
        self.packets.append(packet)
        return len(packet)

    def close(self) -> None:
        self.closed = True


class FailingAimController:
    def __init__(self) -> None:
        self.stopped = False

    def start(self) -> None:
        raise RuntimeError("serial permission denied")

    def stop(self) -> None:
        self.stopped = True


class AimingControllerTests(unittest.TestCase):
    def test_default_head_point_is_twelve_percent_down_player_box(self) -> None:
        target = Detection(0, "person", 0.9, (700, 200, 900, 800))
        self.assertEqual(head_target_point(target), (800.0, 272.0))

    def test_optional_aim_failure_does_not_abort_capture_startup(self) -> None:
        controller = FailingAimController()
        error = io.StringIO()

        with contextlib.redirect_stderr(error):
            active_controller, active_sensor = _start_optional_aiming(controller, None)

        self.assertIsNone(active_controller)
        self.assertIsNone(active_sensor)
        self.assertTrue(controller.stopped)
        self.assertIn("Capture, inference, and preview will continue", error.getvalue())

    def test_target_selection_prefers_matching_detection_nearest_crosshair(self) -> None:
        high_confidence_edge = Detection(0, "person", 0.99, (10, 10, 110, 410))
        lower_confidence_center = Detection(0, "person", 0.70, (860, 400, 1060, 900))
        wrong_class_center = Detection(1, "car", 1.0, (900, 400, 1020, 700))

        selected = choose_target(
            [high_confidence_edge, lower_confidence_center, wrong_class_center],
            label="person",
            frame_shape=(1080, 1920, 3),
        )

        self.assertIs(selected, lower_confidence_center)

    def test_target_tracker_keeps_same_player_and_bridges_three_misses(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=3)
        original = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        closer_rival = Detection(0, "person", 0.9, (930, 350, 1030, 750))
        moved_original = Detection(0, "person", 0.75, (815, 305, 1015, 905))

        self.assertIs(tracker.update([original], (1080, 1920, 3)), original)
        tracked = tracker.update([closer_rival, moved_original], (1080, 1920, 3))
        assert tracked is not None
        self.assertEqual(tracked.class_name, "person")
        self.assertGreater(tracked.x1, original.x1)
        self.assertLess(tracked.x1, moved_original.x1)
        self.assertIsNotNone(tracker.update([], (1080, 1920, 3)))
        self.assertIsNotNone(tracker.update([], (1080, 1920, 3)))
        self.assertIsNotNone(tracker.update([], (1080, 1920, 3)))
        self.assertIsNone(tracker.update([], (1080, 1920, 3)))

    def test_target_tracker_reset_drops_stale_target_immediately(self) -> None:
        tracker = TargetTracker(label="person")
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        tracker.update([target], (1080, 1920, 3))
        tracker.reset()
        self.assertIsNone(tracker.update([], (1080, 1920, 3)))

    def test_target_tracker_can_coast_through_self_filter_uncertainty(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=18)
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        tracker.update([target], (1080, 1920, 3))

        for _ in range(18):
            self.assertIsNotNone(tracker.update((), (1080, 1920, 3)))
        self.assertIsNone(tracker.update((), (1080, 1920, 3)))

    def test_target_tracker_smooths_small_box_jitter(self) -> None:
        tracker = TargetTracker(label="person")
        first = Detection(0, "person", 0.8, (900, 420, 960, 540))
        jittered = Detection(0, "person", 0.8, (920, 400, 980, 520))

        tracker.update([first], (1080, 1920, 3))
        smoothed = tracker.update([jittered], (1080, 1920, 3))

        assert smoothed is not None
        self.assertGreater(smoothed.x1, (first.x1 + jittered.x1) / 2)
        self.assertLess(smoothed.x1, jittered.x1)
        self.assertGreater(smoothed.y1, jittered.y1)
        self.assertLess(smoothed.y1, (first.y1 + jittered.y1) / 2)

    def test_real_uinput_shape_uses_valid_pxn_identity_and_axis_ranges(self) -> None:
        controller = AimingController()

        controller._make_uinput(FakeEvdev)

        self.assertIsNotNone(FakeEvdev.created)
        assert FakeEvdev.created is not None
        events, options = FakeEvdev.created
        self.assertEqual(options["name"], "PXN P5 8K")
        self.assertEqual(options["vendor"], 0x36E6)
        self.assertEqual(options["product"], 0x3016)
        self.assertEqual(options["phys"], "game-detector-aim/uinput")
        axes = events[EV_ABS]
        self.assertEqual(len(axes), 3)
        for _code, info in axes:
            self.assertEqual((info.min, info.max), (0, 255))
            self.assertGreaterEqual(info.value, info.min)
            self.assertLessEqual(info.value, info.max)

    def test_udp_output_sends_target_vector_and_neutral_shutdown(self) -> None:
        sender = FakeSocket()
        key = "0123456789abcdef0123456789abcdef"
        controller = UdpAimingController(
            "192.168.1.20",
            47621,
            key,
            AimConfig(),
            socket_factory=lambda: sender,
        )
        target = Detection(0, "player", 0.9, (700, 200, 900, 800))

        controller.start()
        controller.update(target, (1080, 1920, 3))
        controller.stop()

        self.assertEqual(sender.address, ("192.168.1.20", 47621))
        self.assertTrue(sender.closed)
        active = decode_aim_command(sender.packets[0], key)
        neutral = decode_aim_command(sender.packets[-1], key)
        self.assertTrue(active.active)
        self.assertAlmostEqual(active.x, -1 / 6, places=5)
        self.assertAlmostEqual(active.y, -0.4962963, places=5)
        self.assertFalse(neutral.active)
        self.assertEqual((neutral.x, neutral.y), (0.0, 0.0))

    def test_remote_aim_cli_configuration_parses(self) -> None:
        config = parse_args(
            [
                "--aim",
                "--aim-output",
                "remote",
                "--aim-host",
                "192.168.1.40",
                "--aim-port",
                "47621",
                "--aim-pairing-key",
                "0123456789abcdef0123456789abcdef",
            ]
        )
        self.assertEqual(config.aim_output, "remote")
        self.assertEqual(config.aim_host, "192.168.1.40")
        self.assertEqual(config.aim_port, 47621)

    def test_makcu_aim_cli_configuration_parses(self) -> None:
        config = parse_args(
            [
                "--aim",
                "--aim-output",
                "makcu",
                "--aim-makcu-port",
                "/dev/ttyACM0",
                "--aim-makcu-button",
                "1",
                "--aim-makcu-strength",
                "0.25",
                "--aim-makcu-max-step",
                "80",
                "--aim-makcu-smoothing-alpha",
                "0.72",
                "--aim-makcu-prediction-lead-seconds",
                "0.04",
                "--aim-makcu-derivative-damping-seconds",
                "0.01",
            ]
        )
        self.assertEqual(config.aim_output, "makcu")
        self.assertEqual(config.aim_makcu_port, "/dev/ttyACM0")
        self.assertEqual(config.aim_makcu_button, 1)
        self.assertEqual(config.aim_makcu_strength, 0.25)
        self.assertEqual(config.aim_makcu_max_step, 80)
        self.assertEqual(config.aim_makcu_smoothing_alpha, 0.72)
        self.assertEqual(config.aim_makcu_prediction_lead_seconds, 0.04)
        self.assertEqual(config.aim_makcu_derivative_damping_seconds, 0.01)


if __name__ == "__main__":
    unittest.main()