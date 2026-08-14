from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from aiming.protocol import decode_aim_command
from aiming.controller import (
    AimActivationSensor,
    AimingController,
    AimConfig,
    LOCAL_TARGET_STALE_SECONDS,
    TargetTracker,
    UdpAimingController,
    choose_target,
    head_target_point,
)
from controller_precision.codes import EV_ABS, EV_SYN
from config import parse_args
from detection.types import Detection
from main import (
    _aim_status,
    _start_optional_aiming,
    _update_aim_target,
    _validate_aim_safety,
)
from utils.render import draw_aim_target


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


class RecordingUInput:
    def __init__(self) -> None:
        self.writes: list[tuple[int, int, int]] = []
        self.closed = False

    def write(self, event_type: int, code: int, value: int) -> None:
        self.writes.append((event_type, code, value))

    def close(self) -> None:
        self.closed = True


class ResampledAxisDevice:
    def __init__(self, value: int) -> None:
        self.value = value

    def read_one(self):
        return None

    def absinfo(self, _axis: int) -> FakeAbsInfo:
        return FakeAbsInfo(self.value, 0, 255, 0, 0, 0)


class FailingAimController:
    def __init__(self) -> None:
        self.stopped = False

    def start(self) -> None:
        raise RuntimeError("serial permission denied")

    def stop(self) -> None:
        self.stopped = True


class AimingControllerTests(unittest.TestCase):
    def test_aim_overlay_names_the_configured_physical_gate(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        with (
            patch("cv2.circle"),
            patch("cv2.drawMarker"),
            patch("cv2.putText") as put_text,
        ):
            draw_aim_target(
                frame,
                (50.0, 40.0),
                active=False,
                activation_name="Left",
            )

        self.assertEqual(put_text.call_args.args[1], "HEAD TARGET - HOLD LEFT")

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
        self.assertIsNone(
            _aim_status(
                runtime_enabled=active_controller is not None,
                self_exclusion_ready=True,
                selected_target=Detection(
                    0, "person", 0.9, (700, 200, 900, 800)
                ),
                engaged=True,
                activation_name="LT",
                control_description="output",
            )
        )

    def test_runtime_safety_validation_cannot_be_bypassed_with_direct_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit target label"):
            _validate_aim_safety(
                SimpleNamespace(
                    aim=True,
                    aim_label=" ",
                    aim_output="makcu",
                    aim_activate_path=None,
                )
            )
        with self.assertRaisesRegex(ValueError, "Remote aim is unavailable"):
            _validate_aim_safety(
                SimpleNamespace(
                    aim=True,
                    aim_label="person",
                    aim_output="remote",
                    aim_activate_path="/dev/input/event0",
                    ignore_self=True,
                )
            )
        with self.assertRaisesRegex(ValueError, "physical activation device"):
            _validate_aim_safety(
                SimpleNamespace(
                    aim=True,
                    aim_label="person",
                    aim_output="local",
                    aim_activate_path=None,
                    ignore_self=True,
                )
            )
        with self.assertRaisesRegex(ValueError, "self filter"):
            _validate_aim_safety(
                SimpleNamespace(
                    aim=True,
                    aim_label="person",
                    aim_output="makcu",
                    aim_activate_path=None,
                    ignore_self=False,
                )
            )

        for output in ("LOCAL", "garbage", None):
            with self.subTest(output=output), self.assertRaisesRegex(
                ValueError, "Unsupported safe aim output"
            ):
                _validate_aim_safety(
                    SimpleNamespace(
                        aim=True,
                        aim_label="person",
                        aim_output=output,
                        aim_activate_path="/dev/input/event0",
                        aim_activate_threshold=0.35,
                        ignore_self=True,
                    )
                )

        for threshold in (-1.0, 0.0, float("nan"), float("inf"), 1.01, True):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(
                ValueError, "activation threshold"
            ):
                _validate_aim_safety(
                    SimpleNamespace(
                        aim=True,
                        aim_label="person",
                        aim_output="local",
                        aim_activate_path="/dev/input/event0",
                        aim_activate_threshold=threshold,
                        ignore_self=True,
                    )
                )

    def test_activation_sensor_rejects_fail_open_thresholds(self) -> None:
        for threshold in (-1.0, 0.0, float("nan"), float("inf"), 1.01, True):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(
                ValueError, "activation threshold"
            ):
                AimActivationSensor("/dev/input/fake", threshold=threshold)

        self.assertEqual(
            AimActivationSensor("/dev/input/fake", threshold=0.35).threshold,
            0.35,
        )

    def test_aim_activation_threshold_cli_is_strictly_positive_and_bounded(self) -> None:
        for threshold in ("-1", "0", "nan", "inf", "1.01"):
            with self.subTest(threshold=threshold), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                parse_args(["--aim-activate-threshold", threshold])

        self.assertEqual(
            parse_args(["--aim-activate-threshold", "0.42"]).aim_activate_threshold,
            0.42,
        )

    def test_activation_sensor_resamples_axis_state_after_a_lost_release_event(self) -> None:
        device = ResampledAxisDevice(255)
        sensor = AimActivationSensor("/dev/input/fake", threshold=0.35)
        sensor._device = device
        sensor._minimum = 0
        sensor._maximum = 255
        sensor._active = True
        self.assertTrue(sensor.read())

        device.value = 0
        self.assertFalse(sensor.read())

    def test_local_watchdog_configuration_rejects_nonfinite_interval(self) -> None:
        for invalid in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                AimingController(watchdog_interval=invalid)

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

    def test_target_selection_rejects_barely_confident_center_speck(self) -> None:
        weak_center_speck = Detection(
            0,
            "person",
            0.251,
            (959, 539.76, 961, 541.76),
        )
        credible_nearby_player = Detection(
            0,
            "person",
            0.95,
            (900, 500, 1000, 900),
        )

        selected = choose_target(
            [weak_center_speck, credible_nearby_player],
            label="person",
            frame_shape=(1080, 1920, 3),
        )

        self.assertIs(selected, credible_nearby_player)

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

    def test_target_tracker_can_bridge_ordinary_detection_misses(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=18)
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        tracker.update([target], (1080, 1920, 3))

        for _ in range(18):
            self.assertIsNotNone(tracker.update((), (1080, 1920, 3)))
        self.assertIsNone(tracker.update((), (1080, 1920, 3)))

    def test_unsafe_self_exclusion_resets_tracker_without_grace(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=18)
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        tracker.update([target], (1080, 1920, 3))

        selected = _update_aim_target(
            tracker,
            (),
            (1080, 1920, 3),
            self_exclusion_safe=False,
        )

        self.assertIsNone(selected)
        self.assertIsNone(tracker.update((), (1080, 1920, 3)))

    def test_disabled_aim_runtime_cannot_select_or_retain_a_draw_target(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=18)
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        tracker.update([target], (1080, 1920, 3))

        selected = _update_aim_target(
            tracker,
            (target,),
            (1080, 1920, 3),
            self_exclusion_safe=True,
            aim_runtime_enabled=False,
        )

        self.assertIsNone(selected)
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

    def test_target_tracker_requires_confirmation_before_far_reacquisition(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=0)
        original = Detection(0, "person", 0.9, (250, 250, 450, 850))
        rival = Detection(0, "person", 0.9, (850, 250, 1050, 850))

        self.assertIs(
            tracker.update(
                [original],
                (1080, 1920, 3),
                measurement_ns=1_000_000_000,
            ),
            original,
        )
        self.assertIsNone(
            tracker.update(
                [rival],
                (1080, 1920, 3),
                measurement_ns=1_033_000_000,
            )
        )
        self.assertIsNotNone(
            tracker.update(
                [rival],
                (1080, 1920, 3),
                measurement_ns=1_066_000_000,
            )
        )

    def test_target_tracker_association_keeps_exact_identity_over_confidence(self) -> None:
        tracker = TargetTracker(label="person")
        original = Detection(0, "person", 0.95, (700, 250, 900, 850))
        moved = Detection(0, "person", 0.95, (730, 250, 930, 850))
        stale_duplicate = Detection(0, "person", 0.251, (700, 250, 900, 850))
        tracker.update(
            [original],
            (1080, 1920, 3),
            measurement_ns=1_000_000_000,
        )

        tracked = tracker.update(
            [moved, stale_duplicate],
            (1080, 1920, 3),
            measurement_ns=1_033_000_000,
        )

        assert tracked is not None
        self.assertEqual(tracked.confidence, stale_duplicate.confidence)
        self.assertEqual(tracked.x1, original.x1)

    def test_target_tracker_identity_beats_high_confidence_nearby_rival(self) -> None:
        tracker = TargetTracker(label="person")
        original = Detection(0, "person", 0.9, (700, 250, 900, 850))
        tracked_low_confidence = Detection(0, "person", 0.30, (710, 250, 910, 850))
        nearby_rival = Detection(0, "person", 0.99, (750, 250, 950, 850))
        tracker.update(
            [original],
            (1080, 1920, 3),
            measurement_ns=1_000_000_000,
        )

        tracked = tracker.update(
            [nearby_rival, tracked_low_confidence],
            (1080, 1920, 3),
            measurement_ns=1_033_000_000,
        )

        assert tracked is not None
        self.assertEqual(tracked.confidence, tracked_low_confidence.confidence)
        self.assertLess(head_target_point(tracked)[0], 825.0)

    def test_target_tracker_velocity_is_time_based_across_detector_rates(self) -> None:
        def run(rate_hz: int) -> tuple[float, float]:
            tracker = TargetTracker(label="person")
            speed = 600.0
            sample_count = round(1.0 * rate_hz)
            target = None
            for index in range(sample_count + 1):
                elapsed = index / rate_hz
                center_x = 600.0 + speed * elapsed
                target = tracker.update(
                    [
                        Detection(
                            0,
                            "person",
                            0.9,
                            (center_x - 50, 300, center_x + 50, 800),
                        )
                    ],
                    (1080, 1920, 3),
                    measurement_ns=1_000_000_000 + round(elapsed * 1e9),
                )
            assert target is not None
            return head_target_point(target)[0], tracker._box_velocity[0]

        center_25, velocity_25 = run(25)
        center_100, velocity_100 = run(100)

        self.assertAlmostEqual(center_25, center_100, delta=1.0)
        self.assertAlmostEqual(velocity_25, velocity_100, delta=30.0)

    def test_target_tracker_prediction_grace_is_time_based_across_rates(self) -> None:
        target = Detection(0, "person", 0.9, (700, 250, 900, 850))

        def last_predicted_time(rate_hz: int) -> float:
            tracker = TargetTracker(label="person", lost_grace_frames=3)
            base_ns = 1_000_000_000
            tracker.update([target], (1080, 1920, 3), measurement_ns=base_ns)
            last_seconds = 0.0
            for index in range(1, rate_hz + 1):
                seconds = index / rate_hz
                prediction = tracker.update(
                    [],
                    (1080, 1920, 3),
                    measurement_ns=base_ns + round(seconds * 1e9),
                )
                if prediction is None:
                    break
                last_seconds = seconds
            return last_seconds

        last_25 = last_predicted_time(25)
        last_100 = last_predicted_time(100)

        self.assertGreaterEqual(last_25, 0.04)
        self.assertLessEqual(last_100, 0.05)
        self.assertAlmostEqual(last_25, last_100, delta=0.011)

    def test_target_tracker_rejects_backwards_measurement_timestamp(self) -> None:
        tracker = TargetTracker(label="person")
        target = Detection(0, "person", 0.9, (700, 250, 900, 850))
        tracker.update(
            [target],
            (1080, 1920, 3),
            measurement_ns=2_000_000_000,
        )

        with self.assertRaisesRegex(ValueError, "must not move backwards"):
            tracker.update(
                [target],
                (1080, 1920, 3),
                measurement_ns=1_999_999_999,
            )

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

    def test_local_output_never_synthesizes_trigger_and_watchdog_neutralizes(self) -> None:
        now_ns = [1_000_000_000]
        controller = AimingController(clock_ns=lambda: now_ns[0])
        device = RecordingUInput()
        controller._uinput = device
        target = Detection(0, "person", 0.9, (1400, 300, 1600, 900))

        controller.update(target, (1080, 1920, 3), active=True)

        self.assertTrue(controller._output_active)
        trigger_writes = [
            value
            for event_type, code, value in device.writes
            if event_type == EV_ABS and code == controller.config.trigger_code
        ]
        self.assertEqual(trigger_writes, [controller.config.trigger_released_value])

        now_ns[0] += int((LOCAL_TARGET_STALE_SECONDS + 0.01) * 1e9)
        controller._watchdog_tick()

        self.assertFalse(controller._output_active)
        final_report = device.writes[-4:]
        self.assertEqual(
            final_report,
            [
                (EV_ABS, controller.config.right_x_code, controller.config.neutral_value),
                (EV_ABS, controller.config.right_y_code, controller.config.neutral_value),
                (
                    EV_ABS,
                    controller.config.trigger_code,
                    controller.config.trigger_released_value,
                ),
                (EV_SYN, 0, 0),
            ],
        )

    def test_local_output_neutralizes_and_closes_on_stop(self) -> None:
        controller = AimingController(clock_ns=lambda: 1_000_000_000)
        device = RecordingUInput()
        controller._uinput = device
        target = Detection(0, "person", 0.9, (1400, 300, 1600, 900))
        controller.update(target, (1080, 1920, 3), active=True)

        controller.stop()

        self.assertTrue(device.closed)
        self.assertIsNone(controller._uinput)
        self.assertEqual(device.writes[-1], (EV_SYN, 0, 0))

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

    def test_remote_aim_cli_is_rejected_until_a_safe_receiver_exists(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            parse_args(
                [
                    "--aim",
                    "--aim-label",
                    "person",
                    "--aim-output",
                    "remote",
                    "--ignore-self",
                    "--aim-host",
                    "192.168.1.40",
                    "--aim-port",
                    "47621",
                    "--aim-pairing-key",
                    "0123456789abcdef0123456789abcdef",
                ]
            )
        self.assertIn("no authenticated, physically gated receiver", error.getvalue())

    def test_aim_cli_requires_an_explicit_target_label(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            parse_args(
                [
                    "--aim",
                    "--aim-output",
                    "makcu",
                    "--aim-makcu-port",
                    "/dev/ttyACM0",
                ]
            )
        self.assertIn("--aim-label is required", error.getvalue())

    def test_local_aim_cli_requires_a_physical_activation_device(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            parse_args(
                [
                    "--aim",
                    "--aim-label",
                    "person",
                    "--aim-output",
                    "local",
                    "--ignore-self",
                ]
            )
        self.assertIn("--aim-activate-path is required", error.getvalue())

    def test_makcu_aim_cli_configuration_parses(self) -> None:
        config = parse_args(
            [
                "--aim",
                "--aim-label",
                "player",
                "--aim-output",
                "makcu",
                "--ignore-self",
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
        self.assertTrue(config.ignore_self)
        self.assertEqual(config.aim_makcu_smoothing_alpha, 0.72)
        self.assertEqual(config.aim_makcu_prediction_lead_seconds, 0.04)
        self.assertEqual(config.aim_makcu_derivative_damping_seconds, 0.01)

    def test_aim_cli_requires_self_filter(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            parse_args(
                [
                    "--aim",
                    "--aim-label",
                    "player",
                    "--aim-output",
                    "makcu",
                    "--aim-makcu-port",
                    "/dev/ttyACM0",
                ]
            )
        self.assertIn("--ignore-self is required", error.getvalue())


if __name__ == "__main__":
    unittest.main()
