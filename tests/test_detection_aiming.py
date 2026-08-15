from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from aiming.protocol import decode_aim_command
from aiming.makcu import MakcuTelemetrySnapshot
from aiming.controller import (
    AimActivationSensor,
    AimingController,
    AimConfig,
    LOCAL_TARGET_STALE_SECONDS,
    TargetTracker,
    TargetTrackerTelemetrySnapshot,
    UdpAimingController,
    choose_target,
    head_target_point,
)
from controller_precision.codes import EV_ABS, EV_SYN
from config import parse_args
from detection.types import Detection
from main import (
    AimInputTelemetry,
    _aim_status,
    _aim_input_telemetry_summary,
    _apply_hard_aim_guard,
    _makcu_telemetry_summary,
    _start_optional_aiming,
    _target_tracker_telemetry_summary,
    _update_aim_target,
    _validate_aim_safety,
)
from utils.render import draw_aim_target
from utils.self_filter import NormalizedBottomZone, SelfAvatarFilter


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
    def test_makcu_telemetry_summary_reports_gate_and_real_command_rates(self) -> None:
        previous = MakcuTelemetrySnapshot(
            output_ticks=100,
            button_pressed_ticks=50,
            target_present_ticks=80,
            fresh_target_ticks=75,
            authorized_ticks=40,
            movement_commands=20,
            emitted_x=-5,
            emitted_y=10,
            emitted_abs_x=25,
            emitted_abs_y=30,
            control_samples=10,
            control_error_abs_x=100.0,
            control_error_abs_y=50.0,
            pursuit_abs_x=20.0,
            pursuit_abs_y=10.0,
            saturated_x_samples=1,
            pursuit_resets=2,
        )
        current = MakcuTelemetrySnapshot(
            output_ticks=1100,
            button_pressed_ticks=850,
            target_present_ticks=980,
            fresh_target_ticks=925,
            authorized_ticks=740,
            movement_commands=620,
            emitted_x=395,
            emitted_y=-190,
            emitted_abs_x=2425,
            emitted_abs_y=1230,
            control_samples=60,
            control_error_abs_x=600.0,
            control_error_abs_y=300.0,
            pursuit_abs_x=120.0,
            pursuit_abs_y=60.0,
            saturated_x_samples=6,
            pursuit_resets=5,
        )

        summary = _makcu_telemetry_summary(previous, current, 1.0)

        self.assertIn("MAKCU loop 1000 Hz", summary)
        self.assertIn("button gate 80%", summary)
        self.assertIn("target 90%", summary)
        self.assertIn("fresh 85%", summary)
        self.assertIn("authorized 70%", summary)
        self.assertIn("moves 600/s", summary)
        self.assertIn("abs counts X/Y 2400/1200/s", summary)
        self.assertIn("net X/Y +400/-200/s", summary)
        self.assertIn("CTRL samples 50/s", summary)
        self.assertIn("error abs X/Y 10.0/5.0px", summary)
        self.assertIn("pursuit X/Y 120/60 cps", summary)
        self.assertIn("saturation X/Y 10/0%", summary)
        self.assertIn("pursuit resets 3", summary)

    def test_tracker_telemetry_summary_reports_aggregate_residuals_only(self) -> None:
        previous = TargetTrackerTelemetrySnapshot(
            updates=10,
            candidate_samples=8,
            measurement_samples=7,
            continuation_measurement_samples=2,
            output_samples=7,
            compared_samples=6,
            target_loss_transitions=1,
            residual_x=10.0,
            residual_y=5.0,
            residual_abs_x=20.0,
            residual_abs_y=10.0,
        )
        current = TargetTrackerTelemetrySnapshot(
            updates=110,
            candidate_samples=98,
            measurement_samples=87,
            continuation_measurement_samples=14,
            output_samples=92,
            compared_samples=86,
            target_loss_transitions=3,
            residual_x=50.0,
            residual_y=-11.0,
            residual_abs_x=260.0,
            residual_abs_y=130.0,
        )

        summary = _target_tracker_telemetry_summary(previous, current, 1.0)

        self.assertIn("TRACK samples 100/s", summary)
        self.assertIn("raw/out 80/85/s", summary)
        self.assertIn("continued-low 12/s", summary)
        self.assertIn("rejected 10/s", summary)
        self.assertIn("raw-track abs X/Y 3.0/1.5px", summary)
        self.assertIn("signed +0.5/-0.2px", summary)
        self.assertIn("losses 2", summary)

    def test_hard_aim_guard_attributes_in_zone_exact_label_removal(self) -> None:
        zone = NormalizedBottomZone(left=0.25, width=0.5, height=0.5)
        self_candidate = Detection(0, "player", 0.9, (40, 35, 60, 100))

        result = _apply_hard_aim_guard(
            [self_candidate],
            (100, 100, 3),
            self_zone=zone,
            aim_label="PLAYER",
        )

        self.assertEqual(result.detections, ())
        self.assertEqual(result.removed_exact_label_boxes, 1)
        self.assertTrue(result.targetless_after_exact_removal)

    def test_hard_aim_guard_does_not_block_when_same_label_survives(self) -> None:
        zone = NormalizedBottomZone(left=0.25, width=0.5, height=0.5)
        self_candidate = Detection(0, "player", 0.9, (40, 35, 60, 100))
        opponent = Detection(0, "player", 0.9, (80, 20, 96, 75))

        result = _apply_hard_aim_guard(
            [self_candidate, opponent],
            (100, 100, 3),
            self_zone=zone,
            aim_label="player",
            configured_confidence=0.25,
        )

        self.assertEqual(result.detections, (opponent,))
        self.assertEqual(result.removed_exact_label_boxes, 1)
        self.assertFalse(result.targetless_after_exact_removal)

    def test_confirmed_self_removal_retains_distinct_in_zone_opponent(self) -> None:
        frame_shape = (1080, 1920, 3)
        zone = NormalizedBottomZone(left=0.18, width=0.34, height=0.10)
        self_filter = SelfAvatarFilter(zone)
        avatar = Detection(0, "person", 0.90, (430, 560, 850, 1080))
        opponent = Detection(0, "player", 0.90, (850, 600, 1050, 1080))
        for _ in range(3):
            self_filter.apply((avatar,), frame_shape)

        exclusion = self_filter.apply((avatar, opponent), frame_shape)
        self.assertTrue(exclusion.aim_safe)
        self.assertEqual(exclusion.ignored_count, 1)
        self.assertIs(exclusion.ignored_detection, avatar)
        self.assertEqual(exclusion.detections, (opponent,))

        result = _apply_hard_aim_guard(
            exclusion.detections,
            frame_shape,
            self_zone=zone,
            aim_label="player",
            configured_confidence=0.25,
            confirmed_self_detection=exclusion.ignored_detection,
        )

        self.assertEqual(result.detections, (opponent,))
        self.assertEqual(result.removed_exact_label_boxes, 0)
        self.assertFalse(result.targetless_after_exact_removal)
        selected = _update_aim_target(
            TargetTracker(label="player"),
            result.detections,
            frame_shape,
            self_exclusion_safe=exclusion.aim_safe,
        )
        self.assertIs(selected, opponent)

    def test_confirmed_self_removal_still_guards_overlapping_duplicate(self) -> None:
        frame_shape = (1080, 1920, 3)
        zone = NormalizedBottomZone(left=0.18, width=0.34, height=0.10)
        self_filter = SelfAvatarFilter(zone)
        avatar = Detection(0, "person", 0.90, (430, 560, 850, 1080))
        duplicate = Detection(0, "player", 0.90, (440, 570, 860, 1080))
        for _ in range(3):
            self_filter.apply((avatar,), frame_shape)

        exclusion = self_filter.apply((avatar, duplicate), frame_shape)
        self.assertTrue(exclusion.aim_safe)
        self.assertIs(exclusion.ignored_detection, avatar)
        self.assertEqual(exclusion.detections, (duplicate,))

        result = _apply_hard_aim_guard(
            exclusion.detections,
            frame_shape,
            self_zone=zone,
            aim_label="player",
            configured_confidence=0.25,
            confirmed_self_detection=exclusion.ignored_detection,
        )

        self.assertEqual(result.detections, ())
        self.assertEqual(result.removed_exact_label_boxes, 1)
        self.assertTrue(result.targetless_after_exact_removal)

    def test_ambiguous_same_class_self_boxes_remain_fail_closed(self) -> None:
        frame_shape = (1080, 1920, 3)
        zone = NormalizedBottomZone(left=0.18, width=0.34, height=0.10)
        self_filter = SelfAvatarFilter(zone)
        avatar = Detection(0, "player", 0.90, (430, 560, 850, 1080))
        duplicate = Detection(0, "player", 0.90, (440, 570, 860, 1080))
        for _ in range(3):
            self_filter.apply((avatar,), frame_shape)

        exclusion = self_filter.apply((avatar, duplicate), frame_shape)
        self.assertFalse(exclusion.aim_safe)
        self.assertEqual(exclusion.ignored_count, 0)
        self.assertIsNone(exclusion.ignored_detection)
        guarded = _apply_hard_aim_guard(
            exclusion.detections,
            frame_shape,
            self_zone=zone,
            aim_label="player",
            configured_confidence=0.25,
        )
        self.assertIsNone(
            _update_aim_target(
                TargetTracker(label="player"),
                guarded.detections,
                frame_shape,
                self_exclusion_safe=exclusion.aim_safe,
            )
        )

    def test_confirmed_self_refinement_updates_guard_telemetry_exactly(self) -> None:
        frame_shape = (1080, 1920, 3)
        zone = NormalizedBottomZone(left=0.18, width=0.34, height=0.10)
        confirmed = Detection(0, "person", 0.90, (430, 560, 850, 1080))
        distinct = Detection(0, "player", 0.90, (850, 600, 1050, 1080))
        duplicate = Detection(0, "player", 0.90, (440, 570, 860, 1080))
        telemetry = AimInputTelemetry("player")

        telemetry.record_hard_guard(
            _apply_hard_aim_guard(
                (distinct,),
                frame_shape,
                self_zone=zone,
                aim_label="player",
                configured_confidence=0.25,
                confirmed_self_detection=confirmed,
            )
        )
        after_distinct = telemetry.snapshot()
        self.assertEqual(after_distinct.hard_guard_removed_exact_boxes, 0)
        self.assertEqual(after_distinct.hard_guard_targetless_samples, 0)

        telemetry.record_hard_guard(
            _apply_hard_aim_guard(
                (duplicate,),
                frame_shape,
                self_zone=zone,
                aim_label="player",
                configured_confidence=0.25,
                confirmed_self_detection=confirmed,
            )
        )
        after_duplicate = telemetry.snapshot()
        self.assertEqual(after_duplicate.hard_guard_removed_exact_boxes, 1)
        self.assertEqual(after_duplicate.hard_guard_targetless_samples, 1)

    def test_hard_guard_weak_survivor_cannot_preserve_removed_strong_target(
        self,
    ) -> None:
        zone = NormalizedBottomZone(left=0.25, width=0.5, height=0.5)
        self_candidate = Detection(0, "player", 0.90, (40, 35, 60, 100))
        far_weak = Detection(0, "player", 0.18, (0, 0, 10, 20))

        result = _apply_hard_aim_guard(
            [self_candidate, far_weak],
            (100, 100, 3),
            self_zone=zone,
            aim_label="player",
            configured_confidence=0.25,
        )

        self.assertEqual(result.detections, (far_weak,))
        self.assertEqual(result.removed_exact_label_boxes, 1)
        self.assertTrue(result.targetless_after_exact_removal)

    def test_hard_aim_guard_non_target_label_does_not_revoke_grace(self) -> None:
        zone = NormalizedBottomZone(left=0.25, width=0.5, height=0.5)
        unrelated_player_label = Detection(
            0,
            "person",
            0.9,
            (40, 35, 60, 100),
        )

        result = _apply_hard_aim_guard(
            [unrelated_player_label],
            (100, 100, 3),
            self_zone=zone,
            aim_label="player",
        )

        self.assertEqual(result.detections, ())
        self.assertEqual(result.removed_exact_label_boxes, 0)
        self.assertFalse(result.targetless_after_exact_removal)

    def test_aim_input_telemetry_reports_per_interval_cause_deltas(self) -> None:
        telemetry = AimInputTelemetry("player")
        previous = telemetry.snapshot()
        exact = Detection(0, "PLAYER", 0.9, (80, 20, 96, 75))
        unrelated = Detection(1, "prop", 0.9, (10, 10, 20, 20))
        zone = NormalizedBottomZone(left=0.25, width=0.5, height=0.5)
        guarded = Detection(0, "player", 0.9, (40, 35, 60, 100))

        telemetry.record_sample([exact])
        telemetry.record_self_filter(aim_safe=True)
        telemetry.record_sample([unrelated])
        telemetry.record_self_filter(aim_safe=False)
        telemetry.record_hard_guard(
            _apply_hard_aim_guard(
                [guarded],
                (100, 100, 3),
                self_zone=zone,
                aim_label="player",
            )
        )

        summary = _aim_input_telemetry_summary(
            previous,
            telemetry.snapshot(),
            0.5,
        )

        self.assertIn("AIM INPUT 4/s", summary)
        self.assertIn("exact 2/s", summary)
        self.assertIn("self-unsafe 1", summary)
        self.assertIn("guard exact boxes 1", summary)
        self.assertIn("guard targetless 1", summary)

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

    def test_one_reference_frame_grace_bridges_8_and_16_ms_then_expires(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=1)
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        base_ns = 1_000_000_000

        self.assertIs(
            tracker.update(
                [target],
                (1080, 1920, 3),
                measurement_ns=base_ns,
            ),
            target,
        )
        self.assertIsNotNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 8_000_000,
            )
        )
        self.assertIsNotNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 16_000_000,
            )
        )

        bridged = tracker.telemetry_snapshot()
        self.assertEqual(bridged.updates, 3)
        self.assertEqual(bridged.candidate_samples, 1)
        self.assertEqual(bridged.measurement_samples, 1)
        self.assertEqual(bridged.output_samples, 3)
        self.assertEqual(bridged.compared_samples, 1)
        self.assertEqual(bridged.target_loss_transitions, 0)

        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 16_666_668,
            )
        )
        expired = tracker.telemetry_snapshot()
        self.assertEqual(expired.updates, 4)
        self.assertEqual(expired.candidate_samples, 1)
        self.assertEqual(expired.measurement_samples, 1)
        self.assertEqual(expired.output_samples, 3)
        self.assertEqual(expired.compared_samples, 1)
        self.assertEqual(expired.target_loss_transitions, 1)

    def test_three_reference_frame_grace_bridges_fifty_ms_then_expires(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=3)
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        base_ns = 1_500_000_000

        tracker.update(
            (target,),
            (1080, 1920, 3),
            measurement_ns=base_ns,
        )
        for elapsed_ms in (8, 24, 40, 50):
            with self.subTest(elapsed_ms=elapsed_ms):
                self.assertIsNotNone(
                    tracker.update(
                        (),
                        (1080, 1920, 3),
                        measurement_ns=base_ns + elapsed_ms * 1_000_000,
                    )
                )
                self.assertTrue(tracker.output_is_prediction)

        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                # Three rounded 60 Hz reference frames equal 50,000,001 ns.
                measurement_ns=base_ns + 50_000_002,
            )
        )
        self.assertFalse(tracker.output_is_prediction)
        self.assertEqual(
            tracker.telemetry_snapshot().target_loss_transitions,
            1,
        )

    def test_within_grace_reacquisition_is_measured_without_a_loss(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=1)
        first = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        reacquired = Detection(0, "person", 0.85, (808, 300, 1008, 900))
        base_ns = 2_000_000_000

        tracker.update([first], (1080, 1920, 3), measurement_ns=base_ns)
        prediction = tracker.update(
            (),
            (1080, 1920, 3),
            measurement_ns=base_ns + 8_000_000,
        )
        self.assertIsNotNone(prediction)
        resumed = tracker.update(
            [reacquired],
            (1080, 1920, 3),
            measurement_ns=base_ns + 16_000_000,
        )

        self.assertIsNotNone(resumed)
        telemetry = tracker.telemetry_snapshot()
        self.assertEqual(telemetry.updates, 3)
        self.assertEqual(telemetry.candidate_samples, 2)
        self.assertEqual(telemetry.measurement_samples, 2)
        self.assertEqual(telemetry.continuation_measurement_samples, 0)
        self.assertEqual(telemetry.output_samples, 3)
        self.assertEqual(telemetry.compared_samples, 2)
        self.assertEqual(telemetry.target_loss_transitions, 0)

    def test_low_confidence_box_continues_only_an_active_geometric_track(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=0)
        acquired = Detection(0, "person", 0.90, (700, 250, 900, 850))
        continued = Detection(0, "person", 0.18, (708, 250, 908, 850))
        base_ns = 2_500_000_000

        self.assertIs(
            tracker.update(
                (acquired,),
                (1080, 1920, 3),
                measurement_ns=base_ns,
            ),
            acquired,
        )
        tracked = tracker.update(
            (),
            (1080, 1920, 3),
            measurement_ns=base_ns + 8_000_000,
            continuation_detections=(continued,),
        )

        self.assertIsNotNone(tracked)
        assert tracked is not None
        self.assertEqual(tracked.confidence, continued.confidence)
        self.assertFalse(tracker.output_is_prediction)
        telemetry = tracker.telemetry_snapshot()
        self.assertEqual(telemetry.measurement_samples, 2)
        self.assertEqual(telemetry.continuation_measurement_samples, 1)
        self.assertEqual(telemetry.output_samples, 2)

    def test_low_confidence_box_cannot_acquire_or_enter_reacquisition(self) -> None:
        fresh_tracker = TargetTracker(label="person", lost_grace_frames=0)
        low = Detection(0, "person", 0.18, (700, 250, 900, 850))
        self.assertIsNone(
            fresh_tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=1_000_000_000,
                continuation_detections=(low,),
            )
        )

        tracker = TargetTracker(
            label="person",
            lost_grace_frames=0,
            reacquire_confirmations=2,
        )
        original = Detection(0, "person", 0.90, (200, 250, 400, 850))
        low_rival = Detection(0, "person", 0.18, (1100, 250, 1300, 850))
        high_rival = Detection(0, "person", 0.90, low_rival.xyxy)
        base_ns = 2_000_000_000
        tracker.update((original,), (1080, 1920, 3), measurement_ns=base_ns)
        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 8_000_000,
            )
        )
        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 16_000_000,
                continuation_detections=(low_rival,),
            )
        )
        # The low rival did not enter pending reacquisition, so the first
        # configured-confidence rival still cannot revive output.
        self.assertIsNone(
            tracker.update(
                (high_rival,),
                (1080, 1920, 3),
                measurement_ns=base_ns + 24_000_000,
            )
        )
        self.assertIsNotNone(
            tracker.update(
                (high_rival,),
                (1080, 1920, 3),
                measurement_ns=base_ns + 32_000_000,
            )
        )

    def test_low_confidence_box_cannot_revive_a_dropped_track(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=0)
        high = Detection(0, "person", 0.90, (700, 250, 900, 850))
        low = Detection(0, "person", 0.18, (708, 250, 908, 850))
        base_ns = 3_500_000_000

        tracker.update((high,), (1080, 1920, 3), measurement_ns=base_ns)
        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 8_000_000,
            )
        )
        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 16_000_000,
                continuation_detections=(low,),
            )
        )

    def test_low_confidence_continuation_expires_100_ms_after_strong_measurement(
        self,
    ) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=1)
        strong = Detection(0, "person", 0.90, (700, 250, 900, 850))
        weak = Detection(0, "person", 0.18, (704, 250, 904, 850))
        base_ns = 7_000_000_000
        tracker.update((strong,), (1080, 1920, 3), measurement_ns=base_ns)

        for elapsed_ms in range(10, 101, 10):
            self.assertIsNotNone(
                tracker.update(
                    (),
                    (1080, 1920, 3),
                    measurement_ns=base_ns + elapsed_ms * 1_000_000,
                    continuation_detections=(weak,),
                )
            )
        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 110_000_000,
                continuation_detections=(weak,),
            )
        )
        self.assertEqual(
            tracker.telemetry_snapshot().continuation_measurement_samples,
            10,
        )

    def test_release_then_repress_cannot_authorize_a_weak_only_track(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=1)
        strong = Detection(0, "person", 0.90, (700, 250, 900, 850))
        weak = Detection(0, "person", 0.18, (704, 250, 904, 850))
        base_ns = 8_000_000_000
        tracker.update((strong,), (1080, 1920, 3), measurement_ns=base_ns)
        self.assertIsNotNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 8_000_000,
                continuation_detections=(weak,),
                continuation_allowed=True,
            )
        )

        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 16_000_000,
                continuation_detections=(weak,),
                continuation_allowed=False,
            )
        )
        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 24_000_000,
                continuation_detections=(weak,),
                continuation_allowed=True,
            )
        )

    def test_low_continuation_rejects_zero_iou_box_90_pixels_away(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=3)
        strong = Detection(0, "person", 0.90, (700, 250, 780, 850))
        weak = Detection(0, "person", 0.18, (790, 250, 870, 850))
        base_ns = 9_000_000_000
        tracker.update((strong,), (1080, 1920, 3), measurement_ns=base_ns)

        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 8_000_000,
                continuation_detections=(weak,),
            )
        )

    def test_low_continuation_rejects_large_box_area_jump(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=1)
        strong = Detection(0, "person", 0.90, (700, 250, 900, 850))
        weak = Detection(0, "person", 0.18, (600, 100, 1000, 1000))
        base_ns = 10_000_000_000
        tracker.update((strong,), (1080, 1920, 3), measurement_ns=base_ns)

        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 8_000_000,
                continuation_detections=(weak,),
            )
        )

    def test_continuation_allowed_requires_a_boolean(self) -> None:
        tracker = TargetTracker(label="person")
        with self.assertRaisesRegex(TypeError, "continuation_allowed must be bool"):
            tracker.update((), (1080, 1920, 3), continuation_allowed=1)

    def test_low_confidence_continuation_must_pass_geometric_association(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=0)
        high = Detection(0, "person", 0.90, (200, 250, 400, 850))
        far_low = Detection(0, "person", 0.18, (1300, 250, 1500, 850))
        base_ns = 4_500_000_000

        tracker.update((high,), (1080, 1920, 3), measurement_ns=base_ns)
        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 8_000_000,
                continuation_detections=(far_low,),
            )
        )

    def test_prediction_flag_is_true_only_for_synthetic_grace_output(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=1)
        high = Detection(0, "person", 0.90, (700, 250, 900, 850))
        continued = Detection(0, "person", 0.18, (708, 250, 908, 850))
        base_ns = 5_500_000_000

        tracker.update((high,), (1080, 1920, 3), measurement_ns=base_ns)
        self.assertFalse(tracker.output_is_prediction)
        self.assertIs(tracker.accepted_measurement, high)
        tracker.update(
            (),
            (1080, 1920, 3),
            measurement_ns=base_ns + 8_000_000,
        )
        self.assertTrue(tracker.output_is_prediction)
        self.assertIsNone(tracker.accepted_measurement)
        tracker.update(
            (),
            (1080, 1920, 3),
            measurement_ns=base_ns + 16_000_000,
            continuation_detections=(continued,),
        )
        self.assertFalse(tracker.output_is_prediction)
        self.assertIs(tracker.accepted_measurement, continued)
        tracker.reset()
        self.assertFalse(tracker.output_is_prediction)
        self.assertIsNone(tracker.accepted_measurement)

    def test_accepted_measurement_remains_raw_while_output_is_smoothed(self) -> None:
        tracker = TargetTracker(label="person")
        first = Detection(0, "person", 0.90, (700, 250, 900, 850))
        moved = Detection(0, "person", 0.85, (740, 270, 940, 870))
        base_ns = 5_750_000_000

        tracker.update(
            (first,),
            (1080, 1920, 3),
            measurement_ns=base_ns,
        )
        tracked = tracker.update(
            (moved,),
            (1080, 1920, 3),
            measurement_ns=base_ns + 8_000_000,
        )

        self.assertIsNotNone(tracked)
        assert tracked is not None
        self.assertIs(tracker.accepted_measurement, moved)
        self.assertNotEqual(tracked.xyxy, moved.xyxy)

    def test_track_generation_is_stable_through_measurement_and_prediction(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=3)
        first = Detection(0, "person", 0.9, (700, 250, 900, 850))
        moved = Detection(0, "person", 0.9, (720, 250, 920, 850))
        base_ns = 5_900_000_000

        self.assertEqual(tracker.track_generation, 0)
        tracker.update((first,), (1080, 1920, 3), measurement_ns=base_ns)
        self.assertEqual(tracker.track_generation, 1)
        tracker.update(
            (moved,),
            (1080, 1920, 3),
            measurement_ns=base_ns + 8_000_000,
        )
        self.assertEqual(tracker.track_generation, 1)
        tracker.update(
            (),
            (1080, 1920, 3),
            measurement_ns=base_ns + 16_000_000,
        )
        self.assertTrue(tracker.output_is_prediction)
        self.assertEqual(tracker.track_generation, 1)

    def test_track_generation_increments_on_reacquisition_and_reset_new_track(self) -> None:
        tracker = TargetTracker(
            label="person",
            lost_grace_frames=1,
            reacquire_confirmations=2,
        )
        first = Detection(0, "person", 0.9, (200, 250, 400, 850))
        replacement = Detection(0, "person", 0.9, (1300, 250, 1500, 850))
        base_ns = 6_000_000_000
        tracker.update((first,), (1080, 1920, 3), measurement_ns=base_ns)

        tracker.update(
            (replacement,),
            (1080, 1920, 3),
            measurement_ns=base_ns + 8_000_000,
        )
        self.assertEqual(tracker.track_generation, 1)
        tracker.update(
            (replacement,),
            (1080, 1920, 3),
            measurement_ns=base_ns + 16_000_000,
        )
        self.assertEqual(tracker.track_generation, 2)

        tracker.reset()
        self.assertEqual(tracker.track_generation, 2)
        tracker.update(
            (first,),
            (1080, 1920, 3),
            measurement_ns=base_ns + 24_000_000,
        )
        self.assertEqual(tracker.track_generation, 3)

    def test_prediction_grace_does_not_replace_an_incompatible_detection(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=3)
        original = Detection(0, "person", 0.8, (200, 300, 400, 900))
        incompatible = Detection(0, "person", 0.9, (1200, 300, 1400, 900))
        base_ns = 3_000_000_000

        tracker.update([original], (1080, 1920, 3), measurement_ns=base_ns)

        self.assertIsNone(
            tracker.update(
                [incompatible],
                (1080, 1920, 3),
                measurement_ns=base_ns + 8_000_000,
            )
        )

    def test_target_tracker_counts_one_loss_per_contiguous_missing_interval(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=0)
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))

        tracker.update([target], (1080, 1920, 3))
        tracker.update([], (1080, 1920, 3))
        tracker.update([], (1080, 1920, 3))
        self.assertEqual(tracker.telemetry_snapshot().target_loss_transitions, 1)

        tracker.update([target], (1080, 1920, 3))
        tracker.reset()
        tracker.reset()
        self.assertEqual(tracker.telemetry_snapshot().target_loss_transitions, 2)

    def test_unsafe_self_exclusion_resets_tracker_without_grace(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=3)
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        base_ns = 4_000_000_000
        tracker.update(
            [target],
            (1080, 1920, 3),
            measurement_ns=base_ns,
        )

        selected = _update_aim_target(
            tracker,
            (),
            (1080, 1920, 3),
            self_exclusion_safe=False,
            measurement_ns=base_ns + 8_000_000,
        )

        self.assertIsNone(selected)
        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 16_000_000,
            )
        )

    def test_disabled_aim_runtime_cannot_select_or_retain_a_draw_target(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=1)
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        base_ns = 5_000_000_000
        tracker.update(
            [target],
            (1080, 1920, 3),
            measurement_ns=base_ns,
        )

        selected = _update_aim_target(
            tracker,
            (target,),
            (1080, 1920, 3),
            self_exclusion_safe=True,
            aim_runtime_enabled=False,
            measurement_ns=base_ns + 8_000_000,
        )

        self.assertIsNone(selected)
        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 16_000_000,
            )
        )

    def test_non_detector_empty_sample_revokes_prediction_grace(self) -> None:
        tracker = TargetTracker(label="person", lost_grace_frames=3)
        target = Detection(0, "person", 0.8, (800, 300, 1000, 900))
        base_ns = 6_000_000_000
        tracker.update([target], (1080, 1920, 3), measurement_ns=base_ns)

        selected = _update_aim_target(
            tracker,
            (),
            (1080, 1920, 3),
            self_exclusion_safe=True,
            prediction_grace_safe=False,
            measurement_ns=base_ns + 8_000_000,
        )

        self.assertIsNone(selected)
        self.assertIsNone(
            tracker.update(
                (),
                (1080, 1920, 3),
                measurement_ns=base_ns + 16_000_000,
            )
        )

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
        telemetry = tracker.telemetry_snapshot()
        self.assertEqual(telemetry.updates, 3)
        self.assertEqual(telemetry.candidate_samples, 3)
        self.assertEqual(telemetry.measurement_samples, 2)
        self.assertEqual(telemetry.output_samples, 2)
        self.assertEqual(telemetry.compared_samples, 2)
        self.assertEqual(telemetry.target_loss_transitions, 1)

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
        telemetry = tracker.telemetry_snapshot()
        raw_x, raw_y = head_target_point(tracked_low_confidence)
        tracked_x, tracked_y = head_target_point(tracked)
        # The residual uses the low-confidence identity match actually accepted
        # by the tracker, not the closer high-confidence rival that a second
        # independent target-selection call could choose.
        self.assertEqual(telemetry.measurement_samples, 2)
        self.assertEqual(telemetry.compared_samples, 2)
        self.assertAlmostEqual(telemetry.residual_x, raw_x - tracked_x)
        self.assertAlmostEqual(telemetry.residual_y, raw_y - tracked_y)

    def test_target_tracker_keeps_established_bbox_mode_across_confidence_ties(
        self,
    ) -> None:
        # Automatic MAKCU's longer empty-only lease must not alter measured-box
        # association or its established shape-mode arbitration.
        tracker = TargetTracker(label="person", lost_grace_frames=3)
        frame_shape = (1080, 1920, 3)
        frame_height, frame_width = frame_shape[:2]
        narrow_width = frame_width * 0.078
        narrow_height = frame_height * 0.355
        wide_width = frame_width * 0.092
        wide_height = frame_height * 0.371

        def box(
            center_x: float,
            center_y: float,
            width: float,
            height: float,
        ) -> tuple[float, float, float, float]:
            return (
                center_x - width / 2.0,
                center_y - height / 2.0,
                center_x + width / 2.0,
                center_y + height / 2.0,
            )

        base_ns = 10_000_000_000
        center_x = 850.0
        center_y = 600.0
        initial = Detection(
            0,
            "person",
            0.85,
            box(center_x, center_y + 20.0, narrow_width, narrow_height),
        )
        self.assertIs(
            tracker.update(
                (initial,),
                frame_shape,
                measurement_ns=base_ns,
            ),
            initial,
        )

        tracked_head_y: list[float] = []
        for index in range(1, 9):
            # Fast genuine horizontal motion makes both overlapping boxes a
            # close geometric association. Their confidence ordering flips on
            # every sample, reproducing the observed detector mode alternation.
            center_x += 24.0
            narrow_confidence = 0.78 if index % 2 else 0.91
            wide_confidence = 0.91 if index % 2 else 0.78
            narrow = Detection(
                0,
                "person",
                narrow_confidence,
                box(center_x, center_y + 20.0, narrow_width, narrow_height),
            )
            wide = Detection(
                0,
                "person",
                wide_confidence,
                box(center_x, center_y + 10.0, wide_width, wide_height),
            )

            tracked = tracker.update(
                (wide, narrow),
                frame_shape,
                measurement_ns=base_ns + index * 8_000_000,
            )

            self.assertIsNotNone(tracked)
            assert tracked is not None
            self.assertEqual(tracked.confidence, narrow_confidence)
            self.assertAlmostEqual(tracked.x2 - tracked.x1, narrow_width)
            self.assertAlmostEqual(tracked.y2 - tracked.y1, narrow_height)
            tracked_head_y.append(head_target_point(tracked)[1])

        # Horizontal target motion must not turn confidence-only box-mode
        # flips into vertical head-point motion.
        self.assertAlmostEqual(max(tracked_head_y), min(tracked_head_y))

    def test_target_tracker_allows_sustained_gradual_scale_change_and_motion(
        self,
    ) -> None:
        tracker = TargetTracker(label="person")
        frame_shape = (1080, 1920, 3)
        base_ns = 11_000_000_000
        initial_width = 150.0
        initial_height = 384.0
        center_x = 700.0
        center_y = 600.0
        tracked = None

        for index in range(33):
            scale = 1.0 + index * 0.0125
            center_x += 6.0
            width = initial_width * scale
            height = initial_height * scale
            measurement = Detection(
                0,
                "person",
                0.85,
                (
                    center_x - width / 2.0,
                    center_y - height / 2.0,
                    center_x + width / 2.0,
                    center_y + height / 2.0,
                ),
            )
            tracked = tracker.update(
                (measurement,),
                frame_shape,
                measurement_ns=base_ns + index * 8_000_000,
            )
            self.assertIsNotNone(tracked)
            self.assertFalse(tracker.output_is_prediction)

        assert tracked is not None
        self.assertGreater(tracked.x2 - tracked.x1, initial_width * 1.30)
        self.assertGreater(tracked.y2 - tracked.y1, initial_height * 1.30)
        self.assertGreater(head_target_point(tracked)[0], 850.0)

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
                "--aim-makcu-vertical-rate-ratio",
                "0.63",
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
        self.assertEqual(config.aim_makcu_vertical_rate_ratio, 0.63)

    def test_makcu_vertical_rate_ratio_cli_is_bounded(self) -> None:
        self.assertEqual(parse_args([]).aim_makcu_vertical_rate_ratio, 0.48)
        for value in ("0", "-0.1", "1.01", "nan", "inf"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(["--aim-makcu-vertical-rate-ratio", value])

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
