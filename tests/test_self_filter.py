from __future__ import annotations

import math
from contextlib import redirect_stderr
from io import StringIO
import unittest

from config import (
    DEFAULT_SELF_ZONE_HEIGHT,
    DEFAULT_SELF_ZONE_LEFT,
    DEFAULT_SELF_ZONE_WIDTH,
    parse_args,
)
from detection.types import Detection
from utils.metrics import RollingMetrics
from utils.render import console_summary
from utils.self_filter import (
    NormalizedBottomZone,
    SelfAvatarFilter,
    boxes_are_safely_distinct,
    exclude_self_avatar,
    is_obvious_bottom_shoulder_avatar,
    is_player_like,
)


class SelfFilterConfigTests(unittest.TestCase):
    def test_filter_is_opt_in_with_documented_defaults(self) -> None:
        config = parse_args([])
        self.assertFalse(config.ignore_self)
        self.assertEqual(config.preview_fps, 15.0)
        self.assertEqual(config.self_zone_left, DEFAULT_SELF_ZONE_LEFT)
        self.assertEqual(config.self_zone_width, DEFAULT_SELF_ZONE_WIDTH)
        self.assertEqual(config.self_zone_height, DEFAULT_SELF_ZONE_HEIGHT)

    def test_preview_fps_is_configurable_and_positive(self) -> None:
        self.assertEqual(parse_args(["--preview-fps", "20"]).preview_fps, 20.0)
        with self.assertRaises(SystemExit):
            parse_args(["--preview-fps", "0"])

    def test_custom_zone_is_parsed(self) -> None:
        config = parse_args(
            [
                "--ignore-self",
                "--self-zone-left",
                "0.25",
                "--self-zone-width",
                "0.4",
                "--self-zone-height",
                "0.75",
            ]
        )
        self.assertTrue(config.ignore_self)
        self.assertEqual(config.self_zone_left, 0.25)
        self.assertEqual(config.self_zone_width, 0.4)
        self.assertEqual(config.self_zone_height, 0.75)

    def test_zone_cannot_extend_past_right_edge(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(["--self-zone-left", "0.8", "--self-zone-width", "0.3"])

    def test_non_finite_and_out_of_range_values_are_rejected(self) -> None:
        bad_arguments = (
            ("--self-zone-left", "nan"),
            ("--self-zone-left", "-0.01"),
            ("--self-zone-width", "inf"),
            ("--self-zone-width", "0"),
            ("--self-zone-height", "1.01"),
        )
        for name, value in bad_arguments:
            with (
                self.subTest(name=name, value=value),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args([name, value])


class NormalizedBottomZoneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zone = NormalizedBottomZone(left=0.18, width=0.34, height=0.10)
        self.frame_shape = (1000, 2000, 3)

    @staticmethod
    def detection(box: tuple[float, float, float, float], label: str = "person") -> Detection:
        return Detection(class_id=99, class_name=label, confidence=0.9, xyxy=box)

    def test_pixel_bounds_are_bottom_anchored(self) -> None:
        x1, y1, x2, y2 = self.zone.pixel_bounds(self.frame_shape)
        self.assertEqual((x1, y1), (360, 900))
        self.assertEqual((x2, y2), (1040, 999))

    def test_custom_player_like_label_inside_zone_is_excluded(self) -> None:
        self_detection = self.detection((550, 300, 850, 1000), label="custom_avatar")
        result = exclude_self_avatar([self_detection], self.frame_shape, self.zone)
        self.assertEqual(result.detections, ())
        self.assertEqual(result.ignored_count, 1)

    def test_box_center_inside_but_bottom_center_below_zone_is_retained(self) -> None:
        # The zone starts at normalized y=0.90; this bottom edge is y=0.85.
        detection = self.detection((500, 100, 900, 850))
        result = exclude_self_avatar([detection], self.frame_shape, self.zone)
        self.assertEqual(result.detections, (detection,))
        self.assertEqual(result.ignored_count, 0)

    def test_bottom_center_outside_horizontal_zone_is_retained(self) -> None:
        opponent = self.detection((1200, 400, 1500, 950), label="player")
        result = exclude_self_avatar([opponent], self.frame_shape, self.zone)
        self.assertEqual(result.detections, (opponent,))

    def test_wide_bottom_avatar_is_obvious_only_in_configured_shoulder_band(
        self,
    ) -> None:
        left_avatar = self.detection((4, 340, 760, 1000), "player")
        right_avatar = self.detection((1240, 340, 1996, 1000), "player")

        self.assertTrue(
            is_obvious_bottom_shoulder_avatar(
                left_avatar,
                self.frame_shape,
                self.zone,
            )
        )
        self.assertFalse(
            is_obvious_bottom_shoulder_avatar(
                right_avatar,
                self.frame_shape,
                self.zone,
            )
        )

    def test_obvious_guard_covers_recorded_clipped_outer_edge_avatar(self) -> None:
        # Recorded 1920x1080 self boxes from diagnostic 20260822T032717.  Their
        # centers landed at x=.179 and x=.173, narrowly outside the configured
        # left=.18 boundary because the avatar was clipped by the screen edge.
        recorded_self_boxes = (
            (9.6, 346.2, 676.2, 1079.6),
            (18.3, 630.0, 645.8, 1080.0),
        )
        for box in recorded_self_boxes:
            with self.subTest(box=box):
                self.assertTrue(
                    is_obvious_bottom_shoulder_avatar(
                        self.detection(box, "player"),
                        (1080, 1920, 3),
                        self.zone,
                    )
                )

        # The extra outboard margin is not a generic widening toward nearby
        # players: only a box actually clipped by that screen edge may use it.
        close_opponent = self.detection(
            (100.0, 340.0, 600.0, 1000.0),
            "player",
        )
        self.assertFalse(
            is_obvious_bottom_shoulder_avatar(
                close_opponent,
                self.frame_shape,
                self.zone,
            )
        )

    def test_obvious_shoulder_guard_requires_screen_bottom_and_avatar_height(
        self,
    ) -> None:
        above_bottom = self.detection((4, 300, 760, 980), "player")
        short = self.detection((400, 800, 800, 1000), "player")
        outside_shoulders = self.detection((0, 300, 200, 1000), "player")

        for detection in (above_bottom, short, outside_shoulders):
            with self.subTest(box=detection.box):
                self.assertFalse(
                    is_obvious_bottom_shoulder_avatar(
                        detection,
                        self.frame_shape,
                        self.zone,
                    )
                )

    def test_edges_are_inclusive(self) -> None:
        # Anchor x is exactly 0.18 and bottom y exactly 0.90.
        boundary = self.detection((260, 100, 460, 900))
        self.assertTrue(self.zone.contains_box_bottom_center(boundary.box, self.frame_shape))

    def test_order_is_preserved_and_count_is_current_frame_only(self) -> None:
        left = self.detection((10, 10, 100, 500), "left")
        self_avatar = self.detection((500, 100, 900, 1000), "player")
        right = self.detection((1500, 50, 1900, 900), "right")
        result = exclude_self_avatar([left, self_avatar, right], self.frame_shape, self.zone)
        self.assertEqual(result.detections, (left, right))
        self.assertEqual(result.ignored_count, 1)

    def test_short_box_at_bottom_is_not_suppressed(self) -> None:
        opponent = self.detection((500, 800, 900, 1000), "nearby_opponent")
        result = exclude_self_avatar([opponent], self.frame_shape, self.zone)
        self.assertEqual(result.detections, (opponent,))
        self.assertEqual(result.ignored_count, 0)

    def test_ambiguous_candidates_are_all_retained(self) -> None:
        smaller = self.detection((550, 200, 850, 1000), "person")
        larger = self.detection((600, 300, 900, 1000), "player")
        result = exclude_self_avatar([smaller, larger], self.frame_shape, self.zone)
        self.assertEqual(result.detections, (smaller, larger))
        self.assertEqual(result.ignored_count, 0)
        self.assertEqual(
            result.uncertain_self_detections,
            (smaller, larger),
        )

    def test_uncertain_candidates_are_deduplicated_by_identity(self) -> None:
        avatar = self.detection((550, 200, 850, 1000), "person")

        result = exclude_self_avatar(
            [avatar, avatar],
            self.frame_shape,
            self.zone,
        )

        self.assertEqual(result.detections, (avatar, avatar))
        self.assertEqual(result.uncertain_self_detections, (avatar,))

    def test_non_player_class_is_never_suppressed(self) -> None:
        car = self.detection((500, 300, 1000, 1000), "car")
        result = exclude_self_avatar([car], self.frame_shape, self.zone)
        self.assertEqual(result.detections, (car,))
        self.assertEqual(result.ignored_count, 0)

    def test_removed_detection_is_reported_for_preview_calibration(self) -> None:
        avatar = self.detection((500, 500, 900, 1000), "custom_avatar")
        result = exclude_self_avatar([avatar], self.frame_shape, self.zone)
        self.assertIs(result.ignored_detection, avatar)

    def test_non_finite_detection_anchor_is_safely_retained(self) -> None:
        malformed = self.detection((math.nan, 0, 10, 20))
        result = exclude_self_avatar([malformed], self.frame_shape, self.zone)
        self.assertEqual(result.detections, (malformed,))

    def test_box_crossing_frame_bottom_is_clamped_and_excluded(self) -> None:
        crossing = self.detection((500, 500, 900, 1200))
        result = exclude_self_avatar([crossing], self.frame_shape, self.zone)
        self.assertEqual(result.detections, ())
        self.assertEqual(result.ignored_count, 1)

    def test_invalid_zone_values_are_rejected_directly(self) -> None:
        invalid = (
            (-0.1, 0.2, 0.2),
            (0.1, 0.0, 0.2),
            (0.1, 0.2, 0.0),
            (0.9, 0.2, 0.2),
            (math.nan, 0.2, 0.2),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                NormalizedBottomZone(*values)

    def test_invalid_frame_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.zone.pixel_bounds((0, 1920, 3))

    def test_distinct_box_check_rejects_duplicates_but_allows_separate_player(
        self,
    ) -> None:
        confirmed = (430, 560, 850, 1080)
        overlapping_duplicate = (440, 570, 860, 1080)
        nested_duplicate = (560, 700, 760, 1050)
        separate_opponent = (850, 600, 1050, 1080)

        self.assertFalse(
            boxes_are_safely_distinct(
                confirmed,
                overlapping_duplicate,
                (1080, 1920, 3),
            )
        )
        self.assertFalse(
            boxes_are_safely_distinct(
                confirmed,
                nested_duplicate,
                (1080, 1920, 3),
            )
        )
        self.assertTrue(
            boxes_are_safely_distinct(
                confirmed,
                separate_opponent,
                (1080, 1920, 3),
            )
        )

    def test_malformed_box_is_never_declared_safely_distinct(self) -> None:
        self.assertFalse(
            boxes_are_safely_distinct(
                (430, 560, 850, 1080),
                (math.nan, 600, 1050, 1080),
                (1080, 1920, 3),
            )
        )


class StatefulSelfAvatarFilterTests(unittest.TestCase):
    frame_shape = (1080, 1920, 3)

    def setUp(self) -> None:
        self.zone = NormalizedBottomZone(left=0.18, width=0.34, height=0.10)
        self.filter = SelfAvatarFilter(self.zone)
        self.avatar = self.detection((430, 560, 850, 1080), "person")

    @staticmethod
    def detection(box: tuple[float, float, float, float], label: str) -> Detection:
        return Detection(class_id=0, class_name=label, confidence=0.9, xyxy=box)

    def acquire(self) -> None:
        self.assertEqual(self.filter.apply([self.avatar], self.frame_shape).ignored_count, 0)
        self.assertEqual(self.filter.apply([self.avatar], self.frame_shape).ignored_count, 0)
        self.assertEqual(self.filter.apply([self.avatar], self.frame_shape).ignored_count, 1)

    def test_stable_avatar_requires_three_frames_before_suppression(self) -> None:
        self.acquire()
        self.assertTrue(self.filter.acquired)
        self.assertEqual(self.filter.apply([self.avatar], self.frame_shape).ignored_count, 1)

    def test_aim_can_fail_closed_until_self_is_actually_removed(self) -> None:
        enemy = self.detection((1200, 250, 1500, 850), "person")
        first = self.filter.apply([self.avatar, enemy], self.frame_shape)
        second = self.filter.apply([self.avatar, enemy], self.frame_shape)
        third = self.filter.apply([self.avatar, enemy], self.frame_shape)

        self.assertEqual(first.ignored_count, 0)
        self.assertEqual(second.ignored_count, 0)
        self.assertEqual(third.ignored_count, 1)
        self.assertFalse(first.aim_safe)
        self.assertFalse(second.aim_safe)
        self.assertTrue(third.aim_safe)
        self.assertEqual(first.uncertain_self_detections, (self.avatar,))
        self.assertEqual(second.uncertain_self_detections, (self.avatar,))
        self.assertEqual(third.uncertain_self_detections, ())
        self.assertEqual(third.detections, (enemy,))
        self.assertIs(third.ignored_detection, self.avatar)

    def test_aim_is_allowed_during_acquisition_when_no_self_candidate_is_visible(self) -> None:
        enemy_outside_self_zone = self.detection((1200, 250, 1500, 850), "person")

        first = self.filter.apply([enemy_outside_self_zone], self.frame_shape)
        second = self.filter.apply([enemy_outside_self_zone], self.frame_shape)

        self.assertTrue(first.aim_safe)
        self.assertTrue(second.aim_safe)
        self.assertEqual(first.detections, (enemy_outside_self_zone,))
        self.assertEqual(second.detections, (enemy_outside_self_zone,))

    def test_short_self_dropout_keeps_enemy_aim_available(self) -> None:
        self.acquire()
        enemy = self.detection((1200, 250, 1500, 850), "person")

        first = self.filter.apply([enemy], self.frame_shape)
        second = self.filter.apply([enemy], self.frame_shape)

        self.assertTrue(first.aim_safe)
        self.assertTrue(second.aim_safe)
        self.assertEqual(first.detections, (enemy,))
        self.assertEqual(second.detections, (enemy,))
        self.assertEqual(first.uncertain_self_detections, ())
        self.assertEqual(second.uncertain_self_detections, ())

    def test_ambiguous_self_overlap_blocks_aim(self) -> None:
        self.acquire()
        duplicate = self.detection((440, 570, 860, 1080), "person")
        result = self.filter.apply([self.avatar, duplicate], self.frame_shape)
        self.assertFalse(result.aim_safe)
        self.assertEqual(
            result.uncertain_self_detections,
            (self.avatar, duplicate),
        )

    def test_opposite_shoulder_avatar_blocks_aim_until_reacquired(self) -> None:
        self.acquire()
        opposite_shoulder = self.detection((1200, 550, 1450, 1080), "person")

        first = self.filter.apply([opposite_shoulder], self.frame_shape)
        second = self.filter.apply([opposite_shoulder], self.frame_shape)
        third = self.filter.apply([opposite_shoulder], self.frame_shape)

        self.assertFalse(first.aim_safe)
        self.assertFalse(second.aim_safe)
        self.assertEqual(first.uncertain_self_detections, (opposite_shoulder,))
        self.assertEqual(second.uncertain_self_detections, (opposite_shoulder,))
        self.assertEqual(first.detections, (opposite_shoulder,))
        self.assertEqual(second.detections, (opposite_shoulder,))
        self.assertTrue(third.aim_safe)
        self.assertEqual(third.uncertain_self_detections, ())
        self.assertEqual(third.detections, ())
        self.assertIs(third.ignored_detection, opposite_shoulder)

    def test_bottom_opponent_outside_selected_shoulder_is_not_suppressed(self) -> None:
        opponent = self.detection((1300, 430, 1550, 1080), "person")

        result = self.filter.apply([opponent], self.frame_shape)

        self.assertEqual(result.detections, (opponent,))
        self.assertEqual(result.ignored_count, 0)
        self.assertTrue(result.aim_safe)

    def test_one_or_two_frame_transient_is_retained(self) -> None:
        first = self.filter.apply([self.avatar], self.frame_shape)
        second = self.filter.apply([self.avatar], self.frame_shape)
        self.assertEqual(first.detections, (self.avatar,))
        self.assertEqual(second.detections, (self.avatar,))

    def test_ambiguous_startup_never_acquires(self) -> None:
        opponent = self.detection((350, 400, 700, 1080), "person")
        for _ in range(5):
            result = self.filter.apply([self.avatar, opponent], self.frame_shape)
            self.assertEqual(result.ignored_count, 0)
            self.assertEqual(
                result.uncertain_self_detections,
                (self.avatar, opponent),
            )
        self.assertFalse(self.filter.acquired)

    def test_taller_opponent_does_not_steal_existing_lock(self) -> None:
        self.acquire()
        opponent = self.detection((40, 100, 650, 1080), "person")
        jittered_avatar = self.detection((440, 570, 860, 1080), "person")
        result = self.filter.apply(
            [opponent, jittered_avatar],
            self.frame_shape,
        )
        self.assertEqual(result.detections, (opponent,))
        self.assertIs(result.ignored_detection, jittered_avatar)

    def test_material_track_jump_requires_three_fresh_confirmations(self) -> None:
        self.acquire()
        possible_handoff = self.detection((500, 500, 920, 1050), "person")
        first = self.filter.apply([possible_handoff], self.frame_shape)
        second = self.filter.apply([possible_handoff], self.frame_shape)
        third = self.filter.apply([possible_handoff], self.frame_shape)
        self.assertEqual(first.detections, (possible_handoff,))
        self.assertEqual(second.detections, (possible_handoff,))
        self.assertEqual(first.uncertain_self_detections, (possible_handoff,))
        self.assertEqual(second.uncertain_self_detections, (possible_handoff,))
        self.assertEqual(third.uncertain_self_detections, ())
        self.assertEqual(third.detections, ())
        self.assertIs(third.ignored_detection, possible_handoff)

    def test_reordered_detections_follow_the_locked_box(self) -> None:
        self.acquire()
        outside = self.detection((1200, 200, 1700, 1000), "person")
        result = self.filter.apply([outside, self.avatar], self.frame_shape)
        self.assertEqual(result.detections, (outside,))

    def test_acquired_avatar_is_tracked_just_outside_anchor_zone(self) -> None:
        self.acquire()
        raised_avatar = self.detection((430, 520, 850, 970), "person")
        result = self.filter.apply([raised_avatar], self.frame_shape)
        self.assertTrue(result.aim_safe)
        self.assertEqual(result.ignored_count, 1)
        self.assertEqual(result.detections, ())
        self.assertIs(result.ignored_detection, raised_avatar)

    def test_two_frame_dropout_keeps_lock_without_suppressing_anything(self) -> None:
        self.acquire()
        self.assertEqual(self.filter.apply([], self.frame_shape).ignored_count, 0)
        self.assertEqual(self.filter.apply([], self.frame_shape).ignored_count, 0)
        self.assertEqual(self.filter.apply([self.avatar], self.frame_shape).ignored_count, 1)

    def test_abrupt_merged_box_is_not_suppressed(self) -> None:
        self.acquire()
        merged = self.detection((100, 100, 950, 1080), "person")
        result = self.filter.apply([merged], self.frame_shape)
        self.assertEqual(result.detections, (merged,))
        self.assertEqual(result.ignored_count, 0)
        self.assertFalse(result.aim_safe)
        self.assertEqual(result.uncertain_self_detections, (merged,))

    def test_wide_bottom_handoff_does_not_taint_distinct_opponent(self) -> None:
        self.acquire()
        wide_self = self.detection((4, 370, 750, 1080), "person")
        distinct_opponent = self.detection((903, 438, 954, 522), "person")

        result = self.filter.apply(
            [wide_self, distinct_opponent],
            self.frame_shape,
        )

        self.assertFalse(result.aim_safe)
        self.assertEqual(
            result.detections,
            (wide_self, distinct_opponent),
        )
        self.assertEqual(result.uncertain_self_detections, (wide_self,))

    def test_non_player_never_enters_acquisition(self) -> None:
        vehicle = self.detection((430, 400, 900, 1080), "car")
        for _ in range(4):
            self.assertEqual(
                self.filter.apply([vehicle], self.frame_shape).ignored_count,
                0,
            )
        self.assertFalse(self.filter.acquired)

    def test_explicit_enemy_class_is_never_suppressed(self) -> None:
        enemy = self.detection((430, 400, 900, 1080), "enemy")
        for _ in range(5):
            result = self.filter.apply([enemy], self.frame_shape)
            self.assertEqual(result.detections, (enemy,))
            self.assertEqual(result.ignored_count, 0)

    def test_resolution_change_requires_reacquisition(self) -> None:
        self.acquire()
        resized_shape = (720, 1280, 3)
        resized_avatar = self.detection((287, 373, 567, 720), "person")
        self.assertEqual(self.filter.apply([resized_avatar], resized_shape).ignored_count, 0)
        self.assertEqual(self.filter.apply([resized_avatar], resized_shape).ignored_count, 0)
        self.assertEqual(self.filter.apply([resized_avatar], resized_shape).ignored_count, 1)

    def test_player_label_patterns_are_explicit(self) -> None:
        self.assertTrue(is_player_like(self.detection((0, 0, 1, 1), "player_2")))
        self.assertTrue(is_player_like(self.detection((0, 0, 1, 1), "custom_avatar")))
        self.assertFalse(is_player_like(self.detection((0, 0, 1, 1), "enemy_player_2")))
        self.assertFalse(is_player_like(self.detection((0, 0, 1, 1), "enemy2_player")))
        self.assertFalse(is_player_like(self.detection((0, 0, 1, 1), "NPC")))
        self.assertFalse(is_player_like(self.detection((0, 0, 1, 1), "sports car")))


class SelfFilterReportingTests(unittest.TestCase):
    def test_current_ignored_count_is_separate_from_skipped_frames(self) -> None:
        snapshot = RollingMetrics(2).snapshot()
        summary = console_summary(snapshot, skipped_frames=7, ignored_count=1)
        self.assertIn("skipped 7", summary)
        self.assertIn("self ignored 1", summary)

    def test_disabled_filter_does_not_add_ignored_field(self) -> None:
        snapshot = RollingMetrics(2).snapshot()
        summary = console_summary(snapshot, skipped_frames=0)
        self.assertNotIn("self ignored", summary)


if __name__ == "__main__":
    unittest.main()
