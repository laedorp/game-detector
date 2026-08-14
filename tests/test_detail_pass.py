from __future__ import annotations

from math import gcd
import unittest

import numpy as np

from config import parse_args
from detection.detail_pass import (
    DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT,
    DetailPassStats,
    merge_cross_pass_detections,
    plan_detail_pass,
)
from detection.types import Detection
from utils.preprocess import preprocess_frame


try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None


def detection(
    box: tuple[float, float, float, float],
    confidence: float,
    *,
    class_id: int = 0,
) -> Detection:
    return Detection(class_id, f"class_{class_id}", confidence, box)


class DetailPassGeometryTests(unittest.TestCase):
    def test_cli_defaults_off_and_rejects_two_primary_crop_modes(self) -> None:
        self.assertIsNone(parse_args([]).detail_crop_size)
        self.assertEqual(
            parse_args(["--detail-crop-size", "768"]).detail_crop_size,
            768,
        )
        with self.assertRaises(SystemExit):
            parse_args(
                ["--crop-size", "720", "--detail-crop-size", "768"]
            )

    def test_plan_records_actual_crop_coverage_and_derived_magnification(self) -> None:
        plan = plan_detail_pass((1080, 1920, 3), 768, (416, 416))

        self.assertEqual((plan.applied_crop_width, plan.applied_crop_height), (768, 768))
        self.assertEqual((plan.crop_x, plan.crop_y), (576, 156))
        self.assertAlmostEqual(plan.coverage_fraction, 768 * 768 / (1920 * 1080))
        self.assertAlmostEqual(plan.effective_linear_magnification, 2.5)
        self.assertFalse(plan.clamped)
        self.assertFalse(plan.redundant)

    def test_rectangular_model_uses_exact_letterbox_scale_ratio(self) -> None:
        plan = plan_detail_pass((1080, 1920, 3), 768, (384, 640))

        self.assertAlmostEqual(plan.full_frame_scale, 1.0 / 3.0)
        self.assertEqual(plan.requested_crop_height, 461)
        self.assertEqual((plan.applied_crop_width, plan.applied_crop_height), (765, 459))
        self.assertAlmostEqual(plan.detail_scale, 640.0 / 765.0)
        self.assertAlmostEqual(plan.effective_linear_magnification, 2.51, places=2)
        self.assertEqual((plan.model_height, plan.model_width), (384, 640))

    @unittest.skipIf(cv2 is None, "OpenCV is not installed")
    def test_detail_tensor_box_maps_back_to_full_source_coordinates(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        plan = plan_detail_pass(frame.shape, 640, (384, 640))
        prepared = preprocess_frame(
            frame,
            (384, 640),
            crop_size=(plan.applied_crop_height, plan.applied_crop_width),
        )

        self.assertEqual(
            (prepared.transform.crop_x, prepared.transform.crop_y),
            (plan.crop_x, plan.crop_y),
        )
        # The whole tensor maps to the centered model-aspect ROI in full-frame
        # source coordinates; no square-ROI letterbox is left unused.
        self.assertEqual(
            prepared.transform.to_source_box((0, 0, 640, 384)),
            (320.0, 168.0, 960.0, 552.0),
        )

    def test_oversized_crop_is_clamped_but_widescreen_pass_is_not_redundant(self) -> None:
        plan = plan_detail_pass((1080, 1920, 3), 4000, (416, 416))

        self.assertEqual((plan.applied_crop_width, plan.applied_crop_height), (1080, 1080))
        self.assertTrue(plan.clamped)
        self.assertFalse(plan.redundant)
        self.assertAlmostEqual(plan.effective_linear_magnification, 1920 / 1080)

    def test_square_full_source_is_explicitly_redundant(self) -> None:
        plan = plan_detail_pass((720, 720, 3), 900, (416, 416))

        self.assertTrue(plan.redundant)
        self.assertEqual(plan.effective_linear_magnification, 1.0)

    def test_planned_model_aspect_roi_is_in_bounds_across_source_shapes(self) -> None:
        for model_height, model_width in (
            (416, 416),
            (384, 640),
            (448, 768),
            (320, 640),
        ):
            divisor = gcd(model_width, model_height)
            aspect_width = model_width // divisor
            aspect_height = model_height // divisor
            for source_height, source_width in (
                (1080, 1920),
                (720, 1280),
                (2160, 3840),
                (1080, 1080),
                (100, 80),
                (3, 4),
            ):
                for requested_width in (1, 31, 32, 333, 768, 4000):
                    with self.subTest(
                        model=(model_height, model_width),
                        source=(source_height, source_width),
                        requested_width=requested_width,
                    ):
                        plan = plan_detail_pass(
                            (source_height, source_width, 3),
                            requested_width,
                            (model_height, model_width),
                        )
                        self.assertGreaterEqual(plan.applied_crop_width, 1)
                        self.assertLessEqual(plan.applied_crop_width, source_width)
                        self.assertGreaterEqual(plan.applied_crop_height, 1)
                        self.assertLessEqual(plan.applied_crop_height, source_height)
                        self.assertGreaterEqual(plan.crop_x, 0)
                        self.assertLessEqual(
                            plan.crop_x + plan.applied_crop_width,
                            source_width,
                        )
                        self.assertGreaterEqual(plan.crop_y, 0)
                        self.assertLessEqual(
                            plan.crop_y + plan.applied_crop_height,
                            source_height,
                        )
                        self.assertGreaterEqual(
                            plan.effective_linear_magnification,
                            1.0,
                        )
                        exact_aspect_units_fit = min(
                            requested_width // aspect_width,
                            source_width // aspect_width,
                            source_height // aspect_height,
                        )
                        if exact_aspect_units_fit > 0:
                            self.assertEqual(
                                plan.applied_crop_width * model_height,
                                plan.applied_crop_height * model_width,
                            )

    def test_stats_record_applied_redundant_and_clamped_counts(self) -> None:
        stats = DetailPassStats(900)
        stats.record(plan_detail_pass((720, 1280, 3), 900, (416, 416)))
        stats.record(plan_detail_pass((720, 720, 3), 900, (416, 416)))

        record = stats.snapshot()

        self.assertEqual(record["frames_seen"], 2)
        self.assertEqual(record["duplicate_iou_threshold"], 0.5)
        self.assertEqual(
            record["unmatched_detail_max_reference_height"],
            DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT,
        )
        self.assertEqual(record["frames_applied"], 1)
        self.assertEqual(record["frames_redundant"], 1)
        self.assertEqual(record["frames_clamped"], 2)
        self.assertEqual(record["primary_detections"], 0)
        self.assertEqual(record["unmatched_detail_rejected_large"], 0)

    def test_stats_record_merge_rejects_invalid_counts(self) -> None:
        stats = DetailPassStats(768)
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            stats.record_merge(
                primary=1,
                detail=1,
                matches=0,
                replacements=0,
                unmatched_accepted=-1,
                unmatched_rejected_large=0,
                merged=1,
            )


class CrossPassMergeTests(unittest.TestCase):
    def test_high_overlap_duplicate_keeps_more_confident_detail_result(self) -> None:
        primary = detection((100, 100, 200, 300), 0.60)
        detail = detection((102, 102, 202, 302), 0.91)

        merged = merge_cross_pass_detections([primary], [detail])

        self.assertEqual(merged, [detail])

    def test_measured_low_overlap_same_player_duplicate_is_consolidated(self) -> None:
        primary = detection((100, 100, 200, 300), 0.60)
        # IoU is about 0.54, representative of the lower end observed between
        # full-frame and detail-pass predictions for the same labeled player.
        detail = detection((130, 100, 230, 300), 0.91)

        self.assertEqual(merge_cross_pass_detections([primary], [detail]), [detail])

    def test_exact_confidence_tie_deterministically_keeps_primary(self) -> None:
        primary = detection((100, 100, 200, 300), 0.80)
        detail = detection((101, 101, 201, 301), 0.80)

        self.assertEqual(merge_cross_pass_detections([primary], [detail]), [primary])

    def test_distinct_nearby_players_are_not_suppressed(self) -> None:
        primary = detection((100, 100, 200, 300), 0.80)
        nearby = detection((150, 105, 250, 305), 0.90)

        self.assertEqual(
            merge_cross_pass_detections([primary], [nearby]),
            [primary, nearby],
        )

    def test_different_classes_never_merge(self) -> None:
        first = detection((100, 100, 200, 300), 0.80, class_id=0)
        second = detection((100, 100, 200, 300), 0.90, class_id=1)

        self.assertEqual(merge_cross_pass_detections([first], [second]), [first, second])

    def test_one_to_one_match_preserves_unmatched_same_pass_results(self) -> None:
        primary = detection((100, 100, 200, 300), 0.70)
        detail_best = detection((101, 101, 201, 301), 0.90)
        detail_second = detection((102, 102, 202, 302), 0.80)

        merged = merge_cross_pass_detections(
            [primary],
            [detail_best, detail_second],
        )

        self.assertEqual(merged, [detail_best, detail_second])

    def test_invalid_threshold_is_rejected(self) -> None:
        for threshold in (-0.1, 1.1):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                merge_cross_pass_detections([], [], duplicate_iou=threshold)

    def test_unmatched_detail_is_limited_to_small_source_objects(self) -> None:
        primary = detection((100, 100, 200, 300), 0.80)
        small_detail = detection((400, 100, 430, 180), 0.70)
        large_detail = detection((500, 100, 620, 320), 0.90)
        stats = DetailPassStats(768)

        merged = merge_cross_pass_detections(
            [primary],
            [small_detail, large_detail],
            source_height=1080,
            unmatched_detail_max_reference_height=96,
            stats=stats,
        )

        self.assertEqual(merged, [primary, small_detail])
        record = stats.snapshot()
        self.assertEqual(record["primary_detections"], 1)
        self.assertEqual(record["detail_detections"], 2)
        self.assertEqual(record["cross_pass_matches"], 0)
        self.assertEqual(record["detail_replacements"], 0)
        self.assertEqual(record["unmatched_detail_accepted"], 1)
        self.assertEqual(record["unmatched_detail_rejected_large"], 1)
        self.assertEqual(record["merged_detections"], 2)

    def test_large_matched_detail_can_still_replace_primary(self) -> None:
        primary = detection((100, 100, 300, 500), 0.60)
        detail = detection((102, 102, 302, 502), 0.90)
        stats = DetailPassStats(768)

        merged = merge_cross_pass_detections(
            [primary],
            [detail],
            source_height=1080,
            unmatched_detail_max_reference_height=96,
            stats=stats,
        )

        self.assertEqual(merged, [detail])
        record = stats.snapshot()
        self.assertEqual(record["cross_pass_matches"], 1)
        self.assertEqual(record["detail_replacements"], 1)
        self.assertEqual(record["unmatched_detail_accepted"], 0)
        self.assertEqual(record["unmatched_detail_rejected_large"], 0)

    def test_unmatched_detail_limit_requires_complete_valid_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            merge_cross_pass_detections([], [], source_height=1080)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            merge_cross_pass_detections(
                [],
                [],
                source_height=0,
                unmatched_detail_max_reference_height=96,
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            merge_cross_pass_detections(
                [],
                [],
                source_height=1080,
                unmatched_detail_max_reference_height=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
