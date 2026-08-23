from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from detection.base import OutputDecodeError
from detection.head_detector import (
    DIRECT_HEAD_RUNTIME_MANIFEST_ENV,
    DEFAULT_CLOSE_PLAYER_CROP_SCALE,
    DEFAULT_CROP_SCALE,
    DEFAULT_HEAD_CONFIDENCE,
    HEAD_CLASS_ID,
    HEAD_OUTPUT_CANDIDATES,
    MAX_HEAD_DETECTIONS,
    MAX_HEAD_NMS_CANDIDATES,
    PINNED_HEAD_MODEL_RELATIVE_PATH,
    PLAYER_CLASS_ID,
    DirectHeadLocalizer,
    HeadModelSpec,
    HeadAssociationOutcome,
    HeadCandidate,
    HeadCropTransform,
    adaptive_head_crop_scale,
    associate_head_to_player,
    associate_head_to_player_outcome,
    decode_head_output,
    plan_head_crop,
    pinned_head_model_path,
    prepare_head_input,
    runtime_head_model_spec,
    verify_pinned_head_model,
)
from detection.head_worker import HeadLocalizationReason


try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None


def model_output(
    *rows: tuple[tuple[float, float, float, float], float, float],
) -> np.ndarray:
    output = np.zeros((1, 6, HEAD_OUTPUT_CANDIDATES), dtype=np.float32)
    for index, (box, player_score, head_score) in enumerate(rows):
        x1, y1, x2, y2 = box
        output[0, :, index] = (
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            x2 - x1,
            y2 - y1,
            player_score,
            head_score,
        )
    return output


def candidate(
    box: tuple[float, float, float, float],
    confidence: float,
    *,
    class_id: int = HEAD_CLASS_ID,
    row_index: int = 0,
) -> HeadCandidate:
    return HeadCandidate(
        class_id=class_id,
        class_name="head" if class_id == HEAD_CLASS_ID else "player",
        confidence=confidence,
        box=box,
        row_index=row_index,
    )


class HeadCropTests(unittest.TestCase):
    def test_adaptive_crop_uses_detail_only_for_close_player(self) -> None:
        self.assertEqual(
            adaptive_head_crop_scale(
                (1080, 1920, 3),
                (100.0, 100.0, 200.0, 299.0),
            ),
            DEFAULT_CROP_SCALE,
        )
        self.assertEqual(
            adaptive_head_crop_scale(
                (1080, 1920, 3),
                (100.0, 100.0, 200.0, 300.0),
            ),
            DEFAULT_CLOSE_PLAYER_CROP_SCALE,
        )

    def test_adaptive_crop_threshold_scales_with_frame_height(self) -> None:
        self.assertEqual(
            adaptive_head_crop_scale(
                (540, 960, 3),
                (100.0, 100.0, 200.0, 200.0),
            ),
            DEFAULT_CLOSE_PLAYER_CROP_SCALE,
        )

    def test_adaptive_crop_hysteresis_prevents_boundary_flapping(self) -> None:
        self.assertEqual(
            adaptive_head_crop_scale(
                (1080, 1920, 3),
                (100.0, 100.0, 200.0, 285.0),
                previous_crop_scale=DEFAULT_CLOSE_PLAYER_CROP_SCALE,
            ),
            DEFAULT_CLOSE_PLAYER_CROP_SCALE,
        )
        self.assertEqual(
            adaptive_head_crop_scale(
                (1080, 1920, 3),
                (100.0, 100.0, 200.0, 279.0),
                previous_crop_scale=DEFAULT_CLOSE_PLAYER_CROP_SCALE,
            ),
            DEFAULT_CROP_SCALE,
        )

    def test_crop_is_square_in_bounds_and_keeps_edge_player(self) -> None:
        crop = plan_head_crop(
            (720, 1280, 3),
            (1180.0, 500.0, 1270.0, 700.0),
        )

        self.assertEqual((crop.crop_width, crop.crop_height), (400, 400))
        self.assertEqual((crop.crop_x, crop.crop_y), (880, 320))
        self.assertEqual((crop.resized_width, crop.resized_height), (320, 320))
        self.assertEqual((crop.pad_left, crop.pad_top), (0, 0))
        self.assertAlmostEqual(crop.scale, 0.8)

    def test_rectangular_source_extent_is_letterboxed(self) -> None:
        crop = plan_head_crop((100, 1000, 3), (100, 10, 600, 90))

        self.assertEqual(
            (crop.crop_x, crop.crop_y, crop.crop_width, crop.crop_height),
            (0, 0, 1000, 100),
        )
        self.assertEqual((crop.resized_width, crop.resized_height), (320, 32))
        self.assertEqual((crop.pad_left, crop.pad_top), (0, 144))

    def test_letterbox_transform_round_trip(self) -> None:
        crop = plan_head_crop(
            (100, 1000, 3),
            (100, 10, 600, 90),
        )

        self.assertTrue(
            np.allclose(
                crop.to_source_box((0, 144, 320, 176)),
                (0, 0, 1000, 100),
            )
        )

    @unittest.skipIf(cv2 is None, "OpenCV is not installed")
    def test_preprocess_uses_114_letterbox_and_rgb_chw_float(self) -> None:
        frame = np.empty((2, 4, 3), dtype=np.uint8)
        frame[:] = (10, 20, 30)
        transform = plan_head_crop(
            frame.shape,
            (0, 0, 4, 2),
            crop_scale=1,
            min_crop_side=1,
            model_size=(4, 4),
        )

        prepared = prepare_head_input(frame, transform)

        self.assertEqual(prepared.tensor.shape, (1, 3, 4, 4))
        self.assertEqual(prepared.tensor.dtype, np.float32)
        self.assertTrue(
            np.allclose(prepared.tensor[0, :, 0, 0], 114 / np.float32(255))
        )
        self.assertTrue(
            np.allclose(
                prepared.tensor[0, :, 1, 0],
                (30, 20, 10) / np.float32(255),
            )
        )

    def test_box_outside_source_fails_before_crop(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not intersect"):
            plan_head_crop((100, 100, 3), (120, 20, 140, 60))


class PinnedHeadModelTests(unittest.TestCase):
    def test_pinned_path_is_relative_to_explicit_project_root(self) -> None:
        self.assertEqual(
            pinned_head_model_path("/opt/proaim"),
            Path("/opt/proaim") / PINNED_HEAD_MODEL_RELATIVE_PATH,
        )

    def test_pinned_path_preserves_explicit_project_root_spelling(self) -> None:
        root = Path("portable-root") / "nested" / ".."

        self.assertEqual(
            pinned_head_model_path(root),
            root / PINNED_HEAD_MODEL_RELATIVE_PATH,
        )

    def test_verifier_checks_both_size_and_sha256(self) -> None:
        payload = b"pinned-head-model"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "head.onnx"
            path.write_bytes(payload)
            import hashlib

            expected = hashlib.sha256(payload).hexdigest()
            with (
                patch(
                    "detection.head_detector.PINNED_HEAD_MODEL_SIZE_BYTES",
                    len(payload),
                ),
                patch(
                    "detection.head_detector.PINNED_HEAD_MODEL_SHA256",
                    expected,
                ),
            ):
                self.assertEqual(verify_pinned_head_model(path), path)
                path.write_bytes(payload + b"x")
                with self.assertRaisesRegex(ValueError, "size mismatch"):
                    verify_pinned_head_model(path)

    def test_verifier_rejects_same_size_wrong_digest(self) -> None:
        payload = b"same-size"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "head.onnx"
            path.write_bytes(payload)
            with (
                patch(
                    "detection.head_detector.PINNED_HEAD_MODEL_SIZE_BYTES",
                    len(payload),
                ),
                patch(
                    "detection.head_detector.PINNED_HEAD_MODEL_SHA256",
                    "0" * 64,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    verify_pinned_head_model(path)

    def test_runtime_override_manifest_resolves_verified_contract(self) -> None:
        payload = b"override-head-model"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "override.onnx"
            model.write_bytes(payload)
            import hashlib

            manifest = root / "runtime-head.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model": str(model),
                        "model_size_bytes": len(payload),
                        "model_sha256": hashlib.sha256(payload).hexdigest(),
                        "input_shape_nchw": [1, 3, 640, 640],
                        "output_shape": [1, 6, 8400],
                        "model_name": "Nightly head 640",
                        "evidence_label": "Nightly head box",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {DIRECT_HEAD_RUNTIME_MANIFEST_ENV: str(manifest)},
                clear=False,
            ):
                spec = runtime_head_model_spec(root)

        self.assertIsInstance(spec, HeadModelSpec)
        self.assertEqual(spec.path, model)
        self.assertEqual(spec.input_shape, (1, 3, 640, 640))
        self.assertEqual(spec.output_shape, (1, 6, 8400))
        self.assertEqual(spec.model_name, "Nightly head 640")
        self.assertEqual(spec.evidence_label, "Nightly head box")
        self.assertEqual(spec.confidence_threshold, 0.15)

    def test_runtime_override_preserves_verified_model_path_spelling(self) -> None:
        payload = b"override-head-model"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "existing").mkdir()
            model = root / "existing" / ".." / "override.onnx"
            model.write_bytes(payload)
            import hashlib

            manifest = root / "runtime-head.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model": str(model),
                        "model_size_bytes": len(payload),
                        "model_sha256": hashlib.sha256(payload).hexdigest(),
                        "input_shape_nchw": [1, 3, 640, 640],
                        "output_shape": [1, 6, 8400],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {DIRECT_HEAD_RUNTIME_MANIFEST_ENV: str(manifest)},
                clear=False,
            ):
                spec = runtime_head_model_spec(root)

        self.assertEqual(spec.path, model)


class HeadDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = HeadCropTransform(
            crop_x=100,
            crop_y=50,
            crop_width=320,
            crop_height=320,
            source_width=1280,
            source_height=720,
            resized_width=320,
            resized_height=320,
            pad_left=0,
            pad_top=0,
            scale=1.0,
        )

    def test_decodes_exact_channel_first_contract_and_maps_classes(self) -> None:
        output = model_output(
            ((20, 20, 220, 300), 0.80, 0.05),
            ((80, 30, 120, 70), 0.04, 0.91),
        )

        decoded = decode_head_output(output, self.transform)

        self.assertEqual(
            [item.class_id for item in decoded],
            [HEAD_CLASS_ID, PLAYER_CLASS_ID],
        )
        self.assertEqual([item.row_index for item in decoded], [1, 0])
        self.assertTrue(np.allclose(decoded[0].box, (180, 80, 220, 120)))
        self.assertTrue(np.allclose(decoded[1].box, (120, 70, 320, 350)))

    def test_default_threshold_is_selected_safe_recall_floor(self) -> None:
        self.assertEqual(DEFAULT_HEAD_CONFIDENCE, 0.15)
        output = model_output(((80, 30, 120, 70), 0.01, 0.27))

        decoded = decode_head_output(output, self.transform)

        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0].class_id, HEAD_CLASS_ID)
        self.assertAlmostEqual(decoded[0].confidence, 0.27, places=6)
        self.assertEqual(
            decode_head_output(
                model_output(((80, 30, 120, 70), 0.01, 0.14)),
                self.transform,
            ),
            [],
        )

    def test_nms_is_class_aware_and_suppresses_only_duplicate_heads(self) -> None:
        output = model_output(
            ((80, 30, 120, 70), 0.92, 0.02),
            ((80, 30, 120, 70), 0.02, 0.85),
            ((82, 32, 122, 72), 0.01, 0.90),
        )

        decoded = decode_head_output(output, self.transform)

        self.assertEqual(len(decoded), 2)
        self.assertEqual(
            [(item.class_id, round(item.confidence, 2)) for item in decoded],
            [(PLAYER_CLASS_ID, 0.92), (HEAD_CLASS_ID, 0.90)],
        )

    def test_dense_player_rows_cannot_starve_a_valid_head_candidate(self) -> None:
        output = np.zeros((1, 6, HEAD_OUTPUT_CANDIDATES), dtype=np.float32)
        for index in range(200):
            x = 2.0 + (index % 16) * 19.0
            y = 2.0 + (index // 16) * 19.0
            output[0, :, index] = (x, y, 2, 2, 0.99, 0.01)
        output[0, :, 500] = (100, 50, 40, 40, 0.01, 0.80)

        decoded = decode_head_output(output, self.transform)

        self.assertTrue(
            any(
                item.class_id == HEAD_CLASS_ID and item.row_index == 500
                for item in decoded
            )
        )

    def test_low_nonfinite_impossible_and_out_of_crop_rows_are_skipped(self) -> None:
        output = model_output(
            ((10, 10, 30, 30), 0.01, 0.14),
            ((30, 30, 50, 50), 0.01, 0.90),
            ((60, 60, 60, 80), 0.01, 0.95),
            ((-40, 10, -20, 30), 0.01, 0.95),
            ((90, 90, 110, 110), 0.01, 1.20),
        )
        output[0, 0, 1] = np.nan

        self.assertEqual(decode_head_output(output, self.transform), [])

    def test_wrong_or_ambiguous_output_contract_is_rejected(self) -> None:
        outputs = (
            np.zeros((1, 5, 2100), dtype=np.float32),
            np.zeros((1, 2100, 6), dtype=np.float32),
            np.zeros((6, 2100), dtype=np.float32),
            np.full((1, 6, 2100), "x", dtype="<U1"),
        )
        for output in outputs:
            with self.subTest(shape=output.shape, dtype=output.dtype):
                with self.assertRaises(OutputDecodeError):
                    decode_head_output(output, self.transform)

    def test_candidate_and_result_work_are_hard_bounded(self) -> None:
        output = np.zeros((1, 6, HEAD_OUTPUT_CANDIDATES), dtype=np.float32)
        for index in range(200):
            x = 2.0 + (index % 16) * 19.0
            y = 2.0 + (index // 16) * 19.0
            output[0, :, index] = (x, y, 2, 2, 0.01, 0.90)

        decoded = decode_head_output(output, self.transform)

        self.assertEqual(len(decoded), MAX_HEAD_DETECTIONS)
        self.assertLess(
            max(item.row_index for item in decoded),
            MAX_HEAD_NMS_CANDIDATES,
        )
        with self.assertRaises(ValueError):
            decode_head_output(
                output,
                self.transform,
                max_detections=MAX_HEAD_DETECTIONS + 1,
            )


class HeadAssociationTests(unittest.TestCase):
    def test_outcome_reports_localized_and_is_frozen(self) -> None:
        supporting_player = candidate(
            (95, 95, 205, 305),
            0.88,
            class_id=PLAYER_CLASS_ID,
            row_index=1,
        )
        outcome = associate_head_to_player_outcome(
            [supporting_player, candidate((135, 105, 165, 135), 0.75)],
            (100, 100, 200, 300),
            source_timestamp_ns=10,
        )

        self.assertIsInstance(outcome, HeadAssociationOutcome)
        self.assertIs(outcome.reason, HeadLocalizationReason.LOCALIZED)
        self.assertIsNotNone(outcome.localization)
        with self.assertRaises(FrozenInstanceError):
            outcome.reason = (  # type: ignore[misc]
                HeadLocalizationReason.NO_PLAUSIBLE_HEAD
            )

    def test_outcome_reports_no_decoded_head_candidate(self) -> None:
        player = candidate(
            (95, 95, 205, 305),
            0.88,
            class_id=PLAYER_CLASS_ID,
        )

        outcome = associate_head_to_player_outcome(
            [player],
            (100, 100, 200, 300),
            source_timestamp_ns=11,
        )

        self.assertIs(
            outcome.reason,
            HeadLocalizationReason.NO_DECODED_HEAD_CANDIDATE,
        )
        self.assertIsNone(outcome.localization)

    def test_outcome_reports_no_plausible_head(self) -> None:
        supporting_player = candidate(
            (95, 95, 205, 305),
            0.88,
            class_id=PLAYER_CLASS_ID,
        )
        torso_head = candidate((135, 220, 165, 250), 0.91)

        outcome = associate_head_to_player_outcome(
            [supporting_player, torso_head],
            (100, 100, 200, 300),
            source_timestamp_ns=12,
        )

        self.assertIs(outcome.reason, HeadLocalizationReason.NO_PLAUSIBLE_HEAD)
        self.assertIsNone(outcome.localization)

    def test_outcome_reports_multiple_plausible_heads(self) -> None:
        outcome = associate_head_to_player_outcome(
            [
                candidate((120, 105, 145, 135), 0.80),
                candidate((160, 105, 185, 135), 0.90, row_index=1),
            ],
            (100, 100, 200, 300),
            source_timestamp_ns=13,
        )

        self.assertIs(
            outcome.reason,
            HeadLocalizationReason.MULTIPLE_PLAUSIBLE_HEADS,
        )
        self.assertIsNone(outcome.localization)

    def test_outcome_reports_no_matching_secondary_player(self) -> None:
        outcome = associate_head_to_player_outcome(
            [candidate((135, 105, 165, 135), 0.90)],
            (100, 100, 200, 300),
            source_timestamp_ns=14,
        )

        self.assertIs(
            outcome.reason,
            HeadLocalizationReason.NO_MATCHING_SECONDARY_PLAYER,
        )
        self.assertIsNone(outcome.localization)

    def test_outcome_reports_multiple_matching_secondary_players(self) -> None:
        first_player = candidate(
            (95, 95, 175, 310),
            0.90,
            class_id=PLAYER_CLASS_ID,
        )
        second_player = candidate(
            (125, 95, 205, 310),
            0.91,
            class_id=PLAYER_CLASS_ID,
            row_index=1,
        )
        head = candidate((135, 105, 165, 135), 0.80, row_index=2)

        outcome = associate_head_to_player_outcome(
            [first_player, second_player, head],
            (100, 100, 200, 300),
            source_timestamp_ns=15,
        )

        self.assertIs(
            outcome.reason,
            HeadLocalizationReason.MULTIPLE_MATCHING_SECONDARY_PLAYERS,
        )
        self.assertIsNone(outcome.localization)

    def test_nested_multiscale_secondary_pair_supports_one_head(self) -> None:
        target = (100.0, 100.0, 200.0, 300.0)
        full_scale = candidate(
            target,
            0.90,
            class_id=PLAYER_CLASS_ID,
            row_index=30,
        )
        nested_scale = candidate(
            (110.0, 120.0, 190.0, 220.0),
            0.80,
            class_id=PLAYER_CLASS_ID,
            row_index=31,
        )
        sole_head = candidate(
            (135.0, 125.0, 165.0, 150.0),
            0.75,
            row_index=32,
        )

        outcome = associate_head_to_player_outcome(
            [full_scale, nested_scale, sole_head],
            target,
            source_timestamp_ns=33,
        )

        self.assertIs(outcome.reason, HeadLocalizationReason.LOCALIZED)
        assert outcome.localization is not None
        self.assertEqual(outcome.localization.point, (150.0, 137.5))
        self.assertEqual(outcome.localization.supporting_player_index, 0)

    def test_nested_secondary_pair_requires_both_boxes_to_support_head(self) -> None:
        target = (100.0, 100.0, 200.0, 300.0)
        full_scale = candidate(
            target,
            0.90,
            class_id=PLAYER_CLASS_ID,
            row_index=33,
        )
        nested_scale = candidate(
            (110.0, 120.0, 190.0, 220.0),
            0.80,
            class_id=PLAYER_CLASS_ID,
            row_index=34,
        )
        head_outside_nested = candidate(
            (101.0, 125.0, 108.0, 150.0),
            0.75,
            row_index=35,
        )

        outcome = associate_head_to_player_outcome(
            [full_scale, nested_scale, head_outside_nested],
            target,
            source_timestamp_ns=36,
        )

        self.assertIs(
            outcome.reason,
            HeadLocalizationReason.MULTIPLE_MATCHING_SECONDARY_PLAYERS,
        )
        self.assertIsNone(outcome.localization)

    def test_three_nested_secondary_players_remain_ambiguous(self) -> None:
        target = (100.0, 100.0, 200.0, 300.0)
        players = [
            candidate(
                target,
                0.90,
                class_id=PLAYER_CLASS_ID,
                row_index=40,
            ),
            candidate(
                (110.0, 120.0, 190.0, 220.0),
                0.80,
                class_id=PLAYER_CLASS_ID,
                row_index=41,
            ),
            candidate(
                (111.0, 121.0, 189.0, 219.0),
                0.70,
                class_id=PLAYER_CLASS_ID,
                row_index=42,
            ),
        ]
        sole_head = candidate(
            (135.0, 125.0, 165.0, 150.0),
            0.75,
            row_index=43,
        )

        outcome = associate_head_to_player_outcome(
            [*players, sole_head],
            target,
            source_timestamp_ns=44,
        )

        self.assertIs(
            outcome.reason,
            HeadLocalizationReason.MULTIPLE_MATCHING_SECONDARY_PLAYERS,
        )
        self.assertIsNone(outcome.localization)

    def test_nested_secondary_pair_enforces_every_geometric_boundary(self) -> None:
        target = (100.0, 100.0, 200.0, 300.0)
        sole_head = candidate(
            (135.0, 140.0, 165.0, 160.0),
            0.75,
            row_index=50,
        )
        invalid_pairs = {
            # Nested area ratio/IoU is 0.35, just below the 0.36 floor.
            "iou": (
                target,
                (115.0, 120.0, 185.0, 220.0),
            ),
            # Both boxes move together, but one overlaps only 89% of the
            # primary's smaller area.
            "primary-overlap": (
                (89.0, 100.0, 189.0, 300.0),
                (99.0, 120.0, 179.0, 220.0),
            ),
            # A nested box whose top is more than 0.18 target heights away.
            "top-offset": (
                target,
                (110.0, 137.0, 190.0, 237.0),
            ),
        }

        for label, boxes in invalid_pairs.items():
            with self.subTest(label=label):
                players = [
                    candidate(
                        box,
                        0.90 - index * 0.10,
                        class_id=PLAYER_CLASS_ID,
                        row_index=51 + index,
                    )
                    for index, box in enumerate(boxes)
                ]
                outcome = associate_head_to_player_outcome(
                    [*players, sole_head],
                    target,
                    source_timestamp_ns=53,
                )
                self.assertIs(
                    outcome.reason,
                    HeadLocalizationReason.MULTIPLE_MATCHING_SECONDARY_PLAYERS,
                )
                self.assertIsNone(outcome.localization)

    def test_outcome_reports_head_unsupported_by_matched_player(self) -> None:
        supporting_player = candidate(
            (95, 95, 170, 305),
            0.90,
            class_id=PLAYER_CLASS_ID,
        )
        head_outside_support = candidate((180, 105, 195, 135), 0.88)

        outcome = associate_head_to_player_outcome(
            [supporting_player, head_outside_support],
            (100, 100, 200, 300),
            source_timestamp_ns=16,
        )

        self.assertIs(
            outcome.reason,
            HeadLocalizationReason.HEAD_UNSUPPORTED_BY_MATCHED_PLAYER,
        )
        self.assertIsNone(outcome.localization)

    def test_unique_matching_player_and_head_are_accepted(self) -> None:
        other = candidate((290, 90, 330, 130), 0.99, row_index=3)
        selected = candidate((135, 105, 165, 135), 0.75, row_index=4)
        supporting_player = candidate(
            (95, 95, 205, 305),
            0.88,
            class_id=PLAYER_CLASS_ID,
            row_index=5,
        )

        result = associate_head_to_player(
            [other, selected, supporting_player],
            (100, 100, 200, 300),
            source_timestamp_ns=10,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.candidate_index, 1)
        self.assertEqual(result.point, (150.0, 120.0))
        self.assertEqual(result.confidence, 0.75)
        self.assertEqual(result.supporting_player_index, 2)

    def test_single_plausible_head_without_supporting_player_fails_closed(self) -> None:
        sole_neighbor_head = candidate(
            (135, 105, 165, 135),
            0.99,
            row_index=6,
        )

        self.assertIsNone(
            associate_head_to_player(
                [sole_neighbor_head],
                (100, 100, 200, 300),
                source_timestamp_ns=11,
            )
        )

    def test_unmatched_secondary_player_does_not_support_lone_head(self) -> None:
        unrelated_player = candidate(
            (230, 90, 310, 300),
            0.95,
            class_id=PLAYER_CLASS_ID,
            row_index=7,
        )
        sole_head = candidate((135, 105, 165, 135), 0.99, row_index=8)

        self.assertIsNone(
            associate_head_to_player(
                [unrelated_player, sole_head],
                (100, 100, 200, 300),
                source_timestamp_ns=12,
            )
        )

    def test_player_detection_never_becomes_a_head_fallback(self) -> None:
        player = candidate(
            (100, 100, 200, 300),
            0.99,
            class_id=PLAYER_CLASS_ID,
        )

        self.assertIsNone(
            associate_head_to_player(
                [player],
                (100, 100, 200, 300),
                source_timestamp_ns=13,
            )
        )

    def test_head_outside_or_barely_overlapping_player_fails_closed(self) -> None:
        outside = candidate((220, 80, 260, 120), 0.99)
        partial = candidate((80, 80, 120, 120), 0.98)
        supporting_player = candidate(
            (95, 95, 205, 305),
            0.90,
            class_id=PLAYER_CLASS_ID,
        )

        self.assertIsNone(
            associate_head_to_player(
                [supporting_player, outside, partial],
                (100, 100, 200, 300),
                source_timestamp_ns=14,
            )
        )

    def test_implausibly_large_head_fails_closed(self) -> None:
        large = candidate((100, 100, 200, 260), 0.99)
        supporting_player = candidate(
            (95, 95, 205, 305),
            0.90,
            class_id=PLAYER_CLASS_ID,
        )

        self.assertIsNone(
            associate_head_to_player(
                [supporting_player, large],
                (100, 100, 200, 300),
                source_timestamp_ns=15,
            )
        )

    def test_torso_head_false_positive_fails_closed(self) -> None:
        torso = candidate((135, 200, 165, 230), 0.99)
        supporting_player = candidate(
            (95, 95, 205, 305),
            0.90,
            class_id=PLAYER_CLASS_ID,
        )

        self.assertIsNone(
            associate_head_to_player(
                [supporting_player, torso],
                (100, 100, 200, 300),
                source_timestamp_ns=16,
            )
        )

    def test_multiple_plausible_heads_without_instance_fail_closed(self) -> None:
        first = candidate((135, 105, 165, 135), 0.80, row_index=1)
        second = candidate((170, 105, 195, 135), 0.99, row_index=2)

        self.assertIsNone(
            associate_head_to_player(
                [first, second],
                (100, 100, 200, 300),
                source_timestamp_ns=17,
            )
        )

    def test_supporting_player_cannot_choose_between_two_global_heads(self) -> None:
        supporting_player = candidate(
            (95, 95, 170, 310),
            0.86,
            class_id=PLAYER_CLASS_ID,
            row_index=10,
        )
        first = candidate((120, 105, 145, 135), 0.80, row_index=11)
        second = candidate((180, 105, 205, 135), 0.99, row_index=12)

        self.assertIsNone(
            associate_head_to_player(
                [supporting_player, first, second],
                (100, 100, 220, 320),
                source_timestamp_ns=18,
            )
        )

    def test_half_overlapping_neighbor_cannot_be_sole_player_support(self) -> None:
        overlapping_neighbor = candidate(
            (40, 0, 140, 200),
            0.95,
            class_id=PLAYER_CLASS_ID,
            row_index=13,
        )
        neighbor_head = candidate((60, 10, 80, 30), 0.99, row_index=14)

        self.assertIsNone(
            associate_head_to_player(
                [overlapping_neighbor, neighbor_head],
                (0, 0, 100, 200),
                source_timestamp_ns=19,
            )
        )

    def test_multiple_matching_players_are_ambiguous_with_only_one_head(self) -> None:
        first_player = candidate(
            (95, 95, 175, 310),
            0.90,
            class_id=PLAYER_CLASS_ID,
            row_index=20,
        )
        second_player = candidate(
            (145, 95, 225, 310),
            0.91,
            class_id=PLAYER_CLASS_ID,
            row_index=21,
        )
        sole_head = candidate((120, 105, 150, 135), 0.80, row_index=22)

        self.assertIsNone(
            associate_head_to_player(
                [first_player, second_player, sole_head],
                (100, 100, 220, 320),
                source_timestamp_ns=20,
            )
        )

    def test_result_keeps_absolute_point_and_exact_source_timestamp(self) -> None:
        supporting_player = candidate(
            (95, 95, 205, 305),
            0.90,
            class_id=PLAYER_CLASS_ID,
        )
        result = associate_head_to_player(
            [supporting_player, candidate((140, 110, 160, 130), 0.90)],
            (100, 100, 200, 300),
            source_timestamp_ns=123_456,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.point, (150.0, 120.0))
        self.assertEqual(result.source_timestamp_ns, 123_456)
        self.assertEqual(result.supporting_player_index, 0)
        self.assertFalse(hasattr(result, "player_relative_point"))
        self.assertFalse(hasattr(result, "reproject"))

    def test_source_timestamp_must_be_non_negative_integer(self) -> None:
        head = candidate((140, 110, 160, 130), 0.90)
        with self.assertRaises(TypeError):
            associate_head_to_player(
                [head],
                (100, 100, 200, 300),
                source_timestamp_ns=True,
            )
        with self.assertRaises(ValueError):
            associate_head_to_player(
                [head],
                (100, 100, 200, 300),
                source_timestamp_ns=-1,
            )


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class DirectHeadLocalizerTests(unittest.TestCase):
    def test_one_crop_returns_only_a_direct_contained_head(self) -> None:
        seen: list[np.ndarray] = []

        def infer(tensor: np.ndarray) -> np.ndarray:
            seen.append(tensor)
            return model_output(
                ((80, 32, 240, 288), 0.92, 0.02),
                ((128, 48, 192, 112), 0.03, 0.88),
            )

        localizer = DirectHeadLocalizer(infer)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        result = localizer.localize(
            frame,
            (50, 20, 150, 180),
            source_timestamp_ns=20,
        )

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].shape, (1, 3, 320, 320))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(np.allclose(result.point, (100, 50)))
        self.assertEqual(result.source_timestamp_ns, 20)

    def test_off_frame_head_with_only_player_evidence_fails_closed(self) -> None:
        def infer(_tensor: np.ndarray) -> np.ndarray:
            return model_output(((160, 30, 310, 300), 0.95, 0.01))

        localizer = DirectHeadLocalizer(infer)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        self.assertIsNone(
            localizer.localize(
                frame,
                (120, 0, 199, 180),
                source_timestamp_ns=21,
            )
        )

    def test_malformed_inference_output_is_not_hidden_by_a_fallback(self) -> None:
        localizer = DirectHeadLocalizer(
            lambda _tensor: np.zeros((1, 5, 100), dtype=np.float32)
        )
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        with self.assertRaises(OutputDecodeError):
            localizer.localize(
                frame,
                (50, 20, 150, 180),
                source_timestamp_ns=22,
            )


if __name__ == "__main__":
    unittest.main()
