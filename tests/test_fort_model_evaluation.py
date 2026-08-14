from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.fort_dataset_contract import GROUPED_DATASET_YAML, build_dataset_contract
from scripts.evaluate_fort_model import (
    EvaluationConfigurationError,
    bucket_image_evidence,
    evaluate_checkpoint,
    match_bucket_counts,
    metric_summary,
    summarize_bucket_evidence,
    validate_bucket_evidence_coverage,
)


class _BoxMetrics:
    mp = 0.71
    mr = 0.62
    map50 = 0.73
    map = 0.39


class _Metrics:
    box = _BoxMetrics()
    fitness = 0.424
    speed = {"inference": 3.25, "preprocess": 0.75}


class _FakeModel:
    task = "detect"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def val(self, **kwargs: object) -> _Metrics:
        self.calls.append(dict(kwargs))
        return _Metrics()


class FortModelEvaluationTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path]:
        weights = root / "best.pt"
        weights.write_bytes(b"checkpoint")
        data = root / "dataset.yaml"
        data.write_text(
            "train: images/train\nval: images/valid\ntest: images/test\n"
            "names:\n  0: player\n",
            encoding="utf-8",
        )
        return weights, data

    def test_evaluates_requested_splits_and_writes_pinned_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights, original_data = self._inputs(root)
            data = root / "fort_cuh_grouped.yaml"
            original_data.rename(data)
            data.write_text(GROUPED_DATASET_YAML, encoding="utf-8")
            (root / "labels.txt").write_text("player\n", encoding="utf-8")
            (root / "manifest.json").write_text("{}\n", encoding="utf-8")
            model = _FakeModel()

            record = evaluate_checkpoint(
                weights=weights,
                data=data,
                output=root / "evaluation",
                splits=("val", "test"),
                imgsz=416,
                batch=8,
                device="cpu",
                workers=0,
                threads=2,
                plots=False,
                yolo_class=lambda _weights: model,
            )

            self.assertEqual([call["split"] for call in model.calls], ["val", "test"])
            self.assertEqual(record["metrics"]["test"]["map50_95"], 0.39)
            self.assertEqual(record["configuration"]["imgsz"], 416)
            self.assertFalse(record["configuration"]["exact_static_deployment_shape"])
            self.assertTrue(
                record["configuration"]["deployment_artifact_evaluation_required"]
            )
            self.assertIsNotNone(record["dataset_manifest_sha256"])
            saved = json.loads((root / "evaluation" / "metrics.json").read_text())
            self.assertEqual(saved, record)
            self.assertFalse(model.calls[0]["exist_ok"])

    def test_refuses_to_mix_results_in_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights, data = self._inputs(root)
            output = root / "evaluation"
            output.mkdir()
            with self.assertRaisesRegex(EvaluationConfigurationError, "already exists"):
                evaluate_checkpoint(
                    weights=weights,
                    data=data,
                    output=output,
                    yolo_class=lambda _weights: _FakeModel(),
                )

    def test_real_evaluator_refuses_missing_bucket_validator_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights, data = self._inputs(root)
            with (
                mock.patch(
                    "scripts.evaluate_fort_model._load_yolo_class",
                    return_value=lambda _weights: _FakeModel(),
                ),
                mock.patch(
                    "scripts.evaluate_fort_model._capturing_bucket_validator",
                    return_value=(object, {}),
                ),
                self.assertRaisesRegex(RuntimeError, "not captured"),
            ):
                evaluate_checkpoint(
                    weights=weights,
                    data=data,
                    output=root / "evaluation",
                    splits=("val",),
                )

    def test_grouped_dataset_contract_is_verified_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights, original_data = self._inputs(root)
            data = root / "fort_cuh_grouped.yaml"
            original_data.rename(data)
            data.write_text(GROUPED_DATASET_YAML, encoding="utf-8")
            (root / "labels.txt").write_text("player\n", encoding="utf-8")
            for split in ("train", "valid", "test"):
                image_dir = root / "images" / split
                label_dir = root / "labels" / split
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                (image_dir / f"{split}.jpg").write_bytes(f"{split} image".encode())
                (label_dir / f"{split}.txt").write_text(
                    "0 0.5 0.5 0.2 0.4\n", encoding="utf-8"
                )
            manifest = {
                "cross_split_source_groups": 0,
                "dataset_contract": build_dataset_contract(root),
                "splits": {
                    split: {"images": 1, "boxes": 1}
                    for split in ("train", "valid", "test")
                },
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (root / "images" / "valid" / "valid.jpg").write_bytes(b"tampered")
            model = _FakeModel()

            with self.assertRaisesRegex(
                EvaluationConfigurationError, "image hash mismatch"
            ):
                evaluate_checkpoint(
                    weights=weights,
                    data=data,
                    output=root / "evaluation",
                    yolo_class=lambda _weights: model,
                )
            self.assertEqual(model.calls, [])

    def test_grouped_evaluation_rejects_duplicate_or_alternate_yaml_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights, initial_data = self._inputs(root)
            data = root / "fort_cuh_grouped.yaml"
            initial_data.rename(data)
            canonical = (
                "# Generated by scripts/prepare_fort_cuh_grouped.py\n"
                "train: images/train\nval: images/valid\ntest: images/test\n"
                "names:\n  0: player\n"
            )
            data.write_text(canonical, encoding="utf-8")
            (root / "labels.txt").write_text("player\n", encoding="utf-8")
            for split in ("train", "valid", "test"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                (root / "images" / split / f"{split}.jpg").write_bytes(split.encode())
                (root / "labels" / split / f"{split}.txt").write_text(
                    "0 0.5 0.5 0.2 0.4\n", encoding="utf-8"
                )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "cross_split_source_groups": 0,
                        "dataset_contract": build_dataset_contract(root),
                        "splits": {
                            split: {"images": 1, "boxes": 1}
                            for split in ("train", "valid", "test")
                        },
                    }
                ),
                encoding="utf-8",
            )
            model = _FakeModel()
            for index, text in enumerate(
                (
                    canonical + "test: alternate/test\n",
                    canonical.replace("images/valid", "alternate/valid"),
                )
            ):
                data.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(
                    EvaluationConfigurationError, "exact generated"
                ):
                    evaluate_checkpoint(
                        weights=weights,
                        data=data,
                        output=root / f"evaluation-{index}",
                        yolo_class=lambda _weights: model,
                    )
            self.assertEqual(model.calls, [])

    def test_rejects_duplicate_or_invalid_split_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights, data = self._inputs(root)
            with self.assertRaisesRegex(EvaluationConfigurationError, "duplicates"):
                evaluate_checkpoint(
                    weights=weights,
                    data=data,
                    output=root / "one",
                    splits=("val", "val"),
                    yolo_class=lambda _weights: _FakeModel(),
                )
            with self.assertRaisesRegex(EvaluationConfigurationError, "val and/or test"):
                evaluate_checkpoint(
                    weights=weights,
                    data=data,
                    output=root / "two",
                    splits=(),
                    yolo_class=lambda _weights: _FakeModel(),
                )

    def test_metric_summary_rejects_nonfinite_values(self) -> None:
        metrics = _Metrics()
        metrics.box = type(
            "BadBox",
            (),
            {"mp": math.nan, "mr": 0.0, "map50": 0.0, "map": 0.0},
        )()
        with self.assertRaisesRegex(RuntimeError, "non-finite precision"):
            metric_summary(metrics)

    def test_bucket_matching_is_confidence_sorted_one_to_one(self) -> None:
        ground_truth = [
            ((0.0, 0.0, 20.0, 20.0), 24.0),
            ((100.0, 100.0, 140.0, 160.0), 60.0),
            ((200.0, 200.0, 260.0, 290.0), 90.0),
            ((300.0, 300.0, 400.0, 450.0), 150.0),
        ]
        predictions = [
            ((0.0, 0.0, 20.0, 20.0), 0.90),
            ((0.0, 0.0, 20.0, 20.0), 0.80),
            ((100.0, 100.0, 140.0, 160.0), 0.44),
            ((200.0, 200.0, 260.0, 290.0), 0.70),
            ((300.0, 300.0, 400.0, 450.0), 0.60),
        ]

        counts = match_bucket_counts(ground_truth, predictions, confidence=0.45)

        self.assertEqual(counts["ultra_far_le_32px"], {"targets": 1, "matched": 1})
        self.assertEqual(counts["far_33_to_64px"], {"targets": 1, "matched": 0})
        self.assertEqual(counts["medium_65_to_96px"], {"targets": 1, "matched": 1})
        self.assertEqual(counts["near_gt_96px"], {"targets": 1, "matched": 1})

    def test_bucket_evidence_reports_raw_misses_false_positives_pr_and_uncertainty(self) -> None:
        image = bucket_image_evidence(
            [
                ((0.0, 0.0, 20.0, 20.0), 24.0),
                ((100.0, 100.0, 140.0, 160.0), 60.0),
            ],
            [
                ((0.0, 0.0, 20.0, 20.0), 24.0, 0.90),
                ((200.0, 200.0, 220.0, 224.0), 24.0, 0.80),
                ((100.0, 100.0, 140.0, 160.0), 60.0, 0.20),
            ],
        )

        summary = summarize_bucket_evidence(
            [image], confidence_thresholds=(0.25,), bootstrap_samples=20
        )
        ultra = summary["operating_points"]["0.25"]["ultra_far_le_32px"]
        far = summary["operating_points"]["0.25"]["far_33_to_64px"]
        self.assertEqual(ultra["ground_truth_total"], 1)
        self.assertEqual(ultra["detected_true_positives"], 1)
        self.assertEqual(ultra["missed_false_negatives"], 0)
        self.assertEqual(ultra["predictions"], 2)
        self.assertEqual(ultra["false_positives"], 1)
        self.assertEqual(ultra["detected_over_total"], "1/1")
        self.assertEqual(far["detected_over_total"], "0/1")
        self.assertEqual(far["missed_false_negatives"], 1)
        self.assertIsNotNone(ultra["precision_wilson_95_ci"])
        pr = summary["pr_ap50"]["ultra_far_le_32px"]
        self.assertAlmostEqual(pr["ap50_101_point_interpolated"], 1.0)
        self.assertTrue(pr["precision_recall_curve"])
        self.assertEqual(
            pr["ap50_bootstrap_95_ci"]["samples_with_ground_truth"], 20
        )

    def test_bucket_coverage_rejects_partial_images_or_targets(self) -> None:
        image = {
            "targets": {
                "ultra_far_le_32px": 0,
                "far_33_to_64px": 1,
                "medium_65_to_96px": 0,
                "near_gt_96px": 0,
            },
            "events": {},
        }
        with self.assertRaisesRegex(RuntimeError, "image coverage mismatch"):
            validate_bucket_evidence_coverage(
                [image], expected_images=2, expected_boxes=1, split="val"
            )
        with self.assertRaisesRegex(RuntimeError, "target coverage mismatch"):
            validate_bucket_evidence_coverage(
                [image], expected_images=1, expected_boxes=2, split="val"
            )


if __name__ == "__main__":
    unittest.main()
