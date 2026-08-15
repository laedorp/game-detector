from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from detection.types import Detection
import scripts.evaluate_fort_runtime_model as runtime_evaluator
from scripts.evaluate_fort_runtime_model import (
    RuntimeEvaluationError,
    _publish_record,
    build_parser,
    evaluate_runtime_artifact,
    summarize_aggregate_evidence,
    validate_runtime_coverage,
)
from scripts.fort_dataset_contract import (
    GROUPED_DATASET_YAML,
    build_dataset_contract,
)


class _FakeRuntimeDetector:
    def __init__(
        self,
        input_shape: list[object],
        detections: list[Detection] | None = None,
        *,
        varying_output: bool = False,
    ) -> None:
        self._runtime_summary = {
            "runtime": "synthetic ONNX Runtime",
            "device": "CPUExecutionProvider",
            "requested_device_input": "CPU",
            "requested_provider": "CPUExecutionProvider",
            "active_providers": ["CPUExecutionProvider"],
            "require_full_provider": False,
            "declared_input_shape": list(input_shape),
            "configured_input_shape": list(input_shape),
            "output_format": "auto",
        }
        self.detections = list(detections or [])
        self.varying_output = varying_output
        self.warmup_calls: list[int] = []
        self.inference_shapes: list[tuple[int, ...]] = []

    @property
    def runtime_summary(self) -> dict[str, object]:
        return dict(self._runtime_summary)

    def warmup(self, iterations: int) -> None:
        self.warmup_calls.append(iterations)

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        self.inference_shapes.append(tensor.shape)
        rows = len(self.inference_shapes) if self.varying_output else 1
        return np.zeros((1, rows, 6), dtype=np.float32)

    def postprocess(
        self,
        _raw: np.ndarray,
        transform: object | None = None,
        frame_shape: object | None = None,
    ) -> list[Detection]:
        if transform is None or frame_shape is None:
            raise AssertionError("source mapping arguments were omitted")
        return list(self.detections)


class FortRuntimeArtifactEvaluationTests(unittest.TestCase):
    def _dataset(self, root: Path, *, valid_images: int = 1) -> Path:
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            count = valid_images if split == "valid" else 1
            for index in range(count):
                stem = f"{split}-{index}"
                # Content is exact-contract input. Tests inject a deterministic
                # decoded BGR frame so no runtime/model dependency is needed.
                (root / "images" / split / f"{stem}.jpg").write_bytes(
                    f"{split} image {index}".encode()
                )
                (root / "labels" / split / f"{stem}.txt").write_text(
                    "0 0.5 0.5 0.2 0.05\n", encoding="utf-8"
                )
        data = root / "fort_cuh_grouped.yaml"
        data.write_text(GROUPED_DATASET_YAML, encoding="utf-8")
        (root / "labels.txt").write_text("player\n", encoding="utf-8")
        contract = build_dataset_contract(root)
        manifest = {
            "schema_version": 1,
            "cross_split_source_groups": 0,
            "runtime_class_labels": ["player"],
            "grouping_collision_report": {"visual_union_clusters": []},
            "dataset_contract": contract,
            "splits": {
                name: {
                    "images": contract["splits"][name]["images"],
                    "boxes": contract["splits"][name]["boxes"],
                    "source_groups": contract["splits"][name]["images"],
                }
                for name in ("train", "valid", "test")
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        return data

    @staticmethod
    def _image_loader(_path: Path) -> np.ndarray:
        return np.zeros((100, 200, 3), dtype=np.uint8)

    @staticmethod
    def _detections() -> list[Detection]:
        return [
            # Matches normalized GT (80, 47.5, 120, 52.5), projected height 54.
            Detection(0, "player", 0.90, (80.0, 47.5, 120.0, 52.5)),
            # Unmatched near-range false positive retained only at confidence .25.
            Detection(0, "player", 0.30, (10.0, 10.0, 40.0, 30.0)),
        ]

    def test_detail_cli_defaults_off_and_requires_a_positive_crop(self) -> None:
        required = [
            "--model", "player.onnx",
            "--data", "dataset.yaml",
            "--output", "evidence",
            "--backend", "onnxruntime",
            "--inference-size", "384x640",
        ]
        self.assertIsNone(build_parser().parse_args(required).detail_crop_size)
        self.assertEqual(
            build_parser().parse_args(
                [*required, "--detail-crop-size", "768"]
            ).detail_crop_size,
            768,
        )
        with self.assertRaises(SystemExit):
            build_parser().parse_args([*required, "--detail-crop-size", "0"])

    def test_confidence_thresholds_must_be_canonical_and_strictly_increasing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for thresholds, pattern in (
                ((0.45, 0.25), "strictly increasing"),
                ((0.25, 0.25), "must not contain duplicates"),
            ):
                with self.subTest(thresholds=thresholds), self.assertRaisesRegex(
                    RuntimeEvaluationError, pattern
                ):
                    evaluate_runtime_artifact(
                        model=root / "missing.onnx",
                        data=root / "missing.yaml",
                        output=root / "output",
                        backend="onnxruntime",
                        inference_size=(416, 416),
                        confidence_thresholds=thresholds,
                    )

    def test_exact_detail_pipeline_maps_merges_once_and_reports_primary_ab(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"static synthetic artifact")

            class DetailDetector(_FakeRuntimeDetector):
                def __init__(self) -> None:
                    super().__init__([1, 3, 384, 640])
                    self._runtime_summary["output_format"] = "end2end"
                    self.transforms: list[tuple[int, int, int, int]] = []
                    self.decoded_boxes: list[list[tuple[float, float, float, float]]] = []

                def infer(self, tensor: np.ndarray) -> np.ndarray:
                    self.inference_shapes.append(tensor.shape)
                    if len(self.inference_shapes) == 1:
                        # Full-frame letterbox: 3.2x scale, 32 px top pad.
                        return np.array(
                            [[
                                [256.0, 184.0, 384.0, 200.0, 0.60, 0.0],
                                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                            ]],
                            dtype=np.float32,
                        )
                    # Centered 80x48 model-aspect ROI: exact 8x scale, no pad.
                    return np.array(
                        [[
                            [160.0, 172.0, 480.0, 212.0, 0.95, 0.0],
                            [24.0, 24.0, 48.0, 48.0, 0.70, 0.0],
                            [24.0, 48.0, 72.0, 144.0, 0.99, 0.0],
                        ]],
                        dtype=np.float32,
                    )

                def postprocess(self, raw, transform=None, frame_shape=None):
                    from detection.postprocess import decode_yolo_output

                    if transform is None or frame_shape is None:
                        raise AssertionError("source mapping arguments were omitted")
                    self.transforms.append(
                        (
                            transform.crop_x,
                            transform.crop_y,
                            transform.source_width,
                            transform.source_height,
                        )
                    )
                    detections = decode_yolo_output(
                        raw,
                        transform=transform,
                        frame_shape=frame_shape,
                        labels=("player",),
                        confidence=0.001,
                        iou=0.45,
                        output_format="end2end",
                    )
                    self.decoded_boxes.append([item.xyxy for item in detections])
                    return detections

            detector = DetailDetector()
            output = root / "detail-evaluation"
            with mock.patch(
                "scripts.evaluate_fort_runtime_model.merge_cross_pass_detections",
                wraps=runtime_evaluator.merge_cross_pass_detections,
            ) as merge:
                record = evaluate_runtime_artifact(
                    model=model,
                    data=data,
                    output=output,
                    backend="onnxruntime",
                    inference_size=(384, 640),
                    detail_crop_size=80,
                    output_format="end2end",
                    detector_factory=lambda **_kwargs: detector,
                    image_loader=self._image_loader,
                    warmup=0,
                    bootstrap_samples=20,
                )

            self.assertEqual(merge.call_count, 1)
            self.assertEqual(
                detector.inference_shapes,
                [(1, 3, 384, 640), (1, 3, 384, 640)],
            )
            self.assertEqual(
                detector.transforms,
                [(0, 0, 200, 100), (60, 26, 200, 100)],
            )
            self.assertEqual(len(detector.decoded_boxes), 2)
            for actual, expected in zip(
                detector.decoded_boxes[0][0],
                (80.0, 47.5, 120.0, 52.5),
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(
                detector.decoded_boxes[1][1],
                (80.0, 47.5, 120.0, 52.5),
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected)
            detail = record["configuration"]["detail_pass"]
            self.assertTrue(detail["enabled"])
            self.assertEqual(detail["requested_crop_size_source_pixels"], 80)
            self.assertFalse(detail["test_split_evaluation_permitted"])
            self.assertEqual(detail["stats"]["frames_applied"], 1)
            self.assertEqual(detail["stats"]["cross_pass_matches"], 1)
            self.assertEqual(detail["stats"]["detail_replacements"], 1)
            self.assertEqual(detail["stats"]["unmatched_detail_accepted"], 1)
            self.assertEqual(detail["stats"]["unmatched_detail_rejected_large"], 1)
            self.assertEqual(
                (detail["stats"]["last_plan"]["crop_x"],
                 detail["stats"]["last_plan"]["crop_y"]),
                (60, 26),
            )
            self.assertEqual(
                (
                    detail["stats"]["last_plan"]["applied_crop_width"],
                    detail["stats"]["last_plan"]["applied_crop_height"],
                ),
                (80, 48),
            )

            metrics = record["metrics"]["val"]
            self.assertEqual(
                metrics["configured_pipeline"],
                "full_frame_plus_center_detail_merged",
            )
            merged = metrics["aggregate_detection"]["operating_points"]["0.25"]
            primary = metrics["primary_full_frame_reference"][
                "aggregate_detection"
            ]["operating_points"]["0.25"]
            self.assertEqual(merged["detected_over_total"], "1/1")
            self.assertEqual((merged["predictions"], merged["false_positives"]), (2, 1))
            self.assertEqual(
                (primary["predictions"], primary["false_positives"]),
                (1, 0),
            )
            self.assertIn(
                "ap50_bootstrap_95_ci",
                metrics["primary_full_frame_reference"]["size_bucket_detection"]
                ["pr_ap50"]["far_33_to_64px"],
            )
            paired = metrics["paired_image_operating_points"]
            primary_paired = metrics["primary_full_frame_reference"][
                "paired_image_operating_points"
            ]
            self.assertEqual(paired["member_count"], 1)
            self.assertEqual(len(paired["records"][0]["member_id"]), 64)
            self.assertEqual(paired["source_group_count"], 1)
            self.assertEqual(len(paired["records"][0]["source_group_id"]), 64)
            self.assertEqual(
                len(paired["source_group_sequence_sha256"]), 64
            )
            self.assertNotIn("valid-0", json.dumps(paired))
            self.assertEqual(
                paired["records"][0]["operating_points"]["0.25"]
                ["far_33_to_64px"],
                {"true_positives": 1, "false_positives": 1},
            )
            self.assertEqual(
                primary_paired["records"][0]["operating_points"]["0.25"]
                ["near_gt_96px"]["false_positives"],
                0,
            )
            self.assertEqual(record["schema_version"], 4)
            self.assertEqual(
                json.loads((output / "metrics.json").read_text(encoding="utf-8")),
                record,
            )

    def test_detail_pass_is_validation_only_before_any_input_or_model_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factory_calls: list[object] = []
            for invalid_crop in (0, -1, True, 1.5, "80"):
                with self.subTest(invalid_crop=invalid_crop), self.assertRaisesRegex(
                    RuntimeEvaluationError, "positive integer"
                ):
                    evaluate_runtime_artifact(
                        model=root / "missing.onnx",
                        data=root / "missing.yaml",
                        output=root / f"invalid-{invalid_crop}",
                        backend="onnxruntime",
                        inference_size=(384, 640),
                        detail_crop_size=invalid_crop,  # type: ignore[arg-type]
                        detector_factory=lambda **kwargs: factory_calls.append(kwargs),
                    )

            with self.assertRaisesRegex(RuntimeEvaluationError, "validation-only"):
                evaluate_runtime_artifact(
                    model=root / "missing.onnx",
                    data=root / "missing.yaml",
                    output=root / "test-refused",
                    backend="onnxruntime",
                    inference_size=(384, 640),
                    split="test",
                    acknowledge_development_test=True,
                    detail_crop_size=80,
                    detector_factory=lambda **kwargs: factory_calls.append(kwargs),
                )
            self.assertEqual(factory_calls, [])
            self.assertFalse((root / "test-refused").exists())

    def test_redundant_square_detail_pass_skips_second_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"model")
            detector = _FakeRuntimeDetector([1, 3, 416, 416])

            record = evaluate_runtime_artifact(
                model=model,
                data=data,
                output=root / "redundant",
                backend="onnxruntime",
                inference_size=(416, 416),
                detail_crop_size=200,
                detector_factory=lambda **_kwargs: detector,
                image_loader=lambda _path: np.zeros((100, 100, 3), dtype=np.uint8),
                warmup=0,
                bootstrap_samples=2,
            )

            self.assertEqual(detector.inference_shapes, [(1, 3, 416, 416)])
            stats = record["configuration"]["detail_pass"]["stats"]
            self.assertEqual(stats["frames_redundant"], 1)
            self.assertEqual(stats["frames_applied"], 0)
            timing = record["runtime"]["timing_ms_per_image"]
            self.assertEqual(timing["detail_preprocess"]["mean"], 0.0)
            self.assertEqual(timing["detail_inference"]["mean"], 0.0)
            self.assertEqual(timing["detail_postprocess"]["mean"], 0.0)

    def test_detail_transform_mismatch_fails_without_publishing_evidence(self) -> None:
        from utils.preprocess import FrameTransform, PreprocessedFrame

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"model")
            output = root / "bad-detail-transform"
            detector = _FakeRuntimeDetector([1, 3, 384, 640])
            real_preprocess = runtime_evaluator.preprocess_frame
            calls = 0

            def mismatching_preprocess(*args, **kwargs):
                nonlocal calls
                calls += 1
                prepared = real_preprocess(*args, **kwargs)
                if calls == 1:
                    return prepared
                transform = prepared.transform
                wrong = FrameTransform(
                    scale=transform.scale,
                    pad_left=transform.pad_left,
                    pad_top=transform.pad_top,
                    crop_x=transform.crop_x + 1,
                    crop_y=transform.crop_y,
                    source_width=transform.source_width,
                    source_height=transform.source_height,
                    model_width=transform.model_width,
                    model_height=transform.model_height,
                )
                return PreprocessedFrame(
                    prepared.tensor,
                    wrong,
                    prepared.crop_was_clamped,
                )

            with mock.patch(
                "scripts.evaluate_fort_runtime_model.preprocess_frame",
                side_effect=mismatching_preprocess,
            ):
                with self.assertRaisesRegex(
                    RuntimeEvaluationError, "does not match the centered crop plan"
                ):
                    evaluate_runtime_artifact(
                        model=model,
                        data=data,
                        output=output,
                        backend="onnxruntime",
                        inference_size=(384, 640),
                        detail_crop_size=80,
                        detector_factory=lambda **_kwargs: detector,
                        image_loader=self._image_loader,
                        warmup=0,
                        bootstrap_samples=2,
                    )
            self.assertEqual(calls, 2)
            self.assertFalse(output.exists())

    def test_exact_app_path_reports_aggregate_buckets_hashes_and_rectangular_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"static synthetic artifact")
            detector = _FakeRuntimeDetector([1, 3, 384, 640], self._detections())
            detector._runtime_summary["model_path"] = str(model)
            output = root / "evaluation"

            record = evaluate_runtime_artifact(
                model=model,
                data=data,
                output=output,
                backend="onnxruntime",
                inference_size=(384, 640),
                detector_factory=lambda **_kwargs: detector,
                image_loader=self._image_loader,
                warmup=2,
                bootstrap_samples=20,
            )

            self.assertEqual(detector.warmup_calls, [2])
            self.assertEqual(detector.inference_shapes, [(1, 3, 384, 640)])
            self.assertEqual(record["configuration"]["inference_size"], "384x640")
            self.assertEqual(
                record["configuration"]["declared_static_input_shape_nchw"],
                [1, 3, 384, 640],
            )
            self.assertTrue(record["configuration"]["exact_static_deployment_shape"])
            self.assertFalse(
                record["configuration"]["deployment_artifact_evaluation_required"]
            )
            self.assertEqual(
                record["runtime"]["summary"]["active_providers"],
                ["CPUExecutionProvider"],
            )
            self.assertEqual(record["runtime"]["observed_raw_output_shape"], [1, 1, 6])
            self.assertEqual(record["model_artifact"]["members"][0]["bytes"], 25)
            self.assertEqual(len(record["model_artifact"]["entrypoint_sha256"]), 64)
            self.assertEqual(record["model_artifact"]["entrypoint"], "player.onnx")
            self.assertEqual(
                record["model_artifact"]["members"][0]["path"],
                "player.onnx",
            )
            self.assertEqual(record["dataset"]["yaml"], data.name)
            self.assertEqual(record["dataset"]["manifest"], "manifest.json")
            self.assertEqual(
                record["evaluator"]["path"],
                "evaluate_fort_runtime_model.py",
            )
            self.assertEqual(record["runtime"]["summary"]["model_path"], "player.onnx")
            self.assertNotIn(str(root), json.dumps(record, sort_keys=True))
            self.assertEqual(len(record["dataset"]["content_sha256"]), 64)
            self.assertEqual(len(record["dataset"]["runtime_labels_sha256"]), 64)
            self.assertFalse(record["configuration"]["detail_pass"]["enabled"])
            self.assertIsNone(
                record["configuration"]["detail_pass"][
                    "requested_crop_size_source_pixels"
                ]
            )
            self.assertIn(
                "detail_pass", record["evaluator"]["pipeline_source_sha256"]
            )
            self.assertIn(
                "runtime_orchestration",
                record["evaluator"]["pipeline_source_sha256"],
            )

            metrics = record["metrics"]["val"]
            aggregate_025 = metrics["aggregate_detection"]["operating_points"]["0.25"]
            aggregate_045 = metrics["aggregate_detection"]["operating_points"]["0.45"]
            self.assertEqual(
                aggregate_025,
                {
                    "ground_truth_total": 1,
                    "detected_true_positives": 1,
                    "missed_false_negatives": 0,
                    "predictions": 2,
                    "false_positives": 1,
                    "detected_over_total": "1/1",
                    "precision": 0.5,
                    "recall": 1.0,
                    "precision_wilson_95_ci": aggregate_025["precision_wilson_95_ci"],
                    "recall_wilson_95_ci": aggregate_025["recall_wilson_95_ci"],
                },
            )
            self.assertEqual(aggregate_045["predictions"], 1)
            far = metrics["size_bucket_detection"]["operating_points"]["0.25"][
                "far_33_to_64px"
            ]
            near = metrics["size_bucket_detection"]["operating_points"]["0.25"][
                "near_gt_96px"
            ]
            self.assertEqual(far["detected_over_total"], "1/1")
            self.assertEqual(near["ground_truth_total"], 0)
            self.assertEqual(near["false_positives"], 1)
            self.assertEqual(
                metrics["aggregate_detection"]["pr_ap50"][
                    "ap50_bootstrap_95_ci"
                ]["samples_with_ground_truth"],
                20,
            )
            saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, record)

    def test_label_parsing_is_outside_preprocess_and_pipeline_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"model")
            detector = _FakeRuntimeDetector([1, 3, 416, 416])
            # decode start/end, preprocess start/end, inference end, postprocess end
            ticks = iter((0, 1_000_000, 11_000_000, 13_000_000, 18_000_000, 20_000_000))

            record = evaluate_runtime_artifact(
                model=model,
                data=data,
                output=root / "timed",
                backend="onnxruntime",
                inference_size=(416, 416),
                detector_factory=lambda **_kwargs: detector,
                image_loader=self._image_loader,
                warmup=0,
                bootstrap_samples=2,
                clock=lambda: next(ticks),
            )

            timings = record["runtime"]["timing_ms_per_image"]
            self.assertEqual(timings["decode"]["mean"], 1.0)
            self.assertEqual(timings["preprocess"]["mean"], 2.0)
            self.assertEqual(timings["inference"]["mean"], 5.0)
            self.assertEqual(timings["postprocess"]["mean"], 2.0)
            self.assertEqual(timings["runtime_pipeline"]["mean"], 9.0)

    def test_detail_timing_separates_primary_and_includes_merge_in_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"model")
            detector = _FakeRuntimeDetector([1, 3, 384, 640])
            # Decode 1 ms; primary pre/infer/post 2/5/2 ms; a 1 ms planning
            # interval; detail pre/infer/(decode+merge) 3/7/4 ms.
            ticks = iter(
                (
                    0,
                    1_000_000,
                    10_000_000,
                    12_000_000,
                    17_000_000,
                    19_000_000,
                    20_000_000,
                    23_000_000,
                    30_000_000,
                    34_000_000,
                )
            )

            record = evaluate_runtime_artifact(
                model=model,
                data=data,
                output=root / "detail-timed",
                backend="onnxruntime",
                inference_size=(384, 640),
                detail_crop_size=80,
                detector_factory=lambda **_kwargs: detector,
                image_loader=self._image_loader,
                warmup=0,
                bootstrap_samples=2,
                clock=lambda: next(ticks),
            )

            timings = record["runtime"]["timing_ms_per_image"]
            self.assertEqual(timings["decode"]["mean"], 1.0)
            self.assertEqual(timings["preprocess"]["mean"], 2.0)
            self.assertEqual(timings["inference"]["mean"], 5.0)
            self.assertEqual(timings["postprocess"]["mean"], 2.0)
            self.assertEqual(timings["detail_preprocess"]["mean"], 3.0)
            self.assertEqual(timings["detail_inference"]["mean"], 7.0)
            self.assertEqual(timings["detail_postprocess"]["mean"], 4.0)
            self.assertEqual(timings["runtime_pipeline"]["mean"], 24.0)
            self.assertIn(
                "detail_postprocess includes detail decoding and merge",
                record["runtime"]["timing_scope"],
            )

    def test_real_application_decoders_map_end2end_and_traditional_outputs(self) -> None:
        from detection.postprocess import decode_yolo_output

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"model")

            class DecoderDetector(_FakeRuntimeDetector):
                def __init__(self, output_format: str) -> None:
                    super().__init__([1, 3, 416, 416])
                    self.output_format = output_format
                    self._runtime_summary["output_format"] = output_format

                def infer(self, tensor: np.ndarray) -> np.ndarray:
                    self.inference_shapes.append(tensor.shape)
                    # The 100x200 source is letterboxed into 416 square at 2.08x,
                    # with 104 px top padding. Both layouts describe the exact GT.
                    if self.output_format == "end2end":
                        return np.array(
                            [[[166.4, 202.8, 249.6, 213.2, 0.9, 0.0]]],
                            dtype=np.float32,
                        )
                    return np.array(
                        [[[208.0], [208.0], [83.2], [10.4], [0.9]]],
                        dtype=np.float32,
                    )

                def postprocess(self, raw, transform=None, frame_shape=None):
                    return decode_yolo_output(
                        raw,
                        transform=transform,
                        frame_shape=frame_shape,
                        labels=("player",),
                        confidence=0.001,
                        iou=0.45,
                        output_format=self.output_format,
                    )

            for output_format in ("end2end", "traditional"):
                detector = DecoderDetector(output_format)
                record = evaluate_runtime_artifact(
                    model=model,
                    data=data,
                    output=root / output_format,
                    backend="onnxruntime",
                    inference_size=(416, 416),
                    output_format=output_format,
                    detector_factory=lambda **_kwargs: detector,
                    image_loader=self._image_loader,
                    warmup=0,
                    bootstrap_samples=2,
                )
                point = record["metrics"]["val"]["aggregate_detection"][
                    "operating_points"
                ]["0.25"]
                self.assertEqual(point["detected_over_total"], "1/1")
                self.assertEqual(point["false_positives"], 0)

    def test_square_and_height_width_inputs_both_reach_exact_preprocessor_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            for inference_size in ((416, 416), (384, 640)):
                shape = [1, 3, *inference_size]
                model = root / f"player-{inference_size[0]}-{inference_size[1]}.onnx"
                model.write_bytes(repr(shape).encode("ascii"))
                detector = _FakeRuntimeDetector(shape)
                evaluate_runtime_artifact(
                    model=model,
                    data=data,
                    output=root / f"out-{inference_size[0]}-{inference_size[1]}",
                    backend="onnxruntime",
                    inference_size=inference_size,
                    detector_factory=lambda **_kwargs: detector,
                    image_loader=self._image_loader,
                    warmup=0,
                    bootstrap_samples=2,
                )
                self.assertEqual(detector.inference_shapes, [tuple(shape)])

    def test_openvino_ir_hashes_xml_and_bin_and_checks_on_disk_static_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.xml"
            model.write_text("<synthetic-ir/>", encoding="utf-8")
            model.with_suffix(".bin").write_bytes(b"weights")
            detector = _FakeRuntimeDetector([1, 3, 384, 640])
            del detector._runtime_summary["declared_input_shape"]
            del detector._runtime_summary["configured_input_shape"]
            detector._runtime_summary["input_shape"] = [1, 3, 384, 640]
            detector._runtime_summary["device"] = "CPU"
            inspected: list[Path] = []

            record = evaluate_runtime_artifact(
                model=model,
                data=data,
                output=root / "openvino-evaluation",
                backend="openvino",
                inference_size=(384, 640),
                detector_factory=lambda **_kwargs: detector,
                declared_shape_inspector=lambda path: (
                    inspected.append(path) or [1, 3, 384, 640]
                ),
                image_loader=self._image_loader,
                warmup=0,
                bootstrap_samples=2,
            )

            self.assertEqual(inspected, [model.resolve()])
            self.assertEqual(
                [member["name"] for member in record["model_artifact"]["members"]],
                ["player.xml", "player.bin"],
            )
            self.assertEqual(record["configuration"]["backend"], "openvino")

    def test_test_split_requires_acknowledgement_and_is_stamped_audit_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"model")
            calls: list[object] = []

            with self.assertRaisesRegex(
                RuntimeEvaluationError, "acknowledge-development-test"
            ):
                evaluate_runtime_artifact(
                    model=model,
                    data=data,
                    output=root / "refused",
                    backend="onnxruntime",
                    inference_size=(416, 416),
                    split="test",
                    detector_factory=lambda **kwargs: calls.append(kwargs),
                    image_loader=self._image_loader,
                )
            self.assertEqual(calls, [])
            self.assertFalse((root / "refused").exists())

            detector = _FakeRuntimeDetector([1, 3, 416, 416], self._detections())
            record = evaluate_runtime_artifact(
                model=model,
                data=data,
                output=root / "audit",
                backend="onnxruntime",
                inference_size=(416, 416),
                split="test",
                acknowledge_development_test=True,
                detector_factory=lambda **_kwargs: detector,
                image_loader=self._image_loader,
                warmup=0,
                bootstrap_samples=2,
            )
            self.assertEqual(
                record["configuration"]["selection_role"], "development_audit_only"
            )
            self.assertIn(
                "already inspected", record["configuration"]["test_consumption_warning"]
            )
            self.assertTrue(record["qualification"]["independent_holdout_required"])

    def test_contract_tamper_fails_before_detector_creation_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"model")
            (root / "images" / "valid" / "valid-0.jpg").write_bytes(b"tampered")
            factory_calls: list[object] = []

            with self.assertRaisesRegex(RuntimeEvaluationError, "image hash mismatch"):
                evaluate_runtime_artifact(
                    model=model,
                    data=data,
                    output=root / "evaluation",
                    backend="onnxruntime",
                    inference_size=(416, 416),
                    detector_factory=lambda **kwargs: factory_calls.append(kwargs),
                    image_loader=self._image_loader,
                )
            self.assertEqual(factory_calls, [])
            self.assertFalse((root / "evaluation").exists())

    def test_provider_device_and_full_provider_reports_must_match_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"model")

            for name, mutate, options, pattern in (
                (
                    "wrong-device",
                    lambda detector: detector._runtime_summary.update(
                        requested_device_input="CUDA",
                        requested_provider="CUDAExecutionProvider",
                        active_providers=["CUDAExecutionProvider"],
                    ),
                    {"device": "DIRECTML"},
                    "requested provider is active",
                ),
                (
                    "inactive-provider",
                    lambda detector: detector._runtime_summary.update(
                        requested_device_input="CUDA",
                        requested_provider="CUDAExecutionProvider",
                        active_providers=["CPUExecutionProvider"],
                    ),
                    {"device": "CUDA"},
                    "requested provider is active",
                ),
                (
                    "full-provider-graph-fallback",
                    lambda detector: detector._runtime_summary.update(
                        requested_device_input="CUDA",
                        requested_provider="CUDAExecutionProvider",
                        active_providers=[
                            "CUDAExecutionProvider",
                            "CPUExecutionProvider",
                        ],
                        require_full_provider=True,
                        configured_session_options={
                            "disable_cpu_ep_fallback": False
                        },
                        runtime_ep_fail_fallback_disabled=True,
                    ),
                    {"device": "CUDA", "require_full_provider": True},
                    "graph CPU fallback is disabled",
                ),
                (
                    "full-provider-epfail-fallback",
                    lambda detector: detector._runtime_summary.update(
                        requested_device_input="CUDA",
                        requested_provider="CUDAExecutionProvider",
                        active_providers=[
                            "CUDAExecutionProvider",
                            "CPUExecutionProvider",
                        ],
                        require_full_provider=True,
                        configured_session_options={
                            "disable_cpu_ep_fallback": True
                        },
                        runtime_ep_fail_fallback_disabled=False,
                    ),
                    {"device": "CUDA", "require_full_provider": True},
                    "runtime EP-failure fallback is disabled",
                ),
            ):
                detector = _FakeRuntimeDetector([1, 3, 416, 416])
                mutate(detector)
                with self.assertRaisesRegex(RuntimeEvaluationError, pattern):
                    evaluate_runtime_artifact(
                        model=model,
                        data=data,
                        output=root / name,
                        backend="onnxruntime",
                        inference_size=(416, 416),
                        detector_factory=lambda **_kwargs: detector,
                        image_loader=self._image_loader,
                        warmup=0,
                        bootstrap_samples=2,
                        **options,
                    )
                self.assertFalse((root / name).exists())

            # CPUExecutionProvider is normally registered implicitly.  The
            # two independent fallback controls, not provider-list absence,
            # prove that a qualifying run cannot retry or assign graph nodes
            # to CPU.
            qualified = _FakeRuntimeDetector([1, 3, 416, 416])
            qualified._runtime_summary.update(
                requested_device_input="CUDA",
                requested_provider="CUDAExecutionProvider",
                active_providers=[
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                require_full_provider=True,
                configured_session_options={"disable_cpu_ep_fallback": True},
                runtime_ep_fail_fallback_disabled=True,
            )
            output = root / "qualified-cuda-with-implicit-cpu"
            evaluate_runtime_artifact(
                model=model,
                data=data,
                output=output,
                backend="onnxruntime",
                inference_size=(416, 416),
                device="CUDA",
                require_full_provider=True,
                detector_factory=lambda **_kwargs: qualified,
                image_loader=self._image_loader,
                warmup=0,
                bootstrap_samples=2,
            )
            self.assertTrue((output / "metrics.json").is_file())

            openvino = _FakeRuntimeDetector([1, 3, 416, 416])
            openvino._runtime_summary = {
                "runtime": "OpenVINO",
                "device": "GPU",
                "input_shape": [1, 3, 416, 416],
                "output_format": "auto",
            }
            with self.assertRaisesRegex(RuntimeEvaluationError, "device differs"):
                evaluate_runtime_artifact(
                    model=self._openvino_artifact(root),
                    data=data,
                    output=root / "wrong-openvino-device",
                    backend="openvino",
                    inference_size=(416, 416),
                    device="CPU",
                    detector_factory=lambda **_kwargs: openvino,
                    declared_shape_inspector=lambda _path: [1, 3, 416, 416],
                    image_loader=self._image_loader,
                    warmup=0,
                    bootstrap_samples=2,
                )

    @staticmethod
    def _openvino_artifact(root: Path) -> Path:
        model = root / "player.xml"
        model.write_text("<synthetic-ir/>", encoding="utf-8")
        model.with_suffix(".bin").write_bytes(b"weights")
        return model

    def test_existing_output_and_model_or_output_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"model")
            detector = _FakeRuntimeDetector([1, 3, 416, 416])

            existing = root / "existing"
            existing.mkdir()
            for output in (existing, root / "output-link"):
                if output.name == "output-link":
                    output.symlink_to(existing, target_is_directory=True)
                with self.assertRaisesRegex(RuntimeEvaluationError, "output already exists"):
                    evaluate_runtime_artifact(
                        model=model,
                        data=data,
                        output=output,
                        backend="onnxruntime",
                        inference_size=(416, 416),
                        detector_factory=lambda **_kwargs: detector,
                        image_loader=self._image_loader,
                    )

            model_link = root / "model-link.onnx"
            model_link.symlink_to(model)
            with self.assertRaisesRegex(RuntimeEvaluationError, "must not be a symlink"):
                evaluate_runtime_artifact(
                    model=model_link,
                    data=data,
                    output=root / "model-link-result",
                    backend="onnxruntime",
                    inference_size=(416, 416),
                    detector_factory=lambda **_kwargs: detector,
                    image_loader=self._image_loader,
                )

    def test_atomic_publish_cleans_staging_and_preserves_racing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "evaluation"

            def race(_source: Path, destination: Path) -> None:
                destination.mkdir()
                (destination / "owner.txt").write_text("other", encoding="utf-8")
                raise FileExistsError(str(destination))

            with mock.patch(
                "scripts.evaluate_fort_runtime_model._rename_directory_noreplace",
                side_effect=race,
            ):
                with self.assertRaisesRegex(RuntimeEvaluationError, "appeared"):
                    _publish_record(output, {"valid": True})
            self.assertEqual(
                (output / "owner.txt").read_text(encoding="utf-8"), "other"
            )
            self.assertEqual(
                list(root.glob(".evaluation.runtime-eval-*")), []
            )

            with mock.patch(
                "scripts.evaluate_fort_runtime_model._write_metrics_file",
                side_effect=OSError("synthetic disk failure"),
            ):
                with self.assertRaisesRegex(RuntimeEvaluationError, "could not atomically"):
                    _publish_record(root / "failed", {"valid": True})
            self.assertFalse((root / "failed").exists())
            self.assertEqual(list(root.glob(".failed.runtime-eval-*")), [])

    def test_artifact_and_pipeline_source_mutation_write_no_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root)
            model = root / "player.onnx"
            model.write_bytes(b"model")

            class MutatingDetector(_FakeRuntimeDetector):
                def __init__(self, mutation) -> None:
                    super().__init__([1, 3, 416, 416])
                    self.mutation = mutation

                def infer(self, tensor: np.ndarray) -> np.ndarray:
                    raw = super().infer(tensor)
                    self.mutation()
                    return raw

            artifact = MutatingDetector(lambda: model.write_bytes(b"changed"))
            with self.assertRaisesRegex(RuntimeEvaluationError, "artifact changed"):
                evaluate_runtime_artifact(
                    model=model,
                    data=data,
                    output=root / "artifact-mutated",
                    backend="onnxruntime",
                    inference_size=(416, 416),
                    detector_factory=lambda **_kwargs: artifact,
                    image_loader=self._image_loader,
                    warmup=0,
                    bootstrap_samples=2,
                )
            self.assertFalse((root / "artifact-mutated").exists())

            model.write_bytes(b"model")
            fake_source_snapshot = {"evaluator": {"path": "x", "sha256": "x"}, "pipeline": {}}
            source_detector = MutatingDetector(lambda: None)
            with mock.patch(
                "scripts.evaluate_fort_runtime_model._source_hash_snapshot",
                side_effect=[fake_source_snapshot, {"changed": True}],
            ):
                with self.assertRaisesRegex(RuntimeEvaluationError, "source changed"):
                    evaluate_runtime_artifact(
                        model=model,
                        data=data,
                        output=root / "source-mutated",
                        backend="onnxruntime",
                        inference_size=(416, 416),
                        detector_factory=lambda **_kwargs: source_detector,
                        image_loader=self._image_loader,
                        warmup=0,
                        bootstrap_samples=2,
                    )
            self.assertFalse((root / "source-mutated").exists())

    def test_dynamic_mismatched_or_changing_runtime_shapes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self._dataset(root, valid_images=2)
            model = root / "player.onnx"
            model.write_bytes(b"model")
            for name, declared, pattern in (
                ("dynamic", [1, 3, "height", "width"], "fully static"),
                ("mismatch", [1, 3, 416, 416], "expected exact shape"),
            ):
                detector = _FakeRuntimeDetector(declared)
                with self.assertRaisesRegex(RuntimeEvaluationError, pattern):
                    evaluate_runtime_artifact(
                        model=model,
                        data=data,
                        output=root / name,
                        backend="onnxruntime",
                        inference_size=(384, 640),
                        detector_factory=lambda **_kwargs: detector,
                        image_loader=self._image_loader,
                        warmup=0,
                        bootstrap_samples=2,
                    )
                self.assertEqual(detector.inference_shapes, [])
                self.assertFalse((root / name).exists())

            changing = _FakeRuntimeDetector(
                [1, 3, 384, 640], varying_output=True
            )
            with self.assertRaisesRegex(RuntimeEvaluationError, "changed between"):
                evaluate_runtime_artifact(
                    model=model,
                    data=data,
                    output=root / "changing",
                    backend="onnxruntime",
                    inference_size=(384, 640),
                    detector_factory=lambda **_kwargs: changing,
                    image_loader=self._image_loader,
                    warmup=0,
                    bootstrap_samples=2,
                )
            self.assertFalse((root / "changing").exists())

            changing_between_passes = _FakeRuntimeDetector(
                [1, 3, 384, 640], varying_output=True
            )
            with self.assertRaisesRegex(RuntimeEvaluationError, "changed between"):
                evaluate_runtime_artifact(
                    model=model,
                    data=self._dataset(root / "detail-dataset"),
                    output=root / "changing-between-passes",
                    backend="onnxruntime",
                    inference_size=(384, 640),
                    detail_crop_size=80,
                    detector_factory=lambda **_kwargs: changing_between_passes,
                    image_loader=self._image_loader,
                    warmup=0,
                    bootstrap_samples=2,
                )
            self.assertEqual(len(changing_between_passes.inference_shapes), 2)
            self.assertFalse((root / "changing-between-passes").exists())

    def test_aggregate_summary_and_coverage_are_raw_and_fail_closed(self) -> None:
        evidence = [
            {
                "targets": {
                    "ultra_far_le_32px": 1,
                    "far_33_to_64px": 1,
                    "medium_65_to_96px": 0,
                    "near_gt_96px": 0,
                },
                "events": {
                    "ultra_far_le_32px": [(0.9, True), (0.8, False)],
                    "far_33_to_64px": [(0.2, True)],
                    "medium_65_to_96px": [],
                    "near_gt_96px": [],
                },
            }
        ]
        summary = summarize_aggregate_evidence(
            evidence, confidence_thresholds=(0.25,), bootstrap_samples=5
        )
        point = summary["operating_points"]["0.25"]
        self.assertEqual(point["ground_truth_total"], 2)
        self.assertEqual(point["detected_true_positives"], 1)
        self.assertEqual(point["missed_false_negatives"], 1)
        self.assertEqual(point["predictions"], 2)
        self.assertEqual(point["false_positives"], 1)

        with self.assertRaisesRegex(RuntimeError, "member coverage mismatch"):
            validate_runtime_coverage(
                evidence=evidence,
                processed_members=["wrong.jpg"],
                expected_members=[{"image": "expected.jpg"}],
                expected_boxes=2,
                split="val",
            )


if __name__ == "__main__":
    unittest.main()
