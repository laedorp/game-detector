#!/usr/bin/env python3
"""Evaluate an exported detector through the application's exact runtime path.

This evaluator is intentionally separate from PyTorch checkpoint screening. It
decodes every contracted image as BGR, calls :func:`utils.preprocess.preprocess_frame`,
runs the selected application detector at batch one, and calls that detector's
postprocessor to map predictions back to source pixels.  Only exact static ONNX
or OpenVINO deployment shapes are accepted.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import errno
from hashlib import sha256
from importlib import metadata
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import statistics
import sys
import tempfile
from time import perf_counter_ns
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_fort_model import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BUCKET_IOU_THRESHOLD,
    MINIMUM_PREDICTION_CONFIDENCE,
    REFERENCE_FRAME_HEIGHT,
    SIZE_BUCKETS,
    _bootstrap_ap50_interval,
    _dataset_evidence_scope,
    _pr_summary,
    _wilson_interval,
    bucket_image_evidence,
    summarize_bucket_evidence,
    validate_bucket_evidence_coverage,
)
from scripts.fort_dataset_contract import (  # noqa: E402
    DatasetContractError,
    verify_dataset_contract,
    verify_grouped_dataset_metadata,
)
from scripts.prepare_fort_cuh_grouped import (  # noqa: E402
    _canonical_source_group_key,
)
from detection.detail_pass import (  # noqa: E402
    DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT,
    DetailPassPlan,
    DetailPassStats,
    merge_cross_pass_detections,
    plan_detail_pass,
)
from utils.inference_size import (  # noqa: E402
    InferenceSize,
    format_inference_size,
    parse_inference_size,
    validate_yolo_inference_size,
)
from utils.preprocess import preprocess_frame  # noqa: E402


DEFAULT_CONFIDENCE_THRESHOLDS = (0.25, 0.45)
DEFAULT_NMS_IOU_THRESHOLD = 0.45
DEFAULT_WARMUP_ITERATIONS = 3
SUPPORTED_BACKENDS = ("onnxruntime", "openvino")
SUPPORTED_SPLITS = ("val", "test")


class RuntimeEvaluationError(ValueError):
    """Raised before or during evaluation when evidence would be unsafe."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _inference_size_argument(value: str) -> InferenceSize:
    try:
        return validate_yolo_inference_size(parse_inference_size(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _confidence_argument(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite value from zero to one")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=SUPPORTED_BACKENDS, required=True,
        help="Application detector backend used to load the exported artifact.",
    )
    parser.add_argument(
        "--inference-size", type=_inference_size_argument, required=True,
        metavar="N|HEIGHTxWIDTH",
        help="Exact static artifact input shape; both dimensions must be /32.",
    )
    parser.add_argument(
        "--split", choices=SUPPORTED_SPLITS, default="val",
        help="Evaluate validation by default. Test is never selected implicitly.",
    )
    parser.add_argument(
        "--acknowledge-development-test",
        action="store_true",
        help=(
            "Required with --split test. The current v9 test data was already used "
            "during development, so its result is audit-only and not a release gate."
        ),
    )
    parser.add_argument("--device", default="CPU")
    parser.add_argument(
        "--output-format",
        choices=("auto", "end2end", "traditional"),
        default="auto",
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=_confidence_argument,
        nargs="+",
        default=list(DEFAULT_CONFIDENCE_THRESHOLDS),
        metavar="CONFIDENCE",
    )
    parser.add_argument(
        "--nms-iou", type=_confidence_argument, default=DEFAULT_NMS_IOU_THRESHOLD,
        help="Application postprocessor NMS IoU for traditional exports.",
    )
    parser.add_argument(
        "--warmup", type=_non_negative_int, default=DEFAULT_WARMUP_ITERATIONS,
    )
    parser.add_argument(
        "--bootstrap-samples", type=_positive_int, default=BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--require-full-provider",
        action="store_true",
        help=(
            "ONNX accelerator qualification only: require the requested accelerator "
            "and disable both graph-node CPU fallback and runtime EP-failure retry."
        ),
    )
    parser.add_argument(
        "--detail-crop-size",
        type=_positive_int,
        metavar="SOURCE_WIDTH_PIXELS",
        help=(
            "Validation-only exact application detail pass: run the same static "
            "model on a centered model-aspect ROI up to this source-pixel width, "
            "map results to full source coordinates, and merge them once with "
            "the primary pass."
        ),
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _artifact_members(model: Path, backend: str) -> list[dict[str, Any]]:
    suffix = ".onnx" if backend == "onnxruntime" else ".xml"
    if not model.is_file() or model.suffix.casefold() != suffix:
        raise RuntimeEvaluationError(
            f"{backend} model must be a local {suffix} file: {model}"
        )
    paths = [model]
    if backend == "openvino":
        weights = model.with_suffix(".bin")
        if not weights.is_file():
            raise RuntimeEvaluationError(
                f"OpenVINO IR weights file is missing: {weights}"
            )
        paths.append(weights)
    for path in paths:
        if path.is_symlink():
            raise RuntimeEvaluationError(
                f"deployment artifact member must not be a symlink: {path}"
            )
    return [
        {
            "name": path.name,
            # Evidence is intended to survive publication.  The exact loaded
            # bytes are bound by size/SHA and by pre/post-run snapshots; an
            # absolute workstation path adds no integrity and can disclose a
            # developer account or checkout location.
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def _validated_dataset(
    data: Path,
) -> tuple[Path, Path, Mapping[str, Any], dict[str, Any]]:
    data = data.expanduser().resolve()
    if not data.is_file():
        raise RuntimeEvaluationError(f"dataset YAML not found: {data}")
    manifest_path = data.parent / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeEvaluationError(
            f"hardened grouped dataset manifest not found: {manifest_path}"
        )
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeEvaluationError(f"cannot read dataset manifest: {exc}") from exc
    if not isinstance(manifest_value, Mapping):
        raise RuntimeEvaluationError("dataset manifest must be a JSON object")
    if manifest_value.get("cross_split_source_groups") != 0:
        raise RuntimeEvaluationError(
            "dataset manifest does not prove zero cross-split source groups"
        )
    if manifest_value.get("runtime_class_labels") != ["player"]:
        raise RuntimeEvaluationError(
            "dataset manifest must define exactly one runtime class: player"
        )
    try:
        verify_grouped_dataset_metadata(data)
        contract = verify_dataset_contract(
            data.parent, manifest_value.get("dataset_contract")
        )
    except DatasetContractError as exc:
        raise RuntimeEvaluationError(
            f"grouped dataset exact-file contract failed: {exc}"
        ) from exc

    split_stats = manifest_value.get("splits")
    if not isinstance(split_stats, Mapping):
        raise RuntimeEvaluationError("dataset manifest has no split statistics")
    for split in ("train", "valid", "test"):
        stats = split_stats.get(split)
        contracted = contract["splits"][split]
        if not isinstance(stats, Mapping) or any(
            stats.get(key) != contracted[key] for key in ("images", "boxes")
        ):
            raise RuntimeEvaluationError(
                f"dataset manifest counts disagree with exact contract for {split}"
            )
    return data, manifest_path, manifest_value, contract


def _validate_configuration(
    *,
    backend: str,
    inference_size: InferenceSize,
    split: str,
    acknowledge_development_test: bool,
    confidence_thresholds: Sequence[float],
    nms_iou: float,
    warmup: int,
    bootstrap_samples: int,
    require_full_provider: bool,
    detail_crop_size: int | None,
) -> tuple[InferenceSize, tuple[float, ...]]:
    backend = str(backend).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise RuntimeEvaluationError(f"unsupported runtime backend: {backend!r}")
    if split not in SUPPORTED_SPLITS:
        raise RuntimeEvaluationError("split must be 'val' or 'test'")
    if split == "test" and not acknowledge_development_test:
        raise RuntimeEvaluationError(
            "--split test requires --acknowledge-development-test because this "
            "test split has already been consumed during development"
        )
    if acknowledge_development_test and split != "test":
        raise RuntimeEvaluationError(
            "--acknowledge-development-test is valid only with --split test"
        )
    if detail_crop_size is not None:
        if (
            isinstance(detail_crop_size, bool)
            or not isinstance(detail_crop_size, Integral)
            or detail_crop_size <= 0
        ):
            raise RuntimeEvaluationError(
                "detail crop size must be a positive integer or None"
            )
        if split != "val":
            raise RuntimeEvaluationError(
                "detail-pass artifact evaluation is validation-only; the v9 test "
                "split must not be used for detail-pass selection or tuning"
            )
    try:
        canonical_size = validate_yolo_inference_size(inference_size)
    except (TypeError, ValueError) as exc:
        raise RuntimeEvaluationError(f"invalid inference size: {exc}") from exc
    if not confidence_thresholds:
        raise RuntimeEvaluationError("at least one confidence threshold is required")
    if any(isinstance(value, bool) for value in confidence_thresholds):
        raise RuntimeEvaluationError("confidence thresholds must be numeric, not boolean")
    try:
        thresholds = tuple(float(value) for value in confidence_thresholds)
    except (TypeError, ValueError) as exc:
        raise RuntimeEvaluationError("confidence thresholds must be numeric") from exc
    if len(set(thresholds)) != len(thresholds):
        raise RuntimeEvaluationError("confidence thresholds must not contain duplicates")
    if tuple(sorted(thresholds)) != thresholds:
        raise RuntimeEvaluationError(
            "confidence thresholds must be strictly increasing for canonical evidence"
        )
    if any(
        not math.isfinite(value)
        or value < MINIMUM_PREDICTION_CONFIDENCE
        or value > 1.0
        for value in thresholds
    ):
        raise RuntimeEvaluationError(
            "confidence thresholds must be finite values from "
            f"{MINIMUM_PREDICTION_CONFIDENCE} to one"
        )
    if not math.isfinite(float(nms_iou)) or not 0.0 <= float(nms_iou) <= 1.0:
        raise RuntimeEvaluationError("NMS IoU must be a finite value from zero to one")
    if isinstance(warmup, bool) or not isinstance(warmup, Integral) or warmup < 0:
        raise RuntimeEvaluationError("warmup must be a non-negative integer")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, Integral)
        or bootstrap_samples <= 0
    ):
        raise RuntimeEvaluationError("bootstrap samples must be greater than zero")
    if require_full_provider and backend != "onnxruntime":
        raise RuntimeEvaluationError(
            "full-provider qualification is available only for ONNX Runtime"
        )
    return canonical_size, thresholds


def _load_bgr_image(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeEvaluationError("OpenCV is required to decode dataset images") from exc
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeEvaluationError(f"OpenCV could not decode contracted image: {path}")
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.flags.c_contiguous:
        frame = np.ascontiguousarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeEvaluationError(
            f"contracted image did not decode to BGR HxWx3: {path} ({frame.shape})"
        )
    return frame


def _ground_truth_boxes(
    label_path: Path, frame_shape: Sequence[int]
) -> list[tuple[tuple[float, float, float, float], float]]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise RuntimeEvaluationError(f"invalid decoded frame dimensions: {frame_shape}")
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeEvaluationError(f"cannot read contracted label {label_path}: {exc}") from exc
    targets: list[tuple[tuple[float, float, float, float], float]] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            # The exact contract should already have caught this. Keep the
            # inference loop independently fail-closed if that API changes.
            raise RuntimeEvaluationError(
                f"invalid one-class label at {label_path}:{line_number}"
            )
        try:
            center_x, center_y, box_width, box_height = map(float, fields[1:])
        except ValueError as exc:
            raise RuntimeEvaluationError(
                f"non-numeric label at {label_path}:{line_number}"
            ) from exc
        x1 = max(0.0, (center_x - box_width / 2.0) * width)
        y1 = max(0.0, (center_y - box_height / 2.0) * height)
        x2 = min(float(width), (center_x + box_width / 2.0) * width)
        y2 = min(float(height), (center_y + box_height / 2.0) * height)
        box = (x1, y1, x2, y2)
        if not all(math.isfinite(value) for value in box) or x2 <= x1 or y2 <= y1:
            raise RuntimeEvaluationError(
                f"label maps to an invalid source-space box at {label_path}:{line_number}"
            )
        targets.append((box, (y2 - y1) / height * REFERENCE_FRAME_HEIGHT))
    return targets


def _prediction_boxes(
    detections: Sequence[Any], frame_shape: Sequence[int]
) -> list[tuple[tuple[float, float, float, float], float, float]]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    predictions: list[tuple[tuple[float, float, float, float], float, float]] = []
    for index, detection in enumerate(detections):
        class_id = getattr(detection, "class_id", None)
        if isinstance(class_id, bool) or not isinstance(class_id, Integral) or int(class_id) != 0:
            raise RuntimeEvaluationError(
                f"runtime artifact emitted non-player class id at detection {index}: "
                f"{class_id!r}"
            )
        try:
            score = float(detection.confidence)
            box = tuple(float(value) for value in detection.xyxy)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeEvaluationError(
                f"runtime artifact emitted malformed detection {index}"
            ) from exc
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise RuntimeEvaluationError(
                f"runtime artifact emitted invalid box at detection {index}: {box!r}"
            )
        x1, y1, x2, y2 = box
        if (
            not math.isfinite(score)
            or not MINIMUM_PREDICTION_CONFIDENCE <= score <= 1.0
            or x1 < 0.0
            or y1 < 0.0
            or x2 > width
            or y2 > height
            or x2 <= x1
            or y2 <= y1
        ):
            raise RuntimeEvaluationError(
                f"runtime artifact emitted out-of-contract detection {index}"
            )
        predictions.append((box, (y2 - y1) / height * REFERENCE_FRAME_HEIGHT, score))
    return predictions


def _create_detector(
    *,
    backend: str,
    model: Path,
    labels: Path,
    device: str,
    inference_size: InferenceSize,
    confidence: float,
    iou: float,
    output_format: str,
    require_full_provider: bool,
) -> Any:
    common = dict(
        model_path=model,
        labels_path=labels,
        device=device,
        inference_size=inference_size,
        confidence=confidence,
        iou=iou,
        output_format=output_format,
    )
    if backend == "onnxruntime":
        from detection.onnx_yolo import OnnxRuntimeYoloDetector

        return OnnxRuntimeYoloDetector(
            **common, require_full_provider=require_full_provider
        )
    from detection.openvino_yolo import OpenVINOYoloDetector

    return OpenVINOYoloDetector(**common)


def _openvino_declared_shape(model_path: Path) -> list[int | None]:
    """Inspect the on-disk IR before accepting the detector's compiled shape."""

    from detection.openvino_yolo import _load_openvino, _partial_shape_values

    core_type, _version = _load_openvino()
    try:
        model = core_type().read_model(str(model_path))
        if len(model.inputs) != 1:
            raise RuntimeEvaluationError(
                f"OpenVINO artifact must declare one input, found {len(model.inputs)}"
            )
        return _partial_shape_values(model.input(0))
    except RuntimeEvaluationError:
        raise
    except Exception as exc:
        raise RuntimeEvaluationError(
            f"could not inspect OpenVINO artifact's declared input shape: {exc}"
        ) from exc


def _validated_static_shape(
    *,
    backend: str,
    model_path: Path,
    runtime_summary: Mapping[str, Any],
    inference_size: InferenceSize,
    declared_shape_inspector: Callable[[Path], Sequence[Any]] | None = None,
) -> list[int]:
    expected = [1, 3, inference_size[0], inference_size[1]]
    declared = runtime_summary.get("declared_input_shape")
    if declared is None and backend == "openvino":
        inspector = declared_shape_inspector or _openvino_declared_shape
        declared = inspector(model_path)
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise RuntimeEvaluationError(
            "runtime did not expose the artifact's declared input shape"
        )
    static: list[int] = []
    for value in declared:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise RuntimeEvaluationError(
                f"deployment artifact input must be fully static, got {list(declared)!r}"
            )
        static.append(int(value))
    if static != expected:
        raise RuntimeEvaluationError(
            f"deployment artifact declares input {static}, expected exact shape {expected}"
        )
    configured = runtime_summary.get("configured_input_shape", runtime_summary.get("input_shape"))
    if configured != expected:
        raise RuntimeEvaluationError(
            f"runtime configured input {configured!r}, expected exact shape {expected}"
        )
    return static


def _aggregate_images(images: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    for image in images:
        targets = image.get("targets")
        events = image.get("events")
        if not isinstance(targets, Mapping) or not isinstance(events, Mapping):
            raise RuntimeError("runtime evidence image is malformed")
        aggregated.append(
            {
                "targets": {"aggregate": sum(int(targets[name]) for name, *_ in SIZE_BUCKETS)},
                "events": {
                    "aggregate": [
                        event
                        for name, *_ in SIZE_BUCKETS
                        for event in events[name]
                    ]
                },
            }
        )
    return aggregated


def summarize_aggregate_evidence(
    images: Sequence[Mapping[str, Any]],
    *,
    confidence_thresholds: Sequence[float],
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Return overall metrics from the same one-to-one events as the buckets."""

    aggregate_images = _aggregate_images(images)
    targets = sum(int(image["targets"]["aggregate"]) for image in aggregate_images)
    operating_points: dict[str, Any] = {}
    for confidence in confidence_thresholds:
        events = [
            (float(score), bool(is_true_positive))
            for image in aggregate_images
            for score, is_true_positive in image["events"]["aggregate"]
            if float(score) >= confidence
        ]
        true_positives = sum(is_true_positive for _score, is_true_positive in events)
        predictions = len(events)
        false_positives = predictions - true_positives
        operating_points[str(confidence)] = {
            "ground_truth_total": targets,
            "detected_true_positives": true_positives,
            "missed_false_negatives": targets - true_positives,
            "predictions": predictions,
            "false_positives": false_positives,
            "detected_over_total": f"{true_positives}/{targets}",
            "precision": true_positives / predictions if predictions else None,
            "recall": true_positives / targets if targets else None,
            "precision_wilson_95_ci": _wilson_interval(true_positives, predictions),
            "recall_wilson_95_ci": _wilson_interval(true_positives, targets),
        }
    pr = _pr_summary(aggregate_images, "aggregate")
    pr["ap50_bootstrap_95_ci"] = _bootstrap_ap50_interval(
        aggregate_images, "aggregate", samples=bootstrap_samples
    )
    return {"operating_points": operating_points, "pr_ap50": pr}


def _metric_record(
    images: Sequence[Mapping[str, Any]],
    *,
    confidence_thresholds: Sequence[float],
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Build identical aggregate and size-bucket evidence for one pipeline."""

    aggregate = summarize_aggregate_evidence(
        images,
        confidence_thresholds=confidence_thresholds,
        bootstrap_samples=bootstrap_samples,
    )
    buckets = summarize_bucket_evidence(
        images,
        confidence_thresholds=confidence_thresholds,
        bootstrap_samples=bootstrap_samples,
    )
    return {
        "aggregate_detection": {
            "definition": (
                "One-to-one confidence-sorted source-space matching at IoU >=0.50 "
                "after exact application preprocessing and runtime postprocessing."
            ),
            **aggregate,
        },
        "size_bucket_detection": {
            "definition": (
                "A true positive belongs to its ground-truth projected-height "
                "bucket; an unmatched prediction belongs to its own projected-height "
                "bucket. Matching is one-to-one, confidence-sorted, source-space "
                "IoU >=0.50 after application runtime postprocessing."
            ),
            "reference_height_pixels": REFERENCE_FRAME_HEIGHT,
            "buckets": [name for name, *_ in SIZE_BUCKETS],
            "iou_threshold": BUCKET_IOU_THRESHOLD,
            "minimum_prediction_confidence": MINIMUM_PREDICTION_CONFIDENCE,
            "ap50_definition": (
                "101-point interpolated AP at IoU 0.50 over retained runtime "
                "prediction scores; uncertainty is a deterministic image-level "
                "nonparametric bootstrap percentile interval."
            ),
            **buckets,
            "sampling_caveat": (
                "Raw detected/total, misses, predictions, and false positives must "
                "accompany rates. Tiny or domain-poor buckets are not standalone "
                "release gates."
            ),
        },
    }


def paired_image_operating_point_evidence(
    images: Sequence[Mapping[str, Any]],
    expected_members: Sequence[Mapping[str, Any]],
    *,
    confidence_thresholds: Sequence[float],
    split_content_sha256: str,
    source_group_ids: Sequence[str],
) -> dict[str, Any]:
    """Retain compact, privacy-preserving evidence for paired model comparisons.

    Aggregate rates cannot show whether two candidates improve the same exact
    images.  This record binds each result to a stable digest of the contracted
    image/label member and retains only per-bucket target/TP/FP counts at the
    declared operating points.  It intentionally omits filenames, pixels,
    boxes, and raw prediction coordinates.
    """

    if len(images) != len(expected_members) or len(images) != len(source_group_ids):
        raise RuntimeError(
            "paired evidence image coverage differs from exact dataset membership"
        )
    bucket_names = [name for name, *_ in SIZE_BUCKETS]
    records: list[dict[str, Any]] = []
    member_ids: list[str] = []
    for image, member, source_group_id in zip(
        images, expected_members, source_group_ids, strict=True
    ):
        targets = image.get("targets")
        events = image.get("events")
        if not isinstance(targets, Mapping) or not isinstance(events, Mapping):
            raise RuntimeError("paired runtime evidence image is malformed")
        member_identity = {
            "image_sha256": member.get("image_sha256"),
            "label_sha256": member.get("label_sha256"),
        }
        if not all(
            isinstance(member_identity[key], str) and member_identity[key]
            for key in member_identity
        ):
            raise RuntimeError("paired dataset member identity is incomplete")
        member_id = _canonical_hash(member_identity)
        if (
            not isinstance(source_group_id, str)
            or len(source_group_id) != 64
            or any(character not in "0123456789abcdef" for character in source_group_id)
        ):
            raise RuntimeError("paired source-group identity is invalid")
        member_ids.append(member_id)
        target_counts: dict[str, int] = {}
        for bucket in bucket_names:
            value = targets.get(bucket)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise RuntimeError("paired target count is invalid")
            target_counts[bucket] = int(value)
        operating_points: dict[str, Any] = {}
        for confidence in confidence_thresholds:
            bucket_counts: dict[str, Any] = {}
            for bucket in bucket_names:
                bucket_events = events.get(bucket)
                if isinstance(bucket_events, (str, bytes)) or not isinstance(
                    bucket_events, Sequence
                ):
                    raise RuntimeError("paired prediction events are malformed")
                true_positives = 0
                false_positives = 0
                for event in bucket_events:
                    if (
                        isinstance(event, (str, bytes))
                        or not isinstance(event, Sequence)
                        or len(event) != 2
                    ):
                        raise RuntimeError("paired prediction event is malformed")
                    score = float(event[0])
                    is_true_positive = event[1]
                    if not math.isfinite(score) or not isinstance(is_true_positive, bool):
                        raise RuntimeError("paired prediction event is invalid")
                    if score < confidence:
                        continue
                    if is_true_positive:
                        true_positives += 1
                    else:
                        false_positives += 1
                if true_positives > target_counts[bucket]:
                    raise RuntimeError("paired true positives exceed target count")
                bucket_counts[bucket] = {
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                }
            operating_points[str(confidence)] = bucket_counts
        records.append(
            {
                "member_id": member_id,
                "source_group_id": source_group_id,
                "targets": target_counts,
                "operating_points": operating_points,
            }
        )
    return {
        "schema_version": 1,
        "privacy_scope": (
            "Stable exact-contract member digests and raw counts only; no filenames, "
            "pixels, boxes, or prediction coordinates."
        ),
        "bucket_order": bucket_names,
        "confidence_thresholds": [float(value) for value in confidence_thresholds],
        "split_content_sha256": split_content_sha256,
        "member_count": len(records),
        "member_sequence_sha256": _canonical_hash(member_ids),
        "source_group_count": len(set(source_group_ids)),
        "source_group_sequence_sha256": _canonical_hash(list(source_group_ids)),
        "source_group_scope": (
            "Privacy-safe digest of the grouped-dataset assignment heuristic, including "
            "documented visual unions. It is suitable for paired cluster resampling but "
            "does not prove real-world scene independence."
        ),
        "records": records,
    }


def _paired_source_group_ids(
    manifest: Mapping[str, Any],
    expected_members: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> list[str]:
    """Derive the exact grouped-split cluster for each contracted member.

    The grouped dataset predates paired runtime evidence, so its exact member
    contract does not duplicate source-group strings. The manifest pins the
    grouping algorithm and every perceptual union. Re-derive the canonical key,
    apply those unions, verify the recorded per-split group count, and expose
    only a one-way digest in runtime evidence.
    """

    collision_report = manifest.get("grouping_collision_report")
    split_stats = manifest.get("splits")
    if not isinstance(collision_report, Mapping) or not isinstance(split_stats, Mapping):
        raise RuntimeEvaluationError("dataset manifest has no source-group evidence")
    clusters = collision_report.get("visual_union_clusters")
    if isinstance(clusters, (str, bytes)) or not isinstance(clusters, Sequence):
        raise RuntimeEvaluationError("dataset manifest visual unions are invalid")
    visual_groups: dict[str, str] = {}
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            raise RuntimeEvaluationError("dataset manifest visual union is invalid")
        final_group = cluster.get("final_group")
        files = cluster.get("files")
        if (
            not isinstance(final_group, str)
            or not final_group
            or isinstance(files, (str, bytes))
            or not isinstance(files, Sequence)
            or not files
        ):
            raise RuntimeEvaluationError("dataset manifest visual union is invalid")
        for raw in files:
            if not isinstance(raw, str):
                raise RuntimeEvaluationError("dataset manifest visual-union path is invalid")
            path = PurePosixPath(raw)
            if path.is_absolute() or len(path.parts) != 2 or path.name != path.parts[1]:
                raise RuntimeEvaluationError("dataset manifest visual-union path is unsafe")
            key = path.name.casefold()
            previous = visual_groups.setdefault(key, final_group)
            if previous != final_group:
                raise RuntimeEvaluationError(
                    "dataset manifest assigns one member to multiple visual unions"
                )
    group_ids: list[str] = []
    for member in expected_members:
        image_name = member.get("image")
        if (
            not isinstance(image_name, str)
            or PurePosixPath(image_name).name != image_name
        ):
            raise RuntimeEvaluationError("dataset contract member name is invalid")
        group = visual_groups.get(
            image_name.casefold(), _canonical_source_group_key(image_name)
        )
        group_ids.append(
            _canonical_hash({"schema_version": 1, "source_group": group})
        )
    stats = split_stats.get(split)
    if (
        not isinstance(stats, Mapping)
        or isinstance(stats.get("source_groups"), bool)
        or not isinstance(stats.get("source_groups"), Integral)
        or int(stats["source_groups"]) != len(set(group_ids))
    ):
        raise RuntimeEvaluationError(
            "derived source-group count differs from the grouped dataset manifest"
        )
    return group_ids


def validate_runtime_coverage(
    *,
    evidence: Sequence[Mapping[str, Any]],
    processed_members: Sequence[str],
    expected_members: Sequence[Mapping[str, Any]],
    expected_boxes: int,
    split: str,
) -> None:
    """Reject omitted, duplicated, reordered, or partially parsed dataset members."""

    expected_names = [str(member["image"]) for member in expected_members]
    if list(processed_members) != expected_names:
        raise RuntimeError(
            f"runtime evidence member coverage mismatch for {split}: expected exact "
            f"ordered membership of {len(expected_names)} images"
        )
    validate_bucket_evidence_coverage(
        evidence,
        expected_images=len(expected_names),
        expected_boxes=expected_boxes,
        split=split,
    )


def _timing_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise RuntimeError("cannot summarize empty runtime timings")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise RuntimeError("runtime timings contain non-finite values")
    return {
        "mean": float(statistics.fmean(values)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return str(value)


def _public_runtime_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return runtime identity without workstation-specific filesystem paths."""

    result = _json_safe(value)
    if not isinstance(result, dict):  # Defensive; callers already require a mapping.
        raise RuntimeEvaluationError("runtime summary could not be normalized")
    model_path = result.get("model_path")
    if isinstance(model_path, str) and model_path:
        result["model_path"] = Path(model_path).name
    return result


def _pipeline_source_hashes(backend: str) -> dict[str, str]:
    paths = {
        "runtime_orchestration": PROJECT_ROOT / "main.py",
        "preprocess": PROJECT_ROOT / "utils" / "preprocess.py",
        "inference_size": PROJECT_ROOT / "utils" / "inference_size.py",
        "detail_pass": PROJECT_ROOT / "detection" / "detail_pass.py",
        "postprocess": PROJECT_ROOT / "detection" / "postprocess.py",
        "detection_types": PROJECT_ROOT / "detection" / "types.py",
        "detection_base": PROJECT_ROOT / "detection" / "base.py",
        "detector": PROJECT_ROOT / "detection" / (
            "onnx_yolo.py" if backend == "onnxruntime" else "openvino_yolo.py"
        ),
        "metric_helpers": PROJECT_ROOT / "scripts" / "evaluate_fort_model.py",
        "dataset_contract": PROJECT_ROOT / "scripts" / "fort_dataset_contract.py",
        "dataset_grouping": PROJECT_ROOT / "scripts" / "prepare_fort_cuh_grouped.py",
        "dataset_grouping_base": PROJECT_ROOT / "scripts" / "prepare_fort_cuh.py",
    }
    if backend == "onnxruntime":
        # The ONNX backend imports label/shape helpers from this module.
        paths["onnx_shared_openvino_helpers"] = (
            PROJECT_ROOT / "detection" / "openvino_yolo.py"
        )
    return {name: _sha256_file(path) for name, path in paths.items()}


def _validate_detail_preprocessing(
    prepared: Any,
    plan: DetailPassPlan,
) -> None:
    """Prove the crop transform consumed by the detector matches its plan."""

    transform = getattr(prepared, "transform", None)
    expected = {
        "crop_x": plan.crop_x,
        "crop_y": plan.crop_y,
        "source_width": plan.source_width,
        "source_height": plan.source_height,
        "model_width": plan.model_width,
        "model_height": plan.model_height,
    }
    if transform is None:
        raise RuntimeEvaluationError(
            "detail preprocessing transform does not match the centered crop plan"
        )
    for field, expected_value in expected.items():
        actual_value = getattr(transform, field, None)
        if (
            isinstance(actual_value, bool)
            or not isinstance(actual_value, Integral)
            or int(actual_value) != expected_value
        ):
            raise RuntimeEvaluationError(
                "detail preprocessing transform does not match the centered crop plan"
            )
    clamped = getattr(prepared, "crop_was_clamped", None)
    # The planner owns requested-to-applied clamping. Preprocessing receives
    # the already-applied HxW pair, so any second clamp would mean the executed
    # ROI no longer matches the recorded plan.
    if clamped is not False:
        raise RuntimeEvaluationError(
            "detail preprocessing unexpectedly clamped the applied crop plan"
        )
    scale = getattr(transform, "scale", None)
    if (
        isinstance(scale, bool)
        or not isinstance(scale, Real)
        or not math.isfinite(float(scale))
        or not math.isclose(
            float(scale), plan.detail_scale, rel_tol=1e-12, abs_tol=0.0
        )
    ):
        raise RuntimeEvaluationError(
            "detail preprocessing scale does not match the centered crop plan"
        )


def _source_hash_snapshot(backend: str) -> dict[str, Any]:
    """Pin the loaded evaluator and exact application-path source files."""

    evaluator_path = Path(__file__).resolve()
    return {
        "evaluator": {
            "path": evaluator_path.name,
            "sha256": _sha256_file(evaluator_path),
        },
        "pipeline": _pipeline_source_hashes(backend),
    }


def _dataset_metadata_hashes(data: Path, manifest_path: Path) -> dict[str, str]:
    """Hash metadata that is validated but is outside the image/label contract."""

    return {
        "yaml": _sha256_file(data),
        "manifest": _sha256_file(manifest_path),
        "runtime_labels": _sha256_file(data.parent / "labels.txt"),
    }


def _validate_runtime_binding(
    *,
    backend: str,
    requested_device: str,
    requested_output_format: str,
    require_full_provider: bool,
    runtime_summary: Mapping[str, Any],
) -> None:
    """Cross-check the detector's immutable startup report against the request."""

    if runtime_summary.get("output_format") != requested_output_format:
        raise RuntimeEvaluationError(
            "runtime output decoder differs from the requested output format"
        )
    if backend == "openvino":
        if str(runtime_summary.get("device", "")).strip().upper() != (
            requested_device.strip().upper()
        ):
            raise RuntimeEvaluationError(
                "OpenVINO runtime device differs from the requested device"
            )
        return

    requested_input = runtime_summary.get("requested_device_input")
    requested_provider = runtime_summary.get("requested_provider")
    active_providers = runtime_summary.get("active_providers")
    if (
        not isinstance(requested_input, str)
        or requested_input.strip().casefold()
        != requested_device.strip().casefold()
        or not isinstance(requested_provider, str)
        or isinstance(active_providers, (str, bytes))
        or not isinstance(active_providers, Sequence)
        or not active_providers
        or requested_provider not in active_providers
    ):
        raise RuntimeEvaluationError(
            "ONNX Runtime startup report does not prove the requested provider is active"
        )
    if bool(runtime_summary.get("require_full_provider")) != bool(
        require_full_provider
    ):
        raise RuntimeEvaluationError(
            "ONNX Runtime full-provider status differs from the evaluation request"
        )
    if require_full_provider:
        configured_session_options = runtime_summary.get(
            "configured_session_options"
        )
        if (
            not isinstance(configured_session_options, Mapping)
            or configured_session_options.get("disable_cpu_ep_fallback") is not True
        ):
            raise RuntimeEvaluationError(
                "full-provider qualification does not prove graph CPU fallback is disabled"
            )
        # ONNX Runtime can register CPUExecutionProvider implicitly even when
        # every graph node is required to remain on the requested accelerator.
        # Its Python wrapper has a separate EPFail retry path, so qualification
        # must prove that retry was disabled rather than incorrectly requiring
        # CPUExecutionProvider to disappear from get_providers().
        if runtime_summary.get("runtime_ep_fail_fallback_disabled") is not True:
            raise RuntimeEvaluationError(
                "full-provider qualification does not prove runtime EP-failure "
                "fallback is disabled"
            )


def _assert_inputs_unchanged(
    *,
    backend: str,
    model: Path,
    data: Path,
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    artifact_members: Sequence[Mapping[str, Any]],
    dataset_metadata_hashes: Mapping[str, str],
    source_hash_snapshot: Mapping[str, Any],
) -> None:
    """Revalidate every evidence input, including source loaded before inference."""

    final_data, final_manifest_path, final_manifest, final_contract = (
        _validated_dataset(data)
    )
    if final_data != data or final_manifest != manifest or final_contract != contract:
        raise RuntimeEvaluationError(
            "dataset metadata or exact-file contract changed during evaluation"
        )
    if _dataset_metadata_hashes(final_data, final_manifest_path) != dict(
        dataset_metadata_hashes
    ):
        raise RuntimeEvaluationError(
            "dataset YAML, manifest, or runtime labels changed during evaluation"
        )
    if _artifact_members(model, backend) != list(artifact_members):
        raise RuntimeEvaluationError("deployment artifact changed during evaluation")
    if _source_hash_snapshot(backend) != dict(source_hash_snapshot):
        raise RuntimeEvaluationError(
            "evaluator or application pipeline source changed during evaluation"
        )


def _write_metrics_file(path: Path, record: Mapping[str, Any]) -> None:
    """Create and durably flush the only member of a staged evidence directory."""

    with path.open("x", encoding="utf-8") as destination:
        json.dump(record, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing path."""

    if sys.platform.startswith("linux"):
        # Plain POSIX rename may replace an empty destination directory. Linux's
        # RENAME_NOREPLACE closes that race without exposing a partial result.
        import ctypes

        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeEvaluationError(
                "this Linux runtime lacks atomic no-replace directory publication"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(error, os.strerror(error), str(destination))
            raise OSError(error, os.strerror(error), str(destination))
        return

    if os.name == "nt":
        # MoveFileEx without MOVEFILE_REPLACE_EXISTING is the behavior exposed
        # by os.rename on Windows.
        os.rename(source, destination)
        return
    raise RuntimeEvaluationError(
        "atomic no-replace directory publication is supported only on Linux and Windows"
    )


def _publish_record(output: Path, record: Mapping[str, Any]) -> None:
    """Publish a complete private sibling directory with one atomic rename."""

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.runtime-eval-", dir=output.parent)
    )
    published = False
    try:
        _write_metrics_file(staging / "metrics.json", record)
        if os.path.lexists(output):
            raise RuntimeEvaluationError(
                "output appeared during evaluation; refusing to overwrite it: "
                f"{output}"
            )
        try:
            _rename_directory_noreplace(staging, output)
        except FileExistsError as exc:
            raise RuntimeEvaluationError(
                "output appeared during evaluation; refusing to overwrite it: "
                f"{output}"
            ) from exc
        published = True
    except RuntimeEvaluationError:
        raise
    except Exception as exc:
        raise RuntimeEvaluationError(
            f"could not atomically publish runtime evaluation: {type(exc).__name__}"
        ) from exc
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)


def evaluate_runtime_artifact(
    *,
    model: Path,
    data: Path,
    output: Path,
    backend: str,
    inference_size: InferenceSize,
    split: str = "val",
    acknowledge_development_test: bool = False,
    device: str = "CPU",
    output_format: str = "auto",
    confidence_thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    nms_iou: float = DEFAULT_NMS_IOU_THRESHOLD,
    warmup: int = DEFAULT_WARMUP_ITERATIONS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    require_full_provider: bool = False,
    detail_crop_size: int | None = None,
    detector_factory: Callable[..., Any] | None = None,
    image_loader: Callable[[Path], np.ndarray] | None = None,
    declared_shape_inspector: Callable[[Path], Sequence[Any]] | None = None,
    clock: Callable[[], int] = perf_counter_ns,
) -> dict[str, Any]:
    """Run one exact contracted split and write its evidence record."""

    backend = str(backend).strip().lower()
    inference_size, confidence_thresholds = _validate_configuration(
        backend=backend,
        inference_size=inference_size,
        split=split,
        acknowledge_development_test=acknowledge_development_test,
        confidence_thresholds=confidence_thresholds,
        nms_iou=nms_iou,
        warmup=warmup,
        bootstrap_samples=bootstrap_samples,
        require_full_provider=require_full_provider,
        detail_crop_size=detail_crop_size,
    )
    if output_format not in {"auto", "end2end", "traditional"}:
        raise RuntimeEvaluationError(
            "output format must be auto, end2end, or traditional"
        )
    if not str(device).strip():
        raise RuntimeEvaluationError("device cannot be empty")

    unresolved_output = output.expanduser()
    if os.path.lexists(unresolved_output):
        raise RuntimeEvaluationError(
            "output already exists; refusing to mix evaluation results: "
            f"{unresolved_output.resolve()}"
        )
    output = unresolved_output.resolve()
    unresolved_model = model.expanduser()
    if unresolved_model.is_symlink():
        raise RuntimeEvaluationError(
            f"deployment artifact entrypoint must not be a symlink: {unresolved_model}"
        )
    model = unresolved_model.resolve()
    artifact_members = _artifact_members(model, backend)
    data, manifest_path, manifest, contract = _validated_dataset(data)
    dataset_metadata_hashes = _dataset_metadata_hashes(data, manifest_path)
    source_hash_snapshot = _source_hash_snapshot(backend)

    manifest_split = "valid" if split == "val" else "test"
    split_contract = contract["splits"][manifest_split]
    expected_members = split_contract["members"]
    expected_boxes = int(split_contract["boxes"])
    source_group_ids = _paired_source_group_ids(
        manifest, expected_members, split=manifest_split
    )
    labels = data.parent / "labels.txt"

    factory = detector_factory or _create_detector
    detector = factory(
        backend=backend,
        model=model,
        labels=labels,
        device=str(device),
        inference_size=inference_size,
        confidence=MINIMUM_PREDICTION_CONFIDENCE,
        iou=float(nms_iou),
        output_format=output_format,
        require_full_provider=require_full_provider,
    )
    summary_value = getattr(detector, "runtime_summary", None)
    if not isinstance(summary_value, Mapping):
        raise RuntimeEvaluationError("application detector returned no runtime summary")
    runtime_summary = dict(summary_value)
    _validate_runtime_binding(
        backend=backend,
        requested_device=str(device),
        requested_output_format=output_format,
        require_full_provider=require_full_provider,
        runtime_summary=runtime_summary,
    )
    declared_shape = _validated_static_shape(
        backend=backend,
        model_path=model,
        runtime_summary=runtime_summary,
        inference_size=inference_size,
        declared_shape_inspector=declared_shape_inspector,
    )
    detector.warmup(warmup)

    load_image = image_loader or _load_bgr_image
    evidence: list[dict[str, Any]] = []
    primary_evidence: list[dict[str, Any]] = []
    processed_members: list[str] = []
    detail_stats = DetailPassStats(detail_crop_size)
    timings: dict[str, list[float]] = {
        "decode": [],
        "preprocess": [],
        "inference": [],
        "postprocess": [],
        "detail_preprocess": [],
        "detail_inference": [],
        "detail_postprocess": [],
        "runtime_pipeline": [],
    }
    observed_output_shape: list[int] | None = None
    observed_output_dtype: str | None = None

    for member in expected_members:
        image_name = str(member["image"])
        label_name = str(member["label"])
        image_path = data.parent / "images" / manifest_split / image_name
        label_path = data.parent / "labels" / manifest_split / label_name

        decode_started = clock()
        frame = load_image(image_path)
        decoded = clock()
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise RuntimeEvaluationError(
                f"image loader did not return a uint8 NumPy BGR frame: {image_path}"
            )
        targets = _ground_truth_boxes(label_path, frame.shape)
        preprocessing_started = clock()
        prepared = preprocess_frame(frame, inference_size=inference_size)
        preprocessed = clock()
        raw = np.asarray(detector.infer(prepared.tensor))
        inferred = clock()
        detections = detector.postprocess(
            raw, transform=prepared.transform, frame_shape=frame.shape
        )
        primary_postprocessed = clock()

        current_shape = list(raw.shape)
        current_dtype = str(raw.dtype)
        if observed_output_shape is None:
            observed_output_shape = current_shape
            observed_output_dtype = current_dtype
        elif current_shape != observed_output_shape or current_dtype != observed_output_dtype:
            raise RuntimeEvaluationError(
                "runtime artifact output shape or dtype changed between runtime calls"
            )
        primary_predictions = _prediction_boxes(detections, frame.shape)
        primary_evidence.append(bucket_image_evidence(targets, primary_predictions))

        detail_preprocess_ms = 0.0
        detail_inference_ms = 0.0
        detail_postprocess_ms = 0.0
        detections_ready = primary_postprocessed
        if detail_crop_size is not None:
            detail_plan = plan_detail_pass(
                frame.shape,
                detail_crop_size,
                inference_size,
            )
            detail_stats.record(detail_plan)
            if not detail_plan.redundant:
                detail_preprocessing_started = clock()
                detail_prepared = preprocess_frame(
                    frame,
                    inference_size=inference_size,
                    crop_size=(
                        detail_plan.applied_crop_height,
                        detail_plan.applied_crop_width,
                    ),
                )
                detail_preprocessed = clock()
                detail_raw = np.asarray(detector.infer(detail_prepared.tensor))
                detail_inferred = clock()
                detail_detections = detector.postprocess(
                    detail_raw,
                    transform=detail_prepared.transform,
                    frame_shape=frame.shape,
                )
                detections = merge_cross_pass_detections(
                    detections,
                    detail_detections,
                    source_height=int(frame.shape[0]),
                    unmatched_detail_max_reference_height=(
                        DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT
                    ),
                    stats=detail_stats,
                )
                detections_ready = clock()

                detail_shape = list(detail_raw.shape)
                detail_dtype = str(detail_raw.dtype)
                if (
                    detail_shape != observed_output_shape
                    or detail_dtype != observed_output_dtype
                ):
                    raise RuntimeEvaluationError(
                        "runtime artifact output shape or dtype changed between "
                        "runtime calls"
                    )
                divisor = 1_000_000.0
                detail_preprocess_ms = (
                    detail_preprocessed - detail_preprocessing_started
                ) / divisor
                detail_inference_ms = (
                    detail_inferred - detail_preprocessed
                ) / divisor
                # Match the live pipeline: detail postprocess includes the
                # deterministic source-space cross-pass merge.
                detail_postprocess_ms = (
                    detections_ready - detail_inferred
                ) / divisor
                # Keep audit-only validation outside the production-equivalent
                # timed interval while still refusing to publish bad geometry.
                _validate_detail_preprocessing(detail_prepared, detail_plan)

        predictions = _prediction_boxes(detections, frame.shape)
        evidence.append(bucket_image_evidence(targets, predictions))
        processed_members.append(image_name)

        divisor = 1_000_000.0
        timings["decode"].append((decoded - decode_started) / divisor)
        timings["preprocess"].append((preprocessed - preprocessing_started) / divisor)
        timings["inference"].append((inferred - preprocessed) / divisor)
        timings["postprocess"].append((primary_postprocessed - inferred) / divisor)
        timings["detail_preprocess"].append(detail_preprocess_ms)
        timings["detail_inference"].append(detail_inference_ms)
        timings["detail_postprocess"].append(detail_postprocess_ms)
        timings["runtime_pipeline"].append(
            (detections_ready - preprocessing_started) / divisor
        )

    validate_runtime_coverage(
        evidence=evidence,
        processed_members=processed_members,
        expected_members=expected_members,
        expected_boxes=expected_boxes,
        split=split,
    )
    validate_runtime_coverage(
        evidence=primary_evidence,
        processed_members=processed_members,
        expected_members=expected_members,
        expected_boxes=expected_boxes,
        split=f"{split} primary",
    )
    # Pin evidence to every input that remained unchanged for the entire run,
    # including the Python source that was imported before inference began.
    _assert_inputs_unchanged(
        backend=backend,
        model=model,
        data=data,
        manifest=manifest,
        contract=contract,
        artifact_members=artifact_members,
        dataset_metadata_hashes=dataset_metadata_hashes,
        source_hash_snapshot=source_hash_snapshot,
    )
    selected_metrics = _metric_record(
        evidence,
        confidence_thresholds=confidence_thresholds,
        bootstrap_samples=bootstrap_samples,
    )
    primary_metrics = (
        _metric_record(
            primary_evidence,
            confidence_thresholds=confidence_thresholds,
            bootstrap_samples=bootstrap_samples,
        )
        if detail_crop_size is not None
        else None
    )
    split_content_sha256 = split_contract.get("content_sha256")
    if not isinstance(split_content_sha256, str) or len(split_content_sha256) != 64:
        raise RuntimeEvaluationError(
            "exact dataset split has no valid content digest for paired evidence"
        )
    paired_evidence = paired_image_operating_point_evidence(
        evidence,
        expected_members,
        confidence_thresholds=confidence_thresholds,
        split_content_sha256=split_content_sha256,
        source_group_ids=source_group_ids,
    )
    primary_paired_evidence = (
        paired_image_operating_point_evidence(
            primary_evidence,
            expected_members,
            confidence_thresholds=confidence_thresholds,
            split_content_sha256=split_content_sha256,
            source_group_ids=source_group_ids,
        )
        if detail_crop_size is not None
        else None
    )

    artifact_record = {
        "backend": backend,
        "entrypoint": model.name,
        "entrypoint_sha256": artifact_members[0]["sha256"],
        "members": artifact_members,
        "content_sha256": _canonical_hash(
            [{"name": item["name"], "sha256": item["sha256"]} for item in artifact_members]
        ),
    }
    test_warning = (
        "This dataset's test split was already inspected during model, shape, detail-pass, "
        "and filtering development. This result is development/audit-only, is not an "
        "untouched holdout, and cannot qualify a release. Capture a new independent "
        "holdout for final qualification."
        if split == "test"
        else None
    )
    record = {
        "schema_version": 4,
        "evaluator": {
            **source_hash_snapshot["evaluator"],
            "pipeline_source_sha256": source_hash_snapshot["pipeline"],
        },
        "model_artifact": artifact_record,
        "dataset": {
            "yaml": data.name,
            "yaml_sha256": dataset_metadata_hashes["yaml"],
            "manifest": manifest_path.name,
            "manifest_sha256": dataset_metadata_hashes["manifest"],
            "runtime_labels_sha256": dataset_metadata_hashes["runtime_labels"],
            "content_sha256": contract["content_sha256"],
            "evidence_scope": _dataset_evidence_scope(data),
        },
        "configuration": {
            "split": split,
            "manifest_split": manifest_split,
            "selection_role": (
                "development_validation" if split == "val" else "development_audit_only"
            ),
            "test_consumption_warning": test_warning,
            "backend": backend,
            "device": str(device),
            "inference_size": format_inference_size(inference_size),
            "input_shape_nchw": [1, 3, *inference_size],
            "declared_static_input_shape_nchw": declared_shape,
            "batch_size": 1,
            "output_format": output_format,
            "runtime_nms_iou_threshold": float(nms_iou),
            "matching_iou_threshold": BUCKET_IOU_THRESHOLD,
            "minimum_prediction_confidence": MINIMUM_PREDICTION_CONFIDENCE,
            "reported_confidence_thresholds": list(confidence_thresholds),
            "warmup_iterations": warmup,
            "bootstrap_samples": bootstrap_samples,
            "require_full_provider": bool(require_full_provider),
            "evaluation_mode": "application_runtime_artifact",
            "exact_static_deployment_shape": True,
            "deployment_artifact_evaluation_required": False,
            "preprocess": "utils.preprocess.preprocess_frame, full frame primary",
            "postprocess": "selected application detector.postprocess in source coordinates",
            "detail_pass": {
                "enabled": detail_crop_size is not None,
                "requested_crop_size_source_pixels": detail_crop_size,
                "selection_split_policy": "validation_only",
                "test_split_evaluation_permitted": False,
                "pipeline": (
                    "same static model: full-frame primary plus centered "
                    "model-aspect source ROI; both detector outputs map to full source "
                    "coordinates; detection.detail_pass.merge_cross_pass_detections "
                    "is called exactly once for each non-redundant detail frame"
                    if detail_crop_size is not None
                    else "disabled; full-frame primary pass only"
                ),
                "stats": detail_stats.snapshot(),
            },
        },
        "runtime": {
            "summary": _public_runtime_summary(runtime_summary),
            "observed_raw_output_shape": observed_output_shape,
            "observed_raw_output_dtype": observed_output_dtype,
            "timing_ms_per_image": {
                name: _timing_summary(values) for name, values in timings.items()
            },
            "timing_scope": (
                "Synchronous batch-one local measurements. runtime_pipeline includes "
                "the full-frame primary and, when enabled and non-redundant, centered "
                "detail preprocess/inference/postprocess plus the one cross-pass merge; "
                "it excludes image decode. preprocess/inference/postprocess are primary-"
                "only, while detail_postprocess includes detail decoding and merge. "
                "Redundant or disabled detail components are recorded as zero."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "opencv_python": _package_version("opencv-python"),
            "onnxruntime": _package_version("onnxruntime"),
            "openvino": _package_version("openvino"),
        },
        "metrics": {
            split: {
                "images": len(evidence),
                "ground_truth_boxes": expected_boxes,
                "configured_pipeline": (
                    "full_frame_plus_center_detail_merged"
                    if detail_crop_size is not None
                    else "full_frame_primary"
                ),
                **selected_metrics,
                "paired_image_operating_points": paired_evidence,
                **(
                    {
                        "primary_full_frame_reference": {
                            "definition": (
                                "The unmerged full-frame result from the same model calls, "
                                "images, runtime, and evaluation run. It receives the same "
                                "aggregate and size-bucket scoring as the configured "
                                "detail pipeline."
                            ),
                            **primary_metrics,
                            "paired_image_operating_points": primary_paired_evidence,
                        }
                    }
                    if primary_metrics is not None
                    else {}
                ),
            }
        },
        "qualification": {
            "status": "development_evidence_only",
            "independent_holdout_required": True,
            "reason": (
                test_warning
                or "Validation may select candidates but cannot provide final independent "
                "release qualification."
            ),
        },
    }
    # Recheck after all metric/report computation, immediately before atomic
    # publication. A mutation during bootstrap/report construction cannot be
    # attached to otherwise valid-looking evidence.
    _assert_inputs_unchanged(
        backend=backend,
        model=model,
        data=data,
        manifest=manifest,
        contract=contract,
        artifact_members=artifact_members,
        dataset_metadata_hashes=dataset_metadata_hashes,
        source_hash_snapshot=source_hash_snapshot,
    )
    _publish_record(output, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = evaluate_runtime_artifact(
            model=args.model,
            data=args.data,
            output=args.output,
            backend=args.backend,
            inference_size=args.inference_size,
            split=args.split,
            acknowledge_development_test=args.acknowledge_development_test,
            device=args.device,
            output_format=args.output_format,
            confidence_thresholds=args.confidence_thresholds,
            nms_iou=args.nms_iou,
            warmup=args.warmup,
            bootstrap_samples=args.bootstrap_samples,
            require_full_provider=args.require_full_provider,
            detail_crop_size=args.detail_crop_size,
        )
    except RuntimeEvaluationError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
