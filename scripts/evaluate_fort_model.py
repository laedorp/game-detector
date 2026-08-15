#!/usr/bin/env python3
"""Evaluate a local FORT player checkpoint on explicitly selected data splits.

The output is a machine-readable comparison record plus the plots produced by
Ultralytics.  The command refuses to reuse an output directory so results from
different checkpoints or dataset contracts cannot be mixed accidentally.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
from typing import Any

try:
    from scripts.fort_dataset_contract import (
        DatasetContractError,
        verify_dataset_contract,
        verify_grouped_dataset_metadata,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from fort_dataset_contract import (
        DatasetContractError,
        verify_dataset_contract,
        verify_grouped_dataset_metadata,
    )


DEFAULT_IMAGE_SIZE = 416
DEFAULT_SPLITS = ("val", "test")
REFERENCE_FRAME_HEIGHT = 1080
BUCKET_CONFIDENCE_THRESHOLDS = (0.25, 0.45)
BUCKET_IOU_THRESHOLD = 0.50
MINIMUM_PREDICTION_CONFIDENCE = 0.001
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_SEED = 0
SIZE_BUCKETS = (
    ("ultra_far_le_32px", 0.0, 32.0),
    ("far_33_to_64px", 32.0, 64.0),
    ("medium_65_to_96px", 64.0, 96.0),
    ("near_gt_96px", 96.0, float("inf")),
)


class EvaluationConfigurationError(ValueError):
    """Raised before inference when an evaluation contract is unsafe."""


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val", "test"),
        default=list(DEFAULT_SPLITS),
        help="Dataset split(s) to evaluate (default: val test).",
    )
    parser.add_argument("--imgsz", type=_positive_int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch", type=_positive_int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=_non_negative_int, default=0)
    parser.add_argument("--threads", type=_positive_int, default=6)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic plots (machine-readable metrics are still written).",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _configure_cpu_environment(threads: int) -> None:
    value = str(threads)
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value


def _load_yolo_class(threads: int) -> Callable[[str], Any]:
    try:
        from ultralytics import YOLO
        import ultralytics.data.dataset as ultralytics_dataset
        import torch
    except ImportError as exc:
        raise EvaluationConfigurationError(
            "Ultralytics and PyTorch are required in the active environment."
        ) from exc

    def _ignore_dataset_cache(_path: Path) -> dict[str, Any]:
        raise FileNotFoundError(
            "Ultralytics label caches are disabled by the exact dataset contract"
        )

    def _do_not_write_dataset_cache(
        _prefix: str, _path: Path, value: dict[str, Any], version: str
    ) -> dict[str, Any]:
        value["version"] = version
        return value

    ultralytics_dataset.load_dataset_cache_file = _ignore_dataset_cache
    ultralytics_dataset.save_dataset_cache_file = _do_not_write_dataset_cache
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except RuntimeError:
        pass
    return YOLO


def _validated_paths(weights: Path, data: Path, output: Path) -> tuple[Path, Path, Path]:
    weights = weights.expanduser().resolve()
    data = data.expanduser().resolve()
    output = output.expanduser().resolve()
    if not weights.is_file() or weights.suffix.casefold() != ".pt":
        raise EvaluationConfigurationError(f"local .pt checkpoint not found: {weights}")
    if not data.is_file() or data.suffix.casefold() not in {".yaml", ".yml"}:
        raise EvaluationConfigurationError(f"dataset YAML not found: {data}")
    try:
        yaml_text = data.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EvaluationConfigurationError(f"cannot read dataset YAML: {exc}") from exc
    if "0: player" not in yaml_text:
        raise EvaluationConfigurationError("dataset YAML must define class 0 as player")
    manifest_file = data.parent / "manifest.json"
    if manifest_file.is_file():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationConfigurationError(
                f"cannot read dataset manifest: {exc}"
            ) from exc
        is_grouped = isinstance(manifest, dict) and (
            manifest.get("cross_split_source_groups") == 0
            or "dataset_contract" in manifest
        )
        if is_grouped:
            try:
                verify_grouped_dataset_metadata(data)
                verified_contract = verify_dataset_contract(
                    data.parent, manifest.get("dataset_contract")
                )
            except DatasetContractError as exc:
                raise EvaluationConfigurationError(
                    f"grouped dataset exact-file contract failed: {exc}"
                ) from exc
            split_stats = manifest.get("splits")
            if not isinstance(split_stats, Mapping):
                raise EvaluationConfigurationError(
                    "grouped dataset manifest has no split statistics"
                )
            for split in ("train", "valid", "test"):
                stats = split_stats.get(split)
                contract_split = verified_contract["splits"][split]
                if not isinstance(stats, Mapping) or (
                    stats.get("images") != contract_split["images"]
                    or stats.get("boxes") != contract_split["boxes"]
                ):
                    raise EvaluationConfigurationError(
                        f"grouped dataset manifest counts disagree for {split}"
                    )
    if output.exists() or output.is_symlink() or os.path.lexists(output):
        raise EvaluationConfigurationError(
            f"output already exists; refusing to mix evaluation results: {output}"
        )
    return weights, data, output


def _finite_float(value: Any, description: str) -> float:
    converted = float(value)
    if converted != converted or converted in {float("inf"), float("-inf")}:
        raise RuntimeError(f"Ultralytics returned non-finite {description}: {converted}")
    return converted


def metric_summary(metrics: Any) -> dict[str, Any]:
    """Convert an Ultralytics detection metrics object to stable JSON fields."""

    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("Ultralytics validation returned no detection box metrics")
    speed = getattr(metrics, "speed", {})
    if not isinstance(speed, Mapping):
        speed = {}
    return {
        "precision": _finite_float(getattr(box, "mp"), "precision"),
        "recall": _finite_float(getattr(box, "mr"), "recall"),
        "map50": _finite_float(getattr(box, "map50"), "mAP50"),
        "map50_95": _finite_float(getattr(box, "map"), "mAP50-95"),
        "fitness": _finite_float(getattr(metrics, "fitness"), "fitness"),
        "speed_ms_per_image": {
            str(name): _finite_float(value, f"speed {name}")
            for name, value in sorted(speed.items())
        },
    }


def _bucket_name(projected_height: float) -> str:
    for name, lower, upper in SIZE_BUCKETS:
        if lower < projected_height <= upper or (lower == 0.0 and projected_height <= upper):
            return name
    raise RuntimeError(f"cannot bucket projected height: {projected_height}")


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_bucket_counts(
    ground_truth: Sequence[tuple[Sequence[float], float]],
    predictions: Sequence[tuple[Sequence[float], float]],
    *,
    confidence: float,
    iou_threshold: float = BUCKET_IOU_THRESHOLD,
) -> dict[str, dict[str, int]]:
    """Greedily match predictions by confidence and count GT recall buckets."""

    counts = {name: {"targets": 0, "matched": 0} for name, _low, _high in SIZE_BUCKETS}
    for _box, projected_height in ground_truth:
        counts[_bucket_name(projected_height)]["targets"] += 1
    unmatched = set(range(len(ground_truth)))
    for predicted_box, score in sorted(predictions, key=lambda item: item[1], reverse=True):
        if score < confidence:
            continue
        matches = [
            (_box_iou(predicted_box, ground_truth[index][0]), index)
            for index in unmatched
        ]
        if not matches:
            continue
        iou, selected = max(matches)
        if iou < iou_threshold:
            continue
        unmatched.remove(selected)
        counts[_bucket_name(ground_truth[selected][1])]["matched"] += 1
    return counts


def bucket_image_evidence(
    ground_truth: Sequence[tuple[Sequence[float], float]],
    predictions: Sequence[tuple[Sequence[float], float, float]],
    *,
    iou_threshold: float = BUCKET_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Match one image once and retain raw events for fixed-threshold and PR metrics.

    A true positive belongs to its matched ground-truth size bucket. An unmatched
    prediction belongs to its own projected-height bucket. This makes each
    bucket's precision denominator explicit instead of silently omitting false
    positives that have no matched target.
    """

    targets = {name: 0 for name, _low, _high in SIZE_BUCKETS}
    events: dict[str, list[tuple[float, bool]]] = {
        name: [] for name, _low, _high in SIZE_BUCKETS
    }
    for _box, projected_height in ground_truth:
        targets[_bucket_name(projected_height)] += 1
    unmatched = set(range(len(ground_truth)))
    for predicted_box, predicted_height, score in sorted(
        predictions, key=lambda item: item[2], reverse=True
    ):
        matches = [
            (_box_iou(predicted_box, ground_truth[index][0]), index)
            for index in unmatched
        ]
        iou, selected = max(matches, default=(0.0, -1))
        if selected >= 0 and iou >= iou_threshold:
            unmatched.remove(selected)
            bucket = _bucket_name(ground_truth[selected][1])
            events[bucket].append((float(score), True))
        else:
            bucket = _bucket_name(predicted_height)
            events[bucket].append((float(score), False))
    return {"targets": targets, "events": events}


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _pr_summary(
    images: Sequence[Mapping[str, Any]],
    bucket: str,
    image_weights: Sequence[int] | None = None,
    *,
    include_curve: bool = True,
) -> dict[str, Any]:
    weights = image_weights or [1] * len(images)
    targets = sum(
        int(image["targets"][bucket]) * int(weight)
        for image, weight in zip(images, weights, strict=True)
    )
    weighted_events: list[tuple[float, bool, int]] = []
    for image, weight in zip(images, weights, strict=True):
        if weight <= 0:
            continue
        weighted_events.extend(
            (float(score), bool(is_true_positive), int(weight))
            for score, is_true_positive in image["events"][bucket]
        )
    weighted_events.sort(key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    curve: list[dict[str, Any]] = []
    index = 0
    while index < len(weighted_events):
        confidence = weighted_events[index][0]
        while index < len(weighted_events) and weighted_events[index][0] == confidence:
            _score, is_true_positive, weight = weighted_events[index]
            if is_true_positive:
                true_positives += weight
            else:
                false_positives += weight
            index += 1
        predictions = true_positives + false_positives
        curve.append(
            {
                "confidence": confidence,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": max(0, targets - true_positives),
                "precision": true_positives / predictions if predictions else None,
                "recall": true_positives / targets if targets else None,
            }
        )
    if targets:
        recalls = [float(point["recall"]) for point in curve]
        precision_envelope = [float(point["precision"]) for point in curve]
        for position in range(len(precision_envelope) - 2, -1, -1):
            precision_envelope[position] = max(
                precision_envelope[position], precision_envelope[position + 1]
            )
        position = 0
        sampled_precision = 0.0
        for level in range(101):
            threshold = level / 100
            while position < len(recalls) and recalls[position] < threshold:
                position += 1
            if position < len(recalls):
                sampled_precision += precision_envelope[position]
        ap50 = sampled_precision / 101.0
    else:
        ap50 = None
    return {
        "ground_truth_total": targets,
        "prediction_events": true_positives + false_positives,
        "true_positives_at_minimum_confidence": true_positives,
        "false_positives_at_minimum_confidence": false_positives,
        "false_negatives_at_minimum_confidence": max(0, targets - true_positives),
        "ap50_101_point_interpolated": ap50,
        **({"precision_recall_curve": curve} if include_curve else {}),
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_ap50_interval(
    images: Sequence[Mapping[str, Any]],
    bucket: str,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if not images or samples <= 0:
        return {"samples_requested": samples, "samples_with_ground_truth": 0, "ci95": None}
    bucket_seed = int.from_bytes(
        sha256(f"{seed}:{bucket}".encode("utf-8")).digest()[:8], "big"
    )
    generator = random.Random(bucket_seed)
    estimates: list[float] = []
    for _ in range(samples):
        weights = [0] * len(images)
        for _draw in range(len(images)):
            weights[generator.randrange(len(images))] += 1
        estimate = _pr_summary(
            images, bucket, weights, include_curve=False
        )["ap50_101_point_interpolated"]
        if estimate is not None:
            estimates.append(float(estimate))
    return {
        "method": "deterministic image-level nonparametric bootstrap percentile interval",
        "seed": seed,
        "samples_requested": samples,
        "samples_with_ground_truth": len(estimates),
        "ci95": [
            _percentile(estimates, 0.025),
            _percentile(estimates, 0.975),
        ] if estimates else None,
    }


def summarize_bucket_evidence(
    images: Sequence[Mapping[str, Any]],
    *,
    confidence_thresholds: Sequence[float] = BUCKET_CONFIDENCE_THRESHOLDS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Build raw operating-point counts and target-size AP50/PR evidence."""

    operating_points: dict[str, Any] = {}
    for confidence in confidence_thresholds:
        buckets: dict[str, Any] = {}
        for bucket, _low, _high in SIZE_BUCKETS:
            targets = sum(int(image["targets"][bucket]) for image in images)
            events = [
                (float(score), bool(is_true_positive))
                for image in images
                for score, is_true_positive in image["events"][bucket]
                if float(score) >= confidence
            ]
            true_positives = sum(is_true_positive for _score, is_true_positive in events)
            predictions = len(events)
            false_positives = predictions - true_positives
            false_negatives = targets - true_positives
            buckets[bucket] = {
                "ground_truth_total": targets,
                "detected_true_positives": true_positives,
                "missed_false_negatives": false_negatives,
                "predictions": predictions,
                "false_positives": false_positives,
                "detected_over_total": f"{true_positives}/{targets}",
                "precision": true_positives / predictions if predictions else None,
                "recall": true_positives / targets if targets else None,
                "precision_wilson_95_ci": _wilson_interval(true_positives, predictions),
                "recall_wilson_95_ci": _wilson_interval(true_positives, targets),
            }
        operating_points[str(confidence)] = buckets
    pr_ap50: dict[str, Any] = {}
    for bucket, _low, _high in SIZE_BUCKETS:
        summary = _pr_summary(images, bucket)
        summary["ap50_bootstrap_95_ci"] = _bootstrap_ap50_interval(
            images, bucket, samples=bootstrap_samples
        )
        pr_ap50[bucket] = summary
    return {"operating_points": operating_points, "pr_ap50": pr_ap50}


def validate_bucket_evidence_coverage(
    images: Sequence[Mapping[str, Any]],
    *,
    expected_images: int,
    expected_boxes: int,
    split: str,
) -> None:
    """Fail closed if custom-validator evidence is partial or duplicated."""

    if len(images) != expected_images:
        raise RuntimeError(
            f"bucket evidence image coverage mismatch for {split}: "
            f"expected {expected_images}, got {len(images)}"
        )
    targets = sum(
        int(count)
        for image in images
        for count in image.get("targets", {}).values()
    )
    if targets != expected_boxes:
        raise RuntimeError(
            f"bucket evidence target coverage mismatch for {split}: "
            f"expected {expected_boxes}, got {targets}"
        )


def _bucket_validator_class() -> type[Any]:
    """Create an Ultralytics validator that records fixed-threshold bucket recall."""

    try:
        from ultralytics.models.yolo.detect.val import DetectionValidator
    except ImportError as exc:
        raise EvaluationConfigurationError("Ultralytics detection validator unavailable") from exc

    class BucketDetectionValidator(DetectionValidator):
        bucket_images: list[dict[str, Any]]

        def init_metrics(self, model: Any) -> None:
            super().init_metrics(model)
            self.bucket_images = []

        def update_metrics(self, preds: list[dict[str, Any]], batch: dict[str, Any]) -> None:
            from ultralytics.utils import ops

            for image_index, prediction in enumerate(preds):
                prepared_batch = self._prepare_batch(image_index, batch)
                prepared_prediction = self._prepare_pred(prediction)
                original_height, _original_width = prepared_batch["ori_shape"]
                original_boxes = ops.scale_boxes(
                    prepared_batch["imgsz"],
                    prepared_batch["bboxes"].clone(),
                    prepared_batch["ori_shape"],
                    ratio_pad=prepared_batch["ratio_pad"],
                )
                targets = []
                for input_box, original_box in zip(
                    prepared_batch["bboxes"].detach().cpu().tolist(),
                    original_boxes.detach().cpu().tolist(),
                    strict=True,
                ):
                    normalized_height = (
                        float(original_box[3]) - float(original_box[1])
                    ) / float(original_height)
                    targets.append(
                        (input_box, normalized_height * REFERENCE_FRAME_HEIGHT)
                    )
                predicted_boxes = prepared_prediction["bboxes"].detach().cpu().tolist()
                original_predicted_boxes = ops.scale_boxes(
                    prepared_batch["imgsz"],
                    prepared_prediction["bboxes"].clone(),
                    prepared_batch["ori_shape"],
                    ratio_pad=prepared_batch["ratio_pad"],
                ).detach().cpu().tolist()
                predicted_scores = prepared_prediction["conf"].detach().cpu().tolist()
                predictions = [
                    (
                        input_box,
                        (
                            float(original_box[3]) - float(original_box[1])
                        ) / float(original_height) * REFERENCE_FRAME_HEIGHT,
                        float(score),
                    )
                    for input_box, original_box, score in zip(
                        predicted_boxes,
                        original_predicted_boxes,
                        predicted_scores,
                        strict=True,
                    )
                ]
                self.bucket_images.append(bucket_image_evidence(targets, predictions))
            super().update_metrics(preds, batch)

    return BucketDetectionValidator


def _capturing_bucket_validator() -> tuple[type[Any], dict[str, Any]]:
    """Return a validator class plus a holder for the Model.val-created instance."""

    base = _bucket_validator_class()
    holder: dict[str, Any] = {}

    class CapturingBucketValidator(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            holder["instance"] = self

    return CapturingBucketValidator, holder


def _dataset_evidence_scope(data: Path) -> dict[str, Any] | None:
    manifest_file = data.parent / "manifest.json"
    if not manifest_file.is_file():
        return None
    value = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        return None
    assignment_balance = value.get("assignment_balance")
    strata = (
        assignment_balance.get("strata")
        if isinstance(assignment_balance, Mapping)
        else None
    )
    if not isinstance(strata, Mapping):
        return None
    by_split: dict[str, dict[str, int]] = {"valid": {}, "test": {}}
    for stratum, stats in sorted(strata.items()):
        if not isinstance(stratum, str) or not isinstance(stats, Mapping):
            continue
        splits = stats.get("splits")
        if not isinstance(splits, Mapping):
            continue
        for split in by_split:
            split_stats = splits.get(split)
            if isinstance(split_stats, Mapping) and isinstance(
                split_stats.get("far_boxes"), int
            ):
                by_split[split][stratum] = int(split_stats["far_boxes"])
    if not any(by_split.values()):
        return None
    test_gameplay_far = sum(
        count
        for stratum, count in by_split["test"].items()
        if stratum != "original_file"
    )
    return {
        "far_definition": "ground-truth projected height <=64 px at 1080p",
        "far_ground_truth_by_source_stratum": by_split,
        "test_non_original_file_far_ground_truth": test_gameplay_far,
        "test_selection_status": "development_consumed_not_final_holdout",
        "test_selection_warning": (
            "This v9 test split has already been examined during application-path "
            "A/B and detail/filter development. It is development/audit evidence, "
            "not an untouched final-selection holdout."
        ),
        "test_far_domain_warning": (
            "The test split has no far ground truth from capture_session, "
            "numbered_sequence, or video_sequence groups. Test far metrics measure "
            "original-file imagery only and cannot serve as a gameplay/far release gate."
            if test_gameplay_far == 0
            else None
        ),
    }


def evaluate_checkpoint(
    *,
    weights: Path,
    data: Path,
    output: Path,
    splits: Sequence[str] = DEFAULT_SPLITS,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    batch: int = 16,
    device: str = "cpu",
    workers: int = 0,
    threads: int = 6,
    plots: bool = True,
    yolo_class: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate once per requested split and write a reproducible JSON record."""

    weights, data, output = _validated_paths(weights, data, output)
    manifest_path = data.parent / "manifest.json"
    manifest_value = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    if not splits or any(split not in DEFAULT_SPLITS for split in splits):
        raise EvaluationConfigurationError("splits must contain val and/or test")
    if len(set(splits)) != len(splits):
        raise EvaluationConfigurationError("splits must not contain duplicates")
    if imgsz <= 0 or imgsz % 32:
        raise EvaluationConfigurationError("image size must be positive and divisible by 32")
    if batch <= 0 or workers < 0 or threads <= 0:
        raise EvaluationConfigurationError("invalid batch, worker, or thread count")
    if not str(device).strip():
        raise EvaluationConfigurationError("device cannot be empty")

    _configure_cpu_environment(threads)
    model_factory = yolo_class or _load_yolo_class(threads)
    model = model_factory(str(weights))
    if getattr(model, "task", None) != "detect":
        raise EvaluationConfigurationError(
            f"unsupported checkpoint task {getattr(model, 'task', None)!r}; expected 'detect'"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    split_metrics: dict[str, Any] = {}
    for split in splits:
        if yolo_class is None:
            validator, validator_holder = _capturing_bucket_validator()
        else:
            validator, validator_holder = None, {}
        validation_options = dict(
            data=str(data),
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=str(device),
            workers=workers,
            conf=MINIMUM_PREDICTION_CONFIDENCE,
            project=str(output),
            name=split,
            exist_ok=False,
            plots=plots,
            verbose=True,
        )
        if validator is not None:
            validation_options["validator"] = validator
        metrics = model.val(**validation_options)
        split_metrics[split] = metric_summary(metrics)
        validator_instance = validator_holder.get("instance")
        if yolo_class is None and validator_instance is None:
            raise RuntimeError(
                f"bucket validator instance was not captured for {split}; "
                "refusing to write partial aggregate-only evidence"
            )
        if validator_instance is not None:
            manifest_split = "valid" if split == "val" else split
            manifest_splits = (
                manifest_value.get("splits")
                if isinstance(manifest_value, Mapping)
                else None
            )
            expected_stats = (
                manifest_splits.get(manifest_split)
                if isinstance(manifest_splits, Mapping)
                else None
            )
            if not isinstance(expected_stats, Mapping) or not all(
                isinstance(expected_stats.get(key), int)
                for key in ("images", "boxes")
            ):
                raise RuntimeError(
                    f"dataset manifest has no exact coverage counts for {split}"
                )
            validate_bucket_evidence_coverage(
                validator_instance.bucket_images,
                expected_images=int(expected_stats["images"]),
                expected_boxes=int(expected_stats["boxes"]),
                split=split,
            )
            split_metrics[split]["size_bucket_detection"] = {
                "definition": (
                    "One-to-one confidence-sorted matching at IoU >=0.50. A true "
                    "positive is assigned to its ground-truth projected-height bucket; "
                    "an unmatched prediction is assigned by its own projected height."
                ),
                "reference_height_pixels": REFERENCE_FRAME_HEIGHT,
                "iou_threshold": BUCKET_IOU_THRESHOLD,
                "minimum_prediction_confidence": MINIMUM_PREDICTION_CONFIDENCE,
                "ap50_definition": (
                    "101-point interpolated area under each bucket precision-recall "
                    "curve at IoU 0.50; uncertainty is a deterministic image-level "
                    "bootstrap percentile interval."
                ),
                **summarize_bucket_evidence(validator_instance.bucket_images),
                "sampling_caveat": (
                    "Always report detected/total, misses, predictions, and false "
                    "positives with rates. Tiny buckets and domain-poor holdouts are "
                    "not standalone release gates even when point estimates are high."
                ),
            }

    manifest = manifest_path
    record = {
        "schema_version": 2,
        "evaluator_script_sha256": _sha256_file(Path(__file__).resolve()),
        "weights": str(weights),
        "weights_sha256": _sha256_file(weights),
        "dataset_yaml": str(data),
        "dataset_yaml_sha256": _sha256_file(data),
        "dataset_manifest_sha256": _sha256_file(manifest) if manifest.is_file() else None,
        "dataset_content_sha256": (
            manifest_value.get("dataset_contract", {}).get("content_sha256")
            if isinstance(manifest_value, Mapping)
            and isinstance(manifest_value.get("dataset_contract"), Mapping)
            else None
        ),
        "dataset_evidence_scope": _dataset_evidence_scope(data),
        "configuration": {
            "splits": list(splits),
            "imgsz": imgsz,
            "batch": batch,
            "device": str(device),
            "workers": workers,
            "threads": threads,
            "plots": plots,
            "minimum_prediction_confidence": MINIMUM_PREDICTION_CONFIDENCE,
            "evaluation_mode": "ultralytics_pt_checkpoint_screening",
            "rectangular_validation_batches": True,
            "exact_static_deployment_shape": False,
            "deployment_artifact_evaluation_required": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ultralytics": _package_version("ultralytics"),
            "torch": _package_version("torch"),
        },
        "metrics": split_metrics,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = evaluate_checkpoint(
            weights=args.weights,
            data=args.data,
            output=args.output,
            splits=args.splits,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            threads=args.threads,
            plots=not args.no_plots,
        )
    except EvaluationConfigurationError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
