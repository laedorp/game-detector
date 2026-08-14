#!/usr/bin/env python3
"""Compare two exact ProAim runtime evaluations with paired image evidence.

This is a development-selection tool, not a release approver.  It compares the
same contracted validation images and reports paired recall/false-positive
deltas with deterministic image-level bootstrap intervals.  Independent
gameplay holdout, reviewed negatives, and frozen target-GPU latency remain
separate mandatory release gates.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import ctypes
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from typing import Any


SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 0
BOOTSTRAP_SAMPLES = 2000
ADVANCEMENT_CONFIDENCE = 0.25
DECISION_BUCKET = "far_33_to_64px"
EXPECTED_BUCKETS = (
    "ultra_far_le_32px",
    "far_33_to_64px",
    "medium_65_to_96px",
    "near_gt_96px",
)
MINIMUM_FAR_RECALL_GAIN = 0.10
MAXIMUM_AGGREGATE_REGRESSION = 0.01
MINIMUM_DEVELOPMENT_FAR_TARGETS = 30
MINIMUM_DEVELOPMENT_FAR_IMAGES = 30
MINIMUM_DEVELOPMENT_FAR_SOURCE_GROUPS = 15
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ComparisonError(ValueError):
    """Raised when reports cannot form trustworthy paired evidence."""


def _finite_probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite value from zero to one")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--confidence", type=_finite_probability, default=0.25,
        help="Operating point present in both reports (default: 0.25).",
    )
    parser.add_argument(
        "--baseline-pipeline", choices=("configured", "primary"), default="configured"
    )
    parser.add_argument(
        "--candidate-pipeline", choices=("configured", "primary"), default="configured"
    )
    parser.add_argument(
        "--bootstrap-samples", type=_positive_int, default=BOOTSTRAP_SAMPLES
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ComparisonError(f"non-finite JSON constant: {value}")


def _regular_report(path: Path, description: str) -> Path:
    expanded = path.expanduser().absolute()
    for component in (expanded, *expanded.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ComparisonError(f"{description} path contains a symlink: {component}")
    if expanded.is_dir():
        expanded = expanded / "metrics.json"
    if not expanded.is_file() or expanded.is_symlink():
        raise ComparisonError(f"{description} must be a regular metrics JSON file")
    return expanded.resolve()


def _read_report(path: Path, description: str) -> tuple[Path, dict[str, Any], str]:
    resolved = _regular_report(path, description)
    try:
        snapshot = resolved.read_bytes()
        value = json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ComparisonError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"{description} must contain one JSON object")
    return resolved, value, sha256(snapshot).hexdigest()


def _confidence_key(points: Mapping[str, Any], confidence: float) -> str:
    matches: list[str] = []
    for key in points:
        try:
            numeric = float(key)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric == confidence:
            matches.append(str(key))
    if len(matches) != 1:
        raise ComparisonError(
            f"report must contain exactly one operating point for confidence {confidence}"
        )
    return matches[0]


def _sha256_value(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ComparisonError(f"{description} is not a lowercase SHA-256")
    return value


def _selected_pipeline(
    report: Mapping[str, Any], pipeline: str, confidence: float, description: str
) -> dict[str, Any]:
    if report.get("schema_version") != 4:
        raise ComparisonError(f"{description} uses an unsupported runtime report schema")
    configuration = report.get("configuration")
    dataset = report.get("dataset")
    qualification = report.get("qualification")
    metrics = report.get("metrics")
    if not all(isinstance(item, Mapping) for item in (configuration, dataset, qualification, metrics)):
        raise ComparisonError(f"{description} runtime report contract is incomplete")
    assert isinstance(configuration, Mapping)
    assert isinstance(dataset, Mapping)
    assert isinstance(qualification, Mapping)
    assert isinstance(metrics, Mapping)
    if configuration.get("split") != "val" or configuration.get(
        "selection_role"
    ) != "development_validation":
        raise ComparisonError("paired model selection accepts validation reports only")
    if qualification.get("status") != "development_evidence_only" or qualification.get(
        "independent_holdout_required"
    ) is not True:
        raise ComparisonError(f"{description} misstates its development-only qualification")
    evaluator = report.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ComparisonError(f"{description} evaluator identity is missing")
    evaluator_identity = {
        "source_sha256": _sha256_value(
            evaluator.get("sha256"), f"{description} evaluator source hash"
        ),
        "pipeline_source_sha256": evaluator.get("pipeline_source_sha256"),
    }
    if not isinstance(evaluator_identity["pipeline_source_sha256"], Mapping):
        raise ComparisonError(f"{description} pipeline source identity is missing")
    for source_name, source_hash in evaluator_identity[
        "pipeline_source_sha256"
    ].items():
        if not isinstance(source_name, str):
            raise ComparisonError(f"{description} pipeline source name is invalid")
        _sha256_value(source_hash, f"{description} pipeline source hash")
    selected = metrics.get("val")
    if not isinstance(selected, Mapping):
        raise ComparisonError(f"{description} has no validation metrics")
    if pipeline == "primary":
        selected = selected.get("primary_full_frame_reference")
        if not isinstance(selected, Mapping):
            raise ComparisonError(f"{description} has no primary pipeline evidence")
    paired = selected.get("paired_image_operating_points")
    aggregate = selected.get("aggregate_detection")
    buckets = selected.get("size_bucket_detection")
    if not all(isinstance(item, Mapping) for item in (paired, aggregate, buckets)):
        raise ComparisonError(f"{description} paired/aggregate evidence is incomplete")
    assert isinstance(paired, Mapping)
    assert isinstance(aggregate, Mapping)
    assert isinstance(buckets, Mapping)
    if paired.get("schema_version") != 1:
        raise ComparisonError(f"{description} paired evidence schema is unsupported")
    paired_confidences = paired.get("confidence_thresholds")
    configured_confidences = configuration.get("reported_confidence_thresholds")
    if (
        isinstance(paired_confidences, (str, bytes))
        or not isinstance(paired_confidences, Sequence)
        or list(paired_confidences) != list(configured_confidences or ())
    ):
        raise ComparisonError(
            f"{description} paired confidence thresholds differ from configuration"
        )
    if any(isinstance(value, bool) for value in paired_confidences):
        raise ComparisonError(f"{description} confidence thresholds are invalid")
    try:
        normalized_confidences = [float(value) for value in paired_confidences]
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{description} confidence thresholds are invalid") from exc
    if (
        not normalized_confidences
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized_confidences)
        or any(
            right <= left
            for left, right in zip(
                normalized_confidences, normalized_confidences[1:], strict=False
            )
        )
    ):
        raise ComparisonError(f"{description} confidence thresholds are invalid")
    expected_point_keys = {str(value) for value in normalized_confidences}
    records = paired.get("records")
    bucket_order = paired.get("bucket_order")
    if (
        isinstance(records, (str, bytes))
        or not isinstance(records, Sequence)
        or isinstance(bucket_order, (str, bytes))
        or not isinstance(bucket_order, Sequence)
        or not records
        or list(bucket_order) != list(EXPECTED_BUCKETS)
        or paired.get("member_count") != len(records)
    ):
        raise ComparisonError(f"{description} paired evidence coverage is invalid")
    member_ids: list[str] = []
    source_group_ids: list[str] = []
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ComparisonError(f"{description} paired record is invalid")
        member_id = record.get("member_id")
        source_group_id = record.get("source_group_id")
        targets = record.get("targets")
        points = record.get("operating_points")
        if (
            not isinstance(member_id, str)
            or SHA256_PATTERN.fullmatch(member_id) is None
            or not isinstance(source_group_id, str)
            or SHA256_PATTERN.fullmatch(source_group_id) is None
            or not isinstance(targets, Mapping)
            or not isinstance(points, Mapping)
        ):
            raise ComparisonError(f"{description} paired record is incomplete")
        if set(targets) != set(EXPECTED_BUCKETS) or set(points) != expected_point_keys:
            raise ComparisonError(f"{description} paired record keys are invalid")
        validated_points: dict[str, dict[str, dict[str, int]]] = {}
        for configured_key in expected_point_keys:
            configured_point = points[configured_key]
            if not isinstance(configured_point, Mapping) or set(configured_point) != set(
                EXPECTED_BUCKETS
            ):
                raise ComparisonError(f"{description} paired operating point is invalid")
            validated_buckets: dict[str, dict[str, int]] = {}
            for bucket in EXPECTED_BUCKETS:
                target = targets.get(bucket)
                counts = configured_point.get(bucket)
                if (
                    isinstance(target, bool)
                    or not isinstance(target, int)
                    or target < 0
                    or not isinstance(counts, Mapping)
                    or set(counts) != {"true_positives", "false_positives"}
                ):
                    raise ComparisonError(f"{description} paired bucket is invalid")
                tp = counts.get("true_positives")
                fp = counts.get("false_positives")
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (tp, fp)
                ) or int(tp) > target:
                    raise ComparisonError(f"{description} paired bucket count is invalid")
                validated_buckets[bucket] = {
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                }
            validated_points[configured_key] = validated_buckets
        point_key = _confidence_key(points, confidence)
        normalized_targets: dict[str, int] = {}
        normalized_counts: dict[str, dict[str, int]] = {}
        for bucket in bucket_order:
            normalized_targets[bucket] = int(targets[bucket])
            normalized_counts[bucket] = validated_points[point_key][bucket]
        member_ids.append(member_id)
        source_group_ids.append(source_group_id)
        normalized_records.append(
            {
                "member_id": member_id,
                "source_group_id": source_group_id,
                "targets": normalized_targets,
                "counts": normalized_counts,
                "all_counts": validated_points,
            }
        )
    if len(set(member_ids)) != len(member_ids) or paired.get(
        "member_sequence_sha256"
    ) != _canonical_hash(member_ids):
        raise ComparisonError(f"{description} paired member sequence is invalid")
    if (
        paired.get("source_group_count") != len(set(source_group_ids))
        or paired.get("source_group_sequence_sha256")
        != _canonical_hash(source_group_ids)
    ):
        raise ComparisonError(f"{description} paired source-group sequence is invalid")
    aggregate_points = aggregate.get("operating_points")
    bucket_points = buckets.get("operating_points")
    if (
        not isinstance(aggregate_points, Mapping)
        or not isinstance(bucket_points, Mapping)
        or set(aggregate_points) != expected_point_keys
        or set(bucket_points) != expected_point_keys
    ):
        raise ComparisonError(f"{description} summary operating points are missing")
    for configured_key in expected_point_keys:
        summarized_aggregate = aggregate_points[configured_key]
        summarized_buckets = bucket_points[configured_key]
        if (
            not isinstance(summarized_aggregate, Mapping)
            or not isinstance(summarized_buckets, Mapping)
            or set(summarized_buckets) != set(EXPECTED_BUCKETS)
        ):
            raise ComparisonError(f"{description} summary operating point is invalid")
        for bucket in bucket_order:
            expected = {
                "ground_truth_total": sum(
                    record["targets"][bucket] for record in normalized_records
                ),
                "detected_true_positives": sum(
                    record["all_counts"][configured_key][bucket]["true_positives"]
                    for record in normalized_records
                ),
                "false_positives": sum(
                    record["all_counts"][configured_key][bucket]["false_positives"]
                    for record in normalized_records
                ),
            }
            actual = summarized_buckets.get(bucket)
            if not isinstance(actual, Mapping) or any(
                actual.get(key) != value for key, value in expected.items()
            ):
                raise ComparisonError(f"{description} paired and bucket summaries disagree")
        expected_aggregate = {
            "ground_truth_total": sum(
                sum(record["targets"].values()) for record in normalized_records
            ),
            "detected_true_positives": sum(
                sum(
                    item["true_positives"]
                    for item in record["all_counts"][configured_key].values()
                )
                for record in normalized_records
            ),
            "false_positives": sum(
                sum(
                    item["false_positives"]
                    for item in record["all_counts"][configured_key].values()
                )
                for record in normalized_records
            ),
        }
        if any(
            summarized_aggregate.get(key) != value
            for key, value in expected_aggregate.items()
        ):
            raise ComparisonError(f"{description} paired and aggregate summaries disagree")
    return {
        "records": normalized_records,
        "bucket_order": list(bucket_order),
        "member_ids": member_ids,
        "source_group_ids": source_group_ids,
        "source_group_count": len(set(source_group_ids)),
        "member_sequence_sha256": paired.get("member_sequence_sha256"),
        "split_content_sha256": _sha256_value(
            paired.get("split_content_sha256"), f"{description} split content hash"
        ),
        "dataset_content_sha256": _sha256_value(
            dataset.get("content_sha256"), f"{description} dataset content hash"
        ),
        "dataset_manifest_sha256": _sha256_value(
            dataset.get("manifest_sha256"), f"{description} dataset manifest hash"
        ),
        "model_content_sha256": _sha256_value(
            report.get("model_artifact", {}).get("content_sha256")
            if isinstance(report.get("model_artifact"), Mapping)
            else None,
            f"{description} model content hash",
        ),
        "evaluator_identity": evaluator_identity,
        "matching_contract": {
            key: configuration.get(key)
            for key in (
                "matching_iou_threshold",
                "minimum_prediction_confidence",
                "reported_confidence_thresholds",
                "runtime_nms_iou_threshold",
            )
        },
        "configuration": configuration,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(records: Sequence[Mapping[str, Any]], buckets: Sequence[str]) -> dict[str, Any]:
    targets = sum(sum(int(record["targets"][bucket]) for bucket in buckets) for record in records)
    true_positives = sum(
        sum(int(record["counts"][bucket]["true_positives"]) for bucket in buckets)
        for record in records
    )
    false_positives = sum(
        sum(int(record["counts"][bucket]["false_positives"]) for bucket in buckets)
        for record in records
    )
    predictions = true_positives + false_positives
    return {
        "ground_truth_total": targets,
        "true_positives": true_positives,
        "false_negatives": targets - true_positives,
        "false_positives": false_positives,
        "recall": true_positives / targets if targets else None,
        "precision": true_positives / predictions if predictions else None,
        "false_positives_per_1000_images": false_positives / len(records) * 1000.0,
    }


def _paired_comparison(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    buckets: Sequence[str],
    *,
    samples: int,
    seed_material: str,
) -> dict[str, Any]:
    baseline_summary = _summary(baseline, buckets)
    candidate_summary = _summary(candidate, buckets)
    recall_deltas: list[float] = []
    fp_rate_deltas: list[float] = []
    derived_seed = int.from_bytes(
        sha256(f"{BOOTSTRAP_SEED}:{seed_material}".encode()).digest()[:8], "big"
    )
    generator = random.Random(derived_seed)
    for _ in range(samples):
        indices = [generator.randrange(len(baseline)) for _ in range(len(baseline))]
        sampled_baseline = [baseline[index] for index in indices]
        sampled_candidate = [candidate[index] for index in indices]
        left = _summary(sampled_baseline, buckets)
        right = _summary(sampled_candidate, buckets)
        if left["recall"] is not None and right["recall"] is not None:
            recall_deltas.append(float(right["recall"]) - float(left["recall"]))
        fp_rate_deltas.append(
            float(right["false_positives_per_1000_images"])
            - float(left["false_positives_per_1000_images"])
        )
    recall_delta = (
        float(candidate_summary["recall"]) - float(baseline_summary["recall"])
        if baseline_summary["recall"] is not None and candidate_summary["recall"] is not None
        else None
    )
    precision_delta = (
        float(candidate_summary["precision"]) - float(baseline_summary["precision"])
        if baseline_summary["precision"] is not None
        and candidate_summary["precision"] is not None
        else None
    )
    return {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta_candidate_minus_baseline": {
            "recall": recall_delta,
            "precision": precision_delta,
            "false_positives": candidate_summary["false_positives"]
            - baseline_summary["false_positives"],
            "false_positives_per_1000_images": candidate_summary[
                "false_positives_per_1000_images"
            ]
            - baseline_summary["false_positives_per_1000_images"],
        },
        "paired_image_bootstrap_95_ci": {
            "method": "deterministic paired image-level nonparametric percentile interval",
            "base_seed": BOOTSTRAP_SEED,
            "derived_seed": derived_seed,
            "samples_requested": samples,
            "samples_with_ground_truth": len(recall_deltas),
            "recall_delta": (
                [_percentile(recall_deltas, 0.025), _percentile(recall_deltas, 0.975)]
                if recall_deltas
                else None
            ),
            "false_positives_per_1000_images_delta": [
                _percentile(fp_rate_deltas, 0.025),
                _percentile(fp_rate_deltas, 0.975),
            ],
        },
    }


def _paired_source_group_bootstrap(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    buckets: Sequence[str],
    *,
    samples: int,
    seed_material: str,
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ComparisonError("paired source-group bootstrap requires matching records")
    grouped_indices: dict[str, list[int]] = {}
    for index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
        left_group = left.get("source_group_id")
        if left_group != right.get("source_group_id") or not isinstance(left_group, str):
            raise ComparisonError("baseline and candidate source groups differ")
        grouped_indices.setdefault(left_group, []).append(index)
    group_ids = sorted(grouped_indices)
    generator = random.Random(
        int.from_bytes(
            sha256(f"{BOOTSTRAP_SEED}:{seed_material}".encode()).digest()[:8], "big"
        )
    )
    recall_deltas: list[float] = []
    fp_rate_deltas: list[float] = []
    for _ in range(samples):
        selected_groups = [
            group_ids[generator.randrange(len(group_ids))] for _ in group_ids
        ]
        indices = [
            index for group_id in selected_groups for index in grouped_indices[group_id]
        ]
        sampled_baseline = [baseline[index] for index in indices]
        sampled_candidate = [candidate[index] for index in indices]
        left = _summary(sampled_baseline, buckets)
        right = _summary(sampled_candidate, buckets)
        if left["recall"] is not None and right["recall"] is not None:
            recall_deltas.append(float(right["recall"]) - float(left["recall"]))
        fp_rate_deltas.append(
            float(right["false_positives_per_1000_images"])
            - float(left["false_positives_per_1000_images"])
        )
    return {
        "method": (
            "deterministic paired source-group cluster nonparametric percentile interval; "
            "all images in a sampled grouped-dataset cluster are resampled together"
        ),
        "base_seed": BOOTSTRAP_SEED,
        "derived_seed": int.from_bytes(
            sha256(f"{BOOTSTRAP_SEED}:{seed_material}".encode()).digest()[:8], "big"
        ),
        "source_group_count": len(group_ids),
        "samples_requested": samples,
        "samples_with_ground_truth": len(recall_deltas),
        "recall_delta": (
            [_percentile(recall_deltas, 0.025), _percentile(recall_deltas, 0.975)]
            if recall_deltas
            else None
        ),
        "false_positives_per_1000_images_delta": [
            _percentile(fp_rate_deltas, 0.025),
            _percentile(fp_rate_deltas, 0.975),
        ],
    }


def _publish(output: Path, record: Mapping[str, Any]) -> None:
    destination = output.expanduser().absolute()
    for component in destination.parents:
        if os.path.lexists(component) and component.is_symlink():
            raise ComparisonError(f"output path contains a symlink: {component}")
    if os.path.lexists(destination):
        raise ComparisonError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.comparison-", dir=destination.parent))
    published = False
    try:
        target = staging / "comparison.json"
        with target.open("xb") as stream:
            stream.write(_canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        if sys.platform.startswith("linux"):
            library = ctypes.CDLL(None, use_errno=True)
            renameat2 = getattr(library, "renameat2", None)
            if renameat2 is None:
                raise ComparisonError("atomic no-replace publication is unavailable")
            renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            renameat2.restype = ctypes.c_int
            if renameat2(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) != 0:
                error = ctypes.get_errno()
                if error == errno.EEXIST:
                    raise ComparisonError(f"output appeared during comparison: {destination}")
                raise ComparisonError(f"atomic publication failed: {os.strerror(error)}")
        elif os.name == "nt":
            os.rename(staging, destination)
        else:
            raise ComparisonError("atomic publication is supported only on Linux and Windows")
        published = True
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)


def compare_reports(
    *,
    baseline: Path,
    candidate: Path,
    output: Path,
    confidence: float = 0.25,
    baseline_pipeline: str = "configured",
    candidate_pipeline: str = "configured",
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    source = Path(__file__).resolve()
    source_sha256 = _sha256_file(source)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ComparisonError("confidence must be finite from zero to one")
    if baseline_pipeline not in {"configured", "primary"} or candidate_pipeline not in {
        "configured",
        "primary",
    }:
        raise ComparisonError("pipeline must be configured or primary")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples <= 0:
        raise ComparisonError("bootstrap samples must be a positive integer")
    baseline_path, baseline_report, baseline_sha = _read_report(baseline, "baseline report")
    candidate_path, candidate_report, candidate_sha = _read_report(candidate, "candidate report")
    if baseline_sha == candidate_sha and baseline_pipeline == candidate_pipeline:
        raise ComparisonError(
            "baseline and candidate must use different reports or different pipelines"
        )
    left = _selected_pipeline(baseline_report, baseline_pipeline, confidence, "baseline")
    right = _selected_pipeline(candidate_report, candidate_pipeline, confidence, "candidate")
    for field in (
        "bucket_order",
        "member_ids",
        "source_group_ids",
        "source_group_count",
        "member_sequence_sha256",
        "split_content_sha256",
        "dataset_content_sha256",
        "dataset_manifest_sha256",
        "evaluator_identity",
        "matching_contract",
    ):
        if left[field] != right[field]:
            raise ComparisonError(f"baseline and candidate {field} differ")
    for left_record, right_record in zip(left["records"], right["records"], strict=True):
        if left_record["targets"] != right_record["targets"]:
            raise ComparisonError("baseline and candidate per-image targets differ")
    if DECISION_BUCKET not in left["bucket_order"]:
        raise ComparisonError(f"reports do not contain decision bucket {DECISION_BUCKET}")
    shared_seed_contract = _canonical_hash(
        {
            "schema_version": 1,
            "split_content_sha256": left["split_content_sha256"],
            "member_sequence_sha256": left["member_sequence_sha256"],
            "confidence": confidence,
        }
    )
    bucket_comparisons = {
        bucket: _paired_comparison(
            left["records"],
            right["records"],
            [bucket],
            samples=bootstrap_samples,
            seed_material=f"{shared_seed_contract}:{bootstrap_samples}:{bucket}",
        )
        for bucket in left["bucket_order"]
    }
    aggregate = _paired_comparison(
        left["records"],
        right["records"],
        left["bucket_order"],
        samples=bootstrap_samples,
        seed_material=f"{shared_seed_contract}:{bootstrap_samples}:aggregate",
    )
    far = bucket_comparisons[DECISION_BUCKET]
    far_delta = far["delta_candidate_minus_baseline"]
    far_bootstrap = far["paired_image_bootstrap_95_ci"]
    far_ci = far_bootstrap["recall_delta"]
    far_source_group_bootstrap = _paired_source_group_bootstrap(
        left["records"],
        right["records"],
        [DECISION_BUCKET],
        samples=bootstrap_samples,
        seed_material=(
            f"{shared_seed_contract}:{bootstrap_samples}:{DECISION_BUCKET}:source-group"
        ),
    )
    far_source_group_ci = far_source_group_bootstrap["recall_delta"]
    far["paired_source_group_bootstrap_95_ci"] = far_source_group_bootstrap
    aggregate_delta = aggregate["delta_candidate_minus_baseline"]
    far_target_images = sum(
        record["targets"][DECISION_BUCKET] > 0 for record in left["records"]
    )
    far_target_source_groups = len(
        {
            record["source_group_id"]
            for record in left["records"]
            if record["targets"][DECISION_BUCKET] > 0
        }
    )
    checks = {
        "confidence_is_release_default_0_25": confidence == ADVANCEMENT_CONFIDENCE,
        "bootstrap_uses_exact_2000_samples": bootstrap_samples == BOOTSTRAP_SAMPLES,
        "far_decision_bucket_has_at_least_30_targets": (
            far["baseline"]["ground_truth_total"] >= MINIMUM_DEVELOPMENT_FAR_TARGETS
        ),
        "far_decision_bucket_spans_at_least_30_images": (
            far_target_images >= MINIMUM_DEVELOPMENT_FAR_IMAGES
        ),
        "far_decision_bucket_spans_at_least_15_source_groups": (
            far_target_source_groups >= MINIMUM_DEVELOPMENT_FAR_SOURCE_GROUPS
        ),
        "far_bootstrap_has_no_zero_target_resamples": (
            far_bootstrap["samples_with_ground_truth"] == bootstrap_samples
        ),
        "far_recall_gain_at_least_10_points": (
            far_delta["recall"] is not None
            and far_delta["recall"] >= MINIMUM_FAR_RECALL_GAIN
        ),
        "far_recall_bootstrap_lower_bound_above_zero": (
            far_ci is not None and far_ci[0] > 0.0
        ),
        "far_source_group_bootstrap_has_no_zero_target_resamples": (
            far_source_group_bootstrap["samples_with_ground_truth"]
            == bootstrap_samples
        ),
        "far_source_group_bootstrap_lower_bound_above_zero": (
            far_source_group_ci is not None and far_source_group_ci[0] > 0.0
        ),
        "far_false_positives_do_not_increase": far_delta["false_positives"] <= 0,
        "aggregate_recall_regression_no_more_than_1_point": (
            aggregate_delta["recall"] is not None
            and aggregate_delta["recall"] >= -MAXIMUM_AGGREGATE_REGRESSION
        ),
        "aggregate_precision_regression_no_more_than_1_point": (
            aggregate_delta["precision"] is not None
            and aggregate_delta["precision"] >= -MAXIMUM_AGGREGATE_REGRESSION
        ),
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_selection_evidence_only",
        "comparator": {"path": source.name, "sha256": source_sha256},
        "reports": {
            "baseline": {
                "metrics_sha256": baseline_sha,
                "model_content_sha256": left["model_content_sha256"],
                "pipeline": baseline_pipeline,
            },
            "candidate": {
                "metrics_sha256": candidate_sha,
                "model_content_sha256": right["model_content_sha256"],
                "pipeline": candidate_pipeline,
            },
        },
        "paired_contract": {
            "dataset_content_sha256": left["dataset_content_sha256"],
            "dataset_manifest_sha256": left["dataset_manifest_sha256"],
            "split_content_sha256": left["split_content_sha256"],
            "member_sequence_sha256": left["member_sequence_sha256"],
            "member_count": len(left["records"]),
            "confidence": confidence,
            "bucket_order": left["bucket_order"],
        },
        "comparison": {"aggregate": aggregate, "buckets": bucket_comparisons},
        "development_advancement_policy": {
            "decision_bucket": DECISION_BUCKET,
            "minimum_absolute_recall_gain": MINIMUM_FAR_RECALL_GAIN,
            "minimum_development_far_targets": MINIMUM_DEVELOPMENT_FAR_TARGETS,
            "minimum_development_far_images": MINIMUM_DEVELOPMENT_FAR_IMAGES,
            "minimum_development_far_source_groups": (
                MINIMUM_DEVELOPMENT_FAR_SOURCE_GROUPS
            ),
            "far_target_bearing_images": far_target_images,
            "far_target_bearing_source_groups": far_target_source_groups,
            "required_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "required_advancement_confidence": ADVANCEMENT_CONFIDENCE,
            "shared_bootstrap_seed_contract_sha256": shared_seed_contract,
            "maximum_aggregate_precision_or_recall_regression": MAXIMUM_AGGREGATE_REGRESSION,
            "checks": checks,
            "passed": all(checks.values()),
            "scope": "May advance a validation candidate to independent/hardware testing only.",
        },
        "release_qualification": {
            "qualified": False,
            "independent_holdout_required": True,
            "reviewed_negative_scenes_required": True,
            "exact_frozen_target_gpu_latency_required": True,
            "reason": (
                "This comparison uses development validation evidence. It cannot approve "
                "a release model regardless of the advancement-policy result."
            ),
        },
    }
    # Catch ordinary report mutation during comparison before publishing a
    # result that claims different input bytes.
    if (
        _sha256_file(source) != source_sha256
        or _sha256_file(baseline_path) != baseline_sha
        or _sha256_file(candidate_path) != candidate_sha
    ):
        raise ComparisonError("comparator source or input report changed during comparison")
    _publish(output, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = compare_reports(
            baseline=args.baseline,
            candidate=args.candidate,
            output=args.output,
            confidence=args.confidence,
            baseline_pipeline=args.baseline_pipeline,
            candidate_pipeline=args.candidate_pipeline,
            bootstrap_samples=args.bootstrap_samples,
        )
    except ComparisonError as exc:
        print(f"Runtime comparison failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(record["development_advancement_policy"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
