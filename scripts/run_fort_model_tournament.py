#!/usr/bin/env python3
"""Select a FORT development candidate from sealed runtime evidence.

This command performs no training, export, or inference.  It validates four
completed candidate bundles (YOLO26 n/s x end-to-end/traditional), consumes
their exact 384x640 validation reports, invokes the paired runtime comparator
at the fixed 0.25/2,000 operating point, and atomically publishes one
development-only selection bundle.  The portable plan, all four privacy-safe
runtime reports, and both training-results tables are copied into that sealed
bundle so ignored run directories are not an evidence dependency.  Test
reports are never accepted and no result from this command can qualify a
release.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_fort_runtime_evaluations import (  # noqa: E402
    ADVANCEMENT_CONFIDENCE,
    BOOTSTRAP_SAMPLES,
    compare_reports,
)
from scripts.export_fort_release_candidate import (  # noqa: E402
    CandidateExportError,
    _validate_packaged_training_provenance,
)
from scripts.evaluate_fort_runtime_model import _source_hash_snapshot  # noqa: E402
from utils.public_evidence import contains_nonportable_path  # noqa: E402


SCHEMA_VERSION = 1
PLAN_STATUS = "development_model_tournament_plan"
OUTPUT_STATUS = "development_model_selection_only"
EXACT_INFERENCE_SIZE = "384x640"
EXACT_INPUT_SHAPE = [1, 3, 384, 640]
SCALES = ("n", "s")
HEADS = ("end2end", "traditional")
EXPECTED_RELEASE_GATE = {
    "approved": False,
    "model_accuracy_qualified": False,
    "target_gpu_latency_qualified": False,
    "frozen_build_qualified": False,
    "independent_holdout_qualified": False,
}
REQUIRED_EVIDENCE_CHECKS = {
    "confidence_is_release_default_0_25",
    "bootstrap_uses_exact_2000_samples",
    "far_decision_bucket_has_at_least_30_targets",
    "far_decision_bucket_spans_at_least_30_images",
    "far_decision_bucket_spans_at_least_15_source_groups",
    "far_bootstrap_has_no_zero_target_resamples",
    "far_source_group_bootstrap_has_no_zero_target_resamples",
}
OUTCOME_CHECKS = {
    "far_recall_gain_at_least_10_points",
    "far_recall_bootstrap_lower_bound_above_zero",
    "far_source_group_bootstrap_lower_bound_above_zero",
    "far_false_positives_do_not_increase",
    "aggregate_recall_regression_no_more_than_1_point",
    "aggregate_precision_regression_no_more_than_1_point",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_MEMBER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
SAFE_DEVICE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}(?::[0-9]+)?$")
SEALED_PLAN_NAME = "tournament-plan.json"
SEALED_REPORT_NAME = "validation-metrics.json"
SEALED_RESULTS_NAME = "training-results.csv"


class TournamentError(ValueError):
    """Raised when a plan or evidence bundle cannot support selection."""


@dataclass(frozen=True, slots=True)
class Slot:
    scale: str
    head: str
    candidate_dir: Path
    report_path: Path
    candidate_manifest_sha256: str
    candidate_content_sha256: str
    checkpoint_sha256: str
    initial_weights_sha256: str
    training_identity: Mapping[str, Any]
    onnx_name: str
    onnx_sha256: str
    onnx_bytes: int
    report_sha256: str
    report_model_content_sha256: str
    plan_paths: Mapping[str, str]

    @property
    def name(self) -> str:
        return f"{self.scale}_{self.head}"


@dataclass(frozen=True, slots=True)
class ComparisonOutcome:
    name: str
    passed: bool
    record: Mapping[str, Any]
    relative_path: str
    sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Strict JSON plan binding all four completed candidate/report pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New atomically published development-selection directory.",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TournamentError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
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


def _manifest_content_hash(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    body.pop("candidate_content_sha256", None)
    return _canonical_hash(body)


def _selection_content_hash(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    body.pop("selection_content_sha256", None)
    return _canonical_hash(body)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TournamentError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TournamentError(f"non-finite JSON constant: {value}")


def _reject_symlink_components(path: Path, description: str) -> None:
    absolute = path.expanduser().absolute()
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise TournamentError(
                f"{description} path contains a symlink component: {component}"
            )


def _regular_file(path: Path, description: str, suffix: str | None = None) -> Path:
    unresolved = path.expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    _reject_symlink_components(unresolved, description)
    if not unresolved.is_file() or unresolved.is_symlink():
        raise TournamentError(f"{description} must be a local regular file: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    if suffix is not None and resolved.suffix.casefold() != suffix.casefold():
        raise TournamentError(f"{description} must use the {suffix} extension: {resolved}")
    return resolved


def _regular_directory(path: Path, description: str) -> Path:
    unresolved = path.expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    _reject_symlink_components(unresolved, description)
    if not unresolved.is_dir() or unresolved.is_symlink():
        raise TournamentError(f"{description} must be a local regular directory: {unresolved}")
    return unresolved.resolve(strict=True)


def _read_json_snapshot(path: Path, description: str) -> tuple[dict[str, Any], str]:
    try:
        snapshot = path.read_bytes()
        value = json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except TournamentError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TournamentError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise TournamentError(f"{description} must contain one JSON object")
    return value, sha256(snapshot).hexdigest()


def _exact_keys(value: object, expected: set[str], description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise TournamentError(
            f"{description} keys differ; expected={sorted(expected)}, actual={actual}"
        )
    return value


def _sha256_value(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise TournamentError(f"{description} must be a lowercase SHA-256")
    return value


def _reject_absolute_path_strings(value: object, description: str) -> None:
    """Keep sealed public evidence free of workstation-specific paths."""

    if isinstance(value, str):
        if contains_nonportable_path(value):
            raise TournamentError(
                f"{description} contains a non-portable absolute path or file URI"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TournamentError(
                    f"{description} contains a non-string JSON field name"
                )
            _reject_absolute_path_strings(key, f"{description} field name")
            _reject_absolute_path_strings(item, description)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_absolute_path_strings(item, description)


def _positive_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TournamentError(f"{description} must be a positive integer")
    return value


def _resolve_plan_path(base: Path, value: object, description: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise TournamentError(f"{description} must be a non-empty path string")
    if value != value.strip() or "\\" in value or ":" in value:
        raise TournamentError(
            f"{description} must be a canonical portable relative path"
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or not pure.parts
        or any(
            part in {"", ".", ".."}
            or SAFE_MEMBER_PATTERN.fullmatch(part) is None
            for part in pure.parts
        )
    ):
        raise TournamentError(
            f"{description} must be a canonical portable relative path"
        )
    return base.joinpath(*pure.parts)


def _report_file(path: Path, description: str) -> Path:
    unresolved = path
    _reject_symlink_components(unresolved, description)
    if unresolved.is_dir() and not unresolved.is_symlink():
        unresolved = unresolved / "metrics.json"
    return _regular_file(unresolved, description, ".json")


def _validate_plan(
    plan_path: Path,
) -> tuple[dict[str, Any], str, dict[tuple[str, str], tuple[Path, Path]], Path]:
    plan_path = _regular_file(plan_path, "tournament plan", ".json")
    plan, plan_sha256 = _read_json_snapshot(plan_path, "tournament plan")
    try:
        plan_payload = plan_path.read_bytes()
    except OSError as exc:
        raise TournamentError(f"cannot reread tournament plan: {exc}") from exc
    if sha256(plan_payload).hexdigest() != plan_sha256:
        raise TournamentError("tournament plan changed while it was being parsed")
    if plan_payload != _canonical_bytes(plan):
        raise TournamentError(
            "tournament plan must be canonical sorted JSON with one trailing newline"
        )
    _exact_keys(
        plan,
        {"schema_version", "status", "dataset", "runtime", "models"},
        "tournament plan",
    )
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("status") != PLAN_STATUS:
        raise TournamentError("tournament plan schema/status is unsupported")
    dataset = _exact_keys(
        plan.get("dataset"), {"manifest_sha256", "content_sha256"}, "plan dataset"
    )
    _sha256_value(dataset.get("manifest_sha256"), "plan dataset manifest hash")
    _sha256_value(dataset.get("content_sha256"), "plan dataset content hash")
    runtime = _exact_keys(
        plan.get("runtime"),
        {
            "backend",
            "device",
            "require_full_provider",
            "inference_size",
            "detail_crop_size_source_pixels",
            "confidence",
            "comparator_bootstrap_samples",
        },
        "plan runtime",
    )
    if runtime.get("backend") != "onnxruntime":
        raise TournamentError("tournament backend must be onnxruntime")
    if runtime.get("inference_size") != EXACT_INFERENCE_SIZE:
        raise TournamentError(f"tournament inference size must be {EXACT_INFERENCE_SIZE}")
    device = runtime.get("device")
    if not isinstance(device, str) or SAFE_DEVICE_PATTERN.fullmatch(device) is None:
        raise TournamentError(
            "tournament runtime device must be a portable provider/device token"
        )
    if not isinstance(runtime.get("require_full_provider"), bool):
        raise TournamentError("require_full_provider must be a JSON boolean")
    _positive_int(
        runtime.get("detail_crop_size_source_pixels"),
        "detail crop size",
    )
    confidence = runtime.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or float(confidence) != ADVANCEMENT_CONFIDENCE
    ):
        raise TournamentError(
            f"tournament confidence must be exactly {ADVANCEMENT_CONFIDENCE}"
        )
    samples = runtime.get("comparator_bootstrap_samples")
    if isinstance(samples, bool) or samples != BOOTSTRAP_SAMPLES:
        raise TournamentError(
            f"tournament comparator bootstrap count must be exactly {BOOTSTRAP_SAMPLES}"
        )

    models = _exact_keys(plan.get("models"), set(SCALES), "plan models")
    resolved: dict[tuple[str, str], tuple[Path, Path]] = {}
    candidate_paths: set[Path] = set()
    report_paths: set[Path] = set()
    initial_hashes: dict[str, str] = {}
    for scale in SCALES:
        scale_plan = _exact_keys(
            models.get(scale),
            {"initial_weights_sha256", *HEADS},
            f"plan model {scale}",
        )
        initial_hashes[scale] = _sha256_value(
            scale_plan.get("initial_weights_sha256"),
            f"plan model {scale} initial weights hash",
        )
        for head in HEADS:
            slot = _exact_keys(
                scale_plan.get(head),
                {"candidate_dir", "validation_report"},
                f"plan slot {scale}_{head}",
            )
            candidate = _regular_directory(
                _resolve_plan_path(
                    plan_path.parent,
                    slot.get("candidate_dir"),
                    f"plan slot {scale}_{head} candidate",
                ),
                f"plan slot {scale}_{head} candidate",
            )
            report = _report_file(
                _resolve_plan_path(
                    plan_path.parent,
                    slot.get("validation_report"),
                    f"plan slot {scale}_{head} report",
                ),
                f"plan slot {scale}_{head} report",
            )
            if candidate in candidate_paths or report in report_paths:
                raise TournamentError(
                    "all four candidate and validation-report paths must be distinct"
                )
            candidate_paths.add(candidate)
            report_paths.add(report)
            resolved[(scale, head)] = (candidate, report)
    if initial_hashes["n"] == initial_hashes["s"]:
        raise TournamentError("n and s must use different pinned initial-weight hashes")
    return plan, plan_sha256, resolved, plan_path


def _safe_member_name(value: object, description: str) -> str:
    if not isinstance(value, str) or SAFE_MEMBER_PATTERN.fullmatch(value) is None:
        raise TournamentError(f"{description} has an unsafe member name")
    return value


def _validate_candidate(
    *,
    scale: str,
    head: str,
    directory: Path,
    expected_initial_weights_sha256: str,
    expected_dataset: Mapping[str, Any],
    expected_exporter_sha256: str,
    packaged_training_validator: Callable[..., None],
) -> tuple[dict[str, Any], dict[Path, str]]:
    manifest_path = _regular_file(
        directory / "candidate-manifest.json", f"{scale}_{head} candidate manifest", ".json"
    )
    manifest, manifest_sha256 = _read_json_snapshot(
        manifest_path, f"{scale}_{head} candidate manifest"
    )
    if manifest.get("schema_version") != 1 or manifest.get(
        "status"
    ) != "validated_release_candidate_not_approved":
        raise TournamentError(f"{scale}_{head} candidate schema/status is unsupported")
    content_sha256 = _sha256_value(
        manifest.get("candidate_content_sha256"),
        f"{scale}_{head} candidate content hash",
    )
    if content_sha256 != _manifest_content_hash(manifest):
        raise TournamentError(f"{scale}_{head} candidate content hash mismatch")
    if manifest.get("release_gate") != EXPECTED_RELEASE_GATE:
        raise TournamentError(f"{scale}_{head} candidate release gates are not all false")

    exporter = manifest.get("exporter")
    if not isinstance(exporter, Mapping):
        raise TournamentError(f"{scale}_{head} candidate exporter identity is missing")
    if exporter.get("sha256") != expected_exporter_sha256:
        raise TournamentError(
            f"{scale}_{head} candidate uses a different exporter revision; restage it"
        )

    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TournamentError(f"{scale}_{head} candidate dataset identity is missing")
    for key in ("manifest_sha256", "content_sha256"):
        if dataset.get(key) != expected_dataset.get(key):
            raise TournamentError(f"{scale}_{head} candidate dataset {key} differs from plan")

    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise TournamentError(f"{scale}_{head} candidate configuration is missing")
    basename = configuration.get("basename")
    if (
        not isinstance(basename, str)
        or SAFE_MEMBER_PATTERN.fullmatch(basename) is None
        or "." in basename
    ):
        raise TournamentError(f"{scale}_{head} candidate basename is unsafe")
    if (
        configuration.get("inference_size") != EXACT_INFERENCE_SIZE
        or configuration.get("input_shape_nchw") != EXACT_INPUT_SHAPE
        or configuration.get("head") != head
        or configuration.get("one_class") != {"0": "player"}
    ):
        raise TournamentError(f"{scale}_{head} candidate deployment contract differs")

    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TournamentError(f"{scale}_{head} candidate checkpoint identity is missing")
    checkpoint_sha256 = _sha256_value(
        checkpoint.get("sha256"), f"{scale}_{head} checkpoint hash"
    )
    _positive_int(checkpoint.get("bytes"), f"{scale}_{head} checkpoint size")
    provenance = manifest.get("training_provenance")
    if not isinstance(provenance, Mapping):
        raise TournamentError(f"{scale}_{head} candidate training provenance is missing")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("checkpoint_role") != "completed_run_best"
        or provenance.get("checkpoint_sha256") != checkpoint_sha256
        or provenance.get("initial_weights_sha256") != expected_initial_weights_sha256
    ):
        raise TournamentError(f"{scale}_{head} candidate is not a completed pinned training run")
    _positive_int(provenance.get("completed_epochs"), f"{scale}_{head} completed epochs")
    _positive_int(provenance.get("results_rows"), f"{scale}_{head} training result rows")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TournamentError(f"{scale}_{head} candidate artifact inventory is missing")
    required_artifacts = {
        "onnx",
        "openvino_xml",
        "openvino_bin",
        "labels",
        "attribution",
        "initial_run_contract",
        "training_reproducibility",
        "training_results",
    }
    allowed_artifacts = required_artifacts | {"ultralytics_metadata"}
    if not required_artifacts.issubset(artifacts) or not set(artifacts).issubset(
        allowed_artifacts
    ):
        raise TournamentError(f"{scale}_{head} candidate artifact inventory differs")
    expected_names = {
        "onnx": f"{basename}.onnx",
        "openvino_xml": f"{basename}.xml",
        "openvino_bin": f"{basename}.bin",
        "labels": "labels.txt",
        "attribution": "ATTRIBUTION.md",
        "initial_run_contract": "training-initial-run-contract.json",
        "training_reproducibility": "training-reproducibility.json",
        "training_results": "training-results.csv",
        "ultralytics_metadata": "ultralytics-metadata.yaml",
    }
    snapshots: dict[Path, str] = {manifest_path: manifest_sha256}
    actual_names: set[str] = {manifest_path.name}
    validated_artifacts: dict[str, dict[str, Any]] = {}
    for key, raw_record in artifacts.items():
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"name", "bytes", "sha256"}:
            raise TournamentError(f"{scale}_{head} artifact {key} record is invalid")
        name = _safe_member_name(raw_record.get("name"), f"{scale}_{head} artifact {key}")
        if name != expected_names[key]:
            raise TournamentError(f"{scale}_{head} artifact {key} has the wrong name")
        artifact_path = _regular_file(directory / name, f"{scale}_{head} artifact {key}")
        if artifact_path.parent != directory:
            raise TournamentError(f"{scale}_{head} artifact {key} escapes its candidate")
        artifact_sha256 = _sha256_value(
            raw_record.get("sha256"), f"{scale}_{head} artifact {key} hash"
        )
        artifact_bytes = _positive_int(
            raw_record.get("bytes"), f"{scale}_{head} artifact {key} size"
        )
        if artifact_path.stat().st_size != artifact_bytes or _sha256_file(
            artifact_path
        ) != artifact_sha256:
            raise TournamentError(f"{scale}_{head} artifact {key} hash/size mismatch")
        actual_names.add(name)
        snapshots[artifact_path] = artifact_sha256
        validated_artifacts[key] = dict(raw_record)
    for child in directory.iterdir():
        if child.is_symlink() or not child.is_file():
            raise TournamentError(f"{scale}_{head} candidate has an unsafe member: {child}")
    if {child.name for child in directory.iterdir()} != actual_names:
        raise TournamentError(f"{scale}_{head} candidate has missing or unexpected members")
    if (directory / "labels.txt").read_bytes() != b"player\n":
        raise TournamentError(f"{scale}_{head} labels must contain exactly player")
    try:
        attribution = (directory / "ATTRIBUTION.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TournamentError(f"cannot read {scale}_{head} attribution: {exc}") from exc
    for marker in ("FORT-Cuh", "creativecommons.org/licenses/by/4.0", "AGPL-3.0", "test split"):
        if marker not in attribution:
            raise TournamentError(f"{scale}_{head} attribution is missing {marker!r}")

    parity = manifest.get("parity")
    if not isinstance(parity, Mapping) or parity.get("status") != "passed":
        raise TournamentError(f"{scale}_{head} ONNX/OpenVINO parity did not pass")
    layout = parity.get("output_layout")
    if parity.get("input_shape_nchw") != EXACT_INPUT_SHAPE or (
        (head == "end2end" and layout != "end2end")
        or (head == "traditional" and not str(layout).startswith("traditional_"))
    ):
        raise TournamentError(f"{scale}_{head} parity output contract differs")

    initial_contract_path = directory / "training-initial-run-contract.json"
    initial_contract, _ = _read_json_snapshot(
        initial_contract_path, f"{scale}_{head} packaged initial training contract"
    )
    _read_json_snapshot(
        directory / "training-reproducibility.json",
        f"{scale}_{head} packaged training reproducibility",
    )
    initial_name = (
        str(initial_contract.get("initial_weights", ""))
        .replace("\\", "/")
        .rsplit("/", 1)[-1]
    )
    if (
        initial_name != f"yolo26{scale}.pt"
        or initial_contract.get("initial_weights_sha256") != expected_initial_weights_sha256
    ):
        raise TournamentError(f"{scale}_{head} initial model is not pinned YOLO26{scale}")
    try:
        packaged_training_validator(
            directory,
            manifest=manifest,
            artifacts=validated_artifacts,
        )
    except CandidateExportError as exc:
        raise TournamentError(f"{scale}_{head} packaged training audit failed: {exc}") from exc

    onnx = validated_artifacts["onnx"]
    training_identity = {
        key: provenance.get(key)
        for key in (
            "initial_run_contract_sha256",
            "training_reproducibility_sha256",
            "training_results_sha256",
            "checkpoint_sha256",
            "initial_weights_sha256",
            "dataset_manifest_sha256",
            "dataset_content_sha256",
            "completed_epochs",
            "results_rows",
        )
    }
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "content_sha256": content_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "initial_weights_sha256": expected_initial_weights_sha256,
        "training_identity": training_identity,
        "onnx_name": onnx["name"],
        "onnx_sha256": onnx["sha256"],
        "onnx_bytes": onnx["bytes"],
    }, snapshots


def _validate_report(
    *,
    scale: str,
    head: str,
    path: Path,
    candidate: Mapping[str, Any],
    expected_dataset: Mapping[str, Any],
    runtime_plan: Mapping[str, Any],
    expected_source_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    report, report_sha256 = _read_json_snapshot(path, f"{scale}_{head} runtime report")
    _reject_absolute_path_strings(report, f"{scale}_{head} runtime report")
    if report.get("schema_version") != 4:
        raise TournamentError(f"{scale}_{head} runtime report schema is unsupported")
    evaluator = report.get("evaluator")
    if (
        not isinstance(evaluator, Mapping)
        or evaluator.get("path") != "evaluate_fort_runtime_model.py"
        or evaluator.get("sha256")
        != expected_source_identity["evaluator"]["sha256"]
        or evaluator.get("pipeline_source_sha256")
        != expected_source_identity["pipeline"]
    ):
        raise TournamentError(f"{scale}_{head} report uses a different evaluator revision")
    for name, digest in evaluator["pipeline_source_sha256"].items():
        if not isinstance(name, str):
            raise TournamentError(f"{scale}_{head} report pipeline identity is invalid")
        _sha256_value(digest, f"{scale}_{head} report pipeline source hash")

    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise TournamentError(f"{scale}_{head} report configuration is missing")
    if configuration.get("split") != "val" or configuration.get(
        "selection_role"
    ) != "development_validation":
        raise TournamentError(
            f"{scale}_{head} must use validation evidence; test reports are rejected"
        )
    if (
        configuration.get("backend") != "onnxruntime"
        or configuration.get("device") != runtime_plan.get("device")
        or configuration.get("require_full_provider")
        is not runtime_plan.get("require_full_provider")
        or configuration.get("inference_size") != EXACT_INFERENCE_SIZE
        or configuration.get("input_shape_nchw") != EXACT_INPUT_SHAPE
        or configuration.get("declared_static_input_shape_nchw") != EXACT_INPUT_SHAPE
        or configuration.get("exact_static_deployment_shape") is not True
        or configuration.get("output_format") != head
    ):
        raise TournamentError(f"{scale}_{head} runtime configuration differs from plan")
    confidences = configuration.get("reported_confidence_thresholds")
    if isinstance(confidences, (str, bytes)) or not isinstance(confidences, Sequence):
        raise TournamentError(f"{scale}_{head} reported confidences are invalid")
    try:
        normalized_confidences = [float(value) for value in confidences]
    except (TypeError, ValueError) as exc:
        raise TournamentError(f"{scale}_{head} reported confidences are invalid") from exc
    if (
        any(isinstance(value, bool) for value in confidences)
        or any(not math.isfinite(value) for value in normalized_confidences)
        or normalized_confidences.count(ADVANCEMENT_CONFIDENCE) != 1
    ):
        raise TournamentError(f"{scale}_{head} report lacks one exact 0.25 operating point")
    detail = configuration.get("detail_pass")
    if (
        not isinstance(detail, Mapping)
        or detail.get("enabled") is not True
        or detail.get("requested_crop_size_source_pixels")
        != runtime_plan.get("detail_crop_size_source_pixels")
        or detail.get("selection_split_policy") != "validation_only"
        or detail.get("test_split_evaluation_permitted") is not False
    ):
        raise TournamentError(f"{scale}_{head} report lacks the planned detail A/B")

    dataset = report.get("dataset")
    if not isinstance(dataset, Mapping) or any(
        dataset.get(key) != expected_dataset.get(key)
        for key in ("manifest_sha256", "content_sha256")
    ):
        raise TournamentError(f"{scale}_{head} report dataset differs from plan")
    if (
        dataset.get("yaml") != "fort_cuh_grouped.yaml"
        or dataset.get("manifest") != "manifest.json"
    ):
        raise TournamentError(
            f"{scale}_{head} report contains non-portable dataset paths"
        )
    qualification = report.get("qualification")
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("status") != "development_evidence_only"
        or qualification.get("independent_holdout_required") is not True
    ):
        raise TournamentError(f"{scale}_{head} report overstates qualification")

    artifact = report.get("model_artifact")
    if not isinstance(artifact, Mapping) or artifact.get("backend") != "onnxruntime":
        raise TournamentError(f"{scale}_{head} report model artifact is invalid")
    if artifact.get("entrypoint") != candidate["onnx_name"]:
        raise TournamentError(
            f"{scale}_{head} report contains a non-portable model entrypoint"
        )
    members = artifact.get("members")
    if isinstance(members, (str, bytes)) or not isinstance(members, Sequence) or len(members) != 1:
        raise TournamentError(f"{scale}_{head} report must bind one ONNX artifact")
    member = members[0]
    if not isinstance(member, Mapping):
        raise TournamentError(f"{scale}_{head} report ONNX member is invalid")
    if member.get("path") != candidate["onnx_name"]:
        raise TournamentError(
            f"{scale}_{head} report contains a non-portable model member path"
        )
    expected_member = {
        "name": candidate["onnx_name"],
        "bytes": candidate["onnx_bytes"],
        "sha256": candidate["onnx_sha256"],
    }
    if any(member.get(key) != value for key, value in expected_member.items()) or artifact.get(
        "entrypoint_sha256"
    ) != candidate["onnx_sha256"]:
        raise TournamentError(f"{scale}_{head} report is not bound to its candidate ONNX")
    expected_model_content = _canonical_hash(
        [{"name": candidate["onnx_name"], "sha256": candidate["onnx_sha256"]}]
    )
    if artifact.get("content_sha256") != expected_model_content:
        raise TournamentError(f"{scale}_{head} report model content hash differs")

    runtime = report.get("runtime")
    summary = runtime.get("summary") if isinstance(runtime, Mapping) else None
    if not isinstance(summary, Mapping) or summary.get("output_format") != head:
        raise TournamentError(f"{scale}_{head} runtime decoder identity is invalid")
    if summary.get("model_path", candidate["onnx_name"]) != candidate["onnx_name"]:
        raise TournamentError(
            f"{scale}_{head} runtime summary contains a non-portable model path"
        )
    output_layout = summary.get("output_layout")
    if (head == "end2end" and output_layout != "end2end") or (
        head == "traditional" and not str(output_layout).startswith("traditional_")
    ):
        raise TournamentError(f"{scale}_{head} runtime output layout differs")
    if (
        summary.get("requested_device_input") != runtime_plan.get("device")
        or summary.get("require_full_provider")
        is not runtime_plan.get("require_full_provider")
    ):
        raise TournamentError(f"{scale}_{head} runtime provider request differs")
    requested_provider = summary.get("requested_provider")
    active_providers = summary.get("active_providers")
    if (
        not isinstance(requested_provider, str)
        or isinstance(active_providers, (str, bytes))
        or not isinstance(active_providers, Sequence)
        or requested_provider not in active_providers
    ):
        raise TournamentError(
            f"{scale}_{head} report does not prove the requested provider was active"
        )
    if runtime_plan.get("require_full_provider") is True:
        session_options = summary.get("configured_session_options")
        if (
            not isinstance(session_options, Mapping)
            or session_options.get("disable_cpu_ep_fallback") is not True
            or summary.get("runtime_ep_fail_fallback_disabled") is not True
        ):
            raise TournamentError(
                f"{scale}_{head} report lacks full-provider fallback-disable proof"
            )

    metrics = report.get("metrics")
    selected = metrics.get("val") if isinstance(metrics, Mapping) else None
    if (
        not isinstance(selected, Mapping)
        or selected.get("configured_pipeline") != "full_frame_plus_center_detail_merged"
        or not isinstance(selected.get("primary_full_frame_reference"), Mapping)
    ):
        raise TournamentError(f"{scale}_{head} report lacks paired primary/detail metrics")
    return {
        "report": report,
        "report_sha256": report_sha256,
        "model_content_sha256": expected_model_content,
    }, report_sha256


def _validate_comparison(
    *,
    name: str,
    record: Mapping[str, Any],
    comparison_path: Path,
    comparator_sha256: str,
    baseline: Slot,
    candidate: Slot,
    baseline_pipeline: str,
    candidate_pipeline: str,
) -> ComparisonOutcome:
    comparison_path = _regular_file(
        comparison_path, f"{name} comparison", ".json"
    )
    if {child.name for child in comparison_path.parent.iterdir()} != {
        "comparison.json"
    } or any(
        child.is_symlink() or not child.is_file()
        for child in comparison_path.parent.iterdir()
    ):
        raise TournamentError(f"{name} comparison directory has unexpected members")
    published, comparison_sha256 = _read_json_snapshot(
        comparison_path, f"{name} comparison"
    )
    if published != record:
        raise TournamentError(f"{name} comparator return value differs from published evidence")
    if record.get("schema_version") != 1 or record.get(
        "status"
    ) != "development_selection_evidence_only":
        raise TournamentError(f"{name} comparator status is invalid")
    comparator = record.get("comparator")
    if (
        not isinstance(comparator, Mapping)
        or comparator.get("path") != "compare_fort_runtime_evaluations.py"
        or comparator.get("sha256") != comparator_sha256
    ):
        raise TournamentError(f"{name} comparator source identity differs")
    reports = record.get("reports")
    if not isinstance(reports, Mapping):
        raise TournamentError(f"{name} comparison report bindings are missing")
    expected_reports = {
        "baseline": {
            "metrics_sha256": baseline.report_sha256,
            "model_content_sha256": baseline.report_model_content_sha256,
            "pipeline": baseline_pipeline,
        },
        "candidate": {
            "metrics_sha256": candidate.report_sha256,
            "model_content_sha256": candidate.report_model_content_sha256,
            "pipeline": candidate_pipeline,
        },
    }
    if reports != expected_reports:
        raise TournamentError(f"{name} comparison inputs differ from the bracket")
    paired = record.get("paired_contract")
    if not isinstance(paired, Mapping) or paired.get("confidence") != ADVANCEMENT_CONFIDENCE:
        raise TournamentError(f"{name} comparison operating point is not exactly 0.25")
    policy = record.get("development_advancement_policy")
    checks = policy.get("checks") if isinstance(policy, Mapping) else None
    if (
        not isinstance(checks, Mapping)
        or any(not isinstance(value, bool) for value in checks.values())
        or policy.get("required_bootstrap_samples") != BOOTSTRAP_SAMPLES
        or policy.get("required_advancement_confidence") != ADVANCEMENT_CONFIDENCE
        or policy.get("passed") is not all(checks.values())
    ):
        raise TournamentError(f"{name} advancement policy is invalid")
    required_policy_checks = REQUIRED_EVIDENCE_CHECKS | OUTCOME_CHECKS
    missing_checks = required_policy_checks - set(checks)
    if missing_checks:
        raise TournamentError(
            f"{name} comparator omitted policy checks: {sorted(missing_checks)}"
        )
    evidence_checks = set(checks) - OUTCOME_CHECKS
    failed_evidence = sorted(key for key in evidence_checks if checks[key] is not True)
    if failed_evidence:
        raise TournamentError(
            f"{name} has inadequate development evidence: {failed_evidence}"
        )
    comparison = record.get("comparison")
    if not isinstance(comparison, Mapping):
        raise TournamentError(f"{name} paired comparison metrics are missing")
    metric_groups = [comparison.get("aggregate")]
    buckets = comparison.get("buckets")
    if not isinstance(buckets, Mapping):
        raise TournamentError(f"{name} bucket comparisons are missing")
    metric_groups.extend(buckets.values())
    for metric in metric_groups:
        bootstrap = (
            metric.get("paired_image_bootstrap_95_ci")
            if isinstance(metric, Mapping)
            else None
        )
        if not isinstance(bootstrap, Mapping) or bootstrap.get(
            "samples_requested"
        ) != BOOTSTRAP_SAMPLES:
            raise TournamentError(f"{name} does not contain exact 2,000-sample evidence")
    far = buckets.get("far_33_to_64px")
    source_group_bootstrap = (
        far.get("paired_source_group_bootstrap_95_ci")
        if isinstance(far, Mapping)
        else None
    )
    if not isinstance(source_group_bootstrap, Mapping) or source_group_bootstrap.get(
        "samples_requested"
    ) != BOOTSTRAP_SAMPLES:
        raise TournamentError(
            f"{name} does not contain exact 2,000-sample source-group evidence"
        )
    release = record.get("release_qualification")
    if (
        not isinstance(release, Mapping)
        or release.get("qualified") is not False
        or release.get("independent_holdout_required") is not True
        or release.get("reviewed_negative_scenes_required") is not True
        or release.get("exact_frozen_target_gpu_latency_required") is not True
    ):
        raise TournamentError(f"{name} comparison overstates release qualification")
    return ComparisonOutcome(
        name=name,
        passed=bool(policy["passed"]),
        record=record,
        relative_path=f"comparisons/{name}/comparison.json",
        sha256=comparison_sha256,
    )


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise TournamentError("atomic no-replace publication is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        ) != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise TournamentError(f"output appeared during tournament: {destination}")
            raise TournamentError(f"atomic tournament publication failed: {os.strerror(error)}")
        return
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise TournamentError(f"output appeared during tournament: {destination}") from exc
        return
    raise TournamentError("atomic tournament publication is supported only on Linux and Windows")


def _snapshot_unchanged(snapshots: Mapping[Path, str]) -> None:
    for path, expected in snapshots.items():
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
            raise TournamentError(f"tournament input changed before publication: {path}")


def _copy_sealed_input(
    source: Path,
    target: Path,
    *,
    expected_sha256: str,
    staging: Path,
) -> dict[str, object]:
    """Copy one already validated input into private staging with one hash pass."""

    if target == staging or staging not in target.parents:
        raise TournamentError("sealed tournament input target escapes staging")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    size = 0
    try:
        with source.open("rb") as reader, target.open("xb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
    except OSError as exc:
        raise TournamentError(f"cannot seal tournament input {source}: {exc}") from exc
    if size <= 0 or digest.hexdigest() != expected_sha256:
        raise TournamentError(f"tournament input changed while it was sealed: {source}")
    return {
        "path": target.relative_to(staging).as_posix(),
        "bytes": size,
        "sha256": expected_sha256,
    }


def _verify_sealed_input_records(
    staging: Path,
    records: Mapping[str, Any],
) -> None:
    """Recheck every staged evidence byte immediately before publication."""

    def verify(record: object, description: str) -> None:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise TournamentError(f"{description} sealed-input record is invalid")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise TournamentError(f"{description} sealed-input path is invalid")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise TournamentError(f"{description} sealed-input path is unsafe")
        path = staging.joinpath(*pure.parts)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("bytes")
            or _sha256_file(path) != record.get("sha256")
        ):
            raise TournamentError(f"{description} sealed input changed in staging")

    verify(records.get("plan"), "plan")
    runtime_reports = records.get("runtime_reports")
    training_results = records.get("training_results")
    if (
        not isinstance(runtime_reports, Mapping)
        or set(runtime_reports) != {f"{scale}_{head}" for scale in SCALES for head in HEADS}
        or not isinstance(training_results, Mapping)
        or set(training_results) != set(SCALES)
    ):
        raise TournamentError("sealed tournament input inventory is incomplete")
    for name, record in runtime_reports.items():
        verify(record, f"runtime report {name}")
    for scale, record in training_results.items():
        verify(record, f"training results {scale}")


def run_tournament(
    *,
    plan_path: Path,
    output: Path,
    comparator: Callable[..., Mapping[str, Any]] = compare_reports,
    packaged_training_validator: Callable[..., None] = _validate_packaged_training_provenance,
) -> dict[str, Any]:
    """Validate a fixed four-slot bracket and publish its development winner."""

    source = Path(__file__).resolve()
    source_sha256 = _sha256_file(source)
    comparator_source = PROJECT_ROOT / "scripts" / "compare_fort_runtime_evaluations.py"
    comparator_sha256 = _sha256_file(comparator_source)
    exporter_source = PROJECT_ROOT / "scripts" / "export_fort_release_candidate.py"
    exporter_sha256 = _sha256_file(exporter_source)
    public_evidence_source = PROJECT_ROOT / "utils" / "public_evidence.py"
    public_evidence_sha256 = _sha256_file(public_evidence_source)
    runtime_source_identity = _source_hash_snapshot("onnxruntime")
    plan, plan_sha256, paths, resolved_plan_path = _validate_plan(plan_path)
    destination = output.expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    destination = destination.absolute()
    _reject_symlink_components(destination.parent, "tournament output")
    if os.path.lexists(destination):
        raise TournamentError(f"output already exists; refusing overwrite: {destination}")

    input_snapshots: dict[Path, str] = {resolved_plan_path: plan_sha256}
    slots: dict[tuple[str, str], Slot] = {}
    onnx_hashes: set[str] = set()
    runtime_plan = plan["runtime"]
    dataset_plan = plan["dataset"]
    models_plan = plan["models"]
    assert isinstance(runtime_plan, Mapping)
    assert isinstance(dataset_plan, Mapping)
    assert isinstance(models_plan, Mapping)
    for scale in SCALES:
        scale_plan = models_plan[scale]
        assert isinstance(scale_plan, Mapping)
        for head in HEADS:
            candidate_path, report_path = paths[(scale, head)]
            for protected in (candidate_path, report_path.parent):
                if destination == protected or destination.is_relative_to(protected):
                    raise TournamentError("tournament output must be outside all evidence inputs")
            candidate, candidate_snapshots = _validate_candidate(
                scale=scale,
                head=head,
                directory=candidate_path,
                expected_initial_weights_sha256=scale_plan["initial_weights_sha256"],
                expected_dataset=dataset_plan,
                expected_exporter_sha256=exporter_sha256,
                packaged_training_validator=packaged_training_validator,
            )
            report, report_sha256 = _validate_report(
                scale=scale,
                head=head,
                path=report_path,
                candidate=candidate,
                expected_dataset=dataset_plan,
                runtime_plan=runtime_plan,
                expected_source_identity=runtime_source_identity,
            )
            input_snapshots.update(candidate_snapshots)
            input_snapshots[report_path] = report_sha256
            if candidate["onnx_sha256"] in onnx_hashes:
                raise TournamentError("all four head/scale ONNX artifacts must be distinct")
            onnx_hashes.add(candidate["onnx_sha256"])
            raw_paths = scale_plan[head]
            assert isinstance(raw_paths, Mapping)
            slots[(scale, head)] = Slot(
                scale=scale,
                head=head,
                candidate_dir=candidate_path,
                report_path=report_path,
                candidate_manifest_sha256=candidate["manifest_sha256"],
                candidate_content_sha256=candidate["content_sha256"],
                checkpoint_sha256=candidate["checkpoint_sha256"],
                initial_weights_sha256=candidate["initial_weights_sha256"],
                training_identity=candidate["training_identity"],
                onnx_name=candidate["onnx_name"],
                onnx_sha256=candidate["onnx_sha256"],
                onnx_bytes=candidate["onnx_bytes"],
                report_sha256=report["report_sha256"],
                report_model_content_sha256=report["model_content_sha256"],
                plan_paths={
                    "candidate_dir": str(raw_paths["candidate_dir"]),
                    "validation_report": str(raw_paths["validation_report"]),
                },
            )
    for scale in SCALES:
        left = slots[(scale, "end2end")]
        right = slots[(scale, "traditional")]
        if (
            left.checkpoint_sha256 != right.checkpoint_sha256
            or left.training_identity != right.training_identity
        ):
            raise TournamentError(
                f"{scale} head exports do not share one completed training run"
            )
    if (
        slots[("n", "end2end")].checkpoint_sha256
        == slots[("s", "end2end")].checkpoint_sha256
    ):
        raise TournamentError("n and s completed checkpoints must be distinct")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tournament-", dir=destination.parent)
    )
    published = False
    comparisons: dict[str, ComparisonOutcome] = {}
    try:
        sealed_inputs: dict[str, Any] = {
            "plan": _copy_sealed_input(
                resolved_plan_path,
                staging / "inputs" / SEALED_PLAN_NAME,
                expected_sha256=plan_sha256,
                staging=staging,
            ),
            "runtime_reports": {},
            "training_results": {},
        }
        for scale in SCALES:
            training_results = (
                slots[(scale, "end2end")].candidate_dir / SEALED_RESULTS_NAME
            )
            expected_results_sha256 = input_snapshots.get(training_results)
            if expected_results_sha256 is None:
                raise TournamentError(
                    f"{scale} training results were not sealed by validation"
                )
            sealed_inputs["training_results"][scale] = _copy_sealed_input(
                training_results,
                staging / "inputs" / "training" / scale / SEALED_RESULTS_NAME,
                expected_sha256=expected_results_sha256,
                staging=staging,
            )
            other_results = (
                slots[(scale, "traditional")].candidate_dir / SEALED_RESULTS_NAME
            )
            if input_snapshots.get(other_results) != expected_results_sha256:
                raise TournamentError(
                    f"{scale} head exports do not carry identical training results"
                )
            for head in HEADS:
                slot = slots[(scale, head)]
                sealed_inputs["runtime_reports"][slot.name] = _copy_sealed_input(
                    slot.report_path,
                    staging / "inputs" / "runtime" / slot.name / SEALED_REPORT_NAME,
                    expected_sha256=slot.report_sha256,
                    staging=staging,
                )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    def run_comparison(
        name: str,
        baseline: Slot,
        challenger: Slot,
        baseline_pipeline: str,
        challenger_pipeline: str,
    ) -> ComparisonOutcome:
        comparison_dir = staging / "comparisons" / name
        try:
            result = comparator(
                baseline=baseline.report_path,
                candidate=challenger.report_path,
                output=comparison_dir,
                confidence=ADVANCEMENT_CONFIDENCE,
                baseline_pipeline=baseline_pipeline,
                candidate_pipeline=challenger_pipeline,
                bootstrap_samples=BOOTSTRAP_SAMPLES,
            )
        except Exception as exc:
            raise TournamentError(f"{name} paired comparison failed: {exc}") from exc
        if not isinstance(result, Mapping):
            raise TournamentError(f"{name} comparator returned no evidence object")
        outcome = _validate_comparison(
            name=name,
            record=result,
            comparison_path=comparison_dir / "comparison.json",
            comparator_sha256=comparator_sha256,
            baseline=baseline,
            candidate=challenger,
            baseline_pipeline=baseline_pipeline,
            candidate_pipeline=challenger_pipeline,
        )
        comparisons[name] = outcome
        return outcome

    try:
        selected_pipelines: dict[tuple[str, str], str] = {}
        detail_decisions: dict[str, Any] = {}
        for scale in SCALES:
            for head in HEADS:
                slot = slots[(scale, head)]
                name = f"{slot.name}_primary_vs_detail"
                outcome = run_comparison(name, slot, slot, "primary", "configured")
                selected_pipeline = "configured" if outcome.passed else "primary"
                selected_pipelines[(scale, head)] = selected_pipeline
                detail_decisions[slot.name] = {
                    "incumbent": "primary",
                    "challenger": "configured",
                    "challenger_advanced": outcome.passed,
                    "selected_pipeline": selected_pipeline,
                    "comparison": outcome.relative_path,
                }

        selected_heads: dict[str, str] = {}
        head_decisions: dict[str, Any] = {}
        for scale in SCALES:
            incumbent = slots[(scale, "end2end")]
            challenger = slots[(scale, "traditional")]
            name = f"{scale}_end2end_vs_traditional"
            outcome = run_comparison(
                name,
                incumbent,
                challenger,
                selected_pipelines[(scale, "end2end")],
                selected_pipelines[(scale, "traditional")],
            )
            selected_head = "traditional" if outcome.passed else "end2end"
            selected_heads[scale] = selected_head
            head_decisions[scale] = {
                "incumbent": "end2end",
                "challenger": "traditional",
                "challenger_advanced": outcome.passed,
                "selected_head": selected_head,
                "comparison": outcome.relative_path,
            }

        n_slot = slots[("n", selected_heads["n"])]
        s_slot = slots[("s", selected_heads["s"])]
        final_outcome = run_comparison(
            "n_vs_s",
            n_slot,
            s_slot,
            selected_pipelines[("n", selected_heads["n"])],
            selected_pipelines[("s", selected_heads["s"])],
        )
        winner_scale = "s" if final_outcome.passed else "n"
        winner_head = selected_heads[winner_scale]
        winner = slots[(winner_scale, winner_head)]
        winner_pipeline = selected_pipelines[(winner_scale, winner_head)]

        candidate_records = {
            slot.name: {
                "scale": slot.scale,
                "head": slot.head,
                "plan_paths": dict(slot.plan_paths),
                "candidate_manifest_sha256": slot.candidate_manifest_sha256,
                "candidate_content_sha256": slot.candidate_content_sha256,
                "checkpoint_sha256": slot.checkpoint_sha256,
                "initial_weights_sha256": slot.initial_weights_sha256,
                "training_identity": dict(slot.training_identity),
                "onnx": {
                    "name": slot.onnx_name,
                    "bytes": slot.onnx_bytes,
                    "sha256": slot.onnx_sha256,
                },
                "validation_report_sha256": slot.report_sha256,
                "runtime_model_content_sha256": slot.report_model_content_sha256,
                "selected_pipeline_after_detail_ab": selected_pipelines[(slot.scale, slot.head)],
            }
            for slot in slots.values()
        }
        comparison_records = {
            name: {
                "path": outcome.relative_path,
                "sha256": outcome.sha256,
                "challenger_advanced": outcome.passed,
            }
            for name, outcome in comparisons.items()
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": OUTPUT_STATUS,
            "orchestrator": {"path": source.name, "sha256": source_sha256},
            "comparator": {
                "path": comparator_source.name,
                "sha256": comparator_sha256,
            },
            "candidate_exporter": {
                "path": exporter_source.name,
                "sha256": exporter_sha256,
            },
            "public_evidence_privacy": {
                "path": public_evidence_source.name,
                "sha256": public_evidence_sha256,
            },
            "runtime_evaluator": {
                "path": "evaluate_fort_runtime_model.py",
                "sha256": runtime_source_identity["evaluator"]["sha256"],
                "pipeline_source_sha256": runtime_source_identity["pipeline"],
            },
            "plan": {
                "path": sealed_inputs["plan"]["path"],
                "sha256": plan_sha256,
                "status": PLAN_STATUS,
                "timing_note": (
                    "The plan hash binds the fixed bracket, but this command does not "
                    "claim the plan predates report generation. This is development data."
                ),
            },
            "sealed_inputs": sealed_inputs,
            "fixed_contract": {
                "scales": list(SCALES),
                "heads": list(HEADS),
                "backend": "onnxruntime",
                "device": runtime_plan["device"],
                "require_full_provider": runtime_plan["require_full_provider"],
                "input_shape_nchw": EXACT_INPUT_SHAPE,
                "inference_size": EXACT_INFERENCE_SIZE,
                "detail_crop_size_source_pixels": runtime_plan[
                    "detail_crop_size_source_pixels"
                ],
                "confidence": ADVANCEMENT_CONFIDENCE,
                "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
                "split": "val",
                "bracket": [
                    "detail challenges primary for every scale/head export",
                    "traditional challenges end2end within each scale",
                    "s challenges n after within-scale selection",
                ],
            },
            "dataset": dict(dataset_plan),
            "candidates": candidate_records,
            "comparisons": comparison_records,
            "development_selection": {
                "detail_decisions": detail_decisions,
                "head_decisions": head_decisions,
                "scale_decision": {
                    "incumbent": "n",
                    "challenger": "s",
                    "challenger_advanced": final_outcome.passed,
                    "selected_scale": winner_scale,
                    "comparison": final_outcome.relative_path,
                },
                "winner": {
                    "slot": winner.name,
                    "scale": winner_scale,
                    "head": winner_head,
                    "pipeline": winner_pipeline,
                    "candidate_content_sha256": winner.candidate_content_sha256,
                    "onnx_sha256": winner.onnx_sha256,
                    "validation_report_sha256": winner.report_sha256,
                },
                "scope": (
                    "Development validation winner only; it may advance to independent "
                    "holdout and physical frozen-build GPU qualification."
                ),
            },
            "test_data_policy": {
                "test_split_consumed": False,
                "test_reports_accepted": False,
                "selection_split": "val",
            },
            "release_qualification": {
                "qualified": False,
                "release_model_approved": False,
                "independent_holdout_required": True,
                "reviewed_negative_scenes_required": True,
                "physical_target_gpu_latency_required": True,
                "frozen_build_qualification_required": True,
                "reason": (
                    "The tournament consumes development validation only. No bracket "
                    "outcome, including unanimous paired gains, can qualify a release."
                ),
            },
        }
        manifest["selection_content_sha256"] = _selection_content_hash(manifest)

        _verify_sealed_input_records(staging, sealed_inputs)
        if _sha256_file(source) != source_sha256 or _sha256_file(
            comparator_source
        ) != comparator_sha256:
            raise TournamentError("orchestrator or comparator source changed during tournament")
        if (
            _sha256_file(exporter_source) != exporter_sha256
            or _sha256_file(public_evidence_source) != public_evidence_sha256
            or _source_hash_snapshot("onnxruntime") != runtime_source_identity
        ):
            raise TournamentError(
                "candidate exporter, public-evidence, evaluator, or runtime source "
                "changed during tournament"
            )
        _snapshot_unchanged(input_snapshots)
        for outcome in comparisons.values():
            comparison_path = staging / outcome.relative_path
            if _sha256_file(comparison_path) != outcome.sha256:
                raise TournamentError("comparison evidence changed before tournament publication")
        manifest_path = staging / "selection-manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_canonical_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(destination):
            raise TournamentError(f"output appeared during tournament: {destination}")
        _rename_directory_noreplace(staging, destination)
        published = True
        return manifest
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_tournament(plan_path=args.plan, output=args.output)
    except TournamentError as exc:
        print(f"Model tournament failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["development_selection"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
