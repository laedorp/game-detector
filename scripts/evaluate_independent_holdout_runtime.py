#!/usr/bin/env python3
"""Evaluate one adopted ProAim candidate on one sealed independent holdout.

This is the final-evidence counterpart to ``evaluate_fort_runtime_model.py``.
It deliberately does not accept a grouped-dataset YAML or a val/test split.
Ground truth comes directly from a verified sealed HOLDOUT-MANIFEST/COCO
package.  The package is opened only inside a lock-held, one-time transaction;
successful atomic evidence publication is followed by an evidence-hash-bound
consumption event and immediate retirement.

The resulting record can support manual release review.  It never approves a
release and never marks hardware, frozen-build, legal, or release gates passed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from hashlib import sha256
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import sys
import tempfile
from time import perf_counter_ns
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.detail_pass import (  # noqa: E402
    DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT,
    DetailPassStats,
    merge_cross_pass_detections,
    plan_detail_pass,
)
from scripts.evaluate_fort_model import (  # noqa: E402
    BUCKET_IOU_THRESHOLD,
    MINIMUM_PREDICTION_CONFIDENCE,
    REFERENCE_FRAME_HEIGHT,
    SIZE_BUCKETS,
    _percentile,
    _pr_summary,
    bucket_image_evidence,
)
from scripts.evaluate_fort_runtime_model import (  # noqa: E402
    _artifact_members,
    _create_detector,
    _json_safe,
    _metric_record,
    _prediction_boxes,
    _publish_record,
    _rename_directory_noreplace,
    _source_hash_snapshot,
    _timing_summary,
    _validate_detail_preprocessing,
    _validate_runtime_binding,
    _validated_static_shape,
    paired_image_operating_point_evidence,
)
from scripts.prepare_independent_player_holdout import (  # noqa: E402
    DATASET_KIND,
    GATING_COUNT_KEYS,
    MANIFEST_NAME,
    MAX_IMAGE_ENCODED_BYTES,
    PINNED_RELEASE_MINIMUMS,
    POOL_SEALED,
    HoldoutContractError,
    _canonical_json_bytes as _canonical_holdout_json_bytes,
    _load_json as _load_holdout_json,
    _load_package_manifest,
    _ledger_events,
    _read_regular_bytes,
    _fsync_directory as _fsync_holdout_directory,
    complete_sealed_evaluation,
    verify_holdout,
)
from scripts.write_dependency_manifest import (  # noqa: E402
    LOCK_PROFILES,
    DependencyContractError,
    load_profile_lock,
    runtime_identity,
    validate_runtime,
    verify_installed_set,
)
from utils.inference_size import (  # noqa: E402
    InferenceSize,
    format_inference_size,
    parse_inference_size,
    validate_yolo_inference_size,
)
from utils.independent_holdout_release_contract import (  # noqa: E402
    BUNDLE_KIND,
    BUNDLE_MANIFEST_NAME,
    BUNDLE_MEMBER_NAMES,
    CANONICAL_RELEASE_DECISION_RULE,
    DECISION_RESULT_SCOPE,
    DEFAULT_WARMUP,
    EVIDENCE_KIND,
    EVIDENCE_SCHEMA_VERSION,
    MINIMUM_CAPTURE_SESSIONS,
    MINIMUM_REVIEWED_NEGATIVE_CAPTURE_SESSIONS,
    MINIMUM_TARGET_BUCKET_CAPTURE_SESSIONS,
    PLAN_KIND,
    PLAN_SCHEMA_VERSION,
    PLAN_STATUS,
    RECEIPT_KIND,
    RECEIPT_NAME,
    RECEIPT_SCHEMA_VERSION,
    RELEASE_POLICY_REVIEW_NOTE,
    IndependentHoldoutReleaseContractError,
    receipt_verifier_record as _shared_receipt_verifier_record,
    release_environment_policy_record as _shared_release_environment_policy_record,
    release_environment_record as _shared_release_environment_record,
    release_policy_record as _shared_release_policy_record,
    source_snapshot as _shared_source_snapshot,
    validate_holdout_hardware_identity as _shared_validate_holdout_hardware_identity,
    validate_release_environment_record as _shared_validate_release_environment_record,
)
from utils.preprocess import preprocess_frame  # noqa: E402
from utils.public_evidence import contains_nonportable_path  # noqa: E402
from utils.release_model_contract import (  # noqa: E402
    CONTRACT_RELATIVE,
    QUALIFICATION_RECORD,
    TOURNAMENT_COMPARISON_NAMES,
    TOURNAMENT_SEALED_INPUT_ROLES,
    TOURNAMENT_SLOT_NAMES,
    canonical_hash,
    canonical_json_bytes,
    load_release_default_contract,
)


EVALUATION_PURPOSE = (
    "execute the exact frozen ProAim independent runtime evaluation plan"
)
DEFAULT_CONFIDENCE_THRESHOLDS = (0.25, 0.45)
DEFAULT_NMS_IOU = 0.45
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
SUPPORTED_ACCELERATOR_PROVIDERS = frozenset(
    {
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "DmlExecutionProvider",
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
    }
)
PUBLIC_CANDIDATE_KEYS = (
    "pointer_sha256",
    "pointer_content_sha256",
    "input_shape_nchw",
    "candidate_content_sha256",
    "candidate_manifest_sha256",
    "checkpoint_sha256",
    "dataset_manifest_sha256",
    "dataset_content_sha256",
    "adoption_sha256",
    "adoption_content_sha256",
    "adoption_evidence_replay_sha256",
    "tournament_selection_sha256",
    "tournament_selection_content_sha256",
    "tournament_evidence",
    "candidate_provenance_evidence",
    "candidate_evaluation_sha256",
    "winner_slot",
    "model_artifacts",
    "model_content_sha256",
    "labels",
    "selected_pipeline",
    "selected_backend",
    "output_head",
    "detail_crop_size_source_pixels",
    "exporter_sha256",
    "adoption_source_sha256",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _release_policy_record() -> dict[str, Any]:
    return _shared_release_policy_record()


class IndependentRuntimeEvaluationError(ValueError):
    """Raised when final independent evidence would be incomplete or unsafe."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise IndependentRuntimeEvaluationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _snapshot_exact_regular_file(
    source: Path,
    target: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    description: str,
) -> Path:
    """Copy one bound artifact through no-follow descriptors into private staging."""

    source_path = _regular_file(source, description)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source_path, flags)
    except OSError as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot open {description} for private snapshot: {exc}"
        ) from exc
    try:
        target_descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        os.close(source_descriptor)
        raise IndependentRuntimeEvaluationError(
            f"cannot create private {description} snapshot: {exc}"
        ) from exc
    digest = sha256()
    size = 0
    try:
        source_details = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_details.st_mode):
            raise IndependentRuntimeEvaluationError(
                f"{description} changed to a non-regular file"
            )
        with (
            os.fdopen(source_descriptor, "rb", closefd=False) as reader,
            os.fdopen(target_descriptor, "wb", closefd=False) as writer,
        ):
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
    finally:
        os.close(source_descriptor)
        os.close(target_descriptor)
    if size != expected_bytes or digest.hexdigest() != expected_sha256:
        raise IndependentRuntimeEvaluationError(
            f"{description} bytes differ while creating the private snapshot"
        )
    return _regular_file(target, f"private {description} snapshot")


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise IndependentRuntimeEvaluationError(
            f"{description} must be a lowercase SHA-256"
        )
    return value


def _require_int(value: object, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IndependentRuntimeEvaluationError(
            f"{description} must be an integer >= {minimum}"
        )
    return value


def _probability(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise IndependentRuntimeEvaluationError(
            f"{description} must be a finite probability"
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise IndependentRuntimeEvaluationError(
            f"{description} must be a finite probability"
        )
    return result


def _positive_float(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise IndependentRuntimeEvaluationError(
            f"{description} must be a finite positive number"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise IndependentRuntimeEvaluationError(
            f"{description} must be a finite positive number"
        )
    return result


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise IndependentRuntimeEvaluationError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise IndependentRuntimeEvaluationError(f"non-finite JSON constant: {value}")


def _regular_file(path: Path, description: str) -> Path:
    expanded = path.expanduser().absolute()
    current = Path(expanded.anchor)
    for part in expanded.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise IndependentRuntimeEvaluationError(
                f"{description} path contains a symlink: {current}"
            )
    try:
        details = expanded.stat(follow_symlinks=False)
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise IndependentRuntimeEvaluationError(
            f"missing or unreadable {description}: {expanded}: {exc}"
        ) from exc
    if not stat.S_ISREG(details.st_mode) or details.st_size <= 0:
        raise IndependentRuntimeEvaluationError(
            f"{description} must be a non-empty regular file: {expanded}"
        )
    return resolved


def _strict_json_file(path: Path, description: str) -> tuple[Path, dict[str, Any], bytes]:
    source = _regular_file(path, description)
    try:
        payload = source.read_bytes()
        value = _parse_json_object_payload(payload, description)
    except IndependentRuntimeEvaluationError:
        raise
    except OSError as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot parse {description}: {exc}"
        ) from exc
    if payload != canonical_json_bytes(value):
        raise IndependentRuntimeEvaluationError(
            f"{description} must be canonical sorted JSON with one trailing newline"
        )
    return source, value, payload


def _parse_json_object_payload(payload: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except IndependentRuntimeEvaluationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot parse {description}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise IndependentRuntimeEvaluationError(f"{description} must be one JSON object")
    return value


def _load_release_environment(
    dependency_manifest: Path,
    *,
    project_root: Path,
) -> tuple[Path, bytes, dict[str, Any]]:
    """Read once and reduce an exact locked environment manifest to public evidence."""

    source = _regular_file(
        dependency_manifest, "Windows DirectML dependency manifest"
    )
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot read Windows DirectML dependency manifest: {exc}"
        ) from exc
    manifest = _parse_json_object_payload(
        payload, "Windows DirectML dependency manifest"
    )
    try:
        _validate_current_release_environment(manifest, project_root=project_root)
        record = _shared_release_environment_record(
            manifest,
            dependency_manifest_sha256=sha256(payload).hexdigest(),
            project_root=project_root,
        )
        return (
            source,
            payload,
            _shared_validate_release_environment_record(
                record, project_root=project_root
            ),
        )
    except IndependentHoldoutReleaseContractError as exc:
        raise IndependentRuntimeEvaluationError(
            f"Windows DirectML dependency environment is not release-exact: {exc}"
        ) from exc


def _validate_current_release_environment(
    manifest: Mapping[str, Any], *, project_root: Path
) -> None:
    """Re-hash the executing interpreter and every installed locked payload."""

    try:
        python_record = manifest.get("python")
        if not isinstance(python_record, Mapping):
            raise IndependentRuntimeEvaluationError(
                "dependency manifest Python record is missing"
            )
        executable = Path(sys.executable).resolve(strict=True)
        if _sha256_file(executable) != python_record.get("executable_sha256"):
            raise IndependentRuntimeEvaluationError(
                "the executing Python differs from the dependency manifest"
            )
        profile = LOCK_PROFILES["windows-directml-py313"]
        validate_runtime(profile, runtime_identity())
        locked = load_profile_lock(project_root, profile)
        _, installed = verify_installed_set(
            locked,
            profile.runtime_distribution,
            environment_root=Path(sys.prefix).resolve(),
            verify_installed_files=True,
        )
    except IndependentRuntimeEvaluationError:
        raise
    except (DependencyContractError, KeyError, OSError) as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot verify the executing locked runtime environment: {exc}"
        ) from exc

    expected_distributions = manifest.get("distributions")
    if not isinstance(expected_distributions, list):
        raise IndependentRuntimeEvaluationError(
            "dependency manifest distribution inventory is missing"
        )
    expected_by_name = {
        str(record.get("canonical_name")): record
        for record in expected_distributions
        if isinstance(record, Mapping)
    }
    actual_by_name = {
        str(record["canonical_name"]): record for record in installed
    }
    if set(expected_by_name) != set(actual_by_name):
        raise IndependentRuntimeEvaluationError(
            "executing distribution set differs from the dependency manifest"
        )
    for name, actual in actual_by_name.items():
        expected = expected_by_name[name]
        if any(
            actual.get(key) != expected.get(key)
            for key in (
                "canonical_name",
                "installed_files",
                "installed_metadata_sha256",
                "installed_record_sha256",
                "name",
                "version",
            )
        ):
            raise IndependentRuntimeEvaluationError(
                f"executing installed payload differs from the manifest: {name}"
            )


def _load_holdout_hardware_identity(
    hardware_identity: Path,
) -> tuple[Path, bytes, dict[str, Any]]:
    """Load the canonical redacted RX 6950 XT identity produced pre-access."""

    source, value, payload = _strict_json_file(
        hardware_identity, "RX 6950 XT holdout hardware identity"
    )
    try:
        return source, payload, _shared_validate_holdout_hardware_identity(value)
    except IndependentHoldoutReleaseContractError as exc:
        raise IndependentRuntimeEvaluationError(
            f"holdout hardware identity is invalid: {exc}"
        ) from exc


def _reject_private_path_strings(value: Any, description: str) -> None:
    """Reject local/absolute paths from copied public tournament JSON."""

    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IndependentRuntimeEvaluationError(
                f"{description} contains a non-finite number"
            )
        return
    if isinstance(value, str):
        normalized = value.strip()
        if (
            not normalized
            or any(ord(character) < 32 for character in value)
            or contains_nonportable_path(value)
        ):
            raise IndependentRuntimeEvaluationError(
                f"{description} contains a private or absolute path"
            )
        return
    if isinstance(value, Mapping):
        for key, member in value.items():
            if (
                not isinstance(key, str)
                or not key
                or any(ord(character) < 32 for character in key)
                or contains_nonportable_path(key)
            ):
                raise IndependentRuntimeEvaluationError(
                    f"{description} contains an unsafe field name"
                )
            _reject_private_path_strings(member, f"{description}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, member in enumerate(value):
            _reject_private_path_strings(member, f"{description}[{index}]")
        return
    raise IndependentRuntimeEvaluationError(
        f"{description} contains unsupported public evidence data"
    )


def _assert_public_text_safe(path: Path, description: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot inspect {description}: {exc}"
        ) from exc
    if (
        any(ord(character) < 32 and character not in "\t\r\n" for character in text)
        or contains_nonportable_path(text)
    ):
        raise IndependentRuntimeEvaluationError(
            f"{description} contains a private or absolute path"
        )


def _source_snapshot(backend: str) -> dict[str, Any]:
    if backend != "onnxruntime":
        raise IndependentRuntimeEvaluationError(
            "independent final evidence currently supports ONNX Runtime only"
        )
    try:
        return _shared_source_snapshot(PROJECT_ROOT)
    except Exception as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot construct canonical evaluator/source snapshot: {exc}"
        ) from exc


def _artifact_content(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(
        [
            {"name": record["name"], "sha256": record["sha256"]}
            for record in records
        ]
    )


def _canonical_relative(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise IndependentRuntimeEvaluationError(f"invalid {description}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise IndependentRuntimeEvaluationError(f"unsafe {description}")
    return value


def _adoption_body_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("content_sha256", None)
    return canonical_hash(body)


def _tournament_selection_body_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("selection_content_sha256", None)
    return canonical_hash(body)


def _receipt_body_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("content_sha256", None)
    return canonical_hash(body)


def _validate_adoption_evidence_replay(
    *,
    replay: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_artifacts: Mapping[str, Any],
    tournament_candidates: Mapping[str, Any],
    tournament_comparisons: Mapping[str, Any],
    tournament_plan: Mapping[str, Any],
) -> str:
    """Bind the adoption's deterministic tournament/training replay proof."""

    if (
        set(replay)
        != {
            "schema_version",
            "status",
            "plan",
            "comparison",
            "winner_training_results",
            "qualification",
        }
        or replay.get("schema_version") != 1
        or replay.get("status")
        != (
            "sealed_plan_comparisons_and_winner_training_replayed_"
            "not_release_qualified"
        )
        or replay.get("qualification") != QUALIFICATION_RECORD
    ):
        raise IndependentRuntimeEvaluationError(
            "adoption evidence replay schema/status/qualification is invalid"
        )
    plan = replay.get("plan")
    comparison = replay.get("comparison")
    training = replay.get("winner_training_results")
    if not all(isinstance(item, Mapping) for item in (plan, comparison, training)):
        raise IndependentRuntimeEvaluationError(
            "adoption evidence replay records are incomplete"
        )
    assert isinstance(plan, Mapping)
    assert isinstance(comparison, Mapping)
    assert isinstance(training, Mapping)
    plan_artifact = artifacts.get("tournament_plan")
    if (
        not isinstance(plan_artifact, Mapping)
        or set(plan)
        != {
            "status",
            "sha256",
            "canonical_bytes",
            "dataset_matches",
            "runtime_matches",
            "all_slot_paths_and_initial_weights_match",
        }
        or plan.get("status") != "sealed_tournament_plan_replayed"
        or plan.get("sha256") != plan_artifact.get("sha256")
        or plan.get("sha256") != tournament_plan.get("sha256")
        or plan.get("canonical_bytes") != plan_artifact.get("bytes")
        or plan.get("dataset_matches") is not True
        or plan.get("runtime_matches") is not True
        or plan.get("all_slot_paths_and_initial_weights_match") is not True
    ):
        raise IndependentRuntimeEvaluationError(
            "adoption sealed tournament plan replay proof is invalid"
        )

    records = comparison.get("records")
    if (
        set(comparison)
        != {
            "comparator_sha256",
            "confidence",
            "bootstrap_samples",
            "records",
            "winner_slot",
            "winner_pipeline",
        }
        or comparison.get("comparator_sha256")
        != _sha256_file(
            PROJECT_ROOT / "scripts" / "compare_fort_runtime_evaluations.py"
        )
        or comparison.get("confidence") != 0.25
        or comparison.get("bootstrap_samples") != 2_000
        or comparison.get("winner_slot") != selection.get("winner_slot")
        or comparison.get("winner_pipeline") != selection.get("selected_pipeline")
        or not isinstance(records, Mapping)
        or set(records) != set(TOURNAMENT_COMPARISON_NAMES)
    ):
        raise IndependentRuntimeEvaluationError(
            "adoption deterministic comparison replay proof is invalid"
        )
    assert isinstance(records, Mapping)

    outcomes: dict[str, bool] = {}
    selected_pipelines: dict[str, str] = {}
    selected_heads: dict[str, str] = {}
    for name in TOURNAMENT_COMPARISON_NAMES:
        if name.endswith("_primary_vs_detail"):
            slot = name.removesuffix("_primary_vs_detail")
            expected_slots = (slot, slot)
            expected_pipelines = ("primary", "configured")
        elif name.endswith("_end2end_vs_traditional"):
            scale = name[0]
            expected_slots = (f"{scale}_end2end", f"{scale}_traditional")
            expected_pipelines = tuple(
                selected_pipelines[slot] for slot in expected_slots
            )
        else:
            expected_slots = (
                f"n_{selected_heads['n']}",
                f"s_{selected_heads['s']}",
            )
            expected_pipelines = tuple(
                selected_pipelines[slot] for slot in expected_slots
            )
        baseline_slot, candidate_slot = expected_slots
        baseline = tournament_candidates.get(baseline_slot)
        challenger = tournament_candidates.get(candidate_slot)
        replay_record = records.get(name)
        sealed_record = tournament_comparisons.get(name)
        comparison_artifact = artifacts.get(f"tournament_comparison_{name}")
        if not all(
            isinstance(item, Mapping)
            for item in (baseline, challenger, replay_record, sealed_record, comparison_artifact)
        ):
            raise IndependentRuntimeEvaluationError(
                f"adoption replay {name} cross-link is incomplete"
            )
        assert isinstance(baseline, Mapping)
        assert isinstance(challenger, Mapping)
        assert isinstance(replay_record, Mapping)
        assert isinstance(sealed_record, Mapping)
        assert isinstance(comparison_artifact, Mapping)
        expected_sha = comparison_artifact.get("sha256")
        advanced = sealed_record.get("challenger_advanced")
        if (
            set(replay_record)
            != {
                "baseline_slot",
                "candidate_slot",
                "baseline_pipeline",
                "candidate_pipeline",
                "baseline_report_sha256",
                "candidate_report_sha256",
                "sealed_comparison_sha256",
                "replayed_comparison_sha256",
                "challenger_advanced",
            }
            or replay_record.get("baseline_slot") != baseline_slot
            or replay_record.get("candidate_slot") != candidate_slot
            or replay_record.get("baseline_pipeline") != expected_pipelines[0]
            or replay_record.get("candidate_pipeline") != expected_pipelines[1]
            or replay_record.get("baseline_report_sha256")
            != baseline.get("validation_report_sha256")
            or replay_record.get("candidate_report_sha256")
            != challenger.get("validation_report_sha256")
            or replay_record.get("sealed_comparison_sha256") != expected_sha
            or replay_record.get("replayed_comparison_sha256") != expected_sha
            or replay_record.get("challenger_advanced") is not advanced
            or not isinstance(advanced, bool)
        ):
            raise IndependentRuntimeEvaluationError(
                f"adoption replay {name} differs from the sealed reports/comparison"
            )
        outcomes[name] = advanced
        if name.endswith("_primary_vs_detail"):
            selected_pipelines[baseline_slot] = (
                "configured" if advanced else "primary"
            )
        elif name.endswith("_end2end_vs_traditional"):
            selected_heads[name[0]] = "traditional" if advanced else "end2end"

    winner_scale = "s" if outcomes["n_vs_s"] else "n"
    winner_slot = f"{winner_scale}_{selected_heads[winner_scale]}"
    winner_candidate = tournament_candidates.get(winner_slot)
    training_source = source_artifacts.get("training_results")
    winner_training_artifact = artifacts.get(
        f"tournament_training_results_{winner_scale}"
    )
    adopted_training_artifact = artifacts.get("training_results")
    training_identity = (
        winner_candidate.get("training_identity")
        if isinstance(winner_candidate, Mapping)
        else None
    )
    if not all(
        isinstance(item, Mapping)
        for item in (
            winner_candidate,
            training_source,
            winner_training_artifact,
            adopted_training_artifact,
            training_identity,
        )
    ):
        raise IndependentRuntimeEvaluationError(
            "adoption winner training-results replay cross-link is incomplete"
        )
    assert isinstance(training_source, Mapping)
    assert isinstance(winner_training_artifact, Mapping)
    assert isinstance(adopted_training_artifact, Mapping)
    assert isinstance(training_identity, Mapping)
    if (
        winner_slot != selection.get("winner_slot")
        or selected_pipelines[winner_slot] != selection.get("selected_pipeline")
        or set(training)
        != {"scale", "bytes", "sha256", "completed_epochs", "results_rows"}
        or training.get("scale") != winner_scale
        or training.get("bytes") != training_source.get("bytes")
        or training.get("sha256") != training_source.get("sha256")
        or training.get("bytes") != winner_training_artifact.get("bytes")
        or training.get("sha256") != winner_training_artifact.get("sha256")
        or training.get("bytes") != adopted_training_artifact.get("bytes")
        or training.get("sha256") != adopted_training_artifact.get("sha256")
        or training.get("completed_epochs")
        != training_identity.get("completed_epochs")
        or training.get("results_rows") != training_identity.get("results_rows")
        or _require_int(
            training.get("completed_epochs"),
            "adoption replay completed epochs",
            minimum=1,
        )
        != _require_int(
            training.get("results_rows"),
            "adoption replay training-result rows",
            minimum=1,
        )
    ):
        raise IndependentRuntimeEvaluationError(
            "adoption winner training-results replay proof is invalid"
        )
    return canonical_hash(replay)


def _release_candidate_binding(project_root: Path, backend: str) -> dict[str, Any]:
    """Validate and bind the current adopted, development-selected candidate."""

    if backend != "onnxruntime":
        raise IndependentRuntimeEvaluationError(
            "sealed GPU qualification currently requires the ONNX Runtime artifact"
        )
    root = project_root.expanduser().resolve()
    try:
        pointer = load_release_default_contract(root, verify_files=True)
    except Exception as exc:
        raise IndependentRuntimeEvaluationError(
            f"release-default contract is invalid: {exc}"
        ) from exc
    provenance = pointer["provenance"]
    if provenance.get("kind") != "development_selected_candidate":
        raise IndependentRuntimeEvaluationError(
            "release default was not adopted from frozen development-selection evidence"
        )
    artifacts = pointer["artifacts"]
    adoption_record = artifacts.get("adoption_record")
    selection_record = artifacts.get("tournament_selection_manifest")
    if not isinstance(adoption_record, Mapping) or not isinstance(
        selection_record, Mapping
    ):
        raise IndependentRuntimeEvaluationError(
            "release default lacks adoption/tournament-selection evidence"
        )
    adoption_path, adoption, adoption_payload = _strict_json_file(
        root / str(adoption_record["path"]), "candidate adoption record"
    )
    selection_path, tournament, selection_payload = _strict_json_file(
        root / str(selection_record["path"]), "model-tournament selection"
    )
    if (
        _sha256_file(adoption_path) != adoption_record.get("sha256")
        or len(adoption_payload) != adoption_record.get("bytes")
        or _sha256_file(selection_path) != selection_record.get("sha256")
        or len(selection_payload) != selection_record.get("bytes")
    ):
        raise IndependentRuntimeEvaluationError(
            "adoption/tournament bytes differ from the release-default pointer"
        )
    if set(adoption) != {
        "schema_version",
        "status",
        "candidate",
        "selection",
        "source",
        "qualification",
        "content_sha256",
    }:
        raise IndependentRuntimeEvaluationError("candidate adoption schema is incomplete")
    if (
        adoption.get("schema_version") != 1
        or adoption.get("status")
        != "development_selected_default_not_release_qualified"
        or adoption.get("qualification") != QUALIFICATION_RECORD
        or adoption.get("content_sha256") != _adoption_body_hash(adoption)
    ):
        raise IndependentRuntimeEvaluationError(
            "candidate adoption record is invalid or over-qualified"
        )
    candidate = adoption.get("candidate")
    selection = adoption.get("selection")
    source = adoption.get("source")
    if not all(isinstance(item, Mapping) for item in (candidate, selection, source)):
        raise IndependentRuntimeEvaluationError("candidate adoption binding is incomplete")
    assert isinstance(candidate, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(source, Mapping)
    if set(candidate) != {
        "candidate_content_sha256",
        "candidate_manifest_sha256",
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "dataset_content_sha256",
        "input_shape_nchw",
        "output_head",
        "source_artifacts",
    } or set(selection) != {
        "candidate_evaluation_sha256",
        "selected_backend",
        "selected_pipeline",
        "selected_model_content_sha256",
        "detail_crop_size_source_pixels",
        "tournament_selection_sha256",
        "tournament_selection_content_sha256",
        "winner_slot",
        "evidence_replay",
        "sealed_tournament_winner",
        "release_qualified",
    }:
        raise IndependentRuntimeEvaluationError(
            "candidate adoption winner schema is incomplete or unexpected"
        )
    for key in (
        "candidate_content_sha256",
        "candidate_manifest_sha256",
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "dataset_content_sha256",
    ):
        _require_sha256(candidate.get(key), f"adoption candidate {key}")
    output_head = candidate.get("output_head")
    if output_head not in {"end2end", "traditional"}:
        raise IndependentRuntimeEvaluationError(
            "adoption does not bind the candidate's exact exported output head"
        )
    evidence_replay = selection.get("evidence_replay")
    if not isinstance(evidence_replay, Mapping):
        raise IndependentRuntimeEvaluationError(
            "adoption omits the sealed tournament semantic replay proof"
        )
    detail_crop_size = _require_int(
        pointer.get("detail_crop_size_source_pixels"),
        "adopted production detail crop width",
        minimum=0,
    )
    selected_pipeline = selection.get("selected_pipeline")
    if selected_pipeline not in {"primary", "configured"} or (
        selected_pipeline == "primary" and detail_crop_size != 0
    ) or (selected_pipeline == "configured" and detail_crop_size <= 0):
        raise IndependentRuntimeEvaluationError(
            "adopted tournament pipeline and production detail width disagree"
        )
    if (
        candidate.get("candidate_content_sha256")
        != provenance.get("candidate_content_sha256")
        or candidate.get("candidate_manifest_sha256")
        != provenance.get("candidate_manifest_sha256")
        or candidate.get("input_shape_nchw") != pointer["input_shape_nchw"]
        or selection.get("tournament_selection_sha256")
        != provenance.get("tournament_selection_sha256")
        or selection.get("tournament_selection_sha256")
        != selection_record.get("sha256")
        or selection.get("tournament_selection_content_sha256")
        != tournament.get("selection_content_sha256")
        or selection.get("sealed_tournament_winner") is not True
        or selection.get("release_qualified") is not False
        or selection.get("selected_backend") != backend
        or selection.get("detail_crop_size_source_pixels") != detail_crop_size
    ):
        raise IndependentRuntimeEvaluationError(
            "release pointer, candidate adoption, and tournament winner disagree"
        )
    current_source_paths = {
        "adoption_sha256": PROJECT_ROOT / "scripts" / "adopt_fort_release_candidate.py",
        "candidate_exporter_sha256": PROJECT_ROOT
        / "scripts"
        / "export_fort_release_candidate.py",
        "runtime_comparator_sha256": PROJECT_ROOT
        / "scripts"
        / "compare_fort_runtime_evaluations.py",
        "release_contract_sha256": PROJECT_ROOT
        / "utils"
        / "release_model_contract.py",
        "public_evidence_sha256": PROJECT_ROOT / "utils" / "public_evidence.py",
    }
    if set(source) != set(current_source_paths) or any(
        source.get(key) != _sha256_file(path)
        for key, path in current_source_paths.items()
    ):
        raise IndependentRuntimeEvaluationError(
            "adopted candidate was produced by different exporter/adoption sources"
        )
    expected_tournament_fields = {
        "schema_version",
        "status",
        "orchestrator",
        "comparator",
        "candidate_exporter",
        "runtime_evaluator",
        "public_evidence_privacy",
        "plan",
        "sealed_inputs",
        "fixed_contract",
        "dataset",
        "candidates",
        "comparisons",
        "development_selection",
        "test_data_policy",
        "release_qualification",
        "selection_content_sha256",
    }
    if (
        set(tournament) != expected_tournament_fields
        or tournament.get("schema_version") != 1
        or tournament.get("status") != "development_model_selection_only"
        or tournament.get("selection_content_sha256")
        != _tournament_selection_body_hash(tournament)
        or tournament.get("test_data_policy")
        != {
            "test_split_consumed": False,
            "test_reports_accepted": False,
            "selection_split": "val",
        }
    ):
        raise IndependentRuntimeEvaluationError(
            "copied tournament selection is invalid or consumed development test data"
        )
    tournament_source_paths = {
        "orchestrator": (
            "run_fort_model_tournament.py",
            PROJECT_ROOT / "scripts" / "run_fort_model_tournament.py",
        ),
        "comparator": (
            "compare_fort_runtime_evaluations.py",
            PROJECT_ROOT / "scripts" / "compare_fort_runtime_evaluations.py",
        ),
        "candidate_exporter": (
            "export_fort_release_candidate.py",
            PROJECT_ROOT / "scripts" / "export_fort_release_candidate.py",
        ),
        "public_evidence_privacy": (
            "public_evidence.py",
            PROJECT_ROOT / "utils" / "public_evidence.py",
        ),
    }
    if any(
        tournament.get(key)
        != {"path": name, "sha256": _sha256_file(path)}
        for key, (name, path) in tournament_source_paths.items()
    ):
        raise IndependentRuntimeEvaluationError(
            "tournament selection uses different orchestrator/export sources"
        )
    current_development_runtime = _source_hash_snapshot("onnxruntime")
    if tournament.get("runtime_evaluator") != {
        "path": "evaluate_fort_runtime_model.py",
        "sha256": current_development_runtime["evaluator"]["sha256"],
        "pipeline_source_sha256": current_development_runtime["pipeline"],
    }:
        raise IndependentRuntimeEvaluationError(
            "tournament selection uses a different development runtime pipeline"
        )
    release_qualification = tournament.get("release_qualification")
    fixed_contract = tournament.get("fixed_contract")
    development_selection = tournament.get("development_selection")
    tournament_candidates = tournament.get("candidates")
    tournament_comparisons = tournament.get("comparisons")
    tournament_plan = tournament.get("plan")
    tournament_sealed_inputs = tournament.get("sealed_inputs")
    dataset = tournament.get("dataset")
    if not all(
        isinstance(item, Mapping)
        for item in (
            release_qualification,
            fixed_contract,
            development_selection,
            tournament_candidates,
            tournament_comparisons,
            tournament_plan,
            tournament_sealed_inputs,
            dataset,
        )
    ):
        raise IndependentRuntimeEvaluationError(
            "copied tournament selection contracts are incomplete"
        )
    assert isinstance(release_qualification, Mapping)
    assert isinstance(fixed_contract, Mapping)
    assert isinstance(development_selection, Mapping)
    assert isinstance(tournament_candidates, Mapping)
    assert isinstance(tournament_comparisons, Mapping)
    assert isinstance(tournament_plan, Mapping)
    assert isinstance(tournament_sealed_inputs, Mapping)
    assert isinstance(dataset, Mapping)
    tournament_detail_crop_size = _require_int(
        fixed_contract.get("detail_crop_size_source_pixels"),
        "tournament configured detail crop width",
        minimum=1,
    )
    if (
        release_qualification.get("qualified") is not False
        or release_qualification.get("release_model_approved") is not False
        or release_qualification.get("independent_holdout_required") is not True
        or release_qualification.get("reviewed_negative_scenes_required") is not True
        or release_qualification.get("physical_target_gpu_latency_required") is not True
        or release_qualification.get("frozen_build_qualification_required") is not True
        or fixed_contract.get("backend") != "onnxruntime"
        or fixed_contract.get("input_shape_nchw") != pointer["input_shape_nchw"]
        or fixed_contract.get("inference_size")
        != format_inference_size(
            (pointer["input_shape_nchw"][2], pointer["input_shape_nchw"][3])
        )
        or (
            selected_pipeline == "configured"
            and tournament_detail_crop_size != detail_crop_size
        )
        or fixed_contract.get("split") != "val"
        or set(tournament_plan) != {"path", "sha256", "status", "timing_note"}
        or tournament_plan.get("status") != "development_model_tournament_plan"
        or dataset.get("manifest_sha256")
        != candidate.get("dataset_manifest_sha256")
        or dataset.get("content_sha256")
        != candidate.get("dataset_content_sha256")
    ):
        raise IndependentRuntimeEvaluationError(
            "tournament workload/dataset is not the adopted validation-only contract"
        )
    winner = development_selection.get("winner")
    winner_slot = selection.get("winner_slot")
    winner_candidate = tournament_candidates.get(winner_slot)
    if not isinstance(winner, Mapping) or not isinstance(winner_candidate, Mapping):
        raise IndependentRuntimeEvaluationError("tournament winner slot is missing")
    if (
        winner.get("slot") != winner_slot
        or winner.get("head") != output_head
        or winner.get("pipeline") != selection.get("selected_pipeline")
        or winner.get("candidate_content_sha256")
        != candidate.get("candidate_content_sha256")
        or winner.get("validation_report_sha256")
        != selection.get("candidate_evaluation_sha256")
        or winner_candidate.get("head") != output_head
        or winner_candidate.get("candidate_content_sha256")
        != candidate.get("candidate_content_sha256")
        or winner_candidate.get("candidate_manifest_sha256")
        != candidate.get("candidate_manifest_sha256")
        or winner_candidate.get("checkpoint_sha256")
        != candidate.get("checkpoint_sha256")
    ):
        raise IndependentRuntimeEvaluationError(
            "adoption does not identify the exact tournament winner slot"
        )

    tournament_evidence: list[dict[str, Any]] = [
        {
            "role": "tournament_selection_manifest",
            "relative_path": str(selection_record["path"]),
            "bytes": selection_record["bytes"],
            "sha256": selection_record["sha256"],
        }
    ]
    tournament_evidence_files: list[tuple[Path, str]] = [
        (selection_path, str(selection_record["sha256"]))
    ]
    if set(tournament_comparisons) != set(TOURNAMENT_COMPARISON_NAMES):
        raise IndependentRuntimeEvaluationError(
            "tournament comparison inventory is incomplete"
        )
    for name in TOURNAMENT_COMPARISON_NAMES:
        role = f"tournament_comparison_{name}"
        pointer_record = artifacts.get(role)
        comparison_manifest_record = tournament_comparisons.get(name)
        if not isinstance(pointer_record, Mapping) or not isinstance(
            comparison_manifest_record, Mapping
        ):
            raise IndependentRuntimeEvaluationError(
                f"tournament comparison {name} binding is missing"
            )
        comparison_path, comparison, comparison_payload = _strict_json_file(
            root / str(pointer_record["path"]), f"tournament comparison {name}"
        )
        if (
            _sha256_file(comparison_path) != pointer_record.get("sha256")
            or len(comparison_payload) != pointer_record.get("bytes")
            or comparison_manifest_record.get("sha256")
            != pointer_record.get("sha256")
            or comparison.get("status") != "development_selection_evidence_only"
            or not isinstance(comparison.get("release_qualification"), Mapping)
            or comparison["release_qualification"].get("qualified") is not False
        ):
            raise IndependentRuntimeEvaluationError(
                f"tournament comparison {name} is invalid or over-qualified"
            )
        tournament_evidence.append(
            {
                "role": role,
                "relative_path": str(pointer_record["path"]),
                "bytes": pointer_record["bytes"],
                "sha256": pointer_record["sha256"],
            }
        )
        tournament_evidence_files.append(
            (comparison_path, str(pointer_record["sha256"]))
        )

    if set(tournament_sealed_inputs) != {
        "plan",
        "runtime_reports",
        "training_results",
    }:
        raise IndependentRuntimeEvaluationError(
            "tournament sealed-input inventory is incomplete or unexpected"
        )
    sealed_runtime_reports = tournament_sealed_inputs.get("runtime_reports")
    sealed_training_results = tournament_sealed_inputs.get("training_results")
    if (
        not isinstance(sealed_runtime_reports, Mapping)
        or set(sealed_runtime_reports) != set(TOURNAMENT_SLOT_NAMES)
        or not isinstance(sealed_training_results, Mapping)
        or set(sealed_training_results) != {"n", "s"}
    ):
        raise IndependentRuntimeEvaluationError(
            "tournament sealed runtime/training inventory is incomplete"
        )
    if (
        tournament_plan.get("path") != "inputs/tournament-plan.json"
        or tournament_plan.get("sha256")
        != (
            tournament_sealed_inputs.get("plan", {}).get("sha256")
            if isinstance(tournament_sealed_inputs.get("plan"), Mapping)
            else None
        )
    ):
        raise IndependentRuntimeEvaluationError(
            "tournament plan and its sealed source identity disagree"
        )

    sealed_records: dict[str, tuple[object, str, str]] = {
        "tournament_plan": (
            tournament_sealed_inputs.get("plan"),
            "inputs/tournament-plan.json",
            _require_sha256(tournament_plan.get("sha256"), "tournament plan hash"),
        )
    }
    for name in TOURNAMENT_SLOT_NAMES:
        slot = tournament_candidates.get(name)
        if not isinstance(slot, Mapping):
            raise IndependentRuntimeEvaluationError(
                f"tournament sealed runtime slot {name} is missing"
            )
        sealed_records[f"tournament_runtime_report_{name}"] = (
            sealed_runtime_reports[name],
            f"inputs/runtime/{name}/validation-metrics.json",
            _require_sha256(
                slot.get("validation_report_sha256"),
                f"tournament {name} validation report hash",
            ),
        )
    for scale in ("n", "s"):
        head_hashes: list[str] = []
        for head in ("end2end", "traditional"):
            slot = tournament_candidates.get(f"{scale}_{head}")
            training_identity = (
                slot.get("training_identity") if isinstance(slot, Mapping) else None
            )
            if not isinstance(training_identity, Mapping):
                raise IndependentRuntimeEvaluationError(
                    f"tournament {scale}_{head} training identity is missing"
                )
            head_hashes.append(
                _require_sha256(
                    training_identity.get("training_results_sha256"),
                    f"tournament {scale}_{head} training-results hash",
                )
            )
        if len(set(head_hashes)) != 1:
            raise IndependentRuntimeEvaluationError(
                f"tournament {scale} head exports use different training results"
            )
        sealed_records[f"tournament_training_results_{scale}"] = (
            sealed_training_results[scale],
            f"inputs/training/{scale}/training-results.csv",
            head_hashes[0],
        )
    if set(sealed_records) != set(TOURNAMENT_SEALED_INPUT_ROLES):
        raise IndependentRuntimeEvaluationError(
            "tournament sealed-input role contract differs from the release pointer"
        )

    seen_sealed_paths: set[str] = set()
    for role in sorted(sealed_records):
        manifest_record, expected_source_path, expected_digest = sealed_records[role]
        if not isinstance(manifest_record, Mapping) or set(manifest_record) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise IndependentRuntimeEvaluationError(
                f"tournament {role} manifest record is invalid"
            )
        source_relative = manifest_record.get("path")
        manifest_size = _require_int(
            manifest_record.get("bytes"), f"tournament {role} byte size", minimum=1
        )
        manifest_digest = _require_sha256(
            manifest_record.get("sha256"), f"tournament {role} hash"
        )
        if (
            source_relative != expected_source_path
            or source_relative in seen_sealed_paths
            or manifest_digest != expected_digest
        ):
            raise IndependentRuntimeEvaluationError(
                f"tournament {role} source identity is unexpected or repeated"
            )
        seen_sealed_paths.add(str(source_relative))

        pointer_record = artifacts.get(role)
        if not isinstance(pointer_record, Mapping):
            raise IndependentRuntimeEvaluationError(
                f"release pointer omits sealed tournament role {role}"
            )
        relative = _canonical_relative(pointer_record.get("path"), f"{role} path")
        copied_path = _regular_file(root / relative, f"copied {role}")
        if (
            pointer_record.get("bytes") != manifest_size
            or pointer_record.get("sha256") != manifest_digest
            or copied_path.stat().st_size != manifest_size
            or _sha256_file(copied_path) != manifest_digest
        ):
            raise IndependentRuntimeEvaluationError(
                f"copied {role} differs from the sealed tournament source"
            )
        if copied_path.suffix.casefold() == ".json":
            try:
                public_payload = copied_path.read_bytes()
            except OSError as exc:
                raise IndependentRuntimeEvaluationError(
                    f"cannot read copied {role}: {exc}"
                ) from exc
            if (
                len(public_payload) != manifest_size
                or sha256(public_payload).hexdigest() != manifest_digest
            ):
                raise IndependentRuntimeEvaluationError(
                    f"copied {role} bytes changed during validation"
                )
            public_json = _parse_json_object_payload(
                public_payload, f"copied {role}"
            )
            _reject_private_path_strings(public_json, f"copied {role}")
        else:
            _assert_public_text_safe(copied_path, f"copied {role}")
        tournament_evidence.append(
            {
                "role": role,
                "relative_path": relative,
                "bytes": manifest_size,
                "sha256": manifest_digest,
            }
        )
        tournament_evidence_files.append((copied_path, manifest_digest))

    source_artifacts = candidate.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise IndependentRuntimeEvaluationError("adoption lacks source artifact records")
    evidence_replay_sha256 = _validate_adoption_evidence_replay(
        replay=evidence_replay,
        artifacts=artifacts,
        selection=selection,
        source_artifacts=source_artifacts,
        tournament_candidates=tournament_candidates,
        tournament_comparisons=tournament_comparisons,
        tournament_plan=tournament_plan,
    )
    candidate_provenance_evidence: list[dict[str, Any]] = []
    candidate_provenance_files: list[tuple[Path, str]] = []
    provenance_roles = (
        "candidate_receipt",
        "training_provenance_receipt",
        "training_results",
        "winner_runtime_receipt",
    )
    for role in provenance_roles:
        pointer_record = artifacts.get(role)
        if not isinstance(pointer_record, Mapping):
            raise IndependentRuntimeEvaluationError(
                f"adopted candidate lacks self-contained {role} provenance"
            )
        relative = _canonical_relative(
            pointer_record.get("path"), f"{role} path"
        )
        provenance_path = _regular_file(
            root / relative, f"adopted {role} provenance"
        )
        digest = pointer_record.get("sha256")
        if (
            provenance_path.stat().st_size != pointer_record.get("bytes")
            or _sha256_file(provenance_path) != digest
        ):
            raise IndependentRuntimeEvaluationError(
                f"adopted {role} provenance bytes differ from the pointer"
            )
        if role == "training_results":
            source_record = source_artifacts.get(role)
            if (
                not isinstance(source_record, Mapping)
                or source_record.get("name") != Path(relative).name
                or source_record.get("bytes") != pointer_record.get("bytes")
                or source_record.get("sha256") != digest
            ):
                raise IndependentRuntimeEvaluationError(
                    f"copied {role} differs from candidate-manifest provenance"
                )
        else:
            _receipt_path, receipt, receipt_payload = _strict_json_file(
                provenance_path, f"adopted {role}"
            )
            if len(receipt_payload) != pointer_record.get("bytes"):
                raise IndependentRuntimeEvaluationError(
                    f"adopted {role} size differs from its pointer"
                )
            receipt_contracts = {
                "candidate_receipt": (
                    {
                        "schema_version",
                        "status",
                        "original_candidate_manifest_sha256",
                        "candidate_content_sha256",
                        "configuration",
                        "checkpoint",
                        "dataset",
                        "artifacts",
                        "exporter",
                        "package_versions",
                        "parity",
                        "qualification",
                        "content_sha256",
                    },
                    "redacted_candidate_receipt_not_release_qualified",
                ),
                "training_provenance_receipt": (
                    {
                        "schema_version",
                        "status",
                        "candidate_manifest",
                        "training",
                        "environment_versions",
                        "inputs",
                        "output",
                        "original_local_records",
                        "qualification",
                        "content_sha256",
                    },
                    "redacted_training_provenance_not_release_qualified",
                ),
                "winner_runtime_receipt": (
                    {
                        "schema_version",
                        "status",
                        "original_runtime_evaluation_sha256",
                        "candidate_content_sha256",
                        "selected_pipeline",
                        "detail_crop_size_source_pixels",
                        "evaluator",
                        "model_artifact",
                        "dataset",
                        "configuration",
                        "runtime",
                        "environment_versions",
                        "metrics",
                        "qualification",
                        "content_sha256",
                    },
                    "redacted_winner_runtime_receipt_not_release_qualified",
                ),
            }
            expected_fields, expected_status = receipt_contracts[role]
            if (
                set(receipt) != expected_fields
                or receipt.get("schema_version") != 1
                or receipt.get("status") != expected_status
                or receipt.get("qualification") != QUALIFICATION_RECORD
                or receipt.get("content_sha256") != _receipt_body_hash(receipt)
            ):
                raise IndependentRuntimeEvaluationError(
                    f"adopted {role} schema/self-hash is invalid or over-qualified"
                )
            if role == "candidate_receipt" and (
                receipt.get("original_candidate_manifest_sha256")
                != candidate.get("candidate_manifest_sha256")
                or receipt.get("candidate_content_sha256")
                != candidate.get("candidate_content_sha256")
            ):
                raise IndependentRuntimeEvaluationError(
                    "candidate receipt differs from the adopted candidate"
                )
            if role == "candidate_receipt":
                receipt_configuration = receipt.get("configuration")
                receipt_checkpoint = receipt.get("checkpoint")
                receipt_dataset = receipt.get("dataset")
                if (
                    not isinstance(receipt_configuration, Mapping)
                    or not isinstance(receipt_checkpoint, Mapping)
                    or not isinstance(receipt_dataset, Mapping)
                    or receipt_configuration.get("head") != output_head
                    or receipt_configuration.get("input_shape_nchw")
                    != pointer["input_shape_nchw"]
                    or receipt_configuration.get("one_class") != {"0": "player"}
                    or receipt_checkpoint.get("sha256")
                    != candidate.get("checkpoint_sha256")
                    or receipt_dataset.get("manifest_sha256")
                    != candidate.get("dataset_manifest_sha256")
                    or receipt_dataset.get("content_sha256")
                    != candidate.get("dataset_content_sha256")
                    or receipt.get("artifacts") != source_artifacts
                ):
                    raise IndependentRuntimeEvaluationError(
                        "candidate receipt configuration/artifacts are not the winner"
                    )
            if role == "training_provenance_receipt":
                receipt_candidate = receipt.get("candidate_manifest")
                receipt_inputs = receipt.get("inputs")
                receipt_output = receipt.get("output")
                local_records = receipt.get("original_local_records")
                if not all(
                    isinstance(item, Mapping)
                    for item in (
                        receipt_candidate,
                        receipt_inputs,
                        receipt_output,
                        local_records,
                    )
                ):
                    raise IndependentRuntimeEvaluationError(
                        "training provenance receipt mappings are incomplete"
                    )
                assert isinstance(receipt_candidate, Mapping)
                assert isinstance(receipt_inputs, Mapping)
                assert isinstance(receipt_output, Mapping)
                assert isinstance(local_records, Mapping)
                receipt_checkpoint = receipt_output.get("checkpoint")
                initial_record = source_artifacts.get("initial_run_contract")
                reproducibility_record = source_artifacts.get(
                    "training_reproducibility"
                )
                results_record = source_artifacts.get("training_results")
                if (
                    receipt_candidate.get("original_sha256")
                    != candidate.get("candidate_manifest_sha256")
                    or receipt_candidate.get("candidate_content_sha256")
                    != candidate.get("candidate_content_sha256")
                    or receipt_inputs.get("dataset_manifest_sha256")
                    != candidate.get("dataset_manifest_sha256")
                    or receipt_inputs.get("dataset_content_sha256")
                    != candidate.get("dataset_content_sha256")
                    or not isinstance(receipt_checkpoint, Mapping)
                    or receipt_checkpoint.get("sha256")
                    != candidate.get("checkpoint_sha256")
                    or receipt_output.get("training_results")
                    != results_record
                    or not isinstance(initial_record, Mapping)
                    or not isinstance(reproducibility_record, Mapping)
                    or not isinstance(results_record, Mapping)
                    or local_records.get("initial_run_contract_sha256")
                    != initial_record.get("sha256")
                    or local_records.get("training_reproducibility_sha256")
                    != reproducibility_record.get("sha256")
                    or local_records.get("training_results_sha256")
                    != results_record.get("sha256")
                ):
                    raise IndependentRuntimeEvaluationError(
                        "training provenance receipt does not bind the candidate inputs/output"
                    )
            if role == "winner_runtime_receipt" and (
                receipt.get("original_runtime_evaluation_sha256")
                != selection.get("candidate_evaluation_sha256")
                or receipt.get("candidate_content_sha256")
                != candidate.get("candidate_content_sha256")
                or receipt.get("selected_pipeline") != selected_pipeline
                or receipt.get("detail_crop_size_source_pixels")
                != detail_crop_size
            ):
                raise IndependentRuntimeEvaluationError(
                    "winner runtime receipt differs from tournament adoption"
                )
            if role == "winner_runtime_receipt":
                receipt_artifact = receipt.get("model_artifact")
                receipt_dataset = receipt.get("dataset")
                receipt_configuration = receipt.get("configuration")
                if (
                    not isinstance(receipt_artifact, Mapping)
                    or not isinstance(receipt_dataset, Mapping)
                    or not isinstance(receipt_configuration, Mapping)
                    or receipt_artifact.get("backend") != "onnxruntime"
                    or receipt_artifact.get("content_sha256")
                    != selection.get("selected_model_content_sha256")
                    or receipt_dataset.get("manifest_sha256")
                    != candidate.get("dataset_manifest_sha256")
                    or receipt_dataset.get("content_sha256")
                    != candidate.get("dataset_content_sha256")
                    or receipt_configuration.get("input_shape_nchw")
                    != pointer["input_shape_nchw"]
                    or receipt_configuration.get("output_format") != output_head
                ):
                    raise IndependentRuntimeEvaluationError(
                        "winner runtime receipt does not bind the selected artifact/workload"
                    )
        candidate_provenance_evidence.append(
            {
                "role": role,
                "relative_path": relative,
                "bytes": pointer_record["bytes"],
                "sha256": digest,
            }
        )
        candidate_provenance_files.append((provenance_path, str(digest)))
    roles = ("onnx",)
    model_records: list[dict[str, Any]] = []
    model_paths: list[Path] = []
    for role in roles:
        pointer_record = artifacts.get(role)
        source_record = source_artifacts.get(role)
        if not isinstance(pointer_record, Mapping) or not isinstance(
            source_record, Mapping
        ):
            raise IndependentRuntimeEvaluationError(
                f"adopted candidate lacks {role} artifact binding"
            )
        relative = _canonical_relative(pointer_record.get("path"), f"{role} path")
        path = _regular_file(root / relative, f"adopted {role} artifact")
        normalized = {
            "role": role,
            "name": source_record.get("name"),
            "relative_path": relative,
            "bytes": pointer_record.get("bytes"),
            "sha256": pointer_record.get("sha256"),
        }
        if (
            not isinstance(normalized["name"], str)
            or Path(relative).name != normalized["name"]
            or source_record.get("bytes") != normalized["bytes"]
            or source_record.get("sha256") != normalized["sha256"]
            or path.stat().st_size != normalized["bytes"]
            or _sha256_file(path) != normalized["sha256"]
        ):
            raise IndependentRuntimeEvaluationError(
                f"pointer and candidate {role} artifact identities differ"
            )
        model_records.append(normalized)
        model_paths.append(path)
    model_content = _artifact_content(model_records)
    if selection.get("selected_model_content_sha256") != model_content:
        raise IndependentRuntimeEvaluationError(
            "adopted selection model content differs from the runtime artifact"
        )
    winner_onnx = winner_candidate.get("onnx")
    if (
        not isinstance(winner_onnx, Mapping)
        or winner.get("onnx_sha256") != model_records[0]["sha256"]
        or winner_onnx.get("name") != model_records[0]["name"]
        or winner_onnx.get("bytes") != model_records[0]["bytes"]
        or winner_onnx.get("sha256") != model_records[0]["sha256"]
    ):
        raise IndependentRuntimeEvaluationError(
            "tournament winner ONNX differs from the adopted runtime artifact"
        )
    label_record = artifacts.get("labels")
    source_label_record = source_artifacts.get("labels")
    if not isinstance(label_record, Mapping) or not isinstance(
        source_label_record, Mapping
    ):
        raise IndependentRuntimeEvaluationError("adopted candidate has no label artifact")
    label_relative = _canonical_relative(label_record.get("path"), "labels path")
    labels_path = _regular_file(root / label_relative, "adopted labels")
    try:
        label_text = labels_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise IndependentRuntimeEvaluationError("cannot read adopted labels") from exc
    if (
        label_text != "player\n"
        or source_label_record.get("name") != Path(label_relative).name
        or source_label_record.get("bytes") != label_record.get("bytes")
        or source_label_record.get("sha256") != label_record.get("sha256")
        or labels_path.stat().st_size != label_record.get("bytes")
        or _sha256_file(labels_path) != label_record.get("sha256")
    ):
        raise IndependentRuntimeEvaluationError(
            "adopted labels differ from the candidate source or one-player mapping"
        )
    pointer_path = root.joinpath(*CONTRACT_RELATIVE.parts)
    return {
        "pointer_path": pointer_path,
        "pointer_sha256": _sha256_file(pointer_path),
        "pointer_content_sha256": pointer["content_sha256"],
        "input_shape_nchw": list(pointer["input_shape_nchw"]),
        "candidate_content_sha256": candidate["candidate_content_sha256"],
        "candidate_manifest_sha256": candidate["candidate_manifest_sha256"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "dataset_manifest_sha256": candidate["dataset_manifest_sha256"],
        "dataset_content_sha256": candidate["dataset_content_sha256"],
        "adoption_path": adoption_path,
        "adoption_sha256": adoption_record["sha256"],
        "adoption_content_sha256": adoption["content_sha256"],
        "adoption_evidence_replay_sha256": evidence_replay_sha256,
        "tournament_selection_path": selection_path,
        "tournament_selection_sha256": selection_record["sha256"],
        "tournament_selection_content_sha256": tournament[
            "selection_content_sha256"
        ],
        "tournament_evidence": tournament_evidence,
        "tournament_evidence_files": tournament_evidence_files,
        "candidate_provenance_evidence": candidate_provenance_evidence,
        "candidate_provenance_files": candidate_provenance_files,
        "candidate_evaluation_sha256": selection["candidate_evaluation_sha256"],
        "winner_slot": winner_slot,
        "model_path": model_paths[0],
        "model_artifacts": model_records,
        "model_content_sha256": model_content,
        "labels_path": labels_path,
        "labels": {
            "relative_path": label_relative,
            "bytes": label_record["bytes"],
            "sha256": label_record["sha256"],
        },
        "selected_pipeline": selection.get("selected_pipeline"),
        "selected_backend": selection.get("selected_backend"),
        "output_head": output_head,
        "detail_crop_size_source_pixels": detail_crop_size,
        "exporter_sha256": source["candidate_exporter_sha256"],
        "adoption_source_sha256": source["adoption_sha256"],
    }


def _public_candidate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {key: binding[key] for key in PUBLIC_CANDIDATE_KEYS}


def _validate_sealed_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("kind") != DATASET_KIND or manifest.get("pool") != POOL_SEALED:
        raise IndependentRuntimeEvaluationError(
            "final runtime evaluation requires a sealed independent holdout, not development data"
        )
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise IndependentRuntimeEvaluationError("holdout has no exact inventory counts")
    normalized_counts = {
        key: _require_int(counts.get(key), f"holdout {key}")
        for key in PINNED_RELEASE_MINIMUMS.as_dict()
    }
    required = PINNED_RELEASE_MINIMUMS.as_dict()
    deficits = [
        f"{key}={normalized_counts[key]}<{required[key]}"
        for key in GATING_COUNT_KEYS
        if normalized_counts[key] < required[key]
    ]
    if deficits:
        raise IndependentRuntimeEvaluationError(
            "sealed holdout is below pinned release inventory: " + ", ".join(deficits)
        )
    release_gates = manifest.get("release_gates")
    if (
        not isinstance(release_gates, Mapping)
        or release_gates.get("pinned_minimums") != required
        or release_gates.get("gating_count_keys") != list(GATING_COUNT_KEYS)
        or release_gates.get("meets_pinned_release_gates") is not True
        or release_gates.get("target_le_32_is_descriptive") is not True
    ):
        raise IndependentRuntimeEvaluationError(
            "sealed holdout release-inventory contract is incomplete or altered"
        )
    sessions = manifest.get("sessions")
    images = manifest.get("images")
    annotations = manifest.get("annotations")
    if (
        isinstance(sessions, (str, bytes))
        or not isinstance(sessions, Sequence)
        or not sessions
        or isinstance(images, (str, bytes))
        or not isinstance(images, Sequence)
        or not images
        or not isinstance(annotations, Mapping)
    ):
        raise IndependentRuntimeEvaluationError("holdout inventory is incomplete")
    session_ids: list[str] = []
    for session in sessions:
        if not isinstance(session, Mapping) or not isinstance(
            session.get("session_id"), str
        ):
            raise IndependentRuntimeEvaluationError("holdout session inventory is invalid")
        session_ids.append(str(session["session_id"]))
    if len(set(session_ids)) != len(session_ids):
        raise IndependentRuntimeEvaluationError("holdout source sessions are repeated")
    raw_source_groups = manifest.get("source_group_inventory")
    if not isinstance(raw_source_groups, Mapping) or set(raw_source_groups) != {
        "definition",
        "overall_capture_sessions",
        "target_bearing_capture_sessions",
        "reviewed_negative_capture_sessions",
    }:
        raise IndependentRuntimeEvaluationError(
            "holdout source-group inventory is incomplete or unexpected"
        )
    raw_target_sessions = raw_source_groups.get(
        "target_bearing_capture_sessions"
    )
    target_keys = {
        "target_le_32",
        "target_33_64",
        "target_65_96",
        "target_gt_96",
    }
    if not isinstance(raw_target_sessions, Mapping) or set(
        raw_target_sessions
    ) != target_keys:
        raise IndependentRuntimeEvaluationError(
            "holdout target-bearing source-group inventory is incomplete"
        )
    target_sessions = {
        key: _require_int(
            raw_target_sessions.get(key), f"holdout {key} capture sessions"
        )
        for key in sorted(target_keys)
    }
    overall_sessions = _require_int(
        raw_source_groups.get("overall_capture_sessions"),
        "holdout overall capture sessions",
    )
    negative_sessions = _require_int(
        raw_source_groups.get("reviewed_negative_capture_sessions"),
        "holdout reviewed-negative capture sessions",
    )
    group_deficits: list[str] = []
    if overall_sessions < MINIMUM_CAPTURE_SESSIONS:
        group_deficits.append(
            f"overall_capture_sessions={overall_sessions}<"
            f"{MINIMUM_CAPTURE_SESSIONS}"
        )
    for key in GATING_COUNT_KEYS:
        if key == "reviewed_negatives":
            continue
        if target_sessions[key] < MINIMUM_TARGET_BUCKET_CAPTURE_SESSIONS:
            group_deficits.append(
                f"{key}_capture_sessions={target_sessions[key]}<"
                f"{MINIMUM_TARGET_BUCKET_CAPTURE_SESSIONS}"
            )
    if negative_sessions < MINIMUM_REVIEWED_NEGATIVE_CAPTURE_SESSIONS:
        group_deficits.append(
            f"reviewed_negative_capture_sessions={negative_sessions}<"
            f"{MINIMUM_REVIEWED_NEGATIVE_CAPTURE_SESSIONS}"
        )
    if (
        raw_source_groups.get("definition")
        != "distinct normalized COCO image session_id values"
        or overall_sessions != len(session_ids)
        or group_deficits
    ):
        details = ", ".join(group_deficits) if group_deficits else "schema mismatch"
        raise IndependentRuntimeEvaluationError(
            "sealed holdout is below pinned capture-session inventory: " + details
        )
    source_group_inventory = {
        "definition": "distinct normalized COCO image session_id values",
        "overall_capture_sessions": overall_sessions,
        "target_bearing_capture_sessions": target_sessions,
        "reviewed_negative_capture_sessions": negative_sessions,
    }
    return {
        "package_id": manifest.get("package_id"),
        "manifest_content_sha256": _require_sha256(
            manifest.get("manifest_content_sha256"), "holdout manifest content hash"
        ),
        "pool": POOL_SEALED,
        "counts": normalized_counts,
        "images": len(images),
        "boxes": _require_int(annotations.get("boxes"), "holdout box count"),
        "source_group_definition": "capture_session",
        "source_group_count": len(session_ids),
        "source_group_inventory": source_group_inventory,
        "ultra_far_le_32_is_descriptive_only": True,
        "gating_inventory_keys": list(GATING_COUNT_KEYS),
        "redistribution_permitted_for_all_sessions": bool(
            manifest.get("redistribution_permitted_for_all_sessions")
        ),
    }


def _validate_decision_rule(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "selected_confidence_threshold",
        "minimum_far_recall",
        "maximum_far_false_positives",
        "minimum_medium_recall",
        "minimum_near_recall",
        "minimum_aggregate_precision",
        "minimum_aggregate_recall",
        "maximum_reviewed_negative_false_positives",
        "maximum_runtime_pipeline_p95_ms",
        "manual_review_note",
    }
    if set(value) != expected or value.get("schema_version") != 1 or value.get(
        "kind"
    ) != "proaim-independent-holdout-frozen-decision-rule":
        raise IndependentRuntimeEvaluationError("frozen decision-rule schema is invalid")
    note = value.get("manual_review_note")
    if not isinstance(note, str) or not note.strip() or len(note) > 1_000:
        raise IndependentRuntimeEvaluationError("decision-rule review note is invalid")
    return {
        "schema_version": 1,
        "kind": value["kind"],
        "selected_confidence_threshold": _probability(
            value.get("selected_confidence_threshold"), "selected confidence"
        ),
        "minimum_far_recall": _probability(
            value.get("minimum_far_recall"), "minimum far recall"
        ),
        "maximum_far_false_positives": _require_int(
            value.get("maximum_far_false_positives"), "maximum far false positives"
        ),
        "minimum_medium_recall": _probability(
            value.get("minimum_medium_recall"), "minimum medium recall"
        ),
        "minimum_near_recall": _probability(
            value.get("minimum_near_recall"), "minimum near recall"
        ),
        "minimum_aggregate_precision": _probability(
            value.get("minimum_aggregate_precision"), "minimum aggregate precision"
        ),
        "minimum_aggregate_recall": _probability(
            value.get("minimum_aggregate_recall"), "minimum aggregate recall"
        ),
        "maximum_reviewed_negative_false_positives": _require_int(
            value.get("maximum_reviewed_negative_false_positives"),
            "maximum reviewed-negative false positives",
        ),
        "maximum_runtime_pipeline_p95_ms": _positive_float(
            value.get("maximum_runtime_pipeline_p95_ms"),
            "maximum runtime pipeline p95",
        ),
        "manual_review_note": note.strip(),
    }


def _validate_runtime_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "backend",
        "device",
        "expected_provider",
        "inference_size",
        "input_shape_nchw",
        "output_format",
        "nms_iou_threshold",
        "confidence_thresholds",
        "warmup_iterations",
        "bootstrap_samples",
        "require_full_provider",
        "detail_crop_size_source_pixels",
    }
    if set(value) != expected or value.get("backend") != "onnxruntime":
        raise IndependentRuntimeEvaluationError("runtime plan schema/backend is invalid")
    expected_provider = value.get("expected_provider")
    if expected_provider not in SUPPORTED_ACCELERATOR_PROVIDERS:
        raise IndependentRuntimeEvaluationError(
            "runtime plan must name one exact supported accelerator provider"
        )
    device = value.get("device")
    if not isinstance(device, str) or not device.strip() or device.strip().upper() in {
        "AUTO",
        "GPU",
        "AMD",
        "CPU",
    }:
        raise IndependentRuntimeEvaluationError(
            "runtime plan device must explicitly identify the target accelerator"
        )
    device_family = device.strip().upper().partition(":")[0]
    provider_aliases = {
        "CUDAExecutionProvider": {"CUDA", "NVIDIA", "CUDAEXECUTIONPROVIDER"},
        "TensorrtExecutionProvider": {
            "TENSORRT",
            "TENSORRTEXECUTIONPROVIDER",
        },
        "DmlExecutionProvider": {
            "DML",
            "DIRECTML",
            "DMLEXECUTIONPROVIDER",
        },
        "MIGraphXExecutionProvider": {
            "MIGRAPHX",
            "MIGRAPHXEXECUTIONPROVIDER",
        },
        "ROCMExecutionProvider": {"ROCM", "ROCMEXECUTIONPROVIDER"},
    }
    if device_family not in provider_aliases[expected_provider]:
        raise IndependentRuntimeEvaluationError(
            "runtime plan device alias and expected accelerator provider disagree"
        )
    if expected_provider == "DmlExecutionProvider" and re.fullmatch(
        r"(?i)(?:DML|DIRECTML):[0-9]+", device.strip()
    ) is None:
        raise IndependentRuntimeEvaluationError(
            "DirectML final evidence requires an exact numeric DML adapter id"
        )
    try:
        inference_size = validate_yolo_inference_size(
            parse_inference_size(value.get("inference_size"))
        )
    except (TypeError, ValueError) as exc:
        raise IndependentRuntimeEvaluationError(f"invalid inference size: {exc}") from exc
    if inference_size[0] == inference_size[1]:
        raise IndependentRuntimeEvaluationError(
            "final candidate must use the selected rectangular deployment shape"
        )
    if value.get("input_shape_nchw") != [1, 3, *inference_size]:
        raise IndependentRuntimeEvaluationError("runtime plan static shape disagrees")
    output_format = value.get("output_format")
    if output_format not in {"end2end", "traditional"}:
        raise IndependentRuntimeEvaluationError("runtime output format is invalid")
    thresholds = value.get("confidence_thresholds")
    if isinstance(thresholds, (str, bytes)) or not isinstance(thresholds, Sequence):
        raise IndependentRuntimeEvaluationError("confidence thresholds are invalid")
    normalized_thresholds = tuple(
        _probability(item, "confidence threshold") for item in thresholds
    )
    if (
        normalized_thresholds != tuple(sorted(set(normalized_thresholds)))
        or normalized_thresholds != DEFAULT_CONFIDENCE_THRESHOLDS
    ):
        raise IndependentRuntimeEvaluationError(
            "final evidence requires exact confidence thresholds 0.25 and 0.45"
        )
    nms = _probability(value.get("nms_iou_threshold"), "NMS IoU threshold")
    warmup = _require_int(value.get("warmup_iterations"), "warmup iterations")
    bootstrap = _require_int(
        value.get("bootstrap_samples"), "bootstrap samples", minimum=2_000
    )
    detail_crop = _require_int(
        value.get("detail_crop_size_source_pixels"), "detail crop size"
    )
    if value.get("require_full_provider") is not True:
        raise IndependentRuntimeEvaluationError(
            "final GPU evidence must require the entire graph on its accelerator provider"
        )
    return {
        "backend": "onnxruntime",
        "device": device.strip(),
        "expected_provider": expected_provider,
        "inference_size": format_inference_size(inference_size),
        "input_shape_nchw": [1, 3, *inference_size],
        "output_format": output_format,
        "nms_iou_threshold": nms,
        "confidence_thresholds": list(normalized_thresholds),
        "warmup_iterations": warmup,
        "bootstrap_samples": bootstrap,
        "require_full_provider": True,
        "detail_crop_size_source_pixels": detail_crop,
    }


def _plan_content_hash(plan: Mapping[str, Any]) -> str:
    body = dict(plan)
    body.pop("plan_content_sha256", None)
    return canonical_hash(body)


def _evidence_content_hash(evidence: Mapping[str, Any]) -> str:
    body = dict(evidence)
    body.pop("evidence_content_sha256", None)
    return canonical_hash(body)


def _final_receipt_content_hash(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    body.pop("receipt_content_sha256", None)
    return canonical_hash(body)


def _receipt_verifier_record(source: Mapping[str, Any]) -> dict[str, Any]:
    expected_source = _source_snapshot("onnxruntime")
    if source != expected_source:
        raise IndependentRuntimeEvaluationError(
            "receipt verifier source snapshot differs from the canonical contract"
        )
    try:
        return _shared_receipt_verifier_record(PROJECT_ROOT)
    except Exception as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot construct canonical portable-receipt verifier record: {exc}"
        ) from exc


def build_evaluation_plan(
    *,
    package: Path,
    dependency_manifest: Path,
    hardware_identity: Path,
    project_root: Path,
    device: str,
    expected_provider: str,
    detail_crop_size: int,
    decision_rule: Mapping[str, Any],
    output_format: str | None = None,
    confidence_thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    nms_iou: float = DEFAULT_NMS_IOU,
    warmup: int = DEFAULT_WARMUP,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    candidate_binding_loader: Callable[[Path, str], Mapping[str, Any]] = (
        _release_candidate_binding
    ),
) -> dict[str, Any]:
    """Build an immutable plan without opening any sealed image/annotation member."""

    _, _, environment = _load_release_environment(
        dependency_manifest, project_root=project_root
    )
    _, _, hardware = _load_holdout_hardware_identity(hardware_identity)
    try:
        manifest, _manifest_path, _raw = _load_package_manifest(package)
    except HoldoutContractError as exc:
        raise IndependentRuntimeEvaluationError(str(exc)) from exc
    holdout = _validate_sealed_manifest(manifest)
    binding = dict(candidate_binding_loader(project_root, "onnxruntime"))
    public_binding = _public_candidate_binding(binding)
    shape = public_binding["input_shape_nchw"]
    selected_output_head = public_binding["output_head"]
    if output_format is not None and output_format != selected_output_head:
        raise IndependentRuntimeEvaluationError(
            "plan output format differs from the adopted tournament winner head"
        )
    selected_detail_crop = public_binding["detail_crop_size_source_pixels"]
    if detail_crop_size != selected_detail_crop:
        raise IndependentRuntimeEvaluationError(
            "plan detail crop width differs from the adopted production pipeline"
        )
    runtime = _validate_runtime_plan(
        {
            "backend": "onnxruntime",
            "device": device,
            "expected_provider": expected_provider,
            "inference_size": format_inference_size((shape[2], shape[3])),
            "input_shape_nchw": shape,
            "output_format": selected_output_head,
            "nms_iou_threshold": nms_iou,
            "confidence_thresholds": list(confidence_thresholds),
            "warmup_iterations": warmup,
            "bootstrap_samples": bootstrap_samples,
            "require_full_provider": True,
            "detail_crop_size_source_pixels": selected_detail_crop,
        }
    )
    if (
        runtime["device"] != hardware["directml_device"]
        or runtime["expected_provider"] != "DmlExecutionProvider"
    ):
        raise IndependentRuntimeEvaluationError(
            "plan runtime does not match the exact RX 6950 XT DirectML identity"
        )
    rule = _validate_decision_rule(decision_rule)
    if rule["selected_confidence_threshold"] not in runtime["confidence_thresholds"]:
        raise IndependentRuntimeEvaluationError(
            "decision rule selects an unreported confidence threshold"
        )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "status": PLAN_STATUS,
        "candidate": public_binding,
        "holdout": holdout,
        "runtime": runtime,
        "decision_rule": rule,
        "release_policy": _release_policy_record(),
        "environment": environment["policy"],
        "hardware_identity": hardware,
        "source": _source_snapshot("onnxruntime"),
        "scope": {
            "dataset": "one sealed independent COCO package only",
            "grouped_v9_development_data_permitted": False,
            "candidate_or_threshold_selection_permitted": False,
            "release_approval_permitted": False,
            "ultra_far_le_32_release_gate": False,
            "claim_scope": (
                "absolute_threshold_evidence_only_no_incumbent_comparison"
            ),
        },
    }
    plan["plan_content_sha256"] = _plan_content_hash(plan)
    return plan


def write_evaluation_plan(path: Path, plan: Mapping[str, Any]) -> Path:
    _validate_plan(plan)
    output = path.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as target:
            target.write(canonical_json_bytes(plan))
            target.flush()
            os.fsync(target.fileno())
    except FileExistsError as exc:
        raise IndependentRuntimeEvaluationError(
            f"refusing to overwrite evaluation plan: {output}"
        ) from exc
    return output.resolve()


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "status",
        "candidate",
        "holdout",
        "runtime",
        "decision_rule",
        "release_policy",
        "environment",
        "hardware_identity",
        "source",
        "scope",
        "plan_content_sha256",
    }
    if (
        set(plan) != expected
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("kind") != PLAN_KIND
        or plan.get("status") != PLAN_STATUS
        or plan.get("plan_content_sha256") != _plan_content_hash(plan)
    ):
        raise IndependentRuntimeEvaluationError(
            "evaluation plan schema, status, or content hash is invalid"
        )
    runtime = plan.get("runtime")
    rule = plan.get("decision_rule")
    if not isinstance(runtime, Mapping) or not isinstance(rule, Mapping):
        raise IndependentRuntimeEvaluationError("evaluation plan contracts are incomplete")
    normalized_runtime = _validate_runtime_plan(runtime)
    normalized_rule = _validate_decision_rule(rule)
    if plan.get("release_policy") != _release_policy_record():
        raise IndependentRuntimeEvaluationError(
            "evaluation plan release policy version/hash is invalid"
        )
    try:
        expected_environment_policy = _shared_release_environment_policy_record(
            PROJECT_ROOT
        )
    except IndependentHoldoutReleaseContractError as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot reconstruct the release runtime environment policy: {exc}"
        ) from exc
    if plan.get("environment") != expected_environment_policy:
        raise IndependentRuntimeEvaluationError(
            "evaluation plan runtime environment policy is invalid"
        )
    hardware = plan.get("hardware_identity")
    try:
        normalized_hardware = (
            _shared_validate_holdout_hardware_identity(hardware)
            if isinstance(hardware, Mapping)
            else None
        )
    except IndependentHoldoutReleaseContractError as exc:
        raise IndependentRuntimeEvaluationError(
            f"evaluation plan hardware identity is invalid: {exc}"
        ) from exc
    if (
        normalized_hardware is None
        or normalized_runtime["device"]
        != normalized_hardware["directml_device"]
        or normalized_runtime["expected_provider"] != "DmlExecutionProvider"
    ):
        raise IndependentRuntimeEvaluationError(
            "evaluation plan does not bind the exact RX 6950 XT DirectML runtime"
        )
    if normalized_rule["selected_confidence_threshold"] not in normalized_runtime[
        "confidence_thresholds"
    ]:
        raise IndependentRuntimeEvaluationError(
            "decision rule selects an unreported confidence threshold"
        )
    candidate = plan.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or set(candidate) != set(PUBLIC_CANDIDATE_KEYS)
        or candidate.get("output_head") not in {"end2end", "traditional"}
        or normalized_runtime["output_format"] != candidate.get("output_head")
        or candidate.get("selected_backend") != "onnxruntime"
        or candidate.get("selected_pipeline") not in {"primary", "configured"}
        or normalized_runtime["detail_crop_size_source_pixels"]
        != candidate.get("detail_crop_size_source_pixels")
        or (
            candidate.get("selected_pipeline") == "primary"
            and normalized_runtime["detail_crop_size_source_pixels"] != 0
        )
        or (
            candidate.get("selected_pipeline") == "configured"
            and normalized_runtime["detail_crop_size_source_pixels"] <= 0
        )
    ):
        raise IndependentRuntimeEvaluationError(
            "runtime pipeline differs from the adopted tournament winner"
        )
    expected_scope = {
        "dataset": "one sealed independent COCO package only",
        "grouped_v9_development_data_permitted": False,
        "candidate_or_threshold_selection_permitted": False,
        "release_approval_permitted": False,
        "ultra_far_le_32_release_gate": False,
        "claim_scope": (
            "absolute_threshold_evidence_only_no_incumbent_comparison"
        ),
    }
    if plan.get("scope") != expected_scope:
        raise IndependentRuntimeEvaluationError("evaluation plan scope is unsafe")
    if not isinstance(plan.get("candidate"), Mapping) or not isinstance(
        plan.get("holdout"), Mapping
    ) or not isinstance(plan.get("source"), Mapping):
        raise IndependentRuntimeEvaluationError("evaluation plan bindings are incomplete")
    return dict(plan)


def _load_plan(path: Path) -> tuple[Path, dict[str, Any], bytes]:
    source, value, payload = _strict_json_file(path, "frozen evaluation plan")
    return source, _validate_plan(value), payload


def _load_exact_image(
    path: Path,
    *,
    expected_sha256: str,
    width: int,
    height: int,
) -> np.ndarray:
    try:
        raw = _read_regular_bytes(
            path, "sealed evaluation image", maximum_bytes=MAX_IMAGE_ENCODED_BYTES
        )
    except HoldoutContractError as exc:
        raise IndependentRuntimeEvaluationError(str(exc)) from exc
    actual = sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise IndependentRuntimeEvaluationError(
            "sealed image changed between package verification and decode"
        )
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if (
        frame is None
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape != (height, width, 3)
    ):
        raise IndependentRuntimeEvaluationError(
            "sealed image decode differs from its exact COCO dimensions"
        )
    return np.ascontiguousarray(frame)


def _coco_inventory(
    package: Path, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    annotations_record = manifest.get("annotations")
    if not isinstance(annotations_record, Mapping):
        raise IndependentRuntimeEvaluationError("holdout annotation record is missing")
    relative = annotations_record.get("path")
    if not isinstance(relative, str):
        raise IndependentRuntimeEvaluationError("holdout annotation path is invalid")
    annotations_path = package / relative
    try:
        coco_value, coco_raw = _load_holdout_json(
            annotations_path, "sealed normalized COCO annotations"
        )
    except HoldoutContractError as exc:
        raise IndependentRuntimeEvaluationError(str(exc)) from exc
    if (
        sha256(coco_raw).hexdigest() != annotations_record.get("sha256")
        or not isinstance(coco_value, Mapping)
        or coco_value.get("categories") != manifest.get("classes")
        or coco_value.get("images") != manifest.get("images")
    ):
        raise IndependentRuntimeEvaluationError(
            "sealed COCO bytes/inventory differ from HOLDOUT-MANIFEST"
        )
    raw_annotations = coco_value.get("annotations")
    raw_images = coco_value.get("images")
    if isinstance(raw_annotations, (str, bytes)) or not isinstance(
        raw_annotations, Sequence
    ) or isinstance(raw_images, (str, bytes)) or not isinstance(raw_images, Sequence):
        raise IndependentRuntimeEvaluationError("sealed COCO inventory is malformed")
    by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in raw_annotations:
        if not isinstance(annotation, Mapping):
            raise IndependentRuntimeEvaluationError("sealed COCO annotation is malformed")
        image_id = _require_int(annotation.get("image_id"), "COCO image id")
        by_image[image_id].append(annotation)
    inventory: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for raw_image in raw_images:
        if not isinstance(raw_image, Mapping):
            raise IndependentRuntimeEvaluationError("sealed COCO image is malformed")
        image_id = _require_int(raw_image.get("id"), "COCO image id")
        if image_id in seen_ids:
            raise IndependentRuntimeEvaluationError("sealed COCO repeats an image id")
        seen_ids.add(image_id)
        width = _require_int(raw_image.get("width"), "COCO image width", minimum=1)
        height = _require_int(raw_image.get("height"), "COCO image height", minimum=1)
        file_name = raw_image.get("file_name")
        session_id = raw_image.get("session_id")
        if not isinstance(file_name, str) or not isinstance(session_id, str):
            raise IndependentRuntimeEvaluationError("sealed COCO source mapping is missing")
        annotations = sorted(by_image.pop(image_id, []), key=lambda item: int(item["id"]))
        targets: list[tuple[tuple[float, float, float, float], float]] = []
        target_identity: list[dict[str, Any]] = []
        for annotation in annotations:
            bbox = annotation.get("bbox")
            if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence) or len(bbox) != 4:
                raise IndependentRuntimeEvaluationError("sealed COCO bbox is malformed")
            x, y, box_width, box_height = (float(item) for item in bbox)
            box = (x, y, x + box_width, y + box_height)
            projected = box_height * REFERENCE_FRAME_HEIGHT / height
            recorded_projected = annotation.get("projected_height_px_at_1080p")
            if (
                not all(math.isfinite(item) for item in box)
                or box_width <= 0.0
                or box_height <= 0.0
                or not isinstance(recorded_projected, Real)
                or not math.isclose(
                    float(recorded_projected), projected, rel_tol=1e-12, abs_tol=1e-9
                )
            ):
                raise IndependentRuntimeEvaluationError(
                    "sealed COCO bbox/source-height mapping is invalid"
                )
            targets.append((box, projected))
            target_identity.append(
                {"id": annotation["id"], "bbox": list(bbox), "projected_height": projected}
            )
        inventory.append(
            {
                "image_id": image_id,
                "path": package / file_name,
                "sha256": _require_sha256(raw_image.get("sha256"), "COCO image hash"),
                "width": width,
                "height": height,
                "session_id": session_id,
                "source_frame_index": _require_int(
                    raw_image.get("source_frame_index"), "COCO source frame index"
                ),
                "targets": targets,
                "target_identity_sha256": canonical_hash(target_identity),
                "reviewed_negative": not targets,
            }
        )
    if by_image:
        raise IndependentRuntimeEvaluationError("COCO annotations reference absent images")
    inventory.sort(key=lambda item: item["image_id"])
    return inventory, sha256(coco_raw).hexdigest()


def _inventory_counts(inventory: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {
        "target_le_32": 0,
        "target_33_64": 0,
        "target_65_96": 0,
        "target_gt_96": 0,
        "reviewed_negatives": 0,
    }
    for image in inventory:
        targets = image.get("targets")
        if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
            raise IndependentRuntimeEvaluationError("sealed target inventory is malformed")
        if not targets:
            result["reviewed_negatives"] += 1
        for target in targets:
            if (
                isinstance(target, (str, bytes))
                or not isinstance(target, Sequence)
                or len(target) != 2
            ):
                raise IndependentRuntimeEvaluationError("sealed target is malformed")
            projected = float(target[1])
            if projected <= 32.0:
                result["target_le_32"] += 1
            elif projected <= 64.0:
                result["target_33_64"] += 1
            elif projected <= 96.0:
                result["target_65_96"] += 1
            else:
                result["target_gt_96"] += 1
    return result


def _inventory_source_groups(
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    overall: set[str] = set()
    negatives: set[str] = set()
    targets: dict[str, set[str]] = {
        "target_le_32": set(),
        "target_33_64": set(),
        "target_65_96": set(),
        "target_gt_96": set(),
    }
    for image in inventory:
        session_id = image.get("session_id")
        image_targets = image.get("targets")
        if not isinstance(session_id, str) or isinstance(
            image_targets, (str, bytes)
        ) or not isinstance(image_targets, Sequence):
            raise IndependentRuntimeEvaluationError(
                "sealed source-group inventory is malformed"
            )
        overall.add(session_id)
        if not image_targets:
            negatives.add(session_id)
        for target in image_targets:
            projected = float(target[1])
            if projected <= 32.0:
                bucket = "target_le_32"
            elif projected <= 64.0:
                bucket = "target_33_64"
            elif projected <= 96.0:
                bucket = "target_65_96"
            else:
                bucket = "target_gt_96"
            targets[bucket].add(session_id)
    return {
        "definition": "distinct normalized COCO image session_id values",
        "overall_capture_sessions": len(overall),
        "target_bearing_capture_sessions": {
            key: len(targets[key]) for key in sorted(targets)
        },
        "reviewed_negative_capture_sessions": len(negatives),
    }


def _cluster_weights(
    group_ids: Sequence[str], *, samples: int, seed_material: str
) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        groups[group].append(index)
    ordered = sorted(groups)
    generator = random.Random(
        int.from_bytes(sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    )
    result: list[list[int]] = []
    for _ in range(samples):
        weights = [0] * len(group_ids)
        for _draw in range(len(ordered)):
            selected = ordered[generator.randrange(len(ordered))]
            for index in groups[selected]:
                weights[index] += 1
        result.append(weights)
    return result


def _weighted_counts(
    images: Sequence[Mapping[str, Any]],
    bucket: str,
    confidence: float,
    weights: Sequence[int],
) -> tuple[int, int, int]:
    targets = 0
    true_positives = 0
    false_positives = 0
    bucket_names = [name for name, *_ in SIZE_BUCKETS]
    selected = bucket_names if bucket == "aggregate" else [bucket]
    for image, weight in zip(images, weights, strict=True):
        if weight <= 0:
            continue
        for name in selected:
            targets += int(image["targets"][name]) * weight
            for score, is_true_positive in image["events"][name]:
                if float(score) < confidence:
                    continue
                if bool(is_true_positive):
                    true_positives += weight
                else:
                    false_positives += weight
    return targets, true_positives, false_positives


def _interval(values: Sequence[float]) -> list[float] | None:
    return (
        [_percentile(values, 0.025), _percentile(values, 0.975)]
        if values
        else None
    )


def _source_group_uncertainty(
    images: Sequence[Mapping[str, Any]],
    source_group_ids: Sequence[str],
    *,
    confidence_thresholds: Sequence[float],
    bootstrap_samples: int,
    seed_binding: str,
) -> dict[str, Any]:
    weights = _cluster_weights(
        source_group_ids,
        samples=bootstrap_samples,
        seed_material=f"{seed_binding}:source-group-bootstrap:v1",
    )
    bucket_names = [name for name, *_ in SIZE_BUCKETS]
    operating_points: dict[str, Any] = {}
    for confidence in confidence_thresholds:
        records: dict[str, Any] = {}
        for bucket in [*bucket_names, "aggregate"]:
            recalls: list[float] = []
            precisions: list[float] = []
            false_positives: list[float] = []
            for sample_weights in weights:
                targets, true_positives, false_positive_count = _weighted_counts(
                    images, bucket, confidence, sample_weights
                )
                if targets:
                    recalls.append(true_positives / targets)
                predictions = true_positives + false_positive_count
                if predictions:
                    precisions.append(true_positives / predictions)
                false_positives.append(float(false_positive_count))
            records[bucket] = {
                "recall_samples_with_denominator": len(recalls),
                "recall_ci95": _interval(recalls),
                "precision_samples_with_denominator": len(precisions),
                "precision_ci95": _interval(precisions),
                "false_positive_count_ci95": _interval(false_positives),
            }
        operating_points[str(confidence)] = records
    ap50: dict[str, Any] = {}
    for bucket in bucket_names:
        values = [
            _pr_summary(images, bucket, sample_weights, include_curve=False)[
                "ap50_101_point_interpolated"
            ]
            for sample_weights in weights
        ]
        finite = [float(value) for value in values if value is not None]
        ap50[bucket] = {
            "samples_with_ground_truth": len(finite),
            "ci95": _interval(finite),
        }
    return {
        "method": (
            "deterministic capture-session cluster nonparametric bootstrap; all "
            "frames from a sampled source session move together"
        ),
        "source_group_definition": "capture_session",
        "source_group_count": len(set(source_group_ids)),
        "samples_requested": bootstrap_samples,
        "seed_binding_sha256": sha256(
            f"{seed_binding}:source-group-bootstrap:v1".encode("utf-8")
        ).hexdigest(),
        "operating_points": operating_points,
        "ap50": ap50,
        "caveat": (
            "Capture-session clustering accounts for within-session frame dependence; "
            "it does not prove the sessions span every future game domain."
        ),
    }


def _reviewed_negative_metrics(
    images: Sequence[Mapping[str, Any]],
    negative_mask: Sequence[bool],
    source_group_ids: Sequence[str],
    *,
    confidence_thresholds: Sequence[float],
    bootstrap_samples: int,
    seed_binding: str,
) -> dict[str, Any]:
    negative_images = [image for image, flag in zip(images, negative_mask, strict=True) if flag]
    negative_groups = [
        group for group, flag in zip(source_group_ids, negative_mask, strict=True) if flag
    ]
    if not negative_images:
        raise IndependentRuntimeEvaluationError("reviewed-negative inventory is empty")
    weights = _cluster_weights(
        negative_groups,
        samples=bootstrap_samples,
        seed_material=f"{seed_binding}:negative-source-group-bootstrap:v1",
    )
    buckets = [name for name, *_ in SIZE_BUCKETS]
    operating_points: dict[str, Any] = {}
    for confidence in confidence_thresholds:
        per_image_fp = [
            sum(
                1
                for bucket in buckets
                for score, is_true_positive in image["events"][bucket]
                if float(score) >= confidence and not bool(is_true_positive)
            )
            for image in negative_images
        ]
        false_positives = sum(per_image_fp)
        images_with_fp = sum(value > 0 for value in per_image_fp)
        fp_rates: list[float] = []
        image_rates: list[float] = []
        counts: list[float] = []
        for sample_weights in weights:
            image_total = sum(sample_weights)
            fp_total = sum(
                count * weight
                for count, weight in zip(per_image_fp, sample_weights, strict=True)
            )
            image_with_fp_total = sum(
                int(count > 0) * weight
                for count, weight in zip(per_image_fp, sample_weights, strict=True)
            )
            counts.append(float(fp_total))
            if image_total:
                fp_rates.append(fp_total / image_total)
                image_rates.append(image_with_fp_total / image_total)
        operating_points[str(confidence)] = {
            "reviewed_negative_images": len(negative_images),
            "false_positives": false_positives,
            "negative_images_with_false_positive": images_with_fp,
            "false_positives_per_image": false_positives / len(negative_images),
            "negative_image_false_positive_rate": images_with_fp / len(negative_images),
            "capture_session_cluster_bootstrap_95_ci": {
                "false_positive_count": _interval(counts),
                "false_positives_per_image": _interval(fp_rates),
                "negative_image_false_positive_rate": _interval(image_rates),
            },
        }
    return {
        "definition": (
            "All detections on independently dual-reviewed zero-box images are "
            "false positives; capture sessions are the uncertainty clusters."
        ),
        "source_group_count": len(set(negative_groups)),
        "bootstrap_samples": bootstrap_samples,
        "operating_points": operating_points,
    }


def _redacted_runtime_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    def scrub(item: Any, key: str = "") -> Any:
        if "path" in key.casefold() and isinstance(item, str):
            return "<bound-by-evidence-sha256>"
        if isinstance(item, Mapping):
            return {str(name): scrub(child, str(name)) for name, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            return [scrub(child, key) for child in item]
        return _json_safe(item)

    result = scrub(value)
    assert isinstance(result, dict)
    return result


def _decision_result(
    rule: Mapping[str, Any],
    metrics: Mapping[str, Any],
    negative_metrics: Mapping[str, Any],
    runtime_p95: float,
) -> dict[str, Any]:
    confidence = str(rule["selected_confidence_threshold"])
    buckets = metrics["size_bucket_detection"]["operating_points"][confidence]
    negative = negative_metrics["operating_points"][confidence]
    far = buckets["far_33_to_64px"]
    medium = buckets["medium_65_to_96px"]
    near = buckets["near_gt_96px"]
    ultra_far = buckets["ultra_far_le_32px"]
    gating_buckets = (far, medium, near)
    all_size_buckets = (ultra_far, far, medium, near)
    gating_ground_truth = sum(
        int(bucket["ground_truth_total"]) for bucket in gating_buckets
    )
    gating_true_positives = sum(
        int(bucket["detected_true_positives"]) for bucket in gating_buckets
    )
    all_size_predictions = sum(
        int(bucket["predictions"]) for bucket in all_size_buckets
    )
    all_size_false_positives = sum(
        int(bucket["false_positives"]) for bucket in all_size_buckets
    )
    release_precision_predictions = gating_true_positives + all_size_false_positives
    gating_precision = (
        gating_true_positives / release_precision_predictions
        if release_precision_predictions
        else None
    )
    gating_recall = (
        gating_true_positives / gating_ground_truth if gating_ground_truth else None
    )
    checks = {
        "far_recall": far["recall"] is not None
        and far["recall"] >= rule["minimum_far_recall"],
        "far_false_positives": far["false_positives"]
        <= rule["maximum_far_false_positives"],
        "medium_recall": medium["recall"] is not None
        and medium["recall"] >= rule["minimum_medium_recall"],
        "near_recall": near["recall"] is not None
        and near["recall"] >= rule["minimum_near_recall"],
        "aggregate_precision": gating_precision is not None
        and gating_precision >= rule["minimum_aggregate_precision"],
        "aggregate_recall": gating_recall is not None
        and gating_recall >= rule["minimum_aggregate_recall"],
        "reviewed_negative_false_positives": negative["false_positives"]
        <= rule["maximum_reviewed_negative_false_positives"],
        "runtime_pipeline_p95_ms": runtime_p95
        <= rule["maximum_runtime_pipeline_p95_ms"],
    }
    return {
        "frozen_rule_passed": all(checks.values()),
        "checks": checks,
        "selected_confidence_threshold": rule["selected_confidence_threshold"],
        "raw_inputs": {
            "far_detected_over_total": far["detected_over_total"],
            "far_false_positives": far["false_positives"],
            "medium_detected_over_total": medium["detected_over_total"],
            "near_detected_over_total": near["detected_over_total"],
            "gating_aggregate_detected_over_total": (
                f"{gating_true_positives}/{gating_ground_truth}"
            ),
            "all_size_predictions_observed": all_size_predictions,
            "all_size_false_positives": all_size_false_positives,
            "release_precision_denominator": release_precision_predictions,
            "aggregate_precision": gating_precision,
            "aggregate_recall": gating_recall,
            "reviewed_negative_false_positives": negative["false_positives"],
            "runtime_pipeline_p95_ms": runtime_p95,
        },
        "scope": DECISION_RESULT_SCOPE,
    }


def _release_evidence_eligibility(
    *, plan: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, bool]:
    canonical_policy = _release_policy_record()
    runtime = plan.get("runtime")
    canonical_runtime_match = (
        isinstance(runtime, Mapping)
        and runtime.get("backend") == "onnxruntime"
        and runtime.get("confidence_thresholds")
        == list(DEFAULT_CONFIDENCE_THRESHOLDS)
        and runtime.get("nms_iou_threshold") == DEFAULT_NMS_IOU
        and runtime.get("warmup_iterations") == DEFAULT_WARMUP
        and runtime.get("bootstrap_samples") == DEFAULT_BOOTSTRAP_SAMPLES
        and runtime.get("require_full_provider") is True
    )
    canonical_rule_match = (
        plan.get("release_policy") == canonical_policy
        and plan.get("decision_rule") == canonical_policy["decision_rule"]
        and canonical_runtime_match
    )
    frozen_rule_passed = decision.get("frozen_rule_passed") is True
    return {
        "canonical_release_policy_matched": canonical_rule_match,
        "frozen_metric_rule_passed": frozen_rule_passed,
        "release_evidence_eligible": canonical_rule_match and frozen_rule_passed,
    }


def _evidence_status(eligibility: Mapping[str, bool]) -> str:
    if eligibility.get("release_evidence_eligible") is True:
        return "valid_final_holdout_evidence_meeting_frozen_rule_not_release_approved"
    if eligibility.get("canonical_release_policy_matched") is not True:
        return (
            "valid_diagnostic_holdout_evidence_noncanonical_policy_"
            "not_release_approved"
        )
    return "valid_final_holdout_evidence_failed_frozen_rule_not_release_approved"


def _validate_raw_metric_point(
    value: object,
    *,
    expected_ground_truth: int,
    description: str,
) -> None:
    if not isinstance(value, Mapping):
        raise IndependentRuntimeEvaluationError(f"{description} is missing")
    integer_fields = (
        "ground_truth_total",
        "detected_true_positives",
        "missed_false_negatives",
        "predictions",
        "false_positives",
    )
    normalized = {
        field: _require_int(value.get(field), f"{description} {field}")
        for field in integer_fields
    }
    targets = normalized["ground_truth_total"]
    true_positives = normalized["detected_true_positives"]
    misses = normalized["missed_false_negatives"]
    predictions = normalized["predictions"]
    false_positives = normalized["false_positives"]
    precision = true_positives / predictions if predictions else None
    recall = true_positives / targets if targets else None
    if (
        targets != expected_ground_truth
        or true_positives + misses != targets
        or true_positives + false_positives != predictions
        or value.get("detected_over_total") != f"{true_positives}/{targets}"
        or value.get("precision") != precision
        or value.get("recall") != recall
    ):
        raise IndependentRuntimeEvaluationError(
            f"{description} raw counts/rates are internally inconsistent"
        )


def _validate_evidence_metric_inventory(
    metrics: Mapping[str, Any],
    holdout: Mapping[str, Any],
) -> None:
    if (
        metrics.get("images") != holdout["images"]
        or metrics.get("ground_truth_boxes") != holdout["boxes"]
    ):
        raise IndependentRuntimeEvaluationError(
            "evidence image/ground-truth coverage differs from the sealed holdout"
        )
    size_detection = metrics.get("size_bucket_detection")
    aggregate_detection = metrics.get("aggregate_detection")
    if not isinstance(size_detection, Mapping) or not isinstance(
        aggregate_detection, Mapping
    ):
        raise IndependentRuntimeEvaluationError(
            "evidence raw size/aggregate detection metrics are missing"
        )
    size_points = size_detection.get("operating_points")
    aggregate_points = aggregate_detection.get("operating_points")
    expected_points = {str(item) for item in DEFAULT_CONFIDENCE_THRESHOLDS}
    if (
        not isinstance(size_points, Mapping)
        or not isinstance(aggregate_points, Mapping)
        or set(size_points) != expected_points
        or set(aggregate_points) != expected_points
    ):
        raise IndependentRuntimeEvaluationError(
            "evidence confidence operating-point inventory is altered"
        )
    bucket_counts = {
        "ultra_far_le_32px": int(holdout["counts"]["target_le_32"]),
        "far_33_to_64px": int(holdout["counts"]["target_33_64"]),
        "medium_65_to_96px": int(holdout["counts"]["target_65_96"]),
        "near_gt_96px": int(holdout["counts"]["target_gt_96"]),
    }
    for confidence in sorted(expected_points):
        buckets = size_points.get(confidence)
        if not isinstance(buckets, Mapping) or set(buckets) != set(bucket_counts):
            raise IndependentRuntimeEvaluationError(
                "evidence size-bucket inventory is altered"
            )
        for name, count in bucket_counts.items():
            _validate_raw_metric_point(
                buckets[name],
                expected_ground_truth=count,
                description=f"{confidence} {name}",
            )
        _validate_raw_metric_point(
            aggregate_points[confidence],
            expected_ground_truth=int(holdout["boxes"]),
            description=f"{confidence} aggregate",
        )

    negative = metrics.get("reviewed_negative_detection")
    if not isinstance(negative, Mapping) or not isinstance(
        negative.get("operating_points"), Mapping
    ) or set(negative["operating_points"]) != expected_points:
        raise IndependentRuntimeEvaluationError(
            "reviewed-negative operating-point inventory is altered"
        )
    expected_negatives = int(holdout["counts"]["reviewed_negatives"])
    for confidence in sorted(expected_points):
        point = negative["operating_points"][confidence]
        if not isinstance(point, Mapping):
            raise IndependentRuntimeEvaluationError(
                "reviewed-negative raw metrics are missing"
            )
        false_positives = _require_int(
            point.get("false_positives"),
            "reviewed-negative false positives",
        )
        images_with_fp = _require_int(
            point.get("negative_images_with_false_positive"),
            "reviewed-negative images with false positives",
        )
        if (
            point.get("reviewed_negative_images") != expected_negatives
            or images_with_fp > expected_negatives
            or point.get("false_positives_per_image")
            != false_positives / expected_negatives
            or point.get("negative_image_false_positive_rate")
            != images_with_fp / expected_negatives
        ):
            raise IndependentRuntimeEvaluationError(
                "reviewed-negative counts/rates are internally inconsistent"
            )


def _binding_file_snapshot(binding: Mapping[str, Any]) -> dict[Path, str]:
    paths = {
        Path(binding["pointer_path"]): binding["pointer_sha256"],
        Path(binding["adoption_path"]): binding["adoption_sha256"],
        Path(binding["tournament_selection_path"]): binding[
            "tournament_selection_sha256"
        ],
        Path(binding["labels_path"]): binding["labels"]["sha256"],
        Path(binding["model_path"]): binding["model_artifacts"][0]["sha256"],
    }
    for path, digest in binding["tournament_evidence_files"]:
        paths[Path(path)] = str(digest)
    for path, digest in binding["candidate_provenance_files"]:
        paths[Path(path)] = str(digest)
    return {path: str(digest) for path, digest in paths.items()}


def _assert_unchanged(
    *,
    plan_path: Path,
    plan_sha256: str,
    dependency_manifest_path: Path,
    dependency_manifest_payload: bytes,
    environment_record: Mapping[str, Any],
    hardware_identity_path: Path,
    hardware_identity_payload: bytes,
    hardware_identity_record: Mapping[str, Any],
    project_root: Path,
    binding: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> None:
    if _sha256_file(plan_path) != plan_sha256:
        raise IndependentRuntimeEvaluationError("frozen evaluation plan changed")
    for path, expected in _binding_file_snapshot(binding).items():
        if _sha256_file(path) != expected:
            raise IndependentRuntimeEvaluationError(
                f"adopted candidate input changed during evaluation: {path.name}"
            )
    if _source_snapshot("onnxruntime") != source_snapshot:
        raise IndependentRuntimeEvaluationError(
            "evaluator or exact application pipeline source changed during evaluation"
        )
    current_path, current_payload, current_environment = _load_release_environment(
        dependency_manifest_path, project_root=project_root
    )
    if (
        current_path != dependency_manifest_path
        or current_payload != dependency_manifest_payload
        or current_environment != environment_record
    ):
        raise IndependentRuntimeEvaluationError(
            "locked Windows DirectML dependency environment changed during evaluation"
        )
    current_hardware_path, current_hardware_payload, current_hardware = (
        _load_holdout_hardware_identity(hardware_identity_path)
    )
    if (
        current_hardware_path != hardware_identity_path
        or current_hardware_payload != hardware_identity_payload
        or current_hardware != hardware_identity_record
    ):
        raise IndependentRuntimeEvaluationError(
            "RX 6950 XT hardware identity changed during evaluation"
        )


def _package_root(package: Path) -> Path:
    candidate = Path(os.path.abspath(package.expanduser()))
    if candidate.name == MANIFEST_NAME:
        candidate = candidate.parent
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            details = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise IndependentRuntimeEvaluationError(
                f"cannot inspect holdout package path component {current}: {exc}"
            ) from exc
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            raise IndependentRuntimeEvaluationError(
                f"holdout package path contains a symlink or junction: {current}"
            )
        if current != candidate and not stat.S_ISDIR(details.st_mode):
            raise IndependentRuntimeEvaluationError(
                f"holdout package ancestor is not a directory: {current}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot resolve holdout package: {candidate}: {exc}"
        ) from exc
    if not stat.S_ISDIR(candidate.stat(follow_symlinks=False).st_mode):
        raise IndependentRuntimeEvaluationError("holdout package must be a directory")
    return resolved


def evaluate_independent_holdout(
    *,
    package: Path,
    plan: Path,
    output: Path,
    dependency_manifest: Path,
    hardware_identity: Path,
    project_root: Path,
    event_id: str,
    actor_id: str,
    retirement_event_id: str,
    retirement_reason: str,
    candidate_binding_loader: Callable[[Path, str], Mapping[str, Any]] = (
        _release_candidate_binding
    ),
    detector_factory: Callable[..., Any] | None = None,
    clock: Callable[[], int] = perf_counter_ns,
    utc_now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run and retire one final holdout under the pre-frozen exact plan."""

    plan_path, plan_record, plan_payload = _load_plan(plan)
    plan_sha256 = sha256(plan_payload).hexdigest()
    (
        dependency_manifest_path,
        dependency_manifest_payload,
        environment_record,
    ) = _load_release_environment(dependency_manifest, project_root=project_root)
    if environment_record["policy"] != plan_record["environment"]:
        raise IndependentRuntimeEvaluationError(
            "evaluation environment policy differs from the frozen pre-access plan"
        )
    (
        hardware_identity_path,
        hardware_identity_payload,
        hardware_identity_record,
    ) = _load_holdout_hardware_identity(hardware_identity)
    if hardware_identity_record != plan_record["hardware_identity"]:
        raise IndependentRuntimeEvaluationError(
            "evaluation hardware differs from the frozen pre-access plan"
        )
    runtime_plan = plan_record["runtime"]
    assert isinstance(runtime_plan, Mapping)
    binding = dict(candidate_binding_loader(project_root, "onnxruntime"))
    if _public_candidate_binding(binding) != plan_record["candidate"]:
        raise IndependentRuntimeEvaluationError(
            "current adopted candidate differs from the frozen evaluation plan"
        )
    source_snapshot = _source_snapshot("onnxruntime")
    if source_snapshot != plan_record["source"]:
        raise IndependentRuntimeEvaluationError(
            "evaluator/application sources differ from the frozen plan"
        )
    package_root = _package_root(package)
    try:
        manifest, _, _ = _load_package_manifest(package_root)
    except HoldoutContractError as exc:
        raise IndependentRuntimeEvaluationError(str(exc)) from exc
    holdout_binding = _validate_sealed_manifest(manifest)
    if holdout_binding != plan_record["holdout"]:
        raise IndependentRuntimeEvaluationError(
            "sealed holdout metadata differs from the frozen evaluation plan"
        )
    inference_size: InferenceSize = (
        int(runtime_plan["input_shape_nchw"][2]),
        int(runtime_plan["input_shape_nchw"][3]),
    )
    output_dir = output.expanduser().absolute()
    metrics_path = output_dir / "metrics.json"
    snapshot_root: Path | None = None

    def run_and_publish() -> dict[str, Any]:
        nonlocal snapshot_root
        # The transaction has already verified every sealed byte while holding
        # its exclusive lock. Re-load only normalized package data from here.
        current_manifest, _, _ = _load_package_manifest(package_root)
        if _validate_sealed_manifest(current_manifest) != holdout_binding:
            raise IndependentRuntimeEvaluationError(
                "sealed holdout changed after transaction verification"
            )
        inventory, coco_sha256 = _coco_inventory(package_root, current_manifest)
        if len(inventory) != holdout_binding["images"]:
            raise IndependentRuntimeEvaluationError("sealed image coverage is incomplete")
        if sum(len(item["targets"]) for item in inventory) != holdout_binding["boxes"]:
            raise IndependentRuntimeEvaluationError("sealed target coverage is incomplete")
        if _inventory_counts(inventory) != holdout_binding["counts"]:
            raise IndependentRuntimeEvaluationError(
                "sealed runtime inventory differs from pinned manifest bucket/negative counts"
            )
        if (
            _inventory_source_groups(inventory)
            != holdout_binding["source_group_inventory"]
        ):
            raise IndependentRuntimeEvaluationError(
                "sealed runtime source groups differ from the pinned manifest"
            )

        planned_artifacts = [
            {"name": item["name"], "bytes": item["bytes"], "sha256": item["sha256"]}
            for item in binding["model_artifacts"]
        ]
        if len(planned_artifacts) != 1:
            raise IndependentRuntimeEvaluationError(
                "final ONNX workload must bind exactly one model artifact"
            )
        if snapshot_root is not None:
            raise IndependentRuntimeEvaluationError(
                "private runtime artifact snapshot was unexpectedly reused"
            )
        snapshot_root = Path(
            tempfile.mkdtemp(prefix="proaim-final-holdout-runtime-")
        )
        model_source = Path(binding["model_path"])
        labels_source = Path(binding["labels_path"])
        planned_model = planned_artifacts[0]
        model = _snapshot_exact_regular_file(
            model_source,
            snapshot_root / str(planned_model["name"]),
            expected_bytes=int(planned_model["bytes"]),
            expected_sha256=str(planned_model["sha256"]),
            description="adopted ONNX artifact",
        )
        label_record = binding["labels"]
        labels = _snapshot_exact_regular_file(
            labels_source,
            snapshot_root / Path(str(label_record["relative_path"])).name,
            expected_bytes=int(label_record["bytes"]),
            expected_sha256=str(label_record["sha256"]),
            description="adopted labels artifact",
        )
        artifact_before = _artifact_members(model, "onnxruntime")
        normalized_artifact_before = [
            {"name": item["name"], "bytes": item["bytes"], "sha256": item["sha256"]}
            for item in artifact_before
        ]
        if normalized_artifact_before != planned_artifacts:
            raise IndependentRuntimeEvaluationError(
                "runtime artifact differs from the frozen candidate binding"
            )
        _assert_unchanged(
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            dependency_manifest_path=dependency_manifest_path,
            dependency_manifest_payload=dependency_manifest_payload,
            environment_record=environment_record,
            hardware_identity_path=hardware_identity_path,
            hardware_identity_payload=hardware_identity_payload,
            hardware_identity_record=hardware_identity_record,
            project_root=project_root,
            binding=binding,
            source_snapshot=source_snapshot,
        )
        factory = detector_factory or _create_detector
        detector = factory(
            backend="onnxruntime",
            model=model,
            labels=labels,
            device=str(runtime_plan["device"]),
            inference_size=inference_size,
            confidence=MINIMUM_PREDICTION_CONFIDENCE,
            iou=float(runtime_plan["nms_iou_threshold"]),
            output_format=str(runtime_plan["output_format"]),
            require_full_provider=True,
        )
        if (
            _artifact_members(model, "onnxruntime") != artifact_before
            or labels.stat().st_size != binding["labels"]["bytes"]
            or _sha256_file(labels) != binding["labels"]["sha256"]
        ):
            raise IndependentRuntimeEvaluationError(
                "private runtime artifact snapshot changed during detector creation"
            )
        _assert_unchanged(
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            dependency_manifest_path=dependency_manifest_path,
            dependency_manifest_payload=dependency_manifest_payload,
            environment_record=environment_record,
            hardware_identity_path=hardware_identity_path,
            hardware_identity_payload=hardware_identity_payload,
            hardware_identity_record=hardware_identity_record,
            project_root=project_root,
            binding=binding,
            source_snapshot=source_snapshot,
        )
        summary_value = getattr(detector, "runtime_summary", None)
        if not isinstance(summary_value, Mapping):
            raise IndependentRuntimeEvaluationError(
                "application detector returned no runtime summary"
            )
        runtime_summary = dict(summary_value)
        _validate_runtime_binding(
            backend="onnxruntime",
            requested_device=str(runtime_plan["device"]),
            requested_output_format=str(runtime_plan["output_format"]),
            require_full_provider=True,
            runtime_summary=runtime_summary,
        )
        expected_provider = runtime_plan["expected_provider"]
        active = runtime_summary.get("active_providers")
        if (
            runtime_summary.get("requested_provider") != expected_provider
            or isinstance(active, (str, bytes))
            or not isinstance(active, Sequence)
            or expected_provider not in active
            or expected_provider == "CPUExecutionProvider"
        ):
            raise IndependentRuntimeEvaluationError(
                "runtime did not activate the exact accelerator provider in the plan"
            )
        declared_shape = _validated_static_shape(
            backend="onnxruntime",
            model_path=model,
            runtime_summary=runtime_summary,
            inference_size=inference_size,
        )
        detector.warmup(int(runtime_plan["warmup_iterations"]))

        evidence: list[dict[str, Any]] = []
        primary_evidence: list[dict[str, Any]] = []
        source_group_ids: list[str] = []
        member_records: list[dict[str, Any]] = []
        negative_mask: list[bool] = []
        detail_crop_size = int(runtime_plan["detail_crop_size_source_pixels"])
        detail_enabled = detail_crop_size > 0
        adopted_pipeline = str(plan_record["candidate"]["selected_pipeline"])
        if detail_enabled != (adopted_pipeline == "configured"):
            raise IndependentRuntimeEvaluationError(
                "frozen runtime workload differs from the adopted tournament pipeline"
            )
        detail_stats = DetailPassStats(
            detail_crop_size if detail_enabled else None
        )
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
        observed_shape: list[int] | None = None
        observed_dtype: str | None = None
        for item in inventory:
            decode_started = clock()
            frame = _load_exact_image(
                item["path"],
                expected_sha256=item["sha256"],
                width=item["width"],
                height=item["height"],
            )
            decoded = clock()
            preprocessing_started = clock()
            prepared = preprocess_frame(frame, inference_size=inference_size)
            preprocessed = clock()
            raw = np.asarray(detector.infer(prepared.tensor))
            inferred = clock()
            detections = detector.postprocess(
                raw, transform=prepared.transform, frame_shape=frame.shape
            )
            primary_ready = clock()
            current_shape = list(raw.shape)
            current_dtype = str(raw.dtype)
            if observed_shape is None:
                observed_shape, observed_dtype = current_shape, current_dtype
            elif current_shape != observed_shape or current_dtype != observed_dtype:
                raise IndependentRuntimeEvaluationError(
                    "runtime output shape or dtype changed between calls"
                )
            primary_predictions = _prediction_boxes(detections, frame.shape)
            primary_evidence.append(
                bucket_image_evidence(item["targets"], primary_predictions)
            )

            detail_preprocess_ms = 0.0
            detail_inference_ms = 0.0
            detail_postprocess_ms = 0.0
            detections_ready = primary_ready
            if detail_enabled:
                detail_plan = plan_detail_pass(
                    frame.shape, detail_crop_size, inference_size
                )
                detail_stats.record(detail_plan)
            else:
                detail_plan = None
            if detail_plan is not None and not detail_plan.redundant:
                detail_started = clock()
                detail_prepared = preprocess_frame(
                    frame,
                    inference_size=inference_size,
                    crop_size=(
                        detail_plan.applied_crop_height,
                        detail_plan.applied_crop_width,
                    ),
                )
                detail_prepared_at = clock()
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
                if list(detail_raw.shape) != observed_shape or str(
                    detail_raw.dtype
                ) != observed_dtype:
                    raise IndependentRuntimeEvaluationError(
                        "detail runtime output contract differs from primary"
                    )
                _validate_detail_preprocessing(detail_prepared, detail_plan)
                detail_preprocess_ms = (detail_prepared_at - detail_started) / 1_000_000.0
                detail_inference_ms = (detail_inferred - detail_prepared_at) / 1_000_000.0
                detail_postprocess_ms = (detections_ready - detail_inferred) / 1_000_000.0

            predictions = _prediction_boxes(detections, frame.shape)
            evidence.append(bucket_image_evidence(item["targets"], predictions))
            group_id = canonical_hash(
                {
                    "schema_version": 1,
                    "holdout_manifest_content_sha256": holdout_binding[
                        "manifest_content_sha256"
                    ],
                    "capture_session_id": item["session_id"],
                }
            )
            source_group_ids.append(group_id)
            member_records.append(
                {
                    "image_sha256": item["sha256"],
                    "label_sha256": item["target_identity_sha256"],
                }
            )
            negative_mask.append(bool(item["reviewed_negative"]))
            timings["decode"].append((decoded - decode_started) / 1_000_000.0)
            timings["preprocess"].append((preprocessed - preprocessing_started) / 1_000_000.0)
            timings["inference"].append((inferred - preprocessed) / 1_000_000.0)
            timings["postprocess"].append((primary_ready - inferred) / 1_000_000.0)
            timings["detail_preprocess"].append(detail_preprocess_ms)
            timings["detail_inference"].append(detail_inference_ms)
            timings["detail_postprocess"].append(detail_postprocess_ms)
            timings["runtime_pipeline"].append(
                (detections_ready - preprocessing_started) / 1_000_000.0
            )

        if len(evidence) != holdout_binding["images"] or sum(
            sum(int(count) for count in image["targets"].values()) for image in evidence
        ) != holdout_binding["boxes"]:
            raise IndependentRuntimeEvaluationError(
                "runtime evidence omitted sealed images or targets"
            )
        thresholds = tuple(float(item) for item in runtime_plan["confidence_thresholds"])
        bootstrap_samples = int(runtime_plan["bootstrap_samples"])
        metrics = _metric_record(
            evidence,
            confidence_thresholds=thresholds,
            bootstrap_samples=bootstrap_samples,
        )
        primary_metrics = _metric_record(
            primary_evidence,
            confidence_thresholds=thresholds,
            bootstrap_samples=bootstrap_samples,
        )
        split_content_sha256 = canonical_hash(
            [
                {
                    "image_sha256": member["image_sha256"],
                    "label_sha256": member["label_sha256"],
                }
                for member in member_records
            ]
        )
        paired = paired_image_operating_point_evidence(
            evidence,
            member_records,
            confidence_thresholds=thresholds,
            split_content_sha256=split_content_sha256,
            source_group_ids=source_group_ids,
        )
        primary_paired = paired_image_operating_point_evidence(
            primary_evidence,
            member_records,
            confidence_thresholds=thresholds,
            split_content_sha256=split_content_sha256,
            source_group_ids=source_group_ids,
        )
        uncertainty = _source_group_uncertainty(
            evidence,
            source_group_ids,
            confidence_thresholds=thresholds,
            bootstrap_samples=bootstrap_samples,
            seed_binding=plan_record["plan_content_sha256"],
        )
        negative_metrics = _reviewed_negative_metrics(
            evidence,
            negative_mask,
            source_group_ids,
            confidence_thresholds=thresholds,
            bootstrap_samples=bootstrap_samples,
            seed_binding=plan_record["plan_content_sha256"],
        )
        timing_summary = {
            name: _timing_summary(values) for name, values in timings.items()
        }
        detail_snapshot = detail_stats.snapshot()
        if detail_enabled and detail_snapshot.get("frames_applied", 0) <= 0:
            raise IndependentRuntimeEvaluationError(
                "detail pass was redundant for every sealed image; final evidence "
                "does not exercise the configured production detail pipeline"
            )
        decision = _decision_result(
            plan_record["decision_rule"],
            metrics,
            negative_metrics,
            timing_summary["runtime_pipeline"]["p95"],
        )
        eligibility = _release_evidence_eligibility(
            plan=plan_record, decision=decision
        )
        try:
            final_verification = verify_holdout(
                package_root, access_mode="curator", _allow_ledger_lock=True
            )
        except HoldoutContractError as exc:
            raise IndependentRuntimeEvaluationError(
                f"sealed package changed during runtime evaluation: {exc}"
            ) from exc
        if (
            final_verification["manifest_content_sha256"]
            != holdout_binding["manifest_content_sha256"]
            or _artifact_members(model, "onnxruntime") != artifact_before
            or labels.stat().st_size != binding["labels"]["bytes"]
            or _sha256_file(labels) != binding["labels"]["sha256"]
        ):
            raise IndependentRuntimeEvaluationError(
                "sealed package or runtime artifact changed during evaluation"
            )
        _assert_unchanged(
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            dependency_manifest_path=dependency_manifest_path,
            dependency_manifest_payload=dependency_manifest_payload,
            environment_record=environment_record,
            hardware_identity_path=hardware_identity_path,
            hardware_identity_payload=hardware_identity_payload,
            hardware_identity_record=hardware_identity_record,
            project_root=project_root,
            binding=binding,
            source_snapshot=source_snapshot,
        )
        record: dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "kind": EVIDENCE_KIND,
            "status": _evidence_status(eligibility),
            "evaluation_plan": {
                "sha256": plan_sha256,
                "content_sha256": plan_record["plan_content_sha256"],
                "frozen_decision_rule": plan_record["decision_rule"],
            },
            "release_policy": plan_record["release_policy"],
            "candidate": plan_record["candidate"],
            "model_artifact": {
                "backend": "onnxruntime",
                "content_sha256": binding["model_content_sha256"],
                "members": planned_artifacts,
            },
            "holdout": {
                **holdout_binding,
                "normalized_coco_sha256": coco_sha256,
                "exact_member_verification_before_and_after_inference": True,
                "ground_truth_source": "sealed normalized COCO; no grouped-v9 YAML/split",
            },
            "configuration": {
                **dict(runtime_plan),
                "evaluation_mode": "sealed_independent_exact_application_runtime_artifact",
                "adopted_tournament_pipeline": adopted_pipeline,
                "configured_pipeline": (
                    "rectangular_full_frame_plus_center_model_aspect_detail_merged"
                    if detail_enabled
                    else "rectangular_full_frame_primary_only"
                ),
                "primary_reference_retained": True,
                "detail_merge": (
                    "detection.detail_pass.merge_cross_pass_detections exactly once "
                    "for every non-redundant detail frame"
                    if detail_enabled
                    else "disabled by the adopted primary-only tournament workload"
                ),
                "detail_stats": detail_snapshot,
            },
            "runtime": {
                "summary": _redacted_runtime_summary(runtime_summary),
                "declared_static_input_shape_nchw": declared_shape,
                "observed_raw_output_shape": observed_shape,
                "observed_raw_output_dtype": observed_dtype,
                "timing_ms_per_image": timing_summary,
                "timing_scope": (
                    "Synchronous batch-one local measurement. runtime_pipeline covers "
                    + (
                        "primary plus non-redundant detail preprocessing, inference, "
                        "postprocessing, and merge; "
                        if detail_enabled
                        else "the adopted primary-only preprocessing, inference, and "
                        "postprocessing; "
                    )
                    + "image decode is reported separately."
                ),
            },
            "metrics": {
                "images": len(evidence),
                "ground_truth_boxes": holdout_binding["boxes"],
                **metrics,
                "paired_image_operating_points": paired,
                "capture_session_cluster_uncertainty": uncertainty,
                "reviewed_negative_detection": negative_metrics,
                "primary_full_frame_reference": {
                    **primary_metrics,
                    "paired_image_operating_points": primary_paired,
                },
            },
            "decision_rule_result": decision,
            "one_time_access": {
                "event_id": event_id,
                "actor_id": actor_id,
                "purpose": EVALUATION_PURPOSE,
                "retirement_event_id": retirement_event_id,
                "retirement_reason": retirement_reason,
                "timestamp_authority": (
                    "UTC transition times are generated inside the exclusive "
                    "ledger transaction: consumption before first sealed-member "
                    "read and retirement after durable evidence publication"
                ),
                "publication_order": (
                    "durable pre-access consumption, atomic evidence publication, "
                    "then evidence-hash-bound retirement while the exclusive lock "
                    "remains held"
                ),
            },
            "environment": environment_record,
            "hardware_identity": hardware_identity_record,
            "source": source_snapshot,
            "qualification": {
                **QUALIFICATION_RECORD,
                "final_holdout_evaluation_completed": True,
                **eligibility,
                "comparative_incumbent_improvement_proven": False,
                "manual_release_review_required": True,
                "hardware_gate_passed": False,
                "frozen_build_gate_passed": False,
                "legal_redistribution_gate_passed": False,
                "release_gate_passed": False,
                "reason": (
                    "This is release-eligible evidence for manual review only. It "
                    "cannot approve a release or satisfy separate physical GPU, "
                    "frozen-build, and legal-distribution gates."
                ),
        },
    }
        _reject_private_path_strings(record, "independent runtime evidence")
        record["evidence_content_sha256"] = _evidence_content_hash(record)
        _assert_unchanged(
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            dependency_manifest_path=dependency_manifest_path,
            dependency_manifest_payload=dependency_manifest_payload,
            environment_record=environment_record,
            hardware_identity_path=hardware_identity_path,
            hardware_identity_payload=hardware_identity_payload,
            hardware_identity_record=hardware_identity_record,
            project_root=project_root,
            binding=binding,
            source_snapshot=source_snapshot,
        )
        _publish_record(output_dir, record)
        return record

    try:
        try:
            record, ledger = complete_sealed_evaluation(
                package_root,
                evidence_path=metrics_path,
                publish_evaluation=run_and_publish,
                event_id=event_id,
                actor_id=actor_id,
                purpose=EVALUATION_PURPOSE,
                evaluation_plan_sha256=plan_sha256,
                retirement_event_id=retirement_event_id,
                retirement_reason=retirement_reason,
                utc_now=utc_now,
            )
        except IndependentRuntimeEvaluationError:
            raise
        except Exception as exc:
            raise IndependentRuntimeEvaluationError(
                f"final sealed runtime evaluation failed closed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return record, ledger
    finally:
        if snapshot_root is not None:
            try:
                shutil.rmtree(snapshot_root)
            except OSError as exc:
                raise IndependentRuntimeEvaluationError(
                    "cannot remove private runtime artifact snapshot after the "
                    f"sealed transaction: {exc}"
                ) from exc


def verify_independent_evidence(
    *,
    evidence: Path,
    plan: Path,
    package: Path,
    dependency_manifest: Path,
    hardware_identity: Path,
    project_root: Path,
    candidate_binding_loader: Callable[[Path, str], Mapping[str, Any]] = (
        _release_candidate_binding
    ),
    enforce_current_environment_match: bool = True,
) -> dict[str, Any]:
    """Verify evidence, frozen rule, adopted candidate, and retired ledger chain."""

    evidence_input = evidence.expanduser().absolute()
    if evidence_input.is_dir():
        evidence_root = evidence_input
        evidence_input = evidence_root / "metrics.json"
    else:
        evidence_root = evidence_input.parent
    try:
        children = list(evidence_root.iterdir())
    except OSError as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot inspect evidence directory: {exc}"
        ) from exc
    if len(children) != 1 or children[0].name != "metrics.json":
        raise IndependentRuntimeEvaluationError(
            "evidence directory must contain exactly metrics.json"
        )
    evidence_path, record, evidence_payload = _strict_json_file(
        evidence_input, "independent runtime evidence"
    )
    expected_top_level = {
        "schema_version",
        "kind",
        "status",
        "evaluation_plan",
        "release_policy",
        "candidate",
        "model_artifact",
        "holdout",
        "configuration",
        "runtime",
        "metrics",
        "decision_rule_result",
        "one_time_access",
        "environment",
        "hardware_identity",
        "source",
        "qualification",
        "evidence_content_sha256",
    }
    if (
        set(record) != expected_top_level
        or record.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or record.get("kind") != EVIDENCE_KIND
        or record.get("evidence_content_sha256") != _evidence_content_hash(record)
    ):
        raise IndependentRuntimeEvaluationError(
            "independent evidence schema or content hash is invalid"
        )
    _reject_private_path_strings(record, "independent runtime evidence")
    plan_path, plan_record, plan_payload = _load_plan(plan)
    plan_sha256 = sha256(plan_payload).hexdigest()
    (
        dependency_manifest_path,
        dependency_manifest_payload,
        environment_record,
    ) = _load_release_environment(dependency_manifest, project_root=project_root)
    evidence_environment = record.get("environment")
    try:
        validated_evidence_environment = (
            _shared_validate_release_environment_record(
                evidence_environment, project_root=project_root
            )
            if isinstance(evidence_environment, Mapping)
            else None
        )
    except IndependentHoldoutReleaseContractError as exc:
        raise IndependentRuntimeEvaluationError(
            f"evidence runtime environment record is invalid: {exc}"
        ) from exc
    if (
        validated_evidence_environment is None
        or plan_record.get("environment")
        != validated_evidence_environment["policy"]
        or (
            enforce_current_environment_match
            and validated_evidence_environment != environment_record
        )
    ):
        raise IndependentRuntimeEvaluationError(
            "evidence runtime environment differs from its plan/current exact lock"
        )
    (
        hardware_identity_path,
        hardware_identity_payload,
        hardware_identity_record,
    ) = _load_holdout_hardware_identity(hardware_identity)
    if (
        plan_record.get("hardware_identity") != hardware_identity_record
        or record.get("hardware_identity") != hardware_identity_record
    ):
        raise IndependentRuntimeEvaluationError(
            "evidence/plan hardware identity differs from the pre-access RX 6950 XT record"
        )
    plan_binding = record.get("evaluation_plan")
    if not isinstance(plan_binding, Mapping) or plan_binding != {
        "sha256": plan_sha256,
        "content_sha256": plan_record["plan_content_sha256"],
        "frozen_decision_rule": plan_record["decision_rule"],
    }:
        raise IndependentRuntimeEvaluationError(
            "evidence does not bind the exact frozen evaluation plan"
        )
    if (
        record.get("release_policy") != _release_policy_record()
        or record.get("release_policy") != plan_record.get("release_policy")
    ):
        raise IndependentRuntimeEvaluationError(
            "evidence release policy version/hash differs from the frozen plan"
        )

    binding = dict(candidate_binding_loader(project_root, "onnxruntime"))
    if (
        record.get("candidate") != _public_candidate_binding(binding)
        or record.get("candidate") != plan_record["candidate"]
    ):
        raise IndependentRuntimeEvaluationError(
            "evidence candidate differs from the plan/current adopted candidate"
        )
    model_artifact = record.get("model_artifact")
    planned_members = [
        {"name": item["name"], "bytes": item["bytes"], "sha256": item["sha256"]}
        for item in binding["model_artifacts"]
    ]
    if model_artifact != {
        "backend": "onnxruntime",
        "content_sha256": binding["model_content_sha256"],
        "members": planned_members,
    }:
        raise IndependentRuntimeEvaluationError(
            "evidence model artifact differs from the adopted candidate"
        )
    source_snapshot = _source_snapshot("onnxruntime")
    if record.get("source") != source_snapshot or plan_record.get("source") != source_snapshot:
        raise IndependentRuntimeEvaluationError(
            "evidence/plan sources differ from this exact application checkout"
        )
    _assert_unchanged(
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        dependency_manifest_path=dependency_manifest_path,
        dependency_manifest_payload=dependency_manifest_payload,
        environment_record=environment_record,
        hardware_identity_path=hardware_identity_path,
        hardware_identity_payload=hardware_identity_payload,
        hardware_identity_record=hardware_identity_record,
        project_root=project_root,
        binding=binding,
        source_snapshot=source_snapshot,
    )

    package_root = _package_root(package)
    try:
        verification = verify_holdout(package_root, access_mode="curator")
        manifest, _, _ = _load_package_manifest(package_root)
        events = _ledger_events(
            package_root,
            expected_manifest_sha256=manifest["manifest_content_sha256"],
        )
    except HoldoutContractError as exc:
        raise IndependentRuntimeEvaluationError(
            f"retired sealed holdout or ledger is invalid: {exc}"
        ) from exc
    holdout_binding = _validate_sealed_manifest(manifest)
    evidence_holdout = record.get("holdout")
    annotations_record = manifest.get("annotations")
    expected_evidence_holdout = {
        **holdout_binding,
        "normalized_coco_sha256": (
            annotations_record.get("sha256")
            if isinstance(annotations_record, Mapping)
            else None
        ),
        "exact_member_verification_before_and_after_inference": True,
        "ground_truth_source": "sealed normalized COCO; no grouped-v9 YAML/split",
    }
    if evidence_holdout != expected_evidence_holdout:
        raise IndependentRuntimeEvaluationError(
            "evidence holdout binding differs from the sealed package"
        )
    if plan_record.get("holdout") != holdout_binding:
        raise IndependentRuntimeEvaluationError(
            "frozen plan identifies a different sealed holdout"
        )
    if not verification.get("retired") or len(events) != 2:
        raise IndependentRuntimeEvaluationError(
            "sealed holdout was not consumed exactly once and retired"
        )
    consumed, retired = events
    evidence_sha256 = sha256(evidence_payload).hexdigest()
    if (
        consumed.get("schema_version") != 1
        or consumed.get("operation") != "consumed"
        or consumed.get("evaluation_plan_sha256") != plan_sha256
        or "evaluation_evidence_sha256" in consumed
        or retired.get("schema_version") != 2
        or retired.get("operation") != "retired"
        or retired.get("evaluation_evidence_sha256") != evidence_sha256
        or retired.get("previous_event_sha256")
        != consumed.get("event_content_sha256")
    ):
        raise IndependentRuntimeEvaluationError(
            "ledger does not bind this exact plan and published evidence"
        )
    consumption_event_path = (
        package_root
        / "access-ledger"
        / f"00000001-{consumed['event_id']}.json"
    )
    retirement_event_path = (
        package_root
        / "access-ledger"
        / f"00000002-{retired['event_id']}.json"
    )
    consumed_file, consumed_payload = _load_holdout_json(
        consumption_event_path, "sealed holdout consumption ledger event"
    )
    retired_file, retired_payload = _load_holdout_json(
        retirement_event_path, "sealed holdout retirement ledger event"
    )
    if (
        not isinstance(consumed_file, Mapping)
        or not isinstance(retired_file, Mapping)
        or consumed_payload != _canonical_holdout_json_bytes(consumed_file)
        or retired_payload != _canonical_holdout_json_bytes(retired_file)
    ):
        raise IndependentRuntimeEvaluationError(
            "ledger event files must use the exact canonical holdout encoding"
        )
    if consumed_file != consumed or retired_file != retired:
        raise IndependentRuntimeEvaluationError(
            "ledger event files differ from the verified one-time chain"
        )
    access = record.get("one_time_access")
    expected_access = {
        "event_id": consumed["event_id"],
        "actor_id": consumed["actor_id"],
        "purpose": consumed["purpose"],
        "retirement_event_id": retired["event_id"],
        "retirement_reason": retired["reason"],
        "timestamp_authority": (
            "UTC transition times are generated inside the exclusive ledger "
            "transaction: consumption before first sealed-member read and "
            "retirement after durable evidence publication"
        ),
        "publication_order": (
            "durable pre-access consumption, atomic evidence publication, then "
            "evidence-hash-bound retirement while the exclusive lock remains held"
        ),
    }
    if not isinstance(access, Mapping) or access != expected_access:
        raise IndependentRuntimeEvaluationError(
            "evidence access declaration differs from the retired ledger"
        )

    metrics = record.get("metrics")
    runtime = record.get("runtime")
    configuration = record.get("configuration")
    if (
        not isinstance(metrics, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(configuration, Mapping)
    ):
        raise IndependentRuntimeEvaluationError("evidence metrics/runtime are incomplete")
    for key, value in plan_record["runtime"].items():
        if configuration.get(key) != value:
            raise IndependentRuntimeEvaluationError(
                "evidence runtime configuration differs from the frozen plan"
            )
    selected_pipeline = plan_record["candidate"]["selected_pipeline"]
    detail_enabled = plan_record["runtime"]["detail_crop_size_source_pixels"] > 0
    expected_pipeline = (
        "rectangular_full_frame_plus_center_model_aspect_detail_merged"
        if detail_enabled
        else "rectangular_full_frame_primary_only"
    )
    detail_stats = configuration.get("detail_stats")
    if (
        configuration.get("evaluation_mode")
        != "sealed_independent_exact_application_runtime_artifact"
        or configuration.get("adopted_tournament_pipeline") != selected_pipeline
        or configuration.get("configured_pipeline") != expected_pipeline
        or configuration.get("primary_reference_retained") is not True
        or not isinstance(detail_stats, Mapping)
        or detail_stats.get("enabled") is not detail_enabled
        or (
            detail_enabled
            and (
                detail_stats.get("frames_seen") != holdout_binding["images"]
                or not isinstance(detail_stats.get("frames_applied"), int)
                or detail_stats["frames_applied"] <= 0
            )
        )
        or (
            not detail_enabled
            and (
                detail_stats.get("frames_seen") != 0
                or detail_stats.get("frames_applied") != 0
                or detail_stats.get("last_plan") is not None
            )
        )
    ):
        raise IndependentRuntimeEvaluationError(
            "evidence did not execute the exact adopted primary/detail workload"
        )

    runtime_summary = runtime.get("summary")
    if not isinstance(runtime_summary, Mapping):
        raise IndependentRuntimeEvaluationError("runtime startup summary is missing")
    try:
        _validate_runtime_binding(
            backend="onnxruntime",
            requested_device=str(plan_record["runtime"]["device"]),
            requested_output_format=str(plan_record["runtime"]["output_format"]),
            require_full_provider=True,
            runtime_summary=runtime_summary,
        )
        declared_shape = _validated_static_shape(
            backend="onnxruntime",
            model_path=Path(binding["model_path"]),
            runtime_summary=runtime_summary,
            inference_size=(
                int(plan_record["runtime"]["input_shape_nchw"][2]),
                int(plan_record["runtime"]["input_shape_nchw"][3]),
            ),
        )
    except Exception as exc:
        raise IndependentRuntimeEvaluationError(
            f"evidence runtime provider/shape contract is invalid: {exc}"
        ) from exc
    expected_provider = plan_record["runtime"]["expected_provider"]
    active_providers = runtime_summary.get("active_providers")
    if (
        runtime_summary.get("requested_provider") != expected_provider
        or isinstance(active_providers, (str, bytes))
        or not isinstance(active_providers, Sequence)
        or expected_provider not in active_providers
        or runtime.get("declared_static_input_shape_nchw") != declared_shape
    ):
        raise IndependentRuntimeEvaluationError(
            "evidence does not prove the exact planned accelerator provider/shape"
        )
    _validate_evidence_metric_inventory(metrics, holdout_binding)
    timing = runtime.get("timing_ms_per_image")
    negative = metrics.get("reviewed_negative_detection")
    if not isinstance(timing, Mapping) or not isinstance(negative, Mapping):
        raise IndependentRuntimeEvaluationError(
            "evidence timing/negative metrics are incomplete"
        )
    pipeline_timing = timing.get("runtime_pipeline")
    if not isinstance(pipeline_timing, Mapping):
        raise IndependentRuntimeEvaluationError("runtime pipeline timing is missing")
    recomputed_decision = _decision_result(
        plan_record["decision_rule"],
        metrics,
        negative,
        _positive_float(pipeline_timing.get("p95"), "runtime pipeline p95"),
    )
    if record.get("decision_rule_result") != recomputed_decision:
        raise IndependentRuntimeEvaluationError(
            "machine-verifiable frozen decision result was altered"
        )
    eligibility = _release_evidence_eligibility(
        plan=plan_record, decision=recomputed_decision
    )
    rule_passed = eligibility["frozen_metric_rule_passed"]
    release_evidence_eligible = eligibility["release_evidence_eligible"]
    expected_status = _evidence_status(eligibility)
    if record.get("status") != expected_status:
        raise IndependentRuntimeEvaluationError("evidence status disagrees with its rule")
    expected_qualification = {
        **QUALIFICATION_RECORD,
        "final_holdout_evaluation_completed": True,
        **eligibility,
        "comparative_incumbent_improvement_proven": False,
        "manual_release_review_required": True,
        "hardware_gate_passed": False,
        "frozen_build_gate_passed": False,
        "legal_redistribution_gate_passed": False,
        "release_gate_passed": False,
        "reason": (
            "This is release-eligible evidence for manual review only. It "
            "cannot approve a release or satisfy separate physical GPU, "
            "frozen-build, and legal-distribution gates."
        ),
    }
    if record.get("qualification") != expected_qualification:
        raise IndependentRuntimeEvaluationError(
            "evidence qualification flags are altered or overstate approval"
        )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "status": (
            "verified_release_eligible_evidence_not_release_approved"
            if release_evidence_eligible
            else "verified_valid_evidence_not_release_eligible_or_approved"
        ),
        "claim_scope": "absolute_threshold_evidence_only_no_incumbent_comparison",
        "release_policy": _release_policy_record(),
        "environment": validated_evidence_environment,
        "hardware_identity": hardware_identity_record,
        "verifier": _receipt_verifier_record(source_snapshot),
        "evidence": {
            "name": "metrics.json",
            "bytes": len(evidence_payload),
            "sha256": evidence_sha256,
            "content_sha256": record["evidence_content_sha256"],
        },
        "evaluation_plan": {
            "bytes": len(plan_payload),
            "sha256": plan_sha256,
            "content_sha256": plan_record["plan_content_sha256"],
        },
        "holdout": {
            "package_id": holdout_binding["package_id"],
            "manifest_content_sha256": holdout_binding[
                "manifest_content_sha256"
            ],
            "normalized_coco_sha256": expected_evidence_holdout[
                "normalized_coco_sha256"
            ],
            "counts": holdout_binding["counts"],
            "source_group_inventory": holdout_binding[
                "source_group_inventory"
            ],
        },
        "candidate": {
            "release_default_pointer_sha256": binding["pointer_sha256"],
            "release_default_pointer_content_sha256": binding[
                "pointer_content_sha256"
            ],
            "candidate_content_sha256": binding["candidate_content_sha256"],
            "candidate_manifest_sha256": binding["candidate_manifest_sha256"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "adoption_sha256": binding["adoption_sha256"],
            "adoption_content_sha256": binding["adoption_content_sha256"],
            "adoption_evidence_replay_sha256": binding[
                "adoption_evidence_replay_sha256"
            ],
            "tournament_selection_sha256": binding[
                "tournament_selection_sha256"
            ],
            "tournament_selection_content_sha256": binding[
                "tournament_selection_content_sha256"
            ],
            "winner_slot": binding["winner_slot"],
            "model_content_sha256": binding["model_content_sha256"],
            "labels_sha256": binding["labels"]["sha256"],
            "tournament_evidence_inventory_sha256": canonical_hash(
                binding["tournament_evidence"]
            ),
            "candidate_provenance_inventory_sha256": canonical_hash(
                binding["candidate_provenance_evidence"]
            ),
        },
        "workload": {
            "backend": "onnxruntime",
            "expected_provider": plan_record["runtime"]["expected_provider"],
            "input_shape_nchw": plan_record["runtime"]["input_shape_nchw"],
            "output_format": plan_record["runtime"]["output_format"],
            "selected_pipeline": binding["selected_pipeline"],
            "detail_crop_size_source_pixels": binding[
                "detail_crop_size_source_pixels"
            ],
            "confidence_thresholds": plan_record["runtime"][
                "confidence_thresholds"
            ],
            "nms_iou_threshold": plan_record["runtime"]["nms_iou_threshold"],
            "bootstrap_samples": plan_record["runtime"]["bootstrap_samples"],
            "warmup_iterations": plan_record["runtime"]["warmup_iterations"],
            "require_full_provider": True,
            "runtime_pipeline_p95_ms": pipeline_timing["p95"],
        },
        "decision": {
            "rule": plan_record["decision_rule"],
            "result_sha256": canonical_hash(recomputed_decision),
            "result": recomputed_decision,
        },
        "one_time_ledger": {
            "event_count": 2,
            "consumed_exactly_once": True,
            "retired": True,
            "consumption_event": {
                "name": BUNDLE_MEMBER_NAMES["consumption_event"],
                "bytes": len(consumed_payload),
                "sha256": sha256(consumed_payload).hexdigest(),
                "event_id": consumed["event_id"],
                "recorded_at_utc": consumed["recorded_at_utc"],
                "event_content_sha256": consumed["event_content_sha256"],
                "evaluation_plan_sha256": consumed["evaluation_plan_sha256"],
            },
            "retirement_event": {
                "name": BUNDLE_MEMBER_NAMES["retirement_event"],
                "bytes": len(retired_payload),
                "sha256": sha256(retired_payload).hexdigest(),
                "event_id": retired["event_id"],
                "recorded_at_utc": retired["recorded_at_utc"],
                "event_content_sha256": retired["event_content_sha256"],
                "previous_event_sha256": retired["previous_event_sha256"],
                "evaluation_evidence_sha256": retired[
                    "evaluation_evidence_sha256"
                ],
            },
        },
        "canonical_release_policy_matched": eligibility[
            "canonical_release_policy_matched"
        ],
        "frozen_metric_rule_passed": rule_passed,
        "release_evidence_eligible": release_evidence_eligible,
        "release_approved": False,
        "release_pointer_changed": False,
        "manual_release_review_required": True,
        "separate_hardware_frozen_build_and_legal_gates_required": True,
        "qualification": {
            **QUALIFICATION_RECORD,
            "hardware_gate_passed": False,
            "frozen_build_gate_passed": False,
            "legal_redistribution_gate_passed": False,
            "release_gate_passed": False,
            "comparative_incumbent_improvement_proven": False,
        },
    }
    receipt["receipt_content_sha256"] = _final_receipt_content_hash(receipt)
    _validate_portable_receipt(receipt)
    return receipt


def _receipt_fraction(value: object, description: str) -> tuple[int, int]:
    if not isinstance(value, str) or re.fullmatch(r"\d+/[1-9]\d*", value) is None:
        raise IndependentRuntimeEvaluationError(
            f"{description} must be an exact detected/total count"
        )
    detected_text, total_text = value.split("/", 1)
    detected, total = int(detected_text), int(total_text)
    if detected > total:
        raise IndependentRuntimeEvaluationError(
            f"{description} detected count exceeds total"
        )
    return detected, total


def _validate_receipt_decision(value: Mapping[str, Any]) -> tuple[bool, bool]:
    if set(value) != {"rule", "result", "result_sha256"}:
        raise IndependentRuntimeEvaluationError("receipt decision schema is invalid")
    rule_value = value.get("rule")
    result = value.get("result")
    if not isinstance(rule_value, Mapping) or not isinstance(result, Mapping):
        raise IndependentRuntimeEvaluationError("receipt decision records are incomplete")
    rule = _validate_decision_rule(rule_value)
    if (
        set(result)
        != {
            "frozen_rule_passed",
            "checks",
            "selected_confidence_threshold",
            "raw_inputs",
            "scope",
        }
        or result.get("selected_confidence_threshold")
        != rule["selected_confidence_threshold"]
        or result.get("scope") != DECISION_RESULT_SCOPE
        or value.get("result_sha256") != canonical_hash(result)
    ):
        raise IndependentRuntimeEvaluationError(
            "receipt decision result binding is invalid"
        )
    checks = result.get("checks")
    raw = result.get("raw_inputs")
    expected_check_keys = {
        "far_recall",
        "far_false_positives",
        "medium_recall",
        "near_recall",
        "aggregate_precision",
        "aggregate_recall",
        "reviewed_negative_false_positives",
        "runtime_pipeline_p95_ms",
    }
    expected_raw_keys = {
        "far_detected_over_total",
        "far_false_positives",
        "medium_detected_over_total",
        "near_detected_over_total",
        "gating_aggregate_detected_over_total",
        "all_size_predictions_observed",
        "all_size_false_positives",
        "release_precision_denominator",
        "aggregate_precision",
        "aggregate_recall",
        "reviewed_negative_false_positives",
        "runtime_pipeline_p95_ms",
    }
    if (
        not isinstance(checks, Mapping)
        or set(checks) != expected_check_keys
        or not all(isinstance(item, bool) for item in checks.values())
        or not isinstance(raw, Mapping)
        or set(raw) != expected_raw_keys
    ):
        raise IndependentRuntimeEvaluationError(
            "receipt decision checks/raw-input schema is invalid"
        )
    far_detected, far_total = _receipt_fraction(
        raw["far_detected_over_total"], "receipt far detected/total"
    )
    medium_detected, medium_total = _receipt_fraction(
        raw["medium_detected_over_total"], "receipt medium detected/total"
    )
    near_detected, near_total = _receipt_fraction(
        raw["near_detected_over_total"], "receipt near detected/total"
    )
    aggregate_detected, aggregate_total = _receipt_fraction(
        raw["gating_aggregate_detected_over_total"],
        "receipt aggregate detected/total",
    )
    far_false_positives = _require_int(
        raw["far_false_positives"], "receipt far false positives"
    )
    all_predictions = _require_int(
        raw["all_size_predictions_observed"], "receipt all-size predictions"
    )
    all_false_positives = _require_int(
        raw["all_size_false_positives"], "receipt all-size false positives"
    )
    precision_denominator = _require_int(
        raw["release_precision_denominator"],
        "receipt release-precision denominator",
    )
    reviewed_negative_false_positives = _require_int(
        raw["reviewed_negative_false_positives"],
        "receipt reviewed-negative false positives",
    )
    runtime_p95 = _positive_float(
        raw["runtime_pipeline_p95_ms"], "receipt runtime pipeline p95"
    )
    raw_precision = raw["aggregate_precision"]
    aggregate_precision = (
        None
        if raw_precision is None
        else _probability(raw_precision, "receipt aggregate precision")
    )
    aggregate_recall = _probability(
        raw["aggregate_recall"], "receipt aggregate recall"
    )
    if (
        aggregate_detected
        != far_detected + medium_detected + near_detected
        or aggregate_total != far_total + medium_total + near_total
        or precision_denominator != aggregate_detected + all_false_positives
        or all_false_positives > all_predictions
        or far_false_positives > all_false_positives
        or reviewed_negative_false_positives > all_false_positives
        or precision_denominator > all_predictions
        or (
            aggregate_precision
            != (
                aggregate_detected / precision_denominator
                if precision_denominator
                else None
            )
        )
        or aggregate_recall != aggregate_detected / aggregate_total
    ):
        raise IndependentRuntimeEvaluationError(
            "receipt decision raw counts/rates are internally inconsistent"
        )
    expected_checks = {
        "far_recall": far_detected / far_total >= rule["minimum_far_recall"],
        "far_false_positives": far_false_positives
        <= rule["maximum_far_false_positives"],
        "medium_recall": medium_detected / medium_total
        >= rule["minimum_medium_recall"],
        "near_recall": near_detected / near_total
        >= rule["minimum_near_recall"],
        "aggregate_precision": aggregate_precision is not None
        and aggregate_precision >= rule["minimum_aggregate_precision"],
        "aggregate_recall": aggregate_recall
        >= rule["minimum_aggregate_recall"],
        "reviewed_negative_false_positives": reviewed_negative_false_positives
        <= rule["maximum_reviewed_negative_false_positives"],
        "runtime_pipeline_p95_ms": runtime_p95
        <= rule["maximum_runtime_pipeline_p95_ms"],
    }
    passed = all(expected_checks.values())
    if checks != expected_checks or result.get("frozen_rule_passed") is not passed:
        raise IndependentRuntimeEvaluationError(
            "receipt decision booleans differ from its exact counts/rule"
        )
    return passed, rule == CANONICAL_RELEASE_DECISION_RULE


def _validate_portable_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "status",
        "claim_scope",
        "release_policy",
        "environment",
        "hardware_identity",
        "verifier",
        "evidence",
        "evaluation_plan",
        "holdout",
        "candidate",
        "workload",
        "decision",
        "one_time_ledger",
        "canonical_release_policy_matched",
        "frozen_metric_rule_passed",
        "release_evidence_eligible",
        "release_approved",
        "release_pointer_changed",
        "manual_release_review_required",
        "separate_hardware_frozen_build_and_legal_gates_required",
        "qualification",
        "receipt_content_sha256",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or value.get("kind") != RECEIPT_KIND
        or value.get("claim_scope")
        != "absolute_threshold_evidence_only_no_incumbent_comparison"
        or value.get("release_policy") != _release_policy_record()
        or value.get("receipt_content_sha256")
        != _final_receipt_content_hash(value)
    ):
        raise IndependentRuntimeEvaluationError(
            "portable final-holdout receipt schema/policy/self-hash is invalid"
        )
    _reject_private_path_strings(value, "portable final-holdout receipt")
    evidence = value.get("evidence")
    environment = value.get("environment")
    hardware_identity = value.get("hardware_identity")
    verifier = value.get("verifier")
    plan = value.get("evaluation_plan")
    holdout = value.get("holdout")
    candidate = value.get("candidate")
    workload = value.get("workload")
    decision = value.get("decision")
    ledger = value.get("one_time_ledger")
    qualification = value.get("qualification")
    if not all(
        isinstance(item, Mapping)
        for item in (
            evidence,
            environment,
            hardware_identity,
            verifier,
            plan,
            holdout,
            candidate,
            workload,
            decision,
            ledger,
            qualification,
        )
    ):
        raise IndependentRuntimeEvaluationError(
            "portable final-holdout receipt mappings are incomplete"
        )
    assert isinstance(evidence, Mapping)
    assert isinstance(environment, Mapping)
    assert isinstance(hardware_identity, Mapping)
    assert isinstance(verifier, Mapping)
    assert isinstance(plan, Mapping)
    assert isinstance(holdout, Mapping)
    assert isinstance(candidate, Mapping)
    assert isinstance(workload, Mapping)
    assert isinstance(decision, Mapping)
    assert isinstance(ledger, Mapping)
    assert isinstance(qualification, Mapping)
    try:
        _shared_validate_release_environment_record(
            environment, project_root=PROJECT_ROOT
        )
    except IndependentHoldoutReleaseContractError as exc:
        raise IndependentRuntimeEvaluationError(
            f"portable receipt runtime environment binding is invalid: {exc}"
        ) from exc
    try:
        _shared_validate_holdout_hardware_identity(hardware_identity)
    except IndependentHoldoutReleaseContractError as exc:
        raise IndependentRuntimeEvaluationError(
            f"portable receipt hardware identity binding is invalid: {exc}"
        ) from exc
    expected_verifier = _receipt_verifier_record(_source_snapshot("onnxruntime"))
    if verifier != expected_verifier:
        raise IndependentRuntimeEvaluationError(
            "receipt verifier/evaluator/source snapshot binding is invalid"
        )
    if set(evidence) != {"name", "bytes", "sha256", "content_sha256"} or (
        evidence.get("name") != "metrics.json"
        or _require_int(evidence.get("bytes"), "receipt evidence bytes", minimum=1)
        <= 0
    ):
        raise IndependentRuntimeEvaluationError("receipt evidence record is invalid")
    for record, fields, description in (
        (
            evidence,
            ("sha256", "content_sha256"),
            "receipt evidence",
        ),
        (
            plan,
            ("sha256", "content_sha256"),
            "receipt evaluation plan",
        ),
    ):
        for field in fields:
            _require_sha256(record.get(field), f"{description} {field}")
    if set(plan) != {"bytes", "sha256", "content_sha256"}:
        raise IndependentRuntimeEvaluationError("receipt plan record is invalid")
    _require_int(plan.get("bytes"), "receipt plan bytes", minimum=1)
    expected_candidate_keys = {
        "release_default_pointer_sha256",
        "release_default_pointer_content_sha256",
        "candidate_content_sha256",
        "candidate_manifest_sha256",
        "checkpoint_sha256",
        "adoption_sha256",
        "adoption_content_sha256",
        "adoption_evidence_replay_sha256",
        "tournament_selection_sha256",
        "tournament_selection_content_sha256",
        "winner_slot",
        "model_content_sha256",
        "labels_sha256",
        "tournament_evidence_inventory_sha256",
        "candidate_provenance_inventory_sha256",
    }
    if (
        set(candidate) != expected_candidate_keys
        or candidate.get("winner_slot") not in TOURNAMENT_SLOT_NAMES
    ):
        raise IndependentRuntimeEvaluationError("receipt candidate record is invalid")
    for field in expected_candidate_keys - {"winner_slot"}:
        _require_sha256(candidate.get(field), f"receipt candidate {field}")
    counts = holdout.get("counts")
    source_groups = holdout.get("source_group_inventory")
    if (
        set(holdout)
        != {
            "package_id",
            "manifest_content_sha256",
            "normalized_coco_sha256",
            "counts",
            "source_group_inventory",
        }
        or not isinstance(holdout.get("package_id"), str)
        or not isinstance(counts, Mapping)
        or not isinstance(source_groups, Mapping)
    ):
        raise IndependentRuntimeEvaluationError("receipt holdout record is invalid")
    _require_sha256(
        holdout.get("manifest_content_sha256"), "receipt holdout manifest hash"
    )
    _require_sha256(
        holdout.get("normalized_coco_sha256"), "receipt normalized COCO hash"
    )
    expected_count_keys = {
        "target_le_32",
        "target_33_64",
        "target_65_96",
        "target_gt_96",
        "reviewed_negatives",
    }
    if set(counts) != expected_count_keys:
        raise IndependentRuntimeEvaluationError(
            "receipt holdout count inventory schema is invalid"
        )
    for key in expected_count_keys:
        _require_int(counts.get(key), f"receipt holdout {key}")
    if any(
        _require_int(counts.get(key), f"receipt holdout {key}") < minimum
        for key, minimum in PINNED_RELEASE_MINIMUMS.as_dict().items()
        if key in GATING_COUNT_KEYS
    ):
        raise IndependentRuntimeEvaluationError(
            "receipt holdout count inventory is below release policy"
        )
    target_sessions = source_groups.get("target_bearing_capture_sessions")
    if isinstance(target_sessions, Mapping):
        for key in ("target_le_32", "target_33_64", "target_65_96", "target_gt_96"):
            _require_int(
                target_sessions.get(key), f"receipt {key} capture sessions"
            )
    if (
        set(source_groups)
        != {
            "definition",
            "overall_capture_sessions",
            "target_bearing_capture_sessions",
            "reviewed_negative_capture_sessions",
        }
        or source_groups.get("definition")
        != "distinct normalized COCO image session_id values"
        or _require_int(
            source_groups.get("overall_capture_sessions"),
            "receipt overall capture sessions",
        )
        < MINIMUM_CAPTURE_SESSIONS
        or not isinstance(target_sessions, Mapping)
        or set(target_sessions)
        != {"target_le_32", "target_33_64", "target_65_96", "target_gt_96"}
        or any(
            _require_int(
                target_sessions.get(key), f"receipt {key} capture sessions"
            )
            < MINIMUM_TARGET_BUCKET_CAPTURE_SESSIONS
            for key in ("target_33_64", "target_65_96", "target_gt_96")
        )
        or _require_int(
            source_groups.get("reviewed_negative_capture_sessions"),
            "receipt reviewed-negative capture sessions",
        )
        < MINIMUM_REVIEWED_NEGATIVE_CAPTURE_SESSIONS
    ):
        raise IndependentRuntimeEvaluationError(
            "receipt holdout source-group inventory is below release policy"
        )
    expected_workload_keys = {
        "backend",
        "expected_provider",
        "input_shape_nchw",
        "output_format",
        "selected_pipeline",
        "detail_crop_size_source_pixels",
        "confidence_thresholds",
        "nms_iou_threshold",
        "bootstrap_samples",
        "warmup_iterations",
        "require_full_provider",
        "runtime_pipeline_p95_ms",
    }
    shape = workload.get("input_shape_nchw")
    detail_width = _require_int(
        workload.get("detail_crop_size_source_pixels"),
        "receipt detail crop width",
    )
    if (
        set(workload) != expected_workload_keys
        or workload.get("backend") != "onnxruntime"
        or workload.get("expected_provider") not in SUPPORTED_ACCELERATOR_PROVIDERS
        or not isinstance(shape, list)
        or len(shape) != 4
        or shape[0:2] != [1, 3]
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            for item in shape
        )
        or shape[2] % 32 != 0
        or shape[3] % 32 != 0
        or shape[2] == shape[3]
        or workload.get("output_format") not in {"end2end", "traditional"}
        or workload.get("selected_pipeline") not in {"primary", "configured"}
        or (
            workload.get("selected_pipeline") == "primary" and detail_width != 0
        )
        or (
            workload.get("selected_pipeline") == "configured" and detail_width <= 0
        )
        or workload.get("confidence_thresholds")
        != list(DEFAULT_CONFIDENCE_THRESHOLDS)
        or workload.get("nms_iou_threshold") != DEFAULT_NMS_IOU
        or workload.get("bootstrap_samples") != DEFAULT_BOOTSTRAP_SAMPLES
        or workload.get("warmup_iterations") != DEFAULT_WARMUP
        or workload.get("require_full_provider") is not True
    ):
        raise IndependentRuntimeEvaluationError("receipt workload record is invalid")
    _positive_float(
        workload.get("runtime_pipeline_p95_ms"), "receipt runtime pipeline p95"
    )
    frozen_passed, decision_uses_canonical_rule = _validate_receipt_decision(decision)
    canonical_policy_matched = value.get("canonical_release_policy_matched") is True
    if canonical_policy_matched is not decision_uses_canonical_rule:
        raise IndependentRuntimeEvaluationError(
            "receipt canonical-policy flag differs from the exact frozen rule"
        )
    eligible = value.get("release_evidence_eligible") is True
    expected_status = (
        "verified_release_eligible_evidence_not_release_approved"
        if eligible
        else "verified_valid_evidence_not_release_eligible_or_approved"
    )
    if (
        value.get("frozen_metric_rule_passed") is not frozen_passed
        or eligible is not (frozen_passed and canonical_policy_matched)
        or value.get("status") != expected_status
        or value.get("release_approved") is not False
        or value.get("release_pointer_changed") is not False
        or value.get("manual_release_review_required") is not True
        or value.get("separate_hardware_frozen_build_and_legal_gates_required")
        is not True
    ):
        raise IndependentRuntimeEvaluationError(
            "receipt eligibility/status flags are inconsistent or unsafe"
        )
    consumption_event = ledger.get("consumption_event")
    retirement_event = ledger.get("retirement_event")
    if (
        set(ledger)
        != {
            "event_count",
            "consumed_exactly_once",
            "retired",
            "consumption_event",
            "retirement_event",
        }
        or ledger.get("event_count") != 2
        or ledger.get("consumed_exactly_once") is not True
        or ledger.get("retired") is not True
        or not isinstance(consumption_event, Mapping)
        or not isinstance(retirement_event, Mapping)
    ):
        raise IndependentRuntimeEvaluationError("receipt one-time ledger proof is invalid")
    assert isinstance(consumption_event, Mapping)
    assert isinstance(retirement_event, Mapping)
    if (
        set(consumption_event)
        != {
            "name",
            "bytes",
            "sha256",
            "event_id",
            "recorded_at_utc",
            "event_content_sha256",
            "evaluation_plan_sha256",
        }
        or set(retirement_event)
        != {
            "name",
            "bytes",
            "sha256",
            "event_id",
            "recorded_at_utc",
            "event_content_sha256",
            "previous_event_sha256",
            "evaluation_evidence_sha256",
        }
        or consumption_event.get("name")
        != BUNDLE_MEMBER_NAMES["consumption_event"]
        or retirement_event.get("name")
        != BUNDLE_MEMBER_NAMES["retirement_event"]
        or consumption_event.get("evaluation_plan_sha256") != plan.get("sha256")
        or retirement_event.get("evaluation_evidence_sha256")
        != evidence.get("sha256")
        or retirement_event.get("previous_event_sha256")
        != consumption_event.get("event_content_sha256")
    ):
        raise IndependentRuntimeEvaluationError(
            "receipt ledger event records do not bind the plan/evidence chain"
        )
    event_times: list[datetime] = []
    for event, description in (
        (consumption_event, "consumption"),
        (retirement_event, "retirement"),
    ):
        _require_int(event.get("bytes"), f"receipt {description} event bytes", minimum=1)
        for field in ("sha256", "event_content_sha256"):
            _require_sha256(
                event.get(field), f"receipt {description} event {field}"
            )
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", event_id
        ) is None:
            raise IndependentRuntimeEvaluationError(
                f"receipt {description} event id is invalid"
            )
        timestamp = event.get("recorded_at_utc")
        if (
            not isinstance(timestamp, str)
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
                timestamp,
            )
            is None
        ):
            raise IndependentRuntimeEvaluationError(
                f"receipt {description} event time is invalid"
            )
        event_times.append(datetime.fromisoformat(timestamp.replace("Z", "+00:00")))
    if consumption_event.get("event_id") == retirement_event.get("event_id"):
        raise IndependentRuntimeEvaluationError(
            "receipt ledger event ids must be distinct"
        )
    if event_times[1] <= event_times[0]:
        raise IndependentRuntimeEvaluationError(
            "receipt retirement time must follow pre-access consumption"
        )
    _require_sha256(
        consumption_event.get("evaluation_plan_sha256"),
        "receipt consumption event evaluation plan hash",
    )
    _require_sha256(
        retirement_event.get("previous_event_sha256"),
        "receipt retirement previous event hash",
    )
    _require_sha256(
        retirement_event.get("evaluation_evidence_sha256"),
        "receipt retirement evaluation evidence hash",
    )
    expected_qualification = {
        **QUALIFICATION_RECORD,
        "hardware_gate_passed": False,
        "frozen_build_gate_passed": False,
        "legal_redistribution_gate_passed": False,
        "release_gate_passed": False,
        "comparative_incumbent_improvement_proven": False,
    }
    if qualification != expected_qualification:
        raise IndependentRuntimeEvaluationError(
            "receipt qualification flags overstate release approval"
        )
    return dict(value)


def write_independent_evidence_receipt(
    path: Path, receipt: Mapping[str, Any]
) -> Path:
    """Atomically publish one canonical portable receipt without replacement."""

    normalized = _validate_portable_receipt(receipt)
    payload = canonical_json_bytes(normalized)
    output, parent_identity = _safe_new_output_path(
        path, "portable receipt output"
    )
    if os.path.lexists(output):
        raise IndependentRuntimeEvaluationError(
            f"refusing to overwrite portable receipt: {output}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.receipt-", dir=output.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise IndependentRuntimeEvaluationError(
                f"receipt output appeared during publication: {output}"
            ) from exc
        except OSError as exc:
            raise IndependentRuntimeEvaluationError(
                "atomic no-replace portable receipt publication is unavailable: "
                f"{exc}"
            ) from exc
        published = True
        if output.read_bytes() != payload:
            raise IndependentRuntimeEvaluationError(
                "published portable receipt bytes differ from the verified record"
            )
        _require_same_output_parent(output.parent, parent_identity)
        _fsync_holdout_directory(output.parent)
        return output.resolve()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _safe_new_output_path(
    value: Path, description: str
) -> tuple[Path, tuple[int, int]]:
    output = value.expanduser().absolute()
    current = Path(output.anchor)
    for part in output.parent.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise IndependentRuntimeEvaluationError(
                f"{description} parent contains a symlink: {current}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    current = Path(output.anchor)
    for part in output.parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise IndependentRuntimeEvaluationError(
                f"{description} parent became a symlink: {current}"
            )
    try:
        details = output.parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise IndependentRuntimeEvaluationError(
            f"cannot inspect {description} parent: {exc}"
        ) from exc
    if not stat.S_ISDIR(details.st_mode):
        raise IndependentRuntimeEvaluationError(
            f"{description} parent must be a real directory"
        )
    return output, (details.st_dev, details.st_ino)


def _require_same_output_parent(
    parent: Path, expected_identity: tuple[int, int]
) -> None:
    try:
        details = parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise IndependentRuntimeEvaluationError(
            f"publication output parent changed: {exc}"
        ) from exc
    if (
        parent.is_symlink()
        or not stat.S_ISDIR(details.st_mode)
        or (details.st_dev, details.st_ino) != expected_identity
    ):
        raise IndependentRuntimeEvaluationError(
            "publication output parent identity changed during staging"
        )


def verify_independent_evidence_receipt(
    *,
    receipt: Path,
    evidence: Path,
    plan: Path,
    package: Path,
    dependency_manifest: Path,
    hardware_identity: Path,
    project_root: Path,
    candidate_binding_loader: Callable[[Path, str], Mapping[str, Any]] = (
        _release_candidate_binding
    ),
    enforce_current_environment_match: bool = True,
) -> dict[str, Any]:
    """Rebuild and compare a portable receipt from all authoritative inputs."""

    expected = verify_independent_evidence(
        evidence=evidence,
        plan=plan,
        package=package,
        dependency_manifest=dependency_manifest,
        hardware_identity=hardware_identity,
        project_root=project_root,
        candidate_binding_loader=candidate_binding_loader,
        enforce_current_environment_match=enforce_current_environment_match,
    )
    _receipt_path, supplied, payload = _strict_json_file(
        receipt, "portable final-holdout receipt"
    )
    _validate_portable_receipt(supplied)
    if supplied != expected:
        raise IndependentRuntimeEvaluationError(
            "portable receipt differs from authoritative evidence/plan/pointer/ledger"
        )
    return {
        "schema_version": 1,
        "status": "verified_portable_final_holdout_receipt",
        "receipt_sha256": sha256(payload).hexdigest(),
        "receipt_content_sha256": supplied["receipt_content_sha256"],
        "release_policy_sha256": supplied["release_policy"]["policy_sha256"],
        "canonical_release_policy_matched": supplied[
            "canonical_release_policy_matched"
        ],
        "release_evidence_eligible": supplied["release_evidence_eligible"],
        "release_approved": False,
        "release_pointer_changed": False,
        "consumed_exactly_once": True,
        "retired": True,
    }


def _bundle_content_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("bundle_content_sha256", None)
    return canonical_hash(body)


def _bundle_manifest(
    *, receipt: Mapping[str, Any], members: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    ledger = receipt["one_time_ledger"]
    assert isinstance(ledger, Mapping)
    consumption = ledger["consumption_event"]
    retirement = ledger["retirement_event"]
    assert isinstance(consumption, Mapping)
    assert isinstance(retirement, Mapping)
    value: dict[str, Any] = {
        "schema_version": 1,
        "kind": BUNDLE_KIND,
        "status": (
            "verified_release_eligible_publication_inputs_not_release_approved"
            if receipt["release_evidence_eligible"] is True
            else "verified_diagnostic_publication_inputs_not_release_eligible_or_approved"
        ),
        "members": {name: dict(members[name]) for name in sorted(members)},
        "bindings": {
            "receipt_content_sha256": receipt["receipt_content_sha256"],
            "release_policy_sha256": receipt["release_policy"]["policy_sha256"],
            "source_snapshot_sha256": receipt["verifier"][
                "source_snapshot_sha256"
            ],
            "evidence_sha256": receipt["evidence"]["sha256"],
            "evaluation_plan_sha256": receipt["evaluation_plan"]["sha256"],
            "candidate_binding_sha256": canonical_hash(receipt["candidate"]),
            "holdout_binding_sha256": canonical_hash(receipt["holdout"]),
            "environment_record_sha256": receipt["environment"][
                "record_content_sha256"
            ],
            "hardware_identity_sha256": receipt["hardware_identity"][
                "content_sha256"
            ],
            "consumption_event_content_sha256": consumption[
                "event_content_sha256"
            ],
            "retirement_event_content_sha256": retirement[
                "event_content_sha256"
            ],
        },
        "canonical_release_policy_matched": receipt[
            "canonical_release_policy_matched"
        ],
        "release_evidence_eligible": receipt["release_evidence_eligible"],
        "authenticated_origin_required": True,
        "release_approved": False,
        "release_pointer_changed": False,
        "qualification": dict(QUALIFICATION_RECORD),
    }
    value["bundle_content_sha256"] = _bundle_content_hash(value)
    return value


def _validate_bundle_manifest(
    value: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "members",
        "bindings",
        "canonical_release_policy_matched",
        "release_evidence_eligible",
        "authenticated_origin_required",
        "release_approved",
        "release_pointer_changed",
        "qualification",
        "bundle_content_sha256",
    }
    members = value.get("members")
    bindings = value.get("bindings")
    expected_status = (
        "verified_release_eligible_publication_inputs_not_release_approved"
        if receipt["release_evidence_eligible"] is True
        else "verified_diagnostic_publication_inputs_not_release_eligible_or_approved"
    )
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("kind") != BUNDLE_KIND
        or value.get("status") != expected_status
        or value.get("bundle_content_sha256") != _bundle_content_hash(value)
        or not isinstance(members, Mapping)
        or set(members) != set(BUNDLE_MEMBER_NAMES)
        or not isinstance(bindings, Mapping)
        or value.get("canonical_release_policy_matched")
        is not receipt["canonical_release_policy_matched"]
        or value.get("release_evidence_eligible")
        is not receipt["release_evidence_eligible"]
        or value.get("authenticated_origin_required") is not True
        or value.get("release_approved") is not False
        or value.get("release_pointer_changed") is not False
        or value.get("qualification") != QUALIFICATION_RECORD
    ):
        raise IndependentRuntimeEvaluationError(
            "independent holdout publication-bundle manifest is invalid"
        )
    assert isinstance(members, Mapping)
    assert isinstance(bindings, Mapping)
    for role, expected_path in BUNDLE_MEMBER_NAMES.items():
        record = members.get(role)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "bytes", "sha256"}
            or record.get("path") != expected_path
            or _require_int(
                record.get("bytes"), f"bundle {role} bytes", minimum=1
            )
            <= 0
        ):
            raise IndependentRuntimeEvaluationError(
                f"publication bundle {role} member is invalid"
            )
        _require_sha256(record.get("sha256"), f"bundle {role} hash")
    expected_bindings = {
        "receipt_content_sha256": receipt["receipt_content_sha256"],
        "release_policy_sha256": receipt["release_policy"]["policy_sha256"],
        "source_snapshot_sha256": receipt["verifier"]["source_snapshot_sha256"],
        "evidence_sha256": receipt["evidence"]["sha256"],
        "evaluation_plan_sha256": receipt["evaluation_plan"]["sha256"],
        "candidate_binding_sha256": canonical_hash(receipt["candidate"]),
        "holdout_binding_sha256": canonical_hash(receipt["holdout"]),
        "environment_record_sha256": receipt["environment"][
            "record_content_sha256"
        ],
        "hardware_identity_sha256": receipt["hardware_identity"][
            "content_sha256"
        ],
        "consumption_event_content_sha256": receipt["one_time_ledger"]
        ["consumption_event"]["event_content_sha256"],
        "retirement_event_content_sha256": receipt["one_time_ledger"]
        ["retirement_event"]["event_content_sha256"],
    }
    if bindings != expected_bindings:
        raise IndependentRuntimeEvaluationError(
            "publication bundle bindings differ from its exact receipt"
        )
    _reject_private_path_strings(value, "publication bundle manifest")
    return dict(value)


def _bundle_member_sources(
    *, receipt_path: Path, evidence: Path, plan: Path, package: Path
) -> dict[str, Path]:
    _, receipt, _ = _strict_json_file(receipt_path, "portable final-holdout receipt")
    _validate_portable_receipt(receipt)
    evidence_input = evidence.expanduser().absolute()
    if evidence_input.is_dir():
        evidence_input = evidence_input / "metrics.json"
    ledger = receipt["one_time_ledger"]
    assert isinstance(ledger, Mapping)
    consumption = ledger["consumption_event"]
    retirement = ledger["retirement_event"]
    assert isinstance(consumption, Mapping)
    assert isinstance(retirement, Mapping)
    package_root = _package_root(package)
    return {
        "receipt": receipt_path,
        "evidence": evidence_input,
        "evaluation_plan": plan,
        "consumption_event": package_root
        / "access-ledger"
        / f"00000001-{consumption['event_id']}.json",
        "retirement_event": package_root
        / "access-ledger"
        / f"00000002-{retirement['event_id']}.json",
    }


def publish_independent_evidence_bundle(
    *,
    output: Path,
    receipt: Path,
    evidence: Path,
    plan: Path,
    package: Path,
    dependency_manifest: Path,
    hardware_identity: Path,
    project_root: Path,
    candidate_binding_loader: Callable[[Path, str], Mapping[str, Any]] = (
        _release_candidate_binding
    ),
    enforce_current_environment_match: bool = True,
) -> Path:
    """Publish the fixed public workflow input inventory atomically/no-replace."""

    verify_independent_evidence_receipt(
        receipt=receipt,
        evidence=evidence,
        plan=plan,
        package=package,
        dependency_manifest=dependency_manifest,
        hardware_identity=hardware_identity,
        project_root=project_root,
        candidate_binding_loader=candidate_binding_loader,
        enforce_current_environment_match=enforce_current_environment_match,
    )
    receipt_path, receipt_value, receipt_payload = _strict_json_file(
        receipt, "portable final-holdout receipt"
    )
    _validate_portable_receipt(receipt_value)
    sources = _bundle_member_sources(
        receipt_path=receipt_path, evidence=evidence, plan=plan, package=package
    )
    payloads: dict[str, bytes] = {}
    member_records: dict[str, dict[str, Any]] = {}
    for role in BUNDLE_MEMBER_NAMES:
        source = _regular_file(sources[role], f"publication bundle {role} source")
        try:
            payload = source.read_bytes()
        except OSError as exc:
            raise IndependentRuntimeEvaluationError(
                f"cannot read publication bundle {role}: {exc}"
            ) from exc
        payloads[role] = payload
        if source.suffix.casefold() == ".json":
            public_value = _parse_json_object_payload(
                payload, f"publication bundle {role} source"
            )
            _reject_private_path_strings(
                public_value, f"publication bundle {role} source"
            )
        member_records[role] = {
            "path": BUNDLE_MEMBER_NAMES[role],
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
    if payloads["receipt"] != receipt_payload:
        raise IndependentRuntimeEvaluationError(
            "portable receipt changed while staging the publication bundle"
        )
    expected_member_links = {
        "evidence": receipt_value["evidence"],
        "evaluation_plan": receipt_value["evaluation_plan"],
        "consumption_event": receipt_value["one_time_ledger"]["consumption_event"],
        "retirement_event": receipt_value["one_time_ledger"]["retirement_event"],
    }
    for role, bound in expected_member_links.items():
        if (
            member_records[role]["bytes"] != bound["bytes"]
            or member_records[role]["sha256"] != bound["sha256"]
        ):
            raise IndependentRuntimeEvaluationError(
                f"publication bundle {role} differs from the portable receipt"
            )
    manifest = _bundle_manifest(receipt=receipt_value, members=member_records)
    _validate_bundle_manifest(manifest, receipt_value)
    output_path, parent_identity = _safe_new_output_path(
        output, "publication bundle output"
    )
    if os.path.lexists(output_path):
        raise IndependentRuntimeEvaluationError(
            f"refusing to overwrite publication bundle: {output_path}"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.bundle-", dir=output_path.parent)
    )
    published = False
    try:
        (staging / "ledger").mkdir(mode=0o700)
        for role, destination_name in BUNDLE_MEMBER_NAMES.items():
            destination = staging.joinpath(*PurePosixPath(destination_name).parts)
            with destination.open("xb") as target:
                target.write(payloads[role])
                target.flush()
                os.fsync(target.fileno())
        manifest_path = staging / BUNDLE_MANIFEST_NAME
        with manifest_path.open("xb") as target:
            target.write(canonical_json_bytes(manifest))
            target.flush()
            os.fsync(target.fileno())
        _fsync_holdout_directory(staging / "ledger")
        _fsync_holdout_directory(staging)
        _require_same_output_parent(output_path.parent, parent_identity)
        try:
            _rename_directory_noreplace(staging, output_path)
        except Exception as exc:
            raise IndependentRuntimeEvaluationError(
                f"atomic no-replace publication bundle failed: {exc}"
            ) from exc
        published = True
        _require_same_output_parent(output_path.parent, parent_identity)
        _fsync_holdout_directory(output_path.parent)
        return output_path.resolve()
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def verify_independent_evidence_bundle(
    *,
    bundle: Path,
    evidence: Path,
    plan: Path,
    package: Path,
    dependency_manifest: Path,
    hardware_identity: Path,
    project_root: Path,
    candidate_binding_loader: Callable[[Path, str], Mapping[str, Any]] = (
        _release_candidate_binding
    ),
    enforce_current_environment_match: bool = True,
) -> dict[str, Any]:
    """Verify fixed bundle bytes plus all authoritative private/source bindings."""

    root = bundle.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise IndependentRuntimeEvaluationError(
            "independent holdout publication bundle must be a real directory"
        )
    expected_files = {BUNDLE_MANIFEST_NAME, *BUNDLE_MEMBER_NAMES.values()}
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for child in root.rglob("*"):
        relative = child.relative_to(root).as_posix()
        if child.is_symlink():
            raise IndependentRuntimeEvaluationError(
                f"publication bundle member is a symlink: {relative}"
            )
        if child.is_file():
            actual_files.add(relative)
        elif child.is_dir():
            actual_directories.add(relative)
        else:
            raise IndependentRuntimeEvaluationError(
                f"publication bundle contains an unsafe member: {relative}"
            )
    if actual_files != expected_files or actual_directories != {"ledger"}:
        raise IndependentRuntimeEvaluationError(
            "publication bundle filesystem inventory is incomplete or unexpected"
        )
    receipt_path = root / BUNDLE_MEMBER_NAMES["receipt"]
    authoritative = verify_independent_evidence_receipt(
        receipt=receipt_path,
        evidence=evidence,
        plan=plan,
        package=package,
        dependency_manifest=dependency_manifest,
        hardware_identity=hardware_identity,
        project_root=project_root,
        candidate_binding_loader=candidate_binding_loader,
        enforce_current_environment_match=enforce_current_environment_match,
    )
    _, receipt_value, _ = _strict_json_file(
        receipt_path, "bundled portable final-holdout receipt"
    )
    _, manifest, manifest_payload = _strict_json_file(
        root / BUNDLE_MANIFEST_NAME, "publication bundle manifest"
    )
    _validate_bundle_manifest(manifest, receipt_value)
    members = manifest["members"]
    for role, record in members.items():
        member = _regular_file(
            root.joinpath(*PurePosixPath(record["path"]).parts),
            f"publication bundle {role}",
        )
        if (
            member.stat().st_size != record["bytes"]
            or _sha256_file(member) != record["sha256"]
        ):
            raise IndependentRuntimeEvaluationError(
                f"publication bundle {role} bytes/hash differ from its manifest"
            )
    return {
        "schema_version": 1,
        "status": "verified_independent_holdout_publication_input_bundle",
        "bundle_manifest_sha256": sha256(manifest_payload).hexdigest(),
        "bundle_content_sha256": manifest["bundle_content_sha256"],
        "receipt_content_sha256": receipt_value["receipt_content_sha256"],
        "release_policy_sha256": receipt_value["release_policy"]["policy_sha256"],
        "source_snapshot_sha256": receipt_value["verifier"][
            "source_snapshot_sha256"
        ],
        "environment_record_sha256": receipt_value["environment"][
            "record_content_sha256"
        ],
        "hardware_identity_sha256": receipt_value["hardware_identity"][
            "content_sha256"
        ],
        "canonical_release_policy_matched": authoritative[
            "canonical_release_policy_matched"
        ],
        "release_evidence_eligible": authoritative["release_evidence_eligible"],
        "authenticated_origin_required": True,
        "release_approved": False,
        "release_pointer_changed": False,
        "consumed_exactly_once": True,
        "retired": True,
    }


def _decision_rule_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "proaim-independent-holdout-frozen-decision-rule",
        "selected_confidence_threshold": args.selected_confidence,
        "minimum_far_recall": args.minimum_far_recall,
        "maximum_far_false_positives": args.maximum_far_false_positives,
        "minimum_medium_recall": args.minimum_medium_recall,
        "minimum_near_recall": args.minimum_near_recall,
        "minimum_aggregate_precision": args.minimum_aggregate_precision,
        "minimum_aggregate_recall": args.minimum_aggregate_recall,
        "maximum_reviewed_negative_false_positives": (
            args.maximum_reviewed_negative_false_positives
        ),
        "maximum_runtime_pipeline_p95_ms": args.maximum_runtime_pipeline_p95_ms,
        "manual_review_note": args.manual_review_note,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="freeze a final plan without opening sealed members")
    plan.add_argument("--package", type=Path, required=True)
    plan.add_argument("--dependency-manifest", type=Path, required=True)
    plan.add_argument("--hardware-identity", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    plan.add_argument("--device", required=True)
    plan.add_argument(
        "--expected-provider", choices=sorted(SUPPORTED_ACCELERATOR_PROVIDERS), required=True
    )
    plan.add_argument(
        "--detail-crop-size",
        type=int,
        required=True,
        help=(
            "Assertion matching the adopted workload: 0 for primary-only or "
            "the exact positive production detail ROI width."
        ),
    )
    plan.add_argument(
        "--output-format",
        choices=("end2end", "traditional"),
        help="Optional assertion; defaults to the adopted candidate's bound output head.",
    )
    plan.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    plan.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    plan.add_argument("--selected-confidence", type=float, default=0.25)
    plan.add_argument(
        "--minimum-far-recall",
        type=float,
        default=CANONICAL_RELEASE_DECISION_RULE["minimum_far_recall"],
    )
    plan.add_argument(
        "--maximum-far-false-positives",
        type=int,
        default=CANONICAL_RELEASE_DECISION_RULE["maximum_far_false_positives"],
    )
    plan.add_argument(
        "--minimum-medium-recall",
        type=float,
        default=CANONICAL_RELEASE_DECISION_RULE["minimum_medium_recall"],
    )
    plan.add_argument(
        "--minimum-near-recall",
        type=float,
        default=CANONICAL_RELEASE_DECISION_RULE["minimum_near_recall"],
    )
    plan.add_argument(
        "--minimum-aggregate-precision",
        type=float,
        default=CANONICAL_RELEASE_DECISION_RULE["minimum_aggregate_precision"],
    )
    plan.add_argument(
        "--minimum-aggregate-recall",
        type=float,
        default=CANONICAL_RELEASE_DECISION_RULE["minimum_aggregate_recall"],
    )
    plan.add_argument(
        "--maximum-reviewed-negative-false-positives",
        type=int,
        default=CANONICAL_RELEASE_DECISION_RULE[
            "maximum_reviewed_negative_false_positives"
        ],
    )
    plan.add_argument(
        "--maximum-runtime-pipeline-p95-ms",
        type=float,
        default=CANONICAL_RELEASE_DECISION_RULE[
            "maximum_runtime_pipeline_p95_ms"
        ],
    )
    plan.add_argument(
        "--manual-review-note", default=RELEASE_POLICY_REVIEW_NOTE
    )

    evaluate = commands.add_parser("evaluate", help="run, publish, consume, and retire once")
    evaluate.add_argument("--package", type=Path, required=True)
    evaluate.add_argument("--plan", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--dependency-manifest", type=Path, required=True)
    evaluate.add_argument("--hardware-identity", type=Path, required=True)
    evaluate.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    evaluate.add_argument("--event-id", required=True)
    evaluate.add_argument("--actor-id", required=True)
    evaluate.add_argument("--retirement-event-id", required=True)
    evaluate.add_argument("--retirement-reason", required=True)

    verify = commands.add_parser(
        "verify", help="verify evidence, frozen rule, candidate, and retired ledger"
    )
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--dependency-manifest", type=Path, required=True)
    verify.add_argument("--hardware-identity", type=Path, required=True)
    verify.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    verify.add_argument(
        "--receipt-output",
        type=Path,
        help=(
            "Atomically publish the verified portable receipt at this new file path."
        ),
    )
    verify_receipt = commands.add_parser(
        "verify-receipt",
        help=(
            "rebuild a portable receipt from evidence/plan/pointer/ledger and "
            "require exact equality"
        ),
    )
    verify_receipt.add_argument("--receipt", type=Path, required=True)
    verify_receipt.add_argument("--evidence", type=Path, required=True)
    verify_receipt.add_argument("--plan", type=Path, required=True)
    verify_receipt.add_argument("--package", type=Path, required=True)
    verify_receipt.add_argument("--dependency-manifest", type=Path, required=True)
    verify_receipt.add_argument("--hardware-identity", type=Path, required=True)
    verify_receipt.add_argument(
        "--project-root", type=Path, default=PROJECT_ROOT
    )
    publish_bundle = commands.add_parser(
        "publish-bundle",
        help="atomically publish the fixed receipt/metrics/plan/ledger workflow inventory",
    )
    publish_bundle.add_argument("--output", type=Path, required=True)
    publish_bundle.add_argument("--receipt", type=Path, required=True)
    publish_bundle.add_argument("--evidence", type=Path, required=True)
    publish_bundle.add_argument("--plan", type=Path, required=True)
    publish_bundle.add_argument("--package", type=Path, required=True)
    publish_bundle.add_argument("--dependency-manifest", type=Path, required=True)
    publish_bundle.add_argument("--hardware-identity", type=Path, required=True)
    publish_bundle.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    verify_bundle = commands.add_parser(
        "verify-bundle",
        help="verify the fixed bundle inventory against every authoritative input",
    )
    verify_bundle.add_argument("--bundle", type=Path, required=True)
    verify_bundle.add_argument("--evidence", type=Path, required=True)
    verify_bundle.add_argument("--plan", type=Path, required=True)
    verify_bundle.add_argument("--package", type=Path, required=True)
    verify_bundle.add_argument("--dependency-manifest", type=Path, required=True)
    verify_bundle.add_argument("--hardware-identity", type=Path, required=True)
    verify_bundle.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    verify_bundle.add_argument(
        "--authenticated-evaluation-environment",
        action="store_true",
        help=(
            "For the separately protected attestation job only: validate the "
            "authenticated evaluation environment record against the frozen "
            "policy while independently preflighting this job's exact runtime, "
            "without requiring path-sensitive installed hashes to match across jobs."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            record = build_evaluation_plan(
                package=args.package,
                dependency_manifest=args.dependency_manifest,
                hardware_identity=args.hardware_identity,
                project_root=args.project_root,
                device=args.device,
                expected_provider=args.expected_provider,
                detail_crop_size=args.detail_crop_size,
                decision_rule=_decision_rule_from_args(args),
                output_format=args.output_format,
                warmup=args.warmup,
                bootstrap_samples=args.bootstrap_samples,
            )
            path = write_evaluation_plan(args.output, record)
            result: Mapping[str, Any] = {
                "status": "frozen_without_sealed_member_access",
                "plan_path": str(path),
                "plan_sha256": _sha256_file(path),
                "plan_content_sha256": record["plan_content_sha256"],
            }
        elif args.command == "evaluate":
            record, ledger = evaluate_independent_holdout(
                package=args.package,
                plan=args.plan,
                output=args.output,
                dependency_manifest=args.dependency_manifest,
                hardware_identity=args.hardware_identity,
                project_root=args.project_root,
                event_id=args.event_id,
                actor_id=args.actor_id,
                retirement_event_id=args.retirement_event_id,
                retirement_reason=args.retirement_reason,
            )
            result = {
                "status": record["status"],
                "evidence_path": str(args.output.expanduser().absolute() / "metrics.json"),
                "ledger": ledger,
            }
        elif args.command == "verify":
            receipt = verify_independent_evidence(
                evidence=args.evidence,
                plan=args.plan,
                package=args.package,
                dependency_manifest=args.dependency_manifest,
                hardware_identity=args.hardware_identity,
                project_root=args.project_root,
            )
            if args.receipt_output is not None:
                receipt_path = write_independent_evidence_receipt(
                    args.receipt_output, receipt
                )
                result = {
                    "status": "portable_final_holdout_receipt_published",
                    "receipt_sha256": _sha256_file(receipt_path),
                    "receipt_content_sha256": receipt[
                        "receipt_content_sha256"
                    ],
                    "release_policy_sha256": receipt["release_policy"][
                        "policy_sha256"
                    ],
                    "canonical_release_policy_matched": receipt[
                        "canonical_release_policy_matched"
                    ],
                    "release_evidence_eligible": receipt[
                        "release_evidence_eligible"
                    ],
                    "release_approved": False,
                    "release_pointer_changed": False,
                }
            else:
                result = receipt
        elif args.command == "verify-receipt":
            result = verify_independent_evidence_receipt(
                receipt=args.receipt,
                evidence=args.evidence,
                plan=args.plan,
                package=args.package,
                dependency_manifest=args.dependency_manifest,
                hardware_identity=args.hardware_identity,
                project_root=args.project_root,
            )
        elif args.command == "publish-bundle":
            bundle_path = publish_independent_evidence_bundle(
                output=args.output,
                receipt=args.receipt,
                evidence=args.evidence,
                plan=args.plan,
                package=args.package,
                dependency_manifest=args.dependency_manifest,
                hardware_identity=args.hardware_identity,
                project_root=args.project_root,
            )
            result = verify_independent_evidence_bundle(
                bundle=bundle_path,
                evidence=args.evidence,
                plan=args.plan,
                package=args.package,
                dependency_manifest=args.dependency_manifest,
                hardware_identity=args.hardware_identity,
                project_root=args.project_root,
            )
        else:
            result = verify_independent_evidence_bundle(
                bundle=args.bundle,
                evidence=args.evidence,
                plan=args.plan,
                package=args.package,
                dependency_manifest=args.dependency_manifest,
                hardware_identity=args.hardware_identity,
                project_root=args.project_root,
                enforce_current_environment_match=(
                    not args.authenticated_evaluation_environment
                ),
            )
    except IndependentRuntimeEvaluationError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
