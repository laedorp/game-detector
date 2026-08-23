#!/usr/bin/env python3
"""Adopt a development-selected FORT candidate as the launcher default.

This command is intentionally not a release approver.  It requires a staged
candidate, its exact runtime evaluation, and the current fixed tournament's
sealed development-only winner evidence. Both exported formats are copied into a new immutable,
content-addressed directory; the existing default remains untouched and the
single release-default pointer is replaced only after every other write is
complete.  Independent holdout, frozen-build, and physical-GPU gates remain
explicitly false.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import csv
import ctypes
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_fort_release_candidate import (  # noqa: E402
    CandidateExportError,
    INITIAL_CONTRACT_NAME,
    MANIFEST_NAME as CANDIDATE_MANIFEST_NAME,
    REPRODUCIBILITY_NAME,
    RESULTS_NAME,
    _manifest_content_hash as candidate_manifest_content_hash,
    validate_staged_candidate,
)
from scripts.compare_fort_runtime_evaluations import (  # noqa: E402
    ADVANCEMENT_CONFIDENCE,
    BOOTSTRAP_SAMPLES,
    ComparisonError,
    compare_reports,
)
from scripts.evaluate_fort_runtime_model import _source_hash_snapshot  # noqa: E402
from scripts.run_fort_model_tournament import (  # noqa: E402
    EXACT_INPUT_SHAPE as TOURNAMENT_INPUT_SHAPE,
    HEADS as TOURNAMENT_HEADS,
    OUTPUT_STATUS as TOURNAMENT_OUTPUT_STATUS,
    PLAN_STATUS as TOURNAMENT_PLAN_STATUS,
    SEALED_PLAN_NAME,
    SEALED_REPORT_NAME,
    SEALED_RESULTS_NAME,
    SCALES as TOURNAMENT_SCALES,
    Slot as TournamentSlot,
    TournamentError,
    _selection_content_hash as tournament_selection_content_hash,
    _validate_comparison as validate_tournament_comparison,
)
from utils.release_model_contract import (  # noqa: E402
    CONTRACT_RELATIVE,
    QUALIFICATION_RECORD,
    ReleaseModelContractError,
    TOURNAMENT_COMPARISON_NAMES,
    TOURNAMENT_SLOT_NAMES,
    canonical_hash,
    canonical_json_bytes,
    canonical_relative_path,
    load_release_default_contract,
    make_release_default_contract,
    validate_release_default_contract,
)
from utils.public_evidence import contains_nonportable_path  # noqa: E402


SCHEMA_VERSION = 1
METRICS_NAME = "metrics.json"
ADOPTION_NAME = "ADOPTION.json"
SELECTION_MANIFEST_NAME = "selection-manifest.json"
COPIED_SELECTION_NAME = "MODEL-TOURNAMENT-SELECTION.json"
CANDIDATE_RECEIPT_NAME = "CANDIDATE-RECEIPT.json"
TRAINING_PROVENANCE_RECEIPT_NAME = "TRAINING-PROVENANCE-RECEIPT.json"
WINNER_RUNTIME_RECEIPT_NAME = "WINNER-RUNTIME-RECEIPT.json"
RELEASES_RELATIVE = PurePosixPath("models/release-defaults")
MODEL_MANIFEST_RELATIVE = PurePosixPath("models/RELEASE-MANIFEST.sha256")
LOCK_NAME = ".release-default-adoption.lock"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_BASENAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
ULTRALYTICS_RESULTS_COLUMNS = (
    "epoch",
    "time",
    "train/box_loss",
    "train/cls_loss",
    "train/l1_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/l1_loss",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
)
INTERNAL_SELECTION_KEYS = frozenset(
    {"comparison_sources", "sealed_input_sources", "copy_records"}
)
EXPECTED_CANDIDATE_GATES = dict(QUALIFICATION_RECORD)


class CandidateAdoptionError(RuntimeError):
    """Raised when candidate adoption cannot be proven or published safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--candidate-evaluation",
        type=Path,
        required=True,
        help="Exact runtime metrics.json selected by the comparison record.",
    )
    parser.add_argument(
        "--tournament-selection",
        type=Path,
        required=True,
        help="Sealed fixed-tournament directory or its selection-manifest.json.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root containing models/ (default: this checkout).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all evidence and print the proposed identity without writing.",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CandidateAdoptionError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_value(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise CandidateAdoptionError(f"{description} is not a lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateAdoptionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CandidateAdoptionError(f"non-finite JSON constant: {value}")


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _reject_symlink_components(path: Path, description: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if _path_lexists(component) and component.is_symlink():
            raise CandidateAdoptionError(
                f"{description} path contains a symlink: {component}"
            )


def _regular_file(
    path: Path,
    description: str,
    *,
    suffix: str | None = None,
) -> Path:
    expanded = path.expanduser().absolute()
    _reject_symlink_components(expanded, description)
    try:
        file_stat = expanded.stat(follow_symlinks=False)
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CandidateAdoptionError(f"{description} is missing: {expanded}") from exc
    except OSError as exc:
        raise CandidateAdoptionError(
            f"cannot inspect {description} {expanded}: {exc}"
        ) from exc
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
        raise CandidateAdoptionError(
            f"{description} must be a non-empty regular file: {expanded}"
        )
    if suffix is not None and resolved.suffix.casefold() != suffix.casefold():
        raise CandidateAdoptionError(
            f"{description} must use the {suffix} extension: {resolved}"
        )
    return resolved


def _canonical_json_file(
    path: Path,
    description: str,
    *,
    default_name: str | None = None,
) -> tuple[Path, dict[str, Any], bytes, str]:
    expanded = path.expanduser().absolute()
    if expanded.is_dir() and default_name is not None:
        expanded = expanded / default_name
    resolved = _regular_file(expanded, description, suffix=".json")
    try:
        payload = resolved.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except CandidateAdoptionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateAdoptionError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateAdoptionError(f"{description} must contain one JSON object")
    if payload != canonical_json_bytes(value):
        raise CandidateAdoptionError(
            f"{description} must be canonical sorted JSON with one trailing newline"
        )
    return resolved, value, payload, sha256(payload).hexdigest()


def _candidate_artifact(
    candidate: Path,
    artifacts: Mapping[str, Any],
    role: str,
    expected_name: str,
) -> tuple[Path, dict[str, Any]]:
    record = artifacts.get(role)
    if not isinstance(record, Mapping) or set(record) != {"name", "bytes", "sha256"}:
        raise CandidateAdoptionError(f"candidate {role} record is invalid")
    if record.get("name") != expected_name:
        raise CandidateAdoptionError(f"candidate {role} has an unexpected filename")
    source = _regular_file(candidate / expected_name, f"candidate {role}")
    if source.parent != candidate:
        raise CandidateAdoptionError(f"candidate {role} escapes its staged directory")
    if (
        record.get("bytes") != source.stat().st_size
        or record.get("sha256") != _sha256_file(source)
    ):
        raise CandidateAdoptionError(f"candidate {role} hash/size mismatch")
    return source, dict(record)


def _candidate_contract(
    candidate: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        isinstance(manifest.get("schema_version"), bool)
        or manifest.get("schema_version") != 1
    ):
        raise CandidateAdoptionError("candidate manifest schema is unsupported")
    if manifest.get("status") != "validated_release_candidate_not_approved":
        raise CandidateAdoptionError("candidate is not in the validated, unapproved state")
    release_gate = manifest.get("release_gate")
    if (
        not isinstance(release_gate, Mapping)
        or set(release_gate) != set(EXPECTED_CANDIDATE_GATES)
        or any(release_gate.get(key) is not False for key in EXPECTED_CANDIDATE_GATES)
    ):
        raise CandidateAdoptionError(
            "candidate release/hardware gates must all remain explicitly false"
        )
    content_hash = _sha256_value(
        manifest.get("candidate_content_sha256"), "candidate content hash"
    )
    if content_hash != candidate_manifest_content_hash(manifest):
        raise CandidateAdoptionError("candidate manifest content hash mismatch")
    configuration = manifest.get("configuration")
    artifacts = manifest.get("artifacts")
    if not isinstance(configuration, Mapping) or not isinstance(artifacts, Mapping):
        raise CandidateAdoptionError("candidate configuration/artifact record is missing")
    basename = configuration.get("basename")
    if not isinstance(basename, str) or SAFE_BASENAME.fullmatch(basename) is None:
        raise CandidateAdoptionError("candidate basename is invalid")
    shape = configuration.get("input_shape_nchw")
    if (
        isinstance(shape, (str, bytes))
        or not isinstance(shape, Sequence)
        or len(shape) != 4
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] != 1
        or isinstance(shape[1], bool)
        or not isinstance(shape[1], int)
        or shape[1] != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 32
            or value > 4096
            or value % 32
            for value in shape[2:]
        )
    ):
        raise CandidateAdoptionError("candidate static NCHW input shape is invalid")
    if configuration.get("one_class") != {"0": "player"}:
        raise CandidateAdoptionError("candidate must contain exactly class 0: player")
    head = configuration.get("head")
    if head not in TOURNAMENT_HEADS:
        raise CandidateAdoptionError("candidate output head is invalid")
    required_roles = {
        "onnx",
        "openvino_xml",
        "openvino_bin",
        "labels",
        "attribution",
        "initial_run_contract",
        "training_reproducibility",
        "training_results",
    }
    if not required_roles.issubset(artifacts):
        raise CandidateAdoptionError("candidate artifact records are incomplete")
    sources: dict[str, Path] = {}
    records: dict[str, dict[str, Any]] = {}
    expected = {
        "onnx": f"{basename}.onnx",
        "openvino_xml": f"{basename}.xml",
        "openvino_bin": f"{basename}.bin",
        "labels": "labels.txt",
        "attribution": "ATTRIBUTION.md",
        "initial_run_contract": INITIAL_CONTRACT_NAME,
        "training_reproducibility": REPRODUCIBILITY_NAME,
        "training_results": RESULTS_NAME,
    }
    for role, name in expected.items():
        source, record = _candidate_artifact(candidate, artifacts, role, name)
        sources[role] = source
        records[role] = record
    checkpoint = manifest.get("checkpoint")
    dataset = manifest.get("dataset")
    training_provenance = manifest.get("training_provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (checkpoint, dataset, training_provenance)
    ):
        raise CandidateAdoptionError(
            "candidate checkpoint/dataset/training provenance is incomplete"
        )
    assert isinstance(checkpoint, Mapping)
    assert isinstance(dataset, Mapping)
    assert isinstance(training_provenance, Mapping)
    checkpoint_sha256 = _sha256_value(
        checkpoint.get("sha256"), "candidate checkpoint hash"
    )
    dataset_manifest_sha256 = _sha256_value(
        dataset.get("manifest_sha256"), "candidate dataset manifest hash"
    )
    dataset_content_sha256 = _sha256_value(
        dataset.get("content_sha256"), "candidate dataset content hash"
    )
    initial_weights_sha256 = _sha256_value(
        training_provenance.get("initial_weights_sha256"),
        "candidate initial weights hash",
    )
    planned_epochs = _positive_integer(
        training_provenance.get("planned_epochs"), "candidate planned epochs"
    )
    completed_epochs = _positive_integer(
        training_provenance.get("completed_epochs"), "candidate completed epochs"
    )
    results_rows = _positive_integer(
        training_provenance.get("results_rows"), "candidate training-result rows"
    )
    if (
        training_provenance.get("schema_version") != 1
        or training_provenance.get("checkpoint_role") != "completed_run_best"
        or completed_epochs > planned_epochs
        or training_provenance.get("checkpoint_sha256") != checkpoint_sha256
        or training_provenance.get("dataset_manifest_sha256")
        != dataset_manifest_sha256
        or training_provenance.get("dataset_content_sha256")
        != dataset_content_sha256
        or training_provenance.get("initial_run_contract_sha256")
        != records["initial_run_contract"]["sha256"]
        or training_provenance.get("training_reproducibility_sha256")
        != records["training_reproducibility"]["sha256"]
        or training_provenance.get("training_results_sha256")
        != records["training_results"]["sha256"]
    ):
        raise CandidateAdoptionError(
            "candidate packaged training provenance hashes or epochs differ"
        )
    return {
        "basename": basename,
        "head": head,
        "candidate_content_sha256": content_hash,
        "input_shape_nchw": list(shape),
        "sources": sources,
        "records": records,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_content_sha256": dataset_content_sha256,
        "initial_weights_sha256": initial_weights_sha256,
        "planned_epochs": planned_epochs,
        "completed_epochs": completed_epochs,
        "results_rows": results_rows,
    }


def _safe_basename(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise CandidateAdoptionError(f"{description} has no safe basename")
    name = value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if (
        not name
        or name in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", name) is None
    ):
        raise CandidateAdoptionError(f"{description} has no portable basename")
    return name


def _public_value(value: Any, description: str) -> Any:
    """Copy JSON-safe receipt data while rejecting local-path disclosures."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CandidateAdoptionError(f"{description} contains a non-finite number")
        return value
    if isinstance(value, str):
        normalized = value.strip()
        home = str(Path.home()).replace("\\", "/").rstrip("/")
        slash_value = value.replace("\\", "/")
        if (
            not normalized
            or any(ord(character) < 32 for character in value)
            or "\\" in value
            or contains_nonportable_path(value)
            or "/home/" in slash_value.casefold()
            or "/users/" in slash_value.casefold()
            or (home and home in slash_value)
        ):
            raise CandidateAdoptionError(
                f"{description} contains an unsafe or local path-like string"
            )
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, member in value.items():
            if not isinstance(key, str):
                raise CandidateAdoptionError(f"{description} has an unsafe field name")
            _public_value(key, f"{description} field name")
            copied[key] = _public_value(member, f"{description}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _public_value(member, f"{description}[{index}]")
            for index, member in enumerate(value)
        ]
    raise CandidateAdoptionError(f"{description} is not safe canonical JSON data")


def _receipt(body: Mapping[str, Any], description: str) -> dict[str, Any]:
    receipt = _public_value(dict(body), description)
    assert isinstance(receipt, dict)
    receipt["content_sha256"] = canonical_hash(receipt)
    _public_value(receipt, description)
    return receipt


def _assert_public_text_safe(path: Path, description: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CandidateAdoptionError(f"cannot inspect {description}: {exc}") from exc
    home = str(Path.home()).replace("\\", "/").rstrip("/")
    slash_text = text.replace("\\", "/")
    if (
        "\x00" in text
        or "\\" in text
        or contains_nonportable_path(text)
        or "/home/" in slash_text.casefold()
        or "/users/" in slash_text.casefold()
        or (home and home in slash_text)
    ):
        raise CandidateAdoptionError(
            f"{description} contains a private or absolute path"
        )


def _validated_training_results_contract(
    path: Path,
    description: str,
) -> tuple[int, int]:
    """Require the pinned Ultralytics results schema and finite epoch rows."""

    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != ULTRALYTICS_RESULTS_COLUMNS:
                raise CandidateAdoptionError(
                    f"{description} header differs from the pinned Ultralytics schema"
                )
            rows = list(reader)
    except CandidateAdoptionError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CandidateAdoptionError(f"cannot parse {description}: {exc}") from exc
    if not rows:
        raise CandidateAdoptionError(f"{description} contains no completed epoch")
    epochs: list[int] = []
    expected_fields = set(ULTRALYTICS_RESULTS_COLUMNS)
    for row_index, row in enumerate(rows, start=1):
        if set(row) != expected_fields or any(row[name] is None for name in expected_fields):
            raise CandidateAdoptionError(
                f"{description} row {row_index} has missing or extra values"
            )
        parsed: dict[str, float] = {}
        for name in ULTRALYTICS_RESULTS_COLUMNS:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise CandidateAdoptionError(
                    f"{description} row {row_index} has a non-numeric {name!r}"
                ) from exc
            if not math.isfinite(value):
                raise CandidateAdoptionError(
                    f"{description} row {row_index} has a non-finite {name!r}"
                )
            parsed[name] = value
        epoch = parsed["epoch"]
        if not epoch.is_integer():
            raise CandidateAdoptionError(
                f"{description} row {row_index} has a non-integer epoch"
            )
        epochs.append(int(epoch))
    if epochs != list(range(1, len(rows) + 1)):
        raise CandidateAdoptionError(
            f"{description} epoch sequence is not contiguous from one"
        )
    return epochs[-1], len(rows)


def _read_local_json(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except CandidateAdoptionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateAdoptionError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CandidateAdoptionError(f"{description} must be a JSON object")
    return value


def _candidate_public_receipt(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = manifest.get("checkpoint")
    dataset = manifest.get("dataset")
    exporter = manifest.get("exporter")
    environment = manifest.get("environment")
    parity = manifest.get("parity")
    configuration = manifest.get("configuration")
    if not isinstance(checkpoint, Mapping) or not isinstance(dataset, Mapping):
        raise CandidateAdoptionError("candidate receipt inputs are incomplete")
    checkpoint_bytes = _positive_integer(
        checkpoint.get("bytes"), "candidate checkpoint size"
    )
    checkpoint_record = {
        "logical_name": _safe_basename(
            checkpoint.get("path"), "candidate checkpoint path"
        ),
        "bytes": checkpoint_bytes,
        "sha256": candidate["checkpoint_sha256"],
    }
    dataset_hashes = {
        "manifest_sha256": candidate["dataset_manifest_sha256"],
        "content_sha256": candidate["dataset_content_sha256"],
    }
    for key in ("yaml_sha256", "archive_sha256"):
        if key in dataset:
            dataset_hashes[key] = _sha256_value(
                dataset.get(key), f"candidate dataset {key}"
            )
    export_record: dict[str, Any] = {}
    if isinstance(exporter, Mapping):
        if "path" in exporter:
            export_record["logical_name"] = _safe_basename(
                exporter.get("path"), "candidate exporter path"
            )
        for key in (
            "sha256",
            "dataset_contract_sha256",
            "requirements_export_sha256",
        ):
            if key in exporter:
                export_record[key] = _sha256_value(
                    exporter.get(key), f"candidate exporter {key}"
                )
    packages: dict[str, Any] = {}
    raw_packages = environment.get("packages") if isinstance(environment, Mapping) else None
    if isinstance(raw_packages, Mapping):
        for name, record in raw_packages.items():
            if not isinstance(name, str) or not isinstance(record, Mapping):
                raise CandidateAdoptionError("candidate package receipt is invalid")
            packages[name] = {
                key: record[key]
                for key in ("version", "metadata_sha256", "record_sha256")
                if key in record
            }
    parity_summary: dict[str, Any] = {}
    if isinstance(parity, Mapping):
        for key in (
            "status",
            "input_shape_nchw",
            "output_layout",
            "seeds",
            "confidence_floor",
            "atol",
            "rtol",
        ):
            if key in parity:
                parity_summary[key] = parity[key]
    export_args = (
        configuration.get("export_args")
        if isinstance(configuration, Mapping)
        and isinstance(configuration.get("export_args"), Mapping)
        else {}
    )
    return _receipt(
        {
            "schema_version": 1,
            "status": "redacted_candidate_receipt_not_release_qualified",
            "original_candidate_manifest_sha256": manifest_sha256,
            "candidate_content_sha256": candidate["candidate_content_sha256"],
            "configuration": {
                "basename": candidate["basename"],
                "head": candidate["head"],
                "input_shape_nchw": list(candidate["input_shape_nchw"]),
                "one_class": {"0": "player"},
                "export_args": dict(export_args),
            },
            "checkpoint": checkpoint_record,
            "dataset": dataset_hashes,
            "artifacts": {
                role: dict(record)
                for role, record in sorted(candidate["records"].items())
            },
            "exporter": export_record,
            "package_versions": packages,
            "parity": parity_summary,
            "qualification": dict(QUALIFICATION_RECORD),
        },
        "candidate receipt",
    )


def _training_public_receipt(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    initial = _read_local_json(
        candidate["sources"]["initial_run_contract"],
        "local initial-run contract",
    )
    reproducibility = _read_local_json(
        candidate["sources"]["training_reproducibility"],
        "local training reproducibility record",
    )
    training = initial.get("training")
    if not isinstance(training, Mapping):
        training = reproducibility.get("training")
    hyperparameter_keys = (
        "epochs",
        "patience",
        "batch",
        "imgsz",
        "device",
        "workers",
        "threads",
        "cache",
        "seed",
        "smoke_test",
        "run_test",
        "adopt_interrupted_run",
    )
    hyperparameters = (
        {key: training[key] for key in hyperparameter_keys if key in training}
        if isinstance(training, Mapping)
        else {}
    )
    versions: dict[str, Any] = {}
    for source in (initial.get("environment"), reproducibility.get("environment")):
        if not isinstance(source, Mapping):
            continue
        for key in (
            "python",
            "ultralytics",
            "torch",
            "torchvision",
            "openvino",
            "onnxruntime",
            "numpy",
        ):
            if key in source:
                versions[key] = source[key]
    initial_weights_value = initial.get("initial_weights")
    initial_weights_name = (
        _safe_basename(initial_weights_value, "initial weights path")
        if isinstance(initial_weights_value, str)
        else "initial-weights.pt"
    )
    provenance = manifest.get("training_provenance")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(provenance, Mapping) or not isinstance(checkpoint, Mapping):
        raise CandidateAdoptionError("candidate training receipt inputs are incomplete")
    return _receipt(
        {
            "schema_version": 1,
            "status": "redacted_training_provenance_not_release_qualified",
            "candidate_manifest": {
                "original_sha256": manifest_sha256,
                "candidate_content_sha256": candidate["candidate_content_sha256"],
            },
            "training": {
                "checkpoint_role": "completed_run_best",
                "planned_epochs": candidate["planned_epochs"],
                "completed_epochs": candidate["completed_epochs"],
                "results_rows": candidate["results_rows"],
                "hyperparameters": hyperparameters,
            },
            "environment_versions": versions,
            "inputs": {
                "initial_weights": {
                    "logical_name": initial_weights_name,
                    "sha256": candidate["initial_weights_sha256"],
                },
                "dataset_manifest_sha256": candidate["dataset_manifest_sha256"],
                "dataset_content_sha256": candidate["dataset_content_sha256"],
            },
            "output": {
                "checkpoint": {
                    "logical_name": _safe_basename(
                        checkpoint.get("path"), "candidate checkpoint path"
                    ),
                    "sha256": candidate["checkpoint_sha256"],
                    "bytes": _positive_integer(
                        checkpoint.get("bytes"), "candidate checkpoint size"
                    ),
                },
                "training_results": dict(candidate["records"]["training_results"]),
            },
            "original_local_records": {
                "initial_run_contract_sha256": candidate["records"]
                ["initial_run_contract"]["sha256"],
                "training_reproducibility_sha256": candidate["records"]
                ["training_reproducibility"]["sha256"],
                "training_results_sha256": candidate["records"]["training_results"]
                ["sha256"],
            },
            "qualification": dict(QUALIFICATION_RECORD),
        },
        "training provenance receipt",
    )


def _winner_runtime_public_receipt(
    *,
    evaluation: Mapping[str, Any],
    evaluation_sha256: str,
    candidate: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    evaluator = evaluation.get("evaluator")
    artifact = evaluation.get("model_artifact")
    dataset = evaluation.get("dataset")
    configuration = evaluation.get("configuration")
    runtime = evaluation.get("runtime")
    metrics = evaluation.get("metrics")
    environment = evaluation.get("environment")
    if not all(
        isinstance(value, Mapping)
        for value in (evaluator, artifact, dataset, configuration)
    ):
        raise CandidateAdoptionError("winner runtime receipt inputs are incomplete")
    assert isinstance(evaluator, Mapping)
    assert isinstance(artifact, Mapping)
    assert isinstance(dataset, Mapping)
    assert isinstance(configuration, Mapping)
    evaluator_record = dict(evaluator)
    if "path" in evaluator_record:
        evaluator_record["path"] = _safe_basename(
            evaluator_record["path"], "runtime evaluator path"
        )
    members = artifact.get("members")
    if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
        raise CandidateAdoptionError("winner runtime members are invalid")
    safe_members = [
        {
            "name": _safe_basename(member.get("name"), "runtime model member"),
            "bytes": member.get("bytes"),
            "sha256": member.get("sha256"),
        }
        for member in members
        if isinstance(member, Mapping)
    ]
    if len(safe_members) != len(members):
        raise CandidateAdoptionError("winner runtime member record is invalid")
    configuration_keys = (
        "split",
        "manifest_split",
        "selection_role",
        "backend",
        "device",
        "inference_size",
        "input_shape_nchw",
        "declared_static_input_shape_nchw",
        "batch_size",
        "output_format",
        "runtime_nms_iou_threshold",
        "matching_iou_threshold",
        "minimum_prediction_confidence",
        "reported_confidence_thresholds",
        "warmup_iterations",
        "bootstrap_samples",
        "require_full_provider",
        "evaluation_mode",
        "exact_static_deployment_shape",
    )
    safe_configuration = {
        key: configuration[key] for key in configuration_keys if key in configuration
    }
    detail = configuration.get("detail_pass")
    if isinstance(detail, Mapping):
        safe_configuration["detail_pass"] = {
            key: detail[key]
            for key in (
                "enabled",
                "requested_crop_size_source_pixels",
                "selection_split_policy",
                "test_split_evaluation_permitted",
            )
            if key in detail
        }
    safe_runtime: dict[str, Any] = {}
    if isinstance(runtime, Mapping):
        for key in (
            "observed_raw_output_shape",
            "observed_raw_output_dtype",
            "timing_ms_per_image",
        ):
            if key in runtime:
                safe_runtime[key] = runtime[key]
        summary = runtime.get("summary")
        if isinstance(summary, Mapping):
            safe_runtime["summary"] = {
                key: summary[key]
                for key in (
                    "output_format",
                    "output_layout",
                    "requested_device_input",
                    "requested_provider",
                    "active_providers",
                    "require_full_provider",
                    "runtime_ep_fail_fallback_disabled",
                    "input_shape",
                    "declared_input_shape",
                    "output_shape",
                )
                if key in summary
            }
    safe_metrics: dict[str, Any] = {}
    if isinstance(metrics, Mapping):
        validation = metrics.get("val")
        if isinstance(validation, Mapping):
            for key in (
                "images",
                "ground_truth_boxes",
                "configured_pipeline",
                "aggregate_detection",
                "size_bucket_detection",
            ):
                if key in validation:
                    safe_metrics[key] = validation[key]
            primary = validation.get("primary_full_frame_reference")
            if isinstance(primary, Mapping):
                safe_metrics["primary_full_frame_reference"] = {
                    key: primary[key]
                    for key in ("aggregate_detection", "size_bucket_detection")
                    if key in primary
                }
    versions = {}
    if isinstance(environment, Mapping):
        versions = {
            key: environment[key]
            for key in ("python", "numpy", "opencv_python", "onnxruntime", "openvino")
            if key in environment
        }
    dataset_receipt = {
        key: dataset[key]
        for key in (
            "yaml_sha256",
            "manifest_sha256",
            "runtime_labels_sha256",
            "content_sha256",
            "evidence_scope",
        )
        if key in dataset
    }
    # Historical evaluator reports may contain absolute local dataset paths.
    # The exact report hash remains bound above, while the durable projection
    # carries only portable logical names.
    for key in ("yaml", "manifest"):
        if key in dataset:
            dataset_receipt[key] = _safe_basename(
                dataset[key], f"winner runtime dataset {key}"
            )
    return _receipt(
        {
            "schema_version": 1,
            "status": "redacted_winner_runtime_receipt_not_release_qualified",
            "original_runtime_evaluation_sha256": evaluation_sha256,
            "candidate_content_sha256": candidate["candidate_content_sha256"],
            "selected_pipeline": selection["selected_pipeline"],
            "detail_crop_size_source_pixels": selection[
                "detail_crop_size_source_pixels"
            ],
            "evaluator": evaluator_record,
            "model_artifact": {
                "backend": artifact.get("backend"),
                "entrypoint_name": safe_members[0]["name"],
                "entrypoint_sha256": artifact.get("entrypoint_sha256"),
                "content_sha256": artifact.get("content_sha256"),
                "members": safe_members,
            },
            "dataset": dataset_receipt,
            "configuration": safe_configuration,
            "runtime": safe_runtime,
            "environment_versions": versions,
            "metrics": safe_metrics,
            "qualification": dict(QUALIFICATION_RECORD),
        },
        "winner runtime receipt",
    )


def _artifact_identity(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(
        [{"name": record["name"], "sha256": record["sha256"]} for record in records]
    )


def _exact_mapping(
    value: object,
    keys: set[str],
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise CandidateAdoptionError(f"{description} fields are incomplete or unexpected")
    return value


def _positive_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CandidateAdoptionError(f"{description} must be a positive integer")
    return value


def _tournament_slot(
    name: str,
    record: object,
) -> tuple[TournamentSlot, Mapping[str, Any]]:
    scale, separator, head = name.partition("_")
    if not separator or scale not in TOURNAMENT_SCALES or head not in TOURNAMENT_HEADS:
        raise CandidateAdoptionError(f"tournament candidate slot is invalid: {name!r}")
    candidate = _exact_mapping(
        record,
        {
            "scale",
            "head",
            "plan_paths",
            "candidate_manifest_sha256",
            "candidate_content_sha256",
            "checkpoint_sha256",
            "initial_weights_sha256",
            "training_identity",
            "onnx",
            "validation_report_sha256",
            "runtime_model_content_sha256",
            "selected_pipeline_after_detail_ab",
        },
        f"tournament candidate {name}",
    )
    if candidate.get("scale") != scale or candidate.get("head") != head:
        raise CandidateAdoptionError(f"tournament candidate {name} identity differs")
    plan_paths = _exact_mapping(
        candidate.get("plan_paths"),
        {"candidate_dir", "validation_report"},
        f"tournament candidate {name} plan paths",
    )
    if any(
        not isinstance(plan_paths.get(key), str) or not str(plan_paths[key]).strip()
        for key in plan_paths
    ):
        raise CandidateAdoptionError(f"tournament candidate {name} plan paths are invalid")
    onnx = _exact_mapping(
        candidate.get("onnx"),
        {"name", "bytes", "sha256"},
        f"tournament candidate {name} ONNX",
    )
    onnx_name = onnx.get("name")
    if not isinstance(onnx_name, str) or SAFE_BASENAME.fullmatch(
        PurePosixPath(onnx_name).stem
    ) is None or PurePosixPath(onnx_name).suffix.casefold() != ".onnx":
        raise CandidateAdoptionError(f"tournament candidate {name} ONNX name is unsafe")
    onnx_bytes = _positive_integer(onnx.get("bytes"), f"tournament candidate {name} ONNX size")
    onnx_sha = _sha256_value(onnx.get("sha256"), f"tournament candidate {name} ONNX hash")
    runtime_content = _sha256_value(
        candidate.get("runtime_model_content_sha256"),
        f"tournament candidate {name} runtime model content hash",
    )
    if runtime_content != _artifact_identity([onnx]):
        raise CandidateAdoptionError(
            f"tournament candidate {name} runtime model identity differs from ONNX"
        )
    pipeline = candidate.get("selected_pipeline_after_detail_ab")
    if pipeline not in {"primary", "configured"}:
        raise CandidateAdoptionError(f"tournament candidate {name} pipeline is invalid")
    training_identity = candidate.get("training_identity")
    if not isinstance(training_identity, Mapping):
        raise CandidateAdoptionError(
            f"tournament candidate {name} training identity is invalid"
        )
    slot = TournamentSlot(
        scale=scale,
        head=head,
        candidate_dir=Path("."),
        report_path=Path("."),
        candidate_manifest_sha256=_sha256_value(
            candidate.get("candidate_manifest_sha256"),
            f"tournament candidate {name} manifest hash",
        ),
        candidate_content_sha256=_sha256_value(
            candidate.get("candidate_content_sha256"),
            f"tournament candidate {name} content hash",
        ),
        checkpoint_sha256=_sha256_value(
            candidate.get("checkpoint_sha256"),
            f"tournament candidate {name} checkpoint hash",
        ),
        initial_weights_sha256=_sha256_value(
            candidate.get("initial_weights_sha256"),
            f"tournament candidate {name} initial-weights hash",
        ),
        training_identity=dict(training_identity),
        onnx_name=onnx_name,
        onnx_sha256=onnx_sha,
        onnx_bytes=onnx_bytes,
        report_sha256=_sha256_value(
            candidate.get("validation_report_sha256"),
            f"tournament candidate {name} validation-report hash",
        ),
        report_model_content_sha256=runtime_content,
        plan_paths=dict(plan_paths),
    )
    return slot, candidate


def _sealed_tournament_inputs(
    *,
    selection_root: Path,
    value: object,
    plan_sha256: str,
    slots: Mapping[str, TournamentSlot],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    sealed = _exact_mapping(
        value,
        {"plan", "runtime_reports", "training_results"},
        "tournament sealed inputs",
    )
    runtime_reports = _exact_mapping(
        sealed.get("runtime_reports"),
        set(TOURNAMENT_SLOT_NAMES),
        "tournament sealed runtime reports",
    )
    training_results = _exact_mapping(
        sealed.get("training_results"),
        set(TOURNAMENT_SCALES),
        "tournament sealed training results",
    )
    expected: dict[str, tuple[object, str, str]] = {
        "tournament_plan": (
            sealed.get("plan"),
            f"inputs/{SEALED_PLAN_NAME}",
            plan_sha256,
        ),
    }
    for name in TOURNAMENT_SLOT_NAMES:
        expected[f"tournament_runtime_report_{name}"] = (
            runtime_reports[name],
            f"inputs/runtime/{name}/{SEALED_REPORT_NAME}",
            slots[name].report_sha256,
        )
    for scale in TOURNAMENT_SCALES:
        training_hash = slots[f"{scale}_end2end"].training_identity.get(
            "training_results_sha256"
        )
        if (
            training_hash != slots[f"{scale}_traditional"].training_identity.get(
                "training_results_sha256"
            )
        ):
            raise CandidateAdoptionError(
                f"tournament {scale} training-result identities differ"
            )
        expected[f"tournament_training_results_{scale}"] = (
            training_results[scale],
            f"inputs/training/{scale}/{SEALED_RESULTS_NAME}",
            _sha256_value(training_hash, f"tournament {scale} training-result hash"),
        )

    sources: dict[str, Path] = {}
    copy_records: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for role, (raw_record, expected_path, expected_hash) in expected.items():
        record = _exact_mapping(
            raw_record,
            {"path", "bytes", "sha256"},
            f"{role} sealed-input record",
        )
        relative = record.get("path")
        if relative != expected_path or relative in seen_paths:
            raise CandidateAdoptionError(
                f"{role} sealed-input path is unexpected or repeated"
            )
        seen_paths.add(str(relative))
        size = _positive_integer(record.get("bytes"), f"{role} sealed-input size")
        digest = _sha256_value(record.get("sha256"), f"{role} sealed-input hash")
        if digest != expected_hash:
            raise CandidateAdoptionError(f"{role} sealed-input identity differs")
        pure = PurePosixPath(str(relative))
        source = _regular_file(
            selection_root.joinpath(*pure.parts),
            f"{role} sealed input",
        )
        if source.stat().st_size != size or _sha256_file(source) != digest:
            raise CandidateAdoptionError(f"{role} sealed-input bytes differ")
        if selection_root != source and selection_root not in source.parents:
            raise CandidateAdoptionError(f"{role} sealed input escapes selection")
        if source.suffix.casefold() == ".json":
            public_json = _read_local_json(source, f"{role} public sealed input")
            _public_value(public_json, f"{role} public sealed input")
        else:
            _assert_public_text_safe(source, f"{role} public sealed input")
        sources[role] = source
        copy_records[role] = {"bytes": size, "sha256": digest}
    return sources, copy_records


def _portable_plan_path(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or ":" in value
        or "\x00" in value
    ):
        raise CandidateAdoptionError(
            f"{description} must be a canonical portable relative path"
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or not pure.parts
        or any(
            part in {"", ".", ".."}
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", part) is None
            for part in pure.parts
        )
    ):
        raise CandidateAdoptionError(
            f"{description} must be a canonical portable relative path"
        )
    return value


def _replay_tournament_plan(
    *,
    plan_path: Path,
    plan_sha256: str,
    fixed: Mapping[str, Any],
    dataset: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    slots: Mapping[str, TournamentSlot],
) -> dict[str, Any]:
    _resolved, plan, payload, digest = _canonical_json_file(
        plan_path,
        "sealed tournament plan",
    )
    if digest != plan_sha256:
        raise CandidateAdoptionError("sealed tournament plan hash differs")
    plan = _exact_mapping(
        plan,
        {"schema_version", "status", "dataset", "runtime", "models"},
        "sealed tournament plan",
    )
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("status") != TOURNAMENT_PLAN_STATUS
    ):
        raise CandidateAdoptionError(
            "sealed tournament plan schema/status is unsupported"
        )
    plan_dataset = _exact_mapping(
        plan.get("dataset"),
        {"manifest_sha256", "content_sha256"},
        "sealed tournament plan dataset",
    )
    if dict(plan_dataset) != dict(dataset):
        raise CandidateAdoptionError(
            "sealed tournament plan dataset differs from selection"
        )
    plan_runtime = _exact_mapping(
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
        "sealed tournament plan runtime",
    )
    expected_runtime = {
        "backend": fixed["backend"],
        "device": fixed["device"],
        "require_full_provider": fixed["require_full_provider"],
        "inference_size": fixed["inference_size"],
        "detail_crop_size_source_pixels": fixed[
            "detail_crop_size_source_pixels"
        ],
        "confidence": fixed["confidence"],
        "comparator_bootstrap_samples": fixed["paired_bootstrap_samples"],
    }
    if dict(plan_runtime) != expected_runtime:
        raise CandidateAdoptionError(
            "sealed tournament plan runtime differs from fixed selection contract"
        )
    models = _exact_mapping(
        plan.get("models"),
        set(TOURNAMENT_SCALES),
        "sealed tournament plan models",
    )
    candidate_paths: set[str] = set()
    report_paths: set[str] = set()
    for scale in TOURNAMENT_SCALES:
        scale_plan = _exact_mapping(
            models.get(scale),
            {"initial_weights_sha256", *TOURNAMENT_HEADS},
            f"sealed tournament plan model {scale}",
        )
        expected_initial = slots[f"{scale}_end2end"].initial_weights_sha256
        if (
            scale_plan.get("initial_weights_sha256") != expected_initial
            or slots[f"{scale}_traditional"].initial_weights_sha256
            != expected_initial
        ):
            raise CandidateAdoptionError(
                f"sealed tournament plan {scale} initial weights differ"
            )
        for head in TOURNAMENT_HEADS:
            name = f"{scale}_{head}"
            slot_plan = _exact_mapping(
                scale_plan.get(head),
                {"candidate_dir", "validation_report"},
                f"sealed tournament plan slot {name}",
            )
            candidate_path = _portable_plan_path(
                slot_plan.get("candidate_dir"),
                f"sealed tournament plan {name} candidate path",
            )
            report_path = _portable_plan_path(
                slot_plan.get("validation_report"),
                f"sealed tournament plan {name} report path",
            )
            if candidate_path in candidate_paths or report_path in report_paths:
                raise CandidateAdoptionError(
                    "sealed tournament plan repeats a candidate or report path"
                )
            candidate_paths.add(candidate_path)
            report_paths.add(report_path)
            if candidates[name].get("plan_paths") != {
                "candidate_dir": candidate_path,
                "validation_report": report_path,
            }:
                raise CandidateAdoptionError(
                    f"sealed tournament plan slot {name} differs from selection"
                )
    if (
        slots["n_end2end"].initial_weights_sha256
        == slots["s_end2end"].initial_weights_sha256
    ):
        raise CandidateAdoptionError(
            "sealed tournament plan n/s initial weights must remain distinct"
        )
    return {
        "status": "sealed_tournament_plan_replayed",
        "sha256": digest,
        "canonical_bytes": len(payload),
        "dataset_matches": True,
        "runtime_matches": True,
        "all_slot_paths_and_initial_weights_match": True,
    }


def _replay_tournament_comparison(
    *,
    name: str,
    baseline: TournamentSlot,
    candidate: TournamentSlot,
    baseline_pipeline: str,
    candidate_pipeline: str,
    sealed_report_sources: Mapping[str, Path],
    sealed_comparison: Mapping[str, Any],
    sealed_payload: bytes,
    sealed_sha256: str,
) -> tuple[bool, dict[str, Any]]:
    baseline_report = sealed_report_sources[
        f"tournament_runtime_report_{baseline.name}"
    ]
    candidate_report = sealed_report_sources[
        f"tournament_runtime_report_{candidate.name}"
    ]
    with tempfile.TemporaryDirectory(prefix="proaim-adoption-comparison-replay-") as temporary:
        output = Path(temporary) / name
        try:
            replayed = compare_reports(
                baseline=baseline_report,
                candidate=candidate_report,
                output=output,
                confidence=ADVANCEMENT_CONFIDENCE,
                baseline_pipeline=baseline_pipeline,
                candidate_pipeline=candidate_pipeline,
                bootstrap_samples=BOOTSTRAP_SAMPLES,
            )
        except ComparisonError as exc:
            raise CandidateAdoptionError(
                f"sealed tournament comparison {name} cannot be recomputed: {exc}"
            ) from exc
        _path, published, replay_payload, replay_sha256 = _canonical_json_file(
            output / "comparison.json",
            f"replayed tournament comparison {name}",
        )
    if replayed != published:
        raise CandidateAdoptionError(
            f"replayed tournament comparison {name} return/file differ"
        )
    if (
        published != sealed_comparison
        or replay_payload != sealed_payload
        or replay_sha256 != sealed_sha256
    ):
        raise CandidateAdoptionError(
            f"sealed tournament comparison {name} differs from deterministic "
            "report replay"
        )
    policy = published.get("development_advancement_policy")
    if not isinstance(policy, Mapping) or not isinstance(policy.get("passed"), bool):
        raise CandidateAdoptionError(
            f"replayed tournament comparison {name} has no boolean outcome"
        )
    passed = bool(policy["passed"])
    return passed, {
        "baseline_slot": baseline.name,
        "candidate_slot": candidate.name,
        "baseline_pipeline": baseline_pipeline,
        "candidate_pipeline": candidate_pipeline,
        "baseline_report_sha256": baseline.report_sha256,
        "candidate_report_sha256": candidate.report_sha256,
        "sealed_comparison_sha256": sealed_sha256,
        "replayed_comparison_sha256": replay_sha256,
        "challenger_advanced": passed,
    }


def _winner_runtime_contract(
    *,
    evaluation: Mapping[str, Any],
    evaluation_sha256: str,
    candidate: Mapping[str, Any],
    winner_slot: TournamentSlot,
    winner_pipeline: str,
    detail_crop_size: int,
    tournament_device: str,
    tournament_require_full_provider: bool,
) -> dict[str, Any]:
    if evaluation_sha256 != winner_slot.report_sha256:
        raise CandidateAdoptionError(
            "candidate evaluation bytes differ from the tournament winner report"
        )
    if (
        isinstance(evaluation.get("schema_version"), bool)
        or evaluation.get("schema_version") != 4
    ):
        raise CandidateAdoptionError("winner runtime evaluation schema is unsupported")
    configuration = evaluation.get("configuration")
    artifact = evaluation.get("model_artifact")
    qualification = evaluation.get("qualification")
    dataset = evaluation.get("dataset")
    evaluator = evaluation.get("evaluator")
    if not all(
        isinstance(item, Mapping)
        for item in (configuration, artifact, qualification, dataset, evaluator)
    ):
        raise CandidateAdoptionError("winner runtime evaluation contract is incomplete")
    assert isinstance(configuration, Mapping)
    assert isinstance(artifact, Mapping)
    assert isinstance(qualification, Mapping)
    assert isinstance(dataset, Mapping)
    assert isinstance(evaluator, Mapping)
    if (
        dataset.get("manifest_sha256") != candidate["dataset_manifest_sha256"]
        or dataset.get("content_sha256") != candidate["dataset_content_sha256"]
    ):
        raise CandidateAdoptionError("winner runtime evaluation dataset differs")
    if (
        configuration.get("split") != "val"
        or configuration.get("selection_role") != "development_validation"
        or configuration.get("backend") != "onnxruntime"
        or configuration.get("device") != tournament_device
        or configuration.get("require_full_provider") is not tournament_require_full_provider
        or configuration.get("input_shape_nchw") != candidate["input_shape_nchw"]
        or configuration.get("declared_static_input_shape_nchw")
        != candidate["input_shape_nchw"]
        or configuration.get("exact_static_deployment_shape") is not True
        or configuration.get("output_format") != candidate["head"]
    ):
        raise CandidateAdoptionError(
            "winner runtime evaluation does not bind the tournament deployment contract"
        )
    confidences = configuration.get("reported_confidence_thresholds")
    if isinstance(confidences, (str, bytes)) or not isinstance(confidences, Sequence):
        raise CandidateAdoptionError("winner evaluation confidence contract is invalid")
    try:
        normalized_confidences = [float(value) for value in confidences]
    except (TypeError, ValueError) as exc:
        raise CandidateAdoptionError("winner evaluation confidence contract is invalid") from exc
    if (
        any(isinstance(value, bool) for value in confidences)
        or any(not math.isfinite(value) for value in normalized_confidences)
        or normalized_confidences.count(ADVANCEMENT_CONFIDENCE) != 1
    ):
        raise CandidateAdoptionError(
            "winner evaluation lacks one exact advancement confidence"
        )
    if (
        qualification.get("status") != "development_evidence_only"
        or qualification.get("independent_holdout_required") is not True
    ):
        raise CandidateAdoptionError("winner evaluation overstates release qualification")
    detail = configuration.get("detail_pass")
    if (
        not isinstance(detail, Mapping)
        or detail.get("enabled") is not True
        or detail.get("requested_crop_size_source_pixels") != detail_crop_size
        or detail.get("selection_split_policy") != "validation_only"
        or detail.get("test_split_evaluation_permitted") is not False
    ):
        raise CandidateAdoptionError("winner evaluation detail workload differs from tournament")
    current_source = _source_hash_snapshot("onnxruntime")
    if dict(evaluator) != {
        **current_source["evaluator"],
        "pipeline_source_sha256": current_source["pipeline"],
    }:
        raise CandidateAdoptionError(
            "winner evaluation uses a different evaluator or runtime-pipeline revision"
        )
    if artifact.get("backend") != "onnxruntime":
        raise CandidateAdoptionError("winner evaluation is not an ONNX Runtime artifact")
    members = artifact.get("members")
    if isinstance(members, (str, bytes)) or not isinstance(members, Sequence) or len(members) != 1:
        raise CandidateAdoptionError("winner evaluation must bind exactly one ONNX member")
    member = members[0]
    expected = candidate["records"]["onnx"]
    if not isinstance(member, Mapping) or {
        "name": member.get("name"),
        "bytes": member.get("bytes"),
        "sha256": member.get("sha256"),
    } != expected:
        raise CandidateAdoptionError(
            "winner runtime ONNX differs from the staged candidate manifest"
        )
    model_content = _artifact_identity([expected])
    if (
        artifact.get("entrypoint_sha256") != expected["sha256"]
        or artifact.get("content_sha256") != model_content
        or winner_slot.onnx_name != expected["name"]
        or winner_slot.onnx_bytes != expected["bytes"]
        or winner_slot.onnx_sha256 != expected["sha256"]
        or winner_slot.report_model_content_sha256 != model_content
    ):
        raise CandidateAdoptionError(
            "tournament winner, runtime report, and staged ONNX identities differ"
        )
    pointer_detail = detail_crop_size if winner_pipeline == "configured" else 0
    return {
        "candidate_evaluation_sha256": evaluation_sha256,
        "selected_backend": "onnxruntime",
        "selected_pipeline": winner_pipeline,
        "selected_model_content_sha256": model_content,
        "detail_crop_size_source_pixels": pointer_detail,
    }


def _tournament_selection_contract(
    *,
    selection_path: Path,
    selection: Mapping[str, Any],
    selection_sha256: str,
    evaluation: Mapping[str, Any],
    evaluation_sha256: str,
    candidate: Mapping[str, Any],
    candidate_manifest_sha256: str,
    project_root: Path,
) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "status",
        "orchestrator",
        "comparator",
        "candidate_exporter",
        "public_evidence_privacy",
        "runtime_evaluator",
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
    if set(selection) != expected_top:
        raise CandidateAdoptionError("tournament selection fields are incomplete or unexpected")
    if (
        isinstance(selection.get("schema_version"), bool)
        or selection.get("schema_version") != 1
        or selection.get("status") != TOURNAMENT_OUTPUT_STATUS
        or selection.get("selection_content_sha256")
        != tournament_selection_content_hash(selection)
    ):
        raise CandidateAdoptionError("tournament selection schema/status/self-hash is invalid")
    _public_value(selection, "tournament selection manifest")

    source_contracts = (
        (
            selection.get("orchestrator"),
            "run_fort_model_tournament.py",
            project_root / "scripts" / "run_fort_model_tournament.py",
            "tournament orchestrator",
        ),
        (
            selection.get("comparator"),
            "compare_fort_runtime_evaluations.py",
            project_root / "scripts" / "compare_fort_runtime_evaluations.py",
            "tournament comparator",
        ),
        (
            selection.get("candidate_exporter"),
            "export_fort_release_candidate.py",
            project_root / "scripts" / "export_fort_release_candidate.py",
            "candidate exporter",
        ),
        (
            selection.get("public_evidence_privacy"),
            "public_evidence.py",
            project_root / "utils" / "public_evidence.py",
            "public-evidence privacy validator",
        ),
    )
    for record, name, path, description in source_contracts:
        source = _regular_file(path, f"current {description}", suffix=".py")
        if record != {"path": name, "sha256": _sha256_file(source)}:
            raise CandidateAdoptionError(f"selection uses a different {description} revision")
    current_runtime = _source_hash_snapshot("onnxruntime")
    runtime_evaluator = selection.get("runtime_evaluator")
    if runtime_evaluator != {
        "path": "evaluate_fort_runtime_model.py",
        "sha256": current_runtime["evaluator"]["sha256"],
        "pipeline_source_sha256": current_runtime["pipeline"],
    }:
        raise CandidateAdoptionError("selection uses a different runtime evaluator/pipeline")
    plan = _exact_mapping(
        selection.get("plan"),
        {"path", "sha256", "status", "timing_note"},
        "tournament plan record",
    )
    if (
        plan.get("path") != f"inputs/{SEALED_PLAN_NAME}"
        or plan.get("status") != TOURNAMENT_PLAN_STATUS
        or plan.get("timing_note")
        != (
            "The plan hash binds the fixed bracket, but this command does not "
            "claim the plan predates report generation. This is development data."
        )
    ):
        raise CandidateAdoptionError("tournament plan record is invalid")
    plan_sha256 = _sha256_value(plan.get("sha256"), "tournament plan hash")

    fixed = _exact_mapping(
        selection.get("fixed_contract"),
        {
            "scales",
            "heads",
            "backend",
            "device",
            "require_full_provider",
            "input_shape_nchw",
            "inference_size",
            "detail_crop_size_source_pixels",
            "confidence",
            "paired_bootstrap_samples",
            "split",
            "bracket",
        },
        "tournament fixed contract",
    )
    device = fixed.get("device")
    detail_crop = _positive_integer(
        fixed.get("detail_crop_size_source_pixels"), "tournament detail crop"
    )
    if (
        fixed.get("scales") != list(TOURNAMENT_SCALES)
        or fixed.get("heads") != list(TOURNAMENT_HEADS)
        or fixed.get("backend") != "onnxruntime"
        or not isinstance(device, str)
        or not device.strip()
        or device != device.strip()
        or not isinstance(fixed.get("require_full_provider"), bool)
        or fixed.get("input_shape_nchw") != TOURNAMENT_INPUT_SHAPE
        or fixed.get("inference_size") != "384x640"
        or fixed.get("confidence") != ADVANCEMENT_CONFIDENCE
        or fixed.get("paired_bootstrap_samples") != BOOTSTRAP_SAMPLES
        or fixed.get("split") != "val"
        or fixed.get("bracket")
        != [
            "detail challenges primary for every scale/head export",
            "traditional challenges end2end within each scale",
            "s challenges n after within-scale selection",
        ]
    ):
        raise CandidateAdoptionError("tournament fixed workload contract differs")
    dataset = _exact_mapping(
        selection.get("dataset"),
        {"manifest_sha256", "content_sha256"},
        "tournament dataset",
    )
    if (
        dataset.get("manifest_sha256") != candidate["dataset_manifest_sha256"]
        or dataset.get("content_sha256") != candidate["dataset_content_sha256"]
    ):
        raise CandidateAdoptionError("tournament dataset differs from staged candidate")
    test_policy = selection.get("test_data_policy")
    if test_policy != {
        "test_split_consumed": False,
        "test_reports_accepted": False,
        "selection_split": "val",
    }:
        raise CandidateAdoptionError("tournament selection consumed or accepts test data")
    release = selection.get("release_qualification")
    if (
        not isinstance(release, Mapping)
        or release.get("qualified") is not False
        or release.get("release_model_approved") is not False
        or release.get("independent_holdout_required") is not True
        or release.get("reviewed_negative_scenes_required") is not True
        or release.get("physical_target_gpu_latency_required") is not True
        or release.get("frozen_build_qualification_required") is not True
    ):
        raise CandidateAdoptionError("tournament selection overstates release qualification")

    candidate_records = _exact_mapping(
        selection.get("candidates"),
        {f"{scale}_{head}" for scale in TOURNAMENT_SCALES for head in TOURNAMENT_HEADS},
        "tournament candidates",
    )
    slots: dict[str, TournamentSlot] = {}
    normalized_candidates: dict[str, Mapping[str, Any]] = {}
    for name, raw in candidate_records.items():
        slot, normalized = _tournament_slot(name, raw)
        slots[name] = slot
        normalized_candidates[name] = normalized

    selection_root = selection_path.parent
    sealed_input_sources, sealed_copy_records = _sealed_tournament_inputs(
        selection_root=selection_root,
        value=selection.get("sealed_inputs"),
        plan_sha256=plan_sha256,
        slots=slots,
    )
    plan_replay = _replay_tournament_plan(
        plan_path=sealed_input_sources["tournament_plan"],
        plan_sha256=plan_sha256,
        fixed=fixed,
        dataset=dataset,
        candidates=normalized_candidates,
        slots=slots,
    )

    comparison_records = _exact_mapping(
        selection.get("comparisons"),
        set(TOURNAMENT_COMPARISON_NAMES),
        "tournament comparison inventory",
    )
    comparisons_root = selection_root / "comparisons"
    if (
        {child.name for child in selection_root.iterdir()}
        != {SELECTION_MANIFEST_NAME, "comparisons", "inputs"}
        or comparisons_root.is_symlink()
        or not comparisons_root.is_dir()
        or {child.name for child in comparisons_root.iterdir()}
        != set(TOURNAMENT_COMPARISON_NAMES)
    ):
        raise CandidateAdoptionError("sealed tournament selection has unexpected members")

    def comparison_input(name: str) -> tuple[TournamentSlot, TournamentSlot, str, str]:
        if name.endswith("_primary_vs_detail"):
            slot_name = name.removesuffix("_primary_vs_detail")
            return slots[slot_name], slots[slot_name], "primary", "configured"
        if name == "n_end2end_vs_traditional":
            return (
                slots["n_end2end"],
                slots["n_traditional"],
                selected_pipelines["n_end2end"],
                selected_pipelines["n_traditional"],
            )
        if name == "s_end2end_vs_traditional":
            return (
                slots["s_end2end"],
                slots["s_traditional"],
                selected_pipelines["s_end2end"],
                selected_pipelines["s_traditional"],
            )
        return (
            slots[f"n_{selected_heads['n']}"],
            slots[f"s_{selected_heads['s']}"],
            selected_pipelines[f"n_{selected_heads['n']}"],
            selected_pipelines[f"s_{selected_heads['s']}"],
        )

    comparison_sources: dict[str, Path] = {}
    comparison_copy_records: dict[str, dict[str, Any]] = {}
    comparison_replay_records: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, bool] = {}
    selected_pipelines: dict[str, str] = {}
    selected_heads: dict[str, str] = {}
    comparator_sha = _sha256_file(
        _regular_file(
            project_root / "scripts" / "compare_fort_runtime_evaluations.py",
            "current tournament comparator",
            suffix=".py",
        )
    )
    ordered = [
        *[name for name in TOURNAMENT_COMPARISON_NAMES if name.endswith("_primary_vs_detail")],
        "n_end2end_vs_traditional",
        "s_end2end_vs_traditional",
        "n_vs_s",
    ]
    for name in ordered:
        record = _exact_mapping(
            comparison_records[name],
            {"path", "sha256", "challenger_advanced"},
            f"tournament comparison {name} manifest record",
        )
        expected_relative = f"comparisons/{name}/comparison.json"
        if record.get("path") != expected_relative or not isinstance(
            record.get("challenger_advanced"), bool
        ):
            raise CandidateAdoptionError(f"tournament comparison {name} path/outcome is invalid")
        comparison_path = comparisons_root / name / "comparison.json"
        _resolved, comparison, comparison_payload, digest = _canonical_json_file(
            comparison_path, f"tournament comparison {name}"
        )
        if digest != _sha256_value(record.get("sha256"), f"tournament comparison {name} hash"):
            raise CandidateAdoptionError(f"tournament comparison {name} hash mismatch")
        _public_value(comparison, f"tournament comparison {name}")
        baseline, challenger, baseline_pipeline, challenger_pipeline = comparison_input(name)
        replay_passed, replay_record = _replay_tournament_comparison(
            name=name,
            baseline=baseline,
            candidate=challenger,
            baseline_pipeline=baseline_pipeline,
            candidate_pipeline=challenger_pipeline,
            sealed_report_sources=sealed_input_sources,
            sealed_comparison=comparison,
            sealed_payload=comparison_payload,
            sealed_sha256=digest,
        )
        try:
            outcome = validate_tournament_comparison(
                name=name,
                record=comparison,
                comparison_path=comparison_path,
                comparator_sha256=comparator_sha,
                baseline=baseline,
                candidate=challenger,
                baseline_pipeline=baseline_pipeline,
                candidate_pipeline=challenger_pipeline,
            )
        except TournamentError as exc:
            raise CandidateAdoptionError(
                f"sealed tournament comparison {name} is invalid: {exc}"
            ) from exc
        if (
            outcome.passed is not replay_passed
            or replay_passed is not record.get("challenger_advanced")
        ):
            raise CandidateAdoptionError(f"tournament comparison {name} outcome differs")
        outcomes[name] = replay_passed
        comparison_replay_records[name] = replay_record
        comparison_role = f"tournament_comparison_{name}"
        comparison_sources[comparison_role] = comparison_path
        comparison_copy_records[comparison_role] = {
            "bytes": len(comparison_payload),
            "sha256": digest,
        }
        if name.endswith("_primary_vs_detail"):
            slot_name = name.removesuffix("_primary_vs_detail")
            selected_pipelines[slot_name] = (
                "configured" if replay_passed else "primary"
            )
        elif name.endswith("_end2end_vs_traditional"):
            scale = name[0]
            selected_heads[scale] = "traditional" if replay_passed else "end2end"

    expected_files = {
        selection_path,
        *comparison_sources.values(),
        *sealed_input_sources.values(),
    }
    expected_relatives = {
        path.relative_to(selection_root).as_posix() for path in expected_files
    }
    expected_directories: set[str] = set()
    for relative in expected_relatives:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for child in selection_root.rglob("*"):
        relative = child.relative_to(selection_root).as_posix()
        if child.is_symlink():
            raise CandidateAdoptionError(
                f"sealed tournament member uses a symlink: {relative}"
            )
        if child.is_file():
            actual_files.add(relative)
        elif child.is_dir():
            actual_directories.add(relative)
        else:
            raise CandidateAdoptionError(
                f"sealed tournament has an unsafe member: {relative}"
            )
    if actual_files != expected_relatives or actual_directories != expected_directories:
        raise CandidateAdoptionError(
            "sealed tournament filesystem inventory differs from its manifest"
        )

    decisions = _exact_mapping(
        selection.get("development_selection"),
        {"detail_decisions", "head_decisions", "scale_decision", "winner", "scope"},
        "tournament development selection",
    )
    expected_detail = {
        slot_name: {
            "incumbent": "primary",
            "challenger": "configured",
            "challenger_advanced": outcomes[f"{slot_name}_primary_vs_detail"],
            "selected_pipeline": selected_pipelines[slot_name],
            "comparison": f"comparisons/{slot_name}_primary_vs_detail/comparison.json",
        }
        for slot_name in slots
    }
    expected_heads = {
        scale: {
            "incumbent": "end2end",
            "challenger": "traditional",
            "challenger_advanced": outcomes[f"{scale}_end2end_vs_traditional"],
            "selected_head": selected_heads[scale],
            "comparison": f"comparisons/{scale}_end2end_vs_traditional/comparison.json",
        }
        for scale in TOURNAMENT_SCALES
    }
    winner_scale = "s" if outcomes["n_vs_s"] else "n"
    winner_head = selected_heads[winner_scale]
    winner_name = f"{winner_scale}_{winner_head}"
    winner_slot = slots[winner_name]
    winner_pipeline = selected_pipelines[winner_name]
    expected_scale = {
        "incumbent": "n",
        "challenger": "s",
        "challenger_advanced": outcomes["n_vs_s"],
        "selected_scale": winner_scale,
        "comparison": "comparisons/n_vs_s/comparison.json",
    }
    expected_winner = {
        "slot": winner_name,
        "scale": winner_scale,
        "head": winner_head,
        "pipeline": winner_pipeline,
        "candidate_content_sha256": winner_slot.candidate_content_sha256,
        "onnx_sha256": winner_slot.onnx_sha256,
        "validation_report_sha256": winner_slot.report_sha256,
    }
    if (
        decisions.get("detail_decisions") != expected_detail
        or decisions.get("head_decisions") != expected_heads
        or decisions.get("scale_decision") != expected_scale
        or decisions.get("winner") != expected_winner
        or decisions.get("scope")
        != (
            "Development validation winner only; it may advance to independent "
            "holdout and physical frozen-build GPU qualification."
        )
    ):
        raise CandidateAdoptionError("tournament winner/decision replay differs")
    winner_record = normalized_candidates[winner_name]
    expected_training_identity = {
        "initial_run_contract_sha256": candidate["records"]
        ["initial_run_contract"]["sha256"],
        "training_reproducibility_sha256": candidate["records"]
        ["training_reproducibility"]["sha256"],
        "training_results_sha256": candidate["records"]["training_results"]
        ["sha256"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "initial_weights_sha256": candidate["initial_weights_sha256"],
        "dataset_manifest_sha256": candidate["dataset_manifest_sha256"],
        "dataset_content_sha256": candidate["dataset_content_sha256"],
        "completed_epochs": candidate["completed_epochs"],
        "results_rows": candidate["results_rows"],
    }
    if (
        winner_slot.candidate_manifest_sha256 != candidate_manifest_sha256
        or winner_slot.candidate_content_sha256 != candidate["candidate_content_sha256"]
        or winner_slot.checkpoint_sha256 != candidate["checkpoint_sha256"]
        or winner_slot.initial_weights_sha256 != candidate["initial_weights_sha256"]
        or dict(winner_slot.training_identity) != expected_training_identity
        or winner_record.get("head") != candidate["head"]
        or candidate["input_shape_nchw"] != TOURNAMENT_INPUT_SHAPE
        or winner_record.get("onnx") != candidate["records"]["onnx"]
    ):
        raise CandidateAdoptionError(
            "staged candidate is not the exact sealed tournament winner"
        )
    winner_training_role = f"tournament_training_results_{winner_scale}"
    winner_training_record = sealed_copy_records[winner_training_role]
    candidate_training_record = candidate["records"]["training_results"]
    if (
        winner_training_record.get("bytes") != candidate_training_record["bytes"]
        or winner_training_record.get("sha256")
        != candidate_training_record["sha256"]
    ):
        raise CandidateAdoptionError(
            "sealed winner-scale training results differ from staged candidate"
        )
    completed_epoch, result_rows = _validated_training_results_contract(
        sealed_input_sources[winner_training_role],
        "sealed winner-scale training results",
    )
    if (
        completed_epoch != candidate["completed_epochs"]
        or result_rows != candidate["results_rows"]
    ):
        raise CandidateAdoptionError(
            "sealed winner-scale training results rows/epochs differ from candidate"
        )
    evidence_replay = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "sealed_plan_comparisons_and_winner_training_replayed_"
            "not_release_qualified"
        ),
        "plan": plan_replay,
        "comparison": {
            "comparator_sha256": comparator_sha,
            "confidence": ADVANCEMENT_CONFIDENCE,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "records": comparison_replay_records,
            "winner_slot": winner_name,
            "winner_pipeline": winner_pipeline,
        },
        "winner_training_results": {
            "scale": winner_scale,
            "bytes": candidate_training_record["bytes"],
            "sha256": candidate_training_record["sha256"],
            "completed_epochs": completed_epoch,
            "results_rows": result_rows,
        },
        "qualification": dict(QUALIFICATION_RECORD),
    }
    runtime = _winner_runtime_contract(
        evaluation=evaluation,
        evaluation_sha256=evaluation_sha256,
        candidate=candidate,
        winner_slot=winner_slot,
        winner_pipeline=winner_pipeline,
        detail_crop_size=detail_crop,
        tournament_device=str(device),
        tournament_require_full_provider=bool(fixed["require_full_provider"]),
    )
    return {
        **runtime,
        "tournament_selection_sha256": selection_sha256,
        "tournament_selection_content_sha256": selection["selection_content_sha256"],
        "winner_slot": winner_name,
        "evidence_replay": evidence_replay,
        "comparison_sources": comparison_sources,
        "sealed_input_sources": sealed_input_sources,
        "copy_records": {
            **comparison_copy_records,
            **sealed_copy_records,
        },
    }


def _release_manifest(
    project_root: Path,
) -> tuple[Path, bytes, dict[str, str]]:
    path = _regular_file(
        project_root.joinpath(*MODEL_MANIFEST_RELATIVE.parts),
        "release model SHA-256 manifest",
        suffix=".sha256",
    )
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise CandidateAdoptionError(f"cannot read release model manifest: {exc}") from exc
    if not payload.endswith(b"\n"):
        raise CandidateAdoptionError("release model manifest must end with one newline")
    records: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (models/.+)", line)
        if match is None:
            raise CandidateAdoptionError(
                f"release model manifest line {line_number} is not canonical"
            )
        digest, relative = match.groups()
        try:
            relative = canonical_relative_path(relative, "release manifest member")
        except ReleaseModelContractError as exc:
            raise CandidateAdoptionError(str(exc)) from exc
        if relative in records:
            raise CandidateAdoptionError(
                f"release model manifest repeats {relative!r}"
            )
        target = _regular_file(project_root / relative, f"release manifest member {relative}")
        if _sha256_file(target) != digest:
            raise CandidateAdoptionError(
                f"release model manifest SHA-256 mismatch: {relative}"
            )
        records[relative] = digest
    return path, payload, records


def _copy_verified_new(
    source: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    description: str,
) -> None:
    """Copy one immutable snapshot and reject any transient source mutation."""

    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise CandidateAdoptionError(f"{description} expected size is invalid")
    _sha256_value(expected_sha256, f"{description} expected hash")
    digest = sha256()
    copied = 0
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                output_stream.write(block)
                digest.update(block)
                copied += len(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except OSError as exc:
        raise CandidateAdoptionError(
            f"cannot stage release-default member {destination.name}: {exc}"
        ) from exc
    if copied != expected_bytes or digest.hexdigest() != expected_sha256:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise CandidateAdoptionError(
                f"{description} changed while copying and the rejected staged "
                f"member could not be removed: {exc}"
            ) from exc
        raise CandidateAdoptionError(
            f"{description} changed while copying; expected {expected_bytes} bytes/"
            f"{expected_sha256}, copied {copied} bytes/{digest.hexdigest()}"
        )


def _write_new(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        raise CandidateAdoptionError(f"cannot stage {path.name}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise CandidateAdoptionError("atomic no-replace publication is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise CandidateAdoptionError(
                    f"release-default identity already exists: {destination.name}"
                )
            raise CandidateAdoptionError(
                f"atomic release-default publication failed: {os.strerror(error)}"
            )
        return
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise CandidateAdoptionError(
                f"release-default identity already exists: {destination.name}"
            ) from exc
        return
    raise CandidateAdoptionError(
        "atomic no-replace directory publication is supported only on Linux and Windows"
    )


def _atomic_replace(path: Path, payload: bytes) -> None:
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.adoption-", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise CandidateAdoptionError(f"cannot atomically replace {path}: {exc}") from exc
    finally:
        if _path_lexists(temporary):
            temporary.unlink()


def _artifact_record(path: Path, relative: PurePosixPath) -> dict[str, Any]:
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _assert_inputs_unchanged(snapshots: Mapping[Path, str]) -> None:
    for path, digest in snapshots.items():
        if _sha256_file(path) != digest:
            raise CandidateAdoptionError(
                f"adoption input changed before publication: {path.name}"
            )


def adopt_candidate(
    *,
    candidate: Path,
    candidate_evaluation: Path,
    tournament_selection: Path,
    project_root: Path = PROJECT_ROOT,
    validate_only: bool = False,
    candidate_validator: Callable[..., Mapping[str, Any]] = validate_staged_candidate,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise CandidateAdoptionError(f"project root must be a regular directory: {root}")
    candidate_root = candidate.expanduser().absolute()
    _reject_symlink_components(candidate_root, "candidate")
    if not candidate_root.is_dir() or candidate_root.is_symlink():
        raise CandidateAdoptionError(
            f"candidate must be a non-symlink directory: {candidate_root}"
        )
    candidate_root = candidate_root.resolve()
    manifest_path, manifest, _manifest_payload, candidate_manifest_sha = (
        _canonical_json_file(
            candidate_root / CANDIDATE_MANIFEST_NAME,
            "candidate manifest",
        )
    )
    candidate_record = _candidate_contract(candidate_root, manifest)
    try:
        validation = candidate_validator(candidate_root)
    except CandidateExportError as exc:
        raise CandidateAdoptionError(f"staged candidate validation failed: {exc}") from exc
    if (
        not isinstance(validation, Mapping)
        or validation.get("status") != "validated"
        or validation.get("candidate_content_sha256")
        != candidate_record["candidate_content_sha256"]
    ):
        raise CandidateAdoptionError(
            "staged candidate validator did not confirm this exact candidate"
        )

    evaluation_path, evaluation, _evaluation_payload, evaluation_sha = (
        _canonical_json_file(
            candidate_evaluation,
            "candidate runtime evaluation",
            default_name=METRICS_NAME,
        )
    )
    selection_path, selection_record, selection_payload, selection_sha = (
        _canonical_json_file(
            tournament_selection,
            "model-tournament selection manifest",
            default_name=SELECTION_MANIFEST_NAME,
        )
    )
    selection = _tournament_selection_contract(
        selection_path=selection_path,
        selection=selection_record,
        selection_sha256=selection_sha,
        evaluation=evaluation,
        evaluation_sha256=evaluation_sha,
        candidate=candidate_record,
        candidate_manifest_sha256=candidate_manifest_sha,
        project_root=root,
    )
    current_pointer = load_release_default_contract(root, verify_files=True)
    manifest_target, manifest_snapshot, manifest_records = _release_manifest(root)
    pointer_target = root.joinpath(*CONTRACT_RELATIVE.parts)
    pointer_snapshot = _regular_file(
        pointer_target, "current release-default pointer", suffix=".json"
    ).read_bytes()

    release_id = str(candidate_record["candidate_content_sha256"])
    release_relative = RELEASES_RELATIVE / release_id
    release_target = root.joinpath(*release_relative.parts)
    selected_height = candidate_record["input_shape_nchw"][2]
    selected_width = candidate_record["input_shape_nchw"][3]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "validated_for_adoption_not_published"
            if validate_only
            else "adopted_for_frozen_hardware_eligibility_not_release_qualified"
        ),
        "release_identity": release_id,
        "input_shape_nchw": list(candidate_record["input_shape_nchw"]),
        "selected_runtime": {
            key: value
            for key, value in selection.items()
            if key not in INTERNAL_SELECTION_KEYS
        },
        "candidate_content_sha256": release_id,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "tournament_selection_sha256": selection_sha,
        "previous_pointer_content_sha256": current_pointer["content_sha256"],
        "qualification": dict(QUALIFICATION_RECORD),
    }
    if validate_only:
        return result

    releases_root = root.joinpath(*RELEASES_RELATIVE.parts)
    models_root = root / "models"
    _reject_symlink_components(models_root, "models directory")
    releases_root.mkdir(parents=True, exist_ok=True)
    if releases_root.is_symlink():
        raise CandidateAdoptionError("release-default asset directory must not be a symlink")
    if _path_lexists(release_target):
        raise CandidateAdoptionError(
            f"release-default identity already exists; refusing overwrite: {release_target}"
        )
    lock_path = models_root / LOCK_NAME
    try:
        lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CandidateAdoptionError(
            f"another adoption is active or left a lock: {lock_path}"
        ) from exc
    os.close(lock_descriptor)

    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}.staging-", dir=releases_root))
    published_directory = False
    try:
        basename = str(candidate_record["basename"])
        copied_paths = {
            "onnx": staging / f"{basename}.onnx",
            "openvino_xml": staging / f"{basename}.xml",
            "openvino_bin": staging / f"{basename}.bin",
            "labels": staging / "labels.txt",
            "attribution": staging / "ATTRIBUTION.md",
        }
        for role, target in copied_paths.items():
            source_record = candidate_record["records"][role]
            _copy_verified_new(
                candidate_record["sources"][role],
                target,
                expected_bytes=source_record["bytes"],
                expected_sha256=source_record["sha256"],
                description=f"candidate {role}",
            )
        candidate_receipt_path = staging / CANDIDATE_RECEIPT_NAME
        _write_new(
            candidate_receipt_path,
            canonical_json_bytes(
                _candidate_public_receipt(
                    manifest=manifest,
                    manifest_sha256=candidate_manifest_sha,
                    candidate=candidate_record,
                )
            ),
        )
        training_receipt_path = staging / TRAINING_PROVENANCE_RECEIPT_NAME
        _write_new(
            training_receipt_path,
            canonical_json_bytes(
                _training_public_receipt(
                    manifest=manifest,
                    manifest_sha256=candidate_manifest_sha,
                    candidate=candidate_record,
                )
            ),
        )
        copied_training_results = staging / RESULTS_NAME
        training_results_record = candidate_record["records"]["training_results"]
        _copy_verified_new(
            candidate_record["sources"]["training_results"],
            copied_training_results,
            expected_bytes=training_results_record["bytes"],
            expected_sha256=training_results_record["sha256"],
            description="candidate training_results",
        )
        winner_runtime_receipt_path = staging / WINNER_RUNTIME_RECEIPT_NAME
        _write_new(
            winner_runtime_receipt_path,
            canonical_json_bytes(
                _winner_runtime_public_receipt(
                    evaluation=evaluation,
                    evaluation_sha256=evaluation_sha,
                    candidate=candidate_record,
                    selection=selection,
                )
            ),
        )
        copied_selection = staging / COPIED_SELECTION_NAME
        _write_new(copied_selection, selection_payload)
        copied_comparisons: dict[str, Path] = {}
        for role, source in selection["comparison_sources"].items():
            target = staging / f"{role.upper().replace('_', '-')}.json"
            source_record = selection["copy_records"][role]
            _copy_verified_new(
                source,
                target,
                expected_bytes=source_record["bytes"],
                expected_sha256=source_record["sha256"],
                description=role,
            )
            copied_comparisons[role] = target
        copied_sealed_inputs: dict[str, Path] = {}
        for role, source in selection["sealed_input_sources"].items():
            target = staging / (
                f"{role.upper().replace('_', '-')}{source.suffix.casefold()}"
            )
            source_record = selection["copy_records"][role]
            _copy_verified_new(
                source,
                target,
                expected_bytes=source_record["bytes"],
                expected_sha256=source_record["sha256"],
                description=role,
            )
            copied_sealed_inputs[role] = target

        adoption_source = _regular_file(
            Path(__file__).resolve(), "candidate adoption source", suffix=".py"
        )
        exporter_source = _regular_file(
            root / "scripts" / "export_fort_release_candidate.py",
            "candidate exporter source",
            suffix=".py",
        )
        comparator_source = _regular_file(
            root / "scripts" / "compare_fort_runtime_evaluations.py",
            "runtime comparator source",
            suffix=".py",
        )
        tournament_source = _regular_file(
            root / "scripts" / "run_fort_model_tournament.py",
            "model tournament source",
            suffix=".py",
        )
        contract_source = _regular_file(
            root / "utils" / "release_model_contract.py",
            "release model contract source",
            suffix=".py",
        )
        public_evidence_source = _regular_file(
            root / "utils" / "public_evidence.py",
            "public-evidence privacy validator source",
            suffix=".py",
        )
        adoption_record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "development_selected_default_not_release_qualified",
            "candidate": {
                "candidate_content_sha256": release_id,
                "candidate_manifest_sha256": candidate_manifest_sha,
                "checkpoint_sha256": candidate_record["checkpoint_sha256"],
                "dataset_manifest_sha256": candidate_record[
                    "dataset_manifest_sha256"
                ],
                "dataset_content_sha256": candidate_record[
                    "dataset_content_sha256"
                ],
                "input_shape_nchw": list(candidate_record["input_shape_nchw"]),
                "output_head": candidate_record["head"],
                "source_artifacts": {
                    role: dict(record)
                    for role, record in candidate_record["records"].items()
                },
            },
            "selection": {
                **{
                    key: value
                    for key, value in selection.items()
                    if key not in INTERNAL_SELECTION_KEYS
                },
                "sealed_tournament_winner": True,
                "release_qualified": False,
            },
            "source": {
                "adoption_sha256": _sha256_file(adoption_source),
                "candidate_exporter_sha256": _sha256_file(exporter_source),
                "runtime_comparator_sha256": _sha256_file(comparator_source),
                "release_contract_sha256": _sha256_file(contract_source),
                "public_evidence_sha256": _sha256_file(public_evidence_source),
            },
            "qualification": dict(QUALIFICATION_RECORD),
        }
        adoption_record["content_sha256"] = canonical_hash(adoption_record)
        adoption_path = staging / ADOPTION_NAME
        _write_new(adoption_path, canonical_json_bytes(adoption_record))

        staged_roles = {
            **copied_paths,
            "adoption_record": adoption_path,
            "candidate_receipt": candidate_receipt_path,
            "training_provenance_receipt": training_receipt_path,
            "training_results": copied_training_results,
            "winner_runtime_receipt": winner_runtime_receipt_path,
            "tournament_selection_manifest": copied_selection,
            **copied_comparisons,
            **copied_sealed_inputs,
        }
        final_artifacts = {
            role: _artifact_record(path, release_relative / path.name)
            for role, path in staged_roles.items()
        }
        expected_copied_records = {
            **{
                role: {
                    "bytes": candidate_record["records"][role]["bytes"],
                    "sha256": candidate_record["records"][role]["sha256"],
                }
                for role in (*copied_paths, "training_results")
            },
            "tournament_selection_manifest": {
                "bytes": len(selection_payload),
                "sha256": selection_sha,
            },
            **selection["copy_records"],
        }
        for role, expected_record in expected_copied_records.items():
            staged_record = final_artifacts.get(role)
            if (
                not isinstance(staged_record, Mapping)
                or staged_record.get("bytes") != expected_record["bytes"]
                or staged_record.get("sha256") != expected_record["sha256"]
            ):
                raise CandidateAdoptionError(
                    f"staged release-default {role} differs from its validated "
                    "candidate/tournament snapshot"
                )
        label = (
            f"Game players — Selected {selected_height}×{selected_width} (Recommended)"
        )
        pointer = make_release_default_contract(
            label=label,
            description=(
                "Development-selected one-class player detector; frozen-build "
                "target-GPU performance eligibility and then final independent "
                "holdout remain required."
            ),
            input_shape_nchw=candidate_record["input_shape_nchw"],
            detail_crop_size_source_pixels=selection[
                "detail_crop_size_source_pixels"
            ],
            artifacts=final_artifacts,
            provenance={
                "kind": "development_selected_candidate",
                "candidate_content_sha256": release_id,
                "candidate_manifest_sha256": candidate_manifest_sha,
                "tournament_selection_sha256": selection_sha,
            },
        )
        validate_release_default_contract(pointer)
        for role, path in staged_roles.items():
            record = final_artifacts[role]
            if (
                path.stat().st_size != record["bytes"]
                or _sha256_file(path) != record["sha256"]
            ):
                raise CandidateAdoptionError(
                    f"staged release-default {role} changed before publication"
                )
            try:
                path.chmod(0o644)
            except OSError as exc:
                raise CandidateAdoptionError(
                    f"cannot set portable permissions on staged {role}"
                ) from exc
        try:
            staging.chmod(0o755)
        except OSError as exc:
            raise CandidateAdoptionError(
                "cannot set portable permissions on staged release directory"
            ) from exc
        _fsync_directory(staging)

        snapshots = {
            manifest_path: candidate_manifest_sha,
            evaluation_path: evaluation_sha,
            selection_path: selection_sha,
            adoption_source: _sha256_file(adoption_source),
            exporter_source: _sha256_file(exporter_source),
            comparator_source: _sha256_file(comparator_source),
            tournament_source: _sha256_file(tournament_source),
            contract_source: _sha256_file(contract_source),
            public_evidence_source: _sha256_file(public_evidence_source),
        }
        snapshots.update(
            {
                path: _sha256_file(path)
                for path in selection["comparison_sources"].values()
            }
        )
        snapshots.update(
            {
                path: _sha256_file(path)
                for path in selection["sealed_input_sources"].values()
            }
        )
        snapshots.update(
            {
                path: str(candidate_record["records"][role]["sha256"])
                for role, path in candidate_record["sources"].items()
            }
        )
        _assert_inputs_unchanged(snapshots)
        current_runtime_identity = _source_hash_snapshot("onnxruntime")
        if selection_record.get("runtime_evaluator") != {
            "path": "evaluate_fort_runtime_model.py",
            "sha256": current_runtime_identity["evaluator"]["sha256"],
            "pipeline_source_sha256": current_runtime_identity["pipeline"],
        }:
            raise CandidateAdoptionError(
                "runtime evaluator/pipeline changed before adoption publication"
            )
        if manifest_target.read_bytes() != manifest_snapshot:
            raise CandidateAdoptionError("release model manifest changed during adoption")
        if pointer_target.read_bytes() != pointer_snapshot:
            raise CandidateAdoptionError("release-default pointer changed during adoption")

        # Replay the complete staged inventory at the final publication
        # boundary, after every potentially long source/snapshot check. This is
        # deliberately redundant with the earlier construction check.
        for role, path in staged_roles.items():
            record = final_artifacts[role]
            if (
                path.stat().st_size != record["bytes"]
                or _sha256_file(path) != record["sha256"]
            ):
                raise CandidateAdoptionError(
                    f"staged release-default {role} changed immediately before "
                    "atomic publication"
                )

        _rename_directory_noreplace(staging, release_target)
        published_directory = True
        _fsync_directory(releases_root)
        if manifest_target.read_bytes() != manifest_snapshot:
            raise CandidateAdoptionError(
                "release model manifest changed before additive publication"
            )
        if pointer_target.read_bytes() != pointer_snapshot:
            raise CandidateAdoptionError(
                "release-default pointer changed before additive publication"
            )
        for record in final_artifacts.values():
            relative = str(record["path"])
            if relative in manifest_records:
                raise CandidateAdoptionError(
                    f"new release-default member already appears in manifest: {relative}"
                )
            manifest_records[relative] = str(record["sha256"])
        new_manifest = manifest_snapshot + b"".join(
            f"{record['sha256']}  {record['path']}\n".encode("utf-8")
            for record in final_artifacts.values()
        )
        _atomic_replace(manifest_target, new_manifest)
        try:
            (
                verified_manifest_path,
                verified_manifest_bytes,
                verified_manifest_records,
            ) = _release_manifest(root)
            if (
                verified_manifest_path != manifest_target
                or verified_manifest_bytes != new_manifest
                or any(
                    verified_manifest_records.get(str(record["path"]))
                    != record["sha256"]
                    for record in final_artifacts.values()
                )
            ):
                raise CandidateAdoptionError(
                    "published release manifest differs from the staged additive update"
                )
        except Exception:
            _atomic_replace(manifest_target, manifest_snapshot)
            raise
        if pointer_target.read_bytes() != pointer_snapshot:
            _atomic_replace(manifest_target, manifest_snapshot)
            raise CandidateAdoptionError(
                "release-default pointer changed before its final atomic swap"
            )
        try:
            _atomic_replace(pointer_target, canonical_json_bytes(pointer))
        except Exception:
            # The old pointer is still authoritative. Restore the old additive
            # manifest on ordinary failures; a process crash between the two
            # writes is also safe because extra manifest entries are inert.
            _atomic_replace(manifest_target, manifest_snapshot)
            raise
        try:
            loaded = load_release_default_contract(root, verify_files=True)
            if loaded != pointer:
                raise CandidateAdoptionError(
                    "published release-default pointer differs from its staged contract"
                )
        except Exception as exc:
            try:
                _atomic_replace(pointer_target, pointer_snapshot)
                _atomic_replace(manifest_target, manifest_snapshot)
            except Exception as rollback_exc:
                raise CandidateAdoptionError(
                    "published release-default verification failed and rollback failed"
                ) from rollback_exc
            if isinstance(exc, CandidateAdoptionError):
                raise
            raise CandidateAdoptionError(
                "published release-default verification failed; old pointer restored"
            ) from exc
        result["pointer_content_sha256"] = pointer["content_sha256"]
        result["artifacts"] = final_artifacts
        return result
    finally:
        if not published_directory and _path_lexists(staging):
            shutil.rmtree(staging)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = adopt_candidate(
            candidate=args.candidate,
            candidate_evaluation=args.candidate_evaluation,
            tournament_selection=args.tournament_selection,
            project_root=args.project_root,
            validate_only=args.validate_only,
        )
    except (CandidateAdoptionError, ReleaseModelContractError) as exc:
        print(f"Candidate adoption failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
