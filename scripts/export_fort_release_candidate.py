#!/usr/bin/env python3
"""Stage and verify a reviewed FORT checkpoint without touching release assets.

The command writes a brand-new, self-contained candidate directory containing
one static ONNX graph, one static OpenVINO IR pair, the exact one-class label
file, attribution, and a cryptographic provenance record.  It deliberately
copies the checkpoint into a private temporary directory before calling
Ultralytics because that exporter normally writes beside the input weights.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.postprocess import supported_yolo_output_layout  # noqa: E402
from scripts.fort_dataset_contract import (  # noqa: E402
    DatasetContractError,
    verify_dataset_contract,
    verify_grouped_dataset_metadata,
)
from utils.inference_size import (  # noqa: E402
    InferenceSize,
    compact_inference_size,
    format_inference_size,
    parse_inference_size,
    validate_yolo_inference_size,
)


SCHEMA_VERSION = 1
DEFAULT_INFERENCE_SIZE = (384, 640)
DEFAULT_BASENAME = "fort_player_candidate"
DEFAULT_SEEDS = (0, 2026, 8675309, 429496729)
DEFAULT_PARITY_CONFIDENCE_FLOOR = 0.001
DEFAULT_PARITY_ATOL = 0.002
DEFAULT_PARITY_RTOL = 0.0001
AUDITED_V9_MANIFEST_SHA256 = "f09ad355ead4a4dd4504f550f0b390786950faa6702fcd065c6849d765dbdffb"
AUDITED_V9_CONTENT_SHA256 = "b2979f0ea75e5245944076aab51636ab7173a6e5ef1108d0cfb2e3f0549e0255"
LABEL_TEXT = "player\n"
MANIFEST_NAME = "candidate-manifest.json"
ATTRIBUTION_NAME = "ATTRIBUTION.md"
LABELS_NAME = "labels.txt"
STAGING_MARKER = ".proaim-fort-candidate-incomplete"
INITIAL_CONTRACT_NAME = "training-initial-run-contract.json"
REPRODUCIBILITY_NAME = "training-reproducibility.json"
RESULTS_NAME = "training-results.csv"
RESULT_COLUMN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./()\-]{0,127}$")
MEMBER_NAMES = (
    ATTRIBUTION_NAME,
    LABELS_NAME,
    MANIFEST_NAME,
)
PACKAGE_NAMES = (
    "numpy",
    "onnx",
    "onnxruntime",
    "openvino",
    "torch",
    "ultralytics",
)


class CandidateExportError(RuntimeError):
    """Raised when a candidate cannot be staged or verified safely."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CandidateExportError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
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
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _reject_symlink_components(path: Path, description: str) -> None:
    """Reject an existing symlink anywhere in a security-sensitive path."""

    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if _path_lexists(component) and component.is_symlink():
            raise CandidateExportError(
                f"{description} path contains a symlink component: {component}"
            )


def _require_regular_file(path: Path, description: str, suffix: str | None = None) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    _reject_symlink_components(expanded, description)
    try:
        if not expanded.is_file() or expanded.is_symlink():
            raise CandidateExportError(
                f"{description} must be a local regular file, not a symlink: {expanded}"
            )
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise CandidateExportError(f"cannot inspect {description} {expanded}: {exc}") from exc
    if suffix is not None and resolved.suffix.casefold() != suffix.casefold():
        raise CandidateExportError(f"{description} must use the {suffix} extension: {resolved}")
    return resolved


def _require_new_directory(path: Path, description: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    absolute = expanded.absolute()
    _reject_symlink_components(absolute.parent, description)
    if _path_lexists(absolute):
        raise CandidateExportError(f"{description} already exists; refusing overwrite: {absolute}")
    return absolute


def _basename(value: str) -> str:
    candidate = str(value).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", candidate):
        raise argparse.ArgumentTypeError(
            "must be a 1-80 character lowercase stem using a-z, digits, _ or -"
        )
    return candidate


def _inference_size(value: str) -> InferenceSize:
    try:
        return validate_yolo_inference_size(parse_inference_size(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _finite_nonnegative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and zero or greater")
    return parsed


def _finite_probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and from zero to one")
    return parsed


def _seed(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 4294967295")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Reviewed local .pt checkpoint.")
    parser.add_argument(
        "--data",
        type=Path,
        help="Audited grouped fort_cuh_grouped.yaml used to train the checkpoint.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inference-size",
        type=_inference_size,
        default=DEFAULT_INFERENCE_SIZE,
        metavar="N|HEIGHTxWIDTH",
        help="Exact static batch-one tensor shape (default: 384x640).",
    )
    parser.add_argument("--basename", type=_basename, default=DEFAULT_BASENAME)
    parser.add_argument(
        "--head",
        choices=("end2end", "traditional"),
        default="end2end",
        help="Select the YOLO26 head explicitly; unsupported checkpoints fail closed.",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        default=AUDITED_V9_MANIFEST_SHA256,
        help="Pinned reviewed grouped-manifest SHA-256 (defaults to audited v9).",
    )
    parser.add_argument(
        "--expected-content-sha256",
        default=AUDITED_V9_CONTENT_SHA256,
        help="Pinned exact grouped-content SHA-256 (defaults to audited v9).",
    )
    parser.add_argument("--opset", type=int, help="Optional explicit ONNX opset.")
    parser.add_argument(
        "--seeds",
        type=_seed,
        nargs="+",
        help="Parity tensor seeds (default: four pinned seeds; validate-only reuses the manifest).",
    )
    parser.add_argument(
        "--parity-confidence-floor",
        type=_finite_probability,
        default=DEFAULT_PARITY_CONFIDENCE_FLOOR,
        help="End-to-end rows below this runtime-relevant score are ignored for box parity.",
    )
    parser.add_argument("--parity-atol", type=_finite_nonnegative, default=DEFAULT_PARITY_ATOL)
    parser.add_argument("--parity-rtol", type=_finite_nonnegative, default=DEFAULT_PARITY_RTOL)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Re-hash and re-run both app backends on an already staged candidate.",
    )
    return parser


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateExportError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateExportError(f"{description} must be a JSON object: {path}")
    return value


def _validate_dataset(
    data: Path,
    *,
    expected_manifest_sha256: str,
    expected_content_sha256: str,
) -> dict[str, Any]:
    data = _require_regular_file(data, "grouped dataset YAML", ".yaml")
    dataset_attribution = _require_regular_file(
        data.parent / ATTRIBUTION_NAME, "grouped dataset attribution", ".md"
    )
    try:
        attribution_text = dataset_attribution.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CandidateExportError(f"cannot read grouped dataset attribution: {exc}") from exc
    # Markdown line wrapping is not a semantic attribution change. Normalize
    # whitespace before checking the audited notice so the actual v9 document's
    # ``independent\ntarget-clone`` wrap does not make a valid export impossible.
    normalized_attribution = " ".join(attribution_text.split()).casefold()
    for marker in (
        "FORT-Cuh v1",
        "Aviles Joseph",
        "creativecommons.org/licenses/by/4.0",
        "independent target-clone footage",
    ):
        if marker.casefold() not in normalized_attribution:
            raise CandidateExportError(
                f"grouped dataset attribution is missing required marker {marker!r}"
            )
    manifest_path = _require_regular_file(
        data.parent / "manifest.json", "grouped dataset manifest", ".json"
    )
    manifest = _read_json_object(manifest_path, "grouped dataset manifest")
    if manifest.get("schema_version") != 1:
        raise CandidateExportError("grouped dataset manifest schema is unsupported")
    if manifest.get("cross_split_source_groups") != 0:
        raise CandidateExportError(
            "grouped dataset manifest does not prove zero cross-split source groups"
        )
    if manifest.get("cross_split_visual_similarity_edges") != 0:
        raise CandidateExportError(
            "grouped dataset manifest does not prove zero cross-split visual edges"
        )
    if manifest.get("runtime_class_labels") != ["player"]:
        raise CandidateExportError(
            "grouped dataset manifest must define exactly one runtime class: player"
        )
    try:
        verify_grouped_dataset_metadata(data)
        contract = verify_dataset_contract(data.parent, manifest.get("dataset_contract"))
    except DatasetContractError as exc:
        raise CandidateExportError(f"grouped dataset exact-file audit failed: {exc}") from exc
    source = manifest.get("source")
    if not isinstance(source, Mapping) or source.get("license") != "CC BY 4.0":
        raise CandidateExportError("dataset manifest must retain its stated CC BY 4.0 source")
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise CandidateExportError(
            "grouped dataset manifest is not the explicitly reviewed manifest: "
            f"{manifest_sha256} != {expected_manifest_sha256}"
        )
    if contract["content_sha256"] != expected_content_sha256:
        raise CandidateExportError(
            "grouped dataset content is not the explicitly reviewed content: "
            f"{contract['content_sha256']} != {expected_content_sha256}"
        )
    return {
        "yaml_path": str(data),
        "yaml_sha256": _sha256_file(data),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "attribution_path": str(dataset_attribution),
        "attribution_sha256": _sha256_file(dataset_attribution),
        "content_sha256": contract["content_sha256"],
        "contract_schema_version": contract["schema_version"],
        "source_archive_sha256": manifest.get("source_archive_sha256"),
        "source": dict(source),
        "splits": {
            name: {
                "images": contract["splits"][name]["images"],
                "boxes": contract["splits"][name]["boxes"],
                "content_sha256": contract["splits"][name]["content_sha256"],
            }
            for name in ("train", "valid", "test")
        },
        "limitations": {
            "reviewed_negative_images": len(manifest.get("reviewed_negative_images", [])),
            "far_object_distribution": manifest.get("assignment_balance", {}).get(
                "far_object_distribution"
            ),
            "test_consumed_during_development": True,
            "independent_target_clone_holdout_required": True,
        },
    }


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CandidateExportError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _recorded_path(value: object, description: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise CandidateExportError(f"training provenance has invalid {description}")
    return Path(value).expanduser().resolve()


def _completed_results_epoch(path: Path) -> tuple[int, int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if (
                not reader.fieldnames
                or any(not isinstance(name, str) or not name for name in reader.fieldnames)
                or len(set(reader.fieldnames)) != len(reader.fieldnames)
            ):
                raise CandidateExportError("training results.csv has invalid columns")
            if any(
                RESULT_COLUMN_PATTERN.fullmatch(name) is None
                or name.startswith(("/", "."))
                or name.endswith(("/", "."))
                or "//" in name
                or ".." in name
                for name in reader.fieldnames
            ):
                raise CandidateExportError(
                    "training results.csv has unsafe or non-portable columns"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CandidateExportError(f"cannot read training results.csv: {exc}") from exc
    if not rows:
        raise CandidateExportError("training results.csv contains no completed epoch")
    try:
        raw_epochs = [float(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateExportError("training results.csv has invalid epoch values") from exc
    if not all(math.isfinite(value) and value.is_integer() for value in raw_epochs):
        raise CandidateExportError("training results.csv has invalid epoch values")
    epochs = [int(value) for value in raw_epochs]
    if epochs != list(range(1, len(epochs) + 1)):
        raise CandidateExportError(
            f"training results.csv epoch sequence is not contiguous: {epochs!r}"
        )
    assert reader.fieldnames is not None
    for name in reader.fieldnames:
        if name == "epoch":
            continue
        try:
            values = [float(row[name]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise CandidateExportError(
                f"training results.csv has invalid values for {name!r}"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise CandidateExportError(
                f"training results.csv has non-finite values for {name!r}"
            )
    return epochs[-1], len(rows)


def _validate_initial_training_contract(
    contract: Mapping[str, Any],
    *,
    run_dir: Path,
    data: Path,
    dataset: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any]]:
    if contract.get("schema_version") != 1:
        raise CandidateExportError("initial training contract schema is unsupported")
    if _recorded_path(contract.get("run_dir"), "initial run directory") != run_dir:
        raise CandidateExportError("initial training contract does not identify the checkpoint run")
    if _recorded_path(contract.get("data"), "initial dataset path") != data:
        raise CandidateExportError("initial training contract does not identify the reviewed dataset")
    for contract_key, dataset_key in (
        ("dataset_manifest_sha256", "manifest_sha256"),
        ("dataset_content_sha256", "content_sha256"),
        ("dataset_yaml_sha256", "yaml_sha256"),
    ):
        if contract.get(contract_key) != dataset.get(dataset_key):
            raise CandidateExportError(
                f"initial training contract {contract_key} does not match the reviewed dataset"
            )

    initial_weights = _require_regular_file(
        _recorded_path(contract.get("initial_weights"), "initial weights"),
        "initial pretrained weights",
        ".pt",
    )
    initial_weights_sha256 = _require_sha256(
        contract.get("initial_weights_sha256"), "initial pretrained weights hash"
    )
    if _sha256_file(initial_weights) != initial_weights_sha256:
        raise CandidateExportError("initial pretrained weights changed after training")

    training = contract.get("training")
    arguments = contract.get("training_arguments")
    environment = contract.get("environment")
    if not all(isinstance(value, Mapping) for value in (training, arguments, environment)):
        raise CandidateExportError("initial training contract is incomplete")
    assert isinstance(training, Mapping) and isinstance(arguments, Mapping)
    if training.get("smoke_test") is not False:
        raise CandidateExportError("smoke-test checkpoints cannot become release candidates")
    planned_epochs = training.get("epochs")
    if isinstance(planned_epochs, bool) or not isinstance(planned_epochs, int) or planned_epochs <= 0:
        raise CandidateExportError("initial training contract has invalid planned epochs")
    if training.get("cache") != "none" or arguments.get("cache") is not False:
        raise CandidateExportError("release training must use the exact no-cache dataset contract")
    expected_argument_values = {
        "data": str(data),
        "epochs": planned_epochs,
        "patience": training.get("patience"),
        "batch": training.get("batch"),
        "imgsz": training.get("imgsz"),
        "device": training.get("device"),
        "workers": training.get("workers"),
        "seed": training.get("seed"),
        "deterministic": True,
        "amp": False,
        "rect": False,
        "single_cls": False,
        "save": True,
        "save_period": 1,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
    }
    differing = sorted(
        key for key, expected in expected_argument_values.items() if arguments.get(key) != expected
    )
    if differing:
        raise CandidateExportError(
            f"initial training arguments are inconsistent with the run contract: {differing}"
        )
    for key in ("ultralytics", "torch"):
        if not isinstance(environment.get(key), str) or not environment.get(key):
            raise CandidateExportError(f"initial training contract has no {key} version")

    script_status = contract.get("training_script_sha256_status")
    adoption = contract.get("adoption")
    if script_status == "captured_at_launch":
        _require_sha256(contract.get("training_script_sha256"), "launch training script hash")
        if adoption is not None:
            raise CandidateExportError("ordinary initial training contract has unexpected adoption data")
    elif script_status == "unavailable_pre_contract_run":
        if contract.get("training_script_sha256") is not None or not isinstance(adoption, Mapping):
            raise CandidateExportError("adopted training contract has invalid launch-script provenance")
        adopted_epoch = adoption.get("adopted_checkpoint_epoch")
        if isinstance(adopted_epoch, bool) or not isinstance(adopted_epoch, int) or adopted_epoch <= 0:
            raise CandidateExportError("adopted training contract has invalid checkpoint epoch")
        adopted_sha = _require_sha256(
            adoption.get("adopted_checkpoint_sha256"), "adopted checkpoint hash"
        )
        if adoption.get("epoch_checkpoint_sha256") != adopted_sha:
            raise CandidateExportError("adopted checkpoint and epoch-copy hashes differ")
        epoch_copy = _require_regular_file(
            run_dir / "weights" / f"epoch{adopted_epoch - 1}.pt",
            "adopted epoch checkpoint",
            ".pt",
        )
        if _sha256_file(epoch_copy) != adopted_sha:
            raise CandidateExportError("adopted epoch checkpoint no longer matches its audit contract")
        _require_sha256(adoption.get("adoption_script_sha256"), "adoption script hash")
        if adoption.get("checkpoint_version") != environment.get("ultralytics"):
            raise CandidateExportError("adopted checkpoint version differs from the initial contract")
    else:
        raise CandidateExportError("initial training contract has invalid script provenance status")
    return initial_weights, arguments


def _validate_training_provenance(
    *,
    weights: Path,
    data: Path,
    dataset: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Bind a staged checkpoint to its completed, hash-recorded training run."""

    if weights.name != "best.pt" or weights.parent.name != "weights":
        raise CandidateExportError(
            "reviewed checkpoint must be the exact weights/best.pt from a completed training run"
        )
    run_dir = weights.parent.parent
    contract_path = _require_regular_file(
        run_dir / "initial_run_contract.json", "initial training contract", ".json"
    )
    reproducibility_path = _require_regular_file(
        run_dir / "reproducibility.json", "training reproducibility record", ".json"
    )
    results_path = _require_regular_file(run_dir / "results.csv", "training results", ".csv")
    contract = _read_json_object(contract_path, "initial training contract")
    initial_weights, initial_arguments = _validate_initial_training_contract(
        contract,
        run_dir=run_dir,
        data=data,
        dataset=dataset,
    )
    reproducibility = _read_json_object(reproducibility_path, "training reproducibility record")
    if reproducibility.get("schema_version") != 1:
        raise CandidateExportError("training reproducibility schema is unsupported")
    training = reproducibility.get("training")
    arguments = reproducibility.get("training_arguments")
    inputs = reproducibility.get("inputs")
    output = reproducibility.get("output")
    environment = reproducibility.get("environment")
    if not all(
        isinstance(value, Mapping)
        for value in (training, arguments, inputs, output, environment)
    ):
        raise CandidateExportError("training reproducibility record is incomplete")
    assert all(
        isinstance(value, Mapping)
        for value in (training, arguments, inputs, output, environment)
    )
    training = training  # type: ignore[assignment]
    arguments = arguments  # type: ignore[assignment]
    inputs = inputs  # type: ignore[assignment]
    output = output  # type: ignore[assignment]
    environment = environment  # type: ignore[assignment]

    expected_training = contract["training"]
    assert isinstance(expected_training, Mapping)
    for key in (
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
    ):
        if training.get(key) != expected_training.get(key):
            raise CandidateExportError(f"training reproducibility differs for {key}")
    for key, expected in (
        ("data", data),
        ("weights", initial_weights),
        ("project", run_dir.parent),
        ("run_dir", run_dir),
    ):
        if _recorded_path(training.get(key), f"reproducibility {key}") != expected:
            raise CandidateExportError(f"training reproducibility path differs for {key}")
    if training.get("name") != run_dir.name or arguments != initial_arguments:
        raise CandidateExportError("training reproducibility arguments do not match the initial contract")
    if training.get("adopt_interrupted_run") is not False:
        raise CandidateExportError("reproducibility record represents adoption, not completed training")

    contract_sha256 = _sha256_file(contract_path)
    if inputs.get("initial_run_contract_sha256") != contract_sha256:
        raise CandidateExportError("training reproducibility does not seal the initial run contract")
    if _recorded_path(inputs.get("initial_run_contract"), "recorded initial contract") != contract_path:
        raise CandidateExportError("training reproducibility points at a different initial run contract")
    for input_key, expected in (
        ("dataset_manifest_sha256", dataset["manifest_sha256"]),
        ("dataset_content_sha256", dataset["content_sha256"]),
        ("dataset_yaml_sha256", dataset["yaml_sha256"]),
        ("initial_weights_sha256", contract["initial_weights_sha256"]),
    ):
        if inputs.get(input_key) != expected:
            raise CandidateExportError(f"training reproducibility input hash differs for {input_key}")
    _require_sha256(inputs.get("training_script_sha256"), "final training script hash")
    contract_environment = contract["environment"]
    assert isinstance(contract_environment, Mapping)
    for key in ("ultralytics", "torch"):
        if environment.get(key) != contract_environment.get(key):
            raise CandidateExportError(f"training environment version differs for {key}")

    checkpoint_sha256 = _sha256_file(weights)
    checkpoint_bytes = weights.stat().st_size
    if _recorded_path(output.get("best_weights"), "recorded best weights") != weights:
        raise CandidateExportError("training reproducibility points at a different best checkpoint")
    if output.get("best_weights_sha256") != checkpoint_sha256 or output.get(
        "best_weights_bytes"
    ) != checkpoint_bytes:
        raise CandidateExportError("best checkpoint hash/size differs from training reproducibility")
    if output.get("runtime_class_labels") != ["player"]:
        raise CandidateExportError("training reproducibility does not define one player class")
    if output.get("deployment_inference_size") != expected_training.get("imgsz"):
        raise CandidateExportError("training reproducibility deployment size differs from training")
    if _recorded_path(output.get("results_csv"), "recorded results.csv") != results_path:
        raise CandidateExportError("training reproducibility points at a different results.csv")
    results_sha256 = _sha256_file(results_path)
    results_bytes = results_path.stat().st_size
    completed_epoch, result_rows = _completed_results_epoch(results_path)
    if (
        output.get("results_csv_sha256") != results_sha256
        or output.get("results_csv_bytes") != results_bytes
        or output.get("completed_epochs") != completed_epoch
    ):
        raise CandidateExportError("training results differ from the reproducibility record")
    planned_epochs = expected_training.get("epochs")
    if not isinstance(planned_epochs, int) or completed_epoch > planned_epochs:
        raise CandidateExportError("training results exceed the planned epoch contract")

    resumed_from = inputs.get("resumed_from")
    if isinstance(contract.get("adoption"), Mapping) and resumed_from is None:
        raise CandidateExportError("adopted training run has no stateful-resume provenance")
    if resumed_from is not None:
        if not isinstance(resumed_from, Mapping):
            raise CandidateExportError("training resume provenance is invalid")
        if _recorded_path(resumed_from.get("path"), "starting resume checkpoint") != (
            run_dir / "weights" / "last.pt"
        ).resolve():
            raise CandidateExportError("training resume provenance points outside this run")
        if _recorded_path(training.get("resume_from"), "training resume checkpoint") != (
            run_dir / "weights" / "last.pt"
        ).resolve():
            raise CandidateExportError("training configuration resume path differs from its evidence")
        _require_sha256(resumed_from.get("sha256"), "starting resume checkpoint hash")
        start_epoch = resumed_from.get("completed_epoch")
        if isinstance(start_epoch, bool) or not isinstance(start_epoch, int) or not 0 < start_epoch < completed_epoch:
            raise CandidateExportError("training resume provenance has an invalid starting epoch")
        adoption = contract.get("adoption")
        if isinstance(adoption, Mapping) and start_epoch < adoption.get(
            "adopted_checkpoint_epoch", start_epoch
        ):
            raise CandidateExportError("training resume predates the adopted checkpoint")
        if resumed_from.get("sha256_scope") != "captured_before_resume_training":
            raise CandidateExportError("training resume hash was not captured before training")
    elif training.get("resume_from") is not None:
        raise CandidateExportError("training configuration claims an unrecorded resume checkpoint")

    files = {
        "initial_run_contract": contract_path,
        "training_reproducibility": reproducibility_path,
        "training_results": results_path,
    }
    record = {
        "schema_version": 1,
        "checkpoint_role": "completed_run_best",
        "run_dir": str(run_dir),
        "planned_epochs": planned_epochs,
        "completed_epochs": completed_epoch,
        "results_rows": result_rows,
        "initial_weights_sha256": contract["initial_weights_sha256"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "dataset_content_sha256": dataset["content_sha256"],
        "initial_run_contract_sha256": contract_sha256,
        "training_reproducibility_sha256": _sha256_file(reproducibility_path),
        "training_results_sha256": results_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "script_provenance_status": contract.get("training_script_sha256_status"),
        "adopted_interrupted_run": isinstance(contract.get("adoption"), Mapping),
        "packaged_files": {
            "initial_run_contract": INITIAL_CONTRACT_NAME,
            "training_reproducibility": REPRODUCIBILITY_NAME,
            "training_results": RESULTS_NAME,
        },
    }
    return record, files


def _package_records() -> dict[str, dict[str, str]]:
    """Record versions plus wheel metadata/RECORD hashes without hashing gigabytes."""

    packages: dict[str, dict[str, str]] = {}
    for name in PACKAGE_NAMES:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            packages[name] = {"version": "not-installed"}
            continue
        record: dict[str, str] = {"version": distribution.version}
        for metadata_name, key in (("METADATA", "metadata_sha256"), ("RECORD", "record_sha256")):
            value = distribution.read_text(metadata_name)
            if value is not None:
                record[key] = sha256(value.encode("utf-8")).hexdigest()
        packages[name] = record
    return packages


def _export_call(
    model: Any,
    *,
    format_name: str,
    inference_size: InferenceSize,
    head: str,
    opset: int | None,
) -> Path:
    kwargs: dict[str, Any] = {
        "format": format_name,
        "imgsz": compact_inference_size(inference_size),
        "batch": 1,
        "dynamic": False,
        "device": "cpu",
        "end2end": head == "end2end",
        "nms": False,
    }
    if format_name == "onnx":
        kwargs["simplify"] = False
        if opset is not None:
            kwargs["opset"] = opset
    try:
        exported = Path(model.export(**kwargs))
    except Exception as exc:
        raise CandidateExportError(f"Ultralytics {format_name} export failed: {exc}") from exc
    if not exported.is_absolute():
        exported = Path.cwd() / exported
    return exported.resolve()


def _one_ir_pair(directory: Path) -> tuple[Path, Path, Path | None]:
    if not directory.is_dir() or directory.is_symlink():
        raise CandidateExportError(f"OpenVINO exporter returned an unsafe directory: {directory}")
    regular: list[Path] = []
    for child in directory.iterdir():
        if child.is_symlink() or not child.is_file():
            raise CandidateExportError(f"unexpected OpenVINO export member: {child}")
        regular.append(child)
    xml = [path for path in regular if path.suffix.casefold() == ".xml"]
    binary = [path for path in regular if path.suffix.casefold() == ".bin"]
    metadata_files = [path for path in regular if path.name == "metadata.yaml"]
    allowed = set(xml + binary + metadata_files)
    if len(xml) != 1 or len(binary) != 1 or len(metadata_files) > 1 or set(regular) != allowed:
        raise CandidateExportError(
            "OpenVINO export must contain exactly one .xml/.bin pair and optional metadata.yaml"
        )
    if xml[0].stem != binary[0].stem:
        raise CandidateExportError("OpenVINO .xml and .bin export stems do not match")
    return xml[0], binary[0], metadata_files[0] if metadata_files else None


def _copy_regular_file(source: Path, target: Path) -> None:
    source = _require_regular_file(source, "exported artifact")
    if _path_lexists(target):
        raise CandidateExportError(f"refusing to overwrite staged member: {target}")
    try:
        with source.open("rb") as input_file, target.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
    except OSError as exc:
        raise CandidateExportError(f"cannot stage {target.name}: {exc}") from exc


def _export_into_temporary(
    weights: Path,
    work: Path,
    *,
    expected_checkpoint_sha256: str,
    basename: str,
    inference_size: InferenceSize,
    head: str,
    opset: int | None,
    yolo_factory: Callable[[str], Any] | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    scratch = work / ".ultralytics-export-scratch"
    try:
        scratch.mkdir()
    except OSError as exc:
        raise CandidateExportError(f"cannot create isolated export scratch directory: {exc}") from exc
    checkpoint = scratch / "reviewed.pt"
    _copy_regular_file(weights, checkpoint)
    if _sha256_file(checkpoint) != expected_checkpoint_sha256:
        raise CandidateExportError(
            "reviewed checkpoint changed while copied into isolated export scratch"
        )
    if yolo_factory is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise CandidateExportError(
                "Ultralytics is required only for export; install requirements-export.txt"
            ) from exc
        yolo_factory = YOLO
    def load_checkpoint() -> Any:
        try:
            loaded = yolo_factory(str(checkpoint))
        except Exception as exc:
            raise CandidateExportError(f"cannot load reviewed checkpoint: {exc}") from exc
        if getattr(loaded, "task", None) != "detect":
            raise CandidateExportError(
                f"checkpoint task must be detect, got {getattr(loaded, 'task', None)!r}"
            )
        names = getattr(loaded, "names", None)
        if names not in ({0: "player"}, ["player"], ("player",)):
            raise CandidateExportError(
                f"checkpoint must contain exactly class 0='player', got {names!r}"
            )
        graph = getattr(loaded, "model", None)
        if head == "end2end" and not hasattr(graph, "end2end"):
            raise CandidateExportError(
                "checkpoint architecture does not expose a selectable end-to-end head"
            )
        return loaded

    model = load_checkpoint()

    onnx_export = _export_call(
        model,
        format_name="onnx",
        inference_size=inference_size,
        head=head,
        opset=opset,
    )
    if not onnx_export.is_file() or onnx_export.is_symlink() or onnx_export.suffix != ".onnx":
        raise CandidateExportError(f"Ultralytics returned an invalid ONNX artifact: {onnx_export}")
    staged_onnx = work / f"{basename}.onnx"
    _copy_regular_file(onnx_export, staged_onnx)

    # Load a fresh copy so mutations made while exporting ONNX cannot silently
    # change the OpenVINO graph.
    model = load_checkpoint()
    openvino_export = _export_call(
        model,
        format_name="openvino",
        inference_size=inference_size,
        head=head,
        opset=None,
    )
    directory = openvino_export if openvino_export.is_dir() else openvino_export.parent
    source_xml, source_bin, source_metadata = _one_ir_pair(directory)
    staged_xml = work / f"{basename}.xml"
    staged_bin = work / f"{basename}.bin"
    _copy_regular_file(source_xml, staged_xml)
    _copy_regular_file(source_bin, staged_bin)
    staged_metadata = work / "ultralytics-metadata.yaml"
    if source_metadata is not None:
        _copy_regular_file(source_metadata, staged_metadata)

    artifacts = {
        "onnx": staged_onnx,
        "openvino_xml": staged_xml,
        "openvino_bin": staged_bin,
    }
    if source_metadata is not None:
        artifacts["ultralytics_metadata"] = staged_metadata
    export_args = {
        "batch": 1,
        "device": "cpu",
        "dynamic": False,
        "end2end": head == "end2end",
        "head": head,
        "imgsz": list(inference_size),
        "nms": False,
        "onnx_opset": opset,
        "onnx_simplify": False,
    }
    try:
        shutil.rmtree(scratch)
    except OSError as exc:
        raise CandidateExportError(f"cannot remove isolated export scratch directory: {exc}") from exc
    return artifacts, export_args


def _make_detectors(
    *,
    onnx: Path,
    openvino_xml: Path,
    labels: Path,
    inference_size: InferenceSize,
    head: str,
) -> tuple[Any, Any]:
    from detection.onnx_yolo import OnnxRuntimeYoloDetector
    from detection.openvino_yolo import OpenVINOYoloDetector

    return (
        OnnxRuntimeYoloDetector(
            onnx,
            labels,
            device="CPU",
            inference_size=inference_size,
            confidence=DEFAULT_PARITY_CONFIDENCE_FLOOR,
            output_format=head,
        ),
        OpenVINOYoloDetector(
            openvino_xml,
            labels,
            device="CPU",
            inference_size=inference_size,
            confidence=DEFAULT_PARITY_CONFIDENCE_FLOOR,
            output_format=head,
        ),
    )


def _summary_shape(summary: Mapping[str, Any]) -> list[int] | None:
    for key in ("declared_input_shape", "input_shape"):
        value = summary.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            try:
                return [int(item) for item in value]
            except (TypeError, ValueError):
                return None
    return None


def _parity_arrays(
    onnx: np.ndarray,
    openvino: np.ndarray,
    *,
    layout: str,
    confidence_floor: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if onnx.shape != openvino.shape or onnx.dtype != np.float32 or openvino.dtype != np.float32:
        raise CandidateExportError(
            "ONNX/OpenVINO raw output shape or float32 dtype contract differs: "
            f"{onnx.shape}/{onnx.dtype} vs {openvino.shape}/{openvino.dtype}"
        )
    if not np.isfinite(onnx).all() or not np.isfinite(openvino).all():
        raise CandidateExportError("ONNX/OpenVINO output contains a non-finite value")
    if layout == "end2end":
        onnx_classes = onnx[..., 5]
        openvino_classes = openvino[..., 5]
        if not np.array_equal(onnx_classes, openvino_classes):
            raise CandidateExportError("ONNX/OpenVINO end-to-end class IDs differ")
        if np.any(onnx_classes != 0.0):
            raise CandidateExportError("candidate emitted a class ID other than player (0)")
        # TopK leaves unused rows with tiny scores whose coordinates are not a
        # stable semantic result and may be ordered differently by runtimes.
        # Compare every score/class value, and compare boxes for all rows that
        # reach the application's minimum retained confidence.
        score_class_indices = np.array([4, 5])
        score_class_onnx = onnx[..., score_class_indices]
        score_class_openvino = openvino[..., score_class_indices]
        relevant = np.maximum(onnx[..., 4], openvino[..., 4]) >= confidence_floor
        boxes_onnx = onnx[..., :4][relevant]
        boxes_openvino = openvino[..., :4][relevant]
        return (
            np.concatenate((score_class_onnx.reshape(-1), boxes_onnx.reshape(-1))),
            np.concatenate((score_class_openvino.reshape(-1), boxes_openvino.reshape(-1))),
            int(np.count_nonzero(relevant)),
        )
    return onnx.reshape(-1), openvino.reshape(-1), int(np.prod(onnx.shape[1:]))


def _validate_detector_runtime_contract(
    *,
    name: str,
    detector: Any,
    inference_size: InferenceSize,
    head: str,
) -> dict[str, Any]:
    summary = getattr(detector, "runtime_summary", None)
    if not isinstance(summary, Mapping):
        raise CandidateExportError(f"{name} app detector returned no runtime summary")
    expected_input = [1, 3, *inference_size]
    if _summary_shape(summary) != expected_input:
        raise CandidateExportError(
            f"{name} app detector input contract is {_summary_shape(summary)}, "
            f"expected {expected_input}"
        )
    if summary.get("output_format") != head:
        raise CandidateExportError(
            f"{name} app detector configured output format differs from {head!r}"
        )
    if name == "onnxruntime":
        declared = summary.get("declared_input_shape")
        if declared != expected_input:
            raise CandidateExportError(
                f"ONNX graph must declare exact static input {expected_input}, got {declared!r}"
            )
        output_layout = summary.get("output_layout")
        if output_layout is not None:
            if head == "end2end" and output_layout != "end2end":
                raise CandidateExportError("ONNX graph does not expose the selected end-to-end head")
            if head == "traditional" and not str(output_layout).startswith("traditional_"):
                raise CandidateExportError("ONNX graph does not expose the selected traditional head")
    else:
        # The OpenVINO detector can reshape a graph at load, so its compiled
        # shape alone does not prove this release artifact is static. Inspect
        # the on-disk IR and require the exact declared dimensions too.
        from scripts.evaluate_fort_runtime_model import _openvino_declared_shape

        declared = _openvino_declared_shape(Path(detector.model_path))
        if declared != expected_input:
            raise CandidateExportError(
                f"OpenVINO IR must declare exact static input {expected_input}, got {declared!r}"
            )
    return {
        key: summary.get(key)
        for key in (
            "runtime",
            "onnxruntime_version",
            "openvino_version",
            "input_shape",
            "declared_input_shape",
            "configured_input_shape",
            "output_shape",
            "output_layout",
            "output_format",
            "device",
            "active_providers",
        )
        if key in summary
    }


def validate_runtime_parity(
    *,
    artifacts: Mapping[str, Path],
    labels: Path,
    inference_size: InferenceSize,
    head: str,
    seeds: Sequence[int],
    confidence_floor: float,
    atol: float,
    rtol: float,
    detector_factory: Callable[..., tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise CandidateExportError("parity requires at least three distinct seeded tensors")
    factory = detector_factory or _make_detectors
    onnx_detector, openvino_detector = factory(
        onnx=artifacts["onnx"],
        openvino_xml=artifacts["openvino_xml"],
        labels=labels,
        inference_size=inference_size,
        head=head,
    )
    expected_input = [1, 3, *inference_size]
    summaries: dict[str, dict[str, Any]] = {}
    for name, detector in (("onnxruntime", onnx_detector), ("openvino", openvino_detector)):
        # Test doubles deliberately have no model_path; the static-contract
        # branch remains covered through their declared shape. Real OpenVINO
        # app detectors are additionally inspected on disk.
        if name == "openvino" and not hasattr(detector, "model_path"):
            summary = getattr(detector, "runtime_summary", None)
            if not isinstance(summary, Mapping) or _summary_shape(summary) != expected_input:
                raise CandidateExportError(f"{name} app detector input contract is invalid")
            if summary.get("output_format") != head:
                raise CandidateExportError(
                    f"{name} app detector configured output format differs from {head!r}"
                )
            summaries[name] = dict(summary)
        else:
            summaries[name] = _validate_detector_runtime_contract(
                name=name,
                detector=detector,
                inference_size=inference_size,
                head=head,
            )

    results: list[dict[str, Any]] = []
    observed_shape: list[int] | None = None
    observed_layout: str | None = None
    for seed in seeds:
        tensor = np.random.default_rng(seed).random(
            (1, 3, *inference_size), dtype=np.float32
        )
        onnx_raw = np.asarray(onnx_detector.infer(tensor))
        # OpenVINO may return a view into a reusable request; retain this output
        # before another call can mutate it.
        openvino_raw = np.array(openvino_detector.infer(tensor), copy=True)
        layout = supported_yolo_output_layout(onnx_raw.shape, 1, head)
        if layout is None:
            raise CandidateExportError(
                f"app detector rejects candidate YOLO output shape {onnx_raw.shape}"
            )
        expected_layout = "end2end" if head == "end2end" else None
        if expected_layout is not None and layout != expected_layout:
            raise CandidateExportError(f"candidate exported {layout}, expected {expected_layout}")
        if head == "traditional" and not layout.startswith("traditional_"):
            raise CandidateExportError(f"candidate exported {layout}, expected a traditional layout")
        if observed_shape is None:
            observed_shape = list(onnx_raw.shape)
            observed_layout = layout
        elif observed_shape != list(onnx_raw.shape) or observed_layout != layout:
            raise CandidateExportError("candidate output contract changed between seeded tensors")
        onnx_values, openvino_values, relevant_rows = _parity_arrays(
            onnx_raw,
            openvino_raw,
            layout=layout,
            confidence_floor=confidence_floor,
        )
        absolute = np.abs(onnx_values - openvino_values)
        if not np.allclose(onnx_values, openvino_values, atol=atol, rtol=rtol):
            raise CandidateExportError(
                "ONNX/OpenVINO numerical parity failed for seed "
                f"{seed}: max_abs={float(absolute.max()):.9g}, atol={atol}, rtol={rtol}"
            )
        results.append(
            {
                "seed": int(seed),
                "input_sha256": sha256(tensor.tobytes(order="C")).hexdigest(),
                "compared_values": int(onnx_values.size),
                "runtime_relevant_rows": relevant_rows,
                "max_abs_error": float(absolute.max(initial=0.0)),
                "mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
            }
        )
    return {
        "status": "passed",
        "seeded_tensor_count": len(results),
        "seeds": [int(value) for value in seeds],
        "atol": float(atol),
        "rtol": float(rtol),
        "confidence_floor": float(confidence_floor),
        "confidence_floor_basis": (
            "Matches the exact runtime-artifact evaluator's minimum retained "
            "prediction confidence. End-to-end scores/classes are compared for all "
            "rows; boxes are compared for rows at or above this floor because unused "
            "TopK coordinates have no detection semantics and may be reordered."
        ),
        "input_shape_nchw": expected_input,
        "output_shape": observed_shape,
        "output_layout": observed_layout,
        "detectors": summaries,
        "tensors": results,
    }


def _artifact_records(directory: Path, basename: str) -> dict[str, dict[str, Any]]:
    expected = {
        "onnx": directory / f"{basename}.onnx",
        "openvino_xml": directory / f"{basename}.xml",
        "openvino_bin": directory / f"{basename}.bin",
        "labels": directory / LABELS_NAME,
        "attribution": directory / ATTRIBUTION_NAME,
        "initial_run_contract": directory / INITIAL_CONTRACT_NAME,
        "training_reproducibility": directory / REPRODUCIBILITY_NAME,
        "training_results": directory / RESULTS_NAME,
    }
    optional = directory / "ultralytics-metadata.yaml"
    if optional.exists():
        expected["ultralytics_metadata"] = optional
    return {
        name: {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(_require_regular_file(path, f"candidate {name}")),
        }
        for name, path in expected.items()
    }


def _attribution_text(
    *,
    checkpoint_sha256: str,
    dataset: Mapping[str, Any],
    inference_size: InferenceSize,
    head: str,
) -> str:
    return f"""# FORT player release-candidate attribution

This staged one-class `player` model was exported from reviewed checkpoint
`{checkpoint_sha256}` at static input `{format_inference_size(inference_size)}`
using the `{head}` output head. It is a candidate only; staging does not approve
it for release or replace any ProAim model.

## Training data

The training data is derived from **FORT-Cuh v1** by Aviles Joseph:

https://universe.roboflow.com/aviles-joseph/fort-cuh-mji4f

The source dataset identifies its license as **CC BY 4.0**:

https://creativecommons.org/licenses/by/4.0/

Exact grouped dataset manifest SHA-256: `{dataset['manifest_sha256']}`

Exact grouped dataset content SHA-256: `{dataset['content_sha256']}`

## Base model and exporter

The model is derived from an Ultralytics YOLO checkpoint and exported with
Ultralytics tooling. Ultralytics code and models identify an AGPL-3.0 license;
ProAim source is AGPL-3.0-or-later. Retain this notice, the candidate manifest,
and applicable source/license material when redistributing artifacts.

## Data and evaluation limitations

- `player` is a visual character class, not enemy/team identity.
- FORT-Cuh labels and class remapping may contain omissions or annotation noise.
- Far and ultra-far buckets contain relatively few reviewed objects; report raw
  detected/total counts and confidence intervals, not only point estimates.
- This dataset's test split was consumed during development and is not an
  untouched release holdout.
- Filename/perceptual grouping reduces observed leakage but cannot prove every
  visually related source was separated.
- Independent target-clone footage and real target hardware qualification are
  still required before release. Seeded numerical parity is not an accuracy or
  latency claim.

This notice records stated inputs and limitations; it is not legal advice.
"""


def _write_new(path: Path, contents: bytes) -> None:
    try:
        with path.open("xb") as target:
            target.write(contents)
    except OSError as exc:
        raise CandidateExportError(f"cannot write candidate member {path}: {exc}") from exc


def _manifest_content_hash(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    body.pop("candidate_content_sha256", None)
    return _canonical_hash(body)


def stage_candidate(
    *,
    weights: Path,
    data: Path,
    output: Path,
    inference_size: InferenceSize = DEFAULT_INFERENCE_SIZE,
    basename: str = DEFAULT_BASENAME,
    head: str = "end2end",
    opset: int | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    parity_confidence_floor: float = DEFAULT_PARITY_CONFIDENCE_FLOOR,
    parity_atol: float = DEFAULT_PARITY_ATOL,
    parity_rtol: float = DEFAULT_PARITY_RTOL,
    expected_manifest_sha256: str = AUDITED_V9_MANIFEST_SHA256,
    expected_content_sha256: str = AUDITED_V9_CONTENT_SHA256,
    yolo_factory: Callable[[str], Any] | None = None,
    detector_factory: Callable[..., tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    output = _require_new_directory(output, "candidate output")
    weights = _require_regular_file(weights, "reviewed checkpoint", ".pt")
    inference_size = validate_yolo_inference_size(inference_size)
    if head not in {"end2end", "traditional"}:
        raise CandidateExportError("head must be end2end or traditional")
    if isinstance(opset, bool) or (opset is not None and not 12 <= opset <= 23):
        raise CandidateExportError("ONNX opset must be an integer from 12 to 23")
    if not math.isfinite(parity_confidence_floor) or not 0.0 <= parity_confidence_floor <= 1.0:
        raise CandidateExportError("parity confidence floor must be from zero to one")
    if (
        not math.isfinite(parity_atol)
        or not math.isfinite(parity_rtol)
        or parity_atol < 0.0
        or parity_rtol < 0.0
    ):
        raise CandidateExportError("parity tolerances must be non-negative")
    for description, digest in (
        ("expected manifest", expected_manifest_sha256),
        ("expected dataset content", expected_content_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise CandidateExportError(f"{description} SHA-256 must be 64 lowercase hex characters")
    dataset = _validate_dataset(
        data,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_content_sha256=expected_content_sha256,
    )
    checkpoint_record = {
        "path": str(weights),
        "bytes": weights.stat().st_size,
        "sha256": _sha256_file(weights),
    }
    training_provenance, provenance_files = _validate_training_provenance(
        weights=weights,
        data=_require_regular_file(data, "grouped dataset YAML", ".yaml"),
        dataset=dataset,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _write_new(work / STAGING_MARKER, b"incomplete\n")
        artifacts, export_args = _export_into_temporary(
            weights,
            work,
            expected_checkpoint_sha256=checkpoint_record["sha256"],
            basename=basename,
            inference_size=inference_size,
            head=head,
            opset=opset,
            yolo_factory=yolo_factory,
        )
        labels = work / LABELS_NAME
        _write_new(labels, LABEL_TEXT.encode("utf-8"))
        for key, target_name in (
            ("initial_run_contract", INITIAL_CONTRACT_NAME),
            ("training_reproducibility", REPRODUCIBILITY_NAME),
            ("training_results", RESULTS_NAME),
        ):
            _copy_regular_file(provenance_files[key], work / target_name)
        parity = validate_runtime_parity(
            artifacts=artifacts,
            labels=labels,
            inference_size=inference_size,
            head=head,
            seeds=seeds,
            confidence_floor=parity_confidence_floor,
            atol=parity_atol,
            rtol=parity_rtol,
            detector_factory=detector_factory,
        )
        attribution = _attribution_text(
            checkpoint_sha256=checkpoint_record["sha256"],
            dataset=dataset,
            inference_size=inference_size,
            head=head,
        )
        _write_new(work / ATTRIBUTION_NAME, attribution.encode("utf-8"))
        artifact_records = _artifact_records(work, basename)
        if (
            weights.stat().st_size != checkpoint_record["bytes"]
            or _sha256_file(weights) != checkpoint_record["sha256"]
        ):
            raise CandidateExportError("reviewed checkpoint changed during candidate export")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "validated_release_candidate_not_approved",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint_record,
            "dataset": dataset,
            "training_provenance": training_provenance,
            "configuration": {
                "basename": basename,
                "inference_size": format_inference_size(inference_size),
                "input_shape_nchw": [1, 3, *inference_size],
                "head": head,
                "one_class": {"0": "player"},
                "export_args": export_args,
            },
            "exporter": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
                "dataset_contract_sha256": _sha256_file(
                    PROJECT_ROOT / "scripts" / "fort_dataset_contract.py"
                ),
                "requirements_export_sha256": _sha256_file(
                    PROJECT_ROOT / "requirements-export.txt"
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _package_records(),
            },
            "artifacts": artifact_records,
            "parity": parity,
            "release_gate": {
                "approved": False,
                "model_accuracy_qualified": False,
                "target_gpu_latency_qualified": False,
                "frozen_build_qualified": False,
                "independent_holdout_qualified": False,
            },
        }
        _validate_packaged_training_provenance(
            work,
            manifest=manifest,
            artifacts=artifact_records,
        )
        manifest["candidate_content_sha256"] = _manifest_content_hash(manifest)
        _write_new(work / MANIFEST_NAME, _canonical_json_bytes(manifest))
        try:
            (work / STAGING_MARKER).unlink()
        except OSError as exc:
            raise CandidateExportError(f"cannot finalize staging marker: {exc}") from exc
        if _path_lexists(output):
            raise CandidateExportError(f"candidate output appeared during export: {output}")
        try:
            work.rename(output)
        except OSError as exc:
            raise CandidateExportError(f"cannot publish completed candidate directory: {exc}") from exc
        return manifest
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


def _validate_packaged_training_provenance(
    output: Path,
    *,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> None:
    """Revalidate the sealed training evidence without needing the source run."""

    provenance = manifest.get("training_provenance")
    checkpoint = manifest.get("checkpoint")
    dataset = manifest.get("dataset")
    if not all(isinstance(value, Mapping) for value in (provenance, checkpoint, dataset)):
        raise CandidateExportError("candidate training provenance is incomplete")
    assert isinstance(provenance, Mapping)
    assert isinstance(checkpoint, Mapping)
    assert isinstance(dataset, Mapping)
    if provenance.get("schema_version") != 1 or provenance.get(
        "checkpoint_role"
    ) != "completed_run_best":
        raise CandidateExportError("candidate training provenance schema/role is invalid")
    packaged_files = provenance.get("packaged_files")
    expected_files = {
        "initial_run_contract": INITIAL_CONTRACT_NAME,
        "training_reproducibility": REPRODUCIBILITY_NAME,
        "training_results": RESULTS_NAME,
    }
    if packaged_files != expected_files:
        raise CandidateExportError("candidate training provenance file map is invalid")
    for key, filename in expected_files.items():
        artifact = artifacts.get(key)
        if not isinstance(artifact, Mapping) or artifact.get("name") != filename:
            raise CandidateExportError(f"candidate training provenance artifact is invalid: {key}")

    contract_path = output / INITIAL_CONTRACT_NAME
    reproducibility_path = output / REPRODUCIBILITY_NAME
    results_path = output / RESULTS_NAME
    contract = _read_json_object(contract_path, "packaged initial training contract")
    reproducibility = _read_json_object(
        reproducibility_path, "packaged training reproducibility record"
    )
    contract_sha256 = _sha256_file(contract_path)
    reproducibility_sha256 = _sha256_file(reproducibility_path)
    results_sha256 = _sha256_file(results_path)
    for actual, recorded, description in (
        (
            contract_sha256,
            provenance.get("initial_run_contract_sha256"),
            "initial run contract",
        ),
        (
            reproducibility_sha256,
            provenance.get("training_reproducibility_sha256"),
            "training reproducibility",
        ),
        (results_sha256, provenance.get("training_results_sha256"), "training results"),
    ):
        if actual != recorded:
            raise CandidateExportError(f"packaged {description} hash differs from its manifest")

    if contract.get("schema_version") != 1 or reproducibility.get("schema_version") != 1:
        raise CandidateExportError("packaged training contract schema is unsupported")
    run_dir = _recorded_path(provenance.get("run_dir"), "packaged run directory")
    if _recorded_path(contract.get("run_dir"), "packaged initial run directory") != run_dir:
        raise CandidateExportError("packaged initial contract identifies a different run")
    dataset_manifest_sha256 = _require_sha256(
        dataset.get("manifest_sha256"), "packaged dataset manifest hash"
    )
    dataset_content_sha256 = _require_sha256(
        dataset.get("content_sha256"), "packaged dataset content hash"
    )
    dataset_yaml_sha256 = _require_sha256(
        dataset.get("yaml_sha256"), "packaged dataset YAML hash"
    )
    initial_weights_sha256 = _require_sha256(
        contract.get("initial_weights_sha256"), "packaged initial weights hash"
    )
    checkpoint_sha256 = _require_sha256(
        checkpoint.get("sha256"), "packaged best checkpoint hash"
    )
    checkpoint_bytes = checkpoint.get("bytes")
    if (
        isinstance(checkpoint_bytes, bool)
        or not isinstance(checkpoint_bytes, int)
        or checkpoint_bytes <= 0
    ):
        raise CandidateExportError("packaged best checkpoint size is invalid")
    source = dataset.get("source")
    if not isinstance(source, Mapping) or source.get("license") != "CC BY 4.0":
        raise CandidateExportError("packaged dataset source/license contract is invalid")
    for contract_key, dataset_key in (
        ("dataset_manifest_sha256", "manifest_sha256"),
        ("dataset_content_sha256", "content_sha256"),
        ("dataset_yaml_sha256", "yaml_sha256"),
    ):
        if contract.get(contract_key) != dataset.get(dataset_key):
            raise CandidateExportError(f"packaged training dataset hash differs for {contract_key}")
    if provenance.get("dataset_manifest_sha256") != dataset.get("manifest_sha256"):
        raise CandidateExportError("packaged training manifest hash differs from dataset evidence")
    if provenance.get("dataset_content_sha256") != dataset.get("content_sha256"):
        raise CandidateExportError("packaged training content hash differs from dataset evidence")
    if provenance.get("initial_weights_sha256") != contract.get("initial_weights_sha256"):
        raise CandidateExportError("packaged initial weights hash differs from its contract")

    script_status = contract.get("training_script_sha256_status")
    adoption = contract.get("adoption")
    if script_status == "captured_at_launch":
        _require_sha256(contract.get("training_script_sha256"), "packaged launch script hash")
        if adoption is not None or provenance.get("adopted_interrupted_run") is not False:
            raise CandidateExportError("packaged ordinary run has inconsistent adoption provenance")
    elif script_status == "unavailable_pre_contract_run":
        if contract.get("training_script_sha256") is not None or not isinstance(adoption, Mapping):
            raise CandidateExportError("packaged adopted run has invalid script provenance")
        adopted_epoch = adoption.get("adopted_checkpoint_epoch")
        if (
            isinstance(adopted_epoch, bool)
            or not isinstance(adopted_epoch, int)
            or adopted_epoch <= 0
        ):
            raise CandidateExportError("packaged adopted checkpoint epoch is invalid")
        adopted_sha = _require_sha256(
            adoption.get("adopted_checkpoint_sha256"), "packaged adopted checkpoint hash"
        )
        if adoption.get("epoch_checkpoint_sha256") != adopted_sha:
            raise CandidateExportError("packaged adopted checkpoint hashes differ")
        _require_sha256(
            adoption.get("adoption_script_sha256"), "packaged adoption script hash"
        )
        if provenance.get("adopted_interrupted_run") is not True:
            raise CandidateExportError("packaged adopted-run flag is inconsistent")
    else:
        raise CandidateExportError("packaged training script provenance status is invalid")
    if provenance.get("script_provenance_status") != script_status:
        raise CandidateExportError("packaged training script provenance summary differs")

    training = contract.get("training")
    initial_arguments = contract.get("training_arguments")
    reproduced_training = reproducibility.get("training")
    reproduced_arguments = reproducibility.get("training_arguments")
    inputs = reproducibility.get("inputs")
    trained_output = reproducibility.get("output")
    initial_environment = contract.get("environment")
    reproduced_environment = reproducibility.get("environment")
    if not all(
        isinstance(value, Mapping)
        for value in (
            training,
            initial_arguments,
            reproduced_training,
            reproduced_arguments,
            inputs,
            trained_output,
            initial_environment,
            reproduced_environment,
        )
    ):
        raise CandidateExportError("packaged training records are incomplete")
    assert isinstance(training, Mapping)
    assert isinstance(initial_arguments, Mapping)
    assert isinstance(reproduced_training, Mapping)
    assert isinstance(reproduced_arguments, Mapping)
    assert isinstance(inputs, Mapping)
    assert isinstance(trained_output, Mapping)
    assert isinstance(initial_environment, Mapping)
    assert isinstance(reproduced_environment, Mapping)
    if reproduced_arguments != initial_arguments:
        raise CandidateExportError("packaged training arguments differ between records")
    for key in (
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
    ):
        if reproduced_training.get(key) != training.get(key):
            raise CandidateExportError(f"packaged training configuration differs for {key}")
    if reproduced_training.get("name") != run_dir.name:
        raise CandidateExportError("packaged training run name differs")
    if reproduced_training.get("adopt_interrupted_run") is not False:
        raise CandidateExportError("packaged record represents adoption, not completed training")
    if _recorded_path(reproduced_training.get("run_dir"), "packaged reproduced run") != run_dir:
        raise CandidateExportError("packaged training run directory differs")
    dataset_yaml_path = _recorded_path(
        dataset.get("yaml_path"), "packaged dataset YAML path"
    )
    initial_weights_path = _recorded_path(
        contract.get("initial_weights"), "packaged initial weights path"
    )
    for actual, expected, description in (
        (contract.get("data"), dataset_yaml_path, "initial dataset"),
        (reproduced_training.get("data"), dataset_yaml_path, "reproduced dataset"),
        (reproduced_training.get("weights"), initial_weights_path, "reproduced weights"),
        (reproduced_training.get("project"), run_dir.parent, "reproduced project"),
    ):
        if _recorded_path(actual, f"packaged {description} path") != expected:
            raise CandidateExportError(f"packaged {description} path differs")
    planned_epochs = training.get("epochs")
    if (
        isinstance(planned_epochs, bool)
        or not isinstance(planned_epochs, int)
        or planned_epochs <= 0
        or training.get("smoke_test") is not False
        or training.get("cache") != "none"
    ):
        raise CandidateExportError("packaged initial training policy is invalid")
    expected_argument_values = {
        "data": str(dataset_yaml_path),
        "epochs": planned_epochs,
        "patience": training.get("patience"),
        "batch": training.get("batch"),
        "imgsz": training.get("imgsz"),
        "device": training.get("device"),
        "workers": training.get("workers"),
        "seed": training.get("seed"),
        "cache": False,
        "deterministic": True,
        "amp": False,
        "rect": False,
        "single_cls": False,
        "save": True,
        "save_period": 1,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
    }
    differing_arguments = sorted(
        key
        for key, expected in expected_argument_values.items()
        if initial_arguments.get(key) != expected
    )
    if differing_arguments:
        raise CandidateExportError(
            "packaged initial training arguments violate the release policy: "
            f"{differing_arguments}"
        )
    for key in ("ultralytics", "torch"):
        version = initial_environment.get(key)
        if not isinstance(version, str) or not version:
            raise CandidateExportError(f"packaged initial environment has no {key} version")
        if reproduced_environment.get(key) != version:
            raise CandidateExportError(f"packaged training environment differs for {key}")
    if inputs.get("initial_run_contract_sha256") != contract_sha256:
        raise CandidateExportError("packaged reproducibility does not seal the initial contract")
    if _recorded_path(
        inputs.get("initial_run_contract"), "packaged recorded initial contract"
    ) != (run_dir / "initial_run_contract.json").resolve():
        raise CandidateExportError("packaged reproducibility points at a different initial contract")
    _require_sha256(inputs.get("training_script_sha256"), "packaged final training script hash")
    for key, expected in (
        ("dataset_manifest_sha256", dataset.get("manifest_sha256")),
        ("dataset_content_sha256", dataset.get("content_sha256")),
        ("dataset_yaml_sha256", dataset.get("yaml_sha256")),
        ("initial_weights_sha256", contract.get("initial_weights_sha256")),
    ):
        if inputs.get(key) != expected:
            raise CandidateExportError(f"packaged reproducibility input differs for {key}")
    completed_summary = provenance.get("completed_epochs")
    if (
        isinstance(completed_summary, bool)
        or not isinstance(completed_summary, int)
        or completed_summary <= 0
    ):
        raise CandidateExportError("packaged completed-epoch summary is invalid")
    resumed_from = inputs.get("resumed_from")
    if isinstance(adoption, Mapping) and resumed_from is None:
        raise CandidateExportError("packaged adopted run has no stateful-resume provenance")
    if resumed_from is not None:
        if not isinstance(resumed_from, Mapping):
            raise CandidateExportError("packaged training resume provenance is invalid")
        _require_sha256(
            resumed_from.get("sha256"), "packaged starting resume checkpoint hash"
        )
        start_epoch = resumed_from.get("completed_epoch")
        if (
            isinstance(start_epoch, bool)
            or not isinstance(start_epoch, int)
            or not 0 < start_epoch < completed_summary
            or resumed_from.get("sha256_scope") != "captured_before_resume_training"
        ):
            raise CandidateExportError("packaged training resume provenance is inconsistent")
        if _recorded_path(
            reproduced_training.get("resume_from"), "packaged training resume checkpoint"
        ) != (run_dir / "weights" / "last.pt").resolve():
            raise CandidateExportError("packaged training resume path differs from its evidence")
        if isinstance(adoption, Mapping) and start_epoch < adoption.get(
            "adopted_checkpoint_epoch", start_epoch
        ):
            raise CandidateExportError("packaged training resume predates adoption")
    elif reproduced_training.get("resume_from") is not None:
        raise CandidateExportError("packaged training configuration claims an unrecorded resume")

    if (
        provenance.get("checkpoint_sha256") != checkpoint_sha256
        or trained_output.get("best_weights_sha256") != checkpoint_sha256
        or trained_output.get("best_weights_bytes") != checkpoint_bytes
    ):
        raise CandidateExportError("packaged best checkpoint evidence is inconsistent")
    if trained_output.get("runtime_class_labels") != ["player"] or trained_output.get(
        "deployment_inference_size"
    ) != training.get("imgsz"):
        raise CandidateExportError("packaged training output contract is inconsistent")
    if _recorded_path(trained_output.get("best_weights"), "packaged best weights") != _recorded_path(
        checkpoint.get("path"), "packaged checkpoint path"
    ):
        raise CandidateExportError("packaged checkpoint path differs from training output")
    completed_epoch, result_rows = _completed_results_epoch(results_path)
    if (
        trained_output.get("results_csv_sha256") != results_sha256
        or trained_output.get("results_csv_bytes") != results_path.stat().st_size
        or trained_output.get("completed_epochs") != completed_epoch
        or provenance.get("completed_epochs") != completed_epoch
        or provenance.get("results_rows") != result_rows
        or provenance.get("planned_epochs") != training.get("epochs")
    ):
        raise CandidateExportError("packaged training results evidence is inconsistent")


def validate_staged_candidate(
    output: Path,
    *,
    seeds: Sequence[int] | None = None,
    detector_factory: Callable[..., tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    output = output.expanduser().absolute()
    if not output.is_dir() or output.is_symlink():
        raise CandidateExportError(f"candidate must be a regular directory: {output}")
    if _path_lexists(output / STAGING_MARKER):
        raise CandidateExportError("candidate retains an incomplete staging marker")
    manifest_path = _require_regular_file(output / MANIFEST_NAME, "candidate manifest", ".json")
    manifest = _read_json_object(manifest_path, "candidate manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CandidateExportError("candidate manifest schema is unsupported")
    if manifest.get("candidate_content_sha256") != _manifest_content_hash(manifest):
        raise CandidateExportError("candidate manifest content hash mismatch")
    if manifest.get("status") != "validated_release_candidate_not_approved":
        raise CandidateExportError("candidate status must remain not approved")
    expected_release_gate = {
        "approved": False,
        "model_accuracy_qualified": False,
        "target_gpu_latency_qualified": False,
        "frozen_build_qualified": False,
        "independent_holdout_qualified": False,
    }
    if manifest.get("release_gate") != expected_release_gate:
        raise CandidateExportError("candidate release gates must remain explicitly unqualified")
    exporter_record = manifest.get("exporter")
    if not isinstance(exporter_record, Mapping):
        raise CandidateExportError("candidate manifest exporter record is missing")
    current_exporter_sha256 = _sha256_file(Path(__file__).resolve())
    if exporter_record.get("sha256") != current_exporter_sha256:
        raise CandidateExportError(
            "candidate was staged by a different exporter revision; restage it before release"
        )
    configuration = manifest.get("configuration")
    artifacts_record = manifest.get("artifacts")
    parity_record = manifest.get("parity")
    training_provenance = manifest.get("training_provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (configuration, artifacts_record, parity_record, training_provenance)
    ):
        raise CandidateExportError("candidate manifest contract is incomplete")
    basename = configuration.get("basename")
    if not isinstance(basename, str):
        raise CandidateExportError("candidate manifest basename is invalid")
    try:
        basename = _basename(basename)
        inference_size = validate_yolo_inference_size(
            configuration.get("input_shape_nchw", [None, None, None, None])[2:4]
        )
    except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
        raise CandidateExportError(f"candidate configuration is invalid: {exc}") from exc
    head = configuration.get("head")
    if head not in {"end2end", "traditional"}:
        raise CandidateExportError("candidate manifest head is invalid")
    expected_names = {
        f"{basename}.onnx",
        f"{basename}.xml",
        f"{basename}.bin",
        LABELS_NAME,
        ATTRIBUTION_NAME,
        INITIAL_CONTRACT_NAME,
        REPRODUCIBILITY_NAME,
        RESULTS_NAME,
        MANIFEST_NAME,
    }
    if "ultralytics_metadata" in artifacts_record:
        expected_names.add("ultralytics-metadata.yaml")
    required_artifact_keys = {
        "onnx",
        "openvino_xml",
        "openvino_bin",
        "labels",
        "attribution",
        "initial_run_contract",
        "training_reproducibility",
        "training_results",
    }
    allowed_artifact_keys = required_artifact_keys | {"ultralytics_metadata"}
    if not required_artifact_keys.issubset(artifacts_record) or not set(
        artifacts_record
    ).issubset(allowed_artifact_keys):
        raise CandidateExportError("candidate artifact records are incomplete or unexpected")
    actual_names: set[str] = set()
    for child in output.iterdir():
        if child.is_symlink() or not child.is_file():
            raise CandidateExportError(f"unexpected or unsafe candidate member: {child}")
        actual_names.add(child.name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise CandidateExportError(
            f"candidate member set mismatch; missing={missing}, extra={extra}"
        )
    expected_artifact_names = {
        "onnx": f"{basename}.onnx",
        "openvino_xml": f"{basename}.xml",
        "openvino_bin": f"{basename}.bin",
        "labels": LABELS_NAME,
        "attribution": ATTRIBUTION_NAME,
        "initial_run_contract": INITIAL_CONTRACT_NAME,
        "training_reproducibility": REPRODUCIBILITY_NAME,
        "training_results": RESULTS_NAME,
        "ultralytics_metadata": "ultralytics-metadata.yaml",
    }
    for name, record in artifacts_record.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("name"), str):
            raise CandidateExportError(f"candidate artifact record is invalid: {name}")
        if record["name"] != expected_artifact_names[name]:
            raise CandidateExportError(f"candidate artifact record has wrong name: {name}")
        path = _require_regular_file(output / record["name"], f"candidate {name}")
        if path.parent != output.resolve():
            raise CandidateExportError(f"candidate artifact escapes its directory: {path}")
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != _sha256_file(path):
            raise CandidateExportError(f"candidate artifact hash/size mismatch: {path.name}")
    if (output / LABELS_NAME).read_text(encoding="utf-8") != LABEL_TEXT:
        raise CandidateExportError("candidate labels must contain exactly 'player'")
    attribution = (output / ATTRIBUTION_NAME).read_text(encoding="utf-8")
    for marker in ("FORT-Cuh", "creativecommons.org/licenses/by/4.0", "AGPL-3.0", "test split"):
        if marker not in attribution:
            raise CandidateExportError(f"candidate attribution is missing {marker!r}")
    _validate_packaged_training_provenance(
        output,
        manifest=manifest,
        artifacts=artifacts_record,
    )
    runtime_seeds = tuple(seeds) if seeds is not None else tuple(parity_record.get("seeds", ()))
    artifacts = {
        "onnx": output / f"{basename}.onnx",
        "openvino_xml": output / f"{basename}.xml",
        "openvino_bin": output / f"{basename}.bin",
    }
    try:
        confidence_floor = float(parity_record.get("confidence_floor"))
        atol = float(parity_record.get("atol"))
        rtol = float(parity_record.get("rtol"))
    except (TypeError, ValueError) as exc:
        raise CandidateExportError("candidate parity tolerances are invalid") from exc
    parity = validate_runtime_parity(
        artifacts=artifacts,
        labels=output / LABELS_NAME,
        inference_size=inference_size,
        head=head,
        seeds=runtime_seeds,
        confidence_floor=confidence_floor,
        atol=atol,
        rtol=rtol,
        detector_factory=detector_factory,
    )
    return {
        "status": "validated",
        "candidate": str(output),
        "candidate_content_sha256": manifest["candidate_content_sha256"],
        "artifacts": artifacts_record,
        "parity": parity,
    }


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.validate_only:
            if args.weights is not None or args.data is not None:
                raise CandidateExportError(
                    "--validate-only accepts only --output and optional --seeds; "
                    "stored candidate settings are authoritative"
                )
            validation_only_overrides = (
                args.inference_size != DEFAULT_INFERENCE_SIZE
                or args.basename != DEFAULT_BASENAME
                or args.head != "end2end"
                or args.opset is not None
                or args.expected_manifest_sha256 != AUDITED_V9_MANIFEST_SHA256
                or args.expected_content_sha256 != AUDITED_V9_CONTENT_SHA256
                or args.parity_confidence_floor != DEFAULT_PARITY_CONFIDENCE_FLOOR
                or args.parity_atol != DEFAULT_PARITY_ATOL
                or args.parity_rtol != DEFAULT_PARITY_RTOL
            )
            if validation_only_overrides:
                raise CandidateExportError(
                    "--validate-only cannot override stored shape, basename, head, "
                    "dataset pins, opset, or parity tolerances"
                )
            result = validate_staged_candidate(args.output, seeds=args.seeds)
        else:
            if args.weights is None or args.data is None:
                raise CandidateExportError("export requires both --weights and --data")
            result = stage_candidate(
                weights=args.weights,
                data=args.data,
                output=args.output,
                inference_size=args.inference_size,
                basename=args.basename,
                head=args.head,
                opset=args.opset,
                seeds=args.seeds or DEFAULT_SEEDS,
                parity_confidence_floor=args.parity_confidence_floor,
                parity_atol=args.parity_atol,
                parity_rtol=args.parity_rtol,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_content_sha256=args.expected_content_sha256,
            )
    except (CandidateExportError, OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"FORT candidate export refused: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
