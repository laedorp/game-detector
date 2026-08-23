"""Strict contract for the bundled model selected as ProAim's default.

The contract is a small, atomically replaceable pointer.  Runtime assets live
at immutable, content-addressed paths, so changing the pointer never requires
overwriting the model currently used by the launcher or a frozen bundle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from utils.public_evidence import contains_nonportable_path


SCHEMA_VERSION = 1
CONTRACT_RELATIVE = PurePosixPath("models/RELEASE-DEFAULT.json")
CONTRACT_STATUS = "default_selected_not_release_qualified"
PRESET_KEY = "fort_player_balanced"
BASE_ARTIFACT_ROLES = frozenset(
    {"onnx", "openvino_xml", "openvino_bin", "labels", "attribution"}
)
TOURNAMENT_COMPARISON_NAMES = (
    "n_end2end_primary_vs_detail",
    "n_traditional_primary_vs_detail",
    "s_end2end_primary_vs_detail",
    "s_traditional_primary_vs_detail",
    "n_end2end_vs_traditional",
    "s_end2end_vs_traditional",
    "n_vs_s",
)
TOURNAMENT_COMPARISON_ROLES = frozenset(
    f"tournament_comparison_{name}" for name in TOURNAMENT_COMPARISON_NAMES
)
TOURNAMENT_SLOT_NAMES = (
    "n_end2end",
    "n_traditional",
    "s_end2end",
    "s_traditional",
)
TOURNAMENT_SEALED_INPUT_ROLES = frozenset(
    {"tournament_plan", "tournament_training_results_n", "tournament_training_results_s"}
    | {f"tournament_runtime_report_{name}" for name in TOURNAMENT_SLOT_NAMES}
)
EVIDENCE_ARTIFACT_ROLES = frozenset(
    {
        "adoption_record",
        "candidate_receipt",
        "training_provenance_receipt",
        "training_results",
        "winner_runtime_receipt",
        "tournament_selection_manifest",
    }
) | TOURNAMENT_COMPARISON_ROLES | TOURNAMENT_SEALED_INPUT_ROLES
QUALIFICATION_RECORD = {
    "approved": False,
    "model_accuracy_qualified": False,
    "target_gpu_latency_qualified": False,
    "frozen_build_qualified": False,
    "independent_holdout_qualified": False,
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PRESET_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
PATH_PART_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
WINDOWS_RESERVED_PARTS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


class ReleaseModelContractError(ValueError):
    """Raised when the release-default pointer is unsafe or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
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


def canonical_hash(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def contract_content_hash(contract: Mapping[str, Any]) -> str:
    body = dict(contract)
    body.pop("content_sha256", None)
    return canonical_hash(body)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseModelContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReleaseModelContractError(f"non-finite JSON constant: {value}")


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ReleaseModelContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseModelContractError(
            f"cannot read release-default contract {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReleaseModelContractError("release-default contract must be a JSON object")
    if payload != canonical_json_bytes(value):
        raise ReleaseModelContractError(
            "release-default contract must use canonical sorted JSON with one trailing newline"
        )
    return value


def _sha256_value(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ReleaseModelContractError(f"{description} must be a lowercase SHA-256")
    return value


def canonical_relative_path(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
    ):
        raise ReleaseModelContractError(
            f"{description} must be a canonical repository-relative POSIX path"
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or not pure.parts
        or pure.parts[0] != "models"
    ):
        raise ReleaseModelContractError(
            f"{description} must remain under models/ as a canonical POSIX path"
        )
    for part in pure.parts:
        if (
            PATH_PART_PATTERN.fullmatch(part) is None
            or part.split(".", 1)[0].casefold() in WINDOWS_RESERVED_PARTS
        ):
            raise ReleaseModelContractError(
                f"{description} contains a non-portable path component: {part!r}"
            )
    return value


def _regular_member(root: Path, relative: str, record: Mapping[str, Any]) -> None:
    pure = PurePosixPath(relative)
    candidate = root.joinpath(*pure.parts)
    current = root
    try:
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise ReleaseModelContractError(
                    f"release-default member uses a symlink: {relative}"
                )
        member_stat = candidate.stat(follow_symlinks=False)
        resolved = candidate.resolve(strict=True)
    except ReleaseModelContractError:
        raise
    except FileNotFoundError as exc:
        raise ReleaseModelContractError(
            f"release-default member is missing: {relative}"
        ) from exc
    except OSError as exc:
        raise ReleaseModelContractError(
            f"cannot inspect release-default member {relative}: {exc}"
        ) from exc
    if not stat.S_ISREG(member_stat.st_mode) or member_stat.st_size <= 0:
        raise ReleaseModelContractError(
            f"release-default member must be a non-empty regular file: {relative}"
        )
    if root != resolved and root not in resolved.parents:
        raise ReleaseModelContractError(
            f"release-default member escapes the project root: {relative}"
        )
    if record.get("bytes") != member_stat.st_size:
        raise ReleaseModelContractError(
            f"release-default member size differs from contract: {relative}"
        )
    digest = sha256()
    try:
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReleaseModelContractError(
            f"cannot hash release-default member {relative}: {exc}"
        ) from exc
    if digest.hexdigest() != record.get("sha256"):
        raise ReleaseModelContractError(
            f"release-default member SHA-256 differs from contract: {relative}"
        )


def _current_public_evidence_sha256() -> str:
    """Hash the exact privacy scanner imported by this semantic validator."""

    source = Path(__file__).resolve().with_name("public_evidence.py")
    try:
        source_stat = source.stat(follow_symlinks=False)
        if source.is_symlink() or not stat.S_ISREG(source_stat.st_mode):
            raise ReleaseModelContractError(
                "public-evidence privacy validator source is not a regular file"
            )
        digest = sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except ReleaseModelContractError:
        raise
    except OSError as exc:
        raise ReleaseModelContractError(
            "cannot hash the public-evidence privacy validator source"
        ) from exc
    return digest.hexdigest()


def _exact_mapping(
    value: object,
    keys: set[str] | frozenset[str],
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ReleaseModelContractError(
            f"{description} fields are incomplete or unexpected"
        )
    return value


def _public_evidence_value(value: Any, description: str) -> None:
    """Reject workstation-local paths from evidence copied into a bundle."""

    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ReleaseModelContractError(
                f"{description} contains a non-finite number"
            )
        return
    if isinstance(value, str):
        slash_value = value.replace("\\", "/")
        folded = slash_value.casefold()
        if (
            "\\" in value
            or contains_nonportable_path(value)
            or "/home/" in folded
            or "/users/" in folded
            or any(ord(character) < 32 for character in value)
        ):
            raise ReleaseModelContractError(
                f"{description} contains a private or absolute path-like string"
            )
        return
    if isinstance(value, Mapping):
        for key, member in value.items():
            if not isinstance(key, str):
                raise ReleaseModelContractError(
                    f"{description} contains an unsafe JSON field name"
                )
            _public_evidence_value(key, f"{description} field name")
            _public_evidence_value(member, f"{description}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, member in enumerate(value):
            _public_evidence_value(member, f"{description}[{index}]")
        return
    raise ReleaseModelContractError(
        f"{description} contains unsupported public evidence data"
    )


def _canonical_evidence_json(
    root: Path,
    record: Mapping[str, Any],
    description: str,
    *,
    require_canonical: bool = True,
) -> dict[str, Any]:
    path = root.joinpath(*PurePosixPath(str(record["path"])).parts)
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ReleaseModelContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseModelContractError(
            f"cannot read {description} as strict JSON: {exc}"
        ) from exc
    if not isinstance(value, dict) or (
        require_canonical and payload != canonical_json_bytes(value)
    ):
        raise ReleaseModelContractError(
            f"{description} must be one"
            + (" canonical" if require_canonical else " strict")
            + " JSON object"
        )
    _public_evidence_value(value, description)
    return value


def _self_hashed_evidence(
    value: Mapping[str, Any],
    field: str,
    description: str,
) -> None:
    expected = _sha256_value(value.get(field), f"{description} self-hash")
    body = dict(value)
    body.pop(field, None)
    if canonical_hash(body) != expected:
        raise ReleaseModelContractError(f"{description} self-hash differs")


def _assert_public_evidence_text(
    root: Path,
    record: Mapping[str, Any],
    description: str,
) -> None:
    path = root.joinpath(*PurePosixPath(str(record["path"])).parts)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReleaseModelContractError(
            f"cannot inspect {description} as UTF-8: {exc}"
        ) from exc
    slash_text = text.replace("\\", "/")
    if (
        "\x00" in text
        or "\\" in text
        or contains_nonportable_path(text)
        or "/home/" in slash_text.casefold()
        or "/users/" in slash_text.casefold()
    ):
        raise ReleaseModelContractError(
            f"{description} contains a private or absolute path"
        )


def _validate_adopted_evidence(
    contract: Mapping[str, Any],
    root: Path,
) -> None:
    """Validate the semantic links in a published development adoption.

    Pointer hashes alone prove byte identity.  These checks additionally prove
    that the receipts, tournament winner, workload, and copied sealed inputs all
    describe the same candidate without turning any release gate true.
    """

    artifacts = contract["artifacts"]
    provenance = contract["provenance"]
    json_roles = {
        "adoption_record",
        "candidate_receipt",
        "training_provenance_receipt",
        "winner_runtime_receipt",
        "tournament_selection_manifest",
        *TOURNAMENT_COMPARISON_ROLES,
        *{
            role
            for role in TOURNAMENT_SEALED_INPUT_ROLES
            if not role.startswith("tournament_training_results_")
        },
    }
    evidence_json = {
        role: _canonical_evidence_json(
            root,
            artifacts[role],
            role,
            require_canonical=role not in TOURNAMENT_SEALED_INPUT_ROLES,
        )
        for role in sorted(json_roles)
    }
    for role in (
        "training_results",
        "tournament_training_results_n",
        "tournament_training_results_s",
    ):
        _assert_public_evidence_text(root, artifacts[role], role)

    adoption = _exact_mapping(
        evidence_json["adoption_record"],
        {
            "schema_version",
            "status",
            "candidate",
            "selection",
            "source",
            "qualification",
            "content_sha256",
        },
        "adoption record",
    )
    if (
        adoption.get("schema_version") != SCHEMA_VERSION
        or adoption.get("status")
        != "development_selected_default_not_release_qualified"
        or adoption.get("qualification") != QUALIFICATION_RECORD
    ):
        raise ReleaseModelContractError(
            "adoption record schema/status/qualification is invalid"
        )
    _self_hashed_evidence(adoption, "content_sha256", "adoption record")
    adoption_source = _exact_mapping(
        adoption.get("source"),
        {
            "adoption_sha256",
            "candidate_exporter_sha256",
            "runtime_comparator_sha256",
            "release_contract_sha256",
            "public_evidence_sha256",
        },
        "adoption source revisions",
    )
    for name, digest in adoption_source.items():
        _sha256_value(digest, f"adoption source revision {name}")
    candidate = _exact_mapping(
        adoption.get("candidate"),
        {
            "candidate_content_sha256",
            "candidate_manifest_sha256",
            "checkpoint_sha256",
            "dataset_manifest_sha256",
            "dataset_content_sha256",
            "input_shape_nchw",
            "output_head",
            "source_artifacts",
        },
        "adopted candidate",
    )
    selection = _exact_mapping(
        adoption.get("selection"),
        {
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
        },
        "adopted selection",
    )
    source_artifacts = _exact_mapping(
        candidate.get("source_artifacts"),
        {
            "onnx",
            "openvino_xml",
            "openvino_bin",
            "labels",
            "attribution",
            "initial_run_contract",
            "training_reproducibility",
            "training_results",
        },
        "adopted candidate source artifacts",
    )
    for role, record in source_artifacts.items():
        _exact_mapping(
            record,
            {"name", "bytes", "sha256"},
            f"adopted candidate source artifact {role}",
        )
        _sha256_value(record.get("sha256"), f"adopted candidate {role} hash")
        if (
            not isinstance(record.get("name"), str)
            or PurePosixPath(str(record["name"])).name != record.get("name")
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record.get("bytes") <= 0
        ):
            raise ReleaseModelContractError(
                f"adopted candidate source artifact {role} is invalid"
            )
    if (
        candidate.get("candidate_content_sha256")
        != provenance.get("candidate_content_sha256")
        or candidate.get("candidate_manifest_sha256")
        != provenance.get("candidate_manifest_sha256")
        or candidate.get("input_shape_nchw") != contract.get("input_shape_nchw")
        or candidate.get("output_head") not in {"end2end", "traditional"}
        or selection.get("tournament_selection_sha256")
        != provenance.get("tournament_selection_sha256")
        or selection.get("tournament_selection_sha256")
        != artifacts["tournament_selection_manifest"]["sha256"]
        or selection.get("selected_backend") != "onnxruntime"
        or selection.get("sealed_tournament_winner") is not True
        or selection.get("release_qualified") is not False
        or selection.get("detail_crop_size_source_pixels")
        != contract.get("detail_crop_size_source_pixels")
    ):
        raise ReleaseModelContractError(
            "release pointer, adopted candidate, and selected workload disagree"
        )
    selected_pipeline = selection.get("selected_pipeline")
    detail_crop = contract.get("detail_crop_size_source_pixels")
    if (
        selected_pipeline not in {"primary", "configured"}
        or (selected_pipeline == "primary" and detail_crop != 0)
        or (selected_pipeline == "configured" and (not isinstance(detail_crop, int) or detail_crop <= 0))
    ):
        raise ReleaseModelContractError(
            "adopted pipeline and pointer detail workload disagree"
        )
    evidence_replay = _exact_mapping(
        selection.get("evidence_replay"),
        {
            "schema_version",
            "status",
            "plan",
            "comparison",
            "winner_training_results",
            "qualification",
        },
        "adoption evidence replay",
    )
    if (
        evidence_replay.get("schema_version") != SCHEMA_VERSION
        or evidence_replay.get("status")
        != (
            "sealed_plan_comparisons_and_winner_training_replayed_"
            "not_release_qualified"
        )
        or evidence_replay.get("qualification") != QUALIFICATION_RECORD
    ):
        raise ReleaseModelContractError(
            "adoption evidence replay schema/status/qualification is invalid"
        )
    replay_plan = _exact_mapping(
        evidence_replay.get("plan"),
        {
            "status",
            "sha256",
            "canonical_bytes",
            "dataset_matches",
            "runtime_matches",
            "all_slot_paths_and_initial_weights_match",
        },
        "adoption plan replay",
    )
    if (
        replay_plan.get("status") != "sealed_tournament_plan_replayed"
        or replay_plan.get("sha256") != artifacts["tournament_plan"]["sha256"]
        or isinstance(replay_plan.get("canonical_bytes"), bool)
        or not isinstance(replay_plan.get("canonical_bytes"), int)
        or replay_plan.get("canonical_bytes") != artifacts["tournament_plan"]["bytes"]
        or replay_plan.get("dataset_matches") is not True
        or replay_plan.get("runtime_matches") is not True
        or replay_plan.get("all_slot_paths_and_initial_weights_match") is not True
    ):
        raise ReleaseModelContractError("adoption sealed-plan replay proof is invalid")
    replay_comparison = _exact_mapping(
        evidence_replay.get("comparison"),
        {
            "comparator_sha256",
            "confidence",
            "bootstrap_samples",
            "records",
            "winner_slot",
            "winner_pipeline",
        },
        "adoption comparison replay",
    )
    _sha256_value(
        replay_comparison.get("comparator_sha256"),
        "adoption replay comparator hash",
    )
    replay_records = _exact_mapping(
        replay_comparison.get("records"),
        set(TOURNAMENT_COMPARISON_NAMES),
        "adoption comparison replay records",
    )
    if (
        replay_comparison.get("confidence") != 0.25
        or replay_comparison.get("bootstrap_samples") != 2000
        or replay_comparison.get("winner_slot") != selection.get("winner_slot")
        or replay_comparison.get("winner_pipeline") != selected_pipeline
    ):
        raise ReleaseModelContractError(
            "adoption comparison replay workload/winner differs"
        )
    for name in TOURNAMENT_COMPARISON_NAMES:
        record = _exact_mapping(
            replay_records[name],
            {
                "baseline_slot",
                "candidate_slot",
                "baseline_pipeline",
                "candidate_pipeline",
                "baseline_report_sha256",
                "candidate_report_sha256",
                "sealed_comparison_sha256",
                "replayed_comparison_sha256",
                "challenger_advanced",
            },
            f"adoption comparison replay {name}",
        )
        for key in (
            "baseline_report_sha256",
            "candidate_report_sha256",
            "sealed_comparison_sha256",
            "replayed_comparison_sha256",
        ):
            _sha256_value(record.get(key), f"adoption replay {name} {key}")
        pointer_comparison_sha = artifacts[f"tournament_comparison_{name}"][
            "sha256"
        ]
        if (
            record.get("sealed_comparison_sha256") != pointer_comparison_sha
            or record.get("replayed_comparison_sha256") != pointer_comparison_sha
            or record.get("baseline_slot") not in TOURNAMENT_SLOT_NAMES
            or record.get("candidate_slot") not in TOURNAMENT_SLOT_NAMES
            or record.get("baseline_pipeline") not in {"primary", "configured"}
            or record.get("candidate_pipeline") not in {"primary", "configured"}
            or not isinstance(record.get("challenger_advanced"), bool)
        ):
            raise ReleaseModelContractError(
                f"adoption comparison replay {name} proof is invalid"
            )
    replay_training = _exact_mapping(
        evidence_replay.get("winner_training_results"),
        {"scale", "bytes", "sha256", "completed_epochs", "results_rows"},
        "adoption winner training-results replay",
    )
    winner_slot_name = selection.get("winner_slot")
    if (
        not isinstance(winner_slot_name, str)
        or replay_training.get("scale") != winner_slot_name.partition("_")[0]
        or replay_training.get("bytes")
        != source_artifacts["training_results"]["bytes"]
        or replay_training.get("sha256")
        != source_artifacts["training_results"]["sha256"]
        or isinstance(replay_training.get("completed_epochs"), bool)
        or not isinstance(replay_training.get("completed_epochs"), int)
        or replay_training.get("completed_epochs") <= 0
        or replay_training.get("results_rows")
        != replay_training.get("completed_epochs")
    ):
        raise ReleaseModelContractError(
            "adoption winner training-results replay proof is invalid"
        )
    for role in (*BASE_ARTIFACT_ROLES, "training_results"):
        source_record = source_artifacts[role]
        pointer_record = artifacts[role]
        if (
            source_record.get("name")
            != PurePosixPath(str(pointer_record["path"])).name
            or source_record.get("bytes") != pointer_record.get("bytes")
            or source_record.get("sha256") != pointer_record.get("sha256")
        ):
            raise ReleaseModelContractError(
                f"adopted {role} bytes differ from candidate provenance"
            )
    expected_model_identity = canonical_hash(
        [
            {
                "name": source_artifacts["onnx"]["name"],
                "sha256": artifacts["onnx"]["sha256"],
            }
        ]
    )
    if selection.get("selected_model_content_sha256") != expected_model_identity:
        raise ReleaseModelContractError(
            "adopted runtime model identity differs from pointer ONNX"
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
    receipts: dict[str, Mapping[str, Any]] = {}
    for role, (fields, status) in receipt_contracts.items():
        receipt = _exact_mapping(evidence_json[role], fields, role)
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("status") != status
            or receipt.get("qualification") != QUALIFICATION_RECORD
        ):
            raise ReleaseModelContractError(
                f"{role} schema/status/qualification is invalid"
            )
        _self_hashed_evidence(receipt, "content_sha256", role)
        receipts[role] = receipt

    candidate_receipt = receipts["candidate_receipt"]
    receipt_configuration = _exact_mapping(
        candidate_receipt.get("configuration"),
        {"basename", "head", "input_shape_nchw", "one_class", "export_args"},
        "candidate receipt configuration",
    )
    receipt_checkpoint = candidate_receipt.get("checkpoint")
    receipt_dataset = candidate_receipt.get("dataset")
    if (
        candidate_receipt.get("original_candidate_manifest_sha256")
        != candidate.get("candidate_manifest_sha256")
        or candidate_receipt.get("candidate_content_sha256")
        != candidate.get("candidate_content_sha256")
        or candidate_receipt.get("artifacts") != source_artifacts
        or receipt_configuration.get("head") != candidate.get("output_head")
        or receipt_configuration.get("input_shape_nchw")
        != contract.get("input_shape_nchw")
        or receipt_configuration.get("one_class") != {"0": "player"}
        or not isinstance(receipt_checkpoint, Mapping)
        or receipt_checkpoint.get("sha256") != candidate.get("checkpoint_sha256")
        or not isinstance(receipt_dataset, Mapping)
        or receipt_dataset.get("manifest_sha256")
        != candidate.get("dataset_manifest_sha256")
        or receipt_dataset.get("content_sha256")
        != candidate.get("dataset_content_sha256")
    ):
        raise ReleaseModelContractError(
            "candidate receipt does not bind the adopted candidate"
        )

    training_receipt = receipts["training_provenance_receipt"]
    training_candidate = training_receipt.get("candidate_manifest")
    training_summary = training_receipt.get("training")
    training_inputs = training_receipt.get("inputs")
    training_output = training_receipt.get("output")
    local_records = training_receipt.get("original_local_records")
    if not all(
        isinstance(item, Mapping)
        for item in (
            training_candidate,
            training_summary,
            training_inputs,
            training_output,
            local_records,
        )
    ):
        raise ReleaseModelContractError(
            "training provenance receipt mappings are incomplete"
        )
    assert isinstance(training_candidate, Mapping)
    assert isinstance(training_summary, Mapping)
    assert isinstance(training_inputs, Mapping)
    assert isinstance(training_output, Mapping)
    assert isinstance(local_records, Mapping)
    training_checkpoint = training_output.get("checkpoint")
    if (
        training_candidate.get("original_sha256")
        != candidate.get("candidate_manifest_sha256")
        or training_candidate.get("candidate_content_sha256")
        != candidate.get("candidate_content_sha256")
        or training_summary.get("completed_epochs")
        != replay_training.get("completed_epochs")
        or training_summary.get("results_rows")
        != replay_training.get("results_rows")
        or training_inputs.get("dataset_manifest_sha256")
        != candidate.get("dataset_manifest_sha256")
        or training_inputs.get("dataset_content_sha256")
        != candidate.get("dataset_content_sha256")
        or not isinstance(training_checkpoint, Mapping)
        or training_checkpoint.get("sha256") != candidate.get("checkpoint_sha256")
        or training_output.get("training_results")
        != source_artifacts["training_results"]
        or local_records.get("initial_run_contract_sha256")
        != source_artifacts["initial_run_contract"]["sha256"]
        or local_records.get("training_reproducibility_sha256")
        != source_artifacts["training_reproducibility"]["sha256"]
        or local_records.get("training_results_sha256")
        != source_artifacts["training_results"]["sha256"]
    ):
        raise ReleaseModelContractError(
            "training provenance receipt does not bind the adopted run"
        )

    winner_receipt = receipts["winner_runtime_receipt"]
    winner_artifact = winner_receipt.get("model_artifact")
    winner_dataset = winner_receipt.get("dataset")
    winner_configuration = winner_receipt.get("configuration")
    if not all(
        isinstance(item, Mapping)
        for item in (winner_artifact, winner_dataset, winner_configuration)
    ):
        raise ReleaseModelContractError("winner runtime receipt mappings are incomplete")
    assert isinstance(winner_artifact, Mapping)
    assert isinstance(winner_dataset, Mapping)
    assert isinstance(winner_configuration, Mapping)
    members = winner_artifact.get("members")
    if (
        winner_receipt.get("original_runtime_evaluation_sha256")
        != selection.get("candidate_evaluation_sha256")
        or winner_receipt.get("candidate_content_sha256")
        != candidate.get("candidate_content_sha256")
        or winner_receipt.get("selected_pipeline") != selected_pipeline
        or winner_receipt.get("detail_crop_size_source_pixels") != detail_crop
        or winner_artifact.get("backend") != "onnxruntime"
        or winner_artifact.get("content_sha256") != expected_model_identity
        or winner_artifact.get("entrypoint_sha256")
        != artifacts["onnx"]["sha256"]
        or not isinstance(members, Sequence)
        or isinstance(members, (str, bytes))
        or len(members) != 1
        or not isinstance(members[0], Mapping)
        or members[0].get("name") != source_artifacts["onnx"]["name"]
        or members[0].get("bytes") != artifacts["onnx"]["bytes"]
        or members[0].get("sha256") != artifacts["onnx"]["sha256"]
        or winner_dataset.get("manifest_sha256")
        != candidate.get("dataset_manifest_sha256")
        or winner_dataset.get("content_sha256")
        != candidate.get("dataset_content_sha256")
        or winner_configuration.get("input_shape_nchw")
        != contract.get("input_shape_nchw")
        or winner_configuration.get("output_format") != candidate.get("output_head")
    ):
        raise ReleaseModelContractError(
            "winner runtime receipt does not bind the adopted artifact/workload"
        )

    tournament = _exact_mapping(
        evidence_json["tournament_selection_manifest"],
        {
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
        },
        "tournament selection",
    )
    if (
        tournament.get("schema_version") != SCHEMA_VERSION
        or tournament.get("status") != "development_model_selection_only"
        or tournament.get("test_data_policy")
        != {
            "test_split_consumed": False,
            "test_reports_accepted": False,
            "selection_split": "val",
        }
    ):
        raise ReleaseModelContractError(
            "tournament selection schema/status/test policy is invalid"
        )
    tournament_comparator = _exact_mapping(
        tournament.get("comparator"),
        {"path", "sha256"},
        "tournament comparator",
    )
    tournament_exporter = _exact_mapping(
        tournament.get("candidate_exporter"),
        {"path", "sha256"},
        "tournament candidate exporter",
    )
    tournament_public_evidence = _exact_mapping(
        tournament.get("public_evidence_privacy"),
        {"path", "sha256"},
        "tournament public-evidence privacy validator",
    )
    if (
        tournament_comparator.get("path")
        != "compare_fort_runtime_evaluations.py"
        or tournament_comparator.get("sha256")
        != replay_comparison.get("comparator_sha256")
        or tournament_comparator.get("sha256")
        != adoption_source.get("runtime_comparator_sha256")
        or tournament_exporter.get("path")
        != "export_fort_release_candidate.py"
        or tournament_exporter.get("sha256")
        != adoption_source.get("candidate_exporter_sha256")
        or tournament_public_evidence.get("path") != "public_evidence.py"
        or tournament_public_evidence.get("sha256")
        != adoption_source.get("public_evidence_sha256")
        or tournament_public_evidence.get("sha256")
        != _current_public_evidence_sha256()
    ):
        raise ReleaseModelContractError(
            "tournament comparator/privacy validator differs from adoption evidence"
        )
    _self_hashed_evidence(
        tournament, "selection_content_sha256", "tournament selection"
    )
    if (
        tournament.get("selection_content_sha256")
        != selection.get("tournament_selection_content_sha256")
    ):
        raise ReleaseModelContractError(
            "adoption and tournament selection content hashes disagree"
        )
    tournament_release = tournament.get("release_qualification")
    if (
        not isinstance(tournament_release, Mapping)
        or tournament_release.get("qualified") is not False
        or tournament_release.get("release_model_approved") is not False
        or tournament_release.get("independent_holdout_required") is not True
        or tournament_release.get("physical_target_gpu_latency_required") is not True
        or tournament_release.get("frozen_build_qualification_required") is not True
    ):
        raise ReleaseModelContractError(
            "tournament selection overstates release qualification"
        )
    fixed = tournament.get("fixed_contract")
    decisions = tournament.get("development_selection")
    candidates = tournament.get("candidates")
    comparisons = tournament.get("comparisons")
    sealed_inputs = tournament.get("sealed_inputs")
    if not all(
        isinstance(item, Mapping)
        for item in (fixed, decisions, candidates, comparisons, sealed_inputs)
    ):
        raise ReleaseModelContractError("tournament selection mappings are incomplete")
    assert isinstance(fixed, Mapping)
    assert isinstance(decisions, Mapping)
    assert isinstance(candidates, Mapping)
    assert isinstance(comparisons, Mapping)
    assert isinstance(sealed_inputs, Mapping)
    winner = decisions.get("winner")
    winner_slot = selection.get("winner_slot")
    winner_candidate = candidates.get(winner_slot)
    tournament_detail = fixed.get("detail_crop_size_source_pixels")
    if (
        fixed.get("input_shape_nchw") != contract.get("input_shape_nchw")
        or isinstance(tournament_detail, bool)
        or not isinstance(tournament_detail, int)
        or tournament_detail <= 0
        or (selected_pipeline == "configured" and tournament_detail != detail_crop)
        or not isinstance(winner, Mapping)
        or winner.get("slot") != winner_slot
        or winner.get("head") != candidate.get("output_head")
        or winner.get("pipeline") != selected_pipeline
        or winner.get("candidate_content_sha256")
        != candidate.get("candidate_content_sha256")
        or winner.get("onnx_sha256") != artifacts["onnx"]["sha256"]
        or winner.get("validation_report_sha256")
        != selection.get("candidate_evaluation_sha256")
        or not isinstance(winner_candidate, Mapping)
        or winner_candidate.get("candidate_manifest_sha256")
        != candidate.get("candidate_manifest_sha256")
        or winner_candidate.get("candidate_content_sha256")
        != candidate.get("candidate_content_sha256")
        or winner_candidate.get("head") != candidate.get("output_head")
        or winner_candidate.get("onnx") != source_artifacts["onnx"]
        or winner_candidate.get("validation_report_sha256")
        != selection.get("candidate_evaluation_sha256")
    ):
        raise ReleaseModelContractError(
            "tournament winner does not bind the adopted candidate/workload"
        )

    if set(comparisons) != set(TOURNAMENT_COMPARISON_NAMES):
        raise ReleaseModelContractError("tournament comparison inventory differs")
    for name in TOURNAMENT_COMPARISON_NAMES:
        record = _exact_mapping(
            comparisons[name],
            {"path", "sha256", "challenger_advanced"},
            f"tournament comparison {name}",
        )
        pointer_record = artifacts[f"tournament_comparison_{name}"]
        replay_record = replay_records[name]
        assert isinstance(replay_record, Mapping)
        replay_baseline = str(replay_record["baseline_slot"])
        replay_candidate = str(replay_record["candidate_slot"])
        if (
            record.get("path") != f"comparisons/{name}/comparison.json"
            or record.get("sha256") != pointer_record["sha256"]
            or not isinstance(record.get("challenger_advanced"), bool)
            or record.get("challenger_advanced")
            is not replay_record.get("challenger_advanced")
            or replay_record.get("baseline_report_sha256")
            != artifacts[f"tournament_runtime_report_{replay_baseline}"]["sha256"]
            or replay_record.get("candidate_report_sha256")
            != artifacts[f"tournament_runtime_report_{replay_candidate}"]["sha256"]
        ):
            raise ReleaseModelContractError(
                f"tournament comparison {name} differs from pointer evidence"
            )

    sealed = _exact_mapping(
        sealed_inputs,
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
        {"n", "s"},
        "tournament sealed training results",
    )
    sealed_records: dict[str, tuple[Mapping[str, Any], str]] = {
        "tournament_plan": (
            _exact_mapping(
                sealed.get("plan"),
                {"path", "bytes", "sha256"},
                "tournament sealed plan",
            ),
            "inputs/tournament-plan.json",
        )
    }
    for name in TOURNAMENT_SLOT_NAMES:
        sealed_records[f"tournament_runtime_report_{name}"] = (
            _exact_mapping(
                runtime_reports[name],
                {"path", "bytes", "sha256"},
                f"tournament sealed runtime report {name}",
            ),
            f"inputs/runtime/{name}/validation-metrics.json",
        )
    for scale in ("n", "s"):
        sealed_records[f"tournament_training_results_{scale}"] = (
            _exact_mapping(
                training_results[scale],
                {"path", "bytes", "sha256"},
                f"tournament sealed training results {scale}",
            ),
            f"inputs/training/{scale}/training-results.csv",
        )
    for role, (record, expected_path) in sealed_records.items():
        pointer_record = artifacts[role]
        if (
            record.get("path") != expected_path
            or record.get("bytes") != pointer_record["bytes"]
            or record.get("sha256") != pointer_record["sha256"]
        ):
            raise ReleaseModelContractError(
                f"{role} differs from tournament sealed-input evidence"
            )
    plan = tournament.get("plan")
    if (
        not isinstance(plan, Mapping)
        or plan.get("path") != "inputs/tournament-plan.json"
        or plan.get("sha256") != artifacts["tournament_plan"]["sha256"]
        or runtime_reports[str(winner_slot)].get("sha256")
        != selection.get("candidate_evaluation_sha256")
    ):
        raise ReleaseModelContractError(
            "tournament plan or winner runtime report cross-link differs"
        )


def validate_release_default_contract(
    value: Mapping[str, Any],
    *,
    project_root: Path | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Return a normalized copy after enforcing the complete pointer schema."""

    required_top_level = {
        "schema_version",
        "status",
        "preset",
        "input_shape_nchw",
        "detail_crop_size_source_pixels",
        "artifacts",
        "provenance",
        "qualification",
        "content_sha256",
    }
    if set(value) != required_top_level:
        raise ReleaseModelContractError(
            "release-default contract fields are incomplete or unexpected"
        )
    if (
        isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != SCHEMA_VERSION
    ):
        raise ReleaseModelContractError("release-default contract schema is unsupported")
    if value.get("status") != CONTRACT_STATUS:
        raise ReleaseModelContractError("release-default contract status is invalid")
    qualification = value.get("qualification")
    if (
        not isinstance(qualification, Mapping)
        or set(qualification) != set(QUALIFICATION_RECORD)
        or any(qualification.get(key) is not False for key in QUALIFICATION_RECORD)
    ):
        raise ReleaseModelContractError(
            "release-default contract must leave every release qualification false"
        )
    expected_content_hash = _sha256_value(
        value.get("content_sha256"), "release-default content hash"
    )
    if expected_content_hash != contract_content_hash(value):
        raise ReleaseModelContractError("release-default contract content hash mismatch")

    preset = value.get("preset")
    if not isinstance(preset, Mapping) or set(preset) != {
        "key",
        "label",
        "description",
    }:
        raise ReleaseModelContractError("release-default preset record is invalid")
    if preset.get("key") != PRESET_KEY or PRESET_PATTERN.fullmatch(
        str(preset.get("key", ""))
    ) is None:
        raise ReleaseModelContractError("release-default preset key is invalid")
    for field in ("label", "description"):
        text = preset.get(field)
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > 240
            or any(ord(character) < 32 for character in text)
        ):
            raise ReleaseModelContractError(
                f"release-default preset {field} is invalid"
            )

    shape = value.get("input_shape_nchw")
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
    ):
        raise ReleaseModelContractError(
            "release-default input shape must be static batch-one NCHW"
        )
    height, width = shape[2], shape[3]
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 32
        or dimension > 4096
        or dimension % 32
        for dimension in (height, width)
    ):
        raise ReleaseModelContractError(
            "release-default height and width must be multiples of 32 from 32 to 4096"
        )
    detail_crop = value.get("detail_crop_size_source_pixels")
    if (
        isinstance(detail_crop, bool)
        or not isinstance(detail_crop, int)
        or detail_crop < 0
        or detail_crop > 16384
    ):
        raise ReleaseModelContractError(
            "release-default detail crop must be zero (disabled) or a positive "
            "source-pixel size no greater than 16384"
        )

    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping) or "kind" not in provenance:
        raise ReleaseModelContractError("release-default provenance record is invalid")
    provenance_kind = provenance.get("kind")
    if provenance_kind not in {
        "existing_release_default_migration",
        "development_selected_candidate",
    }:
        raise ReleaseModelContractError("release-default provenance kind is invalid")
    if provenance_kind == "existing_release_default_migration":
        if set(provenance) != {
            "kind",
            "candidate_content_sha256",
            "candidate_manifest_sha256",
            "tournament_selection_sha256",
        }:
            raise ReleaseModelContractError(
                "migrated release-default provenance record is invalid"
            )
        if any(
            provenance.get(key) is not None
            for key in (
                "candidate_content_sha256",
                "candidate_manifest_sha256",
                "tournament_selection_sha256",
            )
        ):
            raise ReleaseModelContractError(
                "migrated release-default provenance must not invent candidate evidence"
            )
        expected_roles = BASE_ARTIFACT_ROLES
    else:
        if set(provenance) != {
            "kind",
            "candidate_content_sha256",
            "candidate_manifest_sha256",
            "tournament_selection_sha256",
        }:
            raise ReleaseModelContractError(
                "selected release-default tournament provenance record is invalid"
            )
        for key in (
            "candidate_content_sha256",
            "candidate_manifest_sha256",
            "tournament_selection_sha256",
        ):
            _sha256_value(provenance.get(key), f"release-default provenance {key}")
        expected_roles = BASE_ARTIFACT_ROLES | EVIDENCE_ARTIFACT_ROLES

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_roles:
        raise ReleaseModelContractError(
            "release-default artifact roles are incomplete or unexpected"
        )
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    expected_suffixes = {
        "onnx": ".onnx",
        "openvino_xml": ".xml",
        "openvino_bin": ".bin",
        "labels": ".txt",
        "attribution": ".md",
        "adoption_record": ".json",
        "candidate_receipt": ".json",
        "training_provenance_receipt": ".json",
        "training_results": ".csv",
        "winner_runtime_receipt": ".json",
        "tournament_selection_manifest": ".json",
        **{role: ".json" for role in TOURNAMENT_COMPARISON_ROLES},
        **{
            role: (".csv" if role.startswith("tournament_training_results_") else ".json")
            for role in TOURNAMENT_SEALED_INPUT_ROLES
        },
    }
    for role in sorted(artifacts):
        record = artifacts[role]
        if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
            raise ReleaseModelContractError(
                f"release-default {role} artifact record is invalid"
            )
        relative = canonical_relative_path(record.get("path"), f"{role} artifact")
        if relative in seen_paths:
            raise ReleaseModelContractError(
                f"release-default artifacts reuse path {relative!r}"
            )
        seen_paths.add(relative)
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ReleaseModelContractError(
                f"release-default {role} artifact size is invalid"
            )
        digest = _sha256_value(record.get("sha256"), f"{role} artifact hash")
        if PurePosixPath(relative).suffix.casefold() != expected_suffixes[role]:
            raise ReleaseModelContractError(
                f"release-default {role} artifact has the wrong extension"
            )
        normalized_artifacts[role] = {
            "path": relative,
            "bytes": size,
            "sha256": digest,
        }

    xml = PurePosixPath(normalized_artifacts["openvino_xml"]["path"])
    binary = PurePosixPath(normalized_artifacts["openvino_bin"]["path"])
    if xml.parent != binary.parent or xml.stem != binary.stem:
        raise ReleaseModelContractError(
            "release-default OpenVINO XML and BIN must share a directory and stem"
        )
    if provenance_kind == "development_selected_candidate":
        selection_hash = normalized_artifacts["tournament_selection_manifest"][
            "sha256"
        ]
        if provenance.get("tournament_selection_sha256") != selection_hash:
            raise ReleaseModelContractError(
                "release-default tournament selection hash differs from its artifact"
            )

    normalized = {
        "schema_version": SCHEMA_VERSION,
        "status": CONTRACT_STATUS,
        "preset": dict(preset),
        "input_shape_nchw": [1, 3, height, width],
        "detail_crop_size_source_pixels": detail_crop,
        "artifacts": normalized_artifacts,
        "provenance": dict(provenance),
        "qualification": dict(QUALIFICATION_RECORD),
        "content_sha256": expected_content_hash,
    }
    if verify_files:
        if project_root is None:
            raise ReleaseModelContractError(
                "project_root is required when release-default files are verified"
            )
        root = project_root.expanduser().resolve()
        if not root.is_dir():
            raise ReleaseModelContractError(f"project root is not a directory: {root}")
        for role, record in normalized_artifacts.items():
            _regular_member(root, record["path"], record)
        if provenance_kind == "development_selected_candidate":
            _validate_adopted_evidence(normalized, root)
    return normalized


def load_release_default_contract(
    project_root: str | Path,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = root.joinpath(*CONTRACT_RELATIVE.parts)
    if path.is_symlink():
        raise ReleaseModelContractError(
            f"release-default contract must not be a symlink: {path}"
        )
    value = _strict_json(path)
    return validate_release_default_contract(
        value,
        project_root=root,
        verify_files=verify_files,
    )


def make_release_default_contract(
    *,
    label: str,
    description: str,
    input_shape_nchw: Sequence[int],
    detail_crop_size_source_pixels: int,
    artifacts: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct and validate a never-qualified release-default pointer."""

    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": CONTRACT_STATUS,
        "preset": {
            "key": PRESET_KEY,
            "label": label,
            "description": description,
        },
        "input_shape_nchw": list(input_shape_nchw),
        "detail_crop_size_source_pixels": detail_crop_size_source_pixels,
        "artifacts": {key: dict(record) for key, record in artifacts.items()},
        "provenance": dict(provenance),
        "qualification": dict(QUALIFICATION_RECORD),
    }
    value["content_sha256"] = contract_content_hash(value)
    return validate_release_default_contract(value)
