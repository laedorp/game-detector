"""Public, standard-library-only contract for final holdout release evidence.

The sealed evaluator depends on OpenCV, NumPy, and an accelerator runtime.  A
GitHub-hosted publication job must not need those packages (or the private
holdout) merely to revalidate the redacted evidence bundle.  This module is
therefore the shared, repository-pinned boundary between the sealed evaluator
and release publication.  It validates only public/redacted artifacts; an
authenticated protected workflow remains required to attest the private
package comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Callable

from utils.public_evidence import contains_nonportable_path
from utils.release_model_contract import (
    CONTRACT_RELATIVE,
    QUALIFICATION_RECORD,
    TOURNAMENT_COMPARISON_NAMES,
    TOURNAMENT_SEALED_INPUT_ROLES,
    canonical_hash,
    canonical_json_bytes,
    load_release_default_contract,
)


PLAN_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
PLAN_KIND = "proaim-sealed-independent-runtime-evaluation-plan"
PLAN_STATUS = "frozen_before_sealed_member_access"
EVIDENCE_KIND = "proaim-sealed-independent-runtime-evidence"
RECEIPT_KIND = "proaim-independent-holdout-release-gate-receipt"
RECEIPT_NAME = "INDEPENDENT-HOLDOUT-RECEIPT.json"
BUNDLE_MANIFEST_NAME = "ARTIFACT-MANIFEST.json"
BUNDLE_KIND = "proaim-independent-holdout-publication-input-bundle"
BUNDLE_MEMBER_NAMES = {
    "receipt": RECEIPT_NAME,
    "evidence": "metrics.json",
    "evaluation_plan": "evaluation-plan.json",
    "consumption_event": "ledger/consumed.json",
    "retirement_event": "ledger/retired.json",
}
PUBLIC_RELEASE_RECEIPT_NAME = "ProAim-Independent-Holdout-Qualification.json"

RELEASE_POLICY_VERSION = "proaim-independent-holdout-v1"
RELEASE_POLICY_KIND = "proaim-independent-holdout-release-policy"
RELEASE_POLICY_REVIEW_NOTE = (
    "Repository-owned final independent-holdout release thresholds; absolute "
    "quality evidence only, with no comparative incumbent-improvement claim."
)
DECISION_RESULT_SCOPE = (
    "A pass means only that this immutable evidence meets the predeclared "
    "metric rule over the 33-64, 65-96, and >96px gating buckets. The "
    "<=32px ground-truth recall is descriptive and excluded, but every "
    "unmatched prediction/false positive across all size buckets enters "
    "release precision. This is not release approval or hardware/legal "
    "qualification."
)
RELEASE_INVENTORY_MINIMUMS = {
    "target_le_32": 150,
    "target_33_64": 400,
    "target_65_96": 250,
    "target_gt_96": 250,
    "reviewed_negatives": 1_000,
}
GATING_INVENTORY_KEYS = (
    "target_33_64",
    "target_65_96",
    "target_gt_96",
    "reviewed_negatives",
)
MINIMUM_CAPTURE_SESSIONS = 15
MINIMUM_TARGET_BUCKET_CAPTURE_SESSIONS = 15
MINIMUM_REVIEWED_NEGATIVE_CAPTURE_SESSIONS = 15
CANONICAL_RELEASE_DECISION_RULE = {
    "schema_version": 1,
    "kind": "proaim-independent-holdout-frozen-decision-rule",
    "selected_confidence_threshold": 0.25,
    "minimum_far_recall": 0.80,
    "maximum_far_false_positives": 10,
    "minimum_medium_recall": 0.90,
    "minimum_near_recall": 0.95,
    "minimum_aggregate_precision": 0.90,
    "minimum_aggregate_recall": 0.90,
    "maximum_reviewed_negative_false_positives": 0,
    "maximum_runtime_pipeline_p95_ms": 20.0,
    "manual_review_note": RELEASE_POLICY_REVIEW_NOTE,
}
DEFAULT_CONFIDENCE_THRESHOLDS = [0.25, 0.45]
DEFAULT_NMS_IOU = 0.45
DEFAULT_WARMUP = 3
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

RELEASE_ENVIRONMENT_SCHEMA_VERSION = 1
RELEASE_ENVIRONMENT_KIND = (
    "proaim-independent-holdout-windows-directml-runtime-environment"
)
RELEASE_ENVIRONMENT_POLICY_KIND = (
    "proaim-independent-holdout-windows-directml-runtime-policy"
)
RELEASE_ENVIRONMENT_LOCK_PROFILE = "windows-directml-py313"
RELEASE_ENVIRONMENT_PYTHON_VERSION = "3.13.14"
RELEASE_ENVIRONMENT_LOCKS = {
    "bootstrap": {
        "path": "requirements-locks/bootstrap-py313.txt",
        "sha256": "8694e18bfee0c2e8f5e1a8f5d143fade288f4c17496c5123e50259c96255333e",
    },
    "windows_directml": {
        "path": "requirements-locks/windows-directml-py313.txt",
        "sha256": "f80c77b24c51c981458834ba6f05ae5fa11c41ce8a08829df313f831defd654d",
    },
}
RELEASE_ENVIRONMENT_PACKAGES = {
    "numpy": {
        "version": "2.5.2",
        "artifact_sha256": "85aaccb24182c25df891ad0ec333585967e115269d5f1b17f2c9ae005bc96657",
    },
    "onnxruntime-directml": {
        "version": "1.24.4",
        "artifact_sha256": "2f1031cb2281e5b27cca9efe0b9399317c7286e4d226f7a79d4ab79bbd94d19e",
    },
    "opencv-python": {
        "version": "4.14.0.94",
        "artifact_sha256": "ace53616cdffc9643e17e075397b493e607c427d388c7508798ef7ad2ed577cc",
    },
}
RELEASE_ENVIRONMENT_DECLARED_REQUIREMENTS = (
    ("dxcam", "0.3.0", "requirements.txt"),
    ("mss", "10.2.0", "requirements.txt"),
    ("numpy", "2.5.2", "requirements.txt"),
    ("onnxruntime-directml", "1.24.4", "requirements-runtime-directml.txt"),
    ("opencv-python", "4.14.0.94", "requirements.txt"),
    ("openvino", "2026.3.0", "requirements.txt"),
    ("pyinstaller", "6.22.0", "requirements-build.txt"),
    ("pyserial", "3.5", "requirements.txt"),
    ("pyside6-essentials", "6.11.1", "requirements.txt"),
)
HOLDOUT_HARDWARE_IDENTITY_KIND = (
    "proaim-independent-holdout-directml-adapter-invariant"
)
HOLDOUT_HARDWARE_IDENTITY_STATUS = "verified_before_sealed_member_access"
HOLDOUT_HARDWARE_GPU_ROLE = "amd_rx_6950_xt"
HOLDOUT_HARDWARE_PRODUCT_NAME = "AMD Radeon RX 6950 XT"
HOLDOUT_HARDWARE_VENDOR_ID = "0x1002"


class IndependentHoldoutReleaseContractError(ValueError):
    """Raised when a public final-holdout artifact is unsafe or inconsistent."""


def _environment_record_hash(record: Mapping[str, Any]) -> str:
    body = dict(record)
    body.pop("record_content_sha256", None)
    return canonical_hash(body)


def release_environment_policy_record(project_root: Path) -> dict[str, Any]:
    """Return and source-verify the exact sealed Windows DirectML environment."""

    root = project_root.expanduser().resolve()
    for lock in RELEASE_ENVIRONMENT_LOCKS.values():
        if sha256_file(root / lock["path"]) != lock["sha256"]:
            raise IndependentHoldoutReleaseContractError(
                f"release environment lock differs from policy: {lock['path']}"
            )
    body: dict[str, Any] = {
        "schema_version": RELEASE_ENVIRONMENT_SCHEMA_VERSION,
        "kind": RELEASE_ENVIRONMENT_POLICY_KIND,
        "lock_profile": RELEASE_ENVIRONMENT_LOCK_PROFILE,
        "platform": {"system": "Windows", "machine": "amd64"},
        "python": {
            "implementation": "CPython",
            "version": RELEASE_ENVIRONMENT_PYTHON_VERSION,
        },
        "locks": {
            name: dict(value) for name, value in RELEASE_ENVIRONMENT_LOCKS.items()
        },
        "packages": {
            name: dict(value) for name, value in RELEASE_ENVIRONMENT_PACKAGES.items()
        },
        "installation": {
            "fresh_virtual_environment": True,
            "pip_require_hashes": True,
            "pip_force_reinstall": True,
            "pip_no_compile": True,
            "pip_no_build_isolation": True,
            "complete_distribution_set_verified": True,
            "installed_record_payload_hashes_verified": True,
        },
    }
    body["policy_sha256"] = canonical_hash(body)
    return body


def _release_locked_environment(project_root: Path) -> dict[str, dict[str, Any]]:
    locked: dict[str, dict[str, Any]] = {}
    for lock in RELEASE_ENVIRONMENT_LOCKS.values():
        path = project_root / lock["path"]
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.partition("#")[0].strip()
            if not line:
                continue
            fields = line.split()
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", fields[0])
            hashes = {
                value.removeprefix("--hash=sha256:").lower()
                for value in fields[1:]
                if value.startswith("--hash=sha256:")
            }
            if (
                match is None
                or len(hashes) != len(fields) - 1
                or not hashes
                or any(SHA256_RE.fullmatch(value) is None for value in hashes)
            ):
                raise IndependentHoldoutReleaseContractError(
                    f"invalid exact release environment lock at {path.name}:{number}"
                )
            canonical_name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
            if canonical_name in locked:
                raise IndependentHoldoutReleaseContractError(
                    f"duplicate release environment distribution: {canonical_name}"
                )
            locked[canonical_name] = {
                "version": match.group(2),
                "artifact_hashes": hashes,
            }
    if not locked:
        raise IndependentHoldoutReleaseContractError(
            "release environment locks contain no distributions"
        )
    return locked


def _manifest_environment_inventory(
    manifest: Mapping[str, Any], project_root: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    expected_manifest_keys = {
        "application",
        "artifact_hash_contract",
        "distributions",
        "declared_requirements",
        "inputs",
        "lock_profile",
        "pip_reports",
        "python",
        "runtime_variant",
        "schema_version",
        "target",
    }
    python_record = manifest.get("python")
    target = manifest.get("target")
    contract = manifest.get("artifact_hash_contract")
    distributions = manifest.get("distributions")
    inputs = manifest.get("inputs")
    declared_requirements = manifest.get("declared_requirements")
    pip_reports = manifest.get("pip_reports")
    exact_artifact_scope = (
        "repository-pinned pip --require-hashes wheel/sdist SHA-256; installed "
        "METADATA and RECORD content SHA-256; every installed RECORD payload "
        "entry SHA-256 and size"
    )
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != 1
        or manifest.get("application") != "ProAim"
        or manifest.get("lock_profile") != RELEASE_ENVIRONMENT_LOCK_PROFILE
        or manifest.get("runtime_variant") != "directml"
        or not isinstance(python_record, Mapping)
        or set(python_record)
        != {"cache_tag", "executable_sha256", "implementation", "version"}
        or python_record.get("cache_tag") != "cpython-313"
        or python_record.get("implementation") != "CPython"
        or python_record.get("version") != RELEASE_ENVIRONMENT_PYTHON_VERSION
        or not isinstance(target, Mapping)
        or target != {"machine": "amd64", "system": "Windows"}
        or not isinstance(contract, Mapping)
        or contract
        != {"enforced_before_install": True, "scope": exact_artifact_scope}
        or not isinstance(distributions, list)
        or not isinstance(inputs, list)
        or not isinstance(declared_requirements, list)
        or not isinstance(pip_reports, list)
    ):
        raise IndependentHoldoutReleaseContractError(
            "dependency manifest is not the exact Windows DirectML environment"
        )
    _sha(python_record.get("executable_sha256"), "environment Python executable hash")

    expected_inputs = {
        "requirements-locks/bootstrap-py313.txt",
        "requirements-locks/windows-directml-py313.txt",
        "requirements.txt",
        "requirements-build.txt",
        "requirements-runtime-directml.txt",
    }
    actual_inputs: dict[str, str] = {}
    for item in inputs:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise IndependentHoldoutReleaseContractError(
                "dependency manifest input inventory is invalid"
            )
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or path in actual_inputs:
            raise IndependentHoldoutReleaseContractError(
                "dependency manifest input path is invalid or duplicated"
            )
        _sha(digest, f"dependency input {path} hash")
        actual_inputs[path] = str(digest)
    if set(actual_inputs) != expected_inputs or any(
        sha256_file(project_root / path) != digest
        for path, digest in actual_inputs.items()
    ):
        raise IndependentHoldoutReleaseContractError(
            "dependency manifest source inputs differ from the repository"
        )

    expected_declared = [
        {
            "canonical_name": name,
            "declared_extras": [],
            "locked_version": version,
            "sources": [source],
        }
        for name, version, source in RELEASE_ENVIRONMENT_DECLARED_REQUIREMENTS
    ]
    if declared_requirements != expected_declared:
        raise IndependentHoldoutReleaseContractError(
            "dependency manifest declared requirements differ from release policy"
        )
    expected_report_names = (
        "pip-bootstrap-windows-directml-py313.json",
        "pip-dependencies-windows-directml-py313.json",
    )
    if len(pip_reports) != 2:
        raise IndependentHoldoutReleaseContractError(
            "dependency manifest must bind bootstrap and final pip reports"
        )
    for index, report in enumerate(pip_reports):
        expected_purpose = "bootstrap" if index == 0 else "final-environment-install"
        if (
            not isinstance(report, Mapping)
            or set(report) != {"filename", "pip_version", "purpose", "sha256"}
            or report.get("filename") != expected_report_names[index]
            or report.get("pip_version") != "26.2.1"
            or report.get("purpose") != expected_purpose
        ):
            raise IndependentHoldoutReleaseContractError(
                "dependency manifest pip report inventory is invalid"
            )
        _sha(report.get("sha256"), f"{expected_purpose} pip report hash")

    inventory: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    locked = _release_locked_environment(project_root)
    for item in distributions:
        installed = item.get("installed_files") if isinstance(item, Mapping) else None
        artifact = item.get("artifact") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "artifact",
                "canonical_name",
                "installed_files",
                "installed_metadata_sha256",
                "installed_record_sha256",
                "locked_extras",
                "name",
                "version",
            }
            or not isinstance(installed, Mapping)
            or set(installed)
            != {
                "aggregate_sha256",
                "record_document_sha256",
                "record_entry_count",
                "record_sha256_entries_verified",
                "total_size_bytes",
                "unhashed_record_entries",
            }
            or not isinstance(artifact, Mapping)
            or set(artifact) != {"filename", "sha256", "source"}
            or artifact.get("source") != "files.pythonhosted.org"
        ):
            raise IndependentHoldoutReleaseContractError(
                "dependency manifest distribution record is invalid"
            )
        canonical_name = item.get("canonical_name")
        if not isinstance(canonical_name, str) or canonical_name in seen:
            raise IndependentHoldoutReleaseContractError(
                "dependency manifest distribution names are invalid or duplicated"
            )
        seen.add(canonical_name)
        expected_distribution = locked.get(canonical_name)
        if (
            expected_distribution is None
            or item.get("version") != expected_distribution["version"]
            or artifact.get("sha256")
            not in expected_distribution["artifact_hashes"]
            or item.get("locked_extras") != []
        ):
            raise IndependentHoldoutReleaseContractError(
                f"dependency manifest differs from the exact lock for {canonical_name}"
            )
        for field, value in (
            ("artifact", artifact.get("sha256")),
            ("metadata", item.get("installed_metadata_sha256")),
            ("record", item.get("installed_record_sha256")),
            ("aggregate", installed.get("aggregate_sha256")),
        ):
            _sha(value, f"{canonical_name} installed {field} hash")
        if (
            _integer(installed.get("record_entry_count"), "installed record entries", 1)
            <= 0
            or _integer(
                installed.get("record_sha256_entries_verified"),
                "verified installed record entries",
                1,
            )
            <= 0
            or _integer(installed.get("total_size_bytes"), "installed package bytes", 1)
            <= 0
            or installed.get("unhashed_record_entries") != 1
            or installed.get("record_document_sha256")
            != item.get("installed_record_sha256")
        ):
            raise IndependentHoldoutReleaseContractError(
                "dependency manifest did not verify complete installed RECORD payloads"
            )
        concise = {
            "canonical_name": canonical_name,
            "version": item.get("version"),
            "artifact_sha256": artifact.get("sha256"),
            "installed_metadata_sha256": item.get("installed_metadata_sha256"),
            "installed_record_sha256": item.get("installed_record_sha256"),
            "installed_files_aggregate_sha256": installed.get("aggregate_sha256"),
        }
        inventory.append(concise)
        if canonical_name in RELEASE_ENVIRONMENT_PACKAGES:
            selected[canonical_name] = {key: value for key, value in concise.items() if key != "canonical_name"}
    inventory.sort(key=lambda item: item["canonical_name"])
    if seen != set(locked):
        raise IndependentHoldoutReleaseContractError(
            "dependency manifest installed distribution set differs from exact locks"
        )
    if set(selected) != set(RELEASE_ENVIRONMENT_PACKAGES):
        raise IndependentHoldoutReleaseContractError(
            "dependency manifest omits a release-critical runtime package"
        )
    for name, expected in RELEASE_ENVIRONMENT_PACKAGES.items():
        if (
            selected[name]["version"] != expected["version"]
            or selected[name]["artifact_sha256"] != expected["artifact_sha256"]
        ):
            raise IndependentHoldoutReleaseContractError(
                f"dependency manifest differs from pinned {name}"
            )
    return inventory, selected


def release_environment_record(
    dependency_manifest: Mapping[str, Any],
    *,
    dependency_manifest_sha256: str,
    project_root: Path,
) -> dict[str, Any]:
    """Reduce a verified dependency manifest to a portable exact runtime record."""

    root = project_root.expanduser().resolve()
    manifest_sha256 = _sha(
        dependency_manifest_sha256, "dependency manifest byte hash"
    )
    inventory, selected = _manifest_environment_inventory(dependency_manifest, root)
    python_record = dependency_manifest["python"]
    body: dict[str, Any] = {
        "schema_version": RELEASE_ENVIRONMENT_SCHEMA_VERSION,
        "kind": RELEASE_ENVIRONMENT_KIND,
        "policy": release_environment_policy_record(root),
        "dependency_manifest": {
            "sha256": manifest_sha256,
            "content_sha256": canonical_hash(dependency_manifest),
            "distribution_count": len(inventory),
            "distribution_inventory_sha256": canonical_hash(inventory),
        },
        "python_executable_sha256": python_record["executable_sha256"],
        "packages": selected,
    }
    body["record_content_sha256"] = _environment_record_hash(body)
    return body


def validate_release_environment_record(
    record: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    """Validate the portable exact runtime record without importing runtime deps."""

    expected_keys = {
        "schema_version",
        "kind",
        "policy",
        "dependency_manifest",
        "python_executable_sha256",
        "packages",
        "record_content_sha256",
    }
    manifest = record.get("dependency_manifest")
    packages = record.get("packages")
    if (
        set(record) != expected_keys
        or record.get("schema_version") != RELEASE_ENVIRONMENT_SCHEMA_VERSION
        or record.get("kind") != RELEASE_ENVIRONMENT_KIND
        or record.get("policy") != release_environment_policy_record(project_root)
        or record.get("record_content_sha256") != _environment_record_hash(record)
        or not isinstance(manifest, Mapping)
        or set(manifest)
        != {
            "sha256",
            "content_sha256",
            "distribution_count",
            "distribution_inventory_sha256",
        }
        or _integer(manifest.get("distribution_count"), "environment distributions", 1)
        <= 0
        or not isinstance(packages, Mapping)
        or set(packages) != set(RELEASE_ENVIRONMENT_PACKAGES)
    ):
        raise IndependentHoldoutReleaseContractError(
            "portable holdout runtime environment record is invalid"
        )
    _sha(record.get("python_executable_sha256"), "environment Python executable hash")
    _sha(manifest.get("sha256"), "dependency manifest byte hash")
    _sha(manifest.get("content_sha256"), "dependency manifest content hash")
    _sha(
        manifest.get("distribution_inventory_sha256"),
        "dependency distribution inventory hash",
    )
    expected_package_keys = {
        "version",
        "artifact_sha256",
        "installed_metadata_sha256",
        "installed_record_sha256",
        "installed_files_aggregate_sha256",
    }
    for name, expected in RELEASE_ENVIRONMENT_PACKAGES.items():
        package = packages.get(name)
        if (
            not isinstance(package, Mapping)
            or set(package) != expected_package_keys
            or package.get("version") != expected["version"]
            or package.get("artifact_sha256") != expected["artifact_sha256"]
        ):
            raise IndependentHoldoutReleaseContractError(
                f"portable runtime record differs from pinned {name}"
            )
        for field in expected_package_keys - {"version"}:
            _sha(package.get(field), f"portable {name} {field}")
    return dict(record)


def validate_holdout_hardware_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the redacted exact RX 6950 XT pre-access invariant record."""

    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "gpu_role",
        "product_name",
        "directml_device",
        "directml_adapter_index",
        "vendor_id",
        "device_id",
        "driver_version",
        "physical_evidence",
        "inventory",
        "cross_checks",
        "privacy",
        "content_sha256",
    }
    index = record.get("directml_adapter_index")
    physical = record.get("physical_evidence")
    inventory = record.get("inventory")
    expected_checks = {
        "selected_index_matches_physical_qualification": True,
        "product_name_matches_across_sources": True,
        "vendor_device_matches_across_sources": True,
        "driver_version_matches_physical_qualification": True,
        "dml_execution_provider_available": True,
    }
    expected_privacy = {
        "redacted": True,
        "adapter_luid_disclosed": False,
        "pnp_device_id_disclosed": False,
        "local_paths_disclosed": False,
    }
    if (
        set(record) != expected_keys
        or record.get("schema_version") != 1
        or record.get("kind") != HOLDOUT_HARDWARE_IDENTITY_KIND
        or record.get("status") != HOLDOUT_HARDWARE_IDENTITY_STATUS
        or record.get("gpu_role") != HOLDOUT_HARDWARE_GPU_ROLE
        or record.get("product_name") != HOLDOUT_HARDWARE_PRODUCT_NAME
        or record.get("vendor_id") != HOLDOUT_HARDWARE_VENDOR_ID
        or isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or record.get("directml_device") != f"DML:{index}"
        or re.fullmatch(r"0x[0-9a-f]{4,8}", str(record.get("device_id") or ""))
        is None
        or not isinstance(record.get("driver_version"), str)
        or not str(record["driver_version"]).strip()
        or any(character in str(record["driver_version"]) for character in "\r\n\0")
        or not isinstance(physical, Mapping)
        or set(physical)
        != {"qualification_run_id", "adapter_index", "public_receipt_sha256"}
        or isinstance(physical.get("qualification_run_id"), bool)
        or not isinstance(physical.get("qualification_run_id"), int)
        or physical.get("qualification_run_id") <= 0
        or physical.get("adapter_index") != index
        or not isinstance(inventory, Mapping)
        or set(inventory)
        != {
            "dxgi_exact_match_count",
            "wmi_exact_match_count",
            "directx_registry_exact_match_count",
            "dedicated_vram_bytes",
            "adapter_luid_present_and_correlated",
        }
        or inventory.get("dxgi_exact_match_count") != 1
        or inventory.get("wmi_exact_match_count") != 1
        or inventory.get("directx_registry_exact_match_count") != 1
        or _integer(inventory.get("dedicated_vram_bytes"), "RX 6950 XT VRAM", 1)
        <= 0
        or inventory.get("adapter_luid_present_and_correlated") is not True
        or record.get("cross_checks") != expected_checks
        or record.get("privacy") != expected_privacy
        or record.get("content_sha256")
        != canonical_hash(
            {key: value for key, value in record.items() if key != "content_sha256"}
        )
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout hardware identity is not the exact RX 6950 XT invariant"
        )
    _sha(physical.get("public_receipt_sha256"), "RX 6950 XT receipt hash")
    reject_private_path_strings(record, "holdout hardware identity")
    return dict(record)


def release_policy_record() -> dict[str, Any]:
    """Return the one canonical repository-owned final-holdout policy."""

    body: dict[str, Any] = {
        "schema_version": 1,
        "kind": RELEASE_POLICY_KIND,
        "policy_version": RELEASE_POLICY_VERSION,
        "decision_rule": dict(CANONICAL_RELEASE_DECISION_RULE),
        "inventory_minimums": dict(RELEASE_INVENTORY_MINIMUMS),
        "source_group_minimums": {
            "overall_capture_sessions": MINIMUM_CAPTURE_SESSIONS,
            "target_bearing_capture_sessions_per_gating_bucket": (
                MINIMUM_TARGET_BUCKET_CAPTURE_SESSIONS
            ),
            "reviewed_negative_capture_sessions": (
                MINIMUM_REVIEWED_NEGATIVE_CAPTURE_SESSIONS
            ),
        },
        "claim_scope": "absolute_threshold_evidence_only_no_incumbent_comparison",
    }
    body["policy_sha256"] = canonical_hash(body)
    return body


def sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        details = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(details.st_mode):
            raise IndependentHoldoutReleaseContractError(
                f"expected a regular file: {path}"
            )
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except IndependentHoldoutReleaseContractError:
        raise
    except OSError as exc:
        raise IndependentHoldoutReleaseContractError(
            f"cannot hash required source file {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def source_snapshot(project_root: Path) -> dict[str, Any]:
    """Reconstruct the exact evaluator/application source snapshot with stdlib."""

    root = project_root.expanduser().resolve()
    sources = {
        "evaluator": root / "scripts" / "evaluate_independent_holdout_runtime.py",
        "development_runtime_evaluator": root
        / "scripts"
        / "evaluate_fort_runtime_model.py",
        "holdout_contract": root
        / "scripts"
        / "prepare_independent_player_holdout.py",
        "candidate_exporter": root / "scripts" / "export_fort_release_candidate.py",
        "candidate_adoption": root / "scripts" / "adopt_fort_release_candidate.py",
        "candidate_tournament": root / "scripts" / "run_fort_model_tournament.py",
        "runtime_comparator": root
        / "scripts"
        / "compare_fort_runtime_evaluations.py",
        "release_default_contract": root / "utils" / "release_model_contract.py",
        "public_evidence_privacy": root / "utils" / "public_evidence.py",
        "independent_holdout_release_contract": Path(__file__).resolve(),
        "holdout_adapter_invariant": root
        / "scripts"
        / "verify_windows_holdout_adapter.py",
        "holdout_hardware_discovery": root / "detection" / "hardware.py",
    }
    pipeline_paths = {
        "runtime_orchestration": root / "main.py",
        "preprocess": root / "utils" / "preprocess.py",
        "inference_size": root / "utils" / "inference_size.py",
        "detail_pass": root / "detection" / "detail_pass.py",
        "postprocess": root / "detection" / "postprocess.py",
        "detection_types": root / "detection" / "types.py",
        "detection_base": root / "detection" / "base.py",
        "detector": root / "detection" / "onnx_yolo.py",
        "onnx_shared_openvino_helpers": root / "detection" / "openvino_yolo.py",
        "metric_helpers": root / "scripts" / "evaluate_fort_model.py",
    }
    return {
        "files": {
            name: {"name": path.name, "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "application_pipeline": {
            name: sha256_file(path) for name, path in pipeline_paths.items()
        },
    }


def receipt_verifier_record(project_root: Path) -> dict[str, Any]:
    """Return the exact public verifier record embedded in a portable receipt."""

    snapshot = source_snapshot(project_root)
    files = snapshot["files"]
    return {
        "schema_version": 1,
        "evaluator": dict(files["evaluator"]),
        "holdout_contract": dict(files["holdout_contract"]),
        "public_evidence_privacy": dict(files["public_evidence_privacy"]),
        "independent_holdout_release_contract": dict(
            files["independent_holdout_release_contract"]
        ),
        "holdout_adapter_invariant": dict(files["holdout_adapter_invariant"]),
        "holdout_hardware_discovery": dict(files["holdout_hardware_discovery"]),
        "source_snapshot_sha256": canonical_hash(snapshot),
        "application_pipeline_sha256": canonical_hash(
            snapshot["application_pipeline"]
        ),
        "release_policy_sha256": release_policy_record()["policy_sha256"],
    }


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IndependentHoldoutReleaseContractError(
                f"duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise IndependentHoldoutReleaseContractError(
        f"non-finite JSON constant: {value}"
    )


def strict_json_file(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except IndependentHoldoutReleaseContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentHoldoutReleaseContractError(
            f"cannot read {description} as strict JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise IndependentHoldoutReleaseContractError(
            f"{description} must be a JSON object"
        )
    reject_private_path_strings(value, description)
    return value, payload


def reject_private_path_strings(value: object, description: str) -> None:
    """Reject local paths/control text recursively in both keys and values."""

    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise IndependentHoldoutReleaseContractError(
                f"{description} contains a non-finite number"
            )
        return
    if isinstance(value, str):
        if (
            not value.strip()
            or any(ord(character) < 32 for character in value)
            or contains_nonportable_path(value)
        ):
            raise IndependentHoldoutReleaseContractError(
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
                raise IndependentHoldoutReleaseContractError(
                    f"{description} contains an unsafe field name"
                )
            reject_private_path_strings(member, f"{description}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, member in enumerate(value):
            reject_private_path_strings(member, f"{description}[{index}]")
        return
    raise IndependentHoldoutReleaseContractError(
        f"{description} contains unsupported public data"
    )


def _sha(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise IndependentHoldoutReleaseContractError(
            f"{description} must be a lowercase SHA-256"
        )
    return value


def _integer(value: object, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IndependentHoldoutReleaseContractError(
            f"{description} must be an integer >= {minimum}"
        )
    return value


def _probability(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndependentHoldoutReleaseContractError(
            f"{description} must be a probability"
        )
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise IndependentHoldoutReleaseContractError(
            f"{description} must be a finite probability"
        )
    return normalized


def _positive(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndependentHoldoutReleaseContractError(
            f"{description} must be a positive number"
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise IndependentHoldoutReleaseContractError(
            f"{description} must be a positive finite number"
        )
    return normalized


def _json_from_pointer(root: Path, record: Mapping[str, Any], description: str) -> dict[str, Any]:
    relative = record.get("path")
    if not isinstance(relative, str):
        raise IndependentHoldoutReleaseContractError(
            f"{description} pointer path is missing"
        )
    path = root.joinpath(*PurePosixPath(relative).parts)
    value, payload = strict_json_file(path, description)
    if (
        len(payload) != record.get("bytes")
        or sha256(payload).hexdigest() != record.get("sha256")
    ):
        raise IndependentHoldoutReleaseContractError(
            f"{description} bytes differ from the release-default pointer"
        )
    return value


def public_candidate_binding(project_root: Path) -> dict[str, Any]:
    """Rebuild the evaluator's public adopted-candidate binding without ML deps."""

    root = project_root.expanduser().resolve()
    try:
        pointer = load_release_default_contract(root, verify_files=True)
    except Exception as exc:
        raise IndependentHoldoutReleaseContractError(
            f"release-default contract is invalid: {exc}"
        ) from exc
    provenance = pointer.get("provenance")
    artifacts = pointer.get("artifacts")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("kind") != "development_selected_candidate"
        or not isinstance(artifacts, Mapping)
    ):
        raise IndependentHoldoutReleaseContractError(
            "release default is not a development-selected candidate"
        )
    adoption_record = artifacts.get("adoption_record")
    selection_record = artifacts.get("tournament_selection_manifest")
    if not isinstance(adoption_record, Mapping) or not isinstance(
        selection_record, Mapping
    ):
        raise IndependentHoldoutReleaseContractError(
            "release default omits adoption/tournament evidence"
        )
    adoption = _json_from_pointer(root, adoption_record, "candidate adoption")
    tournament = _json_from_pointer(root, selection_record, "tournament selection")
    candidate = adoption.get("candidate")
    selection = adoption.get("selection")
    source = adoption.get("source")
    if not all(isinstance(item, Mapping) for item in (candidate, selection, source)):
        raise IndependentHoldoutReleaseContractError(
            "candidate adoption public binding is incomplete"
        )
    assert isinstance(candidate, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(source, Mapping)
    source_artifacts = candidate.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise IndependentHoldoutReleaseContractError(
            "candidate adoption source artifacts are incomplete"
        )
    onnx_pointer = artifacts.get("onnx")
    onnx_source = source_artifacts.get("onnx")
    labels_pointer = artifacts.get("labels")
    labels_source = source_artifacts.get("labels")
    if not all(
        isinstance(item, Mapping)
        for item in (onnx_pointer, onnx_source, labels_pointer, labels_source)
    ):
        raise IndependentHoldoutReleaseContractError(
            "candidate runtime model/labels bindings are incomplete"
        )
    assert isinstance(onnx_pointer, Mapping)
    assert isinstance(onnx_source, Mapping)
    assert isinstance(labels_pointer, Mapping)
    assert isinstance(labels_source, Mapping)
    model_record = {
        "role": "onnx",
        "name": onnx_source.get("name"),
        "relative_path": onnx_pointer.get("path"),
        "bytes": onnx_pointer.get("bytes"),
        "sha256": onnx_pointer.get("sha256"),
    }
    model_content = canonical_hash(
        [{"name": model_record["name"], "sha256": model_record["sha256"]}]
    )
    tournament_roles = [
        "tournament_selection_manifest",
        *(f"tournament_comparison_{name}" for name in TOURNAMENT_COMPARISON_NAMES),
        *sorted(TOURNAMENT_SEALED_INPUT_ROLES),
    ]
    tournament_evidence: list[dict[str, Any]] = []
    for role in tournament_roles:
        record = artifacts.get(role)
        if not isinstance(record, Mapping):
            raise IndependentHoldoutReleaseContractError(
                f"release pointer omits tournament evidence role {role}"
            )
        tournament_evidence.append(
            {
                "role": role,
                "relative_path": record.get("path"),
                "bytes": record.get("bytes"),
                "sha256": record.get("sha256"),
            }
        )
    provenance_roles = (
        "candidate_receipt",
        "training_provenance_receipt",
        "training_results",
        "winner_runtime_receipt",
    )
    candidate_provenance: list[dict[str, Any]] = []
    for role in provenance_roles:
        record = artifacts.get(role)
        if not isinstance(record, Mapping):
            raise IndependentHoldoutReleaseContractError(
                f"release pointer omits candidate provenance role {role}"
            )
        candidate_provenance.append(
            {
                "role": role,
                "relative_path": record.get("path"),
                "bytes": record.get("bytes"),
                "sha256": record.get("sha256"),
            }
        )
    evidence_replay = selection.get("evidence_replay")
    if not isinstance(evidence_replay, Mapping):
        raise IndependentHoldoutReleaseContractError(
            "candidate adoption omits its semantic replay proof"
        )
    pointer_path = root.joinpath(*CONTRACT_RELATIVE.parts)
    return {
        "pointer_sha256": sha256_file(pointer_path),
        "pointer_content_sha256": pointer["content_sha256"],
        "input_shape_nchw": list(pointer["input_shape_nchw"]),
        "candidate_content_sha256": candidate.get("candidate_content_sha256"),
        "candidate_manifest_sha256": candidate.get("candidate_manifest_sha256"),
        "checkpoint_sha256": candidate.get("checkpoint_sha256"),
        "dataset_manifest_sha256": candidate.get("dataset_manifest_sha256"),
        "dataset_content_sha256": candidate.get("dataset_content_sha256"),
        "adoption_sha256": adoption_record.get("sha256"),
        "adoption_content_sha256": adoption.get("content_sha256"),
        "adoption_evidence_replay_sha256": canonical_hash(evidence_replay),
        "tournament_selection_sha256": selection_record.get("sha256"),
        "tournament_selection_content_sha256": tournament.get(
            "selection_content_sha256"
        ),
        "tournament_evidence": tournament_evidence,
        "candidate_provenance_evidence": candidate_provenance,
        "candidate_evaluation_sha256": selection.get(
            "candidate_evaluation_sha256"
        ),
        "winner_slot": selection.get("winner_slot"),
        "model_artifacts": [model_record],
        "model_content_sha256": model_content,
        "labels": {
            "relative_path": labels_pointer.get("path"),
            "bytes": labels_pointer.get("bytes"),
            "sha256": labels_pointer.get("sha256"),
        },
        "selected_pipeline": selection.get("selected_pipeline"),
        "selected_backend": selection.get("selected_backend"),
        "output_head": candidate.get("output_head"),
        "detail_crop_size_source_pixels": pointer.get(
            "detail_crop_size_source_pixels"
        ),
        "exporter_sha256": source.get("candidate_exporter_sha256"),
        "adoption_source_sha256": source.get("adoption_sha256"),
    }


def _receipt_candidate(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
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
        "tournament_selection_sha256": binding["tournament_selection_sha256"],
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
    }


def _receipt_content_hash(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    body.pop("receipt_content_sha256", None)
    return canonical_hash(body)


def _plan_content_hash(plan: Mapping[str, Any]) -> str:
    body = dict(plan)
    body.pop("plan_content_sha256", None)
    return canonical_hash(body)


def _evidence_content_hash(evidence: Mapping[str, Any]) -> str:
    body = dict(evidence)
    body.pop("evidence_content_sha256", None)
    return canonical_hash(body)


def _bundle_content_hash(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    body.pop("bundle_content_sha256", None)
    return canonical_hash(body)


def _fraction(value: object, description: str) -> tuple[int, int]:
    if not isinstance(value, str) or re.fullmatch(r"\d+/[1-9]\d*", value) is None:
        raise IndependentHoldoutReleaseContractError(
            f"{description} must be an exact detected/total count"
        )
    detected_text, total_text = value.split("/", 1)
    detected, total = int(detected_text), int(total_text)
    if detected > total:
        raise IndependentHoldoutReleaseContractError(
            f"{description} detected count exceeds total"
        )
    return detected, total


def _validate_decision(
    decision: Mapping[str, Any], *, holdout_counts: Mapping[str, Any]
) -> None:
    if set(decision) != {"rule", "result", "result_sha256"}:
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt decision schema is invalid"
        )
    rule = decision.get("rule")
    result = decision.get("result")
    if rule != CANONICAL_RELEASE_DECISION_RULE or not isinstance(result, Mapping):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt does not use the canonical release decision rule"
        )
    if decision.get("result_sha256") != canonical_hash(result):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt decision result hash differs"
        )
    checks = result.get("checks")
    raw = result.get("raw_inputs")
    if not isinstance(checks, Mapping) or not isinstance(raw, Mapping):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt decision result is incomplete"
        )
    far_detected, far_total = _fraction(
        raw.get("far_detected_over_total"), "far recall"
    )
    medium_detected, medium_total = _fraction(
        raw.get("medium_detected_over_total"), "medium recall"
    )
    near_detected, near_total = _fraction(
        raw.get("near_detected_over_total"), "near recall"
    )
    aggregate_detected, aggregate_total = _fraction(
        raw.get("gating_aggregate_detected_over_total"), "aggregate recall"
    )
    far_fp = _integer(raw.get("far_false_positives"), "far false positives")
    all_predictions = _integer(
        raw.get("all_size_predictions_observed"), "all-size predictions"
    )
    all_fp = _integer(
        raw.get("all_size_false_positives"), "all-size false positives"
    )
    precision_denominator = _integer(
        raw.get("release_precision_denominator"), "precision denominator"
    )
    negative_fp = _integer(
        raw.get("reviewed_negative_false_positives"),
        "reviewed-negative false positives",
    )
    aggregate_precision = _probability(
        raw.get("aggregate_precision"), "aggregate precision"
    )
    aggregate_recall = _probability(
        raw.get("aggregate_recall"), "aggregate recall"
    )
    runtime_p95 = _positive(
        raw.get("runtime_pipeline_p95_ms"), "runtime pipeline p95"
    )
    if (
        far_total != holdout_counts.get("target_33_64")
        or medium_total != holdout_counts.get("target_65_96")
        or near_total != holdout_counts.get("target_gt_96")
        or aggregate_detected != far_detected + medium_detected + near_detected
        or aggregate_total != far_total + medium_total + near_total
        or precision_denominator != aggregate_detected + all_fp
        or all_fp > all_predictions
        or far_fp > all_fp
        or negative_fp > all_fp
        or precision_denominator > all_predictions
        or not math.isclose(
            aggregate_precision,
            aggregate_detected / precision_denominator,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            aggregate_recall,
            aggregate_detected / aggregate_total,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt raw decision counts are inconsistent"
        )
    expected_checks = {
        "far_recall": far_detected / far_total
        >= CANONICAL_RELEASE_DECISION_RULE["minimum_far_recall"],
        "far_false_positives": far_fp
        <= CANONICAL_RELEASE_DECISION_RULE["maximum_far_false_positives"],
        "medium_recall": medium_detected / medium_total
        >= CANONICAL_RELEASE_DECISION_RULE["minimum_medium_recall"],
        "near_recall": near_detected / near_total
        >= CANONICAL_RELEASE_DECISION_RULE["minimum_near_recall"],
        "aggregate_precision": aggregate_precision
        >= CANONICAL_RELEASE_DECISION_RULE["minimum_aggregate_precision"],
        "aggregate_recall": aggregate_recall
        >= CANONICAL_RELEASE_DECISION_RULE["minimum_aggregate_recall"],
        "reviewed_negative_false_positives": negative_fp
        <= CANONICAL_RELEASE_DECISION_RULE[
            "maximum_reviewed_negative_false_positives"
        ],
        "runtime_pipeline_p95_ms": runtime_p95
        <= CANONICAL_RELEASE_DECISION_RULE["maximum_runtime_pipeline_p95_ms"],
    }
    if (
        checks != expected_checks
        or result.get("frozen_rule_passed") is not True
        or result.get("selected_confidence_threshold") != 0.25
        or result.get("scope") != DECISION_RESULT_SCOPE
        or not all(expected_checks.values())
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt does not prove the canonical metric rule passed"
        )


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    verifier: Mapping[str, Any],
    binding: Mapping[str, Any],
    project_root: Path,
) -> None:
    expected_keys = {
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
        set(receipt) != expected_keys
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("kind") != RECEIPT_KIND
        or receipt.get("status")
        != "verified_release_eligible_evidence_not_release_approved"
        or receipt.get("claim_scope")
        != "absolute_threshold_evidence_only_no_incumbent_comparison"
        or receipt.get("release_policy") != release_policy_record()
        or receipt.get("verifier") != verifier
        or receipt.get("candidate") != _receipt_candidate(binding)
        or receipt.get("receipt_content_sha256") != _receipt_content_hash(receipt)
        or receipt.get("canonical_release_policy_matched") is not True
        or receipt.get("frozen_metric_rule_passed") is not True
        or receipt.get("release_evidence_eligible") is not True
        or receipt.get("release_approved") is not False
        or receipt.get("release_pointer_changed") is not False
        or receipt.get("manual_release_review_required") is not True
        or receipt.get("separate_hardware_frozen_build_and_legal_gates_required")
        is not True
    ):
        raise IndependentHoldoutReleaseContractError(
            "portable holdout receipt is not exact release-eligible evidence"
        )
    qualification = receipt.get("qualification")
    expected_qualification = {
        **QUALIFICATION_RECORD,
        "hardware_gate_passed": False,
        "frozen_build_gate_passed": False,
        "legal_redistribution_gate_passed": False,
        "release_gate_passed": False,
        "comparative_incumbent_improvement_proven": False,
    }
    if qualification != expected_qualification:
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt overstates release qualification"
        )
    environment = receipt.get("environment")
    hardware_identity = receipt.get("hardware_identity")
    if not isinstance(environment, Mapping) or not isinstance(
        hardware_identity, Mapping
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt runtime environment/hardware bindings are incomplete"
        )
    validate_release_environment_record(environment, project_root=project_root)
    validate_holdout_hardware_identity(hardware_identity)
    evidence = receipt.get("evidence")
    plan = receipt.get("evaluation_plan")
    ledger = receipt.get("one_time_ledger")
    workload = receipt.get("workload")
    holdout = receipt.get("holdout")
    decision = receipt.get("decision")
    if not all(
        isinstance(item, Mapping)
        for item in (evidence, plan, ledger, workload, holdout, decision)
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt public bindings are incomplete"
        )
    assert isinstance(evidence, Mapping)
    assert isinstance(plan, Mapping)
    assert isinstance(ledger, Mapping)
    assert isinstance(workload, Mapping)
    assert isinstance(holdout, Mapping)
    assert isinstance(decision, Mapping)
    if set(evidence) != {"name", "bytes", "sha256", "content_sha256"} or evidence.get(
        "name"
    ) != "metrics.json":
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt evidence record is invalid"
        )
    if set(plan) != {"bytes", "sha256", "content_sha256"}:
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt plan record is invalid"
        )
    for record, description in ((evidence, "evidence"), (plan, "plan")):
        _integer(record.get("bytes"), f"receipt {description} bytes", 1)
        _sha(record.get("sha256"), f"receipt {description} hash")
        _sha(record.get("content_sha256"), f"receipt {description} content hash")
    counts = holdout.get("counts")
    groups = holdout.get("source_group_inventory")
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
        or set(counts) != set(RELEASE_INVENTORY_MINIMUMS)
        or not isinstance(groups, Mapping)
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt inventory is incomplete"
        )
    _sha(holdout.get("manifest_content_sha256"), "holdout manifest hash")
    _sha(holdout.get("normalized_coco_sha256"), "holdout COCO hash")
    for name, minimum in RELEASE_INVENTORY_MINIMUMS.items():
        count = _integer(counts.get(name), f"holdout {name}")
        if name in GATING_INVENTORY_KEYS and count < minimum:
            raise IndependentHoldoutReleaseContractError(
                f"holdout receipt is below the {name} inventory gate"
            )
    target_groups = groups.get("target_bearing_capture_sessions")
    if (
        set(groups)
        != {
            "definition",
            "overall_capture_sessions",
            "target_bearing_capture_sessions",
            "reviewed_negative_capture_sessions",
        }
        or groups.get("definition")
        != "distinct normalized COCO image session_id values"
        or _integer(groups.get("overall_capture_sessions"), "capture sessions")
        < MINIMUM_CAPTURE_SESSIONS
        or not isinstance(target_groups, Mapping)
        or set(target_groups)
        != {"target_le_32", "target_33_64", "target_65_96", "target_gt_96"}
        or any(
            _integer(target_groups.get(name), f"{name} sessions")
            < MINIMUM_TARGET_BUCKET_CAPTURE_SESSIONS
            for name in ("target_33_64", "target_65_96", "target_gt_96")
        )
        or _integer(
            groups.get("reviewed_negative_capture_sessions"),
            "reviewed-negative sessions",
        )
        < MINIMUM_REVIEWED_NEGATIVE_CAPTURE_SESSIONS
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt is below the source-group gate"
        )
    shape = binding["input_shape_nchw"]
    expected_workload = {
        "backend": "onnxruntime",
        "expected_provider": "DmlExecutionProvider",
        "input_shape_nchw": shape,
        "output_format": binding["output_head"],
        "selected_pipeline": binding["selected_pipeline"],
        "detail_crop_size_source_pixels": binding[
            "detail_crop_size_source_pixels"
        ],
        "confidence_thresholds": DEFAULT_CONFIDENCE_THRESHOLDS,
        "nms_iou_threshold": DEFAULT_NMS_IOU,
        "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
        "warmup_iterations": DEFAULT_WARMUP,
        "require_full_provider": True,
    }
    if set(workload) != set(expected_workload) | {"runtime_pipeline_p95_ms"} or any(
        workload.get(key) != value for key, value in expected_workload.items()
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt workload differs from the adopted DirectML candidate"
        )
    _positive(workload.get("runtime_pipeline_p95_ms"), "runtime pipeline p95")
    _validate_decision(decision, holdout_counts=counts)
    consumption = ledger.get("consumption_event")
    retirement = ledger.get("retirement_event")
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
        or not isinstance(consumption, Mapping)
        or not isinstance(retirement, Mapping)
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt does not prove one-time consumption and retirement"
        )
    assert isinstance(consumption, Mapping)
    assert isinstance(retirement, Mapping)
    if (
        set(consumption)
        != {
            "name",
            "bytes",
            "sha256",
            "event_id",
            "recorded_at_utc",
            "event_content_sha256",
            "evaluation_plan_sha256",
        }
        or set(retirement)
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
        or consumption.get("evaluation_plan_sha256") != plan.get("sha256")
        or retirement.get("evaluation_evidence_sha256")
        != evidence.get("sha256")
        or retirement.get("previous_event_sha256")
        != consumption.get("event_content_sha256")
        or consumption.get("name") != BUNDLE_MEMBER_NAMES["consumption_event"]
        or retirement.get("name") != BUNDLE_MEMBER_NAMES["retirement_event"]
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt ledger does not bind the exact plan/evidence"
        )
    for record, description in (
        (consumption, "consumption"),
        (retirement, "retirement"),
    ):
        _integer(record.get("bytes"), f"receipt {description} event bytes", 1)
        _sha(record.get("sha256"), f"receipt {description} event byte hash")
        _sha(
            record.get("event_content_sha256"),
            f"receipt {description} event content hash",
        )
        _ledger_utc_timestamp(
            record.get("recorded_at_utc"), f"receipt {description} event"
        )


def _validate_plan(
    plan: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    receipt_holdout: Mapping[str, Any],
    receipt_environment: Mapping[str, Any],
    receipt_hardware_identity: Mapping[str, Any],
    project_root: Path,
) -> None:
    expected_plan_keys = {
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
        set(plan) != expected_plan_keys
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("kind") != PLAN_KIND
        or plan.get("status") != PLAN_STATUS
        or plan.get("plan_content_sha256") != _plan_content_hash(plan)
        or plan.get("candidate") != binding
        or plan.get("release_policy") != release_policy_record()
        or plan.get("environment")
        != release_environment_policy_record(project_root)
        or plan.get("environment") != receipt_environment.get("policy")
        or plan.get("hardware_identity") != receipt_hardware_identity
        or plan.get("decision_rule") != CANONICAL_RELEASE_DECISION_RULE
        or plan.get("source") != snapshot
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout plan is not the exact canonical pre-access plan"
        )
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise IndependentHoldoutReleaseContractError(
            "holdout plan runtime contract is missing"
        )
    holdout = plan.get("holdout")
    expected_holdout_keys = {
        "package_id",
        "manifest_content_sha256",
        "pool",
        "counts",
        "images",
        "boxes",
        "source_group_definition",
        "source_group_count",
        "source_group_inventory",
        "ultra_far_le_32_is_descriptive_only",
        "gating_inventory_keys",
        "redistribution_permitted_for_all_sessions",
    }
    if (
        not isinstance(holdout, Mapping)
        or set(holdout) != expected_holdout_keys
        or holdout.get("package_id") != receipt_holdout.get("package_id")
        or holdout.get("manifest_content_sha256")
        != receipt_holdout.get("manifest_content_sha256")
        or holdout.get("counts") != receipt_holdout.get("counts")
        or holdout.get("source_group_inventory")
        != receipt_holdout.get("source_group_inventory")
        or holdout.get("source_group_count")
        != receipt_holdout["source_group_inventory"].get(
            "overall_capture_sessions"
        )
        or holdout.get("pool") != "sealed_release_holdout"
        or holdout.get("source_group_definition") != "capture_session"
        or holdout.get("ultra_far_le_32_is_descriptive_only") is not True
        or holdout.get("gating_inventory_keys") != list(GATING_INVENTORY_KEYS)
        or holdout.get("redistribution_permitted_for_all_sessions") is not True
        or _integer(holdout.get("images"), "holdout plan images", 1) <= 0
        or _integer(holdout.get("boxes"), "holdout plan boxes", 1) <= 0
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout plan package binding differs from the receipt/release contract"
        )
    shape = binding["input_shape_nchw"]
    inference_size = f"{shape[2]}x{shape[3]}"
    expected_runtime_keys = {
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
    if (
        set(runtime) != expected_runtime_keys
        or runtime.get("backend") != "onnxruntime"
        or runtime.get("expected_provider") != "DmlExecutionProvider"
        or re.fullmatch(
            r"(?i)(?:DML|DIRECTML):[0-9]+", str(runtime.get("device") or "")
        )
        is None
        or runtime.get("inference_size") != inference_size
        or runtime.get("input_shape_nchw") != shape
        or runtime.get("output_format") != binding["output_head"]
        or runtime.get("confidence_thresholds") != DEFAULT_CONFIDENCE_THRESHOLDS
        or runtime.get("nms_iou_threshold") != DEFAULT_NMS_IOU
        or runtime.get("warmup_iterations") != DEFAULT_WARMUP
        or runtime.get("bootstrap_samples") != DEFAULT_BOOTSTRAP_SAMPLES
        or runtime.get("require_full_provider") is not True
        or runtime.get("device")
        != receipt_hardware_identity.get("directml_device")
        or runtime.get("detail_crop_size_source_pixels")
        != binding["detail_crop_size_source_pixels"]
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout plan DirectML workload differs from the adopted candidate"
        )
    expected_scope = {
        "dataset": "one sealed independent COCO package only",
        "grouped_v9_development_data_permitted": False,
        "candidate_or_threshold_selection_permitted": False,
        "release_approval_permitted": False,
        "ultra_far_le_32_release_gate": False,
        "claim_scope": "absolute_threshold_evidence_only_no_incumbent_comparison",
    }
    if plan.get("scope") != expected_scope:
        raise IndependentHoldoutReleaseContractError(
            "holdout plan scope permits unsafe development/release behavior"
        )


def _validate_evidence(
    evidence: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    binding: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    evidence_sha256: str,
    plan_sha256: str,
    receipt: Mapping[str, Any],
    project_root: Path,
) -> None:
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
        set(evidence) != expected_top_level
        or evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("kind") != EVIDENCE_KIND
        or evidence.get("status")
        != "valid_final_holdout_evidence_meeting_frozen_rule_not_release_approved"
        or evidence.get("evidence_content_sha256") != _evidence_content_hash(evidence)
        or evidence.get("release_policy") != release_policy_record()
        or evidence.get("candidate") != binding
        or evidence.get("environment") != receipt.get("environment")
        or evidence.get("hardware_identity") != receipt.get("hardware_identity")
        or evidence.get("environment") is None
        or evidence.get("hardware_identity") is None
        or evidence.get("source") != snapshot
        or evidence.get("decision_rule_result") != receipt["decision"]["result"]
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence semantics differ from its plan/receipt/current source"
        )
    assert isinstance(evidence["environment"], Mapping)
    assert isinstance(evidence["hardware_identity"], Mapping)
    validate_release_environment_record(
        evidence["environment"], project_root=project_root
    )
    validate_holdout_hardware_identity(evidence["hardware_identity"])
    if (
        plan.get("environment") != evidence["environment"].get("policy")
        or plan.get("hardware_identity") != evidence["hardware_identity"]
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence runtime environment/hardware differ from its plan"
        )
    expected_plan = {
        "sha256": plan_sha256,
        "content_sha256": plan["plan_content_sha256"],
        "frozen_decision_rule": CANONICAL_RELEASE_DECISION_RULE,
    }
    if evidence.get("evaluation_plan") != expected_plan:
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence does not bind the exact plan"
        )
    if receipt["evidence"]["sha256"] != evidence_sha256 or receipt["evidence"][
        "content_sha256"
    ] != evidence["evidence_content_sha256"]:
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt does not bind the exact evidence bytes"
        )
    plan_holdout = plan["holdout"]
    expected_evidence_holdout = {
        **plan_holdout,
        "normalized_coco_sha256": receipt["holdout"]["normalized_coco_sha256"],
        "exact_member_verification_before_and_after_inference": True,
        "ground_truth_source": "sealed normalized COCO; no grouped-v9 YAML/split",
    }
    if evidence.get("holdout") != expected_evidence_holdout:
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence package/COCO inventory differs from the plan/receipt"
        )
    model_artifact = evidence.get("model_artifact")
    expected_model = {
        "backend": "onnxruntime",
        "content_sha256": binding["model_content_sha256"],
        "members": [
            {
                "name": item["name"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in binding["model_artifacts"]
        ],
    }
    if model_artifact != expected_model:
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence model artifact differs from the adopted pointer"
        )
    configuration = evidence.get("configuration")
    runtime = evidence.get("runtime")
    metrics = evidence.get("metrics")
    if not all(isinstance(item, Mapping) for item in (configuration, runtime, metrics)):
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence runtime/metrics are incomplete"
        )
    assert isinstance(configuration, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(metrics, Mapping)
    if any(configuration.get(key) != value for key, value in plan["runtime"].items()):
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence configuration differs from the frozen plan"
        )
    detail_enabled = binding["detail_crop_size_source_pixels"] > 0
    expected_pipeline = (
        "rectangular_full_frame_plus_center_model_aspect_detail_merged"
        if detail_enabled
        else "rectangular_full_frame_primary_only"
    )
    detail_stats = configuration.get("detail_stats")
    if (
        configuration.get("evaluation_mode")
        != "sealed_independent_exact_application_runtime_artifact"
        or configuration.get("adopted_tournament_pipeline")
        != binding["selected_pipeline"]
        or configuration.get("configured_pipeline") != expected_pipeline
        or configuration.get("primary_reference_retained") is not True
        or not isinstance(detail_stats, Mapping)
        or detail_stats.get("enabled") is not detail_enabled
        or (
            detail_enabled
            and (
                detail_stats.get("frames_seen") != plan_holdout["images"]
                or not isinstance(detail_stats.get("frames_applied"), int)
                or detail_stats.get("frames_applied") <= 0
                or not isinstance(detail_stats.get("last_plan"), Mapping)
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
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence did not run the adopted primary/detail pipeline"
        )
    summary = runtime.get("summary")
    timing = runtime.get("timing_ms_per_image")
    if not isinstance(summary, Mapping) or not isinstance(timing, Mapping):
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence runtime provider/timing summary is incomplete"
        )
    active = summary.get("active_providers")
    pipeline_timing = timing.get("runtime_pipeline")
    planned_device = str(plan["runtime"]["device"])
    planned_adapter = planned_device.rsplit(":", 1)[1]
    if (
        not isinstance(summary.get("requested_device_input"), str)
        or str(summary.get("requested_device_input")).casefold()
        != planned_device.casefold()
        or summary.get("requested_provider") != "DmlExecutionProvider"
        or summary.get("requested_device") != "DmlExecutionProvider"
        or isinstance(active, (str, bytes))
        or not isinstance(active, Sequence)
        or "DmlExecutionProvider" not in active
        or summary.get("provider_option_overrides")
        != {"DmlExecutionProvider": {"device_id": planned_adapter}}
        or summary.get("require_full_provider") is not True
        or summary.get("runtime_ep_fail_fallback_disabled") is not True
        or not isinstance(summary.get("configured_session_options"), Mapping)
        or summary["configured_session_options"].get("disable_cpu_ep_fallback")
        is not True
        or summary.get("output_format") != binding["output_head"]
        or summary.get("input_shape") != binding["input_shape_nchw"]
        or summary.get("declared_input_shape") != binding["input_shape_nchw"]
        or summary.get("configured_input_shape") != binding["input_shape_nchw"]
        or runtime.get("declared_static_input_shape_nchw")
        != binding["input_shape_nchw"]
        or not isinstance(pipeline_timing, Mapping)
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence does not prove the exact full DirectML graph/shape"
        )
    runtime_p95 = _positive(
        pipeline_timing.get("p95"), "holdout evidence runtime pipeline p95"
    )
    if runtime_p95 != receipt["workload"]["runtime_pipeline_p95_ms"]:
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence runtime p95 differs from the receipt"
        )
    if (
        metrics.get("images") != plan_holdout["images"]
        or metrics.get("ground_truth_boxes") != plan_holdout["boxes"]
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence metric coverage differs from the frozen package"
        )
    size = metrics.get("size_bucket_detection")
    negative = metrics.get("reviewed_negative_detection")
    if not isinstance(size, Mapping) or not isinstance(negative, Mapping):
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence size/negative metrics are missing"
        )
    size_points = size.get("operating_points")
    negative_points = negative.get("operating_points")
    if not isinstance(size_points, Mapping) or not isinstance(
        negative_points, Mapping
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence operating points are missing"
        )
    if set(size_points) != {"0.25", "0.45"} or set(negative_points) != {
        "0.25",
        "0.45",
    }:
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence must contain exactly both frozen operating points"
        )
    selected = size_points.get("0.25")
    selected_negative = negative_points.get("0.25")
    bucket_names = (
        "ultra_far_le_32px",
        "far_33_to_64px",
        "medium_65_to_96px",
        "near_gt_96px",
    )
    if not isinstance(selected, Mapping) or not isinstance(
        selected_negative, Mapping
    ) or any(not isinstance(selected.get(name), Mapping) for name in bucket_names):
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence selected operating point is incomplete"
        )
    bucket_counts = {
        "ultra_far_le_32px": receipt["holdout"]["counts"]["target_le_32"],
        "far_33_to_64px": receipt["holdout"]["counts"]["target_33_64"],
        "medium_65_to_96px": receipt["holdout"]["counts"]["target_65_96"],
        "near_gt_96px": receipt["holdout"]["counts"]["target_gt_96"],
    }
    reviewed_negative_images = receipt["holdout"]["counts"]["reviewed_negatives"]
    for confidence in ("0.25", "0.45"):
        point_set = size_points[confidence]
        negative_point = negative_points[confidence]
        if (
            not isinstance(point_set, Mapping)
            or set(point_set) != set(bucket_counts)
            or not isinstance(negative_point, Mapping)
        ):
            raise IndependentHoldoutReleaseContractError(
                f"holdout evidence {confidence} operating-point inventory is invalid"
            )
        for name, total in bucket_counts.items():
            point = point_set[name]
            if not isinstance(point, Mapping):
                raise IndependentHoldoutReleaseContractError(
                    f"holdout evidence {confidence} {name} point is missing"
                )
            tp = _integer(point.get("detected_true_positives"), f"{confidence} {name} TP")
            misses = _integer(point.get("missed_false_negatives"), f"{confidence} {name} FN")
            fp = _integer(point.get("false_positives"), f"{confidence} {name} FP")
            predictions = _integer(point.get("predictions"), f"{confidence} {name} predictions")
            if (
                point.get("ground_truth_total") != total
                or tp + misses != total
                or tp + fp != predictions
                or point.get("detected_over_total") != f"{tp}/{total}"
                or point.get("precision") != (tp / predictions if predictions else None)
                or point.get("recall") != (tp / total if total else None)
            ):
                raise IndependentHoldoutReleaseContractError(
                    f"holdout evidence {confidence} {name} counts/rates are inconsistent"
                )
        negative_fp_point = _integer(
            negative_point.get("false_positives"),
            f"{confidence} reviewed-negative false positives",
        )
        negative_images_with_fp_point = _integer(
            negative_point.get("negative_images_with_false_positive"),
            f"{confidence} reviewed-negative images with false positives",
        )
        if (
            negative_point.get("reviewed_negative_images")
            != reviewed_negative_images
            or negative_images_with_fp_point > reviewed_negative_images
            or negative_point.get("false_positives_per_image")
            != negative_fp_point / reviewed_negative_images
            or negative_point.get("negative_image_false_positive_rate")
            != negative_images_with_fp_point / reviewed_negative_images
        ):
            raise IndependentHoldoutReleaseContractError(
                f"holdout evidence {confidence} reviewed-negative rates are inconsistent"
            )
    far = selected["far_33_to_64px"]
    medium = selected["medium_65_to_96px"]
    near = selected["near_gt_96px"]
    gating_tp = sum(
        int(selected[name]["detected_true_positives"])
        for name in bucket_names[1:]
    )
    gating_total = sum(bucket_counts[name] for name in bucket_names[1:])
    all_predictions = sum(int(selected[name]["predictions"]) for name in bucket_names)
    all_fp = sum(int(selected[name]["false_positives"]) for name in bucket_names)
    denominator = gating_tp + all_fp
    negative_fp = _integer(
        selected_negative.get("false_positives"),
        "reviewed-negative false positives",
    )
    negative_images_with_fp = _integer(
        selected_negative.get("negative_images_with_false_positive"),
        "reviewed-negative images with false positives",
    )
    if (
        selected_negative.get("reviewed_negative_images")
        != reviewed_negative_images
        or negative_images_with_fp > reviewed_negative_images
        or selected_negative.get("false_positives_per_image")
        != negative_fp / reviewed_negative_images
        or selected_negative.get("negative_image_false_positive_rate")
        != negative_images_with_fp / reviewed_negative_images
    ):
        raise IndependentHoldoutReleaseContractError(
            "reviewed-negative evidence counts/rates are inconsistent"
        )
    expected_raw = {
        "far_detected_over_total": (
            f"{far['detected_true_positives']}/{bucket_counts['far_33_to_64px']}"
        ),
        "far_false_positives": far["false_positives"],
        "medium_detected_over_total": (
            f"{medium['detected_true_positives']}/{bucket_counts['medium_65_to_96px']}"
        ),
        "near_detected_over_total": (
            f"{near['detected_true_positives']}/{bucket_counts['near_gt_96px']}"
        ),
        "gating_aggregate_detected_over_total": f"{gating_tp}/{gating_total}",
        "all_size_predictions_observed": all_predictions,
        "all_size_false_positives": all_fp,
        "release_precision_denominator": denominator,
        "aggregate_precision": gating_tp / denominator if denominator else None,
        "aggregate_recall": gating_tp / gating_total,
        "reviewed_negative_false_positives": negative_fp,
        "runtime_pipeline_p95_ms": runtime_p95,
    }
    if receipt["decision"]["result"]["raw_inputs"] != expected_raw:
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt decision was not recomputed from bundled evidence"
        )
    qualification = evidence.get("qualification")
    expected_qualification = {
        **QUALIFICATION_RECORD,
        "final_holdout_evaluation_completed": True,
        "canonical_release_policy_matched": True,
        "frozen_metric_rule_passed": True,
        "release_evidence_eligible": True,
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
    if qualification != expected_qualification:
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence is not canonical release-eligible evidence"
        )


def _validate_ledger_event(
    event: Mapping[str, Any], *, expected_hash: str, description: str
) -> None:
    body = dict(event)
    body.pop("event_content_sha256", None)
    # The sealed-package ledger has its own compact canonical encoding.  Keep
    # this deliberately distinct from release-model ``canonical_hash``.
    ledger_hash = sha256(
        (
            json.dumps(
                body,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if (
        event.get("event_content_sha256") != ledger_hash
        or event.get("event_content_sha256") != expected_hash
        or not isinstance(event.get("event_id"), str)
        or EVENT_ID_RE.fullmatch(str(event.get("event_id"))) is None
    ):
        raise IndependentHoldoutReleaseContractError(
            f"{description} ledger event is invalid"
        )


def _ledger_utc_timestamp(value: Any, description: str) -> datetime:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value
    ) is None:
        raise IndependentHoldoutReleaseContractError(
            f"{description} ledger UTC timestamp is invalid"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise IndependentHoldoutReleaseContractError(
            f"{description} ledger UTC timestamp is invalid"
        ) from exc
    if parsed.tzinfo != timezone.utc:
        raise IndependentHoldoutReleaseContractError(
            f"{description} ledger UTC timestamp is invalid"
        )
    return parsed


def validate_publication_bundle(
    bundle: Path,
    *,
    project_root: Path,
    candidate_binding_loader: Callable[[Path], Mapping[str, Any]] = (
        public_candidate_binding
    ),
    source_snapshot_loader: Callable[[Path], Mapping[str, Any]] = source_snapshot,
    receipt_verifier_loader: Callable[[Path], Mapping[str, Any]] = (
        receipt_verifier_record
    ),
) -> dict[str, Any]:
    """Revalidate the complete redacted bundle without the sealed package.

    This proves byte/semantic consistency against the current checkout.  It is
    intentionally not an origin proof; callers must additionally authenticate
    the successful protected Actions run and its immutable artifact identity.
    """

    root = bundle.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise IndependentHoldoutReleaseContractError(
            "holdout publication bundle must be a real directory"
        )
    expected_files = {BUNDLE_MANIFEST_NAME, *BUNDLE_MEMBER_NAMES.values()}
    files: set[str] = set()
    directories: set[str] = set()
    for child in root.rglob("*"):
        relative = child.relative_to(root).as_posix()
        if child.is_symlink():
            raise IndependentHoldoutReleaseContractError(
                f"holdout publication bundle contains a symlink: {relative}"
            )
        if child.is_file():
            files.add(relative)
        elif child.is_dir():
            directories.add(relative)
        else:
            raise IndependentHoldoutReleaseContractError(
                f"holdout publication bundle contains a special member: {relative}"
            )
    if files != expected_files or directories != {"ledger"}:
        raise IndependentHoldoutReleaseContractError(
            "holdout publication bundle has an unexpected fixed inventory"
        )
    values: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for role, relative in BUNDLE_MEMBER_NAMES.items():
        value, payload = strict_json_file(
            root.joinpath(*PurePosixPath(relative).parts), f"holdout bundle {role}"
        )
        values[role] = value
        payloads[role] = payload
    for role in ("receipt", "evidence", "evaluation_plan"):
        if payloads[role] != canonical_json_bytes(values[role]):
            raise IndependentHoldoutReleaseContractError(
                f"holdout bundle {role} is not canonical release JSON"
            )
    for role in ("consumption_event", "retirement_event"):
        expected = (
            json.dumps(
                values[role],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if payloads[role] != expected:
            raise IndependentHoldoutReleaseContractError(
                f"holdout bundle {role} is not canonical ledger JSON"
            )
    manifest, manifest_payload = strict_json_file(
        root / BUNDLE_MANIFEST_NAME, "holdout bundle manifest"
    )
    receipt = values["receipt"]
    binding = dict(candidate_binding_loader(project_root))
    snapshot = dict(source_snapshot_loader(project_root))
    verifier = dict(receipt_verifier_loader(project_root))
    _validate_receipt(
        receipt,
        verifier=verifier,
        binding=binding,
        project_root=project_root,
    )
    _validate_plan(
        values["evaluation_plan"],
        binding=binding,
        snapshot=snapshot,
        receipt_holdout=receipt["holdout"],
        receipt_environment=receipt["environment"],
        receipt_hardware_identity=receipt["hardware_identity"],
        project_root=project_root,
    )
    evidence_sha = sha256(payloads["evidence"]).hexdigest()
    plan_sha = sha256(payloads["evaluation_plan"]).hexdigest()
    if (
        receipt["evidence"]["bytes"] != len(payloads["evidence"])
        or receipt["evidence"]["sha256"] != evidence_sha
        or receipt["evaluation_plan"]["bytes"]
        != len(payloads["evaluation_plan"])
        or receipt["evaluation_plan"]["sha256"] != plan_sha
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout receipt byte records differ from the bundle"
        )
    _validate_evidence(
        values["evidence"],
        plan=values["evaluation_plan"],
        binding=binding,
        snapshot=snapshot,
        evidence_sha256=evidence_sha,
        plan_sha256=plan_sha,
        receipt=receipt,
        project_root=project_root,
    )
    consumption = values["consumption_event"]
    retirement = values["retirement_event"]
    receipt_consumption = receipt["one_time_ledger"]["consumption_event"]
    receipt_retirement = receipt["one_time_ledger"]["retirement_event"]
    expected_consumption_keys = {
        "schema_version",
        "sequence",
        "event_id",
        "operation",
        "recorded_at_utc",
        "actor_id",
        "dataset_manifest_content_sha256",
        "previous_event_sha256",
        "purpose",
        "evaluation_plan_sha256",
        "event_content_sha256",
    }
    expected_retirement_keys = {
        "schema_version",
        "sequence",
        "event_id",
        "operation",
        "recorded_at_utc",
        "actor_id",
        "dataset_manifest_content_sha256",
        "previous_event_sha256",
        "reason",
        "evaluation_evidence_sha256",
        "event_content_sha256",
    }
    if (
        set(consumption) != expected_consumption_keys
        or set(retirement) != expected_retirement_keys
        or consumption.get("schema_version") != 1
        or consumption.get("sequence") != 1
        or consumption.get("previous_event_sha256") is not None
        or retirement.get("schema_version") != 2
        or retirement.get("sequence") != 2
        or consumption.get("event_id") != receipt_consumption.get("event_id")
        or retirement.get("event_id") != receipt_retirement.get("event_id")
        or consumption.get("event_id") == retirement.get("event_id")
        or consumption.get("dataset_manifest_content_sha256")
        != receipt["holdout"]["manifest_content_sha256"]
        or retirement.get("dataset_manifest_content_sha256")
        != receipt["holdout"]["manifest_content_sha256"]
        or consumption.get("actor_id") != retirement.get("actor_id")
        or receipt_consumption.get("bytes") != len(payloads["consumption_event"])
        or receipt_consumption.get("sha256")
        != sha256(payloads["consumption_event"]).hexdigest()
        or receipt_retirement.get("bytes") != len(payloads["retirement_event"])
        or receipt_retirement.get("sha256")
        != sha256(payloads["retirement_event"]).hexdigest()
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout bundled ledger schema/bytes differ from the receipt/package"
        )
    consumed_at = _ledger_utc_timestamp(
        consumption.get("recorded_at_utc"), "consumption"
    )
    retired_at = _ledger_utc_timestamp(
        retirement.get("recorded_at_utc"), "retirement"
    )
    if retired_at <= consumed_at:
        raise IndependentHoldoutReleaseContractError(
            "holdout retirement time must be strictly after pre-access consumption"
        )
    _validate_ledger_event(
        consumption,
        expected_hash=receipt_consumption["event_content_sha256"],
        description="consumption",
    )
    _validate_ledger_event(
        retirement,
        expected_hash=receipt_retirement["event_content_sha256"],
        description="retirement",
    )
    if (
        consumption.get("operation") != "consumed"
        or consumption.get("evaluation_plan_sha256") != plan_sha
        or retirement.get("operation") != "retired"
        or retirement.get("evaluation_evidence_sha256") != evidence_sha
        or retirement.get("previous_event_sha256")
        != consumption.get("event_content_sha256")
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout ledger does not bind exact one-time plan/evidence retirement"
        )
    access = values["evidence"].get("one_time_access")
    expected_access = {
        "event_id": consumption["event_id"],
        "actor_id": consumption["actor_id"],
        "purpose": consumption["purpose"],
        "retirement_event_id": retirement["event_id"],
        "retirement_reason": retirement["reason"],
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
        raise IndependentHoldoutReleaseContractError(
            "holdout evidence access declaration differs from its exact ledger"
        )
    members = manifest.get("members")
    bindings = manifest.get("bindings")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != BUNDLE_KIND
        or manifest.get("status")
        != "verified_release_eligible_publication_inputs_not_release_approved"
        or manifest.get("bundle_content_sha256") != _bundle_content_hash(manifest)
        or manifest.get("canonical_release_policy_matched") is not True
        or manifest.get("release_evidence_eligible") is not True
        or manifest.get("authenticated_origin_required") is not True
        or manifest.get("release_approved") is not False
        or manifest.get("release_pointer_changed") is not False
        or manifest.get("qualification") != QUALIFICATION_RECORD
        or not isinstance(members, Mapping)
        or set(members) != set(BUNDLE_MEMBER_NAMES)
        or not isinstance(bindings, Mapping)
    ):
        raise IndependentHoldoutReleaseContractError(
            "holdout publication bundle manifest is invalid or ineligible"
        )
    assert isinstance(members, Mapping)
    for role, relative in BUNDLE_MEMBER_NAMES.items():
        record = members.get(role)
        if record != {
            "path": relative,
            "bytes": len(payloads[role]),
            "sha256": sha256(payloads[role]).hexdigest(),
        }:
            raise IndependentHoldoutReleaseContractError(
                f"holdout bundle manifest does not bind exact {role} bytes"
            )
    expected_bindings = {
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
        "consumption_event_content_sha256": receipt_consumption[
            "event_content_sha256"
        ],
        "retirement_event_content_sha256": receipt_retirement[
            "event_content_sha256"
        ],
    }
    if bindings != expected_bindings:
        raise IndependentHoldoutReleaseContractError(
            "holdout bundle manifest bindings differ from its exact receipt"
        )
    if manifest_payload != canonical_json_bytes(manifest):
        raise IndependentHoldoutReleaseContractError(
            "holdout bundle manifest is not canonical JSON"
        )
    return {
        "schema_version": 1,
        "status": "verified_public_holdout_bundle_requires_authenticated_origin",
        "bundle_manifest_sha256": sha256(manifest_payload).hexdigest(),
        "bundle_content_sha256": manifest["bundle_content_sha256"],
        "receipt_sha256": sha256(payloads["receipt"]).hexdigest(),
        "receipt_content_sha256": receipt["receipt_content_sha256"],
        "release_policy_sha256": release_policy_record()["policy_sha256"],
        "source_snapshot_sha256": receipt["verifier"][
            "source_snapshot_sha256"
        ],
        "candidate_binding_sha256": canonical_hash(receipt["candidate"]),
        "environment_record_sha256": receipt["environment"][
            "record_content_sha256"
        ],
        "hardware_identity_sha256": receipt["hardware_identity"][
            "content_sha256"
        ],
        "hardware_identity": dict(receipt["hardware_identity"]),
        "plan_sha256": plan_sha,
        "evidence_sha256": evidence_sha,
        "consumption_event_content_sha256": consumption[
            "event_content_sha256"
        ],
        "retirement_event_content_sha256": retirement[
            "event_content_sha256"
        ],
        "canonical_release_policy_matched": True,
        "release_evidence_eligible": True,
        "consumed_exactly_once": True,
        "retired": True,
        "authenticated_origin_required": True,
        "release_approved": False,
    }
