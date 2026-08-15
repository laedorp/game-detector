#!/usr/bin/env python3
"""Stage, physically qualify, and transactionally publish DirectML releases.

A tag build is deliberately non-publishing.  It emits one content-manifested
candidate containing Linux CPU and Windows DirectML archives.  Two independent
physical qualification runs then seal evidence for the exact DirectML archive
on the required AMD and NVIDIA products.  Only the protected publication
workflow may create a draft release; it makes the release public only after all
source, artifact, evidence, and uploaded-byte identities have been rechecked.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from math import gcd
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from scripts.manage_cuda_release_attachment import (
    AttachmentError,
    GitHubApi,
    LIVE_TIMING_FIELDS,
    _artifact_digest,
    _asset_id,
    _asset_signature,
    _assets_by_name,
    _finite_number,
    _read_json_object,
    _records_by_key,
    _release_assets,
    _require_bundle_path_suffix,
    _require_close_timing,
    _require_sha,
    _safe_zip_name,
    _single_run_artifact,
    _strict_json_loads,
    _utc_datetime,
    _validate_dependency_distribution_records,
    _validate_release_default_model_record,
    _validate_timing_summary,
    _verify_actions_run,
    _write_json_atomic,
    render_checksum_manifest,
    resolve_tag_commit,
    sha256_bytes,
    sha256_file,
    validate_content_manifest,
    write_content_manifest,
)
from utils.independent_holdout_release_contract import (
    BUNDLE_MANIFEST_NAME as HOLDOUT_BUNDLE_MANIFEST_NAME,
    BUNDLE_MEMBER_NAMES as HOLDOUT_BUNDLE_MEMBER_NAMES,
    PUBLIC_RELEASE_RECEIPT_NAME as PUBLIC_HOLDOUT_RECEIPT_NAME,
    IndependentHoldoutReleaseContractError,
    canonical_hash as holdout_canonical_hash,
    canonical_json_bytes as holdout_canonical_json_bytes,
    public_candidate_binding,
    release_policy_record as independent_holdout_release_policy,
    strict_json_file as strict_holdout_json_file,
    validate_publication_bundle as validate_public_holdout_bundle,
)


SOURCE_WORKFLOW = ".github/workflows/release-bundles.yml"
QUALIFICATION_WORKFLOW = ".github/workflows/qualify-windows-directml.yml"
INDEPENDENT_HOLDOUT_WORKFLOW = ".github/workflows/qualify-independent-holdout.yml"
SOURCE_EVENT = "push"
QUALIFICATION_EVENT = "workflow_dispatch"
INDEPENDENT_HOLDOUT_EVENT = "workflow_dispatch"
SOURCE_ARTIFACT_NAME = "ProAim-Release-Candidate"
CANDIDATE_MANIFEST_NAME = "RELEASE-CANDIDATE-MANIFEST.json"
CANDIDATE_MANIFEST_KIND = "proaim-release-candidate"
LINUX_ARCHIVE_NAME = "ProAim-Linux-x64.zip"
DIRECTML_ARCHIVE_NAME = "ProAim-Windows-x64-DirectML.zip"
CHECKSUM_NAME = "SHA256SUMS.txt"
RAW_CONTENT_MANIFEST_NAME = "RAW-CONTENT-MANIFEST.json"
RAW_CONTENT_KIND = "proaim-directml-raw-qualification"
STAGED_CONTENT_MANIFEST_NAME = "STAGED-CONTENT-MANIFEST.json"
STAGED_CONTENT_KIND = "proaim-directml-release-stage"
QUALIFICATION_MANIFEST_NAME = "qualification-manifest.json"
PHYSICAL_ATTESTATION_NAME = "PHYSICAL-DIRECTML-ATTESTATION.json"
LOCAL_OBSERVATION_NAME = "LOCAL-DIRECTML-OBSERVATION.json"
TELEMETRY_NAME = "directml-gpu-engine-telemetry.jsonl"
RUNNER_INVARIANT_NAME = "DIRECTML-RUNNER-INVARIANT.json"
CANDIDATE_INSPECTION_NAME = "candidate-inspection.json"
SOURCE_RECORD_NAME = "verified-source.json"
RELEASE_ATTESTATION_NAME = "DIRECTML-RELEASE-ATTESTATION.json"
PUBLIC_RELEASE_RECEIPT_NAME = "ProAim-DirectML-Release-Qualification.json"
HOLDOUT_EVIDENCE_ARTIFACT_NAME = "ProAim-Independent-Holdout-Evidence"
HOLDOUT_ATTESTATION_ARTIFACT_NAME = "ProAim-Independent-Holdout-Attestation"
HOLDOUT_PREREQUISITE_ARTIFACT_NAME = "ProAim-Verified-Holdout-Prerequisites"
HOLDOUT_PLAN_ARTIFACT_NAME = "ProAim-Independent-Holdout-Frozen-Plan"
HOLDOUT_ATTESTATION_NAME = "AUTHENTICATED-INDEPENDENT-HOLDOUT-ATTESTATION.json"
HOLDOUT_ATTESTATION_KIND = "proaim-authenticated-independent-holdout-attestation"
HOLDOUT_ATTESTATION_STATUS = (
    "protected_workflow_authenticated_release_eligible_holdout_evidence"
)
PRIVATE_HOLDOUT_ROOT = "private-holdout"
EXPECTED_ARCHIVES = frozenset({LINUX_ARCHIVE_NAME, DIRECTML_ARCHIVE_NAME})
REQUIRED_DIRECTML_BUNDLE_FILES = frozenset(
    {
        "ProAim/ProAimCLI.exe",
        "ProAim/BUILD-INFO.json",
        "ProAim/DEPENDENCY-MANIFEST.json",
        "ProAim/Qualify-ProAimGpu.ps1",
        "ProAim/LICENSE",
        "ProAim/THIRD_PARTY_NOTICES.md",
    }
)
SOFTWARE_EVIDENCE_FILES = frozenset(
    {
        "TASK-MANAGER-INSTRUCTIONS.txt",
        "bundle-BUILD-INFO.json",
        "bundle-DEPENDENCY-MANIFEST.json",
        "runtime-info.json",
        "runtime-info.stderr.txt",
        "benchmark-release-default.json",
        "benchmark-release-default.stderr.txt",
        "live-release-default-no-preview.json",
        "live-release-default-no-preview.stdout.txt",
        "live-release-default-no-preview.stderr.txt",
        "live-release-default-preview-15.json",
        "live-release-default-preview-15.stdout.txt",
        "live-release-default-preview-15.stderr.txt",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[1-9][0-9]{0,19}$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TAG_RE = re.compile(r"^v[A-Za-z0-9][A-Za-z0-9._-]{0,126}$")

GPU_PRODUCTS = {
    "amd_rx_6950_xt": {
        "product_name": "AMD Radeon RX 6950 XT",
        "vendor_id": "0x1002",
        "receipt_name": "ProAim-Windows-DirectML-AMD-RX-6950-XT-Qualification.json",
    },
    "nvidia_rtx_5060_laptop": {
        "product_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "vendor_id": "0x10de",
        "receipt_name": "ProAim-Windows-DirectML-NVIDIA-RTX-5060-Laptop-Qualification.json",
    },
}
REQUIRED_ROLES = tuple(GPU_PRODUCTS)

DIRECTML_POLICY = {
    "benchmark_samples": 32,
    "benchmark_warmup": 30,
    "benchmark_iterations_per_repeat": 100,
    "benchmark_repeats": 3,
    "benchmark_max_p95_inference_ms": 35.0,
    "live_requested_max_frames": 1000,
    "live_max_seconds": 60.0,
    "live_min_processed_frames": 120,
    "live_min_elapsed_seconds": 2.0,
    "live_min_elapsed_fps": 20.0,
    "live_min_update_fps": 20.0,
    "live_max_p95_observed_pipeline_ms": 50.0,
    "live_max_p95_freshness_latency_ms": 50.0,
    "telemetry_interval_milliseconds": 500,
    "telemetry_min_positive_samples_per_run": 1,
    "telemetry_min_total_samples_per_run": 2,
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_sha(value: str, description: str) -> str:
    normalized = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise AttachmentError(f"{description} must be a lowercase SHA-256")
    return normalized


def _run_id(value: str | int, description: str) -> int:
    text = str(value).strip()
    if not RUN_ID_RE.fullmatch(text):
        raise AttachmentError(f"{description} must be a positive numeric Actions run ID")
    return int(text)


def _first_run_attempt(run: Mapping[str, Any], description: str) -> int:
    """Require the exact, non-coerced first Actions run attempt."""

    value = run.get("run_attempt")
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise AttachmentError(f"{description} must be exact Actions run attempt 1")
    return value


def _single_line(value: str, description: str, maximum: int = 240) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > maximum or any(c in normalized for c in "\r\n\0"):
        raise AttachmentError(f"{description} must be one non-empty line of at most {maximum} characters")
    return normalized


def normalize_product_name(value: str) -> str:
    return " ".join(_single_line(value, "GPU product name").split()).casefold()


def expected_physical_confirmation(tag: str, role: str) -> str:
    product = GPU_PRODUCTS[role]["product_name"]
    return f"I ATTEST THAT I OBSERVED {product} RUN DIRECTML FOR {tag}"


def qualification_artifact_name(role: str) -> str:
    return f"ProAim-Windows-DirectML-{role}-Qualification-Evidence"


def qualification_archive_name(role: str) -> str:
    return qualification_artifact_name(role) + ".zip"


def _validate_identity(repository: str, tag: str, role: str | None = None) -> None:
    if not REPOSITORY_RE.fullmatch(repository):
        raise AttachmentError("repository must use owner/name syntax")
    if not TAG_RE.fullmatch(tag):
        raise AttachmentError("tag must be a canonical v* release tag")
    if role is not None and role not in GPU_PRODUCTS:
        raise AttachmentError(f"unsupported DirectML GPU role: {role!r}")


def candidate_context(repository: str, tag: str, tag_commit: str, source_run_id: int) -> dict[str, Any]:
    return {
        "repository": repository,
        "tag": tag,
        "tag_commit": _require_sha(tag_commit, "candidate tag commit"),
        "source_build_run_id": int(source_run_id),
        "public_release_created": False,
    }


def stage_candidate(
    root: Path,
    *,
    repository: str,
    tag: str,
    tag_commit: str,
    source_run_id: int,
) -> dict[str, Any]:
    _validate_identity(repository, tag)
    if not root.is_dir():
        raise AttachmentError(f"candidate stage root not found: {root}")
    actual = {path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()}
    if actual != EXPECTED_ARCHIVES or any(path.is_symlink() or path.is_dir() for path in root.iterdir()):
        raise AttachmentError("candidate stage must initially contain exactly the Linux and DirectML ZIPs")
    for name in EXPECTED_ARCHIVES:
        path = root / name
        if path.stat().st_size <= 0 or not zipfile.is_zipfile(path):
            raise AttachmentError(f"candidate archive is empty or invalid: {name}")
    return write_content_manifest(
        root=root,
        output=root / CANDIDATE_MANIFEST_NAME,
        kind=CANDIDATE_MANIFEST_KIND,
        context=candidate_context(repository, tag, tag_commit, source_run_id),
    )


def _releases(api: GitHubApi) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for page in range(1, 101):
        current = api.get_json_list(f"/releases?per_page=100&page={page}")
        values.extend(current)
        if len(current) < 100:
            return values
    raise AttachmentError("GitHub release listing exceeded its safety limit")


def _matching_releases(api: GitHubApi, tag: str) -> list[dict[str, Any]]:
    return [release for release in _releases(api) if release.get("tag_name") == tag]


def verify_source_contract(
    api: GitHubApi,
    *,
    repository: str,
    tag: str,
    source_run_id: int,
    require_no_release: bool = True,
) -> dict[str, Any]:
    _validate_identity(repository, tag)
    tag_commit = resolve_tag_commit(api, tag)
    releases = _matching_releases(api, tag)
    if require_no_release and releases:
        raise AttachmentError("a GitHub release already exists for this tag; publication is not restart-safe")
    run, workflow_id = _verify_actions_run(
        api,
        repository=repository,
        run_id=source_run_id,
        tag_commit=tag_commit,
        expected_event=SOURCE_EVENT,
        expected_workflow=SOURCE_WORKFLOW,
        description="staged release source run",
    )
    artifact = _single_run_artifact(
        api,
        run_id=source_run_id,
        expected_name=SOURCE_ARTIFACT_NAME,
        description="staged release source run",
        require_digest=True,
    )
    return {
        "tag_commit": tag_commit,
        "release_absent": not releases,
        "source_build_run": {
            "id": source_run_id,
            "event": run["event"],
            "status": run["status"],
            "conclusion": run["conclusion"],
            "head_sha": str(run["head_sha"]).lower(),
            "html_url": str(run.get("html_url") or ""),
            "workflow_id": workflow_id,
            "workflow_path": SOURCE_WORKFLOW,
            "run_attempt": _first_run_attempt(run, "staged release source run"),
        },
        "candidate_artifact": artifact,
    }


def _downloaded_candidate_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise AttachmentError(f"downloaded candidate directory not found: {directory}")
    paths = list(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise AttachmentError("downloaded candidate artifact contains a symbolic link")
    files = {path.relative_to(directory).as_posix(): path for path in paths if path.is_file()}
    expected = EXPECTED_ARCHIVES | {CANDIDATE_MANIFEST_NAME}
    if set(files) != expected:
        raise AttachmentError("downloaded candidate artifact has an unexpected file set")
    return files


def _extract_directml_bundle(
    archive_path: Path,
    *,
    expected_sha256: str,
    expected_commit: str,
    extraction_root: Path,
) -> dict[str, Any]:
    expected_sha256 = _normalize_sha(expected_sha256, "DirectML ZIP SHA-256")
    if archive_path.name != DIRECTML_ARCHIVE_NAME or sha256_file(archive_path) != expected_sha256:
        raise AttachmentError("DirectML archive name or SHA-256 differs from the dispatched candidate")
    if extraction_root.exists():
        raise AttachmentError(f"refusing to reuse candidate extraction path: {extraction_root}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if len(archive.infolist()) > 100_000:
                raise AttachmentError("DirectML candidate exceeds the entry-count limit")
            entries: dict[str, zipfile.ZipInfo] = {}
            folded: set[str] = set()
            total_size = 0
            for info in archive.infolist():
                name = _safe_zip_name(info)
                if not name:
                    continue
                if name.casefold() in folded:
                    raise AttachmentError("DirectML candidate contains a duplicate path")
                folded.add(name.casefold())
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode) if mode else 0
                if file_type not in (0, stat.S_IFREG, stat.S_IFDIR) or info.flag_bits & 0x1:
                    raise AttachmentError("DirectML candidate contains a link, special, or encrypted entry")
                if info.file_size < 0 or info.file_size > 4 * 1024 * 1024 * 1024:
                    raise AttachmentError("DirectML candidate contains an unreasonably large entry")
                total_size += info.file_size
                if total_size > 8 * 1024 * 1024 * 1024:
                    raise AttachmentError("DirectML candidate exceeds the uncompressed-size limit")
                entries[name] = info
            if not entries or any(
                not (name == "ProAim" or name.startswith("ProAim/"))
                for name in entries
            ):
                raise AttachmentError("DirectML candidate must contain one ProAim root")
            missing = REQUIRED_DIRECTML_BUNDLE_FILES.difference(entries)
            if missing:
                raise AttachmentError(
                    "DirectML candidate omitted required bundle files: "
                    + ", ".join(sorted(missing))
                )
            if archive.testzip() is not None:
                raise AttachmentError("DirectML candidate ZIP failed CRC validation")
            build_info = _strict_json_loads(
                archive.read(entries["ProAim/BUILD-INFO.json"]), "DirectML BUILD-INFO.json"
            )
            if not isinstance(build_info, dict):
                raise AttachmentError("DirectML BUILD-INFO.json must be an object")
            if (
                build_info.get("schema") != 2
                or build_info.get("application") != "ProAim"
                or build_info.get("runtime_variant") != "directml"
                or build_info.get("commit") != _require_sha(expected_commit, "tag commit")
                or build_info.get("dirty") is not False
            ):
                raise AttachmentError("DirectML BUILD-INFO does not identify the exact clean tag candidate")
            release_default = _validate_release_default_model_record(
                build_info, archive=archive, entries=entries
            )
            dependency_record = build_info.get("dependency_manifest")
            if not isinstance(dependency_record, dict) or dependency_record.get("path") != "DEPENDENCY-MANIFEST.json":
                raise AttachmentError("DirectML BUILD-INFO omitted the adjacent dependency manifest")
            if dependency_record.get("lock_profile") != "windows-directml-py313":
                raise AttachmentError("DirectML candidate did not use the Windows DirectML release lock")
            dependency_payload = archive.read(entries["ProAim/DEPENDENCY-MANIFEST.json"])
            dependency_sha = sha256_bytes(dependency_payload)
            if dependency_record.get("sha256") != dependency_sha:
                raise AttachmentError("DirectML dependency manifest hash differs from BUILD-INFO")
            dependency = _strict_json_loads(dependency_payload, "DirectML dependency manifest")
            distributions = dependency.get("distributions") if isinstance(dependency, dict) else None
            if (
                not isinstance(dependency, dict)
                or dependency.get("schema_version") != 1
                or dependency.get("application") != "ProAim"
                or dependency.get("runtime_variant") != "directml"
                or dependency.get("lock_profile") != "windows-directml-py313"
                or not isinstance(dependency.get("artifact_hash_contract"), dict)
                or dependency["artifact_hash_contract"].get("enforced_before_install") is not True
                or not isinstance(distributions, list)
                or dependency_record.get("distribution_count") != len(distributions)
            ):
                raise AttachmentError("DirectML dependency manifest does not identify the exact hash lock")
            _validate_dependency_distribution_records(distributions)
            extraction_root.mkdir(parents=True)
            root = extraction_root.resolve()
            for name, info in entries.items():
                destination = extraction_root.joinpath(*PurePosixPath(name).parts)
                resolved = destination.resolve()
                if resolved != root and root not in resolved.parents:
                    raise AttachmentError("DirectML archive member escaped the extraction root")
                if info.is_dir() or info.filename.endswith("/"):
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, destination.open("xb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
    except zipfile.BadZipFile as exc:
        raise AttachmentError("DirectML candidate is not a valid ZIP") from exc
    bundle = extraction_root / "ProAim"
    paths = {
        "build_info": bundle / "BUILD-INFO.json",
        "dependency_manifest": bundle / "DEPENDENCY-MANIFEST.json",
        "frozen_cli": bundle / "ProAimCLI.exe",
        "qualification_helper": bundle / "Qualify-ProAimGpu.ps1",
        "release_default_model": bundle.joinpath(*PurePosixPath(release_default["model_path"]).parts),
        "release_default_labels": bundle.joinpath(*PurePosixPath(release_default["labels_path"]).parts),
    }
    if any(not path.is_file() or path.stat().st_size <= 0 for path in paths.values()):
        raise AttachmentError("DirectML candidate omitted a required non-empty bundle member")
    return {
        "filename": DIRECTML_ARCHIVE_NAME,
        "sha256": expected_sha256,
        "size_bytes": archive_path.stat().st_size,
        "build_info": build_info,
        "build_info_sha256": sha256_file(paths["build_info"]),
        "dependency_manifest_sha256": sha256_file(paths["dependency_manifest"]),
        "frozen_cli_sha256": sha256_file(paths["frozen_cli"]),
        "qualification_helper_sha256": sha256_file(paths["qualification_helper"]),
        "release_default_model": release_default,
    }


def inspect_candidate(
    directory: Path,
    *,
    repository: str,
    tag: str,
    tag_commit: str,
    source_run_id: int,
    candidate_manifest_sha256: str,
    directml_zip_sha256: str,
    extraction_root: Path,
) -> dict[str, Any]:
    files = _downloaded_candidate_files(directory)
    context = candidate_context(repository, tag, tag_commit, source_run_id)
    manifest = validate_content_manifest(
        root=directory,
        manifest_name=CANDIDATE_MANIFEST_NAME,
        expected_sha256=candidate_manifest_sha256,
        expected_kind=CANDIDATE_MANIFEST_KIND,
        expected_context=context,
    )
    archive_records = _records_by_key(manifest.get("files"), key="path", description="candidate archives")
    if set(archive_records) != EXPECTED_ARCHIVES:
        raise AttachmentError("candidate manifest does not bind exactly both release archives")
    candidate = _extract_directml_bundle(
        files[DIRECTML_ARCHIVE_NAME],
        expected_sha256=directml_zip_sha256,
        expected_commit=tag_commit,
        extraction_root=extraction_root,
    )
    candidate["candidate_manifest_sha256"] = _normalize_sha(
        candidate_manifest_sha256, "candidate manifest SHA-256"
    )
    candidate["linux_archive"] = {
        "filename": LINUX_ARCHIVE_NAME,
        "sha256": archive_records[LINUX_ARCHIVE_NAME]["sha256"],
        "size_bytes": archive_records[LINUX_ARCHIVE_NAME]["size_bytes"],
    }
    candidate["source_identity"] = context
    return candidate


def raw_content_context(
    *,
    repository: str,
    tag: str,
    tag_commit: str,
    source_run_id: int,
    qualification_run_id: int,
    qualification_run_attempt: int,
    role: str,
    adapter_index: int,
    candidate_manifest_sha256: str,
    directml_zip_sha256: str,
) -> dict[str, Any]:
    _validate_identity(repository, tag, role)
    if adapter_index < 0:
        raise AttachmentError("DirectML adapter index must be non-negative")
    return {
        "repository": repository,
        "tag": tag,
        "tag_commit": _require_sha(tag_commit, "raw evidence tag commit"),
        "source_build_run_id": int(source_run_id),
        "qualification_run_id": int(qualification_run_id),
        "qualification_run_attempt": int(qualification_run_attempt),
        "gpu_role": role,
        "qualified_product": GPU_PRODUCTS[role]["product_name"],
        "directml_adapter_index": int(adapter_index),
        "candidate_manifest_sha256": _normalize_sha(
            candidate_manifest_sha256, "candidate manifest SHA-256"
        ),
        "directml_zip_sha256": _normalize_sha(
            directml_zip_sha256, "DirectML ZIP SHA-256"
        ),
    }


def _validate_recorded_file(
    record: Mapping[str, Any], files: Mapping[str, Path], description: str
) -> None:
    filename = record.get("file")
    if not isinstance(filename, str) or filename not in files:
        raise AttachmentError(f"{description} references a missing file")
    if record.get("sha256") != sha256_file(files[filename]):
        raise AttachmentError(f"{description} SHA-256 mismatch")
    if "size_bytes" in record and record.get("size_bytes") != files[filename].stat().st_size:
        raise AttachmentError(f"{description} size mismatch")


def _validate_directml_provider(summary: Any, *, device: str, adapter_index: int, description: str) -> None:
    if not isinstance(summary, dict):
        raise AttachmentError(f"{description} omitted its provider summary")
    if summary.get("requested_provider") != "DmlExecutionProvider":
        raise AttachmentError(f"{description} did not request DmlExecutionProvider")
    if summary.get("requested_device_input") != device:
        raise AttachmentError(f"{description} recorded the wrong requested DirectML device")
    active = summary.get("active_providers")
    if not isinstance(active, list) or "DmlExecutionProvider" not in active:
        raise AttachmentError(f"{description} did not activate DmlExecutionProvider")
    if summary.get("require_full_provider") is not True:
        raise AttachmentError(f"{description} omitted require_full_provider=true")
    configured = summary.get("configured_session_options")
    if not isinstance(configured, dict) or configured.get("disable_cpu_ep_fallback") is not True:
        raise AttachmentError(f"{description} did not disable CPU graph-node fallback")
    if summary.get("runtime_ep_fail_fallback_disabled") is not True:
        raise AttachmentError(f"{description} did not disable EPFail runtime fallback")
    if summary.get("provider_options_status") != "ok":
        raise AttachmentError(f"{description} could not report provider options")
    overrides = summary.get("provider_option_overrides")
    dml = overrides.get("DmlExecutionProvider") if isinstance(overrides, dict) else None
    if not isinstance(dml, dict) or str(dml.get("device_id")) != str(adapter_index):
        raise AttachmentError(f"{description} did not bind the exact DirectML adapter index")


def _validate_dxgi_descriptor(
    report: Mapping[str, Any], *, role: str, adapter_index: int, description: str
) -> dict[str, Any]:
    adapter = report.get("directml_adapter")
    if not isinstance(adapter, dict):
        raise AttachmentError(f"{description} omitted its DirectML adapter record")
    if (
        adapter.get("requested_index") != adapter_index
        or adapter.get("configured_index") != adapter_index
        or adapter.get("effective_index") != adapter_index
        or adapter.get("requested_provider_mismatch") is not False
        or adapter.get("enumeration_status") != "matched_dxgi_adapter"
        or adapter.get("task_manager_confirmation_required") is not True
    ):
        raise AttachmentError(f"{description} did not bind the exact requested DXGI adapter")
    descriptor = adapter.get("descriptor")
    if not isinstance(descriptor, dict) or descriptor.get("index") != adapter_index:
        raise AttachmentError(f"{description} omitted the exact DXGI descriptor")
    expected = GPU_PRODUCTS[role]
    if normalize_product_name(str(descriptor.get("name") or "")) != normalize_product_name(
        expected["product_name"]
    ):
        raise AttachmentError(f"{description} DXGI product name differs from {expected['product_name']}")
    vendor = str(descriptor.get("vendor_id") or "").lower().removeprefix("0x")
    if vendor != expected["vendor_id"].removeprefix("0x"):
        raise AttachmentError(f"{description} DXGI vendor ID is wrong for {role}")
    device_id = str(descriptor.get("device_id") or "").lower().removeprefix("0x")
    if not re.fullmatch(r"[0-9a-f]{4,8}", device_id):
        raise AttachmentError(f"{description} DXGI device ID is invalid")
    dedicated = descriptor.get("dedicated_vram_bytes")
    if not isinstance(dedicated, int) or isinstance(dedicated, bool) or dedicated <= 0:
        raise AttachmentError(f"{description} DXGI dedicated VRAM evidence is invalid")
    return dict(descriptor)


def _validate_run_intervals(manifest: Mapping[str, Any], files: Mapping[str, Path]) -> list[tuple[str, datetime, datetime]]:
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 4:
        raise AttachmentError("DirectML qualification must contain runtime info, benchmark, and two live runs")
    expected_names = [
        "frozen runtime info",
        "model benchmark (release-default)",
        "live pipeline (release-default-no-preview)",
        "live pipeline (release-default-preview-15)",
    ]
    intervals: list[tuple[str, datetime, datetime]] = []
    prior: datetime | None = None
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or run.get("name") != expected_names[index] or run.get("exit_code") != 0:
            raise AttachmentError("DirectML qualification run identity or exit status is invalid")
        for field in ("stdout", "stderr"):
            record = run.get(field)
            if not isinstance(record, dict):
                raise AttachmentError(f"DirectML run omitted {field} evidence")
            _validate_recorded_file(record, files, f"DirectML run {field}")
        if index >= 2:
            record = run.get("metrics")
            if not isinstance(record, dict):
                raise AttachmentError("DirectML live run omitted metrics evidence")
            _validate_recorded_file(record, files, "DirectML live metrics")
        started = _utc_datetime(run.get("started_at_utc"), "DirectML run start")
        completed = _utc_datetime(run.get("completed_at_utc"), "DirectML run completion")
        if completed < started or (prior is not None and started < prior):
            raise AttachmentError("DirectML run intervals overlap or are out of order")
        prior = completed
        intervals.append((expected_names[index], started, completed))
    return intervals


def _validate_directml_software_evidence(
    software_root: Path,
    *,
    candidate: Mapping[str, Any],
    tag_commit: str,
    role: str,
    adapter_index: int,
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, datetime, datetime]], dict[str, Any]]:
    if not software_root.is_dir() or any(path.is_symlink() for path in software_root.rglob("*")):
        raise AttachmentError("software evidence directory is missing or contains a symbolic link")
    files = {path.name: path for path in software_root.iterdir() if path.is_file()}
    if len(files) != len([path for path in software_root.iterdir() if path.is_file()]):
        raise AttachmentError("software evidence contains duplicate file names")
    manifest_path = files.get(QUALIFICATION_MANIFEST_NAME)
    if manifest_path is None:
        raise AttachmentError("software evidence omitted qualification-manifest.json")
    expected_file_set = SOFTWARE_EVIDENCE_FILES | {
        QUALIFICATION_MANIFEST_NAME,
        "TASK-MANAGER-CONFIRMATION.txt",
    }
    if set(files) != expected_file_set:
        raise AttachmentError("software evidence has an unexpected file set")
    manifest = _read_json_object(manifest_path, "DirectML qualification manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "software_checks_passed_physical_gpu_confirmation_pending"
        or manifest.get("qualified") is not False
    ):
        raise AttachmentError("DirectML helper manifest status or schema is invalid")
    device = f"DIRECTML:{adapter_index}"
    if manifest.get("provider") != {
        "selection": "DirectML",
        "requested_device": device,
        "expected_execution_provider": "DmlExecutionProvider",
        "directml_adapter_index": adapter_index,
    }:
        raise AttachmentError("qualification manifest is not the exact requested DirectML run")
    build_info = manifest.get("bundle_build_info")
    if not isinstance(build_info, dict) or build_info != candidate.get("build_info"):
        raise AttachmentError("qualification BUILD-INFO differs from the staged DirectML candidate")
    if build_info.get("commit") != tag_commit:
        raise AttachmentError("qualification BUILD-INFO differs from the tag commit")
    release_default = _validate_release_default_model_record(build_info)
    if release_default != candidate.get("release_default_model"):
        raise AttachmentError("qualification selected a different release-default contract")
    if _read_json_object(files["bundle-BUILD-INFO.json"], "evidence BUILD-INFO") != build_info:
        raise AttachmentError("copied BUILD-INFO differs from the qualification manifest")
    if sha256_file(files["bundle-BUILD-INFO.json"]) != candidate.get("build_info_sha256"):
        raise AttachmentError("copied BUILD-INFO hash differs from the staged candidate")
    if sha256_file(files["bundle-DEPENDENCY-MANIFEST.json"]) != candidate.get("dependency_manifest_sha256"):
        raise AttachmentError("copied dependency-manifest hash differs from the staged candidate")

    confirmation = manifest.get("manual_confirmation")
    if not isinstance(confirmation, dict) or confirmation.get("required") is not True or confirmation.get("completed_by_helper") is not False:
        raise AttachmentError("qualification helper improperly claimed physical completion")
    bounds = manifest.get("benchmark_bounds")
    expected_bounds = {
        "samples": DIRECTML_POLICY["benchmark_samples"],
        "warmup": DIRECTML_POLICY["benchmark_warmup"],
        "iterations": DIRECTML_POLICY["benchmark_iterations_per_repeat"],
        "repeats": DIRECTML_POLICY["benchmark_repeats"],
    }
    if bounds != expected_bounds:
        raise AttachmentError("DirectML benchmark bounds differ from release policy")
    live_bounds = manifest.get("live_bounds")
    detail_width = release_default["detail_crop_size_source_pixels"]
    expected_detail_width = detail_width if detail_width > 0 else None
    if (
        not isinstance(live_bounds, dict)
        or live_bounds.get("enabled") is not True
        or live_bounds.get("selected_model") != "release-default"
        or live_bounds.get("release_default_model") != release_default
        or live_bounds.get("detail_crop_size") != expected_detail_width
        or live_bounds.get("max_frames") != DIRECTML_POLICY["live_requested_max_frames"]
        or live_bounds.get("max_seconds") != DIRECTML_POLICY["live_max_seconds"]
        or set(live_bounds.get("modes") or ()) != {"no-preview", "preview-15"}
    ):
        raise AttachmentError("DirectML live bounds differ from the release-default policy")

    artifacts = _records_by_key(manifest.get("input_artifacts"), key="role", description="qualification input artifacts")
    required_artifacts = {
        "frozen_cli": candidate.get("frozen_cli_sha256"),
        "build_info": candidate.get("build_info_sha256"),
        "dependency_manifest": candidate.get("dependency_manifest_sha256"),
        "qualification_helper": candidate.get("qualification_helper_sha256"),
        "original_bundle_archive": candidate.get("sha256"),
        "release_default_model": release_default["model_sha256"],
        "release_default_labels": release_default["labels_sha256"],
    }
    if not set(required_artifacts).issubset(artifacts) or set(artifacts).difference(required_artifacts) not in (set(), {"frozen_gui"}):
        raise AttachmentError("qualification input artifact roles are incomplete or unexpected")
    for role_name, expected_sha in required_artifacts.items():
        if artifacts[role_name].get("sha256") != expected_sha:
            raise AttachmentError(f"qualification input artifact {role_name!r} differs from candidate")
    if artifacts["release_default_model"].get("path") != release_default["model_path"]:
        raise AttachmentError("qualification model path differs from BUILD-INFO")
    if artifacts["release_default_labels"].get("path") != release_default["labels_path"]:
        raise AttachmentError("qualification labels path differs from BUILD-INFO")
    if (
        artifacts["release_default_model"].get("location") != "bundle"
        or artifacts["release_default_labels"].get("location") != "bundle"
    ):
        raise AttachmentError("qualification release-default artifacts were not loaded from the bundle")

    evidence_records = _records_by_key(manifest.get("evidence_files"), key="file", description="software evidence files")
    if set(evidence_records) != SOFTWARE_EVIDENCE_FILES:
        raise AttachmentError("qualification manifest did not bind the exact software evidence set")
    for filename, record in evidence_records.items():
        _validate_recorded_file(record, files, f"software evidence {filename}")
    intervals = _validate_run_intervals(manifest, files)
    intervals_by_name = {name: (start, end) for name, start, end in intervals}
    runs = manifest["runs"]
    expected_adapter_arg = GPU_PRODUCTS[role]["product_name"]
    for run in runs[1:]:
        arguments = run.get("arguments")
        if not isinstance(arguments, list):
            raise AttachmentError("accelerated run omitted its exact arguments")
        pairs = list(zip(arguments, arguments[1:]))
        if ("--device", device) not in pairs or "--require-full-provider" not in arguments:
            raise AttachmentError("accelerated run did not request the exact full DirectML provider")
    # ExpectedAdapterName is a helper parameter rather than a CLI argument; its
    # effect is proved below by both live DXGI descriptors.
    _ = expected_adapter_arg

    runtime = _read_json_object(files["runtime-info.json"], "frozen DirectML runtime report")
    if runtime.get("frozen") is not True or "DmlExecutionProvider" not in (runtime.get("onnxruntime_providers") or []):
        raise AttachmentError("frozen runtime report omitted DmlExecutionProvider")
    benchmark = _read_json_object(files["benchmark-release-default.json"], "DirectML model benchmark")
    methodology = benchmark.get("methodology")
    models = benchmark.get("models")
    benchmark_input = benchmark.get("input")
    if (
        not isinstance(methodology, dict)
        or methodology.get("backend") != "onnxruntime"
        or methodology.get("requested_device") != device
        or methodology.get("require_full_provider") is not True
        or methodology.get("warmup_per_model") != expected_bounds["warmup"]
        or methodology.get("iterations_per_repeat") != expected_bounds["iterations"]
        or methodology.get("repeats") != expected_bounds["repeats"]
        or not isinstance(models, list)
        or len(models) != 1
        or not isinstance(models[0], dict)
        or not isinstance(benchmark_input, dict)
        or benchmark_input.get("kind") != "synthetic"
        or benchmark_input.get("generator")
        != "numpy.default_rng(seed=0), uint8 720x1280 BGR"
        or benchmark_input.get("count") != expected_bounds["samples"]
    ):
        raise AttachmentError("DirectML benchmark methodology differs from release policy")
    generated = _utc_datetime(benchmark.get("generated_at_utc"), "DirectML benchmark generation")
    bench_start, bench_end = intervals_by_name["model benchmark (release-default)"]
    if not bench_start <= generated <= bench_end:
        raise AttachmentError("DirectML benchmark timestamp lies outside its frozen run")
    model = models[0]
    if model.get("key") != "release-default" or model.get("input_shape_hw") != release_default["input_shape_hw"]:
        raise AttachmentError("DirectML benchmark did not use the dynamic release-default shape")
    _validate_directml_provider(model.get("runtime"), device=device, adapter_index=adapter_index, description="DirectML benchmark")
    for group, hash_key, path_key in (
        ("artifact", "model_sha256", "model_path"),
        ("labels_artifact", "labels_sha256", "labels_path"),
    ):
        value = model.get(group)
        members = value.get("files") if isinstance(value, dict) else None
        if not isinstance(members, list) or len(members) != 1 or members[0].get("sha256") != release_default[hash_key]:
            raise AttachmentError(f"DirectML benchmark {group} differs from BUILD-INFO")
        _require_bundle_path_suffix(members[0].get("resolved_path"), release_default[path_key], f"DirectML benchmark {group} path")
    expected_timed_samples = expected_bounds["iterations"] * expected_bounds["repeats"]
    benchmark_payload = model.get("timing_ms")
    if not isinstance(benchmark_payload, dict) or set(benchmark_payload) != {
        "preprocess",
        "inference",
        "postprocess",
        "pipeline",
    }:
        raise AttachmentError("DirectML benchmark timing fields are incomplete")
    benchmark_timings = {
        name: _validate_timing_summary(
            benchmark_payload[name],
            f"DirectML benchmark {name}",
            expected_samples=expected_timed_samples,
        )
        for name in ("preprocess", "inference", "postprocess", "pipeline")
    }
    _require_close_timing(
        float(benchmark_timings["pipeline"]["mean"]),
        sum(
            float(benchmark_timings[name]["mean"])
            for name in ("preprocess", "inference", "postprocess")
        ),
        "DirectML benchmark mean pipeline/component timing",
    )
    repeats = model.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != expected_bounds["repeats"]:
        raise AttachmentError("DirectML benchmark omitted repeat-level latency")
    normalized_repeats: list[dict[str, dict[str, float | int]]] = []
    for repeat_number, repeat in enumerate(repeats, 1):
        if not isinstance(repeat, dict) or repeat.get("repeat") != repeat_number:
            raise AttachmentError("DirectML benchmark repeat identities are invalid")
        timing = repeat.get("timing_ms")
        if not isinstance(timing, dict) or set(timing) != {
            "preprocess",
            "inference",
            "postprocess",
            "pipeline",
        }:
            raise AttachmentError("DirectML benchmark repeat timing fields are incomplete")
        normalized = {
            name: _validate_timing_summary(
                timing[name],
                f"DirectML benchmark repeat {repeat_number} {name}",
                expected_samples=expected_bounds["iterations"],
            )
            for name in ("preprocess", "inference", "postprocess", "pipeline")
        }
        _require_close_timing(
            float(normalized["pipeline"]["mean"]),
            sum(
                float(normalized[name]["mean"])
                for name in ("preprocess", "inference", "postprocess")
            ),
            f"DirectML benchmark repeat {repeat_number} mean pipeline/component timing",
        )
        if float(normalized["inference"]["p95"]) > DIRECTML_POLICY["benchmark_max_p95_inference_ms"]:
            raise AttachmentError(f"DirectML benchmark repeat {repeat_number} p95 inference exceeds policy")
        normalized_repeats.append(normalized)
    for name in ("preprocess", "inference", "postprocess", "pipeline"):
        aggregate = benchmark_timings[name]
        _require_close_timing(
            float(aggregate["mean"]),
            sum(float(repeat[name]["mean"]) for repeat in normalized_repeats)
            / expected_bounds["repeats"],
            f"DirectML benchmark aggregate/repeat {name} mean",
            absolute_tolerance_ms=1e-9,
        )
        if not math.isclose(
            float(aggregate["min"]),
            min(float(repeat[name]["min"]) for repeat in normalized_repeats),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ) or not math.isclose(
            float(aggregate["max"]),
            max(float(repeat[name]["max"]) for repeat in normalized_repeats),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise AttachmentError(f"DirectML benchmark aggregate/repeat {name} extrema differ")
    pipeline_fps = _finite_number(
        model.get("pipeline_fps_from_mean"), "DirectML benchmark pipeline FPS"
    )
    _require_close_timing(
        pipeline_fps,
        1000.0 / float(benchmark_timings["pipeline"]["mean"]),
        "DirectML benchmark pipeline FPS",
        absolute_tolerance_ms=1e-6,
    )
    inference_summary = benchmark_timings["inference"]
    if float(inference_summary["p95"]) > DIRECTML_POLICY["benchmark_max_p95_inference_ms"]:
        raise AttachmentError("DirectML benchmark aggregate p95 inference exceeds policy")

    live_metrics: dict[str, Any] = {}
    descriptor: dict[str, Any] | None = None
    for filename, preview_enabled in (
        ("live-release-default-no-preview.json", False),
        ("live-release-default-preview-15.json", True),
    ):
        report = _read_json_object(files[filename], f"DirectML live report {filename}")
        run_name = "live pipeline (release-default-preview-15)" if preview_enabled else "live pipeline (release-default-no-preview)"
        run_start, run_end = intervals_by_name[run_name]
        report_start = _utc_datetime(report.get("started_utc"), f"{filename} start")
        report_end = _utc_datetime(report.get("completed_utc"), f"{filename} completion")
        if report_end < report_start or not run_start <= report_start <= report_end <= run_end:
            raise AttachmentError(f"{filename} timestamps lie outside its frozen run")
        _validate_directml_provider(report.get("detector_runtime"), device=device, adapter_index=adapter_index, description=filename)
        current_descriptor = _validate_dxgi_descriptor(report, role=role, adapter_index=adapter_index, description=filename)
        if descriptor is not None and current_descriptor != descriptor:
            raise AttachmentError("DirectML DXGI descriptor changed between bounded live runs")
        descriptor = current_descriptor
        config = report.get("config")
        source = report.get("source")
        capture = report.get("capture")
        pipeline = report.get("pipeline")
        termination = report.get("termination")
        preview = report.get("preview")
        config_preview = config.get("preview") if isinstance(config, dict) else None
        if (
            not isinstance(config, dict)
            or config.get("backend") != "onnxruntime"
            or config.get("device") != device
            or config.get("require_full_provider") is not True
            or not isinstance(config.get("source"), dict)
            or config["source"].get("kind") != "screen"
            or config["source"].get("value") is not None
            or not isinstance(config.get("capture"), dict)
            or config["capture"].get("screen_region") is not None
            or config["capture"].get("screen_monitor") != live_bounds.get("screen_monitor")
            or config["capture"].get("screen_fps") != live_bounds.get("screen_fps")
            or not isinstance(config.get("inference"), dict)
            or config["inference"].get("shape_hw") != release_default["input_shape_hw"]
            or config["inference"].get("crop_size") is not None
            or config["inference"].get("detail_crop_size") != expected_detail_width
            or config.get("stats_window") != DIRECTML_POLICY["live_requested_max_frames"]
            or not isinstance(config_preview, dict)
            or config_preview.get("enabled") is not preview_enabled
            or (preview_enabled and config_preview.get("fps_limit") != 15.0)
            or not isinstance(source, dict)
            or source.get("backend") != "dxcam-dxgi"
            or source.get("fallback_reason") not in (None, "")
            or not isinstance(capture, dict)
            or capture.get("read_failures") != 0
        ):
            raise AttachmentError(f"{filename} used a different full-screen workload")
        _require_bundle_path_suffix(
            config.get("model_path"),
            release_default["model_path"],
            f"{filename} configured model path",
        )
        _require_bundle_path_suffix(
            config.get("labels_path"),
            release_default["labels_path"],
            f"{filename} configured labels path",
        )
        model_artifact = report.get("model_artifact")
        if not isinstance(model_artifact, dict) or model_artifact.get("sha256") != release_default["model_sha256"]:
            raise AttachmentError(f"{filename} used a different model")
        labels_artifact = report.get("labels_artifact")
        if not isinstance(labels_artifact, dict) or labels_artifact.get("sha256") != release_default["labels_sha256"]:
            raise AttachmentError(f"{filename} used different labels")
        detail = report.get("detail_pass")
        if not isinstance(detail, dict):
            raise AttachmentError(f"{filename} omitted detail-pass evidence")
        if detail.get("enabled") is not (detail_width > 0) or detail.get(
            "requested_crop_size"
        ) != expected_detail_width:
            raise AttachmentError(
                f"{filename} detail-pass setting differs from BUILD-INFO"
            )
        if detail_width > 0:
            plan = detail.get("last_plan")
            if (
                detail.get("crop_policy") != "centered_model_aspect_roi"
                or not isinstance(plan, dict)
                or plan.get("crop_policy") != "centered_model_aspect_roi"
            ):
                raise AttachmentError(
                    f"{filename} omitted the centered model-aspect detail ROI"
                )
            source_width = plan.get("source_width")
            source_height = plan.get("source_height")
            model_width = plan.get("model_width")
            model_height = plan.get("model_height")
            if (
                not all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                    for value in (
                        source_width,
                        source_height,
                        model_width,
                        model_height,
                    )
                )
                or [model_height, model_width]
                != release_default["input_shape_hw"]
                or plan.get("requested_crop_size") != detail_width
            ):
                raise AttachmentError(
                    f"{filename} detail ROI source/model geometry is invalid"
                )
            expected_requested_height = max(
                1, round(detail_width * model_height / float(model_width))
            )
            common = gcd(model_width, model_height)
            aspect_width = model_width // common
            aspect_height = model_height // common
            aspect_units = min(
                detail_width // aspect_width,
                source_width // aspect_width,
                source_height // aspect_height,
            )
            if aspect_units > 0:
                expected_width = aspect_units * aspect_width
                expected_height = aspect_units * aspect_height
            else:
                expected_width = min(detail_width, source_width)
                expected_height = min(
                    source_height,
                    max(
                        1,
                        round(
                            expected_width
                            * model_height
                            / float(model_width)
                        ),
                    ),
                )
            expected_crop_x = (source_width - expected_width) // 2
            expected_crop_y = (source_height - expected_height) // 2
            expected_full_scale = min(
                model_width / float(source_width),
                model_height / float(source_height),
            )
            expected_detail_scale = min(
                model_width / float(expected_width),
                model_height / float(expected_height),
            )
            expected_magnification = expected_detail_scale / expected_full_scale
            exact_geometry = {
                "requested_crop_height": expected_requested_height,
                "applied_crop_width": expected_width,
                "applied_crop_height": expected_height,
                "crop_x": expected_crop_x,
                "crop_y": expected_crop_y,
                "clamped": (
                    expected_width != detail_width
                    or expected_height != expected_requested_height
                ),
                "redundant": (
                    expected_width == source_width
                    and expected_height == source_height
                ),
            }
            for key, value in exact_geometry.items():
                if plan.get(key) != value:
                    raise AttachmentError(
                        f"{filename} detail ROI {key} differs from exact model-aspect geometry"
                    )
            for key, value in (
                (
                    "coverage_fraction",
                    expected_width
                    * expected_height
                    / float(source_width * source_height),
                ),
                ("full_frame_scale", expected_full_scale),
                ("detail_scale", expected_detail_scale),
                (
                    "effective_linear_magnification",
                    expected_magnification,
                ),
            ):
                actual = _finite_number(plan.get(key), f"{filename} detail ROI {key}")
                if not math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-9):
                    raise AttachmentError(
                        f"{filename} detail ROI {key} is internally inconsistent"
                    )
            if (
                (source_height, source_width) == (1080, 1920)
                and expected_width == 765
                and [model_height, model_width] == [384, 640]
            ):
                if expected_height != 459 or not math.isclose(
                    expected_magnification, 2.5, rel_tol=0.01, abs_tol=0.01
                ):
                    raise AttachmentError(
                        f"{filename} did not preserve the documented 765x459 / ~2.5x detail geometry"
                    )
        elif detail.get("last_plan") is not None:
            raise AttachmentError(
                f"{filename} recorded a detail ROI while BUILD-INFO disables it"
            )
        if not isinstance(preview, dict) or preview.get("enabled") is not preview_enabled:
            raise AttachmentError(f"{filename} recorded the wrong preview state")
        if preview_enabled:
            stats = preview.get("stats")
            if preview.get("fps_limit") != 15.0 or preview.get("mode") == "disabled" or not isinstance(stats, dict):
                raise AttachmentError(f"{filename} did not exercise preview at 15 FPS")
            submitted, displayed, replaced = (stats.get(key) for key in ("submitted_frames", "displayed_frames", "replaced_frames"))
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (submitted, displayed, replaced)) or not 0 < displayed <= submitted:
                raise AttachmentError(f"{filename} preview counters are invalid")
        elif preview.get("mode") != "disabled" or preview.get("stats") != {}:
            raise AttachmentError(f"{filename} performed unexpected preview work")
        if not isinstance(pipeline, dict) or not isinstance(termination, dict):
            raise AttachmentError(f"{filename} omitted pipeline or termination evidence")
        processed = pipeline.get("processed_frames")
        rolling = pipeline.get("rolling_sample_count")
        elapsed = _finite_number(pipeline.get("elapsed_seconds"), f"{filename} elapsed seconds")
        elapsed_fps = _finite_number(pipeline.get("elapsed_fps"), f"{filename} elapsed FPS")
        update_fps = _finite_number(pipeline.get("update_fps"), f"{filename} update FPS")
        if (
            not isinstance(processed, int)
            or isinstance(processed, bool)
            or processed < DIRECTML_POLICY["live_min_processed_frames"]
            or not isinstance(rolling, int)
            or isinstance(rolling, bool)
            or rolling < DIRECTML_POLICY["live_min_processed_frames"]
            or elapsed < DIRECTML_POLICY["live_min_elapsed_seconds"]
            or elapsed_fps < DIRECTML_POLICY["live_min_elapsed_fps"]
            or update_fps < DIRECTML_POLICY["live_min_update_fps"]
        ):
            raise AttachmentError(f"{filename} throughput is below DirectML release policy")
        if detail_width > 0:
            detail_counts = {
                key: detail.get(key)
                for key in (
                    "frames_seen",
                    "frames_applied",
                    "frames_redundant",
                    "frames_clamped",
                )
            }
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in detail_counts.values()
            ):
                raise AttachmentError(f"{filename} detail-pass counters are invalid")
            expected_applied = 0 if plan["redundant"] else processed
            expected_redundant = processed if plan["redundant"] else 0
            expected_clamped = processed if plan["clamped"] else 0
            if detail_counts != {
                "frames_seen": processed,
                "frames_applied": expected_applied,
                "frames_redundant": expected_redundant,
                "frames_clamped": expected_clamped,
            }:
                raise AttachmentError(f"{filename} detail-pass counters differ from its ROI plan")
        if rolling != min(processed, config["stats_window"]):
            raise AttachmentError(f"{filename} rolling sample count is internally inconsistent")
        if preview_enabled and submitted > processed:
            raise AttachmentError(f"{filename} preview counters exceed processed frames")
        report_duration = (report_end - report_start).total_seconds()
        if abs(report_duration - elapsed) > 2.0:
            raise AttachmentError(f"{filename} elapsed duration is internally inconsistent")
        if not math.isclose(processed / elapsed, elapsed_fps, rel_tol=0.02, abs_tol=0.5):
            raise AttachmentError(f"{filename} elapsed FPS is internally inconsistent")
        timings = pipeline.get("timings")
        if not isinstance(timings, dict) or timings.get("unit") != "milliseconds" or timings.get("fields") != list(LIVE_TIMING_FIELDS):
            raise AttachmentError(f"{filename} has the wrong latency schema")
        normalized: dict[str, dict[str, float]] = {}
        for percentile in ("mean", "p50", "p95", "p99"):
            values = timings.get(percentile)
            if not isinstance(values, dict) or set(values) != set(LIVE_TIMING_FIELDS):
                raise AttachmentError(f"{filename} {percentile} latency fields are incomplete")
            normalized[percentile] = {key: _finite_number(values[key], f"{filename} {percentile} {key}") for key in LIVE_TIMING_FIELDS}
        for field in LIVE_TIMING_FIELDS:
            if not (
                normalized["p50"][field]
                <= normalized["p95"][field]
                <= normalized["p99"][field]
            ):
                raise AttachmentError(f"{filename} timing percentile ordering is inconsistent")
        mean_timing = normalized["mean"]
        _require_close_timing(
            mean_timing["processing_ms"],
            sum(
                mean_timing[field]
                for field in (
                    "preprocess_ms",
                    "inference_ms",
                    "postprocess_ms",
                    "detail_preprocess_ms",
                    "detail_inference_ms",
                    "detail_postprocess_ms",
                    "control_ms",
                )
            ),
            f"{filename} mean processing/component timing",
        )
        _require_close_timing(
            mean_timing["freshness_latency_ms"],
            mean_timing["queue_age_ms"] + mean_timing["processing_ms"],
            f"{filename} mean freshness timing",
        )
        _require_close_timing(
            mean_timing["observed_pipeline_ms"],
            mean_timing["capture_ms"] + mean_timing["freshness_latency_ms"],
            f"{filename} mean observed-pipeline timing",
        )
        if normalized["p95"]["observed_pipeline_ms"] > DIRECTML_POLICY["live_max_p95_observed_pipeline_ms"]:
            raise AttachmentError(f"{filename} p95 observed pipeline latency exceeds policy")
        if normalized["p95"]["freshness_latency_ms"] > DIRECTML_POLICY["live_max_p95_freshness_latency_ms"]:
            raise AttachmentError(f"{filename} p95 freshness latency exceeds policy")
        if preview_enabled and normalized["mean"]["preview_service_ms"] <= 0.0:
            raise AttachmentError(f"{filename} recorded no preview service work")
        if not preview_enabled and any(normalized[p]["preview_service_ms"] != 0.0 for p in normalized):
            raise AttachmentError(f"{filename} recorded preview work with preview disabled")
        detail_timing_fields = (
            "detail_preprocess_ms",
            "detail_inference_ms",
            "detail_postprocess_ms",
        )
        if detail_width > 0:
            if normalized["mean"]["detail_inference_ms"] <= 0.0:
                raise AttachmentError(
                    f"{filename} did not time the BUILD-INFO detail inference"
                )
        elif any(
            normalized[percentile][field] != 0.0
            for percentile in normalized
            for field in detail_timing_fields
        ):
            raise AttachmentError(
                f"{filename} recorded detail timings while BUILD-INFO disables detail"
            )
        if (
            termination.get("reason") not in {"max_frames", "max_seconds"}
            or termination.get("requested_max_frames") != DIRECTML_POLICY["live_requested_max_frames"]
            or termination.get("requested_max_seconds") != DIRECTML_POLICY["live_max_seconds"]
            or processed > DIRECTML_POLICY["live_requested_max_frames"]
            or elapsed > DIRECTML_POLICY["live_max_seconds"] + 2.0
        ):
            raise AttachmentError(f"{filename} was not bounded by release policy")
        if termination["reason"] == "max_frames" and processed != termination["requested_max_frames"]:
            raise AttachmentError(f"{filename} did not reach its claimed frame bound")
        if termination["reason"] == "max_seconds" and not math.isclose(
            elapsed,
            float(termination["requested_max_seconds"]),
            rel_tol=0.0,
            abs_tol=2.0,
        ):
            raise AttachmentError(f"{filename} did not reach its claimed time bound")
        live_metrics[filename] = {
            "processed_frames": processed,
            "elapsed_fps": elapsed_fps,
            "update_fps": update_fps,
            "p95_observed_pipeline_ms": normalized["p95"]["observed_pipeline_ms"],
            "p95_freshness_latency_ms": normalized["p95"]["freshness_latency_ms"],
            "preview_enabled": preview_enabled,
            "measurement_limit": "preview submission measured; physical display scanout is not measured",
        }
    if descriptor is None:
        raise AttachmentError("DirectML qualification produced no DXGI descriptor")
    metrics = {
        "policy": dict(DIRECTML_POLICY),
        "benchmark": {
            "timed_samples": expected_bounds["iterations"] * expected_bounds["repeats"],
            "p95_inference_ms": float(inference_summary["p95"]),
        },
        "live": live_metrics,
    }
    return manifest, metrics, intervals, descriptor


def _validate_runner_invariant(
    path: Path, *, role: str, adapter_index: int
) -> dict[str, Any]:
    record = _read_json_object(path, "DirectML runner invariant")
    expected = GPU_PRODUCTS[role]
    if (
        record.get("schema_version") != 1
        or record.get("status") != "passed_before_directml_runs"
        or record.get("gpu_role") != role
        or normalize_product_name(str(record.get("expected_product_name") or ""))
        != normalize_product_name(expected["product_name"])
        or record.get("directml_adapter_index") != adapter_index
        or record.get("preexisting_proaim_cli_count") != 0
        or record.get("telemetry_interval_milliseconds")
        != DIRECTML_POLICY["telemetry_interval_milliseconds"]
    ):
        raise AttachmentError("DirectML runner invariant differs from the requested physical gate")
    _utc_datetime(record.get("checked_at_utc"), "DirectML runner invariant timestamp")
    return record


def _parse_directml_telemetry(
    path: Path,
    *,
    role: str,
    adapter_index: int,
    descriptor: Mapping[str, Any],
    intervals: Sequence[tuple[str, datetime, datetime]],
) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AttachmentError("DirectML GPU Engine telemetry is not readable UTF-8") from exc
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = _strict_json_loads(line, f"DirectML telemetry line {number}")
        if not isinstance(value, dict):
            raise AttachmentError(f"DirectML telemetry line {number} is not an object")
        records.append(value)
    inventory = [record for record in records if record.get("kind") == "adapter_inventory"]
    engines = [record for record in records if record.get("kind") == "proaim_gpu_engine"]
    if len(inventory) != 1 or not engines:
        raise AttachmentError("DirectML telemetry omitted its single adapter inventory or ProAim engine samples")
    unknown = [record for record in records if record.get("kind") not in {"adapter_inventory", "proaim_gpu_engine"}]
    if unknown:
        raise AttachmentError("DirectML telemetry contains an unsupported record kind")
    selected = inventory[0]
    expected = GPU_PRODUCTS[role]
    luid = str(selected.get("adapter_luid") or "").lower()
    if (
        selected.get("schema_version") != 1
        or selected.get("gpu_role") != role
        or selected.get("directml_adapter_index") != adapter_index
        or normalize_product_name(str(selected.get("product_name") or ""))
        != normalize_product_name(expected["product_name"])
        or selected.get("exact_product_match_count") != 1
        or selected.get("telemetry_interval_milliseconds")
        != DIRECTML_POLICY["telemetry_interval_milliseconds"]
        or str(selected.get("vendor_id") or "").lower() != expected["vendor_id"]
        or not re.fullmatch(r"0x[0-9a-f]{8}_0x[0-9a-f]{8}", luid)
    ):
        raise AttachmentError("DirectML telemetry adapter inventory is not the exact requested product")
    telemetry_device = str(selected.get("device_id") or "").lower().removeprefix("0x")
    descriptor_device = str(descriptor.get("device_id") or "").lower().removeprefix("0x")
    if telemetry_device != descriptor_device:
        raise AttachmentError("DirectX telemetry device ID differs from the live DXGI descriptor")
    driver_version = _single_line(str(selected.get("driver_version") or ""), "DirectML driver version")
    inventory_timestamp = _utc_datetime(
        selected.get("captured_at_utc"), "DirectML adapter inventory timestamp"
    )
    parsed_engines: list[tuple[datetime, int, float, str]] = []
    previous: datetime | None = None
    for record in engines:
        timestamp = _utc_datetime(record.get("captured_at_utc"), "DirectML engine timestamp")
        if previous is not None and timestamp < previous:
            raise AttachmentError("DirectML telemetry timestamps are out of order")
        previous = timestamp
        process_id = record.get("pid")
        utilization = _finite_number(record.get("utilization_percent"), "DirectML engine utilization")
        engine_type = _single_line(str(record.get("engine_type") or ""), "DirectML engine type")
        if (
            record.get("schema_version") != 1
            or not isinstance(process_id, int)
            or isinstance(process_id, bool)
            or process_id <= 0
            or record.get("process_name") != "ProAimCLI.exe"
            or str(record.get("adapter_luid") or "").lower() != luid
            or utilization > 100.0
            or not isinstance(record.get("physical_adapter"), int)
            or isinstance(record.get("physical_adapter"), bool)
            or not isinstance(record.get("engine_index"), int)
            or isinstance(record.get("engine_index"), bool)
        ):
            raise AttachmentError("DirectML telemetry contains an invalid PID-correlated engine record")
        parsed_engines.append((timestamp, process_id, utilization, engine_type))

    required = [interval for interval in intervals if interval[0] != "frozen runtime info"]
    if (
        not required
        or inventory_timestamp > required[0][1]
        or required[0][1] - inventory_timestamp > timedelta(minutes=10)
    ):
        raise AttachmentError("DirectML adapter inventory was not captured immediately before the accelerated runs")
    per_run: dict[str, Any] = {}
    pid_sets: list[set[int]] = []
    for name, start, completed in required:
        samples = [record for record in parsed_engines if start <= record[0] <= completed]
        positive = [record for record in samples if record[2] > 0.0]
        pids = {record[1] for record in samples}
        capture_times = {record[0] for record in samples}
        if (
            len(samples) < DIRECTML_POLICY["telemetry_min_total_samples_per_run"]
            or len(capture_times)
            < DIRECTML_POLICY["telemetry_min_total_samples_per_run"]
            or len(positive) < DIRECTML_POLICY["telemetry_min_positive_samples_per_run"]
            or not pids
        ):
            raise AttachmentError(f"DirectML telemetry did not correlate positive ProAim GPU work during {name}")
        if any(pids & prior for prior in pid_sets):
            raise AttachmentError("a ProAimCLI PID was reused across distinct qualification runs")
        pid_sets.append(pids)
        per_run[name] = {
            "sample_count": len(samples),
            "distinct_capture_count": len(capture_times),
            "positive_sample_count": len(positive),
            "max_utilization_percent": max(record[2] for record in samples),
            "distinct_process_count": len(pids),
            "engine_types": sorted({record[3] for record in samples}),
        }
    return {
        "product_name": expected["product_name"],
        "directml_adapter_index": adapter_index,
        "vendor_id": expected["vendor_id"],
        "device_id": "0x" + telemetry_device,
        "driver_version": driver_version,
        "pid_correlated": True,
        "adapter_luid_correlated": True,
        "total_engine_samples": len(parsed_engines),
        "per_run": per_run,
    }


def _validate_observation(
    path: Path,
    *,
    repository: str,
    tag: str,
    role: str,
    adapter_index: int,
    qualification_run_id: int,
    observer_name: str,
    typed_confirmation: str,
    final_run_completed: datetime,
) -> dict[str, Any]:
    record = _read_json_object(path, "local DirectML physical observation")
    expected_product = GPU_PRODUCTS[role]["product_name"]
    if (
        record.get("schema_version") != 1
        or record.get("status") != "completed_after_automated_directml_runs"
        or record.get("completed") is not True
        or record.get("repository") != repository
        or record.get("tag") != tag
        or record.get("gpu_role") != role
        or normalize_product_name(str(record.get("physical_gpu_name") or ""))
        != normalize_product_name(expected_product)
        or record.get("directml_adapter_index") != adapter_index
        or record.get("observer_name") != observer_name
        or record.get("typed_confirmation") != typed_confirmation
        or str(record.get("github_run_id")) != str(qualification_run_id)
        or record.get("observations")
        != {
            "release_default_benchmark": True,
            "live_no_preview": True,
            "live_preview_15": True,
            "automated_luid_telemetry_agreed": True,
        }
    ):
        raise AttachmentError("local DirectML observation does not complete the requested physical gate")
    _single_line(str(record.get("github_actor") or ""), "qualification GitHub actor")
    _single_line(str(record.get("task_manager_gpu_engine") or ""), "Task Manager GPU engine")
    observed = _utc_datetime(record.get("observed_at_utc"), "DirectML observation timestamp")
    if observed < final_run_completed or observed - final_run_completed > timedelta(minutes=30):
        raise AttachmentError("DirectML observation was not completed promptly after the GPU runs")
    return record


def _validate_raw_tree(
    raw_root: Path,
    *,
    expected_manifest_sha256: str,
    context: Mapping[str, Any],
    sealed_additions: frozenset[str] = frozenset(),
) -> None:
    if not sealed_additions:
        validate_content_manifest(
            root=raw_root,
            manifest_name=RAW_CONTENT_MANIFEST_NAME,
            expected_sha256=expected_manifest_sha256,
            expected_kind=RAW_CONTENT_KIND,
            expected_context=context,
        )
    else:
        manifest_path = raw_root / RAW_CONTENT_MANIFEST_NAME
        if sha256_file(manifest_path) != _normalize_sha(
            expected_manifest_sha256, "raw content manifest SHA-256"
        ):
            raise AttachmentError("raw content manifest SHA-256 mismatch")
        manifest = _read_json_object(manifest_path, "raw content manifest")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != RAW_CONTENT_KIND
            or manifest.get("context") != dict(context)
            or manifest.get("manifest_file_excluded_from_records")
            != RAW_CONTENT_MANIFEST_NAME
        ):
            raise AttachmentError("raw content manifest identity mismatch")
        records = _records_by_key(
            manifest.get("files"), key="path", description="raw content files"
        )
        actual = {
            path.relative_to(raw_root).as_posix(): path
            for path in raw_root.rglob("*")
            if path.is_file()
            and path.relative_to(raw_root).as_posix()
            not in sealed_additions | {RAW_CONTENT_MANIFEST_NAME}
        }
        if set(actual) != set(records):
            raise AttachmentError("raw content manifest file set mismatch")
        for relative, path in actual.items():
            record = records[relative]
            if record.get("size_bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
                raise AttachmentError(f"raw content record mismatch for {relative}")
    required_roots = {
        RAW_CONTENT_MANIFEST_NAME,
        CANDIDATE_INSPECTION_NAME,
        SOURCE_RECORD_NAME,
        TELEMETRY_NAME,
        RUNNER_INVARIANT_NAME,
        LOCAL_OBSERVATION_NAME,
        "software-evidence",
    }
    expected_roots = required_roots | {
        PurePosixPath(path).parts[0] for path in sealed_additions
    }
    if {path.name for path in raw_root.iterdir()} != expected_roots:
        raise AttachmentError("raw DirectML qualification has an unexpected top-level file set")
    if not (raw_root / "software-evidence").is_dir():
        raise AttachmentError("raw DirectML qualification omitted software-evidence")


def _zip_tree_atomic(source_root: Path, destination: Path) -> None:
    if destination.exists():
        raise AttachmentError(f"refusing to overwrite evidence archive: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise AttachmentError(f"temporary evidence path already exists: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(source_root.rglob("*")):
                if path.is_symlink():
                    raise AttachmentError("evidence tree contains a symbolic link")
                if path.is_file():
                    archive.write(path, path.relative_to(source_root).as_posix())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def seal_evidence(
    raw_root: Path,
    output_directory: Path,
    *,
    context: Mapping[str, Any],
    raw_content_manifest_sha256: str,
    observer_name: str,
    typed_confirmation: str,
) -> dict[str, Any]:
    role = str(context["gpu_role"])
    adapter_index = int(context["directml_adapter_index"])
    expected_confirmation = expected_physical_confirmation(str(context["tag"]), role)
    if typed_confirmation != expected_confirmation:
        raise AttachmentError(f"typed confirmation must equal: {expected_confirmation}")
    observer_name = _single_line(observer_name, "observer name")
    _validate_raw_tree(
        raw_root, expected_manifest_sha256=raw_content_manifest_sha256, context=context
    )
    candidate = _read_json_object(raw_root / CANDIDATE_INSPECTION_NAME, "candidate inspection")
    if (
        candidate.get("sha256") != context["directml_zip_sha256"]
        or candidate.get("candidate_manifest_sha256") != context["candidate_manifest_sha256"]
        or candidate.get("source_identity")
        != candidate_context(
            str(context["repository"]),
            str(context["tag"]),
            str(context["tag_commit"]),
            int(context["source_build_run_id"]),
        )
    ):
        raise AttachmentError("raw candidate inspection differs from the qualification identity")
    source = _read_json_object(raw_root / SOURCE_RECORD_NAME, "verified source record")
    if (
        source.get("tag_commit") != context["tag_commit"]
        or source.get("source_build_run", {}).get("id") != context["source_build_run_id"]
        or source.get("release_absent") is not True
    ):
        raise AttachmentError("raw source record differs from the exact staged tag run")
    manifest, metrics, intervals, descriptor = _validate_directml_software_evidence(
        raw_root / "software-evidence",
        candidate=candidate,
        tag_commit=str(context["tag_commit"]),
        role=role,
        adapter_index=adapter_index,
    )
    invariant = _validate_runner_invariant(
        raw_root / RUNNER_INVARIANT_NAME, role=role, adapter_index=adapter_index
    )
    checked = _utc_datetime(invariant.get("checked_at_utc"), "DirectML invariant timestamp")
    if checked > intervals[0][1] or intervals[0][1] - checked > timedelta(minutes=10):
        raise AttachmentError("DirectML runner invariant was not recorded immediately before the runs")
    telemetry = _parse_directml_telemetry(
        raw_root / TELEMETRY_NAME,
        role=role,
        adapter_index=adapter_index,
        descriptor=descriptor,
        intervals=intervals,
    )
    observation = _validate_observation(
        raw_root / LOCAL_OBSERVATION_NAME,
        repository=str(context["repository"]),
        tag=str(context["tag"]),
        role=role,
        adapter_index=adapter_index,
        qualification_run_id=int(context["qualification_run_id"]),
        observer_name=observer_name,
        typed_confirmation=typed_confirmation,
        final_run_completed=intervals[-1][2],
    )
    if output_directory.exists():
        raise AttachmentError(f"refusing to overwrite sealed evidence path: {output_directory}")
    temporary = output_directory.with_name(f".{output_directory.name}.tmp")
    if temporary.exists():
        raise AttachmentError(f"temporary sealed evidence path exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        sealed_tree = temporary / "sealed-tree"
        shutil.copytree(raw_root, sealed_tree)
        qualification_manifest_sha = sha256_file(
            raw_root / "software-evidence" / QUALIFICATION_MANIFEST_NAME
        )
        receipt_name = str(GPU_PRODUCTS[role]["receipt_name"])
        receipt = {
            "schema_version": 1,
            "status": "physically_qualified_directml_release_candidate",
            "repository": context["repository"],
            "tag": context["tag"],
            "tag_commit": context["tag_commit"],
            "gpu_role": role,
            "physical_gpu": {
                "product_name": GPU_PRODUCTS[role]["product_name"],
                "directml_adapter_index": adapter_index,
                "vendor_id": telemetry["vendor_id"],
                "device_id": telemetry["device_id"],
                "driver_version": telemetry["driver_version"],
            },
            "candidate": {
                "filename": DIRECTML_ARCHIVE_NAME,
                "sha256": context["directml_zip_sha256"],
                "candidate_manifest_sha256": context["candidate_manifest_sha256"],
                "release_default_model": candidate["release_default_model"],
                "build_info_sha256": candidate["build_info_sha256"],
                "dependency_manifest_sha256": candidate["dependency_manifest_sha256"],
            },
            "qualification_metrics": metrics,
            "telemetry": {
                "pid_correlated": telemetry["pid_correlated"],
                "adapter_luid_correlated": telemetry["adapter_luid_correlated"],
                "per_run": telemetry["per_run"],
            },
            "qualification_run": {
                "id": context["qualification_run_id"],
                "attempt": context["qualification_run_attempt"],
            },
            "privacy": {
                "redacted": True,
                "omitted": [
                    "observer identity",
                    "Task Manager text",
                    "process IDs",
                    "adapter LUID",
                    "raw telemetry",
                    "local filesystem paths",
                ],
            },
        }
        _write_json_atomic(sealed_tree / receipt_name, receipt)
        attestation = {
            "schema_version": 1,
            "status": "sealed_physical_directml_attestation",
            "sealed_at_utc": _now_utc(),
            "context": dict(context),
            "candidate": candidate,
            "qualification_manifest_sha256": qualification_manifest_sha,
            "raw_content_manifest_sha256": _normalize_sha(
                raw_content_manifest_sha256, "raw content manifest SHA-256"
            ),
            "physical_observation": {
                "observer_name": observation["observer_name"],
                "observed_at_utc": observation["observed_at_utc"],
                "task_manager_gpu_engine": observation["task_manager_gpu_engine"],
                "typed_confirmation": observation["typed_confirmation"],
            },
            "dxgi_descriptor": descriptor,
            "telemetry_summary": telemetry,
            "qualification_metrics": metrics,
            "public_receipt": {
                "filename": receipt_name,
                "sha256": sha256_file(sealed_tree / receipt_name),
            },
        }
        _write_json_atomic(sealed_tree / PHYSICAL_ATTESTATION_NAME, attestation)
        archive = temporary / qualification_archive_name(role)
        _zip_tree_atomic(sealed_tree, archive)
        result = {
            "status": "sealed_and_ready_for_dual_gpu_release_review",
            "gpu_role": role,
            "archive_name": archive.name,
            "archive_sha256": sha256_file(archive),
            "archive_size_bytes": archive.stat().st_size,
            "qualification_manifest_sha256": qualification_manifest_sha,
            "physical_attestation_sha256": sha256_file(sealed_tree / PHYSICAL_ATTESTATION_NAME),
            "public_receipt_sha256": sha256_file(sealed_tree / receipt_name),
        }
        # The uploaded Actions artifact contains exactly one immutable ZIP.
        shutil.rmtree(sealed_tree)
        temporary.replace(output_directory)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@dataclass(frozen=True)
class EvidenceInput:
    role: str
    run_id: int
    adapter_index: int
    archive_sha256: str
    qualification_manifest_sha256: str
    physical_attestation_sha256: str
    public_receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        role: str,
        run_id: str | int,
        adapter_index: str | int,
        archive_sha256: str,
        qualification_manifest_sha256: str,
        physical_attestation_sha256: str,
        public_receipt_sha256: str,
    ) -> "EvidenceInput":
        if role not in GPU_PRODUCTS:
            raise AttachmentError(f"unsupported DirectML evidence role {role!r}")
        try:
            index = int(adapter_index)
        except (TypeError, ValueError) as exc:
            raise AttachmentError("DirectML evidence adapter index must be an integer") from exc
        if index < 0:
            raise AttachmentError("DirectML evidence adapter index must be non-negative")
        return cls(
            role=role,
            run_id=_run_id(run_id, f"{role} qualification run ID"),
            adapter_index=index,
            archive_sha256=_normalize_sha(archive_sha256, f"{role} evidence archive SHA-256"),
            qualification_manifest_sha256=_normalize_sha(
                qualification_manifest_sha256, f"{role} qualification manifest SHA-256"
            ),
            physical_attestation_sha256=_normalize_sha(
                physical_attestation_sha256, f"{role} physical attestation SHA-256"
            ),
            public_receipt_sha256=_normalize_sha(
                public_receipt_sha256, f"{role} public receipt SHA-256"
            ),
        )


@dataclass(frozen=True)
class ReleaseInputs:
    repository: str
    tag: str
    source_run_id: int
    candidate_manifest_sha256: str
    directml_zip_sha256: str
    confirmation: str
    evidence: tuple[EvidenceInput, EvidenceInput]

    @classmethod
    def create(
        cls,
        *,
        repository: str,
        tag: str,
        source_run_id: str | int,
        candidate_manifest_sha256: str,
        directml_zip_sha256: str,
        confirmation: str,
        evidence: Iterable[EvidenceInput],
    ) -> "ReleaseInputs":
        _validate_identity(repository, tag)
        records = tuple(evidence)
        if len(records) != 2 or {record.role for record in records} != set(REQUIRED_ROLES):
            raise AttachmentError("release requires exactly one AMD and one NVIDIA DirectML evidence input")
        if len({record.run_id for record in records}) != 2:
            raise AttachmentError("AMD and NVIDIA qualification run IDs must be distinct")
        expected_confirmation = f"PUBLISH DUAL-GPU QUALIFIED DIRECTML RELEASE {tag}"
        if confirmation != expected_confirmation:
            raise AttachmentError(f"publication confirmation must equal: {expected_confirmation}")
        ordered = tuple(sorted(records, key=lambda record: REQUIRED_ROLES.index(record.role)))
        return cls(
            repository=repository,
            tag=tag,
            source_run_id=_run_id(source_run_id, "source build run ID"),
            candidate_manifest_sha256=_normalize_sha(
                candidate_manifest_sha256, "candidate manifest SHA-256"
            ),
            directml_zip_sha256=_normalize_sha(
                directml_zip_sha256, "DirectML ZIP SHA-256"
            ),
            confirmation=confirmation,
            evidence=(ordered[0], ordered[1]),
        )

    def for_role(self, role: str) -> EvidenceInput:
        return next(record for record in self.evidence if record.role == role)


@dataclass(frozen=True)
class HoldoutInput:
    """Authenticated Actions identities for the final independent holdout."""

    run_id: int
    prerequisite_artifact_id: int
    prerequisite_artifact_digest: str
    plan_artifact_id: int
    plan_artifact_digest: str
    evidence_artifact_id: int
    evidence_artifact_digest: str
    attestation_artifact_id: int
    attestation_artifact_digest: str

    @classmethod
    def create(
        cls,
        *,
        run_id: str | int,
        prerequisite_artifact_id: str | int,
        prerequisite_artifact_digest: str,
        plan_artifact_id: str | int,
        plan_artifact_digest: str,
        evidence_artifact_id: str | int,
        evidence_artifact_digest: str,
        attestation_artifact_id: str | int,
        attestation_artifact_digest: str,
    ) -> "HoldoutInput":
        prerequisite_id = _run_id(
            prerequisite_artifact_id,
            "independent holdout prerequisite artifact ID",
        )
        plan_id = _run_id(
            plan_artifact_id, "independent holdout frozen-plan artifact ID"
        )
        evidence_id = _run_id(
            evidence_artifact_id, "independent holdout evidence artifact ID"
        )
        attestation_id = _run_id(
            attestation_artifact_id,
            "independent holdout attestation artifact ID",
        )
        if len({prerequisite_id, plan_id, evidence_id, attestation_id}) != 4:
            raise AttachmentError(
                "holdout prerequisite, plan, evidence, and attestation artifact IDs must be distinct"
            )
        prerequisite_digest = _artifact_digest(prerequisite_artifact_digest)
        plan_digest = _artifact_digest(plan_artifact_digest)
        evidence_digest = _artifact_digest(evidence_artifact_digest)
        attestation_digest = _artifact_digest(attestation_artifact_digest)
        if (
            prerequisite_digest is None
            or plan_digest is None
            or evidence_digest is None
            or attestation_digest is None
        ):
            raise AttachmentError(
                "independent holdout Actions artifacts require SHA-256 digests"
            )
        return cls(
            run_id=_run_id(run_id, "independent holdout workflow run ID"),
            prerequisite_artifact_id=prerequisite_id,
            prerequisite_artifact_digest=prerequisite_digest,
            plan_artifact_id=plan_id,
            plan_artifact_digest=plan_digest,
            evidence_artifact_id=evidence_id,
            evidence_artifact_digest=evidence_digest,
            attestation_artifact_id=attestation_id,
            attestation_artifact_digest=attestation_digest,
        )


def verify_evidence_run(
    api: GitHubApi,
    *,
    inputs: ReleaseInputs,
    evidence: EvidenceInput,
    tag_commit: str,
) -> dict[str, Any]:
    run, workflow_id = _verify_actions_run(
        api,
        repository=inputs.repository,
        run_id=evidence.run_id,
        tag_commit=tag_commit,
        expected_event=QUALIFICATION_EVENT,
        expected_workflow=QUALIFICATION_WORKFLOW,
        description=f"{evidence.role} DirectML qualification run",
    )
    artifact = _single_run_artifact(
        api,
        run_id=evidence.run_id,
        expected_name=qualification_artifact_name(evidence.role),
        description=f"{evidence.role} DirectML qualification run",
        require_digest=True,
    )
    actor = run.get("actor")
    if not isinstance(actor, dict) or not str(actor.get("login") or "").strip():
        raise AttachmentError(f"{evidence.role} qualification run omitted its actor")
    return {
        "run": {
            "id": evidence.run_id,
            "event": run["event"],
            "status": run["status"],
            "conclusion": run["conclusion"],
            "head_sha": str(run["head_sha"]).lower(),
            "html_url": str(run.get("html_url") or ""),
            "workflow_id": workflow_id,
            "workflow_path": QUALIFICATION_WORKFLOW,
            "run_attempt": _first_run_attempt(
                run, f"{evidence.role} DirectML qualification run"
            ),
            "actor": str(actor["login"]),
        },
        "artifact": artifact,
    }


def verify_independent_holdout_run(
    api: GitHubApi,
    *,
    inputs: ReleaseInputs,
    holdout: HoldoutInput,
    tag_commit: str,
) -> dict[str, Any]:
    """Authenticate the successful protected holdout run and both artifacts."""

    run, workflow_id = _verify_actions_run(
        api,
        repository=inputs.repository,
        run_id=holdout.run_id,
        tag_commit=tag_commit,
        expected_event=INDEPENDENT_HOLDOUT_EVENT,
        expected_workflow=INDEPENDENT_HOLDOUT_WORKFLOW,
        description="independent holdout qualification run",
    )
    expected_names = {
        HOLDOUT_PREREQUISITE_ARTIFACT_NAME,
        HOLDOUT_PLAN_ARTIFACT_NAME,
        HOLDOUT_EVIDENCE_ARTIFACT_NAME,
        HOLDOUT_ATTESTATION_ARTIFACT_NAME,
    }
    raw_artifacts = api.get_paginated(
        f"/actions/runs/{holdout.run_id}/artifacts", "artifacts"
    )
    if (
        len(raw_artifacts) != len(expected_names)
        or {artifact.get("name") for artifact in raw_artifacts} != expected_names
    ):
        raise AttachmentError(
            "independent holdout run must contain exactly the four fixed artifacts"
        )
    artifacts: dict[str, dict[str, Any]] = {}
    artifact_ids: set[int] = set()
    artifact_digests: set[str] = set()
    for artifact in raw_artifacts:
        name = str(artifact["name"])
        artifact_id = artifact.get("id")
        digest = _artifact_digest(artifact.get("digest"))
        size = artifact.get("size_in_bytes")
        if (
            bool(artifact.get("expired"))
            or isinstance(artifact_id, bool)
            or not isinstance(artifact_id, int)
            or artifact_id <= 0
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or digest is None
            or artifact_id in artifact_ids
            or digest in artifact_digests
        ):
            raise AttachmentError(
                "independent holdout run artifact identity/digest/inventory is invalid"
            )
        artifact_ids.add(artifact_id)
        artifact_digests.add(digest)
        artifacts[name] = {
            "id": artifact_id,
            "name": name,
            "size_in_bytes": size,
            "digest": digest,
        }
    prerequisite = artifacts[HOLDOUT_PREREQUISITE_ARTIFACT_NAME]
    plan = artifacts[HOLDOUT_PLAN_ARTIFACT_NAME]
    evidence = artifacts[HOLDOUT_EVIDENCE_ARTIFACT_NAME]
    attestation = artifacts[HOLDOUT_ATTESTATION_ARTIFACT_NAME]
    if (
        prerequisite.get("id") != holdout.prerequisite_artifact_id
        or prerequisite.get("digest") != holdout.prerequisite_artifact_digest
        or plan.get("id") != holdout.plan_artifact_id
        or plan.get("digest") != holdout.plan_artifact_digest
        or
        evidence.get("id") != holdout.evidence_artifact_id
        or evidence.get("digest") != holdout.evidence_artifact_digest
        or attestation.get("id") != holdout.attestation_artifact_id
        or attestation.get("digest") != holdout.attestation_artifact_digest
    ):
        raise AttachmentError(
            "independent holdout Actions artifact ID/digest differs from dispatch"
        )
    run_attempt = _first_run_attempt(run, "independent holdout qualification run")
    actor = run.get("actor")
    if not isinstance(actor, Mapping) or not str(actor.get("login") or "").strip():
        raise AttachmentError("independent holdout run omitted its dispatch actor")
    return {
        "run": {
            "id": holdout.run_id,
            "attempt": run_attempt,
            "event": run["event"],
            "status": run["status"],
            "conclusion": run["conclusion"],
            "head_sha": str(run["head_sha"]).lower(),
            "html_url": str(run.get("html_url") or ""),
            "workflow_id": workflow_id,
            "workflow_path": INDEPENDENT_HOLDOUT_WORKFLOW,
            "actor": str(actor["login"]),
        },
        "evidence_artifact": evidence,
        "attestation_artifact": attestation,
        "prerequisite_artifact": prerequisite,
        "plan_artifact": plan,
    }


def verify_release_remote_contract(
    api: GitHubApi,
    inputs: ReleaseInputs,
    holdout: HoldoutInput | None = None,
) -> dict[str, Any]:
    source = verify_source_contract(
        api,
        repository=inputs.repository,
        tag=inputs.tag,
        source_run_id=inputs.source_run_id,
        require_no_release=True,
    )
    evidence = {
        record.role: verify_evidence_run(
            api, inputs=inputs, evidence=record, tag_commit=source["tag_commit"]
        )
        for record in inputs.evidence
    }
    result: dict[str, Any] = {"source": source, "evidence": evidence}
    if holdout is not None:
        if holdout.run_id in {inputs.source_run_id, *(item.run_id for item in inputs.evidence)}:
            raise AttachmentError(
                "source, physical qualification, and holdout run IDs must be distinct"
            )
        result["independent_holdout"] = verify_independent_holdout_run(
            api,
            inputs=inputs,
            holdout=holdout,
            tag_commit=source["tag_commit"],
        )
    return result


def _single_downloaded_evidence(directory: Path, role: str) -> Path:
    if not directory.is_dir():
        raise AttachmentError(f"downloaded {role} evidence directory not found")
    paths = list(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise AttachmentError(f"downloaded {role} evidence contains a symbolic link")
    files = [path for path in paths if path.is_file()]
    expected_name = qualification_archive_name(role)
    if len(files) != 1 or files[0].name != expected_name:
        raise AttachmentError(f"downloaded {role} evidence must contain exactly {expected_name}")
    return files[0]


def _downloaded_holdout_bundle(directory: Path) -> Path:
    """Require one exact extracted Actions artifact containing the fixed bundle."""

    if not directory.is_dir() or directory.is_symlink():
        raise AttachmentError("downloaded independent holdout evidence directory not found")
    # upload-artifact stores the selected directory's contents at the artifact
    # root.  Refuse convenience searching/nesting so an ambiguous second
    # bundle can never be selected.
    if not (directory / HOLDOUT_BUNDLE_MANIFEST_NAME).is_file():
        raise AttachmentError(
            "independent holdout evidence artifact has no fixed root manifest"
        )
    return directory


def _downloaded_holdout_attestation(directory: Path) -> Path:
    if directory.is_file() and not directory.is_symlink():
        if directory.name != HOLDOUT_ATTESTATION_NAME:
            raise AttachmentError(
                "independent holdout attestation has the wrong fixed filename"
            )
        return directory
    if not directory.is_dir() or directory.is_symlink():
        raise AttachmentError(
            "downloaded independent holdout attestation directory not found"
        )
    paths = list(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise AttachmentError("independent holdout attestation artifact contains a symlink")
    files = [path for path in paths if path.is_file()]
    if len(files) != 1 or files[0].relative_to(directory).as_posix() != HOLDOUT_ATTESTATION_NAME:
        raise AttachmentError(
            f"independent holdout attestation artifact must contain exactly {HOLDOUT_ATTESTATION_NAME}"
        )
    return files[0]


def _downloaded_holdout_plan(directory: Path) -> Path:
    if not directory.is_dir() or directory.is_symlink():
        raise AttachmentError("downloaded independent holdout plan directory not found")
    paths = list(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise AttachmentError("independent holdout plan artifact contains a symlink")
    files = [path for path in paths if path.is_file()]
    expected_name = HOLDOUT_BUNDLE_MEMBER_NAMES["evaluation_plan"]
    if len(files) != 1 or files[0].relative_to(directory).as_posix() != expected_name:
        raise AttachmentError(
            f"independent holdout plan artifact must contain exactly {expected_name}"
        )
    return files[0]


def _crosslink_holdout_candidate(
    *,
    candidate: Mapping[str, Any],
    bundle_directory: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Bind the sealed-plan candidate to the exact frozen DirectML archive."""

    binding = public_candidate_binding(project_root)
    plan, _ = strict_holdout_json_file(
        bundle_directory / HOLDOUT_BUNDLE_MEMBER_NAMES["evaluation_plan"],
        "independent holdout evaluation plan",
    )
    if plan.get("candidate") != binding:
        raise AttachmentError(
            "holdout plan candidate differs from the current release-default pointer"
        )
    release_default = candidate.get("release_default_model")
    model = binding.get("model_artifacts")
    labels = binding.get("labels")
    shape = binding.get("input_shape_nchw")
    if (
        not isinstance(release_default, Mapping)
        or not isinstance(model, Sequence)
        or len(model) != 1
        or not isinstance(model[0], Mapping)
        or not isinstance(labels, Mapping)
        or not isinstance(shape, Sequence)
        or len(shape) != 4
        or release_default.get("model_sha256") != model[0].get("sha256")
        or release_default.get("labels_sha256") != labels.get("sha256")
        or release_default.get("input_shape_hw") != list(shape[2:4])
        or release_default.get("detail_crop_size_source_pixels")
        != binding.get("detail_crop_size_source_pixels")
    ):
        raise AttachmentError(
            "holdout candidate/pointer/workload differs from the frozen DirectML archive"
        )
    return {
        "release_default_pointer_sha256": binding["pointer_sha256"],
        "release_default_pointer_content_sha256": binding[
            "pointer_content_sha256"
        ],
        "candidate_content_sha256": binding["candidate_content_sha256"],
        "candidate_manifest_sha256": binding["candidate_manifest_sha256"],
        "model_sha256": model[0]["sha256"],
        "labels_sha256": labels["sha256"],
        "input_shape_nchw": list(shape),
        "output_head": binding["output_head"],
        "selected_pipeline": binding["selected_pipeline"],
        "detail_crop_size_source_pixels": binding[
            "detail_crop_size_source_pixels"
        ],
    }


def _holdout_attestation_content_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("attestation_content_sha256", None)
    return holdout_canonical_hash(body)


def _authoritative_holdout_verification(
    value: Mapping[str, Any], public: Mapping[str, Any]
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "status",
        "bundle_manifest_sha256",
        "bundle_content_sha256",
        "receipt_content_sha256",
        "release_policy_sha256",
        "source_snapshot_sha256",
        "environment_record_sha256",
        "hardware_identity_sha256",
        "canonical_release_policy_matched",
        "release_evidence_eligible",
        "authenticated_origin_required",
        "release_approved",
        "release_pointer_changed",
        "consumed_exactly_once",
        "retired",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("status")
        != "verified_independent_holdout_publication_input_bundle"
        or value.get("bundle_manifest_sha256")
        != public.get("bundle_manifest_sha256")
        or value.get("bundle_content_sha256")
        != public.get("bundle_content_sha256")
        or value.get("receipt_content_sha256")
        != public.get("receipt_content_sha256")
        or value.get("release_policy_sha256")
        != public.get("release_policy_sha256")
        or value.get("source_snapshot_sha256")
        != public.get("source_snapshot_sha256")
        or value.get("environment_record_sha256")
        != public.get("environment_record_sha256")
        or value.get("hardware_identity_sha256")
        != public.get("hardware_identity_sha256")
        or value.get("canonical_release_policy_matched") is not True
        or value.get("release_evidence_eligible") is not True
        or value.get("authenticated_origin_required") is not True
        or value.get("release_approved") is not False
        or value.get("release_pointer_changed") is not False
        or value.get("consumed_exactly_once") is not True
        or value.get("retired") is not True
    ):
        raise AttachmentError(
            "private authoritative holdout verification differs from the public bundle"
        )
    return dict(value)


def _holdout_runtime_prerequisite_crosslink(
    *,
    public: Mapping[str, Any],
    inputs: ReleaseInputs,
    amd_public_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the holdout runtime claim to the authenticated AMD prerequisite."""

    hardware = public.get("hardware_identity")
    if not isinstance(hardware, Mapping):
        raise AttachmentError("holdout bundle omitted its validated hardware identity")
    amd = inputs.for_role("amd_rx_6950_xt")
    expected_physical = {
        "qualification_run_id": amd.run_id,
        "adapter_index": amd.adapter_index,
        "public_receipt_sha256": amd.public_receipt_sha256,
    }
    if (
        hardware.get("physical_evidence") != expected_physical
        or hardware.get("directml_adapter_index") != amd.adapter_index
        or hardware.get("directml_device") != f"DML:{amd.adapter_index}"
        or public.get("hardware_identity_sha256")
        != hardware.get("content_sha256")
        or not isinstance(public.get("environment_record_sha256"), str)
    ):
        raise AttachmentError(
            "holdout runtime hardware/environment does not bind the authenticated AMD prerequisite"
        )
    if amd_public_receipt is not None:
        physical_gpu = amd_public_receipt.get("physical_gpu")
        qualification_run = amd_public_receipt.get("qualification_run")
        if (
            not isinstance(physical_gpu, Mapping)
            or not isinstance(qualification_run, Mapping)
            or physical_gpu.get("product_name") != hardware.get("product_name")
            or physical_gpu.get("directml_adapter_index") != amd.adapter_index
            or str(physical_gpu.get("vendor_id") or "").casefold()
            != str(hardware.get("vendor_id") or "").casefold()
            or str(physical_gpu.get("device_id") or "").casefold()
            != str(hardware.get("device_id") or "").casefold()
            or physical_gpu.get("driver_version") != hardware.get("driver_version")
            or qualification_run.get("id") != amd.run_id
            or qualification_run.get("attempt") != 1
        ):
            raise AttachmentError(
                "holdout RX 6950 XT identity differs from the authenticated physical receipt"
            )
    return {
        "environment_record_sha256": public["environment_record_sha256"],
        "hardware_identity_sha256": public["hardware_identity_sha256"],
        "physical_evidence": expected_physical,
        "product_name": hardware["product_name"],
        "directml_device": hardware["directml_device"],
        "vendor_id": hardware["vendor_id"],
        "device_id": hardware["device_id"],
        "driver_version": hardware["driver_version"],
    }


def create_authenticated_holdout_attestation(
    api: GitHubApi,
    inputs: ReleaseInputs,
    *,
    stage_directory: Path,
    staged_content_manifest_sha256: str,
    holdout_workflow_run_id: int,
    holdout_workflow_run_attempt: int,
    prerequisite_artifact_id: int,
    prerequisite_artifact_digest: str,
    evidence_artifact_id: int,
    evidence_artifact_digest: str,
    plan_artifact_id: int,
    plan_artifact_digest: str,
    plan_artifact_directory: Path,
    bundle_directory: Path,
    authoritative_verification_path: Path,
    output: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Create the redacted attestation inside the protected reviewer job."""

    if holdout_workflow_run_attempt != 1:
        raise AttachmentError("independent holdout attestation requires run attempt 1")
    run_id = _run_id(holdout_workflow_run_id, "holdout workflow run ID")
    prerequisite_id = _run_id(
        prerequisite_artifact_id, "holdout prerequisite artifact ID"
    )
    evidence_id = _run_id(evidence_artifact_id, "holdout evidence artifact ID")
    plan_id = _run_id(plan_artifact_id, "holdout frozen-plan artifact ID")
    prerequisite_digest = _artifact_digest(prerequisite_artifact_digest)
    evidence_digest = _artifact_digest(evidence_artifact_digest)
    plan_digest = _artifact_digest(plan_artifact_digest)
    if (
        prerequisite_digest is None
        or evidence_digest is None
        or plan_digest is None
    ):
        raise AttachmentError(
            "holdout prerequisite/plan/evidence artifacts require digests"
        )
    if len({prerequisite_id, plan_id, evidence_id}) != 3:
        raise AttachmentError(
            "holdout prerequisite, plan, and evidence artifact IDs must be distinct"
        )
    remote = verify_release_remote_contract(api, inputs)
    stage_attestation, _ = _validate_release_stage(
        stage_directory,
        inputs=inputs,
        remote=remote,
        verification_run_id=run_id,
        staged_content_manifest_sha256=staged_content_manifest_sha256,
    )
    try:
        public = validate_public_holdout_bundle(
            _downloaded_holdout_bundle(bundle_directory),
            project_root=project_root,
        )
    except IndependentHoldoutReleaseContractError as exc:
        raise AttachmentError(f"public holdout bundle validation failed: {exc}") from exc
    amd_receipt = _read_json_object(
        stage_directory
        / str(GPU_PRODUCTS["amd_rx_6950_xt"]["receipt_name"]),
        "authenticated AMD RX 6950 XT public receipt",
    )
    runtime_prerequisite_crosslink = _holdout_runtime_prerequisite_crosslink(
        public=public,
        inputs=inputs,
        amd_public_receipt=amd_receipt,
    )
    authoritative_value = _read_json_object(
        authoritative_verification_path,
        "private authoritative holdout verification",
    )
    authoritative = _authoritative_holdout_verification(
        authoritative_value, public
    )
    committed_plan = _downloaded_holdout_plan(plan_artifact_directory)
    bundled_plan = bundle_directory / HOLDOUT_BUNDLE_MEMBER_NAMES["evaluation_plan"]
    if (
        sha256_file(committed_plan) != public["plan_sha256"]
        or committed_plan.read_bytes() != bundled_plan.read_bytes()
    ):
        raise AttachmentError(
            "evaluated plan differs from the exact pre-access Actions artifact"
        )
    candidate = stage_attestation.get("candidate")
    if not isinstance(candidate, Mapping):
        raise AttachmentError("verified hardware prerequisite omitted its candidate")
    candidate_crosslink = _crosslink_holdout_candidate(
        candidate=candidate,
        bundle_directory=bundle_directory,
        project_root=project_root,
    )
    context = release_stage_context(inputs, remote, run_id)
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": HOLDOUT_ATTESTATION_KIND,
        "status": HOLDOUT_ATTESTATION_STATUS,
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": remote["source"]["tag_commit"],
        "holdout_workflow": {
            "run_id": run_id,
            "run_attempt": 1,
            "head_sha": remote["source"]["tag_commit"],
            "workflow_path": INDEPENDENT_HOLDOUT_WORKFLOW,
            "event": INDEPENDENT_HOLDOUT_EVENT,
        },
        "verified_prerequisite": {
            "artifact_id": prerequisite_id,
            "artifact_name": HOLDOUT_PREREQUISITE_ARTIFACT_NAME,
            "artifact_digest": prerequisite_digest,
            "staged_content_manifest_sha256": _normalize_sha(
                staged_content_manifest_sha256,
                "holdout prerequisite staged-content manifest SHA-256",
            ),
            "stage_context_sha256": holdout_canonical_hash(context),
            "both_physical_directml_products_verified_before_access": True,
        },
        "source_candidate": {
            "source_build_run_id": inputs.source_run_id,
            "candidate_manifest_sha256": inputs.candidate_manifest_sha256,
            "directml_zip_sha256": inputs.directml_zip_sha256,
        },
        "physical_directml_evidence": {
            evidence.role: {
                "run_id": evidence.run_id,
                "adapter_index": evidence.adapter_index,
                "archive_sha256": evidence.archive_sha256,
                "qualification_manifest_sha256": evidence.qualification_manifest_sha256,
                "physical_attestation_sha256": evidence.physical_attestation_sha256,
                "public_receipt_sha256": evidence.public_receipt_sha256,
            }
            for evidence in inputs.evidence
        },
        "holdout_evidence_artifact": {
            "artifact_id": evidence_id,
            "artifact_name": HOLDOUT_EVIDENCE_ARTIFACT_NAME,
            "artifact_digest": evidence_digest,
            "public_verification": dict(public),
            "authoritative_private_verification": authoritative,
        },
        "frozen_plan_artifact": {
            "artifact_id": plan_id,
            "artifact_name": HOLDOUT_PLAN_ARTIFACT_NAME,
            "artifact_digest": plan_digest,
            "plan_sha256": public["plan_sha256"],
            "uploaded_before_sealed_member_access": True,
            "downloaded_by_exact_artifact_id_for_evaluation": True,
        },
        "candidate_crosslink": candidate_crosslink,
        "runtime_prerequisite_crosslink": runtime_prerequisite_crosslink,
        "release_policy": independent_holdout_release_policy(),
        "protected_controls": {
            "hardware_verified_before_environment_secret_mapping": True,
            "plan_frozen_before_sealed_member_access": True,
            "sealed_package_path_from_environment_secret_only": True,
            "one_time_consumption_and_retirement": True,
            "independent_attestation_environment_required": True,
            "manual_receipt_or_self_hash_is_not_authority": True,
        },
        "publication": {
            "authenticated_origin_required": True,
            "publish_redacted_receipt_only": True,
            "private_bundle_not_a_release_asset": True,
            "release_approved": False,
        },
    }
    record["attestation_content_sha256"] = _holdout_attestation_content_hash(record)
    if output.exists():
        raise AttachmentError(f"refusing to overwrite holdout attestation: {output}")
    _write_json_atomic(output, record)
    if output.read_bytes() != holdout_canonical_json_bytes(record):
        raise AttachmentError("published holdout attestation is not canonical JSON")
    return {
        "status": "authenticated_holdout_attestation_ready_for_artifact_upload",
        "attestation_sha256": sha256_file(output),
        "attestation_content_sha256": record["attestation_content_sha256"],
        "bundle_manifest_sha256": public["bundle_manifest_sha256"],
        "receipt_sha256": public["receipt_sha256"],
        "release_policy_sha256": public["release_policy_sha256"],
        "environment_record_sha256": public["environment_record_sha256"],
        "hardware_identity_sha256": public["hardware_identity_sha256"],
    }


def validate_authenticated_holdout_evidence(
    *,
    inputs: ReleaseInputs,
    holdout: HoldoutInput,
    remote: Mapping[str, Any],
    candidate: Mapping[str, Any],
    bundle_directory: Path,
    attestation_directory: Path,
    amd_public_receipt: Mapping[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Revalidate downloaded public bytes plus authenticated workflow identity."""

    holdout_remote = remote.get("independent_holdout")
    if not isinstance(holdout_remote, Mapping):
        raise AttachmentError("remote release contract omitted independent holdout origin")
    bundle_root = _downloaded_holdout_bundle(bundle_directory)
    try:
        public = validate_public_holdout_bundle(
            bundle_root,
            project_root=project_root,
        )
    except IndependentHoldoutReleaseContractError as exc:
        raise AttachmentError(f"public holdout bundle validation failed: {exc}") from exc
    runtime_prerequisite_crosslink = _holdout_runtime_prerequisite_crosslink(
        public=public,
        inputs=inputs,
        amd_public_receipt=amd_public_receipt,
    )
    candidate_crosslink = _crosslink_holdout_candidate(
        candidate=candidate,
        bundle_directory=bundle_root,
        project_root=project_root,
    )
    attestation_path = _downloaded_holdout_attestation(attestation_directory)
    attestation = _read_json_object(
        attestation_path, "authenticated independent holdout attestation"
    )
    if attestation_path.read_bytes() != holdout_canonical_json_bytes(attestation):
        raise AttachmentError("authenticated holdout attestation is not canonical JSON")
    workflow = holdout_remote.get("run")
    evidence_artifact = holdout_remote.get("evidence_artifact")
    attestation_artifact = holdout_remote.get("attestation_artifact")
    prerequisite_artifact = holdout_remote.get("prerequisite_artifact")
    plan_artifact = holdout_remote.get("plan_artifact")
    if not all(
        isinstance(item, Mapping)
        for item in (
            workflow,
            evidence_artifact,
            attestation_artifact,
            prerequisite_artifact,
            plan_artifact,
        )
    ):
        raise AttachmentError("authenticated holdout remote artifact records are incomplete")
    assert isinstance(workflow, Mapping)
    assert isinstance(evidence_artifact, Mapping)
    assert isinstance(attestation_artifact, Mapping)
    assert isinstance(prerequisite_artifact, Mapping)
    assert isinstance(plan_artifact, Mapping)
    authoritative = attestation.get("holdout_evidence_artifact", {})
    authoritative_private = (
        authoritative.get("authoritative_private_verification")
        if isinstance(authoritative, Mapping)
        else None
    )
    if not isinstance(authoritative_private, Mapping):
        raise AttachmentError("authenticated holdout attestation omitted private verification")
    _authoritative_holdout_verification(authoritative_private, public)
    expected_physical = {
        evidence.role: {
            "run_id": evidence.run_id,
            "adapter_index": evidence.adapter_index,
            "archive_sha256": evidence.archive_sha256,
            "qualification_manifest_sha256": evidence.qualification_manifest_sha256,
            "physical_attestation_sha256": evidence.physical_attestation_sha256,
            "public_receipt_sha256": evidence.public_receipt_sha256,
        }
        for evidence in inputs.evidence
    }
    prerequisite = attestation.get("verified_prerequisite")
    expected_controls = {
        "hardware_verified_before_environment_secret_mapping": True,
        "plan_frozen_before_sealed_member_access": True,
        "sealed_package_path_from_environment_secret_only": True,
        "one_time_consumption_and_retirement": True,
        "independent_attestation_environment_required": True,
        "manual_receipt_or_self_hash_is_not_authority": True,
    }
    if (
        set(attestation)
        != {
            "schema_version",
            "kind",
            "status",
            "repository",
            "tag",
            "tag_commit",
            "holdout_workflow",
            "verified_prerequisite",
            "source_candidate",
            "physical_directml_evidence",
            "holdout_evidence_artifact",
            "frozen_plan_artifact",
            "candidate_crosslink",
            "runtime_prerequisite_crosslink",
            "release_policy",
            "protected_controls",
            "publication",
            "attestation_content_sha256",
        }
        or attestation.get("schema_version") != 1
        or attestation.get("kind") != HOLDOUT_ATTESTATION_KIND
        or attestation.get("status") != HOLDOUT_ATTESTATION_STATUS
        or attestation.get("repository") != inputs.repository
        or attestation.get("tag") != inputs.tag
        or attestation.get("tag_commit") != remote["source"]["tag_commit"]
        or attestation.get("holdout_workflow")
        != {
            "run_id": holdout.run_id,
            "run_attempt": 1,
            "head_sha": remote["source"]["tag_commit"],
            "workflow_path": INDEPENDENT_HOLDOUT_WORKFLOW,
            "event": INDEPENDENT_HOLDOUT_EVENT,
        }
        or not isinstance(prerequisite, Mapping)
        or prerequisite.get("artifact_id") != prerequisite_artifact.get("id")
        or prerequisite.get("artifact_name") != HOLDOUT_PREREQUISITE_ARTIFACT_NAME
        or prerequisite.get("artifact_digest")
        != prerequisite_artifact.get("digest")
        or _normalize_sha(
            str(prerequisite.get("staged_content_manifest_sha256") or ""),
            "attested prerequisite stage manifest SHA-256",
        )
        != prerequisite.get("staged_content_manifest_sha256")
        or prerequisite.get("stage_context_sha256")
        != holdout_canonical_hash(
            release_stage_context(inputs, remote, holdout.run_id)
        )
        or prerequisite.get(
            "both_physical_directml_products_verified_before_access"
        )
        is not True
        or attestation.get("source_candidate")
        != {
            "source_build_run_id": inputs.source_run_id,
            "candidate_manifest_sha256": inputs.candidate_manifest_sha256,
            "directml_zip_sha256": inputs.directml_zip_sha256,
        }
        or attestation.get("physical_directml_evidence") != expected_physical
        or not isinstance(authoritative, Mapping)
        or authoritative.get("artifact_id") != evidence_artifact.get("id")
        or authoritative.get("artifact_name") != HOLDOUT_EVIDENCE_ARTIFACT_NAME
        or authoritative.get("artifact_digest") != evidence_artifact.get("digest")
        or attestation_artifact.get("id") != holdout.attestation_artifact_id
        or attestation_artifact.get("name") != HOLDOUT_ATTESTATION_ARTIFACT_NAME
        or attestation_artifact.get("digest")
        != holdout.attestation_artifact_digest
        or authoritative.get("public_verification") != public
        or attestation.get("frozen_plan_artifact")
        != {
            "artifact_id": plan_artifact.get("id"),
            "artifact_name": HOLDOUT_PLAN_ARTIFACT_NAME,
            "artifact_digest": plan_artifact.get("digest"),
            "plan_sha256": public["plan_sha256"],
            "uploaded_before_sealed_member_access": True,
            "downloaded_by_exact_artifact_id_for_evaluation": True,
        }
        or attestation.get("candidate_crosslink") != candidate_crosslink
        or attestation.get("runtime_prerequisite_crosslink")
        != runtime_prerequisite_crosslink
        or attestation.get("release_policy") != independent_holdout_release_policy()
        or attestation.get("protected_controls") != expected_controls
        or attestation.get("publication")
        != {
            "authenticated_origin_required": True,
            "publish_redacted_receipt_only": True,
            "private_bundle_not_a_release_asset": True,
            "release_approved": False,
        }
        or attestation.get("attestation_content_sha256")
        != _holdout_attestation_content_hash(attestation)
    ):
        raise AttachmentError(
            "authenticated holdout attestation differs from exact run/artifact/candidate policy"
        )
    return {
        "status": "verified_authenticated_release_eligible_independent_holdout",
        "run": dict(workflow),
        "evidence_artifact": dict(evidence_artifact),
        "attestation_artifact": dict(attestation_artifact),
        "prerequisite_artifact": dict(prerequisite_artifact),
        "plan_artifact": dict(plan_artifact),
        "public_verification": dict(public),
        "candidate_crosslink": candidate_crosslink,
        "runtime_prerequisite_crosslink": runtime_prerequisite_crosslink,
        "attestation_sha256": sha256_file(attestation_path),
        "attestation_content_sha256": attestation[
            "attestation_content_sha256"
        ],
        "receipt_path": bundle_root
        / HOLDOUT_BUNDLE_MEMBER_NAMES["receipt"],
    }


def _public_holdout_origin_summary(
    *,
    inputs: ReleaseInputs,
    holdout: HoldoutInput,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {
        name: {
            "id": record[f"{name}_artifact"]["id"],
            "name": record[f"{name}_artifact"]["name"],
            "digest": record[f"{name}_artifact"]["digest"],
        }
        for name in ("prerequisite", "plan", "evidence", "attestation")
    }
    return {
        "authenticated_origin": {
            "repository": inputs.repository,
            "workflow_path": INDEPENDENT_HOLDOUT_WORKFLOW,
            "event": INDEPENDENT_HOLDOUT_EVENT,
            "run_id": holdout.run_id,
            "run_attempt": record["run"]["attempt"],
            "head_sha": record["run"]["head_sha"],
            "html_url": record["run"]["html_url"],
            "artifacts": artifacts,
            "plan_sha256": record["public_verification"]["plan_sha256"],
            "attestation_sha256": record["attestation_sha256"],
            "attestation_content_sha256": record[
                "attestation_content_sha256"
            ],
        },
        "receipt": {
            "filename": PUBLIC_HOLDOUT_RECEIPT_NAME,
            "sha256": record["public_verification"]["receipt_sha256"],
        },
        "release_policy_sha256": record["public_verification"][
            "release_policy_sha256"
        ],
        "environment_record_sha256": record["public_verification"][
            "environment_record_sha256"
        ],
        "hardware_identity_sha256": record["public_verification"][
            "hardware_identity_sha256"
        ],
        "runtime_prerequisite_crosslink": record[
            "runtime_prerequisite_crosslink"
        ],
        "candidate_crosslink": record["candidate_crosslink"],
        "canonical_release_policy_matched": True,
        "release_evidence_eligible": True,
        "authenticated_origin_verified": True,
    }


def _extract_evidence_archive(archive_path: Path, destination: Path, role: str) -> None:
    if archive_path.name != qualification_archive_name(role):
        raise AttachmentError(f"{role} evidence archive has the wrong fixed name")
    if destination.exists():
        raise AttachmentError(f"refusing to reuse evidence extraction path: {destination}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > 512:
                raise AttachmentError(f"{role} evidence archive has an invalid file count")
            total = 0
            folded: set[str] = set()
            validated: list[tuple[str, zipfile.ZipInfo]] = []
            for info in infos:
                name = _safe_zip_name(info)
                if not name or info.is_dir() or info.filename.endswith("/"):
                    raise AttachmentError(f"{role} evidence archive must contain regular files only")
                if name.casefold() in folded:
                    raise AttachmentError(f"{role} evidence archive contains a duplicate path")
                folded.add(name.casefold())
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode) if mode else 0
                if file_type not in (0, stat.S_IFREG) or info.flag_bits & 0x1:
                    raise AttachmentError(f"{role} evidence contains a link, special, or encrypted entry")
                total += info.file_size
                if info.file_size < 0 or total > 1024 * 1024 * 1024:
                    raise AttachmentError(f"{role} evidence exceeds the extraction limit")
                validated.append((name, info))
            required = {
                RAW_CONTENT_MANIFEST_NAME,
                CANDIDATE_INSPECTION_NAME,
                SOURCE_RECORD_NAME,
                TELEMETRY_NAME,
                RUNNER_INVARIANT_NAME,
                LOCAL_OBSERVATION_NAME,
                PHYSICAL_ATTESTATION_NAME,
                str(GPU_PRODUCTS[role]["receipt_name"]),
                f"software-evidence/{QUALIFICATION_MANIFEST_NAME}",
            }
            names = {name for name, _ in validated}
            if not required.issubset(names):
                raise AttachmentError(f"{role} evidence archive omitted required sealed files")
            if archive.testzip() is not None:
                raise AttachmentError(f"{role} evidence archive failed CRC validation")
            destination.mkdir(parents=True)
            root = destination.resolve()
            for name, info in validated:
                target = destination.joinpath(*PurePosixPath(name).parts)
                resolved = target.resolve()
                if resolved != root and root not in resolved.parents:
                    raise AttachmentError(f"{role} evidence entry escaped extraction root")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except zipfile.BadZipFile as exc:
        raise AttachmentError(f"{role} evidence is not a valid ZIP") from exc


def validate_sealed_evidence(
    archive_path: Path,
    *,
    inputs: ReleaseInputs,
    evidence: EvidenceInput,
    remote: Mapping[str, Any],
    candidate: Mapping[str, Any],
    extraction_root: Path,
) -> dict[str, Any]:
    if sha256_file(archive_path) != evidence.archive_sha256:
        raise AttachmentError(f"{evidence.role} evidence archive SHA-256 mismatch")
    _extract_evidence_archive(archive_path, extraction_root, evidence.role)
    attestation_path = extraction_root / PHYSICAL_ATTESTATION_NAME
    receipt_name = str(GPU_PRODUCTS[evidence.role]["receipt_name"])
    receipt_path = extraction_root / receipt_name
    if sha256_file(attestation_path) != evidence.physical_attestation_sha256:
        raise AttachmentError(f"{evidence.role} physical attestation SHA-256 mismatch")
    if sha256_file(receipt_path) != evidence.public_receipt_sha256:
        raise AttachmentError(f"{evidence.role} public receipt SHA-256 mismatch")
    run_record = remote["evidence"][evidence.role]["run"]
    context = raw_content_context(
        repository=inputs.repository,
        tag=inputs.tag,
        tag_commit=remote["source"]["tag_commit"],
        source_run_id=inputs.source_run_id,
        qualification_run_id=evidence.run_id,
        qualification_run_attempt=run_record["run_attempt"],
        role=evidence.role,
        adapter_index=evidence.adapter_index,
        candidate_manifest_sha256=inputs.candidate_manifest_sha256,
        directml_zip_sha256=inputs.directml_zip_sha256,
    )
    attestation = _read_json_object(attestation_path, f"{evidence.role} physical attestation")
    if (
        set(attestation)
        != {
            "schema_version",
            "status",
            "sealed_at_utc",
            "context",
            "candidate",
            "qualification_manifest_sha256",
            "raw_content_manifest_sha256",
            "physical_observation",
            "dxgi_descriptor",
            "telemetry_summary",
            "qualification_metrics",
            "public_receipt",
        }
        or
        attestation.get("schema_version") != 1
        or attestation.get("status") != "sealed_physical_directml_attestation"
        or attestation.get("context") != context
        or attestation.get("candidate") != candidate
        or attestation.get("qualification_manifest_sha256")
        != evidence.qualification_manifest_sha256
    ):
        raise AttachmentError(f"{evidence.role} physical attestation identity differs from publication")
    raw_manifest_sha = _normalize_sha(
        str(attestation.get("raw_content_manifest_sha256") or ""),
        f"{evidence.role} raw content manifest SHA-256",
    )
    _validate_raw_tree(
        extraction_root,
        expected_manifest_sha256=raw_manifest_sha,
        context=context,
        sealed_additions=frozenset({PHYSICAL_ATTESTATION_NAME, receipt_name}),
    )
    # The sealed attestation and public receipt were added after the raw
    # manifest.  Validate that these are the only two additions.
    actual_paths = {
        path.relative_to(extraction_root).as_posix()
        for path in extraction_root.rglob("*")
        if path.is_file()
    }
    raw_manifest = _read_json_object(
        extraction_root / RAW_CONTENT_MANIFEST_NAME, f"{evidence.role} raw manifest"
    )
    raw_paths = {
        str(record.get("path"))
        for record in raw_manifest.get("files", [])
        if isinstance(record, dict)
    } | {RAW_CONTENT_MANIFEST_NAME}
    if actual_paths != raw_paths | {PHYSICAL_ATTESTATION_NAME, receipt_name}:
        raise AttachmentError(f"{evidence.role} sealed archive has an unexpected file set")
    raw_candidate = _read_json_object(
        extraction_root / CANDIDATE_INSPECTION_NAME, f"{evidence.role} candidate inspection"
    )
    if raw_candidate != candidate:
        raise AttachmentError(f"{evidence.role} evidence binds a different staged candidate")
    manifest, metrics, intervals, descriptor = _validate_directml_software_evidence(
        extraction_root / "software-evidence",
        candidate=candidate,
        tag_commit=remote["source"]["tag_commit"],
        role=evidence.role,
        adapter_index=evidence.adapter_index,
    )
    if sha256_file(extraction_root / "software-evidence" / QUALIFICATION_MANIFEST_NAME) != evidence.qualification_manifest_sha256:
        raise AttachmentError(f"{evidence.role} inner qualification manifest SHA-256 mismatch")
    invariant = _validate_runner_invariant(
        extraction_root / RUNNER_INVARIANT_NAME,
        role=evidence.role,
        adapter_index=evidence.adapter_index,
    )
    checked = _utc_datetime(invariant.get("checked_at_utc"), "DirectML invariant timestamp")
    if checked > intervals[0][1] or intervals[0][1] - checked > timedelta(minutes=10):
        raise AttachmentError(f"{evidence.role} runner invariant is outside the run timeline")
    telemetry = _parse_directml_telemetry(
        extraction_root / TELEMETRY_NAME,
        role=evidence.role,
        adapter_index=evidence.adapter_index,
        descriptor=descriptor,
        intervals=intervals,
    )
    physical = attestation.get("physical_observation")
    if not isinstance(physical, dict):
        raise AttachmentError(f"{evidence.role} attestation omitted the physical observation")
    observation = _validate_observation(
        extraction_root / LOCAL_OBSERVATION_NAME,
        repository=inputs.repository,
        tag=inputs.tag,
        role=evidence.role,
        adapter_index=evidence.adapter_index,
        qualification_run_id=evidence.run_id,
        observer_name=str(physical.get("observer_name") or ""),
        typed_confirmation=expected_physical_confirmation(inputs.tag, evidence.role),
        final_run_completed=intervals[-1][2],
    )
    sealed_at = _utc_datetime(
        attestation.get("sealed_at_utc"), f"{evidence.role} evidence seal timestamp"
    )
    observed_at = _utc_datetime(
        observation.get("observed_at_utc"), f"{evidence.role} observation timestamp"
    )
    if sealed_at < observed_at or sealed_at - observed_at > timedelta(days=7):
        raise AttachmentError(f"{evidence.role} evidence was not sealed after the reviewed observation")
    if physical != {
        "observer_name": observation["observer_name"],
        "observed_at_utc": observation["observed_at_utc"],
        "task_manager_gpu_engine": observation["task_manager_gpu_engine"],
        "typed_confirmation": observation["typed_confirmation"],
    }:
        raise AttachmentError(f"{evidence.role} attestation observation differs from raw evidence")
    if attestation.get("dxgi_descriptor") != descriptor or attestation.get("telemetry_summary") != telemetry or attestation.get("qualification_metrics") != metrics:
        raise AttachmentError(f"{evidence.role} sealed summaries differ from recomputed evidence")
    receipt_record = attestation.get("public_receipt")
    if receipt_record != {"filename": receipt_name, "sha256": evidence.public_receipt_sha256}:
        raise AttachmentError(f"{evidence.role} public receipt record differs from attestation")
    receipt = _read_json_object(receipt_path, f"{evidence.role} public receipt")
    expected_receipt = {
        "schema_version": 1,
        "status": "physically_qualified_directml_release_candidate",
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": remote["source"]["tag_commit"],
        "gpu_role": evidence.role,
        "physical_gpu": {
            "product_name": GPU_PRODUCTS[evidence.role]["product_name"],
            "directml_adapter_index": evidence.adapter_index,
            "vendor_id": telemetry["vendor_id"],
            "device_id": telemetry["device_id"],
            "driver_version": telemetry["driver_version"],
        },
        "candidate": {
            "filename": DIRECTML_ARCHIVE_NAME,
            "sha256": inputs.directml_zip_sha256,
            "candidate_manifest_sha256": inputs.candidate_manifest_sha256,
            "release_default_model": candidate["release_default_model"],
            "build_info_sha256": candidate["build_info_sha256"],
            "dependency_manifest_sha256": candidate["dependency_manifest_sha256"],
        },
        "qualification_metrics": metrics,
        "telemetry": {
            "pid_correlated": telemetry["pid_correlated"],
            "adapter_luid_correlated": telemetry["adapter_luid_correlated"],
            "per_run": telemetry["per_run"],
        },
        "qualification_run": {"id": evidence.run_id, "attempt": run_record["run_attempt"]},
        "privacy": {
            "redacted": True,
            "omitted": [
                "observer identity",
                "Task Manager text",
                "process IDs",
                "adapter LUID",
                "raw telemetry",
                "local filesystem paths",
            ],
        },
    }
    if receipt != expected_receipt:
        raise AttachmentError(f"{evidence.role} public receipt differs from the exact redacted schema")
    serialized = json.dumps(receipt, sort_keys=True).casefold()
    for sensitive in (
        str(observation["observer_name"]),
        str(observation["task_manager_gpu_engine"]),
        str(attestation["telemetry_summary"].get("adapter_luid") or ""),
    ):
        if sensitive and sensitive.casefold() in serialized:
            raise AttachmentError(f"{evidence.role} public receipt leaked private physical evidence")
    return {
        "role": evidence.role,
        "archive_sha256": evidence.archive_sha256,
        "qualification_manifest_sha256": evidence.qualification_manifest_sha256,
        "physical_attestation_sha256": evidence.physical_attestation_sha256,
        "public_receipt_sha256": evidence.public_receipt_sha256,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "qualification_run": run_record,
        "qualification_artifact": remote["evidence"][evidence.role]["artifact"],
    }


def release_stage_context(
    inputs: ReleaseInputs,
    remote: Mapping[str, Any],
    verification_run_id: int,
    holdout: HoldoutInput | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": remote["source"]["tag_commit"],
        "source_build_run_id": inputs.source_run_id,
        "verification_run_id": int(verification_run_id),
        "candidate_manifest_sha256": inputs.candidate_manifest_sha256,
        "directml_zip_sha256": inputs.directml_zip_sha256,
        "evidence": {
            evidence.role: {
                "run_id": evidence.run_id,
                "adapter_index": evidence.adapter_index,
                "archive_sha256": evidence.archive_sha256,
                "qualification_manifest_sha256": evidence.qualification_manifest_sha256,
                "physical_attestation_sha256": evidence.physical_attestation_sha256,
                "public_receipt_sha256": evidence.public_receipt_sha256,
            }
            for evidence in inputs.evidence
        },
    }
    if holdout is not None:
        holdout_remote = remote.get("independent_holdout")
        if not isinstance(holdout_remote, Mapping):
            raise AttachmentError(
                "release-stage context omitted authenticated holdout remote identity"
            )
        context["independent_holdout"] = {
            "run_id": holdout.run_id,
            "prerequisite_artifact": {
                "id": holdout.prerequisite_artifact_id,
                "digest": holdout.prerequisite_artifact_digest,
            },
            "plan_artifact": {
                "id": holdout.plan_artifact_id,
                "digest": holdout.plan_artifact_digest,
            },
            "evidence_artifact": {
                "id": holdout.evidence_artifact_id,
                "digest": holdout.evidence_artifact_digest,
            },
            "attestation_artifact": {
                "id": holdout.attestation_artifact_id,
                "digest": holdout.attestation_artifact_digest,
            },
            "workflow_path": INDEPENDENT_HOLDOUT_WORKFLOW,
        }
    return context


def prepare_release_stage(
    *,
    inputs: ReleaseInputs,
    remote: Mapping[str, Any],
    candidate_directory: Path,
    evidence_directories: Mapping[str, Path],
    stage_directory: Path,
    verification_run_id: int,
    holdout: HoldoutInput | None = None,
    holdout_bundle_directory: Path | None = None,
    holdout_attestation_directory: Path | None = None,
) -> dict[str, Any]:
    holdout_paths_supplied = (
        holdout_bundle_directory is not None
        or holdout_attestation_directory is not None
    )
    if (holdout is None) is holdout_paths_supplied or (
        holdout is not None
        and (
            holdout_bundle_directory is None
            or holdout_attestation_directory is None
        )
    ):
        raise AttachmentError(
            "authenticated holdout input and both downloaded artifact directories are all required together"
        )
    if stage_directory.exists():
        raise AttachmentError(f"refusing to overwrite release stage: {stage_directory}")
    candidate_files = _downloaded_candidate_files(candidate_directory)
    temporary = stage_directory.with_name(f".{stage_directory.name}.tmp")
    if temporary.exists():
        raise AttachmentError(f"temporary release stage already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix="proaim-directml-stage-validate-") as scratch_name:
            scratch = Path(scratch_name)
            candidate = inspect_candidate(
                candidate_directory,
                repository=inputs.repository,
                tag=inputs.tag,
                tag_commit=remote["source"]["tag_commit"],
                source_run_id=inputs.source_run_id,
                candidate_manifest_sha256=inputs.candidate_manifest_sha256,
                directml_zip_sha256=inputs.directml_zip_sha256,
                extraction_root=scratch / "candidate",
            )
            evidence_records: dict[str, dict[str, Any]] = {}
            for evidence in inputs.evidence:
                directory = evidence_directories.get(evidence.role)
                if directory is None:
                    raise AttachmentError(f"missing downloaded {evidence.role} evidence directory")
                archive = _single_downloaded_evidence(directory, evidence.role)
                record = validate_sealed_evidence(
                    archive,
                    inputs=inputs,
                    evidence=evidence,
                    remote=remote,
                    candidate=candidate,
                    extraction_root=scratch / f"evidence-{evidence.role}",
                )
                evidence_records[evidence.role] = record

            holdout_record: dict[str, Any] | None = None
            if holdout is not None:
                assert holdout_bundle_directory is not None
                assert holdout_attestation_directory is not None
                holdout_record = validate_authenticated_holdout_evidence(
                    inputs=inputs,
                    holdout=holdout,
                    remote=remote,
                    candidate=candidate,
                    bundle_directory=holdout_bundle_directory,
                    attestation_directory=holdout_attestation_directory,
                    amd_public_receipt=evidence_records["amd_rx_6950_xt"][
                        "receipt"
                    ],
                )

            for name in EXPECTED_ARCHIVES:
                shutil.copyfile(candidate_files[name], temporary / name)
            private = temporary / "private-evidence"
            private.mkdir()
            for evidence in inputs.evidence:
                archive = _single_downloaded_evidence(
                    evidence_directories[evidence.role], evidence.role
                )
                shutil.copyfile(archive, private / archive.name)
                receipt_source = Path(evidence_records[evidence.role]["receipt_path"])
                shutil.copyfile(receipt_source, temporary / receipt_source.name)
            if holdout_record is not None:
                private_holdout = temporary / PRIVATE_HOLDOUT_ROOT
                private_holdout.mkdir()
                private_bundle = private_holdout / "bundle"
                shutil.copytree(
                    _downloaded_holdout_bundle(holdout_bundle_directory),
                    private_bundle,
                    symlinks=False,
                )
                attestation_source = _downloaded_holdout_attestation(
                    holdout_attestation_directory
                )
                shutil.copyfile(
                    attestation_source, private_holdout / HOLDOUT_ATTESTATION_NAME
                )
                shutil.copyfile(
                    Path(holdout_record["receipt_path"]),
                    temporary / PUBLIC_HOLDOUT_RECEIPT_NAME,
                )

            overall_receipt = {
                "schema_version": 1,
                "status": "dual_gpu_physically_qualified_directml_release_candidate",
                "repository": inputs.repository,
                "tag": inputs.tag,
                "tag_commit": remote["source"]["tag_commit"],
                "source_build_run": {
                    "id": inputs.source_run_id,
                    "html_url": remote["source"]["source_build_run"]["html_url"],
                },
                "candidate": {
                    "manifest_sha256": inputs.candidate_manifest_sha256,
                    "linux_archive": candidate["linux_archive"],
                    "directml_archive": {
                        "filename": DIRECTML_ARCHIVE_NAME,
                        "sha256": inputs.directml_zip_sha256,
                    },
                    "release_default_model": candidate["release_default_model"],
                    "build_info_sha256": candidate["build_info_sha256"],
                    "dependency_manifest_sha256": candidate["dependency_manifest_sha256"],
                },
                "required_physical_products": [
                    {
                        "gpu_role": evidence.role,
                        "product_name": GPU_PRODUCTS[evidence.role]["product_name"],
                        "qualification_run": {
                            "id": evidence.run_id,
                            "html_url": remote["evidence"][evidence.role]["run"]["html_url"],
                        },
                        "evidence_archive_sha256": evidence.archive_sha256,
                        "public_receipt": {
                            "filename": GPU_PRODUCTS[evidence.role]["receipt_name"],
                            "sha256": evidence.public_receipt_sha256,
                        },
                    }
                    for evidence in inputs.evidence
                ],
                "publication_policy": {
                    "both_products_required": True,
                    "draft_until_all_assets_verified": True,
                    "tag_only_build_does_not_publish": True,
                    "cuda_attachment_is_separate": True,
                },
                "privacy": {
                    "redacted": True,
                    "raw_observations_and_telemetry_not_published": True,
                },
            }
            if holdout_record is not None:
                overall_receipt["status"] = (
                    "dual_gpu_and_independent_holdout_qualified_directml_release_candidate"
                )
                overall_receipt["independent_holdout"] = (
                    _public_holdout_origin_summary(
                        inputs=inputs,
                        holdout=holdout,
                        record=holdout_record,
                    )
                )
                overall_receipt["publication_policy"][
                    "independent_holdout_required"
                ] = True
                overall_receipt["privacy"][
                    "holdout_images_coco_metrics_plan_and_ledger_not_published"
                ] = True
            _write_json_atomic(temporary / PUBLIC_RELEASE_RECEIPT_NAME, overall_receipt)
            public_names = [
                LINUX_ARCHIVE_NAME,
                DIRECTML_ARCHIVE_NAME,
                *(str(GPU_PRODUCTS[evidence.role]["receipt_name"]) for evidence in inputs.evidence),
                PUBLIC_RELEASE_RECEIPT_NAME,
            ]
            if holdout_record is not None:
                public_names.append(PUBLIC_HOLDOUT_RECEIPT_NAME)
            checksums = {
                name: sha256_file(temporary / name) for name in public_names
            }
            (temporary / CHECKSUM_NAME).write_bytes(render_checksum_manifest(checksums))
            attestation = {
                "schema_version": 1,
                "status": "verified_dual_gpu_directml_release_stage",
                "created_at_utc": _now_utc(),
                "context": release_stage_context(
                    inputs, remote, verification_run_id, holdout
                ),
                "source": remote["source"],
                "remote_evidence": remote["evidence"],
                "candidate": candidate,
                "sealed_evidence": {
                    role: {
                        key: value
                        for key, value in record.items()
                        if key not in {"receipt_path", "receipt"}
                    }
                    for role, record in evidence_records.items()
                },
                "public_assets": checksums,
                "release_receipt_sha256": sha256_file(
                    temporary / PUBLIC_RELEASE_RECEIPT_NAME
                ),
            }
            if holdout_record is not None:
                attestation["status"] = (
                    "verified_dual_gpu_and_authenticated_holdout_directml_release_stage"
                )
                attestation["independent_holdout"] = {
                    key: value
                    for key, value in holdout_record.items()
                    if key != "receipt_path"
                }
            _write_json_atomic(temporary / RELEASE_ATTESTATION_NAME, attestation)
            manifest_result = write_content_manifest(
                root=temporary,
                output=temporary / STAGED_CONTENT_MANIFEST_NAME,
                kind=STAGED_CONTENT_KIND,
                context=release_stage_context(
                    inputs, remote, verification_run_id, holdout
                ),
            )
        temporary.replace(stage_directory)
        return {
            "status": (
                "verified_dual_gpu_and_holdout_stage_ready_for_protected_publication"
                if holdout is not None
                else "verified_dual_gpu_stage_ready_for_protected_publication"
            ),
            "staged_content_manifest_sha256": manifest_result["manifest_sha256"],
            "release_receipt_sha256": sha256_file(
                stage_directory / PUBLIC_RELEASE_RECEIPT_NAME
            ),
            "public_assets": checksums,
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_release_stage(
    stage_directory: Path,
    *,
    inputs: ReleaseInputs,
    remote: Mapping[str, Any],
    verification_run_id: int,
    staged_content_manifest_sha256: str,
    holdout: HoldoutInput | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    validate_content_manifest(
        root=stage_directory,
        manifest_name=STAGED_CONTENT_MANIFEST_NAME,
        expected_sha256=staged_content_manifest_sha256,
        expected_kind=STAGED_CONTENT_KIND,
        expected_context=release_stage_context(
            inputs, remote, verification_run_id, holdout
        ),
    )
    expected_files = {
        LINUX_ARCHIVE_NAME,
        DIRECTML_ARCHIVE_NAME,
        CHECKSUM_NAME,
        RELEASE_ATTESTATION_NAME,
        PUBLIC_RELEASE_RECEIPT_NAME,
        STAGED_CONTENT_MANIFEST_NAME,
        *(str(record["receipt_name"]) for record in GPU_PRODUCTS.values()),
        *(f"private-evidence/{qualification_archive_name(role)}" for role in REQUIRED_ROLES),
    }
    if holdout is not None:
        expected_files.update(
            {
                PUBLIC_HOLDOUT_RECEIPT_NAME,
                f"{PRIVATE_HOLDOUT_ROOT}/{HOLDOUT_ATTESTATION_NAME}",
                *(
                    f"{PRIVATE_HOLDOUT_ROOT}/bundle/{name}"
                    for name in {
                        HOLDOUT_BUNDLE_MANIFEST_NAME,
                        *HOLDOUT_BUNDLE_MEMBER_NAMES.values(),
                    }
                ),
            }
        )
    actual_files = {
        path.relative_to(stage_directory).as_posix()
        for path in stage_directory.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise AttachmentError("verified release stage has an unexpected file set")
    attestation = _read_json_object(
        stage_directory / RELEASE_ATTESTATION_NAME, "DirectML release attestation"
    )
    if (
        attestation.get("schema_version") != 1
        or attestation.get("status")
        != (
            "verified_dual_gpu_and_authenticated_holdout_directml_release_stage"
            if holdout is not None
            else "verified_dual_gpu_directml_release_stage"
        )
        or attestation.get("context")
        != release_stage_context(inputs, remote, verification_run_id, holdout)
        or attestation.get("source") != remote["source"]
        or attestation.get("remote_evidence") != remote["evidence"]
    ):
        raise AttachmentError("release-stage attestation differs from current remote identities")
    public_assets = attestation.get("public_assets")
    if not isinstance(public_assets, dict):
        raise AttachmentError("release-stage attestation omitted public asset hashes")
    expected_public_names = {
        LINUX_ARCHIVE_NAME,
        DIRECTML_ARCHIVE_NAME,
        PUBLIC_RELEASE_RECEIPT_NAME,
        *(str(record["receipt_name"]) for record in GPU_PRODUCTS.values()),
    }
    if holdout is not None:
        expected_public_names.add(PUBLIC_HOLDOUT_RECEIPT_NAME)
    if set(public_assets) != expected_public_names:
        raise AttachmentError("release-stage public asset set is incomplete")
    normalized_assets: dict[str, str] = {}
    for name, digest in public_assets.items():
        normalized = _normalize_sha(str(digest), f"public asset {name} SHA-256")
        if sha256_file(stage_directory / name) != normalized:
            raise AttachmentError(f"staged public asset hash mismatch: {name}")
        normalized_assets[name] = normalized
    expected_checksums = render_checksum_manifest(normalized_assets)
    if (stage_directory / CHECKSUM_NAME).read_bytes() != expected_checksums:
        raise AttachmentError("staged SHA256SUMS.txt differs from exact public assets")
    if attestation.get("release_receipt_sha256") != sha256_file(
        stage_directory / PUBLIC_RELEASE_RECEIPT_NAME
    ):
        raise AttachmentError("release-stage receipt hash differs from attestation")
    if holdout is None:
        if "independent_holdout" in attestation:
            raise AttachmentError(
                "physical-only release stage unexpectedly contains holdout evidence"
            )
    else:
        candidate = attestation.get("candidate")
        if not isinstance(candidate, Mapping):
            raise AttachmentError("release-stage attestation omitted candidate")
        holdout_record = validate_authenticated_holdout_evidence(
            inputs=inputs,
            holdout=holdout,
            remote=remote,
            candidate=candidate,
            bundle_directory=stage_directory / PRIVATE_HOLDOUT_ROOT / "bundle",
            attestation_directory=stage_directory
            / PRIVATE_HOLDOUT_ROOT
            / HOLDOUT_ATTESTATION_NAME,
            amd_public_receipt=_read_json_object(
                stage_directory
                / str(GPU_PRODUCTS["amd_rx_6950_xt"]["receipt_name"]),
                "staged AMD RX 6950 XT public receipt",
            ),
        )
        expected_record = {
            key: value for key, value in holdout_record.items() if key != "receipt_path"
        }
        if attestation.get("independent_holdout") != expected_record:
            raise AttachmentError(
                "release-stage authenticated holdout summary differs from private evidence"
            )
        receipt_source = Path(holdout_record["receipt_path"])
        if (stage_directory / PUBLIC_HOLDOUT_RECEIPT_NAME).read_bytes() != receipt_source.read_bytes():
            raise AttachmentError(
                "public holdout receipt differs from authenticated private bundle"
            )
        public_release_receipt = _read_json_object(
            stage_directory / PUBLIC_RELEASE_RECEIPT_NAME,
            "public DirectML release receipt",
        )
        if (
            set(public_release_receipt)
            != {
                "schema_version",
                "status",
                "repository",
                "tag",
                "tag_commit",
                "source_build_run",
                "candidate",
                "required_physical_products",
                "publication_policy",
                "privacy",
                "independent_holdout",
            }
            or public_release_receipt.get("independent_holdout")
            != _public_holdout_origin_summary(
                inputs=inputs,
                holdout=holdout,
                record=holdout_record,
            )
        ):
            raise AttachmentError(
                "public release receipt authenticated holdout origin is not exact"
            )
    return attestation, normalized_assets


def verify_existing_release_stage(
    api: GitHubApi,
    inputs: ReleaseInputs,
    *,
    stage_directory: Path,
    verification_run_id: int,
    staged_content_manifest_sha256: str,
) -> dict[str, Any]:
    """Revalidate the physical-only prerequisite stage without publishing."""

    remote = verify_release_remote_contract(api, inputs)
    attestation, public_assets = _validate_release_stage(
        stage_directory,
        inputs=inputs,
        remote=remote,
        verification_run_id=verification_run_id,
        staged_content_manifest_sha256=staged_content_manifest_sha256,
    )
    candidate = attestation.get("candidate")
    if not isinstance(candidate, Mapping):
        raise AttachmentError("verified prerequisite stage omitted candidate")
    return {
        "status": "verified_dual_directml_prerequisite_before_holdout_access",
        "context": release_stage_context(inputs, remote, verification_run_id),
        "source": remote["source"],
        "remote_evidence": remote["evidence"],
        "candidate": candidate,
        "public_assets": public_assets,
        "staged_content_manifest_sha256": _normalize_sha(
            staged_content_manifest_sha256,
            "prerequisite staged-content manifest SHA-256",
        ),
        "holdout_access_permitted": False,
        "reason": (
            "Hardware prerequisites are verified; only the protected workflow may "
            "map the environment-only sealed package path and freeze a plan."
        ),
    }


def _api_json_mutation(
    api: GitHubApi,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None,
    expected_status: int,
) -> dict[str, Any] | None:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    status, _, response = api._request(  # noqa: SLF001 - same repository REST client
        method,
        api._api_root + path,  # noqa: SLF001
        data=body,
        content_type="application/json" if body is not None else None,
    )
    if status != expected_status:
        raise AttachmentError(
            f"GitHub API {method} {path} returned {status}, expected {expected_status}"
        )
    if expected_status == 204:
        if response:
            raise AttachmentError(f"GitHub API {method} {path} returned unexpected content")
        return None
    value = _strict_json_loads(response, f"GitHub API {method} {path} response")
    if not isinstance(value, dict):
        raise AttachmentError(f"GitHub API {method} {path} returned a non-object")
    return value


def _create_draft_release(
    api: GitHubApi, *, tag: str, tag_commit: str, marker: str
) -> dict[str, Any]:
    release = _api_json_mutation(
        api,
        "POST",
        "/releases",
        {
            "tag_name": tag,
            "target_commitish": tag_commit,
            "name": f"ProAim {tag}",
            "body": (
                "This release was staged from immutable Linux and DirectML artifacts and "
                "physically qualified on both required DirectML GPUs. See the attached "
                f"redacted qualification receipts.\n\n<!-- {marker} -->"
            ),
            "draft": True,
            "prerelease": False,
            "generate_release_notes": True,
        },
        201,
    )
    if not isinstance(release, dict):
        raise AttachmentError("GitHub did not return the created draft release")
    if (
        release.get("tag_name") != tag
        or release.get("draft") is not True
        or not isinstance(release.get("id"), int)
    ):
        raise AttachmentError("GitHub created a release with the wrong draft identity")
    return release


def _delete_release(api: GitHubApi, release_id: int) -> None:
    _api_json_mutation(api, "DELETE", f"/releases/{release_id}", None, 204)


def _publish_draft(api: GitHubApi, release_id: int) -> dict[str, Any]:
    value = _api_json_mutation(
        api, "PATCH", f"/releases/{release_id}", {"draft": False}, 200
    )
    if not isinstance(value, dict) or value.get("draft") is not False:
        raise AttachmentError("GitHub did not publish the verified draft release")
    return value


def _rollback_created_release(api: GitHubApi, release_id: int, tag: str) -> list[str]:
    errors: list[str] = []
    try:
        matches = _matching_releases(api, tag)
        target = next((item for item in matches if item.get("id") == release_id), None)
        if target is not None:
            _delete_release(api, release_id)
    except BaseException as exc:  # best-effort rollback must report every failure
        errors.append(str(exc))
    try:
        if any(item.get("id") == release_id for item in _matching_releases(api, tag)):
            errors.append("created release still exists after rollback")
    except BaseException as exc:
        errors.append(f"could not verify release rollback: {exc}")
    return errors


def _rollback_marker_releases(
    api: GitHubApi, *, tag: str, tag_commit: str, marker: str
) -> list[str]:
    """Reconcile a release when creation may have committed before failing.

    A lost HTTP response can leave the caller without the new release ID.  The
    unique publication-run marker allows deletion of only releases created by
    this transaction; unrelated releases are never selected for rollback.
    """

    errors: list[str] = []
    marker_text = f"<!-- {marker} -->"

    def matching() -> list[dict[str, Any]]:
        return [
            release
            for release in _matching_releases(api, tag)
            if marker_text in str(release.get("body") or "")
            and str(release.get("target_commitish") or "").lower()
            in {tag_commit.lower(), tag.lower()}
            and isinstance(release.get("id"), int)
        ]

    try:
        for release in matching():
            _delete_release(api, int(release["id"]))
    except BaseException as exc:  # best-effort rollback reports every failure
        errors.append(str(exc))
    try:
        if matching():
            errors.append("marker-identified release still exists after rollback")
    except BaseException as exc:
        errors.append(f"could not verify marker-based release rollback: {exc}")
    return errors


def _holdout_release_upload_names(
    public_assets: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the only assets a holdout-qualified publication may upload."""

    expected = {
        LINUX_ARCHIVE_NAME,
        DIRECTML_ARCHIVE_NAME,
        PUBLIC_RELEASE_RECEIPT_NAME,
        PUBLIC_HOLDOUT_RECEIPT_NAME,
        *(str(record["receipt_name"]) for record in GPU_PRODUCTS.values()),
    }
    if set(public_assets) != expected:
        raise AttachmentError(
            "holdout-qualified publication has an unexpected public asset set"
        )
    for name, digest in public_assets.items():
        if "/" in name or "\\" in name:
            raise AttachmentError("holdout-qualified public asset name is unsafe")
        _normalize_sha(str(digest), f"holdout-qualified public asset {name} SHA-256")
    return (*sorted(expected), CHECKSUM_NAME)


def publish_release(
    api: GitHubApi,
    inputs: ReleaseInputs,
    *,
    holdout: HoldoutInput,
    stage_directory: Path,
    verification_run_id: int,
    staged_content_manifest_sha256: str,
) -> dict[str, Any]:
    remote = verify_release_remote_contract(api, inputs, holdout)
    attestation, public_assets = _validate_release_stage(
        stage_directory,
        inputs=inputs,
        remote=remote,
        verification_run_id=verification_run_id,
        staged_content_manifest_sha256=staged_content_manifest_sha256,
        holdout=holdout,
    )
    # Revalidate both private evidence archives and the extracted DirectML
    # candidate in the publication job, not merely in the earlier verifier.
    with tempfile.TemporaryDirectory(prefix="proaim-directml-publish-") as scratch_name:
        scratch = Path(scratch_name)
        # The source candidate manifest is not a public asset; reconstruct its
        # exact bytes from the verified source artifact is impossible here.
        # Its SHA and archive records remain bound in the stage attestation,
        # while both archives themselves are rehashed below.
        candidate = attestation.get("candidate")
        if not isinstance(candidate, dict):
            raise AttachmentError("release-stage attestation omitted the candidate record")
        if (
            candidate.get("sha256") != inputs.directml_zip_sha256
            or candidate.get("candidate_manifest_sha256") != inputs.candidate_manifest_sha256
            or sha256_file(stage_directory / DIRECTML_ARCHIVE_NAME) != inputs.directml_zip_sha256
            or sha256_file(stage_directory / LINUX_ARCHIVE_NAME)
            != candidate.get("linux_archive", {}).get("sha256")
        ):
            raise AttachmentError("staged archives differ from the verified source candidate")
        _extract_directml_bundle(
            stage_directory / DIRECTML_ARCHIVE_NAME,
            expected_sha256=inputs.directml_zip_sha256,
            expected_commit=remote["source"]["tag_commit"],
            extraction_root=scratch / "candidate",
        )
        for evidence in inputs.evidence:
            record = validate_sealed_evidence(
                stage_directory / "private-evidence" / qualification_archive_name(evidence.role),
                inputs=inputs,
                evidence=evidence,
                remote=remote,
                candidate=candidate,
                extraction_root=scratch / f"evidence-{evidence.role}",
            )
            if record["public_receipt_sha256"] != public_assets[
                str(GPU_PRODUCTS[evidence.role]["receipt_name"])
            ]:
                raise AttachmentError(f"{evidence.role} staged public receipt differs from sealed evidence")

    if _matching_releases(api, inputs.tag):
        raise AttachmentError("a release appeared during verification; refusing to overwrite it")
    if resolve_tag_commit(api, inputs.tag) != remote["source"]["tag_commit"]:
        raise AttachmentError("tag moved before draft creation")
    marker = f"proaim-directml-publication-run-{verification_run_id}"
    release: Mapping[str, Any] | None = None
    release_id: int | None = None
    try:
        release = _create_draft_release(
            api,
            tag=inputs.tag,
            tag_commit=remote["source"]["tag_commit"],
            marker=marker,
        )
        release_id = int(release["id"])
        upload_names = _holdout_release_upload_names(public_assets)
        uploaded: dict[str, Mapping[str, Any]] = {}
        for name in upload_names:
            content_type = (
                "application/zip"
                if name.endswith(".zip")
                else "application/json; charset=utf-8"
                if name.endswith(".json")
                else "text/plain; charset=utf-8"
            )
            uploaded[name] = api.upload_release_asset(
                release_id, name, (stage_directory / name).read_bytes(), content_type
            )
            current = _assets_by_name(_release_assets(api, release_id))
            if set(current) != set(uploaded):
                raise AttachmentError("draft release asset set changed during upload")
        final_assets = _assets_by_name(_release_assets(api, release_id))
        if set(final_assets) != set(upload_names):
            raise AttachmentError("draft release does not contain the exact public asset set")
        for name in upload_names:
            payload = api.download_release_asset(_asset_id(final_assets[name], name))
            expected = (stage_directory / name).read_bytes()
            if payload != expected:
                raise AttachmentError(f"uploaded release asset failed byte verification: {name}")
        if resolve_tag_commit(api, inputs.tag) != remote["source"]["tag_commit"]:
            raise AttachmentError("tag moved while the draft release was being assembled")
        refreshed_source = verify_source_contract(
            api,
            repository=inputs.repository,
            tag=inputs.tag,
            source_run_id=inputs.source_run_id,
            require_no_release=False,
        )
        if any(
            refreshed_source.get(key) != remote["source"].get(key)
            for key in ("tag_commit", "source_build_run", "candidate_artifact")
        ):
            raise AttachmentError("source run or candidate artifact identity changed before publication")
        for evidence in inputs.evidence:
            refreshed = verify_evidence_run(
                api,
                inputs=inputs,
                evidence=evidence,
                tag_commit=remote["source"]["tag_commit"],
            )
            if refreshed != remote["evidence"][evidence.role]:
                raise AttachmentError(f"{evidence.role} evidence identity changed before publication")
        refreshed_holdout = verify_independent_holdout_run(
            api,
            inputs=inputs,
            holdout=holdout,
            tag_commit=remote["source"]["tag_commit"],
        )
        if refreshed_holdout != remote["independent_holdout"]:
            raise AttachmentError(
                "independent holdout run/artifact identity changed before publication"
            )
        matches = _matching_releases(api, inputs.tag)
        if len(matches) != 1 or matches[0].get("id") != release_id or matches[0].get("draft") is not True:
            raise AttachmentError("draft release identity changed immediately before publication")
        published = _publish_draft(api, release_id)
        matches = _matching_releases(api, inputs.tag)
        if (
            len(matches) != 1
            or matches[0].get("id") != release_id
            or matches[0].get("draft") is not False
            or resolve_tag_commit(api, inputs.tag) != remote["source"]["tag_commit"]
        ):
            raise AttachmentError("public release failed final tag/release verification")
        published_assets = _assets_by_name(_release_assets(api, release_id))
        if _asset_signature(published_assets.values()) != _asset_signature(final_assets.values()):
            raise AttachmentError("public release asset identities changed after publication")
        for name in upload_names:
            if api.download_release_asset(_asset_id(published_assets[name], name)) != (
                stage_directory / name
            ).read_bytes():
                raise AttachmentError(f"public release asset failed final verification: {name}")
    except BaseException as exc:
        rollback_errors: list[str]
        if release_id is not None:
            rollback_errors = _rollback_created_release(api, release_id, inputs.tag)
        else:
            rollback_errors = _rollback_marker_releases(
                api,
                tag=inputs.tag,
                tag_commit=remote["source"]["tag_commit"],
                marker=marker,
            )
        detail = f"DirectML publication failed; created release rollback was attempted: {exc}"
        if rollback_errors:
            detail += "; ROLLBACK ERRORS: " + " | ".join(rollback_errors)
        raise AttachmentError(detail) from exc
    return {
        "status": "published_and_verified",
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": remote["source"]["tag_commit"],
        "release_id": release_id,
        "release_html_url": str(published.get("html_url") or ""),
        "public_assets": {
            **public_assets,
            CHECKSUM_NAME: sha256_file(stage_directory / CHECKSUM_NAME),
        },
        "physical_gpu_roles": list(REQUIRED_ROLES),
        "independent_holdout_run_id": holdout.run_id,
        "independent_holdout_authenticated": True,
        "published_at_utc": _now_utc(),
    }


def _write_result(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise AttachmentError(f"refusing to overwrite output: {path}")
    _write_json_atomic(path, payload)


def _add_release_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-build-run-id", required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--directml-zip-sha256", required=True)
    parser.add_argument("--confirmation", required=True)
    for prefix in ("amd", "nvidia"):
        parser.add_argument(f"--{prefix}-evidence-run-id", required=True)
        parser.add_argument(f"--{prefix}-adapter-index", required=True)
        parser.add_argument(f"--{prefix}-evidence-archive-sha256", required=True)
        parser.add_argument(f"--{prefix}-qualification-manifest-sha256", required=True)
        parser.add_argument(f"--{prefix}-physical-attestation-sha256", required=True)
        parser.add_argument(f"--{prefix}-public-receipt-sha256", required=True)


def _add_holdout_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--holdout-run-id", required=True)
    for role in ("prerequisite", "plan", "evidence", "attestation"):
        parser.add_argument(f"--holdout-{role}-artifact-id", required=True)
        parser.add_argument(f"--holdout-{role}-artifact-digest", required=True)


def _holdout_input(args: argparse.Namespace) -> HoldoutInput:
    return HoldoutInput.create(
        run_id=args.holdout_run_id,
        prerequisite_artifact_id=args.holdout_prerequisite_artifact_id,
        prerequisite_artifact_digest=args.holdout_prerequisite_artifact_digest,
        plan_artifact_id=args.holdout_plan_artifact_id,
        plan_artifact_digest=args.holdout_plan_artifact_digest,
        evidence_artifact_id=args.holdout_evidence_artifact_id,
        evidence_artifact_digest=args.holdout_evidence_artifact_digest,
        attestation_artifact_id=args.holdout_attestation_artifact_id,
        attestation_artifact_digest=args.holdout_attestation_artifact_digest,
    )


def _release_inputs(args: argparse.Namespace) -> ReleaseInputs:
    evidence = (
        EvidenceInput.create(
            role="amd_rx_6950_xt",
            run_id=args.amd_evidence_run_id,
            adapter_index=args.amd_adapter_index,
            archive_sha256=args.amd_evidence_archive_sha256,
            qualification_manifest_sha256=args.amd_qualification_manifest_sha256,
            physical_attestation_sha256=args.amd_physical_attestation_sha256,
            public_receipt_sha256=args.amd_public_receipt_sha256,
        ),
        EvidenceInput.create(
            role="nvidia_rtx_5060_laptop",
            run_id=args.nvidia_evidence_run_id,
            adapter_index=args.nvidia_adapter_index,
            archive_sha256=args.nvidia_evidence_archive_sha256,
            qualification_manifest_sha256=args.nvidia_qualification_manifest_sha256,
            physical_attestation_sha256=args.nvidia_physical_attestation_sha256,
            public_receipt_sha256=args.nvidia_public_receipt_sha256,
        ),
    )
    return ReleaseInputs.create(
        repository=args.repository,
        tag=args.tag,
        source_run_id=args.source_build_run_id,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        directml_zip_sha256=args.directml_zip_sha256,
        confirmation=args.confirmation,
        evidence=evidence,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    stage = commands.add_parser("stage-candidate")
    stage.add_argument("--root", type=Path, required=True)
    stage.add_argument("--repository", required=True)
    stage.add_argument("--tag", required=True)
    stage.add_argument("--tag-commit", required=True)
    stage.add_argument("--source-build-run-id", required=True)
    stage.add_argument("--result", type=Path, required=True)

    source = commands.add_parser("verify-source")
    source.add_argument("--repository", required=True)
    source.add_argument("--tag", required=True)
    source.add_argument("--source-build-run-id", required=True)
    source.add_argument("--output", type=Path, required=True)

    inspect = commands.add_parser("inspect-candidate")
    inspect.add_argument("--downloaded-directory", type=Path, required=True)
    inspect.add_argument("--repository", required=True)
    inspect.add_argument("--tag", required=True)
    inspect.add_argument("--tag-commit", required=True)
    inspect.add_argument("--source-build-run-id", required=True)
    inspect.add_argument("--candidate-manifest-sha256", required=True)
    inspect.add_argument("--directml-zip-sha256", required=True)
    inspect.add_argument("--extract-directory", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)

    content = commands.add_parser("write-content-manifest")
    content.add_argument("--root", type=Path, required=True)
    content.add_argument("--output-name", default=RAW_CONTENT_MANIFEST_NAME)
    content.add_argument("--repository", required=True)
    content.add_argument("--tag", required=True)
    content.add_argument("--tag-commit", required=True)
    content.add_argument("--source-build-run-id", required=True)
    content.add_argument("--qualification-run-id", required=True)
    content.add_argument("--qualification-run-attempt", required=True)
    content.add_argument("--gpu-role", choices=REQUIRED_ROLES, required=True)
    content.add_argument("--adapter-index", required=True)
    content.add_argument("--candidate-manifest-sha256", required=True)
    content.add_argument("--directml-zip-sha256", required=True)
    content.add_argument("--result", type=Path, required=True)

    seal = commands.add_parser("seal-evidence")
    seal.add_argument("--raw-directory", type=Path, required=True)
    seal.add_argument("--output-directory", type=Path, required=True)
    seal.add_argument("--raw-content-manifest-sha256", required=True)
    seal.add_argument("--repository", required=True)
    seal.add_argument("--tag", required=True)
    seal.add_argument("--tag-commit", required=True)
    seal.add_argument("--source-build-run-id", required=True)
    seal.add_argument("--qualification-run-id", required=True)
    seal.add_argument("--qualification-run-attempt", required=True)
    seal.add_argument("--gpu-role", choices=REQUIRED_ROLES, required=True)
    seal.add_argument("--adapter-index", required=True)
    seal.add_argument("--candidate-manifest-sha256", required=True)
    seal.add_argument("--directml-zip-sha256", required=True)
    seal.add_argument("--observer-name", required=True)
    seal.add_argument("--typed-confirmation", required=True)
    seal.add_argument("--result", type=Path, required=True)

    metadata = commands.add_parser("verify-metadata")
    _add_release_inputs(metadata)
    metadata.add_argument("--output", type=Path, required=True)

    publication_metadata = commands.add_parser("verify-publication-metadata")
    _add_release_inputs(publication_metadata)
    _add_holdout_inputs(publication_metadata)
    publication_metadata.add_argument("--output", type=Path, required=True)

    prepare = commands.add_parser("prepare-stage")
    _add_release_inputs(prepare)
    prepare.add_argument("--candidate-directory", type=Path, required=True)
    prepare.add_argument("--amd-evidence-directory", type=Path, required=True)
    prepare.add_argument("--nvidia-evidence-directory", type=Path, required=True)
    prepare.add_argument("--stage-directory", type=Path, required=True)
    prepare.add_argument("--verification-run-id", required=True)
    prepare.add_argument("--result", type=Path, required=True)

    verify_stage = commands.add_parser("verify-stage")
    _add_release_inputs(verify_stage)
    verify_stage.add_argument("--stage-directory", type=Path, required=True)
    verify_stage.add_argument("--verification-run-id", required=True)
    verify_stage.add_argument("--staged-content-manifest-sha256", required=True)
    verify_stage.add_argument("--output", type=Path, required=True)

    attest = commands.add_parser("attest-holdout")
    _add_release_inputs(attest)
    attest.add_argument("--stage-directory", type=Path, required=True)
    attest.add_argument("--staged-content-manifest-sha256", required=True)
    attest.add_argument("--holdout-workflow-run-id", required=True)
    attest.add_argument("--holdout-workflow-run-attempt", required=True)
    attest.add_argument("--prerequisite-artifact-id", required=True)
    attest.add_argument("--prerequisite-artifact-digest", required=True)
    attest.add_argument("--plan-artifact-id", required=True)
    attest.add_argument("--plan-artifact-digest", required=True)
    attest.add_argument("--plan-artifact-directory", type=Path, required=True)
    attest.add_argument("--evidence-artifact-id", required=True)
    attest.add_argument("--evidence-artifact-digest", required=True)
    attest.add_argument("--bundle-directory", type=Path, required=True)
    attest.add_argument("--authoritative-verification", type=Path, required=True)
    attest.add_argument("--output", type=Path, required=True)
    attest.add_argument("--result", type=Path, required=True)

    prepare_publication = commands.add_parser("prepare-publication-stage")
    _add_release_inputs(prepare_publication)
    _add_holdout_inputs(prepare_publication)
    prepare_publication.add_argument("--candidate-directory", type=Path, required=True)
    prepare_publication.add_argument("--amd-evidence-directory", type=Path, required=True)
    prepare_publication.add_argument("--nvidia-evidence-directory", type=Path, required=True)
    prepare_publication.add_argument("--holdout-bundle-directory", type=Path, required=True)
    prepare_publication.add_argument("--holdout-attestation-directory", type=Path, required=True)
    prepare_publication.add_argument("--stage-directory", type=Path, required=True)
    prepare_publication.add_argument("--verification-run-id", required=True)
    prepare_publication.add_argument("--result", type=Path, required=True)

    publish = commands.add_parser("publish")
    _add_release_inputs(publish)
    _add_holdout_inputs(publish)
    publish.add_argument("--stage-directory", type=Path, required=True)
    publish.add_argument("--verification-run-id", required=True)
    publish.add_argument("--staged-content-manifest-sha256", required=True)
    publish.add_argument("--output", type=Path, required=True)
    return parser


def _api(inputs: ReleaseInputs | argparse.Namespace) -> GitHubApi:
    repository = inputs.repository
    return GitHubApi(os.environ.get("GH_TOKEN", ""), repository)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "stage-candidate":
            result = stage_candidate(
                args.root,
                repository=args.repository,
                tag=args.tag,
                tag_commit=args.tag_commit,
                source_run_id=_run_id(args.source_build_run_id, "source build run ID"),
            )
            _write_result(args.result, result)
        elif args.command == "verify-source":
            result = verify_source_contract(
                _api(args),
                repository=args.repository,
                tag=args.tag,
                source_run_id=_run_id(args.source_build_run_id, "source build run ID"),
            )
            _write_result(args.output, result)
        elif args.command == "inspect-candidate":
            result = inspect_candidate(
                args.downloaded_directory,
                repository=args.repository,
                tag=args.tag,
                tag_commit=args.tag_commit,
                source_run_id=_run_id(args.source_build_run_id, "source build run ID"),
                candidate_manifest_sha256=args.candidate_manifest_sha256,
                directml_zip_sha256=args.directml_zip_sha256,
                extraction_root=args.extract_directory,
            )
            _write_result(args.output, result)
        elif args.command == "write-content-manifest":
            try:
                adapter_index = int(args.adapter_index)
            except ValueError as exc:
                raise AttachmentError("adapter index must be an integer") from exc
            context = raw_content_context(
                repository=args.repository,
                tag=args.tag,
                tag_commit=args.tag_commit,
                source_run_id=_run_id(args.source_build_run_id, "source build run ID"),
                qualification_run_id=_run_id(
                    args.qualification_run_id, "qualification run ID"
                ),
                qualification_run_attempt=_run_id(
                    args.qualification_run_attempt, "qualification run attempt"
                ),
                role=args.gpu_role,
                adapter_index=adapter_index,
                candidate_manifest_sha256=args.candidate_manifest_sha256,
                directml_zip_sha256=args.directml_zip_sha256,
            )
            result = write_content_manifest(
                root=args.root,
                output=args.root / args.output_name,
                kind=RAW_CONTENT_KIND,
                context=context,
            )
            _write_result(args.result, result)
        elif args.command == "seal-evidence":
            adapter_index = int(args.adapter_index)
            context = raw_content_context(
                repository=args.repository,
                tag=args.tag,
                tag_commit=args.tag_commit,
                source_run_id=_run_id(args.source_build_run_id, "source build run ID"),
                qualification_run_id=_run_id(
                    args.qualification_run_id, "qualification run ID"
                ),
                qualification_run_attempt=_run_id(
                    args.qualification_run_attempt, "qualification run attempt"
                ),
                role=args.gpu_role,
                adapter_index=adapter_index,
                candidate_manifest_sha256=args.candidate_manifest_sha256,
                directml_zip_sha256=args.directml_zip_sha256,
            )
            result = seal_evidence(
                args.raw_directory,
                args.output_directory,
                context=context,
                raw_content_manifest_sha256=args.raw_content_manifest_sha256,
                observer_name=args.observer_name,
                typed_confirmation=args.typed_confirmation,
            )
            _write_result(args.result, result)
        elif args.command == "verify-metadata":
            inputs = _release_inputs(args)
            _write_result(args.output, verify_release_remote_contract(_api(inputs), inputs))
        elif args.command == "verify-publication-metadata":
            inputs = _release_inputs(args)
            holdout = _holdout_input(args)
            _write_result(
                args.output,
                verify_release_remote_contract(_api(inputs), inputs, holdout),
            )
        elif args.command == "prepare-stage":
            inputs = _release_inputs(args)
            remote = verify_release_remote_contract(_api(inputs), inputs)
            result = prepare_release_stage(
                inputs=inputs,
                remote=remote,
                candidate_directory=args.candidate_directory,
                evidence_directories={
                    "amd_rx_6950_xt": args.amd_evidence_directory,
                    "nvidia_rtx_5060_laptop": args.nvidia_evidence_directory,
                },
                stage_directory=args.stage_directory,
                verification_run_id=_run_id(
                    args.verification_run_id, "verification run ID"
                ),
            )
            _write_result(args.result, result)
        elif args.command == "verify-stage":
            inputs = _release_inputs(args)
            result = verify_existing_release_stage(
                _api(inputs),
                inputs,
                stage_directory=args.stage_directory,
                verification_run_id=_run_id(
                    args.verification_run_id, "verification run ID"
                ),
                staged_content_manifest_sha256=args.staged_content_manifest_sha256,
            )
            _write_result(args.output, result)
        elif args.command == "attest-holdout":
            inputs = _release_inputs(args)
            result = create_authenticated_holdout_attestation(
                _api(inputs),
                inputs,
                stage_directory=args.stage_directory,
                staged_content_manifest_sha256=args.staged_content_manifest_sha256,
                holdout_workflow_run_id=_run_id(
                    args.holdout_workflow_run_id, "holdout workflow run ID"
                ),
                holdout_workflow_run_attempt=_run_id(
                    args.holdout_workflow_run_attempt,
                    "holdout workflow run attempt",
                ),
                prerequisite_artifact_id=_run_id(
                    args.prerequisite_artifact_id,
                    "holdout prerequisite artifact ID",
                ),
                prerequisite_artifact_digest=args.prerequisite_artifact_digest,
                evidence_artifact_id=_run_id(
                    args.evidence_artifact_id,
                    "holdout evidence artifact ID",
                ),
                evidence_artifact_digest=args.evidence_artifact_digest,
                plan_artifact_id=_run_id(
                    args.plan_artifact_id, "holdout plan artifact ID"
                ),
                plan_artifact_digest=args.plan_artifact_digest,
                plan_artifact_directory=args.plan_artifact_directory,
                bundle_directory=args.bundle_directory,
                authoritative_verification_path=args.authoritative_verification,
                output=args.output,
            )
            _write_result(args.result, result)
        elif args.command == "prepare-publication-stage":
            inputs = _release_inputs(args)
            holdout = _holdout_input(args)
            remote = verify_release_remote_contract(_api(inputs), inputs, holdout)
            result = prepare_release_stage(
                inputs=inputs,
                remote=remote,
                candidate_directory=args.candidate_directory,
                evidence_directories={
                    "amd_rx_6950_xt": args.amd_evidence_directory,
                    "nvidia_rtx_5060_laptop": args.nvidia_evidence_directory,
                },
                stage_directory=args.stage_directory,
                verification_run_id=_run_id(
                    args.verification_run_id, "verification run ID"
                ),
                holdout=holdout,
                holdout_bundle_directory=args.holdout_bundle_directory,
                holdout_attestation_directory=args.holdout_attestation_directory,
            )
            _write_result(args.result, result)
        elif args.command == "publish":
            inputs = _release_inputs(args)
            holdout = _holdout_input(args)
            result = publish_release(
                _api(inputs),
                inputs,
                holdout=holdout,
                stage_directory=args.stage_directory,
                verification_run_id=_run_id(
                    args.verification_run_id, "verification run ID"
                ),
                staged_content_manifest_sha256=args.staged_content_manifest_sha256,
            )
            _write_result(args.output, result)
        else:  # pragma: no cover - argparse guarantees a known command
            raise AttachmentError(f"unsupported command: {args.command}")
    except (AttachmentError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
