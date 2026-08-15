#!/usr/bin/env python3
"""Verify and attach a physically qualified Windows CUDA release bundle.

This helper is intentionally narrow.  It supports the manual protected
workflow in ``.github/workflows/attach-qualified-cuda.yml`` and will never
create a tag or release.  The read-only verification phase binds a successful
manual Windows build artifact to the commit behind an existing release tag.
The publication phase rechecks that identity, verifies every existing ZIP
against ``SHA256SUMS.txt``, and adds the CUDA ZIP while replacing only that
checksum asset.  A best-effort rollback restores the original release assets
if a mutation fails.

The GitHub token is read only from ``GH_TOKEN``.  User-controlled workflow
inputs are passed as process arguments, never evaluated as shell source.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
    urlopen,
)
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.write_nvidia_redistribution_manifest import (  # noqa: E402
    NvidiaManifestError,
    validate_manifest as validate_nvidia_manifest,
)


API_VERSION = "2022-11-28"
EXPECTED_BUILD_WORKFLOW = ".github/workflows/build-windows.yml"
EXPECTED_BUILD_EVENT = "workflow_dispatch"
EXPECTED_BUILD_ARTIFACT = "ProAim-Windows-x64-NVIDIA-CUDA"
EXPECTED_QUALIFICATION_WORKFLOW = ".github/workflows/qualify-windows-cuda.yml"
EXPECTED_QUALIFICATION_EVENT = "workflow_dispatch"
EXPECTED_QUALIFICATION_ARTIFACT = "ProAim-Windows-CUDA-Qualification-Evidence"
QUALIFICATION_EVIDENCE_ARCHIVE_NAME = (
    "ProAim-Windows-CUDA-Qualification-Evidence.zip"
)
PHYSICAL_ATTESTATION_NAME = "PHYSICAL-GPU-ATTESTATION.json"
QUALIFICATION_MANIFEST_NAME = "qualification-manifest.json"
TASK_MANAGER_CONFIRMATION_NAME = "TASK-MANAGER-CONFIRMATION.txt"
NVIDIA_TELEMETRY_NAME = "nvidia-smi-telemetry.jsonl"
CUDA_RUNNER_INVARIANT_NAME = "CUDA-RUNNER-INVARIANT.json"
LOCAL_PHYSICAL_OBSERVATION_NAME = "LOCAL-PHYSICAL-OBSERVATION.json"
RAW_CONTENT_MANIFEST_NAME = "RAW-CONTENT-MANIFEST.json"
STAGED_CONTENT_MANIFEST_NAME = "STAGED-CONTENT-MANIFEST.json"
RAW_CANDIDATE_RECORD_NAME = "candidate-inspection.json"
RAW_SOURCE_RECORD_NAME = "verified-source.json"
CUDA_ARCHIVE_NAME = "ProAim-Windows-x64-NVIDIA-CUDA.zip"
CUDA_QUALIFICATION_RECEIPT_NAME = (
    "ProAim-Windows-x64-NVIDIA-CUDA-QUALIFICATION.json"
)
CHECKSUM_ASSET_NAME = "SHA256SUMS.txt"
REQUIRED_EXISTING_ARCHIVES = frozenset(
    {
        "ProAim-Linux-x64.zip",
        "ProAim-Windows-x64-DirectML.zip",
    }
)
VERIFIED_STAGE_FILES = frozenset(
    {
        CUDA_ARCHIVE_NAME,
        CUDA_QUALIFICATION_RECEIPT_NAME,
        QUALIFICATION_EVIDENCE_ARCHIVE_NAME,
        "ATTACHMENT-ATTESTATION.json",
        "hosted-runtime-info.json",
        "hosted-cpu-model-smoke.json",
        STAGED_CONTENT_MANIFEST_NAME,
    }
)
REQUIRED_BUNDLE_FILES = frozenset(
    {
        "ProAim/ProAimCLI.exe",
        "ProAim/BUILD-INFO.json",
        "ProAim/DEPENDENCY-MANIFEST.json",
        "ProAim/Qualify-ProAimGpu.ps1",
        "ProAim/LICENSE",
        "ProAim/THIRD_PARTY_NOTICES.md",
        "ProAim/NVIDIA-REDISTRIBUTION-MANIFEST.json",
    }
)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[1-9][0-9]{0,19}$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TAG_RE = re.compile(r"^v[A-Za-z0-9][A-Za-z0-9._-]{0,126}$")
CHECKSUM_LINE_RE = re.compile(
    r"^([0-9a-fA-F]{64}) [ *]([^/\\\r\n]+)$"
)
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
MAX_ARCHIVE_FILE_BYTES = 4 * 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 100_000
MAX_EVIDENCE_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
CUDA_QUALIFICATION_POLICY = {
    "benchmark_samples": 32,
    "benchmark_warmup": 30,
    "benchmark_iterations_per_repeat": 100,
    "benchmark_repeats": 3,
    "benchmark_max_p95_inference_ms": 35.0,
    "live_min_processed_frames": 120,
    "live_min_elapsed_fps": 20.0,
    "live_min_update_fps": 20.0,
    "live_max_p95_observed_pipeline_ms": 50.0,
    "live_max_p95_freshness_latency_ms": 50.0,
    "live_max_seconds": 60.0,
    "live_requested_max_frames": 1000,
    "live_min_elapsed_seconds": 2.0,
    "telemetry_interval_milliseconds": 500,
    "telemetry_min_correlated_samples_benchmark": 1,
    "telemetry_min_correlated_samples_live": 5,
}
LIVE_TIMING_FIELDS = (
    "capture_ms",
    "queue_age_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "detail_preprocess_ms",
    "detail_inference_ms",
    "detail_postprocess_ms",
    "control_ms",
    "processing_ms",
    "freshness_latency_ms",
    "observed_pipeline_ms",
    "draw_ms",
    "preview_service_ms",
)
REQUIRED_EVIDENCE_FILES = frozenset(
    {
        QUALIFICATION_MANIFEST_NAME,
        PHYSICAL_ATTESTATION_NAME,
        TASK_MANAGER_CONFIRMATION_NAME,
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
        NVIDIA_TELEMETRY_NAME,
        CUDA_RUNNER_INVARIANT_NAME,
        LOCAL_PHYSICAL_OBSERVATION_NAME,
        RAW_CANDIDATE_RECORD_NAME,
        RAW_SOURCE_RECORD_NAME,
        RAW_CONTENT_MANIFEST_NAME,
    }
)
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


class AttachmentError(RuntimeError):
    """Raised when an attachment gate cannot prove its required contract."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _strict_json_loads(payload: str | bytes, description: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AttachmentError(f"{description} is not strict JSON") from exc


def _normalize_sha256(value: str, description: str) -> str:
    normalized = str(value).strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise AttachmentError(f"{description} must be exactly 64 hexadecimal characters")
    return normalized


def _normalize_run_id(value: str) -> int:
    normalized = str(value).strip()
    if not RUN_ID_RE.fullmatch(normalized):
        raise AttachmentError("source build run ID must be a positive decimal integer")
    return int(normalized)


@dataclass(frozen=True, slots=True)
class DispatchInputs:
    repository: str
    tag: str
    build_run_id: int
    evidence_run_id: int
    evidence_artifact_name: str
    cuda_zip_sha256: str
    qualification_evidence_sha256: str
    qualification_manifest_sha256: str
    physical_attestation_sha256: str
    qualified_gpu: str
    confirmation: str
    nvidia_redistribution_confirmation: str

    @property
    def expected_confirmation(self) -> str:
        return f"ATTACH QUALIFIED WINDOWS CUDA TO {self.tag}"

    @property
    def expected_nvidia_confirmation(self) -> str:
        return f"I APPROVE NVIDIA REDISTRIBUTION REVIEW FOR {self.tag}"


def validate_dispatch_inputs(
    *,
    repository: str,
    tag: str,
    build_run_id: str,
    evidence_run_id: str,
    evidence_artifact_name: str,
    cuda_zip_sha256: str,
    qualification_evidence_sha256: str,
    qualification_manifest_sha256: str,
    physical_attestation_sha256: str,
    qualified_gpu: str,
    confirmation: str,
    nvidia_redistribution_confirmation: str,
) -> DispatchInputs:
    normalized_repository = repository.strip()
    if not REPOSITORY_RE.fullmatch(normalized_repository):
        raise AttachmentError("repository must use the exact OWNER/REPOSITORY form")
    normalized_tag = tag.strip()
    if not TAG_RE.fullmatch(normalized_tag):
        raise AttachmentError("tag must be a short v* release tag without slashes or whitespace")
    normalized_gpu = qualified_gpu.strip()
    if not normalized_gpu or len(normalized_gpu) > 160 or any(
        character in normalized_gpu for character in "\r\n\0"
    ):
        raise AttachmentError("qualified GPU must be a single non-empty line of at most 160 characters")
    expected_confirmation = f"ATTACH QUALIFIED WINDOWS CUDA TO {normalized_tag}"
    if confirmation != expected_confirmation:
        raise AttachmentError(
            "typed confirmation mismatch; expected exactly " + repr(expected_confirmation)
        )
    expected_nvidia_confirmation = (
        f"I APPROVE NVIDIA REDISTRIBUTION REVIEW FOR {normalized_tag}"
    )
    if nvidia_redistribution_confirmation != expected_nvidia_confirmation:
        raise AttachmentError(
            "NVIDIA redistribution confirmation mismatch; expected exactly "
            + repr(expected_nvidia_confirmation)
        )
    normalized_artifact_name = evidence_artifact_name.strip()
    if normalized_artifact_name != EXPECTED_QUALIFICATION_ARTIFACT:
        raise AttachmentError(
            "evidence artifact name must equal exactly "
            + repr(EXPECTED_QUALIFICATION_ARTIFACT)
        )
    return DispatchInputs(
        repository=normalized_repository,
        tag=normalized_tag,
        build_run_id=_normalize_run_id(build_run_id),
        evidence_run_id=_normalize_run_id(evidence_run_id),
        evidence_artifact_name=normalized_artifact_name,
        cuda_zip_sha256=_normalize_sha256(cuda_zip_sha256, "CUDA ZIP SHA-256"),
        qualification_evidence_sha256=_normalize_sha256(
            qualification_evidence_sha256,
            "qualification evidence archive SHA-256",
        ),
        qualification_manifest_sha256=_normalize_sha256(
            qualification_manifest_sha256,
            "qualification manifest SHA-256",
        ),
        physical_attestation_sha256=_normalize_sha256(
            physical_attestation_sha256,
            "physical GPU attestation SHA-256",
        ),
        qualified_gpu=normalized_gpu,
        confirmation=confirmation,
        nvidia_redistribution_confirmation=nvidia_redistribution_confirmation,
    )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        request: Request,
        fp: Any,
        code: int,
        message: str,
        headers: Mapping[str, str],
        new_url: str,
    ) -> None:
        return None


class GitHubApi:
    """Small REST client that never forwards the token to asset redirect hosts."""

    def __init__(self, token: str, repository: str) -> None:
        normalized_token = token.strip()
        if not normalized_token:
            raise AttachmentError("GH_TOKEN is required")
        if not REPOSITORY_RE.fullmatch(repository):
            raise AttachmentError("invalid repository identity")
        self._token = normalized_token
        self.repository = repository
        self._api_root = f"https://api.github.com/repos/{repository}"
        self._upload_root = f"https://uploads.github.com/repos/{repository}"

    def _request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        accept: str = "application/vnd.github+json",
        content_type: str | None = None,
        allow_redirects: bool = True,
    ) -> tuple[int, Mapping[str, str], bytes]:
        request = Request(url, data=data, method=method)
        request.add_header("Accept", accept)
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("User-Agent", "ProAim-qualified-CUDA-release-gate")
        request.add_header("X-GitHub-Api-Version", API_VERSION)
        if content_type is not None:
            request.add_header("Content-Type", content_type)
        opener = build_opener() if allow_redirects else build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=60) as response:
                return int(response.status), response.headers, response.read()
        except HTTPError as exc:
            body = exc.read(4096).decode("utf-8", errors="replace")
            raise AttachmentError(
                f"GitHub API {method} {urlparse(url).path} failed with HTTP {exc.code}: {body}"
            ) from exc
        except (OSError, URLError) as exc:
            raise AttachmentError(
                f"GitHub API {method} {urlparse(url).path} failed: {exc}"
            ) from exc

    def get_json(self, path: str) -> dict[str, Any]:
        status, _, payload = self._request("GET", self._api_root + path)
        if status != 200:
            raise AttachmentError(f"unexpected GitHub API status {status} for {path}")
        value = _strict_json_loads(payload, f"GitHub API response for {path}")
        if not isinstance(value, dict):
            raise AttachmentError(f"GitHub API returned a non-object for {path}")
        return value

    def get_json_list(self, path: str) -> list[dict[str, Any]]:
        status, _, payload = self._request("GET", self._api_root + path)
        if status != 200:
            raise AttachmentError(f"unexpected GitHub API status {status} for {path}")
        value = _strict_json_loads(payload, f"GitHub API response for {path}")
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise AttachmentError(f"GitHub API returned a non-object list for {path}")
        return value

    def get_paginated(self, path: str, key: str) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            payload = self.get_json(f"{path}{separator}per_page=100&page={page}")
            values = payload.get(key)
            if not isinstance(values, list):
                raise AttachmentError(f"GitHub API response omitted list {key!r}")
            for value in values:
                if not isinstance(value, dict):
                    raise AttachmentError(f"GitHub API list {key!r} contained a non-object")
                collected.append(value)
            if len(values) < 100:
                return collected
        raise AttachmentError(f"GitHub API pagination exceeded the safety limit for {path}")

    def download_release_asset(self, asset_id: int) -> bytes:
        url = f"{self._api_root}/releases/assets/{asset_id}"
        request = Request(url, method="GET")
        request.add_header("Accept", "application/octet-stream")
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("User-Agent", "ProAim-qualified-CUDA-release-gate")
        request.add_header("X-GitHub-Api-Version", API_VERSION)
        opener = build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=60) as response:
                if int(response.status) != 200:
                    raise AttachmentError(
                        f"release asset {asset_id} returned HTTP {response.status}"
                    )
                return response.read()
        except HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                body = exc.read(4096).decode("utf-8", errors="replace")
                raise AttachmentError(
                    f"release asset {asset_id} download failed with HTTP {exc.code}: {body}"
                ) from exc
            location = exc.headers.get("Location", "")
            parsed = urlparse(location)
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise AttachmentError("GitHub returned an unsafe release-asset redirect") from exc
            # Deliberately create a new request without Authorization.  The
            # signed redirect URL is sufficient and the repository token must
            # never be forwarded to blob storage.
            redirected = Request(location, method="GET")
            redirected.add_header("User-Agent", "ProAim-qualified-CUDA-release-gate")
            try:
                with urlopen(redirected, timeout=120) as response:
                    if int(response.status) != 200:
                        raise AttachmentError(
                            f"release asset redirect returned HTTP {response.status}"
                        )
                    return response.read()
            except (HTTPError, OSError, URLError) as redirect_exc:
                raise AttachmentError(
                    f"release asset {asset_id} redirected download failed: {redirect_exc}"
                ) from redirect_exc
        except (OSError, URLError) as exc:
            raise AttachmentError(f"release asset {asset_id} download failed: {exc}") from exc

    def upload_release_asset(
        self,
        release_id: int,
        name: str,
        payload: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        query = urlencode({"name": name})
        status, _, response = self._request(
            "POST",
            f"{self._upload_root}/releases/{release_id}/assets?{query}",
            data=payload,
            content_type=content_type,
        )
        if status != 201:
            raise AttachmentError(f"upload of {name} returned HTTP {status}, expected 201")
        value = _strict_json_loads(response, f"upload response for {name}")
        if not isinstance(value, dict) or value.get("name") != name:
            raise AttachmentError(f"upload of {name} returned the wrong asset identity")
        return value

    def delete_release_asset(self, asset_id: int) -> None:
        status, _, payload = self._request(
            "DELETE",
            f"{self._api_root}/releases/assets/{asset_id}",
        )
        if status != 204 or payload:
            raise AttachmentError(
                f"delete of release asset {asset_id} returned HTTP {status}, expected 204"
            )


def _require_sha(value: Any, description: str) -> str:
    normalized = str(value or "").lower()
    if not COMMIT_RE.fullmatch(normalized):
        raise AttachmentError(f"{description} was not a full 40-character commit SHA")
    return normalized


def resolve_tag_commit(api: GitHubApi, tag: str) -> str:
    reference = api.get_json(f"/git/ref/tags/{quote(tag, safe='')}")
    if reference.get("ref") != f"refs/tags/{tag}":
        raise AttachmentError("GitHub resolved a different tag reference")
    target = reference.get("object")
    if not isinstance(target, dict):
        raise AttachmentError("tag reference omitted its target object")
    seen: set[str] = set()
    for _ in range(8):
        kind = target.get("type")
        sha = _require_sha(target.get("sha"), "tag target")
        if sha in seen:
            raise AttachmentError("annotated tag chain contained a cycle")
        seen.add(sha)
        if kind == "commit":
            return sha
        if kind != "tag":
            raise AttachmentError(f"tag resolves to unsupported Git object type {kind!r}")
        annotated = api.get_json(f"/git/tags/{sha}")
        target = annotated.get("object")
        if not isinstance(target, dict):
            raise AttachmentError("annotated tag omitted its target object")
    raise AttachmentError("annotated tag chain exceeded the safety limit")


def _release_for_tag(api: GitHubApi, tag: str) -> dict[str, Any]:
    release = api.get_json(f"/releases/tags/{quote(tag, safe='')}")
    if release.get("tag_name") != tag:
        raise AttachmentError("GitHub returned a release for a different tag")
    if bool(release.get("draft")):
        raise AttachmentError("CUDA may be attached only to an already-published release")
    if not isinstance(release.get("id"), int):
        raise AttachmentError("release response omitted its numeric ID")
    return release


def _artifact_digest(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text.startswith("sha256:"):
        text = text.removeprefix("sha256:")
    return _normalize_sha256(text, "GitHub Actions artifact digest")


def _verify_actions_run(
    api: GitHubApi,
    *,
    repository: str,
    run_id: int,
    tag_commit: str,
    expected_event: str,
    expected_workflow: str,
    description: str,
) -> tuple[dict[str, Any], int]:
    run = api.get_json(f"/actions/runs/{run_id}")
    if run.get("event") != expected_event:
        raise AttachmentError(
            f"{description} event must be {expected_event!r}, got {run.get('event')!r}"
        )
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise AttachmentError(f"{description} must be completed successfully")
    if _require_sha(run.get("head_sha"), f"{description} head") != tag_commit:
        raise AttachmentError(f"{description} head SHA does not equal the resolved tag commit")
    run_repository = run.get("repository")
    head_repository = run.get("head_repository")
    if not isinstance(run_repository, dict) or run_repository.get("full_name") != repository:
        raise AttachmentError(f"{description} belongs to a different repository")
    if not isinstance(head_repository, dict) or head_repository.get("full_name") != repository:
        raise AttachmentError(f"{description} used source from a different repository")
    workflow_id = run.get("workflow_id")
    if not isinstance(workflow_id, int):
        raise AttachmentError(f"{description} omitted its workflow ID")
    workflow = api.get_json(f"/actions/workflows/{workflow_id}")
    if workflow.get("path") != expected_workflow:
        raise AttachmentError(
            f"{description} must use {expected_workflow}, got {workflow.get('path')!r}"
        )
    run_path = run.get("path")
    if run_path not in (None, expected_workflow) and not str(run_path).startswith(
        expected_workflow + "@"
    ):
        raise AttachmentError(f"{description} reported the wrong workflow path")
    return run, workflow_id


def _single_run_artifact(
    api: GitHubApi,
    *,
    run_id: int,
    expected_name: str,
    description: str,
    require_digest: bool,
) -> dict[str, Any]:
    artifacts = api.get_paginated(
        f"/actions/runs/{run_id}/artifacts",
        "artifacts",
    )
    matches = [artifact for artifact in artifacts if artifact.get("name") == expected_name]
    if len(matches) != 1:
        raise AttachmentError(
            f"{description} must contain exactly one {expected_name!r} artifact"
        )
    artifact = matches[0]
    if bool(artifact.get("expired")):
        raise AttachmentError(f"{description} artifact has expired")
    if not isinstance(artifact.get("id"), int) or int(artifact.get("size_in_bytes", 0)) <= 0:
        raise AttachmentError(f"{description} artifact has invalid identity or size")
    digest = _artifact_digest(artifact.get("digest"))
    if require_digest and digest is None:
        raise AttachmentError(f"{description} artifact omitted its GitHub SHA-256 digest")
    return {
        "id": artifact["id"],
        "name": expected_name,
        "size_in_bytes": int(artifact["size_in_bytes"]),
        "digest": digest,
    }


def verify_source_build_contract(
    api: GitHubApi,
    *,
    repository: str,
    tag: str,
    build_run_id: int,
) -> dict[str, Any]:
    tag_commit = resolve_tag_commit(api, tag)
    release = _release_for_tag(api, tag)
    run, workflow_id = _verify_actions_run(
        api,
        repository=repository,
        run_id=build_run_id,
        tag_commit=tag_commit,
        expected_event=EXPECTED_BUILD_EVENT,
        expected_workflow=EXPECTED_BUILD_WORKFLOW,
        description="source build run",
    )
    artifact = _single_run_artifact(
        api,
        run_id=build_run_id,
        expected_name=EXPECTED_BUILD_ARTIFACT,
        description="source build run",
        require_digest=False,
    )
    return {
        "tag_commit": tag_commit,
        "release_id": release["id"],
        "release_html_url": str(release.get("html_url") or ""),
        "build_run": {
            "id": build_run_id,
            "event": run["event"],
            "status": run["status"],
            "conclusion": run["conclusion"],
            "head_sha": str(run["head_sha"]).lower(),
            "html_url": str(run.get("html_url") or ""),
            "workflow_id": workflow_id,
            "workflow_path": EXPECTED_BUILD_WORKFLOW,
        },
        "build_artifact": artifact,
    }


def verify_remote_contract(api: GitHubApi, inputs: DispatchInputs) -> dict[str, Any]:
    result = verify_source_build_contract(
        api,
        repository=inputs.repository,
        tag=inputs.tag,
        build_run_id=inputs.build_run_id,
    )
    run, workflow_id = _verify_actions_run(
        api,
        repository=inputs.repository,
        run_id=inputs.evidence_run_id,
        tag_commit=result["tag_commit"],
        expected_event=EXPECTED_QUALIFICATION_EVENT,
        expected_workflow=EXPECTED_QUALIFICATION_WORKFLOW,
        description="physical qualification run",
    )
    artifact = _single_run_artifact(
        api,
        run_id=inputs.evidence_run_id,
        expected_name=inputs.evidence_artifact_name,
        description="physical qualification run",
        require_digest=True,
    )
    actor = run.get("actor")
    if not isinstance(actor, dict) or not str(actor.get("login") or "").strip():
        raise AttachmentError("physical qualification run omitted its GitHub actor")
    result["qualification_run"] = {
        "id": inputs.evidence_run_id,
        "event": run["event"],
        "status": run["status"],
        "conclusion": run["conclusion"],
        "head_sha": str(run["head_sha"]).lower(),
        "html_url": str(run.get("html_url") or ""),
        "workflow_id": workflow_id,
        "workflow_path": EXPECTED_QUALIFICATION_WORKFLOW,
        "run_attempt": int(run.get("run_attempt") or 1),
        "actor": str(actor["login"]),
    }
    result["qualification_artifact"] = artifact
    return result


def _safe_zip_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not name or "\\" in name or "\0" in name:
        raise AttachmentError(f"candidate ZIP contains an unsafe entry name {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise AttachmentError(f"candidate ZIP contains an unsafe path {name!r}")
    for part in path.parts:
        if (
            ":" in part
            or any(ord(character) < 32 for character in part)
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
        ):
            raise AttachmentError(f"candidate ZIP contains a Windows-unsafe path {name!r}")
    return path.as_posix().rstrip("/")


def _validated_zip_entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    if len(archive.infolist()) > MAX_ARCHIVE_ENTRIES:
        raise AttachmentError("candidate ZIP exceeds the entry-count safety limit")
    entries: dict[str, zipfile.ZipInfo] = {}
    casefolded: set[str] = set()
    total_size = 0
    for info in archive.infolist():
        name = _safe_zip_name(info)
        if not name:
            continue
        folded = name.casefold()
        if folded in casefolded:
            raise AttachmentError(f"candidate ZIP contains duplicate path {name!r}")
        casefolded.add(folded)
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode) if mode else 0
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise AttachmentError(f"candidate ZIP contains a link or special file {name!r}")
        if info.flag_bits & 0x1:
            raise AttachmentError(f"candidate ZIP contains encrypted entry {name!r}")
        if info.file_size < 0 or info.file_size > MAX_ARCHIVE_FILE_BYTES:
            raise AttachmentError(f"candidate ZIP entry is unreasonably large: {name!r}")
        total_size += info.file_size
        if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise AttachmentError("candidate ZIP exceeds the uncompressed-size safety limit")
        entries[name] = info
    if not entries:
        raise AttachmentError("candidate ZIP is empty")
    if any(not (name == "ProAim" or name.startswith("ProAim/")) for name in entries):
        raise AttachmentError("candidate ZIP must contain only the single ProAim root")
    missing = REQUIRED_BUNDLE_FILES.difference(entries)
    if missing:
        raise AttachmentError(
            "candidate ZIP is missing required frozen bundle files: " + ", ".join(sorted(missing))
        )
    return entries


def _validate_release_default_model_record(
    build_info: Mapping[str, Any],
    *,
    archive: zipfile.ZipFile | None = None,
    entries: Mapping[str, zipfile.ZipInfo] | None = None,
) -> dict[str, Any]:
    record = build_info.get("release_default_model")
    expected_keys = {
        "preset",
        "model_path",
        "labels_path",
        "input_shape_hw",
        "detail_crop_size_source_pixels",
        "model_sha256",
        "labels_sha256",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise AttachmentError("candidate BUILD-INFO has no exact release-default model contract")
    preset = record.get("preset")
    if (
        not isinstance(preset, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", preset)
    ):
        raise AttachmentError("release-default model preset is invalid")
    paths: dict[str, str] = {}
    for key in ("model_path", "labels_path"):
        value = record.get(key)
        if not isinstance(value, str):
            raise AttachmentError(f"release-default {key} is missing")
        pure = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or pure.is_absolute()
            or pure.as_posix() != value
            or not value.startswith("_internal/")
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise AttachmentError(f"release-default {key} is not a safe bundle-relative path")
        paths[key] = value
    if paths["model_path"] == paths["labels_path"] or not paths["model_path"].endswith(
        ".onnx"
    ):
        raise AttachmentError("release-default model/labels paths are invalid")
    shape = record.get("input_shape_hw")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 32
            or value > 4096
            or value % 32
            for value in shape
        )
    ):
        raise AttachmentError("release-default input shape must be [height,width] in YOLO dimensions")
    normalized = {
        "preset": preset,
        "model_path": paths["model_path"],
        "labels_path": paths["labels_path"],
        "input_shape_hw": list(shape),
        "detail_crop_size_source_pixels": record.get(
            "detail_crop_size_source_pixels"
        ),
        "model_sha256": _normalize_sha256(
            str(record.get("model_sha256") or ""), "release-default model SHA-256"
        ),
        "labels_sha256": _normalize_sha256(
            str(record.get("labels_sha256") or ""), "release-default labels SHA-256"
        ),
    }
    detail_crop = normalized["detail_crop_size_source_pixels"]
    if (
        isinstance(detail_crop, bool)
        or not isinstance(detail_crop, int)
        or detail_crop < 0
        or detail_crop > 16384
    ):
        raise AttachmentError(
            "release-default detail crop must be a source-pixel width from 0 to 16384"
        )
    if archive is not None or entries is not None:
        if archive is None or entries is None:
            raise AttachmentError("internal release-default archive validation is incomplete")
        for path_key, hash_key in (
            ("model_path", "model_sha256"),
            ("labels_path", "labels_sha256"),
        ):
            member_name = "ProAim/" + normalized[path_key]
            info = entries.get(member_name)
            if info is None or info.is_dir() or info.file_size <= 0:
                raise AttachmentError(f"candidate omitted non-empty release-default {path_key}")
            if sha256_bytes(archive.read(info)) != normalized[hash_key]:
                raise AttachmentError(f"candidate release-default {path_key} hash differs from BUILD-INFO")
    return normalized


def _validate_dependency_distribution_records(distributions: object) -> None:
    """Reject dependency manifests lacking complete installed-RECORD evidence."""

    if not isinstance(distributions, list) or not distributions:
        raise AttachmentError("candidate dependency manifest has no distributions")
    expected_installed_file_keys = {
        "aggregate_sha256",
        "record_document_sha256",
        "record_entry_count",
        "record_sha256_entries_verified",
        "total_size_bytes",
        "unhashed_record_entries",
    }
    for index, distribution in enumerate(distributions):
        description = f"candidate dependency distribution {index}"
        if not isinstance(distribution, dict):
            raise AttachmentError(f"{description} must be an object")
        installed_files = distribution.get("installed_files")
        if (
            not isinstance(installed_files, dict)
            or set(installed_files) != expected_installed_file_keys
        ):
            raise AttachmentError(
                f"{description} has no exact installed_files verification record"
            )
        entry_count = installed_files.get("record_entry_count")
        verified_count = installed_files.get("record_sha256_entries_verified")
        unhashed_count = installed_files.get("unhashed_record_entries")
        total_size = installed_files.get("total_size_bytes")
        if (
            not isinstance(entry_count, int)
            or isinstance(entry_count, bool)
            or entry_count < 2
            or not isinstance(verified_count, int)
            or isinstance(verified_count, bool)
            or verified_count != entry_count - 1
            or not isinstance(unhashed_count, int)
            or isinstance(unhashed_count, bool)
            or unhashed_count != 1
            or not isinstance(total_size, int)
            or isinstance(total_size, bool)
            or total_size <= 0
        ):
            raise AttachmentError(f"{description} has invalid installed RECORD counts")
        for key in ("aggregate_sha256", "record_document_sha256"):
            value = installed_files.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise AttachmentError(f"{description} has an invalid installed-file SHA-256")
        installed_record_sha256 = distribution.get("installed_record_sha256")
        if (
            not isinstance(installed_record_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", installed_record_sha256)
            or installed_files["record_document_sha256"] != installed_record_sha256
        ):
            raise AttachmentError(f"{description} installed RECORD digests disagree")


def validate_and_extract_candidate(
    candidate: Path,
    *,
    expected_sha256: str,
    expected_commit: str,
    extract_directory: Path,
) -> dict[str, Any]:
    if not candidate.is_file() or candidate.name != CUDA_ARCHIVE_NAME:
        raise AttachmentError(f"downloaded artifact must be the single file {CUDA_ARCHIVE_NAME}")
    actual_sha256 = sha256_file(candidate)
    if actual_sha256 != expected_sha256:
        raise AttachmentError(
            f"CUDA ZIP SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    if extract_directory.exists():
        raise AttachmentError(f"refusing to reuse extraction path: {extract_directory}")
    try:
        with zipfile.ZipFile(candidate) as archive:
            entries = _validated_zip_entries(archive)
            bad_entry = archive.testzip()
            if bad_entry is not None:
                raise AttachmentError(f"candidate ZIP CRC validation failed at {bad_entry!r}")
            try:
                build_info = _strict_json_loads(
                    archive.read(entries["ProAim/BUILD-INFO.json"]),
                    "candidate BUILD-INFO.json",
                )
            except KeyError as exc:
                raise AttachmentError("candidate BUILD-INFO.json is invalid") from exc
            if not isinstance(build_info, dict):
                raise AttachmentError("candidate BUILD-INFO.json must contain an object")
            if build_info.get("application") != "ProAim":
                raise AttachmentError("candidate BUILD-INFO.json identifies the wrong application")
            if build_info.get("runtime_variant") != "cuda":
                raise AttachmentError("candidate is not a CUDA runtime bundle")
            if build_info.get("commit") != expected_commit:
                raise AttachmentError("candidate BUILD-INFO commit does not equal the tag commit")
            if build_info.get("dirty") is not False:
                raise AttachmentError("candidate must have been built from a clean Git worktree")
            if build_info.get("schema") != 2:
                raise AttachmentError("candidate BUILD-INFO schema is unsupported")
            release_default_model = _validate_release_default_model_record(
                build_info,
                archive=archive,
                entries=entries,
            )
            dependency_record = build_info.get("dependency_manifest")
            if not isinstance(dependency_record, dict):
                raise AttachmentError("candidate BUILD-INFO has no dependency-manifest record")
            if dependency_record.get("path") != "DEPENDENCY-MANIFEST.json":
                raise AttachmentError("candidate dependency-manifest path is unsupported")
            if dependency_record.get("lock_profile") != "windows-cuda-py313":
                raise AttachmentError("candidate was not built from the Windows CUDA release lock")
            dependency_payload = archive.read(entries["ProAim/DEPENDENCY-MANIFEST.json"])
            dependency_sha256 = sha256_bytes(dependency_payload)
            if dependency_record.get("sha256") != dependency_sha256:
                raise AttachmentError("candidate dependency-manifest SHA-256 does not match BUILD-INFO")
            dependency_manifest = _strict_json_loads(
                dependency_payload, "candidate dependency manifest"
            )
            if not isinstance(dependency_manifest, dict):
                raise AttachmentError("candidate dependency manifest must contain an object")
            if (
                dependency_manifest.get("application") != "ProAim"
                or dependency_manifest.get("schema_version") != 1
                or dependency_manifest.get("runtime_variant") != "cuda"
                or dependency_manifest.get("lock_profile") != "windows-cuda-py313"
            ):
                raise AttachmentError("candidate dependency manifest identifies the wrong lock")
            artifact_contract = dependency_manifest.get("artifact_hash_contract")
            if not isinstance(artifact_contract, dict) or artifact_contract.get(
                "enforced_before_install"
            ) is not True:
                raise AttachmentError("candidate dependencies were not installed from a hash lock")
            distributions = dependency_manifest.get("distributions")
            distribution_count = dependency_record.get("distribution_count")
            if (
                not isinstance(distributions, list)
                or not distributions
                or not isinstance(distribution_count, int)
                or isinstance(distribution_count, bool)
                or distribution_count != len(distributions)
            ):
                raise AttachmentError("candidate dependency distribution count is invalid")
            _validate_dependency_distribution_records(distributions)
            extract_directory.mkdir(parents=True)
            resolved_root = extract_directory.resolve()
            for name, info in entries.items():
                destination = extract_directory.joinpath(*PurePosixPath(name).parts)
                resolved_destination = destination.resolve()
                if resolved_destination != resolved_root and resolved_root not in resolved_destination.parents:
                    raise AttachmentError(f"candidate ZIP entry escaped extraction root: {name!r}")
                if info.is_dir() or info.filename.endswith("/"):
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
    except zipfile.BadZipFile as exc:
        raise AttachmentError("candidate artifact is not a valid ZIP archive") from exc
    model_path = extract_directory / "ProAim" / Path(
        *PurePosixPath(release_default_model["model_path"]).parts
    )
    cli_path = extract_directory / "ProAim" / "ProAimCLI.exe"
    build_info_path = extract_directory / "ProAim" / "BUILD-INFO.json"
    dependency_manifest_path = extract_directory / "ProAim" / "DEPENDENCY-MANIFEST.json"
    helper_path = extract_directory / "ProAim" / "Qualify-ProAimGpu.ps1"
    labels_path = extract_directory / "ProAim" / Path(
        *PurePosixPath(release_default_model["labels_path"]).parts
    )
    if cli_path.stat().st_size <= 0 or model_path.stat().st_size <= 0:
        raise AttachmentError("candidate frozen CLI and release-default model must be non-empty")
    try:
        nvidia_manifest = validate_nvidia_manifest(extract_directory / "ProAim")
    except NvidiaManifestError as exc:
        raise AttachmentError(
            "candidate failed its NVIDIA metadata/license/EULA/native-payload gate: "
            + str(exc)
        ) from exc
    return {
        "filename": CUDA_ARCHIVE_NAME,
        "sha256": actual_sha256,
        "size_bytes": candidate.stat().st_size,
        "build_info": build_info,
        "build_info_sha256": sha256_file(build_info_path),
        "dependency_manifest_sha256": sha256_file(dependency_manifest_path),
        "frozen_cli_sha256": sha256_file(cli_path),
        "qualification_helper_sha256": sha256_file(helper_path),
        "release_default_model": release_default_model,
        "nvidia_redistribution_manifest_sha256": sha256_file(
            extract_directory / "ProAim" / "NVIDIA-REDISTRIBUTION-MANIFEST.json"
        ),
        "nvidia_distribution_versions": {
            str(record["name"]): str(record["version"])
            for record in nvidia_manifest["distributions"]
        },
    }


def _single_downloaded_candidate(directory: Path) -> Path:
    if not directory.is_dir():
        raise AttachmentError(f"downloaded artifact directory not found: {directory}")
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise AttachmentError("downloaded artifact directory contains a symbolic link")
    files = [path for path in directory.rglob("*") if path.is_file()]
    directories = [path for path in directory.rglob("*") if path.is_dir()]
    if len(files) != 1 or files[0].parent != directory or directories:
        raise AttachmentError(
            "downloaded CUDA artifact must contain exactly one file and no nested directories"
        )
    if files[0].name != CUDA_ARCHIVE_NAME:
        raise AttachmentError(
            f"downloaded CUDA artifact contained {files[0].name!r}, expected {CUDA_ARCHIVE_NAME!r}"
        )
    return files[0]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    temporary.replace(path)


def _relative_regular_files(root: Path, *, excluded: frozenset[str] = frozenset()) -> dict[str, Path]:
    if not root.is_dir():
        raise AttachmentError(f"content-manifest root not found: {root}")
    files: dict[str, Path] = {}
    folded: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AttachmentError("content-manifest root contains a symbolic link")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if relative.casefold() in folded:
            raise AttachmentError(f"content-manifest root has duplicate path {relative!r}")
        folded.add(relative.casefold())
        files[relative] = path
    if not files:
        raise AttachmentError("content-manifest root contains no files")
    return files


def _content_file_records(files: Mapping[str, Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for relative, path in sorted(files.items())
    ]


def write_content_manifest(
    *,
    root: Path,
    output: Path,
    kind: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    if output.parent.resolve() != root.resolve():
        raise AttachmentError("content manifest must be written at its manifest root")
    if output.exists():
        raise AttachmentError(f"refusing to overwrite content manifest: {output}")
    filename = output.name
    files = _relative_regular_files(root, excluded=frozenset({filename}))
    manifest = {
        "schema_version": 1,
        "kind": _require_single_line(kind, "content manifest kind", maximum=80),
        "context": dict(context),
        "manifest_file_excluded_from_records": filename,
        "files": _content_file_records(files),
    }
    _write_json_atomic(output, manifest)
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(output),
        "manifest_size_bytes": output.stat().st_size,
    }


def validate_content_manifest(
    *,
    root: Path,
    manifest_name: str,
    expected_sha256: str,
    expected_kind: str,
    expected_context: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = root / manifest_name
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise AttachmentError(f"content manifest is missing: {manifest_name}")
    expected_digest = _normalize_sha256(expected_sha256, f"{expected_kind} manifest SHA-256")
    if sha256_file(manifest_path) != expected_digest:
        raise AttachmentError(f"{expected_kind} manifest SHA-256 mismatch")
    manifest = _read_json_object(manifest_path, f"{expected_kind} content manifest")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != expected_kind:
        raise AttachmentError(f"{expected_kind} content manifest schema or kind mismatch")
    if manifest.get("context") != dict(expected_context):
        raise AttachmentError(f"{expected_kind} content manifest context mismatch")
    if manifest.get("manifest_file_excluded_from_records") != manifest_name:
        raise AttachmentError(f"{expected_kind} content manifest exclusion mismatch")
    records = _records_by_key(
        manifest.get("files"), key="path", description=f"{expected_kind} content files"
    )
    actual_files = _relative_regular_files(root, excluded=frozenset({manifest_name}))
    if set(records) != set(actual_files):
        raise AttachmentError(f"{expected_kind} content manifest file set mismatch")
    for relative, path in actual_files.items():
        record = records[relative]
        if record.get("size_bytes") != path.stat().st_size:
            raise AttachmentError(f"{expected_kind} content size mismatch for {relative}")
        if record.get("sha256") != sha256_file(path):
            raise AttachmentError(f"{expected_kind} content SHA-256 mismatch for {relative}")
    return manifest


def prepare_attachment(
    api: GitHubApi,
    inputs: DispatchInputs,
    *,
    downloaded_directory: Path,
    evidence_downloaded_directory: Path,
    stage_directory: Path,
    extract_directory: Path,
    evidence_extract_directory: Path,
    github_run_id: str,
    github_actor: str,
) -> dict[str, Any]:
    remote = verify_remote_contract(api, inputs)
    candidate = _single_downloaded_candidate(downloaded_directory)
    if stage_directory.exists():
        raise AttachmentError(f"refusing to reuse verified stage path: {stage_directory}")
    candidate_record = validate_and_extract_candidate(
        candidate,
        expected_sha256=inputs.cuda_zip_sha256,
        expected_commit=remote["tag_commit"],
        extract_directory=extract_directory,
    )
    evidence_archive = _single_downloaded_evidence(evidence_downloaded_directory)
    evidence_record = validate_physical_evidence(
        evidence_archive,
        inputs=inputs,
        remote=remote,
        candidate_record=candidate_record,
        extract_directory=evidence_extract_directory,
    )
    stage_directory.mkdir(parents=True)
    staged_candidate = stage_directory / CUDA_ARCHIVE_NAME
    shutil.copyfile(candidate, staged_candidate)
    shutil.copyfile(
        evidence_archive,
        stage_directory / QUALIFICATION_EVIDENCE_ARCHIVE_NAME,
    )
    telemetry = evidence_record["nvidia_telemetry_summary"]
    public_receipt = {
        "schema_version": 1,
        "status": "physically_qualified_cuda_release_candidate",
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": remote["tag_commit"],
        "candidate": {"filename": CUDA_ARCHIVE_NAME, "sha256": inputs.cuda_zip_sha256},
        "evidence_hashes": {
            "archive_sha256": inputs.qualification_evidence_sha256,
            "qualification_manifest_sha256": inputs.qualification_manifest_sha256,
            "physical_attestation_sha256": inputs.physical_attestation_sha256,
        },
        "physical_gpu": {
            "product_name": inputs.qualified_gpu,
            "driver_version": telemetry["driver_version"],
            "compute_capability": telemetry["compute_capability"],
        },
        "qualification_metrics": evidence_record["qualification_metrics"],
        "qualification_run": {
            "id": remote["qualification_run"]["id"],
            "html_url": remote["qualification_run"]["html_url"],
        },
        "measurement_limits": [
            "Model benchmark excludes capture, display, and disk I/O.",
            "Live preview timing measures application work and preview submission, not physical display scanout.",
            "Results establish compatibility and repository policy conformance only on the named GPU/product run.",
        ],
        "privacy": {
            "redacted": True,
            "omitted": ["observer identity", "GPU UUID", "process paths", "raw telemetry"],
            "sensitive_raw_and_sealed_evidence_retention_days": 7,
            "verified_staging_retention_days": 7,
        },
    }
    receipt_path = stage_directory / CUDA_QUALIFICATION_RECEIPT_NAME
    _write_json_atomic(receipt_path, public_receipt)
    attestation: dict[str, Any] = {
        "schema_version": 2,
        "verified_at_utc": _now_utc(),
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": remote["tag_commit"],
        "existing_release_id": remote["release_id"],
        "existing_release_html_url": remote["release_html_url"],
        "source_build": remote["build_run"],
        "source_artifact": remote["build_artifact"],
        "candidate": candidate_record,
        "physical_qualification": {
            **evidence_record,
            "release_typed_confirmation": inputs.confirmation,
            "nvidia_redistribution_typed_confirmation": (
                inputs.nvidia_redistribution_confirmation
            ),
            "review_limit": (
                "Verified physical evidence establishes only this candidate's recorded "
                "runtime behavior on the named GPU. The protected publication reviewer "
                "must independently approve NVIDIA license/EULA terms; artifact and "
                "manifest hashes do not establish redistribution rights."
            ),
        },
        "verification_workflow": {
            "run_id": str(github_run_id),
            "actor": str(github_actor),
            "hosted_smoke": None,
        },
        "public_qualification_receipt": {
            "filename": CUDA_QUALIFICATION_RECEIPT_NAME,
            "sha256": sha256_file(receipt_path),
        },
    }
    _write_json_atomic(stage_directory / "ATTACHMENT-ATTESTATION.json", attestation)
    return attestation


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise AttachmentError(f"{description} is not readable UTF-8: {path}") from exc
    value = _strict_json_loads(payload, description)
    if not isinstance(value, dict):
        raise AttachmentError(f"{description} must contain a JSON object")
    return value


def _finite_number(value: Any, description: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise AttachmentError(f"{description} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AttachmentError(f"{description} must be a finite number") from exc
    if not math.isfinite(number) or number < minimum:
        raise AttachmentError(f"{description} must be finite and at least {minimum}")
    return number


def _utc_datetime(value: Any, description: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise AttachmentError(f"{description} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AttachmentError(f"{description} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise AttachmentError(f"{description} must explicitly use UTC")
    return parsed.astimezone(timezone.utc)


def expected_physical_confirmation(tag: str, gpu_name: str) -> str:
    return f"I ATTEST THAT I OBSERVED {gpu_name} RUN CUDA FOR {tag}"


def _single_downloaded_evidence(directory: Path) -> Path:
    if not directory.is_dir():
        raise AttachmentError(f"downloaded evidence artifact directory not found: {directory}")
    entries = list(directory.iterdir())
    if any(path.is_symlink() for path in entries):
        raise AttachmentError("downloaded evidence artifact contains a symbolic link")
    if len(entries) != 1 or not entries[0].is_file():
        raise AttachmentError(
            "downloaded qualification artifact must contain exactly one top-level file"
        )
    if entries[0].name != QUALIFICATION_EVIDENCE_ARCHIVE_NAME:
        raise AttachmentError(
            "downloaded qualification artifact contained the wrong evidence archive name"
        )
    return entries[0]


def _extract_evidence_archive(archive_path: Path, extract_directory: Path) -> dict[str, Path]:
    if archive_path.name != QUALIFICATION_EVIDENCE_ARCHIVE_NAME or not archive_path.is_file():
        raise AttachmentError("qualification evidence must use the fixed archive name")
    if extract_directory.exists():
        raise AttachmentError(f"refusing to reuse evidence extraction path: {extract_directory}")
    extracted: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > 256:
                raise AttachmentError("qualification evidence archive has an invalid file count")
            total = 0
            folded: set[str] = set()
            for info in infos:
                name = _safe_zip_name(info)
                if not name or "/" in name or info.is_dir() or info.filename.endswith("/"):
                    raise AttachmentError(
                        "qualification evidence archive must contain only flat regular files"
                    )
                if name.casefold() in folded:
                    raise AttachmentError(
                        f"qualification evidence archive contains duplicate file {name!r}"
                    )
                folded.add(name.casefold())
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode) if mode else 0
                if file_type not in (0, stat.S_IFREG) or info.flag_bits & 0x1:
                    raise AttachmentError(
                        f"qualification evidence contains a link, special, or encrypted file {name!r}"
                    )
                total += int(info.file_size)
                if info.file_size < 0 or total > MAX_EVIDENCE_UNCOMPRESSED_BYTES:
                    raise AttachmentError("qualification evidence exceeds its size safety limit")
            names = {info.filename for info in infos}
            missing = REQUIRED_EVIDENCE_FILES.difference(names)
            if missing:
                raise AttachmentError(
                    "qualification evidence omitted required files: "
                    + ", ".join(sorted(missing))
                )
            if names != REQUIRED_EVIDENCE_FILES:
                raise AttachmentError(
                    "qualification evidence contained unexpected files: "
                    + ", ".join(sorted(names.difference(REQUIRED_EVIDENCE_FILES)))
                )
            bad_entry = archive.testzip()
            if bad_entry is not None:
                raise AttachmentError(
                    f"qualification evidence CRC validation failed at {bad_entry!r}"
                )
            extract_directory.mkdir(parents=True)
            for info in infos:
                destination = extract_directory / info.filename
                with archive.open(info) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                extracted[info.filename] = destination
    except zipfile.BadZipFile as exc:
        raise AttachmentError("qualification evidence is not a valid ZIP archive") from exc
    return extracted


def _records_by_key(
    values: Any,
    *,
    key: str,
    description: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        raise AttachmentError(f"{description} must be a JSON array")
    records: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise AttachmentError(f"{description} contains a non-object record")
        identity = value.get(key)
        if not isinstance(identity, str) or not identity or identity in records:
            raise AttachmentError(f"{description} contains a missing or duplicate {key}")
        records[identity] = value
    return records


def _validate_recorded_file(
    record: Mapping[str, Any],
    files: Mapping[str, Path],
    *,
    description: str,
) -> None:
    filename = record.get("file")
    if not isinstance(filename, str) or filename not in files:
        raise AttachmentError(f"{description} references a missing evidence file")
    if record.get("sha256") != sha256_file(files[filename]):
        raise AttachmentError(f"{description} SHA-256 does not match {filename}")
    if "size_bytes" in record and record.get("size_bytes") != files[filename].stat().st_size:
        raise AttachmentError(f"{description} size does not match {filename}")


def _validate_provider_summary(summary: Any, description: str) -> None:
    if not isinstance(summary, dict):
        raise AttachmentError(f"{description} omitted its provider summary")
    if summary.get("requested_provider") != "CUDAExecutionProvider":
        raise AttachmentError(f"{description} did not request CUDAExecutionProvider")
    active = summary.get("active_providers")
    if not isinstance(active, list) or "CUDAExecutionProvider" not in active:
        raise AttachmentError(f"{description} did not activate CUDAExecutionProvider")
    if summary.get("require_full_provider") is not True:
        raise AttachmentError(f"{description} omitted require_full_provider=true")
    configured = summary.get("configured_session_options")
    if not isinstance(configured, dict) or configured.get("disable_cpu_ep_fallback") is not True:
        raise AttachmentError(
            f"{description} did not configure disable_cpu_ep_fallback=true"
        )
    if summary.get("runtime_ep_fail_fallback_disabled") is not True:
        raise AttachmentError(
            f"{description} did not disable ONNX Runtime EPFail fallback"
        )
    if summary.get("provider_options_status") != "ok":
        raise AttachmentError(f"{description} could not report CUDA provider options")
    provider_options = summary.get("provider_options")
    cuda_options = (
        provider_options.get("CUDAExecutionProvider")
        if isinstance(provider_options, dict)
        else None
    )
    if not isinstance(cuda_options, dict) or str(cuda_options.get("device_id")) != "0":
        raise AttachmentError(
            f"{description} did not bind CUDA device_id=0 on the dedicated single-GPU runner"
        )


def _validate_cuda_runner_invariant(path: Path, expected_gpu: str) -> dict[str, Any]:
    record = _read_json_object(path, "CUDA runner invariant")
    if record.get("schema_version") != 1 or record.get("status") != "passed_before_gpu_runs":
        raise AttachmentError("CUDA runner invariant is incomplete or unsupported")
    _utc_datetime(record.get("checked_at_utc"), "CUDA runner invariant timestamp")
    expected = {
        "nvidia_gpu_count": 1,
        "nvidia_gpu_names": [expected_gpu],
        "preexisting_proaim_cli_count": 0,
        "cuda_visible_devices": None,
        "nvidia_visible_devices": None,
        "telemetry_interval_milliseconds": CUDA_QUALIFICATION_POLICY[
            "telemetry_interval_milliseconds"
        ],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AttachmentError(f"CUDA runner invariant {key} mismatch")
    return record


def _validate_physical_timeline(
    runner_invariant: Mapping[str, Any],
    run_intervals: Sequence[tuple[str, datetime, datetime]],
    observed_at_utc: Any,
) -> None:
    if not run_intervals:
        raise AttachmentError("physical evidence has no frozen CLI run timeline")
    checked = _utc_datetime(
        runner_invariant.get("checked_at_utc"), "CUDA runner invariant timestamp"
    )
    first_started = run_intervals[0][1]
    final_completed = run_intervals[-1][2]
    observed = _utc_datetime(observed_at_utc, "physical observation timestamp")
    if checked > first_started or first_started - checked > timedelta(minutes=10):
        raise AttachmentError("CUDA runner invariant was not recorded immediately before GPU runs")
    if observed < final_completed:
        raise AttachmentError("physical observation was recorded before automated GPU runs completed")


def _validate_timing_summary(
    summary: Any,
    description: str,
    *,
    expected_samples: int | None = None,
) -> dict[str, float | int]:
    if not isinstance(summary, dict):
        raise AttachmentError(f"{description} is missing")
    if expected_samples is not None and summary.get("samples") != expected_samples:
        raise AttachmentError(f"{description} has the wrong sample count")
    normalized: dict[str, float | int] = {}
    for key in ("mean", "p50", "median", "p95", "p99", "min", "max", "stdev"):
        normalized[key] = _finite_number(summary.get(key), f"{description} {key}")
    if expected_samples is not None:
        normalized["samples"] = expected_samples
    if not (
        float(normalized["min"])
        <= float(normalized["p50"])
        <= float(normalized["p95"])
        <= float(normalized["p99"])
        <= float(normalized["max"])
    ):
        raise AttachmentError(f"{description} percentile ordering is inconsistent")
    if not (
        float(normalized["min"])
        <= float(normalized["mean"])
        <= float(normalized["max"])
    ) or not math.isclose(
        float(normalized["p50"]),
        float(normalized["median"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise AttachmentError(f"{description} summary is internally inconsistent")
    return normalized


def _require_close_timing(
    actual: float,
    expected: float,
    description: str,
    *,
    absolute_tolerance_ms: float = 0.1,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=0.01,
        abs_tol=absolute_tolerance_ms,
    ):
        raise AttachmentError(f"{description} is internally inconsistent")


def _require_bundle_path_suffix(value: Any, relative_path: str, description: str) -> None:
    actual = _require_single_line(str(value or ""), description, maximum=4096).replace(
        "\\", "/"
    )
    expected = relative_path.casefold()
    folded = actual.casefold()
    if folded != expected and not folded.endswith("/" + expected):
        raise AttachmentError(f"{description} differs from the release-default contract")


def _validate_software_evidence(
    files: Mapping[str, Path],
    *,
    candidate_record: Mapping[str, Any],
    expected_commit: str,
    expected_cuda_sha256: str,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, datetime, datetime]]]:
    manifest_path = files.get(QUALIFICATION_MANIFEST_NAME)
    if manifest_path is None:
        raise AttachmentError("software evidence omitted qualification-manifest.json")
    if expected_manifest_sha256 is not None and sha256_file(manifest_path) != expected_manifest_sha256:
        raise AttachmentError("qualification manifest SHA-256 does not match the dispatch input")
    manifest = _read_json_object(manifest_path, "qualification manifest")
    if manifest.get("schema_version") != 1:
        raise AttachmentError("qualification manifest schema is unsupported")
    if manifest.get("status") != "software_checks_passed_physical_gpu_confirmation_pending":
        raise AttachmentError("qualification manifest has the wrong software-check status")
    if manifest.get("qualified") is not False:
        raise AttachmentError(
            "qualification helper must remain qualified=false; physical completion is separate"
        )
    provider = manifest.get("provider")
    if not isinstance(provider, dict) or provider != {
        "selection": "CUDA",
        "requested_device": "CUDA",
        "expected_execution_provider": "CUDAExecutionProvider",
        "directml_adapter_index": None,
    }:
        raise AttachmentError("qualification manifest is not an exact CUDA-provider run")
    build_info = manifest.get("bundle_build_info")
    if not isinstance(build_info, dict) or build_info != candidate_record.get("build_info"):
        raise AttachmentError("qualification manifest BUILD-INFO differs from the candidate")
    if build_info.get("commit") != expected_commit:
        raise AttachmentError("qualification manifest commit differs from the tag commit")
    release_default_model = _validate_release_default_model_record(build_info)
    if release_default_model != candidate_record.get("release_default_model"):
        raise AttachmentError("qualification manifest release-default model differs from candidate")
    if _read_json_object(files["bundle-BUILD-INFO.json"], "evidence BUILD-INFO") != build_info:
        raise AttachmentError("evidence BUILD-INFO copy differs from the manifest")

    confirmation = manifest.get("manual_confirmation")
    if not isinstance(confirmation, dict) or confirmation.get("required") is not True:
        raise AttachmentError("qualification manifest did not require manual confirmation")
    if confirmation.get("completed_by_helper") is not False:
        raise AttachmentError("qualification helper improperly claimed manual completion")
    live_bounds = manifest.get("live_bounds")
    if not isinstance(live_bounds, dict) or live_bounds.get("enabled") is not True:
        raise AttachmentError("CUDA release qualification must include bounded live passes")
    if live_bounds.get("selected_model") != "release-default":
        raise AttachmentError("CUDA release qualification used an unapproved live model")
    if live_bounds.get("release_default_model") != release_default_model:
        raise AttachmentError("CUDA live bounds do not bind the release-default model")
    if set(live_bounds.get("modes") or []) != {"no-preview", "preview-15"}:
        raise AttachmentError("CUDA release qualification omitted a required live preview mode")

    artifacts = _records_by_key(
        manifest.get("input_artifacts"), key="role", description="input artifacts"
    )
    expected_artifacts = {
        "frozen_cli": "frozen_cli_sha256",
        "build_info": "build_info_sha256",
        "dependency_manifest": "dependency_manifest_sha256",
        "qualification_helper": "qualification_helper_sha256",
        "original_bundle_archive": "sha256",
    }
    for role, candidate_key in expected_artifacts.items():
        record = artifacts.get(role)
        if not isinstance(record, dict) or record.get("sha256") != candidate_record.get(candidate_key):
            raise AttachmentError(f"qualification input artifact {role!r} differs from candidate")
    for role, hash_key, path_key in (
        ("release_default_model", "model_sha256", "model_path"),
        ("release_default_labels", "labels_sha256", "labels_path"),
    ):
        record = artifacts.get(role)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != release_default_model[hash_key]
            or record.get("path") != release_default_model[path_key]
            or record.get("location") != "bundle"
        ):
            raise AttachmentError(f"qualification input artifact {role!r} differs from candidate")
    if artifacts["original_bundle_archive"].get("sha256") != expected_cuda_sha256:
        raise AttachmentError("qualification manifest binds a different CUDA archive")

    software_file_names = (
        REQUIRED_EVIDENCE_FILES
        - {
            PHYSICAL_ATTESTATION_NAME,
            TASK_MANAGER_CONFIRMATION_NAME,
            NVIDIA_TELEMETRY_NAME,
            CUDA_RUNNER_INVARIANT_NAME,
            LOCAL_PHYSICAL_OBSERVATION_NAME,
            RAW_CANDIDATE_RECORD_NAME,
            RAW_SOURCE_RECORD_NAME,
            RAW_CONTENT_MANIFEST_NAME,
            QUALIFICATION_MANIFEST_NAME,
        }
    )
    evidence_records = _records_by_key(
        manifest.get("evidence_files"), key="file", description="software evidence files"
    )
    if set(evidence_records) != software_file_names:
        raise AttachmentError("qualification manifest did not bind the exact software evidence set")
    for filename, record in evidence_records.items():
        _validate_recorded_file(record, files, description=f"software evidence {filename}")

    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 4:
        raise AttachmentError("qualification manifest must contain runtime, benchmark, and two live runs")
    run_intervals: list[tuple[str, datetime, datetime]] = []
    previous_completed: datetime | None = None
    for run in runs:
        if not isinstance(run, dict) or run.get("exit_code") != 0:
            raise AttachmentError("qualification manifest contains a failed or malformed run")
        for record_key in ("stdout", "stderr"):
            record = run.get(record_key)
            if not isinstance(record, dict):
                raise AttachmentError(f"qualification run omitted {record_key} identity")
            _validate_recorded_file(record, files, description=f"qualification run {record_key}")
        if "metrics" in run:
            if not isinstance(run["metrics"], dict):
                raise AttachmentError("qualification live run metrics identity is malformed")
            _validate_recorded_file(run["metrics"], files, description="qualification live metrics")
        started = _utc_datetime(run.get("started_at_utc"), "qualification run start")
        completed = _utc_datetime(run.get("completed_at_utc"), "qualification run completion")
        if completed < started:
            raise AttachmentError("qualification run completed before it started")
        if previous_completed is not None and started < previous_completed:
            raise AttachmentError("qualification run time windows overlap or are out of order")
        previous_completed = completed
        run_intervals.append((str(run.get("name") or ""), started, completed))
    expected_run_names = [
        "frozen runtime info",
        "model benchmark (release-default)",
        "live pipeline (release-default-no-preview)",
        "live pipeline (release-default-preview-15)",
    ]
    if [name for name, _, _ in run_intervals] != expected_run_names:
        raise AttachmentError("qualification manifest has the wrong run names or order")
    intervals_by_name = {
        name: (started, completed) for name, started, completed in run_intervals
    }

    runtime = _read_json_object(files["runtime-info.json"], "frozen CUDA runtime report")
    if runtime.get("frozen") is not True:
        raise AttachmentError("CUDA runtime report did not come from the frozen CLI")
    providers = runtime.get("onnxruntime_providers")
    if not isinstance(providers, list) or "CUDAExecutionProvider" not in providers:
        raise AttachmentError("frozen CUDA runtime report omitted CUDAExecutionProvider")

    benchmark = _read_json_object(
        files["benchmark-release-default.json"], "CUDA model benchmark"
    )
    benchmark_generated = _utc_datetime(
        benchmark.get("generated_at_utc"), "CUDA benchmark generation time"
    )
    benchmark_started, benchmark_completed = intervals_by_name[
        "model benchmark (release-default)"
    ]
    if not benchmark_started <= benchmark_generated <= benchmark_completed:
        raise AttachmentError("CUDA benchmark generation time is outside its frozen CLI run")
    methodology = benchmark.get("methodology")
    models = benchmark.get("models")
    if (
        not isinstance(methodology, dict)
        or methodology.get("backend") != "onnxruntime"
        or methodology.get("requested_device") != "CUDA"
    ):
        raise AttachmentError("CUDA benchmark did not request the CUDA device")
    if methodology.get("require_full_provider") is not True:
        raise AttachmentError("CUDA benchmark did not enable full-provider enforcement")
    benchmark_bounds = manifest.get("benchmark_bounds")
    if not isinstance(benchmark_bounds, dict):
        raise AttachmentError("qualification manifest omitted benchmark bounds")
    sample_count = int(benchmark_bounds.get("samples") or 0)
    iterations = int(methodology.get("iterations_per_repeat") or 0)
    repeats = int(methodology.get("repeats") or 0)
    warmup = int(methodology.get("warmup_per_model") or -1)
    expected_benchmark = {
        "samples": CUDA_QUALIFICATION_POLICY["benchmark_samples"],
        "warmup": CUDA_QUALIFICATION_POLICY["benchmark_warmup"],
        "iterations": CUDA_QUALIFICATION_POLICY["benchmark_iterations_per_repeat"],
        "repeats": CUDA_QUALIFICATION_POLICY["benchmark_repeats"],
    }
    actual_benchmark = {
        "samples": sample_count,
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
    }
    if actual_benchmark != expected_benchmark or benchmark_bounds != expected_benchmark:
        raise AttachmentError("CUDA benchmark dimensions differ from repository release policy")
    benchmark_input = benchmark.get("input")
    if (
        not isinstance(benchmark_input, dict)
        or benchmark_input.get("kind") != "synthetic"
        or benchmark_input.get("generator")
        != "numpy.default_rng(seed=0), uint8 720x1280 BGR"
        or int(benchmark_input.get("count") or 0) != sample_count
    ):
        raise AttachmentError("CUDA benchmark input sample count differs from declared bounds")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise AttachmentError("CUDA benchmark must report exactly one release-default model")
    model = models[0]
    if (
        model.get("key") != "release-default"
        or model.get("input_shape_hw") != release_default_model["input_shape_hw"]
    ):
        raise AttachmentError("CUDA benchmark did not use the release-default key and shape")
    _validate_provider_summary(model.get("runtime"), "CUDA benchmark")
    artifact = model.get("artifact")
    artifact_files = artifact.get("files") if isinstance(artifact, dict) else None
    if (
        not isinstance(artifact_files, list)
        or len(artifact_files) != 1
        or not isinstance(artifact_files[0], dict)
        or artifact_files[0].get("sha256") != release_default_model["model_sha256"]
    ):
        raise AttachmentError("CUDA benchmark fingerprinted a different release-default model")
    _require_bundle_path_suffix(
        artifact_files[0].get("resolved_path"),
        release_default_model["model_path"],
        "CUDA benchmark resolved model path",
    )
    labels_artifact = model.get("labels_artifact")
    label_files = labels_artifact.get("files") if isinstance(labels_artifact, dict) else None
    if (
        not isinstance(label_files, list)
        or len(label_files) != 1
        or not isinstance(label_files[0], dict)
        or label_files[0].get("sha256") != release_default_model["labels_sha256"]
    ):
        raise AttachmentError("CUDA benchmark fingerprinted different release-default labels")
    _require_bundle_path_suffix(
        label_files[0].get("resolved_path"),
        release_default_model["labels_path"],
        "CUDA benchmark resolved labels path",
    )
    expected_timed_samples = iterations * repeats
    timing_ms = model.get("timing_ms")
    if not isinstance(timing_ms, dict):
        raise AttachmentError("CUDA benchmark omitted timing summaries")
    benchmark_timings: dict[str, dict[str, float | int]] = {}
    for name in ("preprocess", "inference", "postprocess", "pipeline"):
        benchmark_timings[name] = _validate_timing_summary(
            timing_ms.get(name),
            f"CUDA benchmark {name} timing",
            expected_samples=expected_timed_samples,
        )
    _require_close_timing(
        float(benchmark_timings["pipeline"]["mean"]),
        sum(
            float(benchmark_timings[name]["mean"])
            for name in ("preprocess", "inference", "postprocess")
        ),
        "CUDA benchmark mean pipeline/component timing",
    )
    repeat_records = model.get("repeats")
    if not isinstance(repeat_records, list) or len(repeat_records) != repeats:
        raise AttachmentError("CUDA benchmark omitted exact repeat-level timing evidence")
    repeat_timings: list[dict[str, dict[str, float | int]]] = []
    for repeat_number, repeat_record in enumerate(repeat_records, 1):
        if not isinstance(repeat_record, dict) or repeat_record.get("repeat") != repeat_number:
            raise AttachmentError("CUDA benchmark repeat identities are incomplete or out of order")
        repeat_payload = repeat_record.get("timing_ms")
        if not isinstance(repeat_payload, dict) or set(repeat_payload) != {
            "preprocess",
            "inference",
            "postprocess",
            "pipeline",
        }:
            raise AttachmentError("CUDA benchmark repeat timing fields are incomplete")
        normalized_repeat = {
            name: _validate_timing_summary(
                repeat_payload[name],
                f"CUDA benchmark repeat {repeat_number} {name} timing",
                expected_samples=iterations,
            )
            for name in ("preprocess", "inference", "postprocess", "pipeline")
        }
        _require_close_timing(
            float(normalized_repeat["pipeline"]["mean"]),
            sum(
                float(normalized_repeat[name]["mean"])
                for name in ("preprocess", "inference", "postprocess")
            ),
            f"CUDA benchmark repeat {repeat_number} mean pipeline/component timing",
        )
        if float(normalized_repeat["inference"]["p95"]) > CUDA_QUALIFICATION_POLICY[
            "benchmark_max_p95_inference_ms"
        ]:
            raise AttachmentError(
                f"CUDA benchmark repeat {repeat_number} p95 inference latency exceeds release policy"
            )
        repeat_timings.append(normalized_repeat)
    for name in ("preprocess", "inference", "postprocess", "pipeline"):
        aggregate_summary = benchmark_timings[name]
        _require_close_timing(
            float(aggregate_summary["mean"]),
            sum(float(repeat[name]["mean"]) for repeat in repeat_timings) / repeats,
            f"CUDA benchmark aggregate/repeat {name} mean",
            absolute_tolerance_ms=1e-9,
        )
        if not math.isclose(
            float(aggregate_summary["min"]),
            min(float(repeat[name]["min"]) for repeat in repeat_timings),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ) or not math.isclose(
            float(aggregate_summary["max"]),
            max(float(repeat[name]["max"]) for repeat in repeat_timings),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise AttachmentError(f"CUDA benchmark aggregate/repeat {name} extrema differ")
    pipeline_fps = _finite_number(
        model.get("pipeline_fps_from_mean"), "CUDA benchmark pipeline FPS"
    )
    _require_close_timing(
        pipeline_fps,
        1000.0 / float(benchmark_timings["pipeline"]["mean"]),
        "CUDA benchmark pipeline FPS",
        absolute_tolerance_ms=1e-6,
    )
    benchmark_p95 = float(benchmark_timings["inference"]["p95"])
    if benchmark_p95 > CUDA_QUALIFICATION_POLICY["benchmark_max_p95_inference_ms"]:
        raise AttachmentError("CUDA benchmark p95 inference latency exceeds release policy")

    live_metrics: dict[str, dict[str, Any]] = {}

    for filename, expected_preview in (
        ("live-release-default-no-preview.json", False),
        ("live-release-default-preview-15.json", True),
    ):
        report = _read_json_object(files[filename], f"CUDA live report {filename}")
        run_name = (
            "live pipeline (release-default-preview-15)"
            if expected_preview
            else "live pipeline (release-default-no-preview)"
        )
        run_started, run_completed = intervals_by_name[run_name]
        report_started = _utc_datetime(
            report.get("started_utc"), f"CUDA live report {filename} start"
        )
        report_completed = _utc_datetime(
            report.get("completed_utc"), f"CUDA live report {filename} completion"
        )
        if report_completed < report_started or not (
            run_started <= report_started <= report_completed <= run_completed
        ):
            raise AttachmentError(f"CUDA live report {filename} times are outside its frozen CLI run")
        _validate_provider_summary(report.get("detector_runtime"), f"CUDA live report {filename}")
        config = report.get("config")
        if not isinstance(config, dict) or config.get("require_full_provider") is not True:
            raise AttachmentError(f"CUDA live report {filename} omitted full-provider enforcement")
        config_source = config.get("source")
        config_capture = config.get("capture")
        config_preview = config.get("preview")
        if (
            config.get("backend") != "onnxruntime"
            or config.get("device") != "CUDA"
            or not isinstance(config_source, dict)
            or config_source.get("kind") != "screen"
            or config_source.get("value") is not None
            or not isinstance(config_capture, dict)
            or config_capture.get("screen_region") is not None
            or config_capture.get("screen_monitor") != live_bounds.get("screen_monitor")
            or config_capture.get("screen_fps") != live_bounds.get("screen_fps")
            or not isinstance(config_preview, dict)
            or config_preview.get("enabled") is not expected_preview
        ):
            raise AttachmentError(f"CUDA live report {filename} used a different screen workload")
        if expected_preview and config_preview.get("fps_limit") != 15.0:
            raise AttachmentError(f"CUDA live report {filename} did not request preview at 15 FPS")
        inference_config = config.get("inference")
        if (
            not isinstance(inference_config, dict)
            or inference_config.get("shape_hw") != release_default_model["input_shape_hw"]
            or inference_config.get("crop_size") is not None
            or inference_config.get("detail_crop_size") is not None
            or live_bounds.get("detail_crop_size") is not None
        ):
            raise AttachmentError(f"CUDA live report {filename} used a different full-frame workload")
        model_artifact = report.get("model_artifact")
        if (
            not isinstance(model_artifact, dict)
            or model_artifact.get("sha256") != release_default_model["model_sha256"]
        ):
            raise AttachmentError(f"CUDA live report {filename} used a different model")
        labels_artifact = report.get("labels_artifact")
        if (
            not isinstance(labels_artifact, dict)
            or labels_artifact.get("sha256") != release_default_model["labels_sha256"]
        ):
            raise AttachmentError(f"CUDA live report {filename} used different labels")
        preview = report.get("preview")
        if not isinstance(preview, dict) or preview.get("enabled") is not expected_preview:
            raise AttachmentError(f"CUDA live report {filename} recorded the wrong preview state")
        preview_stats = preview.get("stats")
        submitted: int | None = None
        displayed: int | None = None
        replaced: int | None = None
        if expected_preview:
            if preview.get("fps_limit") != 15.0 or preview.get("mode") == "disabled":
                raise AttachmentError(f"CUDA live report {filename} did not run preview at 15 FPS")
            if not isinstance(preview_stats, dict) or set(preview_stats) != {
                "submitted_frames",
                "displayed_frames",
                "replaced_frames",
            }:
                raise AttachmentError(f"CUDA live report {filename} omitted preview activity counters")
            submitted = preview_stats.get("submitted_frames")
            displayed = preview_stats.get("displayed_frames")
            replaced = preview_stats.get("replaced_frames")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (submitted, displayed, replaced)
            ) or not (0 < displayed <= submitted):
                raise AttachmentError(f"CUDA live report {filename} has impossible preview activity")
        elif preview.get("mode") != "disabled" or preview_stats != {}:
            raise AttachmentError(f"CUDA live report {filename} performed unexpected preview work")
        detail_pass = report.get("detail_pass")
        if (
            not isinstance(detail_pass, dict)
            or detail_pass.get("enabled") is not False
            or detail_pass.get("requested_crop_size") is not None
        ):
            raise AttachmentError(f"CUDA live report {filename} used an unapproved detail pass")
        source = report.get("source")
        if not isinstance(source, dict) or source.get("backend") != "dxcam-dxgi":
            raise AttachmentError(f"CUDA live report {filename} did not use DXcam/DXGI")
        if source.get("fallback_reason") not in (None, ""):
            raise AttachmentError(f"CUDA live report {filename} used a capture fallback")
        capture = report.get("capture")
        pipeline = report.get("pipeline")
        termination = report.get("termination")
        if not isinstance(capture, dict) or capture.get("read_failures") != 0:
            raise AttachmentError(f"CUDA live report {filename} recorded capture read failures")
        if not isinstance(pipeline, dict):
            raise AttachmentError(f"CUDA live report {filename} omitted pipeline metrics")
        processed_frames = int(pipeline.get("processed_frames") or 0)
        if expected_preview and (
            submitted is None or submitted > processed_frames
        ):
            raise AttachmentError(f"CUDA live report {filename} has impossible preview activity")
        rolling_samples = int(pipeline.get("rolling_sample_count") or 0)
        elapsed_fps = _finite_number(
            pipeline.get("elapsed_fps"), f"CUDA live report {filename} elapsed FPS"
        )
        update_fps = _finite_number(
            pipeline.get("update_fps"), f"CUDA live report {filename} update FPS"
        )
        elapsed_seconds = _finite_number(
            pipeline.get("elapsed_seconds"), f"CUDA live report {filename} elapsed seconds"
        )
        if elapsed_seconds < CUDA_QUALIFICATION_POLICY["live_min_elapsed_seconds"]:
            raise AttachmentError(f"CUDA live report {filename} measured too little elapsed time")
        report_duration = (report_completed - report_started).total_seconds()
        if abs(report_duration - elapsed_seconds) > 2.0:
            raise AttachmentError(f"CUDA live report {filename} elapsed time is internally inconsistent")
        calculated_fps = processed_frames / elapsed_seconds
        if not math.isclose(calculated_fps, elapsed_fps, rel_tol=0.02, abs_tol=0.5):
            raise AttachmentError(f"CUDA live report {filename} elapsed FPS is internally inconsistent")
        if processed_frames < CUDA_QUALIFICATION_POLICY["live_min_processed_frames"]:
            raise AttachmentError(f"CUDA live report {filename} processed too few frames")
        stats_window = int(config.get("stats_window") or 0)
        if rolling_samples != min(processed_frames, stats_window):
            raise AttachmentError(f"CUDA live report {filename} rolling sample count is inconsistent")
        if rolling_samples < CUDA_QUALIFICATION_POLICY["live_min_processed_frames"]:
            raise AttachmentError(f"CUDA live report {filename} has too few timing samples")
        if elapsed_fps < CUDA_QUALIFICATION_POLICY["live_min_elapsed_fps"]:
            raise AttachmentError(f"CUDA live report {filename} elapsed FPS is below policy")
        if update_fps < CUDA_QUALIFICATION_POLICY["live_min_update_fps"]:
            raise AttachmentError(f"CUDA live report {filename} update FPS is below policy")
        live_timings = pipeline.get("timings")
        if (
            not isinstance(live_timings, dict)
            or live_timings.get("unit") != "milliseconds"
            or live_timings.get("fields") != list(LIVE_TIMING_FIELDS)
        ):
            raise AttachmentError(f"CUDA live report {filename} has the wrong timing schema")
        timing_sections: dict[str, dict[str, float]] = {}
        for percentile in ("mean", "p50", "p95", "p99"):
            values = live_timings.get(percentile)
            if not isinstance(values, dict) or set(values) != set(LIVE_TIMING_FIELDS):
                raise AttachmentError(
                    f"CUDA live report {filename} {percentile} timing fields are incomplete"
                )
            timing_sections[percentile] = {
                key: _finite_number(
                    values[key], f"CUDA live report {filename} {percentile} {key}"
                )
                for key in LIVE_TIMING_FIELDS
            }
        normalized_p95 = timing_sections["p95"]
        for field in LIVE_TIMING_FIELDS:
            if not (
                timing_sections["p50"][field]
                <= timing_sections["p95"][field]
                <= timing_sections["p99"][field]
            ):
                raise AttachmentError(
                    f"CUDA live report {filename} timing percentile ordering is inconsistent"
                )
        mean_timing = timing_sections["mean"]
        processing_components = (
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "detail_preprocess_ms",
            "detail_inference_ms",
            "detail_postprocess_ms",
            "control_ms",
        )
        _require_close_timing(
            mean_timing["processing_ms"],
            sum(mean_timing[field] for field in processing_components),
            f"CUDA live report {filename} mean processing/component timing",
        )
        _require_close_timing(
            mean_timing["freshness_latency_ms"],
            mean_timing["queue_age_ms"] + mean_timing["processing_ms"],
            f"CUDA live report {filename} mean freshness timing",
        )
        _require_close_timing(
            mean_timing["observed_pipeline_ms"],
            mean_timing["capture_ms"] + mean_timing["freshness_latency_ms"],
            f"CUDA live report {filename} mean observed-pipeline timing",
        )
        for percentile in ("mean", "p50", "p95", "p99"):
            if any(
                timing_sections[percentile][field] != 0.0
                for field in (
                    "detail_preprocess_ms",
                    "detail_inference_ms",
                    "detail_postprocess_ms",
                )
            ):
                raise AttachmentError(
                    f"CUDA live report {filename} contains detail timing while detail is disabled"
                )
        if expected_preview:
            if mean_timing["preview_service_ms"] <= 0.0:
                raise AttachmentError(f"CUDA live report {filename} recorded no preview service work")
        elif any(
            timing_sections[percentile]["preview_service_ms"] != 0.0
            for percentile in ("mean", "p50", "p95", "p99")
        ):
            raise AttachmentError(f"CUDA live report {filename} recorded unexpected preview timing")
        observed_p95 = normalized_p95.get("observed_pipeline_ms")
        freshness_p95 = normalized_p95.get("freshness_latency_ms")
        if observed_p95 is None or observed_p95 > CUDA_QUALIFICATION_POLICY[
            "live_max_p95_observed_pipeline_ms"
        ]:
            raise AttachmentError(f"CUDA live report {filename} p95 pipeline latency exceeds policy")
        if freshness_p95 is None or freshness_p95 > CUDA_QUALIFICATION_POLICY[
            "live_max_p95_freshness_latency_ms"
        ]:
            raise AttachmentError(f"CUDA live report {filename} p95 freshness latency exceeds policy")
        if not isinstance(termination, dict) or termination.get("reason") not in {
            "max_frames",
            "max_seconds",
        }:
            raise AttachmentError(f"CUDA live report {filename} was not bounded normally")
        requested_seconds = _finite_number(
            termination.get("requested_max_seconds"),
            f"CUDA live report {filename} requested maximum seconds",
            minimum=1.0,
        )
        requested_frames = termination.get("requested_max_frames")
        expected_seconds = CUDA_QUALIFICATION_POLICY["live_max_seconds"]
        expected_frames = CUDA_QUALIFICATION_POLICY["live_requested_max_frames"]
        if (
            requested_seconds != expected_seconds
            or requested_frames != expected_frames
            or live_bounds.get("max_seconds") != expected_seconds
            or live_bounds.get("max_frames") != expected_frames
            or stats_window != expected_frames
        ):
            raise AttachmentError(f"CUDA live report {filename} bounds differ from release policy")
        reason = termination["reason"]
        if reason == "max_frames" and processed_frames != requested_frames:
            raise AttachmentError(f"CUDA live report {filename} did not reach its claimed frame bound")
        if reason == "max_seconds" and not math.isclose(
            elapsed_seconds, requested_seconds, rel_tol=0.0, abs_tol=2.0
        ):
            raise AttachmentError(f"CUDA live report {filename} did not reach its claimed time bound")
        if processed_frames > requested_frames or elapsed_seconds > requested_seconds + 2.0:
            raise AttachmentError(f"CUDA live report {filename} exceeded its configured bound")
        live_metrics[filename] = {
            "processed_frames": processed_frames,
            "rolling_sample_count": rolling_samples,
            "elapsed_fps": elapsed_fps,
            "update_fps": update_fps,
            "p95_observed_pipeline_ms": observed_p95,
            "p95_freshness_latency_ms": freshness_p95,
            "preview_enabled": expected_preview,
            "measurement_limit": "preview submission measured; display scanout is not measured",
        }
    public_metrics = {
        "policy": dict(CUDA_QUALIFICATION_POLICY),
        "benchmark": {
            "timed_samples": expected_timed_samples,
            "p95_inference_ms": benchmark_p95,
        },
        "live": live_metrics,
    }
    return manifest, public_metrics, run_intervals


def _parse_nvidia_telemetry(
    path: Path,
    expected_gpu: str,
    *,
    required_intervals: Sequence[tuple[str, datetime, datetime]],
) -> dict[str, Any]:
    gpu_records: list[dict[str, Any]] = []
    process_records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AttachmentError("NVIDIA telemetry is not readable UTF-8") from exc
    previous_timestamp: datetime | None = None
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        record = _strict_json_loads(line, f"NVIDIA telemetry line {line_number}")
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise AttachmentError(f"NVIDIA telemetry line {line_number} has an invalid schema")
        captured = _utc_datetime(
            record.get("captured_at_utc"),
            f"NVIDIA telemetry line {line_number} timestamp",
        )
        if previous_timestamp is not None and captured < previous_timestamp:
            raise AttachmentError("NVIDIA telemetry timestamps are out of order")
        previous_timestamp = captured
        record["_captured_datetime"] = captured
        if record.get("kind") == "gpu":
            if isinstance(record.get("gpu_index"), bool) or not isinstance(
                record.get("gpu_index"), int
            ) or record["gpu_index"] < 0:
                raise AttachmentError(f"NVIDIA telemetry line {line_number} has an invalid GPU index")
            for field in (
                "nvidia_timestamp",
                "gpu_name",
                "gpu_uuid",
                "driver_version",
                "compute_capability",
            ):
                _require_single_line(
                    str(record.get(field) or ""),
                    f"NVIDIA telemetry line {line_number} {field}",
                )
            utilization = _finite_number(
                record.get("utilization_gpu_percent"),
                f"NVIDIA telemetry line {line_number} utilization",
            )
            used = _finite_number(
                record.get("memory_used_mib"),
                f"NVIDIA telemetry line {line_number} used memory",
            )
            total = _finite_number(
                record.get("memory_total_mib"),
                f"NVIDIA telemetry line {line_number} total memory",
                minimum=1.0,
            )
            if utilization > 100.0 or used > total:
                raise AttachmentError(f"NVIDIA telemetry line {line_number} has out-of-range GPU data")
            gpu_records.append(record)
        elif record.get("kind") == "compute_process":
            _require_single_line(
                str(record.get("gpu_uuid") or ""),
                f"NVIDIA telemetry line {line_number} GPU UUID",
            )
            pid = record.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise AttachmentError(f"NVIDIA telemetry line {line_number} has an invalid PID")
            process_name = _require_single_line(
                str(record.get("process_name") or ""),
                f"NVIDIA telemetry line {line_number} process name",
            )
            if "/" in process_name or "\\" in process_name:
                raise AttachmentError("NVIDIA telemetry process names must be privacy-safe basenames")
            memory_supported = record.get("used_gpu_memory_supported")
            if not isinstance(memory_supported, bool):
                raise AttachmentError(
                    f"NVIDIA telemetry line {line_number} omitted process-memory support state"
                )
            if memory_supported:
                _finite_number(
                    record.get("used_gpu_memory_mib"),
                    f"NVIDIA telemetry line {line_number} process GPU memory",
                )
            elif record.get("used_gpu_memory_mib") is not None:
                raise AttachmentError(
                    f"NVIDIA telemetry line {line_number} has inconsistent unavailable process memory"
                )
            process_records.append(record)
        else:
            raise AttachmentError(f"NVIDIA telemetry line {line_number} has an unknown record kind")
    target = [record for record in gpu_records if record.get("gpu_name") == expected_gpu]
    if len(target) < 3:
        raise AttachmentError("NVIDIA telemetry contains fewer than three samples for the named GPU")
    uuids = {str(record.get("gpu_uuid") or "").strip() for record in target}
    drivers = {str(record.get("driver_version") or "").strip() for record in target}
    capabilities = {str(record.get("compute_capability") or "").strip() for record in target}
    if len(uuids) != 1 or "" in uuids or len(drivers) != 1 or "" in drivers:
        raise AttachmentError("NVIDIA telemetry did not record one stable GPU UUID and driver")
    if len(capabilities) != 1 or "" in capabilities:
        raise AttachmentError("NVIDIA telemetry did not record one stable compute capability")
    try:
        utilizations = [float(record["utilization_gpu_percent"]) for record in target]
    except (KeyError, TypeError, ValueError) as exc:
        raise AttachmentError("NVIDIA telemetry has invalid utilization values") from exc
    if max(utilizations) <= 0:
        raise AttachmentError("NVIDIA telemetry never observed non-zero GPU utilization")
    gpu_uuid = next(iter(uuids))
    proaim_records = [
        record
        for record in process_records
        if record.get("gpu_uuid") == gpu_uuid
        and str(record.get("process_name") or "").casefold() == "proaimcli.exe"
    ]
    proaim_processes = sorted({str(record["process_name"]) for record in proaim_records})
    if not proaim_processes:
        raise AttachmentError("NVIDIA telemetry never associated ProAimCLI.exe with the named GPU")
    positive_times = {
        record["_captured_datetime"]
        for record in target
        if float(record["utilization_gpu_percent"]) > 0.0
    }
    process_times = {record["_captured_datetime"] for record in proaim_records}
    correlated_times = positive_times.intersection(process_times)
    if not correlated_times:
        raise AttachmentError(
            "NVIDIA telemetry never correlated positive target-GPU utilization with ProAimCLI.exe"
        )
    accelerated_intervals = [
        interval
        for interval in required_intervals
        if "benchmark" in interval[0] or "live pipeline" in interval[0]
    ]
    if len(accelerated_intervals) != 3:
        raise AttachmentError("qualification manifest has the wrong accelerated-run intervals")
    correlated_runs: list[str] = []
    correlated_sample_counts: dict[str, int] = {}
    for name, started, completed in accelerated_intervals:
        observed_times = sorted(
            observed for observed in correlated_times if started <= observed <= completed
        )
        minimum_samples = (
            CUDA_QUALIFICATION_POLICY["telemetry_min_correlated_samples_live"]
            if "live pipeline" in name
            else CUDA_QUALIFICATION_POLICY[
                "telemetry_min_correlated_samples_benchmark"
            ]
        )
        if len(observed_times) < minimum_samples:
            raise AttachmentError(
                f"NVIDIA telemetry did not meet correlated sample density during {name}"
            )
        correlated_runs.append(name)
        correlated_sample_counts[name] = len(observed_times)
    return {
        "gpu_name": expected_gpu,
        "gpu_uuid": gpu_uuid,
        "driver_version": next(iter(drivers)),
        "compute_capability": next(iter(capabilities)),
        "gpu_sample_count": len(target),
        "max_utilization_gpu_percent": max(utilizations),
        "proaim_compute_process_observed": True,
        "proaim_process_names": proaim_processes,
        "correlated_accelerated_runs": correlated_runs,
        "correlated_sample_counts": correlated_sample_counts,
        "sampling_policy": {
            "interval_milliseconds": CUDA_QUALIFICATION_POLICY[
                "telemetry_interval_milliseconds"
            ],
            "minimum_correlated_samples_benchmark": CUDA_QUALIFICATION_POLICY[
                "telemetry_min_correlated_samples_benchmark"
            ],
            "minimum_correlated_samples_live": CUDA_QUALIFICATION_POLICY[
                "telemetry_min_correlated_samples_live"
            ],
        },
        "telemetry_sha256": sha256_file(path),
    }


def _render_task_manager_confirmation(attestation: Mapping[str, Any]) -> str:
    observer = attestation["observer"]
    gpu = attestation["physical_gpu"]
    observations = attestation["observations"]
    return "\n".join(
        (
            "ProAim physical CUDA confirmation",
            "Schema: 1",
            "Status: COMPLETED",
            f"GitHub actor: {observer['github_actor']}",
            f"Observer: {observer['name']}",
            f"Observed at UTC: {observer['observed_at_utc']}",
            f"Physical GPU full name: {gpu['name']}",
            f"NVIDIA GPU UUID: {gpu['uuid']}",
            f"NVIDIA driver: {gpu['driver_version']}",
            f"CUDA compute capability: {gpu['compute_capability']}",
            f"Task Manager GPU/engine: {gpu['task_manager_gpu_engine']}",
            f"Observed release-default benchmark: {'YES' if observations['release_default_benchmark'] else 'NO'}",
            f"Observed live no-preview: {'YES' if observations['live_no_preview'] else 'NO'}",
            f"Observed live preview-15: {'YES' if observations['live_preview_15'] else 'NO'}",
            f"Observed nvidia-smi ProAim process/utilization: {'YES' if observations['nvidia_telemetry'] else 'NO'}",
            f"Typed confirmation: {observer['typed_confirmation']}",
            "",
        )
    )


def _validate_physical_attestation(
    attestation: Mapping[str, Any],
    *,
    files: Mapping[str, Path],
    inputs: DispatchInputs,
    remote: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    telemetry_summary: Mapping[str, Any],
    qualification_metrics: Mapping[str, Any],
    runner_invariant: Mapping[str, Any],
) -> None:
    if attestation.get("schema_version") != 1:
        raise AttachmentError("physical GPU attestation schema is unsupported")
    if attestation.get("status") != "physical_gpu_observation_attested":
        raise AttachmentError("physical GPU attestation has an incomplete status")
    if attestation.get("physical_observation_completed") is not True:
        raise AttachmentError("physical GPU attestation did not record completed observation")
    expected_top = {
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": remote["tag_commit"],
        "source_build_run_id": inputs.build_run_id,
    }
    for key, expected in expected_top.items():
        if attestation.get(key) != expected:
            raise AttachmentError(f"physical GPU attestation {key} mismatch")
    candidate = attestation.get("candidate")
    expected_candidate = {
        "filename": CUDA_ARCHIVE_NAME,
        "sha256": inputs.cuda_zip_sha256,
        "size_bytes": candidate_record.get("size_bytes"),
        "build_info_sha256": candidate_record.get("build_info_sha256"),
        "dependency_manifest_sha256": candidate_record.get(
            "dependency_manifest_sha256"
        ),
        "frozen_cli_sha256": candidate_record.get("frozen_cli_sha256"),
        "qualification_helper_sha256": candidate_record.get(
            "qualification_helper_sha256"
        ),
        "release_default_model": candidate_record.get("release_default_model"),
    }
    if not isinstance(candidate, dict) or candidate != expected_candidate:
        raise AttachmentError("physical GPU attestation binds a different CUDA candidate")
    manifest = attestation.get("qualification_manifest")
    if not isinstance(manifest, dict) or manifest != {
        "filename": QUALIFICATION_MANIFEST_NAME,
        "sha256": inputs.qualification_manifest_sha256,
        "software_status": "software_checks_passed_physical_gpu_confirmation_pending",
        "helper_qualified": False,
    }:
        raise AttachmentError("physical GPU attestation binds the wrong helper manifest")
    producer = attestation.get("producer")
    qualification_run = remote["qualification_run"]
    if not isinstance(producer, dict):
        raise AttachmentError("physical GPU attestation omitted producer identity")
    expected_producer = {
        "workflow_path": EXPECTED_QUALIFICATION_WORKFLOW,
        "run_id": inputs.evidence_run_id,
        "run_attempt": qualification_run["run_attempt"],
        "head_sha": remote["tag_commit"],
        "github_actor": qualification_run["actor"],
    }
    for key, expected in expected_producer.items():
        if producer.get(key) != expected:
            raise AttachmentError(f"physical GPU attestation producer {key} mismatch")
    if producer.get("attestation_environment") != "cuda-physical-attestation":
        raise AttachmentError("physical GPU attestation omitted its protected environment")
    raw_manifest_sha256 = _normalize_sha256(
        str(producer.get("raw_content_manifest_sha256") or ""),
        "raw content manifest SHA-256",
    )
    if raw_manifest_sha256 != sha256_file(files[RAW_CONTENT_MANIFEST_NAME]):
        raise AttachmentError("physical GPU attestation binds a different raw-content manifest")
    observer = attestation.get("observer")
    if not isinstance(observer, dict) or observer.get("github_actor") != qualification_run["actor"]:
        raise AttachmentError("physical GPU attestation observer is not the dispatching actor")
    if not str(observer.get("name") or "").strip():
        raise AttachmentError("physical GPU attestation omitted the observer name")
    expected_confirmation = expected_physical_confirmation(inputs.tag, inputs.qualified_gpu)
    if observer.get("typed_confirmation") != expected_confirmation:
        raise AttachmentError("physical GPU typed confirmation is not exact")
    if observer.get("authentication") != (
        "GitHub workflow_dispatch actor plus protected environment review"
    ):
        raise AttachmentError("physical GPU attestation omitted its authentication method")
    if observer.get("local_completion_file") != LOCAL_PHYSICAL_OBSERVATION_NAME:
        raise AttachmentError("physical GPU attestation omitted local post-run completion")
    try:
        observed = datetime.fromisoformat(str(observer.get("observed_at_utc") or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise AttachmentError("physical GPU observation time is not ISO-8601") from exc
    if observed.tzinfo is None or observed.utcoffset() != timedelta(0):
        raise AttachmentError("physical GPU observation time must explicitly use UTC")
    physical_gpu = attestation.get("physical_gpu")
    if not isinstance(physical_gpu, dict) or physical_gpu.get("name") != inputs.qualified_gpu:
        raise AttachmentError("physical GPU attestation names a different GPU")
    expected_gpu_fields = {
        "uuid": telemetry_summary["gpu_uuid"],
        "driver_version": telemetry_summary["driver_version"],
        "compute_capability": telemetry_summary["compute_capability"],
    }
    for key, expected in expected_gpu_fields.items():
        if physical_gpu.get(key) != expected:
            raise AttachmentError(f"physical GPU attestation {key} differs from telemetry")
    if not str(physical_gpu.get("task_manager_gpu_engine") or "").strip():
        raise AttachmentError("physical GPU attestation omitted Task Manager engine text")
    observations = attestation.get("observations")
    if not isinstance(observations, dict) or any(
        observations.get(key) is not True
        for key in (
            "release_default_benchmark",
            "live_no_preview",
            "live_preview_15",
            "nvidia_telemetry",
        )
    ):
        raise AttachmentError("physical GPU attestation has an incomplete observation checklist")
    if attestation.get("nvidia_telemetry_summary") != telemetry_summary:
        raise AttachmentError("physical GPU attestation telemetry summary differs from raw telemetry")
    if attestation.get("qualification_metrics") != qualification_metrics:
        raise AttachmentError("physical GPU attestation qualification metrics differ from reports")
    if attestation.get("cuda_runner_invariant") != runner_invariant:
        raise AttachmentError("physical GPU attestation runner invariant differs from preflight")
    sealed_files = _records_by_key(
        attestation.get("sealed_files"), key="file", description="attestation sealed files"
    )
    expected_sealed = set(files).difference({PHYSICAL_ATTESTATION_NAME})
    if set(sealed_files) != expected_sealed:
        raise AttachmentError("physical GPU attestation did not bind the exact evidence file set")
    for filename, record in sealed_files.items():
        _validate_recorded_file(record, files, description=f"sealed evidence {filename}")
    expected_confirmation_text = _render_task_manager_confirmation(attestation)
    try:
        actual_confirmation_text = files[TASK_MANAGER_CONFIRMATION_NAME].read_text(
            encoding="utf-8-sig"
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise AttachmentError("Task Manager confirmation is not readable UTF-8") from exc
    if actual_confirmation_text != expected_confirmation_text:
        raise AttachmentError("Task Manager confirmation text differs from its attestation")
    local_observation = _read_json_object(
        files[LOCAL_PHYSICAL_OBSERVATION_NAME], "local physical observation"
    )
    expected_local = {
        "schema_version": 1,
        "status": "completed_after_automated_gpu_runs",
        "completed": True,
        "tag": inputs.tag,
        "github_actor": qualification_run["actor"],
        "github_run_id": str(inputs.evidence_run_id),
        "observer_name": observer["name"],
        "observed_at_utc": observer["observed_at_utc"],
        "physical_gpu_name": inputs.qualified_gpu,
        "task_manager_gpu_engine": physical_gpu["task_manager_gpu_engine"],
        "typed_confirmation": expected_confirmation,
        "observations": {
            "release_default_benchmark": True,
            "live_no_preview": True,
            "live_preview_15": True,
        },
        "completion_method": "interactive Windows desktop form after automated GPU runs",
    }
    if local_observation != expected_local:
        raise AttachmentError("local physical observation differs from sealed attestation")


def validate_physical_evidence(
    evidence_archive: Path,
    *,
    inputs: DispatchInputs,
    remote: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    extract_directory: Path,
) -> dict[str, Any]:
    actual_archive_sha = sha256_file(evidence_archive)
    if actual_archive_sha != inputs.qualification_evidence_sha256:
        raise AttachmentError("qualification evidence archive SHA-256 does not match input")
    files = _extract_evidence_archive(evidence_archive, extract_directory)
    manifest, public_metrics, run_intervals = _validate_software_evidence(
        files,
        candidate_record=candidate_record,
        expected_commit=remote["tag_commit"],
        expected_cuda_sha256=inputs.cuda_zip_sha256,
        expected_manifest_sha256=inputs.qualification_manifest_sha256,
    )
    raw_candidate = _read_json_object(files[RAW_CANDIDATE_RECORD_NAME], "candidate inspection")
    if raw_candidate != candidate_record:
        raise AttachmentError("qualification evidence candidate inspection differs from candidate")
    raw_source = _read_json_object(files[RAW_SOURCE_RECORD_NAME], "verified source record")
    for key in ("tag_commit", "build_run", "build_artifact"):
        if raw_source.get(key) != remote.get(key):
            raise AttachmentError(f"qualification evidence source record {key} mismatch")
    telemetry_summary = _parse_nvidia_telemetry(
        files[NVIDIA_TELEMETRY_NAME],
        inputs.qualified_gpu,
        required_intervals=run_intervals,
    )
    runner_invariant = _validate_cuda_runner_invariant(
        files[CUDA_RUNNER_INVARIANT_NAME], inputs.qualified_gpu
    )
    attestation_path = files[PHYSICAL_ATTESTATION_NAME]
    if sha256_file(attestation_path) != inputs.physical_attestation_sha256:
        raise AttachmentError("physical GPU attestation SHA-256 does not match input")
    attestation = _read_json_object(attestation_path, "physical GPU attestation")
    attestation_observer = attestation.get("observer")
    _validate_physical_timeline(
        runner_invariant,
        run_intervals,
        attestation_observer.get("observed_at_utc")
        if isinstance(attestation_observer, dict)
        else None,
    )
    _validate_physical_attestation(
        attestation,
        files=files,
        inputs=inputs,
        remote=remote,
        candidate_record=candidate_record,
        telemetry_summary=telemetry_summary,
        qualification_metrics=public_metrics,
        runner_invariant=runner_invariant,
    )
    qualification_artifact = remote["qualification_artifact"]
    return {
        "qualified_gpu": inputs.qualified_gpu,
        "evidence_archive": {
            "filename": QUALIFICATION_EVIDENCE_ARCHIVE_NAME,
            "sha256": actual_archive_sha,
            "size_bytes": evidence_archive.stat().st_size,
        },
        "qualification_manifest": {
            "sha256": inputs.qualification_manifest_sha256,
            "status": manifest["status"],
            "helper_qualified": False,
        },
        "physical_attestation": {
            "sha256": inputs.physical_attestation_sha256,
            "status": attestation["status"],
            "observer": attestation["observer"],
        },
        "qualification_run": remote["qualification_run"],
        "qualification_artifact": qualification_artifact,
        "nvidia_telemetry_summary": telemetry_summary,
        "qualification_metrics": public_metrics,
    }


def _require_single_line(value: str, description: str, *, maximum: int = 240) -> str:
    normalized = str(value).strip()
    if not normalized or len(normalized) > maximum or any(char in normalized for char in "\r\n\0"):
        raise AttachmentError(f"{description} must be one non-empty line of at most {maximum} characters")
    return normalized


def _copy_flat_files(source_files: Mapping[str, Path], destination: Path) -> dict[str, Path]:
    destination.mkdir()
    copied: dict[str, Path] = {}
    for filename, source in sorted(source_files.items()):
        target = destination / filename
        shutil.copyfile(source, target)
        copied[filename] = target
    return copied


def _raw_qualification_files(raw_directory: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    if not raw_directory.is_dir():
        raise AttachmentError(f"raw qualification directory not found: {raw_directory}")
    for path in raw_directory.rglob("*"):
        if path.is_symlink():
            raise AttachmentError("raw qualification artifact contains a symbolic link")
    software_directory = raw_directory / "software-evidence"
    expected_root_files = {
        RAW_CANDIDATE_RECORD_NAME,
        RAW_SOURCE_RECORD_NAME,
        RAW_CONTENT_MANIFEST_NAME,
        NVIDIA_TELEMETRY_NAME,
        CUDA_RUNNER_INVARIANT_NAME,
        LOCAL_PHYSICAL_OBSERVATION_NAME,
    }
    root_files = {path.name: path for path in raw_directory.iterdir() if path.is_file()}
    root_directories = {path.name for path in raw_directory.iterdir() if path.is_dir()}
    if set(root_files) != expected_root_files or root_directories != {"software-evidence"}:
        raise AttachmentError("raw qualification artifact contained the wrong root file set")
    if any(path.is_dir() for path in software_directory.iterdir()):
        raise AttachmentError("software evidence directory must be flat")
    software_files = {path.name: path for path in software_directory.iterdir() if path.is_file()}
    expected_software = REQUIRED_EVIDENCE_FILES - {
        PHYSICAL_ATTESTATION_NAME,
        NVIDIA_TELEMETRY_NAME,
        CUDA_RUNNER_INVARIANT_NAME,
        LOCAL_PHYSICAL_OBSERVATION_NAME,
        RAW_CANDIDATE_RECORD_NAME,
        RAW_SOURCE_RECORD_NAME,
        RAW_CONTENT_MANIFEST_NAME,
    }
    if set(software_files) != expected_software:
        raise AttachmentError("raw helper evidence contained the wrong file set")
    return software_files, root_files


def seal_physical_evidence(
    *,
    raw_directory: Path,
    output_directory: Path,
    repository: str,
    tag: str,
    source_build_run_id: str,
    cuda_zip_sha256: str,
    qualified_gpu: str,
    observer_name: str,
    typed_confirmation: str,
    github_actor: str,
    github_run_id: str,
    github_run_attempt: str,
    github_head_sha: str,
    raw_content_manifest_sha256: str,
) -> dict[str, Any]:
    normalized_repository = repository.strip()
    if not REPOSITORY_RE.fullmatch(normalized_repository):
        raise AttachmentError("repository must use the exact OWNER/REPOSITORY form")
    normalized_tag = tag.strip()
    if not TAG_RE.fullmatch(normalized_tag):
        raise AttachmentError("tag must be a short v* release tag")
    build_run_id = _normalize_run_id(source_build_run_id)
    qualification_run_id = _normalize_run_id(github_run_id)
    run_attempt = _normalize_run_id(github_run_attempt)
    candidate_sha = _normalize_sha256(cuda_zip_sha256, "CUDA ZIP SHA-256")
    head_sha = _require_sha(github_head_sha, "qualification workflow head")
    gpu_name = _require_single_line(qualified_gpu, "qualified GPU", maximum=160)
    observer = _require_single_line(observer_name, "observer name")
    actor = _require_single_line(github_actor, "GitHub actor", maximum=100)
    expected_typed = expected_physical_confirmation(normalized_tag, gpu_name)
    if typed_confirmation != expected_typed:
        raise AttachmentError("physical observation typed confirmation mismatch")
    software_files, root_files = _raw_qualification_files(raw_directory)
    raw_context = {
        "repository": normalized_repository,
        "tag": normalized_tag,
        "tag_commit": head_sha,
        "source_build_run_id": build_run_id,
        "qualification_run_id": qualification_run_id,
        "qualification_run_attempt": run_attempt,
    }
    validate_content_manifest(
        root=raw_directory,
        manifest_name=RAW_CONTENT_MANIFEST_NAME,
        expected_sha256=raw_content_manifest_sha256,
        expected_kind="proaim-cuda-raw-qualification",
        expected_context=raw_context,
    )
    runner_invariant = _validate_cuda_runner_invariant(
        root_files[CUDA_RUNNER_INVARIANT_NAME], gpu_name
    )
    local_observation = _read_json_object(
        root_files[LOCAL_PHYSICAL_OBSERVATION_NAME],
        "local physical observation",
    )
    if (
        local_observation.get("schema_version") != 1
        or local_observation.get("status") != "completed_after_automated_gpu_runs"
        or local_observation.get("completed") is not True
        or local_observation.get("completion_method")
        != "interactive Windows desktop form after automated GPU runs"
    ):
        raise AttachmentError("local physical observation is incomplete or unsupported")
    expected_local = {
        "tag": normalized_tag,
        "github_actor": actor,
        "github_run_id": str(qualification_run_id),
        "observer_name": observer,
        "physical_gpu_name": gpu_name,
        "typed_confirmation": typed_confirmation,
    }
    for key, expected in expected_local.items():
        if local_observation.get(key) != expected:
            raise AttachmentError(f"local physical observation {key} mismatch")
    local_observations = local_observation.get("observations")
    if not isinstance(local_observations, dict) or any(
        local_observations.get(key) is not True
        for key in ("release_default_benchmark", "live_no_preview", "live_preview_15")
    ):
        raise AttachmentError("local physical observation checklist is incomplete")
    engine = _require_single_line(
        str(local_observation.get("task_manager_gpu_engine") or ""),
        "Task Manager GPU engine",
    )
    local_observed_text = str(local_observation.get("observed_at_utc") or "")
    try:
        observed_time = datetime.fromisoformat(
            local_observed_text.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise AttachmentError("local observation time must be an ISO-8601 timestamp") from exc
    if observed_time.tzinfo is None or observed_time.utcoffset() != timedelta(0):
        raise AttachmentError("local observation time must explicitly use UTC")
    now = datetime.now(timezone.utc)
    observed_utc = observed_time.astimezone(timezone.utc)
    if observed_utc > now + timedelta(minutes=5) or observed_utc < now - timedelta(days=1):
        raise AttachmentError("local physical observation time is outside the one-day sealing window")
    observed_text = local_observed_text
    candidate_record = _read_json_object(
        root_files[RAW_CANDIDATE_RECORD_NAME], "candidate inspection"
    )
    source_record = _read_json_object(root_files[RAW_SOURCE_RECORD_NAME], "source verification")
    if source_record.get("tag_commit") != head_sha:
        raise AttachmentError("source verification tag commit differs from workflow head")
    build_run = source_record.get("build_run")
    if not isinstance(build_run, dict) or build_run.get("id") != build_run_id:
        raise AttachmentError("source verification binds a different build run")
    if build_run.get("head_sha") != head_sha:
        raise AttachmentError("source build head differs from qualification workflow head")
    if candidate_record.get("filename") != CUDA_ARCHIVE_NAME:
        raise AttachmentError("candidate inspection has the wrong archive filename")
    if candidate_record.get("sha256") != candidate_sha:
        raise AttachmentError("candidate inspection binds a different CUDA ZIP hash")
    candidate_build = candidate_record.get("build_info")
    if not isinstance(candidate_build, dict) or candidate_build.get("commit") != head_sha:
        raise AttachmentError("candidate inspection BUILD-INFO differs from workflow head")
    combined_for_validation = dict(software_files)
    combined_for_validation.update(root_files)
    manifest, public_metrics, run_intervals = _validate_software_evidence(
        combined_for_validation,
        candidate_record=candidate_record,
        expected_commit=head_sha,
        expected_cuda_sha256=candidate_sha,
    )
    _validate_physical_timeline(runner_invariant, run_intervals, observed_text)
    telemetry_summary = _parse_nvidia_telemetry(
        root_files[NVIDIA_TELEMETRY_NAME],
        gpu_name,
        required_intervals=run_intervals,
    )
    manifest_sha = sha256_file(software_files[QUALIFICATION_MANIFEST_NAME])
    if output_directory.exists():
        raise AttachmentError(f"refusing to overwrite evidence output path: {output_directory}")
    output_parent = output_directory.parent
    if not output_parent.is_dir():
        raise AttachmentError(f"evidence output parent does not exist: {output_parent}")
    with tempfile.TemporaryDirectory(prefix=".cuda-evidence-", dir=output_parent) as temporary:
        temporary_root = Path(temporary)
        evidence_stage = temporary_root / "evidence"
        files = _copy_flat_files(software_files, evidence_stage)
        for filename, path in root_files.items():
            target = evidence_stage / filename
            shutil.copyfile(path, target)
            files[filename] = target
        attestation: dict[str, Any] = {
            "schema_version": 1,
            "status": "physical_gpu_observation_attested",
            "physical_observation_completed": True,
            "repository": normalized_repository,
            "tag": normalized_tag,
            "tag_commit": head_sha,
            "source_build_run_id": build_run_id,
            "candidate": {
                "filename": CUDA_ARCHIVE_NAME,
                "sha256": candidate_sha,
                "size_bytes": candidate_record.get("size_bytes"),
                "build_info_sha256": candidate_record.get("build_info_sha256"),
                "dependency_manifest_sha256": candidate_record.get(
                    "dependency_manifest_sha256"
                ),
                "frozen_cli_sha256": candidate_record.get("frozen_cli_sha256"),
                "qualification_helper_sha256": candidate_record.get(
                    "qualification_helper_sha256"
                ),
                "release_default_model": candidate_record.get("release_default_model"),
            },
            "qualification_manifest": {
                "filename": QUALIFICATION_MANIFEST_NAME,
                "sha256": manifest_sha,
                "software_status": manifest["status"],
                "helper_qualified": False,
            },
            "producer": {
                "workflow_path": EXPECTED_QUALIFICATION_WORKFLOW,
                "run_id": qualification_run_id,
                "run_attempt": run_attempt,
                "head_sha": head_sha,
                "github_actor": actor,
                "raw_content_manifest_sha256": _normalize_sha256(
                    raw_content_manifest_sha256,
                    "raw content manifest SHA-256",
                ),
                "attestation_environment": "cuda-physical-attestation",
            },
            "observer": {
                "github_actor": actor,
                "name": observer,
                "observed_at_utc": observed_text,
                "typed_confirmation": typed_confirmation,
                "authentication": "GitHub workflow_dispatch actor plus protected environment review",
                "local_completion_file": LOCAL_PHYSICAL_OBSERVATION_NAME,
            },
            "physical_gpu": {
                "name": gpu_name,
                "uuid": telemetry_summary["gpu_uuid"],
                "driver_version": telemetry_summary["driver_version"],
                "compute_capability": telemetry_summary["compute_capability"],
                "task_manager_gpu_engine": engine,
            },
            "observations": {
                "release_default_benchmark": True,
                "live_no_preview": True,
                "live_preview_15": True,
                "nvidia_telemetry": True,
            },
            "nvidia_telemetry_summary": telemetry_summary,
            "qualification_metrics": public_metrics,
            "cuda_runner_invariant": runner_invariant,
            "legal_limit": (
                "This attestation records physical execution only. It is not legal advice and "
                "does not establish permission to redistribute NVIDIA software."
            ),
        }
        confirmation_path = files[TASK_MANAGER_CONFIRMATION_NAME]
        confirmation_path.write_text(
            _render_task_manager_confirmation(attestation), encoding="utf-8"
        )
        attestation["sealed_files"] = [
            {
                "file": filename,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for filename, path in sorted(files.items())
        ]
        attestation_path = evidence_stage / PHYSICAL_ATTESTATION_NAME
        _write_json_atomic(attestation_path, attestation)
        files[PHYSICAL_ATTESTATION_NAME] = attestation_path
        final_directory = temporary_root / "final"
        final_directory.mkdir()
        archive_path = final_directory / QUALIFICATION_EVIDENCE_ARCHIVE_NAME
        with zipfile.ZipFile(
            archive_path,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for filename, path in sorted(files.items()):
                info = zipfile.ZipInfo(filename, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, path.read_bytes(), compresslevel=9)
        result = {
            "status": "sealed_physical_gpu_evidence",
            "evidence_archive": {
                "filename": QUALIFICATION_EVIDENCE_ARCHIVE_NAME,
                "sha256": sha256_file(archive_path),
                "size_bytes": archive_path.stat().st_size,
            },
            "qualification_manifest_sha256": manifest_sha,
            "physical_attestation_sha256": sha256_file(attestation_path),
            "qualified_gpu": gpu_name,
            "tag": normalized_tag,
            "tag_commit": head_sha,
            "source_build_run_id": build_run_id,
        }
        final_directory.replace(output_directory)
    return result


def validate_hosted_smoke(
    *,
    stage_directory: Path,
    runtime_info_path: Path,
    benchmark_path: Path,
) -> dict[str, Any]:
    attestation_path = stage_directory / "ATTACHMENT-ATTESTATION.json"
    attestation = _read_json_object(attestation_path, "attachment attestation")
    runtime = _read_json_object(runtime_info_path, "hosted runtime report")
    if runtime.get("frozen") is not True:
        raise AttachmentError("hosted runtime report did not identify a frozen executable")
    if "onnxruntime_error" in runtime:
        raise AttachmentError(f"hosted runtime failed to import ONNX Runtime: {runtime['onnxruntime_error']}")
    providers = runtime.get("onnxruntime_providers")
    if not isinstance(providers, list) or "CUDAExecutionProvider" not in providers:
        raise AttachmentError("hosted runtime does not expose CUDAExecutionProvider")
    if "CPUExecutionProvider" not in providers:
        raise AttachmentError("hosted runtime does not expose CPUExecutionProvider for the CPU smoke")
    benchmark = _read_json_object(benchmark_path, "hosted CPU model smoke report")
    if "error" in benchmark:
        raise AttachmentError(f"hosted CPU model smoke reported an error: {benchmark['error']}")
    methodology = benchmark.get("methodology")
    models = benchmark.get("models")
    if not isinstance(methodology, dict) or methodology.get("backend") != "onnxruntime":
        raise AttachmentError("hosted model smoke did not use ONNX Runtime")
    if methodology.get("requested_device") != "CPU":
        raise AttachmentError("hosted model smoke did not request CPU")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise AttachmentError("hosted model smoke must report exactly one model")
    model = models[0]
    candidate = attestation.get("candidate")
    default_model = (
        candidate.get("release_default_model") if isinstance(candidate, dict) else None
    )
    if not isinstance(default_model, dict):
        raise AttachmentError("attachment attestation omitted the release-default model")
    if (
        model.get("key") != "release-default"
        or model.get("input_shape_hw") != default_model.get("input_shape_hw")
    ):
        raise AttachmentError("hosted model smoke used the wrong release-default key or shape")
    model_runtime = model.get("runtime")
    if not isinstance(model_runtime, dict):
        raise AttachmentError("hosted model smoke omitted runtime identity")
    if model_runtime.get("requested_provider") != "CPUExecutionProvider":
        raise AttachmentError("hosted model smoke did not request CPUExecutionProvider")
    active = model_runtime.get("active_providers")
    if not isinstance(active, list) or "CPUExecutionProvider" not in active:
        raise AttachmentError("hosted model smoke did not activate CPUExecutionProvider")
    artifact = model.get("artifact")
    if not isinstance(artifact, dict):
        raise AttachmentError("hosted model smoke omitted model artifact identity")
    files = artifact.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise AttachmentError("hosted model smoke must fingerprint exactly one model file")
    expected_model_hash = default_model.get("model_sha256")
    if files[0].get("sha256") != expected_model_hash:
        raise AttachmentError("hosted model smoke fingerprinted a different release-default model")
    labels_artifact = model.get("labels_artifact")
    label_files = labels_artifact.get("files") if isinstance(labels_artifact, dict) else None
    if (
        not isinstance(label_files, list)
        or len(label_files) != 1
        or not isinstance(label_files[0], dict)
        or label_files[0].get("sha256") != default_model.get("labels_sha256")
    ):
        raise AttachmentError("hosted model smoke fingerprinted different release-default labels")
    staged_runtime = stage_directory / "hosted-runtime-info.json"
    staged_benchmark = stage_directory / "hosted-cpu-model-smoke.json"
    shutil.copyfile(runtime_info_path, staged_runtime)
    shutil.copyfile(benchmark_path, staged_benchmark)
    workflow = attestation.get("verification_workflow")
    if not isinstance(workflow, dict):
        raise AttachmentError("attachment attestation omitted verification workflow identity")
    workflow["hosted_smoke"] = {
        "scope": "frozen CLI import and real release-default-model inference on CPU; no GPU claim",
        "runtime_info_sha256": sha256_file(staged_runtime),
        "cpu_model_smoke_sha256": sha256_file(staged_benchmark),
    }
    _write_json_atomic(attestation_path, attestation)
    return attestation


def parse_checksum_manifest(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AttachmentError("existing SHA256SUMS.txt is not UTF-8") from exc
    records: dict[str, str] = {}
    folded: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line:
            continue
        match = CHECKSUM_LINE_RE.fullmatch(raw_line)
        if match is None:
            raise AttachmentError(
                f"existing SHA256SUMS.txt has invalid line {line_number}: {raw_line!r}"
            )
        digest, filename = match.groups()
        if not filename.casefold().endswith((".zip", ".json")):
            raise AttachmentError(
                f"SHA256SUMS.txt line {line_number} is not a checksummed release asset: {filename!r}"
            )
        key = filename.casefold()
        if key in folded:
            raise AttachmentError(f"SHA256SUMS.txt contains duplicate filename {filename!r}")
        folded.add(key)
        records[filename] = digest.lower()
    if not records:
        raise AttachmentError("existing SHA256SUMS.txt contains no asset checksums")
    return records


def render_checksum_manifest(records: Mapping[str, str]) -> bytes:
    normalized: list[tuple[str, str]] = []
    folded: set[str] = set()
    for filename, digest in records.items():
        if (
            not filename
            or "/" in filename
            or "\\" in filename
            or "\r" in filename
            or "\n" in filename
            or not filename.casefold().endswith((".zip", ".json"))
        ):
            raise AttachmentError(f"unsafe checksum filename {filename!r}")
        key = filename.casefold()
        if key in folded:
            raise AttachmentError(f"duplicate checksum filename {filename!r}")
        folded.add(key)
        normalized.append((filename, _normalize_sha256(digest, f"checksum for {filename}")))
    return "".join(
        f"{digest}  {filename}\n" for filename, digest in sorted(normalized, key=lambda item: item[0])
    ).encode("utf-8")


def _asset_id(asset: Mapping[str, Any], description: str) -> int:
    value = asset.get("id")
    if not isinstance(value, int):
        raise AttachmentError(f"{description} omitted its numeric asset ID")
    return value


def _release_assets(api: GitHubApi, release_id: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for page in range(1, 101):
        # Release asset listing is a bare JSON array, unlike Actions artifact
        # listing.  Keep this method explicit so a schema change fails closed.
        values = api.get_json_list(
            f"/releases/{release_id}/assets?per_page=100&page={page}"
        )
        collected.extend(values)
        if len(values) < 100:
            return collected
    raise AttachmentError("release asset pagination exceeded the safety limit")


def _assets_by_name(assets: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    mapped: dict[str, Mapping[str, Any]] = {}
    folded: set[str] = set()
    for asset in assets:
        name = asset.get("name")
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise AttachmentError("release contains an asset with an unsafe or missing name")
        key = name.casefold()
        if key in folded:
            raise AttachmentError(f"release contains case-insensitive duplicate asset {name!r}")
        folded.add(key)
        mapped[name] = asset
    return mapped


def _asset_signature(assets: Iterable[Mapping[str, Any]]) -> tuple[tuple[int, str, int], ...]:
    signature: list[tuple[int, str, int]] = []
    for asset in assets:
        asset_id = _asset_id(asset, "release asset")
        name = str(asset.get("name") or "")
        size = asset.get("size")
        if not isinstance(size, int) or size < 0:
            raise AttachmentError(f"release asset {name!r} omitted a valid size")
        signature.append((asset_id, name, size))
    return tuple(sorted(signature))


def _validate_stage(
    stage_directory: Path,
    inputs: DispatchInputs,
    *,
    staged_content_manifest_sha256: str,
    verification_run_id: int,
    tag_commit: str,
) -> tuple[Path, dict[str, Any]]:
    if not stage_directory.is_dir():
        raise AttachmentError(f"verified attachment stage not found: {stage_directory}")
    validate_content_manifest(
        root=stage_directory,
        manifest_name=STAGED_CONTENT_MANIFEST_NAME,
        expected_sha256=staged_content_manifest_sha256,
        expected_kind="proaim-cuda-verified-stage",
        expected_context={
            "repository": inputs.repository,
            "tag": inputs.tag,
            "tag_commit": tag_commit,
            "source_build_run_id": inputs.build_run_id,
            "verification_run_id": verification_run_id,
        },
    )
    for path in stage_directory.rglob("*"):
        if path.is_symlink() or path.is_dir():
            raise AttachmentError("verified attachment artifact contains a link or nested directory")
    files = {path.name: path for path in stage_directory.iterdir() if path.is_file()}
    if set(files) != VERIFIED_STAGE_FILES:
        raise AttachmentError(
            "verified attachment artifact contained the wrong file set: " + ", ".join(sorted(files))
        )
    candidate = files[CUDA_ARCHIVE_NAME]
    if sha256_file(candidate) != inputs.cuda_zip_sha256:
        raise AttachmentError("staged CUDA ZIP hash does not match the dispatch input")
    evidence_archive = files[QUALIFICATION_EVIDENCE_ARCHIVE_NAME]
    if sha256_file(evidence_archive) != inputs.qualification_evidence_sha256:
        raise AttachmentError("staged qualification evidence hash does not match dispatch input")
    attestation = _read_json_object(files["ATTACHMENT-ATTESTATION.json"], "attachment attestation")
    if attestation.get("schema_version") != 2:
        raise AttachmentError("attachment attestation schema is unsupported")
    expected_pairs = {
        "repository": inputs.repository,
        "tag": inputs.tag,
    }
    for key, expected in expected_pairs.items():
        if attestation.get(key) != expected:
            raise AttachmentError(f"attachment attestation {key} mismatch")
    candidate_record = attestation.get("candidate")
    qualification = attestation.get("physical_qualification")
    workflow = attestation.get("verification_workflow")
    if not isinstance(candidate_record, dict) or candidate_record.get("sha256") != inputs.cuda_zip_sha256:
        raise AttachmentError("attachment attestation candidate mismatch")
    if not isinstance(qualification, dict):
        raise AttachmentError("attachment attestation omitted physical qualification")
    if qualification.get("qualified_gpu") != inputs.qualified_gpu:
        raise AttachmentError("attachment attestation qualified GPU mismatch")
    evidence_record = qualification.get("evidence_archive")
    manifest_record = qualification.get("qualification_manifest")
    physical_record = qualification.get("physical_attestation")
    if not isinstance(evidence_record, dict) or evidence_record.get("sha256") != inputs.qualification_evidence_sha256:
        raise AttachmentError("attachment attestation evidence archive mismatch")
    if not isinstance(manifest_record, dict) or manifest_record.get("sha256") != inputs.qualification_manifest_sha256:
        raise AttachmentError("attachment attestation qualification manifest mismatch")
    if not isinstance(physical_record, dict) or physical_record.get("sha256") != inputs.physical_attestation_sha256:
        raise AttachmentError("attachment attestation physical attestation mismatch")
    if qualification.get("release_typed_confirmation") != inputs.confirmation:
        raise AttachmentError("attachment attestation release confirmation mismatch")
    if (
        qualification.get("nvidia_redistribution_typed_confirmation")
        != inputs.nvidia_redistribution_confirmation
    ):
        raise AttachmentError("attachment attestation NVIDIA redistribution confirmation mismatch")
    qualification_run = qualification.get("qualification_run")
    qualification_artifact = qualification.get("qualification_artifact")
    if not isinstance(qualification_run, dict) or qualification_run.get("id") != inputs.evidence_run_id:
        raise AttachmentError("attachment attestation qualification run mismatch")
    if (
        not isinstance(qualification_artifact, dict)
        or qualification_artifact.get("name") != inputs.evidence_artifact_name
    ):
        raise AttachmentError("attachment attestation qualification artifact mismatch")
    if not isinstance(workflow, dict) or not isinstance(workflow.get("hosted_smoke"), dict):
        raise AttachmentError("attachment attestation omitted completed hosted smoke evidence")
    smoke = workflow["hosted_smoke"]
    if smoke.get("runtime_info_sha256") != sha256_file(files["hosted-runtime-info.json"]):
        raise AttachmentError("hosted runtime report hash does not match the attestation")
    if smoke.get("cpu_model_smoke_sha256") != sha256_file(files["hosted-cpu-model-smoke.json"]):
        raise AttachmentError("hosted CPU model report hash does not match the attestation")
    receipt_path = files[CUDA_QUALIFICATION_RECEIPT_NAME]
    receipt = _read_json_object(receipt_path, "public CUDA qualification receipt")
    receipt_record = attestation.get("public_qualification_receipt")
    if not isinstance(receipt_record, dict) or receipt_record != {
        "filename": CUDA_QUALIFICATION_RECEIPT_NAME,
        "sha256": sha256_file(receipt_path),
    }:
        raise AttachmentError("public qualification receipt hash differs from attestation")
    telemetry = qualification.get("nvidia_telemetry_summary")
    if not isinstance(telemetry, dict):
        raise AttachmentError("attachment attestation omitted NVIDIA telemetry summary")
    expected_receipt_fields = {
        "schema_version": 1,
        "status": "physically_qualified_cuda_release_candidate",
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": tag_commit,
        "candidate": {"filename": CUDA_ARCHIVE_NAME, "sha256": inputs.cuda_zip_sha256},
        "evidence_hashes": {
            "archive_sha256": inputs.qualification_evidence_sha256,
            "qualification_manifest_sha256": inputs.qualification_manifest_sha256,
            "physical_attestation_sha256": inputs.physical_attestation_sha256,
        },
        "physical_gpu": {
            "product_name": inputs.qualified_gpu,
            "driver_version": telemetry.get("driver_version"),
            "compute_capability": telemetry.get("compute_capability"),
        },
        "qualification_metrics": qualification.get("qualification_metrics"),
        "qualification_run": {
            "id": qualification_run.get("id"),
            "html_url": qualification_run.get("html_url"),
        },
    }
    for key, expected in expected_receipt_fields.items():
        if receipt.get(key) != expected:
            raise AttachmentError(f"public qualification receipt {key} mismatch")
    privacy = receipt.get("privacy")
    if not isinstance(privacy, dict) or privacy.get("redacted") is not True:
        raise AttachmentError("public qualification receipt omitted its privacy declaration")
    serialized_receipt = json.dumps(receipt, sort_keys=True).casefold()
    observer = physical_record.get("observer") if isinstance(physical_record, dict) else None
    observer_name = str(observer.get("name") or "") if isinstance(observer, dict) else ""
    if observer_name and observer_name.casefold() in serialized_receipt:
        raise AttachmentError("public qualification receipt leaked the observer identity")
    gpu_uuid = str(telemetry.get("gpu_uuid") or "")
    if gpu_uuid and gpu_uuid.casefold() in serialized_receipt:
        raise AttachmentError("public qualification receipt leaked the GPU UUID")
    return candidate, attestation


def _rollback_publication(
    api: GitHubApi,
    release_id: int,
    *,
    original_manifest: bytes,
    attempted_manifest: bytes,
    candidate_sha256: str,
    receipt_sha256: str,
) -> list[str]:
    """Best-effort restore the exact pre-publication checksum/CUDA state.

    REST mutations can succeed server-side while the client observes a timeout.
    Reconcile by name *and content* instead of relying only on returned asset
    objects.  A conflicting payload is never deleted or overwritten.
    """

    errors: list[str] = []

    def current_assets() -> dict[str, Mapping[str, Any]] | None:
        try:
            return _assets_by_name(_release_assets(api, release_id))
        except AttachmentError as exc:
            errors.append(f"rollback could not list release assets: {exc}")
            return None

    current = current_assets()
    if current is None:
        return errors
    checksum = current.get(CHECKSUM_ASSET_NAME)
    if checksum is not None:
        try:
            checksum_payload = api.download_release_asset(
                _asset_id(checksum, CHECKSUM_ASSET_NAME)
            )
            if checksum_payload == attempted_manifest:
                api.delete_release_asset(_asset_id(checksum, CHECKSUM_ASSET_NAME))
            elif checksum_payload != original_manifest:
                errors.append(
                    "rollback found a conflicting SHA256SUMS.txt and refused to delete it"
                )
        except AttachmentError as exc:
            errors.append(f"rollback could not reconcile SHA256SUMS.txt: {exc}")

    current = current_assets()
    if current is None:
        return errors
    checksum = current.get(CHECKSUM_ASSET_NAME)
    checksum_is_original = False
    if checksum is not None:
        try:
            checksum_is_original = (
                api.download_release_asset(
                    _asset_id(checksum, CHECKSUM_ASSET_NAME)
                )
                == original_manifest
            )
        except AttachmentError as exc:
            errors.append(f"rollback could not verify SHA256SUMS.txt: {exc}")
    else:
        try:
            api.upload_release_asset(
                release_id,
                CHECKSUM_ASSET_NAME,
                original_manifest,
                "text/plain; charset=utf-8",
            )
        except AttachmentError as exc:
            # The upload may have succeeded before a transport error. Re-list
            # and accept it only when the exact original bytes are present.
            verification = current_assets()
            restored = False
            if verification is not None and CHECKSUM_ASSET_NAME in verification:
                try:
                    restored = (
                        api.download_release_asset(
                            _asset_id(
                                verification[CHECKSUM_ASSET_NAME],
                                CHECKSUM_ASSET_NAME,
                            )
                        )
                        == original_manifest
                    )
                except AttachmentError:
                    restored = False
            if not restored:
                errors.append(f"rollback could not restore SHA256SUMS.txt: {exc}")
        else:
            checksum_is_original = True
    if checksum is not None and not checksum_is_original:
        errors.append("rollback could not prove that the original SHA256SUMS.txt is restored")

    current = current_assets()
    if current is None:
        return errors
    for asset_name, expected_sha256 in (
        (CUDA_ARCHIVE_NAME, candidate_sha256),
        (CUDA_QUALIFICATION_RECEIPT_NAME, receipt_sha256),
    ):
        asset = current.get(asset_name)
        if asset is None:
            continue
        try:
            payload = api.download_release_asset(
                _asset_id(asset, asset_name)
            )
            if sha256_bytes(payload) != expected_sha256:
                errors.append(
                    f"rollback found a conflicting {asset_name} and refused to delete it"
                )
            else:
                api.delete_release_asset(_asset_id(asset, asset_name))
        except AttachmentError as exc:
            verification = current_assets()
            if verification is None or asset_name in verification:
                errors.append(f"rollback could not remove {asset_name}: {exc}")
    return errors


def publish_attachment(
    api: GitHubApi,
    inputs: DispatchInputs,
    *,
    stage_directory: Path,
    staged_content_manifest_sha256: str,
    verification_run_id: int,
) -> dict[str, Any]:
    remote = verify_remote_contract(api, inputs)
    candidate_path, attestation = _validate_stage(
        stage_directory,
        inputs,
        staged_content_manifest_sha256=staged_content_manifest_sha256,
        verification_run_id=verification_run_id,
        tag_commit=remote["tag_commit"],
    )
    with tempfile.TemporaryDirectory(prefix="proaim-cuda-publish-reverify-") as temporary:
        reverify_root = Path(temporary)
        candidate_record = validate_and_extract_candidate(
            candidate_path,
            expected_sha256=inputs.cuda_zip_sha256,
            expected_commit=remote["tag_commit"],
            extract_directory=reverify_root / "candidate",
        )
        evidence_record = validate_physical_evidence(
            stage_directory / QUALIFICATION_EVIDENCE_ARCHIVE_NAME,
            inputs=inputs,
            remote=remote,
            candidate_record=candidate_record,
            extract_directory=reverify_root / "evidence",
        )
    if attestation.get("candidate") != candidate_record:
        raise AttachmentError("staged candidate attestation differs from publication revalidation")
    staged_qualification = attestation.get("physical_qualification")
    if not isinstance(staged_qualification, dict) or any(
        staged_qualification.get(key) != value for key, value in evidence_record.items()
    ):
        raise AttachmentError("staged physical evidence differs from publication revalidation")
    if attestation.get("tag_commit") != remote["tag_commit"]:
        raise AttachmentError("tag moved after read-only verification; refusing publication")
    if attestation.get("existing_release_id") != remote["release_id"]:
        raise AttachmentError("release identity changed after read-only verification")
    if attestation.get("source_build") != remote["build_run"]:
        raise AttachmentError("source build identity changed after read-only verification")
    if attestation.get("source_artifact") != remote["build_artifact"]:
        raise AttachmentError("source artifact identity changed after read-only verification")
    qualification = attestation.get("physical_qualification")
    if not isinstance(qualification, dict):
        raise AttachmentError("staged attestation omitted physical qualification")
    if qualification.get("qualification_run") != remote["qualification_run"]:
        raise AttachmentError("qualification run identity changed after read-only verification")
    if qualification.get("qualification_artifact") != remote["qualification_artifact"]:
        raise AttachmentError("qualification artifact identity changed after read-only verification")
    release_id = int(remote["release_id"])
    assets_before = _release_assets(api, release_id)
    by_name = _assets_by_name(assets_before)
    if CUDA_ARCHIVE_NAME in by_name or CUDA_QUALIFICATION_RECEIPT_NAME in by_name:
        raise AttachmentError(
            "release already has a CUDA archive or qualification receipt; refusing to overwrite it"
        )
    checksum_asset = by_name.get(CHECKSUM_ASSET_NAME)
    if checksum_asset is None:
        raise AttachmentError(f"existing release is missing {CHECKSUM_ASSET_NAME}")
    missing_archives = REQUIRED_EXISTING_ARCHIVES.difference(by_name)
    if missing_archives:
        raise AttachmentError(
            "existing release is missing required automatic assets: " + ", ".join(sorted(missing_archives))
        )
    original_manifest = api.download_release_asset(_asset_id(checksum_asset, CHECKSUM_ASSET_NAME))
    manifest = parse_checksum_manifest(original_manifest)
    # Every user-supplied release asset except the checksum document itself is
    # covered. This preserves the dual-GPU DirectML qualification receipts
    # when the separately qualified CUDA bundle is attached later.
    existing_checksummed_names = {
        name for name in by_name if name != CHECKSUM_ASSET_NAME
    }
    if set(manifest) != existing_checksummed_names:
        missing = sorted(existing_checksummed_names.difference(manifest))
        stale = sorted(set(manifest).difference(existing_checksummed_names))
        raise AttachmentError(
            "existing checksum/ZIP asset set mismatch"
            + (f"; unlisted assets: {', '.join(missing)}" if missing else "")
            + (f"; stale entries: {', '.join(stale)}" if stale else "")
        )
    for filename, expected_digest in sorted(manifest.items()):
        payload = api.download_release_asset(_asset_id(by_name[filename], filename))
        actual_digest = sha256_bytes(payload)
        if actual_digest != expected_digest:
            raise AttachmentError(
                f"existing release asset {filename} does not match SHA256SUMS.txt"
            )
    new_manifest_records = dict(manifest)
    new_manifest_records[CUDA_ARCHIVE_NAME] = inputs.cuda_zip_sha256
    receipt_path = stage_directory / CUDA_QUALIFICATION_RECEIPT_NAME
    receipt_sha256 = sha256_file(receipt_path)
    new_manifest_records[CUDA_QUALIFICATION_RECEIPT_NAME] = receipt_sha256
    new_manifest = render_checksum_manifest(new_manifest_records)
    if _asset_signature(_release_assets(api, release_id)) != _asset_signature(assets_before):
        raise AttachmentError("release assets changed during verification; retry after reviewing the release")

    candidate_payload = candidate_path.read_bytes()
    receipt_payload = receipt_path.read_bytes()
    candidate_asset: Mapping[str, Any] | None = None
    receipt_asset: Mapping[str, Any] | None = None
    new_checksum_asset: Mapping[str, Any] | None = None
    fresh_tag_commit = resolve_tag_commit(api, inputs.tag)
    fresh_release = _release_for_tag(api, inputs.tag)
    if fresh_tag_commit != remote["tag_commit"] or fresh_release.get("id") != release_id:
        raise AttachmentError("tag or release identity changed immediately before publication")
    try:
        candidate_asset = api.upload_release_asset(
            release_id,
            CUDA_ARCHIVE_NAME,
            candidate_payload,
            "application/zip",
        )
        expected_after_candidate = tuple(
            sorted(
                (*_asset_signature(assets_before), _asset_signature([candidate_asset])[0])
            )
        )
        if _asset_signature(_release_assets(api, release_id)) != expected_after_candidate:
            raise AttachmentError("release assets changed after CUDA upload; aborting")
        receipt_asset = api.upload_release_asset(
            release_id,
            CUDA_QUALIFICATION_RECEIPT_NAME,
            receipt_payload,
            "application/json; charset=utf-8",
        )
        expected_after_receipt = tuple(
            sorted((*expected_after_candidate, _asset_signature([receipt_asset])[0]))
        )
        if _asset_signature(_release_assets(api, release_id)) != expected_after_receipt:
            raise AttachmentError("release assets changed after qualification receipt upload")
        api.delete_release_asset(_asset_id(checksum_asset, CHECKSUM_ASSET_NAME))
        new_checksum_asset = api.upload_release_asset(
            release_id,
            CHECKSUM_ASSET_NAME,
            new_manifest,
            "text/plain; charset=utf-8",
        )
        final_assets = _assets_by_name(_release_assets(api, release_id))
        final_candidate = final_assets.get(CUDA_ARCHIVE_NAME)
        final_receipt = final_assets.get(CUDA_QUALIFICATION_RECEIPT_NAME)
        final_checksum = final_assets.get(CHECKSUM_ASSET_NAME)
        if final_candidate is None or final_receipt is None or final_checksum is None:
            raise AttachmentError("release verification could not find newly uploaded assets")
        if _asset_id(final_candidate, CUDA_ARCHIVE_NAME) != _asset_id(candidate_asset, CUDA_ARCHIVE_NAME):
            raise AttachmentError("release CUDA asset identity changed after upload")
        if _asset_id(final_checksum, CHECKSUM_ASSET_NAME) != _asset_id(
            new_checksum_asset,
            CHECKSUM_ASSET_NAME,
        ):
            raise AttachmentError("release checksum asset identity changed after upload")
        if _asset_id(final_receipt, CUDA_QUALIFICATION_RECEIPT_NAME) != _asset_id(
            receipt_asset,
            CUDA_QUALIFICATION_RECEIPT_NAME,
        ):
            raise AttachmentError("release qualification receipt identity changed after upload")
        expected_final_signature = tuple(
            sorted(
                (
                    *(
                        record
                        for record in _asset_signature(assets_before)
                        if record[0] != _asset_id(checksum_asset, CHECKSUM_ASSET_NAME)
                    ),
                    _asset_signature([candidate_asset])[0],
                    _asset_signature([receipt_asset])[0],
                    _asset_signature([new_checksum_asset])[0],
                )
            )
        )
        if _asset_signature(final_assets.values()) != expected_final_signature:
            raise AttachmentError("release asset set changed during final publication verification")
        if sha256_bytes(api.download_release_asset(_asset_id(final_candidate, CUDA_ARCHIVE_NAME))) != inputs.cuda_zip_sha256:
            raise AttachmentError("published CUDA asset failed final SHA-256 verification")
        if sha256_bytes(
            api.download_release_asset(
                _asset_id(final_receipt, CUDA_QUALIFICATION_RECEIPT_NAME)
            )
        ) != receipt_sha256:
            raise AttachmentError("published qualification receipt failed final SHA-256 verification")
        if api.download_release_asset(_asset_id(final_checksum, CHECKSUM_ASSET_NAME)) != new_manifest:
            raise AttachmentError("published SHA256SUMS.txt failed final byte verification")
        if resolve_tag_commit(api, inputs.tag) != remote["tag_commit"]:
            raise AttachmentError("tag moved during publication")
        final_release = _release_for_tag(api, inputs.tag)
        if final_release.get("id") != release_id:
            raise AttachmentError("release identity changed during publication")
    except BaseException as exc:
        rollback_errors = _rollback_publication(
            api,
            release_id,
            original_manifest=original_manifest,
            attempted_manifest=new_manifest,
            candidate_sha256=inputs.cuda_zip_sha256,
            receipt_sha256=receipt_sha256,
        )
        detail = f"publication failed and rollback was attempted: {exc}"
        if rollback_errors:
            detail += "; ROLLBACK ERRORS: " + " | ".join(rollback_errors)
        raise AttachmentError(detail) from exc

    return {
        "status": "published_and_verified",
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": remote["tag_commit"],
        "release_html_url": remote["release_html_url"],
        "cuda_asset": {
            "name": CUDA_ARCHIVE_NAME,
            "sha256": inputs.cuda_zip_sha256,
            "asset_id": _asset_id(candidate_asset, CUDA_ARCHIVE_NAME),
        },
        "checksum_asset": {
            "name": CHECKSUM_ASSET_NAME,
            "sha256": sha256_bytes(new_manifest),
            "asset_id": _asset_id(new_checksum_asset, CHECKSUM_ASSET_NAME),
            "entries": new_manifest_records,
        },
        "qualification_receipt_asset": {
            "name": CUDA_QUALIFICATION_RECEIPT_NAME,
            "sha256": receipt_sha256,
            "asset_id": _asset_id(receipt_asset, CUDA_QUALIFICATION_RECEIPT_NAME),
        },
        "qualification_evidence_sha256": inputs.qualification_evidence_sha256,
        "qualification_manifest_sha256": inputs.qualification_manifest_sha256,
        "physical_attestation_sha256": inputs.physical_attestation_sha256,
        "published_at_utc": _now_utc(),
    }


def _add_dispatch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-build-run-id", required=True)
    parser.add_argument("--evidence-run-id", required=True)
    parser.add_argument("--evidence-artifact-name", required=True)
    parser.add_argument("--cuda-zip-sha256", required=True)
    parser.add_argument("--qualification-evidence-sha256", required=True)
    parser.add_argument("--qualification-manifest-sha256", required=True)
    parser.add_argument("--physical-attestation-sha256", required=True)
    parser.add_argument("--qualified-gpu", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--nvidia-redistribution-confirmation", required=True)


def _inputs_from_args(args: argparse.Namespace) -> DispatchInputs:
    return validate_dispatch_inputs(
        repository=args.repository,
        tag=args.tag,
        build_run_id=args.source_build_run_id,
        evidence_run_id=args.evidence_run_id,
        evidence_artifact_name=args.evidence_artifact_name,
        cuda_zip_sha256=args.cuda_zip_sha256,
        qualification_evidence_sha256=args.qualification_evidence_sha256,
        qualification_manifest_sha256=args.qualification_manifest_sha256,
        physical_attestation_sha256=args.physical_attestation_sha256,
        qualified_gpu=args.qualified_gpu,
        confirmation=args.confirmation,
        nvidia_redistribution_confirmation=args.nvidia_redistribution_confirmation,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    content_manifest = subparsers.add_parser("write-content-manifest")
    content_manifest.add_argument("--root", required=True, type=Path)
    content_manifest.add_argument("--output-name", required=True)
    content_manifest.add_argument("--kind", required=True)
    content_manifest.add_argument("--repository", required=True)
    content_manifest.add_argument("--tag", required=True)
    content_manifest.add_argument("--tag-commit", required=True)
    content_manifest.add_argument("--source-build-run-id", required=True)
    content_manifest.add_argument("--qualification-run-id")
    content_manifest.add_argument("--qualification-run-attempt")
    content_manifest.add_argument("--verification-run-id")
    content_manifest.add_argument("--result", required=True, type=Path)

    verify_source = subparsers.add_parser("verify-source")
    verify_source.add_argument("--repository", required=True)
    verify_source.add_argument("--tag", required=True)
    verify_source.add_argument("--source-build-run-id", required=True)
    verify_source.add_argument("--output", required=True, type=Path)

    inspect_candidate = subparsers.add_parser("inspect-candidate")
    inspect_candidate.add_argument("--downloaded-directory", required=True, type=Path)
    inspect_candidate.add_argument("--cuda-zip-sha256", required=True)
    inspect_candidate.add_argument("--tag-commit", required=True)
    inspect_candidate.add_argument("--extract-directory", required=True, type=Path)
    inspect_candidate.add_argument("--output", required=True, type=Path)

    seal = subparsers.add_parser("seal-evidence")
    seal.add_argument("--raw-directory", required=True, type=Path)
    seal.add_argument("--output-directory", required=True, type=Path)
    seal.add_argument("--output-record", required=True, type=Path)
    seal.add_argument("--repository", required=True)
    seal.add_argument("--tag", required=True)
    seal.add_argument("--source-build-run-id", required=True)
    seal.add_argument("--cuda-zip-sha256", required=True)
    seal.add_argument("--qualified-gpu", required=True)
    seal.add_argument("--observer-name", required=True)
    seal.add_argument("--typed-confirmation", required=True)
    seal.add_argument("--github-actor", required=True)
    seal.add_argument("--github-run-id", required=True)
    seal.add_argument("--github-run-attempt", required=True)
    seal.add_argument("--github-head-sha", required=True)
    seal.add_argument("--raw-content-manifest-sha256", required=True)

    metadata = subparsers.add_parser("verify-metadata")
    _add_dispatch_arguments(metadata)
    metadata.add_argument("--output", required=True, type=Path)

    prepare = subparsers.add_parser("prepare")
    _add_dispatch_arguments(prepare)
    prepare.add_argument("--downloaded-directory", required=True, type=Path)
    prepare.add_argument("--evidence-downloaded-directory", required=True, type=Path)
    prepare.add_argument("--stage-directory", required=True, type=Path)
    prepare.add_argument("--extract-directory", required=True, type=Path)
    prepare.add_argument("--evidence-extract-directory", required=True, type=Path)
    prepare.add_argument("--github-run-id", required=True)
    prepare.add_argument("--github-actor", required=True)

    smoke = subparsers.add_parser("validate-smoke")
    smoke.add_argument("--stage-directory", required=True, type=Path)
    smoke.add_argument("--runtime-info", required=True, type=Path)
    smoke.add_argument("--benchmark", required=True, type=Path)

    publish = subparsers.add_parser("publish")
    _add_dispatch_arguments(publish)
    publish.add_argument("--stage-directory", required=True, type=Path)
    publish.add_argument("--staged-content-manifest-sha256", required=True)
    publish.add_argument("--verification-run-id", required=True)
    publish.add_argument("--output", required=True, type=Path)
    return parser


def _api_for(inputs: DispatchInputs) -> GitHubApi:
    return GitHubApi(os.environ.get("GH_TOKEN", ""), inputs.repository)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-smoke":
            result = validate_hosted_smoke(
                stage_directory=args.stage_directory.resolve(),
                runtime_info_path=args.runtime_info.resolve(),
                benchmark_path=args.benchmark.resolve(),
            )
        elif args.command == "write-content-manifest":
            repository = args.repository.strip()
            tag = args.tag.strip()
            if not REPOSITORY_RE.fullmatch(repository) or not TAG_RE.fullmatch(tag):
                raise AttachmentError("content manifest repository or tag is invalid")
            context: dict[str, Any] = {
                "repository": repository,
                "tag": tag,
                "tag_commit": _require_sha(args.tag_commit, "content manifest tag commit"),
                "source_build_run_id": _normalize_run_id(args.source_build_run_id),
            }
            if args.kind == "proaim-cuda-raw-qualification":
                context["qualification_run_id"] = _normalize_run_id(
                    str(args.qualification_run_id or "")
                )
                context["qualification_run_attempt"] = _normalize_run_id(
                    str(args.qualification_run_attempt or "")
                )
            elif args.kind == "proaim-cuda-verified-stage":
                context["verification_run_id"] = _normalize_run_id(
                    str(args.verification_run_id or "")
                )
            else:
                raise AttachmentError("unsupported content manifest kind")
            root = args.root.resolve()
            result = write_content_manifest(
                root=root,
                output=root / args.output_name,
                kind=args.kind,
                context=context,
            )
            _write_json_atomic(args.result.resolve(), result)
        elif args.command == "verify-source":
            repository = args.repository.strip()
            tag = args.tag.strip()
            if not REPOSITORY_RE.fullmatch(repository):
                raise AttachmentError("repository must use the exact OWNER/REPOSITORY form")
            if not TAG_RE.fullmatch(tag):
                raise AttachmentError("tag must be a short v* release tag")
            api = GitHubApi(os.environ.get("GH_TOKEN", ""), repository)
            result = verify_source_build_contract(
                api,
                repository=repository,
                tag=tag,
                build_run_id=_normalize_run_id(args.source_build_run_id),
            )
            _write_json_atomic(args.output.resolve(), result)
        elif args.command == "inspect-candidate":
            candidate = _single_downloaded_candidate(args.downloaded_directory.resolve())
            result = validate_and_extract_candidate(
                candidate,
                expected_sha256=_normalize_sha256(
                    args.cuda_zip_sha256, "CUDA ZIP SHA-256"
                ),
                expected_commit=_require_sha(args.tag_commit, "tag commit"),
                extract_directory=args.extract_directory.resolve(),
            )
            _write_json_atomic(args.output.resolve(), result)
        elif args.command == "seal-evidence":
            output_record = args.output_record.resolve()
            if output_record.exists():
                raise AttachmentError(f"refusing to overwrite output record: {output_record}")
            result = seal_physical_evidence(
                raw_directory=args.raw_directory.resolve(),
                output_directory=args.output_directory.resolve(),
                repository=args.repository,
                tag=args.tag,
                source_build_run_id=args.source_build_run_id,
                cuda_zip_sha256=args.cuda_zip_sha256,
                qualified_gpu=args.qualified_gpu,
                observer_name=args.observer_name,
                typed_confirmation=args.typed_confirmation,
                github_actor=args.github_actor,
                github_run_id=args.github_run_id,
                github_run_attempt=args.github_run_attempt,
                github_head_sha=args.github_head_sha,
                raw_content_manifest_sha256=args.raw_content_manifest_sha256,
            )
            _write_json_atomic(output_record, result)
        else:
            inputs = _inputs_from_args(args)
            api = _api_for(inputs)
            if args.command == "verify-metadata":
                result = verify_remote_contract(api, inputs)
                _write_json_atomic(args.output.resolve(), result)
            elif args.command == "prepare":
                result = prepare_attachment(
                    api,
                    inputs,
                    downloaded_directory=args.downloaded_directory.resolve(),
                    evidence_downloaded_directory=args.evidence_downloaded_directory.resolve(),
                    stage_directory=args.stage_directory.resolve(),
                    extract_directory=args.extract_directory.resolve(),
                    evidence_extract_directory=args.evidence_extract_directory.resolve(),
                    github_run_id=args.github_run_id,
                    github_actor=args.github_actor,
                )
            elif args.command == "publish":
                result = publish_attachment(
                    api,
                    inputs,
                    stage_directory=args.stage_directory.resolve(),
                    staged_content_manifest_sha256=_normalize_sha256(
                        args.staged_content_manifest_sha256,
                        "staged content manifest SHA-256",
                    ),
                    verification_run_id=_normalize_run_id(args.verification_run_id),
                )
                _write_json_atomic(args.output.resolve(), result)
            else:  # pragma: no cover - argparse constrains this
                parser.error(f"unsupported command: {args.command}")
                return 2
    except AttachmentError as exc:
        print(f"CUDA release attachment rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
