from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
import zipfile

from scripts.manage_cuda_release_attachment import (
    AttachmentError,
    CUDA_ARCHIVE_NAME,
    CUDA_RUNNER_INVARIANT_NAME,
    CUDA_QUALIFICATION_RECEIPT_NAME,
    EXPECTED_BUILD_ARTIFACT,
    EXPECTED_BUILD_WORKFLOW,
    EXPECTED_QUALIFICATION_ARTIFACT,
    EXPECTED_QUALIFICATION_WORKFLOW,
    PHYSICAL_ATTESTATION_NAME,
    QUALIFICATION_EVIDENCE_ARCHIVE_NAME,
    REQUIRED_EVIDENCE_FILES,
    RAW_CONTENT_MANIFEST_NAME,
    STAGED_CONTENT_MANIFEST_NAME,
    LIVE_TIMING_FIELDS,
    NVIDIA_TELEMETRY_NAME,
    parse_checksum_manifest,
    publish_attachment,
    render_checksum_manifest,
    resolve_tag_commit,
    validate_hosted_smoke,
    validate_and_extract_candidate,
    validate_dispatch_inputs,
    validate_physical_evidence,
    verify_remote_contract,
    seal_physical_evidence,
    write_content_manifest,
)
from scripts.write_nvidia_redistribution_manifest import (
    LIBRARY_FAMILIES,
    MANIFEST_NAME,
    NOTICE_MARKERS,
    NVIDIA_DISTRIBUTIONS,
    NvidiaManifestError,
    validate_manifest,
    write_manifest,
)


COMMIT = "a" * 40
ZIP_SHA = "1" * 64
EVIDENCE_SHA = "2" * 64
QUALIFICATION_MANIFEST_SHA = "3" * 64
PHYSICAL_ATTESTATION_SHA = "4" * 64


def _inputs(**overrides: str):
    values = {
        "repository": "owner/proaim",
        "tag": "v1.2.3",
        "build_run_id": "42",
        "evidence_run_id": "84",
        "evidence_artifact_name": EXPECTED_QUALIFICATION_ARTIFACT,
        "cuda_zip_sha256": ZIP_SHA,
        "qualification_evidence_sha256": EVIDENCE_SHA,
        "qualification_manifest_sha256": QUALIFICATION_MANIFEST_SHA,
        "physical_attestation_sha256": PHYSICAL_ATTESTATION_SHA,
        "qualified_gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "confirmation": "ATTACH QUALIFIED WINDOWS CUDA TO v1.2.3",
        "nvidia_redistribution_confirmation": (
            "I APPROVE NVIDIA REDISTRIBUTION REVIEW FOR v1.2.3"
        ),
    }
    values.update(overrides)
    return validate_dispatch_inputs(**values)


class FakeApi:
    def __init__(self, *, annotated: bool = False) -> None:
        tag_object = {"type": "tag", "sha": "b" * 40} if annotated else {
            "type": "commit",
            "sha": COMMIT,
        }
        self.responses = {
            "/git/ref/tags/v1.2.3": {
                "ref": "refs/tags/v1.2.3",
                "object": tag_object,
            },
            "/git/tags/" + "b" * 40: {
                "object": {"type": "commit", "sha": COMMIT},
            },
            "/releases/tags/v1.2.3": {
                "id": 8,
                "tag_name": "v1.2.3",
                "draft": False,
                "html_url": "https://example.invalid/release",
            },
            "/actions/runs/42": {
                "event": "workflow_dispatch",
                "status": "completed",
                "conclusion": "success",
                "head_sha": COMMIT,
                "repository": {"full_name": "owner/proaim"},
                "head_repository": {"full_name": "owner/proaim"},
                "workflow_id": 7,
                "path": EXPECTED_BUILD_WORKFLOW,
                "html_url": "https://example.invalid/run",
            },
            "/actions/workflows/7": {"path": EXPECTED_BUILD_WORKFLOW},
            "/actions/runs/84": {
                "event": "workflow_dispatch",
                "status": "completed",
                "conclusion": "success",
                "head_sha": COMMIT,
                "repository": {"full_name": "owner/proaim"},
                "head_repository": {"full_name": "owner/proaim"},
                "workflow_id": 11,
                "path": EXPECTED_QUALIFICATION_WORKFLOW,
                "html_url": "https://example.invalid/qualification-run",
                "run_attempt": 1,
                "actor": {"login": "gpu-tester"},
            },
            "/actions/workflows/11": {"path": EXPECTED_QUALIFICATION_WORKFLOW},
        }
        self.artifacts = [
            {
                "id": 9,
                "name": EXPECTED_BUILD_ARTIFACT,
                "expired": False,
                "size_in_bytes": 123,
                "digest": "sha256:" + "4" * 64,
            }
        ]
        self.evidence_artifacts = [
            {
                "id": 12,
                "name": EXPECTED_QUALIFICATION_ARTIFACT,
                "expired": False,
                "size_in_bytes": 456,
                "digest": "sha256:" + "5" * 64,
            }
        ]

    def get_json(self, path: str):
        return self.responses[path]

    def get_paginated(self, path: str, key: str):
        self.last_pagination = (path, key)
        if path == "/actions/runs/84/artifacts":
            return list(self.evidence_artifacts)
        return list(self.artifacts)


class FakeReleaseApi(FakeApi):
    def __init__(
        self,
        *,
        fail_first_checksum_upload: bool = False,
        commit_then_raise_name: str | None = None,
    ) -> None:
        super().__init__()
        linux = b"linux-archive"
        directml = b"directml-archive"
        from scripts.manage_cuda_release_attachment import sha256_bytes

        checksums = render_checksum_manifest(
            {
                "ProAim-Linux-x64.zip": sha256_bytes(linux),
                "ProAim-Windows-x64-DirectML.zip": sha256_bytes(directml),
            }
        )
        self.release_assets = {
            101: {
                "id": 101,
                "name": "ProAim-Linux-x64.zip",
                "size": len(linux),
            },
            102: {
                "id": 102,
                "name": "ProAim-Windows-x64-DirectML.zip",
                "size": len(directml),
            },
            103: {"id": 103, "name": "SHA256SUMS.txt", "size": len(checksums)},
        }
        self.release_payloads = {101: linux, 102: directml, 103: checksums}
        self.next_asset_id = 200
        self.fail_first_checksum_upload = fail_first_checksum_upload
        self.failed_checksum_upload = False
        self.commit_then_raise_name = commit_then_raise_name
        self.committed_failure = False

    def get_json_list(self, path: str):
        self.last_release_page = path
        if "page=1" in path:
            return [dict(asset) for asset in self.release_assets.values()]
        return []

    def download_release_asset(self, asset_id: int) -> bytes:
        return self.release_payloads[asset_id]

    def upload_release_asset(
        self,
        release_id: int,
        name: str,
        payload: bytes,
        content_type: str,
    ):
        if any(asset["name"].casefold() == name.casefold() for asset in self.release_assets.values()):
            raise AttachmentError("duplicate upload")
        if (
            name == "SHA256SUMS.txt"
            and self.fail_first_checksum_upload
            and not self.failed_checksum_upload
        ):
            self.failed_checksum_upload = True
            raise AttachmentError("simulated checksum upload failure")
        asset_id = self.next_asset_id
        self.next_asset_id += 1
        asset = {"id": asset_id, "name": name, "size": len(payload)}
        self.release_assets[asset_id] = asset
        self.release_payloads[asset_id] = bytes(payload)
        if name == self.commit_then_raise_name and not self.committed_failure:
            self.committed_failure = True
            raise AttachmentError("simulated response loss after server-side upload")
        return dict(asset)

    def delete_release_asset(self, asset_id: int) -> None:
        if asset_id not in self.release_assets:
            raise AttachmentError("missing delete target")
        del self.release_assets[asset_id]
        del self.release_payloads[asset_id]


class MovingTagReleaseApi(FakeReleaseApi):
    def get_json(self, path: str):
        if (
            path == "/git/ref/tags/v1.2.3"
            and any(
                asset["name"] == CUDA_ARCHIVE_NAME
                for asset in self.release_assets.values()
            )
            and any(
                asset["name"] == "SHA256SUMS.txt"
                and asset["id"] != 103
                for asset in self.release_assets.values()
            )
        ):
            return {
                "ref": "refs/tags/v1.2.3",
                "object": {"type": "commit", "sha": "c" * 40},
            }
        return super().get_json(path)


def _dependency_distribution() -> dict:
    return {
        "canonical_name": "onnxruntime-gpu",
        "installed_files": {
            "aggregate_sha256": "a" * 64,
            "record_document_sha256": "b" * 64,
            "record_entry_count": 3,
            "record_sha256_entries_verified": 2,
            "total_size_bytes": 123,
            "unhashed_record_entries": 1,
        },
        "installed_record_sha256": "b" * 64,
        "version": "1.28.0",
    }


def _write_fake_cuda_bundle(root: Path, *, commit: str = COMMIT, dirty: bool = False) -> None:
    from scripts.manage_cuda_release_attachment import sha256_file

    (root / "_internal" / "models" / "release_default").mkdir(parents=True)
    (root / "_internal" / "models" / "release_labels.txt").write_text(
        "player\n", encoding="utf-8"
    )
    (root / "_internal" / "models" / "release_default" / "player.onnx").write_bytes(
        b"fake-onnx"
    )
    model_path = root / "_internal" / "models" / "release_default" / "player.onnx"
    labels_path = root / "_internal" / "models" / "release_labels.txt"
    for filename in (
        "ProAimCLI.exe",
        "Qualify-ProAimGpu.ps1",
        "LICENSE",
    ):
        (root / filename).write_bytes((filename + "\n").encode())
    notices = "\n".join(NOTICE_MARKERS) + "\n"
    (root / "THIRD_PARTY_NOTICES.md").write_text(notices, encoding="utf-8")
    dependency_manifest = {
        "application": "ProAim",
        "artifact_hash_contract": {"enforced_before_install": True},
        "distributions": [_dependency_distribution()],
        "lock_profile": "windows-cuda-py313",
        "runtime_variant": "cuda",
        "schema_version": 1,
    }
    dependency_path = root / "DEPENDENCY-MANIFEST.json"
    dependency_path.write_text(
        json.dumps(dependency_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "BUILD-INFO.json").write_text(
        json.dumps(
            {
                "application": "ProAim",
                "commit": commit,
                "dependency_manifest": {
                    "distribution_count": 1,
                    "lock_profile": "windows-cuda-py313",
                    "path": "DEPENDENCY-MANIFEST.json",
                    "schema_version": 1,
                    "sha256": sha256_file(dependency_path),
                },
                "dirty": dirty,
                "release_default_model": {
                    "preset": "release_player_rectangular",
                    "model_path": "_internal/models/release_default/player.onnx",
                    "labels_path": "_internal/models/release_labels.txt",
                    "input_shape_hw": [384, 640],
                    "detail_crop_size_source_pixels": 0,
                    "model_sha256": sha256_file(model_path),
                    "labels_sha256": sha256_file(labels_path),
                },
                "runtime_variant": "cuda",
                "schema": 2,
            }
        ),
        encoding="utf-8",
    )
    metadata_root = root / "_internal"
    for index, distribution in enumerate(NVIDIA_DISTRIBUTIONS):
        dist_info = metadata_root / f"{distribution.replace('-', '_')}-1.0.{index}.dist-info"
        (dist_info / "licenses").mkdir(parents=True)
        (dist_info / "METADATA").write_text(
            "\n".join(
                (
                    "Metadata-Version: 2.4",
                    f"Name: {distribution}",
                    f"Version: 1.0.{index}",
                    "License-Expression: LicenseRef-NVIDIA-Proprietary",
                    "License-File: License.txt",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (dist_info / "WHEEL").write_text("Wheel-Version: 1.0\n", encoding="utf-8")
        (dist_info / "licenses" / "License.txt").write_text(
            f"NVIDIA terms for {distribution}\n", encoding="utf-8"
        )
    nvidia_bin = root / "_internal" / "nvidia" / "bin"
    nvidia_bin.mkdir(parents=True)
    for family, markers in LIBRARY_FAMILIES.items():
        (nvidia_bin / f"{markers[0]}64_test.dll").write_bytes(family.encode())
    write_manifest(root)


def _zip_bundle(bundle: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                output.write(path, "ProAim/" + path.relative_to(bundle).as_posix())


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_raw_physical_evidence(root: Path) -> tuple[Path, dict, dict]:
    from scripts.manage_cuda_release_attachment import sha256_file, verify_source_build_contract

    raw = root / "raw"
    software = raw / "software-evidence"
    software.mkdir(parents=True)
    base_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    def stamp(offset: int) -> str:
        return (base_time + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")
    dependency_manifest = {
        "application": "ProAim",
        "artifact_hash_contract": {"enforced_before_install": True},
        "distributions": [_dependency_distribution()],
        "lock_profile": "windows-cuda-py313",
        "runtime_variant": "cuda",
        "schema_version": 1,
    }
    _write_json(software / "bundle-DEPENDENCY-MANIFEST.json", dependency_manifest)
    build_info = {
        "application": "ProAim",
        "commit": COMMIT,
        "dependency_manifest": {
            "distribution_count": 1,
            "lock_profile": "windows-cuda-py313",
            "path": "DEPENDENCY-MANIFEST.json",
            "schema_version": 1,
            "sha256": sha256_file(software / "bundle-DEPENDENCY-MANIFEST.json"),
        },
        "dirty": False,
        "release_default_model": {
            "preset": "release_player_rectangular",
            "model_path": "_internal/models/release_default/player.onnx",
            "labels_path": "_internal/models/release_labels.txt",
            "input_shape_hw": [384, 640],
            "detail_crop_size_source_pixels": 0,
            "model_sha256": "8" * 64,
            "labels_sha256": "9" * 64,
        },
        "runtime_variant": "cuda",
        "schema": 2,
    }
    _write_json(software / "bundle-BUILD-INFO.json", build_info)
    (software / "TASK-MANAGER-INSTRUCTIONS.txt").write_text("watch Task Manager\n", encoding="utf-8")
    (software / "TASK-MANAGER-CONFIRMATION.txt").write_text("pending\n", encoding="utf-8")
    runtime = {
        "frozen": True,
        "onnxruntime_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }
    _write_json(software / "runtime-info.json", runtime)
    (software / "runtime-info.stderr.txt").write_text("", encoding="utf-8")
    provider = {
        "requested_provider": "CUDAExecutionProvider",
        "active_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "require_full_provider": True,
        "configured_session_options": {"disable_cpu_ep_fallback": True},
        "runtime_ep_fail_fallback_disabled": True,
        "provider_options_status": "ok",
        "provider_options": {"CUDAExecutionProvider": {"device_id": "0"}},
    }
    candidate_record = {
        "filename": CUDA_ARCHIVE_NAME,
        "sha256": ZIP_SHA,
        "size_bytes": 999,
        "build_info": build_info,
        "build_info_sha256": sha256_file(software / "bundle-BUILD-INFO.json"),
        "dependency_manifest_sha256": sha256_file(
            software / "bundle-DEPENDENCY-MANIFEST.json"
        ),
        "frozen_cli_sha256": "6" * 64,
        "qualification_helper_sha256": "7" * 64,
        "release_default_model": build_info["release_default_model"],
        "nvidia_redistribution_manifest_sha256": "a" * 64,
        "nvidia_distribution_versions": {"nvidia-cuda-runtime-cu12": "12.0"},
    }
    def timing_summary(*, samples: int, mean: float, minimum: float, maximum: float, p95: float) -> dict:
        return {
            "samples": samples,
            "mean": mean,
            "p50": mean,
            "median": mean,
            "p95": p95,
            "p99": max(p95, maximum - 0.25),
            "min": minimum,
            "max": maximum,
            "stdev": 0.5,
        }

    aggregate_timings = {
        "preprocess": timing_summary(samples=300, mean=2.0, minimum=1.0, maximum=3.0, p95=2.5),
        "inference": timing_summary(samples=300, mean=4.0, minimum=2.0, maximum=6.0, p95=5.0),
        "postprocess": timing_summary(samples=300, mean=2.0, minimum=1.0, maximum=3.0, p95=2.5),
        "pipeline": timing_summary(samples=300, mean=8.0, minimum=5.0, maximum=12.0, p95=10.0),
    }
    repeat_timings = [
        {
            name: {**summary, "samples": 100}
            for name, summary in aggregate_timings.items()
        }
        for _ in range(3)
    ]
    benchmark = {
        "generated_at_utc": stamp(15),
        "methodology": {
            "backend": "onnxruntime",
            "requested_device": "CUDA",
            "require_full_provider": True,
            "warmup_per_model": 30,
            "iterations_per_repeat": 100,
            "repeats": 3,
        },
        "input": {
            "kind": "synthetic",
            "count": 32,
            "generator": "numpy.default_rng(seed=0), uint8 720x1280 BGR",
        },
        "models": [
            {
                "key": "release-default",
                "input_shape_hw": [384, 640],
                "detail_crop_size_source_pixels": 0,
                "runtime": provider,
                "artifact": {
                    "files": [
                        {
                            "resolved_path": "C:/runner/ProAim/"
                            + build_info["release_default_model"]["model_path"],
                            "sha256": build_info["release_default_model"]["model_sha256"],
                        }
                    ]
                },
                "labels_artifact": {
                    "files": [
                        {
                            "resolved_path": "C:/runner/ProAim/"
                            + build_info["release_default_model"]["labels_path"],
                            "sha256": build_info["release_default_model"]["labels_sha256"],
                        }
                    ]
                },
                "timing_ms": aggregate_timings,
                "pipeline_fps_from_mean": 125.0,
                "repeats": [
                    {
                        "repeat": index,
                        "timing_ms": timing,
                        "detections_mean": 1.0,
                    }
                    for index, timing in enumerate(repeat_timings, 1)
                ],
            }
        ],
    }
    _write_json(software / "benchmark-release-default.json", benchmark)
    (software / "benchmark-release-default.stderr.txt").write_text("", encoding="utf-8")
    for suffix, preview, report_start in (
        ("no-preview", False, 30),
        ("preview-15", True, 80),
    ):
        report = {
            "started_utc": stamp(report_start),
            "completed_utc": stamp(report_start + 40),
            "detector_runtime": provider,
            "config": {
                "backend": "onnxruntime",
                "device": "CUDA",
                "require_full_provider": True,
                "stats_window": 1000,
                "source": {"kind": "screen", "value": None},
                "capture": {
                    "screen_monitor": 1,
                    "screen_region": None,
                    "screen_fps": 60.0,
                },
                "inference": {
                    "shape_hw": [384, 640],
                    "crop_size": None,
                    "detail_crop_size": None,
                },
                "preview": {"enabled": preview, "fps_limit": 15.0},
            },
            "model_artifact": {"sha256": build_info["release_default_model"]["model_sha256"]},
            "labels_artifact": {"sha256": build_info["release_default_model"]["labels_sha256"]},
            "preview": {
                "enabled": preview,
                "fps_limit": 15.0,
                "mode": "threaded" if preview else "disabled",
                "stats": (
                    {
                        "submitted_frames": 600,
                        "displayed_frames": 590,
                        "replaced_frames": 10,
                    }
                    if preview
                    else {}
                ),
            },
            "detail_pass": {"enabled": False, "requested_crop_size": None},
            "source": {"backend": "dxcam-dxgi", "fallback_reason": None},
            "capture": {"read_failures": 0},
            "pipeline": {
                "processed_frames": 1000,
                "rolling_sample_count": 1000,
                "elapsed_seconds": 40.0,
                "elapsed_fps": 25.0,
                "update_fps": 25.0,
                "timings": {
                    "unit": "milliseconds",
                    "fields": list(LIVE_TIMING_FIELDS),
                    **{
                        percentile: {
                            field: {
                                "capture_ms": 1.0,
                                "queue_age_ms": 2.0,
                                "preprocess_ms": 2.0,
                                "inference_ms": 4.0,
                                "postprocess_ms": 1.0,
                                "detail_preprocess_ms": 0.0,
                                "detail_inference_ms": 0.0,
                                "detail_postprocess_ms": 0.0,
                                "control_ms": 1.0,
                                "processing_ms": 8.0,
                                "freshness_latency_ms": 10.0,
                                "observed_pipeline_ms": 11.0,
                                "draw_ms": 1.0 if preview else 0.0,
                                "preview_service_ms": 0.5 if preview else 0.0,
                            }[field]
                            for field in LIVE_TIMING_FIELDS
                        }
                        for percentile in ("mean", "p50", "p95", "p99")
                    },
                },
            },
            "termination": {
                "reason": "max_frames",
                "requested_max_frames": 1000,
                "requested_max_seconds": 60.0,
            },
        }
        _write_json(software / f"live-release-default-{suffix}.json", report)
        (software / f"live-release-default-{suffix}.stdout.txt").write_text("done\n", encoding="utf-8")
        (software / f"live-release-default-{suffix}.stderr.txt").write_text("", encoding="utf-8")

    software_hashes = []
    for path in sorted(software.iterdir()):
        if path.name == "TASK-MANAGER-CONFIRMATION.txt":
            continue
        software_hashes.append(
            {"file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    def output_record(filename: str):
        path = software / filename
        return {"file": filename, "sha256": sha256_file(path)}
    runs = [
        {
            "name": "frozen runtime info",
            "exit_code": 0,
            "stdout": output_record("runtime-info.json"),
            "stderr": output_record("runtime-info.stderr.txt"),
            "started_at_utc": stamp(0),
            "completed_at_utc": stamp(5),
        },
        {
            "name": "model benchmark (release-default)",
            "exit_code": 0,
            "stdout": output_record("benchmark-release-default.json"),
            "stderr": output_record("benchmark-release-default.stderr.txt"),
            "started_at_utc": stamp(10),
            "completed_at_utc": stamp(20),
        },
    ]
    for suffix, start in (("no-preview", 30), ("preview-15", 80)):
        runs.append(
            {
                "name": f"live pipeline (release-default-{suffix})",
                "exit_code": 0,
                "stdout": output_record(f"live-release-default-{suffix}.stdout.txt"),
                "stderr": output_record(f"live-release-default-{suffix}.stderr.txt"),
                "metrics": output_record(f"live-release-default-{suffix}.json"),
                "started_at_utc": stamp(start),
                "completed_at_utc": stamp(start + 45),
            }
        )
    artifact_hashes = {
        "frozen_cli": candidate_record["frozen_cli_sha256"],
        "build_info": candidate_record["build_info_sha256"],
        "dependency_manifest": candidate_record["dependency_manifest_sha256"],
        "qualification_helper": candidate_record["qualification_helper_sha256"],
        "release_default_model": build_info["release_default_model"]["model_sha256"],
        "release_default_labels": build_info["release_default_model"]["labels_sha256"],
        "original_bundle_archive": ZIP_SHA,
    }
    manifest = {
        "schema_version": 1,
        "status": "software_checks_passed_physical_gpu_confirmation_pending",
        "qualified": False,
        "provider": {
            "selection": "CUDA",
            "requested_device": "CUDA",
            "expected_execution_provider": "CUDAExecutionProvider",
            "directml_adapter_index": None,
        },
        "bundle_build_info": build_info,
        "benchmark_bounds": {"samples": 32, "warmup": 30, "iterations": 100, "repeats": 3},
        "live_bounds": {
            "enabled": True,
            "selected_model": "release-default",
            "release_default_model": build_info["release_default_model"],
            "screen_monitor": 1,
            "screen_fps": 60.0,
            "max_frames": 1000,
            "max_seconds": 60.0,
            "detail_crop_size": None,
            "modes": ["no-preview", "preview-15"],
        },
        "input_artifacts": [
            {
                "role": role,
                "sha256": digest,
                **(
                    {
                        "path": build_info["release_default_model"][
                            "model_path" if role == "release_default_model" else "labels_path"
                        ],
                        "location": "bundle",
                    }
                    if role in {"release_default_model", "release_default_labels"}
                    else {}
                ),
            }
            for role, digest in artifact_hashes.items()
        ],
        "runs": runs,
        "evidence_files": software_hashes,
        "manual_confirmation": {"required": True, "completed_by_helper": False},
    }
    _write_json(software / "qualification-manifest.json", manifest)
    _write_json(raw / "candidate-inspection.json", candidate_record)
    source_record = verify_source_build_contract(
        FakeApi(), repository="owner/proaim", tag="v1.2.3", build_run_id=42
    )
    _write_json(raw / "verified-source.json", source_record)
    telemetry = []
    telemetry_samples = (
        (15, 35),
        *((offset, 32) for offset in range(35, 40)),
        *((offset, 30) for offset in range(85, 90)),
    )
    for offset, utilization in telemetry_samples:
        captured_at = stamp(offset)
        telemetry.append(
            {
                "schema_version": 1,
                "kind": "gpu",
                "captured_at_utc": captured_at,
                "nvidia_timestamp": captured_at,
                "gpu_index": 0,
                "gpu_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
                "gpu_uuid": "GPU-1234",
                "driver_version": "600.00",
                "compute_capability": "12.0",
                "utilization_gpu_percent": utilization,
                "memory_used_mib": 1024.0,
                "memory_total_mib": 8192.0,
            }
        )
        telemetry.append(
            {
                "schema_version": 1,
                "kind": "compute_process",
                "captured_at_utc": captured_at,
                "gpu_uuid": "GPU-1234",
                "pid": 1234,
                "process_name": "ProAimCLI.exe",
                "used_gpu_memory_mib": None,
                "used_gpu_memory_supported": False,
            }
        )
    (raw / "nvidia-smi-telemetry.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in telemetry), encoding="utf-8"
    )
    _write_json(
        raw / CUDA_RUNNER_INVARIANT_NAME,
        {
            "schema_version": 1,
            "status": "passed_before_gpu_runs",
            "checked_at_utc": stamp(0),
            "nvidia_gpu_count": 1,
            "nvidia_gpu_names": ["NVIDIA GeForce RTX 5060 Laptop GPU"],
            "preexisting_proaim_cli_count": 0,
            "cuda_visible_devices": None,
            "nvidia_visible_devices": None,
            "telemetry_interval_milliseconds": 500,
        },
    )
    _write_json(
        raw / "LOCAL-PHYSICAL-OBSERVATION.json",
        {
            "schema_version": 1,
            "status": "completed_after_automated_gpu_runs",
            "completed": True,
            "tag": "v1.2.3",
            "github_actor": "gpu-tester",
            "github_run_id": "84",
            "observer_name": "Physical Tester",
            "observed_at_utc": _recent_utc(),
            "physical_gpu_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
            "task_manager_gpu_engine": "GPU 1 - CUDA",
            "typed_confirmation": (
                "I ATTEST THAT I OBSERVED NVIDIA GeForce RTX 5060 Laptop GPU "
                "RUN CUDA FOR v1.2.3"
            ),
            "observations": {
                "release_default_benchmark": True,
                "live_no_preview": True,
                "live_preview_15": True,
            },
            "completion_method": (
                "interactive Windows desktop form after automated GPU runs"
            ),
        },
    )
    write_content_manifest(
        root=raw,
        output=raw / RAW_CONTENT_MANIFEST_NAME,
        kind="proaim-cuda-raw-qualification",
        context={
            "repository": "owner/proaim",
            "tag": "v1.2.3",
            "tag_commit": COMMIT,
            "source_build_run_id": 42,
            "qualification_run_id": 84,
            "qualification_run_attempt": 1,
        },
    )
    return raw, candidate_record, source_record


def _recent_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rewrite_raw_content_manifest(raw: Path) -> str:
    from scripts.manage_cuda_release_attachment import sha256_file

    manifest_path = raw / RAW_CONTENT_MANIFEST_NAME
    manifest_path.unlink(missing_ok=True)
    write_content_manifest(
        root=raw,
        output=manifest_path,
        kind="proaim-cuda-raw-qualification",
        context={
            "repository": "owner/proaim",
            "tag": "v1.2.3",
            "tag_commit": COMMIT,
            "source_build_run_id": 42,
            "qualification_run_id": 84,
            "qualification_run_attempt": 1,
        },
    )
    return sha256_file(manifest_path)


def _refresh_software_manifest(raw: Path) -> None:
    from scripts.manage_cuda_release_attachment import sha256_file

    software = raw / "software-evidence"
    path = software / "qualification-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for record in manifest["evidence_files"]:
        evidence_path = software / record["file"]
        record["sha256"] = sha256_file(evidence_path)
        record["size_bytes"] = evidence_path.stat().st_size
    for run in manifest["runs"]:
        for key in ("stdout", "stderr", "metrics"):
            record = run.get(key)
            if record is not None:
                record["sha256"] = sha256_file(software / record["file"])
    _write_json(path, manifest)


def _seal_fixture(raw: Path, output: Path):
    return seal_physical_evidence(
        raw_directory=raw,
        output_directory=output,
        repository="owner/proaim",
        tag="v1.2.3",
        source_build_run_id="42",
        cuda_zip_sha256=ZIP_SHA,
        qualified_gpu="NVIDIA GeForce RTX 5060 Laptop GPU",
        observer_name="Physical Tester",
        typed_confirmation=(
            "I ATTEST THAT I OBSERVED NVIDIA GeForce RTX 5060 Laptop GPU "
            "RUN CUDA FOR v1.2.3"
        ),
        github_actor="gpu-tester",
        github_run_id="84",
        github_run_attempt="1",
        github_head_sha=COMMIT,
        raw_content_manifest_sha256=_rewrite_raw_content_manifest(raw),
    )


class DispatchInputTests(unittest.TestCase):
    def test_exact_confirmation_runs_artifact_and_all_hashes_are_required(self) -> None:
        inputs = _inputs()
        self.assertEqual(inputs.build_run_id, 42)
        self.assertEqual(inputs.expected_confirmation, inputs.confirmation)
        self.assertEqual(inputs.cuda_zip_sha256, ZIP_SHA)
        self.assertEqual(inputs.evidence_run_id, 84)
        self.assertEqual(inputs.evidence_artifact_name, EXPECTED_QUALIFICATION_ARTIFACT)
        self.assertEqual(inputs.qualification_evidence_sha256, EVIDENCE_SHA)
        self.assertEqual(
            inputs.qualification_manifest_sha256,
            QUALIFICATION_MANIFEST_SHA,
        )
        self.assertEqual(inputs.physical_attestation_sha256, PHYSICAL_ATTESTATION_SHA)
        self.assertEqual(
            inputs.nvidia_redistribution_confirmation,
            inputs.expected_nvidia_confirmation,
        )

    def test_near_miss_confirmation_and_malformed_hash_fail_closed(self) -> None:
        with self.assertRaisesRegex(AttachmentError, "typed confirmation mismatch"):
            _inputs(confirmation="ATTACH QUALIFIED WINDOWS CUDA TO v1.2.3 ")
        with self.assertRaisesRegex(AttachmentError, "64 hexadecimal"):
            _inputs(cuda_zip_sha256="abc")
        with self.assertRaisesRegex(AttachmentError, "positive decimal"):
            _inputs(build_run_id="0")
        with self.assertRaisesRegex(AttachmentError, "must equal exactly"):
            _inputs(evidence_artifact_name="almost-the-right-artifact")
        with self.assertRaisesRegex(AttachmentError, "redistribution confirmation mismatch"):
            _inputs(nvidia_redistribution_confirmation="I skimmed it")


class RemoteContractTests(unittest.TestCase):
    def test_lightweight_and_annotated_tags_resolve_to_commit(self) -> None:
        self.assertEqual(resolve_tag_commit(FakeApi(), "v1.2.3"), COMMIT)
        self.assertEqual(resolve_tag_commit(FakeApi(annotated=True), "v1.2.3"), COMMIT)

    def test_successful_manual_exact_workflow_artifact_is_accepted(self) -> None:
        api = FakeApi(annotated=True)
        result = verify_remote_contract(api, _inputs())
        self.assertEqual(result["tag_commit"], COMMIT)
        self.assertEqual(result["build_run"]["workflow_path"], EXPECTED_BUILD_WORKFLOW)
        self.assertEqual(result["build_artifact"]["name"], EXPECTED_BUILD_ARTIFACT)
        self.assertEqual(
            result["qualification_run"]["workflow_path"],
            EXPECTED_QUALIFICATION_WORKFLOW,
        )
        self.assertEqual(
            result["qualification_artifact"]["name"],
            EXPECTED_QUALIFICATION_ARTIFACT,
        )

    def test_wrong_event_workflow_sha_or_duplicate_artifact_is_rejected(self) -> None:
        cases = []
        api = FakeApi()
        api.responses["/actions/runs/42"]["event"] = "push"
        cases.append(api)
        api = FakeApi()
        api.responses["/actions/workflows/7"]["path"] = ".github/workflows/other.yml"
        cases.append(api)
        api = FakeApi()
        api.responses["/actions/runs/42"]["head_sha"] = "c" * 40
        cases.append(api)
        api = FakeApi()
        api.artifacts.append(dict(api.artifacts[0], id=10))
        cases.append(api)
        api = FakeApi()
        api.responses["/actions/runs/84"]["head_sha"] = "d" * 40
        cases.append(api)
        api = FakeApi()
        api.responses["/actions/workflows/11"]["path"] = ".github/workflows/untrusted.yml"
        cases.append(api)
        api = FakeApi()
        api.evidence_artifacts[0]["digest"] = None
        cases.append(api)
        for api in cases:
            with self.subTest(api=api), self.assertRaises(AttachmentError):
                verify_remote_contract(api, _inputs())

    def test_evidence_wrong_event_conclusion_or_duplicate_artifact_is_rejected(self) -> None:
        cases = []
        api = FakeApi()
        api.responses["/actions/runs/84"]["event"] = "push"
        cases.append(api)
        api = FakeApi()
        api.responses["/actions/runs/84"]["conclusion"] = "failure"
        cases.append(api)
        api = FakeApi()
        api.evidence_artifacts.append(dict(api.evidence_artifacts[0], id=99))
        cases.append(api)
        api = FakeApi()
        api.evidence_artifacts[0]["expired"] = True
        cases.append(api)
        for api in cases:
            with self.subTest(api=api), self.assertRaises(AttachmentError):
                verify_remote_contract(api, _inputs())


class PhysicalEvidenceChainTests(unittest.TestCase):
    def test_sealed_evidence_round_trip_binds_reports_gpu_actor_and_candidate(self) -> None:
        from scripts.manage_cuda_release_attachment import sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, candidate_record, _ = _make_raw_physical_evidence(root)
            output = root / "sealed-output"
            sealed = seal_physical_evidence(
                raw_directory=raw,
                output_directory=output,
                repository="owner/proaim",
                tag="v1.2.3",
                source_build_run_id="42",
                cuda_zip_sha256=ZIP_SHA,
                qualified_gpu="NVIDIA GeForce RTX 5060 Laptop GPU",
                observer_name="Physical Tester",
                typed_confirmation=(
                    "I ATTEST THAT I OBSERVED NVIDIA GeForce RTX 5060 Laptop GPU "
                    "RUN CUDA FOR v1.2.3"
                ),
                github_actor="gpu-tester",
                github_run_id="84",
                github_run_attempt="1",
                github_head_sha=COMMIT,
                raw_content_manifest_sha256=sha256_file(
                    raw / RAW_CONTENT_MANIFEST_NAME
                ),
            )
            archive = output / QUALIFICATION_EVIDENCE_ARCHIVE_NAME
            inputs = _inputs(
                cuda_zip_sha256=ZIP_SHA,
                qualification_evidence_sha256=sha256_file(archive),
                qualification_manifest_sha256=sealed["qualification_manifest_sha256"],
                physical_attestation_sha256=sealed["physical_attestation_sha256"],
            )
            remote = verify_remote_contract(FakeApi(), inputs)
            result = validate_physical_evidence(
                archive,
                inputs=inputs,
                remote=remote,
                candidate_record=candidate_record,
                extract_directory=root / "extracted-evidence",
            )
            self.assertEqual(result["qualified_gpu"], inputs.qualified_gpu)
            self.assertTrue(
                result["nvidia_telemetry_summary"]["proaim_compute_process_observed"]
            )
            self.assertEqual(result["qualification_run"]["actor"], "gpu-tester")
            wrong_attestation = _inputs(
                cuda_zip_sha256=ZIP_SHA,
                qualification_evidence_sha256=sha256_file(archive),
                qualification_manifest_sha256=sealed["qualification_manifest_sha256"],
                physical_attestation_sha256="0" * 64,
            )
            with self.assertRaisesRegex(AttachmentError, "attestation SHA-256"):
                validate_physical_evidence(
                    archive,
                    inputs=wrong_attestation,
                    remote=verify_remote_contract(FakeApi(), wrong_attestation),
                    candidate_record=candidate_record,
                    extract_directory=root / "wrong-attestation-extract",
                )

    def test_helper_cannot_mark_its_pending_manifest_qualified(self) -> None:
        from scripts.manage_cuda_release_attachment import sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, _, _ = _make_raw_physical_evidence(root)
            manifest_path = raw / "software-evidence" / "qualification-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["qualified"] = True
            _write_json(manifest_path, manifest)
            (raw / RAW_CONTENT_MANIFEST_NAME).unlink()
            write_content_manifest(
                root=raw,
                output=raw / RAW_CONTENT_MANIFEST_NAME,
                kind="proaim-cuda-raw-qualification",
                context={
                    "repository": "owner/proaim",
                    "tag": "v1.2.3",
                    "tag_commit": COMMIT,
                    "source_build_run_id": 42,
                    "qualification_run_id": 84,
                    "qualification_run_attempt": 1,
                },
            )
            with self.assertRaisesRegex(AttachmentError, "qualified=false"):
                seal_physical_evidence(
                    raw_directory=raw,
                    output_directory=root / "rejected-output",
                    repository="owner/proaim",
                    tag="v1.2.3",
                    source_build_run_id="42",
                    cuda_zip_sha256=ZIP_SHA,
                    qualified_gpu="NVIDIA GeForce RTX 5060 Laptop GPU",
                    observer_name="Physical Tester",
                    typed_confirmation=(
                        "I ATTEST THAT I OBSERVED NVIDIA GeForce RTX 5060 Laptop GPU "
                        "RUN CUDA FOR v1.2.3"
                    ),
                    github_actor="gpu-tester",
                    github_run_id="84",
                    github_run_attempt="1",
                    github_head_sha=COMMIT,
                    raw_content_manifest_sha256=sha256_file(
                        raw / RAW_CONTENT_MANIFEST_NAME
                    ),
                )

    def test_evidence_archive_rejects_path_traversal_and_unexpected_files(self) -> None:
        from scripts.manage_cuda_release_attachment import _extract_evidence_archive

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / QUALIFICATION_EVIDENCE_ARCHIVE_NAME
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.txt", "bad")
            with self.assertRaisesRegex(AttachmentError, "unsafe path"):
                _extract_evidence_archive(archive, root / "extract-traversal")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / QUALIFICATION_EVIDENCE_ARCHIVE_NAME
            with zipfile.ZipFile(archive, "w") as output:
                for filename in REQUIRED_EVIDENCE_FILES:
                    output.writestr(filename, "{}" if filename.endswith(".json") else "x")
                output.writestr("not-allowed.txt", "bad")
            with self.assertRaisesRegex(AttachmentError, "unexpected files"):
                _extract_evidence_archive(archive, root / "extract-extra")

    def test_raw_content_manifest_rejects_post_upload_byte_mutation(self) -> None:
        from scripts.manage_cuda_release_attachment import sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, _, _ = _make_raw_physical_evidence(root)
            manifest_sha = sha256_file(raw / RAW_CONTENT_MANIFEST_NAME)
            with (raw / NVIDIA_TELEMETRY_NAME).open("a", encoding="utf-8") as stream:
                stream.write(" \n")
            with self.assertRaisesRegex(AttachmentError, "content size mismatch"):
                seal_physical_evidence(
                    raw_directory=raw,
                    output_directory=root / "rejected",
                    repository="owner/proaim",
                    tag="v1.2.3",
                    source_build_run_id="42",
                    cuda_zip_sha256=ZIP_SHA,
                    qualified_gpu="NVIDIA GeForce RTX 5060 Laptop GPU",
                    observer_name="Physical Tester",
                    typed_confirmation=(
                        "I ATTEST THAT I OBSERVED NVIDIA GeForce RTX 5060 Laptop GPU "
                        "RUN CUDA FOR v1.2.3"
                    ),
                    github_actor="gpu-tester",
                    github_run_id="84",
                    github_run_attempt="1",
                    github_head_sha=COMMIT,
                    raw_content_manifest_sha256=manifest_sha,
                )

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_telemetry(self) -> None:
        from scripts.manage_cuda_release_attachment import _read_json_object

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ambiguous = root / "ambiguous.json"
            ambiguous.write_text('{"qualified":true,"qualified":false}\n', encoding="utf-8")
            with self.assertRaisesRegex(AttachmentError, "strict JSON"):
                _read_json_object(ambiguous, "ambiguous evidence")

            raw, _, _ = _make_raw_physical_evidence(root)
            telemetry_path = raw / NVIDIA_TELEMETRY_NAME
            telemetry = telemetry_path.read_text(encoding="utf-8").replace(
                '"utilization_gpu_percent": 35',
                '"utilization_gpu_percent": NaN',
                1,
            )
            telemetry_path.write_text(telemetry, encoding="utf-8")
            with self.assertRaisesRegex(AttachmentError, "strict JSON"):
                _seal_fixture(raw, root / "rejected-nan")

    def test_performance_and_timeline_policy_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, _, _ = _make_raw_physical_evidence(root)
            benchmark_path = (
                raw / "software-evidence" / "benchmark-release-default.json"
            )
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            benchmark["models"][0]["timing_ms"]["inference"]["p95"] = 100.0
            benchmark["models"][0]["timing_ms"]["inference"]["p99"] = 110.0
            benchmark["models"][0]["timing_ms"]["inference"]["max"] = 120.0
            for repeat in benchmark["models"][0]["repeats"]:
                repeat["timing_ms"]["inference"]["p95"] = 100.0
                repeat["timing_ms"]["inference"]["p99"] = 110.0
                repeat["timing_ms"]["inference"]["max"] = 120.0
            _write_json(benchmark_path, benchmark)
            _refresh_software_manifest(raw)
            with self.assertRaisesRegex(AttachmentError, "p95 inference latency"):
                _seal_fixture(raw, root / "rejected-slow")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, _, _ = _make_raw_physical_evidence(root)
            invariant_path = raw / CUDA_RUNNER_INVARIANT_NAME
            invariant = json.loads(invariant_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (raw / "software-evidence" / "qualification-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            invariant["checked_at_utc"] = manifest["runs"][1]["started_at_utc"]
            _write_json(invariant_path, invariant)
            with self.assertRaisesRegex(AttachmentError, "immediately before GPU runs"):
                _seal_fixture(raw, root / "rejected-timeline")

    def test_benchmark_repeat_and_live_workload_mutations_fail_closed(self) -> None:
        mutations = (
            (
                "bad-repeat-p95",
                "benchmark-release-default.json",
                lambda report: report["models"][0]["repeats"][1]["timing_ms"][
                    "inference"
                ].update({"p95": 40.0, "p99": 41.0, "max": 42.0}),
                "repeat 2 p95 inference latency",
            ),
            (
                "wrong-benchmark-generator",
                "benchmark-release-default.json",
                lambda report: report["input"].__setitem__("generator", "untrusted"),
                "benchmark input",
            ),
            (
                "wrong-preview-fps",
                "live-release-default-preview-15.json",
                lambda report: report["preview"].__setitem__("fps_limit", 1.0),
                "preview at 15 FPS",
            ),
            (
                "no-preview-activity",
                "live-release-default-preview-15.json",
                lambda report: report["preview"]["stats"].__setitem__(
                    "displayed_frames", 0
                ),
                "preview activity",
            ),
            (
                "unexpected-crop",
                "live-release-default-no-preview.json",
                lambda report: report["config"]["inference"].__setitem__(
                    "crop_size", 640
                ),
                "full-frame workload",
            ),
            (
                "impossible-mean-timing",
                "live-release-default-no-preview.json",
                lambda report: report["pipeline"]["timings"]["mean"].__setitem__(
                    "freshness_latency_ms", 20.0
                ),
                "freshness timing",
            ),
        )
        for name, filename, mutate, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                raw, _, _ = _make_raw_physical_evidence(root)
                report_path = raw / "software-evidence" / filename
                report = json.loads(report_path.read_text(encoding="utf-8"))
                mutate(report)
                _write_json(report_path, report)
                _refresh_software_manifest(raw)
                with self.assertRaisesRegex(AttachmentError, message):
                    _seal_fixture(raw, root / f"rejected-{name}")

    def test_release_default_report_shape_model_and_labels_are_exactly_bound(self) -> None:
        mutations = (
            (
                "benchmark-shape",
                "benchmark-release-default.json",
                lambda report: report["models"][0].__setitem__(
                    "input_shape_hw", [640, 640]
                ),
                "release-default key and shape",
            ),
            (
                "benchmark-labels",
                "benchmark-release-default.json",
                lambda report: report["models"][0]["labels_artifact"]["files"][0].__setitem__(
                    "sha256", "0" * 64
                ),
                "release-default labels",
            ),
            (
                "live-model",
                "live-release-default-no-preview.json",
                lambda report: report["model_artifact"].__setitem__(
                    "sha256", "0" * 64
                ),
                "different model",
            ),
            (
                "live-labels",
                "live-release-default-preview-15.json",
                lambda report: report["labels_artifact"].__setitem__(
                    "sha256", "0" * 64
                ),
                "different labels",
            ),
        )
        for name, filename, mutate, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                raw, _, _ = _make_raw_physical_evidence(root)
                report_path = raw / "software-evidence" / filename
                report = json.loads(report_path.read_text(encoding="utf-8"))
                mutate(report)
                _write_json(report_path, report)
                _refresh_software_manifest(raw)
                with self.assertRaisesRegex(AttachmentError, message):
                    _seal_fixture(raw, root / f"rejected-{name}")



class NvidiaRedistributionManifestTests(unittest.TestCase):
    def test_manifest_binds_every_distribution_legal_file_notice_and_dll(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "ProAim"
            bundle.mkdir()
            _write_fake_cuda_bundle(bundle)
            payload = validate_manifest(bundle)
            self.assertEqual(
                [record["name"] for record in payload["distributions"]],
                list(NVIDIA_DISTRIBUTIONS),
            )
            self.assertEqual(
                {family for record in payload["native_libraries"] for family in record["families"]},
                set(LIBRARY_FAMILIES),
            )
            self.assertEqual(
                payload["third_party_notices"]["required_inventory_markers"],
                list(NOTICE_MARKERS),
            )

    def test_manifest_rejects_changed_legal_payload_or_incomplete_notice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "ProAim"
            bundle.mkdir()
            _write_fake_cuda_bundle(bundle)
            license_path = next(bundle.rglob("License.txt"))
            license_path.write_text("changed terms\n", encoding="utf-8")
            with self.assertRaisesRegex(NvidiaManifestError, "does not exactly match"):
                validate_manifest(bundle)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "ProAim"
            bundle.mkdir()
            _write_fake_cuda_bundle(bundle)
            (bundle / "THIRD_PARTY_NOTICES.md").write_text("CUDA only\n", encoding="utf-8")
            with self.assertRaisesRegex(NvidiaManifestError, "omits NVIDIA inventory"):
                validate_manifest(bundle)

    def test_manifest_rejects_ambiguous_or_nonstandard_json(self) -> None:
        for ambiguous_member in (
            '"schema_version": 1, "schema_version": 2',
            '"schema_version": NaN',
            '"schema_version": Infinity',
        ):
            with self.subTest(ambiguous_member=ambiguous_member), tempfile.TemporaryDirectory() as temporary:
                bundle = Path(temporary) / "ProAim"
                bundle.mkdir()
                _write_fake_cuda_bundle(bundle)
                (bundle / MANIFEST_NAME).write_text(
                    "{" + ambiguous_member + "}\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(NvidiaManifestError, "invalid or missing"):
                    validate_manifest(bundle)


class CandidateArchiveTests(unittest.TestCase):
    def test_candidate_archive_binds_build_and_redistribution_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            _write_fake_cuda_bundle(bundle)
            candidate = root / CUDA_ARCHIVE_NAME
            _zip_bundle(bundle, candidate)
            from scripts.manage_cuda_release_attachment import sha256_file

            record = validate_and_extract_candidate(
                candidate,
                expected_sha256=sha256_file(candidate),
                expected_commit=COMMIT,
                extract_directory=root / "extracted",
            )
            self.assertEqual(record["build_info"]["runtime_variant"], "cuda")
            self.assertEqual(
                record["release_default_model"]["input_shape_hw"], [384, 640]
            )
            self.assertEqual(
                record["release_default_model"]["model_path"],
                "_internal/models/release_default/player.onnx",
            )
            self.assertEqual(
                set(record["nvidia_distribution_versions"]),
                set(NVIDIA_DISTRIBUTIONS),
            )
            self.assertEqual(len(record["nvidia_redistribution_manifest_sha256"]), 64)

    def test_dirty_build_and_path_traversal_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            _write_fake_cuda_bundle(bundle, dirty=True)
            candidate = root / CUDA_ARCHIVE_NAME
            _zip_bundle(bundle, candidate)
            from scripts.manage_cuda_release_attachment import sha256_file

            with self.assertRaisesRegex(AttachmentError, "clean Git worktree"):
                validate_and_extract_candidate(
                    candidate,
                    expected_sha256=sha256_file(candidate),
                    expected_commit=COMMIT,
                    extract_directory=root / "extract-dirty",
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / CUDA_ARCHIVE_NAME
            with zipfile.ZipFile(candidate, "w") as output:
                output.writestr("../escape", b"bad")
            from scripts.manage_cuda_release_attachment import sha256_file

            with self.assertRaisesRegex(AttachmentError, "unsafe path"):
                validate_and_extract_candidate(
                    candidate,
                    expected_sha256=sha256_file(candidate),
                    expected_commit=COMMIT,
                    extract_directory=root / "extract-traversal",
                )

    def test_release_default_contract_rejects_unsafe_shape_and_hash_mutations(self) -> None:
        mutations = (
            ("model_path", "../outside.onnx", "safe bundle-relative path"),
            ("input_shape_hw", [384, 641], "input shape"),
            ("model_sha256", "0" * 64, "hash differs"),
            ("labels_sha256", "0" * 64, "hash differs"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle = root / "bundle"
                bundle.mkdir()
                _write_fake_cuda_bundle(bundle)
                build_info_path = bundle / "BUILD-INFO.json"
                build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
                build_info["release_default_model"][key] = value
                _write_json(build_info_path, build_info)
                candidate = root / CUDA_ARCHIVE_NAME
                _zip_bundle(bundle, candidate)
                from scripts.manage_cuda_release_attachment import sha256_file

                with self.assertRaisesRegex(AttachmentError, message):
                    validate_and_extract_candidate(
                        candidate,
                        expected_sha256=sha256_file(candidate),
                        expected_commit=COMMIT,
                        extract_directory=root / "extract-mutated",
                    )

    def test_dependency_manifest_rejects_weakened_installed_file_evidence(self) -> None:
        mutations = (
            (
                "missing-installed-files",
                lambda distribution, build_info: distribution.pop("installed_files"),
                "exact installed_files",
            ),
            (
                "boolean-entry-count",
                lambda distribution, build_info: distribution["installed_files"].__setitem__(
                    "record_entry_count", True
                ),
                "RECORD counts",
            ),
            (
                "wrong-verified-count",
                lambda distribution, build_info: distribution["installed_files"].__setitem__(
                    "record_sha256_entries_verified", 3
                ),
                "RECORD counts",
            ),
            (
                "boolean-unhashed-count",
                lambda distribution, build_info: distribution["installed_files"].__setitem__(
                    "unhashed_record_entries", False
                ),
                "RECORD counts",
            ),
            (
                "empty-total-size",
                lambda distribution, build_info: distribution["installed_files"].__setitem__(
                    "total_size_bytes", 0
                ),
                "RECORD counts",
            ),
            (
                "record-digest-disagreement",
                lambda distribution, build_info: distribution["installed_files"].__setitem__(
                    "record_document_sha256", "c" * 64
                ),
                "RECORD digests disagree",
            ),
            (
                "uppercase-aggregate",
                lambda distribution, build_info: distribution["installed_files"].__setitem__(
                    "aggregate_sha256", "A" * 64
                ),
                "invalid installed-file SHA-256",
            ),
            (
                "boolean-distribution-count",
                lambda distribution, build_info: build_info["dependency_manifest"].__setitem__(
                    "distribution_count", True
                ),
                "distribution count",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle = root / "bundle"
                bundle.mkdir()
                _write_fake_cuda_bundle(bundle)
                dependency_path = bundle / "DEPENDENCY-MANIFEST.json"
                build_info_path = bundle / "BUILD-INFO.json"
                dependency_manifest = json.loads(
                    dependency_path.read_text(encoding="utf-8")
                )
                build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
                mutate(dependency_manifest["distributions"][0], build_info)
                _write_json(dependency_path, dependency_manifest)
                from scripts.manage_cuda_release_attachment import sha256_file

                build_info["dependency_manifest"]["sha256"] = sha256_file(dependency_path)
                _write_json(build_info_path, build_info)
                candidate = root / CUDA_ARCHIVE_NAME
                _zip_bundle(bundle, candidate)
                with self.assertRaisesRegex(AttachmentError, message):
                    validate_and_extract_candidate(
                        candidate,
                        expected_sha256=sha256_file(candidate),
                        expected_commit=COMMIT,
                        extract_directory=root / "extract-mutated",
                    )


class HostedCpuSmokeTests(unittest.TestCase):
    def _records(self, root: Path, *, requested_provider: str = "CPUExecutionProvider"):
        from scripts.manage_cuda_release_attachment import sha256_file

        stage = root / "stage"
        stage.mkdir()
        model_hash = "f" * 64
        labels_hash = "e" * 64
        release_default = {
            "preset": "fort_player_balanced",
            "model_path": "_internal/models/default.onnx",
            "labels_path": "_internal/models/player.txt",
            "input_shape_hw": [384, 640],
            "detail_crop_size_source_pixels": 0,
            "model_sha256": model_hash,
            "labels_sha256": labels_hash,
        }
        (stage / "ATTACHMENT-ATTESTATION.json").write_text(
            json.dumps(
                {
                    "candidate": {"release_default_model": release_default},
                    "verification_workflow": {"hosted_smoke": None},
                }
            ),
            encoding="utf-8",
        )
        runtime = root / "runtime.json"
        runtime.write_text(
            json.dumps(
                {
                    "frozen": True,
                    "onnxruntime_providers": [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                }
            ),
            encoding="utf-8",
        )
        benchmark = root / "benchmark.json"
        benchmark.write_text(
            json.dumps(
                {
                    "methodology": {
                        "backend": "onnxruntime",
                        "requested_device": "CPU",
                        # CPU is the hosted archive smoke, so accelerator-only
                        # full-provider mode is intentionally disabled.
                        "require_full_provider": False,
                    },
                    "models": [
                        {
                            "key": "release-default",
                            "input_shape_hw": [384, 640],
                            "detail_crop_size_source_pixels": 0,
                            "runtime": {
                                "requested_provider": requested_provider,
                                "active_providers": ["CPUExecutionProvider"],
                            },
                            "artifact": {
                                "files": [{"sha256": model_hash}],
                            },
                            "labels_artifact": {
                                "files": [{"sha256": labels_hash}],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return stage, runtime, benchmark, sha256_file

    def test_cpu_smoke_accepts_cpu_without_accelerator_full_provider_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage, runtime, benchmark, sha256_file = self._records(root)
            result = validate_hosted_smoke(
                stage_directory=stage,
                runtime_info_path=runtime,
                benchmark_path=benchmark,
            )
            smoke = result["verification_workflow"]["hosted_smoke"]
            self.assertEqual(smoke["runtime_info_sha256"], sha256_file(runtime))
            self.assertIn("no GPU claim", smoke["scope"])

    def test_cpu_smoke_rejects_accelerator_requested_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage, runtime, benchmark, _ = self._records(
                root,
                requested_provider="CUDAExecutionProvider",
            )
            with self.assertRaisesRegex(AttachmentError, "did not request CPU"):
                validate_hosted_smoke(
                    stage_directory=stage,
                    runtime_info_path=runtime,
                    benchmark_path=benchmark,
                )


class ChecksumManifestTests(unittest.TestCase):
    def test_checksum_rebuild_preserves_existing_entries_and_adds_cuda(self) -> None:
        existing = {
            "ProAim-Linux-x64.zip": "a" * 64,
            "ProAim-Windows-x64-DirectML.zip": "b" * 64,
        }
        payload = render_checksum_manifest(existing)
        self.assertEqual(parse_checksum_manifest(payload), existing)
        updated = dict(existing)
        updated[CUDA_ARCHIVE_NAME] = "c" * 64
        reparsed = parse_checksum_manifest(render_checksum_manifest(updated))
        self.assertEqual(reparsed, updated)

    def test_duplicate_stale_format_and_paths_are_rejected(self) -> None:
        duplicate = (
            ("a" * 64) + "  A.zip\n" + ("b" * 64) + "  a.zip\n"
        ).encode()
        with self.assertRaisesRegex(AttachmentError, "duplicate"):
            parse_checksum_manifest(duplicate)
        with self.assertRaisesRegex(AttachmentError, "invalid line"):
            parse_checksum_manifest((("a" * 64) + " ../bad.zip\n").encode())
        with self.assertRaisesRegex(AttachmentError, "unsafe checksum filename"):
            render_checksum_manifest({"nested/bad.zip": "a" * 64})


def _write_verified_stage(root: Path, api: FakeApi, inputs) -> Path:
    from scripts.manage_cuda_release_attachment import sha256_bytes, sha256_file

    stage = root / "verified-attachment"
    stage.mkdir()
    candidate = b"already-read-only-verified-cuda-zip"
    (stage / CUDA_ARCHIVE_NAME).write_bytes(candidate)
    evidence = b"verified-physical-evidence"
    (stage / QUALIFICATION_EVIDENCE_ARCHIVE_NAME).write_bytes(evidence)
    runtime = b'{"frozen": true}\n'
    benchmark = b'{"models": [{}]}\n'
    (stage / "hosted-runtime-info.json").write_bytes(runtime)
    (stage / "hosted-cpu-model-smoke.json").write_bytes(benchmark)
    remote = verify_remote_contract(api, inputs)
    telemetry = {
        "driver_version": "600.00",
        "compute_capability": "12.0",
        "gpu_uuid": "GPU-private",
    }
    qualification_metrics = {
        "policy": {"fixture": True},
        "benchmark": {"p95_inference_ms": 10.0},
        "live": {},
    }
    receipt = {
        "schema_version": 1,
        "status": "physically_qualified_cuda_release_candidate",
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": remote["tag_commit"],
        "candidate": {"filename": CUDA_ARCHIVE_NAME, "sha256": sha256_bytes(candidate)},
        "evidence_hashes": {
            "archive_sha256": sha256_bytes(evidence),
            "qualification_manifest_sha256": inputs.qualification_manifest_sha256,
            "physical_attestation_sha256": inputs.physical_attestation_sha256,
        },
        "physical_gpu": {
            "product_name": inputs.qualified_gpu,
            "driver_version": telemetry["driver_version"],
            "compute_capability": telemetry["compute_capability"],
        },
        "qualification_metrics": qualification_metrics,
        "qualification_run": {
            "id": remote["qualification_run"]["id"],
            "html_url": remote["qualification_run"]["html_url"],
        },
        "measurement_limits": ["fixture"],
        "privacy": {"redacted": True},
    }
    _write_json(stage / CUDA_QUALIFICATION_RECEIPT_NAME, receipt)
    attestation = {
        "schema_version": 2,
        "repository": inputs.repository,
        "tag": inputs.tag,
        "tag_commit": remote["tag_commit"],
        "existing_release_id": remote["release_id"],
        "source_build": remote["build_run"],
        "source_artifact": remote["build_artifact"],
        "candidate": {"sha256": sha256_bytes(candidate)},
        "physical_qualification": {
            "qualified_gpu": inputs.qualified_gpu,
            "evidence_archive": {
                "sha256": sha256_bytes(evidence),
            },
            "qualification_manifest": {
                "sha256": inputs.qualification_manifest_sha256,
            },
            "physical_attestation": {
                "sha256": inputs.physical_attestation_sha256,
                "observer": {"name": "Private Observer"},
            },
            "qualification_run": remote["qualification_run"],
            "qualification_artifact": remote["qualification_artifact"],
            "release_typed_confirmation": inputs.confirmation,
            "nvidia_redistribution_typed_confirmation": (
                inputs.nvidia_redistribution_confirmation
            ),
            "nvidia_telemetry_summary": telemetry,
            "qualification_metrics": qualification_metrics,
        },
        "verification_workflow": {
            "hosted_smoke": {
                "runtime_info_sha256": sha256_bytes(runtime),
                "cpu_model_smoke_sha256": sha256_bytes(benchmark),
            }
        },
        "public_qualification_receipt": {
            "filename": CUDA_QUALIFICATION_RECEIPT_NAME,
            "sha256": sha256_file(stage / CUDA_QUALIFICATION_RECEIPT_NAME),
        },
    }
    (stage / "ATTACHMENT-ATTESTATION.json").write_text(
        json.dumps(attestation), encoding="utf-8"
    )
    write_content_manifest(
        root=stage,
        output=stage / STAGED_CONTENT_MANIFEST_NAME,
        kind="proaim-cuda-verified-stage",
        context={
            "repository": inputs.repository,
            "tag": inputs.tag,
            "tag_commit": remote["tag_commit"],
            "source_build_run_id": inputs.build_run_id,
            "verification_run_id": 999,
        },
    )
    return stage


def _publish_transaction(api: FakeApi, inputs, stage: Path):
    from scripts.manage_cuda_release_attachment import sha256_file

    attestation = json.loads(
        (stage / "ATTACHMENT-ATTESTATION.json").read_text(encoding="utf-8")
    )
    with patch(
        "scripts.manage_cuda_release_attachment.validate_and_extract_candidate",
        return_value=attestation["candidate"],
    ), patch(
        "scripts.manage_cuda_release_attachment.validate_physical_evidence",
        return_value=attestation["physical_qualification"],
    ):
        return publish_attachment(
            api,
            inputs,
            stage_directory=stage,
            staged_content_manifest_sha256=sha256_file(
                stage / STAGED_CONTENT_MANIFEST_NAME
            ),
            verification_run_id=999,
        )


class PublicationTransactionTests(unittest.TestCase):
    def test_publish_preserves_existing_assets_and_rebuilds_checksum_manifest(self) -> None:
        from scripts.manage_cuda_release_attachment import sha256_bytes

        with tempfile.TemporaryDirectory() as temporary:
            api = FakeReleaseApi()
            candidate_hash = sha256_bytes(b"already-read-only-verified-cuda-zip")
            inputs = _inputs(
                cuda_zip_sha256=candidate_hash,
                qualification_evidence_sha256=sha256_bytes(b"verified-physical-evidence"),
            )
            stage = _write_verified_stage(Path(temporary), api, inputs)
            result = _publish_transaction(api, inputs, stage)
            self.assertEqual(result["status"], "published_and_verified")
            assets = {asset["name"]: asset for asset in api.release_assets.values()}
            self.assertEqual(
                set(assets),
                {
                    "ProAim-Linux-x64.zip",
                    "ProAim-Windows-x64-DirectML.zip",
                    CUDA_ARCHIVE_NAME,
                    CUDA_QUALIFICATION_RECEIPT_NAME,
                    "SHA256SUMS.txt",
                },
            )
            updated = parse_checksum_manifest(
                api.release_payloads[assets["SHA256SUMS.txt"]["id"]]
            )
            self.assertEqual(set(updated), set(assets).difference({"SHA256SUMS.txt"}))
            self.assertEqual(updated[CUDA_ARCHIVE_NAME], candidate_hash)

    def test_failed_checksum_replacement_restores_manifest_and_removes_cuda(self) -> None:
        from scripts.manage_cuda_release_attachment import sha256_bytes

        with tempfile.TemporaryDirectory() as temporary:
            api = FakeReleaseApi(fail_first_checksum_upload=True)
            original_manifest = api.release_payloads[103]
            inputs = _inputs(
                cuda_zip_sha256=sha256_bytes(b"already-read-only-verified-cuda-zip"),
                qualification_evidence_sha256=sha256_bytes(b"verified-physical-evidence"),
            )
            stage = _write_verified_stage(Path(temporary), api, inputs)
            with self.assertRaisesRegex(AttachmentError, "rollback was attempted"):
                _publish_transaction(api, inputs, stage)
            assets = {asset["name"]: asset for asset in api.release_assets.values()}
            self.assertNotIn(CUDA_ARCHIVE_NAME, assets)
            self.assertNotIn(CUDA_QUALIFICATION_RECEIPT_NAME, assets)
            self.assertEqual(
                api.release_payloads[assets["SHA256SUMS.txt"]["id"]],
                original_manifest,
            )

    def test_ambiguous_upload_success_is_reconciled_by_exact_payload(self) -> None:
        from scripts.manage_cuda_release_attachment import sha256_bytes

        for ambiguous_name in (
            CUDA_ARCHIVE_NAME,
            CUDA_QUALIFICATION_RECEIPT_NAME,
            "SHA256SUMS.txt",
        ):
            with self.subTest(ambiguous_name=ambiguous_name), tempfile.TemporaryDirectory() as temporary:
                api = FakeReleaseApi(commit_then_raise_name=ambiguous_name)
                original_manifest = api.release_payloads[103]
                inputs = _inputs(
                    cuda_zip_sha256=sha256_bytes(
                        b"already-read-only-verified-cuda-zip"
                    ),
                    qualification_evidence_sha256=sha256_bytes(b"verified-physical-evidence"),
                )
                stage = _write_verified_stage(Path(temporary), api, inputs)
                with self.assertRaisesRegex(AttachmentError, "rollback was attempted"):
                    _publish_transaction(api, inputs, stage)
                assets = {
                    asset["name"]: asset for asset in api.release_assets.values()
                }
                self.assertNotIn(CUDA_ARCHIVE_NAME, assets)
                self.assertNotIn(CUDA_QUALIFICATION_RECEIPT_NAME, assets)
                self.assertEqual(
                    api.release_payloads[assets["SHA256SUMS.txt"]["id"]],
                    original_manifest,
                )

    def test_staged_manifest_mutation_is_rejected_before_release_mutation(self) -> None:
        from scripts.manage_cuda_release_attachment import sha256_bytes

        with tempfile.TemporaryDirectory() as temporary:
            api = FakeReleaseApi()
            inputs = _inputs(
                cuda_zip_sha256=sha256_bytes(b"already-read-only-verified-cuda-zip"),
                qualification_evidence_sha256=sha256_bytes(b"verified-physical-evidence"),
            )
            stage = _write_verified_stage(Path(temporary), api, inputs)
            (stage / "hosted-runtime-info.json").write_text(
                '{"frozen":false}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(AttachmentError, "content .* mismatch"):
                _publish_transaction(api, inputs, stage)
            self.assertNotIn(
                CUDA_ARCHIVE_NAME,
                {asset["name"] for asset in api.release_assets.values()},
            )

    def test_tag_move_after_upload_rolls_back_cuda_receipt_and_checksum(self) -> None:
        from scripts.manage_cuda_release_attachment import sha256_bytes

        with tempfile.TemporaryDirectory() as temporary:
            api = MovingTagReleaseApi()
            original_manifest = api.release_payloads[103]
            inputs = _inputs(
                cuda_zip_sha256=sha256_bytes(b"already-read-only-verified-cuda-zip"),
                qualification_evidence_sha256=sha256_bytes(b"verified-physical-evidence"),
            )
            stage = _write_verified_stage(Path(temporary), api, inputs)
            with self.assertRaisesRegex(AttachmentError, "tag moved during publication"):
                _publish_transaction(api, inputs, stage)
            assets = {asset["name"]: asset for asset in api.release_assets.values()}
            self.assertNotIn(CUDA_ARCHIVE_NAME, assets)
            self.assertNotIn(CUDA_QUALIFICATION_RECEIPT_NAME, assets)
            self.assertEqual(
                api.release_payloads[assets["SHA256SUMS.txt"]["id"]],
                original_manifest,
            )


if __name__ == "__main__":
    unittest.main()
