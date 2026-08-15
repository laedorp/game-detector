from __future__ import annotations

from datetime import datetime, timedelta, timezone
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.manage_cuda_release_attachment import AttachmentError, LIVE_TIMING_FIELDS, sha256_file
from scripts.manage_directml_release import (
    CANDIDATE_MANIFEST_NAME,
    CHECKSUM_NAME,
    DIRECTML_ARCHIVE_NAME,
    DIRECTML_POLICY,
    GPU_PRODUCTS,
    HOLDOUT_ATTESTATION_ARTIFACT_NAME,
    HOLDOUT_ATTESTATION_NAME,
    HOLDOUT_EVIDENCE_ARTIFACT_NAME,
    HOLDOUT_PLAN_ARTIFACT_NAME,
    HOLDOUT_PREREQUISITE_ARTIFACT_NAME,
    LINUX_ARCHIVE_NAME,
    LOCAL_OBSERVATION_NAME,
    PHYSICAL_ATTESTATION_NAME,
    PUBLIC_HOLDOUT_RECEIPT_NAME,
    PUBLIC_RELEASE_RECEIPT_NAME,
    QUALIFICATION_MANIFEST_NAME,
    RAW_CONTENT_KIND,
    RAW_CONTENT_MANIFEST_NAME,
    RUNNER_INVARIANT_NAME,
    SOFTWARE_EVIDENCE_FILES,
    SOURCE_RECORD_NAME,
    TELEMETRY_NAME,
    EvidenceInput,
    HoldoutInput,
    ReleaseInputs,
    _parse_directml_telemetry,
    _holdout_attestation_content_hash,
    _holdout_release_upload_names,
    _rollback_marker_releases,
    _validate_release_stage,
    expected_physical_confirmation,
    create_authenticated_holdout_attestation,
    inspect_candidate,
    prepare_release_stage,
    qualification_archive_name,
    raw_content_context,
    seal_evidence,
    stage_candidate,
    validate_sealed_evidence,
    validate_authenticated_holdout_evidence,
    verify_release_remote_contract,
)
from tests.independent_holdout_hardware_fixture import (
    valid_rx6950_holdout_hardware_identity,
)
from utils.independent_holdout_release_contract import (
    BUNDLE_MANIFEST_NAME as HOLDOUT_BUNDLE_MANIFEST_NAME,
    BUNDLE_MEMBER_NAMES as HOLDOUT_BUNDLE_MEMBER_NAMES,
    canonical_json_bytes as holdout_canonical_json_bytes,
)
from scripts.manage_cuda_release_attachment import write_content_manifest


COMMIT = "a" * 40
REPOSITORY = "owner/proaim"
TAG = "v1.2.3"
SOURCE_RUN_ID = 42
AMD_RUN_ID = 84
NVIDIA_RUN_ID = 85
HOLDOUT_RUN_ID = 86


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stamp(base: datetime, seconds: int) -> str:
    return (base + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _distribution() -> dict[str, object]:
    return {
        "canonical_name": "onnxruntime-directml",
        "installed_files": {
            "aggregate_sha256": "b" * 64,
            "record_document_sha256": "c" * 64,
            "record_entry_count": 3,
            "record_sha256_entries_verified": 2,
            "total_size_bytes": 123,
            "unhashed_record_entries": 1,
        },
        "installed_record_sha256": "c" * 64,
        "version": "1.24.4",
    }


def _write_directml_bundle(bundle: Path, *, detail_width: int = 768) -> None:
    model = bundle / "_internal" / "models" / "release_default" / "player.onnx"
    labels = bundle / "_internal" / "models" / "release_labels.txt"
    model.parent.mkdir(parents=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"fake-directml-onnx")
    labels.write_text("player\n", encoding="utf-8")
    for name in ("ProAimCLI.exe", "Qualify-ProAimGpu.ps1", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        (bundle / name).write_text(name + "\n", encoding="utf-8")
    dependency = {
        "application": "ProAim",
        "artifact_hash_contract": {"enforced_before_install": True},
        "distributions": [_distribution()],
        "lock_profile": "windows-directml-py313",
        "runtime_variant": "directml",
        "schema_version": 1,
    }
    dependency_path = bundle / "DEPENDENCY-MANIFEST.json"
    _write_json(dependency_path, dependency)
    build_info = {
        "application": "ProAim",
        "commit": COMMIT,
        "dependency_manifest": {
            "distribution_count": 1,
            "lock_profile": "windows-directml-py313",
            "path": "DEPENDENCY-MANIFEST.json",
            "schema_version": 1,
            "sha256": sha256_file(dependency_path),
        },
        "dirty": False,
        "release_default_model": {
            "preset": "release_player_rectangular",
            "model_path": "_internal/models/release_default/player.onnx",
            "labels_path": "_internal/models/release_labels.txt",
            "input_shape_hw": [384, 640],
            "detail_crop_size_source_pixels": detail_width,
            "model_sha256": sha256_file(model),
            "labels_sha256": sha256_file(labels),
        },
        "runtime_variant": "directml",
        "schema": 2,
    }
    _write_json(bundle / "BUILD-INFO.json", build_info)


def _zip_bundle(bundle: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                output.write(path, "ProAim/" + path.relative_to(bundle).as_posix())


def _make_candidate(root: Path) -> tuple[Path, Path, dict[str, object], str, str]:
    bundle = root / "bundle"
    bundle.mkdir()
    _write_directml_bundle(bundle)
    candidate_directory = root / "candidate"
    candidate_directory.mkdir()
    directml_zip = candidate_directory / DIRECTML_ARCHIVE_NAME
    _zip_bundle(bundle, directml_zip)
    with zipfile.ZipFile(candidate_directory / LINUX_ARCHIVE_NAME, "w") as archive:
        archive.writestr("ProAim/ProAim", b"linux")
    staged = stage_candidate(
        candidate_directory,
        repository=REPOSITORY,
        tag=TAG,
        tag_commit=COMMIT,
        source_run_id=SOURCE_RUN_ID,
    )
    candidate_manifest_sha = str(staged["manifest_sha256"])
    directml_sha = sha256_file(directml_zip)
    candidate = inspect_candidate(
        candidate_directory,
        repository=REPOSITORY,
        tag=TAG,
        tag_commit=COMMIT,
        source_run_id=SOURCE_RUN_ID,
        candidate_manifest_sha256=candidate_manifest_sha,
        directml_zip_sha256=directml_sha,
        extraction_root=root / "extracted-candidate",
    )
    return candidate_directory, bundle, candidate, candidate_manifest_sha, directml_sha


def _timing(samples: int, mean: float, minimum: float, maximum: float, p95: float) -> dict[str, float | int]:
    return {
        "samples": samples,
        "mean": mean,
        "p50": mean,
        "median": mean,
        "p95": p95,
        "p99": max(p95, maximum),
        "min": minimum,
        "max": max(p95, maximum),
        "stdev": 0.5,
    }


def _provider(adapter_index: int) -> dict[str, object]:
    device = f"DIRECTML:{adapter_index}"
    return {
        "requested_provider": "DmlExecutionProvider",
        "requested_device_input": device,
        "active_providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "require_full_provider": True,
        "configured_session_options": {"disable_cpu_ep_fallback": True},
        "runtime_ep_fail_fallback_disabled": True,
        "provider_options_status": "ok",
        "provider_option_overrides": {"DmlExecutionProvider": {"device_id": str(adapter_index)}},
    }


def _detail_plan() -> dict[str, object]:
    width, height = 765, 459
    full_scale = min(640.0 / 1920.0, 384.0 / 1080.0)
    detail_scale = min(640.0 / width, 384.0 / height)
    return {
        "crop_policy": "centered_model_aspect_roi",
        "requested_crop_size": 768,
        "requested_crop_height": 461,
        "applied_crop_width": width,
        "applied_crop_height": height,
        "source_width": 1920,
        "source_height": 1080,
        "model_width": 640,
        "model_height": 384,
        "crop_x": (1920 - width) // 2,
        "crop_y": (1080 - height) // 2,
        "coverage_fraction": width * height / float(1920 * 1080),
        "full_frame_scale": full_scale,
        "detail_scale": detail_scale,
        "effective_linear_magnification": detail_scale / full_scale,
        "clamped": True,
        "redundant": False,
    }


def _live_timings(preview: bool) -> dict[str, object]:
    values = {
        "capture_ms": 1.0,
        "queue_age_ms": 2.0,
        "preprocess_ms": 2.0,
        "inference_ms": 8.0,
        "postprocess_ms": 1.0,
        "detail_preprocess_ms": 1.0,
        "detail_inference_ms": 8.0,
        "detail_postprocess_ms": 1.0,
        "control_ms": 1.0,
        "processing_ms": 22.0,
        "freshness_latency_ms": 24.0,
        "observed_pipeline_ms": 25.0,
        "draw_ms": 1.0 if preview else 0.0,
        "preview_service_ms": 0.5 if preview else 0.0,
    }
    return {
        "unit": "milliseconds",
        "fields": list(LIVE_TIMING_FIELDS),
        **{name: dict(values) for name in ("mean", "p50", "p95", "p99")},
    }


def _refresh_software_manifest(software: Path) -> None:
    manifest_path = software / QUALIFICATION_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["evidence_files"]:
        path = software / record["file"]
        record["sha256"] = sha256_file(path)
        record["size_bytes"] = path.stat().st_size
    for run in manifest["runs"]:
        for key in ("stdout", "stderr", "metrics"):
            record = run.get(key)
            if record is not None:
                path = software / record["file"]
                record["sha256"] = sha256_file(path)
                record["size_bytes"] = path.stat().st_size
    _write_json(manifest_path, manifest)


def _make_raw_evidence(
    root: Path,
    *,
    bundle: Path,
    candidate: dict[str, object],
    candidate_manifest_sha: str,
    directml_sha: str,
    role: str = "amd_rx_6950_xt",
    adapter_index: int = 0,
) -> tuple[Path, dict[str, object], datetime]:
    run_id = AMD_RUN_ID if role == "amd_rx_6950_xt" else NVIDIA_RUN_ID
    raw = root / f"raw-{role}"
    software = raw / "software-evidence"
    software.mkdir(parents=True)
    shutil.copyfile(bundle / "BUILD-INFO.json", software / "bundle-BUILD-INFO.json")
    shutil.copyfile(
        bundle / "DEPENDENCY-MANIFEST.json",
        software / "bundle-DEPENDENCY-MANIFEST.json",
    )
    (software / "TASK-MANAGER-INSTRUCTIONS.txt").write_text("observe GPU\n", encoding="utf-8")
    (software / "TASK-MANAGER-CONFIRMATION.txt").write_text("pending\n", encoding="utf-8")
    _write_json(
        software / "runtime-info.json",
        {"frozen": True, "onnxruntime_providers": ["DmlExecutionProvider", "CPUExecutionProvider"]},
    )
    (software / "runtime-info.stderr.txt").write_text("", encoding="utf-8")
    base = datetime.now(timezone.utc) - timedelta(seconds=130)
    provider = _provider(adapter_index)
    aggregate = {
        "preprocess": _timing(300, 1.0, 0.5, 2.0, 1.5),
        "inference": _timing(300, 10.0, 8.0, 14.0, 12.0),
        "postprocess": _timing(300, 1.0, 0.5, 2.0, 1.5),
        "pipeline": _timing(300, 12.0, 9.0, 18.0, 15.0),
    }
    repeats = [
        {
            "repeat": number,
            "timing_ms": {name: {**summary, "samples": 100} for name, summary in aggregate.items()},
            "detections_mean": 1.0,
        }
        for number in range(1, 4)
    ]
    release_default = candidate["release_default_model"]
    benchmark = {
        "generated_at_utc": _stamp(base, 10),
        "methodology": {
            "backend": "onnxruntime",
            "requested_device": f"DIRECTML:{adapter_index}",
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
                "runtime": provider,
                "artifact": {
                    "files": [
                        {
                            "resolved_path": "C:/runner/ProAim/" + release_default["model_path"],
                            "sha256": release_default["model_sha256"],
                        }
                    ]
                },
                "labels_artifact": {
                    "files": [
                        {
                            "resolved_path": "C:/runner/ProAim/" + release_default["labels_path"],
                            "sha256": release_default["labels_sha256"],
                        }
                    ]
                },
                "timing_ms": aggregate,
                "pipeline_fps_from_mean": 1000.0 / 12.0,
                "repeats": repeats,
            }
        ],
    }
    _write_json(software / "benchmark-release-default.json", benchmark)
    (software / "benchmark-release-default.stderr.txt").write_text("", encoding="utf-8")
    descriptor = {
        "index": adapter_index,
        "name": GPU_PRODUCTS[role]["product_name"],
        "vendor_id": GPU_PRODUCTS[role]["vendor_id"],
        "device_id": "0x73bf" if role == "amd_rx_6950_xt" else "0x2d58",
        "dedicated_vram_bytes": 8 * 1024**3,
    }
    for suffix, preview, start in (("no-preview", False, 22), ("preview-15", True, 72)):
        report = {
            "started_utc": _stamp(base, start),
            "completed_utc": _stamp(base, start + 40),
            "detector_runtime": provider,
            "directml_adapter": {
                "requested_index": adapter_index,
                "configured_index": adapter_index,
                "effective_index": adapter_index,
                "requested_provider_mismatch": False,
                "enumeration_status": "matched_dxgi_adapter",
                "task_manager_confirmation_required": True,
                "descriptor": descriptor,
            },
            "config": {
                "backend": "onnxruntime",
                "device": f"DIRECTML:{adapter_index}",
                "require_full_provider": True,
                "model_path": "C:/runner/ProAim/" + release_default["model_path"],
                "labels_path": "C:/runner/ProAim/" + release_default["labels_path"],
                "source": {"kind": "screen", "value": None},
                "capture": {"screen_region": None, "screen_monitor": 1, "screen_fps": 60.0},
                "inference": {"shape_hw": [384, 640], "crop_size": None, "detail_crop_size": 768},
                "preview": {"enabled": preview, "fps_limit": 15.0},
                "stats_window": 1000,
            },
            "model_artifact": {"sha256": release_default["model_sha256"]},
            "labels_artifact": {"sha256": release_default["labels_sha256"]},
            "source": {"backend": "dxcam-dxgi", "fallback_reason": None},
            "capture": {"read_failures": 0},
            "preview": {
                "enabled": preview,
                "fps_limit": 15.0,
                "mode": "threaded" if preview else "disabled",
                "stats": (
                    {"submitted_frames": 600, "displayed_frames": 590, "replaced_frames": 10}
                    if preview
                    else {}
                ),
            },
            "detail_pass": {
                "enabled": True,
                "crop_policy": "centered_model_aspect_roi",
                "requested_crop_size": 768,
                "frames_seen": 1000,
                "frames_applied": 1000,
                "frames_redundant": 0,
                "frames_clamped": 1000,
                "last_plan": _detail_plan(),
            },
            "pipeline": {
                "processed_frames": 1000,
                "rolling_sample_count": 1000,
                "elapsed_seconds": 40.0,
                "elapsed_fps": 25.0,
                "update_fps": 25.0,
                "timings": _live_timings(preview),
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

    def record(filename: str) -> dict[str, object]:
        path = software / filename
        return {"file": filename, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    runs = [
        {
            "name": "frozen runtime info",
            "arguments": ["--runtime-info"],
            "exit_code": 0,
            "stdout": record("runtime-info.json"),
            "stderr": record("runtime-info.stderr.txt"),
            "started_at_utc": _stamp(base, 0),
            "completed_at_utc": _stamp(base, 2),
        },
        {
            "name": "model benchmark (release-default)",
            "arguments": ["--benchmark-models", "--device", f"DIRECTML:{adapter_index}", "--require-full-provider"],
            "exit_code": 0,
            "stdout": record("benchmark-release-default.json"),
            "stderr": record("benchmark-release-default.stderr.txt"),
            "started_at_utc": _stamp(base, 5),
            "completed_at_utc": _stamp(base, 15),
        },
    ]
    for suffix, start in (("no-preview", 20), ("preview-15", 70)):
        runs.append(
            {
                "name": f"live pipeline (release-default-{suffix})",
                "arguments": ["--cli", "--device", f"DIRECTML:{adapter_index}", "--require-full-provider"],
                "exit_code": 0,
                "stdout": record(f"live-release-default-{suffix}.stdout.txt"),
                "stderr": record(f"live-release-default-{suffix}.stderr.txt"),
                "metrics": record(f"live-release-default-{suffix}.json"),
                "started_at_utc": _stamp(base, start),
                "completed_at_utc": _stamp(base, start + 45),
            }
        )
    artifacts = {
        "frozen_cli": candidate["frozen_cli_sha256"],
        "build_info": candidate["build_info_sha256"],
        "dependency_manifest": candidate["dependency_manifest_sha256"],
        "qualification_helper": candidate["qualification_helper_sha256"],
        "original_bundle_archive": directml_sha,
        "release_default_model": release_default["model_sha256"],
        "release_default_labels": release_default["labels_sha256"],
    }
    evidence_files = [record(name) for name in sorted(SOFTWARE_EVIDENCE_FILES)]
    manifest = {
        "schema_version": 1,
        "status": "software_checks_passed_physical_gpu_confirmation_pending",
        "qualified": False,
        "provider": {
            "selection": "DirectML",
            "requested_device": f"DIRECTML:{adapter_index}",
            "expected_execution_provider": "DmlExecutionProvider",
            "directml_adapter_index": adapter_index,
        },
        "bundle_build_info": candidate["build_info"],
        "benchmark_bounds": {"samples": 32, "warmup": 30, "iterations": 100, "repeats": 3},
        "live_bounds": {
            "enabled": True,
            "selected_model": "release-default",
            "release_default_model": release_default,
            "screen_monitor": 1,
            "screen_fps": 60.0,
            "max_frames": 1000,
            "max_seconds": 60.0,
            "detail_crop_size": 768,
            "modes": ["no-preview", "preview-15"],
        },
        "input_artifacts": [
            {
                "role": name,
                "sha256": digest,
                **(
                    {
                        "path": release_default[
                            "model_path" if name == "release_default_model" else "labels_path"
                        ],
                        "location": "bundle",
                    }
                    if name in {"release_default_model", "release_default_labels"}
                    else {}
                ),
            }
            for name, digest in artifacts.items()
        ],
        "runs": runs,
        "evidence_files": evidence_files,
        "manual_confirmation": {"required": True, "completed_by_helper": False},
    }
    _write_json(software / QUALIFICATION_MANIFEST_NAME, manifest)
    _write_json(raw / "candidate-inspection.json", candidate)
    _write_json(
        raw / SOURCE_RECORD_NAME,
        {
            "tag_commit": COMMIT,
            "release_absent": True,
            "source_build_run": {"id": SOURCE_RUN_ID},
        },
    )
    _write_json(
        raw / RUNNER_INVARIANT_NAME,
        {
            "schema_version": 1,
            "status": "passed_before_directml_runs",
            "checked_at_utc": _stamp(base, -1),
            "gpu_role": role,
            "expected_product_name": GPU_PRODUCTS[role]["product_name"],
            "directml_adapter_index": adapter_index,
            "preexisting_proaim_cli_count": 0,
            "telemetry_interval_milliseconds": 500,
        },
    )
    luid = "0x00000000_0x00001234"
    telemetry: list[dict[str, object]] = [
        {
            "schema_version": 1,
            "kind": "adapter_inventory",
            "captured_at_utc": _stamp(base, 0),
            "gpu_role": role,
            "directml_adapter_index": adapter_index,
            "product_name": GPU_PRODUCTS[role]["product_name"],
            "adapter_luid": luid,
            "vendor_id": GPU_PRODUCTS[role]["vendor_id"],
            "device_id": descriptor["device_id"],
            "driver_version": "32.0.1",
            "exact_product_match_count": 1,
            "telemetry_interval_milliseconds": 500,
        }
    ]
    for pid, offsets in ((101, (7, 9)), (102, (25, 30)), (103, (75, 80))):
        for offset in offsets:
            telemetry.append(
                {
                    "schema_version": 1,
                    "kind": "proaim_gpu_engine",
                    "captured_at_utc": _stamp(base, offset),
                    "pid": pid,
                    "process_name": "ProAimCLI.exe",
                    "adapter_luid": luid,
                    "physical_adapter": 0,
                    "engine_index": 1,
                    "engine_type": "Compute_0",
                    "utilization_percent": 25.0,
                }
            )
    (raw / TELEMETRY_NAME).write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in telemetry),
        encoding="utf-8",
    )
    confirmation = expected_physical_confirmation(TAG, role)
    _write_json(
        raw / LOCAL_OBSERVATION_NAME,
        {
            "schema_version": 1,
            "status": "completed_after_automated_directml_runs",
            "completed": True,
            "repository": REPOSITORY,
            "tag": TAG,
            "github_actor": "gpu-tester",
            "github_run_id": str(run_id),
            "observer_name": "Physical Tester",
            "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "gpu_role": role,
            "physical_gpu_name": GPU_PRODUCTS[role]["product_name"],
            "directml_adapter_index": adapter_index,
            "task_manager_gpu_engine": "GPU 1 - Compute_0",
            "typed_confirmation": confirmation,
            "observations": {
                "release_default_benchmark": True,
                "live_no_preview": True,
                "live_preview_15": True,
                "automated_luid_telemetry_agreed": True,
            },
        },
    )
    context = raw_content_context(
        repository=REPOSITORY,
        tag=TAG,
        tag_commit=COMMIT,
        source_run_id=SOURCE_RUN_ID,
        qualification_run_id=run_id,
        qualification_run_attempt=1,
        role=role,
        adapter_index=adapter_index,
        candidate_manifest_sha256=candidate_manifest_sha,
        directml_zip_sha256=directml_sha,
    )
    write_content_manifest(
        root=raw,
        output=raw / RAW_CONTENT_MANIFEST_NAME,
        kind=RAW_CONTENT_KIND,
        context=context,
    )
    return raw, context, base


def _release_inputs(
    *,
    candidate_manifest_sha: str,
    directml_sha: str,
    amd: EvidenceInput,
    nvidia: EvidenceInput | None = None,
) -> ReleaseInputs:
    if nvidia is None:
        nvidia = EvidenceInput.create(
            role="nvidia_rtx_5060_laptop",
            run_id=NVIDIA_RUN_ID,
            adapter_index=0,
            archive_sha256="1" * 64,
            qualification_manifest_sha256="2" * 64,
            physical_attestation_sha256="3" * 64,
            public_receipt_sha256="4" * 64,
        )
    return ReleaseInputs.create(
        repository=REPOSITORY,
        tag=TAG,
        source_run_id=SOURCE_RUN_ID,
        candidate_manifest_sha256=candidate_manifest_sha,
        directml_zip_sha256=directml_sha,
        confirmation=f"PUBLISH DUAL-GPU QUALIFIED DIRECTML RELEASE {TAG}",
        evidence=(amd, nvidia),
    )


class CandidateContractTests(unittest.TestCase):
    def test_candidate_round_trip_binds_dynamic_rectangular_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, _, candidate, manifest_sha, directml_sha = _make_candidate(root)
            self.assertEqual(candidate["sha256"], directml_sha)
            self.assertEqual(candidate["candidate_manifest_sha256"], manifest_sha)
            self.assertEqual(candidate["release_default_model"]["input_shape_hw"], [384, 640])
            self.assertEqual(candidate["release_default_model"]["detail_crop_size_source_pixels"], 768)
            with (directory / DIRECTML_ARCHIVE_NAME).open("ab") as output:
                output.write(b"tamper")
            with self.assertRaises(AttachmentError):
                inspect_candidate(
                    directory,
                    repository=REPOSITORY,
                    tag=TAG,
                    tag_commit=COMMIT,
                    source_run_id=SOURCE_RUN_ID,
                    candidate_manifest_sha256=manifest_sha,
                    directml_zip_sha256=directml_sha,
                    extraction_root=root / "tampered",
                )

    def test_stage_rejects_any_third_file_or_missing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (LINUX_ARCHIVE_NAME, DIRECTML_ARCHIVE_NAME):
                with zipfile.ZipFile(root / name, "w") as archive:
                    archive.writestr("x", b"x")
            (root / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(AttachmentError, "exactly"):
                stage_candidate(
                    root,
                    repository=REPOSITORY,
                    tag=TAG,
                    tag_commit=COMMIT,
                    source_run_id=SOURCE_RUN_ID,
                )


class InputAndTelemetryTests(unittest.TestCase):
    def test_publication_requires_distinct_amd_and_nvidia_runs_and_exact_phrase(self) -> None:
        amd = EvidenceInput.create(
            role="amd_rx_6950_xt",
            run_id=AMD_RUN_ID,
            adapter_index=0,
            archive_sha256="1" * 64,
            qualification_manifest_sha256="2" * 64,
            physical_attestation_sha256="3" * 64,
            public_receipt_sha256="4" * 64,
        )
        nvidia = EvidenceInput.create(
            role="nvidia_rtx_5060_laptop",
            run_id=NVIDIA_RUN_ID,
            adapter_index=1,
            archive_sha256="5" * 64,
            qualification_manifest_sha256="6" * 64,
            physical_attestation_sha256="7" * 64,
            public_receipt_sha256="8" * 64,
        )
        result = _release_inputs(candidate_manifest_sha="9" * 64, directml_sha="a" * 64, amd=amd, nvidia=nvidia)
        self.assertEqual([record.role for record in result.evidence], list(GPU_PRODUCTS))
        with self.assertRaisesRegex(AttachmentError, "distinct"):
            ReleaseInputs.create(
                repository=REPOSITORY,
                tag=TAG,
                source_run_id=SOURCE_RUN_ID,
                candidate_manifest_sha256="9" * 64,
                directml_zip_sha256="a" * 64,
                confirmation=f"PUBLISH DUAL-GPU QUALIFIED DIRECTML RELEASE {TAG}",
                evidence=(amd, EvidenceInput.create(**{**nvidia.__dict__, "run_id": AMD_RUN_ID})),
            )
        with self.assertRaisesRegex(AttachmentError, "confirmation"):
            ReleaseInputs.create(
                repository=REPOSITORY,
                tag=TAG,
                source_run_id=SOURCE_RUN_ID,
                candidate_manifest_sha256="9" * 64,
                directml_zip_sha256="a" * 64,
                confirmation="publish it",
                evidence=(amd, nvidia),
            )

    def test_telemetry_requires_two_distinct_captures_pid_luid_and_positive_work(self) -> None:
        base = datetime.now(timezone.utc)
        intervals = [
            ("frozen runtime info", base, base + timedelta(seconds=1)),
            ("model benchmark (release-default)", base + timedelta(seconds=2), base + timedelta(seconds=5)),
            ("live pipeline (release-default-no-preview)", base + timedelta(seconds=6), base + timedelta(seconds=9)),
            ("live pipeline (release-default-preview-15)", base + timedelta(seconds=10), base + timedelta(seconds=13)),
        ]
        descriptor = {"device_id": "0x73bf"}
        inventory = {
            "schema_version": 1,
            "kind": "adapter_inventory",
            "captured_at_utc": _stamp(base, 1),
            "gpu_role": "amd_rx_6950_xt",
            "directml_adapter_index": 0,
            "product_name": "AMD Radeon RX 6950 XT",
            "exact_product_match_count": 1,
            "telemetry_interval_milliseconds": 500,
            "vendor_id": "0x1002",
            "device_id": "0x73bf",
            "driver_version": "1.2.3",
            "adapter_luid": "0x00000000_0x00001234",
        }
        records = [inventory]
        for pid, starts in ((101, (3, 4)), (102, (7, 8)), (103, (11, 12))):
            for offset in starts:
                records.append(
                    {
                        "schema_version": 1,
                        "kind": "proaim_gpu_engine",
                        "captured_at_utc": _stamp(base, offset),
                        "pid": pid,
                        "process_name": "ProAimCLI.exe",
                        "adapter_luid": inventory["adapter_luid"],
                        "physical_adapter": 0,
                        "engine_index": 1,
                        "engine_type": "Compute_0",
                        "utilization_percent": 10.0,
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / TELEMETRY_NAME
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            summary = _parse_directml_telemetry(
                path,
                role="amd_rx_6950_xt",
                adapter_index=0,
                descriptor=descriptor,
                intervals=intervals,
            )
            self.assertTrue(summary["pid_correlated"])
            self.assertTrue(all(value["distinct_capture_count"] == 2 for value in summary["per_run"].values()))
            records[2]["captured_at_utc"] = records[1]["captured_at_utc"]
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            with self.assertRaisesRegex(AttachmentError, "positive ProAim GPU work"):
                _parse_directml_telemetry(
                    path,
                    role="amd_rx_6950_xt",
                    adapter_index=0,
                    descriptor=descriptor,
                    intervals=intervals,
                )


class _RemoteApi:
    def __init__(self) -> None:
        self.responses = {
            f"/git/ref/tags/{TAG}": {
                "ref": f"refs/tags/{TAG}",
                "object": {"type": "commit", "sha": COMMIT},
            },
            f"/actions/runs/{SOURCE_RUN_ID}": {
                "event": "push",
                "status": "completed",
                "conclusion": "success",
                "head_sha": COMMIT,
                "repository": {"full_name": REPOSITORY},
                "head_repository": {"full_name": REPOSITORY},
                "workflow_id": 10,
                "path": ".github/workflows/release-bundles.yml",
                "html_url": "https://example.invalid/source",
                "run_attempt": 1,
            },
            "/actions/workflows/10": {"path": ".github/workflows/release-bundles.yml"},
        }
        for run_id, workflow_id in ((AMD_RUN_ID, 11), (NVIDIA_RUN_ID, 12)):
            self.responses[f"/actions/runs/{run_id}"] = {
                "event": "workflow_dispatch",
                "status": "completed",
                "conclusion": "success",
                "head_sha": COMMIT,
                "repository": {"full_name": REPOSITORY},
                "head_repository": {"full_name": REPOSITORY},
                "workflow_id": workflow_id,
                "path": ".github/workflows/qualify-windows-directml.yml",
                "html_url": f"https://example.invalid/{run_id}",
                "run_attempt": 1,
                "actor": {"login": f"tester-{run_id}"},
            }
            self.responses[f"/actions/workflows/{workflow_id}"] = {
                "path": ".github/workflows/qualify-windows-directml.yml"
            }
        self.responses[f"/actions/runs/{HOLDOUT_RUN_ID}"] = {
            "event": "workflow_dispatch",
            "status": "completed",
            "conclusion": "success",
            "head_sha": COMMIT,
            "repository": {"full_name": REPOSITORY},
            "head_repository": {"full_name": REPOSITORY},
            "workflow_id": 13,
            "path": ".github/workflows/qualify-independent-holdout.yml",
            "html_url": "https://example.invalid/holdout",
            "run_attempt": 1,
            "actor": {"login": "holdout-dispatcher"},
        }
        self.responses["/actions/workflows/13"] = {
            "path": ".github/workflows/qualify-independent-holdout.yml"
        }
        self.releases: list[dict[str, object]] = []
        self.artifacts = {
            SOURCE_RUN_ID: [
                {
                    "id": 90,
                    "name": "ProAim-Release-Candidate",
                    "expired": False,
                    "size_in_bytes": 100,
                    "digest": "sha256:" + "1" * 64,
                }
            ],
            AMD_RUN_ID: [
                {
                    "id": 91,
                    "name": "ProAim-Windows-DirectML-amd_rx_6950_xt-Qualification-Evidence",
                    "expired": False,
                    "size_in_bytes": 100,
                    "digest": "sha256:" + "2" * 64,
                }
            ],
            NVIDIA_RUN_ID: [
                {
                    "id": 92,
                    "name": "ProAim-Windows-DirectML-nvidia_rtx_5060_laptop-Qualification-Evidence",
                    "expired": False,
                    "size_in_bytes": 100,
                    "digest": "sha256:" + "3" * 64,
                }
            ],
            HOLDOUT_RUN_ID: [
                {
                    "id": 93 + index,
                    "name": name,
                    "expired": False,
                    "size_in_bytes": 100,
                    "digest": "sha256:" + str(4 + index) * 64,
                }
                for index, name in enumerate(
                    (
                        HOLDOUT_PREREQUISITE_ARTIFACT_NAME,
                        HOLDOUT_PLAN_ARTIFACT_NAME,
                        HOLDOUT_EVIDENCE_ARTIFACT_NAME,
                        HOLDOUT_ATTESTATION_ARTIFACT_NAME,
                    )
                )
            ],
        }

    def get_json(self, path: str) -> dict[str, object]:
        return self.responses[path]

    def get_json_list(self, path: str) -> list[dict[str, object]]:
        return list(self.releases) if "page=1" in path else []

    def get_paginated(self, path: str, key: str) -> list[dict[str, object]]:
        del key
        run_id = int(path.split("/")[3])
        return list(self.artifacts[run_id])


class RemoteIdentityTests(unittest.TestCase):
    def _inputs(self) -> ReleaseInputs:
        records = []
        for role, run_id in (("amd_rx_6950_xt", AMD_RUN_ID), ("nvidia_rtx_5060_laptop", NVIDIA_RUN_ID)):
            records.append(
                EvidenceInput.create(
                    role=role,
                    run_id=run_id,
                    adapter_index=0,
                    archive_sha256="4" * 64,
                    qualification_manifest_sha256="5" * 64,
                    physical_attestation_sha256="6" * 64,
                    public_receipt_sha256="7" * 64,
                )
            )
        return ReleaseInputs.create(
            repository=REPOSITORY,
            tag=TAG,
            source_run_id=SOURCE_RUN_ID,
            candidate_manifest_sha256="8" * 64,
            directml_zip_sha256="9" * 64,
            confirmation=f"PUBLISH DUAL-GPU QUALIFIED DIRECTML RELEASE {TAG}",
            evidence=records,
        )

    def test_remote_contract_requires_tag_stage_and_two_exact_qualification_artifacts(self) -> None:
        api = _RemoteApi()
        result = verify_release_remote_contract(api, self._inputs())
        self.assertTrue(result["source"]["release_absent"])
        self.assertEqual(set(result["evidence"]), set(GPU_PRODUCTS))
        api.releases.append({"id": 1, "tag_name": TAG, "draft": True})
        with self.assertRaisesRegex(AttachmentError, "already exists"):
            verify_release_remote_contract(api, self._inputs())

    def test_wrong_or_duplicate_evidence_artifact_fails_closed(self) -> None:
        api = _RemoteApi()
        api.artifacts[AMD_RUN_ID].append(dict(api.artifacts[AMD_RUN_ID][0], id=99))
        with self.assertRaises(AttachmentError):
            verify_release_remote_contract(api, self._inputs())

    @staticmethod
    def _holdout() -> HoldoutInput:
        return HoldoutInput.create(
            run_id=HOLDOUT_RUN_ID,
            prerequisite_artifact_id=93,
            prerequisite_artifact_digest="sha256:" + "4" * 64,
            plan_artifact_id=94,
            plan_artifact_digest="sha256:" + "5" * 64,
            evidence_artifact_id=95,
            evidence_artifact_digest="sha256:" + "6" * 64,
            attestation_artifact_id=96,
            attestation_artifact_digest="sha256:" + "7" * 64,
        )

    def test_every_authenticated_run_requires_exact_noncoerced_first_attempt(self) -> None:
        for run_id in (SOURCE_RUN_ID, AMD_RUN_ID, HOLDOUT_RUN_ID):
            for value in (None, 0, False, "1"):
                with self.subTest(run_id=run_id, value=value):
                    api = _RemoteApi()
                    run = api.responses[f"/actions/runs/{run_id}"]
                    if value is None:
                        run.pop("run_attempt")
                    else:
                        run["run_attempt"] = value
                    with self.assertRaisesRegex(AttachmentError, "run attempt 1"):
                        verify_release_remote_contract(
                            api, self._inputs(), self._holdout()
                        )

        verified = verify_release_remote_contract(
            _RemoteApi(), self._inputs(), self._holdout()
        )
        self.assertEqual(verified["source"]["source_build_run"]["run_attempt"], 1)
        self.assertEqual(verified["independent_holdout"]["run"]["attempt"], 1)

    def test_holdout_run_rejects_any_extra_or_duplicate_artifact_identity(self) -> None:
        api = _RemoteApi()
        api.artifacts[HOLDOUT_RUN_ID].append(
            {
                "id": 97,
                "name": "unsafe-private-debug-output",
                "expired": False,
                "size_in_bytes": 1,
                "digest": "sha256:" + "8" * 64,
            }
        )
        with self.assertRaisesRegex(AttachmentError, "exactly the four fixed"):
            verify_release_remote_contract(api, self._inputs(), self._holdout())

        api = _RemoteApi()
        api.artifacts[HOLDOUT_RUN_ID][1]["digest"] = api.artifacts[
            HOLDOUT_RUN_ID
        ][0]["digest"]
        with self.assertRaisesRegex(AttachmentError, "identity/digest/inventory"):
            verify_release_remote_contract(api, self._inputs(), self._holdout())


class AuthenticatedHoldoutManagerTests(unittest.TestCase):
    def test_holdout_publication_upload_inventory_excludes_every_private_member(self) -> None:
        public_assets = {
            LINUX_ARCHIVE_NAME: "1" * 64,
            DIRECTML_ARCHIVE_NAME: "2" * 64,
            PUBLIC_RELEASE_RECEIPT_NAME: "3" * 64,
            PUBLIC_HOLDOUT_RECEIPT_NAME: "4" * 64,
            **{
                str(record["receipt_name"]): str(index) * 64
                for index, record in enumerate(GPU_PRODUCTS.values(), start=5)
            },
        }
        upload_names = _holdout_release_upload_names(public_assets)
        self.assertEqual(set(upload_names), set(public_assets) | {CHECKSUM_NAME})
        forbidden_tokens = (
            "private-holdout",
            "metrics.json",
            "evaluation-plan.json",
            "ledger/",
            HOLDOUT_ATTESTATION_NAME,
        )
        for name in upload_names:
            self.assertFalse(any(token in name for token in forbidden_tokens))

        for private_name in (
            "private-holdout/bundle/metrics.json",
            "metrics.json",
            "evaluation-plan.json",
            "ledger/consumed.json",
            "ledger/retired.json",
            HOLDOUT_ATTESTATION_NAME,
        ):
            with self.subTest(private_name=private_name):
                changed = dict(public_assets)
                changed[private_name] = "f" * 64
                with self.assertRaisesRegex(AttachmentError, "unexpected public asset set"):
                    _holdout_release_upload_names(changed)

        missing_redacted_receipt = dict(public_assets)
        missing_redacted_receipt.pop(PUBLIC_HOLDOUT_RECEIPT_NAME)
        with self.assertRaisesRegex(AttachmentError, "unexpected public asset set"):
            _holdout_release_upload_names(missing_redacted_receipt)

    def test_attestation_round_trip_revalidates_hardware_environment_and_artifacts(self) -> None:
        inputs = RemoteIdentityTests()._inputs()
        holdout = RemoteIdentityTests._holdout()
        api = _RemoteApi()
        remote_without_holdout = verify_release_remote_contract(api, inputs)
        remote = verify_release_remote_contract(api, inputs, holdout)
        hardware = valid_rx6950_holdout_hardware_identity(
            adapter_index=0,
            qualification_run_id=AMD_RUN_ID,
            public_receipt_sha256="7" * 64,
        )
        plan_bytes = b'{"fixture":"exact-pre-access-plan"}\n'
        public = {
            "schema_version": 1,
            "status": "verified_public_holdout_bundle_requires_authenticated_origin",
            "bundle_manifest_sha256": "1" * 64,
            "bundle_content_sha256": "2" * 64,
            "receipt_sha256": "3" * 64,
            "receipt_content_sha256": "4" * 64,
            "release_policy_sha256": "5" * 64,
            "source_snapshot_sha256": "6" * 64,
            "candidate_binding_sha256": "7" * 64,
            "environment_record_sha256": "8" * 64,
            "hardware_identity_sha256": hardware["content_sha256"],
            "hardware_identity": hardware,
            "plan_sha256": sha256(plan_bytes).hexdigest(),
            "evidence_sha256": "9" * 64,
            "consumption_event_content_sha256": "a" * 64,
            "retirement_event_content_sha256": "b" * 64,
            "canonical_release_policy_matched": True,
            "release_evidence_eligible": True,
            "consumed_exactly_once": True,
            "retired": True,
            "authenticated_origin_required": True,
            "release_approved": False,
        }
        authoritative = {
            "schema_version": 1,
            "status": "verified_independent_holdout_publication_input_bundle",
            "bundle_manifest_sha256": public["bundle_manifest_sha256"],
            "bundle_content_sha256": public["bundle_content_sha256"],
            "receipt_content_sha256": public["receipt_content_sha256"],
            "release_policy_sha256": public["release_policy_sha256"],
            "source_snapshot_sha256": public["source_snapshot_sha256"],
            "environment_record_sha256": public["environment_record_sha256"],
            "hardware_identity_sha256": public["hardware_identity_sha256"],
            "canonical_release_policy_matched": True,
            "release_evidence_eligible": True,
            "authenticated_origin_required": True,
            "release_approved": False,
            "release_pointer_changed": False,
            "consumed_exactly_once": True,
            "retired": True,
        }
        candidate = {"fixture": "validated-frozen-candidate"}
        candidate_crosslink = {"fixture": "exact-candidate-crosslink"}
        amd_receipt = {
            "physical_gpu": {
                "product_name": hardware["product_name"],
                "directml_adapter_index": 0,
                "vendor_id": hardware["vendor_id"],
                "device_id": hardware["device_id"],
                "driver_version": hardware["driver_version"],
            },
            "qualification_run": {"id": AMD_RUN_ID, "attempt": 1},
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "prerequisite-stage"
            stage.mkdir()
            _write_json(
                stage / str(GPU_PRODUCTS["amd_rx_6950_xt"]["receipt_name"]),
                amd_receipt,
            )
            plan_artifact = root / "plan-artifact"
            plan_artifact.mkdir()
            (plan_artifact / HOLDOUT_BUNDLE_MEMBER_NAMES["evaluation_plan"]).write_bytes(
                plan_bytes
            )
            bundle = root / "bundle"
            (bundle / "ledger").mkdir(parents=True)
            (bundle / HOLDOUT_BUNDLE_MANIFEST_NAME).write_bytes(b"{}\n")
            (bundle / HOLDOUT_BUNDLE_MEMBER_NAMES["evaluation_plan"]).write_bytes(
                plan_bytes
            )
            authoritative_path = root / "authoritative.json"
            _write_json(authoritative_path, authoritative)
            attestation_path = root / HOLDOUT_ATTESTATION_NAME

            with (
                patch(
                    "scripts.manage_directml_release.verify_release_remote_contract",
                    return_value=remote_without_holdout,
                ),
                patch(
                    "scripts.manage_directml_release._validate_release_stage",
                    return_value=({"candidate": candidate}, {}),
                ),
                patch(
                    "scripts.manage_directml_release.validate_public_holdout_bundle",
                    return_value=public,
                ),
                patch(
                    "scripts.manage_directml_release._crosslink_holdout_candidate",
                    return_value=candidate_crosslink,
                ),
            ):
                create_authenticated_holdout_attestation(
                    api,
                    inputs,
                    stage_directory=stage,
                    staged_content_manifest_sha256="c" * 64,
                    holdout_workflow_run_id=HOLDOUT_RUN_ID,
                    holdout_workflow_run_attempt=1,
                    prerequisite_artifact_id=holdout.prerequisite_artifact_id,
                    prerequisite_artifact_digest=holdout.prerequisite_artifact_digest,
                    evidence_artifact_id=holdout.evidence_artifact_id,
                    evidence_artifact_digest=holdout.evidence_artifact_digest,
                    plan_artifact_id=holdout.plan_artifact_id,
                    plan_artifact_digest=holdout.plan_artifact_digest,
                    plan_artifact_directory=plan_artifact,
                    bundle_directory=bundle,
                    authoritative_verification_path=authoritative_path,
                    output=attestation_path,
                )

            def validate(
                *,
                selected_inputs: ReleaseInputs = inputs,
                selected_public: dict[str, object] = public,
                selected_remote: dict[str, object] = remote,
                receipt: dict[str, object] = amd_receipt,
            ) -> dict[str, object]:
                with (
                    patch(
                        "scripts.manage_directml_release.validate_public_holdout_bundle",
                        return_value=selected_public,
                    ),
                    patch(
                        "scripts.manage_directml_release._crosslink_holdout_candidate",
                        return_value=candidate_crosslink,
                    ),
                ):
                    return validate_authenticated_holdout_evidence(
                        inputs=selected_inputs,
                        holdout=holdout,
                        remote=selected_remote,
                        candidate=candidate,
                        bundle_directory=bundle,
                        attestation_directory=attestation_path,
                        amd_public_receipt=receipt,
                    )

            verified = validate()
            self.assertEqual(
                verified["runtime_prerequisite_crosslink"][
                    "environment_record_sha256"
                ],
                public["environment_record_sha256"],
            )

            for artifact_name in (
                "prerequisite_artifact",
                "plan_artifact",
                "evidence_artifact",
                "attestation_artifact",
            ):
                for field in ("id", "digest"):
                    with self.subTest(artifact=artifact_name, field=field):
                        changed_remote = deepcopy(remote)
                        artifact = changed_remote["independent_holdout"][artifact_name]
                        artifact[field] = (
                            int(artifact[field]) + 100
                            if field == "id"
                            else "sha256:" + "d" * 64
                        )
                        with self.assertRaises(AttachmentError):
                            validate(selected_remote=changed_remote)

            for field in ("environment_record_sha256", "hardware_identity_sha256"):
                with self.subTest(field=field):
                    changed_public = deepcopy(public)
                    changed_public[field] = "e" * 64
                    with self.assertRaises(AttachmentError):
                        validate(selected_public=changed_public)

            changed_receipt = deepcopy(amd_receipt)
            changed_receipt["physical_gpu"]["driver_version"] = "different"
            with self.assertRaisesRegex(AttachmentError, "physical receipt"):
                validate(receipt=changed_receipt)

            for changed_field, changed_value in (
                ("run_id", AMD_RUN_ID + 100),
                ("adapter_index", 1),
                ("public_receipt_sha256", "e" * 64),
            ):
                with self.subTest(amd_binding=changed_field):
                    changed_evidence = []
                    for evidence in inputs.evidence:
                        values = {
                            "role": evidence.role,
                            "run_id": evidence.run_id,
                            "adapter_index": evidence.adapter_index,
                            "archive_sha256": evidence.archive_sha256,
                            "qualification_manifest_sha256": (
                                evidence.qualification_manifest_sha256
                            ),
                            "physical_attestation_sha256": (
                                evidence.physical_attestation_sha256
                            ),
                            "public_receipt_sha256": evidence.public_receipt_sha256,
                        }
                        if evidence.role == "amd_rx_6950_xt":
                            values[changed_field] = changed_value
                        changed_evidence.append(EvidenceInput.create(**values))
                    changed_inputs = ReleaseInputs.create(
                        repository=inputs.repository,
                        tag=inputs.tag,
                        source_run_id=inputs.source_run_id,
                        candidate_manifest_sha256=inputs.candidate_manifest_sha256,
                        directml_zip_sha256=inputs.directml_zip_sha256,
                        confirmation=inputs.confirmation,
                        evidence=changed_evidence,
                    )
                    with self.assertRaises(AttachmentError):
                        validate(selected_inputs=changed_inputs)

            original_attestation = attestation_path.read_bytes()
            changed_attestation = json.loads(original_attestation)
            changed_attestation["runtime_prerequisite_crosslink"][
                "environment_record_sha256"
            ] = "f" * 64
            changed_attestation["attestation_content_sha256"] = (
                _holdout_attestation_content_hash(changed_attestation)
            )
            attestation_path.write_bytes(
                holdout_canonical_json_bytes(changed_attestation)
            )
            with self.assertRaises(AttachmentError):
                validate()
            attestation_path.write_bytes(original_attestation)

            (plan_artifact / HOLDOUT_BUNDLE_MEMBER_NAMES["evaluation_plan"]).write_bytes(
                b'{"fixture":"different-plan"}\n'
            )
            with (
                patch(
                    "scripts.manage_directml_release.verify_release_remote_contract",
                    return_value=remote_without_holdout,
                ),
                patch(
                    "scripts.manage_directml_release._validate_release_stage",
                    return_value=({"candidate": candidate}, {}),
                ),
                patch(
                    "scripts.manage_directml_release.validate_public_holdout_bundle",
                    return_value=public,
                ),
            ):
                with self.assertRaisesRegex(AttachmentError, "pre-access Actions artifact"):
                    create_authenticated_holdout_attestation(
                        api,
                        inputs,
                        stage_directory=stage,
                        staged_content_manifest_sha256="c" * 64,
                        holdout_workflow_run_id=HOLDOUT_RUN_ID,
                        holdout_workflow_run_attempt=1,
                        prerequisite_artifact_id=holdout.prerequisite_artifact_id,
                        prerequisite_artifact_digest=holdout.prerequisite_artifact_digest,
                        evidence_artifact_id=holdout.evidence_artifact_id,
                        evidence_artifact_digest=holdout.evidence_artifact_digest,
                        plan_artifact_id=holdout.plan_artifact_id,
                        plan_artifact_digest=holdout.plan_artifact_digest,
                        plan_artifact_directory=plan_artifact,
                        bundle_directory=bundle,
                        authoritative_verification_path=authoritative_path,
                        output=root / "second-attestation.json",
                    )


class EvidenceRoundTripTests(unittest.TestCase):
    def test_sealed_evidence_recomputes_geometry_telemetry_and_redacted_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, bundle, candidate, candidate_manifest_sha, directml_sha = _make_candidate(root)
            raw, context, _ = _make_raw_evidence(
                root,
                bundle=bundle,
                candidate=candidate,
                candidate_manifest_sha=candidate_manifest_sha,
                directml_sha=directml_sha,
            )
            sealed_directory = root / "sealed"
            result = seal_evidence(
                raw,
                sealed_directory,
                context=context,
                raw_content_manifest_sha256=sha256_file(raw / RAW_CONTENT_MANIFEST_NAME),
                observer_name="Physical Tester",
                typed_confirmation=expected_physical_confirmation(TAG, "amd_rx_6950_xt"),
            )
            archive = sealed_directory / qualification_archive_name("amd_rx_6950_xt")
            amd = EvidenceInput.create(
                role="amd_rx_6950_xt",
                run_id=AMD_RUN_ID,
                adapter_index=0,
                archive_sha256=result["archive_sha256"],
                qualification_manifest_sha256=result["qualification_manifest_sha256"],
                physical_attestation_sha256=result["physical_attestation_sha256"],
                public_receipt_sha256=result["public_receipt_sha256"],
            )
            inputs = _release_inputs(
                candidate_manifest_sha=candidate_manifest_sha,
                directml_sha=directml_sha,
                amd=amd,
            )
            remote = {
                "source": {"tag_commit": COMMIT},
                "evidence": {
                    "amd_rx_6950_xt": {
                        "run": {"id": AMD_RUN_ID, "run_attempt": 1, "actor": "gpu-tester"},
                        "artifact": {"id": 91},
                    },
                    "nvidia_rtx_5060_laptop": {
                        "run": {"id": NVIDIA_RUN_ID, "run_attempt": 1, "actor": "gpu-tester-2"},
                        "artifact": {"id": 92},
                    },
                },
            }
            verified = validate_sealed_evidence(
                archive,
                inputs=inputs,
                evidence=amd,
                remote=remote,
                candidate=candidate,
                extraction_root=root / "verified-evidence",
            )
            serialized = json.dumps(verified["receipt"], sort_keys=True)
            self.assertNotIn("Physical Tester", serialized)
            self.assertNotIn("GPU 1 - Compute_0", serialized)
            self.assertEqual(
                verified["receipt"]["candidate"]["release_default_model"]["detail_crop_size_source_pixels"],
                768,
            )

    def test_tampered_detail_geometry_fails_even_when_all_outer_manifests_are_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, bundle, candidate, candidate_manifest_sha, directml_sha = _make_candidate(root)
            raw, context, _ = _make_raw_evidence(
                root,
                bundle=bundle,
                candidate=candidate,
                candidate_manifest_sha=candidate_manifest_sha,
                directml_sha=directml_sha,
            )
            report_path = raw / "software-evidence" / "live-release-default-no-preview.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["detail_pass"]["last_plan"]["applied_crop_width"] = 768
            _write_json(report_path, report)
            _refresh_software_manifest(raw / "software-evidence")
            (raw / RAW_CONTENT_MANIFEST_NAME).unlink()
            write_content_manifest(
                root=raw,
                output=raw / RAW_CONTENT_MANIFEST_NAME,
                kind=RAW_CONTENT_KIND,
                context=context,
            )
            with self.assertRaisesRegex(AttachmentError, "exact model-aspect geometry"):
                seal_evidence(
                    raw,
                    root / "rejected",
                    context=context,
                    raw_content_manifest_sha256=sha256_file(raw / RAW_CONTENT_MANIFEST_NAME),
                    observer_name="Physical Tester",
                    typed_confirmation=expected_physical_confirmation(TAG, "amd_rx_6950_xt"),
                )

    def test_publication_stage_requires_and_preserves_both_physical_products(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_directory, bundle, candidate, candidate_manifest_sha, directml_sha = _make_candidate(root)
            evidence_inputs: list[EvidenceInput] = []
            evidence_directories: dict[str, Path] = {}
            for role, run_id in (("amd_rx_6950_xt", AMD_RUN_ID), ("nvidia_rtx_5060_laptop", NVIDIA_RUN_ID)):
                raw, context, _ = _make_raw_evidence(
                    root,
                    bundle=bundle,
                    candidate=candidate,
                    candidate_manifest_sha=candidate_manifest_sha,
                    directml_sha=directml_sha,
                    role=role,
                )
                sealed_directory = root / f"sealed-{role}"
                result = seal_evidence(
                    raw,
                    sealed_directory,
                    context=context,
                    raw_content_manifest_sha256=sha256_file(raw / RAW_CONTENT_MANIFEST_NAME),
                    observer_name="Physical Tester",
                    typed_confirmation=expected_physical_confirmation(TAG, role),
                )
                evidence_inputs.append(
                    EvidenceInput.create(
                        role=role,
                        run_id=run_id,
                        adapter_index=0,
                        archive_sha256=result["archive_sha256"],
                        qualification_manifest_sha256=result["qualification_manifest_sha256"],
                        physical_attestation_sha256=result["physical_attestation_sha256"],
                        public_receipt_sha256=result["public_receipt_sha256"],
                    )
                )
                evidence_directories[role] = sealed_directory
            inputs = ReleaseInputs.create(
                repository=REPOSITORY,
                tag=TAG,
                source_run_id=SOURCE_RUN_ID,
                candidate_manifest_sha256=candidate_manifest_sha,
                directml_zip_sha256=directml_sha,
                confirmation=f"PUBLISH DUAL-GPU QUALIFIED DIRECTML RELEASE {TAG}",
                evidence=evidence_inputs,
            )
            remote = {
                "source": {
                    "tag_commit": COMMIT,
                    "release_absent": True,
                    "source_build_run": {
                        "id": SOURCE_RUN_ID,
                        "html_url": "https://example.invalid/source",
                    },
                    "candidate_artifact": {"id": 90, "name": "ProAim-Release-Candidate"},
                },
                "evidence": {
                    record.role: {
                        "run": {
                            "id": record.run_id,
                            "run_attempt": 1,
                            "actor": f"tester-{record.role}",
                            "html_url": f"https://example.invalid/{record.role}",
                        },
                        "artifact": {"id": 100 + index},
                    }
                    for index, record in enumerate(inputs.evidence)
                },
            }
            stage_directory = root / "release-stage"
            staged = prepare_release_stage(
                inputs=inputs,
                remote=remote,
                candidate_directory=candidate_directory,
                evidence_directories=evidence_directories,
                stage_directory=stage_directory,
                verification_run_id=500,
            )
            _, public_assets = _validate_release_stage(
                stage_directory,
                inputs=inputs,
                remote=remote,
                verification_run_id=500,
                staged_content_manifest_sha256=staged["staged_content_manifest_sha256"],
            )
            self.assertEqual(
                set(public_assets),
                {
                    LINUX_ARCHIVE_NAME,
                    DIRECTML_ARCHIVE_NAME,
                    "ProAim-Windows-DirectML-AMD-RX-6950-XT-Qualification.json",
                    "ProAim-Windows-DirectML-NVIDIA-RTX-5060-Laptop-Qualification.json",
                    "ProAim-DirectML-Release-Qualification.json",
                },
            )


class _MarkerRollbackApi:
    _api_root = "https://api.example.invalid"

    def __init__(self) -> None:
        self.releases = [
            {
                "id": 7,
                "tag_name": TAG,
                "draft": True,
                "target_commitish": COMMIT,
                "body": "body\n<!-- proaim-directml-publication-run-500 -->",
            },
            {
                "id": 8,
                "tag_name": "v9.9.9",
                "draft": True,
                "target_commitish": COMMIT,
                "body": "unrelated",
            },
        ]

    def get_json_list(self, path: str) -> list[dict[str, object]]:
        return list(self.releases) if "page=1" in path else []

    def _request(self, method: str, url: str, **_: object) -> tuple[int, dict[str, str], bytes]:
        if method != "DELETE":
            raise AssertionError(method)
        release_id = int(url.rsplit("/", 1)[1])
        self.releases = [release for release in self.releases if release["id"] != release_id]
        return 204, {}, b""


class PublicationRollbackTests(unittest.TestCase):
    def test_lost_create_response_can_rollback_only_marker_identified_release(self) -> None:
        api = _MarkerRollbackApi()
        errors = _rollback_marker_releases(
            api,
            tag=TAG,
            tag_commit=COMMIT,
            marker="proaim-directml-publication-run-500",
        )
        self.assertEqual(errors, [])
        self.assertEqual([release["id"] for release in api.releases], [8])


if __name__ == "__main__":
    unittest.main()
