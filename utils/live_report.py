"""Reproducible, privacy-bounded reports for the normal live pipeline.

The model-only benchmark deliberately excludes capture and preview.  This
module records the complementary evidence from the real application loop
without creating a second implementation of that loop.  Reports are written
only after measurement has stopped, and publication uses a new hard link so a
pre-existing result can never be replaced accidentally.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

from config import AppConfig
from utils.inference_size import compact_inference_size, normalize_inference_size
from utils.metrics import FrameTimings, MetricsSnapshot


REPORT_SCHEMA = "proaim.live-pipeline-metrics"
REPORT_SCHEMA_VERSION = 2
_DIRECTML_REQUEST = re.compile(
    r"^(?:DIRECTML|DML|DMLEXECUTIONPROVIDER)(?::(?P<index>\d+))?$",
    re.IGNORECASE,
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp without a locale-dependent suffix."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def prepare_report_destination(path: str | Path) -> Path:
    """Create the parent directory and reject an existing destination early."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise ValueError(
            f"Refusing to overwrite metrics report: {destination}. "
            "Choose a new --metrics-json path."
        )
    return destination


def write_json_atomic_new(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish JSON while refusing to replace any existing path.

    The complete, flushed temporary file is hard-linked into place.  Creating
    that destination link is atomic and fails when another process won the
    filename race.  NTFS and the ordinary Linux release filesystems support
    hard links; an unsupported filesystem fails clearly instead of degrading
    to a partial or overwrite-prone write.
    """

    destination = prepare_report_destination(path)
    document = json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ValueError(
                f"Refusing to overwrite metrics report: {destination}. "
                "Choose a new --metrics-json path."
            ) from exc
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def build_live_report(
    *,
    config: AppConfig,
    detector_summary: Mapping[str, Any],
    source_description: str,
    source_settings: Mapping[str, Any],
    capture_stats: Any,
    preview_mode: str,
    preview_stats: Any,
    metrics: MetricsSnapshot,
    elapsed_seconds: float,
    started_utc: str,
    completed_utc: str,
    termination_reason: str,
    detail_pass_stats: Mapping[str, Any] | None = None,
    directml_adapter_factory: Callable[[], Sequence[Any]] | None = None,
    model_artifact_snapshot: Mapping[str, Any] | None = None,
    labels_artifact_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe report from final live-loop state.

    Configuration is intentionally allow-listed.  Pairing keys, network
    destinations, serial paths, and activation-device paths never enter this
    object even when aim output was configured.
    """

    safe_elapsed = max(0.0, float(elapsed_seconds))
    processed_frames = int(metrics.processed_frames)
    elapsed_fps = processed_frames / safe_elapsed if safe_elapsed > 0.0 else 0.0
    source_record = _source_record(
        description=source_description,
        settings=source_settings,
    )
    preview_record = {
        "enabled": bool(config.preview),
        "draw_enabled": bool(config.draw),
        "fps_limit": float(config.preview_fps),
        "mode": str(preview_mode),
        "stats": _stats_record(preview_stats),
    }
    timing_fields = tuple(FrameTimings.__dataclass_fields__)
    timings = {
        "unit": "milliseconds",
        "fields": list(timing_fields),
        "mean": _timings_record(metrics.average),
        "p50": _timings_record(metrics.p50),
        "p95": _timings_record(metrics.p95),
        "p99": _timings_record(metrics.p99),
    }

    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "started_utc": str(started_utc),
        "completed_utc": str(completed_utc),
        "application": _application_record(),
        "config": _config_record(config),
        "model_artifact": _json_safe(
            model_artifact_snapshot
            if model_artifact_snapshot is not None
            else snapshot_artifact(config.model_path)
        ),
        "labels_artifact": _json_safe(
            labels_artifact_snapshot
            if labels_artifact_snapshot is not None
            else snapshot_artifact(config.labels_path)
        ),
        "detector_runtime": _json_safe(detector_summary),
        "directml_adapter": _directml_adapter_record(
            detector_summary,
            adapter_factory=directml_adapter_factory,
        ),
        "source": source_record,
        "capture": _stats_record(capture_stats),
        "preview": preview_record,
        "detail_pass": _json_safe(
            detail_pass_stats
            if detail_pass_stats is not None
            else {
                "enabled": config.detail_crop_size is not None,
                "requested_crop_size": config.detail_crop_size,
                # A caller that omits runtime detail stats has no authoritative
                # merge threshold to report; the normal application always
                # supplies the concrete value from DetailPassStats.
                "duplicate_iou_threshold": None,
                "unmatched_detail_reference_height": None,
                "unmatched_detail_max_reference_height": None,
                "frames_seen": 0,
                "frames_applied": 0,
                "frames_redundant": 0,
                "frames_clamped": 0,
                "primary_detections": 0,
                "detail_detections": 0,
                "cross_pass_matches": 0,
                "detail_replacements": 0,
                "unmatched_detail_accepted": 0,
                "unmatched_detail_rejected_large": 0,
                "merged_detections": 0,
                "last_plan": None,
            }
        ),
        "pipeline": {
            "processed_frames": processed_frames,
            "rolling_window_size": int(config.stats_window),
            "rolling_sample_count": min(processed_frames, int(config.stats_window)),
            "elapsed_seconds": safe_elapsed,
            "elapsed_fps": elapsed_fps,
            "update_fps": float(metrics.moving_fps),
            "timings": timings,
        },
        "termination": {
            "reason": str(termination_reason),
            "requested_max_frames": config.max_frames,
            "requested_max_seconds": config.max_seconds,
        },
        "privacy": {
            "allow_listed_config": True,
            "omitted_sensitive_fields": [
                "aim_activate_path",
                "aim_host",
                "aim_makcu_port",
                "aim_pairing_key",
            ],
        },
    }


def _config_record(config: AppConfig) -> dict[str, Any]:
    source_value = config.source.value
    if isinstance(source_value, Path):
        source_value = str(source_value)
    input_height, input_width = normalize_inference_size(config.inference_size)
    return {
        "backend": config.backend,
        "device": config.device,
        "require_full_provider": bool(config.require_full_provider),
        "model_path": str(config.model_path),
        "labels_path": str(config.labels_path),
        "source": {
            "kind": config.source.kind,
            # A file path or device index is already explicit CLI input.  No
            # auto-detected serial/input paths are copied into the report.
            "value": source_value,
        },
        "capture": {
            "size": list(config.capture_size) if config.capture_size else None,
            "fps": config.capture_fps,
            "pixel_format": config.capture_format,
            "screen_monitor": config.screen_monitor,
            "screen_region": (
                list(config.screen_region) if config.screen_region else None
            ),
            "screen_fps": config.screen_fps,
        },
        "inference": {
            # ``size`` preserves the scalar shape of existing square reports;
            # ``shape_hw`` is the unambiguous canonical form for automation.
            "size": compact_inference_size(config.inference_size),
            "shape_hw": [input_height, input_width],
            "crop_size": config.crop_size,
            "detail_crop_size": config.detail_crop_size,
            "confidence": config.confidence,
            "iou_threshold": config.iou_threshold,
            "output_format": config.output_format,
        },
        "preview": {
            "enabled": config.preview,
            "draw_enabled": config.draw,
            "fps_limit": config.preview_fps,
        },
        "stats_window": config.stats_window,
        "self_filter": {
            "enabled": config.ignore_self,
            "zone_left": config.self_zone_left,
            "zone_width": config.self_zone_width,
            "zone_height": config.self_zone_height,
        },
        "aim": {
            "enabled": config.aim,
            "target_label": config.aim_label if config.aim else None,
            "output_type": config.aim_output if config.aim else None,
        },
    }


def _source_record(
    *, description: str, settings: Mapping[str, Any]
) -> dict[str, Any]:
    copied = dict(settings)
    return {
        "description": str(description),
        "actual_settings": _json_safe(copied),
        "backend": _json_safe(copied.get("backend")),
        "preferred_backend": _json_safe(copied.get("preferred_backend")),
        "fallback_reason": _json_safe(copied.get("fallback_reason")),
    }


def snapshot_artifact(path: str | Path) -> dict[str, Any]:
    """Hash one model/labels artifact and its required companion files."""

    artifact = Path(path).expanduser()
    record = _file_record(artifact)
    companions: list[dict[str, Any]] = []
    if artifact.suffix.casefold() == ".xml":
        weights = artifact.with_suffix(".bin")
        if weights.is_file():
            companions.append(_file_record(weights))
    record["companions"] = companions
    return record


def verify_artifact_unchanged(
    path: str | Path,
    expected: Mapping[str, Any],
    *,
    description: str,
) -> None:
    """Fail when a live session's path no longer matches its initial bytes."""

    actual = snapshot_artifact(path)
    if actual != dict(expected):
        raise RuntimeError(
            f"{description} changed while the live pipeline was running; "
            "refusing to publish qualification evidence"
        )


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {
        # Keep reports comparable across extraction directories and avoid
        # publishing a tester's account/home path. The configured model/source
        # fields still identify what the tester explicitly selected.
        "name": path.name,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _directml_adapter_record(
    summary: Mapping[str, Any],
    *,
    adapter_factory: Callable[[], Sequence[Any]] | None,
) -> dict[str, Any] | None:
    active = tuple(str(item) for item in summary.get("active_providers", ()))
    requested_provider = str(summary.get("requested_provider", ""))
    if (
        "DmlExecutionProvider" not in active
        and requested_provider != "DmlExecutionProvider"
    ):
        return None

    requested_input = str(summary.get("requested_device_input", ""))
    match = _DIRECTML_REQUEST.fullmatch(requested_input.strip())
    requested_index = int(match.group("index")) if match and match.group("index") else None
    configured_index = _provider_device_id(
        summary.get("provider_option_overrides")
    )
    provider_index = _provider_device_id(summary.get("provider_options"))
    effective_index = (
        requested_index
        if requested_index is not None
        else provider_index if provider_index is not None else 0
    )
    source = (
        "explicit_device_request"
        if requested_index is not None
        else "provider_report"
        if provider_index is not None
        else "DirectML_default_device_0"
    )

    if adapter_factory is None:
        try:
            from detection.hardware import scan_windows_directml_adapters

            adapter_factory = scan_windows_directml_adapters
        except Exception:
            adapter_factory = lambda: ()
    try:
        adapters = tuple(adapter_factory())
    except Exception:
        adapters = ()
    selected = next(
        (item for item in adapters if int(getattr(item, "index", -1)) == effective_index),
        None,
    )
    descriptor = None
    if selected is not None:
        descriptor = {
            "index": int(getattr(selected, "index")),
            "name": str(getattr(selected, "name", "")),
            "vendor_id": str(getattr(selected, "vendor_id", "")),
            "device_id": str(getattr(selected, "device_id", "")),
            "dedicated_vram_bytes": int(getattr(selected, "dedicated_vram", 0)),
        }
    mismatch = (
        requested_index is not None
        and provider_index is not None
        and requested_index != provider_index
    )
    return {
        "requested_index": requested_index,
        "configured_index": configured_index,
        "provider_reported_index": provider_index,
        "effective_index": effective_index,
        "selection_source": source,
        "enumeration_status": (
            "matched_dxgi_adapter" if descriptor is not None else "not_enumerated"
        ),
        "requested_provider_mismatch": mismatch,
        "qualification_status": (
            "failed_provider_index_mismatch" if mismatch else "requires_task_manager_confirmation"
        ),
        "descriptor": descriptor,
        # Provider activation and adapter enumeration are software evidence.
        # Release qualification still keeps an independent GPU-activity check.
        "task_manager_confirmation_required": True,
    }


def _provider_device_id(groups: Any) -> int | None:
    if not isinstance(groups, Mapping):
        return None
    options = groups.get("DmlExecutionProvider")
    if not isinstance(options, Mapping):
        return None
    value = options.get("device_id")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    if parsed >= 0:
        return parsed
    return None


def _application_record() -> dict[str, Any]:
    build_info = None
    candidates = [Path(sys.executable).resolve().parent / "BUILD-INFO.json"]
    if not bool(getattr(sys, "frozen", False)):
        candidates.append(Path(__file__).resolve().parents[1] / "BUILD-INFO.json")
    for candidate in candidates:
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(raw, Mapping):
            allowed = (
                "application",
                "commit",
                "commit_time",
                "dirty",
                "runtime_variant",
                "schema",
            )
            build_info = {key: _json_safe(raw.get(key)) for key in allowed}
            break
    return {
        "name": "ProAim",
        "frozen": bool(getattr(sys, "frozen", False)),
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "build_info": build_info,
    }


def _stats_record(stats: Any) -> dict[str, Any]:
    if stats is None:
        return {}
    try:
        values = asdict(stats)
    except TypeError:
        if isinstance(stats, Mapping):
            values = dict(stats)
        else:
            values = {
                name: getattr(stats, name)
                for name in dir(stats)
                if not name.startswith("_")
                and isinstance(getattr(stats, name), (bool, int, float, str, type(None)))
            }
    return _json_safe(values)


def _timings_record(timings: FrameTimings) -> dict[str, float]:
    return {
        field: float(getattr(timings, field))
        for field in FrameTimings.__dataclass_fields__
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
