from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

from config import AppConfig, parse_args
from utils.inference_size import compact_inference_size


WINDOW_NAME = "ProAim"
AIM_CONTINUATION_CONFIDENCE_FLOOR = 0.15


def _calibration_requested(config: AppConfig) -> bool:
    return config.aim_calibration_evidence is not None


def _active_profile_requested(config: AppConfig) -> bool:
    return config.aim_makcu_active_profile is not None


def _calibrated_controller_from_active_profile(
    profile,
    *,
    max_step: int,
    vertical_rate_ratio: float,
):
    """Build the bounded numeric controller represented by one strict profile."""

    from aiming.makcu_calibrated_control import (
        CalibratedControlConfig,
        CalibratedPlant,
        MakcuCalibratedController,
    )
    from aiming.makcu_calibration_activation import ActiveMakcuCalibrationProfile

    if not isinstance(profile, ActiveMakcuCalibrationProfile):
        raise TypeError("profile must be an ActiveMakcuCalibrationProfile")
    if isinstance(max_step, bool) or not isinstance(max_step, int) or max_step <= 0:
        raise ValueError("calibrated maximum step must be a positive integer")
    ratio = float(vertical_rate_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("calibrated vertical rate ratio must be in (0,1]")
    maximum_rate_x = float(max_step) * 60.0
    maximum_rate_y = maximum_rate_x * ratio
    return MakcuCalibratedController(
        CalibratedPlant(
            profile.fit.x.gain_pixels_per_count,
            profile.fit.y.gain_pixels_per_count,
            profile.fit.delay_seconds,
        ),
        CalibratedControlConfig(
            maximum_rate_x_counts_per_second=maximum_rate_x,
            maximum_rate_y_counts_per_second=maximum_rate_y,
        ),
    )


def _calibration_model_sha256(snapshot: object) -> str:
    """Return the exact selected-model identity, including OpenVINO weights."""

    from hashlib import sha256
    from typing import Mapping

    if not isinstance(snapshot, Mapping):
        raise ValueError("Model artifact snapshot is malformed")
    digest = snapshot.get("sha256")
    companions = snapshot.get("companions")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(companions, list)
    ):
        raise ValueError("Model artifact snapshot has no verified SHA-256")
    if not companions:
        # ONNX and other single-file models retain their ordinary file hash.
        return digest

    records: list[tuple[str, str]] = []
    for record in [snapshot, *companions]:
        if not isinstance(record, Mapping):
            raise ValueError("Model companion snapshot is malformed")
        name = record.get("name")
        record_digest = record.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(record_digest, str)
            or len(record_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in record_digest
            )
        ):
            raise ValueError("Model companion snapshot has no verified identity")
        records.append((name, record_digest))
    if len({name for name, _digest in records}) != len(records):
        raise ValueError("Model artifact snapshot contains duplicate filenames")

    composite = sha256(b"proaim-model-artifact-set-v1\0")
    for name, record_digest in sorted(records):
        encoded_name = name.encode("utf-8")
        composite.update(len(encoded_name).to_bytes(4, "big"))
        composite.update(encoded_name)
        composite.update(bytes.fromhex(record_digest))
    return composite.hexdigest()


def _calibration_source_identity(
    project_root: Path | None = None,
) -> tuple[str, str]:
    """Bind calibration to one exact frozen build or development source tree."""

    from hashlib import sha256
    import json
    import os
    import re
    import stat
    import subprocess

    root = (project_root or Path(__file__).resolve().parent).resolve()
    if bool(getattr(sys, "frozen", False)):
        build_info = Path(sys.executable).resolve().parent / "BUILD-INFO.json"
        try:
            payload = build_info.read_bytes()
            document = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Calibration requires a valid adjacent BUILD-INFO.json"
            ) from exc
        commit = document.get("commit") if isinstance(document, dict) else None
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{7,64}", commit) is None:
            raise RuntimeError("Calibration build metadata has no exact source commit")
        executable = Path(sys.executable).resolve()
        identity = sha256(b"proaim-frozen-build-v1\0")
        identity.update(payload)
        try:
            with executable.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    identity.update(chunk)
        except OSError as exc:
            raise RuntimeError("Could not hash the calibration executable") from exc
        return commit, f"frozen-executable-sha256:{identity.hexdigest()}"

    def git(*arguments: str) -> bytes:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=False,
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "Calibration requires readable source-control identity"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                "Calibration requires readable source-control identity"
            )
        return completed.stdout

    commit = git("rev-parse", "--verify", "HEAD").decode("ascii", "strict").strip()
    if re.fullmatch(r"[0-9a-f]{7,64}", commit) is None:
        raise RuntimeError("Calibration source commit is invalid")
    diff = git("diff", "--binary", "--no-ext-diff", "HEAD", "--")
    untracked = tuple(
        value.decode("utf-8", "surrogateescape")
        for value in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if value
    )
    if not diff and not untracked:
        return commit, f"source-clean:{commit}"

    identity = sha256(b"proaim-development-tree-v1\0")
    identity.update(commit.encode("ascii"))
    identity.update(b"\0tracked-diff\0")
    identity.update(diff)
    for relative_text in sorted(untracked):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Git reported an unsafe untracked source path")
        candidate = root / relative
        try:
            metadata = os.lstat(candidate)
        except OSError as exc:
            raise RuntimeError(
                "Calibration source changed while its identity was captured"
            ) from exc
        identity.update(b"\0untracked\0")
        identity.update(relative_text.encode("utf-8", "surrogateescape"))
        identity.update(b"\0")
        identity.update(str(stat.S_IMODE(metadata.st_mode)).encode("ascii"))
        identity.update(b"\0")
        if stat.S_ISLNK(metadata.st_mode):
            identity.update(os.readlink(candidate).encode("utf-8", "surrogateescape"))
        elif stat.S_ISREG(metadata.st_mode):
            try:
                with candidate.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        identity.update(chunk)
            except OSError as exc:
                raise RuntimeError(
                    "Calibration source changed while its identity was captured"
                ) from exc
        else:
            raise RuntimeError("Calibration source contains an unsupported untracked entry")
    return commit, f"source-tree-sha256:{identity.hexdigest()}"


def _positive_integral_runtime_value(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Calibration {name} is not a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Calibration {name} is unavailable") from exc
    if not math.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"Calibration {name} is not a positive integer")
    return int(numeric)


def _calibration_capture_identity(
    config: AppConfig,
    settings: object,
) -> tuple[str, str, int, str, int, int, float, str, int]:
    """Extract only negotiated live-capture identity from the running source."""

    import json
    from typing import Mapping

    if not isinstance(settings, Mapping):
        raise ValueError("Calibration capture settings are unavailable")
    if config.source.kind not in {"screen", "device"}:
        raise ValueError("Calibration requires a live screen or capture-card source")
    width = _positive_integral_runtime_value(settings.get("width"), "capture width")
    height = _positive_integral_runtime_value(settings.get("height"), "capture height")
    try:
        fps = float(settings.get("fps"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration capture frame rate is unavailable") from exc
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("Calibration capture frame rate must be positive")
    try:
        rotation = int(settings.get("rotation_degrees", 180 if config.capture_rotate_180 else 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration capture rotation is unavailable") from exc
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Calibration capture rotation is invalid")

    if config.source.kind == "device":
        backend_value = settings.get("backend")
        if not isinstance(backend_value, str) or not backend_value.strip():
            raise ValueError("Calibration capture-card backend is unavailable")
        capture_backend = backend_value.strip()
        capture_buffer_size = _positive_integral_runtime_value(
            settings.get("buffer_size"), "capture buffer size"
        )
        actual_source = settings.get("source", config.source.value)
        if actual_source is None:
            raise ValueError("Calibration capture-card index is unavailable")
        capture_index = str(actual_source)
        pixel_format_value = settings.get("pixel_format")
        if not isinstance(pixel_format_value, str) or not pixel_format_value.strip():
            raise ValueError("Calibration capture-card pixel format is unavailable")
        pixel_format = pixel_format_value.strip().upper()
        capture_kind = "camera"
    else:
        # Screen backends expose different adapter/output fields.  Canonical
        # JSON retains their actual selection without pretending they share an
        # integer index namespace.
        index_record = {
            key: settings[key]
            for key in (
                "backend",
                "device_index",
                "left",
                "monitor",
                "output_index",
                "top",
            )
            if key in settings and settings[key] is not None
        }
        capture_index = json.dumps(
            index_record,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        pixel_format = str(settings.get("pixel_format") or "BGR8").strip().upper()
        capture_backend = str(settings.get("backend") or "").strip()
        if not capture_backend:
            raise ValueError("Calibration screen-capture backend is unavailable")
        capture_buffer_size = 0
        capture_kind = "screen"
    if not capture_index or len(capture_index) > 256:
        raise ValueError("Calibration capture identity is unavailable")
    return (
        capture_kind,
        capture_backend,
        capture_buffer_size,
        capture_index,
        width,
        height,
        fps,
        pixel_format,
        rotation,
    )


def _build_calibration_runtime_binding(
    config: AppConfig,
    *,
    detector_summary: object,
    capture_settings: object,
    makcu_identity_token: object,
    model_artifact_snapshot: object,
    labels_artifact_snapshot: object,
    source_identity: tuple[str, str] | None = None,
    context_name: str | None = None,
    aim_mode: str | None = None,
):
    """Create the exact immutable identity used by evidence and activation."""

    from typing import Mapping

    from aiming.makcu_calibration_session import (
        CalibrationRuntimeBinding,
        normalize_calibration_context,
    )
    from hashlib import sha256
    import json
    import os
    import re

    if not isinstance(detector_summary, Mapping):
        raise ValueError("Calibration detector runtime summary is unavailable")
    if not isinstance(labels_artifact_snapshot, Mapping):
        raise ValueError("Labels artifact snapshot is malformed")
    labels_sha256 = labels_artifact_snapshot.get("sha256")
    if (
        not isinstance(labels_sha256, str)
        or len(labels_sha256) != 64
        or any(character not in "0123456789abcdef" for character in labels_sha256)
    ):
        raise ValueError("Labels artifact snapshot has no verified SHA-256")
    if not isinstance(makcu_identity_token, str) or len(makcu_identity_token) != 64:
        raise RuntimeError("Verified MAKCU identity is unavailable for calibration")

    shape = detector_summary.get("input_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise ValueError("Calibration detector input shape is unavailable")
    inference_height = _positive_integral_runtime_value(shape[-2], "inference height")
    inference_width = _positive_integral_runtime_value(shape[-1], "inference width")
    active_values = detector_summary.get("active_providers")
    active_provider = ""
    if isinstance(active_values, (list, tuple)) and active_values:
        active_provider = str(active_values[0]).strip()
    if not active_provider:
        active_provider = str(detector_summary.get("device") or "").strip()
    requested_provider = str(
        detector_summary.get("requested_provider")
        or detector_summary.get("requested_device")
        or config.device
    ).strip()
    active_device = str(detector_summary.get("device") or active_provider).strip()
    if not requested_provider or not active_provider or not active_device:
        raise ValueError("Calibration detector provider/device identity is incomplete")

    runtime_version = str(
        detector_summary.get("onnxruntime_version")
        or detector_summary.get("openvino_version")
        or ""
    ).strip()
    if runtime_version.casefold() in {"", "unknown", "unavailable", "n/a", "none"}:
        raise ValueError("Calibration runtime version identity is unavailable")
    provider_identity_record = {
        key: detector_summary.get(key)
        for key in (
            "active_providers",
            "configured_session_options",
            "execution_devices",
            "num_streams",
            "num_streams_requested",
            "output_format",
            "performance_hint",
            "provider_chain",
            "provider_option_overrides",
            "provider_options",
            "require_full_provider",
            "runtime_ep_fail_fallback_disabled",
        )
        if key in detector_summary
    }
    if config.backend == "onnxruntime":
        if detector_summary.get("provider_options_status") != "ok":
            raise ValueError(
                "Calibration requires a successful ONNX provider-options query"
            )
        if (
            detector_summary.get("require_full_provider") is not True
            or detector_summary.get("runtime_ep_fail_fallback_disabled") is not True
        ):
            raise ValueError(
                "Calibration requires strict ONNX GPU provider fallback controls"
            )
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION") is not None:
            raise ValueError("Calibration refuses HSA_OVERRIDE_GFX_VERSION")
        provider_identity_record["runtime_environment"] = {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("MIGRAPHX_", "ROCR_", "HIP_", "HSA_"))
            or key in {"CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}
        }
    try:
        provider_identity_bytes = json.dumps(
            provider_identity_record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Calibration provider options are not canonically serializable"
        ) from exc
    provider_options_sha256 = sha256(provider_identity_bytes).hexdigest()

    physical_identity = str(
        detector_summary.get("physical_device_identity") or ""
    ).strip()
    if not physical_identity and any(
        token in active_provider.casefold() for token in ("migraphx", "rocm")
    ):
        rocr_selector = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
        if re.fullmatch(r"GPU-[0-9A-Fa-f]{16,64}", rocr_selector):
            physical_identity = f"rocr:{rocr_selector}"
    if (
        not physical_identity
        or len(physical_identity) > 256
        or any(ord(character) < 0x20 for character in physical_identity)
    ):
        raise ValueError(
            "Calibration requires an unambiguous physical accelerator identity"
        )
    physical_device_token = sha256(physical_identity.encode("utf-8")).hexdigest()

    (
        capture_kind,
        capture_backend,
        capture_buffer_size,
        capture_index,
        capture_width,
        capture_height,
        capture_fps,
        pixel_format,
        rotation_degrees,
    ) = _calibration_capture_identity(config, capture_settings)
    source_commit, build_identity = source_identity or _calibration_source_identity()
    selected_context = context_name or config.aim_calibration_context
    normalized_mode = normalize_calibration_context(selected_context)
    selected_mode = aim_mode or normalized_mode
    if selected_mode != normalized_mode:
        raise ValueError("Calibration context and aim mode do not match")
    return CalibrationRuntimeBinding(
        model_sha256=_calibration_model_sha256(model_artifact_snapshot),
        labels_sha256=labels_sha256,
        source_commit=source_commit,
        build_identity=build_identity,
        backend=config.backend,
        runtime_version=runtime_version,
        requested_provider=requested_provider,
        active_provider=active_provider,
        active_device=active_device,
        provider_options_sha256=provider_options_sha256,
        physical_device_token=physical_device_token,
        inference_width=inference_width,
        inference_height=inference_height,
        detail_pass_enabled=False,
        capture_kind=capture_kind,
        capture_backend=capture_backend,
        capture_buffer_size=capture_buffer_size,
        capture_index=capture_index,
        capture_width=capture_width,
        capture_height=capture_height,
        capture_fps=capture_fps,
        pixel_format=pixel_format,
        rotation_degrees=rotation_degrees,
        makcu_identity_token=makcu_identity_token,
        activation_button=config.aim_makcu_button,
        aim_label=config.aim_label or "",
        head_ratio=config.aim_head_ratio,
        invert_x=config.aim_invert_x,
        invert_y=config.aim_invert_y,
        context_name=selected_context,
        aim_mode=selected_mode,
    )


def _calibration_observation_and_target(
    detections,
    frame_shape: tuple[int, ...],
    *,
    aim_label: str,
    head_ratio: float,
    configured_confidence: float,
    invert_x: bool,
    invert_y: bool,
    self_exclusion_safe: bool,
    measurement_ns: int,
):
    """Return one raw, strong, exact-label target in 1920x1080 coordinates."""

    from aiming import head_target_point
    from aiming.makcu_calibration_session import CalibrationObservation

    if not self_exclusion_safe:
        return None, None
    if len(frame_shape) < 2:
        return None, None
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return None, None
    threshold = float(configured_confidence)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Calibration confidence must be finite and in [0,1]")
    normalized_label = aim_label.strip().casefold()
    candidates = tuple(
        detection
        for detection in detections
        if detection.class_name.strip().casefold() == normalized_label
        and math.isfinite(float(detection.confidence))
        and float(detection.confidence) >= threshold
    )
    if len(candidates) != 1:
        return None, None
    target = candidates[0]
    try:
        left, top, right, bottom = (float(value) for value in target.box)
    except (TypeError, ValueError):
        return None, None
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None, None
    if not (0.0 <= left < right <= width and 0.0 <= top < bottom <= height):
        return None, None
    target_x, target_y = head_target_point(target, head_ratio)
    reference_x = (target_x - width / 2.0) * (1920.0 / width)
    reference_y = (target_y - height / 2.0) * (1080.0 / height)
    if invert_x:
        reference_x = -reference_x
    if invert_y:
        reference_y = -reference_y
    observation = CalibrationObservation(
        measurement_ns=measurement_ns,
        error_x=reference_x,
        error_y=reference_y,
        confidence=float(target.confidence),
        exact_label=True,
        unique_candidates=1,
        self_safe=True,
        is_prediction=False,
        target_identity="unique-exact-target",
        normalized_bbox=(
            left / width,
            top / height,
            right / width,
            bottom / height,
        ),
    )
    return observation, target


def _partition_detections_by_confidence(
    detections,
    configured_confidence: float | None,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Separate normal detections from aim-only continuation evidence."""

    source = tuple(detections)
    if configured_confidence is None:
        return source, ()
    normal: list[object] = []
    continuation: list[object] = []
    for detection in source:
        destination = (
            normal
            if float(detection.confidence) >= configured_confidence
            else continuation
        )
        destination.append(detection)
    return tuple(normal), tuple(continuation)


@dataclass(frozen=True, slots=True)
class HardAimGuardResult:
    """Aim-only detections and exact-label attribution for the hard self guard."""

    detections: tuple[object, ...]
    removed_exact_label_boxes: int = 0
    targetless_after_exact_removal: bool = False


def _apply_hard_aim_guard(
    detections,
    frame_shape: tuple[int, ...],
    *,
    self_zone,
    aim_label: str,
    configured_confidence: float | None = None,
) -> HardAimGuardResult:
    """Apply the existing conservative aim guard and attribute exact labels.

    The guard remains player-like rather than label-specific.  Attribution is
    exact-label-specific so an unrelated guarded box cannot revoke prediction
    grace for a genuinely empty target-label sample.
    """

    from utils.self_filter import is_player_like

    source = tuple(detections)
    if self_zone is None or not source:
        return HardAimGuardResult(source)

    normalized_aim_label = aim_label.strip().lower()
    if configured_confidence is not None:
        configured_confidence = float(configured_confidence)
        if not math.isfinite(configured_confidence) or not (
            0.0 <= configured_confidence <= 1.0
        ):
            raise ValueError(
                "configured_confidence must be finite and between 0 and 1"
            )
    guarded: list[object] = []
    removed_exact_label_boxes = 0
    for detection in source:
        drop_for_aim = (
            is_player_like(detection)
            and self_zone.candidate_score(detection.box, frame_shape) is not None
        )
        if drop_for_aim:
            if detection.class_name.strip().lower() == normalized_aim_label:
                removed_exact_label_boxes += 1
            continue
        guarded.append(detection)

    targetless_after_exact_removal = (
        removed_exact_label_boxes > 0
        and not any(
            detection.class_name.strip().lower() == normalized_aim_label
            and (
                configured_confidence is None
                or float(detection.confidence) >= configured_confidence
            )
            for detection in guarded
        )
    )
    return HardAimGuardResult(
        tuple(guarded),
        removed_exact_label_boxes=removed_exact_label_boxes,
        targetless_after_exact_removal=targetless_after_exact_removal,
    )


@dataclass(frozen=True, slots=True)
class AimInputTelemetrySnapshot:
    samples: int = 0
    exact_label_samples: int = 0
    self_filter_unsafe_samples: int = 0
    hard_guard_removed_exact_boxes: int = 0
    hard_guard_targetless_samples: int = 0


class AimInputTelemetry:
    """Cumulative non-spatial counters explaining missing aim inputs."""

    def __init__(self, aim_label: str) -> None:
        self._normalized_aim_label = aim_label.strip().lower()
        self._samples = 0
        self._exact_label_samples = 0
        self._self_filter_unsafe_samples = 0
        self._hard_guard_removed_exact_boxes = 0
        self._hard_guard_targetless_samples = 0

    def record_sample(self, detections) -> None:
        self._samples += 1
        if any(
            detection.class_name.strip().lower() == self._normalized_aim_label
            for detection in detections
        ):
            self._exact_label_samples += 1

    def record_self_filter(self, *, aim_safe: bool) -> None:
        if not aim_safe:
            self._self_filter_unsafe_samples += 1

    def record_hard_guard(self, result: HardAimGuardResult) -> None:
        self._hard_guard_removed_exact_boxes += result.removed_exact_label_boxes
        if result.targetless_after_exact_removal:
            self._hard_guard_targetless_samples += 1

    def snapshot(self) -> AimInputTelemetrySnapshot:
        return AimInputTelemetrySnapshot(
            samples=self._samples,
            exact_label_samples=self._exact_label_samples,
            self_filter_unsafe_samples=self._self_filter_unsafe_samples,
            hard_guard_removed_exact_boxes=self._hard_guard_removed_exact_boxes,
            hard_guard_targetless_samples=self._hard_guard_targetless_samples,
        )


def _build_capture(config: AppConfig):
    from capture import DesktopCaptureSource, OpenCVCaptureSource

    if config.source.kind == "screen":
        return DesktopCaptureSource(
            monitor=config.screen_monitor,
            region=config.screen_region,
            fps=config.screen_fps,
        )

    width = config.capture_size[0] if config.capture_size else None
    height = config.capture_size[1] if config.capture_size else None
    if config.source.kind == "device":
        assert isinstance(config.source.value, int)
        return OpenCVCaptureSource(
            config.source.value,
            width=width,
            height=height,
            fps=config.capture_fps,
            # High-rate UVC devices require at least double buffering to keep
            # one transfer in flight while the previous frame is consumed.
            # The source still publishes into a one-frame latest-only mailbox,
            # so this does not create an application-side stale-frame queue.
            buffer_size=2,
            pixel_format=config.capture_format,
            rotate_180=config.capture_rotate_180,
        )

    assert isinstance(config.source.value, Path)
    if not config.source.value.is_file():
        raise FileNotFoundError(f"Video file not found: {config.source.value}")
    return OpenCVCaptureSource(
        config.source.value,
        rotate_180=config.capture_rotate_180,
    )


def _print_startup(detector, source) -> None:
    summary = detector.runtime_summary
    # Either backend may be running here, so report whichever one built this
    # detector rather than assuming OpenVINO.
    runtime = summary.get("runtime", "OpenVINO")
    version = summary.get("openvino_version") or summary.get(
        "onnxruntime_version", "unknown"
    )
    print(f"{runtime} {version}")
    devices = ", ".join(detector.available_devices) or "none"
    print(f"Detected {runtime} devices: {devices}")
    requested = summary.get("requested_device")
    active_device = summary.get("device")
    if requested and active_device and requested != active_device:
        inference_label = f"{active_device} (requested {requested})"
    else:
        inference_label = active_device
    hint = summary.get("performance_hint") or ", ".join(
        summary.get("active_providers", ())
    )
    print(
        f"Inference: {inference_label} | input {summary.get('input_shape')} | "
        f"hint {hint} | one synchronous request"
    )
    print(f"Model: {summary.get('model_path')}")
    print(f"Source: {source.description}")


def _warn_on_capture_mismatch(config: AppConfig, settings) -> None:
    """Say so when the driver refused the requested format or frame rate.

    Capture properties are hints.  A card asked for a mode it cannot provide
    silently falls back, and the usual symptom is a frame rate far below the
    advertised one, so the difference is worth stating rather than leaving the
    user to infer it from the numbers.
    """

    requested_format = settings.get("requested_pixel_format")
    granted_format = settings.get("pixel_format")
    if requested_format and granted_format and requested_format != granted_format:
        print(
            f"Warning: requested pixel format {requested_format} but the device "
            f"is running {granted_format}. The frame rate is limited by the "
            f"format the driver actually granted.",
            file=sys.stderr,
        )

    requested_fps = config.capture_fps
    granted_fps = settings.get("fps")
    if (
        requested_fps
        and granted_fps
        # Drivers routinely round; only a real shortfall is worth a warning.
        and granted_fps < requested_fps * 0.9
    ):
        print(
            f"Warning: requested {requested_fps:g} fps but the device reports "
            f"{granted_fps:g} fps. Uncompressed modes such as NV12 and YUY2 are "
            f"limited by USB bandwidth; MJPG usually reaches higher rates.",
            file=sys.stderr,
        )


def _format_settings(settings) -> str:
    return ", ".join(f"{key}={value}" for key, value in settings.items() if value is not None)


def _target_tracker_telemetry_summary(previous, current, elapsed_seconds: float) -> str:
    """Format aggregate raw-measurement versus tracker-output diagnostics."""

    elapsed = max(float(elapsed_seconds), 1e-9)

    def count_delta(name: str) -> int:
        return max(0, int(getattr(current, name)) - int(getattr(previous, name)))

    def total_delta(name: str) -> float:
        return float(getattr(current, name)) - float(getattr(previous, name))

    updates = count_delta("updates")
    candidates = count_delta("candidate_samples")
    measurements = count_delta("measurement_samples")
    continuation_measurements = count_delta("continuation_measurement_samples")
    outputs = count_delta("output_samples")
    compared = count_delta("compared_samples")

    def mean(name: str, *, absolute: bool = False) -> float:
        value = total_delta(name)
        if absolute:
            value = max(0.0, value)
        return value / compared if compared else 0.0

    rejected = max(0, candidates - measurements)
    return (
        f"TRACK samples {updates / elapsed:.0f}/s | "
        f"raw/out {measurements / elapsed:.0f}/{outputs / elapsed:.0f}/s | "
        f"continued-low {continuation_measurements / elapsed:.0f}/s | "
        f"rejected {rejected / elapsed:.0f}/s | "
        f"raw-track abs X/Y "
        f"{mean('residual_abs_x', absolute=True):.1f}/"
        f"{mean('residual_abs_y', absolute=True):.1f}px | "
        f"signed {mean('residual_x'):+.1f}/{mean('residual_y'):+.1f}px | "
        f"losses {count_delta('target_loss_transitions')}"
    )


def _aim_input_telemetry_summary(previous, current, elapsed_seconds: float) -> str:
    """Format per-report deltas for non-spatial aim input causes."""

    elapsed = max(float(elapsed_seconds), 1e-9)

    def delta(name: str) -> int:
        return max(0, int(getattr(current, name)) - int(getattr(previous, name)))

    return (
        f"AIM INPUT {delta('samples') / elapsed:.0f}/s | "
        f"exact {delta('exact_label_samples') / elapsed:.0f}/s | "
        f"self-unsafe {delta('self_filter_unsafe_samples')} | "
        f"guard exact boxes {delta('hard_guard_removed_exact_boxes')} | "
        f"guard targetless {delta('hard_guard_targetless_samples')}"
    )


def _makcu_telemetry_summary(previous, current, elapsed_seconds: float) -> str:
    """Format passive output-loop counters collected since the prior report."""

    elapsed = max(float(elapsed_seconds), 1e-9)

    def delta(name: str) -> int:
        return max(0, int(getattr(current, name)) - int(getattr(previous, name)))

    ticks = delta("output_ticks")

    def duty(name: str) -> float:
        return 100.0 * delta(name) / ticks if ticks else 0.0

    commands = delta("movement_commands")
    abs_x = delta("emitted_abs_x")
    abs_y = delta("emitted_abs_y")
    net_x = int(getattr(current, "emitted_x")) - int(getattr(previous, "emitted_x"))
    net_y = int(getattr(current, "emitted_y")) - int(getattr(previous, "emitted_y"))
    control_samples = delta("control_samples")

    def control_mean(name: str) -> float:
        total = max(
            0.0,
            float(getattr(current, name)) - float(getattr(previous, name)),
        )
        return total / control_samples if control_samples else 0.0

    saturation_x = (
        100.0 * delta("saturated_x_samples") / control_samples
        if control_samples
        else 0.0
    )
    saturation_y = (
        100.0 * delta("saturated_y_samples") / control_samples
        if control_samples
        else 0.0
    )
    return (
        f"MAKCU loop {ticks / elapsed:.0f} Hz | button {duty('button_pressed_ticks'):.0f}% | "
        f"target {duty('target_present_ticks'):.0f}% | "
        f"fresh {duty('fresh_target_ticks'):.0f}% | "
        f"authorized {duty('authorized_ticks'):.0f}% | "
        f"moves {commands / elapsed:.0f}/s | "
        f"abs counts X/Y {abs_x / elapsed:.0f}/{abs_y / elapsed:.0f}/s | "
        f"net X/Y {net_x / elapsed:+.0f}/{net_y / elapsed:+.0f}/s | "
        f"CTRL samples {control_samples / elapsed:.0f}/s | "
        f"error abs X/Y {control_mean('control_error_abs_x'):.1f}/"
        f"{control_mean('control_error_abs_y'):.1f}px | "
        f"pursuit X/Y {control_mean('pursuit_abs_x') * 60.0:.0f}/"
        f"{control_mean('pursuit_abs_y') * 60.0:.0f} cps | "
        f"saturation X/Y {saturation_x:.0f}/{saturation_y:.0f}% | "
        f"pursuit resets {delta('pursuit_resets')}"
    )


def _start_optional_aiming(aim_controller, aim_sensor):
    """Start optional aim devices without preventing capture and preview."""

    if aim_controller is None:
        return None, None
    try:
        if aim_sensor is not None:
            aim_sensor.start()
        aim_controller.start()
    except (RuntimeError, OSError, ValueError) as exc:
        try:
            aim_controller.stop()
        except (RuntimeError, OSError, ValueError):
            pass
        if aim_sensor is not None:
            try:
                aim_sensor.stop()
            except (RuntimeError, OSError, ValueError):
                pass
        print(
            f"Warning: detection-driven aim is disabled: {exc}. "
            "Capture, inference, and preview will continue.",
            file=sys.stderr,
        )
        return None, None
    return aim_controller, aim_sensor


def _validate_aim_safety(config: AppConfig) -> None:
    """Reject fail-open aiming configurations even when parse_args was bypassed."""

    if not config.aim:
        return
    if not (config.aim_label or "").strip():
        raise ValueError("Detection-driven aim requires an explicit target label")
    if not config.ignore_self:
        raise ValueError(
            "Detection-driven aim requires the third-person self filter to be enabled"
        )
    if config.aim_output == "remote":
        raise ValueError(
            "Remote aim is unavailable because no authenticated, physically gated "
            "receiver is included"
        )
    if config.aim_output not in {"local", "makcu"}:
        raise ValueError(
            f"Unsupported safe aim output: {config.aim_output!r}; "
            "expected 'local' or 'makcu'"
        )
    if config.aim_output == "local":
        if not (config.aim_activate_path or "").strip():
            raise ValueError("Local aim requires an explicit physical activation device")
        if isinstance(config.aim_activate_threshold, bool):
            raise ValueError(
                "Local aim activation threshold must be finite, greater than 0, "
                "and at most 1"
            )
        try:
            activation_threshold = float(config.aim_activate_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Local aim activation threshold must be a finite number in (0,1]"
            ) from exc
        if not math.isfinite(activation_threshold) or not 0.0 < activation_threshold <= 1.0:
            raise ValueError(
                "Local aim activation threshold must be finite, greater than 0, "
                "and at most 1"
            )


def _validate_calibration_safety(config: AppConfig) -> None:
    """Repeat the CLI's calibration gates for direct ``run(AppConfig)`` callers."""

    if not _calibration_requested(config):
        return
    import os

    from aiming.makcu_calibration_session import normalize_calibration_context

    if not config.aim or config.aim_output != "makcu":
        raise ValueError("MAKCU calibration requires live MAKCU aiming")
    if config.source.kind not in {"screen", "device"}:
        raise ValueError("MAKCU calibration requires a live capture source")
    if not (config.aim_makcu_port or "").strip():
        raise ValueError("MAKCU calibration requires an explicit verified port")
    if not config.ignore_self:
        raise ValueError("MAKCU calibration requires self-avatar exclusion")
    if config.detail_crop_size is not None:
        raise ValueError("MAKCU calibration cannot use the detail pass")
    if config.aim_makcu_active_profile is not None:
        raise ValueError("MAKCU calibration cannot load an active aim profile")
    if config.metrics_json is not None:
        raise ValueError("MAKCU calibration cannot write a live metrics report")
    if config.max_frames is not None or config.max_seconds is not None:
        raise ValueError("MAKCU calibration cannot use frame or time bounds")
    assert config.aim_calibration_evidence is not None
    if os.path.lexists(config.aim_calibration_evidence):
        raise ValueError(
            "MAKCU calibration evidence already exists; refusing to overwrite"
        )
    if not bool(getattr(sys, "frozen", False)):
        source_root = Path(__file__).resolve().parent
        evidence_path = config.aim_calibration_evidence.resolve(strict=False)
        if evidence_path == source_root or source_root in evidence_path.parents:
            raise ValueError(
                "MAKCU calibration evidence must be kept outside the source tree"
            )
    normalize_calibration_context(config.aim_calibration_context)


def _validate_active_profile_safety(config: AppConfig) -> None:
    """Reject explicit calibrated mode unless its full live boundary exists."""

    if not _active_profile_requested(config):
        return
    if not config.aim or config.aim_output != "makcu":
        raise ValueError("An active MAKCU profile requires live MAKCU aiming")
    if config.aim_calibration_evidence is not None:
        raise ValueError(
            "An active MAKCU profile cannot be combined with calibration"
        )
    if config.source.kind not in {"screen", "device"}:
        raise ValueError("An active MAKCU profile requires a live capture source")
    if config.detail_crop_size is not None:
        raise ValueError("An active MAKCU profile cannot use the detail pass")


def _update_aim_target(
    tracker,
    detections,
    frame_shape: tuple[int, ...],
    *,
    continuation_detections=(),
    continuation_allowed: bool = True,
    self_exclusion_safe: bool,
    aim_runtime_enabled: bool = True,
    prediction_grace_safe: bool = True,
    measurement_ns: int | None = None,
):
    """Select a target, dropping history whenever the physical sample is unsafe."""

    if tracker is None:
        return None
    if not aim_runtime_enabled:
        tracker.reset()
        return None
    if not self_exclusion_safe:
        tracker.reset()
        return None
    if not prediction_grace_safe:
        tracker.reset()
        return None
    return tracker.update(
        detections,
        frame_shape,
        measurement_ns=measurement_ns,
        continuation_detections=continuation_detections,
        continuation_allowed=continuation_allowed,
    )


def _aim_status(
    *,
    runtime_enabled: bool,
    self_exclusion_ready: bool,
    selected_target,
    engaged: bool,
    activation_name: str,
    control_description: str,
) -> str | None:
    """Describe live aiming only when its optional output actually started."""

    if not runtime_enabled:
        return None
    if not self_exclusion_ready and selected_target is None:
        return "aim blocked: waiting for confident self-avatar exclusion"
    if selected_target is None:
        return (
            f"aim armed: {activation_name} held, waiting for target"
            if engaged
            else "aim: no matching target"
        )
    if engaged:
        return (
            f"aim active: {activation_name} held, {control_description}"
            if not self_exclusion_ready
            else (
                f"aim active: {activation_name} held, {control_description}, "
                "tracking selected head"
            )
        )
    return f"aim ready: hold {activation_name} to track selected head"


def run(config: AppConfig) -> int:
    _validate_aim_safety(config)
    _validate_calibration_safety(config)
    _validate_active_profile_safety(config)
    calibration_requested = _calibration_requested(config)
    active_profile_requested = _active_profile_requested(config)
    active_profile = None
    calibrated_numeric_controller = None
    if active_profile_requested:
        from aiming.makcu_calibration_activation import load_active_profile

        assert config.aim_makcu_active_profile is not None
        # Explicit profile selection is not a cache lookup. Any malformed,
        # insecure, missing, or tampered file is a terminal startup error.
        active_profile = load_active_profile(config.aim_makcu_active_profile)
        calibrated_numeric_controller = _calibrated_controller_from_active_profile(
            active_profile,
            max_step=config.aim_makcu_max_step,
            vertical_rate_ratio=config.aim_makcu_vertical_rate_ratio,
        )
    if config.crop_size is not None and config.detail_crop_size is not None:
        raise ValueError(
            "The detail pass requires a full-frame primary inference; "
            "crop_size and detail_crop_size cannot both be enabled"
        )
    report_destination = None
    model_artifact_snapshot = None
    labels_artifact_snapshot = None
    if config.metrics_json is not None:
        # Reject a reused qualification filename before model startup or live
        # capture can consume minutes of the tester's time. Publication repeats
        # the check atomically to close the race between startup and shutdown.
        from utils.live_report import prepare_report_destination, snapshot_artifact

        report_destination = prepare_report_destination(config.metrics_json)
        # Bind the report to the bytes that are about to be loaded, rather than
        # hashing a mutable path only after the run has finished.
        model_artifact_snapshot = snapshot_artifact(config.model_path)
        labels_artifact_snapshot = snapshot_artifact(config.labels_path)
    elif calibration_requested or active_profile_requested:
        # Hash before detector construction so evidence is bound to the exact
        # bytes that this session is about to load.
        from utils.live_report import snapshot_artifact

        model_artifact_snapshot = snapshot_artifact(config.model_path)
        labels_artifact_snapshot = snapshot_artifact(config.labels_path)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Create a virtual environment and install requirements.txt."
        ) from exc

    from detection import (
        DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT,
        DetailPassStats,
        OpenVINOYoloDetector,
        merge_cross_pass_detections,
        plan_detail_pass,
    )
    from aiming import (
        AimActivationSensor,
        AimConfig,
        AimingController,
        AimingControllerError,
        AimActivationError,
        MakcuAimConfig,
        MakcuAimingController,
        TargetTracker,
        head_target_point,
    )
    from aiming.makcu import BUTTON_NAMES
    from utils.metrics import FrameTimings, RollingMetrics
    from utils.preview import PreviewPacer, create_preview_window
    from utils.preprocess import preprocess_frame
    from utils.render import (
        console_summary,
        draw_detections,
        draw_aim_target,
        draw_ignore_zone,
        draw_metrics,
    )
    from utils.self_filter import NormalizedBottomZone, SelfAvatarFilter

    if config.backend == "onnxruntime":
        # AMD and NVIDIA GPUs have no OpenVINO plugin, so they run the same
        # graph through ONNX Runtime.  Both backends share the decoder, so the
        # rest of the pipeline is unchanged by this choice.
        from detection.onnx_yolo import OnnxRuntimeYoloDetector

        detector_type = OnnxRuntimeYoloDetector
    else:
        detector_type = OpenVINOYoloDetector

    detector_arguments = dict(
        model_path=config.model_path,
        labels_path=config.labels_path,
        device=config.device,
        # Keep square detector-constructor integrations source-compatible;
        # rectangular shapes remain an explicit (height, width) pair.
        inference_size=compact_inference_size(config.inference_size),
        confidence=config.confidence,
        iou=config.iou_threshold,
        output_format=config.output_format,
    )
    if config.backend == "onnxruntime":
        detector_arguments["require_full_provider"] = config.require_full_provider
    detector = detector_type(**detector_arguments)
    detector.warmup()
    aim_configured_confidence = config.confidence if config.aim else None
    postprocess_options = (
        {
            "confidence": min(
                config.confidence,
                AIM_CONTINUATION_CONFIDENCE_FLOOR,
            )
        }
        if config.aim and not calibration_requested
        else {}
    )
    if report_destination is not None or calibration_requested or active_profile_requested:
        from utils.live_report import verify_artifact_unchanged

        assert model_artifact_snapshot is not None
        assert labels_artifact_snapshot is not None
        verify_artifact_unchanged(
            config.model_path,
            model_artifact_snapshot,
            description="Model artifact",
        )
        verify_artifact_unchanged(
            config.labels_path,
            labels_artifact_snapshot,
            description="Labels artifact",
        )

    source = _build_capture(config)
    metrics = RollingMetrics(config.stats_window)
    display_snapshot = metrics.snapshot()
    preview_pacer = PreviewPacer(config.preview_fps) if config.preview else None
    preview_window = create_preview_window(cv2, WINDOW_NAME) if config.preview else None
    crop_warning_printed = False
    detail_geometry_printed = False
    detail_clamp_warning_printed = False
    detail_redundant_warning_printed = False
    detail_pass_stats = DetailPassStats(config.detail_crop_size)
    last_report_ns = perf_counter_ns()
    pipeline_started_ns: int | None = None
    pipeline_completed_ns: int | None = None
    pipeline_started_utc: str | None = None
    termination_reason = "source_ended"
    processed_frames = 0
    last_source_settings: dict[str, object] = {}
    cleanup_failures: list[str] = []
    self_zone = (
        NormalizedBottomZone(
            left=config.self_zone_left,
            width=config.self_zone_width,
            height=config.self_zone_height,
        )
        if config.ignore_self
        else None
    )
    self_filter = SelfAvatarFilter(self_zone) if self_zone is not None else None
    last_ignored_count: int | None = 0 if self_zone is not None else None
    last_ignored_detection = None
    aim_controller: AimingController | MakcuAimingController | None = None
    makcu_report_snapshot = None
    makcu_report_ns: int | None = None
    tracker_report_snapshot = None
    tracker_report_ns: int | None = None
    aim_input_telemetry: AimInputTelemetry | None = None
    aim_input_report_snapshot: AimInputTelemetrySnapshot | None = None
    aim_input_report_ns: int | None = None
    aim_sensor: AimActivationSensor | None = None
    target_tracker: TargetTracker | None = None
    calibration_session = None
    calibration_status = None
    calibration_evidence_written = False
    calibration_last_log: tuple[object, str] | None = None
    active_profile_bound = False
    aim_runtime_enabled = False
    aim_activation_was_active = False
    aim_activation_name = "physical control"
    aim_control_description = "gated output"
    if config.aim:
        aim_input_telemetry = AimInputTelemetry(config.aim_label)
        if not calibration_requested:
            target_tracker = TargetTracker(
                label=config.aim_label,
                head_ratio=config.aim_head_ratio,
                # Bridge only one 60 Hz reference frame (16.7 ms). At the measured
                # ~130 Hz detector rate this covers an isolated empty result
                # without carrying physical movement through a sustained loss.
                lost_grace_frames=1,
            )
        aim_config = AimConfig(
            invert_x=config.aim_invert_x,
            invert_y=config.aim_invert_y,
            head_ratio=config.aim_head_ratio,
        )
        if config.aim_output == "makcu":
            aim_activation_name = BUTTON_NAMES[config.aim_makcu_button]
            makcu_config = MakcuAimConfig(
                port=config.aim_makcu_port or "",
                activation_button=config.aim_makcu_button,
                strength=config.aim_makcu_strength,
                max_step=config.aim_makcu_max_step,
                smoothing_alpha=config.aim_makcu_smoothing_alpha,
                prediction_lead_seconds=config.aim_makcu_prediction_lead_seconds,
                derivative_damping_seconds=config.aim_makcu_derivative_damping_seconds,
                vertical_rate_ratio=config.aim_makcu_vertical_rate_ratio,
                invert_x=config.aim_invert_x,
                invert_y=config.aim_invert_y,
                head_ratio=config.aim_head_ratio,
            )
            if active_profile is None:
                aim_controller = MakcuAimingController(makcu_config)
            else:
                assert calibrated_numeric_controller is not None
                aim_controller = MakcuAimingController(
                    makcu_config,
                    calibrated_controller=calibrated_numeric_controller,
                    expected_identity_token=(
                        active_profile.binding.makcu_identity_token
                    ),
                )
            if active_profile is None:
                aim_control_description = (
                    f"{aim_controller.config.output_hz} Hz control"
                )
            else:
                aim_control_description = (
                    "calibrated profile "
                    f"{active_profile.profile_sha256[:12]} at "
                    f"{aim_controller.config.output_hz} Hz"
                )
        else:
            aim_controller = AimingController(aim_config)
        if config.aim_output == "local":
            aim_activation_name = "LT"
            assert config.aim_activate_path is not None
            aim_sensor = AimActivationSensor(
                config.aim_activate_path,
                axis=config.aim_activate_axis,
                threshold=config.aim_activate_threshold,
            )

    _print_startup(detector, source)
    if config.detail_crop_size is not None:
        print(
            "Detail pass: enabled | same model/input | centered model-aspect "
            f"ROI up to {config.detail_crop_size} source px wide | "
            "class-aware one-to-one "
            "cross-pass deduplication | unmatched detail additions <= "
            f"{DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT:.0f}px at 1080p reference"
        )
    if self_zone is not None:
        print(
            "Self-avatar filter: enabled heuristic | player-like labels | "
            "3-frame lock/relock | max one/frame | box height >= 0.250 | "
            "box width >= 0.060 | bottom-center zone: "
            f"left {self_zone.left:.3f} | width {self_zone.width:.3f} | "
            f"height {self_zone.height:.3f}"
        )
    try:
        source.start()
        if active_profile is None:
            aim_controller, aim_sensor = _start_optional_aiming(
                aim_controller,
                aim_sensor,
            )
        else:
            # An explicitly selected profile is a strict operating mode, not
            # an optional acceleration hint.  In particular, an identity
            # rejection must stop startup instead of printing the legacy
            # fail-soft message that capture will continue.
            assert isinstance(aim_controller, MakcuAimingController)
            assert aim_sensor is None
            try:
                aim_controller.start()
            except (RuntimeError, OSError, ValueError) as exc:
                try:
                    aim_controller.stop()
                except (RuntimeError, OSError, ValueError):
                    pass
                raise RuntimeError(
                    "Calibrated MAKCU profile-bound controller failed strict "
                    f"startup: {exc}"
                ) from exc
        aim_runtime_enabled = aim_controller is not None
        if calibration_requested and not isinstance(
            aim_controller, MakcuAimingController
        ):
            raise RuntimeError(
                "MAKCU calibration cannot continue because the verified live "
                "controller did not start"
            )
        if active_profile is not None and not isinstance(
            aim_controller, MakcuAimingController
        ):
            raise RuntimeError(
                "Calibrated MAKCU aiming cannot continue because the exact "
                "profile-bound controller did not start"
            )
        if not aim_runtime_enabled:
            # A configured but unavailable output must fail closed. In
            # particular, do not retain a tracker that could still drive the
            # "aim ready" overlay or draw a selected aim point.
            target_tracker = None
            aim_input_telemetry = None
        elif config.aim_output == "makcu" and calibration_requested:
            assert isinstance(aim_controller, MakcuAimingController)
            print(
                f"MAKCU calibration: enabled | target {config.aim_label} | "
                f"activation {aim_activation_name} | exact full-pass detections | "
                "bounded exclusive pulses | no automatic profile activation"
            )
        elif config.aim_output == "makcu" and active_profile is not None:
            # Announce calibrated operation only after capture/runtime binding
            # is verified below. The output worker has no target yet and
            # therefore cannot move during this startup boundary.
            pass
        elif config.aim_output == "makcu":
            activation = (
                f"MAKCU mouse button {config.aim_makcu_button} | "
                f"control loop {aim_controller.config.output_hz} Hz"
            )
            output = f"MAKCU {config.aim_makcu_port or 'auto-detect'}"
            print(
                f"Detection-driven aim: enabled | target {config.aim_label} | "
                f"output {output} | activation {activation} | "
                f"strength {aim_controller.config.strength:g} | "
                f"max step {aim_controller.config.max_step} | "
                f"smoothing {aim_controller.config.smoothing_alpha:g} | "
                f"prediction {aim_controller.config.prediction_lead_seconds:g}s | "
                f"damping {aim_controller.config.derivative_damping_seconds:g}s | "
                f"vertical cap {aim_controller.config.vertical_rate_ratio:g}"
            )
        else:
            activation = (
                f"LT axis {config.aim_activate_axis} on {config.aim_activate_path}"
                if aim_sensor is not None
                else "always active"
            )
            print(
                f"Detection-driven aim: enabled | target {config.aim_label} | "
                f"output local uinput | activation {activation}"
            )
        print(f"Capture settings: {_format_settings(source.actual_settings)}")
        _warn_on_capture_mismatch(config, source.actual_settings)
        last_source_settings = dict(source.actual_settings)
        if active_profile is not None:
            from aiming.makcu_calibration_activation import (
                CalibrationActivationBindingError,
            )
            from utils.live_report import verify_artifact_unchanged

            assert isinstance(aim_controller, MakcuAimingController)
            assert calibrated_numeric_controller is not None
            assert model_artifact_snapshot is not None
            assert labels_artifact_snapshot is not None
            # Close the detector-construction/startup interval before allowing
            # the first normal target publication into the output worker.
            verify_artifact_unchanged(
                config.model_path,
                model_artifact_snapshot,
                description="Model artifact",
            )
            verify_artifact_unchanged(
                config.labels_path,
                labels_artifact_snapshot,
                description="Labels artifact",
            )
            runtime_binding = _build_calibration_runtime_binding(
                config,
                detector_summary=detector.runtime_summary,
                capture_settings=last_source_settings,
                makcu_identity_token=aim_controller.identity_token,
                model_artifact_snapshot=model_artifact_snapshot,
                labels_artifact_snapshot=labels_artifact_snapshot,
            )
            if active_profile.binding != runtime_binding:
                raise CalibrationActivationBindingError(
                    "Active calibration profile does not exactly match the "
                    "current runtime binding"
                )
            active_profile_bound = True
            profile_control = calibrated_numeric_controller.config
            print(
                f"Detection-driven aim: enabled | target {config.aim_label} | "
                f"output MAKCU | activation {aim_activation_name} | "
                f"control calibrated | profile {active_profile.profile_sha256[:12]} | "
                f"context {active_profile.binding.context_name} | gains X/Y "
                f"{active_profile.fit.x.gain_pixels_per_count:.6g}/"
                f"{active_profile.fit.y.gain_pixels_per_count:.6g} px/count | "
                f"delay {active_profile.fit.delay_seconds * 1000.0:.2f} ms | "
                "caps X/Y "
                f"{profile_control.maximum_rate_x_counts_per_second:.0f}/"
                f"{profile_control.maximum_rate_y_counts_per_second:.0f} counts/s"
            )
        if config.preview:
            assert preview_window is not None
            preview_behavior = (
                "latest-only Windows worker"
                if preview_window.mode == "threaded"
                else "main-thread HighGUI compatibility mode"
            )
            print(
                f"Preview: capped at {config.preview_fps:g} fps; detection and "
                f"control continue between refreshes | {preview_behavior} | "
                "service cost measured separately"
            )
            preview_window.start()

        if calibration_requested:
            from aiming.makcu_calibration_session import MakcuCalibrationSession

            assert isinstance(aim_controller, MakcuAimingController)
            assert model_artifact_snapshot is not None
            assert labels_artifact_snapshot is not None
            binding = _build_calibration_runtime_binding(
                config,
                detector_summary=detector.runtime_summary,
                capture_settings=last_source_settings,
                makcu_identity_token=aim_controller.identity_token,
                model_artifact_snapshot=model_artifact_snapshot,
                labels_artifact_snapshot=labels_artifact_snapshot,
            )
            calibration_started_ns = perf_counter_ns()
            calibration_session = MakcuCalibrationSession(
                aim_controller,
                binding,
                started_ns=calibration_started_ns,
            )
            calibration_status = calibration_session.status()
            calibration_last_log = (
                calibration_status.state,
                calibration_status.message,
            )
            print(f"MAKCU calibration: {calibration_status.message}")

        pipeline_started_ns = perf_counter_ns()
        if isinstance(aim_controller, MakcuAimingController):
            makcu_report_snapshot = aim_controller.telemetry_snapshot()
            makcu_report_ns = pipeline_started_ns
        if target_tracker is not None:
            tracker_telemetry_snapshot = getattr(
                target_tracker,
                "telemetry_snapshot",
                None,
            )
            if callable(tracker_telemetry_snapshot):
                tracker_report_snapshot = tracker_telemetry_snapshot()
                tracker_report_ns = pipeline_started_ns
        if aim_input_telemetry is not None:
            aim_input_report_snapshot = aim_input_telemetry.snapshot()
            aim_input_report_ns = pipeline_started_ns
        if report_destination is not None:
            from utils.live_report import utc_now

            pipeline_started_utc = utc_now()
        deadline_ns = (
            pipeline_started_ns + round(config.max_seconds * 1_000_000_000)
            if config.max_seconds is not None
            else None
        )
        while True:
            loop_started_ns = perf_counter_ns()
            if deadline_ns is not None and loop_started_ns >= deadline_ns:
                termination_reason = "max_seconds"
                break
            read_timeout = 0.25
            if deadline_ns is not None:
                # Keep the existing bounded read and shorten its final wait so
                # an unchanged DXGI desktop still obeys --max-seconds.
                read_timeout = min(
                    read_timeout,
                    max(0.0, (deadline_ns - loop_started_ns) / 1_000_000_000),
                )
            preview_service_ms = 0.0
            packet = source.read(timeout=read_timeout)
            read_returned_ns = perf_counter_ns()
            # A static DXGI source still returns from this bounded read every
            # 250 ms, so inline HighGUI can service Escape/window close even
            # when there is no paced preview frame to submit.
            if packet is None:
                if source.error:
                    raise RuntimeError(source.error)
                if deadline_ns is not None and read_returned_ns >= deadline_ns:
                    termination_reason = "max_seconds"
                    break
                if source.ended:
                    termination_reason = "source_ended"
                    break
                if preview_window is not None:
                    preview_service_started_ns = perf_counter_ns()
                    continue_running = preview_window.poll()
                    preview_service_ms += (
                        perf_counter_ns() - preview_service_started_ns
                    ) / 1e6
                    if not continue_running:
                        termination_reason = "preview_closed"
                        break
                continue

            # A source may return its final packet as the deadline is reached.
            # A packet already delivered by a source is still a
            # legitimate sample; max-seconds is checked again immediately
            # after it finishes, and this preserves exact max-frames behavior
            # when both bounds are supplied.

            processing_started_ns = perf_counter_ns()
            preprocessing_started_ns = processing_started_ns
            prepared = preprocess_frame(
                packet.image,
                inference_size=config.inference_size,
                crop_size=config.crop_size,
            )
            preprocessing_completed_ns = perf_counter_ns()

            inference_started_ns = perf_counter_ns()
            raw = detector.infer(prepared.tensor)
            inference_completed_ns = perf_counter_ns()
            detections = detector.postprocess(
                raw,
                transform=prepared.transform,
                frame_shape=packet.image.shape,
                **postprocess_options,
            )
            all_detections = tuple(detections)
            detections, continuation_detections = (
                _partition_detections_by_confidence(
                    all_detections,
                    aim_configured_confidence,
                )
            )
            postprocess_completed_ns = perf_counter_ns()
            detail_preprocess_ms = 0.0
            detail_inference_ms = 0.0
            detail_postprocess_ms = 0.0
            detections_ready_ns = postprocess_completed_ns
            if config.detail_crop_size is not None:
                detail_plan = plan_detail_pass(
                    packet.image.shape,
                    config.detail_crop_size,
                    config.inference_size,
                )
                detail_pass_stats.record(detail_plan)
                if not detail_geometry_printed:
                    print(
                        "Detail pass coverage: centered "
                        f"{detail_plan.applied_crop_width}x"
                        f"{detail_plan.applied_crop_height} of "
                        f"{detail_plan.source_width}x{detail_plan.source_height} "
                        f"({detail_plan.coverage_fraction * 100.0:.1f}% of frame area; "
                        f"{detail_plan.effective_linear_magnification:.2f}x "
                        "derived linear detail versus the full pass)"
                    )
                    detail_geometry_printed = True
                if detail_plan.clamped and not detail_clamp_warning_printed:
                    print(
                        "Warning: --detail-crop-size was reduced to the largest "
                        "exact model-aspect ROI that fits this source: "
                        f"{detail_plan.applied_crop_width}x"
                        f"{detail_plan.applied_crop_height}px.",
                        file=sys.stderr,
                    )
                    detail_clamp_warning_printed = True
                if detail_plan.redundant:
                    if not detail_redundant_warning_printed:
                        print(
                            "Warning: detail pass is identical to this square source; "
                            "the redundant second inference is disabled.",
                            file=sys.stderr,
                        )
                        detail_redundant_warning_printed = True
                else:
                    detail_preprocessing_started_ns = perf_counter_ns()
                    detail_prepared = preprocess_frame(
                        packet.image,
                        inference_size=config.inference_size,
                        crop_size=(
                            detail_plan.applied_crop_height,
                            detail_plan.applied_crop_width,
                        ),
                    )
                    detail_preprocessing_completed_ns = perf_counter_ns()

                    detail_inference_started_ns = detail_preprocessing_completed_ns
                    detail_raw = detector.infer(detail_prepared.tensor)
                    detail_inference_completed_ns = perf_counter_ns()
                    detail_detections = detector.postprocess(
                        detail_raw,
                        transform=detail_prepared.transform,
                        frame_shape=packet.image.shape,
                        **postprocess_options,
                    )
                    if config.aim:
                        detail_normal, _detail_continuation = (
                            _partition_detections_by_confidence(
                                detail_detections,
                                aim_configured_confidence,
                            )
                        )
                        all_detections = tuple(
                            merge_cross_pass_detections(
                                all_detections,
                                detail_detections,
                                source_height=packet.image.shape[0],
                                unmatched_detail_max_reference_height=(
                                    DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT
                                ),
                            )
                        )
                        detections = tuple(
                            merge_cross_pass_detections(
                                detections,
                                detail_normal,
                                source_height=packet.image.shape[0],
                                unmatched_detail_max_reference_height=(
                                    DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT
                                ),
                                stats=detail_pass_stats,
                            )
                        )
                        _all_normal, continuation_detections = (
                            _partition_detections_by_confidence(
                                all_detections,
                                aim_configured_confidence,
                            )
                        )
                        all_detections = tuple(detections) + tuple(
                            continuation_detections
                        )
                    else:
                        detections = tuple(
                            merge_cross_pass_detections(
                                detections,
                                detail_detections,
                                source_height=packet.image.shape[0],
                                unmatched_detail_max_reference_height=(
                                    DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT
                                ),
                                stats=detail_pass_stats,
                            )
                        )
                        all_detections = tuple(detections)
                    detections_ready_ns = perf_counter_ns()
                    detail_preprocess_ms = (
                        detail_preprocessing_completed_ns
                        - detail_preprocessing_started_ns
                    ) / 1e6
                    detail_inference_ms = (
                        detail_inference_completed_ns - detail_inference_started_ns
                    ) / 1e6
                    # Includes source-space cross-pass consolidation, the final
                    # deterministic postprocessing operation of this pass.
                    detail_postprocess_ms = (
                        detections_ready_ns - detail_inference_completed_ns
                    ) / 1e6
            if aim_input_telemetry is not None:
                aim_input_telemetry.record_sample(detections)

            self_exclusion_ready = self_filter is None
            if self_filter is not None:
                exclusion = self_filter.apply(
                    all_detections,
                    packet.image.shape,
                )
                all_detections = tuple(exclusion.detections)
                detections, continuation_detections = (
                    _partition_detections_by_confidence(
                        all_detections,
                        aim_configured_confidence,
                    )
                )
                ignored_detection = exclusion.ignored_detection
                ignored_is_display_detection = bool(
                    ignored_detection is not None
                    and (
                        aim_configured_confidence is None
                        or float(ignored_detection.confidence)
                        >= aim_configured_confidence
                    )
                )
                last_ignored_count = (
                    exclusion.ignored_count if ignored_is_display_detection else 0
                )
                last_ignored_detection = (
                    ignored_detection if ignored_is_display_detection else None
                )
                self_exclusion_ready = exclusion.aim_safe
            if aim_input_telemetry is not None:
                aim_input_telemetry.record_self_filter(
                    aim_safe=self_exclusion_ready,
                )

            # Hard self guard for aim selection: never select a likely self-avatar
            # candidate from the configured bottom zone, even if temporal lock is
            # not currently confident enough to hide it from the preview list.
            # Opposite-shoulder ambiguity is handled temporally by SelfAvatarFilter;
            # do not guess that an arbitrary large bottom opponent is self.
            aim_detections = detections
            aim_continuation_detections = continuation_detections
            hard_guard_revoked_prediction_grace = False
            if self_zone is not None and all_detections:
                hard_guard_result = _apply_hard_aim_guard(
                    all_detections,
                    packet.image.shape,
                    self_zone=self_zone,
                    aim_label=config.aim_label,
                    configured_confidence=aim_configured_confidence,
                )
                aim_detections, aim_continuation_detections = (
                    _partition_detections_by_confidence(
                        hard_guard_result.detections,
                        aim_configured_confidence,
                    )
                )
                # This was not a genuine detector-empty sample. Never bridge
                # an old physical target when an exact aim-label candidate was
                # consumed and no exact aim-label target survived.
                hard_guard_revoked_prediction_grace = (
                    hard_guard_result.targetless_after_exact_removal
                )
                if aim_input_telemetry is not None:
                    aim_input_telemetry.record_hard_guard(hard_guard_result)
            if calibration_session is not None:
                # Calibration consumes only this frame's configured-confidence,
                # exact-label, full-pass result.  It deliberately bypasses the
                # target tracker, prediction grace, continuation detections, and
                # the normal aim controller update path.
                assert isinstance(aim_controller, MakcuAimingController)
                assert aim_configured_confidence is not None
                assert config.aim_label is not None
                calibration_observation, selected_aim_target = (
                    _calibration_observation_and_target(
                        aim_detections,
                        packet.image.shape,
                        aim_label=config.aim_label,
                        head_ratio=config.aim_head_ratio,
                        configured_confidence=max(
                            aim_configured_confidence,
                            calibration_session.config.minimum_confidence,
                        ),
                        invert_x=config.aim_invert_x,
                        invert_y=config.aim_invert_y,
                        self_exclusion_safe=self_exclusion_ready,
                        measurement_ns=packet.read_started_ns,
                    )
                )
                calibration_status = calibration_session.update_from_controller(
                    perf_counter_ns(),
                    observation=calibration_observation,
                )
                raw_known, raw_pressed = aim_controller.raw_activation_state
                aim_engaged = raw_known and raw_pressed
                current_calibration_log = (
                    calibration_status.state,
                    calibration_status.message,
                )
                if current_calibration_log != calibration_last_log:
                    print(f"MAKCU calibration: {calibration_status.message}")
                    calibration_last_log = current_calibration_log
                aim_status = (
                    f"calibration {calibration_status.state.value}: "
                    f"{calibration_status.message}"
                )
            else:
                if active_profile is not None and not active_profile_bound:
                    raise RuntimeError(
                        "Calibrated MAKCU profile was not bound before target update"
                    )
                tracking_activation_active = aim_runtime_enabled
                if aim_sensor is not None:
                    tracking_activation_active = aim_sensor.read()
                elif isinstance(aim_controller, MakcuAimingController):
                    tracking_activation_active = aim_controller.activation_pressed
                activation_transition = (
                    tracking_activation_active != aim_activation_was_active
                )
                if activation_transition and target_tracker is not None:
                    # A new physical hold must establish configured-confidence
                    # provenance in this hold; never inherit a weak-only track
                    # maintained while output was inactive.
                    target_tracker.reset()
                aim_activation_was_active = tracking_activation_active

                selected_aim_target = _update_aim_target(
                    target_tracker,
                    aim_detections,
                    packet.image.shape,
                    continuation_detections=aim_continuation_detections,
                    continuation_allowed=(
                        aim_runtime_enabled and tracking_activation_active
                    ),
                    self_exclusion_safe=self_exclusion_ready,
                    aim_runtime_enabled=aim_runtime_enabled,
                    prediction_grace_safe=not hard_guard_revoked_prediction_grace,
                    measurement_ns=packet.read_started_ns,
                )
                aim_engaged = False
                if aim_controller is not None:
                    active = (
                        tracking_activation_active if aim_sensor is not None else True
                    )
                    aim_controller.update(
                        selected_aim_target,
                        packet.image.shape,
                        active=active,
                        **(
                            {
                                "measurement_ns": packet.read_started_ns,
                                "measurement_observed": not (
                                    target_tracker is not None
                                    and target_tracker.output_is_prediction
                                ),
                            }
                            if isinstance(aim_controller, MakcuAimingController)
                            else {}
                        ),
                    )
                    aim_engaged = (
                        aim_controller.activation_pressed
                        if isinstance(aim_controller, MakcuAimingController)
                        else active
                    )
                aim_status = _aim_status(
                    runtime_enabled=aim_runtime_enabled,
                    self_exclusion_ready=self_exclusion_ready,
                    selected_target=selected_aim_target,
                    engaged=aim_engaged,
                    activation_name=aim_activation_name,
                    control_description=aim_control_description,
                )
            result_ready_ns = perf_counter_ns()

            if prepared.crop_was_clamped and not crop_warning_printed:
                print(
                    "Warning: --crop-size exceeded the source dimensions and was "
                    "clamped to the largest centered square.",
                    file=sys.stderr,
                )
                crop_warning_printed = True

            skipped_frames = source.stats.frames_overwritten
            render_preview = bool(
                preview_pacer is not None
                and preview_pacer.should_render(result_ready_ns)
            )
            draw_started_ns = result_ready_ns
            if config.draw and render_preview:
                # The overlay intentionally shows the completed prior sample so
                # its own drawing cost can be measured in the current sample.
                if self_zone is not None:
                    assert last_ignored_count is not None
                    draw_ignore_zone(
                        packet.image,
                        self_zone,
                        last_ignored_count,
                        last_ignored_detection,
                    )
                draw_detections(packet.image, detections)
                if aim_runtime_enabled and selected_aim_target is not None:
                    draw_aim_target(
                        packet.image,
                        head_target_point(selected_aim_target, config.aim_head_ratio),
                        active=aim_engaged,
                        activation_name=aim_activation_name,
                    )
                draw_metrics(
                    packet.image,
                    display_snapshot,
                    skipped_frames,
                    ignored_count=last_ignored_count,
                    aim_status=aim_status,
                )
            draw_completed_ns = perf_counter_ns()

            continue_running = True
            if render_preview:
                assert preview_window is not None
                preview_service_started_ns = perf_counter_ns()
                continue_running = preview_window.submit(packet.image)
                preview_service_ms = (
                    perf_counter_ns() - preview_service_started_ns
                ) / 1e6
            elif preview_window is not None:
                # Inline HighGUI is serviced by submit at preview_fps; polling
                # it on every faster inference frame costs milliseconds on Qt.
                # The threaded Windows implementation polls independently.
                continue_running = preview_window.should_continue()

            timings = FrameTimings(
                capture_ms=(packet.read_completed_ns - packet.read_started_ns) / 1e6,
                queue_age_ms=max(0, processing_started_ns - packet.read_completed_ns) / 1e6,
                preprocess_ms=(preprocessing_completed_ns - preprocessing_started_ns) / 1e6,
                inference_ms=(inference_completed_ns - inference_started_ns) / 1e6,
                postprocess_ms=(postprocess_completed_ns - inference_completed_ns) / 1e6,
                detail_preprocess_ms=detail_preprocess_ms,
                detail_inference_ms=detail_inference_ms,
                detail_postprocess_ms=detail_postprocess_ms,
                control_ms=(result_ready_ns - detections_ready_ns) / 1e6,
                processing_ms=(result_ready_ns - processing_started_ns) / 1e6,
                freshness_latency_ms=max(0, result_ready_ns - packet.read_completed_ns) / 1e6,
                observed_pipeline_ms=max(0, result_ready_ns - packet.read_started_ns) / 1e6,
                draw_ms=(draw_completed_ns - draw_started_ns) / 1e6,
                # Threaded Windows mode measures owned copy/mailbox submission;
                # inline compatibility mode measures HighGUI submission/event
                # service. Neither is a measurement of display scanout.
                preview_service_ms=preview_service_ms,
            )
            metrics.record(timings, result_ready_ns)
            processed_frames += 1

            if result_ready_ns - last_report_ns >= 1_000_000_000:
                display_snapshot = metrics.snapshot()
                summary = console_summary(
                    display_snapshot,
                    skipped_frames,
                    ignored_count=last_ignored_count,
                )
                if (
                    target_tracker is not None
                    and tracker_report_snapshot is not None
                    and tracker_report_ns is not None
                ):
                    current_tracker_telemetry = target_tracker.telemetry_snapshot()
                    summary += " | " + _target_tracker_telemetry_summary(
                        tracker_report_snapshot,
                        current_tracker_telemetry,
                        (result_ready_ns - tracker_report_ns) / 1_000_000_000,
                    )
                    tracker_report_snapshot = current_tracker_telemetry
                    tracker_report_ns = result_ready_ns
                if (
                    aim_input_telemetry is not None
                    and aim_input_report_snapshot is not None
                    and aim_input_report_ns is not None
                ):
                    current_aim_input = aim_input_telemetry.snapshot()
                    summary += " | " + _aim_input_telemetry_summary(
                        aim_input_report_snapshot,
                        current_aim_input,
                        (result_ready_ns - aim_input_report_ns) / 1_000_000_000,
                    )
                    aim_input_report_snapshot = current_aim_input
                    aim_input_report_ns = result_ready_ns
                if (
                    isinstance(aim_controller, MakcuAimingController)
                    and makcu_report_snapshot is not None
                    and makcu_report_ns is not None
                ):
                    current_telemetry = aim_controller.telemetry_snapshot()
                    telemetry_snapshot_ns = perf_counter_ns()
                    summary += " | " + _makcu_telemetry_summary(
                        makcu_report_snapshot,
                        current_telemetry,
                        (telemetry_snapshot_ns - makcu_report_ns) / 1_000_000_000,
                    )
                    makcu_report_snapshot = current_telemetry
                    makcu_report_ns = telemetry_snapshot_ns
                if calibration_status is not None:
                    summary += (
                        f" | CAL {calibration_status.state.value} | "
                        f"counts {calibration_status.emitted_abs_counts}/2400 | "
                        "qualifying X +/- "
                        f"{calibration_status.qualifying_x_positive}/"
                        f"{calibration_status.qualifying_x_negative} | Y +/- "
                        f"{calibration_status.qualifying_y_positive}/"
                        f"{calibration_status.qualifying_y_negative}"
                    )
                print(summary)
                last_report_ns = result_ready_ns
            if calibration_status is not None and calibration_status.terminal:
                assert calibration_session is not None
                assert calibration_session.result is not None
                termination_reason = (
                    "aim_calibration_success"
                    if calibration_session.result.outcome == "success"
                    else "aim_calibration_aborted"
                )
                break
            if (
                config.max_frames is not None
                and processed_frames >= config.max_frames
            ):
                termination_reason = "max_frames"
                break
            if deadline_ns is not None and perf_counter_ns() >= deadline_ns:
                termination_reason = "max_seconds"
                break
            if not continue_running:
                termination_reason = "preview_closed"
                break
        pipeline_completed_ns = perf_counter_ns()
    finally:
        primary_exception_active = sys.exc_info()[0] is not None

        def record_cleanup_failure(component: str, detail: object) -> None:
            message = f"{component}: {detail}"
            cleanup_failures.append(message)
            if primary_exception_active:
                # Preserve the original detector/capture exception while still
                # exposing every failed cleanup operation to the operator.
                print(f"Warning: cleanup also failed: {message}", file=sys.stderr)

        if calibration_session is not None:
            if not calibration_session.terminal:
                try:
                    calibration_status = calibration_session.abort(
                        "pipeline stopped before calibration completed",
                        now_ns=perf_counter_ns(),
                    )
                except Exception as exc:  # noqa: BLE001 - still stop physical output
                    record_cleanup_failure("calibration abort", exc)
            result = calibration_session.result
            if result is None:
                record_cleanup_failure(
                    "calibration evidence",
                    "the session produced no terminal result",
                )
            elif not calibration_evidence_written:
                try:
                    from aiming.makcu_calibration_session import (
                        write_session_evidence_exclusive,
                    )

                    assert config.aim_calibration_evidence is not None
                    write_session_evidence_exclusive(
                        config.aim_calibration_evidence,
                        result.evidence,
                    )
                    calibration_evidence_written = True
                except Exception as exc:  # noqa: BLE001 - still stop physical output
                    record_cleanup_failure("calibration evidence publication", exc)

        if aim_controller is not None:
            try:
                aim_controller.stop()
            except Exception as exc:  # noqa: BLE001 - aggregate bounded cleanup failures
                record_cleanup_failure("aim output shutdown", exc)
        if aim_sensor is not None:
            try:
                aim_sensor.stop()
            except Exception as exc:  # noqa: BLE001 - aggregate bounded cleanup failures
                record_cleanup_failure("activation sensor shutdown", exc)
        source_error_before_close = source.error
        try:
            source.close()
        except Exception as exc:  # noqa: BLE001 - still attempt preview shutdown
            record_cleanup_failure("capture shutdown", exc)
        else:
            if source.error and (
                not primary_exception_active or source.error != source_error_before_close
            ):
                record_cleanup_failure("capture shutdown", source.error)
        if preview_window is not None:
            try:
                preview_stopped = preview_window.stop()
            except Exception as exc:  # noqa: BLE001 - report as qualification failure
                record_cleanup_failure("preview shutdown", exc)
            else:
                if not preview_stopped:
                    record_cleanup_failure(
                        "preview shutdown",
                        "the OpenCV preview worker did not stop within its bounded timeout",
                    )
                try:
                    preview_window.raise_if_failed()
                except Exception as exc:  # noqa: BLE001 - report worker failure after cleanup
                    record_cleanup_failure("preview worker", exc)

    if cleanup_failures:
        raise RuntimeError("Pipeline cleanup failed: " + "; ".join(cleanup_failures))

    if report_destination is not None or active_profile is not None:
        from utils.live_report import verify_artifact_unchanged

        assert model_artifact_snapshot is not None
        assert labels_artifact_snapshot is not None
        verify_artifact_unchanged(
            config.model_path,
            model_artifact_snapshot,
            description="Model artifact",
        )
        verify_artifact_unchanged(
            config.labels_path,
            labels_artifact_snapshot,
            description="Labels artifact",
        )

    final = metrics.snapshot()
    capture_stats = source.stats
    if final.processed_frames:
        print(
            console_summary(
                final,
                capture_stats.frames_overwritten,
                ignored_count=last_ignored_count,
            )
        )
    print(
        f"Stopped after {final.processed_frames} processed frame(s); "
        f"{capture_stats.frames_overwritten} application overwrite(s), "
        f"{capture_stats.read_failures} capture failure(s)."
    )
    if report_destination is not None:
        from utils.live_report import build_live_report, utc_now, write_json_atomic_new

        if pipeline_started_ns is None:
            # Source/preview startup errors propagate before this point, so the
            # branch is defensive for custom CaptureSource implementations.
            elapsed_seconds = 0.0
        else:
            completed_ns = pipeline_completed_ns or perf_counter_ns()
            elapsed_seconds = max(0.0, completed_ns - pipeline_started_ns) / 1e9
        preview_mode = preview_window.mode if preview_window is not None else "disabled"
        preview_stats = preview_window.stats if preview_window is not None else {}
        report = build_live_report(
            config=config,
            detector_summary=detector.runtime_summary,
            source_description=source.description,
            source_settings=last_source_settings or source.actual_settings,
            capture_stats=capture_stats,
            preview_mode=preview_mode,
            preview_stats=preview_stats,
            metrics=final,
            elapsed_seconds=elapsed_seconds,
            started_utc=pipeline_started_utc or utc_now(),
            completed_utc=utc_now(),
            termination_reason=termination_reason,
            detail_pass_stats=detail_pass_stats.snapshot(),
            model_artifact_snapshot=model_artifact_snapshot,
            labels_artifact_snapshot=labels_artifact_snapshot,
        )
        written = write_json_atomic_new(report_destination, report)
        print(f"Live pipeline metrics written to {written}")
    if calibration_requested:
        assert calibration_session is not None
        assert calibration_session.result is not None
        assert calibration_evidence_written
        calibration_result = calibration_session.result
        assert config.aim_calibration_evidence is not None
        print(
            "MAKCU calibration evidence written: "
            f"{config.aim_calibration_evidence} | "
            f"outcome {calibration_result.outcome} | "
            f"artifact {calibration_result.evidence.artifact_sha256}"
        )
        if calibration_result.outcome != "success":
            raise RuntimeError(
                f"MAKCU calibration aborted: {calibration_result.reason}"
            )
        assert calibration_result.fit is not None
        print(
            "MAKCU calibration fit passed: "
            f"X {calibration_result.fit.x.gain_pixels_per_count:.6g} px/count | "
            f"Y {calibration_result.fit.y.gain_pixels_per_count:.6g} px/count | "
            f"delay {calibration_result.fit.delay_seconds * 1000.0:.2f} ms | "
            "profile remains inactive pending explicit review"
        )
    return 0


def main() -> int:
    config = parse_args()
    try:
        return run(config)
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
