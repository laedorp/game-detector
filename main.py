from __future__ import annotations

import math
import sys
from pathlib import Path
from time import perf_counter_ns

from config import AppConfig, parse_args
from utils.inference_size import compact_inference_size


WINDOW_NAME = "ProAim"


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
            buffer_size=1,
            pixel_format=config.capture_format,
        )

    assert isinstance(config.source.value, Path)
    if not config.source.value.is_file():
        raise FileNotFoundError(f"Video file not found: {config.source.value}")
    return OpenCVCaptureSource(config.source.value)


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


def _update_aim_target(
    tracker,
    detections,
    frame_shape: tuple[int, ...],
    *,
    self_exclusion_safe: bool,
    aim_runtime_enabled: bool = True,
    measurement_ns: int | None = None,
):
    """Select a target, dropping all tracker history when self filtering is unsafe."""

    if tracker is None:
        return None
    if not aim_runtime_enabled:
        tracker.reset()
        return None
    if not self_exclusion_safe:
        tracker.reset()
        return None
    return tracker.update(
        detections,
        frame_shape,
        measurement_ns=measurement_ns,
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
    from utils.self_filter import NormalizedBottomZone, SelfAvatarFilter, is_player_like

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
    if report_destination is not None:
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
    aim_sensor: AimActivationSensor | None = None
    target_tracker: TargetTracker | None = None
    aim_runtime_enabled = False
    aim_activation_name = "physical control"
    aim_control_description = "gated output"
    if config.aim:
        target_tracker = TargetTracker(
            label=config.aim_label,
            head_ratio=config.aim_head_ratio,
            # Output must fail closed on the first missed measurement.  The
            # general tracker supports prediction grace for visualization,
            # but carrying a predicted target into a physical output path can
            # keep stale movement alive when inference drops a frame.
            lost_grace_frames=0,
        )
        aim_config = AimConfig(
            invert_x=config.aim_invert_x,
            invert_y=config.aim_invert_y,
            head_ratio=config.aim_head_ratio,
        )
        if config.aim_output == "makcu":
            aim_activation_name = BUTTON_NAMES[config.aim_makcu_button]
            aim_controller = MakcuAimingController(
                MakcuAimConfig(
                    port=config.aim_makcu_port or "",
                    activation_button=config.aim_makcu_button,
                    strength=config.aim_makcu_strength,
                    max_step=config.aim_makcu_max_step,
                    smoothing_alpha=config.aim_makcu_smoothing_alpha,
                    prediction_lead_seconds=config.aim_makcu_prediction_lead_seconds,
                    derivative_damping_seconds=config.aim_makcu_derivative_damping_seconds,
                    invert_x=config.aim_invert_x,
                    invert_y=config.aim_invert_y,
                    head_ratio=config.aim_head_ratio,
                )
            )
            aim_control_description = f"{aim_controller.config.output_hz} Hz control"
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
        aim_controller, aim_sensor = _start_optional_aiming(
            aim_controller,
            aim_sensor,
        )
        aim_runtime_enabled = aim_controller is not None
        if not aim_runtime_enabled:
            # A configured but unavailable output must fail closed. In
            # particular, do not retain a tracker that could still drive the
            # "aim ready" overlay or draw a selected aim point.
            target_tracker = None
        elif config.aim_output == "makcu":
            activation = (
                f"MAKCU mouse button {config.aim_makcu_button} | "
                f"control loop {aim_controller.config.output_hz} Hz"
            )
            output = f"MAKCU {config.aim_makcu_port or 'auto-detect'}"
            print(
                f"Detection-driven aim: enabled | target {config.aim_label} | "
                f"output {output} | activation {activation}"
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

        pipeline_started_ns = perf_counter_ns()
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
                    )
                    detections = merge_cross_pass_detections(
                        detections,
                        detail_detections,
                        source_height=packet.image.shape[0],
                        unmatched_detail_max_reference_height=(
                            DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT
                        ),
                        stats=detail_pass_stats,
                    )
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
            self_exclusion_ready = self_filter is None
            if self_filter is not None:
                exclusion = self_filter.apply(
                    detections,
                    packet.image.shape,
                )
                detections = exclusion.detections
                last_ignored_count = exclusion.ignored_count
                last_ignored_detection = exclusion.ignored_detection
                self_exclusion_ready = exclusion.aim_safe

            # Hard self guard for aim selection: never select a likely self-avatar
            # candidate from the configured bottom zone, even if temporal lock is
            # not currently confident enough to hide it from the preview list.
            # Opposite-shoulder ambiguity is handled temporally by SelfAvatarFilter;
            # do not guess that an arbitrary large bottom opponent is self.
            aim_detections = detections
            if self_zone is not None and detections:
                guarded: list = []
                for detection in detections:
                    drop_for_aim = (
                        is_player_like(detection)
                        and self_zone.candidate_score(detection.box, packet.image.shape)
                        is not None
                    )
                    if drop_for_aim:
                        continue
                    guarded.append(detection)
                aim_detections = tuple(guarded)
            selected_aim_target = _update_aim_target(
                target_tracker,
                aim_detections,
                packet.image.shape,
                self_exclusion_safe=self_exclusion_ready,
                aim_runtime_enabled=aim_runtime_enabled,
                measurement_ns=packet.read_started_ns,
            )
            aim_engaged = False
            if aim_controller is not None:
                active = True
                if aim_sensor is not None:
                    active = aim_sensor.read()
                aim_controller.update(
                    selected_aim_target,
                    packet.image.shape,
                    active=active,
                    **(
                        {"measurement_ns": packet.read_started_ns}
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
                print(
                    console_summary(
                        display_snapshot,
                        skipped_frames,
                        ignored_count=last_ignored_count,
                    )
                )
                last_report_ns = result_ready_ns
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

    if report_destination is not None:
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
