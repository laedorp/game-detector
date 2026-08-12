from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter_ns

from config import AppConfig, parse_args


WINDOW_NAME = "Game Detector"


def _build_capture(config: AppConfig):
    from capture import OpenCVCaptureSource, ScreenCaptureSource

    if config.source.kind == "screen":
        return ScreenCaptureSource(
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


def _show_frame(cv2, frame) -> bool:
    try:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:
            # Some Linux GUI backends do not implement WND_PROP_VISIBLE.
            pass
        return True
    except cv2.error as exc:
        raise RuntimeError(
            f"OpenCV preview failed: {exc}. Use --no-preview on a headless session."
        ) from exc


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


def run(config: AppConfig) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Create a virtual environment and install requirements.txt."
        ) from exc

    from detection import OpenVINOYoloDetector
    from aiming import (
        AimActivationSensor,
        AimConfig,
        AimingController,
        AimingControllerError,
        AimActivationError,
        MakcuAimConfig,
        MakcuAimingController,
        TargetTracker,
        UdpAimingController,
        choose_target,
        head_target_point,
    )
    from utils.metrics import FrameTimings, RollingMetrics
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

    detector = detector_type(
        model_path=config.model_path,
        labels_path=config.labels_path,
        device=config.device,
        inference_size=config.inference_size,
        confidence=config.confidence,
        iou=config.iou_threshold,
        output_format=config.output_format,
    )
    detector.warmup()

    source = _build_capture(config)
    metrics = RollingMetrics(config.stats_window)
    crop_warning_printed = False
    last_report_ns = perf_counter_ns()
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
    aim_controller: AimingController | UdpAimingController | MakcuAimingController | None = None
    aim_sensor: AimActivationSensor | None = None
    target_tracker: TargetTracker | None = None
    if config.aim:
        target_tracker = TargetTracker(
            label=config.aim_label,
            head_ratio=config.aim_head_ratio,
            lost_grace_frames=18,
        )
        aim_config = AimConfig(
            invert_x=config.aim_invert_x,
            invert_y=config.aim_invert_y,
            head_ratio=config.aim_head_ratio,
        )
        if config.aim_output == "remote":
            assert config.aim_host is not None
            assert config.aim_pairing_key is not None
            aim_controller = UdpAimingController(
                config.aim_host,
                config.aim_port,
                config.aim_pairing_key,
                aim_config,
            )
        elif config.aim_output == "makcu":
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
        else:
            aim_controller = AimingController(aim_config)
        if config.aim_output == "local" and config.aim_activate_path:
            aim_sensor = AimActivationSensor(
                config.aim_activate_path,
                axis=config.aim_activate_axis,
                threshold=config.aim_activate_threshold,
            )

    _print_startup(detector, source)
    if aim_controller is not None:
        if config.aim_output == "remote":
            activation = "gaming-PC receiver LT"
            output = f"remote {config.aim_host}:{config.aim_port}"
        elif config.aim_output == "makcu":
            activation = (
                f"MAKCU mouse button {config.aim_makcu_button} | "
                f"control loop {aim_controller.config.output_hz} Hz"
            )
            output = f"MAKCU {config.aim_makcu_port or 'auto-detect'}"
        else:
            activation = (
                f"LT axis {config.aim_activate_axis} on {config.aim_activate_path}"
                if aim_sensor is not None
                else "always active"
            )
            output = "local uinput"
        target = config.aim_label or "highest-confidence detection"
        print(
            f"Detection-driven aim: enabled | target {target} | "
            f"output {output} | activation {activation}"
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
        print(f"Capture settings: {_format_settings(source.actual_settings)}")
        _warn_on_capture_mismatch(config, source.actual_settings)
        if config.preview:
            try:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            except cv2.error as exc:
                raise RuntimeError(
                    f"Could not create the preview window: {exc}. Use --no-preview."
                ) from exc

        while True:
            packet = source.read(timeout=0.25)
            if packet is None:
                if source.error:
                    raise RuntimeError(source.error)
                if source.ended:
                    break
                continue

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
            frame_counter = getattr(main, "_diag_frame_counter", 0) + 1
            self_exclusion_ready = self_filter is None
            if self_filter is not None:
                raw_count = len(detections)
                exclusion = self_filter.apply(
                    detections,
                    packet.image.shape,
                )
                detections = exclusion.detections
                last_ignored_count = exclusion.ignored_count
                last_ignored_detection = exclusion.ignored_detection
                self_exclusion_ready = exclusion.aim_safe
                # Diagnostic: log filter state every ~30 frames
                if frame_counter % 30 == 0:
                    filtered_count = len(detections)
                    ignored_msg = ""
                    if last_ignored_detection:
                        x1, y1, x2, y2 = last_ignored_detection.box
                        h = y2 - y1
                        w = x2 - x1
                        ignored_msg = (
                            " | currently_locked: "
                            f"{last_ignored_detection.label} "
                            f"h={int(h)} w={int(w)} conf={last_ignored_detection.confidence:.2f}"
                        )
                    print(f"[FILTER] raw={raw_count} after_filter={filtered_count} aim_safe={self_exclusion_ready}{ignored_msg}", flush=True)

            # Hard self guard for aim selection: never select a likely self-avatar
            # candidate from the configured bottom zone, even if temporal lock is
            # not currently confident enough to hide it from the preview list.
            aim_detections = detections
            self_guard_dropped = 0
            if self_zone is not None and detections:
                guarded: list = []
                frame_h, frame_w = packet.image.shape[:2]
                fallback_candidates: list[tuple[float, object]] = []
                for detection in detections:
                    drop_for_aim = (
                        is_player_like(detection)
                        and self_zone.candidate_score(detection.box, packet.image.shape)
                        is not None
                    )
                    if not drop_for_aim and is_player_like(detection) and frame_h > 0 and frame_w > 0:
                        x1, y1, x2, y2 = detection.box
                        width_ratio = max(0.0, (x2 - x1) / frame_w)
                        height_ratio = max(0.0, (y2 - y1) / frame_h)
                        bottom_ratio = y2 / frame_h
                        if (
                            bottom_ratio >= 0.92
                            and height_ratio >= 0.40
                            and width_ratio >= 0.10
                        ):
                            # Fallback self candidate when the user-picked zone
                            # is imperfect for current framing.
                            fallback_candidates.append((bottom_ratio + height_ratio, detection))
                    if drop_for_aim:
                        self_guard_dropped += 1
                        continue
                    guarded.append(detection)
                if fallback_candidates and guarded:
                    fallback_target = max(fallback_candidates, key=lambda item: item[0])[1]
                    guarded = [item for item in guarded if item is not fallback_target]
                    self_guard_dropped += 1
                aim_detections = tuple(guarded)
                if frame_counter % 30 == 0 and self_guard_dropped:
                    print(
                        f"[AIM_GUARD] dropped_self_like={self_guard_dropped}",
                        flush=True,
                    )
            main._diag_frame_counter = frame_counter
            if target_tracker is not None and self_exclusion_ready:
                selected_aim_target = target_tracker.update(
                    aim_detections,
                    packet.image.shape,
                )
            else:
                selected_aim_target = (
                    target_tracker.update((), packet.image.shape)
                    if target_tracker is not None
                    else None
                )
                # Diagnostic: log when filter blocks aim but tracker still returns a target
                if target_tracker is not None and frame_counter % 30 == 0:
                    print(f"[AIM] aim_safe={self_exclusion_ready} selected_target={selected_aim_target is not None}", flush=True)
            aim_engaged = False
            if aim_controller is not None:
                active = True
                if aim_sensor is not None:
                    active = aim_sensor.read()
                aim_controller.update(
                    selected_aim_target,
                    packet.image.shape,
                    active=active,
                )
                aim_engaged = (
                    aim_controller.activation_pressed
                    if isinstance(aim_controller, MakcuAimingController)
                    else active
                )
            if config.aim and not self_exclusion_ready and selected_aim_target is None:
                aim_status = "aim blocked: waiting for confident self-avatar exclusion"
            elif selected_aim_target is None:
                aim_status = (
                    "aim armed: Right held, waiting for target"
                    if aim_engaged
                    else "aim: no matching target"
                )
            elif aim_engaged:
                aim_status = (
                    "aim active: Right held, 1000 Hz control, target grace"
                    if not self_exclusion_ready
                    else "aim active: Right held, 1000 Hz control, tracking selected head"
                )
            else:
                aim_status = "aim ready: hold Right to track selected head"
            result_ready_ns = perf_counter_ns()

            if prepared.crop_was_clamped and not crop_warning_printed:
                print(
                    "Warning: --crop-size exceeded the source dimensions and was "
                    "clamped to the largest centered square.",
                    file=sys.stderr,
                )
                crop_warning_printed = True

            skipped_frames = source.stats.frames_overwritten
            draw_started_ns = result_ready_ns
            if config.draw:
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
                if selected_aim_target is not None:
                    draw_aim_target(
                        packet.image,
                        head_target_point(selected_aim_target, config.aim_head_ratio),
                        active=aim_engaged,
                    )
                draw_metrics(
                    packet.image,
                    metrics.snapshot(),
                    skipped_frames,
                    ignored_count=last_ignored_count,
                    aim_status=aim_status if config.aim else None,
                )
            draw_completed_ns = perf_counter_ns()

            continue_running = True
            if config.preview:
                continue_running = _show_frame(cv2, packet.image)
            display_completed_ns = perf_counter_ns()

            timings = FrameTimings(
                capture_ms=(packet.read_completed_ns - packet.read_started_ns) / 1e6,
                queue_age_ms=max(0, processing_started_ns - packet.read_completed_ns) / 1e6,
                preprocess_ms=(preprocessing_completed_ns - preprocessing_started_ns) / 1e6,
                inference_ms=(inference_completed_ns - inference_started_ns) / 1e6,
                postprocess_ms=(result_ready_ns - inference_completed_ns) / 1e6,
                processing_ms=(result_ready_ns - processing_started_ns) / 1e6,
                freshness_latency_ms=max(0, result_ready_ns - packet.read_completed_ns) / 1e6,
                observed_pipeline_ms=max(0, result_ready_ns - packet.read_started_ns) / 1e6,
                draw_ms=(draw_completed_ns - draw_started_ns) / 1e6,
                display_ms=(display_completed_ns - draw_completed_ns) / 1e6,
            )
            snapshot = metrics.record(timings, result_ready_ns)

            if result_ready_ns - last_report_ns >= 1_000_000_000:
                print(
                    console_summary(
                        snapshot,
                        skipped_frames,
                        ignored_count=last_ignored_count,
                    )
                )
                last_report_ns = result_ready_ns
            if not continue_running:
                break
    finally:
        if aim_controller is not None:
            aim_controller.stop()
        if aim_sensor is not None:
            aim_sensor.stop()
        source.close()
        if config.preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

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
