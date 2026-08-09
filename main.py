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
        )

    assert isinstance(config.source.value, Path)
    if not config.source.value.is_file():
        raise FileNotFoundError(f"Video file not found: {config.source.value}")
    return OpenCVCaptureSource(config.source.value)


def _print_startup(detector, source) -> None:
    summary = detector.runtime_summary
    print(f"OpenVINO {summary.get('openvino_version', 'unknown')}")
    devices = ", ".join(detector.available_devices) or "none"
    print(f"Detected OpenVINO devices: {devices}")
    print(
        f"Inference: {summary.get('device')} | input {summary.get('input_shape')} | "
        f"hint {summary.get('performance_hint')} | one synchronous request"
    )
    print(f"Model: {summary.get('model_path')}")
    print(f"Source: {source.description}")


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


def run(config: AppConfig) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Create a virtual environment and install requirements.txt."
        ) from exc

    from detection import OpenVINOYoloDetector
    from utils.metrics import FrameTimings, RollingMetrics
    from utils.preprocess import preprocess_frame
    from utils.render import (
        console_summary,
        draw_detections,
        draw_ignore_zone,
        draw_metrics,
    )
    from utils.self_filter import NormalizedBottomZone, SelfAvatarFilter

    detector = OpenVINOYoloDetector(
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

    _print_startup(detector, source)
    if self_zone is not None:
        print(
            "Self-avatar filter: enabled heuristic | player-like labels | "
            "3-frame lock/relock | max one/frame | box height >= 0.280 | "
            "box width >= 0.060 | bottom-center zone: "
            f"left {self_zone.left:.3f} | width {self_zone.width:.3f} | "
            f"height {self_zone.height:.3f}"
        )
    try:
        source.start()
        print(f"Capture settings: {_format_settings(source.actual_settings)}")
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
            if self_filter is not None:
                exclusion = self_filter.apply(
                    detections,
                    packet.image.shape,
                )
                detections = exclusion.detections
                last_ignored_count = exclusion.ignored_count
                last_ignored_detection = exclusion.ignored_detection
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
                draw_metrics(
                    packet.image,
                    metrics.snapshot(),
                    skipped_frames,
                    ignored_count=last_ignored_count,
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
