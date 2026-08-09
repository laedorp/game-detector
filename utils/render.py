from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from utils.metrics import MetricsSnapshot
from utils.self_filter import NormalizedBottomZone


def _color(class_id: int) -> tuple[int, int, int]:
    return (
        80 + (class_id * 67) % 176,
        80 + (class_id * 131) % 176,
        80 + (class_id * 29) % 176,
    )


def draw_detections(frame: np.ndarray, detections: Iterable[Any]) -> None:
    import cv2

    line_width = max(1, round(min(frame.shape[:2]) / 500))
    font_scale = max(0.45, min(frame.shape[:2]) / 1400)
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.box)
        color = _color(detection.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
        text = f"{detection.label} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        text_y = max(text_height + baseline + 2, y1)
        cv2.rectangle(
            frame,
            (x1, text_y - text_height - baseline - 4),
            (x1 + text_width + 4, text_y),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 2, text_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 15, 15),
            line_width,
            cv2.LINE_AA,
        )


def draw_ignore_zone(
    frame: np.ndarray,
    zone: NormalizedBottomZone,
    ignored_count: int,
    ignored_detection: Any | None = None,
) -> None:
    """Draw the active zone and identify the exact box hidden this frame."""

    import cv2

    x1, y1, x2, y2 = zone.pixel_bounds(frame.shape)
    color = (40, 150, 255)
    region = frame[y1 : y2 + 1, x1 : x2 + 1]
    tint = region.copy()
    tint[...] = color
    cv2.addWeighted(tint, 0.12, region, 0.88, 0.0, region)
    line_width = max(1, round(min(frame.shape[:2]) / 450))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
    font_scale = max(0.45, min(frame.shape[:2]) / 1400)
    label = f"SELF ANCHOR ZONE  ignored {ignored_count}"
    text_y = min(y2 - 4, y1 + max(18, round(25 * font_scale / 0.6)))
    cv2.putText(
        frame,
        label,
        (x1 + 6, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        max(1, line_width),
        cv2.LINE_AA,
    )
    if ignored_detection is None:
        return

    ignored_box = getattr(ignored_detection, "box", None)
    if ignored_box is None:
        ignored_box = getattr(ignored_detection, "xyxy", None)
    if ignored_box is None:
        return
    box_x1, box_y1, box_x2, box_y2 = (
        int(round(float(value))) for value in ignored_box
    )
    box_x1 = min(frame.shape[1] - 1, max(0, box_x1))
    box_x2 = min(frame.shape[1] - 1, max(0, box_x2))
    box_y1 = min(frame.shape[0] - 1, max(0, box_y1))
    box_y2 = min(frame.shape[0] - 1, max(0, box_y2))
    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x2, box_y2),
        color,
        line_width,
        cv2.LINE_AA,
    )
    ignored_label = "IGNORED SELF?"
    ignored_text_y = max(16, box_y1 - 6)
    cv2.putText(
        frame,
        ignored_label,
        (box_x1 + 3, ignored_text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        max(1, line_width),
        cv2.LINE_AA,
    )


def draw_metrics(
    frame: np.ndarray,
    snapshot: MetricsSnapshot,
    skipped_frames: int,
    ignored_count: int | None = None,
) -> None:
    import cv2

    avg = snapshot.average
    latest = snapshot.latest
    lines = [
        f"FPS {snapshot.moving_fps:5.1f}  processed {snapshot.processed_frames}  skipped {skipped_frames}",
        f"capture {latest.capture_ms:5.1f}  pre {latest.preprocess_ms:5.1f}  infer {latest.inference_ms:5.1f} ms",
        f"post {latest.postprocess_ms:5.1f}  pipeline {latest.observed_pipeline_ms:5.1f} ms",
        f"draw {latest.draw_ms:5.1f}  display {latest.display_ms:5.1f} ms",
        f"moving: infer {avg.inference_ms:5.1f}  pipeline {avg.observed_pipeline_ms:5.1f} ms",
    ]
    if ignored_count is not None:
        lines.append(f"self-avatar filter: ignored {ignored_count} this frame")
    scale = max(0.48, min(frame.shape[:2]) / 1300)
    line_height = max(18, int(round(25 * scale / 0.6)))
    overlay_width = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for line in lines
    )
    cv2.rectangle(frame, (5, 5), (overlay_width + 17, 14 + line_height * len(lines)), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (11, 5 + line_height * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def console_summary(
    snapshot: MetricsSnapshot,
    skipped_frames: int,
    ignored_count: int | None = None,
) -> str:
    avg = snapshot.average
    summary = (
        f"FPS {snapshot.moving_fps:.1f} | infer {avg.inference_ms:.1f} ms | "
        f"pre {avg.preprocess_ms:.1f} ms | post {avg.postprocess_ms:.1f} ms | "
        f"pipeline {avg.observed_pipeline_ms:.1f} ms | "
        f"draw {avg.draw_ms:.1f} ms | display {avg.display_ms:.1f} ms | "
        f"skipped {skipped_frames}"
    )
    if ignored_count is not None:
        summary += f" | self ignored {ignored_count}"
    return summary
