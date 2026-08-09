"""YOLO output decoding and class-aware non-maximum suppression."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .base import OutputDecodeError
from .types import Detection


class FrameTransformLike(Protocol):
    """Structural contract supplied by ``utils.preprocess.FrameTransform``."""

    scale: float
    pad_left: int | float
    pad_top: int | float
    crop_x: int | float
    crop_y: int | float
    source_width: int
    source_height: int
    model_width: int
    model_height: int


def _validate_threshold(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value between 0 and 1, got {value!r}.")
    return value


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return confidence-ordered indices kept by per-class NMS."""

    iou_threshold = _validate_threshold(iou_threshold, "iou_threshold")
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    class_ids = np.asarray(class_ids)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {boxes.shape}.")
    if scores.ndim != 1 or class_ids.ndim != 1:
        raise ValueError("scores and class_ids must be one-dimensional.")
    if not (len(boxes) == len(scores) == len(class_ids)):
        raise ValueError("boxes, scores, and class_ids must have the same length.")
    if not len(boxes):
        return np.empty(0, dtype=np.int64)

    kept: list[int] = []
    for class_id in np.unique(class_ids):
        candidates = np.flatnonzero(class_ids == class_id)
        order = candidates[np.argsort(-scores[candidates], kind="stable")]

        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break

            remaining = order[1:]
            current_box = boxes[current]
            other_boxes = boxes[remaining]

            inter_x1 = np.maximum(current_box[0], other_boxes[:, 0])
            inter_y1 = np.maximum(current_box[1], other_boxes[:, 1])
            inter_x2 = np.minimum(current_box[2], other_boxes[:, 2])
            inter_y2 = np.minimum(current_box[3], other_boxes[:, 3])
            inter_w = np.maximum(0.0, inter_x2 - inter_x1)
            inter_h = np.maximum(0.0, inter_y2 - inter_y1)
            intersection = inter_w * inter_h

            current_area = max(
                0.0,
                float(current_box[2] - current_box[0]),
            ) * max(0.0, float(current_box[3] - current_box[1]))
            other_areas = np.maximum(0.0, other_boxes[:, 2] - other_boxes[:, 0]) * np.maximum(
                0.0,
                other_boxes[:, 3] - other_boxes[:, 1],
            )
            union = current_area + other_areas - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0.0,
            )
            order = remaining[iou <= iou_threshold]

    result = np.asarray(kept, dtype=np.int64)
    return result[np.argsort(-scores[result], kind="stable")]


def _as_prediction_matrix(raw: np.ndarray) -> np.ndarray:
    output = np.asarray(raw)
    if output.ndim == 3:
        if output.shape[0] != 1:
            raise OutputDecodeError(
                f"Only batch size 1 is supported, but output shape is {output.shape}."
            )
        output = output[0]
    if output.ndim != 2:
        raise OutputDecodeError(
            "Expected one YOLO output shaped [1, N, 6], [1, N, 4+C], or "
            f"[1, 4+C, N]; got {output.shape}."
        )
    if output.size == 0:
        return output.astype(np.float32, copy=False)
    if not np.issubdtype(output.dtype, np.number):
        raise OutputDecodeError(f"YOLO output must be numeric, got dtype {output.dtype}.")
    return output.astype(np.float32, copy=False)


def _traditional_predictions(output: np.ndarray, label_count: int) -> np.ndarray:
    rows, columns = output.shape
    expected_attributes = label_count + 4 if label_count else None

    if expected_attributes and columns == expected_attributes:
        predictions = output
    elif expected_attributes and rows == expected_attributes:
        predictions = output.T
    elif columns >= 5 and (rows < 5 or columns <= rows):
        predictions = output
    elif rows >= 5:
        predictions = output.T
    else:
        raise OutputDecodeError(
            f"Traditional YOLO output needs at least five attributes, got {output.shape}."
        )

    attributes = predictions.shape[1]
    if attributes < 5:
        raise OutputDecodeError(
            f"Traditional YOLO output needs xywh plus class scores, got {predictions.shape}."
        )
    if label_count and attributes == label_count + 5:
        raise OutputDecodeError(
            "This appears to be an objectness-style YOLO output (xywh + objectness + "
            "class scores). This detector supports YOLOv8/11 outputs without a separate "
            "objectness column. Export the model in the supported layout or supply a "
            "custom decoder."
        )
    if label_count and attributes != label_count + 4:
        raise OutputDecodeError(
            f"Model output contains {attributes - 4} class scores, but {label_count} "
            "labels were loaded."
        )
    return predictions


def _frame_dimensions(
    transform: FrameTransformLike | None,
    frame_shape: Sequence[int] | None,
) -> tuple[float, float] | None:
    if transform is not None:
        try:
            width = float(transform.source_width)
            height = float(transform.source_height)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "transform must provide numeric source_width and source_height attributes."
            ) from exc
        if width <= 0.0 or height <= 0.0:
            raise ValueError("transform source dimensions must be positive.")
        return width, height

    if frame_shape is None:
        return None
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain at least height and width.")
    height = float(frame_shape[0])
    width = float(frame_shape[1])
    if width <= 0.0 or height <= 0.0:
        raise ValueError("frame_shape dimensions must be positive.")
    return width, height


def map_boxes_to_source(
    boxes: np.ndarray,
    transform: FrameTransformLike | None = None,
    frame_shape: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map model-space boxes to source space and return boxes plus valid mask."""

    mapped = np.asarray(boxes, dtype=np.float32).copy()
    if mapped.ndim != 2 or mapped.shape[1] != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {mapped.shape}.")

    if transform is not None:
        try:
            scale = float(transform.scale)
            pad_left = float(transform.pad_left)
            pad_top = float(transform.pad_top)
            crop_x = float(transform.crop_x)
            crop_y = float(transform.crop_y)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "transform must provide numeric scale, pad_left, pad_top, crop_x, and "
                "crop_y attributes."
            ) from exc
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"transform.scale must be positive, got {scale!r}.")
        mapped[:, (0, 2)] = (mapped[:, (0, 2)] - pad_left) / scale + crop_x
        mapped[:, (1, 3)] = (mapped[:, (1, 3)] - pad_top) / scale + crop_y

    dimensions = _frame_dimensions(transform, frame_shape)
    if dimensions is not None:
        width, height = dimensions
        mapped[:, (0, 2)] = np.clip(mapped[:, (0, 2)], 0.0, width)
        mapped[:, (1, 3)] = np.clip(mapped[:, (1, 3)], 0.0, height)

    valid = (
        np.isfinite(mapped).all(axis=1)
        & (mapped[:, 2] > mapped[:, 0])
        & (mapped[:, 3] > mapped[:, 1])
    )
    return mapped, valid


def _label_for(class_id: int, labels: Sequence[str]) -> str:
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class_{class_id}"


def _to_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    labels: Sequence[str],
) -> list[Detection]:
    return [
        Detection(
            class_id=int(class_id),
            class_name=_label_for(int(class_id), labels),
            confidence=float(score),
            xyxy=tuple(float(value) for value in box),
        )
        for box, score, class_id in zip(boxes, scores, class_ids, strict=True)
    ]


def decode_yolo_output(
    raw: np.ndarray,
    *,
    transform: FrameTransformLike | None = None,
    frame_shape: Sequence[int] | None = None,
    labels: Sequence[str] = (),
    confidence: float = 0.25,
    iou: float = 0.45,
    output_format: str = "auto",
) -> list[Detection]:
    """Decode supported YOLO output layouts into source-space detections.

    End-to-end ``[1, N, 6]`` outputs are interpreted as ``xyxy, confidence,
    class_id`` and are not passed through NMS. Traditional YOLOv8/11 layouts
    contain ``xywh`` and per-class scores and receive class-aware NMS.
    """

    confidence = _validate_threshold(confidence, "confidence")
    iou = _validate_threshold(iou, "iou")
    output_format = str(output_format).lower()
    if output_format not in {"auto", "end2end", "traditional"}:
        raise ValueError("output_format must be 'auto', 'end2end', or 'traditional'.")
    output = _as_prediction_matrix(raw)
    if output.size == 0:
        return []

    # YOLO26 end-to-end exports put the six result attributes last. A [6, N]
    # tensor remains available to two-class traditional YOLO exports.
    decode_end_to_end = output_format == "end2end" or (
        output_format == "auto" and output.shape[1] == 6
    )
    if decode_end_to_end:
        if output.shape[1] != 6:
            raise OutputDecodeError(
                f"End-to-end YOLO output must have shape [N, 6], got {output.shape}."
            )
        finite = np.isfinite(output).all(axis=1)
        class_values = output[:, 5]
        integral_classes = np.isclose(class_values, np.rint(class_values), atol=1e-4)
        selected = finite & integral_classes & (class_values >= 0.0) & (output[:, 4] >= confidence)
        predictions = output[selected]
        if not len(predictions):
            return []

        scores = predictions[:, 4]
        class_ids = np.rint(predictions[:, 5]).astype(np.int64)
        boxes, valid = map_boxes_to_source(predictions[:, :4], transform, frame_shape)
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        order = np.argsort(-scores, kind="stable")
        return _to_detections(boxes[order], scores[order], class_ids[order], labels)

    predictions = _traditional_predictions(output, len(labels))
    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    finite = np.isfinite(boxes_xywh).all(axis=1) & np.isfinite(class_scores).all(axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    selected = finite & (scores >= confidence)
    if not np.any(selected):
        return []

    boxes_xywh = boxes_xywh[selected]
    scores = scores[selected].astype(np.float32, copy=False)
    class_ids = class_ids[selected].astype(np.int64, copy=False)
    boxes = np.empty_like(boxes_xywh, dtype=np.float32)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

    valid_model_boxes = (
        np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    boxes = boxes[valid_model_boxes]
    scores = scores[valid_model_boxes]
    class_ids = class_ids[valid_model_boxes]
    if not len(boxes):
        return []

    kept = class_aware_nms(boxes, scores, class_ids, iou)
    boxes, valid_source_boxes = map_boxes_to_source(boxes[kept], transform, frame_shape)
    scores = scores[kept][valid_source_boxes]
    class_ids = class_ids[kept][valid_source_boxes]
    boxes = boxes[valid_source_boxes]
    return _to_detections(boxes, scores, class_ids, labels)
