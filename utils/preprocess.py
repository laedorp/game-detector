from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameTransform:
    scale: float
    pad_left: int
    pad_top: int
    crop_x: int
    crop_y: int
    source_width: int
    source_height: int
    model_width: int
    model_height: int

    def to_source_box(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        x1 = (x1 - self.pad_left) / self.scale + self.crop_x
        y1 = (y1 - self.pad_top) / self.scale + self.crop_y
        x2 = (x2 - self.pad_left) / self.scale + self.crop_x
        y2 = (y2 - self.pad_top) / self.scale + self.crop_y
        return (
            min(max(x1, 0.0), float(self.source_width - 1)),
            min(max(y1, 0.0), float(self.source_height - 1)),
            min(max(x2, 0.0), float(self.source_width - 1)),
            min(max(y2, 0.0), float(self.source_height - 1)),
        )


@dataclass(slots=True)
class PreprocessedFrame:
    tensor: np.ndarray
    transform: FrameTransform
    crop_was_clamped: bool


def preprocess_frame(
    frame: np.ndarray,
    inference_size: int,
    crop_size: int | None = None,
) -> PreprocessedFrame:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install the packages from requirements.txt."
        ) from exc

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected a BGR image with shape HxWx3, got {frame.shape}")

    source_height, source_width = frame.shape[:2]
    crop_x = 0
    crop_y = 0
    crop_was_clamped = False
    roi = frame

    if crop_size is not None:
        applied_crop = min(crop_size, source_width, source_height)
        crop_was_clamped = applied_crop != crop_size
        crop_x = (source_width - applied_crop) // 2
        crop_y = (source_height - applied_crop) // 2
        roi = frame[crop_y : crop_y + applied_crop, crop_x : crop_x + applied_crop]

    roi_height, roi_width = roi.shape[:2]
    scale = min(inference_size / roi_width, inference_size / roi_height)
    resized_width = max(1, int(round(roi_width * scale)))
    resized_height = max(1, int(round(roi_height * scale)))

    if resized_width == roi_width and resized_height == roi_height:
        resized = roi
    else:
        resized = cv2.resize(
            roi,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    horizontal_padding = inference_size - resized_width
    vertical_padding = inference_size - resized_height
    pad_left = int(round(horizontal_padding / 2 - 0.1))
    pad_top = int(round(vertical_padding / 2 - 0.1))

    if horizontal_padding or vertical_padding:
        letterboxed = np.full(
            (inference_size, inference_size, 3),
            114,
            dtype=np.uint8,
        )
        letterboxed[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ] = resized
    else:
        letterboxed = resized

    # Channel reversal and transpose are views; ascontiguousarray performs the
    # one required conversion into the model's contiguous float32 input.
    rgb_chw = letterboxed[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(rgb_chw, dtype=np.float32)
    tensor *= 1.0 / 255.0
    tensor = tensor[np.newaxis, ...]

    transform = FrameTransform(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        crop_x=crop_x,
        crop_y=crop_y,
        source_width=source_width,
        source_height=source_height,
        model_width=inference_size,
        model_height=inference_size,
    )
    return PreprocessedFrame(tensor, transform, crop_was_clamped)
