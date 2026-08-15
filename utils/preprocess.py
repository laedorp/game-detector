from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from threading import local

import numpy as np

from .inference_size import InferenceSize, InferenceSizeLike, normalize_inference_size


_LETTERBOX_LOCAL = local()


def _letterbox_workspace(inference_size: InferenceSize) -> np.ndarray:
    workspaces = getattr(_LETTERBOX_LOCAL, "workspaces", None)
    if workspaces is None:
        workspaces = {}
        _LETTERBOX_LOCAL.workspaces = workspaces
    workspace = workspaces.get(inference_size)
    if workspace is None:
        height, width = inference_size
        workspace = np.empty((height, width, 3), dtype=np.uint8)
        workspaces[inference_size] = workspace
    workspace.fill(114)
    return workspace


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
    inference_size: InferenceSizeLike,
    crop_size: int | tuple[int, int] | None = None,
) -> PreprocessedFrame:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install the packages from requirements.txt."
        ) from exc

    model_height, model_width = normalize_inference_size(inference_size)
    crop_height: int | None = None
    crop_width: int | None = None
    if crop_size is not None:
        if isinstance(crop_size, bool):
            raise TypeError(
                "crop_size must be a positive integer, an (height, width) pair, or None"
            )
        if isinstance(crop_size, Integral):
            crop_height = crop_width = int(crop_size)
        elif isinstance(crop_size, tuple) and len(crop_size) == 2:
            raw_height, raw_width = crop_size
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in (raw_height, raw_width)
            ):
                raise TypeError(
                    "crop_size pair must contain positive integer height and width"
                )
            crop_height, crop_width = int(raw_height), int(raw_width)
        else:
            raise TypeError(
                "crop_size must be a positive integer, an (height, width) pair, or None"
            )
        if crop_height <= 0 or crop_width <= 0:
            raise ValueError(
                "crop_size must contain positive dimensions or be None"
            )

    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected a BGR image with shape HxWx3, got {frame.shape}")

    source_height, source_width = frame.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError(
            f"expected a non-empty BGR image, got shape {frame.shape}"
        )
    crop_x = 0
    crop_y = 0
    crop_was_clamped = False
    roi = frame

    if crop_height is not None and crop_width is not None:
        applied_crop_height = min(crop_height, source_height)
        applied_crop_width = min(crop_width, source_width)
        crop_was_clamped = (
            applied_crop_height != crop_height
            or applied_crop_width != crop_width
        )
        crop_x = (source_width - applied_crop_width) // 2
        crop_y = (source_height - applied_crop_height) // 2
        roi = frame[
            crop_y : crop_y + applied_crop_height,
            crop_x : crop_x + applied_crop_width,
        ]

    roi_height, roi_width = roi.shape[:2]
    scale = min(model_width / roi_width, model_height / roi_height)
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

    horizontal_padding = model_width - resized_width
    vertical_padding = model_height - resized_height
    pad_left = int(round(horizontal_padding / 2 - 0.1))
    pad_top = int(round(vertical_padding / 2 - 0.1))

    if horizontal_padding or vertical_padding:
        letterboxed = _letterbox_workspace((model_height, model_width))
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
        model_width=model_width,
        model_height=model_height,
    )
    return PreprocessedFrame(tensor, transform, crop_was_clamped)
