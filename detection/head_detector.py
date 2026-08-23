"""Fail-closed second-stage localization for a direct player/head detector.

The primary detector owns target identity.  This module crops that selected
player, runs one fixed 320-pixel detector, and accepts only a directly detected
``head`` box geometrically contained by both the supplied primary player box
and exactly one matching ``player`` instance from the same inference pass.  It
never synthesizes a head point from a player box.

The decoder intentionally implements the exact inspected SunXDS 0.8.0 export
contract instead of guessing among YOLO layouts:

* input: letterboxed RGB float32 ``[1, 3, 320, 320]`` scaled to ``[0, 1]``;
* output: float ``[1, 6, 2100]``;
* row after transpose: ``cx, cy, width, height, player_score, head_score``;
* class ids: ``0 = player`` and ``1 = head``.

Pure crop, preprocessing, decoding, and association functions keep inference
and newest-frame-only scheduling outside this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil, hypot, isfinite
from numbers import Integral
import os
from pathlib import Path
from typing import TypeAlias

import numpy as np

from .base import OutputDecodeError
from .head_worker import HeadLocalizationReason
from .postprocess import class_aware_nms


Box: TypeAlias = tuple[float, float, float, float]
Point: TypeAlias = tuple[float, float]

HEAD_INPUT_HEIGHT = 320
HEAD_INPUT_WIDTH = 320
HEAD_OUTPUT_ATTRIBUTES = 6
HEAD_OUTPUT_CANDIDATES = 2100
PLAYER_CLASS_ID = 0
HEAD_CLASS_ID = 1
HEAD_CLASS_NAMES = ("player", "head")

# Exact-runtime held-out evaluation selected 0.15 as the lowest threshold that
# retained 100% accepted-point precision while improving direct-head recall from
# 54.1% at 0.25 to 64.9%. The lower 0.10 candidate introduced a false location.
DEFAULT_HEAD_CONFIDENCE = 0.15
DEFAULT_NMS_IOU = 0.45
DEFAULT_CROP_SCALE = 2.00
# A large, close player already supplies ample context inside its primary box.
# Keeping the 2x crop used by distant players makes that close player needlessly
# small at the model input and admits more neighbouring/build geometry.  These
# thresholds are resolution-normalized to the 1080p capture used for the live
# evaluation.  A small exit hysteresis prevents detector-size jitter around the
# boundary from alternating crop geometry frame by frame.
DEFAULT_CLOSE_PLAYER_CROP_SCALE = 1.25
DEFAULT_CLOSE_PLAYER_ENTER_HEIGHT_AT_1080P = 200.0
DEFAULT_CLOSE_PLAYER_EXIT_HEIGHT_AT_1080P = 180.0
HEAD_CROP_REFERENCE_FRAME_HEIGHT = 1080.0
DEFAULT_MIN_CROP_SIDE = 64
DEFAULT_MIN_HEAD_CONTAINMENT = 0.60
DEFAULT_MIN_PLAYER_OVERLAP = 0.50
DEFAULT_MAX_PLAYER_CENTER_DISPLACEMENT_RATIO = 0.35
DEFAULT_PLAYER_BOX_MARGIN = 0.08
DEFAULT_MAX_HEAD_AREA_RATIO = 0.50
DEFAULT_MAX_HEAD_CENTER_Y_RATIO = 0.45
# The head export can retain a tight, fully nested pair of differently scaled
# player boxes just below its class-aware NMS threshold. This is not permission
# to choose between people: only the exact two-box geometry below, with one
# sole head supported by both boxes, is treated as duplicate evidence.
NESTED_PLAYER_DUPLICATE_MIN_IOU = 0.36
NESTED_PLAYER_DUPLICATE_MIN_SMALLER_OVERLAP = 0.99
NESTED_PLAYER_DUPLICATE_MAX_CENTER_DISPLACEMENT_RATIO = 0.15
NESTED_PLAYER_DUPLICATE_MIN_PRIMARY_OVERLAP = 0.90
NESTED_PLAYER_DUPLICATE_MAX_PRIMARY_TOP_DELTA_RATIO = 0.18
MAX_HEAD_NMS_CANDIDATES = 128
MAX_HEAD_DETECTIONS = 32

PINNED_HEAD_MODEL_RELATIVE_PATH = Path(
    "models/sunxds_head_onnx/sunxds_0.8.0.onnx"
)
PINNED_HEAD_MODEL_SIZE_BYTES = 10_392_860
PINNED_HEAD_MODEL_SHA256 = (
    "93264ec61b86b8459ef64c85a31ab3da294327ee1f95337076e57d8af24bb192"
)
DIRECT_HEAD_RUNTIME_MANIFEST_ENV = "PROAIM_DIRECT_HEAD_RUNTIME_MANIFEST"


@dataclass(frozen=True, slots=True)
class HeadModelSpec:
    path: Path
    input_height: int
    input_width: int
    output_attributes: int
    output_candidates: int
    model_name: str
    evidence_label: str
    confidence_threshold: float = DEFAULT_HEAD_CONFIDENCE

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.is_file():
            raise FileNotFoundError(f"head model not found: {path}")
        input_height = _non_negative_integer(self.input_height, "input_height")
        input_width = _non_negative_integer(self.input_width, "input_width")
        output_attributes = _non_negative_integer(
            self.output_attributes,
            "output_attributes",
        )
        output_candidates = _non_negative_integer(
            self.output_candidates,
            "output_candidates",
        )
        if input_height <= 0 or input_width <= 0:
            raise ValueError("head model input dimensions must be positive")
        if output_attributes != HEAD_OUTPUT_ATTRIBUTES:
            raise ValueError(
                "head model output_attributes must be exactly "
                f"{HEAD_OUTPUT_ATTRIBUTES}"
            )
        if output_candidates <= 0:
            raise ValueError("head model output_candidates must be positive")
        model_name = str(self.model_name).strip()
        if not model_name:
            raise ValueError("head model name must not be empty")
        evidence_label = str(self.evidence_label).strip()
        if not evidence_label:
            raise ValueError("head model evidence label must not be empty")
        confidence_threshold = _finite_threshold(
            self.confidence_threshold,
            "confidence_threshold",
        )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "input_height", input_height)
        object.__setattr__(self, "input_width", input_width)
        object.__setattr__(self, "output_attributes", output_attributes)
        object.__setattr__(self, "output_candidates", output_candidates)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "evidence_label", evidence_label)
        object.__setattr__(self, "confidence_threshold", confidence_threshold)

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.input_height, self.input_width)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (1, self.output_attributes, self.output_candidates)


def _project_root(project_root: str | Path | None = None) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        if project_root is None
        else Path(project_root).expanduser().resolve()
    )


def _verify_model_file(
    model_path: Path,
    *,
    expected_size_bytes: int | None,
    expected_sha256: str | None,
    description: str,
) -> Path:
    if not model_path.is_file():
        raise FileNotFoundError(f"{description} not found: {model_path}")
    size = model_path.stat().st_size
    if expected_size_bytes is not None and size != int(expected_size_bytes):
        raise ValueError(
            f"{description} size mismatch: expected {expected_size_bytes}, got {size}"
        )
    if expected_sha256 is not None:
        digest = sha256()
        with model_path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != str(expected_sha256):
            raise ValueError(
                f"{description} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            )
    return model_path


def _default_head_model_spec(project_root: str | Path | None = None) -> HeadModelSpec:
    root = _project_root(project_root)
    return HeadModelSpec(
        path=_verify_model_file(
            root / PINNED_HEAD_MODEL_RELATIVE_PATH,
            expected_size_bytes=PINNED_HEAD_MODEL_SIZE_BYTES,
            expected_sha256=PINNED_HEAD_MODEL_SHA256,
            description="Pinned head model",
        ),
        input_height=HEAD_INPUT_HEIGHT,
        input_width=HEAD_INPUT_WIDTH,
        output_attributes=HEAD_OUTPUT_ATTRIBUTES,
        output_candidates=HEAD_OUTPUT_CANDIDATES,
        model_name="SunXDS 0.8.0",
        evidence_label="SunXDS 0.8.0 direct head box",
        confidence_threshold=DEFAULT_HEAD_CONFIDENCE,
    )


def runtime_head_model_spec(project_root: str | Path | None = None) -> HeadModelSpec:
    """Resolve the runtime direct-head model, optionally from a local manifest."""

    root = _project_root(project_root)
    manifest_value = str(os.environ.get(DIRECT_HEAD_RUNTIME_MANIFEST_ENV, "")).strip()
    if not manifest_value:
        return _default_head_model_spec(root)
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Direct-head runtime manifest not found: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("direct-head runtime manifest must be a JSON object")
    model_value = payload.get("model") or payload.get("onnx")
    if not isinstance(model_value, str) or not model_value.strip():
        raise ValueError("direct-head runtime manifest must declare model/onnx")
    model_path = Path(model_value).expanduser()
    if not model_path.is_absolute():
        model_path = root / model_path
    input_shape = payload.get("input_shape_nchw")
    output_shape = payload.get("output_shape")
    if not isinstance(input_shape, list) or len(input_shape) != 4:
        raise ValueError(
            "direct-head runtime manifest input_shape_nchw must be [1, 3, H, W]"
        )
    if not isinstance(output_shape, list) or len(output_shape) != 3:
        raise ValueError(
            "direct-head runtime manifest output_shape must be [1, 6, N]"
        )
    input_shape = [int(value) for value in input_shape]
    output_shape = [int(value) for value in output_shape]
    if input_shape[0:2] != [1, 3]:
        raise ValueError(
            "direct-head runtime manifest input_shape_nchw must begin with [1, 3]"
        )
    if output_shape[0:2] != [1, HEAD_OUTPUT_ATTRIBUTES]:
        raise ValueError(
            "direct-head runtime manifest output_shape must begin with "
            f"[1, {HEAD_OUTPUT_ATTRIBUTES}]"
        )
    return HeadModelSpec(
        path=_verify_model_file(
            model_path.resolve(),
            expected_size_bytes=(
                int(payload["model_size_bytes"])
                if payload.get("model_size_bytes") is not None
                else None
            ),
            expected_sha256=(
                str(payload.get("model_sha256") or payload.get("onnx_sha256"))
                if payload.get("model_sha256") is not None
                or payload.get("onnx_sha256") is not None
                else None
            ),
            description="Direct-head runtime override model",
        ),
        input_height=input_shape[2],
        input_width=input_shape[3],
        output_attributes=output_shape[1],
        output_candidates=output_shape[2],
        model_name=str(payload.get("model_name") or payload.get("name") or model_path.stem),
        evidence_label=str(
            payload.get("evidence_label")
            or f"{payload.get('model_name') or payload.get('name') or model_path.stem} direct head box"
        ),
        confidence_threshold=float(
            payload.get("confidence_threshold")
            if payload.get("confidence_threshold") is not None
            else payload.get("confidence")
            if payload.get("confidence") is not None
            else DEFAULT_HEAD_CONFIDENCE
        ),
    )


def _finite_threshold(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite value between 0 and 1")
    return result


def _validated_box(box: Sequence[float], name: str) -> Box:
    if len(box) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    result = tuple(float(value) for value in box)
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite coordinates")
    x1, y1, x2, y2 = result
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must have positive width and height")
    return x1, y1, x2, y2


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def pinned_head_model_path(project_root: str | Path | None = None) -> Path:
    """Return the repository/package path of the pinned direct-head model."""

    root = _project_root(project_root)
    return root / PINNED_HEAD_MODEL_RELATIVE_PATH


def verify_pinned_head_model(path: str | Path | None = None) -> Path:
    """Verify exact size and SHA-256 before a runtime session loads the model."""

    model_path = pinned_head_model_path() if path is None else Path(path).expanduser()
    return _verify_model_file(
        model_path,
        expected_size_bytes=PINNED_HEAD_MODEL_SIZE_BYTES,
        expected_sha256=PINNED_HEAD_MODEL_SHA256,
        description="Pinned head model",
    )


@dataclass(frozen=True, slots=True)
class HeadCropTransform:
    """One selected-player crop and its 320-pixel letterbox transform."""

    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    scale: float
    model_width: int = HEAD_INPUT_WIDTH
    model_height: int = HEAD_INPUT_HEIGHT

    @property
    def right(self) -> int:
        return self.crop_x + self.crop_width

    @property
    def bottom(self) -> int:
        return self.crop_y + self.crop_height

    def to_source_point(self, point: Point) -> Point:
        x, y = point
        return (
            (float(x) - self.pad_left) / self.scale + self.crop_x,
            (float(y) - self.pad_top) / self.scale + self.crop_y,
        )

    def to_source_box(self, box: Box) -> Box:
        """Map and clip a model box to the pixels actually present in the crop."""

        x1, y1 = self.to_source_point((box[0], box[1]))
        x2, y2 = self.to_source_point((box[2], box[3]))
        return (
            min(max(x1, float(self.crop_x)), float(self.right)),
            min(max(y1, float(self.crop_y)), float(self.bottom)),
            min(max(x2, float(self.crop_x)), float(self.right)),
            min(max(y2, float(self.crop_y)), float(self.bottom)),
        )


@dataclass(frozen=True, slots=True)
class PreparedHeadInput:
    tensor: np.ndarray
    transform: HeadCropTransform


@dataclass(frozen=True, slots=True)
class HeadCandidate:
    class_id: int
    class_name: str
    confidence: float
    box: Box
    row_index: int


@dataclass(frozen=True, slots=True)
class HeadLocalization:
    """One direct head-box center tied to the observed primary player box."""

    point: Point
    source_timestamp_ns: int
    confidence: float
    head_box: Box
    containment: float
    candidate_index: int
    supporting_player_index: int | None


@dataclass(frozen=True, slots=True)
class HeadAssociationOutcome:
    """One direct-head association result with an immutable disposition."""

    reason: HeadLocalizationReason
    localization: HeadLocalization | None

    def __post_init__(self) -> None:
        try:
            reason = HeadLocalizationReason(self.reason)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown head-association reason") from exc
        localization = self.localization
        if localization is not None and not isinstance(localization, HeadLocalization):
            raise TypeError("localization must be HeadLocalization or None")
        if (localization is not None) != (reason is HeadLocalizationReason.LOCALIZED):
            raise ValueError(
                "localized association outcomes require a localization and "
                "rejected outcomes forbid one"
            )
        object.__setattr__(self, "reason", reason)


def adaptive_head_crop_scale(
    frame_shape: Sequence[int],
    player_box: Sequence[float],
    *,
    previous_crop_scale: float | None = None,
) -> float:
    """Choose close-player detail or distant-player context with hysteresis.

    The player-height boundary is expressed at a 1080p reference so the same
    apparent target size selects the same crop on other capture resolutions.
    ``previous_crop_scale`` is optional; supplying the last returned value
    enables the 180--200 reference-pixel hysteresis band.
    """

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    source_height = int(frame_shape[0])
    source_width = int(frame_shape[1])
    if source_height <= 0 or source_width <= 0:
        raise ValueError("frame dimensions must be positive")
    raw_x1, raw_y1, raw_x2, raw_y2 = _validated_box(
        player_box,
        "player_box",
    )
    x1 = min(max(raw_x1, 0.0), float(source_width))
    y1 = min(max(raw_y1, 0.0), float(source_height))
    x2 = min(max(raw_x2, 0.0), float(source_width))
    y2 = min(max(raw_y2, 0.0), float(source_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("player_box does not intersect the source frame")

    prior_scale = DEFAULT_CROP_SCALE
    if previous_crop_scale is not None:
        prior_scale = float(previous_crop_scale)
        if not isfinite(prior_scale) or prior_scale < 1.0:
            raise ValueError(
                "previous_crop_scale must be finite and at least 1"
            )
    reference_height = (
        (y2 - y1) * HEAD_CROP_REFERENCE_FRAME_HEIGHT / source_height
    )
    close_crop_active = prior_scale == DEFAULT_CLOSE_PLAYER_CROP_SCALE
    threshold = (
        DEFAULT_CLOSE_PLAYER_EXIT_HEIGHT_AT_1080P
        if close_crop_active
        else DEFAULT_CLOSE_PLAYER_ENTER_HEIGHT_AT_1080P
    )
    return (
        DEFAULT_CLOSE_PLAYER_CROP_SCALE
        if reference_height >= threshold
        else DEFAULT_CROP_SCALE
    )


def plan_head_crop(
    frame_shape: Sequence[int],
    player_box: Sequence[float],
    *,
    crop_scale: float = DEFAULT_CROP_SCALE,
    min_crop_side: int = DEFAULT_MIN_CROP_SIDE,
    model_size: tuple[int, int] = (HEAD_INPUT_HEIGHT, HEAD_INPUT_WIDTH),
) -> HeadCropTransform:
    """Plan an in-bounds context crop containing the supplied player box."""

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    source_height = int(frame_shape[0])
    source_width = int(frame_shape[1])
    if source_height <= 0 or source_width <= 0:
        raise ValueError("frame dimensions must be positive")
    if not isfinite(float(crop_scale)) or float(crop_scale) < 1.0:
        raise ValueError("crop_scale must be finite and at least 1")
    if isinstance(min_crop_side, bool) or not isinstance(min_crop_side, Integral):
        raise TypeError("min_crop_side must be a positive integer")
    if min_crop_side <= 0:
        raise ValueError("min_crop_side must be a positive integer")
    if len(model_size) != 2:
        raise ValueError("model_size must contain height and width")
    model_height, model_width = (int(model_size[0]), int(model_size[1]))
    if model_height <= 0 or model_width <= 0:
        raise ValueError("model dimensions must be positive")

    raw_x1, raw_y1, raw_x2, raw_y2 = _validated_box(player_box, "player_box")
    x1 = min(max(raw_x1, 0.0), float(source_width))
    y1 = min(max(raw_y1, 0.0), float(source_height))
    x2 = min(max(raw_x2, 0.0), float(source_width))
    y2 = min(max(raw_y2, 0.0), float(source_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("player_box does not intersect the source frame")

    requested_side = max(
        int(min_crop_side),
        int(ceil(max(x2 - x1, y2 - y1) * float(crop_scale))),
    )
    crop_width = min(source_width, requested_side)
    crop_height = min(source_height, requested_side)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5

    left_min = max(0, int(ceil(x2)) - crop_width)
    left_max = min(int(x1), source_width - crop_width)
    top_min = max(0, int(ceil(y2)) - crop_height)
    top_max = min(int(y1), source_height - crop_height)
    crop_x = min(max(int(round(center_x - crop_width * 0.5)), left_min), left_max)
    crop_y = min(max(int(round(center_y - crop_height * 0.5)), top_min), top_max)

    scale = min(model_width / crop_width, model_height / crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    pad_left = int(round((model_width - resized_width) / 2 - 0.1))
    pad_top = int(round((model_height - resized_height) / 2 - 0.1))
    return HeadCropTransform(
        crop_x=crop_x,
        crop_y=crop_y,
        crop_width=crop_width,
        crop_height=crop_height,
        source_width=source_width,
        source_height=source_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        scale=scale,
        model_width=model_width,
        model_height=model_height,
    )


def prepare_head_input(
    frame: np.ndarray,
    transform: HeadCropTransform,
) -> PreparedHeadInput:
    """Crop and apply the detector's exact letterbox/RGB preprocessing."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to prepare head input") from exc

    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected a BGR image with shape HxWx3, got {frame.shape}")
    if (
        frame.shape[0] != transform.source_height
        or frame.shape[1] != transform.source_width
    ):
        raise ValueError("frame dimensions do not match the crop transform")

    crop = frame[
        transform.crop_y : transform.bottom,
        transform.crop_x : transform.right,
    ]
    resized = cv2.resize(
        crop,
        (transform.resized_width, transform.resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    if (
        transform.resized_width == transform.model_width
        and transform.resized_height == transform.model_height
    ):
        letterboxed = resized
    else:
        letterboxed = np.full(
            (transform.model_height, transform.model_width, 3),
            114,
            dtype=np.uint8,
        )
        letterboxed[
            transform.pad_top : transform.pad_top + transform.resized_height,
            transform.pad_left : transform.pad_left + transform.resized_width,
        ] = resized

    rgb_chw = letterboxed[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(rgb_chw, dtype=np.float32)
    tensor *= 1.0 / 255.0
    return PreparedHeadInput(tensor[np.newaxis, ...], transform)


def _head_rows(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if (
        array.ndim != 3
        or array.shape[0] != 1
        or array.shape[1] != HEAD_OUTPUT_ATTRIBUTES
        or array.shape[2] <= 0
    ):
        raise OutputDecodeError(
            "direct head output must have exact shape [1, "
            f"{HEAD_OUTPUT_ATTRIBUTES}, N], got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise OutputDecodeError(
            f"direct head output must be numeric, got dtype {array.dtype}"
        )
    return np.asarray(array[0].T, dtype=np.float32)


def decode_head_output(
    output: np.ndarray,
    transform: HeadCropTransform,
    *,
    confidence: float = DEFAULT_HEAD_CONFIDENCE,
    nms_iou: float = DEFAULT_NMS_IOU,
    max_detections: int = MAX_HEAD_DETECTIONS,
) -> list[HeadCandidate]:
    """Decode, bound, suppress, and map the exact two-class model output."""

    confidence = _finite_threshold(confidence, "confidence")
    nms_iou = _finite_threshold(nms_iou, "nms_iou")
    if isinstance(max_detections, bool) or not isinstance(max_detections, Integral):
        raise TypeError("max_detections must be a positive bounded integer")
    if not 1 <= int(max_detections) <= MAX_HEAD_DETECTIONS:
        raise ValueError(
            f"max_detections must be between 1 and {MAX_HEAD_DETECTIONS}"
        )

    rows = _head_rows(output)
    finite = np.isfinite(rows).all(axis=1)
    class_ids = np.argmax(rows[:, 4:6], axis=1)
    scores = rows[np.arange(len(rows)), class_ids + 4]
    eligible = np.flatnonzero(
        finite & (scores >= confidence) & (scores <= 1.0)
    )
    if not len(eligible):
        return []
    per_class_limit = max(
        1,
        MAX_HEAD_NMS_CANDIDATES // len(HEAD_CLASS_NAMES),
    )
    class_orders = []
    for class_id in range(len(HEAD_CLASS_NAMES)):
        class_eligible = eligible[class_ids[eligible] == class_id]
        class_order = class_eligible[
            np.argsort(-scores[class_eligible], kind="stable")
        ]
        class_orders.append(class_order[:per_class_limit])
    order = np.concatenate(class_orders)
    order = order[np.argsort(-scores[order], kind="stable")]

    candidates: list[HeadCandidate] = []
    boxes: list[Box] = []
    kept_scores: list[float] = []
    kept_classes: list[int] = []
    for row_index in order:
        row = rows[int(row_index)]
        center_x, center_y, width, height = (float(value) for value in row[:4])
        if width <= 0.0 or height <= 0.0:
            continue
        model_box = (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        )
        source_box = transform.to_source_box(model_box)
        if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
            continue
        class_id = int(class_ids[int(row_index)])
        score = float(scores[int(row_index)])
        candidates.append(
            HeadCandidate(
                class_id=class_id,
                class_name=HEAD_CLASS_NAMES[class_id],
                confidence=score,
                box=source_box,
                row_index=int(row_index),
            )
        )
        boxes.append(source_box)
        kept_scores.append(score)
        kept_classes.append(class_id)

    if not candidates:
        return []
    kept = class_aware_nms(
        np.asarray(boxes, dtype=np.float32),
        np.asarray(kept_scores, dtype=np.float32),
        np.asarray(kept_classes, dtype=np.int64),
        nms_iou,
    )
    kept_indices = [int(index) for index in kept]
    present_classes = {
        candidates[index].class_id for index in kept_indices
    }
    if len(present_classes) > 1:
        reserved_per_class = max(1, int(max_detections) // len(present_classes))
        selected: list[int] = []
        for class_id in sorted(present_classes):
            selected.extend(
                [
                    index
                    for index in kept_indices
                    if candidates[index].class_id == class_id
                ][:reserved_per_class]
            )
        selected_set = set(selected)
        remaining_slots = max(0, int(max_detections) - len(selected))
        selected.extend(
            [
                index
                for index in kept_indices
                if index not in selected_set
            ][:remaining_slots]
        )
        selected.sort(
            key=lambda index: candidates[index].confidence,
            reverse=True,
        )
        kept_indices = selected
    return [
        candidates[index]
        for index in kept_indices[: int(max_detections)]
    ]


def _box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(first: Box, second: Box) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )


def _strict_nested_player_duplicate_pair(
    matching_players: Sequence[tuple[int, Box]],
    target: Box,
    *,
    head_box: Box,
    head_center: Point,
    min_head_containment: float,
) -> bool:
    """Recognize only a fully corroborated two-box multiscale duplicate."""

    if len(matching_players) != 2:
        return False
    first = matching_players[0][1]
    second = matching_players[1][1]
    first_area = _box_area(first)
    second_area = _box_area(second)
    target_area = _box_area(target)
    head_area = _box_area(head_box)
    if min(first_area, second_area, target_area, head_area) <= 0.0:
        return False

    pair_intersection = _intersection_area(first, second)
    pair_union = first_area + second_area - pair_intersection
    if (
        pair_union <= 0.0
        or pair_intersection / pair_union < NESTED_PLAYER_DUPLICATE_MIN_IOU
        or pair_intersection / min(first_area, second_area)
        < NESTED_PLAYER_DUPLICATE_MIN_SMALLER_OVERLAP
    ):
        return False

    target_width = target[2] - target[0]
    target_height = target[3] - target[1]
    first_center = (
        (first[0] + first[2]) * 0.5,
        (first[1] + first[3]) * 0.5,
    )
    second_center = (
        (second[0] + second[2]) * 0.5,
        (second[1] + second[3]) * 0.5,
    )
    if hypot(
        (first_center[0] - second_center[0]) / target_width,
        (first_center[1] - second_center[1]) / target_height,
    ) > NESTED_PLAYER_DUPLICATE_MAX_CENTER_DISPLACEMENT_RATIO:
        return False

    for secondary, secondary_area in (
        (first, first_area),
        (second, second_area),
    ):
        target_overlap = _intersection_area(target, secondary) / min(
            target_area,
            secondary_area,
        )
        if target_overlap < NESTED_PLAYER_DUPLICATE_MIN_PRIMARY_OVERLAP:
            return False
        if (
            abs(secondary[1] - target[1]) / target_height
            > NESTED_PLAYER_DUPLICATE_MAX_PRIMARY_TOP_DELTA_RATIO
        ):
            return False
        if not (
            secondary[0] <= head_center[0] <= secondary[2]
            and secondary[1] <= head_center[1] <= secondary[3]
        ):
            return False
        if (
            _intersection_area(head_box, secondary) / head_area
            < min_head_containment
        ):
            return False
    return True


def associate_head_to_player_outcome(
    candidates: Sequence[HeadCandidate],
    player_box: Sequence[float],
    *,
    source_timestamp_ns: int,
    min_containment: float = DEFAULT_MIN_HEAD_CONTAINMENT,
    min_player_overlap: float = DEFAULT_MIN_PLAYER_OVERLAP,
    max_player_center_displacement_ratio: float = (
        DEFAULT_MAX_PLAYER_CENTER_DISPLACEMENT_RATIO
    ),
    player_box_margin: float = DEFAULT_PLAYER_BOX_MARGIN,
    max_head_area_ratio: float = DEFAULT_MAX_HEAD_AREA_RATIO,
    max_head_center_y_ratio: float = DEFAULT_MAX_HEAD_CENTER_Y_RATIO,
) -> HeadAssociationOutcome:
    """Classify one head association without changing its acceptance policy."""

    target = _validated_box(player_box, "player_box")
    source_timestamp_ns = _non_negative_integer(
        source_timestamp_ns,
        "source_timestamp_ns",
    )
    min_containment = _finite_threshold(min_containment, "min_containment")
    min_player_overlap = _finite_threshold(
        min_player_overlap,
        "min_player_overlap",
    )
    max_player_center_displacement_ratio = _finite_threshold(
        max_player_center_displacement_ratio,
        "max_player_center_displacement_ratio",
    )
    player_box_margin = _finite_threshold(player_box_margin, "player_box_margin")
    max_head_area_ratio = _finite_threshold(
        max_head_area_ratio,
        "max_head_area_ratio",
    )
    max_head_center_y_ratio = _finite_threshold(
        max_head_center_y_ratio,
        "max_head_center_y_ratio",
    )
    if max_head_area_ratio <= 0.0:
        raise ValueError("max_head_area_ratio must be greater than zero")

    target_width = target[2] - target[0]
    target_height = target[3] - target[1]
    target_area = target_width * target_height
    target_center = (
        (target[0] + target[2]) * 0.5,
        (target[1] + target[3]) * 0.5,
    )
    margin_x = target_width * player_box_margin
    margin_y = target_height * player_box_margin
    expanded = (
        target[0] - margin_x,
        target[1] - margin_y,
        target[2] + margin_x,
        target[3] + margin_y,
    )

    decoded_head_present = False
    plausible_heads: list[
        tuple[int, HeadCandidate, Box, float, Point, float]
    ] = []
    for index, candidate in enumerate(candidates):
        if candidate.class_id != HEAD_CLASS_ID:
            continue
        decoded_head_present = True
        try:
            head_box = _validated_box(candidate.box, "candidate.box")
        except ValueError:
            continue
        head_area = _box_area(head_box)
        if head_area <= 0.0 or head_area / target_area > max_head_area_ratio:
            continue
        center = (
            (head_box[0] + head_box[2]) * 0.5,
            (head_box[1] + head_box[3]) * 0.5,
        )
        if not (
            expanded[0] <= center[0] <= expanded[2]
            and expanded[1] <= center[1] <= expanded[3]
        ):
            continue
        normalized_center_y = (center[1] - target[1]) / target_height
        if normalized_center_y > max_head_center_y_ratio:
            continue
        containment = _intersection_area(head_box, target) / head_area
        if containment < min_containment:
            continue
        if not (
            isfinite(float(candidate.confidence))
            and 0.0 <= float(candidate.confidence) <= 1.0
        ):
            continue
        plausible_heads.append(
            (index, candidate, head_box, head_area, center, containment)
        )

    if not decoded_head_present:
        return HeadAssociationOutcome(
            HeadLocalizationReason.NO_DECODED_HEAD_CANDIDATE,
            None,
        )
    if not plausible_heads:
        return HeadAssociationOutcome(
            HeadLocalizationReason.NO_PLAUSIBLE_HEAD,
            None,
        )

    # Any second head that independently passes the primary target's anatomy
    # gates makes the crop ambiguous.  A player box must not be allowed to
    # select between globally plausible heads after the fact.
    if len(plausible_heads) > 1:
        return HeadAssociationOutcome(
            HeadLocalizationReason.MULTIPLE_PLAUSIBLE_HEADS,
            None,
        )

    # A head candidate alone cannot establish that it belongs to the primary
    # detector's selected target.  Always require exactly one player instance
    # from this same inference pass to geometrically corroborate the primary
    # player.  This stays fail-closed in crowded crops even when only one head
    # happens to survive the anatomical gates.
    matching_players: list[tuple[int, Box]] = []
    for index, candidate in enumerate(candidates):
        if candidate.class_id != PLAYER_CLASS_ID:
            continue
        if not (
            isfinite(float(candidate.confidence))
            and 0.0 <= float(candidate.confidence) <= 1.0
        ):
            continue
        try:
            secondary_box = _validated_box(candidate.box, "candidate.box")
        except ValueError:
            continue
        secondary_area = _box_area(secondary_box)
        smaller_area = min(target_area, secondary_area)
        if smaller_area <= 0.0:
            continue
        intersection = _intersection_area(target, secondary_box)
        overlap = intersection / smaller_area
        secondary_center = (
            (secondary_box[0] + secondary_box[2]) * 0.5,
            (secondary_box[1] + secondary_box[3]) * 0.5,
        )
        normalized_center_displacement = hypot(
            (secondary_center[0] - target_center[0]) / target_width,
            (secondary_center[1] - target_center[1]) / target_height,
        )
        if (
            overlap >= min_player_overlap
            and normalized_center_displacement
            <= max_player_center_displacement_ratio
        ):
            matching_players.append((index, secondary_box))
    if not matching_players:
        return HeadAssociationOutcome(
            HeadLocalizationReason.NO_MATCHING_SECONDARY_PLAYER,
            None,
        )
    if len(matching_players) > 1:
        # Never rank competing secondary players. The only exception is an
        # exact pair of nearly fully nested multiscale duplicates, and even
        # that pair must independently support the sole plausible head. Any
        # shifted pair, partial support, or three-way ambiguity stays closed.
        sole_head = plausible_heads[0]
        if not _strict_nested_player_duplicate_pair(
            matching_players,
            target,
            head_box=sole_head[2],
            head_center=sole_head[4],
            min_head_containment=min_containment,
        ):
            return HeadAssociationOutcome(
                HeadLocalizationReason.MULTIPLE_MATCHING_SECONDARY_PLAYERS,
                None,
            )

    supporting_player_index, supporting_box = matching_players[0]
    supporting_area = _box_area(supporting_box)
    supported_heads = []
    for head in plausible_heads:
        head_area = head[3]
        center = head[4]
        if not (
            supporting_box[0] <= center[0] <= supporting_box[2]
            and supporting_box[1] <= center[1] <= supporting_box[3]
        ):
            continue
        containment = _intersection_area(head[2], supporting_box) / head_area
        if containment >= min_containment and supporting_area > 0.0:
            supported_heads.append(head)
    if not supported_heads:
        return HeadAssociationOutcome(
            HeadLocalizationReason.HEAD_UNSUPPORTED_BY_MATCHED_PLAYER,
            None,
        )
    plausible_heads = supported_heads

    index, selected, head_box, _area, center, containment = plausible_heads[0]
    return HeadAssociationOutcome(
        HeadLocalizationReason.LOCALIZED,
        HeadLocalization(
            point=center,
            source_timestamp_ns=source_timestamp_ns,
            confidence=float(selected.confidence),
            head_box=head_box,
            containment=containment,
            candidate_index=index,
            supporting_player_index=supporting_player_index,
        ),
    )


def associate_head_to_player(
    candidates: Sequence[HeadCandidate],
    player_box: Sequence[float],
    *,
    source_timestamp_ns: int,
    min_containment: float = DEFAULT_MIN_HEAD_CONTAINMENT,
    min_player_overlap: float = DEFAULT_MIN_PLAYER_OVERLAP,
    max_player_center_displacement_ratio: float = (
        DEFAULT_MAX_PLAYER_CENTER_DISPLACEMENT_RATIO
    ),
    player_box_margin: float = DEFAULT_PLAYER_BOX_MARGIN,
    max_head_area_ratio: float = DEFAULT_MAX_HEAD_AREA_RATIO,
    max_head_center_y_ratio: float = DEFAULT_MAX_HEAD_CENTER_Y_RATIO,
) -> HeadLocalization | None:
    """Preserve the public localization-only API over the typed outcome."""

    return associate_head_to_player_outcome(
        candidates,
        player_box,
        source_timestamp_ns=source_timestamp_ns,
        min_containment=min_containment,
        min_player_overlap=min_player_overlap,
        max_player_center_displacement_ratio=max_player_center_displacement_ratio,
        player_box_margin=player_box_margin,
        max_head_area_ratio=max_head_area_ratio,
        max_head_center_y_ratio=max_head_center_y_ratio,
    ).localization


class DirectHeadLocalizer:
    """Synchronous unit intended for one latest-frame-only async worker."""

    def __init__(
        self,
        infer: Callable[[np.ndarray], np.ndarray],
        *,
        confidence: float = DEFAULT_HEAD_CONFIDENCE,
        nms_iou: float = DEFAULT_NMS_IOU,
        crop_scale: float = DEFAULT_CROP_SCALE,
        min_containment: float = DEFAULT_MIN_HEAD_CONTAINMENT,
        max_head_center_y_ratio: float = DEFAULT_MAX_HEAD_CENTER_Y_RATIO,
    ) -> None:
        if not callable(infer):
            raise TypeError("infer must be callable")
        self._infer = infer
        self.confidence = _finite_threshold(confidence, "confidence")
        self.nms_iou = _finite_threshold(nms_iou, "nms_iou")
        self.min_containment = _finite_threshold(
            min_containment,
            "min_containment",
        )
        self.max_head_center_y_ratio = _finite_threshold(
            max_head_center_y_ratio,
            "max_head_center_y_ratio",
        )
        if not isfinite(float(crop_scale)) or float(crop_scale) < 1.0:
            raise ValueError("crop_scale must be finite and at least 1")
        self.crop_scale = float(crop_scale)

    def localize(
        self,
        frame: np.ndarray,
        player_box: Sequence[float],
        *,
        source_timestamp_ns: int,
    ) -> HeadLocalization | None:
        transform = plan_head_crop(
            frame.shape,
            player_box,
            crop_scale=self.crop_scale,
        )
        prepared = prepare_head_input(frame, transform)
        output = self._infer(prepared.tensor)
        candidates = decode_head_output(
            output,
            transform,
            confidence=self.confidence,
            nms_iou=self.nms_iou,
        )
        return associate_head_to_player(
            candidates,
            player_box,
            source_timestamp_ns=source_timestamp_ns,
            min_containment=self.min_containment,
            max_head_center_y_ratio=self.max_head_center_y_ratio,
        )
