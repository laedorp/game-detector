"""Canonical model-input dimension parsing and formatting.

Model tensor dimensions are always represented as ``(height, width)`` inside
the application.  A single integer remains accepted at public boundaries as a
backwards-compatible shorthand for a square input.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
import re
from typing import TypeAlias


InferenceSize: TypeAlias = tuple[int, int]
InferenceSizeLike: TypeAlias = int | Sequence[int]
YOLO_INPUT_STRIDE = 32


def normalize_inference_size(value: InferenceSizeLike) -> InferenceSize:
    """Return ``value`` as a positive ``(height, width)`` pair.

    Booleans and lossy numeric coercions are deliberately rejected.  This
    keeps a transposed or malformed tensor shape from surviving until runtime.
    """

    if isinstance(value, bool):
        raise TypeError(
            "inference_size must be a positive integer or (height, width) pair"
        )
    if isinstance(value, Integral):
        size = int(value)
        if size <= 0:
            raise ValueError("inference_size dimensions must be greater than zero")
        return size, size
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "inference_size must be a positive integer or (height, width) pair"
        )
    if len(value) != 2:
        raise ValueError("inference_size must contain exactly (height, width)")

    dimensions: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, Integral):
            raise TypeError("inference_size dimensions must be positive integers")
        parsed = int(dimension)
        if parsed <= 0:
            raise ValueError("inference_size dimensions must be greater than zero")
        dimensions.append(parsed)
    return dimensions[0], dimensions[1]


def parse_inference_size(value: str) -> InferenceSize:
    """Parse legacy ``N`` or explicit ``HEIGHTxWIDTH`` text."""

    text = str(value).strip()
    square = re.fullmatch(r"\d+", text)
    if square is not None:
        return normalize_inference_size(int(text))
    rectangular = re.fullmatch(r"(\d+)\s*[xX\N{MULTIPLICATION SIGN}]\s*(\d+)", text)
    if rectangular is None:
        raise ValueError("expected N or HEIGHTxWIDTH, for example 416 or 384x640")
    return normalize_inference_size(tuple(int(part) for part in rectangular.groups()))


def format_inference_size(value: InferenceSizeLike) -> str:
    """Format a square compatibly as ``N`` and a rectangle as ``HEIGHTxWIDTH``."""

    height, width = normalize_inference_size(value)
    return str(height) if height == width else f"{height}x{width}"


def compact_inference_size(value: InferenceSizeLike) -> int | InferenceSize:
    """Use the legacy scalar representation only when both dimensions match."""

    height, width = normalize_inference_size(value)
    return height if height == width else (height, width)


def validate_yolo_inference_size(value: InferenceSizeLike) -> InferenceSize:
    """Require a shape compatible with the runtime's supported YOLO exports."""

    height, width = normalize_inference_size(value)
    if height % YOLO_INPUT_STRIDE or width % YOLO_INPUT_STRIDE:
        raise ValueError(
            "inference size height and width must each be divisible by "
            f"{YOLO_INPUT_STRIDE}"
        )
    return height, width
