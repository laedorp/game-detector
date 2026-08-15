"""Replaceable object-detection backends."""

from .base import (
    DependencyError,
    Detector,
    DetectorError,
    DeviceNotAvailableError,
    ModelError,
    OutputDecodeError,
)
from .openvino_yolo import OpenVINOYoloDetector
from .detail_pass import (
    CROSS_PASS_DUPLICATE_IOU,
    DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT,
    DetailPassPlan,
    DetailPassStats,
    merge_cross_pass_detections,
    plan_detail_pass,
)
from .postprocess import FrameTransformLike, class_aware_nms, decode_yolo_output
from .types import Detection

__all__ = [
    "DependencyError",
    "CROSS_PASS_DUPLICATE_IOU",
    "DETAIL_UNMATCHED_MAX_REFERENCE_HEIGHT",
    "Detection",
    "DetailPassPlan",
    "DetailPassStats",
    "Detector",
    "DetectorError",
    "DeviceNotAvailableError",
    "FrameTransformLike",
    "ModelError",
    "OpenVINOYoloDetector",
    "OutputDecodeError",
    "class_aware_nms",
    "decode_yolo_output",
    "merge_cross_pass_detections",
    "plan_detail_pass",
]
