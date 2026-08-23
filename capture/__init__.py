"""Frame capture sources for live devices, video files, and the desktop."""

from .base import CaptureSource, CaptureStats, FramePacket
from .desktop_source import DesktopCaptureSource
from .dxcam_source import DXCamCaptureSource
from .opencv_source import OpenCVCaptureSource
from .screen_source import ScreenCaptureSource

__all__ = [
    "CaptureSource",
    "CaptureStats",
    "DesktopCaptureSource",
    "DXCamCaptureSource",
    "FramePacket",
    "OpenCVCaptureSource",
    "ScreenCaptureSource",
]
