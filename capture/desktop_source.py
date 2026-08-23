"""Platform-preferred screen capture with a transparent MSS fallback."""

from __future__ import annotations

from collections.abc import Callable, Mapping as MappingABC, Sequence
import math
from numbers import Integral
import sys
from typing import Mapping

from .base import CaptureSource, CaptureStats, FramePacket
from .dxcam_source import DXCamCaptureSource
from .screen_source import ScreenCaptureSource, _normalise_region


class DesktopCaptureSource(CaptureSource):
    """Prefer DXcam on Windows and retain MSS everywhere as a safe fallback."""

    def __init__(
        self,
        *,
        monitor: int = 1,
        region: Sequence[int] | MappingABC[str, int] | None = None,
        fps: float = 60.0,
        startup_timeout: float = 5.0,
        close_timeout: float = 2.0,
        _platform: str | None = None,
        _dxcam_factory: Callable[..., CaptureSource] = DXCamCaptureSource,
        _mss_factory: Callable[..., CaptureSource] = ScreenCaptureSource,
    ) -> None:
        if (
            isinstance(monitor, bool)
            or not isinstance(monitor, Integral)
            or monitor < 0
        ):
            raise ValueError("monitor must be a non-negative integer")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be finite and positive")
        if not math.isfinite(close_timeout) or close_timeout < 0:
            raise ValueError("close_timeout must be finite and non-negative")

        # Region is normalized once so both candidates receive identical
        # desktop-space semantics.
        self._monitor = int(monitor)
        self._region = _normalise_region(region) if region is not None else None
        self._fps = float(fps)
        self._startup_timeout = float(startup_timeout)
        self._close_timeout = float(close_timeout)
        self._platform = sys.platform if _platform is None else _platform
        self._dxcam_factory = _dxcam_factory
        self._mss_factory = _mss_factory
        self._source: CaptureSource | None = None
        self._fallback_reason: str | None = None

        if self._region is None:
            description = f"screen monitor {monitor}"
        else:
            left, top, width, height = self._region
            description = f"screen region {width}x{height}+{left}+{top}"
        super().__init__(description)

    @property
    def actual_settings(self) -> Mapping[str, object]:
        source = self._source
        if source is None:
            return {}
        settings = dict(source.actual_settings)
        if self._fallback_reason is not None:
            settings["preferred_backend"] = "dxcam-dxgi"
            settings["fallback_reason"] = self._fallback_reason
        return settings

    @property
    def ended(self) -> bool:
        source = self._source
        return source.ended if source is not None else super().ended

    @property
    def error(self) -> str | None:
        source = self._source
        return source.error if source is not None else super().error

    @property
    def stats(self) -> CaptureStats:
        source = self._source
        return source.stats if source is not None else super().stats

    def start(self) -> None:
        if not self._start_once():
            return

        if self._platform == "win32":
            accelerated = self._new_source(self._dxcam_factory)
            try:
                accelerated.start()
            except Exception as exc:
                self._fallback_reason = str(exc)
                try:
                    accelerated.close()
                except Exception:
                    pass
            else:
                self._source = accelerated
                return

        fallback = self._new_source(self._mss_factory)
        try:
            fallback.start()
        except Exception as exc:
            try:
                fallback.close()
            except Exception:
                pass
            if self._fallback_reason is None:
                message = str(exc)
            else:
                message = (
                    "DXcam screen capture failed and MSS fallback also failed. "
                    f"DXcam: {self._fallback_reason}; MSS: {exc}"
                )
            self._finish(message)
            raise RuntimeError(message) from exc
        self._source = fallback

    def read(self, timeout: float | None = None) -> FramePacket | None:
        self._require_started()
        source = self._source
        if source is None:
            return None
        return source.read(timeout)

    def peek_latest(self) -> FramePacket | None:
        self._require_started()
        source = self._source
        if source is None:
            return None
        return source.peek_latest()

    def close(self) -> None:
        self._begin_close()
        source = self._source
        if source is not None:
            source.close()
            if not source.ended:
                self._record_close_timeout(source.description)
                return
        self._mark_closed()

    def _new_source(self, factory: Callable[..., CaptureSource]) -> CaptureSource:
        return factory(
            monitor=self._monitor,
            region=self._region,
            fps=self._fps,
            startup_timeout=self._startup_timeout,
            close_timeout=self._close_timeout,
        )
