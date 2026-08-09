"""Low-latency screen capture backed by the optional python-mss package."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence
import math
import os
from threading import Event, Lock, Thread, current_thread
from time import perf_counter_ns
from typing import Any, Mapping

import numpy as np

from .base import CaptureSource, FramePacket

try:
    import mss
except ImportError:  # Screen capture is an optional runtime feature.
    mss = None  # type: ignore[assignment]


class ScreenCaptureSource(CaptureSource):
    """Capture a monitor or rectangular desktop region into a latest mailbox."""

    def __init__(
        self,
        *,
        monitor: int = 1,
        region: Sequence[int] | Mapping[str, int] | None = None,
        fps: float = 30.0,
        startup_timeout: float = 5.0,
        close_timeout: float = 2.0,
    ) -> None:
        if isinstance(monitor, bool) or monitor < 0:
            raise ValueError("monitor must be a non-negative integer")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        if close_timeout < 0:
            raise ValueError("close_timeout must be non-negative")

        self._monitor = int(monitor)
        self._region = _normalise_region(region) if region is not None else None
        self._target_fps = float(fps)
        self._startup_timeout = float(startup_timeout)
        self._close_timeout = float(close_timeout)

        if self._region is None:
            description = f"screen monitor {self._monitor}"
        else:
            left, top, width, height = self._region
            description = f"screen region {width}x{height}+{left}+{top}"
        super().__init__(description)

        self._stop_event = Event()
        self._startup_event = Event()
        self._startup_error: str | None = None
        self._thread: Thread | None = None
        self._settings_lock = Lock()
        self._actual_settings: dict[str, object] = {}
        self._sequence = 0

    @property
    def actual_settings(self) -> Mapping[str, object]:
        with self._settings_lock:
            return dict(self._actual_settings)

    def start(self) -> None:
        if not self._start_once():
            return
        if mss is None:
            message = (
                "Screen capture requires the optional 'mss' package; "
                "install it with: pip install mss"
            )
            self._finish(message)
            raise RuntimeError(message)
        if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            message = (
                "Moonlight screen capture currently requires an Xorg/X11 session. "
                "Log into Xorg instead of Wayland, then try again."
            )
            self._finish(message)
            raise RuntimeError(message)

        self._thread = Thread(
            target=self._capture_loop,
            name="capture-mss",
            daemon=True,
        )
        try:
            self._thread.start()
        except Exception as exc:
            message = f"Failed to start screen capture thread: {exc}"
            self._finish(message)
            raise RuntimeError(message) from exc

        if not self._startup_event.wait(self._startup_timeout):
            message = f"Timed out while starting {self.description}"
            self._stop_event.set()
            self._finish(message)
            self._join_thread()
            raise RuntimeError(message)
        if self._startup_error is not None:
            self._join_thread()
            raise RuntimeError(self._startup_error)

    def read(self, timeout: float | None = None) -> FramePacket | None:
        return self._read_latest(timeout)

    def close(self) -> None:
        self._stop_event.set()
        self._mark_closed()
        self._join_thread()

    def _capture_loop(self) -> None:
        screenshotter: Any | None = None
        try:
            assert mss is not None
            screenshotter = mss.mss()
            target = self._resolve_target(screenshotter)
            with self._settings_lock:
                self._actual_settings = {
                    "backend": "mss",
                    "monitor": None if self._region is not None else self._monitor,
                    "left": target["left"],
                    "top": target["top"],
                    "width": target["width"],
                    "height": target["height"],
                    "fps": self._target_fps,
                    "display": os.environ.get("DISPLAY"),
                    "session_type": os.environ.get("XDG_SESSION_TYPE"),
                }

            period_ns = max(1, round(1_000_000_000 / self._target_fps))
            next_capture_ns = perf_counter_ns()
            while not self._stop_event.is_set():
                now_ns = perf_counter_ns()
                if now_ns < next_capture_ns:
                    if self._stop_event.wait((next_capture_ns - now_ns) / 1e9):
                        break

                read_started_ns = perf_counter_ns()
                try:
                    shot = screenshotter.grab(target)
                    pixels = np.asarray(shot)
                    if pixels.ndim != 3 or pixels.shape[2] < 3:
                        raise RuntimeError(
                            f"mss returned an unexpected frame shape: {pixels.shape}"
                        )
                    # MSS supplies BGRA. Dropping alpha and making the slice
                    # contiguous yields the BGR layout expected by OpenCV.
                    image = np.ascontiguousarray(pixels[:, :, :3])
                except Exception as exc:
                    if not self._stop_event.is_set():
                        self._record_read_failure()
                        self._finish(f"Read failed for {self.description}: {exc}")
                    break
                read_completed_ns = perf_counter_ns()

                packet = FramePacket(
                    image=image,
                    sequence=self._sequence,
                    read_started_ns=read_started_ns,
                    read_completed_ns=read_completed_ns,
                )
                self._sequence += 1
                self._publish_latest(packet)
                if not self._startup_event.is_set():
                    self._startup_event.set()

                next_capture_ns += period_ns
                now_ns = perf_counter_ns()
                if next_capture_ns < now_ns:
                    next_capture_ns = now_ns
        except Exception as exc:
            if not self._startup_event.is_set():
                self._startup_error = f"Failed to start {self.description}: {exc}"
                self._finish(self._startup_error)
            elif not self._stop_event.is_set():
                self._record_read_failure()
                self._finish(f"Screen capture failed: {exc}")
        finally:
            self._startup_event.set()
            if screenshotter is not None:
                try:
                    screenshotter.close()
                except Exception:
                    pass
            if not self._stop_event.is_set():
                self._finish()

    def _resolve_target(self, screenshotter: Any) -> dict[str, int]:
        if self._region is not None:
            left, top, width, height = self._region
            return {"left": left, "top": top, "width": width, "height": height}

        monitors = screenshotter.monitors
        if self._monitor >= len(monitors):
            available = max(0, len(monitors) - 1)
            raise ValueError(
                f"monitor {self._monitor} is unavailable; "
                f"mss reports {available} physical monitor(s)"
            )
        selected = monitors[self._monitor]
        return {
            "left": int(selected["left"]),
            "top": int(selected["top"]),
            "width": int(selected["width"]),
            "height": int(selected["height"]),
        }

    def _join_thread(self) -> None:
        thread = self._thread
        if (
            thread is not None
            and thread is not current_thread()
            and thread.is_alive()
        ):
            thread.join(self._close_timeout)


def _normalise_region(
    region: Sequence[int] | Mapping[str, int],
) -> tuple[int, int, int, int]:
    if isinstance(region, MappingABC):
        try:
            left = region["left"] if "left" in region else region["x"]
            top = region["top"] if "top" in region else region["y"]
            width = region["width"] if "width" in region else region["w"]
            height = region["height"] if "height" in region else region["h"]
            values = (left, top, width, height)
        except KeyError as exc:
            raise ValueError(
                "region mapping requires left/top/width/height (or x/y/w/h)"
            ) from exc
    else:
        if isinstance(region, (str, bytes)) or len(region) != 4:
            raise ValueError("region must contain x, y, width, and height")
        values = tuple(region)

    if any(isinstance(value, bool) for value in values):
        raise ValueError("region values must be integers")
    try:
        left, top, width, height = (int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("region values must be integers") from exc
    if width <= 0 or height <= 0:
        raise ValueError("region width and height must be positive")
    return left, top, width, height
