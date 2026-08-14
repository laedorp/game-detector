"""Low-latency Windows desktop capture backed by DXcam and DXGI."""

from __future__ import annotations

from collections.abc import Callable, Mapping as MappingABC, Sequence
import math
from numbers import Integral
import re
import sys
from threading import Event, Lock, Thread, current_thread
from time import perf_counter_ns
from typing import Any, Mapping

import numpy as np

from .base import CaptureSource, FramePacket
from .screen_source import _normalise_region


if sys.platform == "win32":
    try:
        import dxcam
    except Exception as exc:  # Driver/COM initialization can fail at import time.
        dxcam = None  # type: ignore[assignment]
        _DXCAM_IMPORT_ERROR: str | None = str(exc)
    else:
        _DXCAM_IMPORT_ERROR = None
else:
    dxcam = None  # type: ignore[assignment]
    _DXCAM_IMPORT_ERROR = None


_OUTPUT_RECORD = re.compile(r"Device\[(\d+)]\s+Output\[(\d+)]")


class DXCamCaptureSource(CaptureSource):
    """Capture one Windows monitor or one global desktop rectangle via DXGI.

    DXcam owns the DXGI producer thread.  A small consumer thread transfers
    only its newest frame into :class:`CaptureSource`'s one-packet mailbox, so
    slow inference cannot turn captured frames into a latency-growing queue.
    """

    def __init__(
        self,
        *,
        monitor: int = 1,
        region: Sequence[int] | Mapping[str, int] | None = None,
        fps: float = 60.0,
        startup_timeout: float = 5.0,
        close_timeout: float = 2.0,
        _dxcam_module: Any | None = None,
        _monitor_bounds_provider: Callable[[int], tuple[int, int, int, int]] | None = None,
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

        self._monitor = int(monitor)
        self._region = _normalise_region(region) if region is not None else None
        self._target_fps = float(fps)
        self._startup_timeout = float(startup_timeout)
        self._close_timeout = float(close_timeout)
        self._dxcam = dxcam if _dxcam_module is None else _dxcam_module
        self._monitor_bounds_provider = (
            _mss_monitor_bounds
            if _monitor_bounds_provider is None
            else _monitor_bounds_provider
        )

        if self._region is None:
            description = f"Windows screen monitor {self._monitor}"
        else:
            left, top, width, height = self._region
            description = f"Windows screen region {width}x{height}+{left}+{top}"
        super().__init__(description)

        self._stop_event = Event()
        self._startup_event = Event()
        self._startup_error: str | None = None
        self._thread: Thread | None = None
        self._shutdown_thread: Thread | None = None
        self._shutdown_lock = Lock()
        self._camera: Any | None = None
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
        if self._dxcam is None:
            detail = f" ({_DXCAM_IMPORT_ERROR})" if _DXCAM_IMPORT_ERROR else ""
            message = (
                "Windows DXGI screen capture requires the optional 'dxcam' "
                f"package; install it with: pip install dxcam{detail}"
            )
            self._finish(message)
            raise RuntimeError(message)

        cameras_to_release: list[Any] = []
        try:
            target = self._target_global_bounds()
            camera, device_index, output_index, output_bounds = self._select_camera(
                target,
                cameras_to_release,
            )
            self._camera = camera
            local_region = _relative_region(target, output_bounds)
            left, top, right, bottom = target
            with self._settings_lock:
                self._actual_settings = {
                    "backend": "dxcam-dxgi",
                    "monitor": None if self._region is not None else self._monitor,
                    "device_index": device_index,
                    "output_index": output_index,
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top,
                    "fps": self._target_fps,
                    "buffer_frames": 2,
                }

            # DXcam's ring is deliberately kept at its minimum supported size.
            # video_mode=False means unchanged/duplicate desktop frames are not
            # manufactured just to satisfy the requested rate.
            camera.start(
                region=local_region,
                target_fps=max(1, round(self._target_fps)),
                video_mode=False,
            )
        except Exception as exc:
            self._release_cameras(cameras_to_release)
            message = f"Failed to start {self.description}: {exc}"
            self._finish(message)
            raise RuntimeError(message) from exc
        finally:
            # Non-selected probes are no longer needed after geometry matching.
            self._release_cameras(cameras_to_release, except_camera=self._camera)

        self._thread = Thread(
            target=self._capture_loop,
            name="capture-dxcam",
            daemon=True,
        )
        try:
            self._thread.start()
        except Exception as exc:
            self._shutdown_camera()
            message = f"Failed to start DXcam consumer thread: {exc}"
            self._finish(message)
            raise RuntimeError(message) from exc

        if not self._startup_event.wait(self._startup_timeout):
            message = f"Timed out while starting {self.description}"
            self._stop_event.set()
            self._shutdown_camera()
            self._finish(message)
            self._join_thread()
            raise RuntimeError(message)
        if self._startup_error is not None:
            self._shutdown_camera()
            self._join_thread()
            raise RuntimeError(self._startup_error)

    def read(self, timeout: float | None = None) -> FramePacket | None:
        return self._read_latest(timeout)

    def close(self) -> None:
        self._begin_close()
        self._stop_event.set()
        self._shutdown_camera()
        self._join_thread()
        thread = self._thread
        shutdown_thread = self._shutdown_thread
        if thread is not None and thread.is_alive():
            self._record_close_timeout(thread.name)
            return
        if shutdown_thread is not None and shutdown_thread.is_alive():
            self._record_close_timeout(shutdown_thread.name)
            return
        self._camera = None
        self._mark_closed()

    def _capture_loop(self) -> None:
        try:
            camera = self._camera
            if camera is None:
                raise RuntimeError("DXcam camera was not initialized")

            while not self._stop_event.is_set():
                wait_started_ns = perf_counter_ns()
                latest = camera.get_latest_frame(copy=True, with_timestamp=True)
                returned_ns = perf_counter_ns()
                if latest is None:
                    if self._stop_event.is_set():
                        break
                    if not bool(getattr(camera, "is_capturing", True)):
                        raise RuntimeError("DXcam capture stopped unexpectedly")
                    continue

                image, timestamp_seconds = latest
                pixels = np.asarray(image)
                if pixels.ndim != 3 or pixels.shape[2] != 3:
                    raise RuntimeError(
                        f"DXcam returned an unexpected BGR frame shape: {pixels.shape}"
                    )

                # copy=True above gives this consumer ownership.  Only force a
                # second copy if the installed DXcam processor returned a
                # strided view despite that contract.
                if not pixels.flags.c_contiguous:
                    pixels = np.ascontiguousarray(pixels)

                presented_ns = _timestamp_ns(timestamp_seconds)
                if presented_ns is None or presented_ns > returned_ns:
                    read_started_ns = wait_started_ns
                else:
                    # DXGI LastPresentTime uses the same monotonic performance
                    # counter epoch as perf_counter on Windows. A presentation
                    # commonly happens after this consumer begins waiting; use
                    # the presentation itself so freshness is frame age rather
                    # than the blocking call's duration.
                    read_started_ns = presented_ns

                packet = FramePacket(
                    image=pixels,
                    sequence=self._sequence,
                    read_started_ns=read_started_ns,
                    read_completed_ns=returned_ns,
                )
                self._sequence += 1
                self._publish_latest(packet)
                if not self._startup_event.is_set():
                    self._startup_event.set()
        except Exception as exc:
            if not self._stop_event.is_set():
                self._record_read_failure()
                message = f"Read failed for {self.description}: {exc}"
                if not self._startup_event.is_set():
                    self._startup_error = message
                self._finish(message)
        finally:
            self._startup_event.set()
            if self._stop_event.is_set():
                shutdown_thread = self._shutdown_thread
                if shutdown_thread is None or not shutdown_thread.is_alive():
                    self._complete_close_from_worker()
            else:
                self._finish()

    def _target_global_bounds(self) -> tuple[int, int, int, int]:
        if self._region is not None:
            left, top, width, height = self._region
            return left, top, left + width, top + height
        return self._monitor_bounds_provider(self._monitor)

    def _select_camera(
        self,
        target: tuple[int, int, int, int],
        probes: list[Any],
    ) -> tuple[Any, int, int, tuple[int, int, int, int]]:
        assert self._dxcam is not None
        records = _output_records(self._dxcam)
        errors: list[str] = []
        for device_index, output_index in records:
            try:
                camera = self._dxcam.create(
                    device_idx=device_index,
                    output_idx=output_index,
                    output_color="BGR",
                    max_buffer_len=2,
                    backend="dxgi",
                    processor_backend="cv2",
                )
                probes.append(camera)
                bounds = _camera_global_bounds(camera)
            except Exception as exc:
                errors.append(f"device {device_index} output {output_index}: {exc}")
                continue

            if _contains(bounds, target):
                return camera, device_index, output_index, bounds

        detail = f" ({'; '.join(errors)})" if errors else ""
        left, top, right, bottom = target
        raise ValueError(
            "the requested desktop rectangle "
            f"({left},{top})-({right},{bottom}) is not contained within one "
            f"DXGI output{detail}"
        )

    def _shutdown_camera(self) -> None:
        """Start one bounded, asynchronous native-camera shutdown.

        DXcam 0.3.0 can wait up to ten seconds inside ``stop()``. Running it
        on a helper keeps our public ``close_timeout`` meaningful and avoids a
        second ten-second wait through ``release()``. The helper owns the
        camera until both native cleanup and the consumer thread finish.
        """

        camera = self._camera
        if camera is None:
            return
        with self._shutdown_lock:
            thread = self._shutdown_thread
            if thread is None:
                thread = Thread(
                    target=self._camera_shutdown_loop,
                    args=(camera,),
                    name="capture-dxcam-shutdown",
                    daemon=True,
                )
                self._shutdown_thread = thread
                thread.start()
        if thread is not current_thread() and thread.is_alive():
            thread.join(self._close_timeout)

    def _camera_shutdown_loop(self, camera: Any) -> None:
        stop_succeeded = False
        try:
            camera.stop()
            stop_succeeded = True
        except Exception:
            # A timed-out DXcam stop still owns live capture resources. Do not
            # immediately call release(), which invokes stop() a second time.
            pass
        if stop_succeeded:
            try:
                camera.release()
            except Exception:
                pass

        capture_thread = self._thread
        if (
            capture_thread is not None
            and capture_thread is not current_thread()
            and capture_thread.is_alive()
        ):
            # This is a daemon cleanup worker, not the caller. Once native stop
            # succeeds, wait for the app consumer before declaring closure.
            capture_thread.join()
        self._camera = None
        self._complete_close_from_worker()

    def _release_cameras(
        self,
        cameras: Sequence[Any],
        *,
        except_camera: Any | None = None,
    ) -> None:
        for camera in cameras:
            if camera is except_camera:
                continue
            try:
                camera.release()
            except Exception:
                pass

    def _join_thread(self) -> None:
        thread = self._thread
        if (
            thread is not None
            and thread is not current_thread()
            and thread.is_alive()
        ):
            thread.join(self._close_timeout)


def _output_records(dxcam_module: Any) -> list[tuple[int, int]]:
    """Return all advertised DXGI device/output pairs in stable order."""

    try:
        text = str(dxcam_module.output_info())
    except Exception:
        text = ""
    records = [
        (int(device), int(output))
        for device, output in _OUTPUT_RECORD.findall(text)
    ]
    # Old DXcam versions expose create() but not parseable output_info().
    return list(dict.fromkeys(records)) or [(0, 0)]


def _camera_global_bounds(camera: Any) -> tuple[int, int, int, int]:
    """Read the desktop-space DXGI output rectangle from a camera."""

    explicit = getattr(camera, "output_bounds", None)
    if explicit is not None:
        values = tuple(int(value) for value in explicit)
        if len(values) == 4:
            return values  # type: ignore[return-value]

    output = getattr(camera, "_output", None)
    desc = getattr(output, "desc", None)
    coordinates = getattr(desc, "DesktopCoordinates", None)
    if coordinates is None:
        # A primary output without private geometry metadata is necessarily at
        # local origin.  This is sufficient for old DXcam on one-display PCs;
        # non-primary/global regions fail closed and use the MSS fallback.
        width = int(getattr(camera, "width"))
        height = int(getattr(camera, "height"))
        return 0, 0, width, height
    return (
        int(coordinates.left),
        int(coordinates.top),
        int(coordinates.right),
        int(coordinates.bottom),
    )


def _mss_monitor_bounds(monitor: int) -> tuple[int, int, int, int]:
    """Resolve a monitor using exactly the same numbering MSS exposes today."""

    from . import screen_source

    if screen_source.mss is None:
        raise RuntimeError("monitor discovery requires the 'mss' package")
    screenshotter = screen_source.mss.mss()
    try:
        monitors = screenshotter.monitors
        if monitor >= len(monitors):
            available = max(0, len(monitors) - 1)
            raise ValueError(
                f"monitor {monitor} is unavailable; "
                f"mss reports {available} physical monitor(s)"
            )
        selected: MappingABC[str, Any] = monitors[monitor]
        left = int(selected["left"])
        top = int(selected["top"])
        width = int(selected["width"])
        height = int(selected["height"])
        return left, top, left + width, top + height
    finally:
        try:
            screenshotter.close()
        except Exception:
            pass


def _contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _relative_region(
    target: tuple[int, int, int, int],
    output: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        target[0] - output[0],
        target[1] - output[1],
        target[2] - output[0],
        target[3] - output[1],
    )


def _timestamp_ns(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return round(seconds * 1_000_000_000)
