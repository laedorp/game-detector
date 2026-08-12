"""OpenCV-backed capture for camera devices and video files."""

from __future__ import annotations

import math
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from time import perf_counter_ns
from typing import Any, Mapping

try:
    import cv2
except ImportError:  # Keep the package importable for non-OpenCV tooling.
    cv2 = None  # type: ignore[assignment]

from .base import CaptureSource, FramePacket, _validate_timeout


class OpenCVCaptureSource(CaptureSource):
    """Capture from an OpenCV device or file.

    Integer sources are treated as live devices and use a producer thread with
    a one-packet latest-frame mailbox. String and ``Path`` sources default to
    synchronous, sequential file decoding. Pass ``live=True`` explicitly for a
    URL or other live source represented by a string.
    """

    def __init__(
        self,
        source: int | str | Path,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        backend: int | None = None,
        buffer_size: int = 1,
        live: bool | None = None,
        close_timeout: float = 2.0,
        pixel_format: str | None = None,
    ) -> None:
        if isinstance(source, bool):
            raise TypeError("source must be a device index or file path, not bool")
        if width is not None and width <= 0:
            raise ValueError("width must be positive")
        if height is not None and height <= 0:
            raise ValueError("height must be positive")
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("fps must be finite and positive")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if close_timeout < 0:
            raise ValueError("close_timeout must be non-negative")
        self._pixel_format = _normalized_pixel_format(pixel_format)

        self._source: int | str = str(source) if isinstance(source, Path) else source
        self._live = isinstance(source, int) if live is None else bool(live)
        description = _describe_source(self._source, self._live)
        super().__init__(description)

        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps
        self._backend = backend
        self._buffer_size = buffer_size
        self._close_timeout = close_timeout
        self._capture: Any | None = None
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._settings_lock = Lock()
        self._actual_settings: dict[str, object] = {}
        self._sequence = 0

    @property
    def actual_settings(self) -> Mapping[str, object]:
        with self._settings_lock:
            return dict(self._actual_settings)

    @property
    def is_live(self) -> bool:
        return self._live

    def start(self) -> None:
        if not self._start_once():
            return
        if cv2 is None:
            message = (
                "OpenCV capture requires the 'opencv-python' package; "
                "install it with: pip install opencv-python"
            )
            self._finish(message)
            raise RuntimeError(message)

        capture = None
        try:
            capture = self._open_capture()
            opened = capture is not None and capture.isOpened()
        except Exception as exc:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            message = f"Failed to open {self.description}: {exc}"
            self._finish(message)
            raise RuntimeError(message) from exc

        if not opened:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            message = f"Failed to open {self.description}"
            self._finish(message)
            raise RuntimeError(message)

        self._capture = capture
        if self._live:
            self._apply_live_requests(capture)
        self._store_actual_settings(capture)

        if not self._live:
            return

        self._thread = Thread(
            target=self._device_loop,
            name=f"capture-opencv-{self._source}",
            daemon=True,
        )
        try:
            self._thread.start()
        except Exception as exc:
            capture.release()
            self._capture = None
            message = f"Failed to start capture thread for {self.description}: {exc}"
            self._finish(message)
            raise RuntimeError(message) from exc

    def read(self, timeout: float | None = None) -> FramePacket | None:
        if self._live:
            return self._read_latest(timeout)

        self._require_started()
        _validate_timeout(timeout)  # File decoding is intentionally synchronous.
        if self._is_finished():
            return None

        capture = self._capture
        if capture is None:
            return None

        read_started_ns = perf_counter_ns()
        try:
            ok, image = capture.read()
        except Exception as exc:
            self._record_read_failure()
            self._finish(f"Read failed for {self.description}: {exc}")
            capture.release()
            self._capture = None
            return None
        read_completed_ns = perf_counter_ns()

        if not ok or image is None:
            error = self._early_file_failure(capture)
            if error is not None:
                self._record_read_failure()
            self._finish(error)
            capture.release()
            self._capture = None
            return None

        packet = FramePacket(
            image=image,
            sequence=self._sequence,
            read_started_ns=read_started_ns,
            read_completed_ns=read_completed_ns,
        )
        self._sequence += 1
        self._record_direct_delivery()
        return packet

    def close(self) -> None:
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            # release() is best effort and often unblocks a device read in flight.
            try:
                capture.release()
            except Exception:
                pass

        thread = self._thread
        if (
            thread is not None
            and thread is not current_thread()
            and thread.is_alive()
        ):
            thread.join(self._close_timeout)

        self._capture = None
        self._mark_closed()

    def _open_capture(self) -> Any:
        assert cv2 is not None
        if self._backend is None:
            return cv2.VideoCapture(self._source)
        return cv2.VideoCapture(self._source, self._backend)

    def _apply_live_requests(self, capture: Any) -> None:
        # The pixel format must be negotiated before size and rate.  A card's
        # advertised high frame rates usually exist only in a compressed or
        # subsampled mode, and drivers clamp the rate to whatever the *current*
        # format can sustain, so requesting 240 fps while still in the default
        # uncompressed mode silently yields 30 or 60.
        if self._pixel_format is not None:
            fourcc_id = getattr(cv2, "CAP_PROP_FOURCC", None)
            writer_fourcc = getattr(cv2, "VideoWriter_fourcc", None)
            if fourcc_id is not None and writer_fourcc is not None:
                try:
                    capture.set(fourcc_id, writer_fourcc(*self._pixel_format))
                except Exception:
                    pass

        requested = (
            ("CAP_PROP_FRAME_WIDTH", self._requested_width),
            ("CAP_PROP_FRAME_HEIGHT", self._requested_height),
            ("CAP_PROP_FPS", self._requested_fps),
            ("CAP_PROP_BUFFERSIZE", self._buffer_size),
            # Ask for BGR explicitly so both compressed (MJPG) and subsampled
            # (NV12, YUY2) modes are decoded by the backend.  Without it a
            # backend that hands back raw planar data would reach the detector
            # as a wrongly shaped array rather than an image.
            ("CAP_PROP_CONVERT_RGB", 1),
        )
        for property_name, value in requested:
            property_id = getattr(cv2, property_name, None)
            if value is None or property_id is None:
                continue
            try:
                capture.set(property_id, value)
            except Exception:
                # OpenCV properties are backend hints; unsupported hints should
                # not prevent capture from starting.
                continue

    def _store_actual_settings(self, capture: Any) -> None:
        settings: dict[str, object] = {
            "mode": "live" if self._live else "file",
            "source": self._source,
            "backend": _backend_name(capture),
            "width": _integer_property(capture, "CAP_PROP_FRAME_WIDTH"),
            "height": _integer_property(capture, "CAP_PROP_FRAME_HEIGHT"),
            "fps": _float_property(capture, "CAP_PROP_FPS"),
            "frame_count": _integer_property(capture, "CAP_PROP_FRAME_COUNT"),
            "buffer_size": _integer_property(capture, "CAP_PROP_BUFFERSIZE"),
            # Report what the driver actually granted.  A request is only a
            # hint, and silently running in the wrong format is the single
            # most common reason a capture card misses its rated frame rate.
            "pixel_format": _fourcc_text(capture),
            "requested_pixel_format": self._pixel_format,
        }
        with self._settings_lock:
            self._actual_settings = settings

    def _device_loop(self) -> None:
        capture = self._capture
        if capture is None:
            self._finish(f"Capture backend disappeared for {self.description}")
            return

        try:
            while not self._stop_event.is_set():
                read_started_ns = perf_counter_ns()
                try:
                    ok, image = capture.read()
                except Exception as exc:
                    if not self._stop_event.is_set():
                        self._record_read_failure()
                        self._finish(f"Read failed for {self.description}: {exc}")
                    break
                read_completed_ns = perf_counter_ns()

                if self._stop_event.is_set():
                    break
                if not ok or image is None:
                    self._record_read_failure()
                    self._finish(f"Read failed for {self.description}")
                    break

                packet = FramePacket(
                    image=image,
                    sequence=self._sequence,
                    read_started_ns=read_started_ns,
                    read_completed_ns=read_completed_ns,
                )
                self._sequence += 1
                self._publish_latest(packet)
        finally:
            try:
                capture.release()
            except Exception:
                pass
            if not self._stop_event.is_set():
                self._finish()

    def _early_file_failure(self, capture: Any) -> str | None:
        """Distinguish ordinary EOF from a decoder failure when possible."""

        frame_count = _float_property(capture, "CAP_PROP_FRAME_COUNT")
        position = _float_property(capture, "CAP_PROP_POS_FRAMES")
        if frame_count is None or position is None or frame_count <= 0:
            return None
        if position + 0.5 >= frame_count:
            return None
        return (
            f"Read failed before end of {self.description} "
            f"(frame {position:g} of {frame_count:g})"
        )


def _describe_source(source: int | str, live: bool) -> str:
    if isinstance(source, int):
        return f"camera device {source}"
    if live:
        return f"live OpenCV source {source!r}"
    return f"video file {source!r}"


# Formats worth naming.  NV12 and YUY2 are uncompressed, so their bandwidth is
# fixed by resolution and rate; MJPG is compressed and is usually the only way a
# USB card reaches its highest advertised rates.
KNOWN_PIXEL_FORMATS = ("MJPG", "NV12", "YUY2", "YUYV", "UYVY", "BGR3", "RGB3", "H264")


def _normalized_pixel_format(value: str | None) -> str | None:
    """Validate a FOURCC request without requiring it to be one we know."""

    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if len(text) != 4:
        raise ValueError(
            f"pixel format must be a four-character FOURCC code, got {value!r}"
        )
    if not text.isalnum():
        raise ValueError(f"pixel format must be alphanumeric, got {value!r}")
    return text


def _fourcc_text(capture: Any) -> str | None:
    """Decode the driver's active FOURCC into readable characters."""

    raw = _integer_property(capture, "CAP_PROP_FOURCC")
    if not raw:
        return None
    try:
        characters = [chr((raw >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
    except (TypeError, ValueError):
        return None
    text = "".join(characters).strip()
    return text if text.isprintable() and text else None


def _backend_name(capture: Any) -> str | None:
    try:
        return str(capture.getBackendName())
    except Exception:
        return None


def _float_property(capture: Any, property_name: str) -> float | None:
    property_id = getattr(cv2, property_name, None)
    if property_id is None:
        return None
    try:
        value = float(capture.get(property_id))
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None
    return value if math.isfinite(value) else None


def _integer_property(capture: Any, property_name: str) -> int | None:
    value = _float_property(capture, property_name)
    if value is None or value < 0:
        return None
    return int(round(value))
