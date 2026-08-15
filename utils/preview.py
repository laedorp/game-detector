"""Paced OpenCV preview support with a backend-safe threading policy.

OpenCV HighGUI has backend-specific thread affinity.  In particular, Qt on
Linux requires window calls on the process main thread.  The factory below
therefore uses an inline implementation unless the native Windows HighGUI
backend explicitly identifies itself as ``WIN32``.  Both implementations
share the same lifecycle and event-polling interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from threading import Event, Lock, Thread
from typing import Any


class PreviewPacer:
    """Select at most ``fps`` frames per second for preview submission."""

    def __init__(self, fps: float) -> None:
        value = float(fps)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("preview fps must be finite and greater than zero")
        self.interval_ns = max(1, round(1_000_000_000 / value))
        self._next_due_ns = 0

    def should_render(self, now_ns: int) -> bool:
        now = int(now_ns)
        if now < self._next_due_ns:
            return False
        # Schedule from the actual submission decision instead of trying to
        # catch up after a slow frame. Catch-up would create a display burst.
        self._next_due_ns = now + self.interval_ns
        return True


@dataclass(frozen=True, slots=True)
class PreviewStats:
    """Small diagnostic snapshot for the latest-only mailbox."""

    submitted_frames: int
    displayed_frames: int
    replaced_frames: int


class InlinePreviewWindow:
    """Run HighGUI on the caller thread for thread-affine GUI backends."""

    mode = "inline"

    def __init__(self, cv2_module: Any, window_name: str) -> None:
        self._cv2 = cv2_module
        self.window_name = str(window_name)
        self._started = False
        self._closed = False
        self._invisible_polls = 0
        self._visibility_confirmed = False
        self._submitted_frames = 0
        self._displayed_frames = 0

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Preview window has already been started")
        try:
            self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)
        except self._cv2.error as exc:
            raise RuntimeError(
                f"Could not create the OpenCV preview window: {exc}. "
                "Use --no-preview on a headless session."
            ) from exc
        self._started = True
        # Give the native backend one startup event turn. This establishes
        # visibility on ordinary backends and lets the first later static poll
        # recognize a real close without requiring a second 250 ms timeout.
        self.poll()

    def submit(self, frame: Any) -> bool:
        """Submit one paced frame on the caller thread."""

        if not self.should_continue():
            return False
        try:
            self._cv2.imshow(self.window_name, frame)
        except self._cv2.error as exc:
            raise RuntimeError(
                f"OpenCV preview failed: {exc}. Use --no-preview on a "
                "headless session."
            ) from exc
        self._submitted_frames += 1
        self._displayed_frames += 1
        return self.poll()

    def poll(self) -> bool:
        """Service key/window events, including when capture is static."""

        if not self.should_continue():
            return False
        try:
            key = self._cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self._closed = True
                return False
            try:
                visible = self._cv2.getWindowProperty(
                    self.window_name,
                    self._cv2.WND_PROP_VISIBLE,
                )
            except self._cv2.error:
                # Some Linux GUI backends do not implement this property.
                visible = 1
        except self._cv2.error as exc:
            raise RuntimeError(
                f"OpenCV preview failed: {exc}. Use --no-preview on a "
                "headless session."
            ) from exc

        if visible < 1:
            # Some backends briefly report a new, not-yet-painted window as
            # invisible. Before visibility has ever been confirmed, require
            # two consecutive polls; after confirmation, one poll is a close.
            self._invisible_polls += 1
            if self._visibility_confirmed or self._invisible_polls >= 2:
                self._closed = True
                return False
        else:
            self._visibility_confirmed = True
            self._invisible_polls = 0
        return True

    def should_continue(self) -> bool:
        return self._started and not self._closed

    def raise_if_failed(self) -> None:
        # Inline failures are raised synchronously by start/submit/poll.
        return None

    def stop(self, timeout_s: float | None = None) -> bool:
        del timeout_s
        if not self._started:
            return True
        self._closed = True
        try:
            self._cv2.destroyWindow(self.window_name)
        except self._cv2.error:
            pass
        self._started = False
        return True

    @property
    def stats(self) -> PreviewStats:
        return PreviewStats(
            submitted_frames=self._submitted_frames,
            displayed_frames=self._displayed_frames,
            replaced_frames=0,
        )


class AsyncPreviewWindow:
    """Own an OpenCV HighGUI window on a bounded-lifecycle worker thread.

    ``submit`` copies the input before publishing it.  At most one unpublished
    frame is retained; a newer submission replaces the older one.  This makes
    producer cost explicit and bounded by one frame copy, independent of the
    time the OS window manager takes to repaint.
    """

    mode = "threaded"

    def __init__(
        self,
        cv2_module: Any,
        window_name: str,
        *,
        startup_timeout_s: float = 2.0,
        stop_timeout_s: float = 1.0,
        idle_poll_s: float = 0.01,
    ) -> None:
        if startup_timeout_s <= 0:
            raise ValueError("preview startup timeout must be greater than zero")
        if stop_timeout_s <= 0:
            raise ValueError("preview stop timeout must be greater than zero")
        if idle_poll_s <= 0:
            raise ValueError("preview poll interval must be greater than zero")

        self._cv2 = cv2_module
        self.window_name = str(window_name)
        self.startup_timeout_s = float(startup_timeout_s)
        self.stop_timeout_s = float(stop_timeout_s)
        self.idle_poll_s = float(idle_poll_s)

        self._lock = Lock()
        self._ready = Event()
        self._stopped = Event()
        self._stop_requested = Event()
        self._wake = Event()
        self._close_requested = Event()
        self._thread: Thread | None = None
        self._latest_frame: Any | None = None
        self._error: RuntimeError | None = None
        self._submitted_frames = 0
        self._displayed_frames = 0
        self._replaced_frames = 0

    def start(self) -> None:
        """Create the worker and wait a bounded time for window creation."""

        with self._lock:
            if self._thread is not None:
                raise RuntimeError("Preview window has already been started")
            thread = Thread(
                target=self._run,
                name="proaim-preview",
                daemon=True,
            )
            self._thread = thread
            thread.start()

        if not self._ready.wait(self.startup_timeout_s):
            self._stop_requested.set()
            self._wake.set()
            raise RuntimeError(
                "Timed out while creating the OpenCV preview window. "
                "Use --no-preview on a headless session."
            )
        self.raise_if_failed()
        if self._stopped.is_set():
            raise RuntimeError("The OpenCV preview window closed during startup")

    def submit(self, frame: Any) -> bool:
        """Copy and publish ``frame`` without waiting for a repaint.

        Returns ``False`` when the window has already closed or is stopping.
        The copy deliberately happens on the producer: capture buffers and
        overlay images may be reused or mutated immediately after this call.
        """

        self.raise_if_failed()
        if self._close_requested.is_set() or self._stop_requested.is_set():
            return False

        owned_frame = frame.copy()
        with self._lock:
            if self._close_requested.is_set() or self._stop_requested.is_set():
                return False
            if self._latest_frame is not None:
                self._replaced_frames += 1
            self._latest_frame = owned_frame
            self._submitted_frames += 1
        self._wake.set()
        return True

    def should_continue(self) -> bool:
        """Return whether the producer should continue, propagating failures."""

        self.raise_if_failed()
        return not self._close_requested.is_set() and not self._stopped.is_set()

    def poll(self) -> bool:
        """The worker polls HighGUI; the producer only checks its signals."""

        return self.should_continue()

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error

    def stop(self, timeout_s: float | None = None) -> bool:
        """Request shutdown and wait no longer than ``timeout_s``.

        A stuck native GUI backend cannot be forcefully killed safely.  The
        worker is a daemon, so returning ``False`` preserves bounded shutdown
        for the application while accurately reporting the condition.
        """

        self._stop_requested.set()
        self._wake.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(self.stop_timeout_s if timeout_s is None else max(0.0, timeout_s))
        return not thread.is_alive()

    @property
    def stats(self) -> PreviewStats:
        with self._lock:
            return PreviewStats(
                submitted_frames=self._submitted_frames,
                displayed_frames=self._displayed_frames,
                replaced_frames=self._replaced_frames,
            )

    def _take_latest(self) -> Any | None:
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    def _record_error(self, exc: BaseException) -> None:
        error = RuntimeError(
            f"OpenCV preview failed: {exc}. Use --no-preview on a headless session."
        )
        with self._lock:
            self._error = error

    def _run(self) -> None:
        window_created = False
        frame_was_shown = False
        invisible_polls = 0
        try:
            self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)
            window_created = True
            self._ready.set()

            while not self._stop_requested.is_set():
                frame = self._take_latest()
                if frame is not None:
                    self._cv2.imshow(self.window_name, frame)
                    frame_was_shown = True
                    with self._lock:
                        self._displayed_frames += 1

                key = self._cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self._close_requested.set()
                    break

                try:
                    visible = self._cv2.getWindowProperty(
                        self.window_name,
                        self._cv2.WND_PROP_VISIBLE,
                    )
                except self._cv2.error:
                    # Not every Linux HighGUI backend implements this flag.
                    visible = 1
                if visible < 1 and frame_was_shown:
                    # Native Win32 can report an unpainted new window as not
                    # visible until the first imshow. Treat visibility as a
                    # close signal only after content has actually been shown;
                    # otherwise a slow first capture self-closes the preview.
                    invisible_polls += 1
                    if invisible_polls >= 2:
                        self._close_requested.set()
                        break
                elif visible >= 1:
                    invisible_polls = 0

                self._wake.clear()
                if self._stop_requested.is_set():
                    break
                self._wake.wait(self.idle_poll_s)
        except BaseException as exc:
            # OpenCV exposes backend failures as cv2.error, but containing an
            # unexpected worker exception is safer than silently losing the
            # preview/event loop.
            self._record_error(exc)
            self._close_requested.set()
        finally:
            self._ready.set()
            if window_created:
                try:
                    self._cv2.destroyWindow(self.window_name)
                except BaseException:
                    # Shutdown must remain bounded even on a broken GUI backend.
                    pass
            self._stopped.set()
            self._wake.set()


def highgui_backend(cv2_module: Any) -> str:
    """Return OpenCV's active UI framework without making it a requirement."""

    reporter = getattr(cv2_module, "currentUIFramework", None)
    if reporter is None:
        return ""
    try:
        return str(reporter() or "").strip().upper()
    except Exception:
        return ""


def preview_mode(cv2_module: Any, *, platform: str | None = None) -> str:
    """Select threading only for the native Windows HighGUI backend.

    Qt, GTK, Cocoa, and unknown backends remain inline because their main-thread
    requirements cannot be safely inferred away.  ``platform`` is injectable
    so this safety policy is deterministic in tests.
    """

    active_platform = sys.platform if platform is None else platform
    if active_platform == "win32" and highgui_backend(cv2_module) == "WIN32":
        return "threaded"
    return "inline"


def create_preview_window(
    cv2_module: Any,
    window_name: str,
    *,
    platform: str | None = None,
) -> InlinePreviewWindow | AsyncPreviewWindow:
    """Build the backend-safe preview implementation for this runtime."""

    if preview_mode(cv2_module, platform=platform) == "threaded":
        return AsyncPreviewWindow(cv2_module, window_name)
    return InlinePreviewWindow(cv2_module, window_name)
