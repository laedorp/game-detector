from __future__ import annotations

import ctypes
from ctypes.util import find_library
import os
from pathlib import Path
from queue import SimpleQueue
import subprocess
import sys
from threading import Event, get_ident
import time
import unittest

import numpy as np

from utils.preview import (
    AsyncPreviewWindow,
    InlinePreviewWindow,
    PreviewPacer,
    create_preview_window,
    highgui_backend,
    preview_mode,
)


class FakeCv2:
    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 1

    class error(Exception):
        pass

    def __init__(self, ui_framework: str = "") -> None:
        self.calls: list[tuple[str, int]] = []
        self.frames: list[np.ndarray] = []
        self.shown = Event()
        self.keys: SimpleQueue[int] = SimpleQueue()
        self.visible = 1.0
        self.ui_framework = ui_framework

    def currentUIFramework(self) -> str:
        return self.ui_framework

    def _record(self, name: str) -> None:
        self.calls.append((name, get_ident()))

    def namedWindow(self, _name: str, _flags: int) -> None:
        self._record("namedWindow")

    def imshow(self, _name: str, frame: np.ndarray) -> None:
        self._record("imshow")
        self.frames.append(frame.copy())
        self.shown.set()

    def waitKey(self, _delay: int) -> int:
        self._record("waitKey")
        return self.keys.get() if not self.keys.empty() else -1

    def getWindowProperty(self, _name: str, _property: int) -> float:
        self._record("getWindowProperty")
        return self.visible

    def destroyWindow(self, _name: str) -> None:
        self._record("destroyWindow")


class BlockingWaitCv2(FakeCv2):
    def __init__(self) -> None:
        super().__init__()
        self.wait_entered = Event()
        self.release_wait = Event()
        self._first_wait = True

    def waitKey(self, delay: int) -> int:
        self._record("waitKey")
        if self._first_wait:
            self._first_wait = False
            self.wait_entered.set()
            self.release_wait.wait(2.0)
        return -1


class BlockingCreateCv2(FakeCv2):
    def __init__(self) -> None:
        super().__init__()
        self.create_entered = Event()
        self.release_create = Event()

    def namedWindow(self, name: str, flags: int) -> None:
        self._record("namedWindow")
        self.create_entered.set()
        self.release_create.wait(2.0)


class FailingShowCv2(FakeCv2):
    def imshow(self, _name: str, _frame: np.ndarray) -> None:
        self._record("imshow")
        self.shown.set()
        raise self.error("simulated GUI failure")


def wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class AsyncPreviewWindowTests(unittest.TestCase):
    def test_worker_owns_highgui_and_submission_uses_an_owned_copy(self) -> None:
        cv2 = FakeCv2()
        preview = AsyncPreviewWindow(cv2, "test", idle_poll_s=0.001)
        producer_thread = get_ident()
        preview.start()

        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        self.assertTrue(preview.submit(frame))
        frame.fill(255)

        self.assertTrue(cv2.shown.wait(1.0))
        self.assertTrue(preview.stop())
        self.assertTrue(np.all(cv2.frames[0] == 0))
        highgui_threads = {thread_id for _, thread_id in cv2.calls}
        self.assertEqual(len(highgui_threads), 1)
        self.assertNotIn(producer_thread, highgui_threads)

    def test_latest_only_mailbox_replaces_undisplayed_frames(self) -> None:
        cv2 = BlockingWaitCv2()
        preview = AsyncPreviewWindow(cv2, "test", idle_poll_s=0.001)
        preview.start()
        self.assertTrue(cv2.wait_entered.wait(1.0))

        for value in (1, 2, 3):
            self.assertTrue(
                preview.submit(np.full((2, 2, 3), value, dtype=np.uint8))
            )
        cv2.release_wait.set()

        self.assertTrue(cv2.shown.wait(1.0))
        self.assertEqual(int(cv2.frames[0][0, 0, 0]), 3)
        stats = preview.stats
        self.assertEqual(stats.submitted_frames, 3)
        self.assertEqual(stats.displayed_frames, 1)
        self.assertEqual(stats.replaced_frames, 2)
        self.assertTrue(preview.stop())

    def test_escape_is_signaled_without_a_new_submission(self) -> None:
        cv2 = FakeCv2()
        preview = AsyncPreviewWindow(cv2, "test", idle_poll_s=0.001)
        preview.start()

        cv2.keys.put(27)
        self.assertTrue(wait_until(lambda: not preview.should_continue()))
        self.assertTrue(preview.stop())

    def test_window_close_is_signaled_without_a_new_submission(self) -> None:
        cv2 = FakeCv2()
        preview = AsyncPreviewWindow(cv2, "test", idle_poll_s=0.001)
        preview.start()

        self.assertTrue(preview.submit(np.zeros((2, 2, 3), dtype=np.uint8)))
        self.assertTrue(cv2.shown.wait(1.0))

        cv2.visible = 0.0
        self.assertTrue(wait_until(lambda: not preview.should_continue()))
        self.assertTrue(preview.stop())

    def test_initially_invisible_win32_window_waits_for_the_first_frame(self) -> None:
        cv2 = FakeCv2("WIN32")
        cv2.visible = 0.0
        preview = AsyncPreviewWindow(cv2, "test", idle_poll_s=0.001)
        preview.start()

        time.sleep(0.02)
        self.assertTrue(preview.should_continue())
        cv2.visible = 1.0
        self.assertTrue(preview.submit(np.zeros((2, 2, 3), dtype=np.uint8)))
        self.assertTrue(cv2.shown.wait(1.0))
        self.assertTrue(preview.stop())

    def test_worker_error_is_propagated_to_the_producer(self) -> None:
        cv2 = FailingShowCv2()
        preview = AsyncPreviewWindow(cv2, "test", idle_poll_s=0.001)
        preview.start()
        preview.submit(np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertTrue(cv2.shown.wait(1.0))
        self.assertTrue(preview.stop())
        with self.assertRaisesRegex(RuntimeError, "simulated GUI failure"):
            preview.raise_if_failed()

    def test_startup_and_stop_waits_are_bounded(self) -> None:
        create_cv2 = BlockingCreateCv2()
        preview = AsyncPreviewWindow(
            create_cv2,
            "test",
            startup_timeout_s=0.02,
            stop_timeout_s=0.02,
        )
        with self.assertRaisesRegex(RuntimeError, "Timed out"):
            preview.start()
        self.assertTrue(create_cv2.create_entered.is_set())
        create_cv2.release_create.set()
        self.assertTrue(preview.stop(timeout_s=1.0))

        wait_cv2 = BlockingWaitCv2()
        preview = AsyncPreviewWindow(wait_cv2, "test", stop_timeout_s=0.01)
        preview.start()
        self.assertTrue(wait_cv2.wait_entered.wait(1.0))
        self.assertFalse(preview.stop())
        wait_cv2.release_wait.set()
        self.assertTrue(preview.stop(timeout_s=1.0))


class PreviewPolicyTests(unittest.TestCase):
    def test_linux_qt_and_unknown_backends_remain_inline(self) -> None:
        for platform, framework in (
            ("linux", "QT"),
            ("linux", "GTK3"),
            ("linux", ""),
            ("darwin", "COCOA"),
            ("win32", "QT"),
            ("win32", ""),
        ):
            with self.subTest(platform=platform, framework=framework):
                cv2 = FakeCv2(framework)
                self.assertEqual(
                    preview_mode(cv2, platform=platform),
                    "inline",
                )
                self.assertIsInstance(
                    create_preview_window(cv2, "test", platform=platform),
                    InlinePreviewWindow,
                )

    def test_native_windows_highgui_selects_threaded_worker(self) -> None:
        cv2 = FakeCv2("WIN32")

        self.assertEqual(highgui_backend(cv2), "WIN32")
        self.assertEqual(preview_mode(cv2, platform="win32"), "threaded")
        self.assertIsInstance(
            create_preview_window(cv2, "test", platform="win32"),
            AsyncPreviewWindow,
        )

    def test_inline_lifecycle_and_static_poll_run_on_caller_thread(self) -> None:
        cv2 = FakeCv2("QT")
        preview = create_preview_window(cv2, "test", platform="linux")
        caller_thread = get_ident()

        preview.start()
        self.assertTrue(preview.poll())
        self.assertTrue(preview.submit(np.zeros((2, 3, 3), dtype=np.uint8)))
        cv2.keys.put(ord("q"))
        self.assertFalse(preview.poll())
        self.assertTrue(preview.stop())

        self.assertEqual({thread for _, thread in cv2.calls}, {caller_thread})

    def test_inline_static_window_close_needs_one_timeout_poll(self) -> None:
        cv2 = FakeCv2("QT")
        preview = InlinePreviewWindow(cv2, "test")
        preview.start()

        cv2.visible = 0.0
        self.assertFalse(preview.poll())
        self.assertTrue(preview.stop())

    def test_inline_highgui_calls_follow_preview_pacing_not_inference_rate(self) -> None:
        cv2 = FakeCv2("QT")
        preview = InlinePreviewWindow(cv2, "test")
        pacer = PreviewPacer(15)
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        preview.start()

        # Model a 100 Hz inference loop for 200 ms. Only the frames selected by
        # the 15 FPS pacer service HighGUI; intervening frames perform a cheap
        # signal check. Startup and a static-capture timeout each add one poll.
        for now_ns in range(0, 200_000_000, 10_000_000):
            if pacer.should_render(now_ns):
                self.assertTrue(preview.submit(frame))
            else:
                self.assertTrue(preview.should_continue())
        self.assertTrue(preview.poll())
        preview.stop()

        names = [name for name, _ in cv2.calls]
        self.assertEqual(names.count("imshow"), 3)
        self.assertEqual(names.count("waitKey"), 5)
        self.assertEqual(names.count("getWindowProperty"), 5)


def _live_x11_display_available() -> bool:
    """Probe X11 without letting Qt abort on a stale DISPLAY variable."""

    display = os.environ.get("DISPLAY")
    if os.name != "posix" or not display:
        return False
    library = find_library("X11")
    if not library:
        return False
    try:
        x11 = ctypes.CDLL(library)
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.restype = ctypes.c_int
        handle = x11.XOpenDisplay(os.fsencode(display))
        if not handle:
            return False
        x11.XCloseDisplay(handle)
        return True
    except (AttributeError, OSError):
        return False


@unittest.skipUnless(_live_x11_display_available(), "requires a reachable X11 display")
class RealHighGuiIntegrationTests(unittest.TestCase):
    def test_real_linux_backend_uses_inline_main_thread_lifecycle(self) -> None:
        # Isolate OpenCV's bundled Qt from PySide's QApplication. The launcher
        # tests intentionally select an offscreen PySide plugin during unittest
        # discovery, while this real backend check must exercise OpenCV xcb.
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "xcb"
        code = """
import cv2
import numpy as np
from utils.preview import InlinePreviewWindow, create_preview_window, highgui_backend

preview = create_preview_window(cv2, "ProAim preview integration test", platform="linux")
assert isinstance(preview, InlinePreviewWindow)
assert highgui_backend(cv2) != "WIN32"
try:
    preview.start()
    assert preview.poll()
    assert preview.submit(np.zeros((24, 32, 3), dtype=np.uint8))
    assert preview.poll()
finally:
    assert preview.stop()
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
