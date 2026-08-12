from __future__ import annotations

import math
from threading import Event
import unittest
from unittest import mock

import numpy as np

from capture.base import _validate_timeout
from capture.opencv_source import OpenCVCaptureSource
from capture.screen_source import ScreenCaptureSource, _normalise_region


class _BlockingCapture:
    def __init__(self, entered: Event, unblock: Event) -> None:
        self.entered = entered
        self.unblock = unblock

    def isOpened(self) -> bool:
        return True

    def read(self):
        self.entered.set()
        self.unblock.wait()
        return False, None

    def release(self) -> None:
        # Some real backends do not unblock a read when release is called from a
        # different thread; model that failure mode explicitly.
        return None

    def set(self, *_args) -> bool:
        return True

    def get(self, *_args) -> float:
        return 0.0

    def getBackendName(self) -> str:
        return "blocking-fake"


class _FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_FOURCC = 6
    CAP_PROP_CONVERT_RGB = 16

    def __init__(self, capture: _BlockingCapture) -> None:
        self.capture = capture

    def VideoCapture(self, *_args):
        return self.capture


class _BlockingScreenshotter:
    monitors = [
        {"left": 0, "top": 0, "width": 8, "height": 8},
        {"left": 0, "top": 0, "width": 8, "height": 8},
    ]

    def __init__(self, entered: Event, unblock: Event) -> None:
        self.entered = entered
        self.unblock = unblock
        self.grabs = 0

    def grab(self, _target):
        self.grabs += 1
        if self.grabs > 1:
            self.entered.set()
            self.unblock.wait()
        return np.zeros((8, 8, 4), dtype=np.uint8)

    def close(self) -> None:
        return None


class _FakeMssModule:
    def __init__(self, screenshotter: _BlockingScreenshotter) -> None:
        self.screenshotter = screenshotter

    def mss(self):
        return self.screenshotter


class CaptureLifecycleTests(unittest.TestCase):
    def test_close_timeout_does_not_claim_blocked_worker_has_ended(self) -> None:
        entered = Event()
        unblock = Event()
        backend = _BlockingCapture(entered, unblock)
        with mock.patch("capture.opencv_source.cv2", _FakeCv2(backend)):
            source = OpenCVCaptureSource(0, close_timeout=0.01)
            source.start()
            self.assertTrue(entered.wait(1.0))

            source.close()

            self.assertFalse(source.ended)
            self.assertFalse(source._closed)
            self.assertTrue(source._closing)
            self.assertIn("still closing", source.error or "")
            with self.assertRaises(RuntimeError):
                source.start()

            unblock.set()
            assert source._thread is not None
            source._thread.join(1.0)
            self.assertFalse(source._thread.is_alive())
            self.assertTrue(source.ended)
            self.assertTrue(source._closed)

    def test_screen_close_timeout_also_waits_for_worker_truthfully(self) -> None:
        entered = Event()
        unblock = Event()
        screenshotter = _BlockingScreenshotter(entered, unblock)
        with mock.patch("capture.screen_source.mss", _FakeMssModule(screenshotter)):
            source = ScreenCaptureSource(fps=10_000, close_timeout=0.01)
            source.start()
            self.assertTrue(entered.wait(1.0))

            source.close()

            self.assertFalse(source.ended)
            self.assertFalse(source._closed)
            self.assertTrue(source._closing)
            with self.assertRaises(RuntimeError):
                source.start()

            unblock.set()
            assert source._thread is not None
            source._thread.join(1.0)
            self.assertFalse(source._thread.is_alive())
            self.assertTrue(source.ended)
            self.assertTrue(source._closed)

    def test_nonfinite_timeout_is_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validate_timeout(value)
        with self.assertRaises(TypeError):
            _validate_timeout(True)

    def test_region_requires_actual_integers(self) -> None:
        self.assertEqual(_normalise_region((1, 2, 3, 4)), (1, 2, 3, 4))
        self.assertEqual(
            _normalise_region(tuple(np.int64(value) for value in (1, 2, 3, 4))),
            (1, 2, 3, 4),
        )
        for region in ((1.5, 2, 3, 4), ("1", 2, 3, 4), (True, 2, 3, 4)):
            with self.subTest(region=region), self.assertRaises(ValueError):
                _normalise_region(region)

    def test_capture_timeouts_must_be_finite(self) -> None:
        for constructor in (OpenCVCaptureSource, ScreenCaptureSource):
            with self.subTest(constructor=constructor), self.assertRaises(ValueError):
                constructor(0, close_timeout=math.nan) if constructor is OpenCVCaptureSource else constructor(close_timeout=math.nan)


if __name__ == "__main__":
    unittest.main()
