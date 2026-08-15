from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from capture.opencv_source import OpenCVCaptureSource, _rotate_frame_180
from config import parse_args
from main import _build_capture


class _FrameCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._delivered = False
        self.released = False

    def read(self):
        if self._delivered:
            return False, None
        self._delivered = True
        return True, self._frame.copy()

    def release(self) -> None:
        self.released = True


class CaptureRotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        self.expected = self.frame[::-1, ::-1]

    def test_rotation_is_exact_contiguous_and_independent(self) -> None:
        rotated = _rotate_frame_180(self.frame)

        np.testing.assert_array_equal(rotated, self.expected)
        self.assertTrue(rotated.flags.c_contiguous)
        self.assertFalse(np.shares_memory(rotated, self.frame))

    def test_constructor_requires_a_boolean_rotation_flag(self) -> None:
        for value in (1, "yes", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                OpenCVCaptureSource(0, rotate_180=value)  # type: ignore[arg-type]

    def test_synchronous_frames_are_rotated_before_delivery(self) -> None:
        capture = _FrameCapture(self.frame)
        source = OpenCVCaptureSource("clip", live=False, rotate_180=True)
        source._start_once()
        source._capture = capture

        packet = source.read()

        self.assertIsNotNone(packet)
        assert packet is not None
        np.testing.assert_array_equal(packet.image, self.expected)

    def test_live_frames_are_rotated_before_publication(self) -> None:
        capture = _FrameCapture(self.frame)
        source = OpenCVCaptureSource(0, rotate_180=True)
        source._capture = capture

        with mock.patch.object(source, "_publish_latest") as publish:
            source._device_loop()

        publish.assert_called_once()
        packet = publish.call_args.args[0]
        np.testing.assert_array_equal(packet.image, self.expected)
        self.assertTrue(capture.released)

    def test_cli_accepts_device_rotation_and_rejects_screen_rotation(self) -> None:
        config = parse_args(["--source", "0", "--capture-rotate-180"])
        self.assertTrue(config.capture_rotate_180)

        with self.assertRaises(SystemExit):
            parse_args(["--source", "screen", "--capture-rotate-180"])

    def test_capture_builder_passes_rotation_and_pixel_format(self) -> None:
        config = parse_args(
            [
                "--source",
                "0",
                "--capture-format",
                "NV12",
                "--capture-rotate-180",
            ]
        )
        with mock.patch("capture.OpenCVCaptureSource") as constructor:
            _build_capture(config)

        self.assertEqual(constructor.call_args.kwargs["buffer_size"], 2)
        self.assertEqual(constructor.call_args.kwargs["pixel_format"], "NV12")
        self.assertTrue(constructor.call_args.kwargs["rotate_180"])


if __name__ == "__main__":
    unittest.main()
