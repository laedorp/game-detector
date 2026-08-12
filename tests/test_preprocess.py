from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from utils.preprocess import preprocess_frame


try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None


class PreprocessTests(unittest.TestCase):
    @unittest.skipIf(cv2 is None, "OpenCV is not installed")
    def test_preprocess_returns_expected_tensor_shape_and_dtype(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        prepared = preprocess_frame(frame, inference_size=320)

        self.assertEqual(prepared.tensor.shape, (1, 3, 320, 320))
        self.assertEqual(prepared.tensor.dtype, np.float32)

    def test_letterbox_workspace_is_reused_between_calls(self) -> None:
        if cv2 is None:
            self.skipTest("OpenCV is not installed")
        # Use a non-square frame so letterboxing executes.
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)

        first = preprocess_frame(frame, inference_size=320)
        second = preprocess_frame(frame, inference_size=320)

        np.testing.assert_array_equal(first.tensor, second.tensor)

    def test_inference_size_must_be_a_positive_integer(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                preprocess_frame(frame, inference_size=value)
        for value in (True, 320.5):
            with self.subTest(value=value), self.assertRaises(TypeError):
                preprocess_frame(frame, inference_size=value)

    def test_crop_size_must_be_a_positive_integer(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                preprocess_frame(frame, inference_size=320, crop_size=value)
        for value in (True, 4.5):
            with self.subTest(value=value), self.assertRaises(TypeError):
                preprocess_frame(frame, inference_size=320, crop_size=value)

    def test_empty_frames_are_rejected_before_resize(self) -> None:
        for shape in ((0, 10, 3), (10, 0, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                preprocess_frame(np.zeros(shape, dtype=np.uint8), inference_size=320)

    def test_letterbox_workspaces_are_thread_local(self) -> None:
        if cv2 is None:
            self.skipTest("OpenCV is not installed")
        dark = np.zeros((20, 40, 3), dtype=np.uint8)
        bright = np.full((20, 40, 3), 255, dtype=np.uint8)

        def run(frame: np.ndarray) -> np.ndarray:
            expected_sum = preprocess_frame(frame, 64).tensor.sum()
            for _ in range(50):
                prepared = preprocess_frame(frame, 64)
                self.assertEqual(prepared.tensor.sum(), expected_sum)
            return prepared.tensor

        with ThreadPoolExecutor(max_workers=2) as executor:
            dark_result, bright_result = (
                future.result()
                for future in (executor.submit(run, dark), executor.submit(run, bright))
            )

        self.assertLess(float(dark_result.mean()), float(bright_result.mean()))


if __name__ == "__main__":
    unittest.main()
