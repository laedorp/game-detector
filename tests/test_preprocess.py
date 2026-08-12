from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
