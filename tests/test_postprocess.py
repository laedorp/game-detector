from __future__ import annotations

from time import perf_counter
import unittest

import numpy as np

from detection.postprocess import (
    MAX_DETECTIONS,
    MAX_NMS_CANDIDATES,
    class_aware_nms,
    decode_yolo_output,
    supported_yolo_output_layout,
)


class NmsTests(unittest.TestCase):
    def test_nms_is_class_aware_and_confidence_ordered(self) -> None:
        boxes = np.asarray(
            [[0, 0, 10, 10], [1, 1, 11, 11], [1, 1, 11, 11]],
            dtype=np.float32,
        )
        scores = np.asarray([0.8, 0.9, 0.7], dtype=np.float32)
        classes = np.asarray([0, 0, 1], dtype=np.int64)

        kept = class_aware_nms(boxes, scores, classes, 0.5)

        self.assertEqual(kept.tolist(), [1, 2])

    def test_equal_scores_use_original_index_as_tie_breaker(self) -> None:
        boxes = np.asarray(
            [[index * 2, 0, index * 2 + 1, 1] for index in range(10)],
            dtype=np.float32,
        )
        kept = class_aware_nms(
            boxes,
            np.ones(10, dtype=np.float32),
            np.zeros(10, dtype=np.int64),
            0.5,
        )

        self.assertEqual(kept.tolist(), list(range(10)))

    def test_large_candidate_set_is_bounded_and_fast(self) -> None:
        count = 8_400
        centers = np.arange(count, dtype=np.float32) * 2.0
        raw = np.zeros((1, count, 5), dtype=np.float32)
        raw[0, :, 0] = centers
        raw[0, :, 1] = 1.0
        raw[0, :, 2:4] = 1.0
        raw[0, :, 4] = 0.9

        started = perf_counter()
        detections = decode_yolo_output(
            raw,
            labels=("person",),
            confidence=0.0,
            output_format="traditional",
        )
        elapsed = perf_counter() - started

        self.assertEqual(len(detections), MAX_DETECTIONS)
        self.assertLessEqual(MAX_DETECTIONS, MAX_NMS_CANDIDATES)
        # This is deliberately generous for shared CI while still catching the
        # previous 1.4-second quadratic path on the development host.
        self.assertLess(elapsed, 0.75)

    def test_nonfinite_nms_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            class_aware_nms(
                np.asarray([[0, 0, np.nan, 1]], dtype=np.float32),
                np.asarray([0.9], dtype=np.float32),
                np.asarray([0], dtype=np.int64),
                0.5,
            )


class OutputLayoutTests(unittest.TestCase):
    def test_supported_layouts_are_identified(self) -> None:
        self.assertEqual(
            supported_yolo_output_layout([1, "detections", 6], 1),
            "end2end",
        )
        self.assertEqual(
            supported_yolo_output_layout([1, 84, 8400], 80),
            "traditional_columns",
        )
        self.assertEqual(
            supported_yolo_output_layout([1, 8400, 84], 80),
            "traditional_rows",
        )

    def test_wrong_batch_rank_and_attribute_count_are_rejected(self) -> None:
        for shape in ([2, 300, 6], [1, 300], [1, 10, 10]):
            with self.subTest(shape=shape):
                self.assertIsNone(supported_yolo_output_layout(shape, 80))


if __name__ == "__main__":
    unittest.main()
