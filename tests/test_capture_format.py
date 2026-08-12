from __future__ import annotations

import unittest
from unittest import mock

from capture.opencv_source import (
    KNOWN_PIXEL_FORMATS,
    OpenCVCaptureSource,
    _fourcc_text,
    _normalized_pixel_format,
)


def fourcc_value(code: str) -> int:
    """Pack four characters the way V4L2 and OpenCV do."""

    return sum(ord(char) << (8 * index) for index, char in enumerate(code))


class FakeCapture:
    """Records property writes and answers reads from a fixed state."""

    def __init__(self, state: dict[int, float] | None = None) -> None:
        self.writes: list[tuple[int, float]] = []
        self.state = state or {}

    def set(self, property_id: int, value: float) -> bool:
        self.writes.append((property_id, value))
        return True

    def get(self, property_id: int) -> float:
        return self.state.get(property_id, 0.0)


class PixelFormatValidationTests(unittest.TestCase):
    def test_none_and_blank_mean_no_request(self) -> None:
        self.assertIsNone(_normalized_pixel_format(None))
        self.assertIsNone(_normalized_pixel_format("   "))

    def test_codes_are_upper_cased(self) -> None:
        self.assertEqual(_normalized_pixel_format("nv12"), "NV12")

    def test_wrong_length_is_rejected(self) -> None:
        for value in ("NV1", "NV122", ""):
            if value == "":
                continue
            with self.subTest(value=value), self.assertRaises(ValueError):
                _normalized_pixel_format(value)

    def test_non_alphanumeric_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _normalized_pixel_format("NV-2")

    def test_unknown_but_well_formed_codes_are_allowed(self) -> None:
        # The catalogue is for documentation; drivers expose many more codes and
        # refusing an unlisted one would block valid hardware.
        self.assertEqual(_normalized_pixel_format("ABCD"), "ABCD")
        self.assertIn("NV12", KNOWN_PIXEL_FORMATS)

    def test_constructor_rejects_a_malformed_format(self) -> None:
        with self.assertRaises(ValueError):
            OpenCVCaptureSource(0, pixel_format="BAD")


class FourccDecodingTests(unittest.TestCase):
    def test_packed_value_decodes_to_characters(self) -> None:
        capture = FakeCapture()
        with mock.patch("capture.opencv_source._integer_property") as reader:
            reader.return_value = fourcc_value("NV12")
            self.assertEqual(_fourcc_text(capture), "NV12")

    def test_zero_means_the_driver_reported_nothing(self) -> None:
        capture = FakeCapture()
        with mock.patch("capture.opencv_source._integer_property") as reader:
            reader.return_value = 0
            self.assertIsNone(_fourcc_text(capture))

    def test_mjpg_round_trips(self) -> None:
        capture = FakeCapture()
        with mock.patch("capture.opencv_source._integer_property") as reader:
            reader.return_value = fourcc_value("MJPG")
            self.assertEqual(_fourcc_text(capture), "MJPG")


class FormatNegotiationOrderTests(unittest.TestCase):
    def _apply(self, pixel_format: str | None) -> list[tuple[int, float]]:
        source = OpenCVCaptureSource(
            0, width=1920, height=1080, fps=240, pixel_format=pixel_format
        )
        capture = FakeCapture()

        fake_cv2 = mock.MagicMock()
        fake_cv2.CAP_PROP_FOURCC = 6
        fake_cv2.CAP_PROP_FRAME_WIDTH = 3
        fake_cv2.CAP_PROP_FRAME_HEIGHT = 4
        fake_cv2.CAP_PROP_FPS = 5
        fake_cv2.CAP_PROP_BUFFERSIZE = 38
        fake_cv2.VideoWriter_fourcc = lambda *chars: fourcc_value("".join(chars))

        with mock.patch("capture.opencv_source.cv2", fake_cv2):
            source._apply_live_requests(capture)
        return capture.writes

    def test_format_is_set_before_size_and_rate(self) -> None:
        writes = self._apply("NV12")

        property_order = [property_id for property_id, _ in writes]
        self.assertEqual(property_order[0], 6, "FOURCC must be negotiated first")
        self.assertIn(5, property_order, "frame rate must still be requested")
        self.assertLess(
            property_order.index(6),
            property_order.index(5),
            "a rate requested before the format is clamped to the old mode",
        )

    def test_the_requested_code_is_packed_correctly(self) -> None:
        writes = self._apply("NV12")

        self.assertEqual(writes[0], (6, fourcc_value("NV12")))

    def test_no_format_request_leaves_fourcc_untouched(self) -> None:
        writes = self._apply(None)

        self.assertNotIn(6, [property_id for property_id, _ in writes])


if __name__ == "__main__":
    unittest.main()
