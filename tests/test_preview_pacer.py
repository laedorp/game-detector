from __future__ import annotations

import unittest

from utils.preview import PreviewPacer


class PreviewPacerTests(unittest.TestCase):
    def test_first_frame_renders_and_intermediate_frames_are_skipped(self) -> None:
        pacer = PreviewPacer(20)

        self.assertTrue(pacer.should_render(1_000_000_000))
        self.assertFalse(pacer.should_render(1_049_999_999))
        self.assertTrue(pacer.should_render(1_050_000_000))

    def test_slow_frame_does_not_trigger_catch_up_burst(self) -> None:
        pacer = PreviewPacer(30)

        self.assertTrue(pacer.should_render(0))
        self.assertTrue(pacer.should_render(1_000_000_000))
        self.assertFalse(pacer.should_render(1_000_000_001))

    def test_invalid_rates_are_rejected(self) -> None:
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PreviewPacer(value)


if __name__ == "__main__":
    unittest.main()
