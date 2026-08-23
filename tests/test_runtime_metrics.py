from __future__ import annotations

import unittest

from utils.metrics import FrameTimings, RollingMetrics
from utils.render import console_summary


def sample(value: float) -> FrameTimings:
    return FrameTimings(*(value for _ in FrameTimings.__dataclass_fields__))


class RuntimeMetricsTests(unittest.TestCase):
    def test_snapshot_reports_latency_percentiles(self) -> None:
        metrics = RollingMetrics(10)
        for index in range(1, 11):
            metrics.record(sample(float(index)), index * 10_000_000)

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot.p50.inference_ms, 5.5)
        self.assertAlmostEqual(snapshot.p95.inference_ms, 9.55)
        self.assertAlmostEqual(snapshot.p99.inference_ms, 9.91)

    def test_console_distinguishes_queue_processing_and_freshness(self) -> None:
        metrics = RollingMetrics(2)
        metrics.record(sample(3.0), 1)
        metrics.record(sample(5.0), 2)

        summary = console_summary(metrics.snapshot(), skipped_frames=4)

        self.assertIn("capture 4.0 ms", summary)
        self.assertIn("queue 4.0 ms", summary)
        self.assertIn("processing 4.0 ms", summary)
        self.assertIn("fresh p50 4.0", summary)
        self.assertIn("p95 4.9 ms", summary)
        self.assertIn("capture-to-result 4.0 ms", summary)
        self.assertIn("preview service avg/frame 4.0 ms", summary)
        self.assertNotIn("display 4.0 ms", summary)


if __name__ == "__main__":
    unittest.main()
