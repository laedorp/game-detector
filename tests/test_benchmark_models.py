from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from scripts.benchmark_models import (
    PROJECT_ROOT,
    _timed_pipeline,
    deterministic_sample,
    main,
    selection_sha256,
    summarize_ms,
)


class BenchmarkUtilityTests(unittest.TestCase):
    def test_int8_release_record_identifies_quantization_and_exact_artifacts(self) -> None:
        model_dir = PROJECT_ROOT / "models" / "fort_player_416_int8_openvino_model"
        metadata = model_dir.joinpath("metadata.yaml").read_text(encoding="utf-8")
        attribution = model_dir.joinpath("ATTRIBUTION.md").read_text(encoding="utf-8")

        self.assertIn("precision: INT8", metadata)
        self.assertIn("method: NNCF post-training quantization", metadata)
        self.assertIn("calibration_samples: 300", metadata)
        self.assertIn(
            "e29876a30238511dae38382d358cf36592023f9183c2d35bd2c5a1714b71ee84",
            metadata,
        )
        self.assertIn("fort_player_416_int8.xml", attribution)
        self.assertIn("byte-identical", attribution)
        self.assertNotIn(
            "This OpenVINO model (`fort_player_416.xml` / `fort_player_416.bin`)",
            attribution,
        )

    def test_deterministic_sample_spans_sorted_input(self) -> None:
        paths = [Path(name) for name in ("9.jpg", "1.jpg", "5.jpg", "3.jpg", "7.jpg")]

        selected = deterministic_sample(paths, 3)

        self.assertEqual(selected, [Path("1.jpg"), Path("5.jpg"), Path("9.jpg")])
        self.assertEqual(deterministic_sample(paths, 9), sorted(paths))
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            deterministic_sample(paths, 0)

    def test_selection_fingerprint_changes_with_name_or_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a.jpg"
            second = root / "b.jpg"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            baseline = selection_sha256([first, second])

            second.write_bytes(b"changed")

            self.assertNotEqual(selection_sha256([first, second]), baseline)
            self.assertNotEqual(selection_sha256([second, first]), baseline)

    def test_timing_summary_uses_population_statistics_and_percentiles(self) -> None:
        summary = summarize_ms([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["median"], 2.5)
        self.assertAlmostEqual(float(summary["p95"]), 3.85)
        self.assertAlmostEqual(float(summary["stdev"]), np.sqrt(1.25))
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize_ms([])

    def test_pipeline_times_each_stage_and_reports_repeat_distribution(self) -> None:
        class Detector:
            runtime_summary = {}

            def infer(self, tensor: np.ndarray) -> np.ndarray:
                self.tensor = tensor
                return np.zeros((1, 1, 6), dtype=np.float32)

            def postprocess(self, raw: np.ndarray, **kwargs: object) -> list[object]:
                return [object(), object()]

        ticks = iter(range(0, 10_000_000, 1_000_000))

        def preprocess(frame: np.ndarray, size: int) -> SimpleNamespace:
            self.assertEqual(size, 32)
            return SimpleNamespace(
                tensor=np.zeros((1, 3, size, size), dtype=np.float32),
                transform=object(),
            )

        result = _timed_pipeline(
            Detector(),
            [np.zeros((10, 20, 3), dtype=np.uint8)],
            32,
            warmup=0,
            iterations=2,
            repeats=1,
            preprocess=preprocess,
            clock=lambda: next(ticks),
        )

        self.assertEqual(result["timing_ms"]["preprocess"]["mean"], 1.0)
        self.assertEqual(result["timing_ms"]["inference"]["mean"], 1.0)
        self.assertEqual(result["timing_ms"]["postprocess"]["mean"], 1.0)
        self.assertEqual(result["timing_ms"]["pipeline"]["mean"], 3.0)
        self.assertAlmostEqual(result["pipeline_fps_from_mean"], 1000.0 / 3.0)
        self.assertEqual(result["detections_mean"], 2.0)
        self.assertEqual(len(result["repeats"]), 1)

    def test_cli_errors_are_machine_readable_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.xml"
            from contextlib import redirect_stderr, redirect_stdout
            from io import StringIO

            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                return_code = main(
                    [
                        "--model",
                        str(missing),
                        "--inference-size",
                        "320",
                        "--synthetic",
                        "--samples",
                        "1",
                    ]
                )

        self.assertEqual(return_code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("not found", payload["error"])
        self.assertIn("Benchmark failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
