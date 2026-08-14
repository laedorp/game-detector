from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest import mock

import app


class AppEntrypointTests(unittest.TestCase):
    def test_benchmark_dispatches_without_rewriting_process_argv(self) -> None:
        benchmark = ModuleType("scripts.benchmark_models")
        benchmark_main = mock.Mock(return_value=17)
        benchmark.main = benchmark_main  # type: ignore[attr-defined]
        original_argv = list(sys.argv)

        with (
            mock.patch(
                "detection.runtime_setup.activate_configured_runtime"
            ) as activate,
            mock.patch.dict(
                sys.modules,
                {"scripts.benchmark_models": benchmark},
            ),
        ):
            result = app.main(
                [
                    "--benchmark-models",
                    "--backend",
                    "onnxruntime",
                    "--device",
                    "CPU",
                ]
            )

        self.assertEqual(result, 17)
        benchmark_main.assert_called_once_with(
            ["--backend", "onnxruntime", "--device", "CPU"]
        )
        activate.assert_called_once_with()
        self.assertEqual(sys.argv, original_argv)


if __name__ == "__main__":
    unittest.main()
