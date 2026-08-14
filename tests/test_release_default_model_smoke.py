from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.smoke_release_default_model import (
    ReleaseDefaultSmokeError,
    load_release_default,
    smoke_release_default,
    validate_benchmark,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ReleaseDefaultModelSmokeTests(unittest.TestCase):
    def _bundle(self, root: Path) -> tuple[Path, dict]:
        bundle = root / "ProAim"
        model = bundle / "_internal" / "models" / "player" / "default.onnx"
        labels = bundle / "_internal" / "models" / "player.txt"
        model.parent.mkdir(parents=True)
        labels.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"onnx")
        labels.write_bytes(b"player\n")
        (bundle / "ProAimCLI.exe").write_bytes(b"exe")
        record = {
            "detail_crop_size_source_pixels": 768,
            "input_shape_hw": [384, 640],
            "labels_path": "_internal/models/player.txt",
            "labels_sha256": _sha(labels.read_bytes()),
            "model_path": "_internal/models/player/default.onnx",
            "model_sha256": _sha(model.read_bytes()),
            "preset": "future-default",
        }
        (bundle / "BUILD-INFO.json").write_text(
            json.dumps(
                {
                    "application": "ProAim",
                    "release_default_model": record,
                    "schema": 2,
                }
            ),
            encoding="utf-8",
        )
        return bundle, record

    def _benchmark(self, record: dict, bundle: Path) -> dict:
        model_path = bundle / record["model_path"]
        labels_path = bundle / record["labels_path"]
        return {
            "methodology": {
                "backend": "onnxruntime",
                "requested_device": "CPU",
                "require_full_provider": False,
            },
            "models": [
                {
                    "artifact": {
                        "files": [
                            {
                                "resolved_path": str(model_path.resolve()),
                                "sha256": record["model_sha256"],
                            }
                        ]
                    },
                    "input_shape_hw": list(record["input_shape_hw"]),
                    "key": "release-default",
                    "labels_artifact": {
                        "files": [
                            {
                                "resolved_path": str(labels_path.resolve()),
                                "sha256": record["labels_sha256"],
                            }
                        ]
                    },
                    "runtime": {
                        "active_providers": ["CPUExecutionProvider"],
                        "requested_provider": "CPUExecutionProvider",
                    },
                }
            ],
            "schema_version": 1,
        }

    def test_smoke_uses_dynamic_paths_shape_and_validates_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, record = self._bundle(root)
            benchmark = self._benchmark(record, bundle)
            completed = subprocess.CompletedProcess(
                ["ProAimCLI.exe"], 0, stdout=json.dumps(benchmark), stderr=""
            )
            with mock.patch(
                "scripts.smoke_release_default_model.subprocess.run",
                return_value=completed,
            ) as run:
                output = root / "smoke.json"
                smoke_release_default(
                    bundle, bundle / "ProAimCLI.exe", output, device="CPU"
                )
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--name") + 1], "release-default")
            self.assertEqual(command[command.index("--inference-size") + 1], "384x640")
            self.assertEqual(
                Path(command[command.index("--model") + 1]),
                bundle / "_internal" / "models" / "player" / "default.onnx",
            )
            self.assertEqual(
                Path(command[command.index("--labels") + 1]),
                bundle / "_internal" / "models" / "player.txt",
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), benchmark)

    def test_bundle_mutation_fails_before_running_frozen_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, _ = self._bundle(root)
            (bundle / "_internal" / "models" / "player.txt").write_bytes(b"changed\n")
            with mock.patch(
                "scripts.smoke_release_default_model.subprocess.run"
            ) as run, self.assertRaisesRegex(ReleaseDefaultSmokeError, "labels differs"):
                load_release_default(bundle)
            run.assert_not_called()

    def test_build_info_rejects_a_false_like_detail_workload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle, _record = self._bundle(Path(temporary))
            build_info = bundle / "BUILD-INFO.json"
            payload = json.loads(build_info.read_text())
            payload["release_default_model"]["detail_crop_size_source_pixels"] = True
            build_info.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ReleaseDefaultSmokeError, "detail workload"):
                load_release_default(bundle)

    def test_report_model_labels_and_shape_must_match_build_info(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, record = self._bundle(root)
            for key, mutation, message in (
                ("input_shape_hw", [416, 416], "input shape"),
                ("model_sha256", "0" * 64, "different model"),
                ("labels_sha256", "1" * 64, "different labels"),
                ("labels_path", str((root / "wrong-labels.txt").resolve()), "labels path"),
            ):
                benchmark = self._benchmark(record, bundle)
                model = benchmark["models"][0]
                if key == "input_shape_hw":
                    model[key] = mutation
                elif key == "model_sha256":
                    model["artifact"]["files"][0]["sha256"] = mutation
                elif key == "labels_sha256":
                    model["labels_artifact"]["files"][0]["sha256"] = mutation
                else:
                    model["labels_artifact"]["files"][0]["resolved_path"] = mutation
                with self.subTest(key=key), self.assertRaisesRegex(
                    ReleaseDefaultSmokeError, message
                ):
                    validate_benchmark(
                        benchmark,
                        record,
                        bundle / record["model_path"],
                        bundle / record["labels_path"],
                        expected_device="CPU",
                    )

    def test_cpu_smoke_rejects_full_provider_or_wrong_active_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, record = self._bundle(root)
            model_path = bundle / record["model_path"]
            labels_path = bundle / record["labels_path"]
            benchmark = self._benchmark(record, bundle)
            benchmark["methodology"]["require_full_provider"] = True
            with self.assertRaisesRegex(ReleaseDefaultSmokeError, "full-provider"):
                validate_benchmark(
                    benchmark,
                    record,
                    model_path,
                    labels_path,
                    expected_device="CPU",
                )

            benchmark = self._benchmark(record, bundle)
            benchmark["models"][0]["runtime"]["active_providers"] = [
                "CUDAExecutionProvider"
            ]
            with self.assertRaisesRegex(ReleaseDefaultSmokeError, "CPUExecutionProvider"):
                validate_benchmark(
                    benchmark,
                    record,
                    model_path,
                    labels_path,
                    expected_device="CPU",
                )


if __name__ == "__main__":
    unittest.main()
