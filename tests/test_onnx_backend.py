from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from detection.base import DeviceNotAvailableError, DetectorError, ModelError
from detection.onnx_yolo import (
    PROVIDER_PREFERENCE,
    OnnxRuntimeYoloDetector,
    resolve_providers,
)


ALL_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)
CPU_ONLY = ("CPUExecutionProvider",)


class ProviderResolutionTests(unittest.TestCase):
    def test_auto_picks_the_fastest_installed_provider(self) -> None:
        chain, target = resolve_providers("AUTO", ALL_PROVIDERS)

        self.assertEqual(target, "TensorrtExecutionProvider")
        self.assertEqual(chain[0], "TensorrtExecutionProvider")

    def test_tensorrt_chain_keeps_cuda_before_cpu(self) -> None:
        chain, target = resolve_providers("AUTO", ALL_PROVIDERS)

        self.assertEqual(target, "TensorrtExecutionProvider")
        self.assertEqual(
            chain,
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def test_auto_falls_back_to_cpu_when_nothing_else_exists(self) -> None:
        chain, target = resolve_providers("AUTO", CPU_ONLY)

        self.assertEqual(target, "CPUExecutionProvider")
        self.assertEqual(chain, ["CPUExecutionProvider"])

    def test_generic_gpu_alias_uses_auto_selection(self) -> None:
        chain, target = resolve_providers("GPU", ALL_PROVIDERS)

        self.assertEqual(target, "TensorrtExecutionProvider")
        self.assertEqual(chain[0], "TensorrtExecutionProvider")

    def test_every_chain_ends_with_cpu_so_partial_graphs_still_run(self) -> None:
        for requested in ("ROCM", "CUDA", "DIRECTML"):
            chain, _ = resolve_providers(requested, ALL_PROVIDERS)
            self.assertEqual(chain[-1], "CPUExecutionProvider", requested)
            self.assertEqual(len(chain), 2, requested)

    def test_amd_aliases_resolve_to_rocm(self) -> None:
        for alias in ("AMD", "ROCM", "rocm"):
            _, target = resolve_providers(alias, ALL_PROVIDERS)
            self.assertEqual(target, "ROCMExecutionProvider", alias)

    def test_windows_amd_alias_resolves_to_directml(self) -> None:
        for alias in ("DIRECTML", "DML"):
            _, target = resolve_providers(alias, ALL_PROVIDERS)
            self.assertEqual(target, "DmlExecutionProvider", alias)

    def test_nvidia_alias_resolves_to_cuda(self) -> None:
        _, target = resolve_providers("NVIDIA", ALL_PROVIDERS)

        self.assertEqual(target, "CUDAExecutionProvider")

    def test_requesting_an_uninstalled_provider_is_actionable(self) -> None:
        with self.assertRaises(DeviceNotAvailableError) as raised:
            resolve_providers("ROCM", CPU_ONLY)

        message = str(raised.exception)
        self.assertIn("ROCMExecutionProvider", message)
        self.assertIn("not installed", message)

    def test_unknown_device_names_are_rejected_with_the_valid_set(self) -> None:
        with self.assertRaises(DeviceNotAvailableError) as raised:
            resolve_providers("QUANTUM", ALL_PROVIDERS)

        self.assertIn("AUTO", str(raised.exception))

    def test_empty_device_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_providers("   ", ALL_PROVIDERS)

    def test_cpu_provider_is_ranked_last_in_the_preference_order(self) -> None:
        self.assertEqual(PROVIDER_PREFERENCE[-1], "CPUExecutionProvider")


class FakeTensorSpec:
    def __init__(self, name: str, shape: list) -> None:
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(
        self,
        path: str,
        sess_options=None,
        providers=None,
        *,
        input_shape: list | None = None,
        output: np.ndarray | None = None,
        fail: Exception | None = None,
    ) -> None:
        self.path = path
        self.providers = list(providers or [])
        self._inputs = [FakeTensorSpec("images", input_shape or [1, 3, 416, 416])]
        self._outputs = [FakeTensorSpec("output0", [1, 300, 6])]
        self._output = output if output is not None else np.zeros((1, 300, 6), np.float32)
        self._fail = fail
        self.runs = 0

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return self.providers

    def run(self, names, feed):
        if self._fail is not None:
            raise self._fail
        self.runs += 1
        return [self._output]


def make_detector(**kwargs) -> OnnxRuntimeYoloDetector:
    session_kwargs = kwargs.pop("session_kwargs", {})

    def factory(path, sess_options=None, providers=None):
        return FakeSession(path, sess_options, providers, **session_kwargs)

    defaults = dict(
        model_path="unused.onnx",
        labels_path=None,
        device="CPU",
        inference_size=416,
        session_factory=factory,
    )
    defaults.update(kwargs)
    return OnnxRuntimeYoloDetector(**defaults)


class DetectorContractTests(unittest.TestCase):
    def test_summary_reports_the_backend_and_active_providers(self) -> None:
        detector = make_detector()

        summary = detector.runtime_summary

        self.assertEqual(summary["runtime"], "ONNX Runtime")
        self.assertEqual(summary["device"], "CPUExecutionProvider")
        self.assertIn("CPUExecutionProvider", summary["active_providers"])

    def test_wrong_tensor_shape_is_rejected_before_the_session_runs(self) -> None:
        detector = make_detector()

        with self.assertRaises(ValueError):
            detector.infer(np.zeros((1, 3, 320, 320), np.float32))

    def test_non_float32_tensor_is_rejected(self) -> None:
        detector = make_detector()

        with self.assertRaises(TypeError):
            detector.infer(np.zeros((1, 3, 416, 416), np.float64))

    def test_a_static_model_that_disagrees_with_the_size_is_refused(self) -> None:
        with self.assertRaises(ModelError) as raised:
            make_detector(session_kwargs={"input_shape": [1, 3, 320, 320]})

        self.assertIn("does not match", str(raised.exception))

    def test_a_dynamic_axis_is_accepted_because_any_size_is_valid(self) -> None:
        detector = make_detector(
            session_kwargs={"input_shape": [1, 3, "height", "width"]}
        )

        self.assertEqual(detector.inference_size, 416)

    def test_session_failure_is_reported_as_a_detector_error(self) -> None:
        detector = make_detector(session_kwargs={"fail": RuntimeError("gpu fell over")})

        with self.assertRaises(DetectorError) as raised:
            detector.infer(np.zeros((1, 3, 416, 416), np.float32))

        self.assertIn("gpu fell over", str(raised.exception))

    def test_warmup_runs_the_requested_iterations(self) -> None:
        sessions: list[FakeSession] = []

        def factory(path, sess_options=None, providers=None):
            session = FakeSession(path, sess_options, providers)
            sessions.append(session)
            return session

        detector = OnnxRuntimeYoloDetector(
            model_path="unused.onnx", device="CPU", inference_size=416,
            session_factory=factory,
        )
        detector.warmup(3)

        self.assertEqual(sessions[0].runs, 3)

    def test_warmup_with_no_iterations_does_nothing(self) -> None:
        sessions: list[FakeSession] = []

        def factory(path, sess_options=None, providers=None):
            session = FakeSession(path, sess_options, providers)
            sessions.append(session)
            return session

        detector = OnnxRuntimeYoloDetector(
            model_path="unused.onnx", device="CPU", inference_size=416,
            session_factory=factory,
        )
        detector.warmup(0)

        self.assertEqual(sessions[0].runs, 0)

    def test_invalid_thresholds_are_rejected(self) -> None:
        for kwargs in ({"confidence": 1.5}, {"iou": -0.1}, {"confidence": float("nan")}):
            with self.assertRaises(ValueError):
                make_detector(**kwargs)

    def test_missing_model_file_is_reported_without_a_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "absent.onnx"
            with self.assertRaises(FileNotFoundError):
                OnnxRuntimeYoloDetector(model_path=missing, inference_size=416)

    def test_end_to_end_output_decodes_into_detections(self) -> None:
        raw = np.zeros((1, 300, 6), np.float32)
        # One confident box covering the middle of a 416x416 frame.
        raw[0, 0] = (100.0, 120.0, 200.0, 300.0, 0.9, 0.0)
        detector = make_detector(
            confidence=0.35, session_kwargs={"output": raw}
        )

        detections = detector.postprocess(
            detector.infer(np.zeros((1, 3, 416, 416), np.float32)),
            frame_shape=(416, 416, 3),
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
