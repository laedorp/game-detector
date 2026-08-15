from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from config import parse_args
from detection.base import DeviceNotAvailableError, DetectorError, ModelError
from detection.onnx_yolo import (
    PROVIDER_PREFERENCE,
    OnnxRuntimeYoloDetector,
    _configure_session_options,
    _preload_nvidia_libraries,
    _provider_options_record,
    _require_active_provider,
    resolve_providers,
)


ALL_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)
CPU_ONLY = ("CPUExecutionProvider",)


class ProviderResolutionTests(unittest.TestCase):
    def test_auto_picks_the_fastest_installed_provider(self) -> None:
        chain, target = resolve_providers("AUTO", ALL_PROVIDERS)

        self.assertEqual(target, "CUDAExecutionProvider")
        self.assertEqual(chain[0], "CUDAExecutionProvider")

    def test_tensorrt_chain_keeps_cuda_before_cpu(self) -> None:
        chain, target = resolve_providers("TENSORRT", ALL_PROVIDERS)

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

        self.assertEqual(target, "CUDAExecutionProvider")
        self.assertEqual(chain[0], "CUDAExecutionProvider")

    def test_generic_gpu_alias_never_silently_falls_back_to_cpu(self) -> None:
        with self.assertRaisesRegex(
            DeviceNotAvailableError, "no GPU execution provider"
        ):
            resolve_providers("GPU", CPU_ONLY)

    def test_full_provider_name_is_case_insensitive_for_old_launcher_settings(self) -> None:
        chain, target = resolve_providers("TENSORRTEXECUTIONPROVIDER", ALL_PROVIDERS)

        self.assertEqual(target, "TensorrtExecutionProvider")
        self.assertEqual(
            chain,
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def test_scanner_provider_name_survives_cli_uppercase_boundary(self) -> None:
        parsed = parse_args(
            [
                "--backend",
                "onnxruntime",
                "--device",
                "TensorrtExecutionProvider",
            ]
        )
        self.assertEqual(parsed.device, "TENSORRTEXECUTIONPROVIDER")

        chain, target = resolve_providers(parsed.device, ALL_PROVIDERS)
        self.assertEqual(target, "TensorrtExecutionProvider")
        self.assertEqual(chain[:2], ["TensorrtExecutionProvider", "CUDAExecutionProvider"])

    def test_full_provider_names_resolve_case_insensitively(self) -> None:
        expected = {
            "cudaexecutionprovider": "CUDAExecutionProvider",
            "RoCmExecutionProvider": "ROCMExecutionProvider",
            "DMLEXECUTIONPROVIDER": "DmlExecutionProvider",
            "cpuexecutionprovider": "CPUExecutionProvider",
        }
        for requested, provider in expected.items():
            with self.subTest(requested=requested):
                _, target = resolve_providers(requested, ALL_PROVIDERS)
                self.assertEqual(target, provider)

    def test_known_full_provider_name_reports_not_installed(self) -> None:
        with self.assertRaisesRegex(DeviceNotAvailableError, "not installed"):
            resolve_providers("TENSORRTEXECUTIONPROVIDER", CPU_ONLY)

    def test_every_chain_ends_with_cpu_so_partial_graphs_still_run(self) -> None:
        for requested in ("ROCM", "CUDA", "DIRECTML"):
            chain, _ = resolve_providers(requested, ALL_PROVIDERS)
            self.assertEqual(chain[-1], "CPUExecutionProvider", requested)
            self.assertEqual(len(chain), 2, requested)

    def test_rocm_aliases_resolve_to_legacy_rocm_provider(self) -> None:
        for alias in ("ROCM", "rocm"):
            _, target = resolve_providers(alias, ALL_PROVIDERS)
            self.assertEqual(target, "ROCMExecutionProvider", alias)

    def test_amd_alias_prefers_migraphx_then_legacy_rocm(self) -> None:
        _, target = resolve_providers("AMD", ALL_PROVIDERS)
        self.assertEqual(target, "MIGraphXExecutionProvider")

        _, target = resolve_providers(
            "AMD", ("ROCMExecutionProvider", "CPUExecutionProvider")
        )
        self.assertEqual(target, "ROCMExecutionProvider")

        with self.assertRaisesRegex(DeviceNotAvailableError, "neither the MIGraphX"):
            resolve_providers("AMD", CPU_ONLY)

    def test_windows_amd_alias_resolves_to_directml(self) -> None:
        for alias in ("DIRECTML", "DML", "DIRECTML:1", "dml:2"):
            _, target = resolve_providers(alias, ALL_PROVIDERS)
            self.assertEqual(target, "DmlExecutionProvider", alias)

    def test_directml_adapter_index_must_be_non_negative_integer(self) -> None:
        for requested in ("DIRECTML:-1", "DML:gpu", "DIRECTML:"):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                resolve_providers(requested, ALL_PROVIDERS)

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


class SessionOptionsTests(unittest.TestCase):
    class _FakeRuntime:
        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = object()

        class ExecutionMode:
            ORT_SEQUENTIAL = object()

    class _Options:
        pass

    def test_directml_chain_uses_its_mandatory_session_options(self) -> None:
        options = self._Options()
        configured = _configure_session_options(
            self._FakeRuntime(),
            options,
            ["DmlExecutionProvider", "CPUExecutionProvider"],
        )

        self.assertIsNotNone(options.graph_optimization_level)
        self.assertIsNotNone(options.execution_mode)
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(configured["execution_mode"], "ORT_SEQUENTIAL")
        self.assertFalse(configured["enable_mem_pattern"])
        self.assertFalse(hasattr(options, "intra_op_num_threads"))

    def test_cpu_only_chain_keeps_default_thread_pool(self) -> None:
        options = self._Options()
        configured = _configure_session_options(
            self._FakeRuntime(),
            options,
            ["CPUExecutionProvider"],
        )

        self.assertIsNotNone(options.graph_optimization_level)
        self.assertFalse(hasattr(options, "execution_mode"))
        self.assertFalse(hasattr(options, "enable_mem_pattern"))
        self.assertFalse(hasattr(options, "intra_op_num_threads"))
        self.assertFalse(hasattr(options, "inter_op_num_threads"))
        self.assertNotIn("enable_mem_pattern", configured)

    def test_cuda_chain_keeps_cuda_memory_and_thread_defaults(self) -> None:
        options = self._Options()
        configured = _configure_session_options(
            self._FakeRuntime(),
            options,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        self.assertFalse(hasattr(options, "enable_mem_pattern"))
        self.assertFalse(hasattr(options, "intra_op_num_threads"))
        self.assertEqual(
            configured,
            {"graph_optimization_level": "ORT_ENABLE_ALL"},
        )


class NvidiaRuntimePreparationTests(unittest.TestCase):
    class _Runtime:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def preload_dlls(self, **kwargs) -> None:
            self.calls.append(kwargs)

    def test_cuda_preloads_nvidia_site_package_libraries(self) -> None:
        runtime = self._Runtime()

        attempted = _preload_nvidia_libraries(
            runtime,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        self.assertTrue(attempted)
        self.assertEqual(runtime.calls, [{"directory": ""}])

    def test_tensorrt_chain_preloads_once(self) -> None:
        runtime = self._Runtime()

        attempted = _preload_nvidia_libraries(
            runtime,
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        self.assertTrue(attempted)
        self.assertEqual(len(runtime.calls), 1)

    def test_non_nvidia_provider_does_not_preload(self) -> None:
        runtime = self._Runtime()

        attempted = _preload_nvidia_libraries(
            runtime,
            ["DmlExecutionProvider", "CPUExecutionProvider"],
        )

        self.assertFalse(attempted)
        self.assertEqual(runtime.calls, [])

    def test_preload_failure_is_actionable(self) -> None:
        class BrokenRuntime:
            @staticmethod
            def preload_dlls(**_kwargs) -> None:
                raise OSError("missing CUDA DLL")

        with self.assertRaisesRegex(DetectorError, "matching NVIDIA CUDA build"):
            _preload_nvidia_libraries(
                BrokenRuntime(),
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

    def test_missing_preload_api_keeps_source_install_compatible(self) -> None:
        self.assertFalse(
            _preload_nvidia_libraries(
                object(),
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        )

    def test_requested_accelerator_must_be_active(self) -> None:
        with self.assertRaisesRegex(
            DeviceNotAvailableError,
            "did not initialize",
        ):
            _require_active_provider(
                "CUDAExecutionProvider",
                ["CPUExecutionProvider"],
            )

    def test_requested_accelerator_accepts_active_fallback_chain(self) -> None:
        _require_active_provider(
            "CUDAExecutionProvider",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


class ProviderDiagnosticsTests(unittest.TestCase):
    def test_provider_options_are_normalized_for_json(self) -> None:
        class Session:
            @staticmethod
            def get_provider_options():
                return {
                    "CUDAExecutionProvider": {
                        "device_id": "0",
                        "opaque": Path("cache"),
                    },
                    "CPUExecutionProvider": {},
                }

        options, status = _provider_options_record(Session())

        self.assertEqual(status, "ok")
        self.assertEqual(options["CUDAExecutionProvider"]["device_id"], "0")
        self.assertEqual(options["CUDAExecutionProvider"]["opaque"], "cache")

    def test_provider_diagnostics_do_not_break_an_older_runtime(self) -> None:
        self.assertEqual(_provider_options_record(object()), ({}, "unavailable"))

    def test_provider_diagnostic_failure_omits_machine_specific_message(self) -> None:
        class Session:
            @staticmethod
            def get_provider_options():
                raise OSError("secret/install/path")

        options, status = _provider_options_record(Session())

        self.assertEqual(options, {})
        self.assertEqual(status, "query_failed:OSError")
        self.assertNotIn("secret", status)


class FakeTensorSpec:
    def __init__(
        self,
        name: str,
        shape: list,
        tensor_type: str = "tensor(float)",
    ) -> None:
        self.name = name
        self.shape = shape
        self.type = tensor_type


class FakeSession:
    def __init__(
        self,
        path: str,
        sess_options=None,
        providers=None,
        *,
        input_shape: list | None = None,
        input_type: str = "tensor(float)",
        input_count: int = 1,
        output_shape: list | None = None,
        output_count: int = 1,
        output: np.ndarray | None = None,
        fail: Exception | None = None,
        provider_options: dict | None = None,
    ) -> None:
        self.path = path
        self.provider_specs = list(providers or [])
        self.providers = [
            provider[0] if isinstance(provider, tuple) else provider
            for provider in self.provider_specs
        ]
        self._inputs = [
            FakeTensorSpec(
                "images" if index == 0 else f"input{index}",
                input_shape or [1, 3, 416, 416],
                input_type,
            )
            for index in range(input_count)
        ]
        self._outputs = [
            FakeTensorSpec(
                "output0" if index == 0 else f"output{index}",
                output_shape or [1, 300, 6],
            )
            for index in range(output_count)
        ]
        self._output = output if output is not None else np.zeros((1, 300, 6), np.float32)
        self._fail = fail
        self._provider_options = (
            provider_options
            if provider_options is not None
            else {provider: {} for provider in self.providers}
        )
        self.fallback_disabled = False
        self.runs = 0

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return self.providers

    def get_provider_options(self):
        return self._provider_options

    def disable_fallback(self):
        self.fallback_disabled = True

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
        self.assertEqual(summary["requested_device_input"], "CPU")
        self.assertEqual(summary["requested_provider"], "CPUExecutionProvider")
        self.assertEqual(summary["provider_options_status"], "ok")
        self.assertEqual(summary["provider_options"], {"CPUExecutionProvider": {}})
        self.assertEqual(summary["provider_option_overrides"], {})
        self.assertNotIn("execution_mode", summary["configured_session_options"])

    def test_summary_records_runtime_reported_provider_options(self) -> None:
        detector = make_detector(
            session_kwargs={
                "provider_options": {
                    "CPUExecutionProvider": {"arena_extend_strategy": "0"}
                }
            }
        )

        self.assertEqual(
            detector.runtime_summary["provider_options"],
            {"CPUExecutionProvider": {"arena_extend_strategy": "0"}},
        )

    def test_directml_device_id_is_bound_and_reported(self) -> None:
        captured: dict[str, object] = {}

        class Runtime:
            class GraphOptimizationLevel:
                ORT_ENABLE_ALL = object()

            class ExecutionMode:
                ORT_SEQUENTIAL = object()

            class SessionOptions:
                pass

            @staticmethod
            def get_available_providers():
                return ["DmlExecutionProvider", "CPUExecutionProvider"]

        def factory(path, sess_options=None, providers=None):
            captured["providers"] = providers
            return FakeSession(path, sess_options, providers)

        with patch(
            "detection.onnx_yolo._load_onnxruntime",
            return_value=(Runtime, "test"),
        ):
            detector = OnnxRuntimeYoloDetector(
                model_path="unused.onnx",
                labels_path=None,
                device="DIRECTML:1",
                inference_size=416,
                session_factory=factory,
            )

        self.assertEqual(
            captured["providers"],
            [
                ("DmlExecutionProvider", {"device_id": "1"}),
                "CPUExecutionProvider",
            ],
        )
        self.assertEqual(
            detector.runtime_summary["provider_option_overrides"],
            {"DmlExecutionProvider": {"device_id": "1"}},
        )

    def test_full_provider_qualification_disables_cpu_graph_fallback(self) -> None:
        captured: dict[str, object] = {}

        class Runtime:
            class GraphOptimizationLevel:
                ORT_ENABLE_ALL = object()

            class ExecutionMode:
                ORT_SEQUENTIAL = object()

            class SessionOptions:
                def __init__(self) -> None:
                    self.entries: list[tuple[str, str]] = []

                def add_session_config_entry(self, key: str, value: str) -> None:
                    self.entries.append((key, value))

            @staticmethod
            def get_available_providers():
                return ["DmlExecutionProvider", "CPUExecutionProvider"]

        def factory(path, sess_options=None, providers=None):
            captured["providers"] = providers
            captured["entries"] = list(sess_options.entries)
            session = FakeSession(path, sess_options, providers)
            captured["session"] = session
            return session

        with patch(
            "detection.onnx_yolo._load_onnxruntime",
            return_value=(Runtime, "test"),
        ):
            detector = OnnxRuntimeYoloDetector(
                model_path="unused.onnx",
                device="DIRECTML:1",
                inference_size=416,
                session_factory=factory,
                require_full_provider=True,
            )

        self.assertEqual(
            captured["providers"],
            [("DmlExecutionProvider", {"device_id": "1"})],
        )
        self.assertEqual(
            captured["entries"],
            [("session.disable_cpu_ep_fallback", "1")],
        )
        self.assertTrue(detector.runtime_summary["require_full_provider"])
        self.assertTrue(
            detector.runtime_summary["configured_session_options"][
                "disable_cpu_ep_fallback"
            ]
        )
        self.assertTrue(captured["session"].fallback_disabled)
        self.assertTrue(
            detector.runtime_summary["runtime_ep_fail_fallback_disabled"]
        )

    def test_full_provider_qualification_requires_epfail_fallback_control(self) -> None:
        class Runtime:
            class GraphOptimizationLevel:
                ORT_ENABLE_ALL = object()

            class ExecutionMode:
                ORT_SEQUENTIAL = object()

            class SessionOptions:
                def add_session_config_entry(self, key: str, value: str) -> None:
                    pass

            @staticmethod
            def get_available_providers():
                return ["DmlExecutionProvider", "CPUExecutionProvider"]

        class SessionWithoutFallbackControl:
            def get_providers(self):
                return ["DmlExecutionProvider", "CPUExecutionProvider"]

        with patch(
            "detection.onnx_yolo._load_onnxruntime",
            return_value=(Runtime, "test"),
        ), self.assertRaisesRegex(ModelError, "execution-provider failure fallback"):
            OnnxRuntimeYoloDetector(
                model_path="unused.onnx",
                device="DIRECTML",
                inference_size=416,
                require_full_provider=True,
                session_factory=lambda *args, **kwargs: SessionWithoutFallbackControl(),
            )

    def test_full_provider_qualification_rejects_cpu(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an accelerator"):
            make_detector(require_full_provider=True)

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

    def test_input_contract_requires_rank4_batch1_and_three_channels(self) -> None:
        invalid_shapes = (
            [2, 3, 416, 416],
            [1, 1, 416, 416],
            [1, 3, 416],
            [1, 3, 416, 416, 1],
            ["batch", 3, 416, 416],
        )
        for shape in invalid_shapes:
            with self.subTest(shape=shape), self.assertRaises(ModelError):
                make_detector(session_kwargs={"input_shape": shape})

    def test_input_contract_requires_float32(self) -> None:
        with self.assertRaises(ModelError) as raised:
            make_detector(session_kwargs={"input_type": "tensor(float16)"})

        self.assertIn("float32", str(raised.exception))

    def test_model_must_have_exactly_one_input_and_output(self) -> None:
        for session_kwargs in ({"input_count": 2}, {"output_count": 2}):
            with self.subTest(session_kwargs=session_kwargs), self.assertRaises(ModelError):
                make_detector(session_kwargs=session_kwargs)

    def test_output_shape_must_match_a_supported_decoder_layout(self) -> None:
        with self.assertRaises(ModelError) as raised:
            make_detector(
                labels_path="models/coco80.txt",
                session_kwargs={"output_shape": [1, 10, 10]},
            )

        self.assertIn("Unsupported ONNX YOLO output shape", str(raised.exception))

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

    def test_postprocess_override_does_not_change_configured_default(self) -> None:
        raw = np.zeros((1, 300, 6), np.float32)
        raw[0, 0] = (100.0, 120.0, 200.0, 300.0, 0.9, 0.0)
        raw[0, 1] = (220.0, 120.0, 320.0, 300.0, 0.18, 0.0)
        detector = make_detector(
            confidence=0.25,
            session_kwargs={"output": raw},
        )
        inferred = detector.infer(np.zeros((1, 3, 416, 416), np.float32))

        configured = detector.postprocess(inferred, frame_shape=(416, 416, 3))
        continuation_decode = detector.postprocess(
            inferred,
            frame_shape=(416, 416, 3),
            confidence=0.15,
        )

        self.assertEqual(len(configured), 1)
        self.assertAlmostEqual(configured[0].confidence, 0.9, places=5)
        self.assertEqual(len(continuation_decode), 2)
        self.assertAlmostEqual(continuation_decode[1].confidence, 0.18, places=5)
        self.assertEqual(detector.confidence, 0.25)

        for invalid in (-0.1, 1.1, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                detector.postprocess(
                    inferred,
                    frame_shape=(416, 416, 3),
                    confidence=invalid,
                )


if __name__ == "__main__":
    unittest.main()
