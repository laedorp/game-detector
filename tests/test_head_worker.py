from __future__ import annotations

from dataclasses import FrozenInstanceError
from threading import Event
import time
import unittest

import numpy as np

from detection.base import DeviceNotAvailableError, ModelError
from detection.head_worker import (
    CPU_PROVIDER,
    MIGRAPHX_PROVIDER,
    CpuOnnxSession,
    HeadObservation,
    LatestHeadWorker,
    OnnxModelContract,
    OnnxTensorContract,
    StrictProviderOnnxSession,
)


def wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


DIRECT_HEAD_CONTRACT = OnnxModelContract(
    input=OnnxTensorContract("images", (1, 3, 320, 320)),
    output=OnnxTensorContract("output0", (1, 6, 2100)),
)


class FakeValueInfo:
    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        element_type: str = "tensor(float)",
    ) -> None:
        self.name = name
        self.shape = list(shape)
        self.type = element_type


class FakeSession:
    def __init__(
        self,
        contract: OnnxModelContract = DIRECT_HEAD_CONTRACT,
        *,
        providers: tuple[str, ...] = (CPU_PROVIDER,),
    ) -> None:
        self.contract = contract
        self.providers = providers
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []
        self.fallback_disabled = False

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def get_inputs(self) -> list[FakeValueInfo]:
        value = self.contract.input
        return [FakeValueInfo(value.name, value.shape, value.element_type)]

    def get_outputs(self) -> list[FakeValueInfo]:
        value = self.contract.output
        return [FakeValueInfo(value.name, value.shape, value.element_type)]

    def disable_fallback(self) -> None:
        self.fallback_disabled = True

    def run(
        self,
        names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        self.calls.append((names, inputs))
        return [np.zeros(self.contract.output.shape, dtype=np.float32)]


class FakeRuntime:
    __version__ = "test-runtime"

    class ExecutionMode:
        ORT_SEQUENTIAL = object()

    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = object()

    class SessionOptions:
        def __init__(self) -> None:
            self.entries: list[tuple[str, str]] = []

        def add_session_config_entry(self, key: str, value: str) -> None:
            self.entries.append((key, value))

    def __init__(self, providers: tuple[str, ...] = (CPU_PROVIDER,)) -> None:
        self.providers = providers

    def get_available_providers(self) -> list[str]:
        return list(self.providers)


class CpuOnnxSessionTests(unittest.TestCase):
    def test_factory_pins_cpu_provider_threads_and_exact_contract(self) -> None:
        runtime = FakeRuntime()
        fake_session = FakeSession()
        factory_calls: list[tuple[str, object, list[str]]] = []

        def factory(path, *, sess_options, providers):
            factory_calls.append((path, sess_options, providers))
            return fake_session

        session = CpuOnnxSession(
            "synthetic.onnx",
            DIRECT_HEAD_CONTRACT,
            intra_op_threads=6,
            runtime_module=runtime,
            session_factory=factory,
        )

        self.assertEqual(len(factory_calls), 1)
        _, options, providers = factory_calls[0]
        self.assertEqual(providers, [CPU_PROVIDER])
        self.assertEqual(options.intra_op_num_threads, 6)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertIs(
            options.execution_mode,
            runtime.ExecutionMode.ORT_SEQUENTIAL,
        )
        self.assertFalse(hasattr(options, "graph_optimization_level"))
        self.assertEqual(session.info.provider, CPU_PROVIDER)
        self.assertEqual(session.info.runtime_version, "test-runtime")

        tensor = np.zeros((1, 3, 320, 320), dtype=np.float32)
        output = session.infer(tensor)
        self.assertEqual(output.shape, (1, 6, 2100))
        self.assertEqual(fake_session.calls[0][0], ["output0"])
        self.assertEqual(list(fake_session.calls[0][1]), ["images"])

    def test_factory_rejects_missing_or_extra_active_provider(self) -> None:
        runtime = FakeRuntime(providers=("AzureExecutionProvider",))
        with self.assertRaises(DeviceNotAvailableError):
            CpuOnnxSession(
                "synthetic.onnx",
                DIRECT_HEAD_CONTRACT,
                runtime_module=runtime,
                session_factory=lambda *_args, **_kwargs: FakeSession(),
            )

        runtime = FakeRuntime()
        with self.assertRaisesRegex(ModelError, "unexpected provider chain"):
            CpuOnnxSession(
                "synthetic.onnx",
                DIRECT_HEAD_CONTRACT,
                runtime_module=runtime,
                session_factory=lambda *_args, **_kwargs: FakeSession(
                    providers=(CPU_PROVIDER, "AzureExecutionProvider")
                ),
            )

    def test_factory_rejects_any_contract_drift(self) -> None:
        wrong = OnnxModelContract(
            input=OnnxTensorContract("images", (1, 3, 640, 640)),
            output=DIRECT_HEAD_CONTRACT.output,
        )
        runtime = FakeRuntime()

        with self.assertRaisesRegex(ModelError, "input shape"):
            CpuOnnxSession(
                "synthetic.onnx",
                DIRECT_HEAD_CONTRACT,
                runtime_module=runtime,
                session_factory=lambda *_args, **_kwargs: FakeSession(wrong),
            )

    def test_infer_rejects_wrong_shape_and_dtype_before_runtime(self) -> None:
        session = CpuOnnxSession(
            "synthetic.onnx",
            DIRECT_HEAD_CONTRACT,
            runtime_module=FakeRuntime(),
            session_factory=lambda *_args, **_kwargs: FakeSession(),
        )

        with self.assertRaisesRegex(ValueError, "shape"):
            session.infer(np.zeros((1, 3, 640, 640), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "float32"):
            session.infer(np.zeros((1, 3, 320, 320), dtype=np.float16))


class StrictProviderOnnxSessionTests(unittest.TestCase):
    def test_factory_registers_only_migraphx_and_disables_both_fallbacks(self) -> None:
        runtime = FakeRuntime(providers=(MIGRAPHX_PROVIDER, CPU_PROVIDER))
        fake_session = FakeSession(providers=(MIGRAPHX_PROVIDER,))
        captured: dict[str, object] = {}

        def factory(path, *, sess_options, providers):
            captured["path"] = path
            captured["providers"] = providers
            captured["options"] = sess_options
            return fake_session

        session = StrictProviderOnnxSession(
            "synthetic.onnx",
            DIRECT_HEAD_CONTRACT,
            provider=MIGRAPHX_PROVIDER,
            runtime_module=runtime,
            session_factory=factory,
        )

        self.assertEqual(captured["providers"], [MIGRAPHX_PROVIDER])
        options = captured["options"]
        self.assertEqual(
            options.entries,
            [("session.disable_cpu_ep_fallback", "1")],
        )
        self.assertIs(
            options.graph_optimization_level,
            runtime.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )
        self.assertTrue(fake_session.fallback_disabled)
        self.assertEqual(session.info.provider, MIGRAPHX_PROVIDER)
        self.assertTrue(session.info.cpu_graph_fallback_disabled)
        self.assertTrue(session.info.ep_failure_fallback_disabled)
        self.assertEqual(session.info.active_providers, (MIGRAPHX_PROVIDER,))
        self.assertNotIn(CPU_PROVIDER, captured["providers"])

        output = session.infer(
            np.zeros(DIRECT_HEAD_CONTRACT.input.shape, dtype=np.float32)
        )
        self.assertEqual(output.shape, DIRECT_HEAD_CONTRACT.output.shape)

    def test_unavailable_migraphx_and_explicit_cpu_fail_closed(self) -> None:
        runtime = FakeRuntime(providers=(CPU_PROVIDER,))
        with self.assertRaisesRegex(DeviceNotAvailableError, "not installed"):
            StrictProviderOnnxSession(
                "synthetic.onnx",
                DIRECT_HEAD_CONTRACT,
                provider=MIGRAPHX_PROVIDER,
                runtime_module=runtime,
                session_factory=lambda *_args, **_kwargs: FakeSession(),
            )

        with self.assertRaisesRegex(ValueError, "refuses CPUExecutionProvider"):
            StrictProviderOnnxSession(
                "synthetic.onnx",
                DIRECT_HEAD_CONTRACT,
                provider=CPU_PROVIDER,
                runtime_module=runtime,
                session_factory=lambda *_args, **_kwargs: FakeSession(),
            )

    def test_implicit_cpu_registration_is_allowed_but_never_requested(self) -> None:
        runtime = FakeRuntime(providers=(MIGRAPHX_PROVIDER, CPU_PROVIDER))
        captured: dict[str, object] = {}

        def factory(_path, *, sess_options, providers):
            captured["providers"] = providers
            return FakeSession(providers=(MIGRAPHX_PROVIDER, CPU_PROVIDER))

        session = StrictProviderOnnxSession(
            "synthetic.onnx",
            DIRECT_HEAD_CONTRACT,
            provider=MIGRAPHX_PROVIDER,
            runtime_module=runtime,
            session_factory=factory,
        )

        self.assertEqual(captured["providers"], [MIGRAPHX_PROVIDER])
        self.assertEqual(
            session.info.active_providers,
            (MIGRAPHX_PROVIDER, CPU_PROVIDER),
        )
        self.assertTrue(session.info.cpu_graph_fallback_disabled)

    def test_alternate_or_reordered_active_provider_is_rejected(self) -> None:
        runtime = FakeRuntime(providers=(MIGRAPHX_PROVIDER, CPU_PROVIDER))
        invalid_chains = (
            (CPU_PROVIDER,),
            (CPU_PROVIDER, MIGRAPHX_PROVIDER),
            (MIGRAPHX_PROVIDER, "DnnlExecutionProvider", CPU_PROVIDER),
        )
        for active in invalid_chains:
            with self.subTest(active=active), self.assertRaisesRegex(
                ModelError,
                "unexpected provider chain",
            ):
                StrictProviderOnnxSession(
                    "synthetic.onnx",
                    DIRECT_HEAD_CONTRACT,
                    provider=MIGRAPHX_PROVIDER,
                    runtime_module=runtime,
                    session_factory=lambda *_args, **_kwargs: FakeSession(
                        providers=active
                    ),
                )

    def test_missing_runtime_ep_failure_control_is_rejected(self) -> None:
        runtime = FakeRuntime(providers=(MIGRAPHX_PROVIDER, CPU_PROVIDER))

        class SessionWithoutFallbackControl:
            def get_providers(self):
                return [MIGRAPHX_PROVIDER]

        with self.assertRaisesRegex(ModelError, "failure fallback"):
            StrictProviderOnnxSession(
                "synthetic.onnx",
                DIRECT_HEAD_CONTRACT,
                provider=MIGRAPHX_PROVIDER,
                runtime_module=runtime,
                session_factory=lambda *_args, **_kwargs: SessionWithoutFallbackControl(),
            )

    def test_provider_drift_after_run_fails_closed(self) -> None:
        runtime = FakeRuntime(providers=(MIGRAPHX_PROVIDER, CPU_PROVIDER))

        class DriftingSession(FakeSession):
            def run(self, names, inputs):
                output = super().run(names, inputs)
                self.providers = (CPU_PROVIDER,)
                return output

        session = StrictProviderOnnxSession(
            "synthetic.onnx",
            DIRECT_HEAD_CONTRACT,
            provider=MIGRAPHX_PROVIDER,
            runtime_module=runtime,
            session_factory=lambda *_args, **_kwargs: DriftingSession(
                providers=(MIGRAPHX_PROVIDER,)
            ),
        )

        with self.assertRaisesRegex(ModelError, "unexpected provider chain"):
            session.infer(
                np.zeros(DIRECT_HEAD_CONTRACT.input.shape, dtype=np.float32)
            )

    def test_exact_pinned_contract_is_still_required(self) -> None:
        runtime = FakeRuntime(providers=(MIGRAPHX_PROVIDER, CPU_PROVIDER))
        wrong = OnnxModelContract(
            input=OnnxTensorContract("images", (1, 3, 640, 640)),
            output=DIRECT_HEAD_CONTRACT.output,
        )
        with self.assertRaisesRegex(ModelError, "Strict GPU head model input shape"):
            StrictProviderOnnxSession(
                "synthetic.onnx",
                DIRECT_HEAD_CONTRACT,
                provider=MIGRAPHX_PROVIDER,
                runtime_module=runtime,
                session_factory=lambda *_args, **_kwargs: FakeSession(
                    wrong,
                    providers=(MIGRAPHX_PROVIDER,),
                ),
            )


def observation_for(payload: np.ndarray) -> HeadObservation:
    value = float(payload.flat[0])
    return HeadObservation(
        point=(value, value + 0.5),
        confidence=0.9,
        evidence="synthetic-head",
    )


class BlockingLocalizer:
    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()
        self.seen: list[int] = []
        self._first = True

    def __call__(self, payload: np.ndarray, _box) -> HeadObservation:
        if self._first:
            self._first = False
            self.entered.set()
            self.release.wait(2.0)
        self.seen.append(int(payload.flat[0]))
        return observation_for(payload)


class LatestHeadWorkerTests(unittest.TestCase):
    def test_pending_mailbox_overwrites_and_payload_is_owned(self) -> None:
        localizer = BlockingLocalizer()
        worker = LatestHeadWorker(localizer, stop_timeout_s=0.1)
        self.addCleanup(localizer.release.set)
        self.addCleanup(lambda: worker.stop(timeout_s=1.0))
        worker.start()

        first = np.full((2, 2), 1, dtype=np.uint8)
        self.assertTrue(
            worker.submit(
                first,
                source_timestamp_ns=1,
                identity_generation=4,
                selected_player_box=(10, 20, 30, 60),
            )
        )
        first.fill(99)
        self.assertTrue(localizer.entered.wait(1.0))
        for timestamp, value in ((2, 2), (3, 3)):
            self.assertTrue(
                worker.submit(
                    np.full((2, 2), value, dtype=np.uint8),
                    source_timestamp_ns=timestamp,
                    identity_generation=4,
                    selected_player_box=(10, 20, 30, 60),
                )
            )
        localizer.release.set()

        self.assertTrue(wait_until(lambda: worker.status.jobs_completed == 2))
        result = worker.take_latest(4)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(localizer.seen, [1, 3])
        self.assertEqual(result.source_timestamp_ns, 3)
        self.assertEqual(result.selected_player_box, (10.0, 20.0, 30.0, 60.0))
        self.assertEqual(result.head_point, (3.0, 3.5))
        status = worker.status
        self.assertEqual(status.accepted_submissions, 3)
        self.assertEqual(status.pending_overwrites, 1)
        self.assertEqual(status.jobs_started, 2)
        self.assertEqual(status.result_overwrites, 1)

    def test_changed_identity_drops_inflight_result(self) -> None:
        localizer = BlockingLocalizer()
        worker = LatestHeadWorker(localizer)
        self.addCleanup(localizer.release.set)
        self.addCleanup(lambda: worker.stop(timeout_s=1.0))
        worker.start()
        self.assertTrue(
            worker.submit(
                np.full((1, 1), 1, dtype=np.uint8),
                source_timestamp_ns=10,
                identity_generation=1,
                selected_player_box=(0, 0, 10, 20),
            )
        )
        self.assertTrue(localizer.entered.wait(1.0))
        self.assertTrue(
            worker.submit(
                np.full((1, 1), 2, dtype=np.uint8),
                source_timestamp_ns=20,
                identity_generation=2,
                selected_player_box=(20, 20, 40, 60),
            )
        )
        localizer.release.set()

        self.assertTrue(wait_until(lambda: worker.status.jobs_completed == 2))
        result = worker.take_latest(2)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.identity_generation, 2)
        self.assertEqual(result.head_point_if_current(1), None)
        self.assertEqual(result.head_point_if_current(2), (2.0, 2.5))
        self.assertGreaterEqual(worker.status.stale_results_dropped, 1)

    def test_old_identity_and_out_of_order_frame_are_rejected(self) -> None:
        worker = LatestHeadWorker(lambda payload, _box: observation_for(payload))
        self.addCleanup(lambda: worker.stop(timeout_s=1.0))
        worker.start()
        self.assertTrue(
            worker.submit(
                np.zeros((1, 1), dtype=np.uint8),
                source_timestamp_ns=20,
                identity_generation=2,
                selected_player_box=(0, 0, 10, 20),
            )
        )
        self.assertFalse(
            worker.submit(
                np.zeros((1, 1), dtype=np.uint8),
                source_timestamp_ns=30,
                identity_generation=1,
                selected_player_box=(0, 0, 10, 20),
            )
        )
        self.assertFalse(
            worker.submit(
                np.zeros((1, 1), dtype=np.uint8),
                source_timestamp_ns=19,
                identity_generation=2,
                selected_player_box=(0, 0, 10, 20),
            )
        )
        self.assertEqual(worker.status.stale_submissions, 2)

    def test_no_head_publishes_none_without_box_heuristic(self) -> None:
        worker = LatestHeadWorker(lambda _payload, _box: None)
        self.addCleanup(lambda: worker.stop(timeout_s=1.0))
        worker.start()
        selected_box = (100, 50, 200, 300)
        self.assertTrue(
            worker.submit(
                np.zeros((1, 1), dtype=np.uint8),
                source_timestamp_ns=123,
                identity_generation=7,
                selected_player_box=selected_box,
            )
        )

        self.assertTrue(wait_until(lambda: worker.status.jobs_completed == 1))
        result = worker.take_latest(7)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIsNone(result.observation)
        self.assertIsNone(result.head_point)
        self.assertEqual(result.selected_player_box, selected_box)
        self.assertEqual(worker.status.no_head_results, 1)

    def test_result_is_frozen_and_keeps_absolute_source_point(self) -> None:
        worker = LatestHeadWorker(lambda payload, _box: observation_for(payload))
        self.addCleanup(lambda: worker.stop(timeout_s=1.0))
        worker.start()
        self.assertTrue(
            worker.submit(
                np.full((1, 1), 8, dtype=np.uint8),
                source_timestamp_ns=50,
                identity_generation=3,
                selected_player_box=(0, 0, 10, 20),
            )
        )
        self.assertTrue(wait_until(lambda: worker.status.result_pending))
        result = worker.take_latest(3)
        assert result is not None

        self.assertEqual(result.head_point, (8.0, 8.5))
        with self.assertRaises(FrozenInstanceError):
            result.source_timestamp_ns = 51  # type: ignore[misc]

    def test_stop_is_bounded_and_discards_result_from_inflight_work(self) -> None:
        localizer = BlockingLocalizer()
        worker = LatestHeadWorker(localizer, stop_timeout_s=0.01)
        self.addCleanup(localizer.release.set)
        self.addCleanup(lambda: worker.stop(timeout_s=1.0))
        worker.start()
        self.assertTrue(
            worker.submit(
                np.ones((1, 1), dtype=np.uint8),
                source_timestamp_ns=1,
                identity_generation=1,
                selected_player_box=(0, 0, 10, 20),
            )
        )
        self.assertTrue(localizer.entered.wait(1.0))

        self.assertFalse(worker.stop())
        localizer.release.set()
        self.assertTrue(worker.stop(timeout_s=1.0))
        self.assertEqual(worker.status.lifecycle, "stopped")
        self.assertIsNone(worker.take_latest(1))
        self.assertGreaterEqual(worker.status.stopped_results_dropped, 1)
        self.assertFalse(
            worker.submit(
                np.ones((1, 1), dtype=np.uint8),
                source_timestamp_ns=2,
                identity_generation=1,
                selected_player_box=(0, 0, 10, 20),
            )
        )

    def test_worker_exception_is_contained_and_reported(self) -> None:
        def fail(_payload, _box):
            raise ValueError("synthetic inference failure")

        worker = LatestHeadWorker(fail)
        self.addCleanup(lambda: worker.stop(timeout_s=1.0))
        worker.start()
        self.assertTrue(
            worker.submit(
                np.zeros((1, 1), dtype=np.uint8),
                source_timestamp_ns=1,
                identity_generation=1,
                selected_player_box=(0, 0, 10, 20),
            )
        )

        self.assertTrue(wait_until(lambda: worker.status.lifecycle == "failed"))
        status = worker.status
        self.assertEqual(status.failures, 1)
        self.assertIn("synthetic inference failure", status.last_error or "")
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            worker.raise_if_failed()
        self.assertTrue(worker.stop())


if __name__ == "__main__":
    unittest.main()
