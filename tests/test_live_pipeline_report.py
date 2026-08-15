from __future__ import annotations

from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import tempfile
from time import perf_counter_ns, sleep
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from aiming.makcu import MakcuTelemetrySnapshot
from capture.base import CaptureStats, FramePacket
from config import parse_args
from detection.hardware import DirectMLAdapter
from detection.types import Detection
from main import run
from utils.live_report import build_live_report, write_json_atomic_new
from utils.metrics import FrameTimings, RollingMetrics
from utils.preview import PreviewStats
from utils.inference_size import normalize_inference_size


class _FakeDetector:
    instances: list["_FakeDetector"] = []
    detection_batches: list[list[Detection]] = []
    fail_infer_call: int | None = None
    inference_delay_seconds: float = 0.0

    def __init__(self, **arguments) -> None:
        self.arguments = arguments
        height, width = normalize_inference_size(arguments["inference_size"])
        self.available_devices = ("CPU",)
        self.infer_calls = 0
        self.postprocess_calls = 0
        self.postprocess_confidences: list[float | None] = []
        self.tensor_shapes: list[tuple[int, ...]] = []
        self.runtime_summary = {
            "runtime": "test-runtime",
            "openvino_version": "1",
            "model_path": str(arguments["model_path"]),
            "requested_device": arguments["device"],
            "device": arguments["device"],
            "input_shape": [1, 3, height, width],
        }
        self.__class__.instances.append(self)

    def warmup(self) -> None:
        return None

    def infer(self, _tensor):
        self.infer_calls += 1
        self.tensor_shapes.append(tuple(_tensor.shape))
        if self.inference_delay_seconds:
            sleep(self.inference_delay_seconds)
        if self.fail_infer_call == self.infer_calls:
            raise RuntimeError("synthetic detail inference failure")
        return np.zeros((1, 0, 6), dtype=np.float32)

    def postprocess(self, _raw, *, transform, frame_shape, confidence=None):
        del transform, frame_shape
        self.postprocess_confidences.append(confidence)
        index = self.postprocess_calls
        self.postprocess_calls += 1
        if index < len(self.detection_batches):
            return list(self.detection_batches[index])
        return []


class _FakeSource:
    description = "synthetic live source"

    def __init__(
        self,
        *,
        static: bool = False,
        shape: tuple[int, int, int] = (32, 32, 3),
    ) -> None:
        self.static = static
        self.shape = shape
        self.started = False
        self.closed = False
        self.read_calls = 0
        self.read_timeouts: list[float | None] = []
        self.error = None
        self.ended = False

    @property
    def actual_settings(self):
        return {
            "backend": "fake-latest-only",
            "width": self.shape[1],
            "height": self.shape[0],
            "fps": 120.0,
        }

    @property
    def stats(self) -> CaptureStats:
        delivered = 0 if self.static else self.read_calls
        return CaptureStats(
            frames_read=delivered,
            frames_delivered=delivered,
            frames_overwritten=2 if delivered else 0,
            read_failures=0,
        )

    def start(self) -> None:
        self.started = True

    def read(self, timeout: float | None = None):
        self.read_calls += 1
        self.read_timeouts.append(timeout)
        if self.static:
            sleep(float(timeout or 0.0))
            return None
        completed_ns = perf_counter_ns()
        return FramePacket(
            image=np.zeros(self.shape, dtype=np.uint8),
            sequence=self.read_calls - 1,
            read_started_ns=completed_ns - 100_000,
            read_completed_ns=completed_ns,
        )

    def close(self) -> None:
        self.closed = True


class _CloseTimeoutSource(_FakeSource):
    def close(self) -> None:
        super().close()
        self.error = (
            "Timed out waiting for capture-worker to stop; "
            "the capture source is still closing."
        )


class _ArtifactMutatingSource(_FakeSource):
    def __init__(self, artifact: Path) -> None:
        super().__init__()
        self.artifact = artifact

    def read(self, timeout: float | None = None):
        packet = super().read(timeout)
        self.artifact.write_bytes(b"mutated model")
        return packet


class _StuckPreview:
    mode = "threaded"
    stats: dict[str, object] = {}

    def start(self) -> None:
        return None

    def poll(self) -> bool:
        return True

    def submit(self, _frame) -> bool:
        return True

    def should_continue(self) -> bool:
        return True

    def stop(self) -> bool:
        return False

    def raise_if_failed(self) -> None:
        return None


def _sample(value: float) -> FrameTimings:
    return FrameTimings(*(value for _ in FrameTimings.__dataclass_fields__))


class LiveReportUnitTests(unittest.TestCase):
    def test_cli_parses_bounds_and_full_provider_gate(self) -> None:
        config = parse_args(
            [
                "--backend",
                "onnxruntime",
                "--metrics-json",
                "result.json",
                "--max-frames",
                "25",
                "--max-seconds",
                "3.5",
                "--require-full-provider",
            ]
        )

        self.assertEqual(config.metrics_json, Path("result.json"))
        self.assertEqual(config.max_frames, 25)
        self.assertEqual(config.max_seconds, 3.5)
        self.assertTrue(config.require_full_provider)

        for arguments in (("--max-frames", "0"), ("--max-seconds", "0")):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                parse_args(list(arguments))
        with self.assertRaises(SystemExit):
            parse_args(["--require-full-provider"])

    def test_atomic_writer_refuses_overwrite_and_leaves_no_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "nested" / "metrics.json"

            written = write_json_atomic_new(destination, {"schema": 1})

            self.assertEqual(written, destination)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"schema": 1})
            self.assertFalse(list(destination.parent.glob(".*.tmp")))
            original = destination.read_bytes()
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                write_json_atomic_new(destination, {"schema": 2})
            self.assertEqual(destination.read_bytes(), original)

    def test_report_records_every_timing_and_omits_aim_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            labels = root / "labels.txt"
            model.write_bytes(b"model")
            labels.write_text("player\n", encoding="utf-8")
            config = parse_args(
                [
                    "--model",
                    str(model),
                    "--labels",
                    str(labels),
                    "--backend",
                    "onnxruntime",
                    "--device",
                    "DIRECTML:1",
                    "--inference-size",
                    "416",
                    "--no-preview",
                    "--max-frames",
                    "3",
                ]
            )
            config = replace(
                config,
                aim_pairing_key="do-not-leak",
                aim_activate_path="/private/controller/device",
                aim_makcu_port="/private/serial/device",
                capture_rotate_180=True,
            )
            metrics = RollingMetrics(4)
            for index in range(1, 4):
                metrics.record(_sample(float(index)), index * 1_000_000)
            runtime = {
                "runtime": "ONNX Runtime",
                "requested_device_input": "DIRECTML:1",
                "requested_provider": "DmlExecutionProvider",
                "active_providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
                "provider_option_overrides": {
                    "DmlExecutionProvider": {"device_id": "1"}
                },
                "provider_options": {
                    "DmlExecutionProvider": {"device_id": "1"}
                },
            }

            report = build_live_report(
                config=config,
                detector_summary=runtime,
                source_description="Windows screen monitor 1",
                source_settings={
                    "backend": "dxcam-dxgi",
                    "preferred_backend": "dxcam-dxgi",
                    "fallback_reason": None,
                    "width": 1920,
                    "height": 1080,
                },
                capture_stats=CaptureStats(20, 3, 17, 0),
                preview_mode="disabled",
                preview_stats=PreviewStats(0, 0, 0),
                metrics=metrics.snapshot(),
                elapsed_seconds=0.25,
                started_utc="2026-08-13T00:00:00.000Z",
                completed_utc="2026-08-13T00:00:00.250Z",
                termination_reason="max_frames",
                directml_adapter_factory=lambda: (
                    DirectMLAdapter(1, "GeForce RTX 5060 Laptop GPU", "10de", "2d59", 8),
                ),
            )

            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(report["pipeline"]["processed_frames"], 3)
            self.assertEqual(report["pipeline"]["elapsed_fps"], 12.0)
            self.assertEqual(report["pipeline"]["update_fps"], 1000.0)
            self.assertEqual(report["capture"]["frames_overwritten"], 17)
            self.assertEqual(report["source"]["backend"], "dxcam-dxgi")
            self.assertEqual(
                report["config"]["capture"]["rotation_degrees"], 180
            )
            self.assertEqual(report["directml_adapter"]["effective_index"], 1)
            self.assertFalse(report["directml_adapter"]["requested_provider_mismatch"])
            self.assertEqual(
                report["directml_adapter"]["descriptor"]["name"],
                "GeForce RTX 5060 Laptop GPU",
            )
            expected_fields = set(FrameTimings.__dataclass_fields__)
            for summary in ("mean", "p50", "p95", "p99"):
                self.assertEqual(
                    set(report["pipeline"]["timings"][summary]), expected_fields
                )
            serialized = json.dumps(report)
            self.assertNotIn("do-not-leak", serialized)
            self.assertNotIn("/private/controller/device", serialized)
            self.assertNotIn("/private/serial/device", serialized)

    def test_runtime_provider_index_cannot_be_masked_by_configured_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            labels = root / "labels.txt"
            model.write_bytes(b"model")
            labels.write_text("player\n", encoding="utf-8")
            config = parse_args(
                [
                    "--model",
                    str(model),
                    "--labels",
                    str(labels),
                    "--backend",
                    "onnxruntime",
                    "--device",
                    "DIRECTML:1",
                    "--no-preview",
                ]
            )
            metrics = RollingMetrics(2).snapshot()
            report = build_live_report(
                config=config,
                detector_summary={
                    "runtime": "ONNX Runtime",
                    "requested_device_input": "DIRECTML:1",
                    "requested_provider": "DmlExecutionProvider",
                    "active_providers": ["DmlExecutionProvider"],
                    "provider_option_overrides": {
                        "DmlExecutionProvider": {"device_id": "1"}
                    },
                    "provider_options": {
                        "DmlExecutionProvider": {"device_id": "0"}
                    },
                },
                source_description="screen",
                source_settings={"backend": "dxcam-dxgi"},
                capture_stats=CaptureStats(),
                preview_mode="disabled",
                preview_stats=PreviewStats(0, 0, 0),
                metrics=metrics,
                elapsed_seconds=0.0,
                started_utc="2026-08-13T00:00:00.000Z",
                completed_utc="2026-08-13T00:00:00.000Z",
                termination_reason="source_ended",
                directml_adapter_factory=lambda: (),
            )

            adapter = report["directml_adapter"]
            self.assertEqual(adapter["requested_index"], 1)
            self.assertEqual(adapter["configured_index"], 1)
            self.assertEqual(adapter["provider_reported_index"], 0)
            self.assertEqual(adapter["effective_index"], 1)
            self.assertTrue(adapter["requested_provider_mismatch"])
            self.assertEqual(
                adapter["qualification_status"],
                "failed_provider_index_mismatch",
            )


class LivePipelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.xml"
        self.model.write_text("<model />", encoding="utf-8")
        self.model.with_suffix(".bin").write_bytes(b"weights")
        self.labels = self.root / "labels.txt"
        self.labels.write_text("player\n", encoding="utf-8")
        self.video = self.root / "capture.mp4"
        self.video.write_bytes(b"placeholder")
        _FakeDetector.instances.clear()
        _FakeDetector.detection_batches = []
        _FakeDetector.fail_infer_call = None
        _FakeDetector.inference_delay_seconds = 0.0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self, report: Path, *extra: str):
        return parse_args(
            [
                "--source",
                str(self.video),
                "--model",
                str(self.model),
                "--labels",
                str(self.labels),
                "--inference-size",
                "32",
                "--no-preview",
                "--metrics-json",
                str(report),
                *extra,
            ]
        )

    def _run_with(self, source: _FakeSource, config) -> int:
        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
        ):
            return run(config)

    def test_max_frames_uses_the_real_pipeline_and_is_exact(self) -> None:
        report_path = self.root / "three-frames.json"
        source = _FakeSource()

        result = self._run_with(
            source,
            self._config(report_path, "--max-frames", "3"),
        )

        self.assertEqual(result, 0)
        self.assertTrue(source.closed)
        self.assertEqual(source.read_calls, 3)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["termination"]["reason"], "max_frames")
        self.assertEqual(report["pipeline"]["processed_frames"], 3)
        self.assertEqual(report["capture"]["frames_delivered"], 3)
        self.assertEqual(
            report["model_artifact"]["companions"][0]["sha256"],
            "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
        )
        detector = _FakeDetector.instances[0]
        self.assertEqual(detector.infer_calls, 3)
        self.assertEqual(detector.postprocess_calls, 3)
        self.assertFalse(report["detail_pass"]["enabled"])
        self.assertEqual(report["config"]["inference"]["detail_crop_size"], None)
        for summary in ("mean", "p50", "p95", "p99"):
            timings = report["pipeline"]["timings"][summary]
            self.assertEqual(timings["detail_preprocess_ms"], 0.0)
            self.assertEqual(timings["detail_inference_ms"], 0.0)
            self.assertEqual(timings["detail_postprocess_ms"], 0.0)

    def test_makcu_automatic_startup_uses_equal_axis_caps(self) -> None:
        from aiming.controller import TargetTracker as RealTargetTracker

        report_path = self.root / "makcu-vertical-cap.json"

        class TimestampedSource(_FakeSource):
            def __init__(self) -> None:
                super().__init__()
                self.base_ns = perf_counter_ns() - 20_000_000

            def read(self, timeout: float | None = None):
                self.read_calls += 1
                self.read_timeouts.append(timeout)
                started_ns = self.base_ns + (self.read_calls - 1) * 8_000_000
                return FramePacket(
                    image=np.zeros(self.shape, dtype=np.uint8),
                    sequence=self.read_calls - 1,
                    read_started_ns=started_ns,
                    read_completed_ns=started_ns + 100_000,
                )

        source = TimestampedSource()
        raw_target = Detection(0, "player", 0.90, (20, 2, 28, 22))
        _FakeDetector.detection_batches = [[raw_target], []]
        tracker_options: list[dict[str, object]] = []
        cleanup_order: list[str] = []

        def recording_tracker(**options):
            tracker_options.append(options)
            return RealTargetTracker(**options)

        class RecordingMakcuController:
            instances: list["RecordingMakcuController"] = []

            def __init__(self, config, *, calibrated_controller=None) -> None:
                self.config = config
                self.calibrated_controller = calibrated_controller
                self.activation_pressed = True
                self.started = False
                self.stopped = False
                self.updates: list[
                    tuple[
                        Detection | None,
                        tuple[int, int, int],
                        bool,
                        dict[str, object],
                    ]
                ] = []
                self.__class__.instances.append(self)

            def start(self) -> None:
                self.started = True

            def update(
                self,
                target,
                frame_shape,
                *,
                active=True,
                **kwargs,
            ) -> None:
                if self.stopped:
                    raise AssertionError("aim controller stopped before frame update")
                self.updates.append(
                    (target, frame_shape, active, dict(kwargs))
                )

            def telemetry_snapshot(self) -> MakcuTelemetrySnapshot:
                return MakcuTelemetrySnapshot()

            def stop(self) -> None:
                self.stopped = True
                cleanup_order.append("aim")

        direct_sample = SimpleNamespace(
            point=(24.0, 5.0),
            source_timestamp_ns=source.base_ns,
            confidence=0.9,
            evidence="direct head box",
            bridging=False,
            corroboration_point=(24.0, 12.0),
        )
        head_runtime = mock.Mock()
        head_runtime.status = SimpleNamespace()
        head_runtime.identity_generation = 1
        head_runtime.accept_body.return_value = False
        head_runtime.take_latest.side_effect = [direct_sample, None]
        head_runtime.visible_sample.return_value = direct_sample
        head_runtime.stop.side_effect = lambda: cleanup_order.append("head") or True

        config = self._config(
            report_path,
            "--max-frames",
            "2",
            "--aim",
            "--aim-label",
            "player",
            "--ignore-self",
            "--aim-output",
            "makcu",
            "--aim-makcu-port",
            "/dev/serial/by-id/test-makcu",
            "--aim-makcu-vertical-rate-ratio",
            "0.63",
        )
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("aiming.MakcuAimingController", RecordingMakcuController),
            mock.patch("aiming.TargetTracker", side_effect=recording_tracker),
            mock.patch(
                "main._build_automatic_head_runtime",
                return_value=head_runtime,
            ),
        ):
            result = run(config)

        self.assertEqual(result, 0)
        controller = RecordingMakcuController.instances[0]
        self.assertTrue(controller.started)
        self.assertTrue(controller.stopped)
        self.assertEqual(controller.config.vertical_rate_ratio, 1.0)
        self.assertEqual(len(tracker_options), 1)
        self.assertEqual(tracker_options[0]["lost_grace_frames"], 3)
        numeric = controller.calibrated_controller
        self.assertIsNotNone(numeric)
        self.assertEqual(
            numeric.config.maximum_rate_x_counts_per_second,
            numeric.config.maximum_rate_y_counts_per_second,
        )
        self.assertEqual(numeric.config.velocity_median_window, 5)
        self.assertEqual(
            numeric.config.velocity_filter_time_constant_seconds,
            0.018,
        )
        self.assertEqual(
            numeric.config.maximum_target_acceleration_pixels_per_second_squared,
            20_000.0,
        )
        self.assertEqual(numeric.config.stale_after_seconds, 0.065)
        self.assertEqual(
            numeric.config.maximum_observation_interval_seconds,
            0.040,
        )
        self.assertEqual(numeric.config.position_time_constant_seconds, 0.022)
        self.assertEqual(numeric.config.feedback_deadzone_pixels, 3.0)
        self.assertEqual(numeric.config.maximum_velocity_feedforward_fraction, 0.95)
        self.assertTrue(
            numeric.config.require_motion_corroboration_for_feedforward
        )
        self.assertEqual(len(controller.updates), 2)
        first_target, _shape, _active, first_keywords = controller.updates[0]
        self.assertIsNone(first_target)
        self.assertNotIn("aim_point", first_keywords)
        direct_target, _shape, _active, direct_keywords = controller.updates[1]
        self.assertIs(direct_target, raw_target)
        self.assertEqual(direct_keywords["aim_point"], direct_sample.point)
        self.assertEqual(
            direct_keywords["measurement_ns"],
            direct_sample.source_timestamp_ns,
        )
        self.assertNotIn("velocity_target", direct_keywords)
        self.assertTrue(direct_keywords["measurement_observed"])
        self.assertEqual(
            direct_keywords["motion_corroboration_point"],
            direct_sample.corroboration_point,
        )
        head_runtime.submit.assert_called_once()
        self.assertEqual(cleanup_order, ["aim", "head"])
        startup = output.getvalue()
        self.assertIn("control automatic command-aware observer", startup)
        self.assertIn("head source pinned SunXDS 0.8.0 direct boxes", startup)
        self.assertIn(
            "direct-head prediction gated by same-frame player motion",
            startup,
        )
        self.assertIn("latest-only 90 Hz", startup)

    def test_capture_starvation_surfaces_automatic_head_worker_failure(self) -> None:
        report_path = self.root / "head-worker-starvation.json"
        source = _FakeSource(static=True)

        class RecordingMakcuController:
            instances: list["RecordingMakcuController"] = []

            def __init__(self, config, *, calibrated_controller=None) -> None:
                self.config = config
                self.calibrated_controller = calibrated_controller
                self.activation_pressed = False
                self.started = False
                self.stopped = False
                self.__class__.instances.append(self)

            def start(self) -> None:
                self.started = True

            def stop(self) -> None:
                self.stopped = True

            def telemetry_snapshot(self) -> MakcuTelemetrySnapshot:
                return MakcuTelemetrySnapshot()

        head_runtime = mock.Mock()
        head_runtime.status = SimpleNamespace()
        head_runtime.stop.return_value = True
        head_runtime.raise_if_failed.side_effect = RuntimeError(
            "synthetic starved head failure"
        )
        config = self._config(
            report_path,
            "--max-seconds",
            "0.1",
            "--aim",
            "--aim-label",
            "player",
            "--ignore-self",
            "--aim-output",
            "makcu",
            "--aim-makcu-port",
            "/dev/serial/by-id/test-makcu",
        )

        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("aiming.MakcuAimingController", RecordingMakcuController),
            mock.patch(
                "main._build_automatic_head_runtime",
                return_value=head_runtime,
            ),
            self.assertRaisesRegex(RuntimeError, "synthetic starved head failure"),
        ):
            run(config)

        self.assertEqual(source.read_calls, 1)
        # One call comes from the packet=None branch and the second from the
        # bounded cleanup audit.  Without starvation polling this would be one.
        self.assertEqual(head_runtime.raise_if_failed.call_count, 2)
        head_runtime.stop.assert_called_once_with()
        controller = RecordingMakcuController.instances[0]
        self.assertTrue(controller.started)
        self.assertTrue(controller.stopped)

    def test_detail_pass_runs_same_model_twice_and_reports_actual_geometry(self) -> None:
        report_path = self.root / "detail.json"
        source = _FakeSource(shape=(72, 128, 3))
        _FakeDetector.inference_delay_seconds = 0.001

        result = self._run_with(
            source,
            self._config(
                report_path,
                "--detail-crop-size",
                "64",
                "--max-frames",
                "2",
            ),
        )

        self.assertEqual(result, 0)
        detector = _FakeDetector.instances[0]
        self.assertEqual(detector.infer_calls, 4)
        self.assertEqual(detector.postprocess_calls, 4)
        self.assertEqual(detector.tensor_shapes, [(1, 3, 32, 32)] * 4)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        detail = report["detail_pass"]
        self.assertTrue(detail["enabled"])
        self.assertEqual(detail["requested_crop_size"], 64)
        self.assertEqual(detail["duplicate_iou_threshold"], 0.5)
        self.assertEqual(detail["unmatched_detail_reference_height"], 1080.0)
        self.assertEqual(detail["unmatched_detail_max_reference_height"], 96.0)
        self.assertEqual(detail["frames_applied"], 2)
        self.assertEqual(detail["crop_policy"], "centered_model_aspect_roi")
        self.assertEqual(detail["last_plan"]["applied_crop_width"], 64)
        self.assertEqual(detail["last_plan"]["applied_crop_height"], 64)
        self.assertEqual(detail["last_plan"]["source_width"], 128)
        self.assertEqual(detail["last_plan"]["source_height"], 72)
        self.assertEqual(detail["last_plan"]["effective_linear_magnification"], 2.0)
        timings = report["pipeline"]["timings"]["mean"]
        self.assertGreater(timings["detail_preprocess_ms"], 0.0)
        self.assertGreater(timings["detail_inference_ms"], 0.0)
        self.assertGreaterEqual(timings["detail_postprocess_ms"], 0.0)

    def test_detail_pass_failure_fails_closed_before_report_publication(self) -> None:
        report_path = self.root / "detail-failure.json"
        source = _FakeSource(shape=(72, 128, 3))
        _FakeDetector.fail_infer_call = 2

        with self.assertRaisesRegex(RuntimeError, "detail inference failure"):
            self._run_with(
                source,
                self._config(
                    report_path,
                    "--detail-crop-size",
                    "64",
                    "--max-frames",
                    "1",
                ),
            )

        self.assertTrue(source.closed)
        self.assertFalse(report_path.exists())

    def test_capture_close_timeout_fails_before_report_publication(self) -> None:
        report_path = self.root / "capture-close-timeout.json"
        source = _CloseTimeoutSource()

        with self.assertRaisesRegex(RuntimeError, "capture shutdown.*Timed out"):
            self._run_with(
                source,
                self._config(report_path, "--max-frames", "1"),
            )

        self.assertTrue(source.closed)
        self.assertFalse(report_path.exists())

    def test_preview_close_timeout_fails_before_report_publication(self) -> None:
        report_path = self.root / "preview-close-timeout.json"
        source = _FakeSource()
        config = replace(
            self._config(report_path, "--max-frames", "1"),
            preview=True,
        )

        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("utils.preview.create_preview_window", return_value=_StuckPreview()),
            self.assertRaisesRegex(RuntimeError, "preview shutdown.*bounded timeout"),
        ):
            run(config)

        self.assertTrue(source.closed)
        self.assertFalse(report_path.exists())

    def test_model_mutation_fails_before_report_publication(self) -> None:
        report_path = self.root / "mutated-model.json"
        source = _ArtifactMutatingSource(self.model)

        with self.assertRaisesRegex(RuntimeError, "Model artifact changed"):
            self._run_with(
                source,
                self._config(report_path, "--max-frames", "1"),
            )

        self.assertTrue(source.closed)
        self.assertFalse(report_path.exists())

    def test_redundant_square_detail_pass_is_skipped(self) -> None:
        report_path = self.root / "detail-redundant.json"
        source = _FakeSource(shape=(32, 32, 3))

        result = self._run_with(
            source,
            self._config(
                report_path,
                "--detail-crop-size",
                "64",
                "--max-frames",
                "1",
            ),
        )

        self.assertEqual(result, 0)
        detector = _FakeDetector.instances[0]
        self.assertEqual(detector.infer_calls, 1)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["detail_pass"]["frames_redundant"], 1)
        self.assertEqual(report["detail_pass"]["frames_applied"], 0)

    def test_aim_continuation_is_safety_filtered_but_not_a_display_detection(self) -> None:
        report_path = self.root / "detail-aim.json"
        source = _FakeSource(shape=(72, 128, 3))
        primary = Detection(0, "player", 0.60, (40, 15, 70, 60))
        detail = Detection(0, "player", 0.90, (41, 16, 71, 61))
        low_continuation = Detection(0, "player", 0.18, (90, 10, 110, 60))
        guarded_low = Detection(0, "player", 0.17, (40, 35, 60, 72))
        _FakeDetector.detection_batches = [
            [primary, low_continuation, guarded_low],
            [detail],
        ]

        class FakeTracker:
            instances: list["FakeTracker"] = []

            def __init__(self, **options) -> None:
                self.options = options
                self.updates: list[list[Detection]] = []
                self.continuation_updates: list[list[Detection]] = []
                self.__class__.instances.append(self)

            def update(
                self,
                detections,
                _frame_shape,
                *,
                continuation_detections=(),
                **_kwargs,
            ):
                copied = list(detections)
                self.updates.append(copied)
                self.continuation_updates.append(list(continuation_detections))
                return copied[0] if copied else None

            def reset(self) -> None:
                return None

        class FakeController:
            instances: list["FakeController"] = []

            def __init__(self, _config) -> None:
                self.updates: list[Detection | None] = []
                self.__class__.instances.append(self)

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def update(self, target, _frame_shape, *, active=True) -> None:
                self.updates.append(target if active else None)

        class FakeSensor:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def read(self) -> bool:
                return True

        class FakeSelfFilter:
            instances: list["FakeSelfFilter"] = []

            def __init__(self, _zone) -> None:
                self.calls: list[list[Detection]] = []
                self.__class__.instances.append(self)

            def apply(self, detections, _frame_shape):
                copied = list(detections)
                self.calls.append(copied)
                return SimpleNamespace(
                    detections=copied,
                    ignored_count=0,
                    ignored_detection=None,
                    aim_safe=True,
                )

        config = self._config(
            report_path,
            "--detail-crop-size",
            "64",
            "--max-frames",
            "1",
            "--aim",
            "--aim-label",
            "player",
            "--ignore-self",
            "--aim-output",
            "local",
            "--aim-activate-path",
            "/synthetic/controller",
        )
        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("aiming.AimingController", FakeController),
            mock.patch("aiming.AimActivationSensor", FakeSensor),
            mock.patch("aiming.TargetTracker", FakeTracker),
            mock.patch("utils.self_filter.SelfAvatarFilter", FakeSelfFilter),
        ):
            result = run(config)

        self.assertEqual(result, 0)
        self.assertEqual(
            FakeSelfFilter.instances[0].calls,
            [[detail, low_continuation, guarded_low]],
        )
        self.assertEqual(FakeTracker.instances[0].options["lost_grace_frames"], 1)
        self.assertEqual(FakeTracker.instances[0].updates, [[detail]])
        self.assertEqual(
            FakeTracker.instances[0].continuation_updates,
            [[low_continuation]],
        )
        self.assertEqual(FakeController.instances[0].updates, [detail])
        detector = _FakeDetector.instances[0]
        self.assertEqual(detector.postprocess_confidences, [0.15, 0.15])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["detail_pass"]["primary_detections"], 1)
        self.assertEqual(report["detail_pass"]["detail_detections"], 1)

    def test_release_repress_requires_configured_confidence_again(self) -> None:
        report_path = self.root / "release-repress.json"
        source = _FakeSource(shape=(72, 128, 3))
        strong = Detection(0, "player", 0.90, (80, 15, 100, 65))
        weak = Detection(0, "player", 0.18, (81, 15, 101, 65))
        _FakeDetector.detection_batches = [[strong], [weak], [weak]]

        class FakeController:
            instances: list["FakeController"] = []

            def __init__(self, _config) -> None:
                self.updates: list[tuple[Detection | None, bool]] = []
                self.__class__.instances.append(self)

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def update(self, target, _frame_shape, *, active=True) -> None:
                self.updates.append((target, active))

        class SequencedSensor:
            def __init__(self, *_args, **_kwargs) -> None:
                self.states = iter((True, False, True))

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def read(self) -> bool:
                return next(self.states)

        class AlwaysSafeSelfFilter:
            def __init__(self, _zone) -> None:
                pass

            def apply(self, detections, _frame_shape):
                copied = tuple(detections)
                return SimpleNamespace(
                    detections=copied,
                    ignored_count=0,
                    ignored_detection=None,
                    aim_safe=True,
                )

        config = self._config(
            report_path,
            "--max-frames",
            "3",
            "--aim",
            "--aim-label",
            "player",
            "--ignore-self",
            "--aim-output",
            "local",
            "--aim-activate-path",
            "/synthetic/controller",
        )
        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("aiming.AimingController", FakeController),
            mock.patch("aiming.AimActivationSensor", SequencedSensor),
            mock.patch("utils.self_filter.SelfAvatarFilter", AlwaysSafeSelfFilter),
        ):
            result = run(config)

        self.assertEqual(result, 0)
        updates = FakeController.instances[0].updates
        self.assertEqual(updates[0], (strong, True))
        self.assertEqual(updates[1], (None, False))
        self.assertEqual(updates[2], (None, True))

    def test_low_continuation_boxes_never_reach_detection_drawing(self) -> None:
        report_path = self.root / "draw-confidence.json"
        source = _FakeSource(shape=(72, 128, 3))
        strong = Detection(0, "player", 0.90, (80, 15, 100, 65))
        weak = Detection(0, "player", 0.18, (100, 15, 120, 65))
        _FakeDetector.detection_batches = [[strong, weak]]
        drawn: list[tuple[Detection, ...]] = []

        class FakeController:
            def __init__(self, _config) -> None:
                pass

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def update(self, _target, _frame_shape, *, active=True) -> None:
                del active

        class ActiveSensor:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def read(self) -> bool:
                return True

        class AlwaysSafeSelfFilter:
            def __init__(self, _zone) -> None:
                pass

            def apply(self, detections, _frame_shape):
                copied = tuple(detections)
                return SimpleNamespace(
                    detections=copied,
                    ignored_count=0,
                    ignored_detection=None,
                    aim_safe=True,
                )

        class RecordingPreview:
            mode = "inline"
            stats = PreviewStats(1, 1, 0)

            def start(self) -> None:
                return None

            def poll(self) -> bool:
                return True

            def submit(self, _frame) -> bool:
                return True

            def should_continue(self) -> bool:
                return True

            def stop(self) -> bool:
                return True

            def raise_if_failed(self) -> None:
                return None

        config = replace(
            self._config(
                report_path,
                "--max-frames",
                "1",
                "--aim",
                "--aim-label",
                "player",
                "--ignore-self",
                "--aim-output",
                "local",
                "--aim-activate-path",
                "/synthetic/controller",
            ),
            preview=True,
            draw=True,
        )
        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("aiming.AimingController", FakeController),
            mock.patch("aiming.AimActivationSensor", ActiveSensor),
            mock.patch("utils.self_filter.SelfAvatarFilter", AlwaysSafeSelfFilter),
            mock.patch(
                "utils.preview.create_preview_window",
                return_value=RecordingPreview(),
            ),
            mock.patch(
                "utils.render.draw_detections",
                side_effect=lambda _frame, detections: drawn.append(tuple(detections)),
            ),
        ):
            result = run(config)

        self.assertEqual(result, 0)
        self.assertEqual(drawn, [(strong,)])

    def test_hard_self_guard_empty_sample_cannot_use_physical_prediction_grace(
        self,
    ) -> None:
        report_path = self.root / "hard-guard-aim.json"

        class TimestampedSource(_FakeSource):
            def __init__(self) -> None:
                super().__init__(shape=(72, 128, 3))
                self.base_ns = perf_counter_ns() - 20_000_000

            def read(self, timeout: float | None = None):
                self.read_calls += 1
                self.read_timeouts.append(timeout)
                started_ns = self.base_ns + (self.read_calls - 1) * 8_000_000
                return FramePacket(
                    image=np.zeros(self.shape, dtype=np.uint8),
                    sequence=self.read_calls - 1,
                    read_started_ns=started_ns,
                    read_completed_ns=started_ns + 100_000,
                )

        class FakeController:
            instances: list["FakeController"] = []

            def __init__(self, _config) -> None:
                self.updates: list[Detection | None] = []
                self.__class__.instances.append(self)

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def update(self, target, _frame_shape, *, active=True) -> None:
                self.updates.append(target if active else None)

        class FakeSensor:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def read(self) -> bool:
                return True

        class AlwaysSafeSelfFilter:
            def __init__(self, _zone) -> None:
                pass

            def apply(self, detections, _frame_shape):
                copied = tuple(detections)
                return SimpleNamespace(
                    detections=copied,
                    ignored_count=0,
                    ignored_detection=None,
                    aim_safe=True,
                )

        # The first player is outside the bottom-center guard and establishes
        # a physical target. A guarded non-target player-like label at +8 ms
        # must leave genuine empty-target prediction grace intact. An exact
        # target-label self candidate at +16 ms must revoke that grace.
        opponent = Detection(0, "player", 0.9, (80, 20, 100, 65))
        self_candidate = Detection(0, "player", 0.9, (40, 35, 60, 72))
        guarded_non_target = Detection(1, "person", 0.9, (40, 35, 60, 72))
        _FakeDetector.detection_batches = [
            [opponent],
            [guarded_non_target],
            [self_candidate],
        ]
        source = TimestampedSource()
        config = self._config(
            report_path,
            "--max-frames",
            "3",
            "--aim",
            "--aim-label",
            "player",
            "--ignore-self",
            "--aim-output",
            "local",
            "--aim-activate-path",
            "/synthetic/controller",
        )

        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("aiming.AimingController", FakeController),
            mock.patch("aiming.AimActivationSensor", FakeSensor),
            mock.patch("utils.self_filter.SelfAvatarFilter", AlwaysSafeSelfFilter),
        ):
            result = run(config)

        self.assertEqual(result, 0)
        updates = FakeController.instances[0].updates
        self.assertEqual(len(updates), 3)
        self.assertEqual(updates[0], opponent)
        self.assertIsNotNone(updates[1])
        self.assertEqual(updates[1].class_name, "player")
        self.assertIsNone(updates[2])

    def test_max_seconds_stops_a_static_source_on_the_bounded_read(self) -> None:
        report_path = self.root / "static.json"
        source = _FakeSource(static=True)
        started = perf_counter_ns()

        result = self._run_with(
            source,
            self._config(report_path, "--max-seconds", "0.03"),
        )
        elapsed = (perf_counter_ns() - started) / 1e9

        self.assertEqual(result, 0)
        self.assertLess(elapsed, 0.5)
        self.assertTrue(source.read_timeouts)
        self.assertLessEqual(max(float(value or 0.0) for value in source.read_timeouts), 0.25)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["termination"]["reason"], "max_seconds")
        self.assertEqual(report["pipeline"]["processed_frames"], 0)
        self.assertGreaterEqual(report["pipeline"]["elapsed_seconds"], 0.02)

    def test_existing_report_is_rejected_before_detector_or_capture_start(self) -> None:
        report_path = self.root / "already-there.json"
        report_path.write_text("keep", encoding="utf-8")
        source = _FakeSource()

        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self._run_with(
                source,
                self._config(report_path, "--max-frames", "1"),
            )

        self.assertFalse(source.started)
        self.assertFalse(_FakeDetector.instances)
        self.assertEqual(report_path.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
