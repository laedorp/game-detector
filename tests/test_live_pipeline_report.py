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

from aiming.direct_head_anchor import DirectHeadProvenance
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
        self.transforms: list[object] = []
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
        del frame_shape
        self.transforms.append(transform)
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

    def _run_automatic_detail_case(
        self,
        name: str,
        detection_batches: list[list[Detection]],
        *,
        activation_pressed: bool = True,
        raw_activation_state: tuple[bool, bool] | None = None,
        activation_requires_release: bool = False,
        live_measured_anchor: bool = False,
        suspend_body_gap: bool = False,
        max_frames: int = 1,
        extra_arguments: tuple[str, ...] = (),
    ) -> tuple[_FakeDetector, dict[str, object], str, mock.Mock]:
        report_path = self.root / f"automatic-detail-{name}.json"
        source = _FakeSource(shape=(1080, 1920, 3))
        _FakeDetector.detection_batches = detection_batches

        class RecordingMakcuController:
            def __init__(self, config, *, calibrated_controller=None) -> None:
                self.config = config
                self.calibrated_controller = calibrated_controller
                self.activation_pressed = activation_pressed
                self.raw_activation_state = raw_activation_state or (
                    True,
                    activation_pressed,
                )
                self.activation_requires_release = activation_requires_release
                self.updates: list[tuple[Detection | None, dict[str, object]]] = []

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def update(self, target, _frame_shape, **keywords) -> None:
                self.updates.append((target, dict(keywords)))

            def revoke_motion_corroboration(self) -> None:
                return None

            def telemetry_snapshot(self) -> MakcuTelemetrySnapshot:
                return MakcuTelemetrySnapshot()

        head_runtime = mock.Mock()
        head_runtime.provider = "MIGraphXExecutionProvider"
        head_runtime.status = SimpleNamespace()
        head_runtime.identity_generation = 0
        head_runtime.accept_body.return_value = False
        head_runtime.consume_motion_corroboration_revocation.return_value = False
        head_runtime.take_latest.return_value = None
        head_runtime.visible_sample.return_value = None
        head_runtime.has_live_measured_anchor.return_value = live_measured_anchor
        head_runtime.suspend_body_gap.return_value = suspend_body_gap
        head_runtime.revoke_body.return_value = False
        head_runtime.stop.return_value = True
        config = self._config(
            report_path,
            "--backend",
            "onnxruntime",
            "--device",
            "MIGRAPHX",
            "--require-full-provider",
            "--max-frames",
            str(max_frames),
            "--aim",
            "--aim-label",
            "player",
            "--ignore-self",
            "--aim-output",
            "makcu",
            "--aim-makcu-port",
            "/dev/serial/by-id/test-makcu",
            "--aim-makcu-tracking-mode",
            "direct-head",
            *extra_arguments,
        )
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("detection.onnx_yolo.OnnxRuntimeYoloDetector", _FakeDetector),
            mock.patch("aiming.MakcuAimingController", RecordingMakcuController),
            mock.patch(
                "main._build_automatic_head_runtime",
                return_value=head_runtime,
            ),
        ):
            self.assertEqual(run(config), 0)
        detector = _FakeDetector.instances[-1]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return detector, report, output.getvalue(), head_runtime

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

    def test_automatic_detail_rescues_a_missing_held_target(self) -> None:
        rescued = Detection(0, "player", 0.88, (930, 500, 990, 580))

        detector, report, startup, _head_runtime = self._run_automatic_detail_case(
            "no-target",
            [[], [rescued]],
        )

        self.assertEqual(detector.infer_calls, 2)
        detail = report["detail_pass"]
        self.assertEqual(detail["mode"], "automatic_activation_need_gated")
        self.assertIsNone(detail["configured_crop_size"])
        self.assertEqual(detail["effective_crop_size"], 640)
        self.assertEqual(detail["frames_applied"], 1)
        self.assertEqual(detail["unmatched_detail_accepted"], 1)
        self.assertEqual(
            (detector.transforms[1].crop_x, detector.transforms[1].crop_y),
            (640, 220),
        )
        self.assertEqual(
            detail["last_plan"]["crop_policy"],
            "centered_model_aspect_roi",
        )
        self.assertEqual(
            detail["automatic_frames_triggered_no_exact_target"],
            1,
        )
        self.assertIn("activation/need gated", startup)
        self.assertIn("released preview skip", startup)

    def test_missing_primary_detail_follows_recent_accepted_target(self) -> None:
        # This first exact target is outside the ordinary centered 640 ROI, so
        # its full-primary acceptance is the only safe source for the next
        # frame's crop location. The detail-only rescue cannot renew the hint.
        primary = Detection(0, "player", 0.88, (300, 500, 400, 580))
        rescued = Detection(0, "player", 0.90, (302, 500, 402, 580))

        detector, report, _startup, _head_runtime = (
            self._run_automatic_detail_case(
                "recent-target-roi",
                [[primary], [], [rescued]],
                max_frames=2,
            )
        )

        self.assertEqual(detector.infer_calls, 3)
        self.assertEqual(len(detector.transforms), 3)
        detail_transform = detector.transforms[2]
        self.assertEqual(
            (detail_transform.crop_x, detail_transform.crop_y),
            (30, 220),
        )
        detail = report["detail_pass"]
        self.assertEqual(detail["frames_applied"], 1)
        self.assertEqual(
            detail["last_plan"]["crop_policy"],
            "target_centered_model_aspect_roi",
        )
        self.assertEqual(
            (detail["last_plan"]["crop_x"], detail["last_plan"]["crop_y"]),
            (30, 220),
        )

    def test_stable_body_mode_skips_head_worker_and_detail_inference(self) -> None:
        target = Detection(0, "player", 0.88, (830, 260, 1030, 860))
        diagnostic_root = self.root / "stable-body-diagnostics"

        detector, report, startup, head_runtime = self._run_automatic_detail_case(
            "stable-body",
            [[target]],
            extra_arguments=(
                "--aim-makcu-tracking-mode",
                "stable-body",
                "--aim-diagnostic-dir",
                str(diagnostic_root),
            ),
        )

        self.assertEqual(detector.infer_calls, 1)
        self.assertEqual(report["detail_pass"]["mode"], "disabled")
        self.assertFalse(report["detail_pass"]["enabled"])
        head_runtime.start.assert_not_called()
        self.assertIn("stable measured-body tracking", startup)
        sessions = list(diagnostic_root.iterdir())
        self.assertEqual(len(sessions), 1)
        manifest = json.loads(
            (sessions[0] / "manifest.json").read_text(encoding="utf-8")
        )
        metadata = manifest["metadata"]
        self.assertIsNone(metadata["plant_calibrated_delay_seconds"])
        self.assertEqual(metadata["plant_effective_delay_seconds"], 0.006)
        self.assertIsNone(metadata["plant_delay_upper_seconds"])
        self.assertEqual(metadata["plant_delay_seconds"], 0.006)
        replay = json.loads(
            (sessions[0] / "replay-report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(replay["status"], "insufficient-data")
        self.assertEqual(replay["records"], 1)

    def test_expired_continuous_hold_reports_latch_and_release_instruction(self) -> None:
        diagnostic_root = self.root / "expired-hold-diagnostics"

        _detector, _report, _startup, _head_runtime = (
            self._run_automatic_detail_case(
                "expired-hold",
                [[]],
                activation_pressed=False,
                raw_activation_state=(True, True),
                activation_requires_release=True,
                extra_arguments=(
                    "--aim-diagnostic-dir",
                    str(diagnostic_root),
                ),
            )
        )

        sessions = list(diagnostic_root.iterdir())
        self.assertEqual(len(sessions), 1)
        records = [
            json.loads(line)
            for line in (sessions[0] / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertFalse(record["activation_pressed"])
        self.assertTrue(record["raw_activation_known"])
        self.assertTrue(record["raw_activation_pressed"])
        self.assertTrue(record["activation_requires_release"])
        self.assertEqual(
            record["activation_denial_reason"],
            "continuous-hold-expired",
        )
        self.assertEqual(
            record["aim_status"],
            "aim paused: continuous hold safety limit reached; "
            "release Right, then press it again",
        )

    def test_visible_full_pass_self_cannot_suppress_opponent_rescue(self) -> None:
        visible_self = Detection(
            0,
            "player",
            0.92,
            (850.0, 650.0, 1070.0, 1080.0),
        )
        distinct_small_opponent = Detection(
            0,
            "player",
            0.88,
            (1100.0, 500.0, 1160.0, 580.0),
        )

        batches: list[list[Detection]] = []
        for _ in range(3):
            batches.extend(([visible_self], [distinct_small_opponent]))

        detector, report, startup, head_runtime = (
            self._run_automatic_detail_case(
                "self-does-not-suppress",
                batches,
                max_frames=3,
            )
        )

        self.assertEqual(detector.infer_calls, 6)
        detail = report["detail_pass"]
        self.assertEqual(
            detail["automatic_frames_triggered_no_exact_target"],
            3,
        )
        self.assertEqual(detail["unmatched_detail_accepted"], 3)
        self.assertTrue(detail["automatic_self_guarded_need"])
        self.assertTrue(
            detail["automatic_self_relative_detail_exclusion_enabled"]
        )
        self.assertTrue(
            detail["automatic_lower_edge_self_fragment_exclusion_enabled"]
        )
        self.assertIn("self-guarded full-pass need decision", startup)
        self.assertIn(
            "lower-ROI-edge detail-only self fragments within 4 model px",
            startup,
        )
        head_runtime.accept_body.assert_called()
        accepted_box = head_runtime.accept_body.call_args.args[0]
        self.assertEqual(accepted_box, distinct_small_opponent.box)

    def test_detail_fragment_of_guarded_self_never_enters_aim_authority(
        self,
    ) -> None:
        weak_full_self = Detection(
            0,
            "player",
            0.24,
            (850.0, 650.0, 1070.0, 1080.0),
        )
        strong_upper_self_fragment = Detection(
            0,
            "player",
            0.88,
            (850.0, 850.0, 910.0, 920.0),
        )
        batches: list[list[Detection]] = []
        for _ in range(3):
            batches.extend(([weak_full_self], [strong_upper_self_fragment]))
        # Once confirmed, the weak parent can disappear for a frame without
        # letting its detail-only child inherit aim authority.
        batches.extend(([], [strong_upper_self_fragment]))

        detector, report, _startup, head_runtime = (
            self._run_automatic_detail_case(
                "guarded-self-fragment",
                batches,
                max_frames=4,
            )
        )

        self.assertEqual(detector.infer_calls, 8)
        detail = report["detail_pass"]
        self.assertEqual(
            detail["automatic_frames_triggered_no_exact_target"],
            4,
        )
        self.assertEqual(detail["unmatched_detail_accepted"], 0)
        head_runtime.accept_body.assert_not_called()
        head_runtime.submit.assert_not_called()

    def test_cold_start_detail_only_self_fragment_never_enters_aim_authority(
        self,
    ) -> None:
        strong_upper_self_fragment = Detection(
            0,
            "player",
            0.88,
            (850.0, 850.0, 910.0, 920.0),
        )

        detector, report, startup, head_runtime = (
            self._run_automatic_detail_case(
                "cold-start-self-fragment",
                [[], [strong_upper_self_fragment]],
            )
        )

        self.assertEqual(detector.infer_calls, 2)
        detail = report["detail_pass"]
        self.assertEqual(
            detail["automatic_frames_triggered_no_exact_target"],
            1,
        )
        self.assertEqual(detail["unmatched_detail_accepted"], 0)
        self.assertTrue(
            detail["automatic_lower_edge_self_fragment_exclusion_enabled"]
        )
        self.assertEqual(
            detail["automatic_lower_edge_self_fragment_margin_model_pixels"],
            4.0,
        )
        self.assertIn(
            "lower-ROI-edge detail-only self fragments within 4 model px",
            startup,
        )
        head_runtime.accept_body.assert_not_called()
        head_runtime.submit.assert_not_called()

    def test_automatic_detail_refreshes_a_small_central_target(self) -> None:
        primary = Detection(0, "player", 0.60, (930, 500, 990, 580))
        refined = Detection(0, "player", 0.91, (931, 501, 991, 581))

        detector, report, _startup, _head_runtime = self._run_automatic_detail_case(
            "small-central",
            [[primary], [refined]],
        )

        self.assertEqual(detector.infer_calls, 2)
        detail = report["detail_pass"]
        self.assertEqual(
            detail["automatic_frames_triggered_small_central_target"],
            1,
        )
        self.assertEqual(detail["cross_pass_matches"], 1)
        self.assertEqual(detail["detail_replacements"], 1)

    def test_live_verified_anchor_skips_redundant_small_target_detail_pass(
        self,
    ) -> None:
        primary = Detection(0, "player", 0.60, (930, 500, 990, 580))

        detector, report, startup, _head_runtime = (
            self._run_automatic_detail_case(
                "verified-anchor-carry",
                [[primary]],
                live_measured_anchor=True,
            )
        )

        self.assertEqual(detector.infer_calls, 1)
        detail = report["detail_pass"]
        self.assertEqual(detail["automatic_frames_triggered"], 0)
        self.assertEqual(
            detail["automatic_frames_skipped_verified_anchor"],
            1,
        )
        self.assertIn("live verified head", startup)

    def test_live_anchor_still_runs_detail_when_full_pass_target_is_missing(
        self,
    ) -> None:
        rescued = Detection(0, "player", 0.88, (930, 500, 990, 580))

        detector, report, _startup, _head_runtime = (
            self._run_automatic_detail_case(
                "verified-anchor-missing-primary",
                [[], [rescued]],
                live_measured_anchor=True,
            )
        )

        self.assertEqual(detector.infer_calls, 2)
        self.assertEqual(
            report["detail_pass"][
                "automatic_frames_triggered_no_exact_target"
            ],
            1,
        )

    def test_automatic_detail_skips_close_and_off_center_targets(self) -> None:
        cases = {
            "close": Detection(0, "player", 0.90, (900, 440, 1020, 640)),
            "off-center": Detection(0, "player", 0.90, (420, 500, 500, 580)),
        }
        for name, primary in cases.items():
            with self.subTest(name=name):
                detector, report, _startup, _head_runtime = (
                    self._run_automatic_detail_case(
                        name,
                        [[primary]],
                    )
                )

                self.assertEqual(detector.infer_calls, 1)
                detail = report["detail_pass"]
                self.assertEqual(detail["frames_applied"], 0)
                self.assertEqual(detail["automatic_frames_triggered"], 0)
                self.assertEqual(detail["automatic_frames_skipped_not_needed"], 1)

    def test_automatic_detail_uses_direct_acquisition_threshold_not_ui_threshold(
        self,
    ) -> None:
        # Direct-head body evidence is deliberately admitted at 0.15 for head
        # verification.  A high UI display/aim threshold must not turn the
        # conditional detail rescue into an unconditional second inference on
        # every held frame when a large, usable primary target already exists.
        primary = Detection(0, "player", 0.40, (900, 440, 1020, 640))

        detector, report, _startup, _head_runtime = (
            self._run_automatic_detail_case(
                "acquisition-threshold",
                [[primary]],
                extra_arguments=("--confidence", "0.95"),
            )
        )

        self.assertEqual(detector.infer_calls, 1)
        detail = report["detail_pass"]
        self.assertEqual(detail["automatic_need_confidence"], 0.15)
        self.assertEqual(detail["automatic_frames_triggered"], 0)
        self.assertEqual(detail["automatic_frames_skipped_not_needed"], 1)

    def test_automatic_detail_skips_released_preview(self) -> None:
        detector, report, _startup, _head_runtime = self._run_automatic_detail_case(
            "released",
            [[]],
            activation_pressed=False,
        )

        self.assertEqual(detector.infer_calls, 1)
        detail = report["detail_pass"]
        self.assertEqual(detail["automatic_frames_triggered"], 0)
        self.assertEqual(detail["automatic_frames_activation_released"], 1)
        self.assertEqual(detail["frames_applied"], 0)

    def test_explicit_primary_crop_and_detail_settings_take_precedence(self) -> None:
        rescued = Detection(0, "player", 0.88, (930, 500, 990, 580))
        detector, report, startup, _head_runtime = self._run_automatic_detail_case(
            "explicit-detail",
            [[], [rescued]],
            activation_pressed=False,
            extra_arguments=("--detail-crop-size", "768"),
        )
        self.assertEqual(detector.infer_calls, 2)
        detail = report["detail_pass"]
        self.assertEqual(detail["mode"], "explicit_always")
        self.assertEqual(detail["configured_crop_size"], 768)
        self.assertEqual(detail["automatic_frames_evaluated"], 0)
        self.assertIn("explicit always-on", startup)

        detector, report, startup, _head_runtime = self._run_automatic_detail_case(
            "explicit-primary-crop",
            [[]],
            extra_arguments=("--crop-size", "768"),
        )
        self.assertEqual(detector.infer_calls, 1)
        detail = report["detail_pass"]
        self.assertFalse(detail["enabled"])
        self.assertEqual(detail["mode"], "disabled")
        self.assertNotIn("Automatic MAKCU detail rescue", startup)

    def test_ordinary_target_gap_suspends_anchor_without_hard_revocation(
        self,
    ) -> None:
        opponent = Detection(0, "player", 0.90, (900, 438, 960, 558))
        # Early empty samples stay inside the automatic tracker's bounded
        # prediction bridge. The later ones pause output while its 350 ms
        # identity memory remains live; the returning exact box therefore
        # keeps generation 1.
        batches = [[opponent], *([[]] * 18), [opponent]]
        _FakeDetector.inference_delay_seconds = 0.01
        try:
            detector, _report, _startup, head_runtime = (
                self._run_automatic_detail_case(
                    "same-identity-primary-gap",
                    batches,
                    max_frames=len(batches),
                    # Disable the independent detail rescue so one scripted
                    # batch corresponds to exactly one primary detector frame.
                    extra_arguments=("--crop-size", "768"),
                    suspend_body_gap=True,
                )
            )
        finally:
            _FakeDetector.inference_delay_seconds = 0.0

        self.assertEqual(detector.infer_calls, len(batches))
        self.assertGreaterEqual(
            head_runtime.suspend_body_gap.call_count,
            1,
        )
        for suspended_call in head_runtime.suspend_body_gap.call_args_list:
            self.assertEqual(suspended_call.args, ())
            self.assertIsInstance(suspended_call.kwargs.get("now_ns"), int)
            self.assertGreaterEqual(suspended_call.kwargs["now_ns"], 0)
        head_runtime.revoke_body.assert_not_called()
        accepted_calls = head_runtime.accept_body.call_args_list
        self.assertGreaterEqual(len(accepted_calls), 2)
        self.assertEqual(
            accepted_calls[0].kwargs["track_generation"],
            accepted_calls[-1].kwargs["track_generation"],
        )
        self.assertIsNotNone(
            accepted_calls[-1].kwargs["corroboration_box"]
        )

    def test_automatic_detail_does_not_leak_into_detector_only_mode(self) -> None:
        report_path = self.root / "automatic-detail-no-mode-leak.json"
        source = _FakeSource(shape=(1080, 1920, 3))

        self.assertEqual(
            self._run_with(
                source,
                self._config(report_path, "--max-frames", "1"),
            ),
            0,
        )

        detector = _FakeDetector.instances[0]
        self.assertEqual(detector.infer_calls, 1)
        detail = json.loads(report_path.read_text(encoding="utf-8"))["detail_pass"]
        self.assertFalse(detail["enabled"])
        self.assertEqual(detail["mode"], "disabled")
        self.assertIsNone(detail["effective_crop_size"])

    def test_local_aim_uses_longer_loss_grace_than_single_frame(self) -> None:
        from aiming.controller import TargetTracker as RealTargetTracker

        source = _FakeSource(shape=(1080, 1920, 3))
        tracker_options: list[dict[str, object]] = []

        def recording_tracker(**options):
            tracker_options.append(options)
            return RealTargetTracker(**options)

        class FakeController:
            def __init__(self, _config) -> None:
                pass

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def update(self, _target, _frame_shape, *, active=True) -> None:
                return None

        class FakeSensor:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def read(self) -> bool:
                return True

        config = self._config(
            self.root / "local-aim-grace.json",
            "--backend",
            "onnxruntime",
            "--device",
            "MIGRAPHX",
            "--require-full-provider",
            "--max-frames",
            "1",
            "--aim",
            "--aim-label",
            "player",
            "--ignore-self",
            "--aim-output",
            "local",
            "--aim-activate-path",
            "/tmp/test-aim-activation",
        )

        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("detection.onnx_yolo.OnnxRuntimeYoloDetector", _FakeDetector),
            mock.patch("aiming.AimingController", FakeController),
            mock.patch("aiming.AimActivationSensor", FakeSensor),
            mock.patch("aiming.TargetTracker", side_effect=recording_tracker),
        ):
            result = run(config)

        self.assertEqual(result, 0)
        self.assertEqual(len(tracker_options), 1)
        self.assertEqual(tracker_options[0]["lost_grace_frames"], 6)

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
        _FakeDetector.detection_batches = [[raw_target], [raw_target]]
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
                self.control_events: list[str] = []
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
                self.control_events.append("update")
                self.updates.append(
                    (target, frame_shape, active, dict(kwargs))
                )

            def revoke_motion_corroboration(self) -> None:
                self.control_events.append("revoke_motion_corroboration")

            def telemetry_snapshot(self) -> MakcuTelemetrySnapshot:
                return MakcuTelemetrySnapshot()

            def stop(self) -> None:
                self.stopped = True
                cleanup_order.append("aim")

        direct_sample = SimpleNamespace(
            point=(24.0, 5.0),
            velocity_point=(24.0, 5.0),
            source_timestamp_ns=source.base_ns,
            direct_source_timestamp_ns=source.base_ns,
            identity_deadline_ns=source.base_ns + 180_000_000,
            track_generation=1,
            provenance=DirectHeadProvenance.DIRECT,
            confidence=0.9,
            evidence="direct test head box",
            bridging=False,
            body_derived_motion_permitted=False,
            body_derived_motion_deadline_ns=None,
            corroboration_point=(16.0, 16.0),
        )
        measured_head_anchor = SimpleNamespace(
            point=(24.0, 5.0),
            velocity_point=(26.0, 6.0),
            source_timestamp_ns=source.base_ns,
            direct_source_timestamp_ns=source.base_ns,
            identity_deadline_ns=source.base_ns + 180_000_000,
            track_generation=1,
            provenance=DirectHeadProvenance.MEASURED_PRIMARY,
            confidence=0.9,
            evidence="filtered direct-head anchor",
            bridging=False,
            body_derived_motion_permitted=True,
            body_derived_motion_deadline_ns=source.base_ns + 180_000_000,
            corroboration_point=None,
        )
        continued_head_anchor = SimpleNamespace(
            point=(29.0, 9.0),
            velocity_point=(34.0, 11.0),
            source_timestamp_ns=source.base_ns + 8_000_000,
            direct_source_timestamp_ns=source.base_ns,
            identity_deadline_ns=source.base_ns + 180_000_000,
            track_generation=1,
            provenance=DirectHeadProvenance.MEASURED_PRIMARY,
            confidence=0.86,
            evidence="filtered direct-head anchor",
            bridging=False,
            body_derived_motion_permitted=True,
            body_derived_motion_deadline_ns=source.base_ns + 180_000_000,
            corroboration_point=None,
        )
        head_runtime = mock.Mock()
        head_runtime.provider = "MIGraphXExecutionProvider"
        head_runtime.status = SimpleNamespace()
        head_runtime.identity_generation = 1
        head_runtime.accept_body.return_value = False
        head_runtime.consume_motion_corroboration_revocation.side_effect = [
            False,
            False,
            False,
            False,
        ]
        head_runtime.take_latest.side_effect = [direct_sample, None]
        head_runtime.visible_sample.side_effect = [
            measured_head_anchor,
            continued_head_anchor,
        ]
        head_runtime.stop.side_effect = lambda: cleanup_order.append("head") or True

        config = self._config(
            report_path,
            "--backend",
            "onnxruntime",
            "--device",
            "MIGRAPHX",
            "--require-full-provider",
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
            "--aim-makcu-tracking-mode",
            "direct-head",
            "--aim-makcu-vertical-rate-ratio",
            "0.63",
        )
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("detection.onnx_yolo.OnnxRuntimeYoloDetector", _FakeDetector),
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
        self.assertEqual(tracker_options[0]["lost_grace_frames"], 8)
        numeric = controller.calibrated_controller
        self.assertIsNotNone(numeric)
        self.assertEqual(
            numeric.config.maximum_rate_x_counts_per_second,
            numeric.config.maximum_rate_y_counts_per_second,
        )
        self.assertEqual(numeric.config.velocity_median_window, 3)
        self.assertEqual(
            numeric.config.velocity_filter_time_constant_seconds,
            0.014,
        )
        self.assertEqual(
            numeric.config.maximum_target_acceleration_pixels_per_second_squared,
            40_000.0,
        )
        self.assertEqual(numeric.config.stale_after_seconds, 0.110)
        self.assertEqual(
            numeric.config.maximum_observation_interval_seconds,
            0.040,
        )
        self.assertEqual(numeric.config.position_time_constant_seconds, 0.028)
        self.assertEqual(numeric.config.feedback_deadzone_pixels, 4.5)
        self.assertTrue(numeric.config.continuous_feedback_deadband)
        self.assertEqual(
            numeric.config.continuous_feedback_shoulder_pixels,
            6.0,
        )
        self.assertEqual(numeric.config.velocity_median_window, 3)
        self.assertEqual(numeric.config.maximum_velocity_feedforward_fraction, 1.0)
        self.assertTrue(
            numeric.config.require_motion_corroboration_for_feedforward
        )
        self.assertEqual(
            numeric.config.maximum_body_derived_projection_fraction,
            1.0,
        )
        self.assertEqual(
            numeric.config.maximum_body_derived_feedforward_fraction,
            0.50,
        )
        self.assertEqual(
            numeric.config.maximum_body_derived_pursuit_feedforward_fraction,
            0.82,
        )
        self.assertEqual(
            numeric.config.maximum_residual_pursuit_feedforward_fraction,
            0.65,
        )
        self.assertEqual(
            numeric.config.maximum_correlated_lookahead_pursuit_feedforward_fraction,
            0.60,
        )
        self.assertEqual(
            numeric.config.maximum_verified_flow_pursuit_feedforward_fraction,
            0.0,
        )
        self.assertEqual(len(controller.updates), 3)
        first_target, _shape, _active, first_keywords = controller.updates[0]
        self.assertIsNone(first_target)
        self.assertNotIn("aim_point", first_keywords)
        mapped_target, _shape, _active, mapped_keywords = controller.updates[1]
        self.assertIs(mapped_target, raw_target)
        self.assertEqual(mapped_keywords["aim_point"], measured_head_anchor.point)
        self.assertEqual(
            mapped_keywords["measurement_ns"],
            measured_head_anchor.source_timestamp_ns,
        )
        self.assertNotIn("velocity_target", mapped_keywords)
        self.assertEqual(
            mapped_keywords["velocity_point"],
            measured_head_anchor.velocity_point,
        )
        self.assertTrue(mapped_keywords["body_derived_motion_permitted"])
        self.assertEqual(
            mapped_keywords["body_derived_motion_deadline_ns"],
            measured_head_anchor.identity_deadline_ns,
        )
        self.assertEqual(
            mapped_keywords["identity_deadline_ns"],
            measured_head_anchor.identity_deadline_ns,
        )
        self.assertTrue(mapped_keywords["measurement_observed"])
        self.assertNotIn("motion_corroboration_point", mapped_keywords)

        continued_target, _shape, _active, continued_keywords = controller.updates[2]
        self.assertEqual(continued_target, raw_target)
        self.assertEqual(
            continued_keywords["aim_point"],
            continued_head_anchor.point,
        )
        self.assertEqual(
            continued_keywords["measurement_ns"],
            continued_head_anchor.source_timestamp_ns,
        )
        self.assertEqual(
            continued_keywords["velocity_point"],
            continued_head_anchor.velocity_point,
        )
        self.assertTrue(continued_keywords["body_derived_motion_permitted"])
        self.assertEqual(
            continued_keywords["body_derived_motion_deadline_ns"],
            continued_head_anchor.identity_deadline_ns,
        )
        self.assertEqual(
            continued_keywords["identity_deadline_ns"],
            continued_head_anchor.identity_deadline_ns,
        )
        self.assertTrue(continued_keywords["measurement_observed"])
        self.assertNotIn("velocity_target", continued_keywords)
        self.assertNotIn("motion_corroboration_point", continued_keywords)
        self.assertEqual(
            controller.control_events,
            [
                "update",
                "update",
                "update",
            ],
        )
        self.assertEqual(head_runtime.submit.call_count, 2)
        self.assertEqual(cleanup_order, ["aim", "head"])
        startup = output.getvalue()
        self.assertIn("control automatic command-aware observer", startup)
        self.assertIn(
            "head source pinned SunXDS 0.8.0 direct boxes on "
            "MIGraphXExecutionProvider GPU-only (CPU fallback disabled)",
            startup,
        )
        self.assertIn("direct-head confidence >= 0.15", startup)
        self.assertIn("direct results establish the head anchor", startup)
        self.assertIn("already-qualified fast pursuit <= 60%", startup)
        self.assertIn(
            "current measured primary geometry carries position for at most 750 ms",
            startup,
        )
        self.assertIn(
            "a fresh verified no-head decoder miss may use a position-only body proxy",
            startup,
        )
        self.assertIn("predicted primary geometry remains display-only", startup)
        self.assertIn(
            "verified mapped-motion source-age projection 100% | "
            "feed-forward baseline/aligned/fast max 25%/50%/82%",
            startup,
        )
        self.assertIn("closed-loop trailing-residual max 65%", startup)
        self.assertIn(
            "latest-only 90 Hz acquisition / 24 Hz anchored maintenance "
            "(acquisition recovery on stale/repeated model misses or within "
            "300 ms of lease expiry)",
            startup,
        )

    def test_capture_phase_lookahead_is_atomic_and_uncorroborated_falls_back(
        self,
    ) -> None:
        def run_case(
            name: str,
            *,
            root_corroboration_point: tuple[float, float] | None,
        ):
            report_path = self.root / f"capture-phase-{name}.json"
            diagnostic_root = self.root / f"capture-phase-{name}-diagnostics"

            class PeekSource(_FakeSource):
                def __init__(self) -> None:
                    super().__init__()
                    self.base_ns = perf_counter_ns() - 20_000_000
                    self.current_image = np.zeros(self.shape, dtype=np.uint8)
                    self.newest_image = np.ones(self.shape, dtype=np.uint8)
                    self.newest_packet = FramePacket(
                        image=self.newest_image,
                        sequence=1,
                        read_started_ns=self.base_ns + 8_000_000,
                        read_completed_ns=self.base_ns + 8_100_000,
                    )

                def read(self, timeout: float | None = None):
                    self.read_calls += 1
                    self.read_timeouts.append(timeout)
                    return FramePacket(
                        image=self.current_image,
                        sequence=0,
                        read_started_ns=self.base_ns,
                        read_completed_ns=self.base_ns + 100_000,
                    )

                def peek_latest(self):
                    return self.newest_packet

            class RecordingMakcuController:
                instances: list["RecordingMakcuController"] = []

                def __init__(self, config, *, calibrated_controller=None) -> None:
                    self.config = config
                    self.calibrated_controller = calibrated_controller
                    self.activation_pressed = True
                    self.raw_activation_state = (True, True)
                    self.activation_requires_release = False
                    self.updates: list[
                        tuple[
                            Detection | None,
                            tuple[int, int, int],
                            bool,
                            dict[str, object],
                        ]
                    ] = []
                    self.correlated_updates: list[
                        tuple[
                            Detection,
                            tuple[int, int, int],
                            bool,
                            dict[str, object],
                        ]
                    ] = []
                    self.__class__.instances.append(self)

                def start(self) -> None:
                    return None

                def stop(self) -> None:
                    return None

                def update(
                    self,
                    target,
                    frame_shape,
                    *,
                    active=True,
                    **keywords,
                ) -> None:
                    self.updates.append(
                        (target, frame_shape, active, dict(keywords))
                    )

                def update_correlated_lookahead(
                    self,
                    target,
                    frame_shape,
                    *,
                    active=True,
                    **keywords,
                ) -> None:
                    self.correlated_updates.append(
                        (target, frame_shape, active, dict(keywords))
                    )

                def revoke_motion_corroboration(self) -> None:
                    return None

                def telemetry_snapshot(self) -> MakcuTelemetrySnapshot:
                    return MakcuTelemetrySnapshot()

            source = PeekSource()
            target = Detection(0, "player", 0.90, (20, 2, 28, 22))
            _FakeDetector.detection_batches = [[target]]
            identity_deadline_ns = source.base_ns + 750_000_000
            root_sample = SimpleNamespace(
                point=(23.0, 5.0),
                velocity_point=(24.0, 5.0),
                source_timestamp_ns=source.base_ns,
                direct_source_timestamp_ns=source.base_ns - 10_000_000,
                identity_deadline_ns=identity_deadline_ns,
                track_generation=3,
                provenance=DirectHeadProvenance.MEASURED_PRIMARY,
                confidence=0.90,
                evidence="same-frame mapped head",
                bridging=False,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=identity_deadline_ns,
                corroboration_point=root_corroboration_point,
                phase_advanced=False,
                phase_hops=0,
            )
            lookahead_sample = SimpleNamespace(
                point=(25.0, 5.0),
                velocity_point=(26.0, 5.0),
                source_timestamp_ns=source.newest_packet.read_started_ns,
                direct_source_timestamp_ns=root_sample.direct_source_timestamp_ns,
                identity_deadline_ns=identity_deadline_ns,
                track_generation=3,
                provenance=DirectHeadProvenance.MEASURED_PRIMARY,
                confidence=0.90,
                evidence="capture-phase LK endpoint",
                bridging=False,
                body_derived_motion_permitted=False,
                body_derived_motion_deadline_ns=None,
                corroboration_point=None,
                phase_advanced=True,
                phase_hops=1,
            )
            head_runtime = mock.Mock()
            head_runtime.provider = "MIGraphXExecutionProvider"
            head_runtime.status = SimpleNamespace()
            head_runtime.identity_generation = 7
            head_runtime.accept_body.return_value = False
            head_runtime.consume_motion_corroboration_revocation.return_value = False
            head_runtime.take_latest.return_value = None
            head_runtime.visible_sample.side_effect = [
                root_sample,
                lookahead_sample,
            ]
            head_runtime.has_live_measured_anchor.return_value = True
            head_runtime.revoke_body.return_value = False
            head_runtime.stop.return_value = True

            config = self._config(
                report_path,
                "--backend",
                "onnxruntime",
                "--device",
                "MIGRAPHX",
                "--require-full-provider",
                "--max-frames",
                "1",
                "--aim",
                "--aim-label",
                "player",
                "--ignore-self",
                "--aim-output",
                "makcu",
                "--aim-makcu-port",
                "/dev/serial/by-id/test-makcu",
                "--aim-makcu-tracking-mode",
                "direct-head",
                "--aim-diagnostic-dir",
                str(diagnostic_root),
            )
            with (
                mock.patch("main._build_capture", return_value=source),
                mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
                mock.patch(
                    "detection.onnx_yolo.OnnxRuntimeYoloDetector",
                    _FakeDetector,
                ),
                mock.patch(
                    "aiming.MakcuAimingController",
                    RecordingMakcuController,
                ),
                mock.patch(
                    "main._build_automatic_head_runtime",
                    return_value=head_runtime,
                ),
            ):
                self.assertEqual(run(config), 0)

            sessions = list(diagnostic_root.iterdir())
            self.assertEqual(len(sessions), 1)
            records = [
                json.loads(line)
                for line in (sessions[0] / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(records), 1)
            return (
                RecordingMakcuController.instances[0],
                head_runtime,
                source,
                target,
                root_sample,
                lookahead_sample,
                records[0],
            )

        (
            controller,
            head_runtime,
            source,
            target,
            root_sample,
            lookahead_sample,
            record,
        ) = run_case(
            "correlated",
            root_corroboration_point=(16.0, 16.0),
        )
        self.assertEqual(len(controller.correlated_updates), 1)
        published_target, shape, active, keywords = (
            controller.correlated_updates[0]
        )
        self.assertIs(published_target, target)
        self.assertEqual(shape, source.shape)
        self.assertTrue(active)
        self.assertEqual(
            keywords,
            {
                "primary_measurement_ns": root_sample.source_timestamp_ns,
                "primary_aim_point": root_sample.point,
                "primary_velocity_point": root_sample.velocity_point,
                "primary_motion_corroboration_point": (
                    root_sample.corroboration_point
                ),
                "lookahead_measurement_ns": (
                    lookahead_sample.source_timestamp_ns
                ),
                "lookahead_aim_point": lookahead_sample.point,
                "lookahead_velocity_point": lookahead_sample.velocity_point,
                "identity_deadline_ns": root_sample.identity_deadline_ns,
                "runtime_identity_generation": 7,
                "track_generation": root_sample.track_generation,
                "verified_flow_motion": False,
            },
        )
        self.assertFalse(any(update[0] is not None for update in controller.updates))
        phase_events = [
            call[0]
            for call in head_runtime.method_calls
            if call[0]
            in {"visible_sample", "remember_newer_capture_frame"}
        ]
        self.assertEqual(
            phase_events,
            [
                "visible_sample",
                "remember_newer_capture_frame",
                "visible_sample",
            ],
        )
        remember_call = head_runtime.remember_newer_capture_frame.call_args
        assert remember_call is not None
        self.assertIs(remember_call.args[0], source.newest_image)
        self.assertEqual(
            remember_call.kwargs["source_timestamp_ns"],
            lookahead_sample.source_timestamp_ns,
        )
        self.assertEqual(record["control_source"], "capture-phase-correlated")
        self.assertEqual(
            record["correlated_root_sample"]["source_timestamp_ns"],
            root_sample.source_timestamp_ns,
        )

        (
            fallback_controller,
            _fallback_runtime,
            _fallback_source,
            fallback_target,
            _fallback_root,
            fallback_lookahead,
            fallback_record,
        ) = run_case(
            "uncorroborated",
            root_corroboration_point=None,
        )
        self.assertEqual(fallback_controller.correlated_updates, [])
        target_updates = [
            update
            for update in fallback_controller.updates
            if update[0] is not None
        ]
        self.assertEqual(len(target_updates), 1)
        ordinary_target, _shape, _active, ordinary_keywords = target_updates[0]
        self.assertIs(ordinary_target, fallback_target)
        self.assertEqual(
            ordinary_keywords["measurement_ns"],
            fallback_lookahead.source_timestamp_ns,
        )
        self.assertEqual(
            ordinary_keywords["aim_point"],
            fallback_lookahead.point,
        )
        self.assertNotIn("motion_corroboration_point", ordinary_keywords)
        self.assertEqual(fallback_record["control_source"], "capture-phase")
        self.assertIsNone(fallback_record["correlated_root_sample"])

    def test_uncertain_wide_self_keeps_only_distinct_mapped_opponent(self) -> None:
        from aiming.controller import TargetTracker as RealTargetTracker
        from utils.self_filter import SelfAvatarFilter as RealSelfAvatarFilter

        report_path = self.root / "uncertain-wide-self-direct-head.json"

        class TimestampedSource(_FakeSource):
            def __init__(self) -> None:
                super().__init__(shape=(1080, 1920, 3))
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
        narrow_self = Detection(
            0,
            "player",
            0.90,
            (430, 560, 850, 1080),
        )
        wide_self = Detection(
            0,
            "player",
            0.90,
            (4, 370, 750, 1080),
        )
        wide_self_duplicate = Detection(
            0,
            "player",
            0.82,
            (8, 388, 744, 1076),
        )
        opponent = Detection(
            0,
            "player",
            0.90,
            (900, 438, 960, 558),
        )
        _FakeDetector.detection_batches = [
            [narrow_self, opponent],
            [narrow_self, opponent],
            [narrow_self, opponent],
            [wide_self, wide_self_duplicate, opponent],
        ]

        class RecordingSelfAvatarFilter(RealSelfAvatarFilter):
            instances: list["RecordingSelfAvatarFilter"] = []

            def __init__(self, zone) -> None:
                super().__init__(zone)
                self.results = []
                self.__class__.instances.append(self)

            def apply(self, detections, frame_shape):
                result = super().apply(detections, frame_shape)
                self.results.append(result)
                return result

        class RecordingTracker(RealTargetTracker):
            instances: list["RecordingTracker"] = []

            def __init__(self, **options) -> None:
                super().__init__(**options)
                self.detection_inputs: list[tuple[Detection, ...]] = []
                self.continuation_inputs: list[tuple[Detection, ...]] = []
                self.__class__.instances.append(self)

            def update(
                self,
                detections,
                frame_shape,
                *,
                continuation_detections=(),
                **keywords,
            ):
                self.detection_inputs.append(tuple(detections))
                self.continuation_inputs.append(tuple(continuation_detections))
                return super().update(
                    detections,
                    frame_shape,
                    continuation_detections=continuation_detections,
                    **keywords,
                )

        class RecordingMakcuController:
            instances: list["RecordingMakcuController"] = []

            def __init__(self, config, *, calibrated_controller=None) -> None:
                self.config = config
                self.calibrated_controller = calibrated_controller
                self.activation_pressed = True
                self.updates: list[
                    tuple[Detection | None, dict[str, object]]
                ] = []
                self.__class__.instances.append(self)

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def update(self, target, _frame_shape, **keywords) -> None:
                self.updates.append((target, dict(keywords)))

            def revoke_motion_corroboration(self) -> None:
                return None

            def telemetry_snapshot(self) -> MakcuTelemetrySnapshot:
                return MakcuTelemetrySnapshot()

        identity_deadline_ns = source.base_ns + 180_000_000
        direct_sample = SimpleNamespace(
            point=(930.0, 450.0),
            source_timestamp_ns=source.base_ns,
            direct_source_timestamp_ns=source.base_ns,
            identity_deadline_ns=identity_deadline_ns,
            track_generation=1,
            provenance=DirectHeadProvenance.DIRECT,
            confidence=0.9,
            evidence="direct test head box",
            bridging=False,
            body_derived_motion_permitted=False,
            body_derived_motion_deadline_ns=None,
            corroboration_point=(930.0, 498.0),
        )
        mapped_samples = [
            SimpleNamespace(
                point=(930.0 + index, 450.0 + index),
                source_timestamp_ns=source.base_ns + index * 8_000_000,
                direct_source_timestamp_ns=source.base_ns,
                identity_deadline_ns=identity_deadline_ns,
                track_generation=1,
                provenance=DirectHeadProvenance.MEASURED_PRIMARY,
                confidence=0.90 - index * 0.02,
                evidence="filtered direct-head anchor",
                bridging=False,
                body_derived_motion_permitted=True,
                body_derived_motion_deadline_ns=identity_deadline_ns,
                corroboration_point=None,
            )
            for index in range(4)
        ]
        head_runtime = mock.Mock()
        head_runtime.provider = "MIGraphXExecutionProvider"
        head_runtime.status = SimpleNamespace()
        head_runtime.identity_generation = 1
        head_runtime.accept_body.return_value = False
        head_runtime.consume_motion_corroboration_revocation.return_value = False
        head_runtime.take_latest.side_effect = [direct_sample, None, None, None]
        head_runtime.visible_sample.side_effect = mapped_samples
        head_runtime.revoke_body.return_value = False
        head_runtime.stop.return_value = True

        config = self._config(
            report_path,
            "--backend",
            "onnxruntime",
            "--device",
            "MIGRAPHX",
            "--require-full-provider",
            "--max-frames",
            "4",
            "--aim",
            "--aim-label",
            "player",
            "--ignore-self",
            "--aim-output",
            "makcu",
            "--aim-makcu-port",
            "/dev/serial/by-id/test-makcu",
            "--aim-makcu-tracking-mode",
            "direct-head",
        )
        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("detection.onnx_yolo.OnnxRuntimeYoloDetector", _FakeDetector),
            mock.patch("aiming.MakcuAimingController", RecordingMakcuController),
            mock.patch("aiming.TargetTracker", RecordingTracker),
            mock.patch(
                "utils.self_filter.SelfAvatarFilter",
                RecordingSelfAvatarFilter,
            ),
            mock.patch(
                "main._build_automatic_head_runtime",
                return_value=head_runtime,
            ),
        ):
            result = run(config)

        self.assertEqual(result, 0)
        exclusion_results = RecordingSelfAvatarFilter.instances[0].results
        self.assertEqual(len(exclusion_results), 4)
        self.assertFalse(exclusion_results[-1].aim_safe)
        self.assertEqual(
            exclusion_results[-1].uncertain_self_detections,
            (wide_self, wide_self_duplicate),
        )

        tracker = RecordingTracker.instances[0]
        self.assertEqual(
            tracker.detection_inputs,
            [(opponent,), (opponent,), (opponent,), (opponent,)],
        )
        self.assertEqual(tracker.continuation_inputs, [(), (), (), ()])

        self.assertEqual(head_runtime.accept_body.call_count, 4)
        for accepted_call in head_runtime.accept_body.call_args_list:
            self.assertEqual(accepted_call.args[0], opponent.box)
            self.assertEqual(accepted_call.kwargs["aim_box"], opponent.box)
            self.assertEqual(
                accepted_call.kwargs["corroboration_box"],
                opponent.box,
            )
        self.assertEqual(head_runtime.submit.call_count, 4)
        for submit_call in head_runtime.submit.call_args_list:
            self.assertIs(submit_call.args[1], opponent)
        head_runtime.revoke_body.assert_not_called()

        controller = RecordingMakcuController.instances[0]
        self.assertEqual(len(controller.updates), 5)
        self.assertIsNone(controller.updates[0][0])
        mapped_updates = controller.updates[1:]
        self.assertEqual(len(mapped_updates), 4)
        for index, (target, keywords) in enumerate(mapped_updates):
            self.assertIsNotNone(target)
            assert target is not None
            self.assertGreater(float(target.x1), float(wide_self.x2))
            self.assertEqual(keywords["aim_point"], mapped_samples[index].point)
            self.assertTrue(keywords["measurement_observed"])
            self.assertTrue(keywords["body_derived_motion_permitted"])
            self.assertEqual(
                keywords["identity_deadline_ns"],
                identity_deadline_ns,
            )

    def test_obvious_wide_self_guard_preserves_opponent_then_revokes_alone(
        self,
    ) -> None:
        from aiming.controller import TargetTracker as RealTargetTracker
        from utils.self_filter import SelfAvatarFilter as RealSelfAvatarFilter

        report_path = self.root / "obvious-wide-self-direct-head.json"

        class TimestampedSource(_FakeSource):
            def __init__(self) -> None:
                super().__init__(shape=(1080, 1920, 3))
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
        # Normalized width/height is about 0.65, intentionally above the
        # temporal filter's 0.60 acquisition ceiling.  Its bottom, height, and
        # configured-shoulder anchor satisfy the automatic obvious-self guard.
        wide_self = Detection(
            0,
            "player",
            0.90,
            (0, 370, 820, 1080),
        )
        opponent = Detection(
            0,
            "player",
            0.90,
            (900, 438, 960, 558),
        )
        _FakeDetector.detection_batches = [
            [wide_self, opponent],
            [wide_self],
        ]
        normalized_self_width = (wide_self.x2 - wide_self.x1) / source.shape[1]
        normalized_self_height = (wide_self.y2 - wide_self.y1) / source.shape[0]
        self.assertGreater(normalized_self_width / normalized_self_height, 0.60)
        self.assertGreaterEqual(wide_self.y2 / source.shape[0], 0.985)
        self.assertGreaterEqual(normalized_self_height, 0.25)

        class RecordingSelfAvatarFilter(RealSelfAvatarFilter):
            instances: list["RecordingSelfAvatarFilter"] = []

            def __init__(self, zone) -> None:
                super().__init__(zone)
                self.results = []
                self.__class__.instances.append(self)

            def apply(self, detections, frame_shape):
                result = super().apply(detections, frame_shape)
                self.results.append(result)
                return result

        class RecordingTracker(RealTargetTracker):
            instances: list["RecordingTracker"] = []

            def __init__(self, **options) -> None:
                super().__init__(**options)
                self.detection_inputs: list[tuple[Detection, ...]] = []
                self.reset_calls = 0
                self.__class__.instances.append(self)

            def reset(self) -> None:
                self.reset_calls += 1
                super().reset()

            def update(self, detections, frame_shape, **keywords):
                self.detection_inputs.append(tuple(detections))
                return super().update(detections, frame_shape, **keywords)

        class RecordingMakcuController:
            instances: list["RecordingMakcuController"] = []

            def __init__(self, config, *, calibrated_controller=None) -> None:
                self.config = config
                self.calibrated_controller = calibrated_controller
                self.activation_pressed = True
                self.updates: list[
                    tuple[Detection | None, dict[str, object]]
                ] = []
                self.__class__.instances.append(self)

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def update(self, target, _frame_shape, **keywords) -> None:
                self.updates.append((target, dict(keywords)))

            def revoke_motion_corroboration(self) -> None:
                return None

            def telemetry_snapshot(self) -> MakcuTelemetrySnapshot:
                return MakcuTelemetrySnapshot()

        identity_deadline_ns = source.base_ns + 180_000_000
        direct_sample = SimpleNamespace(
            point=(930.0, 450.0),
            source_timestamp_ns=source.base_ns,
            direct_source_timestamp_ns=source.base_ns,
            identity_deadline_ns=identity_deadline_ns,
            track_generation=1,
            provenance=DirectHeadProvenance.DIRECT,
            confidence=0.9,
            evidence="direct test head box",
            bridging=False,
            body_derived_motion_permitted=False,
            body_derived_motion_deadline_ns=None,
            corroboration_point=(930.0, 498.0),
        )
        mapped_sample = SimpleNamespace(
            point=(930.0, 450.0),
            source_timestamp_ns=source.base_ns,
            direct_source_timestamp_ns=source.base_ns,
            identity_deadline_ns=identity_deadline_ns,
            track_generation=1,
            provenance=DirectHeadProvenance.MEASURED_PRIMARY,
            confidence=0.9,
            evidence="filtered direct-head anchor",
            bridging=False,
            body_derived_motion_permitted=True,
            body_derived_motion_deadline_ns=identity_deadline_ns,
            corroboration_point=None,
        )
        head_runtime = mock.Mock()
        head_runtime.provider = "MIGraphXExecutionProvider"
        head_runtime.status = SimpleNamespace()
        head_runtime.identity_generation = 1
        head_runtime.accept_body.return_value = False
        head_runtime.consume_motion_corroboration_revocation.return_value = False
        head_runtime.take_latest.return_value = direct_sample
        head_runtime.visible_sample.return_value = mapped_sample
        head_runtime.revoke_body.return_value = True
        head_runtime.stop.return_value = True

        config = self._config(
            report_path,
            "--backend",
            "onnxruntime",
            "--device",
            "MIGRAPHX",
            "--require-full-provider",
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
            "--aim-makcu-tracking-mode",
            "direct-head",
        )
        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("detection.onnx_yolo.OnnxRuntimeYoloDetector", _FakeDetector),
            mock.patch("aiming.MakcuAimingController", RecordingMakcuController),
            mock.patch("aiming.TargetTracker", RecordingTracker),
            mock.patch(
                "utils.self_filter.SelfAvatarFilter",
                RecordingSelfAvatarFilter,
            ),
            mock.patch(
                "main._build_automatic_head_runtime",
                return_value=head_runtime,
            ),
        ):
            result = run(config)

        self.assertEqual(result, 0)
        exclusion_results = RecordingSelfAvatarFilter.instances[0].results
        self.assertEqual(len(exclusion_results), 2)
        for exclusion in exclusion_results:
            self.assertTrue(exclusion.aim_safe)
            self.assertEqual(exclusion.ignored_count, 0)
            self.assertEqual(exclusion.uncertain_self_detections, ())

        tracker = RecordingTracker.instances[0]
        self.assertEqual(tracker.detection_inputs, [(opponent,)])
        # One reset establishes the physical hold epoch; the second proves the
        # guarded exact self-only frame cannot borrow tracker prediction grace.
        self.assertEqual(tracker.reset_calls, 2)

        head_runtime.accept_body.assert_called_once()
        accepted_call = head_runtime.accept_body.call_args
        assert accepted_call is not None
        self.assertEqual(accepted_call.args[0], opponent.box)
        self.assertEqual(accepted_call.kwargs["aim_box"], opponent.box)
        self.assertEqual(
            accepted_call.kwargs["corroboration_box"],
            opponent.box,
        )
        head_runtime.submit.assert_called_once()
        self.assertIs(head_runtime.submit.call_args.args[1], opponent)
        head_runtime.take_latest.assert_called_once()
        head_runtime.visible_sample.assert_called_once()
        head_runtime.revoke_body.assert_called_once()

        controller = RecordingMakcuController.instances[0]
        self.assertEqual(len(controller.updates), 3)
        self.assertIsNone(controller.updates[0][0])
        mapped_target, mapped_keywords = controller.updates[1]
        self.assertIsNotNone(mapped_target)
        assert mapped_target is not None
        self.assertGreater(float(mapped_target.x1), float(wide_self.x2))
        self.assertEqual(mapped_keywords["aim_point"], mapped_sample.point)
        self.assertTrue(mapped_keywords["body_derived_motion_permitted"])
        self.assertIsNone(controller.updates[2][0])

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
        head_runtime.provider = "MIGraphXExecutionProvider"
        head_runtime.status = SimpleNamespace()
        head_runtime.stop.return_value = True
        head_runtime.raise_if_failed.side_effect = RuntimeError(
            "synthetic starved head failure"
        )
        config = self._config(
            report_path,
            "--backend",
            "onnxruntime",
            "--device",
            "MIGRAPHX",
            "--require-full-provider",
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
            "--aim-makcu-tracking-mode",
            "direct-head",
        )

        with (
            mock.patch("main._build_capture", return_value=source),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("detection.onnx_yolo.OnnxRuntimeYoloDetector", _FakeDetector),
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
                self.output_is_prediction = False
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
        self.assertEqual(FakeTracker.instances[0].options["lost_grace_frames"], 6)
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
