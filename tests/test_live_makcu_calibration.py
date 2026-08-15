from __future__ import annotations

from dataclasses import replace
import contextlib
from hashlib import sha256
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
from time import perf_counter_ns
import unittest
from unittest import mock

import numpy as np

from aiming.makcu import MakcuTelemetrySnapshot
from aiming.makcu_calibration import AxisCalibrationFit, MakcuCalibrationFit
from aiming.makcu_calibration_activation import (
    ActiveMakcuCalibrationProfile,
    _profile_digest,
    write_active_profile_atomic,
)
from capture.base import CaptureStats, FramePacket
from config import parse_args
from detection.types import Detection
from main import (
    _build_calibration_runtime_binding,
    _calibrated_controller_from_active_profile,
    _calibration_model_sha256,
    _calibration_observation_and_target,
    run,
)
from utils.live_report import snapshot_artifact


def _active_profile(binding) -> ActiveMakcuCalibrationProfile:
    def axis(name: str, gain: float) -> AxisCalibrationFit:
        return AxisCalibrationFit(
            axis=name,
            gain_pixels_per_count=gain,
            delay_seconds=0.012,
            drift_pixels_per_second=0.0,
            r_squared=0.99,
            gain_cv=0.02,
            polarity_mismatch=0.01,
            cross_axis_ratio=0.01,
            delay_ambiguity_seconds=0.001,
            pulse_delay_spread_seconds=0.001,
            minimum_excursion_pixels=20.0,
            maximum_excursion_pixels=40.0,
            positive_pulses=2,
            negative_pulses=2,
        )

    fit = MakcuCalibrationFit(
        x=axis("x", 0.075),
        y=axis("y", 0.14),
        delay_seconds=0.012,
        detector_period_seconds=1.0 / 130.0,
        observation_duty=0.99,
        evidence_sha256="d" * 64,
    )
    placeholder = ActiveMakcuCalibrationProfile(
        binding=binding,
        fit=fit,
        session_artifact_sha256="e" * 64,
        core_evidence_sha256=fit.evidence_sha256,
        profile_sha256="0" * 64,
    )
    return replace(placeholder, profile_sha256=_profile_digest(placeholder))


class CalibrationInputHelperTests(unittest.TestCase):
    def test_one_strong_exact_target_is_normalized_to_reference_pixels(self) -> None:
        target = Detection(0, "player", 0.85, (760.0, 300.0, 1160.0, 900.0))

        observation, selected = _calibration_observation_and_target(
            [target],
            (1080, 1920, 3),
            aim_label="player",
            head_ratio=0.12,
            configured_confidence=0.70,
            invert_x=False,
            invert_y=False,
            self_exclusion_safe=True,
            measurement_ns=123,
        )

        self.assertIs(selected, target)
        assert observation is not None
        self.assertEqual(observation.measurement_ns, 123)
        self.assertEqual(observation.error_x, 0.0)
        self.assertEqual(observation.error_y, -168.0)
        self.assertEqual(
            observation.normalized_bbox,
            (760.0 / 1920.0, 300.0 / 1080.0, 1160.0 / 1920.0, 900.0 / 1080.0),
        )
        self.assertTrue(observation.exact_label)
        self.assertTrue(observation.self_safe)
        self.assertFalse(observation.is_prediction)
        self.assertEqual(observation.unique_candidates, 1)

    def test_unsafe_weak_ambiguous_or_out_of_frame_inputs_are_not_observations(self) -> None:
        valid = Detection(0, "player", 0.85, (760.0, 300.0, 1160.0, 900.0))
        cases = (
            ([valid], False),
            ([replace(valid, confidence=0.69)], True),
            ([valid, replace(valid, xyxy=(500.0, 300.0, 700.0, 900.0))], True),
            ([replace(valid, xyxy=(-1.0, 300.0, 1160.0, 900.0))], True),
            ([replace(valid, class_name="person")], True),
        )
        for detections, self_safe in cases:
            with self.subTest(detections=detections, self_safe=self_safe):
                self.assertEqual(
                    _calibration_observation_and_target(
                        detections,
                        (1080, 1920, 3),
                        aim_label="player",
                        head_ratio=0.12,
                        configured_confidence=0.70,
                        invert_x=False,
                        invert_y=False,
                        self_exclusion_safe=self_safe,
                        measurement_ns=123,
                    ),
                    (None, None),
                )

    def test_axis_inversion_uses_the_same_logical_error_as_normal_control(self) -> None:
        target = Detection(0, "player", 0.90, (960.0, 540.0, 1160.0, 1040.0))

        for invert_x, invert_y, expected in (
            (False, False, (100.0, 60.0)),
            (True, False, (-100.0, 60.0)),
            (False, True, (100.0, -60.0)),
            (True, True, (-100.0, -60.0)),
        ):
            with self.subTest(invert_x=invert_x, invert_y=invert_y):
                observation, selected = _calibration_observation_and_target(
                    [target],
                    (1080, 1920, 3),
                    aim_label="player",
                    head_ratio=0.12,
                    configured_confidence=0.70,
                    invert_x=invert_x,
                    invert_y=invert_y,
                    self_exclusion_safe=True,
                    measurement_ns=123,
                )
                self.assertIs(selected, target)
                assert observation is not None
                self.assertEqual((observation.error_x, observation.error_y), expected)

    def test_inverted_logical_error_has_positive_gain_for_reversed_physical_axis(self) -> None:
        initial = Detection(0, "player", 0.90, (960.0, 540.0, 1160.0, 1040.0))
        # On a reversed physical mapping, a positive raw count moves the
        # captured crosshair left/up, so target-minus-crosshair grows right/down.
        responses = (
            (replace(initial, xyxy=(970.0, 540.0, 1170.0, 1040.0)), True, False, "x"),
            (replace(initial, xyxy=(960.0, 550.0, 1160.0, 1050.0)), False, True, "y"),
        )
        for moved, invert_x, invert_y, axis in responses:
            with self.subTest(axis=axis):
                before, _ = _calibration_observation_and_target(
                    [initial],
                    (1080, 1920, 3),
                    aim_label="player",
                    head_ratio=0.12,
                    configured_confidence=0.70,
                    invert_x=invert_x,
                    invert_y=invert_y,
                    self_exclusion_safe=True,
                    measurement_ns=100,
                )
                after, _ = _calibration_observation_and_target(
                    [moved],
                    (1080, 1920, 3),
                    aim_label="player",
                    head_ratio=0.12,
                    configured_confidence=0.70,
                    invert_x=invert_x,
                    invert_y=invert_y,
                    self_exclusion_safe=True,
                    measurement_ns=200,
                )
                assert before is not None and after is not None
                before_error = before.error_x if axis == "x" else before.error_y
                after_error = after.error_x if axis == "x" else after.error_y
                fitted_gain = -(after_error - before_error) / 10.0
                self.assertEqual(fitted_gain, 1.0)

    @mock.patch.dict(
        "os.environ",
        {"ROCR_VISIBLE_DEVICES": "GPU-1621aa76fbfff6bf"},
        clear=True,
    )
    def test_binding_uses_runtime_capture_and_rocr_physical_identity(self) -> None:
        config = parse_args(
            [
                "--source",
                "4",
                "--backend",
                "onnxruntime",
                "--device",
                "MIGRAPHX",
                "--no-preview",
                "--aim",
                "--aim-label",
                "player",
                "--ignore-self",
                "--aim-output",
                "makcu",
                "--aim-makcu-port",
                "/dev/serial/by-id/test-makcu",
                "--aim-calibration-context",
                "ads-scope",
            ]
        )
        binding = _build_calibration_runtime_binding(
            config,
            detector_summary={
                "runtime": "ONNX Runtime",
                "onnxruntime_version": "1.28.0",
                "requested_provider": "MIGraphXExecutionProvider",
                "active_providers": ["MIGraphXExecutionProvider"],
                "device": "gfx1030-RX6950XT",
                "input_shape": [1, 3, 384, 640],
                "provider_options": {
                    "MIGraphXExecutionProvider": {"device_id": "0"}
                },
                "provider_options_status": "ok",
                "require_full_provider": True,
                "runtime_ep_fail_fallback_disabled": True,
            },
            capture_settings={
                "source": 4,
                "backend": "V4L2",
                "buffer_size": 2,
                "width": 1920,
                "height": 1080,
                "fps": 240.0,
                "pixel_format": "NV12",
                "rotation_degrees": 180,
            },
            makcu_identity_token="c" * 64,
            model_artifact_snapshot={
                "name": "model.onnx",
                "sha256": "a" * 64,
                "companions": [],
            },
            labels_artifact_snapshot={
                "name": "labels.txt",
                "sha256": "b" * 64,
                "companions": [],
            },
            source_identity=(
                "7be5eb145c38dac3495d15c4693392570561cb99",
                "source-tree-sha256:" + "d" * 64,
            ),
        )

        self.assertEqual(binding.model_sha256, "a" * 64)
        self.assertEqual(binding.labels_sha256, "b" * 64)
        self.assertEqual(binding.runtime_version, "1.28.0")
        self.assertEqual(binding.requested_provider, "MIGraphXExecutionProvider")
        self.assertEqual(len(binding.provider_options_sha256), 64)
        self.assertEqual(
            binding.physical_device_token,
            sha256(b"rocr:GPU-1621aa76fbfff6bf").hexdigest(),
        )
        self.assertEqual(binding.active_provider, "MIGraphXExecutionProvider")
        self.assertEqual(binding.active_device, "gfx1030-RX6950XT")
        self.assertEqual((binding.inference_width, binding.inference_height), (640, 384))
        self.assertEqual(binding.capture_kind, "camera")
        self.assertEqual(binding.capture_index, "4")
        self.assertEqual(binding.capture_fps, 240.0)
        self.assertEqual(binding.pixel_format, "NV12")
        self.assertEqual(binding.rotation_degrees, 180)
        self.assertEqual(binding.aim_mode, "ads")
        self.assertFalse(binding.detail_pass_enabled)

    def test_openvino_model_identity_includes_xml_and_bin(self) -> None:
        first = {
            "name": "model.xml",
            "sha256": "a" * 64,
            "companions": [{"name": "model.bin", "sha256": "b" * 64}],
        }
        changed_weights = {
            "name": "model.xml",
            "sha256": "a" * 64,
            "companions": [{"name": "model.bin", "sha256": "c" * 64}],
        }

        self.assertEqual(_calibration_model_sha256(first), _calibration_model_sha256(first))
        self.assertNotEqual(
            _calibration_model_sha256(first),
            _calibration_model_sha256(changed_weights),
        )


class _FakeDetector:
    def __init__(self, **arguments) -> None:
        size = arguments["inference_size"]
        height, width = (size, size) if isinstance(size, int) else size
        self.available_devices = ("GPU",)
        self.runtime_summary = {
            "runtime": "OpenVINO",
            "openvino_version": "test",
            "physical_device_identity": "test-openvino-gpu",
            "model_path": str(arguments["model_path"]),
            "requested_device": "GPU",
            "device": "GPU",
            "input_shape": [1, 3, height, width],
        }

    def warmup(self) -> None:
        return None

    def infer(self, _tensor):
        return np.zeros((1, 0, 6), dtype=np.float32)

    def postprocess(self, _raw, **_arguments):
        return []


class _FakeSource:
    description = "synthetic capture card"

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.error = None
        self.ended = False
        self.end_after_first = False
        self.settings_overrides: dict[str, object] = {}
        self.mutate_on_start: tuple[Path, bytes] | None = None
        self.read_calls = 0

    @property
    def actual_settings(self):
        return {
            "mode": "live",
            "source": 4,
            "backend": "synthetic",
            "buffer_size": 2,
            "width": 1920,
            "height": 1080,
            "fps": 240.0,
            "pixel_format": "NV12",
            "rotation_degrees": 0,
        } | self.settings_overrides

    @property
    def stats(self) -> CaptureStats:
        return CaptureStats(
            frames_read=self.read_calls,
            frames_delivered=self.read_calls,
        )

    def start(self) -> None:
        self.started = True
        if self.mutate_on_start is not None:
            path, payload = self.mutate_on_start
            path.write_bytes(payload)

    def read(self, timeout=None):
        del timeout
        self.read_calls += 1
        if self.end_after_first and self.read_calls > 1:
            self.ended = True
            return None
        completed = perf_counter_ns()
        return FramePacket(
            image=np.zeros((1080, 1920, 3), dtype=np.uint8),
            sequence=self.read_calls - 1,
            read_started_ns=completed - 1_000_000,
            read_completed_ns=completed,
        )

    def close(self) -> None:
        self.closed = True


class _FakeMakcuController:
    instances: list["_FakeMakcuController"] = []
    actual_identity_token = "c" * 64
    reject_updates = True

    def __init__(
        self,
        config,
        *,
        calibrated_controller=None,
        expected_identity_token=None,
    ) -> None:
        self.config = config
        self.calibrated_controller = calibrated_controller
        self.expected_identity_token = expected_identity_token
        self.identity_token = self.actual_identity_token
        self.raw_activation_state = (True, False)
        self.started = False
        self.stopped = False
        self.normal_updates = 0
        self.__class__.instances.append(self)

    @property
    def activation_pressed(self) -> bool:
        return False

    def start(self) -> None:
        if (
            self.expected_identity_token is not None
            and self.identity_token != self.expected_identity_token
        ):
            raise RuntimeError("synthetic MAKCU identity mismatch")
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def update(self, *_arguments, **_keywords) -> None:
        self.normal_updates += 1
        if self.reject_updates:
            raise AssertionError("normal MAKCU update ran during calibration")

    def telemetry_snapshot(self) -> MakcuTelemetrySnapshot:
        return MakcuTelemetrySnapshot()


class _State:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeCalibrationSession:
    outcome = "success"
    terminal_on_update = True
    instances: list["_FakeCalibrationSession"] = []

    def __init__(self, controller, binding, *, started_ns: int) -> None:
        self.controller = controller
        self.binding = binding
        self.started_ns = started_ns
        self.config = SimpleNamespace(minimum_confidence=0.70)
        self.terminal = False
        self.result = None
        self.update_calls = 0
        self.abort_calls = 0
        self.controller_stopped_at_abort = None
        self.__class__.instances.append(self)

    @staticmethod
    def _status(terminal: bool, outcome: str = "waiting"):
        return SimpleNamespace(
            state=_State(outcome),
            message=outcome,
            terminal=terminal,
            emitted_abs_counts=0,
            qualifying_x_positive=0,
            qualifying_x_negative=0,
            qualifying_y_positive=0,
            qualifying_y_negative=0,
        )

    def status(self):
        return self._status(False)

    def update_from_controller(self, _now_ns: int, *, observation) -> object:
        del observation
        self.update_calls += 1
        if not self.terminal_on_update:
            return self._status(False)
        self.terminal = True
        fit = None
        if self.outcome == "success":
            axis = SimpleNamespace(gain_pixels_per_count=0.1)
            fit = SimpleNamespace(x=axis, y=axis, delay_seconds=0.012)
        evidence = SimpleNamespace(artifact_sha256="e" * 64)
        self.result = SimpleNamespace(
            outcome=self.outcome,
            reason="synthetic result",
            fit=fit,
            evidence=evidence,
        )
        return self._status(True, self.outcome)

    def abort(self, _reason: str, *, now_ns: int) -> object:
        del now_ns
        self.abort_calls += 1
        if self.terminal:
            raise AssertionError("a terminal fake session should not be aborted")
        self.controller_stopped_at_abort = self.controller.stopped
        self.terminal = True
        evidence = SimpleNamespace(artifact_sha256="f" * 64)
        self.result = SimpleNamespace(
            outcome="aborted",
            reason="pipeline stopped",
            fit=None,
            evidence=evidence,
        )
        return self._status(True, "aborted")


class LiveCalibrationArbitrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.onnx"
        self.model.write_bytes(b"model")
        self.labels = self.root / "labels.txt"
        self.labels.write_text("player\n", encoding="utf-8")
        self.evidence_dir = self.root / "private"
        self.evidence_dir.mkdir(mode=0o700)
        self.evidence = self.evidence_dir / "calibration.json"
        self.source = _FakeSource()
        _FakeMakcuController.instances.clear()
        _FakeMakcuController.actual_identity_token = "c" * 64
        _FakeMakcuController.reject_updates = True
        _FakeCalibrationSession.instances.clear()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self):
        return parse_args(
            [
                "--source",
                "4",
                "--model",
                str(self.model),
                "--labels",
                str(self.labels),
                "--device",
                "GPU",
                "--inference-size",
                "32",
                "--no-preview",
                "--aim",
                "--aim-label",
                "player",
                "--ignore-self",
                "--aim-output",
                "makcu",
                "--aim-makcu-port",
                "/dev/serial/by-id/test-makcu",
                "--aim-calibration-evidence",
                str(self.evidence),
            ]
        )

    def _run(self, outcome: str, *, terminal_on_update: bool = True):
        _FakeCalibrationSession.outcome = outcome
        _FakeCalibrationSession.terminal_on_update = terminal_on_update
        writes: list[tuple[Path, object]] = []
        with (
            mock.patch("main._build_capture", return_value=self.source),
            mock.patch("main._calibration_source_identity", return_value=("7be5eb1", "test-build")),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("aiming.MakcuAimingController", _FakeMakcuController),
            mock.patch(
                "aiming.TargetTracker",
                side_effect=AssertionError("tracker was constructed during calibration"),
            ),
            mock.patch(
                "aiming.makcu_calibration_session.MakcuCalibrationSession",
                _FakeCalibrationSession,
            ),
            mock.patch(
                "aiming.makcu_calibration_session.write_session_evidence_exclusive",
                side_effect=lambda path, evidence: writes.append((Path(path), evidence)),
            ),
        ):
            if outcome == "success":
                result = run(self._config())
            else:
                with self.assertRaisesRegex(RuntimeError, "calibration aborted"):
                    run(self._config())
                result = None
        return result, writes

    def test_calibration_bypasses_tracker_and_normal_aim_and_writes_evidence(self) -> None:
        result, writes = self._run("success")

        self.assertEqual(result, 0)
        controller = _FakeMakcuController.instances[0]
        session = _FakeCalibrationSession.instances[0]
        self.assertTrue(controller.started)
        self.assertTrue(controller.stopped)
        self.assertEqual(controller.normal_updates, 0)
        self.assertEqual(session.update_calls, 1)
        self.assertEqual(session.abort_calls, 0)
        self.assertEqual(writes[0][0], self.evidence)
        self.assertTrue(self.source.closed)

    def test_aborted_calibration_still_writes_evidence_and_raises(self) -> None:
        result, writes = self._run("aborted")

        self.assertIsNone(result)
        self.assertEqual(len(writes), 1)
        self.assertTrue(_FakeMakcuController.instances[0].stopped)
        self.assertEqual(_FakeMakcuController.instances[0].normal_updates, 0)

    def test_pipeline_end_aborts_before_controller_stop_and_writes_once(self) -> None:
        self.source.end_after_first = True

        result, writes = self._run("aborted", terminal_on_update=False)

        self.assertIsNone(result)
        session = _FakeCalibrationSession.instances[0]
        self.assertEqual(session.update_calls, 1)
        self.assertEqual(session.abort_calls, 1)
        self.assertFalse(session.controller_stopped_at_abort)
        self.assertTrue(_FakeMakcuController.instances[0].stopped)
        self.assertEqual(len(writes), 1)


class ActiveProfileRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.onnx"
        self.model.write_bytes(b"model")
        self.labels = self.root / "labels.txt"
        self.labels.write_text("player\n", encoding="utf-8")
        self.profile_path = self.root / "active.json"
        self.source = _FakeSource()
        _FakeMakcuController.instances.clear()
        _FakeMakcuController.actual_identity_token = "c" * 64
        _FakeMakcuController.reject_updates = False
        self.base_config = parse_args(
            [
                "--source",
                "4",
                "--model",
                str(self.model),
                "--labels",
                str(self.labels),
                "--device",
                "GPU",
                "--inference-size",
                "32",
                "--no-preview",
                "--max-frames",
                "1",
                "--aim",
                "--aim-label",
                "player",
                "--aim-invert-x",
                "--aim-invert-y",
                "--ignore-self",
                "--aim-output",
                "makcu",
                "--aim-makcu-port",
                "/dev/serial/by-id/test-makcu",
                "--aim-makcu-max-step",
                "200",
                "--aim-makcu-vertical-rate-ratio",
                "0.5",
            ]
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _binding(self):
        return _build_calibration_runtime_binding(
            self.base_config,
            detector_summary={
                "runtime": "OpenVINO",
                "openvino_version": "test",
                "physical_device_identity": "test-openvino-gpu",
                "model_path": str(self.model),
                "requested_device": "GPU",
                "device": "GPU",
                "input_shape": [1, 3, 32, 32],
            },
            capture_settings=self.source.actual_settings,
            makcu_identity_token="c" * 64,
            model_artifact_snapshot=snapshot_artifact(self.model),
            labels_artifact_snapshot=snapshot_artifact(self.labels),
            source_identity=("7be5eb1", "test-build"),
            context_name="ads-scope",
            aim_mode="ads",
        )

    def _write_profile(self, *, binding=None):
        profile = _active_profile(binding or self._binding())
        write_active_profile_atomic(self.profile_path, profile)
        return profile

    def _config_with_profile(self):
        return replace(
            self.base_config,
            aim_makcu_active_profile=self.profile_path,
            aim_calibration_context="ads-scope",
        )

    def _run(self, config=None):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch("main._build_capture", return_value=self.source),
            mock.patch(
                "main._calibration_source_identity",
                return_value=("7be5eb1", "test-build"),
            ),
            mock.patch("detection.OpenVINOYoloDetector", _FakeDetector),
            mock.patch("aiming.MakcuAimingController", _FakeMakcuController),
        ):
            result = run(config or self._config_with_profile())
        return result, output.getvalue()

    def test_valid_ads_inverted_profile_selects_calibrated_controller_and_caps(self) -> None:
        profile = self._write_profile()

        result, output = self._run()

        self.assertEqual(result, 0)
        controller = _FakeMakcuController.instances[0]
        numeric = controller.calibrated_controller
        self.assertIsNotNone(numeric)
        self.assertEqual(controller.expected_identity_token, "c" * 64)
        self.assertEqual(controller.normal_updates, 1)
        self.assertTrue(controller.stopped)
        self.assertEqual(numeric.plant.gain_x_pixels_per_count, 0.075)
        self.assertEqual(numeric.plant.gain_y_pixels_per_count, 0.14)
        self.assertEqual(numeric.plant.delay_seconds, 0.012)
        self.assertEqual(
            numeric.config.maximum_rate_x_counts_per_second,
            12_000.0,
        )
        self.assertEqual(
            numeric.config.maximum_rate_y_counts_per_second,
            6_000.0,
        )
        self.assertEqual(self.base_config.aim_calibration_context, "hip-fire")
        self.assertIn("control calibrated", output)
        self.assertIn(f"profile {profile.profile_sha256[:12]}", output)
        self.assertIn("context ads-scope", output)
        self.assertNotIn(str(self.profile_path), output)
        self.assertNotIn("c" * 64, output)

    def test_active_ads_profile_rejects_implicit_hip_fire_context(self) -> None:
        self._write_profile()

        hip_fire_config = replace(
            self.base_config,
            aim_makcu_active_profile=self.profile_path,
            aim_calibration_context="hip-fire",
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self._run(hip_fire_config)

        controller = _FakeMakcuController.instances[0]
        self.assertEqual(controller.normal_updates, 0)
        self.assertTrue(controller.stopped)

    def test_capture_binding_mismatch_fails_before_update_without_legacy_fallback(self) -> None:
        self._write_profile()
        self.source.settings_overrides["fps"] = 120.0

        with self.assertRaisesRegex(ValueError, "exactly match"):
            self._run()

        controller = _FakeMakcuController.instances[0]
        self.assertIsNotNone(controller.calibrated_controller)
        self.assertEqual(controller.normal_updates, 0)
        self.assertTrue(controller.stopped)

    def test_makcu_identity_mismatch_fails_without_legacy_fallback(self) -> None:
        self._write_profile()
        _FakeMakcuController.actual_identity_token = "f" * 64

        with self.assertRaisesRegex(RuntimeError, "profile-bound controller"):
            self._run()

        controller = _FakeMakcuController.instances[0]
        self.assertIsNotNone(controller.calibrated_controller)
        self.assertEqual(controller.normal_updates, 0)
        self.assertTrue(controller.stopped)

    def test_model_change_during_startup_is_rejected_before_update(self) -> None:
        self._write_profile()
        self.source.mutate_on_start = (self.model, b"changed model")

        with self.assertRaisesRegex(RuntimeError, "Model artifact changed"):
            self._run()

        controller = _FakeMakcuController.instances[0]
        self.assertEqual(controller.normal_updates, 0)
        self.assertTrue(controller.stopped)

    def test_malformed_or_missing_profile_fails_before_detector_start(self) -> None:
        cases = ("malformed", "missing")
        for case in cases:
            with self.subTest(case=case):
                if case == "malformed":
                    self.profile_path.write_bytes(b"{}\n")
                    self.profile_path.chmod(0o600)
                config = self._config_with_profile()
                with (
                    mock.patch(
                        "detection.OpenVINOYoloDetector",
                        side_effect=AssertionError("detector started before profile load"),
                    ),
                    self.assertRaises((ValueError, FileNotFoundError)),
                ):
                    run(config)
                if self.profile_path.exists():
                    self.profile_path.unlink()

    def test_active_profile_rejects_detail_pass_before_detector_start(self) -> None:
        self._write_profile()
        config = replace(self._config_with_profile(), detail_crop_size=64)

        with (
            mock.patch(
                "detection.OpenVINOYoloDetector",
                side_effect=AssertionError("detector started with active detail pass"),
            ),
            self.assertRaisesRegex(ValueError, "cannot use the detail pass"),
        ):
            run(config)

    def test_absent_profile_preserves_legacy_controller_selection(self) -> None:
        result, output = self._run(self.base_config)

        self.assertEqual(result, 0)
        controller = _FakeMakcuController.instances[0]
        self.assertIsNone(controller.calibrated_controller)
        self.assertIsNone(controller.expected_identity_token)
        self.assertEqual(controller.normal_updates, 1)
        self.assertIn("strength", output)
        self.assertNotIn("control calibrated", output)


if __name__ == "__main__":
    unittest.main()
