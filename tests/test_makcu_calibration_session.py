from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import unittest

from aiming.makcu import MakcuCalibrationSnapshot
from aiming.makcu_calibration_session import (
    CalibrationEvidenceError,
    CalibrationObservation,
    CalibrationRuntimeBinding,
    CalibrationSessionConfig,
    CalibrationSessionState,
    MAX_SESSION_EVIDENCE_BYTES,
    MakcuCalibrationSession,
    evidence_matches_binding,
    normalize_calibration_context,
    load_session_evidence,
    session_evidence_bytes,
    session_evidence_from_bytes,
    target_within_safe_roi,
    write_session_evidence_exclusive,
)


MS = 1_000_000


def _binding(**changes: object) -> CalibrationRuntimeBinding:
    values: dict[str, object] = {
        "model_sha256": "a" * 64,
        "labels_sha256": "b" * 64,
        "source_commit": "7be5eb145c38dac3495d15c4693392570561cb99",
        "build_identity": "linux-dev+dirty",
        "backend": "onnxruntime",
        "runtime_version": "1.28.0",
        "requested_provider": "MIGraphXExecutionProvider",
        "active_provider": "MIGraphXExecutionProvider",
        "active_device": "gfx1030-RX6950XT",
        "provider_options_sha256": "1" * 64,
        "physical_device_token": "2" * 64,
        "inference_width": 416,
        "inference_height": 416,
        "detail_pass_enabled": False,
        "capture_kind": "camera",
        "capture_backend": "V4L2",
        "capture_buffer_size": 2,
        "capture_index": "/dev/video0",
        "capture_width": 1920,
        "capture_height": 1080,
        "capture_fps": 125.0,
        "pixel_format": "NV12",
        "rotation_degrees": 0,
        "makcu_identity_token": "c" * 64,
        "activation_button": 1,
        "aim_label": "player",
        "head_ratio": 0.20,
        "invert_x": False,
        "invert_y": False,
        "context_name": "hip-fire+solo",
        "aim_mode": "hip",
    }
    values.update(changes)
    return CalibrationRuntimeBinding(**values)  # type: ignore[arg-type]


class FakeCalibrationController:
    def __init__(self) -> None:
        self.active = False
        self.events: list[tuple[int, int, int]] = []
        self.pending_axis: str | None = None
        self.pending_counts = 0
        self.pending_rate = 0.0
        self.request_ns = 0
        self.last_tick_ns = 0
        self.emitted_for_request = 0
        self.abort_reason: str | None = None
        self.enter_calls = 0
        self.exit_calls = 0
        self.requests: list[tuple[str, int, float]] = []
        self.corrupt_aggregate = False
        self.activation_requires_release = False
        self.physical_pressed = False
        self.activation_known = True
        self.emit_during_next_snapshot = False

    @property
    def raw_activation_state(self) -> tuple[bool, bool]:
        return self.activation_known, self.physical_pressed

    def enter_calibration_mode(self) -> object:
        if self.active:
            raise RuntimeError("already active")
        self.active = True
        self.enter_calls += 1
        self.events.clear()
        self.abort_reason = None
        self.activation_requires_release = True
        return object()

    def observe_activation(self, *, known: bool, pressed: bool) -> tuple[bool, bool]:
        if not known:
            self.activation_known = False
            self.physical_pressed = False
            return False, False
        self.activation_known = True
        self.physical_pressed = pressed
        if not pressed:
            self.activation_requires_release = False
        return True, pressed

    def publish_calibration_lease(
        self,
        valid: bool,
        measurement_ns: int,
        token: object,
    ) -> None:
        del measurement_ns, token
        if not self.active:
            raise RuntimeError("inactive")
        if not valid:
            self.abort_reason = self.abort_reason or "lease invalidated"
            self.pending_axis = None
            self.pending_counts = 0

    def request_calibration_pulse(
        self,
        axis: str,
        signed_counts: int,
        bounded_rate: float,
        token: object,
    ) -> None:
        del token
        if (
            not self.active
            or self.pending_counts
            or self.activation_requires_release
            or not self.physical_pressed
        ):
            raise RuntimeError("cannot queue")
        self.pending_axis = axis
        self.pending_counts = signed_counts
        self.pending_rate = bounded_rate
        self.request_ns = self.last_tick_ns
        self.emitted_for_request = 0
        self.requests.append((axis, signed_counts, bounded_rate))

    def advance_to(self, now_ns: int) -> None:
        self.last_tick_ns = now_ns
        if not self.pending_counts:
            return
        sign = 1 if self.pending_counts > 0 else -1
        absolute_requested = abs(self.pending_counts) + self.emitted_for_request
        elapsed_ms = max(0, (now_ns - self.request_ns) // MS)
        should_have_emitted = min(
            absolute_requested,
            math.floor(self.pending_rate * elapsed_ms / 1000.0),
        )
        while self.emitted_for_request < should_have_emitted:
            step = min(2, should_have_emitted - self.emitted_for_request)
            new_total = self.emitted_for_request + step
            timestamp_ns = self.request_ns + math.ceil(
                new_total * 1_000_000_000 / self.pending_rate
            )
            delta_x = sign * step if self.pending_axis == "x" else 0
            delta_y = sign * step if self.pending_axis == "y" else 0
            if self.events and timestamp_ns <= self.events[-1][0]:
                timestamp_ns = self.events[-1][0] + 1
            self.events.append((timestamp_ns, delta_x, delta_y))
            self.emitted_for_request += step
            self.pending_counts -= sign * step
        if not self.pending_counts:
            self.pending_axis = None
            self.pending_rate = 0.0

    def calibration_snapshot(self) -> MakcuCalibrationSnapshot:
        captured_ns = self.last_tick_ns
        if self.emit_during_next_snapshot and self.pending_counts:
            self.emit_during_next_snapshot = False
            sign = 1 if self.pending_counts > 0 else -1
            timestamp_ns = self.last_tick_ns + 1
            delta_x = sign if self.pending_axis == "x" else 0
            delta_y = sign if self.pending_axis == "y" else 0
            self.events.append((timestamp_ns, delta_x, delta_y))
            self.pending_counts -= sign
            self.emitted_for_request += 1
            captured_ns = timestamp_ns + 1
            if not self.pending_counts:
                self.pending_axis = None
                self.pending_rate = 0.0
        emitted_x = sum(value[1] for value in self.events)
        emitted_y = sum(value[2] for value in self.events)
        emitted_abs = sum(abs(value[1]) + abs(value[2]) for value in self.events)
        return MakcuCalibrationSnapshot(
            active=self.active,
            captured_ns=captured_ns,
            emitted_x=emitted_x + (1 if self.corrupt_aggregate else 0),
            emitted_y=emitted_y,
            emitted_abs_counts=emitted_abs,
            movement_commands=len(self.events),
            first_emitted_ns=self.events[0][0] if self.events else None,
            last_emitted_ns=self.events[-1][0] if self.events else None,
            emitted_events=tuple(self.events),
            pending_axis=self.pending_axis,
            pending_counts=self.pending_counts,
            pending_rate_counts_per_second=self.pending_rate,
            abort_reason=self.abort_reason,
        )

    def exit_calibration_mode(self, token: object) -> None:
        del token
        self.exit_calls += 1
        self.active = False
        self.pending_axis = None
        self.pending_counts = 0


class SessionHarness:
    def __init__(
        self,
        *,
        gain_x: float = 0.14,
        gain_y: float = 0.10,
        delay_ms: int = 24,
        noise_pixels: float = 0.02,
        cross_x_to_y: float = 0.0,
        cross_y_to_x: float = 0.0,
        sample_period_ms: int = 8,
        config: CalibrationSessionConfig | None = None,
    ) -> None:
        self.controller = FakeCalibrationController()
        self.session = MakcuCalibrationSession(
            self.controller,
            _binding(capture_fps=1000.0 / sample_period_ms),
            config=config,
            started_ns=1 * MS,
        )
        self.now_ns = 1 * MS
        self.gain_x = gain_x
        self.gain_y = gain_y
        self.delay_ns = delay_ms * MS
        self.noise_pixels = noise_pixels
        self.cross_x_to_y = cross_x_to_y
        self.cross_y_to_x = cross_y_to_x
        self.sample_period_ms = sample_period_ms
        self.sample_index = 0

    def _error(self) -> tuple[float, float]:
        visible_x = 0
        visible_y = 0
        for timestamp_ns, delta_x, delta_y in self.controller.events:
            if timestamp_ns + self.delay_ns <= self.now_ns:
                visible_x += delta_x
                visible_y += delta_y
        noise_x = self.noise_pixels * math.sin(self.sample_index * 0.71)
        noise_y = self.noise_pixels * math.cos(self.sample_index * 0.63)
        return (
            20.0
            - self.gain_x * visible_x
            - self.cross_y_to_x * visible_y
            + noise_x,
            -16.0
            - self.gain_y * visible_y
            - self.cross_x_to_y * visible_x
            + noise_y,
        )

    def observation(self, **changes: object) -> CalibrationObservation:
        error_x, error_y = self._error()
        values: dict[str, object] = {
            "measurement_ns": self.now_ns,
            "error_x": error_x,
            "error_y": error_y,
            "confidence": 0.95,
            "exact_label": True,
            "unique_candidates": 1,
            "self_safe": True,
            "is_prediction": False,
            "target_identity": "stationary-target-1",
            "normalized_bbox": (0.40, 0.20, 0.60, 0.86),
        }
        values.update(changes)
        return CalibrationObservation(**values)  # type: ignore[arg-type]

    def step(
        self,
        *,
        pressed: bool,
        known: bool = True,
        observation: CalibrationObservation | None | Literal["default"] = "default",
        milliseconds: int | None = None,
    ) -> CalibrationSessionState:
        self.now_ns += (milliseconds or self.sample_period_ms) * MS
        self.controller.advance_to(self.now_ns)
        self.sample_index += 1
        known, pressed = self.controller.observe_activation(
            known=known,
            pressed=pressed,
        )
        actual = self.observation() if observation == "default" else observation
        status = self.session.update(
            self.now_ns,
            activation_known=known,
            activation_pressed=pressed,
            observation=actual,
        )
        return status.state

    def arm(self) -> None:
        self.step(pressed=False)
        self.step(pressed=False)
        self.step(pressed=False, milliseconds=80)
        self.step(pressed=True)

    def run(self, maximum_steps: int = 1000) -> None:
        for _index in range(maximum_steps):
            if self.session.terminal:
                return
            self.step(pressed=True)
        raise AssertionError("session did not terminate")


class MakcuCalibrationSessionTests(unittest.TestCase):
    def test_happy_path_fits_unequal_gains_from_actual_emitted_events(self) -> None:
        harness = SessionHarness(gain_x=0.14, gain_y=0.10, delay_ms=24)
        harness.arm()
        harness.run()

        result = harness.session.result
        assert result is not None and result.fit is not None
        self.assertEqual(result.outcome, "success")
        self.assertAlmostEqual(result.fit.x.gain_pixels_per_count, 0.14, delta=0.014)
        self.assertAlmostEqual(result.fit.y.gain_pixels_per_count, 0.10, delta=0.010)
        self.assertAlmostEqual(result.fit.delay_seconds, 0.024, delta=0.008)
        self.assertLessEqual(
            sum(abs(dx) + abs(dy) for _timestamp, dx, dy in result.evidence.emitted_events),
            2400,
        )
        self.assertTrue(result.evidence.evidence_complete)
        self.assertTrue(evidence_matches_binding(result.evidence, _binding()))
        self.assertEqual(harness.controller.exit_calls, 1)

    def test_stationary_window_works_at_240hz_and_approximately_143hz(self) -> None:
        for sample_period_ms in (4, 7):
            with self.subTest(sample_period_ms=sample_period_ms):
                harness = SessionHarness(
                    sample_period_ms=sample_period_ms,
                    delay_ms=50,
                )
                harness.arm()
                harness.run(maximum_steps=3000)
                result = harness.session.result
                assert result is not None
                self.assertEqual(result.outcome, "success")

    def test_snapshot_capture_ceiling_accepts_concurrent_worker_event(self) -> None:
        harness = SessionHarness()
        harness.arm()
        while harness.session.state is not CalibrationSessionState.PULSE:
            harness.step(pressed=True)
        harness.controller.emit_during_next_snapshot = True
        harness.step(pressed=True, milliseconds=1)
        self.assertNotEqual(harness.session.state, CalibrationSessionState.ABORTED)

    def test_known_release_dwell_is_required_before_a_new_hold(self) -> None:
        harness = SessionHarness()
        for _index in range(20):
            harness.step(pressed=True)
        self.assertEqual(harness.session.state, CalibrationSessionState.WAIT_RELEASE)
        self.assertEqual(harness.controller.enter_calls, 1)
        harness.step(pressed=False, known=False)
        self.assertEqual(harness.session.state, CalibrationSessionState.WAIT_RELEASE)
        harness.step(pressed=False)
        harness.step(pressed=False, milliseconds=80)
        self.assertEqual(harness.session.state, CalibrationSessionState.WAIT_HOLD)
        harness.step(pressed=True)
        self.assertEqual(harness.controller.enter_calls, 1)
        self.assertEqual(harness.session.state, CalibrationSessionState.BASELINE_SETTLE)

    def test_prediction_after_exclusive_entry_aborts_and_exits(self) -> None:
        harness = SessionHarness()
        harness.arm()
        predicted = harness.observation(is_prediction=True)
        harness.step(pressed=True, observation=predicted)
        result = harness.session.result
        assert result is not None
        self.assertEqual(result.outcome, "aborted")
        self.assertIn("predicted", result.reason)
        self.assertFalse(harness.controller.active)
        self.assertEqual(harness.controller.requests, [])

    def test_release_during_pulse_aborts_without_later_request(self) -> None:
        harness = SessionHarness()
        harness.arm()
        while harness.session.state is not CalibrationSessionState.PULSE:
            harness.step(pressed=True)
        request_count = len(harness.controller.requests)
        harness.step(pressed=False)
        self.assertEqual(harness.session.state, CalibrationSessionState.ABORTED)
        self.assertEqual(len(harness.controller.requests), request_count)
        self.assertFalse(harness.controller.active)

    def test_complete_target_bbox_must_remain_inside_safe_roi(self) -> None:
        harness = SessionHarness()
        harness.arm()
        clipped = harness.observation(normalized_bbox=(0.0, 0.20, 0.60, 0.86))
        harness.step(pressed=True, observation=clipped)
        result = harness.session.result
        assert result is not None
        self.assertIn("safe ROI", result.reason)

    def test_discontinuous_bbox_aborts_even_when_caller_identity_is_constant(self) -> None:
        harness = SessionHarness()
        harness.arm()
        replacement = harness.observation(
            measurement_ns=harness.now_ns + harness.sample_period_ms * MS,
            target_identity="stationary-target-1",
            normalized_bbox=(0.65, 0.20, 0.85, 0.86),
        )
        harness.step(pressed=True, observation=replacement)
        result = harness.session.result
        assert result is not None
        self.assertEqual(result.outcome, "aborted")
        self.assertIn("bounding box changed discontinuously", result.reason)
        self.assertFalse(harness.controller.active)
        self.assertEqual(harness.controller.requests, [])

    def test_overlapping_bbox_jitter_preserves_target_continuity(self) -> None:
        harness = SessionHarness()
        harness.arm()
        jittered = harness.observation(
            measurement_ns=harness.now_ns + harness.sample_period_ms * MS,
            normalized_bbox=(0.41, 0.19, 0.61, 0.85),
        )
        harness.step(pressed=True, observation=jittered)
        self.assertNotEqual(harness.session.state, CalibrationSessionState.ABORTED)
        self.assertTrue(harness.controller.active)

    def test_corrupted_controller_aggregate_aborts(self) -> None:
        harness = SessionHarness()
        harness.arm()
        while harness.session.state is not CalibrationSessionState.PULSE:
            harness.step(pressed=True)
        harness.controller.corrupt_aggregate = True
        harness.step(pressed=True)
        result = harness.session.result
        assert result is not None
        self.assertEqual(result.outcome, "aborted")
        self.assertIn("aggregate", result.reason)

    def test_excursion_over_100_pixels_aborts(self) -> None:
        harness = SessionHarness(gain_x=3.0, gain_y=0.10)
        harness.arm()
        harness.run()
        result = harness.session.result
        assert result is not None
        self.assertEqual(result.outcome, "aborted")
        self.assertIn("100px", result.reason)

    def test_cross_axis_excursion_over_100_pixels_aborts(self) -> None:
        harness = SessionHarness(cross_x_to_y=3.0)
        harness.arm()
        harness.run()
        result = harness.session.result
        assert result is not None
        self.assertEqual(result.outcome, "aborted")
        self.assertIn("cross-axis", result.reason)

    def test_arm_timeout_bounds_prehold_observation_retention(self) -> None:
        config = CalibrationSessionConfig(
            arm_timeout_seconds=0.20,
            maximum_prehold_records=3,
        )
        harness = SessionHarness(config=config)
        for _index in range(50):
            if harness.session.terminal:
                break
            harness.step(pressed=True)
        result = harness.session.result
        assert result is not None
        self.assertEqual(result.outcome, "aborted")
        self.assertIn("arming timed out", result.reason)
        self.assertLessEqual(len(result.evidence.observations), 3)


class CalibrationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        harness = SessionHarness()
        harness.arm()
        harness.run()
        assert harness.session.result is not None
        cls.evidence = harness.session.result.evidence

    def test_context_normalization_and_complete_bbox_helper(self) -> None:
        self.assertEqual(normalize_calibration_context("hip-fire+solo"), "hip")
        self.assertEqual(normalize_calibration_context("ads_ranked"), "ads")
        with self.assertRaises(ValueError):
            normalize_calibration_context("ranked")
        self.assertTrue(target_within_safe_roi((0.1, 0.1, 0.9, 0.9), 0.08))
        self.assertFalse(target_within_safe_roi((0.0, 0.1, 0.9, 0.9), 0.08))

    def test_canonical_evidence_is_deterministic_and_binding_sensitive(self) -> None:
        first = session_evidence_bytes(self.evidence)
        self.assertEqual(first, session_evidence_bytes(self.evidence))
        self.assertTrue(first.endswith(b"\n"))
        self.assertEqual(session_evidence_from_bytes(first), self.evidence)
        self.assertFalse(
            evidence_matches_binding(
                self.evidence,
                replace(_binding(), active_device="different-device"),
            )
        )

    def test_exclusive_writer_is_mode_0600_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "evidence.json"
            write_session_evidence_exclusive(destination, self.evidence)
            previous = destination.read_bytes()
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            self.assertEqual(load_session_evidence(destination), self.evidence)
            with self.assertRaises(FileExistsError):
                write_session_evidence_exclusive(destination, self.evidence)
            self.assertEqual(destination.read_bytes(), previous)
            self.assertEqual(os.listdir(temporary), [destination.name])

    def test_file_loader_rejects_oversized_and_nonregular_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oversized = root / "oversized.json"
            with oversized.open("wb") as stream:
                stream.truncate(MAX_SESSION_EVIDENCE_BYTES + 1)
            if os.name == "posix":
                oversized.chmod(0o600)
            with self.assertRaisesRegex(CalibrationEvidenceError, "large"):
                load_session_evidence(oversized)
            with self.assertRaisesRegex(CalibrationEvidenceError, "regular file"):
                load_session_evidence(root)

    @unittest.skipUnless(os.name == "posix", "POSIX file modes and symlinks")
    def test_file_loader_requires_private_mode_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence_path = root / "evidence.json"
            evidence_path.write_bytes(session_evidence_bytes(self.evidence))
            evidence_path.chmod(0o644)
            with self.assertRaisesRegex(PermissionError, "mode 0600"):
                load_session_evidence(evidence_path)
            evidence_path.chmod(0o600)
            link = root / "evidence-link.json"
            link.symlink_to(evidence_path)
            with self.assertRaisesRegex(CalibrationEvidenceError, "regular file"):
                load_session_evidence(link)

    def test_loader_rejects_noncanonical_and_hash_tampering(self) -> None:
        canonical = session_evidence_bytes(self.evidence)
        document = json.loads(canonical)
        document["binding"]["active_device"] = "tampered-device"
        tampered = (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self.assertRaisesRegex(CalibrationEvidenceError, "artifact_sha256"):
            session_evidence_from_bytes(tampered)
        with self.assertRaisesRegex(CalibrationEvidenceError, "canonical"):
            session_evidence_from_bytes(json.dumps(json.loads(canonical)).encode())

    def test_loader_refits_even_if_attacker_recomputes_artifact_hash(self) -> None:
        document = json.loads(session_evidence_bytes(self.evidence))
        document["fit"]["x"]["gain_pixels_per_count"] += 0.01
        unsigned = dict(document)
        del unsigned["artifact_sha256"]
        unsigned_bytes = (
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        document["artifact_sha256"] = sha256(unsigned_bytes).hexdigest()
        forged = (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self.assertRaisesRegex(CalibrationEvidenceError, "fresh evidence refit"):
            session_evidence_from_bytes(forged)

    def test_binding_rejects_detail_pass_and_invalid_activation_or_head_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "detail pass"):
            _binding(detail_pass_enabled=True)
        with self.assertRaisesRegex(ValueError, "activation_button"):
            _binding(activation_button=-1)
        with self.assertRaisesRegex(ValueError, "head_ratio"):
            _binding(head_ratio=0.75)
        with self.assertRaisesRegex(ValueError, "rotation_degrees"):
            _binding(rotation_degrees=False)
        with self.assertRaisesRegex(ValueError, "runtime_version"):
            _binding(runtime_version="unknown")
        self.assertEqual(_binding(activation_button=0).activation_button, 0)

    def test_aborted_evidence_never_matches_even_with_identical_binding(self) -> None:
        harness = SessionHarness()
        harness.arm()
        harness.step(
            pressed=True,
            observation=harness.observation(is_prediction=True),
        )
        result = harness.session.result
        assert result is not None
        self.assertEqual(result.outcome, "aborted")
        self.assertFalse(evidence_matches_binding(result.evidence, _binding()))


if __name__ == "__main__":
    unittest.main()
