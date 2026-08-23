from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import cv2
import numpy as np

from aiming.direct_head_anchor import DirectHeadAnchorSample, DirectHeadProvenance
from aiming.makcu import (
    MakcuNormalCommandRecord,
    MakcuNormalControlSnapshot,
    MakcuTelemetrySnapshot,
)
from aiming.makcu_calibrated_control import CalibratedControlOutput
from detection.head_detector import (
    HeadModelSpec,
    HeadAssociationOutcome,
    HeadCandidate,
    HeadLocalization,
    PreparedHeadInput,
    associate_head_to_player,
)
from detection.head_worker import (
    HeadLocalizationOutcome,
    HeadLocalizationReason,
    HeadObservation,
    HeadWorkerResult,
)
from detection.head_flow import PhaseAdvancedHead
from detection.types import Detection
from main import (
    AUTOMATIC_HEAD_EXACT_BODY_ASSOCIATION_MAX_SPEED_PIXELS_PER_SECOND,
    AUTOMATIC_HEAD_FLOW_MAX_BODY_RESIDUAL_PIXELS,
    AUTOMATIC_HEAD_FLOW_MAX_CONSECUTIVE_FAILURES,
    AUTOMATIC_HEAD_FLOW_MAX_DYNAMIC_RESIDUAL_PIXELS,
    AUTOMATIC_HEAD_FLOW_SMALL_PLAYER_LONGITUDINAL_RESIDUAL_CAP_PIXELS,
    AUTOMATIC_HEAD_FLOW_SMALL_PLAYER_RESIDUAL_CAP_PIXELS,
    AUTOMATIC_HEAD_LOCALIZATION_HZ,
    AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS,
    AUTOMATIC_HEAD_MAPPED_MAX_SPEED_PIXELS_PER_SECOND,
    AUTOMATIC_HEAD_MAPPED_STEP_ALLOWANCE_PIXELS,
    AUTOMATIC_HEAD_MAPPED_VELOCITY_FILTER_TIME_CONSTANT_SECONDS,
    AUTOMATIC_HEAD_NORMALIZED_ANCHOR_FILTER_TIME_CONSTANT_SECONDS,
    AUTOMATIC_HEAD_PROVIDER,
    AUTOMATIC_HEAD_TRACKING_LOCALIZATION_HZ,
    AUTOMATIC_HEAD_TRACKING_MAX_CONSECUTIVE_MISSES,
    AUTOMATIC_HEAD_TRACKING_MINIMUM_LEASE_REMAINING_SECONDS,
    AUTOMATIC_HEAD_VELOCITY_RECONCILIATION_TIME_CONSTANT_SECONDS,
    _AutomaticHeadRuntime,
    _AutomaticBodyFallbackGate,
    _PreparedDirectHeadLocalizer,
    _TimestampedPreparedHeadInput,
    _aim_diagnostic_head_sample,
    _aim_diagnostic_makcu_control,
    _build_automatic_head_runtime,
    _head_runtime_telemetry_summary,
    _publish_automatic_head_loss_once,
)


def _strict_primary_runtime_summary() -> dict[str, object]:
    return {
        "active_providers": [
            "MIGraphXExecutionProvider",
            "CPUExecutionProvider",
        ],
        "require_full_provider": True,
        "configured_session_options": {
            "disable_cpu_ep_fallback": True,
        },
        "runtime_ep_fail_fallback_disabled": True,
    }


class _FakeWorker:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.active_generation: int | None = None
        self.submissions: list[dict[str, object]] = []
        self.result: HeadWorkerResult | None = None
        self.failed = False

    def start(self) -> None:
        self.started = True

    def submit(self, payload, **metadata) -> bool:
        self.submissions.append({"payload": payload, **metadata})
        self.active_generation = int(metadata["identity_generation"])
        return True

    def advance_identity(self, generation: int) -> bool:
        self.active_generation = generation
        self.result = None
        return True

    def take_latest(self, generation: int):
        result = self.result
        self.result = None
        if result is None or result.identity_generation != generation:
            return None
        return result

    def raise_if_failed(self) -> None:
        if self.failed:
            raise RuntimeError("synthetic head failure")

    def stop(self) -> bool:
        self.stopped = True
        return True


class _ScriptedPhaseAdvancer:
    """Small deterministic flow source for runtime handoff regressions."""

    def __init__(self, actions) -> None:
        self.actions = list(actions)
        self.remembered: list[tuple[int, int]] = []
        self.feature_boxes: list[
            tuple[float, float, float, float] | None
        ] = []

    def remember(
        self,
        _frame,
        *,
        source_timestamp_ns: int,
        identity_generation: int,
    ) -> None:
        self.remembered.append(
            (int(source_timestamp_ns), int(identity_generation))
        )

    def advance(
        self,
        head_box,
        *,
        feature_box=None,
        anchor_point,
        anchor_timestamp_ns: int,
        identity_generation: int,
    ):
        self.feature_boxes.append(
            None
            if feature_box is None
            else tuple(float(value) for value in feature_box)
        )
        if not self.actions or not self.remembered:
            return None
        action = self.actions.pop(0)
        if action is None:
            return None
        if action == "step-right":
            point = (float(anchor_point[0]) + 3.0, float(anchor_point[1]))
        else:
            point = tuple(float(value) for value in action)
        dx = point[0] - float(anchor_point[0])
        dy = point[1] - float(anchor_point[1])
        source_timestamp_ns = self.remembered[-1][0]
        hops = sum(
            timestamp > int(anchor_timestamp_ns)
            for timestamp, generation in self.remembered
            if generation == int(identity_generation)
        )
        return PhaseAdvancedHead(
            point=point,
            head_box=(
                float(head_box[0]) + dx,
                float(head_box[1]) + dy,
                float(head_box[2]) + dx,
                float(head_box[3]) + dy,
            ),
            anchor_timestamp_ns=int(anchor_timestamp_ns),
            source_timestamp_ns=source_timestamp_ns,
            identity_generation=int(identity_generation),
            hops=hops,
            frames_spanned=hops,
            flow_measurements=1 if hops > 0 else 0,
            strategy="scripted",
            minimum_inlier_fraction=1.0,
            maximum_forward_backward_error=0.0,
            feature_box=(
                None
                if feature_box is None
                else tuple(
                    float(value) + (dx if index % 2 == 0 else dy)
                    for index, value in enumerate(feature_box)
                )
            ),
        )

    def clear(self) -> None:
        self.remembered.clear()


def _result(
    *,
    source_ns: int,
    generation: int = 0,
    point: tuple[float, float] | None = (321.0, 123.0),
    selected_player_box: tuple[float, float, float, float] = (
        100.0,
        100.0,
        300.0,
        700.0,
    ),
    head_box: tuple[float, float, float, float] | None = None,
    localization_reason: HeadLocalizationReason | None = None,
) -> HeadWorkerResult:
    observation = (
        None
        if point is None
        else HeadObservation(
            point,
            0.8,
            "direct test head box",
            head_box=head_box,
        )
    )
    return HeadWorkerResult(
        submission_id=1,
        source_timestamp_ns=source_ns,
        completed_timestamp_ns=source_ns + 1,
        identity_generation=generation,
        selected_player_box=selected_player_box,
        observation=observation,
        localization_reason=localization_reason,
    )


class AutomaticBodyFallbackGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = _AutomaticBodyFallbackGate(
            confidence=0.40,
            confirmations=2,
        )
        self.measurement_ns = 100_000_000

    def _observe(self, **changes) -> bool:
        values = {
            "tracker_generation": 3,
            "runtime_generation": 7,
            "measurement_ns": self.measurement_ns,
            "accepted_confidence": 0.8,
            "strict_self_safe": True,
            "body_update_deferred": False,
            "no_decoded_head_verified": True,
            "direct_seen": False,
        }
        self.measurement_ns += 1
        values.update(changes)
        return self.gate.observe(**values)

    def test_requires_two_strong_exact_measurements(self) -> None:
        self.assertFalse(self._observe())
        self.assertTrue(self._observe())
        self.assertTrue(self._observe())

    def test_every_unsafe_or_unverified_sample_withdraws_qualification(self) -> None:
        self.assertFalse(self._observe())
        self.assertTrue(self._observe())
        for changes in (
            {"accepted_confidence": None},
            {"accepted_confidence": 0.39},
            {"strict_self_safe": False},
            {"body_update_deferred": True},
            {"no_decoded_head_verified": False},
        ):
            self.assertFalse(self._observe(**changes))
            self.assertFalse(self._observe())
            self.assertTrue(self._observe())

    def test_direct_seen_latches_closed_until_identity_changes(self) -> None:
        self.assertFalse(self._observe())
        self.assertTrue(self._observe())
        self.assertFalse(self._observe(direct_seen=True))
        self.assertFalse(self._observe())
        self.assertFalse(self._observe())
        self.assertFalse(self._observe(runtime_generation=8))
        self.assertTrue(self._observe(runtime_generation=8))

    def test_duplicate_measurement_cannot_supply_two_confirmations(self) -> None:
        source_ns = 222_000_000
        self.assertFalse(self._observe(measurement_ns=source_ns))
        self.assertFalse(self._observe(measurement_ns=source_ns))
        self.assertFalse(self._observe(measurement_ns=source_ns + 1))
        self.assertTrue(self._observe(measurement_ns=source_ns + 2))


class AutomaticHeadRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = _FakeWorker()
        self.runtime = _AutomaticHeadRuntime(
            self.worker,
            submission_hz=60.0,
            stale_after_seconds=0.065,
        )
        self.player = Detection(
            0,
            "player",
            0.9,
            (100.0, 100.0, 300.0, 700.0),
        )
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def _accept_and_remember_adaptive_frame(
        self,
        runtime: _AutomaticHeadRuntime,
        timestamp_ns: int,
    ) -> None:
        self.assertFalse(
            runtime.accept_body(
                self.player.box,
                aim_box=self.player.box,
                corroboration_box=self.player.box,
                track_generation=3,
                source_timestamp_ns=timestamp_ns,
            )
        )
        self.assertTrue(
            runtime.remember_frame(
                self.frame,
                source_timestamp_ns=timestamp_ns,
            )
        )

    def _seed_adaptive_maintenance_cadence(
        self,
        worker: _FakeWorker,
        runtime: _AutomaticHeadRuntime,
        *,
        source_ns: int = 100_000_000,
    ) -> int:
        """Establish one direct anchor and three deterministic LK samples."""

        self._accept_and_remember_adaptive_frame(runtime, source_ns)
        self.assertTrue(
            runtime.submit(
                self.frame,
                self.player,
                source_timestamp_ns=source_ns,
            )
        )
        worker.result = _result(
            source_ns=source_ns,
            point=(200.0, 150.0),
            selected_player_box=self.player.box,
            head_box=(180.0, 120.0, 220.0, 180.0),
        )
        self.assertIsNotNone(runtime.take_latest(now_ns=source_ns + 1))
        for offset_ms, expected_submission in (
            (10, False),
            (20, False),
            (30, False),
        ):
            timestamp_ns = source_ns + offset_ms * 1_000_000
            self._accept_and_remember_adaptive_frame(runtime, timestamp_ns)
            self.assertEqual(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=timestamp_ns,
                ),
                expected_submission,
            )
        return source_ns

    def test_flow_body_corridor_scales_with_time_and_player_size_but_is_capped(
        self,
    ) -> None:
        large = self.player.box
        tiny = (100.0, 100.0, 180.0, 180.0)

        self.assertEqual(
            self.runtime._flow_body_residual_tolerance(
                large,
                large,
                elapsed_ns=0,
            ),
            AUTOMATIC_HEAD_FLOW_MAX_BODY_RESIDUAL_PIXELS,
        )
        self.assertEqual(
            self.runtime._flow_body_residual_tolerance(
                large,
                large,
                elapsed_ns=10_000_000,
            ),
            21.0,
        )
        self.assertEqual(
            self.runtime._flow_body_residual_tolerance(
                large,
                large,
                elapsed_ns=75_000_000,
            ),
            AUTOMATIC_HEAD_FLOW_MAX_DYNAMIC_RESIDUAL_PIXELS,
        )
        self.assertEqual(
            self.runtime._flow_body_residual_tolerance(
                tiny,
                tiny,
                elapsed_ns=75_000_000,
            ),
            18.0,
        )
        self.assertEqual(
            self.runtime._flow_body_residual_tolerance(
                large,
                (100.0, 100.0, 330.0, 740.0),
                elapsed_ns=10_000_000,
            ),
            29.0,
        )

    def test_small_player_directional_corridor_requires_exact_coherent_motion(
        self,
    ) -> None:
        runtime = _AutomaticHeadRuntime(
            _FakeWorker(),
            stale_after_seconds=0.2,
        )
        prior_box = (100.0, 100.0, 180.0, 180.0)
        previous_box = (108.0, 100.0, 188.0, 180.0)
        current_box = (116.0, 100.0, 196.0, 180.0)
        prior_ns = 90_000_000
        previous_ns = 100_000_000
        current_ns = 175_000_000
        reference = (146.0, 110.0)

        # Without an established exact trajectory, the helper is exactly the
        # legacy circular gate: 18 px is accepted and 20 px is not.
        self.assertTrue(
            runtime._flow_body_residual_is_safe(
                (reference[0] + 18.0, reference[1]),
                reference,
                previous_box,
                current_box,
                elapsed_ns=75_000_000,
                previous_body_source_timestamp_ns=previous_ns,
                current_body_source_timestamp_ns=current_ns,
            )
        )
        self.assertFalse(
            runtime._flow_body_residual_is_safe(
                (reference[0] + 20.0, reference[1]),
                reference,
                previous_box,
                current_box,
                elapsed_ns=75_000_000,
                previous_body_source_timestamp_ns=previous_ns,
                current_body_source_timestamp_ns=current_ns,
            )
        )
        self.assertFalse(
            runtime._flow_body_residual_is_safe(
                (reference[0] + 13.0, reference[1]),
                reference,
                previous_box,
                previous_box,
                elapsed_ns=0,
                previous_body_source_timestamp_ns=previous_ns,
                current_body_source_timestamp_ns=previous_ns,
            )
        )

        for timestamp_ns, body_box in (
            (prior_ns, prior_box),
            (previous_ns, previous_box),
            (current_ns, current_box),
        ):
            self.assertFalse(
                runtime.accept_body(
                    body_box,
                    aim_box=body_box,
                    corroboration_box=body_box,
                    track_generation=7,
                    source_timestamp_ns=timestamp_ns,
                )
            )
        self.assertIsNotNone(
            runtime.anchor.observe_direct(
                reference,
                current_box,
                track_generation=7,
                source_timestamp_ns=current_ns,
                confidence=0.8,
            )
        )
        self.assertEqual(
            runtime._flow_body_residual_tolerance(
                previous_box,
                current_box,
                elapsed_ns=current_ns - previous_ns,
            ),
            AUTOMATIC_HEAD_FLOW_SMALL_PLAYER_RESIDUAL_CAP_PIXELS,
        )
        self.assertTrue(
            runtime._flow_body_residual_is_safe(
                (
                    reference[0]
                    + AUTOMATIC_HEAD_FLOW_SMALL_PLAYER_LONGITUDINAL_RESIDUAL_CAP_PIXELS,
                    reference[1],
                ),
                reference,
                previous_box,
                current_box,
                elapsed_ns=current_ns - previous_ns,
                previous_body_source_timestamp_ns=previous_ns,
                current_body_source_timestamp_ns=current_ns,
            )
        )
        self.assertFalse(
            runtime._flow_body_residual_is_safe(
                (reference[0] + 25.0, reference[1]),
                reference,
                previous_box,
                current_box,
                elapsed_ns=current_ns - previous_ns,
                previous_body_source_timestamp_ns=previous_ns,
                current_body_source_timestamp_ns=current_ns,
            )
        )
        self.assertFalse(
            runtime._flow_body_residual_is_safe(
                (reference[0], reference[1] + 19.0),
                reference,
                previous_box,
                current_box,
                elapsed_ns=current_ns - previous_ns,
                previous_body_source_timestamp_ns=previous_ns,
                current_body_source_timestamp_ns=current_ns,
            )
        )

    def test_coherent_small_player_motion_accepts_aligned_flow_residual(self) -> None:
        worker = _FakeWorker()
        phase_advancer = _ScriptedPhaseAdvancer(
            ((130.0, 110.0), (138.0, 110.0), (166.0, 110.0))
        )
        runtime = _AutomaticHeadRuntime(
            worker,
            stale_after_seconds=0.2,
            phase_advancer=phase_advancer,
        )
        body_boxes = (
            (100.0, 100.0, 180.0, 180.0),
            (108.0, 100.0, 188.0, 180.0),
            (116.0, 100.0, 196.0, 180.0),
        )
        timestamps = (90_000_000, 100_000_000, 110_000_000)

        runtime.accept_body(
            body_boxes[0],
            aim_box=body_boxes[0],
            corroboration_box=body_boxes[0],
            track_generation=7,
            source_timestamp_ns=timestamps[0],
        )
        runtime.remember_frame(self.frame, source_timestamp_ns=timestamps[0])
        worker.result = _result(
            source_ns=timestamps[0],
            point=(130.0, 110.0),
            selected_player_box=body_boxes[0],
            head_box=(125.0, 102.0, 135.0, 114.0),
        )
        self.assertIsNotNone(runtime.take_latest(now_ns=timestamps[0] + 1))

        for timestamp_ns, body_box in zip(timestamps[1:], body_boxes[1:]):
            runtime.accept_body(
                body_box,
                aim_box=body_box,
                corroboration_box=body_box,
                track_generation=7,
                source_timestamp_ns=timestamp_ns,
            )
            runtime.remember_frame(self.frame, source_timestamp_ns=timestamp_ns)
            self.assertIsNone(runtime.take_latest(now_ns=timestamp_ns + 1))

        sample = runtime.visible_sample(now_ns=timestamps[-1] + 1)
        assert sample is not None
        self.assertEqual(sample.point, (166.0, 110.0))
        self.assertTrue(sample.phase_advanced)
        self.assertTrue(runtime._flow_pixel_observed_current)

    def test_transverse_small_player_flow_residual_falls_back_to_mapped_point(
        self,
    ) -> None:
        worker = _FakeWorker()
        phase_advancer = _ScriptedPhaseAdvancer(
            ((130.0, 110.0), (138.0, 110.0), (146.0, 129.0))
        )
        runtime = _AutomaticHeadRuntime(
            worker,
            stale_after_seconds=0.2,
            phase_advancer=phase_advancer,
        )
        body_boxes = (
            (100.0, 100.0, 180.0, 180.0),
            (108.0, 100.0, 188.0, 180.0),
            (116.0, 100.0, 196.0, 180.0),
        )
        timestamps = (90_000_000, 100_000_000, 110_000_000)

        runtime.accept_body(
            body_boxes[0],
            aim_box=body_boxes[0],
            corroboration_box=body_boxes[0],
            track_generation=7,
            source_timestamp_ns=timestamps[0],
        )
        runtime.remember_frame(self.frame, source_timestamp_ns=timestamps[0])
        worker.result = _result(
            source_ns=timestamps[0],
            point=(130.0, 110.0),
            selected_player_box=body_boxes[0],
            head_box=(125.0, 102.0, 135.0, 114.0),
        )
        self.assertIsNotNone(runtime.take_latest(now_ns=timestamps[0] + 1))

        for timestamp_ns, body_box in zip(timestamps[1:], body_boxes[1:]):
            runtime.accept_body(
                body_box,
                aim_box=body_box,
                corroboration_box=body_box,
                track_generation=7,
                source_timestamp_ns=timestamp_ns,
            )
            runtime.remember_frame(self.frame, source_timestamp_ns=timestamp_ns)
            self.assertIsNone(runtime.take_latest(now_ns=timestamp_ns + 1))

        sample = runtime.visible_sample(now_ns=timestamps[-1] + 1)
        assert sample is not None
        self.assertNotEqual(sample.point, (146.0, 129.0))
        self.assertFalse(sample.phase_advanced)
        self.assertTrue(sample.body_derived_motion_permitted)
        self.assertFalse(runtime._flow_pixel_observed_current)
        self.assertEqual(runtime._flow_point, (146.0, 110.0))

    def test_directional_corridor_does_not_widen_direct_flow_correction(self) -> None:
        worker = _FakeWorker()
        phase_advancer = _ScriptedPhaseAdvancer(
            (
                (130.0, 110.0),
                (138.0, 110.0),
                (146.0, 110.0),
                (154.0, 110.0),
                (167.0, 110.0),
            )
        )
        runtime = _AutomaticHeadRuntime(
            worker,
            stale_after_seconds=0.2,
            phase_advancer=phase_advancer,
        )
        body_boxes = tuple(
            (100.0 + offset, 100.0, 180.0 + offset, 180.0)
            for offset in (0.0, 8.0, 16.0, 24.0)
        )
        timestamps = (90_000_000, 100_000_000, 110_000_000, 120_000_000)

        runtime.accept_body(
            body_boxes[0],
            aim_box=body_boxes[0],
            corroboration_box=body_boxes[0],
            track_generation=7,
            source_timestamp_ns=timestamps[0],
        )
        runtime.remember_frame(self.frame, source_timestamp_ns=timestamps[0])
        worker.result = _result(
            source_ns=timestamps[0],
            point=(130.0, 110.0),
            selected_player_box=body_boxes[0],
            head_box=(125.0, 102.0, 135.0, 114.0),
        )
        self.assertIsNotNone(runtime.take_latest(now_ns=timestamps[0] + 1))

        for timestamp_ns, body_box in zip(timestamps[1:], body_boxes[1:]):
            runtime.accept_body(
                body_box,
                aim_box=body_box,
                corroboration_box=body_box,
                track_generation=7,
                source_timestamp_ns=timestamp_ns,
            )
            runtime.remember_frame(self.frame, source_timestamp_ns=timestamp_ns)
            if timestamp_ns != timestamps[-1]:
                self.assertIsNone(runtime.take_latest(now_ns=timestamp_ns + 1))

        self.assertEqual(runtime._flow_point, (154.0, 110.0))
        worker.result = _result(
            source_ns=timestamps[1],
            point=(138.0, 110.0),
            selected_player_box=body_boxes[1],
            head_box=(133.0, 102.0, 143.0, 114.0),
        )
        direct = runtime.take_latest(now_ns=timestamps[-1] + 1)
        assert direct is not None

        # The delayed direct phase is 13 px from healthy current LK.  It is
        # inside the association corridor, but still exceeds the deliberately
        # fixed 12 px direct-correction gate and must not nudge current flow.
        self.assertEqual(direct.point, (138.0, 110.0))
        self.assertEqual(runtime._flow_point, (154.0, 110.0))
        visible = runtime.visible_sample(now_ns=timestamps[-1] + 1)
        assert visible is not None
        self.assertEqual(visible.point, (154.0, 110.0))

    def test_live_measured_anchor_query_is_bounded_and_prediction_cannot_set_it(
        self,
    ) -> None:
        self.assertFalse(self.runtime.has_live_measured_anchor(now_ns=0))
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=1,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(200.0, 150.0),
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=110_000_000))
        self.assertTrue(
            self.runtime.has_live_measured_anchor(now_ns=110_000_000)
        )

        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=None,
            track_generation=1,
            source_timestamp_ns=120_000_000,
        )
        self.assertFalse(
            self.runtime.has_live_measured_anchor(now_ns=120_000_000)
        )
        self.assertFalse(
            self.runtime.has_live_measured_anchor(now_ns=300_000_000)
        )

    def test_submission_gate_prepares_only_owned_bounded_tensor(self) -> None:
        self.runtime.start()
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=1,
            source_timestamp_ns=100_000_000,
        )

        self.assertTrue(
            self.runtime.submit(
                self.frame,
                self.player,
                source_timestamp_ns=100_000_000,
            )
        )
        self.assertFalse(
            self.runtime.submit(
                self.frame,
                self.player,
                source_timestamp_ns=110_000_000,
            )
        )
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=1,
            source_timestamp_ns=117_000_000,
        )
        self.assertTrue(
            self.runtime.submit(
                self.frame,
                self.player,
                source_timestamp_ns=117_000_000,
            )
        )

        self.assertEqual(len(self.worker.submissions), 2)
        payload = self.worker.submissions[0]["payload"]
        self.assertIsInstance(payload, _TimestampedPreparedHeadInput)
        self.assertEqual(payload.source_timestamp_ns, 100_000_000)
        self.assertIsInstance(payload.prepared, PreparedHeadInput)
        self.assertEqual(payload.prepared.tensor.shape, (1, 3, 320, 320))
        self.assertEqual(payload.prepared.tensor.dtype, np.float32)
        self.assertLess(payload.prepared.tensor.nbytes, self.frame.nbytes)
        self.assertIsNot(payload.prepared.tensor, self.frame)

    def test_submission_gate_preserves_sixty_hz_phase_at_125_and_130_fps(self) -> None:
        for source_fps in (125.0, 130.0):
            with self.subTest(source_fps=source_fps):
                worker = _FakeWorker()
                runtime = _AutomaticHeadRuntime(worker, submission_hz=60.0)
                frame_count = round(source_fps * 10.0)
                with (
                    mock.patch(
                        "detection.head_detector.plan_head_crop",
                        return_value=object(),
                    ),
                    mock.patch(
                        "detection.head_detector.prepare_head_input",
                        return_value=object(),
                    ),
                ):
                    for index in range(frame_count):
                        timestamp_ns = round(
                            index * 1_000_000_000 / source_fps
                        )
                        runtime.accept_body(
                            self.player.box,
                            corroboration_box=self.player.box,
                            track_generation=1,
                            source_timestamp_ns=timestamp_ns,
                        )
                        runtime.submit(
                            self.frame,
                            self.player,
                            source_timestamp_ns=timestamp_ns,
                        )

                self.assertGreaterEqual(len(worker.submissions), 599)
                self.assertLessEqual(len(worker.submissions), 601)

    def test_default_submission_gate_preserves_90_hz_phase_at_130_fps(self) -> None:
        self.assertEqual(AUTOMATIC_HEAD_LOCALIZATION_HZ, 90.0)
        self.assertEqual(AUTOMATIC_HEAD_PROVIDER, "MIGraphXExecutionProvider")
        worker = _FakeWorker()
        runtime = _AutomaticHeadRuntime(worker)
        with (
            mock.patch(
                "detection.head_detector.plan_head_crop",
                return_value=object(),
            ),
            mock.patch(
                "detection.head_detector.prepare_head_input",
                return_value=object(),
            ),
        ):
            for index in range(1300):
                timestamp_ns = round(index * 1_000_000_000 / 130.0)
                runtime.accept_body(
                    self.player.box,
                    corroboration_box=self.player.box,
                    track_generation=1,
                    source_timestamp_ns=timestamp_ns,
                )
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=timestamp_ns,
                )

        self.assertGreaterEqual(len(worker.submissions), 899)
        self.assertLessEqual(len(worker.submissions), 901)

    def test_long_range_flow_fallback_tracks_exact_upper_body_roi(self) -> None:
        worker = _FakeWorker()
        phase_advancer = _ScriptedPhaseAdvancer(
            ((133.0, 107.0), (136.0, 107.0))
        )
        runtime = _AutomaticHeadRuntime(
            worker,
            stale_after_seconds=0.2,
            phase_advancer=phase_advancer,
        )
        source_box = (100.0, 100.0, 180.0, 176.0)
        current_box = (103.0, 100.0, 183.0, 176.0)
        next_box = (106.0, 100.0, 186.0, 176.0)
        head_box = (125.0, 102.0, 135.0, 112.0)

        for timestamp_ns, body_box in (
            (100_000_000, source_box),
            (110_000_000, current_box),
        ):
            runtime.accept_body(
                body_box,
                aim_box=body_box,
                corroboration_box=body_box,
                track_generation=7,
                source_timestamp_ns=timestamp_ns,
            )
            self.assertTrue(
                runtime.remember_frame(
                    self.frame,
                    source_timestamp_ns=timestamp_ns,
                )
            )
        worker.result = _result(
            source_ns=100_000_000,
            point=(130.0, 107.0),
            selected_player_box=source_box,
            head_box=head_box,
        )
        self.assertIsNotNone(runtime.take_latest(now_ns=110_000_001))

        runtime.accept_body(
            next_box,
            aim_box=next_box,
            corroboration_box=next_box,
            track_generation=7,
            source_timestamp_ns=120_000_000,
        )
        self.assertTrue(
            runtime.remember_frame(
                self.frame,
                source_timestamp_ns=120_000_000,
            )
        )

        self.assertEqual(
            phase_advancer.feature_boxes,
            [
                (108.0, 100.76, 172.0, 134.2),
                (111.0, 100.76, 175.0, 134.2),
            ],
        )
        self.assertEqual(
            runtime._flow_feature_box,
            (114.0, 100.76, 178.0, 134.2),
        )
        self.assertIsNone(
            runtime._flow_feature_box_for_player(
                self.player.box,
                (180.0, 120.0, 220.0, 180.0),
            )
        )

    def test_anchored_maintenance_survives_single_lk_loss_without_gpu_burst(
        self,
    ) -> None:
        self.assertEqual(AUTOMATIC_HEAD_TRACKING_LOCALIZATION_HZ, 24.0)
        worker = _FakeWorker()
        phase_advancer = _ScriptedPhaseAdvancer(
            (
                (200.0, 150.0),
                (203.0, 150.0),
                (206.0, 150.0),
                (209.0, 150.0),
                (212.0, 150.0),
                None,
                (215.0, 150.0),
                (218.0, 150.0),
                (221.0, 150.0),
            )
        )
        runtime = _AutomaticHeadRuntime(
            worker,
            submission_hz=90.0,
            tracking_submission_hz=24.0,
            stale_after_seconds=0.2,
            phase_advancer=phase_advancer,
        )
        body = self.player.box
        head_box = (180.0, 120.0, 220.0, 180.0)

        def accept_and_remember(timestamp_ns: int) -> None:
            self.assertFalse(
                runtime.accept_body(
                    body,
                    aim_box=body,
                    corroboration_box=body,
                    track_generation=3,
                    source_timestamp_ns=timestamp_ns,
                )
            )
            self.assertTrue(
                runtime.remember_frame(
                    self.frame,
                    source_timestamp_ns=timestamp_ns,
                )
            )

        with (
            mock.patch(
                "detection.head_detector.plan_head_crop",
                return_value=object(),
            ),
            mock.patch(
                "detection.head_detector.prepare_head_input",
                return_value=object(),
            ),
        ):
            source_ns = 100_000_000
            accept_and_remember(source_ns)
            self.assertTrue(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=source_ns,
                )
            )
            worker.result = _result(
                source_ns=source_ns,
                point=(200.0, 150.0),
                selected_player_box=body,
                head_box=head_box,
            )
            self.assertIsNotNone(runtime.take_latest(now_ns=source_ns + 1))

            # A measured direct anchor immediately moves the shared-GPU model
            # to its achievable maintenance cadence. Pixel flow may still earn
            # controller authority independently, but it no longer controls
            # how aggressively the 640 model contends with the primary model.
            for offset_ms, expected_submission in (
                (10, False),
                (20, False),
                (30, False),
            ):
                timestamp_ns = source_ns + offset_ms * 1_000_000
                accept_and_remember(timestamp_ns)
                self.assertEqual(
                    runtime.submit(
                        self.frame,
                        self.player,
                        source_timestamp_ns=timestamp_ns,
                    ),
                    expected_submission,
                )
            maintenance_ns = source_ns + 62_000_000
            accept_and_remember(maintenance_ns)
            self.assertTrue(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=maintenance_ns,
                )
            )
            self.assertEqual(len(worker.submissions), 2)

            # A failed LK step immediately revokes pixel-flow authority, but
            # one recovered frame does not trigger a futile 90 Hz submission
            # burst on a worker that can complete only at maintenance rate.
            rejected_ns = source_ns + 70_000_000
            accept_and_remember(rejected_ns)
            self.assertFalse(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=rejected_ns,
                )
            )
            recovery_ns = source_ns + 74_000_000
            accept_and_remember(recovery_ns)
            self.assertFalse(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=recovery_ns,
                )
            )
            self.assertEqual(len(worker.submissions), 2)
            self.assertEqual(runtime._tracking_flow_failure_streak, 0)
            self.assertFalse(runtime._tracking_cadence_requires_direct_refresh)

            correction_ns = source_ns + 104_000_000
            accept_and_remember(correction_ns)
            self.assertTrue(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=correction_ns,
                )
            )
            worker.result = _result(
                source_ns=correction_ns,
                point=(215.0, 150.0),
                selected_player_box=body,
                head_box=(195.0, 120.0, 235.0, 180.0),
            )
            self.assertIsNotNone(
                runtime.take_latest(now_ns=correction_ns + 1_000_000)
            )
            resumed_ns = source_ns + 114_000_000
            accept_and_remember(resumed_ns)
            self.assertFalse(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=resumed_ns,
                )
            )

    def test_stale_maintenance_result_pulls_next_submission_forward(self) -> None:
        worker = _FakeWorker()
        runtime = _AutomaticHeadRuntime(
            worker,
            submission_hz=90.0,
            tracking_submission_hz=24.0,
            stale_after_seconds=0.110,
            phase_advancer=_ScriptedPhaseAdvancer(((200.0, 150.0),) * 16),
        )

        with (
            mock.patch(
                "detection.head_detector.plan_head_crop",
                return_value=object(),
            ),
            mock.patch(
                "detection.head_detector.prepare_head_input",
                return_value=object(),
            ),
        ):
            source_ns = self._seed_adaptive_maintenance_cadence(worker, runtime)
            for offset_ms in (62, 104):
                timestamp_ns = source_ns + offset_ms * 1_000_000
                self._accept_and_remember_adaptive_frame(runtime, timestamp_ns)
                self.assertTrue(
                    runtime.submit(
                        self.frame,
                        self.player,
                        source_timestamp_ns=timestamp_ns,
                    )
                )

            # The 120 ms source was an accepted acquisition submission. A
            # delayed completion for it is now just beyond the 110 ms freshness
            # limit, while the existing 24 Hz deadline is still in the future.
            probe_ns = source_ns + 132_000_000
            self._accept_and_remember_adaptive_frame(runtime, probe_ns)
            worker.result = _result(
                source_ns=source_ns + 20_000_000,
                point=(200.0, 150.0),
                selected_player_box=self.player.box,
                head_box=(180.0, 120.0, 220.0, 180.0),
            )
            self.assertIsNone(runtime.take_latest(now_ns=probe_ns + 1))
            self.assertTrue(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=probe_ns,
                )
            )

    def test_three_consecutive_no_head_results_restore_acquisition_cadence(
        self,
    ) -> None:
        self.assertEqual(AUTOMATIC_HEAD_TRACKING_MAX_CONSECUTIVE_MISSES, 3)
        worker = _FakeWorker()
        runtime = _AutomaticHeadRuntime(
            worker,
            submission_hz=90.0,
            tracking_submission_hz=24.0,
            stale_after_seconds=0.2,
            phase_advancer=_ScriptedPhaseAdvancer(((200.0, 150.0),) * 20),
        )

        with (
            mock.patch(
                "detection.head_detector.plan_head_crop",
                return_value=object(),
            ),
            mock.patch(
                "detection.head_detector.prepare_head_input",
                return_value=object(),
            ),
        ):
            source_ns = self._seed_adaptive_maintenance_cadence(worker, runtime)
            for maintenance_ms, early_probe_ms in (
                (62, 74),
                (104, 116),
            ):
                maintenance_ns = source_ns + maintenance_ms * 1_000_000
                self._accept_and_remember_adaptive_frame(runtime, maintenance_ns)
                self.assertTrue(
                    runtime.submit(
                        self.frame,
                        self.player,
                        source_timestamp_ns=maintenance_ns,
                    )
                )
                worker.result = _result(
                    source_ns=maintenance_ns,
                    point=None,
                    selected_player_box=self.player.box,
                )
                self.assertIsNone(
                    runtime.take_latest(now_ns=maintenance_ns + 1)
                )
                early_probe_ns = source_ns + early_probe_ms * 1_000_000
                self._accept_and_remember_adaptive_frame(runtime, early_probe_ns)
                self.assertFalse(
                    runtime.submit(
                        self.frame,
                        self.player,
                        source_timestamp_ns=early_probe_ns,
                    )
                )

            third_maintenance_ns = source_ns + 146_000_000
            self._accept_and_remember_adaptive_frame(
                runtime,
                third_maintenance_ns,
            )
            self.assertTrue(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=third_maintenance_ns,
                )
            )
            worker.result = _result(
                source_ns=third_maintenance_ns,
                point=None,
                selected_player_box=self.player.box,
            )
            self.assertIsNone(
                runtime.take_latest(now_ns=third_maintenance_ns + 1)
            )
            recovery_ns = source_ns + 158_000_000
            self._accept_and_remember_adaptive_frame(runtime, recovery_ns)
            self.assertTrue(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=recovery_ns,
                )
            )

    def test_body_fallback_proof_accepts_only_fresh_no_decoded_outcome(self) -> None:
        first_ns = 100_000_000
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=4,
            source_timestamp_ns=first_ns,
        )
        self.worker.result = _result(
            source_ns=first_ns,
            point=None,
            selected_player_box=self.player.box,
            localization_reason=(
                HeadLocalizationReason.NO_DECODED_HEAD_CANDIDATE
            ),
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=first_ns + 1))
        self.assertTrue(
            self.runtime.body_fallback_no_decoded_verified(
                now_ns=first_ns + 1,
            )
        )
        self.assertEqual(
            self.runtime.body_fallback_no_decoded_deadline_ns(
                now_ns=first_ns + 1,
            ),
            first_ns + self.runtime.stale_after_ns,
        )

        ambiguous_ns = 120_000_000
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=4,
            source_timestamp_ns=ambiguous_ns,
        )
        self.worker.result = _result(
            source_ns=ambiguous_ns,
            point=None,
            selected_player_box=self.player.box,
            localization_reason=HeadLocalizationReason.MULTIPLE_PLAUSIBLE_HEADS,
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=ambiguous_ns + 1))
        self.assertFalse(
            self.runtime.body_fallback_no_decoded_verified(
                now_ns=ambiguous_ns + 1,
            )
        )

        decoded_miss_ns = 140_000_000
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=4,
            source_timestamp_ns=decoded_miss_ns,
        )
        self.worker.result = _result(
            source_ns=decoded_miss_ns,
            point=None,
            selected_player_box=self.player.box,
            localization_reason=(
                HeadLocalizationReason.NO_DECODED_HEAD_CANDIDATE
            ),
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=decoded_miss_ns + 1))
        self.assertFalse(
            self.runtime.body_fallback_no_decoded_verified(
                now_ns=(
                    decoded_miss_ns + self.runtime.stale_after_ns + 1
                ),
            )
        )
        self.assertIsNone(
            self.runtime.body_fallback_no_decoded_deadline_ns(
                now_ns=(
                    decoded_miss_ns + self.runtime.stale_after_ns + 1
                ),
            )
        )

    def test_lease_margin_pulls_maintenance_deadline_back_to_90_hz(self) -> None:
        self.assertEqual(
            AUTOMATIC_HEAD_TRACKING_MINIMUM_LEASE_REMAINING_SECONDS,
            0.300,
        )
        worker = _FakeWorker()
        runtime = _AutomaticHeadRuntime(
            worker,
            submission_hz=90.0,
            tracking_submission_hz=24.0,
            stale_after_seconds=0.2,
            phase_advancer=_ScriptedPhaseAdvancer(((200.0, 150.0),) * 64),
        )

        with (
            mock.patch(
                "detection.head_detector.plan_head_crop",
                return_value=object(),
            ),
            mock.patch(
                "detection.head_detector.prepare_head_input",
                return_value=object(),
            ),
        ):
            source_ns = self._seed_adaptive_maintenance_cadence(worker, runtime)
            for offset_ms in range(40, 441, 10):
                timestamp_ns = source_ns + offset_ms * 1_000_000
                self._accept_and_remember_adaptive_frame(runtime, timestamp_ns)
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=timestamp_ns,
                )

            # The direct anchor expires at 850 ms. Exactly 300 ms remains here,
            # so the strict lease-margin comparison retains maintenance cadence.
            boundary_ns = source_ns + 450_000_000
            self._accept_and_remember_adaptive_frame(runtime, boundary_ns)
            self.assertFalse(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=boundary_ns,
                )
            )

            # Two milliseconds later the lease margin has been crossed. The
            # scheduler must pull the 24 Hz deadline forward to last+11.1 ms.
            recovery_ns = source_ns + 452_000_000
            self._accept_and_remember_adaptive_frame(runtime, recovery_ns)
            self.assertTrue(
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=recovery_ns,
                )
            )

    def test_submission_requires_an_explicit_same_frame_player(self) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=4,
            source_timestamp_ns=100_000_000,
        )
        with (
            mock.patch(
                "detection.head_detector.plan_head_crop",
                return_value=object(),
            ),
            mock.patch(
                "detection.head_detector.prepare_head_input",
                return_value=object(),
            ),
        ):
            with self.assertRaises(TypeError):
                self.runtime.submit(
                    self.frame,
                    source_timestamp_ns=120_000_000,
                )
            self.runtime.accept_body(
                self.player.box,
                corroboration_box=None,
                track_generation=4,
                source_timestamp_ns=120_000_000,
            )
            with self.assertRaises(TypeError):
                self.runtime.submit(
                    self.frame,
                    source_timestamp_ns=140_000_000,
                )

        self.assertEqual(self.worker.submissions, [])
        self.assertTrue(
            self.runtime.consume_motion_corroboration_revocation()
        )

    def test_direct_head_without_exact_measured_primary_binding_is_rejected(self) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=None,
            track_generation=6,
            source_timestamp_ns=108_000_000,
        )
        self.runtime.consume_motion_corroboration_revocation()
        self.worker.result = _result(
            source_ns=108_000_000,
            point=(200.0, 150.0),
        )

        self.assertIsNone(self.runtime.take_latest(now_ns=118_000_000))
        self.assertIsNone(self.runtime.visible_sample(now_ns=118_000_000))
        self.assertEqual(self.runtime.identity_generation, 1)

    def test_direct_absolute_point_and_original_timestamp_are_preserved(self) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(source_ns=100_000_000)

        sample = self.runtime.take_latest(now_ns=120_000_000)

        assert sample is not None
        self.assertEqual(sample.point, (321.0, 123.0))
        self.assertEqual(sample.source_timestamp_ns, 100_000_000)
        self.assertIs(sample.provenance, DirectHeadProvenance.DIRECT)
        self.assertFalse(sample.body_derived_motion_permitted)
        self.assertIsNone(sample.body_derived_motion_deadline_ns)
        self.assertEqual(sample.corroboration_point, (200.0, 400.0))
        # This is intentionally unrelated to the body-box ratio proxy.
        self.assertNotEqual(sample.point, (200.0, 172.0))

    def test_delayed_direct_box_seeds_current_pixel_track_without_body_remap(
        self,
    ) -> None:
        source_ns = 100_000_000
        current_ns = 116_000_000
        source_box = self.player.box
        translated_box = tuple(
            value + offset
            for value, offset in zip(
                source_box,
                (12.0, 5.0, 12.0, 5.0),
            )
        )
        head_box = (150.0, 120.0, 210.0, 190.0)
        head_point = (180.0, 155.0)
        rng = np.random.default_rng(81)
        source_frame = rng.integers(
            0,
            255,
            size=(1080, 1920, 3),
            dtype=np.uint8,
        )
        transform = np.asarray(
            ((1.0, 0.0, 12.0), (0.0, 1.0, 5.0)),
            dtype=np.float32,
        )
        current_frame = cv2.warpAffine(
            source_frame,
            transform,
            (source_frame.shape[1], source_frame.shape[0]),
            borderMode=cv2.BORDER_REFLECT101,
        )

        self.runtime.accept_body(
            source_box,
            aim_box=source_box,
            corroboration_box=source_box,
            track_generation=9,
            source_timestamp_ns=source_ns,
        )
        self.assertTrue(
            self.runtime.remember_frame(
                source_frame,
                source_timestamp_ns=source_ns,
            )
        )
        self.runtime.accept_body(
            translated_box,
            aim_box=translated_box,
            corroboration_box=translated_box,
            track_generation=9,
            source_timestamp_ns=current_ns,
        )
        self.assertTrue(
            self.runtime.remember_frame(
                current_frame,
                source_timestamp_ns=current_ns,
            )
        )
        self.worker.result = _result(
            source_ns=source_ns,
            point=head_point,
            selected_player_box=source_box,
            head_box=head_box,
        )

        direct = self.runtime.take_latest(now_ns=current_ns + 1_000_000)
        visible = self.runtime.visible_sample(now_ns=current_ns + 1_000_000)

        assert direct is not None
        assert visible is not None
        expected = (192.0, 160.0)
        self.assertAlmostEqual(direct.point[0], expected[0], delta=0.5)
        self.assertAlmostEqual(direct.point[1], expected[1], delta=0.5)
        self.assertEqual(direct.source_timestamp_ns, current_ns)
        self.assertEqual(direct.direct_source_timestamp_ns, source_ns)
        self.assertTrue(direct.phase_advanced)
        self.assertAlmostEqual(visible.point[0], expected[0], delta=0.5)
        self.assertAlmostEqual(visible.point[1], expected[1], delta=0.5)
        self.assertEqual(visible.velocity_point, visible.point)
        self.assertFalse(visible.body_derived_motion_permitted)
        self.assertIsNone(visible.body_derived_motion_deadline_ns)
        self.assertEqual(visible.corroboration_point, (212.0, 405.0))
        self.assertTrue(visible.phase_advanced)

        # The phase-corrected seed becomes a live one-hop tracker.  A newer
        # frame therefore stays on observed head pixels even with no new model
        # result; current body geometry remains identity/corroboration only.
        next_ns = 124_000_000
        next_box = tuple(
            value + offset
            for value, offset in zip(
                source_box,
                (20.0, 8.0, 20.0, 8.0),
            )
        )
        next_transform = np.asarray(
            ((1.0, 0.0, 20.0), (0.0, 1.0, 8.0)),
            dtype=np.float32,
        )
        next_frame = cv2.warpAffine(
            source_frame,
            next_transform,
            (source_frame.shape[1], source_frame.shape[0]),
            borderMode=cv2.BORDER_REFLECT101,
        )
        self.runtime.accept_body(
            next_box,
            aim_box=next_box,
            corroboration_box=next_box,
            track_generation=9,
            source_timestamp_ns=next_ns,
        )
        self.assertTrue(
            self.runtime.remember_frame(
                next_frame,
                source_timestamp_ns=next_ns,
            )
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=next_ns + 1_000_000))
        continued = self.runtime.visible_sample(now_ns=next_ns + 1_000_000)
        assert continued is not None
        self.assertAlmostEqual(continued.point[0], 200.0, delta=0.7)
        self.assertAlmostEqual(continued.point[1], 163.0, delta=0.7)
        self.assertEqual(continued.velocity_point, continued.point)
        self.assertFalse(continued.body_derived_motion_permitted)
        self.assertEqual(continued.corroboration_point, (220.0, 408.0))
        self.assertTrue(continued.phase_advanced)

    def test_flow_rejection_carries_one_coordinate_without_claiming_pixels(
        self,
    ) -> None:
        worker = _FakeWorker()
        phase_advancer = _ScriptedPhaseAdvancer(
            (
                (210.0, 150.0),
                None,
                (210.0, 150.0),
                None,
                None,
                None,
            )
        )
        runtime = _AutomaticHeadRuntime(
            worker,
            stale_after_seconds=0.2,
            phase_advancer=phase_advancer,
        )
        body_box = self.player.box
        head_box = (180.0, 120.0, 220.0, 180.0)
        source_ns = 100_000_000
        seed_ns = 110_000_000

        for timestamp_ns in (source_ns, seed_ns):
            runtime.accept_body(
                body_box,
                aim_box=body_box,
                corroboration_box=body_box,
                track_generation=3,
                source_timestamp_ns=timestamp_ns,
            )
            self.assertTrue(
                runtime.remember_frame(
                    self.frame,
                    source_timestamp_ns=timestamp_ns,
                )
            )
        worker.result = _result(
            source_ns=source_ns,
            point=(200.0, 150.0),
            selected_player_box=body_box,
            head_box=head_box,
        )
        self.assertIsNotNone(runtime.take_latest(now_ns=seed_ns + 1))
        seeded = runtime.visible_sample(now_ns=seed_ns + 1)
        assert seeded is not None
        self.assertEqual(seeded.point, (210.0, 150.0))
        self.assertTrue(seeded.phase_advanced)
        # Phase correction must not rewrite the exact-source anchor.
        self.assertEqual(runtime.anchor.normalized_point, (0.5, 1.0 / 12.0))

        rejected_ns = 120_000_000
        runtime.accept_body(
            body_box,
            aim_box=body_box,
            corroboration_box=body_box,
            track_generation=3,
            source_timestamp_ns=rejected_ns,
        )
        runtime.remember_frame(self.frame, source_timestamp_ns=rejected_ns)
        self.assertIsNone(runtime.take_latest(now_ns=rejected_ns + 1))
        carried = runtime.visible_sample(now_ns=rejected_ns + 1)
        assert carried is not None
        fallback_alpha = 1.0 - math.exp(
            -0.010 / AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS
        )
        expected_carried = (
            seeded.point[0]
            + fallback_alpha * (200.0 - seeded.point[0]),
            seeded.point[1]
            + fallback_alpha * (150.0 - seeded.point[1]),
        )
        self.assertAlmostEqual(carried.point[0], expected_carried[0])
        self.assertAlmostEqual(carried.point[1], expected_carried[1])
        self.assertEqual(carried.point, runtime._mapped_filter_point)
        self.assertEqual(runtime._flow_point, seeded.point)
        self.assertTrue(runtime._flow_coordinate_current)
        self.assertFalse(runtime._flow_pixel_observed_current)
        self.assertFalse(carried.phase_advanced)
        self.assertTrue(carried.body_derived_motion_permitted)
        self.assertIsNone(carried.corroboration_point)
        self.assertEqual(runtime._tracking_flow_failure_streak, 1)

        recovered_ns = 130_000_000
        runtime.accept_body(
            body_box,
            aim_box=body_box,
            corroboration_box=body_box,
            track_generation=3,
            source_timestamp_ns=recovered_ns,
        )
        runtime.remember_frame(self.frame, source_timestamp_ns=recovered_ns)
        self.assertIsNone(runtime.take_latest(now_ns=recovered_ns + 1))
        recovered = runtime.visible_sample(now_ns=recovered_ns + 1)
        assert recovered is not None
        self.assertEqual(recovered.point, seeded.point)
        self.assertEqual(runtime._mapped_filter_point, recovered.point)
        self.assertTrue(recovered.phase_advanced)
        self.assertFalse(recovered.body_derived_motion_permitted)
        self.assertEqual(runtime._tracking_flow_failure_streak, 0)

        # Repeated failures cannot turn the short private continuity bridge
        # into raw body translation for the whole 750 ms direct-head lease.
        # Both retained rejections publish the mapped-position LP immediately;
        # the third reaches the explicit cap and clears even the private seed.
        self.assertEqual(AUTOMATIC_HEAD_FLOW_MAX_CONSECUTIVE_FAILURES, 2)
        failure_samples = []
        private_flow_points = []
        failure_streaks = []
        flow_current = []
        for timestamp_ns in (140_000_000, 150_000_000, 160_000_000):
            runtime.accept_body(
                body_box,
                aim_box=body_box,
                corroboration_box=body_box,
                track_generation=3,
                source_timestamp_ns=timestamp_ns,
            )
            runtime.remember_frame(self.frame, source_timestamp_ns=timestamp_ns)
            self.assertIsNone(runtime.take_latest(now_ns=timestamp_ns + 1))
            sample = runtime.visible_sample(now_ns=timestamp_ns + 1)
            assert sample is not None
            failure_samples.append(sample)
            private_flow_points.append(runtime._flow_point)
            failure_streaks.append(runtime._tracking_flow_failure_streak)
            flow_current.append(runtime._flow_coordinate_current)

        expected_point = recovered.point
        for sample in failure_samples:
            expected_point = (
                expected_point[0]
                + fallback_alpha * (200.0 - expected_point[0]),
                expected_point[1]
                + fallback_alpha * (150.0 - expected_point[1]),
            )
            self.assertAlmostEqual(sample.point[0], expected_point[0])
            self.assertAlmostEqual(sample.point[1], expected_point[1])
            self.assertTrue(sample.body_derived_motion_permitted)
            self.assertFalse(sample.phase_advanced)
        self.assertEqual(private_flow_points[:2], [recovered.point] * 2)
        self.assertEqual(failure_streaks[:2], [1, 2])
        self.assertEqual(flow_current[:2], [True, True])
        self.assertIsNone(private_flow_points[2])
        self.assertEqual(failure_streaks[2], 0)
        self.assertFalse(flow_current[2])
        self.assertIsNone(runtime._flow_point)

    def test_failed_flow_body_jitter_publishes_mapped_filter_and_clears_bounded(
        self,
    ) -> None:
        worker = _FakeWorker()
        runtime = _AutomaticHeadRuntime(
            worker,
            stale_after_seconds=0.2,
            phase_advancer=_ScriptedPhaseAdvancer(
                ((210.0, 150.0), None, None, None)
            ),
        )
        body_box = self.player.box
        head_box = (180.0, 120.0, 220.0, 180.0)
        source_ns = 100_000_000
        seed_ns = 110_000_000

        for timestamp_ns in (source_ns, seed_ns):
            runtime.accept_body(
                body_box,
                aim_box=body_box,
                corroboration_box=body_box,
                track_generation=3,
                source_timestamp_ns=timestamp_ns,
            )
            self.assertTrue(
                runtime.remember_frame(
                    self.frame,
                    source_timestamp_ns=timestamp_ns,
                )
            )
        worker.result = _result(
            source_ns=source_ns,
            point=(200.0, 150.0),
            selected_player_box=body_box,
            head_box=head_box,
        )
        self.assertIsNotNone(runtime.take_latest(now_ns=seed_ns + 1))
        seeded = runtime.visible_sample(now_ns=seed_ns + 1)
        assert seeded is not None
        self.assertEqual(seeded.point, (210.0, 150.0))
        self.assertTrue(seeded.phase_advanced)

        def jittered_box(
            center_dx: float,
            center_dy: float,
            width_delta: float,
            height_delta: float,
        ) -> tuple[float, float, float, float]:
            center_x = (body_box[0] + body_box[2]) * 0.5 + center_dx
            center_y = (body_box[1] + body_box[3]) * 0.5 + center_dy
            width = body_box[2] - body_box[0] + width_delta
            height = body_box[3] - body_box[1] + height_delta
            return (
                center_x - width * 0.5,
                center_y - height * 0.5,
                center_x + width * 0.5,
                center_y + height * 0.5,
            )

        # Alternate both measured center and shape. The retained LK seed sees
        # center translation only, while the immutable normalized anchor maps
        # through the full current body geometry; these paths intentionally
        # disagree so publishing the raw private seed is observable.
        positive_jitter = jittered_box(6.0, 4.0, 8.0, -12.0)
        negative_jitter = jittered_box(-6.0, -4.0, -8.0, 12.0)
        boxes = (positive_jitter, negative_jitter, positive_jitter)
        timestamps = (120_000_000, 130_000_000, 140_000_000)
        fallback_alpha = 1.0 - math.exp(
            -0.010 / AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS
        )
        expected_public = seeded.point
        previous_public = seeded.point
        previous_input = (200.0, 150.0)
        maximum_input_step = (
            AUTOMATIC_HEAD_MAPPED_STEP_ALLOWANCE_PIXELS
            + AUTOMATIC_HEAD_MAPPED_MAX_SPEED_PIXELS_PER_SECOND * 0.010
        )
        maximum_public_step = fallback_alpha * maximum_input_step

        for index, (timestamp_ns, current_box) in enumerate(
            zip(timestamps, boxes, strict=True)
        ):
            runtime.accept_body(
                current_box,
                aim_box=current_box,
                corroboration_box=current_box,
                track_generation=3,
                source_timestamp_ns=timestamp_ns,
            )
            self.assertTrue(
                runtime.remember_frame(
                    self.frame,
                    source_timestamp_ns=timestamp_ns,
                )
            )
            self.assertIsNone(runtime.take_latest(now_ns=timestamp_ns + 1))
            sample = runtime.visible_sample(now_ns=timestamp_ns + 1)
            assert sample is not None

            mapped_input = (
                (current_box[0] + current_box[2]) * 0.5,
                current_box[1]
                + (current_box[3] - current_box[1]) / 12.0,
            )
            input_dx = mapped_input[0] - previous_input[0]
            input_dy = mapped_input[1] - previous_input[1]
            input_distance = math.hypot(input_dx, input_dy)
            if input_distance > maximum_input_step:
                input_scale = maximum_input_step / input_distance
                mapped_input = (
                    previous_input[0] + input_dx * input_scale,
                    previous_input[1] + input_dy * input_scale,
                )
            expected_public = (
                expected_public[0]
                + fallback_alpha * (mapped_input[0] - expected_public[0]),
                expected_public[1]
                + fallback_alpha * (mapped_input[1] - expected_public[1]),
            )
            self.assertAlmostEqual(sample.point[0], expected_public[0])
            self.assertAlmostEqual(sample.point[1], expected_public[1])
            self.assertEqual(sample.point, runtime._mapped_filter_point)
            self.assertFalse(sample.phase_advanced)
            self.assertTrue(sample.body_derived_motion_permitted)
            self.assertLessEqual(
                math.dist(sample.point, previous_public),
                maximum_public_step + 1e-9,
            )

            if index < AUTOMATIC_HEAD_FLOW_MAX_CONSECUTIVE_FAILURES:
                self.assertIsNotNone(runtime._flow_point)
                assert runtime._flow_point is not None
                self.assertNotEqual(sample.point, runtime._flow_point)
                self.assertTrue(runtime._flow_coordinate_current)
                self.assertFalse(runtime._flow_pixel_observed_current)
                self.assertEqual(runtime._tracking_flow_failure_streak, index + 1)
            else:
                # The third failure clears private flow without a public snap.
                self.assertIsNone(runtime._flow_point)
                self.assertFalse(runtime._flow_coordinate_current)
                self.assertEqual(runtime._tracking_flow_failure_streak, 0)
                self.assertLessEqual(
                    math.dist(sample.point, previous_public),
                    maximum_public_step + 1e-9,
                )
            previous_public = sample.point
            previous_input = mapped_input

    def test_newest_capture_phase_is_one_step_position_only_and_monotonic(
        self,
    ) -> None:
        worker = _FakeWorker()
        phase_advancer = _ScriptedPhaseAdvancer(
            ((210.0, 150.0), (220.0, 150.0), (230.0, 150.0))
        )
        runtime = _AutomaticHeadRuntime(
            worker,
            stale_after_seconds=0.2,
            phase_advancer=phase_advancer,
        )
        source_box = self.player.box
        source_ns = 100_000_000
        measured_ns = 110_000_000
        measured_box = tuple(
            value + offset
            for value, offset in zip(
                source_box,
                (10.0, 0.0, 10.0, 0.0),
            )
        )
        runtime.accept_body(
            source_box,
            aim_box=source_box,
            corroboration_box=source_box,
            track_generation=4,
            source_timestamp_ns=source_ns,
        )
        self.assertTrue(
            runtime.remember_frame(self.frame, source_timestamp_ns=source_ns)
        )
        runtime.accept_body(
            measured_box,
            aim_box=measured_box,
            corroboration_box=measured_box,
            track_generation=4,
            source_timestamp_ns=measured_ns,
        )
        self.assertTrue(
            runtime.remember_frame(self.frame, source_timestamp_ns=measured_ns)
        )
        worker.result = _result(
            source_ns=source_ns,
            point=(200.0, 150.0),
            selected_player_box=source_box,
            head_box=(180.0, 120.0, 220.0, 180.0),
        )
        self.assertIsNotNone(runtime.take_latest(now_ns=measured_ns + 1))

        peek_ns = 120_000_000
        self.assertTrue(
            runtime.remember_newer_capture_frame(
                self.frame,
                source_timestamp_ns=peek_ns,
            )
        )
        peeked = runtime.visible_sample(now_ns=peek_ns + 1)
        assert peeked is not None
        self.assertEqual(peeked.point, (220.0, 150.0))
        self.assertEqual(peeked.source_timestamp_ns, peek_ns)
        self.assertTrue(peeked.phase_advanced)
        self.assertFalse(peeked.body_derived_motion_permitted)
        self.assertIsNone(peeked.body_derived_motion_deadline_ns)
        self.assertIsNone(peeked.corroboration_point)

        # A second uninferred endpoint cannot chain from the first one.
        self.assertFalse(
            runtime.remember_newer_capture_frame(
                self.frame,
                source_timestamp_ns=peek_ns + 1_000_000,
            )
        )

        # When the peeked packet becomes the next inferred body input, its
        # already remembered pixels are validated without a zero-time LK step.
        confirmed_box = tuple(
            value + offset
            for value, offset in zip(
                source_box,
                (20.0, 0.0, 20.0, 0.0),
            )
        )
        runtime.accept_body(
            confirmed_box,
            aim_box=confirmed_box,
            corroboration_box=confirmed_box,
            track_generation=4,
            source_timestamp_ns=peek_ns,
        )
        self.assertTrue(
            runtime.remember_frame(self.frame, source_timestamp_ns=peek_ns)
        )
        confirmed = runtime.visible_sample(now_ns=peek_ns + 1)
        assert confirmed is not None
        self.assertEqual(confirmed.source_timestamp_ns, peek_ns)
        self.assertEqual(confirmed.point, peeked.point)

        next_peek_ns = 130_000_000
        self.assertTrue(
            runtime.remember_newer_capture_frame(
                self.frame,
                source_timestamp_ns=next_peek_ns,
            )
        )
        next_peeked = runtime.visible_sample(now_ns=next_peek_ns + 1)
        assert next_peeked is not None
        self.assertEqual(next_peeked.source_timestamp_ns, next_peek_ns)
        self.assertEqual(next_peeked.point, (230.0, 150.0))
        self.assertIsNone(next_peeked.corroboration_point)

    def test_newest_capture_phase_rejects_lead_beyond_one_inference(self) -> None:
        worker = _FakeWorker()
        phase_advancer = _ScriptedPhaseAdvancer(((210.0, 150.0),))
        runtime = _AutomaticHeadRuntime(
            worker,
            stale_after_seconds=0.2,
            phase_advancer=phase_advancer,
        )
        source_ns = 100_000_000
        runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=7,
            source_timestamp_ns=source_ns,
        )
        runtime.remember_frame(self.frame, source_timestamp_ns=source_ns)
        worker.result = _result(
            source_ns=source_ns,
            point=(200.0, 150.0),
            selected_player_box=self.player.box,
            head_box=(180.0, 120.0, 220.0, 180.0),
        )
        # A same-frame direct result has no pixel-qualified LK hop and cannot
        # authorize a newer-frame transient on its own.
        self.assertIsNotNone(runtime.take_latest(now_ns=source_ns + 1))
        self.assertFalse(
            runtime.remember_newer_capture_frame(
                self.frame,
                source_timestamp_ns=source_ns + 26_000_000,
            )
        )

    def test_continued_flow_uses_bounded_source_time_anchor_corridor(self) -> None:
        worker = _FakeWorker()
        phase_advancer = _ScriptedPhaseAdvancer(
            ((200.0, 150.0), *("step-right" for _ in range(12)))
        )
        runtime = _AutomaticHeadRuntime(
            worker,
            stale_after_seconds=0.2,
            phase_advancer=phase_advancer,
        )
        body_box = self.player.box
        source_ns = 100_000_000
        runtime.accept_body(
            body_box,
            aim_box=body_box,
            corroboration_box=body_box,
            track_generation=5,
            source_timestamp_ns=source_ns,
        )
        runtime.remember_frame(self.frame, source_timestamp_ns=source_ns)
        worker.result = _result(
            source_ns=source_ns,
            point=(200.0, 150.0),
            selected_player_box=body_box,
            head_box=(180.0, 120.0, 220.0, 180.0),
        )
        self.assertIsNotNone(runtime.take_latest(now_ns=source_ns + 1))

        visible_samples = []
        for index in range(1, 13):
            timestamp_ns = source_ns + index * 10_000_000
            runtime.accept_body(
                body_box,
                aim_box=body_box,
                corroboration_box=body_box,
                track_generation=5,
                source_timestamp_ns=timestamp_ns,
            )
            runtime.remember_frame(self.frame, source_timestamp_ns=timestamp_ns)
            self.assertIsNone(runtime.take_latest(now_ns=timestamp_ns + 1))
            sample = runtime.visible_sample(now_ns=timestamp_ns + 1)
            assert sample is not None
            visible_samples.append(sample)

        expected_corridor = runtime._flow_body_residual_tolerance(
            body_box,
            body_box,
            elapsed_ns=10_000_000,
        )
        self.assertEqual(expected_corridor, 21.0)
        self.assertLessEqual(
            max(abs(sample.point[0] - 200.0) for sample in visible_samples),
            expected_corridor,
        )
        self.assertLessEqual(
            expected_corridor,
            AUTOMATIC_HEAD_FLOW_MAX_DYNAMIC_RESIDUAL_PIXELS,
        )
        self.assertTrue(
            any(sample.body_derived_motion_permitted for sample in visible_samples)
        )

    def test_body_frames_and_late_direct_result_never_republish_new_direct_sample(
        self,
    ) -> None:
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=4,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(source_ns=100_000_000, point=(200.0, 150.0))
        first = self.runtime.take_latest(now_ns=105_000_000)
        assert first is not None

        for timestamp_ns in (130_000_000, 140_000_000):
            current_box = tuple(
                value + (timestamp_ns - 100_000_000) / 10_000_000
                for value in self.player.box
            )
            self.runtime.accept_body(
                current_box,
                aim_box=current_box,
                corroboration_box=current_box,
                track_generation=4,
                source_timestamp_ns=timestamp_ns,
            )
            if timestamp_ns == 140_000_000:
                # An old async result completes after source-130 was already
                # published. It may only revisit the anchor, never its time.
                self.worker.result = _result(
                    source_ns=100_000_000,
                    point=(201.0, 151.0),
                )
            self.assertIsNone(
                self.runtime.take_latest(now_ns=timestamp_ns + 1_000_000)
            )
            visible = self.runtime.visible_sample(
                now_ns=timestamp_ns + 1_000_000
            )
            assert visible is not None
            self.assertEqual(visible.source_timestamp_ns, timestamp_ns)
            self.assertEqual(visible.direct_source_timestamp_ns, 100_000_000)

        self.assertGreater(visible.source_timestamp_ns, first.source_timestamp_ns)

    def test_mapped_position_stays_stable_while_velocity_passes_translation(
        self,
    ) -> None:
        self.assertEqual(AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS, 0.012)
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=6,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(source_ns=100_000_000, point=(200.0, 150.0))
        seeded = self.runtime.take_latest(now_ns=100_000_000)
        assert seeded is not None

        moved = (112.0, 100.0, 312.0, 700.0)
        self.runtime.accept_body(
            moved,
            aim_box=moved,
            corroboration_box=moved,
            track_generation=6,
            source_timestamp_ns=112_000_000,
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=112_000_000))
        filtered = self.runtime.visible_sample(now_ns=112_000_000)

        assert filtered is not None
        position_alpha = 1.0 - math.exp(
            -0.012 / AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS
        )
        expected_position = (200.0 + position_alpha * 12.0, 150.0)
        self.assertAlmostEqual(filtered.point[0], expected_position[0])
        self.assertAlmostEqual(filtered.point[1], expected_position[1])
        self.assertEqual(self.runtime._mapped_filter_input_point, (212.0, 150.0))
        # The fallback position remains intentionally damped for lock stability;
        # its separately qualified velocity channel retains coherent body
        # translation for the bounded body-derived pursuit path.
        expected_velocity_point = (212.0, 150.0)
        self.assertEqual(filtered.velocity_point, expected_velocity_point)
        diagnostic = _aim_diagnostic_head_sample(filtered)
        assert diagnostic is not None
        self.assertEqual(diagnostic["point"], list(filtered.point))
        self.assertEqual(
            diagnostic["velocity_point"],
            list(expected_velocity_point),
        )
        self.assertEqual(filtered.source_timestamp_ns, 112_000_000)
        self.assertTrue(filtered.body_derived_motion_permitted)
        self.assertEqual(
            filtered.body_derived_motion_deadline_ns,
            filtered.identity_deadline_ns,
        )
        self.assertEqual(filtered.identity_deadline_ns, 850_000_000)
        self.assertIsNone(filtered.corroboration_point)
        self.assertFalse(
            self.runtime.consume_motion_corroboration_revocation()
        )

    def test_local_head_refresh_is_smoothed_without_delaying_body_translation(
        self,
    ) -> None:
        body = self.player.box
        first_ns = 100_000_000
        second_ns = 112_000_000
        self.runtime.accept_body(
            body,
            aim_box=body,
            corroboration_box=body,
            track_generation=6,
            source_timestamp_ns=first_ns,
        )
        self.worker.result = _result(source_ns=first_ns, point=(200.0, 150.0))
        self.assertIsNotNone(self.runtime.take_latest(now_ns=first_ns))

        self.runtime.accept_body(
            body,
            aim_box=body,
            corroboration_box=body,
            track_generation=6,
            source_timestamp_ns=second_ns,
        )
        self.worker.result = _result(source_ns=second_ns, point=(220.0, 180.0))
        self.assertIsNotNone(self.runtime.take_latest(now_ns=second_ns))
        visible = self.runtime.visible_sample(now_ns=second_ns)
        assert visible is not None

        # The anchor's two-sample median moves from (200,150) to (210,165).
        # Only that local offset receives the 60 ms LP. The ordinary mapped
        # position applies its 12 ms LP only to the remaining local change,
        # while the velocity channel reconciles that local change at 30 ms
        # before its own 12 ms LP.
        anchor_alpha = 1.0 - math.exp(
            -0.012
            / AUTOMATIC_HEAD_NORMALIZED_ANCHOR_FILTER_TIME_CONSTANT_SECONDS
        )
        stabilized = (
            200.0 + anchor_alpha * 10.0,
            150.0 + anchor_alpha * 15.0,
        )
        mapped_alpha = 1.0 - math.exp(
            -0.012 / AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS
        )
        expected_position = (
            200.0 + mapped_alpha * (stabilized[0] - 200.0),
            150.0 + mapped_alpha * (stabilized[1] - 150.0),
        )
        reconcile_alpha = 1.0 - math.exp(
            -0.012
            / AUTOMATIC_HEAD_VELOCITY_RECONCILIATION_TIME_CONSTANT_SECONDS
        )
        reconciled = (
            200.0 + reconcile_alpha * (stabilized[0] - 200.0),
            150.0 + reconcile_alpha * (stabilized[1] - 150.0),
        )
        expected_velocity = (
            200.0 + mapped_alpha * (reconciled[0] - 200.0),
            150.0 + mapped_alpha * (reconciled[1] - 150.0),
        )
        self.assertAlmostEqual(visible.point[0], expected_position[0])
        self.assertAlmostEqual(visible.point[1], expected_position[1])
        self.assertAlmostEqual(visible.velocity_point[0], expected_velocity[0])
        self.assertAlmostEqual(visible.velocity_point[1], expected_velocity[1])
        self.assertLess(
            math.dist(visible.point, (200.0, 150.0)),
            math.dist(stabilized, (200.0, 150.0)),
        )
        self.assertLess(
            math.dist(visible.velocity_point, (200.0, 150.0)),
            math.dist(visible.point, (200.0, 150.0)),
        )

    def test_velocity_lp_duplicate_stale_and_prediction_boundaries(self) -> None:
        def sample(
            point: tuple[float, float],
            timestamp_ns: int,
            provenance: DirectHeadProvenance = (
                DirectHeadProvenance.MEASURED_PRIMARY
            ),
        ) -> DirectHeadAnchorSample:
            return DirectHeadAnchorSample(
                point=point,
                source_timestamp_ns=timestamp_ns,
                direct_source_timestamp_ns=100_000_000,
                identity_deadline_ns=300_000_000,
                track_generation=1,
                provenance=provenance,
                confidence=1.0,
                motion_corroboration_permitted=False,
            )

        first = self.runtime._filter_mapped_point(
            sample((200.0, 150.0), 100_000_000)
        )
        self.assertEqual(first, ((200.0, 150.0), (200.0, 150.0)))
        state_before_duplicate = (
            self.runtime._mapped_filter_point,
            self.runtime._mapped_filter_input_point,
            self.runtime._mapped_velocity_filter_point,
            self.runtime._mapped_filter_timestamp_ns,
        )
        duplicate = self.runtime._filter_mapped_point(
            sample((240.0, 170.0), 100_000_000)
        )
        self.assertEqual(duplicate, first)
        self.assertEqual(
            (
                self.runtime._mapped_filter_point,
                self.runtime._mapped_filter_input_point,
                self.runtime._mapped_velocity_filter_point,
                self.runtime._mapped_filter_timestamp_ns,
            ),
            state_before_duplicate,
        )

        # The test runtime's stale threshold is 65 ms. A longer source gap is
        # a discontinuity and seeds both channels directly, without a velocity
        # warm-up tail from the old target coordinate.
        stale = self.runtime._filter_mapped_point(
            sample((240.0, 170.0), 170_000_000)
        )
        self.assertEqual(stale, ((240.0, 170.0), (240.0, 170.0)))
        self.assertEqual(self.runtime._mapped_filter_input_point, (240.0, 170.0))

        predicted = self.runtime._filter_mapped_point(
            sample(
                (245.0, 175.0),
                180_000_000,
                DirectHeadProvenance.PREDICTED_PRIMARY,
            )
        )
        self.assertEqual(predicted, ((245.0, 175.0), (245.0, 175.0)))
        self.assertIsNone(self.runtime._mapped_filter_point)
        self.assertIsNone(self.runtime._mapped_filter_input_point)
        self.assertIsNone(self.runtime._mapped_velocity_filter_point)
        self.assertIsNone(self.runtime._mapped_filter_timestamp_ns)

    def test_shared_twelve_ms_channels_damp_four_ms_jitter_identically(
        self,
    ) -> None:
        def sample(point_x: float, timestamp_ns: int) -> DirectHeadAnchorSample:
            return DirectHeadAnchorSample(
                point=(point_x, 0.0),
                source_timestamp_ns=timestamp_ns,
                direct_source_timestamp_ns=0,
                identity_deadline_ns=1_000_000_000,
                track_generation=1,
                provenance=DirectHeadProvenance.MEASURED_PRIMARY,
                confidence=1.0,
                motion_corroboration_permitted=False,
            )

        position_excursions: list[float] = []
        velocity_excursions: list[float] = []
        qualified_excursions: list[float] = []
        for index in range(40):
            raw_x = 4.0 if index % 2 == 0 else -4.0
            position, velocity = self.runtime._filter_mapped_point(
                sample(raw_x, index * 4_000_000)
            )
            if index >= 20:
                position_excursions.append(abs(position[0]))
                velocity_excursions.append(abs(velocity[0]))
                assert self.runtime._mapped_filter_input_point is not None
                qualified_excursions.append(
                    abs(self.runtime._mapped_filter_input_point[0])
                )

        self.assertEqual(position_excursions, velocity_excursions)
        self.assertLess(max(position_excursions), max(qualified_excursions))

    def test_circular_body_jitter_is_damped_and_velocity_stays_bounded(self) -> None:
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=7,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(source_ns=100_000_000, point=(200.0, 150.0))
        self.assertIsNotNone(self.runtime.take_latest(now_ns=100_000_000))

        radii = []
        offsets = ((3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -3.0)) * 3
        last = None
        expected_position = (200.0, 150.0)
        position_alpha = 1.0 - math.exp(
            -0.008 / AUTOMATIC_HEAD_MAPPED_FILTER_TIME_CONSTANT_SECONDS
        )
        for index, (offset_x, offset_y) in enumerate(offsets, 1):
            timestamp_ns = 100_000_000 + index * 8_000_000
            jittered = (
                self.player.box[0] + offset_x,
                self.player.box[1] + offset_y,
                self.player.box[2] + offset_x,
                self.player.box[3] + offset_y,
            )
            self.runtime.accept_body(
                jittered,
                aim_box=jittered,
                corroboration_box=jittered,
                track_generation=7,
                source_timestamp_ns=timestamp_ns,
            )
            self.assertIsNone(self.runtime.take_latest(now_ns=timestamp_ns))
            last = self.runtime.visible_sample(now_ns=timestamp_ns)
            assert last is not None
            expected_input = (200.0 + offset_x, 150.0 + offset_y)
            expected_position = (
                expected_position[0]
                + position_alpha * (expected_input[0] - expected_position[0]),
                expected_position[1]
                + position_alpha * (expected_input[1] - expected_position[1]),
            )
            self.assertAlmostEqual(last.point[0], expected_position[0])
            self.assertAlmostEqual(last.point[1], expected_position[1])
            self.assertAlmostEqual(last.velocity_point[0], expected_input[0])
            self.assertAlmostEqual(last.velocity_point[1], expected_input[1])
            radii.append(math.dist(last.point, (200.0, 150.0)))

        # The fallback position must attenuate rather than amplify a repeating
        # three-pixel detector wobble. Its separately bounded velocity channel
        # remains available to the body-derived controller path.
        for radius in radii:
            self.assertLess(radius, 3.0)
        assert last is not None
        self.assertTrue(last.body_derived_motion_permitted)
        self.assertEqual(
            last.body_derived_motion_deadline_ns,
            last.identity_deadline_ns,
        )
        self.assertIsNone(last.corroboration_point)

    def test_recorded_one_edge_box_collapse_stays_continuous_and_slew_bounded(
        self,
    ) -> None:
        previous = (
            913.2299194335938,
            522.3317260742188,
            979.3130493164062,
            636.155029296875,
        )
        collapsed = (
            910.9881591796875,
            558.9227905273438,
            971.1378173828125,
            641.5419311523438,
        )
        recovered = (
            911.0,
            524.0,
            978.0,
            638.0,
        )
        first_ns = 100_000_000
        collapsed_ns = first_ns + 4_087_992
        recovered_ns = collapsed_ns + 4_100_000
        self.runtime.accept_body(
            previous,
            aim_box=previous,
            corroboration_box=previous,
            track_generation=58,
            source_timestamp_ns=first_ns,
        )
        self.worker.result = _result(
            source_ns=first_ns,
            point=(944.8208760805455, 530.0),
            selected_player_box=previous,
        )
        direct = self.runtime.take_latest(now_ns=first_ns + 1_000_000)
        assert direct is not None
        self.assertEqual(direct.velocity_point, direct.point)
        deadline_ns = direct.identity_deadline_ns
        filter_point = self.runtime._mapped_filter_point
        filter_input_point = self.runtime._mapped_filter_input_point
        filter_timestamp_ns = self.runtime._mapped_filter_timestamp_ns
        assert filter_point is not None
        assert filter_input_point is not None

        self.assertFalse(
            self.runtime.accept_body(
                collapsed,
                aim_box=collapsed,
                corroboration_box=collapsed,
                track_generation=58,
                source_timestamp_ns=collapsed_ns,
            )
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=collapsed_ns))
        collapsed_sample = self.runtime.visible_sample(now_ns=collapsed_ns)
        assert collapsed_sample is not None
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertEqual(self.runtime.anchor.identity_deadline_ns, deadline_ns)
        self.assertEqual(collapsed_sample.identity_deadline_ns, deadline_ns)
        self.assertEqual(
            self.runtime._mapped_filter_timestamp_ns,
            collapsed_ns,
        )
        maximum_step = (
            AUTOMATIC_HEAD_MAPPED_STEP_ALLOWANCE_PIXELS
            + AUTOMATIC_HEAD_MAPPED_MAX_SPEED_PIXELS_PER_SECOND
            * (collapsed_ns - filter_timestamp_ns)
            / 1_000_000_000
        )
        self.assertLessEqual(
            math.dist(collapsed_sample.velocity_point, filter_input_point),
            maximum_step + 1e-9,
        )
        # Coherent body translation is no longer alpha-scaled, but its center
        # delta is still qualified by the same per-frame physical step bound.
        self.assertLessEqual(
            math.dist(collapsed_sample.point, filter_point),
            maximum_step + 1e-9,
        )

        # Recovery remains in the same anchor epoch too. Neither half of the
        # recorded collapse/recovery pair may publish a loss or a raw 51 px
        # reseed jump.
        self.assertFalse(
            self.runtime.accept_body(
                recovered,
                aim_box=recovered,
                corroboration_box=recovered,
                track_generation=58,
                source_timestamp_ns=recovered_ns,
            )
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=recovered_ns))
        resumed = self.runtime.visible_sample(now_ns=recovered_ns)
        assert resumed is not None
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertEqual(resumed.identity_deadline_ns, deadline_ns)
        self.assertLess(
            math.dist(resumed.point, filter_point),
            8.0,
        )

    def test_repeated_one_edge_collapse_converges_without_identity_reset(
        self,
    ) -> None:
        previous = (
            878.1204833984375,
            503.1250305175781,
            1002.0667724609375,
            621.5841064453125,
        )
        collapsed = (
            868.4522094726562,
            555.4608764648438,
            945.2994384765625,
            625.739990234375,
        )
        self.runtime.accept_body(
            previous,
            aim_box=previous,
            corroboration_box=previous,
            track_generation=58,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(940.0, 520.0),
            selected_player_box=previous,
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=101_000_000))

        self.assertFalse(
            self.runtime.accept_body(
                collapsed,
                aim_box=collapsed,
                corroboration_box=collapsed,
                track_generation=58,
                source_timestamp_ns=112_000_000,
            )
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=112_000_000))
        first_collapsed = self.runtime.visible_sample(now_ns=112_000_000)
        assert first_collapsed is not None
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(self.runtime.anchor.active)

        self.assertFalse(
            self.runtime.accept_body(
                collapsed,
                aim_box=collapsed,
                corroboration_box=collapsed,
                track_generation=58,
                source_timestamp_ns=124_000_000,
            )
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=124_000_000))
        second_collapsed = self.runtime.visible_sample(now_ns=124_000_000)
        assert second_collapsed is not None
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(self.runtime.anchor.active)
        maximum_step = (
            AUTOMATIC_HEAD_MAPPED_STEP_ALLOWANCE_PIXELS
            + AUTOMATIC_HEAD_MAPPED_MAX_SPEED_PIXELS_PER_SECOND * 0.012
        )
        self.assertLessEqual(
            math.dist(
                second_collapsed.velocity_point,
                first_collapsed.velocity_point,
            ),
            maximum_step + 1e-9,
        )

    def test_recorded_ordinary_box_flex_is_unmodified_and_never_deferred(
        self,
    ) -> None:
        timestamps = (
            342_639_317_781_770,
            342_639_321_908_512,
            342_639_334_411_950,
        )
        boxes = (
            (
                848.682373046875,
                419.85260009765625,
                886.4513549804688,
                480.5121154785156,
            ),
            (
                848.836669921875,
                416.7593078613281,
                892.9866333007812,
                485.8830261230469,
            ),
            (
                848.8801879882812,
                416.093505859375,
                892.5843505859375,
                484.77899169921875,
            ),
        )
        direct_point = (867.5, 425.0)
        first_box = boxes[0]
        normalized_point = (
            (direct_point[0] - first_box[0]) / (first_box[2] - first_box[0]),
            (direct_point[1] - first_box[1]) / (first_box[3] - first_box[1]),
        )

        self.runtime.accept_body(
            first_box,
            aim_box=first_box,
            corroboration_box=first_box,
            track_generation=43,
            source_timestamp_ns=timestamps[0],
        )
        self.worker.result = _result(
            source_ns=timestamps[0],
            point=direct_point,
            selected_player_box=first_box,
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=timestamps[0]))

        expected_velocity_point = direct_point
        expected_reconcile_point = direct_point
        previous_center = (
            (first_box[0] + first_box[2]) * 0.5,
            (first_box[1] + first_box[3]) * 0.5,
        )
        previous_timestamp_ns = timestamps[0]
        for timestamp_ns, box in zip(timestamps[1:], boxes[1:]):
            self.assertFalse(
                self.runtime.accept_body(
                    box,
                    aim_box=box,
                    corroboration_box=box,
                    track_generation=43,
                    source_timestamp_ns=timestamp_ns,
                )
            )
            self.assertIsNone(self.runtime.take_latest(now_ns=timestamp_ns))
            visible = self.runtime.visible_sample(now_ns=timestamp_ns)
            assert visible is not None
            expected_raw_mapping = (
                box[0] + normalized_point[0] * (box[2] - box[0]),
                box[1] + normalized_point[1] * (box[3] - box[1]),
            )
            velocity_alpha = 1.0 - math.exp(
                -(
                    timestamp_ns - previous_timestamp_ns
                )
                / (
                    AUTOMATIC_HEAD_MAPPED_VELOCITY_FILTER_TIME_CONSTANT_SECONDS
                    * 1_000_000_000
                )
            )
            reconcile_alpha = 1.0 - math.exp(
                -(
                    timestamp_ns - previous_timestamp_ns
                )
                / (
                    AUTOMATIC_HEAD_VELOCITY_RECONCILIATION_TIME_CONSTANT_SECONDS
                    * 1_000_000_000
                )
            )
            current_center = (
                (box[0] + box[2]) * 0.5,
                (box[1] + box[3]) * 0.5,
            )
            center_delta = (
                current_center[0] - previous_center[0],
                current_center[1] - previous_center[1],
            )
            reconcile_prediction = (
                expected_reconcile_point[0] + center_delta[0],
                expected_reconcile_point[1] + center_delta[1],
            )
            expected_reconcile_point = (
                reconcile_prediction[0]
                + reconcile_alpha
                * (expected_raw_mapping[0] - reconcile_prediction[0]),
                reconcile_prediction[1]
                + reconcile_alpha
                * (expected_raw_mapping[1] - reconcile_prediction[1]),
            )
            velocity_prediction = (
                expected_velocity_point[0] + center_delta[0],
                expected_velocity_point[1] + center_delta[1],
            )
            expected_velocity_point = (
                velocity_prediction[0]
                + velocity_alpha
                * (expected_reconcile_point[0] - velocity_prediction[0]),
                velocity_prediction[1]
                + velocity_alpha
                * (expected_reconcile_point[1] - velocity_prediction[1]),
            )
            self.assertAlmostEqual(
                visible.velocity_point[0],
                expected_velocity_point[0],
            )
            self.assertAlmostEqual(
                visible.velocity_point[1],
                expected_velocity_point[1],
            )
            previous_timestamp_ns = timestamp_ns
            previous_center = current_center
            self.assertFalse(self.runtime.body_update_deferred)
            self.assertEqual(self.runtime.identity_generation, 0)
            self.assertTrue(self.runtime.anchor.active)

    def test_anchor_expires_at_seven_hundred_fifty_ms_and_advances_safety_epoch(
        self,
    ) -> None:
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=8,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(source_ns=100_000_000, point=(200.0, 150.0))
        self.assertIsNotNone(self.runtime.take_latest(now_ns=105_000_000))

        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=8,
            source_timestamp_ns=849_000_000,
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=849_000_000))
        self.assertIsNotNone(self.runtime.visible_sample(now_ns=849_000_000))
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=8,
            source_timestamp_ns=850_000_000,
        )

        self.assertIsNone(self.runtime.take_latest(now_ns=850_000_000))
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.body_valid)
        self.assertFalse(self.runtime.anchor.active)
        self.assertIsNone(self.runtime.visible_sample(now_ns=850_000_000))

    def test_stale_worker_result_cannot_revoke_a_newer_live_anchor(self) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=5,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(source_ns=100_000_000, point=(200.0, 150.0))
        self.assertIsNotNone(self.runtime.take_latest(now_ns=105_000_000))

        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=5,
            source_timestamp_ns=260_000_000,
        )
        # Source-100 is stale enough that its exact binding has been pruned.
        # It must be ignored before binding validation, not reset generation 5.
        self.worker.result = _result(source_ns=100_000_000, point=(210.0, 155.0))
        self.assertIsNone(self.runtime.take_latest(now_ns=260_000_000))
        mapped = self.runtime.visible_sample(now_ns=260_000_000)

        assert mapped is not None
        self.assertEqual(mapped.source_timestamp_ns, 260_000_000)
        self.assertEqual(mapped.direct_source_timestamp_ns, 100_000_000)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(self.runtime.anchor.active)

    def test_incompatible_late_result_cannot_revoke_newer_verified_anchor(
        self,
    ) -> None:
        first = (800.0, 200.0, 1000.0, 800.0)
        newer = (840.0, 200.0, 1040.0, 800.0)
        current = (880.0, 200.0, 1080.0, 800.0)
        self.runtime.accept_body(
            first,
            corroboration_box=first,
            track_generation=5,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(900.0, 260.0),
            selected_player_box=first,
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=104_000_000))

        self.runtime.accept_body(
            newer,
            corroboration_box=newer,
            track_generation=5,
            source_timestamp_ns=108_000_000,
        )
        self.worker.result = _result(
            source_ns=108_000_000,
            point=(940.0, 260.0),
            selected_player_box=newer,
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=112_000_000))
        self.runtime.accept_body(
            current,
            corroboration_box=current,
            track_generation=5,
            source_timestamp_ns=116_000_000,
        )

        # The source-100 result is too far from the current box over its full
        # interval. The independent source-108 anchor is still exactly bound
        # and advances to current through two physically bounded steps.
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(900.0, 260.0),
            selected_player_box=first,
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=120_000_000))
        retained = self.runtime.visible_sample(now_ns=120_000_000)

        assert retained is not None
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(self.runtime.body_valid)
        self.assertTrue(self.runtime.anchor.active)
        self.assertEqual(retained.direct_source_timestamp_ns, 108_000_000)
        self.assertEqual(retained.source_timestamp_ns, 116_000_000)

    def test_late_direct_result_follows_exact_fast_measured_body_trace(
        self,
    ) -> None:
        """A fast same-track translation may make async endpoint boxes disjoint."""

        worker = _FakeWorker()
        runtime = _AutomaticHeadRuntime(
            worker,
            submission_hz=90.0,
            stale_after_seconds=0.110,
        )
        source_ns = 339_444_574_408_007
        current_ns = 339_444_636_970_329
        source = (
            840.1611328125,
            474.2440490722656,
            900.7807006835938,
            570.9955444335938,
        )
        current = (
            922.89990234375,
            501.0458984375,
            994.4312744140625,
            603.6637573242188,
        )

        # Consecutive detector measurements overlap and remain in tracker
        # generation 32.  Only the async worker's source/current endpoints are
        # disjoint after 62.6 ms of commanded-camera motion.
        for fraction in (0.0, 0.32, 0.66, 1.0):
            timestamp_ns = round(
                source_ns + (current_ns - source_ns) * fraction
            )
            box = tuple(
                start + (end - start) * fraction
                for start, end in zip(source, current)
            )
            self.assertFalse(
                runtime.accept_body(
                    box,
                    aim_box=box,
                    corroboration_box=box,
                    track_generation=32,
                    source_timestamp_ns=timestamp_ns,
                )
            )
            self.assertFalse(runtime.body_update_deferred)

        self.assertFalse(
            runtime._player_boxes_associate_over_interval(
                source,
                current,
                elapsed_ns=current_ns - source_ns,
            )
        )
        self.assertTrue(
            runtime._player_boxes_associate_over_interval(
                source,
                current,
                elapsed_ns=current_ns - source_ns,
                allow_disjoint_measured_motion=True,
            )
        )
        worker.result = _result(
            source_ns=source_ns,
            point=(870.0, 490.0),
            selected_player_box=source,
        )

        direct = runtime.take_latest(now_ns=current_ns + 1_000_000)
        visible = runtime.visible_sample(now_ns=current_ns + 1_000_000)

        self.assertIsNotNone(direct)
        self.assertIsNotNone(visible)
        self.assertEqual(runtime.identity_generation, 0)
        self.assertTrue(runtime.anchor.active)
        assert visible is not None
        self.assertEqual(visible.source_timestamp_ns, current_ns)
        self.assertEqual(visible.direct_source_timestamp_ns, source_ns)

    def test_disjoint_late_result_fallback_rejects_fragment_and_teleport(
        self,
    ) -> None:
        source = (840.0, 474.0, 901.0, 571.0)
        fragment = (923.0, 501.0, 943.0, 542.0)
        teleported = (1240.0, 474.0, 1301.0, 571.0)

        self.assertFalse(
            self.runtime._player_boxes_associate_over_interval(
                source,
                fragment,
                elapsed_ns=63_000_000,
                allow_disjoint_measured_motion=True,
            )
        )
        self.assertFalse(
            self.runtime._player_boxes_associate_over_interval(
                source,
                teleported,
                elapsed_ns=63_000_000,
                allow_disjoint_measured_motion=True,
            )
        )

    def test_raw_primary_mapping_recovers_smoothed_box_anatomy_lag(self) -> None:
        original = self.player.box
        self.runtime.accept_body(
            original,
            aim_box=original,
            corroboration_box=original,
            track_generation=3,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(110.0, 150.0),
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=104_000_000))

        current = (180.0, 100.0, 380.0, 700.0)
        self.assertFalse(
            self.runtime.accept_body(
                current,
                # Simulate a valid tracker output whose smoothing trails the
                # exact accepted measurement by one high-motion frame.
                aim_box=original,
                corroboration_box=current,
                track_generation=3,
                source_timestamp_ns=150_000_000,
            )
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=150_000_000))
        mapped = self.runtime.visible_sample(now_ns=150_000_000)

        assert mapped is not None
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertAlmostEqual(mapped.point[0], 190.0, delta=2.0)
        self.assertEqual(mapped.point[1], 150.0)
        self.assertTrue(
            self.runtime._head_point_belongs_to_player(mapped.point, current)
        )

    def test_direct_sample_uses_same_source_body_center_only_as_corroboration(
        self,
    ) -> None:
        previous = self.player.box
        current = (110.0, 100.0, 310.0, 700.0)
        self.runtime.accept_body(
            previous,
            corroboration_box=previous,
            track_generation=5,
            source_timestamp_ns=92_000_000,
        )
        self.runtime.accept_body(
            current,
            corroboration_box=current,
            track_generation=5,
            source_timestamp_ns=100_000_000,
        )
        with (
            mock.patch(
                "detection.head_detector.plan_head_crop",
                return_value=object(),
            ),
            mock.patch(
                "detection.head_detector.prepare_head_input",
                return_value=object(),
            ),
        ):
            self.assertTrue(
                self.runtime.submit(
                    self.frame,
                    Detection(0, "player", 0.9, current),
                    source_timestamp_ns=100_000_000,
                )
            )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(205.0, 150.0),
            selected_player_box=current,
        )

        sample = self.runtime.take_latest(now_ns=108_000_000)

        assert sample is not None
        self.assertEqual(sample.point, (205.0, 150.0))
        self.assertEqual(sample.corroboration_point, (210.0, 400.0))
        self.assertIs(sample.provenance, DirectHeadProvenance.DIRECT)
        self.assertFalse(sample.body_derived_motion_permitted)
        self.assertFalse(
            self.runtime.consume_motion_corroboration_revocation()
        )

    def test_wrong_timestamp_or_predicted_body_cannot_bind_direct_result(self) -> None:
        for body_timestamp_ns, corroboration_box in (
            (101_000_000, self.player.box),
            (100_000_000, None),
        ):
            with self.subTest(body_timestamp_ns=body_timestamp_ns):
                worker = _FakeWorker()
                runtime = _AutomaticHeadRuntime(worker, submission_hz=60.0)
                runtime.accept_body(
                    self.player.box,
                    corroboration_box=corroboration_box,
                    track_generation=1,
                    source_timestamp_ns=body_timestamp_ns,
                )
                worker.result = _result(
                    source_ns=100_000_000,
                    point=(200.0, 150.0),
                )

                self.assertIsNone(runtime.take_latest(now_ns=110_000_000))
                self.assertIsNone(runtime.visible_sample(now_ns=110_000_000))

    def test_primary_miss_can_bridge_position_but_withdraws_corroboration(
        self,
    ) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=2,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(200.0, 150.0),
        )
        observed = self.runtime.take_latest(now_ns=102_000_000)
        assert observed is not None
        self.assertEqual(observed.corroboration_point, (200.0, 400.0))

        # A tracker prediction may retain the bounded direct-head lease but is
        # not an accepted primary measurement and therefore cannot grant FF.
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=None,
            track_generation=2,
            source_timestamp_ns=108_000_000,
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=110_000_000))
        bridged = self.runtime.visible_sample(now_ns=110_000_000)

        assert bridged is not None
        self.assertTrue(bridged.bridging)
        self.assertEqual(bridged.point, observed.point)
        self.assertIsNone(bridged.corroboration_point)

    def test_inflight_exact_head_during_prediction_keeps_same_source_corroboration(
        self,
    ) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=2,
            source_timestamp_ns=100_000_000,
        )
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=None,
            track_generation=2,
            source_timestamp_ns=108_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(200.0, 150.0),
        )

        direct = self.runtime.take_latest(now_ns=110_000_000)
        assert direct is not None
        self.assertEqual(direct.point, (200.0, 150.0))
        self.assertEqual(direct.source_timestamp_ns, 100_000_000)
        self.assertEqual(direct.corroboration_point, (200.0, 400.0))
        self.assertIs(direct.provenance, DirectHeadProvenance.DIRECT)
        sample = self.runtime.visible_sample(now_ns=110_000_000)
        assert sample is not None
        self.assertEqual(sample.point, (200.0, 150.0))
        self.assertIsNone(sample.corroboration_point)
        self.assertTrue(sample.bridging)
        self.assertIs(
            sample.provenance,
            DirectHeadProvenance.PREDICTED_PRIMARY,
        )
        self.assertTrue(
            self.runtime.consume_motion_corroboration_revocation()
        )

    def test_body_reacquisition_cannot_relabel_old_head_bridge_as_observed(
        self,
    ) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=2,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(200.0, 150.0),
        )
        observed = self.runtime.take_latest(now_ns=102_000_000)
        assert observed is not None
        self.assertFalse(observed.bridging)
        self.assertEqual(observed.corroboration_point, (200.0, 400.0))

        self.runtime.accept_body(
            self.player.box,
            corroboration_box=None,
            track_generation=2,
            source_timestamp_ns=108_000_000,
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=110_000_000))
        predicted = self.runtime.visible_sample(now_ns=110_000_000)
        assert predicted is not None
        self.assertTrue(predicted.bridging)
        self.assertIsNone(predicted.corroboration_point)
        self.assertTrue(
            self.runtime.consume_motion_corroboration_revocation()
        )

        # A measured primary may carry the still-live direct anchor again, but
        # it cannot renew the direct deadline or resurrect corroboration.
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=2,
            source_timestamp_ns=116_000_000,
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=118_000_000))
        body_only_reacquired = self.runtime.visible_sample(now_ns=118_000_000)
        assert body_only_reacquired is not None
        self.assertFalse(body_only_reacquired.bridging)
        self.assertIsNone(body_only_reacquired.corroboration_point)
        self.assertEqual(
            body_only_reacquired.source_timestamp_ns,
            116_000_000,
        )
        self.assertEqual(
            body_only_reacquired.direct_source_timestamp_ns,
            observed.direct_source_timestamp_ns,
        )

    def test_submission_uses_current_same_frame_box_not_previous_box(self) -> None:
        first = (800.0, 200.0, 1000.0, 800.0)
        current = (920.0, 200.0, 1120.0, 800.0)
        self.runtime.accept_body(
            first,
            corroboration_box=first,
            track_generation=9,
            source_timestamp_ns=100_000_000,
        )
        self.runtime.accept_body(
            current,
            corroboration_box=current,
            track_generation=9,
            source_timestamp_ns=150_000_000,
        )
        current_player = Detection(0, "player", 0.9, current)
        with (
            mock.patch(
                "detection.head_detector.plan_head_crop",
                return_value=object(),
            ) as plan_crop,
            mock.patch(
                "detection.head_detector.prepare_head_input",
                return_value=object(),
            ),
        ):
            self.assertTrue(
                self.runtime.submit(
                    self.frame,
                    current_player,
                    source_timestamp_ns=150_000_000,
                )
            )
        plan_crop.assert_called_once_with(
            self.frame.shape,
            current,
            crop_scale=1.25,
            model_size=(320, 320),
        )
        self.assertEqual(
            self.worker.submissions[0]["selected_player_box"],
            current,
        )
        self.worker.result = _result(
            source_ns=150_000_000,
            point=(1000.0, 260.0),
            selected_player_box=current,
        )

        sample = self.runtime.take_latest(now_ns=150_000_000)

        assert sample is not None
        self.assertEqual(sample.point, (1000.0, 260.0))
        self.assertEqual(sample.corroboration_point, (1020.0, 500.0))

    def test_stale_previous_box_rejects_motion_that_same_frame_box_accepts(
        self,
    ) -> None:
        previous = (900.0, 300.0, 960.0, 600.0)
        current = (922.0, 300.0, 982.0, 600.0)
        candidates = [
            HeadCandidate(0, "player", 0.9, current, 0),
            HeadCandidate(1, "head", 0.9, (940.0, 315.0, 964.0, 355.0), 1),
        ]

        self.assertIsNone(
            associate_head_to_player(
                candidates,
                previous,
                source_timestamp_ns=150_000_000,
            )
        )
        self.assertIsNotNone(
            associate_head_to_player(
                candidates,
                current,
                source_timestamp_ns=150_000_000,
            )
        )

    def test_stale_or_missing_head_never_becomes_visible(self) -> None:
        self.runtime.accept_body(self.player.box)
        self.worker.result = _result(source_ns=100_000_000)
        self.assertIsNone(self.runtime.take_latest(now_ns=165_000_001))
        self.assertIsNone(self.runtime.visible_sample(now_ns=165_000_001))

        self.worker.result = _result(source_ns=200_000_000, point=None)
        self.assertIsNone(self.runtime.take_latest(now_ns=210_000_000))
        self.assertIsNone(self.runtime.visible_sample(now_ns=210_000_000))

    def test_clustered_head_misses_keep_bounded_mapped_motion_without_sawtooth_revoke(
        self,
    ) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=3,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(source_ns=100_000_000)
        original = self.runtime.take_latest(now_ns=110_000_000)
        assert original is not None
        samples = []
        for index, timestamp_ns in enumerate(
            (
                120_000_000,
                150_000_000,
                164_000_000,
                190_000_000,
                250_000_000,
            ),
            1,
        ):
            translated = tuple(value + index * 5.0 for value in self.player.box)
            self.runtime.accept_body(
                translated,
                aim_box=translated,
                corroboration_box=translated,
                track_generation=3,
                source_timestamp_ns=timestamp_ns,
            )
            if index == 1:
                self.worker.result = _result(
                    source_ns=timestamp_ns,
                    point=None,
                    selected_player_box=translated,
                )
            self.assertIsNone(
                self.runtime.take_latest(now_ns=timestamp_ns + 1_000_000)
            )
            sample = self.runtime.visible_sample(
                now_ns=timestamp_ns + 1_000_000
            )
            assert sample is not None
            samples.append(sample)
            self.assertEqual(sample.source_timestamp_ns, timestamp_ns)
            self.assertEqual(sample.direct_source_timestamp_ns, 100_000_000)
            self.assertFalse(sample.bridging)
            self.assertIsNone(sample.corroboration_point)
            self.assertTrue(sample.body_derived_motion_permitted)
            self.assertEqual(
                sample.body_derived_motion_deadline_ns,
                sample.identity_deadline_ns,
            )

        self.assertGreater(samples[-1].point[0], original.point[0])
        # A non-spatial decoder miss cannot contradict motion already mapped
        # through this same measured primary.  The immutable direct deadline
        # still bounds every sample above, without repeatedly erasing and
        # rebuilding the pursuit observer between positive head results.
        self.assertFalse(
            self.runtime.consume_motion_corroboration_revocation()
        )

    def test_identity_advance_and_body_loss_clear_point_and_pending_result(self) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(source_ns=100_000_000)
        self.assertIsNotNone(self.runtime.take_latest(now_ns=110_000_000))

        self.assertTrue(self.runtime.revoke_body())
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.body_valid)
        self.assertIsNone(self.runtime.visible_sample(now_ns=111_000_000))
        self.assertFalse(self.runtime.revoke_body())

        # Even if a test double exposes an old result later, the generation
        # checked by take_latest prevents it from arming the new identity.
        self.runtime.accept_body(self.player.box)
        self.worker.result = _result(source_ns=112_000_000, generation=0)
        self.assertIsNone(self.runtime.take_latest(now_ns=113_000_000))

    def test_same_generation_body_gap_pauses_and_resumes_verified_anchor(
        self,
    ) -> None:
        generation = 7
        direct_ns = 100_000_000
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=generation,
            source_timestamp_ns=direct_ns,
        )
        self.worker.result = _result(source_ns=direct_ns)
        direct = self.runtime.take_latest(now_ns=101_000_000)
        assert direct is not None
        original_deadline_ns = direct.identity_deadline_ns
        original_point = direct.point

        self.assertTrue(
            self.runtime.suspend_body_gap(now_ns=110_000_000)
        )
        self.assertFalse(self.runtime.body_valid)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(self.runtime.anchor.active)
        self.assertEqual(self.runtime.anchor.track_generation, generation)
        self.assertEqual(
            self.runtime.anchor.identity_deadline_ns,
            original_deadline_ns,
        )
        self.assertIsNone(self.runtime.visible_sample(now_ns=120_000_000))
        self.assertTrue(
            self.runtime.suspend_body_gap(now_ns=120_000_000)
        )
        self.assertTrue(self.runtime.body_gap_suspended)
        self.assertEqual(self.runtime.identity_generation, 0)

        resumed_ns = 150_000_000
        resumed_box = tuple(value + 8.0 for value in self.player.box)
        self.assertFalse(
            self.runtime.accept_body(
                resumed_box,
                aim_box=resumed_box,
                corroboration_box=resumed_box,
                track_generation=generation,
                source_timestamp_ns=resumed_ns,
            )
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=resumed_ns + 1))
        resumed = self.runtime.visible_sample(now_ns=resumed_ns + 1)
        assert resumed is not None
        self.assertTrue(self.runtime.body_valid)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertEqual(resumed.source_timestamp_ns, resumed_ns)
        self.assertEqual(resumed.direct_source_timestamp_ns, direct_ns)
        self.assertEqual(resumed.identity_deadline_ns, original_deadline_ns)
        self.assertAlmostEqual(resumed.point[0], original_point[0] + 8.0)
        self.assertAlmostEqual(resumed.point[1], original_point[1] + 8.0)

    def test_suspended_anchor_cannot_cross_generation_or_unsafe_revoke(self) -> None:
        generation = 11
        direct_ns = 100_000_000
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=generation,
            source_timestamp_ns=direct_ns,
        )
        self.worker.result = _result(source_ns=direct_ns)
        self.assertIsNotNone(self.runtime.take_latest(now_ns=101_000_000))
        self.assertTrue(
            self.runtime.suspend_body_gap(now_ns=110_000_000)
        )

        self.assertTrue(
            self.runtime.accept_body(
                self.player.box,
                aim_box=self.player.box,
                corroboration_box=self.player.box,
                track_generation=generation + 1,
                source_timestamp_ns=120_000_000,
            )
        )
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.anchor.active)
        self.assertIsNone(self.runtime.visible_sample(now_ns=120_000_001))

        # Suspension is not a release/self-guard bypass. A hard revocation
        # after a later verified anchor still erases the retained identity.
        replacement_ns = 130_000_000
        self.worker.result = _result(
            source_ns=replacement_ns,
            generation=1,
        )
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=generation + 1,
            source_timestamp_ns=replacement_ns,
        )
        self.assertIsNotNone(
            self.runtime.take_latest(now_ns=replacement_ns + 1)
        )
        self.assertTrue(
            self.runtime.suspend_body_gap(now_ns=140_000_000)
        )
        self.assertTrue(self.runtime.revoke_body())
        self.assertEqual(self.runtime.identity_generation, 2)
        self.assertFalse(self.runtime.anchor.active)

    def test_body_gap_without_live_verified_anchor_cannot_suspend(self) -> None:
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=5,
            source_timestamp_ns=100_000_000,
        )

        self.assertFalse(
            self.runtime.suspend_body_gap(now_ns=110_000_000)
        )
        self.assertTrue(self.runtime.body_valid)

    def test_expired_verified_anchor_cannot_suspend_body_gap(self) -> None:
        direct_ns = 100_000_000
        self.runtime.accept_body(
            self.player.box,
            aim_box=self.player.box,
            corroboration_box=self.player.box,
            track_generation=5,
            source_timestamp_ns=direct_ns,
        )
        self.worker.result = _result(source_ns=direct_ns)
        direct = self.runtime.take_latest(now_ns=direct_ns + 1)
        assert direct is not None

        self.assertFalse(
            self.runtime.suspend_body_gap(
                now_ns=direct.identity_deadline_ns,
            )
        )
        self.assertTrue(self.runtime.body_valid)

    def test_late_result_from_geometrically_replaced_player_never_arms(self) -> None:
        self.runtime.accept_body(self.player.box)
        self.worker.result = _result(source_ns=100_000_000)

        replacement = (900.0, 100.0, 1100.0, 700.0)
        # One incompatible body frame is quarantined: it cannot consume the
        # pending old crop, publish geometry, or erase the established epoch.
        self.assertFalse(self.runtime.accept_body(replacement))
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertIsNone(self.runtime.take_latest(now_ns=105_000_000))
        # A second compatible replacement confirms the transition and starts
        # a fresh epoch before accepting its geometry.
        self.assertTrue(self.runtime.accept_body(replacement))
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 1)
        self.worker.result = _result(source_ns=101_000_000, generation=0)
        self.assertIsNone(self.runtime.take_latest(now_ns=110_000_000))
        self.assertIsNone(self.runtime.visible_sample(now_ns=110_000_000))

    def test_one_frame_body_mismatch_preserves_bounded_anchor_without_crop(
        self,
    ) -> None:
        original = self.player.box
        self.runtime.accept_body(
            original,
            aim_box=original,
            corroboration_box=original,
            track_generation=4,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(200.0, 150.0),
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=104_000_000))

        mismatch = (900.0, 100.0, 1100.0, 700.0)
        self.assertFalse(
            self.runtime.accept_body(
                mismatch,
                aim_box=mismatch,
                corroboration_box=mismatch,
                track_generation=4,
                source_timestamp_ns=108_000_000,
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertTrue(self.runtime.body_valid)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(
            self.runtime.consume_motion_corroboration_revocation()
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=108_000_000))
        self.assertIsNone(self.runtime.visible_sample(now_ns=108_000_000))
        self.assertFalse(
            self.runtime.submit(
                self.frame,
                Detection(0, "player", 0.9, mismatch),
                source_timestamp_ns=108_000_000,
            )
        )

        # Returning to the trusted geometry clears the quarantine and resumes
        # the same immutable direct-head lease without an identity reset.
        self.assertFalse(
            self.runtime.accept_body(
                original,
                aim_box=original,
                corroboration_box=original,
                track_generation=4,
                source_timestamp_ns=116_000_000,
            )
        )
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertIsNotNone(self.runtime.visible_sample(now_ns=116_000_000))

    def test_recorded_singleton_body_excursion_keeps_same_generation_head_lease(
        self,
    ) -> None:
        trusted_ns = 100_000_000
        excursion_ns = trusted_ns + 16_764_019
        pending_ns = trusted_ns + 25_007_822
        confirming_ns = trusted_ns + 37_622_209
        trusted = (
            977.8775634765625,
            508.55767822265625,
            1016.749755859375,
            589.260498046875,
        )
        excursion = (
            996.0299072265625,
            483.6278076171875,
            1047.69287109375,
            556.3037719726562,
        )
        pending = (
            955.7386474609375,
            532.6640625,
            999.8295288085938,
            593.3821411132812,
        )
        confirming = (
            960.395751953125,
            528.9095458984375,
            990.4163208007812,
            590.27978515625,
        )
        generation = 95
        self.assertFalse(
            self.runtime.accept_body(
                trusted,
                aim_box=trusted,
                corroboration_box=trusted,
                track_generation=generation,
                source_timestamp_ns=trusted_ns,
            )
        )
        self.worker.result = _result(
            source_ns=trusted_ns,
            point=(1005.0, 520.0),
            selected_player_box=trusted,
        )
        initial = self.runtime.take_latest(now_ns=trusted_ns + 1_000_000)
        self.assertIsNotNone(initial)
        assert initial is not None

        self.assertFalse(
            self.runtime.accept_body(
                excursion,
                aim_box=excursion,
                corroboration_box=excursion,
                track_generation=generation,
                source_timestamp_ns=excursion_ns,
            )
        )
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertFalse(
            self.runtime.accept_body(
                pending,
                aim_box=pending,
                corroboration_box=pending,
                track_generation=generation,
                source_timestamp_ns=pending_ns,
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertFalse(
            self.runtime.accept_body(
                confirming,
                aim_box=confirming,
                corroboration_box=confirming,
                track_generation=generation,
                source_timestamp_ns=confirming_ns,
            )
        )

        self.assertFalse(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(self.runtime.anchor.active)
        self.assertEqual(
            self.runtime.anchor.identity_deadline_ns,
            initial.identity_deadline_ns,
        )
        self.assertNotIn(
            excursion_ns,
            self.runtime._observed_primary_sources,
        )
        self.assertIn(
            excursion_ns,
            self.runtime._rejected_body_outlier_source_timestamps,
        )
        # ``accept_body`` updates identity geometry only.  The normal runtime
        # tick maps the immutable direct anchor through that new geometry even
        # when no asynchronous head result completed on this exact frame.
        self.assertIsNone(
            self.runtime.take_latest(now_ns=confirming_ns + 1_000_000)
        )
        visible = self.runtime.visible_sample(now_ns=confirming_ns)
        self.assertIsNotNone(visible)
        assert visible is not None
        self.assertEqual(visible.source_timestamp_ns, confirming_ns)
        self.assertEqual(visible.track_generation, generation)

        # An async result already in flight for the removed excursion is
        # discarded without revoking the recovered clean anchor.
        self.worker.result = _result(
            source_ns=excursion_ns,
            point=(1020.0, 500.0),
            selected_player_box=excursion,
        )
        self.assertIsNone(
            self.runtime.take_latest(now_ns=confirming_ns + 1_000_000)
        )
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(self.runtime.anchor.active)

    def test_directly_verified_body_excursion_still_starts_new_identity(
        self,
    ) -> None:
        trusted_ns = 100_000_000
        excursion_ns = 116_000_000
        pending_ns = 124_000_000
        confirming_ns = 136_000_000
        trusted = (978.0, 509.0, 1017.0, 589.0)
        excursion = (996.0, 484.0, 1048.0, 556.0)
        pending = (956.0, 533.0, 1000.0, 593.0)
        confirming = (960.0, 529.0, 990.0, 590.0)
        generation = 95
        self.runtime.accept_body(
            trusted,
            corroboration_box=trusted,
            track_generation=generation,
            source_timestamp_ns=trusted_ns,
        )
        self.worker.result = _result(
            source_ns=trusted_ns,
            point=(1005.0, 520.0),
            selected_player_box=trusted,
        )
        self.assertIsNotNone(
            self.runtime.take_latest(now_ns=trusted_ns + 1_000_000)
        )
        self.runtime.accept_body(
            excursion,
            corroboration_box=excursion,
            track_generation=generation,
            source_timestamp_ns=excursion_ns,
        )
        self.worker.result = _result(
            source_ns=excursion_ns,
            point=(1020.0, 500.0),
            selected_player_box=excursion,
        )
        self.assertIsNotNone(
            self.runtime.take_latest(now_ns=excursion_ns + 1_000_000)
        )
        self.assertFalse(
            self.runtime.accept_body(
                pending,
                corroboration_box=pending,
                track_generation=generation,
                source_timestamp_ns=pending_ns,
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertTrue(
            self.runtime.accept_body(
                confirming,
                corroboration_box=confirming,
                track_generation=generation,
                source_timestamp_ns=confirming_ns,
            )
        )
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.anchor.active)

    def test_phase_verified_body_excursion_still_starts_new_identity(self) -> None:
        anchor_ns = 90_000_000
        trusted_ns = 100_000_000
        excursion_ns = trusted_ns + 16_764_019
        pending_ns = trusted_ns + 25_007_822
        confirming_ns = trusted_ns + 37_622_209
        trusted = (
            977.8775634765625,
            508.55767822265625,
            1016.749755859375,
            589.260498046875,
        )
        excursion = (
            996.0299072265625,
            483.6278076171875,
            1047.69287109375,
            556.3037719726562,
        )
        pending = (
            955.7386474609375,
            532.6640625,
            999.8295288085938,
            593.3821411132812,
        )
        confirming = (
            960.395751953125,
            528.9095458984375,
            990.4163208007812,
            590.27978515625,
        )
        generation = 95
        direct_point = (1005.0, 520.0)
        normalized = (
            (direct_point[0] - trusted[0]) / (trusted[2] - trusted[0]),
            (direct_point[1] - trusted[1]) / (trusted[3] - trusted[1]),
        )
        phase_point = (
            excursion[0] + normalized[0] * (excursion[2] - excursion[0]),
            excursion[1] + normalized[1] * (excursion[3] - excursion[1]),
        )
        worker = _FakeWorker()
        runtime = _AutomaticHeadRuntime(
            worker,
            submission_hz=60.0,
            stale_after_seconds=0.065,
            phase_advancer=_ScriptedPhaseAdvancer((phase_point,)),
        )
        runtime.accept_body(
            trusted,
            corroboration_box=trusted,
            track_generation=generation,
            source_timestamp_ns=anchor_ns,
        )
        worker.result = _result(
            source_ns=anchor_ns,
            point=direct_point,
            selected_player_box=trusted,
        )
        self.assertIsNotNone(runtime.take_latest(now_ns=anchor_ns + 1))

        runtime.accept_body(
            trusted,
            corroboration_box=trusted,
            track_generation=generation,
            source_timestamp_ns=trusted_ns,
        )
        self.assertTrue(
            runtime.remember_frame(self.frame, source_timestamp_ns=trusted_ns)
        )
        runtime.accept_body(
            excursion,
            corroboration_box=excursion,
            track_generation=generation,
            source_timestamp_ns=excursion_ns,
        )
        self.assertTrue(
            runtime.remember_frame(self.frame, source_timestamp_ns=excursion_ns)
        )
        worker.result = _result(
            source_ns=trusted_ns,
            point=direct_point,
            selected_player_box=trusted,
            head_box=(990.0, 505.0, 1020.0, 535.0),
        )
        phase_verified = runtime.take_latest(now_ns=excursion_ns + 1)
        self.assertIsNotNone(phase_verified)
        assert phase_verified is not None
        self.assertTrue(phase_verified.phase_advanced)
        self.assertEqual(phase_verified.direct_source_timestamp_ns, trusted_ns)
        self.assertEqual(phase_verified.source_timestamp_ns, excursion_ns)
        self.assertEqual(runtime.anchor.last_direct_source_timestamp_ns, trusted_ns)
        self.assertEqual(
            runtime._last_physical_source_timestamp_ns,
            excursion_ns,
        )

        self.assertFalse(
            runtime.accept_body(
                pending,
                corroboration_box=pending,
                track_generation=generation,
                source_timestamp_ns=pending_ns,
            )
        )
        self.assertTrue(runtime.body_update_deferred)
        self.assertTrue(
            runtime.accept_body(
                confirming,
                corroboration_box=confirming,
                track_generation=generation,
                source_timestamp_ns=confirming_ns,
            )
        )
        self.assertEqual(runtime.identity_generation, 1)
        self.assertFalse(runtime.anchor.active)
        self.assertNotIn(
            excursion_ns,
            runtime._rejected_body_outlier_source_timestamps,
        )

    def test_active_head_lease_does_not_cross_to_confirmed_far_rival(self) -> None:
        trusted_ns = 100_000_000
        current_ns = 108_000_000
        rival_pending_ns = 116_000_000
        rival_confirming_ns = 124_000_000
        trusted = (900.0, 420.0, 980.0, 650.0)
        current = (908.0, 420.0, 988.0, 650.0)
        rival_pending = (1220.0, 390.0, 1300.0, 640.0)
        rival_confirming = (1228.0, 390.0, 1308.0, 640.0)
        generation = 96
        self.runtime.accept_body(
            trusted,
            corroboration_box=trusted,
            track_generation=generation,
            source_timestamp_ns=trusted_ns,
        )
        self.worker.result = _result(
            source_ns=trusted_ns,
            point=(940.0, 445.0),
            selected_player_box=trusted,
        )
        self.assertIsNotNone(
            self.runtime.take_latest(now_ns=trusted_ns + 1_000_000)
        )
        self.runtime.accept_body(
            current,
            corroboration_box=current,
            track_generation=generation,
            source_timestamp_ns=current_ns,
        )

        self.assertFalse(
            self.runtime.accept_body(
                rival_pending,
                corroboration_box=rival_pending,
                track_generation=generation,
                source_timestamp_ns=rival_pending_ns,
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertTrue(
            self.runtime.accept_body(
                rival_confirming,
                corroboration_box=rival_confirming,
                track_generation=generation,
                source_timestamp_ns=rival_confirming_ns,
            )
        )
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.anchor.active)
        self.assertNotIn(
            current_ns,
            self.runtime._rejected_body_outlier_source_timestamps,
        )

    def test_alternating_body_mismatches_never_confirm_replacement(self) -> None:
        original = self.player.box
        rival_a = (700.0, 100.0, 900.0, 700.0)
        rival_b = (1200.0, 100.0, 1400.0, 700.0)
        self.runtime.accept_body(
            original,
            corroboration_box=original,
            track_generation=9,
            source_timestamp_ns=100_000_000,
        )

        for index, rival in enumerate((rival_a, rival_b, rival_a, rival_b), 1):
            self.assertFalse(
                self.runtime.accept_body(
                    rival,
                    corroboration_box=rival,
                    track_generation=9,
                    source_timestamp_ns=100_000_000 + index * 8_000_000,
                )
            )
            self.assertTrue(self.runtime.body_update_deferred)
            self.assertEqual(self.runtime.identity_generation, 0)

        self.assertTrue(self.runtime.body_valid)
        self.assertIsNone(self.runtime.visible_sample(now_ns=140_000_000))

    def test_track_generation_rejects_reacquisition_even_with_same_geometry(self) -> None:
        self.assertFalse(
            self.runtime.accept_body(
                self.player.box,
                corroboration_box=self.player.box,
                track_generation=1,
                source_timestamp_ns=100_000_000,
            )
        )
        self.worker.result = _result(source_ns=100_000_000)
        self.assertIsNotNone(self.runtime.take_latest(now_ns=105_000_000))
        self.assertFalse(
            self.runtime.accept_body(self.player.box, track_generation=1)
        )
        self.assertTrue(
            self.runtime.accept_body(self.player.box, track_generation=2)
        )
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.anchor.active)
        self.assertIsNone(self.runtime.visible_sample(now_ns=110_000_000))

    def test_geometry_rejects_eighty_percent_replacement_but_allows_motion(self) -> None:
        base = self.player.box
        plausible_motion_and_scale = (170.0, 70.0, 410.0, 790.0)
        horizontal_replacement = (260.0, 100.0, 460.0, 700.0)
        vertical_replacement = (100.0, 580.0, 300.0, 1180.0)

        self.assertTrue(
            self.runtime._player_boxes_associate(
                base,
                plausible_motion_and_scale,
            )
        )
        self.assertFalse(
            self.runtime._player_boxes_associate(base, horizontal_replacement)
        )
        self.assertFalse(
            self.runtime._player_boxes_associate(base, vertical_replacement)
        )

    def test_same_generation_allows_2400_px_per_second_for_fifty_ms(self) -> None:
        self.assertFalse(
            self.runtime.accept_body(
                self.player.box,
                track_generation=7,
                source_timestamp_ns=100_000_000,
            )
        )
        moved_120_pixels = (220.0, 100.0, 420.0, 700.0)

        self.assertFalse(
            self.runtime.accept_body(
                moved_120_pixels,
                track_generation=7,
                source_timestamp_ns=150_000_000,
            )
        )
        self.assertEqual(self.runtime.identity_generation, 0)

    def test_recorded_fast_overlapping_measurement_keeps_existing_head(self) -> None:
        first_ns = 100_000_000
        second_ns = first_ns + 12_515_000
        first = (
            952.9,
            514.0,
            1010.4,
            613.7,
        )
        second = (
            955.0,
            456.4,
            1027.5,
            554.9,
        )
        self.assertFalse(
            self.runtime.accept_body(
                first,
                corroboration_box=first,
                track_generation=58,
                source_timestamp_ns=first_ns,
            )
        )
        self.worker.result = _result(
            source_ns=first_ns,
            point=(980.0, 525.0),
            selected_player_box=first,
        )
        initial = self.runtime.take_latest(now_ns=first_ns + 1_000_000)
        assert initial is not None

        # The legacy radial envelope rejects this measured 3697 px/s body
        # transition even though the boxes overlap and TargetTracker retained
        # generation 58.  The widened envelope is restricted to this exact
        # physical-measurement path.
        self.assertFalse(
            self.runtime._player_boxes_associate_over_interval(
                first,
                second,
                elapsed_ns=second_ns - first_ns,
            )
        )
        self.assertTrue(
            self.runtime._player_boxes_associate_over_interval(
                first,
                second,
                elapsed_ns=second_ns - first_ns,
                maximum_speed_pixels_per_second=(
                    AUTOMATIC_HEAD_EXACT_BODY_ASSOCIATION_MAX_SPEED_PIXELS_PER_SECOND
                ),
            )
        )
        self.assertFalse(
            self.runtime.accept_body(
                second,
                corroboration_box=second,
                track_generation=58,
                source_timestamp_ns=second_ns,
            )
        )
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertEqual(
            self.runtime.anchor.identity_deadline_ns,
            initial.identity_deadline_ns,
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=second_ns))
        self.assertIsNotNone(self.runtime.visible_sample(now_ns=second_ns))

    def test_recorded_exact_disjoint_run_confirms_once_then_continues(
        self,
    ) -> None:
        self.runtime = _AutomaticHeadRuntime(
            self.worker,
            submission_hz=90.0,
            stale_after_seconds=0.110,
        )
        trusted_ns = 351_769_351_206_949
        pending_ns = 351_769_367_904_317
        confirming_ns = 351_769_376_205_776
        trusted = (
            1080.39990234375,
            514.327392578125,
            1128.1488037109375,
            578.6832275390625,
        )
        pending = (
            1035.618408203125,
            523.1901245117188,
            1078.441162109375,
            596.0577392578125,
        )
        confirming = (
            1009.535400390625,
            531.8250732421875,
            1038.920166015625,
            586.6737060546875,
        )
        generation = 70
        self.runtime.accept_body(
            trusted,
            aim_box=trusted,
            corroboration_box=trusted,
            track_generation=generation,
            source_timestamp_ns=trusted_ns,
        )
        self.worker.result = _result(
            source_ns=trusted_ns,
            point=(1104.0, 525.0),
            selected_player_box=trusted,
        )
        anchored = self.runtime.take_latest(now_ns=trusted_ns + 1_000_000)
        self.assertIsNotNone(anchored)

        # The first disjoint exact box is still quarantined and publishes no
        # coordinate.  The next shape-continuous exact box continues in the
        # same direction inside the measured-motion envelope, proving this was
        # the recorded fast traversal rather than a nearby rival.
        self.assertFalse(
            self.runtime.accept_body(
                pending,
                aim_box=pending,
                corroboration_box=pending,
                track_generation=generation,
                source_timestamp_ns=pending_ns,
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertIsNone(self.runtime.visible_sample(now_ns=pending_ns))
        self.assertFalse(
            self.runtime.accept_body(
                confirming,
                aim_box=confirming,
                corroboration_box=confirming,
                track_generation=generation,
                source_timestamp_ns=confirming_ns,
            )
        )
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(self.runtime.anchor.active)
        self.assertIsNone(
            self.runtime.take_latest(now_ns=confirming_ns + 1_000_000)
        )
        visible = self.runtime.visible_sample(now_ns=confirming_ns)
        self.assertIsNotNone(visible)
        assert visible is not None
        self.assertEqual(visible.track_generation, generation)
        self.assertEqual(visible.source_timestamp_ns, confirming_ns)
        self.assertIn(
            pending_ns,
            self.runtime._confirmed_disjoint_source_timestamps,
        )
        self.assertIn(
            confirming_ns,
            self.runtime._confirmed_disjoint_source_timestamps,
        )
        self.assertTrue(
            self.runtime._exact_measured_binding_chain_associates(
                trusted,
                confirming,
                submitted_timestamp_ns=trusted_ns,
                current_timestamp_ns=confirming_ns,
                track_generation=generation,
            )
        )

        # The recorded traversal continues with several disjoint endpoints.
        # Once the first three exact samples establish a coherent trajectory,
        # each next same-direction exact sample carries without another
        # one-frame quarantine or identity change.
        continuation = (
            (
                351_769_388_693_790,
                (973.3836669921875, 537.5855712890625,
                 999.31689453125, 587.973388671875),
            ),
            (
                351_769_401_194_343,
                (935.1617431640625, 534.6859741210938,
                 962.914794921875, 588.9052734375),
            ),
            (
                351_769_413_690_297,
                (888.41455078125, 531.9902954101562,
                 920.5862426757812, 586.4720458984375),
            ),
            (
                351_769_426_195_481,
                (861.7445068359375, 526.6485595703125,
                 892.5630493164062, 583.0189208984375),
            ),
        )
        for timestamp_ns, box in continuation:
            self.assertFalse(
                self.runtime.accept_body(
                    box,
                    aim_box=box,
                    corroboration_box=box,
                    track_generation=generation,
                    source_timestamp_ns=timestamp_ns,
                )
            )
            self.assertFalse(self.runtime.body_update_deferred)
            self.assertIsNone(
                self.runtime.take_latest(now_ns=timestamp_ns + 1_000_000)
            )
            self.assertIsNotNone(
                self.runtime.visible_sample(now_ns=timestamp_ns)
            )
        last_timestamp_ns, last_box = continuation[-1]
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(
            self.runtime._exact_measured_binding_chain_associates(
                trusted,
                last_box,
                submitted_timestamp_ns=trusted_ns,
                current_timestamp_ns=last_timestamp_ns,
                track_generation=generation,
            )
        )

    def test_disjoint_confirmation_rejects_near_perpendicular_turn(self) -> None:
        self.assertFalse(
            self.runtime._confirmed_disjoint_measured_continuation(
                (0.0, 0.0, 100.0, 200.0),
                (105.0, 0.0, 205.0, 200.0),
                (106.0, 105.0, 206.0, 305.0),
                trusted_timestamp_ns=100_000_000,
                pending_timestamp_ns=125_000_000,
                confirming_timestamp_ns=150_000_000,
                maximum_chain_interval_ns=self.runtime.stale_after_ns,
            )
        )

    def test_ordinary_overlap_ends_confirmed_disjoint_trajectory(self) -> None:
        boxes = (
            (200.0, 100.0, 240.0, 160.0),
            (150.0, 100.0, 190.0, 160.0),
            (100.0, 100.0, 140.0, 160.0),
            (95.0, 100.0, 135.0, 160.0),
            (45.0, 100.0, 85.0, 160.0),
        )
        timestamps = (
            100_000_000,
            112_000_000,
            124_000_000,
            136_000_000,
            148_000_000,
        )
        self.runtime.accept_body(
            boxes[0],
            corroboration_box=boxes[0],
            track_generation=8,
            source_timestamp_ns=timestamps[0],
        )
        self.assertFalse(
            self.runtime.accept_body(
                boxes[1],
                corroboration_box=boxes[1],
                track_generation=8,
                source_timestamp_ns=timestamps[1],
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertFalse(
            self.runtime.accept_body(
                boxes[2],
                corroboration_box=boxes[2],
                track_generation=8,
                source_timestamp_ns=timestamps[2],
            )
        )
        self.assertEqual(
            self.runtime._confirmed_disjoint_trajectory_endpoint_ns,
            timestamps[2],
        )

        self.assertFalse(
            self.runtime.accept_body(
                boxes[3],
                corroboration_box=boxes[3],
                track_generation=8,
                source_timestamp_ns=timestamps[3],
            )
        )
        self.assertIsNone(
            self.runtime._confirmed_disjoint_trajectory_endpoint_ns
        )
        self.assertFalse(
            self.runtime.accept_body(
                boxes[4],
                corroboration_box=boxes[4],
                track_generation=8,
                source_timestamp_ns=timestamps[4],
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)

    def test_predicted_disjoint_sample_cannot_confirm_exact_motion(self) -> None:
        trusted = (1080.4, 514.3, 1128.1, 578.7)
        predicted = (1035.6, 523.2, 1078.4, 596.1)
        confirming = (1009.5, 531.8, 1038.9, 586.7)
        generation = 70
        self.runtime.accept_body(
            trusted,
            aim_box=trusted,
            corroboration_box=trusted,
            track_generation=generation,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(1104.0, 525.0),
            selected_player_box=trusted,
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=101_000_000))

        self.assertFalse(
            self.runtime.accept_body(
                predicted,
                aim_box=predicted,
                track_generation=generation,
                source_timestamp_ns=117_000_000,
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertTrue(
            self.runtime.accept_body(
                confirming,
                aim_box=confirming,
                corroboration_box=confirming,
                track_generation=generation,
                source_timestamp_ns=126_000_000,
            )
        )
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.anchor.active)
        self.assertNotIn(
            117_000_000,
            self.runtime._confirmed_disjoint_source_timestamps,
        )

    def test_stale_disjoint_chain_cannot_retain_anchor(self) -> None:
        trusted = (1080.4, 514.3, 1128.1, 578.7)
        pending = (1035.6, 523.2, 1078.4, 596.1)
        confirming = (1009.5, 531.8, 1038.9, 586.7)
        generation = 70
        self.runtime.accept_body(
            trusted,
            aim_box=trusted,
            corroboration_box=trusted,
            track_generation=generation,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(1104.0, 525.0),
            selected_player_box=trusted,
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=101_000_000))

        self.assertFalse(
            self.runtime.accept_body(
                pending,
                aim_box=pending,
                corroboration_box=pending,
                track_generation=generation,
                source_timestamp_ns=170_000_000,
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertTrue(
            self.runtime.accept_body(
                confirming,
                aim_box=confirming,
                corroboration_box=confirming,
                track_generation=generation,
                source_timestamp_ns=180_000_000,
            )
        )
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.anchor.active)

    def test_first_async_head_result_follows_recorded_exact_measurement_chain(
        self,
    ) -> None:
        first_ns = 100_000_000
        second_ns = first_ns + 12_515_000
        first = (952.9, 514.0, 1010.4, 613.7)
        second = (955.0, 456.4, 1027.5, 554.9)
        self.runtime.accept_body(
            first,
            corroboration_box=first,
            track_generation=58,
            source_timestamp_ns=first_ns,
        )
        self.assertFalse(
            self.runtime.accept_body(
                second,
                corroboration_box=second,
                track_generation=58,
                source_timestamp_ns=second_ns,
            )
        )
        self.assertFalse(
            self.runtime._player_boxes_associate_over_interval(
                first,
                second,
                elapsed_ns=second_ns - first_ns,
            )
        )
        self.worker.result = _result(
            source_ns=first_ns,
            point=(980.0, 525.0),
            selected_player_box=first,
        )

        direct = self.runtime.take_latest(now_ns=second_ns + 1_000_000)
        visible = self.runtime.visible_sample(now_ns=second_ns + 1_000_000)

        self.assertIsNotNone(direct)
        self.assertIsNotNone(visible)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(self.runtime.anchor.active)
        assert visible is not None
        self.assertEqual(visible.source_timestamp_ns, second_ns)
        self.assertEqual(visible.direct_source_timestamp_ns, first_ns)
        self.assertTrue(
            self.runtime._head_point_belongs_to_player(visible.point, second)
        )

    def test_async_no_head_result_across_fast_exact_chain_keeps_anchor(
        self,
    ) -> None:
        first_ns = 100_000_000
        second_ns = first_ns + 12_515_000
        first = (952.9, 514.0, 1010.4, 613.7)
        second = (955.0, 456.4, 1027.5, 554.9)
        self.runtime.accept_body(
            first,
            corroboration_box=first,
            track_generation=58,
            source_timestamp_ns=first_ns,
        )
        self.worker.result = _result(
            source_ns=first_ns,
            point=(980.0, 525.0),
            selected_player_box=first,
        )
        initial = self.runtime.take_latest(now_ns=first_ns + 1_000_000)
        assert initial is not None
        self.runtime.accept_body(
            second,
            corroboration_box=second,
            track_generation=58,
            source_timestamp_ns=second_ns,
        )
        self.worker.result = _result(
            source_ns=first_ns,
            point=None,
            selected_player_box=first,
        )

        self.assertIsNone(
            self.runtime.take_latest(now_ns=second_ns + 1_000_000)
        )
        visible = self.runtime.visible_sample(now_ns=second_ns + 1_000_000)

        self.assertIsNotNone(visible)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(self.runtime.anchor.active)
        self.assertEqual(
            self.runtime.anchor.identity_deadline_ns,
            initial.identity_deadline_ns,
        )
        assert visible is not None
        self.assertEqual(visible.source_timestamp_ns, second_ns)
        self.assertEqual(visible.direct_source_timestamp_ns, first_ns)

    def test_exact_measurement_chain_rejects_overlapping_crossing_above_4800(
        self,
    ) -> None:
        first_ns = 100_000_000
        first = (800.0, 200.0, 1000.0, 800.0)
        crossing = (950.0, 200.0, 1150.0, 800.0)
        confirmed_crossing = (954.0, 200.0, 1154.0, 800.0)
        self.runtime.accept_body(
            first,
            corroboration_box=first,
            track_generation=61,
            source_timestamp_ns=first_ns,
        )
        self.assertTrue(self.runtime._player_boxes_associate(first, crossing))
        self.assertFalse(
            self.runtime._player_boxes_associate_over_interval(
                first,
                crossing,
                elapsed_ns=8_000_000,
                maximum_speed_pixels_per_second=(
                    AUTOMATIC_HEAD_EXACT_BODY_ASSOCIATION_MAX_SPEED_PIXELS_PER_SECOND
                ),
            )
        )

        self.assertFalse(
            self.runtime.accept_body(
                crossing,
                corroboration_box=crossing,
                track_generation=61,
                source_timestamp_ns=first_ns + 8_000_000,
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)
        self.assertTrue(
            self.runtime.accept_body(
                confirmed_crossing,
                corroboration_box=confirmed_crossing,
                track_generation=61,
                source_timestamp_ns=first_ns + 16_000_000,
            )
        )
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.anchor.active)

    def test_predicted_body_cannot_use_widened_measured_speed_envelope(self) -> None:
        first_ns = 100_000_000
        first = (952.9, 514.0, 1010.4, 613.7)
        fast_prediction = (955.0, 456.4, 1027.5, 554.9)
        self.runtime.accept_body(
            first,
            corroboration_box=first,
            track_generation=58,
            source_timestamp_ns=first_ns,
        )

        self.assertFalse(
            self.runtime.accept_body(
                fast_prediction,
                track_generation=58,
                source_timestamp_ns=first_ns + 12_515_000,
            )
        )
        self.assertTrue(self.runtime.body_update_deferred)
        self.assertEqual(self.runtime.identity_generation, 0)

    def test_predicted_frame_breaks_async_exact_measurement_chain(self) -> None:
        source_ns = 100_000_000
        source = (800.0, 200.0, 900.0, 400.0)
        predicted = (830.0, 200.0, 930.0, 400.0)
        current = (860.0, 200.0, 960.0, 400.0)
        self.runtime.accept_body(
            source,
            corroboration_box=source,
            track_generation=62,
            source_timestamp_ns=source_ns,
        )
        self.assertFalse(
            self.runtime.accept_body(
                predicted,
                track_generation=62,
                source_timestamp_ns=source_ns + 8_000_000,
            )
        )
        self.assertFalse(self.runtime.body_update_deferred)
        self.assertFalse(
            self.runtime.accept_body(
                current,
                corroboration_box=current,
                track_generation=62,
                source_timestamp_ns=source_ns + 16_000_000,
            )
        )
        self.assertFalse(self.runtime.body_update_deferred)
        # Both short endpoint steps satisfy the ordinary envelope, while the
        # whole source/current interval needs the widened one.  The intervening
        # prediction must prevent the exact-binding chain from granting it.
        self.assertFalse(
            self.runtime._player_boxes_associate_over_interval(
                source,
                current,
                elapsed_ns=16_000_000,
            )
        )
        self.assertTrue(
            self.runtime._player_boxes_associate_over_interval(
                source,
                current,
                elapsed_ns=16_000_000,
                maximum_speed_pixels_per_second=(
                    AUTOMATIC_HEAD_EXACT_BODY_ASSOCIATION_MAX_SPEED_PIXELS_PER_SECOND
                ),
            )
        )
        self.worker.result = _result(
            source_ns=source_ns,
            point=(850.0, 220.0),
            selected_player_box=source,
        )

        self.assertIsNone(
            self.runtime.take_latest(now_ns=source_ns + 17_000_000)
        )
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.body_valid)
        self.assertFalse(self.runtime.anchor.active)

    def test_crossing_target_old_head_outside_current_body_never_arms(self) -> None:
        first = (800.0, 200.0, 1000.0, 800.0)
        crossing = (950.0, 200.0, 1150.0, 800.0)
        self.runtime.accept_body(first, track_generation=1)
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(900.0, 260.0),
            selected_player_box=first,
        )

        # This close crossing remains inside the tracker's logical epoch and
        # deliberately passes the loose motion sanity check.  The direct point
        # itself must still belong to the current primary body.
        self.assertFalse(
            self.runtime.accept_body(crossing, track_generation=1)
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=108_000_000))
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.body_valid)
        self.assertIsNone(self.runtime.visible_sample(now_ns=108_000_000))

    def test_overlapping_crossing_cannot_inherit_shared_upper_head_point(self) -> None:
        first = (800.0, 200.0, 1000.0, 800.0)
        crossing = (900.0, 200.0, 1100.0, 800.0)
        self.runtime.accept_body(first, track_generation=1)
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(940.0, 260.0),
            selected_player_box=first,
        )

        # The point belongs anatomically to both overlapping players and the
        # tracker has not advanced its logical generation.  A 100 px transition
        # in eight milliseconds is nevertheless incompatible with the submitted
        # body and must revoke the entire direct-head identity.
        self.assertFalse(
            self.runtime.accept_body(crossing, track_generation=1)
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=108_000_000))
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.body_valid)

    def test_crossing_target_revokes_existing_visible_head_lease(self) -> None:
        first = (800.0, 200.0, 1000.0, 800.0)
        crossing = (950.0, 200.0, 1150.0, 800.0)
        self.runtime.accept_body(
            first,
            corroboration_box=first,
            track_generation=1,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(900.0, 260.0),
            selected_player_box=first,
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=104_000_000))

        self.assertFalse(
            self.runtime.accept_body(crossing, track_generation=1)
        )
        self.assertIsNone(self.runtime.visible_sample(now_ns=108_000_000))
        self.assertEqual(self.runtime.identity_generation, 1)
        self.assertFalse(self.runtime.body_valid)

    def test_fast_same_target_motion_keeps_head_inside_current_body(self) -> None:
        first = (800.0, 200.0, 1000.0, 800.0)
        moved_120_pixels = (920.0, 200.0, 1120.0, 800.0)
        self.runtime.accept_body(
            first,
            corroboration_box=first,
            track_generation=7,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(940.0, 260.0),
            selected_player_box=first,
        )

        self.assertFalse(
            self.runtime.accept_body(
                moved_120_pixels,
                corroboration_box=moved_120_pixels,
                track_generation=7,
                source_timestamp_ns=150_000_000,
            )
        )
        sample = self.runtime.take_latest(now_ns=150_000_000)
        assert sample is not None
        self.assertEqual(sample.point, (940.0, 260.0))
        self.assertEqual(sample.source_timestamp_ns, 100_000_000)
        self.assertEqual(sample.direct_source_timestamp_ns, 100_000_000)
        self.assertEqual(sample.corroboration_point, (900.0, 500.0))
        visible = self.runtime.visible_sample(now_ns=150_000_000)
        assert visible is not None
        self.assertEqual(visible.point, (1060.0, 260.0))
        self.assertEqual(visible.velocity_point, (1060.0, 260.0))
        self.assertEqual(visible.source_timestamp_ns, 150_000_000)
        self.assertEqual(self.runtime.identity_generation, 0)

    def test_body_map_jitter_is_filtered_for_control_while_take_latest_stays_direct_only(
        self,
    ) -> None:
        base = self.player.box
        self.runtime.accept_body(
            base,
            corroboration_box=base,
            track_generation=3,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(
            source_ns=100_000_000,
            point=(200.0, 150.0),
            selected_player_box=base,
        )
        self.assertIsNotNone(self.runtime.take_latest(now_ns=100_000_000))

        mapped_points: list[tuple[float, float]] = []
        for index, offset in enumerate((4.0, -4.0, 3.5, -3.5, 4.0, -4.0), 1):
            oscillating_box = (
                base[0] + offset,
                base[1] - offset,
                base[2] + offset,
                base[3] - offset,
            )
            self.assertFalse(
                self.runtime.accept_body(
                    oscillating_box,
                    aim_box=oscillating_box,
                    corroboration_box=oscillating_box,
                    track_generation=3,
                    source_timestamp_ns=100_000_000 + index * 8_000_000,
                )
            )
            now_ns = 100_000_000 + index * 8_000_000
            self.assertIsNone(self.runtime.take_latest(now_ns=now_ns))
            visible = self.runtime.visible_sample(
                now_ns=now_ns
            )
            assert visible is not None
            mapped_points.append(visible.point)
            self.assertIs(
                visible.provenance,
                DirectHeadProvenance.MEASURED_PRIMARY,
            )
            self.assertTrue(visible.body_derived_motion_permitted)
            self.assertEqual(
                visible.body_derived_motion_deadline_ns,
                visible.identity_deadline_ns,
            )

        # The controller-eligible mapped anchor follows filtered measured body
        # geometry, while take_latest remains new-direct-result-only.
        self.assertTrue(
            any(point != (200.0, 150.0) for point in mapped_points)
        )

    def test_nearby_tracker_prediction_retains_bounded_direct_head_lease(self) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(source_ns=100_000_000)
        original = self.runtime.take_latest(now_ns=110_000_000)
        assert original is not None

        predicted_box = (106.0, 102.0, 306.0, 702.0)
        self.assertFalse(
            self.runtime.accept_body(
                predicted_box,
                aim_box=predicted_box,
                corroboration_box=None,
                track_generation=0,
                source_timestamp_ns=130_000_000,
            )
        )
        self.assertIsNone(self.runtime.take_latest(now_ns=130_000_000))
        self.assertEqual(self.runtime.identity_generation, 0)
        retained = self.runtime.visible_sample(now_ns=130_000_000)
        assert retained is not None
        self.assertNotEqual(retained.point, original.point)
        self.assertTrue(retained.bridging)
        self.assertIs(
            retained.provenance,
            DirectHeadProvenance.PREDICTED_PRIMARY,
        )
        self.assertEqual(
            retained.direct_source_timestamp_ns,
            original.direct_source_timestamp_ns,
        )
        self.assertFalse(retained.body_derived_motion_permitted)
        self.assertIsNone(retained.body_derived_motion_deadline_ns)
        self.assertIsNone(retained.corroboration_point)

    def test_controller_loss_publication_is_idempotent_within_one_frame(self) -> None:
        controller = mock.Mock()
        published = _publish_automatic_head_loss_once(
            controller,
            (1080, 1920, 3),
            source_timestamp_ns=123,
            already_published=False,
        )
        published = _publish_automatic_head_loss_once(
            controller,
            (1080, 1920, 3),
            source_timestamp_ns=123,
            already_published=published,
        )

        self.assertTrue(published)
        controller.update.assert_called_once_with(
            None,
            (1080, 1920, 3),
            active=True,
            measurement_ns=123,
        )

    def test_worker_cleanup_is_bounded_and_failure_is_exposed(self) -> None:
        self.runtime.start()
        self.assertTrue(self.runtime.stop())
        self.assertTrue(self.worker.stopped)
        self.worker.failed = True
        with self.assertRaisesRegex(RuntimeError, "synthetic head failure"):
            self.runtime.raise_if_failed()


class PreparedDirectHeadLocalizerTests(unittest.TestCase):
    def test_no_decoded_head_returns_typed_rejection_without_box_proxy(self) -> None:
        prepared = SimpleNamespace(
            tensor=np.zeros((1, 3, 320, 320), np.float32),
            transform=object(),
        )
        payload = _TimestampedPreparedHeadInput(prepared, 777)
        session = SimpleNamespace(infer=mock.Mock(return_value=np.zeros((1, 6, 2100))))
        localizer = _PreparedDirectHeadLocalizer(session)

        with mock.patch(
            "detection.head_detector.PreparedHeadInput",
            SimpleNamespace,
        ), mock.patch(
            "detection.head_detector.decode_head_output",
            return_value=[],
        ), mock.patch(
            "detection.head_detector.associate_head_to_player_outcome",
            return_value=HeadAssociationOutcome(
                HeadLocalizationReason.NO_DECODED_HEAD_CANDIDATE,
                None,
            ),
        ):
            outcome = localizer(payload, (100.0, 100.0, 300.0, 700.0))

        self.assertIsInstance(outcome, HeadLocalizationOutcome)
        self.assertIs(
            outcome.reason,
            HeadLocalizationReason.NO_DECODED_HEAD_CANDIDATE,
        )
        self.assertIsNone(outcome.observation)

    def test_localization_is_wrapped_as_direct_evidence(self) -> None:
        prepared = SimpleNamespace(
            tensor=np.zeros((1, 3, 320, 320), np.float32),
            transform=object(),
        )
        session = SimpleNamespace(infer=mock.Mock(return_value=np.zeros((1, 6, 2100))))
        localization = HeadLocalization(
            point=(400.0, 210.0),
            source_timestamp_ns=777,
            confidence=0.91,
            head_box=(380.0, 190.0, 420.0, 230.0),
            containment=1.0,
            candidate_index=0,
            supporting_player_index=None,
        )
        localizer = _PreparedDirectHeadLocalizer(session)
        payload = _TimestampedPreparedHeadInput(prepared, 777)

        with mock.patch(
            "detection.head_detector.PreparedHeadInput",
            SimpleNamespace,
        ), mock.patch(
            "detection.head_detector.decode_head_output",
            return_value=[object()],
        ), mock.patch(
            "detection.head_detector.associate_head_to_player_outcome",
            return_value=HeadAssociationOutcome(
                HeadLocalizationReason.LOCALIZED,
                localization,
            ),
        ):
            outcome = localizer(
                payload,
                (100.0, 100.0, 500.0, 800.0),
            )

        self.assertIs(outcome.reason, HeadLocalizationReason.LOCALIZED)
        observation = outcome.observation
        assert observation is not None
        self.assertEqual(observation.point, localization.point)
        self.assertEqual(observation.head_box, localization.head_box)
        self.assertIn("direct head box", observation.evidence)


class AimDiagnosticControlTests(unittest.TestCase):
    def test_snapshot_serializes_current_output_and_bounded_recent_commands(
        self,
    ) -> None:
        output = CalibratedControlOutput(
            timestamp_ns=900,
            rate_x_counts_per_second=1200.0,
            rate_y_counts_per_second=-300.0,
            target_velocity_x_pixels_per_second=450.0,
            target_velocity_y_pixels_per_second=-80.0,
            projected_error_x_pixels=12.0,
            projected_error_y_pixels=-3.0,
            valid=True,
            velocity_feedforward_confidence_x=0.75,
            position_channel_agreement=0.25,
            position_feedback_confidence_x=1.0,
            ambiguous_lookahead_projection_retained_x=True,
        )
        commands = tuple(
            MakcuNormalCommandRecord(index, 1_000 + index, 1, -1)
            for index in range(1, 66)
        )
        normal = MakcuNormalControlSnapshot(
            captured_ns=2_000,
            connection_epoch=3,
            calibrated_output=output,
            successful_commands=65,
            emitted_x=65,
            emitted_y=-65,
            emitted_abs_x=65,
            emitted_abs_y=65,
            first_emitted_ns=1_001,
            last_emitted_ns=1_065,
            commands=commands,
        )
        controller = SimpleNamespace(
            normal_control_snapshot=lambda: normal,
            telemetry_snapshot=lambda: MakcuTelemetrySnapshot(
                movement_commands=65,
                control_samples=7,
            ),
        )

        diagnostic = _aim_diagnostic_makcu_control(controller)

        assert diagnostic is not None
        self.assertEqual(diagnostic["successful_commands"], 65)
        recent = diagnostic["recent_commands"]
        assert isinstance(recent, list)
        self.assertEqual(len(recent), 64)
        self.assertEqual(recent[0]["sequence"], 2)
        self.assertEqual(
            diagnostic["calibrated_output"][
                "velocity_feedforward_confidence_x"
            ],
            0.75,
        )
        self.assertEqual(
            diagnostic["calibrated_output"]["position_channel_agreement"],
            0.25,
        )
        self.assertEqual(
            diagnostic["calibrated_output"]["position_feedback_confidence_x"],
            1.0,
        )
        self.assertTrue(
            diagnostic["calibrated_output"][
                "ambiguous_lookahead_projection_retained_x"
            ]
        )
        self.assertEqual(
            diagnostic["cumulative_telemetry"]["control_samples"],
            7,
        )
        json.dumps(diagnostic, allow_nan=False)


class AutomaticHeadBuilderTests(unittest.TestCase):
    def test_exact_gpu_contract_is_warmed_and_worker_owns_prepared_payload(self) -> None:
        sessions: list[object] = []
        workers: list[object] = []

        class FakeSession:
            def __init__(self, path, contract, *, provider):
                self.path = Path(path)
                self.contract = contract
                self.provider = provider
                self.info = SimpleNamespace(provider=provider)
                self.inputs: list[np.ndarray] = []
                sessions.append(self)

            def infer(self, tensor):
                self.inputs.append(tensor)
                return np.zeros(self.contract.output.shape, dtype=np.float32)

        class FakeLatestWorker(_FakeWorker):
            def __init__(self, localize, *, payload_copier):
                super().__init__()
                self.localize = localize
                self.payload_copier = payload_copier
                workers.append(self)

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "sunxds_0.8.0.onnx"
            model_path.write_bytes(b"fake")
            with (
                mock.patch(
                    "detection.head_detector.runtime_head_model_spec",
                    return_value=HeadModelSpec(
                        path=model_path,
                        input_height=320,
                        input_width=320,
                        output_attributes=6,
                        output_candidates=2100,
                        model_name="SunXDS 0.8.0",
                        evidence_label="SunXDS 0.8.0 direct head box",
                    ),
                ),
                mock.patch(
                    "detection.head_worker.StrictProviderOnnxSession",
                    FakeSession,
                ),
                mock.patch(
                    "detection.head_worker.LatestHeadWorker",
                    FakeLatestWorker,
                ),
            ):
                runtime = _build_automatic_head_runtime(
                    _strict_primary_runtime_summary()
                )

        self.assertIsInstance(runtime, _AutomaticHeadRuntime)
        self.assertEqual(len(sessions), 1)
        session = sessions[0]
        self.assertEqual(session.path.name, "sunxds_0.8.0.onnx")
        self.assertEqual(session.provider, AUTOMATIC_HEAD_PROVIDER)
        self.assertEqual(runtime.provider, AUTOMATIC_HEAD_PROVIDER)
        self.assertEqual(session.contract.input.name, "images")
        self.assertEqual(session.contract.input.shape, (1, 3, 320, 320))
        self.assertEqual(session.contract.output.name, "output0")
        self.assertEqual(session.contract.output.shape, (1, 6, 2100))
        self.assertEqual(len(session.inputs), 1)
        self.assertEqual(session.inputs[0].dtype, np.float32)
        self.assertIs(workers[0].payload_copier("owned"), "owned")

    def test_builder_rejects_any_non_strict_primary_before_model_load(self) -> None:
        cases = {
            "wrong_provider": {
                **_strict_primary_runtime_summary(),
                "active_providers": ["CPUExecutionProvider"],
            },
            "full_provider_disabled": {
                **_strict_primary_runtime_summary(),
                "require_full_provider": False,
            },
            "full_provider_missing": {
                key: value
                for key, value in _strict_primary_runtime_summary().items()
                if key != "require_full_provider"
            },
            "cpu_graph_fallback_enabled": {
                **_strict_primary_runtime_summary(),
                "configured_session_options": {
                    "disable_cpu_ep_fallback": False,
                },
            },
            "cpu_graph_fallback_missing": {
                **_strict_primary_runtime_summary(),
                "configured_session_options": {},
            },
            "ep_failure_fallback_enabled": {
                **_strict_primary_runtime_summary(),
                "runtime_ep_fail_fallback_disabled": False,
            },
            "ep_failure_fallback_missing": {
                key: value
                for key, value in _strict_primary_runtime_summary().items()
                if key != "runtime_ep_fail_fallback_disabled"
            },
        }
        for name, summary in cases.items():
            with (
                self.subTest(name=name),
                mock.patch(
                    "detection.head_detector.verify_pinned_head_model",
                    side_effect=AssertionError(
                        "model loaded before provider validation"
                    ),
                ),
                self.assertRaises(RuntimeError),
            ):
                _build_automatic_head_runtime(summary)


class HeadRuntimeTelemetryTests(unittest.TestCase):
    def test_summary_reports_only_counts_and_freshness_not_coordinates(self) -> None:
        fields = dict(
            jobs_completed=7,
            localized_heads=1,
            no_head_results=6,
            no_decoded_head_candidates=1,
            no_plausible_heads=1,
            multiple_plausible_heads=1,
            no_matching_secondary_players=1,
            multiple_matching_secondary_players=1,
            head_unsupported_by_matched_player=1,
            unspecified_no_head_results=0,
            pending_overwrites=1,
            result_overwrites=2,
            stale_submissions=1,
            stale_pending_dropped=1,
            stale_results_dropped=1,
        )
        previous = SimpleNamespace(**{name: 0 for name in fields})
        current = SimpleNamespace(**fields)
        sample = SimpleNamespace(
            source_timestamp_ns=1_980_000_000,
            bridging=True,
        )

        summary = _head_runtime_telemetry_summary(
            previous,
            current,
            1.0,
            now_ns=2_000_000_000,
            visible_sample=sample,
        )

        self.assertIn("HEAD completed 7/s", summary)
        self.assertIn("localized 1/s", summary)
        self.assertIn("no-head 6/s", summary)
        self.assertIn(
            "why no-decoded/no-plausible/multi-head 1/1/1/s",
            summary,
        )
        self.assertIn("secondary none/multi/unsupported 1/1/1/s", summary)
        self.assertIn("other 0/s", summary)
        self.assertIn("overwrites 3", summary)
        self.assertIn("stale 3", summary)
        self.assertIn("point age 20ms bridge", summary)
        self.assertNotIn("321", summary)


if __name__ == "__main__":
    unittest.main()
