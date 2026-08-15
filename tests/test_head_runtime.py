from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from detection.head_detector import (
    HeadCandidate,
    HeadLocalization,
    PreparedHeadInput,
    associate_head_to_player,
)
from detection.head_worker import HeadObservation, HeadWorkerResult
from detection.types import Detection
from main import (
    AUTOMATIC_HEAD_LOCALIZATION_HZ,
    AUTOMATIC_HEAD_PROVIDER,
    _AutomaticHeadRuntime,
    _PreparedDirectHeadLocalizer,
    _TimestampedPreparedHeadInput,
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
) -> HeadWorkerResult:
    observation = (
        None
        if point is None
        else HeadObservation(point, 0.8, "direct test head box")
    )
    return HeadWorkerResult(
        submission_id=1,
        source_timestamp_ns=source_ns,
        completed_timestamp_ns=source_ns + 1,
        identity_generation=generation,
        selected_player_box=selected_player_box,
        observation=observation,
    )


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

    def test_submission_gate_prepares_only_owned_bounded_tensor(self) -> None:
        self.runtime.start()
        self.runtime.accept_body(self.player.box)

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
                runtime.accept_body(self.player.box)
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
                        runtime.submit(
                            self.frame,
                            self.player,
                            source_timestamp_ns=round(
                                index * 1_000_000_000 / source_fps
                            ),
                        )

                self.assertGreaterEqual(len(worker.submissions), 599)
                self.assertLessEqual(len(worker.submissions), 601)

    def test_default_submission_gate_preserves_90_hz_phase_at_130_fps(self) -> None:
        self.assertEqual(AUTOMATIC_HEAD_LOCALIZATION_HZ, 90.0)
        self.assertEqual(AUTOMATIC_HEAD_PROVIDER, "MIGraphXExecutionProvider")
        worker = _FakeWorker()
        runtime = _AutomaticHeadRuntime(worker)
        runtime.accept_body(self.player.box)
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
                runtime.submit(
                    self.frame,
                    self.player,
                    source_timestamp_ns=round(index * 1_000_000_000 / 130.0),
                )

        self.assertGreaterEqual(len(worker.submissions), 899)
        self.assertLessEqual(len(worker.submissions), 901)

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

    def test_direct_head_without_corroboration_still_publishes_position(self) -> None:
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

        sample = self.runtime.take_latest(now_ns=118_000_000)

        assert sample is not None
        self.assertEqual(sample.point, (200.0, 150.0))
        self.assertIsNone(sample.corroboration_point)
        self.assertEqual(self.runtime.visible_sample(now_ns=118_000_000), sample)
        self.assertTrue(
            self.runtime.consume_motion_corroboration_revocation()
        )

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
        # This is intentionally unrelated to the body-box ratio proxy.
        self.assertNotEqual(sample.point, (200.0, 172.0))

    def test_same_source_current_body_center_is_independent_corroboration(
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
        # The previous crop box center was (200, 400); it must never masquerade
        # as same-source motion evidence for the current image.
        self.assertEqual(sample.corroboration_point, (210.0, 400.0))

    def test_wrong_timestamp_or_predicted_body_only_withholds_corroboration(self) -> None:
        cases = (
            (101_000_000, self.player.box),
            (100_000_000, None),
        )
        for body_timestamp_ns, corroboration_box in cases:
            with self.subTest(
                body_timestamp_ns=body_timestamp_ns,
                corroboration=corroboration_box is not None,
            ):
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

                sample = runtime.take_latest(now_ns=110_000_000)
                assert sample is not None
                self.assertEqual(sample.point, (200.0, 150.0))
                self.assertIsNone(sample.corroboration_point)
                self.assertTrue(
                    runtime.consume_motion_corroboration_revocation()
                )

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
        self.worker.result = _result(source_ns=108_000_000, point=None)
        self.assertIsNone(self.runtime.take_latest(now_ns=110_000_000))
        bridged = self.runtime.visible_sample(now_ns=110_000_000)

        assert bridged is not None
        self.assertTrue(bridged.bridging)
        self.assertEqual(bridged.point, observed.point)
        self.assertIsNone(bridged.corroboration_point)

    def test_inflight_exact_head_during_current_prediction_has_no_corroboration(
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

        sample = self.runtime.take_latest(now_ns=110_000_000)

        assert sample is not None
        self.assertEqual(sample.point, (200.0, 150.0))
        self.assertIsNone(sample.corroboration_point)
        self.assertTrue(sample.bridging)
        self.assertTrue(
            self.runtime.visible_sample(now_ns=110_000_000).bridging
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
        predicted = self.runtime.visible_sample(now_ns=110_000_000)
        assert predicted is not None
        self.assertTrue(predicted.bridging)
        self.assertIsNone(predicted.corroboration_point)
        self.assertTrue(
            self.runtime.consume_motion_corroboration_revocation()
        )

        # The primary body is physical again, but no newer direct-head result
        # exists. Status must continue to report the old point as a bridge and
        # cannot resurrect its source-100 corroboration record.
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            track_generation=2,
            source_timestamp_ns=116_000_000,
        )
        body_only_reacquired = self.runtime.visible_sample(now_ns=118_000_000)
        assert body_only_reacquired is not None
        self.assertTrue(body_only_reacquired.bridging)
        self.assertIsNone(body_only_reacquired.corroboration_point)
        self.assertEqual(
            body_only_reacquired.source_timestamp_ns,
            observed.source_timestamp_ns,
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
        plan_crop.assert_called_once_with(self.frame.shape, current)
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

    def test_explicit_no_head_truthfully_bridges_only_existing_fresh_lease(self) -> None:
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            source_timestamp_ns=100_000_000,
        )
        self.worker.result = _result(source_ns=100_000_000)
        original = self.runtime.take_latest(now_ns=110_000_000)
        assert original is not None
        self.runtime.accept_body(
            self.player.box,
            corroboration_box=self.player.box,
            source_timestamp_ns=120_000_000,
        )
        self.worker.result = _result(source_ns=120_000_000, point=None)

        self.assertIsNone(self.runtime.take_latest(now_ns=125_000_000))
        bridged = self.runtime.visible_sample(now_ns=125_000_000)
        assert bridged is not None
        self.assertTrue(bridged.bridging)
        self.assertEqual(bridged.point, original.point)
        self.assertEqual(
            bridged.source_timestamp_ns,
            original.source_timestamp_ns,
        )
        self.assertTrue(
            self.runtime.consume_motion_corroboration_revocation()
        )
        self.assertIsNone(self.runtime.visible_sample(now_ns=165_000_001))

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

    def test_late_result_from_geometrically_replaced_player_never_arms(self) -> None:
        self.runtime.accept_body(self.player.box)
        self.worker.result = _result(source_ns=100_000_000)

        replacement = (900.0, 100.0, 1100.0, 700.0)
        self.assertTrue(self.runtime.accept_body(replacement))
        self.assertEqual(self.runtime.identity_generation, 1)
        self.worker.result = _result(source_ns=101_000_000, generation=0)
        self.assertIsNone(self.runtime.take_latest(now_ns=110_000_000))
        self.assertIsNone(self.runtime.visible_sample(now_ns=110_000_000))

    def test_track_generation_rejects_reacquisition_even_with_same_geometry(self) -> None:
        self.assertFalse(
            self.runtime.accept_body(self.player.box, track_generation=1)
        )
        self.assertFalse(
            self.runtime.accept_body(self.player.box, track_generation=1)
        )
        self.assertTrue(
            self.runtime.accept_body(self.player.box, track_generation=2)
        )
        self.assertEqual(self.runtime.identity_generation, 1)

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
        self.assertEqual(self.runtime.identity_generation, 0)

    def test_stationary_head_is_not_reprojected_by_oscillating_primary_boxes(
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

        points: list[tuple[float, float]] = []
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
                    track_generation=3,
                    source_timestamp_ns=100_000_000 + index * 8_000_000,
                )
            )
            sample = self.runtime.visible_sample(
                now_ns=100_000_000 + index * 8_000_000
            )
            assert sample is not None
            points.append(sample.point)

        # The direct head remains exactly fixed, hence far below the 3 px
        # controller deadband and incapable of producing an orbit command.
        self.assertEqual(points, [(200.0, 150.0)] * len(points))

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
        self.assertFalse(self.runtime.accept_body(predicted_box))
        self.assertEqual(self.runtime.identity_generation, 0)
        retained = self.runtime.visible_sample(now_ns=130_000_000)
        assert retained is not None
        self.assertEqual(retained.point, original.point)
        self.assertTrue(retained.bridging)
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
    def test_no_decoded_head_returns_none_without_box_proxy(self) -> None:
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
            "detection.head_detector.associate_head_to_player",
            return_value=None,
        ):
            self.assertIsNone(
                localizer(payload, (100.0, 100.0, 300.0, 700.0))
            )

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
            "detection.head_detector.associate_head_to_player",
            return_value=localization,
        ):
            observation = localizer(
                payload,
                (100.0, 100.0, 500.0, 800.0),
            )

        assert observation is not None
        self.assertEqual(observation.point, localization.point)
        self.assertIn("direct head box", observation.evidence)


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

        with (
            mock.patch(
                "detection.head_detector.verify_pinned_head_model",
                return_value=Path("/pinned/sunxds_0.8.0.onnx"),
            ),
            mock.patch(
                "detection.head_worker.StrictProviderOnnxSession",
                FakeSession,
            ),
            mock.patch("detection.head_worker.LatestHeadWorker", FakeLatestWorker),
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
            jobs_completed=10,
            localized_heads=7,
            no_head_results=3,
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

        self.assertIn("HEAD completed 10/s", summary)
        self.assertIn("localized 7/s", summary)
        self.assertIn("no-head 3/s", summary)
        self.assertIn("overwrites 3", summary)
        self.assertIn("stale 3", summary)
        self.assertIn("point age 20ms bridge", summary)
        self.assertNotIn("321", summary)


if __name__ == "__main__":
    unittest.main()
