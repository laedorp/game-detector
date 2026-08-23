from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.replay_aim_diagnostic import replay_session
from utils.aim_diagnostic import AimDiagnosticConfig, AimDiagnosticRecorder


class AimDiagnosticReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def detection(box=(100.0, 100.0, 200.0, 400.0)) -> dict[str, object]:
        return {
            "class_id": 0,
            "class_name": "player",
            "confidence": 0.9,
            "box": list(box),
        }

    def make_session(self) -> Path:
        recorder = AimDiagnosticRecorder(
            AimDiagnosticConfig(
                output_root=self.root,
                sample_hz=60.0,
                max_duration_seconds=1.0,
            ),
            metadata={
                "aim_label": "player",
                "head_ratio": 0.12,
                "tracking_mode": "stable-body",
            },
            image_encoder=lambda image, _quality: bytes(image.reshape(-1)),
        )
        recorder.start()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        first = {
            "frame_sequence": 1,
            "source_timestamp_ns": 1_000_000_000,
            "frame_shape": [480, 640, 3],
            "tracking_mode": "stable-body",
            "activation_pressed": True,
            "self_exclusion_ready": True,
            "hard_guard_revoked_prediction_grace": False,
            "aim_candidates": [self.detection()],
            "continuation_candidates": [],
            "accepted_measurement": self.detection(),
            "selected_target": self.detection(),
            "selected_is_prediction": False,
            "direct_head_sample": None,
            "control_source": "stable-body",
        }
        second = {
            **first,
            "frame_sequence": 2,
            "source_timestamp_ns": 1_016_666_667,
            "aim_candidates": [],
            "accepted_measurement": None,
            "selected_is_prediction": True,
            "control_source": "predicted-body",
        }
        self.assertTrue(recorder.submit(frame, first))
        self.assertTrue(recorder.submit(frame, second))
        self.assertTrue(recorder.stop())
        return recorder.session_dir

    @staticmethod
    def read_records(session: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in (session / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    @staticmethod
    def rewrite_session(
        session: Path,
        records: list[dict[str, object]],
        *,
        tracking_mode: str | None = None,
    ) -> None:
        (session / "records.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        manifest_path = session / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["statistics"]["written"] = len(records)
        if tracking_mode is not None:
            manifest["metadata"]["tracking_mode"] = tracking_mode
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_replay_validates_artifacts_and_matches_tracker_decisions(self) -> None:
        report = replay_session(self.make_session())

        self.assertEqual(report["schema"], "proaim.aim_diagnostic.replay")
        self.assertTrue(report["trace_complete"])
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["replay_divergences"], 0)
        self.assertEqual(report["activated_frames"], 2)
        self.assertEqual(report["primary_measurement_rate"], 0.5)
        self.assertEqual(report["target_output_rate"], 1.0)
        self.assertEqual(report["controller_target_publication_rate"], 1.0)
        self.assertEqual(report["control_availability_rate"], 1.0)
        self.assertEqual(report["new_direct_head_sample_rate"], 0.0)
        self.assertEqual(report["visible_head_anchor_coverage"], 0.0)
        self.assertEqual(report["visible_head_phase_metadata_frames"], 0)
        self.assertEqual(report["phase_advanced_frames"], 0)
        self.assertIsNone(report["phase_hops_p95"])
        self.assertEqual(report["makcu_control_frames"], 0)
        self.assertEqual(report["makcu_successful_commands"], 0)
        self.assertEqual(report["makcu_ledger_commands"], 0)
        self.assertIsNone(report["makcu_command_trace_coverage"])
        self.assertEqual(report["makcu_calibrated_output_summary"]["samples"], 0)
        self.assertEqual(report["makcu_cumulative_telemetry_totals"], {})
        self.assertEqual(report["status"], "insufficient-data")

    def test_direct_hold_is_a_publication_not_an_authorization_gate(self) -> None:
        session = self.make_session()
        records = self.read_records(session)
        for record in records:
            record["control_source"] = "direct-hold"
        self.rewrite_session(session, records)

        report = replay_session(session)

        self.assertEqual(report["controller_target_publications"], 2)
        self.assertEqual(report["controller_target_publication_rate"], 1.0)
        self.assertEqual(
            report["legacy_metric_aliases"]["control_availability_rate"],
            "controller_target_publication_rate",
        )
        self.assertNotIn("control_availability_rate", report["failed_gates"])
        self.assertNotIn("inspect-authorization-gates", report["recommendations"])

    def test_effective_per_candidate_self_safety_overrides_raw_filter_wait(self) -> None:
        session = self.make_session()
        records = self.read_records(session)
        for record in records:
            record["self_exclusion_ready"] = False
            record["aim_self_exclusion_safe"] = True
        self.rewrite_session(session, records)

        report = replay_session(session)

        self.assertEqual(report["replay_divergences"], 0)
        self.assertEqual(report["prediction_divergences"], 0)

    def test_visible_anchor_coverage_outlives_new_direct_samples(self) -> None:
        session = self.make_session()
        records = self.read_records(session)
        records[0].update(
            {
                "control_source": "direct-head",
                "direct_head_sample": {"point": [150.0, 120.0]},
                "visible_head_sample": {"point": [150.0, 120.0]},
            }
        )
        records[1].update(
            {
                "control_source": "carried-head",
                "direct_head_sample": None,
                "visible_head_sample": {
                    "point": [150.0, 120.0],
                    "bridging": True,
                },
            }
        )
        self.rewrite_session(session, records, tracking_mode="direct-head")

        report = replay_session(session)

        self.assertEqual(report["controller_target_publication_rate"], 1.0)
        self.assertEqual(report["new_direct_head_sample_rate"], 0.5)
        self.assertEqual(report["visible_head_anchor_coverage"], 1.0)
        self.assertNotIn("direct_head_rate", report["failed_gates"])
        self.assertNotIn("visible_head_anchor_coverage", report["failed_gates"])
        self.assertIn("new_direct_head_sample_rate", report["non_gating_metrics"])

    def test_replay_summarizes_optional_visible_head_phase_metadata(self) -> None:
        session = self.make_session()
        records = self.read_records(session)
        records[0].update(
            {
                "control_source": "direct-head",
                "direct_head_sample": {"point": [150.0, 120.0]},
                "visible_head_sample": {
                    "point": [151.0, 120.0],
                    "phase_advanced": True,
                    "phase_hops": 2,
                },
            }
        )
        records[1].update(
            {
                "control_source": "carried-head",
                "visible_head_sample": {
                    "point": [151.0, 120.0],
                    "bridging": True,
                    "phase_advanced": False,
                    "phase_hops": 0,
                },
            }
        )
        self.rewrite_session(session, records, tracking_mode="direct-head")

        report = replay_session(session)

        self.assertEqual(report["visible_head_anchor_coverage"], 1.0)
        self.assertEqual(report["visible_head_phase_metadata_frames"], 2)
        self.assertEqual(report["visible_head_phase_partial_frames"], 0)
        self.assertEqual(report["visible_head_phase_metadata_coverage"], 1.0)
        self.assertEqual(report["phase_advanced_frames"], 1)
        self.assertEqual(report["phase_advanced_rate"], 0.5)
        self.assertEqual(report["phase_hops_total"], 2)
        self.assertEqual(report["phase_hops_mean"], 1.0)
        self.assertEqual(report["phase_hops_p50"], 1.0)
        self.assertAlmostEqual(report["phase_hops_p95"], 1.9)
        self.assertEqual(report["phase_hops_max"], 2.0)

    def test_replay_rejects_malformed_visible_head_phase_metadata(self) -> None:
        for field, value, message in (
            ("phase_advanced", 1, "phase_advanced"),
            ("phase_hops", True, "phase_hops"),
            ("phase_hops", -1, "phase_hops"),
        ):
            with self.subTest(field=field, value=value):
                session = self.make_session()
                records = self.read_records(session)
                records[0]["visible_head_sample"] = {
                    "point": [150.0, 120.0],
                    field: value,
                }
                self.rewrite_session(session, records)

                with self.assertRaisesRegex(ValueError, message):
                    replay_session(session)

    def test_explicit_visible_sample_overrides_legacy_status_fallback(self) -> None:
        session = self.make_session()
        records = self.read_records(session)
        for record in records:
            record["direct_head_sample"] = {"point": [150.0, 120.0]}
            record["visible_head_sample"] = None
            record["aim_status"] = "aim anchored: legacy text must not win"
        self.rewrite_session(session, records, tracking_mode="direct-head")

        report = replay_session(session)

        self.assertEqual(report["new_direct_head_sample_rate"], 1.0)
        self.assertEqual(report["visible_head_anchor_coverage"], 0.0)
        self.assertIn("visible_head_anchor_coverage", report["failed_gates"])
        self.assertNotIn("direct_head_rate", report["failed_gates"])

    def test_v1_status_fallback_recovers_anchor_and_bridge_coverage(self) -> None:
        session = self.make_session()
        records = self.read_records(session)
        records[0].update(
            {
                "direct_head_sample": {"point": [150.0, 120.0]},
                "aim_status": "aim anchored: direct head carried by player",
            }
        )
        records[1].update(
            {
                "direct_head_sample": None,
                "aim_status": "aim bridge visible: primary is predicted",
            }
        )
        self.rewrite_session(session, records, tracking_mode="direct-head")

        report = replay_session(session)

        self.assertEqual(report["new_direct_head_sample_rate"], 0.5)
        self.assertEqual(report["visible_head_anchor_coverage"], 1.0)
        self.assertNotIn("visible_head_anchor_coverage", report["failed_gates"])

    def test_gaps_do_not_span_released_activation_epochs(self) -> None:
        session = self.make_session()
        first = self.read_records(session)[0]
        records: list[dict[str, object]] = []
        for sequence, timestamp, activation in (
            (1, 1_000_000_000, True),
            (2, 1_010_000_000, True),
            (3, 1_080_000_000, False),
            (4, 1_090_000_000, True),
            (5, 1_110_000_000, True),
        ):
            record = dict(first)
            record.update(
                {
                    "frame_sequence": sequence,
                    "source_timestamp_ns": timestamp,
                    "activation_pressed": activation,
                    "aim_candidates": [self.detection()] if activation else [],
                    "accepted_measurement": (
                        self.detection() if activation else None
                    ),
                    "selected_target": self.detection() if activation else None,
                    "selected_is_prediction": False,
                    "direct_head_sample": (
                        {"point": [150.0, 120.0]} if activation else None
                    ),
                    "control_source": "direct-head" if activation else "none",
                }
            )
            records.append(record)
        self.rewrite_session(session, records, tracking_mode="direct-head")

        report = replay_session(session)

        self.assertEqual(report["primary_gap_max_ms"], 20.0)
        self.assertEqual(report["new_direct_head_gap_max_ms"], 20.0)
        self.assertEqual(report["direct_head_gap_max_ms"], 20.0)
        self.assertAlmostEqual(report["primary_gap_p95_ms"], 19.5)
        self.assertAlmostEqual(report["new_direct_head_gap_p95_ms"], 19.5)

    def test_replay_deduplicates_makcu_ledger_and_aggregates_control(self) -> None:
        session = self.make_session()
        records = self.read_records(session)

        def command(
            sequence: int,
            timestamp_ns: int,
            delta_x: int,
            delta_y: int,
        ) -> dict[str, int]:
            return {
                "sequence": sequence,
                "timestamp_ns": timestamp_ns,
                "delta_x_counts": delta_x,
                "delta_y_counts": delta_y,
            }

        first_command = command(1, 1_000_100_000, 2, -1)
        overlapping_command = command(2, 1_001_100_000, -1, 3)
        last_command = command(3, 1_003_100_000, 4, -2)
        records[0]["makcu_control"] = {
            "captured_ns": 1_005_000_000,
            "connection_epoch": 7,
            "successful_commands": 2,
            "emitted_x": 1,
            "emitted_y": 2,
            "emitted_abs_x": 3,
            "emitted_abs_y": 4,
            "first_emitted_ns": first_command["timestamp_ns"],
            "last_emitted_ns": overlapping_command["timestamp_ns"],
            "dropped_commands": 0,
            "recent_commands": [first_command, overlapping_command],
            "calibrated_output": {
                "timestamp_ns": 1_004_000_000,
                "rate_x_counts_per_second": 1000.0,
                "target_velocity_x_pixels_per_second": 100.0,
                "projected_error_x_pixels": 10.0,
                "velocity_feedforward_confidence_x": 0.5,
                "valid": True,
                "saturated_x": False,
            },
            "cumulative_telemetry": {
                "output_ticks": 10,
                "authorized_ticks": 5,
                "movement_commands": 2,
                "control_samples": 2,
                "target_velocity_abs_x_pixels_per_second": 300.0,
                "saturated_x_samples": 1,
            },
        }
        records[1]["makcu_control"] = {
            "captured_ns": 1_015_000_000,
            "connection_epoch": 7,
            "successful_commands": 3,
            "emitted_x": 5,
            "emitted_y": 0,
            "emitted_abs_x": 7,
            "emitted_abs_y": 6,
            "first_emitted_ns": first_command["timestamp_ns"],
            "last_emitted_ns": last_command["timestamp_ns"],
            "dropped_commands": 0,
            "recent_commands": [overlapping_command, last_command],
            "calibrated_output": {
                "timestamp_ns": 1_014_000_000,
                "rate_x_counts_per_second": 2000.0,
                "target_velocity_x_pixels_per_second": 200.0,
                "projected_error_x_pixels": 20.0,
                "velocity_feedforward_confidence_x": 0.75,
                "valid": True,
                "saturated_x": True,
            },
            "cumulative_telemetry": {
                "output_ticks": 20,
                "authorized_ticks": 12,
                "movement_commands": 3,
                "control_samples": 4,
                "target_velocity_abs_x_pixels_per_second": 800.0,
                "saturated_x_samples": 2,
            },
        }
        self.rewrite_session(session, records)

        report = replay_session(session)

        self.assertEqual(report["makcu_control_frames"], 2)
        self.assertEqual(report["makcu_control_trace_coverage"], 1.0)
        self.assertEqual(report["makcu_ledger_snapshot_frames"], 2)
        self.assertEqual(report["makcu_successful_commands"], 3)
        self.assertEqual(report["makcu_ledger_commands"], 3)
        self.assertEqual(report["makcu_command_trace_coverage"], 1.0)
        self.assertEqual(report["makcu_command_timing_trace_coverage"], 1.0)
        self.assertEqual(report["makcu_unobserved_commands"], 0)
        self.assertEqual(report["makcu_ledger_emitted_x"], 5)
        self.assertEqual(report["makcu_ledger_emitted_y"], 0)
        self.assertEqual(report["makcu_ledger_emitted_abs_x"], 7)
        self.assertEqual(report["makcu_ledger_emitted_abs_y"], 6)
        self.assertEqual(report["makcu_first_command_ns"], 1_000_100_000)
        self.assertEqual(report["makcu_last_command_ns"], 1_003_100_000)
        self.assertAlmostEqual(report["makcu_command_gap_mean_ms"], 1.5)
        self.assertAlmostEqual(report["makcu_command_gap_p50_ms"], 1.5)
        self.assertAlmostEqual(report["makcu_command_gap_p95_ms"], 1.95)
        self.assertEqual(report["makcu_command_gap_max_ms"], 2.0)
        self.assertAlmostEqual(report["makcu_command_rate_hz"], 2000 / 3)
        self.assertAlmostEqual(
            report["makcu_cumulative_command_rate_hz"],
            2000 / 3,
        )
        self.assertAlmostEqual(
            report["makcu_commands_per_trace_second"],
            3 / report["trace_duration_seconds"],
        )
        self.assertEqual(
            report["makcu_connection_epochs"],
            [
                {
                    "connection_epoch": 7,
                    "snapshot_frames": 2,
                    "successful_commands": 3,
                    "ledger_commands": 3,
                    "command_trace_coverage": 1.0,
                    "unobserved_commands": 0,
                    "dropped_commands": 0,
                    "first_emitted_ns": 1_000_100_000,
                    "last_emitted_ns": 1_003_100_000,
                    "emitted_x": 5,
                    "emitted_y": 0,
                    "emitted_abs_x": 7,
                    "emitted_abs_y": 6,
                }
            ],
        )
        output_summary = report["makcu_calibrated_output_summary"]
        self.assertEqual(output_summary["samples"], 2)
        self.assertEqual(
            output_summary["numeric"]["rate_x_counts_per_second"]["mean"],
            1500.0,
        )
        self.assertEqual(
            output_summary["boolean"]["saturated_x"]["true_rate"],
            0.5,
        )
        self.assertEqual(
            report["makcu_cumulative_telemetry_totals"]["output_ticks"],
            20,
        )
        self.assertEqual(
            report["makcu_telemetry_summary"][
                "authorized_ticks_per_output_tick"
            ],
            0.6,
        )
        self.assertEqual(
            report["makcu_telemetry_summary"][
                "mean_target_velocity_abs_x_pixels_per_second"
            ],
            200.0,
        )
        self.assertEqual(
            report["makcu_telemetry_summary"]["saturated_x_samples_rate"],
            0.5,
        )

    def test_makcu_command_identity_includes_connection_epoch(self) -> None:
        session = self.make_session()
        records = self.read_records(session)
        for record, epoch, timestamp, delta_x in (
            (records[0], 7, 1_001_000_000, 1),
            (records[1], 8, 1_010_000_000, -1),
        ):
            record["makcu_control"] = {
                "connection_epoch": epoch,
                "successful_commands": 1,
                "dropped_commands": 0,
                "first_emitted_ns": timestamp,
                "last_emitted_ns": timestamp,
                "emitted_x": delta_x,
                "emitted_y": 0,
                "emitted_abs_x": 1,
                "emitted_abs_y": 0,
                "recent_commands": [
                    {
                        "sequence": 1,
                        "timestamp_ns": timestamp,
                        "delta_x_counts": delta_x,
                        "delta_y_counts": 0,
                    }
                ],
            }
        self.rewrite_session(session, records)

        report = replay_session(session)

        self.assertEqual(report["makcu_successful_commands"], 2)
        self.assertEqual(report["makcu_ledger_commands"], 2)
        self.assertEqual(
            [
                epoch["connection_epoch"]
                for epoch in report["makcu_connection_epochs"]
            ],
            [7, 8],
        )
        self.assertIsNone(report["makcu_command_rate_hz"])

    def test_replay_reports_incomplete_makcu_command_trace_coverage(self) -> None:
        session = self.make_session()
        records = self.read_records(session)
        for record, sequence, timestamp in (
            (records[0], 2, 1_002_000_000),
            (records[1], 3, 1_003_000_000),
        ):
            record["makcu_control"] = {
                "connection_epoch": 5,
                "successful_commands": 3,
                "dropped_commands": 1,
                "emitted_x": 3,
                "emitted_y": 0,
                "emitted_abs_x": 3,
                "emitted_abs_y": 0,
                "recent_commands": [
                    {
                        "sequence": sequence,
                        "timestamp_ns": timestamp,
                        "delta_x_counts": 1,
                        "delta_y_counts": 0,
                    }
                ],
            }
        self.rewrite_session(session, records)

        report = replay_session(session)

        self.assertEqual(report["makcu_successful_commands"], 3)
        self.assertEqual(report["makcu_ledger_commands"], 2)
        self.assertAlmostEqual(report["makcu_command_trace_coverage"], 2 / 3)
        self.assertEqual(report["makcu_command_timing_trace_coverage"], 0.5)
        self.assertEqual(report["makcu_unobserved_commands"], 1)
        self.assertEqual(report["makcu_internal_dropped_commands"], 1)
        self.assertAlmostEqual(
            report["makcu_connection_epochs"][0]["command_trace_coverage"],
            2 / 3,
        )

    def test_replay_rejects_conflicting_makcu_command_identity(self) -> None:
        session = self.make_session()
        records = self.read_records(session)
        for record, delta_x in zip(records, (1, 2), strict=True):
            record["makcu_control"] = {
                "connection_epoch": 4,
                "successful_commands": 1,
                "recent_commands": [
                    {
                        "sequence": 1,
                        "timestamp_ns": 1_001_000_000,
                        "delta_x_counts": delta_x,
                        "delta_y_counts": 0,
                    }
                ],
            }
        self.rewrite_session(session, records)

        with self.assertRaisesRegex(ValueError, "identity conflicts"):
            replay_session(session)

    def test_replay_reports_continuous_hold_safety_latch_separately(self) -> None:
        session = self.make_session()
        records = self.read_records(session)
        records[0].update(
            {
                "raw_activation_known": True,
                "raw_activation_pressed": True,
                "activation_requires_release": False,
                "activation_denial_reason": None,
            }
        )
        records[1].update(
            {
                "activation_pressed": False,
                "raw_activation_known": True,
                "raw_activation_pressed": True,
                "activation_requires_release": True,
                "activation_denial_reason": "continuous-hold-expired",
                "aim_candidates": [self.detection()],
                "accepted_measurement": self.detection(),
                "selected_target": self.detection(),
                "selected_is_prediction": False,
                "control_source": "none",
            }
        )
        self.rewrite_session(session, records)

        report = replay_session(session)

        self.assertEqual(report["raw_activation_known_frames"], 2)
        self.assertEqual(report["raw_activation_pressed_frames"], 2)
        self.assertEqual(report["raw_pressed_authorized_frames"], 1)
        self.assertEqual(
            report["filtered_activation_rate_while_raw_pressed"],
            0.5,
        )
        self.assertEqual(report["continuous_hold_expired_frames"], 1)
        self.assertEqual(report["continuous_hold_expired_events"], 1)
        self.assertIn(
            "release-and-repress-after-continuous-hold-limit",
            report["recommendations"],
        )

    def test_replay_detects_recorded_selection_divergence(self) -> None:
        session = self.make_session()
        records_path = session / "records.jsonl"
        records = self.read_records(session)
        records[1]["selected_target"] = None
        records_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        report = replay_session(session)

        self.assertEqual(report["replay_divergences"], 1)

    def test_replay_rejects_tampered_frame_artifact(self) -> None:
        session = self.make_session()
        first_record = json.loads(
            (session / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        (session / first_record["image_path"]).write_bytes(b"tampered")

        with self.assertRaisesRegex(ValueError, "hash"):
            replay_session(session)


if __name__ == "__main__":
    unittest.main()
