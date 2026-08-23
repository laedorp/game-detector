from __future__ import annotations

import json
from pathlib import Path
import tempfile
from threading import Event
import unittest

import numpy as np

from utils.aim_diagnostic import AimDiagnosticConfig, AimDiagnosticRecorder


class AimDiagnosticRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def record(sequence: int, timestamp_ns: int) -> dict[str, object]:
        return {
            "frame_sequence": sequence,
            "source_timestamp_ns": timestamp_ns,
            "frame_shape": [4, 6, 3],
            "tracking_mode": "stable-body",
            "activation_pressed": True,
            "aim_candidates": [],
            "continuation_candidates": [],
            "accepted_measurement": None,
            "selected_target": None,
            "selected_is_prediction": False,
            "control_source": "none",
        }

    def test_every_decision_is_written_while_frames_are_source_time_sampled(self) -> None:
        recorder = AimDiagnosticRecorder(
            AimDiagnosticConfig(
                output_root=self.root,
                sample_hz=10.0,
                max_duration_seconds=1.0,
            ),
            metadata={"aim_label": "player"},
            image_encoder=lambda image, _quality: bytes(image.reshape(-1)),
        )
        recorder.start()
        frame = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)

        self.assertTrue(recorder.submit(frame, self.record(1, 1_000_000_000)))
        self.assertTrue(recorder.submit(frame, self.record(2, 1_050_000_000)))
        self.assertTrue(recorder.stop())

        manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "proaim.aim_diagnostic.session")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["metadata"], {"aim_label": "player"})
        self.assertEqual(manifest["statistics"]["submitted"], 2)
        self.assertEqual(manifest["statistics"]["sample_gate_skips"], 1)
        self.assertEqual(manifest["statistics"]["written"], 2)
        lines = recorder.records_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        records = [json.loads(line) for line in lines]
        self.assertEqual([item["frame_sequence"] for item in records], [1, 2])
        image = recorder.session_dir / records[0]["image_path"]
        self.assertTrue(image.is_file())
        self.assertRegex(records[0]["image_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("image_path", records[1])

    def test_pending_queue_is_bounded_and_reports_overwrites(self) -> None:
        encoder_started = Event()
        release_encoder = Event()

        def blocking_encoder(image, _quality):
            encoder_started.set()
            self.assertTrue(release_encoder.wait(1.0))
            return bytes(image.reshape(-1))

        recorder = AimDiagnosticRecorder(
            AimDiagnosticConfig(
                output_root=self.root,
                sample_hz=50.0,
                max_duration_seconds=1.0,
                max_pending_records=1,
            ),
            image_encoder=blocking_encoder,
        )
        recorder.start()
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        self.assertTrue(recorder.submit(frame, self.record(1, 1_000_000_000)))
        self.assertTrue(encoder_started.wait(1.0))
        self.assertTrue(recorder.submit(frame, self.record(2, 1_020_000_000)))
        self.assertTrue(recorder.submit(frame, self.record(3, 1_040_000_000)))
        release_encoder.set()
        self.assertTrue(recorder.stop())

        manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["statistics"]["pending_overwrites"], 1)
        lines = recorder.records_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            [json.loads(line)["frame_sequence"] for line in lines],
            [1, 3],
        )

    def test_duration_limit_is_bounded_in_source_time(self) -> None:
        recorder = AimDiagnosticRecorder(
            AimDiagnosticConfig(
                output_root=self.root,
                sample_hz=20.0,
                max_duration_seconds=0.1,
            ),
            image_encoder=lambda image, _quality: bytes(image.reshape(-1)),
        )
        recorder.start()
        frame = np.zeros((4, 6, 3), dtype=np.uint8)

        self.assertTrue(recorder.submit(frame, self.record(1, 1_000_000_000)))
        self.assertFalse(recorder.submit(frame, self.record(2, 1_100_000_001)))
        self.assertTrue(recorder.stop())
        manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["statistics"]["duration_limit_skips"], 1)

    def test_activation_armed_duration_begins_on_first_pressed_record(self) -> None:
        recorder = AimDiagnosticRecorder(
            AimDiagnosticConfig(
                output_root=self.root,
                sample_hz=20.0,
                max_duration_seconds=0.1,
                wait_for_activation=True,
            ),
            image_encoder=lambda image, _quality: bytes(image.reshape(-1)),
        )
        recorder.start()
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        released = self.record(1, 1_000_000_000)
        released["activation_pressed"] = False
        released["raw_activation_pressed"] = False

        self.assertFalse(recorder.submit(frame, released))
        self.assertEqual(recorder.status.arming_skips, 1)
        pressed = self.record(2, 2_000_000_000)
        pressed["raw_activation_pressed"] = True
        self.assertTrue(recorder.submit(frame, pressed))
        released_after_arm = self.record(3, 2_050_000_000)
        released_after_arm["activation_pressed"] = False
        released_after_arm["raw_activation_pressed"] = False
        self.assertTrue(recorder.submit(frame, released_after_arm))
        expired = self.record(4, 2_100_000_001)
        self.assertFalse(recorder.submit(frame, expired))
        self.assertTrue(recorder.stop())

        manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["configuration"]["wait_for_activation"])
        self.assertEqual(manifest["statistics"]["arming_skips"], 1)
        records = [
            json.loads(line)
            for line in recorder.records_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [record["frame_sequence"] for record in records],
            [2, 3],
        )


if __name__ == "__main__":
    unittest.main()
