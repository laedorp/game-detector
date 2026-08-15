from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from capture.base import FramePacket
from scripts import record_capture_player_head as recorder


PNG_BYTES = b"\x89PNG\r\n\x1a\nlossless-test-payload"
JPEG_BYTES = b"\xff\xd8lossy-test-payload\xff\xd9"
FIXED_UTC = datetime(2026, 8, 15, 20, 30, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _Stats:
    frames_read: int = 12
    frames_delivered: int = 3
    frames_overwritten: int = 9
    read_failures: int = 0


class _FakeCapture:
    description = "fake latest-only capture"

    def __init__(
        self,
        events: list[FramePacket | None | BaseException],
        *,
        close_error: str | None = None,
    ) -> None:
        self.events = list(events)
        self.close_error = close_error
        self.started = False
        self.closed = False
        self.ended = False
        self.error: str | None = None
        self.actual_settings = {
            "width": np.int64(1920),
            "height": 1080,
            "fps": 239.76,
            "pixel_format": "NV12",
            "backend": "FAKE-V4L2",
        }
        self.stats = _Stats()
        self.read_timeouts: list[float | None] = []

    def start(self) -> None:
        self.started = True

    def read(self, timeout: float | None = None) -> FramePacket | None:
        self.read_timeouts.append(timeout)
        if not self.events:
            self.ended = True
            return None
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            self.error = self.close_error


def _packet(sequence: int, completed_ns: int, value: int = 0) -> FramePacket:
    image = np.full((6, 8, 3), value, dtype=np.uint8)
    return FramePacket(
        image=image,
        sequence=sequence,
        read_started_ns=max(0, completed_ns - 1_000_000),
        read_completed_ns=completed_ns,
    )


def _config(root: Path, **overrides: object) -> recorder.RecorderConfig:
    values: dict[str, object] = {
        "range_label": "medium",
        "motion": "moving",
        "scenario": "friend-strafe-left-right",
        "output_root": root,
        "duration_seconds": 30.0,
        "max_frames": 2,
        "sample_fps": 10.0,
    }
    values.update(overrides)
    return recorder.RecorderConfig(**values)


def _run(
    root: Path,
    source: _FakeCapture,
    *,
    config: recorder.RecorderConfig | None = None,
    encoder=lambda _image, _config: PNG_BYTES,
    session_id: str = "20260815T203000.000000Z-abcdef012345",
    clock=lambda: 0,
) -> recorder.RecorderResult:
    return recorder.record_session(
        config or _config(root),
        capture_factory=lambda _config: source,
        encoder=encoder,
        clock_ns=clock,
        utc_now=lambda: FIXED_UTC,
        session_id_factory=lambda: session_id,
    )


class RecorderManifestTests(unittest.TestCase):
    def test_png_is_lossless_default_and_manifest_hashes_exact_published_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture([_packet(7, 100_000_000), _packet(8, 200_000_000)])

            result = _run(root, source)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(result.status, "complete")
            self.assertEqual(result.completion_reason, "frame-limit")
            self.assertEqual(result.frame_count, 2)
            self.assertEqual(manifest["schema"], recorder.SCHEMA_NAME)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["range"], "medium")
            self.assertEqual(manifest["motion"], "moving")
            self.assertEqual(manifest["scenario"], "friend-strafe-left-right")
            self.assertEqual(manifest["encoding"]["format"], "png")
            self.assertEqual(manifest["encoding"]["extension"], ".png")
            self.assertTrue(manifest["encoding"]["lossless"])
            self.assertIsNone(manifest["encoding"]["jpeg_quality"])
            self.assertEqual(manifest["capture"]["actual"]["pixel_format"], "NV12")
            self.assertEqual(manifest["capture"]["stats"]["frames_overwritten"], 9)
            self.assertTrue(manifest["capture"]["requested"]["latest_only"])

            first = manifest["frames"][0]
            image_path = result.session_dir / first["file_name"]
            self.assertEqual(image_path.suffix, ".png")
            self.assertEqual(image_path.read_bytes(), PNG_BYTES)
            self.assertEqual(first["sha256"], sha256(PNG_BYTES).hexdigest())
            self.assertEqual(first["byte_size"], len(PNG_BYTES))
            self.assertEqual(first["encoding"], "png")
            self.assertEqual(first["width"], 8)
            self.assertEqual(first["height"], 6)
            self.assertEqual(first["source_sequence"], 7)
            self.assertEqual(first["source_read_completed_ns"], 100_000_000)
            self.assertTrue(source.started)
            self.assertTrue(source.closed)

    def test_jpeg_is_explicit_and_extension_bytes_quality_agree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            config = _config(
                root,
                encoding="jpeg",
                jpeg_quality=91,
                max_frames=1,
                range_label="long",
                motion="stationary",
            )
            source = _FakeCapture([_packet(2, 50_000_000)])

            result = _run(
                root,
                source,
                config=config,
                encoder=lambda _image, _config: JPEG_BYTES,
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            frame = manifest["frames"][0]

            self.assertEqual(manifest["encoding"]["format"], "jpeg")
            self.assertFalse(manifest["encoding"]["lossless"])
            self.assertEqual(manifest["encoding"]["jpeg_quality"], 91)
            self.assertIsNone(manifest["encoding"]["png_compression"])
            self.assertTrue(frame["file_name"].endswith(".jpg"))
            self.assertEqual((result.session_dir / frame["file_name"]).read_bytes(), JPEG_BYTES)
            self.assertEqual(frame["sha256"], sha256(JPEG_BYTES).hexdigest())

    def test_sampling_uses_source_timestamps_not_every_capture_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture(
                [
                    _packet(0, 0, 0),
                    _packet(1, 20_000_000, 1),
                    _packet(2, 99_999_999, 2),
                    _packet(3, 100_000_000, 3),
                ]
            )

            result = _run(root, source)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(
                [frame["source_sequence"] for frame in manifest["frames"]], [0, 3]
            )
            self.assertTrue(all(timeout <= 0.25 for timeout in source.read_timeouts if timeout))

    def test_actual_settings_and_all_source_provenance_are_atomic_manifest_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture([_packet(123, 456_000_000), _packet(124, 556_000_000)])

            result = _run(root, source)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["capture"]["actual"]["width"], 1920)
            self.assertEqual(manifest["capture"]["actual"]["fps"], 239.76)
            frame = manifest["frames"][0]
            self.assertEqual(frame["source_sequence"], 123)
            self.assertEqual(frame["source_read_started_ns"], 455_000_000)
            self.assertEqual(frame["source_read_completed_ns"], 456_000_000)
            self.assertRegex(frame["saved_utc"], r"Z$")
            self.assertFalse(any(result.session_dir.glob(".*.tmp")))
            self.assertFalse(any((result.session_dir / "images").glob(".*.tmp")))


class InterruptionAndBoundsTests(unittest.TestCase):
    def test_ctrl_c_leaves_valid_explicitly_incomplete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture([_packet(0, 0), KeyboardInterrupt()])

            result = _run(root, source)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(result.status, "interrupted")
            self.assertEqual(result.completion_reason, "keyboard-interrupt")
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(len(manifest["frames"]), 1)
            self.assertIsNotNone(manifest["ended_utc"])
            self.assertTrue(source.closed)
            self.assertEqual(os.stat(result.manifest_path).st_mode & 0o777, 0o400)
            self.assertEqual(os.stat(result.session_dir).st_mode & 0o777, 0o500)

    def test_zero_frame_duration_bound_is_failed_not_false_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture([_packet(0, 0)])
            ticks = iter((0, 200_000_000))
            config = _config(root, duration_seconds=0.1)

            result = _run(root, source, config=config, clock=lambda: next(ticks))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.completion_reason, "duration-limit-no-frames")
            self.assertEqual(result.frame_count, 0)
            self.assertEqual(source.read_timeouts, [])
            self.assertFalse(manifest["complete"])

    def test_duration_bound_is_complete_after_at_least_one_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture([_packet(0, 0)])
            ticks = iter((0, 0, 0, 200_000_000))
            config = _config(root, duration_seconds=0.1, max_frames=2)

            result = _run(root, source, config=config, clock=lambda: next(ticks))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(result.status, "complete")
            self.assertEqual(result.completion_reason, "duration-limit")
            self.assertEqual(result.frame_count, 1)
            self.assertTrue(manifest["complete"])

    def test_clock_contract_failure_finalizes_and_seals_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture([_packet(0, 0)])
            session_id = "bad-clock-session"

            with self.assertRaisesRegex(recorder.RecorderError, "clock_ns"):
                _run(root, source, session_id=session_id, clock=lambda: -1)

            manifest_path = root / session_id / "session.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(manifest["complete"])
            self.assertIn("clock_ns", manifest["completion_reason"])
            self.assertEqual(os.stat(manifest_path).st_mode & 0o777, 0o400)
            self.assertFalse(source.started)

    def test_unexpected_source_end_is_valid_but_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture([])

            result = _run(root, source)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.completion_reason, "source-ended")
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["frames"], [])

    def test_close_reported_error_downgrades_frame_limit_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture(
                [_packet(0, 0)],
                close_error="Timed out waiting for capture worker to stop",
            )
            config = _config(root, max_frames=1)

            result = _run(root, source, config=config)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.completion_reason, "capture-error")
            self.assertEqual(result.frame_count, 1)
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completion_reason"], "capture-error")
            self.assertEqual(
                manifest["capture"]["error"],
                "Timed out waiting for capture worker to stop",
            )
            self.assertEqual(os.stat(result.manifest_path).st_mode & 0o777, 0o400)

    def test_encoder_failure_is_recorded_before_error_is_raised(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture([_packet(0, 0)])
            session_id = "failed-session"

            with self.assertRaisesRegex(recorder.RecorderError, "declared PNG"):
                _run(
                    root,
                    source,
                    encoder=lambda _image, _config: JPEG_BYTES,
                    session_id=session_id,
                )

            manifest_path = root / session_id / "session.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(manifest["complete"])
            self.assertIn("declared PNG", manifest["completion_reason"])
            self.assertTrue(source.closed)

    def test_primary_failure_reason_survives_a_separate_close_reported_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture(
                [_packet(0, 0)],
                close_error="capture worker still closing",
            )
            session_id = "primary-and-close-failure"

            with self.assertRaisesRegex(recorder.RecorderError, "declared PNG"):
                _run(
                    root,
                    source,
                    encoder=lambda _image, _config: JPEG_BYTES,
                    session_id=session_id,
                )

            manifest = json.loads(
                (root / session_id / "session.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("declared PNG", manifest["completion_reason"])
            self.assertNotEqual(manifest["completion_reason"], "capture-error")
            self.assertEqual(
                manifest["capture"]["error"], "capture worker still closing"
            )


class OutputSafetyTests(unittest.TestCase):
    def test_existing_session_is_never_reused_or_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            session = root / "claimed-session"
            session.mkdir(parents=True)
            marker = session / "keep.txt"
            marker.write_text("do not overwrite", encoding="utf-8")
            source = _FakeCapture([_packet(0, 0)])

            with self.assertRaisesRegex(recorder.RecorderError, "refusing overwrite"):
                _run(root, source, session_id="claimed-session")

            self.assertEqual(marker.read_text(encoding="utf-8"), "do not overwrite")
            self.assertFalse(source.started)

    def test_generated_session_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture([_packet(0, 0)])

            with self.assertRaisesRegex(recorder.RecorderError, "unsafe generated"):
                _run(root, source, session_id="../escape")

            self.assertFalse(source.started)
            self.assertFalse((Path(temporary) / "escape").exists())

    def test_symlink_in_output_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            real = base / "real"
            real.mkdir()
            linked = base / "linked"
            linked.symlink_to(real, target_is_directory=True)
            root = linked / "raw"
            source = _FakeCapture([_packet(0, 0)])

            with self.assertRaisesRegex(recorder.RecorderError, "refusing symlink"):
                _run(root, source)

            self.assertFalse(source.started)
            self.assertEqual(list(real.iterdir()), [])

    def test_output_root_parent_traversal_is_rejected_by_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            _config(Path("safe") / ".." / "escape")

    def test_duplicate_or_backwards_packets_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            source = _FakeCapture(
                [
                    _packet(4, 100_000_000),
                    _packet(4, 200_000_000),
                    _packet(5, 50_000_000),
                    _packet(6, 200_000_000),
                ]
            )

            result = _run(root, source)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["skipped_stale_packets"], 2)
            self.assertEqual(
                [frame["source_sequence"] for frame in manifest["frames"]], [4, 6]
            )


class DefaultsAndIsolationTests(unittest.TestCase):
    def test_cli_requires_range_motion_and_scenario(self) -> None:
        parser = recorder.build_parser()
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parser.parse_args([])

    def test_host_defaults_are_device_zero_1080p240_nv12_and_lossless_png(self) -> None:
        args = recorder.build_parser().parse_args(
            [
                "--range",
                "close",
                "--motion",
                "stationary",
                "--scenario",
                "friend-standing",
            ]
        )
        config = recorder._config_from_args(args)

        self.assertEqual(config.device, 0)
        self.assertEqual((config.width, config.height), (1920, 1080))
        self.assertEqual(config.capture_fps, 240.0)
        self.assertEqual(config.pixel_format, "NV12")
        self.assertEqual(config.sample_fps, 10.0)
        self.assertEqual(config.encoding, "png")
        self.assertEqual(config.duration_seconds, 120.0)
        self.assertEqual(config.max_frames, 1200)
        self.assertEqual(config.output_root, recorder.DEFAULT_OUTPUT_ROOT)

    def test_cli_output_root_is_visible_and_wired_without_capture(self) -> None:
        parser = recorder.build_parser()
        custom_root = Path("/mnt/training-data/raw")
        args = parser.parse_args(
            [
                "--range",
                "medium",
                "--motion",
                "moving",
                "--scenario",
                "friend-running",
                "--output-root",
                str(custom_root),
            ]
        )

        config = recorder._config_from_args(args)

        self.assertEqual(config.output_root, custom_root)
        help_text = parser.format_help()
        self.assertIn("--output-root PATH", help_text)
        self.assertIn(
            str(recorder.DEFAULT_OUTPUT_ROOT), "".join(help_text.split())
        )
        self.assertIn("lossless PNG sessions can be large", help_text)

    def test_default_factory_constructs_only_latest_opencv_capture(self) -> None:
        config = _config(Path("datasets/capture_player_head/raw"))
        sentinel = object()
        with mock.patch.object(
            recorder, "OpenCVCaptureSource", return_value=sentinel
        ) as constructor:
            result = recorder._default_capture_factory(config)

        self.assertIs(result, sentinel)
        constructor.assert_called_once_with(
            0,
            width=1920,
            height=1080,
            fps=240.0,
            buffer_size=1,
            pixel_format="NV12",
            rotate_180=False,
            live=True,
        )

    def test_module_has_no_detector_model_aiming_or_makcu_import(self) -> None:
        source_path = Path(recorder.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])

        self.assertTrue({"capture", "numpy"}.issubset(imported_roots))
        self.assertTrue(
            imported_roots.isdisjoint(
                {"aiming", "detection", "main", "ultralytics", "onnxruntime"}
            )
        )

    def test_bounds_and_required_labels_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid = (
                {"range_label": "far"},
                {"motion": "sometimes"},
                {"scenario": "../unsafe"},
                {"duration_seconds": 0},
                {"duration_seconds": recorder.MAX_DURATION_SECONDS + 1},
                {"max_frames": recorder.MAX_FRAME_COUNT + 1},
                {"sample_fps": recorder.MAX_SAMPLE_FPS + 1},
                {"encoding": "raw"},
                {"png_compression": 10},
                {"jpeg_quality": 0},
            )
            for override in invalid:
                with self.subTest(override=override), self.assertRaises((ValueError, TypeError)):
                    _config(root, **override)

    def test_main_returns_nonzero_when_source_session_failed(self) -> None:
        result = recorder.RecorderResult(
            session_dir=Path("session"),
            manifest_path=Path("session/session.json"),
            status="failed",
            completion_reason="source-ended",
            frame_count=0,
        )
        argv = [
            "--range",
            "negative",
            "--motion",
            "stationary",
            "--scenario",
            "empty-scene",
        ]
        with mock.patch.object(recorder, "record_session", return_value=result):
            with mock.patch("builtins.print"):
                self.assertEqual(recorder.main(argv), 1)


if __name__ == "__main__":
    unittest.main()
