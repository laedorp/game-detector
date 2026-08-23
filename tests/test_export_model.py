from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.export_model import (
    ExportArtifactError,
    _artifact_basename,
    build_parser as build_export_parser,
    main as export_main,
    normalize_ir_basename,
)
from scripts.train_fort_model import TrainingConfig, TrainingOutcome, _print_outcome


class ExportModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_pair(self, directory: Path, stem: str = "best") -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        xml_file = directory / f"{stem}.xml"
        bin_file = directory / f"{stem}.bin"
        xml_file.write_bytes(b"xml graph")
        bin_file.write_bytes(b"binary weights")
        return xml_file, bin_file

    def test_basename_accepts_plain_stems_and_rejects_paths_or_extensions(self) -> None:
        self.assertEqual(_artifact_basename("fort_player"), "fort_player")
        self.assertEqual(_artifact_basename("YOLO-26n"), "YOLO-26n")
        for invalid in ("", "../player", "player/model", "player.xml", "player model"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _artifact_basename(invalid)

    def test_image_size_accepts_legacy_square_and_explicit_height_width(self) -> None:
        self.assertEqual(build_export_parser().parse_args([]).imgsz, (320, 320))
        self.assertEqual(
            build_export_parser().parse_args(["--imgsz", "416"]).imgsz,
            (416, 416),
        )
        self.assertEqual(
            build_export_parser().parse_args(["--imgsz", "384x640"]).imgsz,
            (384, 640),
        )

    def test_normalizes_both_ir_artifacts_and_preserves_other_files(self) -> None:
        source_xml, source_bin = self.write_pair(self.root)
        metadata = self.root / "metadata.yaml"
        metadata.write_text("task: detect\n", encoding="utf-8")

        xml_file, bin_file = normalize_ir_basename(self.root, "fort_player")

        self.assertEqual(xml_file, self.root / "fort_player.xml")
        self.assertEqual(bin_file, self.root / "fort_player.bin")
        self.assertEqual(xml_file.read_bytes(), b"xml graph")
        self.assertEqual(bin_file.read_bytes(), b"binary weights")
        self.assertFalse(source_xml.exists())
        self.assertFalse(source_bin.exists())
        self.assertEqual(metadata.read_text(encoding="utf-8"), "task: detect\n")

    def test_refuses_a_target_collision_without_mutating_the_source_pair(self) -> None:
        source_xml, source_bin = self.write_pair(self.root)
        (self.root / "fort_player.xml").mkdir()

        with self.assertRaisesRegex(ExportArtifactError, "Refusing to overwrite"):
            normalize_ir_basename(self.root, "fort_player")

        self.assertEqual(source_xml.read_bytes(), b"xml graph")
        self.assertEqual(source_bin.read_bytes(), b"binary weights")
        self.assertTrue((self.root / "fort_player.xml").is_dir())
        self.assertFalse((self.root / "fort_player.bin").exists())

    def test_rejects_an_unmatched_or_ambiguous_ir_export(self) -> None:
        self.root.joinpath("best.xml").write_bytes(b"xml")
        self.root.joinpath("other.bin").write_bytes(b"bin")
        with self.assertRaisesRegex(ExportArtifactError, "same stem"):
            normalize_ir_basename(self.root, "fort_player")

        self.root.joinpath("other.xml").write_bytes(b"xml")
        with self.assertRaisesRegex(ExportArtifactError, "exactly one"):
            normalize_ir_basename(self.root, "fort_player")

    def test_main_moves_export_with_requested_final_basename(self) -> None:
        exported = self.root / "best_openvino_model"
        output = self.root / "models" / "fort_player_openvino_model"
        test_case = self

        class FakeYOLO:
            task = "detect"

            def __init__(self, source: str) -> None:
                self.source = source

            def export(self, **kwargs: object) -> str:
                test_case.write_pair(exported)
                exported.joinpath("metadata.yaml").write_text(
                    "task: detect\n", encoding="utf-8"
                )
                return str(exported)

        fake_ultralytics = SimpleNamespace(YOLO=FakeYOLO)
        argv = [
            "export_model.py",
            "--weights",
            str(self.root / "best.pt"),
            "--output",
            str(output),
            "--basename",
            "fort_player",
        ]
        stdout = StringIO()
        with (
            mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}),
            mock.patch.object(sys, "argv", argv),
            redirect_stdout(stdout),
        ):
            export_main()

        self.assertFalse(exported.exists())
        self.assertEqual((output / "fort_player.xml").read_bytes(), b"xml graph")
        self.assertEqual((output / "fort_player.bin").read_bytes(), b"binary weights")
        self.assertTrue((output / "metadata.yaml").is_file())
        self.assertIn("fort_player.xml, fort_player.bin", stdout.getvalue())


class FortExportHandoffTests(unittest.TestCase):
    def test_training_outcome_prints_the_required_fort_artifact_basename(self) -> None:
        root = Path("/tmp/fort-export-command-test")
        config = TrainingConfig(
            data=root / "fort_cuh_player.yaml",
            weights=root / "yolo26n.pt",
            project=root / "runs",
            name="player_v1",
        )
        outcome = TrainingOutcome(
            run_dir=config.run_dir,
            best_weights=config.run_dir / "weights" / "best.pt",
            test_run_dir=None,
        )
        stdout = StringIO()

        with redirect_stdout(stdout):
            _print_outcome(outcome, config)

        self.assertIn("--basename fort_player", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
