from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts.prepare_fort_cuh import (
    ATTRIBUTION_TEXT,
    DatasetPreparationError,
    prepare_dataset,
)
from scripts.train_fort_model import (
    TrainingConfig,
    TrainingConfigurationError,
    build_parser as build_training_parser,
    config_from_args,
    run_training,
    training_arguments,
    validate_training_config,
)


class SyntheticFortArchive:
    CATEGORIES = [
        {"id": 0, "name": "Person"},
        {"id": 1, "name": "PLAYER"},
        {"id": 2, "name": "head"},
        {"id": 3, "name": "body"},
        {"id": 4, "name": "ally"},
        {"id": 5, "name": "Yourself"},
        {"id": 6, "name": "Fortnite"},
        {"id": 7, "name": "bots"},
        {"id": 8, "name": "enemy"},
        {"id": 9, "name": "hello"},
        {"id": 10, "name": "people"},
        {"id": 11, "name": "player"},
        {"id": 12, "name": "0"},
    ]

    @classmethod
    def write(
        cls,
        path: Path,
        *,
        unknown_category: bool = False,
        unsafe_member: str | None = None,
        video_overlap: bool = False,
    ) -> None:
        categories = list(cls.CATEGORIES)
        if unknown_category:
            categories.append({"id": 99, "name": "vehicle"})
        split_values = {
            "train": {
                "images": [
                    {
                        "id": 1,
                        "file_name": "shared_jpg.rf.trainhash.jpg",
                        "width": 100,
                        "height": 100,
                    },
                    {
                        "id": 2,
                        "file_name": "head_only.jpg",
                        "width": 100,
                        "height": 100,
                    },
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [-5, -2, 20, 22],
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 2,
                        "bbox": [2, 2, 5, 5],
                        "iscrowd": 0,
                    },
                    {
                        "id": 3,
                        "image_id": 2,
                        "category_id": 2,
                        "bbox": [10, 10, 20, 20],
                        "iscrowd": 0,
                    },
                ],
            },
            "valid": {
                "images": [
                    {
                        "id": 10,
                        "file_name": "shared_jpg.rf.validhash.jpg",
                        "width": 200,
                        "height": 100,
                    }
                ],
                "annotations": [
                    {
                        "id": 10,
                        "image_id": 10,
                        "category_id": 6,
                        "bbox": [20, 10, 40, 50],
                        "iscrowd": 0,
                    }
                ],
            },
            "test": {
                "images": [
                    {
                        "id": 20,
                        "file_name": "test_player.png",
                        "width": 100,
                        "height": 100,
                    }
                ],
                "annotations": [
                    {
                        "id": 20,
                        "image_id": 20,
                        "category_id": 8,
                        "bbox": [30, 30, -5, 20],
                        "iscrowd": 0,
                    },
                    {
                        "id": 21,
                        "image_id": 20,
                        "category_id": 11,
                        "bbox": [25, 20, 20, 40],
                        "iscrowd": 0,
                    },
                ],
            },
        }
        image_bytes = {
            "train/shared_jpg.rf.trainhash.jpg": b"train player image",
            "train/head_only.jpg": b"head-only image",
            "valid/shared_jpg.rf.validhash.jpg": b"valid player image",
            "test/test_player.png": b"test player image",
        }
        if video_overlap:
            split_values["train"]["images"].append(
                {
                    "id": 3,
                    "file_name": "gameplay_mp4-10_jpg.rf.trainvideo.jpg",
                    "width": 100,
                    "height": 100,
                }
            )
            split_values["train"]["annotations"].append(
                {
                    "id": 30,
                    "image_id": 3,
                    "category_id": 8,
                    "bbox": [10, 10, 20, 40],
                    "iscrowd": 0,
                }
            )
            split_values["valid"]["images"].append(
                {
                    "id": 11,
                    "file_name": "gameplay_mp4-20_jpg.rf.validvideo.jpg",
                    "width": 100,
                    "height": 100,
                }
            )
            split_values["valid"]["annotations"].append(
                {
                    "id": 11,
                    "image_id": 11,
                    "category_id": 8,
                    "bbox": [20, 20, 25, 50],
                    "iscrowd": 0,
                }
            )
            image_bytes.update(
                {
                    "train/gameplay_mp4-10_jpg.rf.trainvideo.jpg": b"video frame ten",
                    "valid/gameplay_mp4-20_jpg.rf.validvideo.jpg": b"video frame twenty",
                }
            )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for split, values in split_values.items():
                payload = {
                    "images": values["images"],
                    "annotations": values["annotations"],
                    "categories": categories,
                }
                archive.writestr(
                    f"{split}/_annotations.coco.json",
                    json.dumps(payload, sort_keys=True),
                )
            for name, data in image_bytes.items():
                archive.writestr(name, data)
            if unsafe_member is not None:
                archive.writestr(unsafe_member, b"must never be extracted")


class FortDatasetPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "fort.zip"
        SyntheticFortArchive.write(self.archive)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_converts_one_class_clips_boxes_and_excludes_parts(self) -> None:
        output = self.root / "prepared"
        result = prepare_dataset(self.archive, output, expected_sha256=None)

        self.assertEqual(result.reports["train"].source_images, 2)
        self.assertEqual(result.reports["train"].retained_images, 1)
        self.assertEqual(result.reports["train"].retained_boxes, 1)
        self.assertEqual(result.reports["train"].clipped_boxes, 1)
        self.assertEqual(result.reports["train"].excluded_annotations["head"], 2)
        self.assertEqual(result.reports["test"].invalid_boxes_skipped, 1)
        self.assertFalse((output / "images" / "train" / "head_only.jpg").exists())
        self.assertEqual(
            (output / "labels" / "train" / "shared_jpg.rf.trainhash.txt").read_text(
                encoding="utf-8"
            ),
            "0 0.075 0.1 0.15 0.2\n",
        )
        self.assertEqual((output / "labels.txt").read_text(encoding="utf-8"), "player\n")
        self.assertIn("0: player", (output / "fort_cuh_player.yaml").read_text("utf-8"))
        self.assertEqual(
            (output / "ATTRIBUTION.md").read_text(encoding="utf-8"),
            ATTRIBUTION_TEXT,
        )
        self.assertEqual(
            (Path(__file__).resolve().parents[1] / "docs" / "FORT_CUH_ATTRIBUTION.md").read_text(
                encoding="utf-8"
            ),
            ATTRIBUTION_TEXT,
        )

        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["conversion"]["output_class_names"], ["player"])
        self.assertEqual(
            manifest["leakage"]["cross_split_original_basename_groups"],
            1,
        )
        self.assertEqual(manifest["leakage"]["cross_split_exact_duplicate_groups"], 0)
        self.assertEqual(manifest["leakage"]["cross_split_source_groups"], 1)
        self.assertEqual(manifest["leakage"]["cross_split_source_group_images"], 2)
        self.assertEqual(manifest["leakage"]["cross_split_video_sequence_groups"], 0)
        self.assertIn(
            "No visual-similarity or timestamp guessing",
            manifest["leakage"]["source_grouping_heuristic"]["description"],
        )

    def test_reports_explicit_video_sequence_overlap_conservatively(self) -> None:
        archive = self.root / "video-overlap.zip"
        SyntheticFortArchive.write(archive, video_overlap=True)
        output = self.root / "video-overlap"
        prepare_dataset(archive, output, expected_sha256=None)

        leakage = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )["leakage"]
        self.assertEqual(leakage["cross_split_original_basename_groups"], 1)
        self.assertEqual(leakage["cross_split_source_groups"], 2)
        self.assertEqual(leakage["cross_split_source_group_images"], 4)
        self.assertEqual(leakage["cross_split_video_sequence_groups"], 1)
        self.assertEqual(leakage["cross_split_video_sequence_images"], 2)
        self.assertEqual(
            leakage["cross_split_video_sequence_details"][0]["key"],
            "gameplay_mp4",
        )

    def test_output_is_deterministic_for_the_same_archive(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        prepare_dataset(self.archive, first, expected_sha256=None)
        prepare_dataset(self.archive, second, expected_sha256=None)

        def contents(root: Path) -> dict[str, str]:
            return {
                str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }

        self.assertEqual(contents(first), contents(second))

    def test_refuses_existing_output_and_wrong_hash(self) -> None:
        output = self.root / "prepared"
        output.mkdir()
        with self.assertRaisesRegex(DatasetPreparationError, "already exists"):
            prepare_dataset(self.archive, output, expected_sha256=None)
        with self.assertRaisesRegex(DatasetPreparationError, "SHA-256 mismatch"):
            prepare_dataset(
                self.archive,
                self.root / "other",
                expected_sha256="0" * 64,
            )

    def test_rejects_any_unsafe_archive_member(self) -> None:
        archive = self.root / "unsafe.zip"
        SyntheticFortArchive.write(archive, unsafe_member="../escape.txt")
        output = self.root / "prepared"
        with self.assertRaisesRegex(DatasetPreparationError, "unsafe zip member"):
            prepare_dataset(archive, output, expected_sha256=None)
        self.assertFalse(output.exists())
        self.assertFalse((self.root / "escape.txt").exists())

    def test_unknown_category_requires_explicit_review(self) -> None:
        archive = self.root / "unknown.zip"
        SyntheticFortArchive.write(archive, unknown_category=True)
        with self.assertRaisesRegex(DatasetPreparationError, "unknown categories"):
            prepare_dataset(archive, self.root / "prepared", expected_sha256=None)


class FortTrainingScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        archive = self.root / "fort.zip"
        SyntheticFortArchive.write(archive)
        self.dataset = self.root / "prepared"
        prepare_dataset(archive, self.dataset, expected_sha256=None)
        self.weights = self.root / "yolo26n.pt"
        self.weights.write_bytes(b"local weights")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, **changes: object) -> TrainingConfig:
        values: dict[str, object] = {
            "data": self.dataset / "fort_cuh_player.yaml",
            "weights": self.weights,
            "project": self.root / "runs",
            "name": "unit_run",
            "epochs": 2,
            "patience": 1,
            "batch": 4,
            "workers": 1,
            "threads": 3,
            "cache": "none",
        }
        values.update(changes)
        return TrainingConfig(**values)  # type: ignore[arg-type]

    def test_smoke_flag_forces_one_epoch_small_fraction_and_no_test(self) -> None:
        args = build_training_parser().parse_args(
            [
                "--data",
                str(self.dataset / "fort_cuh_player.yaml"),
                "--weights",
                str(self.weights),
                "--project",
                str(self.root / "runs"),
                "--name",
                "trial",
                "--epochs",
                "99",
                "--threads",
                "5",
                "--smoke-test",
            ]
        )
        config = config_from_args(args)
        options = training_arguments(config)
        self.assertEqual(config.name, "trial_smoke")
        self.assertEqual(config.epochs, 1)
        self.assertEqual(config.threads, 5)
        self.assertFalse(config.run_test)
        self.assertEqual(config.cache, "none")
        self.assertEqual(options["fraction"], 0.02)
        self.assertFalse(options["plots"])

    def test_training_arguments_pin_reproducible_cpu_defaults(self) -> None:
        config = self.config()
        options = training_arguments(config)
        self.assertTrue(options["deterministic"])
        self.assertFalse(options["amp"])
        self.assertEqual(options["seed"], 0)
        self.assertEqual(options["imgsz"], 320)
        self.assertIs(options["single_cls"], False)
        self.assertEqual(options["cache"], False)

    def test_validation_refuses_reusing_a_run_directory(self) -> None:
        config = self.config()
        config.run_dir.mkdir(parents=True)
        with self.assertRaisesRegex(TrainingConfigurationError, "already exists"):
            validate_training_config(config)

    def test_mock_training_sets_threads_validates_test_and_records_inputs(self) -> None:
        config = self.config()

        class FakeYOLO:
            instances: list["FakeYOLO"] = []

            def __init__(self, source: str) -> None:
                self.source = source
                self.task = "detect"
                self.trainer = None
                self.train_options: dict[str, object] | None = None
                self.val_options: dict[str, object] | None = None
                self.__class__.instances.append(self)

            def train(self, **kwargs: object) -> None:
                self.train_options = kwargs
                run_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                weights_dir = run_dir / "weights"
                weights_dir.mkdir(parents=True)
                best = weights_dir / "best.pt"
                best.write_bytes(b"fine-tuned")
                self.trainer = SimpleNamespace(best=best)

            def val(self, **kwargs: object) -> None:
                self.val_options = kwargs
                (Path(str(kwargs["project"])) / str(kwargs["name"])).mkdir(parents=True)

        output = StringIO()
        with mock.patch.dict(os.environ, {}, clear=False), redirect_stdout(output):
            outcome = run_training(config, yolo_class=FakeYOLO)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "3")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "3")

        self.assertIn("metrics are not independent and may be optimistic", output.getvalue())
        self.assertEqual(outcome.best_weights.name, "best.pt")
        self.assertEqual(len(FakeYOLO.instances), 2)
        self.assertEqual(FakeYOLO.instances[0].train_options["epochs"], 2)
        self.assertEqual(FakeYOLO.instances[1].val_options["split"], "test")
        record = json.loads(
            (config.run_dir / "reproducibility.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["training"]["threads"], 3)
        self.assertEqual(
            record["inputs"]["initial_weights_sha256"],
            sha256(b"local weights").hexdigest(),
        )
        self.assertEqual(record["output"]["runtime_class_labels"], ["player"])
        self.assertFalse(
            record["evaluation_caveat"]["supplied_splits_are_source_independent"]
        )
        self.assertTrue(record["evaluation_caveat"]["metrics_may_be_optimistic"])
        self.assertEqual(record["evaluation_caveat"]["cross_split_source_groups"], 1)


if __name__ == "__main__":
    unittest.main()
