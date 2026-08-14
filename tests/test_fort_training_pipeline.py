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
from scripts.fort_dataset_contract import GROUPED_DATASET_YAML, build_dataset_contract
from scripts.train_fort_model import (
    TrainingConfig,
    TrainingConfigurationError,
    _initial_run_contract,
    _write_reproducibility_record,
    build_parser as build_training_parser,
    config_from_args,
    adopt_interrupted_run,
    resume_training_arguments,
    run_training,
    training_arguments,
    validate_resume_checkpoint,
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
        self.assertIs(options["rect"], False)
        self.assertEqual(options["cache"], False)
        self.assertEqual(options["save_period"], 1)

    def test_grouped_zero_leakage_manifest_is_accepted_and_recorded(self) -> None:
        grouped = self.root / "grouped"
        for kind in ("images", "labels"):
            for split in ("train", "valid", "test"):
                (grouped / kind / split).mkdir(parents=True)
        for split in ("train", "valid", "test"):
            (grouped / "images" / split / f"{split}.jpg").write_bytes(
                f"{split} image".encode()
            )
            (grouped / "labels" / split / f"{split}.txt").write_text(
                "0 0.5 0.5 0.2 0.4\n", encoding="utf-8"
            )
        (grouped / "labels.txt").write_text("player\n", encoding="utf-8")
        (grouped / "fort_cuh_grouped.yaml").write_text(
            GROUPED_DATASET_YAML,
            encoding="utf-8",
        )
        (grouped / "manifest.json").write_text(
            json.dumps(
                {
                    "source_archive_sha256": "a" * 64,
                    "cross_split_source_groups": 0,
                    "cross_split_visual_similarity_edges": 0,
                    "visual_grouping": {"enabled": True, "version": 1},
                    "ambiguous_partial_label_images_are_never_negatives": True,
                    "assignment": {"version": 3, "seed": 0},
                    "dataset_contract": build_dataset_contract(grouped),
                    "splits": {
                        split: {"images": 1, "boxes": 1, "source_groups": 1}
                        for split in ("train", "valid", "test")
                    },
                }
            ),
            encoding="utf-8",
        )
        config = self.config(data=grouped / "fort_cuh_grouped.yaml", run_test=False)

        manifest = validate_training_config(config)

        self.assertEqual(manifest["cross_split_source_groups"], 0)
        (grouped / "labels" / "test" / "test.txt").write_text(
            "0 0.4 0.5 0.2 0.4\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(TrainingConfigurationError, "label hash mismatch"):
            validate_training_config(config)

    def test_grouped_training_rejects_duplicate_or_alternate_yaml_paths(self) -> None:
        grouped = self.root / "grouped-yaml"
        for kind in ("images", "labels"):
            for split in ("train", "valid", "test"):
                (grouped / kind / split).mkdir(parents=True)
        for split in ("train", "valid", "test"):
            (grouped / "images" / split / f"{split}.jpg").write_bytes(split.encode())
            (grouped / "labels" / split / f"{split}.txt").write_text(
                "0 0.5 0.5 0.2 0.4\n", encoding="utf-8"
            )
        (grouped / "labels.txt").write_text("player\n", encoding="utf-8")
        data = grouped / "fort_cuh_grouped.yaml"
        canonical = (
            "# Generated by scripts/prepare_fort_cuh_grouped.py\n"
            "train: images/train\nval: images/valid\ntest: images/test\n"
            "names:\n  0: player\n"
        )
        data.write_text(canonical, encoding="utf-8")
        (grouped / "manifest.json").write_text(
            json.dumps(
                {
                    "cross_split_source_groups": 0,
                    "cross_split_visual_similarity_edges": 0,
                    "visual_grouping": {"enabled": True},
                    "ambiguous_partial_label_images_are_never_negatives": True,
                    "assignment": {"version": 3},
                    "dataset_contract": build_dataset_contract(grouped),
                    "splits": {
                        split: {"images": 1, "boxes": 1, "source_groups": 1}
                        for split in ("train", "valid", "test")
                    },
                }
            ),
            encoding="utf-8",
        )
        config = self.config(data=data, run_test=False)
        data.write_text(canonical + "train: alternate/images\n", encoding="utf-8")
        with self.assertRaisesRegex(TrainingConfigurationError, "exact generated"):
            validate_training_config(config)
        data.write_text(canonical.replace("images/valid", "alternate/valid"), encoding="utf-8")
        with self.assertRaisesRegex(TrainingConfigurationError, "exact generated"):
            validate_training_config(config)

    def test_exact_grouped_training_forbids_decoded_image_cache_mode(self) -> None:
        grouped = self.root / "grouped-cache-mode"
        for kind in ("images", "labels"):
            for split in ("train", "valid", "test"):
                (grouped / kind / split).mkdir(parents=True)
        for split in ("train", "valid", "test"):
            (grouped / "images" / split / f"{split}.jpg").write_bytes(split.encode())
            (grouped / "labels" / split / f"{split}.txt").write_text(
                "0 0.5 0.5 0.2 0.4\n", encoding="utf-8"
            )
        (grouped / "labels.txt").write_text("player\n", encoding="utf-8")
        (grouped / "fort_cuh_grouped.yaml").write_text(
            GROUPED_DATASET_YAML, encoding="utf-8"
        )
        (grouped / "manifest.json").write_text(
            json.dumps(
                {
                    "cross_split_source_groups": 0,
                    "cross_split_visual_similarity_edges": 0,
                    "visual_grouping": {"enabled": True},
                    "ambiguous_partial_label_images_are_never_negatives": True,
                    "assignment": {"version": 3},
                    "dataset_contract": build_dataset_contract(grouped),
                    "splits": {
                        split: {"images": 1, "boxes": 1, "source_groups": 1}
                        for split in ("train", "valid", "test")
                    },
                }
            ),
            encoding="utf-8",
        )

        config = self.config(
            data=grouped / "fort_cuh_grouped.yaml", cache="disk", run_test=False
        )

        with self.assertRaisesRegex(TrainingConfigurationError, "requires --cache none"):
            validate_training_config(config)

    def test_grouped_reproducibility_record_does_not_overclaim_independence(self) -> None:
        grouped = self.root / "grouped-record"
        for kind in ("images", "labels"):
            for split in ("train", "valid", "test"):
                (grouped / kind / split).mkdir(parents=True)
        for split in ("train", "valid", "test"):
            (grouped / "images" / split / f"{split}.jpg").write_bytes(
                f"{split} image".encode()
            )
            (grouped / "labels" / split / f"{split}.txt").write_text(
                "0 0.5 0.5 0.2 0.4\n", encoding="utf-8"
            )
        (grouped / "labels.txt").write_text("player\n", encoding="utf-8")
        (grouped / "fort_cuh_grouped.yaml").write_text(
            GROUPED_DATASET_YAML,
            encoding="utf-8",
        )
        (grouped / "manifest.json").write_text(
            json.dumps(
                {
                    "source_archive_sha256": "a" * 64,
                    "cross_split_source_groups": 0,
                    "cross_split_visual_similarity_edges": 0,
                    "visual_grouping": {"enabled": True, "version": 1},
                    "ambiguous_partial_label_images_are_never_negatives": True,
                    "assignment": {"version": 3, "seed": 0},
                    "dataset_contract": build_dataset_contract(grouped),
                    "splits": {
                        split: {"images": 1, "boxes": 1, "source_groups": 1}
                        for split in ("train", "valid", "test")
                    },
                }
            ),
            encoding="utf-8",
        )
        config = self.config(
            data=grouped / "fort_cuh_grouped.yaml", name="grouped_record", run_test=False
        )

        class FakeGroupedYOLO:
            def __init__(self, _source: str) -> None:
                self.task = "detect"
                self.trainer = None

            def train(self, **kwargs: object) -> None:
                run_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                weights = run_dir / "weights"
                weights.mkdir(parents=True)
                (run_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
                best = weights / "best.pt"
                best.write_bytes(b"grouped")
                self.trainer = SimpleNamespace(best=best)

        run_training(config, yolo_class=FakeGroupedYOLO)
        record = json.loads(
            (config.run_dir / "reproducibility.json").read_text(encoding="utf-8")
        )

        caveat = record["evaluation_caveat"]
        self.assertFalse(caveat["supplied_splits_are_source_independent"])
        self.assertEqual(caveat["conservative_filename_source_groups_cross_splits"], 0)
        self.assertEqual(caveat["known_visual_similarity_edges_cross_splits"], 0)
        self.assertTrue(caveat["metrics_may_be_optimistic"])
        self.assertIn("cannot prove visual independence", caveat["reason"])
        self.assertEqual(record["inputs"]["dataset_split_assignment"], {"version": 3, "seed": 0})

    def test_validation_refuses_reusing_a_run_directory(self) -> None:
        config = self.config()
        config.run_dir.mkdir(parents=True)
        with self.assertRaisesRegex(TrainingConfigurationError, "already exists"):
            validate_training_config(config)

    def test_adoption_is_explicit_audit_only_and_resume_is_stateful(self) -> None:
        # ``slots=True`` has no __dict__; reconstruct from serializable fields.
        base = self.config(name="resume_run", epochs=4, patience=2, run_test=False)
        manifest = validate_training_config(base)
        base.run_dir.mkdir(parents=True)
        contract = _initial_run_contract(base, manifest)
        weights = base.run_dir / "weights"
        weights.mkdir()
        (base.run_dir / "results.csv").write_text(
            "epoch,time,metrics/mAP50-95(B)\n1,1.0,0.1\n", encoding="utf-8"
        )
        checkpoint_args = dict(contract["training_arguments"])
        checkpoint_args.update(
            {
                "data": str(base.data), "model": str(base.weights),
                "project": str(base.project), "name": base.name,
                "save_dir": str(base.run_dir), "workers": 0,
            }
        )
        checkpoint_args["warmup_bias_lr"] = 0.0
        checkpoint: dict[str, object] = {
            "epoch": 0, "optimizer": {"state": {1: {"step": 1}}},
            "scaler": {}, "ema": object(), "updates": 1,
            "version": contract["environment"]["ultralytics"],
            "train_args": checkpoint_args,
            "train_results": {
                "epoch": [1],
                "time": [1.0],
                "metrics/mAP50-95(B)": [0.1],
            },
            "best_fitness": 0.1,
        }
        for name in ("last.pt", "epoch0.pt", "best.pt"):
            (weights / name).write_bytes(b"checkpoint")
        (base.run_dir / "args.yaml").write_text("audited fixture\n", encoding="utf-8")
        resume = TrainingConfig(
            data=base.data, weights=base.weights, project=base.project, name=base.name,
            epochs=base.epochs, patience=base.patience, batch=base.batch,
            imgsz=base.imgsz, device=base.device, workers=base.workers,
            threads=base.threads, cache=base.cache, seed=base.seed,
            smoke_test=False, run_test=False,
            resume_from=(weights / "last.pt").resolve(), adopt_interrupted_run=True,
        )

        manifest = validate_training_config(resume)
        with mock.patch(
            "scripts.train_fort_model._read_flat_yaml",
            return_value={**checkpoint_args, "workers": base.workers, "warmup_bias_lr": 0.1},
        ):
            path = adopt_interrupted_run(
                resume, manifest, checkpoint_loader=lambda _path: checkpoint
            )
        self.assertTrue(path.is_file())
        adopted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(adopted["adoption"]["adopted_checkpoint_epoch"], 1)
        self.assertEqual(
            adopted["adoption"]["adopted_checkpoint_sha256"],
            sha256(b"checkpoint").hexdigest(),
        )

        resume = TrainingConfig(
            data=base.data, weights=base.weights, project=base.project, name=base.name,
            epochs=base.epochs, patience=base.patience, batch=base.batch,
            imgsz=base.imgsz, device=base.device, workers=base.workers,
            threads=base.threads, cache=base.cache, seed=base.seed,
            smoke_test=False, run_test=False,
            resume_from=(weights / "last.pt").resolve(),
        )
        state = validate_resume_checkpoint(
            resume, validate_training_config(resume),
            checkpoint_loader=lambda _path: checkpoint,
        )
        self.assertEqual(state.completed_epoch, 1)
        options = resume_training_arguments(resume, state)
        self.assertEqual(options["resume"], str(weights / "last.pt"))
        self.assertNotIn("data", options)
        self.assertNotIn("project", options)

        checkpoint_results = checkpoint["train_results"]
        assert isinstance(checkpoint_results, dict)
        checkpoint_results["time"] = [2.0]
        with self.assertRaisesRegex(TrainingConfigurationError, "exactly match"):
            validate_resume_checkpoint(
                resume,
                validate_training_config(resume),
                checkpoint_loader=lambda _path: checkpoint,
            )
        checkpoint_results["time"] = [1.0]

        # Ultralytics rewrites train_args.model to last.pt after the first
        # resume. That exact evolution must survive another interruption.
        checkpoint_args["model"] = str(weights / "last.pt")
        repeated = validate_resume_checkpoint(
            resume,
            validate_training_config(resume),
            checkpoint_loader=lambda _path: checkpoint,
        )
        self.assertEqual(repeated.completed_epoch, 1)
        checkpoint_args["model"] = str(self.root / "unrelated.pt")
        with self.assertRaisesRegex(TrainingConfigurationError, "neither the pinned"):
            validate_resume_checkpoint(
                resume,
                validate_training_config(resume),
                checkpoint_loader=lambda _path: checkpoint,
            )
        checkpoint_args["model"] = str(weights / "last.pt")

        # last.pt is overwritten by resumed training. Provenance must retain
        # the starting hash captured by validation, not hash the final file.
        starting_sha256 = repeated.sha256
        (weights / "last.pt").write_bytes(b"later checkpoint")
        _write_reproducibility_record(
            resume,
            manifest,
            weights / "best.pt",
            resume_state=repeated,
        )
        record = json.loads((base.run_dir / "reproducibility.json").read_text())
        self.assertEqual(record["inputs"]["resumed_from"]["sha256"], starting_sha256)
        self.assertEqual(
            record["inputs"]["resumed_from"]["sha256_scope"],
            "captured_before_resume_training",
        )

    def test_fresh_contract_has_canonical_fields_and_accepts_first_resume(self) -> None:
        base = self.config(name="fresh_resume", epochs=4, run_test=False)
        manifest = validate_training_config(base)
        base.run_dir.mkdir(parents=True)
        contract = _initial_run_contract(base, manifest)
        self.assertEqual(contract["training_script_sha256_status"], "captured_at_launch")
        self.assertIsNone(contract["adoption"])
        (base.run_dir / "initial_run_contract.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
        weights = base.run_dir / "weights"
        weights.mkdir()
        for name in ("last.pt", "epoch0.pt", "best.pt"):
            (weights / name).write_bytes(b"fresh checkpoint")
        (base.run_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
        checkpoint_args = dict(contract["training_arguments"])
        checkpoint_args.update(
            {
                "data": str(base.data),
                "model": str(base.weights),
                "project": str(base.project),
                "name": base.name,
                "save_dir": str(base.run_dir),
                "workers": 0,
            }
        )
        checkpoint = {
            "epoch": 0,
            "optimizer": {"state": {1: {"step": 1}}},
            "scaler": {},
            "ema": object(),
            "updates": 1,
            "version": contract["environment"]["ultralytics"],
            "train_args": checkpoint_args,
            "train_results": {"epoch": [1]},
        }
        resume = TrainingConfig(
            data=base.data,
            weights=base.weights,
            project=base.project,
            name=base.name,
            epochs=base.epochs,
            patience=base.patience,
            batch=base.batch,
            imgsz=base.imgsz,
            device=base.device,
            workers=base.workers,
            threads=base.threads,
            cache=base.cache,
            seed=base.seed,
            smoke_test=False,
            run_test=False,
            resume_from=(weights / "last.pt").resolve(),
        )
        state = validate_resume_checkpoint(
            resume,
            validate_training_config(resume),
            checkpoint_loader=lambda _path: checkpoint,
        )
        self.assertEqual(state.completed_epoch, 1)

    def test_adoption_rejects_tampered_args_and_never_replaces_contract(self) -> None:
        base = self.config(name="tampered_resume", epochs=4, run_test=False)
        manifest = validate_training_config(base)
        base.run_dir.mkdir(parents=True)
        contract = _initial_run_contract(base, manifest)
        weights = base.run_dir / "weights"
        weights.mkdir()
        for name in ("last.pt", "epoch0.pt", "best.pt"):
            (weights / name).write_bytes(b"same")
        (base.run_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
        checkpoint_args = dict(contract["training_arguments"])
        checkpoint_args.update(
            {"data": str(base.data), "model": str(base.weights),
             "project": str(base.project), "name": base.name,
             "save_dir": str(base.run_dir), "workers": 0}
        )
        checkpoint_args["warmup_bias_lr"] = 0.0
        (base.run_dir / "args.yaml").write_text(
            "tampered fixture\n", encoding="utf-8"
        )
        checkpoint = {
            "epoch": 0, "optimizer": {"x": 1}, "scaler": {}, "ema": object(),
            "updates": 1, "version": contract["environment"]["ultralytics"],
            "train_args": checkpoint_args, "train_results": {"epoch": [1]},
        }
        resume = TrainingConfig(
            data=base.data, weights=base.weights, project=base.project, name=base.name,
            epochs=base.epochs, patience=base.patience, batch=base.batch,
            imgsz=base.imgsz, device=base.device, workers=base.workers,
            threads=base.threads, cache=base.cache, seed=base.seed,
            smoke_test=False, run_test=False,
            resume_from=(weights / "last.pt").resolve(), adopt_interrupted_run=True,
        )
        with mock.patch(
            "scripts.train_fort_model._read_flat_yaml",
            return_value={**checkpoint_args, "imgsz": 640, "workers": base.workers, "warmup_bias_lr": 0.1},
        ), self.assertRaisesRegex(TrainingConfigurationError, "outside"):
            adopt_interrupted_run(resume, manifest, checkpoint_loader=lambda _: checkpoint)
        (base.run_dir / "initial_run_contract.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(TrainingConfigurationError, "refusing to replace"):
            adopt_interrupted_run(resume, manifest, checkpoint_loader=lambda _: checkpoint)

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
                (run_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
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
