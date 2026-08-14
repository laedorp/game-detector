from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_strict_fort_eval import (
    StrictEvaluationError,
    build_strict_membership,
    write_strict_evaluation,
)


class StrictFortEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        for split in ("train", "valid", "test"):
            (self.dataset / "images" / split).mkdir(parents=True)
            (self.dataset / "labels" / split).mkdir(parents=True)
        files = {
            "train": ("train-sequence-a.jpg", "train-only.jpg"),
            "valid": ("valid-sequence-a.jpg", "valid-sequence-b.jpg", "valid-only.jpg"),
            "test": (
                "test-sequence-a.jpg",
                "test-sequence-b.jpg",
                "test-only.jpg",
            ),
        }
        for split, names in files.items():
            for index, name in enumerate(names, start=1):
                (self.dataset / "images" / split / name).write_bytes(b"image")
                (self.dataset / "labels" / split / f"{Path(name).stem}.txt").write_text(
                    "0 0.5 0.5 0.25 0.5\n" * index,
                    encoding="utf-8",
                )
        manifest = {
            "conversion": {"output_class_names": ["player"]},
            "leakage": {
                "source_grouping_heuristic": {"version": 1},
                "cross_split_source_group_details": [
                    {
                        "key": "sequence-a",
                        "files": [
                            "train/train-sequence-a.jpg",
                            "valid/valid-sequence-a.jpg",
                            "test/test-sequence-a.jpg",
                        ],
                    },
                    {
                        "key": "sequence-b",
                        "files": [
                            "valid/valid-sequence-b.jpg",
                            "test/test-sequence-b.jpg",
                        ],
                    },
                ],
            },
        }
        (self.dataset / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_filters_validation_against_train_and_test_against_train_and_validation(self) -> None:
        memberships, summaries, heuristic = build_strict_membership(self.dataset)

        self.assertEqual(
            [path.name for path in memberships["valid"]],
            ["valid-only.jpg", "valid-sequence-b.jpg"],
        )
        self.assertEqual(
            [path.name for path in memberships["test"]],
            ["test-only.jpg"],
        )
        self.assertEqual(summaries["valid"].supplied_images, 3)
        self.assertEqual(summaries["valid"].excluded_source_overlap_images, 1)
        self.assertEqual(summaries["test"].excluded_source_overlap_images, 2)
        self.assertEqual(summaries["test"].strict_boxes, 3)
        self.assertEqual(heuristic, {"version": 1})

    def test_writes_portable_evaluation_contract_without_overwriting(self) -> None:
        output = self.root / "strict"
        report = write_strict_evaluation(self.dataset, output)

        self.assertEqual(report["splits"]["test"]["strict_images"], 1)
        self.assertEqual(
            (output / "strict_test.txt").read_text(encoding="utf-8"),
            f"{self.dataset.resolve() / 'images' / 'test' / 'test-only.jpg'}\n",
        )
        yaml_text = (output / "fort_cuh_strict_eval.yaml").read_text(encoding="utf-8")
        self.assertIn(str(output.resolve() / "strict_valid.txt"), yaml_text)
        self.assertIn("0: player", yaml_text)
        with self.assertRaisesRegex(StrictEvaluationError, "refusing to overwrite"):
            write_strict_evaluation(self.dataset, output)

    def test_rejects_unsafe_manifest_membership(self) -> None:
        manifest_file = self.dataset / "manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest["leakage"]["cross_split_source_group_details"][0]["files"][0] = (
            "../outside.jpg"
        )
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(StrictEvaluationError, "unsafe source-group"):
            build_strict_membership(self.dataset)


if __name__ == "__main__":
    unittest.main()
