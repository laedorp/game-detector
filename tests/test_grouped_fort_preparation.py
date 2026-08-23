from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.prepare_fort_cuh_grouped import (
    Candidate,
    GroupedPreparationError,
    VisualSignature,
    _canonical_source_group_key,
    assignment_balance_report,
    assign_source_groups,
    prepare_grouped_dataset,
    refine_visual_source_groups,
    visual_similarity_edges,
)
from scripts.prepare_fort_cuh import prepare_dataset
from tests.test_fort_training_pipeline import SyntheticFortArchive


class GroupedFortPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "fort.zip"
        SyntheticFortArchive.write(self.archive, video_overlap=True)
        self._add_unique_positive_groups(self.archive)
        self.prepared = self.root / "prepared"
        prepare_dataset(self.archive, self.prepared, expected_sha256=None)

    @staticmethod
    def _add_unique_positive_groups(archive_path: Path) -> None:
        """Make the tiny synthetic archive feasible under strict eval capacity."""

        revised = archive_path.with_suffix(".revised.zip")
        extra_counts = {"train": 30, "valid": 6, "test": 4}
        with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(revised, "w") as target:
            for info in source.infolist():
                data = source.read(info.filename)
                split = info.filename.partition("/")[0]
                if info.filename == f"{split}/_annotations.coco.json" and split in extra_counts:
                    coco = json.loads(data)
                    starting_image_id = 10_000 + {"train": 0, "valid": 100, "test": 200}[split]
                    starting_annotation_id = 20_000 + {"train": 0, "valid": 100, "test": 200}[split]
                    for index in range(extra_counts[split]):
                        image_id = starting_image_id + index
                        name = f"unique_{split}_{index:03d}.jpg"
                        coco["images"].append(
                            {"id": image_id, "file_name": name, "width": 100, "height": 100}
                        )
                        coco["annotations"].append(
                            {
                                "id": starting_annotation_id + index,
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": [20, 10, 30, 60],
                                "iscrowd": 0,
                            }
                        )
                    data = json.dumps(coco, sort_keys=True).encode()
                target.writestr(info, data)
            for split, count in extra_counts.items():
                for index in range(count):
                    target.writestr(
                        f"{split}/unique_{split}_{index:03d}.jpg",
                        f"unique {split} image {index}".encode(),
                    )
        revised.replace(archive_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_assignment_is_deterministic_and_never_splits_a_group(self) -> None:
        candidates = [
            Candidate(
                original_split="train",
                archive_name=f"train/{group}-{index}.jpg",
                file_name=f"{group}-{index}.jpg",
                source_group=group,
                label_lines=("0 0.5 0.5 0.2 0.4",) * (index + 1),
            )
            for group in ("a", "b", "c", "d", "e", "f")
            for index in range(2)
        ]

        ratios = {"train": 0.34, "valid": 0.33, "test": 0.33}
        first = assign_source_groups(candidates, seed=7, ratios=ratios)
        second = assign_source_groups(tuple(reversed(candidates)), seed=7, ratios=ratios)

        self.assertEqual(first, second)
        self.assertEqual(set(first), {"a", "b", "c", "d", "e", "f"})
        self.assertEqual(set(first.values()), {"train", "valid", "test"})

    def test_grouped_output_has_no_cross_split_source_group(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        first_report = prepare_grouped_dataset(
            self.archive,
            self.prepared,
            first,
            expected_sha256=None,
            visual_grouping=False,
        )
        second_report = prepare_grouped_dataset(
            self.archive,
            self.prepared,
            second,
            expected_sha256=None,
            visual_grouping=False,
        )

        self.assertEqual(first_report, second_report)
        self.assertEqual(first_report["cross_split_source_groups"], 0)
        self.assertTrue(first_report["ambiguous_partial_label_images_are_never_negatives"])
        self.assertEqual(first_report["reviewed_negative_images"], [])
        self.assertGreater(first_report["splits"]["train"]["images"], 0)
        self.assertGreater(first_report["splits"]["valid"]["images"], 0)
        self.assertGreater(first_report["splits"]["test"]["images"], 0)
        grouped_attribution = (first / "ATTRIBUTION.md").read_text(encoding="utf-8")
        self.assertIn(
            (self.prepared / "ATTRIBUTION.md").read_text(encoding="utf-8").rstrip(),
            grouped_attribution,
        )
        self.assertIn("Group-aware split refinement", grouped_attribution)

        def snapshot(root: Path) -> dict[str, bytes]:
            return {
                str(path.relative_to(root)): path.read_bytes()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }

        # Manifests differ only because they pin the prepared manifest, not paths.
        self.assertEqual(snapshot(first), snapshot(second))

    def test_reviewed_negative_must_be_annotation_free(self) -> None:
        archive_digest = __import__("hashlib").sha256(self.archive.read_bytes()).hexdigest()
        review = self.root / "negative-review.json"
        review.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "archive_sha256": archive_digest,
                    "reviewer": "unit-test",
                    "reviewed_at": "2026-08-12T00:00:00Z",
                    "verified_empty_images": ["train/head_only.jpg"],
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(GroupedPreparationError, "ambiguous"):
            prepare_grouped_dataset(
                self.archive,
                self.prepared,
                self.root / "grouped",
                negative_review=review,
                expected_sha256=None,
                visual_grouping=False,
            )

    def test_reviewed_zero_annotation_image_becomes_empty_label(self) -> None:
        # Add a source image with no annotations, then prepare the positive-only base.
        augmented = self.root / "with-background.zip"
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(augmented, "w") as target:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == "train/_annotations.coco.json":
                    coco = json.loads(data)
                    coco["images"].append(
                        {"id": 999, "file_name": "empty.jpg", "width": 100, "height": 100}
                    )
                    data = json.dumps(coco, sort_keys=True).encode()
                target.writestr(info, data)
            target.writestr("train/empty.jpg", b"empty background")
        prepared = self.root / "prepared-background"
        prepare_dataset(augmented, prepared, expected_sha256=None)
        digest = __import__("hashlib").sha256(augmented.read_bytes()).hexdigest()
        review = self.root / "review.json"
        review.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "archive_sha256": digest,
                    "reviewer": "unit-test",
                    "reviewed_at": "2026-08-12T00:00:00Z",
                    "verified_empty_images": ["train/empty.jpg"],
                }
            ),
            encoding="utf-8",
        )
        output = self.root / "grouped-background"

        report = prepare_grouped_dataset(
            augmented,
            prepared,
            output,
            negative_review=review,
            expected_sha256=None,
            visual_grouping=False,
        )

        empty_labels = [path for path in output.glob("labels/*/empty.txt")]
        self.assertEqual(len(empty_labels), 1)
        self.assertEqual(empty_labels[0].read_text(encoding="utf-8"), "")
        self.assertEqual(
            sum(split["reviewed_negative_images"] for split in report["splits"].values()),
            1,
        )

    def test_filename_canonicalization_merges_export_and_capture_variants(self) -> None:
        self.assertEqual(
            _canonical_source_group_key(
                "Fortnite-Screenshot-2022-08-20-11-51-07-00_png_jpg.rf.aaa.jpg"
            ),
            _canonical_source_group_key(
                "Fortnite-Screenshot-2022-08-20-11-51-08-22_png.rf.bbb.jpg"
            ),
        )
        self.assertEqual(
            _canonical_source_group_key("Screenshot-2023-02-11-102716_png.rf.a.jpg"),
            _canonical_source_group_key("Screenshot-2023-02-11-103213_png_jpg.rf.b.jpg"),
        )
        self.assertEqual(
            _canonical_source_group_key("frame_000888_PNG.rf.a.jpg"),
            _canonical_source_group_key("frame_000891_PNG.rf.b.jpg"),
        )

    def test_visual_edges_are_transitively_unioned(self) -> None:
        candidates = [
            Candidate(
                original_split="train",
                archive_name=f"train/{name}.jpg",
                file_name=f"{name}.jpg",
                source_group=f"original_file:{name}",
                label_lines=("0 0.5 0.5 0.2 0.4",),
            )
            for name in ("a", "b", "c", "d")
        ]
        signatures = {
            "train/a.jpg": VisualSignature(0, bytes([0]) * 768),
            "train/b.jpg": VisualSignature(0, bytes([6]) * 768),
            "train/c.jpg": VisualSignature(0, bytes([12]) * 768),
            "train/d.jpg": VisualSignature((1 << 20) - 1, bytes([100]) * 768),
        }

        edges = visual_similarity_edges(candidates, signatures)
        refined, clusters = refine_visual_source_groups(candidates, edges)

        self.assertEqual([(edge.left, edge.right) for edge in edges], [
            ("train/a.jpg", "train/b.jpg"),
            ("train/b.jpg", "train/c.jpg"),
        ])
        groups = {candidate.file_name: candidate.source_group for candidate in refined}
        self.assertEqual(groups["a.jpg"], groups["b.jpg"])
        self.assertEqual(groups["b.jpg"], groups["c.jpg"])
        self.assertNotEqual(groups["c.jpg"], groups["d.jpg"])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(
            clusters[0]["source_groups"],
            ["original_file:a", "original_file:b", "original_file:c"],
        )

    def test_giant_groups_are_train_only_and_assignment_is_deterministic(self) -> None:
        candidates: list[Candidate] = []
        sizes = {
            "giant": 40,
            "valid-only": 10,
            **{f"small-{index}": 2 for index in range(18)},
        }
        for group, size in sizes.items():
            candidates.extend(
                Candidate(
                    original_split="train",
                    archive_name=f"train/{group}-{index}.jpg",
                    file_name=f"{group}-{index}.jpg",
                    source_group=f"capture_session:{group}",
                    label_lines=("0 0.5 0.5 0.2 0.4",),
                )
                for index in range(size)
            )

        first = assign_source_groups(candidates, seed=11)
        second = assign_source_groups(tuple(reversed(candidates)), seed=11)
        report = assignment_balance_report(candidates, first, {"train": .75, "valid": .15, "test": .1})

        self.assertEqual(first, second)
        self.assertEqual(first["capture_session:giant"], "train")
        self.assertNotEqual(first["capture_session:valid-only"], "test")
        self.assertEqual(set(first.values()), {"train", "valid", "test"})
        self.assertEqual(report["train_only_giant_group_count"], 1)
        self.assertEqual(
            report["largest_source_group"]["source_group"],
            "capture_session:giant",
        )
        valid_only = next(
            group
            for group in report["train_only_giant_groups"]
            if group["source_group"] == "capture_session:giant"
        )
        self.assertEqual(valid_only["eligible_evaluation_splits"], [])
        self.assertIn("global target capacity", report["policy"]["giant_group_rule"])
        self.assertIn("neither evaluation split", report["policy"]["giant_group_rule"])
        distribution = report["far_object_distribution"]
        self.assertEqual(
            sum(distribution["total"].values()),
            sum(len(candidate.label_lines) for candidate in candidates),
        )
        self.assertEqual(
            distribution["total"],
            {
                bucket: sum(
                    distribution["splits"][split][bucket]
                    for split in ("train", "valid", "test")
                )
                for bucket in distribution["total"]
            },
        )
        self.assertIn("bootstrap confidence intervals", distribution["sampling_caveat"])


if __name__ == "__main__":
    unittest.main()
