from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scripts.build_direct_head_crop_dataset import build_crop_dataset


class BuildDirectHeadCropDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_builds_exact_320_crops_with_player_and_head_labels(self) -> None:
        source = self.root / "source"
        for split in ("train", "val", "test"):
            (source / "images" / split).mkdir(parents=True)
            (source / "labels" / split).mkdir(parents=True)
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(source / "images" / split / "sample.jpg"), image))
            (source / "labels" / split / "sample.txt").write_text(
                "0 0.5 0.5 0.25 0.5\n"
                "1 0.5 0.32 0.08 0.10\n",
                encoding="utf-8",
            )
        (source / "dataset.yaml").write_text("names: {0: player, 1: head}\n", encoding="utf-8")
        output = self.root / "crops"

        report = build_crop_dataset(source, output)

        self.assertEqual(report["crops"], 3)
        self.assertEqual(report["positive_head_crops"], 3)
        for split in ("train", "val", "test"):
            images = list((output / "images" / split).glob("*.jpg"))
            labels = list((output / "labels" / split).glob("*.txt"))
            self.assertEqual(len(images), 1)
            self.assertEqual(len(labels), 1)
            crop = cv2.imread(str(images[0]))
            self.assertEqual(crop.shape, (320, 320, 3))
            classes = [
                line.split()[0]
                for line in labels[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(classes, ["0", "1"])

    def test_builds_640_crops_when_requested(self) -> None:
        source = self.root / "source640"
        for split in ("train", "val", "test"):
            (source / "images" / split).mkdir(parents=True)
            (source / "labels" / split).mkdir(parents=True)
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            self.assertTrue(
                cv2.imwrite(str(source / "images" / split / "sample.jpg"), image)
            )
            (source / "labels" / split / "sample.txt").write_text(
                "0 0.5 0.5 0.20 0.40\n"
                "1 0.5 0.36 0.05 0.06\n",
                encoding="utf-8",
            )
        (source / "dataset.yaml").write_text(
            "names: {0: player, 1: head}\n", encoding="utf-8"
        )
        output = self.root / "crops640"

        report = build_crop_dataset(source, output, crop_size=640)

        self.assertEqual(report["crop_size"], 640)
        for split in ("train", "val", "test"):
            images = list((output / "images" / split).glob("*.jpg"))
            self.assertEqual(len(images), 1)
            crop = cv2.imread(str(images[0]))
            self.assertEqual(crop.shape, (640, 640, 3))


if __name__ == "__main__":
    unittest.main()