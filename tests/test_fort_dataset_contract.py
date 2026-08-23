from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.fort_dataset_contract import (
    DatasetContractError,
    GROUPED_DATASET_YAML,
    build_dataset_contract,
    verify_grouped_dataset_metadata,
    verify_dataset_contract,
)


class FortDatasetContractTests(unittest.TestCase):
    @staticmethod
    def _dataset(root: Path) -> None:
        for split in ("train", "valid", "test"):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / f"{split}.jpg").write_bytes(f"image-{split}".encode())
            (label_dir / f"{split}.txt").write_text(
                "0 0.5 0.5 0.25 0.5\n", encoding="utf-8"
            )

    def test_contract_is_deterministic_and_verifies_exact_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)

            first = build_dataset_contract(root)
            second = build_dataset_contract(root)

            self.assertEqual(first, second)
            self.assertEqual(verify_dataset_contract(root, first), first)
            self.assertEqual(first["splits"]["train"]["images"], 1)
            self.assertEqual(first["splits"]["train"]["boxes"], 1)
            self.assertRegex(first["content_sha256"], r"^[0-9a-f]{64}$")

    def test_rejects_empty_split_when_building_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind in ("images", "labels"):
                for split in ("train", "valid", "test"):
                    (root / kind / split).mkdir(parents=True)

            with self.assertRaisesRegex(DatasetContractError, "image split is empty"):
                build_dataset_contract(root)

    def test_rejects_tampered_missing_or_extra_members(self) -> None:
        mutations = {
            "tampered image": lambda root: (root / "images/train/train.jpg").write_bytes(
                b"changed image"
            ),
            "tampered label": lambda root: (root / "labels/train/train.txt").write_text(
                "0 0.4 0.5 0.25 0.5\n", encoding="utf-8"
            ),
            "missing image": lambda root: (root / "images/train/train.jpg").unlink(),
            "extra image": lambda root: (root / "images/train/extra.jpg").write_bytes(
                b"extra image"
            ),
            "extra alternate-format image": lambda root: (
                root / "images/train/extra.bmp"
            ).write_bytes(b"extra image"),
            "extra decoded cache": lambda root: (
                root / "images/train/train.npy"
            ).write_bytes(b"untrusted cache"),
            "nested image directory": lambda root: (
                root / "images/train/nested"
            ).mkdir(),
            "missing label": lambda root: (root / "labels/train/train.txt").unlink(),
            "extra label": lambda root: (root / "labels/train/extra.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._dataset(root)
                contract = build_dataset_contract(root)

                mutate(root)

                with self.assertRaises(DatasetContractError):
                    verify_dataset_contract(root, contract)

    def test_rejects_contract_aggregate_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)
            contract = build_dataset_contract(root)
            contract["splits"]["train"]["images"] = 99

            with self.assertRaisesRegex(DatasetContractError, "aggregate mismatch"):
                verify_dataset_contract(root, contract)

    def test_grouped_yaml_is_an_exact_semantic_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "fort_cuh_grouped.yaml"
            data.write_text(GROUPED_DATASET_YAML, encoding="utf-8")
            (root / "labels.txt").write_text("player\n", encoding="utf-8")

            verify_grouped_dataset_metadata(data)
            data.write_text(
                GROUPED_DATASET_YAML + "train: alternate/images\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(DatasetContractError, "exact generated"):
                verify_grouped_dataset_metadata(data)

    def test_rejects_unsealed_ultralytics_root_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)
            contract = build_dataset_contract(root)
            (root / "labels" / "valid.cache").write_bytes(b"pickle-like cache")

            with self.assertRaisesRegex(
                DatasetContractError, "uncontracted dataset runtime member"
            ):
                verify_dataset_contract(root, contract)


if __name__ == "__main__":
    unittest.main()
