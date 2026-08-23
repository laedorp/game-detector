from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.prepare_valorant_player_head import prepare_dataset


class PrepareValorantPlayerHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_archive(self, *, unsafe: bool = False) -> Path:
        archive = self.root / "source.zip"
        prefix = "Valorant object detection image dataset/labeled data"
        with zipfile.ZipFile(archive, "w") as output:
            for group in range(20):
                stem = f"frame_{group * 300:05d}_png.rf.{group:032x}"
                output.writestr(f"{prefix}/images/{stem}.jpg", b"jpeg" + bytes([group]))
                output.writestr(
                    f"{prefix}/labels/{stem}.txt",
                    "0 0.5 0.5 0.1 0.1\n"
                    "1 0.5 0.6 0.3 0.6\n"
                    "2 0.5 0.3 0.1 0.1\n"
                    "3 0.2 0.5 0.2 0.4\n",
                )
            output.writestr(
                "Valorant object detection image dataset/data.yaml",
                "nc: 4\nnames: ['crosshair', 'enemy', 'enemyhead', 'teammate']\n",
            )
            if unsafe:
                output.writestr("../escape.jpg", b"unsafe")
        return archive

    def test_remaps_player_and_head_and_keeps_temporal_groups_together(self) -> None:
        archive = self.make_archive()
        output = self.root / "prepared"

        report = prepare_dataset(
            archive,
            output,
            expected_sha256=sha256(archive.read_bytes()).hexdigest(),
        )

        self.assertEqual(report["images"], 20)
        self.assertEqual(report["annotations"], {"player": 20, "head": 20})
        self.assertGreater(report["splits"]["train"], 0)
        self.assertGreater(report["splits"]["val"], 0)
        self.assertGreater(report["splits"]["test"], 0)
        labels = list((output / "labels").glob("*/*.txt"))
        self.assertEqual(len(labels), 20)
        for label in labels:
            lines = label.read_text(encoding="utf-8").splitlines()
            self.assertEqual([line.split()[0] for line in lines], ["0", "1"])
        self.assertTrue((output / "dataset.yaml").is_file())
        attribution = (output / "ATTRIBUTION.md").read_text(encoding="utf-8")
        self.assertIn("CC BY 4.0", attribution)
        self.assertIn("Dasun01/Valorant-Object-Detection-Dataset", attribution)

    def test_rejects_archive_hash_mismatch(self) -> None:
        archive = self.make_archive()

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare_dataset(archive, self.root / "prepared", expected_sha256="0" * 64)

    def test_rejects_unsafe_zip_paths(self) -> None:
        archive = self.make_archive(unsafe=True)

        with self.assertRaisesRegex(ValueError, "unsafe"):
            prepare_dataset(
                archive,
                self.root / "prepared",
                expected_sha256=sha256(archive.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()