from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.validate_release_assets import (
    COCO80_LABELS,
    EXPECTED_INPUT_SHAPE,
    FORT_SOURCE_URL,
    ReleaseAssetError,
    validate_release_assets,
)


class FakePort:
    def __init__(self, shape: tuple[int | None, ...]) -> None:
        self.partial_shape = tuple(-1 if value is None else value for value in shape)


class FakeModel:
    def __init__(
        self,
        input_shape: tuple[int | None, ...] = EXPECTED_INPUT_SHAPE,
        output_shapes: tuple[tuple[int | None, ...], ...] = ((1, 300, 6),),
    ) -> None:
        self.inputs = (FakePort(input_shape),)
        self.outputs = tuple(FakePort(shape) for shape in output_shapes)


class FakeCore:
    def __init__(
        self,
        *,
        coco: FakeModel | Exception | None = None,
        fort: FakeModel | Exception | None = None,
    ) -> None:
        self.models = {
            "yolo26n.xml": coco or FakeModel(output_shapes=((1, 84, 2100),)),
            "fort_player.xml": fort or FakeModel(output_shapes=((1, 2100, 5),)),
        }
        self.calls: list[tuple[Path, Path]] = []

    def read_model(self, *, model: str, weights: str) -> FakeModel:
        xml_path = Path(model)
        bin_path = Path(weights)
        self.calls.append((xml_path, bin_path))
        outcome = self.models[xml_path.name]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ReleaseAssetValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self._write_valid_files()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_valid_files(self) -> None:
        coco_dir = self.root / "models" / "yolo26n_openvino_model"
        fort_dir = self.root / "models" / "fort_player_openvino_model"
        coco_dir.mkdir(parents=True)
        fort_dir.mkdir(parents=True)
        (coco_dir / "yolo26n.xml").write_text("<net/>", encoding="utf-8")
        (coco_dir / "yolo26n.bin").write_bytes(b"coco weights")
        (fort_dir / "fort_player.xml").write_text("<net/>", encoding="utf-8")
        (fort_dir / "fort_player.bin").write_bytes(b"fort weights")
        (self.root / "models" / "coco80.txt").write_text(
            "\n".join(COCO80_LABELS) + "\n", encoding="utf-8"
        )
        (self.root / "models" / "fort_player.txt").write_text(
            "player\n", encoding="utf-8"
        )
        (fort_dir / "ATTRIBUTION.md").write_text(
            "FORT-Cuh from Roboflow\n"
            f"Source: {FORT_SOURCE_URL}\n"
            "License: CC BY 4.0\n",
            encoding="utf-8",
        )

    def assert_fails(self, expected: str, core: FakeCore | None = None) -> str:
        with self.assertRaises(ReleaseAssetError) as raised:
            validate_release_assets(self.root, core_factory=lambda: core or FakeCore())
        message = str(raised.exception)
        self.assertIn("Release asset preflight failed", message)
        self.assertIn(expected, message)
        return message

    def test_accepts_explicit_matching_pairs_and_supported_yolo_layouts(self) -> None:
        core = FakeCore()

        summaries = validate_release_assets(self.root, core_factory=lambda: core)

        self.assertEqual(len(summaries), 2)
        self.assertIn("traditional [1,84,N]", summaries[0])
        self.assertIn("traditional [1,N,5]", summaries[1])
        self.assertEqual(len(core.calls), 2)
        for xml_path, bin_path in core.calls:
            self.assertEqual(xml_path.stem, bin_path.stem)
            self.assertTrue(xml_path.is_absolute())
            self.assertTrue(bin_path.is_absolute())

    def test_accepts_end_to_end_output_with_dynamic_detection_count(self) -> None:
        core = FakeCore(
            coco=FakeModel(output_shapes=((1, None, 6),)),
            fort=FakeModel(output_shapes=((1, None, 6),)),
        )

        summaries = validate_release_assets(self.root, core_factory=lambda: core)

        self.assertTrue(all("end-to-end [1,N,6]" in item for item in summaries))

    def test_rejects_empty_or_missing_ir_without_constructing_openvino(self) -> None:
        fort_bin = (
            self.root
            / "models"
            / "fort_player_openvino_model"
            / "fort_player.bin"
        )
        fort_bin.write_bytes(b"")
        constructed = False

        def factory() -> FakeCore:
            nonlocal constructed
            constructed = True
            return FakeCore()

        with self.assertRaisesRegex(ReleaseAssetError, "Fort player detector BIN is empty"):
            validate_release_assets(self.root, core_factory=factory)
        self.assertFalse(constructed)

    def test_rejects_an_ir_pair_openvino_cannot_read(self) -> None:
        core = FakeCore(fort=RuntimeError("weights do not match graph"))

        message = self.assert_fails("OpenVINO could not read the matching", core)

        self.assertIn("fort_player.xml, fort_player.bin", message)
        self.assertIn("weights do not match graph", message)

    def test_rejects_nonstatic_or_wrong_input_shape(self) -> None:
        for shape in ((1, 3, 640, 640), (None, 3, 320, 320), (1, 320, 320, 3)):
            with self.subTest(shape=shape):
                core = FakeCore(fort=FakeModel(input_shape=shape))
                message = self.assert_fails("expected static NCHW", core)
                self.assertIn(str(EXPECTED_INPUT_SHAPE), message)

    def test_rejects_multiple_or_unsupported_outputs(self) -> None:
        cases = (
            (
                FakeModel(output_shapes=((1, 300, 6), (1, 10, 4))),
                "exactly one YOLO output",
            ),
            (
                FakeModel(output_shapes=((1, 7, 100),)),
                "output shape (1, 7, 100) is unsupported",
            ),
            (
                FakeModel(output_shapes=((2, 300, 6),)),
                "output shape (2, 300, 6) is unsupported",
            ),
        )
        for model, expected in cases:
            with self.subTest(expected=expected):
                self.assert_fails(expected, FakeCore(fort=model))

    def test_requires_exact_coco_and_fort_labels(self) -> None:
        fort_labels = self.root / "models" / "fort_player.txt"
        fort_labels.write_text("Player\n", encoding="utf-8")
        self.assert_fails("Fort player detector labels are wrong")

        fort_labels.write_text("player\n", encoding="utf-8")
        coco_labels = self.root / "models" / "coco80.txt"
        changed = list(COCO80_LABELS)
        changed[0] = "human"
        coco_labels.write_text("\n".join(changed) + "\n", encoding="utf-8")
        self.assert_fails("COCO detector labels are wrong")

    def test_requires_complete_license_and_source_attribution(self) -> None:
        attribution = (
            self.root
            / "models"
            / "fort_player_openvino_model"
            / "ATTRIBUTION.md"
        )
        attribution.write_text("Roboflow source, but no license or URL\n", encoding="utf-8")

        message = self.assert_fails("attribution", FakeCore())

        self.assertIn("CC BY 4.0", message)
        self.assertIn(FORT_SOURCE_URL, message)


if __name__ == "__main__":
    unittest.main()
