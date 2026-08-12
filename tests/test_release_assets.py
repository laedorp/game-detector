from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.validate_release_assets import (
    BALANCED_INPUT_SHAPE,
    COCO80_LABELS,
    EXPECTED_INPUT_SHAPE,
    PLAYER_ATTRIBUTION_MARKERS,
    PLAYER_LABELS,
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
        fast: FakeModel | Exception | None = None,
        balanced: FakeModel | Exception | None = None,
        high_end: FakeModel | Exception | None = None,
        player_fast: FakeModel | Exception | None = None,
        player_balanced: FakeModel | Exception | None = None,
    ) -> None:
        self.models = {
            "yolo26n.xml": fast or FakeModel(output_shapes=((1, 84, 2100),)),
            "yolo26n_416.xml": balanced
            or FakeModel(
                input_shape=BALANCED_INPUT_SHAPE,
                output_shapes=((1, 84, 3549),),
            ),
            "yolo11l.xml": high_end
            or FakeModel(
                input_shape=(1, 3, 640, 640),
                output_shapes=((1, 84, 5670),),
            ),
            "fort_player.xml": player_fast or FakeModel(output_shapes=((1, 300, 6),)),
            "fort_player_416.xml": player_balanced
            or FakeModel(
                input_shape=BALANCED_INPUT_SHAPE,
                output_shapes=((1, 300, 6),),
            ),
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
        balanced_dir = self.root / "models" / "yolo26n_416_openvino_model"
        high_end_dir = self.root / "models" / "yolo11l_openvino_model"
        coco_dir.mkdir(parents=True)
        balanced_dir.mkdir(parents=True)
        high_end_dir.mkdir(parents=True)
        (coco_dir / "yolo26n.xml").write_text("<net/>", encoding="utf-8")
        (coco_dir / "yolo26n.bin").write_bytes(b"coco weights")
        (balanced_dir / "yolo26n_416.xml").write_text("<net/>", encoding="utf-8")
        (balanced_dir / "yolo26n_416.bin").write_bytes(b"balanced weights")
        (high_end_dir / "yolo11l.xml").write_text("<net/>", encoding="utf-8")
        (high_end_dir / "yolo11l.bin").write_bytes(b"high-end weights")
        (self.root / "models" / "coco80.txt").write_text(
            "\n".join(COCO80_LABELS) + "\n", encoding="utf-8"
        )
        high_end_onnx = self.root / "models" / "yolo11l_onnx"
        high_end_onnx.mkdir(parents=True)
        (high_end_onnx / "yolo11l.onnx").write_bytes(b"onnx weights")

        player_dir = self.root / "models" / "fort_player_openvino_model"
        player_balanced_dir = self.root / "models" / "fort_player_416_openvino_model"
        player_dir.mkdir(parents=True)
        player_balanced_dir.mkdir(parents=True)
        (player_dir / "fort_player.xml").write_text("<net/>", encoding="utf-8")
        (player_dir / "fort_player.bin").write_bytes(b"player weights")
        (player_balanced_dir / "fort_player_416.xml").write_text(
            "<net/>", encoding="utf-8"
        )
        (player_balanced_dir / "fort_player_416.bin").write_bytes(
            b"balanced player weights"
        )
        (self.root / "models" / "fort_player.txt").write_text(
            "\n".join(PLAYER_LABELS) + "\n", encoding="utf-8"
        )
        attribution = "\n".join(PLAYER_ATTRIBUTION_MARKERS) + "\n"
        (player_dir / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
        (player_balanced_dir / "ATTRIBUTION.md").write_text(
            attribution, encoding="utf-8"
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

        self.assertEqual(len(summaries), 5)
        # The two player detectors are end-to-end; the COCO pair in this fixture
        # uses the traditional layout, while the high-end YOLO11l bundle keeps
        # the dynamic end-to-end path exercised.
        player_summaries = [text for text in summaries if "player detector" in text]
        coco_summaries = [text for text in summaries if "COCO detector" in text]
        high_end_summaries = [text for text in summaries if "YOLO11l detector" in text]
        self.assertEqual(len(player_summaries), 2)
        self.assertEqual(len(coco_summaries), 2)
        self.assertEqual(len(high_end_summaries), 1)
        for text in player_summaries:
            self.assertIn("end-to-end [1,N,6]", text)
        for text in coco_summaries:
            self.assertIn("traditional [1,84,N]", text)
        self.assertIn("traditional [1,84,N]", high_end_summaries[0])
        self.assertEqual(len(core.calls), 5)
        for xml_path, bin_path in core.calls:
            self.assertEqual(xml_path.stem, bin_path.stem)
            self.assertTrue(xml_path.is_absolute())
            self.assertTrue(bin_path.is_absolute())

    def test_accepts_end_to_end_output_with_dynamic_detection_count(self) -> None:
        core = FakeCore(
            fast=FakeModel(output_shapes=((1, None, 6),)),
            balanced=FakeModel(
                input_shape=BALANCED_INPUT_SHAPE,
                output_shapes=((1, None, 6),),
            ),
            high_end=FakeModel(
                input_shape=(1, 3, 640, 640),
                output_shapes=((1, None, 6),),
            ),
        )

        summaries = validate_release_assets(self.root, core_factory=lambda: core)

        self.assertTrue(all("end-to-end [1,N,6]" in item for item in summaries))

    def test_rejects_empty_or_missing_ir_without_constructing_openvino(self) -> None:
        balanced_bin = (
            self.root
            / "models"
            / "yolo26n_416_openvino_model"
            / "yolo26n_416.bin"
        )
        balanced_bin.write_bytes(b"")
        constructed = False

        def factory() -> FakeCore:
            nonlocal constructed
            constructed = True
            return FakeCore()

        with self.assertRaisesRegex(ReleaseAssetError, "Balanced COCO detector BIN is empty"):
            validate_release_assets(self.root, core_factory=factory)
        self.assertFalse(constructed)

    def test_rejects_an_ir_pair_openvino_cannot_read(self) -> None:
        core = FakeCore(balanced=RuntimeError("weights do not match graph"))

        message = self.assert_fails("OpenVINO could not read the matching", core)

        self.assertIn("yolo26n_416.xml, yolo26n_416.bin", message)
        self.assertIn("weights do not match graph", message)

    def test_rejects_nonstatic_or_wrong_input_shape(self) -> None:
        for shape in ((1, 3, 640, 640), (None, 3, 320, 320), (1, 320, 320, 3)):
            with self.subTest(shape=shape):
                core = FakeCore(balanced=FakeModel(input_shape=shape))
                message = self.assert_fails("expected static NCHW", core)
                self.assertIn(str(BALANCED_INPUT_SHAPE), message)

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
                self.assert_fails(expected, FakeCore(balanced=model))

    def test_requires_exact_shared_coco_labels(self) -> None:
        coco_labels = self.root / "models" / "coco80.txt"
        changed = list(COCO80_LABELS)
        changed[0] = "human"
        coco_labels.write_text("\n".join(changed) + "\n", encoding="utf-8")
        self.assert_fails("COCO detector labels are wrong")

if __name__ == "__main__":
    unittest.main()
