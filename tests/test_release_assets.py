from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from scripts.validate_release_assets import (
    BALANCED_INPUT_SHAPE,
    COCO80_LABELS,
    EXPECTED_INPUT_SHAPE,
    PLAYER_ATTRIBUTION_MARKERS,
    PLAYER_LABELS,
    RELEASE_MODELS,
    ReleaseAssetError,
    ULTRALYTICS_METADATA_MARKERS,
    validate_release_assets,
)
from utils.release_model_contract import (
    canonical_json_bytes,
    make_release_default_contract,
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
        player_int8: FakeModel | Exception | None = None,
        onnx_failure: Exception | None = None,
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
            "fort_player_416_int8.xml": player_int8
            or FakeModel(
                input_shape=BALANCED_INPUT_SHAPE,
                output_shapes=((1, 300, 6),),
            ),
        }
        self.calls: list[tuple[Path, Path]] = []
        self.onnx_failure = onnx_failure

    def read_model(self, *, model: str, weights: str | None = None) -> FakeModel:
        model_path = Path(model)
        weights_path = Path(weights) if weights is not None else Path()
        self.calls.append((model_path, weights_path))
        if model_path.suffix == ".onnx" and self.onnx_failure is not None:
            raise self.onnx_failure
        aliases = {
            "yolo26n.onnx": "yolo26n.xml",
            "yolo26n_416.onnx": "yolo26n_416.xml",
            "yolo11l.onnx": "yolo11l.xml",
            "fort_player.onnx": "fort_player.xml",
            "fort_player_416.onnx": "fort_player_416.xml",
        }
        lookup = aliases.get(model_path.name, model_path.name)
        if model_path.suffix == ".onnx" and lookup == model_path.name:
            lookup = model_path.with_suffix(".xml").name
        outcome = self.models[lookup]
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
        for directory, size in (
            (coco_dir, 320),
            (balanced_dir, 416),
            (high_end_dir, 640),
        ):
            directory.joinpath("metadata.yaml").write_text(
                "\n".join(ULTRALYTICS_METADATA_MARKERS)
                + f"\nimgsz:\n- {size}\n- {size}\n",
                encoding="utf-8",
            )
        (self.root / "models" / "coco80.txt").write_text(
            "\n".join(COCO80_LABELS) + "\n", encoding="utf-8"
        )
        onnx_files = (
            ("yolo11l_onnx", "yolo11l.onnx"),
            ("yolo26n_onnx", "yolo26n.onnx"),
            ("yolo26n_416_onnx", "yolo26n_416.onnx"),
            ("fort_player_onnx", "fort_player.onnx"),
            ("fort_player_416_onnx", "fort_player_416.onnx"),
        )
        for directory, filename in onnx_files:
            target = self.root / "models" / directory
            target.mkdir(parents=True)
            (target / filename).write_bytes(b"onnx weights")

        player_dir = self.root / "models" / "fort_player_openvino_model"
        player_balanced_dir = self.root / "models" / "fort_player_416_openvino_model"
        player_int8_dir = self.root / "models" / "fort_player_416_int8_openvino_model"
        player_dir.mkdir(parents=True)
        player_balanced_dir.mkdir(parents=True)
        player_int8_dir.mkdir(parents=True)
        (player_dir / "fort_player.xml").write_text("<net/>", encoding="utf-8")
        (player_dir / "fort_player.bin").write_bytes(b"player weights")
        (player_balanced_dir / "fort_player_416.xml").write_text(
            "<net/>", encoding="utf-8"
        )
        (player_balanced_dir / "fort_player_416.bin").write_bytes(
            b"balanced player weights"
        )
        (player_int8_dir / "fort_player_416_int8.xml").write_text(
            "<net/>", encoding="utf-8"
        )
        (player_int8_dir / "fort_player_416_int8.bin").write_bytes(
            b"int8 player weights"
        )
        (self.root / "models" / "fort_player.txt").write_text(
            "\n".join(PLAYER_LABELS) + "\n", encoding="utf-8"
        )
        attribution = "\n".join(PLAYER_ATTRIBUTION_MARKERS) + "\n"
        (player_dir / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
        (player_balanced_dir / "ATTRIBUTION.md").write_text(
            attribution, encoding="utf-8"
        )
        (player_int8_dir / "ATTRIBUTION.md").write_text(
            attribution, encoding="utf-8"
        )
        (player_int8_dir / "metadata.yaml").write_text(
            "precision: INT8\nmethod: NNCF\noutput_xml_sha256: fixture\n",
            encoding="utf-8",
        )
        self._write_default_contract()
        self._write_model_manifest()

    def _file_record(self, relative: str) -> dict[str, object]:
        path = self.root / relative
        return {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _write_default_contract(
        self,
        *,
        shape: tuple[int, int] = (416, 416),
        onnx: str = "models/fort_player_416_onnx/fort_player_416.onnx",
        xml: str = "models/fort_player_416_openvino_model/fort_player_416.xml",
        binary: str = "models/fort_player_416_openvino_model/fort_player_416.bin",
        labels: str = "models/fort_player.txt",
        attribution: str = "models/fort_player_416_openvino_model/ATTRIBUTION.md",
        detail_crop_size_source_pixels: int = 0,
    ) -> None:
        contract = make_release_default_contract(
            label=f"Game players — Selected {shape[0]}×{shape[1]} (Recommended)",
            description="Fixture release-default player detector.",
            input_shape_nchw=[1, 3, *shape],
            detail_crop_size_source_pixels=detail_crop_size_source_pixels,
            artifacts={
                "onnx": self._file_record(onnx),
                "openvino_xml": self._file_record(xml),
                "openvino_bin": self._file_record(binary),
                "labels": self._file_record(labels),
                "attribution": self._file_record(attribution),
            },
            provenance={
                "kind": "existing_release_default_migration",
                "candidate_content_sha256": None,
                "candidate_manifest_sha256": None,
                "tournament_selection_sha256": None,
            },
        )
        (self.root / "models" / "RELEASE-DEFAULT.json").write_bytes(
            canonical_json_bytes(contract)
        )

    def _write_model_manifest(self) -> None:
        manifest_paths: set[Path] = set()
        for asset in RELEASE_MODELS:
            manifest_paths.update(
                (asset.xml_relative, asset.bin_relative, asset.labels_relative)
            )
            if asset.onnx_relative is not None:
                manifest_paths.add(asset.onnx_relative)
            if asset.attribution_relative is not None:
                manifest_paths.add(asset.attribution_relative)
            if asset.metadata_relative is not None:
                manifest_paths.add(asset.metadata_relative)
        contract = json.loads(
            (self.root / "models" / "RELEASE-DEFAULT.json").read_text(
                encoding="utf-8"
            )
        )
        manifest_paths.update(
            Path(record["path"]) for record in contract["artifacts"].values()
        )
        lines = []
        for relative in sorted(manifest_paths, key=lambda value: value.as_posix()):
            digest = hashlib.sha256((self.root / relative).read_bytes()).hexdigest()
            lines.append(f"{digest}  {relative.as_posix()}")
        (self.root / "models" / "RELEASE-MANIFEST.sha256").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
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

        self.assertEqual(len(summaries), 6)
        # The two player detectors are end-to-end; the COCO pair in this fixture
        # uses the traditional layout, while the high-end YOLO11l bundle keeps
        # the dynamic end-to-end path exercised.
        player_summaries = [
            text
            for text in summaries
            if "player detector" in text or text.startswith("Game players — Selected")
        ]
        coco_summaries = [text for text in summaries if "COCO detector" in text]
        high_end_summaries = [text for text in summaries if "YOLO11l detector" in text]
        self.assertEqual(len(player_summaries), 3)
        self.assertEqual(len(coco_summaries), 2)
        self.assertEqual(len(high_end_summaries), 1)
        for text in player_summaries:
            self.assertIn("end-to-end [1,N,6]", text)
        for text in coco_summaries:
            self.assertIn("traditional [1,84,N]", text)
        self.assertIn("traditional [1,84,N]", high_end_summaries[0])
        self.assertEqual(len(core.calls), 11)
        for model_path, weights_path in core.calls:
            self.assertTrue(model_path.is_absolute())
            if model_path.suffix == ".xml":
                self.assertEqual(model_path.stem, weights_path.stem)
                self.assertTrue(weights_path.is_absolute())

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

    def test_pointer_selects_a_new_rectangular_model_without_validator_edits(self) -> None:
        release = self.root / "models" / "release-defaults" / ("a" * 64)
        release.mkdir(parents=True)
        (release / "candidate.xml").write_text("<net/>", encoding="utf-8")
        (release / "candidate.bin").write_bytes(b"rectangular weights")
        (release / "candidate.onnx").write_bytes(b"rectangular onnx")
        (release / "labels.txt").write_text("player\n", encoding="utf-8")
        (release / "ATTRIBUTION.md").write_text(
            "FORT-Cuh\ncreativecommons.org/licenses/by/4.0\nAGPL-3.0\n",
            encoding="utf-8",
        )
        prefix = f"models/release-defaults/{'a' * 64}"
        self._write_default_contract(
            shape=(384, 640),
            onnx=f"{prefix}/candidate.onnx",
            xml=f"{prefix}/candidate.xml",
            binary=f"{prefix}/candidate.bin",
            labels=f"{prefix}/labels.txt",
            attribution=f"{prefix}/ATTRIBUTION.md",
            detail_crop_size_source_pixels=768,
        )
        self._write_model_manifest()
        core = FakeCore()
        core.models["candidate.xml"] = FakeModel(
            input_shape=(1, 3, 384, 640), output_shapes=((1, 300, 6),)
        )

        summaries = validate_release_assets(self.root, core_factory=lambda: core)

        selected = [item for item in summaries if "Selected 384×640" in item]
        self.assertEqual(len(selected), 1)
        self.assertIn("input (1, 3, 384, 640)", selected[0])
        self.assertIn("detail ROI requested width 768 source px", selected[0])

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

    def test_rejects_a_nonempty_onnx_graph_openvino_cannot_read(self) -> None:
        message = self.assert_fails(
            "OpenVINO could not read Game players — Selected 416×416 (Recommended) ONNX",
            FakeCore(onnx_failure=RuntimeError("corrupt protobuf")),
        )
        self.assertIn("corrupt protobuf", message)

    def test_rejects_a_missing_bundled_onnx_graph_without_constructing_openvino(self) -> None:
        path = self.root / "models" / "fort_player_onnx" / "fort_player.onnx"
        path.unlink()
        constructed = False

        def factory() -> FakeCore:
            nonlocal constructed
            constructed = True
            return FakeCore()

        with self.assertRaisesRegex(ReleaseAssetError, "Fast player detector ONNX is missing"):
            validate_release_assets(self.root, core_factory=factory)
        self.assertFalse(constructed)

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

    def test_requires_generic_yolo_export_metadata_and_provenance(self) -> None:
        metadata = (
            self.root / "models" / "yolo26n_openvino_model" / "metadata.yaml"
        )
        valid_metadata = metadata.read_bytes()
        metadata.unlink()
        self.assert_fails("COCO detector metadata is missing")

        metadata.write_text("task: detect\n", encoding="utf-8")
        self.assert_fails("missing required marker(s)")

        metadata.write_bytes(valid_metadata)

    def test_rejects_local_paths_in_bundled_model_metadata(self) -> None:
        metadata = (
            self.root / "models" / "yolo26n_openvino_model" / "metadata.yaml"
        )
        metadata.write_text(
            "\n".join(ULTRALYTICS_METADATA_MARKERS)
            + "\ndescription: trained on /home/private/dataset.yaml\n",
            encoding="utf-8",
        )
        self.assert_fails("contains a local or nonportable path", FakeCore())

    def test_rejects_a_missing_or_tampered_hash_manifest(self) -> None:
        manifest = self.root / "models" / "RELEASE-MANIFEST.sha256"
        valid_manifest = manifest.read_bytes()
        manifest.unlink()
        self.assert_fails("release model SHA-256 manifest is missing")

        manifest.write_bytes(valid_manifest)
        model = self.root / "models" / "fort_player_openvino_model" / "fort_player.bin"
        model.write_bytes(b"tampered weights")
        self.assert_fails("SHA-256 mismatch for models/fort_player_openvino_model/fort_player.bin")

    def test_rejects_a_missing_or_false_like_release_default_pointer(self) -> None:
        pointer = self.root / "models" / "RELEASE-DEFAULT.json"
        valid_pointer = pointer.read_bytes()
        pointer.unlink()
        self.assert_fails("release-default model contract is invalid")

        pointer.write_bytes(valid_pointer)
        value = json.loads(pointer.read_text(encoding="utf-8"))
        value["qualification"] = {
            key: 0 for key in value["qualification"]
        }
        from utils.release_model_contract import contract_content_hash

        value["content_sha256"] = contract_content_hash(value)
        pointer.write_bytes(canonical_json_bytes(value))
        self.assert_fails("qualification")

        pointer.write_bytes(valid_pointer)
        value = json.loads(pointer.read_text(encoding="utf-8"))
        value["detail_crop_size_source_pixels"] = -1
        value["content_sha256"] = contract_content_hash(value)
        pointer.write_bytes(canonical_json_bytes(value))
        self.assert_fails("detail crop")

if __name__ == "__main__":
    unittest.main()
