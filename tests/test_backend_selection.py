from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import tkinter  # noqa: F401
except ImportError:
    stub = types.ModuleType("tkinter")
    stub.filedialog = types.SimpleNamespace()
    stub.messagebox = types.SimpleNamespace()
    stub.ttk = types.SimpleNamespace()
    sys.modules["tkinter"] = stub

from config import parse_args
from launcher.settings import (
    DEFAULT_MODEL_PRESET,
    LauncherSettings,
    MODEL_PRESETS,
    MODEL_PRESET_CUSTOM,
    MODEL_PRESET_FORT_PLAYER_BALANCED_INT8,
    SettingsError,
    model_preset,
    model_preset_paths,
    release_default_model_contract,
)
from utils.inference_size import normalize_inference_size


def make_bundled_tree(root: Path, backend: str) -> tuple[Path, Path]:
    with mock.patch("launcher.settings.resource_root", return_value=root):
        model_text, labels_text = model_preset_paths(DEFAULT_MODEL_PRESET, backend)
    model = Path(model_text)
    labels = Path(labels_text)
    model.parent.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("<net/>", encoding="utf-8")
    if model.suffix == ".xml":
        model.with_suffix(".bin").write_bytes(b"weights")
    labels.write_text("player\n", encoding="utf-8")
    return model, labels


class PresetFormatTests(unittest.TestCase):
    def test_release_default_contract_is_derived_from_launcher_default(self) -> None:
        contract = release_default_model_contract()
        preset = model_preset(DEFAULT_MODEL_PRESET)

        self.assertEqual(contract["preset"], DEFAULT_MODEL_PRESET)
        self.assertEqual(contract["model_path"], preset.model_for("onnxruntime"))
        self.assertEqual(contract["labels_path"], preset.labels_relative)
        self.assertEqual(
            contract["input_shape_hw"],
            list(normalize_inference_size(preset.inference_size)),
        )

    def test_every_portable_bundled_preset_offers_both_model_formats(self) -> None:
        for preset in MODEL_PRESETS:
            if not preset.bundled or preset.key == MODEL_PRESET_FORT_PLAYER_BALANCED_INT8:
                continue
            with self.subTest(preset=preset.key):
                self.assertIsNotNone(preset.model_for("openvino"), preset.key)
                self.assertIsNotNone(preset.model_for("onnxruntime"), preset.key)

    def test_openvino_resolves_xml_and_onnxruntime_resolves_onnx(self) -> None:
        preset = model_preset(DEFAULT_MODEL_PRESET)

        self.assertTrue(preset.model_for("openvino").endswith(".xml"))
        self.assertTrue(preset.model_for("onnxruntime").endswith(".onnx"))

    def test_labels_are_shared_across_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch("launcher.settings.resource_root", return_value=root):
                _, openvino_labels = model_preset_paths(DEFAULT_MODEL_PRESET, "openvino")
                _, onnx_labels = model_preset_paths(DEFAULT_MODEL_PRESET, "onnxruntime")

        self.assertEqual(openvino_labels, onnx_labels)

    def test_a_preset_without_an_onnx_form_is_refused_rather_than_guessed(self) -> None:
        with self.assertRaises(ValueError):
            model_preset_paths(MODEL_PRESET_FORT_PLAYER_BALANCED_INT8, "onnxruntime")


class DetectorArgumentTests(unittest.TestCase):
    def test_openvino_settings_build_an_xml_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, _ = make_bundled_tree(root, "openvino")
            with mock.patch("launcher.settings.resource_root", return_value=root):
                args = LauncherSettings(backend="openvino", device="CPU").detector_arguments()

        self.assertEqual(args[args.index("--backend") + 1], "openvino")
        self.assertEqual(args[args.index("--model") + 1], str(model.resolve()))

    def test_onnx_settings_build_an_onnx_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, _ = make_bundled_tree(root, "onnxruntime")
            with mock.patch("launcher.settings.resource_root", return_value=root):
                args = LauncherSettings(
                    backend="onnxruntime", device="ROCM"
                ).detector_arguments()

        self.assertEqual(args[args.index("--backend") + 1], "onnxruntime")
        self.assertEqual(args[args.index("--model") + 1], str(model.resolve()))
        self.assertEqual(args[args.index("--device") + 1], "ROCM")

    def test_onnx_backend_does_not_demand_a_bin_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, _ = make_bundled_tree(root, "onnxruntime")
            self.assertFalse(model.with_suffix(".bin").exists())
            with mock.patch("launcher.settings.resource_root", return_value=root):
                # Must not raise: an .onnx graph is self-contained.
                LauncherSettings(backend="onnxruntime").detector_arguments()

    def test_unknown_backend_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_bundled_tree(root, "openvino")
            with mock.patch("launcher.settings.resource_root", return_value=root):
                with self.assertRaises(SettingsError):
                    LauncherSettings(backend="tensorflow").detector_arguments()

    def test_absent_backend_in_an_old_profile_reads_as_openvino(self) -> None:
        loaded = LauncherSettings.from_mapping({"version": 4})

        self.assertEqual(loaded.backend, "openvino")


class CliBackendTests(unittest.TestCase):
    def _parse(self, extra: list[str]):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "m.xml"
            model.write_text("<net/>", encoding="utf-8")
            model.with_suffix(".bin").write_bytes(b"w")
            labels = root / "l.txt"
            labels.write_text("player\n", encoding="utf-8")
            video = root / "clip.mp4"
            video.write_bytes(b"\x00")
            return parse_args(
                [
                    "--source", str(video),
                    "--model", str(model),
                    "--labels", str(labels),
                    *extra,
                ]
            )

    def test_backend_defaults_to_openvino(self) -> None:
        self.assertEqual(self._parse([]).backend, "openvino")

    def test_backend_can_select_onnxruntime(self) -> None:
        config = self._parse(["--backend", "onnxruntime", "--device", "ROCM"])

        self.assertEqual(config.backend, "onnxruntime")
        self.assertEqual(config.device, "ROCM")

    def test_unknown_backend_is_rejected_by_the_parser(self) -> None:
        with self.assertRaises(SystemExit):
            self._parse(["--backend", "tensorflow"])


if __name__ == "__main__":
    unittest.main()
