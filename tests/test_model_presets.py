from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

# The production bundle includes Tk.  The CPU-only test venv may not have its
# shared Tk library, and these tests exercise only preset state methods (no
# widgets or display), so provide the smallest import-time stand-in there.
try:
    import tkinter  # noqa: F401
except ImportError:
    tkinter_stub = types.ModuleType("tkinter")
    tkinter_stub.filedialog = types.SimpleNamespace()
    tkinter_stub.messagebox = types.SimpleNamespace()
    tkinter_stub.ttk = types.SimpleNamespace()
    sys.modules["tkinter"] = tkinter_stub

from launcher.application import (
    DetectorLauncher,
    MODEL_PRESET_LABELS,
)
from launcher.settings import (
    BUNDLED_LABELS,
    BUNDLED_MODEL,
    DEFAULT_MODEL_PRESET,
    LauncherSettings,
    MODEL_PRESETS,
    MODEL_PRESET_COCO,
    MODEL_PRESET_CUSTOM,
    MODEL_PRESET_FORT_PLAYER,
    SETTINGS_VERSION,
    load_settings,
    model_preset,
    model_preset_paths,
    save_settings,
)


class FakeVariable:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class FakeWidget:
    def __init__(self) -> None:
        self.state = ""

    def configure(self, *, state: str) -> None:
        self.state = state


def create_bundled_files(root: Path, preset_key: str) -> tuple[Path, Path]:
    with mock.patch("launcher.settings.resource_root", return_value=root):
        model_text, labels_text = model_preset_paths(preset_key)
    model = Path(model_text)
    labels = Path(labels_text)
    model.parent.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("<xml />", encoding="utf-8")
    model.with_suffix(".bin").write_bytes(b"weights")
    labels.write_text("player\n", encoding="utf-8")
    return model, labels


class ModelPresetSettingsTests(unittest.TestCase):
    def test_preset_catalog_has_stable_keys_labels_and_player_description(self) -> None:
        self.assertEqual(
            [preset.key for preset in MODEL_PRESETS],
            [MODEL_PRESET_FORT_PLAYER, MODEL_PRESET_COCO, MODEL_PRESET_CUSTOM],
        )
        self.assertEqual(
            model_preset(MODEL_PRESET_FORT_PLAYER).label,
            "Fortnite-style players (Recommended)",
        )
        self.assertEqual(
            model_preset(MODEL_PRESET_COCO).label,
            "General objects (COCO fallback)",
        )
        description = model_preset(MODEL_PRESET_FORT_PLAYER).description.lower()
        self.assertIn("one class: player", description)
        self.assertIn("auto", description)
        self.assertIn("self filter", description)

    def test_fresh_settings_choose_recommended_pair_from_resource_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch("launcher.settings.resource_root", return_value=root):
                settings = LauncherSettings()
                expected = model_preset_paths(DEFAULT_MODEL_PRESET)
        self.assertEqual(settings.model_preset, MODEL_PRESET_FORT_PLAYER)
        self.assertEqual((settings.model_path, settings.labels_path), expected)

    def test_bundled_pair_is_resolved_atomically_when_arguments_are_built(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, labels = create_bundled_files(root, MODEL_PRESET_FORT_PLAYER)
            with mock.patch("launcher.settings.resource_root", return_value=root):
                settings = LauncherSettings(model_preset=MODEL_PRESET_FORT_PLAYER)
                # Simulate stale or manually mixed UI state.  The semantic key
                # remains authoritative when a detector command is created.
                settings.model_path = "/stale/coco.xml"
                settings.labels_path = "/stale/custom.txt"
                args = settings.detector_arguments()
        self.assertEqual(args[args.index("--model") + 1], str(model.resolve()))
        self.assertEqual(args[args.index("--labels") + 1], str(labels.resolve()))
        self.assertEqual(args[args.index("--output-format") + 1], "auto")

    def test_custom_files_remain_custom_and_are_used_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "mine.xml"
            labels = root / "mine.txt"
            model.write_text("<xml />", encoding="utf-8")
            model.with_suffix(".bin").write_bytes(b"weights")
            labels.write_text("target\n", encoding="utf-8")
            settings = LauncherSettings(
                model_path=str(model),
                labels_path=str(labels),
            )
            args = settings.detector_arguments()
        self.assertEqual(settings.model_preset, MODEL_PRESET_CUSTOM)
        self.assertEqual(args[args.index("--model") + 1], str(model.resolve()))
        self.assertEqual(args[args.index("--labels") + 1], str(labels.resolve()))

    def test_v4_bundled_round_trip_is_semantic_and_relocatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            first_root = temporary_root / "first bundle"
            second_root = temporary_root / "second bundle"
            target = temporary_root / "settings.json"
            with mock.patch("launcher.settings.resource_root", return_value=first_root):
                save_settings(
                    LauncherSettings(model_preset=MODEL_PRESET_FORT_PLAYER),
                    target,
                )
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], SETTINGS_VERSION)
            self.assertEqual(payload["model_preset"], MODEL_PRESET_FORT_PLAYER)
            self.assertNotIn("model_path", payload)
            self.assertNotIn("labels_path", payload)

            with mock.patch("launcher.settings.resource_root", return_value=second_root):
                loaded = load_settings(target)
                expected = model_preset_paths(MODEL_PRESET_FORT_PLAYER)
        self.assertEqual((loaded.model_path, loaded.labels_path), expected)

    def test_each_legacy_schema_keeps_old_bundled_profiles_on_coco(self) -> None:
        cases = (
            {},
            {"model_path": BUNDLED_MODEL, "labels_path": BUNDLED_LABELS},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for version in (1, 2, 3):
                for old_model_fields in cases:
                    with self.subTest(version=version, fields=old_model_fields):
                        payload = {"version": version, **old_model_fields}
                        with mock.patch("launcher.settings.resource_root", return_value=root):
                            loaded = LauncherSettings.from_mapping(payload)
                            expected = model_preset_paths(MODEL_PRESET_COCO)
                        self.assertEqual(loaded.model_preset, MODEL_PRESET_COCO)
                        self.assertEqual(
                            (loaded.model_path, loaded.labels_path),
                            expected,
                        )

    def test_explicit_legacy_paths_migrate_to_custom(self) -> None:
        loaded = LauncherSettings.from_mapping(
            {
                "version": 3,
                "model_path": "/models/old.xml",
                "labels_path": "/models/old.txt",
            }
        )
        self.assertEqual(loaded.model_preset, MODEL_PRESET_CUSTOM)
        self.assertEqual(loaded.model_path, "/models/old.xml")
        self.assertEqual(loaded.labels_path, "/models/old.txt")

    def test_known_v4_preset_ignores_stale_serialized_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch("launcher.settings.resource_root", return_value=root):
                loaded = LauncherSettings.from_mapping(
                    {
                        "version": 4,
                        "model_preset": MODEL_PRESET_COCO,
                        "model_path": "/mixed/player.xml",
                        "labels_path": "/mixed/player.txt",
                    }
                )
                expected = model_preset_paths(MODEL_PRESET_COCO)
        self.assertEqual(loaded.model_preset, MODEL_PRESET_COCO)
        self.assertEqual((loaded.model_path, loaded.labels_path), expected)

    def test_v4_custom_paths_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "settings.json"
            original = LauncherSettings(
                model_preset=MODEL_PRESET_CUSTOM,
                model_path="/models/custom.xml",
                labels_path="/models/custom.txt",
            )
            save_settings(original, target)
            payload = json.loads(target.read_text(encoding="utf-8"))
            loaded = load_settings(target)
        self.assertEqual(payload["model_path"], "/models/custom.xml")
        self.assertEqual(payload["labels_path"], "/models/custom.txt")
        self.assertEqual(loaded.model_preset, MODEL_PRESET_CUSTOM)
        self.assertEqual(loaded.model_path, original.model_path)
        self.assertEqual(loaded.labels_path, original.labels_path)


class ModelPresetUiStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.launcher = DetectorLauncher.__new__(DetectorLauncher)
        self.launcher.model_preset = FakeVariable(
            MODEL_PRESET_LABELS[MODEL_PRESET_CUSTOM]
        )
        self.launcher.model_preset_description = FakeVariable("")
        self.launcher.model_path = FakeVariable("/custom/player.xml")
        self.launcher.labels_path = FakeVariable("/custom/player.txt")
        self.launcher.output_format = FakeVariable("traditional")
        self.launcher._active_model_preset = MODEL_PRESET_CUSTOM
        self.launcher._custom_model_path = "/custom/player.xml"
        self.launcher._custom_labels_path = "/custom/player.txt"
        self.launcher._custom_output_format = "traditional"
        self.launcher.model_path_entry = FakeWidget()
        self.launcher.labels_path_entry = FakeWidget()
        self.launcher.model_browse_button = FakeWidget()
        self.launcher.labels_browse_button = FakeWidget()

    def test_bundled_presets_lock_paths_and_custom_restores_cached_pair(self) -> None:
        fort_pair = ("/bundle/fort.xml", "/bundle/fort.txt")
        coco_pair = ("/bundle/coco.xml", "/bundle/coco.txt")

        def resolve(key: str) -> tuple[str, str]:
            return fort_pair if key == MODEL_PRESET_FORT_PLAYER else coco_pair

        self.launcher.model_preset.set(
            MODEL_PRESET_LABELS[MODEL_PRESET_FORT_PLAYER]
        )
        with mock.patch("launcher.application.model_preset_paths", side_effect=resolve):
            self.launcher._model_preset_changed()
            self.assertEqual(
                (
                    self.launcher.model_path.get(),
                    self.launcher.labels_path.get(),
                ),
                fort_pair,
            )
            self.assertEqual(self.launcher.model_path_entry.state, "readonly")
            self.assertEqual(self.launcher.labels_path_entry.state, "readonly")
            self.assertEqual(self.launcher.model_browse_button.state, "disabled")
            self.assertEqual(self.launcher.labels_browse_button.state, "disabled")
            self.assertEqual(self.launcher.output_format.get(), "auto")

            # Even adversarial stale UI state cannot carry a custom decoder
            # from one bundled preset into the other.
            self.launcher.output_format.set("end2end")
            self.launcher.model_preset.set(MODEL_PRESET_LABELS[MODEL_PRESET_COCO])
            self.launcher._model_preset_changed()
            self.assertEqual(
                (
                    self.launcher.model_path.get(),
                    self.launcher.labels_path.get(),
                ),
                coco_pair,
            )
            self.assertEqual(self.launcher.output_format.get(), "auto")

            self.launcher.model_preset.set(MODEL_PRESET_LABELS[MODEL_PRESET_CUSTOM])
            self.launcher._model_preset_changed()

        self.assertEqual(
            (
                self.launcher.model_path.get(),
                self.launcher.labels_path.get(),
            ),
            ("/custom/player.xml", "/custom/player.txt"),
        )
        self.assertEqual(self.launcher.model_path_entry.state, "normal")
        self.assertEqual(self.launcher.labels_path_entry.state, "normal")
        self.assertEqual(self.launcher.model_browse_button.state, "normal")
        self.assertEqual(self.launcher.labels_browse_button.state, "normal")
        self.assertEqual(self.launcher.output_format.get(), "traditional")

    def test_latest_custom_edits_are_cached_when_switching_away(self) -> None:
        self.launcher.model_path.set("/custom/new.xml")
        self.launcher.labels_path.set("/custom/new.txt")
        self.launcher.model_preset.set(
            MODEL_PRESET_LABELS[MODEL_PRESET_FORT_PLAYER]
        )
        with mock.patch(
            "launcher.application.model_preset_paths",
            return_value=("/bundle/fort.xml", "/bundle/fort.txt"),
        ):
            self.launcher._model_preset_changed()
            self.launcher.model_preset.set(MODEL_PRESET_LABELS[MODEL_PRESET_CUSTOM])
            self.launcher._model_preset_changed()
        self.assertEqual(self.launcher.model_path.get(), "/custom/new.xml")
        self.assertEqual(self.launcher.labels_path.get(), "/custom/new.txt")

    def test_bundled_settings_override_incompatible_serialized_decoder(self) -> None:
        settings = LauncherSettings(
            model_preset=MODEL_PRESET_FORT_PLAYER,
            output_format="traditional",
        )
        self.assertEqual(settings.output_format, "auto")


if __name__ == "__main__":
    unittest.main()
