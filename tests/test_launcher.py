from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from launcher.process import external_process_environment
from launcher.settings import (
    BUNDLED_LABELS,
    BUNDLED_MODEL,
    DEFAULT_MODEL_PRESET,
    LauncherSettings,
    MODEL_PRESET_CUSTOM,
    MODEL_PRESET_FORT_PLAYER,
    SETTINGS_VERSION,
    SELF_POSITION_CENTER,
    SELF_POSITION_LEFT,
    SELF_POSITION_RIGHT,
    SettingsError,
    launcher_command,
    load_settings,
    save_settings,
)


class LauncherSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "detector.xml"
        self.model.write_text("<xml />", encoding="utf-8")
        self.model.with_suffix(".bin").write_bytes(b"weights")
        self.labels = self.root / "labels.txt"
        self.labels.write_text("target\n", encoding="utf-8")
        self.video = self.root / "game clip.mp4"
        self.video.write_bytes(b"video")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def settings(self, **changes: object) -> LauncherSettings:
        values: dict[str, object] = {
            "model_path": str(self.model),
            "labels_path": str(self.labels),
        }
        values.update(changes)
        return LauncherSettings(**values)  # type: ignore[arg-type]

    def test_screen_monitor_arguments(self) -> None:
        args = self.settings(source_mode="screen", screen_monitor="2").detector_arguments()
        self.assertEqual(args[args.index("--source") + 1], "screen")
        self.assertEqual(args[args.index("--screen-monitor") + 1], "2")
        self.assertNotIn("--screen-region", args)

    def test_third_person_filter_uses_left_preset_when_enabled(self) -> None:
        settings = self.settings(ignore_self=True)
        args = settings.detector_arguments()
        self.assertTrue(settings.ignore_self)
        self.assertEqual(settings.self_position, SELF_POSITION_LEFT)
        self.assertIn("--ignore-self", args)
        self.assertEqual(args[args.index("--self-zone-left") + 1], "0.18")
        self.assertEqual(args[args.index("--self-zone-width") + 1], "0.34")
        self.assertEqual(args[args.index("--self-zone-height") + 1], "0.10")

    def test_third_person_position_presets_map_to_cli_geometry(self) -> None:
        expected = {
            SELF_POSITION_LEFT: "0.18",
            SELF_POSITION_CENTER: "0.33",
            SELF_POSITION_RIGHT: "0.48",
        }
        for position, left in expected.items():
            with self.subTest(position=position):
                args = self.settings(
                    ignore_self=True,
                    self_position=position,
                ).detector_arguments()
                self.assertEqual(args[args.index("--self-zone-left") + 1], left)
                self.assertEqual(args[args.index("--self-zone-width") + 1], "0.34")
                self.assertEqual(args[args.index("--self-zone-height") + 1], "0.10")

    def test_disabled_third_person_filter_emits_no_self_arguments(self) -> None:
        args = self.settings(ignore_self=False).detector_arguments()
        self.assertNotIn("--ignore-self", args)
        self.assertNotIn("--self-zone-left", args)

    def test_invalid_enabled_self_position_is_rejected(self) -> None:
        with self.assertRaisesRegex(SettingsError, "on-screen character position"):
            self.settings(
                ignore_self=True,
                self_position="somewhere",
            ).detector_arguments()

    def test_screen_region_accepts_negative_desktop_coordinates(self) -> None:
        args = self.settings(
            source_mode="screen",
            use_screen_region=True,
            screen_x="-1920",
            screen_y="0",
            screen_width="1920",
            screen_height="1080",
        ).detector_arguments()
        self.assertEqual(args[args.index("--screen-region") + 1], "-1920,0,1920,1080")
        self.assertNotIn("--screen-monitor", args)

    def test_camera_arguments_and_no_preview(self) -> None:
        args = self.settings(
            source_mode="camera",
            camera_index="3",
            capture_width="1920",
            capture_height="1080",
            capture_fps="59.94",
            preview=False,
        ).detector_arguments()
        self.assertEqual(args[args.index("--source") + 1], "3")
        self.assertEqual(args[args.index("--capture-size") + 1], "1920x1080")
        self.assertEqual(args[args.index("--capture-fps") + 1], "59.94")
        self.assertIn("--no-preview", args)
        self.assertNotIn("--no-draw", args)

    def test_video_path_is_one_argument_even_with_spaces(self) -> None:
        settings = self.settings(source_mode="video", video_path=str(self.video))
        command = launcher_command(
            settings,
            executable="/runtime/python",
            app_script="/checkout/app.py",
            frozen=False,
        )
        self.assertEqual(command[:3], ["/runtime/python", "/checkout/app.py", "--cli"])
        self.assertEqual(command[command.index("--source") + 1], str(self.video.resolve()))

    def test_frozen_command_reuses_executable(self) -> None:
        settings = self.settings(source_mode="video", video_path=str(self.video))
        command = launcher_command(settings, executable="GameDetector.exe", frozen=True)
        self.assertEqual(command[:2], ["GameDetector.exe", "--cli"])
        self.assertNotIn("app.py", command)

    def test_frozen_windows_command_uses_console_helper(self) -> None:
        gui_executable = self.root / "GameDetector.exe"
        gui_executable.write_bytes(b"gui")
        helper_executable = self.root / "GameDetectorCLI.exe"
        helper_executable.write_bytes(b"cli")
        settings = self.settings(source_mode="video", video_path=str(self.video))
        with mock.patch("launcher.settings.sys.platform", "win32"):
            command = launcher_command(
                settings,
                executable=gui_executable,
                frozen=True,
            )
        self.assertEqual(command[:2], [str(helper_executable), "--cli"])

    def test_model_requires_matching_bin_file(self) -> None:
        self.model.with_suffix(".bin").unlink()
        with self.assertRaisesRegex(SettingsError, "weights were not found"):
            self.settings().detector_arguments()

    def test_capture_size_must_be_complete(self) -> None:
        with self.assertRaisesRegex(SettingsError, "both capture width and height"):
            self.settings(
                source_mode="camera", capture_width="1280", capture_height=""
            ).detector_arguments()

    def test_numeric_ranges_are_validated(self) -> None:
        with self.assertRaisesRegex(SettingsError, "Confidence must be between"):
            self.settings(confidence="1.5").detector_arguments()

    def test_settings_round_trip(self) -> None:
        target = self.root / "preferences" / "settings.json"
        original = self.settings(
            source_mode="video",
            video_path=str(self.video),
            draw=False,
            ignore_self=True,
            self_position=SELF_POSITION_RIGHT,
        )
        self.assertEqual(save_settings(original, target), target)
        loaded = load_settings(target)
        self.assertEqual(loaded.source_mode, "video")
        self.assertEqual(loaded.video_path, str(self.video))
        self.assertFalse(loaded.draw)
        self.assertTrue(loaded.ignore_self)
        self.assertEqual(loaded.self_position, SELF_POSITION_RIGHT)

    def test_older_settings_do_not_silently_enable_filtering(self) -> None:
        target = self.root / "old-settings.json"
        target.write_text(
            json.dumps(
                {
                    "version": 1,
                    "model_path": str(self.model),
                    "labels_path": str(self.labels),
                    "source_mode": "screen",
                }
            ),
            encoding="utf-8",
        )
        loaded = load_settings(target)
        self.assertFalse(loaded.ignore_self)
        self.assertEqual(loaded.self_position, SELF_POSITION_LEFT)
        self.assertEqual(loaded.model_preset, MODEL_PRESET_CUSTOM)

    def test_bundled_paths_are_saved_semantically(self) -> None:
        target = self.root / "settings.json"
        original = LauncherSettings()
        save_settings(original, target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], SETTINGS_VERSION)
        self.assertEqual(payload["model_preset"], DEFAULT_MODEL_PRESET)
        self.assertNotIn("model_path", payload)
        self.assertNotIn("labels_path", payload)
        restored = load_settings(target)
        self.assertEqual(restored.model_preset, MODEL_PRESET_FORT_PLAYER)
        self.assertEqual(restored.model_path, original.model_path)
        self.assertEqual(restored.labels_path, original.labels_path)

    def test_bad_json_falls_back_to_defaults(self) -> None:
        target = self.root / "settings.json"
        target.write_text("not json", encoding="utf-8")
        self.assertEqual(load_settings(target).source_mode, "screen")


class ExternalProcessTests(unittest.TestCase):
    def test_frozen_linux_restores_original_library_path(self) -> None:
        environment = {
            "LD_LIBRARY_PATH": "/bundle/lib",
            "LD_LIBRARY_PATH_ORIG": "/system/lib",
            "DISPLAY": ":0",
        }
        result = external_process_environment(
            environment, frozen=True, platform="linux"
        )
        self.assertEqual(result["LD_LIBRARY_PATH"], "/system/lib")
        self.assertEqual(result["DISPLAY"], ":0")

    def test_frozen_linux_removes_injected_path_when_no_original_exists(self) -> None:
        result = external_process_environment(
            {"LD_LIBRARY_PATH": "/bundle/lib"}, frozen=True, platform="linux"
        )
        self.assertNotIn("LD_LIBRARY_PATH", result)

    def test_source_environment_is_unchanged(self) -> None:
        environment = {"LD_LIBRARY_PATH": "/developer/lib"}
        result = external_process_environment(
            environment, frozen=False, platform="linux"
        )
        self.assertEqual(result, environment)


if __name__ == "__main__":
    unittest.main()
