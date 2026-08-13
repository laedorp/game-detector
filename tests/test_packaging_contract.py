from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "packaging" / "game_detector.spec"
LINUX_BUILD = PROJECT_ROOT / "scripts" / "build_linux_app.sh"
LINUX_BUNDLE_VALIDATOR = PROJECT_ROOT / "scripts" / "validate_linux_bundle.py"
MAKCU_ACCESS_INSTALLER = PROJECT_ROOT / "scripts" / "install_makcu_access.sh"
WINDOWS_BUILD = PROJECT_ROOT / "scripts" / "build_windows_app.ps1"
WINDOWS_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "build-windows.yml"
WINDOWS_GUIDE = PROJECT_ROOT / "packaging" / "windows" / "README-Windows.txt"
RELEASE_PREFLIGHT = PROJECT_ROOT / "scripts" / "validate_release_assets.py"
README = PROJECT_ROOT / "README.md"
MAKCU_RULE = PROJECT_ROOT / "packaging" / "linux" / "70-game-detector-makcu.rules"

REQUIRED_MODEL_ASSETS = (
    "models/coco80.txt",
    "models/fort_player.txt",
    "models/fort_player_openvino_model/fort_player.xml",
    "models/fort_player_openvino_model/fort_player.bin",
    "models/fort_player_openvino_model/ATTRIBUTION.md",
    "models/fort_player_onnx/fort_player.onnx",
    "models/fort_player_onnx/ATTRIBUTION.md",
    "models/fort_player_416_openvino_model/fort_player_416.xml",
    "models/fort_player_416_openvino_model/fort_player_416.bin",
    "models/fort_player_416_openvino_model/ATTRIBUTION.md",
    "models/fort_player_416_onnx/fort_player_416.onnx",
    "models/fort_player_416_onnx/ATTRIBUTION.md",
    "models/fort_player_416_int8_openvino_model/fort_player_416_int8.xml",
    "models/fort_player_416_int8_openvino_model/fort_player_416_int8.bin",
    "models/fort_player_416_int8_openvino_model/metadata.yaml",
    "models/fort_player_416_int8_openvino_model/ATTRIBUTION.md",
    "models/RELEASE-MANIFEST.sha256",
    "models/yolo26n_openvino_model/yolo26n.xml",
    "models/yolo26n_openvino_model/yolo26n.bin",
    "models/yolo26n_openvino_model/metadata.yaml",
    "models/yolo26n_onnx/yolo26n.onnx",
    "models/yolo26n_416_openvino_model/yolo26n_416.xml",
    "models/yolo26n_416_openvino_model/yolo26n_416.bin",
    "models/yolo26n_416_openvino_model/metadata.yaml",
    "models/yolo26n_416_onnx/yolo26n_416.onnx",
    "models/yolo11l_openvino_model/yolo11l.xml",
    "models/yolo11l_openvino_model/yolo11l.bin",
    "models/yolo11l_openvino_model/metadata.yaml",
    "models/yolo11l_onnx/yolo11l.onnx",
)
REQUIRED_ROOT_ASSETS = (
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/MODEL_BENCHMARKS.md",
    "docs/RELEASE_CHECKLIST.md",
)


class PackagingContractTests(unittest.TestCase):
    def test_spec_is_valid_python_and_mentions_every_required_model_asset(self) -> None:
        source = SPEC_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(SPEC_PATH))
        string_literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        for asset in REQUIRED_MODEL_ASSETS:
            with self.subTest(asset=asset):
                relative = Path(asset)
                directory = relative.parent.as_posix()
                self.assertIn(relative.name, string_literals)
                self.assertTrue(
                    directory in string_literals
                    or all(component in string_literals for component in relative.parent.parts),
                    f"directory {directory!r} is absent from {SPEC_PATH}",
                )

        self.assertIn('"models/yolo26n_416_openvino_model"', source)
        self.assertIn('"models/yolo26n_openvino_model"', source)
        self.assertIn('"models/yolo11l_openvino_model"', source)
        self.assertIn('"models/yolo11l_onnx"', source)
        self.assertIn('"models/fort_player_416_openvino_model"', source)
        self.assertIn('"models/fort_player_openvino_model"', source)

    def test_license_and_readme_are_required_and_bundled_at_bundle_root(self) -> None:
        source = SPEC_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(SPEC_PATH))
        normalized = ast.unparse(tree)
        for asset in REQUIRED_ROOT_ASSETS[:2]:
            with self.subTest(asset=asset):
                self.assertIn(f"PROJECT_ROOT / '{asset}'", normalized)
                self.assertIn(f"(str(PROJECT_ROOT / '{asset}'), '.')", normalized)
        self.assertIn("THIRD_PARTY_NOTICES = PROJECT_ROOT / 'THIRD_PARTY_NOTICES.md'", normalized)
        self.assertIn("(str(THIRD_PARTY_NOTICES), '.')", normalized)
        for document in ("MODEL_BENCHMARKS.md", "RELEASE_CHECKLIST.md"):
            self.assertIn(document, source)
        self.assertIn('"docs"', source)

        linux = LINUX_BUILD.read_text(encoding="utf-8")
        windows = WINDOWS_BUILD.read_text(encoding="utf-8")
        for asset in REQUIRED_ROOT_ASSETS:
            with self.subTest(helper_asset=asset):
                name = Path(asset).name
                self.assertIn(name, linux)
                self.assertIn(name, windows)
        self.assertIn('"$PROJECT_DIR/dist/ProAim/LICENSE"', linux)
        self.assertIn('(Join-Path $BundleDir "LICENSE")', windows)

    def test_qt_lgpl_license_is_required_and_bundled(self) -> None:
        source = SPEC_PATH.read_text(encoding="utf-8")
        normalized = ast.unparse(ast.parse(source, filename=str(SPEC_PATH)))
        self.assertIn("PROJECT_ROOT / 'packaging' / 'licenses' / 'LGPL-3.0-only.txt'", normalized)
        self.assertIn("'LGPL-3.0-only.txt'), 'licenses')", normalized)
        self.assertIn("PROJECT_ROOT / 'packaging' / 'licenses' / 'GPL-3.0-only.txt'", normalized)
        self.assertIn("'GPL-3.0-only.txt'), 'licenses')", normalized)

    def test_third_party_license_payloads_are_collected(self) -> None:
        source = SPEC_PATH.read_text(encoding="utf-8")
        self.assertIn("copy_metadata(license_distribution, recursive=True)", source)
        self.assertIn('sysconfig.get_path("stdlib")', source)
        self.assertIn("sys.base_prefix", source)
        self.assertIn("sys.executable", source)
        self.assertIn('"Resources/English.lproj/License.rtf"', source)
        self.assertIn('"licenses/third-party/python"', source)
        self.assertIn("pyserial-3.5-BSD-3-Clause.txt", source)
        self.assertIn('"licenses/third-party/pyserial"', source)
        for distribution in (
            "numpy",
            "opencv-python",
            "openvino",
            "PySide6-Essentials",
            "shiboken6",
            "mss",
            "pyserial",
            "evdev",
        ):
            with self.subTest(distribution=distribution):
                self.assertIn(f'"{distribution}"', source)
        self.assertIn("libqtiff.so", source)
        self.assertIn("for entry in a.binaries", source)
        self.assertIn('"launcher.application"', source)
        self.assertIn('"tkinter"', source)
        self.assertIn("libqgtk3.so", source)
        self.assertIn("nvidia-cudnn-cu13", source)
        self.assertIn('collect_dynamic_libs(\n        "nvidia"', source)

        pyserial_license = (
            PROJECT_ROOT
            / "packaging"
            / "licenses"
            / "pyserial-3.5-BSD-3-Clause.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2001-2020 Chris Liechti", pyserial_license)
        self.assertIn(
            "Redistribution and use in source and binary forms",
            pyserial_license,
        )
        self.assertIn("THIS SOFTWARE IS PROVIDED", pyserial_license)

    def test_generic_yolo_metadata_is_bundled_beside_ir_and_onnx(self) -> None:
        source = SPEC_PATH.read_text(encoding="utf-8")
        expected_copies = (
            ('str(MODEL_DIR / "metadata.yaml")', "models/yolo26n_openvino_model"),
            ('str(MODEL_DIR / "metadata.yaml")', "models/yolo26n_onnx"),
            (
                'str(BALANCED_MODEL_DIR / "metadata.yaml")',
                "models/yolo26n_416_openvino_model",
            ),
            (
                'str(BALANCED_MODEL_DIR / "metadata.yaml")',
                "models/yolo26n_416_onnx",
            ),
            (
                'str(HIGH_END_MODEL_DIR / "metadata.yaml")',
                "models/yolo11l_openvino_model",
            ),
            ('str(HIGH_END_MODEL_DIR / "metadata.yaml")', "models/yolo11l_onnx"),
        )
        for source_expression, destination in expected_copies:
            with self.subTest(destination=destination):
                self.assertIn(source_expression, source)
                self.assertIn(f'"{destination}"', source)

        manifest = (PROJECT_ROOT / "models" / "RELEASE-MANIFEST.sha256").read_text(
            encoding="utf-8"
        )
        for relative in (
            "models/yolo26n_openvino_model/metadata.yaml",
            "models/yolo26n_416_openvino_model/metadata.yaml",
            "models/yolo11l_openvino_model/metadata.yaml",
        ):
            self.assertIn(relative, manifest)

    def test_linux_evdev_collection_is_platform_guarded(self) -> None:
        source = SPEC_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(SPEC_PATH))
        guarded_blocks = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "sys.platform.startswith" in ast.unparse(node.test)
            and "linux" in ast.unparse(node.test)
            and "collect_submodules('evdev')" in ast.unparse(node)
        ]
        self.assertEqual(len(guarded_blocks), 1)
        guarded_source = ast.unparse(guarded_blocks[0])
        self.assertIn("collect_submodules('evdev')", guarded_source)
        self.assertIn("collect_dynamic_libs", guarded_source)
        self.assertIn("find_spec('evdev')", guarded_source)
        self.assertIn('excludes.append("evdev")', source)

    def test_build_helpers_run_release_preflight_before_pyinstaller(self) -> None:
        linux_source = LINUX_BUILD.read_text(encoding="utf-8")
        windows_source = WINDOWS_BUILD.read_text(encoding="utf-8")
        for asset in REQUIRED_MODEL_ASSETS:
            with self.subTest(asset=asset):
                # Packaging-only copies such as format-specific attribution may
                # share the already validated OpenVINO attribution content.
                if asset.endswith("_onnx/ATTRIBUTION.md"):
                    continue
                self.assertIn(asset, RELEASE_PREFLIGHT.read_text(encoding="utf-8"))

        self.assertIn("scripts/validate_release_assets.py", linux_source)
        self.assertIn("scripts\\validate_release_assets.py", windows_source)
        self.assertLess(
            linux_source.index("validate_release_assets.py"),
            linux_source.index('-c "import PyInstaller"'),
        )
        self.assertLess(
            windows_source.index("validate_release_assets.py"),
            windows_source.index('-c "import PyInstaller"'),
        )
        self.assertIn("evdev._ecodes, evdev._input, evdev._uinput", linux_source)
        self.assertIn("import serial, serial.tools.list_ports", linux_source)
        self.assertIn("import serial, serial.tools.list_ports", windows_source)
        self.assertIn("70-game-detector-makcu.rules", linux_source)
        self.assertIn("validate_linux_bundle.py", linux_source)
        self.assertNotIn("import evdev", windows_source)

    def test_windows_build_creates_shareable_zip_and_ci_artifact(self) -> None:
        build = WINDOWS_BUILD.read_text(encoding="utf-8")
        workflow = WINDOWS_WORKFLOW.read_text(encoding="utf-8")
        guide = WINDOWS_GUIDE.read_text(encoding="utf-8")
        self.assertIn("ProAim-Windows-x64-", build)
        self.assertIn("NVIDIA-CUDA", build)
        self.assertIn("DirectML", build)
        self.assertIn("Compress-Archive", build)
        self.assertIn("RuntimeVariant", build)
        self.assertIn("README-Windows.txt", build)
        self.assertIn("workflow_dispatch", workflow)
        self.assertIn("windows-2022", workflow)
        self.assertIn("ProAim-Windows-x64-NVIDIA-CUDA", workflow)
        self.assertIn("ProAim-Windows-x64-DirectML", workflow)
        self.assertIn("Scan hardware", guide)
        self.assertIn("GPU", guide)
        self.assertIn("NPU", guide)

    def test_makcu_rule_is_narrow_and_documented(self) -> None:
        rule = MAKCU_RULE.read_text(encoding="utf-8")
        self.assertIn('ATTRS{idVendor}=="1a86"', rule)
        self.assertIn('ATTRS{idProduct}=="55d3"', rule)
        self.assertIn('TAG+="uaccess"', rule)
        readme = README.read_text(encoding="utf-8")
        self.assertIn("Detection → MAKCU aim", readme)
        self.assertIn("Start detection", readme)
        self.assertIn("70-game-detector-makcu.rules", readme)

    @unittest.skipIf(os.name == "nt", "Linux shell syntax is validated on Linux CI")
    def test_linux_build_helper_has_valid_shell_syntax(self) -> None:
        for script in (LINUX_BUILD, MAKCU_ACCESS_INSTALLER):
            with self.subTest(script=script):
                result = subprocess.run(
                    ["bash", "-n", script.as_posix()],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_bundled_makcu_installer_supports_source_and_setup_layouts(self) -> None:
        source = MAKCU_ACCESS_INSTALLER.read_text(encoding="utf-8")
        self.assertIn("packaging/linux/70-game-detector-makcu.rules", source)
        self.assertIn('$SCRIPT_DIR/70-game-detector-makcu.rules', source)
        readme = README.read_text(encoding="utf-8")
        self.assertIn("setup/install_makcu_access.sh", readme)

    def test_readme_distinguishes_precision_from_detection_driven_input(self) -> None:
        source = README.read_text(encoding="utf-8")
        self.assertIn("Optional MAKCU aim", source)
        self.assertIn("explicit **Target label**", source)
        self.assertIn("**Controller precision**", source)
        self.assertIn("Verify LT + right stick", source)
        self.assertIn("Start controller precision", source)
        self.assertIn("before opening Moonlight", source)
        self.assertIn("It does not read\ndetections or choose a direction", source)
        self.assertIn("`--aim-output remote` is deliberately unavailable", source)
        self.assertIn("PXN P5 8K", source)
        self.assertIn("/dev/uinput", source)
        self.assertNotIn(
            "It never reads game memory or generates mouse, keyboard, or controller input.",
            source,
        )


if __name__ == "__main__":
    unittest.main()
