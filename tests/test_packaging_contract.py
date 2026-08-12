from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "packaging" / "game_detector.spec"
LINUX_BUILD = PROJECT_ROOT / "scripts" / "build_linux_app.sh"
MAKCU_ACCESS_INSTALLER = PROJECT_ROOT / "scripts" / "install_makcu_access.sh"
WINDOWS_BUILD = PROJECT_ROOT / "scripts" / "build_windows_app.ps1"
WINDOWS_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "build-windows.yml"
WINDOWS_GUIDE = PROJECT_ROOT / "packaging" / "windows" / "README-Windows.txt"
RELEASE_PREFLIGHT = PROJECT_ROOT / "scripts" / "validate_release_assets.py"
README = PROJECT_ROOT / "README.md"
MAKCU_RULE = PROJECT_ROOT / "packaging" / "linux" / "70-game-detector-makcu.rules"

REQUIRED_MODEL_ASSETS = (
    "models/coco80.txt",
    "models/yolo26n_openvino_model/yolo26n.xml",
    "models/yolo26n_openvino_model/yolo26n.bin",
    "models/yolo26n_416_openvino_model/yolo26n_416.xml",
    "models/yolo26n_416_openvino_model/yolo26n_416.bin",
    "models/yolo11l_openvino_model/yolo11l.xml",
    "models/yolo11l_openvino_model/yolo11l.bin",
    "models/yolo11l_onnx/yolo11l.onnx",
)


class PackagingContractTests(unittest.TestCase):
    def test_spec_is_valid_python_and_mentions_every_required_model_asset(self) -> None:
        source = SPEC_PATH.read_text(encoding="utf-8")
        ast.parse(source, filename=str(SPEC_PATH))

        for asset in REQUIRED_MODEL_ASSETS:
            with self.subTest(asset=asset):
                for component in Path(asset).parts:
                    self.assertTrue(
                        f'"{component}"' in source or f"'{component}'" in source,
                        f"{component!r} is absent from {SPEC_PATH}",
                    )

        self.assertIn('"models/yolo26n_416_openvino_model"', source)
        self.assertIn('"models/yolo26n_openvino_model"', source)
        self.assertIn('"models/yolo11l_openvino_model"', source)
        self.assertIn('"models/yolo11l_onnx"', source)

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
        self.assertIn("Refresh devices", guide)
        self.assertIn("GPU", guide)
        self.assertIn("NPU", guide)

    def test_makcu_rule_is_narrow_and_documented(self) -> None:
        rule = MAKCU_RULE.read_text(encoding="utf-8")
        self.assertIn('ATTRS{idVendor}=="1a86"', rule)
        self.assertIn('ATTRS{idProduct}=="55d3"', rule)
        self.assertIn('TAG+="uaccess"', rule)
        readme = README.read_text(encoding="utf-8")
        self.assertIn("MAKCU detection aim", readme)
        self.assertIn("Start capture + AI preview", readme)
        self.assertIn("70-game-detector-makcu.rules", readme)

    def test_linux_build_helper_has_valid_shell_syntax(self) -> None:
        for script in (LINUX_BUILD, MAKCU_ACCESS_INSTALLER):
            with self.subTest(script=script):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_readme_distinguishes_precision_from_detection_driven_input(self) -> None:
        source = README.read_text(encoding="utf-8")
        self.assertIn("MAKCU detection aim", source)
        self.assertIn("km.move(dx,dy)", source)
        self.assertIn("Linux Controller precision", source)
        self.assertIn("Verify LT + right stick", source)
        self.assertIn("Start controller precision", source)
        self.assertIn("Only after the status says it is active, open Moonlight", source)
        self.assertIn("does not import video or detector modules, listen on UDP", source)
        self.assertIn("PXN P5 8K", source)
        self.assertIn("/dev/uinput", source)
        self.assertNotIn(
            "It never reads game memory or generates mouse, keyboard, or controller input.",
            source,
        )


if __name__ == "__main__":
    unittest.main()
