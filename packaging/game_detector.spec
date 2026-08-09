# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller definition for the cross-platform Game Detector desktop app.

Build this file on the target operating system. PyInstaller does not
cross-compile, so a Windows .exe must be produced on Windows and the Linux
binary must be produced on Linux.
"""

from importlib.util import find_spec
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


PROJECT_ROOT = Path(SPECPATH).resolve().parent
ENTRY_POINT = PROJECT_ROOT / "app.py"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_DIR = PROJECT_ROOT / "models" / "yolo26n_openvino_model"
FORT_MODEL_DIR = PROJECT_ROOT / "models" / "fort_player_openvino_model"
WINDOWS_ICON = ASSETS_DIR / "game-detector.ico"

required_files = (
    ENTRY_POINT,
    ASSETS_DIR / "game-detector.svg",
    PROJECT_ROOT / "models" / "coco80.txt",
    MODEL_DIR / "yolo26n.xml",
    MODEL_DIR / "yolo26n.bin",
    PROJECT_ROOT / "models" / "fort_player.txt",
    FORT_MODEL_DIR / "fort_player.xml",
    FORT_MODEL_DIR / "fort_player.bin",
    FORT_MODEL_DIR / "ATTRIBUTION.md",
)
if sys.platform == "win32":
    required_files += (WINDOWS_ICON,)
missing_files = [str(path) for path in required_files if not path.is_file()]
if missing_files:
    raise SystemExit(
        "Cannot build Game Detector; required file(s) are missing:\n  "
        + "\n  ".join(missing_files)
    )


datas = [
    (str(ASSETS_DIR / "game-detector.svg"), "assets"),
    (str(PROJECT_ROOT / "models" / "coco80.txt"), "models"),
    (str(MODEL_DIR / "yolo26n.xml"), "models/yolo26n_openvino_model"),
    (str(MODEL_DIR / "yolo26n.bin"), "models/yolo26n_openvino_model"),
    (str(PROJECT_ROOT / "models" / "fort_player.txt"), "models"),
    (str(FORT_MODEL_DIR / "fort_player.xml"), "models/fort_player_openvino_model"),
    (str(FORT_MODEL_DIR / "fort_player.bin"), "models/fort_player_openvino_model"),
    (str(FORT_MODEL_DIR / "ATTRIBUTION.md"), "models/fort_player_openvino_model"),
]

# OpenVINO locates device plugins and model frontends dynamically in its
# package-local libs directory. Keep both its data index and native libraries
# in that same relative layout inside the bundle.
datas += collect_data_files("openvino", subdir="libs")
openvino_library_patterns = [
    "*.dll",
    "*.dylib",
    "*.so",
    "*.so.*",
    "lib*.so",
    "lib*.so.*",
]
binaries = collect_dynamic_libs(
    "openvino",
    search_patterns=openvino_library_patterns,
)

# MSS chooses a backend at runtime based on the host platform. Keeping all of
# its small backend modules makes the same spec usable on Linux and Windows.
hiddenimports = collect_submodules("mss")

# Controller precision is Linux-only.  Its backend deliberately imports evdev
# optionally so source runs and Windows bundles remain usable without it.
# PyInstaller therefore needs the package's runtime-selected submodules and C
# extensions added explicitly on Linux, while Windows must not try to resolve
# or bundle evdev at all.
if sys.platform.startswith("linux"):
    if find_spec("evdev") is None:
        raise SystemExit(
            "Cannot build Game Detector for Linux; the evdev package required "
            "by Controller precision is missing. Install requirements.txt in "
            "the build environment first."
        )
    hiddenimports += collect_submodules("evdev")
    binaries += collect_dynamic_libs(
        "evdev",
        search_patterns=["*.so", "*.so.*"],
    )

# Training/export packages are intentionally not runtime dependencies. These
# exclusions also prevent optional OpenVINO frontends from pulling them into a
# desktop build when they happen to exist in the build environment.
excludes = [
    "matplotlib",
    # The desktop runtime is intentionally offline.  OpenVINO's conversion
    # package otherwise initializes its optional analytics client during the
    # top-level ``openvino`` import; excluding it activates OpenVINO's bundled
    # no-op telemetry stub while leaving inference unchanged.
    "openvino_telemetry",
    "pandas",
    "scipy",
    "tensorflow",
    "torch",
    "torchvision",
    "ultralytics",
]
if not sys.platform.startswith("linux"):
    excludes.append("evdev")

a = Analysis(
    [str(ENTRY_POINT)],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

gui_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GameDetector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(WINDOWS_ICON) if sys.platform == "win32" else None,
)

executables = [gui_exe]
if sys.platform == "win32":
    # The windowed GUI cannot provide Python stdout/stderr on Windows.  A
    # console-subsystem sibling is launched invisibly with redirected pipes
    # for detector work, preserving the GUI's live log and useful errors.
    cli_exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="GameDetectorCLI",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        icon=str(WINDOWS_ICON),
    )
    executables.append(cli_exe)

coll = COLLECT(
    *executables,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="GameDetector",
)
