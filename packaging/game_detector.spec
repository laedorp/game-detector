# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller definition for the cross-platform ProAim desktop app.

Build this file on the target operating system. PyInstaller does not
cross-compile, so a Windows .exe must be produced on Windows and the Linux
binary must be produced on Linux.
"""

from importlib import metadata
from importlib.util import find_spec
import os
from pathlib import Path, PurePosixPath
import sysconfig
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


PROJECT_ROOT = Path(SPECPATH).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.release_model_contract import (  # noqa: E402
    CONTRACT_RELATIVE as RELEASE_DEFAULT_CONTRACT_RELATIVE,
    load_release_default_contract,
)

ENTRY_POINT = PROJECT_ROOT / "app.py"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_DIR = PROJECT_ROOT / "models" / "yolo26n_openvino_model"
BALANCED_MODEL_DIR = PROJECT_ROOT / "models" / "yolo26n_416_openvino_model"
HIGH_END_MODEL_DIR = PROJECT_ROOT / "models" / "yolo11l_openvino_model"
HIGH_END_ONNX_DIR = PROJECT_ROOT / "models" / "yolo11l_onnx"
PLAYER_MODEL_DIR = PROJECT_ROOT / "models" / "fort_player_openvino_model"
PLAYER_INT8_MODEL_DIR = PROJECT_ROOT / "models" / "fort_player_416_int8_openvino_model"
# ONNX copies of every bundled model.  OpenVINO cannot drive AMD or NVIDIA
# GPUs, so a build without these cannot run on that hardware at all.
ONNX_MODEL_DIRS = {
    "models/fort_player_onnx": "fort_player.onnx",
    "models/yolo26n_416_onnx": "yolo26n_416.onnx",
    "models/yolo26n_onnx": "yolo26n.onnx",
}
THIRD_PARTY_NOTICES = PROJECT_ROOT / "THIRD_PARTY_NOTICES.md"
MODEL_MANIFEST = PROJECT_ROOT / "models" / "RELEASE-MANIFEST.sha256"
RELEASE_DEFAULT_CONTRACT = PROJECT_ROOT.joinpath(
    *RELEASE_DEFAULT_CONTRACT_RELATIVE.parts
)
RELEASE_DEFAULT_POINTER = load_release_default_contract(
    PROJECT_ROOT, verify_files=True
)
RELEASE_DEFAULT_MEMBER_PATHS = tuple(
    PROJECT_ROOT.joinpath(*PurePosixPath(record["path"]).parts)
    for record in RELEASE_DEFAULT_POINTER["artifacts"].values()
)
PYSERIAL_LICENSE = (
    PROJECT_ROOT / "packaging" / "licenses" / "pyserial-3.5-BSD-3-Clause.txt"
)
DXCAM_LICENSE = (
    PROJECT_ROOT / "packaging" / "licenses" / "dxcam-0.3.0-MIT.txt"
)
WINDOWS_ICON = ASSETS_DIR / "game-detector.ico"
RUNTIME_VARIANT = os.environ.get("PROAIM_RUNTIME_VARIANT", "cpu").strip().lower()
RUNTIME_DISTRIBUTIONS = {
    "cpu": "onnxruntime",
    "cuda": "onnxruntime-gpu",
    "directml": "onnxruntime-directml",
    "rocm": "onnxruntime-rocm",
}
if RUNTIME_VARIANT not in RUNTIME_DISTRIBUTIONS:
    raise SystemExit(
        f"Unknown PROAIM_RUNTIME_VARIANT={RUNTIME_VARIANT!r}; choose one of "
        + ", ".join(sorted(RUNTIME_DISTRIBUTIONS))
    )
expected_runtime_distribution = RUNTIME_DISTRIBUTIONS[RUNTIME_VARIANT]
installed_runtime_distributions = []
for distribution in RUNTIME_DISTRIBUTIONS.values():
    try:
        metadata.version(distribution)
    except metadata.PackageNotFoundError:
        continue
    installed_runtime_distributions.append(distribution)
if installed_runtime_distributions != [expected_runtime_distribution]:
    raise SystemExit(
        "Cannot build ProAim: expected exactly "
        f"{expected_runtime_distribution!r} for the {RUNTIME_VARIANT!r} runtime, "
        f"found {installed_runtime_distributions or 'none'}. Install only the matching "
        "requirements-runtime-*.txt file in this build environment."
    )

required_files = (
    ENTRY_POINT,
    PROJECT_ROOT / "LICENSE",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "DEPENDENCY_LOCKS.md",
    PROJECT_ROOT / "docs" / "MODEL_ACCURACY_EVALUATION.md",
    PROJECT_ROOT / "docs" / "MODEL_BENCHMARKS.md",
    PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md",
    THIRD_PARTY_NOTICES,
    MODEL_MANIFEST,
    PROJECT_ROOT / "packaging" / "licenses" / "LGPL-3.0-only.txt",
    PROJECT_ROOT / "packaging" / "licenses" / "GPL-3.0-only.txt",
    PYSERIAL_LICENSE,
    ASSETS_DIR / "game-detector.svg",
    ASSETS_DIR / "chevron-down.svg",
    PROJECT_ROOT / "models" / "coco80.txt",
    MODEL_DIR / "yolo26n.xml",
    MODEL_DIR / "yolo26n.bin",
    MODEL_DIR / "metadata.yaml",
    BALANCED_MODEL_DIR / "yolo26n_416.xml",
    BALANCED_MODEL_DIR / "yolo26n_416.bin",
    BALANCED_MODEL_DIR / "metadata.yaml",
    HIGH_END_MODEL_DIR / "yolo11l.xml",
    HIGH_END_MODEL_DIR / "yolo11l.bin",
    HIGH_END_MODEL_DIR / "metadata.yaml",
    HIGH_END_ONNX_DIR / "yolo11l.onnx",
    PROJECT_ROOT / "models" / "fort_player.txt",
    PLAYER_MODEL_DIR / "fort_player.xml",
    PLAYER_MODEL_DIR / "fort_player.bin",
    PLAYER_MODEL_DIR / "ATTRIBUTION.md",
    PLAYER_INT8_MODEL_DIR / "fort_player_416_int8.xml",
    PLAYER_INT8_MODEL_DIR / "fort_player_416_int8.bin",
    PLAYER_INT8_MODEL_DIR / "metadata.yaml",
    PLAYER_INT8_MODEL_DIR / "ATTRIBUTION.md",
) + tuple(
    PROJECT_ROOT / directory / filename
    for directory, filename in ONNX_MODEL_DIRS.items()
) + tuple(
    PROJECT_ROOT / directory / "ATTRIBUTION.md"
    for directory in ("models/fort_player_onnx",)
) + (RELEASE_DEFAULT_CONTRACT,) + RELEASE_DEFAULT_MEMBER_PATHS
required_files = tuple(dict.fromkeys(required_files))
if sys.platform == "win32":
    required_files += (WINDOWS_ICON, DXCAM_LICENSE)
missing_files = [str(path) for path in required_files if not path.is_file()]
if missing_files:
    raise SystemExit(
        "Cannot build ProAim; required file(s) are missing:\n  "
        + "\n  ".join(missing_files)
    )


datas = [
    (str(PROJECT_ROOT / "LICENSE"), "."),
    (str(PROJECT_ROOT / "README.md"), "."),
    (str(PROJECT_ROOT / "docs" / "DEPENDENCY_LOCKS.md"), "docs"),
    (str(PROJECT_ROOT / "docs" / "MODEL_ACCURACY_EVALUATION.md"), "docs"),
    (str(PROJECT_ROOT / "docs" / "MODEL_BENCHMARKS.md"), "docs"),
    (str(PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md"), "docs"),
    (str(THIRD_PARTY_NOTICES), "."),
    (str(MODEL_MANIFEST), "models"),
    (
        str(PROJECT_ROOT / "packaging" / "licenses" / "LGPL-3.0-only.txt"),
        "licenses",
    ),
    (
        str(PROJECT_ROOT / "packaging" / "licenses" / "GPL-3.0-only.txt"),
        "licenses",
    ),
    (str(PYSERIAL_LICENSE), "licenses/third-party/pyserial"),
    (str(ASSETS_DIR / "game-detector.svg"), "assets"),
    (str(ASSETS_DIR / "chevron-down.svg"), "assets"),
    (str(PROJECT_ROOT / "models" / "coco80.txt"), "models"),
    (str(MODEL_DIR / "yolo26n.xml"), "models/yolo26n_openvino_model"),
    (str(MODEL_DIR / "yolo26n.bin"), "models/yolo26n_openvino_model"),
    (str(MODEL_DIR / "metadata.yaml"), "models/yolo26n_openvino_model"),
    (str(MODEL_DIR / "metadata.yaml"), "models/yolo26n_onnx"),
    (str(BALANCED_MODEL_DIR / "yolo26n_416.xml"), "models/yolo26n_416_openvino_model"),
    (str(BALANCED_MODEL_DIR / "yolo26n_416.bin"), "models/yolo26n_416_openvino_model"),
    (
        str(BALANCED_MODEL_DIR / "metadata.yaml"),
        "models/yolo26n_416_openvino_model",
    ),
    (str(BALANCED_MODEL_DIR / "metadata.yaml"), "models/yolo26n_416_onnx"),
    (str(HIGH_END_MODEL_DIR / "yolo11l.xml"), "models/yolo11l_openvino_model"),
    (str(HIGH_END_MODEL_DIR / "yolo11l.bin"), "models/yolo11l_openvino_model"),
    (str(HIGH_END_MODEL_DIR / "metadata.yaml"), "models/yolo11l_openvino_model"),
    (str(HIGH_END_MODEL_DIR / "metadata.yaml"), "models/yolo11l_onnx"),
    (str(PROJECT_ROOT / "models" / "fort_player.txt"), "models"),
    (str(PLAYER_MODEL_DIR / "fort_player.xml"), "models/fort_player_openvino_model"),
    (str(PLAYER_MODEL_DIR / "fort_player.bin"), "models/fort_player_openvino_model"),
    (str(PLAYER_MODEL_DIR / "ATTRIBUTION.md"), "models/fort_player_openvino_model"),
    (
        str(PLAYER_INT8_MODEL_DIR / "fort_player_416_int8.xml"),
        "models/fort_player_416_int8_openvino_model",
    ),
    (
        str(PLAYER_INT8_MODEL_DIR / "fort_player_416_int8.bin"),
        "models/fort_player_416_int8_openvino_model",
    ),
    (
        str(PLAYER_INT8_MODEL_DIR / "metadata.yaml"),
        "models/fort_player_416_int8_openvino_model",
    ),
    (
        str(PLAYER_INT8_MODEL_DIR / "ATTRIBUTION.md"),
        "models/fort_player_416_int8_openvino_model",
    ),
]

def _find_python_license():
    """Locate the CPython license in official Unix, Windows, and macOS layouts."""

    configured_roots = (
        sysconfig.get_path("stdlib"),
        sysconfig.get_config_var("installed_base"),
        sysconfig.get_config_var("base"),
        sys.base_prefix,
        sys.prefix,
        sys.exec_prefix,
        Path(sys.executable).resolve().parent,
    )
    checked = []
    for configured_root in configured_roots:
        if not configured_root:
            continue
        root = Path(configured_root).resolve()
        for relative in (
            "LICENSE.txt",
            "LICENSE",
            "LICENSE.rst",
            "Doc/license.rst",
            "Resources/English.lproj/License.rtf",
        ):
            candidate = root / relative
            if candidate in checked:
                continue
            checked.append(candidate)
            if candidate.is_file():
                return candidate
    raise SystemExit(
        "Cannot build ProAim; Python license text is missing. Checked:\n  "
        + "\n  ".join(str(path) for path in checked)
    )


python_license = _find_python_license()
datas.append((str(python_license), "licenses/third-party/python"))
datas += [
    (str(PROJECT_ROOT / directory / filename), directory)
    for directory, filename in ONNX_MODEL_DIRS.items()
]
datas += [
    (str(PROJECT_ROOT / directory / "ATTRIBUTION.md"), directory)
    for directory in ("models/fort_player_onnx",)
]
datas += [(str(HIGH_END_ONNX_DIR / "yolo11l.onnx"), "models/yolo11l_onnx")]
datas.append((str(RELEASE_DEFAULT_CONTRACT), "models"))
packaged_data_sources = {str(Path(source).resolve()) for source, _destination in datas}
for member in RELEASE_DEFAULT_MEMBER_PATHS:
    source = str(member.resolve())
    if source in packaged_data_sources:
        continue
    relative_parent = member.relative_to(PROJECT_ROOT).parent.as_posix()
    datas.append((source, relative_parent))
    packaged_data_sources.add(source)

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

# Windows desktop capture prefers DXcam's Desktop Duplication path. Its
# processor kernels and COM helpers are selected dynamically, so explicitly
# retain the whole package rather than relying on static import discovery.
if sys.platform == "win32":
    dxcam_spec = find_spec("dxcam")
    if dxcam_spec is None or dxcam_spec.origin is None:
        raise SystemExit(
            "Cannot build ProAim for Windows; DXcam is missing. "
            "Install requirements.txt in the build environment first."
        )
    # Do not import DXcam while building: its package initializer enumerates
    # DXGI outputs, which need not exist on a headless Windows build runner.
    # Enumerate import names directly from the installed wheel instead.
    dxcam_root = Path(dxcam_spec.origin).resolve().parent
    dxcam_modules = []
    for python_file in dxcam_root.rglob("*.py"):
        module_parts = list(python_file.relative_to(dxcam_root).with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts.pop()
        module_name = ".".join(("dxcam", *module_parts))
        if module_name:
            dxcam_modules.append(module_name)
    hiddenimports += sorted(set(dxcam_modules))
    hiddenimports.append("dxcam.processor._numpy_kernels")
    binaries += collect_dynamic_libs("dxcam")
    datas += collect_data_files("dxcam")
    datas.append((str(DXCAM_LICENSE), "licenses/third-party/dxcam"))

# Every bundle carries exactly one ONNX Runtime provider selected by its build
# variant. The detector imports it lazily, so make the native provider package
# explicit and let PyInstaller's hook retain its platform libraries.
if find_spec("onnxruntime") is None:
    raise SystemExit(
        f"Cannot build ProAim; {expected_runtime_distribution} is not importable."
    )
hiddenimports += collect_submodules("onnxruntime")
binaries += collect_dynamic_libs("onnxruntime")
datas += collect_data_files("onnxruntime")

if RUNTIME_VARIANT == "cuda":
    nvidia_distributions = (
        "nvidia-cuda-nvrtc",
        "nvidia-cuda-runtime",
        "nvidia-cufft",
        "nvidia-curand",
        "nvidia-cudnn-cu13",
        "nvidia-cublas",
        "nvidia-nvjitlink",
    )
    missing_nvidia = []
    for distribution in nvidia_distributions:
        try:
            metadata.version(distribution)
        except metadata.PackageNotFoundError:
            missing_nvidia.append(distribution)
    if missing_nvidia:
        raise SystemExit(
            "Cannot build ProAim CUDA bundle; required user-space runtime "
            "package(s) are missing: " + ", ".join(missing_nvidia)
        )
    if find_spec("nvidia") is None:
        raise SystemExit("Cannot build ProAim CUDA bundle; nvidia namespace is missing.")
    hiddenimports += collect_submodules("nvidia")
    binaries += collect_dynamic_libs(
        "nvidia", search_patterns=["*.dll", "*.so", "*.so.*"]
    )
    datas += collect_data_files("nvidia", include_py_files=True)
    for distribution in nvidia_distributions:
        datas += copy_metadata(distribution, recursive=True)

# Collect the installed distributions' complete license payloads.  PyInstaller
# hooks primarily retain importable code and native libraries; dist-info
# license directories are otherwise easy to omit from a redistributed bundle.
license_distributions = [
    "numpy",
    "opencv-python",
    "openvino",
    expected_runtime_distribution,
    "PySide6-Essentials",
    "shiboken6",
    "mss",
    "pyserial",
]
if sys.platform == "win32":
    license_distributions.extend(["dxcam", "comtypes"])
if sys.platform.startswith("linux"):
    license_distributions.append("evdev")
for license_distribution in license_distributions:
    datas += copy_metadata(license_distribution, recursive=True)

# Some wheels keep their license at package root instead of dist-info.
for package_license in ("onnxruntime/LICENSE", "cv2/LICENSE.txt", "cv2/LICENSE-3RD-PARTY.txt"):
    module_name, relative_name = package_license.split("/", 1)
    module_spec = find_spec(module_name)
    if module_spec is not None and module_spec.submodule_search_locations:
        candidate = Path(next(iter(module_spec.submodule_search_locations))) / relative_name
        if candidate.is_file():
            datas.append(
                (
                    str(candidate),
                    f"licenses/third-party/{module_name}",
                )
            )

# app.py imports the Qt launcher lazily at runtime. Name that entry point and
# its direct helpers explicitly so the frozen app always carries them.
hiddenimports += [
    "launcher.qt_app",
    "launcher.qt_theme",
    "launcher.process",
    "launcher.settings",
    "scripts.benchmark_models",
    "utils.live_report",
]

# app.py imports the Qt front end lazily inside a function so that a source
# checkout without PySide6 still runs the CLI.  PyInstaller's static analysis
# cannot see through that, so the Qt modules must be named explicitly or the
# bundled application would start and immediately fail to find its interface.
if find_spec("PySide6") is None:
    raise SystemExit(
        "Cannot build ProAim; PySide6 is required for the desktop interface. "
        "Install requirements-build.txt in the build environment first."
    )
hiddenimports += [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
]

if find_spec("serial") is None:
    raise SystemExit(
        "Cannot build ProAim; pyserial required for MAKCU output is missing. "
        "Install requirements.txt in the build environment first."
    )
hiddenimports += collect_submodules("serial")

# Controller precision is Linux-only.  Its backend deliberately imports evdev
# optionally so source runs and Windows bundles remain usable without it.
# PyInstaller therefore needs the package's runtime-selected submodules and C
# extensions added explicitly on Linux, while Windows must not try to resolve
# or bundle evdev at all.
if sys.platform.startswith("linux"):
    if find_spec("evdev") is None:
        raise SystemExit(
            "Cannot build ProAim for Linux; the evdev package required "
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
    "launcher.application",
    "matplotlib",
    # The desktop runtime is intentionally offline.  OpenVINO's conversion
    # package otherwise initializes its optional analytics client during the
    # top-level ``openvino`` import; excluding it activates OpenVINO's bundled
    # no-op telemetry stub while leaving inference unchanged.
    "openvino_telemetry",
    "pandas",
    "scipy",
    "tensorflow",
    "tkinter",
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

# PySide's hook discovers plugins during Analysis, after the manually prepared
# binary list above. Remove the unused TIFF plugin from the finalized TOC so a
# wheel linked to a legacy libtiff ABI cannot leave the bundle with a broken
# native dependency.
a.binaries = [
    entry
    for entry in a.binaries
    if Path(str(entry[0])).name
    not in {"libqtiff.so", "qtiff.dll", "libqgtk3.so", "qgtk3.dll"}
]
pyz = PYZ(a.pure)

gui_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ProAim",
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
        name="ProAimCLI",
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
    name="ProAim",
)
