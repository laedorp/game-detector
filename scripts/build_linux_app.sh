#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_PYTHON="${GAME_DETECTOR_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
RUNTIME_VARIANT="${PROAIM_RUNTIME_VARIANT:-cpu}"

case "$RUNTIME_VARIANT" in
    cpu) EXPECTED_RUNTIME="onnxruntime" ;;
    cuda) EXPECTED_RUNTIME="onnxruntime-gpu" ;;
    rocm) EXPECTED_RUNTIME="onnxruntime-rocm" ;;
    *)
        echo "Unsupported Linux runtime variant: $RUNTIME_VARIANT (use cpu, cuda, or rocm)." >&2
        exit 2
        ;;
esac
export PROAIM_RUNTIME_VARIANT="$RUNTIME_VARIANT"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This helper builds the Linux app. Use build_windows_app.ps1 on Windows." >&2
    exit 2
fi

if [[ ! -f "$PROJECT_DIR/app.py" ]]; then
    echo "Desktop entry point is missing: $PROJECT_DIR/app.py" >&2
    exit 2
fi

if [[ ! -x "$PROJECT_PYTHON" ]]; then
    echo "Python environment not found at: $PROJECT_PYTHON" >&2
    echo "Create .venv and install requirements-build.txt first." >&2
    exit 2
fi

"$PROJECT_PYTHON" "$PROJECT_DIR/scripts/validate_release_assets.py" \
    --project-root "$PROJECT_DIR"

if ! "$PROJECT_PYTHON" -c "import PyInstaller" >/dev/null 2>&1; then
    echo "PyInstaller is missing. Run:" >&2
    echo "  '$PROJECT_PYTHON' -m pip install -r '$PROJECT_DIR/requirements-build.txt'" >&2
    exit 2
fi

if ! "$PROJECT_PYTHON" -c "import evdev, evdev._ecodes, evdev._input, evdev._uinput" >/dev/null 2>&1; then
    echo "Linux Controller precision cannot be bundled because evdev is missing or incomplete." >&2
    echo "Install the Linux runtime dependencies first:" >&2
    echo "  '$PROJECT_PYTHON' -m pip install -r '$PROJECT_DIR/requirements.txt'" >&2
    exit 2
fi

if ! "$PROJECT_PYTHON" -c "import serial, serial.tools.list_ports" >/dev/null 2>&1; then
    echo "MAKCU support cannot be bundled because pyserial is missing." >&2
    echo "Install the runtime dependencies first:" >&2
    echo "  '$PROJECT_PYTHON' -m pip install -r '$PROJECT_DIR/requirements.txt'" >&2
    exit 2
fi

if ! "$PROJECT_PYTHON" -c "from importlib import metadata; metadata.version('$EXPECTED_RUNTIME'); import onnxruntime" >/dev/null 2>&1; then
    echo "The $RUNTIME_VARIANT build requires $EXPECTED_RUNTIME." >&2
    echo "Install the matching requirements-runtime-$RUNTIME_VARIANT.txt file first." >&2
    exit 2
fi

cd "$PROJECT_DIR"
"$PROJECT_PYTHON" -m PyInstaller --noconfirm --clean packaging/game_detector.spec

install -d "$PROJECT_DIR/dist/ProAim/setup"
install -m 0755 "$PROJECT_DIR/scripts/install_linux_desktop.py" \
    "$PROJECT_DIR/dist/ProAim/setup/install_linux_desktop.py"
install -m 0644 "$PROJECT_DIR/packaging/linux/game-detector.desktop.in" \
    "$PROJECT_DIR/dist/ProAim/setup/game-detector.desktop.in"
install -m 0644 "$PROJECT_DIR/assets/game-detector.svg" \
    "$PROJECT_DIR/dist/ProAim/setup/game-detector.svg"
install -m 0644 "$PROJECT_DIR/packaging/linux/70-game-detector-makcu.rules" \
    "$PROJECT_DIR/dist/ProAim/setup/70-game-detector-makcu.rules"
install -m 0755 "$PROJECT_DIR/scripts/install_makcu_access.sh" \
    "$PROJECT_DIR/dist/ProAim/setup/install_makcu_access.sh"
install -m 0755 "$PROJECT_DIR/packaging/linux/install.sh" \
    "$PROJECT_DIR/dist/ProAim/install.sh"

# PyInstaller stores datas under its private _internal directory. Keep copies
# there for resource_root()/About, and also put the user-facing legal and help
# documents beside the executable so an extracted release is self-explanatory.
install -m 0644 "$PROJECT_DIR/LICENSE" \
    "$PROJECT_DIR/dist/ProAim/LICENSE"
install -m 0644 "$PROJECT_DIR/README.md" \
    "$PROJECT_DIR/dist/ProAim/README.md"
install -m 0644 "$PROJECT_DIR/THIRD_PARTY_NOTICES.md" \
    "$PROJECT_DIR/dist/ProAim/THIRD_PARTY_NOTICES.md"
install -d "$PROJECT_DIR/dist/ProAim/docs"
install -m 0644 "$PROJECT_DIR/docs/MODEL_BENCHMARKS.md" \
    "$PROJECT_DIR/dist/ProAim/docs/MODEL_BENCHMARKS.md"
install -m 0644 "$PROJECT_DIR/docs/RELEASE_CHECKLIST.md" \
    "$PROJECT_DIR/dist/ProAim/docs/RELEASE_CHECKLIST.md"

# Qt is dynamically linked in the one-folder bundle. Keep the reviewed,
# repository-pinned LGPL text next to the application for redistribution and
# relinking notice.  Do not depend on distribution-specific system paths: the
# same build helper runs on release CI images that may not install system Qt.
install -d "$PROJECT_DIR/dist/ProAim/licenses"
QT_LGPL_LICENSE="$PROJECT_DIR/packaging/licenses/LGPL-3.0-only.txt"
if [[ ! -f "$QT_LGPL_LICENSE" ]]; then
    echo "Could not find LGPL-3.0-only.txt required for the Qt bundle notice." >&2
    exit 2
fi
install -m 0644 "$QT_LGPL_LICENSE" \
    "$PROJECT_DIR/dist/ProAim/licenses/LGPL-3.0-only.txt"
install -m 0644 "$PROJECT_DIR/packaging/licenses/GPL-3.0-only.txt" \
    "$PROJECT_DIR/dist/ProAim/licenses/GPL-3.0-only.txt"

"$PROJECT_PYTHON" "$PROJECT_DIR/scripts/write_build_info.py" \
    --bundle "$PROJECT_DIR/dist/ProAim" \
    --runtime-variant "$RUNTIME_VARIANT"

VALIDATE_BUNDLE_ARGS=("$PROJECT_DIR/dist/ProAim")
if [[ "$RUNTIME_VARIANT" == "cuda" ]]; then
    # The redistributable CUDA user-space libraries are bundled, but the
    # kernel driver library is supplied by the target NVIDIA host and is not
    # present on GitHub's CPU-only build runner.
    VALIDATE_BUNDLE_ARGS+=(--allow-missing libcuda.so.1)
fi
"$PROJECT_PYTHON" "$PROJECT_DIR/scripts/validate_linux_bundle.py" \
    "${VALIDATE_BUNDLE_ARGS[@]}"

echo
echo "Linux desktop bundle created:"
echo "  $PROJECT_DIR/dist/ProAim/ProAim"
echo "Install it in your application menu with:"
echo "  $PROJECT_DIR/dist/ProAim/install.sh"
