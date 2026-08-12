#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_PYTHON="${GAME_DETECTOR_PYTHON:-$PROJECT_DIR/.venv/bin/python}"

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

if ! "$PROJECT_PYTHON" -c "import tkinter; tkinter.Tcl()" >/dev/null 2>&1; then
    echo "Tk is missing, so the desktop UI cannot be bundled." >&2
    echo "On Arch/CachyOS: sudo pacman -S tk" >&2
    echo "On Debian/Ubuntu: sudo apt install python3-tk" >&2
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

echo
echo "Linux desktop bundle created:"
echo "  $PROJECT_DIR/dist/ProAim/ProAim"
echo "Install it in your application menu with:"
echo "  $PROJECT_DIR/dist/ProAim/install.sh"
