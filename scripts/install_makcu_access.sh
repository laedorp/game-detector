#!/usr/bin/env bash
set -euo pipefail

if [[ "$EUID" -ne 0 ]]; then
    echo "Run this script with sudo." >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_RULE="$PROJECT_DIR/packaging/linux/70-game-detector-makcu.rules"
TARGET_RULE="/etc/udev/rules.d/70-game-detector-makcu.rules"

if [[ ! -f "$SOURCE_RULE" ]]; then
    echo "MAKCU access rule not found: $SOURCE_RULE" >&2
    exit 2
fi

install -Dm0644 "$SOURCE_RULE" "$TARGET_RULE"
udevadm control --reload-rules

echo "Installed $TARGET_RULE"
echo "Unplug and reconnect only the MAKCU cable attached to this laptop."