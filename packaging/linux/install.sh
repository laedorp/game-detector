#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/env python3 "$BUNDLE_DIR/setup/install_linux_desktop.py" \
    --bundle "$BUNDLE_DIR" "$@"
