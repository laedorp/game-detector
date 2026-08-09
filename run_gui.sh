#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_executable="$script_dir/.venv/bin/python"

if [[ ! -x "$python_executable" ]]; then
    python_executable="python3"
fi

exec "$python_executable" "$script_dir/app.py" --gui
