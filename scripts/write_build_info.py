#!/usr/bin/env python3
"""Write a deterministic identity record into a completed ProAim bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence


RUNTIME_VARIANTS = ("cpu", "cuda", "directml", "rocm")


def _git_value(root: Path, *arguments: str) -> str | None:
    """Return stripped git output, preserving an empty successful result."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def write_build_info(bundle: Path, runtime_variant: str, project_root: Path) -> Path:
    resolved = bundle.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {resolved}")
    normalized_variant = runtime_variant.strip().lower()
    if normalized_variant not in RUNTIME_VARIANTS:
        raise ValueError(
            f"unknown runtime variant {runtime_variant!r}; choose one of "
            + ", ".join(RUNTIME_VARIANTS)
        )
    executable = resolved / ("ProAim.exe" if (resolved / "ProAim.exe").is_file() else "ProAim")
    if not executable.is_file():
        raise FileNotFoundError(f"bundle executable not found: {executable}")
    project_root = project_root.expanduser().resolve()
    commit = _git_value(project_root, "rev-parse", "HEAD")
    if not commit:
        commit = os.environ.get("GITHUB_SHA", "").strip() or "unknown"
    commit_time = _git_value(project_root, "show", "-s", "--format=%cI", "HEAD")
    status = _git_value(project_root, "status", "--porcelain", "--untracked-files=normal")
    payload = {
        "application": "ProAim",
        "commit": commit,
        "commit_time": commit_time or "unknown",
        # ``null`` is intentionally different from a clean checkout: it means
        # the bundle was built without usable source-control metadata.
        "dirty": None if status is None else bool(status),
        "runtime_variant": normalized_variant,
        "schema": 1,
    }
    target = resolved / "BUILD-INFO.json"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--runtime-variant", required=True, choices=RUNTIME_VARIANTS)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    target = write_build_info(args.bundle, args.runtime_variant, args.project_root.resolve())
    print(f"Build identity written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
