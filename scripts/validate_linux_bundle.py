#!/usr/bin/env python3
"""Validate native dependency closure and glibc floor of a Linux bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


GLIBC_PATTERN = re.compile(rb"GLIBC_(\d+)\.(\d+)")


def _version(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("expected MAJOR.MINOR, such as 2.35")
    return int(match.group(1)), int(match.group(2))


def _elf_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            with path.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    files.append(path)
        except OSError:
            continue
    return files


def validate_bundle(
    root: Path,
    *,
    max_glibc: tuple[int, int] | None = None,
    allowed_missing: frozenset[str] = frozenset(),
) -> tuple[int, tuple[int, int]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"bundle directory not found: {root}")
    elf_files = _elf_files(root)
    if not elf_files:
        raise RuntimeError(f"no ELF files found in bundle: {root}")

    missing: list[str] = []
    highest = (0, 0)
    for path in elf_files:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"cannot read {path}: {exc}") from exc
        versions = [
            (int(major), int(minor))
            for major, minor in GLIBC_PATTERN.findall(payload)
        ]
        if versions:
            highest = max(highest, max(versions))

        completed = subprocess.run(
            ["ldd", str(path)],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        output = completed.stdout + completed.stderr
        for line in output.splitlines():
            if "=> not found" not in line:
                continue
            library = line.split("=>", 1)[0].strip()
            if library not in allowed_missing:
                missing.append(f"{path.relative_to(root)}: {library}")

    if missing:
        raise RuntimeError(
            "unresolved native bundle dependencies:\n  - " + "\n  - ".join(missing)
        )
    if max_glibc is not None and highest > max_glibc:
        raise RuntimeError(
            f"bundle requires GLIBC_{highest[0]}.{highest[1]}, above allowed "
            f"GLIBC_{max_glibc[0]}.{max_glibc[1]}"
        )
    return len(elf_files), highest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--max-glibc", type=_version)
    parser.add_argument(
        "--allow-missing",
        action="append",
        default=[],
        metavar="LIBRARY",
        help="External driver library allowed to be absent (repeatable).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        count, highest = validate_bundle(
            args.bundle,
            max_glibc=args.max_glibc,
            allowed_missing=frozenset(args.allow_missing),
        )
    except RuntimeError as exc:
        print(f"Linux bundle validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"Linux bundle native validation passed: {count} ELF files; "
        f"highest GLIBC_{highest[0]}.{highest[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
