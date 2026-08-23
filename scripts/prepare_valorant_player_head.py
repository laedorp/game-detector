#!/usr/bin/env python3
"""Prepare the licensed Valorant enemy/head archive as two-class YOLO data."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Sequence
import zipfile


EXPECTED_ARCHIVE_SHA256 = (
    "e662f25fd39dcee317bd00db6f40acdc6c5e3e2677d142816535bd0a2ab543b4"
)
EXPECTED_ARCHIVE_SIZE = 125_978_971
SOURCE_URL = (
    "https://huggingface.co/datasets/Dasun01/"
    "Valorant-Object-Detection-Dataset"
)
SOURCE_LICENSE = "CC BY 4.0"
ARCHIVE_PREFIX = PurePosixPath(
    "Valorant object detection image dataset/labeled data"
)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
SPLITS = ("train", "val", "test")
MAX_ENTRY_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_FRAME_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > 100_000:
        raise ValueError("dataset archive contains too many entries")
    total = 0
    seen: set[str] = set()
    for member in members:
        path = PurePosixPath(member.filename)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or "\\" in member.filename
        ):
            raise ValueError(f"unsafe dataset archive path: {member.filename}")
        if member.filename in seen:
            raise ValueError(f"duplicate dataset archive path: {member.filename}")
        seen.add(member.filename)
        mode = member.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"unsafe dataset archive entry: {member.filename}")
        if member.flag_bits & 0x1:
            raise ValueError("encrypted dataset archives are not supported")
        if member.file_size > MAX_ENTRY_BYTES:
            raise ValueError(f"dataset archive entry is too large: {member.filename}")
        total += member.file_size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("dataset archive expands beyond the safety limit")
    return members


def _source_group(stem: str) -> str:
    match = _FRAME_PATTERN.search(stem)
    if match is None:
        return stem.split(".rf.", 1)[0]
    return f"frame-window-{int(match.group(1)) // 300:06d}"


def _group_splits(groups: set[str]) -> dict[str, str]:
    ordered = sorted(groups)
    if len(ordered) < 3:
        raise ValueError("dataset needs at least three temporal source groups")
    result = {}
    for index, group in enumerate(ordered):
        residue = index % 10
        result[group] = "test" if residue == 0 else "val" if residue == 1 else "train"
    if set(result.values()) != set(SPLITS):
        raise ValueError("temporal grouping did not produce all dataset splits")
    return result


def _remap_label(payload: bytes) -> tuple[str, Counter[str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("dataset label is not UTF-8") from exc
    output = []
    counts: Counter[str] = Counter()
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(f"malformed YOLO label line {line_number}")
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"malformed YOLO label line {line_number}") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("YOLO labels must contain finite values")
        center_x, center_y, width, height = values
        if not (
            0.0 <= center_x <= 1.0
            and 0.0 <= center_y <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError("YOLO label coordinates are outside [0,1]")
        if class_id == 1:
            output.append("0 " + " ".join(fields[1:]))
            counts["player"] += 1
        elif class_id == 2:
            output.append("1 " + " ".join(fields[1:]))
            counts["head"] += 1
    return ("\n".join(output) + ("\n" if output else ""), counts)


def prepare_dataset(
    archive_path: str | Path,
    output_path: str | Path,
    *,
    expected_sha256: str = EXPECTED_ARCHIVE_SHA256,
) -> dict[str, Any]:
    archive_file = Path(archive_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not archive_file.is_file() or archive_file.is_symlink():
        raise ValueError(f"dataset archive is missing or unsafe: {archive_file}")
    actual_sha256 = _sha256_file(archive_file)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"dataset archive SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    if output.exists():
        raise ValueError(f"refusing to overwrite dataset output: {output}")

    with zipfile.ZipFile(archive_file) as archive:
        members = _safe_members(archive)
        yaml_members = [
            member
            for member in members
            if PurePosixPath(member.filename).name == "data.yaml"
        ]
        if len(yaml_members) != 1:
            raise ValueError("dataset archive must contain exactly one data.yaml")
        source_yaml = archive.read(yaml_members[0]).decode("utf-8", "strict")
        for expected_class in ("crosshair", "enemy", "enemyhead", "teammate"):
            if expected_class not in source_yaml:
                raise ValueError(
                    f"dataset source class contract is missing {expected_class!r}"
                )

        image_members: dict[str, zipfile.ZipInfo] = {}
        label_members: dict[str, zipfile.ZipInfo] = {}
        for member in members:
            path = PurePosixPath(member.filename)
            if member.is_dir() or len(path.parts) < len(ARCHIVE_PREFIX.parts) + 2:
                continue
            if path.parts[: len(ARCHIVE_PREFIX.parts)] != ARCHIVE_PREFIX.parts:
                continue
            kind = path.parts[len(ARCHIVE_PREFIX.parts)]
            if kind == "images" and path.suffix.lower() in IMAGE_SUFFIXES:
                if path.stem in image_members:
                    raise ValueError(f"duplicate dataset image stem: {path.stem}")
                image_members[path.stem] = member
            elif kind == "labels" and path.suffix.lower() == ".txt":
                if path.stem in label_members:
                    raise ValueError(f"duplicate dataset label stem: {path.stem}")
                label_members[path.stem] = member
        if not image_members or set(image_members) != set(label_members):
            raise ValueError("dataset images and labels are not one-to-one")
        groups = {_source_group(stem) for stem in image_members}
        group_splits = _group_splits(groups)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
        )
        try:
            for split in SPLITS:
                (temporary / "images" / split).mkdir(parents=True)
                (temporary / "labels" / split).mkdir(parents=True)
            split_counts: Counter[str] = Counter()
            annotation_counts: Counter[str] = Counter()
            for stem in sorted(image_members):
                split = group_splits[_source_group(stem)]
                image_member = image_members[stem]
                image_name = PurePosixPath(image_member.filename).name
                image_payload = archive.read(image_member)
                label_text, counts = _remap_label(archive.read(label_members[stem]))
                (temporary / "images" / split / image_name).write_bytes(image_payload)
                (temporary / "labels" / split / f"{stem}.txt").write_text(
                    label_text,
                    encoding="utf-8",
                )
                split_counts[split] += 1
                annotation_counts.update(counts)

            final_root = output.as_posix()
            (temporary / "dataset.yaml").write_text(
                f"path: {final_root}\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "names:\n"
                "  0: player\n"
                "  1: head\n",
                encoding="utf-8",
            )
            (temporary / "ATTRIBUTION.md").write_text(
                "# Valorant player/head dataset attribution\n\n"
                "Derived from **Valorant Object Detection Dataset** by Dasun01, "
                f"licensed [{SOURCE_LICENSE}](https://creativecommons.org/licenses/by/4.0/).\n\n"
                f"Source: {SOURCE_URL}\n\n"
                f"Source archive SHA-256: `{actual_sha256}`\n\n"
                "Class mapping: `enemy` to `player`; `enemyhead` to `head`. "
                "Crosshair and teammate annotations are excluded.\n",
                encoding="utf-8",
            )
            report = {
                "schema": "proaim.valorant_player_head.dataset",
                "schema_version": 1,
                "source_url": SOURCE_URL,
                "source_license": SOURCE_LICENSE,
                "source_archive_sha256": actual_sha256,
                "source_archive_size": archive_file.stat().st_size,
                "images": len(image_members),
                "annotations": {
                    "player": annotation_counts["player"],
                    "head": annotation_counts["head"],
                },
                "splits": {split: split_counts[split] for split in SPLITS},
                "temporal_groups": len(groups),
            }
            (temporary / "DATASET-MANIFEST.json").write_text(
                json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, output)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-sha256", default=EXPECTED_ARCHIVE_SHA256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = prepare_dataset(
        args.archive,
        args.output,
        expected_sha256=args.expected_sha256,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())