#!/usr/bin/env python3
"""Build conservative source-independent FORT-Cuh evaluation lists.

The supplied Roboflow splits contain frames and augmented copies from the same
underlying screenshots/video sequences.  ``prepare_fort_cuh.py`` records those
groups in ``manifest.json``.  This tool uses that record to create:

* a validation list with every source group seen in training removed; and
* a test list with every source group seen in training *or validation* removed.

The grouping is deliberately conservative but filename-based.  It is stronger
than using the supplied splits unchanged; it is not a substitute for footage
captured independently from the target game clone.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
EVAL_RULES: Mapping[str, tuple[str, ...]] = {
    "valid": ("train",),
    "test": ("train", "valid"),
}


class StrictEvaluationError(ValueError):
    """Raised when the prepared dataset cannot produce a trustworthy list."""


@dataclass(frozen=True, slots=True)
class SplitSummary:
    supplied_images: int
    supplied_boxes: int
    excluded_source_overlap_images: int
    excluded_source_overlap_boxes: int
    strict_images: int
    strict_boxes: int
    blocked_by_splits: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/fort_cuh_player"),
        help="Prepared dataset root containing manifest.json (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory for strict lists, YAML, and report JSON.",
    )
    return parser


def _load_manifest(dataset: Path) -> dict[str, Any]:
    manifest_file = dataset / "manifest.json"
    try:
        value = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StrictEvaluationError(f"cannot read prepared manifest {manifest_file}: {exc}") from exc
    if not isinstance(value, dict):
        raise StrictEvaluationError("prepared manifest root must be an object")
    conversion = value.get("conversion")
    if not isinstance(conversion, dict) or conversion.get("output_class_names") != ["player"]:
        raise StrictEvaluationError("prepared manifest is not one-class player data")
    leakage = value.get("leakage")
    if not isinstance(leakage, dict) or not isinstance(
        leakage.get("cross_split_source_group_details"), list
    ):
        raise StrictEvaluationError("prepared manifest has no source-group leakage details")
    return value


def _validated_group_files(manifest: Mapping[str, Any]) -> list[tuple[str, ...]]:
    details = manifest["leakage"]["cross_split_source_group_details"]
    groups: list[tuple[str, ...]] = []
    for index, detail in enumerate(details):
        if not isinstance(detail, dict) or not isinstance(detail.get("files"), list):
            raise StrictEvaluationError(f"invalid source-group detail at index {index}")
        files: list[str] = []
        for raw in detail["files"]:
            if not isinstance(raw, str):
                raise StrictEvaluationError(f"non-string source-group file at index {index}")
            path = PurePosixPath(raw)
            if (
                path.is_absolute()
                or len(path.parts) != 2
                or path.parts[0] not in {"train", "valid", "test"}
                or path.name != path.parts[1]
                or path.suffix.casefold() not in IMAGE_SUFFIXES
            ):
                raise StrictEvaluationError(f"unsafe source-group file path: {raw!r}")
            files.append(raw)
        if len(files) < 2 or len(files) != len(set(files)):
            raise StrictEvaluationError(f"invalid source-group membership at index {index}")
        groups.append(tuple(files))
    return groups


def _image_box_counts(dataset: Path, split: str) -> dict[str, int]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise StrictEvaluationError(f"prepared split directories are missing for {split}")

    counts: dict[str, int] = {}
    for image in sorted(image_dir.iterdir(), key=lambda path: path.name.casefold()):
        if not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        label = label_dir / f"{image.stem}.txt"
        try:
            lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError) as exc:
            raise StrictEvaluationError(f"cannot read label for {image}: {exc}") from exc
        if not lines or any(not line for line in lines):
            raise StrictEvaluationError(f"prepared image has an empty/invalid label: {image}")
        for line in lines:
            fields = line.split()
            if len(fields) != 5 or fields[0] != "0":
                raise StrictEvaluationError(f"unexpected one-class label in {label}: {line!r}")
            try:
                coordinates = tuple(float(value) for value in fields[1:])
            except ValueError as exc:
                raise StrictEvaluationError(f"non-numeric box in {label}: {line!r}") from exc
            if any(not 0.0 <= value <= 1.0 for value in coordinates) or any(
                value <= 0.0 for value in coordinates[2:]
            ):
                raise StrictEvaluationError(f"out-of-range box in {label}: {line!r}")
        counts[image.name] = len(lines)
    if not counts:
        raise StrictEvaluationError(f"prepared split contains no images: {split}")
    return counts


def build_strict_membership(
    dataset: Path,
) -> tuple[dict[str, tuple[Path, ...]], dict[str, SplitSummary], Mapping[str, Any]]:
    """Return strict image paths and summaries after validating the dataset."""

    dataset = dataset.expanduser().resolve()
    manifest = _load_manifest(dataset)
    groups = _validated_group_files(manifest)
    memberships: dict[str, tuple[Path, ...]] = {}
    summaries: dict[str, SplitSummary] = {}

    for split, blocked_splits in EVAL_RULES.items():
        counts = _image_box_counts(dataset, split)
        excluded: set[str] = set()
        for group in groups:
            if not any(path.partition("/")[0] in blocked_splits for path in group):
                continue
            excluded.update(
                path.partition("/")[2]
                for path in group
                if path.partition("/")[0] == split
            )
        unknown = excluded.difference(counts)
        if unknown:
            raise StrictEvaluationError(
                f"manifest references missing retained {split} image: {sorted(unknown)[0]}"
            )
        retained_names = tuple(sorted(set(counts).difference(excluded), key=str.casefold))
        if not retained_names:
            raise StrictEvaluationError(f"source-overlap filtering emptied the {split} split")
        memberships[split] = tuple(dataset / "images" / split / name for name in retained_names)
        summaries[split] = SplitSummary(
            supplied_images=len(counts),
            supplied_boxes=sum(counts.values()),
            excluded_source_overlap_images=len(excluded),
            excluded_source_overlap_boxes=sum(counts[name] for name in excluded),
            strict_images=len(retained_names),
            strict_boxes=sum(counts[name] for name in retained_names),
            blocked_by_splits=blocked_splits,
        )
    return memberships, summaries, manifest["leakage"].get("source_grouping_heuristic", {})


def write_strict_evaluation(dataset: Path, output: Path) -> dict[str, Any]:
    """Create strict list/YAML/report artifacts in a previously unused directory."""

    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink() or os.path.lexists(output):
        raise StrictEvaluationError(f"output already exists; refusing to overwrite it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    memberships, summaries, heuristic = build_strict_membership(dataset)
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(dataset),
        "method": (
            "filename-based conservative source groups from the prepared dataset manifest; "
            "not a substitute for independently captured evaluation footage"
        ),
        "source_grouping_heuristic": heuristic,
        "splits": {name: asdict(summary) for name, summary in summaries.items()},
    }

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for split, paths in memberships.items():
            (temporary / f"strict_{split}.txt").write_text(
                "".join(f"{path}\n" for path in paths), encoding="utf-8"
            )
        yaml_lines = [
            "# Generated by scripts/build_strict_fort_eval.py",
            f"path: {json.dumps(str(dataset))}",
            f"train: {json.dumps(str(dataset / 'images' / 'train'))}",
            f"val: {json.dumps(str(temporary / 'strict_valid.txt'))}",
            f"test: {json.dumps(str(temporary / 'strict_test.txt'))}",
            "names:",
            "  0: player",
            "",
        ]
        # The temporary directory name changes on rename, so point list entries at
        # their final absolute paths before publishing the directory atomically.
        yaml_lines[3] = f"val: {json.dumps(str(output / 'strict_valid.txt'))}"
        yaml_lines[4] = f"test: {json.dumps(str(output / 'strict_test.txt'))}"
        (temporary / "fort_cuh_strict_eval.yaml").write_text(
            "\n".join(yaml_lines), encoding="utf-8"
        )
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = write_strict_evaluation(args.dataset, args.output)
    except StrictEvaluationError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
