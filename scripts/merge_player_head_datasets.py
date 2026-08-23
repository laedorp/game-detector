#!/usr/bin/env python3
"""Merge multiple two-class YOLO player/head datasets into one dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile


SPLITS = ("train", "val", "test")


def _dataset_path(dataset_root: Path) -> Path:
    yaml_path = dataset_root / "dataset.yaml"
    if not yaml_path.is_file():
        raise ValueError(f"dataset.yaml missing: {dataset_root}")
    return yaml_path


def _copy_dataset(
    dataset_root: Path,
    merged_root: Path,
    prefix: str,
    counters: dict[str, int],
) -> None:
    for split in SPLITS:
        source_images = dataset_root / "images" / split
        source_labels = dataset_root / "labels" / split
        if not source_images.is_dir() or not source_labels.is_dir():
            raise ValueError(f"missing split directories in {dataset_root}: {split}")
        destination_images = merged_root / "images" / split
        destination_labels = merged_root / "labels" / split
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)
        for label_path in sorted(source_labels.glob("*.txt")):
            image_candidates = [
                source_images / f"{label_path.stem}.jpg",
                source_images / f"{label_path.stem}.jpeg",
                source_images / f"{label_path.stem}.png",
            ]
            image_path = next((path for path in image_candidates if path.is_file()), None)
            if image_path is None:
                continue
            destination_stem = f"{prefix}__{label_path.stem}"
            destination_image = destination_images / f"{destination_stem}{image_path.suffix.lower()}"
            destination_label = destination_labels / f"{destination_stem}.txt"
            shutil.copy2(image_path, destination_image)
            shutil.copy2(label_path, destination_label)
            counters[f"{split}_samples"] = counters.get(f"{split}_samples", 0) + 1


def merge_datasets(inputs: list[Path], output: Path) -> dict[str, int]:
    if output.exists():
        raise ValueError(f"refusing to overwrite output dataset: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    counters: dict[str, int] = {}
    try:
        for index, dataset_root in enumerate(inputs):
            _dataset_path(dataset_root)
            _copy_dataset(
                dataset_root,
                temporary,
                prefix=f"d{index:02d}",
                counters=counters,
            )

        (temporary / "dataset.yaml").write_text(
            f"path: {output.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: player\n"
            "  1: head\n",
            encoding="utf-8",
        )
        (temporary / "DATASET-MANIFEST.json").write_text(
            json.dumps(
                {
                    "schema": "proaim.merged_player_head_dataset",
                    "schema_version": 1,
                    "inputs": [str(path) for path in inputs],
                    "output": str(output),
                    "counts": counters,
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return counters


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    inputs = [path.expanduser().resolve() for path in args.input]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = merge_datasets(inputs, output)
    print(json.dumps(counts, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())