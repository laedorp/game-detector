#!/usr/bin/env python3
"""Build direct-head player crops from player/head YOLO data."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.head_detector import plan_head_crop, prepare_head_input  # noqa: E402


SPLITS = ("train", "val", "test")
DEFAULT_CROP_SIZE = 320
DEFAULT_CROP_SCALE = 2.0
DEFAULT_MIN_CROP_SIDE = 64


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_labels(path: Path, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"malformed label {path}:{line_number}")
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = (
            float(value) for value in fields[1:]
        )
        if not all(
            math.isfinite(value)
            for value in (center_x, center_y, box_width, box_height)
        ):
            raise ValueError(f"non-finite label {path}:{line_number}")
        result.append(
            (
                class_id,
                (
                    (center_x - box_width * 0.5) * width,
                    (center_y - box_height * 0.5) * height,
                    (center_x + box_width * 0.5) * width,
                    (center_y + box_height * 0.5) * height,
                ),
            )
        )
    return result


def _center_in(box: tuple[float, ...], container: tuple[float, ...]) -> bool:
    center_x = (box[0] + box[2]) * 0.5
    center_y = (box[1] + box[3]) * 0.5
    return bool(
        container[0] <= center_x <= container[2]
        and container[1] <= center_y <= container[3]
    )


def _map_box(box: tuple[float, ...], transform) -> tuple[float, float, float, float] | None:
    x1 = (box[0] - transform.crop_x) * transform.scale + transform.pad_left
    y1 = (box[1] - transform.crop_y) * transform.scale + transform.pad_top
    x2 = (box[2] - transform.crop_x) * transform.scale + transform.pad_left
    y2 = (box[3] - transform.crop_y) * transform.scale + transform.pad_top
    x1 = min(max(x1, 0.0), float(transform.model_width))
    y1 = min(max(y1, 0.0), float(transform.model_height))
    x2 = min(max(x2, 0.0), float(transform.model_width))
    y2 = min(max(y2, 0.0), float(transform.model_height))
    if x2 <= x1 or y2 <= y1:
        return None
    width = x2 - x1
    height = y2 - y1
    return (
        ((x1 + x2) * 0.5) / transform.model_width,
        ((y1 + y2) * 0.5) / transform.model_height,
        width / transform.model_width,
        height / transform.model_height,
    )


def build_crop_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    crop_size: int = DEFAULT_CROP_SIZE,
    crop_scale: float = DEFAULT_CROP_SCALE,
    min_crop_side: int = DEFAULT_MIN_CROP_SIDE,
) -> dict[str, Any]:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if crop_size <= 0 or crop_size % 32 != 0:
        raise ValueError("crop_size must be a positive multiple of 32")
    if not math.isfinite(float(crop_scale)) or float(crop_scale) < 1.0:
        raise ValueError("crop_scale must be finite and at least 1")
    if min_crop_side <= 0:
        raise ValueError("min_crop_side must be positive")
    if not (source / "dataset.yaml").is_file():
        raise ValueError(f"source dataset YAML is missing: {source}")
    if output.exists():
        raise ValueError(f"refusing to overwrite crop dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    split_counts: Counter[str] = Counter()
    positive_crops = 0
    negative_crops = 0
    try:
        for split in SPLITS:
            source_images = source / "images" / split
            source_labels = source / "labels" / split
            if not source_images.is_dir() or not source_labels.is_dir():
                raise ValueError(f"source dataset split is missing: {split}")
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)
            image_paths = sorted(
                path
                for path in source_images.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for image_path in image_paths:
                frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"cannot decode source image: {image_path}")
                height, width = frame.shape[:2]
                label_path = source_labels / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    raise ValueError(f"source label is missing: {label_path}")
                labels = _read_labels(label_path, width, height)
                players = [box for class_id, box in labels if class_id == 0]
                heads = [box for class_id, box in labels if class_id == 1]
                for player_index, player in enumerate(players):
                    transform = plan_head_crop(
                        frame.shape,
                        player,
                        crop_scale=float(crop_scale),
                        min_crop_side=min_crop_side,
                        model_size=(crop_size, crop_size),
                    )
                    prepared = prepare_head_input(frame, transform)
                    rgb = prepared.tensor[0].transpose(1, 2, 0)
                    crop = np.rint(rgb[:, :, ::-1] * 255.0).astype(np.uint8)
                    stem = f"{image_path.stem}--player-{player_index:02d}"
                    output_image = temporary / "images" / split / f"{stem}.jpg"
                    if not cv2.imwrite(
                        str(output_image),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    ):
                        raise RuntimeError(f"could not write crop image: {output_image}")
                    output_lines = []
                    mapped_player = _map_box(player, transform)
                    if mapped_player is None:
                        raise ValueError("selected player vanished from its runtime crop")
                    output_lines.append(
                        "0 " + " ".join(f"{value:.8f}" for value in mapped_player)
                    )
                    matching_heads = [head for head in heads if _center_in(head, player)]
                    for head in matching_heads:
                        mapped_head = _map_box(head, transform)
                        if mapped_head is not None:
                            output_lines.append(
                                "1 "
                                + " ".join(f"{value:.8f}" for value in mapped_head)
                            )
                    if len(output_lines) > 1:
                        positive_crops += 1
                    else:
                        negative_crops += 1
                    (temporary / "labels" / split / f"{stem}.txt").write_text(
                        "\n".join(output_lines) + "\n",
                        encoding="utf-8",
                    )
                    split_counts[split] += 1
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
        report = {
            "schema": "proaim.direct_head_crop_dataset",
            "schema_version": 1,
            "source": str(source),
            "source_dataset_yaml_sha256": _sha256_file(source / "dataset.yaml"),
            "crop_preprocessor": "detection.head_detector.prepare_head_input",
            "crop_size": crop_size,
            "crop_scale": float(crop_scale),
            "min_crop_side": int(min_crop_side),
            "crops": sum(split_counts.values()),
            "positive_head_crops": positive_crops,
            "negative_head_crops": negative_crops,
            "splits": {split: split_counts[split] for split in SPLITS},
        }
        (temporary / "DATASET-MANIFEST.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crop-size", type=_positive_int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--crop-scale", type=float, default=DEFAULT_CROP_SCALE)
    parser.add_argument("--min-crop-side", type=_positive_int, default=DEFAULT_MIN_CROP_SIDE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_crop_dataset(
        args.source,
        args.output,
        crop_size=args.crop_size,
        crop_scale=args.crop_scale,
        min_crop_side=args.min_crop_side,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())