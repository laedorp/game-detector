#!/usr/bin/env python3
"""Convert Elysium Roboflow COCO into two-class YOLO player/head data.

The source dataset provides only enemy-head boxes. This converter keeps those
as class 1 (head) and synthesizes conservative class 0 (player) boxes around
each head so the direct-head runtime model contract remains two-class.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


SPLIT_MAP = (("train", "train"), ("valid", "val"), ("test", "test"))


def _clip_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    return f"{cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def _synthetic_player_from_head(
    head_box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    hx1, hy1, hx2, hy2 = head_box
    hw = hx2 - hx1
    hh = hy2 - hy1
    if hw <= 0.0 or hh <= 0.0:
        return None
    hcx = (hx1 + hx2) * 0.5
    hcy = (hy1 + hy2) * 0.5

    # Build a conservative torso-inclusive proxy from a head box.
    player_w = max(hw * 4.0, hh * 2.0)
    player_h = max(hh * 7.0, hw * 3.5)
    player_cx = hcx
    player_cy = hcy + hh * 2.0
    px1 = player_cx - player_w * 0.5
    py1 = player_cy - player_h * 0.5
    px2 = player_cx + player_w * 0.5
    py2 = player_cy + player_h * 0.5
    return _clip_box(px1, py1, px2, py2, width, height)


def convert_dataset(source: Path, output: Path) -> dict[str, Any]:
    if not source.is_dir():
        raise ValueError(f"source directory is missing: {source}")
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    report: Counter[str] = Counter()
    try:
        for source_split, target_split in SPLIT_MAP:
            split_dir = source / source_split
            annotation_path = split_dir / "_annotations.coco.json"
            if not annotation_path.is_file():
                raise ValueError(f"missing COCO annotations: {annotation_path}")
            document = json.loads(annotation_path.read_text(encoding="utf-8"))

            categories = {
                int(item["id"]): str(item["name"]).strip().casefold()
                for item in document.get("categories", [])
            }
            head_category_ids = {
                category_id
                for category_id, name in categories.items()
                if "head" in name
            }
            if not head_category_ids:
                raise ValueError(
                    f"no head-like categories found in {annotation_path}"
                )

            images_by_id: dict[int, dict[str, Any]] = {
                int(item["id"]): item for item in document.get("images", [])
            }
            head_boxes_by_image: dict[int, list[tuple[float, float, float, float]]] = {}
            for annotation in document.get("annotations", []):
                category_id = int(annotation.get("category_id", -1))
                if category_id not in head_category_ids:
                    continue
                image_id = int(annotation.get("image_id", -1))
                if image_id not in images_by_id:
                    continue
                bbox = annotation.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                left, top, box_width, box_height = (float(value) for value in bbox)
                if not all(math.isfinite(value) for value in (left, top, box_width, box_height)):
                    continue
                if box_width <= 0.0 or box_height <= 0.0:
                    continue
                head_boxes_by_image.setdefault(image_id, []).append(
                    (left, top, left + box_width, top + box_height)
                )

            image_output_dir = temporary / "images" / target_split
            label_output_dir = temporary / "labels" / target_split
            image_output_dir.mkdir(parents=True, exist_ok=True)
            label_output_dir.mkdir(parents=True, exist_ok=True)

            for image_id, image in images_by_id.items():
                file_name = str(image.get("file_name", "")).strip()
                if not file_name:
                    continue
                width = int(image.get("width", 0))
                height = int(image.get("height", 0))
                if width <= 0 or height <= 0:
                    continue
                source_image_path = split_dir / file_name
                if not source_image_path.is_file():
                    continue
                clipped_heads = []
                for box in head_boxes_by_image.get(image_id, []):
                    clipped = _clip_box(*box, width, height)
                    if clipped is not None:
                        clipped_heads.append(clipped)
                if not clipped_heads:
                    continue

                stem = Path(file_name).stem
                destination_image = image_output_dir / source_image_path.name
                shutil.copy2(source_image_path, destination_image)
                report[f"{target_split}_images"] += 1

                label_lines: list[str] = []
                for head_box in clipped_heads:
                    player_box = _synthetic_player_from_head(head_box, width, height)
                    if player_box is None:
                        continue
                    label_lines.append("0 " + _to_yolo(player_box, width, height))
                    label_lines.append("1 " + _to_yolo(head_box, width, height))
                    report["synthetic_player_boxes"] += 1
                    report["head_boxes"] += 1

                if not label_lines:
                    continue
                (label_output_dir / f"{stem}.txt").write_text(
                    "\n".join(label_lines) + "\n",
                    encoding="utf-8",
                )

        dataset_yaml = (
            f"path: {output.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: player\n"
            "  1: head\n"
        )
        (temporary / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
        (temporary / "DATASET-MANIFEST.json").write_text(
            json.dumps(
                {
                    "schema": "proaim.elysium_enemy_head_dataset",
                    "schema_version": 1,
                    "source": str(source),
                    "output": str(output),
                    "report": dict(report),
                    "notes": (
                        "player boxes are synthetic proxies inferred from enemy-head "
                        "annotations to satisfy two-class direct-head training"
                    ),
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
    return dict(report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = convert_dataset(source, output)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())