#!/usr/bin/env python3
"""Evaluate the pinned direct-head runtime against annotated player/head data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns
from typing import Any, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.head_detector import (  # noqa: E402
    DEFAULT_HEAD_CONFIDENCE,
    HEAD_INPUT_HEIGHT,
    HEAD_INPUT_WIDTH,
    HEAD_OUTPUT_ATTRIBUTES,
    HEAD_OUTPUT_CANDIDATES,
    associate_head_to_player_outcome,
    decode_head_output,
    plan_head_crop,
    prepare_head_input,
    runtime_head_model_spec,
    verify_pinned_head_model,
)
from detection.head_worker import (  # noqa: E402
    MIGRAPHX_PROVIDER,
    OnnxModelContract,
    OnnxTensorContract,
    StrictProviderOnnxSession,
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _labels(path: Path, width: int, height: int) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
    players = []
    heads = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"malformed label file: {path}")
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = (
            float(value) for value in fields[1:]
        )
        box = (
            (center_x - box_width * 0.5) * width,
            (center_y - box_height * 0.5) * height,
            (center_x + box_width * 0.5) * width,
            (center_y + box_height * 0.5) * height,
        )
        if class_id == 0:
            players.append(box)
        elif class_id == 1:
            heads.append(box)
    return players, heads


def _matching_head(
    player: tuple[float, ...],
    heads: list[tuple[float, ...]],
) -> tuple[float, ...] | None:
    matches = []
    player_center_x = (player[0] + player[2]) * 0.5
    for head in heads:
        center_x = (head[0] + head[2]) * 0.5
        center_y = (head[1] + head[3]) * 0.5
        if player[0] <= center_x <= player[2] and player[1] <= center_y <= player[3]:
            matches.append((abs(center_x - player_center_x), head))
    return min(matches, default=(0.0, None), key=lambda item: item[0])[1]


def _point_inside(point: tuple[float, float], box: tuple[float, ...]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def evaluate(
    dataset_root: str | Path,
    *,
    model_path: str | Path | None = None,
    split: str = "test",
    provider: str = MIGRAPHX_PROVIDER,
    confidence: float = DEFAULT_HEAD_CONFIDENCE,
    maximum_images: int | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"dataset split is missing: {split}")
    if model_path is None:
        model_spec = runtime_head_model_spec()
        model = model_spec.path
        input_height = model_spec.input_height
        input_width = model_spec.input_width
        output_candidates = model_spec.output_candidates
        if confidence == DEFAULT_HEAD_CONFIDENCE:
            confidence = model_spec.confidence_threshold
    else:
        model = Path(model_path).expanduser().resolve()
        input_height = HEAD_INPUT_HEIGHT
        input_width = HEAD_INPUT_WIDTH
        output_candidates = HEAD_OUTPUT_CANDIDATES
    if not model.is_file() or model.is_symlink():
        raise ValueError(f"head model is missing or unsafe: {model}")
    contract = OnnxModelContract(
        input=OnnxTensorContract(
            "images",
            (1, 3, input_height, input_width),
        ),
        output=OnnxTensorContract(
            "output0",
            (1, HEAD_OUTPUT_ATTRIBUTES, output_candidates),
        ),
    )
    session = StrictProviderOnnxSession(
        model,
        contract,
        provider=provider,
    )
    session.infer(np.zeros(contract.input.shape, dtype=np.float32))

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if maximum_images is not None:
        image_paths = image_paths[:maximum_images]
    reasons: Counter[str] = Counter()
    inference_ms: list[float] = []
    positive_players = 0
    negative_players = 0
    accepted = 0
    correct = 0
    false_positive = 0
    unpaired_heads = 0
    evaluated_images = 0
    for image_path in image_paths:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"could not decode dataset image: {image_path}")
        height, width = frame.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"dataset label is missing: {label_path}")
        players, heads = _labels(label_path, width, height)
        paired_head_ids: set[int] = set()
        for player in players:
            ground_truth = _matching_head(player, heads)
            if ground_truth is None:
                negative_players += 1
            else:
                positive_players += 1
                paired_head_ids.add(id(ground_truth))
            transform = plan_head_crop(
                frame.shape,
                player,
                model_size=(input_height, input_width),
            )
            prepared = prepare_head_input(frame, transform)
            started = perf_counter_ns()
            output = session.infer(prepared.tensor)
            inference_ms.append((perf_counter_ns() - started) / 1e6)
            candidates = decode_head_output(
                output,
                transform,
                confidence=confidence,
            )
            outcome = associate_head_to_player_outcome(
                candidates,
                player,
                source_timestamp_ns=evaluated_images,
            )
            reasons[outcome.reason.value] += 1
            localization = outcome.localization
            if localization is not None:
                accepted += 1
                if ground_truth is not None and _point_inside(
                    localization.point,
                    ground_truth,
                ):
                    correct += 1
                else:
                    false_positive += 1
        unpaired_heads += max(0, len(heads) - len(paired_head_ids))
        evaluated_images += 1

    recall = correct / positive_players if positive_players else 0.0
    precision = correct / accepted if accepted else 0.0
    return {
        "schema": "proaim.direct_head_runtime_evaluation",
        "schema_version": 1,
        "dataset": str(root),
        "split": split,
        "provider": session.info.provider,
        "model": str(model),
        "confidence": confidence,
        "images": evaluated_images,
        "positive_players": positive_players,
        "negative_players": negative_players,
        "unpaired_heads": unpaired_heads,
        "accepted_localizations": accepted,
        "correct_localizations": correct,
        "false_positive_localizations": false_positive,
        "recall": recall,
        "precision": precision,
        "reasons": dict(sorted(reasons.items())),
        "inference_ms": {
            "mean": statistics.fmean(inference_ms) if inference_ms else 0.0,
            "p50": _percentile(inference_ms, 0.50),
            "p95": _percentile(inference_ms, 0.95),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--provider", default=MIGRAPHX_PROVIDER)
    parser.add_argument("--confidence", type=float, default=DEFAULT_HEAD_CONFIDENCE)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.dataset,
        model_path=args.model,
        split=args.split,
        provider=args.provider,
        confidence=args.confidence,
        maximum_images=args.max_images,
    )
    encoded = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())