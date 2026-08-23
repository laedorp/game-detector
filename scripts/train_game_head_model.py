#!/usr/bin/env python3
"""Train and export a two-class game player/head detector on local data."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "valorant_player_head_v1" / "dataset.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "game_head"
MODEL_NAME_RE = re.compile(r"^yolo\d+[a-z0-9_.-]*\.pt$", flags=re.IGNORECASE)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="yolo26n_player_head")
    parser.add_argument("--epochs", type=_positive_int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=_positive_int, default=16)
    parser.add_argument("--imgsz", type=_positive_int, default=320)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def train(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    data = args.data.expanduser().resolve()
    requested_weights = str(args.weights)
    weights_path = Path(requested_weights).expanduser()
    if weights_path.is_file() and not weights_path.is_symlink():
        weights_for_training = str(weights_path.resolve())
        base_weights_sha256 = _sha256_file(weights_path.resolve())
    elif MODEL_NAME_RE.fullmatch(requested_weights):
        # Let Ultralytics resolve/download canonical upstream weights when a
        # model name like yolo11l.pt is supplied.
        weights_for_training = requested_weights
        base_weights_sha256 = None
    else:
        raise ValueError(
            "base weights are missing or unsafe; provide a local .pt file or "
            "an upstream model name like yolo11l.pt"
        )
    project = args.project.expanduser().resolve()
    if not data.is_file() or data.is_symlink():
        raise ValueError(f"dataset YAML is missing or unsafe: {data}")
    if not args.name or Path(args.name).name != args.name:
        raise ValueError("training name must be a plain directory name")
    if args.imgsz % 32 != 0:
        raise ValueError("--imgsz must be a positive multiple of 32")
    run_name = f"{args.name}_{args.imgsz}" + ("_smoke" if args.smoke_test else "")
    run_dir = project / run_name
    if run_dir.exists():
        raise ValueError(f"refusing to overwrite training run: {run_dir}")
    project.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project / ".ultralytics"))

    import torch
    import ultralytics
    from ultralytics import YOLO

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif args.allow_cpu:
        args.device = "cpu"
        device_name = "cpu"
    else:
        raise RuntimeError(
            "ROCm PyTorch did not expose the AMD GPU. Use --allow-cpu to run "
            "a slower CPU training fallback."
        )
    model = YOLO(weights_for_training)
    epochs = 1 if args.smoke_test else args.epochs
    results = model.train(
        data=str(data),
        epochs=epochs,
        patience=0 if args.smoke_test else args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=max(0, args.workers),
        project=str(project),
        name=run_name,
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        seed=0,
        deterministic=True,
        cache="disk",
        amp=True,
        close_mosaic=min(10, max(0, epochs // 4)),
        plots=not args.smoke_test,
        verbose=True,
    )
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"training did not produce best weights: {best}")
    trained = YOLO(str(best))
    test_metrics = None
    if not args.smoke_test:
        validation = trained.val(
            data=str(data),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=max(0, args.workers),
            project=str(project),
            name=run_name + "_test",
            plots=True,
        )
        test_metrics = {
            "map50": float(validation.box.map50),
            "map50_95": float(validation.box.map),
            "precision": float(validation.box.mp),
            "recall": float(validation.box.mr),
        }
    exported = Path(
        trained.export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=False,
            simplify=True,
            opset=19,
            batch=1,
        )
    ).resolve()
    if not exported.is_file():
        raise RuntimeError("ONNX export did not produce a file")
    manifest = {
        "schema": "proaim.game_head_training",
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "device": device_name,
        "data": str(data),
        "data_sha256": _sha256_file(data),
        "base_weights": weights_for_training,
        "base_weights_sha256": base_weights_sha256,
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "best_weights": str(best),
        "best_weights_sha256": _sha256_file(best),
        "onnx": str(exported),
        "onnx_sha256": _sha256_file(exported),
        "test_metrics": test_metrics,
        "results_type": type(results).__name__,
    }
    manifest_path = run_dir / "PROAIM-TRAINING-MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    manifest = train(argv)
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())