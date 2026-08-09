"""Reproducibly fine-tune the local YOLO26n weights on prepared FORT-Cuh data."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path, PurePath
import platform
import re
import shlex
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "fort_cuh_player" / "fort_cuh_player.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "fort_cuh"
DEFAULT_NAME = "yolo26n_320"


class TrainingConfigurationError(ValueError):
    """Raised before Ultralytics is imported when a run is unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    data: Path
    weights: Path
    project: Path
    name: str
    epochs: int = 50
    patience: int = 10
    batch: int = 16
    imgsz: int = 320
    device: str = "cpu"
    workers: int = 4
    threads: int = 6
    cache: str = "disk"
    seed: int = 0
    smoke_test: bool = False
    run_test: bool = True

    @property
    def run_dir(self) -> Path:
        return self.project / self.name

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("data", "weights", "project"):
            values[key] = str(values[key])
        values["run_dir"] = str(self.run_dir)
        return values


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    run_dir: Path
    best_weights: Path
    test_run_dir: Path | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune local YOLO26n weights for one-class, 320x320 player "
            "detection. Training uses local weights/data and is CPU-only by default."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Existing local detect weights; remote model names are not accepted.",
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--epochs", type=_positive_int, default=50)
    parser.add_argument("--patience", type=_non_negative_int, default=10)
    parser.add_argument("--batch", type=_positive_int, default=16)
    parser.add_argument("--imgsz", type=_positive_int, default=320)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=_non_negative_int, default=4)
    parser.add_argument(
        "--threads",
        type=_positive_int,
        default=6,
        help=(
            "PyTorch/OpenMP CPU threads (default: 6 physical cores on this laptop). "
            "Applied before importing torch or Ultralytics."
        ),
    )
    parser.add_argument(
        "--cache",
        choices=("ram", "disk", "none"),
        default="disk",
        help="Image cache mode (default: disk, which preserves deterministic ordering).",
    )
    parser.add_argument("--seed", type=_non_negative_int, default=0)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help=(
            "Do not evaluate best.pt on the supplied non-independent test split "
            "after training."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one epoch on 2%% of training images, disable caching/plots, use "
            "a deterministic _smoke run name, and skip the full test evaluation."
        ),
    )
    return parser


def _safe_run_name(value: str) -> str:
    name = value.strip()
    if (
        not name
        or name in {".", ".."}
        or PurePath(name).name != name
        or "/" in name
        or "\\" in name
        or any(ord(character) < 32 for character in name)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name)
    ):
        raise TrainingConfigurationError(
            "run name must be a plain 1-80 character name using letters, digits, ., _, or -"
        )
    return name


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    name = _safe_run_name(str(args.name))
    smoke_test = bool(args.smoke_test)
    if smoke_test and not name.endswith("_smoke"):
        name = _safe_run_name(f"{name}_smoke")
    imgsz = int(args.imgsz)
    if imgsz % 32:
        raise TrainingConfigurationError("--imgsz must be divisible by 32")
    device = str(args.device).strip()
    if not device:
        raise TrainingConfigurationError("--device cannot be empty")
    return TrainingConfig(
        data=Path(args.data).expanduser().resolve(),
        weights=Path(args.weights).expanduser().resolve(),
        project=Path(args.project).expanduser().resolve(),
        name=name,
        epochs=1 if smoke_test else int(args.epochs),
        patience=0 if smoke_test else int(args.patience),
        batch=int(args.batch),
        imgsz=imgsz,
        device=device,
        workers=int(args.workers),
        threads=int(args.threads),
        cache="none" if smoke_test else str(args.cache),
        seed=int(args.seed),
        smoke_test=smoke_test,
        run_test=not bool(args.skip_test) and not smoke_test,
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_bundle(data_file: Path) -> Mapping[str, Any]:
    if not data_file.is_file():
        raise TrainingConfigurationError(f"prepared dataset YAML not found: {data_file}")
    dataset_root = data_file.parent
    labels_file = dataset_root / "labels.txt"
    manifest_file = dataset_root / "manifest.json"
    if not labels_file.is_file() or labels_file.read_text(encoding="utf-8") != "player\n":
        raise TrainingConfigurationError(
            f"prepared runtime labels must contain exactly 'player': {labels_file}"
        )
    if not manifest_file.is_file():
        raise TrainingConfigurationError(f"prepared dataset manifest not found: {manifest_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingConfigurationError(f"invalid prepared dataset manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TrainingConfigurationError("prepared dataset manifest root must be an object")
    conversion = manifest.get("conversion")
    if not isinstance(conversion, dict) or conversion.get("output_class_names") != ["player"]:
        raise TrainingConfigurationError("prepared dataset manifest is not one-class player data")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise TrainingConfigurationError("prepared dataset manifest has no split statistics")
    for split in ("train", "valid", "test"):
        stats = splits.get(split)
        if (
            not isinstance(stats, dict)
            or not isinstance(stats.get("retained_images"), int)
            or stats["retained_images"] <= 0
            or not isinstance(stats.get("retained_boxes"), int)
            or stats["retained_boxes"] <= 0
        ):
            raise TrainingConfigurationError(
                f"prepared dataset split is empty or invalid: {split}"
            )
        for kind in ("images", "labels"):
            directory = dataset_root / kind / split
            if not directory.is_dir():
                raise TrainingConfigurationError(
                    f"prepared dataset directory missing: {directory}"
                )
    yaml_text = data_file.read_text(encoding="utf-8")
    for required in ("train: images/train", "val: images/valid", "test: images/test", "0: player"):
        if required not in yaml_text:
            raise TrainingConfigurationError(
                f"prepared dataset YAML is missing required entry: {required}"
            )
    return manifest


def validate_training_config(config: TrainingConfig) -> Mapping[str, Any]:
    _safe_run_name(config.name)
    for value, description in (
        (config.epochs, "epochs"),
        (config.batch, "batch"),
        (config.imgsz, "image size"),
        (config.threads, "CPU threads"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TrainingConfigurationError(f"{description} must be a positive integer")
    for value, description in (
        (config.patience, "patience"),
        (config.workers, "workers"),
        (config.seed, "seed"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TrainingConfigurationError(f"{description} must be a non-negative integer")
    if config.imgsz % 32:
        raise TrainingConfigurationError("image size must be divisible by 32")
    if config.cache not in {"ram", "disk", "none"}:
        raise TrainingConfigurationError("cache must be 'ram', 'disk', or 'none'")
    if not config.device.strip():
        raise TrainingConfigurationError("device cannot be empty")

    manifest = validate_dataset_bundle(config.data)
    if not config.weights.is_file():
        raise TrainingConfigurationError(
            f"local pretrained weights not found: {config.weights}"
        )
    if config.weights.suffix.casefold() != ".pt":
        raise TrainingConfigurationError("pretrained weights must be a local .pt file")
    if config.run_dir.exists() or config.run_dir.is_symlink() or os.path.lexists(config.run_dir):
        raise TrainingConfigurationError(
            f"run directory already exists; refusing to reuse it: {config.run_dir}"
        )
    if config.run_test:
        test_run = config.run_dir / "test"
        if test_run.exists() or test_run.is_symlink() or os.path.lexists(test_run):
            raise TrainingConfigurationError(
                f"test run directory already exists; refusing to reuse it: {test_run}"
            )
    return manifest


def training_arguments(config: TrainingConfig) -> dict[str, Any]:
    """Return the explicit arguments passed to Ultralytics YOLO.train."""

    return {
        "data": str(config.data),
        "epochs": config.epochs,
        "patience": config.patience,
        "batch": config.batch,
        "imgsz": config.imgsz,
        "device": config.device,
        "workers": config.workers,
        "cache": False if config.cache == "none" else config.cache,
        "seed": config.seed,
        "deterministic": True,
        "amp": False,
        "optimizer": "auto",
        "pretrained": True,
        # The generated YAML already has one class. Explicitly keep single_cls
        # disabled because Ultralytics otherwise renames that class to "item".
        "single_cls": False,
        "close_mosaic": 10 if config.epochs > 10 else 0,
        "fraction": 0.02 if config.smoke_test else 1.0,
        "plots": not config.smoke_test,
        "save": True,
        "project": str(config.project),
        "name": config.name,
        "exist_ok": True,
        "verbose": True,
    }


def validation_arguments(config: TrainingConfig) -> dict[str, Any]:
    """Return arguments for the supplied test split, which is not source-independent."""

    return {
        "data": str(config.data),
        "split": "test",
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "workers": config.workers,
        "project": str(config.run_dir),
        "name": "test",
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }


def _configure_cpu_environment(threads: int) -> None:
    # Ultralytics sets OMP_NUM_THREADS=1 when it is absent. Set both knobs before
    # importing Ultralytics/torch so this CPU run can use the six physical cores.
    value = str(threads)
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value


def _load_yolo_class(threads: int) -> Callable[[str], Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainingConfigurationError(
            "Ultralytics is unavailable. Activate .venv-export and install "
            "requirements-export.txt."
        ) from exc

    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, min(2, threads)))
        except RuntimeError:
            # PyTorch permits setting inter-op threads only before parallel work
            # begins. OMP/MKL and intra-op threads remain explicitly configured.
            pass
    except ImportError as exc:
        raise TrainingConfigurationError(
            "PyTorch is unavailable in the export/training environment."
        ) from exc
    return YOLO


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _write_reproducibility_record(
    config: TrainingConfig,
    manifest: Mapping[str, Any],
    best_weights: Path,
) -> None:
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    leakage = manifest.get("leakage") if isinstance(manifest.get("leakage"), dict) else {}
    record = {
        "schema_version": 1,
        "training": config.serializable(),
        "training_arguments": training_arguments(config),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ultralytics": _package_version("ultralytics"),
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "openvino": _package_version("openvino"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "inputs": {
            "dataset_archive_sha256": source.get("archive_sha256"),
            "dataset_manifest_sha256": _sha256_file(config.data.parent / "manifest.json"),
            "initial_weights_sha256": _sha256_file(config.weights),
        },
        "output": {
            "best_weights": str(best_weights),
            "runtime_class_labels": ["player"],
            "expected_openvino_output_format": "auto",
            "deployment_inference_size": config.imgsz,
        },
        "evaluation_caveat": {
            "supplied_splits_are_source_independent": False,
            "metrics_may_be_optimistic": True,
            "cross_split_original_basename_groups": leakage.get(
                "cross_split_original_basename_groups"
            ),
            "cross_split_source_groups": leakage.get("cross_split_source_groups"),
            "cross_split_video_sequence_groups": leakage.get(
                "cross_split_video_sequence_groups"
            ),
        },
    }
    (config.run_dir / "reproducibility.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_split_overlap_warning(manifest: Mapping[str, Any]) -> None:
    leakage = manifest.get("leakage")
    leakage = leakage if isinstance(leakage, dict) else {}
    original_groups = leakage.get("cross_split_original_basename_groups", "unknown")
    source_groups = leakage.get("cross_split_source_groups")
    source_images = leakage.get("cross_split_source_group_images")
    video_groups = leakage.get("cross_split_video_sequence_groups")

    detail = f"{original_groups} repeated original-basename group(s)"
    if isinstance(source_groups, int):
        detail += f", {source_groups} conservative source group(s)"
    if isinstance(source_images, int):
        detail += f" spanning {source_images} image(s)"
    if isinstance(video_groups, int):
        detail += f", including {video_groups} explicit video sequence(s)"
    print(f"WARNING: Supplied train/validation/test sources overlap: {detail}.")
    print(
        "WARNING: Validation and supplied-test metrics are not independent and may "
        "be optimistic; use a separately captured evaluation set for final claims."
    )


def run_training(
    config: TrainingConfig,
    *,
    yolo_class: Callable[[str], Any] | None = None,
) -> TrainingOutcome:
    """Train once in an unused run directory and optionally evaluate best.pt."""

    manifest = validate_training_config(config)
    _configure_cpu_environment(config.threads)
    model_factory = yolo_class or _load_yolo_class(config.threads)
    model = model_factory(str(config.weights))
    if getattr(model, "task", None) != "detect":
        raise TrainingConfigurationError(
            f"unsupported pretrained task {getattr(model, 'task', None)!r}; expected 'detect'"
        )

    print(f"Training output: {config.run_dir}")
    print(
        f"CPU configuration: {config.threads} compute thread(s), "
        f"{config.workers} data worker(s)"
    )
    if config.smoke_test:
        print("Smoke mode: one epoch, 2% training fraction, no cache/plots/test evaluation")
    _print_split_overlap_warning(manifest)
    model.train(**training_arguments(config))

    trainer = getattr(model, "trainer", None)
    best_value = getattr(trainer, "best", None)
    if not best_value:
        raise RuntimeError("Ultralytics training completed without reporting best.pt")
    best_weights = Path(best_value).expanduser().resolve()
    if not best_weights.is_file():
        raise RuntimeError(f"Ultralytics best weights were not found: {best_weights}")

    test_run_dir: Path | None = None
    if config.run_test:
        test_run_dir = config.run_dir / "test"
        best_model = model_factory(str(best_weights))
        best_model.val(**validation_arguments(config))

    _write_reproducibility_record(config, manifest, best_weights)
    return TrainingOutcome(
        run_dir=config.run_dir,
        best_weights=best_weights,
        test_run_dir=test_run_dir,
    )


def _print_outcome(outcome: TrainingOutcome, config: TrainingConfig) -> None:
    print(f"Best weights: {outcome.best_weights}")
    if outcome.test_run_dir is not None:
        print(f"Supplied-split test report (not independent): {outcome.test_run_dir}")
    export_command = [
        str(PROJECT_ROOT / ".venv-export" / "bin" / "python"),
        str(PROJECT_ROOT / "scripts" / "export_model.py"),
        "--weights",
        str(outcome.best_weights),
        "--imgsz",
        str(config.imgsz),
        "--output",
        str(PROJECT_ROOT / "models" / "fort_player_openvino_model"),
        "--basename",
        "fort_player",
    ]
    print("Export when the run has been reviewed:")
    print(shlex.join(export_command))
    print(f"Runtime labels: {config.data.parent / 'labels.txt'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        outcome = run_training(config)
    except TrainingConfigurationError as exc:
        parser.error(str(exc))
    _print_outcome(outcome, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
