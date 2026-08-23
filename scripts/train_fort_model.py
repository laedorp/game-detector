"""Reproducibly fine-tune the local YOLO26n weights on prepared FORT-Cuh data."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path, PurePath
import platform
import re
import shlex
import sys
from typing import Any

try:
    from scripts.fort_dataset_contract import (
        DatasetContractError,
        verify_dataset_contract,
        verify_grouped_dataset_metadata,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from fort_dataset_contract import (
        DatasetContractError,
        verify_dataset_contract,
        verify_grouped_dataset_metadata,
    )


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
    resume_from: Path | None = None
    adopt_interrupted_run: bool = False

    @property
    def run_dir(self) -> Path:
        return self.project / self.name

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("data", "weights", "project", "resume_from"):
            if values[key] is None:
                continue
            values[key] = str(values[key])
        values["run_dir"] = str(self.run_dir)
        return values


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    run_dir: Path
    best_weights: Path
    test_run_dir: Path | None


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    """Verified state needed for a true Ultralytics optimizer/scheduler resume."""

    path: Path
    sha256: str
    completed_epoch: int
    initial_weights: Path
    initial_weights_sha256: str
    dataset_manifest_sha256: str
    dataset_content_sha256: str | None
    dataset_yaml_sha256: str
    initial_training_script_sha256: str


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
    parser.add_argument(
        "--adopt-interrupted-run",
        action="store_true",
        help=(
            "One-time recovery for a run created before initial_run_contract.json "
            "existed. Audits args.yaml, results.csv, last/epoch/best checkpoints, "
            "dataset, source weights, package versions, and records their hashes; "
            "requires --resume-from and never starts training in the same invocation."
        ),
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
        default="none",
        help=(
            "Image cache mode (default: none). Exact-contract grouped datasets "
            "forbid decoded caches outside their hashed file inventory."
        ),
    )
    parser.add_argument("--seed", type=_non_negative_int, default=0)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Statefully resume this exact run from its weights/last.pt. The "
            "dataset contract, initial-run contract, embedded checkpoint arguments, "
            "optimizer state, completed epoch, and matching epoch checkpoint are "
            "verified before Ultralytics is invoked."
        ),
    )
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
    resume_from = (
        Path(args.resume_from).expanduser().resolve()
        if args.resume_from is not None
        else None
    )
    if smoke_test and resume_from is not None:
        raise TrainingConfigurationError("--smoke-test cannot be combined with --resume-from")
    adopt_interrupted_run = bool(args.adopt_interrupted_run)
    if adopt_interrupted_run and resume_from is None:
        raise TrainingConfigurationError("--adopt-interrupted-run requires --resume-from")
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
        resume_from=resume_from,
        adopt_interrupted_run=adopt_interrupted_run,
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingConfigurationError(f"invalid {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"{description} root must be an object: {path}")
    return value


def _read_training_results(path: Path) -> dict[str, list[int | float]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if (
                not reader.fieldnames
                or any(not isinstance(name, str) or not name for name in reader.fieldnames)
                or len(set(reader.fieldnames)) != len(reader.fieldnames)
            ):
                raise TrainingConfigurationError("resume results.csv has invalid columns")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise TrainingConfigurationError(f"invalid resume results.csv: {exc}") from exc
    if not rows:
        raise TrainingConfigurationError("resume results.csv contains no completed epoch")
    try:
        raw_epochs = [float(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingConfigurationError("resume results.csv has invalid epoch values") from exc
    if not all(math.isfinite(value) and value.is_integer() for value in raw_epochs):
        raise TrainingConfigurationError("resume results.csv has invalid epoch values")
    epochs = [int(value) for value in raw_epochs]
    expected = list(range(1, len(epochs) + 1))
    if epochs != expected:
        raise TrainingConfigurationError(
            f"resume results.csv epoch sequence is not contiguous: {epochs!r}"
        )
    values: dict[str, list[int | float]] = {"epoch": epochs}
    assert reader.fieldnames is not None
    for name in reader.fieldnames:
        if name == "epoch":
            continue
        try:
            column = [float(row[name]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingConfigurationError(
                f"resume results.csv has invalid values for {name!r}"
            ) from exc
        if not all(math.isfinite(value) for value in column):
            raise TrainingConfigurationError(
                f"resume results.csv has non-finite values for {name!r}"
            )
        values[name] = column
    return values


def _read_completed_results_epoch(path: Path) -> int:
    return int(_read_training_results(path)["epoch"][-1])


def _verify_checkpoint_training_results(checkpoint: Mapping[str, Any], path: Path) -> None:
    embedded = checkpoint.get("train_results")
    if not isinstance(embedded, dict) or embedded != _read_training_results(path):
        raise TrainingConfigurationError(
            "resume checkpoint embedded results do not exactly match results.csv"
        )


def _normalized_path(value: object, description: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise TrainingConfigurationError(f"resume checkpoint has invalid {description}")
    return Path(value).expanduser().resolve()


def _torch_load_resume_checkpoint(path: Path) -> dict[str, Any]:
    """Load a local checkpoint after path/ownership checks; injectable in tests."""

    try:
        import torch
    except ImportError as exc:
        raise TrainingConfigurationError(
            "PyTorch is required to verify a resume checkpoint."
        ) from exc
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrainingConfigurationError(f"could not load resume checkpoint: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingConfigurationError("resume checkpoint root must be a dictionary")
    return value


def _initial_run_contract(config: TrainingConfig, manifest: Mapping[str, Any]) -> dict[str, Any]:
    contract = manifest.get("dataset_contract")
    content_sha = contract.get("content_sha256") if isinstance(contract, dict) else None
    return {
        "schema_version": 1,
        "run_dir": str(config.run_dir),
        "data": str(config.data),
        "dataset_manifest_sha256": _sha256_file(config.data.parent / "manifest.json"),
        "dataset_content_sha256": content_sha,
        "dataset_yaml_sha256": _sha256_file(config.data),
        "initial_weights": str(config.weights),
        "initial_weights_sha256": _sha256_file(config.weights),
        "training_script_sha256": _sha256_file(Path(__file__).resolve()),
        "training_script_sha256_status": "captured_at_launch",
        "adoption": None,
        "environment": {
            "ultralytics": _package_version("ultralytics"),
            "torch": _package_version("torch"),
        },
        "training": {
            "epochs": config.epochs,
            "patience": config.patience,
            "batch": config.batch,
            "imgsz": config.imgsz,
            "device": config.device,
            "workers": config.workers,
            "threads": config.threads,
            "cache": config.cache,
            "seed": config.seed,
            "smoke_test": config.smoke_test,
            "run_test": config.run_test,
        },
        "training_arguments": training_arguments(config),
    }


def _contract_for_comparison(config: TrainingConfig, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return current immutable inputs while preserving launch-time provenance fields."""

    return _initial_run_contract(config, manifest)


def _write_initial_run_contract(config: TrainingConfig, manifest: Mapping[str, Any]) -> None:
    path = config.run_dir / "initial_run_contract.json"
    if path.exists() or path.is_symlink() or os.path.lexists(path):
        raise TrainingConfigurationError(f"initial run contract already exists: {path}")
    path.write_text(
        json.dumps(_initial_run_contract(config, manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_flat_yaml(path: Path) -> dict[str, object]:
    """Parse Ultralytics args.yaml with its safe loader, only for adoption audit."""

    try:
        import yaml
    except ImportError as exc:
        raise TrainingConfigurationError("PyYAML is required to audit args.yaml") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TrainingConfigurationError(f"invalid interrupted-run args.yaml: {exc}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TrainingConfigurationError("interrupted-run args.yaml must be a mapping")
    return value


def adopt_interrupted_run(
    config: TrainingConfig,
    manifest: Mapping[str, Any],
    *,
    checkpoint_loader: Callable[[Path], dict[str, Any]] | None = None,
) -> Path:
    """Create a one-time, hash-bound contract; this function never trains."""

    if not config.adopt_interrupted_run or config.resume_from is None:
        raise TrainingConfigurationError("adoption requires --adopt-interrupted-run and --resume-from")
    contract_path = config.run_dir / "initial_run_contract.json"
    if contract_path.exists() or contract_path.is_symlink() or os.path.lexists(contract_path):
        raise TrainingConfigurationError(
            f"refusing to replace an existing initial run contract: {contract_path}"
        )
    checkpoint_path = config.resume_from
    expected_checkpoint_entry = config.run_dir / "weights" / "last.pt"
    if expected_checkpoint_entry.is_symlink():
        raise TrainingConfigurationError(
            f"adoption checkpoint must not be a symlink: {expected_checkpoint_entry}"
        )
    expected_checkpoint_path = expected_checkpoint_entry.resolve()
    if checkpoint_path != expected_checkpoint_path or not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise TrainingConfigurationError(
            f"adoption checkpoint must be this run's regular weights/last.pt: {expected_checkpoint_path}"
        )
    args_path = config.run_dir / "args.yaml"
    if not args_path.is_file() or args_path.is_symlink():
        raise TrainingConfigurationError(f"interrupted-run args.yaml is missing or unsafe: {args_path}")
    checkpoint = (checkpoint_loader or _torch_load_resume_checkpoint)(checkpoint_path)
    checkpoint_args = checkpoint.get("train_args")
    if not isinstance(checkpoint_args, dict):
        raise TrainingConfigurationError("interrupted checkpoint has no embedded train_args")
    yaml_args = _read_flat_yaml(args_path)
    # Ultralytics records args.yaml before CPU worker normalization and before
    # YOLO26's embedded recipe sets warmup_bias_lr=0.0 in the trainer/checkpoint.
    allowed_runtime_transforms = {
        "workers": 0 if str(config.device).casefold() in {"cpu", "mps"} else yaml_args.get("workers"),
        "warmup_bias_lr": 0.0,
    }
    normalized_yaml_args = {**yaml_args, **allowed_runtime_transforms}
    if normalized_yaml_args != checkpoint_args:
        differing = sorted(
            key
            for key in set(normalized_yaml_args) | set(checkpoint_args)
            if normalized_yaml_args.get(key) != checkpoint_args.get(key)
        )
        raise TrainingConfigurationError(
            "interrupted-run args.yaml differs from checkpoint train_args outside "
            f"the pinned Ultralytics runtime transforms: {differing}"
        )
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise TrainingConfigurationError("interrupted checkpoint has no completed epoch")
    completed_epoch = epoch + 1
    if _read_completed_results_epoch(config.run_dir / "results.csv") != completed_epoch:
        raise TrainingConfigurationError(
            "interrupted-run results.csv does not end at the checkpoint epoch"
        )
    _verify_checkpoint_training_results(checkpoint, config.run_dir / "results.csv")
    checkpoint_sha = _sha256_file(checkpoint_path)
    epoch_path = config.run_dir / "weights" / f"epoch{epoch}.pt"
    if not epoch_path.is_file() or epoch_path.is_symlink() or _sha256_file(epoch_path) != checkpoint_sha:
        raise TrainingConfigurationError(
            "interrupted last.pt does not match its saved epoch checkpoint"
        )
    best_path = config.run_dir / "weights" / "best.pt"
    if not best_path.is_file() or best_path.is_symlink():
        raise TrainingConfigurationError("interrupted run has no safe best.pt")
    if checkpoint.get("optimizer") is None or checkpoint.get("scaler") is None:
        raise TrainingConfigurationError("interrupted checkpoint lacks optimizer/scaler state")
    if checkpoint.get("ema") is None or checkpoint.get("updates") is None:
        raise TrainingConfigurationError("interrupted checkpoint lacks EMA state")
    if checkpoint.get("version") != _package_version("ultralytics"):
        raise TrainingConfigurationError("interrupted checkpoint Ultralytics version differs")

    contract = _initial_run_contract(config, manifest)
    contract["training_script_sha256"] = None
    contract["training_script_sha256_status"] = "unavailable_pre_contract_run"
    expected_arguments = contract["training_arguments"]
    assert isinstance(expected_arguments, dict)
    expected_embedded: dict[str, object] = {
        key: value for key, value in expected_arguments.items() if key != "workers"
    }
    expected_embedded.update(
        {
            "data": str(config.data),
            "model": str(config.weights),
            "project": str(config.project),
            "name": config.name,
            "save_dir": str(config.run_dir),
        }
    )
    for key, expected in expected_embedded.items():
        actual = checkpoint_args.get(key)
        if key in {"data", "model", "project", "save_dir"}:
            try:
                matches = _normalized_path(actual, key) == Path(str(expected)).resolve()
            except TrainingConfigurationError:
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise TrainingConfigurationError(
                f"interrupted checkpoint train_args mismatch for {key}: {actual!r} != {expected!r}"
            )
    # Ultralytics forces workers=0 after recording args.yaml on CPU. Pin both
    # values so this documented transformation cannot hide an arbitrary change.
    expected_worker = 0 if str(config.device).casefold() in {"cpu", "mps"} else config.workers
    if checkpoint_args.get("workers") != expected_worker:
        raise TrainingConfigurationError("interrupted checkpoint has unexpected worker count")

    contract["adoption"] = {
        "reason": "battery_stop_after_epoch_boundary_before_contract_feature_existed",
        "adopted_checkpoint_epoch": completed_epoch,
        "adopted_checkpoint_sha256": checkpoint_sha,
        "epoch_checkpoint_sha256": _sha256_file(epoch_path),
        "best_checkpoint_sha256": _sha256_file(best_path),
        "args_yaml_sha256": _sha256_file(args_path),
        "results_csv_sha256": _sha256_file(config.run_dir / "results.csv"),
        "checkpoint_version": checkpoint.get("version"),
        "checkpoint_updates": checkpoint.get("updates"),
        "checkpoint_best_fitness": checkpoint.get("best_fitness"),
        "adoption_script_sha256": _sha256_file(Path(__file__).resolve()),
        "launch_training_script_sha256": None,
        "launch_training_script_hash_reason": (
            "Run began before launch contracts existed; no immutable launch-time "
            "script copy/hash was recorded, so adoption does not fabricate one."
        ),
    }
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract_path


def _positive_manifest_split(stats: object) -> bool:
    return (
        isinstance(stats, dict)
        and isinstance(stats.get("retained_images"), int)
        and stats["retained_images"] > 0
        and isinstance(stats.get("retained_boxes"), int)
        and stats["retained_boxes"] > 0
    )


def _positive_grouped_manifest_split(stats: object) -> bool:
    return (
        isinstance(stats, dict)
        and isinstance(stats.get("images"), int)
        and stats["images"] > 0
        and isinstance(stats.get("boxes"), int)
        and stats["boxes"] > 0
        and isinstance(stats.get("source_groups"), int)
        and stats["source_groups"] > 0
    )


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
    is_legacy_prepared = (
        isinstance(conversion, dict)
        and conversion.get("output_class_names") == ["player"]
    )
    is_grouped = (
        manifest.get("cross_split_source_groups") == 0
        and manifest.get("cross_split_visual_similarity_edges") == 0
        and isinstance(manifest.get("visual_grouping"), dict)
        and manifest["visual_grouping"].get("enabled") is True
        and manifest.get("ambiguous_partial_label_images_are_never_negatives") is True
        and isinstance(manifest.get("assignment"), dict)
    )
    if not is_legacy_prepared and not is_grouped:
        raise TrainingConfigurationError("prepared dataset manifest is not one-class player data")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise TrainingConfigurationError("prepared dataset manifest has no split statistics")
    for split in ("train", "valid", "test"):
        stats = splits.get(split)
        if not (
            _positive_manifest_split(stats)
            if is_legacy_prepared
            else _positive_grouped_manifest_split(stats)
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
    if is_grouped:
        try:
            verify_grouped_dataset_metadata(data_file)
            verified_contract = verify_dataset_contract(
                dataset_root, manifest.get("dataset_contract")
            )
        except DatasetContractError as exc:
            raise TrainingConfigurationError(
                f"prepared grouped dataset exact-file contract failed: {exc}"
            ) from exc
        for split in ("train", "valid", "test"):
            stats = splits[split]
            contract_split = verified_contract["splits"][split]
            if (
                stats.get("images") != contract_split["images"]
                or stats.get("boxes") != contract_split["boxes"]
            ):
                raise TrainingConfigurationError(
                    f"prepared grouped dataset manifest counts disagree for {split}"
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
    if isinstance(manifest.get("dataset_contract"), dict) and config.cache != "none":
        raise TrainingConfigurationError(
            "exact-contract grouped training requires --cache none"
        )
    if not config.weights.is_file():
        raise TrainingConfigurationError(
            f"local pretrained weights not found: {config.weights}"
        )
    if config.weights.suffix.casefold() != ".pt":
        raise TrainingConfigurationError("pretrained weights must be a local .pt file")
    if config.resume_from is None:
        if config.run_dir.exists() or config.run_dir.is_symlink() or os.path.lexists(config.run_dir):
            raise TrainingConfigurationError(
                f"run directory already exists; refusing to reuse it: {config.run_dir}"
            )
    elif not config.run_dir.is_dir() or config.run_dir.is_symlink():
        raise TrainingConfigurationError(
            f"resume run directory is missing or unsafe: {config.run_dir}"
        )
    if config.run_test and config.resume_from is None:
        test_run = config.run_dir / "test"
        if test_run.exists() or test_run.is_symlink() or os.path.lexists(test_run):
            raise TrainingConfigurationError(
                f"test run directory already exists; refusing to reuse it: {test_run}"
            )
    return manifest


def validate_resume_checkpoint(
    config: TrainingConfig,
    manifest: Mapping[str, Any],
    *,
    checkpoint_loader: Callable[[Path], dict[str, Any]] | None = None,
) -> ResumeCheckpoint:
    """Fail closed unless ``last.pt`` is exact, resumable state for this run."""

    path = config.resume_from
    if path is None:
        raise TrainingConfigurationError("no --resume-from checkpoint was supplied")
    if not path.is_file() or path.is_symlink():
        raise TrainingConfigurationError(f"resume checkpoint is missing or unsafe: {path}")
    expected_entry = config.run_dir / "weights" / "last.pt"
    if expected_entry.is_symlink():
        raise TrainingConfigurationError(
            f"resume checkpoint must not be a symlink: {expected_entry}"
        )
    expected_path = expected_entry.resolve()
    if path != expected_path:
        raise TrainingConfigurationError(
            f"--resume-from must be this run's weights/last.pt: {expected_path}"
        )
    contract_path = config.run_dir / "initial_run_contract.json"
    contract = _read_json_object(contract_path, "initial run contract")
    if contract.get("schema_version") != 1:
        raise TrainingConfigurationError("unsupported initial run contract schema")
    expected_contract = _contract_for_comparison(config, manifest)
    # resume_from is deliberately absent from the pinned initial configuration.
    expected_contract["training_script_sha256"] = contract.get("training_script_sha256")
    expected_contract["training_script_sha256_status"] = contract.get(
        "training_script_sha256_status"
    )
    expected_contract["adoption"] = contract.get("adoption")
    if contract != expected_contract:
        raise TrainingConfigurationError(
            "current resume configuration/data/initial weights do not match initial_run_contract.json"
        )
    initial_script_sha = contract.get("training_script_sha256")
    script_hash_status = contract.get("training_script_sha256_status", "captured_at_launch")
    if script_hash_status == "captured_at_launch":
        if not isinstance(initial_script_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", initial_script_sha):
            raise TrainingConfigurationError("initial run contract has an invalid training script hash")
    elif script_hash_status == "unavailable_pre_contract_run":
        if initial_script_sha is not None:
            raise TrainingConfigurationError(
                "pre-contract adoption must not claim a launch-time training script hash"
            )
    else:
        raise TrainingConfigurationError("initial run contract has an invalid script hash status")

    checkpoint = (checkpoint_loader or _torch_load_resume_checkpoint)(path)
    checkpoint_version = checkpoint.get("version")
    contract_environment = contract.get("environment")
    if not isinstance(contract_environment, dict):
        raise TrainingConfigurationError("initial run contract has no environment versions")
    if checkpoint_version != contract_environment.get("ultralytics"):
        raise TrainingConfigurationError(
            "resume checkpoint Ultralytics version does not match the initial run contract"
        )
    if _package_version("ultralytics") != checkpoint_version:
        raise TrainingConfigurationError(
            "installed Ultralytics version does not match the resume checkpoint"
        )
    if _package_version("torch") != contract_environment.get("torch"):
        raise TrainingConfigurationError(
            "installed PyTorch version does not match the initial run contract"
        )
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise TrainingConfigurationError("resume checkpoint has no completed epoch")
    completed_epoch = epoch + 1
    if completed_epoch >= config.epochs:
        raise TrainingConfigurationError(
            f"resume checkpoint already completed {completed_epoch}/{config.epochs} epochs"
        )
    if not isinstance(checkpoint.get("optimizer"), dict) or not checkpoint["optimizer"]:
        raise TrainingConfigurationError("resume checkpoint has no optimizer state")
    if not isinstance(checkpoint.get("scaler"), dict):
        raise TrainingConfigurationError("resume checkpoint has no scaler state")
    if checkpoint.get("ema") is None or checkpoint.get("updates") is None:
        raise TrainingConfigurationError("resume checkpoint has no EMA training state")
    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, dict):
        raise TrainingConfigurationError("resume checkpoint has no embedded train_args")

    expected_args = contract.get("training_arguments")
    if not isinstance(expected_args, dict):
        raise TrainingConfigurationError("initial run contract has no training_arguments")
    embedded_checks: dict[str, object] = {
        key: value for key, value in expected_args.items() if key != "workers"
    }
    embedded_checks.update({
        "data": str(config.data),
        "project": str(config.project),
        "name": config.name,
        "save_dir": str(config.run_dir),
    })
    for key, expected in embedded_checks.items():
        actual = train_args.get(key)
        if key in {"data", "model", "project", "save_dir"}:
            try:
                matches = _normalized_path(actual, key) == Path(str(expected)).resolve()
            except TrainingConfigurationError:
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise TrainingConfigurationError(
                f"resume checkpoint train_args mismatch for {key}: {actual!r} != {expected!r}"
            )
    # The first checkpoint records the original pretrained weights as ``model``.
    # Once Ultralytics resumes, it intentionally rewrites that field to the exact
    # run-local last.pt path in every subsequent checkpoint. Accept only those two
    # path states; the contract above still pins the original weights and hash.
    try:
        embedded_model = _normalized_path(train_args.get("model"), "model")
    except TrainingConfigurationError:
        embedded_model = Path()
    allowed_embedded_models = {config.weights.resolve(), path}
    if embedded_model not in allowed_embedded_models:
        raise TrainingConfigurationError(
            "resume checkpoint train_args mismatch for model: "
            f"{train_args.get('model')!r} is neither the pinned initial weights "
            "nor this run's exact weights/last.pt"
        )
    expected_worker = (
        0 if str(config.device).casefold() in {"cpu", "mps"} else config.workers
    )
    if train_args.get("workers") != expected_worker:
        raise TrainingConfigurationError(
            "resume checkpoint has unexpected worker count"
        )

    csv_epoch = _read_completed_results_epoch(config.run_dir / "results.csv")
    if csv_epoch != completed_epoch:
        raise TrainingConfigurationError(
            f"resume results.csv ends at epoch {csv_epoch}, checkpoint ends at {completed_epoch}"
        )
    _verify_checkpoint_training_results(checkpoint, config.run_dir / "results.csv")
    epoch_checkpoint = config.run_dir / "weights" / f"epoch{epoch}.pt"
    if not epoch_checkpoint.is_file() or epoch_checkpoint.is_symlink():
        raise TrainingConfigurationError(
            f"matching saved epoch checkpoint is missing or unsafe: {epoch_checkpoint}"
        )
    checkpoint_sha = _sha256_file(path)
    if _sha256_file(epoch_checkpoint) != checkpoint_sha:
        raise TrainingConfigurationError(
            "last.pt does not match the saved checkpoint for its completed epoch"
        )
    best = config.run_dir / "weights" / "best.pt"
    if not best.is_file() or best.is_symlink():
        raise TrainingConfigurationError(f"resume best checkpoint is missing or unsafe: {best}")

    return ResumeCheckpoint(
        path=path,
        sha256=checkpoint_sha,
        completed_epoch=completed_epoch,
        initial_weights=config.weights,
        initial_weights_sha256=str(contract["initial_weights_sha256"]),
        dataset_manifest_sha256=str(contract["dataset_manifest_sha256"]),
        dataset_content_sha256=contract.get("dataset_content_sha256"),
        dataset_yaml_sha256=str(contract["dataset_yaml_sha256"]),
        initial_training_script_sha256=initial_script_sha or "unavailable_pre_contract_run",
    )


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
        # Standard randomized square training preserves mosaic and shuffle. In
        # Ultralytics, rect=True disables mosaic/mixup/cutmix and may disable
        # shuffle for mixed batch shapes, which is a poor small-object trade here.
        "rect": False,
        # The generated YAML already has one class. Explicitly keep single_cls
        # disabled because Ultralytics otherwise renames that class to "item".
        "single_cls": False,
        "close_mosaic": 10 if config.epochs > 10 else 0,
        "fraction": 0.02 if config.smoke_test else 1.0,
        "plots": not config.smoke_test,
        "save": True,
        "save_period": 1,
        "project": str(config.project),
        "name": config.name,
        "exist_ok": True,
        "verbose": True,
    }


def resume_training_arguments(config: TrainingConfig, state: ResumeCheckpoint) -> dict[str, Any]:
    """Return only Ultralytics' documented resume-safe overrides."""

    return {
        "resume": str(state.path),
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "workers": config.workers,
        "cache": False,
        "patience": config.patience,
        "close_mosaic": 10 if config.epochs > 10 else 0,
        "save_period": 1,
        "plots": not config.smoke_test,
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
        import ultralytics.data.dataset as ultralytics_dataset
    except ImportError as exc:
        raise TrainingConfigurationError(
            "Ultralytics is unavailable. Activate .venv-export and install "
            "requirements-export.txt."
        ) from exc

    def _ignore_dataset_cache(_path: Path) -> dict[str, Any]:
        raise FileNotFoundError(
            "Ultralytics label caches are disabled by the exact dataset contract"
        )

    def _do_not_write_dataset_cache(
        _prefix: str, _path: Path, value: dict[str, Any], version: str
    ) -> dict[str, Any]:
        value["version"] = version
        return value

    # These caches are loaded through numpy allow_pickle=True and sit outside
    # the hashed split folders. Always rescan verified labels and never persist
    # an implicit cache for a later process.
    ultralytics_dataset.load_dataset_cache_file = _ignore_dataset_cache
    ultralytics_dataset.save_dataset_cache_file = _do_not_write_dataset_cache

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
    *,
    resume_state: ResumeCheckpoint | None = None,
) -> None:
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    leakage = manifest.get("leakage") if isinstance(manifest.get("leakage"), dict) else {}
    initial_contract = config.run_dir / "initial_run_contract.json"
    results_csv = config.run_dir / "results.csv"
    completed_epochs = _read_completed_results_epoch(results_csv)
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
            "dataset_archive_sha256": source.get("archive_sha256")
            or manifest.get("source_archive_sha256"),
            "dataset_manifest_sha256": _sha256_file(config.data.parent / "manifest.json"),
            "dataset_content_sha256": (
                manifest.get("dataset_contract", {}).get("content_sha256")
                if isinstance(manifest.get("dataset_contract"), dict)
                else None
            ),
            "dataset_yaml_sha256": _sha256_file(config.data),
            "training_script_sha256": _sha256_file(Path(__file__).resolve()),
            "initial_run_contract": str(initial_contract),
            "initial_run_contract_sha256": _sha256_file(initial_contract),
            "dataset_split_assignment": manifest.get("assignment"),
            "reviewed_negative_images": manifest.get("reviewed_negative_images"),
            "initial_weights_sha256": _sha256_file(config.weights),
            "resumed_from": (
                {
                    "path": str(resume_state.path),
                    "sha256": resume_state.sha256,
                    "completed_epoch": resume_state.completed_epoch,
                    "sha256_scope": "captured_before_resume_training",
                }
                if resume_state is not None
                else None
            ),
        },
        "output": {
            "best_weights": str(best_weights),
            "best_weights_sha256": _sha256_file(best_weights),
            "best_weights_bytes": best_weights.stat().st_size,
            "results_csv": str(results_csv),
            "results_csv_sha256": _sha256_file(results_csv),
            "results_csv_bytes": results_csv.stat().st_size,
            "completed_epochs": completed_epochs,
            "runtime_class_labels": ["player"],
            "expected_openvino_output_format": "auto",
            "deployment_inference_size": config.imgsz,
        },
        "evaluation_caveat": {
            "supplied_splits_are_source_independent": False,
            "conservative_filename_source_groups_cross_splits": manifest.get(
                "cross_split_source_groups"
            ),
            "known_visual_similarity_edges_cross_splits": manifest.get(
                "cross_split_visual_similarity_edges"
            ),
            "visual_grouping": manifest.get("visual_grouping"),
            "metrics_may_be_optimistic": True,
            "reason": (
                "Filename and conservative perceptual grouping reduce known source "
                "overlap but cannot prove visual independence; independently captured "
                "clone footage is required."
                if manifest.get("cross_split_source_groups") == 0
                else "Supplied train/validation/test sources overlap."
            ),
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
    if manifest.get("cross_split_source_groups") == 0:
        print(
            "Group-aware dataset: no conservative filename/perceptual source group "
            "or known visual-similarity edge crosses train/validation/test."
        )
        print(
            "WARNING: Heuristic grouping cannot prove visual independence; use separately "
            "captured clone footage for final accuracy claims."
        )
        return
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

    if config.adopt_interrupted_run:
        raise TrainingConfigurationError(
            "--adopt-interrupted-run is audit-only and never starts training"
        )
    manifest = validate_training_config(config)
    resume_state = (
        validate_resume_checkpoint(config, manifest)
        if config.resume_from is not None
        else None
    )
    if resume_state is None:
        config.run_dir.mkdir(parents=True, exist_ok=False)
        _write_initial_run_contract(config, manifest)
    _configure_cpu_environment(config.threads)
    model_factory = yolo_class or _load_yolo_class(config.threads)
    model = model_factory(str(resume_state.path if resume_state else config.weights))
    if getattr(model, "task", None) != "detect":
        raise TrainingConfigurationError(
            f"unsupported pretrained task {getattr(model, 'task', None)!r}; expected 'detect'"
        )

    print(f"Training output: {config.run_dir}")
    if resume_state is not None:
        print(
            f"Stateful resume: completed epoch {resume_state.completed_epoch}, "
            f"checkpoint SHA-256 {resume_state.sha256}"
        )
    print(
        f"CPU configuration: {config.threads} compute thread(s), "
        f"{config.workers} data worker(s)"
    )
    if config.smoke_test:
        print("Smoke mode: one epoch, 2% training fraction, no cache/plots/test evaluation")
    _print_split_overlap_warning(manifest)
    model.train(
        **(
            resume_training_arguments(config, resume_state)
            if resume_state is not None
            else training_arguments(config)
        )
    )

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

    _write_reproducibility_record(
        config,
        manifest,
        best_weights,
        resume_state=resume_state,
    )
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
        if config.adopt_interrupted_run:
            manifest = validate_training_config(config)
            contract = adopt_interrupted_run(config, manifest)
            print(f"Adopted interrupted run without starting training: {contract}")
            return 0
        outcome = run_training(config)
    except TrainingConfigurationError as exc:
        parser.error(str(exc))
    _print_outcome(outcome, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
