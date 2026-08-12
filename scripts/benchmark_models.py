#!/usr/bin/env python3
"""Benchmark bundled OpenVINO detectors without changing project state.

The benchmark preloads a deterministic set of BGR images, warms each model,
then measures application-equivalent preprocessing, synchronous batch-one
inference, and postprocessing separately.  Results are emitted as one JSON
document on stdout; progress and errors go to stderr.

This is a latency benchmark, not a training/evaluation runner.  It deliberately
does not create Ultralytics run directories, rewrite label caches, or download
models.  Accuracy results are documented in ``docs/MODEL_BENCHMARKS.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
from typing import Any, Protocol

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = PROJECT_ROOT / "datasets" / "fort_cuh_player" / "images" / "test"
DEFAULT_LABELS = PROJECT_ROOT / "models" / "fort_player.txt"
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png"})


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    model: Path
    inference_size: int
    precision: str


MODEL_PRESETS: dict[str, ModelSpec] = {
    "fort-320-fp32": ModelSpec(
        key="fort-320-fp32",
        model=PROJECT_ROOT / "models" / "fort_player_openvino_model" / "fort_player.xml",
        inference_size=320,
        precision="FP32",
    ),
    "fort-416-fp32": ModelSpec(
        key="fort-416-fp32",
        model=(
            PROJECT_ROOT
            / "models"
            / "fort_player_416_openvino_model"
            / "fort_player_416.xml"
        ),
        inference_size=416,
        precision="FP32",
    ),
    "fort-416-int8": ModelSpec(
        key="fort-416-int8",
        model=(
            PROJECT_ROOT
            / "models"
            / "fort_player_416_int8_openvino_model"
            / "fort_player_416_int8.xml"
        ),
        inference_size=416,
        precision="INT8 PTQ",
    ),
}


class DetectorLike(Protocol):
    @property
    def runtime_summary(self) -> Any: ...

    def infer(self, tensor: np.ndarray) -> np.ndarray: ...

    def postprocess(
        self,
        raw: np.ndarray,
        transform: Any | None = None,
        frame_shape: Sequence[int] | None = None,
    ) -> list[Any]: ...


class PreprocessedLike(Protocol):
    tensor: np.ndarray
    transform: Any


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_sample(paths: Iterable[Path], limit: int) -> list[Path]:
    """Return at most ``limit`` sorted paths spread across the whole input."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    ordered = sorted(Path(path) for path in paths)
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[0]]
    last = len(ordered) - 1
    return [ordered[round(index * last / (limit - 1))] for index in range(limit)]


def image_paths(directory: Path, limit: int) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    candidates = (
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    selected = deterministic_sample(candidates, limit)
    if not selected:
        raise ValueError(f"no supported images found in {directory}")
    return selected


def selection_sha256(paths: Sequence[Path]) -> str:
    """Fingerprint names and contents so two benchmark inputs can be compared."""

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def summarize_ms(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty timing sequence")
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
        "stdev": float(array.std(ddof=0)),
    }


def _timed_pipeline(
    detector: DetectorLike,
    frames: Sequence[np.ndarray],
    inference_size: int,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    preprocess: Callable[[np.ndarray, int], PreprocessedLike],
    clock: Callable[[], int] = perf_counter_ns,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("at least one frame is required")
    if inference_size <= 0 or warmup < 0 or iterations <= 0 or repeats <= 0:
        raise ValueError("invalid benchmark dimensions or iteration counts")

    def one(index: int) -> tuple[float, float, float, float, int]:
        frame = frames[index % len(frames)]
        started = clock()
        prepared = preprocess(frame, inference_size)
        after_preprocess = clock()
        raw = detector.infer(prepared.tensor)
        after_inference = clock()
        detections = detector.postprocess(
            raw,
            transform=prepared.transform,
            frame_shape=frame.shape,
        )
        finished = clock()
        divisor = 1_000_000.0
        return (
            (after_preprocess - started) / divisor,
            (after_inference - after_preprocess) / divisor,
            (finished - after_inference) / divisor,
            (finished - started) / divisor,
            len(detections),
        )

    for index in range(warmup):
        one(index)

    aggregate: dict[str, list[float]] = {
        "preprocess": [],
        "inference": [],
        "postprocess": [],
        "pipeline": [],
    }
    repeat_summaries: list[dict[str, Any]] = []
    detection_counts: list[int] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(repeats):
            current: dict[str, list[float]] = {name: [] for name in aggregate}
            current_counts: list[int] = []
            offset = repeat * iterations
            for index in range(iterations):
                values = one(offset + index)
                for name, value in zip(current, values[:4], strict=True):
                    current[name].append(value)
                    aggregate[name].append(value)
                current_counts.append(values[4])
                detection_counts.append(values[4])
            repeat_summaries.append(
                {
                    "repeat": repeat + 1,
                    "timing_ms": {
                        name: summarize_ms(values) for name, values in current.items()
                    },
                    "detections_mean": float(statistics.fmean(current_counts)),
                }
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    pipeline = summarize_ms(aggregate["pipeline"])
    pipeline_mean = float(pipeline["mean"])
    return {
        "timing_ms": {name: summarize_ms(values) for name, values in aggregate.items()},
        "pipeline_fps_from_mean": 1000.0 / pipeline_mean,
        "detections_mean": float(statistics.fmean(detection_counts)),
        "repeats": repeat_summaries,
    }


def _load_frames(paths: Sequence[Path]) -> list[np.ndarray]:
    import cv2

    frames: list[np.ndarray] = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"OpenCV could not decode benchmark image: {path}")
        frames.append(frame)
    return frames


def _synthetic_frames(count: int) -> list[np.ndarray]:
    """Generate deterministic nonuniform 16:9 frames for runtime-only checks."""

    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8) for _ in range(count)]


def _artifact_record(model_xml: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in (model_xml, model_xml.with_suffix(".bin")):
        if not path.is_file():
            raise FileNotFoundError(f"model artifact not found: {path}")
        try:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            relative = str(path)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "bytes_total": sum(item["bytes"] for item in files)}


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _host_record() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENVINO_NUM_THREADS")
        },
    }


def _specs_from_args(args: argparse.Namespace) -> list[ModelSpec]:
    if args.model is not None:
        if args.inference_size is None:
            raise ValueError("--inference-size is required with --model")
        return [
            ModelSpec(
                key=args.name,
                model=args.model.expanduser().resolve(),
                inference_size=args.inference_size,
                precision=args.precision,
            )
        ]
    keys = args.preset or list(MODEL_PRESETS)
    return [MODEL_PRESETS[key] for key in keys]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        action="append",
        choices=tuple(MODEL_PRESETS),
        help="Bundled model to test; repeat for multiple models (default: all three).",
    )
    parser.add_argument("--model", type=Path, help="Benchmark one custom OpenVINO XML model.")
    parser.add_argument("--inference-size", type=positive_int)
    parser.add_argument("--name", default="custom", help="JSON key for a custom model.")
    parser.add_argument("--precision", default="unknown", help="Custom model precision label.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--device", default="CPU", help="OpenVINO device (default: CPU).")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--samples", type=positive_int, default=32)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic generated frames instead of dataset images.",
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--repeats", type=positive_int, default=3)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.model is not None and args.preset:
        raise ValueError("--model and --preset cannot be used together")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from detection.openvino_yolo import OpenVINOYoloDetector
    from utils.preprocess import preprocess_frame

    specs = _specs_from_args(args)
    labels = args.labels.expanduser().resolve()
    if not labels.is_file():
        raise FileNotFoundError(f"labels file not found: {labels}")

    selected_paths: list[Path] = []
    if args.synthetic:
        frames = _synthetic_frames(min(args.samples, 32))
        input_record: dict[str, Any] = {
            "kind": "synthetic",
            "count": len(frames),
            "generator": "numpy.default_rng(seed=0), uint8 720x1280 BGR",
        }
    else:
        selected_paths = image_paths(args.images.expanduser().resolve(), args.samples)
        frames = _load_frames(selected_paths)
        input_record = {
            "kind": "images",
            "directory": str(args.images.expanduser().resolve()),
            "available_count": sum(
                1
                for path in args.images.expanduser().resolve().iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            "selected_count": len(selected_paths),
            "selection": "sorted paths, evenly spaced including first and last",
            "selection_sha256": selection_sha256(selected_paths),
            "first": selected_paths[0].name,
            "last": selected_paths[-1].name,
        }

    results: list[dict[str, Any]] = []
    for spec in specs:
        print(f"Benchmarking {spec.key} on {args.device}...", file=sys.stderr, flush=True)
        detector = OpenVINOYoloDetector(
            model_path=spec.model,
            labels_path=labels,
            device=args.device,
            inference_size=spec.inference_size,
            # Use the application's ordinary thresholds; this keeps
            # postprocessing work representative without affecting inference.
            confidence=0.25,
            iou=0.45,
            output_format="auto",
        )
        measured = _timed_pipeline(
            detector,
            frames,
            spec.inference_size,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            preprocess=preprocess_frame,
        )
        results.append(
            {
                "key": spec.key,
                "precision": spec.precision,
                "inference_size": spec.inference_size,
                "artifact": _artifact_record(spec.model),
                "runtime": dict(detector.runtime_summary),
                **measured,
            }
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "batch": 1,
            "execution": "synchronous",
            "performance_hint": "LATENCY",
            "streams": 1,
            "warmup_per_model": args.warmup,
            "iterations_per_repeat": args.iterations,
            "repeats": args.repeats,
            "timer": "time.perf_counter_ns",
            "garbage_collection_during_measurement": "disabled",
            "scope": "preloaded frame; capture, decode, display, and disk I/O excluded",
        },
        "host": _host_record(),
        "input": input_record,
        "models": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run(args)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        json.dump(
            {"schema_version": 1, "error": f"{type(exc).__name__}: {exc}"},
            sys.stdout,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 2
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
