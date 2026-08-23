#!/usr/bin/env python3
"""Smoke-test the frozen model bound by BUILD-INFO, then verify its report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_KEYS = frozenset(
    {
        "preset",
        "model_path",
        "labels_path",
        "input_shape_hw",
        "detail_crop_size_source_pixels",
        "model_sha256",
        "labels_sha256",
    }
)


class ReleaseDefaultSmokeError(RuntimeError):
    """Raised when the frozen release-default smoke cannot prove its binding."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _strict_json_loads(payload: str, description: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReleaseDefaultSmokeError(f"{description} is not strict JSON") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_bundle_file(bundle: Path, relative: object, description: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ReleaseDefaultSmokeError(f"{description} path is not bundle-relative POSIX")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ReleaseDefaultSmokeError(f"{description} path is unsafe")
    candidate = bundle.joinpath(*pure.parts)
    current = bundle
    try:
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise ReleaseDefaultSmokeError(f"{description} uses a symlink")
        file_stat = candidate.stat(follow_symlinks=False)
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ReleaseDefaultSmokeError(f"{description} is missing: {relative}") from exc
    except OSError as exc:
        raise ReleaseDefaultSmokeError(f"could not validate {description}") from exc
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
        raise ReleaseDefaultSmokeError(f"{description} must be a non-empty regular file")
    if bundle != resolved and bundle not in resolved.parents:
        raise ReleaseDefaultSmokeError(f"{description} escapes the bundle")
    return resolved


def load_release_default(bundle: Path) -> tuple[dict[str, Any], Path, Path]:
    root = bundle.expanduser().resolve()
    if not root.is_dir():
        raise ReleaseDefaultSmokeError(f"bundle directory not found: {root}")
    build_info_path = _regular_bundle_file(root, "BUILD-INFO.json", "BUILD-INFO")
    try:
        build_info = _strict_json_loads(
            build_info_path.read_text(encoding="utf-8"), "BUILD-INFO.json"
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseDefaultSmokeError("could not read BUILD-INFO.json") from exc
    if (
        not isinstance(build_info, dict)
        or build_info.get("application") != "ProAim"
        or build_info.get("schema") != 2
    ):
        raise ReleaseDefaultSmokeError("BUILD-INFO.json has the wrong schema or application")
    record = build_info.get("release_default_model")
    if not isinstance(record, dict) or set(record) != MODEL_KEYS:
        raise ReleaseDefaultSmokeError("BUILD-INFO has no exact release-default model record")
    preset = record.get("preset")
    if not isinstance(preset, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", preset
    ):
        raise ReleaseDefaultSmokeError("release-default preset key is invalid")
    shape = record.get("input_shape_hw")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            or dimension % 32
            for dimension in shape
        )
    ):
        raise ReleaseDefaultSmokeError("release-default input shape is invalid")
    detail_crop = record.get("detail_crop_size_source_pixels")
    if (
        isinstance(detail_crop, bool)
        or not isinstance(detail_crop, int)
        or detail_crop < 0
        or detail_crop > 16384
    ):
        raise ReleaseDefaultSmokeError("release-default detail workload is invalid")
    model = _regular_bundle_file(root, record.get("model_path"), "release-default model")
    labels = _regular_bundle_file(root, record.get("labels_path"), "release-default labels")
    if model == labels or model.suffix.lower() != ".onnx":
        raise ReleaseDefaultSmokeError("release-default model and labels paths are invalid")
    for path, key, description in (
        (model, "model_sha256", "model"),
        (labels, "labels_sha256", "labels"),
    ):
        expected = record.get(key)
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            raise ReleaseDefaultSmokeError(f"release-default {description} SHA-256 is invalid")
        if _sha256_file(path) != expected:
            raise ReleaseDefaultSmokeError(
                f"release-default {description} differs from BUILD-INFO SHA-256"
            )
    return dict(record), model, labels


def _single_artifact_hash(value: object, description: str, expected_path: Path) -> str:
    if not isinstance(value, Mapping):
        raise ReleaseDefaultSmokeError(f"benchmark omitted {description} artifact")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], Mapping):
        raise ReleaseDefaultSmokeError(f"benchmark {description} artifact is not one file")
    if files[0].get("resolved_path") != str(expected_path.resolve()):
        raise ReleaseDefaultSmokeError(
            f"benchmark {description} path differs from the requested bundle file"
        )
    digest = files[0].get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ReleaseDefaultSmokeError(f"benchmark {description} SHA-256 is invalid")
    return digest


def validate_benchmark(
    payload: object,
    record: Mapping[str, Any],
    model_path: Path,
    labels_path: Path,
    *,
    expected_device: str,
) -> None:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ReleaseDefaultSmokeError("frozen benchmark has an unsupported schema")
    methodology = payload.get("methodology")
    if (
        not isinstance(methodology, Mapping)
        or methodology.get("backend") != "onnxruntime"
        or methodology.get("requested_device") != expected_device
    ):
        raise ReleaseDefaultSmokeError(
            "frozen benchmark did not use the requested ONNX Runtime device"
        )
    if expected_device.strip().upper() == "CPU" and methodology.get(
        "require_full_provider"
    ) is not False:
        raise ReleaseDefaultSmokeError(
            "frozen CPU smoke must not request the accelerator-only full-provider gate"
        )
    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], Mapping):
        raise ReleaseDefaultSmokeError("frozen benchmark must report exactly one model")
    model = models[0]
    if model.get("key") != "release-default":
        raise ReleaseDefaultSmokeError("frozen benchmark did not use the release-default key")
    if model.get("input_shape_hw") != record["input_shape_hw"]:
        raise ReleaseDefaultSmokeError("frozen benchmark used a different input shape")
    if expected_device.strip().upper() == "CPU":
        runtime = model.get("runtime")
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("requested_provider") != "CPUExecutionProvider"
            or not isinstance(runtime.get("active_providers"), list)
            or "CPUExecutionProvider" not in runtime["active_providers"]
        ):
            raise ReleaseDefaultSmokeError(
                "frozen CPU smoke did not activate CPUExecutionProvider"
            )
    if (
        _single_artifact_hash(model.get("artifact"), "model", model_path)
        != record["model_sha256"]
    ):
        raise ReleaseDefaultSmokeError("frozen benchmark used a different model")
    if (
        _single_artifact_hash(model.get("labels_artifact"), "labels", labels_path)
        != record["labels_sha256"]
    ):
        raise ReleaseDefaultSmokeError("frozen benchmark used different labels")


def smoke_release_default(
    bundle: Path,
    executable: Path,
    output: Path,
    *,
    device: str = "CPU",
) -> dict[str, Any]:
    root = bundle.expanduser().resolve()
    record, model, labels = load_release_default(root)
    executable_path = executable.expanduser().resolve()
    try:
        relative_executable = executable_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReleaseDefaultSmokeError("frozen executable must be inside the bundle") from exc
    frozen = _regular_bundle_file(root, relative_executable, "frozen executable")
    height, width = record["input_shape_hw"]
    command = [
        str(frozen),
        "--benchmark-models",
        "--backend",
        "onnxruntime",
        "--model",
        str(model),
        "--labels",
        str(labels),
        "--inference-size",
        f"{height}x{width}",
        "--name",
        "release-default",
        "--precision",
        "release-default",
        "--device",
        device,
        "--synthetic",
        "--samples",
        "1",
        "--warmup",
        "1",
        "--iterations",
        "1",
        "--repeats",
        "1",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseDefaultSmokeError("could not run the frozen benchmark") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:1000]
        raise ReleaseDefaultSmokeError(
            f"frozen release-default benchmark failed ({completed.returncode}): {detail}"
        )
    payload = _strict_json_loads(completed.stdout, "frozen benchmark output")
    validate_benchmark(
        payload,
        record,
        model,
        labels,
        expected_device=device,
    )
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args(argv)
    try:
        smoke_release_default(
            args.bundle,
            args.executable,
            args.output,
            device=args.device,
        )
    except ReleaseDefaultSmokeError as exc:
        parser.error(str(exc))
    print(f"Verified frozen release-default smoke: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
