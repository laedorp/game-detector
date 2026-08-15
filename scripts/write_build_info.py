#!/usr/bin/env python3
"""Write a deterministic identity record into a completed ProAim bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from launcher.settings import release_default_model_contract  # noqa: E402


RUNTIME_VARIANTS = ("cpu", "cuda", "directml", "rocm")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _bundle_regular_file(bundle: Path, relative: str, description: str) -> Path:
    """Resolve one canonical bundle-relative POSIX path without following links."""

    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError(f"{description} path must be a bundle-relative POSIX path")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError(f"{description} path must be a canonical bundle-relative POSIX path")
    candidate = bundle.joinpath(*pure.parts)
    current = bundle
    try:
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"{description} must not use a symlink: {relative}")
        file_stat = candidate.stat(follow_symlinks=False)
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{description} not found in bundle: {candidate}") from exc
    except OSError as exc:
        raise ValueError(f"could not validate {description}: {candidate}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{description} must be a regular file: {candidate}")
    if file_stat.st_size <= 0:
        raise ValueError(f"{description} must be non-empty: {candidate}")
    if bundle != resolved and bundle not in resolved.parents:
        raise ValueError(f"{description} escapes the bundle: {relative}")
    return resolved


def _release_default_model_record(bundle: Path) -> dict[str, object]:
    contract = release_default_model_contract()
    if not isinstance(contract, dict):
        raise ValueError("release default model contract must be an object")
    preset = contract.get("preset")
    if not isinstance(preset, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", preset
    ):
        raise ValueError("release default model contract has no preset")
    input_shape = contract.get("input_shape_hw")
    if (
        not isinstance(input_shape, list)
        or len(input_shape) != 2
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 32
            or dimension > 4096
            or dimension % 32
            for dimension in input_shape
        )
    ):
        raise ValueError("release default model contract has an invalid [height, width] shape")
    detail_crop = contract.get("detail_crop_size_source_pixels")
    if (
        isinstance(detail_crop, bool)
        or not isinstance(detail_crop, int)
        or detail_crop < 0
        or detail_crop > 16384
    ):
        raise ValueError("release default model contract has an invalid detail workload")
    resource_paths: dict[str, str] = {}
    for key in ("model_path", "labels_path"):
        value = contract.get(key)
        if not isinstance(value, str):
            raise ValueError(f"release default model contract has no {key}")
        pure = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or pure.is_absolute()
            or pure.as_posix() != value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError(f"release default model contract has an unsafe {key}")
        resource_paths[key] = value
    model_relative = (PurePosixPath("_internal") / resource_paths["model_path"]).as_posix()
    labels_relative = (PurePosixPath("_internal") / resource_paths["labels_path"]).as_posix()
    if model_relative == labels_relative:
        raise ValueError("release default model and labels must be different files")
    model = _bundle_regular_file(bundle, model_relative, "release default ONNX model")
    labels = _bundle_regular_file(bundle, labels_relative, "release default labels")
    return {
        "detail_crop_size_source_pixels": detail_crop,
        "input_shape_hw": list(input_shape),
        "labels_path": labels_relative,
        "labels_sha256": _sha256_file(labels),
        "model_path": model_relative,
        "model_sha256": _sha256_file(model),
        "preset": preset,
    }


def _dependency_record(
    dependency_manifest: Path | None,
    bundle: Path,
    runtime_variant: str,
) -> dict[str, object] | None:
    if dependency_manifest is None:
        return None
    resolved = dependency_manifest.expanduser().resolve()
    if resolved.parent != bundle or resolved.name != "DEPENDENCY-MANIFEST.json":
        raise ValueError(
            "dependency manifest must be the adjacent bundle file "
            f"{bundle / 'DEPENDENCY-MANIFEST.json'}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"dependency manifest not found: {resolved}")
    try:
        manifest = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"dependency manifest is invalid JSON: {resolved}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("dependency manifest schema is unsupported")
    if manifest.get("application") != "ProAim":
        raise ValueError("dependency manifest identifies the wrong application")
    if manifest.get("runtime_variant") != runtime_variant:
        raise ValueError(
            "dependency manifest runtime variant does not match BUILD-INFO: "
            f"{manifest.get('runtime_variant')!r} != {runtime_variant!r}"
        )
    profile = manifest.get("lock_profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("dependency manifest has no lock profile")
    expected_profiles = {
        "cpu": "linux-cpu-py313",
        "cuda": "windows-cuda-py313",
        "directml": "windows-directml-py313",
    }
    expected_profile = expected_profiles.get(runtime_variant)
    if expected_profile is not None and profile != expected_profile:
        raise ValueError(
            f"dependency manifest profile {profile!r} does not match {expected_profile!r}"
        )
    distributions = manifest.get("distributions")
    if not isinstance(distributions, list) or not distributions:
        raise ValueError("dependency manifest has no distribution records")
    artifact_contract = manifest.get("artifact_hash_contract")
    if not isinstance(artifact_contract, dict) or artifact_contract.get(
        "enforced_before_install"
    ) is not True:
        raise ValueError("dependency manifest does not enforce artifact hashes before install")
    for distribution in distributions:
        if not isinstance(distribution, dict):
            raise ValueError("dependency manifest contains a non-object distribution record")
        installed_files = distribution.get("installed_files")
        if not isinstance(installed_files, dict) or set(installed_files) != {
            "aggregate_sha256",
            "record_document_sha256",
            "record_entry_count",
            "record_sha256_entries_verified",
            "total_size_bytes",
            "unhashed_record_entries",
        }:
            raise ValueError("dependency manifest lacks installed-file verification")
        entry_count = installed_files.get("record_entry_count")
        verified_count = installed_files.get("record_sha256_entries_verified")
        unhashed_count = installed_files.get("unhashed_record_entries")
        total_size = installed_files.get("total_size_bytes")
        if (
            not isinstance(entry_count, int)
            or isinstance(entry_count, bool)
            or entry_count < 2
            or not isinstance(verified_count, int)
            or isinstance(verified_count, bool)
            or verified_count != entry_count - 1
            or not isinstance(unhashed_count, int)
            or isinstance(unhashed_count, bool)
            or unhashed_count != 1
            or not isinstance(total_size, int)
            or isinstance(total_size, bool)
            or total_size <= 0
        ):
            raise ValueError("dependency manifest has an invalid installed-file count")
        for key in ("aggregate_sha256", "record_document_sha256"):
            if not isinstance(installed_files.get(key), str) or not SHA256_RE.fullmatch(
                installed_files[key]
            ):
                raise ValueError("dependency manifest has an invalid installed-file digest")
        if installed_files["record_document_sha256"] != distribution.get(
            "installed_record_sha256"
        ):
            raise ValueError("dependency manifest RECORD digests disagree")
    return {
        "distribution_count": len(distributions),
        "lock_profile": profile,
        "path": resolved.name,
        "schema_version": 1,
        "sha256": _sha256_file(resolved),
    }


def _git_value(root: Path, *arguments: str) -> str | None:
    """Return stripped git output, preserving an empty successful result."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def write_build_info(
    bundle: Path,
    runtime_variant: str,
    project_root: Path,
    dependency_manifest: Path | None = None,
) -> Path:
    resolved = bundle.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {resolved}")
    normalized_variant = runtime_variant.strip().lower()
    if normalized_variant not in RUNTIME_VARIANTS:
        raise ValueError(
            f"unknown runtime variant {runtime_variant!r}; choose one of "
            + ", ".join(RUNTIME_VARIANTS)
        )
    executable = resolved / ("ProAim.exe" if (resolved / "ProAim.exe").is_file() else "ProAim")
    if not executable.is_file():
        raise FileNotFoundError(f"bundle executable not found: {executable}")
    project_root = project_root.expanduser().resolve()
    commit = _git_value(project_root, "rev-parse", "HEAD")
    if not commit:
        commit = os.environ.get("GITHUB_SHA", "").strip() or "unknown"
    commit_time = _git_value(project_root, "show", "-s", "--format=%cI", "HEAD")
    status = _git_value(project_root, "status", "--porcelain", "--untracked-files=normal")
    dependency_record = _dependency_record(dependency_manifest, resolved, normalized_variant)
    release_default_model = _release_default_model_record(resolved)
    payload = {
        "application": "ProAim",
        "commit": commit,
        "commit_time": commit_time or "unknown",
        "dependency_manifest": dependency_record,
        # ``null`` is intentionally different from a clean checkout: it means
        # the bundle was built without usable source-control metadata.
        "dirty": None if status is None else bool(status),
        "release_default_model": release_default_model,
        "runtime_variant": normalized_variant,
        "schema": 2,
    }
    target = resolved / "BUILD-INFO.json"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--runtime-variant", required=True, choices=RUNTIME_VARIANTS)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dependency-manifest", type=Path)
    args = parser.parse_args(argv)
    target = write_build_info(
        args.bundle,
        args.runtime_variant,
        args.project_root.resolve(),
        dependency_manifest=args.dependency_manifest,
    )
    print(f"Build identity written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
