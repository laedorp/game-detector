#!/usr/bin/env python3
"""Create or validate the legal/payload inventory for a frozen CUDA bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


MANIFEST_NAME = "NVIDIA-REDISTRIBUTION-MANIFEST.json"
EXPECTED_LICENSE_EXPRESSION = "LicenseRef-NVIDIA-Proprietary"
NVIDIA_DISTRIBUTIONS = (
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-cudnn-cu13",
    "nvidia-cublas",
    "nvidia-nvjitlink",
)
NOTICE_MARKERS = (
    *NVIDIA_DISTRIBUTIONS,
    "NVIDIA CUDA Toolkit End User License Agreement",
    "NVIDIA cuDNN Software License Agreement",
)
LIBRARY_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "cuda_runtime": ("cudart",),
    "cuda_nvrtc": ("nvrtc",),
    "cufft": ("cufft",),
    "curand": ("curand",),
    "cudnn": ("cudnn",),
    "cublas": ("cublas",),
    "nvjitlink": ("nvjitlink",),
}
LEGAL_NAME_RE = re.compile(r"(?:license|licence|eula|notice|third[-_ ]?party)", re.I)
SAFE_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_-]{0,127}$")


class NvidiaManifestError(RuntimeError):
    """Raised when required NVIDIA redistribution evidence is incomplete."""


def _reject_json_constant(value: str) -> None:
    raise NvidiaManifestError(
        f"{MANIFEST_NAME} contains non-standard JSON number {value!r}"
    )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NvidiaManifestError(
                f"{MANIFEST_NAME} contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _strict_json_loads(payload: str) -> Any:
    return json.loads(
        payload,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_strict_json_object,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _relative_file_record(path: Path, bundle: Path) -> dict[str, Any]:
    resolved = path.resolve()
    root = bundle.resolve()
    if root not in resolved.parents:
        raise NvidiaManifestError(f"payload escaped bundle root: {path}")
    relative = resolved.relative_to(root).as_posix()
    if path.is_symlink() or not path.is_file():
        raise NvidiaManifestError(f"payload must be a regular file: {relative}")
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _metadata_directories(bundle: Path) -> dict[str, tuple[Path, Any]]:
    found: dict[str, tuple[Path, Any]] = {}
    for directory in sorted(bundle.rglob("*.dist-info")):
        if directory.is_symlink() or not directory.is_dir():
            continue
        metadata_path = directory / "METADATA"
        if not metadata_path.is_file() or metadata_path.is_symlink():
            continue
        try:
            message = BytesParser(policy=email_policy).parsebytes(metadata_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise NvidiaManifestError(f"invalid package metadata: {metadata_path}") from exc
        name = _canonical_name(str(message.get("Name") or ""))
        if name not in NVIDIA_DISTRIBUTIONS:
            continue
        if name in found:
            raise NvidiaManifestError(f"multiple bundled metadata directories identify {name}")
        found[name] = (directory, message)
    missing = set(NVIDIA_DISTRIBUTIONS).difference(found)
    if missing:
        raise NvidiaManifestError(
            "bundle is missing NVIDIA distribution metadata for: " + ", ".join(sorted(missing))
        )
    return found


def _distribution_record(
    name: str,
    directory: Path,
    message: Any,
    bundle: Path,
) -> dict[str, Any]:
    version = str(message.get("Version") or "").strip()
    if not SAFE_VERSION_RE.fullmatch(version):
        raise NvidiaManifestError(f"{name} has a missing or unsafe version in METADATA")
    license_expression = str(message.get("License-Expression") or "").strip()
    if license_expression != EXPECTED_LICENSE_EXPRESSION:
        raise NvidiaManifestError(
            f"{name} must declare {EXPECTED_LICENSE_EXPRESSION!r}; got {license_expression!r}"
        )
    declared = [str(value).strip() for value in (message.get_all("License-File") or [])]
    if not declared:
        raise NvidiaManifestError(f"{name} METADATA declares no License-File payload")
    legal_paths: dict[str, Path] = {}
    for value in declared:
        pure = PurePosixPath(value.replace("\\", "/"))
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
            raise NvidiaManifestError(f"{name} declares unsafe License-File path {value!r}")
        candidates = (
            directory.joinpath(*pure.parts),
            directory.joinpath("licenses", *pure.parts),
        )
        present = [
            candidate
            for candidate in candidates
            if candidate.is_file() and not candidate.is_symlink()
        ]
        if len(present) != 1:
            raise NvidiaManifestError(f"{name} is missing declared License-File {value!r}")
        candidate = present[0]
        legal_paths[candidate.resolve().as_posix()] = candidate
    for candidate in directory.rglob("*"):
        if candidate.is_file() and not candidate.is_symlink() and LEGAL_NAME_RE.search(candidate.name):
            legal_paths[candidate.resolve().as_posix()] = candidate
    if not legal_paths:
        raise NvidiaManifestError(f"{name} contains no bundled license/EULA/notice payload")
    metadata_path = directory / "METADATA"
    wheel_path = directory / "WHEEL"
    if not wheel_path.is_file() or wheel_path.is_symlink():
        raise NvidiaManifestError(f"{name} is missing its WHEEL metadata")
    return {
        "name": name,
        "version": version,
        "license_expression": license_expression,
        "metadata": _relative_file_record(metadata_path, bundle),
        "wheel_metadata": _relative_file_record(wheel_path, bundle),
        "declared_license_files": sorted(declared),
        "legal_payloads": [
            _relative_file_record(path, bundle)
            for path in sorted(legal_paths.values(), key=lambda item: item.as_posix().casefold())
        ],
    }


def _native_library_records(bundle: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    covered: set[str] = set()
    for path in sorted(bundle.rglob("*.dll"), key=lambda item: item.as_posix().casefold()):
        relative = path.relative_to(bundle).as_posix()
        folded_path = relative.casefold()
        folded_name = path.name.casefold()
        families = sorted(
            family
            for family, markers in LIBRARY_FAMILIES.items()
            if any(marker in folded_name for marker in markers)
        )
        under_nvidia_namespace = any(
            part.casefold() == "nvidia" or part.casefold().startswith("nvidia_")
            for part in PurePosixPath(relative).parts
        )
        if not families and not under_nvidia_namespace and "nvidia" not in folded_path:
            continue
        record = _relative_file_record(path, bundle)
        record["families"] = families
        records.append(record)
        covered.update(families)
    missing = set(LIBRARY_FAMILIES).difference(covered)
    if missing:
        raise NvidiaManifestError(
            "bundle is missing required NVIDIA native library families: "
            + ", ".join(sorted(missing))
        )
    if not records:
        raise NvidiaManifestError("bundle contains no inventoried NVIDIA DLL payloads")
    return records


def collect_manifest_payload(bundle: Path) -> dict[str, Any]:
    root = bundle.expanduser().resolve()
    if not root.is_dir():
        raise NvidiaManifestError(f"bundle directory not found: {root}")
    notices = root / "THIRD_PARTY_NOTICES.md"
    if not notices.is_file() or notices.is_symlink():
        raise NvidiaManifestError("bundle is missing top-level THIRD_PARTY_NOTICES.md")
    try:
        notice_text = notices.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise NvidiaManifestError("top-level THIRD_PARTY_NOTICES.md is not valid UTF-8") from exc
    missing_markers = [marker for marker in NOTICE_MARKERS if marker not in notice_text]
    if missing_markers:
        raise NvidiaManifestError(
            "top-level THIRD_PARTY_NOTICES.md omits NVIDIA inventory marker(s): "
            + ", ".join(missing_markers)
        )
    metadata = _metadata_directories(root)
    distributions = [
        _distribution_record(name, *metadata[name], root)
        for name in NVIDIA_DISTRIBUTIONS
    ]
    return {
        "schema_version": 1,
        "runtime_variant": "cuda",
        "license_review_limit": (
            "This inventory proves payload presence and identity only. It is not a legal "
            "opinion or a substitute for reviewing NVIDIA redistribution terms."
        ),
        "third_party_notices": {
            **_relative_file_record(notices, root),
            "required_inventory_markers": list(NOTICE_MARKERS),
        },
        "distributions": distributions,
        "native_libraries": _native_library_records(root),
    }


def write_manifest(bundle: Path) -> Path:
    root = bundle.expanduser().resolve()
    payload = collect_manifest_payload(root)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    target = root / MANIFEST_NAME
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def validate_manifest(bundle: Path) -> dict[str, Any]:
    root = bundle.expanduser().resolve()
    target = root / MANIFEST_NAME
    try:
        recorded = _strict_json_loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, NvidiaManifestError) as exc:
        raise NvidiaManifestError(f"invalid or missing {MANIFEST_NAME}") from exc
    if not isinstance(recorded, dict):
        raise NvidiaManifestError(f"{MANIFEST_NAME} must contain a JSON object")
    generated_at = recorded.pop("generated_at_utc", None)
    if not isinstance(generated_at, str) or not generated_at.strip():
        raise NvidiaManifestError(f"{MANIFEST_NAME} omits its generation time")
    expected = collect_manifest_payload(root)
    if recorded != expected:
        raise NvidiaManifestError(
            f"{MANIFEST_NAME} does not exactly match the bundled metadata, legal payloads, notices, and DLLs"
        )
    recorded["generated_at_utc"] = generated_at
    return recorded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--validate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.validate:
            validate_manifest(args.bundle)
            print(f"NVIDIA redistribution manifest validated: {args.bundle / MANIFEST_NAME}")
        else:
            target = write_manifest(args.bundle)
            validate_manifest(args.bundle)
            print(f"NVIDIA redistribution manifest written and validated: {target}")
    except NvidiaManifestError as exc:
        print(f"NVIDIA redistribution manifest rejected: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
