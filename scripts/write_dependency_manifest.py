#!/usr/bin/env python3
"""Verify a locked release environment and record its installed artifacts.

Windows wheels and Linux wheels/sdists are different artifacts, so every
supported release target has its own ``--require-hashes`` lock. The workflows
install those locks into a fresh virtual environment and retain pip's JSON
reports. This tool then fails closed on platform, interpreter,
distribution-set, version, runtime-variant, dependency, and artifact drift
before PyInstaller runs.

Each resulting manifest cross-checks pip's downloaded wheel/sdist SHA-256
against the repository lock, verifies every installed file against its PEP 376
RECORD hash and size, then records deterministic installed-file aggregates.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
from dataclasses import dataclass
import hashlib
from importlib import metadata
import io
import json
import ntpath
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_DISTRIBUTIONS = frozenset(
    {"onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-rocm"}
)
PYPI_ARTIFACT_HOST = "files.pythonhosted.org"


class DependencyContractError(RuntimeError):
    """Raised when a release environment cannot satisfy its pinned contract."""


@dataclass(frozen=True)
class LockProfile:
    name: str
    system: str
    machines: tuple[str, ...]
    python_version: str
    runtime_variant: str
    runtime_distribution: str
    bootstrap_lock_path: str
    platform_lock_path: str
    requirements_paths: tuple[str, ...]


@dataclass(frozen=True)
class LockedDistribution:
    name: str
    version: str
    hashes: frozenset[str]
    extras: frozenset[str] = frozenset()


LOCK_PROFILES: dict[str, LockProfile] = {
    "linux-cpu-py313": LockProfile(
        name="linux-cpu-py313",
        system="Linux",
        machines=("amd64", "x86_64"),
        python_version="3.13.14",
        runtime_variant="cpu",
        runtime_distribution="onnxruntime",
        bootstrap_lock_path="requirements-locks/bootstrap-py313.txt",
        platform_lock_path="requirements-locks/linux-cpu-py313.txt",
        requirements_paths=(
            "requirements.txt",
            "requirements-build.txt",
            "requirements-runtime-cpu.txt",
        ),
    ),
    "windows-directml-py313": LockProfile(
        name="windows-directml-py313",
        system="Windows",
        machines=("amd64", "x86_64"),
        python_version="3.13.14",
        runtime_variant="directml",
        runtime_distribution="onnxruntime-directml",
        bootstrap_lock_path="requirements-locks/bootstrap-py313.txt",
        platform_lock_path="requirements-locks/windows-directml-py313.txt",
        requirements_paths=(
            "requirements.txt",
            "requirements-build.txt",
            "requirements-runtime-directml.txt",
        ),
    ),
    "windows-cuda-py313": LockProfile(
        name="windows-cuda-py313",
        system="Windows",
        machines=("amd64", "x86_64"),
        python_version="3.13.14",
        runtime_variant="cuda",
        runtime_distribution="onnxruntime-gpu",
        bootstrap_lock_path="requirements-locks/bootstrap-py313.txt",
        platform_lock_path="requirements-locks/windows-cuda-py313.txt",
        requirements_paths=(
            "requirements.txt",
            "requirements-build.txt",
            "requirements-runtime-cuda.txt",
        ),
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: str) -> str:
    return str(canonicalize_name(value))


def parse_hashed_lock(path: Path) -> dict[str, LockedDistribution]:
    """Return canonical name -> exact version and permitted artifact hashes."""

    if not path.is_file():
        raise DependencyContractError(f"release lock file not found: {path}")
    locked: dict[str, LockedDistribution] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        if line.startswith(("-", "--")):
            raise DependencyContractError(
                f"{path}:{number}: nested files and pip options are forbidden in a release lock"
            )
        fields = line.split()
        requirement_text, hash_fields = fields[0], fields[1:]
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement as exc:
            raise DependencyContractError(
                f"{path}:{number}: invalid locked requirement {requirement_text!r}"
            ) from exc
        specifiers = list(requirement.specifier)
        if (
            requirement.marker is not None
            or requirement.url is not None
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise DependencyContractError(
                f"{path}:{number}: release locks must be unconditional name==version pins"
            )
        hashes: set[str] = set()
        for field in hash_fields:
            prefix = "--hash=sha256:"
            if not field.startswith(prefix):
                raise DependencyContractError(
                    f"{path}:{number}: only --hash=sha256:<digest> options are permitted"
                )
            digest = field[len(prefix) :].lower()
            if not SHA256_RE.fullmatch(digest):
                raise DependencyContractError(
                    f"{path}:{number}: invalid SHA-256 artifact hash"
                )
            hashes.add(digest)
        if not hashes:
            raise DependencyContractError(
                f"{path}:{number}: every locked distribution requires an artifact hash"
            )
        canonical = _canonical(requirement.name)
        if canonical in locked:
            raise DependencyContractError(
                f"{path}:{number}: duplicate distribution {requirement.name!r}"
            )
        locked[canonical] = LockedDistribution(
            requirement.name,
            specifiers[0].version,
            frozenset(hashes),
            frozenset(sorted(requirement.extras)),
        )
    if not locked:
        raise DependencyContractError(f"release lock file is empty: {path}")
    return locked


def load_profile_lock(
    project_root: Path, profile: LockProfile
) -> dict[str, LockedDistribution]:
    locked: dict[str, LockedDistribution] = {}
    for relative in (profile.bootstrap_lock_path, profile.platform_lock_path):
        for canonical, requirement in parse_hashed_lock(project_root / relative).items():
            if canonical in locked:
                raise DependencyContractError(
                    f"distribution {requirement.name!r} appears in more than one profile lock"
                )
            locked[canonical] = requirement
    return locked


def runtime_identity() -> dict[str, str]:
    return {
        "implementation": platform.python_implementation(),
        "machine": platform.machine().lower(),
        "python_version": platform.python_version(),
        "system": platform.system(),
    }


def validate_runtime(profile: LockProfile, identity: Mapping[str, str]) -> None:
    failures: list[str] = []
    if identity.get("implementation") != "CPython":
        failures.append(f"implementation={identity.get('implementation')!r}")
    if identity.get("python_version") != profile.python_version:
        failures.append(
            f"python={identity.get('python_version')!r} (expected {profile.python_version})"
        )
    if identity.get("system") != profile.system:
        failures.append(f"system={identity.get('system')!r} (expected {profile.system})")
    if identity.get("machine", "").lower() not in profile.machines:
        failures.append(
            f"machine={identity.get('machine')!r} (expected one of {profile.machines})"
        )
    if failures:
        raise DependencyContractError(
            f"host does not match release lock {profile.name}: " + "; ".join(failures)
        )


def _marker_environment(profile: LockProfile) -> dict[str, str]:
    environment = default_environment()
    environment.update(
        {
            "implementation_name": "cpython",
            "implementation_version": profile.python_version,
            "os_name": "nt" if profile.system == "Windows" else "posix",
            "platform_machine": "AMD64" if profile.system == "Windows" else "x86_64",
            "platform_python_implementation": "CPython",
            "platform_system": profile.system,
            "python_full_version": profile.python_version,
            "python_version": ".".join(profile.python_version.split(".")[:2]),
            "sys_platform": "win32" if profile.system == "Windows" else "linux",
        }
    )
    return environment


def validate_declared_requirements(
    project_root: Path,
    profile: LockProfile,
    locked: Mapping[str, LockedDistribution],
) -> list[dict[str, Any]]:
    """Prove the target-active source/build/runtime requirements are in the lock."""

    project_root = project_root.expanduser().resolve()
    marker_environment = _marker_environment(profile)
    visited: set[Path] = set()
    active: dict[str, dict[str, Any]] = {}

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        if resolved != project_root and project_root not in resolved.parents:
            raise DependencyContractError(f"requirement include escapes project root: {path}")
        if not resolved.is_file():
            raise DependencyContractError(f"declared requirement file not found: {resolved}")
        visited.add(resolved)
        for number, raw_line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.partition("#")[0].strip()
            if not line:
                continue
            if line.startswith(("-r ", "--requirement ")):
                include = line.split(maxsplit=1)[1].strip()
                if not include:
                    raise DependencyContractError(
                        f"{resolved}:{number}: empty requirement include"
                    )
                visit((resolved.parent / include).resolve())
                continue
            if line.startswith(("-", "--")):
                raise DependencyContractError(
                    f"{resolved}:{number}: unsupported pip option in declared requirements"
                )
            try:
                requirement = Requirement(line)
            except InvalidRequirement as exc:
                raise DependencyContractError(
                    f"{resolved}:{number}: invalid declared requirement {line!r}"
                ) from exc
            if requirement.url is not None:
                raise DependencyContractError(
                    f"{resolved}:{number}: direct URL requirements are forbidden in releases"
                )
            if requirement.marker is not None and not requirement.marker.evaluate(
                marker_environment
            ):
                continue
            canonical = _canonical(requirement.name)
            pinned = locked.get(canonical)
            if pinned is None:
                raise DependencyContractError(
                    f"active declared requirement {requirement.name!r} is missing from "
                    f"release lock {profile.name}"
                )
            try:
                pinned_version = Version(pinned.version)
            except InvalidVersion as exc:
                raise DependencyContractError(
                    f"release lock has an invalid version for {pinned.name}: {pinned.version}"
                ) from exc
            if requirement.specifier and not requirement.specifier.contains(
                pinned_version, prereleases=True
            ):
                raise DependencyContractError(
                    f"release lock pins {pinned.name}=={pinned.version}, which does not satisfy "
                    f"declared requirement {requirement}"
                )
            missing_extras = set(requirement.extras).difference(pinned.extras)
            if missing_extras:
                raise DependencyContractError(
                    f"release lock for {pinned.name} omits declared extras: "
                    + ", ".join(sorted(missing_extras))
                )
            relative = resolved.relative_to(project_root).as_posix()
            record = active.setdefault(
                canonical,
                {
                    "canonical_name": canonical,
                    "declared_extras": set(),
                    "locked_version": pinned.version,
                    "sources": set(),
                },
            )
            record["declared_extras"].update(requirement.extras)
            record["sources"].add(relative)

    for relative in profile.requirements_paths:
        visit(project_root / relative)
    return [
        {
            **record,
            "declared_extras": sorted(record["declared_extras"]),
            "sources": sorted(record["sources"]),
        }
        for _, record in sorted(active.items())
    ]


def _report_environment_matches(
    report: Mapping[str, Any], profile: LockProfile, report_path: Path
) -> None:
    environment = report.get("environment")
    if not isinstance(environment, Mapping):
        raise DependencyContractError(f"pip report has no environment object: {report_path}")
    expected = {
        "implementation_name": "cpython",
        "platform_system": profile.system,
        "python_full_version": profile.python_version,
    }
    for key, expected_value in expected.items():
        if environment.get(key) != expected_value:
            raise DependencyContractError(
                f"pip report {report_path} has {key}={environment.get(key)!r}; "
                f"expected {expected_value!r}"
            )
    if str(environment.get("platform_machine", "")).lower() not in profile.machines:
        raise DependencyContractError(
            f"pip report {report_path} has unsupported machine "
            f"{environment.get('platform_machine')!r}"
        )


def load_artifact_reports(
    report_paths: Sequence[Path],
    profile: LockProfile,
    locked: Mapping[str, LockedDistribution],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    if not report_paths:
        raise DependencyContractError("at least one pip --report JSON file is required")
    artifacts: dict[str, dict[str, Any]] = {}
    report_records: list[dict[str, str]] = []
    resolved_report_paths = [path.expanduser().resolve() for path in report_paths]
    if len(set(resolved_report_paths)) != len(resolved_report_paths):
        raise DependencyContractError("each pip --report JSON path must be unique")
    final_report_distributions: set[str] | None = None
    final_report_pip_version: str | None = None
    for report_index, resolved in enumerate(resolved_report_paths):
        if not resolved.is_file():
            raise DependencyContractError(f"pip report not found: {resolved}")
        try:
            report = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DependencyContractError(f"invalid pip report JSON: {resolved}") from exc
        if not isinstance(report, Mapping) or report.get("version") != "1":
            raise DependencyContractError(f"unsupported pip report schema: {resolved}")
        _report_environment_matches(report, profile, resolved)
        installs = report.get("install")
        if not isinstance(installs, list):
            raise DependencyContractError(f"pip report install field is not a list: {resolved}")
        pip_version = str(report.get("pip_version", "unknown"))
        purpose = (
            "final-environment-install"
            if report_index == len(resolved_report_paths) - 1
            else "bootstrap"
        )
        report_records.append(
            {
                "filename": resolved.name,
                "pip_version": pip_version,
                "purpose": purpose,
                "sha256": sha256_file(resolved),
            }
        )
        report_distributions: set[str] = set()
        for entry in installs:
            if not isinstance(entry, Mapping):
                raise DependencyContractError(f"pip report contains a non-object install: {resolved}")
            package_metadata = entry.get("metadata")
            if not isinstance(package_metadata, Mapping):
                raise DependencyContractError(f"pip report install has no metadata: {resolved}")
            name = package_metadata.get("name")
            version = package_metadata.get("version")
            if not isinstance(name, str) or not isinstance(version, str):
                raise DependencyContractError(
                    f"pip report install lacks a string name/version: {resolved}"
                )
            canonical = _canonical(name)
            if canonical in report_distributions:
                raise DependencyContractError(
                    f"pip report contains duplicate distribution {name!r}: {resolved}"
                )
            report_distributions.add(canonical)
            expected = locked.get(canonical)
            if expected is None:
                raise DependencyContractError(
                    f"pip report installed unlisted distribution {name}=={version}"
                )
            if version != expected.version:
                raise DependencyContractError(
                    f"pip report installed {name}=={version}; lock requires {expected.version}"
                )
            if entry.get("is_direct") is not False or entry.get("is_yanked") is not False:
                raise DependencyContractError(
                    f"release artifact for {name} must be a non-yanked, non-direct PyPI download"
                )
            download = entry.get("download_info")
            if not isinstance(download, Mapping):
                raise DependencyContractError(f"pip report lacks download_info for {name}")
            url = download.get("url")
            parsed = urlparse(url) if isinstance(url, str) else None
            if (
                parsed is None
                or parsed.scheme != "https"
                or parsed.hostname != PYPI_ARTIFACT_HOST
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise DependencyContractError(
                    f"release artifact for {name} is not a plain HTTPS files.pythonhosted.org URL"
                )
            archive_info = download.get("archive_info")
            hashes = archive_info.get("hashes") if isinstance(archive_info, Mapping) else None
            sha256 = hashes.get("sha256") if isinstance(hashes, Mapping) else None
            if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256.lower()):
                raise DependencyContractError(f"pip report lacks a SHA-256 artifact hash for {name}")
            if sha256.lower() not in expected.hashes:
                raise DependencyContractError(
                    f"pip report artifact SHA-256 for {name} is not permitted by the release lock"
                )
            artifact = {
                "filename": unquote(Path(parsed.path).name),
                "sha256": sha256.lower(),
                "source": PYPI_ARTIFACT_HOST,
            }
            if not artifact["filename"]:
                raise DependencyContractError(
                    f"pip report artifact URL has no filename for {name}"
                )
            previous = artifacts.get(canonical)
            if previous is not None and previous != artifact:
                raise DependencyContractError(
                    f"pip reports disagree about the artifact installed for {name}"
                )
            artifacts[canonical] = artifact
        if purpose == "final-environment-install":
            final_report_distributions = report_distributions
            final_report_pip_version = pip_version
    expected_distributions = set(locked)
    if final_report_distributions != expected_distributions:
        observed_final = final_report_distributions or set()
        missing_final = sorted(expected_distributions.difference(observed_final))
        unlisted_final = sorted(observed_final.difference(expected_distributions))
        details: list[str] = []
        if missing_final:
            details.append("missing=" + ",".join(missing_final))
        if unlisted_final:
            details.append("unlisted=" + ",".join(unlisted_final))
        raise DependencyContractError(
            "final pip report must account for the complete exact release environment: "
            + " | ".join(details)
        )
    locked_pip = locked.get("pip")
    if locked_pip is None:
        raise DependencyContractError("release lock must include pip itself")
    if final_report_pip_version != locked_pip.version:
        raise DependencyContractError(
            f"final pip report was produced by pip {final_report_pip_version!r}; "
            f"release lock requires pip {locked_pip.version!r}"
        )
    missing = sorted(set(locked).difference(artifacts))
    if missing:
        raise DependencyContractError(
            "pip reports do not account for every locked distribution: " + ", ".join(missing)
        )
    return artifacts, report_records


def installed_distributions(
    distributions: Iterable[metadata.Distribution] | None = None,
) -> dict[str, metadata.Distribution]:
    installed: dict[str, metadata.Distribution] = {}
    for distribution in metadata.distributions() if distributions is None else distributions:
        name = distribution.metadata.get("Name")
        if not name:
            raise DependencyContractError("installed distribution has no Name metadata")
        canonical = _canonical(name)
        if canonical in installed:
            raise DependencyContractError(f"duplicate installed distribution metadata for {name}")
        installed[canonical] = distribution
    return installed


def _distribution_document(distribution: metadata.Distribution, filename: str) -> bytes:
    try:
        value = distribution.read_text(filename)
    except (OSError, UnicodeDecodeError) as exc:
        raise DependencyContractError(
            f"could not read {filename} for {distribution.metadata.get('Name', 'unknown')}"
        ) from exc
    if value is None:
        raise DependencyContractError(
            f"installed distribution {distribution.metadata.get('Name', 'unknown')} "
            f"has no {filename}"
        )
    return value.encode("utf-8")


def _hash_installed_file(path: Path) -> tuple[str, int]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DependencyContractError(f"could not hash installed file: {path}") from exc
    return sha256_bytes(payload), len(payload)


def _decode_record_sha256(value: str, *, distribution_name: str, record_path: str) -> str:
    try:
        algorithm, encoded = value.split("=", 1)
    except ValueError as exc:
        raise DependencyContractError(
            f"installed RECORD for {distribution_name} has a malformed hash for {record_path}"
        ) from exc
    if algorithm != "sha256" or not encoded or "=" in encoded:
        raise DependencyContractError(
            f"installed RECORD for {distribution_name} must use unpadded sha256 hashes: "
            f"{record_path}"
        )
    try:
        raw_digest = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
    except (binascii.Error, ValueError, TypeError) as exc:
        raise DependencyContractError(
            f"installed RECORD for {distribution_name} has an invalid SHA-256 for {record_path}"
        ) from exc
    if len(raw_digest) != hashlib.sha256().digest_size:
        raise DependencyContractError(
            f"installed RECORD for {distribution_name} has an invalid SHA-256 for {record_path}"
        )
    return raw_digest.hex()


def verify_distribution_record(
    distribution: metadata.Distribution,
    environment_root: Path,
) -> tuple[dict[str, Any], set[Path]]:
    """Verify every installed file named by one distribution's PEP 376 RECORD."""

    name = str(distribution.metadata.get("Name", "unknown"))
    root = environment_root.expanduser().resolve()
    if not root.is_dir():
        raise DependencyContractError(f"release environment root is not a directory: {root}")
    record_document = _distribution_document(distribution, "RECORD")
    try:
        rows = list(csv.reader(io.StringIO(record_document.decode("utf-8")), strict=True))
    except (csv.Error, UnicodeDecodeError) as exc:
        raise DependencyContractError(f"installed RECORD for {name} is malformed") from exc
    if not rows:
        raise DependencyContractError(f"installed RECORD for {name} is empty")

    seen_record_paths: set[str] = set()
    seen_files: set[Path] = set()
    aggregate_entries: list[dict[str, Any]] = []
    self_entries = 0
    record_document_sha256: str | None = None
    verified_entries = 0
    for row_number, row in enumerate(rows, 1):
        if len(row) != 3:
            raise DependencyContractError(
                f"installed RECORD for {name} row {row_number} must have exactly three fields"
            )
        record_path, record_hash, record_size = row
        if not record_path or record_path in seen_record_paths:
            raise DependencyContractError(
                f"installed RECORD for {name} has an empty or duplicate path at row {row_number}"
            )
        seen_record_paths.add(record_path)
        candidate = Path(record_path)
        windows_drive, _ = ntpath.splitdrive(record_path)
        if (
            candidate.is_absolute()
            or ntpath.isabs(record_path)
            or windows_drive
            or "\x00" in record_path
        ):
            raise DependencyContractError(
                f"installed RECORD for {name} contains an unsafe path: {record_path!r}"
            )
        try:
            located = Path(distribution.locate_file(record_path))
            resolved = located.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise DependencyContractError(
                f"installed RECORD file for {name} is missing or invalid: {record_path}"
            ) from exc
        if root != resolved and root not in resolved.parents:
            raise DependencyContractError(
                f"installed RECORD for {name} escapes the release environment: {record_path}"
            )
        if located.is_symlink() or not resolved.is_file() or resolved in seen_files:
            raise DependencyContractError(
                f"installed RECORD for {name} has a symlink, non-file, or alias: {record_path}"
            )
        seen_files.add(resolved)
        actual_sha256, actual_size = _hash_installed_file(resolved)
        normalized_parts = record_path.replace("\\", "/").split("/")
        is_record_path = (
            len(normalized_parts) >= 2
            and normalized_parts[-1] == "RECORD"
            and normalized_parts[-2].endswith(".dist-info")
        )
        is_record_self = is_record_path and not record_hash and not record_size
        if is_record_self:
            self_entries += 1
            record_document_sha256 = actual_sha256
            try:
                located_record_document = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise DependencyContractError(
                    f"could not read installed RECORD self-entry for {name}"
                ) from exc
            if located_record_document != record_document.decode("utf-8"):
                raise DependencyContractError(
                    f"installed RECORD self-entry for {name} does not identify the read document"
                )
        else:
            if not record_hash or not record_size:
                raise DependencyContractError(
                    f"installed RECORD for {name} has an unhashed non-RECORD file: {record_path}"
                )
            expected_sha256 = _decode_record_sha256(
                record_hash, distribution_name=name, record_path=record_path
            )
            if (
                not record_size.isascii()
                or not record_size.isdecimal()
                or str(int(record_size)) != record_size
            ):
                raise DependencyContractError(
                    f"installed RECORD for {name} has an invalid size for {record_path}"
                )
            if actual_size != int(record_size):
                raise DependencyContractError(
                    f"installed file size differs from RECORD for {name}: {record_path}"
                )
            if actual_sha256 != expected_sha256:
                raise DependencyContractError(
                    f"installed file SHA-256 differs from RECORD for {name}: {record_path}"
                )
            verified_entries += 1
        aggregate_entries.append(
            {
                "path": resolved.relative_to(root).as_posix(),
                "sha256": actual_sha256,
                "size": actual_size,
            }
        )
    if self_entries != 1:
        raise DependencyContractError(
            f"installed RECORD for {name} must contain exactly one unhashed self-entry"
        )
    canonical_entries = sorted(aggregate_entries, key=lambda item: item["path"])
    aggregate_payload = json.dumps(
        canonical_entries, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return (
        {
            "aggregate_sha256": sha256_bytes(aggregate_payload),
            "record_entry_count": len(canonical_entries),
            "record_document_sha256": record_document_sha256,
            "record_sha256_entries_verified": verified_entries,
            "total_size_bytes": sum(item["size"] for item in canonical_entries),
            "unhashed_record_entries": 1,
        },
        seen_files,
    )


def verify_installed_payload_coverage(
    distributions: Iterable[metadata.Distribution],
    environment_root: Path,
    record_files: Mapping[Path, str],
) -> None:
    """Reject files importable from site-packages but absent from every RECORD."""

    root = environment_root.expanduser().resolve()
    install_roots: set[Path] = set()
    for distribution in distributions:
        name = str(distribution.metadata.get("Name", "unknown"))
        try:
            located_root = Path(distribution.locate_file(""))
            resolved_root = located_root.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise DependencyContractError(
                f"could not resolve installed package root for {name}"
            ) from exc
        if root != resolved_root and root not in resolved_root.parents:
            raise DependencyContractError(
                f"installed package root for {name} escapes the release environment"
            )
        if located_root.is_symlink() or not resolved_root.is_dir():
            raise DependencyContractError(
                f"installed package root for {name} must be a real directory"
            )
        install_roots.add(resolved_root)

    actual_files: set[Path] = set()
    for install_root in sorted(install_roots):
        for path in install_root.rglob("*"):
            if path.is_symlink():
                raise DependencyContractError(
                    f"release package directory contains an unverified symlink: {path}"
                )
            if not path.is_file():
                continue
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise DependencyContractError(
                    f"could not resolve installed package payload: {path}"
                ) from exc
            if install_root != resolved and install_root not in resolved.parents:
                raise DependencyContractError(
                    f"installed package payload escapes its package root: {path}"
                )
            actual_files.add(resolved)
    claimed_files = {
        path
        for path in record_files
        if any(
            path == install_root or install_root in path.parents
            for install_root in install_roots
        )
    }
    unrecorded = sorted(actual_files.difference(claimed_files))
    missing = sorted(claimed_files.difference(actual_files))
    if unrecorded or missing:
        details: list[str] = []
        if unrecorded:
            details.append(
                "unrecorded="
                + ",".join(path.relative_to(root).as_posix() for path in unrecorded[:20])
            )
        if missing:
            details.append(
                "missing="
                + ",".join(path.relative_to(root).as_posix() for path in missing[:20])
            )
        raise DependencyContractError(
            "installed package roots differ from the complete RECORD file set: "
            + " | ".join(details)
        )


def verify_installed_set(
    locked: Mapping[str, LockedDistribution],
    expected_runtime: str,
    distributions: Iterable[metadata.Distribution] | None = None,
    *,
    environment_root: Path | None = None,
    verify_installed_files: bool = True,
) -> tuple[dict[str, metadata.Distribution], list[dict[str, Any]]]:
    installed = installed_distributions(distributions)
    missing = sorted(set(locked).difference(installed))
    unexpected = sorted(set(installed).difference(locked))
    mismatches = sorted(
        f"{canonical}: installed {installed[canonical].version}, locked {locked[canonical].version}"
        for canonical in set(locked).intersection(installed)
        if installed[canonical].version != locked[canonical].version
    )
    installed_runtimes = sorted(set(installed).intersection(RUNTIME_DISTRIBUTIONS))
    failures: list[str] = []
    if missing:
        failures.append("missing=" + ",".join(missing))
    if unexpected:
        failures.append("unlisted=" + ",".join(unexpected))
    if mismatches:
        failures.append("version_mismatch=" + ";".join(mismatches))
    if installed_runtimes != [_canonical(expected_runtime)]:
        failures.append(
            f"runtime_distributions={installed_runtimes!r} "
            f"(expected [{_canonical(expected_runtime)!r}])"
        )
    if failures:
        raise DependencyContractError(
            "installed release environment differs from its exact lock: " + " | ".join(failures)
        )
    records: list[dict[str, Any]] = []
    resolved_environment_root = (
        Path(sys.prefix).resolve()
        if environment_root is None
        else environment_root.expanduser().resolve()
    )
    all_record_files: dict[Path, str] = {}
    for canonical in sorted(installed):
        distribution = installed[canonical]
        installed_files = None
        if verify_installed_files:
            installed_files, distribution_files = verify_distribution_record(
                distribution, resolved_environment_root
            )
            overlapping = sorted(set(all_record_files).intersection(distribution_files))
            if overlapping:
                previous = all_record_files[overlapping[0]]
                raise DependencyContractError(
                    f"installed file is claimed by both {previous} and {canonical}: "
                    f"{overlapping[0]}"
                )
            all_record_files.update({path: canonical for path in distribution_files})
        records.append(
            {
                "canonical_name": canonical,
                "installed_files": installed_files,
                "installed_metadata_sha256": sha256_bytes(
                    _distribution_document(distribution, "METADATA")
                ),
                "installed_record_sha256": (
                    installed_files["record_document_sha256"]
                    if installed_files is not None
                    else sha256_bytes(_distribution_document(distribution, "RECORD"))
                ),
                "name": str(distribution.metadata["Name"]),
                "version": distribution.version,
            }
        )
    if verify_installed_files:
        verify_installed_payload_coverage(
            installed.values(), resolved_environment_root, all_record_files
        )
    return installed, records


def run_pip_check() -> None:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DependencyContractError("could not run pip check") from exc
    if completed.returncode != 0:
        details = (completed.stdout + completed.stderr).strip()
        raise DependencyContractError("pip check failed: " + (details or "no diagnostics"))


def build_manifest(
    *,
    profile: LockProfile,
    project_root: Path,
    report_paths: Sequence[Path],
    distributions: Iterable[metadata.Distribution] | None = None,
    identity: Mapping[str, str] | None = None,
    check_dependencies: bool = True,
    verify_installed_files: bool = True,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    validate_runtime(profile, runtime_identity() if identity is None else identity)
    locked = load_profile_lock(root, profile)
    declared_requirements = validate_declared_requirements(root, profile, locked)
    artifacts, reports = load_artifact_reports(report_paths, profile, locked)
    _, records = verify_installed_set(
        locked,
        profile.runtime_distribution,
        distributions=distributions,
        verify_installed_files=verify_installed_files,
    )
    if check_dependencies:
        run_pip_check()
    for record in records:
        record["artifact"] = artifacts[record["canonical_name"]]
        record["locked_extras"] = sorted(locked[record["canonical_name"]].extras)
    input_paths = (
        profile.bootstrap_lock_path,
        profile.platform_lock_path,
        *profile.requirements_paths,
    )
    inputs: list[dict[str, str]] = []
    seen_input_paths: set[str] = set()
    for relative in input_paths:
        if relative in seen_input_paths:
            continue
        seen_input_paths.add(relative)
        path = root / relative
        if not path.is_file():
            raise DependencyContractError(f"release dependency input not found: {path}")
        inputs.append({"path": relative, "sha256": sha256_file(path)})
    executable = Path(sys.executable).resolve()
    return {
        "application": "ProAim",
        "artifact_hash_contract": {
            "enforced_before_install": True,
            "scope": (
                "repository-pinned pip --require-hashes wheel/sdist SHA-256; installed "
                "METADATA and RECORD content SHA-256; every installed RECORD payload "
                "entry SHA-256 and size"
            ),
        },
        "distributions": records,
        "declared_requirements": declared_requirements,
        "inputs": inputs,
        "lock_profile": profile.name,
        "pip_reports": reports,
        "python": {
            "cache_tag": sys.implementation.cache_tag,
            "executable_sha256": sha256_file(executable),
            "implementation": "CPython",
            "version": profile.python_version,
        },
        "runtime_variant": profile.runtime_variant,
        "schema_version": 1,
        "target": {
            "machine": runtime_identity()["machine"] if identity is None else identity["machine"],
            "system": profile.system,
        },
    }


def write_manifest(payload: Mapping[str, Any], output: Path) -> Path:
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=tuple(sorted(LOCK_PROFILES)))
    parser.add_argument("--pip-report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    try:
        payload = build_manifest(
            profile=LOCK_PROFILES[args.profile],
            project_root=args.project_root,
            report_paths=args.pip_report,
        )
        target = write_manifest(payload, args.output)
    except DependencyContractError as exc:
        parser.error(str(exc))
    print(f"Verified dependency manifest written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
