"""Canonical synthetic dependency-manifest fixture for holdout contract tests."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from utils import independent_holdout_release_contract as contract


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def valid_windows_directml_dependency_manifest(
    project_root: Path,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    locked = contract._release_locked_environment(root)
    distributions = []
    for canonical_name, expected in sorted(locked.items()):
        record_sha = _digest(f"{canonical_name}:record")
        distributions.append(
            {
                "artifact": {
                    "filename": f"{canonical_name}-{expected['version']}-fixture.whl",
                    "sha256": sorted(expected["artifact_hashes"])[0],
                    "source": "files.pythonhosted.org",
                },
                "canonical_name": canonical_name,
                "installed_files": {
                    "aggregate_sha256": _digest(f"{canonical_name}:aggregate"),
                    "record_document_sha256": record_sha,
                    "record_entry_count": 2,
                    "record_sha256_entries_verified": 1,
                    "total_size_bytes": 1,
                    "unhashed_record_entries": 1,
                },
                "installed_metadata_sha256": _digest(f"{canonical_name}:metadata"),
                "installed_record_sha256": record_sha,
                "locked_extras": [],
                "name": canonical_name,
                "version": expected["version"],
            }
        )
    inputs = [
        {
            "path": relative,
            "sha256": contract.sha256_file(root / relative),
        }
        for relative in (
            "requirements-locks/bootstrap-py313.txt",
            "requirements-locks/windows-directml-py313.txt",
            "requirements.txt",
            "requirements-build.txt",
            "requirements-runtime-directml.txt",
        )
    ]
    declared = [
        {
            "canonical_name": name,
            "declared_extras": [],
            "locked_version": version,
            "sources": [source],
        }
        for name, version, source in contract.RELEASE_ENVIRONMENT_DECLARED_REQUIREMENTS
    ]
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
        "distributions": distributions,
        "declared_requirements": declared,
        "inputs": inputs,
        "lock_profile": contract.RELEASE_ENVIRONMENT_LOCK_PROFILE,
        "pip_reports": [
            {
                "filename": "pip-bootstrap-windows-directml-py313.json",
                "pip_version": "26.2.1",
                "purpose": "bootstrap",
                "sha256": _digest("bootstrap-report"),
            },
            {
                "filename": "pip-dependencies-windows-directml-py313.json",
                "pip_version": "26.2.1",
                "purpose": "final-environment-install",
                "sha256": _digest("dependency-report"),
            },
        ],
        "python": {
            "cache_tag": "cpython-313",
            "executable_sha256": _digest("python.exe"),
            "implementation": "CPython",
            "version": contract.RELEASE_ENVIRONMENT_PYTHON_VERSION,
        },
        "runtime_variant": "directml",
        "schema_version": 1,
        "target": {"machine": "amd64", "system": "Windows"},
    }
