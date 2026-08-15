from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.write_build_info import _git_value, main, write_build_info
from launcher.settings import DEFAULT_MODEL_PRESET, release_default_model_contract


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


class BuildInfoTests(unittest.TestCase):
    def _make_bundle(self, root: Path, executable_name: str = "ProAim") -> Path:
        bundle = root / "bundle"
        bundle.mkdir()
        (bundle / executable_name).write_bytes(b"test executable")
        contract = release_default_model_contract()
        model = bundle / "_internal" / str(contract["model_path"])
        labels = bundle / "_internal" / str(contract["labels_path"])
        model.parent.mkdir(parents=True, exist_ok=True)
        labels.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"test onnx model")
        labels.write_bytes(b"player\n")
        return bundle

    def _make_repository(self, root: Path) -> Path:
        repository = root / "repository"
        repository.mkdir()
        _run_git(repository, "init", "--quiet")
        _run_git(repository, "config", "user.name", "ProAim Tests")
        _run_git(repository, "config", "user.email", "tests@example.invalid")
        tracked = repository / "tracked.txt"
        tracked.write_text("clean\n", encoding="utf-8")
        _run_git(repository, "add", "tracked.txt")
        _run_git(repository, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "test")
        return repository

    def _make_dependency_manifest(
        self, bundle: Path, runtime_variant: str = "directml"
    ) -> Path:
        target = bundle / "DEPENDENCY-MANIFEST.json"
        target.write_text(
            json.dumps(
                {
                    "application": "ProAim",
                    "artifact_hash_contract": {"enforced_before_install": True},
                    "distributions": [
                        {
                            "canonical_name": "example",
                            "installed_files": {
                                "aggregate_sha256": "a" * 64,
                                "record_document_sha256": "b" * 64,
                                "record_entry_count": 2,
                                "record_sha256_entries_verified": 1,
                                "total_size_bytes": 123,
                                "unhashed_record_entries": 1,
                            },
                            "installed_record_sha256": "b" * 64,
                            "name": "example",
                            "version": "1",
                        }
                    ],
                    "lock_profile": f"windows-{runtime_variant}-py313",
                    "runtime_variant": runtime_variant,
                    "schema_version": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    def test_clean_and_dirty_repositories_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self._make_repository(root)
            bundle = self._make_bundle(root)

            target = write_build_info(bundle, " CPU ", repository)
            clean = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(clean["application"], "ProAim")
            self.assertEqual(clean["commit"], _run_git(repository, "rev-parse", "HEAD"))
            self.assertNotEqual(clean["commit_time"], "unknown")
            self.assertIs(clean["dirty"], False)
            self.assertEqual(clean["runtime_variant"], "cpu")
            self.assertEqual(clean["schema"], 2)
            self.assertIsNone(clean["dependency_manifest"])
            default_model = clean["release_default_model"]
            contract = release_default_model_contract()
            self.assertEqual(default_model["preset"], DEFAULT_MODEL_PRESET)
            self.assertEqual(default_model["preset"], contract["preset"])
            self.assertEqual(default_model["input_shape_hw"], contract["input_shape_hw"])
            self.assertEqual(
                default_model["detail_crop_size_source_pixels"],
                contract["detail_crop_size_source_pixels"],
            )
            self.assertEqual(
                default_model["model_path"], f"_internal/{contract['model_path']}"
            )
            self.assertEqual(
                default_model["labels_path"], f"_internal/{contract['labels_path']}"
            )
            model = bundle.joinpath(*default_model["model_path"].split("/"))
            labels = bundle.joinpath(*default_model["labels_path"].split("/"))
            self.assertEqual(
                default_model["model_sha256"], hashlib.sha256(model.read_bytes()).hexdigest()
            )
            self.assertEqual(
                default_model["labels_sha256"], hashlib.sha256(labels.read_bytes()).hexdigest()
            )
            self.assertTrue(target.read_bytes().endswith(b"\n"))
            self.assertFalse((bundle / ".BUILD-INFO.json.tmp").exists())

            (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
            write_build_info(bundle, "cuda", repository)
            dirty = json.loads(target.read_text(encoding="utf-8"))
            self.assertIs(dirty["dirty"], True)
            self.assertEqual(dirty["runtime_variant"], "cuda")

    def test_missing_git_metadata_is_not_reported_as_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._make_bundle(root)
            source_archive = root / "source"
            source_archive.mkdir()
            with mock.patch.dict(os.environ, {"GITHUB_SHA": "abc123"}):
                target = write_build_info(bundle, "directml", source_archive)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["commit"], "abc123")
            self.assertEqual(payload["commit_time"], "unknown")
            self.assertIsNone(payload["dirty"])

    def test_git_value_preserves_empty_successful_output(self) -> None:
        completed = subprocess.CompletedProcess(["git"], 0, stdout="\n", stderr="")
        with mock.patch("scripts.write_build_info.subprocess.run", return_value=completed):
            self.assertEqual(_git_value(Path("."), "status", "--porcelain"), "")

    def test_invalid_bundle_and_variant_fail_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                write_build_info(root / "missing", "cpu", root)
            bundle = self._make_bundle(root)
            with self.assertRaisesRegex(ValueError, "unknown runtime variant"):
                write_build_info(bundle, "vulkan", root)
            self.assertFalse((bundle / "BUILD-INFO.json").exists())

    def test_release_default_model_contract_fails_closed_on_missing_or_unsafe_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._make_bundle(root)
            contract = release_default_model_contract()
            model = bundle / "_internal" / str(contract["model_path"])
            model.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "default ONNX model"):
                write_build_info(bundle, "cpu", root)
            self.assertFalse((bundle / "BUILD-INFO.json").exists())

            model.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                write_build_info(bundle, "cpu", root)
            model.write_bytes(b"test onnx model")
            with mock.patch(
                "scripts.write_build_info.release_default_model_contract",
                return_value={**contract, "model_path": "../outside.onnx"},
            ), self.assertRaisesRegex(ValueError, "unsafe model_path"):
                write_build_info(bundle, "cpu", root)
            with mock.patch(
                "scripts.write_build_info.release_default_model_contract",
                return_value={**contract, "input_shape_hw": [True, 416]},
            ), self.assertRaisesRegex(ValueError, "invalid.*shape"):
                write_build_info(bundle, "cpu", root)
            with mock.patch(
                "scripts.write_build_info.release_default_model_contract",
                return_value={**contract, "detail_crop_size_source_pixels": True},
            ), self.assertRaisesRegex(ValueError, "invalid detail workload"):
                write_build_info(bundle, "cpu", root)

            model.unlink()
            model.symlink_to(root / "outside.onnx")
            (root / "outside.onnx").write_bytes(b"outside")
            with self.assertRaisesRegex(ValueError, "symlink"):
                write_build_info(bundle, "cpu", root)

    def test_dependency_manifest_is_hash_bound_and_variant_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._make_bundle(root)
            manifest = self._make_dependency_manifest(bundle)
            target = write_build_info(
                bundle, "directml", root, dependency_manifest=manifest
            )
            payload = json.loads(target.read_text(encoding="utf-8"))
            record = payload["dependency_manifest"]
            self.assertEqual(record["path"], "DEPENDENCY-MANIFEST.json")
            self.assertEqual(record["lock_profile"], "windows-directml-py313")
            self.assertEqual(record["distribution_count"], 1)
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")

            with self.assertRaisesRegex(ValueError, "runtime variant"):
                write_build_info(bundle, "cuda", root, dependency_manifest=manifest)
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_payload["lock_profile"] = "windows-cuda-py313"
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "profile"):
                write_build_info(bundle, "directml", root, dependency_manifest=manifest)
            manifest_payload["lock_profile"] = "windows-directml-py313"
            manifest_payload["distributions"][0]["installed_files"] = None
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "installed-file verification"):
                write_build_info(bundle, "directml", root, dependency_manifest=manifest)
            manifest_payload["distributions"][0]["installed_files"] = {
                "aggregate_sha256": "a" * 64,
                "record_document_sha256": "b" * 64,
                "record_entry_count": 2,
                "record_sha256_entries_verified": 1,
                "total_size_bytes": 0,
                "unhashed_record_entries": 1,
            }
            manifest_payload["distributions"][0]["installed_record_sha256"] = "b" * 64
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "installed-file count"):
                write_build_info(bundle, "directml", root, dependency_manifest=manifest)
            manifest_payload["distributions"][0]["installed_files"]["total_size_bytes"] = 1
            manifest_payload["distributions"][0]["installed_files"][
                "unhashed_record_entries"
            ] = True
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "installed-file count"):
                write_build_info(bundle, "directml", root, dependency_manifest=manifest)
            manifest.write_text(
                '{"schema_version":1,"schema_version":1}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                write_build_info(bundle, "directml", root, dependency_manifest=manifest)
            outside = root / "DEPENDENCY-MANIFEST.json"
            outside.write_bytes(manifest.read_bytes())
            with self.assertRaisesRegex(ValueError, "adjacent bundle file"):
                write_build_info(bundle, "directml", root, dependency_manifest=outside)

    def test_windows_executable_and_cli_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._make_bundle(root, executable_name="ProAim.exe")
            manifest = self._make_dependency_manifest(bundle)
            with mock.patch.dict(os.environ, {"GITHUB_SHA": "feedface"}):
                result = main(
                    [
                        "--bundle",
                        str(bundle),
                        "--runtime-variant",
                        "directml",
                        "--project-root",
                        str(root),
                        "--dependency-manifest",
                        str(manifest),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertTrue((bundle / "BUILD-INFO.json").is_file())


if __name__ == "__main__":
    unittest.main()
