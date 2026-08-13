from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.write_build_info import _git_value, main, write_build_info


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
            self.assertEqual(clean["schema"], 1)
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

    def test_windows_executable_and_cli_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._make_bundle(root, executable_name="ProAim.exe")
            with mock.patch.dict(os.environ, {"GITHUB_SHA": "feedface"}):
                result = main(
                    [
                        "--bundle",
                        str(bundle),
                        "--runtime-variant",
                        "directml",
                        "--project-root",
                        str(root),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertTrue((bundle / "BUILD-INFO.json").is_file())


if __name__ == "__main__":
    unittest.main()
