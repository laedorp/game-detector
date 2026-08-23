from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.write_dependency_manifest import (
    DependencyContractError,
    LOCK_PROFILES,
    LockedDistribution,
    LockProfile,
    build_manifest,
    load_artifact_reports,
    load_profile_lock,
    parse_hashed_lock,
    run_pip_check,
    validate_declared_requirements,
    verify_distribution_record,
    verify_installed_payload_coverage,
    verify_installed_set,
    write_manifest,
)


class FakeDistribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.version = version

    def read_text(self, filename: str) -> str | None:
        if filename == "METADATA":
            return f"Metadata-Version: 2.4\nName: {self.metadata['Name']}\nVersion: {self.version}\n"
        if filename == "RECORD":
            return f"{self.metadata['Name']}/__init__.py,sha256=example,1\n"
        return None


class RecordDistribution:
    def __init__(self, root: Path, name: str = "example", version: str = "1.0") -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.base = root / "lib" / "site-packages"
        self.info = self.base / f"{name}-{version}.dist-info"
        self.payload = self.base / name / "payload.bin"
        self.info.mkdir(parents=True)
        self.payload.parent.mkdir(parents=True)
        self.payload.write_bytes(b"reviewed payload\n")
        (self.info / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
        self.write_record()

    @staticmethod
    def _record_hash(path: Path) -> str:
        encoded = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
        return "sha256=" + encoded.rstrip(b"=").decode("ascii")

    def write_record(self, extra_rows: list[list[str]] | None = None) -> None:
        rows: list[list[str]] = []
        for path in (self.payload, self.info / "METADATA"):
            relative = path.relative_to(self.base).as_posix()
            rows.append([relative, self._record_hash(path), str(path.stat().st_size)])
        rows.extend(extra_rows or [])
        rows.append([(self.info / "RECORD").relative_to(self.base).as_posix(), "", ""])
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        (self.info / "RECORD").write_text(stream.getvalue(), encoding="utf-8")

    def read_text(self, filename: str) -> str | None:
        path = self.info / filename
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def locate_file(self, path: str) -> Path:
        return self.base / path


def _profile(root_name: str = "test") -> LockProfile:
    return LockProfile(
        name=root_name,
        system="Linux",
        machines=("x86_64",),
        python_version="3.13.14",
        runtime_variant="cpu",
        runtime_distribution="onnxruntime",
        bootstrap_lock_path="bootstrap.txt",
        platform_lock_path="platform.txt",
        requirements_paths=("requirements.txt", "requirements-runtime-cpu.txt"),
    )


def _identity(**overrides: str) -> dict[str, str]:
    values = {
        "implementation": "CPython",
        "machine": "x86_64",
        "python_version": "3.13.14",
        "system": "Linux",
    }
    values.update(overrides)
    return values


def _report_entry(name: str, version: str, sha: str) -> dict[str, object]:
    return {
        "download_info": {
            "archive_info": {"hashes": {"sha256": sha}},
            "url": f"https://files.pythonhosted.org/packages/{name}-{version}.whl",
        },
        "is_direct": False,
        "is_yanked": False,
        "metadata": {"name": name, "version": version},
    }


def _locked(name: str, version: str, sha: str) -> LockedDistribution:
    return LockedDistribution(name=name, version=version, hashes=frozenset({sha}))


def _write_report(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "environment": {
                    "implementation_name": "cpython",
                    "platform_machine": "x86_64",
                    "platform_system": "Linux",
                    "python_full_version": "3.13.14",
                },
                "install": entries,
                "pip_version": "26.2.1",
                "version": "1",
            }
        ),
        encoding="utf-8",
    )


class ReleaseLockContractTests(unittest.TestCase):
    def test_parser_accepts_only_unique_unconditional_exact_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.txt"
            valid.write_text(
                "# comment\n"
                f"Foo_Bar==1.2.3 --hash=sha256:{'a' * 64}\n"
                f"baz==4 --hash=sha256:{'b' * 64}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_hashed_lock(valid),
                {
                    "foo-bar": _locked("Foo_Bar", "1.2.3", "a" * 64),
                    "baz": _locked("baz", "4", "b" * 64),
                },
            )
            extras = root / "extras.txt"
            extras.write_text(
                f"foo[cuda,cudnn]==1 --hash=sha256:{'c' * 64}\n", encoding="utf-8"
            )
            self.assertEqual(
                parse_hashed_lock(extras)["foo"].extras,
                frozenset({"cuda", "cudnn"}),
            )
            for name, content, pattern in (
                ("range", f"foo>=1 --hash=sha256:{'a' * 64}\n", "name==version"),
                (
                    "marker",
                    f'foo==1;sys_platform=="linux" --hash=sha256:{"a" * 64}\n',
                    "name==version",
                ),
                (
                    "duplicate",
                    f"foo==1 --hash=sha256:{'a' * 64}\nFoo==1 --hash=sha256:{'a' * 64}\n",
                    "duplicate",
                ),
                ("include", "-r other.txt\n", "forbidden"),
                ("unhashed", "foo==1\n", "requires an artifact hash"),
            ):
                candidate = root / f"{name}.txt"
                candidate.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(DependencyContractError, pattern):
                    parse_hashed_lock(candidate)

    def test_repository_profiles_pin_toolchain_runtime_and_cuda_transitives(self) -> None:
        project = Path(__file__).resolve().parents[1]
        expected_nvidia = {
            "nvidia-cublas": "13.6.0.2",
            "nvidia-cuda-nvrtc": "13.3.33",
            "nvidia-cuda-runtime": "13.3.29",
            "nvidia-cudnn-cu13": "9.24.0.43",
            "nvidia-cufft": "12.3.0.29",
            "nvidia-curand": "10.4.3.29",
            "nvidia-nvjitlink": "13.3.33",
        }
        for profile in LOCK_PROFILES.values():
            pins = load_profile_lock(project, profile)
            self.assertEqual(pins["pip"].version, "26.2.1")
            self.assertEqual(pins["setuptools"].version, "84.0.0")
            self.assertEqual(pins["wheel"].version, "0.48.0")
            self.assertTrue(pins["pip"].hashes)
            self.assertEqual(pins[profile.runtime_distribution].version, {
                "onnxruntime": "1.28.0",
                "onnxruntime-directml": "1.24.4",
                "onnxruntime-gpu": "1.28.0",
            }[profile.runtime_distribution])
            self.assertEqual(
                set(pins).intersection(
                    {"onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-rocm"}
                ),
                {profile.runtime_distribution},
            )
            declared = validate_declared_requirements(project, profile, pins)
            self.assertTrue(declared)
            self.assertIn(
                profile.runtime_distribution,
                {record["canonical_name"] for record in declared},
            )
        cuda = load_profile_lock(project, LOCK_PROFILES["windows-cuda-py313"])
        self.assertEqual(
            {name: cuda[name].version for name in expected_nvidia}, expected_nvidia
        )
        self.assertEqual(cuda["onnxruntime-gpu"].extras, frozenset({"cuda", "cudnn"}))

    def test_active_declared_requirements_must_match_lock_versions_and_extras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile()
            (root / "requirements.txt").write_text(
                "common>=1,<2\n"
                'win-only==9; sys_platform == "win32"\n',
                encoding="utf-8",
            )
            (root / "requirements-runtime-cpu.txt").write_text(
                "onnxruntime[example]==1.28.0\n", encoding="utf-8"
            )
            locked = {
                "common": _locked("common", "1.5", "a" * 64),
                "onnxruntime": LockedDistribution(
                    "onnxruntime", "1.28.0", frozenset({"b" * 64}), frozenset({"example"})
                ),
            }
            records = validate_declared_requirements(root, profile, locked)
            self.assertEqual(
                [record["canonical_name"] for record in records],
                ["common", "onnxruntime"],
            )
            self.assertEqual(records[1]["declared_extras"], ["example"])

            bad_extra = dict(locked)
            bad_extra["onnxruntime"] = _locked("onnxruntime", "1.28.0", "b" * 64)
            with self.assertRaisesRegex(DependencyContractError, "omits declared extras"):
                validate_declared_requirements(root, profile, bad_extra)
            bad_version = dict(locked)
            bad_version["common"] = _locked("common", "2.0", "a" * 64)
            with self.assertRaisesRegex(DependencyContractError, "does not satisfy"):
                validate_declared_requirements(root, profile, bad_version)


class ArtifactReportTests(unittest.TestCase):
    def test_reports_account_for_every_lock_with_pypi_sha256(self) -> None:
        locked = {
            "pip": _locked("pip", "26.2.1", "a" * 64),
            "onnxruntime": _locked("onnxruntime", "1.28.0", "b" * 64),
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "pip-report.json"
            _write_report(
                report,
                [
                    _report_entry("pip", "26.2.1", "a" * 64),
                    _report_entry("onnxruntime", "1.28.0", "b" * 64),
                ],
            )
            artifacts, reports = load_artifact_reports([report], _profile(), locked)
            self.assertEqual(artifacts["onnxruntime"]["sha256"], "b" * 64)
            self.assertEqual(reports[0]["filename"], "pip-report.json")
            self.assertEqual(reports[0]["purpose"], "final-environment-install")
            self.assertRegex(reports[0]["sha256"], r"^[0-9a-f]{64}$")

    def test_reports_reject_missing_hash_wrong_source_direct_and_host_drift(self) -> None:
        locked = {"onnxruntime": _locked("onnxruntime", "1.28.0", "a" * 64)}
        mutations = (
            ("missing", lambda entry: entry["download_info"]["archive_info"].update({"hashes": {}}), "SHA-256"),
            ("direct", lambda entry: entry.update({"is_direct": True}), "non-yanked"),
            (
                "host",
                lambda entry: entry["download_info"].update({"url": "https://example.invalid/pkg.whl"}),
                "files.pythonhosted",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename, mutate, pattern in mutations:
                entry = _report_entry("onnxruntime", "1.28.0", "a" * 64)
                mutate(entry)
                report = root / f"{filename}.json"
                _write_report(report, [entry])
                with self.assertRaisesRegex(DependencyContractError, pattern):
                    load_artifact_reports([report], _profile(), locked)

    def test_report_platform_and_complete_coverage_are_required(self) -> None:
        locked = {
            "pip": _locked("pip", "26.2.1", "a" * 64),
            "onnxruntime": _locked("onnxruntime", "1.28.0", "b" * 64),
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            _write_report(report, [_report_entry("pip", "26.2.1", "a" * 64)])
            with self.assertRaisesRegex(DependencyContractError, "final pip report"):
                load_artifact_reports([report], _profile(), locked)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["environment"]["python_full_version"] = "3.13.15"
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DependencyContractError, "python_full_version"):
                load_artifact_reports([report], _profile(), {"pip": locked["pip"]})

    def test_last_report_must_be_complete_unique_and_created_by_locked_pip(self) -> None:
        locked = {
            "pip": _locked("pip", "26.2.1", "a" * 64),
            "onnxruntime": _locked("onnxruntime", "1.28.0", "b" * 64),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bootstrap = root / "bootstrap.json"
            final = root / "final.json"
            _write_report(bootstrap, [_report_entry("pip", "26.2.1", "a" * 64)])
            bootstrap_payload = json.loads(bootstrap.read_text(encoding="utf-8"))
            bootstrap_payload["pip_version"] = "26.1.2"
            bootstrap.write_text(json.dumps(bootstrap_payload), encoding="utf-8")
            _write_report(
                final,
                [
                    _report_entry("pip", "26.2.1", "a" * 64),
                    _report_entry("onnxruntime", "1.28.0", "b" * 64),
                ],
            )
            _, records = load_artifact_reports([bootstrap, final], _profile(), locked)
            self.assertEqual(
                [record["purpose"] for record in records],
                ["bootstrap", "final-environment-install"],
            )

            with self.assertRaisesRegex(DependencyContractError, "path must be unique"):
                load_artifact_reports([final, final], _profile(), locked)

            payload = json.loads(final.read_text(encoding="utf-8"))
            payload["pip_version"] = "26.2"
            final.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DependencyContractError, "produced by pip"):
                load_artifact_reports([bootstrap, final], _profile(), locked)

            payload["pip_version"] = "26.2.1"
            payload["install"].append(_report_entry("pip", "26.2.1", "a" * 64))
            final.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DependencyContractError, "duplicate distribution"):
                load_artifact_reports([bootstrap, final], _profile(), locked)


class InstalledEnvironmentTests(unittest.TestCase):
    def test_record_payloads_are_verified_and_aggregated_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            distribution = RecordDistribution(root)
            first, files = verify_distribution_record(distribution, root)
            second, _ = verify_distribution_record(distribution, root)
            self.assertEqual(first, second)
            self.assertEqual(first["record_entry_count"], 3)
            self.assertEqual(first["record_sha256_entries_verified"], 2)
            self.assertEqual(first["unhashed_record_entries"], 1)
            self.assertRegex(first["aggregate_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(len(files), 3)

            injected = distribution.base / "example" / "injected.py"
            injected.write_text("raise RuntimeError('unrecorded')\n", encoding="utf-8")
            with self.assertRaisesRegex(DependencyContractError, "unrecorded"):
                verify_installed_payload_coverage([distribution], root, {path: "example" for path in files})
            injected.unlink()

            distribution.payload.write_bytes(b"mutated after pip install\n")
            with self.assertRaisesRegex(DependencyContractError, "differs from RECORD"):
                verify_distribution_record(distribution, root)

    def test_record_rejects_unhashed_payload_and_environment_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            distribution = RecordDistribution(root)
            unhashed = distribution.base / "example" / "unhashed.pyc"
            unhashed.write_bytes(b"untrusted")
            distribution.write_record(
                [[unhashed.relative_to(distribution.base).as_posix(), "", ""]]
            )
            with self.assertRaisesRegex(DependencyContractError, "unhashed non-RECORD"):
                verify_distribution_record(distribution, root)

            distribution.write_record(
                [["C:/outside/injected.pth", "sha256=" + "a" * 43, "1"]]
            )
            with self.assertRaisesRegex(DependencyContractError, "unsafe path"):
                verify_distribution_record(distribution, root)

            outside = root.parent / f"{root.name}-outside.bin"
            outside.write_bytes(b"outside")
            try:
                distribution.write_record(
                    [[
                        Path("..", "..", "..", outside.name).as_posix(),
                        RecordDistribution._record_hash(outside),
                        str(outside.stat().st_size),
                    ]]
                )
                with self.assertRaisesRegex(DependencyContractError, "escapes"):
                    verify_distribution_record(distribution, root)
            finally:
                outside.unlink()

    def test_exact_set_versions_and_single_runtime_are_required(self) -> None:
        locked = {
            "pip": _locked("pip", "26.2.1", "a" * 64),
            "onnxruntime": _locked("onnxruntime", "1.28.0", "b" * 64),
        }
        _, records = verify_installed_set(
            locked,
            "onnxruntime",
            [FakeDistribution("pip", "26.2.1"), FakeDistribution("onnxruntime", "1.28.0")],
            verify_installed_files=False,
        )
        self.assertEqual([record["canonical_name"] for record in records], ["onnxruntime", "pip"])
        self.assertRegex(records[0]["installed_metadata_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(records[0]["installed_record_sha256"], r"^[0-9a-f]{64}$")

        bad_sets = (
            ([FakeDistribution("pip", "26.2.1")], "missing"),
            (
                [
                    FakeDistribution("pip", "26.2.1"),
                    FakeDistribution("onnxruntime", "1.27.0"),
                ],
                "version_mismatch",
            ),
            (
                [
                    FakeDistribution("pip", "26.2.1"),
                    FakeDistribution("onnxruntime", "1.28.0"),
                    FakeDistribution("extra", "1"),
                ],
                "unlisted",
            ),
        )
        for distributions, pattern in bad_sets:
            with self.assertRaisesRegex(DependencyContractError, pattern):
                verify_installed_set(
                    locked,
                    "onnxruntime",
                    distributions,
                    verify_installed_files=False,
                )

    def test_pip_check_failure_is_a_contract_error(self) -> None:
        completed = subprocess.CompletedProcess(
            ["python", "-m", "pip", "check"], 1, stdout="broken requirement\n", stderr=""
        )
        with mock.patch(
            "scripts.write_dependency_manifest.subprocess.run", return_value=completed
        ):
            with self.assertRaisesRegex(DependencyContractError, "broken requirement"):
                run_pip_check()


class ManifestOutputTests(unittest.TestCase):
    def test_manifest_binds_inputs_reports_artifacts_and_installed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bootstrap.txt").write_text(
                f"pip==26.2.1 --hash=sha256:{'a' * 64}\n", encoding="utf-8"
            )
            (root / "platform.txt").write_text(
                f"onnxruntime==1.28.0 --hash=sha256:{'b' * 64}\n", encoding="utf-8"
            )
            (root / "requirements.txt").write_text("onnxruntime>=1\n", encoding="utf-8")
            (root / "requirements-runtime-cpu.txt").write_text(
                "onnxruntime==1.28.0\n", encoding="utf-8"
            )
            report = root / "dependencies.json"
            _write_report(
                report,
                [
                    _report_entry("pip", "26.2.1", "a" * 64),
                    _report_entry("onnxruntime", "1.28.0", "b" * 64),
                ],
            )
            payload = build_manifest(
                profile=_profile(),
                project_root=root,
                report_paths=[report],
                distributions=[
                    FakeDistribution("pip", "26.2.1"),
                    FakeDistribution("onnxruntime", "1.28.0"),
                ],
                identity=_identity(),
                check_dependencies=False,
                verify_installed_files=False,
            )
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["lock_profile"], "test")
            self.assertTrue(payload["artifact_hash_contract"]["enforced_before_install"])
            by_name = {item["canonical_name"]: item for item in payload["distributions"]}
            self.assertEqual(by_name["onnxruntime"]["artifact"]["sha256"], "b" * 64)
            self.assertEqual(
                [item["path"] for item in payload["inputs"]],
                [
                    "bootstrap.txt",
                    "platform.txt",
                    "requirements.txt",
                    "requirements-runtime-cpu.txt",
                ],
            )
            self.assertEqual(
                [item["canonical_name"] for item in payload["declared_requirements"]],
                ["onnxruntime"],
            )
            output = write_manifest(payload, root / "out" / "DEPENDENCY-MANIFEST.json")
            first = output.read_bytes()
            write_manifest(payload, output)
            self.assertEqual(output.read_bytes(), first)
            self.assertTrue(first.endswith(b"\n"))

    def test_wrong_python_patch_fails_before_manifest_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(DependencyContractError, "expected 3.13.14"):
                build_manifest(
                    profile=_profile(),
                    project_root=Path(temporary),
                    report_paths=[],
                    identity=_identity(python_version="3.13.13"),
                    check_dependencies=False,
                )


if __name__ == "__main__":
    unittest.main()
