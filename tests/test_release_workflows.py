from __future__ import annotations

from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = PROJECT_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOW_DIR / "ci.yml"
RELEASE_WORKFLOW = WORKFLOW_DIR / "release-bundles.yml"
WINDOWS_WORKFLOW = WORKFLOW_DIR / "build-windows.yml"
LINUX_BUILD = PROJECT_ROOT / "scripts" / "build_linux_app.sh"
WINDOWS_BUILD = PROJECT_ROOT / "scripts" / "build_windows_app.ps1"

RUNTIME_REQUIREMENTS = {
    "cpu": "onnxruntime==1.28.0",
    "cuda": "onnxruntime-gpu[cuda,cudnn]==1.28.0",
    "directml": "onnxruntime-directml==1.24.4",
    "rocm": "onnxruntime-rocm==1.22.2.post3",
}

RELEASE_ARCHIVES = (
    "ProAim-Linux-x64.zip",
    "ProAim-Windows-x64-DirectML.zip",
)

CUDA_ARCHIVES = (
    "ProAim-Linux-x64-NVIDIA-CUDA.zip",
    "ProAim-Windows-x64-NVIDIA-CUDA.zip",
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _job(source: str, name: str, next_name: str | None = None) -> str:
    start_marker = f"  {name}:\n"
    start = source.index(start_marker)
    if next_name is None:
        return source[start:]
    end = source.index(f"  {next_name}:\n", start + len(start_marker))
    return source[start:end]


class RuntimeRequirementContractTests(unittest.TestCase):
    def test_each_runtime_variant_is_exactly_pinned_and_isolated(self) -> None:
        actual_lines: list[str] = []
        for variant, expected in RUNTIME_REQUIREMENTS.items():
            path = PROJECT_ROOT / f"requirements-runtime-{variant}.txt"
            lines = [
                line.strip()
                for line in _source(path).splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            with self.subTest(variant=variant):
                self.assertEqual(lines, [expected])
            actual_lines.extend(lines)
        self.assertEqual(len(actual_lines), len(set(actual_lines)))

    def test_base_build_requirements_do_not_install_a_runtime_variant(self) -> None:
        source = _source(PROJECT_ROOT / "requirements-build.txt")
        package_lines = [
            line.strip().lower()
            for line in source.splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "-r"))
        ]
        self.assertFalse(any(line.startswith("onnxruntime") for line in package_lines))


class ReleaseWorkflowContractTests(unittest.TestCase):
    def test_ci_tests_linux_cpu_and_windows_directml(self) -> None:
        source = _source(CI_WORKFLOW)
        self.assertIn("branches: [main]", source)
        self.assertIn("os: ubuntu-24.04\n            runtime_variant: cpu", source)
        self.assertIn("os: windows-2022\n            runtime_variant: directml", source)
        self.assertIn(
            "python -m pip install -r requirements-runtime-${{ matrix.runtime_variant }}.txt",
            source,
        )
        self.assertNotIn("shell: bash", source)
        self.assertIn("python -m unittest discover -s tests", source)
        self.assertIn("python scripts/validate_release_assets.py", source)
        self.assertIn("python app.py --cli --help", source)
        self.assertIn("python app.py --runtime-info", source)
        self.assertIn("QT_QPA_PLATFORM: offscreen", source)

    def test_release_builds_are_gated_by_cross_platform_tests(self) -> None:
        source = _source(RELEASE_WORKFLOW)
        test_job = _job(source, "test", "build-linux")
        linux_job = _job(source, "build-linux", "build-windows")
        windows_job = _job(source, "build-windows", "publish-release")
        self.assertIn("ubuntu-24.04", test_job)
        self.assertIn("windows-2022", test_job)
        self.assertIn("python -m unittest discover -s tests", test_job)
        self.assertRegex(linux_job, r"(?m)^    needs: test$")
        self.assertIn("runs-on: ubuntu-22.04", linux_job)
        self.assertIn("--max-glibc 2.35", linux_job)
        self.assertRegex(windows_job, r"(?m)^    needs: test$")
        self.assertIn("dist/ProAim/ProAim --cli --help", linux_job)
        self.assertIn("dist/ProAim/ProAim --runtime-info", linux_job)
        self.assertIn("./dist/ProAim/ProAimCLI.exe --cli --help", windows_job)
        self.assertIn("./dist/ProAim/ProAimCLI.exe --runtime-info", windows_job)
        self.assertIn('zip -yr "${{ matrix.zip_name }}" ProAim', linux_job)
        self.assertIn("runtime_variant: cpu", linux_job)
        self.assertIn("runtime_variant: directml", windows_job)
        self.assertNotIn("runtime_variant: cuda", linux_job + windows_job)

    def test_linux_helper_allows_only_driver_library_for_cuda_bundle(self) -> None:
        source = _source(LINUX_BUILD)
        self.assertIn('[[ "$RUNTIME_VARIANT" == "cuda" ]]', source)
        self.assertIn("VALIDATE_BUNDLE_ARGS+=(--allow-missing libcuda.so.1)", source)

    def test_manual_release_run_cannot_publish_without_a_version_tag(self) -> None:
        source = _source(RELEASE_WORKFLOW)
        publish_job = _job(source, "publish-release")
        before_publish = source[: source.index("  publish-release:\n")]
        self.assertIn("workflow_dispatch:", source)
        self.assertIn("permissions:\n  contents: read", before_publish)
        self.assertNotIn("contents: write", before_publish)
        self.assertIn(
            "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')",
            publish_job,
        )
        self.assertIn("permissions:\n      contents: write", publish_job)
        self.assertIn("tag_name: ${{ github.ref_name }}", publish_job)

    def test_release_publishes_every_archive_and_a_checksum_manifest(self) -> None:
        source = _source(RELEASE_WORKFLOW)
        publish_job = _job(source, "publish-release")
        for archive in RELEASE_ARCHIVES:
            with self.subTest(archive=archive):
                self.assertGreaterEqual(source.count(archive), 2)
                self.assertIn(f"release-assets/{archive}", publish_job)
        checksum = "sha256sum *.zip > SHA256SUMS.txt"
        self.assertIn(checksum, publish_job)
        self.assertIn("release-assets/SHA256SUMS.txt", publish_job)
        self.assertIn("fail_on_unmatched_files: true", publish_job)
        self.assertLess(publish_job.index(checksum), publish_job.index("softprops/action-gh-release"))

    def test_tagged_release_excludes_cuda_until_a_real_nvidia_gate_exists(self) -> None:
        release = _source(RELEASE_WORKFLOW)
        manual_windows = _source(WINDOWS_WORKFLOW)
        for archive in CUDA_ARCHIVES:
            with self.subTest(archive=archive):
                self.assertNotIn(archive, release)
        self.assertNotIn("runtime_variant: cuda", release)
        self.assertIn("runtime_variant: cuda", manual_windows)
        self.assertIn("ProAim-Windows-x64-NVIDIA-CUDA", manual_windows)

    def test_windows_manual_build_tests_before_packaging_and_upload(self) -> None:
        source = _source(WINDOWS_WORKFLOW)
        self.assertIn("timeout-minutes: 60", source)
        self.assertIn("runtime_variant: cuda", source)
        self.assertIn("runtime_variant: directml", source)
        test_index = source.index("python -m unittest discover -s tests")
        build_index = source.index("./scripts/build_windows_app.ps1")
        help_smoke_index = source.index("./dist/ProAim/ProAimCLI.exe --cli --help")
        runtime_smoke_index = source.index("./dist/ProAim/ProAimCLI.exe --runtime-info")
        upload_index = source.index("actions/upload-artifact")
        self.assertLess(test_index, build_index)
        self.assertLess(build_index, help_smoke_index)
        self.assertLess(help_smoke_index, runtime_smoke_index)
        self.assertLess(runtime_smoke_index, upload_index)

    def test_build_info_is_written_before_archives_are_created(self) -> None:
        linux = _source(LINUX_BUILD)
        windows = _source(WINDOWS_BUILD)
        self.assertIn("scripts/write_build_info.py", linux)
        self.assertIn("scripts\\write_build_info.py", windows)
        # Linux is archived by the workflow after its helper returns. Windows
        # creates the ZIP itself, so metadata must be written first.
        self.assertLess(
            windows.index("$BuildInfo"),
            windows.index("Compress-Archive"),
        )

    def test_workflow_expressions_are_not_accidentally_shell_expansions(self) -> None:
        expression = re.compile(r"\$\{\{[^{}]+\}\}")
        for path in (CI_WORKFLOW, RELEASE_WORKFLOW, WINDOWS_WORKFLOW):
            with self.subTest(path=path.name):
                source = _source(path)
                self.assertNotIn("${{ }}", source)
                self.assertGreater(len(expression.findall(source)), 0)

    def test_powershell_native_failures_cannot_be_masked(self) -> None:
        for path in (CI_WORKFLOW, RELEASE_WORKFLOW, WINDOWS_WORKFLOW):
            with self.subTest(path=path.name):
                source = _source(path)
                pwsh_blocks = re.findall(
                    r"shell: pwsh\n(?:\s+[^\n]+\n)*?\s+run: \|\n((?:\s{10,}.*\n)+)",
                    source,
                )
                self.assertTrue(pwsh_blocks)
                for block in pwsh_blocks:
                    self.assertIn(
                        "$PSNativeCommandUseErrorActionPreference = $true",
                        block,
                    )


if __name__ == "__main__":
    unittest.main()
