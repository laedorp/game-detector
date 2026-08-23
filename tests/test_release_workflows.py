from __future__ import annotations

from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = PROJECT_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOW_DIR / "ci.yml"
RELEASE_WORKFLOW = WORKFLOW_DIR / "release-bundles.yml"
WINDOWS_WORKFLOW = WORKFLOW_DIR / "build-windows.yml"
CUDA_ATTACHMENT_WORKFLOW = WORKFLOW_DIR / "attach-qualified-cuda.yml"
CUDA_QUALIFICATION_WORKFLOW = WORKFLOW_DIR / "qualify-windows-cuda.yml"
DIRECTML_QUALIFICATION_WORKFLOW = WORKFLOW_DIR / "qualify-windows-directml.yml"
DIRECTML_PUBLICATION_WORKFLOW = WORKFLOW_DIR / "publish-qualified-directml-release.yml"
INDEPENDENT_HOLDOUT_WORKFLOW = WORKFLOW_DIR / "qualify-independent-holdout.yml"
DIRECTML_TELEMETRY_HELPER = (
    PROJECT_ROOT / "packaging" / "windows" / "Record-ProAimDirectMlTelemetry.ps1"
)
DIRECTML_OBSERVATION_HELPER = (
    PROJECT_ROOT / "packaging" / "windows" / "Complete-ProAimDirectMlObservation.ps1"
)
CUDA_TELEMETRY_HELPER = (
    PROJECT_ROOT / "packaging" / "windows" / "Record-ProAimCudaTelemetry.ps1"
)
CUDA_OBSERVATION_HELPER = (
    PROJECT_ROOT / "packaging" / "windows" / "Complete-ProAimPhysicalObservation.ps1"
)
LINUX_BUILD = PROJECT_ROOT / "scripts" / "build_linux_app.sh"
WINDOWS_BUILD = PROJECT_ROOT / "scripts" / "build_windows_app.ps1"
GIT_ATTRIBUTES = PROJECT_ROOT / ".gitattributes"

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

PINNED_ACTIONS = {
    "actions/checkout": "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",  # v5
    "actions/setup-python": "ece7cb06caefa5fff74198d8649806c4678c61a1",  # v6
    "actions/upload-artifact": "330a01c490aca151604b8cf639adc76d48f6c5d4",  # v5
    "actions/download-artifact": "634f93cb2916e3fdff6788551b99b062d0335ce0",  # v5
    "softprops/action-gh-release": "3bb12739c298aeb8a4eeaf626c5b8d85266b0e65",  # v2
}


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
    def test_dependency_and_release_workflows_pin_actions_to_reviewed_commits(self) -> None:
        for path in (
            CI_WORKFLOW,
            RELEASE_WORKFLOW,
            WINDOWS_WORKFLOW,
            DIRECTML_QUALIFICATION_WORKFLOW,
            DIRECTML_PUBLICATION_WORKFLOW,
            INDEPENDENT_HOLDOUT_WORKFLOW,
        ):
            source = _source(path)
            uses = re.findall(r"(?m)^\s*-?\s*uses:\s*([^\s#]+)", source)
            self.assertTrue(uses)
            for action_reference in uses:
                with self.subTest(workflow=path.name, action=action_reference):
                    action, separator, revision = action_reference.rpartition("@")
                    self.assertTrue(separator)
                    self.assertIn(action, PINNED_ACTIONS)
                    self.assertEqual(revision, PINNED_ACTIONS[action])
                    self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_release_assets_have_cross_platform_checkout_bytes(self) -> None:
        source = _source(GIT_ATTRIBUTES)
        self.assertIn("* text=auto eol=lf", source)
        self.assertIn("*.bin binary", source)
        self.assertIn("*.onnx binary", source)

    def test_ci_tests_linux_cpu_and_windows_directml(self) -> None:
        source = _source(CI_WORKFLOW)
        self.assertIn("branches: [main]", source)
        self.assertIn("os: ubuntu-24.04\n            runtime_variant: cpu", source)
        self.assertIn("os: windows-2022\n            runtime_variant: directml", source)
        self.assertIn('python-version: "3.13.14"', source)
        self.assertIn("python -m venv .release-venv", source)
        self.assertIn("--require-hashes", source)
        self.assertIn("requirements-locks/bootstrap-py313.txt", source)
        self.assertIn("${{ matrix.lock_file }}", source)
        self.assertIn("scripts/write_dependency_manifest.py", source)
        self.assertNotIn("shell: bash", source)
        self.assertIn("python -m unittest discover -s tests", source)
        self.assertIn("python scripts/validate_release_assets.py", source)
        self.assertIn("python app.py --cli --help", source)
        self.assertIn("python app.py --runtime-info", source)
        self.assertIn("QT_QPA_PLATFORM: offscreen", source)
        self.assertIn("if: runner.os == 'Linux'", source)
        self.assertIn("sudo apt-get install -y libegl1 libgl1", source)
        self.assertIn("timeout-minutes: 30", source)
        self.assertIn("- name: Compile Python sources", source)
        self.assertIn("- name: Run unit tests", source)
        self.assertIn("- name: Validate release assets", source)
        self.assertIn("- name: Smoke-test source CLI", source)
        self.assertIn("- name: Report source runtime", source)
        self.assertNotIn("- name: Compile and test", source)

    def test_release_builds_are_gated_by_cross_platform_tests(self) -> None:
        source = _source(RELEASE_WORKFLOW)
        test_job = _job(source, "test", "build-linux")
        linux_job = _job(source, "build-linux", "build-windows")
        windows_job = _job(source, "build-windows", "stage-release-candidate")
        self.assertIn("ubuntu-24.04", test_job)
        self.assertIn("windows-2022", test_job)
        self.assertIn("python -m unittest discover -s tests", test_job)
        self.assertRegex(linux_job, r"(?m)^    needs: test$")
        self.assertIn("runs-on: ubuntu-22.04", linux_job)
        self.assertIn("--max-glibc 2.35", linux_job)
        self.assertRegex(windows_job, r"(?m)^    needs: test$")
        self.assertIn("dist/ProAim/ProAim --cli --help", linux_job)
        self.assertIn("dist/ProAim/ProAim --runtime-info", linux_job)
        self.assertIn("python scripts/smoke_release_default_model.py", linux_job)
        self.assertIn("./dist/ProAim/ProAimCLI.exe --cli --help", windows_job)
        self.assertIn("./dist/ProAim/ProAimCLI.exe --runtime-info", windows_job)
        self.assertIn("python scripts/smoke_release_default_model.py", windows_job)
        for job in (linux_job, windows_job):
            self.assertIn("--bundle dist/ProAim", job)
            self.assertIn("--device CPU", job)
            self.assertNotIn("--preset fort-416-fp32", job)
        self.assertIn('zip -yr "${{ matrix.zip_name }}" ProAim', linux_job)
        self.assertIn("runtime_variant: cpu", linux_job)
        self.assertIn("runtime_variant: directml", windows_job)
        self.assertNotIn("runtime_variant: cuda", linux_job + windows_job)
        # PySide6 links libEGL even with QT_QPA_PLATFORM=offscreen. Keep the
        # source-test and oldest-supported Linux build runners explicit.
        self.assertIn("sudo apt-get install -y libegl1 libgl1", test_job)
        self.assertIn("sudo apt-get install -y --no-install-recommends", linux_job)
        for package in (
            "libegl1",
            "libgl1",
            "libxcb-cursor0",
            "libxcb-icccm4",
            "libxcb-keysyms1",
            "libxcb-shape0",
            "libxkbcommon-x11-0",
            "ocl-icd-libopencl1",
        ):
            with self.subTest(package=package):
                self.assertIn(package, linux_job)
        self.assertIn("timeout-minutes: 30", test_job)
        self.assertIn("- name: Compile Python sources", test_job)
        self.assertIn("- name: Run unit tests", test_job)
        self.assertIn("- name: Validate release assets", test_job)
        self.assertIn("- name: Smoke-test source CLI", test_job)
        self.assertIn("- name: Report source runtime", test_job)
        self.assertNotIn("- name: Run unit and release-contract tests", test_job)

    def test_linux_helper_allows_only_driver_library_for_cuda_bundle(self) -> None:
        source = _source(LINUX_BUILD)
        self.assertIn('[[ "$RUNTIME_VARIANT" == "cuda" ]]', source)
        self.assertIn("VALIDATE_BUNDLE_ARGS+=(--allow-missing libcuda.so.1)", source)

    def test_release_dependency_installs_are_fresh_exact_and_hash_locked(self) -> None:
        profiles = {
            CI_WORKFLOW: (
                "requirements-locks/linux-cpu-py313.txt",
                "requirements-locks/windows-directml-py313.txt",
            ),
            RELEASE_WORKFLOW: (
                "requirements-locks/linux-cpu-py313.txt",
                "requirements-locks/windows-directml-py313.txt",
            ),
            WINDOWS_WORKFLOW: (
                "requirements-locks/windows-directml-py313.txt",
                "requirements-locks/windows-cuda-py313.txt",
            ),
        }
        for path, expected_locks in profiles.items():
            with self.subTest(workflow=path.name):
                source = _source(path)
                self.assertIn('python-version: "3.13.14"', source)
                self.assertIn("python -m venv .release-venv", source)
                self.assertIn("--require-hashes", source)
                self.assertIn("--force-reinstall", source)
                self.assertIn("--no-compile", source)
                self.assertIn("--no-build-isolation", source)
                self.assertIn('PYTHONDONTWRITEBYTECODE: "1"', source)
                self.assertIn("requirements-locks/bootstrap-py313.txt", source)
                self.assertIn("--report", source)
                self.assertIn("scripts/write_dependency_manifest.py", source)
                self.assertNotIn("python -m pip install --upgrade pip", source)
                for lock in expected_locks:
                    self.assertIn(lock, source)

    def test_release_build_helpers_fail_closed_and_embed_dependency_manifest(self) -> None:
        linux = _source(LINUX_BUILD)
        windows = _source(WINDOWS_BUILD)
        self.assertIn("linux-cpu-py313", linux)
        self.assertIn("scripts/write_dependency_manifest.py", linux)
        self.assertIn("DEPENDENCY-MANIFEST.json", linux)
        self.assertLess(
            linux.index("scripts/write_dependency_manifest.py"),
            linux.index("-m PyInstaller"),
        )
        for profile in ("windows-$RuntimeVariant-py313", "DEPENDENCY-MANIFEST.json"):
            self.assertIn(profile, windows)
        self.assertIn("scripts\\write_dependency_manifest.py", windows)
        self.assertLess(
            windows.index("scripts\\write_dependency_manifest.py"),
            windows.index("-m PyInstaller"),
        )

    def test_tag_and_manual_builds_cannot_publish_a_release(self) -> None:
        source = _source(RELEASE_WORKFLOW)
        stage_job = _job(source, "stage-release-candidate")
        self.assertIn("workflow_dispatch:", source)
        self.assertIn("permissions:\n  actions: read\n  contents: read", source)
        self.assertNotIn("contents: write", source)
        self.assertIn(
            "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')",
            stage_job,
        )
        self.assertNotIn("softprops/action-gh-release", source)
        self.assertNotIn("gh release", source)
        self.assertIn("Publication status: **not published**", stage_job)

    def test_tag_workflow_stages_both_archives_and_content_manifest(self) -> None:
        source = _source(RELEASE_WORKFLOW)
        stage_job = _job(source, "stage-release-candidate")
        for archive in RELEASE_ARCHIVES:
            with self.subTest(archive=archive):
                self.assertGreaterEqual(source.count(archive), 1)
        self.assertIn("scripts/manage_directml_release.py stage-candidate", stage_job)
        self.assertIn("RELEASE-CANDIDATE-MANIFEST.json", stage_job)
        self.assertIn("name: ProAim-Release-Candidate", stage_job)
        self.assertIn("retention-days: 30", stage_job)
        self.assertNotIn("SHA256SUMS.txt", stage_job)

    def test_tagged_release_excludes_cuda_until_a_real_nvidia_gate_exists(self) -> None:
        release = _source(RELEASE_WORKFLOW)
        manual_windows = _source(WINDOWS_WORKFLOW)
        for archive in CUDA_ARCHIVES:
            with self.subTest(archive=archive):
                self.assertNotIn(archive, release)
        self.assertNotIn("runtime_variant: cuda", release)
        self.assertIn("runtime_variant: cuda", manual_windows)
        self.assertIn("ProAim-Windows-x64-NVIDIA-CUDA", manual_windows)

    def test_physical_directml_gate_requires_both_fixed_products_in_separate_runs(self) -> None:
        source = _source(DIRECTML_QUALIFICATION_WORKFLOW)
        self.assertIn("workflow_dispatch:", source)
        self.assertNotIn("push:\n", source)
        self.assertIn(
            "runs-on: [self-hosted, Windows, X64, proaim-directml-qualification]",
            source,
        )
        for value in (
            "amd_rx_6950_xt",
            "nvidia_rtx_5060_laptop",
            "AMD Radeon RX 6950 XT",
            "NVIDIA GeForce RTX 5060 Laptop GPU",
            "candidate_manifest_sha256",
            "directml_zip_sha256",
            "adapter_index",
            "observer_name",
            "physical_confirmation",
        ):
            with self.subTest(value=value):
                self.assertIn(value, source)
        self.assertIn(
            "name: directml-${{ inputs.gpu_role }}-physical-attestation",
            source,
        )
        self.assertIn("DIRECTML_PHYSICAL_ATTESTATION_GUARD", source)
        self.assertIn("DIRECTML_INDEPENDENT_REVIEWER_GROUP", source)
        self.assertIn("required-reviewers-v1", source)
        self.assertIn("needs: collect-physical-evidence", source)
        self.assertIn("scripts/manage_directml_release.py inspect-candidate", source)
        self.assertIn("scripts/manage_directml_release.py seal-evidence", source)
        self.assertIn("-Provider DirectML", source)
        self.assertIn("-ExpectedAdapterName", source)
        self.assertIn("-RunLive", source)
        self.assertIn("RAW-CONTENT-MANIFEST.json", source)
        self.assertNotIn("contents: write", source)

    def test_directml_telemetry_is_pid_luid_and_product_correlated(self) -> None:
        workflow = _source(DIRECTML_QUALIFICATION_WORKFLOW)
        telemetry = _source(DIRECTML_TELEMETRY_HELPER)
        observation = _source(DIRECTML_OBSERVATION_HELPER)
        self.assertIn("Record-ProAimDirectMlTelemetry.ps1", workflow)
        self.assertIn("Complete-ProAimDirectMlObservation.ps1", workflow)
        self.assertIn("\\GPU Engine(*)\\Utilization Percentage", telemetry)
        self.assertIn("HKLM:\\SOFTWARE\\Microsoft\\DirectX", telemetry)
        self.assertIn("AdapterLuid", telemetry)
        self.assertIn('kind = "adapter_inventory"', telemetry)
        self.assertIn('kind = "proaim_gpu_engine"', telemetry)
        self.assertIn('process_name = "ProAimCLI.exe"', telemetry)
        self.assertIn("ExpectedExecutablePath", telemetry)
        self.assertIn("exactly one", telemetry)
        self.assertIn("completed_after_automated_directml_runs", observation)
        self.assertIn("automated_luid_telemetry_agreed", observation)
        self.assertIn("MessageBoxButtons]::YesNo", observation)

    def test_directml_publication_requires_two_sealed_bundles_and_is_draft_transactional(self) -> None:
        source = _source(DIRECTML_PUBLICATION_WORKFLOW)
        self.assertIn("workflow_dispatch:", source)
        self.assertNotIn("push:\n", source)
        for field in (
            "amd_evidence_run_id",
            "amd_evidence_archive_sha256",
            "amd_qualification_manifest_sha256",
            "amd_physical_attestation_sha256",
            "amd_public_receipt_sha256",
            "nvidia_evidence_run_id",
            "nvidia_evidence_archive_sha256",
            "nvidia_qualification_manifest_sha256",
            "nvidia_physical_attestation_sha256",
            "nvidia_public_receipt_sha256",
            "holdout_run_id",
            "holdout_prerequisite_artifact_id",
            "holdout_prerequisite_artifact_digest",
            "holdout_plan_artifact_id",
            "holdout_plan_artifact_digest",
            "holdout_evidence_artifact_id",
            "holdout_evidence_artifact_digest",
            "holdout_attestation_artifact_id",
            "holdout_attestation_artifact_digest",
        ):
            with self.subTest(field=field):
                self.assertRegex(source, rf"(?m)^      {field}:$")
        verification = _job(source, "verify-and-stage", "publish-draft-transactionally")
        publication = _job(source, "publish-draft-transactionally")
        self.assertNotIn("contents: write", verification)
        self.assertIn("environment:\n      name: directml-release-publication", publication)
        self.assertIn("contents: write", publication)
        self.assertIn("DIRECTML_RELEASE_ENVIRONMENT_GUARD", publication)
        self.assertIn("DIRECTML_RELEASE_REVIEWER_GROUP", publication)
        self.assertIn("required-reviewers-v1", publication)
        self.assertIn('"verify-publication-metadata"', verification)
        self.assertIn('"prepare-publication-stage"', verification)
        self.assertIn("downloaded-holdout-bundle", verification)
        self.assertIn("downloaded-holdout-attestation", verification)
        self.assertNotRegex(source, r"(?m)^      confirmation:$")
        self.assertNotIn("holdout_receipt_sha256", source)
        self.assertIn("python scripts/manage_directml_release.py @Arguments", verification)
        self.assertIn("scripts/manage_directml_release.py publish", publication)
        for argument in (
            "--holdout-prerequisite-artifact-id",
            "--holdout-prerequisite-artifact-digest",
            "--holdout-plan-artifact-id",
            "--holdout-plan-artifact-digest",
            "--holdout-evidence-artifact-id",
            "--holdout-evidence-artifact-digest",
            "--holdout-attestation-artifact-id",
            "--holdout-attestation-artifact-digest",
        ):
            with self.subTest(argument=argument):
                self.assertEqual(source.count(argument), 3)
        self.assertIn("staged_content_manifest_sha256", source)
        self.assertIn("verified-directml-release-stage", source)
        self.assertNotIn("private-holdout/*", publication)
        self.assertNotIn("private-holdout/**", publication)
        self.assertNotIn("softprops/action-gh-release", source)
        self.assertNotIn("NVIDIA-CUDA.zip", source)

    def test_independent_holdout_is_one_time_plan_first_exact_rx6950_and_fail_closed(self) -> None:
        source = _source(INDEPENDENT_HOLDOUT_WORKFLOW)
        self.assertIn("workflow_dispatch:", source)
        self.assertNotIn("push:\n", source)
        self.assertIn("permissions:\n  actions: read\n  contents: read", source)
        self.assertNotIn("contents: write", source)
        self.assertNotIn("gh release", source)
        self.assertNotIn("softprops/action-gh-release", source)
        for job in (
            "verify-physical-prerequisites",
            "freeze-plan-before-access",
            "evaluate-once",
            "attest-independently",
        ):
            self.assertIn(f"  {job}:\n", source)
        self.assertLess(
            source.index("  verify-physical-prerequisites:\n"),
            source.index("  freeze-plan-before-access:\n"),
        )
        self.assertLess(
            source.index("  freeze-plan-before-access:\n"),
            source.index("  evaluate-once:\n"),
        )
        self.assertLess(
            source.index("  evaluate-once:\n"),
            source.index("  attest-independently:\n"),
        )
        self.assertGreaterEqual(source.count('GITHUB_RUN_ATTEMPT -cne "1"'), 3)
        self.assertNotIn("--recorded-at-utc", source)
        self.assertNotIn("--retired-at-utc", source)
        self.assertEqual(source.count("python -m venv .holdout-venv"), 3)
        self.assertEqual(source.count("scripts/write_dependency_manifest.py"), 3)
        self.assertEqual(source.count("pip-bootstrap-windows-directml-py313.json"), 6)
        self.assertEqual(source.count("--no-build-isolation"), 6)
        self.assertIn("proaim-rx-6950-xt-holdout", source)
        actionlint = _source(PROJECT_ROOT / ".github" / "actionlint.yaml")
        self.assertIn("- proaim-independent-holdout-directml", actionlint)
        self.assertIn("- proaim-rx-6950-xt-holdout", actionlint)
        self.assertIn("name: independent-holdout-access", source)
        self.assertIn("name: independent-holdout-attestation", source)
        self.assertIn("--require-hashes", source)
        self.assertIn("windows-directml-py313-DEPENDENCY-MANIFEST.json", source)
        self.assertIn("scripts/verify_windows_holdout_adapter.py", source)
        self.assertIn("--adapter-index $env:AMD_ADAPTER_INDEX_INPUT", source)
        self.assertNotIn("INDEPENDENT_HOLDOUT_DIRECTML_ADAPTER_INDEX", source)
        self.assertIn("--dependency-manifest", source)
        self.assertIn("--hardware-identity verified-holdout-adapter.json", source)
        self.assertIn("::add-mask::", source)
        self.assertIn("ProAim-Verified-Holdout-Prerequisites", source)
        self.assertIn("ProAim-Independent-Holdout-Frozen-Plan", source)
        self.assertIn("ProAim-Independent-Holdout-Evidence", source)
        self.assertIn("ProAim-Independent-Holdout-Attestation", source)
        plan_job = _job(source, "freeze-plan-before-access", "evaluate-once")
        self.assertIn("needs: verify-physical-prerequisites", plan_job)
        self.assertLess(
            plan_job.index("verify-stage"),
            plan_job.index("INDEPENDENT_HOLDOUT_PACKAGE_PATH"),
        )
        self.assertLess(
            plan_job.index("scripts/verify_windows_holdout_adapter.py"),
            plan_job.index("INDEPENDENT_HOLDOUT_PACKAGE_PATH"),
        )
        self.assertLess(
            plan_job.index("ProAim-Independent-Holdout-Frozen-Plan"),
            source.index("  evaluate-once:\n"),
        )
        evaluation = _job(source, "evaluate-once", "attest-independently")
        self.assertIn(
            "needs: [verify-physical-prerequisites, freeze-plan-before-access]",
            evaluation,
        )
        self.assertIn("artifact-ids: ${{ needs.freeze-plan-before-access.outputs.plan_artifact_id }}", evaluation)
        self.assertLess(
            evaluation.index("verify-stage"),
            evaluation.index("INDEPENDENT_HOLDOUT_PACKAGE_PATH"),
        )
        self.assertLess(
            evaluation.index("scripts/verify_windows_holdout_adapter.py"),
            evaluation.index("INDEPENDENT_HOLDOUT_PACKAGE_PATH"),
        )
        attestation = _job(source, "attest-independently")
        self.assertIn(
            "needs: [verify-physical-prerequisites, freeze-plan-before-access, evaluate-once]",
            attestation,
        )
        self.assertIn("authoritative-evidence/metrics.json", attestation)
        self.assertIn("--plan downloaded-frozen-plan/evaluation-plan.json", attestation)
        self.assertLess(
            attestation.index("verify-stage"),
            attestation.index("INDEPENDENT_HOLDOUT_PACKAGE_PATH"),
        )
        self.assertLess(
            attestation.index("scripts/verify_windows_holdout_adapter.py"),
            attestation.index("INDEPENDENT_HOLDOUT_PACKAGE_PATH"),
        )
        self.assertEqual(source.count("--authenticated-evaluation-environment"), 1)
        self.assertNotIn("--authenticated-evaluation-environment", plan_job)
        self.assertNotIn("--authenticated-evaluation-environment", evaluation)

    def test_workflow_dispatch_input_count_never_exceeds_github_limit(self) -> None:
        for path in sorted(WORKFLOW_DIR.glob("*.yml")):
            source = _source(path)
            marker = "  workflow_dispatch:"
            if marker not in source:
                continue
            dispatch = source.split(marker, 1)[1]
            if "    inputs:\n" not in dispatch:
                count = 0
            else:
                input_block = dispatch.split("    inputs:\n", 1)[1]
                count = len(
                    re.findall(r"(?m)^      [A-Za-z0-9_-]+:\s*$", input_block)
                )
            with self.subTest(workflow=path.name):
                self.assertLessEqual(count, 25)

    def test_cuda_attachment_is_manual_protected_and_does_not_create_releases(self) -> None:
        source = _source(CUDA_ATTACHMENT_WORKFLOW)
        publish_job = _job(source, "publish-existing-release")
        before_publish = source[: source.index("  publish-existing-release:\n")]
        self.assertIn("workflow_dispatch:", source)
        self.assertNotIn("push:\n", source)
        self.assertNotIn("pull_request", source)
        self.assertIn("actions: read\n  contents: read", before_publish)
        self.assertNotIn("contents: write", before_publish)
        self.assertIn("environment:\n      name: cuda-release-publication", publish_job)
        self.assertIn("contents: write", publish_job)
        self.assertIn("CUDA_RELEASE_ENVIRONMENT_GUARD", publish_job)
        self.assertIn("CUDA_RELEASE_ENVIRONMENT_POLICY_VERSION", publish_job)
        self.assertIn("required-reviewers-v1", publish_job)
        self.assertIn("Require dispatch from protected main workflow source", source)
        self.assertIn('refs/heads/main', source)
        self.assertIn("cancel-in-progress: false", source)
        self.assertEqual(source.count("actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1 # v6"), 2)
        self.assertEqual(source.count('python-version: "3.13.14"'), 2)
        self.assertNotIn("softprops/action-gh-release", source)

    def test_cuda_attachment_binds_source_run_candidate_and_physical_evidence(self) -> None:
        source = _source(CUDA_ATTACHMENT_WORKFLOW)
        for workflow_input in (
            "tag",
            "source_build_run_id",
            "evidence_run_id",
            "evidence_artifact_name",
            "cuda_zip_sha256",
            "qualification_evidence_sha256",
            "qualification_manifest_sha256",
            "physical_attestation_sha256",
            "qualified_gpu",
            "confirmation",
            "nvidia_redistribution_confirmation",
        ):
            with self.subTest(workflow_input=workflow_input):
                self.assertRegex(source, rf"(?m)^      {workflow_input}:$")
        self.assertIn("ProAim-Windows-x64-NVIDIA-CUDA", source)
        self.assertIn("run-id: ${{ inputs.source_build_run_id }}", source)
        self.assertIn("run-id: ${{ inputs.evidence_run_id }}", source)
        self.assertIn("ProAim-Windows-CUDA-Qualification-Evidence", source)
        self.assertIn(
            "artifact-ids: ${{ steps.identity.outputs.build_artifact_id }}",
            source,
        )
        self.assertIn(
            "artifact-ids: ${{ steps.identity.outputs.evidence_artifact_id }}",
            source,
        )
        self.assertIn("github-token: ${{ github.token }}", source)
        self.assertIn("scripts/manage_cuda_release_attachment.py", source)
        self.assertIn("verify-metadata", source)
        self.assertIn("validate-smoke", source)
        self.assertIn("$BuildInfo.release_default_model", source)
        self.assertIn("--model $Model", source)
        self.assertIn("--labels $Labels", source)
        self.assertIn("--name release-default", source)
        self.assertIn("--inference-size $InferenceSize", source)
        self.assertNotIn("fort-416-fp32", source)
        self.assertIn("--device CPU", source)
        self.assertIn("real release-default-model inference on CPU only", source)
        self.assertIn("no legal-rights inference", source)
        self.assertIn("I APPROVE NVIDIA REDISTRIBUTION REVIEW FOR", source)

    def test_physical_cuda_qualification_is_manual_self_hosted_and_protected(self) -> None:
        source = _source(CUDA_QUALIFICATION_WORKFLOW)
        self.assertIn("workflow_dispatch:", source)
        self.assertNotIn("push:\n", source)
        self.assertNotIn("pull_request", source)
        self.assertIn("actions: read\n  contents: read", source)
        self.assertNotIn("contents: write", source)
        self.assertIn(
            "runs-on: [self-hosted, Windows, X64, proaim-cuda-qualification]",
            source,
        )
        self.assertIn("environment:\n      name: cuda-physical-attestation", source)
        self.assertIn("CUDA_PHYSICAL_ATTESTATION_GUARD", source)
        self.assertIn("CUDA_PHYSICAL_ATTESTATION_POLICY_VERSION", source)
        self.assertIn("required-reviewers-v1", source)
        self.assertIn("needs: collect-physical-evidence", source)
        self.assertIn("artifact-ids: ${{ needs.collect-physical-evidence.outputs.raw_artifact_id }}", source)
        self.assertNotIn("digest-mismatch", source)
        self.assertIn("RAW-CONTENT-MANIFEST.json", source)
        self.assertIn("raw_content_manifest_sha256", source)

    def test_physical_cuda_artifact_is_immutable_structured_and_exactly_named(self) -> None:
        source = _source(CUDA_QUALIFICATION_WORKFLOW)
        for workflow_input in (
            "tag",
            "source_build_run_id",
            "cuda_zip_sha256",
            "qualified_gpu",
            "observer_name",
            "physical_confirmation",
            "legal_review_acknowledgement",
        ):
            with self.subTest(workflow_input=workflow_input):
                self.assertRegex(source, rf"(?m)^      {workflow_input}:$")
        self.assertIn("verify-source", source)
        self.assertIn("inspect-candidate", source)
        self.assertIn(
            "artifact-ids: ${{ steps.source.outputs.build_artifact_id }}",
            source,
        )
        self.assertIn("seal-evidence", source)
        self.assertIn("Complete-ProAimPhysicalObservation.ps1", source)
        self.assertIn("LOCAL-PHYSICAL-OBSERVATION.json", source)
        observation = _source(CUDA_OBSERVATION_HELPER)
        self.assertIn("completed_after_automated_gpu_runs", observation)
        self.assertIn("ExpectedTypedConfirmation", observation)
        self.assertIn("MessageBoxButtons]::YesNo", observation)
        self.assertIn("PHYSICAL-GPU-ATTESTATION.json", source)
        self.assertIn("qualification-manifest.json", source)
        self.assertIn("ProAim-Windows-CUDA-Qualification-Evidence.zip", source)
        self.assertIn("name: ProAim-Windows-CUDA-Qualification-Evidence", source)
        self.assertIn("NVIDIA redistribution permission still requires separate human legal approval", source)
        self.assertIn(
            "PHYSICAL QUALIFICATION ONLY - NVIDIA LEGAL REVIEW REMAINS REQUIRED",
            source,
        )

    def test_physical_cuda_workflow_records_gpu_process_and_utilization(self) -> None:
        workflow = _source(CUDA_QUALIFICATION_WORKFLOW)
        telemetry = _source(CUDA_TELEMETRY_HELPER)
        self.assertIn("Record-ProAimCudaTelemetry.ps1", workflow)
        self.assertIn("-RunLive", workflow)
        self.assertIn("nvidia-smi-telemetry.jsonl", workflow)
        self.assertIn("telemetry_interval_milliseconds = 500", workflow)
        self.assertIn("-IntervalMilliseconds", workflow)
        self.assertIn("--query-gpu=timestamp,index,name,uuid,driver_version,compute_cap,utilization.gpu", telemetry)
        self.assertIn("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", telemetry)
        self.assertIn('kind = "gpu"', telemetry)
        self.assertIn('kind = "compute_process"', telemetry)

    def test_cuda_hosted_cpu_smoke_does_not_request_accelerator_only_gate(self) -> None:
        source = _source(CUDA_ATTACHMENT_WORKFLOW)
        smoke_start = source.index(
            "      - name: Smoke-test the extracted frozen CLI and release-default model on CPU\n"
        )
        smoke_end = source.index(
            "      - name: Upload immutable verified attachment staging artifact\n",
            smoke_start,
        )
        smoke = source[smoke_start:smoke_end]
        self.assertIn("--device CPU", smoke)
        self.assertIn("validate-smoke", smoke)
        self.assertNotIn("--require-full-provider", smoke)
        validator = _source(
            PROJECT_ROOT / "scripts" / "manage_cuda_release_attachment.py"
        )
        self.assertIn('methodology.get("requested_device") != "CPU"', validator)
        self.assertIn(
            'model_runtime.get("requested_provider") != "CPUExecutionProvider"',
            validator,
        )
        self.assertIn('"CPUExecutionProvider" not in active', validator)

    def test_cuda_attachment_verifies_before_the_protected_publish_job(self) -> None:
        source = _source(CUDA_ATTACHMENT_WORKFLOW)
        verify_job = _job(source, "verify-candidate", "publish-existing-release")
        publish_job = _job(source, "publish-existing-release")
        metadata_index = verify_job.index("verify-metadata")
        download_index = verify_job.index("actions/download-artifact@")
        prepare_index = verify_job.index('"prepare"')
        frozen_index = verify_job.index("Smoke-test the extracted frozen CLI")
        upload_index = verify_job.index("actions/upload-artifact@")
        self.assertLess(metadata_index, download_index)
        self.assertLess(download_index, prepare_index)
        self.assertLess(prepare_index, frozen_index)
        self.assertLess(frozen_index, upload_index)
        self.assertIn("needs: verify-candidate", publish_job)
        self.assertIn("Reverify and attach to the existing release with rollback", publish_job)
        self.assertIn(
            "artifact-ids: ${{ needs.verify-candidate.outputs.verified_artifact_id }}",
            publish_job,
        )
        self.assertNotIn("digest-mismatch", source)
        self.assertIn("STAGED-CONTENT-MANIFEST.json", source)
        self.assertIn("staged_content_manifest_sha256", publish_job)
        self.assertIn(
            "staged_content_manifest_sha256: ${{ steps.staged-content.outputs.staged_content_manifest_sha256 }}",
            verify_job,
        )

    def test_cuda_attachment_requires_nvidia_redistribution_inventory(self) -> None:
        helper = _source(PROJECT_ROOT / "scripts" / "manage_cuda_release_attachment.py")
        manifest = _source(
            PROJECT_ROOT / "scripts" / "write_nvidia_redistribution_manifest.py"
        )
        windows_build = _source(WINDOWS_BUILD)
        notices = _source(PROJECT_ROOT / "THIRD_PARTY_NOTICES.md")
        self.assertIn("NVIDIA-REDISTRIBUTION-MANIFEST.json", helper)
        self.assertIn("validate_nvidia_manifest", helper)
        self.assertIn("EXPECTED_LICENSE_EXPRESSION", manifest)
        self.assertIn("LicenseRef-NVIDIA-Proprietary", manifest)
        self.assertIn("declared_license_files", manifest)
        self.assertIn("native_libraries", manifest)
        self.assertIn("write_nvidia_redistribution_manifest.py", windows_build)
        self.assertLess(
            windows_build.index("write_nvidia_redistribution_manifest.py"),
            windows_build.index("scripts\\write_build_info.py"),
        )
        for distribution in (
            "nvidia-cuda-nvrtc",
            "nvidia-cuda-runtime",
            "nvidia-cufft",
            "nvidia-curand",
            "nvidia-cudnn-cu13",
            "nvidia-cublas",
            "nvidia-nvjitlink",
        ):
            with self.subTest(distribution=distribution):
                self.assertIn(distribution, notices)

    def test_windows_manual_build_tests_before_packaging_and_upload(self) -> None:
        source = _source(WINDOWS_WORKFLOW)
        self.assertIn("timeout-minutes: 60", source)
        self.assertIn("runtime_variant: cuda", source)
        self.assertIn("runtime_variant: directml", source)
        test_index = source.index("python -m unittest discover -s tests")
        build_index = source.index("./scripts/build_windows_app.ps1")
        help_smoke_index = source.index("./dist/ProAim/ProAimCLI.exe --cli --help")
        runtime_smoke_index = source.index("./dist/ProAim/ProAimCLI.exe --runtime-info")
        model_smoke_index = source.index("python scripts/smoke_release_default_model.py")
        upload_index = source.index("actions/upload-artifact")
        self.assertLess(test_index, build_index)
        self.assertLess(build_index, help_smoke_index)
        self.assertLess(help_smoke_index, runtime_smoke_index)
        self.assertLess(runtime_smoke_index, model_smoke_index)
        self.assertLess(model_smoke_index, upload_index)

    def test_build_info_is_written_before_archives_are_created(self) -> None:
        linux = _source(LINUX_BUILD)
        windows = _source(WINDOWS_BUILD)
        self.assertIn("scripts/write_build_info.py", linux)
        self.assertIn("scripts\\write_build_info.py", windows)
        self.assertIn("--dependency-manifest", linux)
        self.assertIn("--dependency-manifest", windows)
        # Linux is archived by the workflow after its helper returns. Windows
        # creates the ZIP itself, so metadata must be written first.
        self.assertLess(
            windows.index("$BuildInfo"),
            windows.index("Compress-Archive"),
        )

    def test_workflow_expressions_are_not_accidentally_shell_expansions(self) -> None:
        expression = re.compile(r"\$\{\{[^{}]+\}\}")
        for path in (
            CI_WORKFLOW,
            RELEASE_WORKFLOW,
            WINDOWS_WORKFLOW,
            CUDA_ATTACHMENT_WORKFLOW,
        ):
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
