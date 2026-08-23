from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER = PROJECT_ROOT / "packaging" / "windows" / "Qualify-ProAimGpu.ps1"
BUILD_HELPER = PROJECT_ROOT / "scripts" / "build_windows_app.ps1"
WINDOWS_GUIDE = PROJECT_ROOT / "packaging" / "windows" / "README-Windows.txt"


class WindowsGpuQualificationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = HELPER.read_text(encoding="utf-8")

    def test_provider_and_adapter_inputs_are_narrow(self) -> None:
        self.assertIn('[ValidateSet("DirectML", "CUDA")]', self.source)
        self.assertIn("[int]$AdapterIndex", self.source)
        self.assertIn('requires -AdapterIndex N', self.source)
        self.assertIn('applies only to DirectML', self.source)
        self.assertNotIn("Invoke-Expression", self.source)
        self.assertIn("& $CliPath @Arguments", self.source)

    def test_every_accelerated_run_uses_the_full_provider_gate(self) -> None:
        self.assertGreaterEqual(self.source.count('"--require-full-provider"'), 2)
        self.assertIn("Assert-ProviderSummary", self.source)
        self.assertIn("CPU graph-node fallback disabled", self.source)
        self.assertIn('"CUDAExecutionProvider"', self.source)
        self.assertIn('"DmlExecutionProvider"', self.source)
        self.assertIn('provider_option_overrides.DmlExecutionProvider.device_id', self.source)
        self.assertIn("$LASTEXITCODE", self.source)
        self.assertIn('failed with exit code', self.source)

    def test_release_default_model_is_resolved_from_build_info_and_hash_bound(self) -> None:
        self.assertIn("$BuildInfo.release_default_model", self.source)
        self.assertIn("Resolve-BundleContractFile", self.source)
        self.assertIn("$ReleaseDefault.model_path", self.source)
        self.assertIn("$ReleaseDefault.labels_path", self.source)
        self.assertIn("$ReleaseDefault.input_shape_hw", self.source)
        self.assertIn("$ReleaseDefault.model_sha256", self.source)
        self.assertIn("$ReleaseDefault.labels_sha256", self.source)
        self.assertNotIn("fort_player_416", self.source)
        guide = WINDOWS_GUIDE.read_text(encoding="utf-8")
        self.assertIn("release-default model, labels, and input shape", guide)
        self.assertNotIn("-CandidateModelPath", guide)
        self.assertIn('Get-FileHash -LiteralPath $Path -Algorithm SHA256', self.source)
        self.assertIn('did not fingerprint the exact selected ONNX artifact', self.source)
        self.assertIn('New-ArtifactRecord -Role "build_info"', self.source)
        self.assertIn('New-ArtifactRecord -Role "release_default_model"', self.source)
        self.assertIn('New-ArtifactRecord -Role "release_default_labels"', self.source)
        self.assertIn('bundle_build_info = $BuildInfo', self.source)

    def test_dependency_manifest_requires_complete_installed_record_evidence(self) -> None:
        self.assertIn("Test-StrictJsonInteger", self.source)
        self.assertIn("$Distribution.installed_files", self.source)
        self.assertIn("record_entry_count", self.source)
        self.assertIn("record_sha256_entries_verified", self.source)
        self.assertIn("unhashed_record_entries", self.source)
        self.assertIn("total_size_bytes", self.source)
        self.assertIn("aggregate_sha256", self.source)
        self.assertIn("record_document_sha256", self.source)
        self.assertIn("installed_record_sha256", self.source)
        self.assertIn("installed RECORD digests disagree", self.source)

    def test_optional_live_ab_is_bounded_and_uses_release_default_model(self) -> None:
        for argument in (
            '"--max-frames"',
            '"--max-seconds"',
            '"--metrics-json"',
            '"--no-preview"',
            '"--preview-fps"',
        ):
            with self.subTest(argument=argument):
                self.assertIn(argument, self.source)
        self.assertIn('"dxcam-dxgi"', self.source)
        self.assertIn("$LiveModel = $ReleaseDefaultModel", self.source)
        self.assertIn("$LiveLabels = $ReleaseDefaultLabels", self.source)
        self.assertIn("$LiveSize = $ReleaseDefaultInferenceSize", self.source)
        self.assertIn("$LiveHash = $ReleaseDefaultModelHash", self.source)
        self.assertIn("too few frames/timing samples", self.source)
        self.assertIn("CultureInfo]::InvariantCulture", self.source)

    def test_release_qualification_does_not_switch_to_supplemental_candidate(self) -> None:
        self.assertNotIn("CandidateModelPath", self.source)
        self.assertNotIn("CandidateLabelsPath", self.source)
        self.assertNotIn("CandidateInferenceSize", self.source)
        self.assertNotIn('$LiveModel = $CandidateModel', self.source)
        self.assertNotIn('$LiveLabels = $CandidateLabels', self.source)
        self.assertNotIn('$LiveSize = $NormalizedCandidateSize', self.source)
        self.assertIn('[string]$ModelPath', self.source)
        self.assertIn('[string]$InferenceSize', self.source)
        self.assertIn('selected_model = "release-default"', self.source)
        self.assertIn('release_default_model = $ReleaseDefault', self.source)

    def test_experimental_detail_pass_is_explicit_and_evidence_bound(self) -> None:
        self.assertIn('[ValidateRange(-1, 16384)]', self.source)
        self.assertIn('[int]$DetailCropSize = -1', self.source)
        self.assertIn('$ReleaseDefault.detail_crop_size_source_pixels', self.source)
        self.assertIn('cannot override the release-default BUILD-INFO contract', self.source)
        self.assertIn('"--detail-crop-size"', self.source)
        self.assertIn('$Report.detail_pass.enabled', self.source)
        self.assertIn('$Report.detail_pass.requested_crop_size', self.source)
        self.assertIn('$Plan.applied_crop_width', self.source)
        self.assertIn('$Plan.applied_crop_height', self.source)
        self.assertIn('$Plan.effective_linear_magnification', self.source)
        self.assertIn('centered_model_aspect_roi', self.source)
        self.assertIn('detail_crop_size = if ($DetailCropSize -gt 0)', self.source)

    def test_directml_instructions_are_vendor_neutral(self) -> None:
        guide = WINDOWS_GUIDE.read_text(encoding="utf-8")
        self.assertIn("Radeon RX", guide)
        self.assertIn("Windows does not require ROCm", guide)
        self.assertIn("AMD, Intel,", guide)
        self.assertNotIn("RTX", self.source)
        self.assertNotIn("ROCMExecutionProvider", self.source)
        self.assertIn("-ExecutionPolicy Bypass -File", guide)
        self.assertIn("does not change the machine's saved policy", guide)

    def test_evidence_publish_is_new_atomic_and_cleans_only_its_stage(self) -> None:
        self.assertIn("Refusing to overwrite existing evidence path", self.source)
        self.assertIn('partial-{1}', self.source)
        self.assertIn('[Guid]::NewGuid().ToString("N")', self.source)
        self.assertIn('[System.IO.Directory]::Move($StageDirectory, $FinalDirectory)', self.source)
        self.assertIn('Remove-Item -LiteralPath $StageDirectory -Recurse -Force', self.source)
        self.assertIn('if ($File.Name -eq "TASK-MANAGER-CONFIRMATION.txt")', self.source)
        self.assertLess(
            self.source.index('Write-Utf8File -Path (Join-Path $StageDirectory "qualification-manifest.json")'),
            self.source.index('[System.IO.Directory]::Move($StageDirectory, $FinalDirectory)'),
        )

    def test_no_automatic_physical_gpu_claim_is_emitted(self) -> None:
        self.assertIn('software_checks_passed_physical_gpu_confirmation_pending', self.source)
        self.assertIn('qualified = $false', self.source)
        self.assertIn('completed_by_helper = $false', self.source)
        self.assertIn("Task Manager", self.source)
        self.assertIn("Provider activation alone is not physical-GPU proof", self.source)

    def test_windows_build_copies_helper_to_archive_root(self) -> None:
        build = BUILD_HELPER.read_text(encoding="utf-8")
        self.assertIn('packaging\\windows\\Qualify-ProAimGpu.ps1', build)
        self.assertIn('(Join-Path $BundleDir "Qualify-ProAimGpu.ps1")', build)
        self.assertLess(
            build.index('(Join-Path $BundleDir "Qualify-ProAimGpu.ps1")'),
            build.index("Compress-Archive"),
        )


if __name__ == "__main__":
    unittest.main()
