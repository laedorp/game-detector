from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from detection.runtime_setup import (
    DISTRIBUTION_CPU,
    DISTRIBUTION_DIRECTML,
    DISTRIBUTION_NVIDIA,
    DISTRIBUTION_ROCM,
    RUNTIME_REQUIREMENTS,
    RUNTIME_ROOT_ENV,
    RuntimeSetupError,
    activate,
    activate_configured_runtime,
    describe,
    ensure_runtime,
    install_root,
    plan_for,
)


def completed(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


class PlanTests(unittest.TestCase):
    def test_amd_on_linux_marks_legacy_rocm_as_experimental(self) -> None:
        plan = plan_for("amd", "linux")

        self.assertEqual(plan.distribution, DISTRIBUTION_ROCM)
        self.assertTrue(plan.needs_driver)
        self.assertIn("experimental", plan.driver_note)
        self.assertIn("RX 6950 XT", plan.driver_note)

    def test_amd_on_windows_uses_directml(self) -> None:
        plan = plan_for("amd", "windows")

        self.assertEqual(plan.distribution, DISTRIBUTION_DIRECTML)

    def test_nvidia_on_linux_uses_the_cuda_build(self) -> None:
        plan = plan_for("nvidia", "linux")

        self.assertEqual(plan.distribution, DISTRIBUTION_NVIDIA)
        self.assertTrue(plan.needs_driver)

    def test_nvidia_on_windows_prefers_cuda_tensorrt_runtime(self) -> None:
        plan = plan_for("nvidia", "windows")

        self.assertEqual(plan.distribution, DISTRIBUTION_NVIDIA)
        self.assertTrue(plan.needs_driver)

    def test_intel_and_unknown_vendors_fall_back_to_the_cpu_build(self) -> None:
        for vendor in ("intel", "unknown", ""):
            with self.subTest(vendor=vendor):
                plan = plan_for(vendor, "linux")
                self.assertEqual(plan.distribution, DISTRIBUTION_CPU)
                self.assertFalse(plan.needs_driver)

    def test_vendor_and_system_are_case_insensitive(self) -> None:
        self.assertEqual(plan_for("AMD", "Linux").distribution, DISTRIBUTION_ROCM)

    def test_source_nvidia_install_includes_cuda_and_cudnn_user_libraries(self) -> None:
        self.assertEqual(
            RUNTIME_REQUIREMENTS[DISTRIBUTION_NVIDIA],
            "onnxruntime-gpu[cuda,cudnn]==1.28.0",
        )


class EnsureRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = Path(self.temporary.name)
        self._original_path = list(sys.path)
        self._original_runtime_root = os.environ.get(RUNTIME_ROOT_ENV)

    def tearDown(self) -> None:
        sys.path[:] = self._original_path
        if self._original_runtime_root is None:
            os.environ.pop(RUNTIME_ROOT_ENV, None)
        else:
            os.environ[RUNTIME_ROOT_ENV] = self._original_runtime_root
        self.temporary.cleanup()

    def test_an_already_correct_install_downloads_nothing(self) -> None:
        calls: list[list[str]] = []

        def runner(command):
            calls.append(list(command))
            return completed()

        install_root(self.settings).mkdir(parents=True)
        ensure_runtime(
            plan_for("amd", "linux"),
            self.settings,
            runner=runner,
            already_installed=DISTRIBUTION_ROCM,
        )

        self.assertEqual(calls, [])

    def test_a_conflicting_install_is_refused_rather_than_stacked(self) -> None:
        def runner(command):
            raise AssertionError("must not install over a conflicting distribution")

        with self.assertRaises(RuntimeSetupError) as raised:
            ensure_runtime(
                plan_for("amd", "linux"),
                self.settings,
                runner=runner,
                already_installed=DISTRIBUTION_NVIDIA,
            )

        message = str(raised.exception)
        self.assertIn(DISTRIBUTION_NVIDIA, message)
        self.assertIn("cannot be installed together", message)

    def test_a_missing_runtime_is_installed_into_the_user_directory(self) -> None:
        calls: list[list[str]] = []

        def runner(command):
            calls.append(list(command))
            return completed()

        target = ensure_runtime(
            plan_for("amd", "linux"),
            self.settings,
            runner=runner,
            already_installed=None,
        )

        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertIn(RUNTIME_REQUIREMENTS[DISTRIBUTION_ROCM], command)
        self.assertIn("--target", command)
        # The install must land beside the user's settings, never inside the
        # application directory, which may be read-only or shared.
        self.assertEqual(command[command.index("--target") + 1], str(target))
        self.assertTrue(str(target).startswith(str(self.settings)))

    def test_a_failed_install_reports_the_reason(self) -> None:
        def runner(command):
            return completed(returncode=1, stderr="No matching distribution found")

        with self.assertRaises(RuntimeSetupError) as raised:
            ensure_runtime(
                plan_for("amd", "linux"),
                self.settings,
                runner=runner,
                already_installed=None,
            )

        self.assertIn("No matching distribution found", str(raised.exception))

    def test_an_unlaunchable_installer_is_reported_not_raised_raw(self) -> None:
        def runner(command):
            raise OSError("pip is missing")

        with self.assertRaises(RuntimeSetupError) as raised:
            ensure_runtime(
                plan_for("nvidia", "linux"),
                self.settings,
                runner=runner,
                already_installed=None,
            )

        self.assertIn("pip is missing", str(raised.exception))

    def test_installing_puts_the_directory_first_on_the_import_path(self) -> None:
        target = ensure_runtime(
            plan_for("amd", "linux"),
            self.settings,
            runner=lambda command: completed(),
            already_installed=None,
        )

        self.assertEqual(sys.path[0], str(target))
        self.assertEqual(os.environ[RUNTIME_ROOT_ENV], str(target))

    def test_frozen_app_refuses_to_invoke_itself_as_pip(self) -> None:
        calls = []
        with self.assertRaisesRegex(RuntimeSetupError, "bundle that matches this GPU"):
            ensure_runtime(
                plan_for("nvidia", "linux"),
                self.settings,
                runner=lambda command: calls.append(list(command)),
                already_installed=None,
                frozen=True,
            )
        self.assertEqual(calls, [])

    def test_explicit_interpreter_is_used_for_source_install(self) -> None:
        calls: list[list[str]] = []
        ensure_runtime(
            plan_for("amd", "linux"),
            self.settings,
            runner=lambda command: calls.append(list(command)) or completed(),
            already_installed=None,
            interpreter="/opt/proaim-python",
            frozen=False,
        )
        self.assertEqual(calls[0][0], "/opt/proaim-python")


class ActivateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_path = list(sys.path)
        self._original_runtime_root = os.environ.get(RUNTIME_ROOT_ENV)

    def tearDown(self) -> None:
        sys.path[:] = self._original_path
        if self._original_runtime_root is None:
            os.environ.pop(RUNTIME_ROOT_ENV, None)
        else:
            os.environ[RUNTIME_ROOT_ENV] = self._original_runtime_root

    def test_a_missing_directory_is_not_added(self) -> None:
        self.assertFalse(activate(Path("/nonexistent-runtime-directory")))

    def test_activation_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            self.assertTrue(activate(target))
            self.assertTrue(activate(target))
            self.assertEqual(sys.path.count(str(target)), 1)

    def test_configured_runtime_environment_is_activated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            os.environ[RUNTIME_ROOT_ENV] = temporary
            self.assertTrue(activate_configured_runtime(frozen=False))
            self.assertEqual(sys.path[0], temporary)

    def test_frozen_bundle_never_activates_external_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original_path = list(sys.path)
            os.environ[RUNTIME_ROOT_ENV] = temporary

            self.assertFalse(activate_configured_runtime(frozen=True))
            self.assertEqual(sys.path, original_path)
            self.assertEqual(os.environ[RUNTIME_ROOT_ENV], temporary)


class DescribeTests(unittest.TestCase):
    def test_already_installed_says_nothing_will_download(self) -> None:
        text = describe(plan_for("amd", "linux"), DISTRIBUTION_ROCM)

        self.assertIn("already installed", text)

    def test_missing_runtime_says_it_will_download_on_approval(self) -> None:
        text = describe(plan_for("amd", "linux"), None)

        self.assertIn("downloaded on approval", text)
        self.assertIn("ROCm", text)

    def test_a_conflict_is_explained_before_anything_is_fetched(self) -> None:
        text = describe(plan_for("amd", "linux"), DISTRIBUTION_NVIDIA)

        self.assertIn("conflicts", text)


if __name__ == "__main__":
    unittest.main()
