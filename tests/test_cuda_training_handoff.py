from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts import prepare_cuda_training_handoff as handoff
from scripts import train_fort_model as trainer


class _FakeCudnn:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def version() -> int:
        return 91002


class _FakeCuda:
    name = handoff.EXPECTED_DEVICE_NAME
    capability = handoff.EXPECTED_DEVICE_CAPABILITY
    total = 8 * 1024**3
    free = 7 * 1024**3
    available = True

    def is_available(self) -> bool:
        return self.available

    @staticmethod
    def device_count() -> int:
        return 1

    def get_device_name(self, _index: int) -> str:
        return self.name

    def get_device_capability(self, _index: int) -> tuple[int, int]:
        return self.capability

    @staticmethod
    def get_arch_list() -> list[str]:
        return ["sm_80", "sm_90", "sm_120"]

    def get_device_properties(self, _index: int):
        return type("Properties", (), {"total_memory": self.total})()

    def mem_get_info(self, _index: int) -> tuple[int, int]:
        return self.free, self.total

    @staticmethod
    def set_device(_index: int) -> None:
        return None

    @staticmethod
    def synchronize(_index: int) -> None:
        return None


class _FakeTensor:
    def add_(self, _value: int):
        return self


class _FakeTorch:
    __version__ = handoff.EXPECTED_PACKAGE_VERSIONS["torch"]
    version = type("Version", (), {"cuda": "13.0", "hip": None})()
    cuda = _FakeCuda()
    backends = type("Backends", (), {"cudnn": _FakeCudnn()})()

    @staticmethod
    def empty(_size: int, *, device: str) -> _FakeTensor:
        if device != "cuda:0":
            raise RuntimeError("wrong device")
        return _FakeTensor()


def _versions(name: str) -> str:
    return handoff.EXPECTED_PACKAGE_VERSIONS[name]


def _isolated_gpu_evidence() -> dict[str, object]:
    gpu = handoff.verify_cuda_device(_FakeTorch(), 0)
    gpu["torchvision_import_version"] = handoff.EXPECTED_PACKAGE_VERSIONS["torchvision"]
    gpu["training_smoke"] = dict(handoff.EXPECTED_TRAINING_SMOKE)
    gpu["selected_model_smoke"] = {
        "filename": "yolo26n.pt",
        "sha256": handoff.MODEL_CONTRACTS["n"].sha256,
        "task": "detect",
        "training_batch": handoff.MODEL_CONTRACTS["n"].training_batch,
        "image_size": 640,
        "precision": "fp32",
        "output_tensor_count": 6,
        "gradient_tensor_count": 42,
        "peak_allocated_vram_bytes": 1024,
        "peak_reserved_vram_bytes": 2048,
    }
    return gpu


class CudaTrainingHandoffTests(unittest.TestCase):
    def test_isolated_ultralytics_settings_are_offline_and_schema_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = handoff._write_isolated_ultralytics_settings(Path(temporary))
            settings = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            settings["settings_version"],
            handoff.ULTRALYTICS_SETTINGS_SCHEMA_VERSION,
        )
        self.assertEqual(settings["uuid"], "0" * 64)
        for key in (
            "sync",
            "clearml",
            "comet",
            "dvc",
            "mlflow",
            "neptune",
            "raytune",
            "tensorboard",
            "wandb",
            "vscode_msg",
            "openvino_msg",
        ):
            self.assertFalse(settings[key])

    def test_cuda_gate_records_exact_5060_and_vram(self) -> None:
        record = handoff.verify_cuda_device(_FakeTorch(), 0)
        self.assertEqual(record["name"], handoff.EXPECTED_DEVICE_NAME)
        self.assertEqual(record["capability"], [12, 0])
        self.assertGreaterEqual(record["free_vram_bytes"], 6 * 1024**3)

    def test_isolated_cuda_probe_accepts_one_exact_evidence_record(self) -> None:
        gpu = _isolated_gpu_evidence()
        model = handoff.MODEL_CONTRACTS["n"]
        weights = handoff.PROJECT_ROOT / model.filename
        payload = {
            "ok": True,
            "torch_import_version": handoff.EXPECTED_PACKAGE_VERSIONS["torch"],
            "torchvision_import_version": handoff.EXPECTED_PACKAGE_VERSIONS["torchvision"],
            "gpu": gpu,
        }
        completed = handoff.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=handoff.CUDA_PROBE_MARKER + json.dumps(payload) + "\n",
            stderr="",
        )
        runner = mock.Mock(return_value=completed)
        self.assertEqual(
            handoff.probe_cuda_device_isolated(
                0,
                weights,
                model.sha256,
                model.training_batch,
                runner=runner,
            ),
            gpu,
        )
        args, kwargs = runner.call_args
        self.assertEqual(
            args[0],
            [
                sys.executable,
                "-u",
                "-c",
                handoff.CUDA_PROBE_CODE,
                "0",
                str(weights),
                model.sha256,
                str(model.training_batch),
            ],
        )
        self.assertEqual(kwargs["cwd"], handoff.PROJECT_ROOT)
        self.assertFalse(kwargs["check"])
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["timeout"], 180)
        self.assertEqual(kwargs["env"]["PYTHONNOUSERSITE"], "1")
        self.assertNotIn("PYTHONPATH", kwargs["env"])
        self.assertFalse(Path(kwargs["env"]["YOLO_CONFIG_DIR"]).exists())

    def test_isolated_cuda_probe_fails_closed_on_error_or_unaudited_output(self) -> None:
        cases = (
            (
                "nonzero exit",
                handoff.subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr=""
                ),
                "unaudited output",
            ),
            (
                "reported failure",
                handoff.subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=handoff.CUDA_PROBE_MARKER
                    + json.dumps({"ok": False, "error_type": "RuntimeError", "error": "bad"})
                    + "\n",
                    stderr="",
                ),
                "RuntimeError: bad",
            ),
            (
                "stderr",
                handoff.subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr="warning"
                ),
                "unaudited output",
            ),
            (
                "malformed",
                handoff.subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=handoff.CUDA_PROBE_MARKER + "not-json\n",
                    stderr="",
                ),
                "invalid JSON",
            ),
            (
                "extra stdout",
                handoff.subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="unexpected\n"
                    + handoff.CUDA_PROBE_MARKER
                    + json.dumps({"ok": True})
                    + "\n",
                    stderr="",
                ),
                "unaudited output",
            ),
        )
        for label, completed, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, message):
                    handoff.probe_cuda_device_isolated(
                        0,
                        handoff.PROJECT_ROOT / "yolo26n.pt",
                        handoff.MODEL_CONTRACTS["n"].sha256,
                        8,
                        runner=mock.Mock(return_value=completed),
                    )

    def test_isolated_cuda_probe_rejects_malformed_success_evidence(self) -> None:
        gpu = _isolated_gpu_evidence()
        gpu["training_smoke"] = {"fp32_convolution_forward": "passed"}
        payload = {
            "ok": True,
            "torch_import_version": handoff.EXPECTED_PACKAGE_VERSIONS["torch"],
            "torchvision_import_version": handoff.EXPECTED_PACKAGE_VERSIONS["torchvision"],
            "gpu": gpu,
        }
        completed = handoff.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=handoff.CUDA_PROBE_MARKER + json.dumps(payload) + "\n",
            stderr="",
        )
        with self.assertRaisesRegex(
            handoff.CudaTrainingHandoffError, "training-smoke evidence"
        ):
            handoff.probe_cuda_device_isolated(
                0,
                handoff.PROJECT_ROOT / "yolo26n.pt",
                handoff.MODEL_CONTRACTS["n"].sha256,
                8,
                runner=mock.Mock(return_value=completed),
            )

        gpu = _isolated_gpu_evidence()
        del gpu["selected_model_smoke"]["gradient_tensor_count"]
        payload["gpu"] = gpu
        completed.stdout = handoff.CUDA_PROBE_MARKER + json.dumps(payload) + "\n"
        with self.assertRaisesRegex(
            handoff.CudaTrainingHandoffError, "gradient_tensor_count"
        ):
            handoff.probe_cuda_device_isolated(
                0,
                handoff.PROJECT_ROOT / "yolo26n.pt",
                handoff.MODEL_CONTRACTS["n"].sha256,
                8,
                runner=mock.Mock(return_value=completed),
            )

    def test_cuda_gate_rejects_cpu_amd_wrong_identity_and_low_vram(self) -> None:
        cases = (
            ("CPU", {"available": False}, "CUDA is unavailable"),
            ("AMD", {"name": "AMD Radeon RX 6950 XT"}, "identity mismatch"),
            ("low VRAM", {"total": 4 * 1024**3}, "total VRAM"),
            ("busy", {"free": 2 * 1024**3}, "free VRAM"),
            ("inconsistent", {"free": 9 * 1024**3}, "inconsistent values"),
        )
        for label, changes, message in cases:
            with self.subTest(label=label):
                torch = _FakeTorch()
                cuda = _FakeCuda()
                for key, value in changes.items():
                    setattr(cuda, key, value)
                torch.cuda = cuda
                with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, message):
                    handoff.verify_cuda_device(torch, 0)

    def test_package_gate_is_exact(self) -> None:
        self.assertEqual(handoff.verify_packages(_versions)["ultralytics"], "8.4.116")

        def wrong(name: str) -> str:
            return "0.0" if name == "torch" else _versions(name)

        with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, "torch"):
            handoff.verify_packages(wrong)

    def test_hash_gate_rejects_tampering(self) -> None:
        self.assertEqual(
            handoff._sha256_file(handoff.TRAINING_SCRIPT),
            handoff.AUDITED_TRAINING_SCRIPT_SHA256,
        )
        self.assertEqual(
            handoff._sha256_file(handoff.DATASET_CONTRACT_SCRIPT),
            handoff.AUDITED_DATASET_CONTRACT_SCRIPT_SHA256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.pt"
            path.write_bytes(b"expected")
            expected = handoff._sha256_file(path)
            self.assertEqual(handoff._require_hash(path, expected, "asset"), expected)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, "mismatch"):
                handoff._require_hash(path, expected, "asset")

    def test_preflight_emits_fresh_command_without_resume_or_test(self) -> None:
        name = "unit_cuda5060_fresh"
        report, command = handoff.prepare_handoff(
            model_size="n",
            run_name=name,
            device_index=0,
            torch_module=_FakeTorch(),
            version_getter=_versions,
            power_probe=lambda: True,
            dataset_verifier=lambda: {"content_sha256": handoff.AUDITED_DATASET_CONTENT_SHA256},
        )
        self.assertFalse(report["training_started"])
        self.assertEqual(report["power"]["ac"], "online")
        self.assertEqual(
            report["dataset_contract_script"]["sha256"],
            handoff.AUDITED_DATASET_CONTRACT_SCRIPT_SHA256,
        )
        self.assertIn("--skip-test", command)
        self.assertNotIn("--resume-from", command)
        self.assertNotIn("--adopt-interrupted-run", command)
        self.assertEqual(command[command.index("--name") + 1], name)
        self.assertEqual(command[command.index("--device") + 1], "0")
        parsed = trainer.config_from_args(trainer.build_parser().parse_args(command[3:]))
        self.assertEqual(parsed.epochs, 60)
        self.assertEqual(parsed.patience, 15)
        self.assertEqual(parsed.batch, 8)
        self.assertEqual(parsed.imgsz, 640)
        self.assertEqual(parsed.device, "0")
        self.assertEqual(parsed.workers, 4)
        self.assertEqual(parsed.threads, 6)
        self.assertEqual(parsed.cache, "none")
        self.assertEqual(parsed.seed, 0)
        self.assertFalse(parsed.run_test)
        self.assertIsNone(parsed.resume_from)
        training_arguments = trainer.training_arguments(parsed)
        self.assertTrue(training_arguments["deterministic"])
        self.assertFalse(training_arguments["amp"])
        self.assertFalse(training_arguments["rect"])
        self.assertEqual(training_arguments["save_period"], 1)

    def test_larger_candidate_uses_conservative_fixed_batch_on_eight_gb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            weights = project_root / "yolo26s.pt"
            weights.write_bytes(b"fake yolo26s unit-test checkpoint")
            original_contract = handoff.MODEL_CONTRACTS["s"]
            fake_contract = handoff.ModelContract(
                filename=original_contract.filename,
                sha256=handoff._sha256_file(weights),
                default_run_name=original_contract.default_run_name,
                training_batch=original_contract.training_batch,
            )
            with mock.patch.object(
                handoff, "PROJECT_ROOT", project_root
            ), mock.patch.dict(handoff.MODEL_CONTRACTS, {"s": fake_contract}):
                _report, command = handoff.prepare_handoff(
                    model_size="s",
                    run_name="unit_cuda5060_s_fresh",
                    device_index=0,
                    torch_module=_FakeTorch(),
                    version_getter=_versions,
                    power_probe=lambda: True,
                    dataset_verifier=lambda: {},
                )
            self.assertEqual(command[command.index("--batch") + 1], "4")
            self.assertEqual(command[command.index("--weights") + 1], str(weights))

    def test_production_preflight_uses_isolated_probe_without_importing_torch(self) -> None:
        gpu = handoff.verify_cuda_device(_FakeTorch(), 0)
        probe = mock.Mock(return_value=gpu)
        report, _command = handoff.prepare_handoff(
            model_size="n",
            run_name="unit_cuda5060_fresh",
            device_index=0,
            cuda_probe=probe,
            version_getter=_versions,
            power_probe=lambda: True,
            dataset_verifier=lambda: {},
        )
        probe.assert_called_once_with(
            0,
            handoff.PROJECT_ROOT / "yolo26n.pt",
            handoff.MODEL_CONTRACTS["n"].sha256,
            8,
        )
        self.assertEqual(report["gpu"], gpu)

    def test_preflight_rejects_offline_ac_and_existing_run(self) -> None:
        common = dict(
            model_size="n",
            run_name="unit_cuda5060_fresh",
            device_index=0,
            torch_module=_FakeTorch(),
            version_getter=_versions,
            dataset_verifier=lambda: {},
        )
        with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, "AC power is offline"):
            handoff.prepare_handoff(**common, power_probe=lambda: False)
        with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, "could not be confirmed"):
            handoff.prepare_handoff(**common, power_probe=lambda: None)

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / common["run_name"]).mkdir()
            with mock.patch.object(handoff, "TRAINING_PROJECT", project):
                with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, "already exists"):
                    handoff.prepare_handoff(**common, power_probe=lambda: True)

    def test_preflight_checks_ac_before_the_cuda_child(self) -> None:
        probe = mock.Mock(side_effect=AssertionError("CUDA probe must not run on battery"))
        with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, "AC power is offline"):
            handoff.prepare_handoff(
                model_size="n",
                run_name="unit_cuda5060_fresh",
                device_index=0,
                cuda_probe=probe,
                version_getter=_versions,
                power_probe=lambda: False,
                dataset_verifier=lambda: {},
            )
        probe.assert_not_called()

    def test_windows_sleep_inhibitor_is_scoped_and_fails_closed(self) -> None:
        set_state = mock.Mock(side_effect=(1, 1))
        kernel32 = type("Kernel32", (), {"SetThreadExecutionState": set_state})()
        windll = type("Windll", (), {"kernel32": kernel32})()
        with mock.patch.object(
            handoff.platform, "system", return_value="Windows"
        ), mock.patch.object(handoff.ctypes, "windll", windll, create=True):
            with handoff.training_sleep_inhibitor() as evidence:
                self.assertEqual(
                    evidence["status"], "active_until_training_process_exits"
                )
                self.assertEqual(set_state.call_args_list, [mock.call(0x80000001)])
        self.assertEqual(
            set_state.call_args_list,
            [mock.call(0x80000001), mock.call(0x80000000)],
        )

        set_state = mock.Mock(return_value=0)
        kernel32 = type("Kernel32", (), {"SetThreadExecutionState": set_state})()
        windll = type("Windll", (), {"kernel32": kernel32})()
        with mock.patch.object(
            handoff.platform, "system", return_value="Windows"
        ), mock.patch.object(handoff.ctypes, "windll", windll, create=True):
            with self.assertRaisesRegex(
                handoff.CudaTrainingHandoffError,
                "could not inhibit Windows idle sleep",
            ):
                with handoff.training_sleep_inhibitor():
                    self.fail("sleep inhibition failure must block execution")

    def test_exact_launch_snapshot_rehashes_and_binds_command(self) -> None:
        dataset = {"content_sha256": handoff.AUDITED_DATASET_CONTENT_SHA256}
        report, command = handoff.prepare_handoff(
            model_size="n",
            run_name="unit_cuda5060_fresh",
            device_index=0,
            torch_module=_FakeTorch(),
            version_getter=_versions,
            power_probe=lambda: True,
            dataset_verifier=lambda: dataset,
        )
        snapshot = handoff.revalidate_exact_launch_snapshot(
            report,
            command,
            model_size="n",
            run_name="unit_cuda5060_fresh",
            device_index=0,
            version_getter=_versions,
            power_probe=lambda: True,
            dataset_verifier=lambda: dataset,
        )
        self.assertEqual(snapshot["status"], "exact_launch_snapshot_revalidated")
        self.assertEqual(snapshot["base_checkpoint_sha256"], handoff.MODEL_CONTRACTS["n"].sha256)
        self.assertEqual(snapshot["ac"], "online")

        changed = list(command)
        changed[changed.index("--batch") + 1] = "99"
        with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, "command changed"):
            handoff.revalidate_exact_launch_snapshot(
                report,
                changed,
                model_size="n",
                run_name="unit_cuda5060_fresh",
                device_index=0,
                version_getter=_versions,
                power_probe=lambda: True,
                dataset_verifier=lambda: dataset,
            )

    def test_missing_training_project_is_safe_and_dry_run_does_not_create_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "missing" / "fort_cuh"
            with mock.patch.object(handoff, "TRAINING_PROJECT", project):
                report, _command = handoff.prepare_handoff(
                    model_size="n",
                    run_name="unit_cuda5060_fresh",
                    device_index=0,
                    torch_module=_FakeTorch(),
                    version_getter=_versions,
                    power_probe=lambda: True,
                    dataset_verifier=lambda: {},
                )
            self.assertEqual(
                report["fresh_run_directory"],
                str(project / "unit_cuda5060_fresh"),
            )
            self.assertFalse(project.exists())

    def test_project_path_rejects_existing_symlink_component(self) -> None:
        if os.name == "nt":
            self.skipTest("creating a symlink may require Windows Developer Mode")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            unsafe = root / "runs"
            unsafe.symlink_to(target, target_is_directory=True)
            with mock.patch.object(handoff, "TRAINING_PROJECT", unsafe / "fort_cuh"):
                with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, "unsafe"):
                    handoff._verify_new_run_directory("unit_cuda5060_fresh")

    def test_evidence_mkdir_never_follows_an_existing_symlink_component(self) -> None:
        if os.name == "nt":
            self.skipTest("creating a symlink may require Windows Developer Mode")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            unsafe = root / "runs"
            unsafe.symlink_to(target, target_is_directory=True)
            with mock.patch.object(handoff, "TRAINING_PROJECT", unsafe / "fort_cuh"):
                with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, "unsafe"):
                    handoff._ensure_evidence_directory()
            self.assertFalse((target / "fort_cuh").exists())

    def test_run_name_rejects_windows_aliases_on_every_platform(self) -> None:
        for name in ("fresh.", "CON.fresh", "nul.FRESH", "LPT1.fresh"):
            with self.subTest(name=name), self.assertRaises(
                handoff.CudaTrainingHandoffError
            ):
                handoff._safe_run_name(name)

    def test_execution_authorization_is_persistent_exclusive_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "runs" / "fort_cuh"
            report = {
                "status": "ready_fresh_cuda_training_not_started",
                "training_started": False,
                "base_checkpoint": {"sha256": "abc"},
                "command_argv": ["python", "train"],
            }
            with mock.patch.object(handoff, "TRAINING_PROJECT", project):
                evidence_path, evidence_sha = handoff.persist_execution_authorization(
                    report, "unit_cuda5060_fresh", power_probe=lambda: True
                )
                self.assertTrue(evidence_path.is_file())
                self.assertEqual(evidence_sha, handoff._sha256_file(evidence_path))
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                self.assertEqual(evidence["base_checkpoint"]["sha256"], "abc")
                self.assertEqual(
                    evidence["execution_authorization"]["run_name_confirmation"],
                    "unit_cuda5060_fresh",
                )
                with self.assertRaisesRegex(handoff.CudaTrainingHandoffError, "already exists"):
                    handoff.persist_execution_authorization(
                        report, "unit_cuda5060_fresh", power_probe=lambda: True
                    )

    def test_execution_authorization_rechecks_ac_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "runs" / "fort_cuh"
            with mock.patch.object(handoff, "TRAINING_PROJECT", project):
                with self.assertRaisesRegex(
                    handoff.CudaTrainingHandoffError, "not online at launch"
                ):
                    handoff.persist_execution_authorization(
                        {}, "unit_cuda5060_fresh", power_probe=lambda: False
                    )
            self.assertFalse(project.exists())

    def test_execution_authorization_rechecks_unused_run_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "runs" / "fort_cuh"
            (project / "unit_cuda5060_fresh").mkdir(parents=True)
            with mock.patch.object(handoff, "TRAINING_PROJECT", project):
                with self.assertRaisesRegex(
                    handoff.CudaTrainingHandoffError, "already exists"
                ):
                    handoff.persist_execution_authorization(
                        {}, "unit_cuda5060_fresh", power_probe=lambda: True
                    )
            self.assertFalse((project / ".cuda-training-handoffs").exists())

    def test_execution_lock_blocks_same_device_and_releases(self) -> None:
        if os.name == "nt":
            self.skipTest("the Linux test runner cannot validate msvcrt locking")
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "runs" / "fort_cuh"
            with mock.patch.object(handoff, "TRAINING_PROJECT", project):
                with handoff.training_execution_lock(0):
                    with self.assertRaisesRegex(
                        handoff.CudaTrainingHandoffError, "already holds device 0"
                    ):
                        with handoff.training_execution_lock(0):
                            self.fail("a second same-device lock was acquired")
                with handoff.training_execution_lock(0):
                    pass

    def test_execution_lock_rejects_symlink_without_touching_target(self) -> None:
        if os.name == "nt":
            self.skipTest("creating a symlink may require Windows Developer Mode")
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "runs" / "fort_cuh"
            evidence_directory = project / ".cuda-training-handoffs"
            evidence_directory.mkdir(parents=True)
            victim = Path(temporary) / "victim.txt"
            victim.write_text("unchanged\n", encoding="utf-8")
            (evidence_directory / "cuda-device-0.lock").symlink_to(victim)
            with mock.patch.object(handoff, "TRAINING_PROJECT", project):
                with self.assertRaisesRegex(
                    handoff.CudaTrainingHandoffError, "lock path is unsafe"
                ):
                    with handoff.training_execution_lock(0):
                        self.fail("an unsafe lock path was opened")
            self.assertEqual(victim.read_text(encoding="utf-8"), "unchanged\n")

    def test_execution_lock_rejects_hardlink_without_touching_target(self) -> None:
        if os.name == "nt":
            self.skipTest("hard-link permissions vary on Windows test hosts")
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "runs" / "fort_cuh"
            evidence_directory = project / ".cuda-training-handoffs"
            evidence_directory.mkdir(parents=True)
            victim = Path(temporary) / "victim.txt"
            victim.write_text("unchanged\n", encoding="utf-8")
            os.link(victim, evidence_directory / "cuda-device-0.lock")
            with mock.patch.object(handoff, "TRAINING_PROJECT", project):
                with self.assertRaisesRegex(
                    handoff.CudaTrainingHandoffError, "changed or is unsafe"
                ):
                    with handoff.training_execution_lock(0):
                        self.fail("an unsafe lock path was opened")
            self.assertEqual(victim.read_text(encoding="utf-8"), "unchanged\n")

    def test_execution_lock_uses_windows_nonblocking_byte_lock(self) -> None:
        calls: list[tuple[int, int]] = []

        class FakeMsvcrt:
            LK_NBLCK = 2
            LK_UNLCK = 0

            @staticmethod
            def locking(_file_descriptor: int, mode: int, length: int) -> None:
                calls.append((mode, length))

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "runs" / "fort_cuh"
            with mock.patch.object(handoff, "TRAINING_PROJECT", project), mock.patch.object(
                handoff.platform, "system", return_value="Windows"
            ), mock.patch.dict(sys.modules, {"msvcrt": FakeMsvcrt}):
                with handoff.training_execution_lock(0):
                    self.assertEqual(calls, [(FakeMsvcrt.LK_NBLCK, 1)])
        self.assertEqual(
            calls,
            [(FakeMsvcrt.LK_NBLCK, 1), (FakeMsvcrt.LK_UNLCK, 1)],
        )

    def test_execute_requires_exact_run_name_confirmation(self) -> None:
        report = {"command": "python train", "training_started": False}
        command = ["python", "train"]
        completed = type("Completed", (), {"returncode": 0})()
        with mock.patch.object(
            handoff, "prepare_handoff", return_value=(report, command)
        ) as prepare, mock.patch.object(
            handoff,
            "revalidate_exact_launch_snapshot",
            return_value={"status": "exact_launch_snapshot_revalidated"},
        ) as revalidate, mock.patch.object(
            handoff,
            "persist_execution_authorization",
            return_value=(Path("record.json"), "abc"),
        ) as persist, mock.patch.object(
            handoff,
            "training_execution_lock",
            return_value=nullcontext(Path("device.lock")),
        ), mock.patch.object(
            handoff,
            "training_sleep_inhibitor",
            return_value=nullcontext({"status": "active"}),
        ), mock.patch.object(
            handoff.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(handoff.main(["--execute"]), 2)
            run.assert_not_called()
            persist.assert_not_called()
            expected = handoff.MODEL_CONTRACTS["n"].default_run_name
            self.assertEqual(
                handoff.main(["--execute", "--confirm-run-name", expected]),
                0,
            )
            self.assertEqual(prepare.call_count, 3)
            revalidate.assert_called_once_with(
                report,
                command,
                model_size="n",
                run_name=expected,
                device_index=0,
            )
            self.assertEqual(
                report["final_launch_revalidation"]["status"],
                "exact_launch_snapshot_revalidated",
            )
            persist.assert_called_once_with(report, expected)
            run.assert_called_once_with(command, cwd=handoff.PROJECT_ROOT, check=False)

    def test_emission_only_main_never_locks_persists_or_launches(self) -> None:
        report = {"command": "python train", "training_started": False}
        command = ["python", "train"]
        with mock.patch.object(
            handoff, "prepare_handoff", return_value=(report, command)
        ), mock.patch.object(handoff, "training_execution_lock") as lock, mock.patch.object(
            handoff, "training_sleep_inhibitor"
        ) as sleep, mock.patch.object(
            handoff, "persist_execution_authorization"
        ) as persist, mock.patch.object(handoff.subprocess, "run") as run:
            self.assertEqual(handoff.main([]), 0)
        lock.assert_not_called()
        sleep.assert_not_called()
        persist.assert_not_called()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
