from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication, QMessageBox

from aiming.makcu import MakcuError
from aiming.makcu_calibration_session import CalibrationEvidenceError
from controller_precision.linux_evdev import ControllerCandidate
from detection.hardware import (
    Accelerator,
    AcceleratorKind,
    DirectMLAdapter,
    HardwareProfile,
    ProcessorInfo,
    Recommendation,
    Vendor,
)
from launcher.precision import candidate_identity
from launcher.qt_app import (
    LauncherWindow,
    PROAIM_BUILD_TAG,
    _has_exact_directml_binding,
    _unique_ready_gpu_plan,
)
import launcher.qt_app as qt_app
from launcher.settings import (
    AIM_OUTPUT_MAKCU,
    LauncherSettings,
    MODEL_PRESET_CUSTOM,
    MODEL_PRESET_FORT_PLAYER_BALANCED,
    MODEL_PRESET_FORT_PLAYER_BALANCED_INT8,
)


def controller() -> ControllerCandidate:
    return ControllerCandidate(
        path=Path("/dev/input/by-id/usb-PXN-test-event-joystick"),
        event_path=Path("/dev/input/event99"),
        name="PXN P5 8K",
        vendor=0x36E6,
        product=0x3016,
        serial="test-serial",
        phys="usb-test/input0",
        readable=True,
    )


def hardware_profile(
    *gpus: Accelerator,
    adapters: tuple[DirectMLAdapter, ...] = (),
) -> HardwareProfile:
    cpu = Accelerator(AcceleratorKind.CPU, Vendor.INTEL, "Test CPU")
    return HardwareProfile(
        system="windows",
        processor=ProcessorInfo("Test CPU", 16),
        accelerators=(cpu, *gpus),
        runtime_devices=("CPU",),
        directml_adapters=adapters,
    )


def recommendation(
    accelerator: Accelerator,
    *,
    backend: str,
    device: str,
    ready: bool = True,
) -> Recommendation:
    return Recommendation(
        accelerator=accelerator,
        backend=backend,
        device=device,
        precision="fp32",
        inference_size=416,
        ready=ready,
        reason="test",
    )


def successful_calibration_evidence() -> SimpleNamespace:
    x_fit = SimpleNamespace(gain_pixels_per_count=0.14)
    y_fit = SimpleNamespace(gain_pixels_per_count=0.10)
    fit = SimpleNamespace(x=x_fit, y=y_fit, delay_seconds=0.024)
    binding = SimpleNamespace(
        active_provider="MIGraphXExecutionProvider",
        active_device="gfx1030-RX6950XT",
        capture_width=1920,
        capture_height=1080,
        capture_fps=240.0,
        pixel_format="NV12",
        context_name="ads",
        aim_mode="ads",
    )
    return SimpleNamespace(
        outcome="success",
        evidence_complete=True,
        cleanup_error=None,
        fit=fit,
        binding=binding,
        artifact_sha256="a" * 64,
        core_evidence_sha256="b" * 64,
    )


class FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.pid = 4242
        self.stdout = None

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, _signal: int) -> None:
        self.returncode = -2

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class FakeMakcuVerifier:
    def __init__(self, masks: list[int]) -> None:
        self.masks = masks
        self.started = False
        self.stopped = False

    def start(self, *, output_loop: bool = True) -> None:
        self.started = True

    def poll_button_mask(self) -> int:
        return self.masks.pop(0) if self.masks else 0

    def stop(self) -> None:
        self.stopped = True


class QtLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def window(
        self,
        settings: LauncherSettings | None = None,
        *,
        controllers: tuple[ControllerCandidate, ...] = (),
    ) -> LauncherWindow:
        patcher = mock.patch("launcher.qt_app.pxn_controllers", return_value=controllers)
        with patcher:
            result = LauncherWindow(settings or LauncherSettings())
        self.addCleanup(result.close)
        return result

    def calibration_settings(self, **changes: object) -> LauncherSettings:
        values: dict[str, object] = {
            "source_mode": "screen",
            "aim": True,
            "aim_output": AIM_OUTPUT_MAKCU,
            "aim_label": "player",
            "ignore_self": True,
            "aim_makcu_port": "/dev/ttyACM0",
            "aim_makcu_button": "1",
            "aim_makcu_verified_port": "/dev/ttyACM0",
            "aim_makcu_verified_button": "1",
            "aim_makcu_context": "ads",
        }
        values.update(changes)
        return LauncherSettings(**values)  # type: ignore[arg-type]

    def test_screen_region_crop_output_and_shortcuts_are_exposed_and_collected(self) -> None:
        settings = LauncherSettings(
            model_preset=MODEL_PRESET_CUSTOM,
            model_path="/tmp/custom.xml",
            labels_path="/tmp/custom.txt",
            use_screen_region=True,
            screen_x="-1920",
            screen_y="25",
            screen_width="1600",
            screen_height="900",
            crop_size="720",
            detail_crop_size="640",
            output_format="traditional",
        )
        window = self.window(settings)

        collected = window.collect()

        self.assertTrue(window.use_screen_region.isChecked())
        self.assertTrue(all(widget.isEnabled() for widget in window._region_widgets))
        self.assertEqual(
            (
                collected.screen_x,
                collected.screen_y,
                collected.screen_width,
                collected.screen_height,
            ),
            ("-1920", "25", "1600", "900"),
        )
        self.assertEqual(collected.crop_size, "720")
        self.assertEqual(collected.detail_crop_size, "640")
        self.assertEqual(collected.output_format, "traditional")
        self.assertEqual(window._start_key.key().toString(), "F5")
        self.assertEqual(window._stop_key.key().toString(), "Esc")
        self.assertEqual(window.stack.count(), 4)

    def test_about_dialog_identifies_license_warranty_and_source(self) -> None:
        window = self.window()
        with mock.patch("launcher.qt_app.QMessageBox.about") as about:
            window._show_about()

        _parent, title, text = about.call_args.args
        self.assertEqual(title, "About ProAim")
        self.assertIn("AGPL-3.0-or-later", text)
        self.assertIn("without any warranty", text)
        self.assertIn("github.com/laedorp/game-detector", text)
        self.assertIn(PROAIM_BUILD_TAG, text)
        self.assertNotIn("2026-08-10-makcu-monitor-v1", text)

    def test_build_tag_is_neutral_in_source_and_uses_frozen_build_info(self) -> None:
        with mock.patch.object(qt_app.sys, "frozen", False, create=True):
            self.assertEqual(qt_app._build_tag(), "source checkout")

        payload = '{"commit":"0123456789abcdef","runtime_variant":"directml"}'
        with (
            mock.patch.object(qt_app.sys, "frozen", True, create=True),
            mock.patch.object(qt_app.Path, "read_text", return_value=payload),
        ):
            self.assertEqual(qt_app._build_tag(), "directml · 0123456789ab")

    def test_detail_pass_control_is_only_exposed_for_advanced_or_custom(self) -> None:
        window = self.window()

        self.assertTrue(window.detail_crop_size.isHidden())
        advanced_index = window.model_tier.findData("high")
        window.model_tier.setCurrentIndex(advanced_index)
        self.assertFalse(window.detail_crop_size.isHidden())

        balanced_index = window.model_tier.findData("mid")
        window.model_tier.setCurrentIndex(balanced_index)
        self.assertTrue(window.detail_crop_size.isHidden())

        custom_index = window.model_preset.findData(MODEL_PRESET_CUSTOM)
        window.model_preset.setCurrentIndex(custom_index)
        self.assertFalse(window.detail_crop_size.isHidden())

    def test_persisted_detail_pass_is_presented_as_advanced_workload(self) -> None:
        window = self.window(LauncherSettings(detail_crop_size="768"))

        self.assertEqual(window.model_tier.currentData(), "high")
        self.assertEqual(window.detail_crop_size.text(), "768")
        self.assertFalse(window.detail_crop_size.isHidden())

    def test_explicit_recommended_choice_applies_its_complete_detail_workload(self) -> None:
        window = self.window()
        custom_index = window.model_preset.findData(MODEL_PRESET_CUSTOM)
        recommended_index = window.model_preset.findData(
            MODEL_PRESET_FORT_PLAYER_BALANCED
        )
        with mock.patch(
            "launcher.qt_app.model_preset_detail_crop_text",
            side_effect=lambda key: (
                "768" if key == MODEL_PRESET_FORT_PLAYER_BALANCED else ""
            ),
        ):
            window.model_preset.setCurrentIndex(custom_index)
            window.model_preset.setCurrentIndex(recommended_index)

        self.assertEqual(window.detail_crop_size.text(), "768")
        self.assertFalse(window.detail_crop_size.isHidden())

    @unittest.skipIf(os.name == "nt", "Wayland preflight applies only on Linux")
    def test_wayland_screen_capture_is_rejected_before_launch(self) -> None:
        window = self.window(LauncherSettings(source_mode="screen"))
        with (
            mock.patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            mock.patch("launcher.qt_app.start_detector") as start,
            mock.patch.object(QApplication, "processEvents"),
            mock.patch("launcher.qt_app.QMessageBox.critical") as critical,
        ):
            window._start()

        start.assert_not_called()
        critical.assert_called_once()
        self.assertIn("X11/Xorg", window.status.text())

    def test_makcu_start_requires_matching_verified_port_and_button(self) -> None:
        settings = LauncherSettings(
            aim=True,
            aim_output=AIM_OUTPUT_MAKCU,
            aim_label="player",
            aim_makcu_port="/dev/ttyACM0",
            aim_makcu_button="1",
        )
        window = self.window(settings)
        with (
            mock.patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}),
            mock.patch("launcher.qt_app.start_detector") as start,
            mock.patch("launcher.qt_app.QMessageBox.warning") as warning,
        ):
            window._start()

        start.assert_not_called()
        self.assertIn("Verify MAKCU", warning.call_args.args[1])

    def test_makcu_verification_waits_for_release_before_binding(self) -> None:
        settings = LauncherSettings(
            aim=True,
            aim_output=AIM_OUTPUT_MAKCU,
            aim_label="player",
            aim_makcu_port="/dev/ttyACM0",
            aim_makcu_button="1",
        )
        window = self.window(settings)
        verifier = FakeMakcuVerifier([0, 0b00010, 0])
        with (
            mock.patch("launcher.qt_app.MakcuAimingController", return_value=verifier),
            mock.patch("launcher.qt_app.save_settings"),
        ):
            window._verify_makcu_activation_worker("/dev/ttyACM0", 1)

        self.assertTrue(verifier.started)
        self.assertTrue(verifier.stopped)
        self.assertEqual(window._makcu_verified_port, "/dev/ttyACM0")
        self.assertEqual(window._makcu_verified_button, "1")
        self.assertTrue(window._makcu_verification_matches())

    def test_makcu_shutdown_failure_overrides_verified_result_once(self) -> None:
        settings = LauncherSettings(
            aim=True,
            aim_output=AIM_OUTPUT_MAKCU,
            aim_label="player",
            aim_makcu_port="/dev/ttyACM0",
            aim_makcu_button="1",
        )
        window = self.window(settings)
        verifier = FakeMakcuVerifier([0, 0b00010, 0])
        verifier.stop = mock.Mock(  # type: ignore[method-assign]
            side_effect=MakcuError("serial worker did not stop")
        )
        results: list[tuple[bool, str, str, str]] = []
        window.makcu_verification_done.connect(
            lambda verified, port, button, detail: results.append(
                (verified, port, button, detail)
            )
        )

        with (
            mock.patch("launcher.qt_app.MakcuAimingController", return_value=verifier),
            mock.patch("launcher.qt_app.save_settings") as save,
            mock.patch("launcher.qt_app.QMessageBox.critical") as critical,
        ):
            window._verify_makcu_activation_worker("/dev/ttyACM0", 1)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0][0])
        self.assertIn("shutdown failed", results[0][3].lower())
        self.assertIn("serial worker did not stop", results[0][3])
        self.assertEqual(window._makcu_verified_port, "")
        self.assertEqual(window._makcu_verified_button, "")
        verifier.stop.assert_called_once_with()
        save.assert_not_called()
        critical.assert_called_once()

    def test_makcu_monitor_shutdown_failure_still_emits_done_once(self) -> None:
        window = self.window()
        window._makcu_monitor_cancel.set()
        verifier = FakeMakcuVerifier([])
        verifier.stop = mock.Mock(  # type: ignore[method-assign]
            side_effect=MakcuError("serial close timed out")
        )
        progress: list[str] = []
        completions: list[bool] = []
        window.makcu_verification_progress.connect(progress.append)
        window.makcu_monitor_done.connect(lambda: completions.append(True))

        with mock.patch(
            "launcher.qt_app.MakcuAimingController", return_value=verifier
        ):
            window._monitor_makcu_buttons_worker("/dev/ttyACM0")

        self.assertEqual(completions, [True])
        self.assertTrue(any("shutdown failed" in message.lower() for message in progress))
        self.assertTrue(any("serial close timed out" in message for message in progress))
        verifier.stop.assert_called_once_with()

    def test_makcu_verification_cannot_overlap_button_monitor(self) -> None:
        settings = LauncherSettings(
            aim=True,
            aim_output=AIM_OUTPUT_MAKCU,
            aim_makcu_port="/dev/ttyACM0",
        )
        window = self.window(settings)
        monitor_thread = mock.Mock()
        monitor_thread.is_alive.return_value = True
        window._makcu_monitor_thread = monitor_thread

        window._update_aim_state()

        self.assertFalse(window.verify_makcu_button.isEnabled())
        with mock.patch("launcher.qt_app.threading.Thread") as thread:
            window.verify_makcu_activation()
        thread.assert_not_called()

    def test_makcu_button_monitor_cannot_overlap_verification(self) -> None:
        settings = LauncherSettings(
            aim=True,
            aim_output=AIM_OUTPUT_MAKCU,
            aim_makcu_port="/dev/ttyACM0",
        )
        window = self.window(settings)
        verify_thread = mock.Mock()
        verify_thread.is_alive.return_value = True

        with mock.patch(
            "launcher.qt_app.threading.Thread", return_value=verify_thread
        ) as thread:
            window.verify_makcu_activation()
            self.assertFalse(window.verify_makcu_button.isEnabled())
            self.assertFalse(window.monitor_makcu_button.isEnabled())
            window.monitor_makcu_buttons()

        thread.assert_called_once()
        verify_thread.start.assert_called_once_with()
        self.assertIsNone(window._makcu_monitor_thread)

    def test_cancelled_makcu_monitor_completes_normally_once(self) -> None:
        window = self.window()
        window._makcu_monitor_cancel.set()
        verifier = FakeMakcuVerifier([])
        completions: list[bool] = []
        window.makcu_monitor_done.connect(lambda: completions.append(True))

        with mock.patch(
            "launcher.qt_app.MakcuAimingController", return_value=verifier
        ):
            window._monitor_makcu_buttons_worker("/dev/ttyACM0")

        self.assertEqual(completions, [True])
        self.assertTrue(verifier.started)
        self.assertTrue(verifier.stopped)

    def test_makcu_press_without_release_never_binds_verification(self) -> None:
        settings = LauncherSettings(
            aim=True,
            aim_output=AIM_OUTPUT_MAKCU,
            aim_label="player",
            aim_makcu_port="/dev/ttyACM0",
            aim_makcu_button="1",
        )
        window = self.window(settings)
        verifier = FakeMakcuVerifier([0, 0b00010])

        def press_then_cancel() -> int:
            window._makcu_verify_cancel.set()
            return 0b00010

        verifier.poll_button_mask = press_then_cancel  # type: ignore[method-assign]
        with (
            mock.patch("launcher.qt_app.MakcuAimingController", return_value=verifier),
            mock.patch("launcher.qt_app.QMessageBox.critical"),
        ):
            window._verify_makcu_activation_worker("/dev/ttyACM0", 1)

        self.assertEqual(window._makcu_verified_port, "")
        self.assertEqual(window._makcu_verified_button, "")
        self.assertFalse(window._makcu_verification_matches())

    def test_left_makcu_button_keeps_zero_index(self) -> None:
        window = self.window()
        window.aim_makcu_button.setCurrentIndex(0)

        self.assertEqual(window._selected_makcu_button(), 0)

    def test_makcu_verification_rejects_held_or_wrong_button(self) -> None:
        settings = LauncherSettings(
            aim=True,
            aim_output=AIM_OUTPUT_MAKCU,
            aim_label="player",
            aim_makcu_port="/dev/ttyACM0",
            aim_makcu_button="1",
        )
        for masks in ([0b00010], [0, 0b00001, 0]):
            with self.subTest(masks=masks):
                window = self.window(settings)
                verifier = FakeMakcuVerifier(list(masks))
                with (
                    mock.patch(
                        "launcher.qt_app.MakcuAimingController",
                        return_value=verifier,
                    ),
                    mock.patch("launcher.qt_app.QMessageBox.critical"),
                ):
                    window._verify_makcu_activation_worker("/dev/ttyACM0", 1)

                self.assertFalse(window._makcu_verification_matches())

    def test_private_calibration_paths_are_randomized_private_and_nonexisting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings_target = Path(temporary) / "proaim" / "settings.json"
            with (
                mock.patch("launcher.qt_app.settings_path", return_value=settings_target),
                mock.patch(
                    "launcher.qt_app.secrets.token_hex",
                    side_effect=("1" * 32, "2" * 32),
                ),
            ):
                first = qt_app._private_calibration_path("evidence", "session")
                self.assertFalse(os.path.lexists(first))
                self.assertEqual(first.name, f"session-{'1' * 32}.json")
                if os.name == "posix":
                    self.assertEqual(stat.S_IMODE(first.parent.stat().st_mode), 0o700)
                first.write_bytes(b"reserved")
                second = qt_app._private_calibration_path("evidence", "session")

            self.assertNotEqual(first, second)
            self.assertFalse(os.path.lexists(second))
            # Windows may canonicalize RUNNER~1 to its long path spelling
            # inside _private_calibration_path(). Compare resolved components,
            # not lexical prefixes, while still proving containment.
            self.assertEqual(
                second.parents[2],
                settings_target.parent.resolve(),
            )

    def test_calibration_motion_warning_defaults_to_no_and_creates_nothing(self) -> None:
        window = self.window(self.calibration_settings())
        with (
            mock.patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}),
            mock.patch(
                "launcher.qt_app.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ) as question,
            mock.patch("launcher.qt_app._private_calibration_path") as private_path,
            mock.patch("launcher.qt_app.start_detector") as start,
        ):
            window.start_makcu_calibration()

        self.assertEqual(
            question.call_args.args[-1],
            QMessageBox.StandardButton.No,
        )
        self.assertIn("Normal Start works without", question.call_args.args[2])
        self.assertIn("automatic plant-aware", question.call_args.args[2])
        self.assertIn("never activated automatically", question.call_args.args[2])
        self.assertIn("KEEP Right Mouse RELEASED", question.call_args.args[2])
        self.assertIn("Release confirmation is automatic", question.call_args.args[2])
        self.assertNotIn("press Right once", question.call_args.args[2])
        private_path.assert_not_called()
        start.assert_not_called()
        self.assertIsNone(window.calibration_process)
        self.assertEqual(window.settings.aim_makcu_active_profile, "")

    def test_calibration_ui_marks_advanced_measurement_optional(self) -> None:
        window = self.window(self.calibration_settings())

        self.assertEqual(
            window.calibrate_makcu_button.text(),
            "Optional advanced response calibration…",
        )
        self.assertIn(
            "Normal Start does not require",
            window.aim_makcu_calibration_help.text(),
        )
        self.assertIn(
            "automatic plant-aware",
            window.aim_makcu_calibration_help.text(),
        )
        self.assertIn(
            "Optional advanced response calibration",
            window.aim_makcu_calibration_step.text(),
        )
        self.assertIn(
            "Normal Start already works without calibration",
            window.aim_makcu_calibration_instruction.text(),
        )

    def test_calibration_guide_makes_each_physical_action_visible(self) -> None:
        window = self.window(self.calibration_settings())

        self.assertIn(
            "Optional advanced response calibration",
            window.aim_makcu_calibration_step.text(),
        )
        self.assertIn(
            "Before calibration",
            window.aim_makcu_calibration_instruction.text(),
        )
        self.assertIn("Keep Right Mouse released", window.aim_makcu_calibration_instruction.text())

        window._handle_calibration_output_line(
            "MAKCU calibration: Exclusive mode armed. Keep activation released; "
            "after Release confirmed, press and continuously hold it.\n"
        )
        self.assertIn("MAKCU armed", window.aim_makcu_calibration_step.text())
        self.assertIn("Keep Right Mouse fully released", window.aim_makcu_calibration_instruction.text())
        self.assertEqual(window.aim_makcu_calibration_progress.value(), 2)

        window._handle_calibration_output_line(
            "MAKCU calibration: Waiting for a fresh MAKCU activation report; "
            "movement is disarmed.\n"
        )
        self.assertIn("fresh button report", window.aim_makcu_calibration_step.text())
        self.assertIn("Tap Right Mouse once", window.aim_makcu_calibration_instruction.text())

        window._handle_calibration_output_line(
            "MAKCU calibration: Release confirmed and target ready. Press and "
            "continuously hold activation.\n"
        )
        self.assertIn("READY TO HOLD", window.aim_makcu_calibration_step.text())
        self.assertIn("stable target is ready", window.aim_makcu_calibration_instruction.text())
        self.assertEqual(window.aim_makcu_calibration_progress.value(), 3)

        window._handle_calibration_output_line(
            "MAKCU calibration target: target wait: no exact player detection\n"
        )
        self.assertIn("Waiting for a safe target", window.aim_makcu_calibration_step.text())
        self.assertIn("Keep Right Mouse released", window.aim_makcu_calibration_instruction.text())

        window._handle_calibration_output_line(
            "MAKCU calibration target: target ready: center-nearest of 2 exact detections\n"
        )
        self.assertIn("READY TO HOLD", window.aim_makcu_calibration_step.text())

        window._handle_calibration_output_line(
            "MAKCU calibration: Hold detected. Keep holding while the selected "
            "aim mode settles. No movement is authorized yet.\n"
        )
        self.assertIn("settling the aim view", window.aim_makcu_calibration_step.text())
        self.assertIn("300 ms", window.aim_makcu_calibration_instruction.text())
        self.assertIn(
            "No movement is authorized",
            window.aim_makcu_calibration_instruction.text(),
        )

        window._handle_calibration_output_line(
            "MAKCU calibration: Keep holding activation; waiting for one safe exact "
            "target (target wait: no exact player detection). No movement is "
            "authorized yet.\n"
        )
        self.assertIn("waiting for a safe target", window.aim_makcu_calibration_step.text())
        self.assertIn("No movement is authorized", window.aim_makcu_calibration_instruction.text())

        window._handle_calibration_output_line(
            "MAKCU calibration: Hold still while the stationary baseline settles.\n"
        )
        self.assertIn("Measuring response", window.aim_makcu_calibration_step.text())
        self.assertIn("Keep Right Mouse held", window.aim_makcu_calibration_instruction.text())
        self.assertEqual(window.aim_makcu_calibration_progress.value(), 4)

    def test_calibration_summary_parses_readiness_between_state_and_button(self) -> None:
        window = self.window(self.calibration_settings())

        window._handle_calibration_output_line(
            "FPS 160 | CAL wait_hold | target wait: self-avatar safety is not "
            "ready | raw button released | counts 0/2400\n"
        )
        self.assertIn("Waiting for a safe target", window.aim_makcu_calibration_step.text())
        window._handle_calibration_output_line(
            "FPS 160 | CAL wait_hold | target ready: center-nearest of 2 exact "
            "detections | raw button released | counts 0/2400\n"
        )
        self.assertIn("READY TO HOLD", window.aim_makcu_calibration_step.text())

    def test_known_release_without_post_entry_frame_prompts_one_tap(self) -> None:
        window = self.window(self.calibration_settings())

        window._handle_calibration_output_line(
            "MAKCU calibration: Waiting for a post-entry framed MAKCU button "
            "report; movement is disarmed.\n"
        )

        self.assertIn("fresh button report", window.aim_makcu_calibration_step.text())
        self.assertIn("Tap Right Mouse once", window.aim_makcu_calibration_instruction.text())

    def test_release_confirmed_waiting_message_beats_keep_released_parser(self) -> None:
        window = self.window(self.calibration_settings())

        window._handle_calibration_output_line(
            "MAKCU calibration: Release confirmed. Keep activation released; "
            "waiting for a safe target (no exact target observation was available).\n"
        )

        self.assertTrue(window._calibration_release_confirmed)
        self.assertIn("Waiting for a safe target", window.aim_makcu_calibration_step.text())
        self.assertNotIn("MAKCU armed", window.aim_makcu_calibration_step.text())

    def test_target_ready_updates_do_not_override_post_hold_settle(self) -> None:
        window = self.window(self.calibration_settings())
        window._handle_calibration_output_line(
            "MAKCU calibration: Release confirmed and target ready. Press and "
            "continuously hold activation.\n"
        )
        window._handle_calibration_output_line(
            "MAKCU calibration: Hold detected. Keep holding while the selected "
            "aim mode settles. No movement is authorized yet.\n"
        )

        window._handle_calibration_output_line(
            "MAKCU calibration target: target ready: center-nearest of 2 exact "
            "detections\n"
        )
        self.assertIn("settling the aim view", window.aim_makcu_calibration_step.text())
        window._handle_calibration_output_line(
            "FPS 160 | CAL wait_hold | target ready: center-nearest of 2 exact "
            "detections | raw button pressed | counts 0/2400\n"
        )
        self.assertIn("settling the aim view", window.aim_makcu_calibration_step.text())
        self.assertNotIn("READY TO HOLD", window.aim_makcu_calibration_step.text())

    def test_calibration_runtime_failure_stays_actionable_after_child_exit(self) -> None:
        window = self.window(self.calibration_settings())
        process = FakeProcess(returncode=2)
        window.calibration_process = process  # type: ignore[assignment]
        window._calibration_evidence_path = Path("/private/aborted.json")
        window._calibration_launch_arguments = tuple(
            window.collect().detector_arguments()
        )
        window._calibration_context = "ads"

        window._handle_calibration_output_line(
            "\x1b[31mMAKCU calibration: fresh activation lacked a safe target: "
            "no exact target observation was available\x1b[0m\n"
        )
        with mock.patch("launcher.qt_app.QMessageBox.warning"):
            window._finish_calibration_process(process, 2)  # type: ignore[arg-type]

        self.assertIn("nothing was activated", window.aim_makcu_calibration_step.text())
        self.assertIn("fully visible stationary player box", window.aim_makcu_calibration_instruction.text())
        self.assertIn("safe target", window.aim_makcu_calibration_status.text())

    def test_calibration_context_is_explicit_noneditable_and_collected(self) -> None:
        window = self.window(self.calibration_settings())

        self.assertFalse(window.aim_makcu_context.isEditable())
        self.assertEqual(
            [
                window.aim_makcu_context.itemData(index)
                for index in range(window.aim_makcu_context.count())
            ],
            ["hip", "ads"],
        )
        self.assertEqual(window.aim_makcu_context.currentData(), "ads")
        self.assertEqual(window.collect().aim_makcu_context, "ads")

    def test_calibration_uses_a_separate_managed_child_and_exact_private_path(self) -> None:
        window = self.window(self.calibration_settings())
        process = FakeProcess()
        with tempfile.TemporaryDirectory() as temporary:
            settings_target = Path(temporary) / "proaim" / "settings.json"
            with (
                mock.patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}),
                mock.patch(
                    "launcher.qt_app.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Yes,
                ),
                mock.patch(
                    "launcher.qt_app.settings_path",
                    return_value=settings_target,
                ),
                mock.patch(
                    "launcher.qt_app.start_detector",
                    return_value=process,
                ) as start,
            ):
                window.start_makcu_calibration()

            command = start.call_args.args[0]
            evidence = Path(
                command[command.index("--aim-calibration-evidence") + 1]
            )
            self.assertTrue(evidence.is_absolute())
            self.assertFalse(os.path.lexists(evidence))
            self.assertEqual(evidence.parent.name, "evidence")
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(evidence.parent.stat().st_mode), 0o700)
            self.assertNotIn("--aim-makcu-active-profile", command)
            self.assertEqual(command.count("--aim-calibration-context"), 1)
            self.assertEqual(
                command[command.index("--aim-calibration-context") + 1],
                "ads",
            )
            self.assertIs(window.calibration_process, process)
            self.assertIsNone(window.process)
            self.assertFalse(window.start_button.isEnabled())
            self.assertTrue(window.stop_button.isEnabled())
            self.assertIn("Starting the GPU model", window.aim_makcu_calibration_step.text())
            self.assertIn("30–40 seconds", window.aim_makcu_calibration_instruction.text())

        window.calibration_process = None

    def test_exit_zero_strictly_stages_success_without_auto_activation(self) -> None:
        window = self.window(self.calibration_settings())
        process = FakeProcess(returncode=0)
        evidence_path = Path("/private/session.json")
        evidence = successful_calibration_evidence()
        window.calibration_process = process  # type: ignore[assignment]
        window._calibration_evidence_path = evidence_path
        window._calibration_launch_arguments = tuple(
            window.collect().detector_arguments()
        )
        window._calibration_context = "ads"

        with (
            mock.patch(
                "launcher.qt_app.load_session_evidence",
                return_value=evidence,
            ) as strict_load,
            mock.patch("launcher.qt_app.activate_session_evidence_file") as activate,
            mock.patch("launcher.qt_app.QMessageBox.information"),
        ):
            window._poll_calibration_process()

        strict_load.assert_called_once_with(evidence_path)
        activate.assert_not_called()
        self.assertIs(window._pending_calibration_evidence, evidence)
        self.assertEqual(window._pending_calibration_path, evidence_path)
        self.assertEqual(window.settings.aim_makcu_active_profile, "")
        self.assertTrue(window.activate_makcu_calibration_button.isEnabled())
        self.assertIn("not active yet", window.aim_makcu_calibration_status.text())

    def test_activation_requires_second_default_no_confirmation(self) -> None:
        window = self.window(self.calibration_settings())
        evidence = successful_calibration_evidence()
        evidence_path = Path("/private/session.json")
        window._pending_calibration_evidence = evidence  # type: ignore[assignment]
        window._pending_calibration_path = evidence_path
        window._calibration_launch_arguments = tuple(
            window.collect().detector_arguments()
        )
        window._calibration_context = "ads"
        with (
            mock.patch(
                "launcher.qt_app.load_session_evidence",
                return_value=evidence,
            ),
            mock.patch(
                "launcher.qt_app.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ) as question,
            mock.patch("launcher.qt_app.activate_session_evidence_file") as activate,
        ):
            window.activate_staged_makcu_calibration()

        self.assertEqual(
            question.call_args.args[-1],
            QMessageBox.StandardButton.No,
        )
        activate.assert_not_called()
        self.assertIs(window._pending_calibration_evidence, evidence)
        self.assertEqual(window.settings.aim_makcu_active_profile, "")

    def test_second_confirmation_activates_privately_and_persists_selection(self) -> None:
        window = self.window(self.calibration_settings())
        evidence = successful_calibration_evidence()
        evidence_path = Path("/private/session.json")
        window._pending_calibration_evidence = evidence  # type: ignore[assignment]
        window._pending_calibration_path = evidence_path
        window._calibration_launch_arguments = tuple(
            window.collect().detector_arguments()
        )
        window._calibration_context = "ads"
        profile = SimpleNamespace(
            session_artifact_sha256=evidence.artifact_sha256,
            core_evidence_sha256=evidence.core_evidence_sha256,
            binding=evidence.binding,
            fit=evidence.fit,
        )
        with tempfile.TemporaryDirectory() as temporary:
            active_path = Path(temporary) / "private" / "profile.json"
            settings_target = Path(temporary) / "settings.json"
            with (
                mock.patch(
                    "launcher.qt_app.load_session_evidence",
                    return_value=evidence,
                ) as strict_load,
                mock.patch(
                    "launcher.qt_app.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Yes,
                ),
                mock.patch(
                    "launcher.qt_app._private_calibration_path",
                    return_value=active_path,
                ),
                mock.patch(
                    "launcher.qt_app.activate_session_evidence_file",
                    return_value=profile,
                ) as activate,
                mock.patch(
                    "launcher.qt_app.settings_path",
                    return_value=settings_target,
                ),
                mock.patch("launcher.qt_app.save_settings") as save,
                mock.patch("launcher.qt_app.QMessageBox.information"),
            ):
                window.activate_staged_makcu_calibration()

        strict_load.assert_called_once_with(evidence_path)
        activate.assert_called_once_with(
            evidence_path,
            active_path,
            expected_binding=evidence.binding,
        )
        save.assert_called_once()
        self.assertEqual(
            window.settings.aim_makcu_active_profile,
            str(active_path.resolve()),
        )
        self.assertIsNone(window._pending_calibration_evidence)
        self.assertFalse(window.activate_makcu_calibration_button.isEnabled())
        self.assertTrue(window.aim_makcu_vertical_rate_ratio.isEnabled())
        self.assertIn(
            "Measured plant-aware profile",
            window.aim_makcu_control_mode_note.text(),
        )

    def test_activation_save_failure_restores_selection_and_removes_new_profile(self) -> None:
        window = self.window(self.calibration_settings())
        evidence = successful_calibration_evidence()
        evidence_path = Path("/private/session.json")
        window._pending_calibration_evidence = evidence  # type: ignore[assignment]
        window._pending_calibration_path = evidence_path
        window._calibration_launch_arguments = tuple(
            window.collect().detector_arguments()
        )
        window._calibration_context = "ads"
        old_selection = "/private/old-profile.json"
        window.settings.aim_makcu_active_profile = old_selection
        profile = SimpleNamespace(
            session_artifact_sha256=evidence.artifact_sha256,
            core_evidence_sha256=evidence.core_evidence_sha256,
            binding=evidence.binding,
            fit=evidence.fit,
        )
        with tempfile.TemporaryDirectory() as temporary:
            active_path = Path(temporary) / "profile.json"

            def publish(*_args: object, **_kwargs: object) -> object:
                active_path.write_bytes(b"published")
                return profile

            with (
                mock.patch(
                    "launcher.qt_app.load_session_evidence",
                    return_value=evidence,
                ),
                mock.patch(
                    "launcher.qt_app.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Yes,
                ),
                mock.patch(
                    "launcher.qt_app._private_calibration_path",
                    return_value=active_path,
                ),
                mock.patch(
                    "launcher.qt_app.activate_session_evidence_file",
                    side_effect=publish,
                ),
                mock.patch(
                    "launcher.qt_app.save_settings",
                    side_effect=OSError("settings unavailable"),
                ),
                mock.patch("launcher.qt_app.QMessageBox.critical") as critical,
            ):
                window.activate_staged_makcu_calibration()

            self.assertFalse(active_path.exists())
        self.assertEqual(window.settings.aim_makcu_active_profile, old_selection)
        self.assertIs(window._pending_calibration_evidence, evidence)
        critical.assert_called_once()

    def test_activation_rejects_changed_or_malformed_staged_evidence(self) -> None:
        window = self.window(self.calibration_settings())
        evidence = successful_calibration_evidence()
        window._pending_calibration_evidence = evidence  # type: ignore[assignment]
        window._pending_calibration_path = Path("/private/session.json")
        window._calibration_launch_arguments = tuple(
            window.collect().detector_arguments()
        )
        window._calibration_context = "ads"
        with (
            mock.patch(
                "launcher.qt_app.load_session_evidence",
                side_effect=CalibrationEvidenceError("artifact hash changed"),
            ),
            mock.patch("launcher.qt_app.activate_session_evidence_file") as activate,
            mock.patch("launcher.qt_app.save_settings") as save,
            mock.patch("launcher.qt_app.QMessageBox.critical") as critical,
        ):
            window.activate_staged_makcu_calibration()

        activate.assert_not_called()
        save.assert_not_called()
        critical.assert_called_once()
        self.assertIsNone(window._pending_calibration_evidence)
        self.assertEqual(window.settings.aim_makcu_active_profile, "")

    def test_nonzero_stopped_missing_malformed_or_aborted_never_stage_or_activate(self) -> None:
        cases: tuple[tuple[str, int, bool, object], ...] = (
            ("nonzero", 7, False, successful_calibration_evidence()),
            ("stopped", 0, True, successful_calibration_evidence()),
            ("missing", 0, False, FileNotFoundError("missing")),
            (
                "malformed",
                0,
                False,
                CalibrationEvidenceError("malformed"),
            ),
            (
                "aborted",
                0,
                False,
                SimpleNamespace(
                    outcome="aborted",
                    evidence_complete=True,
                    cleanup_error=None,
                    fit=None,
                    core_evidence_sha256=None,
                ),
            ),
        )
        for name, return_code, stopped, load_result in cases:
            with self.subTest(name=name):
                window = self.window(self.calibration_settings())
                process = FakeProcess(returncode=return_code)
                window.calibration_process = process  # type: ignore[assignment]
                window._calibration_stop_requested = stopped
                window._calibration_evidence_path = Path(f"/{name}.json")
                window._calibration_launch_arguments = ("launch",)
                window._calibration_context = "ads"
                load_patch = mock.patch("launcher.qt_app.load_session_evidence")
                with (
                    load_patch as strict_load,
                    mock.patch("launcher.qt_app.activate_session_evidence_file") as activate,
                    mock.patch("launcher.qt_app.QMessageBox.warning"),
                ):
                    if isinstance(load_result, BaseException):
                        strict_load.side_effect = load_result
                    else:
                        strict_load.return_value = load_result
                    window._finish_calibration_process(process, return_code)  # type: ignore[arg-type]

                if name in {"nonzero", "stopped"}:
                    strict_load.assert_not_called()
                activate.assert_not_called()
                self.assertIsNone(window._pending_calibration_evidence)
                self.assertEqual(window.settings.aim_makcu_active_profile, "")

    def test_calibration_is_mutually_exclusive_with_all_live_workers(self) -> None:
        window = self.window(self.calibration_settings())
        running = FakeProcess()
        window.process = running  # type: ignore[assignment]
        with (
            mock.patch("launcher.qt_app.QMessageBox.information") as information,
            mock.patch("launcher.qt_app.QMessageBox.question") as question,
            mock.patch("launcher.qt_app.start_detector") as start,
        ):
            window.start_makcu_calibration()
        information.assert_called_once()
        question.assert_not_called()
        start.assert_not_called()
        window.process = None

        window.calibration_process = running  # type: ignore[assignment]
        with (
            mock.patch("launcher.qt_app.QMessageBox.information"),
            mock.patch("launcher.qt_app.start_detector") as start,
            mock.patch("launcher.qt_app.threading.Thread") as thread,
            mock.patch.object(window, "_start_precision_child") as precision,
        ):
            window._start()
            window.verify_makcu_activation()
            window.monitor_makcu_buttons()
            window.verify_precision_mapping()
            window.start_precision()

        start.assert_not_called()
        thread.assert_not_called()
        precision.assert_not_called()
        window.calibration_process = None

    def test_detail_pass_blocks_calibration_instead_of_silently_changing_it(self) -> None:
        window = self.window(self.calibration_settings(detail_crop_size="640"))
        with (
            mock.patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}),
            mock.patch("launcher.qt_app.QMessageBox.warning") as warning,
            mock.patch("launcher.qt_app.QMessageBox.question") as question,
            mock.patch("launcher.qt_app.start_detector") as start,
        ):
            window.start_makcu_calibration()

        self.assertIn("detail pass", warning.call_args.args[1].lower())
        question.assert_not_called()
        start.assert_not_called()

    def test_changing_context_unstages_evidence(self) -> None:
        window = self.window(self.calibration_settings())
        window._pending_calibration_evidence = successful_calibration_evidence()  # type: ignore[assignment]
        window._pending_calibration_path = Path("/private/session.json")
        window._calibration_launch_arguments = ("launch",)
        window._calibration_context = "ads"

        window.aim_makcu_context.setCurrentIndex(
            window.aim_makcu_context.findData("hip")
        )

        self.assertIsNone(window._pending_calibration_evidence)
        self.assertIsNone(window._pending_calibration_path)
        self.assertIn("no longer staged", window.aim_makcu_calibration_status.text())

    def test_normal_start_waits_for_live_makcu_monitor_but_allows_precision(self) -> None:
        window = self.window(self.calibration_settings())
        monitor = mock.Mock()
        monitor.is_alive.return_value = True
        window._makcu_monitor_thread = monitor
        with (
            mock.patch("launcher.qt_app.QMessageBox.information") as information,
            mock.patch("launcher.qt_app.start_detector") as start,
        ):
            window._start()

        start.assert_not_called()
        self.assertIn("monitor", information.call_args.args[1].lower())

        window._makcu_monitor_thread = None
        window.precision_process = FakeProcess()
        launched = FakeProcess()
        with (
            mock.patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}),
            mock.patch("launcher.qt_app.save_settings"),
            mock.patch("launcher.qt_app.start_detector", return_value=launched) as start,
        ):
            window._start()

        start.assert_called_once()
        self.assertIs(window.process, launched)
        window.process = None
        window.precision_process = None

    def test_escape_stop_escalates_calibration_without_activating(self) -> None:
        window = self.window(self.calibration_settings())
        process = FakeProcess()
        window.calibration_process = process  # type: ignore[assignment]
        callbacks: list[object] = []
        with (
            mock.patch("launcher.qt_app.request_stop") as request,
            mock.patch("launcher.qt_app.force_stop") as force,
            mock.patch("launcher.qt_app.kill_process") as kill,
            mock.patch("launcher.qt_app.activate_session_evidence_file") as activate,
            mock.patch(
                "launcher.qt_app.QTimer.singleShot",
                side_effect=lambda _delay, callback: callbacks.append(callback),
            ),
        ):
            window._stop()
            callbacks[0]()  # type: ignore[operator]
            callbacks[1]()  # type: ignore[operator]

        request.assert_called_once_with(process)
        force.assert_called_once_with(process)
        kill.assert_called_once_with(process)
        activate.assert_not_called()
        self.assertTrue(window._calibration_stop_requested)
        window.calibration_process = None

    def test_hardware_selection_preserves_player_model_semantics(self) -> None:
        window = self.window()
        window.detected_accelerator.clear()
        window.detected_accelerator.addItem(
            "GPU",
            {
                "ready": True,
                "backend": "onnxruntime",
                "device": "CUDAExecutionProvider",
                "inference_size": 640,
                "precision": "fp16",
                "tier": "high",
            },
        )

        window._apply_detected_accelerator()

        self.assertEqual(window.model_preset.currentData(), "fort_player_balanced")
        self.assertEqual(window.inference_size.currentText(), "416")

    def test_fresh_profile_schedules_nonblocking_first_hardware_scan(self) -> None:
        callbacks: list[object] = []
        settings = LauncherSettings()
        with (
            mock.patch("launcher.qt_app.load_settings", return_value=settings),
            mock.patch("launcher.qt_app.pxn_controllers", return_value=()),
            mock.patch(
                "launcher.qt_app.QTimer.singleShot",
                side_effect=lambda _delay, callback: callbacks.append(callback),
            ),
        ):
            window = LauncherWindow()
        self.addCleanup(window.close)

        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0].__name__, "_begin_first_hardware_scan")
        self.assertFalse(window._hardware_selection_configured)
        self.assertIsNone(window.process)

    def test_explicit_loaded_runtime_is_never_scheduled_for_auto_replacement(self) -> None:
        settings = LauncherSettings(
            backend="onnxruntime",
            device="CUDA",
            hardware_selection_configured=True,
        )
        with (
            mock.patch("launcher.qt_app.load_settings", return_value=settings),
            mock.patch("launcher.qt_app.pxn_controllers", return_value=()),
            mock.patch("launcher.qt_app.QTimer.singleShot") as single_shot,
        ):
            window = LauncherWindow()
        self.addCleanup(window.close)

        single_shot.assert_not_called()
        self.assertEqual(window.backend.currentText(), "onnxruntime")
        self.assertEqual(window.device.currentText(), "CUDA")

    def test_default_qt_load_adopts_legacy_profile_without_auto_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "proaim" / "settings.json"
            legacy = root / "game-detector" / "settings.json"
            legacy.parent.mkdir(parents=True)
            legacy.write_text(
                json.dumps(
                    {
                        "version": 6,
                        "backend": "onnxruntime",
                        "device": "CUDA",
                        "screen_width": "1777",
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch(
                    "launcher.settings._settings_paths",
                    return_value=(current, legacy),
                ),
                mock.patch("launcher.qt_app.pxn_controllers", return_value=()),
                mock.patch("launcher.qt_app.QTimer.singleShot") as single_shot,
            ):
                window = LauncherWindow()
        self.addCleanup(window.close)

        self.assertEqual(window.backend.currentText(), "onnxruntime")
        self.assertEqual(window.device.currentText(), "CUDA")
        self.assertEqual(window.screen_width.text(), "1777")
        self.assertTrue(window._hardware_selection_configured)
        single_shot.assert_not_called()

    def test_first_run_applies_only_exact_unique_directml_gpu(self) -> None:
        gpu = Accelerator(
            AcceleratorKind.GPU,
            Vendor.NVIDIA,
            "RTX 5060 Laptop GPU",
            "10de:2d59",
            True,
        )
        adapter = DirectMLAdapter(1, gpu.name, "10de", "2d59", 8 << 30)
        profile = hardware_profile(gpu, adapters=(adapter,))
        cpu_plan = recommendation(
            profile.accelerators[0], backend="openvino", device="CPU"
        )
        gpu_plan = recommendation(
            gpu, backend="onnxruntime", device="DIRECTML:1"
        )
        window = self.window(LauncherSettings(aim=True))
        window._hardware_selection_configured = False
        window.settings.hardware_selection_configured = False

        with mock.patch("launcher.qt_app.start_detector") as start:
            window._show_hardware_scan(
                profile,
                (gpu_plan, cpu_plan),
                automatic=True,
            )

        self.assertIs(_unique_ready_gpu_plan(profile, (gpu_plan, cpu_plan)), gpu_plan)
        self.assertTrue(_has_exact_directml_binding(profile, gpu_plan))
        self.assertEqual(window.backend.currentText(), "onnxruntime")
        self.assertEqual(window.device.currentText(), "DIRECTML:1")
        self.assertTrue(window._hardware_selection_configured)
        self.assertTrue(window.aim.isChecked())
        self.assertIsNone(window.process)
        start.assert_not_called()

    def test_unbound_or_multiple_ready_gpus_keep_unconfigured_cpu_fallback(self) -> None:
        first = Accelerator(
            AcceleratorKind.GPU, Vendor.AMD, "Radeon RX 6950 XT", "1002:73a5", True
        )
        second = Accelerator(
            AcceleratorKind.GPU, Vendor.NVIDIA, "RTX 5060", "10de:2d59", True
        )
        adapters = (
            DirectMLAdapter(0, first.name, "1002", "73a5", 16 << 30),
            DirectMLAdapter(1, second.name, "10de", "2d59", 8 << 30),
        )
        profile = hardware_profile(first, second, adapters=adapters)
        cpu_plan = recommendation(
            profile.accelerators[0], backend="openvino", device="CPU"
        )
        cases = (
            (recommendation(first, backend="onnxruntime", device="DIRECTML"),),
            (
                recommendation(first, backend="onnxruntime", device="DIRECTML:0"),
                recommendation(second, backend="onnxruntime", device="DIRECTML:1"),
            ),
        )
        for gpu_plans in cases:
            with self.subTest(devices=[plan.device for plan in gpu_plans]):
                window = self.window()
                window._hardware_selection_configured = False
                window.settings.hardware_selection_configured = False
                window._show_hardware_scan(
                    profile,
                    (*gpu_plans, cpu_plan),
                    automatic=True,
                )

                self.assertIsNone(
                    _unique_ready_gpu_plan(profile, (*gpu_plans, cpu_plan))
                )
                self.assertEqual(window.backend.currentText(), "openvino")
                self.assertEqual(window.device.currentText(), "CPU")
                self.assertFalse(window._hardware_selection_configured)
                self.assertIn("explicitly", window._hardware_start_notice)

    def test_provider_presence_without_one_scanned_physical_gpu_is_not_auto_selected(self) -> None:
        reported_only = Accelerator(
            AcceleratorKind.GPU,
            Vendor.NVIDIA,
            "Provider-only GPU claim",
            "10de:ffff",
            True,
        )
        profile = hardware_profile()
        cuda = recommendation(
            reported_only,
            backend="onnxruntime",
            device="CUDA",
        )

        self.assertIsNone(_unique_ready_gpu_plan(profile, (cuda,)))

    def test_hybrid_intel_igpu_does_not_make_exact_rtx_directml_ambiguous(self) -> None:
        intel = Accelerator(
            AcceleratorKind.GPU,
            Vendor.INTEL,
            "Intel Graphics",
            "8086:1234",
            False,
        )
        rtx = Accelerator(
            AcceleratorKind.GPU,
            Vendor.NVIDIA,
            "RTX 5060 Laptop GPU",
            "10de:2d59",
            # Exercise the WMI AdapterRAM-truncation path: exact DXGI dedicated
            # memory still proves that this is the discrete accelerator.
            None,
        )
        adapters = (
            DirectMLAdapter(0, intel.name, "8086", "1234", 0),
            DirectMLAdapter(1, rtx.name, "10de", "2d59", 8 << 30),
        )
        profile = hardware_profile(intel, rtx, adapters=adapters)
        intel_plan = recommendation(
            intel, backend="onnxruntime", device="DIRECTML:0"
        )
        rtx_plan = recommendation(
            rtx, backend="onnxruntime", device="DIRECTML:1"
        )

        self.assertIs(
            _unique_ready_gpu_plan(profile, (intel_plan, rtx_plan)),
            rtx_plan,
        )

    def test_close_cancels_and_joins_first_scan_worker(self) -> None:
        window = self.window()
        worker = threading.Thread(
            target=window._first_hardware_scan_cancel.wait,
            daemon=True,
        )
        window._first_hardware_scan_thread = worker
        worker.start()
        event = mock.Mock()

        window.closeEvent(event)

        self.assertTrue(window._first_hardware_scan_cancel.is_set())
        self.assertFalse(worker.is_alive())
        event.accept.assert_called_once_with()

    def test_late_first_scan_result_does_not_touch_closing_window(self) -> None:
        gpu = Accelerator(
            AcceleratorKind.GPU,
            Vendor.NVIDIA,
            "RTX 5060",
            "10de:2d59",
            True,
        )
        profile = hardware_profile(gpu)
        plan = recommendation(gpu, backend="onnxruntime", device="CUDA")
        window = self.window()
        original_report = window.hardware_report.toPlainText()
        window._closing = True

        window._finish_first_hardware_scan(profile, (plan,))

        self.assertEqual(window.hardware_report.toPlainText(), original_report)
        self.assertEqual(window.backend.currentText(), "openvino")
        self.assertEqual(window.device.currentText(), "CPU")

    def test_inconclusive_first_run_requires_cpu_confirmation_before_start(self) -> None:
        window = self.window()
        window._hardware_selection_configured = False
        window.settings.hardware_selection_configured = False
        window._hardware_start_notice = "GPU adapter binding is ambiguous."

        with mock.patch(
            "launcher.qt_app.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ) as question:
            accepted = window._confirm_hardware_before_start()

        self.assertFalse(accepted)
        self.assertFalse(window._hardware_selection_configured)
        self.assertEqual(window.stack.currentIndex(), 2)
        self.assertIn("ambiguous", question.call_args.args[2])

    def test_accuracy_workload_does_not_overwrite_saved_capture_rate(self) -> None:
        window = self.window(
            LauncherSettings(
                model_tier="high",
                screen_fps="60",
                capture_fps="60",
            )
        )

        self.assertEqual(window.screen_fps.text(), "60")
        self.assertEqual(window.capture_fps.text(), "60")
        self.assertEqual(window.model_preset.currentData(), "fort_player_balanced")

    def test_capture_orientation_and_format_load_and_collect(self) -> None:
        window = self.window(
            LauncherSettings(
                source_mode="camera",
                capture_format="NV12",
                capture_rotate_180=True,
            )
        )

        self.assertEqual(window.capture_format.currentText(), "NV12")
        self.assertTrue(window.capture_rotate_180.isChecked())
        collected = window.collect()
        self.assertEqual(collected.capture_format, "NV12")
        self.assertTrue(collected.capture_rotate_180)

    def test_automatic_makcu_controls_show_only_effective_rate_envelope(self) -> None:
        window = self.window(
            LauncherSettings(
                aim=True,
                aim_makcu_strength="0.91",
                aim_makcu_smoothing_alpha="0.67",
                aim_makcu_prediction_lead_seconds="0.047",
                aim_makcu_derivative_damping_seconds="0.013",
                aim_makcu_vertical_rate_ratio="0.63",
            )
        )

        self.assertFalse(window.aim_makcu_prediction_lead_seconds.isHidden())
        self.assertFalse(window.aim_makcu_derivative_damping_seconds.isHidden())
        self.assertFalse(window.aim_makcu_vertical_rate_ratio.isHidden())
        self.assertTrue(window.aim_makcu_max_step.isEnabled())
        self.assertFalse(window.aim_makcu_strength.isEnabled())
        self.assertFalse(window.aim_makcu_smoothing.isEnabled())
        self.assertFalse(window.aim_makcu_prediction_lead_seconds.isEnabled())
        self.assertFalse(window.aim_makcu_derivative_damping_seconds.isEnabled())
        self.assertFalse(window.aim_makcu_vertical_rate_ratio.isEnabled())
        self.assertIn(
            "Automatic plant-aware control",
            window.aim_makcu_control_mode_note.text(),
        )
        self.assertIn("Max step", window.aim_makcu_control_mode_note.text())
        self.assertIn("equal X/Y", window.aim_makcu_control_mode_note.text())
        self.assertEqual(window.collect().aim_makcu_strength, "0.91")
        self.assertEqual(window.collect().aim_makcu_smoothing_alpha, "0.67")
        self.assertAlmostEqual(window.aim_makcu_prediction_lead_seconds.value(), 0.047)
        self.assertAlmostEqual(window.aim_makcu_derivative_damping_seconds.value(), 0.013)
        self.assertAlmostEqual(window.aim_makcu_vertical_rate_ratio.value(), 0.63)
        self.assertEqual(window.aim_makcu_prediction_lead_seconds.minimum(), 0.0)
        self.assertEqual(window.aim_makcu_prediction_lead_seconds.maximum(), 0.25)
        self.assertEqual(window.aim_makcu_derivative_damping_seconds.minimum(), 0.0)
        self.assertEqual(window.aim_makcu_derivative_damping_seconds.maximum(), 0.25)
        self.assertEqual(window.aim_makcu_vertical_rate_ratio.minimum(), 0.10)
        self.assertEqual(window.aim_makcu_vertical_rate_ratio.maximum(), 1.00)
        self.assertEqual(window.aim_makcu_vertical_rate_ratio.singleStep(), 0.05)
        self.assertEqual(window.aim_makcu_vertical_rate_ratio.suffix(), " ×")

        window.aim_makcu_prediction_lead_seconds.setValue(0.061)
        window.aim_makcu_derivative_damping_seconds.setValue(0.015)
        window.aim_makcu_vertical_rate_ratio.setValue(0.70)
        collected = window.collect()
        self.assertEqual(collected.aim_makcu_strength, "0.91")
        self.assertEqual(collected.aim_makcu_smoothing_alpha, "0.67")
        self.assertEqual(collected.aim_makcu_prediction_lead_seconds, "0.061")
        self.assertEqual(collected.aim_makcu_derivative_damping_seconds, "0.015")
        self.assertEqual(collected.aim_makcu_vertical_rate_ratio, "0.7")

        restored = self.window(collected)
        self.assertEqual(restored.collect().aim_makcu_strength, "0.91")
        self.assertEqual(restored.collect().aim_makcu_smoothing_alpha, "0.67")
        self.assertAlmostEqual(restored.aim_makcu_prediction_lead_seconds.value(), 0.061)
        self.assertAlmostEqual(restored.aim_makcu_derivative_damping_seconds.value(), 0.015)
        self.assertAlmostEqual(restored.aim_makcu_vertical_rate_ratio.value(), 0.70)
        self.assertFalse(restored.aim_makcu_prediction_lead_seconds.isEnabled())
        self.assertFalse(restored.aim_makcu_derivative_damping_seconds.isEnabled())
        self.assertFalse(restored.aim_makcu_vertical_rate_ratio.isEnabled())

        restored.aim.setChecked(False)
        self.assertFalse(restored.aim_makcu_vertical_rate_ratio.isEnabled())

    def test_active_profile_exposes_only_its_effective_envelope_controls(self) -> None:
        window = self.window(
            LauncherSettings(
                aim=True,
                aim_makcu_active_profile="/private/profile.json",
                aim_makcu_vertical_rate_ratio="0.63",
            )
        )

        self.assertTrue(window.aim_makcu_max_step.isEnabled())
        self.assertTrue(window.aim_makcu_vertical_rate_ratio.isEnabled())
        self.assertFalse(window.aim_makcu_strength.isEnabled())
        self.assertFalse(window.aim_makcu_smoothing.isEnabled())
        self.assertFalse(window.aim_makcu_prediction_lead_seconds.isEnabled())
        self.assertFalse(window.aim_makcu_derivative_damping_seconds.isEnabled())
        self.assertIn(
            "Measured plant-aware profile",
            window.aim_makcu_control_mode_note.text(),
        )
        self.assertIn(
            "Max step and Vertical cap",
            window.aim_makcu_control_mode_note.text(),
        )

    def test_nonfinite_persisted_motion_controls_use_safe_defaults(self) -> None:
        window = self.window(
            LauncherSettings(
                aim=True,
                aim_makcu_prediction_lead_seconds="inf",
                aim_makcu_derivative_damping_seconds="-inf",
                aim_makcu_vertical_rate_ratio="nan",
            )
        )

        self.assertAlmostEqual(window.aim_makcu_prediction_lead_seconds.value(), 0.03)
        self.assertAlmostEqual(window.aim_makcu_derivative_damping_seconds.value(), 0.008)
        self.assertAlmostEqual(window.aim_makcu_vertical_rate_ratio.value(), 0.48)

    def test_int8_cpu_hardware_selects_int8_but_onnx_hides_it(self) -> None:
        window = self.window()
        window.detected_accelerator.clear()
        window.detected_accelerator.addItem(
            "CPU",
            {
                "ready": True,
                "backend": "openvino",
                "device": "CPU",
                "inference_size": 416,
                "precision": "int8",
                "tier": "low",
            },
        )

        window._apply_detected_accelerator()

        self.assertEqual(
            window.model_preset.currentData(),
            MODEL_PRESET_FORT_PLAYER_BALANCED_INT8,
        )
        window.backend.setCurrentText("onnxruntime")
        visible = {
            window.model_preset.itemData(index)
            for index in range(window.model_preset.count())
        }
        self.assertNotIn(MODEL_PRESET_FORT_PLAYER_BALANCED_INT8, visible)
        self.assertEqual(window.model_preset.currentData(), "fort_player")

    def test_int8_hardware_does_not_change_coco_class_semantics(self) -> None:
        window = self.window()
        window.model_preset.setCurrentIndex(
            window.model_preset.findData("coco_balanced")
        )
        window.detected_accelerator.clear()
        window.detected_accelerator.addItem(
            "CPU",
            {
                "ready": True,
                "backend": "openvino",
                "device": "CPU",
                "precision": "int8",
                "tier": "low",
            },
        )

        window._apply_detected_accelerator()

        self.assertEqual(window.model_preset.currentData(), "coco")

    def test_stop_escalates_from_interrupt_to_terminate_and_kill(self) -> None:
        window = self.window()
        process = FakeProcess()
        window.process = process  # type: ignore[assignment]
        callbacks: list[object] = []
        with (
            mock.patch("launcher.qt_app.request_stop") as request,
            mock.patch("launcher.qt_app.force_stop") as force,
            mock.patch("launcher.qt_app.kill_process") as kill,
            mock.patch(
                "launcher.qt_app.QTimer.singleShot",
                side_effect=lambda _delay, callback: callbacks.append(callback),
            ),
        ):
            window._stop()
            self.assertEqual(len(callbacks), 2)
            callbacks[0]()  # type: ignore[operator]
            callbacks[1]()  # type: ignore[operator]

        request.assert_called_once_with(process)
        force.assert_called_once_with(process)
        kill.assert_called_once_with(process)
        self.assertTrue(window._stop_requested)
        window.process = None

    @unittest.skipIf(os.name == "nt", "Controller precision is Linux-only")
    def test_precision_verification_binds_device_and_calibration(self) -> None:
        selected = controller()
        window = self.window(
            LauncherSettings(precision_device_path=str(selected.path)),
            controllers=(selected,),
        )
        process = FakeProcess(returncode=0)
        window.precision_process = process  # type: ignore[assignment]
        window._precision_mode = "verify"
        window._precision_pending_identity = candidate_identity(selected)
        window._precision_recent_output = [
            "Mapping verified: trigger=ABS_BRAKE (rest 255, pressed 0, decreasing polarity); "
            "right stick=ABS_Z/ABS_RZ.\n"
        ]

        with (
            mock.patch("launcher.qt_app.save_settings"),
            mock.patch("launcher.qt_app.QMessageBox.information"),
        ):
            window._finish_precision_process(process, 0)  # type: ignore[arg-type]

        self.assertTrue(window._precision_mapping_verified)
        self.assertEqual(window._precision_verified_identity, candidate_identity(selected))
        self.assertEqual(window._precision_trigger_rest, "255")
        self.assertEqual(window._precision_trigger_pressed, "0")
        collected = window.collect()
        self.assertTrue(collected.precision_mapping_verified)
        self.assertEqual(collected.precision_device_identity, candidate_identity(selected))


if __name__ == "__main__":
    unittest.main()
