from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest import mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from controller_precision.linux_evdev import ControllerCandidate
from launcher.precision import candidate_identity
from launcher.qt_app import LauncherWindow
from launcher.settings import (
    AIM_OUTPUT_MAKCU,
    LauncherSettings,
    MODEL_PRESET_CUSTOM,
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


class FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.pid = 4242
        self.stdout = None

    def poll(self) -> int | None:
        return self.returncode


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
