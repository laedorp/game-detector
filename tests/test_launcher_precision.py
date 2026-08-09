from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

# These tests exercise launcher state only; the CPU test interpreter may not
# have Tk's shared library installed.
try:
    import tkinter  # noqa: F401
except ImportError:
    tkinter_stub = types.ModuleType("tkinter")
    tkinter_stub.filedialog = types.SimpleNamespace()
    tkinter_stub.messagebox = types.SimpleNamespace()
    tkinter_stub.ttk = types.SimpleNamespace()
    sys.modules["tkinter"] = tkinter_stub

import app
from controller_precision.linux_evdev import ControllerCandidate
from controller_precision.protocol import CONTROLLER_READY_SENTINEL
from launcher.application import DetectorLauncher
import launcher.precision as precision_module
from launcher.precision import (
    DEFAULT_PRECISION_PRESET,
    candidate_identity,
    precision_command,
    precision_preset,
    precision_readiness,
    pxn_controllers,
    select_saved_candidate,
    verification_calibration,
)
from launcher.settings import LauncherSettings, load_settings, save_settings


def controller(
    path: str = "/dev/input/by-id/usb-PXN-event-joystick",
    *,
    serial: str = "081410",
    vendor: int = 0x36E6,
    product: int = 0x3016,
    readable: bool = True,
) -> ControllerCandidate:
    return ControllerCandidate(
        path=Path(path),
        event_path=Path("/dev/input/event17"),
        name="PXN P5 8K",
        vendor=vendor,
        product=product,
        serial=serial,
        phys="usb-test/input0",
        readable=readable,
    )


class PrecisionLauncherHelperTests(unittest.TestCase):
    def test_discovery_keeps_only_physical_pxn_and_sorts_paths(self) -> None:
        other = controller("/dev/input/other", vendor=0x1234)
        virtual = ControllerCandidate(
            path=Path("/dev/input/virtual"),
            event_path=Path("/dev/input/event20"),
            name="PXN P5 8K",
            vendor=0x36E6,
            product=0x3016,
            serial="",
            phys="game-detector-precision/uinput",
            readable=True,
        )
        second = controller("/dev/input/z-controller", serial="two")
        first = controller("/dev/input/a-controller", serial="one")
        found = pxn_controllers(lambda: (other, second, virtual, first))
        self.assertEqual([item.path for item in found], [first.path, second.path])

    def test_verification_identity_is_stable_for_serialized_device(self) -> None:
        first = controller("/dev/input/event17")
        moved = controller("/dev/input/event22")
        self.assertEqual(candidate_identity(first), candidate_identity(moved))
        different = controller("/dev/input/event17", serial="different")
        self.assertNotEqual(candidate_identity(first), candidate_identity(different))

    def test_unserialized_identity_is_bound_to_locator(self) -> None:
        first = controller("/dev/input/by-id/first", serial="")
        second = controller("/dev/input/by-id/second", serial="")
        self.assertNotEqual(candidate_identity(first), candidate_identity(second))

        replaced_port = ControllerCandidate(
            path=first.path,
            event_path=first.event_path,
            name=first.name,
            vendor=first.vendor,
            product=first.product,
            serial="",
            phys="usb-replacement/input0",
            readable=True,
        )
        self.assertNotEqual(candidate_identity(first), candidate_identity(replaced_port))

    def test_saved_selection_does_not_guess_when_multiple_devices_exist(self) -> None:
        first = controller("/dev/input/first", serial="one")
        second = controller("/dev/input/second", serial="two")
        self.assertIs(select_saved_candidate((first, second), str(second.path)), second)
        self.assertIsNone(select_saved_candidate((first, second), ""))
        self.assertIs(select_saved_candidate((first,), ""), first)

    def test_run_command_is_internal_confirmed_and_parent_bound(self) -> None:
        command = precision_command(
            controller(),
            mode="run",
            preset_key="strong",
            parent_pid=4242,
            executable="/runtime/python",
            app_script="/checkout/app.py",
            frozen=False,
            platform="linux",
        )
        self.assertEqual(
            command[:5],
            [
                "/runtime/python",
                "/checkout/app.py",
                "--controller-precision",
                "--run",
                "--confirm-default-mapping",
            ],
        )
        self.assertEqual(command[command.index("--parent-pid") + 1], "4242")
        self.assertEqual(command[command.index("--strength") + 1], "0.22")
        self.assertEqual(command[command.index("--device") + 1], str(controller().path))
        self.assertEqual(command[command.index("--vendor") + 1], "0x36e6")
        self.assertEqual(command[command.index("--product") + 1], "0x3016")
        self.assertEqual(command[command.index("--serial") + 1], "081410")
        self.assertEqual(
            command[command.index("--expected-fingerprint") + 1],
            candidate_identity(controller()),
        )

    def test_unserialized_run_command_propagates_path_and_phys_fingerprint(self) -> None:
        selected = controller(serial="")
        command = precision_command(
            selected,
            mode="run",
            parent_pid=4242,
            executable="GameDetector",
            frozen=True,
            platform="linux",
        )
        self.assertNotIn("--serial", command)
        self.assertEqual(
            command[command.index("--expected-fingerprint") + 1],
            candidate_identity(selected),
        )

    def test_run_command_carries_verified_trigger_calibration(self) -> None:
        command = precision_command(
            controller(),
            mode="run",
            parent_pid=4242,
            trigger_rest=255,
            trigger_pressed=0,
            executable="GameDetector",
            frozen=True,
            platform="linux",
        )
        self.assertEqual(command[command.index("--trigger-rest") + 1], "255")
        self.assertEqual(command[command.index("--trigger-pressed") + 1], "0")

    def test_verification_output_calibration_is_parsed_only_once(self) -> None:
        report = (
            "Mapping verified: trigger=ABS_BRAKE (rest 255, pressed 0, decreasing polarity); "
            "right stick=ABS_Z/ABS_RZ.\n"
        )
        self.assertEqual(verification_calibration(report), (255, 0))
        self.assertIsNone(verification_calibration("Mapping was not verified"))
        self.assertIsNone(verification_calibration(report + report))

    def test_verify_command_is_read_only_mode_without_run_confirmation(self) -> None:
        command = precision_command(
            controller(),
            mode="verify",
            verification_seconds=5,
            executable="GameDetector",
            frozen=True,
            platform="linux",
        )
        self.assertEqual(command[:3], ["GameDetector", "--controller-precision", "--verify-mapping"])
        self.assertNotIn("--run", command)
        self.assertNotIn("--confirm-default-mapping", command)
        self.assertEqual(command[command.index("--verification-seconds") + 1], "5")
        self.assertEqual(command[command.index("--vendor") + 1], "0x36e6")
        self.assertEqual(command[command.index("--product") + 1], "0x3016")
        self.assertEqual(command[command.index("--serial") + 1], "081410")

    def test_non_linux_command_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only on Linux"):
            precision_command(controller(), mode="verify", platform="win32")

    def test_unknown_preset_falls_back_to_balanced(self) -> None:
        self.assertEqual(precision_preset("unknown").key, DEFAULT_PRECISION_PRESET)

    def test_missing_uinput_has_actionable_non_privileged_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "uinput"
            readiness = precision_readiness(
                controller(),
                platform="linux",
                uinput_path=missing,
                evdev_check=lambda: object(),
            )
        self.assertFalse(readiness.ready)
        self.assertIn("missing", readiness.summary)
        self.assertIn("sudo modprobe uinput", readiness.action)
        self.assertIn("never runs privileged", readiness.action)

    def test_unreadable_controller_is_rejected_before_uinput(self) -> None:
        readiness = precision_readiness(
            controller(readable=False),
            platform="linux",
            evdev_check=lambda: object(),
        )
        self.assertFalse(readiness.ready)
        self.assertIn("cannot read", readiness.summary)

    def test_launcher_helper_has_no_detector_capture_or_network_import(self) -> None:
        tree = ast.parse(
            Path(precision_module.__file__).read_text(encoding="utf-8"),
            filename=str(precision_module.__file__),
        )
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertFalse(imported & {"capture", "cv2", "detection", "socket", "udp"})


class PrecisionSettingsTests(unittest.TestCase):
    def test_verified_identity_and_preset_round_trip_without_running_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "settings.json"
            original = LauncherSettings(
                precision_device_path="/dev/input/by-id/pxn",
                precision_device_identity="abc123",
                precision_mapping_verified=True,
                precision_preset="strong",
                precision_trigger_rest="255",
                precision_trigger_pressed="0",
            )
            save_settings(original, target)
            payload = json.loads(target.read_text(encoding="utf-8"))
            loaded = load_settings(target)
        self.assertNotIn("precision_running", payload)
        self.assertTrue(loaded.precision_mapping_verified)
        self.assertEqual(loaded.precision_device_identity, "abc123")
        self.assertEqual(loaded.precision_preset, "strong")
        self.assertEqual(loaded.precision_trigger_rest, "255")
        self.assertEqual(loaded.precision_trigger_pressed, "0")

    def test_bare_verified_boolean_is_never_trusted(self) -> None:
        loaded = LauncherSettings.from_mapping(
            {"precision_mapping_verified": True, "precision_device_identity": ""}
        )
        self.assertFalse(loaded.precision_mapping_verified)


class PrecisionEntryPointTests(unittest.TestCase):
    def test_internal_dispatch_does_not_enter_detector(self) -> None:
        with mock.patch("controller_precision.cli.main", return_value=17) as worker:
            result = app.main(["--controller-precision", "--diagnose"])
        self.assertEqual(result, 17)
        worker.assert_called_once_with(["--diagnose"])


class _FakeVariable:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class _FakeWidget:
    def __init__(self) -> None:
        self.state = ""

    def configure(self, *, state: str) -> None:
        self.state = state


class PrecisionReadyHandshakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.launcher = DetectorLauncher.__new__(DetectorLauncher)
        self.process = mock.Mock()
        self.process.poll.return_value = None
        self.launcher.precision_process = self.process
        self.launcher._precision_mode = "run"
        self.launcher._precision_stop_requested = False
        self.launcher._precision_ready = False
        self.launcher._precision_guidance_shown = False
        self.launcher._precision_recent_output = []
        self.launcher._closing = False
        self.launcher.precision_status = _FakeVariable("Starting…")
        self.launcher.root = object()
        self.launcher._append_log = mock.Mock()
        self.launcher._update_precision_controls = mock.Mock()
        self.launcher._precision_moonlight_guidance = mock.Mock()

    def test_only_exact_internal_sentinel_marks_worker_active(self) -> None:
        self.launcher._handle_precision_output_line(
            f"prefix {CONTROLLER_READY_SENTINEL} suffix\n"
        )
        self.assertFalse(self.launcher._precision_ready)
        self.assertEqual(self.launcher.precision_status.get(), "Starting…")
        self.launcher._append_log.reset_mock()

        self.launcher._handle_precision_output_line(CONTROLLER_READY_SENTINEL + "\n")

        self.assertTrue(self.launcher._precision_ready)
        self.assertEqual(
            self.launcher.precision_status.get(),
            "Active — hold LT for fine manual control",
        )
        self.launcher._precision_moonlight_guidance.assert_called_once_with(self.process)
        logged = "".join(
            str(call.args[0]) for call in self.launcher._append_log.call_args_list
        )
        self.assertNotIn(CONTROLLER_READY_SENTINEL, logged)

    def test_moonlight_guidance_is_blocked_until_ready(self) -> None:
        # Exercise the real method instead of the setUp spy.
        del self.launcher._precision_moonlight_guidance
        with mock.patch("launcher.application.messagebox.showinfo", create=True) as showinfo:
            self.launcher._precision_moonlight_guidance(self.process)
            showinfo.assert_not_called()
            self.assertEqual(self.launcher.precision_status.get(), "Starting…")

            self.launcher._precision_ready = True
            self.launcher._precision_moonlight_guidance(self.process)
            showinfo.assert_called_once()
            self.assertEqual(
                self.launcher.precision_status.get(),
                "Active — hold LT for fine manual control",
            )

    def test_precision_tab_moonlight_button_enables_only_after_handshake(self) -> None:
        selected = controller()
        self.launcher._precision_supported = True
        self.launcher._precision_candidates = {"selected": selected}
        self.launcher._selected_precision_candidate = lambda: selected
        self.launcher.precision_mapping_verified = mock.Mock()
        self.launcher.precision_mapping_verified.get.return_value = True
        self.launcher._precision_verified_identity = candidate_identity(selected)
        for name in (
            "precision_device_combo",
            "precision_refresh_button",
            "precision_verify_button",
            "precision_preset_combo",
            "precision_start_button",
            "precision_stop_button",
            "precision_moonlight_button",
        ):
            setattr(self.launcher, name, _FakeWidget())

        # Restore the real state updater hidden by setUp's spy.
        del self.launcher._update_precision_controls
        self.launcher._update_precision_controls()
        self.assertEqual(self.launcher.precision_moonlight_button.state, "disabled")

        self.launcher._precision_ready = True
        self.launcher._update_precision_controls()
        self.assertEqual(self.launcher.precision_moonlight_button.state, "normal")
if __name__ == "__main__":
    unittest.main()
