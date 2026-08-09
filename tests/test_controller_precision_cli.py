from __future__ import annotations

import ast
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import unittest
from unittest import mock

import controller_precision
from controller_precision.cli import build_parser, main
from controller_precision.codes import (
    ABS_BRAKE,
    ABS_GAS,
    ABS_RZ,
    ABS_X,
    ABS_Y,
    ABS_Z,
)
from controller_precision.linux_evdev import (
    AxisObservation,
    ControllerBackendError,
    ControllerCandidate,
    EvdevUnavailableError,
    MappingVerificationReport,
    PXN_P5_8K_PRODUCT_ID,
    PXN_VENDOR_ID,
    require_evdev,
)
from controller_precision.protocol import CONTROLLER_READY_SENTINEL


def physical_candidate() -> ControllerCandidate:
    return ControllerCandidate(
        path=Path("/dev/input/by-id/usb-PXN_P5_8K_081410-event-joystick"),
        event_path=Path("/dev/input/event17"),
        name="PXN P5 8K",
        vendor=PXN_VENDOR_ID,
        product=PXN_P5_8K_PRODUCT_ID,
        serial="081410",
        phys="usb-test/input0",
        readable=True,
    )


def axis_observation(
    code: int,
    start: int,
    minimum: int,
    maximum: int,
    declared_minimum: int = 0,
    declared_maximum: int = 255,
) -> AxisObservation:
    return AxisObservation(
        code,
        start,
        minimum,
        maximum,
        declared_minimum,
        declared_maximum,
    )


class ControllerPrecisionCliTests(unittest.TestCase):
    def run_cli(self, arguments: list[str], **changes: object) -> tuple[int, str, str]:
        output = StringIO()
        error = StringIO()
        options: dict[str, object] = {
            "output": output,
            "error": error,
            "discovery": lambda: (physical_candidate(),),
        }
        options.update(changes)
        result = main(arguments, **options)  # type: ignore[arg-type]
        return result, output.getvalue(), error.getvalue()

    def test_dry_run_never_opens_or_grabs_controller(self) -> None:
        calls: list[str] = []

        def forbidden_factory(*_args: object, **_kwargs: object) -> object:
            calls.append("opened")
            raise AssertionError("dry run must not create a controller backend")

        result, output, error = self.run_cli(
            ["--dry-run"],
            controller_factory=forbidden_factory,
        )
        self.assertEqual(result, 0)
        self.assertEqual(error, "")
        self.assertIn("Dry run only", output)
        self.assertIn("LT=ABS_BRAKE", output)
        self.assertEqual(calls, [])

    def test_run_refuses_unconfirmed_default_mapping(self) -> None:
        calls: list[str] = []
        result, _output, error = self.run_cli(
            ["--run"],
            controller_factory=lambda *_args, **_kwargs: calls.append("opened"),
        )
        self.assertEqual(result, 2)
        self.assertIn("Refusing to grab", error)
        self.assertEqual(calls, [])

    def test_read_only_verification_accepts_expected_axes(self) -> None:
        phases: list[str] = []

        def observer(
            _path: Path,
            *,
            phase: str,
            seconds: float,
            expected_identity: object,
        ) -> MappingVerificationReport:
            phases.append(phase)
            self.assertEqual(seconds, 0.01)
            self.assertEqual(expected_identity, physical_candidate().identity)
            if phase == "LT":
                observations = (
                    axis_observation(ABS_BRAKE, 0, 0, 255),
                    axis_observation(ABS_Z, 128, 128, 128),
                    axis_observation(ABS_RZ, 128, 128, 128),
                )
            else:
                observations = (
                    axis_observation(ABS_BRAKE, 0, 0, 0),
                    axis_observation(ABS_Z, 128, 0, 255),
                    axis_observation(ABS_RZ, 128, 0, 255),
                )
            return MappingVerificationReport(phase, observations, ())

        result, output, error = self.run_cli(
            ["--verify-mapping", "--verification-seconds", "0.01"],
            observer=observer,
        )
        self.assertEqual(result, 0)
        self.assertEqual(error, "")
        self.assertEqual(phases, ["LT", "right stick"])
        self.assertIn("Mapping verified", output)
        self.assertIn("rest 0, pressed 255, increasing polarity", output)
        self.assertIn("--trigger-rest 0 --trigger-pressed 255", output)
        self.assertIn("--lt-axis ABS_BRAKE", output)
        self.assertIn("--vendor 0x36e6 --product 0x3016 --serial 081410", output)

    def test_verification_uses_custom_configured_axes_and_reversed_trigger(self) -> None:
        def observer(
            _path: Path,
            *,
            phase: str,
            seconds: float,
            expected_identity: object,
        ) -> MappingVerificationReport:
            del seconds
            self.assertEqual(expected_identity, physical_candidate().identity)
            if phase == "LT":
                observations = (
                    axis_observation(ABS_GAS, 255, 0, 255),
                    axis_observation(ABS_X, 128, 128, 128),
                    axis_observation(ABS_Y, 128, 128, 128),
                )
            else:
                observations = (
                    axis_observation(ABS_GAS, 255, 255, 255),
                    axis_observation(ABS_X, 128, 0, 255),
                    axis_observation(ABS_Y, 128, 0, 255),
                )
            return MappingVerificationReport(phase, observations, ())

        result, output, error = self.run_cli(
            [
                "--verify-mapping",
                "--verification-seconds",
                "0.01",
                "--lt-axis",
                "ABS_GAS",
                "--right-x",
                "ABS_X",
                "--right-y",
                "ABS_Y",
            ],
            observer=observer,
        )
        self.assertEqual(result, 0)
        self.assertEqual(error, "")
        self.assertIn("ABS_GAS (rest 255, pressed 0, decreasing polarity)", output)
        self.assertIn("right stick=ABS_X/ABS_Y", output)
        self.assertIn("--lt-axis ABS_GAS", output)
        self.assertIn("--trigger-rest 255 --trigger-pressed 0", output)

    def test_verification_rejects_wrong_axes(self) -> None:
        def observer(
            _path: Path,
            *,
            phase: str,
            seconds: float,
            expected_identity: object,
        ) -> MappingVerificationReport:
            del seconds, expected_identity
            observations = (axis_observation(ABS_Z, 128, 0, 255),)
            return MappingVerificationReport(phase, observations, ())

        result, output, _error = self.run_cli(
            ["--verify-mapping", "--verification-seconds", "0.01"],
            observer=observer,
        )
        self.assertEqual(result, 3)
        self.assertIn("not verified", output)

    def test_verification_rejects_significant_unexpected_axis_movement(self) -> None:
        def observer(
            _path: Path,
            *,
            phase: str,
            seconds: float,
            expected_identity: object,
        ) -> MappingVerificationReport:
            del seconds, expected_identity
            if phase == "LT":
                observations = (
                    axis_observation(ABS_BRAKE, 0, 0, 255),
                    axis_observation(ABS_Z, 128, 50, 220),
                    axis_observation(ABS_RZ, 128, 128, 128),
                )
            else:
                observations = (
                    axis_observation(ABS_BRAKE, 0, 0, 0),
                    axis_observation(ABS_Z, 128, 0, 255),
                    axis_observation(ABS_RZ, 128, 0, 255),
                )
            return MappingVerificationReport(phase, observations, ())

        result, output, _error = self.run_cli(
            ["--verify-mapping", "--verification-seconds", "0.01"],
            observer=observer,
        )
        self.assertEqual(result, 3)
        self.assertIn("unexpected ABS_Z movement during LT", output)

    def test_verification_thresholds_scale_with_declared_axis_range(self) -> None:
        def observer(
            _path: Path,
            *,
            phase: str,
            seconds: float,
            expected_identity: object,
        ) -> MappingVerificationReport:
            del seconds, expected_identity
            if phase == "LT":
                # A 100-count change would pass the old absolute threshold but
                # is less than 10% of this declared 0..1023 trigger range.
                observations = (
                    axis_observation(ABS_BRAKE, 0, 0, 100, 0, 1023),
                    axis_observation(ABS_Z, 512, 512, 512, 0, 1023),
                    axis_observation(ABS_RZ, 512, 512, 512, 0, 1023),
                )
            else:
                observations = (
                    axis_observation(ABS_BRAKE, 0, 0, 0, 0, 1023),
                    axis_observation(ABS_Z, 512, 0, 1023, 0, 1023),
                    axis_observation(ABS_RZ, 512, 0, 1023, 0, 1023),
                )
            return MappingVerificationReport(phase, observations, ())

        result, output, _error = self.run_cli(
            ["--verify-mapping", "--verification-seconds", "0.01"],
            observer=observer,
        )
        self.assertEqual(result, 3)
        self.assertIn("moved only 10%", output)

    def test_confirmed_run_constructs_isolated_worker_with_user_curve(self) -> None:
        created: list[tuple[object, dict[str, object]]] = []

        class FakeController:
            def __init__(self, path: object, **kwargs: object) -> None:
                created.append((path, kwargs))
                self.ran = False

            def run(self, *, should_stop: object, on_ready: object) -> None:
                self.ran = True
                self.should_stop = should_stop
                on_ready()  # type: ignore[operator]

        result, output, error = self.run_cli(
            [
                "--run",
                "--confirm-default-mapping",
                "--strength",
                "0.25",
                "--exponent",
                "1.6",
            ],
            controller_factory=FakeController,
        )
        self.assertEqual(result, 0)
        self.assertEqual(error, "")
        self.assertEqual(output.count(CONTROLLER_READY_SENTINEL), 1)
        self.assertIn("active", output)
        self.assertEqual(len(created), 1)
        path, kwargs = created[0]
        self.assertEqual(path, physical_candidate().path)
        self.assertTrue(kwargs["mapping_verified"])
        self.assertEqual(kwargs["expected_identity"], physical_candidate().identity)
        config = kwargs["config"]
        self.assertEqual(config.strength, 0.25)  # type: ignore[union-attr]
        self.assertEqual(config.exponent, 1.6)  # type: ignore[union-attr]
        self.assertIsNone(config.trigger_button_code)  # type: ignore[union-attr]

    def test_open_failure_emits_no_ready_sentinel_or_active_claim(self) -> None:
        class FailingController:
            def __init__(self, _path: object, **_kwargs: object) -> None:
                pass

            def run(self, *, should_stop: object, on_ready: object) -> None:
                del should_stop, on_ready
                raise ControllerBackendError("simulated uinput open failure")

        result, output, error = self.run_cli(
            ["--run", "--confirm-default-mapping"],
            controller_factory=FailingController,
        )

        self.assertEqual(result, 3)
        self.assertNotIn(CONTROLLER_READY_SENTINEL, output)
        self.assertNotIn("controller active", output.lower())
        self.assertIn("simulated uinput open failure", error)

    def test_launcher_fingerprint_rejects_unserialized_replacement_before_factory(self) -> None:
        original = ControllerCandidate(
            path=Path("/dev/input/by-id/usb-PXN-event-joystick"),
            event_path=Path("/dev/input/event17"),
            name="PXN P5 8K",
            vendor=PXN_VENDOR_ID,
            product=PXN_P5_8K_PRODUCT_ID,
            serial="",
            phys="usb-original/input0",
            readable=True,
        )
        replacement = ControllerCandidate(
            path=original.path,
            event_path=original.event_path,
            name=original.name,
            vendor=original.vendor,
            product=original.product,
            serial="",
            phys="usb-replacement/input0",
            readable=True,
        )
        factory_calls: list[str] = []

        result, output, error = self.run_cli(
            [
                "--run",
                "--confirm-default-mapping",
                "--device",
                str(original.path),
                "--expected-fingerprint",
                original.identity.fingerprint(),
            ],
            discovery=lambda: (replacement,),
            controller_factory=lambda *_args, **_kwargs: factory_calls.append("created"),
        )

        self.assertEqual(result, 3)
        self.assertEqual(output, "")
        self.assertIn("no longer matches", error)
        self.assertEqual(factory_calls, [])

    def test_parent_pid_change_requests_worker_stop(self) -> None:
        stop_values: list[bool] = []

        class FakeController:
            def __init__(self, _path: object, **_kwargs: object) -> None:
                pass

            def run(self, *, should_stop: object, on_ready: object) -> None:
                on_ready()  # type: ignore[operator]
                stop_values.append(should_stop())  # type: ignore[operator]

        with mock.patch("controller_precision.cli.os.getppid", return_value=4321):
            result, _output, error = self.run_cli(
                ["--run", "--confirm-default-mapping", "--parent-pid", "1234"],
                controller_factory=FakeController,
            )
        self.assertEqual(result, 0)
        self.assertEqual(error, "")
        self.assertEqual(stop_values, [True])

    def test_parent_pid_must_be_positive(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--run", "--parent-pid", "0"])

    def test_invalid_curve_is_rejected_before_hardware_selection(self) -> None:
        discoveries: list[str] = []
        result, _output, error = self.run_cli(
            ["--dry-run", "--strength", "2"],
            discovery=lambda: discoveries.append("called"),
        )
        self.assertEqual(result, 2)
        self.assertIn("strength", error)
        self.assertEqual(discoveries, [])


class ControllerPrecisionIsolationTests(unittest.TestCase):
    def test_package_has_no_detection_capture_network_or_image_dependencies(self) -> None:
        package_root = Path(controller_precision.__file__).resolve().parent
        prohibited = {
            "capture",
            "cv2",
            "detection",
            "main",
            "numpy",
            "socket",
            "utils",
        }
        imported: set[str] = set()
        for path in package_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".", 1)[0])
        self.assertEqual(imported & prohibited, set())

    def test_missing_optional_evdev_has_clear_error(self) -> None:
        with (
            mock.patch("controller_precision.linux_evdev._evdev", None),
            mock.patch(
                "controller_precision.linux_evdev.EVDEV_IMPORT_ERROR",
                ImportError("not installed"),
            ),
            self.assertRaisesRegex(EvdevUnavailableError, "optional 'evdev'"),
        ):
            require_evdev()


if __name__ == "__main__":
    unittest.main()
