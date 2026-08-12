from __future__ import annotations

from dataclasses import dataclass
import errno
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from controller_precision.codes import (
    ABS_BRAKE,
    ABS_RZ,
    ABS_Z,
    EV_ABS,
    EV_KEY,
    EV_SYN,
    SYN_DROPPED,
    SYN_REPORT,
)
from controller_precision.core import DroppedEventsError, TriggerCalibration
from controller_precision.linux_evdev import (
    ControllerBackendError,
    ControllerCandidate,
    ControllerCapabilityError,
    ControllerIdentity,
    ControllerIdentityError,
    ControllerNotFoundError,
    EvdevPrecisionController,
    MappingNotVerifiedError,
    PXN_P5_8K_PRODUCT_ID,
    PXN_VENDOR_ID,
    _default_uinput_factory,
    discover_controllers,
    observe_phase,
    select_controller,
)


@dataclass(frozen=True, slots=True)
class FakeEvent:
    type: int
    code: int
    value: int
    marker: str = ""


@dataclass(slots=True)
class FakeAbsInfo:
    value: int
    min: int
    max: int
    fuzz: int = 0
    flat: int = 0
    resolution: int = 0


class FakeSource:
    def __init__(
        self,
        _path: str = "/dev/input/by-id/fake-event-joystick",
        *,
        axes: dict[int, FakeAbsInfo] | None = None,
        active_keys: tuple[int, ...] = (),
        phys: str = "usb-test/input0",
        name: str = "PXN P5 8K",
        vendor: int = PXN_VENDOR_ID,
        product: int = PXN_P5_8K_PRODUCT_ID,
        serial: str = "081410",
    ) -> None:
        self.name = name
        self.phys = phys
        self.uniq = serial
        self.info = SimpleNamespace(
            vendor=vendor,
            product=product,
            version=0x0111,
            bustype=0x0003,
        )
        self.axes = axes or {
            ABS_Z: FakeAbsInfo(128, 0, 255),
            ABS_RZ: FakeAbsInfo(128, 0, 255),
            ABS_BRAKE: FakeAbsInfo(0, 0, 255),
        }
        self.keys = tuple(active_keys)
        self.grabbed = False
        self.ungrabbed = False
        self.closed = False
        self.read_batches: list[list[FakeEvent]] = []

    def capabilities(self, *, absinfo: bool = True) -> dict[int, list[object]]:
        self.assert_absinfo = absinfo
        return {EV_ABS: list(self.axes.items())}

    def absinfo(self, code: int) -> FakeAbsInfo:
        return self.axes[code]

    def active_keys(self) -> list[int]:
        return list(self.keys)

    def grab(self) -> None:
        self.grabbed = True

    def ungrab(self) -> None:
        self.ungrabbed = True
        self.grabbed = False

    def close(self) -> None:
        self.closed = True

    def read(self) -> list[FakeEvent]:
        if not self.read_batches:
            raise BlockingIOError
        return self.read_batches.pop(0)


class FakeUInput:
    def __init__(self) -> None:
        self.writes: list[tuple[int, int, int]] = []
        self.source_events: list[FakeEvent] = []
        self.syn_count = 0
        self.closed = False

    def write(self, event_type: int, code: int, value: int) -> None:
        self.writes.append((event_type, code, value))

    def write_event(self, event: FakeEvent) -> None:
        self.source_events.append(event)

    def syn(self) -> None:
        self.syn_count += 1

    def close(self) -> None:
        self.closed = True


def candidate(path: str, *, serial: str = "081410") -> ControllerCandidate:
    device_path = Path(path)
    return ControllerCandidate(
        path=device_path,
        event_path=Path("/dev/input/event17"),
        name="PXN P5 8K",
        vendor=PXN_VENDOR_ID,
        product=PXN_P5_8K_PRODUCT_ID,
        serial=serial,
        phys="usb-test/input0",
        readable=True,
    )


class ControllerDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_root = self.root / "dev" / "input"
        self.by_id = self.input_root / "by-id"
        self.sys_input = self.root / "sys" / "class" / "input"
        self.by_id.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def add_event(
        self,
        event_name: str,
        *,
        name: str,
        vendor: str,
        product: str,
        serial: str,
        phys: str,
        abs_mask: str = "30627",
        key_mask: str = "1000000000000 0 0 0 0",
    ) -> Path:
        event_path = self.input_root / event_name
        event_path.write_bytes(b"")
        device = self.sys_input / event_name / "device"
        (device / "id").mkdir(parents=True)
        (device / "capabilities").mkdir()
        (device / "name").write_text(name, encoding="utf-8")
        (device / "id" / "vendor").write_text(vendor, encoding="utf-8")
        (device / "id" / "product").write_text(product, encoding="utf-8")
        (device / "uniq").write_text(serial, encoding="utf-8")
        (device / "phys").write_text(phys, encoding="utf-8")
        (device / "capabilities" / "abs").write_text(abs_mask, encoding="utf-8")
        (device / "capabilities" / "key").write_text(key_mask, encoding="utf-8")
        return event_path

    def test_persistent_by_id_path_is_preferred_and_metadata_is_read(self) -> None:
        self.add_event(
            "event17",
            name="PXN P5 8K",
            vendor="36e6",
            product="3016",
            serial="081410",
            phys="usb-test/input0",
        )
        stable = self.by_id / "usb-PXN_P5_8K_081410-event-joystick"
        stable.symlink_to("../event17")
        found = discover_controllers(
            by_id_root=self.by_id,
            input_root=self.input_root,
            sys_class_input=self.sys_input,
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].path, stable)
        self.assertEqual(found[0].usb_id, "36e6:3016")
        self.assertEqual(found[0].serial, "081410")

    def test_fallback_ignores_non_gamepad_event_devices(self) -> None:
        self.add_event(
            "event3",
            name="Keyboard",
            vendor="0001",
            product="0001",
            serial="",
            phys="isa/input0",
            abs_mask="0",
            key_mask="1",
        )
        self.add_event(
            "event17",
            name="PXN P5 8K",
            vendor="36e6",
            product="3016",
            serial="081410",
            phys="usb-test/input0",
        )
        found = discover_controllers(
            by_id_root=self.by_id,
            input_root=self.input_root,
            sys_class_input=self.sys_input,
        )
        self.assertEqual([item.name for item in found], ["PXN P5 8K"])
        self.assertEqual(found[0].path.name, "event17")

    def test_virtual_precision_device_is_never_selected_as_source(self) -> None:
        self.add_event(
            "event20",
            name="PXN P5 8K",
            vendor="36e6",
            product="3016",
            serial="",
            phys="game-detector-precision/uinput",
        )
        found = discover_controllers(
            by_id_root=self.by_id,
            input_root=self.input_root,
            sys_class_input=self.sys_input,
        )
        self.assertEqual(found, ())

    def test_selector_requires_disambiguation(self) -> None:
        devices = (
            candidate("/dev/input/by-id/first", serial="one"),
            candidate("/dev/input/by-id/second", serial="two"),
        )
        with self.assertRaisesRegex(ControllerNotFoundError, "more than one"):
            select_controller(devices)
        self.assertEqual(select_controller(devices, serial="two"), devices[1])
        self.assertEqual(
            select_controller(devices, device="/dev/input/by-id/first"),
            devices[0],
        )

    def test_explicit_path_still_requires_requested_hardware_profile(self) -> None:
        other = ControllerCandidate(
            path=Path("/dev/input/by-id/other"),
            event_path=Path("/dev/input/event19"),
            name="Other Controller",
            vendor=0x1234,
            product=0x5678,
            serial="other",
            phys="usb-other/input0",
            readable=True,
        )
        with self.assertRaises(ControllerNotFoundError):
            select_controller((other,), device=other.path)


class EvdevPrecisionControllerTests(unittest.TestCase):
    def make_controller(
        self,
        source: FakeSource | None = None,
        ui: FakeUInput | None = None,
        **changes: object,
    ) -> tuple[EvdevPrecisionController, FakeSource, FakeUInput]:
        physical = source or FakeSource()
        virtual = ui or FakeUInput()
        values: dict[str, object] = {
            "expected_identity": ControllerIdentity(
                "PXN P5 8K",
                PXN_VENDOR_ID,
                PXN_P5_8K_PRODUCT_ID,
                "081410",
            ),
            "mapping_verified": True,
            "input_device_factory": lambda _path: physical,
            "uinput_factory": lambda _source: virtual,
        }
        values.update(changes)
        controller = EvdevPrecisionController("/dev/input/by-id/fake", **values)
        return controller, physical, virtual

    def test_unverified_mapping_refuses_before_open_or_grab(self) -> None:
        calls: list[str] = []
        controller = EvdevPrecisionController(
            "/dev/input/by-id/fake",
            expected_identity=ControllerIdentity(
                "PXN P5 8K",
                PXN_VENDOR_ID,
                PXN_P5_8K_PRODUCT_ID,
                "081410",
            ),
            input_device_factory=lambda path: calls.append(path),
            mapping_verified=False,
        )
        with self.assertRaises(MappingNotVerifiedError):
            controller.open()
        self.assertEqual(calls, [])

    def test_physical_device_permission_error_is_actionable(self) -> None:
        controller = EvdevPrecisionController(
            "/dev/input/by-id/fake",
            expected_identity=ControllerIdentity(
                "PXN P5 8K",
                PXN_VENDOR_ID,
                PXN_P5_8K_PRODUCT_ID,
                "081410",
            ),
            input_device_factory=lambda _path: (_ for _ in ()).throw(
                PermissionError(errno.EACCES, "denied")
            ),
            mapping_verified=True,
        )
        with self.assertRaisesRegex(ControllerBackendError, "active desktop session"):
            controller.open()

    def test_event_node_reuse_is_rejected_before_grab_or_uinput(self) -> None:
        mismatches = (
            {"name": "Different Controller"},
            {"vendor": 0x1234},
            {"product": 0x5678},
            {"serial": "other"},
        )
        for changes in mismatches:
            with self.subTest(changes=changes):
                reused = FakeSource(**changes)  # type: ignore[arg-type]
                uinput_calls: list[str] = []
                controller, reused, _ui = self.make_controller(
                    source=reused,
                    uinput_factory=lambda _source: uinput_calls.append("created"),
                )
                with self.assertRaisesRegex(ControllerIdentityError, "identity changed"):
                    controller.open()
                self.assertFalse(reused.grabbed)
                self.assertTrue(reused.closed)
                self.assertEqual(uinput_calls, [])

    def test_unserialized_controller_changed_phys_is_rejected_before_grab_or_uinput(self) -> None:
        path = "/dev/input/by-id/usb-PXN-event-joystick"
        replacement = FakeSource(serial="", phys="usb-replacement/input0")
        uinput_calls: list[str] = []
        controller = EvdevPrecisionController(
            path,
            expected_identity=ControllerIdentity(
                "PXN P5 8K",
                PXN_VENDOR_ID,
                PXN_P5_8K_PRODUCT_ID,
                "",
                "usb-original/input0",
                path,
            ),
            mapping_verified=True,
            input_device_factory=lambda _path: replacement,
            uinput_factory=lambda _source: uinput_calls.append("created"),
        )

        with self.assertRaisesRegex(ControllerIdentityError, "identity changed"):
            controller.open()

        self.assertFalse(replacement.grabbed)
        self.assertTrue(replacement.closed)
        self.assertEqual(uinput_calls, [])

    def test_open_grabs_source_then_initializes_virtual_state(self) -> None:
        controller, source, ui = self.make_controller()
        controller.open()
        self.assertTrue(source.grabbed)
        self.assertTrue(controller.active)
        self.assertEqual(ui.syn_count, 1)
        self.assertIn((EV_ABS, ABS_Z, 128), ui.writes)
        controller.close()
        self.assertTrue(source.ungrabbed)
        self.assertTrue(source.closed)
        self.assertTrue(ui.closed)

    def test_close_restores_decreasing_polarity_trigger_to_calibrated_rest(self) -> None:
        source = FakeSource(
            axes={
                ABS_Z: FakeAbsInfo(128, 0, 255),
                ABS_RZ: FakeAbsInfo(128, 0, 255),
                ABS_BRAKE: FakeAbsInfo(255, 0, 255),
            }
        )
        controller, _source, ui = self.make_controller(
            source=source,
            trigger_calibration=TriggerCalibration(rest=255, pressed=0),
        )

        controller.open()
        controller.close()

        final_axes = {
            code: value
            for event_type, code, value in ui.writes
            if event_type == EV_ABS
        }
        self.assertEqual(final_axes[ABS_BRAKE], 255)
        self.assertEqual(final_axes[ABS_Z], 128)
        self.assertEqual(final_axes[ABS_RZ], 128)
        self.assertTrue(ui.closed)

    def test_run_announces_ready_only_after_grab_uinput_and_initial_sync(self) -> None:
        controller, source, ui = self.make_controller()
        observations: list[tuple[bool, bool, int]] = []

        controller.run(
            events=(),
            on_ready=lambda: observations.append(
                (source.grabbed, controller.active, ui.syn_count)
            ),
        )

        self.assertEqual(observations, [(True, True, 1)])
        self.assertTrue(source.ungrabbed)
        self.assertTrue(ui.closed)

    def test_open_failure_never_invokes_ready_callback(self) -> None:
        source = FakeSource()
        ready_calls: list[str] = []
        controller, source, _ui = self.make_controller(
            source=source,
            uinput_factory=lambda _source: (_ for _ in ()).throw(
                OSError("uinput unavailable")
            ),
        )

        with self.assertRaisesRegex(OSError, "uinput unavailable"):
            controller.run(events=(), on_ready=lambda: ready_calls.append("ready"))

        self.assertEqual(ready_calls, [])
        self.assertTrue(source.ungrabbed)
        self.assertTrue(source.closed)

    def test_ready_callback_failure_still_releases_both_devices(self) -> None:
        controller, source, ui = self.make_controller()

        with self.assertRaisesRegex(RuntimeError, "handshake pipe closed"):
            controller.run(
                events=(),
                on_ready=lambda: (_ for _ in ()).throw(
                    RuntimeError("handshake pipe closed")
                ),
            )

        self.assertTrue(source.ungrabbed)
        self.assertTrue(source.closed)
        self.assertTrue(ui.closed)

    def test_uinput_creation_failure_always_ungrabs_and_closes_source(self) -> None:
        source = FakeSource()

        def fail(_source: FakeSource) -> FakeUInput:
            raise OSError("uinput unavailable")

        controller = EvdevPrecisionController(
            "/dev/input/by-id/fake",
            expected_identity=ControllerIdentity(
                "PXN P5 8K",
                PXN_VENDOR_ID,
                PXN_P5_8K_PRODUCT_ID,
                "081410",
            ),
            mapping_verified=True,
            input_device_factory=lambda _path: source,
            uinput_factory=fail,
        )
        with self.assertRaisesRegex(OSError, "uinput unavailable"):
            controller.open()
        self.assertTrue(source.ungrabbed)
        self.assertTrue(source.closed)
        self.assertFalse(controller.active)

    def test_default_uinput_factory_explains_missing_device_node(self) -> None:
        source = FakeSource()

        class MissingUInput:
            @staticmethod
            def from_device(*_args: object, **_kwargs: object) -> object:
                raise OSError(errno.ENOENT, "missing")

        fake_evdev = SimpleNamespace(UInput=MissingUInput)
        with (
            mock.patch(
                "controller_precision.linux_evdev.require_evdev",
                return_value=fake_evdev,
            ),
            self.assertRaisesRegex(ControllerBackendError, "uinput module"),
        ):
            _default_uinput_factory(source)

    def test_default_uinput_factory_explains_permission_denial(self) -> None:
        source = FakeSource()

        class DeniedUInput:
            @staticmethod
            def from_device(*_args: object, **_kwargs: object) -> object:
                raise OSError(errno.EACCES, "denied")

        fake_evdev = SimpleNamespace(UInput=DeniedUInput)
        with (
            mock.patch(
                "controller_precision.linux_evdev.require_evdev",
                return_value=fake_evdev,
            ),
            self.assertRaisesRegex(ControllerBackendError, "uaccess"),
        ):
            _default_uinput_factory(source)

    def test_missing_required_axis_fails_before_grab(self) -> None:
        source = FakeSource(axes={ABS_Z: FakeAbsInfo(128, 0, 255)})
        controller, source, _ui = self.make_controller(source=source)
        with self.assertRaises(ControllerCapabilityError):
            controller.open()
        self.assertFalse(source.grabbed)
        self.assertTrue(source.closed)

    def test_trigger_calibration_must_fit_declared_axis_range(self) -> None:
        controller, source, _ui = self.make_controller(
            trigger_calibration=TriggerCalibration(rest=-1, pressed=255),
        )
        with self.assertRaisesRegex(ControllerCapabilityError, "declared range"):
            controller.open()
        self.assertFalse(source.grabbed)
        self.assertTrue(source.closed)

    def test_virtual_device_is_rejected_as_physical_input(self) -> None:
        source = FakeSource(phys="game-detector-precision/uinput")
        controller, source, _ui = self.make_controller(source=source)
        with self.assertRaisesRegex(ControllerCapabilityError, "virtual"):
            controller.open()
        self.assertFalse(source.grabbed)
        self.assertTrue(source.closed)

    def test_inactive_reports_are_forwarded_as_original_event_objects(self) -> None:
        controller, source, ui = self.make_controller()
        button = FakeEvent(EV_KEY, 304, 1, "button")
        right = FakeEvent(EV_ABS, ABS_Z, 190, "right")
        sync = FakeEvent(EV_SYN, SYN_REPORT, 0, "sync")
        controller.run(events=(button, right, sync))
        self.assertEqual(ui.source_events, [button, right, sync])
        self.assertTrue(source.ungrabbed)
        self.assertTrue(ui.closed)
        # Cleanup releases any button that had been forwarded as pressed.
        self.assertIn((EV_KEY, 304, 0), ui.writes)

    def test_lt_held_replaces_only_right_stick_axes(self) -> None:
        controller, _source, ui = self.make_controller()
        lt = FakeEvent(EV_ABS, ABS_BRAKE, 255, "lt")
        right = FakeEvent(EV_ABS, ABS_Z, 255, "right")
        sync = FakeEvent(EV_SYN, SYN_REPORT, 0, "sync")
        controller.run(events=(lt, right, sync))
        self.assertIn(lt, ui.source_events)
        self.assertIn(sync, ui.source_events)
        self.assertNotIn(right, ui.source_events)
        precise = [value for etype, code, value in ui.writes if etype == EV_ABS and code == ABS_Z]
        self.assertTrue(any(128 < value < 200 for value in precise))

    def test_syn_dropped_fails_closed(self) -> None:
        controller, source, ui = self.make_controller()
        dropped = FakeEvent(EV_SYN, SYN_DROPPED, 0)
        with self.assertRaises(DroppedEventsError):
            controller.run(events=(dropped,))
        self.assertTrue(source.ungrabbed)
        self.assertTrue(source.closed)
        self.assertTrue(ui.closed)

    def test_arbitrary_event_source_failure_still_cleans_up(self) -> None:
        controller, source, ui = self.make_controller()

        def broken_events():
            yield FakeEvent(EV_KEY, 304, 1)
            raise RuntimeError("test failure")

        with self.assertRaisesRegex(RuntimeError, "test failure"):
            controller.run(events=broken_events())
        self.assertTrue(source.ungrabbed)
        self.assertTrue(source.closed)
        self.assertTrue(ui.closed)


class ReadOnlyObservationTests(unittest.TestCase):
    def test_observation_records_changes_without_grabbing(self) -> None:
        source = FakeSource()
        source.read_batches = [
            [
                FakeEvent(EV_ABS, ABS_BRAKE, 255),
                FakeEvent(EV_KEY, 312, 1),
            ]
        ]
        current = 0.0

        def clock() -> float:
            nonlocal current
            current += 0.05
            return current

        def ready(_read: object, _write: object, _error: object, _timeout: float):
            return ([source], [], []) if source.read_batches else ([], [], [])

        report = observe_phase(
            "/dev/input/by-id/fake",
            phase="LT",
            seconds=0.3,
            input_device_factory=lambda _path: source,
            select_fn=ready,
            clock=clock,
        )
        brake = next(item for item in report.observations if item.code == ABS_BRAKE)
        self.assertEqual((brake.minimum, brake.maximum), (0, 255))
        self.assertEqual((brake.declared_minimum, brake.declared_maximum), (0, 255))
        self.assertEqual(brake.movement_fraction, 1.0)
        self.assertEqual(report.changed_buttons, (312,))
        self.assertFalse(source.grabbed)
        self.assertTrue(source.closed)

    def test_observation_rejects_replaced_unserialized_device_before_reading(self) -> None:
        path = "/dev/input/by-id/usb-PXN-event-joystick"
        replacement = FakeSource(serial="", phys="usb-replacement/input0")
        expected = ControllerIdentity(
            "PXN P5 8K",
            PXN_VENDOR_ID,
            PXN_P5_8K_PRODUCT_ID,
            "",
            "usb-original/input0",
            path,
        )

        with self.assertRaisesRegex(ControllerIdentityError, "identity changed"):
            observe_phase(
                path,
                phase="LT",
                seconds=0.1,
                expected_identity=expected,
                input_device_factory=lambda _path: replacement,
            )

        self.assertFalse(replacement.grabbed)
        self.assertTrue(replacement.closed)


if __name__ == "__main__":
    unittest.main()
