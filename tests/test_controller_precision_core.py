from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

from controller_precision.codes import (
    ABS_BRAKE,
    ABS_RZ,
    ABS_Z,
    BTN_TL2,
    EV_ABS,
    EV_KEY,
    EV_SYN,
    SYN_DROPPED,
    SYN_REPORT,
)
from controller_precision.core import (
    AxisRange,
    DroppedEventsError,
    EventMapper,
    PrecisionConfig,
    TriggerCalibration,
    TriggerHysteresis,
    apply_precision_curve,
)


@dataclass(frozen=True, slots=True)
class FakeEvent:
    type: int
    code: int
    value: int
    marker: str = ""


def event(event_type: int, code: int, value: int, marker: str = "") -> FakeEvent:
    return FakeEvent(event_type, code, value, marker)


SYNC = event(EV_SYN, SYN_REPORT, 0, "sync")


class AxisRangeTests(unittest.TestCase):
    def test_unsigned_axis_round_trips_endpoints_and_center(self) -> None:
        axis = AxisRange(0, 255)
        self.assertEqual(axis.normalize(0), -1.0)
        self.assertEqual(axis.normalize(255), 1.0)
        self.assertIn(axis.encode(0.0), (127, 128))
        for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
            self.assertAlmostEqual(axis.normalize(axis.encode(value)), value, delta=0.01)

    def test_calibrated_off_center_axis_is_asymmetric(self) -> None:
        axis = AxisRange(10, 250, center=130)
        self.assertEqual(axis.normalize(130), 0.0)
        self.assertEqual(axis.encode(-1.0), 10)
        self.assertEqual(axis.encode(1.0), 250)

    def test_bad_range_or_center_is_rejected(self) -> None:
        for values in ((1, 1, None), (2, 1, None), (0, 10, 11), (0, 10, math.nan)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AxisRange(*values)


class TriggerTests(unittest.TestCase):
    def test_hysteresis_does_not_chatter_between_thresholds(self) -> None:
        trigger = TriggerHysteresis(
            TriggerCalibration(0, 100),
            activate_at=0.6,
            release_at=0.4,
        )
        self.assertFalse(trigger.update(59))
        self.assertTrue(trigger.update(60))
        self.assertTrue(trigger.update(41))
        self.assertFalse(trigger.update(40))

    def test_reversed_trigger_calibration_is_supported(self) -> None:
        calibration = TriggerCalibration(rest=255, pressed=0)
        self.assertEqual(calibration.pressure(255), 0.0)
        self.assertEqual(calibration.pressure(0), 1.0)
        self.assertAlmostEqual(calibration.pressure(128), 127 / 255)


class PrecisionCurveTests(unittest.TestCase):
    def test_deadzone_centers_small_input(self) -> None:
        config = PrecisionConfig(deadzone=0.1)
        self.assertEqual(apply_precision_curve(0.05, -0.04, config), (0.0, 0.0))

    def test_full_input_is_limited_to_configured_strength(self) -> None:
        config = PrecisionConfig(strength=0.35, exponent=1.4, deadzone=0.04)
        x, y = apply_precision_curve(1.0, 0.0, config)
        self.assertAlmostEqual(x, 0.35)
        self.assertEqual(y, 0.0)

    def test_curve_is_radial_symmetric_and_preserves_direction(self) -> None:
        config = PrecisionConfig(strength=0.5, exponent=1.7, deadzone=0.0)
        x, y = apply_precision_curve(0.3, 0.4, config)
        self.assertAlmostEqual(x / y, 0.75)
        self.assertAlmostEqual(math.hypot(x, y), 0.5 * (0.5**1.7))
        neg_x, neg_y = apply_precision_curve(-0.3, -0.4, config)
        self.assertAlmostEqual(neg_x, -x)
        self.assertAlmostEqual(neg_y, -y)

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid = (
            {"strength": 0.0},
            {"strength": 1.01},
            {"exponent": 0.0},
            {"deadzone": 1.0},
            {"activate_at": 0.2, "release_at": 0.3},
            {"strength": math.nan},
            {"right_x_code": ABS_Z, "right_y_code": ABS_Z},
            {"right_x_code": ABS_Z, "trigger_axis_code": ABS_Z},
            {"right_y_code": ABS_RZ, "trigger_axis_code": ABS_RZ},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PrecisionConfig(**changes)


class EventMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper: EventMapper[FakeEvent] = EventMapper(
            PrecisionConfig(strength=0.35, exponent=1.0, deadzone=0.0),
            initial_x=128,
            initial_y=128,
            initial_trigger=0,
        )

    def test_steady_inactive_report_is_exact_passthrough(self) -> None:
        report = (
            event(EV_KEY, 304, 1, "button"),
            event(EV_ABS, ABS_Z, 201, "right-x"),
            event(EV_ABS, ABS_RZ, 61, "right-y"),
            event(EV_ABS, ABS_BRAKE, 20, "lt-noise"),
            SYNC,
        )
        output = self.mapper.map_report(report)
        self.assertEqual(len(output), len(report))
        self.assertTrue(all(item.is_passthrough for item in output))
        self.assertEqual(tuple(item.source for item in output), report)
        self.assertTrue(all(item.source is source for item, source in zip(output, report)))

    def test_lt_activation_recomputes_both_axes_without_stick_event(self) -> None:
        lt = event(EV_ABS, ABS_BRAKE, 255, "lt")
        output = self.mapper.map_report((lt, SYNC))
        self.assertTrue(self.mapper.active)
        self.assertIs(output[0].source, lt)
        replacements = [item for item in output if not item.is_passthrough]
        self.assertEqual([item.code for item in replacements], [ABS_Z, ABS_RZ])
        self.assertEqual(replacements[0].value, 128)
        self.assertEqual(replacements[1].value, 128)
        self.assertIs(output[-1].source, SYNC)

    def test_active_right_stick_events_are_replaced_with_precision_values(self) -> None:
        self.mapper.map_report((event(EV_ABS, ABS_BRAKE, 255), SYNC))
        raw_x = event(EV_ABS, ABS_Z, 255, "raw-x")
        raw_y = event(EV_ABS, ABS_RZ, 128, "raw-y")
        output = self.mapper.map_report((raw_x, raw_y, SYNC))
        self.assertNotIn(raw_x, (item.source for item in output))
        self.assertNotIn(raw_y, (item.source for item in output))
        replacements = [item for item in output if not item.is_passthrough]
        self.assertEqual(replacements[0].code, ABS_Z)
        self.assertAlmostEqual(
            AxisRange(0, 255).normalize(replacements[0].value),
            0.35,
            delta=0.01,
        )
        self.assertAlmostEqual(
            AxisRange(0, 255).normalize(replacements[1].value),
            0.0,
            delta=0.01,
        )

    def test_lt_release_restores_exact_raw_axes_immediately(self) -> None:
        self.mapper.map_report((event(EV_ABS, ABS_BRAKE, 255), SYNC))
        self.mapper.map_report(
            (event(EV_ABS, ABS_Z, 240), event(EV_ABS, ABS_RZ, 30), SYNC)
        )
        lt_release = event(EV_ABS, ABS_BRAKE, 0, "release")
        output = self.mapper.map_report((lt_release, SYNC))
        replacements = [item for item in output if not item.is_passthrough]
        self.assertFalse(self.mapper.active)
        self.assertEqual(
            [(item.code, item.value) for item in replacements],
            [(ABS_Z, 240), (ABS_RZ, 30)],
        )

        ordinary = (event(EV_KEY, 305, 1, "ordinary"), SYNC)
        steady_output = self.mapper.map_report(ordinary)
        self.assertEqual(tuple(item.source for item in steady_output), ordinary)

    def test_default_mapping_ignores_unverified_digital_button(self) -> None:
        output = self.mapper.map_report((event(EV_KEY, BTN_TL2, 1), SYNC))
        self.assertFalse(self.mapper.active)
        self.assertTrue(all(item.is_passthrough for item in output))

    def test_verified_digital_companion_can_be_enabled_explicitly(self) -> None:
        mapper: EventMapper[FakeEvent] = EventMapper(
            PrecisionConfig(trigger_button_code=BTN_TL2),
        )
        mapper.map_report((event(EV_KEY, BTN_TL2, 1), SYNC))
        self.assertTrue(mapper.active)
        mapper.map_report((event(EV_KEY, BTN_TL2, 0), SYNC))
        self.assertFalse(mapper.active)

    def test_feed_buffers_until_syn_report(self) -> None:
        first = event(EV_KEY, 304, 1)
        self.assertEqual(self.mapper.feed(first), ())
        output = self.mapper.feed(SYNC)
        self.assertEqual(tuple(item.source for item in output), (first, SYNC))

    def test_syn_dropped_fails_closed_and_discards_partial_report(self) -> None:
        self.mapper.feed(event(EV_KEY, 304, 1))
        with self.assertRaises(DroppedEventsError):
            self.mapper.feed(event(EV_SYN, SYN_DROPPED, 0))
        output = self.mapper.feed(SYNC)
        self.assertEqual(tuple(item.source for item in output), (SYNC,))


if __name__ == "__main__":
    unittest.main()
