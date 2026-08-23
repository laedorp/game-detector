from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import stat
import statistics
import tempfile
import unittest
from unittest import mock

from aiming.makcu_calibration import (
    CalibrationDataError,
    CalibrationMeasurement,
    CalibrationPulse,
    CalibrationQualityError,
    EmittedCount,
    MakcuCalibrationFit,
    MakcuCalibrationProfile,
    calibration_evidence_sha256,
    canonical_profile_bytes,
    fit_makcu_calibration,
    load_profile,
    make_profile,
    profile_from_bytes,
    write_profile_atomic,
    _axis_candidates,
    _prefix_counts,
    _regression_window_intervals,
)


NANOSECONDS_PER_MILLISECOND = 1_000_000


def _synthetic_evidence(
    *,
    delay_ms: float = 24.0,
    gain_x: float = 0.14,
    gain_y: float = 0.10,
    noise_pixels: float = 0.04,
    dropout_every: int | None = None,
    dropout_run: tuple[int, int] | None = None,
    x_negative_gain: float | None = None,
    y_negative_gain: float | None = None,
    cross_x_to_y: float = 0.004,
    cross_y_to_x: float = 0.003,
    nonlinear_motion_pixels: float = 0.0,
    pulse_delay_offsets_ms: tuple[float, ...] | None = None,
    adaptive_scouts: bool = False,
    scout_counts: tuple[int, int] = (40, 80),
    missing_x_negative: bool = False,
) -> tuple[
    tuple[CalibrationMeasurement, ...],
    tuple[EmittedCount, ...],
    tuple[CalibrationPulse, ...],
]:
    """Build deterministic detector samples from timestamped emitted counts."""

    x_negative_gain = gain_x if x_negative_gain is None else x_negative_gain
    y_negative_gain = gain_y if y_negative_gain is None else y_negative_gain
    if adaptive_scouts:
        per_axis = (
            (1, scout_counts[0]),
            (-1, scout_counts[0]),
            (1, scout_counts[1]),
            (-1, scout_counts[1]),
            (1, 140),
            (-1, 140),
            (1, 140),
            (-1, 140),
        )
        schedule = tuple(
            (axis, polarity, count)
            for axis in ("x", "y")
            for polarity, count in per_axis
        )
    else:
        schedule = tuple(
            (axis, polarity, 160)
            for axis, polarity in (
                ("x", 1),
                ("x", -1),
                ("y", 1),
                ("y", -1),
                ("x", -1),
                ("x", 1),
                ("y", -1),
                ("y", 1),
                ("x", 1),
                ("x", -1),
                ("y", 1),
                ("y", -1),
            )
        )
    if missing_x_negative:
        schedule = tuple(
            pulse
            for pulse in schedule
            if not (pulse[0] == "x" and pulse[1] < 0)
        )
    start_ns = 400 * NANOSECONDS_PER_MILLISECOND
    spacing_ns = 250 * NANOSECONDS_PER_MILLISECOND
    base_delay_ns = round(delay_ms * NANOSECONDS_PER_MILLISECOND)
    pulses: list[CalibrationPulse] = []
    commands: list[EmittedCount] = []
    # (screen-visible timestamp, error-X increment, error-Y increment)
    response_events: list[tuple[int, float, float]] = []
    for pulse_index, (axis, polarity, absolute_counts) in enumerate(schedule):
        pulse_start = start_ns + pulse_index * spacing_ns
        command_samples = absolute_counts // 2
        pulse_span_ns = (command_samples - 1) * NANOSECONDS_PER_MILLISECOND
        pulse_end = pulse_start + pulse_span_ns
        pulse = CalibrationPulse(axis, polarity, pulse_start, pulse_end)
        pulses.append(pulse)
        pulse_delay_ns = base_delay_ns
        if pulse_delay_offsets_ms is not None:
            pulse_delay_ns += round(
                pulse_delay_offsets_ms[pulse_index % len(pulse_delay_offsets_ms)]
                * NANOSECONDS_PER_MILLISECOND
            )
        for offset_ms in range(command_samples):
            command_timestamp = pulse_start + offset_ms * NANOSECONDS_PER_MILLISECOND
            delta_x = 2 * polarity if axis == "x" else 0
            delta_y = 2 * polarity if axis == "y" else 0
            commands.append(EmittedCount(command_timestamp, delta_x, delta_y))
            x_gain_for_command = gain_x if delta_x >= 0 else x_negative_gain
            y_gain_for_command = gain_y if delta_y >= 0 else y_negative_gain
            response_events.append(
                (
                    command_timestamp + pulse_delay_ns,
                    -x_gain_for_command * delta_x - cross_y_to_x * delta_y,
                    -y_gain_for_command * delta_y - cross_x_to_y * delta_x,
                )
            )

    response_events.sort(key=lambda value: value[0])
    detector_period_ns = 8 * NANOSECONDS_PER_MILLISECOND
    final_timestamp = pulses[-1].end_ns + 250 * NANOSECONDS_PER_MILLISECOND
    response_index = 0
    response_x = 0.0
    response_y = 0.0
    measurements: list[CalibrationMeasurement] = []
    for sample_index, timestamp_ns in enumerate(
        range(0, final_timestamp + 1, detector_period_ns)
    ):
        while (
            response_index < len(response_events)
            and response_events[response_index][0] <= timestamp_ns
        ):
            (
                _visible_timestamp,
                increment_x,
                increment_y,
            ) = response_events[response_index]
            response_x += increment_x
            response_y += increment_y
            response_index += 1
        seconds = timestamp_ns / 1_000_000_000
        target_motion = nonlinear_motion_pixels * math.sin(
            2.0 * math.pi * 1.7 * seconds
        )
        error_x = (
            36.0
            + response_x
            + 0.35 * seconds
            + target_motion
            + noise_pixels * math.sin(sample_index * 0.731)
        )
        error_y = (
            -24.0
            + response_y
            - 0.28 * seconds
            - target_motion * 0.83
            + noise_pixels * math.cos(sample_index * 0.619)
        )
        observed = not (
            dropout_every is not None
            and (sample_index + 17) % dropout_every == 0
        )
        if dropout_run is not None and dropout_run[0] <= sample_index < (
            dropout_run[0] + dropout_run[1]
        ):
            observed = False
        measurements.append(
            CalibrationMeasurement(timestamp_ns, error_x, error_y, observed)
        )
    return tuple(measurements), tuple(commands), tuple(pulses)


def _fit(**options: object) -> MakcuCalibrationFit:
    return fit_makcu_calibration(*_synthetic_evidence(**options))


def _high_rate_quantized_evidence() -> tuple[
    tuple[CalibrationMeasurement, ...],
    tuple[EmittedCount, ...],
    tuple[CalibrationPulse, ...],
]:
    """Model high-rate box quantization with sparse deterministic jumps."""

    measurements, commands, pulses = _synthetic_evidence(
        gain_x=0.115,
        gain_y=0.105,
        noise_pixels=0.0,
    )
    noisy: list[CalibrationMeasurement] = []
    for index, measurement in enumerate(measurements):
        x_noise = (
            1.8
            if index % 53 == 11
            else (-0.63 if index % 53 == 12 else 0.0)
        )
        y_noise = (
            2.8
            if index % 37 == 7
            else (-1.12 if index % 37 == 8 else 0.0)
        )
        noisy.append(
            replace(
                measurement,
                error_x=round((measurement.error_x + x_noise) * 4.0) / 4.0,
                error_y=round((measurement.error_y + y_noise) * 4.0) / 4.0,
            )
        )
    return tuple(noisy), commands, pulses


def _best_candidate_r_squared(
    evidence: tuple[
        tuple[CalibrationMeasurement, ...],
        tuple[EmittedCount, ...],
        tuple[CalibrationPulse, ...],
    ],
    axis: str,
    regression_window_intervals: int,
) -> float:
    measurements, commands, _pulses = evidence
    timestamp_steps = [
        current.timestamp_ns - previous.timestamp_ns
        for previous, current in zip(measurements, measurements[1:])
    ]
    detector_period_ns = round(statistics.median(timestamp_steps))
    maximum_delay_ns = 100_000_000
    delays = list(range(0, maximum_delay_ns + 1, detector_period_ns))
    if delays[-1] != maximum_delay_ns:
        delays.append(maximum_delay_ns)
    command_timestamps, prefix_x, prefix_y = _prefix_counts(commands)
    candidates = _axis_candidates(
        axis,
        measurements,
        command_timestamps,
        prefix_x,
        prefix_y,
        delays,
        regression_window_intervals,
    )
    return max(
        candidate.r_squared for candidate in candidates if candidate.gain > 0.0
    )


def _profile(fit: MakcuCalibrationFit) -> MakcuCalibrationProfile:
    return make_profile(
        fit,
        profile_name="RX6950XT capture",
        aim_mode="ads",
        capture_width=1920,
        capture_height=1080,
        capture_fps=125.0,
        makcu_identity_token="usb-1a86_USB_Single_Serial_TEST000000-if00",
        model_sha256="a" * 64,
        source_commit="7be5eb145c38dac3495d15c4693392570561cb99",
    )


class MakcuCalibrationFitTests(unittest.TestCase):
    def test_integrated_regression_survives_high_rate_quantized_box_noise(self) -> None:
        evidence = _high_rate_quantized_evidence()
        measurements, _commands, _pulses = evidence
        detector_period_ns = round(
            statistics.median(
                current.timestamp_ns - previous.timestamp_ns
                for previous, current in zip(measurements, measurements[1:])
            )
        )

        legacy_x_r_squared = _best_candidate_r_squared(evidence, "x", 1)
        legacy_y_r_squared = _best_candidate_r_squared(evidence, "y", 1)
        self.assertLess(legacy_x_r_squared, 0.85)
        self.assertLess(legacy_y_r_squared, 0.85)
        self.assertAlmostEqual(legacy_x_r_squared, 0.72, delta=0.08)
        self.assertAlmostEqual(legacy_y_r_squared, 0.35, delta=0.08)
        self.assertGreaterEqual(
            _regression_window_intervals(detector_period_ns),
            12,
        )

        fit = fit_makcu_calibration(*evidence)

        self.assertGreaterEqual(fit.x.r_squared, 0.85)
        self.assertGreaterEqual(fit.y.r_squared, 0.85)
        self.assertAlmostEqual(fit.x.gain_pixels_per_count, 0.115, delta=0.012)
        self.assertAlmostEqual(fit.y.gain_pixels_per_count, 0.105, delta=0.012)

    def test_unequal_axis_gains_and_12_to_50_ms_delays_recover_with_noise(self) -> None:
        for delay_ms in (12.0, 24.0, 50.0):
            with self.subTest(delay_ms=delay_ms):
                fit = _fit(
                    delay_ms=delay_ms,
                    gain_x=0.14,
                    gain_y=0.10,
                    noise_pixels=0.08,
                    dropout_every=151,
                )
                self.assertLessEqual(
                    abs(fit.x.gain_pixels_per_count / 0.14 - 1.0), 0.10
                )
                self.assertLessEqual(
                    abs(fit.y.gain_pixels_per_count / 0.10 - 1.0), 0.10
                )
                self.assertLessEqual(abs(fit.delay_seconds - delay_ms / 1000.0), 0.008)
                self.assertGreaterEqual(fit.x.r_squared, 0.85)
                self.assertGreaterEqual(fit.y.r_squared, 0.85)
                self.assertGreaterEqual(fit.observation_duty, 0.98)
                self.assertAlmostEqual(fit.x.drift_pixels_per_second, 0.35, delta=0.08)
                self.assertAlmostEqual(fit.y.drift_pixels_per_second, -0.28, delta=0.08)

    def test_adaptive_scouts_fit_within_2400_count_session_envelope(self) -> None:
        measurements, commands, pulses = _synthetic_evidence(
            adaptive_scouts=True,
            gain_x=0.10,
            gain_y=0.10,
            noise_pixels=0.06,
            dropout_every=173,
        )
        emitted_absolute_counts = sum(
            abs(command.delta_x) + abs(command.delta_y) for command in commands
        )
        self.assertEqual(emitted_absolute_counts, 1600)
        self.assertLessEqual(emitted_absolute_counts, 2400)

        fit = fit_makcu_calibration(measurements, commands, pulses)

        self.assertAlmostEqual(fit.x.gain_pixels_per_count, 0.10, delta=0.01)
        self.assertAlmostEqual(fit.y.gain_pixels_per_count, 0.10, delta=0.01)
        self.assertEqual((fit.x.positive_pulses, fit.x.negative_pulses), (2, 2))
        self.assertEqual((fit.y.positive_pulses, fit.y.negative_pulses), (2, 2))
        self.assertGreaterEqual(fit.x.minimum_excursion_pixels, 12.0)
        self.assertGreaterEqual(fit.y.minimum_excursion_pixels, 12.0)
        self.assertEqual(
            fit.evidence_sha256,
            calibration_evidence_sha256(measurements, commands, pulses),
        )

    def test_noise_flipped_tiny_scout_does_not_poison_qualifying_pulses(self) -> None:
        measurements, commands, pulses = _synthetic_evidence(
            adaptive_scouts=True,
            scout_counts=(4, 80),
            gain_x=0.10,
            gain_y=0.10,
        )
        first_scout = pulses[0]
        modified = list(measurements)
        # The +X four-count scout truly moves error by -0.4px. A short +0.8px
        # detector disturbance makes its settled median look like a wrong-sign
        # sub-pixel response. It remains in regression/evidence but is ignored
        # by the qualifying-pulse quality metrics.
        settled_start = first_scout.end_ns + 24 * NANOSECONDS_PER_MILLISECOND
        settled_end = settled_start + 16 * NANOSECONDS_PER_MILLISECOND
        for index, measurement in enumerate(modified):
            if settled_start <= measurement.timestamp_ns <= settled_end:
                modified[index] = replace(
                    measurement,
                    error_x=measurement.error_x + 0.8,
                )

        fit = fit_makcu_calibration(modified, commands, pulses)

        self.assertAlmostEqual(fit.x.gain_pixels_per_count, 0.10, delta=0.01)
        self.assertEqual((fit.x.positive_pulses, fit.x.negative_pulses), (2, 2))

    def test_endpoint_quality_uses_one_repeated_count_magnitude(self) -> None:
        fit = fit_makcu_calibration(
            *_synthetic_evidence(
                adaptive_scouts=True,
                scout_counts=(130, 150),
                gain_x=0.10,
                gain_y=0.10,
            )
        )

        # The 130- and 150-count scouts also exceed 12 px, but each has only
        # one pulse per polarity. Only the final repeated 140-count group may
        # establish repeatability quality.
        self.assertEqual((fit.x.positive_pulses, fit.x.negative_pulses), (2, 2))
        self.assertEqual((fit.y.positive_pulses, fit.y.negative_pulses), (2, 2))

    def test_evidence_hash_binds_exact_emitted_history(self) -> None:
        measurements, commands, pulses = _synthetic_evidence()
        digest = calibration_evidence_sha256(measurements, commands, pulses)
        changed = list(commands)
        changed[0] = replace(changed[0], delta_x=changed[0].delta_x + 1)
        self.assertEqual(
            digest,
            calibration_evidence_sha256(measurements, commands, pulses),
        )
        self.assertNotEqual(
            digest,
            calibration_evidence_sha256(measurements, changed, pulses),
        )

    def test_rejects_insufficient_excursion(self) -> None:
        with self.assertRaisesRegex(CalibrationQualityError, "excursion"):
            _fit(gain_x=0.06, gain_y=0.05)

    def test_rejects_excessive_excursion(self) -> None:
        with self.assertRaisesRegex(CalibrationQualityError, "excursion"):
            _fit(gain_x=0.70, gain_y=0.65)

    def test_rejects_low_observation_duty(self) -> None:
        # A sustained unbroken run of unobserved frames starves the axis fit
        # even when the surrounding data is dense; this must still reject.
        with self.assertRaisesRegex(CalibrationQualityError, "observation duty"):
            _fit(dropout_run=(100, 40))

    def test_accepts_isolated_single_frame_drops(self) -> None:
        # Periodic single-frame motion-blur drops (as seen from a 235 Hz
        # detector during pulse reversals) leave the fit fully constrained.
        fit = _fit(dropout_every=10)
        self.assertGreaterEqual(fit.observation_duty, 0.70)

    def test_rejects_wrong_response_sign(self) -> None:
        with self.assertRaisesRegex(CalibrationQualityError, "wrong sign"):
            _fit(gain_x=-0.14)

    def test_rejects_asymmetric_polarity_response(self) -> None:
        with self.assertRaises(CalibrationQualityError):
            _fit(x_negative_gain=0.08, y_negative_gain=0.055)

    def test_rejects_cross_axis_response(self) -> None:
        with self.assertRaisesRegex(CalibrationQualityError, "cross response"):
            _fit(cross_x_to_y=0.027, cross_y_to_x=0.032)

    def test_rejects_nonstationary_target(self) -> None:
        with self.assertRaisesRegex(CalibrationQualityError, "R-squared"):
            _fit(nonlinear_motion_pixels=18.0)

    def test_rejects_pulse_to_pulse_delay_spread(self) -> None:
        with self.assertRaisesRegex(CalibrationQualityError, "delay"):
            _fit(pulse_delay_offsets_ms=(0.0, 40.0), noise_pixels=0.12)

    def test_accepts_local_delays_one_frame_each_side_of_global_fit(self) -> None:
        fit = _fit(pulse_delay_offsets_ms=(0.0, 16.0), noise_pixels=0.12)

        self.assertAlmostEqual(fit.x.pulse_delay_spread_seconds, 0.008)
        self.assertAlmostEqual(fit.y.pulse_delay_spread_seconds, 0.008)

    def test_accepts_pulse_delay_spread_up_to_one_and_a_half_frames(self) -> None:
        fit = _fit(pulse_delay_offsets_ms=(0.0, 20.0), noise_pixels=0.12)

        self.assertLessEqual(fit.x.pulse_delay_spread_seconds, 0.012)
        self.assertLessEqual(fit.y.pulse_delay_spread_seconds, 0.012)

    def test_rejects_nonfinite_or_out_of_order_evidence(self) -> None:
        with self.assertRaises(CalibrationDataError):
            CalibrationMeasurement(0, math.nan, 0.0)
        measurements, commands, pulses = _synthetic_evidence()
        disordered = list(measurements)
        disordered[20], disordered[21] = disordered[21], disordered[20]
        with self.assertRaisesRegex(CalibrationDataError, "strictly increasing"):
            fit_makcu_calibration(disordered, commands, pulses)

    def test_rejects_missing_symmetric_polarity(self) -> None:
        with self.assertRaisesRegex(CalibrationQualityError, "symmetric"):
            fit_makcu_calibration(*_synthetic_evidence(missing_x_negative=True))

    def test_rejects_an_axis_without_command_excitation(self) -> None:
        measurements, commands, pulses = _synthetic_evidence()
        x_pulses = tuple(pulse for pulse in pulses if pulse.axis == "x")
        x_commands = tuple(
            command
            for command in commands
            if any(
                pulse.start_ns <= command.timestamp_ns <= pulse.end_ns
                for pulse in x_pulses
            )
        )

        with self.assertRaisesRegex(CalibrationDataError, "no solvable [xy]-axis"):
            fit_makcu_calibration(measurements, x_commands, x_pulses)


class MakcuCalibrationProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fit = _fit()
        cls.profile = _profile(cls.fit)

    def test_profile_binds_hardware_runtime_mode_and_quality(self) -> None:
        profile = self.profile
        self.assertEqual(profile.aim_mode, "ads")
        self.assertEqual(profile.capture_width, 1920)
        self.assertEqual(profile.capture_height, 1080)
        self.assertEqual(profile.capture_fps, 125.0)
        self.assertIn("TEST000000", profile.makcu_identity_token)
        self.assertEqual(profile.model_sha256, "a" * 64)
        self.assertEqual(
            profile.source_commit,
            "7be5eb145c38dac3495d15c4693392570561cb99",
        )
        self.assertEqual(profile.evidence_sha256, self.fit.evidence_sha256)
        self.assertEqual(profile.detector_period_seconds, 0.008)
        self.assertGreaterEqual(profile.observation_duty, 0.98)
        self.assertEqual(profile.x_quality, self.fit.x)
        self.assertEqual(profile.y_quality, self.fit.y)

    def test_canonical_bytes_are_deterministic_and_round_trip(self) -> None:
        first = canonical_profile_bytes(self.profile)
        second = canonical_profile_bytes(self.profile)
        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertNotIn(b": ", first)
        self.assertNotIn(b", ", first)
        self.assertEqual(profile_from_bytes(first), self.profile)
        with self.assertRaisesRegex(CalibrationDataError, "canonical"):
            profile_from_bytes(json.dumps(json.loads(first)).encode("utf-8"))

    def test_malformed_profile_is_rejected(self) -> None:
        document = json.loads(canonical_profile_bytes(self.profile))
        document["capture"]["width"] = True
        malformed = (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self.assertRaises(CalibrationDataError):
            profile_from_bytes(malformed)
        document = json.loads(canonical_profile_bytes(self.profile))
        document["makcu_identity_token"] = 123
        malformed = (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self.assertRaises(CalibrationDataError):
            profile_from_bytes(malformed)
        with self.assertRaisesRegex(CalibrationDataError, "observation duty"):
            replace(self.profile, observation_duty=0.90)

    def test_atomic_write_is_mode_0600_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "profiles" / "ads.json"
            write_profile_atomic(destination, self.profile)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            self.assertEqual(
                destination.read_bytes(), canonical_profile_bytes(self.profile)
            )
            self.assertEqual(load_profile(destination), self.profile)

    def test_atomic_write_failure_preserves_last_good_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "ads.json"
            write_profile_atomic(destination, self.profile)
            previous = destination.read_bytes()
            updated = replace(self.profile, profile_name="replacement")
            with mock.patch(
                "aiming.makcu_calibration.os.replace",
                side_effect=OSError("simulated rename failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated rename failure"):
                    write_profile_atomic(destination, updated)
            self.assertEqual(destination.read_bytes(), previous)
            self.assertEqual(
                [entry.name for entry in destination.parent.iterdir()],
                [destination.name],
            )
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
