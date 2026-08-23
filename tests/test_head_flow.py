from __future__ import annotations

import unittest

import cv2
import numpy as np

from detection.head_flow import (
    HeadFlowConfig,
    HeadFlowPhaseAdvancer,
    direct_head_box_center,
    measure_head_translation,
)


def _textured_scene(
    *,
    shape: tuple[int, int] = (180, 260),
    box: tuple[int, int, int, int] = (80, 50, 140, 120),
    seed: int = 19,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 35, size=shape, dtype=np.uint8)
    x1, y1, x2, y2 = box
    texture = rng.integers(20, 245, size=(y2 - y1, x2 - x1), dtype=np.uint8)
    # Add stable corners at several scales instead of relying only on noise.
    for y in range(4, texture.shape[0] - 4, 9):
        for x in range(4, texture.shape[1] - 4, 9):
            cv2.circle(texture, (x, y), 2, int((x * 17 + y * 23) % 220 + 20), -1)
    gray[y1:y2, x1:x2] = texture
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _translate(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    transform = np.asarray(((1.0, 0.0, dx), (0.0, 1.0, dy)), dtype=np.float32)
    return cv2.warpAffine(
        frame,
        transform,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


class HeadTranslationTests(unittest.TestCase):
    def test_recovers_subpixel_translation(self) -> None:
        previous = _textured_scene()
        current = _translate(previous, 6.25, -3.5)

        measured = measure_head_translation(
            previous,
            current,
            (80, 50, 140, 120),
        )

        self.assertIsNotNone(measured)
        assert measured is not None
        self.assertAlmostEqual(measured.displacement[0], 6.25, delta=0.35)
        self.assertAlmostEqual(measured.displacement[1], -3.5, delta=0.35)
        self.assertGreaterEqual(measured.inlier_fraction, 0.6)

    def test_head_roi_ignores_differently_moving_background(self) -> None:
        previous = _textured_scene()
        background = _translate(previous, 7.0, 2.0)
        target_only = _translate(previous, -4.0, -2.0)
        current = background.copy()
        current[48:122, 74:142] = target_only[48:122, 74:142]

        measured = measure_head_translation(
            previous,
            current,
            (80, 50, 140, 120),
        )

        self.assertIsNotNone(measured)
        assert measured is not None
        self.assertAlmostEqual(measured.displacement[0], -4.0, delta=0.45)
        self.assertAlmostEqual(measured.displacement[1], -2.0, delta=0.45)

    def test_textureless_head_fails_closed(self) -> None:
        frame = np.zeros((120, 160), dtype=np.uint8)
        self.assertIsNone(
            measure_head_translation(frame, frame, (40, 30, 100, 90))
        )

    def test_excessive_motion_fails_closed(self) -> None:
        previous = _textured_scene()
        current = _translate(previous, 15.0, 0.0)
        config = HeadFlowConfig(max_frame_displacement_pixels=4.0)

        self.assertIsNone(
            measure_head_translation(
                previous,
                current,
                (80, 50, 140, 120),
                config=config,
            )
        )

    def test_stationary_sensor_noise_does_not_manufacture_motion(self) -> None:
        previous = _textured_scene()
        rng = np.random.default_rng(41)
        noise = rng.integers(-1, 2, size=previous.shape, dtype=np.int16)
        current = np.clip(previous.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        measured = measure_head_translation(
            previous,
            current,
            (80, 50, 140, 120),
        )

        self.assertIsNotNone(measured)
        assert measured is not None
        self.assertLess(np.hypot(*measured.displacement), 0.10)

    def test_bounded_feature_region_scale_can_use_same_target_surround(self) -> None:
        previous = _textured_scene()
        previous[50:120, 80:140] = 100
        current = _translate(previous, 3.0, -2.0)

        self.assertIsNone(
            measure_head_translation(previous, current, (80, 50, 140, 120))
        )
        measured = measure_head_translation(
            previous,
            current,
            (80, 50, 140, 120),
            config=HeadFlowConfig(feature_roi_scale=1.5),
        )

        self.assertIsNotNone(measured)
        assert measured is not None
        self.assertAlmostEqual(measured.displacement[0], 3.0, delta=0.45)
        self.assertAlmostEqual(measured.displacement[1], -2.0, delta=0.45)


class PhaseAdvancerTests(unittest.TestCase):
    def test_upper_body_features_translate_a_textureless_tiny_head(self) -> None:
        rng = np.random.default_rng(83)
        base = np.zeros((180, 260), dtype=np.uint8)
        upper_body_box = (78, 32, 142, 118)
        head_box = (102, 38, 110, 46)
        base[32:118, 78:142] = rng.integers(
            15,
            245,
            size=(86, 64),
            dtype=np.uint8,
        )
        # The anatomical localization stays textureless even though the
        # verified player's upper body has ample same-motion features.
        base[38:46, 102:110] = 100
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        current = _translate(base, 7.0, -3.0)
        config = HeadFlowConfig(
            min_features=3,
            min_feature_distance=1.5,
            feature_block_size=3,
            roi_inset_fraction=0.01,
            crosshair_exclusion_radius_pixels=0.0,
            min_feature_span_fraction=0.05,
        )

        head_only = HeadFlowPhaseAdvancer(config)
        body_features = HeadFlowPhaseAdvancer(config)
        for advancer in (head_only, body_features):
            advancer.remember(
                base,
                source_timestamp_ns=100,
                identity_generation=4,
            )
            advancer.remember(
                current,
                source_timestamp_ns=110,
                identity_generation=4,
            )

        self.assertIsNone(
            head_only.advance(
                head_box,
                anchor_point=(106.0, 42.0),
                anchor_timestamp_ns=100,
                identity_generation=4,
            )
        )
        result = body_features.advance(
            head_box,
            feature_box=upper_body_box,
            anchor_point=(106.0, 42.0),
            anchor_timestamp_ns=100,
            identity_generation=4,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.point[0], 113.0, delta=0.45)
        self.assertAlmostEqual(result.point[1], 39.0, delta=0.45)
        self.assertAlmostEqual(result.head_box[0], 109.0, delta=0.45)
        self.assertAlmostEqual(result.head_box[1], 35.0, delta=0.45)
        assert result.feature_box is not None
        self.assertAlmostEqual(result.feature_box[0], 85.0, delta=0.45)
        self.assertAlmostEqual(result.feature_box[1], 29.0, delta=0.45)

    def test_replays_explicit_observed_point_to_newest_frame(self) -> None:
        base = _textured_scene()
        advancer = HeadFlowPhaseAdvancer()
        start = 1_000_000_000
        for index, (dx, dy) in enumerate(((0, 0), (4, -2), (9, -3))):
            advancer.remember(
                _translate(base, dx, dy),
                source_timestamp_ns=start + index * 8_000_000,
                identity_generation=7,
            )

        result = advancer.advance(
            (80, 50, 140, 120),
            anchor_point=(101.0, 61.0),
            anchor_timestamp_ns=start,
            identity_generation=7,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.point[0], 110.0, delta=0.5)
        self.assertAlmostEqual(result.point[1], 58.0, delta=0.5)
        self.assertEqual(result.source_timestamp_ns, start + 16_000_000)
        self.assertEqual(result.hops, 2)
        self.assertEqual(result.frames_spanned, 2)
        self.assertEqual(result.flow_measurements, 1)
        self.assertEqual(result.strategy, "direct")

    def test_direct_endpoint_fallback_survives_one_bad_intermediate_frame(self) -> None:
        base = _textured_scene()
        advancer = HeadFlowPhaseAdvancer()
        start = 1_500_000_000
        for index, frame in enumerate(
            (
                base,
                np.zeros_like(base),
                _translate(base, 8.0, -3.0),
            )
        ):
            advancer.remember(
                frame,
                source_timestamp_ns=start + index * 8_000_000,
                identity_generation=9,
            )

        result = advancer.advance(
            (80, 50, 140, 120),
            anchor_point=(110.0, 85.0),
            anchor_timestamp_ns=start,
            identity_generation=9,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.strategy, "direct")
        self.assertEqual(result.frames_spanned, 2)
        self.assertEqual(result.flow_measurements, 1)
        self.assertAlmostEqual(result.point[0], 118.0, delta=0.5)
        self.assertAlmostEqual(result.point[1], 82.0, delta=0.5)

    def test_sequential_fallback_covers_net_motion_above_endpoint_bound(self) -> None:
        base = _textured_scene(shape=(220, 420), box=(80, 50, 140, 120))
        advancer = HeadFlowPhaseAdvancer(
            HeadFlowConfig(max_frame_displacement_pixels=55.0)
        )
        start = 3_000_000_000
        for index, dx in enumerate((0.0, 40.0, 80.0)):
            advancer.remember(
                _translate(base, dx, 0.0),
                source_timestamp_ns=start + index * 8_000_000,
                identity_generation=4,
            )

        result = advancer.advance(
            (80, 50, 140, 120),
            anchor_point=(110.0, 85.0),
            anchor_timestamp_ns=start,
            identity_generation=4,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.strategy, "sequential")
        self.assertEqual(result.flow_measurements, 2)
        self.assertAlmostEqual(result.point[0], 190.0, delta=1.0)

    def test_default_point_preserves_current_box_center_semantics(self) -> None:
        base = _textured_scene()
        advancer = HeadFlowPhaseAdvancer()
        advancer.remember(
            base,
            source_timestamp_ns=10,
            identity_generation=2,
        )
        advancer.remember(
            _translate(base, 3.0, 5.0),
            source_timestamp_ns=20,
            identity_generation=2,
        )

        result = advancer.advance(
            (80, 50, 140, 120),
            anchor_timestamp_ns=10,
            identity_generation=2,
        )

        self.assertIsNotNone(result)
        assert result is not None
        center = direct_head_box_center((80, 50, 140, 120))
        self.assertAlmostEqual(result.point[0], center[0] + 3.0, delta=0.4)
        self.assertAlmostEqual(result.point[1], center[1] + 5.0, delta=0.4)

    def test_default_bounds_cover_measured_fifty_five_ms_result_age(self) -> None:
        base = _textured_scene()
        advancer = HeadFlowPhaseAdvancer()
        start = 2_000_000_000
        # Twelve frames at a 5 ms interval model the observed capture cadence
        # and put the newest image 55 ms after the detector's source frame.
        for index in range(12):
            advancer.remember(
                _translate(base, float(index), 0.0),
                source_timestamp_ns=start + index * 5_000_000,
                identity_generation=3,
            )

        result = advancer.advance(
            (80, 50, 140, 120),
            anchor_point=(110.0, 85.0),
            anchor_timestamp_ns=start,
            identity_generation=3,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.hops, 11)
        self.assertAlmostEqual(result.point[0], 121.0, delta=0.8)

    def test_age_and_hop_bounds_fail_closed(self) -> None:
        base = _textured_scene()
        config = HeadFlowConfig(max_hops=2, max_phase_advance_seconds=0.02)
        advancer = HeadFlowPhaseAdvancer(config)
        start = 1_000_000_000
        for index in range(4):
            advancer.remember(
                _translate(base, float(index), 0.0),
                source_timestamp_ns=start + index * 8_000_000,
                identity_generation=1,
            )
        self.assertIsNone(
            advancer.advance(
                (80, 50, 140, 120),
                anchor_point=(110, 85),
                anchor_timestamp_ns=start,
                identity_generation=1,
            )
        )

    def test_identity_crossing_and_missing_anchor_fail_closed(self) -> None:
        base = _textured_scene()
        advancer = HeadFlowPhaseAdvancer()
        advancer.remember(
            base,
            source_timestamp_ns=1_000,
            identity_generation=1,
        )
        advancer.remember(
            base,
            source_timestamp_ns=2_000,
            identity_generation=2,
        )

        self.assertIsNone(
            advancer.advance(
                (80, 50, 140, 120),
                anchor_timestamp_ns=1_000,
                identity_generation=1,
            )
        )
        self.assertIsNone(
            advancer.advance(
                (80, 50, 140, 120),
                anchor_timestamp_ns=1_500,
                identity_generation=2,
            )
        )

    def test_history_is_memory_and_frame_bounded(self) -> None:
        config = HeadFlowConfig(
            max_history_frames=4,
            max_history_bytes=1024 * 1024,
        )
        advancer = HeadFlowPhaseAdvancer(config)
        frame = _textured_scene(shape=(120, 160), box=(40, 20, 100, 90))
        for index in range(10):
            advancer.remember(
                frame,
                source_timestamp_ns=index,
                identity_generation=1,
            )

        self.assertEqual(advancer.history_size, 4)
        self.assertLessEqual(advancer.history_bytes, config.max_history_bytes)

    def test_resolution_change_invalidates_old_history(self) -> None:
        advancer = HeadFlowPhaseAdvancer()
        advancer.remember(
            np.zeros((100, 120), dtype=np.uint8),
            source_timestamp_ns=100,
            identity_generation=1,
        )
        advancer.remember(
            np.zeros((80, 90), dtype=np.uint8),
            source_timestamp_ns=200,
            identity_generation=1,
        )
        self.assertEqual(advancer.history_size, 1)


if __name__ == "__main__":
    unittest.main()
