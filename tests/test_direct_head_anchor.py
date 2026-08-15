from __future__ import annotations

import math
import unittest

from aiming.direct_head_anchor import (
    DIRECT_HEAD_ANCHOR_MAX_AGE_SECONDS,
    DirectHeadAnchor,
    DirectHeadProvenance,
)


NS_PER_MS = 1_000_000


class DirectHeadAnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = DirectHeadAnchor()
        self.body = (100.0, 100.0, 300.0, 700.0)
        self.head = (220.0, 172.0)

    def seed(
        self,
        *,
        point: tuple[float, float] | None = None,
        body: tuple[float, float, float, float] | None = None,
        generation: int = 4,
        timestamp_ns: int = 100 * NS_PER_MS,
        confidence: float = 0.80,
    ):
        return self.anchor.observe_direct(
            point or self.head,
            body or self.body,
            track_generation=generation,
            source_timestamp_ns=timestamp_ns,
            confidence=confidence,
        )

    def test_direct_seed_binds_exact_source_identity_and_normalized_point(
        self,
    ) -> None:
        sample = self.seed()

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.point, self.head)
        self.assertEqual(sample.source_timestamp_ns, 100 * NS_PER_MS)
        self.assertEqual(sample.direct_source_timestamp_ns, 100 * NS_PER_MS)
        self.assertEqual(sample.identity_deadline_ns, 300 * NS_PER_MS)
        self.assertEqual(sample.track_generation, 4)
        self.assertIs(sample.provenance, DirectHeadProvenance.DIRECT)
        self.assertFalse(sample.body_derived)
        self.assertTrue(sample.primary_observed)
        self.assertTrue(sample.motion_corroboration_permitted)
        self.assertEqual(sample.confidence, 0.80)
        normalized = self.anchor.normalized_point
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertAlmostEqual(normalized[0], 0.60)
        self.assertAlmostEqual(normalized[1], 0.12)

    def test_translation_and_scale_use_observed_anchor_not_head_ratio(self) -> None:
        self.seed()
        # Translate by (+200, +50) and scale both axes by 1.5 around the box's
        # top-left coordinate. The observed normalized head location maps to
        # (480, 258), which is deliberately not a fixed 12%-from-top heuristic.
        mapped = self.anchor.map_primary(
            (300.0, 150.0, 600.0, 1050.0),
            track_generation=4,
            source_timestamp_ns=140 * NS_PER_MS,
            primary_observed=True,
        )

        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertEqual(mapped.point, (480.0, 258.0))
        self.assertIs(
            mapped.provenance,
            DirectHeadProvenance.MEASURED_PRIMARY,
        )
        self.assertTrue(mapped.body_derived)
        self.assertTrue(mapped.primary_observed)
        self.assertFalse(mapped.motion_corroboration_permitted)
        self.assertAlmostEqual(mapped.confidence, 0.64)

    def test_isolated_direct_jitter_is_rejected_by_rolling_median(self) -> None:
        normalized_points = (
            (0.600, 0.120),
            (0.608, 0.116),
            (0.594, 0.125),
            (0.603, 0.118),
            # One otherwise-valid but severe localization excursion.
            (0.900, 0.430),
        )
        raw_y = []
        for index, normalized in enumerate(normalized_points):
            point = (
                self.body[0] + normalized[0] * 200.0,
                self.body[1] + normalized[1] * 600.0,
            )
            raw_y.append(point[1])
            sample = self.anchor.observe_direct(
                point,
                self.body,
                track_generation=4,
                source_timestamp_ns=(100 + index * 10) * NS_PER_MS,
                confidence=0.80,
            )
            self.assertIsNotNone(sample)

        filtered = self.anchor.normalized_point
        self.assertIsNotNone(filtered)
        assert filtered is not None
        self.assertAlmostEqual(filtered[0], 0.603)
        self.assertAlmostEqual(filtered[1], 0.120)
        mapped = self.anchor.map_primary(
            self.body,
            track_generation=4,
            source_timestamp_ns=150 * NS_PER_MS,
            primary_observed=True,
        )
        assert mapped is not None
        self.assertAlmostEqual(mapped.point[1], 172.0)
        self.assertGreater(max(raw_y) - mapped.point[1], 180.0)

    def test_clustered_direct_misses_are_bridged_by_measured_primary_boxes(
        self,
    ) -> None:
        self.seed()
        points = []
        for age_ms in (25, 50, 75, 100, 125, 150, 175):
            translated = tuple(value + 2.0 * age_ms for value in self.body)
            sample = self.anchor.map_primary(
                translated,
                track_generation=4,
                source_timestamp_ns=(100 + age_ms) * NS_PER_MS,
                primary_observed=True,
            )
            self.assertIsNotNone(sample)
            assert sample is not None
            points.append(sample.point)
            self.assertEqual(
                sample.direct_source_timestamp_ns,
                100 * NS_PER_MS,
            )
            self.assertEqual(sample.identity_deadline_ns, 300 * NS_PER_MS)

        self.assertEqual(points[0], (270.0, 222.0))
        self.assertEqual(points[-1], (570.0, 522.0))
        self.assertEqual(
            self.anchor.last_direct_source_timestamp_ns,
            100 * NS_PER_MS,
        )

    def test_body_mapping_expires_at_two_hundred_ms_without_direct_renewal(
        self,
    ) -> None:
        self.assertEqual(DIRECT_HEAD_ANCHOR_MAX_AGE_SECONDS, 0.200)
        self.seed()
        before = self.anchor.map_primary(
            self.body,
            track_generation=4,
            source_timestamp_ns=299 * NS_PER_MS,
            primary_observed=True,
        )
        expired = self.anchor.map_primary(
            self.body,
            track_generation=4,
            source_timestamp_ns=300 * NS_PER_MS,
            primary_observed=True,
        )

        self.assertIsNotNone(before)
        assert before is not None
        self.assertAlmostEqual(before.confidence, 0.004)
        self.assertIsNone(expired)
        self.assertFalse(self.anchor.active)
        self.assertIsNone(self.anchor.identity_deadline_ns)

    def test_prediction_only_has_distinct_provenance_and_cannot_renew(self) -> None:
        self.seed()
        first = self.anchor.map_primary(
            (110.0, 105.0, 310.0, 705.0),
            track_generation=4,
            source_timestamp_ns=150 * NS_PER_MS,
            primary_observed=False,
        )
        second = self.anchor.map_primary(
            (120.0, 110.0, 320.0, 710.0),
            track_generation=4,
            source_timestamp_ns=250 * NS_PER_MS,
            primary_observed=False,
        )
        expired = self.anchor.map_primary(
            (130.0, 115.0, 330.0, 715.0),
            track_generation=4,
            source_timestamp_ns=300 * NS_PER_MS,
            primary_observed=False,
        )

        for sample in (first, second):
            self.assertIsNotNone(sample)
            assert sample is not None
            self.assertIs(
                sample.provenance,
                DirectHeadProvenance.PREDICTED_PRIMARY,
            )
            self.assertFalse(sample.primary_observed)
            self.assertFalse(sample.motion_corroboration_permitted)
            self.assertEqual(sample.identity_deadline_ns, 300 * NS_PER_MS)
            self.assertEqual(
                sample.direct_source_timestamp_ns,
                100 * NS_PER_MS,
            )
        self.assertIsNone(expired)

    def test_crossing_generation_cannot_inherit_previous_head_anchor(self) -> None:
        self.seed(generation=4)
        crossed = self.anchor.map_primary(
            (500.0, 100.0, 700.0, 700.0),
            track_generation=5,
            source_timestamp_ns=140 * NS_PER_MS,
            primary_observed=True,
        )
        stale_direct = self.anchor.observe_direct(
            self.head,
            self.body,
            track_generation=4,
            source_timestamp_ns=150 * NS_PER_MS,
            confidence=0.90,
        )

        self.assertIsNone(crossed)
        self.assertIsNone(stale_direct)
        self.assertEqual(self.anchor.track_generation, 5)
        self.assertFalse(self.anchor.active)
        replacement = self.seed(
            point=(620.0, 172.0),
            body=(500.0, 100.0, 700.0, 700.0),
            generation=5,
            timestamp_ns=160 * NS_PER_MS,
        )
        self.assertIsNotNone(replacement)
        assert replacement is not None
        self.assertEqual(replacement.point, (620.0, 172.0))

    def test_newer_direct_generation_clears_filter_before_reseeding(self) -> None:
        for index, x in enumerate((210.0, 220.0, 230.0)):
            self.seed(
                point=(x, 172.0),
                generation=4,
                timestamp_ns=(100 + index * 10) * NS_PER_MS,
            )
        replacement = self.seed(
            point=(520.0, 172.0),
            body=(400.0, 100.0, 600.0, 700.0),
            generation=5,
            timestamp_ns=140 * NS_PER_MS,
        )

        self.assertIsNotNone(replacement)
        assert replacement is not None
        self.assertEqual(replacement.point, (520.0, 172.0))
        self.assertEqual(self.anchor.normalized_point, (0.60, 0.12))

    def test_hard_reset_and_explicit_generation_advance_clear_every_coordinate(
        self,
    ) -> None:
        self.seed()
        self.assertTrue(self.anchor.advance_generation(5))
        self.assertEqual(self.anchor.track_generation, 5)
        self.assertFalse(self.anchor.active)
        self.assertIsNone(self.anchor.normalized_point)
        self.assertFalse(self.anchor.advance_generation(4))
        self.assertEqual(self.anchor.track_generation, 5)

        self.seed(
            point=(520.0, 172.0),
            body=(400.0, 100.0, 600.0, 700.0),
            generation=5,
            timestamp_ns=200 * NS_PER_MS,
        )
        self.anchor.reset()

        self.assertIsNone(self.anchor.track_generation)
        self.assertIsNone(self.anchor.last_direct_source_timestamp_ns)
        self.assertIsNone(self.anchor.identity_deadline_ns)
        self.assertIsNone(
            self.anchor.map_primary(
                self.body,
                track_generation=5,
                source_timestamp_ns=210 * NS_PER_MS,
                primary_observed=True,
            )
        )

    def test_stale_direct_and_invalid_geometry_fail_without_corrupting_anchor(
        self,
    ) -> None:
        self.seed()
        stale = self.anchor.observe_direct(
            (250.0, 180.0),
            self.body,
            track_generation=4,
            source_timestamp_ns=90 * NS_PER_MS,
            confidence=0.90,
        )
        outside = self.anchor.observe_direct(
            (500.0, 50.0),
            self.body,
            track_generation=4,
            source_timestamp_ns=110 * NS_PER_MS,
            confidence=0.90,
        )

        self.assertIsNone(stale)
        self.assertIsNone(outside)
        self.assertEqual(
            self.anchor.last_direct_source_timestamp_ns,
            100 * NS_PER_MS,
        )
        self.assertEqual(self.anchor.normalized_point, (0.60, 0.12))

    def test_direct_center_inside_existing_head_region_margin_is_retained(self) -> None:
        sample = self.seed(point=(88.0, 88.0))

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.point, (88.0, 88.0))
        normalized = self.anchor.normalized_point
        assert normalized is not None
        self.assertAlmostEqual(normalized[0], -0.06)
        self.assertAlmostEqual(normalized[1], -0.02)

    def test_constructor_and_public_inputs_reject_ambiguous_values(self) -> None:
        for kwargs in (
            {"max_direct_age_seconds": math.nan},
            {"max_direct_age_seconds": 0.0},
            {"filter_samples": 2},
            {"filter_samples": 4},
            {"filter_samples": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DirectHeadAnchor(**kwargs)
        with self.assertRaises(ValueError):
            self.seed(generation=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.seed(timestamp_ns=-1)
        with self.assertRaises(ValueError):
            self.seed(confidence=math.inf)
        self.seed()
        with self.assertRaises(TypeError):
            self.anchor.map_primary(
                self.body,
                track_generation=4,
                source_timestamp_ns=110 * NS_PER_MS,
                primary_observed=1,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
