from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import scripts.compare_fort_runtime_evaluations as comparator_module
from scripts.compare_fort_runtime_evaluations import ComparisonError, compare_reports


BUCKETS = [
    "ultra_far_le_32px",
    "far_33_to_64px",
    "medium_65_to_96px",
    "near_gt_96px",
]


def _counts(target: int = 0, tp: int = 0, fp: int = 0):
    return target, {"true_positives": tp, "false_positives": fp}


def _report(*, candidate: bool, image_count: int = 2) -> dict[str, object]:
    specifications = [
        {
            "ultra_far_le_32px": _counts(),
            "far_33_to_64px": _counts(1, 1 if candidate else 0, 0),
            "medium_65_to_96px": _counts(1, 1, 0),
            "near_gt_96px": _counts(1, 1, 0),
        }
        for _ in range(image_count)
    ]
    records = []
    for index, specification in enumerate(specifications):
        point = {bucket: specification[bucket][1] for bucket in BUCKETS}
        records.append(
            {
                "member_id": f"{index + 1:064x}",
                "source_group_id": f"{index + 10_000:064x}",
                "targets": {bucket: specification[bucket][0] for bucket in BUCKETS},
                "operating_points": {
                    "0.25": deepcopy(point),
                    "0.45": deepcopy(point),
                },
            }
        )
    from scripts.compare_fort_runtime_evaluations import _canonical_hash

    paired = {
        "schema_version": 1,
        "bucket_order": BUCKETS,
        "member_count": len(records),
        "member_sequence_sha256": _canonical_hash(
            [record["member_id"] for record in records]
        ),
        "source_group_count": len(records),
        "source_group_sequence_sha256": _canonical_hash(
            [record["source_group_id"] for record in records]
        ),
        "confidence_thresholds": [0.25, 0.45],
        "split_content_sha256": "a" * 64,
        "records": records,
    }
    bucket_points = {}
    for bucket in BUCKETS:
        bucket_points[bucket] = {
            "ground_truth_total": sum(record["targets"][bucket] for record in records),
            "detected_true_positives": sum(
                record["operating_points"]["0.25"][bucket]["true_positives"]
                for record in records
            ),
            "false_positives": sum(
                record["operating_points"]["0.25"][bucket]["false_positives"]
                for record in records
            ),
        }
    aggregate = {
        "ground_truth_total": sum(item["ground_truth_total"] for item in bucket_points.values()),
        "detected_true_positives": sum(
            item["detected_true_positives"] for item in bucket_points.values()
        ),
        "false_positives": sum(item["false_positives"] for item in bucket_points.values()),
    }
    metrics = {
        "paired_image_operating_points": paired,
        "aggregate_detection": {
            "operating_points": {
                "0.25": deepcopy(aggregate),
                "0.45": deepcopy(aggregate),
            }
        },
        "size_bucket_detection": {
            "operating_points": {
                "0.25": deepcopy(bucket_points),
                "0.45": deepcopy(bucket_points),
            }
        },
    }
    return {
        "schema_version": 4,
        "evaluator": {
            "sha256": "f" * 64,
            "pipeline_source_sha256": {"preprocess": "1" * 64},
        },
        "configuration": {
            "split": "val",
            "selection_role": "development_validation",
            "matching_iou_threshold": 0.5,
            "minimum_prediction_confidence": 0.001,
            "reported_confidence_thresholds": [0.25, 0.45],
            "runtime_nms_iou_threshold": 0.45,
        },
        "dataset": {
            "content_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
        },
        "model_artifact": {
            "content_sha256": ("d" if candidate else "e") * 64,
        },
        "qualification": {
            "status": "development_evidence_only",
            "independent_holdout_required": True,
        },
        "metrics": {"val": metrics},
    }


class CompareFortRuntimeEvaluationsTests(unittest.TestCase):
    def _write(self, root: Path, name: str, value: object) -> Path:
        path = root / name
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    def test_paired_gain_advances_development_candidate_but_never_qualifies_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._write(root, "baseline.json", _report(candidate=False))
            candidate = self._write(root, "candidate.json", _report(candidate=True))
            output = root / "comparison"
            record = compare_reports(
                baseline=baseline,
                candidate=candidate,
                output=output,
                bootstrap_samples=100,
            )
            far = record["comparison"]["buckets"]["far_33_to_64px"]
            self.assertEqual(far["baseline"]["true_positives"], 0)
            self.assertEqual(far["candidate"]["true_positives"], 2)
            self.assertEqual(far["delta_candidate_minus_baseline"]["recall"], 1.0)
            self.assertFalse(record["development_advancement_policy"]["passed"])
            self.assertFalse(
                record["development_advancement_policy"]["checks"]
                ["far_decision_bucket_has_at_least_30_targets"]
            )
            self.assertFalse(record["release_qualification"]["qualified"])
            self.assertTrue(record["release_qualification"]["independent_holdout_required"])
            self.assertEqual(
                json.loads((output / "comparison.json").read_text(encoding="utf-8")),
                record,
            )

    def test_sufficient_consistent_far_gain_can_pass_only_development_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._write(
                root, "baseline.json", _report(candidate=False, image_count=30)
            )
            candidate = self._write(
                root, "candidate.json", _report(candidate=True, image_count=30)
            )
            record = compare_reports(
                baseline=baseline,
                candidate=candidate,
                output=root / "comparison",
                bootstrap_samples=comparator_module.BOOTSTRAP_SAMPLES,
            )
            self.assertTrue(record["development_advancement_policy"]["passed"])
            self.assertTrue(all(record["development_advancement_policy"]["checks"].values()))
            self.assertFalse(record["release_qualification"]["qualified"])

    def test_nonstandard_bootstrap_and_concentrated_far_targets_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_value = _report(candidate=False, image_count=30)
            candidate_value = _report(candidate=True, image_count=30)
            baseline = self._write(root, "baseline.json", baseline_value)
            candidate = self._write(root, "candidate.json", candidate_value)
            diagnostic = compare_reports(
                baseline=baseline,
                candidate=candidate,
                output=root / "diagnostic",
                bootstrap_samples=1,
            )
            self.assertFalse(diagnostic["development_advancement_policy"]["passed"])
            self.assertFalse(
                diagnostic["development_advancement_policy"]["checks"]
                ["bootstrap_uses_exact_2000_samples"]
            )

            alternate_confidence = compare_reports(
                baseline=baseline,
                candidate=candidate,
                output=root / "alternate-confidence",
                confidence=0.45,
                bootstrap_samples=comparator_module.BOOTSTRAP_SAMPLES,
            )
            self.assertFalse(
                alternate_confidence["development_advancement_policy"]["passed"]
            )
            self.assertFalse(
                alternate_confidence["development_advancement_policy"]["checks"]
                ["confidence_is_release_default_0_25"]
            )

            # Preserve 30 targets but concentrate them in one member. Aggregate
            # counts are updated so only the distinct-image policy rejects it.
            for value in (baseline_value, candidate_value):
                records = value["metrics"]["val"]["paired_image_operating_points"][
                    "records"
                ]
                for index, record in enumerate(records):
                    record["targets"]["far_33_to_64px"] = 30 if index == 0 else 0
                    for point in record["operating_points"].values():
                        point["far_33_to_64px"]["true_positives"] = (
                            30 if index == 0 and value is candidate_value else 0
                        )
                for point_key in ("0.25", "0.45"):
                    bucket = value["metrics"]["val"]["size_bucket_detection"][
                        "operating_points"
                    ][point_key]["far_33_to_64px"]
                    bucket["ground_truth_total"] = 30
                    bucket["detected_true_positives"] = (
                        30 if value is candidate_value else 0
                    )
                    aggregate = value["metrics"]["val"]["aggregate_detection"][
                        "operating_points"
                    ][point_key]
                    aggregate["ground_truth_total"] = 90
                    aggregate["detected_true_positives"] = (
                        90 if value is candidate_value else 60
                    )
            concentrated = compare_reports(
                baseline=self._write(root, "baseline-concentrated.json", baseline_value),
                candidate=self._write(root, "candidate-concentrated.json", candidate_value),
                output=root / "concentrated",
                bootstrap_samples=comparator_module.BOOTSTRAP_SAMPLES,
            )
            self.assertFalse(concentrated["development_advancement_policy"]["passed"])
            self.assertFalse(
                concentrated["development_advancement_policy"]["checks"]
                ["far_decision_bucket_spans_at_least_30_images"]
            )

    def test_source_group_cluster_bootstrap_and_minimum_group_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_value = _report(candidate=False, image_count=30)
            candidate_value = _report(candidate=True, image_count=30)
            # Collapse the 30 otherwise qualifying far-bearing images into one
            # source cluster without changing any image-level evidence.
            for value in (baseline_value, candidate_value):
                paired = value["metrics"]["val"]["paired_image_operating_points"]
                for record in paired["records"]:
                    record["source_group_id"] = "a" * 64
                paired["source_group_count"] = 1
                from scripts.compare_fort_runtime_evaluations import _canonical_hash

                paired["source_group_sequence_sha256"] = _canonical_hash(
                    [record["source_group_id"] for record in paired["records"]]
                )
            result = compare_reports(
                baseline=self._write(root, "baseline.json", baseline_value),
                candidate=self._write(root, "candidate.json", candidate_value),
                output=root / "comparison",
                bootstrap_samples=comparator_module.BOOTSTRAP_SAMPLES,
            )
            policy = result["development_advancement_policy"]
            self.assertFalse(policy["passed"])
            self.assertEqual(policy["far_target_bearing_source_groups"], 1)
            self.assertFalse(
                policy["checks"][
                    "far_decision_bucket_spans_at_least_15_source_groups"
                ]
            )
            cluster = result["comparison"]["buckets"]["far_33_to_64px"][
                "paired_source_group_bootstrap_95_ci"
            ]
            self.assertEqual(cluster["source_group_count"], 1)
            self.assertEqual(cluster["samples_requested"], 2000)

    def test_exact_bucket_and_operating_point_schema_is_required(self) -> None:
        for scenario in ("bucket", "operating-point"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                candidate_value = _report(candidate=True)
                paired = candidate_value["metrics"]["val"][
                    "paired_image_operating_points"
                ]
                if scenario == "bucket":
                    paired["bucket_order"] = paired["bucket_order"][:-1]
                else:
                    paired["records"][0]["operating_points"].pop("0.45")
                with self.assertRaises(ComparisonError):
                    compare_reports(
                        baseline=self._write(
                            root, "baseline.json", _report(candidate=False)
                        ),
                        candidate=self._write(root, "candidate.json", candidate_value),
                        output=root / "comparison",
                        bootstrap_samples=20,
                    )

    def test_bootstrap_seed_ignores_irrelevant_report_formatting_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_value = _report(candidate=False)
            candidate_value = _report(candidate=True)
            first = compare_reports(
                baseline=self._write(root, "baseline-a.json", baseline_value),
                candidate=self._write(root, "candidate-a.json", candidate_value),
                output=root / "first",
                bootstrap_samples=20,
            )
            baseline_value["irrelevant_note"] = "formatting and metadata do not choose RNG"
            candidate_value["irrelevant_note"] = "another irrelevant value"
            baseline_b = root / "baseline-b.json"
            candidate_b = root / "candidate-b.json"
            baseline_b.write_text(json.dumps(baseline_value, indent=4), encoding="utf-8")
            candidate_b.write_text(json.dumps(candidate_value, separators=(",", ":")), encoding="utf-8")
            second = compare_reports(
                baseline=baseline_b,
                candidate=candidate_b,
                output=root / "second",
                bootstrap_samples=20,
            )
            first_seed = first["comparison"]["buckets"]["far_33_to_64px"][
                "paired_image_bootstrap_95_ci"
            ]["derived_seed"]
            second_seed = second["comparison"]["buckets"]["far_33_to_64px"][
                "paired_image_bootstrap_95_ci"
            ]["derived_seed"]
            self.assertEqual(first_seed, second_seed)

    def test_mismatched_members_targets_or_summaries_fail_without_output(self) -> None:
        for scenario in ("member", "target", "summary"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                baseline_value = _report(candidate=False)
                candidate_value = _report(candidate=True)
                candidate_metrics = candidate_value["metrics"]["val"]
                paired = candidate_metrics["paired_image_operating_points"]
                if scenario == "member":
                    paired["records"][0]["member_id"] = "f" * 64
                    from scripts.compare_fort_runtime_evaluations import _canonical_hash

                    paired["member_sequence_sha256"] = _canonical_hash(
                        [record["member_id"] for record in paired["records"]]
                    )
                elif scenario == "target":
                    paired["records"][0]["targets"]["far_33_to_64px"] = 2
                    candidate_metrics["size_bucket_detection"]["operating_points"]["0.25"][
                        "far_33_to_64px"
                    ]["ground_truth_total"] = 3
                    candidate_metrics["aggregate_detection"]["operating_points"]["0.25"][
                        "ground_truth_total"
                    ] = 7
                else:
                    candidate_metrics["size_bucket_detection"]["operating_points"]["0.25"][
                        "far_33_to_64px"
                    ]["detected_true_positives"] = 1
                baseline = self._write(root, "baseline.json", baseline_value)
                candidate = self._write(root, "candidate.json", candidate_value)
                output = root / "comparison"
                with self.assertRaises(ComparisonError):
                    compare_reports(
                        baseline=baseline,
                        candidate=candidate,
                        output=output,
                        bootstrap_samples=20,
                    )
                self.assertFalse(output.exists())

    def test_same_report_may_compare_primary_against_configured_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = _report(candidate=True)
            metrics = value["metrics"]["val"]
            primary = deepcopy(metrics)
            for record in primary["paired_image_operating_points"]["records"]:
                record["operating_points"]["0.25"]["far_33_to_64px"][
                    "true_positives"
                ] = 0
            primary["size_bucket_detection"]["operating_points"]["0.25"][
                "far_33_to_64px"
            ]["detected_true_positives"] = 0
            primary["aggregate_detection"]["operating_points"]["0.25"][
                "detected_true_positives"
            ] = 4
            metrics["primary_full_frame_reference"] = primary
            report = self._write(root, "detail.json", value)
            output = root / "comparison"
            result = compare_reports(
                baseline=report,
                candidate=report,
                output=output,
                baseline_pipeline="primary",
                candidate_pipeline="configured",
                bootstrap_samples=50,
            )
            self.assertEqual(
                result["comparison"]["buckets"]["far_33_to_64px"]
                ["delta_candidate_minus_baseline"]["recall"],
                1.0,
            )

    def test_test_split_duplicate_json_existing_output_and_same_report_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_value = _report(candidate=False)
            candidate_value = _report(candidate=True)
            candidate_value["configuration"]["split"] = "test"
            baseline = self._write(root, "baseline.json", baseline_value)
            candidate = self._write(root, "candidate.json", candidate_value)
            with self.assertRaisesRegex(ComparisonError, "validation reports only"):
                compare_reports(
                    baseline=baseline,
                    candidate=candidate,
                    output=root / "test-output",
                    bootstrap_samples=20,
                )

            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema_version":4,"schema_version":4}', encoding="utf-8")
            with self.assertRaisesRegex(ComparisonError, "duplicate JSON key"):
                compare_reports(
                    baseline=duplicate,
                    candidate=baseline,
                    output=root / "duplicate-output",
                    bootstrap_samples=20,
                )

            existing = root / "existing"
            existing.mkdir()
            valid_candidate = self._write(root, "valid-candidate.json", _report(candidate=True))
            with self.assertRaisesRegex(ComparisonError, "output already exists"):
                compare_reports(
                    baseline=baseline,
                    candidate=valid_candidate,
                    output=existing,
                    bootstrap_samples=20,
                )
            with self.assertRaisesRegex(ComparisonError, "different reports or different"):
                compare_reports(
                    baseline=baseline,
                    candidate=baseline,
                    output=root / "same",
                    bootstrap_samples=20,
                )

    def test_comparator_source_change_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._write(root, "baseline.json", _report(candidate=False))
            candidate = self._write(root, "candidate.json", _report(candidate=True))
            output = root / "comparison"
            source = Path(comparator_module.__file__).resolve()
            original_sha256_file = comparator_module._sha256_file
            source_reads = 0

            def changed_source(path: Path) -> str:
                nonlocal source_reads
                if Path(path).resolve() == source:
                    source_reads += 1
                    return ("a" if source_reads == 1 else "b") * 64
                return original_sha256_file(path)

            with mock.patch.object(
                comparator_module, "_sha256_file", side_effect=changed_source
            ):
                with self.assertRaisesRegex(ComparisonError, "source or input report"):
                    compare_reports(
                        baseline=baseline,
                        candidate=candidate,
                        output=output,
                        bootstrap_samples=20,
                    )
            self.assertFalse(output.exists())

    def test_report_snapshot_hashes_the_same_bytes_that_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            original = json.dumps(_report(candidate=False), sort_keys=True).encode()
            replacement = json.dumps(_report(candidate=True), sort_keys=True).encode()
            report.write_bytes(original)
            original_read_bytes = Path.read_bytes

            def mutate_after_snapshot(path: Path) -> bytes:
                if path.resolve() == report.resolve():
                    report.write_bytes(replacement)
                    return original
                return original_read_bytes(path)

            with mock.patch.object(Path, "read_bytes", new=mutate_after_snapshot):
                _path, parsed, digest = comparator_module._read_report(
                    report, "snapshot report"
                )
            self.assertEqual(parsed["model_artifact"]["content_sha256"], "e" * 64)
            self.assertEqual(digest, sha256(original).hexdigest())
            self.assertNotEqual(digest, sha256(replacement).hexdigest())


if __name__ == "__main__":
    unittest.main()
