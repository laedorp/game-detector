from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from detection.types import Detection
from scripts import evaluate_independent_holdout_runtime as evaluator
from scripts import prepare_independent_player_holdout as holdout_module
from scripts.evaluate_fort_model import bucket_image_evidence
from scripts.evaluate_independent_holdout_runtime import (
    IndependentRuntimeEvaluationError,
    _decision_result,
    _inventory_counts,
    _source_group_uncertainty,
    _validate_plan,
    _validate_sealed_manifest,
    build_evaluation_plan,
    evaluate_independent_holdout,
    publish_independent_evidence_bundle,
    verify_independent_evidence,
    verify_independent_evidence_bundle,
    verify_independent_evidence_receipt,
    write_independent_evidence_receipt,
    write_evaluation_plan,
)
from scripts.evaluate_fort_runtime_model import _metric_record
from scripts.prepare_independent_player_holdout import (
    GATING_COUNT_KEYS,
    PINNED_RELEASE_MINIMUMS,
    POOL_DEVELOPMENT,
    POOL_SEALED,
)
from tests.independent_holdout_environment_fixture import (
    valid_windows_directml_dependency_manifest,
)
from tests.independent_holdout_hardware_fixture import (
    valid_rx6950_holdout_hardware_identity,
)
from utils.release_model_contract import (
    TOURNAMENT_COMPARISON_NAMES,
    TOURNAMENT_SEALED_INPUT_ROLES,
    canonical_hash,
)


def _manifest(*, pool: str = POOL_SEALED) -> dict[str, object]:
    counts = PINNED_RELEASE_MINIMUMS.as_dict()
    return {
        "kind": "independent-player-detection-holdout",
        "pool": pool,
        "package_id": "sealed-final-a",
        "manifest_content_sha256": "1" * 64,
        "counts": counts,
        "release_gates": {
            "pinned_minimums": counts,
            "gating_count_keys": list(GATING_COUNT_KEYS),
            "meets_pinned_release_gates": True,
            "target_le_32_is_descriptive": True,
        },
        "sessions": [
            {"session_id": f"capture-session-{index:02d}"}
            for index in range(15)
        ],
        "source_group_inventory": {
            "definition": "distinct normalized COCO image session_id values",
            "overall_capture_sessions": 15,
            "target_bearing_capture_sessions": {
                "target_le_32": 15,
                "target_33_64": 15,
                "target_65_96": 15,
                "target_gt_96": 15,
            },
            "reviewed_negative_capture_sessions": 15,
        },
        "images": [{}],
        "annotations": {"boxes": sum(counts[key] for key in counts if key != "reviewed_negatives")},
        "redistribution_permitted_for_all_sessions": False,
    }


def _binding(root: Path, *, shape: tuple[int, int] = (384, 640)) -> dict[str, object]:
    model_sha = "2" * 64
    return {
        "pointer_path": root / "models" / "RELEASE-DEFAULT.json",
        "pointer_sha256": "3" * 64,
        "pointer_content_sha256": "4" * 64,
        "input_shape_nchw": [1, 3, *shape],
        "output_head": "end2end",
        "candidate_content_sha256": "5" * 64,
        "candidate_manifest_sha256": "6" * 64,
        "checkpoint_sha256": "7" * 64,
        "dataset_manifest_sha256": "8" * 64,
        "dataset_content_sha256": "9" * 64,
        "adoption_path": root / "ADOPTION.json",
        "adoption_sha256": "a" * 64,
        "adoption_content_sha256": "b" * 64,
        "adoption_evidence_replay_sha256": "e" * 64,
        "tournament_selection_path": root / "MODEL-TOURNAMENT-SELECTION.json",
        "tournament_selection_sha256": "c" * 64,
        "tournament_selection_content_sha256": "0" * 64,
        "tournament_evidence": [
            {
                "role": "tournament_selection_manifest",
                "relative_path": "models/release-defaults/candidate/MODEL-TOURNAMENT-SELECTION.json",
                "bytes": 10,
                "sha256": "c" * 64,
            }
        ],
        "tournament_evidence_files": [
            (root / "MODEL-TOURNAMENT-SELECTION.json", "c" * 64)
        ],
        "candidate_provenance_evidence": [
            {
                "role": "candidate_receipt",
                "relative_path": "models/release-defaults/candidate/CANDIDATE-RECEIPT.json",
                "bytes": 10,
                "sha256": "f" * 64,
            }
        ],
        "candidate_provenance_files": [
            (root / "CANDIDATE-RECEIPT.json", "f" * 64)
        ],
        "candidate_evaluation_sha256": "1" * 64,
        "winner_slot": "s_end2end",
        "model_path": root / "candidate.onnx",
        "model_artifacts": [
            {
                "role": "onnx",
                "name": "candidate.onnx",
                "relative_path": "models/release-defaults/candidate/candidate.onnx",
                "bytes": 123,
                "sha256": model_sha,
            }
        ],
        "model_content_sha256": canonical_hash(
            [{"name": "candidate.onnx", "sha256": model_sha}]
        ),
        "labels_path": root / "labels.txt",
        "labels": {
            "relative_path": "models/release-defaults/candidate/labels.txt",
            "bytes": 7,
            "sha256": "d" * 64,
        },
        "selected_pipeline": "configured",
        "selected_backend": "onnxruntime",
        "detail_crop_size_source_pixels": 768,
        "exporter_sha256": "e" * 64,
        "adoption_source_sha256": "f" * 64,
    }


def _rule() -> dict[str, object]:
    return dict(evaluator.CANONICAL_RELEASE_DECISION_RULE)


def _dependency_manifest(root: Path) -> Path:
    path = root / "DEPENDENCY-MANIFEST.json"
    path.write_text(
        json.dumps(
            valid_windows_directml_dependency_manifest(evaluator.PROJECT_ROOT),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _hardware_identity(root: Path, *, adapter_index: int = 0) -> Path:
    path = root / "HOLDOUT-HARDWARE-IDENTITY.json"
    path.write_bytes(
        evaluator.canonical_json_bytes(
            valid_rx6950_holdout_hardware_identity(
                adapter_index=adapter_index
            )
        )
    )
    return path


_ORIGINAL_VALIDATE_CURRENT_RELEASE_ENVIRONMENT = (
    evaluator._validate_current_release_environment
)


class IndependentHoldoutRuntimeEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        # The unit suite runs on Linux; production re-hashes the executing
        # Windows release environment before any sealed-member access.
        patcher = mock.patch.object(
            evaluator, "_validate_current_release_environment"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_runtime_manifest_cannot_attest_a_different_interpreter(self) -> None:
        manifest = valid_windows_directml_dependency_manifest(evaluator.PROJECT_ROOT)
        with self.assertRaisesRegex(
            IndependentRuntimeEvaluationError, "executing Python differs"
        ):
            _ORIGINAL_VALIDATE_CURRENT_RELEASE_ENVIRONMENT(
                manifest, project_root=evaluator.PROJECT_ROOT
            )

    def test_holdout_package_rejects_final_and_ancestor_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            package = real / "sealed"
            package.mkdir(parents=True)
            final_link = root / "sealed-link"
            ancestor_link = root / "ancestor-link"
            try:
                final_link.symlink_to(package, target_is_directory=True)
                ancestor_link.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks are not available")
            for candidate in (final_link, ancestor_link / "sealed"):
                with self.subTest(candidate=candidate), self.assertRaisesRegex(
                    IndependentRuntimeEvaluationError, "symlink or junction"
                ):
                    evaluator._package_root(candidate)

    def test_public_tournament_evidence_rejects_embedded_local_paths(self) -> None:
        disclosures = (
            "loaded_from=/tmp/private/model.onnx",
            r"loaded_from=C:\Users\private\model.onnx",
            r"loaded_from=\\workstation\share\model.onnx",
            "loaded_from=file:///etc/passwd",
            "note: /opt/private/model.onnx",
            "loaded_from@/tmp/private/model.onnx",
        )
        for disclosure in disclosures:
            with self.subTest(disclosure=disclosure), self.assertRaisesRegex(
                IndependentRuntimeEvaluationError,
                "private or absolute path",
            ):
                evaluator._reject_private_path_strings(
                    {"debug_origin": disclosure}, "sealed runtime report"
                )
        with self.assertRaisesRegex(
            IndependentRuntimeEvaluationError, "unsafe field name"
        ):
            evaluator._reject_private_path_strings(
                {"/tmp/private/model.onnx": "value"},
                "sealed runtime report",
            )
        redacted_summary = evaluator._redacted_runtime_summary(
            {"loaded_from": "/tmp/private/model.onnx"}
        )
        with self.assertRaisesRegex(
            IndependentRuntimeEvaluationError, "private or absolute path"
        ):
            evaluator._reject_private_path_strings(
                {"runtime": {"summary": redacted_summary}},
                "final public evidence",
            )

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "training-results.csv"
            for disclosure in disclosures:
                with self.subTest(text_disclosure=disclosure):
                    source.write_text(
                        f"epoch,origin\n1,{disclosure}\n", encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        IndependentRuntimeEvaluationError,
                        "private or absolute path",
                    ):
                        evaluator._assert_public_text_safe(
                            source, "sealed training results"
                        )
            source.write_text("epoch,metric\n1,0.5\n", encoding="utf-8")
            evaluator._assert_public_text_safe(source, "sealed training results")

    def test_binding_consumes_current_sealed_tournament_adoption_contract(self) -> None:
        # Reuse the repository's complete adoption fixture so this adapter cannot
        # silently drift back to the retired single-comparison pointer schema.
        from tests.test_release_model_adoption import ReleaseModelAdoptionTests

        fixture = ReleaseModelAdoptionTests(
            methodName="test_adopts_exact_configured_winner_and_both_portable_formats"
        )
        fixture.setUp()
        try:
            fixture._adopt()
            binding = evaluator._release_candidate_binding(
                fixture.root, "onnxruntime"
            )
        finally:
            fixture.tearDown()

        self.assertEqual(binding["winner_slot"], "s_traditional")
        self.assertEqual(binding["output_head"], "traditional")
        self.assertEqual(binding["selected_pipeline"], "configured")
        self.assertEqual(binding["detail_crop_size_source_pixels"], 768)
        self.assertEqual(
            {record["role"] for record in binding["tournament_evidence"]},
            {
                "tournament_selection_manifest",
                *{
                    f"tournament_comparison_{name}"
                    for name in TOURNAMENT_COMPARISON_NAMES
                },
                *TOURNAMENT_SEALED_INPUT_ROLES,
            },
        )
        self.assertEqual(len(binding["candidate_provenance_evidence"]), 4)

    def test_binding_and_plan_preserve_a_primary_only_tournament_winner(self) -> None:
        from tests.test_release_model_adoption import ReleaseModelAdoptionTests

        fixture = ReleaseModelAdoptionTests(
            methodName="test_primary_tournament_winner_maps_to_disabled_detail_workload"
        )
        fixture.setUp()
        try:
            candidate, evaluation, selection = fixture._selection(
                fixture.root / "primary-final-evidence",
                challenger_advances=False,
            )
            fixture._adopt(
                candidate=candidate,
                candidate_evaluation=evaluation,
                tournament_selection=selection,
            )
            binding = evaluator._release_candidate_binding(
                fixture.root, "onnxruntime"
            )
        finally:
            fixture.tearDown()

        self.assertEqual(binding["selected_pipeline"], "primary")
        self.assertEqual(binding["detail_crop_size_source_pixels"], 0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_binding = _binding(root)
            primary_binding["selected_pipeline"] = "primary"
            primary_binding["detail_crop_size_source_pixels"] = 0
            with mock.patch.object(
                evaluator,
                "_load_package_manifest",
                return_value=(
                    _manifest(),
                    root / "HOLDOUT-MANIFEST.json",
                    b"manifest",
                ),
            ):
                plan = build_evaluation_plan(
                    package=root / "sealed",
                    dependency_manifest=_dependency_manifest(root),
                    hardware_identity=_hardware_identity(root),
                    project_root=evaluator.PROJECT_ROOT,
                    device="DML:0",
                    expected_provider="DmlExecutionProvider",
                    detail_crop_size=0,
                    decision_rule=_rule(),
                    candidate_binding_loader=lambda _root, _backend: primary_binding,
                )
            self.assertEqual(plan["candidate"]["selected_pipeline"], "primary")
            self.assertEqual(plan["runtime"]["detail_crop_size_source_pixels"], 0)
            _validate_plan(plan)

    def test_binding_rejects_a_changed_copied_sealed_tournament_member(self) -> None:
        from tests.test_release_model_adoption import ReleaseModelAdoptionTests

        fixture = ReleaseModelAdoptionTests(
            methodName="test_adopts_exact_configured_winner_and_both_portable_formats"
        )
        fixture.setUp()
        try:
            fixture._adopt()
            pointer_path = fixture.root / "models" / "RELEASE-DEFAULT.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            member_record = pointer["artifacts"][
                "tournament_runtime_report_n_end2end"
            ]
            member = fixture.root / member_record["path"]
            member.write_bytes(member.read_bytes() + b"\n")

            with self.assertRaisesRegex(
                IndependentRuntimeEvaluationError,
                "release-default contract is invalid",
            ):
                evaluator._release_candidate_binding(fixture.root, "onnxruntime")
        finally:
            fixture.tearDown()

    def test_plan_binds_dynamic_adopted_rectangular_candidate_without_opening_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = _binding(root)
            with mock.patch.object(
                evaluator, "_load_package_manifest", return_value=(_manifest(), root / "HOLDOUT-MANIFEST.json", b"manifest")
            ), mock.patch.object(
                evaluator,
                "verify_holdout",
                side_effect=AssertionError("plan must not open sealed members"),
            ):
                plan = build_evaluation_plan(
                    package=root / "sealed",
                    dependency_manifest=_dependency_manifest(root),
                    hardware_identity=_hardware_identity(root),
                    project_root=evaluator.PROJECT_ROOT,
                    device="DML:0",
                    expected_provider="DmlExecutionProvider",
                    detail_crop_size=768,
                    decision_rule=_rule(),
                    candidate_binding_loader=lambda _root, _backend: binding,
                )

            self.assertEqual(plan["candidate"]["input_shape_nchw"], [1, 3, 384, 640])
            self.assertEqual(plan["candidate"]["output_head"], "end2end")
            self.assertEqual(plan["runtime"]["inference_size"], "384x640")
            self.assertEqual(plan["runtime"]["output_format"], "end2end")
            self.assertEqual(plan["runtime"]["detail_crop_size_source_pixels"], 768)
            self.assertFalse(plan["scope"]["grouped_v9_development_data_permitted"])
            self.assertFalse(plan["scope"]["ultra_far_le_32_release_gate"])
            self.assertEqual(
                plan["plan_content_sha256"], evaluator._plan_content_hash(plan)
            )

    def test_plan_rejects_development_underfilled_and_square_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for manifest, binding, pattern in (
                (_manifest(pool=POOL_DEVELOPMENT), _binding(root), "sealed independent"),
                (_manifest(), _binding(root, shape=(640, 640)), "rectangular"),
            ):
                with self.subTest(pattern=pattern), mock.patch.object(
                    evaluator,
                    "_load_package_manifest",
                    return_value=(manifest, root / "HOLDOUT-MANIFEST.json", b"manifest"),
                ), self.assertRaisesRegex(IndependentRuntimeEvaluationError, pattern):
                    build_evaluation_plan(
                        package=root / "sealed",
                        dependency_manifest=_dependency_manifest(root),
                        hardware_identity=_hardware_identity(root),
                        project_root=evaluator.PROJECT_ROOT,
                        device="DML:0",
                        expected_provider="DmlExecutionProvider",
                        detail_crop_size=768,
                        decision_rule=_rule(),
                        candidate_binding_loader=lambda _root, _backend, value=binding: value,
                    )

            underfilled = _manifest()
            underfilled["counts"] = dict(underfilled["counts"])
            underfilled["counts"]["target_33_64"] = 399
            with self.assertRaisesRegex(
                IndependentRuntimeEvaluationError, "target_33_64=399<400"
            ):
                _validate_sealed_manifest(underfilled)

            missing_negatives = _manifest()
            missing_negatives["counts"] = dict(missing_negatives["counts"])
            missing_negatives["counts"]["reviewed_negatives"] = 999
            with self.assertRaisesRegex(
                IndependentRuntimeEvaluationError, "reviewed_negatives=999<1000"
            ):
                _validate_sealed_manifest(missing_negatives)

            descriptive_only = _manifest()
            descriptive_only["counts"] = dict(descriptive_only["counts"])
            descriptive_only["counts"]["target_le_32"] = 0
            self.assertEqual(
                _validate_sealed_manifest(descriptive_only)["counts"]["target_le_32"],
                0,
            )

    def test_plan_requires_exact_gpu_provider_detail_and_fixed_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = _binding(root)
            base = dict(
                package=root / "sealed",
                dependency_manifest=_dependency_manifest(root),
                hardware_identity=_hardware_identity(root),
                project_root=evaluator.PROJECT_ROOT,
                detail_crop_size=768,
                decision_rule=_rule(),
                candidate_binding_loader=lambda _root, _backend: binding,
            )
            with mock.patch.object(
                evaluator,
                "_load_package_manifest",
                return_value=(_manifest(), root / "HOLDOUT-MANIFEST.json", b"manifest"),
            ):
                for overrides, pattern in (
                    ({"device": "AUTO", "expected_provider": "DmlExecutionProvider"}, "explicitly"),
                    ({"device": "DML:0", "expected_provider": "CPUExecutionProvider"}, "supported accelerator"),
                    ({"device": "DML:0", "expected_provider": "CUDAExecutionProvider"}, "disagree"),
                    ({"device": "DML:evil", "expected_provider": "DmlExecutionProvider"}, "numeric DML"),
                    ({"device": "DML:0", "expected_provider": "DmlExecutionProvider", "detail_crop_size": 0}, "detail crop"),
                    ({"device": "DML:0", "expected_provider": "DmlExecutionProvider", "confidence_thresholds": (0.25,)}, "exact confidence"),
                    ({"device": "DML:0", "expected_provider": "DmlExecutionProvider", "output_format": "traditional"}, "winner head"),
                ):
                    with self.subTest(overrides=overrides), self.assertRaisesRegex(
                        IndependentRuntimeEvaluationError, pattern
                    ):
                        build_evaluation_plan(**{**base, **overrides})
                diagnostic_warmup = build_evaluation_plan(
                    **{
                        **base,
                        "device": "DML:0",
                        "expected_provider": "DmlExecutionProvider",
                        "warmup": evaluator.DEFAULT_WARMUP + 100,
                    }
                )
                eligibility = evaluator._release_evidence_eligibility(
                    plan=diagnostic_warmup,
                    decision={"frozen_rule_passed": True},
                )
                self.assertFalse(eligibility["canonical_release_policy_matched"])
                self.assertFalse(eligibility["release_evidence_eligible"])

    def test_canonical_plan_is_no_replace_and_schema_rejects_extra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                evaluator,
                "_load_package_manifest",
                return_value=(_manifest(), root / "HOLDOUT-MANIFEST.json", b"manifest"),
            ):
                plan = build_evaluation_plan(
                    package=root / "sealed",
                    dependency_manifest=_dependency_manifest(root),
                    hardware_identity=_hardware_identity(root),
                    project_root=evaluator.PROJECT_ROOT,
                    device="DML:0",
                    expected_provider="DmlExecutionProvider",
                    detail_crop_size=768,
                    decision_rule=_rule(),
                    candidate_binding_loader=lambda _root, _backend: _binding(root),
                )
            output = root / "plan.json"
            write_evaluation_plan(output, plan)
            self.assertEqual(output.read_bytes(), evaluator.canonical_json_bytes(plan))
            with self.assertRaisesRegex(IndependentRuntimeEvaluationError, "overwrite"):
                write_evaluation_plan(output, plan)
            altered = deepcopy(plan)
            altered["split"] = "test"
            altered["plan_content_sha256"] = evaluator._plan_content_hash(altered)
            with self.assertRaisesRegex(IndependentRuntimeEvaluationError, "schema"):
                _validate_plan(altered)

            pipeline_drift = deepcopy(plan)
            pipeline_drift["candidate"]["selected_pipeline"] = "primary"
            pipeline_drift["plan_content_sha256"] = evaluator._plan_content_hash(
                pipeline_drift
            )
            with self.assertRaisesRegex(
                IndependentRuntimeEvaluationError, "runtime pipeline"
            ):
                _validate_plan(pipeline_drift)

            environment_drift = deepcopy(plan)
            environment_drift["environment"]["packages"]["numpy"][
                "artifact_sha256"
            ] = "0" * 64
            environment_drift["plan_content_sha256"] = evaluator._plan_content_hash(
                environment_drift
            )
            with self.assertRaisesRegex(
                IndependentRuntimeEvaluationError, "environment policy"
            ):
                _validate_plan(environment_drift)

            hardware_drift = deepcopy(plan)
            hardware_drift["hardware_identity"]["directml_adapter_index"] = 9
            hardware_drift["hardware_identity"]["content_sha256"] = canonical_hash(
                {
                    key: value
                    for key, value in hardware_drift["hardware_identity"].items()
                    if key != "content_sha256"
                }
            )
            hardware_drift["plan_content_sha256"] = evaluator._plan_content_hash(
                hardware_drift
            )
            with self.assertRaisesRegex(
                IndependentRuntimeEvaluationError, "hardware identity"
            ):
                _validate_plan(hardware_drift)

    def test_inventory_rederives_all_pinned_buckets_and_negatives(self) -> None:
        inventory = [
            {"targets": [((0, 0, 1, 1), 32.0)]},
            {"targets": [((0, 0, 1, 1), 64.0)]},
            {"targets": [((0, 0, 1, 1), 96.0)]},
            {"targets": [((0, 0, 1, 1), 97.0)]},
            {"targets": []},
        ]
        self.assertEqual(
            _inventory_counts(inventory),
            {
                "target_le_32": 1,
                "target_33_64": 1,
                "target_65_96": 1,
                "target_gt_96": 1,
                "reviewed_negatives": 1,
            },
        )

    def test_capture_session_bootstrap_moves_dependent_frames_together(self) -> None:
        targets = [((0.0, 0.0, 10.0, 48.0), 48.0)]
        images = [
            bucket_image_evidence(targets, [((0.0, 0.0, 10.0, 48.0), 48.0, 0.9)]),
            bucket_image_evidence(targets, []),
        ]
        uncertainty = _source_group_uncertainty(
            images,
            ["same-session", "same-session"],
            confidence_thresholds=(0.25,),
            bootstrap_samples=50,
            seed_binding="a" * 64,
        )
        far = uncertainty["operating_points"]["0.25"]["far_33_to_64px"]
        self.assertEqual(far["recall_ci95"], [0.5, 0.5])
        self.assertEqual(uncertainty["source_group_count"], 1)

    def test_frozen_rule_result_is_recomputable_from_raw_counts(self) -> None:
        targets = [
            ((0.0, 0.0, 10.0, 48.0), 48.0),
            ((20.0, 0.0, 30.0, 80.0), 80.0),
            ((40.0, 0.0, 50.0, 120.0), 120.0),
        ]
        detections = [
            ((0.0, 0.0, 10.0, 48.0), 48.0, 0.9),
            ((20.0, 0.0, 30.0, 80.0), 80.0, 0.9),
            ((40.0, 0.0, 50.0, 120.0), 120.0, 0.9),
        ]
        images = [bucket_image_evidence(targets, detections)]
        metrics = _metric_record(
            images, confidence_thresholds=(0.25, 0.45), bootstrap_samples=10
        )
        negative = {
            "operating_points": {
                "0.25": {"false_positives": 0},
                "0.45": {"false_positives": 0},
            }
        }
        result = _decision_result(_rule(), metrics, negative, 10.0)
        self.assertTrue(result["frozen_rule_passed"])
        self.assertEqual(result["raw_inputs"]["far_detected_over_total"], "1/1")
        self.assertEqual(result["raw_inputs"]["reviewed_negative_false_positives"], 0)
        self.assertTrue(all(result["checks"].values()))

    def test_permissive_operator_rule_is_diagnostic_never_release_eligible(self) -> None:
        permissive = {
            **_rule(),
            "minimum_far_recall": 0.0,
            "maximum_far_false_positives": 1_000_000,
            "minimum_medium_recall": 0.0,
            "minimum_near_recall": 0.0,
            "minimum_aggregate_precision": 0.0,
            "minimum_aggregate_recall": 0.0,
            "maximum_reviewed_negative_false_positives": 1_000_000,
            "maximum_runtime_pipeline_p95_ms": 1_000_000.0,
            "manual_review_note": "diagnostic-only permissive rule",
        }
        decision = {"frozen_rule_passed": True}
        eligibility = evaluator._release_evidence_eligibility(
            plan={
                "release_policy": evaluator._release_policy_record(),
                "decision_rule": evaluator._validate_decision_rule(permissive),
            },
            decision=decision,
        )
        self.assertTrue(eligibility["frozen_metric_rule_passed"])
        self.assertFalse(eligibility["canonical_release_policy_matched"])
        self.assertFalse(eligibility["release_evidence_eligible"])

    def test_ultra_far_misses_cannot_enter_any_frozen_release_gate(self) -> None:
        targets = [
            ((0.0, 0.0, 10.0, 20.0), 20.0),
            ((20.0, 0.0, 30.0, 48.0), 48.0),
            ((40.0, 0.0, 50.0, 80.0), 80.0),
            ((60.0, 0.0, 70.0, 120.0), 120.0),
        ]
        detections = [
            ((20.0, 0.0, 30.0, 48.0), 48.0, 0.9),
            ((40.0, 0.0, 50.0, 80.0), 80.0, 0.9),
            ((60.0, 0.0, 70.0, 120.0), 120.0, 0.9),
        ]
        metrics = _metric_record(
            [bucket_image_evidence(targets, detections)],
            confidence_thresholds=(0.25, 0.45),
            bootstrap_samples=10,
        )
        negative = {
            "operating_points": {
                "0.25": {"false_positives": 0},
                "0.45": {"false_positives": 0},
            }
        }
        strict_rule = {
            **_rule(),
            "minimum_aggregate_precision": 1.0,
            "minimum_aggregate_recall": 1.0,
        }
        result = _decision_result(strict_rule, metrics, negative, 10.0)

        self.assertEqual(
            metrics["aggregate_detection"]["operating_points"]["0.25"]["recall"],
            0.75,
        )
        self.assertTrue(result["frozen_rule_passed"])
        self.assertEqual(
            result["raw_inputs"]["gating_aggregate_detected_over_total"], "3/3"
        )

    def test_ultra_far_matches_are_neutral_but_every_unmatched_prediction_penalizes_precision(self) -> None:
        targets = [
            ((0.0, 0.0, 10.0, 20.0), 20.0),
            ((20.0, 0.0, 30.0, 48.0), 48.0),
            ((40.0, 0.0, 50.0, 80.0), 80.0),
            ((60.0, 0.0, 70.0, 120.0), 120.0),
        ]
        perfect = [
            ((0.0, 0.0, 10.0, 20.0), 20.0, 0.9),
            ((20.0, 0.0, 30.0, 48.0), 48.0, 0.9),
            ((40.0, 0.0, 50.0, 80.0), 80.0, 0.9),
            ((60.0, 0.0, 70.0, 120.0), 120.0, 0.9),
        ]
        negative = {
            "operating_points": {
                "0.25": {"false_positives": 0},
                "0.45": {"false_positives": 0},
            }
        }
        matched_result = _decision_result(
            _rule(),
            _metric_record(
                [bucket_image_evidence(targets, perfect)],
                confidence_thresholds=(0.25, 0.45),
                bootstrap_samples=10,
            ),
            negative,
            10.0,
        )
        self.assertEqual(matched_result["raw_inputs"]["aggregate_precision"], 1.0)
        self.assertEqual(
            matched_result["raw_inputs"]["release_precision_denominator"], 3
        )

        with_unmatched_ultra = [
            *perfect,
            ((100.0, 0.0, 110.0, 18.0), 18.0, 0.9),
            ((120.0, 0.0, 130.0, 19.0), 19.0, 0.9),
        ]
        penalized = _decision_result(
            _rule(),
            _metric_record(
                [bucket_image_evidence(targets, with_unmatched_ultra)],
                confidence_thresholds=(0.25, 0.45),
                bootstrap_samples=10,
            ),
            negative,
            10.0,
        )
        self.assertEqual(penalized["raw_inputs"]["all_size_false_positives"], 2)
        self.assertEqual(penalized["raw_inputs"]["release_precision_denominator"], 5)
        self.assertAlmostEqual(penalized["raw_inputs"]["aggregate_precision"], 0.6)
        self.assertFalse(penalized["checks"]["aggregate_precision"])

    def test_exact_rectangular_primary_detail_run_publishes_verifies_and_never_approves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "sealed"
            package.mkdir()
            model = root / "candidate.onnx"
            model.write_bytes(b"exact synthetic adopted ONNX artifact")
            labels = root / "labels.txt"
            labels.write_text("player\n", encoding="utf-8")
            pointer = root / "RELEASE-DEFAULT.json"
            adoption = root / "ADOPTION.json"
            comparison = root / "MODEL-TOURNAMENT-SELECTION.json"
            pointer.write_bytes(b"pointer\n")
            adoption.write_bytes(b"adoption\n")
            comparison.write_bytes(b"comparison\n")
            model_sha = sha256(model.read_bytes()).hexdigest()
            binding = _binding(root)
            binding.update(
                {
                    "pointer_path": pointer,
                    "pointer_sha256": sha256(pointer.read_bytes()).hexdigest(),
                    "adoption_path": adoption,
                    "adoption_sha256": sha256(adoption.read_bytes()).hexdigest(),
                    "tournament_selection_path": comparison,
                    "tournament_selection_sha256": sha256(
                        comparison.read_bytes()
                    ).hexdigest(),
                    "tournament_evidence": [
                        {
                            "role": "tournament_selection_manifest",
                            "relative_path": "models/release-defaults/fake/MODEL-TOURNAMENT-SELECTION.json",
                            "bytes": comparison.stat().st_size,
                            "sha256": sha256(comparison.read_bytes()).hexdigest(),
                        }
                    ],
                    "tournament_evidence_files": [
                        (comparison, sha256(comparison.read_bytes()).hexdigest())
                    ],
                    "candidate_provenance_evidence": [
                        {
                            "role": "candidate_receipt",
                            "relative_path": "models/release-defaults/fake/CANDIDATE-RECEIPT.json",
                            "bytes": comparison.stat().st_size,
                            "sha256": sha256(comparison.read_bytes()).hexdigest(),
                        }
                    ],
                    "candidate_provenance_files": [
                        (comparison, sha256(comparison.read_bytes()).hexdigest())
                    ],
                    "model_path": model,
                    "model_artifacts": [
                        {
                            "role": "onnx",
                            "name": model.name,
                            "relative_path": "models/release-defaults/fake/candidate.onnx",
                            "bytes": model.stat().st_size,
                            "sha256": model_sha,
                        }
                    ],
                    "model_content_sha256": canonical_hash(
                        [{"name": model.name, "sha256": model_sha}]
                    ),
                    "labels_path": labels,
                    "labels": {
                        "relative_path": "models/release-defaults/fake/labels.txt",
                        "bytes": labels.stat().st_size,
                        "sha256": sha256(labels.read_bytes()).hexdigest(),
                    },
                }
            )
            small_holdout = {
                "package_id": "sealed-runtime-fixture",
                "manifest_content_sha256": "1" * 64,
                "pool": POOL_SEALED,
                "counts": PINNED_RELEASE_MINIMUMS.as_dict(),
                "images": 2,
                "boxes": 4,
                "source_group_definition": "capture_session",
                "source_group_count": 15,
                "source_group_inventory": {
                    "definition": "distinct normalized COCO image session_id values",
                    "overall_capture_sessions": 15,
                    "target_bearing_capture_sessions": {
                        "target_le_32": 15,
                        "target_33_64": 15,
                        "target_65_96": 15,
                        "target_gt_96": 15,
                    },
                    "reviewed_negative_capture_sessions": 15,
                },
                "ultra_far_le_32_is_descriptive_only": True,
                "gating_inventory_keys": list(GATING_COUNT_KEYS),
                "redistribution_permitted_for_all_sessions": False,
            }
            manifest = {
                "manifest_content_sha256": "1" * 64,
                "pool": POOL_SEALED,
                "images": [{}, {}],
                "annotations": {"sha256": "6" * 64},
            }
            inventory = [
                {
                    "image_id": 1,
                    "path": package / "positive.png",
                    "sha256": "2" * 64,
                    "width": 1920,
                    "height": 1080,
                    "session_id": "capture-session-a",
                    "source_frame_index": 1,
                    "targets": [
                        ((700.0, 100.0, 720.0, 120.0), 20.0),
                        ((800.0, 100.0, 820.0, 148.0), 48.0),
                        ((900.0, 100.0, 920.0, 180.0), 80.0),
                        ((1000.0, 100.0, 1020.0, 220.0), 120.0),
                    ],
                    "target_identity_sha256": "3" * 64,
                    "reviewed_negative": False,
                },
                {
                    "image_id": 2,
                    "path": package / "negative.png",
                    "sha256": "4" * 64,
                    "width": 1920,
                    "height": 1080,
                    "session_id": "capture-session-a",
                    "source_frame_index": 2,
                    "targets": [],
                    "target_identity_sha256": "5" * 64,
                    "reviewed_negative": True,
                },
            ]

            class FakeGpuDetector:
                def __init__(self) -> None:
                    self.calls = 0
                    self.runtime_summary = {
                        "requested_device_input": "DML:0",
                        "requested_provider": "DmlExecutionProvider",
                        "active_providers": [
                            "DmlExecutionProvider",
                            "CPUExecutionProvider",
                        ],
                        "require_full_provider": True,
                        "configured_session_options": {
                            "disable_cpu_ep_fallback": True
                        },
                        "runtime_ep_fail_fallback_disabled": True,
                        "declared_input_shape": [1, 3, 384, 640],
                        "configured_input_shape": [1, 3, 384, 640],
                        "output_format": "end2end",
                        "model_path": "/private/path/candidate.onnx",
                    }

                def warmup(self, _iterations: int) -> None:
                    return None

                def infer(self, tensor: np.ndarray) -> np.ndarray:
                    self.calls += 1
                    self.assert_shape = tensor.shape
                    return np.zeros((1, 1, 6), dtype=np.float32)

                def postprocess(self, _raw, transform=None, frame_shape=None):
                    if transform is None or frame_shape is None:
                        raise AssertionError("source transform was omitted")
                    # Primary/detail are calls 1/2 for the positive image and
                    # 3/4 for the independently reviewed negative image.
                    if self.calls <= 2:
                        return [
                            Detection(0, "player", 0.90, target[0])
                            for target in inventory[0]["targets"]
                        ]
                    return []

            snapshot_paths: list[Path] = []
            original_model_bytes = model.read_bytes()
            original_label_bytes = labels.read_bytes()

            def detector_factory(**kwargs):
                snapshot_model = Path(kwargs["model"])
                snapshot_labels = Path(kwargs["labels"])
                snapshot_paths.extend((snapshot_model, snapshot_labels))
                self.assertNotEqual(snapshot_model, model)
                self.assertNotEqual(snapshot_labels, labels)
                self.assertEqual(snapshot_model.parent, snapshot_labels.parent)
                self.assertEqual(snapshot_model.read_bytes(), original_model_bytes)
                self.assertEqual(snapshot_labels.read_bytes(), original_label_bytes)
                # A transient swap/restore of the mutable original cannot alter
                # what the detector loads because it receives only the snapshot.
                model.write_bytes(b"transient alternate ONNX bytes")
                try:
                    self.assertEqual(snapshot_model.read_bytes(), original_model_bytes)
                finally:
                    model.write_bytes(original_model_bytes)
                return FakeGpuDetector()

            decision_rule = _rule()
            dependency_manifest = _dependency_manifest(root)
            hardware_identity = _hardware_identity(root)
            _, _, release_environment = evaluator._load_release_environment(
                dependency_manifest, project_root=evaluator.PROJECT_ROOT
            )
            _, _, release_hardware = evaluator._load_holdout_hardware_identity(
                hardware_identity
            )
            plan_record = {
                "schema_version": 1,
                "kind": evaluator.PLAN_KIND,
                "status": evaluator.PLAN_STATUS,
                "candidate": evaluator._public_candidate_binding(binding),
                "holdout": small_holdout,
                "runtime": evaluator._validate_runtime_plan(
                    {
                        "backend": "onnxruntime",
                        "device": "DML:0",
                        "expected_provider": "DmlExecutionProvider",
                        "inference_size": "384x640",
                        "input_shape_nchw": [1, 3, 384, 640],
                        "output_format": "end2end",
                        "nms_iou_threshold": 0.45,
                        "confidence_thresholds": [0.25, 0.45],
                        "warmup_iterations": evaluator.DEFAULT_WARMUP,
                        "bootstrap_samples": 2_000,
                        "require_full_provider": True,
                        "detail_crop_size_source_pixels": 768,
                    }
                ),
                "decision_rule": evaluator._validate_decision_rule(decision_rule),
                "release_policy": evaluator._release_policy_record(),
                "environment": release_environment["policy"],
                "hardware_identity": release_hardware,
                "source": evaluator._source_snapshot("onnxruntime"),
                "scope": {
                    "dataset": "one sealed independent COCO package only",
                    "grouped_v9_development_data_permitted": False,
                    "candidate_or_threshold_selection_permitted": False,
                    "release_approval_permitted": False,
                    "ultra_far_le_32_release_gate": False,
                    "claim_scope": (
                        "absolute_threshold_evidence_only_no_incumbent_comparison"
                    ),
                },
            }
            plan_record["plan_content_sha256"] = evaluator._plan_content_hash(
                plan_record
            )
            plan_path = root / "plan.json"
            write_evaluation_plan(plan_path, plan_record)
            output = root / "evidence"

            def fake_verify(*_args, **_kwargs):
                ledger = package / "access-ledger"
                events = list(ledger.glob("*.json")) if ledger.exists() else []
                return {
                    "manifest_content_sha256": "1" * 64,
                    "retired": len(events) == 2,
                    "access_events": len(events),
                }

            patches = (
                mock.patch.object(
                    evaluator,
                    "_load_package_manifest",
                    return_value=(manifest, package / "HOLDOUT-MANIFEST.json", b"manifest"),
                ),
                mock.patch.object(
                    evaluator,
                    "_validate_sealed_manifest",
                    return_value=small_holdout,
                ),
                mock.patch.object(
                    evaluator,
                    "_coco_inventory",
                    return_value=(inventory, "6" * 64),
                ),
                mock.patch.object(
                    evaluator,
                    "_load_exact_image",
                    return_value=np.zeros((1080, 1920, 3), dtype=np.uint8),
                ),
                mock.patch.object(evaluator, "verify_holdout", side_effect=fake_verify),
                mock.patch.object(
                    holdout_module,
                    "_load_package_manifest",
                    return_value=(manifest, package / "HOLDOUT-MANIFEST.json", b"manifest"),
                ),
                mock.patch.object(
                    holdout_module, "verify_holdout", side_effect=fake_verify
                ),
                mock.patch.object(
                    evaluator,
                    "_inventory_counts",
                    return_value=small_holdout["counts"],
                ),
                mock.patch.object(
                    evaluator,
                    "_inventory_source_groups",
                    return_value=small_holdout["source_group_inventory"],
                ),
                mock.patch.object(
                    evaluator, "_validate_evidence_metric_inventory"
                ),
            )
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                patches[7],
                patches[8],
                patches[9],
            ):
                clock_value = 0

                def fake_clock() -> int:
                    nonlocal clock_value
                    clock_value += 1_000_000
                    return clock_value

                transition_start = datetime(
                    2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc
                )
                transition_times = iter(
                    (transition_start, transition_start + timedelta(seconds=1))
                )

                record, ledger = evaluate_independent_holdout(
                    package=package,
                    plan=plan_path,
                    output=output,
                    dependency_manifest=dependency_manifest,
                    hardware_identity=hardware_identity,
                    project_root=evaluator.PROJECT_ROOT,
                    event_id="final-eval",
                    actor_id="release-operator",
                    retirement_event_id="final-eval-retired",
                    retirement_reason="exact final evaluation completed",
                    candidate_binding_loader=lambda _root, _backend: binding,
                    detector_factory=detector_factory,
                    clock=fake_clock,
                    utc_now=lambda: next(transition_times),
                )

                verification = verify_independent_evidence(
                    evidence=output,
                    plan=plan_path,
                    package=package,
                    dependency_manifest=dependency_manifest,
                    hardware_identity=hardware_identity,
                    project_root=evaluator.PROJECT_ROOT,
                    candidate_binding_loader=lambda _root, _backend: binding,
                )
                receipt_path = root / evaluator.RECEIPT_NAME
                write_independent_evidence_receipt(receipt_path, verification)
                portable_verification = verify_independent_evidence_receipt(
                    receipt=receipt_path,
                    evidence=output,
                    plan=plan_path,
                    package=package,
                    dependency_manifest=dependency_manifest,
                    hardware_identity=hardware_identity,
                    project_root=evaluator.PROJECT_ROOT,
                    candidate_binding_loader=lambda _root, _backend: binding,
                )
                forged_receipt = deepcopy(verification)
                forged_receipt["decision"]["rule"]["minimum_far_recall"] = 0.0
                forged_receipt["receipt_content_sha256"] = (
                    evaluator._final_receipt_content_hash(forged_receipt)
                )
                with self.assertRaisesRegex(
                    IndependentRuntimeEvaluationError,
                    "canonical-policy flag",
                ):
                    evaluator._validate_portable_receipt(forged_receipt)
                malformed_receipts = []
                fake_checks = deepcopy(verification)
                fake_checks["decision"]["result"]["checks"] = {"fake": True}
                fake_checks["decision"]["result_sha256"] = canonical_hash(
                    fake_checks["decision"]["result"]
                )
                malformed_receipts.append(fake_checks)
                bool_ultra_count = deepcopy(verification)
                bool_ultra_count["holdout"]["counts"]["target_le_32"] = True
                malformed_receipts.append(bool_ultra_count)
                bool_ultra_session = deepcopy(verification)
                bool_ultra_session["holdout"]["source_group_inventory"][
                    "target_bearing_capture_sessions"
                ]["target_le_32"] = True
                malformed_receipts.append(bool_ultra_session)
                for malformed in malformed_receipts:
                    malformed["receipt_content_sha256"] = (
                        evaluator._final_receipt_content_hash(malformed)
                    )
                    with self.assertRaises(IndependentRuntimeEvaluationError):
                        evaluator._validate_portable_receipt(malformed)

                bundle_path = publish_independent_evidence_bundle(
                    output=root / "publication-bundle",
                    receipt=receipt_path,
                    evidence=output,
                    plan=plan_path,
                    package=package,
                    dependency_manifest=dependency_manifest,
                    hardware_identity=hardware_identity,
                    project_root=evaluator.PROJECT_ROOT,
                    candidate_binding_loader=lambda _root, _backend: binding,
                )
                bundle_verification = verify_independent_evidence_bundle(
                    bundle=bundle_path,
                    evidence=output,
                    plan=plan_path,
                    package=package,
                    dependency_manifest=dependency_manifest,
                    hardware_identity=hardware_identity,
                    project_root=evaluator.PROJECT_ROOT,
                    candidate_binding_loader=lambda _root, _backend: binding,
                )
                attester_manifest = root / "ATTESTER-DEPENDENCY-MANIFEST.json"
                attester_manifest_value = (
                    valid_windows_directml_dependency_manifest(
                        evaluator.PROJECT_ROOT
                    )
                )
                # A separately created, policy-compatible release environment
                # can have path-sensitive pip-report / RECORD identities.  Use
                # a valid dynamic-only difference to exercise that boundary.
                attester_manifest_value["pip_reports"][0]["sha256"] = "f" * 64
                attester_manifest.write_bytes(
                    evaluator.canonical_json_bytes(
                        attester_manifest_value
                    )
                )
                self.assertNotEqual(
                    attester_manifest.read_bytes(), dependency_manifest.read_bytes()
                )
                with self.assertRaisesRegex(
                    IndependentRuntimeEvaluationError, "environment differs"
                ):
                    verify_independent_evidence_bundle(
                        bundle=bundle_path,
                        evidence=output,
                        plan=plan_path,
                        package=package,
                        dependency_manifest=attester_manifest,
                        hardware_identity=hardware_identity,
                        project_root=evaluator.PROJECT_ROOT,
                        candidate_binding_loader=lambda _root, _backend: binding,
                    )
                attestation_verification = verify_independent_evidence_bundle(
                    bundle=bundle_path,
                    evidence=output,
                    plan=plan_path,
                    package=package,
                    dependency_manifest=attester_manifest,
                    hardware_identity=hardware_identity,
                    project_root=evaluator.PROJECT_ROOT,
                    candidate_binding_loader=lambda _root, _backend: binding,
                    enforce_current_environment_match=False,
                )
                with self.assertRaisesRegex(
                    IndependentRuntimeEvaluationError, "refusing to overwrite"
                ):
                    publish_independent_evidence_bundle(
                        output=bundle_path,
                        receipt=receipt_path,
                        evidence=output,
                        plan=plan_path,
                        package=package,
                        dependency_manifest=dependency_manifest,
                        hardware_identity=hardware_identity,
                        project_root=evaluator.PROJECT_ROOT,
                        candidate_binding_loader=lambda _root, _backend: binding,
                    )
                real_parent = root / "real-bundle-parent"
                real_parent.mkdir()
                symlink_parent = root / "symlink-bundle-parent"
                try:
                    symlink_parent.symlink_to(real_parent, target_is_directory=True)
                except OSError:
                    pass  # Windows runners may lack symlink privilege.
                else:
                    with self.assertRaisesRegex(
                        IndependentRuntimeEvaluationError,
                        "parent contains a symlink",
                    ):
                        publish_independent_evidence_bundle(
                            output=symlink_parent / "bundle",
                            receipt=receipt_path,
                            evidence=output,
                            plan=plan_path,
                            package=package,
                            dependency_manifest=dependency_manifest,
                            hardware_identity=hardware_identity,
                            project_root=evaluator.PROJECT_ROOT,
                            candidate_binding_loader=lambda _root, _backend: binding,
                        )
                with self.assertRaisesRegex(
                    IndependentRuntimeEvaluationError, "refusing to overwrite"
                ):
                    write_independent_evidence_receipt(receipt_path, verification)
                original_evidence = (output / "metrics.json").read_bytes()
                tampered = json.loads(original_evidence.decode("utf-8"))
                tampered["qualification"]["release_gate_passed"] = True
                tampered["evidence_content_sha256"] = evaluator._evidence_content_hash(
                    tampered
                )
                (output / "metrics.json").write_bytes(
                    evaluator.canonical_json_bytes(tampered)
                )
                with self.assertRaisesRegex(
                    IndependentRuntimeEvaluationError,
                    "ledger does not bind",
                ):
                    verify_independent_evidence(
                        evidence=output,
                        plan=plan_path,
                        package=package,
                        dependency_manifest=dependency_manifest,
                        hardware_identity=hardware_identity,
                        project_root=evaluator.PROJECT_ROOT,
                        candidate_binding_loader=lambda _root, _backend: binding,
                    )
                (output / "metrics.json").write_bytes(original_evidence)

            self.assertTrue(record["decision_rule_result"]["frozen_rule_passed"])
            self.assertEqual(
                record["metrics"]["size_bucket_detection"]["operating_points"]
                ["0.25"]["far_33_to_64px"]["detected_over_total"],
                "1/1",
            )
            self.assertEqual(
                record["metrics"]["reviewed_negative_detection"]["operating_points"]
                ["0.25"]["false_positives"],
                0,
            )
            self.assertEqual(
                record["configuration"]["input_shape_nchw"], [1, 3, 384, 640]
            )
            self.assertGreater(
                record["configuration"]["detail_stats"]["frames_applied"], 0
            )
            self.assertEqual(
                record["configuration"]["detail_stats"]["last_plan"][
                    "crop_policy"
                ],
                "centered_model_aspect_roi",
            )
            self.assertEqual(
                (
                    record["configuration"]["detail_stats"]["last_plan"][
                        "applied_crop_width"
                    ],
                    record["configuration"]["detail_stats"]["last_plan"][
                        "applied_crop_height"
                    ],
                ),
                (765, 459),
            )
            self.assertNotIn("/private/path", json.dumps(record["runtime"]["summary"]))
            self.assertFalse(record["qualification"]["approved"])
            self.assertFalse(record["qualification"]["hardware_gate_passed"])
            self.assertFalse(record["qualification"]["release_gate_passed"])
            self.assertEqual(record["environment"], release_environment)
            self.assertEqual(record["hardware_identity"], release_hardware)
            self.assertEqual(
                record["one_time_access"]["timestamp_authority"],
                "UTC transition times are generated inside the exclusive ledger "
                "transaction: consumption before first sealed-member read and "
                "retirement after durable evidence publication",
            )
            self.assertEqual(
                ledger["evaluation_evidence_sha256"],
                sha256((output / "metrics.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                ledger["consumed_at_utc"], "2026-08-13T12:00:00.000000Z"
            )
            self.assertEqual(
                ledger["retired_at_utc"], "2026-08-13T12:00:01.000000Z"
            )
            self.assertEqual(verification["environment"], release_environment)
            self.assertEqual(verification["hardware_identity"], release_hardware)
            self.assertTrue(verification["release_evidence_eligible"])
            self.assertTrue(
                verification["canonical_release_policy_matched"]
            )
            self.assertFalse(verification["release_approved"])
            self.assertFalse(verification["release_pointer_changed"])
            self.assertEqual(
                portable_verification["status"],
                "verified_portable_final_holdout_receipt",
            )
            self.assertTrue(portable_verification["consumed_exactly_once"])
            self.assertTrue(portable_verification["retired"])
            self.assertEqual(len(snapshot_paths), 2)
            self.assertFalse(snapshot_paths[0].parent.exists())
            self.assertFalse(snapshot_paths[0].exists())
            self.assertFalse(snapshot_paths[1].exists())
            self.assertTrue(bundle_verification["release_evidence_eligible"])
            self.assertTrue(
                attestation_verification["release_evidence_eligible"]
            )
            self.assertFalse(bundle_verification["release_approved"])
            self.assertEqual(
                {
                    path.relative_to(root / "publication-bundle").as_posix()
                    for path in (root / "publication-bundle").rglob("*")
                    if path.is_file()
                },
                {
                    evaluator.BUNDLE_MANIFEST_NAME,
                    *evaluator.BUNDLE_MEMBER_NAMES.values(),
                },
            )


if __name__ == "__main__":
    unittest.main()
