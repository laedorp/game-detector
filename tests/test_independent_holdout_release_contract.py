from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import utils.independent_holdout_release_contract as contract
from tests.independent_holdout_environment_fixture import (
    valid_windows_directml_dependency_manifest,
)
from tests.independent_holdout_hardware_fixture import (
    valid_rx6950_holdout_hardware_identity,
)
from utils.independent_holdout_release_contract import (
    CANONICAL_RELEASE_DECISION_RULE,
    DEFAULT_WARMUP,
    GATING_INVENTORY_KEYS,
    IndependentHoldoutReleaseContractError,
    RELEASE_INVENTORY_MINIMUMS,
    receipt_verifier_record,
    reject_private_path_strings,
    release_policy_record,
    source_snapshot,
    validate_publication_bundle,
)
from utils.release_model_contract import QUALIFICATION_RECORD, canonical_hash, canonical_json_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class IndependentHoldoutReleaseContractTests(unittest.TestCase):
    def test_exact_windows_directml_environment_manifest_is_reduced_and_validated(self) -> None:
        manifest = valid_windows_directml_dependency_manifest(PROJECT_ROOT)
        payload = canonical_json_bytes(manifest)
        record = contract.release_environment_record(
            manifest,
            dependency_manifest_sha256=sha256(payload).hexdigest(),
            project_root=PROJECT_ROOT,
        )
        self.assertEqual(
            contract.validate_release_environment_record(
                record, project_root=PROJECT_ROOT
            ),
            record,
        )
        self.assertEqual(record["policy"]["python"]["version"], "3.13.14")
        self.assertEqual(record["dependency_manifest"]["distribution_count"], 24)

    def _valid_bundle(self, root: Path) -> tuple[Path, dict, dict, dict]:
        h = lambda character: character * 64
        dependency_manifest = valid_windows_directml_dependency_manifest(PROJECT_ROOT)
        environment = contract.release_environment_record(
            dependency_manifest,
            dependency_manifest_sha256=sha256(
                canonical_json_bytes(dependency_manifest)
            ).hexdigest(),
            project_root=PROJECT_ROOT,
        )
        hardware_identity = valid_rx6950_holdout_hardware_identity(
            adapter_index=0,
            qualification_run_id=1234,
            public_receipt_sha256=h("a"),
        )
        binding = {
            "pointer_sha256": h("1"),
            "pointer_content_sha256": h("2"),
            "input_shape_nchw": [1, 3, 384, 640],
            "candidate_content_sha256": h("3"),
            "candidate_manifest_sha256": h("4"),
            "checkpoint_sha256": h("5"),
            "dataset_manifest_sha256": h("6"),
            "dataset_content_sha256": h("7"),
            "adoption_sha256": h("8"),
            "adoption_content_sha256": h("9"),
            "adoption_evidence_replay_sha256": h("a"),
            "tournament_selection_sha256": h("b"),
            "tournament_selection_content_sha256": h("c"),
            "tournament_evidence": [
                {
                    "role": "tournament_selection_manifest",
                    "relative_path": "models/release-defaults/fixture/selection.json",
                    "bytes": 10,
                    "sha256": h("b"),
                }
            ],
            "candidate_provenance_evidence": [
                {
                    "role": "candidate_receipt",
                    "relative_path": "models/release-defaults/fixture/receipt.json",
                    "bytes": 10,
                    "sha256": h("d"),
                }
            ],
            "candidate_evaluation_sha256": h("e"),
            "winner_slot": "s_end2end",
            "model_artifacts": [
                {
                    "role": "onnx",
                    "name": "candidate.onnx",
                    "relative_path": "models/release-defaults/fixture/candidate.onnx",
                    "bytes": 123,
                    "sha256": h("f"),
                }
            ],
            "model_content_sha256": canonical_hash(
                [{"name": "candidate.onnx", "sha256": h("f")}]
            ),
            "labels": {
                "relative_path": "models/release-defaults/fixture/labels.txt",
                "bytes": 7,
                "sha256": h("0"),
            },
            "selected_pipeline": "configured",
            "selected_backend": "onnxruntime",
            "output_head": "end2end",
            "detail_crop_size_source_pixels": 768,
            "exporter_sha256": h("1"),
            "adoption_source_sha256": h("2"),
        }
        snapshot = {
            "files": {"fixture": {"name": "fixture.py", "sha256": h("3")}},
            "application_pipeline": {"fixture": h("4")},
        }
        verifier = {
            "schema_version": 1,
            "evaluator": {"name": "evaluator.py", "sha256": h("5")},
            "holdout_contract": {"name": "holdout.py", "sha256": h("6")},
            "public_evidence_privacy": {"name": "privacy.py", "sha256": h("7")},
            "independent_holdout_release_contract": {
                "name": "independent_holdout_release_contract.py",
                "sha256": h("8"),
            },
            "source_snapshot_sha256": canonical_hash(snapshot),
            "application_pipeline_sha256": canonical_hash(
                snapshot["application_pipeline"]
            ),
            "release_policy_sha256": release_policy_record()["policy_sha256"],
        }
        counts = {
            "target_le_32": 0,
            "target_33_64": 400,
            "target_65_96": 250,
            "target_gt_96": 250,
            "reviewed_negatives": 1_000,
        }
        source_groups = {
            "definition": "distinct normalized COCO image session_id values",
            "overall_capture_sessions": 15,
            "target_bearing_capture_sessions": {
                "target_le_32": 0,
                "target_33_64": 15,
                "target_65_96": 15,
                "target_gt_96": 15,
            },
            "reviewed_negative_capture_sessions": 15,
        }
        holdout = {
            "package_id": "sealed-fixture",
            "manifest_content_sha256": h("9"),
            "pool": "sealed_release_holdout",
            "counts": counts,
            "images": 1_900,
            "boxes": 900,
            "source_group_definition": "capture_session",
            "source_group_count": 15,
            "source_group_inventory": source_groups,
            "ultra_far_le_32_is_descriptive_only": True,
            "gating_inventory_keys": list(contract.GATING_INVENTORY_KEYS),
            "redistribution_permitted_for_all_sessions": True,
        }
        runtime_plan = {
            "backend": "onnxruntime",
            "device": "DML:0",
            "expected_provider": "DmlExecutionProvider",
            "inference_size": "384x640",
            "input_shape_nchw": [1, 3, 384, 640],
            "output_format": "end2end",
            "nms_iou_threshold": 0.45,
            "confidence_thresholds": [0.25, 0.45],
            "warmup_iterations": 3,
            "bootstrap_samples": 2_000,
            "require_full_provider": True,
            "detail_crop_size_source_pixels": 768,
        }
        plan = {
            "schema_version": 1,
            "kind": contract.PLAN_KIND,
            "status": contract.PLAN_STATUS,
            "candidate": binding,
            "holdout": holdout,
            "runtime": runtime_plan,
            "decision_rule": dict(CANONICAL_RELEASE_DECISION_RULE),
            "release_policy": release_policy_record(),
            "environment": environment["policy"],
            "hardware_identity": hardware_identity,
            "source": snapshot,
            "scope": {
                "dataset": "one sealed independent COCO package only",
                "grouped_v9_development_data_permitted": False,
                "candidate_or_threshold_selection_permitted": False,
                "release_approval_permitted": False,
                "ultra_far_le_32_release_gate": False,
                "claim_scope": "absolute_threshold_evidence_only_no_incumbent_comparison",
            },
        }
        plan["plan_content_sha256"] = canonical_hash(plan)
        plan_payload = canonical_json_bytes(plan)
        plan_sha = sha256(plan_payload).hexdigest()

        def point(total: int) -> dict:
            return {
                "ground_truth_total": total,
                "detected_true_positives": total,
                "missed_false_negatives": 0,
                "predictions": total,
                "false_positives": 0,
                "detected_over_total": f"{total}/{total}",
                "precision": 1.0 if total else None,
                "recall": 1.0 if total else None,
            }

        buckets = {
            "ultra_far_le_32px": point(0),
            "far_33_to_64px": point(400),
            "medium_65_to_96px": point(250),
            "near_gt_96px": point(250),
        }
        negative_point = {
            "reviewed_negative_images": 1_000,
            "false_positives": 0,
            "negative_images_with_false_positive": 0,
            "false_positives_per_image": 0.0,
            "negative_image_false_positive_rate": 0.0,
            "capture_session_cluster_bootstrap_95_ci": {},
        }
        decision_result = {
            "frozen_rule_passed": True,
            "checks": {
                "far_recall": True,
                "far_false_positives": True,
                "medium_recall": True,
                "near_recall": True,
                "aggregate_precision": True,
                "aggregate_recall": True,
                "reviewed_negative_false_positives": True,
                "runtime_pipeline_p95_ms": True,
            },
            "selected_confidence_threshold": 0.25,
            "raw_inputs": {
                "far_detected_over_total": "400/400",
                "far_false_positives": 0,
                "medium_detected_over_total": "250/250",
                "near_detected_over_total": "250/250",
                "gating_aggregate_detected_over_total": "900/900",
                "all_size_predictions_observed": 900,
                "all_size_false_positives": 0,
                "release_precision_denominator": 900,
                "aggregate_precision": 1.0,
                "aggregate_recall": 1.0,
                "reviewed_negative_false_positives": 0,
                "runtime_pipeline_p95_ms": 10.0,
            },
            "scope": contract.DECISION_RESULT_SCOPE,
        }
        evidence_holdout = {
            **holdout,
            "normalized_coco_sha256": h("a"),
            "exact_member_verification_before_and_after_inference": True,
            "ground_truth_source": "sealed normalized COCO; no grouped-v9 YAML/split",
        }
        evidence_qualification = {
            **QUALIFICATION_RECORD,
            "final_holdout_evaluation_completed": True,
            "canonical_release_policy_matched": True,
            "frozen_metric_rule_passed": True,
            "release_evidence_eligible": True,
            "comparative_incumbent_improvement_proven": False,
            "manual_release_review_required": True,
            "hardware_gate_passed": False,
            "frozen_build_gate_passed": False,
            "legal_redistribution_gate_passed": False,
            "release_gate_passed": False,
            "reason": (
                "This is release-eligible evidence for manual review only. It "
                "cannot approve a release or satisfy separate physical GPU, "
                "frozen-build, and legal-distribution gates."
            ),
        }
        evidence = {
            "schema_version": 1,
            "kind": contract.EVIDENCE_KIND,
            "status": "valid_final_holdout_evidence_meeting_frozen_rule_not_release_approved",
            "evaluation_plan": {
                "sha256": plan_sha,
                "content_sha256": plan["plan_content_sha256"],
                "frozen_decision_rule": dict(CANONICAL_RELEASE_DECISION_RULE),
            },
            "release_policy": release_policy_record(),
            "candidate": binding,
            "model_artifact": {
                "backend": "onnxruntime",
                "content_sha256": binding["model_content_sha256"],
                "members": [
                    {"name": "candidate.onnx", "bytes": 123, "sha256": h("f")}
                ],
            },
            "holdout": evidence_holdout,
            "configuration": {
                **runtime_plan,
                "evaluation_mode": "sealed_independent_exact_application_runtime_artifact",
                "adopted_tournament_pipeline": "configured",
                "configured_pipeline": "rectangular_full_frame_plus_center_model_aspect_detail_merged",
                "primary_reference_retained": True,
                "detail_merge": "detection.detail_pass.merge_cross_pass_detections exactly once for every non-redundant detail frame",
                "detail_stats": {
                    "enabled": True,
                    "frames_seen": 1_900,
                    "frames_applied": 1_900,
                    "last_plan": {"crop_policy": "centered_model_aspect_roi"},
                },
            },
            "runtime": {
                "summary": {
                    "requested_device_input": "DML:0",
                    "requested_provider": "DmlExecutionProvider",
                    "requested_device": "DmlExecutionProvider",
                    "active_providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
                    "provider_option_overrides": {
                        "DmlExecutionProvider": {"device_id": "0"}
                    },
                    "require_full_provider": True,
                    "runtime_ep_fail_fallback_disabled": True,
                    "configured_session_options": {"disable_cpu_ep_fallback": True},
                    "output_format": "end2end",
                    "input_shape": [1, 3, 384, 640],
                    "declared_input_shape": [1, 3, 384, 640],
                    "configured_input_shape": [1, 3, 384, 640],
                },
                "declared_static_input_shape_nchw": [1, 3, 384, 640],
                "observed_raw_output_shape": [1, 1, 6],
                "observed_raw_output_dtype": "float32",
                "timing_ms_per_image": {"runtime_pipeline": {"p95": 10.0}},
                "timing_scope": "synchronous batch-one measurement",
            },
            "metrics": {
                "images": 1_900,
                "ground_truth_boxes": 900,
                "size_bucket_detection": {
                    "operating_points": {"0.25": buckets, "0.45": buckets}
                },
                "reviewed_negative_detection": {
                    "operating_points": {
                        "0.25": negative_point,
                        "0.45": negative_point,
                    }
                },
            },
            "decision_rule_result": decision_result,
            "one_time_access": {
                "event_id": "holdout-consumed",
                "actor_id": "protected-holdout-workflow",
                "purpose": "execute the exact frozen ProAim independent runtime evaluation plan",
                "retirement_event_id": "holdout-retired",
                "retirement_reason": "canonical final evaluation completed",
                "timestamp_authority": (
                    "UTC transition times are generated inside the exclusive "
                    "ledger transaction: consumption before first sealed-member "
                    "read and retirement after durable evidence publication"
                ),
                "publication_order": (
                    "durable pre-access consumption, atomic evidence publication, "
                    "then evidence-hash-bound retirement while the exclusive lock "
                    "remains held"
                ),
            },
            "environment": environment,
            "hardware_identity": hardware_identity,
            "source": snapshot,
            "qualification": evidence_qualification,
        }
        evidence["evidence_content_sha256"] = canonical_hash(evidence)
        evidence_payload = canonical_json_bytes(evidence)
        evidence_sha = sha256(evidence_payload).hexdigest()

        def ledger_hash(value: dict) -> str:
            payload = (
                json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
                + "\n"
            ).encode()
            return sha256(payload).hexdigest()

        consumed = {
            "schema_version": 1,
            "sequence": 1,
            "event_id": "holdout-consumed",
            "operation": "consumed",
            "recorded_at_utc": "2026-08-13T12:00:00Z",
            "actor_id": "protected-holdout-workflow",
            "dataset_manifest_content_sha256": h("9"),
            "previous_event_sha256": None,
            "purpose": "execute the exact frozen ProAim independent runtime evaluation plan",
            "evaluation_plan_sha256": plan_sha,
        }
        consumed["event_content_sha256"] = ledger_hash(consumed)
        retired = {
            "schema_version": 2,
            "sequence": 2,
            "event_id": "holdout-retired",
            "operation": "retired",
            "recorded_at_utc": "2026-08-13T12:00:01Z",
            "actor_id": "protected-holdout-workflow",
            "dataset_manifest_content_sha256": h("9"),
            "previous_event_sha256": consumed["event_content_sha256"],
            "reason": "canonical final evaluation completed",
            "evaluation_evidence_sha256": evidence_sha,
        }
        retired["event_content_sha256"] = ledger_hash(retired)
        consumed_payload = (
            json.dumps(consumed, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode()
        retired_payload = (
            json.dumps(retired, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode()
        receipt_holdout = {
            "package_id": "sealed-fixture",
            "manifest_content_sha256": h("9"),
            "normalized_coco_sha256": h("a"),
            "counts": counts,
            "source_group_inventory": source_groups,
        }
        receipt = {
            "schema_version": 1,
            "kind": contract.RECEIPT_KIND,
            "status": "verified_release_eligible_evidence_not_release_approved",
            "claim_scope": "absolute_threshold_evidence_only_no_incumbent_comparison",
            "release_policy": release_policy_record(),
            "environment": environment,
            "hardware_identity": hardware_identity,
            "verifier": verifier,
            "evidence": {
                "name": "metrics.json",
                "bytes": len(evidence_payload),
                "sha256": evidence_sha,
                "content_sha256": evidence["evidence_content_sha256"],
            },
            "evaluation_plan": {
                "bytes": len(plan_payload),
                "sha256": plan_sha,
                "content_sha256": plan["plan_content_sha256"],
            },
            "holdout": receipt_holdout,
            "candidate": contract._receipt_candidate(binding),
            "workload": {
                "backend": "onnxruntime",
                "expected_provider": "DmlExecutionProvider",
                "input_shape_nchw": [1, 3, 384, 640],
                "output_format": "end2end",
                "selected_pipeline": "configured",
                "detail_crop_size_source_pixels": 768,
                "confidence_thresholds": [0.25, 0.45],
                "nms_iou_threshold": 0.45,
                "bootstrap_samples": 2_000,
                "warmup_iterations": 3,
                "require_full_provider": True,
                "runtime_pipeline_p95_ms": 10.0,
            },
            "decision": {
                "rule": dict(CANONICAL_RELEASE_DECISION_RULE),
                "result": decision_result,
                "result_sha256": canonical_hash(decision_result),
            },
            "one_time_ledger": {
                "event_count": 2,
                "consumed_exactly_once": True,
                "retired": True,
                "consumption_event": {
                    "name": "ledger/consumed.json",
                    "bytes": len(consumed_payload),
                    "sha256": sha256(consumed_payload).hexdigest(),
                    "event_id": consumed["event_id"],
                    "recorded_at_utc": consumed["recorded_at_utc"],
                    "event_content_sha256": consumed["event_content_sha256"],
                    "evaluation_plan_sha256": plan_sha,
                },
                "retirement_event": {
                    "name": "ledger/retired.json",
                    "bytes": len(retired_payload),
                    "sha256": sha256(retired_payload).hexdigest(),
                    "event_id": retired["event_id"],
                    "recorded_at_utc": retired["recorded_at_utc"],
                    "event_content_sha256": retired["event_content_sha256"],
                    "previous_event_sha256": consumed["event_content_sha256"],
                    "evaluation_evidence_sha256": evidence_sha,
                },
            },
            "canonical_release_policy_matched": True,
            "frozen_metric_rule_passed": True,
            "release_evidence_eligible": True,
            "release_approved": False,
            "release_pointer_changed": False,
            "manual_release_review_required": True,
            "separate_hardware_frozen_build_and_legal_gates_required": True,
            "qualification": {
                **QUALIFICATION_RECORD,
                "hardware_gate_passed": False,
                "frozen_build_gate_passed": False,
                "legal_redistribution_gate_passed": False,
                "release_gate_passed": False,
                "comparative_incumbent_improvement_proven": False,
            },
        }
        receipt["receipt_content_sha256"] = canonical_hash(receipt)
        receipt_payload = canonical_json_bytes(receipt)
        bundle = root / "bundle"
        (bundle / "ledger").mkdir(parents=True)
        payloads = {
            "receipt": receipt_payload,
            "evidence": evidence_payload,
            "evaluation_plan": plan_payload,
            "consumption_event": consumed_payload,
            "retirement_event": retired_payload,
        }
        for role, payload in payloads.items():
            (bundle / contract.BUNDLE_MEMBER_NAMES[role]).write_bytes(payload)
        members = {
            role: {
                "path": contract.BUNDLE_MEMBER_NAMES[role],
                "bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
            for role, payload in payloads.items()
        }
        manifest = {
            "schema_version": 1,
            "kind": contract.BUNDLE_KIND,
            "status": "verified_release_eligible_publication_inputs_not_release_approved",
            "members": {name: members[name] for name in sorted(members)},
            "bindings": {
                "receipt_content_sha256": receipt["receipt_content_sha256"],
                "release_policy_sha256": release_policy_record()["policy_sha256"],
                "source_snapshot_sha256": verifier["source_snapshot_sha256"],
                "evidence_sha256": evidence_sha,
                "evaluation_plan_sha256": plan_sha,
                "candidate_binding_sha256": canonical_hash(receipt["candidate"]),
                "holdout_binding_sha256": canonical_hash(receipt_holdout),
                "environment_record_sha256": environment[
                    "record_content_sha256"
                ],
                "hardware_identity_sha256": hardware_identity[
                    "content_sha256"
                ],
                "consumption_event_content_sha256": consumed["event_content_sha256"],
                "retirement_event_content_sha256": retired["event_content_sha256"],
            },
            "canonical_release_policy_matched": True,
            "release_evidence_eligible": True,
            "authenticated_origin_required": True,
            "release_approved": False,
            "release_pointer_changed": False,
            "qualification": dict(QUALIFICATION_RECORD),
        }
        manifest["bundle_content_sha256"] = canonical_hash(manifest)
        (bundle / contract.BUNDLE_MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest))
        return bundle, binding, snapshot, verifier

    def test_policy_is_canonical_absolute_only_and_tiny_bucket_is_descriptive(self) -> None:
        policy = release_policy_record()

        self.assertEqual(policy["policy_version"], "proaim-independent-holdout-v1")
        self.assertEqual(policy["decision_rule"], CANONICAL_RELEASE_DECISION_RULE)
        self.assertEqual(policy["inventory_minimums"], RELEASE_INVENTORY_MINIMUMS)
        self.assertNotIn("target_le_32", GATING_INVENTORY_KEYS)
        self.assertEqual(policy["inventory_minimums"]["reviewed_negatives"], 1_000)
        self.assertEqual(DEFAULT_WARMUP, 3)
        unsigned = dict(policy)
        unsigned.pop("policy_sha256")
        self.assertEqual(policy["policy_sha256"], canonical_hash(unsigned))
        self.assertEqual(
            policy["claim_scope"],
            "absolute_threshold_evidence_only_no_incumbent_comparison",
        )

    def test_source_snapshot_and_receipt_verifier_bind_shared_contract(self) -> None:
        snapshot = source_snapshot(PROJECT_ROOT)
        verifier = receipt_verifier_record(PROJECT_ROOT)

        shared = snapshot["files"]["independent_holdout_release_contract"]
        self.assertEqual(shared["name"], "independent_holdout_release_contract.py")
        self.assertEqual(verifier["independent_holdout_release_contract"], shared)
        self.assertEqual(
            verifier["source_snapshot_sha256"], canonical_hash(snapshot)
        )
        self.assertEqual(
            verifier["application_pipeline_sha256"],
            canonical_hash(snapshot["application_pipeline"]),
        )
        self.assertEqual(
            verifier["release_policy_sha256"],
            release_policy_record()["policy_sha256"],
        )

    def test_hosted_contract_imports_no_ml_runtime(self) -> None:
        source = PROJECT_ROOT / "utils" / "independent_holdout_release_contract.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = {
            node.names[0].name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) and node.names
        } | {
            (node.module or "").split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertTrue({"cv2", "numpy", "onnxruntime"}.isdisjoint(imported))

    def test_privacy_scans_mapping_keys_values_and_embedded_path_tokens(self) -> None:
        safe = {
            "source": "https://github.com/RootKit-Org/AI-Aimbot",
            "metric": "33-64px",
        }
        reject_private_path_strings(safe, "safe fixture")

        for disclosure in (
            {"/tmp/secret": "value"},
            {"origin": "C:\\Users\\person\\holdout.json"},
            {"origin": "metric=/home/person/holdout.json"},
            {"origin": "https://example.test/?source=/private/holdout"},
        ):
            with self.subTest(disclosure=disclosure), self.assertRaises(
                IndependentHoldoutReleaseContractError
            ):
                reject_private_path_strings(disclosure, "public fixture")

    def test_full_publication_bundle_accepts_zero_descriptive_ultra_far(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, binding, snapshot, verifier = self._valid_bundle(root)
            result = validate_publication_bundle(
                bundle,
                project_root=PROJECT_ROOT,
                candidate_binding_loader=lambda _root: binding,
                source_snapshot_loader=lambda _root: snapshot,
                receipt_verifier_loader=lambda _root: verifier,
            )

        self.assertTrue(result["release_evidence_eligible"])
        self.assertTrue(result["authenticated_origin_required"])
        self.assertTrue(result["consumed_exactly_once"])
        self.assertTrue(result["retired"])
        self.assertEqual(
            result["hardware_identity_sha256"],
            result["hardware_identity"]["content_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
