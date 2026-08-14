from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import scripts.adopt_fort_release_candidate as adoption_module
from scripts.adopt_fort_release_candidate import CandidateAdoptionError, adopt_candidate
from scripts.export_fort_release_candidate import _manifest_content_hash
import scripts.run_fort_model_tournament as tournament_module
from scripts.run_fort_model_tournament import (
    OUTCOME_CHECKS,
    REQUIRED_EVIDENCE_CHECKS,
    run_tournament,
)
from tests.test_fort_model_tournament import _fixture
from utils.release_model_contract import (
    QUALIFICATION_RECORD,
    ReleaseModelContractError,
    TOURNAMENT_COMPARISON_NAMES,
    canonical_hash,
    canonical_json_bytes,
    contract_content_hash,
    load_release_default_contract,
    make_release_default_contract,
    validate_release_default_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _file_record(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


class ReleaseModelAdoptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self._project()
        self.candidate, self.evaluation, self.selection = self._selection(
            self.root / "configured-evidence"
        )
        self.pointer_before = (self.root / "models" / "RELEASE-DEFAULT.json").read_bytes()
        self.manifest_before = (
            self.root / "models" / "RELEASE-MANIFEST.sha256"
        ).read_bytes()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _skip_packaged_validation(*_args: object, **_kwargs: object) -> None:
        return None

    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(value))

    def _project(self) -> None:
        for relative in (
            "scripts/export_fort_release_candidate.py",
            "scripts/compare_fort_runtime_evaluations.py",
            "scripts/run_fort_model_tournament.py",
            "utils/release_model_contract.py",
            "utils/public_evidence.py",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((PROJECT_ROOT / relative).read_bytes())
        current = self.root / "models" / "current"
        current.mkdir(parents=True)
        files = {
            "onnx": current / "current.onnx",
            "openvino_xml": current / "current.xml",
            "openvino_bin": current / "current.bin",
            "labels": current / "labels.txt",
            "attribution": current / "ATTRIBUTION.md",
        }
        files["onnx"].write_bytes(b"current onnx")
        files["openvino_xml"].write_bytes(b"current xml")
        files["openvino_bin"].write_bytes(b"current bin")
        files["labels"].write_text("player\n", encoding="utf-8")
        files["attribution"].write_text(
            "FORT-Cuh\ncreativecommons.org/licenses/by/4.0\nAGPL-3.0\n",
            encoding="utf-8",
        )
        artifacts = {
            role: {"path": path.relative_to(self.root).as_posix(), **_file_record(path)}
            for role, path in files.items()
        }
        pointer = make_release_default_contract(
            label="Game players — Existing 416 (Recommended)",
            description="Existing fixture default.",
            input_shape_nchw=[1, 3, 416, 416],
            detail_crop_size_source_pixels=0,
            artifacts=artifacts,
            provenance={
                "kind": "existing_release_default_migration",
                "candidate_content_sha256": None,
                "candidate_manifest_sha256": None,
                "tournament_selection_sha256": None,
            },
        )
        self._write_json(self.root / "models" / "RELEASE-DEFAULT.json", pointer)
        (self.root / "models" / "RELEASE-MANIFEST.sha256").write_bytes(
            b"".join(
                f"{record['sha256']}  {record['path']}\n".encode()
                for record in artifacts.values()
            )
        )

    def _selection(
        self,
        evidence_root: Path,
        *,
        challenger_advances: bool = True,
    ) -> tuple[Path, Path, Path]:
        evidence_root.mkdir(parents=True)
        levels = None
        if not challenger_advances:
            levels = {
                (scale, head): 0
                for scale in ("n", "s")
                for head in ("end2end", "traditional")
            }
        plan_path = _fixture(evidence_root, levels=levels)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for scale in ("n", "s"):
            for head in ("end2end", "traditional"):
                candidate = evidence_root / plan["models"][scale][head]["candidate_dir"]
                report = evidence_root / plan["models"][scale][head]["validation_report"]
                candidate_manifest = json.loads(
                    (candidate / "candidate-manifest.json").read_text()
                )
                candidate_manifest["training_provenance"].update(
                    {
                        "initial_run_contract_sha256": candidate_manifest["artifacts"]
                        ["initial_run_contract"]["sha256"],
                        "training_reproducibility_sha256": candidate_manifest[
                            "artifacts"
                        ]["training_reproducibility"]["sha256"],
                        "training_results_sha256": candidate_manifest["artifacts"]
                        ["training_results"]["sha256"],
                    }
                )
                candidate_manifest["candidate_content_sha256"] = _manifest_content_hash(
                    candidate_manifest
                )
                self._write_json(
                    candidate / "candidate-manifest.json",
                    candidate_manifest,
                )
                self._write_json(report, json.loads(report.read_text()))
        selection = evidence_root / "selection"
        manifest = run_tournament(
            plan_path=plan_path,
            output=selection,
            packaged_training_validator=self._skip_packaged_validation,
        )
        winner = manifest["development_selection"]["winner"]
        slot = plan["models"][winner["scale"]][winner["head"]]
        return (
            evidence_root / slot["candidate_dir"],
            evidence_root / slot["validation_report"],
            selection,
        )

    def _fake_validator(self, candidate: Path) -> dict[str, object]:
        manifest = json.loads(
            (candidate / "candidate-manifest.json").read_text(encoding="utf-8")
        )
        return {
            "status": "validated",
            "candidate_content_sha256": manifest["candidate_content_sha256"],
        }

    def _adopt(self, **kwargs: object) -> dict[str, object]:
        arguments = {
            "candidate": self.candidate,
            "candidate_evaluation": self.evaluation,
            "tournament_selection": self.selection,
            "project_root": self.root,
            "candidate_validator": self._fake_validator,
        }
        arguments.update(kwargs)
        return adopt_candidate(**arguments)  # type: ignore[arg-type]

    def test_adopts_exact_configured_winner_and_both_portable_formats(self) -> None:
        old_assets = {
            path: path.read_bytes() for path in (self.root / "models" / "current").iterdir()
        }

        result = self._adopt()

        pointer = load_release_default_contract(self.root, verify_files=True)
        self.assertEqual(pointer["input_shape_nchw"], [1, 3, 384, 640])
        self.assertEqual(pointer["detail_crop_size_source_pixels"], 768)
        self.assertEqual(result["qualification"], QUALIFICATION_RECORD)
        self.assertTrue(all(value is False for value in pointer["qualification"].values()))
        expected_roles = {
            "onnx",
            "openvino_xml",
            "openvino_bin",
            "labels",
            "attribution",
            "adoption_record",
            "candidate_receipt",
            "training_provenance_receipt",
            "training_results",
            "winner_runtime_receipt",
            "tournament_selection_manifest",
            "tournament_plan",
            "tournament_training_results_n",
            "tournament_training_results_s",
            "tournament_runtime_report_n_end2end",
            "tournament_runtime_report_n_traditional",
            "tournament_runtime_report_s_end2end",
            "tournament_runtime_report_s_traditional",
            *(f"tournament_comparison_{name}" for name in TOURNAMENT_COMPARISON_NAMES),
        }
        self.assertEqual(set(pointer["artifacts"]), expected_roles)
        for record in pointer["artifacts"].values():
            self.assertTrue((self.root / record["path"]).is_file())
        for path, payload in old_assets.items():
            self.assertEqual(path.read_bytes(), payload)
        adoption = json.loads(
            (self.root / pointer["artifacts"]["adoption_record"]["path"]).read_text()
        )
        self.assertTrue(adoption["selection"]["sealed_tournament_winner"])
        self.assertFalse(adoption["selection"]["release_qualified"])
        self.assertEqual(adoption["candidate"]["output_head"], "traditional")
        self.assertEqual(
            adoption["source"]["public_evidence_sha256"],
            tournament_selection_sha := sha256(
                (self.root / "utils" / "public_evidence.py").read_bytes()
            ).hexdigest(),
        )
        replay = adoption["selection"]["evidence_replay"]
        self.assertEqual(
            replay["status"],
            "sealed_plan_comparisons_and_winner_training_replayed_not_release_qualified",
        )
        self.assertEqual(
            set(replay["comparison"]["records"]), set(TOURNAMENT_COMPARISON_NAMES)
        )
        self.assertEqual(replay["winner_training_results"]["scale"], "s")
        self.assertTrue(
            all(value is False for value in replay["qualification"].values())
        )
        candidate_receipt = json.loads(
            (self.root / pointer["artifacts"]["candidate_receipt"]["path"]).read_text()
        )
        training_receipt = json.loads(
            (
                self.root
                / pointer["artifacts"]["training_provenance_receipt"]["path"]
            ).read_text()
        )
        winner_receipt = json.loads(
            (
                self.root / pointer["artifacts"]["winner_runtime_receipt"]["path"]
            ).read_text()
        )
        self.assertEqual(
            candidate_receipt["original_candidate_manifest_sha256"],
            adoption["candidate"]["candidate_manifest_sha256"],
        )
        self.assertEqual(
            winner_receipt["original_runtime_evaluation_sha256"],
            adoption["selection"]["candidate_evaluation_sha256"],
        )
        source_artifacts = adoption["candidate"]["source_artifacts"]
        for role in ("initial_run_contract", "training_reproducibility"):
            self.assertEqual(
                training_receipt["original_local_records"][f"{role}_sha256"],
                source_artifacts[role]["sha256"],
            )
        self.assertEqual(
            pointer["artifacts"]["training_results"]["sha256"],
            source_artifacts["training_results"]["sha256"],
        )
        tournament_selection = json.loads(
            (
                self.root
                / pointer["artifacts"]["tournament_selection_manifest"]["path"]
            ).read_text()
        )
        self.assertEqual(
            tournament_selection["public_evidence_privacy"],
            {"path": "public_evidence.py", "sha256": tournament_selection_sha},
        )
        self.assertEqual(
            pointer["artifacts"]["tournament_plan"]["sha256"],
            tournament_selection["sealed_inputs"]["plan"]["sha256"],
        )
        for name, record in tournament_selection["sealed_inputs"][
            "runtime_reports"
        ].items():
            self.assertEqual(
                pointer["artifacts"][f"tournament_runtime_report_{name}"]["sha256"],
                record["sha256"],
            )
        for scale, record in tournament_selection["sealed_inputs"][
            "training_results"
        ].items():
            self.assertEqual(
                pointer["artifacts"][f"tournament_training_results_{scale}"][
                    "sha256"
                ],
                record["sha256"],
            )
        for receipt in (candidate_receipt, training_receipt, winner_receipt):
            content_sha256 = receipt.pop("content_sha256")
            self.assertEqual(content_sha256, canonical_hash(receipt))
            receipt["content_sha256"] = content_sha256
        public_receipt_text = "".join(
            (self.root / record["path"]).read_text(encoding="utf-8")
            for record in pointer["artifacts"].values()
            if Path(record["path"]).suffix in {".json", ".csv"}
        )
        # Relative logical roles such as inputs/training/n/... are intentionally
        # portable public evidence.  Reject the fixture root, home directories,
        # source-candidate absolute paths, and Windows separators without
        # treating a safe relative directory name as a privacy leak.
        for private_marker in (
            self.root.as_posix(),
            Path.home().as_posix(),
            "/candidate/",
            "/home/",
            "\\",
        ):
            self.assertNotIn(private_marker, public_receipt_text)
        missing_receipt = deepcopy(pointer)
        missing_receipt["artifacts"].pop("candidate_receipt")
        missing_receipt["content_sha256"] = contract_content_hash(missing_receipt)
        with self.assertRaisesRegex(ReleaseModelContractError, "artifact roles"):
            validate_release_default_contract(missing_receipt)

    def test_primary_tournament_winner_maps_to_disabled_detail_workload(self) -> None:
        candidate, evaluation, selection = self._selection(
            self.root / "primary-evidence", challenger_advances=False
        )

        result = self._adopt(
            candidate=candidate,
            candidate_evaluation=evaluation,
            tournament_selection=selection,
            validate_only=True,
        )

        self.assertEqual(result["selected_runtime"]["selected_pipeline"], "primary")
        self.assertEqual(
            result["selected_runtime"]["detail_crop_size_source_pixels"], 0
        )

    def test_validate_only_is_read_only(self) -> None:
        result = self._adopt(validate_only=True)
        self.assertEqual(result["status"], "validated_for_adoption_not_published")
        self.assertEqual(
            (self.root / "models" / "RELEASE-DEFAULT.json").read_bytes(),
            self.pointer_before,
        )
        self.assertEqual(
            (self.root / "models" / "RELEASE-MANIFEST.sha256").read_bytes(),
            self.manifest_before,
        )
        self.assertFalse((self.root / "models" / "release-defaults").exists())

    def test_rejects_a_nonwinner_candidate_or_runtime_report(self) -> None:
        plan = json.loads(
            (self.root / "configured-evidence" / "tournament-plan.json").read_text()
        )
        loser = plan["models"]["n"]["end2end"]
        for scenario in ("candidate", "report"):
            kwargs: dict[str, object] = {"validate_only": True}
            if scenario == "candidate":
                kwargs["candidate"] = self.root / "configured-evidence" / loser["candidate_dir"]
            else:
                kwargs["candidate_evaluation"] = (
                    self.root / "configured-evidence" / loser["validation_report"]
                )
            with self.subTest(scenario=scenario), self.assertRaises(CandidateAdoptionError):
                self._adopt(**kwargs)

    def test_tampered_selection_or_comparison_is_rejected(self) -> None:
        manifest_path = self.selection / "selection-manifest.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        manifest["development_selection"]["winner"]["pipeline"] = "primary"
        self._write_json(manifest_path, manifest)
        with self.assertRaisesRegex(CandidateAdoptionError, "self-hash"):
            self._adopt(validate_only=True)
        manifest_path.write_bytes(original)

        comparison = next((self.selection / "comparisons").glob("*/comparison.json"))
        comparison.write_bytes(comparison.read_bytes() + b" ")
        with self.assertRaises(CandidateAdoptionError):
            self._adopt(validate_only=True)

    def test_fully_rehashed_forged_bracket_is_rejected_by_report_replay(self) -> None:
        manifest_path = self.selection / "selection-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidates = manifest["candidates"]
        for name in TOURNAMENT_COMPARISON_NAMES:
            comparison_path = self.selection / manifest["comparisons"][name]["path"]
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            checks = comparison["development_advancement_policy"]["checks"]
            for key in REQUIRED_EVIDENCE_CHECKS:
                checks[key] = True
            for key in OUTCOME_CHECKS:
                checks[key] = False
            comparison["development_advancement_policy"]["passed"] = False
            if name in {
                "n_end2end_vs_traditional",
                "s_end2end_vs_traditional",
            }:
                comparison["reports"]["baseline"]["pipeline"] = "primary"
                comparison["reports"]["candidate"]["pipeline"] = "primary"
            elif name == "n_vs_s":
                for side, slot_name in (
                    ("baseline", "n_end2end"),
                    ("candidate", "s_end2end"),
                ):
                    slot = candidates[slot_name]
                    comparison["reports"][side] = {
                        "metrics_sha256": slot["validation_report_sha256"],
                        "model_content_sha256": slot[
                            "runtime_model_content_sha256"
                        ],
                        "pipeline": "primary",
                    }
            self._write_json(comparison_path, comparison)
            manifest["comparisons"][name].update(
                {
                    "sha256": sha256(comparison_path.read_bytes()).hexdigest(),
                    "challenger_advanced": False,
                }
            )
        for slot in candidates.values():
            slot["selected_pipeline_after_detail_ab"] = "primary"
        decisions = manifest["development_selection"]
        for decision in decisions["detail_decisions"].values():
            decision["challenger_advanced"] = False
            decision["selected_pipeline"] = "primary"
        for decision in decisions["head_decisions"].values():
            decision["challenger_advanced"] = False
            decision["selected_head"] = "end2end"
        decisions["scale_decision"]["challenger_advanced"] = False
        decisions["scale_decision"]["selected_scale"] = "n"
        forged = candidates["n_end2end"]
        decisions["winner"] = {
            "slot": "n_end2end",
            "scale": "n",
            "head": "end2end",
            "pipeline": "primary",
            "candidate_content_sha256": forged["candidate_content_sha256"],
            "onnx_sha256": forged["onnx"]["sha256"],
            "validation_report_sha256": forged["validation_report_sha256"],
        }
        manifest["selection_content_sha256"] = (
            tournament_module._selection_content_hash(manifest)
        )
        self._write_json(manifest_path, manifest)

        with self.assertRaisesRegex(
            CandidateAdoptionError, "differs from deterministic report replay"
        ):
            self._adopt(validate_only=True)

    def test_rehashed_unrelated_sealed_plan_is_rejected(self) -> None:
        manifest_path = self.selection / "selection-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        plan_path = self.selection / manifest["sealed_inputs"]["plan"]["path"]
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["status"] = "UNRELATED_PLAN"
        self._write_json(plan_path, plan)
        plan_record = _file_record(plan_path)
        manifest["sealed_inputs"]["plan"].update(plan_record)
        manifest["plan"]["sha256"] = plan_record["sha256"]
        manifest["selection_content_sha256"] = (
            tournament_module._selection_content_hash(manifest)
        )
        self._write_json(manifest_path, manifest)

        with self.assertRaisesRegex(
            CandidateAdoptionError, "plan schema/status"
        ):
            self._adopt(validate_only=True)

    def test_winner_training_results_must_equal_staged_candidate(self) -> None:
        manifest_path = self.selection / "selection-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sealed = manifest["sealed_inputs"]["training_results"]
        n_results = self.selection / sealed["n"]["path"]
        s_results = self.selection / sealed["s"]["path"]
        self.assertNotEqual(n_results.read_bytes(), s_results.read_bytes())
        s_results.write_bytes(n_results.read_bytes())
        replacement = _file_record(s_results)
        sealed["s"].update(replacement)
        for slot_name in ("s_end2end", "s_traditional"):
            manifest["candidates"][slot_name]["training_identity"][
                "training_results_sha256"
            ] = replacement["sha256"]
        manifest["selection_content_sha256"] = (
            tournament_module._selection_content_hash(manifest)
        )
        self._write_json(manifest_path, manifest)

        with self.assertRaisesRegex(
            CandidateAdoptionError, "exact sealed tournament winner|training results"
        ):
            self._adopt(validate_only=True)

    def test_candidate_gates_and_mid_validation_mutation_fail_before_publication(self) -> None:
        manifest_path = self.candidate / "candidate-manifest.json"
        original_bytes = manifest_path.read_bytes()
        original = json.loads(original_bytes)
        for release_gate in (
            {**original["release_gate"], "approved": True},
            {key: 0 for key in QUALIFICATION_RECORD},
        ):
            manifest = deepcopy(original)
            manifest["release_gate"] = release_gate
            manifest["candidate_content_sha256"] = _manifest_content_hash(manifest)
            self._write_json(manifest_path, manifest)
            with self.subTest(release_gate=release_gate), self.assertRaisesRegex(
                CandidateAdoptionError, "gates"
            ):
                self._adopt(validate_only=True)
        manifest_path.write_bytes(original_bytes)

        candidate_onnx = next(self.candidate.glob("*.onnx"))
        candidate_onnx_bytes = candidate_onnx.read_bytes()

        def mutating_validator(candidate: Path) -> dict[str, object]:
            result = self._fake_validator(candidate)
            onnx = next(candidate.glob("*.onnx"))
            onnx.write_bytes(b"mutated after validation")
            return result

        with self.assertRaisesRegex(
            CandidateAdoptionError, "changed (?:while copying|before publication)"
        ):
            self._adopt(candidate_validator=mutating_validator)
        candidate_onnx.write_bytes(candidate_onnx_bytes)
        self.assertEqual(
            (self.root / "models" / "RELEASE-DEFAULT.json").read_bytes(),
            self.pointer_before,
        )

        original_snapshot_check = adoption_module._assert_inputs_unchanged
        privacy_source = self.root / "utils" / "public_evidence.py"

        def mutate_privacy_source(snapshots: object) -> None:
            self.assertIn(privacy_source.resolve(), snapshots)  # type: ignore[operator]
            original = privacy_source.read_bytes()
            privacy_source.write_bytes(original + b"\n# transient mutation\n")
            try:
                original_snapshot_check(snapshots)  # type: ignore[arg-type]
            finally:
                privacy_source.write_bytes(original)

        with (
            mock.patch.object(
                adoption_module,
                "_assert_inputs_unchanged",
                side_effect=mutate_privacy_source,
            ),
            self.assertRaisesRegex(
                CandidateAdoptionError,
                "changed before publication: public_evidence.py",
            ),
        ):
            self._adopt()
        self.assertEqual(
            (self.root / "models" / "RELEASE-DEFAULT.json").read_bytes(),
            self.pointer_before,
        )

    def test_transient_candidate_comparison_and_sealed_input_copy_mutation_is_rejected(
        self,
    ) -> None:
        original_copy = adoption_module._copy_verified_new
        targets = {
            "candidate": next(self.candidate.glob("*.onnx")),
            "comparison": next(
                (self.selection / "comparisons").glob("*/comparison.json")
            ),
            "sealed_input": self.selection / "inputs" / "tournament-plan.json",
        }
        for scenario, target in targets.items():
            target = target.resolve()

            def transient_copy(
                source: Path,
                destination: Path,
                **kwargs: object,
            ) -> None:
                if source.resolve() != target:
                    original_copy(source, destination, **kwargs)  # type: ignore[arg-type]
                    return
                original = source.read_bytes()
                source.write_bytes(b"transient unvalidated replacement")
                try:
                    original_copy(source, destination, **kwargs)  # type: ignore[arg-type]
                finally:
                    source.write_bytes(original)

            with (
                self.subTest(scenario=scenario),
                mock.patch.object(
                    adoption_module,
                    "_copy_verified_new",
                    side_effect=transient_copy,
                ),
                self.assertRaisesRegex(CandidateAdoptionError, "changed while copying"),
            ):
                self._adopt()
            self.assertEqual(
                (self.root / "models" / "RELEASE-DEFAULT.json").read_bytes(),
                self.pointer_before,
            )
            self.assertEqual(
                (self.root / "models" / "RELEASE-MANIFEST.sha256").read_bytes(),
                self.manifest_before,
            )

    def test_contract_rejects_false_like_flags_bad_detail_and_nonportable_paths(self) -> None:
        pointer = json.loads(self.pointer_before)
        pointer["qualification"] = {key: 0 for key in QUALIFICATION_RECORD}
        pointer["content_sha256"] = contract_content_hash(pointer)
        with self.assertRaisesRegex(ReleaseModelContractError, "qualification"):
            validate_release_default_contract(pointer)

        pointer = json.loads(self.pointer_before)
        pointer["detail_crop_size_source_pixels"] = True
        pointer["content_sha256"] = contract_content_hash(pointer)
        with self.assertRaisesRegex(ReleaseModelContractError, "detail crop"):
            validate_release_default_contract(pointer)

        pointer = json.loads(self.pointer_before)
        pointer["artifacts"]["onnx"]["path"] = "models/CON/file.onnx"
        pointer["content_sha256"] = contract_content_hash(pointer)
        with self.assertRaisesRegex(ReleaseModelContractError, "non-portable"):
            validate_release_default_contract(pointer)

    def test_contract_self_hash_excludes_only_the_self_field(self) -> None:
        pointer = json.loads(self.pointer_before)
        self.assertEqual(pointer["content_sha256"], contract_content_hash(pointer))
        pointer["content_sha256"] = "0" * 64
        self.assertNotEqual(pointer["content_sha256"], contract_content_hash(pointer))

    def test_public_receipt_projection_rejects_embedded_local_path_forms(self) -> None:
        for value in (
            "loaded_from=/tmp/private/model.onnx",
            "loaded_from=C:/private/model.onnx",
            "loaded_from=\\\\server\\share\\model.onnx",
            "source=file:///tmp/private/model.onnx",
            "source=file://server/share/model.onnx",
            "source=file:/tmp/private/model.onnx",
            "origin->/tmp/private/model.onnx",
            "loaded_from@/tmp/private/model.onnx",
            "query?local=/tmp/private/model.onnx",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                CandidateAdoptionError, "unsafe or local path-like"
            ):
                adoption_module._public_value(value, "fixture")
        with self.assertRaisesRegex(CandidateAdoptionError, "unsafe or local"):
            adoption_module._public_value(
                {"/tmp/private/model.onnx": "safe"}, "fixture"
            )
        csv_path = self.root / "private-header.csv"
        csv_path.write_text("epoch,/tmp/private/model\n1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(CandidateAdoptionError, "private or absolute"):
            adoption_module._assert_public_text_safe(csv_path, "fixture CSV")
        with self.assertRaisesRegex(CandidateAdoptionError, "header differs"):
            adoption_module._validated_training_results_contract(
                csv_path, "fixture CSV"
            )
        self.assertEqual(
            adoption_module._public_value(
                "https://example.com/public/model", "fixture"
            ),
            "https://example.com/public/model",
        )
        self.assertEqual(
            adoption_module._public_value(
                "inputs/runtime/n_end2end/validation-metrics.json", "fixture"
            ),
            "inputs/runtime/n_end2end/validation-metrics.json",
        )

    def test_rehashed_pointer_cannot_hide_semantic_or_privacy_receipt_tamper(
        self,
    ) -> None:
        self._adopt()
        pointer_path = self.root / "models" / "RELEASE-DEFAULT.json"
        pointer_bytes = pointer_path.read_bytes()
        pointer = json.loads(pointer_bytes)
        receipt_path = self.root / pointer["artifacts"]["candidate_receipt"]["path"]
        receipt_bytes = receipt_path.read_bytes()

        for scenario in ("winner_identity", "embedded_private_path"):
            receipt = json.loads(receipt_bytes)
            if scenario == "winner_identity":
                receipt["configuration"]["head"] = "end2end"
                expected = "candidate receipt does not bind"
            else:
                receipt["exporter"]["logical_name"] = (
                    "loaded_from=/tmp/private/exporter.py"
                )
                expected = "private or absolute path-like"
            receipt.pop("content_sha256")
            receipt["content_sha256"] = canonical_hash(receipt)
            self._write_json(receipt_path, receipt)
            changed_pointer = json.loads(pointer_bytes)
            changed_pointer["artifacts"]["candidate_receipt"].update(
                _file_record(receipt_path)
            )
            changed_pointer["content_sha256"] = contract_content_hash(
                changed_pointer
            )
            self._write_json(pointer_path, changed_pointer)
            with self.subTest(scenario=scenario), self.assertRaisesRegex(
                ReleaseModelContractError, expected
            ):
                load_release_default_contract(self.root, verify_files=True)
            receipt_path.write_bytes(receipt_bytes)
            pointer_path.write_bytes(pointer_bytes)

    def test_rehashed_adoption_cannot_rebind_privacy_validator_revision(self) -> None:
        self._adopt()
        pointer_path = self.root / "models" / "RELEASE-DEFAULT.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        adoption_path = self.root / pointer["artifacts"]["adoption_record"]["path"]
        adoption = json.loads(adoption_path.read_text(encoding="utf-8"))
        adoption["source"]["public_evidence_sha256"] = "0" * 64
        adoption.pop("content_sha256")
        adoption["content_sha256"] = canonical_hash(adoption)
        self._write_json(adoption_path, adoption)
        pointer["artifacts"]["adoption_record"].update(_file_record(adoption_path))
        pointer["content_sha256"] = contract_content_hash(pointer)
        self._write_json(pointer_path, pointer)

        with self.assertRaisesRegex(
            ReleaseModelContractError,
            "privacy validator differs",
        ):
            load_release_default_contract(self.root, verify_files=True)


if __name__ == "__main__":
    unittest.main()
