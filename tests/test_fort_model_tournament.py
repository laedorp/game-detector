from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_fort_runtime_evaluations import _canonical_hash as comparator_hash
from scripts.evaluate_fort_runtime_model import _source_hash_snapshot
import scripts.run_fort_model_tournament as tournament_module
from scripts.run_fort_model_tournament import TournamentError, run_tournament


BUCKETS = [
    "ultra_far_le_32px",
    "far_33_to_64px",
    "medium_65_to_96px",
    "near_gt_96px",
]
DATASET_MANIFEST_SHA256 = "a" * 64
DATASET_CONTENT_SHA256 = "b" * 64
INITIAL_HASHES = {"n": "c" * 64, "s": "d" * 64}
CHECKPOINT_HASHES = {"n": "e" * 64, "s": "f" * 64}
RESULT_COLUMNS = (
    "epoch",
    "time",
    "train/box_loss",
    "train/cls_loss",
    "train/l1_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/l1_loss",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
)


def _file_record(path: Path) -> dict[str, object]:
    contents = path.read_bytes()
    return {
        "name": path.name,
        "bytes": len(contents),
        "sha256": sha256(contents).hexdigest(),
    }


def _candidate(root: Path, scale: str, head: str) -> tuple[Path, dict[str, object]]:
    directory = root / f"candidate-{scale}-{head}"
    directory.mkdir()
    basename = f"player-{scale}-{head}"
    members = {
        "onnx": (f"{basename}.onnx", f"onnx-{scale}-{head}".encode()),
        "openvino_xml": (f"{basename}.xml", f"xml-{scale}-{head}".encode()),
        "openvino_bin": (f"{basename}.bin", f"bin-{scale}-{head}".encode()),
        "labels": ("labels.txt", b"player\n"),
        "attribution": (
            "ATTRIBUTION.md",
            b"FORT-Cuh https://creativecommons.org/licenses/by/4.0 AGPL-3.0 test split\n",
        ),
        "initial_run_contract": (
            "training-initial-run-contract.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "initial_weights": f"/models/yolo26{scale}.pt",
                    "initial_weights_sha256": INITIAL_HASHES[scale],
                },
                sort_keys=True,
            ).encode(),
        ),
        "training_reproducibility": (
            "training-reproducibility.json",
            b'{"schema_version": 1}\n',
        ),
        "training_results": (
            "training-results.csv",
            (
                ",".join(RESULT_COLUMNS)
                + "\n"
                + "".join(
                    ",".join(
                        [str(epoch)]
                        + [str(0.5 if scale == "n" else 0.6)] * 14
                    )
                    + "\n"
                    for epoch in range(1, 21)
                )
            ).encode("utf-8"),
        ),
    }
    artifacts: dict[str, dict[str, object]] = {}
    for key, (name, contents) in members.items():
        path = directory / name
        path.write_bytes(contents)
        artifacts[key] = _file_record(path)

    provenance = {
        "schema_version": 1,
        "checkpoint_role": "completed_run_best",
        "planned_epochs": 60,
        "completed_epochs": 20,
        "results_rows": 20,
        "initial_weights_sha256": INITIAL_HASHES[scale],
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_content_sha256": DATASET_CONTENT_SHA256,
        "initial_run_contract_sha256": f"{1 if scale == 'n' else 2:064x}",
        "training_reproducibility_sha256": f"{3 if scale == 'n' else 4:064x}",
        "training_results_sha256": f"{5 if scale == 'n' else 6:064x}",
        "checkpoint_sha256": CHECKPOINT_HASHES[scale],
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "validated_release_candidate_not_approved",
        "checkpoint": {
            "path": f"/training/{scale}/weights/best.pt",
            "bytes": 1234,
            "sha256": CHECKPOINT_HASHES[scale],
        },
        "dataset": {
            "manifest_sha256": DATASET_MANIFEST_SHA256,
            "content_sha256": DATASET_CONTENT_SHA256,
        },
        "training_provenance": provenance,
        "configuration": {
            "basename": basename,
            "inference_size": "384x640",
            "input_shape_nchw": [1, 3, 384, 640],
            "head": head,
            "one_class": {"0": "player"},
        },
        "exporter": {
            "sha256": tournament_module._sha256_file(
                tournament_module.PROJECT_ROOT
                / "scripts"
                / "export_fort_release_candidate.py"
            )
        },
        "artifacts": artifacts,
        "parity": {
            "status": "passed",
            "input_shape_nchw": [1, 3, 384, 640],
            "output_layout": "end2end" if head == "end2end" else "traditional_bcn",
        },
        "release_gate": deepcopy(tournament_module.EXPECTED_RELEASE_GATE),
    }
    manifest["candidate_content_sha256"] = tournament_module._manifest_content_hash(
        manifest
    )
    manifest_path = directory / "candidate-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return directory, artifacts["onnx"]


def _pipeline_metrics(far_true_positives: int, image_count: int) -> dict[str, object]:
    records = []
    for index in range(image_count):
        targets = {
            "ultra_far_le_32px": 0,
            "far_33_to_64px": 1,
            "medium_65_to_96px": 1,
            "near_gt_96px": 1,
        }
        counts = {
            "ultra_far_le_32px": {"true_positives": 0, "false_positives": 0},
            "far_33_to_64px": {
                "true_positives": int(index < far_true_positives),
                "false_positives": 0,
            },
            "medium_65_to_96px": {"true_positives": 1, "false_positives": 0},
            "near_gt_96px": {"true_positives": 1, "false_positives": 0},
        }
        records.append(
            {
                "member_id": f"{index + 1:064x}",
                "source_group_id": f"{index + 10_000:064x}",
                "targets": targets,
                "operating_points": {
                    "0.25": deepcopy(counts),
                    "0.45": deepcopy(counts),
                },
            }
        )
    paired = {
        "schema_version": 1,
        "bucket_order": BUCKETS,
        "member_count": image_count,
        "member_sequence_sha256": comparator_hash(
            [record["member_id"] for record in records]
        ),
        "source_group_count": image_count,
        "source_group_sequence_sha256": comparator_hash(
            [record["source_group_id"] for record in records]
        ),
        "confidence_thresholds": [0.25, 0.45],
        "split_content_sha256": "9" * 64,
        "records": records,
    }
    bucket_points: dict[str, dict[str, int]] = {}
    for bucket in BUCKETS:
        bucket_points[bucket] = {
            "ground_truth_total": sum(record["targets"][bucket] for record in records),
            "detected_true_positives": sum(
                record["operating_points"]["0.25"][bucket]["true_positives"]
                for record in records
            ),
            "false_positives": 0,
        }
    aggregate = {
        "ground_truth_total": sum(item["ground_truth_total"] for item in bucket_points.values()),
        "detected_true_positives": sum(
            item["detected_true_positives"] for item in bucket_points.values()
        ),
        "false_positives": 0,
    }
    return {
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


def _report(
    root: Path,
    scale: str,
    head: str,
    onnx: dict[str, object],
    *,
    configured_far_true_positives: int,
    image_count: int,
) -> Path:
    directory = root / "reports" / f"{scale}-{head}"
    directory.mkdir(parents=True)
    source_identity = _source_hash_snapshot("onnxruntime")
    model_content_sha256 = comparator_hash(
        [{"name": onnx["name"], "sha256": onnx["sha256"]}]
    )
    configured = _pipeline_metrics(
        min(configured_far_true_positives, image_count), image_count
    )
    primary = _pipeline_metrics(0, image_count)
    report = {
        "schema_version": 4,
        "evaluator": {
            **source_identity["evaluator"],
            "pipeline_source_sha256": source_identity["pipeline"],
        },
        "model_artifact": {
            "backend": "onnxruntime",
            "entrypoint": onnx["name"],
            "entrypoint_sha256": onnx["sha256"],
            "members": [
                {
                    "name": onnx["name"],
                    "path": onnx["name"],
                    "bytes": onnx["bytes"],
                    "sha256": onnx["sha256"],
                }
            ],
            "content_sha256": model_content_sha256,
        },
        "dataset": {
            "yaml": "fort_cuh_grouped.yaml",
            "manifest_sha256": DATASET_MANIFEST_SHA256,
            "manifest": "manifest.json",
            "content_sha256": DATASET_CONTENT_SHA256,
        },
        "configuration": {
            "split": "val",
            "selection_role": "development_validation",
            "backend": "onnxruntime",
            "device": "CPU",
            "inference_size": "384x640",
            "input_shape_nchw": [1, 3, 384, 640],
            "declared_static_input_shape_nchw": [1, 3, 384, 640],
            "exact_static_deployment_shape": True,
            "output_format": head,
            "matching_iou_threshold": 0.5,
            "minimum_prediction_confidence": 0.001,
            "reported_confidence_thresholds": [0.25, 0.45],
            "runtime_nms_iou_threshold": 0.45,
            "require_full_provider": False,
            "detail_pass": {
                "enabled": True,
                "requested_crop_size_source_pixels": 768,
                "selection_split_policy": "validation_only",
                "test_split_evaluation_permitted": False,
            },
        },
        "runtime": {
            "summary": {
                "output_format": head,
                "output_layout": (
                    "end2end" if head == "end2end" else "traditional_bcn"
                ),
                "requested_device_input": "CPU",
                "requested_provider": "CPUExecutionProvider",
                "active_providers": ["CPUExecutionProvider"],
                "require_full_provider": False,
            }
        },
        "metrics": {
            "val": {
                "configured_pipeline": "full_frame_plus_center_detail_merged",
                **configured,
                "primary_full_frame_reference": primary,
            }
        },
        "qualification": {
            "status": "development_evidence_only",
            "independent_holdout_required": True,
        },
    }
    path = directory / "metrics.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def _fixture(
    root: Path,
    *,
    image_count: int = 30,
    levels: dict[tuple[str, str], int] | None = None,
) -> Path:
    levels = levels or {
        ("n", "end2end"): 12,
        ("n", "traditional"): 18,
        ("s", "end2end"): 24,
        ("s", "traditional"): 30,
    }
    slots: dict[tuple[str, str], tuple[Path, Path]] = {}
    for scale in ("n", "s"):
        for head in ("end2end", "traditional"):
            candidate, onnx = _candidate(root, scale, head)
            report = _report(
                root,
                scale,
                head,
                onnx,
                configured_far_true_positives=levels[(scale, head)],
                image_count=image_count,
            )
            slots[(scale, head)] = (candidate, report)
    plan = {
        "schema_version": 1,
        "status": "development_model_tournament_plan",
        "dataset": {
            "manifest_sha256": DATASET_MANIFEST_SHA256,
            "content_sha256": DATASET_CONTENT_SHA256,
        },
        "runtime": {
            "backend": "onnxruntime",
            "device": "CPU",
            "require_full_provider": False,
            "inference_size": "384x640",
            "detail_crop_size_source_pixels": 768,
            "confidence": 0.25,
            "comparator_bootstrap_samples": 2000,
        },
        "models": {},
    }
    for scale in ("n", "s"):
        plan["models"][scale] = {"initial_weights_sha256": INITIAL_HASHES[scale]}
        for head in ("end2end", "traditional"):
            candidate, report = slots[(scale, head)]
            plan["models"][scale][head] = {
                "candidate_dir": candidate.relative_to(root).as_posix(),
                "validation_report": report.relative_to(root).as_posix(),
            }
    path = root / "tournament-plan.json"
    path.write_bytes(tournament_module._canonical_bytes(plan))
    return path


class FortModelTournamentTests(unittest.TestCase):
    @staticmethod
    def _skip_packaged_validation(*_args: object, **_kwargs: object) -> None:
        return None

    def test_fixed_bracket_selects_s_traditional_detail_but_never_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _fixture(root)
            output = root / "selection"

            manifest = run_tournament(
                plan_path=plan,
                output=output,
                packaged_training_validator=self._skip_packaged_validation,
            )

            winner = manifest["development_selection"]["winner"]
            self.assertEqual(
                (winner["slot"], winner["pipeline"]),
                ("s_traditional", "configured"),
            )
            self.assertEqual(len(manifest["comparisons"]), 7)
            self.assertTrue(
                all(
                    item["challenger_advanced"]
                    for item in manifest["comparisons"].values()
                )
            )
            self.assertFalse(manifest["release_qualification"]["qualified"])
            self.assertFalse(manifest["release_qualification"]["release_model_approved"])
            self.assertFalse(manifest["test_data_policy"]["test_split_consumed"])
            self.assertEqual(
                json.loads((output / "selection-manifest.json").read_text()), manifest
            )
            self.assertEqual(
                manifest["selection_content_sha256"],
                tournament_module._selection_content_hash(manifest),
            )
            self.assertEqual(
                manifest["public_evidence_privacy"],
                {
                    "path": "public_evidence.py",
                    "sha256": tournament_module._sha256_file(
                        tournament_module.PROJECT_ROOT / "utils" / "public_evidence.py"
                    ),
                },
            )
            self.assertEqual(
                len(list((output / "comparisons").glob("*/comparison.json"))), 7
            )
            sealed = manifest["sealed_inputs"]
            self.assertEqual(set(sealed["runtime_reports"]), {
                "n_end2end",
                "n_traditional",
                "s_end2end",
                "s_traditional",
            })
            self.assertEqual(set(sealed["training_results"]), {"n", "s"})
            records = [
                sealed["plan"],
                *sealed["runtime_reports"].values(),
                *sealed["training_results"].values(),
            ]
            for record in records:
                path = output / record["path"]
                self.assertTrue(path.is_file())
                payload = path.read_bytes()
                self.assertEqual(len(payload), record["bytes"])
                self.assertEqual(sha256(payload).hexdigest(), record["sha256"])
            public_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*")
                if path.is_file()
            )
            self.assertNotIn(str(root), public_text)

    def test_plan_requires_canonical_json_before_any_evidence_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = _fixture(root)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            calls: list[object] = []

            with self.assertRaisesRegex(TournamentError, "canonical sorted JSON"):
                run_tournament(
                    plan_path=plan_path,
                    output=root / "selection",
                    comparator=lambda **kwargs: calls.append(kwargs),
                    packaged_training_validator=self._skip_packaged_validation,
                )

            self.assertEqual(calls, [])
            self.assertFalse((root / "selection").exists())

    def test_inadequate_far_evidence_aborts_and_removes_all_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _fixture(root, image_count=2)
            output = root / "selection"

            with self.assertRaisesRegex(TournamentError, "inadequate development evidence"):
                run_tournament(
                    plan_path=plan,
                    output=output,
                    packaged_training_validator=self._skip_packaged_validation,
                )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".selection.tournament-*")), [])

    def test_valid_non_improvement_keeps_every_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            no_far_gain = {
                (scale, head): 0
                for scale in ("n", "s")
                for head in ("end2end", "traditional")
            }
            manifest = run_tournament(
                plan_path=_fixture(root, levels=no_far_gain),
                output=root / "selection",
                packaged_training_validator=self._skip_packaged_validation,
            )

            winner = manifest["development_selection"]["winner"]
            self.assertEqual((winner["slot"], winner["pipeline"]), ("n_end2end", "primary"))
            self.assertFalse(
                any(
                    item["challenger_advanced"]
                    for item in manifest["comparisons"].values()
                )
            )

    def test_test_report_and_nonstandard_operating_contract_fail_before_comparison(self) -> None:
        for scenario in ("test", "confidence", "samples"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan_path = _fixture(root)
                plan = json.loads(plan_path.read_text())
                if scenario == "test":
                    report_path = root / plan["models"]["n"]["end2end"][
                        "validation_report"
                    ]
                    report = json.loads(report_path.read_text())
                    report["configuration"]["split"] = "test"
                    report["configuration"]["selection_role"] = "development_audit_only"
                    report_path.write_text(json.dumps(report), encoding="utf-8")
                elif scenario == "confidence":
                    plan["runtime"]["confidence"] = 0.45
                    plan_path.write_bytes(tournament_module._canonical_bytes(plan))
                else:
                    plan["runtime"]["comparator_bootstrap_samples"] = 1999
                    plan_path.write_bytes(tournament_module._canonical_bytes(plan))
                calls: list[object] = []

                with self.assertRaises(TournamentError):
                    run_tournament(
                        plan_path=plan_path,
                        output=root / "selection",
                        comparator=lambda **kwargs: calls.append(kwargs),
                        packaged_training_validator=self._skip_packaged_validation,
                    )

                self.assertEqual(calls, [])
                self.assertFalse((root / "selection").exists())

    def test_candidate_report_binding_and_existing_output_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = _fixture(root)
            plan = json.loads(plan_path.read_text())
            report_path = root / plan["models"]["s"]["traditional"]["validation_report"]
            report = json.loads(report_path.read_text())
            report["model_artifact"]["entrypoint_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(TournamentError, "not bound"):
                run_tournament(
                    plan_path=plan_path,
                    output=root / "selection",
                    packaged_training_validator=self._skip_packaged_validation,
                )

        for disclosure in (
            "loaded_from=/tmp/private/model.onnx",
            r"loaded_from=C:\\Users\\private\\model.onnx",
            r"loaded_from=\\\\workstation\\share\\model.onnx",
            "loaded_from=file:///tmp/private/model.onnx",
        ):
            with self.subTest(disclosure=disclosure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan_path = _fixture(root)
                plan = json.loads(plan_path.read_text())
                report_path = root / plan["models"]["n"]["end2end"][
                    "validation_report"
                ]
                report = json.loads(report_path.read_text())
                report["debug_origin"] = disclosure
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(TournamentError, "non-portable absolute path"):
                    run_tournament(
                        plan_path=plan_path,
                        output=root / "selection",
                        packaged_training_validator=self._skip_packaged_validation,
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = _fixture(root)
            plan = json.loads(plan_path.read_text())
            report_path = root / plan["models"]["n"]["end2end"][
                "validation_report"
            ]
            report = json.loads(report_path.read_text())
            report["/tmp/private/model.onnx"] = "safe-looking value"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(TournamentError, "non-portable absolute path"):
                run_tournament(
                    plan_path=plan_path,
                    output=root / "selection",
                    packaged_training_validator=self._skip_packaged_validation,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = _fixture(root)
            output = root / "selection"
            output.mkdir()
            with self.assertRaisesRegex(TournamentError, "already exists"):
                run_tournament(
                    plan_path=plan_path,
                    output=output,
                    packaged_training_validator=self._skip_packaged_validation,
                )

    def test_full_provider_requires_both_fallback_disable_proofs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = _fixture(root)
            plan = json.loads(plan_path.read_text())
            plan["runtime"]["require_full_provider"] = True
            plan_path.write_bytes(tournament_module._canonical_bytes(plan))
            report_path = root / plan["models"]["n"]["end2end"][
                "validation_report"
            ]
            report = json.loads(report_path.read_text())
            report["configuration"]["require_full_provider"] = True
            report["runtime"]["summary"]["require_full_provider"] = True
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(TournamentError, "fallback-disable proof"):
                run_tournament(
                    plan_path=plan_path,
                    output=root / "selection",
                    packaged_training_validator=self._skip_packaged_validation,
                )

    def test_duplicate_plan_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(TournamentError, "duplicate JSON key"):
                run_tournament(plan_path=plan, output=root / "selection")

    def test_plan_and_runtime_evidence_reject_private_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = _fixture(root)
            plan = json.loads(plan_path.read_text())
            plan["models"]["n"]["end2end"]["candidate_dir"] = str(
                (root / "candidate-n-end2end").resolve()
            )
            plan_path.write_bytes(tournament_module._canonical_bytes(plan))
            with self.assertRaisesRegex(TournamentError, "portable relative path"):
                    run_tournament(
                        plan_path=plan_path,
                        output=root / "selection",
                        packaged_training_validator=self._skip_packaged_validation,
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = _fixture(root)
            plan = json.loads(plan_path.read_text())
            report_path = root / plan["models"]["n"]["end2end"][
                "validation_report"
            ]
            report = json.loads(report_path.read_text())
            report["public_reference"] = "https://example.com/release/model"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            result = run_tournament(
                plan_path=plan_path,
                output=root / "selection",
                packaged_training_validator=self._skip_packaged_validation,
            )
            self.assertEqual(result["status"], "development_model_selection_only")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = _fixture(root)
            plan = json.loads(plan_path.read_text())
            report_path = root / plan["models"]["n"]["end2end"][
                "validation_report"
            ]
            report = json.loads(report_path.read_text())
            report["model_artifact"]["entrypoint"] = str(
                (root / "private-player.onnx").resolve()
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(TournamentError, "non-portable absolute path"):
                run_tournament(
                    plan_path=plan_path,
                    output=root / "selection",
                    packaged_training_validator=self._skip_packaged_validation,
                )


if __name__ == "__main__":
    unittest.main()
