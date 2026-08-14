from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.export_fort_release_candidate import (
    ATTRIBUTION_NAME,
    CandidateExportError,
    DEFAULT_BASENAME,
    LABELS_NAME,
    MANIFEST_NAME,
    STAGING_MARKER,
    _basename,
    _completed_results_epoch,
    _manifest_content_hash,
    build_parser,
    main,
    stage_candidate,
    validate_runtime_parity,
    validate_staged_candidate,
)
from scripts.fort_dataset_contract import GROUPED_DATASET_YAML, build_dataset_contract
from scripts.train_fort_model import (
    ResumeCheckpoint,
    TrainingConfig,
    _initial_run_contract,
    _write_reproducibility_record,
)


class _FakeDetector:
    def __init__(
        self,
        inference_size: tuple[int, int],
        *,
        delta: float = 0.0,
        shape: tuple[int, int, int] = (1, 4, 6),
    ) -> None:
        self.inference_size = inference_size
        self.delta = delta
        self.shape = shape
        self.runtime_summary = {
            "runtime": "fake",
            "declared_input_shape": [1, 3, *inference_size],
            "configured_input_shape": [1, 3, *inference_size],
            "output_shape": list(shape),
            "output_format": "end2end",
        }

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.shape != (1, 3, *self.inference_size):
            raise AssertionError("wrong seeded tensor shape")
        result = np.zeros(self.shape, dtype=np.float32)
        if self.shape[-1] == 6:
            result[..., 0] = 10.0 + self.delta
            result[..., 1] = 20.0
            result[..., 2] = 30.0
            result[..., 3] = 40.0
            result[..., 4] = np.array([0.5, 0.1, 0.0005, 0.0])[: self.shape[1]]
            result[..., 5] = 0.0
        return result


class FortReleaseCandidateExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self._dataset(self.root / "dataset")
        self.dataset_manifest_sha256 = sha256(
            (self.data.parent / "manifest.json").read_bytes()
        ).hexdigest()
        self.dataset_content_sha256 = build_dataset_contract(self.data.parent)[
            "content_sha256"
        ]
        self.initial_weights = self.root / "yolo26n.pt"
        self.initial_weights.write_bytes(b"initial pretrained weights")
        self.run_dir = self.root / "runs" / "reviewed_run"
        (self.run_dir / "weights").mkdir(parents=True)
        self.weights = self.run_dir / "weights" / "best.pt"
        self.weights.write_bytes(b"reviewed checkpoint")
        self.training_config = TrainingConfig(
            data=self.data,
            weights=self.initial_weights,
            project=self.run_dir.parent,
            name=self.run_dir.name,
            epochs=4,
            patience=2,
            batch=4,
            imgsz=640,
            device="cpu",
            workers=1,
            threads=2,
            cache="none",
            seed=0,
            smoke_test=False,
            run_test=False,
        )
        dataset_manifest = json.loads(
            (self.data.parent / "manifest.json").read_text(encoding="utf-8")
        )
        (self.run_dir / "initial_run_contract.json").write_text(
            json.dumps(_initial_run_contract(self.training_config, dataset_manifest)),
            encoding="utf-8",
        )
        (self.run_dir / "results.csv").write_text(
            "epoch,metrics/mAP50(B)\n1,0.5\n", encoding="utf-8"
        )
        _write_reproducibility_record(
            self.training_config,
            dataset_manifest,
            self.weights,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _dataset(root: Path) -> Path:
        for split in ("train", "valid", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            (root / "images" / split / f"{split}.jpg").write_bytes(
                f"{split} image".encode()
            )
            (root / "labels" / split / f"{split}.txt").write_text(
                "0 0.5 0.5 0.2 0.1\n", encoding="utf-8"
            )
        data = root / "fort_cuh_grouped.yaml"
        data.write_text(GROUPED_DATASET_YAML, encoding="utf-8")
        (root / "labels.txt").write_text("player\n", encoding="utf-8")
        (root / ATTRIBUTION_NAME).write_text(
            "FORT-Cuh v1 by Aviles Joseph\n"
            "https://creativecommons.org/licenses/by/4.0/\n"
            "independent\n  target-clone footage is required.\n",
            encoding="utf-8",
        )
        contract = build_dataset_contract(root)
        manifest = {
            "schema_version": 1,
            "cross_split_source_groups": 0,
            "cross_split_visual_similarity_edges": 0,
            "runtime_class_labels": ["player"],
            "source_archive_sha256": "a" * 64,
            "source": {
                "dataset": "FORT-Cuh v1",
                "license": "CC BY 4.0",
                "url": "https://example.invalid/fort",
            },
            "dataset_contract": contract,
            "reviewed_negative_images": [],
            "splits": {
                split: {
                    "images": contract["splits"][split]["images"],
                    "boxes": contract["splits"][split]["boxes"],
                }
                for split in ("train", "valid", "test")
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        return data

    @staticmethod
    def _detectors(**kwargs: object) -> tuple[_FakeDetector, _FakeDetector]:
        size = kwargs["inference_size"]
        assert isinstance(size, tuple)
        return _FakeDetector(size), _FakeDetector(size, delta=0.0001)

    @staticmethod
    def _fake_yolo_factory(work_root: Path):
        class FakeYOLO:
            task = "detect"
            names = {0: "player"}
            model = SimpleNamespace(end2end=True)

            def __init__(self, source: str) -> None:
                self.source = source

            def export(self, **kwargs: object) -> str:
                source = Path(self.source)
                if source.parent != work_root:
                    raise AssertionError("export did not use a private checkpoint copy")
                if kwargs["dynamic"] is not False or kwargs["batch"] != 1:
                    raise AssertionError("export was not exact static batch one")
                if kwargs["format"] == "onnx":
                    result = work_root / "reviewed.onnx"
                    result.write_bytes(b"onnx graph")
                    return str(result)
                result = work_root / "reviewed_openvino_model"
                result.mkdir()
                (result / "reviewed.xml").write_bytes(b"openvino graph")
                (result / "reviewed.bin").write_bytes(b"openvino weights")
                (result / "metadata.yaml").write_text(
                    "task: detect\nnames:\n  0: player\nend2end: true\n",
                    encoding="utf-8",
                )
                return str(result)

        return FakeYOLO

    def _stage(self, output: Path) -> dict[str, object]:
        # The exporter owns this parent-scoped temporary directory name. Patch
        # the factory lazily by deriving it from the private checkpoint path.
        factories: list[type] = []

        def factory(source: str):
            cls = self._fake_yolo_factory(Path(source).parent)
            factories.append(cls)
            return cls(source)

        return stage_candidate(
            weights=self.weights,
            data=self.data,
            output=output,
            inference_size=(384, 640),
            yolo_factory=factory,
            detector_factory=self._detectors,
            expected_manifest_sha256=self.dataset_manifest_sha256,
            expected_content_sha256=self.dataset_content_sha256,
        )

    def test_stages_both_formats_provenance_attribution_and_validates_again(self) -> None:
        output = self.root / "candidate"

        manifest = self._stage(output)

        self.assertTrue((output / f"{DEFAULT_BASENAME}.onnx").is_file())
        self.assertTrue((output / f"{DEFAULT_BASENAME}.xml").is_file())
        self.assertTrue((output / f"{DEFAULT_BASENAME}.bin").is_file())
        self.assertEqual((output / LABELS_NAME).read_text(), "player\n")
        self.assertFalse((output / STAGING_MARKER).exists())
        self.assertFalse((self.weights.parent / "best.onnx").exists())
        self.assertEqual(manifest["configuration"]["input_shape_nchw"], [1, 3, 384, 640])
        self.assertEqual(manifest["configuration"]["head"], "end2end")
        self.assertEqual(manifest["dataset"]["content_sha256"], build_dataset_contract(self.data.parent)["content_sha256"])
        self.assertEqual(manifest["checkpoint"]["sha256"], sha256(self.weights.read_bytes()).hexdigest())
        self.assertFalse(manifest["release_gate"]["approved"])
        attribution = (output / ATTRIBUTION_NAME).read_text()
        self.assertIn("CC BY 4.0", attribution)
        self.assertIn("AGPL-3.0", attribution)
        self.assertIn("test split", attribution)

        result = validate_staged_candidate(output, detector_factory=self._detectors)

        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["parity"]["seeded_tensor_count"], 4)

    def test_refuses_existing_output_symlink_checkpoint_and_dataset_tamper(self) -> None:
        output = self.root / "candidate"
        output.mkdir()
        with self.assertRaisesRegex(CandidateExportError, "already exists"):
            self._stage(output)

        link = self.root / "linked.pt"
        link.symlink_to(self.weights)
        with self.assertRaisesRegex(CandidateExportError, "symlink component"):
            stage_candidate(
                weights=link,
                data=self.data,
                output=self.root / "linked-output",
                yolo_factory=lambda _source: None,
                detector_factory=self._detectors,
                expected_manifest_sha256=self.dataset_manifest_sha256,
                expected_content_sha256=self.dataset_content_sha256,
            )

        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(CandidateExportError, "symlink component"):
            stage_candidate(
                weights=self.weights,
                data=self.data,
                output=linked_parent / "candidate",
                yolo_factory=lambda _source: None,
                detector_factory=self._detectors,
                expected_manifest_sha256=self.dataset_manifest_sha256,
                expected_content_sha256=self.dataset_content_sha256,
            )

        (self.data.parent / "labels" / "valid" / "valid.txt").write_text(
            "0 0.4 0.5 0.2 0.1\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(CandidateExportError, "exact-file audit failed"):
            stage_candidate(
                weights=self.weights,
                data=self.data,
                output=self.root / "tampered-output",
                yolo_factory=lambda _source: None,
                detector_factory=self._detectors,
                expected_manifest_sha256=self.dataset_manifest_sha256,
                expected_content_sha256=self.dataset_content_sha256,
            )

    def test_export_failure_leaves_no_output_or_partial_staging_directory(self) -> None:
        output = self.root / "candidate"

        class BrokenYOLO:
            task = "detect"
            names = {0: "player"}
            model = SimpleNamespace(end2end=True)

            def __init__(self, _source: str) -> None:
                pass

            def export(self, **_kwargs: object) -> str:
                raise RuntimeError("synthetic export failure")

        with self.assertRaisesRegex(CandidateExportError, "synthetic export failure"):
            stage_candidate(
                weights=self.weights,
                data=self.data,
                output=output,
                yolo_factory=BrokenYOLO,
                detector_factory=self._detectors,
                expected_manifest_sha256=self.dataset_manifest_sha256,
                expected_content_sha256=self.dataset_content_sha256,
            )

        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".candidate.staging-*")), [])

    def test_default_audit_pin_refuses_a_different_valid_grouped_dataset(self) -> None:
        with self.assertRaisesRegex(CandidateExportError, "explicitly reviewed manifest"):
            stage_candidate(
                weights=self.weights,
                data=self.data,
                output=self.root / "not-v9",
                yolo_factory=lambda _source: None,
                detector_factory=self._detectors,
            )

    def test_refuses_wrong_task_classes_and_unsupported_end2end_selection(self) -> None:
        cases = (
            (SimpleNamespace(task="segment", names={0: "player"}, model=SimpleNamespace(end2end=True)), "task must be detect"),
            (SimpleNamespace(task="detect", names={0: "person"}, model=SimpleNamespace(end2end=True)), "exactly class 0='player'"),
            (SimpleNamespace(task="detect", names={0: "player"}, model=object()), "does not expose"),
        )
        for index, (model, message) in enumerate(cases):
            with self.subTest(message=message):
                with self.assertRaisesRegex(CandidateExportError, message):
                    stage_candidate(
                        weights=self.weights,
                        data=self.data,
                        output=self.root / f"bad-{index}",
                        yolo_factory=lambda _source, model=model: model,
                        detector_factory=self._detectors,
                        expected_manifest_sha256=self.dataset_manifest_sha256,
                        expected_content_sha256=self.dataset_content_sha256,
                    )

    def test_refuses_checkpoint_without_completed_run_provenance(self) -> None:
        arbitrary = self.root / "best.pt"
        arbitrary.write_bytes(b"unbound checkpoint")
        with self.assertRaisesRegex(CandidateExportError, "exact weights/best.pt"):
            stage_candidate(
                weights=arbitrary,
                data=self.data,
                output=self.root / "unbound-output",
                yolo_factory=lambda _source: None,
                detector_factory=self._detectors,
                expected_manifest_sha256=self.dataset_manifest_sha256,
                expected_content_sha256=self.dataset_content_sha256,
            )

    def test_training_results_columns_must_be_portable_metric_names(self) -> None:
        results = self.root / "unsafe-results.csv"
        for header in (
            "epoch,/tmp/private/model.onnx",
            r"epoch,C:\\Users\\private\\model.onnx",
            "epoch,metrics/../private",
            "epoch,metric=value",
        ):
            with self.subTest(header=header):
                results.write_text(f"{header}\n1,0.5\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    CandidateExportError, "unsafe or non-portable columns"
                ):
                    _completed_results_epoch(results)

        results.write_text(
            "epoch,metrics/mAP50(B),train/box_loss,lr/pg0\n1,0.5,1.0,0.001\n",
            encoding="utf-8",
        )
        self.assertEqual(_completed_results_epoch(results), (1, 1))

    def test_refuses_tampered_source_training_provenance(self) -> None:
        path = self.run_dir / "reproducibility.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["output"]["best_weights_sha256"] = "0" * 64
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(CandidateExportError, "best checkpoint hash/size"):
            self._stage(self.root / "tampered-provenance")

    def test_explicit_adoption_contract_is_bound_to_stateful_resume(self) -> None:
        contract_path = self.run_dir / "initial_run_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        adopted_bytes = b"adopted epoch checkpoint"
        adopted_sha256 = sha256(adopted_bytes).hexdigest()
        epoch_copy = self.run_dir / "weights" / "epoch0.pt"
        epoch_copy.write_bytes(adopted_bytes)
        last = self.run_dir / "weights" / "last.pt"
        last.write_bytes(b"later mutable last checkpoint")
        contract["training_script_sha256"] = None
        contract["training_script_sha256_status"] = "unavailable_pre_contract_run"
        contract["adoption"] = {
            "adopted_checkpoint_epoch": 1,
            "adopted_checkpoint_sha256": adopted_sha256,
            "epoch_checkpoint_sha256": adopted_sha256,
            "adoption_script_sha256": "a" * 64,
            "checkpoint_version": contract["environment"]["ultralytics"],
        }
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        (self.run_dir / "results.csv").write_text(
            "epoch,metrics/mAP50(B)\n1,0.5\n2,0.6\n", encoding="utf-8"
        )
        resumed_config = replace(self.training_config, resume_from=last.resolve())
        resume_state = ResumeCheckpoint(
            path=last.resolve(),
            sha256=adopted_sha256,
            completed_epoch=1,
            initial_weights=self.initial_weights,
            initial_weights_sha256=contract["initial_weights_sha256"],
            dataset_manifest_sha256=contract["dataset_manifest_sha256"],
            dataset_content_sha256=contract["dataset_content_sha256"],
            dataset_yaml_sha256=contract["dataset_yaml_sha256"],
            initial_training_script_sha256="unavailable_pre_contract_run",
        )
        dataset_manifest = json.loads(
            (self.data.parent / "manifest.json").read_text(encoding="utf-8")
        )
        _write_reproducibility_record(
            resumed_config,
            dataset_manifest,
            self.weights,
            resume_state=resume_state,
        )

        output = self.root / "adopted-candidate"
        manifest = self._stage(output)
        self.assertTrue(manifest["training_provenance"]["adopted_interrupted_run"])
        self.assertEqual(
            validate_staged_candidate(output, detector_factory=self._detectors)["status"],
            "validated",
        )

    def test_parity_requires_multiple_distinct_seeds_and_fails_on_difference(self) -> None:
        artifacts = {
            "onnx": self.root / "a.onnx",
            "openvino_xml": self.root / "a.xml",
            "openvino_bin": self.root / "a.bin",
        }
        for path in artifacts.values():
            path.write_bytes(b"x")
        labels = self.root / "labels.txt"
        labels.write_text("player\n")

        with self.assertRaisesRegex(CandidateExportError, "at least three distinct"):
            validate_runtime_parity(
                artifacts=artifacts,
                labels=labels,
                inference_size=(384, 640),
                head="end2end",
                seeds=(0, 0, 1),
                confidence_floor=0.001,
                atol=0.002,
                rtol=0.0001,
                detector_factory=self._detectors,
            )

        def divergent(**kwargs: object) -> tuple[_FakeDetector, _FakeDetector]:
            size = kwargs["inference_size"]
            assert isinstance(size, tuple)
            return _FakeDetector(size), _FakeDetector(size, delta=1.0)

        with self.assertRaisesRegex(CandidateExportError, "numerical parity failed"):
            validate_runtime_parity(
                artifacts=artifacts,
                labels=labels,
                inference_size=(384, 640),
                head="end2end",
                seeds=(0, 1, 2),
                confidence_floor=0.001,
                atol=0.002,
                rtol=0.0001,
                detector_factory=divergent,
            )

        def wrong_dtype(**kwargs: object) -> tuple[_FakeDetector, _FakeDetector]:
            size = kwargs["inference_size"]
            assert isinstance(size, tuple)
            first = _FakeDetector(size)
            second = _FakeDetector(size)
            original = second.infer
            second.infer = lambda tensor: original(tensor).astype(np.float64)  # type: ignore[method-assign]
            return first, second

        with self.assertRaisesRegex(CandidateExportError, "float32 dtype contract differs"):
            validate_runtime_parity(
                artifacts=artifacts,
                labels=labels,
                inference_size=(384, 640),
                head="end2end",
                seeds=(0, 1, 2),
                confidence_floor=0.001,
                atol=0.002,
                rtol=0.0001,
                detector_factory=wrong_dtype,
            )

    def test_validate_only_rejects_artifact_tamper_extra_member_and_manifest_tamper(self) -> None:
        for scenario in ("artifact", "extra", "manifest"):
            with self.subTest(scenario=scenario):
                output = self.root / f"candidate-{scenario}"
                self._stage(output)
                if scenario == "artifact":
                    (output / f"{DEFAULT_BASENAME}.onnx").write_bytes(b"tampered")
                    expected = "hash/size mismatch"
                elif scenario == "extra":
                    (output / "surprise.txt").write_text("unexpected")
                    expected = "member set mismatch"
                else:
                    manifest_path = output / MANIFEST_NAME
                    manifest = json.loads(manifest_path.read_text())
                    manifest["status"] = "approved"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    expected = "content hash mismatch"
                with self.assertRaisesRegex(CandidateExportError, expected):
                    validate_staged_candidate(output, detector_factory=self._detectors)

    def test_validate_only_rejects_semantic_provenance_tamper_even_when_rehashed(self) -> None:
        output = self.root / "candidate-provenance"
        self._stage(output)
        reproducibility_path = output / "training-reproducibility.json"
        reproducibility = json.loads(reproducibility_path.read_text(encoding="utf-8"))
        reproducibility["output"]["best_weights_sha256"] = "0" * 64
        reproducibility_path.write_text(json.dumps(reproducibility), encoding="utf-8")
        new_sha256 = sha256(reproducibility_path.read_bytes()).hexdigest()
        manifest_path = output / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["training_reproducibility"]["sha256"] = new_sha256
        manifest["artifacts"]["training_reproducibility"]["bytes"] = (
            reproducibility_path.stat().st_size
        )
        manifest["training_provenance"]["training_reproducibility_sha256"] = new_sha256
        manifest["candidate_content_sha256"] = _manifest_content_hash(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(CandidateExportError, "checkpoint evidence is inconsistent"):
            validate_staged_candidate(output, detector_factory=self._detectors)

    def test_validate_only_rejects_rehashed_approval_claim(self) -> None:
        output = self.root / "candidate-status"
        self._stage(output)
        manifest_path = output / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "approved"
        manifest["candidate_content_sha256"] = _manifest_content_hash(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(CandidateExportError, "must remain not approved"):
            validate_staged_candidate(output, detector_factory=self._detectors)

    def test_basename_rejects_paths_extensions_uppercase_and_empty(self) -> None:
        self.assertEqual(_basename("fort-player_v9"), "fort-player_v9")
        for invalid in ("", "../player", "player.onnx", "Player", "player model"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _basename(invalid)

    def test_manifest_hash_excludes_only_its_self_hash(self) -> None:
        value = {"schema_version": 1, "status": "candidate"}
        value["candidate_content_sha256"] = _manifest_content_hash(value)
        before = value["candidate_content_sha256"]
        value["candidate_content_sha256"] = "ignored self value"
        self.assertEqual(_manifest_content_hash(value), before)
        value["status"] = "changed"
        self.assertNotEqual(_manifest_content_hash(value), before)

    def test_validate_only_reuses_manifest_seeds_and_rejects_export_overrides(self) -> None:
        parsed = build_parser().parse_args(["--validate-only", "--output", "candidate"])
        self.assertIsNone(parsed.seeds)

        argv = [
            "export_fort_release_candidate.py",
            "--validate-only",
            "--output",
            str(self.root / "candidate"),
            "--inference-size",
            "416",
        ]
        with (
            mock.patch("sys.argv", argv),
            self.assertRaisesRegex(SystemExit, "cannot override stored shape"),
        ):
            main()


if __name__ == "__main__":
    unittest.main()
