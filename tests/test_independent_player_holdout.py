from __future__ import annotations

from hashlib import sha256
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from scripts import prepare_independent_player_holdout as holdout_module
from scripts.prepare_independent_player_holdout import (
    GateMinimums,
    HoldoutContractError,
    MANIFEST_NAME,
    PINNED_RELEASE_MINIMUMS,
    POOL_DEVELOPMENT,
    POOL_SEALED,
    complete_sealed_evaluation,
    prepare_holdout,
    record_sealed_consumption,
    retire_sealed_holdout,
    verify_holdout,
)


SMALL_GATES = GateMinimums(
    target_le_32=1,
    target_33_64=1,
    target_65_96=1,
    target_gt_96=1,
    reviewed_negatives=1,
)
NOW = "2026-08-13T12:00:00Z"


def _utc_clock(*values: str):
    timestamps = iter(
        datetime.fromisoformat(value.replace("Z", "+00:00")) for value in values
    )
    return lambda: next(timestamps)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class HoldoutFixture:
    def __init__(
        self,
        root: Path,
        *,
        package_id: str = "holdout-a",
        pool: str = POOL_DEVELOPMENT,
        session_id: str = "session-a",
        seed: int = 1,
        image_height: int = 128,
    ) -> None:
        self.root = root
        self.package_id = package_id
        self.pool = pool
        self.session_id = session_id
        root.mkdir(parents=True)
        (root / "media").mkdir()
        (root / "capture.bin").write_bytes(f"offline-capture-{seed}".encode())
        (root / "LICENSE.txt").write_text(
            "Owned local capture; evaluation use permitted.\n", encoding="utf-8"
        )
        (root / "capture-tool.bin").write_bytes(f"capture-tool-{seed}".encode())
        (root / "capture-config.json").write_text(
            json.dumps({"seed": seed}, sort_keys=True), encoding="utf-8"
        )
        (root / "review-protocol.txt").write_text(
            "Two reviewers inspect every no-box frame; disagreements are adjudicated.\n",
            encoding="utf-8",
        )
        self.image_height = image_height
        self._make_image(root / "media" / "positive.png", seed, image_height)
        self._make_image(root / "media" / "negative.png", seed + 31, image_height)
        self.coco = {
            "images": [
                {
                    "id": 1,
                    "file_name": "media/positive.png",
                    "width": 128,
                    "height": image_height,
                    "session_id": session_id,
                    "source_frame_index": 10,
                    "sha256": _sha(root / "media" / "positive.png"),
                },
                {
                    "id": 2,
                    "file_name": "media/negative.png",
                    "width": 128,
                    "height": image_height,
                    "session_id": session_id,
                    "source_frame_index": 20,
                    "sha256": _sha(root / "media" / "negative.png"),
                    "negative_review": {
                        "reviewer_1": {
                            "reviewer_id": "reviewer-a",
                            "decision": "negative",
                            "reviewed_at_utc": NOW,
                        },
                        "reviewer_2": {
                            "reviewer_id": "reviewer-b",
                            "decision": "negative",
                            "reviewed_at_utc": NOW,
                        },
                        "adjudication": {"status": "not_required"},
                    },
                },
            ],
            "annotations": [
                self._box(1, 20, 0, image_height),
                self._box(2, 40, 20, image_height),
                self._box(3, 80, 30, image_height),
                self._box(4, 100, 20, image_height),
            ],
            "categories": [{"id": 1, "name": "player"}],
        }
        self.manifest_path = root / "input.json"
        self.write()

    @staticmethod
    def _make_image(path: Path, seed: int, height: int) -> None:
        image = np.zeros((height, 128, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(128):
                image[y, x] = (
                    (x * 7 + y * 3 + seed * 17) % 256,
                    (x * 2 + y * 11 + seed * 29) % 256,
                    (x * 13 + y * 5 + seed * 37) % 256,
                )
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to create test image: {path}")

    @staticmethod
    def _box(
        annotation_id: int,
        projected_height: int,
        y: int,
        source_height: int,
    ) -> dict[str, object]:
        height = projected_height * source_height / 1080
        return {
            "id": annotation_id,
            "image_id": 1,
            "category_id": 1,
            "bbox": [5, y, 10, height],
            "area": 10 * height,
            "iscrowd": 0,
            "ignore": 0,
            "occluded": False,
            "truncated": False,
        }

    def manifest(self) -> dict[str, object]:
        annotations = self.root / "annotations.json"
        _write_json(annotations, self.coco)
        return {
            "schema_version": 1,
            "package_id": self.package_id,
            "pool": self.pool,
            "sessions": [
                {
                    "session_id": self.session_id,
                    "assigned_pool": self.pool,
                    "captured_at_utc": NOW,
                    "source": {
                        "kind": "video",
                        "path": "capture.bin",
                        "sha256": _sha(self.root / "capture.bin"),
                    },
                    "license": {
                        "path": "LICENSE.txt",
                        "sha256": _sha(self.root / "LICENSE.txt"),
                        "identifier": "owned-local-capture",
                        "authorization_basis": "captured and supplied by the owner",
                        "holdout_use_permitted": True,
                        "redistribution_permitted": False,
                    },
                    "capture_environment_commit": f"{self.package_id.encode().hex()[0]:0<1}" * 40,
                    "acquisition": {
                        "tool_name": "offline-capture-tool",
                        "tool_path": "capture-tool.bin",
                        "tool_sha256": _sha(self.root / "capture-tool.bin"),
                        "config_path": "capture-config.json",
                        "config_sha256": _sha(self.root / "capture-config.json"),
                        "operator_id": "operator-a",
                    },
                }
            ],
            "annotations": {"path": "annotations.json", "sha256": _sha(annotations)},
            "human_review": {
                "annotation_author_ids": ["annotator-a"],
                "reviewer_ids": ["reviewer-a", "reviewer-b"],
                "adjudicator_ids": ["adjudicator-a"],
                "completed_at_utc": NOW,
                "protocol_path": "review-protocol.txt",
                "protocol_sha256": _sha(self.root / "review-protocol.txt"),
            },
        }

    def write(self) -> None:
        _write_json(self.manifest_path, self.manifest())

    def mutate_image_one_pixel(self) -> None:
        for name in ("positive.png", "negative.png"):
            path = self.root / "media" / name
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read test image: {path}")
            image[0, 0, 0] = (int(image[0, 0, 0]) + 1) % 256
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"failed to mutate test image: {path}")
        self.coco["images"][0]["sha256"] = _sha(self.root / "media" / "positive.png")
        self.coco["images"][1]["sha256"] = _sha(self.root / "media" / "negative.png")
        self.write()


class IndependentPlayerHoldoutTests(unittest.TestCase):
    def test_defaults_are_the_pinned_release_inventory(self) -> None:
        self.assertEqual(
            PINNED_RELEASE_MINIMUMS.as_dict(),
            {
                "target_le_32": 150,
                "target_33_64": 400,
                "target_65_96": 250,
                "target_gt_96": 250,
                "reviewed_negatives": 1_000,
            },
        )

    def test_prepare_is_deterministic_and_verifies_all_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input")
            first = root / "first"
            second = root / "second"
            first_manifest = prepare_holdout(fixture.manifest_path, first, minimums=SMALL_GATES)
            second_manifest = prepare_holdout(fixture.manifest_path, second, minimums=SMALL_GATES)

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual((first / MANIFEST_NAME).read_bytes(), (second / MANIFEST_NAME).read_bytes())
            self.assertEqual(verify_holdout(first, access_mode="development")["counts"], {
                "target_le_32": 1,
                "target_33_64": 1,
                "target_65_96": 1,
                "target_gt_96": 1,
                "reviewed_negatives": 1,
            })
            self.assertFalse(first_manifest["release_gates"]["meets_pinned_release_gates"])
            self.assertFalse(
                first_manifest["release_gates"]["pinned_descriptive_inventory_target_met"]
            )
            self.assertEqual(first_manifest["split_contract"]["unit"], "capture_session")
            self.assertFalse(first_manifest["split_contract"]["frame_level_random_split_allowed"])
            self.assertEqual(
                first_manifest["source_group_inventory"],
                {
                    "definition": "distinct normalized COCO image session_id values",
                    "overall_capture_sessions": 1,
                    "target_bearing_capture_sessions": {
                        "target_le_32": 1,
                        "target_33_64": 1,
                        "target_65_96": 1,
                        "target_gt_96": 1,
                    },
                    "reviewed_negative_capture_sessions": 1,
                },
            )

    def test_buckets_use_height_projected_to_1080p_across_source_resolutions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input", image_height=720)
            # A native 32px box at 720p projects to 48px at the pinned 1080p
            # reference and must therefore enter (32, 64], not <=32.
            fixture.coco["annotations"][1]["bbox"][3] = 32
            fixture.coco["annotations"][1]["area"] = 320
            fixture.write()
            output = root / "package"
            manifest = prepare_holdout(fixture.manifest_path, output, minimums=SMALL_GATES)

            self.assertEqual(manifest["target_height_definition"]["reference_height_px"], 1080)
            self.assertEqual(manifest["counts"]["target_33_64"], 1)
            annotations = json.loads(
                (output / "annotations" / "instances.json").read_text(encoding="utf-8")
            )
            projected = next(
                item["projected_height_px_at_1080p"]
                for item in annotations["annotations"]
                if item["id"] == 2
            )
            self.assertEqual(projected, 48.0)

    def test_ultra_far_inventory_target_is_descriptive_not_release_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input")
            fixture.coco["annotations"] = fixture.coco["annotations"][1:]
            fixture.write()
            output = root / "package"
            manifest = prepare_holdout(fixture.manifest_path, output, minimums=SMALL_GATES)

            self.assertEqual(manifest["counts"]["target_le_32"], 0)
            gates = manifest["release_gates"]
            self.assertTrue(gates["configured_gates_pass"])
            self.assertFalse(gates["configured_descriptive_inventory_target_met"])
            self.assertNotIn("target_le_32", gates["gating_count_keys"])

    def test_refuses_overwrite_and_detects_member_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input")
            output = root / "package"
            prepare_holdout(fixture.manifest_path, output, minimums=SMALL_GATES)
            with self.assertRaisesRegex(HoldoutContractError, "overwrite"):
                prepare_holdout(fixture.manifest_path, output, minimums=SMALL_GATES)
            (output / "images" / "000000000001.png").write_bytes(b"tampered")
            with self.assertRaisesRegex(HoldoutContractError, "tampered"):
                verify_holdout(output)

    def test_rejects_path_traversal_and_symlink_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "traversal")
            fixture.coco["images"][0]["file_name"] = "../escape.png"
            fixture.write()
            with self.assertRaisesRegex(HoldoutContractError, "unsafe"):
                prepare_holdout(fixture.manifest_path, root / "bad", minimums=SMALL_GATES)

            link_fixture = HoldoutFixture(root / "link")
            target = link_fixture.root / "real.png"
            target.write_bytes((link_fixture.root / "media" / "positive.png").read_bytes())
            (link_fixture.root / "media" / "positive.png").unlink()
            try:
                (link_fixture.root / "media" / "positive.png").symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable")
            link_fixture.coco["images"][0]["sha256"] = _sha(target)
            link_fixture.write()
            with self.assertRaisesRegex(HoldoutContractError, "symlink"):
                prepare_holdout(link_fixture.manifest_path, root / "bad-link", minimums=SMALL_GATES)

    def test_rejects_session_leakage_against_reference_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_fixture = HoldoutFixture(root / "reference-input", package_id="reference", session_id="same-session")
            reference = root / "reference"
            prepare_holdout(reference_fixture.manifest_path, reference, minimums=SMALL_GATES)
            candidate = HoldoutFixture(
                root / "candidate-input", package_id="candidate", session_id="same-session", seed=77
            )
            with self.assertRaisesRegex(HoldoutContractError, "session crosses"):
                prepare_holdout(
                    candidate.manifest_path,
                    root / "candidate",
                    minimums=SMALL_GATES,
                    reference_manifests=[reference / MANIFEST_NAME],
                )

    def test_rejects_raw_source_reuse_and_sealed_reference_during_development(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_fixture = HoldoutFixture(
                root / "reference-input",
                package_id="sealed-reference",
                pool=POOL_SEALED,
                session_id="session-reference",
                seed=101,
            )
            reference = root / "reference"
            prepare_holdout(reference_fixture.manifest_path, reference, minimums=SMALL_GATES)

            development = HoldoutFixture(
                root / "development-input",
                package_id="development-new",
                session_id="session-development",
                seed=202,
            )
            with self.assertRaisesRegex(HoldoutContractError, "cannot inspect a sealed"):
                prepare_holdout(
                    development.manifest_path,
                    root / "development",
                    minimums=SMALL_GATES,
                    reference_manifests=[reference],
                )

            sealed_candidate = HoldoutFixture(
                root / "sealed-candidate-input",
                package_id="sealed-candidate",
                pool=POOL_SEALED,
                session_id="session-new",
                seed=303,
            )
            (sealed_candidate.root / "capture.bin").write_bytes(
                (reference_fixture.root / "capture.bin").read_bytes()
            )
            sealed_candidate.write()
            with self.assertRaisesRegex(HoldoutContractError, "raw capture source crosses"):
                prepare_holdout(
                    sealed_candidate.manifest_path,
                    root / "sealed-candidate",
                    minimums=SMALL_GATES,
                    reference_manifests=[reference],
                )

    def test_rejects_exact_and_perceptual_cross_pool_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = HoldoutFixture(root / "source", package_id="reference", session_id="session-ref")
            reference = root / "reference"
            prepare_holdout(source.manifest_path, reference, minimums=SMALL_GATES)

            exact = HoldoutFixture(root / "exact", package_id="exact", session_id="session-exact")
            (exact.root / "capture.bin").write_bytes(b"distinct-raw-capture-exact")
            exact.write()
            with self.assertRaisesRegex(HoldoutContractError, "exact image leakage"):
                prepare_holdout(
                    exact.manifest_path, root / "exact-output", minimums=SMALL_GATES,
                    reference_manifests=[reference],
                )

            perceptual = HoldoutFixture(root / "perceptual", package_id="perceptual", session_id="session-perceptual")
            (perceptual.root / "capture.bin").write_bytes(b"distinct-raw-capture-perceptual")
            perceptual.mutate_image_one_pixel()
            with self.assertRaisesRegex(HoldoutContractError, "perceptual image leakage"):
                prepare_holdout(
                    perceptual.manifest_path, root / "perceptual-output", minimums=SMALL_GATES,
                    reference_manifests=[reference],
                )

    def test_requires_two_reviewer_negative_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input")
            review = fixture.coco["images"][1]["negative_review"]
            review["reviewer_2"]["reviewer_id"] = "reviewer-a"
            fixture.write()
            with self.assertRaisesRegex(HoldoutContractError, "two distinct reviewers"):
                prepare_holdout(fixture.manifest_path, root / "bad", minimums=SMALL_GATES)

            fixture = HoldoutFixture(root / "input-2")
            review = fixture.coco["images"][1]["negative_review"]
            review["reviewer_2"]["decision"] = "player_present"
            fixture.write()
            with self.assertRaisesRegex(HoldoutContractError, "resolved adjudication"):
                prepare_holdout(fixture.manifest_path, root / "bad-2", minimums=SMALL_GATES)

    def test_rejects_insufficient_buckets_and_invalid_coco_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input")
            strict = SMALL_GATES._replace(target_33_64=2)
            with self.assertRaisesRegex(HoldoutContractError, "target_33_64=1<2"):
                prepare_holdout(fixture.manifest_path, root / "small", minimums=strict)

            fixture.coco["annotations"][0]["iscrowd"] = False
            fixture.write()
            with self.assertRaisesRegex(HoldoutContractError, "iscrowd=0"):
                prepare_holdout(fixture.manifest_path, root / "bad-flag", minimums=SMALL_GATES)

    def test_annotations_are_hash_checked_from_the_same_bytes_that_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input")
            real_load_json = holdout_module._load_json
            swapped = False

            def swap_before_annotation_read(path: Path, description: str):
                nonlocal swapped
                if description == "source annotations" and not swapped:
                    swapped = True
                    changed = json.loads(path.read_text(encoding="utf-8"))
                    changed["annotations"][0]["occluded"] = True
                    _write_json(path, changed)
                return real_load_json(path, description)

            with mock.patch.object(
                holdout_module,
                "_load_json",
                side_effect=swap_before_annotation_read,
            ):
                with self.assertRaisesRegex(HoldoutContractError, "annotations hash mismatch"):
                    prepare_holdout(
                        fixture.manifest_path,
                        root / "swapped",
                        minimums=SMALL_GATES,
                    )

    def test_sealed_pool_rejects_development_access_and_has_append_only_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input", package_id="sealed-a", pool=POOL_SEALED)
            package = root / "sealed"
            prepare_holdout(fixture.manifest_path, package, minimums=SMALL_GATES)
            with self.assertRaisesRegex(HoldoutContractError, "cannot be opened"):
                verify_holdout(package, access_mode="development")

            consumed = record_sealed_consumption(
                package,
                event_id="release-eval",
                recorded_at_utc=NOW,
                actor_id="release-operator",
                purpose="predeclared candidate tournament",
                evaluation_plan_sha256="a" * 64,
            )
            self.assertTrue(consumed.is_file())
            with self.assertRaisesRegex(HoldoutContractError, "already been consumed"):
                record_sealed_consumption(
                    package,
                    event_id="peek-again",
                    recorded_at_utc=NOW,
                    actor_id="release-operator",
                    purpose="second look",
                    evaluation_plan_sha256="b" * 64,
                )
            retired = retire_sealed_holdout(
                package,
                event_id="retire-after-eval",
                recorded_at_utc="2026-08-13T12:00:01Z",
                actor_id="release-operator",
                reason="release decision completed",
            )
            self.assertTrue(retired.is_file())
            status = verify_holdout(package, access_mode="curator")
            self.assertEqual(status["access_events"], 2)
            self.assertTrue(status["retired"])
            with self.assertRaisesRegex(HoldoutContractError, "retired"):
                retire_sealed_holdout(
                    package,
                    event_id="retire-again",
                    recorded_at_utc=NOW,
                    actor_id="release-operator",
                    reason="duplicate retirement",
                )

    def test_final_evaluation_transaction_publishes_then_binds_and_retires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(
                root / "input",
                package_id="sealed-final",
                pool=POOL_SEALED,
            )
            package = root / "sealed"
            prepare_holdout(fixture.manifest_path, package, minimums=SMALL_GATES)
            evidence = root / "evidence" / "metrics.json"
            durability_events: list[str] = []
            original_fsync_directory = holdout_module._fsync_directory
            original_fsync_file = holdout_module._fsync_regular_file
            original_write_new = holdout_module._write_new

            def publish() -> dict[str, str]:
                durability_events.append("publish")
                evidence.parent.mkdir()
                evidence.write_bytes(b'{"result":"final"}\n')
                return {"result": "final"}

            def tracked_directory(path: Path) -> None:
                durability_events.append(f"dir:{Path(path).resolve()}")
                original_fsync_directory(path)

            def tracked_file(path: Path, description: str) -> None:
                durability_events.append(f"file:{Path(path).resolve()}")
                original_fsync_file(path, description)

            def tracked_write(path: Path, payload: bytes) -> None:
                durability_events.append(f"write:{path.name}")
                original_write_new(path, payload)

            with (
                mock.patch.object(
                    holdout_module, "_fsync_directory", side_effect=tracked_directory
                ),
                mock.patch.object(
                    holdout_module, "_fsync_regular_file", side_effect=tracked_file
                ),
                mock.patch.object(
                    holdout_module, "_write_new", side_effect=tracked_write
                ),
            ):
                result, ledger = complete_sealed_evaluation(
                    package,
                    evidence_path=evidence,
                    publish_evaluation=publish,
                    event_id="release-final",
                    actor_id="release-operator",
                    purpose="exact frozen independent evaluation",
                    evaluation_plan_sha256="a" * 64,
                    retirement_event_id="release-final-retired",
                    retirement_reason="final evaluation completed",
                    utc_now=_utc_clock(NOW, "2026-08-13T12:00:01Z"),
                )

            publish_index = durability_events.index("publish")
            evidence_file_index = durability_events.index(
                f"file:{evidence.resolve()}", publish_index
            )
            evidence_dir_index = durability_events.index(
                f"dir:{evidence.parent.resolve()}", evidence_file_index
            )
            evidence_parent_index = durability_events.index(
                f"dir:{evidence.parent.parent.resolve()}", evidence_dir_index + 1
            )
            consumed_index = durability_events.index("write:00000001-release-final.json")
            consumed_dir_index = durability_events.index(
                f"dir:{(package / 'access-ledger').resolve()}", consumed_index
            )
            retired_index = durability_events.index(
                "write:00000002-release-final-retired.json"
            )
            retired_dir_index = durability_events.index(
                f"dir:{(package / 'access-ledger').resolve()}", retired_index
            )
            self.assertLess(consumed_index, consumed_dir_index)
            self.assertLess(consumed_dir_index, publish_index)
            self.assertLess(publish_index, evidence_file_index)
            self.assertLess(evidence_file_index, evidence_dir_index)
            self.assertLess(evidence_dir_index, evidence_parent_index)
            self.assertLess(evidence_parent_index, retired_index)
            self.assertLess(retired_index, retired_dir_index)

            self.assertEqual(result, {"result": "final"})
            self.assertEqual(
                ledger["evaluation_evidence_sha256"], sha256(evidence.read_bytes()).hexdigest()
            )
            events = holdout_module._ledger_events(
                package,
                expected_manifest_sha256=verify_holdout(
                    package, access_mode="curator"
                )["manifest_content_sha256"],
            )
            self.assertEqual([item["operation"] for item in events], ["consumed", "retired"])
            self.assertEqual(events[0]["schema_version"], 1)
            self.assertNotIn("evaluation_evidence_sha256", events[0])
            self.assertEqual(events[1]["schema_version"], 2)
            self.assertEqual(
                events[1]["evaluation_evidence_sha256"],
                ledger["evaluation_evidence_sha256"],
            )
            self.assertEqual(
                events[1]["previous_event_sha256"], events[0]["event_content_sha256"]
            )
            self.assertTrue(verify_holdout(package, access_mode="curator")["retired"])

            second_called = False

            def second_publish() -> None:
                nonlocal second_called
                second_called = True

            with self.assertRaisesRegex(HoldoutContractError, "already been consumed"):
                complete_sealed_evaluation(
                    package,
                    evidence_path=root / "second" / "metrics.json",
                    publish_evaluation=second_publish,
                    event_id="second-final",
                    actor_id="release-operator",
                    purpose="unsafe repeated evaluation",
                    evaluation_plan_sha256="b" * 64,
                    retirement_event_id="second-retired",
                    retirement_reason="must not happen",
                    utc_now=_utc_clock(
                        "2026-08-13T12:00:02Z", "2026-08-13T12:00:03Z"
                    ),
                )
            self.assertFalse(second_called)

    def test_final_evaluation_failure_after_access_burns_holdout_permanently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(
                root / "input",
                package_id="sealed-prepublish-failure",
                pool=POOL_SEALED,
            )
            package = root / "sealed"
            prepare_holdout(fixture.manifest_path, package, minimums=SMALL_GATES)

            def fail() -> None:
                raise RuntimeError("inference failed before publication")

            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                complete_sealed_evaluation(
                    package,
                    evidence_path=root / "evidence" / "metrics.json",
                    publish_evaluation=fail,
                    event_id="release-failed",
                    actor_id="release-operator",
                    purpose="exact frozen independent evaluation",
                    evaluation_plan_sha256="a" * 64,
                    retirement_event_id="release-failed-retired",
                    retirement_reason="evaluation completed",
                    utc_now=_utc_clock(NOW),
                )
            events = holdout_module._ledger_events(package, allow_lock=True)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["operation"], "consumed")
            self.assertTrue((package / "access-ledger" / ".write.lock").exists())
            with self.assertRaisesRegex(HoldoutContractError, "another process"):
                complete_sealed_evaluation(
                    package,
                    evidence_path=root / "second" / "metrics.json",
                    publish_evaluation=lambda: self.fail("must not run"),
                    event_id="second-failed",
                    actor_id="release-operator",
                    purpose="unsafe retry",
                    evaluation_plan_sha256="b" * 64,
                    retirement_event_id="second-failed-retired",
                    retirement_reason="must not happen",
                    utc_now=_utc_clock("2026-08-13T12:00:02Z"),
                )

    def test_postpublication_failure_preserves_fail_closed_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(
                root / "input",
                package_id="sealed-postpublish-failure",
                pool=POOL_SEALED,
            )
            package = root / "sealed"
            prepare_holdout(fixture.manifest_path, package, minimums=SMALL_GATES)
            evidence = root / "evidence" / "metrics.json"

            def publish_then_fail() -> None:
                evidence.parent.mkdir()
                evidence.write_bytes(b"published but callback failed\n")
                raise RuntimeError("failure after atomic publication")

            with self.assertRaisesRegex(RuntimeError, "after atomic publication"):
                complete_sealed_evaluation(
                    package,
                    evidence_path=evidence,
                    publish_evaluation=publish_then_fail,
                    event_id="release-partial",
                    actor_id="release-operator",
                    purpose="exact frozen independent evaluation",
                    evaluation_plan_sha256="a" * 64,
                    retirement_event_id="release-partial-retired",
                    retirement_reason="evaluation completed",
                    utc_now=_utc_clock(NOW),
                )
            self.assertTrue((package / "access-ledger" / ".write.lock").is_file())
            with self.assertRaisesRegex(HoldoutContractError, "another process"):
                complete_sealed_evaluation(
                    package,
                    evidence_path=root / "second" / "metrics.json",
                    publish_evaluation=lambda: None,
                    event_id="second",
                    actor_id="release-operator",
                    purpose="must remain blocked",
                    evaluation_plan_sha256="b" * 64,
                    retirement_event_id="second-retired",
                    retirement_reason="must not happen",
                    utc_now=_utc_clock(
                        "2026-08-13T12:00:02Z", "2026-08-13T12:00:03Z"
                    ),
                )

    def test_directory_flush_failure_after_consumption_preserves_forensic_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(
                root / "input",
                package_id="sealed-durability-failure",
                pool=POOL_SEALED,
            )
            package = root / "sealed"
            prepare_holdout(fixture.manifest_path, package, minimums=SMALL_GATES)
            evidence = root / "evidence" / "metrics.json"
            original_fsync_directory = holdout_module._fsync_directory

            def publish() -> None:
                evidence.parent.mkdir()
                evidence.write_bytes(b"durable evidence before ledger failure\n")

            def fail_after_consumption(path: Path) -> None:
                original_fsync_directory(path)
                ledger = package / "access-ledger"
                if (
                    Path(path).resolve() == ledger.resolve()
                    and (ledger / "00000001-final.json").exists()
                    and not (ledger / "00000002-final-retired.json").exists()
                ):
                    raise HoldoutContractError("injected ledger directory flush failure")

            with (
                mock.patch.object(
                    holdout_module,
                    "_fsync_directory",
                    side_effect=fail_after_consumption,
                ),
                self.assertRaisesRegex(HoldoutContractError, "injected ledger"),
            ):
                complete_sealed_evaluation(
                    package,
                    evidence_path=evidence,
                    publish_evaluation=publish,
                    event_id="final",
                    actor_id="release-operator",
                    purpose="exact frozen independent evaluation",
                    evaluation_plan_sha256="a" * 64,
                    retirement_event_id="final-retired",
                    retirement_reason="must fail closed",
                    utc_now=_utc_clock(NOW, "2026-08-13T12:00:01Z"),
                )
            self.assertFalse(evidence.exists())
            self.assertTrue((package / "access-ledger" / "00000001-final.json").is_file())
            self.assertFalse(
                (package / "access-ledger" / "00000002-final-retired.json").exists()
            )
            self.assertTrue((package / "access-ledger" / ".write.lock").is_file())

    def test_win32_flush_helpers_request_write_capable_handles(self) -> None:
        class FakeFunction:
            def __init__(self, result: int) -> None:
                self.result = result
                self.calls: list[tuple[object, ...]] = []
                self.argtypes = None
                self.restype = None

            def __call__(self, *args):
                self.calls.append(args)
                return self.result

        class FakeKernel:
            def __init__(self) -> None:
                self.CreateFileW = FakeFunction(123)
                self.FlushFileBuffers = FakeFunction(1)
                self.CloseHandle = FakeFunction(1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "metrics.json"
            evidence.write_bytes(b"evidence\n")
            kernel = FakeKernel()
            with (
                mock.patch.object(holdout_module.os, "name", "nt"),
                mock.patch.object(
                    holdout_module.ctypes,
                    "WinDLL",
                    return_value=kernel,
                    create=True,
                ),
                mock.patch.object(
                    holdout_module,
                    "_require_regular_file",
                    return_value=evidence,
                ),
                mock.patch.object(
                    holdout_module, "_require_directory", return_value=root
                ),
            ):
                holdout_module._fsync_regular_file(evidence, "evidence")
                holdout_module._fsync_directory(root)

            self.assertEqual(len(kernel.CreateFileW.calls), 2)
            self.assertTrue(
                all(call[1] == 0x40000000 for call in kernel.CreateFileW.calls)
            )
            self.assertEqual(len(kernel.FlushFileBuffers.calls), 2)
            self.assertEqual(len(kernel.CloseHandle.calls), 2)

    def test_ledger_timestamps_must_be_strictly_increasing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input", package_id="sealed-time", pool=POOL_SEALED)
            package = root / "sealed"
            prepare_holdout(fixture.manifest_path, package, minimums=SMALL_GATES)
            record_sealed_consumption(
                package,
                event_id="release-eval",
                recorded_at_utc=NOW,
                actor_id="release-operator",
                purpose="predeclared candidate tournament",
                evaluation_plan_sha256="a" * 64,
            )
            with self.assertRaisesRegex(HoldoutContractError, "later than"):
                retire_sealed_holdout(
                    package,
                    event_id="retroactive-retirement",
                    recorded_at_utc="2026-08-13T11:59:59Z",
                    actor_id="release-operator",
                    reason="invalid retrospective event",
                )

    def test_manifest_and_ledger_tamper_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input", package_id="sealed-tamper", pool=POOL_SEALED)
            package = root / "sealed"
            prepare_holdout(fixture.manifest_path, package, minimums=SMALL_GATES)

            manifest_path = package / MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["counts"]["target_33_64"] = 999
            manifest_path.write_bytes(
                (json.dumps(manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode()
            )
            with self.assertRaisesRegex(HoldoutContractError, "self-hash mismatch"):
                verify_holdout(package)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = HoldoutFixture(root / "input", package_id="ledger-tamper", pool=POOL_SEALED)
            package = root / "sealed"
            prepare_holdout(fixture.manifest_path, package, minimums=SMALL_GATES)
            event_path = record_sealed_consumption(
                package,
                event_id="release-eval",
                recorded_at_utc=NOW,
                actor_id="release-operator",
                purpose="predeclared candidate tournament",
                evaluation_plan_sha256="a" * 64,
            )
            event = json.loads(event_path.read_text(encoding="utf-8"))
            event["purpose"] = "tampered purpose"
            event_path.write_bytes(
                (json.dumps(event, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode()
            )
            with self.assertRaisesRegex(HoldoutContractError, "hash chain mismatch"):
                verify_holdout(package, access_mode="curator")


if __name__ == "__main__":
    unittest.main()
