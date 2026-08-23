from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from capture.base import FramePacket
from scripts import capture_player_head_dataset as dataset_contract
from scripts.capture_player_head_dataset import (
    DATASET_YAML_NAME,
    NORMALIZED_COCO_NAME,
    OUTPUT_MANIFEST_NAME,
    DatasetContractError,
    export_dataset,
    main,
    validate_dataset,
)
from scripts import record_capture_player_head as recorder


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


class CaptureDatasetFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw = root / "datasets" / "capture_player_head" / "raw"
        self.raw.mkdir(parents=True)
        self.document: dict[str, object] = {
            "schema_version": 1,
            "categories": [{"id": 0, "name": "player"}, {"id": 1, "name": "head"}],
            "sessions": [],
            "images": [],
            "annotations": [],
        }
        self.arrays: dict[tuple[str, str], np.ndarray] = {}
        self._next_image_id = 1
        self._next_annotation_id = 1
        for seed, (session_id, split) in enumerate(
            (("session-train", "train"), ("session-val", "val"), ("session-test", "test")),
            start=11,
        ):
            self._add_session(session_id, split, seed)
        self._add_frame("session-train", seed=97, negative=True)
        self.write_annotations()

    @property
    def annotations_path(self) -> Path:
        return self.raw / "annotations.json"

    def _pixels(self, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).integers(0, 256, size=(80, 96, 3), dtype=np.uint8)

    def _encode(self, pixels: np.ndarray, compression: int = 3) -> bytes:
        success, encoded = cv2.imencode(
            ".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, compression]
        )
        if not success:
            raise AssertionError("test PNG encoding failed")
        return encoded.tobytes()

    def _add_session(self, session_id: str, split: str, seed: int) -> None:
        sessions = self.document["sessions"]
        assert isinstance(sessions, list)
        sessions.append({"session_id": session_id, "split": split})
        session_root = self.raw / session_id
        (session_root / "images").mkdir(parents=True)
        self._add_frame(session_id, seed=seed, negative=False)

    def _add_frame(self, session_id: str, *, seed: int, negative: bool) -> int:
        session_root = self.raw / session_id
        frame_index = len(list((session_root / "images").glob("*.png")))
        file_name = f"images/frame-{frame_index:06d}.png"
        pixels = self._pixels(seed)
        encoded = self._encode(pixels)
        (session_root / file_name).write_bytes(encoded)
        self.arrays[(session_id, file_name)] = pixels
        image_id = self._next_image_id
        self._next_image_id += 1
        images = self.document["images"]
        assert isinstance(images, list)
        images.append(
            {
                "id": image_id,
                "session_id": session_id,
                "file_name": file_name,
                "width": 96,
                "height": 80,
                "sha256": sha256(encoded).hexdigest(),
                "negative": negative,
            }
        )
        if not negative:
            player_id = self._next_annotation_id
            self._next_annotation_id += 1
            head_id = self._next_annotation_id
            self._next_annotation_id += 1
            annotations = self.document["annotations"]
            assert isinstance(annotations, list)
            annotations.extend(
                [
                    {
                        "id": player_id,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [5, 5, 80, 70],
                        "area": 5600,
                        "iscrowd": 0,
                        "ignore": 0,
                        "visibility": 1.0,
                        "occluded": False,
                        "truncated": False,
                    },
                    {
                        "id": head_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "parent_player_annotation_id": player_id,
                        "bbox": [30, 8, 24, 18],
                        "area": 432,
                        "iscrowd": 0,
                        "ignore": 0,
                        "visibility": 1.0,
                        "occluded": False,
                        "truncated": False,
                    },
                ]
            )
        self.write_session(session_id)
        return image_id

    def image(self, session_id: str) -> dict[str, object]:
        images = self.document["images"]
        assert isinstance(images, list)
        return next(image for image in images if image["session_id"] == session_id)

    def head(self, session_id: str) -> dict[str, object]:
        image_id = self.image(session_id)["id"]
        annotations = self.document["annotations"]
        assert isinstance(annotations, list)
        return next(
            annotation
            for annotation in annotations
            if annotation["image_id"] == image_id and annotation["category_id"] == 1
        )

    def write_session(self, session_id: str) -> None:
        session_root = self.raw / session_id
        existing_path = session_root / "session.json"
        existing: dict[str, object] = {}
        if existing_path.exists():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
        frames = []
        for sequence, path in enumerate(sorted((session_root / "images").glob("*.png"))):
            relative = path.relative_to(session_root).as_posix()
            content = path.read_bytes()
            pixels = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert pixels is not None
            frames.append(
                {
                    "file_name": relative,
                    "sha256": sha256(content).hexdigest(),
                    "byte_size": len(content),
                    "width": int(pixels.shape[1]),
                    "height": int(pixels.shape[0]),
                    "channels": 3,
                    "source_sequence": sequence,
                    "source_read_started_ns": 1000 + sequence,
                    "source_read_completed_ns": 1100 + sequence,
                    "saved_monotonic_ns": 1200 + sequence,
                    "saved_utc": "2026-08-15T20:00:00Z",
                }
            )
        manifest = {
            "schema": "proaim.capture_player_head.session",
            "schema_version": 1,
            "session_id": session_id,
            "status": "complete",
            "complete": True,
            "completion_reason": "requested",
            "created_utc": "2026-08-15T20:00:00Z",
            "ended_utc": "2026-08-15T20:00:10Z",
            "range": existing.get("range", "medium"),
            "motion": existing.get("motion", "moving"),
            "scenario": "fixture",
            "limits": {"duration_seconds": 10, "max_frames": 10, "sample_fps": 1},
            "capture": {
                "requested": {},
                "actual": {},
                "description": "fixture",
                "stats": {},
                "error": None,
            },
            "encoding": {"format": "png", "extension": ".png", "png_compression": 3},
            "frames": frames,
            "skipped_stale_packets": 0,
        }
        existing_path.write_bytes(_canonical(manifest))

    def replace_frame(self, session_id: str, pixels: np.ndarray, compression: int) -> None:
        image = self.image(session_id)
        file_name = str(image["file_name"])
        encoded = self._encode(pixels, compression)
        (self.raw / session_id / file_name).write_bytes(encoded)
        self.arrays[(session_id, file_name)] = pixels.copy()
        image["sha256"] = sha256(encoded).hexdigest()
        image["width"] = int(pixels.shape[1])
        image["height"] = int(pixels.shape[0])
        self.write_session(session_id)
        self.write_annotations()

    def write_annotations(self) -> None:
        self.annotations_path.write_bytes(_canonical(self.document))


class CapturePlayerHeadDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = CaptureDatasetFixture(Path(self.temp.name))

    def test_validates_and_exports_exact_deterministic_two_class_dataset(self) -> None:
        dataset = validate_dataset(self.fixture.raw)
        report = dataset.report()
        self.assertEqual(report["sessions"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(report["images"], {"train": 2, "val": 1, "test": 1})
        self.assertEqual(report["negative_images"]["train"], 1)

        output_a = Path(self.temp.name) / "export-a"
        output_b = Path(self.temp.name) / "export-b"
        manifest_a = export_dataset(dataset, output_a)
        manifest_b = export_dataset(dataset, output_b)
        self.assertEqual(manifest_a, manifest_b)

        files_a = {
            path.relative_to(output_a).as_posix(): path.read_bytes()
            for path in output_a.rglob("*")
            if path.is_file()
        }
        files_b = {
            path.relative_to(output_b).as_posix(): path.read_bytes()
            for path in output_b.rglob("*")
            if path.is_file()
        }
        self.assertEqual(files_a, files_b)
        self.assertIn(b"  0: player\n  1: head\n", files_a[DATASET_YAML_NAME])

        train_positive = output_a / "labels" / "train" / "000000000001.txt"
        label_lines = train_positive.read_text(encoding="ascii").splitlines()
        self.assertEqual([line.split()[0] for line in label_lines], ["0", "1"])
        negative_id = max(
            int(image["id"])
            for image in self.fixture.document["images"]  # type: ignore[index]
            if image["negative"]
        )
        self.assertEqual(
            (output_a / "labels" / "train" / f"{negative_id:012d}.txt").read_bytes(),
            b"",
        )
        coco = json.loads((output_a / NORMALIZED_COCO_NAME).read_text(encoding="utf-8"))
        self.assertEqual(coco["categories"], [{"id": 0, "name": "player"}, {"id": 1, "name": "head"}])

        stored = json.loads((output_a / OUTPUT_MANIFEST_NAME).read_text(encoding="utf-8"))
        payload = dict(stored)
        recorded_payload_hash = payload.pop("manifest_payload_sha256")
        self.assertEqual(recorded_payload_hash, sha256(_canonical(payload)).hexdigest())
        for member in stored["members"]:
            content = (output_a / member["path"]).read_bytes()
            self.assertEqual(member["bytes"], len(content))
            self.assertEqual(member["sha256"], sha256(content).hexdigest())

    def test_ultralytics_resolves_export_from_yaml_parent_not_process_cwd(self) -> None:
        dataset = validate_dataset(self.fixture.raw)
        output = Path(self.temp.name) / "portable-export"
        export_dataset(dataset, output)
        yaml_path = output / DATASET_YAML_NAME
        self.assertNotIn("path:", yaml_path.read_text(encoding="utf-8"))

        unrelated = Path(self.temp.name) / "unrelated-working-directory"
        unrelated.mkdir()
        matplotlib_config = Path(self.temp.name) / "matplotlib-config"
        matplotlib_config.mkdir()
        previous = Path.cwd()
        try:
            with mock.patch.dict(os.environ, {"MPLCONFIGDIR": str(matplotlib_config)}):
                try:
                    from ultralytics.data.utils import check_det_dataset
                except ModuleNotFoundError as exc:
                    if exc.name == "ultralytics":
                        self.skipTest(
                            "requires the optional Ultralytics export dependency"
                        )
                    raise

                os.chdir(unrelated)
                loaded = check_det_dataset(str(yaml_path.resolve()), autodownload=False)
        finally:
            os.chdir(previous)

        self.assertEqual(Path(loaded["path"]), output.resolve())
        self.assertEqual(Path(loaded["train"]), (output / "images" / "train").resolve())
        self.assertEqual(Path(loaded["val"]), (output / "images" / "val").resolve())
        self.assertEqual(Path(loaded["test"]), (output / "images" / "test").resolve())
        self.assertEqual(loaded["names"], {0: "player", 1: "head"})

    def test_consumes_finalized_manifests_written_by_real_recorder_core(self) -> None:
        raw = Path(self.temp.name) / "recorder-e2e" / "raw"
        session_ids = ("recorded-train", "recorded-val", "recorded-test")
        splits = ("train", "val", "test")

        class OneFrameCapture:
            description = "contract integration capture"
            actual_settings = {"width": 96, "height": 80, "pixel_format": "BGR3"}
            stats = {"frames_read": 1, "frames_delivered": 1}
            ended = False
            error = None

            def __init__(self, image: np.ndarray) -> None:
                self.image = image
                self.delivered = False

            def start(self) -> None:
                pass

            def read(self, timeout: float | None = None) -> FramePacket | None:
                del timeout
                if self.delivered:
                    return None
                self.delivered = True
                return FramePacket(self.image, 1, 1_000_000, 2_000_000)

            def close(self) -> None:
                pass

        document: dict[str, object] = {
            "schema_version": 1,
            "categories": [{"id": 0, "name": "player"}, {"id": 1, "name": "head"}],
            "sessions": [],
            "images": [],
            "annotations": [],
        }
        annotation_id = 1
        for image_id, (session_id, split) in enumerate(zip(session_ids, splits), start=1):
            pixels = np.random.default_rng(200 + image_id).integers(
                0, 256, size=(80, 96, 3), dtype=np.uint8
            )
            source = OneFrameCapture(pixels)
            tick = 0

            def clock() -> int:
                nonlocal tick
                tick += 1_000_000
                return tick

            result = recorder.record_session(
                recorder.RecorderConfig(
                    range_label="medium",
                    motion="moving",
                    scenario="contract-integration",
                    output_root=raw,
                    duration_seconds=1.0,
                    max_frames=1,
                    sample_fps=1.0,
                ),
                capture_factory=lambda _config, capture=source: capture,
                clock_ns=clock,
                utc_now=lambda: datetime(2026, 8, 15, 20, 30, tzinfo=timezone.utc),
                session_id_factory=lambda value=session_id: value,
            )
            self.assertEqual(result.status, "complete")
            raw_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            frame = raw_manifest["frames"][0]
            sessions = document["sessions"]
            images = document["images"]
            annotations = document["annotations"]
            assert isinstance(sessions, list)
            assert isinstance(images, list)
            assert isinstance(annotations, list)
            sessions.append({"session_id": session_id, "split": split})
            images.append(
                {
                    "id": image_id,
                    "session_id": session_id,
                    "file_name": frame["file_name"],
                    "width": frame["width"],
                    "height": frame["height"],
                    "sha256": frame["sha256"],
                    "negative": False,
                }
            )
            annotations.extend(
                [
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [5, 5, 80, 70],
                        "area": 5600,
                        "iscrowd": 0,
                        "ignore": 0,
                        "visibility": 1.0,
                        "occluded": False,
                        "truncated": False,
                    },
                    {
                        "id": annotation_id + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "parent_player_annotation_id": annotation_id,
                        "bbox": [30, 8, 24, 18],
                        "area": 432,
                        "iscrowd": 0,
                        "ignore": 0,
                        "visibility": 1.0,
                        "occluded": False,
                        "truncated": False,
                    },
                ]
            )
            annotation_id += 2

        (raw / "annotations.json").write_bytes(_canonical(document))
        validated = validate_dataset(raw)
        self.assertEqual([session.session_id for session in validated.sessions], sorted(session_ids))
        self.assertEqual(len(validated.images), 3)
        self.assertTrue(all(session.range_label == "medium" for session in validated.sessions))
        self.assertTrue(all(session.motion == "moving" for session in validated.sessions))
        self.assertTrue(
            all(session.scenario == "contract-integration" for session in validated.sessions)
        )
        self.assertTrue(
            all(session.capture_description == "contract integration capture" for session in validated.sessions)
        )

    def test_rejects_complete_manifest_that_reports_capture_error(self) -> None:
        manifest_path = self.fixture.raw / "session-train" / "session.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["complete"])
        manifest["capture"]["error"] = "capture backend failed after last frame"
        manifest_path.write_bytes(_canonical(manifest))
        with self.assertRaisesRegex(
            DatasetContractError, "marked complete but reports a capture error"
        ):
            validate_dataset(self.fixture.raw)

    def test_ignored_zero_visibility_head_never_enters_train_yolo(self) -> None:
        self.fixture._add_frame("session-train", seed=98, negative=False)
        self.fixture.write_annotations()
        head = self.fixture.head("session-train")
        head["visibility"] = 0.0
        head["ignore"] = 1
        head["occluded"] = True
        self.fixture.write_annotations()
        dataset = validate_dataset(self.fixture.raw)
        output = Path(self.temp.name) / "export"
        export_dataset(dataset, output)
        labels = (output / "labels" / "train" / "000000000001.txt").read_text().splitlines()
        self.assertEqual([line.split()[0] for line in labels], ["0"])
        normalized = json.loads((output / NORMALIZED_COCO_NAME).read_text())
        normalized_head = next(item for item in normalized["annotations"] if item["id"] == head["id"])
        self.assertEqual(normalized_head["ignore"], 1)

    def test_each_split_requires_exportable_player_and_head_ground_truth(self) -> None:
        annotations = self.fixture.document["annotations"]
        assert isinstance(annotations, list)
        for annotation in annotations:
            if annotation["category_id"] == 0:
                annotation["occluded"] = True
        self.fixture.document["annotations"] = [
            annotation for annotation in annotations if annotation["category_id"] != 1
        ]
        self.fixture.write_annotations()
        with self.assertRaisesRegex(
            DatasetContractError, "split train has no exportable head annotations"
        ):
            validate_dataset(self.fixture.raw)

        self.fixture = CaptureDatasetFixture(Path(self.temp.name) / "ignored-heads")
        annotations = self.fixture.document["annotations"]
        assert isinstance(annotations, list)
        for annotation in annotations:
            if annotation["category_id"] == 1:
                annotation["ignore"] = 1
                annotation["visibility"] = 0.0
                annotation["occluded"] = True
        self.fixture.write_annotations()
        with self.assertRaisesRegex(
            DatasetContractError, "split train has no exportable head annotations"
        ):
            validate_dataset(self.fixture.raw)

    def test_fully_visible_player_requires_explicit_head_record(self) -> None:
        train_image_id = self.fixture.image("session-train")["id"]
        annotations = self.fixture.document["annotations"]
        assert isinstance(annotations, list)
        self.fixture.document["annotations"] = [
            annotation
            for annotation in annotations
            if not (
                annotation["image_id"] == train_image_id
                and annotation["category_id"] == 1
            )
        ]
        self.fixture.write_annotations()
        with self.assertRaisesRegex(
            DatasetContractError,
            "fully visible player annotation .* requires exactly one associated head record",
        ):
            validate_dataset(self.fixture.raw)

    def test_occluded_or_truncated_player_may_lack_head_record(self) -> None:
        for flag in ("occluded", "truncated"):
            with self.subTest(flag=flag):
                fixture = CaptureDatasetFixture(Path(self.temp.name) / f"missing-{flag}")
                fixture._add_frame("session-train", seed=123, negative=False)
                train_image_id = fixture.image("session-train")["id"]
                annotations = fixture.document["annotations"]
                assert isinstance(annotations, list)
                player = next(
                    annotation
                    for annotation in annotations
                    if annotation["image_id"] == train_image_id
                    and annotation["category_id"] == 0
                )
                player[flag] = True
                fixture.document["annotations"] = [
                    annotation
                    for annotation in annotations
                    if not (
                        annotation["image_id"] == train_image_id
                        and annotation["category_id"] == 1
                    )
                ]
                fixture.write_annotations()
                validate_dataset(fixture.raw)

    def test_requires_explicit_containing_parent_but_allows_crowded_overlap(self) -> None:
        head = self.fixture.head("session-train")
        head.pop("parent_player_annotation_id")
        self.fixture.write_annotations()
        with self.assertRaisesRegex(DatasetContractError, "parent_player_annotation_id"):
            validate_dataset(self.fixture.raw)

        self.fixture = CaptureDatasetFixture(Path(self.temp.name) / "second")
        head = self.fixture.head("session-train")
        annotations = self.fixture.document["annotations"]
        assert isinstance(annotations, list)
        annotations.append(
            {
                "id": 1000,
                "image_id": head["image_id"],
                "category_id": 0,
                "bbox": [20, 6, 50, 40],
                "area": 2000,
                "iscrowd": 0,
                "ignore": 0,
                "visibility": 1.0,
                "occluded": False,
                "truncated": False,
            }
        )
        annotations.append(
            {
                "id": 1002,
                "image_id": head["image_id"],
                "category_id": 1,
                "parent_player_annotation_id": 1000,
                "bbox": [55, 10, 10, 10],
                "area": 100,
                "iscrowd": 0,
                "ignore": 0,
                "visibility": 1.0,
                "occluded": False,
                "truncated": False,
            }
        )
        self.fixture.write_annotations()
        validate_dataset(self.fixture.raw)

        annotations.append(
            {
                "id": 1001,
                "image_id": head["image_id"],
                "category_id": 0,
                "bbox": [60, 50, 20, 20],
                "area": 400,
                "iscrowd": 0,
                "ignore": 0,
                "visibility": 1.0,
                "occluded": False,
                "truncated": False,
            }
        )
        head["parent_player_annotation_id"] = 1001
        self.fixture.write_annotations()
        with self.assertRaisesRegex(DatasetContractError, "does not contain"):
            validate_dataset(self.fixture.raw)

    def test_rejects_exact_and_perceptual_cross_split_leakage(self) -> None:
        train_image = self.fixture.image("session-train")
        train_pixels = self.fixture.arrays[("session-train", str(train_image["file_name"]))]
        self.fixture.replace_frame("session-val", train_pixels, compression=3)
        with self.assertRaisesRegex(DatasetContractError, "cross-split exact SHA-256 leakage"):
            validate_dataset(self.fixture.raw)

        self.fixture = CaptureDatasetFixture(Path(self.temp.name) / "dhash")
        train_image = self.fixture.image("session-train")
        train_pixels = self.fixture.arrays[("session-train", str(train_image["file_name"]))]
        self.fixture.replace_frame("session-val", train_pixels, compression=9)
        self.assertNotEqual(
            self.fixture.image("session-train")["sha256"],
            self.fixture.image("session-val")["sha256"],
        )
        with self.assertRaisesRegex(DatasetContractError, "cross-split dHash64 leakage"):
            validate_dataset(self.fixture.raw)

    def test_dhash_hamming_threshold_rejects_near_cross_split_only(self) -> None:
        real_image_details = dataset_contract._image_details

        def details_with(values: dict[tuple[str, str], str]):
            def implementation(path: Path, expected_sha: str) -> tuple[int, int, int, str]:
                width, height, byte_size, _fingerprint = real_image_details(path, expected_sha)
                key = (path.parent.parent.name, path.name)
                return width, height, byte_size, values[key]

            return implementation

        near_values = {
            ("session-train", "frame-000000.png"): "0000000000000000",
            ("session-train", "frame-000001.png"): "0000000000000002",
            ("session-val", "frame-000000.png"): "0000000000000001",
            ("session-test", "frame-000000.png"): "ffffffffffffffff",
        }
        with mock.patch.object(
            dataset_contract, "_image_details", side_effect=details_with(near_values)
        ):
            with self.assertRaisesRegex(
                DatasetContractError, r"dHash64 leakage \(Hamming distance 1\)"
            ):
                validate_dataset(self.fixture.raw)

        allowed_values = {
            ("session-train", "frame-000000.png"): "0000000000000000",
            # Distance one is deliberately retained inside the train split.
            ("session-train", "frame-000001.png"): "0000000000000001",
            # Both train fingerprints are more than four bits from validation.
            ("session-val", "frame-000000.png"): "00000000000000ff",
            ("session-test", "frame-000000.png"): "ffffffffffffffff",
        }
        with mock.patch.object(
            dataset_contract, "_image_details", side_effect=details_with(allowed_values)
        ):
            validated = validate_dataset(self.fixture.raw)
        output = Path(self.temp.name) / "hamming-threshold-export"
        manifest = export_dataset(validated, output)
        self.assertEqual(manifest["leakage_checks"]["dhash_hamming_threshold"], 4)
        self.assertEqual(
            manifest["leakage_checks"]["cross_split_dhash64_hamming"], "passed"
        )

    def test_rejects_frame_level_split_alias_path_traversal_and_overwrite(self) -> None:
        sessions = self.fixture.document["sessions"]
        assert isinstance(sessions, list)
        sessions.append({"session_id": "session-train", "split": "val"})
        self.fixture.write_annotations()
        with self.assertRaisesRegex(DatasetContractError, "more than one split"):
            validate_dataset(self.fixture.raw)

        self.fixture = CaptureDatasetFixture(Path(self.temp.name) / "traversal")
        self.fixture.image("session-train")["file_name"] = "../session-val/images/frame-000000.png"
        self.fixture.write_annotations()
        with self.assertRaisesRegex(DatasetContractError, "path traversal"):
            validate_dataset(self.fixture.raw)

        self.fixture = CaptureDatasetFixture(Path(self.temp.name) / "overwrite")
        dataset = validate_dataset(self.fixture.raw)
        output = Path(self.temp.name) / "already-exists"
        output.mkdir()
        with self.assertRaisesRegex(DatasetContractError, "refusing to overwrite"):
            export_dataset(dataset, output)
        with self.assertRaisesRegex(DatasetContractError, "path traversal"):
            export_dataset(dataset, Path(self.temp.name) / "safe" / ".." / "escape")

    def test_rejects_symlinked_capture_image(self) -> None:
        image = self.fixture.image("session-train")
        source = self.fixture.raw / "session-train" / str(image["file_name"])
        replacement = Path(self.temp.name) / "replacement.png"
        replacement.write_bytes(source.read_bytes())
        source.unlink()
        try:
            source.symlink_to(replacement)
        except (NotImplementedError, OSError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(DatasetContractError, "contains a symlink"):
            validate_dataset(self.fixture.raw)

    def test_cli_validate_and_export(self) -> None:
        self.assertEqual(main(["validate", "--raw-root", str(self.fixture.raw)]), 0)
        output = Path(self.temp.name) / "cli-export"
        self.assertEqual(
            main(
                [
                    "export",
                    "--raw-root",
                    str(self.fixture.raw),
                    "--output",
                    str(output),
                ]
            ),
            0,
        )
        self.assertTrue((output / OUTPUT_MANIFEST_NAME).is_file())


if __name__ == "__main__":
    unittest.main()
