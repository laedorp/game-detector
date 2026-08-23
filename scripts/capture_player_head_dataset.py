#!/usr/bin/env python3
"""Validate capture sessions and export a deterministic two-class YOLO dataset.

The capture recorder owns immutable raw sessions below ``raw/<session_id>``.
Human review owns a separate ``annotations.json`` at the raw root.  Keeping
those responsibilities separate prevents a recorder label or a frame-level
random split from silently becoming training truth.

Annotation document schema (version 1)::

    {
      "schema_version": 1,
      "categories": [{"id": 0, "name": "player"},
                     {"id": 1, "name": "head"}],
      "sessions": [{"session_id": "session-a", "split": "train"}],
      "images": [{
        "id": 1, "session_id": "session-a",
        "file_name": "images/000001.png", "width": 1920,
        "height": 1080, "sha256": "...", "negative": false
      }],
      "annotations": [{
        "id": 1, "image_id": 1, "category_id": 0,
        "bbox": [100, 100, 200, 500], "area": 100000,
        "iscrowd": 0, "ignore": 0, "visibility": 1.0,
        "occluded": false, "truncated": false
      }, {
        "id": 2, "image_id": 1, "category_id": 1,
        "parent_player_annotation_id": 1,
        "bbox": [150, 100, 80, 80], "area": 6400,
        "iscrowd": 0, "ignore": 0, "visibility": 1.0,
        "occluded": false, "truncated": false
      }]
    }

``visibility`` is the visible fraction in ``[0, 1]``.  A zero-visibility
annotation must be explicitly ignored, and ignored annotations must have zero
visibility.  Ignored annotations remain in the
normalized COCO audit file but are never written to YOLO labels.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = 1
RAW_SCHEMA_NAME = "proaim.capture_player_head.session"
RAW_MANIFEST_NAME = "session.json"
DEFAULT_ANNOTATIONS_NAME = "annotations.json"
OUTPUT_MANIFEST_NAME = "DATASET-MANIFEST.json"
NORMALIZED_COCO_NAME = "annotations.coco.json"
DATASET_YAML_NAME = "dataset.yaml"
SPLITS = ("train", "val", "test")
CATEGORIES = ((0, "player"), (1, "head"))
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SCENARIO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.()-]{0,95}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DHASH_RE = re.compile(r"^[0-9a-f]{16}$")
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 512 * 1024 * 1024
MAX_IMAGE_DIMENSION = 32768
MAX_IMAGE_PIXELS = 200_000_000
PERCEPTUAL_HAMMING_THRESHOLD = 4


class DatasetContractError(ValueError):
    """Raised when capture data is unsafe, ambiguous, or non-reproducible."""


@dataclass
class _DHashNode:
    value: int
    image_by_split: dict[str, int]
    children: dict[int, "_DHashNode"]


class _DHashIndex:
    """Small deterministic BK-tree for bounded 64-bit Hamming queries."""

    def __init__(self) -> None:
        self._root: _DHashNode | None = None

    @staticmethod
    def _distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: str, split: str, image_id: int) -> None:
        numeric = int(value, 16)
        if self._root is None:
            self._root = _DHashNode(numeric, {split: image_id}, {})
            return
        node = self._root
        while True:
            distance = self._distance(numeric, node.value)
            if distance == 0:
                node.image_by_split.setdefault(split, image_id)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _DHashNode(numeric, {split: image_id}, {})
                return
            node = child

    def cross_split_match(
        self, value: str, split: str, maximum_distance: int
    ) -> tuple[int, str, int] | None:
        numeric = int(value, 16)
        if self._root is None:
            return None
        matches: list[tuple[int, str, int]] = []
        pending = [self._root]
        while pending:
            node = pending.pop()
            distance = self._distance(numeric, node.value)
            if distance <= maximum_distance:
                for other_split, image_id in node.image_by_split.items():
                    if other_split != split:
                        matches.append((distance, other_split, image_id))
            lower = max(0, distance - maximum_distance)
            upper = min(64, distance + maximum_distance)
            pending.extend(
                child
                for edge, child in node.children.items()
                if lower <= edge <= upper
            )
        return min(matches, default=None)


@dataclass(frozen=True)
class SourceSession:
    session_id: str
    split: str
    root: Path
    manifest_path: Path
    manifest_sha256: str
    range_label: str
    motion: str
    scenario: str
    capture_description: str
    capture_requested: Mapping[str, Any]
    capture_actual: Mapping[str, Any]


@dataclass(frozen=True)
class SourceImage:
    image_id: int
    session_id: str
    split: str
    file_name: str
    source_path: Path
    width: int
    height: int
    sha256: str
    dhash64: str
    byte_size: int
    negative: bool


@dataclass(frozen=True)
class SourceAnnotation:
    annotation_id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    area: float
    iscrowd: int
    ignore: int
    visibility: float
    occluded: bool
    truncated: bool
    parent_player_annotation_id: int | None


@dataclass(frozen=True)
class ValidatedDataset:
    raw_root: Path
    annotations_path: Path
    annotations_sha256: str
    sessions: tuple[SourceSession, ...]
    images: tuple[SourceImage, ...]
    annotations: tuple[SourceAnnotation, ...]

    def report(self) -> dict[str, Any]:
        image_counts = {split: 0 for split in SPLITS}
        negative_counts = {split: 0 for split in SPLITS}
        session_counts = {split: 0 for split in SPLITS}
        annotation_counts = {
            split: {"player": 0, "head": 0, "ignored": 0} for split in SPLITS
        }
        image_split = {image.image_id: image.split for image in self.images}
        for session in self.sessions:
            session_counts[session.split] += 1
        for image in self.images:
            image_counts[image.split] += 1
            negative_counts[image.split] += int(image.negative)
        for annotation in self.annotations:
            split = image_split[annotation.image_id]
            if annotation.ignore:
                annotation_counts[split]["ignored"] += 1
            else:
                annotation_counts[split][CATEGORIES[annotation.category_id][1]] += 1
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": True,
            "annotations_sha256": self.annotations_sha256,
            "sessions": session_counts,
            "images": image_counts,
            "negative_images": negative_counts,
            "annotations": annotation_counts,
            "session_strata": [
                {
                    "session_id": session.session_id,
                    "split": session.split,
                    "range": session.range_label,
                    "motion": session.motion,
                    "scenario": session.scenario,
                }
                for session in self.sessions
            ],
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_parent_traversal(path: Path, description: str) -> None:
    if ".." in path.parts:
        raise DatasetContractError(f"path traversal in {description}: {path}")


def _reject_symlink_components(path: Path, description: str) -> None:
    _reject_parent_traversal(path, description)
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise DatasetContractError(f"cannot inspect {description}: {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise DatasetContractError(f"{description} contains a symlink: {current}")


def _require_directory(path: Path, description: str) -> Path:
    _reject_parent_traversal(path, description)
    absolute = _absolute(path)
    _reject_symlink_components(absolute, description)
    try:
        mode = os.lstat(absolute).st_mode
    except OSError as exc:
        raise DatasetContractError(f"missing {description}: {absolute}: {exc}") from exc
    if not stat.S_ISDIR(mode):
        raise DatasetContractError(f"{description} is not a directory: {absolute}")
    return absolute


def _require_regular_file(path: Path, description: str) -> Path:
    _reject_parent_traversal(path, description)
    absolute = _absolute(path)
    _reject_symlink_components(absolute, description)
    try:
        mode = os.lstat(absolute).st_mode
    except OSError as exc:
        raise DatasetContractError(f"missing {description}: {absolute}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise DatasetContractError(f"{description} is not a regular file: {absolute}")
    return absolute


def _read_regular_bytes(path: Path, description: str, maximum_bytes: int) -> bytes:
    source = _require_regular_file(path, description)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise DatasetContractError(f"cannot open {description}: {source}: {exc}") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise DatasetContractError(f"{description} is not a regular file: {source}")
        if details.st_size > maximum_bytes:
            raise DatasetContractError(
                f"{description} exceeds the {maximum_bytes}-byte safety limit"
            )
        chunks: list[bytes] = []
        total = 0
        with os.fdopen(descriptor, "rb", closefd=False) as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                total += len(chunk)
                if total > maximum_bytes:
                    raise DatasetContractError(
                        f"{description} exceeds the {maximum_bytes}-byte safety limit"
                    )
                chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json(path: Path, description: str) -> tuple[Mapping[str, Any], bytes]:
    raw = _read_regular_bytes(path, description, MAX_JSON_BYTES)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"invalid UTF-8 JSON in {description}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise DatasetContractError(f"{description} must contain a JSON object")
    return decoded, raw


def _safe_relative(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise DatasetContractError(f"{description} must be a non-empty POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value:
        raise DatasetContractError(f"unsafe {description}: {value!r}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise DatasetContractError(f"path traversal in {description}: {value!r}")
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{value}" for value in range(1, 10)),
        *(f"LPT{value}" for value in range(1, 10)),
    }
    for part in pure.parts:
        if (
            ":" in part
            or part.endswith((" ", "."))
            or any(ord(character) < 32 for character in part)
            or part.split(".", 1)[0].upper() in reserved
        ):
            raise DatasetContractError(f"cross-platform unsafe {description}: {value!r}")
    return value


def _safe_member(root: Path, relative: str, description: str) -> Path:
    safe = _safe_relative(relative, description)
    candidate = root.joinpath(*PurePosixPath(safe).parts)
    _reject_symlink_components(candidate, description)
    return _require_regular_file(candidate, description)


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise DatasetContractError(f"{description} must be a safe identifier")
    return value


def _integer(value: object, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DatasetContractError(f"{description} must be an integer >= {minimum}")
    return value


def _number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetContractError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DatasetContractError(f"{description} must be finite")
    return result


def _sha(value: object, description: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise DatasetContractError(f"{description} must be a lowercase SHA-256")
    return value


def _list(value: object, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise DatasetContractError(f"{description} must be an array")
    return value


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DatasetContractError(f"{description} must be an object with string keys")
    return value


def _image_details(path: Path, expected_sha: str) -> tuple[int, int, int, str]:
    encoded = _read_regular_bytes(path, "capture image", MAX_IMAGE_BYTES)
    actual_sha = _sha256_bytes(encoded)
    if actual_sha != expected_sha:
        raise DatasetContractError(
            f"capture image hash mismatch for {path}: {actual_sha} != {expected_sha}"
        )
    try:
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        raise DatasetContractError(f"cannot decode capture image {path}: {exc}") from exc
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise DatasetContractError(f"cannot decode capture image as three-channel pixels: {path}")
    height, width = (int(image.shape[0]), int(image.shape[1]))
    if (
        width <= 0
        or height <= 0
        or width > MAX_IMAGE_DIMENSION
        or height > MAX_IMAGE_DIMENSION
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise DatasetContractError(f"unsafe decoded image dimensions for {path}: {width}x{height}")

    # Fixed integer BT.601 luminance and integer block means make this dHash
    # independent of Pillow and interpolation implementations.
    blue = image[:, :, 0].astype(np.uint32)
    green = image[:, :, 1].astype(np.uint32)
    red = image[:, :, 2].astype(np.uint32)
    grayscale = ((29 * blue + 150 * green + 77 * red + 128) >> 8).astype(np.uint8)
    cells: list[int] = []
    for row in range(8):
        top = row * height // 8
        bottom = max(top + 1, (row + 1) * height // 8)
        for column in range(9):
            left = column * width // 9
            right = max(left + 1, (column + 1) * width // 9)
            block = grayscale[top:bottom, left:right]
            cells.append(int(block.sum(dtype=np.uint64)) // int(block.size))
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(
                cells[row * 9 + column] > cells[row * 9 + column + 1]
            )
    return width, height, len(encoded), f"{bits:016x}"


def _session_assignments(document: Mapping[str, Any]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for position, raw in enumerate(_list(document.get("sessions"), "annotation sessions")):
        item = _mapping(raw, f"annotation session {position}")
        session_id = _identifier(item.get("session_id"), f"annotation session {position} id")
        split = item.get("split")
        if split not in SPLITS:
            raise DatasetContractError(
                f"session {session_id} split must be train, val, or test"
            )
        if session_id in assignments:
            raise DatasetContractError(
                f"session {session_id} has more than one split assignment"
            )
        assignments[session_id] = str(split)
    if not assignments:
        raise DatasetContractError("annotation sessions are empty")
    missing = [split for split in SPLITS if split not in assignments.values()]
    if missing:
        raise DatasetContractError(
            f"whole-session assignments must populate train, val, and test; missing {missing}"
        )
    return assignments


def _load_raw_session(raw_root: Path, session_id: str, split: str) -> tuple[SourceSession, dict[str, Mapping[str, Any]]]:
    session_root = raw_root / session_id
    _require_directory(session_root, f"raw session {session_id}")
    manifest_path = _require_regular_file(
        session_root / RAW_MANIFEST_NAME, f"raw session {session_id} manifest"
    )
    manifest, manifest_bytes = _read_json(manifest_path, f"raw session {session_id} manifest")
    if (
        manifest.get("schema") != RAW_SCHEMA_NAME
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise DatasetContractError(
            f"raw session {session_id} must use {RAW_SCHEMA_NAME!r} schema version 1"
        )
    if manifest.get("session_id") != session_id:
        raise DatasetContractError(
            f"raw session directory {session_id} disagrees with its manifest session_id"
        )
    if manifest.get("status") != "complete" or manifest.get("complete") is not True:
        raise DatasetContractError(f"raw session {session_id} is not complete")
    range_label = manifest.get("range")
    if range_label not in {"close", "medium", "long", "negative"}:
        raise DatasetContractError(
            f"raw session {session_id} range must be close, medium, long, or negative"
        )
    motion = manifest.get("motion")
    if motion not in {"moving", "stationary"}:
        raise DatasetContractError(
            f"raw session {session_id} motion must be moving or stationary"
        )
    scenario = manifest.get("scenario")
    if not isinstance(scenario, str) or not SCENARIO_RE.fullmatch(scenario):
        raise DatasetContractError(f"raw session {session_id} has an invalid scenario")
    capture = _mapping(manifest.get("capture"), f"raw session {session_id} capture")
    if "error" not in capture or capture.get("error") not in (None, ""):
        raise DatasetContractError(
            f"raw session {session_id} is marked complete but reports a capture error"
        )
    description = capture.get("description")
    if not isinstance(description, str) or not description:
        raise DatasetContractError(
            f"raw session {session_id} capture description must be non-empty"
        )
    requested = _mapping(
        capture.get("requested"), f"raw session {session_id} requested capture settings"
    )
    actual = _mapping(
        capture.get("actual"), f"raw session {session_id} actual capture settings"
    )
    frames: dict[str, Mapping[str, Any]] = {}
    source_sequences: set[int] = set()
    for position, raw_frame in enumerate(_list(manifest.get("frames"), f"raw session {session_id} frames")):
        frame = _mapping(raw_frame, f"raw session {session_id} frame {position}")
        file_name = _safe_relative(
            frame.get("file_name"), f"raw session {session_id} frame file_name"
        )
        if file_name in frames:
            raise DatasetContractError(f"raw session {session_id} repeats frame {file_name}")
        sequence = _integer(
            frame.get("source_sequence"),
            f"raw session {session_id} frame {file_name} source_sequence",
        )
        if sequence in source_sequences:
            raise DatasetContractError(
                f"raw session {session_id} repeats source_sequence {sequence}"
            )
        source_sequences.add(sequence)
        _sha(frame.get("sha256"), f"raw session {session_id} frame {file_name} hash")
        _integer(frame.get("width"), f"raw session {session_id} frame {file_name} width", 1)
        _integer(frame.get("height"), f"raw session {session_id} frame {file_name} height", 1)
        _integer(frame.get("byte_size"), f"raw session {session_id} frame {file_name} byte_size", 1)
        frames[file_name] = frame
    if not frames:
        raise DatasetContractError(f"raw session {session_id} has no recorded frames")
    return (
        SourceSession(
            session_id=session_id,
            split=split,
            root=session_root,
            manifest_path=manifest_path,
            manifest_sha256=_sha256_bytes(manifest_bytes),
            range_label=str(range_label),
            motion=str(motion),
            scenario=scenario,
            capture_description=description,
            capture_requested=dict(requested),
            capture_actual=dict(actual),
        ),
        frames,
    )


def _normalize_bbox(value: object, annotation_id: int, image: SourceImage) -> tuple[float, float, float, float]:
    raw = _list(value, f"annotation {annotation_id} bbox")
    if len(raw) != 4:
        raise DatasetContractError(f"annotation {annotation_id} bbox must be [x,y,width,height]")
    x, y, width, height = (
        _number(item, f"annotation {annotation_id} bbox coordinate") for item in raw
    )
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise DatasetContractError(f"annotation {annotation_id} has an invalid bbox")
    if x + width > image.width + 1e-6 or y + height > image.height + 1e-6:
        raise DatasetContractError(f"annotation {annotation_id} bbox leaves image {image.image_id}")
    return (x, y, width, height)


def _contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    tolerance = 1e-6
    return (
        ix + tolerance >= ox
        and iy + tolerance >= oy
        and ix + iw <= ox + ow + tolerance
        and iy + ih <= oy + oh + tolerance
    )


def validate_dataset(raw_root: Path, annotations_path: Path | None = None) -> ValidatedDataset:
    """Validate raw sessions and their separate COCO annotation document."""

    raw_root = _require_directory(raw_root, "capture dataset raw root")
    annotations_path = _require_regular_file(
        annotations_path if annotations_path is not None else raw_root / DEFAULT_ANNOTATIONS_NAME,
        "capture annotation document",
    )
    document, annotation_bytes = _read_json(annotations_path, "capture annotation document")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise DatasetContractError("capture annotation schema_version must be 1")
    categories = _list(document.get("categories"), "COCO categories")
    normalized_categories = []
    for position, raw_category in enumerate(categories):
        category = _mapping(raw_category, f"COCO category {position}")
        normalized_categories.append((category.get("id"), category.get("name")))
    if normalized_categories != list(CATEGORIES):
        raise DatasetContractError(
            "COCO categories must be exactly player=0 and head=1 in that order"
        )

    assignments = _session_assignments(document)
    sessions: list[SourceSession] = []
    raw_frames: dict[str, dict[str, Mapping[str, Any]]] = {}
    for session_id in sorted(assignments):
        session, frames = _load_raw_session(raw_root, session_id, assignments[session_id])
        sessions.append(session)
        raw_frames[session_id] = frames

    images_by_id: dict[int, SourceImage] = {}
    selected_sources: set[tuple[str, str]] = set()
    seen_session_images = {session_id: 0 for session_id in assignments}
    sha_splits: dict[str, tuple[str, int]] = {}
    dhash_index = _DHashIndex()
    for position, raw_image in enumerate(_list(document.get("images"), "COCO images")):
        image = _mapping(raw_image, f"COCO image {position}")
        image_id = _integer(image.get("id"), f"COCO image {position} id", 1)
        if image_id in images_by_id:
            raise DatasetContractError(f"duplicate COCO image id: {image_id}")
        session_id = _identifier(image.get("session_id"), f"COCO image {image_id} session_id")
        if session_id not in assignments:
            raise DatasetContractError(f"COCO image {image_id} uses an unassigned session")
        file_name = _safe_relative(image.get("file_name"), f"COCO image {image_id} file_name")
        if Path(file_name).suffix.casefold() not in IMAGE_SUFFIXES:
            raise DatasetContractError(
                f"COCO image {image_id} has an unsupported image suffix"
            )
        source_identity = (session_id, file_name)
        if source_identity in selected_sources:
            raise DatasetContractError(f"raw frame selected more than once: {source_identity}")
        selected_sources.add(source_identity)
        frame = raw_frames[session_id].get(file_name)
        if frame is None:
            raise DatasetContractError(
                f"COCO image {image_id} is absent from raw session {session_id}: {file_name}"
            )
        width = _integer(image.get("width"), f"COCO image {image_id} width", 1)
        height = _integer(image.get("height"), f"COCO image {image_id} height", 1)
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION or width * height > MAX_IMAGE_PIXELS:
            raise DatasetContractError(f"unsafe COCO dimensions for image {image_id}")
        expected_sha = _sha(image.get("sha256"), f"COCO image {image_id} hash")
        if expected_sha != frame.get("sha256"):
            raise DatasetContractError(
                f"COCO image {image_id} hash disagrees with raw session manifest"
            )
        if width != frame.get("width") or height != frame.get("height"):
            raise DatasetContractError(
                f"COCO image {image_id} dimensions disagree with raw session manifest"
            )
        if not isinstance(image.get("negative"), bool):
            raise DatasetContractError(f"COCO image {image_id} must explicitly set boolean negative")
        source_path = _safe_member(
            raw_root / session_id, file_name, f"COCO image {image_id} source"
        )
        decoded_width, decoded_height, byte_size, dhash64 = _image_details(source_path, expected_sha)
        if (decoded_width, decoded_height) != (width, height):
            raise DatasetContractError(
                f"COCO image {image_id} dimensions disagree with decoded bytes: "
                f"{width}x{height} != {decoded_width}x{decoded_height}"
            )
        if frame.get("byte_size") != byte_size:
            raise DatasetContractError(
                f"COCO image {image_id} byte size disagrees with raw session manifest"
            )
        supplied_dhash = image.get("perceptual_dhash64")
        if supplied_dhash is not None:
            if not isinstance(supplied_dhash, str) or not DHASH_RE.fullmatch(supplied_dhash):
                raise DatasetContractError(f"COCO image {image_id} has invalid perceptual_dhash64")
            if supplied_dhash != dhash64:
                raise DatasetContractError(f"COCO image {image_id} perceptual hash mismatch")
        split = assignments[session_id]
        previous_sha = sha_splits.get(expected_sha)
        if previous_sha is not None and previous_sha[0] != split:
            raise DatasetContractError(
                f"cross-split exact SHA-256 leakage between images {previous_sha[1]} "
                f"({previous_sha[0]}) and {image_id} ({split})"
            )
        sha_splits.setdefault(expected_sha, (split, image_id))
        perceptual_match = dhash_index.cross_split_match(
            dhash64, split, PERCEPTUAL_HAMMING_THRESHOLD
        )
        if perceptual_match is not None:
            distance, previous_split, previous_image_id = perceptual_match
            raise DatasetContractError(
                f"cross-split dHash64 leakage (Hamming distance {distance}) between "
                f"images {previous_image_id} ({previous_split}) and {image_id} ({split})"
            )
        dhash_index.add(dhash64, split, image_id)
        images_by_id[image_id] = SourceImage(
            image_id=image_id,
            session_id=session_id,
            split=split,
            file_name=file_name,
            source_path=source_path,
            width=width,
            height=height,
            sha256=expected_sha,
            dhash64=dhash64,
            byte_size=byte_size,
            negative=bool(image["negative"]),
        )
        seen_session_images[session_id] += 1
    if not images_by_id:
        raise DatasetContractError("COCO image inventory is empty")
    unused_sessions = sorted(
        session_id for session_id, count in seen_session_images.items() if count == 0
    )
    if unused_sessions:
        raise DatasetContractError(f"assigned sessions contain no selected images: {unused_sessions}")

    annotations_by_id: dict[int, SourceAnnotation] = {}
    annotations_by_image: dict[int, list[SourceAnnotation]] = {
        image_id: [] for image_id in images_by_id
    }
    unique_boxes: set[tuple[int, int, float, float, float, float]] = set()
    for position, raw_annotation in enumerate(
        _list(document.get("annotations"), "COCO annotations")
    ):
        item = _mapping(raw_annotation, f"COCO annotation {position}")
        annotation_id = _integer(item.get("id"), f"COCO annotation {position} id", 1)
        if annotation_id in annotations_by_id:
            raise DatasetContractError(f"duplicate COCO annotation id: {annotation_id}")
        image_id = _integer(item.get("image_id"), f"annotation {annotation_id} image_id", 1)
        image = images_by_id.get(image_id)
        if image is None:
            raise DatasetContractError(f"annotation {annotation_id} references an unknown image")
        category_id = _integer(item.get("category_id"), f"annotation {annotation_id} category_id")
        if category_id not in (0, 1):
            raise DatasetContractError(f"annotation {annotation_id} category must be player=0 or head=1")
        bbox = _normalize_bbox(item.get("bbox"), annotation_id, image)
        box_key = (image_id, category_id, *bbox)
        if box_key in unique_boxes:
            raise DatasetContractError(f"annotation {annotation_id} duplicates an existing bbox")
        unique_boxes.add(box_key)
        area = bbox[2] * bbox[3]
        if "area" not in item or not math.isclose(
            _number(item.get("area"), f"annotation {annotation_id} area"),
            area,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise DatasetContractError(f"annotation {annotation_id} area differs from bbox")
        iscrowd = item.get("iscrowd")
        ignore = item.get("ignore")
        if isinstance(iscrowd, bool) or iscrowd != 0:
            raise DatasetContractError(f"annotation {annotation_id} must explicitly set iscrowd=0")
        if isinstance(ignore, bool) or ignore not in (0, 1):
            raise DatasetContractError(f"annotation {annotation_id} ignore must be integer 0 or 1")
        visibility = _number(item.get("visibility"), f"annotation {annotation_id} visibility")
        if visibility < 0.0 or visibility > 1.0:
            raise DatasetContractError(f"annotation {annotation_id} visibility must be in [0,1]")
        if visibility == 0.0 and ignore != 1:
            raise DatasetContractError(
                f"unresolvable annotation {annotation_id} must be explicitly ignored"
            )
        if ignore == 1 and visibility != 0.0:
            raise DatasetContractError(
                f"ignored annotation {annotation_id} must have zero visibility"
            )
        if not isinstance(item.get("occluded"), bool):
            raise DatasetContractError(f"annotation {annotation_id} must set boolean occluded")
        if not isinstance(item.get("truncated"), bool):
            raise DatasetContractError(f"annotation {annotation_id} must set boolean truncated")
        if visibility < 1.0 and visibility > 0.0 and not (
            item["occluded"] or item["truncated"]
        ):
            raise DatasetContractError(
                f"partially visible annotation {annotation_id} must be occluded or truncated"
            )
        parent: int | None
        if category_id == 1:
            parent = _integer(
                item.get("parent_player_annotation_id"),
                f"head annotation {annotation_id} parent_player_annotation_id",
                1,
            )
        else:
            if "parent_player_annotation_id" in item:
                raise DatasetContractError(
                    f"player annotation {annotation_id} must not have a head parent field"
                )
            parent = None
        normalized = SourceAnnotation(
            annotation_id=annotation_id,
            image_id=image_id,
            category_id=category_id,
            bbox=bbox,
            area=area,
            iscrowd=0,
            ignore=int(ignore),
            visibility=visibility,
            occluded=bool(item["occluded"]),
            truncated=bool(item["truncated"]),
            parent_player_annotation_id=parent,
        )
        annotations_by_id[annotation_id] = normalized
        annotations_by_image[image_id].append(normalized)

    associated_heads: dict[int, SourceAnnotation] = {}
    for annotation in annotations_by_id.values():
        if annotation.category_id != 1:
            continue
        assert annotation.parent_player_annotation_id is not None
        parent = annotations_by_id.get(annotation.parent_player_annotation_id)
        if parent is None or parent.category_id != 0:
            raise DatasetContractError(
                f"head annotation {annotation.annotation_id} parent is not a player annotation"
            )
        if parent.image_id != annotation.image_id:
            raise DatasetContractError(
                f"head annotation {annotation.annotation_id} parent is on another image"
            )
        if not _contains(parent.bbox, annotation.bbox):
            raise DatasetContractError(
                f"head annotation {annotation.annotation_id} parent player does not contain its bbox"
            )
        if parent.annotation_id in associated_heads:
            raise DatasetContractError(
                f"player annotation {parent.annotation_id} has more than one associated head"
            )
        associated_heads[parent.annotation_id] = annotation
        if annotation.ignore == 0 and parent.ignore != 0:
            raise DatasetContractError(
                f"exportable head annotation {annotation.annotation_id} has an ignored parent"
            )
        if (
            images_by_id[annotation.image_id].split == "train"
            and annotation.ignore == 0
            and annotation.visibility <= 0.0
        ):
            raise DatasetContractError(
                f"unresolvable head annotation {annotation.annotation_id} would enter train output"
            )

    for image_id, image in images_by_id.items():
        image_annotations = annotations_by_image[image_id]
        if image.negative:
            if image_annotations:
                raise DatasetContractError(
                    f"negative image {image_id} must have zero annotations"
                )
        elif not any(
            annotation.category_id == 0
            and annotation.ignore == 0
            and annotation.visibility > 0.0
            for annotation in image_annotations
        ):
            raise DatasetContractError(
                f"positive image {image_id} must contain at least one exportable player annotation"
            )

    for player in annotations_by_id.values():
        if (
            player.category_id != 0
            or player.ignore != 0
            or player.visibility <= 0.0
            or player.occluded
            or player.truncated
        ):
            continue
        head = associated_heads.get(player.annotation_id)
        if head is None:
            raise DatasetContractError(
                f"fully visible player annotation {player.annotation_id} requires exactly one "
                "associated head record"
            )
        head_is_exportable = head.ignore == 0 and head.visibility > 0.0
        head_is_explicitly_unresolvable = head.ignore == 1 and head.visibility == 0.0
        if not (head_is_exportable or head_is_explicitly_unresolvable):
            raise DatasetContractError(
                f"fully visible player annotation {player.annotation_id} head must be "
                "exportable or explicitly ignored with zero visibility"
            )

    exportable_by_split = {
        split: {0: 0, 1: 0} for split in SPLITS
    }
    for annotation in annotations_by_id.values():
        if annotation.ignore == 0 and annotation.visibility > 0.0:
            split = images_by_id[annotation.image_id].split
            exportable_by_split[split][annotation.category_id] += 1
    for split in SPLITS:
        for category_id, category_name in CATEGORIES:
            if exportable_by_split[split][category_id] == 0:
                raise DatasetContractError(
                    f"split {split} has no exportable {category_name} annotations"
                )

    return ValidatedDataset(
        raw_root=raw_root,
        annotations_path=annotations_path,
        annotations_sha256=_sha256_bytes(annotation_bytes),
        sessions=tuple(sorted(sessions, key=lambda item: item.session_id)),
        images=tuple(sorted(images_by_id.values(), key=lambda item: item.image_id)),
        annotations=tuple(
            sorted(annotations_by_id.values(), key=lambda item: item.annotation_id)
        ),
    )


def _format_float(value: float) -> str:
    # Ten fixed decimal places are more precise than image coordinates require,
    # deterministic across supported CPython versions, and accepted by YOLO.
    return f"{value:.10f}"


def _output_image_name(image: SourceImage) -> str:
    suffix = Path(image.file_name).suffix.casefold()
    if suffix not in IMAGE_SUFFIXES:
        raise DatasetContractError(f"unsupported capture image suffix: {image.file_name}")
    return f"{image.image_id:012d}{suffix}"


def _yolo_line(annotation: SourceAnnotation, image: SourceImage) -> str:
    x, y, width, height = annotation.bbox
    values = (
        (x + width / 2.0) / image.width,
        (y + height / 2.0) / image.height,
        width / image.width,
        height / image.height,
    )
    return f"{annotation.category_id} " + " ".join(_format_float(value) for value in values)


def _write_new(path: Path, content: bytes) -> None:
    _reject_symlink_components(path, "dataset output member")
    try:
        with path.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as exc:
        raise DatasetContractError(f"refusing to overwrite output member: {path}") from exc


def _copy_new(source: Path, destination: Path, expected_sha: str) -> None:
    content = _read_regular_bytes(source, "capture image for export", MAX_IMAGE_BYTES)
    actual = _sha256_bytes(content)
    if actual != expected_sha:
        raise DatasetContractError(
            f"capture image changed after validation: {source}: {actual} != {expected_sha}"
        )
    _write_new(destination, content)


def _normalized_coco(dataset: ValidatedDataset) -> dict[str, Any]:
    images = []
    for image in dataset.images:
        images.append(
            {
                "id": image.image_id,
                "session_id": image.session_id,
                "split": image.split,
                "file_name": f"images/{image.split}/{_output_image_name(image)}",
                "width": image.width,
                "height": image.height,
                "sha256": image.sha256,
                "perceptual_dhash64": image.dhash64,
                "negative": image.negative,
            }
        )
    annotations = []
    for annotation in dataset.annotations:
        record: dict[str, Any] = {
            "id": annotation.annotation_id,
            "image_id": annotation.image_id,
            "category_id": annotation.category_id,
            "bbox": list(annotation.bbox),
            "area": annotation.area,
            "iscrowd": annotation.iscrowd,
            "ignore": annotation.ignore,
            "visibility": annotation.visibility,
            "occluded": annotation.occluded,
            "truncated": annotation.truncated,
        }
        if annotation.parent_player_annotation_id is not None:
            record["parent_player_annotation_id"] = annotation.parent_player_annotation_id
        annotations.append(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "categories": [{"id": category_id, "name": name} for category_id, name in CATEGORIES],
        "images": images,
        "annotations": annotations,
    }


def _dataset_yaml() -> bytes:
    return (
        "# Deterministic two-class capture-card dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names:\n"
        "  0: player\n"
        "  1: head\n"
    ).encode("utf-8")


def _member_record(root: Path, path: Path) -> dict[str, Any]:
    content = _read_regular_bytes(path, "exported dataset member", MAX_IMAGE_BYTES)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(content),
        "sha256": _sha256_bytes(content),
    }


def _publish_new_directory(staging: Path, output: Path) -> None:
    """Atomically publish ``staging`` without replacing an existing path."""

    if os.path.lexists(output):
        raise DatasetContractError(f"refusing to overwrite existing output path: {output}")
    if os.name == "posix":
        parent_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            parent_descriptor = os.open(output.parent, parent_flags)
        except OSError as exc:
            raise DatasetContractError(
                f"cannot anchor output parent for atomic publication: {output.parent}: {exc}"
            ) from exc
        try:
            library = ctypes.CDLL(None, use_errno=True)
            renameat2 = getattr(library, "renameat2", None)
            if renameat2 is None:
                raise DatasetContractError(
                    "this POSIX platform lacks atomic no-replace directory publication"
                )
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                parent_descriptor,
                os.fsencode(staging.name),
                parent_descriptor,
                os.fsencode(output.name),
                1,  # RENAME_NOREPLACE
            )
            if result == 0:
                return
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise DatasetContractError(
                    f"refusing to overwrite existing output path: {output}"
                )
            raise DatasetContractError(
                f"atomic no-replace publication failed for {output}: "
                f"[{error}] {os.strerror(error)}"
            )
        finally:
            os.close(parent_descriptor)
    else:
        # os.rename is no-replace for a directory on Windows.
        try:
            os.rename(staging, output)
        except FileExistsError as exc:
            raise DatasetContractError(
                f"refusing to overwrite existing output path: {output}"
            ) from exc


def export_dataset(dataset: ValidatedDataset, output_root: Path) -> dict[str, Any]:
    """Export a validated dataset without overwriting any existing path."""

    _reject_parent_traversal(output_root, "dataset output path")
    output_root = _absolute(output_root)
    _reject_symlink_components(output_root, "dataset output path")
    if os.path.lexists(output_root):
        raise DatasetContractError(f"refusing to overwrite existing output path: {output_root}")
    parent = _require_directory(output_root.parent, "dataset output parent")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=parent))
    published = False
    try:
        current_annotations = _read_regular_bytes(
            dataset.annotations_path, "capture annotation document", MAX_JSON_BYTES
        )
        if _sha256_bytes(current_annotations) != dataset.annotations_sha256:
            raise DatasetContractError("capture annotation document changed after validation")
        for session in dataset.sessions:
            current_manifest = _read_regular_bytes(
                session.manifest_path,
                f"raw session {session.session_id} manifest",
                MAX_JSON_BYTES,
            )
            if _sha256_bytes(current_manifest) != session.manifest_sha256:
                raise DatasetContractError(
                    f"raw session {session.session_id} manifest changed after validation"
                )
        for split in SPLITS:
            (staging / "images" / split).mkdir(parents=True, exist_ok=False)
            (staging / "labels" / split).mkdir(parents=True, exist_ok=False)

        annotations_by_image: dict[int, list[SourceAnnotation]] = {
            image.image_id: [] for image in dataset.images
        }
        for annotation in dataset.annotations:
            annotations_by_image[annotation.image_id].append(annotation)
        for image in dataset.images:
            name = _output_image_name(image)
            image_destination = staging / "images" / image.split / name
            label_destination = staging / "labels" / image.split / f"{Path(name).stem}.txt"
            _copy_new(image.source_path, image_destination, image.sha256)
            exportable = [
                annotation
                for annotation in annotations_by_image[image.image_id]
                if annotation.ignore == 0 and annotation.visibility > 0.0
            ]
            if image.split == "train" and any(
                annotation.category_id == 1 and annotation.visibility <= 0.0
                for annotation in exportable
            ):
                raise DatasetContractError(
                    f"unresolvable head would enter train output for image {image.image_id}"
                )
            lines = [
                _yolo_line(annotation, image)
                for annotation in sorted(exportable, key=lambda item: item.annotation_id)
            ]
            label_content = (("\n".join(lines) + "\n") if lines else "").encode("ascii")
            _write_new(label_destination, label_content)

        _write_new(staging / DATASET_YAML_NAME, _dataset_yaml())
        _write_new(staging / NORMALIZED_COCO_NAME, _canonical_json_bytes(_normalized_coco(dataset)))

        member_paths = sorted(
            path
            for path in staging.rglob("*")
            if path.is_file() and path.name != OUTPUT_MANIFEST_NAME
        )
        members = [_member_record(staging, path) for path in member_paths]
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_kind": "capture-card-player-head-yolo",
            "classes": [{"id": category_id, "name": name} for category_id, name in CATEGORIES],
            "source": {
                "annotations_path": dataset.annotations_path.name,
                "annotations_sha256": dataset.annotations_sha256,
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "split": session.split,
                        "manifest_sha256": session.manifest_sha256,
                        "range": session.range_label,
                        "motion": session.motion,
                        "scenario": session.scenario,
                        "capture": {
                            "description": session.capture_description,
                            "requested": dict(session.capture_requested),
                            "actual": dict(session.capture_actual),
                        },
                    }
                    for session in dataset.sessions
                ],
            },
            "split_unit": "capture_session",
            "frame_level_random_split_allowed": False,
            "leakage_checks": {
                "cross_split_exact_sha256": "passed",
                "cross_split_dhash64_hamming": "passed",
                "dhash_algorithm": "dhash64-9x8-blockmean-bt601-integer",
                "dhash_hamming_threshold": PERCEPTUAL_HAMMING_THRESHOLD,
            },
            "summary": dataset.report(),
            "members": members,
        }
        manifest = dict(payload)
        manifest["manifest_payload_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
        _write_new(staging / OUTPUT_MANIFEST_NAME, _canonical_json_bytes(manifest))
        _publish_new_directory(staging, output_root)
        published = True
        return manifest
    except Exception:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or export immutable capture-card player/head sessions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "export"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--raw-root",
            type=Path,
            default=Path("datasets/capture_player_head/raw"),
            help="raw session root (default: datasets/capture_player_head/raw)",
        )
        subparser.add_argument(
            "--annotations",
            type=Path,
            help="separate COCO annotation document (default: RAW_ROOT/annotations.json)",
        )
        if command == "export":
            subparser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        dataset = validate_dataset(args.raw_root, args.annotations)
        if args.command == "validate":
            result = dataset.report()
        else:
            result = export_dataset(dataset, args.output)
    except DatasetContractError as exc:
        print(f"capture dataset contract failed: {exc}", file=sys.stderr)
        return 2
    print(_canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
