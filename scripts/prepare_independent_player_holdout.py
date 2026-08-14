#!/usr/bin/env python3
"""Prepare and verify an offline, session-isolated player-detection holdout.

This tool deliberately has no downloader and no frame-splitting operation.  It
packages only caller-supplied local files, records exact hashes, and rejects a
capture session that appears in any supplied reference pool.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence, TypeVar

import cv2
import numpy as np


SCHEMA_VERSION = 1
MANIFEST_NAME = "HOLDOUT-MANIFEST.json"
DATASET_KIND = "independent-player-detection-holdout"
POOL_DEVELOPMENT = "development"
POOL_SEALED = "sealed_release_holdout"
POOLS = frozenset({POOL_DEVELOPMENT, POOL_SEALED})
IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_IMAGE_ENCODED_BYTES = 512 * 1024 * 1024
MAX_IMAGE_DIMENSION = 32768
MAX_IMAGE_PIXELS = 200_000_000
DEFAULT_PERCEPTUAL_HAMMING_THRESHOLD = 4
PROJECTED_HEIGHT_REFERENCE_PX = 1080
GATING_COUNT_KEYS = (
    "target_33_64",
    "target_65_96",
    "target_gt_96",
    "reviewed_negatives",
)


class HoldoutContractError(ValueError):
    """Raised when a holdout violates the isolation or content contract."""


class GateMinimums(NamedTuple):
    target_le_32: int = 150
    target_33_64: int = 400
    target_65_96: int = 250
    target_gt_96: int = 250
    reviewed_negatives: int = 1_000

    def as_dict(self) -> dict[str, int]:
        return self._asdict()


PINNED_RELEASE_MINIMUMS = GateMinimums()
_EvaluationResult = TypeVar("_EvaluationResult")


def _source_group_inventory(document: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize capture-session diversity from an already normalized COCO object."""

    images = _require_list(document.get("images"), "normalized COCO images")
    annotations = _require_list(
        document.get("annotations"), "normalized COCO annotations"
    )
    image_sessions: dict[int, str] = {}
    positive_images: set[int] = set()
    for raw_image in images:
        image = _require_mapping(raw_image, "normalized COCO image")
        image_id = _require_int(image.get("id"), "normalized COCO image id")
        image_sessions[image_id] = _require_identifier(
            image.get("session_id"), f"normalized COCO image {image_id} session"
        )
    target_sessions: dict[str, set[str]] = {
        "target_le_32": set(),
        "target_33_64": set(),
        "target_65_96": set(),
        "target_gt_96": set(),
    }
    for raw_annotation in annotations:
        annotation = _require_mapping(raw_annotation, "normalized COCO annotation")
        image_id = _require_int(
            annotation.get("image_id"), "normalized COCO annotation image_id"
        )
        projected_height = _number(
            annotation.get("projected_height_px_at_1080p"),
            "normalized projected target height",
        )
        if projected_height <= 32:
            bucket = "target_le_32"
        elif projected_height <= 64:
            bucket = "target_33_64"
        elif projected_height <= 96:
            bucket = "target_65_96"
        else:
            bucket = "target_gt_96"
        target_sessions[bucket].add(image_sessions[image_id])
        positive_images.add(image_id)
    negative_sessions = {
        session_id
        for image_id, session_id in image_sessions.items()
        if image_id not in positive_images
    }
    return {
        "definition": "distinct normalized COCO image session_id values",
        "overall_capture_sessions": len(set(image_sessions.values())),
        "target_bearing_capture_sessions": {
            key: len(target_sessions[key]) for key in sorted(target_sessions)
        },
        "reviewed_negative_capture_sessions": len(negative_sessions),
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HoldoutContractError(f"cannot open regular file for hashing: {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise HoldoutContractError(f"cannot hash non-regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path, description: str) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise HoldoutContractError(f"cannot inspect {description}: {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise HoldoutContractError(f"{description} contains a symlink: {current}")


def _require_regular_file(path: Path, description: str) -> Path:
    absolute = _absolute(path)
    _reject_symlink_components(absolute, description)
    try:
        mode = os.lstat(absolute).st_mode
    except OSError as exc:
        raise HoldoutContractError(f"missing {description}: {absolute}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise HoldoutContractError(f"{description} is not a regular file: {absolute}")
    return absolute


def _require_directory(path: Path, description: str) -> Path:
    absolute = _absolute(path)
    _reject_symlink_components(absolute, description)
    try:
        mode = os.lstat(absolute).st_mode
    except OSError as exc:
        raise HoldoutContractError(f"missing {description}: {absolute}: {exc}") from exc
    if not stat.S_ISDIR(mode):
        raise HoldoutContractError(f"{description} is not a directory: {absolute}")
    return absolute


def _read_regular_bytes(
    path: Path,
    description: str,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    source = _require_regular_file(path, description)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise HoldoutContractError(f"cannot open {description}: {source}: {exc}") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise HoldoutContractError(f"{description} is not a regular file: {source}")
        if maximum_bytes is not None and details.st_size > maximum_bytes:
            raise HoldoutContractError(
                f"{description} exceeds the {maximum_bytes}-byte limit"
            )
        chunks: list[bytes] = []
        total = 0
        with os.fdopen(descriptor, "rb", closefd=False) as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                total += len(chunk)
                if maximum_bytes is not None and total > maximum_bytes:
                    raise HoldoutContractError(
                        f"{description} exceeds the {maximum_bytes}-byte limit"
                    )
                chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _safe_relative(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise HoldoutContractError(f"{description} must be a non-empty POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise HoldoutContractError(f"unsafe {description}: {value!r}")
    if pure.as_posix() != value:
        raise HoldoutContractError(f"non-canonical {description}: {value!r}")
    windows_reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    for part in pure.parts:
        if (
            ":" in part
            or part.endswith((" ", "."))
            or any(ord(character) < 32 for character in part)
            or part.split(".", 1)[0].upper() in windows_reserved
        ):
            raise HoldoutContractError(f"cross-platform unsafe {description}: {value!r}")
    return value


def _safe_member(root: Path, value: object, description: str) -> Path:
    relative = _safe_relative(value, description)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    return _require_regular_file(candidate, description)


def _require_mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HoldoutContractError(f"{description} must be an object")
    return value


def _require_exact_keys(
    value: object,
    expected: set[str],
    description: str,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, description)
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise HoldoutContractError(
            f"{description} fields differ from the schema; "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )
    return mapping


def _require_list(value: object, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise HoldoutContractError(f"{description} must be an array")
    return value


def _require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HoldoutContractError(f"{description} must be a non-empty string")
    return value


def _require_identifier(value: object, description: str) -> str:
    text = _require_string(value, description)
    if not IDENTIFIER_RE.fullmatch(text) or text in {".", ".."}:
        raise HoldoutContractError(f"invalid {description}: {text!r}")
    _safe_relative(text, description)
    return text


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise HoldoutContractError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: object, description: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise HoldoutContractError(f"{description} must be an exact lowercase 40/64-hex commit")
    return value


def _parse_utc(text: str) -> datetime:
    return datetime.strptime(
        text,
        "%Y-%m-%dT%H:%M:%S.%fZ" if "." in text else "%Y-%m-%dT%H:%M:%SZ",
    )


def _require_utc(value: object, description: str) -> str:
    text = _require_string(value, description)
    if not UTC_RE.fullmatch(text):
        raise HoldoutContractError(f"{description} must be an explicit UTC timestamp ending in Z")
    try:
        _parse_utc(text)
    except ValueError as exc:
        raise HoldoutContractError(f"{description} is not a real UTC calendar time") from exc
    return text


def _require_int(value: object, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HoldoutContractError(f"{description} must be an integer >= {minimum}")
    return value


def _load_json(path: Path, description: str) -> tuple[Any, bytes]:
    source = _require_regular_file(path, description)
    try:
        raw = _read_regular_bytes(source, description, maximum_bytes=MAX_JSON_BYTES)

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise HoldoutContractError(
                        f"{description} contains duplicate JSON key {key!r}"
                    )
                result[key] = item
            return result

        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys), raw
    except HoldoutContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HoldoutContractError(f"cannot parse {description}: {source}: {exc}") from exc


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as exc:
        raise HoldoutContractError(f"refusing to overwrite output member: {path}") from exc


def _fsync_regular_file(path: Path, description: str) -> None:
    """Flush one already-published regular file through a no-follow handle."""

    source = _require_regular_file(path, description)
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        flush = kernel32.FlushFileBuffers
        flush.argtypes = [ctypes.c_void_p]
        flush.restype = ctypes.c_int
        close = kernel32.CloseHandle
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int
        handle = create_file(
            str(source),
            0x40000000,  # GENERIC_WRITE; required by FlushFileBuffers
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,  # OPEN_EXISTING
            0,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle in (None, invalid_handle):
            error = ctypes.get_last_error()
            raise HoldoutContractError(
                f"cannot open {description} for durable flush: Windows error {error}"
            )
        try:
            if not flush(handle):
                error = ctypes.get_last_error()
                raise HoldoutContractError(
                    f"cannot durably flush {description}: Windows error {error}"
                )
        finally:
            close(handle)
        return
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise HoldoutContractError(f"{description} is not a regular file")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except HoldoutContractError:
        raise
    except OSError as exc:
        raise HoldoutContractError(
            f"cannot durably flush {description}: {exc}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    """Durably flush directory-entry changes or fail closed on this platform."""

    directory = _require_directory(path, "durability directory")
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(directory, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise HoldoutContractError(
                f"cannot durably flush directory {directory}: {exc}"
            ) from exc
        return

    # Windows requires a directory handle opened with BACKUP_SEMANTICS.  Do
    # not silently pretend durability if the filesystem cannot flush it.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    flush = kernel32.FlushFileBuffers
    flush.argtypes = [ctypes.c_void_p]
    flush.restype = ctypes.c_int
    close = kernel32.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    close.restype = ctypes.c_int
    handle = create_file(
        str(directory),
        0x40000000,  # GENERIC_WRITE; required by FlushFileBuffers
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,  # OPEN_EXISTING
        0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle in (None, invalid_handle):
        error = ctypes.get_last_error()
        raise HoldoutContractError(
            f"cannot open directory for durable flush: Windows error {error}"
        )
    try:
        if not flush(handle):
            error = ctypes.get_last_error()
            raise HoldoutContractError(
                f"cannot durably flush directory: Windows error {error}"
            )
    finally:
        close(handle)


def _copy_verified(source: Path, destination: Path, expected_sha256: str, description: str) -> dict[str, Any]:
    source = _require_regular_file(source, description)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = sha256()
    total = 0
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise HoldoutContractError(f"cannot open {description}: {source}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise HoldoutContractError(f"{description} is not a regular file: {source}")
        with os.fdopen(descriptor, "rb", closefd=False) as input_file, destination.open("xb") as output:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as exc:
        raise HoldoutContractError(f"refusing to overwrite output member: {destination}") from exc
    finally:
        os.close(descriptor)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        try:
            destination.unlink()
        except OSError:
            pass
        raise HoldoutContractError(
            f"{description} hash mismatch: {actual} != {expected_sha256}"
        )
    return {"bytes": total, "sha256": actual}


def _validate_minimums(minimums: GateMinimums) -> GateMinimums:
    if not isinstance(minimums, GateMinimums):
        try:
            minimums = GateMinimums(**dict(minimums))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise HoldoutContractError("gate minimums have invalid keys") from exc
    for key, value in minimums.as_dict().items():
        _require_int(value, f"minimum {key}")
    return minimums


def _gate_pass(counts: Mapping[str, int], minimums: GateMinimums) -> bool:
    values = minimums.as_dict()
    return all(counts[key] >= values[key] for key in GATING_COUNT_KEYS)


def _image_fingerprint(
    path: Path,
    expected_width: int,
    expected_height: int,
    expected_sha256: str,
) -> str:
    try:
        raw = _read_regular_bytes(
            path,
            "image for perceptual hashing",
            maximum_bytes=MAX_IMAGE_ENCODED_BYTES,
        )
        actual_sha256 = sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise HoldoutContractError(
                f"image hash mismatch while decoding {path}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        encoded = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise HoldoutContractError(f"cannot decode a three-channel image: {path}")
        height, width = image.shape[:2]
        if width != expected_width or height != expected_height:
            raise HoldoutContractError(
                f"image dimensions differ from COCO metadata for {path}: "
                f"{width}x{height} != {expected_width}x{expected_height}"
            )
        if width <= 0 or height <= 0 or width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise HoldoutContractError(f"unsafe image dimensions for {path}: {width}x{height}")
        if width * height > MAX_IMAGE_PIXELS:
            raise HoldoutContractError(f"image exceeds decoded-pixel limit: {path}")
        # Fixed integer BT.601-ish luminance and 9x8 block means avoid making
        # the hash depend on Pillow or an interpolation implementation.
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
    except HoldoutContractError:
        raise
    except Exception as exc:
        raise HoldoutContractError(f"cannot decode image {path}: {exc}") from exc
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(
                cells[row * 9 + column] > cells[row * 9 + column + 1]
            )
    return f"{bits:016x}"


def _normalize_id_list(value: object, description: str, *, minimum: int = 1) -> list[str]:
    values = _require_list(value, description)
    normalized = [_require_identifier(item, f"{description} member") for item in values]
    if len(normalized) < minimum or len(set(normalized)) != len(normalized):
        raise HoldoutContractError(f"{description} must have >= {minimum} distinct members")
    return sorted(normalized)


def _normalize_human_review(value: object) -> dict[str, Any]:
    review = _require_mapping(value, "human_review")
    required_keys = {
        "annotation_author_ids", "reviewer_ids", "adjudicator_ids",
        "completed_at_utc", "protocol_path", "protocol_sha256",
    }
    if not required_keys.issubset(review) or set(review) - (
        required_keys | {"protocol_packaged_path", "protocol_bytes"}
    ):
        raise HoldoutContractError("human_review fields differ from the schema")
    annotators = _normalize_id_list(review.get("annotation_author_ids"), "annotation_author_ids")
    reviewers = _normalize_id_list(review.get("reviewer_ids"), "reviewer_ids", minimum=2)
    adjudicators = _normalize_id_list(review.get("adjudicator_ids"), "adjudicator_ids")
    if set(reviewers).intersection(adjudicators):
        raise HoldoutContractError("adjudicators must be independent of the two reviewers")
    return {
        "annotation_author_ids": annotators,
        "reviewer_ids": reviewers,
        "adjudicator_ids": adjudicators,
        "completed_at_utc": _require_utc(review.get("completed_at_utc"), "review completion time"),
        "protocol_path": _safe_relative(review.get("protocol_path"), "review protocol path"),
        "protocol_sha256": _require_sha256(review.get("protocol_sha256"), "review protocol hash"),
    }


def _normalize_review_decision(
    value: object,
    description: str,
    allowed_reviewers: set[str],
) -> dict[str, str]:
    record = _require_mapping(value, description)
    reviewer = _require_identifier(record.get("reviewer_id"), f"{description} reviewer")
    if reviewer not in allowed_reviewers:
        raise HoldoutContractError(f"{description} reviewer is absent from review provenance")
    decision = record.get("decision")
    if decision not in {"negative", "player_present"}:
        raise HoldoutContractError(f"{description} decision must be negative or player_present")
    return {
        "reviewer_id": reviewer,
        "decision": decision,
        "reviewed_at_utc": _require_utc(record.get("reviewed_at_utc"), f"{description} time"),
    }


def _normalize_negative_review(value: object, human_review: Mapping[str, Any]) -> dict[str, Any]:
    review = _require_mapping(value, "negative image review")
    allowed_reviewers = set(human_review["reviewer_ids"])
    first = _normalize_review_decision(review.get("reviewer_1"), "first negative review", allowed_reviewers)
    second = _normalize_review_decision(review.get("reviewer_2"), "second negative review", allowed_reviewers)
    if first["reviewer_id"] == second["reviewer_id"]:
        raise HoldoutContractError("negative review requires two distinct reviewers")
    adjudication = _require_mapping(review.get("adjudication"), "negative adjudication")
    if first["decision"] == second["decision"]:
        if first["decision"] != "negative":
            raise HoldoutContractError("zero-box image was rejected by both negative reviewers")
        if adjudication.get("status") != "not_required" or set(adjudication) != {"status"}:
            raise HoldoutContractError("agreeing negative reviews require exact not_required adjudication")
        normalized_adjudication: dict[str, Any] = {"status": "not_required"}
    else:
        if adjudication.get("status") != "resolved":
            raise HoldoutContractError("disagreeing negative reviews require resolved adjudication")
        adjudicator = _require_identifier(adjudication.get("adjudicator_id"), "adjudicator")
        if adjudicator not in set(human_review["adjudicator_ids"]):
            raise HoldoutContractError("negative adjudicator is absent from review provenance")
        if adjudicator in {first["reviewer_id"], second["reviewer_id"]}:
            raise HoldoutContractError("negative adjudicator must be independent")
        if adjudication.get("decision") != "negative":
            raise HoldoutContractError("zero-box image adjudication must resolve to negative")
        normalized_adjudication = {
            "status": "resolved",
            "adjudicator_id": adjudicator,
            "decision": "negative",
            "reviewed_at_utc": _require_utc(
                adjudication.get("reviewed_at_utc"), "adjudication time"
            ),
        }
    return {
        "reviewer_1": first,
        "reviewer_2": second,
        "adjudication": normalized_adjudication,
        "final_status": "reviewed_negative",
    }


def _number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HoldoutContractError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HoldoutContractError(f"{description} must be finite")
    return result


def _analyze_coco(
    coco: object,
    image_root: Path,
    session_ids: set[str],
    human_review: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    document = _require_mapping(coco, "COCO annotations")
    categories = _require_list(document.get("categories"), "COCO categories")
    if len(categories) != 1:
        raise HoldoutContractError("COCO must contain exactly one player category")
    category = _require_mapping(categories[0], "COCO player category")
    category_id = _require_int(category.get("id"), "player category id")
    if category.get("name") != "player":
        raise HoldoutContractError("the only class must be named exactly 'player'")

    raw_images = _require_list(document.get("images"), "COCO images")
    if not raw_images:
        raise HoldoutContractError("COCO image inventory is empty")
    images: dict[int, dict[str, Any]] = {}
    frames: set[tuple[str, int]] = set()
    for position, value in enumerate(raw_images):
        image = _require_mapping(value, f"COCO image {position}")
        image_id = _require_int(image.get("id"), f"COCO image {position} id")
        if image_id in images:
            raise HoldoutContractError(f"duplicate COCO image id: {image_id}")
        file_name = _safe_relative(image.get("file_name"), f"COCO image {image_id} file_name")
        suffix = Path(file_name).suffix.casefold()
        if suffix not in IMAGE_SUFFIXES:
            raise HoldoutContractError(f"unsupported image suffix for {file_name}")
        width = _require_int(image.get("width"), f"COCO image {image_id} width", minimum=1)
        height = _require_int(image.get("height"), f"COCO image {image_id} height", minimum=1)
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION or width * height > MAX_IMAGE_PIXELS:
            raise HoldoutContractError(f"unsafe COCO dimensions for image {image_id}")
        session_id = _require_identifier(image.get("session_id"), f"COCO image {image_id} session")
        if session_id not in session_ids:
            raise HoldoutContractError(f"COCO image {image_id} uses an undeclared capture session")
        frame_index = _require_int(
            image.get("source_frame_index"), f"COCO image {image_id} source frame index"
        )
        frame = (session_id, frame_index)
        if frame in frames:
            raise HoldoutContractError(f"duplicate source frame in COCO inventory: {frame}")
        frames.add(frame)
        expected_sha = _require_sha256(image.get("sha256"), f"COCO image {image_id} hash")
        source = _safe_member(image_root, file_name, f"COCO image {image_id}")
        fingerprint = _image_fingerprint(source, width, height, expected_sha)
        supplied_fingerprint = image.get("perceptual_dhash64")
        if supplied_fingerprint is not None and supplied_fingerprint != fingerprint:
            raise HoldoutContractError(f"COCO image {image_id} perceptual hash mismatch")
        images[image_id] = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "session_id": session_id,
            "source_frame_index": frame_index,
            "sha256": expected_sha,
            "perceptual_dhash64": fingerprint,
            "negative_review": image.get("negative_review"),
        }

    raw_annotations = _require_list(document.get("annotations"), "COCO annotations array")
    annotation_ids: set[int] = set()
    unique_boxes: set[tuple[int, int, float, float, float, float]] = set()
    annotation_count_by_image = {image_id: 0 for image_id in images}
    normalized_annotations: list[dict[str, Any]] = []
    buckets = {"target_le_32": 0, "target_33_64": 0, "target_65_96": 0, "target_gt_96": 0}
    forbidden_keys = {"enemy", "is_enemy", "ally", "team", "team_id", "identity", "gamertag", "player_name"}
    for position, value in enumerate(raw_annotations):
        annotation = _require_mapping(value, f"COCO annotation {position}")
        if forbidden_keys.intersection(str(key).casefold() for key in annotation):
            raise HoldoutContractError("identity/team claims are outside the player-detection contract")
        annotation_id = _require_int(annotation.get("id"), f"COCO annotation {position} id")
        if annotation_id in annotation_ids:
            raise HoldoutContractError(f"duplicate COCO annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = _require_int(annotation.get("image_id"), f"annotation {annotation_id} image_id")
        if image_id not in images:
            raise HoldoutContractError(f"annotation {annotation_id} references an unknown image")
        annotation_category_id = _require_int(
            annotation.get("category_id"), f"annotation {annotation_id} category_id"
        )
        if annotation_category_id != category_id:
            raise HoldoutContractError(f"annotation {annotation_id} is not in the player class")
        bbox = _require_list(annotation.get("bbox"), f"annotation {annotation_id} bbox")
        if len(bbox) != 4:
            raise HoldoutContractError(f"annotation {annotation_id} bbox must be [x,y,width,height]")
        x, y, width, height = (
            _number(item, f"annotation {annotation_id} bbox coordinate") for item in bbox
        )
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise HoldoutContractError(f"annotation {annotation_id} has an invalid bbox")
        image = images[image_id]
        if x + width > image["width"] + 1e-6 or y + height > image["height"] + 1e-6:
            raise HoldoutContractError(f"annotation {annotation_id} bbox leaves the source image")
        if annotation.get("iscrowd") != 0 or isinstance(annotation.get("iscrowd"), bool):
            raise HoldoutContractError(f"annotation {annotation_id} must explicitly set iscrowd=0")
        if annotation.get("ignore") != 0 or isinstance(annotation.get("ignore"), bool):
            raise HoldoutContractError(f"annotation {annotation_id} must explicitly set ignore=0")
        for flag in ("occluded", "truncated"):
            if not isinstance(annotation.get(flag), bool):
                raise HoldoutContractError(f"annotation {annotation_id} must set boolean {flag}")
        area = width * height
        if "area" in annotation and not math.isclose(
            _number(annotation["area"], f"annotation {annotation_id} area"), area,
            rel_tol=1e-9, abs_tol=1e-6,
        ):
            raise HoldoutContractError(f"annotation {annotation_id} area differs from bbox")
        projected_height = height * PROJECTED_HEIGHT_REFERENCE_PX / image["height"]
        if projected_height <= 32:
            bucket = "target_le_32"
        elif projected_height <= 64:
            bucket = "target_33_64"
        elif projected_height <= 96:
            bucket = "target_65_96"
        else:
            bucket = "target_gt_96"
        buckets[bucket] += 1
        box_identity = (image_id, category_id, x, y, width, height)
        if box_identity in unique_boxes:
            raise HoldoutContractError(f"annotation {annotation_id} duplicates an existing bbox")
        unique_boxes.add(box_identity)
        annotation_count_by_image[image_id] += 1
        normalized_annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, width, height],
                "area": area,
                "projected_height_px_at_1080p": projected_height,
                "iscrowd": 0,
                "ignore": 0,
                "occluded": annotation["occluded"],
                "truncated": annotation["truncated"],
            }
        )

    normalized_images: list[dict[str, Any]] = []
    negative_count = 0
    for image_id in sorted(images):
        image = images[image_id]
        if annotation_count_by_image[image_id] == 0:
            review = _normalize_negative_review(image["negative_review"], human_review)
            negative_count += 1
        else:
            if image["negative_review"] is not None:
                raise HoldoutContractError(
                    f"positive image {image_id} must not carry a negative review"
                )
            review = None
        normalized = {key: image[key] for key in (
            "id", "file_name", "width", "height", "session_id",
            "source_frame_index", "sha256", "perceptual_dhash64",
        )}
        if review is not None:
            normalized["negative_review"] = review
        normalized_images.append(normalized)
    buckets["reviewed_negatives"] = negative_count
    normalized_coco = {
        "images": normalized_images,
        "annotations": sorted(normalized_annotations, key=lambda item: item["id"]),
        "categories": [{"id": category_id, "name": "player"}],
    }
    return normalized_coco, normalized_images, buckets


def _normalize_session(
    value: object,
    pool: str,
    input_root: Path,
    staging: Path,
) -> dict[str, Any]:
    session = _require_exact_keys(
        value,
        {
            "session_id", "assigned_pool", "captured_at_utc", "source",
            "license", "capture_environment_commit", "acquisition",
        },
        "capture session",
    )
    session_id = _require_identifier(session.get("session_id"), "session_id")
    if session.get("assigned_pool") != pool:
        raise HoldoutContractError(
            f"session {session_id} must be assigned wholly to pool {pool}; frame-level splitting is forbidden"
        )
    source = _require_exact_keys(
        session.get("source"), {"kind", "path", "sha256"}, f"session {session_id} source"
    )
    source_kind = source.get("kind")
    if source_kind not in {"archive", "video", "image"}:
        raise HoldoutContractError(f"session {session_id} source kind must be archive, video, or image")
    license_record = _require_exact_keys(
        session.get("license"),
        {
            "path", "sha256", "identifier", "authorization_basis",
            "holdout_use_permitted", "redistribution_permitted",
        },
        f"session {session_id} license",
    )
    if license_record.get("holdout_use_permitted") is not True:
        raise HoldoutContractError(f"session {session_id} lacks asserted holdout-use permission")
    if not isinstance(license_record.get("redistribution_permitted"), bool):
        raise HoldoutContractError(f"session {session_id} license must state redistribution_permitted")
    acquisition = _require_exact_keys(
        session.get("acquisition"),
        {
            "tool_name", "tool_path", "tool_sha256", "config_path",
            "config_sha256", "operator_id",
        },
        f"session {session_id} acquisition",
    )
    target_root = staging / "provenance" / "sessions" / session_id

    artifacts: dict[str, dict[str, Any]] = {}
    artifact_specs = (
        ("source", source, "path", "sha256", "source.bin"),
        ("license", license_record, "path", "sha256", "license.txt"),
        ("acquisition_tool", acquisition, "tool_path", "tool_sha256", "acquisition-tool.bin"),
        ("acquisition_config", acquisition, "config_path", "config_sha256", "acquisition-config.bin"),
    )
    for name, record, path_key, sha_key, output_name in artifact_specs:
        relative = _safe_relative(record.get(path_key), f"session {session_id} {name} path")
        expected = _require_sha256(record.get(sha_key), f"session {session_id} {name} hash")
        source_path = _safe_member(input_root, relative, f"session {session_id} {name}")
        destination = target_root / output_name
        copied = _copy_verified(source_path, destination, expected, f"session {session_id} {name}")
        artifacts[name] = {
            "original_path": relative,
            "packaged_path": destination.relative_to(staging).as_posix(),
            **copied,
        }
    try:
        license_text = (target_root / "license.txt").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise HoldoutContractError(f"session {session_id} license evidence is not UTF-8 text") from exc
    if not license_text.strip():
        raise HoldoutContractError(f"session {session_id} license evidence is empty")
    return {
        "session_id": session_id,
        "assigned_pool": pool,
        "captured_at_utc": _require_utc(session.get("captured_at_utc"), f"session {session_id} capture time"),
        "source_kind": source_kind,
        "source": artifacts["source"],
        "license": {
            **artifacts["license"],
            "identifier": _require_string(license_record.get("identifier"), f"session {session_id} license identifier"),
            "authorization_basis": _require_string(
                license_record.get("authorization_basis"), f"session {session_id} authorization basis"
            ),
            "holdout_use_permitted": True,
            "redistribution_permitted": license_record["redistribution_permitted"],
        },
        "capture_environment_commit": _require_commit(
            session.get("capture_environment_commit"), f"session {session_id} capture commit"
        ),
        "acquisition": {
            "tool_name": _require_string(acquisition.get("tool_name"), f"session {session_id} tool name"),
            "operator_id": _require_identifier(acquisition.get("operator_id"), f"session {session_id} operator"),
            "tool": artifacts["acquisition_tool"],
            "config": artifacts["acquisition_config"],
        },
    }


class _BKNode:
    __slots__ = ("value", "records", "children")

    def __init__(self, value: int, record: tuple[str, str, str]) -> None:
        self.value = value
        self.records = [record]
        self.children: dict[int, _BKNode] = {}

    def add(self, value: int, record: tuple[str, str, str]) -> None:
        distance = (self.value ^ value).bit_count()
        if distance == 0:
            self.records.append(record)
            return
        child = self.children.get(distance)
        if child is None:
            self.children[distance] = _BKNode(value, record)
        else:
            child.add(value, record)

    def query(self, value: int, threshold: int) -> Iterable[tuple[int, tuple[str, str, str]]]:
        distance = (self.value ^ value).bit_count()
        if distance <= threshold:
            for record in self.records:
                yield distance, record
        lower, upper = distance - threshold, distance + threshold
        for edge, child in self.children.items():
            if lower <= edge <= upper:
                yield from child.query(value, threshold)


def _load_package_manifest(path: Path, description: str = "holdout manifest") -> tuple[dict[str, Any], Path, bytes]:
    candidate = _absolute(path)
    if candidate.is_dir() and not candidate.is_symlink():
        candidate /= MANIFEST_NAME
    value, raw = _load_json(candidate, description)
    manifest = dict(_require_mapping(value, description))
    if raw != _canonical_json_bytes(manifest):
        raise HoldoutContractError(f"{description} is not canonical JSON")
    expected = manifest.get("manifest_content_sha256")
    body = dict(manifest)
    body.pop("manifest_content_sha256", None)
    if expected != _canonical_hash(body):
        raise HoldoutContractError(f"{description} self-hash mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != DATASET_KIND:
        raise HoldoutContractError(f"unsupported {description} schema")
    expected_keys = {
        "schema_version", "kind", "package_id", "pool", "split_contract", "classes",
        "target_height_definition", "source_input", "sessions", "human_review",
        "annotations", "images", "counts", "source_group_inventory",
        "release_gates", "leakage_check", "redistribution_permitted_for_all_sessions",
        "members", "members_content_sha256", "manifest_content_sha256",
    }
    if set(manifest) != expected_keys:
        raise HoldoutContractError(f"{description} fields differ from the schema contract")
    return manifest, candidate, raw


def _reference_inventory(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    references: list[dict[str, Any]] = []
    records: list[dict[str, str]] = []
    package_ids: set[str] = set()
    for path in paths:
        manifest, _, _ = _load_package_manifest(path, "reference holdout manifest")
        package_id = _require_identifier(manifest.get("package_id"), "reference package_id")
        if package_id in package_ids:
            raise HoldoutContractError(f"duplicate reference package: {package_id}")
        package_ids.add(package_id)
        pool = manifest.get("pool")
        if pool not in POOLS:
            raise HoldoutContractError(f"reference package {package_id} has invalid pool")
        sessions = _require_list(manifest.get("sessions"), f"reference {package_id} sessions")
        session_ids: list[str] = []
        source_sha256s: list[str] = []
        for item in sessions:
            session = _require_mapping(item, "reference session")
            session_ids.append(
                _require_identifier(session.get("session_id"), "reference session_id")
            )
            source_sha256s.append(
                _require_sha256(
                    _require_mapping(session.get("source"), "reference session source").get("sha256"),
                    "reference session source hash",
                )
            )
        images = _require_list(manifest.get("images"), f"reference {package_id} images")
        for image in images:
            entry = _require_mapping(image, "reference image")
            records.append(
                {
                    "package_id": package_id,
                    "pool": pool,
                    "image_id": str(_require_int(entry.get("id"), "reference image id")),
                    "sha256": _require_sha256(entry.get("sha256"), "reference image hash"),
                    "perceptual_dhash64": _require_string(
                        entry.get("perceptual_dhash64"), "reference perceptual hash"
                    ),
                }
            )
            if not re.fullmatch(r"[0-9a-f]{16}", records[-1]["perceptual_dhash64"]):
                raise HoldoutContractError("reference perceptual hash must be lowercase dHash64")
        references.append(
            {
                "package_id": package_id,
                "pool": pool,
                "manifest_content_sha256": manifest["manifest_content_sha256"],
                "session_ids": sorted(session_ids),
                "source_sha256s": sorted(source_sha256s),
                "images": len(images),
            }
        )
    return sorted(references, key=lambda item: item["package_id"]), records


def _check_leakage(
    package_id: str,
    pool: str,
    session_ids: set[str],
    source_sha256s: set[str],
    images: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    reference_images: Sequence[Mapping[str, str]],
    threshold: int,
) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 16:
        raise HoldoutContractError("perceptual Hamming threshold must be an integer from 0 to 16")
    if any(reference["package_id"] == package_id for reference in references):
        raise HoldoutContractError(f"package_id duplicates a reference package: {package_id}")
    reference_sessions: dict[str, str] = {}
    reference_sources: dict[str, str] = {}
    for reference in references:
        for session_id in reference["session_ids"]:
            previous = reference_sessions.setdefault(session_id, reference["package_id"])
            if previous != reference["package_id"]:
                raise HoldoutContractError(
                    f"capture session {session_id} crosses reference packages {previous} and {reference['package_id']}"
                )
        for source_sha in reference["source_sha256s"]:
            previous = reference_sources.setdefault(source_sha, reference["package_id"])
            if previous != reference["package_id"]:
                raise HoldoutContractError(
                    f"raw capture source crosses reference packages {previous} and {reference['package_id']}"
                )
    overlap = sorted(session_ids.intersection(reference_sessions))
    if overlap:
        raise HoldoutContractError(
            f"capture session crosses immutable pools/packages: {overlap[0]}"
        )
    source_overlap = sorted(source_sha256s.intersection(reference_sources))
    if source_overlap:
        raise HoldoutContractError(
            f"raw capture source crosses immutable pools/packages: {source_overlap[0]}"
        )

    exact: dict[str, tuple[str, str, str]] = {}
    tree: _BKNode | None = None
    all_records: list[tuple[str, str, str, str, str]] = []
    for entry in reference_images:
        all_records.append(
            (entry["package_id"], entry["pool"], entry["image_id"], entry["sha256"], entry["perceptual_dhash64"])
        )
    for image in images:
        all_records.append(
            (package_id, pool, str(image["id"]), str(image["sha256"]), str(image["perceptual_dhash64"]))
        )
    for record_package, record_pool, image_id, image_sha, fingerprint in all_records:
        identity = (record_package, record_pool, image_id)
        prior_exact = exact.get(image_sha)
        if prior_exact is not None:
            raise HoldoutContractError(
                f"exact image leakage between {prior_exact[0]}:{prior_exact[2]} and {record_package}:{image_id}"
            )
        exact.setdefault(image_sha, identity)
        value = int(fingerprint, 16)
        if tree is not None:
            for distance, prior in tree.query(value, threshold):
                if prior[0] != record_package:
                    raise HoldoutContractError(
                        f"perceptual image leakage (dHash distance {distance}) between "
                        f"{prior[0]}:{prior[2]} and {record_package}:{image_id}"
                    )
        if tree is None:
            tree = _BKNode(value, identity)
        else:
            tree.add(value, identity)


def _collect_members(staging: Path) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(staging, followlinks=False):
        base = Path(directory)
        for name in list(directory_names):
            child = base / name
            if child.is_symlink():
                raise HoldoutContractError(f"output staging contains a symlink: {child}")
        for name in file_names:
            path = _require_regular_file(base / name, "output member")
            relative = path.relative_to(staging).as_posix()
            members.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            )
    return sorted(members, key=lambda item: item["path"])


def _publish_new_directory(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise HoldoutContractError(f"refusing to overwrite existing output: {output}")
    # Never fall back to a check-then-replacing POSIX rename: another process
    # could create an empty destination in that window and lose it. Linux's
    # renameat2 provides the required atomic no-replace primitive. Windows
    # os.rename is already no-replace when the destination exists.
    if os.name == "posix":
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            parent_fd = os.open(output.parent, parent_flags)
        except OSError as exc:
            raise HoldoutContractError(
                f"cannot anchor output parent for atomic publication: {output.parent}: {exc}"
            ) from exc
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            try:
                renameat2 = libc.renameat2
            except AttributeError as exc:
                raise HoldoutContractError(
                    "this POSIX platform lacks atomic no-replace directory publication"
                ) from exc
            renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                parent_fd,
                os.fsencode(staging.name),
                parent_fd,
                os.fsencode(output.name),
                1,
            )
            if result == 0:
                return
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise HoldoutContractError(f"refusing to overwrite existing output: {output}")
            raise HoldoutContractError(
                f"atomic no-replace publication failed for {output}: "
                f"[{error}] {os.strerror(error)}"
            )
        finally:
            os.close(parent_fd)
    else:
        try:
            os.rename(staging, output)
        except FileExistsError as exc:
            raise HoldoutContractError(f"refusing to overwrite existing output: {output}") from exc


def prepare_holdout(
    input_manifest: Path,
    output_dir: Path,
    *,
    reference_manifests: Sequence[Path] = (),
    minimums: GateMinimums = PINNED_RELEASE_MINIMUMS,
    perceptual_hamming_threshold: int = DEFAULT_PERCEPTUAL_HAMMING_THRESHOLD,
) -> dict[str, Any]:
    """Create a new deterministic package from legally obtained local inputs."""

    minimums = _validate_minimums(minimums)
    manifest_path = _require_regular_file(input_manifest, "input manifest")
    input_root = _require_directory(manifest_path.parent, "input manifest directory")
    value, raw_manifest = _load_json(manifest_path, "input manifest")
    source_manifest = _require_mapping(value, "input manifest")
    _require_exact_keys(
        source_manifest,
        {"schema_version", "package_id", "pool", "sessions", "annotations", "human_review"},
        "input manifest",
    )
    if source_manifest.get("schema_version") != SCHEMA_VERSION:
        raise HoldoutContractError("unsupported input manifest schema_version")
    package_id = _require_identifier(source_manifest.get("package_id"), "package_id")
    pool = source_manifest.get("pool")
    if pool not in POOLS:
        raise HoldoutContractError(f"pool must be one of: {', '.join(sorted(POOLS))}")
    output = _absolute(output_dir)
    parent = _require_directory(output.parent, "output parent")
    if output.exists() or output.is_symlink():
        raise HoldoutContractError(f"refusing to overwrite existing output: {output}")
    references, reference_images = _reference_inventory(reference_manifests)
    if pool == POOL_DEVELOPMENT and any(
        reference["pool"] == POOL_SEALED for reference in references
    ):
        raise HoldoutContractError(
            "development preparation cannot inspect a sealed release-holdout reference"
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    published = False
    try:
        sessions_raw = _require_list(source_manifest.get("sessions"), "capture sessions")
        if not sessions_raw:
            raise HoldoutContractError("at least one capture session is required")
        sessions = [_normalize_session(item, pool, input_root, staging) for item in sessions_raw]
        session_ids = [item["session_id"] for item in sessions]
        if len(session_ids) != len(set(session_ids)):
            raise HoldoutContractError("capture session ids must be unique")
        source_sha256s = [item["source"]["sha256"] for item in sessions]
        if len(source_sha256s) != len(set(source_sha256s)):
            raise HoldoutContractError("one raw capture source cannot masquerade as multiple sessions")
        sessions.sort(key=lambda item: item["session_id"])

        human_review = _normalize_human_review(source_manifest.get("human_review"))
        protocol_source = _safe_member(input_root, human_review["protocol_path"], "review protocol")
        protocol_destination = staging / "provenance" / "review-protocol.txt"
        protocol_member = _copy_verified(
            protocol_source, protocol_destination, human_review["protocol_sha256"], "review protocol"
        )
        try:
            if not protocol_destination.read_text(encoding="utf-8").strip():
                raise HoldoutContractError("review protocol is empty")
        except UnicodeError as exc:
            raise HoldoutContractError("review protocol must be UTF-8 text") from exc
        human_review = {
            **human_review,
            "protocol_packaged_path": protocol_destination.relative_to(staging).as_posix(),
            "protocol_bytes": protocol_member["bytes"],
        }

        annotations_record = _require_exact_keys(
            source_manifest.get("annotations"),
            {"path", "sha256"},
            "annotations record",
        )
        annotations_relative = _safe_relative(annotations_record.get("path"), "annotations path")
        annotations_sha = _require_sha256(annotations_record.get("sha256"), "annotations hash")
        annotations_source = _safe_member(input_root, annotations_relative, "source annotations")
        coco, raw_annotations = _load_json(annotations_source, "source annotations")
        actual_annotations_sha = sha256(raw_annotations).hexdigest()
        if actual_annotations_sha != annotations_sha:
            raise HoldoutContractError(
                "source annotations hash mismatch: "
                f"{actual_annotations_sha} != {annotations_sha}"
            )

        # First copy images to deterministic package names, then validate the
        # normalized COCO against those exact copied bytes.
        coco_mapping = dict(_require_mapping(coco, "COCO annotations"))
        copied_images: list[dict[str, Any]] = []
        for raw_image in _require_list(coco_mapping.get("images"), "COCO images"):
            image = dict(_require_mapping(raw_image, "COCO image"))
            image_id = _require_int(image.get("id"), "COCO image id")
            original_name = _safe_relative(image.get("file_name"), f"COCO image {image_id} file_name")
            suffix = Path(original_name).suffix.casefold()
            if suffix not in IMAGE_SUFFIXES:
                raise HoldoutContractError(f"unsupported image suffix for {original_name}")
            expected = _require_sha256(image.get("sha256"), f"COCO image {image_id} hash")
            source = _safe_member(input_root, original_name, f"COCO image {image_id}")
            packaged_relative = f"images/{image_id:012d}{suffix}"
            destination = staging / packaged_relative
            _copy_verified(source, destination, expected, f"COCO image {image_id}")
            image["file_name"] = packaged_relative
            copied_images.append(image)
        coco_mapping["images"] = copied_images
        normalized_coco, images, counts = _analyze_coco(
            coco_mapping, staging, set(session_ids), human_review
        )
        used_session_ids = {image["session_id"] for image in images}
        unused_sessions = sorted(set(session_ids) - used_session_ids)
        if unused_sessions:
            raise HoldoutContractError(
                f"declared capture session has no extracted image: {unused_sessions[0]}"
            )

        if not _gate_pass(counts, minimums):
            deficits = [
                f"{key}={counts[key]}<{value}"
                for key, value in minimums.as_dict().items()
                if key in GATING_COUNT_KEYS and counts[key] < value
            ]
            raise HoldoutContractError("holdout does not meet configured minimums: " + ", ".join(deficits))

        _check_leakage(
            package_id, pool, set(session_ids), set(source_sha256s), images, references, reference_images,
            perceptual_hamming_threshold,
        )

        annotations_destination = staging / "annotations" / "instances.json"
        _write_new(annotations_destination, _canonical_json_bytes(normalized_coco))
        source_annotations_destination = staging / "provenance" / "source-annotations.json"
        _copy_verified(
            annotations_source, source_annotations_destination, annotations_sha, "source annotations"
        )
        source_manifest_destination = staging / "provenance" / "input-manifest.json"
        _write_new(source_manifest_destination, raw_manifest)

        members = _collect_members(staging)
        member_by_path = {item["path"]: item for item in members}
        annotations_member = member_by_path["annotations/instances.json"]
        pinned_pass = _gate_pass(counts, PINNED_RELEASE_MINIMUMS)
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": DATASET_KIND,
            "package_id": package_id,
            "pool": pool,
            "split_contract": {
                "unit": "capture_session",
                "frame_level_random_split_allowed": False,
                "session_ids": session_ids,
            },
            "classes": [{"id": normalized_coco["categories"][0]["id"], "name": "player"}],
            "target_height_definition": {
                "name": "projected_bbox_height",
                "reference_height_px": PROJECTED_HEIGHT_REFERENCE_PX,
                "formula": "bbox_height_px * reference_height_px / source_image_height_px",
                "bucket_intervals_px": {
                    "target_le_32": "(0,32]",
                    "target_33_64": "(32,64]",
                    "target_65_96": "(64,96]",
                    "target_gt_96": "(96,infinity)",
                },
            },
            "source_input": {
                "manifest_path": "provenance/input-manifest.json",
                "manifest_sha256": sha256(raw_manifest).hexdigest(),
                "annotations_original_path": annotations_relative,
                "annotations_original_sha256": annotations_sha,
                "annotations_original_packaged_path": "provenance/source-annotations.json",
            },
            "sessions": sessions,
            "human_review": human_review,
            "annotations": {
                "path": "annotations/instances.json",
                "sha256": annotations_member["sha256"],
                "images": len(images),
                "boxes": len(normalized_coco["annotations"]),
            },
            "images": images,
            "counts": counts,
            "source_group_inventory": _source_group_inventory(normalized_coco),
            "release_gates": {
                "pinned_minimums": PINNED_RELEASE_MINIMUMS.as_dict(),
                "configured_minimums": minimums.as_dict(),
                "gating_count_keys": list(GATING_COUNT_KEYS),
                "configured_gates_pass": True,
                "meets_pinned_release_gates": pinned_pass,
                "target_le_32_is_descriptive": True,
                "configured_descriptive_inventory_target_met": (
                    counts["target_le_32"] >= minimums.target_le_32
                ),
                "pinned_descriptive_inventory_target_met": (
                    counts["target_le_32"] >= PINNED_RELEASE_MINIMUMS.target_le_32
                ),
            },
            "leakage_check": {
                "exact_sha256": True,
                "perceptual_algorithm": "dhash64-9x8-blockmean-bt601-integer",
                "perceptual_hamming_threshold": perceptual_hamming_threshold,
                "reference_manifests": references,
                "reference_manifest_members_opened": False,
                "result": "no_cross_package_collision",
            },
            "redistribution_permitted_for_all_sessions": all(
                session["license"]["redistribution_permitted"] for session in sessions
            ),
            "members": members,
            "members_content_sha256": _canonical_hash(members),
        }
        body["manifest_content_sha256"] = _canonical_hash(body)
        _write_new(staging / MANIFEST_NAME, _canonical_json_bytes(body))
        _publish_new_directory(staging, output)
        published = True
        return body
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _ledger_events(
    root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    allow_lock: bool = False,
) -> list[dict[str, Any]]:
    ledger = root / "access-ledger"
    if not ledger.exists():
        return []
    ledger = _require_directory(ledger, "access ledger")
    files: list[Path] = []
    for child in sorted(ledger.iterdir(), key=lambda item: item.name):
        if allow_lock and child.name == ".write.lock":
            continue
        if child.is_symlink() or not child.is_file() or not re.fullmatch(r"\d{8}-[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.json", child.name):
            raise HoldoutContractError(f"unexpected access-ledger member: {child}")
        files.append(child)
    events: list[dict[str, Any]] = []
    previous: str | None = None
    previous_time: datetime | None = None
    consumed = False
    retired = False
    for sequence, path in enumerate(files, start=1):
        value, raw = _load_json(path, "access-ledger event")
        event = dict(_require_mapping(value, "access-ledger event"))
        if raw != _canonical_json_bytes(event):
            raise HoldoutContractError(f"access-ledger event is not canonical: {path}")
        if path.name != f"{sequence:08d}-{event.get('event_id')}.json":
            raise HoldoutContractError(f"access-ledger sequence/name mismatch: {path}")
        body = dict(event)
        event_hash = body.pop("event_content_sha256", None)
        if event_hash != _canonical_hash(body) or event.get("previous_event_sha256") != previous:
            raise HoldoutContractError(f"access-ledger hash chain mismatch: {path}")
        schema_version = event.get("schema_version")
        if schema_version not in {1, 2} or event.get("sequence") != sequence:
            raise HoldoutContractError(f"access-ledger event schema/sequence mismatch: {path}")
        event_id = _require_string(event.get("event_id"), "ledger event_id")
        if not EVENT_ID_RE.fullmatch(event_id):
            raise HoldoutContractError(f"access-ledger event_id is unsafe: {path}")
        recorded_at = _require_utc(event.get("recorded_at_utc"), "ledger event time")
        recorded_time = _parse_utc(recorded_at)
        if previous_time is not None and recorded_time <= previous_time:
            raise HoldoutContractError(
                "access-ledger timestamps must be strictly increasing"
            )
        _require_identifier(event.get("actor_id"), "ledger actor_id")
        event_manifest_sha = _require_sha256(
            event.get("dataset_manifest_content_sha256"), "ledger dataset manifest hash"
        )
        if expected_manifest_sha256 is not None and event_manifest_sha != expected_manifest_sha256:
            raise HoldoutContractError(f"access-ledger event binds a different package: {path}")
        operation = event.get("operation")
        if operation == "consumed":
            if schema_version != 1:
                raise HoldoutContractError(
                    "access-ledger consumption events must use schema version 1"
                )
            _require_string(event.get("purpose"), "ledger consumption purpose")
            _require_sha256(event.get("evaluation_plan_sha256"), "ledger evaluation plan hash")
            expected_keys = {
                "schema_version", "sequence", "event_id", "operation", "recorded_at_utc",
                "actor_id", "dataset_manifest_content_sha256", "previous_event_sha256",
                "purpose", "evaluation_plan_sha256", "event_content_sha256",
            }
        elif operation == "retired":
            _require_string(event.get("reason"), "ledger retirement reason")
            expected_keys = {
                "schema_version", "sequence", "event_id", "operation", "recorded_at_utc",
                "actor_id", "dataset_manifest_content_sha256", "previous_event_sha256",
                "reason", "event_content_sha256",
            }
            if schema_version == 2:
                _require_sha256(
                    event.get("evaluation_evidence_sha256"),
                    "ledger evaluation evidence hash",
                )
                expected_keys.add("evaluation_evidence_sha256")
        else:
            raise HoldoutContractError(f"unknown access-ledger operation: {path}")
        if set(event) != expected_keys:
            raise HoldoutContractError(f"access-ledger event fields differ from contract: {path}")
        if retired:
            raise HoldoutContractError("access-ledger contains an event after retirement")
        if operation == "consumed":
            if consumed:
                raise HoldoutContractError("access-ledger contains repeated sealed consumption")
            consumed = True
        elif not consumed:
            raise HoldoutContractError("access-ledger retires a holdout before consumption")
        retired = operation == "retired"
        previous = event_hash
        previous_time = recorded_time
        events.append(event)
    return events


def verify_holdout(
    root: Path,
    *,
    access_mode: str = "development",
    _allow_ledger_lock: bool = False,
) -> dict[str, Any]:
    """Verify every packaged byte; development mode refuses sealed member access."""

    package_root = _require_directory(root, "holdout package")
    manifest, manifest_path, _ = _load_package_manifest(package_root)
    pool = manifest.get("pool")
    if pool not in POOLS:
        raise HoldoutContractError("holdout manifest has an invalid pool")
    if access_mode not in {"curator", "development"}:
        raise HoldoutContractError("access_mode must be curator or development")
    if access_mode == "development" and pool == POOL_SEALED:
        raise HoldoutContractError(
            "sealed release holdout cannot be opened or used in development mode"
        )
    _require_identifier(manifest.get("package_id"), "holdout package_id")

    members = _require_list(manifest.get("members"), "holdout members")
    if manifest.get("members_content_sha256") != _canonical_hash(members):
        raise HoldoutContractError("holdout member inventory hash mismatch")
    expected_paths: set[str] = set()
    member_records: dict[str, Mapping[str, Any]] = {}
    for value in members:
        member = _require_exact_keys(
            value, {"path", "bytes", "sha256"}, "holdout member"
        )
        relative = _safe_relative(member.get("path"), "holdout member path")
        if relative in expected_paths:
            raise HoldoutContractError(f"duplicate holdout member path: {relative}")
        expected_paths.add(relative)
        member_records[relative] = member
        path = _safe_member(package_root, relative, "holdout member")
        expected_size = _require_int(member.get("bytes"), f"member {relative} size")
        expected_sha = _require_sha256(member.get("sha256"), f"member {relative} hash")
        if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha:
            raise HoldoutContractError(f"holdout member tampered: {relative}")

    actual_paths: set[str] = set()
    actual_directories: set[str] = set()
    for directory, directory_names, file_names in os.walk(package_root, followlinks=False):
        base = Path(directory)
        for name in list(directory_names):
            child = base / name
            if child.is_symlink():
                raise HoldoutContractError(f"holdout contains a symlink: {child}")
            relative_dir = child.relative_to(package_root).as_posix()
            if relative_dir == "access-ledger":
                directory_names.remove(name)
            else:
                actual_directories.add(relative_dir)
        for name in file_names:
            path = _require_regular_file(base / name, "holdout file")
            relative = path.relative_to(package_root).as_posix()
            if relative == MANIFEST_NAME:
                continue
            actual_paths.add(relative)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise HoldoutContractError(f"holdout member inventory differs; missing={missing[:1]}, extra={extra[:1]}")
    expected_directories: set[str] = set()
    for relative in expected_paths:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_directories != expected_directories:
        extra = sorted(actual_directories - expected_directories)
        missing = sorted(expected_directories - actual_directories)
        raise HoldoutContractError(
            f"holdout directory inventory differs; missing={missing[:1]}, extra={extra[:1]}"
        )

    sessions = _require_list(manifest.get("sessions"), "holdout sessions")
    session_ids = {
        _require_identifier(_require_mapping(item, "holdout session").get("session_id"), "session_id")
        for item in sessions
    }
    if len(session_ids) != len(sessions):
        raise HoldoutContractError("holdout session ids are not unique")
    if [
        _require_mapping(item, "holdout session").get("session_id") for item in sessions
    ] != sorted(session_ids):
        raise HoldoutContractError("holdout sessions are not in canonical order")
    packaged_source_hashes: set[str] = set()
    for value in sessions:
        session = _require_exact_keys(
            value,
            {
                "session_id", "assigned_pool", "captured_at_utc", "source_kind",
                "source", "license", "capture_environment_commit", "acquisition",
            },
            "holdout session",
        )
        session_id = _require_identifier(session.get("session_id"), "session_id")
        if session.get("assigned_pool") != pool:
            raise HoldoutContractError(f"session {session_id} is assigned to a different pool")
        _require_utc(session.get("captured_at_utc"), f"session {session_id} capture time")
        if session.get("source_kind") not in {"archive", "video", "image"}:
            raise HoldoutContractError(f"session {session_id} has invalid source kind")
        _require_commit(session.get("capture_environment_commit"), f"session {session_id} capture commit")
        source = _require_exact_keys(
            session.get("source"),
            {"original_path", "packaged_path", "bytes", "sha256"},
            f"session {session_id} source",
        )
        source_sha = _require_sha256(source.get("sha256"), f"session {session_id} source hash")
        if source_sha in packaged_source_hashes:
            raise HoldoutContractError("one raw capture source appears in multiple sessions")
        packaged_source_hashes.add(source_sha)
        license_record = _require_exact_keys(
            session.get("license"),
            {
                "original_path", "packaged_path", "bytes", "sha256", "identifier",
                "authorization_basis", "holdout_use_permitted", "redistribution_permitted",
            },
            f"session {session_id} license",
        )
        acquisition = _require_exact_keys(
            session.get("acquisition"),
            {"tool_name", "operator_id", "tool", "config"},
            f"session {session_id} acquisition",
        )
        _require_string(license_record.get("identifier"), f"session {session_id} license identifier")
        _require_string(license_record.get("authorization_basis"), f"session {session_id} authorization basis")
        if license_record.get("holdout_use_permitted") is not True or not isinstance(
            license_record.get("redistribution_permitted"), bool
        ):
            raise HoldoutContractError(f"session {session_id} license assertions are invalid")
        _require_string(acquisition.get("tool_name"), f"session {session_id} tool name")
        _require_identifier(acquisition.get("operator_id"), f"session {session_id} operator")
        for description, artifact in (
            ("source", source),
            ("license", license_record),
            (
                "acquisition tool",
                _require_exact_keys(
                    acquisition.get("tool"),
                    {"original_path", "packaged_path", "bytes", "sha256"},
                    "acquisition tool",
                ),
            ),
            (
                "acquisition config",
                _require_exact_keys(
                    acquisition.get("config"),
                    {"original_path", "packaged_path", "bytes", "sha256"},
                    "acquisition config",
                ),
            ),
        ):
            packaged_path = _safe_relative(
                artifact.get("packaged_path"), f"session {session_id} {description} packaged path"
            )
            _safe_relative(artifact.get("original_path"), f"session {session_id} {description} original path")
            inventory = member_records.get(packaged_path)
            if inventory is None:
                raise HoldoutContractError(f"session {session_id} {description} is absent from members")
            if artifact.get("sha256") != inventory.get("sha256") or artifact.get("bytes") != inventory.get("bytes"):
                raise HoldoutContractError(f"session {session_id} {description} provenance mismatch")
    split = _require_exact_keys(
        manifest.get("split_contract"),
        {"unit", "frame_level_random_split_allowed", "session_ids"},
        "split contract",
    )
    if split.get("unit") != "capture_session" or split.get("frame_level_random_split_allowed") is not False:
        raise HoldoutContractError("holdout does not enforce a session-level split")
    if sorted(session_ids) != split.get("session_ids"):
        raise HoldoutContractError("split session inventory mismatch")

    review = _require_mapping(manifest.get("human_review"), "human review")
    normalized_review = _normalize_human_review(review)
    for key, value in normalized_review.items():
        if review.get(key) != value:
            raise HoldoutContractError(f"human review provenance mismatch: {key}")
    protocol_path = _safe_relative(review.get("protocol_packaged_path"), "packaged review protocol")
    protocol_member = member_records.get(protocol_path)
    if (
        protocol_member is None
        or review.get("protocol_sha256") != protocol_member.get("sha256")
        or review.get("protocol_bytes") != protocol_member.get("bytes")
    ):
        raise HoldoutContractError("review protocol provenance mismatch")
    annotations = _require_exact_keys(
        manifest.get("annotations"),
        {"path", "sha256", "images", "boxes"},
        "annotations record",
    )
    annotations_path = _safe_member(package_root, annotations.get("path"), "packaged annotations")
    coco, raw_coco = _load_json(annotations_path, "packaged annotations")
    if raw_coco != _canonical_json_bytes(coco):
        raise HoldoutContractError("packaged annotations are not canonical JSON")
    normalized_coco, images, counts = _analyze_coco(
        coco, package_root, session_ids, review
    )
    if normalized_coco != coco:
        raise HoldoutContractError("packaged COCO annotations are not normalized")
    if images != manifest.get("images") or counts != manifest.get("counts"):
        raise HoldoutContractError("holdout image/count summary mismatch")
    if manifest.get("source_group_inventory") != _source_group_inventory(
        normalized_coco
    ):
        raise HoldoutContractError("holdout source-group inventory mismatch")
    annotation_member = member_records.get(str(annotations.get("path")))
    if (
        annotation_member is None
        or annotations.get("sha256") != annotation_member.get("sha256")
        or annotations.get("images") != len(images)
        or annotations.get("boxes") != len(normalized_coco["annotations"])
    ):
        raise HoldoutContractError("packaged annotation provenance mismatch")
    if manifest.get("classes") != normalized_coco["categories"]:
        raise HoldoutContractError("holdout class summary mismatch")
    expected_height_definition = {
        "name": "projected_bbox_height",
        "reference_height_px": PROJECTED_HEIGHT_REFERENCE_PX,
        "formula": "bbox_height_px * reference_height_px / source_image_height_px",
        "bucket_intervals_px": {
            "target_le_32": "(0,32]",
            "target_33_64": "(32,64]",
            "target_65_96": "(64,96]",
            "target_gt_96": "(96,infinity)",
        },
    }
    if manifest.get("target_height_definition") != expected_height_definition:
        raise HoldoutContractError("projected target-height definition mismatch")
    source_input = _require_exact_keys(
        manifest.get("source_input"),
        {
            "manifest_path", "manifest_sha256", "annotations_original_path",
            "annotations_original_sha256", "annotations_original_packaged_path",
        },
        "source input provenance",
    )
    for path_key, sha_key in (
        ("manifest_path", "manifest_sha256"),
        ("annotations_original_packaged_path", "annotations_original_sha256"),
    ):
        provenance_path = _safe_relative(source_input.get(path_key), f"source input {path_key}")
        provenance_member = member_records.get(provenance_path)
        if provenance_member is None or source_input.get(sha_key) != provenance_member.get("sha256"):
            raise HoldoutContractError(f"source input provenance mismatch: {path_key}")
    _safe_relative(source_input.get("annotations_original_path"), "original annotations path")
    gates = _require_exact_keys(
        manifest.get("release_gates"),
        {
            "pinned_minimums", "configured_minimums", "gating_count_keys",
            "configured_gates_pass", "meets_pinned_release_gates",
            "target_le_32_is_descriptive", "configured_descriptive_inventory_target_met",
            "pinned_descriptive_inventory_target_met",
        },
        "release gates",
    )
    try:
        pinned = GateMinimums(**dict(_require_mapping(gates.get("pinned_minimums"), "pinned minimums")))
        configured = GateMinimums(**dict(_require_mapping(gates.get("configured_minimums"), "configured minimums")))
    except TypeError as exc:
        raise HoldoutContractError("release-gate minimum keys differ from the contract") from exc
    _validate_minimums(configured)
    if pinned != PINNED_RELEASE_MINIMUMS:
        raise HoldoutContractError("pinned release minimums were altered")
    if gates.get("gating_count_keys") != list(GATING_COUNT_KEYS):
        raise HoldoutContractError("release-gate key inventory mismatch")
    if gates.get("target_le_32_is_descriptive") is not True:
        raise HoldoutContractError("ultra-far bucket must remain descriptive")
    if gates.get("configured_gates_pass") is not True or not _gate_pass(counts, configured):
        raise HoldoutContractError("configured release-gate result is false")
    if gates.get("meets_pinned_release_gates") is not _gate_pass(counts, PINNED_RELEASE_MINIMUMS):
        raise HoldoutContractError("pinned release-gate result mismatch")
    if gates.get("configured_descriptive_inventory_target_met") is not (
        counts["target_le_32"] >= configured.target_le_32
    ):
        raise HoldoutContractError("configured descriptive inventory result mismatch")
    if gates.get("pinned_descriptive_inventory_target_met") is not (
        counts["target_le_32"] >= PINNED_RELEASE_MINIMUMS.target_le_32
    ):
        raise HoldoutContractError("pinned descriptive inventory result mismatch")
    leakage = _require_exact_keys(
        manifest.get("leakage_check"),
        {
            "exact_sha256", "perceptual_algorithm", "perceptual_hamming_threshold",
            "reference_manifests", "reference_manifest_members_opened", "result",
        },
        "leakage check",
    )
    if (
        leakage.get("exact_sha256") is not True
        or leakage.get("perceptual_algorithm") != "dhash64-9x8-blockmean-bt601-integer"
        or leakage.get("result") != "no_cross_package_collision"
        or leakage.get("reference_manifest_members_opened") is not False
    ):
        raise HoldoutContractError("leakage-check provenance mismatch")
    threshold = leakage.get("perceptual_hamming_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 16:
        raise HoldoutContractError("leakage-check threshold is invalid")
    reference_records = _require_list(
        leakage.get("reference_manifests"), "leakage reference manifests"
    )
    reference_ids: list[str] = []
    reference_sessions: set[str] = set()
    reference_sources: set[str] = set()
    for value in reference_records:
        reference = _require_exact_keys(
            value,
            {
                "package_id", "pool", "manifest_content_sha256", "session_ids",
                "source_sha256s", "images",
            },
            "leakage reference manifest",
        )
        reference_id = _require_identifier(reference.get("package_id"), "reference package_id")
        if reference_id == manifest["package_id"] or reference_id in reference_ids:
            raise HoldoutContractError("reference package ids must be unique and external")
        reference_ids.append(reference_id)
        reference_pool = reference.get("pool")
        if reference_pool not in POOLS:
            raise HoldoutContractError("leakage reference has an invalid pool")
        if pool == POOL_DEVELOPMENT and reference_pool == POOL_SEALED:
            raise HoldoutContractError("development package records a sealed-pool reference")
        _require_sha256(reference.get("manifest_content_sha256"), "reference manifest hash")
        _require_int(reference.get("images"), "reference image count", minimum=1)
        listed_sessions = _require_list(reference.get("session_ids"), "reference session ids")
        listed_sources = _require_list(reference.get("source_sha256s"), "reference source hashes")
        normalized_sessions = [
            _require_identifier(item, "reference session id") for item in listed_sessions
        ]
        normalized_sources = [
            _require_sha256(item, "reference source hash") for item in listed_sources
        ]
        if normalized_sessions != sorted(set(normalized_sessions)):
            raise HoldoutContractError("reference session ids are not unique/canonical")
        if normalized_sources != sorted(set(normalized_sources)):
            raise HoldoutContractError("reference source hashes are not unique/canonical")
        if reference_sessions.intersection(normalized_sessions) or session_ids.intersection(
            normalized_sessions
        ):
            raise HoldoutContractError("reference session inventory contains leakage")
        if reference_sources.intersection(normalized_sources) or packaged_source_hashes.intersection(
            normalized_sources
        ):
            raise HoldoutContractError("reference raw-source inventory contains leakage")
        reference_sessions.update(normalized_sessions)
        reference_sources.update(normalized_sources)
    if reference_ids != sorted(reference_ids):
        raise HoldoutContractError("reference manifests are not in canonical order")
    expected_redistribution = all(
        _require_mapping(item, "holdout session")["license"]["redistribution_permitted"]
        for item in sessions
    )
    if manifest.get("redistribution_permitted_for_all_sessions") is not expected_redistribution:
        raise HoldoutContractError("aggregate redistribution assertion mismatch")
    events = _ledger_events(
        package_root,
        expected_manifest_sha256=manifest["manifest_content_sha256"],
        allow_lock=_allow_ledger_lock,
    )
    return {
        "package_id": manifest["package_id"],
        "pool": pool,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "counts": counts,
        "meets_pinned_release_gates": gates["meets_pinned_release_gates"],
        "pinned_descriptive_inventory_target_met": gates[
            "pinned_descriptive_inventory_target_met"
        ],
        "access_events": len(events),
        "retired": bool(events and events[-1].get("operation") == "retired"),
        "manifest_path": str(manifest_path),
    }


def _append_ledger_event(root: Path, body_fields: Mapping[str, Any]) -> Path:
    package_root = _require_directory(root, "holdout package")
    manifest, _, _ = _load_package_manifest(package_root)
    if manifest.get("pool") != POOL_SEALED:
        raise HoldoutContractError("release access ledger is only valid for a sealed release holdout")
    ledger = package_root / "access-ledger"
    try:
        ledger.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _require_directory(ledger, "access ledger")
    _fsync_directory(package_root)
    lock = ledger / ".write.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc:
        raise HoldoutContractError("another process is updating the access ledger") from exc
    try:
        os.fsync(lock_fd)
        _fsync_directory(ledger)
        verified = verify_holdout(
            package_root,
            access_mode="curator",
            _allow_ledger_lock=True,
        )
        if verified["manifest_content_sha256"] != manifest["manifest_content_sha256"]:
            raise HoldoutContractError("holdout manifest changed during access validation")
        events = _ledger_events(
            package_root,
            expected_manifest_sha256=manifest["manifest_content_sha256"],
            allow_lock=True,
        )
        if events and events[-1].get("operation") == "retired":
            raise HoldoutContractError("sealed holdout is retired")
        operation = body_fields["operation"]
        if operation == "consumed" and any(event.get("operation") == "consumed" for event in events):
            raise HoldoutContractError("sealed holdout has already been consumed")
        if operation == "retired" and not any(event.get("operation") == "consumed" for event in events):
            raise HoldoutContractError("sealed holdout cannot be retired before its recorded consumption")
        event_id = _require_string(body_fields.get("event_id"), "event_id")
        if not EVENT_ID_RE.fullmatch(event_id):
            raise HoldoutContractError("event_id contains unsafe characters")
        sequence = len(events) + 1
        previous = events[-1]["event_content_sha256"] if events else None
        recorded_at = _require_utc(body_fields.get("recorded_at_utc"), "event time")
        if events and _parse_utc(recorded_at) <= _parse_utc(events[-1]["recorded_at_utc"]):
            raise HoldoutContractError(
                "new access-ledger timestamp must be later than the previous event"
            )
        evidence_sha256 = body_fields.get("evaluation_evidence_sha256")
        event: dict[str, Any] = {
            "schema_version": 2 if evidence_sha256 is not None else 1,
            "sequence": sequence,
            "event_id": event_id,
            "operation": operation,
            "recorded_at_utc": recorded_at,
            "actor_id": _require_identifier(body_fields.get("actor_id"), "event actor_id"),
            "dataset_manifest_content_sha256": manifest["manifest_content_sha256"],
            "previous_event_sha256": previous,
        }
        if operation == "consumed":
            event["purpose"] = _require_string(body_fields.get("purpose"), "consumption purpose")
            event["evaluation_plan_sha256"] = _require_sha256(
                body_fields.get("evaluation_plan_sha256"), "evaluation plan hash"
            )
            if evidence_sha256 is not None:
                event["evaluation_evidence_sha256"] = _require_sha256(
                    evidence_sha256, "evaluation evidence hash"
                )
        elif operation == "retired":
            event["reason"] = _require_string(body_fields.get("reason"), "retirement reason")
        else:
            raise HoldoutContractError("unknown access-ledger operation")
        event["event_content_sha256"] = _canonical_hash(event)
        path = ledger / f"{sequence:08d}-{event_id}.json"
        _write_new(path, _canonical_json_bytes(event))
        _fsync_directory(ledger)
        return path
    finally:
        os.close(lock_fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(ledger)


def record_sealed_consumption(
    root: Path,
    *,
    event_id: str,
    recorded_at_utc: str,
    actor_id: str,
    purpose: str,
    evaluation_plan_sha256: str,
) -> Path:
    """Record the one permitted, predeclared release-holdout consumption."""

    return _append_ledger_event(
        root,
        {
            "event_id": event_id,
            "operation": "consumed",
            "recorded_at_utc": recorded_at_utc,
            "actor_id": actor_id,
            "purpose": purpose,
            "evaluation_plan_sha256": evaluation_plan_sha256,
        },
    )


def retire_sealed_holdout(
    root: Path,
    *,
    event_id: str,
    recorded_at_utc: str,
    actor_id: str,
    reason: str,
) -> Path:
    """Append an irreversible retirement record after release evaluation."""

    return _append_ledger_event(
        root,
        {
            "event_id": event_id,
            "operation": "retired",
            "recorded_at_utc": recorded_at_utc,
            "actor_id": actor_id,
            "reason": reason,
        },
    )


def complete_sealed_evaluation(
    root: Path,
    *,
    evidence_path: Path,
    publish_evaluation: Callable[[], _EvaluationResult],
    event_id: str,
    actor_id: str,
    purpose: str,
    evaluation_plan_sha256: str,
    retirement_event_id: str,
    retirement_reason: str,
    utc_now: Callable[[], datetime] | None = None,
) -> tuple[_EvaluationResult, dict[str, Any]]:
    """Irreversibly consume before member access, then publish and retire once.

    The exclusive ledger lock is held before any package member is verified and
    remains held throughout the caller's evaluation.  A durable consumption
    event is written before ``verify_holdout`` or the callback can read a sealed
    member.  Any later failure therefore burns the only permitted access and
    preserves the forensic lock.  On success, evidence is durably published
    before a second, evidence-hash-bound retirement event is written.
    """

    package_root = _require_directory(root, "holdout package")
    manifest, _, _ = _load_package_manifest(package_root)
    if manifest.get("pool") != POOL_SEALED:
        raise HoldoutContractError(
            "final evaluation transaction requires a sealed release holdout"
        )

    expected_evidence = _absolute(evidence_path)
    _reject_symlink_components(expected_evidence, "evaluation evidence path")
    if package_root == expected_evidence or package_root in expected_evidence.parents:
        raise HoldoutContractError(
            "evaluation evidence must be published outside the sealed package"
        )
    if os.path.lexists(expected_evidence):
        raise HoldoutContractError(
            f"refusing to overwrite evaluation evidence: {expected_evidence}"
        )

    # Validate every caller-supplied ledger value before opening sealed members.
    event_id = _require_string(event_id, "event_id")
    retirement_event_id = _require_string(retirement_event_id, "retirement event_id")
    if (
        not EVENT_ID_RE.fullmatch(event_id)
        or not EVENT_ID_RE.fullmatch(retirement_event_id)
        or event_id == retirement_event_id
    ):
        raise HoldoutContractError("evaluation ledger event ids are invalid or repeated")
    actor_id = _require_identifier(actor_id, "event actor_id")
    purpose = _require_string(purpose, "consumption purpose")
    evaluation_plan_sha256 = _require_sha256(
        evaluation_plan_sha256, "evaluation plan hash"
    )
    retirement_reason = _require_string(retirement_reason, "retirement reason")
    clock = utc_now or (lambda: datetime.now(timezone.utc))

    def transition_time(description: str) -> str:
        value = clock()
        if not isinstance(value, datetime):
            raise HoldoutContractError(f"{description} clock must return datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise HoldoutContractError(f"{description} clock must return timezone-aware UTC")
        normalized = value.astimezone(timezone.utc)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")

    ledger = package_root / "access-ledger"
    try:
        ledger.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _require_directory(ledger, "access ledger")
    _fsync_directory(package_root)
    lock = ledger / ".write.lock"
    try:
        lock_fd = os.open(
            lock,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise HoldoutContractError(
            "another process is evaluating or updating the sealed holdout"
        ) from exc

    transaction_succeeded = False
    preserve_forensic_lock = False
    try:
        os.fsync(lock_fd)
        _fsync_directory(ledger)
        events = _ledger_events(
            package_root,
            expected_manifest_sha256=manifest["manifest_content_sha256"],
            allow_lock=True,
        )
        if events:
            raise HoldoutContractError(
                "sealed holdout has already been consumed or retired"
            )

        consumed_at_utc = transition_time("consumption event")
        consumed = {
            "schema_version": 1,
            "sequence": 1,
            "event_id": event_id,
            "operation": "consumed",
            "recorded_at_utc": consumed_at_utc,
            "actor_id": actor_id,
            "dataset_manifest_content_sha256": manifest[
                "manifest_content_sha256"
            ],
            "previous_event_sha256": None,
            "purpose": purpose,
            "evaluation_plan_sha256": evaluation_plan_sha256,
        }
        consumed["event_content_sha256"] = _canonical_hash(consumed)
        preserve_forensic_lock = True
        _write_new(
            ledger / f"00000001-{event_id}.json",
            _canonical_json_bytes(consumed),
        )
        _fsync_directory(ledger)
        if _ledger_events(
            package_root,
            expected_manifest_sha256=manifest["manifest_content_sha256"],
            allow_lock=True,
        ) != [consumed]:
            raise HoldoutContractError(
                "durable pre-access consumption event differs from transaction"
            )

        verified = verify_holdout(
            package_root,
            access_mode="curator",
            _allow_ledger_lock=True,
        )
        if verified["manifest_content_sha256"] != manifest["manifest_content_sha256"]:
            raise HoldoutContractError(
                "holdout manifest changed during final-evaluation validation"
            )

        result = publish_evaluation()
        evidence = _require_regular_file(
            expected_evidence, "published evaluation evidence"
        )
        _fsync_regular_file(evidence, "published evaluation evidence")
        _fsync_directory(evidence.parent)
        _fsync_directory(evidence.parent.parent)
        evidence_sha256 = _sha256_file(evidence)
        retired_at_utc = transition_time("retirement event")
        if _parse_utc(retired_at_utc) <= _parse_utc(consumed_at_utc):
            raise HoldoutContractError(
                "internally recorded retirement time did not follow consumption"
            )
        retired = {
            "schema_version": 2,
            "sequence": 2,
            "event_id": retirement_event_id,
            "operation": "retired",
            "recorded_at_utc": retired_at_utc,
            "actor_id": actor_id,
            "dataset_manifest_content_sha256": manifest[
                "manifest_content_sha256"
            ],
            "previous_event_sha256": consumed["event_content_sha256"],
            "reason": retirement_reason,
            "evaluation_evidence_sha256": evidence_sha256,
        }
        retired["event_content_sha256"] = _canonical_hash(retired)
        _write_new(
            ledger / f"00000002-{retirement_event_id}.json",
            _canonical_json_bytes(retired),
        )
        _fsync_directory(ledger)
        written = _ledger_events(
            package_root,
            expected_manifest_sha256=manifest["manifest_content_sha256"],
            allow_lock=True,
        )
        if written != [consumed, retired]:
            raise HoldoutContractError(
                "final-evaluation ledger differs from its predeclared transaction"
            )
        _fsync_directory(ledger)
        transaction_succeeded = True
        return result, {
            "evaluation_evidence_sha256": evidence_sha256,
            "consumption_event_sha256": consumed["event_content_sha256"],
            "retirement_event_sha256": retired["event_content_sha256"],
            "consumed_at_utc": consumed_at_utc,
            "retired_at_utc": retired_at_utc,
        }
    finally:
        os.close(lock_fd)
        if transaction_succeeded or not preserve_forensic_lock:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            _fsync_directory(ledger)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="prepare one immutable session-level pool")
    prepare.add_argument("--input-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--reference-manifest", type=Path, action="append", default=[])
    prepare.add_argument("--perceptual-hamming-threshold", type=int, default=DEFAULT_PERCEPTUAL_HAMMING_THRESHOLD)
    for key, value in PINNED_RELEASE_MINIMUMS.as_dict().items():
        prepare.add_argument(f"--min-{key.replace('_', '-')}", type=int, default=value)

    verify = commands.add_parser("verify", help="verify a package without recording release consumption")
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--mode", choices=("curator", "development"), required=True)

    consume = commands.add_parser("consume", help="record the single predeclared sealed evaluation")
    consume.add_argument("--package", type=Path, required=True)
    consume.add_argument("--event-id", required=True)
    consume.add_argument("--recorded-at-utc", required=True)
    consume.add_argument("--actor-id", required=True)
    consume.add_argument("--purpose", required=True)
    consume.add_argument("--evaluation-plan-sha256", required=True)

    retire = commands.add_parser("retire", help="retire a consumed sealed holdout")
    retire.add_argument("--package", type=Path, required=True)
    retire.add_argument("--event-id", required=True)
    retire.add_argument("--recorded-at-utc", required=True)
    retire.add_argument("--actor-id", required=True)
    retire.add_argument("--reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        minimums = GateMinimums(
            target_le_32=args.min_target_le_32,
            target_33_64=args.min_target_33_64,
            target_65_96=args.min_target_65_96,
            target_gt_96=args.min_target_gt_96,
            reviewed_negatives=args.min_reviewed_negatives,
        )
        result = prepare_holdout(
            args.input_manifest,
            args.output,
            reference_manifests=args.reference_manifest,
            minimums=minimums,
            perceptual_hamming_threshold=args.perceptual_hamming_threshold,
        )
    elif args.command == "verify":
        result = verify_holdout(args.package, access_mode=args.mode)
    elif args.command == "consume":
        path = record_sealed_consumption(
            args.package,
            event_id=args.event_id,
            recorded_at_utc=args.recorded_at_utc,
            actor_id=args.actor_id,
            purpose=args.purpose,
            evaluation_plan_sha256=args.evaluation_plan_sha256,
        )
        result = {"event_path": str(path)}
    else:
        path = retire_sealed_holdout(
            args.package,
            event_id=args.event_id,
            recorded_at_utc=args.recorded_at_utc,
            actor_id=args.actor_id,
            reason=args.reason,
        )
        result = {"event_path": str(path)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
