#!/usr/bin/env python3
"""Create deterministic source-grouped FORT-Cuh train/validation/test splits.

This is a refinement of the prepared one-class dataset. It keeps every existing
positive annotation unchanged, assigns each conservative source group wholly to
one split, and optionally adds only manually reviewed empty background images.
Images whose only annotations are excluded partial/identity labels are never
treated as negatives because doing so would teach false negatives.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

try:
    from scripts.prepare_fort_cuh import (
        EXPECTED_ARCHIVE_SHA256,
        IMAGE_SUFFIXES,
        MAX_IMAGE_BYTES,
        SPLITS,
        _archive_index,
        _category_names,
        _copy_image_member,
        _images_by_id,
        _load_coco_json,
        _original_basename,
        _read_member,
        _sha256_file,
        _clip_and_normalize_box,
        _source_group_key,
        _validate_expected_hash,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from prepare_fort_cuh import (
        EXPECTED_ARCHIVE_SHA256,
        IMAGE_SUFFIXES,
        MAX_IMAGE_BYTES,
        SPLITS,
        _archive_index,
        _category_names,
        _copy_image_member,
        _images_by_id,
        _load_coco_json,
        _original_basename,
        _read_member,
        _sha256_file,
        _clip_and_normalize_box,
        _source_group_key,
        _validate_expected_hash,
    )

try:
    from scripts.fort_dataset_contract import (
        DatasetContractError,
        GROUPED_DATASET_YAML,
        build_dataset_contract,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from fort_dataset_contract import (
        DatasetContractError,
        GROUPED_DATASET_YAML,
        build_dataset_contract,
    )


ASSIGNMENT_VERSION = 3
DEFAULT_SEED = 0
DEFAULT_RATIOS: Mapping[str, float] = {"train": 0.75, "valid": 0.15, "test": 0.10}
NEGATIVE_REVIEW_SCHEMA_VERSION = 1
VISUAL_GROUPING_VERSION = 1
VISUAL_PHASH_MAX_DISTANCE = 4
VISUAL_THUMBNAIL_MAX_MAE = 8.0
VISUAL_PHASH_SIZE = 32
VISUAL_THUMBNAIL_SIZE = 16
GIANT_GROUP_EVAL_CAPACITY_FACTOR = 1.0
REFERENCE_FRAME_HEIGHT = 1080
FAR_PROJECTED_HEIGHT_MAX = 64
FAR_BALANCE_WEIGHT = 0.5
FORMAT_CHAIN_PATTERN = re.compile(
    r"_(?P<source>png|jpe?g)_(?:jpe?g)(?=\.[^.]+$)", re.IGNORECASE
)
DATED_SCREENSHOT_PATTERNS = (
    re.compile(
        r"^(?P<prefix>.*?screenshot)[-_](?P<date>[0-9]{4}[-_][0-9]{2}[-_][0-9]{2})[-_].+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<prefix>.*?screenshot)[-_](?P<date>[0-9]{8})[-_].+",
        re.IGNORECASE,
    ),
)
GENERIC_FRAME_PATTERN = re.compile(
    r"^(?P<prefix>frame|screenshot)[-_]?[0-9]+(?:[-_].*)?\.[^.]+$", re.IGNORECASE
)
NUMERIC_FRAME_PATTERN = re.compile(r"^[0-9]+(?:_jpg)?\.[^.]+$", re.IGNORECASE)
GROUPED_ATTRIBUTION_NOTE = """

## Group-aware split refinement

This derived copy additionally replaces the supplied split assignments with a
deterministic source-grouped 75/15/10 assignment. Explicit video/capture
sequences, canonicalized original-file variants, and conservatively similar
image clusters belong to only one output split. Sequence and individual-image
groups are balanced separately. The filename and perceptual-image heuristics
cannot prove that every visually related image is separated, so independent
target-clone footage is still required for final accuracy claims.

Images with only excluded partial/identity labels are not treated as
backgrounds. Any zero-annotation background included by this tool must appear
in the archive-pinned manual-review allowlist recorded in manifest.json.
"""


class GroupedPreparationError(ValueError):
    """Raised when group-aware preparation cannot be performed safely."""


@dataclass(frozen=True, slots=True)
class Candidate:
    original_split: str
    archive_name: str
    file_name: str
    source_group: str
    label_lines: tuple[str, ...]
    original_source_group: str | None = None
    source_annotation_ids: tuple[int, ...] = ()

    @property
    def is_negative(self) -> bool:
        return not self.label_lines


@dataclass(frozen=True, slots=True)
class SplitSummary:
    source_groups: int
    images: int
    positive_images: int
    reviewed_negative_images: int
    boxes: int


@dataclass(frozen=True, slots=True)
class VisualSignature:
    """Small deterministic image signature used only for conservative grouping."""

    phash: int
    thumbnail_rgb: bytes


@dataclass(frozen=True, slots=True)
class VisualSimilarityEdge:
    left: str
    right: str
    left_group: str
    right_group: str
    phash_distance: int
    thumbnail_mae: float


@dataclass(frozen=True, slots=True)
class AnnotationExclusion:
    source_split: str
    file_name: str
    annotation_id: int
    reason: str


SOURCE_ANNOTATION_EXCLUSIONS = (
    AnnotationExclusion(
        source_split="train",
        file_name="dd33831e-507c-4d66-b272-23cd04f9852b_jpg.rf.4163a393a2f2e694f50da3f7533c9f4b.jpg",
        annotation_id=3393,
        reason=(
            "manual visual review: malformed 0.38x0.48-native-pixel box on non-player pixels"
        ),
    ),
    AnnotationExclusion(
        source_split="train",
        file_name="80c3684c-e4a0-43cb-ba73-6f6b0caee2c0_jpg.rf.3de191f9a910a03b43d8bd11c092ba22.jpg",
        annotation_id=228,
        reason=(
            "manual visual review: malformed 6.12x1.25-native-pixel box on non-player pixels"
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to FORT-Cuh.v1i.coco.zip.")
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("datasets/fort_cuh_player"),
        help="Existing audited one-class conversion (default: %(default)s).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--negative-review",
        type=Path,
        help=(
            "JSON allowlist of manually verified zero-annotation background images. "
            "Omit to add no negative images."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_RATIOS["valid"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS["test"])
    parser.add_argument("--expected-sha256", default=EXPECTED_ARCHIVE_SHA256)
    return parser


def _canonical_source_group_key(file_name: str) -> str:
    """Return a conservative v2 filename group for known export/session patterns."""

    legacy = _source_group_key(file_name)
    if legacy.startswith("video_sequence:"):
        return legacy

    original = _original_basename(file_name)
    while True:
        revised = FORMAT_CHAIN_PATTERN.sub(lambda match: f"_{match.group('source')}", original)
        if revised == original:
            break
        original = revised

    for pattern in DATED_SCREENSHOT_PATTERNS:
        match = pattern.match(original)
        if match is not None:
            prefix = re.sub(r"[-_]+", "-", match.group("prefix").casefold())
            date = re.sub(r"[-_]", "", match.group("date"))
            return f"capture_session:{prefix}:{date}"

    match = GENERIC_FRAME_PATTERN.match(original)
    if match is not None:
        return f"numbered_sequence:{match.group('prefix').casefold()}"
    if NUMERIC_FRAME_PATTERN.match(original) is not None:
        return "numbered_sequence:numeric-frame"
    return f"original_file:{original.casefold()}"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _visual_signature(image_bytes: bytes, archive_name: str) -> VisualSignature:
    """Decode one image and create a pHash plus color-sensitive thumbnail."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise GroupedPreparationError(
            "visual grouping requires numpy and opencv-python"
        ) from exc
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise GroupedPreparationError(f"cannot decode image for visual grouping: {archive_name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    phash_input = cv2.resize(
        gray,
        (VISUAL_PHASH_SIZE, VISUAL_PHASH_SIZE),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    coefficients = cv2.dct(phash_input)[:8, :8].reshape(-1)[1:]
    median = float(np.median(coefficients))
    phash = 0
    for value in coefficients:
        phash = (phash << 1) | int(float(value) > median)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    thumbnail = cv2.resize(
        rgb,
        (VISUAL_THUMBNAIL_SIZE, VISUAL_THUMBNAIL_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return VisualSignature(phash=phash, thumbnail_rgb=thumbnail.tobytes())


def build_visual_signatures(
    archive: zipfile.ZipFile,
    index: Mapping[str, zipfile.ZipInfo],
    candidates: Sequence[Candidate],
) -> dict[str, VisualSignature]:
    """Build deterministic signatures in archive-name order with bounded reads."""

    signatures: dict[str, VisualSignature] = {}
    for candidate in sorted(candidates, key=lambda item: item.archive_name.casefold()):
        info = index.get(candidate.archive_name)
        if info is None:
            raise GroupedPreparationError(
                f"candidate image missing from archive: {candidate.archive_name}"
            )
        image_bytes = _read_member(archive, info, maximum_bytes=MAX_IMAGE_BYTES)
        signatures[candidate.archive_name] = _visual_signature(
            image_bytes, candidate.archive_name
        )
    return signatures


def _thumbnail_mae(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise GroupedPreparationError("visual signature thumbnail sizes do not match")
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)


def visual_similarity_edges(
    candidates: Sequence[Candidate],
    signatures: Mapping[str, VisualSignature],
) -> tuple[VisualSimilarityEdge, ...]:
    """Return every strict cross-group near-duplicate edge in stable order."""

    ordered = sorted(candidates, key=lambda item: item.archive_name.casefold())
    if len({item.archive_name for item in ordered}) != len(ordered):
        raise GroupedPreparationError("candidate archive names must be unique")
    missing = {item.archive_name for item in ordered}.difference(signatures)
    if missing:
        raise GroupedPreparationError(
            f"visual signature missing for candidate: {sorted(missing)[0]}"
        )
    edges: list[VisualSimilarityEdge] = []
    for index, left in enumerate(ordered):
        left_signature = signatures[left.archive_name]
        for right in ordered[index + 1 :]:
            if left.source_group == right.source_group:
                continue
            right_signature = signatures[right.archive_name]
            distance = (left_signature.phash ^ right_signature.phash).bit_count()
            if distance > VISUAL_PHASH_MAX_DISTANCE:
                continue
            mae = _thumbnail_mae(
                left_signature.thumbnail_rgb, right_signature.thumbnail_rgb
            )
            if mae > VISUAL_THUMBNAIL_MAX_MAE:
                continue
            edges.append(
                VisualSimilarityEdge(
                    left=left.archive_name,
                    right=right.archive_name,
                    left_group=left.source_group,
                    right_group=right.source_group,
                    phash_distance=distance,
                    thumbnail_mae=round(mae, 6),
                )
            )
    return tuple(edges)


class _DisjointSet:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def refine_visual_source_groups(
    candidates: Sequence[Candidate],
    edges: Sequence[VisualSimilarityEdge],
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Transitively union every visual edge and return stable cluster details."""

    from dataclasses import replace

    groups = sorted({candidate.source_group for candidate in candidates})
    disjoint = _DisjointSet(groups)
    for edge in sorted(edges, key=lambda item: (item.left, item.right)):
        if edge.left_group not in disjoint.parent or edge.right_group not in disjoint.parent:
            raise GroupedPreparationError("visual edge references an unknown source group")
        disjoint.union(edge.left_group, edge.right_group)

    members: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        members[disjoint.find(group)].append(group)
    final_keys: dict[str, str] = {}
    clusters: list[dict[str, Any]] = []
    for root, source_groups in sorted(members.items()):
        source_groups = sorted(source_groups)
        if len(source_groups) == 1:
            final_key = source_groups[0]
        else:
            prefixes = {value.partition(":")[0] for value in source_groups}
            for preferred in (
                "video_sequence",
                "capture_session",
                "numbered_sequence",
                "original_file",
            ):
                if preferred in prefixes:
                    kind = preferred
                    break
            else:
                kind = "visual_union"
            digest = sha256("\n".join(source_groups).encode()).hexdigest()[:20]
            final_key = f"{kind}:visual-union-{digest}"
            clusters.append(
                {
                    "final_group": final_key,
                    "source_groups": source_groups,
                }
            )
        for source_group in source_groups:
            final_keys[source_group] = final_key
    refined = [
        replace(candidate, source_group=final_keys[candidate.source_group])
        for candidate in candidates
    ]
    return refined, clusters


def _load_prepared_manifest(prepared: Path, archive_digest: str) -> dict[str, Any]:
    try:
        manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GroupedPreparationError(f"invalid prepared dataset manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise GroupedPreparationError("prepared dataset manifest root must be an object")
    conversion = manifest.get("conversion")
    source = manifest.get("source")
    if not isinstance(conversion, dict) or conversion.get("output_class_names") != ["player"]:
        raise GroupedPreparationError("prepared dataset is not one-class player data")
    if not isinstance(source, dict) or source.get("archive_sha256") != archive_digest:
        raise GroupedPreparationError("prepared dataset and source archive hashes do not match")
    return manifest


def _positive_candidates(prepared: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen_names: set[str] = set()
    for split in SPLITS:
        image_dir = prepared / "images" / split
        label_dir = prepared / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise GroupedPreparationError(f"prepared split directories missing: {split}")
        for image in sorted(image_dir.iterdir(), key=lambda path: path.name.casefold()):
            if not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            label = label_dir / f"{image.stem}.txt"
            try:
                lines = tuple(
                    line.strip()
                    for line in label.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            except (OSError, UnicodeError) as exc:
                raise GroupedPreparationError(f"cannot read prepared label {label}: {exc}") from exc
            if not lines or any(len(line.split()) != 5 or not line.startswith("0 ") for line in lines):
                raise GroupedPreparationError(f"unexpected prepared label content: {label}")
            normalized_name = image.name.casefold()
            if normalized_name in seen_names:
                raise GroupedPreparationError(f"duplicate prepared image name: {image.name}")
            seen_names.add(normalized_name)
            original_source_group = _source_group_key(image.name)
            candidates.append(
                Candidate(
                    original_split=split,
                    archive_name=f"{split}/{image.name}",
                    file_name=image.name,
                    source_group=_canonical_source_group_key(image.name),
                    label_lines=lines,
                    original_source_group=original_source_group,
                )
            )
    if not candidates:
        raise GroupedPreparationError("prepared dataset contains no positive images")
    return candidates


def _source_annotation_records(
    archive: zipfile.ZipFile,
    index: Mapping[str, zipfile.ZipInfo],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Return pinned full-player annotation provenance with effective native size."""

    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for split in SPLITS:
        coco = _load_coco_json(archive, index, split)
        categories = _category_names(coco["categories"], split)
        images = _images_by_id(coco["images"], split)
        for annotation in coco["annotations"]:
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            annotation_id = annotation.get("id")
            if not isinstance(annotation_id, int):
                raise GroupedPreparationError(f"annotation without integer id in {split}")
            if not isinstance(image_id, int) or image_id not in images:
                raise GroupedPreparationError(f"invalid annotation image id in {split}")
            if not isinstance(category_id, int) or category_id not in categories:
                raise GroupedPreparationError(f"invalid annotation category id in {split}")
            category = categories[category_id]
            if category not in {
                "0", "fortnite", "player", "bots", "enemy", "hello", "people", "person"
            } or bool(annotation.get("iscrowd", False)):
                continue
            image = images[image_id]
            normalized, clipped = _clip_and_normalize_box(
                annotation.get("bbox"),
                image_width=float(image["width"]),
                image_height=float(image["height"]),
            )
            if normalized is None:
                continue
            width = normalized[2] * float(image["width"])
            height = normalized[3] * float(image["height"])
            key = (split, str(image["file_name"]), annotation_id)
            if key in records:
                raise GroupedPreparationError(f"duplicate source annotation key: {key}")
            records[key] = {
                "source_split": split,
                "file_name": str(image["file_name"]),
                "annotation_id": annotation_id,
                "category": category,
                "coco_bbox": annotation.get("bbox"),
                "image_width": image["width"],
                "image_height": image["height"],
                "effective_native_width": width,
                "effective_native_height": height,
                "effective_native_area": width * height,
                "clipped": clipped,
                "label_line": (
                    "0 " + " ".join(f"{value:.10g}" for value in normalized)
                ),
            }
    return records


def _apply_annotation_exclusions(
    candidates: Sequence[Candidate],
    records: Mapping[tuple[str, str, int], Mapping[str, Any]],
    exclusions: Sequence[AnnotationExclusion] = SOURCE_ANNOTATION_EXCLUSIONS,
) -> tuple[list[Candidate], list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove only reviewed malformed boxes and report all native sub-2px candidates."""

    from dataclasses import replace

    reviewed = {
        (item.source_split, item.file_name, item.annotation_id): item
        for item in exclusions
    }
    discovered_tiny = {
        key: record
        for key, record in records.items()
        if float(record["effective_native_width"]) < 2.0
        or float(record["effective_native_height"]) < 2.0
    }
    if set(discovered_tiny) != set(reviewed):
        unexpected = sorted(set(discovered_tiny).symmetric_difference(reviewed))
        raise GroupedPreparationError(
            "native sub-2px annotation audit changed; review required: "
            f"{unexpected[0] if unexpected else 'unknown'}"
        )
    candidates_by_source = {
        (item.original_split, item.file_name): item for item in candidates
    }
    revised: list[Candidate] = []
    exclusion_report: list[dict[str, Any]] = []
    for candidate in candidates:
        image_records = sorted(
            (
                record
                for key, record in records.items()
                if key[:2] == (candidate.original_split, candidate.file_name)
            ),
            key=lambda record: int(record["annotation_id"]),
        )
        retained = [
            record
            for record in image_records
            if (
                candidate.original_split,
                candidate.file_name,
                int(record["annotation_id"]),
            )
            not in reviewed
        ]
        source_lines = tuple(record["label_line"] for record in image_records)
        if source_lines != candidate.label_lines:
            raise GroupedPreparationError(
                f"source annotation provenance does not reproduce prepared label: {candidate.file_name}"
            )
        if not retained:
            continue
        revised.append(
            replace(
                candidate,
                label_lines=tuple(str(record["label_line"]) for record in retained),
                source_annotation_ids=tuple(
                    int(record["annotation_id"]) for record in retained
                ),
            )
        )
    for key, exclusion in sorted(reviewed.items()):
        record = records.get(key)
        if record is None or key[:2] not in candidates_by_source:
            raise GroupedPreparationError(f"reviewed annotation exclusion is missing: {key}")
        exclusion_report.append({**record, "reason": exclusion.reason})
    audit_report = [dict(record) for _key, record in sorted(discovered_tiny.items())]
    return revised, exclusion_report, audit_report


def _load_negative_review(path: Path | None, archive_digest: str) -> tuple[str, ...]:
    if path is None:
        return ()
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GroupedPreparationError(f"invalid negative review file: {exc}") from exc
    if not isinstance(value, dict):
        raise GroupedPreparationError("negative review root must be an object")
    if value.get("schema_version") != NEGATIVE_REVIEW_SCHEMA_VERSION:
        raise GroupedPreparationError("unsupported negative review schema version")
    if value.get("archive_sha256") != archive_digest:
        raise GroupedPreparationError("negative review was made for a different archive")
    reviewer = value.get("reviewer")
    reviewed_at = value.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(reviewed_at, str):
        raise GroupedPreparationError("negative review must identify reviewer and review time")
    raw_images = value.get("verified_empty_images")
    if not isinstance(raw_images, list):
        raise GroupedPreparationError("negative review has no verified_empty_images list")
    images: list[str] = []
    for raw in raw_images:
        if not isinstance(raw, str):
            raise GroupedPreparationError("negative review image paths must be strings")
        pure = PurePosixPath(raw)
        if (
            pure.is_absolute()
            or len(pure.parts) != 2
            or pure.parts[0] not in SPLITS
            or pure.name != pure.parts[1]
            or pure.suffix.casefold() not in IMAGE_SUFFIXES
        ):
            raise GroupedPreparationError(f"unsafe reviewed negative path: {raw!r}")
        images.append(raw)
    if len(images) != len(set(images)):
        raise GroupedPreparationError("negative review contains duplicate image paths")
    return tuple(sorted(images, key=str.casefold))


def _negative_candidates(
    archive: zipfile.ZipFile,
    index: Mapping[str, zipfile.ZipInfo],
    reviewed_paths: Sequence[str],
) -> list[Candidate]:
    requested = set(reviewed_paths)
    discovered: dict[str, str] = {}
    for split in SPLITS:
        coco = _load_coco_json(archive, index, split)
        categories = _category_names(coco["categories"], split)
        images = _images_by_id(coco["images"], split)
        annotated_ids: set[int] = set()
        for annotation in coco["annotations"]:
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            if not isinstance(image_id, int) or image_id not in images:
                raise GroupedPreparationError(f"invalid annotation image id in {split}")
            if not isinstance(category_id, int) or category_id not in categories:
                raise GroupedPreparationError(f"invalid annotation category id in {split}")
            annotated_ids.add(image_id)
        for image_id, image in images.items():
            archive_name = f"{split}/{image['file_name']}"
            if archive_name not in requested:
                continue
            if image_id in annotated_ids:
                raise GroupedPreparationError(
                    f"reviewed negative has source annotations and is ambiguous: {archive_name}"
                )
            if archive_name not in index:
                raise GroupedPreparationError(f"reviewed negative is missing: {archive_name}")
            discovered[archive_name] = str(image["file_name"])
    missing = requested.difference(discovered)
    if missing:
        raise GroupedPreparationError(f"reviewed negative not present in archive: {sorted(missing)[0]}")
    return [
        Candidate(
            original_split=archive_name.partition("/")[0],
            archive_name=archive_name,
            file_name=file_name,
            source_group=_canonical_source_group_key(file_name),
            label_lines=(),
            original_source_group=_source_group_key(file_name),
        )
        for archive_name, file_name in sorted(discovered.items())
    ]


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    if set(ratios) != set(SPLITS) or any(
        isinstance(value, bool) or not isinstance(value, (float, int)) or value <= 0.0
        for value in ratios.values()
    ):
        raise GroupedPreparationError("split ratios must be positive numbers for train/valid/test")
    if abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-9:
        raise GroupedPreparationError("split ratios must sum to 1.0")


def _tie_break(seed: int, group: str, split: str) -> bytes:
    return sha256(f"{ASSIGNMENT_VERSION}:{seed}:{group}:{split}".encode()).digest()


def assign_source_groups(
    candidates: Sequence[Candidate],
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Deterministically assign complete groups, reserving giant groups for train."""

    _validate_ratios(ratios)
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.source_group].append(candidate)
    if len(grouped) < len(SPLITS):
        raise GroupedPreparationError("too few source groups for three non-empty splits")
    assignments: dict[str, str] = {}
    eligibility_by_group: dict[str, set[str]] = {}

    # Balance video/capture sequences separately from individual screenshots.
    # Eligibility is computed against each output split's global image and box
    # capacity. Train is always eligible; val/test are eligible independently.
    # Remaining groups use deterministic best-fit balancing with largest groups
    # placed first.
    def group_values(members: Sequence[Candidate]) -> dict[str, int]:
        projected_heights = [
            float(line.split()[4]) * REFERENCE_FRAME_HEIGHT
            for member in members
            for line in member.label_lines
        ]
        return {
            "images": len(members),
            "boxes": len(projected_heights),
            "groups": 1,
            "far_boxes": sum(
                height <= FAR_PROJECTED_HEIGHT_MAX
                for height in projected_heights
            ),
        }

    values_by_all_group = {
        group: group_values(members) for group, members in grouped.items()
    }
    global_totals = {
        "images": len(candidates),
        "boxes": sum(len(candidate.label_lines) for candidate in candidates),
        "groups": len(grouped),
        "far_boxes": sum(
            values["far_boxes"] for values in values_by_all_group.values()
        ),
    }
    global_eval_capacities = {
        split: {
            metric: global_totals[metric] * float(ratios[split])
            for metric in ("images", "boxes")
        }
        for split in ("valid", "test")
    }
    for group, members in grouped.items():
        values = values_by_all_group[group]
        eligibility_by_group[group] = {"train"}
        for split in ("valid", "test"):
            if all(
                values[metric]
                <= global_eval_capacities[split][metric]
                * GIANT_GROUP_EVAL_CAPACITY_FACTOR
                for metric in ("images", "boxes")
            ):
                eligibility_by_group[group].add(split)

    strata: dict[str, dict[str, list[Candidate]]] = defaultdict(dict)
    for group, members in grouped.items():
        kind = group.partition(":")[0]
        strata[kind][group] = members

    global_targets = {
        split: {
            metric: global_totals[metric] * float(ratios[split])
            for metric in global_totals
        }
        for split in SPLITS
    }
    global_counts = {
        split: {metric: 0 for metric in global_totals} for split in SPLITS
    }
    stratum_order = [kind for kind in sorted(strata) if kind != "original_file"]
    if "original_file" in strata:
        stratum_order.append("original_file")
    for kind in stratum_order:
        groups = strata[kind]
        totals = {
            "images": sum(len(members) for members in groups.values()),
            "boxes": sum(
                len(item.label_lines)
                for members in groups.values()
                for item in members
            ),
            "groups": len(groups),
            "far_boxes": sum(
                values_by_all_group[group]["far_boxes"] for group in groups
            ),
        }
        targets = {
            split: {metric: totals[metric] * float(ratios[split]) for metric in totals}
            for split in SPLITS
        }
        counts = {split: {metric: 0 for metric in totals} for split in SPLITS}
        values_by_group = {
            group: values_by_all_group[group] for group in groups
        }
        train_only_groups = {
            group for group in groups if eligibility_by_group[group] == {"train"}
        }
        for group in sorted(train_only_groups):
            assignments[group] = "train"
            for metric, value in values_by_group[group].items():
                counts["train"][metric] += value
                global_counts["train"][metric] += value
        order = sorted(
            set(groups).difference(train_only_groups),
            key=lambda group: (
                -values_by_group[group]["far_boxes"],
                -values_by_group[group]["images"],
                -values_by_group[group]["boxes"],
                _tie_break(seed, group, "order"),
            ),
        )

        for index, group in enumerate(order):
            values = values_by_group[group]
            remaining = len(order) - index
            empty = [split for split in SPLITS if counts[split]["groups"] == 0]
            eligible = empty if empty and remaining <= len(empty) else list(SPLITS)

            def cost(split: str) -> tuple[float, bytes]:
                normalized_error = 0.0
                for name in SPLITS:
                    for metric, group_value in values.items():
                        simulated = counts[name][metric]
                        if name == split:
                            simulated += group_value
                        delta = (simulated - targets[name][metric]) / max(
                            totals[metric], 1
                        )
                        weight = FAR_BALANCE_WEIGHT if metric == "far_boxes" else 1.0
                        normalized_error += delta * delta * weight
                return normalized_error, _tie_break(seed, group, split)

            eligible = [
                split for split in eligible if split in eligibility_by_group[group]
            ]
            if not eligible:
                eligible = sorted(eligibility_by_group[group])
            if kind == "original_file":
                # Individual screenshots are the most divisible stratum. Place
                # them last to fill the global deficits left by indivisible
                # capture/video groups instead of preserving a locally perfect
                # ratio while the final dataset remains imbalanced.
                def remaining_deficit(split: str) -> tuple[float, bytes]:
                    score = sum(
                        (
                            global_targets[split][metric]
                            - global_counts[split][metric]
                            - values[metric]
                        )
                        / max(global_targets[split][metric], 1.0)
                        * (FAR_BALANCE_WEIGHT if metric == "far_boxes" else 1.0)
                        for metric in global_totals
                    )
                    return score, _tie_break(seed, group, split)

                selected = max(eligible, key=remaining_deficit)
            else:
                selected = min(eligible, key=cost)
            assignments[group] = selected
            for metric, value in values.items():
                counts[selected][metric] += value
                global_counts[selected][metric] += value

    missing_splits = set(SPLITS).difference(assignments.values())
    for missing in sorted(missing_splits):
        candidates_to_move = [
            group
            for group, current in assignments.items()
            if missing in eligibility_by_group[group]
            if sum(value == current for value in assignments.values()) > 1
        ]
        if not candidates_to_move:
            break
        group = min(
            candidates_to_move,
            key=lambda value: (
                len(grouped[value]),
                sum(len(item.label_lines) for item in grouped[value]),
                _tie_break(seed, value, f"ensure-{missing}"),
            ),
        )
        assignments[group] = missing
    if set(assignments.values()) != set(SPLITS):
        raise GroupedPreparationError("group assignment produced an empty output split")
    return assignments


def assignment_balance_report(
    candidates: Sequence[Candidate],
    assignments: Mapping[str, str],
    ratios: Mapping[str, float],
) -> dict[str, Any]:
    """Describe the pinned giant-group policy and resulting per-stratum balance."""

    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.source_group].append(candidate)
    if set(grouped) != set(assignments):
        raise GroupedPreparationError("assignment report source groups do not match")

    strata: dict[str, dict[str, list[Candidate]]] = defaultdict(dict)
    for group, members in grouped.items():
        strata[group.partition(":")[0]][group] = members

    train_only: list[dict[str, Any]] = []
    stratum_reports: dict[str, Any] = {}
    largest_group: dict[str, Any] | None = None
    def projected_heights(members: Sequence[Candidate]) -> list[float]:
        return [
            float(line.split()[4]) * REFERENCE_FRAME_HEIGHT
            for member in members
            for line in member.label_lines
        ]

    global_heights = projected_heights(candidates)
    def height_buckets(heights: Sequence[float]) -> dict[str, int]:
        return {
            "ultra_far_le_32px": sum(height <= 32 for height in heights),
            "far_33_to_64px": sum(32 < height <= 64 for height in heights),
            "medium_65_to_96px": sum(64 < height <= 96 for height in heights),
            "near_gt_96px": sum(height > 96 for height in heights),
        }

    global_totals = {
        "images": len(candidates),
        "boxes": sum(len(candidate.label_lines) for candidate in candidates),
        "far_boxes": sum(
            height <= FAR_PROJECTED_HEIGHT_MAX for height in global_heights
        ),
    }
    eval_capacities = {
        split: {
            metric: global_totals[metric] * float(ratios[split])
            for metric in global_totals
        }
        for split in ("valid", "test")
    }
    for kind, groups in sorted(strata.items()):
        totals = {
            "images": sum(len(members) for members in groups.values()),
            "boxes": sum(
                len(item.label_lines)
                for members in groups.values()
                for item in members
            ),
            "groups": len(groups),
            "far_boxes": sum(
                height <= FAR_PROJECTED_HEIGHT_MAX
                for members in groups.values()
                for height in projected_heights(members)
            ),
        }
        split_reports: dict[str, Any] = {}
        for split in SPLITS:
            selected = {
                group: members
                for group, members in groups.items()
                if assignments[group] == split
            }
            split_reports[split] = {
                "source_groups": len(selected),
                "images": sum(len(members) for members in selected.values()),
                "boxes": sum(
                    len(item.label_lines)
                    for members in selected.values()
                    for item in members
                ),
                "far_boxes": sum(
                    height <= FAR_PROJECTED_HEIGHT_MAX
                    for members in selected.values()
                    for height in projected_heights(members)
                ),
            }
        for group, members in groups.items():
            group_record = {
                "source_group": group,
                "stratum": kind,
                "images": len(members),
                "boxes": sum(len(item.label_lines) for item in members),
                "far_boxes": sum(
                    height <= FAR_PROJECTED_HEIGHT_MAX
                    for height in projected_heights(members)
                ),
                "assigned_split": assignments[group],
            }
            eligible_eval_splits = [
                split
                for split in ("valid", "test")
                if all(
                    group_record[metric]
                    <= eval_capacities[split][metric]
                    * GIANT_GROUP_EVAL_CAPACITY_FACTOR
                    for metric in ("images", "boxes")
                )
            ]
            group_record["eligible_evaluation_splits"] = eligible_eval_splits
            if largest_group is None or (
                group_record["images"], group_record["boxes"], group
            ) > (
                largest_group["images"],
                largest_group["boxes"],
                largest_group["source_group"],
            ):
                largest_group = group_record
            if not eligible_eval_splits:
                if assignments[group] != "train":
                    raise GroupedPreparationError(
                        f"giant source group was assigned outside training: {group}"
                    )
                train_only.append(group_record)
        stratum_reports[kind] = {
            "totals": totals,
            "evaluation_eligible_source_groups": {
                split: sum(
                    all(
                        (len(members) if metric == "images" else sum(
                            len(item.label_lines) for item in members
                        ))
                        <= eval_capacities[split][metric]
                        * GIANT_GROUP_EVAL_CAPACITY_FACTOR
                        for metric in ("images", "boxes")
                    )
                    for members in groups.values()
                )
                for split in ("valid", "test")
            },
            "splits": split_reports,
        }
    return {
        "policy": {
            "version": 1,
            "giant_group_eval_capacity_factor": GIANT_GROUP_EVAL_CAPACITY_FACTOR,
            "giant_group_rule": (
                "Train is always eligible. Validation or test is independently "
                "eligible only when both group images and boxes fit that split's "
                "global target capacity times the pinned factor. A group fitting "
                "neither evaluation split is train-only."
            ),
            "remaining_group_order": (
                "descending far-box count, image count, box count, then seeded SHA-256 tie-break"
            ),
            "remaining_group_selection": (
                "minimum normalized squared error over per-stratum image, box, and group targets"
            ),
            "individual_image_fill": (
                "Assign original_file groups last to the eligible split with the "
                "largest normalized remaining global image, box, and group deficit."
            ),
            "far_bucket": {
                "reference_frame": "1920x1080 (height-normalized boxes)",
                "definition": (
                    "box projected height <=64 pixels at 1080p; equivalent to "
                    "normalized box height <=64/1080"
                ),
                "reference_height_pixels": REFERENCE_FRAME_HEIGHT,
                "maximum_projected_height_pixels": FAR_PROJECTED_HEIGHT_MAX,
                "balance_weight": FAR_BALANCE_WEIGHT,
            },
        },
        "train_only_giant_groups": sorted(
            train_only, key=lambda item: item["source_group"]
        ),
        "train_only_giant_group_count": len(train_only),
        "largest_source_group": largest_group,
        "global_totals": global_totals,
        "evaluation_target_capacity": eval_capacities,
        "far_object_distribution": {
            "definition": (
                "Height-normalized boxes projected onto a 1920x1080 reference frame; "
                "bucket boundaries are inclusive as named."
            ),
            "reference_height_pixels": REFERENCE_FRAME_HEIGHT,
            "total": height_buckets(global_heights),
            "splits": {
                split: height_buckets(
                    projected_heights(
                        [
                            candidate
                            for candidate in candidates
                            if assignments[candidate.source_group] == split
                        ]
                    )
                )
                for split in SPLITS
            },
            "sampling_caveat": (
                "Ultra-far validation/test buckets contain very few boxes; report "
                "exact counts and bootstrap confidence intervals, and do not rank "
                "models on ultra-far point estimates alone."
            ),
        },
        "strata": stratum_reports,
    }


def _safe_output_name(candidate: Candidate, seen: set[str]) -> str:
    name = candidate.file_name
    normalized = name.casefold()
    if normalized not in seen:
        seen.add(normalized)
        return name
    stem = PurePosixPath(name).stem
    suffix = PurePosixPath(name).suffix
    digest = sha256(candidate.archive_name.encode()).hexdigest()[:12]
    revised = f"{stem}.{digest}{suffix}"
    if revised.casefold() in seen:
        raise GroupedPreparationError(f"cannot disambiguate output image name: {name}")
    seen.add(revised.casefold())
    return revised


def _filename_collision_report(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.source_group].append(candidate)
    report: list[dict[str, Any]] = []
    for canonical_group, members in sorted(grouped.items()):
        original_groups = sorted(
            {
                member.original_source_group or member.source_group
                for member in members
            }
        )
        if len(original_groups) <= 1:
            continue
        report.append(
            {
                "canonical_group": canonical_group,
                "original_source_groups": original_groups,
                "files": sorted(member.archive_name for member in members),
            }
        )
    return report


def _signature_digest(signatures: Mapping[str, VisualSignature]) -> str:
    digest = sha256()
    for name, signature in sorted(signatures.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(signature.phash.to_bytes(8, "big"))
        digest.update(signature.thumbnail_rgb)
    return digest.hexdigest()


def _edge_report(edge: VisualSimilarityEdge) -> dict[str, Any]:
    return {
        "left": edge.left,
        "right": edge.right,
        "left_group": edge.left_group,
        "right_group": edge.right_group,
        "phash_distance": edge.phash_distance,
        "thumbnail_mae": edge.thumbnail_mae,
    }


def prepare_grouped_dataset(
    archive_path: Path,
    prepared_path: Path,
    output_path: Path,
    *,
    negative_review: Path | None = None,
    expected_sha256: str | None = EXPECTED_ARCHIVE_SHA256,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
    visual_grouping: bool = True,
) -> dict[str, Any]:
    """Write a new source-grouped dataset without modifying source artifacts."""

    archive_file = archive_path.expanduser().resolve()
    prepared = prepared_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if not archive_file.is_file() or archive_file.suffix.casefold() != ".zip":
        raise GroupedPreparationError(f"source archive not found or not a zip: {archive_file}")
    if output.exists() or output.is_symlink() or os.path.lexists(output):
        raise GroupedPreparationError(f"output already exists; refusing to overwrite: {output}")
    expected = _validate_expected_hash(expected_sha256)
    archive_digest = _sha256_file(archive_file)
    if expected is not None and expected != archive_digest:
        raise GroupedPreparationError(
            f"archive SHA-256 mismatch: expected {expected}, got {archive_digest}"
        )
    prepared_manifest = _load_prepared_manifest(prepared, archive_digest)
    positive = _positive_candidates(prepared)
    reviewed_paths = _load_negative_review(negative_review, archive_digest)

    try:
        with zipfile.ZipFile(archive_file, "r") as archive:
            index = _archive_index(archive)
            negative = _negative_candidates(archive, index, reviewed_paths)
            annotation_records = _source_annotation_records(archive, index)
            exclusions = (
                SOURCE_ANNOTATION_EXCLUSIONS
                if archive_digest == EXPECTED_ARCHIVE_SHA256
                else tuple(
                    AnnotationExclusion(
                        source_split=split,
                        file_name=file_name,
                        annotation_id=annotation_id,
                        reason=(
                            "synthetic/unpinned archive: deterministic effective native "
                            "width-or-height <2px exclusion"
                        ),
                    )
                    for (split, file_name, annotation_id), record in annotation_records.items()
                    if float(record["effective_native_width"]) < 2.0
                    or float(record["effective_native_height"]) < 2.0
                )
            )
            cleaned_positive, annotation_exclusions, native_tiny_audit = (
                _apply_annotation_exclusions(positive, annotation_records, exclusions)
            )
            filename_candidates = cleaned_positive + negative
            filename_collisions = _filename_collision_report(filename_candidates)
            if visual_grouping:
                signatures = build_visual_signatures(
                    archive, index, filename_candidates
                )
                visual_edges = visual_similarity_edges(filename_candidates, signatures)
                candidates, visual_clusters = refine_visual_source_groups(
                    filename_candidates, visual_edges
                )
            else:
                signatures = {}
                visual_edges = ()
                candidates = list(filename_candidates)
                visual_clusters = []
            assignments = assign_source_groups(candidates, ratios=ratios, seed=seed)
            balance_report = assignment_balance_report(candidates, assignments, ratios)
            missing_archive_members = sorted(
                {candidate.archive_name for candidate in candidates}.difference(index)
            )
            if missing_archive_members:
                raise GroupedPreparationError(
                    f"prepared image is missing from archive: {missing_archive_members[0]}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
            try:
                by_split: dict[str, list[Candidate]] = defaultdict(list)
                seen_output_names: set[str] = set()
                output_names: dict[str, str] = {}
                for candidate in candidates:
                    by_split[assignments[candidate.source_group]].append(candidate)
                    output_names[candidate.archive_name] = _safe_output_name(
                        candidate, seen_output_names
                    )
                summaries: dict[str, SplitSummary] = {}
                group_members: dict[str, list[str]] = defaultdict(list)
                for split in SPLITS:
                    image_dir = temporary / "images" / split
                    label_dir = temporary / "labels" / split
                    image_dir.mkdir(parents=True)
                    label_dir.mkdir(parents=True)
                    items = sorted(by_split[split], key=lambda item: item.archive_name.casefold())
                    for item in items:
                        name = output_names[item.archive_name]
                        _copy_image_member(archive, index[item.archive_name], image_dir / name)
                        label = label_dir / f"{PurePosixPath(name).stem}.txt"
                        label.write_text(
                            "" if item.is_negative else "\n".join(item.label_lines) + "\n",
                            encoding="utf-8",
                        )
                        group_members[item.source_group].append(f"{split}/{name}")
                    summaries[split] = SplitSummary(
                        source_groups=len({item.source_group for item in items}),
                        images=len(items),
                        positive_images=sum(not item.is_negative for item in items),
                        reviewed_negative_images=sum(item.is_negative for item in items),
                        boxes=sum(len(item.label_lines) for item in items),
                    )
                overlap = {
                    group: members
                    for group, members in group_members.items()
                    if len({member.partition("/")[0] for member in members}) > 1
                }
                if overlap:
                    raise GroupedPreparationError(
                        f"internal error: {len(overlap)} source groups cross output splits"
                    )
                refined_by_name = {item.archive_name: item for item in candidates}
                cross_split_visual_edges = [
                    edge
                    for edge in visual_edges
                    if assignments[refined_by_name[edge.left].source_group]
                    != assignments[refined_by_name[edge.right].source_group]
                ]
                if cross_split_visual_edges:
                    raise GroupedPreparationError(
                        "internal error: visual similarity edge crosses output splits"
                    )
                files_by_final_group: dict[str, list[str]] = defaultdict(list)
                for item in candidates:
                    files_by_final_group[item.source_group].append(item.archive_name)
                visual_clusters = [
                    {
                        **cluster,
                        "files": sorted(files_by_final_group[cluster["final_group"]]),
                    }
                    for cluster in visual_clusters
                ]
                (temporary / "fort_cuh_grouped.yaml").write_text(
                    GROUPED_DATASET_YAML, encoding="utf-8"
                )
                (temporary / "labels.txt").write_text("player\n", encoding="utf-8")
                attribution = prepared / "ATTRIBUTION.md"
                try:
                    attribution_text = attribution.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    raise GroupedPreparationError(
                        f"cannot read prepared dataset attribution: {exc}"
                    ) from exc
                (temporary / "ATTRIBUTION.md").write_text(
                    attribution_text.rstrip() + GROUPED_ATTRIBUTION_NOTE,
                    encoding="utf-8",
                )
                try:
                    dataset_contract = build_dataset_contract(temporary)
                except DatasetContractError as exc:
                    raise GroupedPreparationError(
                        f"generated dataset failed exact-file validation: {exc}"
                    ) from exc
                for split in SPLITS:
                    contract_split = dataset_contract["splits"][split]
                    if (
                        contract_split["images"] != summaries[split].images
                        or contract_split["boxes"] != summaries[split].boxes
                    ):
                        raise GroupedPreparationError(
                            f"generated dataset contract disagrees with {split} summary"
                        )
                report = {
                    "schema_version": 1,
                    "source": prepared_manifest.get("source"),
                    "source_archive_sha256": archive_digest,
                    "prepared_manifest_sha256": _sha256_file(prepared / "manifest.json"),
                    "runtime_class_labels": ["player"],
                    "assignment": {
                        "version": ASSIGNMENT_VERSION,
                        "seed": seed,
                        "ratios": dict(ratios),
                        "source_grouping_heuristic": {
                            "version": 2,
                            "base": prepared_manifest["leakage"][
                                "source_grouping_heuristic"
                            ],
                            "filename_canonicalization": (
                                "Collapse redundant encoded-extension chains; group dated "
                                "Screenshot names by capture date; group generic frame and "
                                "numbered Screenshot sequences conservatively."
                            ),
                            "format_chain_regex": FORMAT_CHAIN_PATTERN.pattern,
                            "dated_screenshot_regexes": [
                                pattern.pattern for pattern in DATED_SCREENSHOT_PATTERNS
                            ],
                            "generic_frame_regex": GENERIC_FRAME_PATTERN.pattern,
                            "numeric_frame_regex": NUMERIC_FRAME_PATTERN.pattern,
                        },
                    },
                    "assignment_balance": balance_report,
                    "grouping_collision_report": {
                        "filename_canonicalization_collisions": filename_collisions,
                        "filename_canonicalization_collision_count": len(
                            filename_collisions
                        ),
                        "visual_similarity_edges": [
                            _edge_report(edge) for edge in visual_edges
                        ],
                        "visual_similarity_edge_count": len(visual_edges),
                        "visual_union_clusters": visual_clusters,
                        "visual_union_cluster_count": len(visual_clusters),
                    },
                    "visual_grouping": {
                        "enabled": visual_grouping,
                        "version": VISUAL_GROUPING_VERSION,
                        "phash_max_hamming_distance": VISUAL_PHASH_MAX_DISTANCE,
                        "thumbnail_rgb_max_mae": VISUAL_THUMBNAIL_MAX_MAE,
                        "phash_input_size": VISUAL_PHASH_SIZE,
                        "thumbnail_size": VISUAL_THUMBNAIL_SIZE,
                        "signature_sha256": (
                            _signature_digest(signatures) if visual_grouping else None
                        ),
                        "opencv_version": (
                            _package_version("opencv-python") if visual_grouping else None
                        ),
                        "numpy_version": (
                            _package_version("numpy") if visual_grouping else None
                        ),
                    },
                    "reviewed_negative_images": list(reviewed_paths),
                    "annotation_cleanup": {
                        "version": 1,
                        "rule": (
                            "Audit every retained full-player annotation whose clipped "
                            "effective native width or height is <2px; preparation fails "
                            "if that exact set differs from the archive-pinned reviewed list."
                        ),
                        "excluded_annotations": annotation_exclusions,
                        "excluded_annotation_count": len(annotation_exclusions),
                        "native_sub_2px_audit": native_tiny_audit,
                    },
                    "ambiguous_partial_label_images_are_never_negatives": True,
                    "dataset_contract": dataset_contract,
                    "splits": {split: asdict(summaries[split]) for split in SPLITS},
                    "cross_split_source_groups": 0,
                    "cross_split_visual_similarity_edges": len(
                        cross_split_visual_edges
                    ),
                }
                (temporary / "manifest.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                temporary.rename(output)
                return report
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
    except zipfile.BadZipFile as exc:
        raise GroupedPreparationError(f"invalid zip archive: {archive_file}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ratios = {
        "train": args.train_ratio,
        "valid": args.valid_ratio,
        "test": args.test_ratio,
    }
    try:
        report = prepare_grouped_dataset(
            args.archive,
            args.prepared,
            args.output,
            negative_review=args.negative_review,
            expected_sha256=args.expected_sha256,
            ratios=ratios,
            seed=args.seed,
        )
    except (GroupedPreparationError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
