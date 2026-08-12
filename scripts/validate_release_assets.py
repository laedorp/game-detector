#!/usr/bin/env python3
"""Read-only validation for model assets included in a desktop release.

This preflight deliberately uses OpenVINO's ``read_model`` API with each XML
and BIN path supplied explicitly.  It therefore catches truncated, unrelated,
or otherwise unreadable IR pairs before PyInstaller copies them into a bundle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Sequence


EXPECTED_INPUT_SHAPE = (1, 3, 320, 320)
BALANCED_INPUT_SHAPE = (1, 3, 416, 416)
HIGH_END_INPUT_SHAPE = (1, 3, 640, 640)
FORT_SOURCE_URL = "https://universe.roboflow.com/aviles-joseph/fort-cuh-mji4f"

COCO80_LABELS = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


@dataclass(frozen=True)
class ModelAsset:
    """One release model and the class names its output must use."""

    display_name: str
    xml_relative: Path
    bin_relative: Path
    labels_relative: Path
    expected_labels: tuple[str, ...]
    expected_input_shape: tuple[int, int, int, int]
    attribution_relative: Path | None = None
    attribution_markers: tuple[str, ...] = ()


PLAYER_LABELS = ("player",)

# The FORT-Cuh source data is CC BY 4.0 and the base weights are Ultralytics
# AGPL-3.0, so a release must carry both notices next to the derived model.
PLAYER_ATTRIBUTION_MARKERS = (
    "FORT-Cuh",
    "creativecommons.org/licenses/by/4.0",
    "AGPL-3.0",
)


RELEASE_MODELS = (
    ModelAsset(
        display_name="Balanced player detector",
        xml_relative=Path("models/fort_player_416_openvino_model/fort_player_416.xml"),
        bin_relative=Path("models/fort_player_416_openvino_model/fort_player_416.bin"),
        labels_relative=Path("models/fort_player.txt"),
        expected_labels=PLAYER_LABELS,
        expected_input_shape=BALANCED_INPUT_SHAPE,
        attribution_relative=Path("models/fort_player_416_openvino_model/ATTRIBUTION.md"),
        attribution_markers=PLAYER_ATTRIBUTION_MARKERS,
    ),
    ModelAsset(
        display_name="Fast player detector",
        xml_relative=Path("models/fort_player_openvino_model/fort_player.xml"),
        bin_relative=Path("models/fort_player_openvino_model/fort_player.bin"),
        labels_relative=Path("models/fort_player.txt"),
        expected_labels=PLAYER_LABELS,
        expected_input_shape=EXPECTED_INPUT_SHAPE,
        attribution_relative=Path("models/fort_player_openvino_model/ATTRIBUTION.md"),
        attribution_markers=PLAYER_ATTRIBUTION_MARKERS,
    ),
    ModelAsset(
        display_name="COCO detector",
        xml_relative=Path("models/yolo26n_openvino_model/yolo26n.xml"),
        bin_relative=Path("models/yolo26n_openvino_model/yolo26n.bin"),
        labels_relative=Path("models/coco80.txt"),
        expected_labels=COCO80_LABELS,
        expected_input_shape=EXPECTED_INPUT_SHAPE,
    ),
    ModelAsset(
        display_name="Balanced COCO detector",
        xml_relative=Path("models/yolo26n_416_openvino_model/yolo26n_416.xml"),
        bin_relative=Path("models/yolo26n_416_openvino_model/yolo26n_416.bin"),
        labels_relative=Path("models/coco80.txt"),
        expected_labels=COCO80_LABELS,
        expected_input_shape=BALANCED_INPUT_SHAPE,
    ),
    ModelAsset(
        display_name="High-end YOLO11x detector",
        xml_relative=Path("models/yolo11x_openvino_model/yolo11x.xml"),
        bin_relative=Path("models/yolo11x_openvino_model/yolo11x.bin"),
        labels_relative=Path("models/coco80.txt"),
        expected_labels=COCO80_LABELS,
        expected_input_shape=HIGH_END_INPUT_SHAPE,
    ),
)


EXTRA_RELEASE_FILES = (
    Path("models/yolo11x_onnx/yolo11x.onnx"),
)


class ReleaseAssetError(RuntimeError):
    """Raised when one or more files are unsafe to include in a release."""


def _regular_nonempty_file(path: Path, description: str, errors: list[str]) -> bool:
    try:
        if not path.is_file():
            errors.append(f"{description} is missing or is not a file: {path}")
            return False
        size = path.stat().st_size
    except OSError as exc:
        errors.append(f"cannot inspect {description} {path}: {exc}")
        return False
    if size <= 0:
        errors.append(f"{description} is empty: {path}")
        return False
    return True


def _read_text(path: Path, description: str, errors: list[str]) -> str | None:
    if not _regular_nonempty_file(path, description, errors):
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {description} as UTF-8 ({path}): {exc}")
        return None
    if not text.strip():
        errors.append(f"{description} contains only whitespace: {path}")
        return None
    return text


def _validate_labels(root: Path, asset: ModelAsset, errors: list[str]) -> None:
    path = root / asset.labels_relative
    text = _read_text(path, f"{asset.display_name} label file", errors)
    if text is None:
        return
    actual = tuple(text.splitlines())
    if actual == asset.expected_labels:
        return

    if len(actual) != len(asset.expected_labels):
        detail = f"expected {len(asset.expected_labels)} line(s), found {len(actual)}"
    else:
        index = next(
            index
            for index, (found, expected) in enumerate(
                zip(actual, asset.expected_labels), start=1
            )
            if found != expected
        )
        detail = (
            f"line {index} must be {asset.expected_labels[index - 1]!r}, "
            f"found {actual[index - 1]!r}"
        )
    errors.append(f"{asset.display_name} labels are wrong in {path}: {detail}")


def _validate_attribution(root: Path, asset: ModelAsset, errors: list[str]) -> None:
    if asset.attribution_relative is None:
        return
    path = root / asset.attribution_relative
    text = _read_text(path, f"{asset.display_name} attribution", errors)
    if text is None:
        return
    folded = text.casefold()
    missing = [
        marker
        for marker in asset.attribution_markers
        if marker.casefold() not in folded
    ]
    if missing:
        errors.append(
            f"{asset.display_name} attribution {path} is missing required marker(s): "
            + ", ".join(repr(marker) for marker in missing)
        )


def _dimension_value(dimension: Any) -> int | None:
    """Return a static OpenVINO dimension, or ``None`` when it is dynamic."""

    try:
        if bool(dimension.is_dynamic):
            return None
    except AttributeError:
        pass
    try:
        return int(dimension.get_length())
    except AttributeError:
        pass
    except (RuntimeError, TypeError, ValueError):
        return None
    try:
        value = int(dimension)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _port_shape(port: Any) -> tuple[int | None, ...]:
    try:
        return tuple(_dimension_value(dimension) for dimension in port.partial_shape)
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise ValueError(f"cannot inspect partial shape: {exc}") from exc


def _supported_output_layout(
    shape: Sequence[int | None], label_count: int
) -> str | None:
    """Identify an output layout understood by ``detection.postprocess``."""

    if len(shape) != 3 or shape[0] != 1:
        return None
    rows, columns = shape[1], shape[2]
    if (rows is not None and rows <= 0) or (columns is not None and columns <= 0):
        return None
    if columns == 6:
        return "end-to-end [1,N,6]"
    attributes = 4 + label_count
    if columns == attributes:
        return f"traditional [1,N,{attributes}]"
    if rows == attributes:
        return f"traditional [1,{attributes},N]"
    return None


def _load_openvino_core() -> Any:
    try:
        import openvino as ov
    except ImportError as exc:
        raise ReleaseAssetError(
            "OpenVINO is not installed in the build environment. Install "
            "requirements.txt before validating release assets."
        ) from exc
    core_type = getattr(ov, "Core", None)
    if core_type is None:
        try:
            from openvino.runtime import Core as core_type
        except ImportError as exc:
            raise ReleaseAssetError(
                "The installed OpenVINO package does not expose the Runtime Core API."
            ) from exc
    try:
        return core_type()
    except Exception as exc:
        raise ReleaseAssetError(f"OpenVINO Runtime could not initialize: {exc}") from exc


def _all_ir_files_ready(root: Path) -> bool:
    """Check whether loading is possible without producing duplicate errors."""

    for asset in RELEASE_MODELS:
        for relative_path in (asset.xml_relative, asset.bin_relative):
            path = root / relative_path
            try:
                if not path.is_file() or path.stat().st_size <= 0:
                    return False
            except OSError:
                return False
    return True


def _validate_ir(
    root: Path,
    asset: ModelAsset,
    core: Any,
    errors: list[str],
) -> str | None:
    xml_path = root / asset.xml_relative
    bin_path = root / asset.bin_relative
    xml_ok = _regular_nonempty_file(xml_path, f"{asset.display_name} XML", errors)
    bin_ok = _regular_nonempty_file(bin_path, f"{asset.display_name} BIN", errors)
    if xml_path.stem != bin_path.stem:
        errors.append(
            f"{asset.display_name} XML/BIN names do not have the same stem: "
            f"{xml_path.name}, {bin_path.name}"
        )
        return None
    if not (xml_ok and bin_ok):
        return None

    try:
        model = core.read_model(model=str(xml_path), weights=str(bin_path))
    except Exception as exc:
        errors.append(
            f"OpenVINO could not read the matching {asset.display_name} XML/BIN pair "
            f"({xml_path.name}, {bin_path.name}): {exc}"
        )
        return None

    try:
        inputs = tuple(model.inputs)
        outputs = tuple(model.outputs)
    except Exception as exc:
        errors.append(f"cannot inspect {asset.display_name} model ports: {exc}")
        return None

    if len(inputs) != 1:
        errors.append(
            f"{asset.display_name} must have exactly one input, found {len(inputs)}"
        )
    else:
        try:
            input_shape = _port_shape(inputs[0])
        except ValueError as exc:
            errors.append(f"cannot inspect {asset.display_name} input shape: {exc}")
        else:
            if input_shape != asset.expected_input_shape:
                errors.append(
                    f"{asset.display_name} input shape is {input_shape}; expected static "
                    f"NCHW {asset.expected_input_shape}"
                )

    if len(outputs) != 1:
        errors.append(
            f"{asset.display_name} must have exactly one YOLO output, found {len(outputs)}"
        )
        return None
    try:
        output_shape = _port_shape(outputs[0])
    except ValueError as exc:
        errors.append(f"cannot inspect {asset.display_name} output shape: {exc}")
        return None
    layout = _supported_output_layout(output_shape, len(asset.expected_labels))
    if layout is None:
        attributes = 4 + len(asset.expected_labels)
        errors.append(
            f"{asset.display_name} output shape {output_shape} is unsupported; expected "
            f"batch 1 with [1,N,6], [1,N,{attributes}], or [1,{attributes},N]"
        )
        return None
    return (
        f"{asset.display_name}: input {asset.expected_input_shape}, "
        f"output {output_shape} ({layout})"
    )


def validate_release_assets(
    project_root: str | Path,
    *,
    core_factory: Callable[[], Any] | None = None,
) -> tuple[str, ...]:
    """Validate every bundled model without modifying or compiling anything.

    ``core_factory`` exists for isolated tests; production callers use the
    installed OpenVINO Runtime.  All readily discoverable errors are reported
    together so a release operator can fix them in one pass.
    """

    root = Path(project_root).expanduser().resolve()
    errors: list[str] = []
    for asset in RELEASE_MODELS:
        _validate_labels(root, asset, errors)
        _validate_attribution(root, asset, errors)
    for relative_path in EXTRA_RELEASE_FILES:
        _regular_nonempty_file(root / relative_path, f"release file {relative_path}", errors)

    summaries: list[str] = []
    if _all_ir_files_ready(root):
        try:
            core = (core_factory or _load_openvino_core)()
        except ReleaseAssetError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"OpenVINO Runtime could not initialize: {exc}")
        else:
            for asset in RELEASE_MODELS:
                summary = _validate_ir(root, asset, core, errors)
                if summary is not None:
                    summaries.append(summary)
    else:
        # Run the common checks to produce path-specific diagnostics.  A core
        # is intentionally unnecessary when an IR pair is not present yet.
        for asset in RELEASE_MODELS:
            _regular_nonempty_file(
                root / asset.xml_relative, f"{asset.display_name} XML", errors
            )
            _regular_nonempty_file(
                root / asset.bin_relative, f"{asset.display_name} BIN", errors
            )

    if errors:
        unique_errors = tuple(dict.fromkeys(errors))
        raise ReleaseAssetError(
            "Release asset preflight failed:\n  - " + "\n  - ".join(unique_errors)
        )
    return tuple(summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every model, label file, and attribution bundled in a release."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project directory containing models/ (default: repository root).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summaries = validate_release_assets(args.project_root)
    except ReleaseAssetError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("Release asset preflight passed:")
    for summary in summaries:
        print(f"  - {summary}")
    print("  - bundled COCO label files are exact and complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
