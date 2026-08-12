"""Quantize an exported OpenVINO detector to INT8 with NNCF post-training PTQ.

Quantization is an export-time step: it needs NNCF and the prepared dataset, so
it lives in the export environment beside ``export_model.py`` rather than in the
lean runtime install.

Calibration images are drawn from the dataset's validation split and are
letterboxed exactly the way the runtime preprocesses frames, so the collected
activation statistics match what the detector actually sees.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = PROJECT_ROOT / "datasets" / "fort_cuh_player" / "images" / "valid"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class QuantizationError(RuntimeError):
    """Raised before any file is written when a request cannot be satisfied."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _letterboxed(image, size: int):
    """Resize preserving aspect ratio onto a centered gray canvas."""

    import cv2
    import numpy as np

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise QuantizationError("calibration image has an empty dimension")
    scale = min(size / height, size / width)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - resized_h) // 2
    left = (size - resized_w) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def calibration_images(directory: Path, limit: int) -> list[Path]:
    if not directory.is_dir():
        raise QuantizationError(f"calibration image directory not found: {directory}")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise QuantizationError(f"no calibration images found in {directory}")
    # Take an even stride so calibration spans the split instead of sampling one
    # contiguous, possibly single-scene, run of frames.
    if len(files) <= limit:
        return files
    stride = len(files) / limit
    return [files[int(index * stride)] for index in range(limit)]


def _tensors(paths: Sequence[Path], size: int) -> Iterator:
    import cv2
    import numpy as np

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        canvas = _letterboxed(image, size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        yield np.expand_dims(tensor, 0)


def quantize(
    model_xml: Path,
    output_dir: Path,
    images_dir: Path,
    samples: int,
    basename: str | None,
) -> Path:
    if not model_xml.is_file():
        raise QuantizationError(f"model XML not found: {model_xml}")
    weights = model_xml.with_suffix(".bin")
    if not weights.is_file():
        raise QuantizationError(f"model weights not found beside XML: {weights}")
    if output_dir.exists():
        raise QuantizationError(
            f"output directory already exists; refusing to overwrite it: {output_dir}"
        )

    import nncf
    import openvino as ov

    core = ov.Core()
    model = core.read_model(model=str(model_xml), weights=str(weights))
    static_shape = model.inputs[0].partial_shape
    if static_shape.is_dynamic:
        raise QuantizationError("only static-shaped exports can be calibrated here")
    size = int(static_shape[2].get_length())
    if int(static_shape[3].get_length()) != size:
        raise QuantizationError("only square inputs are supported")

    paths = calibration_images(images_dir, samples)
    print(f"Calibrating on {len(paths)} images at {size}x{size} from {images_dir}")

    dataset = nncf.Dataset(list(_tensors(paths, size)))
    quantized = nncf.quantize(
        model,
        dataset,
        # Detection heads are numerically sensitive; the transformer preset keeps
        # more of the model in higher precision than the default performance
        # preset and costs little on this size of network.
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=len(paths),
    )

    final_name = basename or f"{model_xml.stem}_int8"
    output_dir.mkdir(parents=True)
    target = output_dir / f"{final_name}.xml"
    ov.save_model(quantized, target)

    source_attribution = model_xml.parent / "ATTRIBUTION.md"
    if source_attribution.is_file():
        shutil.copy2(source_attribution, output_dir / "ATTRIBUTION.md")
        print(f"Copied attribution to {output_dir / 'ATTRIBUTION.md'}")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Exported .xml to quantize.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--samples", type=_positive_int, default=300)
    parser.add_argument("--basename", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        target = quantize(
            args.model.resolve(),
            args.output.resolve(),
            args.images.resolve(),
            args.samples,
            args.basename,
        )
    except QuantizationError as exc:
        print(f"Quantization refused: {exc}", file=sys.stderr)
        return 2
    print(f"INT8 model written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
