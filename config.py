from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence


SourceKind = Literal["device", "file", "screen"]
APPLICATION_ROOT = Path(__file__).resolve().parent
DEFAULT_SELF_ZONE_LEFT = 0.18
DEFAULT_SELF_ZONE_WIDTH = 0.34
DEFAULT_SELF_ZONE_HEIGHT = 0.10


@dataclass(frozen=True, slots=True)
class SourceSpec:
    kind: SourceKind
    value: int | Path | None


@dataclass(frozen=True, slots=True)
class AppConfig:
    source: SourceSpec
    model_path: Path
    labels_path: Path
    device: str
    capture_size: tuple[int, int] | None
    capture_fps: float | None
    inference_size: int
    crop_size: int | None
    confidence: float
    iou_threshold: float
    output_format: Literal["auto", "end2end", "traditional"]
    preview: bool
    draw: bool
    stats_window: int
    screen_monitor: int
    screen_region: tuple[int, int, int, int] | None
    screen_fps: float
    ignore_self: bool = False
    self_zone_left: float = DEFAULT_SELF_ZONE_LEFT
    self_zone_width: float = DEFAULT_SELF_ZONE_WIDTH
    self_zone_height: float = DEFAULT_SELF_ZONE_HEIGHT


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1")
    return parsed


def _positive_unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite, greater than 0, and at most 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _window_size(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def _size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX,]\s*(\d+)\s*", value)
    if match is None:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, for example 1280x720")
    width, height = (int(part) for part in match.groups())
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be greater than zero")
    return width, height


def _region(value: str) -> tuple[int, int, int, int]:
    match = re.fullmatch(
        r"\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*",
        value,
    )
    if match is None:
        raise argparse.ArgumentTypeError("expected X,Y,WIDTH,HEIGHT")
    left, top, width, height = (int(part) for part in match.groups())
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("region width and height must be greater than zero")
    return left, top, width, height


def parse_source(value: str) -> SourceSpec:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered == "screen":
        return SourceSpec("screen", None)
    if lowered.startswith("screen:"):
        monitor_text = lowered.partition(":")[2]
        if not monitor_text.isdigit():
            raise argparse.ArgumentTypeError("screen source must be 'screen' or 'screen:N'")
        return SourceSpec("screen", int(monitor_text))

    path = Path(normalized).expanduser()
    if path.exists():
        return SourceSpec("file", path)
    if re.fullmatch(r"\d+", normalized):
        return SourceSpec("device", int(normalized))
    return SourceSpec("file", path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Low-latency offline game-footage object detection with OpenVINO."
    )
    parser.add_argument(
        "--source",
        type=parse_source,
        default=parse_source("0"),
        help="Camera index, video path, 'screen', or 'screen:N' for a Moonlight monitor.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=APPLICATION_ROOT / "models" / "yolo26n_openvino_model" / "yolo26n.xml",
        help="OpenVINO IR .xml model path.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=APPLICATION_ROOT / "models" / "coco80.txt",
        help="Text file with one class name per line.",
    )
    parser.add_argument("--device", default="CPU", help="OpenVINO device, such as CPU or GPU.")
    parser.add_argument(
        "--capture-size",
        type=_size,
        metavar="WIDTHxHEIGHT",
        help="Requested camera/capture-card resolution.",
    )
    parser.add_argument(
        "--capture-fps",
        type=_positive_float,
        help="Requested camera/capture-card frame rate.",
    )
    parser.add_argument(
        "--inference-size",
        type=_positive_int,
        default=320,
        metavar="N",
        help="Square model input size (default: 320).",
    )
    parser.add_argument(
        "--crop-size",
        type=_positive_int,
        metavar="N",
        help="Centered square crop size in source pixels before inference.",
    )
    parser.add_argument(
        "--confidence",
        type=_unit_float,
        default=0.25,
        help="Minimum detection confidence (default: 0.25).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=_unit_float,
        default=0.45,
        help="Class-aware NMS IoU threshold for traditional YOLO outputs.",
    )
    parser.add_argument(
        "--output-format",
        choices=("auto", "end2end", "traditional"),
        default="auto",
        help=(
            "YOLO output decoder. Use 'traditional' for an Nx6 two-class "
            "YOLOv8/11 export; the default auto mode treats Nx6 as end-to-end."
        ),
    )
    parser.add_argument("--no-preview", action="store_true", help="Disable drawing and preview.")
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Show raw preview without boxes or timing overlay.",
    )
    parser.add_argument(
        "--stats-window",
        type=_window_size,
        default=120,
        metavar="N",
        help="Number of recent processed frames used for moving statistics.",
    )
    parser.add_argument(
        "--screen-monitor",
        type=_non_negative_int,
        default=1,
        metavar="N",
        help="MSS monitor index for --source screen (1 is usually the first monitor).",
    )
    parser.add_argument(
        "--screen-region",
        type=_region,
        metavar="X,Y,WIDTH,HEIGHT",
        help=(
            "Global desktop rectangle containing Moonlight; when set, monitor "
            "selection is ignored."
        ),
    )
    parser.add_argument(
        "--screen-fps",
        type=_positive_float,
        default=60.0,
        help="Maximum desktop capture rate for a screen source (default: 60).",
    )
    parser.add_argument(
        "--ignore-self",
        action="store_true",
        help=(
            "After a 3-frame lock, ignore one persistent player-like detection "
            "whose bottom-center reaches a configurable bottom-anchored avatar "
            "zone and whose box is at least 28%% high and 6%% wide (heuristic)."
        ),
    )
    parser.add_argument(
        "--self-zone-left",
        type=_unit_float,
        default=DEFAULT_SELF_ZONE_LEFT,
        metavar="FRACTION",
        help=(
            "Normalized left edge of the self-avatar zone "
            f"(default: {DEFAULT_SELF_ZONE_LEFT})."
        ),
    )
    parser.add_argument(
        "--self-zone-width",
        type=_positive_unit_float,
        default=DEFAULT_SELF_ZONE_WIDTH,
        metavar="FRACTION",
        help=(
            "Normalized width of the self-avatar zone "
            f"(default: {DEFAULT_SELF_ZONE_WIDTH})."
        ),
    )
    parser.add_argument(
        "--self-zone-height",
        type=_positive_unit_float,
        default=DEFAULT_SELF_ZONE_HEIGHT,
        metavar="FRACTION",
        help=(
            "Normalized height measured upward from the frame bottom "
            f"(default: {DEFAULT_SELF_ZONE_HEIGHT})."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AppConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_zone_left + args.self_zone_width > 1.0:
        parser.error("--self-zone-left plus --self-zone-width must be at most 1")
    source: SourceSpec = args.source
    screen_monitor = (
        int(source.value)
        if source.kind == "screen" and source.value is not None
        else args.screen_monitor
    )
    preview = not args.no_preview
    draw = preview and not args.no_draw
    return AppConfig(
        source=source,
        model_path=args.model.expanduser(),
        labels_path=args.labels.expanduser(),
        device=args.device.strip().upper(),
        capture_size=args.capture_size,
        capture_fps=args.capture_fps,
        inference_size=args.inference_size,
        crop_size=args.crop_size,
        confidence=args.confidence,
        iou_threshold=args.iou_threshold,
        output_format=args.output_format,
        preview=preview,
        draw=draw,
        stats_window=args.stats_window,
        screen_monitor=screen_monitor,
        screen_region=args.screen_region,
        screen_fps=args.screen_fps,
        ignore_self=args.ignore_self,
        self_zone_left=args.self_zone_left,
        self_zone_width=args.self_zone_width,
        self_zone_height=args.self_zone_height,
    )
