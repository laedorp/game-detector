from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from controller_precision.codes import ABS_BRAKE
from aiming.controller import DEFAULT_HEAD_RATIO
from utils.inference_size import (
    InferenceSize,
    parse_inference_size,
    validate_yolo_inference_size,
)


SourceKind = Literal["device", "file", "screen"]
AimOutput = Literal["local", "remote", "makcu"]
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
    # NCHW spatial dimensions in unambiguous tensor order: (height, width).
    inference_size: InferenceSize
    crop_size: int | None
    # Optional second inference over a centered, model-aspect source ROI. The
    # integer is the requested maximum source-pixel width.
    # It uses the exact same model and static input shape as the primary pass.
    detail_crop_size: int | None
    confidence: float
    iou_threshold: float
    output_format: Literal["auto", "end2end", "traditional"]
    preview: bool
    draw: bool
    preview_fps: float
    stats_window: int
    screen_monitor: int
    screen_region: tuple[int, int, int, int] | None
    screen_fps: float
    # OpenVINO drives Intel CPUs, integrated graphics, and NPUs; AMD and NVIDIA
    # GPUs have no OpenVINO plugin and are reached through ONNX Runtime instead.
    backend: str = "openvino"
    # FOURCC requested from a capture card.  High advertised frame rates
    # usually exist only in a specific mode, so the format must be asked
    # for explicitly rather than left at the driver default.
    capture_format: str | None = None
    ignore_self: bool = False
    self_zone_left: float = DEFAULT_SELF_ZONE_LEFT
    self_zone_width: float = DEFAULT_SELF_ZONE_WIDTH
    self_zone_height: float = DEFAULT_SELF_ZONE_HEIGHT
    aim: bool = False
    aim_label: str | None = None
    aim_invert_x: bool = False
    aim_invert_y: bool = False
    aim_head_ratio: float = DEFAULT_HEAD_RATIO
    aim_output: AimOutput = "local"
    aim_host: str | None = None
    aim_port: int = 47621
    aim_pairing_key: str | None = None
    aim_makcu_port: str | None = None
    aim_makcu_button: int = 1
    aim_makcu_strength: float = 0.50
    aim_makcu_max_step: int = 160
    aim_makcu_smoothing_alpha: float = 0.78
    aim_makcu_prediction_lead_seconds: float = 0.03
    aim_makcu_derivative_damping_seconds: float = 0.008
    aim_activate_path: str | None = None
    aim_activate_axis: int = ABS_BRAKE
    aim_activate_threshold: float = 0.35
    # Optional live-pipeline qualification controls.  They are deliberately
    # absent from launcher profiles: release testers pass them to the normal
    # frozen CLI, while ordinary one-click behavior remains unbounded.
    metrics_json: Path | None = None
    max_frames: int | None = None
    max_seconds: float | None = None
    require_full_provider: bool = False


def _fourcc(value: str) -> str:
    text = str(value).strip().upper()
    if len(text) != 4 or not text.isalnum():
        raise argparse.ArgumentTypeError(
            "must be a four-character alphanumeric FOURCC code, such as NV12"
        )
    return text


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _inference_size(value: str) -> InferenceSize:
    try:
        return validate_yolo_inference_size(parse_inference_size(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _axis_code(value: str) -> int:
    normalized = value.strip().upper()
    if normalized == "ABS_BRAKE":
        return ABS_BRAKE
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use a non-negative axis number or ABS_BRAKE"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("axis code cannot be negative")
    return parsed


def _head_ratio(value: str) -> float:
    parsed = _finite_float(value)
    if not 0.0 <= parsed <= 0.5:
        raise argparse.ArgumentTypeError("must be between 0 and 0.5")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
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
        "--capture-format",
        type=_fourcc,
        default=None,
        metavar="FOURCC",
        help=(
            "Pixel format to request from a capture card, such as NV12 or "
            "MJPG. Cards commonly reach their highest frame rates only in a "
            "specific mode; the startup banner reports what was granted."
        ),
    )
    parser.add_argument(
        "--backend",
        default="openvino",
        choices=("openvino", "onnxruntime"),
        help=(
            "Inference runtime. Use 'onnxruntime' for AMD or NVIDIA GPUs, which "
            "OpenVINO cannot drive; --device then names an execution provider "
            "such as AUTO, ROCM, CUDA, or DIRECTML."
        ),
    )
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
        type=_inference_size,
        default=(320, 320),
        metavar="N|HEIGHTxWIDTH",
        help=(
            "Model input dimensions: legacy N means a square, or use explicit "
            "HEIGHTxWIDTH tensor order; each dimension must be divisible by 32 "
            "(default: 320)."
        ),
    )
    parser.add_argument(
        "--crop-size",
        type=_positive_int,
        metavar="N",
        help="Centered square crop size in source pixels before inference.",
    )
    parser.add_argument(
        "--detail-crop-size",
        type=_positive_int,
        metavar="SOURCE_WIDTH_PIXELS",
        help=(
            "Advanced: add a second inference over a centered ROI up to this "
            "source-pixel width while retaining the full-frame primary pass. "
            "ROI height matches the model input aspect, so rectangular models "
            "use their full tensor. Disabled by default."
        ),
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
        "--preview-fps",
        type=_positive_float,
        default=15.0,
        metavar="FPS",
        help=(
            "Maximum preview refresh rate (default: 15). Detection and control "
            "continue at full speed between preview refreshes."
        ),
    )
    parser.add_argument(
        "--stats-window",
        type=_window_size,
        default=120,
        metavar="N",
        help="Number of recent processed frames used for moving statistics.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        metavar="PATH",
        help=(
            "Write an atomic, non-overwriting JSON report for this normal live "
            "pipeline run. Pairing keys and physical device paths are omitted."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        metavar="N",
        help=(
            "Stop after exactly N processed frames. When combined with "
            "--max-seconds, whichever bound is reached first stops the run."
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=_positive_float,
        metavar="SECONDS",
        help=(
            "Stop the live pipeline after this many elapsed seconds. When "
            "combined with --max-frames, whichever bound is reached first wins."
        ),
    )
    parser.add_argument(
        "--require-full-provider",
        action="store_true",
        help=(
            "ONNX Runtime qualification only: fail session creation if any graph "
            "node would fall back to CPU."
        ),
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
            "zone and whose box is at least 25%% high and 6%% wide (heuristic)."
        ),
    )
    parser.add_argument(
        "--aim",
        action="store_true",
        help=(
            "Enable physically gated detection output. This requires an explicit "
            "target label and a supported local or MAKCU activation source."
        ),
    )
    parser.add_argument(
        "--aim-label",
        metavar="LABEL",
        help="Required with --aim; only detections matching this exact label are eligible.",
    )
    parser.add_argument(
        "--aim-invert-x",
        action="store_true",
        help="Invert the horizontal aiming axis.",
    )
    parser.add_argument(
        "--aim-invert-y",
        action="store_true",
        help="Invert the vertical aiming axis.",
    )
    parser.add_argument(
        "--aim-head-ratio",
        type=_head_ratio,
        default=DEFAULT_HEAD_RATIO,
        help="Vertical head aim point within a player box (default: 0.12).",
    )
    parser.add_argument(
        "--aim-output",
        choices=("local", "remote", "makcu"),
        default="local",
        help="Send aim to local uinput or a MAKCU mouse board; remote is unavailable.",
    )
    parser.add_argument(
        "--aim-host",
        help="Reserved for remote output, which is unavailable in this release.",
    )
    parser.add_argument(
        "--aim-port",
        type=_port,
        default=47621,
        help="Reserved for remote output, which is unavailable in this release.",
    )
    parser.add_argument(
        "--aim-pairing-key",
        help="Reserved for remote output, which is unavailable in this release.",
    )
    parser.add_argument(
        "--aim-makcu-port",
        help="Optional MAKCU serial path; auto-detected by USB ID when omitted.",
    )
    parser.add_argument(
        "--aim-makcu-button",
        type=int,
        choices=range(5),
        default=1,
        metavar="N",
        help="Physical mouse button that activates MAKCU aim: 0 left, 1 right, 2 middle, 3/4 side.",
    )
    parser.add_argument(
        "--aim-makcu-strength",
        type=_positive_float,
        default=0.50,
        help="Visual-error gain used by the MAKCU control loop (default: 0.50, max: 4).",
    )
    parser.add_argument(
        "--aim-makcu-max-step",
        type=_positive_int,
        default=160,
        help="Maximum relative mouse movement per frame (default: 160).",
    )
    parser.add_argument(
        "--aim-makcu-smoothing-alpha",
        type=_positive_unit_float,
        default=0.78,
        help="Threaded MAKCU smoothing alpha in (0,1] (default: 0.78).",
    )
    parser.add_argument(
        "--aim-makcu-prediction-lead-seconds",
        type=_finite_float,
        default=0.03,
        help="Base predictive lead in seconds for moving targets (default: 0.03).",
    )
    parser.add_argument(
        "--aim-makcu-derivative-damping-seconds",
        type=_finite_float,
        default=0.008,
        help="Velocity damping horizon in seconds (default: 0.008).",
    )
    parser.add_argument(
        "--aim-activate-path",
        metavar="PATH",
        help="Required for local output: physical input path whose LT axis gates every update.",
    )
    parser.add_argument(
        "--aim-activate-axis",
        type=_axis_code,
        default=ABS_BRAKE,
        metavar="AXIS",
        help="Analog axis used to activate aim when held (default: ABS_BRAKE).",
    )
    parser.add_argument(
        "--aim-activate-threshold",
        type=_positive_unit_float,
        default=0.35,
        help="Normalized LT pressure threshold in (0,1] to activate aim (default: 0.35).",
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
    if args.aim_makcu_strength > 4.0:
        parser.error("--aim-makcu-strength must be at most 4")
    if not 0.0 <= args.aim_makcu_prediction_lead_seconds <= 0.25:
        parser.error("--aim-makcu-prediction-lead-seconds must be between 0 and 0.25")
    if not 0.0 <= args.aim_makcu_derivative_damping_seconds <= 0.25:
        parser.error("--aim-makcu-derivative-damping-seconds must be between 0 and 0.25")
    if args.require_full_provider and args.backend != "onnxruntime":
        parser.error("--require-full-provider requires --backend onnxruntime")
    if args.crop_size is not None and args.detail_crop_size is not None:
        parser.error(
            "--crop-size and --detail-crop-size cannot be combined; the detail "
            "mode requires a full-frame primary pass"
        )
    if args.aim:
        args.aim_label = (args.aim_label or "").strip()
        if not args.aim_label:
            parser.error("--aim-label is required when --aim is enabled")
        if not args.ignore_self:
            parser.error("--ignore-self is required when --aim is enabled")
        if args.aim_output == "remote":
            parser.error(
                "--aim-output remote is unavailable because this project has no "
                "authenticated, physically gated receiver"
            )
        if args.aim_output == "local" and not (args.aim_activate_path or "").strip():
            parser.error(
                "--aim-activate-path is required for local aim so a physical control "
                "can gate every output"
            )
        args.aim_activate_path = (args.aim_activate_path or "").strip() or None
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
        backend=args.backend.strip().lower(),
        capture_format=args.capture_format,
        capture_size=args.capture_size,
        capture_fps=args.capture_fps,
        inference_size=args.inference_size,
        crop_size=args.crop_size,
        detail_crop_size=args.detail_crop_size,
        confidence=args.confidence,
        iou_threshold=args.iou_threshold,
        output_format=args.output_format,
        preview=preview,
        draw=draw,
        preview_fps=args.preview_fps,
        stats_window=args.stats_window,
        screen_monitor=screen_monitor,
        screen_region=args.screen_region,
        screen_fps=args.screen_fps,
        ignore_self=args.ignore_self,
        self_zone_left=args.self_zone_left,
        self_zone_width=args.self_zone_width,
        self_zone_height=args.self_zone_height,
        aim=args.aim,
        aim_label=args.aim_label,
        aim_invert_x=args.aim_invert_x,
        aim_invert_y=args.aim_invert_y,
        aim_head_ratio=args.aim_head_ratio,
        aim_output=args.aim_output,
        aim_host=(args.aim_host or "").strip() or None,
        aim_port=args.aim_port,
        aim_pairing_key=args.aim_pairing_key,
        aim_makcu_port=(args.aim_makcu_port or "").strip() or None,
        aim_makcu_button=args.aim_makcu_button,
        aim_makcu_strength=args.aim_makcu_strength,
        aim_makcu_max_step=args.aim_makcu_max_step,
        aim_makcu_smoothing_alpha=args.aim_makcu_smoothing_alpha,
        aim_makcu_prediction_lead_seconds=args.aim_makcu_prediction_lead_seconds,
        aim_makcu_derivative_damping_seconds=args.aim_makcu_derivative_damping_seconds,
        aim_activate_path=args.aim_activate_path,
        aim_activate_axis=args.aim_activate_axis,
        aim_activate_threshold=args.aim_activate_threshold,
        metrics_json=(args.metrics_json.expanduser() if args.metrics_json else None),
        max_frames=args.max_frames,
        max_seconds=args.max_seconds,
        require_full_provider=args.require_full_provider,
    )
