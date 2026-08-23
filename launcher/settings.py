"""Persistent settings and command construction for the desktop launcher.

This module deliberately has no GUI imports.  Keeping the validation and command
builder independent from Tk makes them straightforward to test in a headless
environment and safe to reuse from another front end later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

from utils.inference_size import (
    InferenceSizeLike,
    format_inference_size,
    normalize_inference_size,
    parse_inference_size,
    validate_yolo_inference_size,
)
from utils.release_model_contract import load_release_default_contract

SOURCE_SCREEN = "screen"
SOURCE_CAMERA = "camera"
SOURCE_VIDEO = "video"
SOURCE_MODES = (SOURCE_SCREEN, SOURCE_CAMERA, SOURCE_VIDEO)
AIM_OUTPUT_LOCAL = "local"
AIM_OUTPUT_REMOTE = "remote"
AIM_OUTPUT_MAKCU = "makcu"
AIM_OUTPUTS = (AIM_OUTPUT_LOCAL, AIM_OUTPUT_REMOTE, AIM_OUTPUT_MAKCU)
DEFAULT_AIM_CALIBRATION_CONTEXT = "hip-fire"
_AIM_CALIBRATION_CONTEXT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")
SELF_POSITION_LEFT = "left"
SELF_POSITION_CENTER = "center"
SELF_POSITION_RIGHT = "right"
SELF_POSITIONS = (SELF_POSITION_LEFT, SELF_POSITION_CENTER, SELF_POSITION_RIGHT)
SELF_POSITION_GEOMETRY = {
    SELF_POSITION_LEFT: ("0.18", "0.34", "0.10"),
    SELF_POSITION_CENTER: ("0.33", "0.34", "0.10"),
    SELF_POSITION_RIGHT: ("0.48", "0.34", "0.10"),
}
# Version 4 made bundled detector choices semantic. Version 5 added independent
# preview pacing; version 6 added the opt-in centered detail pass; version 7
# records whether the user has explicitly chosen (or safely auto-selected) an
# inference runtime; version 8 binds a fresh profile's optional detail workload
# to the release-default pointer while preserving every persisted profile;
# version 9 adds persisted camera/capture-card format and orientation controls;
# version 10 adds an explicit, fail-closed MAKCU calibration profile path;
# version 11 persists the physical hip-fire/ADS context that profile is bound
# to, so an ADS calibration can never be reused by an implicit hip-fire run.
# A settings file records the preset key rather than paths into a particular
# checkout/PyInstaller extraction directory, so its model and labels always move
# together.
SETTINGS_VERSION = 12

MODEL_PRESET_FORT_PLAYER_BALANCED = "fort_player_balanced"
MODEL_PRESET_FORT_PLAYER_BALANCED_INT8 = "fort_player_balanced_int8"
MODEL_PRESET_FORT_PLAYER = "fort_player"
MODEL_PRESET_COCO_BALANCED = "coco_balanced"
MODEL_PRESET_COCO_HIGH = "coco_high"
MODEL_PRESET_COCO = "coco"
MODEL_PRESET_CUSTOM = "custom"
DEFAULT_MODEL_PRESET = MODEL_PRESET_FORT_PLAYER_BALANCED

# These tokens were written by settings versions 1--3.  Keep them as migration
# constants: existing profiles that used the old bundled detector must continue
# to select COCO after the Fortnite-style preset becomes the fresh default.
BUNDLED_MODEL = "@bundled/yolo26n.xml"
BUNDLED_LABELS = "@bundled/coco80.txt"


_INITIAL_RESOURCE_ROOT = Path(
    getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)
).resolve()
_RELEASE_DEFAULT_POINTER = load_release_default_contract(_INITIAL_RESOURCE_ROOT)
_RELEASE_DEFAULT_ARTIFACTS = _RELEASE_DEFAULT_POINTER["artifacts"]
_RELEASE_DEFAULT_SHAPE = _RELEASE_DEFAULT_POINTER["input_shape_nchw"]
_RELEASE_DEFAULT_HEIGHT = int(_RELEASE_DEFAULT_SHAPE[2])
_RELEASE_DEFAULT_WIDTH = int(_RELEASE_DEFAULT_SHAPE[3])
_RELEASE_DEFAULT_INFERENCE_SIZE: InferenceSizeLike = (
    _RELEASE_DEFAULT_HEIGHT
    if _RELEASE_DEFAULT_HEIGHT == _RELEASE_DEFAULT_WIDTH
    else (_RELEASE_DEFAULT_HEIGHT, _RELEASE_DEFAULT_WIDTH)
)
_RELEASE_DEFAULT_DETAIL_CROP_SIZE = int(
    _RELEASE_DEFAULT_POINTER["detail_crop_size_source_pixels"]
)
@dataclass(frozen=True, slots=True)
class ModelPreset:
    """A model choice shown in the launcher.

    ``model_relative`` and ``labels_relative`` are deliberately relative to
    :func:`resource_root`.  Frozen applications extract resources to a new
    location, and absolute paths saved by a previous run are not portable.
    """

    key: str
    label: str
    description: str
    model_relative: str | None
    labels_relative: str | None
    inference_size: InferenceSizeLike | None
    # The same trained weights in ONNX form.  OpenVINO cannot drive AMD or
    # NVIDIA GPUs, so a preset must be able to hand ONNX Runtime an .onnx graph
    # instead of an OpenVINO .xml when the hardware scan selects that backend.
    onnx_relative: str | None = None
    # Zero disables the second pass. A positive value is the requested source
    # ROI width; runtime derives the height from the static model aspect ratio.
    detail_crop_size_source_pixels: int = 0

    @property
    def bundled(self) -> bool:
        return self.model_relative is not None and self.labels_relative is not None

    def model_for(self, backend: str) -> str | None:
        if str(backend).strip().lower() == "onnxruntime":
            return self.onnx_relative
        return self.model_relative


MODEL_PRESETS = (
    ModelPreset(
        key=MODEL_PRESET_FORT_PLAYER_BALANCED,
        label=str(_RELEASE_DEFAULT_POINTER["preset"]["label"]),
        description=str(_RELEASE_DEFAULT_POINTER["preset"]["description"]),
        model_relative=str(_RELEASE_DEFAULT_ARTIFACTS["openvino_xml"]["path"]),
        labels_relative=str(_RELEASE_DEFAULT_ARTIFACTS["labels"]["path"]),
        inference_size=_RELEASE_DEFAULT_INFERENCE_SIZE,
        onnx_relative=str(_RELEASE_DEFAULT_ARTIFACTS["onnx"]["path"]),
        detail_crop_size_source_pixels=_RELEASE_DEFAULT_DETAIL_CROP_SIZE,
    ),
    ModelPreset(
        key=MODEL_PRESET_FORT_PLAYER_BALANCED_INT8,
        label="Game players — Responsive 416 INT8 (OpenVINO CPU)",
        description=(
            "Quantized one-class player detector for lower-latency OpenVINO CPU use."
        ),
        model_relative=(
            "models/fort_player_416_int8_openvino_model/fort_player_416_int8.xml"
        ),
        labels_relative="models/fort_player.txt",
        inference_size=416,
        onnx_relative=None,
    ),
    ModelPreset(
        key=MODEL_PRESET_FORT_PLAYER,
        label="Game players — Fast 320",
        description=(
            "Lower-latency player detector; less accurate on small or distant players."
        ),
        model_relative="models/fort_player_openvino_model/fort_player.xml",
        labels_relative="models/fort_player.txt",
        inference_size=320,
        onnx_relative="models/fort_player_onnx/fort_player.onnx",
    ),
    ModelPreset(
        key=MODEL_PRESET_COCO_BALANCED,
        label="People — Balanced 416 (COCO fallback)",
        description=(
            "Higher-detail COCO person detection for balanced CPU latency and range."
        ),
        model_relative="models/yolo26n_416_openvino_model/yolo26n_416.xml",
        labels_relative="models/coco80.txt",
        inference_size=416,
        onnx_relative="models/yolo26n_416_onnx/yolo26n_416.onnx",
    ),
    ModelPreset(
        key=MODEL_PRESET_COCO_HIGH,
        label="Benchmark only — YOLO11l 640 (slow, generic COCO)",
        description=(
            "Accuracy benchmark for explicit testing. It is much heavier than the "
            "clone-trained player models and is not recommended for responsive tracking."
        ),
        model_relative="models/yolo11l_openvino_model/yolo11l.xml",
        labels_relative="models/coco80.txt",
        inference_size=640,
        onnx_relative="models/yolo11l_onnx/yolo11l.onnx",
    ),
    ModelPreset(
        key=MODEL_PRESET_COCO,
        label="People — Fast 320",
        description="Lower-latency COCO person detection when update rate matters most.",
        model_relative="models/yolo26n_openvino_model/yolo26n.xml",
        labels_relative="models/coco80.txt",
        inference_size=320,
        onnx_relative="models/yolo26n_onnx/yolo26n.onnx",
    ),
    ModelPreset(
        key=MODEL_PRESET_CUSTOM,
        label="Custom model files",
        description="Use your own matching OpenVINO .xml/.bin model and labels file.",
        model_relative=None,
        labels_relative=None,
        inference_size=None,
    ),
)
_MODEL_PRESETS_BY_KEY = {preset.key: preset for preset in MODEL_PRESETS}


class SettingsError(ValueError):
    """Raised when launcher settings cannot form a valid detector command."""


def resource_root() -> Path:
    """Return the checkout or PyInstaller resource directory."""

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).resolve()
    return Path(__file__).resolve().parent.parent


def model_preset(value: str) -> ModelPreset:
    """Return a known preset, falling back to the fresh-install default."""

    return _MODEL_PRESETS_BY_KEY.get(value, _MODEL_PRESETS_BY_KEY[DEFAULT_MODEL_PRESET])


def model_preset_detail_crop_text(value: str) -> str:
    """Return the preset workload's CLI text (empty means detail disabled)."""

    crop = model_preset(value).detail_crop_size_source_pixels
    return str(crop) if crop > 0 else ""


def model_preset_paths(value: str, backend: str = "openvino") -> tuple[str, str]:
    """Resolve both files for a bundled preset in one operation.

    ``backend`` selects the model format, because the two runtimes read
    different files: OpenVINO reads an ``.xml``/``.bin`` pair and ONNX Runtime
    reads an ``.onnx`` graph.  The labels file is shared by both.
    """

    preset = _MODEL_PRESETS_BY_KEY.get(value)
    if preset is None or not preset.bundled:
        raise ValueError(f"Model preset does not provide bundled files: {value!r}")
    assert preset.labels_relative is not None
    model_relative = preset.model_for(backend)
    if model_relative is None:
        raise ValueError(
            f"Model preset {value!r} has no model for the {backend!r} backend."
        )
    root = resource_root()
    return str(root / model_relative), str(root / preset.labels_relative)


def release_default_model_contract() -> dict[str, object]:
    """Return the canonical bundled ONNX model deployed as the release default.

    Paths in this contract are resource-root-relative POSIX paths. Build
    tooling deliberately derives them from the same preset catalog used by the
    launcher instead of maintaining a second model/shape table.
    """

    preset = _MODEL_PRESETS_BY_KEY[DEFAULT_MODEL_PRESET]
    model_relative = preset.model_for("onnxruntime")
    labels_relative = preset.labels_relative
    if model_relative is None or labels_relative is None or preset.inference_size is None:
        raise ValueError(
            f"Release default preset {preset.key!r} must provide ONNX, labels, and input size"
        )
    for description, value in (
        ("model", model_relative),
        ("labels", labels_relative),
    ):
        normalized = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or normalized.is_absolute()
            or normalized.as_posix() != value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError(
                f"Release default {description} path must be canonical resource-relative POSIX"
            )
    if PurePosixPath(model_relative).suffix.lower() != ".onnx":
        raise ValueError("Release default model must be an ONNX graph")
    height, width = validate_yolo_inference_size(preset.inference_size)
    return {
        "preset": preset.key,
        "model_path": model_relative,
        "labels_path": labels_relative,
        "input_shape_hw": [height, width],
        "detail_crop_size_source_pixels": preset.detail_crop_size_source_pixels,
        "pointer_content_sha256": _RELEASE_DEFAULT_POINTER["content_sha256"],
        "qualification": dict(_RELEASE_DEFAULT_POINTER["qualification"]),
    }


def default_model_path() -> str:
    return model_preset_paths(DEFAULT_MODEL_PRESET)[0]


def default_labels_path() -> str:
    return model_preset_paths(DEFAULT_MODEL_PRESET)[1]


def coco_model_path() -> str:
    return model_preset_paths(MODEL_PRESET_COCO)[0]


def coco_labels_path() -> str:
    return model_preset_paths(MODEL_PRESET_COCO)[1]


def _settings_paths() -> tuple[Path, Path]:
    """Return the current settings path and the pre-rename one.

    The application was named Game Detector before it became ProAim.  Returning
    both lets a first run adopt an existing profile instead of silently starting
    from defaults and appearing to have lost the user's configuration.
    """

    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / "ProAim" / "settings.json", root / "GameDetector" / "settings.json"
    if sys.platform == "darwin":
        support = Path.home() / "Library" / "Application Support"
        return (
            support / "ProAim" / "settings.json",
            support / "GameDetector" / "settings.json",
        )

    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "proaim" / "settings.json", root / "game-detector" / "settings.json"


def settings_path() -> Path:
    """Return the per-user JSON settings path on the current platform.

    This is always the current location so that the next save completes the
    migration; reading falls back to the legacy file separately.
    """

    return _settings_paths()[0]


def legacy_settings_path() -> Path:
    """Return the pre-rename settings path."""

    return _settings_paths()[1]


@dataclass(slots=True)
class LauncherSettings:
    """User-editable launcher state.

    Entry-widget values remain strings so partially edited values can be saved
    without losing the user's work.  ``detector_arguments`` performs strict
    conversion only when Start is pressed.
    """

    version: int = SETTINGS_VERSION
    source_mode: str = SOURCE_SCREEN
    video_path: str = ""
    camera_index: str = "0"
    capture_width: str = "1280"
    capture_height: str = "720"
    capture_fps: str = "120"
    capture_format: str = ""
    capture_rotate_180: bool = False
    screen_monitor: str = "1"
    use_screen_region: bool = False
    screen_x: str = "0"
    screen_y: str = "0"
    screen_width: str = "1920"
    screen_height: str = "1080"
    screen_fps: str = "120"
    # Empty is an initialization sentinel: no paths means the fresh bundled
    # preset, while explicitly supplied paths mean a custom model.  Every live
    # LauncherSettings instance is normalized to a semantic key in
    # ``__post_init__``.
    model_preset: str = ""
    model_tier: str = "mid"
    model_path: str = ""
    labels_path: str = ""
    device: str = "CPU"
    # OpenVINO cannot drive AMD or NVIDIA GPUs; those run through ONNX
    # Runtime, and the hardware scan sets this automatically.
    backend: str = "openvino"
    # False is a fresh-profile sentinel used only by the Qt launcher. Existing
    # profiles migrate to True so a new release never overwrites an explicit
    # device choice merely because this field did not exist when it was saved.
    hardware_selection_configured: bool = False
    inference_size: str = "320"
    crop_size: str = ""
    detail_crop_size: str = field(
        default_factory=lambda: model_preset_detail_crop_text(DEFAULT_MODEL_PRESET)
    )
    confidence: str = "0.25"
    iou_threshold: str = "0.45"
    output_format: str = "auto"
    # Filtering must be explicitly enabled in a profile because camera/video
    # users should never silently lose a foreground person detection.
    ignore_self: bool = False
    self_position: str = SELF_POSITION_LEFT
    preview: bool = True
    draw: bool = True
    preview_fps: str = "30"
    aim: bool = False
    aim_label: str = ""
    aim_invert_x: bool = False
    aim_invert_y: bool = False
    aim_head_ratio: str = "0.12"
    aim_output: str = AIM_OUTPUT_LOCAL
    aim_host: str = ""
    aim_port: str = "47621"
    aim_pairing_key: str = field(default_factory=lambda: secrets.token_hex(16))
    aim_makcu_port: str = ""
    aim_makcu_button: str = "1"
    aim_makcu_strength: str = "0.50"
    aim_makcu_max_step: str = "160"
    aim_makcu_smoothing_alpha: str = "0.78"
    aim_makcu_prediction_lead_seconds: str = "0.03"
    aim_makcu_derivative_damping_seconds: str = "0.008"
    aim_makcu_vertical_rate_ratio: str = "0.48"
    aim_makcu_tracking_mode: str = "stable-body"
    aim_diagnostics: bool = True
    aim_diagnostic_sample_hz: str = "20"
    aim_diagnostic_max_duration_seconds: str = "30"
    aim_makcu_context: str = "hip"
    aim_makcu_active_profile: str = ""
    aim_makcu_verified_port: str = ""
    aim_makcu_verified_button: str = ""
    aim_activate_path: str = ""
    aim_activate_axis: int = 10
    aim_activate_threshold: str = "0.35"
    # Controller precision is a separate, user-driven process.  Only the
    # selected device, curve preset, and a device-bound verification record are
    # persistent; whether the worker is running is intentionally never saved.
    precision_device_path: str = ""
    precision_device_identity: str = ""
    precision_mapping_verified: bool = False
    precision_preset: str = "balanced"
    precision_trigger_rest: str = ""
    precision_trigger_pressed: str = ""

    def __post_init__(self) -> None:
        # ``fort_player`` is a bundled preset again, so a profile that stored it
        # while the 320 player model was the only bundled detector still selects
        # that same 320 player model rather than silently changing detector.
        if not self.model_preset:
            self.model_preset = (
                MODEL_PRESET_CUSTOM
                if self.model_path or self.labels_path
                else DEFAULT_MODEL_PRESET
            )
        elif self.model_preset not in _MODEL_PRESETS_BY_KEY:
            self.model_preset = (
                MODEL_PRESET_CUSTOM
                if self.model_path or self.labels_path
                else DEFAULT_MODEL_PRESET
            )

        preset = model_preset(self.model_preset)
        if preset.bundled and preset.model_for(self.backend) is None:
            # An OpenVINO-only preset can survive in a profile after the user
            # switches to an ONNX GPU. Keep the player class semantics while
            # falling back to the portable FP32 416 preset.
            self.model_preset = DEFAULT_MODEL_PRESET
            preset = model_preset(self.model_preset)
        if preset.bundled:
            # Ignore stale/partial serialized paths.  A bundled preset is an
            # atomic model+labels choice and always resolves against this run's
            # resource root.  The included YOLO26 models must use automatic
            # output decoding; a custom traditional/end2end choice is not
            # compatible state for either bundled preset.
            self.model_path, self.labels_path = model_preset_paths(
                preset.key, self.backend
            )
            self.output_format = "auto"
            assert preset.inference_size is not None
            self.inference_size = format_inference_size(preset.inference_size)
        else:
            # Canonicalize valid custom values before they are serialized,
            # while preserving malformed profile text for the ordinary form
            # validation message instead of silently changing it.
            try:
                self.inference_size = format_inference_size(
                    parse_inference_size(self.inference_size)
                )
            except (TypeError, ValueError):
                pass
        if self.precision_mapping_verified and not self.precision_device_identity.strip():
            self.precision_mapping_verified = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LauncherSettings":
        """Load known fields while tolerating old, new, or malformed JSON."""

        defaults = cls()
        accepted = {field.name: field for field in fields(cls)}
        converted: dict[str, Any] = {}
        for name, value in values.items():
            if name not in accepted or name == "version":
                continue
            default_value = getattr(defaults, name)
            if isinstance(default_value, bool):
                if isinstance(value, bool):
                    converted[name] = value
            elif isinstance(default_value, int):
                # JSON keeps integer values distinct from strings, but bool is
                # an ``int`` subclass in Python.  Preserve real integer fields
                # such as ``aim_activate_axis`` without accepting true/false as
                # an axis number.
                if isinstance(value, int) and not isinstance(value, bool):
                    converted[name] = value
                elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                    converted[name] = int(value)
            elif isinstance(default_value, str) and isinstance(value, (str, int, float)):
                converted[name] = str(value)

        version = _settings_version(values.get("version"))
        if converted.get("aim_makcu_context") not in {"hip", "ads"}:
            # This field first exists in v11 and the desktop UI deliberately
            # offers only the two physical modes. Never silently reinterpret
            # an arbitrary context string as one of them.
            converted.pop("aim_makcu_context", None)
        # Loading a persisted profile must never opt it into a newly selected
        # release workload. Only a genuinely fresh LauncherSettings instance
        # inherits the pointer's detail crop; an old/partial profile without
        # this field retains the historical full-frame-only behavior.
        if "detail_crop_size" not in converted:
            converted["detail_crop_size"] = ""
        hardware_selection_key_present = "hardware_selection_configured" in values
        if "hardware_selection_configured" not in converted:
            if hardware_selection_key_present:
                # The current sentinel is security-sensitive.  A non-boolean
                # value must never fall through to legacy inference and turn a
                # malformed profile into implicit consent for CPU (or any
                # other runtime).  Reset only the runtime choice so the rest of
                # the user's profile can still be recovered.
                converted["hardware_selection_configured"] = False
                converted["backend"] = "openvino"
                converted["device"] = "CPU"
            elif version < 7:
                # Preserve a pre-v7 runtime only when the serialized object
                # really contains a usable device/backend choice. Empty or
                # partial JSON is not evidence that default CPU was
                # intentional; treating it as a fresh profile lets Qt discover
                # one safely bound GPU and still requires confirmation before
                # a CPU fallback.
                legacy_backend = str(converted.get("backend", "")).strip().lower()
                legacy_device = str(converted.get("device", "")).strip()
                converted["hardware_selection_configured"] = bool(
                    legacy_backend in {"openvino", "onnxruntime"} and legacy_device
                )
            else:
                # A current/future profile missing the sentinel is incomplete,
                # not a legacy runtime selection.
                converted["hardware_selection_configured"] = False
        if converted.get("hardware_selection_configured") is True:
            # A hand-edited/corrupted v7 file must not authenticate a runtime
            # value that the launcher cannot represent. Preserve valid explicit
            # choices, but send malformed ones through fresh-profile discovery.
            serialized_backend = str(converted.get("backend", "openvino")).strip().lower()
            serialized_device = str(converted.get("device", "CPU")).strip()
            if (
                serialized_backend not in {"openvino", "onnxruntime"}
                or not serialized_device
            ):
                converted["hardware_selection_configured"] = False
                converted["backend"] = "openvino"
                converted["device"] = "CPU"
        elif str(converted.get("backend", "openvino")).strip().lower() not in {
            "openvino",
            "onnxruntime",
        }:
            # Do not carry malformed legacy runtime pairs into the form.  They
            # are not an intentional CPU choice and must remain eligible for
            # fresh-profile hardware discovery.
            converted["backend"] = "openvino"
            converted["device"] = "CPU"
        model_value = converted.get("model_path", "")
        labels_value = converted.get("labels_path", "")
        if version < 4:
            # Versions 1--3 had one bundled model (COCO).  Missing paths and
            # the two semantic tokens both represent that default.  Any real
            # path was user-selected and therefore becomes Custom.
            legacy_bundled = (
                model_value in ("", BUNDLED_MODEL)
                and labels_value in ("", BUNDLED_LABELS)
            )
            converted["model_preset"] = (
                MODEL_PRESET_COCO if legacy_bundled else MODEL_PRESET_CUSTOM
            )
            if legacy_bundled:
                converted.pop("model_path", None)
                converted.pop("labels_path", None)
        else:
            requested = converted.get("model_preset", "")
            if requested not in _MODEL_PRESETS_BY_KEY:
                requested = (
                    MODEL_PRESET_CUSTOM
                    if model_value or labels_value
                    else DEFAULT_MODEL_PRESET
                )
            converted["model_preset"] = requested
            if requested != MODEL_PRESET_CUSTOM:
                # The preset key is authoritative; never allow a mixed model
                # and labels pair from stale or hand-edited JSON.
                converted.pop("model_path", None)
                converted.pop("labels_path", None)
        return cls(version=SETTINGS_VERSION, **converted)

    def detector_arguments(self) -> list[str]:
        """Validate the settings and return arguments accepted by ``main.py``."""

        if self.source_mode not in SOURCE_MODES:
            raise SettingsError("Choose Moonlight screen, camera/capture card, or video file.")

        try:
            model_path, labels_path = self.resolved_model_files()
        except ValueError as exc:
            raise SettingsError(str(exc)) from exc
        backend = _backend_name(self.backend)
        device = _device_name(self.device)
        if backend == "onnxruntime":
            # An ONNX graph is a single self-contained file, so there is no
            # separate weights file to keep beside it.
            model = _existing_file(model_path, "ONNX model")
            if model.suffix.lower() != ".onnx":
                raise SettingsError("The ONNX Runtime model must be an .onnx file.")
        else:
            model = _existing_file(model_path, "OpenVINO model")
            if model.suffix.lower() != ".xml":
                raise SettingsError("The OpenVINO model must be an .xml file.")
            weights = model.with_suffix(".bin")
            if not weights.is_file():
                raise SettingsError(
                    f"The model weights were not found: {weights}. "
                    "Keep the matching .xml and .bin files together."
                )
        labels = _existing_file(labels_path, "labels file")

        preset = model_preset(self.model_preset)
        output_format = "auto" if preset.bundled else self.output_format.lower()
        inference_size_value = _inference_size(self.inference_size)
        if preset.bundled and preset.inference_size is not None:
            # Bundled presets are exported at fixed static HxW shapes; allowing
            # stale/edited dimensions can produce internal anchor mismatches at
            # compile time. Always run them at their exact deployment shape.
            inference_size_value = normalize_inference_size(preset.inference_size)
        args = [
            "--model",
            str(model),
            "--labels",
            str(labels),
            "--device",
            device,
            "--backend",
            backend,
            "--inference-size",
            format_inference_size(inference_size_value),
            "--confidence",
            _unit_float_text(self.confidence, "confidence"),
            "--iou-threshold",
            _unit_float_text(self.iou_threshold, "IoU threshold"),
            "--output-format",
            _choice(
                output_format,
                ("auto", "end2end", "traditional"),
                "model output format",
            ),
        ]
        if backend == "onnxruntime" and _requires_full_gpu_provider(device):
            # Explicit accelerator choices must fail closed.  ONNX Runtime can
            # otherwise assign unsupported graph nodes to CPU or retry a failed
            # provider with a CPU-only session while the launcher still appears
            # to be running the selected GPU.
            args.append("--require-full-provider")

        crop = self.crop_size.strip()
        detail_crop = self.detail_crop_size.strip()
        if crop and detail_crop:
            raise SettingsError(
                "Choose either the legacy centered crop or the detail pass, not "
                "both. The detail pass keeps a full-frame primary inference."
            )
        if crop:
            args.extend(("--crop-size", str(_positive_int(crop, "crop size"))))
        if detail_crop:
            args.extend(
                (
                    "--detail-crop-size",
                    str(_positive_int(detail_crop, "detail crop size")),
                )
            )

        if self.ignore_self:
            position = _choice(
                self.self_position,
                SELF_POSITIONS,
                "on-screen character position",
            )
            left, width, height = SELF_POSITION_GEOMETRY[position]
            args.extend(
                (
                    "--ignore-self",
                    "--self-zone-left",
                    left,
                    "--self-zone-width",
                    width,
                    "--self-zone-height",
                    height,
                )
            )

        if self.source_mode == SOURCE_SCREEN:
            args.extend(("--source", "screen"))
            args.extend(
                (
                    "--screen-fps",
                    _positive_float_text(self.screen_fps, "screen capture FPS"),
                )
            )
            if self.use_screen_region:
                left = _integer(self.screen_x, "region X")
                top = _integer(self.screen_y, "region Y")
                width = _positive_int(self.screen_width, "region width")
                height = _positive_int(self.screen_height, "region height")
                args.extend(("--screen-region", f"{left},{top},{width},{height}"))
            else:
                args.extend(
                    ("--screen-monitor", str(_non_negative_int(self.screen_monitor, "monitor")))
                )
        elif self.source_mode == SOURCE_CAMERA:
            args.extend(("--source", str(_non_negative_int(self.camera_index, "camera index"))))
            width = self.capture_width.strip()
            height = self.capture_height.strip()
            if bool(width) != bool(height):
                raise SettingsError("Enter both capture width and height, or leave both blank.")
            if width:
                args.extend(
                    (
                        "--capture-size",
                        f"{_positive_int(width, 'capture width')}x"
                        f"{_positive_int(height, 'capture height')}",
                    )
                )
            fps = self.capture_fps.strip()
            if fps:
                args.extend(("--capture-fps", _positive_float_text(fps, "capture FPS")))
            pixel_format = self.capture_format.strip().upper()
            if pixel_format:
                args.extend(("--capture-format", _fourcc_text(pixel_format)))
            if self.capture_rotate_180:
                args.append("--capture-rotate-180")
        else:
            video = _existing_file(self.video_path, "video file")
            args.extend(("--source", str(video)))

        if not self.preview:
            args.append("--no-preview")
        else:
            args.extend(
                (
                    "--preview-fps",
                    _positive_float_text(self.preview_fps, "preview FPS"),
                )
            )
            if not self.draw:
                args.append("--no-draw")
        aim_enabled = self.aim
        if aim_enabled and self.model_preset in {
            MODEL_PRESET_COCO_HIGH,
            MODEL_PRESET_COCO_BALANCED,
            MODEL_PRESET_COCO,
        }:
            print(
                "Warning: the selected COCO model is generic and is not "
                "validated for clone-player aim output; launching detection-only "
                "instead.",
                file=sys.stderr,
            )
            aim_enabled = False
        if aim_enabled:
            aim_label = self.aim_label.strip()
            if not aim_label:
                raise SettingsError(
                    "Choose an explicit target label before enabling aim output."
                )
            if not self.ignore_self:
                raise SettingsError(
                    "Enable 'Ignore my on-screen character' before enabling aim output."
                )
            args.append("--aim")
            args.extend(("--aim-label", aim_label))
            if self.aim_invert_x:
                args.append("--aim-invert-x")
            if self.aim_invert_y:
                args.append("--aim-invert-y")
            head_ratio = _float(self.aim_head_ratio, "head aim point")
            if not 0.0 <= head_ratio <= 0.5:
                raise SettingsError("Head aim point must be between 0 and 0.5.")
            args.extend(("--aim-head-ratio", f"{head_ratio:g}"))
            aim_output = _choice(self.aim_output, AIM_OUTPUTS, "aim output")
            args.extend(("--aim-output", aim_output))
            if aim_output == AIM_OUTPUT_REMOTE:
                raise SettingsError(
                    "Remote aim output is unavailable until a safe, physically "
                    "activated receiver is implemented."
                )
            elif aim_output == AIM_OUTPUT_MAKCU:
                makcu_port = self.aim_makcu_port.strip()
                if not makcu_port:
                    raise SettingsError("Select or detect a MAKCU serial device before starting.")
                makcu_button = str(
                    _range_int(self.aim_makcu_button, "MAKCU activation button", 0, 4)
                )
                if (
                    self.aim_makcu_verified_port.strip() != makcu_port
                    or self.aim_makcu_verified_button.strip() != makcu_button
                ):
                    raise SettingsError(
                        "Verify the selected MAKCU device and activation button with "
                        "a complete physical press and release before starting."
                    )
                args.extend(("--aim-makcu-port", makcu_port))
                args.extend(
                    (
                        "--aim-makcu-button",
                        makcu_button,
                        "--aim-makcu-strength",
                        _bounded_positive_float_text(
                            self.aim_makcu_strength,
                            "MAKCU aim strength",
                            4.0,
                        ),
                        "--aim-makcu-max-step",
                        str(_positive_int(self.aim_makcu_max_step, "MAKCU maximum step")),
                        "--aim-makcu-smoothing-alpha",
                        _bounded_positive_float_text(
                            self.aim_makcu_smoothing_alpha,
                            "MAKCU smoothing",
                            1.0,
                        ),
                        "--aim-makcu-prediction-lead-seconds",
                        _bounded_non_negative_float_text(
                            self.aim_makcu_prediction_lead_seconds,
                            "MAKCU prediction lead",
                            0.25,
                        ),
                        "--aim-makcu-derivative-damping-seconds",
                        _bounded_non_negative_float_text(
                            self.aim_makcu_derivative_damping_seconds,
                            "MAKCU derivative damping",
                            0.25,
                        ),
                        "--aim-makcu-vertical-rate-ratio",
                        _bounded_positive_float_text(
                            self.aim_makcu_vertical_rate_ratio,
                            "MAKCU vertical cap",
                            1.0,
                        ),
                        "--aim-makcu-tracking-mode",
                        _choice(
                            self.aim_makcu_tracking_mode,
                            ("stable-body", "direct-head"),
                            "MAKCU tracking mode",
                        ),
                        "--aim-calibration-context",
                        _launcher_aim_context_text(self.aim_makcu_context),
                    )
                )
                if self.aim_diagnostics:
                    state_root = Path(
                        os.environ.get(
                            "XDG_STATE_HOME",
                            str(Path.home() / ".local" / "state"),
                        )
                    )
                    args.extend(
                        (
                            "--aim-diagnostic-dir",
                            str(state_root / "proaim" / "diagnostics"),
                            "--aim-diagnostic-sample-hz",
                            _bounded_positive_float_text(
                                self.aim_diagnostic_sample_hz,
                                "aim diagnostic sample rate",
                                60.0,
                            ),
                            "--aim-diagnostic-max-duration-seconds",
                            _bounded_positive_float_text(
                                self.aim_diagnostic_max_duration_seconds,
                                "aim diagnostic duration",
                                300.0,
                            ),
                        )
                    )
                active_profile_text = self.aim_makcu_active_profile.strip()
                if active_profile_text:
                    active_profile = _existing_non_symlink_file(
                        active_profile_text,
                        "MAKCU active calibration profile",
                    )
                    args.extend(
                        (
                            "--aim-makcu-active-profile",
                            str(active_profile),
                        )
                    )
            else:
                activation_path = self.aim_activate_path.strip()
                if not activation_path:
                    raise SettingsError(
                        "Local aim output requires an explicit physical activation device."
                    )
                args.extend(("--aim-activate-path", activation_path))
                if self.aim_activate_axis is not None:
                    args.extend(("--aim-activate-axis", str(self.aim_activate_axis)))
                if self.aim_activate_threshold.strip():
                    args.extend(
                        (
                            "--aim-activate-threshold",
                            _positive_unit_float_text(
                                self.aim_activate_threshold,
                                "aim activation threshold",
                            ),
                        )
                    )
        return args

    def resolved_model_files(self) -> tuple[str, str]:
        """Return the active model and labels, keeping presets atomic."""

        preset = model_preset(self.model_preset)
        if preset.bundled:
            return model_preset_paths(preset.key, self.backend)
        return self.model_path, self.labels_path


def launcher_command(
    settings: LauncherSettings,
    *,
    executable: str | Path | None = None,
    app_script: str | Path | None = None,
    frozen: bool | None = None,
) -> list[str]:
    """Return a shell-free command for a source or frozen application."""

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    program = str(executable or sys.executable)
    if is_frozen and sys.platform == "win32":
        # The Windows GUI uses PyInstaller's no-console bootloader, whose
        # stdout/stderr objects are unavailable.  Send detector work to the
        # sibling console-helper build so its output can be piped back into
        # the GUI without opening a terminal window.
        helper = Path(program).with_name("ProAimCLI.exe")
        if helper.is_file():
            program = str(helper)
    prefix = [program]
    if not is_frozen:
        prefix.append(str(app_script or (resource_root() / "app.py")))
    return [*prefix, "--cli", *settings.detector_arguments()]


def makcu_calibration_command(
    settings: LauncherSettings,
    evidence_path: str | Path,
    context: str = DEFAULT_AIM_CALIBRATION_CONTEXT,
    *,
    executable: str | Path | None = None,
    app_script: str | Path | None = None,
    frozen: bool | None = None,
) -> list[str]:
    """Return a strict, one-shot live MAKCU calibration command.

    Calibration deliberately reuses all ordinary detector/aim validation.  Its
    evidence destination and context are ephemeral command inputs rather than
    persisted launcher settings, and the secondary detail pass is removed so
    every measurement has one unambiguous inference timestamp.
    """

    if not settings.aim:
        raise SettingsError("Enable aim before starting MAKCU calibration.")
    if settings.aim_output != AIM_OUTPUT_MAKCU:
        raise SettingsError("MAKCU calibration requires MAKCU aim output.")
    if settings.source_mode not in {SOURCE_SCREEN, SOURCE_CAMERA}:
        raise SettingsError(
            "MAKCU calibration requires a live screen or camera/capture-card source."
        )
    if settings.aim_makcu_active_profile.strip():
        raise SettingsError(
            "MAKCU calibration cannot run while an active calibration profile "
            "is selected. Clear it before collecting new evidence."
        )
    evidence = _new_calibration_output(evidence_path)
    calibration_context = _aim_calibration_context_text(context)
    command = launcher_command(
        settings,
        executable=executable,
        app_script=app_script,
        frozen=frozen,
    )
    if "--detail-crop-size" in command:
        detail_index = command.index("--detail-crop-size")
        del command[detail_index : detail_index + 2]
    if "--aim-calibration-context" in command:
        context_index = command.index("--aim-calibration-context")
        del command[context_index : context_index + 2]
    command.extend(
        (
            "--aim-calibration-evidence",
            str(evidence),
            "--aim-calibration-context",
            calibration_context,
        )
    )
    return command


def load_settings(path: Path | None = None) -> LauncherSettings:
    target = path or settings_path()
    if not target.exists() and path is None:
        # Adopt a profile written before the application was renamed.  The next
        # save lands in the current location, which completes the move without
        # touching the old file.
        legacy = legacy_settings_path()
        if legacy.is_file():
            target = legacy
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return LauncherSettings()
    except (OSError, UnicodeError, json.JSONDecodeError):
        # A damaged preference file must never keep the launcher from opening.
        # Its contents are unavailable, so the default CPU value is not an
        # authenticated user choice. Treat this exactly like a fresh profile:
        # Qt may scan for one safely bound GPU and otherwise requires explicit
        # CPU confirmation before Start.
        return LauncherSettings()
    if not isinstance(raw, Mapping):
        return LauncherSettings()
    return LauncherSettings.from_mapping(raw)


def save_settings(settings: LauncherSettings, path: Path | None = None) -> Path:
    """Atomically save settings and return the written path."""

    target = path or settings_path()
    parent_existed = target.parent.exists()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    is_default_location = path is None or target == settings_path()
    if is_default_location or not parent_existed:
        try:
            target.parent.chmod(0o700)
        except OSError:
            # Some platforms/filesystems do not implement POSIX permission bits.
            pass
    temporary: Path | None = None
    serialized = asdict(settings)
    serialized["version"] = SETTINGS_VERSION
    if settings.model_preset != MODEL_PRESET_CUSTOM:
        # The key is the complete bundled selection.  Omitting paths avoids
        # persisting PyInstaller extraction locations and prevents mismatched
        # model/labels pairs in the preference file.
        serialized.pop("model_path", None)
        serialized.pop("labels_path", None)
    payload = json.dumps(serialized, indent=2, sort_keys=True) + "\n"
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, target)
        try:
            target.chmod(0o600)
        except OSError:
            pass
    except OSError:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def _existing_file(value: str, label: str) -> Path:
    text = value.strip()
    if not text:
        raise SettingsError(f"Choose a {label}.")
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        raise SettingsError(f"The {label} was not found: {path}")
    return path


def _existing_non_symlink_file(value: str, label: str) -> Path:
    text = value.strip()
    if not text:
        raise SettingsError(f"Choose a {label}.")
    selected = Path(text).expanduser()
    try:
        metadata = selected.lstat()
    except OSError:
        raise SettingsError(
            f"The {label} must be an existing non-symlink regular file: "
            f"{selected.absolute()}"
        ) from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SettingsError(
            f"The {label} must be an existing non-symlink regular file: "
            f"{selected.absolute()}"
        )
    try:
        path = selected.resolve(strict=True)
    except OSError:
        raise SettingsError(
            f"The {label} could not be resolved to an exact absolute path: "
            f"{selected.absolute()}"
        ) from None
    try:
        resolved_metadata = path.lstat()
    except OSError:
        raise SettingsError(f"The {label} was not found: {path}") from None
    if not stat.S_ISREG(resolved_metadata.st_mode):
        raise SettingsError(
            f"The {label} must resolve to a regular file: {path}"
        )
    return path


def _new_calibration_output(value: str | Path) -> Path:
    text = os.fspath(value).strip()
    if not text:
        raise SettingsError("Choose a MAKCU calibration evidence output path.")
    path = Path(text).expanduser().resolve()
    if os.path.lexists(path):
        raise SettingsError(
            f"The MAKCU calibration evidence output already exists: {path}"
        )
    return path


def _aim_calibration_context_text(value: str) -> str:
    text = str(value)
    if _AIM_CALIBRATION_CONTEXT_RE.fullmatch(text) is None:
        raise SettingsError(
            "Calibration context must be 1-64 characters, begin with a letter "
            "or digit, and contain only letters, digits, '.', '_', '+', or '-'."
        )
    return text


def _launcher_aim_context_text(value: str) -> str:
    """Return the explicit physical mode offered by the desktop launchers."""

    text = str(value).strip().casefold()
    if text not in {"hip", "ads"}:
        raise SettingsError("MAKCU aim context must be either hip fire or ADS.")
    return text


def _integer(value: str, label: str) -> int:
    text = value.strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise SettingsError(f"{label.capitalize()} must be a whole number.")
    return int(text)


def _positive_int(value: str, label: str) -> int:
    parsed = _integer(value, label)
    if parsed <= 0:
        raise SettingsError(f"{label.capitalize()} must be greater than zero.")
    return parsed


def _inference_size(value: str) -> tuple[int, int]:
    try:
        return validate_yolo_inference_size(parse_inference_size(value))
    except (TypeError, ValueError) as exc:
        raise SettingsError(
            "Inference size must be N or HEIGHTxWIDTH, with both dimensions "
            "divisible by 32 (for example 416 or 384x640)."
        ) from exc


def _non_negative_int(value: str, label: str) -> int:
    parsed = _integer(value, label)
    if parsed < 0:
        raise SettingsError(f"{label.capitalize()} cannot be negative.")
    return parsed


def _range_int(value: str, label: str, minimum: int, maximum: int) -> int:
    parsed = _integer(value, label)
    if not minimum <= parsed <= maximum:
        raise SettingsError(
            f"{label.capitalize()} must be between {minimum} and {maximum}."
        )
    return parsed


def _float(value: str, label: str) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise SettingsError(f"{label.capitalize()} must be a number.") from exc
    if not math.isfinite(parsed):
        raise SettingsError(f"{label.capitalize()} must be a finite number.")
    return parsed


def _positive_float_text(value: str, label: str) -> str:
    parsed = _float(value, label)
    if parsed <= 0:
        raise SettingsError(f"{label.capitalize()} must be greater than zero.")
    return f"{parsed:g}"


def _fourcc_text(value: str) -> str:
    text = str(value).strip().upper()
    if len(text) != 4 or not text.isalnum():
        raise SettingsError(
            "Capture pixel format must be four letters or digits, such as NV12."
        )
    return text


def _unit_float_text(value: str, label: str) -> str:
    parsed = _float(value, label)
    if not 0.0 <= parsed <= 1.0:
        raise SettingsError(f"{label.capitalize()} must be between 0 and 1.")
    return f"{parsed:g}"


def _positive_unit_float_text(value: str, label: str) -> str:
    parsed = _float(value, label)
    if not 0.0 < parsed <= 1.0:
        raise SettingsError(f"{label.capitalize()} must be greater than 0 and at most 1.")
    return f"{parsed:g}"


def _bounded_positive_float_text(value: str, label: str, maximum: float) -> str:
    parsed = _float(value, label)
    if not 0.0 < parsed <= maximum:
        raise SettingsError(
            f"{label.capitalize()} must be greater than 0 and at most {maximum:g}."
        )
    return f"{parsed:g}"


def _bounded_non_negative_float_text(value: str, label: str, maximum: float) -> str:
    parsed = _float(value, label)
    if not 0.0 <= parsed <= maximum:
        raise SettingsError(
            f"{label.capitalize()} must be between 0 and {maximum:g}."
        )
    return f"{parsed:g}"


def _choice(value: str, choices: Sequence[str], label: str) -> str:
    if value not in choices:
        allowed = ", ".join(choices)
        raise SettingsError(f"Choose a valid {label}: {allowed}.")
    return value


def _device_name(value: str) -> str:
    parsed = value.strip().upper()
    if not parsed:
        raise SettingsError("Choose an inference device.")
    # Hardware scans return ONNX Runtime's class names while the CLI exposes
    # stable short aliases. Accept old profiles and older release output too.
    return {
        "TENSORRTEXECUTIONPROVIDER": "TENSORRT",
        "CUDAEXECUTIONPROVIDER": "CUDA",
        "ROCMEXECUTIONPROVIDER": "ROCM",
        "MIGRAPHXEXECUTIONPROVIDER": "MIGRAPHX",
        "DMLEXECUTIONPROVIDER": "DIRECTML",
        "CPUEXECUTIONPROVIDER": "CPU",
        "OPENVINOEXECUTIONPROVIDER": "OPENVINO",
    }.get(parsed, parsed)


def _requires_full_gpu_provider(device: str) -> bool:
    """Return whether an explicit ONNX device promises GPU-only inference."""

    if device in {
        "GPU",
        "AMD",
        "NVIDIA",
        "CUDA",
        "TENSORRT",
        "ROCM",
        "MIGRAPHX",
        "DIRECTML",
        "DML",
    }:
        return True
    return device.startswith(("DIRECTML:", "DML:"))


def _backend_name(value: str) -> str:
    parsed = value.strip().lower()
    if not parsed:
        # An older settings file predates the choice; OpenVINO was the only
        # backend then, so that is the faithful reading of an absent value.
        return "openvino"
    if parsed not in ("openvino", "onnxruntime"):
        raise SettingsError(
            f"Unknown inference backend {value!r}; expected 'openvino' or 'onnxruntime'."
        )
    return parsed


def _same_file_text(first: str, second: str) -> bool:
    try:
        return Path(first).expanduser().resolve() == Path(second).expanduser().resolve()
    except (OSError, RuntimeError):
        return first == second


def _settings_version(value: Any) -> int:
    """Parse a stored schema version; malformed values are legacy-safe."""

    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 0
