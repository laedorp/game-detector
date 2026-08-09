"""Persistent settings and command construction for the desktop launcher.

This module deliberately has no GUI imports.  Keeping the validation and command
builder independent from Tk makes them straightforward to test in a headless
environment and safe to reuse from another front end later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


SOURCE_SCREEN = "screen"
SOURCE_CAMERA = "camera"
SOURCE_VIDEO = "video"
SOURCE_MODES = (SOURCE_SCREEN, SOURCE_CAMERA, SOURCE_VIDEO)
SELF_POSITION_LEFT = "left"
SELF_POSITION_CENTER = "center"
SELF_POSITION_RIGHT = "right"
SELF_POSITIONS = (SELF_POSITION_LEFT, SELF_POSITION_CENTER, SELF_POSITION_RIGHT)
SELF_POSITION_GEOMETRY = {
    SELF_POSITION_LEFT: ("0.18", "0.34", "0.10"),
    SELF_POSITION_CENTER: ("0.33", "0.34", "0.10"),
    SELF_POSITION_RIGHT: ("0.48", "0.34", "0.10"),
}
# Version 4 makes bundled detector choices semantic.  A settings file records
# the preset key rather than paths into a particular checkout/PyInstaller
# extraction directory, so its model and labels always move together.
SETTINGS_VERSION = 4

MODEL_PRESET_FORT_PLAYER = "fort_player"
MODEL_PRESET_COCO = "coco"
MODEL_PRESET_CUSTOM = "custom"
DEFAULT_MODEL_PRESET = MODEL_PRESET_FORT_PLAYER

# These tokens were written by settings versions 1--3.  Keep them as migration
# constants: existing profiles that used the old bundled detector must continue
# to select COCO after the Fortnite-style preset becomes the fresh default.
BUNDLED_MODEL = "@bundled/yolo26n.xml"
BUNDLED_LABELS = "@bundled/coco80.txt"


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

    @property
    def bundled(self) -> bool:
        return self.model_relative is not None and self.labels_relative is not None


MODEL_PRESETS = (
    ModelPreset(
        key=MODEL_PRESET_FORT_PLAYER,
        label="Fortnite-style players (Recommended)",
        description=(
            "One class: player. Works with Auto output and the third-person self filter."
        ),
        model_relative="models/fort_player_openvino_model/fort_player.xml",
        labels_relative="models/fort_player.txt",
    ),
    ModelPreset(
        key=MODEL_PRESET_COCO,
        label="General objects (COCO fallback)",
        description="General 80-class object detection using the included COCO model.",
        model_relative="models/yolo26n_openvino_model/yolo26n.xml",
        labels_relative="models/coco80.txt",
    ),
    ModelPreset(
        key=MODEL_PRESET_CUSTOM,
        label="Custom model files",
        description="Use your own matching OpenVINO .xml/.bin model and labels file.",
        model_relative=None,
        labels_relative=None,
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


def model_preset_paths(value: str) -> tuple[str, str]:
    """Resolve both files for a bundled preset in one operation."""

    preset = _MODEL_PRESETS_BY_KEY.get(value)
    if preset is None or not preset.bundled:
        raise ValueError(f"Model preset does not provide bundled files: {value!r}")
    assert preset.model_relative is not None
    assert preset.labels_relative is not None
    root = resource_root()
    return str(root / preset.model_relative), str(root / preset.labels_relative)


def default_model_path() -> str:
    return model_preset_paths(DEFAULT_MODEL_PRESET)[0]


def default_labels_path() -> str:
    return model_preset_paths(DEFAULT_MODEL_PRESET)[1]


def coco_model_path() -> str:
    return model_preset_paths(MODEL_PRESET_COCO)[0]


def coco_labels_path() -> str:
    return model_preset_paths(MODEL_PRESET_COCO)[1]


def settings_path() -> Path:
    """Return the per-user JSON settings path on the current platform."""

    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / "GameDetector" / "settings.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "GameDetector" / "settings.json"

    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "game-detector" / "settings.json"


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
    capture_fps: str = "60"
    screen_monitor: str = "1"
    use_screen_region: bool = False
    screen_x: str = "0"
    screen_y: str = "0"
    screen_width: str = "1920"
    screen_height: str = "1080"
    screen_fps: str = "60"
    # Empty is an initialization sentinel: no paths means the fresh bundled
    # preset, while explicitly supplied paths mean a custom model.  Every live
    # LauncherSettings instance is normalized to a semantic key in
    # ``__post_init__``.
    model_preset: str = ""
    model_path: str = ""
    labels_path: str = ""
    device: str = "CPU"
    inference_size: str = "320"
    crop_size: str = ""
    confidence: str = "0.25"
    iou_threshold: str = "0.45"
    output_format: str = "auto"
    # Filtering must be explicitly enabled in a profile because camera/video
    # users should never silently lose a foreground person detection.
    ignore_self: bool = False
    self_position: str = SELF_POSITION_LEFT
    preview: bool = True
    draw: bool = True
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
        if preset.bundled:
            # Ignore stale/partial serialized paths.  A bundled preset is an
            # atomic model+labels choice and always resolves against this run's
            # resource root.  The included YOLO26 models must use automatic
            # output decoding; a custom traditional/end2end choice is not
            # compatible state for either bundled preset.
            self.model_path, self.labels_path = model_preset_paths(preset.key)
            self.output_format = "auto"
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
            elif isinstance(default_value, str) and isinstance(value, (str, int, float)):
                converted[name] = str(value)

        version = _settings_version(values.get("version"))
        model_value = converted.get("model_path", "")
        labels_value = converted.get("labels_path", "")
        if version < SETTINGS_VERSION:
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

        model_path, labels_path = self.resolved_model_files()
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

        output_format = (
            "auto"
            if model_preset(self.model_preset).bundled
            else self.output_format.lower()
        )
        args = [
            "--model",
            str(model),
            "--labels",
            str(labels),
            "--device",
            _device_name(self.device),
            "--inference-size",
            str(_positive_int(self.inference_size, "inference size")),
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

        crop = self.crop_size.strip()
        if crop:
            args.extend(("--crop-size", str(_positive_int(crop, "crop size"))))

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
        else:
            video = _existing_file(self.video_path, "video file")
            args.extend(("--source", str(video)))

        if not self.preview:
            args.append("--no-preview")
        elif not self.draw:
            args.append("--no-draw")
        return args

    def resolved_model_files(self) -> tuple[str, str]:
        """Return the active model and labels, keeping presets atomic."""

        preset = model_preset(self.model_preset)
        if preset.bundled:
            return model_preset_paths(preset.key)
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
        helper = Path(program).with_name("GameDetectorCLI.exe")
        if helper.is_file():
            program = str(helper)
    prefix = [program]
    if not is_frozen:
        prefix.append(str(app_script or (resource_root() / "app.py")))
    return [*prefix, "--cli", *settings.detector_arguments()]


def load_settings(path: Path | None = None) -> LauncherSettings:
    target = path or settings_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return LauncherSettings()
    except (OSError, UnicodeError, json.JSONDecodeError):
        # A damaged preference file must never keep the launcher from opening.
        return LauncherSettings()
    if not isinstance(raw, Mapping):
        return LauncherSettings()
    return LauncherSettings.from_mapping(raw)


def save_settings(settings: LauncherSettings, path: Path | None = None) -> Path:
    """Atomically save settings and return the written path."""

    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
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
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    except OSError:
        try:
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


def _non_negative_int(value: str, label: str) -> int:
    parsed = _integer(value, label)
    if parsed < 0:
        raise SettingsError(f"{label.capitalize()} cannot be negative.")
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


def _unit_float_text(value: str, label: str) -> str:
    parsed = _float(value, label)
    if not 0.0 <= parsed <= 1.0:
        raise SettingsError(f"{label.capitalize()} must be between 0 and 1.")
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
