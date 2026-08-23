#!/usr/bin/env python3
"""Record bounded, capture-only player/head training sessions.

This tool deliberately has no detector, model, controller, or MAKCU dependency.
It samples the existing latest-only OpenCV capture source into a new raw session
that can be annotated and split by a later, separate dataset-preparation step.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from time import monotonic_ns
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the capture implementation directly.  In particular, this script must
# never import the application entry point or anything below aiming/detection.
from capture.base import FramePacket  # noqa: E402
from capture.opencv_source import OpenCVCaptureSource  # noqa: E402


SCHEMA_NAME = "proaim.capture_player_head.session"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "datasets" / "capture_player_head" / "raw"
)
RANGE_LABELS = ("close", "medium", "long", "negative")
MOTION_LABELS = ("moving", "stationary")
ENCODINGS = ("png", "jpeg")
MAX_DURATION_SECONDS = 3_600.0
MAX_FRAME_COUNT = 36_000
MAX_SAMPLE_FPS = 60.0
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_SCENARIO_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.()-]{0,95}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class RecorderError(RuntimeError):
    """Raised when a recording session cannot be created safely."""


class CaptureLike(Protocol):
    """The small capture-only surface used by the recorder core."""

    description: str

    @property
    def actual_settings(self) -> Mapping[str, object]: ...

    @property
    def stats(self) -> object: ...

    @property
    def ended(self) -> bool: ...

    @property
    def error(self) -> str | None: ...

    def start(self) -> None: ...

    def read(self, timeout: float | None = None) -> FramePacket | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RecorderConfig:
    """Validated bounds, labels, and capture/encoding requests for one session."""

    range_label: str
    motion: str
    scenario: str
    output_root: Path = DEFAULT_OUTPUT_ROOT
    device: int = 0
    width: int = 1_920
    height: int = 1_080
    capture_fps: float = 240.0
    pixel_format: str = "NV12"
    sample_fps: float = 10.0
    duration_seconds: float = 120.0
    max_frames: int = 1_200
    encoding: str = "png"
    png_compression: int = 3
    jpeg_quality: int = 95
    rotate_180: bool = False
    read_timeout_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.range_label not in RANGE_LABELS:
            raise ValueError(f"range_label must be one of {RANGE_LABELS}")
        if self.motion not in MOTION_LABELS:
            raise ValueError(f"motion must be one of {MOTION_LABELS}")
        if not _SCENARIO_PATTERN.fullmatch(self.scenario):
            raise ValueError(
                "scenario must be 1-96 safe characters and start with a letter or digit"
            )
        if isinstance(self.device, bool) or self.device < 0:
            raise ValueError("device must be a non-negative integer")
        for name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _bounded_finite("capture_fps", self.capture_fps, 1.0, 1_000.0)
        _bounded_finite("sample_fps", self.sample_fps, 0.1, MAX_SAMPLE_FPS)
        _bounded_finite(
            "duration_seconds", self.duration_seconds, 0.1, MAX_DURATION_SECONDS
        )
        _bounded_finite(
            "read_timeout_seconds", self.read_timeout_seconds, 0.001, 5.0
        )
        if isinstance(self.max_frames, bool) or not 1 <= self.max_frames <= MAX_FRAME_COUNT:
            raise ValueError(f"max_frames must be between 1 and {MAX_FRAME_COUNT}")
        if not re.fullmatch(r"[A-Z0-9]{4}", self.pixel_format):
            raise ValueError("pixel_format must be a four-character uppercase FOURCC")
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        if isinstance(self.png_compression, bool) or not 0 <= self.png_compression <= 9:
            raise ValueError("png_compression must be between 0 and 9")
        if isinstance(self.jpeg_quality, bool) or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not isinstance(self.rotate_180, bool):
            raise TypeError("rotate_180 must be bool")
        _validate_output_root(self.output_root)


@dataclass(frozen=True, slots=True)
class RecorderResult:
    session_dir: Path
    manifest_path: Path
    status: str
    completion_reason: str
    frame_count: int


CaptureFactory = Callable[[RecorderConfig], CaptureLike]
JpegPngEncoder = Callable[[NDArray[np.uint8], RecorderConfig], bytes]
UtcNow = Callable[[], datetime]
SessionIdFactory = Callable[[], str]


def _bounded_finite(name: str, value: float, minimum: float, maximum: float) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")


def _validate_output_root(path: Path) -> None:
    if not isinstance(path, Path):
        raise TypeError("output_root must be pathlib.Path")
    if any(part == ".." for part in path.parts):
        raise ValueError("output_root must not contain parent traversal")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("utc_now must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _generated_session_id() -> str:
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid4().hex[:12]}"


def _validate_session_id(session_id: str) -> str:
    if not _SESSION_ID_PATTERN.fullmatch(session_id) or session_id in {".", ".."}:
        raise RecorderError("unsafe generated session id")
    return session_id


def _default_capture_factory(config: RecorderConfig) -> CaptureLike:
    return OpenCVCaptureSource(
        config.device,
        width=config.width,
        height=config.height,
        fps=config.capture_fps,
        buffer_size=1,
        pixel_format=config.pixel_format,
        rotate_180=config.rotate_180,
        live=True,
    )


def _default_encoder(image: NDArray[np.uint8], config: RecorderConfig) -> bytes:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - capture itself also requires cv2.
        raise RecorderError("image encoding requires opencv-python") from exc

    if config.encoding == "png":
        extension = ".png"
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, config.png_compression]
    else:
        extension = ".jpg"
        parameters = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok or encoded is None:
        raise RecorderError(f"OpenCV failed to encode {config.encoding}")
    return bytes(encoded)


def _encoding_record(config: RecorderConfig) -> dict[str, object]:
    if config.encoding == "png":
        return {
            "format": "png",
            "extension": ".png",
            "lossless": True,
            "png_compression": config.png_compression,
            "jpeg_quality": None,
        }
    return {
        "format": "jpeg",
        "extension": ".jpg",
        "lossless": False,
        "png_compression": None,
        "jpeg_quality": config.jpeg_quality,
    }


def _validate_encoded_bytes(payload: bytes, encoding: str) -> None:
    if not payload:
        raise RecorderError("encoder returned an empty image")
    if encoding == "png" and not payload.startswith(_PNG_SIGNATURE):
        raise RecorderError("encoder bytes do not match declared PNG encoding")
    if encoding == "jpeg" and not (
        payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
    ):
        raise RecorderError("encoder bytes do not match declared JPEG encoding")


def _safe_mkdir_chain(path: Path) -> Path:
    """Create a directory chain while rejecting every symlink component."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                metadata = current.lstat()
            else:
                metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RecorderError(f"refusing symlink in output path: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise RecorderError(f"output path component is not a directory: {current}")
    return absolute


def _prepare_session(output_root: Path, session_id: str) -> tuple[Path, Path]:
    root = _safe_mkdir_chain(output_root)
    session_dir = root / _validate_session_id(session_id)
    try:
        session_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RecorderError(f"session already exists; refusing overwrite: {session_dir}") from exc
    images_dir = session_dir / "images"
    try:
        images_dir.mkdir(mode=0o700)
    except BaseException:
        # Do not remove the exclusive session directory: retaining the claimed
        # ID is safer than allowing a retry to overwrite a partially made run.
        raise
    return session_dir, images_dir


def _directory_fsync(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_file(path: Path, payload: bytes, mode: int = 0o400) -> None:
    """Publish immutable bytes atomically without ever replacing a destination."""

    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RecorderError(f"unsafe image directory: {path.parent}")
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode, follow_symlinks=False)
        # Hard-link publication has atomic visibility and fails if the final
        # name appeared concurrently; unlike replace(), it cannot overwrite.
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _directory_fsync(path.parent)
    except FileExistsError as exc:
        raise RecorderError(f"refusing to overwrite existing file: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    """Atomically create/update session.json; every visible version is valid JSON."""

    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RecorderError(f"unsafe session directory: {path.parent}")
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RecorderError(f"refusing unsafe manifest target: {path}")
    payload = (
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.parent / f".session.json.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600, follow_symlinks=False)
        _directory_fsync(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _source_property(source: CaptureLike, name: str, fallback: object) -> object:
    try:
        return getattr(source, name)
    except BaseException:
        return fallback


def _validate_packet(packet: FramePacket) -> NDArray[np.uint8]:
    if isinstance(packet.sequence, bool) or packet.sequence < 0:
        raise RecorderError("capture returned an invalid source sequence")
    if packet.read_started_ns < 0 or packet.read_completed_ns < packet.read_started_ns:
        raise RecorderError("capture returned invalid source timestamps")
    image = packet.image
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise RecorderError("capture frame must be a uint8 numpy array")
    if image.ndim != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise RecorderError("capture frame must be a non-empty HxWxC image")
    if image.shape[2] != 3:
        raise RecorderError("capture frame must contain exactly three BGR channels")
    return image


def _seal_session(session_dir: Path, images_dir: Path, manifest_path: Path) -> None:
    """Make a finalized raw session read-only against accidental mutation."""

    os.chmod(manifest_path, 0o400, follow_symlinks=False)
    os.chmod(images_dir, 0o500, follow_symlinks=False)
    os.chmod(session_dir, 0o500, follow_symlinks=False)
    _directory_fsync(session_dir.parent)


def record_session(
    config: RecorderConfig,
    *,
    capture_factory: CaptureFactory = _default_capture_factory,
    encoder: JpegPngEncoder = _default_encoder,
    clock_ns: Callable[[], int] = monotonic_ns,
    utc_now: UtcNow = _utc_now,
    session_id_factory: SessionIdFactory = _generated_session_id,
    should_stop: Callable[[], bool] = lambda: False,
) -> RecorderResult:
    """Record one exclusive raw session using injected, deterministic edges."""

    session_id = _validate_session_id(session_id_factory())
    session_dir, images_dir = _prepare_session(config.output_root, session_id)
    manifest_path = session_dir / "session.json"
    created_utc = _iso_utc(utc_now())
    encoding = _encoding_record(config)
    manifest: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "status": "recording",
        "complete": False,
        "completion_reason": None,
        "created_utc": created_utc,
        "ended_utc": None,
        "range": config.range_label,
        "motion": config.motion,
        "scenario": config.scenario,
        "limits": {
            "duration_seconds": config.duration_seconds,
            "max_frames": config.max_frames,
            "sample_fps": config.sample_fps,
        },
        "encoding": encoding,
        "capture": {
            "description": None,
            "requested": {
                "device": config.device,
                "width": config.width,
                "height": config.height,
                "fps": config.capture_fps,
                "pixel_format": config.pixel_format,
                "buffer_size": 1,
                "rotation_degrees": 180 if config.rotate_180 else 0,
                "latest_only": True,
            },
            "actual": {},
            "stats": {},
            "error": None,
        },
        "frames": [],
        "skipped_stale_packets": 0,
    }
    _atomic_manifest(manifest_path, manifest)

    source: CaptureLike | None = None
    status = "failed"
    reason = "initialization-failed"
    pending_error: BaseException | None = None
    sample_interval_ns = max(1, round(1_000_000_000 / config.sample_fps))
    deadline_ns = 0
    next_sample_source_ns: int | None = None
    last_sequence = -1
    last_source_ns = -1

    try:
        started_ns = clock_ns()
        if isinstance(started_ns, bool) or not isinstance(started_ns, int) or started_ns < 0:
            raise RecorderError("clock_ns must return a non-negative integer")
        deadline_ns = started_ns + round(config.duration_seconds * 1_000_000_000)
        source = capture_factory(config)
        manifest["capture"]["description"] = str(source.description)
        source.start()
        manifest["capture"]["actual"] = _json_safe(source.actual_settings)
        _atomic_manifest(manifest_path, manifest)

        while len(manifest["frames"]) < config.max_frames:
            if should_stop():
                status = "interrupted"
                reason = "stop-requested"
                break
            now_ns = clock_ns()
            if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
                raise RecorderError("clock_ns must return a non-negative integer")
            remaining_ns = deadline_ns - now_ns
            if remaining_ns <= 0:
                if manifest["frames"]:
                    status = "complete"
                    reason = "duration-limit"
                else:
                    status = "failed"
                    reason = "duration-limit-no-frames"
                break
            timeout = min(config.read_timeout_seconds, remaining_ns / 1_000_000_000)
            packet = source.read(timeout=timeout)
            if packet is None:
                if bool(_source_property(source, "ended", False)):
                    status = "failed"
                    reason = (
                        "capture-error"
                        if _source_property(source, "error", None)
                        else "source-ended"
                    )
                    break
                continue

            image = _validate_packet(packet)
            if packet.sequence <= last_sequence or packet.read_completed_ns < last_source_ns:
                manifest["skipped_stale_packets"] += 1
                continue
            last_sequence = packet.sequence
            last_source_ns = packet.read_completed_ns
            if (
                next_sample_source_ns is not None
                and packet.read_completed_ns < next_sample_source_ns
            ):
                continue

            frame_number = len(manifest["frames"])
            extension = str(encoding["extension"])
            frame_id = f"frame-{frame_number:06d}"
            file_name = (
                f"{frame_id}-source-{packet.sequence:010d}{extension}"
            )
            relative_name = f"images/{file_name}"
            encoded = encoder(image, config)
            if not isinstance(encoded, bytes):
                raise RecorderError("encoder must return bytes")
            _validate_encoded_bytes(encoded, config.encoding)
            _exclusive_file(images_dir / file_name, encoded)
            saved_ns = clock_ns()
            if isinstance(saved_ns, bool) or not isinstance(saved_ns, int) or saved_ns < 0:
                raise RecorderError("clock_ns must return a non-negative integer")
            manifest["frames"].append(
                {
                    "id": frame_id,
                    "file_name": relative_name,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "channels": int(image.shape[2]),
                    "encoding": config.encoding,
                    "byte_size": len(encoded),
                    "sha256": sha256(encoded).hexdigest(),
                    "source_sequence": int(packet.sequence),
                    "source_read_started_ns": int(packet.read_started_ns),
                    "source_read_completed_ns": int(packet.read_completed_ns),
                    "saved_monotonic_ns": int(saved_ns),
                    "saved_utc": _iso_utc(utc_now()),
                }
            )
            _atomic_manifest(manifest_path, manifest)
            if next_sample_source_ns is None:
                next_sample_source_ns = packet.read_completed_ns + sample_interval_ns
            else:
                intervals = (
                    (packet.read_completed_ns - next_sample_source_ns)
                    // sample_interval_ns
                    + 1
                )
                next_sample_source_ns += intervals * sample_interval_ns

        else:
            status = "complete"
            reason = "frame-limit"
    except KeyboardInterrupt:
        status = "interrupted"
        reason = "keyboard-interrupt"
    except BaseException as exc:
        status = "failed"
        reason = f"{type(exc).__name__}: {exc}"
        pending_error = exc
    finally:
        if source is not None:
            try:
                source.close()
            except BaseException as exc:
                if pending_error is None and status != "interrupted":
                    status = "failed"
                    reason = f"capture-close-{type(exc).__name__}: {exc}"
                    pending_error = exc
            manifest["capture"]["actual"] = _json_safe(
                _source_property(source, "actual_settings", {})
            )
            manifest["capture"]["stats"] = _json_safe(
                _source_property(source, "stats", {})
            )
            capture_error = _source_property(source, "error", None)
            manifest["capture"]["error"] = _json_safe(capture_error)
            # A live backend can return normally from close() while reporting a
            # close timeout through its error property.  Never seal such a run
            # as complete: the producer may still be alive and its teardown is
            # not trustworthy even though all requested images were published.
            if (
                status == "complete"
                and capture_error is not None
                and str(capture_error).strip()
            ):
                status = "failed"
                reason = "capture-error"
        manifest["status"] = status
        manifest["complete"] = status == "complete"
        manifest["completion_reason"] = reason
        manifest["ended_utc"] = _iso_utc(utc_now())
        try:
            _atomic_manifest(manifest_path, manifest)
            _seal_session(session_dir, images_dir, manifest_path)
        except BaseException as finalization_error:
            if pending_error is None:
                pending_error = finalization_error

    if pending_error is not None:
        raise pending_error
    return RecorderResult(
        session_dir=session_dir,
        manifest_path=manifest_path,
        status=status,
        completion_reason=reason,
        frame_count=len(manifest["frames"]),
    )


def build_parser() -> argparse.ArgumentParser:
    try:
        parser = argparse.ArgumentParser(description=__doc__, color=False)
    except TypeError:
        parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", dest="range_label", choices=RANGE_LABELS, required=True)
    parser.add_argument("--motion", choices=MOTION_LABELS, required=True)
    parser.add_argument(
        "--scenario",
        required=True,
        help="Short safe description such as friend-strafe-left-right",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="PATH",
        help=(
            "Directory that receives immutable raw sessions; lossless PNG sessions "
            "can be large (default: %(default)s)"
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1_920)
    parser.add_argument("--height", type=int, default=1_080)
    parser.add_argument("--capture-fps", type=float, default=240.0)
    parser.add_argument("--pixel-format", default="NV12")
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--max-frames", type=int, default=1_200)
    parser.add_argument(
        "--encoding",
        choices=ENCODINGS,
        default="png",
        help="PNG is the lossless default; JPEG must be selected explicitly",
    )
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--rotate-180", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> RecorderConfig:
    return RecorderConfig(
        range_label=args.range_label,
        motion=args.motion,
        scenario=args.scenario,
        output_root=args.output_root,
        device=args.device,
        width=args.width,
        height=args.height,
        capture_fps=args.capture_fps,
        pixel_format=str(args.pixel_format).upper(),
        sample_fps=args.sample_fps,
        duration_seconds=args.duration_seconds,
        max_frames=args.max_frames,
        encoding=args.encoding,
        png_compression=args.png_compression,
        jpeg_quality=args.jpeg_quality,
        rotate_180=args.rotate_180,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        result = record_session(config)
    except (ValueError, TypeError, RecorderError) as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"{result.status}: recorded {result.frame_count} images to {result.session_dir}\n"
        f"Manifest: {result.manifest_path}"
    )
    if result.status == "interrupted":
        return 130
    if result.status != "complete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
