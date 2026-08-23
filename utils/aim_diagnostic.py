"""Bounded asynchronous frame and decision traces for automatic aiming."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from threading import Condition, Thread
from typing import Any
from uuid import uuid4

import numpy as np


SCHEMA_NAME = "proaim.aim_diagnostic.session"
SCHEMA_VERSION = 1
DEFAULT_SAMPLE_HZ = 20.0
DEFAULT_MAX_DURATION_SECONDS = 30.0
DEFAULT_JPEG_QUALITY = 85


@dataclass(frozen=True, slots=True)
class AimDiagnosticConfig:
    output_root: Path
    sample_hz: float = DEFAULT_SAMPLE_HZ
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS
    wait_for_activation: bool = False
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    max_pending_records: int = 512
    stop_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
        if any(part == ".." for part in self.output_root.parts):
            raise ValueError("output_root must not contain parent traversal")
        for name, value in (
            ("sample_hz", self.sample_hz),
            ("max_duration_seconds", self.max_duration_seconds),
            ("stop_timeout_seconds", self.stop_timeout_seconds),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.sample_hz > 60.0:
            raise ValueError("sample_hz must not exceed 60")
        if self.max_duration_seconds > 300.0:
            raise ValueError("max_duration_seconds must not exceed 300")
        if not isinstance(self.wait_for_activation, bool):
            raise TypeError("wait_for_activation must be bool")
        if (
            isinstance(self.jpeg_quality, bool)
            or not isinstance(self.jpeg_quality, int)
            or not 1 <= self.jpeg_quality <= 100
        ):
            raise ValueError("jpeg_quality must be an integer from 1 to 100")
        if (
            isinstance(self.max_pending_records, bool)
            or not isinstance(self.max_pending_records, int)
            or not 1 <= self.max_pending_records <= 10_000
        ):
            raise ValueError("max_pending_records must be between 1 and 10000")


@dataclass(frozen=True, slots=True)
class AimDiagnosticStatus:
    submitted: int
    written: int
    arming_skips: int
    sample_gate_skips: int
    duration_limit_skips: int
    pending_overwrites: int
    failures: int
    error: str | None


@dataclass(frozen=True, slots=True)
class _PendingSample:
    image: np.ndarray | None
    record: dict[str, Any]


ImageEncoder = Callable[[np.ndarray, int], bytes]


def _default_image_encoder(image: np.ndarray, quality: int) -> bytes:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("diagnostic frame encoding requires OpenCV") from exc
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise RuntimeError("OpenCV failed to encode diagnostic frame")
    return bytes(encoded)


def _json_copy(value: Mapping[str, object]) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False)
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("diagnostic record must contain finite JSON data") from exc
    if not isinstance(copied, dict):  # pragma: no cover - Mapping always object
        raise TypeError("diagnostic record must be an object")
    return copied


class AimDiagnosticRecorder:
    """Write every decision through a bounded queue and sample frame images."""

    def __init__(
        self,
        config: AimDiagnosticConfig,
        *,
        metadata: Mapping[str, object] | None = None,
        image_encoder: ImageEncoder = _default_image_encoder,
    ) -> None:
        if not isinstance(config, AimDiagnosticConfig):
            raise TypeError("config must be AimDiagnosticConfig")
        if not callable(image_encoder):
            raise TypeError("image_encoder must be callable")
        self.config = config
        self.metadata = _json_copy(metadata or {})
        self._image_encoder = image_encoder
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.session_dir = config.output_root / f"{timestamp}-{uuid4().hex[:12]}"
        self.frames_dir = self.session_dir / "frames"
        self.records_path = self.session_dir / "records.jsonl"
        self.manifest_path = self.session_dir / "manifest.json"

        self._condition = Condition()
        self._thread: Thread | None = None
        self._pending: deque[_PendingSample] = deque()
        self._started = False
        self._stop_requested = False
        self._first_source_ns: int | None = None
        self._next_sample_ns: int | None = None
        self._sample_interval_ns = max(
            1,
            round(1_000_000_000 / config.sample_hz),
        )
        self._maximum_duration_ns = max(
            1,
            round(config.max_duration_seconds * 1_000_000_000),
        )
        self._submitted = 0
        self._written = 0
        self._arming_skips = 0
        self._sample_gate_skips = 0
        self._duration_limit_skips = 0
        self._pending_overwrites = 0
        self._failures = 0
        self._error: RuntimeError | None = None

    def start(self) -> None:
        with self._condition:
            if self._started:
                raise RuntimeError("diagnostic recorder cannot be restarted")
            self.config.output_root.mkdir(parents=True, exist_ok=True)
            self.session_dir.mkdir(mode=0o700)
            self.frames_dir.mkdir(mode=0o700)
            self.records_path.touch(mode=0o600)
            self._started = True
            self._thread = Thread(
                target=self._run,
                name="proaim-aim-diagnostic",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        image: np.ndarray,
        record: Mapping[str, object],
    ) -> bool:
        if not isinstance(image, np.ndarray):
            raise TypeError("diagnostic image must be a numpy array")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("diagnostic image must be a uint8 HxWx3 array")
        owned_record = _json_copy(record)
        timestamp = owned_record.get("source_timestamp_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise TypeError("source_timestamp_ns must be an integer")
        if timestamp < 0:
            raise ValueError("source_timestamp_ns cannot be negative")

        with self._condition:
            if not self._started or self._stop_requested or self._error is not None:
                return False
            if self._first_source_ns is None:
                if self.config.wait_for_activation and not (
                    owned_record.get("raw_activation_pressed") is True
                    or owned_record.get("activation_pressed") is True
                ):
                    self._arming_skips += 1
                    return False
                self._first_source_ns = timestamp
                self._next_sample_ns = timestamp
            assert self._next_sample_ns is not None
            if timestamp - self._first_source_ns > self._maximum_duration_ns:
                self._duration_limit_skips += 1
                return False
            capture_image = timestamp >= self._next_sample_ns
            if not capture_image:
                self._sample_gate_skips += 1
            else:
                elapsed_intervals = (
                    max(0, timestamp - self._next_sample_ns)
                    // self._sample_interval_ns
                ) + 1
                self._next_sample_ns += (
                    elapsed_intervals * self._sample_interval_ns
                )
            self._submitted += 1

        owned_image = np.ascontiguousarray(image).copy() if capture_image else None
        with self._condition:
            if self._stop_requested or self._error is not None:
                return False
            if len(self._pending) >= self.config.max_pending_records:
                self._pending.popleft()
                self._pending_overwrites += 1
            self._pending.append(_PendingSample(owned_image, owned_record))
            self._condition.notify_all()
        return True

    @property
    def status(self) -> AimDiagnosticStatus:
        with self._condition:
            return AimDiagnosticStatus(
                submitted=self._submitted,
                written=self._written,
                arming_skips=self._arming_skips,
                sample_gate_skips=self._sample_gate_skips,
                duration_limit_skips=self._duration_limit_skips,
                pending_overwrites=self._pending_overwrites,
                failures=self._failures,
                error=str(self._error) if self._error is not None else None,
            )

    def stop(self) -> bool:
        with self._condition:
            if not self._started:
                return True
            self._stop_requested = True
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(self.config.stop_timeout_seconds)
        stopped = thread is None or not thread.is_alive()
        self._write_manifest(stopped=stopped)
        return bool(stopped and self._error is None)

    def _run(self) -> None:
        try:
            with self.records_path.open("a", encoding="utf-8", buffering=1) as stream:
                while True:
                    with self._condition:
                        while not self._pending and not self._stop_requested:
                            self._condition.wait()
                        if not self._pending and self._stop_requested:
                            return
                        pending = self._pending.popleft()
                    sequence = pending.record.get("frame_sequence")
                    if isinstance(sequence, bool) or not isinstance(sequence, int):
                        raise TypeError("frame_sequence must be an integer")
                    complete_record = dict(pending.record)
                    if pending.image is not None:
                        payload = self._image_encoder(
                            pending.image,
                            self.config.jpeg_quality,
                        )
                        if not isinstance(payload, bytes) or not payload:
                            raise RuntimeError(
                                "diagnostic image encoder returned no bytes"
                            )
                        image_name = f"frame-{sequence:010d}.jpg"
                        image_path = self.frames_dir / image_name
                        with image_path.open("xb") as image_file:
                            image_file.write(payload)
                            image_file.flush()
                            os.fsync(image_file.fileno())
                        complete_record.update(
                            {
                                "image_path": f"frames/{image_name}",
                                "image_sha256": sha256(payload).hexdigest(),
                                "image_size_bytes": len(payload),
                            }
                        )
                    stream.write(
                        json.dumps(
                            complete_record,
                            ensure_ascii=True,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    with self._condition:
                        self._written += 1
        except BaseException as exc:
            with self._condition:
                self._failures += 1
                self._error = RuntimeError(
                    f"aim diagnostic recorder failed: {type(exc).__name__}: {exc}"
                )
                self._pending.clear()
                self._stop_requested = True
                self._condition.notify_all()

    def _write_manifest(self, *, stopped: bool) -> None:
        status = self.status
        manifest = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "complete": bool(stopped and status.error is None),
            "metadata": self.metadata,
            "configuration": {
                "sample_hz": self.config.sample_hz,
                "max_duration_seconds": self.config.max_duration_seconds,
                "wait_for_activation": self.config.wait_for_activation,
                "jpeg_quality": self.config.jpeg_quality,
                "max_pending_records": self.config.max_pending_records,
            },
            "statistics": asdict(status),
            "records_path": self.records_path.name,
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)
