"""Detection-driven relative mouse output for a MAKCU passthrough board."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import errno
from hashlib import sha256
import os
from pathlib import Path
import math
import threading
import time
from typing import Any

from detection.types import Detection
from .controller import DEFAULT_HEAD_RATIO, head_target_point
from .makcu_calibrated_control import (
    CalibratedControlOutput,
    EmittedMouseCommand,
    MakcuCalibratedController,
    ScreenErrorObservation,
)

try:
    import serial as _serial
    from serial.tools import list_ports as _list_ports
except ImportError as exc:  # pragma: no cover - depends on optional runtime package
    _serial = None  # type: ignore[assignment]
    _list_ports = None  # type: ignore[assignment]
    SERIAL_IMPORT_ERROR: BaseException | None = exc
else:
    SERIAL_IMPORT_ERROR = None


MAKCU_VENDOR_ID = 0x1A86
MAKCU_PRODUCT_ID = 0x55D3
MAKCU_BAUD_RATES = (4_000_000, 2_000_000, 115_200)
MAKCU_FAST_BAUD = 4_000_000
MAKCU_BAUD_CHANGE = bytes((0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00))
BUTTON_NAMES = ("Left", "Right", "Middle", "Side 4", "Side 5")
DEFAULT_OUTPUT_HZ = 1000
REFERENCE_CONTROL_HZ = 60.0
# Aim tuning is expressed in 1080p-reference pixels.  Scaling visual error to
# this coordinate space keeps identical normalized offsets equivalent when a
# Moonlight stream or capture backend negotiates 720p, 1080p, or 4K.
REFERENCE_FRAME_WIDTH = 1920.0
REFERENCE_FRAME_HEIGHT = 1080.0
TARGET_STALE_SECONDS = 0.15
# MAKCU button telemetry is edge-driven: the board sends a frame when the
# physical state changes, not as a heartbeat.  The one-argument command is
# supported by both older field firmware and the current protocol.
BUTTON_STREAM_COMMAND = "km.buttons(1)"
# The button stream is event-driven, so silence normally means "still held".
# Bound that authority anyway: after an unusually long uninterrupted hold the
# user must release and press again. This prevents a lost release byte from
# authorizing movement indefinitely.
MAX_CONTINUOUS_ACTIVATION_SECONDS = 10.0
THREADED_RATE_SMOOTHING_ALPHA = 0.82
PREDICTION_LEAD_SECONDS = 0.028
DERIVATIVE_DAMPING_SECONDS = 0.016
MAX_SAMPLE_AGE_LEAD_SECONDS = 0.055
MAX_TRACKED_VELOCITY_PX_PER_SEC = 2600.0
ERROR_JUMP_RESET_PIXELS = 240.0
# Persistent screen-space error is unavoidable with a purely proportional
# pursuit controller: a moving target must stay behind the crosshair to keep
# generating a mouse rate. Build a bounded learned rate over this time instead.
# It is cleared on every loss of physical authorization or target continuity,
# so it cannot outlive the evidence that created it.
PURSUIT_INTEGRAL_TIME_SECONDS = 0.12
MAX_PURSUIT_INTEGRAL_RATIO = 0.50
# A moving target requires a nonzero mouse rate even while its measured head is
# centered.  Keep the learned pursuit rate through that zero-error interval and
# let it fade gradually only while fresh detector measurements remain inside
# the deadzone.  This is deliberately much slower than the detector cadence so
# one centered sample cannot erase the rate which brought the target there.
PURSUIT_DEADZONE_LEAK_TIME_SECONDS = 0.35
MAX_VERTICAL_RATE_RATIO = 0.48
# Calibration deliberately has a much smaller envelope than normal aiming.
# The detector renews a short physical-evidence lease while a bounded pulse is
# in flight; the 1 kHz worker independently checks that lease and the selected
# mouse button before every command.
CALIBRATION_OUTPUT_HZ = 1000
CALIBRATION_LEASE_MAX_AGE_SECONDS = 0.025
CALIBRATION_MAX_EXCURSION_COUNTS = 200
CALIBRATION_MAX_SESSION_ABS_COUNTS = 2400
CALIBRATION_MAX_RATE_COUNTS_PER_SECOND = 2400.0
CALIBRATION_MAX_STEP_COUNTS = math.ceil(
    CALIBRATION_MAX_RATE_COUNTS_PER_SECOND / CALIBRATION_OUTPUT_HZ
)
# ``stop()`` owns one end-to-end deadline.  The output worker normally exits in
# a few milliseconds because serial reads and writes use 50 ms timeouts, but a
# broken USB driver is allowed only this long before shutdown raises and leaves
# the still-live worker/connection visible for diagnostics.  Daemon breaker
# threads keep even a pathological native ``close()`` call inside this bound.
MAKCU_STOP_TIMEOUT_SECONDS = 0.75
MAKCU_STOP_PHASE_GRACE_SECONDS = 0.10


class MakcuError(RuntimeError):
    """User-facing MAKCU discovery, connection, or command failure."""


_BUTTON_FRAME_PREFIX = b"km."
_BUTTON_LONG_SUFFIX = b"buttons"


class _ButtonStreamParser:
    """Decode framed and legacy MAKCU button-state events across reads."""

    def __init__(self) -> None:
        self._pending = bytearray()
        self._framed_mode = False

    def reset(self) -> None:
        self._pending.clear()
        self._framed_mode = False

    @staticmethod
    def _prefix_tail_length(data: bytes | bytearray) -> int:
        """Return the longest suffix which can begin the next ``km.`` frame."""

        maximum = min(len(data), len(_BUTTON_FRAME_PREFIX) - 1)
        for length in range(maximum, 0, -1):
            if data[-length:] == _BUTTON_FRAME_PREFIX[:length]:
                return length
        return 0

    def feed(self, data: bytes) -> tuple[tuple[int, bool], ...]:
        bare_mask = (
            not self._framed_mode
            and not self._pending
            and len(data) == 1
            and data[0] <= 0x1F
        )
        if data:
            self._pending.extend(data)
        reports: list[tuple[int, bool]] = []
        while self._pending:
            index = self._pending.find(_BUTTON_FRAME_PREFIX)
            if index < 0:
                # Field firmware can emit a naked one-byte five-bit mask. Only
                # accept a standalone byte received before any framed traffic.
                # Once a framed stream is observed, split CR/LF/prompt bytes
                # must never be reclassified as physical mouse state.
                if bare_mask and len(self._pending) == 1:
                    reports.append((self._pending[0], False))
                    self._pending.clear()
                    break
                tail_length = self._prefix_tail_length(self._pending)
                if tail_length:
                    del self._pending[:-tail_length]
                else:
                    self._pending.clear()
                break

            if index:
                # Prefix-adjacent bytes are command ACK/prompt noise, not a
                # naked legacy event.
                del self._pending[:index]
            self._framed_mode = True

            after_prefix = bytes(self._pending[len(_BUTTON_FRAME_PREFIX) :])
            if not after_prefix:
                break

            # ``km.`` is itself a valid short frame prefix, but it is also the
            # beginning of ``km.buttons``. Wait across arbitrary serial-read
            # boundaries while the available suffix can still become the long
            # form; otherwise a split such as ``km.but`` would lose the frame.
            if _BUTTON_LONG_SUFFIX.startswith(after_prefix):
                break

            if after_prefix.startswith(_BUTTON_LONG_SUFFIX):
                value_index = len(_BUTTON_FRAME_PREFIX) + len(_BUTTON_LONG_SUFFIX)
            else:
                value_index = len(_BUTTON_FRAME_PREFIX)

            if value_index >= len(self._pending):
                break
            value = self._pending[value_index]
            if value <= 0x1F:
                reports.append((value, True))
                del self._pending[: value_index + 1]
                continue

            # Textual command response (for example ``km.buttons(1)``) or an
            # unrelated ``km.*`` message. Drop this prefix and search for the
            # next structured event without interpreting its control bytes.
            del self._pending[: len(_BUTTON_FRAME_PREFIX)]

        if len(self._pending) > 256:
            # A malformed/noisy device must not grow this buffer forever.
            del self._pending[:-32]
        return tuple(reports)


def _canonical_port(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _stable_linux_alias(path: str) -> str:
    if os.name != "posix":
        return path
    selected_key = _canonical_port(path)
    by_id = Path("/dev/serial/by-id")
    if by_id.is_dir():
        for candidate in sorted(by_id.iterdir()):
            if _canonical_port(str(candidate)) == selected_key:
                return str(candidate)
    return path


def _makcu_identity_token(path: str) -> str:
    """Return a stable, non-reversible binding token for the connected board path."""

    stable_path = _stable_linux_alias(path)
    normalized = os.path.normcase(os.path.normpath(stable_path))
    material = f"proaim-makcu-identity-v1\0{normalized}".encode(
        "utf-8", errors="strict"
    )
    return sha256(material).hexdigest()


def _likely_makcu_path(port: Any) -> bool:
    description = " ".join(
        str(getattr(port, field, "") or "")
        for field in ("device", "description", "hwid", "product")
    ).casefold()
    return any(
        marker in description
        for marker in ("ch343", "usb single serial", "usb-enhanced-serial")
    )


def detect_makcu_port(
    *,
    requested: str = "",
    ports_provider: Callable[[], Iterable[Any]] | None = None,
) -> str:
    """Return one MAKCU serial path or raise a user-facing discovery error."""

    provider = ports_provider
    use_system_candidates = provider is None
    if provider is None:
        if _list_ports is None:
            detail = f": {SERIAL_IMPORT_ERROR}" if SERIAL_IMPORT_ERROR else ""
            raise MakcuError("MAKCU discovery requires the 'pyserial' package" + detail)
        provider = lambda: _list_ports.comports(include_links=True)

    available = tuple(provider())
    selected = requested.strip()
    if selected:
        if selected.startswith("/dev/") and not Path(selected).exists():
            raise MakcuError(f"MAKCU serial device not found: {selected}")
        selected_key = _canonical_port(selected)
        matching = [
            port
            for port in available
            if _canonical_port(str(getattr(port, "device", ""))) == selected_key
        ]
        known_ids = {
            (getattr(port, "vid", None), getattr(port, "pid", None))
            for port in matching
            if getattr(port, "vid", None) is not None
            or getattr(port, "pid", None) is not None
        }
        if known_ids and (MAKCU_VENDOR_ID, MAKCU_PRODUCT_ID) not in known_ids:
            raise MakcuError(
                f"Selected serial device is not the expected MAKCU 1a86:55d3: {selected}"
            )
        # Some CH343 drivers expose a usable COM/tty path without attaching USB
        # metadata to pyserial's record (and pyserial 3.5 does not glob the
        # official Linux /dev/ttyCH343USB* name).  Explicit choices are allowed
        # through here; controller startup still requires a MAKCU firmware
        # response before the path is trusted or any output is enabled.
        return selected

    exact_matches = [
        str(port.device)
        for port in available
        if getattr(port, "vid", None) == MAKCU_VENDOR_ID
        and getattr(port, "pid", None) == MAKCU_PRODUCT_ID
    ]
    candidate_paths = list(exact_matches)
    if not candidate_paths:
        candidate_paths.extend(
            str(port.device) for port in available if _likely_makcu_path(port)
        )
    if use_system_candidates and os.name == "posix":
        candidate_paths.extend(
            str(path) for path in sorted(Path("/dev").glob("ttyCH343USB*"))
        )

    unique: dict[str, str] = {}
    for path in candidate_paths:
        if path:
            display_path = _stable_linux_alias(path) if use_system_candidates else path
            unique.setdefault(_canonical_port(path), display_path)
    matches = tuple(unique.values())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise MakcuError(
            "MAKCU 1a86:55d3 serial device was not found. Connect the board's "
            "USB2 control port and confirm the CH343 COM/serial driver is loaded."
        )
    raise MakcuError("More than one MAKCU device was found; choose its serial path")


@dataclass(frozen=True, slots=True)
class MakcuAimConfig:
    port: str = ""
    activation_button: int = 1
    strength: float = 0.50
    max_step: int = 160
    deadzone_pixels: float = 2.0
    invert_x: bool = False
    invert_y: bool = False
    head_ratio: float = DEFAULT_HEAD_RATIO
    output_hz: int = DEFAULT_OUTPUT_HZ
    smoothing_alpha: float = THREADED_RATE_SMOOTHING_ALPHA
    prediction_lead_seconds: float = PREDICTION_LEAD_SECONDS
    derivative_damping_seconds: float = DERIVATIVE_DAMPING_SECONDS
    max_sample_age_lead_seconds: float = MAX_SAMPLE_AGE_LEAD_SECONDS
    vertical_rate_ratio: float = MAX_VERTICAL_RATE_RATIO

    def __post_init__(self) -> None:
        if not 0 <= self.activation_button < len(BUTTON_NAMES):
            raise ValueError("MAKCU activation button must be between 0 and 4")
        if not math.isfinite(self.strength) or not 0.0 < self.strength <= 4.0:
            raise ValueError("MAKCU aim strength must be greater than 0 and at most 4")
        if self.max_step <= 0:
            raise ValueError("MAKCU maximum step must be greater than zero")
        if not math.isfinite(self.deadzone_pixels) or self.deadzone_pixels < 0.0:
            raise ValueError("MAKCU deadzone must be finite and non-negative")
        if not math.isfinite(self.head_ratio) or not 0.0 <= self.head_ratio <= 0.5:
            raise ValueError("MAKCU head ratio must be between 0 and 0.5")
        if not 100 <= self.output_hz <= 2000:
            raise ValueError("MAKCU output rate must be between 100 and 2000 Hz")
        if not math.isfinite(self.smoothing_alpha) or not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("MAKCU smoothing alpha must be greater than 0 and at most 1")
        if (
            not math.isfinite(self.prediction_lead_seconds)
            or self.prediction_lead_seconds < 0.0
            or self.prediction_lead_seconds > 0.25
        ):
            raise ValueError("MAKCU prediction lead must be between 0 and 0.25 seconds")
        if (
            not math.isfinite(self.derivative_damping_seconds)
            or self.derivative_damping_seconds < 0.0
            or self.derivative_damping_seconds > 0.25
        ):
            raise ValueError("MAKCU derivative damping must be between 0 and 0.25 seconds")
        if (
            not math.isfinite(self.max_sample_age_lead_seconds)
            or self.max_sample_age_lead_seconds < 0.0
            or self.max_sample_age_lead_seconds > 0.25
        ):
            raise ValueError("MAKCU sample-age lead cap must be between 0 and 0.25 seconds")
        if (
            isinstance(self.vertical_rate_ratio, bool)
            or not isinstance(self.vertical_rate_ratio, (int, float))
            or not math.isfinite(self.vertical_rate_ratio)
            or self.vertical_rate_ratio <= 0.0
            or self.vertical_rate_ratio > 1.0
        ):
            raise ValueError("MAKCU vertical rate ratio must be greater than 0 and at most 1")


@dataclass(frozen=True, slots=True)
class MakcuTelemetrySnapshot:
    """Monotonic counters describing commands written by the output loop.

    These counters observe the existing control path; collecting a snapshot
    never polls the board or sends a serial command. Signed totals expose net
    direction while absolute totals remain meaningful when motion reverses.
    """

    output_ticks: int = 0
    active_input_ticks: int = 0
    button_pressed_ticks: int = 0
    target_present_ticks: int = 0
    fresh_target_ticks: int = 0
    authorized_ticks: int = 0
    movement_commands: int = 0
    emitted_x: int = 0
    emitted_y: int = 0
    emitted_abs_x: int = 0
    emitted_abs_y: int = 0
    control_samples: int = 0
    control_error_x: float = 0.0
    control_error_y: float = 0.0
    control_error_abs_x: float = 0.0
    control_error_abs_y: float = 0.0
    pursuit_x: float = 0.0
    pursuit_y: float = 0.0
    pursuit_abs_x: float = 0.0
    pursuit_abs_y: float = 0.0
    saturated_x_samples: int = 0
    saturated_y_samples: int = 0
    pursuit_resets: int = 0


@dataclass(frozen=True, slots=True)
class MakcuCalibrationSnapshot:
    """Actual, successfully written motion for one exclusive calibration session."""

    active: bool = False
    # This controller-clock ceiling is captured under the same state lock as
    # the event tuple. It prevents a detector-thread timestamp taken just before
    # the snapshot from falsely classifying a concurrent 1 kHz write as future.
    # It is operational metadata, not part of snapshot value identity.
    captured_ns: int = field(default=0, compare=False)
    emitted_x: int = 0
    emitted_y: int = 0
    emitted_abs_counts: int = 0
    movement_commands: int = 0
    first_emitted_ns: int | None = None
    last_emitted_ns: int | None = None
    emitted_events: tuple[tuple[int, int, int], ...] = ()
    pending_axis: str | None = None
    pending_counts: int = 0
    pending_rate_counts_per_second: float = 0.0
    abort_reason: str | None = None


@dataclass(frozen=True, slots=True)
class MakcuRawActivationSnapshot:
    """Immutable raw button evidence for one exclusive calibration epoch.

    The serial worker can observe a complete release/repress cycle between two
    detector callbacks. Keeping its transition timestamps lets calibration
    prove the released dwell instead of sampling only the final button level.
    """

    captured_ns: int
    calibration_epoch: int
    calibration_entered_ns: int
    calibration_entry_report_sequence: int
    calibration_entry_framed_report_sequence: int
    calibration_entry_transition_sequence: int
    active: bool
    known: bool
    pressed: bool
    report_sequence: int
    framed_report_sequence: int
    transition_sequence: int
    last_report_framed: bool | None = None
    post_entry_press_seen: bool = False
    continuous_state_since_ns: int | None = None
    release_started_ns: int | None = None
    release_started_report_sequence: int | None = None
    completed_release_started_ns: int | None = None
    completed_release_report_sequence: int | None = None
    completed_press_ns: int | None = None
    completed_press_report_sequence: int | None = None
    completed_press_transition_sequence: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "captured_ns",
            "calibration_epoch",
            "calibration_entered_ns",
            "calibration_entry_report_sequence",
            "calibration_entry_framed_report_sequence",
            "calibration_entry_transition_sequence",
            "report_sequence",
            "framed_report_sequence",
            "transition_sequence",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("active", "known", "pressed"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.pressed and not self.known:
            raise ValueError("an unknown raw activation state cannot be pressed")
        if self.active and self.calibration_entered_ns <= 0:
            raise ValueError("active calibration requires its entry timestamp")
        if self.calibration_entered_ns > self.captured_ns:
            raise ValueError("calibration entry cannot be later than captured_ns")
        if self.calibration_entry_report_sequence > self.report_sequence:
            raise ValueError("calibration entry report sequence exceeds current reports")
        if (
            self.calibration_entry_framed_report_sequence
            > self.framed_report_sequence
        ):
            raise ValueError(
                "calibration entry framed-report sequence exceeds framed reports"
            )
        if self.framed_report_sequence > self.report_sequence:
            raise ValueError("framed report sequence exceeds all reports")
        if self.calibration_entry_transition_sequence > self.transition_sequence:
            raise ValueError(
                "calibration entry transition sequence exceeds current transitions"
            )
        if self.transition_sequence > self.report_sequence:
            raise ValueError("activation transition sequence exceeds reports")
        if self.last_report_framed is not None and not isinstance(
            self.last_report_framed, bool
        ):
            raise TypeError("last_report_framed must be bool or None")
        if (self.report_sequence == 0) != (self.last_report_framed is None):
            raise ValueError("last-report provenance must match report availability")
        if not isinstance(self.post_entry_press_seen, bool):
            raise TypeError("post_entry_press_seen must be bool")
        if self.post_entry_press_seen and (
            not self.active
            or self.framed_report_sequence
            <= self.calibration_entry_framed_report_sequence
        ):
            raise ValueError("post-entry press requires a fresh framed report")
        for name in (
            "continuous_state_since_ns",
            "release_started_ns",
            "completed_release_started_ns",
            "completed_press_ns",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
            if value is not None and value > self.captured_ns:
                raise ValueError(f"{name} cannot be later than captured_ns")
            if (
                self.active
                and value is not None
                and value < self.calibration_entered_ns
                and name != "continuous_state_since_ns"
            ):
                raise ValueError(f"{name} cannot predate calibration entry")
        for name in (
            "release_started_report_sequence",
            "completed_release_report_sequence",
            "completed_press_report_sequence",
            "completed_press_transition_sequence",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if (self.release_started_ns is None) != (
            self.release_started_report_sequence is None
        ):
            raise ValueError("live release timestamp/report sequence must be paired")
        completed = (
            self.completed_release_started_ns,
            self.completed_release_report_sequence,
            self.completed_press_ns,
            self.completed_press_report_sequence,
            self.completed_press_transition_sequence,
        )
        if any(value is None for value in completed) != all(
            value is None for value in completed
        ):
            raise ValueError("completed release/press evidence must be complete")
        if (
            completed[0] is not None
            and completed[2] is not None
            and completed[2] < completed[0]
        ):
            raise ValueError("completed press cannot precede its release")
        if self.release_started_report_sequence is not None and not (
            self.calibration_entry_framed_report_sequence
            < self.release_started_report_sequence
            <= self.framed_report_sequence
        ):
            raise ValueError(
                "live release framed report is outside the calibration epoch"
            )
        if completed[1] is not None and completed[3] is not None:
            if not (
                self.calibration_entry_framed_report_sequence
                < completed[1]
                <= completed[3]
                <= self.framed_report_sequence
            ):
                raise ValueError(
                    "completed release/press framed reports are outside the "
                    "calibration epoch"
                )
        if completed[4] is not None and not (
            self.calibration_entry_transition_sequence
            < completed[4]
            <= self.transition_sequence
        ):
            raise ValueError(
                "completed press transition is outside the calibration epoch"
            )
        if self.pressed and completed[4] is not None and (
            completed[4] != self.transition_sequence
        ):
            raise ValueError("pressed state does not match the completed hold transition")
        if not self.active and any(value is not None for value in completed):
            raise ValueError("inactive calibration cannot retain a completed cycle")
        if not self.active and self.release_started_ns is not None:
            raise ValueError("inactive calibration cannot retain a live release")
        if self.release_started_ns is not None and (
            not self.active or not self.known or self.pressed
        ):
            raise ValueError("a live release requires known released calibration state")
        if self.known and self.continuous_state_since_ns is None:
            raise ValueError("known raw activation requires a state timestamp")


def makcu_target_delta(
    target: Detection | None,
    frame_shape: tuple[int, ...],
    config: MakcuAimConfig,
) -> tuple[int, int]:
    """Return one bounded relative mouse correction for the current frame."""

    error_x, error_y = _target_error_pixels(target, frame_shape, config)
    delta_x = 0 if abs(error_x) <= config.deadzone_pixels else round(error_x * config.strength)
    delta_y = 0 if abs(error_y) <= config.deadzone_pixels else round(error_y * config.strength)
    limit = config.max_step
    return (
        min(max(delta_x, -limit), limit),
        min(max(delta_y, -limit), limit),
    )


def _makcu_target_rate(
    target: Detection | None,
    frame_shape: tuple[int, ...],
    config: MakcuAimConfig,
) -> tuple[float, float]:
    """Return mouse counts per second for the current visual error."""

    error_x, error_y = _target_error_pixels(target, frame_shape, config)
    correction_x = 0.0 if abs(error_x) <= config.deadzone_pixels else error_x * config.strength
    correction_y = 0.0 if abs(error_y) <= config.deadzone_pixels else error_y * config.strength
    limit = float(config.max_step)
    return (
        min(max(correction_x, -limit), limit) * REFERENCE_CONTROL_HZ,
        min(max(correction_y, -limit), limit) * REFERENCE_CONTROL_HZ,
    )


def _combine_pursuit_correction(
    error: float,
    base: float,
    accumulated: float,
    *,
    in_deadzone: bool,
) -> float:
    """Combine pursuit without letting retained history stall a fresh reversal."""

    correction = base + accumulated
    if not in_deadzone and correction * error < 0.0:
        # Retain and unwind the learned rate internally, but once fresh error
        # is outside the deadzone never emit a command which would move farther
        # away from that measurement. The independently safe positional term
        # (including only same-direction motion assist) must remain live.
        return base
    return correction


def _target_error_pixels(
    target: Detection | None,
    frame_shape: tuple[int, ...],
    config: MakcuAimConfig,
) -> tuple[float, float]:
    """Return error in resolution-independent 1920x1080 reference pixels."""

    if target is None:
        return 0.0, 0.0
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    target_x, target_y = head_target_point(target, config.head_ratio)
    error_x = (target_x - width / 2.0) * (REFERENCE_FRAME_WIDTH / width)
    error_y = (target_y - height / 2.0) * (REFERENCE_FRAME_HEIGHT / height)
    if config.invert_x:
        error_x = -error_x
    if config.invert_y:
        error_y = -error_y
    return error_x, error_y


class MakcuAimingController:
    """Publish latest detections to a 1 kHz, button-gated MAKCU loop."""

    def __init__(
        self,
        config: MakcuAimConfig | None = None,
        *,
        calibrated_controller: MakcuCalibratedController | None = None,
        expected_identity_token: str | None = None,
        serial_factory: Callable[..., Any] | None = None,
        ports_provider: Callable[[], Iterable[Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        threaded_output: bool = True,
        stop_timeout: float = MAKCU_STOP_TIMEOUT_SECONDS,
    ) -> None:
        if (
            isinstance(stop_timeout, bool)
            or not math.isfinite(stop_timeout)
            or stop_timeout <= 0.0
        ):
            raise ValueError("MAKCU stop timeout must be finite and greater than zero")
        if calibrated_controller is not None and not isinstance(
            calibrated_controller, MakcuCalibratedController
        ):
            raise TypeError(
                "calibrated_controller must be a MakcuCalibratedController or None"
            )
        if expected_identity_token is not None and (
            not isinstance(expected_identity_token, str)
            or len(expected_identity_token) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_identity_token
            )
        ):
            raise ValueError(
                "expected_identity_token must be a lowercase SHA-256 token or None"
            )
        self.config = config or MakcuAimConfig()
        self._calibrated_controller = calibrated_controller
        self._expected_identity_token = expected_identity_token
        self._calibrated_processed_sample_id = 0
        self._calibrated_last_output = None
        self._serial_factory = serial_factory
        self._ports_provider = ports_provider
        self._sleep = sleep
        self._threaded_output = threaded_output
        self._serial: Any | None = None
        self._button_mask = 0
        self._button_state_known = False
        self._activation_started_ns = 0
        self._activation_requires_release = False
        self._activation_report_sequence = 0
        self._activation_framed_report_sequence = 0
        self._activation_transition_sequence = 0
        self._activation_last_report_ns = 0
        self._activation_last_report_framed: bool | None = None
        self._activation_continuous_state_since_ns: int | None = None
        self._calibration_activation_epoch = 0
        self._calibration_activation_entered_ns = 0
        self._calibration_entry_report_sequence = 0
        self._calibration_entry_framed_report_sequence = 0
        self._calibration_entry_transition_sequence = 0
        self._calibration_post_entry_press_seen = False
        self._calibration_release_started_ns: int | None = None
        self._calibration_release_started_report_sequence: int | None = None
        self._calibration_completed_release_started_ns: int | None = None
        self._calibration_completed_release_report_sequence: int | None = None
        self._calibration_completed_press_ns: int | None = None
        self._calibration_completed_press_report_sequence: int | None = None
        self._calibration_completed_press_transition_sequence: int | None = None
        self._button_parser = _ButtonStreamParser()
        self._serial_lock = threading.Lock()
        self._connection_close_lock = threading.Lock()
        self._state_lock = threading.Lock()
        # The numeric calibrated controller is intentionally lock-free.  The
        # output worker normally owns it, while lifecycle calls must still be
        # able to invalidate it synchronously.  Keep this lock separate from
        # _state_lock and always acquire it first when both are needed.
        self._calibrated_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._output_thread: threading.Thread | None = None
        self._shutdown_threads: list[threading.Thread] = []
        self._stop_timeout = float(stop_timeout)
        self._worker_error: MakcuError | None = None
        self._latest_target: Detection | None = None
        self._latest_frame_shape: tuple[int, int, int] = (0, 0, 0)
        self._latest_active = False
        self._latest_update_ns = 0
        self._fractional_x = 0.0
        self._fractional_y = 0.0
        self._smoothed_rate_x = 0.0
        self._smoothed_rate_y = 0.0
        self._latest_measurement_ns = 0
        self._latest_source_ns = 0
        self._latest_measurement_observed = True
        self._measurement_target_present = False
        self._latest_velocity_x = 0.0
        self._latest_velocity_y = 0.0
        self._latest_sample_id = 0
        self._normal_motion_generation = 0
        self._processed_sample_id = 0
        self._control_error_x = 0.0
        self._control_error_y = 0.0
        self._measurement_error_x = 0.0
        self._measurement_error_y = 0.0
        self._pursuit_correction_x = 0.0
        self._pursuit_correction_y = 0.0
        self._pursuit_measurement_ns = 0
        self._calibration_token: object | None = None
        self._calibration_lease_token: object | None = None
        self._calibration_lease_valid = False
        self._calibration_lease_measurement_ns = 0
        self._calibration_hold_transition_sequence: int | None = None
        self._calibration_pending_token: object | None = None
        self._calibration_pending_axis: str | None = None
        self._calibration_pending_counts = 0
        self._calibration_pending_rate = 0.0
        self._calibration_fractional_counts = 0.0
        self._calibration_emitted_x = 0
        self._calibration_emitted_y = 0
        self._calibration_emitted_abs_counts = 0
        self._calibration_movement_commands = 0
        self._calibration_first_emitted_ns: int | None = None
        self._calibration_last_emitted_ns: int | None = None
        self._calibration_emitted_events: list[tuple[int, int, int]] = []
        self._calibration_abort_reason: str | None = None
        self._telemetry_output_ticks = 0
        self._telemetry_active_input_ticks = 0
        self._telemetry_button_pressed_ticks = 0
        self._telemetry_target_present_ticks = 0
        self._telemetry_fresh_target_ticks = 0
        self._telemetry_authorized_ticks = 0
        self._telemetry_movement_commands = 0
        self._telemetry_emitted_x = 0
        self._telemetry_emitted_y = 0
        self._telemetry_emitted_abs_x = 0
        self._telemetry_emitted_abs_y = 0
        self._telemetry_control_samples = 0
        self._telemetry_control_error_x = 0.0
        self._telemetry_control_error_y = 0.0
        self._telemetry_control_error_abs_x = 0.0
        self._telemetry_control_error_abs_y = 0.0
        self._telemetry_pursuit_x = 0.0
        self._telemetry_pursuit_y = 0.0
        self._telemetry_pursuit_abs_x = 0.0
        self._telemetry_pursuit_abs_y = 0.0
        self._telemetry_saturated_x_samples = 0
        self._telemetry_saturated_y_samples = 0
        self._telemetry_pursuit_resets = 0
        self.connected_port: str | None = None
        self._identity_token: str | None = None

    @property
    def identity_token(self) -> str | None:
        """Opaque stable device binding, available only after firmware verification."""

        with self._state_lock:
            return self._identity_token

    @property
    def control_mode(self) -> str:
        """Return the selected normal-control law without exposing profile secrets."""

        return "calibrated" if self._calibrated_controller is not None else "legacy"

    @property
    def calibrated_control_output(self) -> CalibratedControlOutput | None:
        """Return the last immutable calibrated decision, or ``None`` in legacy mode."""

        with self._calibrated_lock:
            return self._calibrated_last_output

    def _reset_calibrated_control_locked(self) -> None:
        """Reset calibrated state while the caller owns _calibrated_lock."""

        self._calibrated_processed_sample_id = 0
        self._calibrated_last_output = None
        if self._calibrated_controller is not None:
            self._calibrated_controller.reset()

    def _reset_calibrated_tracking_locked(self) -> None:
        """Revoke learned pursuit without erasing successful physical facts."""

        self._calibrated_processed_sample_id = 0
        self._calibrated_last_output = None
        if self._calibrated_controller is not None:
            self._calibrated_controller.reset(clear_command_history=False)

    def _reset_calibrated_control(self) -> None:
        """Synchronously invalidate every calibrated normal-motion source."""

        with self._calibrated_lock:
            self._reset_calibrated_control_locked()

    def telemetry_snapshot(self) -> MakcuTelemetrySnapshot:
        """Return a consistent read-only snapshot without touching the board."""

        with self._telemetry_lock:
            return MakcuTelemetrySnapshot(
                output_ticks=self._telemetry_output_ticks,
                active_input_ticks=self._telemetry_active_input_ticks,
                button_pressed_ticks=self._telemetry_button_pressed_ticks,
                target_present_ticks=self._telemetry_target_present_ticks,
                fresh_target_ticks=self._telemetry_fresh_target_ticks,
                authorized_ticks=self._telemetry_authorized_ticks,
                movement_commands=self._telemetry_movement_commands,
                emitted_x=self._telemetry_emitted_x,
                emitted_y=self._telemetry_emitted_y,
                emitted_abs_x=self._telemetry_emitted_abs_x,
                emitted_abs_y=self._telemetry_emitted_abs_y,
                control_samples=self._telemetry_control_samples,
                control_error_x=self._telemetry_control_error_x,
                control_error_y=self._telemetry_control_error_y,
                control_error_abs_x=self._telemetry_control_error_abs_x,
                control_error_abs_y=self._telemetry_control_error_abs_y,
                pursuit_x=self._telemetry_pursuit_x,
                pursuit_y=self._telemetry_pursuit_y,
                pursuit_abs_x=self._telemetry_pursuit_abs_x,
                pursuit_abs_y=self._telemetry_pursuit_abs_y,
                saturated_x_samples=self._telemetry_saturated_x_samples,
                saturated_y_samples=self._telemetry_saturated_y_samples,
                pursuit_resets=self._telemetry_pursuit_resets,
            )

    def calibration_snapshot(self) -> MakcuCalibrationSnapshot:
        """Return only counts which the calibration worker successfully wrote."""

        with self._state_lock:
            return MakcuCalibrationSnapshot(
                active=self._calibration_token is not None,
                captured_ns=time.perf_counter_ns(),
                emitted_x=self._calibration_emitted_x,
                emitted_y=self._calibration_emitted_y,
                emitted_abs_counts=self._calibration_emitted_abs_counts,
                movement_commands=self._calibration_movement_commands,
                first_emitted_ns=self._calibration_first_emitted_ns,
                last_emitted_ns=self._calibration_last_emitted_ns,
                emitted_events=tuple(self._calibration_emitted_events),
                pending_axis=self._calibration_pending_axis,
                pending_counts=self._calibration_pending_counts,
                pending_rate_counts_per_second=self._calibration_pending_rate,
                abort_reason=self._calibration_abort_reason,
            )

    def _clear_normal_motion_locked(self) -> None:
        """Discard every normal-control input and every source of stored motion."""

        self._normal_motion_generation += 1
        self._latest_target = None
        self._latest_frame_shape = (0, 0, 0)
        self._latest_active = False
        self._latest_update_ns = 0
        self._fractional_x = 0.0
        self._fractional_y = 0.0
        self._smoothed_rate_x = 0.0
        self._smoothed_rate_y = 0.0
        self._latest_measurement_ns = 0
        self._latest_source_ns = 0
        self._latest_measurement_observed = True
        self._measurement_target_present = False
        self._latest_velocity_x = 0.0
        self._latest_velocity_y = 0.0
        self._latest_sample_id = 0
        self._processed_sample_id = 0
        self._control_error_x = 0.0
        self._control_error_y = 0.0
        self._measurement_error_x = 0.0
        self._measurement_error_y = 0.0
        self._pursuit_correction_x = 0.0
        self._pursuit_correction_y = 0.0
        self._pursuit_measurement_ns = 0

    def _clear_calibration_pending_locked(self) -> None:
        self._calibration_pending_token = None
        self._calibration_pending_axis = None
        self._calibration_pending_counts = 0
        self._calibration_pending_rate = 0.0
        self._calibration_fractional_counts = 0.0

    def _reset_calibration_session_locked(self) -> None:
        """Invalidate every prior token and erase its evidence before a restart."""

        self._calibration_token = None
        self._calibration_activation_entered_ns = 0
        self._calibration_entry_report_sequence = 0
        self._calibration_entry_framed_report_sequence = 0
        self._calibration_entry_transition_sequence = 0
        self._calibration_post_entry_press_seen = False
        self._calibration_release_started_ns = None
        self._calibration_release_started_report_sequence = None
        self._calibration_completed_release_started_ns = None
        self._calibration_completed_release_report_sequence = None
        self._calibration_completed_press_ns = None
        self._calibration_completed_press_report_sequence = None
        self._calibration_completed_press_transition_sequence = None
        self._calibration_lease_token = None
        self._calibration_lease_valid = False
        self._calibration_lease_measurement_ns = 0
        self._calibration_hold_transition_sequence = None
        self._clear_calibration_pending_locked()
        self._calibration_emitted_x = 0
        self._calibration_emitted_y = 0
        self._calibration_emitted_abs_counts = 0
        self._calibration_movement_commands = 0
        self._calibration_first_emitted_ns = None
        self._calibration_last_emitted_ns = None
        self._calibration_emitted_events.clear()
        self._calibration_abort_reason = None

    def _abort_calibration_locked(
        self,
        reason: str,
        *,
        deactivate: bool = False,
    ) -> None:
        """Fail closed without inventing a compensating or return movement."""

        if self._calibration_token is None:
            return
        if self._calibration_abort_reason is None:
            self._calibration_abort_reason = reason
        self._calibration_lease_token = None
        self._calibration_lease_valid = False
        self._calibration_lease_measurement_ns = 0
        self._calibration_hold_transition_sequence = None
        self._clear_calibration_pending_locked()
        self._clear_normal_motion_locked()
        if deactivate:
            self._calibration_token = None

    def _require_calibration_token_locked(self, token: object) -> None:
        if self._calibration_token is None or token is not self._calibration_token:
            raise MakcuError("invalid or inactive MAKCU calibration token")

    def _calibration_worker_is_live(self) -> bool:
        worker = self._output_thread
        worker_is_alive = getattr(worker, "is_alive", None)
        return bool(
            self._serial is not None
            and self.config.output_hz == CALIBRATION_OUTPUT_HZ
            and not self._stop_event.is_set()
            and callable(worker_is_alive)
            and worker_is_alive()
        )

    def enter_calibration_mode(self) -> object:
        """Exclusively reserve the live 1 kHz worker for bounded calibration pulses."""

        with self._state_lock:
            if self._serial is None:
                raise MakcuError("MAKCU serial connection is not open")
            if self._worker_error is not None:
                raise self._worker_error
            if not self._calibration_worker_is_live():
                raise MakcuError(
                    "MAKCU calibration requires the live 1 kHz output worker"
                )
            if self._calibration_token is not None:
                raise MakcuError("MAKCU calibration mode is already active")
            token = object()
            self._reset_calibration_session_locked()
            self._calibration_token = token
            self._calibration_activation_epoch += 1
            entered_ns = time.perf_counter_ns()
            self._calibration_activation_entered_ns = entered_ns
            self._calibration_entry_report_sequence = self._activation_report_sequence
            self._calibration_entry_framed_report_sequence = (
                self._activation_framed_report_sequence
            )
            self._calibration_entry_transition_sequence = (
                self._activation_transition_sequence
            )
            self._calibration_post_entry_press_seen = False
            # Only a report received after exclusive entry can begin the
            # release proof. A cached pre-entry zero is not fresh evidence.
            self._calibration_release_started_ns = None
            self._calibration_release_started_report_sequence = None
            self._calibration_completed_release_started_ns = None
            self._calibration_completed_release_report_sequence = None
            self._calibration_completed_press_ns = None
            self._calibration_completed_press_report_sequence = None
            self._calibration_completed_press_transition_sequence = None
            # Calibration requires an unambiguous physical action after the
            # exclusive mode begins. A button which was already held cannot
            # authorize a pulse until its release and a new press are observed.
            self._activation_started_ns = 0
            self._activation_requires_release = True
            self._clear_normal_motion_locked()
        # The token is already an exclusive commit barrier.  Reset outside the
        # state lock so an in-flight calibrated tick can finish its lock order
        # (_calibrated_lock -> _state_lock), observe the token, and discard its
        # command without deadlocking this lifecycle call.
        self._reset_calibrated_control()
        with self._state_lock:
            if self._calibration_token is not token:
                raise MakcuError(
                    "MAKCU calibration mode was interrupted while entering"
                )
        return token

    def publish_calibration_lease(
        self,
        valid: bool,
        measurement_ns: int,
        token: object,
        *,
        activation_transition_sequence: int | None = None,
    ) -> None:
        """Publish one detector-timestamped quality lease for the pulse worker."""

        if not isinstance(valid, bool):
            raise TypeError("calibration lease valid must be bool")
        if isinstance(measurement_ns, bool) or not isinstance(measurement_ns, int):
            raise TypeError("calibration lease timestamp must be an integer")
        if measurement_ns < 0:
            raise ValueError("calibration lease timestamp cannot be negative")
        if activation_transition_sequence is not None and (
            isinstance(activation_transition_sequence, bool)
            or not isinstance(activation_transition_sequence, int)
            or activation_transition_sequence < 0
        ):
            raise ValueError(
                "activation_transition_sequence must be a non-negative integer or None"
            )
        with self._state_lock:
            self._require_calibration_token_locked(token)
            if self._worker_error is not None:
                raise self._worker_error
            if not self._calibration_worker_is_live():
                self._abort_calibration_locked(
                    "calibration output worker is unavailable"
                )
                raise MakcuError("MAKCU calibration output worker is unavailable")
            if self._calibration_abort_reason is not None:
                raise MakcuError(
                    "MAKCU calibration session is aborted: "
                    f"{self._calibration_abort_reason}"
                )
            if (
                self._calibration_lease_measurement_ns
                and measurement_ns < self._calibration_lease_measurement_ns
            ):
                raise ValueError("calibration lease timestamps must not move backwards")
            if valid:
                raw_pressed = bool(
                    self._button_state_known
                    and self._button_mask
                    & (1 << self.config.activation_button)
                )
                if not raw_pressed or self._activation_requires_release:
                    self._abort_calibration_locked(
                        "calibration lease requires a fresh physical hold"
                    )
                    raise MakcuError(
                        "MAKCU calibration lease requires a fresh physical hold"
                    )
                if (
                    not self._calibration_post_entry_press_seen
                    or self._calibration_completed_release_started_ns is None
                    or self._calibration_completed_release_report_sequence is None
                    or self._calibration_completed_press_ns is None
                    or self._calibration_completed_press_report_sequence is None
                    or self._calibration_completed_press_transition_sequence is None
                    or self._calibration_completed_release_started_ns
                    < self._calibration_activation_entered_ns
                    or self._calibration_completed_release_report_sequence
                    <= self._calibration_entry_framed_report_sequence
                    or self._calibration_completed_press_report_sequence
                    < self._calibration_completed_release_report_sequence
                    or self._calibration_completed_press_transition_sequence
                    != self._activation_transition_sequence
                    or self._activation_framed_report_sequence
                    <= self._calibration_entry_framed_report_sequence
                ):
                    self._abort_calibration_locked(
                        "calibration lease lacks a fresh release/hold transition"
                    )
                    raise MakcuError(
                        "MAKCU calibration lease lacks a fresh release/hold transition"
                    )
                if (
                    activation_transition_sequence is not None
                    and activation_transition_sequence
                    != self._calibration_completed_press_transition_sequence
                ):
                    self._abort_calibration_locked(
                        "calibration hold transition changed before lease commit"
                    )
                    raise MakcuError(
                        "MAKCU calibration hold transition changed before lease commit"
                    )
                if self._calibration_hold_transition_sequence is None:
                    self._calibration_hold_transition_sequence = (
                        self._calibration_completed_press_transition_sequence
                    )
                elif (
                    self._activation_transition_sequence
                    != self._calibration_hold_transition_sequence
                ):
                    self._abort_calibration_locked(
                        "physical activation changed after calibration hold"
                    )
                    raise MakcuError(
                        "MAKCU calibration activation changed after its hold"
                    )
            self._calibration_lease_token = token
            self._calibration_lease_valid = valid
            self._calibration_lease_measurement_ns = measurement_ns
            if not valid:
                self._abort_calibration_locked("calibration lease was invalidated")

    def request_calibration_pulse(
        self,
        axis: str,
        signed_counts: int,
        bounded_rate: float,
        token: object,
    ) -> None:
        """Queue one hard-bounded, single-axis excursion for the 1 kHz worker."""

        with self._state_lock:
            self._require_calibration_token_locked(token)
            if self._worker_error is not None:
                raise self._worker_error
            if not self._calibration_worker_is_live():
                self._abort_calibration_locked(
                    "calibration output worker is unavailable"
                )
                raise MakcuError("MAKCU calibration output worker is unavailable")
            if self._calibration_abort_reason is not None:
                raise MakcuError(
                    "MAKCU calibration session is aborted: "
                    f"{self._calibration_abort_reason}"
                )
            if axis not in ("x", "y"):
                raise ValueError("calibration pulse axis must be 'x' or 'y'")
            if isinstance(signed_counts, bool) or not isinstance(signed_counts, int):
                raise TypeError("calibration pulse counts must be an integer")
            if signed_counts == 0:
                raise ValueError("calibration pulse counts cannot be zero")
            if abs(signed_counts) > CALIBRATION_MAX_EXCURSION_COUNTS:
                raise ValueError(
                    "calibration pulse cannot exceed "
                    f"{CALIBRATION_MAX_EXCURSION_COUNTS} counts"
                )
            if (
                isinstance(bounded_rate, bool)
                or not isinstance(bounded_rate, (int, float))
                or not math.isfinite(float(bounded_rate))
                or float(bounded_rate) <= 0.0
                or float(bounded_rate) > CALIBRATION_MAX_RATE_COUNTS_PER_SECOND
            ):
                raise ValueError(
                    "calibration pulse rate must be greater than zero and at most "
                    f"{CALIBRATION_MAX_RATE_COUNTS_PER_SECOND:g} counts/s"
                )
            if self._calibration_pending_counts:
                raise MakcuError("a MAKCU calibration pulse is already pending")
            if (
                not self._calibration_lease_valid
                or self._calibration_lease_token is not token
            ):
                raise MakcuError("a valid calibration lease is required before a pulse")
            if (
                self._calibration_hold_transition_sequence is None
                or self._activation_transition_sequence
                != self._calibration_hold_transition_sequence
            ):
                self._abort_calibration_locked(
                    "physical activation changed before calibration pulse"
                )
                raise MakcuError(
                    "physical activation changed before calibration pulse"
                )
            if (
                self._calibration_emitted_abs_counts + abs(signed_counts)
                > CALIBRATION_MAX_SESSION_ABS_COUNTS
            ):
                raise ValueError(
                    "calibration session cannot exceed "
                    f"{CALIBRATION_MAX_SESSION_ABS_COUNTS} absolute counts"
                )
            self._calibration_pending_token = token
            self._calibration_pending_axis = axis
            self._calibration_pending_counts = signed_counts
            self._calibration_pending_rate = float(bounded_rate)
            self._calibration_fractional_counts = 0.0

    def exit_calibration_mode(self, token: object) -> None:
        """End exclusive mode and discard all queued and learned movement."""

        with self._state_lock:
            self._require_calibration_token_locked(token)
            if self._calibration_pending_counts:
                self._abort_calibration_locked(
                    "calibration mode exited with a pending pulse"
                )
            self._calibration_lease_token = None
            self._calibration_lease_valid = False
            self._calibration_lease_measurement_ns = 0
            self._clear_calibration_pending_locked()
            self._clear_normal_motion_locked()
        # Keep the exclusive token installed until calibrated state is fully
        # invalidated.  No post-calibration normal tick can otherwise reuse
        # pursuit/command history between token removal and this reset.
        self._reset_calibrated_control()
        with self._state_lock:
            self._require_calibration_token_locked(token)
            self._calibration_token = None
            self._calibration_lease_token = None
            self._calibration_lease_valid = False
            self._calibration_lease_measurement_ns = 0
            self._clear_calibration_pending_locked()
            self._clear_normal_motion_locked()

    def _reset_telemetry(self) -> None:
        with self._telemetry_lock:
            self._telemetry_output_ticks = 0
            self._telemetry_active_input_ticks = 0
            self._telemetry_button_pressed_ticks = 0
            self._telemetry_target_present_ticks = 0
            self._telemetry_fresh_target_ticks = 0
            self._telemetry_authorized_ticks = 0
            self._telemetry_movement_commands = 0
            self._telemetry_emitted_x = 0
            self._telemetry_emitted_y = 0
            self._telemetry_emitted_abs_x = 0
            self._telemetry_emitted_abs_y = 0
            self._telemetry_control_samples = 0
            self._telemetry_control_error_x = 0.0
            self._telemetry_control_error_y = 0.0
            self._telemetry_control_error_abs_x = 0.0
            self._telemetry_control_error_abs_y = 0.0
            self._telemetry_pursuit_x = 0.0
            self._telemetry_pursuit_y = 0.0
            self._telemetry_pursuit_abs_x = 0.0
            self._telemetry_pursuit_abs_y = 0.0
            self._telemetry_saturated_x_samples = 0
            self._telemetry_saturated_y_samples = 0
            self._telemetry_pursuit_resets = 0

    def _record_pursuit_reset(self, *, reset_x: bool = True, reset_y: bool = True) -> None:
        """Count an actual accumulator clear, never a repeated empty tick."""

        reset_nonzero = (
            reset_x and self._pursuit_correction_x != 0.0
        ) or (
            reset_y and self._pursuit_correction_y != 0.0
        )
        if reset_nonzero:
            with self._telemetry_lock:
                self._telemetry_pursuit_resets += 1

    def _record_control_sample(
        self,
        error_x: float,
        error_y: float,
        pursuit_x: float,
        pursuit_y: float,
        correction_x: float,
        correction_y: float,
        limit_x: float,
        limit_y: float,
    ) -> None:
        """Aggregate one new authorized detector sample without device I/O."""

        with self._telemetry_lock:
            self._telemetry_control_samples += 1
            self._telemetry_control_error_x += error_x
            self._telemetry_control_error_y += error_y
            self._telemetry_control_error_abs_x += abs(error_x)
            self._telemetry_control_error_abs_y += abs(error_y)
            self._telemetry_pursuit_x += pursuit_x
            self._telemetry_pursuit_y += pursuit_y
            self._telemetry_pursuit_abs_x += abs(pursuit_x)
            self._telemetry_pursuit_abs_y += abs(pursuit_y)
            self._telemetry_saturated_x_samples += int(abs(correction_x) >= limit_x)
            self._telemetry_saturated_y_samples += int(abs(correction_y) >= limit_y)

    def _require_serial(self) -> None:
        if self._serial_factory is not None:
            return
        if _serial is None:
            detail = f": {SERIAL_IMPORT_ERROR}" if SERIAL_IMPORT_ERROR else ""
            raise MakcuError("MAKCU output requires the 'pyserial' package" + detail)

    def _available_ports(self) -> Iterable[Any]:
        if self._ports_provider is not None:
            return self._ports_provider()
        assert _list_ports is not None
        return _list_ports.comports()

    def _find_port(self) -> str:
        if self._ports_provider is None:
            return detect_makcu_port(requested=self.config.port)
        return detect_makcu_port(
            requested=self.config.port,
            ports_provider=self._available_ports,
        )

    def _open(self, port: str, baudrate: int) -> Any:
        factory = self._serial_factory
        if factory is None:
            assert _serial is not None
            factory = _serial.Serial
        return factory(
            port=port,
            baudrate=baudrate,
            timeout=0.05,
            write_timeout=0.05,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

    def _command(self, command: str) -> None:
        if self._serial is None:
            raise MakcuError("MAKCU serial connection is not open")
        try:
            with self._serial_lock:
                self._serial.write(f"{command}\r".encode("ascii"))
                self._serial.flush()
        except (OSError, ValueError) as exc:
            raise MakcuError(f"MAKCU command failed: {exc}") from exc

    def _version_ok(self, connection: Any) -> bool:
        try:
            connection.reset_input_buffer()
            connection.write(b"km.version()\r")
            connection.flush()
            self._sleep(0.06)
            available = int(getattr(connection, "in_waiting", 0))
            response = connection.read(available) if available else b""
        except (OSError, ValueError):
            return False
        return b"MAKCU" in response.upper()

    def _connect(self, port: str) -> Any:
        permission_error: BaseException | None = None
        for baudrate in MAKCU_BAUD_RATES:
            connection = None
            try:
                connection = self._open(port, baudrate)
                if not self._version_ok(connection):
                    connection.close()
                    continue
                if baudrate == 115_200:
                    connection.write(MAKCU_BAUD_CHANGE)
                    connection.flush()
                    self._sleep(0.03)
                    connection.baudrate = MAKCU_FAST_BAUD
                    self._sleep(0.03)
                    if not self._version_ok(connection):
                        connection.close()
                        raise MakcuError("MAKCU did not respond after switching to 4M baud")
                return connection
            except PermissionError as exc:
                permission_error = exc
                if connection is not None:
                    connection.close()
                break
            except MakcuError:
                raise
            except (OSError, ValueError) as exc:
                if connection is not None:
                    connection.close()
                if getattr(exc, "errno", None) in (errno.EACCES, errno.EPERM):
                    permission_error = exc
                    break
        if permission_error is not None:
            raise MakcuError(
                f"Permission denied opening {port}. Install the ProAim MAKCU "
                "udev rule, then reconnect the board."
            ) from permission_error
        raise MakcuError(f"No MAKCU firmware response was received from {port}")

    def start(self, *, output_loop: bool | None = None) -> None:
        if self._serial is not None:
            if self._stop_event.is_set():
                raise MakcuError(
                    "MAKCU cannot restart because its previous shutdown is incomplete"
                )
            return
        self._require_serial()
        port = self._find_port()
        self._serial = self._connect(port)
        self.connected_port = port
        self._identity_token = _makcu_identity_token(port)
        if (
            self._expected_identity_token is not None
            and self._identity_token != self._expected_identity_token
        ):
            connection = self._serial
            self._serial = None
            self.connected_port = None
            self._identity_token = None
            try:
                connection.close()
            finally:
                raise MakcuError(
                    "Connected MAKCU identity does not match the active calibration profile"
                )
        self._button_mask = 0
        self._button_state_known = False
        self._activation_started_ns = 0
        self._activation_requires_release = False
        self._activation_report_sequence = 0
        self._activation_framed_report_sequence = 0
        self._activation_transition_sequence = 0
        self._activation_last_report_ns = 0
        self._activation_last_report_framed = None
        self._activation_continuous_state_since_ns = None
        self._calibration_activation_entered_ns = 0
        self._calibration_entry_report_sequence = 0
        self._calibration_entry_framed_report_sequence = 0
        self._calibration_entry_transition_sequence = 0
        self._calibration_post_entry_press_seen = False
        self._calibration_release_started_ns = None
        self._calibration_release_started_report_sequence = None
        self._calibration_completed_release_started_ns = None
        self._calibration_completed_release_report_sequence = None
        self._calibration_completed_press_ns = None
        self._calibration_completed_press_report_sequence = None
        self._calibration_completed_press_transition_sequence = None
        self._calibration_hold_transition_sequence = None
        self._button_parser.reset()
        self._worker_error = None
        self._latest_target = None
        self._latest_active = False
        self._latest_update_ns = 0
        self._fractional_x = 0.0
        self._fractional_y = 0.0
        self._smoothed_rate_x = 0.0
        self._smoothed_rate_y = 0.0
        self._latest_measurement_ns = 0
        self._latest_source_ns = 0
        self._latest_measurement_observed = True
        self._measurement_target_present = False
        self._latest_velocity_x = 0.0
        self._latest_velocity_y = 0.0
        self._latest_sample_id = 0
        self._normal_motion_generation = 0
        self._processed_sample_id = 0
        self._control_error_x = 0.0
        self._control_error_y = 0.0
        self._measurement_error_x = 0.0
        self._measurement_error_y = 0.0
        self._pursuit_correction_x = 0.0
        self._pursuit_correction_y = 0.0
        self._pursuit_measurement_ns = 0
        with self._state_lock:
            self._reset_calibration_session_locked()
        self._reset_calibrated_control()
        self._reset_telemetry()
        self._stop_event.clear()
        self._command(BUTTON_STREAM_COMMAND)
        should_start_output = self._threaded_output if output_loop is None else output_loop
        if should_start_output:
            self._output_thread = threading.Thread(
                target=self._run_output_loop,
                name="makcu-1000hz-output",
                daemon=True,
            )
            self._output_thread.start()

    def _read_buttons(self, *, now_ns: int | None = None) -> None:
        if self._serial is None:
            raise MakcuError("MAKCU serial connection is not open")
        current_ns = time.perf_counter_ns() if now_ns is None else now_ns
        try:
            with self._serial_lock:
                available = int(getattr(self._serial, "in_waiting", 0))
                data = self._serial.read(available) if available else b""
        except (OSError, ValueError) as exc:
            raise MakcuError(f"MAKCU button read failed: {exc}") from exc
        reports = self._button_parser.feed(data)
        if not reports:
            return
        # Apply one serial read atomically to controller state. In particular,
        # a coalesced release/repress pair must latch the release before its
        # final pressed level can be considered by the pulse worker.
        with self._state_lock:
            for value, is_framed in reports:
                event_ns = max(
                    current_ns,
                    self._activation_last_report_ns,
                    self._calibration_activation_entered_ns,
                )
                self._activation_last_report_ns = event_ns
                was_known = self._button_state_known
                was_pressed = bool(
                    was_known
                    and self._button_mask & (1 << self.config.activation_button)
                )
                self._activation_report_sequence += 1
                if is_framed:
                    self._activation_framed_report_sequence += 1
                self._activation_last_report_framed = is_framed
                self._button_mask = value & 0x1F
                self._button_state_known = True
                is_pressed = bool(
                    self._button_mask & (1 << self.config.activation_button)
                )
                if not was_known or is_pressed != was_pressed:
                    self._activation_transition_sequence += 1
                    self._activation_continuous_state_since_ns = event_ns
                if (
                    self._calibration_token is not None
                    and not is_pressed
                    and (
                        self._calibration_lease_valid
                        or self._calibration_pending_counts
                        or self._calibration_hold_transition_sequence is not None
                    )
                ):
                    # Release safety applies to every accepted report, including
                    # legacy naked masks. Framing is required only for arming
                    # new calibration movement, never for revoking it.
                    self._abort_calibration_locked(
                        "physical activation was released"
                    )
                calibration_report_is_fresh_and_framed = bool(
                    self._calibration_token is not None
                    and is_framed
                    and current_ns >= self._calibration_activation_entered_ns
                )
                if calibration_report_is_fresh_and_framed:
                    if is_pressed:
                        self._calibration_post_entry_press_seen = True
                    if not is_pressed:
                        if (
                            self._calibration_post_entry_press_seen
                            and self._calibration_release_started_ns is None
                        ):
                            self._calibration_release_started_ns = event_ns
                            self._calibration_release_started_report_sequence = (
                                self._activation_framed_report_sequence
                            )
                    elif self._calibration_release_started_ns is not None:
                        self._calibration_completed_release_started_ns = (
                            self._calibration_release_started_ns
                        )
                        self._calibration_completed_release_report_sequence = (
                            self._calibration_release_started_report_sequence
                        )
                        self._calibration_completed_press_ns = event_ns
                        self._calibration_completed_press_report_sequence = (
                            self._activation_framed_report_sequence
                        )
                        self._calibration_completed_press_transition_sequence = (
                            self._activation_transition_sequence
                        )
                        self._calibration_release_started_ns = None
                        self._calibration_release_started_report_sequence = None
                if not is_pressed:
                    self._activation_started_ns = 0
                    self._activation_requires_release = False
                elif not was_pressed and not self._activation_requires_release:
                    self._activation_started_ns = event_ns
                # Even a release/repress pair which ends in the same mask must
                # invalidate a normal decision computed before those physical
                # events.
                self._normal_motion_generation += 1
                # A valid release report is authoritative. Do not keep movement
                # active after the physical button is released.

    def _activation_pressed_locked(self, now_ns: int) -> bool:
        """Return live authorization while the caller owns _state_lock."""

        pressed = bool(
            self._button_state_known
            and self._button_mask & (1 << self.config.activation_button)
        )
        if not pressed or self._activation_requires_release:
            return False
        if (
            self._activation_started_ns
            and now_ns - self._activation_started_ns
            > int(MAX_CONTINUOUS_ACTIVATION_SECONDS * 1_000_000_000)
        ):
            self._activation_requires_release = True
            self._normal_motion_generation += 1
            return False
        return True

    def _activation_pressed_at(self, now_ns: int) -> bool:
        with self._state_lock:
            return self._activation_pressed_locked(now_ns)

    @property
    def activation_pressed(self) -> bool:
        return self._activation_pressed_at(time.perf_counter_ns())

    @property
    def raw_activation_state(self) -> tuple[bool, bool]:
        """Return the latest known physical button state without authorization filtering.

        ``activation_pressed`` intentionally becomes false while calibration is
        waiting for a fresh post-entry release.  Calibration orchestration must
        still be able to distinguish that software latch from an actual button
        release, so this passive snapshot exposes both facts explicitly.  It
        performs no serial I/O and never changes the activation state.
        """

        with self._state_lock:
            known = self._button_state_known
            pressed = bool(
                known
                and self._button_mask & (1 << self.config.activation_button)
            )
            return known, pressed

    @property
    def raw_activation_snapshot(self) -> MakcuRawActivationSnapshot:
        """Return lossless raw transition evidence for active calibration."""

        with self._state_lock:
            captured_ns = max(
                time.perf_counter_ns(),
                self._calibration_activation_entered_ns,
                self._activation_last_report_ns,
                self._activation_continuous_state_since_ns or 0,
                self._calibration_release_started_ns or 0,
                self._calibration_completed_press_ns or 0,
            )
            active = self._calibration_token is not None
            known = self._button_state_known
            pressed = bool(
                known
                and self._button_mask & (1 << self.config.activation_button)
            )
            return MakcuRawActivationSnapshot(
                captured_ns=captured_ns,
                calibration_epoch=self._calibration_activation_epoch,
                calibration_entered_ns=(
                    self._calibration_activation_entered_ns if active else 0
                ),
                calibration_entry_report_sequence=(
                    self._calibration_entry_report_sequence if active else 0
                ),
                calibration_entry_framed_report_sequence=(
                    self._calibration_entry_framed_report_sequence
                    if active
                    else 0
                ),
                calibration_entry_transition_sequence=(
                    self._calibration_entry_transition_sequence if active else 0
                ),
                active=active,
                known=known,
                pressed=pressed,
                report_sequence=self._activation_report_sequence,
                framed_report_sequence=self._activation_framed_report_sequence,
                transition_sequence=self._activation_transition_sequence,
                last_report_framed=self._activation_last_report_framed,
                post_entry_press_seen=(
                    self._calibration_post_entry_press_seen if active else False
                ),
                continuous_state_since_ns=(
                    self._activation_continuous_state_since_ns if known else None
                ),
                release_started_ns=(
                    self._calibration_release_started_ns if active else None
                ),
                release_started_report_sequence=(
                    self._calibration_release_started_report_sequence
                    if active
                    else None
                ),
                completed_release_started_ns=(
                    self._calibration_completed_release_started_ns
                    if active
                    else None
                ),
                completed_release_report_sequence=(
                    self._calibration_completed_release_report_sequence
                    if active
                    else None
                ),
                completed_press_ns=(
                    self._calibration_completed_press_ns if active else None
                ),
                completed_press_report_sequence=(
                    self._calibration_completed_press_report_sequence
                    if active
                    else None
                ),
                completed_press_transition_sequence=(
                    self._calibration_completed_press_transition_sequence
                    if active
                    else None
                ),
            )

    def poll_button_mask(self, *, now_ns: int | None = None) -> int:
        """Read pending physical button reports and return the latest 5-bit mask."""

        current_ns = time.perf_counter_ns() if now_ns is None else now_ns
        self._read_buttons(now_ns=current_ns)
        self._activation_pressed_at(current_ns)
        with self._state_lock:
            return self._button_mask

    def poll_activation(self, *, now_ns: int | None = None) -> bool:
        """Read pending physical button reports without sending mouse movement."""

        current_ns = time.perf_counter_ns() if now_ns is None else now_ns
        self.poll_button_mask(now_ns=current_ns)
        return self._activation_pressed_at(current_ns)

    def update(
        self,
        target: Detection | None,
        frame_shape: tuple[int, int, int],
        active: bool = True,
        *,
        measurement_ns: int | None = None,
        measurement_observed: bool = True,
    ) -> None:
        if self._serial is None:
            raise MakcuError("MAKCU serial connection is not open")
        if self._worker_error is not None:
            raise self._worker_error
        published_ns = time.perf_counter_ns()
        if measurement_ns is not None and (
            isinstance(measurement_ns, bool) or not isinstance(measurement_ns, int)
        ):
            raise TypeError("measurement_ns must be an integer monotonic timestamp")
        if not isinstance(measurement_observed, bool):
            raise TypeError("measurement_observed must be bool")
        if not measurement_observed and target is None:
            raise ValueError("an unobserved measurement requires a predicted target")
        source_ns = published_ns if measurement_ns is None else measurement_ns
        if source_ns < 0:
            raise ValueError("measurement_ns cannot be negative")
        error_x, error_y = _target_error_pixels(target, frame_shape, self.config)
        with self._state_lock:
            if self._calibration_token is not None:
                raise MakcuError(
                    "normal MAKCU aiming is unavailable during calibration"
                )
            if self._latest_source_ns and source_ns < self._latest_source_ns:
                raise ValueError("measurement_ns must not move backwards")
            if not measurement_observed and not self._measurement_target_present:
                raise ValueError(
                    "a predicted target requires a prior observed target"
                )
            velocity_x = self._latest_velocity_x
            velocity_y = self._latest_velocity_y
            same_time_geometry_changed = (
                self._latest_measurement_ns
                and source_ns == self._latest_measurement_ns
                and measurement_observed
                and (
                    error_x != self._measurement_error_x
                    or error_y != self._measurement_error_y
                )
            )
            if (
                measurement_observed
                and target is not None
                and self._measurement_target_present
                and self._latest_measurement_ns
                and source_ns > self._latest_measurement_ns
            ):
                delta_t = (source_ns - self._latest_measurement_ns) / 1_000_000_000
                delta_x = error_x - self._measurement_error_x
                delta_y = error_y - self._measurement_error_y
                sample_discontinuity = (
                    delta_t > TARGET_STALE_SECONDS
                    or abs(delta_x) > ERROR_JUMP_RESET_PIXELS
                    or abs(delta_y) > ERROR_JUMP_RESET_PIXELS
                )
                if sample_discontinuity:
                    velocity_x = 0.0
                    velocity_y = 0.0
                else:
                    velocity_x = delta_x / delta_t
                    velocity_y = delta_y / delta_t
                    velocity_x = min(
                        max(velocity_x, -MAX_TRACKED_VELOCITY_PX_PER_SEC),
                        MAX_TRACKED_VELOCITY_PX_PER_SEC,
                    )
                    velocity_y = min(
                        max(velocity_y, -MAX_TRACKED_VELOCITY_PX_PER_SEC),
                        MAX_TRACKED_VELOCITY_PX_PER_SEC,
                    )
            elif measurement_observed and (
                target is None
                or not self._measurement_target_present
                or same_time_geometry_changed
            ):
                velocity_x = 0.0
                velocity_y = 0.0
            # Every publication changes the normal-motion generation, even a
            # tracker prediction.  A worker which already computed from an
            # older snapshot must re-evaluate instead of committing stale
            # motion after this call returns.
            self._normal_motion_generation += 1
            self._latest_target = target
            self._latest_frame_shape = frame_shape
            self._latest_active = bool(active)
            self._latest_update_ns = published_ns
            self._latest_source_ns = source_ns
            self._latest_measurement_observed = measurement_observed
            if measurement_observed:
                self._latest_measurement_ns = source_ns
                self._measurement_target_present = target is not None
                self._measurement_error_x = error_x
                self._measurement_error_y = error_y
            self._latest_velocity_x = velocity_x
            self._latest_velocity_y = velocity_y
            self._latest_sample_id += 1
        if self._output_thread is None:
            self._output_tick(1.0 / REFERENCE_CONTROL_HZ)

    def _run_output_loop(self) -> None:
        period_ns = max(1, round(1_000_000_000 / self.config.output_hz))
        previous_ns = time.perf_counter_ns()
        next_ns = previous_ns
        while not self._stop_event.is_set():
            now_ns = time.perf_counter_ns()
            if now_ns < next_ns:
                self._stop_event.wait((next_ns - now_ns) / 1_000_000_000)
                continue
            elapsed = min(max((now_ns - previous_ns) / 1_000_000_000, 0.0), 0.01)
            try:
                self._output_tick(elapsed, now_ns=now_ns)
            except MakcuError as exc:
                with self._state_lock:
                    if self._calibration_token is not None:
                        self._abort_calibration_locked(
                            f"calibration output worker failed: {exc}"
                        )
                self._worker_error = exc
                return
            previous_ns = now_ns
            next_ns += period_ns
            if now_ns - next_ns > period_ns * 4:
                next_ns = now_ns + period_ns

    def _output_calibration_tick(
        self,
        elapsed: float,
        *,
        current_ns: int,
        button_pressed: bool,
        token: object,
    ) -> None:
        """Emit at most one bounded pulse step while every live gate is valid."""

        with self._state_lock:
            if self._calibration_token is not token:
                return
            if not self._calibration_pending_counts:
                return
            if self._calibration_abort_reason is not None:
                self._clear_calibration_pending_locked()
                return
            if (
                self._calibration_pending_token is not token
                or self._calibration_lease_token is not token
            ):
                self._abort_calibration_locked("calibration token state changed")
                return
            if not button_pressed:
                self._abort_calibration_locked("physical activation was released")
                return
            if not self._calibration_lease_valid:
                self._abort_calibration_locked("calibration lease is invalid")
                return
            if (
                self._calibration_hold_transition_sequence is None
                or self._activation_transition_sequence
                != self._calibration_hold_transition_sequence
            ):
                self._abort_calibration_locked(
                    "physical activation changed during calibration pulse"
                )
                return
            lease_age_ns = current_ns - self._calibration_lease_measurement_ns
            maximum_lease_age_ns = round(
                CALIBRATION_LEASE_MAX_AGE_SECONDS * 1_000_000_000
            )
            if lease_age_ns < 0:
                self._abort_calibration_locked(
                    "calibration lease timestamp is in the future"
                )
                return
            if lease_age_ns > maximum_lease_age_ns:
                self._abort_calibration_locked("calibration lease expired")
                return

            rate = self._calibration_pending_rate
            bounded_elapsed = min(max(float(elapsed), 0.0), 0.01)
            available = self._calibration_fractional_counts + rate * bounded_elapsed
            whole_counts = math.floor(available)
            if whole_counts <= 0:
                self._calibration_fractional_counts = available
                return
            # A delayed scheduler tick may not collapse several milliseconds of
            # requested motion into one large command. Discard that integer
            # backlog and retain only the true fractional remainder.
            self._calibration_fractional_counts = available - whole_counts
            rate_step_limit = max(
                1,
                math.ceil(rate / CALIBRATION_OUTPUT_HZ),
            )
            step = min(
                whole_counts,
                rate_step_limit,
                CALIBRATION_MAX_STEP_COUNTS,
                abs(self._calibration_pending_counts),
            )
            signed_step = step if self._calibration_pending_counts > 0 else -step
            axis = self._calibration_pending_axis
            delta_x = signed_step if axis == "x" else 0
            delta_y = signed_step if axis == "y" else 0
            if not delta_x and not delta_y:
                self._abort_calibration_locked("calibration pulse axis is invalid")
                return

            # Hold the state lock through the physical write. enter/exit/stop
            # therefore cannot return while an earlier calibration decision is
            # still capable of reaching the board.
            try:
                self._command(f"km.move({delta_x},{delta_y})")
            except MakcuError as exc:
                self._abort_calibration_locked(
                    f"calibration movement write failed: {exc}"
                )
                raise

            self._calibration_pending_counts -= signed_step
            self._calibration_emitted_x += delta_x
            self._calibration_emitted_y += delta_y
            self._calibration_emitted_abs_counts += step
            self._calibration_movement_commands += 1
            if self._calibration_first_emitted_ns is None:
                self._calibration_first_emitted_ns = current_ns
            self._calibration_last_emitted_ns = current_ns
            # A session can contain at most 2,400 successful one-count writes,
            # so retaining every immutable event is itself hard-bounded by the
            # session motion budget. Failed writes never reach this point.
            self._calibration_emitted_events.append((current_ns, delta_x, delta_y))
            if not self._calibration_pending_counts:
                self._clear_calibration_pending_locked()
            with self._telemetry_lock:
                self._telemetry_movement_commands += 1
                self._telemetry_emitted_x += delta_x
                self._telemetry_emitted_y += delta_y
                self._telemetry_emitted_abs_x += abs(delta_x)
                self._telemetry_emitted_abs_y += abs(delta_y)

    def _output_calibrated_tick(
        self,
        elapsed: float,
        *,
        current_ns: int,
        button_pressed: bool,
        target: Detection | None,
        frame_shape: tuple[int, int, int],
        active: bool,
        measurement_observed: bool,
        source_ns: int,
        sample_id: int,
        generation: int,
        decision_started_ns: int,
    ) -> None:
        """Run the calibrated plant-aware path without touching legacy PI state."""

        controller = self._calibrated_controller
        if controller is None:  # pragma: no cover - caller guards this branch
            return
        with self._calibrated_lock:
            new_sample = sample_id != self._calibrated_processed_sample_id
            observation = None
            target_lost = False
            if new_sample:
                self._calibrated_processed_sample_id = sample_id
                if measurement_observed and target is not None:
                    error_x, error_y = _target_error_pixels(
                        target,
                        frame_shape,
                        self.config,
                    )
                    observation = ScreenErrorObservation(
                        source_ns,
                        error_x,
                        error_y,
                    )
                elif measurement_observed:
                    # This is explicit real detector/tracker loss. A synthetic
                    # grace sample instead supplies neither observation nor
                    # loss and can bridge only the core's freshness interval.
                    target_lost = True
            try:
                output = controller.step(
                    current_ns,
                    engaged=bool(active and button_pressed),
                    observation=observation,
                    target_lost=target_lost,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                self._fractional_x = 0.0
                self._fractional_y = 0.0
                raise MakcuError(
                    f"calibrated MAKCU control failed closed: {exc}"
                ) from exc

            # Calibration changes the control law, not the visible/user-owned
            # safety envelope.  Never exceed the configured legacy-equivalent
            # X rate or its explicit vertical fraction, even when a supplied
            # calibrated core was constructed with broader defaults.
            visible_rate_x = self.config.max_step * REFERENCE_CONTROL_HZ
            visible_rate_y = visible_rate_x * self.config.vertical_rate_ratio
            rate_limit_x = min(
                controller.config.maximum_rate_x_counts_per_second,
                visible_rate_x,
            )
            rate_limit_y = min(
                controller.config.maximum_rate_y_counts_per_second,
                visible_rate_y,
            )
            bounded_rate_x = min(
                max(output.rate_x_counts_per_second, -rate_limit_x),
                rate_limit_x,
            )
            bounded_rate_y = min(
                max(output.rate_y_counts_per_second, -rate_limit_y),
                rate_limit_y,
            )
            if (
                bounded_rate_x != output.rate_x_counts_per_second
                or bounded_rate_y != output.rate_y_counts_per_second
            ):
                output = CalibratedControlOutput(
                    timestamp_ns=output.timestamp_ns,
                    rate_x_counts_per_second=bounded_rate_x,
                    rate_y_counts_per_second=bounded_rate_y,
                    target_velocity_x_pixels_per_second=(
                        output.target_velocity_x_pixels_per_second
                    ),
                    target_velocity_y_pixels_per_second=(
                        output.target_velocity_y_pixels_per_second
                    ),
                    projected_error_x_pixels=output.projected_error_x_pixels,
                    projected_error_y_pixels=output.projected_error_y_pixels,
                    valid=output.valid,
                    saturated_x=(
                        output.saturated_x
                        or bounded_rate_x != output.rate_x_counts_per_second
                    ),
                    saturated_y=(
                        output.saturated_y
                        or bounded_rate_y != output.rate_y_counts_per_second
                    ),
                    reset_reason=output.reset_reason,
                )
            self._calibrated_last_output = output
            if not output.valid:
                self._fractional_x = 0.0
                self._fractional_y = 0.0
                return

            bounded_elapsed = min(max(float(elapsed), 0.0), 0.01)
            self._fractional_x += bounded_rate_x * bounded_elapsed
            self._fractional_y += bounded_rate_y * bounded_elapsed
            raw_delta_x = math.trunc(self._fractional_x)
            raw_delta_y = math.trunc(self._fractional_y)
            # A delayed scheduler tick may not collapse its whole backlog into
            # one physical command.  Discard clamped-away integer backlog.
            tick_limit_x = max(
                1,
                math.ceil(rate_limit_x / self.config.output_hz),
            )
            tick_limit_y = max(
                1,
                math.ceil(rate_limit_y / self.config.output_hz),
            )
            delta_x = min(max(raw_delta_x, -tick_limit_x), tick_limit_x)
            delta_y = min(max(raw_delta_y, -tick_limit_y), tick_limit_y)
            self._fractional_x -= raw_delta_x
            self._fractional_y -= raw_delta_y

            with self._state_lock:
                decision_age_ns = max(
                    time.perf_counter_ns() - decision_started_ns,
                    0,
                )
                commit_ns = current_ns + decision_age_ns
                commit_button_pressed = self._activation_pressed_locked(commit_ns)
                publication_age_ns = commit_ns - self._latest_update_ns
                measurement_age_ns = commit_ns - self._latest_measurement_ns
                publication_fresh = bool(
                    self._latest_update_ns
                    and (
                        publication_age_ns < 0
                        or publication_age_ns
                        <= round(TARGET_STALE_SECONDS * 1_000_000_000)
                    )
                )
                measurement_fresh = bool(
                    self._latest_measurement_ns
                    and (
                        measurement_age_ns < 0
                        or measurement_age_ns
                        <= round(
                            controller.config.stale_after_seconds
                            * 1_000_000_000
                        )
                    )
                )
                commit_authorized = bool(
                    not self._stop_event.is_set()
                    and self._calibration_token is None
                    and self._normal_motion_generation == generation
                    and self._latest_sample_id == sample_id
                    and self._latest_active
                    and self._latest_target is not None
                    and commit_button_pressed
                    and publication_fresh
                    and measurement_fresh
                    and decision_age_ns
                    <= round(
                        controller.config.stale_after_seconds * 1_000_000_000
                    )
                )
                if not commit_authorized:
                    self._fractional_x = 0.0
                    self._fractional_y = 0.0
                    self._reset_calibrated_tracking_locked()
                    return
                if not delta_x and not delta_y:
                    return
                emitted = EmittedMouseCommand(current_ns, delta_x, delta_y)
                try:
                    controller.preflight_emitted(emitted)
                except (RuntimeError, TypeError, ValueError) as exc:
                    self._fractional_x = 0.0
                    self._fractional_y = 0.0
                    self._reset_calibrated_control_locked()
                    raise MakcuError(
                        "calibrated MAKCU command preflight failed closed: "
                        f"{exc}"
                    ) from exc
                self._command(f"km.move({delta_x},{delta_y})")
                try:
                    controller.record_emitted(emitted)
                except (RuntimeError, TypeError, ValueError) as exc:
                    self._fractional_x = 0.0
                    self._fractional_y = 0.0
                    raise MakcuError(
                        "calibrated MAKCU command accounting failed closed: "
                        f"{exc}"
                    ) from exc
                with self._telemetry_lock:
                    self._telemetry_movement_commands += 1
                    self._telemetry_emitted_x += delta_x
                    self._telemetry_emitted_y += delta_y
                    self._telemetry_emitted_abs_x += abs(delta_x)
                    self._telemetry_emitted_abs_y += abs(delta_y)

    def _output_tick(self, elapsed: float, *, now_ns: int | None = None) -> None:
        decision_started_ns = time.perf_counter_ns()
        current_ns = decision_started_ns if now_ns is None else now_ns
        self._read_buttons(now_ns=current_ns)
        button_pressed = self._activation_pressed_at(current_ns)
        with self._state_lock:
            calibration_token = self._calibration_token
        if calibration_token is not None:
            self._reset_calibrated_control()
            with self._telemetry_lock:
                self._telemetry_output_ticks += 1
                self._telemetry_button_pressed_ticks += int(button_pressed)
            self._output_calibration_tick(
                elapsed,
                current_ns=current_ns,
                button_pressed=button_pressed,
                token=calibration_token,
            )
            return
        with self._state_lock:
            target = self._latest_target
            frame_shape = self._latest_frame_shape
            active = self._latest_active
            updated_ns = self._latest_update_ns
            measurement_ns = self._latest_measurement_ns
            source_ns = self._latest_source_ns
            measurement_observed = self._latest_measurement_observed
            velocity_x = self._latest_velocity_x
            velocity_y = self._latest_velocity_y
            sample_id = self._latest_sample_id
            generation = self._normal_motion_generation
        publication_stale = (
            updated_ns == 0
            or (current_ns - updated_ns) / 1_000_000_000 > TARGET_STALE_SECONDS
        )
        measurement_stale = (
            measurement_ns > 0
            and current_ns >= measurement_ns
            and (current_ns - measurement_ns) / 1_000_000_000
            > TARGET_STALE_SECONDS
        )
        stale = publication_stale or measurement_stale
        target_present = target is not None
        fresh_target = target_present and not stale
        authorized = active and button_pressed and fresh_target
        with self._telemetry_lock:
            self._telemetry_output_ticks += 1
            self._telemetry_active_input_ticks += int(active)
            self._telemetry_button_pressed_ticks += int(button_pressed)
            self._telemetry_target_present_ticks += int(target_present)
            self._telemetry_fresh_target_ticks += int(fresh_target)
            self._telemetry_authorized_ticks += int(authorized)
        if self._calibrated_controller is not None:
            self._output_calibrated_tick(
                elapsed,
                current_ns=current_ns,
                button_pressed=button_pressed,
                target=target,
                frame_shape=frame_shape,
                active=active,
                measurement_observed=measurement_observed,
                source_ns=source_ns,
                sample_id=sample_id,
                generation=generation,
                decision_started_ns=decision_started_ns,
            )
            return
        if not authorized:
            # Missing target, physical release, and inactive input all revoke
            # the same pursuit history. Count only a real nonzero clear, so a
            # contiguous missing interval cannot inflate this at 1 kHz.
            self._record_pursuit_reset()
            self._fractional_x = 0.0
            self._fractional_y = 0.0
            self._smoothed_rate_x = 0.0
            self._smoothed_rate_y = 0.0
            self._control_error_x = 0.0
            self._control_error_y = 0.0
            self._pursuit_correction_x = 0.0
            self._pursuit_correction_y = 0.0
            self._pursuit_measurement_ns = measurement_ns
            self._processed_sample_id = sample_id
            return
        rate_x, rate_y = _makcu_target_rate(target, frame_shape, self.config)
        if self._output_thread is not None:
            error_x, error_y = _target_error_pixels(target, frame_shape, self.config)
            previous_error_x = self._control_error_x
            previous_error_y = self._control_error_y

            # Reset momentum once per new detector sample, never once per 1 kHz
            # output tick. Velocity is computed from source-frame timestamps in
            # update(), so a held sample does not turn into a 1 ms derivative
            # burst followed by zero.
            new_sample = sample_id != self._processed_sample_id
            new_measurement = new_sample and measurement_observed
            pursuit_elapsed = 0.0
            if new_sample:
                if (
                    new_measurement
                    and self._pursuit_measurement_ns
                    and source_ns > self._pursuit_measurement_ns
                ):
                    pursuit_elapsed = min(
                        (source_ns - self._pursuit_measurement_ns)
                        / 1_000_000_000,
                        0.025,
                    )
                # Synthetic tracker output advances the integration clock but
                # contributes no area.  A reacquired measurement therefore
                # cannot retroactively integrate across an unobserved gap.
                self._pursuit_measurement_ns = source_ns
            sample_jump = new_sample and (
                abs(error_x - previous_error_x) > ERROR_JUMP_RESET_PIXELS
                or abs(error_y - previous_error_y) > ERROR_JUMP_RESET_PIXELS
            )
            if sample_jump:
                self._record_pursuit_reset()
                self._smoothed_rate_x = 0.0
                self._smoothed_rate_y = 0.0
                self._fractional_x = 0.0
                self._fractional_y = 0.0
                self._pursuit_correction_x = 0.0
                self._pursuit_correction_y = 0.0
                pursuit_elapsed = 0.0

            self._control_error_x = error_x
            self._control_error_y = error_y
            self._processed_sample_id = sample_id

            # Predict ahead by control lead plus bounded sample age so the
            # control point lands where the moving head is now, not where the
            # detector last saw it.
            sample_age_s = max((current_ns - source_ns) / 1_000_000_000, 0.0)
            lead_s = self.config.prediction_lead_seconds + min(
                sample_age_s,
                self.config.max_sample_age_lead_seconds,
            )
            predicted_x = error_x + velocity_x * lead_s
            predicted_y = error_y + velocity_y * lead_s

            limit = float(self.config.max_step)

            def pursuit_correction(
                error: float,
                predicted: float,
                velocity: float,
                accumulated: float,
                correction_limit: float,
            ) -> tuple[float, float]:
                in_deadzone = abs(error) <= self.config.deadzone_pixels
                base = 0.0 if in_deadzone else error * self.config.strength
                predicted_motion = (predicted - error) * self.config.strength
                derivative_motion = velocity * self.config.derivative_damping_seconds
                motion = predicted_motion + derivative_motion

                # Screen-error velocity contains both independent target
                # motion and camera motion caused by our own prior commands.
                # It is useful as feed-forward only while the visual error is
                # growing in the same direction; it must never brake the
                # measured positional correction.
                if not in_deadzone and error * motion > 0.0:
                    base += motion

                integral_limit = correction_limit * MAX_PURSUIT_INTEGRAL_RATIO
                if pursuit_elapsed > 0.0 and in_deadzone:
                    accumulated *= math.exp(
                        -pursuit_elapsed / PURSUIT_DEADZONE_LEAK_TIME_SECONDS
                    )
                elif pursuit_elapsed > 0.0:
                    increment = (
                        error
                        * self.config.strength
                        * pursuit_elapsed
                        / PURSUIT_INTEGRAL_TIME_SECONDS
                    )
                    unsaturated = base + accumulated
                    # Conditional integration is anti-windup: an error may
                    # unwind a saturated rate, but never build a hidden
                    # same-direction backlog behind the configured clamp.
                    if increment > 0.0 and unsaturated < correction_limit:
                        accumulated += min(
                            increment,
                            correction_limit - unsaturated,
                        )
                    elif increment < 0.0 and unsaturated > -correction_limit:
                        accumulated += max(
                            increment,
                            -correction_limit - unsaturated,
                        )
                accumulated = min(max(accumulated, -integral_limit), integral_limit)
                correction = _combine_pursuit_correction(
                    error,
                    base,
                    accumulated,
                    in_deadzone=in_deadzone,
                )
                return correction, accumulated

            correction_x, self._pursuit_correction_x = pursuit_correction(
                error_x,
                predicted_x,
                velocity_x,
                self._pursuit_correction_x,
                limit,
            )
            correction_y, self._pursuit_correction_y = pursuit_correction(
                error_y,
                predicted_y,
                velocity_y,
                self._pursuit_correction_y,
                limit * self.config.vertical_rate_ratio,
            )
            if new_measurement:
                self._record_control_sample(
                    error_x,
                    error_y,
                    self._pursuit_correction_x,
                    self._pursuit_correction_y,
                    correction_x,
                    correction_y,
                    limit,
                    limit * self.config.vertical_rate_ratio,
                )
            rate_x = min(max(correction_x, -limit), limit) * REFERENCE_CONTROL_HZ
            rate_y = min(max(correction_y, -limit), limit) * REFERENCE_CONTROL_HZ
            vertical_limit = (
                limit * REFERENCE_CONTROL_HZ * self.config.vertical_rate_ratio
            )
            rate_y = min(max(rate_y, -vertical_limit), vertical_limit)

            # Interpret the user value at the historical 60 Hz reference rate,
            # then convert it to the actual tick duration. Response therefore
            # stays the same at 100, 1000, or 2000 Hz.
            alpha = 1.0 - math.pow(
                1.0 - self.config.smoothing_alpha,
                max(elapsed, 0.0) * REFERENCE_CONTROL_HZ,
            )
            self._smoothed_rate_x += (rate_x - self._smoothed_rate_x) * alpha
            self._smoothed_rate_y += (rate_y - self._smoothed_rate_y) * alpha
            rate_x = self._smoothed_rate_x
            rate_y = self._smoothed_rate_y
        self._fractional_x += rate_x * elapsed
        self._fractional_y += rate_y * elapsed
        raw_delta_x = math.trunc(self._fractional_x)
        raw_delta_y = math.trunc(self._fractional_y)
        delta_x = raw_delta_x
        delta_y = raw_delta_y
        if self._output_thread is not None:
            tick_limit = max(
                1,
                int(
                    round(
                        self.config.max_step
                        * elapsed
                        * REFERENCE_CONTROL_HZ
                    )
                ),
            )
            vertical_tick_limit = max(
                1,
                math.ceil(
                    self.config.max_step
                    * self.config.vertical_rate_ratio
                    * elapsed
                    * REFERENCE_CONTROL_HZ
                ),
            )
            vertical_tick_limit = min(tick_limit, vertical_tick_limit)
            delta_x = min(max(delta_x, -tick_limit), tick_limit)
            delta_y = min(max(delta_y, -vertical_tick_limit), vertical_tick_limit)
        # Retain only the true sub-count remainder. If a delayed scheduler tick
        # hits a safety clamp, do not replay the discarded integer backlog on
        # later ticks after the visual error may already have changed.
        self._fractional_x -= raw_delta_x
        self._fractional_y -= raw_delta_y
        if delta_x or delta_y:
            # Calibration may be entered from the detector thread while this
            # output tick is computing. Recheck exclusivity while holding the
            # state lock through the write, so no normal command can land after
            # enter_calibration_mode() returns.
            with self._state_lock:
                if self._calibration_token is not None:
                    self._clear_normal_motion_locked()
                    return
                self._command(f"km.move({delta_x},{delta_y})")
                with self._telemetry_lock:
                    self._telemetry_movement_commands += 1
                    self._telemetry_emitted_x += delta_x
                    self._telemetry_emitted_y += delta_y
                    self._telemetry_emitted_abs_x += abs(delta_x)
                    self._telemetry_emitted_abs_y += abs(delta_y)

    def _disarm_for_shutdown(self) -> None:
        """Remove every software authorization before touching blocking I/O."""

        with self._state_lock:
            if self._calibration_token is not None:
                self._abort_calibration_locked(
                    "MAKCU controller stopped during calibration",
                    deactivate=True,
                )
            self._button_mask = 0
            self._button_state_known = False
            self._activation_started_ns = 0
            self._activation_requires_release = True
            self._clear_normal_motion_locked()
        self._reset_calibrated_control()

    def _close_connection(self, connection: Any, closed: threading.Event) -> None:
        """Cancel pending pyserial operations and close from a daemon breaker."""

        # Serialize native close itself so the graceful and breaker threads can
        # never race two close calls.  The breaker can still cancel/close a
        # write stuck under _serial_lock because that lock is intentionally not
        # involved here.
        with self._connection_close_lock:
            # PySerial documents cancel_read/cancel_write for calls from another
            # thread.  Not every serial-compatible test/driver implements them,
            # so absence and disconnect errors are harmless; close remains the
            # authoritative operation.
            for operation_name in ("cancel_read", "cancel_write"):
                operation = getattr(connection, operation_name, None)
                if not callable(operation):
                    continue
                try:
                    operation()
                except Exception:  # noqa: BLE001 - contain driver failures
                    pass
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - stop() reports an incomplete close
                return
        closed.set()

    def _graceful_connection_shutdown(
        self,
        connection: Any,
        closed: threading.Event,
    ) -> None:
        """Disable button telemetry and close after the output worker exits."""

        try:
            with self._serial_lock:
                connection.write(b"km.buttons(0)\r")
                connection.flush()
        except Exception:  # noqa: BLE001 - closing is the bounded fallback
            # Closing the port is the fail-closed fallback when the optional
            # stream-disable command cannot be delivered.
            pass
        with self._connection_close_lock:
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - stop() reports an incomplete close
                return
        closed.set()

    def _start_shutdown_thread(
        self,
        target: Callable[[], None],
        *,
        name: str,
    ) -> threading.Thread:
        thread = threading.Thread(target=target, name=name, daemon=True)
        self._shutdown_threads.append(thread)
        thread.start()
        return thread

    @staticmethod
    def _wait_for_threads(
        threads: Iterable[threading.Thread],
        *,
        deadline: float,
    ) -> None:
        """Give several cooperating threads fair slices of one shared deadline."""

        candidates = tuple(dict.fromkeys(threads))
        while True:
            live = [
                thread
                for thread in candidates
                if thread is not threading.current_thread() and thread.is_alive()
            ]
            remaining = deadline - time.monotonic()
            if not live or remaining <= 0.0:
                return
            slice_seconds = min(0.01, remaining / len(live))
            for thread in live:
                thread.join(timeout=max(0.0, slice_seconds))

    def stop(self) -> None:
        """Stop output and serial I/O within ``stop_timeout`` or fail explicitly.

        The timeout covers worker exit, the normal ``km.buttons(0)`` command,
        cancellation, and serial close.  If native I/O remains stuck, this
        method raises :class:`MakcuError` and deliberately retains references to
        every live worker instead of claiming the controller has stopped.
        """

        connection = self._serial
        if connection is None:
            return
        deadline = time.monotonic() + self._stop_timeout
        self._stop_event.set()
        self._disarm_for_shutdown()
        self._shutdown_threads = [
            thread for thread in self._shutdown_threads if thread.is_alive()
        ]
        worker = self._output_thread
        worker_grace = min(
            MAKCU_STOP_PHASE_GRACE_SECONDS,
            self._stop_timeout / 3.0,
        )
        if (
            worker is not None
            and worker is not threading.current_thread()
            and worker.is_alive()
        ):
            worker.join(timeout=worker_grace)

        closed = threading.Event()
        if worker is None or not worker.is_alive():
            graceful = self._start_shutdown_thread(
                lambda: self._graceful_connection_shutdown(connection, closed),
                name="makcu-graceful-shutdown",
            )
            io_grace = min(
                MAKCU_STOP_PHASE_GRACE_SECONDS,
                max(0.0, deadline - time.monotonic()) / 2.0,
            )
            graceful.join(timeout=io_grace)

        # A live output worker or stuck graceful write may hold _serial_lock.
        # Never wait on that lock from the caller: pyserial cancellation/close
        # is specifically performed from another thread to break pending I/O.
        if not closed.is_set():
            self._start_shutdown_thread(
                lambda: self._close_connection(connection, closed),
                name="makcu-forced-serial-close",
            )

        self._wait_for_threads(
            (
                *((worker,) if worker is not None else ()),
                *self._shutdown_threads,
            ),
            deadline=deadline,
        )
        live_worker = worker is not None and worker.is_alive()
        live_shutdown = [
            thread for thread in self._shutdown_threads if thread.is_alive()
        ]
        if live_worker or live_shutdown or not closed.is_set():
            problems: list[str] = []
            if live_worker:
                problems.append("the 1 kHz output worker is still running")
            if live_shutdown:
                problems.append("serial cancellation/close is still running")
            if not closed.is_set():
                problems.append("the serial connection did not confirm close")
            error = MakcuError(
                "MAKCU shutdown did not finish within "
                f"{self._stop_timeout:g} seconds; output was disarmed, but "
                + "; ".join(problems)
            )
            self._worker_error = error
            # Keep _output_thread, _serial, connected_port, and live shutdown
            # thread references truthful so a caller cannot mistake this for a
            # completed close or silently restart over the old worker.
            raise error

        self._output_thread = None
        self._shutdown_threads.clear()
        self._serial = None
        self.connected_port = None
        self._identity_token = None
        with self._state_lock:
            self._button_mask = 0
            self._button_state_known = False
            self._activation_started_ns = 0
            self._activation_requires_release = False
            self._latest_target = None
            self._latest_active = False
            self._latest_update_ns = 0
        self._fractional_x = 0.0
        self._fractional_y = 0.0
        self._smoothed_rate_x = 0.0
        self._smoothed_rate_y = 0.0
        self._latest_measurement_ns = 0
        self._latest_source_ns = 0
        self._latest_measurement_observed = True
        self._measurement_target_present = False
        self._latest_velocity_x = 0.0
        self._latest_velocity_y = 0.0
        self._latest_sample_id = 0
        self._processed_sample_id = 0
        self._reset_calibrated_control()
        self._control_error_x = 0.0
        self._control_error_y = 0.0
        self._measurement_error_x = 0.0
        self._measurement_error_y = 0.0
        self._pursuit_correction_x = 0.0
        self._pursuit_correction_y = 0.0
        self._pursuit_measurement_ns = 0
        self._button_parser.reset()
