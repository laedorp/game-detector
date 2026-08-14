"""Detection-driven relative mouse output for a MAKCU passthrough board."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import math
import threading
import time
from typing import Any

from detection.types import Detection
from .controller import DEFAULT_HEAD_RATIO, head_target_point

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
CLOSE_RANGE_SLOWDOWN_PIXELS = 130.0
MIN_CLOSE_RANGE_SCALE = 0.72
THREADED_RATE_SMOOTHING_ALPHA = 0.82
PREDICTION_LEAD_SECONDS = 0.028
DERIVATIVE_DAMPING_SECONDS = 0.016
MAX_SAMPLE_AGE_LEAD_SECONDS = 0.055
MOTION_SLOWDOWN_BYPASS_PX_PER_SEC = 95.0
MAX_TRACKED_VELOCITY_PX_PER_SEC = 2600.0
ERROR_JUMP_RESET_PIXELS = 240.0
LOW_CONFIDENCE_GAIN_FLOOR = 0.45
MAX_VERTICAL_RATE_RATIO = 0.48
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

    def feed(self, data: bytes) -> tuple[int, ...]:
        bare_mask = (
            not self._framed_mode
            and not self._pending
            and len(data) == 1
            and data[0] <= 0x1F
        )
        if data:
            self._pending.extend(data)
        masks: list[int] = []
        while self._pending:
            index = self._pending.find(_BUTTON_FRAME_PREFIX)
            if index < 0:
                # Field firmware can emit a naked one-byte five-bit mask. Only
                # accept a standalone byte received before any framed traffic.
                # Once a framed stream is observed, split CR/LF/prompt bytes
                # must never be reclassified as physical mouse state.
                if bare_mask and len(self._pending) == 1:
                    masks.append(self._pending[0])
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
                masks.append(value)
                del self._pending[: value_index + 1]
                continue

            # Textual command response (for example ``km.buttons(1)``) or an
            # unrelated ``km.*`` message. Drop this prefix and search for the
            # next structured event without interpreting its control bytes.
            del self._pending[: len(_BUTTON_FRAME_PREFIX)]

        if len(self._pending) > 256:
            # A malformed/noisy device must not grow this buffer forever.
            del self._pending[:-32]
        return tuple(masks)


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
        self.config = config or MakcuAimConfig()
        self._serial_factory = serial_factory
        self._ports_provider = ports_provider
        self._sleep = sleep
        self._threaded_output = threaded_output
        self._serial: Any | None = None
        self._button_mask = 0
        self._button_state_known = False
        self._activation_started_ns = 0
        self._activation_requires_release = False
        self._button_parser = _ButtonStreamParser()
        self._serial_lock = threading.Lock()
        self._connection_close_lock = threading.Lock()
        self._state_lock = threading.Lock()
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
        self._latest_velocity_x = 0.0
        self._latest_velocity_y = 0.0
        self._latest_sample_id = 0
        self._processed_sample_id = 0
        self._control_error_x = 0.0
        self._control_error_y = 0.0
        self._measurement_error_x = 0.0
        self._measurement_error_y = 0.0
        self.connected_port: str | None = None

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
        self._button_mask = 0
        self._button_state_known = False
        self._activation_started_ns = 0
        self._activation_requires_release = False
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
        self._latest_velocity_x = 0.0
        self._latest_velocity_y = 0.0
        self._latest_sample_id = 0
        self._processed_sample_id = 0
        self._control_error_x = 0.0
        self._control_error_y = 0.0
        self._measurement_error_x = 0.0
        self._measurement_error_y = 0.0
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
        for value in self._button_parser.feed(data):
            with self._state_lock:
                was_pressed = bool(
                    self._button_state_known
                    and self._button_mask & (1 << self.config.activation_button)
                )
                self._button_mask = value & 0x1F
                self._button_state_known = True
                is_pressed = bool(
                    self._button_mask & (1 << self.config.activation_button)
                )
                if not is_pressed:
                    self._activation_started_ns = 0
                    self._activation_requires_release = False
                elif not was_pressed and not self._activation_requires_release:
                    self._activation_started_ns = current_ns
                # A valid release report is authoritative. Do not keep movement
                # active after the physical button is released.

    def _activation_pressed_at(self, now_ns: int) -> bool:
        with self._state_lock:
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
                return False
            return True

    @property
    def activation_pressed(self) -> bool:
        return self._activation_pressed_at(time.perf_counter_ns())

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
        source_ns = published_ns if measurement_ns is None else measurement_ns
        if source_ns < 0:
            raise ValueError("measurement_ns cannot be negative")
        error_x, error_y = _target_error_pixels(target, frame_shape, self.config)
        with self._state_lock:
            if self._latest_measurement_ns and source_ns < self._latest_measurement_ns:
                raise ValueError("measurement_ns must not move backwards")
            velocity_x = self._latest_velocity_x
            velocity_y = self._latest_velocity_y
            same_time_geometry_changed = (
                self._latest_measurement_ns
                and source_ns == self._latest_measurement_ns
                and (
                    error_x != self._measurement_error_x
                    or error_y != self._measurement_error_y
                )
            )
            if (
                target is not None
                and self._latest_target is not None
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
            elif target is None or self._latest_target is None or same_time_geometry_changed:
                velocity_x = 0.0
                velocity_y = 0.0
            self._latest_target = target
            self._latest_frame_shape = frame_shape
            self._latest_active = bool(active)
            self._latest_update_ns = published_ns
            self._latest_measurement_ns = source_ns
            self._latest_velocity_x = velocity_x
            self._latest_velocity_y = velocity_y
            self._measurement_error_x = error_x
            self._measurement_error_y = error_y
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
                self._worker_error = exc
                return
            previous_ns = now_ns
            next_ns += period_ns
            if now_ns - next_ns > period_ns * 4:
                next_ns = now_ns + period_ns

    def _output_tick(self, elapsed: float, *, now_ns: int | None = None) -> None:
        current_ns = time.perf_counter_ns() if now_ns is None else now_ns
        self._read_buttons(now_ns=current_ns)
        button_pressed = self._activation_pressed_at(current_ns)
        with self._state_lock:
            target = self._latest_target
            frame_shape = self._latest_frame_shape
            active = self._latest_active
            updated_ns = self._latest_update_ns
            measurement_ns = self._latest_measurement_ns
            velocity_x = self._latest_velocity_x
            velocity_y = self._latest_velocity_y
            sample_id = self._latest_sample_id
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
        if not active or not button_pressed or target is None or stale:
            self._fractional_x = 0.0
            self._fractional_y = 0.0
            self._smoothed_rate_x = 0.0
            self._smoothed_rate_y = 0.0
            self._control_error_x = 0.0
            self._control_error_y = 0.0
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
            if new_sample and (
                abs(error_x - previous_error_x) > ERROR_JUMP_RESET_PIXELS
                or abs(error_y - previous_error_y) > ERROR_JUMP_RESET_PIXELS
            ):
                self._smoothed_rate_x = 0.0
                self._smoothed_rate_y = 0.0
                self._fractional_x = 0.0
                self._fractional_y = 0.0

            # Only fully settle when near center and the target is not moving,
            # so we avoid trailing a moving head.
            settle_band = self.config.deadzone_pixels * 1.25
            velocity_mag = math.hypot(velocity_x, velocity_y)
            if (
                abs(error_x) <= settle_band
                and abs(error_y) <= settle_band
                and velocity_mag <= 120.0
            ):
                self._smoothed_rate_x = 0.0
                self._smoothed_rate_y = 0.0
                self._fractional_x = 0.0
                self._fractional_y = 0.0
                self._control_error_x = error_x
                self._control_error_y = error_y
                self._processed_sample_id = sample_id
                return

            self._control_error_x = error_x
            self._control_error_y = error_y
            self._processed_sample_id = sample_id

            # Predict ahead by control lead plus bounded sample age so the
            # control point lands where the moving head is now, not where the
            # detector last saw it.
            sample_age_s = max((current_ns - measurement_ns) / 1_000_000_000, 0.0)
            lead_s = self.config.prediction_lead_seconds + min(
                sample_age_s,
                self.config.max_sample_age_lead_seconds,
            )
            predicted_x = error_x + velocity_x * lead_s
            predicted_y = error_y + velocity_y * lead_s

            confidence = min(max(float(getattr(target, "confidence", 1.0)), 0.0), 1.0)
            gain_scale = LOW_CONFIDENCE_GAIN_FLOOR + (1.0 - LOW_CONFIDENCE_GAIN_FLOOR) * confidence

            correction_x = (
                predicted_x * self.config.strength * gain_scale
                + velocity_x * self.config.derivative_damping_seconds
            )
            correction_y = (
                predicted_y * self.config.strength * gain_scale
                + velocity_y * self.config.derivative_damping_seconds
            )
            limit = float(self.config.max_step)
            rate_x = min(max(correction_x, -limit), limit) * REFERENCE_CONTROL_HZ
            rate_y = min(max(correction_y, -limit), limit) * REFERENCE_CONTROL_HZ
            vertical_limit = limit * REFERENCE_CONTROL_HZ * MAX_VERTICAL_RATE_RATIO
            rate_y = min(max(rate_y, -vertical_limit), vertical_limit)

            # If error crosses center, clear accumulated momentum immediately.
            if error_x and previous_error_x and (error_x > 0) != (previous_error_x > 0):
                self._smoothed_rate_x = 0.0
                self._fractional_x = 0.0
            if error_y and previous_error_y and (error_y > 0) != (previous_error_y > 0):
                self._smoothed_rate_y = 0.0
                self._fractional_y = 0.0

            distance = math.hypot(error_x, error_y)
            if (
                distance < CLOSE_RANGE_SLOWDOWN_PIXELS
                and velocity_mag < MOTION_SLOWDOWN_BYPASS_PX_PER_SEC
            ):
                scale = max(distance / CLOSE_RANGE_SLOWDOWN_PIXELS, MIN_CLOSE_RANGE_SCALE)
                rate_x *= scale
                rate_y *= scale
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
                    * MAX_VERTICAL_RATE_RATIO
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
            self._command(f"km.move({delta_x},{delta_y})")

    def _disarm_for_shutdown(self) -> None:
        """Remove every software authorization before touching blocking I/O."""

        with self._state_lock:
            self._button_mask = 0
            self._button_state_known = False
            self._activation_started_ns = 0
            self._activation_requires_release = True
            self._latest_target = None
            self._latest_active = False
            self._latest_update_ns = 0

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
        self._latest_velocity_x = 0.0
        self._latest_velocity_y = 0.0
        self._latest_sample_id = 0
        self._processed_sample_id = 0
        self._control_error_x = 0.0
        self._control_error_y = 0.0
        self._measurement_error_x = 0.0
        self._measurement_error_y = 0.0
        self._button_parser.reset()
