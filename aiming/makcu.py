"""Detection-driven relative mouse output for a MAKCU passthrough board."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import errno
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
TARGET_STALE_SECONDS = 0.15
BUTTON_RELEASE_GRACE_SECONDS = 0.30
CLOSE_RANGE_SLOWDOWN_PIXELS = 130.0
MIN_CLOSE_RANGE_SCALE = 0.72
THREADED_RATE_SMOOTHING_ALPHA = 0.82
THREADED_TICK_STEP_FRACTION = 0.78
PREDICTION_LEAD_SECONDS = 0.028
DERIVATIVE_DAMPING_SECONDS = 0.016
MAX_SAMPLE_AGE_LEAD_SECONDS = 0.055
MOTION_SLOWDOWN_BYPASS_PX_PER_SEC = 95.0
MAX_TRACKED_VELOCITY_PX_PER_SEC = 2600.0
ERROR_JUMP_RESET_PIXELS = 240.0
LOW_CONFIDENCE_GAIN_FLOOR = 0.45
MAX_VERTICAL_RATE_RATIO = 0.48
MAX_VERTICAL_TICK_STEP = 6


class MakcuError(RuntimeError):
    """User-facing MAKCU discovery, connection, or command failure."""


def detect_makcu_port(
    *,
    requested: str = "",
    ports_provider: Callable[[], Iterable[Any]] | None = None,
) -> str:
    """Return one MAKCU serial path or raise a user-facing discovery error."""

    selected = requested.strip()
    if selected:
        if selected.startswith("/dev/") and not Path(selected).exists():
            raise MakcuError(f"MAKCU serial device not found: {selected}")
        return selected

    provider = ports_provider
    if provider is None:
        if _list_ports is None:
            detail = f": {SERIAL_IMPORT_ERROR}" if SERIAL_IMPORT_ERROR else ""
            raise MakcuError("MAKCU discovery requires the 'pyserial' package" + detail)
        provider = _list_ports.comports

    matches = [
        str(port.device)
        for port in provider()
        if getattr(port, "vid", None) == MAKCU_VENDOR_ID
        and getattr(port, "pid", None) == MAKCU_PRODUCT_ID
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise MakcuError("MAKCU 1a86:55d3 serial device was not found")
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

    if target is None:
        return 0, 0
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return 0, 0
    target_x, target_y = head_target_point(target, config.head_ratio)
    error_x = target_x - width / 2.0
    error_y = target_y - height / 2.0
    if config.invert_x:
        error_x = -error_x
    if config.invert_y:
        error_y = -error_y
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

    if target is None:
        return 0.0, 0.0
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    target_x, target_y = head_target_point(target, config.head_ratio)
    error_x = target_x - width / 2.0
    error_y = target_y - height / 2.0
    if config.invert_x:
        error_x = -error_x
    if config.invert_y:
        error_y = -error_y
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
    if target is None:
        return 0.0, 0.0
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    target_x, target_y = head_target_point(target, config.head_ratio)
    error_x = target_x - width / 2.0
    error_y = target_y - height / 2.0
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
    ) -> None:
        self.config = config or MakcuAimConfig()
        self._serial_factory = serial_factory
        self._ports_provider = ports_provider
        self._sleep = sleep
        self._threaded_output = threaded_output
        self._serial: Any | None = None
        self._button_mask = 0
        self._activation_latched = False
        self._release_started_ns = 0
        self._serial_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._output_thread: threading.Thread | None = None
        self._worker_error: MakcuError | None = None
        self._latest_target: Detection | None = None
        self._latest_frame_shape: tuple[int, int, int] = (0, 0, 0)
        self._latest_active = False
        self._latest_update_ns = 0
        self._fractional_x = 0.0
        self._fractional_y = 0.0
        self._smoothed_rate_x = 0.0
        self._smoothed_rate_y = 0.0
        self._previous_error_x = 0.0
        self._previous_error_y = 0.0
        self._previous_error_ns = 0
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
                f"Permission denied opening {port}. Install the Game Detector MAKCU "
                "udev rule, then reconnect the board."
            ) from permission_error
        raise MakcuError(f"No MAKCU firmware response was received from {port}")

    def start(self, *, output_loop: bool | None = None) -> None:
        if self._serial is not None:
            return
        self._require_serial()
        port = self._find_port()
        self._serial = self._connect(port)
        self.connected_port = port
        self._button_mask = 0
        self._activation_latched = False
        self._release_started_ns = 0
        self._worker_error = None
        self._latest_target = None
        self._latest_active = False
        self._latest_update_ns = 0
        self._fractional_x = 0.0
        self._fractional_y = 0.0
        self._smoothed_rate_x = 0.0
        self._smoothed_rate_y = 0.0
        self._previous_error_x = 0.0
        self._previous_error_y = 0.0
        self._previous_error_ns = 0
        self._command("km.buttons(1)")
        should_start_output = self._threaded_output if output_loop is None else output_loop
        if should_start_output:
            self._stop_event.clear()
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
        for value in data:
            if value <= 0x1F and value not in (0x0A, 0x0D):
                with self._state_lock:
                    self._button_mask = value
                    if value & (1 << self.config.activation_button):
                        self._activation_latched = True
                        self._release_started_ns = 0
                    elif self._activation_latched and self._release_started_ns == 0:
                        self._release_started_ns = current_ns

    def _activation_pressed_at(self, now_ns: int) -> bool:
        with self._state_lock:
            if (
                self._activation_latched
                and self._release_started_ns
                and now_ns - self._release_started_ns
                >= int(BUTTON_RELEASE_GRACE_SECONDS * 1_000_000_000)
            ):
                self._activation_latched = False
                self._release_started_ns = 0
            return self._activation_latched

    @property
    def activation_pressed(self) -> bool:
        return self._activation_pressed_at(time.perf_counter_ns())

    def poll_button_mask(self, *, now_ns: int | None = None) -> int:
        """Read pending physical button reports and return the latest 5-bit mask."""

        current_ns = time.perf_counter_ns() if now_ns is None else now_ns
        self._read_buttons(now_ns=current_ns)
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
    ) -> None:
        if self._serial is None:
            raise MakcuError("MAKCU serial connection is not open")
        if self._worker_error is not None:
            raise self._worker_error
        with self._state_lock:
            self._latest_target = target
            self._latest_frame_shape = frame_shape
            self._latest_active = bool(active)
            self._latest_update_ns = time.perf_counter_ns()
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
        stale = updated_ns == 0 or (current_ns - updated_ns) / 1_000_000_000 > TARGET_STALE_SECONDS
        if not active or not button_pressed or target is None or stale:
            self._fractional_x = 0.0
            self._fractional_y = 0.0
            self._smoothed_rate_x = 0.0
            self._smoothed_rate_y = 0.0
            self._previous_error_x = 0.0
            self._previous_error_y = 0.0
            self._previous_error_ns = 0
            return
        rate_x, rate_y = _makcu_target_rate(target, frame_shape, self.config)
        if self._output_thread is not None:
            error_x, error_y = _target_error_pixels(target, frame_shape, self.config)
            previous_error_x = self._previous_error_x
            previous_error_y = self._previous_error_y

            velocity_x = 0.0
            velocity_y = 0.0
            if self._previous_error_ns:
                delta_t = (current_ns - self._previous_error_ns) / 1_000_000_000
                if delta_t > 0.0:
                    velocity_x = (error_x - self._previous_error_x) / delta_t
                    velocity_y = (error_y - self._previous_error_y) / delta_t

            # Sudden detector jumps can otherwise convert into one-frame flicks.
            if (
                abs(error_x - self._previous_error_x) > ERROR_JUMP_RESET_PIXELS
                or abs(error_y - self._previous_error_y) > ERROR_JUMP_RESET_PIXELS
            ):
                self._smoothed_rate_x = 0.0
                self._smoothed_rate_y = 0.0
                self._fractional_x = 0.0
                self._fractional_y = 0.0

            velocity_x = min(max(velocity_x, -MAX_TRACKED_VELOCITY_PX_PER_SEC), MAX_TRACKED_VELOCITY_PX_PER_SEC)
            velocity_y = min(max(velocity_y, -MAX_TRACKED_VELOCITY_PX_PER_SEC), MAX_TRACKED_VELOCITY_PX_PER_SEC)

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
                self._previous_error_x = error_x
                self._previous_error_y = error_y
                self._previous_error_ns = current_ns
                return

            self._previous_error_x = error_x
            self._previous_error_y = error_y
            self._previous_error_ns = current_ns

            # Predict ahead by control lead plus bounded sample age so the
            # control point lands where the moving head is now, not where the
            # detector last saw it.
            sample_age_s = max((current_ns - updated_ns) / 1_000_000_000, 0.0)
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
                - velocity_x * self.config.derivative_damping_seconds
            )
            correction_y = (
                predicted_y * self.config.strength * gain_scale
                - velocity_y * self.config.derivative_damping_seconds
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
            alpha = self.config.smoothing_alpha
            self._smoothed_rate_x += (rate_x - self._smoothed_rate_x) * alpha
            self._smoothed_rate_y += (rate_y - self._smoothed_rate_y) * alpha
            rate_x = self._smoothed_rate_x
            rate_y = self._smoothed_rate_y
        self._fractional_x += rate_x * elapsed
        self._fractional_y += rate_y * elapsed
        delta_x = math.trunc(self._fractional_x)
        delta_y = math.trunc(self._fractional_y)
        if self._output_thread is not None:
            tick_limit = max(
                1,
                int(
                    round(
                        self.config.max_step
                        * elapsed
                        * REFERENCE_CONTROL_HZ
                        * THREADED_TICK_STEP_FRACTION
                    )
                ),
            )
            vertical_tick_limit = min(tick_limit, MAX_VERTICAL_TICK_STEP)
            delta_x = min(max(delta_x, -tick_limit), tick_limit)
            delta_y = min(max(delta_y, -vertical_tick_limit), vertical_tick_limit)
        self._fractional_x -= delta_x
        self._fractional_y -= delta_y
        if delta_x or delta_y:
            self._command(f"km.move({delta_x},{delta_y})")

    def stop(self) -> None:
        connection = self._serial
        if connection is None:
            return
        self._stop_event.set()
        worker = self._output_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=0.25)
        self._output_thread = None
        try:
            self._command("km.buttons(0)")
        except MakcuError:
            pass
        try:
            connection.close()
        except OSError:
            pass
        self._serial = None
        self.connected_port = None
        with self._state_lock:
            self._button_mask = 0
            self._activation_latched = False
            self._release_started_ns = 0
            self._latest_target = None
            self._latest_active = False
            self._latest_update_ns = 0
        self._fractional_x = 0.0
        self._fractional_y = 0.0
        self._smoothed_rate_x = 0.0
        self._smoothed_rate_y = 0.0
        self._previous_error_x = 0.0
        self._previous_error_y = 0.0
        self._previous_error_ns = 0