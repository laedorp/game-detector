"""Detection-driven virtual right-stick aiming through Linux uinput."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import socket
import threading
import time
from typing import Any

from detection.types import Detection
from controller_precision.codes import ABS_BRAKE, ABS_RZ, ABS_Z, EV_ABS, EV_SYN
from .protocol import AimCommand, encode_aim_command, validate_pairing_key


DEFAULT_HEAD_RATIO = 0.12
LOCAL_TARGET_STALE_SECONDS = 0.15
LOCAL_WATCHDOG_INTERVAL_SECONDS = 0.025
TARGET_TRACK_REFERENCE_HZ = 60.0
TARGET_TRACK_MEMORY_SECONDS = 0.20
TARGET_CONTINUATION_STRONG_WINDOW_SECONDS = 0.10
TARGET_REACQUIRE_CONFIRMATIONS = 2
TARGET_POSITION_TIME_CONSTANT_SECONDS = 0.020
TARGET_VELOCITY_TIME_CONSTANT_SECONDS = 0.060
TARGET_MAX_SPEED_DIAGONALS_PER_SECOND = 2.0
TARGET_MAX_ACCELERATION_DIAGONALS_PER_SECOND_SQUARED = 6.0
TARGET_TELEMETRY_REFERENCE_WIDTH = 1920.0
TARGET_TELEMETRY_REFERENCE_HEIGHT = 1080.0
TARGET_ASSOCIATION_IDENTITY_TOLERANCE = 0.03
TARGET_ASSOCIATION_SHAPE_TOLERANCE = 0.06

try:
    import evdev as _evdev
except ImportError as exc:  # pragma: no cover
    _evdev = None  # type: ignore[assignment]
    EVDEV_IMPORT_ERROR: BaseException | None = exc
else:
    EVDEV_IMPORT_ERROR = None


class AimingControllerError(RuntimeError):
    """User-facing failure to start or run the detection aiming output."""


class AimActivationError(RuntimeError):
    """Failure while reading a physical controller axis for aim activation."""


class AimActivationSensor:
    """Read an analog controller axis and expose whether aim should be active."""

    def __init__(
        self,
        device_path: str,
        axis: int = ABS_BRAKE,
        threshold: float = 0.35,
    ) -> None:
        self.device_path = device_path
        self.axis = axis
        if isinstance(threshold, bool):
            raise ValueError(
                "activation threshold must be finite, greater than 0, and at most 1"
            )
        parsed_threshold = float(threshold)
        if not math.isfinite(parsed_threshold) or not 0.0 < parsed_threshold <= 1.0:
            raise ValueError(
                "activation threshold must be finite, greater than 0, and at most 1"
            )
        self.threshold = parsed_threshold
        self._device: Any | None = None
        self._minimum = 0
        self._maximum = 255
        self._active = False

    def _require_evdev(self) -> Any:
        if _evdev is None:
            detail = f": {EVDEV_IMPORT_ERROR}" if EVDEV_IMPORT_ERROR else ""
            raise AimActivationError(
                "Detection-driven aiming requires the optional 'evdev' package" + detail
            )
        return _evdev

    def _pressure(self, raw_value: int) -> float:
        if self._maximum <= self._minimum:
            return 0.0
        return min(max((raw_value - self._minimum) / (self._maximum - self._minimum), 0.0), 1.0)

    def start(self) -> None:
        if self._device is not None:
            return
        if not Path(self.device_path).exists():
            raise AimActivationError(
                f"Activation device not found: {self.device_path}"
            )
        evdev = self._require_evdev()
        try:
            device = evdev.InputDevice(self.device_path)
        except OSError as exc:
            raise AimActivationError(
                f"Cannot open activation device {self.device_path}: {exc}"
            ) from exc
        info = device.absinfo(self.axis)
        if info is None:
            raise AimActivationError(
                f"Activation axis {self.axis} not found on {self.device_path}."
            )
        self._device = device
        self._minimum = info.min
        self._maximum = info.max
        self._active = self._pressure(info.value) >= self.threshold
        try:
            device.set_nonblocking(True)
        except OSError:
            pass

    def stop(self) -> None:
        if self._device is None:
            return
        try:
            self._device.close()
        except OSError:
            pass
        self._device = None

    def read(self) -> bool:
        if self._device is None:
            raise AimActivationError("Activation sensor is not started")
        while True:
            try:
                event = self._device.read_one()
            except OSError as exc:
                raise AimActivationError(f"Activation device read failed: {exc}") from exc
            if event is None:
                break
            if event.type == EV_ABS and event.code == self.axis:
                self._active = self._pressure(event.value) >= self.threshold
        # Do not rely solely on edge events. A dropped release event would
        # otherwise leave the cached state active indefinitely. EVIOCGABS
        # returns the kernel's current axis value, so resample it every frame
        # and fail closed if the device can no longer be queried.
        try:
            info = self._device.absinfo(self.axis)
        except OSError as exc:
            self._active = False
            raise AimActivationError(
                f"Activation device state query failed: {exc}"
            ) from exc
        if info is None:
            self._active = False
            raise AimActivationError(
                f"Activation axis {self.axis} disappeared from {self.device_path}."
            )
        self._active = self._pressure(info.value) >= self.threshold
        return self._active


@dataclass(frozen=True, slots=True)
class AimConfig:
    """Parameters controlling the aiming output sent to a virtual controller."""

    right_x_code: int = ABS_Z
    right_y_code: int = ABS_RZ
    trigger_code: int = ABS_BRAKE
    invert_x: bool = False
    invert_y: bool = False
    head_ratio: float = DEFAULT_HEAD_RATIO
    neutral_value: int = 128
    minimum_value: int = 0
    maximum_value: int = 255
    trigger_released_value: int = 0
    trigger_pressed_value: int = 255
    vendor_id: int = 0x36E6
    product_id: int = 0x3016
    version: int = 0x0111
    bustype: int = 0x0003

    def __post_init__(self) -> None:
        if self.right_x_code == self.right_y_code:
            raise ValueError("right-stick axes must use different event codes")
        if self.trigger_code in (self.right_x_code, self.right_y_code):
            raise ValueError("trigger axis must be different from both right-stick axes")
        if not 0.0 <= self.head_ratio <= 0.5:
            raise ValueError("head_ratio must be between 0 and 0.5")
        if not self.minimum_value < self.neutral_value < self.maximum_value:
            raise ValueError("neutral_value must lie strictly between min and max")
        if not self.minimum_value <= self.trigger_released_value <= self.maximum_value:
            raise ValueError("trigger_released_value must be within min..max")
        if not self.minimum_value <= self.trigger_pressed_value <= self.maximum_value:
            raise ValueError("trigger_pressed_value must be within min..max")


def head_target_point(
    target: Detection,
    head_ratio: float = DEFAULT_HEAD_RATIO,
) -> tuple[float, float]:
    """Return the horizontal center and configured head height of a box."""

    return (
        (target.x1 + target.x2) / 2.0,
        target.y1 + (target.y2 - target.y1) * head_ratio,
    )


def target_aim_vector(
    target: Detection | None,
    frame_shape: tuple[int, ...],
    config: AimConfig,
    *,
    active: bool = True,
) -> tuple[bool, float, float]:
    """Return active and normalized right/down target direction."""

    if not active or target is None:
        return False, 0.0, 0.0
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return False, 0.0, 0.0
    center_x, head_y = head_target_point(target, config.head_ratio)
    normalized_x = min(max((center_x - width / 2.0) / (width / 2.0), -1.0), 1.0)
    normalized_y = min(max((head_y - height / 2.0) / (height / 2.0), -1.0), 1.0)
    if config.invert_x:
        normalized_x = -normalized_x
    if config.invert_y:
        normalized_y = -normalized_y
    return True, normalized_x, normalized_y


class AimingController:
    """Send right-stick coordinates to a Linux virtual controller."""

    def __init__(
        self,
        config: AimConfig | None = None,
        *,
        device_name: str = "PXN P5 8K",
        uinput_factory: Any | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        watchdog_interval: float = LOCAL_WATCHDOG_INTERVAL_SECONDS,
    ) -> None:
        if not math.isfinite(watchdog_interval) or watchdog_interval <= 0.0:
            raise ValueError("aim watchdog interval must be finite and greater than zero")
        self.config = config or AimConfig()
        self._device_name = device_name
        self._uinput_factory = uinput_factory
        self._clock_ns = clock_ns
        self._watchdog_interval = float(watchdog_interval)
        self._uinput: Any | None = None
        self._last_x = self.config.neutral_value
        self._last_y = self.config.neutral_value
        self._last_trigger = self.config.trigger_released_value
        self._latest_active_update_ns = 0
        self._output_active = False
        self._device_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._worker_error: AimingControllerError | None = None

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return min(max(value, minimum), maximum)

    @staticmethod
    def _normalize_coordinate(value: float, length: int) -> float:
        if length <= 0:
            return 0.0
        center = length / 2.0
        return (value - center) / center

    @staticmethod
    def _normalized_to_raw(normalized: float, minimum: int, maximum: int) -> int:
        clamped = min(max(normalized, -1.0), 1.0)
        span = maximum - minimum
        return int(round((clamped + 1.0) / 2.0 * span + minimum))

    def _require_evdev(self) -> Any:
        if self._uinput_factory is not None:
            return _evdev
        if _evdev is None:
            detail = f": {EVDEV_IMPORT_ERROR}" if EVDEV_IMPORT_ERROR else ""
            raise AimingControllerError(
                "Detection-driven aiming requires the optional 'evdev' package" + detail
            )
        return _evdev

    def _make_uinput(self, evdev: Any) -> Any:
        if self._uinput_factory is not None:
            return self._uinput_factory(
                {
                    EV_ABS: [
                        (
                            self.config.right_x_code,
                            evdev.AbsInfo(
                                self.config.neutral_value,
                                self.config.minimum_value,
                                self.config.maximum_value,
                                0,
                                15,
                                0,
                            ),
                        ),
                        (
                            self.config.right_y_code,
                            evdev.AbsInfo(
                                self.config.neutral_value,
                                self.config.minimum_value,
                                self.config.maximum_value,
                                0,
                                15,
                                0,
                            ),
                        ),
                        (
                            self.config.trigger_code,
                            evdev.AbsInfo(
                                self.config.trigger_released_value,
                                self.config.minimum_value,
                                self.config.maximum_value,
                                0,
                                15,
                                0,
                            ),
                        ),
                    ]
                }
            )

        return evdev.UInput(
            {
                EV_ABS: [
                    (
                        self.config.right_x_code,
                        evdev.AbsInfo(
                            self.config.neutral_value,
                            self.config.minimum_value,
                            self.config.maximum_value,
                            0,
                            15,
                            0,
                        ),
                    ),
                    (
                        self.config.right_y_code,
                        evdev.AbsInfo(
                            self.config.neutral_value,
                            self.config.minimum_value,
                            self.config.maximum_value,
                            0,
                            15,
                            0,
                        ),
                    ),
                    (
                        self.config.trigger_code,
                        evdev.AbsInfo(
                            self.config.trigger_released_value,
                            self.config.minimum_value,
                            self.config.maximum_value,
                            0,
                            15,
                            0,
                        ),
                    ),
                ]
            },
            name=self._device_name,
            vendor=self.config.vendor_id,
            product=self.config.product_id,
            version=self.config.version,
            bustype=self.config.bustype,
            phys="game-detector-aim/uinput",
        )

    def start(self) -> None:
        if self._uinput is not None:
            return
        if self._uinput_factory is None and not Path("/dev/uinput").exists():
            raise AimingControllerError(
                "/dev/uinput is missing. Load the Linux uinput module before using --aim."
            )
        self._uinput = self._make_uinput(self._require_evdev())
        self._worker_error = None
        self._latest_active_update_ns = 0
        self._output_active = False
        self._stop_event.clear()
        self._write_axes(
            self.config.neutral_value,
            self.config.neutral_value,
            self.config.trigger_released_value,
        )
        self._watchdog_thread = threading.Thread(
            target=self._run_watchdog,
            name="aim-uinput-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        worker = self._watchdog_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(0.25, self._watchdog_interval * 2.0))
        self._watchdog_thread = None
        with self._device_lock:
            device = self._uinput
            if device is None:
                return
            try:
                self._write_axes_unlocked(
                    self.config.neutral_value,
                    self.config.neutral_value,
                    self.config.trigger_released_value,
                )
            except (OSError, ValueError):
                pass
            try:
                device.close()
            except OSError:
                pass
            self._uinput = None
            self._record_neutral_state()

    def _write_axes(self, x: int, y: int, trigger: int) -> None:
        with self._device_lock:
            self._write_axes_unlocked(x, y, trigger)

    def _write_axes_unlocked(self, x: int, y: int, trigger: int) -> None:
        if self._uinput is None:
            raise AimingControllerError("Virtual controller is not open")
        self._uinput.write(EV_ABS, self.config.right_x_code, x)
        self._uinput.write(EV_ABS, self.config.right_y_code, y)
        self._uinput.write(EV_ABS, self.config.trigger_code, trigger)
        self._uinput.write(EV_SYN, 0, 0)

    def _record_neutral_state(self) -> None:
        self._last_x = self.config.neutral_value
        self._last_y = self.config.neutral_value
        self._last_trigger = self.config.trigger_released_value
        self._latest_active_update_ns = 0
        self._output_active = False

    def _neutralize_unlocked(self) -> None:
        if self._uinput is None:
            return
        if (
            self._last_x != self.config.neutral_value
            or self._last_y != self.config.neutral_value
            or self._last_trigger != self.config.trigger_released_value
        ):
            self._write_axes_unlocked(
                self.config.neutral_value,
                self.config.neutral_value,
                self.config.trigger_released_value,
            )
        self._record_neutral_state()

    def _watchdog_tick(self, now_ns: int | None = None) -> None:
        """Neutralize an active local output whose detector state stopped refreshing."""

        current_ns = self._clock_ns() if now_ns is None else int(now_ns)
        with self._device_lock:
            if not self._output_active or self._latest_active_update_ns == 0:
                return
            age = (current_ns - self._latest_active_update_ns) / 1_000_000_000
            if age <= LOCAL_TARGET_STALE_SECONDS:
                return
            try:
                self._neutralize_unlocked()
            except (OSError, ValueError) as exc:
                self._worker_error = AimingControllerError(
                    f"Local aim watchdog could not neutralize output: {exc}"
                )

    def _run_watchdog(self) -> None:
        while not self._stop_event.wait(self._watchdog_interval):
            self._watchdog_tick()

    def update(
        self,
        target: Detection | None,
        frame_shape: tuple[int, int, int],
        active: bool = True,
    ) -> None:
        has_target, normalized_x, normalized_y = target_aim_vector(
            target,
            frame_shape,
            self.config,
            active=active,
        )
        if not has_target:
            x = self.config.neutral_value
            y = self.config.neutral_value
            trigger = self.config.trigger_released_value
        else:
            x = self._normalized_to_raw(
                normalized_x,
                self.config.minimum_value,
                self.config.maximum_value,
            )
            y = self._normalized_to_raw(
                normalized_y,
                self.config.minimum_value,
                self.config.maximum_value,
            )
            # Detection output may move only the two stick axes.  It never
            # synthesizes a trigger press from target presence.
            trigger = self.config.trigger_released_value

        with self._device_lock:
            if self._uinput is None:
                raise AimingControllerError("Aiming controller has not been started")
            if self._worker_error is not None:
                raise self._worker_error
            if x != self._last_x or y != self._last_y or trigger != self._last_trigger:
                self._write_axes_unlocked(x, y, trigger)
                self._last_x = x
                self._last_y = y
                self._last_trigger = trigger
            if has_target:
                self._latest_active_update_ns = self._clock_ns()
                self._output_active = True
            else:
                self._latest_active_update_ns = 0
                self._output_active = False


class UdpAimingController:
    """Send authenticated latest-frame aim vectors to a gaming-PC receiver."""

    def __init__(
        self,
        host: str,
        port: int,
        pairing_key: str,
        config: AimConfig | None = None,
        *,
        socket_factory: Any | None = None,
    ) -> None:
        self.host = host.strip()
        if not self.host:
            raise ValueError("gaming PC host cannot be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("gaming PC UDP port must be between 1 and 65535")
        self.port = int(port)
        self.pairing_key = validate_pairing_key(pairing_key)
        self.config = config or AimConfig()
        self._socket_factory = socket_factory
        self._socket: Any | None = None
        self._sequence = 0

    def start(self) -> None:
        if self._socket is not None:
            return
        try:
            sender = (
                self._socket_factory()
                if self._socket_factory is not None
                else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            )
            sender.connect((self.host, self.port))
        except OSError as exc:
            raise AimingControllerError(
                f"Could not open remote aim output to {self.host}:{self.port}: {exc}"
            ) from exc
        self._socket = sender

    def _send(self, active: bool, x: float, y: float) -> None:
        if self._socket is None:
            raise AimingControllerError("Remote aiming output has not been started")
        command = AimCommand(self._sequence, active, x, y)
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        try:
            self._socket.send(encode_aim_command(command, self.pairing_key))
        except OSError as exc:
            raise AimingControllerError(f"Remote aim output failed: {exc}") from exc

    def update(
        self,
        target: Detection | None,
        frame_shape: tuple[int, int, int],
        active: bool = True,
    ) -> None:
        has_target, x, y = target_aim_vector(
            target,
            frame_shape,
            self.config,
            active=active,
        )
        self._send(has_target, x, y)

    def stop(self) -> None:
        sender = self._socket
        if sender is None:
            return
        try:
            self._send(False, 0.0, 0.0)
        except (AimingControllerError, OSError):
            pass
        try:
            sender.close()
        except OSError:
            pass
        self._socket = None


def choose_target(
    detections: list[Detection],
    *,
    label: str | None = None,
    frame_shape: tuple[int, ...] | None = None,
    head_ratio: float = DEFAULT_HEAD_RATIO,
) -> Detection | None:
    candidates = detections
    if label is not None:
        normalized_label = label.strip().lower()
        candidates = [
            detection
            for detection in detections
            if detection.class_name.strip().lower() == normalized_label
        ]
    if not candidates:
        return None
    if frame_shape is None:
        return max(candidates, key=lambda detection: detection.confidence)
    height, width = frame_shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0

    diagonal = math.hypot(width, height)

    def target_key(detection: Detection) -> tuple[float, float, float]:
        target_x, target_y = head_target_point(detection, head_ratio)
        distance = math.hypot(target_x - center_x, target_y - center_y)
        confidence = min(max(float(detection.confidence), 0.0), 1.0)
        # Crosshair distance remains the primary intent signal, but a barely
        # accepted speck must not beat a nearby, high-confidence player merely
        # because its noisy box happens to cover the exact center pixel.
        confidence_penalty = diagonal * 0.15 * (1.0 - confidence)
        return distance + confidence_penalty, distance, -confidence

    return min(candidates, key=target_key)


@dataclass(frozen=True, slots=True)
class TargetTrackerTelemetrySnapshot:
    """Monotonic, aggregate observations of the target tracking path.

    Residual totals compare the exact accepted detector measurement with the
    tracker output from that same sample.  They are expressed in 1920x1080
    reference pixels, and contain no absolute screen coordinates.
    """

    updates: int = 0
    candidate_samples: int = 0
    measurement_samples: int = 0
    continuation_measurement_samples: int = 0
    output_samples: int = 0
    compared_samples: int = 0
    target_loss_transitions: int = 0
    residual_x: float = 0.0
    residual_y: float = 0.0
    residual_abs_x: float = 0.0
    residual_abs_y: float = 0.0


class TargetTracker:
    """Keep one measured target stable without frame-rate-dependent motion state."""

    def __init__(
        self,
        *,
        label: str | None = None,
        head_ratio: float = DEFAULT_HEAD_RATIO,
        lost_grace_frames: int = 3,
        reacquire_confirmations: int = TARGET_REACQUIRE_CONFIRMATIONS,
        track_memory_seconds: float = TARGET_TRACK_MEMORY_SECONDS,
    ) -> None:
        if not 0.0 <= head_ratio <= 0.5:
            raise ValueError("head_ratio must be between 0 and 0.5")
        if lost_grace_frames < 0:
            raise ValueError("lost_grace_frames cannot be negative")
        if reacquire_confirmations <= 0:
            raise ValueError("reacquire_confirmations must be greater than zero")
        if not math.isfinite(track_memory_seconds) or track_memory_seconds < 0.0:
            raise ValueError("track_memory_seconds must be finite and non-negative")
        self.label = label
        self.head_ratio = head_ratio
        self.lost_grace_frames = int(lost_grace_frames)
        self.reacquire_confirmations = int(reacquire_confirmations)
        self.track_memory_seconds = float(track_memory_seconds)
        self._target: Detection | None = None
        self._smoothed_box: tuple[float, float, float, float] | None = None
        self._last_observed_box: tuple[float, float, float, float] | None = None
        self._box_velocity = (0.0, 0.0, 0.0, 0.0)  # pixels per second
        self._last_measurement_ns: int | None = None
        self._last_strong_measurement_ns: int | None = None
        self._last_sample_ns: int | None = None
        self._synthetic_measurement_ns = 0
        self._pending_target: Detection | None = None
        self._pending_hits = 0
        self._ever_tracked = False
        self._misses = 0
        self._frame_dimensions: tuple[int, int] | None = None
        self._telemetry_updates = 0
        self._telemetry_candidate_samples = 0
        self._telemetry_measurement_samples = 0
        self._telemetry_continuation_measurement_samples = 0
        self._telemetry_output_samples = 0
        self._telemetry_compared_samples = 0
        self._telemetry_target_loss_transitions = 0
        self._telemetry_residual_x = 0.0
        self._telemetry_residual_y = 0.0
        self._telemetry_residual_abs_x = 0.0
        self._telemetry_residual_abs_y = 0.0
        self._telemetry_output_present = False
        self._output_is_prediction = False

    def reset(self) -> None:
        self._reset_tracking_state()
        # A safety reset is an observable output loss, but repeated resets
        # while already missing remain one contiguous loss interval.
        self._record_observation(None, None, None, None, count_update=False)

    def _reset_tracking_state(self) -> None:
        self._target = None
        self._smoothed_box = None
        self._last_observed_box = None
        self._box_velocity = (0.0, 0.0, 0.0, 0.0)
        self._last_measurement_ns = None
        self._last_strong_measurement_ns = None
        self._last_sample_ns = None
        self._synthetic_measurement_ns = 0
        self._clear_pending()
        self._ever_tracked = False
        self._misses = 0
        self._frame_dimensions = None
        self._output_is_prediction = False

    @property
    def output_is_prediction(self) -> bool:
        """Whether the most recent output was synthesized through loss grace."""

        return self._output_is_prediction

    def telemetry_snapshot(self) -> TargetTrackerTelemetrySnapshot:
        """Return aggregate tracker diagnostics without changing tracker state."""

        return TargetTrackerTelemetrySnapshot(
            updates=self._telemetry_updates,
            candidate_samples=self._telemetry_candidate_samples,
            measurement_samples=self._telemetry_measurement_samples,
            continuation_measurement_samples=(
                self._telemetry_continuation_measurement_samples
            ),
            output_samples=self._telemetry_output_samples,
            compared_samples=self._telemetry_compared_samples,
            target_loss_transitions=self._telemetry_target_loss_transitions,
            residual_x=self._telemetry_residual_x,
            residual_y=self._telemetry_residual_y,
            residual_abs_x=self._telemetry_residual_abs_x,
            residual_abs_y=self._telemetry_residual_abs_y,
        )

    def _record_observation(
        self,
        candidate: Detection | None,
        measurement: Detection | None,
        output: Detection | None,
        dimensions: tuple[int, int] | None,
        *,
        count_update: bool = True,
        continuation_measurement: bool = False,
    ) -> Detection | None:
        """Record one already-computed result without affecting selection."""

        if count_update:
            self._telemetry_updates += 1
            self._telemetry_candidate_samples += int(candidate is not None)
            self._telemetry_measurement_samples += int(measurement is not None)
            self._telemetry_continuation_measurement_samples += int(
                continuation_measurement and measurement is not None
            )
            self._telemetry_output_samples += int(output is not None)

        output_present = output is not None
        if self._telemetry_output_present and not output_present:
            self._telemetry_target_loss_transitions += 1
        self._telemetry_output_present = output_present

        if count_update and measurement is not None and output is not None:
            assert dimensions is not None
            height, width = dimensions
            if width > 0 and height > 0:
                raw_x, raw_y = head_target_point(measurement, self.head_ratio)
                tracked_x, tracked_y = head_target_point(output, self.head_ratio)
                # Positive means the accepted raw point lies right/below the
                # tracker output. Absolute totals remain useful across turns.
                residual_x = (raw_x - tracked_x) * (
                    TARGET_TELEMETRY_REFERENCE_WIDTH / width
                )
                residual_y = (raw_y - tracked_y) * (
                    TARGET_TELEMETRY_REFERENCE_HEIGHT / height
                )
                self._telemetry_compared_samples += 1
                self._telemetry_residual_x += residual_x
                self._telemetry_residual_y += residual_y
                self._telemetry_residual_abs_x += abs(residual_x)
                self._telemetry_residual_abs_y += abs(residual_y)
        return output

    def update(
        self,
        detections: list[Detection] | tuple[Detection, ...],
        frame_shape: tuple[int, ...],
        *,
        measurement_ns: int | None = None,
        continuation_detections: list[Detection] | tuple[Detection, ...] = (),
        continuation_allowed: bool = True,
    ) -> Detection | None:
        if not isinstance(continuation_allowed, bool):
            raise TypeError("continuation_allowed must be bool")
        # This state describes only the result of the current call. Every path
        # other than the explicit missing-target predictor remains measured or
        # targetless.
        self._output_is_prediction = False
        dimensions = (int(frame_shape[0]), int(frame_shape[1]))
        if self._frame_dimensions is not None and dimensions != self._frame_dimensions:
            # A resolution change resets temporal geometry, but the target
            # selected in this same update is not a missing-output interval.
            self._reset_tracking_state()
        self._frame_dimensions = dimensions
        sample_ns = self._sample_time_ns(measurement_ns)
        candidates = list(detections)
        continuation_candidates = list(continuation_detections)
        if self.label is not None:
            normalized_label = self.label.strip().lower()
            candidates = [
                detection
                for detection in candidates
                if detection.class_name.strip().lower() == normalized_label
            ]
            continuation_candidates = [
                detection
                for detection in continuation_candidates
                if detection.class_name.strip().lower() == normalized_label
            ]

        if self._track_memory_expired(sample_ns):
            self._forget_track()

        # Below-threshold detections are continuation evidence only. They can
        # be geometrically associated while a physical output is still active,
        # but can never start a track, enter pending reacquisition, or revive a
        # target after output has dropped.
        associated_from_continuation = False
        associated = self._associated_candidate(candidates, dimensions, sample_ns)
        if (
            associated is None
            and self._target is not None
            and continuation_allowed
            and self._continuation_is_recent(sample_ns)
            and continuation_candidates
        ):
            associated = self._associated_candidate(
                continuation_candidates,
                dimensions,
                sample_ns,
                strict_continuation=True,
            )
            associated_from_continuation = associated is not None

        if not candidates and associated is None:
            self._clear_pending()
            if continuation_candidates:
                # A weak box was physically present but failed provenance,
                # lease, or strict identity checks. This is not a genuine
                # detector-empty sample and must not borrow prediction grace.
                self._target = None
                self._misses += 1
                return self._record_observation(None, None, None, dimensions)
            if self._target is not None and self._within_prediction_grace(sample_ns):
                self._misses += 1
                self._target = self._predict_missing_target(dimensions, sample_ns)
                self._output_is_prediction = True
                return self._record_observation(
                    None,
                    None,
                    self._target,
                    dimensions,
                )
            # Stop output immediately after grace, but retain the last measured
            # identity briefly. A returning nearby box can resume the same
            # track; a different player must pass reacquisition confirmation.
            self._target = None
            self._misses += 1
            return self._record_observation(None, None, None, dimensions)

        candidate: Detection
        measurement: Detection | None
        if associated is None:
            selected = choose_target(
                candidates,
                frame_shape=frame_shape,
                head_ratio=self.head_ratio,
            )
            assert selected is not None
            candidate = selected
            if self._smoothed_box is None and not self._ever_tracked:
                measurement = selected
                self._target = self._start_track(selected, sample_ns)
            elif self._confirm_reacquisition(selected, dimensions):
                measurement = selected
                self._target = self._start_track(selected, sample_ns)
            else:
                # Never snap a physical output to an incompatible detection in
                # one frame. Keeping the old box internally is only identity
                # memory; returning None makes this interval fail closed.
                self._target = None
                self._misses += 1
                return self._record_observation(
                    candidate,
                    None,
                    None,
                    dimensions,
                )
        else:
            candidate = associated
            measurement = associated
            self._clear_pending()
            self._target = self._smooth_measurement(
                associated,
                dimensions,
                sample_ns,
            )
        self._misses = 0
        if not associated_from_continuation:
            self._last_strong_measurement_ns = sample_ns
        return self._record_observation(
            candidate,
            measurement,
            self._target,
            dimensions,
            continuation_measurement=associated_from_continuation,
        )

    def _start_track(self, detection: Detection, measurement_ns: int) -> Detection:
        self._smoothed_box = detection.xyxy
        self._last_observed_box = detection.xyxy
        self._box_velocity = (0.0, 0.0, 0.0, 0.0)
        self._last_measurement_ns = measurement_ns
        self._ever_tracked = True
        self._clear_pending()
        return detection

    def _smooth_measurement(
        self,
        detection: Detection,
        dimensions: tuple[int, int],
        measurement_ns: int,
    ) -> Detection:
        if self._smoothed_box is None:
            return self._start_track(detection, measurement_ns)
        elapsed = self._measurement_elapsed_seconds(measurement_ns)
        predicted = tuple(
            value + velocity * elapsed
            for value, velocity in zip(self._smoothed_box, self._box_velocity)
        )
        residual = tuple(
            measured - estimate
            for measured, estimate in zip(detection.xyxy, predicted)
        )
        position_alpha = (
            1.0 - math.exp(-elapsed / TARGET_POSITION_TIME_CONSTANT_SECONDS)
            if elapsed > 0.0
            else 1.0
        )
        smoothed = tuple(
            estimate + position_alpha * difference
            for estimate, difference in zip(predicted, residual)
        )
        if elapsed > 0.0:
            previous_observation = self._last_observed_box or self._smoothed_box
            raw_velocity = tuple(
                (measured - previous) / elapsed
                for measured, previous in zip(detection.xyxy, previous_observation)
            )
            maximum_speed = (
                math.hypot(dimensions[1], dimensions[0])
                * TARGET_MAX_SPEED_DIAGONALS_PER_SECOND
            )
            raw_velocity = tuple(
                min(max(value, -maximum_speed), maximum_speed)
                for value in raw_velocity
            )
            velocity_alpha = 1.0 - math.exp(
                -elapsed / TARGET_VELOCITY_TIME_CONSTANT_SECONDS
            )
            blended_velocity = tuple(
                velocity + (measured - velocity) * velocity_alpha
                for velocity, measured in zip(self._box_velocity, raw_velocity)
            )
            maximum_velocity_change = (
                math.hypot(dimensions[1], dimensions[0])
                * TARGET_MAX_ACCELERATION_DIAGONALS_PER_SECOND_SQUARED
                * elapsed
            )
            self._box_velocity = tuple(
                velocity
                + min(
                    max(blended - velocity, -maximum_velocity_change),
                    maximum_velocity_change,
                )
                for velocity, blended in zip(
                    self._box_velocity,
                    blended_velocity,
                )
            )
        self._smoothed_box = _clamp_box(smoothed, dimensions)
        self._last_observed_box = detection.xyxy
        self._last_measurement_ns = measurement_ns
        return Detection(
            detection.class_id,
            detection.class_name,
            detection.confidence,
            self._smoothed_box,
        )

    def _predict_missing_target(
        self,
        dimensions: tuple[int, int],
        measurement_ns: int,
    ) -> Detection:
        assert self._target is not None
        if self._smoothed_box is None:
            self._smoothed_box = self._target.xyxy
        elapsed = self._measurement_elapsed_seconds(measurement_ns)
        predicted = tuple(
            value + velocity * elapsed
            for value, velocity in zip(self._smoothed_box, self._box_velocity)
        )
        return Detection(
            self._target.class_id,
            self._target.class_name,
            self._target.confidence,
            _clamp_box(predicted, dimensions),
        )

    def _associated_candidate(
        self,
        candidates: list[Detection],
        dimensions: tuple[int, int],
        measurement_ns: int,
        *,
        strict_continuation: bool = False,
    ) -> Detection | None:
        if self._smoothed_box is None:
            return None
        frame_height, frame_width = dimensions
        diagonal = math.hypot(frame_width, frame_height)
        elapsed = self._measurement_elapsed_seconds(measurement_ns)
        predicted_box = _clamp_box(
            tuple(
                value + velocity * elapsed
                for value, velocity in zip(self._smoothed_box, self._box_velocity)
            ),
            dimensions,
        )
        predicted_point = _box_target_point(predicted_box, self.head_ratio)
        predicted_area = _box_area(predicted_box)
        distance_gate = min(max(0.035 + elapsed * 1.5, 0.045), 0.10)
        # The last raw measurement is a better mode reference than the
        # smoothed/predicted box. Some detectors emit two heavily-overlapping
        # boxes for the same player (for example, a narrow torso box and a
        # slightly wider full-player box). Mixing those shapes into the filter
        # lets confidence noise alternate the selected measurement and moves
        # the derived head point even when the physical player is steady.
        mode_reference = self._last_observed_box or predicted_box
        scored: list[tuple[float, float, float, float, Detection]] = []
        for candidate in candidates:
            iou = _box_iou_tuple(predicted_box, candidate.xyxy)
            if strict_continuation:
                candidate_area = _box_area(candidate.xyxy)
                area_ratio = (
                    candidate_area / predicted_area
                    if predicted_area > 0.0
                    else math.inf
                )
                if iou < 0.15 or not 0.5 <= area_ratio <= 2.0:
                    continue
            point = head_target_point(candidate, self.head_ratio)
            distance = math.hypot(
                point[0] - predicted_point[0],
                point[1] - predicted_point[1],
            )
            normalized_distance = distance / diagonal if diagonal > 0.0 else 1.0
            if iou >= 0.10 or normalized_distance <= distance_gate:
                proximity = max(0.0, 1.0 - normalized_distance / distance_gate)
                confidence = min(max(float(candidate.confidence), 0.0), 1.0)
                # Identity continuity dominates detector confidence. Confidence
                # breaks close association ties, but cannot make a nearby rival
                # steal a well-overlapping established track in one frame.
                identity_score = iou * 0.70 + proximity * 0.30
                shape_continuity = _box_shape_similarity(
                    mode_reference,
                    candidate.xyxy,
                )
                scored.append(
                    (
                        identity_score,
                        shape_continuity,
                        confidence,
                        -normalized_distance,
                        candidate,
                    )
                )
        if not scored:
            return None
        best_identity = max(item[0] for item in scored)
        finalists = [
            item
            for item in scored
            if best_identity - item[0] <= TARGET_ASSOCIATION_IDENTITY_TOLERANCE
        ]

        # Apply shape hysteresis only inside the existing close identity tie.
        # A clearly better geometric association still wins, and a sole box
        # remains free to move or change scale. Within a near-tie, however,
        # retain the established width/height mode unless its shape continuity
        # is itself nearly tied; only then may confidence break the tie.
        best_shape = max(item[1] for item in finalists)
        mode_finalists = [
            item
            for item in finalists
            if best_shape - item[1] <= TARGET_ASSOCIATION_SHAPE_TOLERANCE
        ]
        return max(
            mode_finalists,
            key=lambda item: (item[2], item[0], item[1], item[3]),
        )[4]

    def _continuation_is_recent(self, measurement_ns: int) -> bool:
        if self._last_strong_measurement_ns is None:
            return False
        elapsed_ns = measurement_ns - self._last_strong_measurement_ns
        return (
            elapsed_ns >= 0
            and elapsed_ns
            <= round(TARGET_CONTINUATION_STRONG_WINDOW_SECONDS * 1_000_000_000)
        )

    def _sample_time_ns(self, measurement_ns: int | None) -> int:
        if measurement_ns is not None:
            if isinstance(measurement_ns, bool) or not isinstance(measurement_ns, int):
                raise TypeError("measurement_ns must be an integer monotonic timestamp")
            if measurement_ns < 0:
                raise ValueError("measurement_ns cannot be negative")
            sample_ns = measurement_ns
        else:
            step_ns = round(1_000_000_000 / TARGET_TRACK_REFERENCE_HZ)
            sample_ns = max(
                self._synthetic_measurement_ns,
                self._last_sample_ns or 0,
            ) + step_ns
            self._synthetic_measurement_ns = sample_ns
        if self._last_sample_ns is not None and sample_ns < self._last_sample_ns:
            raise ValueError("measurement_ns must not move backwards")
        self._last_sample_ns = sample_ns
        return sample_ns

    def _measurement_elapsed_seconds(self, measurement_ns: int) -> float:
        if self._last_measurement_ns is None or measurement_ns <= self._last_measurement_ns:
            return 0.0
        return min((measurement_ns - self._last_measurement_ns) / 1_000_000_000, 0.25)

    def _track_memory_expired(self, measurement_ns: int) -> bool:
        if self._last_measurement_ns is None:
            return False
        memory_seconds = max(
            self.track_memory_seconds,
            (self.lost_grace_frames + 1) / TARGET_TRACK_REFERENCE_HZ,
        )
        return (
            measurement_ns > self._last_measurement_ns
            and (measurement_ns - self._last_measurement_ns) / 1_000_000_000
            > memory_seconds
        )

    def _within_prediction_grace(self, measurement_ns: int) -> bool:
        if self._last_measurement_ns is None or self.lost_grace_frames <= 0:
            return False
        reference_step_ns = round(1_000_000_000 / TARGET_TRACK_REFERENCE_HZ)
        grace_ns = self.lost_grace_frames * reference_step_ns
        return (
            measurement_ns >= self._last_measurement_ns
            and measurement_ns - self._last_measurement_ns <= grace_ns
        )

    def _confirm_reacquisition(
        self,
        candidate: Detection,
        dimensions: tuple[int, int],
    ) -> bool:
        previous = self._pending_target
        if previous is None or not _detections_are_near(
            previous,
            candidate,
            dimensions,
            self.head_ratio,
        ):
            self._pending_target = candidate
            self._pending_hits = 1
        else:
            self._pending_target = candidate
            self._pending_hits += 1
        return self._pending_hits >= self.reacquire_confirmations

    def _clear_pending(self) -> None:
        self._pending_target = None
        self._pending_hits = 0

    def _forget_track(self) -> None:
        self._target = None
        self._smoothed_box = None
        self._last_observed_box = None
        self._box_velocity = (0.0, 0.0, 0.0, 0.0)
        self._last_measurement_ns = None
        self._last_strong_measurement_ns = None
        self._misses = 0
        self._clear_pending()


def _detection_iou(first: Detection, second: Detection) -> float:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _box_iou_tuple(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _box_shape_similarity(
    reference: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
) -> float:
    """Return translation-independent width/height continuity in ``[0, 1]``."""

    reference_width = max(0.0, reference[2] - reference[0])
    reference_height = max(0.0, reference[3] - reference[1])
    candidate_width = max(0.0, candidate[2] - candidate[0])
    candidate_height = max(0.0, candidate[3] - candidate[1])
    if min(
        reference_width,
        reference_height,
        candidate_width,
        candidate_height,
    ) <= 0.0:
        return 0.0
    log_shape_change = abs(math.log(candidate_width / reference_width)) + abs(
        math.log(candidate_height / reference_height)
    )
    return math.exp(-log_shape_change)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_target_point(
    box: tuple[float, float, float, float],
    head_ratio: float,
) -> tuple[float, float]:
    return (
        (box[0] + box[2]) / 2.0,
        box[1] + (box[3] - box[1]) * head_ratio,
    )


def _detections_are_near(
    first: Detection,
    second: Detection,
    dimensions: tuple[int, int],
    head_ratio: float,
) -> bool:
    if first.class_name.strip().casefold() != second.class_name.strip().casefold():
        return False
    iou = _detection_iou(first, second)
    diagonal = math.hypot(dimensions[1], dimensions[0])
    first_point = head_target_point(first, head_ratio)
    second_point = head_target_point(second, head_ratio)
    distance = math.hypot(
        first_point[0] - second_point[0],
        first_point[1] - second_point[1],
    )
    return iou >= 0.20 or (diagonal > 0.0 and distance / diagonal <= 0.04)


def _clamp_box(
    box: tuple[float, float, float, float],
    dimensions: tuple[int, int],
) -> tuple[float, float, float, float]:
    frame_height, frame_width = dimensions
    maximum_x = float(max(1, frame_width - 1))
    maximum_y = float(max(1, frame_height - 1))
    x1 = min(max(box[0], 0.0), maximum_x - 1.0)
    y1 = min(max(box[1], 0.0), maximum_y - 1.0)
    x2 = min(max(box[2], x1 + 1.0), maximum_x)
    y2 = min(max(box[3], y1 + 1.0), maximum_y)
    return x1, y1, x2, y2
