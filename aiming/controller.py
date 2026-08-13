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
        self.threshold = max(0.0, min(float(threshold), 1.0))
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

    def target_key(detection: Detection) -> tuple[float, float]:
        target_x, target_y = head_target_point(detection, head_ratio)
        distance_squared = (target_x - center_x) ** 2 + (target_y - center_y) ** 2
        return distance_squared, -detection.confidence

    return min(candidates, key=target_key)


class TargetTracker:
    """Keep one latest detection stable through short misses and nearby rivals."""

    def __init__(
        self,
        *,
        label: str | None = None,
        head_ratio: float = DEFAULT_HEAD_RATIO,
        lost_grace_frames: int = 3,
    ) -> None:
        if not 0.0 <= head_ratio <= 0.5:
            raise ValueError("head_ratio must be between 0 and 0.5")
        if lost_grace_frames < 0:
            raise ValueError("lost_grace_frames cannot be negative")
        self.label = label
        self.head_ratio = head_ratio
        self.lost_grace_frames = int(lost_grace_frames)
        self._target: Detection | None = None
        self._smoothed_box: tuple[float, float, float, float] | None = None
        self._box_velocity = (0.0, 0.0, 0.0, 0.0)
        self._misses = 0
        self._frame_dimensions: tuple[int, int] | None = None

    def reset(self) -> None:
        self._target = None
        self._smoothed_box = None
        self._box_velocity = (0.0, 0.0, 0.0, 0.0)
        self._misses = 0
        self._frame_dimensions = None

    def update(
        self,
        detections: list[Detection] | tuple[Detection, ...],
        frame_shape: tuple[int, ...],
    ) -> Detection | None:
        dimensions = (int(frame_shape[0]), int(frame_shape[1]))
        if self._frame_dimensions is not None and dimensions != self._frame_dimensions:
            self.reset()
        self._frame_dimensions = dimensions
        candidates = list(detections)
        if self.label is not None:
            normalized_label = self.label.strip().lower()
            candidates = [
                detection
                for detection in candidates
                if detection.class_name.strip().lower() == normalized_label
            ]

        if not candidates:
            if self._target is not None and self._misses < self.lost_grace_frames:
                self._misses += 1
                self._target = self._predict_missing_target(dimensions)
                return self._target
            self._target = None
            self._smoothed_box = None
            self._box_velocity = (0.0, 0.0, 0.0, 0.0)
            self._misses = 0
            return None

        associated = self._associated_candidate(candidates, dimensions)
        if associated is None:
            selected = choose_target(
                candidates,
                frame_shape=frame_shape,
                head_ratio=self.head_ratio,
            )
            assert selected is not None
            self._target = self._start_track(selected)
        else:
            self._target = self._smooth_measurement(associated, dimensions)
        self._misses = 0
        return self._target

    def _start_track(self, detection: Detection) -> Detection:
        self._smoothed_box = detection.xyxy
        self._box_velocity = (0.0, 0.0, 0.0, 0.0)
        return detection

    def _smooth_measurement(
        self,
        detection: Detection,
        dimensions: tuple[int, int],
    ) -> Detection:
        if self._smoothed_box is None:
            return self._start_track(detection)
        frame_height, _frame_width = dimensions
        height_fraction = max(0.0, detection.y2 - detection.y1) / frame_height
        alpha = min(max(0.52 + height_fraction * 1.4, 0.58), 0.82)
        beta = alpha * 0.22
        predicted = tuple(
            value + velocity
            for value, velocity in zip(self._smoothed_box, self._box_velocity)
        )
        residual = tuple(
            measured - estimate
            for measured, estimate in zip(detection.xyxy, predicted)
        )
        smoothed = tuple(
            estimate + alpha * difference
            for estimate, difference in zip(predicted, residual)
        )
        self._box_velocity = tuple(
            (velocity + beta * difference) * 0.88
            for velocity, difference in zip(self._box_velocity, residual)
        )
        self._smoothed_box = _clamp_box(smoothed, dimensions)
        led_box = _clamp_box(
            tuple(
                value + velocity * 0.65
                for value, velocity in zip(self._smoothed_box, self._box_velocity)
            ),
            dimensions,
        )
        return Detection(
            detection.class_id,
            detection.class_name,
            detection.confidence,
            led_box,
        )

    def _predict_missing_target(
        self,
        dimensions: tuple[int, int],
    ) -> Detection:
        assert self._target is not None
        if self._smoothed_box is None:
            self._smoothed_box = self._target.xyxy
        predicted = tuple(
            value + velocity
            for value, velocity in zip(self._smoothed_box, self._box_velocity)
        )
        self._smoothed_box = _clamp_box(predicted, dimensions)
        self._box_velocity = tuple(velocity * 0.72 for velocity in self._box_velocity)
        return Detection(
            self._target.class_id,
            self._target.class_name,
            self._target.confidence,
            self._smoothed_box,
        )

    def _associated_candidate(
        self,
        candidates: list[Detection],
        dimensions: tuple[int, int],
    ) -> Detection | None:
        if self._target is None:
            return None
        frame_height, frame_width = dimensions
        diagonal = math.hypot(frame_width, frame_height)
        previous_point = head_target_point(self._target, self.head_ratio)
        scored: list[tuple[float, float, Detection]] = []
        for candidate in candidates:
            iou = _detection_iou(self._target, candidate)
            point = head_target_point(candidate, self.head_ratio)
            distance = math.hypot(
                point[0] - previous_point[0],
                point[1] - previous_point[1],
            )
            normalized_distance = distance / diagonal if diagonal > 0.0 else 1.0
            if iou >= 0.10 or normalized_distance <= 0.08:
                scored.append((-iou, normalized_distance, candidate))
        if not scored:
            return None
        return min(scored, key=lambda item: (item[0], item[1]))[2]


def _detection_iou(first: Detection, second: Detection) -> float:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


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
