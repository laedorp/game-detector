"""Optional Linux evdev/uinput backend for user-driven precision control.

Importing this module is safe without ``python-evdev``.  Hardware access only
occurs when an explicit diagnostic, verification, or remapper operation is
requested.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import select
import time
from typing import Any, TextIO

from .codes import (
    ABS_BRAKE,
    ABS_GAS,
    ABS_HAT0X,
    ABS_HAT0Y,
    EV_ABS,
    EV_KEY,
    EV_SYN,
)
from .core import (
    AxisRange,
    DroppedEventsError,
    EventMapper,
    PrecisionConfig,
    TriggerCalibration,
)

try:  # Kept optional so the rest of the cross-platform app can import safely.
    import evdev as _evdev
except (ImportError, OSError) as exc:  # pragma: no cover - exact loader error is platform-specific
    _evdev = None
    EVDEV_IMPORT_ERROR: BaseException | None = exc
else:
    EVDEV_IMPORT_ERROR = None


PXN_VENDOR_ID = 0x36E6
PXN_P5_8K_PRODUCT_ID = 0x3016
DEFAULT_BY_ID_NAME = "usb-PXN_P5_8K_081410-event-joystick"
VIRTUAL_PHYS = "game-detector-precision/uinput"


class ControllerBackendError(RuntimeError):
    """Base class for safe, user-facing controller worker failures."""


class EvdevUnavailableError(ControllerBackendError):
    """Raised when the optional Linux input dependency is absent."""


class ControllerNotFoundError(ControllerBackendError):
    """Raised when no unambiguous requested physical controller is present."""


class MappingNotVerifiedError(ControllerBackendError):
    """Raised before grabbing hardware whose control mapping was not confirmed."""


class ControllerCapabilityError(ControllerBackendError):
    """Raised when a device does not expose the configured axes."""


class ControllerIdentityError(ControllerBackendError):
    """Raised when an event node no longer refers to the selected controller."""


@dataclass(frozen=True, slots=True)
class ControllerIdentity:
    """Immutable hardware identity checked again immediately before a grab."""

    name: str
    vendor: int | None
    product: int | None
    serial: str
    # A serial is the stable locator when one exists.  Hardware without a
    # serial must also remain on the exact persistent node and physical USB
    # port that the user verified; otherwise another unit with the same VID/PID
    # could inherit that authorization.
    phys: str = ""
    locator: str = ""

    def fingerprint(self) -> str:
        """Return the persisted, non-secret verification fingerprint."""

        serial_or_locator = self.serial.strip()
        if not serial_or_locator:
            serial_or_locator = f"{self.locator}\x1f{self.phys}"
        payload = "\x1f".join(
            (
                "game-detector-controller-mapping-v1",
                self.name.strip(),
                str(self.vendor),
                str(self.product),
                serial_or_locator,
            )
        )
        return hashlib.sha256(
            payload.encode("utf-8", errors="surrogatepass")
        ).hexdigest()

    def describe(self) -> str:
        usb_id = (
            f"{self.vendor:04x}:{self.product:04x}"
            if self.vendor is not None and self.product is not None
            else "unknown"
        )
        serial = self.serial or "<none>"
        locator = ""
        if not self.serial:
            locator = f", locator {self.locator!r}, phys {self.phys!r}"
        return f"{self.name!r} [{usb_id}, serial {serial}{locator}]"


@dataclass(frozen=True, slots=True)
class ControllerCandidate:
    """Stable controller identity discovered without taking an input grab."""

    path: Path
    event_path: Path
    name: str
    vendor: int | None
    product: int | None
    serial: str
    phys: str
    readable: bool

    @property
    def usb_id(self) -> str:
        if self.vendor is None or self.product is None:
            return "unknown"
        return f"{self.vendor:04x}:{self.product:04x}"

    @property
    def is_virtual(self) -> bool:
        return self.phys.startswith("game-detector-")

    @property
    def identity(self) -> ControllerIdentity:
        if self.serial.strip():
            return ControllerIdentity(self.name, self.vendor, self.product, self.serial)
        return ControllerIdentity(
            self.name,
            self.vendor,
            self.product,
            "",
            self.phys,
            str(self.path),
        )


@dataclass(frozen=True, slots=True)
class AxisObservation:
    code: int
    start: int
    minimum: int
    maximum: int
    declared_minimum: int
    declared_maximum: int

    def __post_init__(self) -> None:
        if self.declared_maximum <= self.declared_minimum:
            raise ValueError("declared axis maximum must be greater than minimum")
        if self.maximum < self.minimum:
            raise ValueError("observed axis maximum must not be less than minimum")
        observed = (self.start, self.minimum, self.maximum)
        if any(
            value < self.declared_minimum or value > self.declared_maximum
            for value in observed
        ):
            raise ValueError("observed axis values must be within the declared range")

    @property
    def span(self) -> int:
        return self.maximum - self.minimum

    @property
    def maximum_delta(self) -> int:
        return max(abs(self.minimum - self.start), abs(self.maximum - self.start))

    @property
    def declared_span(self) -> int:
        return self.declared_maximum - self.declared_minimum

    @property
    def movement_fraction(self) -> float:
        return self.span / self.declared_span

    @property
    def maximum_delta_fraction(self) -> float:
        return self.maximum_delta / self.declared_span


@dataclass(frozen=True, slots=True)
class MappingVerificationReport:
    """Observed physical changes used to confirm, never infer silently, a mapping."""

    phase: str
    observations: tuple[AxisObservation, ...]
    changed_buttons: tuple[int, ...]

    @property
    def changed_axes(self) -> tuple[int, ...]:
        return tuple(item.code for item in self.observations if item.span > 0)


def require_evdev() -> Any:
    if _evdev is None:
        detail = f": {EVDEV_IMPORT_ERROR}" if EVDEV_IMPORT_ERROR else ""
        raise EvdevUnavailableError(
            "Linux controller precision requires the optional 'evdev' package" + detail
        )
    return _evdev


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _read_hex(path: Path) -> int | None:
    text = _read_text(path)
    try:
        return int(text, 16)
    except ValueError:
        return None


def _candidate_from_event(
    stable_path: Path,
    event_path: Path,
    *,
    sys_class_input: Path,
) -> ControllerCandidate | None:
    event_name = event_path.name
    if not event_name.startswith("event"):
        return None
    device_root = sys_class_input / event_name / "device"
    name = _read_text(device_root / "name")
    if not name:
        return None
    return ControllerCandidate(
        path=stable_path,
        event_path=event_path,
        name=name,
        vendor=_read_hex(device_root / "id" / "vendor"),
        product=_read_hex(device_root / "id" / "product"),
        serial=_read_text(device_root / "uniq"),
        phys=_read_text(device_root / "phys"),
        readable=os.access(stable_path, os.R_OK),
    )


def _capability_bit_is_set(text: str, code: int) -> bool:
    """Decode space-separated native-word bitmaps exposed by sysfs."""

    try:
        words = [int(part, 16) for part in text.split()]
    except ValueError:
        return False
    word_index, bit_index = divmod(code, 64)
    if word_index >= len(words):
        return False
    return bool(words[-1 - word_index] & (1 << bit_index))


def _looks_like_gamepad(device_root: Path) -> bool:
    try:
        absolute_mask = int(_read_text(device_root / "capabilities" / "abs"), 16)
    except ValueError:
        return False
    # BTN_GAMEPAD is an alias of BTN_SOUTH at code 304.  The Linux gamepad
    # specification recommends it as the identifying capability.
    return absolute_mask != 0 and _capability_bit_is_set(
        _read_text(device_root / "capabilities" / "key"),
        304,
    )


def discover_controllers(
    *,
    by_id_root: Path = Path("/dev/input/by-id"),
    input_root: Path = Path("/dev/input"),
    sys_class_input: Path = Path("/sys/class/input"),
) -> tuple[ControllerCandidate, ...]:
    """Discover joystick event nodes, preferring persistent ``by-id`` paths."""

    candidates: list[ControllerCandidate] = []
    seen_events: set[Path] = set()
    try:
        stable_paths = sorted(by_id_root.glob("*-event-joystick"))
    except OSError:
        stable_paths = []

    for stable_path in stable_paths:
        try:
            event_path = stable_path.resolve(strict=True)
        except OSError:
            continue
        candidate = _candidate_from_event(
            stable_path,
            event_path,
            sys_class_input=sys_class_input,
        )
        if candidate is not None and not candidate.is_virtual:
            candidates.append(candidate)
            seen_events.add(event_path)

    try:
        event_paths = sorted(input_root.glob("event*"))
    except OSError:
        event_paths = []
    for event_path in event_paths:
        try:
            resolved = event_path.resolve(strict=True)
        except OSError:
            continue
        if resolved in seen_events:
            continue
        device_root = sys_class_input / resolved.name / "device"
        if not _looks_like_gamepad(device_root):
            continue
        candidate = _candidate_from_event(
            event_path,
            resolved,
            sys_class_input=sys_class_input,
        )
        if candidate is not None and not candidate.is_virtual:
            candidates.append(candidate)

    return tuple(candidates)


def select_controller(
    candidates: Sequence[ControllerCandidate],
    *,
    device: str | os.PathLike[str] | None = None,
    vendor: int | None = PXN_VENDOR_ID,
    product: int | None = PXN_P5_8K_PRODUCT_ID,
    serial: str | None = None,
) -> ControllerCandidate:
    """Select one physical controller without guessing through ambiguity."""

    pool = [candidate for candidate in candidates if not candidate.is_virtual]
    if device is not None:
        requested = Path(device).expanduser()
        try:
            requested_resolved = requested.resolve(strict=False)
        except OSError:
            requested_resolved = requested
        pool = [
            candidate
            for candidate in pool
            if candidate.path == requested
            or candidate.event_path == requested
            or candidate.event_path == requested_resolved
        ]
    # An explicit path identifies where to open, not what hardware is trusted.
    # Apply the requested known-profile identity in either selection mode.
    if vendor is not None:
        pool = [candidate for candidate in pool if candidate.vendor == vendor]
    if product is not None:
        pool = [candidate for candidate in pool if candidate.product == product]
    if serial:
        pool = [candidate for candidate in pool if candidate.serial == serial]

    if not pool:
        if device:
            requested_name = str(device)
        elif vendor is not None and product is not None:
            requested_name = f"{vendor:04x}:{product:04x}"
        else:
            requested_name = "the requested profile"
        raise ControllerNotFoundError(f"controller {requested_name} was not found")
    if len(pool) > 1:
        paths = ", ".join(str(candidate.path) for candidate in pool)
        raise ControllerNotFoundError(
            "more than one controller matched; choose --device or --serial: " + paths
        )
    return pool[0]


def describe_candidate(candidate: ControllerCandidate) -> str:
    access = "readable" if candidate.readable else "permission denied"
    identity = f", serial {candidate.serial}" if candidate.serial else ""
    return f"{candidate.name} [{candidate.usb_id}{identity}] at {candidate.path} ({access})"


def _capability_axis_map(source: Any) -> dict[int, Any]:
    capabilities = source.capabilities(absinfo=True)
    result: dict[int, Any] = {}
    for entry in capabilities.get(EV_ABS, ()):
        if isinstance(entry, tuple) and len(entry) == 2:
            result[int(entry[0])] = entry[1]
        else:
            result[int(entry)] = source.absinfo(int(entry))
    return result


def _info_value(info: Any, short_name: str, long_name: str) -> int:
    if hasattr(info, short_name):
        return int(getattr(info, short_name))
    return int(getattr(info, long_name))


def _axis_range(info: Any) -> AxisRange:
    minimum = _info_value(info, "min", "minimum")
    maximum = _info_value(info, "max", "maximum")
    return AxisRange(minimum, maximum)


def _axis_current(info: Any) -> int:
    return int(getattr(info, "value"))


def _open_device_identity(
    source: Any,
    device_path: str | os.PathLike[str],
) -> ControllerIdentity:
    """Build an identity from an already-open event node.

    ``uniq`` is not populated by every gamepad.  In that case the physical
    input locator and the exact path selected by the launcher are part of the
    identity, matching :attr:`ControllerCandidate.identity`.
    """

    info = source.info
    raw_serial = getattr(source, "uniq", "")
    serial = "" if raw_serial is None else str(raw_serial).strip()
    if serial:
        return ControllerIdentity(
            name=str(getattr(source, "name", "")),
            vendor=int(info.vendor),
            product=int(info.product),
            serial=serial,
        )
    raw_phys = getattr(source, "phys", "")
    return ControllerIdentity(
        name=str(getattr(source, "name", "")),
        vendor=int(info.vendor),
        product=int(info.product),
        serial="",
        phys="" if raw_phys is None else str(raw_phys),
        locator=str(device_path),
    )


def _validate_open_device_identity(
    source: Any,
    device_path: str | os.PathLike[str],
    expected_identity: ControllerIdentity,
) -> None:
    actual = _open_device_identity(source, device_path)
    if actual != expected_identity:
        raise ControllerIdentityError(
            "selected controller identity changed before input access; "
            f"expected {expected_identity.describe()}, got {actual.describe()}. "
            "Reconnect the PXN controller and select it again"
        )


def _default_uinput_factory(source: Any) -> Any:
    evdev = require_evdev()
    info = source.info
    try:
        return evdev.UInput.from_device(
            source,
            name=source.name,
            vendor=int(info.vendor),
            product=int(info.product),
            version=int(info.version),
            bustype=int(info.bustype),
            phys=VIRTUAL_PHYS,
        )
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise ControllerBackendError(
                "/dev/uinput is missing. Load the Linux uinput module, then rerun "
                "--diagnose before starting precision control"
            ) from exc
        if exc.errno in (errno.EACCES, errno.EPERM):
            raise ControllerBackendError(
                "permission to /dev/uinput was denied. Check that the active desktop "
                "session has uaccess permission; do not run the whole app as root"
            ) from exc
        raise


class EvdevPrecisionController:
    """Exclusive physical-controller proxy with deterministic cleanup."""

    def __init__(
        self,
        device_path: str | os.PathLike[str],
        *,
        expected_identity: ControllerIdentity,
        config: PrecisionConfig | None = None,
        mapping_verified: bool = False,
        trigger_calibration: TriggerCalibration | None = None,
        input_device_factory: Callable[[str], Any] | None = None,
        uinput_factory: Callable[[Any], Any] | None = None,
        select_fn: Callable[..., Any] = select.select,
    ) -> None:
        self.device_path = str(device_path)
        self.expected_identity = expected_identity
        self.config = config or PrecisionConfig()
        self.mapping_verified = bool(mapping_verified)
        self.trigger_calibration = trigger_calibration
        self._input_device_factory = input_device_factory
        self._uinput_factory = uinput_factory or _default_uinput_factory
        self._select = select_fn
        self.source: Any | None = None
        self.virtual: Any | None = None
        self.mapper: EventMapper[Any] | None = None
        self.axis_info: dict[int, Any] = {}
        self._grabbed = False
        self._forwarded_keys: set[int] = set()

    @property
    def active(self) -> bool:
        return self.source is not None and self.virtual is not None

    def _make_source(self) -> Any:
        try:
            if self._input_device_factory is not None:
                return self._input_device_factory(self.device_path)
            return require_evdev().InputDevice(self.device_path)
        except FileNotFoundError as exc:
            raise ControllerNotFoundError(
                f"controller disappeared or is not connected: {self.device_path}"
            ) from exc
        except PermissionError as exc:
            raise ControllerBackendError(
                "permission to the physical controller was denied. Reconnect it in "
                "the active desktop session and rerun --diagnose"
            ) from exc

    def _validate_source_identity(self) -> None:
        if self.source is None:
            raise ControllerIdentityError("physical controller is not open")
        _validate_open_device_identity(
            self.source,
            self.device_path,
            self.expected_identity,
        )

    def open(self) -> None:
        if self.active:
            return
        if not self.mapping_verified:
            raise MappingNotVerifiedError(
                "controller mapping is not verified; run --verify-mapping, then "
                "start with --confirm-default-mapping"
            )

        try:
            self.source = self._make_source()
            # A stable by-id symlink can disappear and an event number can be
            # reused between discovery and open.  Revalidate all known identity
            # fields before capabilities, EVIOCGRAB, or uinput creation.
            self._validate_source_identity()
            if str(getattr(self.source, "phys", "")).startswith(
                "game-detector-precision/"
            ):
                raise ControllerCapabilityError("refusing to use the virtual controller as input")

            self.axis_info = _capability_axis_map(self.source)
            required = (
                self.config.right_x_code,
                self.config.right_y_code,
                self.config.trigger_axis_code,
            )
            missing = [code for code in required if code not in self.axis_info]
            if missing:
                joined = ", ".join(str(code) for code in missing)
                raise ControllerCapabilityError(
                    "controller is missing configured absolute axis code(s): " + joined
                )

            x_info = self.axis_info[self.config.right_x_code]
            y_info = self.axis_info[self.config.right_y_code]
            trigger_info = self.axis_info[self.config.trigger_axis_code]
            trigger_minimum = _info_value(trigger_info, "min", "minimum")
            trigger_maximum = _info_value(trigger_info, "max", "maximum")
            trigger_range = self.trigger_calibration or TriggerCalibration(
                rest=trigger_minimum,
                pressed=trigger_maximum,
            )
            if not (
                trigger_minimum <= trigger_range.rest <= trigger_maximum
                and trigger_minimum <= trigger_range.pressed <= trigger_maximum
            ):
                raise ControllerCapabilityError(
                    "calibrated trigger values must be within the declared range "
                    f"{trigger_minimum}..{trigger_maximum}"
                )
            active_keys = set(self.source.active_keys())
            self.mapper = EventMapper(
                self.config,
                x_range=_axis_range(x_info),
                y_range=_axis_range(y_info),
                trigger_calibration=trigger_range,
                initial_x=_axis_current(x_info),
                initial_y=_axis_current(y_info),
                initial_trigger=_axis_current(trigger_info),
                trigger_button_pressed=(
                    self.config.trigger_button_code in active_keys
                    if self.config.trigger_button_code is not None
                    else False
                ),
            )

            self.source.grab()
            self._grabbed = True
            self.virtual = self._uinput_factory(self.source)
            self._emit_initial_state(active_keys)
        except BaseException:
            self.close()
            raise

    def _emit_initial_state(self, active_keys: set[int]) -> None:
        if self.virtual is None or self.mapper is None:
            return
        right_x, right_y = self.mapper.current_output()
        for code, info in self.axis_info.items():
            value = _axis_current(info)
            if code == self.config.right_x_code:
                value = right_x
            elif code == self.config.right_y_code:
                value = right_y
            self.virtual.write(EV_ABS, code, value)
        for code in active_keys:
            self.virtual.write(EV_KEY, code, 1)
            self._forwarded_keys.add(code)
        self.virtual.syn()

    def _emit(self, mapped: Any) -> None:
        if self.virtual is None:
            raise ControllerBackendError("virtual controller is not open")
        if mapped.source is not None:
            self.virtual.write_event(mapped.source)
        else:
            self.virtual.write(mapped.type, mapped.code, mapped.value)
        if mapped.type == EV_KEY:
            if mapped.value:
                self._forwarded_keys.add(mapped.code)
            else:
                self._forwarded_keys.discard(mapped.code)

    def _live_events(self, should_stop: Callable[[], bool]) -> Iterator[Any]:
        if self.source is None:
            return
        while not should_stop():
            try:
                readable, _, _ = self._select([self.source], [], [], 0.25)
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                raise
            if not readable:
                continue
            try:
                yield from self.source.read()
            except BlockingIOError:
                continue

    def run(
        self,
        *,
        events: Iterable[Any] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        """Run until stopped, input ends, or a safe-fail exception occurs."""

        stop = should_stop or (lambda: False)
        self.open()
        try:
            # Readiness is observable only after the physical source has been
            # identity-checked and grabbed, uinput has been created, and its
            # initial state has been synchronized.  Keeping the callback in
            # this cleanup scope ensures even an output/IPC failure releases
            # both devices.
            if on_ready is not None:
                on_ready()
            source_events = self._live_events(stop) if events is None else iter(events)
            for event in source_events:
                if stop():
                    break
                if self.mapper is None:
                    raise ControllerBackendError("controller event mapper is not initialized")
                for mapped in self.mapper.feed(event):
                    self._emit(mapped)
        except DroppedEventsError:
            # Continuing after SYN_DROPPED risks a stuck or incorrect control
            # state.  Closing the virtual controller is the conservative path.
            raise
        finally:
            self.close()

    def _neutral_value(self, code: int, info: Any) -> int:
        minimum = _info_value(info, "min", "minimum")
        maximum = _info_value(info, "max", "maximum")
        if code in (ABS_HAT0X, ABS_HAT0Y):
            return 0
        if code in (ABS_GAS, ABS_BRAKE):
            return minimum
        return int(round((minimum + maximum) / 2.0))

    def _release_virtual_state(self) -> None:
        if self.virtual is None:
            return
        for code in sorted(self._forwarded_keys):
            self.virtual.write(EV_KEY, code, 0)
        for code, info in self.axis_info.items():
            self.virtual.write(EV_ABS, code, self._neutral_value(code, info))
        self.virtual.syn()
        self._forwarded_keys.clear()

    def close(self) -> None:
        virtual, source = self.virtual, self.source
        try:
            if virtual is not None:
                try:
                    self._release_virtual_state()
                except (OSError, ValueError):
                    pass
                try:
                    virtual.close()
                except OSError:
                    pass
        finally:
            self.virtual = None
            if source is not None and self._grabbed:
                try:
                    source.ungrab()
                except OSError:
                    pass
            self._grabbed = False
            if source is not None:
                try:
                    source.close()
                except OSError:
                    pass
            self.source = None
            self.mapper = None
            self.axis_info = {}
            self._forwarded_keys.clear()

    def __enter__(self) -> "EvdevPrecisionController":
        self.open()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def observe_phase(
    device_path: str | os.PathLike[str],
    *,
    phase: str,
    seconds: float,
    expected_identity: ControllerIdentity | None = None,
    input_device_factory: Callable[[str], Any] | None = None,
    select_fn: Callable[..., Any] = select.select,
    clock: Callable[[], float] = time.monotonic,
) -> MappingVerificationReport:
    """Observe controls read-only for a calibration phase without grabbing."""

    if not seconds > 0.0:
        raise ValueError("observation time must be greater than zero")
    source = (
        input_device_factory(str(device_path))
        if input_device_factory is not None
        else require_evdev().InputDevice(str(device_path))
    )
    try:
        if expected_identity is not None:
            _validate_open_device_identity(source, device_path, expected_identity)
        axis_info = _capability_axis_map(source)
        values = {code: _axis_current(info) for code, info in axis_info.items()}
        minima = dict(values)
        maxima = dict(values)
        changed_buttons: set[int] = set()
        deadline = clock() + seconds
        while clock() < deadline:
            timeout = max(0.0, min(0.1, deadline - clock()))
            readable, _, _ = select_fn([source], [], [], timeout)
            if not readable:
                continue
            try:
                events = source.read()
            except BlockingIOError:
                continue
            for event in events:
                if event.type == EV_ABS and event.code in values:
                    minima[event.code] = min(minima[event.code], event.value)
                    maxima[event.code] = max(maxima[event.code], event.value)
                elif event.type == EV_KEY and event.value:
                    changed_buttons.add(event.code)
        observations = tuple(
            AxisObservation(
                code,
                values[code],
                minima[code],
                maxima[code],
                _info_value(axis_info[code], "min", "minimum"),
                _info_value(axis_info[code], "max", "maximum"),
            )
            for code in sorted(values)
        )
        return MappingVerificationReport(
            phase=phase,
            observations=observations,
            changed_buttons=tuple(sorted(changed_buttons)),
        )
    finally:
        source.close()


def print_diagnostics(
    candidates: Sequence[ControllerCandidate],
    *,
    output: TextIO,
    uinput_path: Path = Path("/dev/uinput"),
) -> None:
    """Print read-only device and dependency diagnostics."""

    if _evdev is None:
        print(f"evdev: unavailable ({EVDEV_IMPORT_ERROR})", file=output)
    else:
        print("evdev: available", file=output)
    if not candidates:
        print("controllers: none found", file=output)
    else:
        print("controllers:", file=output)
        for candidate in candidates:
            print(f"  - {describe_candidate(candidate)}", file=output)
    if uinput_path.exists():
        status = "writable" if os.access(uinput_path, os.W_OK) else "permission denied"
        print(f"uinput: {uinput_path} ({status})", file=output)
    else:
        print(f"uinput: {uinput_path} not present (the uinput module may not be loaded)", file=output)
