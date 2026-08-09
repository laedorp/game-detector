"""Command-line diagnostics and worker entry point for controller precision."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import math
import os
from pathlib import Path
import re
import shlex
import signal
import sys
from typing import Any, TextIO

from .codes import (
    ABSOLUTE_CODE_NAMES,
    ABSOLUTE_NAME_CODES,
    ABS_BRAKE,
    ABS_RZ,
    ABS_Z,
)
from .core import DroppedEventsError, PrecisionConfig, TriggerCalibration
from .linux_evdev import (
    AxisObservation,
    ControllerBackendError,
    ControllerCandidate,
    ControllerIdentityError,
    EvdevPrecisionController,
    MappingVerificationReport,
    PXN_P5_8K_PRODUCT_ID,
    PXN_VENDOR_ID,
    describe_candidate,
    discover_controllers,
    observe_phase,
    print_diagnostics,
    select_controller,
)
from .protocol import CONTROLLER_READY_SENTINEL


def _usb_id(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use a decimal or 0x-prefixed USB ID") from exc
    if not 0 <= parsed <= 0xFFFF:
        raise argparse.ArgumentTypeError("USB ID must be between 0 and 0xffff")
    return parsed


def _axis_code(value: str) -> int:
    normalized = value.strip().upper()
    if normalized in ABSOLUTE_NAME_CODES:
        return ABSOLUTE_NAME_CODES[normalized]
    try:
        code = int(value, 0)
    except ValueError as exc:
        choices = ", ".join(sorted(ABSOLUTE_NAME_CODES))
        raise argparse.ArgumentTypeError(f"use an axis number or one of: {choices}") from exc
    if code < 0:
        raise argparse.ArgumentTypeError("axis code cannot be negative")
    return code


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _positive_pid(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("parent PID must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("parent PID must be greater than zero")
    return parsed


def _expected_fingerprint(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise argparse.ArgumentTypeError(
            "expected controller fingerprint must be 64 hexadecimal characters"
        )
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m controller_precision",
        description=(
            "User-driven LT precision controller for Linux. It never reads "
            "video, detections, target coordinates, or network messages."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--diagnose",
        action="store_true",
        help="list controller and uinput readiness without opening either device",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="show the selected mapping without grabbing or creating a controller",
    )
    mode.add_argument(
        "--verify-mapping",
        action="store_true",
        help="observe LT and right-stick movement read-only before first use",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="run the exclusive physical-to-virtual controller worker",
    )
    parser.add_argument("--device", help="explicit /dev/input/by-id/...-event-joystick path")
    parser.add_argument("--vendor", type=_usb_id, default=PXN_VENDOR_ID)
    parser.add_argument("--product", type=_usb_id, default=PXN_P5_8K_PRODUCT_ID)
    parser.add_argument("--serial", help="USB serial used to disambiguate identical controllers")
    parser.add_argument(
        "--expected-fingerprint",
        type=_expected_fingerprint,
        help=(
            "device-bound mapping fingerprint supplied by the launcher; optional "
            "for direct command-line use"
        ),
    )
    parser.add_argument(
        "--right-x",
        type=_axis_code,
        default=ABS_Z,
        metavar="AXIS",
        help="right-stick horizontal axis (default: ABS_Z)",
    )
    parser.add_argument(
        "--right-y",
        type=_axis_code,
        default=ABS_RZ,
        metavar="AXIS",
        help="right-stick vertical axis (default: ABS_RZ)",
    )
    parser.add_argument(
        "--lt-axis",
        type=_axis_code,
        default=ABS_BRAKE,
        metavar="AXIS",
        help="analog activation axis (default: ABS_BRAKE)",
    )
    parser.add_argument(
        "--strength",
        type=_finite_float,
        default=0.35,
        help="maximum right-stick magnitude while LT is held (default: 0.35)",
    )
    parser.add_argument(
        "--exponent",
        type=_finite_float,
        default=1.4,
        help="radial fine-control curve exponent (default: 1.4)",
    )
    parser.add_argument(
        "--deadzone",
        type=_finite_float,
        default=0.04,
        help="physical right-stick deadzone while LT is held (default: 0.04)",
    )
    parser.add_argument(
        "--trigger-rest",
        type=int,
        help="calibrated LT rest value; defaults to the device-reported minimum",
    )
    parser.add_argument(
        "--trigger-pressed",
        type=int,
        help="calibrated fully pressed LT value; defaults to device-reported maximum",
    )
    parser.add_argument(
        "--verification-seconds",
        type=_finite_float,
        default=3.0,
        help="seconds per read-only verification phase (default: 3)",
    )
    parser.add_argument(
        "--confirm-default-mapping",
        action="store_true",
        help=(
            "confirm that verification showed the selected trigger and right-stick "
            "axes (the known PXN default is ABS_BRAKE with ABS_Z/ABS_RZ); required "
            "by --run"
        ),
    )
    parser.add_argument(
        "--parent-pid",
        type=_positive_pid,
        help=(
            "exit and release the controller if this launcher PID is no longer "
            "the worker's parent"
        ),
    )
    return parser


def _select_from_arguments(
    args: argparse.Namespace,
    candidates: Sequence[ControllerCandidate],
) -> ControllerCandidate:
    selected = select_controller(
        candidates,
        device=args.device,
        vendor=args.vendor,
        product=args.product,
        serial=args.serial,
    )
    if (
        args.expected_fingerprint is not None
        and selected.identity.fingerprint() != args.expected_fingerprint
    ):
        raise ControllerIdentityError(
            "the selected controller no longer matches the device whose mapping "
            "was verified; reconnect it and verify the mapping again"
        )
    return selected


def _axis_name(code: int) -> str:
    return ABSOLUTE_CODE_NAMES.get(code, f"ABS_{code}")


def _axis_argument(code: int) -> str:
    return ABSOLUTE_CODE_NAMES.get(code, str(code))


def _print_verification_report(report: MappingVerificationReport, output: TextIO) -> None:
    changed = [item for item in report.observations if item.span > 0]
    print(f"{report.phase} observed axis changes:", file=output)
    if not changed:
        print("  none", file=output)
    for item in sorted(changed, key=lambda observation: observation.maximum_delta, reverse=True):
        print(
            f"  {_axis_name(item.code)}: start={item.start}, "
            f"observed={item.minimum}..{item.maximum}, "
            f"declared={item.declared_minimum}..{item.declared_maximum}, "
            f"movement={item.movement_fraction:.0%}",
            file=output,
        )
    if report.changed_buttons:
        codes = ", ".join(str(code) for code in report.changed_buttons)
        print(f"  pressed button codes: {codes}", file=output)


MIN_EXPECTED_MOVEMENT_FRACTION = 0.50
MAX_UNEXPECTED_MOVEMENT_FRACTION = 0.12
MAX_TRIGGER_REST_ENDPOINT_FRACTION = 0.15


def _observation(
    report: MappingVerificationReport,
    code: int,
) -> AxisObservation | None:
    return next((item for item in report.observations if item.code == code), None)


def _unexpected_movements(
    report: MappingVerificationReport,
    allowed_codes: set[int],
) -> list[AxisObservation]:
    return [
        item
        for item in report.observations
        if item.code not in allowed_codes
        and item.movement_fraction >= MAX_UNEXPECTED_MOVEMENT_FRACTION
    ]


def _calibrated_run_command(
    candidate: ControllerCandidate,
    config: PrecisionConfig,
    calibration: TriggerCalibration,
) -> str:
    arguments = [
        "python",
        "-m",
        "controller_precision",
        "--run",
        "--device",
        str(candidate.path),
    ]
    if candidate.vendor is not None:
        arguments.extend(("--vendor", hex(candidate.vendor)))
    if candidate.product is not None:
        arguments.extend(("--product", hex(candidate.product)))
    if candidate.serial:
        arguments.extend(("--serial", candidate.serial))
    arguments.extend(("--expected-fingerprint", candidate.identity.fingerprint()))
    arguments.extend(
        (
            "--lt-axis",
            _axis_argument(config.trigger_axis_code),
            "--right-x",
            _axis_argument(config.right_x_code),
            "--right-y",
            _axis_argument(config.right_y_code),
            "--trigger-rest",
            str(calibration.rest),
            "--trigger-pressed",
            str(calibration.pressed),
            "--strength",
            str(config.strength),
            "--exponent",
            str(config.exponent),
            "--deadzone",
            str(config.deadzone),
            "--confirm-default-mapping",
        )
    )
    return shlex.join(arguments)


def verify_default_mapping(
    candidate: ControllerCandidate,
    *,
    config: PrecisionConfig,
    seconds: float,
    output: TextIO,
    observer: Callable[..., MappingVerificationReport] = observe_phase,
) -> TriggerCalibration | None:
    """Verify configured axes and derive the analog trigger's real polarity."""

    if not seconds > 0.0:
        raise ValueError("verification seconds must be greater than zero")
    print(
        "Verification is read-only: it will not grab the controller or create input.\n"
        f"Release {_axis_name(config.trigger_axis_code)}, then repeatedly press it "
        "fully during the first phase. Do not move either stick.",
        file=output,
    )
    lt_report = observer(
        candidate.path,
        phase="LT",
        seconds=seconds,
        expected_identity=candidate.identity,
    )
    _print_verification_report(lt_report, output)
    print(
        "Now release the trigger and move only the configured right stick in full circles.",
        file=output,
    )
    stick_report = observer(
        candidate.path,
        phase="right stick",
        seconds=seconds,
        expected_identity=candidate.identity,
    )
    _print_verification_report(stick_report, output)

    failures: list[str] = []
    trigger = _observation(lt_report, config.trigger_axis_code)
    calibration: TriggerCalibration | None = None
    if trigger is None:
        failures.append(
            f"configured trigger {_axis_name(config.trigger_axis_code)} was not reported"
        )
    else:
        distance_from_min = trigger.start - trigger.declared_minimum
        distance_from_max = trigger.declared_maximum - trigger.start
        nearest_endpoint_fraction = min(distance_from_min, distance_from_max) / trigger.declared_span
        if nearest_endpoint_fraction > MAX_TRIGGER_REST_ENDPOINT_FRACTION:
            failures.append(
                f"{_axis_name(trigger.code)} rest value {trigger.start} is not near either "
                f"declared endpoint {trigger.declared_minimum}..{trigger.declared_maximum}"
            )
        elif trigger.maximum_delta_fraction < MIN_EXPECTED_MOVEMENT_FRACTION:
            failures.append(
                f"{_axis_name(trigger.code)} moved only "
                f"{trigger.maximum_delta_fraction:.0%} of its declared range"
            )
        else:
            rests_at_minimum = distance_from_min <= distance_from_max
            pressed = trigger.maximum if rests_at_minimum else trigger.minimum
            try:
                calibration = TriggerCalibration(rest=trigger.start, pressed=pressed)
            except ValueError as exc:
                failures.append(str(exc))

    for code in (config.right_x_code, config.right_y_code):
        stick_axis = _observation(stick_report, code)
        if stick_axis is None:
            failures.append(f"configured right-stick axis {_axis_name(code)} was not reported")
        elif stick_axis.movement_fraction < MIN_EXPECTED_MOVEMENT_FRACTION:
            failures.append(
                f"{_axis_name(code)} covered only {stick_axis.movement_fraction:.0%} "
                "of its declared range"
            )

    unexpected = [
        (lt_report.phase, item)
        for item in _unexpected_movements(lt_report, {config.trigger_axis_code})
    ]
    unexpected.extend(
        (stick_report.phase, item)
        for item in _unexpected_movements(
            stick_report,
            {config.right_x_code, config.right_y_code},
        )
    )
    for phase, item in unexpected:
        failures.append(
            f"unexpected {_axis_name(item.code)} movement during {phase} "
            f"({item.movement_fraction:.0%} of its declared range)"
        )

    if not failures and calibration is not None:
        polarity = "increasing" if calibration.pressed > calibration.rest else "decreasing"
        print(
            "Mapping verified: trigger={} (rest {}, pressed {}, {} polarity); "
            "right stick={}/{}.".format(
                _axis_name(config.trigger_axis_code),
                calibration.rest,
                calibration.pressed,
                polarity,
                _axis_name(config.right_x_code),
                _axis_name(config.right_y_code),
            ),
            file=output,
        )
        print("Calibrated run command:", file=output)
        print(_calibrated_run_command(candidate, config, calibration), file=output)
        return calibration
    else:
        print("Mapping was not verified:", file=output)
        for failure in failures:
            print(f"  - {failure}", file=output)
        print(
            "Do not start the worker with this mapping; repeat verification or "
            "choose explicit axis options.",
            file=output,
        )
        return None


def _trigger_calibration(args: argparse.Namespace) -> TriggerCalibration | None:
    if args.trigger_rest is None and args.trigger_pressed is None:
        return None
    if args.trigger_rest is None or args.trigger_pressed is None:
        raise ValueError("provide both --trigger-rest and --trigger-pressed")
    return TriggerCalibration(args.trigger_rest, args.trigger_pressed)


def _config(args: argparse.Namespace) -> PrecisionConfig:
    return PrecisionConfig(
        strength=args.strength,
        exponent=args.exponent,
        deadzone=args.deadzone,
        right_x_code=args.right_x,
        right_y_code=args.right_y,
        trigger_axis_code=args.lt_axis,
        # Digital LT remains disabled unless a future calibration identifies a
        # trustworthy companion button.  The analog axis is authoritative.
        trigger_button_code=None,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    output: TextIO | None = None,
    error: TextIO | None = None,
    discovery: Callable[[], tuple[ControllerCandidate, ...]] = discover_controllers,
    observer: Callable[..., MappingVerificationReport] = observe_phase,
    controller_factory: Callable[..., EvdevPrecisionController] = EvdevPrecisionController,
) -> int:
    out = output or sys.stdout
    err = error or sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        config = _config(args)
        calibration = _trigger_calibration(args)
        if args.verification_seconds <= 0.0:
            raise ValueError("verification seconds must be greater than zero")
    except (ValueError, argparse.ArgumentTypeError) as exc:
        print(f"Cannot configure controller precision: {exc}", file=err)
        return 2

    try:
        candidates = discovery()
        if args.diagnose or not any((args.dry_run, args.verify_mapping, args.run)):
            print_diagnostics(candidates, output=out)
            return 0

        selected = _select_from_arguments(args, candidates)
        print(f"Selected: {describe_candidate(selected)}", file=out)
        print(
            "Mapping: LT={} | right stick={}/{} | strength={:.0%} | curve={:.2f}".format(
                _axis_name(config.trigger_axis_code),
                _axis_name(config.right_x_code),
                _axis_name(config.right_y_code),
                config.strength,
                config.exponent,
            ),
            file=out,
        )

        if args.dry_run:
            print(
                "Dry run only: mapping remains unverified; no device was opened, "
                "grabbed, or created.",
                file=out,
            )
            return 0
        if args.verify_mapping:
            return 0 if verify_default_mapping(
                selected,
                config=config,
                seconds=args.verification_seconds,
                output=out,
                observer=observer,
            ) else 3
        if not args.confirm_default_mapping:
            print(
                "Refusing to grab the controller until its mapping is verified. "
                "Run --verify-mapping first, then add --confirm-default-mapping.",
                file=err,
            )
            return 2

        stopped = False

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stopped
            stopped = True

        previous_handlers: dict[int, Any] = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        try:
            controller = controller_factory(
                selected.path,
                expected_identity=selected.identity,
                config=config,
                mapping_verified=True,
                trigger_calibration=calibration,
            )

            def announce_ready() -> None:
                # The launcher consumes the exact first line as its readiness
                # handshake.  It is emitted only by EvdevPrecisionController
                # after evdev/uinput have opened and synchronized successfully.
                print(CONTROLLER_READY_SENTINEL, file=out, flush=True)
                print(
                    "Precision controller active. Hold LT for fine right-stick control.",
                    file=out,
                    flush=True,
                )

            controller.run(
                should_stop=lambda: stopped
                or (args.parent_pid is not None and os.getppid() != args.parent_pid),
                on_ready=announce_ready,
            )
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
        return 0
    except (ControllerBackendError, DroppedEventsError, OSError) as exc:
        print(f"Controller precision stopped safely: {exc}", file=err)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
