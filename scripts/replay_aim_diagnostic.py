#!/usr/bin/env python3
"""Validate and replay one bounded automatic-aim diagnostic session."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiming.controller import TargetTracker  # noqa: E402
from detection.types import Detection  # noqa: E402
from utils.aim_diagnostic import SCHEMA_NAME, SCHEMA_VERSION  # noqa: E402


REPLAY_SCHEMA = "proaim.aim_diagnostic.replay"
REPLAY_SCHEMA_VERSION = 1
MINIMUM_ACTIVATED_FRAMES = 100
MINIMUM_PRIMARY_MEASUREMENT_RATE = 0.75
MINIMUM_TARGET_OUTPUT_RATE = 0.85
MINIMUM_CONTROLLER_TARGET_PUBLICATION_RATE = 0.85
MAXIMUM_REPLAY_DIVERGENCE_RATE = 0.01
MINIMUM_VISIBLE_HEAD_ANCHOR_COVERAGE = 0.70


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read diagnostic JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"diagnostic JSON must be an object: {path}")
    return value


def _relative_artifact(session: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("diagnostic image path must be a string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("diagnostic image path is unsafe")
    path = session.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"diagnostic image is missing or unsafe: {value}")
    return path


def _detection(value: object) -> Detection | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("diagnostic detection must be an object or null")
    try:
        class_id = int(value["class_id"])
        class_name = str(value["class_name"])
        confidence = float(value["confidence"])
        raw_box = value["box"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("diagnostic detection is malformed") from exc
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        raise ValueError("diagnostic detection box must contain four values")
    box = tuple(float(item) for item in raw_box)
    if not all(math.isfinite(item) for item in (*box, confidence)):
        raise ValueError("diagnostic detection contains non-finite values")
    return Detection(class_id, class_name, confidence, box)


def _detection_list(value: object) -> list[Detection]:
    if not isinstance(value, list):
        raise ValueError("diagnostic candidate stream must be a list")
    result = []
    for item in value:
        detection = _detection(item)
        if detection is None:
            raise ValueError("diagnostic candidate stream cannot contain null")
        result.append(detection)
    return result


def _detections_match(first: Detection | None, second: Detection | None) -> bool:
    if first is None or second is None:
        return first is second
    return bool(
        first.class_id == second.class_id
        and first.class_name == second.class_name
        and max(
            abs(left - right)
            for left, right in zip(first.xyxy, second.xyxy)
        )
        <= 1e-3
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Return one strict JSON integer, rejecting bool and lossy coercion."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _signed_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_integer(
    value: object,
    name: str,
    *,
    minimum: int = 0,
) -> int | None:
    if value is None:
        return None
    return _integer(value, name, minimum=minimum)


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    """Summarize finite samples without inventing values for an empty trace."""

    if not values:
        return {
            "samples": 0,
            "mean": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }
    return {
        "samples": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _mapping_summary(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate additive diagnostic mappings while preserving field names."""

    numeric: dict[str, list[float]] = {}
    boolean: dict[str, list[bool]] = {}
    categorical: dict[str, dict[str, int]] = {}
    for sample in samples:
        for raw_name, value in sample.items():
            name = str(raw_name)
            if isinstance(value, bool):
                boolean.setdefault(name, []).append(value)
            elif isinstance(value, (int, float)):
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError(f"diagnostic field {name} must be finite")
                numeric.setdefault(name, []).append(parsed)
            elif isinstance(value, str):
                counts = categorical.setdefault(name, {})
                counts[value] = counts.get(value, 0) + 1
            elif value is not None:
                # Future structured fields remain forward-compatible; the
                # generic scalar aggregate simply does not interpret them.
                continue
    return {
        "samples": len(samples),
        "numeric": {
            name: _numeric_summary(values)
            for name, values in sorted(numeric.items())
        },
        "boolean": {
            name: {
                "samples": len(values),
                "true": sum(values),
                "true_rate": _rate(sum(values), len(values)),
            }
            for name, values in sorted(boolean.items())
        },
        "categorical": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(categorical.items())
        },
    }


def _numeric_mapping(value: object, name: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result: dict[str, int | float] = {}
    for raw_field, raw_value in value.items():
        field = str(raw_field)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{name}.{field} must be numeric")
        if not math.isfinite(float(raw_value)):
            raise ValueError(f"{name}.{field} must be finite")
        result[field] = raw_value
    return result


def _sum_numeric_mappings(
    samples: Sequence[Mapping[str, int | float]],
) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for sample in samples:
        for name, value in sample.items():
            totals[name] = totals.get(name, 0) + value
    return dict(sorted(totals.items()))


def _telemetry_rates(totals: Mapping[str, int | float]) -> dict[str, float]:
    """Derive duty/mean values from cumulative controller telemetry."""

    result: dict[str, float] = {}

    def ratio(output_name: str, numerator: str, denominator: str) -> None:
        raw_numerator = totals.get(numerator)
        raw_denominator = totals.get(denominator)
        if raw_numerator is None or raw_denominator is None:
            return
        denominator_value = float(raw_denominator)
        if denominator_value <= 0.0:
            return
        result[output_name] = float(raw_numerator) / denominator_value

    for field in (
        "active_input_ticks",
        "button_pressed_ticks",
        "target_present_ticks",
        "fresh_target_ticks",
        "authorized_ticks",
        "movement_commands",
    ):
        ratio(f"{field}_per_output_tick", field, "output_ticks")

    for field in (
        "control_error_x",
        "control_error_y",
        "control_error_abs_x",
        "control_error_abs_y",
        "pursuit_x",
        "pursuit_y",
        "pursuit_abs_x",
        "pursuit_abs_y",
        "target_velocity_abs_x_pixels_per_second",
        "target_velocity_abs_y_pixels_per_second",
        "velocity_feedforward_confidence_x",
        "velocity_feedforward_confidence_y",
        "pursuit_reserve_abs_x_counts_per_second",
        "pursuit_reserve_abs_y_counts_per_second",
        "motion_corroboration_confidence",
        "body_derived_motion_confidence_x",
        "body_derived_motion_confidence_y",
    ):
        ratio(f"mean_{field}", field, "control_samples")

    for field in (
        "pursuit_reserve_active_x_samples",
        "pursuit_reserve_active_y_samples",
        "saturated_x_samples",
        "saturated_y_samples",
    ):
        ratio(f"{field}_rate", field, "control_samples")
    return dict(sorted(result.items()))


def _controller_target_was_published(record: Mapping[str, Any]) -> bool:
    """Return whether this frame published any target to the controller.

    ``control_source`` describes the target supplied to the controller; it is
    not proof that the controller's independent button, freshness, and safety
    gates authorized physical motion.  In particular, ``direct-hold`` is a
    body-fallback publication while a new direct-head result is pending.
    """

    source = record.get("control_source")
    return bool(
        isinstance(source, str)
        and source.strip()
        and source.strip().lower() != "none"
    )


def _visible_head_anchor(record: Mapping[str, Any]) -> bool:
    """Return visible direct-head lease coverage for current and v1 traces."""

    if "visible_head_sample" in record:
        sample = record["visible_head_sample"]
        if isinstance(sample, bool):
            return sample
        return sample is not None

    # Schema-v1 traces did not serialize the held visible sample.  Their status
    # text did distinguish both valid forms of the same identity-bound lease.
    status = record.get("aim_status")
    if not isinstance(status, str):
        return False
    normalized = " ".join(status.strip().lower().split())
    return normalized.startswith(("aim anchored:", "aim bridge visible:"))


def replay_session(session_dir: str | Path) -> dict[str, Any]:
    session = Path(session_dir).expanduser().resolve()
    if not session.is_dir() or session.is_symlink():
        raise ValueError(f"diagnostic session is missing or unsafe: {session}")
    manifest = _read_json_object(session / "manifest.json")
    if manifest.get("schema") != SCHEMA_NAME:
        raise ValueError("diagnostic manifest schema is not recognized")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("diagnostic manifest version is not supported")
    metadata = manifest.get("metadata")
    statistics = manifest.get("statistics")
    if not isinstance(metadata, Mapping) or not isinstance(statistics, Mapping):
        raise ValueError("diagnostic manifest metadata/statistics are malformed")
    records_name = manifest.get("records_path")
    if not isinstance(records_name, str) or PurePosixPath(records_name).name != records_name:
        raise ValueError("diagnostic records path is unsafe")
    records_path = session / records_name
    try:
        lines = records_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read diagnostic records: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"diagnostic record {line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"diagnostic record {line_number} is not an object")
        if "image_path" in record:
            image = _relative_artifact(session, record["image_path"])
            digest = sha256(image.read_bytes()).hexdigest()
            if digest != record.get("image_sha256"):
                raise ValueError(
                    f"diagnostic frame hash mismatch at record {line_number}"
                )
            if image.stat().st_size != record.get("image_size_bytes"):
                raise ValueError(
                    f"diagnostic frame size mismatch at record {line_number}"
                )
        records.append(record)

    written = int(statistics.get("written", -1))
    pending_overwrites = int(statistics.get("pending_overwrites", -1))
    trace_complete = bool(
        manifest.get("complete") is True
        and written == len(records)
        and pending_overwrites == 0
    )
    aim_label = str(metadata.get("aim_label") or "").strip()
    if not aim_label:
        raise ValueError("diagnostic metadata has no aim label")
    head_ratio = float(metadata.get("head_ratio", 0.12))
    tracking_mode = str(metadata.get("tracking_mode") or "")
    lost_grace_frames = int(metadata.get("lost_grace_frames", 3))
    tracker = TargetTracker(
        label=aim_label,
        head_ratio=head_ratio,
        lost_grace_frames=lost_grace_frames,
    )

    activated_frames = 0
    primary_measurements = 0
    target_outputs = 0
    controller_target_publications = 0
    new_direct_head_samples = 0
    visible_head_anchor_frames = 0
    raw_activation_known_frames = 0
    raw_activation_pressed_frames = 0
    raw_pressed_authorized_frames = 0
    continuous_hold_expired_frames = 0
    continuous_hold_expired_events = 0
    replay_divergences = 0
    prediction_divergences = 0
    previous_activation = False
    first_timestamp: int | None = None
    previous_timestamp: int | None = None
    previous_measurement_timestamp_in_epoch: int | None = None
    previous_new_direct_timestamp_in_epoch: int | None = None
    previous_continuous_hold_expired = False
    measurement_gaps: list[float] = []
    new_direct_head_gaps: list[float] = []
    visible_head_phase_metadata_frames = 0
    visible_head_phase_partial_frames = 0
    phase_advanced_frames = 0
    phase_hops: list[float] = []
    makcu_control_frames = 0
    makcu_ledger_snapshot_frames = 0
    makcu_calibrated_output_frames = 0
    makcu_telemetry_frames = 0
    makcu_epochs: dict[int, dict[str, Any]] = {}
    makcu_commands: dict[tuple[int, int], dict[str, int]] = {}
    makcu_outputs: dict[tuple[int, int], dict[str, Any]] = {}
    makcu_telemetry_latest_by_epoch: dict[int, dict[str, int | float]] = {}

    for index, record in enumerate(records, 1):
        timestamp = record.get("source_timestamp_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError(f"record {index} has an invalid source timestamp")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("diagnostic source timestamps must strictly increase")
        if first_timestamp is None:
            first_timestamp = timestamp
        previous_timestamp = timestamp
        shape = record.get("frame_shape")
        if (
            not isinstance(shape, list)
            or len(shape) < 2
            or int(shape[0]) <= 0
            or int(shape[1]) <= 0
        ):
            raise ValueError(f"record {index} has an invalid frame shape")
        frame_shape = tuple(int(value) for value in shape)
        activation = bool(record.get("activation_pressed"))
        raw_activation_known = record.get("raw_activation_known") is True
        raw_activation_pressed = bool(
            raw_activation_known
            and record.get("raw_activation_pressed") is True
        )
        continuous_hold_expired = bool(
            record.get("activation_denial_reason") == "continuous-hold-expired"
            or (
                raw_activation_pressed
                and record.get("activation_requires_release") is True
                and not activation
            )
        )
        if raw_activation_known:
            raw_activation_known_frames += 1
        if raw_activation_pressed:
            raw_activation_pressed_frames += 1
            if activation:
                raw_pressed_authorized_frames += 1
        if continuous_hold_expired:
            continuous_hold_expired_frames += 1
            if not previous_continuous_hold_expired:
                continuous_hold_expired_events += 1
        previous_continuous_hold_expired = continuous_hold_expired

        visible_sample = record.get("visible_head_sample")
        if isinstance(visible_sample, Mapping):
            has_phase_flag = "phase_advanced" in visible_sample
            has_phase_hops = "phase_hops" in visible_sample
            parsed_phase_advanced: bool | None = None
            parsed_phase_hops: int | None = None
            if has_phase_flag:
                raw_phase_advanced = visible_sample["phase_advanced"]
                if not isinstance(raw_phase_advanced, bool):
                    raise ValueError(
                        f"record {index} visible phase_advanced must be bool"
                    )
                parsed_phase_advanced = raw_phase_advanced
            if has_phase_hops:
                parsed_phase_hops = _integer(
                    visible_sample["phase_hops"],
                    f"record {index} visible phase_hops",
                )
            if activation and has_phase_flag and has_phase_hops:
                assert parsed_phase_advanced is not None
                assert parsed_phase_hops is not None
                visible_head_phase_metadata_frames += 1
                phase_advanced_frames += int(parsed_phase_advanced)
                phase_hops.append(float(parsed_phase_hops))
            elif activation and (has_phase_flag or has_phase_hops):
                visible_head_phase_partial_frames += 1

        makcu_control = record.get("makcu_control")
        if makcu_control is not None:
            if not isinstance(makcu_control, Mapping):
                raise ValueError(f"record {index} makcu_control must be an object")
            makcu_control_frames += 1
            ledger_fields = (
                "connection_epoch",
                "successful_commands",
                "recent_commands",
            )
            ledger_fields_present = tuple(
                name in makcu_control for name in ledger_fields
            )
            if any(ledger_fields_present) and not all(ledger_fields_present):
                raise ValueError(
                    f"record {index} makcu_control ledger is incomplete"
                )
            needs_epoch = bool(
                all(ledger_fields_present)
                or makcu_control.get("calibrated_output") is not None
                or makcu_control.get("cumulative_telemetry") is not None
            )
            epoch: int | None = None
            if needs_epoch:
                epoch = _integer(
                    makcu_control.get("connection_epoch"),
                    f"record {index} MAKCU connection epoch",
                )
            if "captured_ns" in makcu_control:
                _integer(
                    makcu_control["captured_ns"],
                    f"record {index} MAKCU capture timestamp",
                )

            if all(ledger_fields_present):
                assert epoch is not None
                makcu_ledger_snapshot_frames += 1
                successful_commands = _integer(
                    makcu_control["successful_commands"],
                    f"record {index} MAKCU successful command count",
                )
                dropped_commands = _integer(
                    makcu_control.get("dropped_commands", 0),
                    f"record {index} MAKCU dropped command count",
                )
                recent_commands = makcu_control["recent_commands"]
                if not isinstance(recent_commands, list):
                    raise ValueError(
                        f"record {index} MAKCU recent_commands must be a list"
                    )
                epoch_state = makcu_epochs.setdefault(
                    epoch,
                    {
                        "snapshot_frames": 0,
                        "successful_commands": 0,
                        "dropped_commands": 0,
                        "first_emitted_ns": None,
                        "last_emitted_ns": None,
                        "emitted_x": 0,
                        "emitted_y": 0,
                        "emitted_abs_x": 0,
                        "emitted_abs_y": 0,
                    },
                )
                if successful_commands < epoch_state["successful_commands"]:
                    raise ValueError(
                        f"record {index} MAKCU command count moved backwards"
                    )
                epoch_state["snapshot_frames"] += 1
                epoch_state["successful_commands"] = successful_commands
                epoch_state["dropped_commands"] = max(
                    epoch_state["dropped_commands"],
                    dropped_commands,
                )
                first_emitted_ns = _optional_integer(
                    makcu_control.get("first_emitted_ns"),
                    f"record {index} MAKCU first command timestamp",
                )
                last_emitted_ns = _optional_integer(
                    makcu_control.get("last_emitted_ns"),
                    f"record {index} MAKCU last command timestamp",
                )
                if (
                    first_emitted_ns is not None
                    and last_emitted_ns is not None
                    and last_emitted_ns < first_emitted_ns
                ):
                    raise ValueError(
                        f"record {index} MAKCU command timestamps are reversed"
                    )
                if first_emitted_ns is not None:
                    epoch_state["first_emitted_ns"] = first_emitted_ns
                if last_emitted_ns is not None:
                    epoch_state["last_emitted_ns"] = last_emitted_ns
                for field in ("emitted_x", "emitted_y"):
                    epoch_state[field] = _signed_integer(
                        makcu_control.get(field, 0),
                        f"record {index} MAKCU {field}",
                    )
                for field in ("emitted_abs_x", "emitted_abs_y"):
                    epoch_state[field] = _integer(
                        makcu_control.get(field, 0),
                        f"record {index} MAKCU {field}",
                    )

                previous_sequence = 0
                for command_index, raw_command in enumerate(recent_commands, 1):
                    if not isinstance(raw_command, Mapping):
                        raise ValueError(
                            f"record {index} MAKCU command {command_index} "
                            "must be an object"
                        )
                    sequence = _integer(
                        raw_command.get("sequence"),
                        f"record {index} MAKCU command sequence",
                        minimum=1,
                    )
                    if sequence <= previous_sequence:
                        raise ValueError(
                            f"record {index} MAKCU command sequence is not "
                            "strictly increasing"
                        )
                    if sequence > successful_commands:
                        raise ValueError(
                            f"record {index} MAKCU command sequence exceeds "
                            "the cumulative count"
                        )
                    previous_sequence = sequence
                    command = {
                        "sequence": sequence,
                        "timestamp_ns": _integer(
                            raw_command.get("timestamp_ns"),
                            f"record {index} MAKCU command timestamp",
                        ),
                        "delta_x_counts": _signed_integer(
                            raw_command.get("delta_x_counts"),
                            f"record {index} MAKCU command X delta",
                        ),
                        "delta_y_counts": _signed_integer(
                            raw_command.get("delta_y_counts"),
                            f"record {index} MAKCU command Y delta",
                        ),
                    }
                    if (
                        command["delta_x_counts"] == 0
                        and command["delta_y_counts"] == 0
                    ):
                        raise ValueError(
                            f"record {index} MAKCU command cannot be zero"
                        )
                    key = (epoch, sequence)
                    prior_command = makcu_commands.get(key)
                    if prior_command is not None and prior_command != command:
                        raise ValueError(
                            f"record {index} MAKCU command identity conflicts "
                            "with an earlier snapshot"
                        )
                    makcu_commands[key] = command

            calibrated_output = makcu_control.get("calibrated_output")
            if calibrated_output is not None:
                assert epoch is not None
                if not isinstance(calibrated_output, Mapping):
                    raise ValueError(
                        f"record {index} calibrated output must be an object"
                    )
                output_timestamp = _integer(
                    calibrated_output.get("timestamp_ns"),
                    f"record {index} calibrated output timestamp",
                )
                output = dict(calibrated_output)
                output_key = (epoch, output_timestamp)
                prior_output = makcu_outputs.get(output_key)
                if prior_output is not None and prior_output != output:
                    raise ValueError(
                        f"record {index} calibrated output identity conflicts"
                    )
                makcu_outputs[output_key] = output
                makcu_calibrated_output_frames += 1

            cumulative_telemetry = makcu_control.get("cumulative_telemetry")
            if cumulative_telemetry is not None:
                assert epoch is not None
                makcu_telemetry_latest_by_epoch[epoch] = _numeric_mapping(
                    cumulative_telemetry,
                    f"record {index} cumulative MAKCU telemetry",
                )
                makcu_telemetry_frames += 1

        if activation != previous_activation:
            tracker.reset()
        previous_activation = activation
        # v2 records distinguish the temporal filter's raw global readiness
        # from the effective aim-only decision.  A target proven spatially
        # distinct from every uncertain self identity is safe even while the
        # preview-oriented temporal filter remains unresolved.  Retain v1
        # compatibility when the effective field is absent.
        self_safe = bool(
            record.get(
                "aim_self_exclusion_safe",
                record.get("self_exclusion_ready"),
            )
        )
        grace_safe = not bool(record.get("hard_guard_revoked_prediction_grace"))
        if not self_safe or not grace_safe:
            tracker.reset()
            replayed = None
        else:
            replayed = tracker.update(
                _detection_list(record.get("aim_candidates")),
                frame_shape,
                measurement_ns=timestamp,
                continuation_detections=_detection_list(
                    record.get("continuation_candidates")
                ),
                continuation_allowed=activation,
            )
        recorded = _detection(record.get("selected_target"))
        if not _detections_match(replayed, recorded):
            replay_divergences += 1
        recorded_prediction = bool(record.get("selected_is_prediction"))
        if tracker.output_is_prediction != recorded_prediction:
            prediction_divergences += 1
        if not activation:
            # An activation edge starts a new authorization/identity epoch.
            # Time spent released must never inflate an in-hold detector gap.
            previous_measurement_timestamp_in_epoch = None
            previous_new_direct_timestamp_in_epoch = None
            continue

        activated_frames += 1
        if _detection(record.get("accepted_measurement")) is not None:
            primary_measurements += 1
            if previous_measurement_timestamp_in_epoch is not None:
                measurement_gaps.append(
                    (timestamp - previous_measurement_timestamp_in_epoch) / 1e6
                )
            previous_measurement_timestamp_in_epoch = timestamp
        if recorded is not None:
            target_outputs += 1
        if _controller_target_was_published(record):
            controller_target_publications += 1
        if record.get("direct_head_sample") is not None:
            new_direct_head_samples += 1
            if previous_new_direct_timestamp_in_epoch is not None:
                new_direct_head_gaps.append(
                    (timestamp - previous_new_direct_timestamp_in_epoch) / 1e6
                )
            previous_new_direct_timestamp_in_epoch = timestamp
        if _visible_head_anchor(record):
            visible_head_anchor_frames += 1

    primary_rate = _rate(primary_measurements, activated_frames)
    target_rate = _rate(target_outputs, activated_frames)
    controller_target_publication_rate = _rate(
        controller_target_publications,
        activated_frames,
    )
    new_direct_head_sample_rate = _rate(
        new_direct_head_samples,
        activated_frames,
    )
    visible_head_anchor_coverage = _rate(
        visible_head_anchor_frames,
        activated_frames,
    )
    filtered_activation_rate_while_raw_pressed = (
        _rate(raw_pressed_authorized_frames, raw_activation_pressed_frames)
        if raw_activation_pressed_frames
        else None
    )
    divergence_rate = _rate(
        replay_divergences + prediction_divergences,
        max(1, len(records) * 2),
    )

    phase_hops_summary = _numeric_summary(phase_hops)
    visible_head_phase_metadata_coverage = _rate(
        visible_head_phase_metadata_frames,
        visible_head_anchor_frames,
    )
    phase_advanced_rate = _rate(
        phase_advanced_frames,
        visible_head_phase_metadata_frames,
    )

    makcu_commands_by_epoch: dict[int, list[dict[str, int]]] = {}
    for (epoch, _sequence), command in sorted(makcu_commands.items()):
        makcu_commands_by_epoch.setdefault(epoch, []).append(command)
    makcu_connection_epochs: list[dict[str, Any]] = []
    makcu_successful_commands = 0
    makcu_internal_dropped_commands = 0
    makcu_command_gaps_ms: list[float] = []
    makcu_command_intervals = 0
    makcu_command_span_ns = 0
    makcu_expected_command_intervals = 0
    makcu_cumulative_command_intervals = 0
    makcu_cumulative_command_span_ns = 0
    makcu_cumulative_first_timestamps: list[int] = []
    makcu_cumulative_last_timestamps: list[int] = []
    for epoch, state in sorted(makcu_epochs.items()):
        commands = makcu_commands_by_epoch.get(epoch, [])
        for earlier, later in zip(commands, commands[1:]):
            if later["timestamp_ns"] < earlier["timestamp_ns"]:
                raise ValueError(
                    f"MAKCU command timestamps move backwards in epoch {epoch}"
                )
            # A missing sequence means the true adjacent-command interval is
            # unknown. Coverage reports the gap; timing statistics must not
            # misclassify the wider visible span as one physical interval.
            if later["sequence"] == earlier["sequence"] + 1:
                interval_ns = (
                    later["timestamp_ns"] - earlier["timestamp_ns"]
                )
                makcu_command_gaps_ms.append(interval_ns / 1e6)
                makcu_command_intervals += 1
                makcu_command_span_ns += interval_ns
        successful_commands = int(state["successful_commands"])
        ledger_commands = len(commands)
        command_coverage = (
            _rate(ledger_commands, successful_commands)
            if successful_commands
            else 1.0
        )
        makcu_successful_commands += successful_commands
        makcu_expected_command_intervals += max(successful_commands - 1, 0)
        makcu_internal_dropped_commands += int(state["dropped_commands"])
        first_emitted_ns = state["first_emitted_ns"]
        last_emitted_ns = state["last_emitted_ns"]
        if isinstance(first_emitted_ns, int):
            makcu_cumulative_first_timestamps.append(first_emitted_ns)
        if isinstance(last_emitted_ns, int):
            makcu_cumulative_last_timestamps.append(last_emitted_ns)
        if (
            successful_commands > 1
            and isinstance(first_emitted_ns, int)
            and isinstance(last_emitted_ns, int)
            and last_emitted_ns > first_emitted_ns
        ):
            makcu_cumulative_command_intervals += successful_commands - 1
            makcu_cumulative_command_span_ns += (
                last_emitted_ns - first_emitted_ns
            )
        makcu_connection_epochs.append(
            {
                "connection_epoch": epoch,
                "snapshot_frames": int(state["snapshot_frames"]),
                "successful_commands": successful_commands,
                "ledger_commands": ledger_commands,
                "command_trace_coverage": command_coverage,
                "unobserved_commands": successful_commands - ledger_commands,
                "dropped_commands": int(state["dropped_commands"]),
                "first_emitted_ns": state["first_emitted_ns"],
                "last_emitted_ns": state["last_emitted_ns"],
                "emitted_x": int(state["emitted_x"]),
                "emitted_y": int(state["emitted_y"]),
                "emitted_abs_x": int(state["emitted_abs_x"]),
                "emitted_abs_y": int(state["emitted_abs_y"]),
            }
        )
    makcu_ledger_commands = len(makcu_commands)
    makcu_command_trace_coverage = (
        _rate(makcu_ledger_commands, makcu_successful_commands)
        if makcu_successful_commands
        else (1.0 if makcu_ledger_snapshot_frames else None)
    )
    makcu_unobserved_commands = (
        makcu_successful_commands - makcu_ledger_commands
    )
    makcu_command_timing_trace_coverage = (
        _rate(makcu_command_intervals, makcu_expected_command_intervals)
        if makcu_expected_command_intervals
        else (1.0 if makcu_ledger_snapshot_frames else None)
    )
    ordered_makcu_commands = sorted(
        makcu_commands.values(),
        key=lambda command: command["timestamp_ns"],
    )
    makcu_ledger_first_command_ns = (
        ordered_makcu_commands[0]["timestamp_ns"]
        if ordered_makcu_commands
        else None
    )
    makcu_ledger_last_command_ns = (
        ordered_makcu_commands[-1]["timestamp_ns"]
        if ordered_makcu_commands
        else None
    )
    makcu_first_command_ns = (
        min(makcu_cumulative_first_timestamps)
        if makcu_cumulative_first_timestamps
        else makcu_ledger_first_command_ns
    )
    makcu_last_command_ns = (
        max(makcu_cumulative_last_timestamps)
        if makcu_cumulative_last_timestamps
        else makcu_ledger_last_command_ns
    )
    makcu_ledger_emitted_x = sum(
        command["delta_x_counts"] for command in makcu_commands.values()
    )
    makcu_ledger_emitted_y = sum(
        command["delta_y_counts"] for command in makcu_commands.values()
    )
    makcu_ledger_emitted_abs_x = sum(
        abs(command["delta_x_counts"]) for command in makcu_commands.values()
    )
    makcu_ledger_emitted_abs_y = sum(
        abs(command["delta_y_counts"]) for command in makcu_commands.values()
    )
    makcu_command_rate_hz = (
        makcu_command_intervals / (makcu_command_span_ns / 1e9)
        if makcu_command_span_ns > 0
        else None
    )
    makcu_cumulative_command_rate_hz = (
        makcu_cumulative_command_intervals
        / (makcu_cumulative_command_span_ns / 1e9)
        if makcu_cumulative_command_span_ns > 0
        else None
    )
    trace_duration_seconds = (
        (previous_timestamp - first_timestamp) / 1e9
        if previous_timestamp is not None
        and first_timestamp is not None
        and previous_timestamp > first_timestamp
        else 0.0
    )
    makcu_commands_per_trace_second = (
        makcu_ledger_commands / trace_duration_seconds
        if trace_duration_seconds > 0.0
        else None
    )
    makcu_command_gap_summary = _numeric_summary(makcu_command_gaps_ms)
    makcu_calibrated_output_summary = _mapping_summary(
        list(makcu_outputs.values())
    )
    makcu_telemetry_by_epoch = [
        {
            "connection_epoch": epoch,
            "cumulative_telemetry": telemetry,
        }
        for epoch, telemetry in sorted(makcu_telemetry_latest_by_epoch.items())
    ]
    makcu_cumulative_telemetry_totals = _sum_numeric_mappings(
        list(makcu_telemetry_latest_by_epoch.values())
    )
    makcu_cumulative_telemetry_latest = (
        makcu_telemetry_latest_by_epoch[
            max(makcu_telemetry_latest_by_epoch)
        ]
        if makcu_telemetry_latest_by_epoch
        else None
    )
    makcu_telemetry_summary = _telemetry_rates(
        makcu_cumulative_telemetry_totals
    )

    failed_gates: list[str] = []
    if not trace_complete:
        failed_gates.append("trace_complete")
    if primary_rate < MINIMUM_PRIMARY_MEASUREMENT_RATE:
        failed_gates.append("primary_measurement_rate")
    if target_rate < MINIMUM_TARGET_OUTPUT_RATE:
        failed_gates.append("target_output_rate")
    if (
        controller_target_publication_rate
        < MINIMUM_CONTROLLER_TARGET_PUBLICATION_RATE
    ):
        failed_gates.append("controller_target_publication_rate")
    if divergence_rate > MAXIMUM_REPLAY_DIVERGENCE_RATE:
        failed_gates.append("replay_divergence_rate")
    if (
        tracking_mode == "direct-head"
        and visible_head_anchor_coverage < MINIMUM_VISIBLE_HEAD_ANCHOR_COVERAGE
    ):
        failed_gates.append("visible_head_anchor_coverage")

    if not trace_complete:
        status = "invalid-trace"
    elif activated_frames < MINIMUM_ACTIVATED_FRAMES:
        status = "insufficient-data"
    elif failed_gates:
        status = "fail"
    else:
        status = "pass"
    recommendations = []
    if "primary_measurement_rate" in failed_gates:
        recommendations.append("evaluate-primary-detector-recall")
    if "target_output_rate" in failed_gates or "replay_divergence_rate" in failed_gates:
        recommendations.append("inspect-target-association")
    if "visible_head_anchor_coverage" in failed_gates:
        recommendations.append("replace-or-train-direct-head-model")
    if "controller_target_publication_rate" in failed_gates:
        recommendations.append("inspect-controller-target-publication")
    if continuous_hold_expired_events:
        recommendations.append("release-and-repress-after-continuous-hold-limit")

    primary_gap_p95_ms = _percentile(measurement_gaps, 0.95)
    primary_gap_max_ms = max(measurement_gaps, default=None)
    new_direct_head_gap_p95_ms = _percentile(new_direct_head_gaps, 0.95)
    new_direct_head_gap_max_ms = max(new_direct_head_gaps, default=None)

    return {
        "schema": REPLAY_SCHEMA,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "session": str(session),
        "status": status,
        "trace_complete": trace_complete,
        "records": len(records),
        "activated_frames": activated_frames,
        "primary_measurements": primary_measurements,
        "primary_measurement_rate": primary_rate,
        "target_outputs": target_outputs,
        "target_output_rate": target_rate,
        "controller_target_publications": controller_target_publications,
        "controller_target_publication_rate": controller_target_publication_rate,
        "new_direct_head_samples": new_direct_head_samples,
        "new_direct_head_sample_rate": new_direct_head_sample_rate,
        "visible_head_anchor_frames": visible_head_anchor_frames,
        "visible_head_anchor_coverage": visible_head_anchor_coverage,
        "visible_head_phase_metadata_frames": (
            visible_head_phase_metadata_frames
        ),
        "visible_head_phase_partial_frames": visible_head_phase_partial_frames,
        "visible_head_phase_metadata_coverage": (
            visible_head_phase_metadata_coverage
        ),
        "phase_advanced_frames": phase_advanced_frames,
        "phase_advanced_rate": phase_advanced_rate,
        "phase_hops_total": int(sum(phase_hops)),
        "phase_hops_mean": phase_hops_summary["mean"],
        "phase_hops_p50": phase_hops_summary["p50"],
        "phase_hops_p95": phase_hops_summary["p95"],
        "phase_hops_max": phase_hops_summary["max"],
        "trace_duration_seconds": trace_duration_seconds,
        "makcu_control_frames": makcu_control_frames,
        "makcu_control_trace_coverage": _rate(
            makcu_control_frames,
            len(records),
        ),
        "makcu_ledger_snapshot_frames": makcu_ledger_snapshot_frames,
        "makcu_ledger_trace_coverage": _rate(
            makcu_ledger_snapshot_frames,
            len(records),
        ),
        "makcu_connection_epochs": makcu_connection_epochs,
        "makcu_successful_commands": makcu_successful_commands,
        "makcu_ledger_commands": makcu_ledger_commands,
        "makcu_command_trace_coverage": makcu_command_trace_coverage,
        "makcu_command_timing_trace_coverage": (
            makcu_command_timing_trace_coverage
        ),
        "makcu_unobserved_commands": makcu_unobserved_commands,
        "makcu_internal_dropped_commands": (
            makcu_internal_dropped_commands
        ),
        "makcu_ledger_emitted_x": makcu_ledger_emitted_x,
        "makcu_ledger_emitted_y": makcu_ledger_emitted_y,
        "makcu_ledger_emitted_abs_x": makcu_ledger_emitted_abs_x,
        "makcu_ledger_emitted_abs_y": makcu_ledger_emitted_abs_y,
        "makcu_first_command_ns": makcu_first_command_ns,
        "makcu_last_command_ns": makcu_last_command_ns,
        "makcu_ledger_first_command_ns": makcu_ledger_first_command_ns,
        "makcu_ledger_last_command_ns": makcu_ledger_last_command_ns,
        "makcu_command_rate_hz": makcu_command_rate_hz,
        "makcu_cumulative_command_rate_hz": (
            makcu_cumulative_command_rate_hz
        ),
        "makcu_commands_per_trace_second": makcu_commands_per_trace_second,
        "makcu_command_gap_mean_ms": makcu_command_gap_summary["mean"],
        "makcu_command_gap_p50_ms": makcu_command_gap_summary["p50"],
        "makcu_command_gap_p95_ms": makcu_command_gap_summary["p95"],
        "makcu_command_gap_max_ms": makcu_command_gap_summary["max"],
        "makcu_calibrated_output_frames": makcu_calibrated_output_frames,
        "makcu_calibrated_output_trace_coverage": _rate(
            makcu_calibrated_output_frames,
            len(records),
        ),
        "makcu_calibrated_output_unique_samples": len(makcu_outputs),
        "makcu_calibrated_output_summary": (
            makcu_calibrated_output_summary
        ),
        "makcu_telemetry_frames": makcu_telemetry_frames,
        "makcu_telemetry_trace_coverage": _rate(
            makcu_telemetry_frames,
            len(records),
        ),
        "makcu_cumulative_telemetry_latest": (
            makcu_cumulative_telemetry_latest
        ),
        "makcu_cumulative_telemetry_totals": (
            makcu_cumulative_telemetry_totals
        ),
        "makcu_cumulative_telemetry_by_epoch": makcu_telemetry_by_epoch,
        "makcu_telemetry_summary": makcu_telemetry_summary,
        "raw_activation_known_frames": raw_activation_known_frames,
        "raw_activation_pressed_frames": raw_activation_pressed_frames,
        "raw_pressed_authorized_frames": raw_pressed_authorized_frames,
        "filtered_activation_rate_while_raw_pressed": (
            filtered_activation_rate_while_raw_pressed
        ),
        "continuous_hold_expired_frames": continuous_hold_expired_frames,
        "continuous_hold_expired_events": continuous_hold_expired_events,
        # Compatibility aliases only.  Neither legacy name is an authorization
        # measurement, and neither legacy rate is used as a quality gate.
        "control_available_frames": controller_target_publications,
        "control_availability_rate": controller_target_publication_rate,
        "direct_head_samples": new_direct_head_samples,
        "direct_head_rate": new_direct_head_sample_rate,
        "replay_divergences": replay_divergences,
        "prediction_divergences": prediction_divergences,
        "replay_divergence_rate": divergence_rate,
        "primary_gap_p95_ms": primary_gap_p95_ms,
        "primary_gap_max_ms": primary_gap_max_ms,
        "new_direct_head_gap_p95_ms": new_direct_head_gap_p95_ms,
        "new_direct_head_gap_max_ms": new_direct_head_gap_max_ms,
        "direct_head_gap_p95_ms": new_direct_head_gap_p95_ms,
        "direct_head_gap_max_ms": new_direct_head_gap_max_ms,
        "legacy_metric_aliases": {
            "control_available_frames": "controller_target_publications",
            "control_availability_rate": "controller_target_publication_rate",
            "direct_head_samples": "new_direct_head_samples",
            "direct_head_rate": "new_direct_head_sample_rate",
            "direct_head_gap_p95_ms": "new_direct_head_gap_p95_ms",
            "direct_head_gap_max_ms": "new_direct_head_gap_max_ms",
        },
        "non_gating_metrics": [
            "new_direct_head_sample_rate",
            "control_availability_rate",
            "direct_head_rate",
            "visible_head_phase_metadata_coverage",
            "phase_advanced_rate",
            "makcu_control_trace_coverage",
            "makcu_command_trace_coverage",
            "makcu_command_timing_trace_coverage",
        ],
        "failed_gates": failed_gates,
        "recommendations": recommendations,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = replay_session(args.session)
    encoded = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 1 if report["status"] in {"fail", "invalid-trace"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
