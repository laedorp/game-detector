"""Model-agnostic latest-only head-localization worker and ONNX sessions.

Model-specific code prepares a bounded payload (normally an owned crop or
input tensor) and supplies a callable that returns an absolute source-frame
head observation.  This module owns only scheduling, identity/freshness
metadata, lifecycle, and strict ONNX Runtime boundaries.  It never derives an
aim point from a player box and never reprojects an old result through a newer,
potentially jittery box.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Integral
from pathlib import Path
from threading import Condition, Thread
from time import monotonic, monotonic_ns
from typing import Any, TypeAlias

import numpy as np

from .base import DependencyError, DeviceNotAvailableError, ModelError


Box: TypeAlias = tuple[float, float, float, float]
Point: TypeAlias = tuple[float, float]
CPU_PROVIDER = "CPUExecutionProvider"
MIGRAPHX_PROVIDER = "MIGraphXExecutionProvider"


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_integer(value: int, name: str) -> int:
    result = _non_negative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _box(value: Sequence[float], name: str = "selected_player_box") -> Box:
    if len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    result = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite coordinates")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{name} must have positive width and height")
    return result


@dataclass(frozen=True, slots=True)
class HeadObservation:
    """One model-observed absolute head point in the source frame.

    ``head_box`` is optional worker metadata for consumers which can replay
    the exact localization through newer captured pixels.  Older/localizer-
    agnostic callers remain point-only; an absent box grants no extra motion
    authority.
    """

    point: Point
    confidence: float
    evidence: str
    head_box: Box | None = None

    def __post_init__(self) -> None:
        point = tuple(float(value) for value in self.point)
        if len(point) != 2 or not all(isfinite(value) for value in point):
            raise ValueError("head point must contain two finite coordinates")
        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("head confidence must be finite and between 0 and 1")
        evidence = str(self.evidence).strip()
        if not evidence:
            raise ValueError("head evidence must be a non-empty string")
        head_box = self.head_box
        if head_box is not None:
            head_box = _box(head_box, "head_box")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "head_box", head_box)


class HeadLocalizationReason(str, Enum):
    """Finite, non-spatial disposition for one completed localization job."""

    LOCALIZED = "localized"
    NO_DECODED_HEAD_CANDIDATE = "no_decoded_head_candidate"
    NO_PLAUSIBLE_HEAD = "no_plausible_head"
    MULTIPLE_PLAUSIBLE_HEADS = "multiple_plausible_heads"
    NO_MATCHING_SECONDARY_PLAYER = "no_matching_secondary_player"
    MULTIPLE_MATCHING_SECONDARY_PLAYERS = "multiple_matching_secondary_players"
    HEAD_UNSUPPORTED_BY_MATCHED_PLAYER = "head_unsupported_by_matched_player"
    UNSPECIFIED_NO_HEAD = "unspecified_no_head"


@dataclass(frozen=True, slots=True)
class HeadLocalizationOutcome:
    """One callable result whose disposition travels with its observation."""

    reason: HeadLocalizationReason
    observation: HeadObservation | None

    def __post_init__(self) -> None:
        try:
            reason = HeadLocalizationReason(self.reason)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown head-localization reason") from exc
        observation = self.observation
        if observation is not None and not isinstance(observation, HeadObservation):
            raise TypeError("head-localization observation must be HeadObservation or None")
        if (observation is not None) != (reason is HeadLocalizationReason.LOCALIZED):
            raise ValueError(
                "localized outcomes require an observation and rejected outcomes forbid one"
            )
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class HeadWorkerResult:
    """Immutable localization result tied to one exact source observation."""

    submission_id: int
    source_timestamp_ns: int
    completed_timestamp_ns: int
    identity_generation: int
    selected_player_box: Box
    observation: HeadObservation | None
    localization_reason: HeadLocalizationReason | None = None

    def __post_init__(self) -> None:
        reason = self.localization_reason
        if reason is None:
            reason = (
                HeadLocalizationReason.LOCALIZED
                if self.observation is not None
                else HeadLocalizationReason.UNSPECIFIED_NO_HEAD
            )
        outcome = HeadLocalizationOutcome(reason, self.observation)
        object.__setattr__(self, "localization_reason", outcome.reason)

    @property
    def head_point(self) -> Point | None:
        return self.observation.point if self.observation is not None else None

    def head_point_if_current(self, identity_generation: int) -> Point | None:
        """Return the original absolute point only for the same identity."""

        current = _non_negative_integer(
            identity_generation,
            "identity_generation",
        )
        if current != self.identity_generation:
            return None
        return self.head_point


@dataclass(frozen=True, slots=True)
class HeadWorkerStatus:
    lifecycle: str
    worker_alive: bool
    pending: bool
    processing: bool
    result_pending: bool
    active_identity_generation: int | None
    submissions: int
    accepted_submissions: int
    pending_overwrites: int
    stale_submissions: int
    stale_pending_dropped: int
    jobs_started: int
    jobs_completed: int
    localized_heads: int
    no_head_results: int
    no_decoded_head_candidates: int
    no_plausible_heads: int
    multiple_plausible_heads: int
    no_matching_secondary_players: int
    multiple_matching_secondary_players: int
    head_unsupported_by_matched_player: int
    unspecified_no_head_results: int
    stale_results_dropped: int
    stopped_results_dropped: int
    result_overwrites: int
    results_taken: int
    failures: int
    last_error: str | None
    latest_source_timestamp_ns: int | None
    last_completed_timestamp_ns: int | None


@dataclass(frozen=True, slots=True)
class OnnxTensorContract:
    name: str
    shape: tuple[int, ...]
    element_type: str = "tensor(float)"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("ONNX tensor contract name must not be empty")
        shape = tuple(
            _positive_integer(value, "ONNX tensor dimension")
            for value in self.shape
        )
        if not shape:
            raise ValueError("ONNX tensor contract shape must not be empty")
        element_type = str(self.element_type).strip()
        if not element_type:
            raise ValueError("ONNX tensor element type must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "element_type", element_type)


@dataclass(frozen=True, slots=True)
class OnnxModelContract:
    input: OnnxTensorContract
    output: OnnxTensorContract


@dataclass(frozen=True, slots=True)
class CpuOnnxSessionInfo:
    model_path: str
    runtime_version: str
    provider: str
    intra_op_threads: int
    inter_op_threads: int
    execution_mode: str
    input: OnnxTensorContract
    output: OnnxTensorContract


class CpuOnnxSession:
    """Strict one-input/one-output CPU session for a pinned model contract."""

    def __init__(
        self,
        model_path: str | Path,
        contract: OnnxModelContract,
        *,
        intra_op_threads: int = 6,
        runtime_module: Any | None = None,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        if session_factory is None and not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")
        self.contract = contract
        threads = _positive_integer(intra_op_threads, "intra_op_threads")

        if runtime_module is None:
            try:
                import onnxruntime as runtime_module  # type: ignore[import-not-found]
            except ImportError as exc:
                raise DependencyError(
                    "ONNX Runtime is required for the CPU head localizer"
                ) from exc
        runtime = runtime_module
        try:
            available = tuple(runtime.get_available_providers())
        except Exception as exc:
            raise DependencyError(
                f"Could not query ONNX Runtime providers: {exc}"
            ) from exc
        if CPU_PROVIDER not in available:
            raise DeviceNotAvailableError(
                f"{CPU_PROVIDER} is not installed; available providers: "
                + (", ".join(available) or "none")
            )

        try:
            options = runtime.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
            factory = session_factory or runtime.InferenceSession
            session = factory(
                str(self.model_path),
                sess_options=options,
                providers=[CPU_PROVIDER],
            )
        except Exception as exc:
            raise ModelError(
                f"Could not load CPU ONNX model {self.model_path}: {exc}"
            ) from exc

        self._validate_session(session, contract)
        self._session = session
        self._input_name = contract.input.name
        self._output_name = contract.output.name
        self.info = CpuOnnxSessionInfo(
            model_path=str(self.model_path),
            runtime_version=str(getattr(runtime, "__version__", "unknown")),
            provider=CPU_PROVIDER,
            intra_op_threads=threads,
            inter_op_threads=1,
            execution_mode="ORT_SEQUENTIAL",
            input=contract.input,
            output=contract.output,
        )

    @staticmethod
    def _validate_session(session: Any, contract: OnnxModelContract) -> None:
        try:
            active = tuple(str(value) for value in session.get_providers())
            inputs = list(session.get_inputs())
            outputs = list(session.get_outputs())
        except Exception as exc:
            raise ModelError(f"Could not inspect CPU ONNX session: {exc}") from exc
        if active != (CPU_PROVIDER,):
            raise ModelError(
                "CPU head session activated an unexpected provider chain: "
                + (", ".join(active) or "none")
            )
        if len(inputs) != 1 or len(outputs) != 1:
            raise ModelError(
                "CPU head model must expose exactly one input and one output"
            )
        CpuOnnxSession._validate_tensor(inputs[0], contract.input, "input")
        CpuOnnxSession._validate_tensor(outputs[0], contract.output, "output")

    @staticmethod
    def _validate_tensor(
        actual: Any,
        expected: OnnxTensorContract,
        role: str,
        *,
        model_label: str = "CPU head model",
    ) -> None:
        name = str(getattr(actual, "name", ""))
        shape = tuple(getattr(actual, "shape", ()))
        element_type = str(getattr(actual, "type", ""))
        if name != expected.name:
            raise ModelError(
                f"{model_label} {role} name {name!r} does not match "
                f"the pinned contract {expected.name!r}"
            )
        if shape != expected.shape:
            raise ModelError(
                f"{model_label} {role} shape {list(shape)} does not match "
                f"the pinned contract {list(expected.shape)}"
            )
        if element_type != expected.element_type:
            raise ModelError(
                f"{model_label} {role} type {element_type!r} does not match "
                f"the pinned contract {expected.element_type!r}"
            )

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        array = np.asarray(tensor)
        if array.shape != self.contract.input.shape:
            raise ValueError(
                f"head model input must have shape {self.contract.input.shape}, "
                f"got {array.shape}"
            )
        if array.dtype != np.float32:
            raise ValueError(
                f"head model input must use float32, got {array.dtype}"
            )
        try:
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: array},
            )
        except Exception as exc:
            raise ModelError(f"CPU head-model inference failed: {exc}") from exc
        if len(outputs) != 1:
            raise ModelError(
                f"CPU head-model inference returned {len(outputs)} outputs"
            )
        output = np.asarray(outputs[0])
        if output.shape != self.contract.output.shape:
            raise ModelError(
                f"CPU head-model output must have shape {self.contract.output.shape}, "
                f"got {output.shape}"
            )
        if output.dtype != np.float32:
            raise ModelError(
                f"CPU head-model output must use float32, got {output.dtype}"
            )
        return output

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        return self.infer(tensor)


@dataclass(frozen=True, slots=True)
class StrictProviderOnnxSessionInfo:
    """Auditable identity for one accelerator-only ONNX session."""

    model_path: str
    runtime_version: str
    provider: str
    active_providers: tuple[str, ...]
    configured_session_options: tuple[tuple[str, str], ...]
    cpu_graph_fallback_disabled: bool
    ep_failure_fallback_disabled: bool
    input: OnnxTensorContract
    output: OnnxTensorContract


class StrictProviderOnnxSession:
    """Run one pinned model on exactly one non-CPU execution provider.

    ONNX Runtime has two independent fallback mechanisms.  The session config
    entry prevents unsupported graph nodes from being assigned to CPU during
    construction, while :meth:`InferenceSession.disable_fallback` prevents the
    Python wrapper from retrying a failed accelerator run on CPU.  Both are
    mandatory here; an older runtime that cannot provide either control is
    rejected before warmup.
    """

    def __init__(
        self,
        model_path: str | Path,
        contract: OnnxModelContract,
        *,
        provider: str,
        runtime_module: Any | None = None,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        if session_factory is None and not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")
        self.contract = contract
        requested_provider = str(provider).strip()
        if not requested_provider:
            raise ValueError("strict ONNX provider must not be empty")
        if requested_provider == CPU_PROVIDER:
            raise ValueError("strict GPU head session refuses CPUExecutionProvider")

        if runtime_module is None:
            try:
                import onnxruntime as runtime_module  # type: ignore[import-not-found]
            except ImportError as exc:
                raise DependencyError(
                    "ONNX Runtime with the requested GPU provider is required "
                    "for the strict head localizer"
                ) from exc
        runtime = runtime_module
        try:
            available = tuple(str(value) for value in runtime.get_available_providers())
        except Exception as exc:
            raise DependencyError(
                f"Could not query ONNX Runtime providers: {exc}"
            ) from exc
        if requested_provider not in available:
            raise DeviceNotAvailableError(
                f"Strict GPU head provider {requested_provider!r} is not installed; "
                "available providers: " + (", ".join(available) or "none")
            )

        configured_session_options: dict[str, Any]
        try:
            # Reuse the same provider-compatible options as the primary ONNX
            # detector instead of creating a second, subtly different GPU
            # configuration path.
            from .onnx_yolo import _configure_session_options

            options = runtime.SessionOptions()
            configured_session_options = _configure_session_options(
                runtime,
                options,
                (requested_provider,),
            )
            add_config = getattr(options, "add_session_config_entry", None)
            if not callable(add_config):
                raise RuntimeError(
                    "this ONNX Runtime cannot disable CPU graph fallback"
                )
            add_config("session.disable_cpu_ep_fallback", "1")
            configured_session_options["disable_cpu_ep_fallback"] = True
            factory = session_factory or runtime.InferenceSession
            # Do not register CPU at all.  Provider availability does not grant
            # it permission to execute any part of this model.
            session = factory(
                str(self.model_path),
                sess_options=options,
                providers=[requested_provider],
            )
        except Exception as exc:
            raise ModelError(
                f"Could not load strict GPU ONNX model {self.model_path} on "
                f"{requested_provider}: {exc}"
            ) from exc

        disable_runtime_fallback = getattr(session, "disable_fallback", None)
        if not callable(disable_runtime_fallback):
            raise ModelError(
                "Strict GPU head inference requires an ONNX Runtime session "
                "that can disable execution-provider failure fallback"
            )
        try:
            disable_runtime_fallback()
        except Exception as exc:
            raise ModelError(
                "Could not disable ONNX Runtime execution-provider failure "
                f"fallback for the GPU head session: {exc}"
            ) from exc

        active_providers = self._validate_session(
            session,
            contract,
            requested_provider,
        )
        self._session = session
        self._provider = requested_provider
        self._input_name = contract.input.name
        self._output_name = contract.output.name
        self.info = StrictProviderOnnxSessionInfo(
            model_path=str(self.model_path),
            runtime_version=str(getattr(runtime, "__version__", "unknown")),
            provider=requested_provider,
            active_providers=active_providers,
            configured_session_options=tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in configured_session_options.items()
                )
            ),
            cpu_graph_fallback_disabled=True,
            ep_failure_fallback_disabled=True,
            input=contract.input,
            output=contract.output,
        )

    @staticmethod
    def _require_strict_provider_chain(
        session: Any,
        provider: str,
    ) -> tuple[str, ...]:
        try:
            active = tuple(str(value) for value in session.get_providers())
        except Exception as exc:
            raise ModelError(
                f"Could not inspect strict GPU head providers: {exc}"
            ) from exc
        # ORT 1.28 implicitly reports CPUExecutionProvider after the explicitly
        # registered accelerator even when providers=[provider].  The required
        # session.disable_cpu_ep_fallback=1 option prevents it from receiving
        # graph nodes.  Accept that one documented implicit registration, but
        # no alternate accelerator, reordered provider, or CPU-only chain.
        if active not in ((provider,), (provider, CPU_PROVIDER)):
            raise ModelError(
                "Strict GPU head session activated an unexpected provider chain: "
                + (", ".join(active) or "none")
                + f"; required {provider} first with CPU execution disabled"
            )
        return active

    @staticmethod
    def _validate_session(
        session: Any,
        contract: OnnxModelContract,
        provider: str,
    ) -> tuple[str, ...]:
        active = StrictProviderOnnxSession._require_strict_provider_chain(
            session,
            provider,
        )
        try:
            inputs = list(session.get_inputs())
            outputs = list(session.get_outputs())
        except Exception as exc:
            raise ModelError(f"Could not inspect strict GPU ONNX session: {exc}") from exc
        if len(inputs) != 1 or len(outputs) != 1:
            raise ModelError(
                "Strict GPU head model must expose exactly one input and one output"
            )
        CpuOnnxSession._validate_tensor(
            inputs[0],
            contract.input,
            "input",
            model_label="Strict GPU head model",
        )
        CpuOnnxSession._validate_tensor(
            outputs[0],
            contract.output,
            "output",
            model_label="Strict GPU head model",
        )
        return active

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        array = np.asarray(tensor)
        if array.shape != self.contract.input.shape:
            raise ValueError(
                f"head model input must have shape {self.contract.input.shape}, "
                f"got {array.shape}"
            )
        if array.dtype != np.float32:
            raise ValueError(f"head model input must use float32, got {array.dtype}")
        # Check before and after every inference.  ``disable_fallback`` is the
        # prevention mechanism; these checks make any runtime contract drift
        # visible and permanently fail the worker instead of continuing.
        self._require_strict_provider_chain(self._session, self._provider)
        try:
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: array},
            )
        except Exception as exc:
            raise ModelError(
                f"Strict GPU head-model inference failed on {self._provider}: {exc}"
            ) from exc
        self._require_strict_provider_chain(self._session, self._provider)
        if len(outputs) != 1:
            raise ModelError(
                f"Strict GPU head-model inference returned {len(outputs)} outputs"
            )
        output = np.asarray(outputs[0])
        if output.shape != self.contract.output.shape:
            raise ModelError(
                f"Strict GPU head-model output must have shape "
                f"{self.contract.output.shape}, got {output.shape}"
            )
        if output.dtype != np.float32:
            raise ModelError(
                f"Strict GPU head-model output must use float32, got {output.dtype}"
            )
        return output

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        return self.infer(tensor)


@dataclass(frozen=True, slots=True)
class _PendingJob:
    submission_id: int
    source_timestamp_ns: int
    identity_generation: int
    selected_player_box: Box
    payload: Any


class LatestHeadWorker:
    """Run one model-specific callable with at most one pending payload.

    ``localize(payload, selected_player_box)`` may return a typed
    :class:`HeadLocalizationOutcome`, or the legacy ``HeadObservation | None``
    shape.  A typed outcome keeps the exact non-spatial rejection disposition
    attached to the completed job instead of publishing a racy side channel.
    The default payload copier is ``deepcopy``; integrations should submit a
    bounded prepared crop/tensor rather than a full 240 FPS capture frame.
    """

    def __init__(
        self,
        localize: Callable[
            [Any, Box],
            HeadLocalizationOutcome | HeadObservation | None,
        ],
        *,
        payload_copier: Callable[[Any], Any] = deepcopy,
        thread_name: str = "proaim-head-localizer",
        stop_timeout_s: float = 1.0,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if not callable(localize):
            raise TypeError("localize must be callable")
        if not callable(payload_copier):
            raise TypeError("payload_copier must be callable")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        timeout = float(stop_timeout_s)
        if not isfinite(timeout) or timeout <= 0.0:
            raise ValueError("stop_timeout_s must be finite and positive")

        self._localize = localize
        self._payload_copier = payload_copier
        self._thread_name = str(thread_name)
        self.stop_timeout_s = timeout
        self._clock_ns = clock_ns
        self._condition = Condition()
        self._thread: Thread | None = None
        self._stop_requested = False
        self._lifecycle = "new"
        self._pending: _PendingJob | None = None
        self._latest_result: HeadWorkerResult | None = None
        self._processing = False
        self._active_identity_generation: int | None = None
        self._latest_source_timestamp_ns: int | None = None
        self._next_submission_id = 1
        self._error: RuntimeError | None = None

        self._submissions = 0
        self._accepted_submissions = 0
        self._pending_overwrites = 0
        self._stale_submissions = 0
        self._stale_pending_dropped = 0
        self._jobs_started = 0
        self._jobs_completed = 0
        self._localized_heads = 0
        self._no_head_results = 0
        self._no_decoded_head_candidates = 0
        self._no_plausible_heads = 0
        self._multiple_plausible_heads = 0
        self._no_matching_secondary_players = 0
        self._multiple_matching_secondary_players = 0
        self._head_unsupported_by_matched_player = 0
        self._unspecified_no_head_results = 0
        self._stale_results_dropped = 0
        self._stopped_results_dropped = 0
        self._result_overwrites = 0
        self._results_taken = 0
        self._failures = 0
        self._last_completed_timestamp_ns: int | None = None

    def start(self) -> None:
        with self._condition:
            if self._lifecycle != "new":
                raise RuntimeError("head-localization worker cannot be restarted")
            self._lifecycle = "running"
            thread = Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def submit(
        self,
        payload: Any,
        *,
        source_timestamp_ns: int,
        identity_generation: int,
        selected_player_box: Sequence[float],
    ) -> bool:
        """Own and publish one prepared frame payload without queueing."""

        timestamp = _non_negative_integer(
            source_timestamp_ns,
            "source_timestamp_ns",
        )
        generation = _non_negative_integer(
            identity_generation,
            "identity_generation",
        )
        player_box = _box(selected_player_box)

        with self._condition:
            self._submissions += 1
            if self._error is not None:
                raise self._error
            if self._lifecycle != "running" or self._stop_requested:
                return False
            if self._submission_is_stale(generation, timestamp):
                self._stale_submissions += 1
                return False

        owned_payload = self._payload_copier(payload)
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._lifecycle != "running" or self._stop_requested:
                return False
            if self._submission_is_stale(generation, timestamp):
                self._stale_submissions += 1
                return False

            if (
                self._active_identity_generation is None
                or generation > self._active_identity_generation
            ):
                self._activate_identity(generation)
            if self._pending is not None:
                self._pending_overwrites += 1
            submission_id = self._next_submission_id
            self._next_submission_id += 1
            self._pending = _PendingJob(
                submission_id=submission_id,
                source_timestamp_ns=timestamp,
                identity_generation=generation,
                selected_player_box=player_box,
                payload=owned_payload,
            )
            self._latest_source_timestamp_ns = timestamp
            self._accepted_submissions += 1
            self._condition.notify_all()
        return True

    def _submission_is_stale(self, generation: int, timestamp: int) -> bool:
        active = self._active_identity_generation
        if active is None or generation > active:
            return False
        if generation < active:
            return True
        latest_timestamp = self._latest_source_timestamp_ns
        return latest_timestamp is not None and timestamp <= latest_timestamp

    def _activate_identity(self, generation: int) -> None:
        self._active_identity_generation = generation
        self._latest_source_timestamp_ns = None
        if self._pending is not None:
            self._pending = None
            self._stale_pending_dropped += 1
        if self._latest_result is not None:
            self._latest_result = None
            self._stale_results_dropped += 1

    def advance_identity(self, identity_generation: int) -> bool:
        """Invalidate pending/results when selection changes or disappears."""

        generation = _non_negative_integer(
            identity_generation,
            "identity_generation",
        )
        with self._condition:
            active = self._active_identity_generation
            if active is not None and generation < active:
                return False
            if active == generation:
                return False
            self._activate_identity(generation)
            self._condition.notify_all()
            return True

    def take_latest(self, identity_generation: int) -> HeadWorkerResult | None:
        """Take a result only when its selected identity is still current."""

        generation = _non_negative_integer(
            identity_generation,
            "identity_generation",
        )
        with self._condition:
            result = self._latest_result
            self._latest_result = None
            if result is None:
                return None
            if result.identity_generation != generation:
                self._stale_results_dropped += 1
                return None
            self._results_taken += 1
            return result

    def raise_if_failed(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise error

    def stop(self, timeout_s: float | None = None) -> bool:
        """Discard pending work, request exit, and join for a bounded time."""

        if timeout_s is None:
            timeout = self.stop_timeout_s
        else:
            timeout = float(timeout_s)
            if not isfinite(timeout) or timeout < 0.0:
                raise ValueError("timeout_s must be finite and non-negative")

        with self._condition:
            if self._lifecycle == "new":
                self._lifecycle = "stopped"
                return True
            self._stop_requested = True
            if self._lifecycle not in {"failed", "stopped"}:
                self._lifecycle = "stopping"
            if self._pending is not None:
                self._stopped_results_dropped += 1
            if self._latest_result is not None:
                self._stopped_results_dropped += 1
            self._pending = None
            self._latest_result = None
            thread = self._thread
            self._condition.notify_all()
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    @property
    def status(self) -> HeadWorkerStatus:
        with self._condition:
            thread = self._thread
            return HeadWorkerStatus(
                lifecycle=self._lifecycle,
                worker_alive=bool(thread is not None and thread.is_alive()),
                pending=self._pending is not None,
                processing=self._processing,
                result_pending=self._latest_result is not None,
                active_identity_generation=self._active_identity_generation,
                submissions=self._submissions,
                accepted_submissions=self._accepted_submissions,
                pending_overwrites=self._pending_overwrites,
                stale_submissions=self._stale_submissions,
                stale_pending_dropped=self._stale_pending_dropped,
                jobs_started=self._jobs_started,
                jobs_completed=self._jobs_completed,
                localized_heads=self._localized_heads,
                no_head_results=self._no_head_results,
                no_decoded_head_candidates=self._no_decoded_head_candidates,
                no_plausible_heads=self._no_plausible_heads,
                multiple_plausible_heads=self._multiple_plausible_heads,
                no_matching_secondary_players=(
                    self._no_matching_secondary_players
                ),
                multiple_matching_secondary_players=(
                    self._multiple_matching_secondary_players
                ),
                head_unsupported_by_matched_player=(
                    self._head_unsupported_by_matched_player
                ),
                unspecified_no_head_results=self._unspecified_no_head_results,
                stale_results_dropped=self._stale_results_dropped,
                stopped_results_dropped=self._stopped_results_dropped,
                result_overwrites=self._result_overwrites,
                results_taken=self._results_taken,
                failures=self._failures,
                last_error=str(self._error) if self._error is not None else None,
                latest_source_timestamp_ns=self._latest_source_timestamp_ns,
                last_completed_timestamp_ns=self._last_completed_timestamp_ns,
            )

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._stop_requested:
                        self._condition.wait()
                    if self._stop_requested:
                        break
                    job = self._pending
                    self._pending = None
                    self._processing = True
                    self._jobs_started += 1
                assert job is not None

                try:
                    raw_outcome = self._localize(
                        job.payload,
                        job.selected_player_box,
                    )
                    if isinstance(raw_outcome, HeadLocalizationOutcome):
                        outcome = raw_outcome
                    elif isinstance(raw_outcome, HeadObservation):
                        outcome = HeadLocalizationOutcome(
                            HeadLocalizationReason.LOCALIZED,
                            raw_outcome,
                        )
                    elif raw_outcome is None:
                        outcome = HeadLocalizationOutcome(
                            HeadLocalizationReason.UNSPECIFIED_NO_HEAD,
                            None,
                        )
                    else:
                        raise TypeError(
                            "localize must return HeadLocalizationOutcome, "
                            "HeadObservation, or None"
                        )
                    completed_timestamp = _non_negative_integer(
                        self._clock_ns(),
                        "clock_ns result",
                    )
                except BaseException as exc:
                    self._record_failure(job, exc)
                    break

                with self._condition:
                    self._processing = False
                    self._jobs_completed += 1
                    self._last_completed_timestamp_ns = completed_timestamp
                    if self._stop_requested:
                        self._stopped_results_dropped += 1
                        continue
                    if job.identity_generation != self._active_identity_generation:
                        self._stale_results_dropped += 1
                        continue
                    result = HeadWorkerResult(
                        submission_id=job.submission_id,
                        source_timestamp_ns=job.source_timestamp_ns,
                        completed_timestamp_ns=completed_timestamp,
                        identity_generation=job.identity_generation,
                        selected_player_box=job.selected_player_box,
                        observation=outcome.observation,
                        localization_reason=outcome.reason,
                    )
                    if self._latest_result is not None:
                        self._result_overwrites += 1
                    self._latest_result = result
                    if outcome.observation is None:
                        self._no_head_results += 1
                    else:
                        self._localized_heads += 1
                    self._record_localization_reason(outcome.reason)
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._processing = False
                self._pending = None
                if self._lifecycle != "failed":
                    self._lifecycle = "stopped"
                self._condition.notify_all()

    def _record_localization_reason(self, reason: HeadLocalizationReason) -> None:
        """Increment one exact disposition counter while holding the condition."""

        if reason is HeadLocalizationReason.LOCALIZED:
            return
        if reason is HeadLocalizationReason.NO_DECODED_HEAD_CANDIDATE:
            self._no_decoded_head_candidates += 1
        elif reason is HeadLocalizationReason.NO_PLAUSIBLE_HEAD:
            self._no_plausible_heads += 1
        elif reason is HeadLocalizationReason.MULTIPLE_PLAUSIBLE_HEADS:
            self._multiple_plausible_heads += 1
        elif reason is HeadLocalizationReason.NO_MATCHING_SECONDARY_PLAYER:
            self._no_matching_secondary_players += 1
        elif reason is HeadLocalizationReason.MULTIPLE_MATCHING_SECONDARY_PLAYERS:
            self._multiple_matching_secondary_players += 1
        elif reason is HeadLocalizationReason.HEAD_UNSUPPORTED_BY_MATCHED_PLAYER:
            self._head_unsupported_by_matched_player += 1
        elif reason is HeadLocalizationReason.UNSPECIFIED_NO_HEAD:
            self._unspecified_no_head_results += 1
        else:  # pragma: no cover - enum exhaustiveness guard
            raise AssertionError(f"unhandled head-localization reason: {reason}")

    def _record_failure(self, job: _PendingJob, exc: BaseException) -> None:
        error = RuntimeError(
            "Head-localization worker failed for submission "
            f"{job.submission_id}: {type(exc).__name__}: {exc}"
        )
        with self._condition:
            self._processing = False
            self._failures += 1
            self._error = error
            self._stop_requested = True
            self._lifecycle = "failed"
            self._pending = None
            self._latest_result = None
            self._condition.notify_all()
