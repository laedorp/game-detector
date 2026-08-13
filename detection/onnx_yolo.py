"""ONNX Runtime detector for the hardware OpenVINO cannot drive.

OpenVINO covers Intel CPUs, integrated graphics, and NPUs, but it has no AMD or
NVIDIA GPU plugin at all.  ONNX Runtime reaches those parts through execution
providers, so this backend is what makes a Radeon or GeForce card usable without
changing anything else in the pipeline.

The two backends deliberately share :func:`decode_yolo_output`, so a model
exported to both formats produces identical detections and the rest of the
application cannot tell which one is running.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from .base import (
    DependencyError,
    DetectorError,
    DeviceNotAvailableError,
    Detector,
    ModelError,
    OutputDecodeError,
)
from .postprocess import (
    FrameTransformLike,
    decode_yolo_output,
    supported_yolo_output_layout,
)
from .openvino_yolo import Decoder, _load_labels, _normalize_inference_size
from .types import Detection


# Ordered by expected throughput.  ``CPUExecutionProvider`` is always last and
# always present, so a session can still be created when nothing else is.
PROVIDER_PREFERENCE = (
    "CUDAExecutionProvider",
    # TensorRT can outperform CUDA after engine construction, but merely being
    # listed by ONNX Runtime does not prove the separate TensorRT libraries are
    # installed. Prefer CUDA for a reliable one-click default; TensorRT remains
    # available as an explicit advanced choice.
    "TensorrtExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
)

# Friendly aliases so callers and saved settings never have to spell out the
# full provider class names.
PROVIDER_ALIASES = {
    "AUTO": None,
    # Launcher/device settings historically used OpenVINO-style tokens (CPU/GPU).
    # Treat generic GPU as AUTO so ONNX Runtime picks the best installed GPU
    # provider (DirectML/CUDA/ROCM/TensorRT) instead of failing hard.
    "GPU": None,
    "CPU": "CPUExecutionProvider",
    "CUDA": "CUDAExecutionProvider",
    "NVIDIA": "CUDAExecutionProvider",
    "TENSORRT": "TensorrtExecutionProvider",
    "ROCM": "ROCMExecutionProvider",
    "AMD": "ROCMExecutionProvider",
    "MIGRAPHX": "MIGraphXExecutionProvider",
    "DIRECTML": "DmlExecutionProvider",
    "DML": "DmlExecutionProvider",
    "OPENVINO": "OpenVINOExecutionProvider",
}

NVIDIA_EXECUTION_PROVIDERS = frozenset(
    {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
)


def _load_onnxruntime() -> tuple[Any, str]:
    try:
        import onnxruntime  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyError(
            "ONNX Runtime is required for non-Intel accelerators. Install "
            "'onnxruntime' for CPU, 'onnxruntime-gpu' for NVIDIA, "
            "'onnxruntime-directml' for AMD/NVIDIA on Windows, or "
            "'onnxruntime-rocm' for AMD on Linux."
        ) from exc
    return onnxruntime, getattr(onnxruntime, "__version__", "unknown")


def _configure_session_options(runtime: Any, options: Any, providers: Sequence[str]) -> None:
    """Apply latency-oriented defaults that work across GPU providers.

    ONNX Runtime defaults are throughput-friendly on some platforms. Real-time
    single-frame detection benefits from sequential execution and lower host
    scheduling overhead, especially when CUDA/ROCm/DirectML do the heavy work.
    """

    options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL

    execution_mode = getattr(runtime, "ExecutionMode", None)
    if execution_mode is not None and hasattr(execution_mode, "ORT_SEQUENTIAL"):
        options.execution_mode = execution_mode.ORT_SEQUENTIAL

    # DirectML paths in particular can stutter with memory pattern reuse.
    options.enable_mem_pattern = False

    gpu_chain = any(
        provider
        in {
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "MIGraphXExecutionProvider",
            "DmlExecutionProvider",
        }
        for provider in providers
    )
    if gpu_chain:
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1


def _preload_nvidia_libraries(runtime: Any, providers: Sequence[str]) -> bool:
    """Ask ONNX Runtime to load bundled CUDA/cuDNN libraries before a session.

    ONNX Runtime 1.21+ can locate the ``nvidia`` site-package libraries that
    ship beside ``onnxruntime-gpu``.  This is especially important in a frozen
    bundle, where those libraries are not necessarily on the operating
    system's default loader path.  Older source installations can still rely
    on their system loader; session creation and the active-provider check
    below remain the final authority.
    """

    if not NVIDIA_EXECUTION_PROVIDERS.intersection(providers):
        return False

    preload = getattr(runtime, "preload_dlls", None)
    if not callable(preload):
        return False
    try:
        # An empty directory tells ORT to search its bundled ``nvidia`` Python
        # packages first, without preferring an unrelated PyTorch install.
        preload(directory="")
    except Exception as exc:
        raise DependencyError(
            "Could not preload the CUDA/cuDNN libraries bundled with ONNX Runtime: "
            f"{exc}. Use the matching NVIDIA CUDA build or select DirectML/CPU."
        ) from exc
    return True


def _require_active_provider(requested: str, active: Sequence[str]) -> None:
    """Refuse ONNX Runtime's silent fallback when acceleration did not start."""

    if requested in active:
        return
    reported = ", ".join(str(provider) for provider in active) or "none"
    raise DeviceNotAvailableError(
        f"Requested ONNX Runtime provider {requested!r} did not initialize; "
        f"the session activated: {reported}. Check the GPU driver and use the "
        "matching ProAim runtime build, or select another device."
    )


def resolve_providers(
    requested: str, available: Sequence[str]
) -> tuple[list[str], str]:
    """Choose the provider chain for ``requested`` against what is installed.

    Returns the ordered chain handed to ONNX Runtime and the provider that is
    expected to actually execute the graph.  A chain always ends with the CPU
    provider so a partially supported graph still runs instead of failing.
    """

    normalized = str(requested).strip().upper()
    if not normalized:
        raise ValueError("device must not be empty.")

    available_set = set(available)
    available_by_normalized = {
        str(provider).strip().upper(): str(provider) for provider in available
    }
    known_by_normalized = {
        provider.upper(): provider for provider in PROVIDER_PREFERENCE
    }
    if normalized in PROVIDER_ALIASES:
        target = PROVIDER_ALIASES[normalized]
    elif normalized in available_by_normalized:
        # Launcher settings were historically uppercased, turning for example
        # ``TensorrtExecutionProvider`` into ``TENSORRTEXECUTIONPROVIDER``.
        # Provider class names are identifiers, not case-sensitive user input,
        # so recover the exact spelling reported by the installed runtime.
        target = available_by_normalized[normalized]
    elif normalized in known_by_normalized:
        target = known_by_normalized[normalized]
    else:
        raise DeviceNotAvailableError(
            f"Unknown ONNX Runtime device {requested!r}. Use one of: "
            + ", ".join(sorted(PROVIDER_ALIASES))
        )

    if target is None:
        for candidate in PROVIDER_PREFERENCE:
            if candidate in available_set:
                target = candidate
                break
        else:  # pragma: no cover - onnxruntime always ships a CPU provider
            raise DeviceNotAvailableError("ONNX Runtime reports no execution providers.")
    elif target not in available_set:
        raise DeviceNotAvailableError(
            f"ONNX Runtime provider {target!r} is not installed. Available: "
            + (", ".join(sorted(available_set)) or "none")
        )

    chain = [target]
    # TensorRT on Windows often appears as installed even when its runtime DLLs
    # are missing from PATH. Keep CUDA ahead of CPU so failed TensorRT setup
    # falls back to a fast NVIDIA path instead of collapsing to CPU.
    if target == "TensorrtExecutionProvider" and "CUDAExecutionProvider" in available_set:
        chain.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available_set and target != "CPUExecutionProvider":
        chain.append("CPUExecutionProvider")
    return chain, target


class OnnxRuntimeYoloDetector(Detector):
    """Run a YOLO ONNX graph on whichever accelerator this machine exposes."""

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path | None = None,
        device: str = "AUTO",
        inference_size: int | Sequence[int] = 416,
        confidence: float = 0.25,
        iou: float = 0.45,
        output_format: str = "auto",
        *,
        decoder: Decoder | None = None,
        session_factory: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        if session_factory is None and not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")
        self.labels = _load_labels(labels_path)
        self.inference_size = _normalize_inference_size(inference_size)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.output_format = str(output_format).lower()
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be a finite value between 0 and 1.")
        if not np.isfinite(self.iou) or not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou must be a finite value between 0 and 1.")
        if self.output_format not in {"auto", "end2end", "traditional"}:
            raise ValueError("output_format must be 'auto', 'end2end', or 'traditional'.")
        self._decoder = decoder

        runtime, version = _load_onnxruntime()
        try:
            self._available_devices = tuple(runtime.get_available_providers())
        except Exception as exc:
            raise DependencyError(f"Could not query ONNX Runtime providers: {exc}") from exc

        chain, requested_device = resolve_providers(device, self._available_devices)
        nvidia_preload_requested = _preload_nvidia_libraries(runtime, chain)

        try:
            options = runtime.SessionOptions()
            _configure_session_options(runtime, options, chain)
            factory = session_factory or runtime.InferenceSession
            self._session = factory(
                str(self.model_path), sess_options=options, providers=chain
            )
        except Exception as exc:
            raise ModelError(
                f"Could not load ONNX model {self.model_path} on {requested_device}: {exc}"
            ) from exc

        try:
            active = list(self._session.get_providers())
        except Exception as exc:
            raise ModelError(
                f"Could not inspect active ONNX Runtime providers: {exc}"
            ) from exc
        _require_active_provider(requested_device, active)

        try:
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if len(inputs) != 1:
                raise ModelError(
                    f"Expected exactly one ONNX model input, found {len(inputs)}."
                )
            if len(outputs) != 1:
                raise ModelError(
                    f"Expected exactly one decoded YOLO output, found {len(outputs)}."
                )
            self._input_name = inputs[0].name
            self._output_name = outputs[0].name
            declared = list(inputs[0].shape)
            output_shape = list(outputs[0].shape)
        except Exception as exc:
            if isinstance(exc, ModelError):
                raise
            raise ModelError(f"Could not inspect ONNX model {self.model_path}: {exc}") from exc

        input_type = str(getattr(inputs[0], "type", ""))
        if input_type != "tensor(float)":
            raise ModelError(
                "ONNX model input must use float32 tensor(float), "
                f"but {self._input_name!r} declares {input_type or 'an unknown type'}."
            )
        if len(declared) != 4:
            raise ModelError(
                f"Expected rank-4 NCHW input [1, 3, H, W], got {declared}."
            )
        if declared[0] != 1 or declared[1] != 3:
            raise ModelError(
                f"Expected batch-one, three-channel NCHW input [1, 3, H, W], got {declared}."
            )

        # A static export pins spatial dimensions; a dynamic axis arrives as a
        # string or None. Dynamic H/W legitimately accept the configured size.
        for value in declared[2:4]:
            if isinstance(value, bool):
                raise ModelError(f"Invalid ONNX model input shape {declared}.")
            if isinstance(value, Integral):
                if int(value) <= 0 or int(value) != self.inference_size:
                    raise ModelError(
                        f"ONNX model expects input {declared}, which does not match the "
                        f"configured inference size {self.inference_size}."
                    )
            elif value is not None and not isinstance(value, str):
                raise ModelError(f"Could not interpret ONNX model input shape {declared}.")

        output_layout = supported_yolo_output_layout(
            output_shape,
            len(self.labels),
            self.output_format,
        )
        if output_layout is None:
            label_detail = f" for {len(self.labels)} loaded label(s)" if self.labels else ""
            raise ModelError(
                f"Unsupported ONNX YOLO output shape {output_shape}{label_detail}; "
                "expected batch 1 in [1, N, 6], [1, N, 4+C], or [1, 4+C, N] layout."
            )

        active_device = active[0] if active else requested_device
        self.device = active_device
        self._runtime_summary: dict[str, Any] = {
            "runtime": "ONNX Runtime",
            "onnxruntime_version": version,
            "model_path": str(self.model_path),
            "device": active_device,
            "requested_device": requested_device,
            "available_devices": list(self._available_devices),
            "provider_chain": chain,
            "active_providers": active,
            "nvidia_preload_requested": nvidia_preload_requested,
            "input_name": self._input_name,
            "input_shape": declared,
            "input_type": input_type,
            "output_name": self._output_name,
            "output_shape": output_shape,
            "output_layout": output_layout,
            "output_format": self.output_format,
        }

    @property
    def available_devices(self) -> tuple[str, ...]:
        return self._available_devices

    @property
    def runtime_summary(self) -> Mapping[str, Any]:
        return dict(self._runtime_summary)

    def summary(self) -> dict[str, Any]:
        return dict(self._runtime_summary)

    def warmup(self, iterations: int = 3) -> None:
        if iterations <= 0:
            return
        blank = np.zeros((1, 3, self.inference_size, self.inference_size), dtype=np.float32)
        for _ in range(iterations):
            self.infer(blank)

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        array = np.asarray(tensor)
        expected_shape = (1, 3, self.inference_size, self.inference_size)
        if array.shape != expected_shape:
            raise ValueError(
                f"Inference tensor must have shape {expected_shape}, got {array.shape}."
            )
        if array.dtype != np.float32:
            raise TypeError(f"Inference tensor must use float32, got {array.dtype}.")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        try:
            outputs = self._session.run([self._output_name], {self._input_name: array})
        except Exception as exc:
            raise DetectorError(
                f"ONNX Runtime inference failed on {self.device}: {exc}"
            ) from exc
        return np.asarray(outputs[0])

    def postprocess(
        self,
        raw: np.ndarray,
        transform: FrameTransformLike | None = None,
        frame_shape: Sequence[int] | None = None,
    ) -> list[Detection]:
        if self._decoder is None:
            return decode_yolo_output(
                raw,
                transform=transform,
                frame_shape=frame_shape,
                labels=self.labels,
                confidence=self.confidence,
                iou=self.iou,
                output_format=self.output_format,
            )
        try:
            return list(
                self._decoder(
                    raw,
                    transform=transform,
                    frame_shape=frame_shape,
                    labels=self.labels,
                    confidence=self.confidence,
                    iou=self.iou,
                )
            )
        except OutputDecodeError:
            raise
        except Exception as exc:
            raise OutputDecodeError(f"Custom YOLO decoder failed: {exc}") from exc
