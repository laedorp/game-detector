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
from .postprocess import FrameTransformLike, decode_yolo_output
from .openvino_yolo import Decoder, _load_labels, _normalize_inference_size
from .types import Detection


# Ordered by expected throughput.  ``CPUExecutionProvider`` is always last and
# always present, so a session can still be created when nothing else is.
PROVIDER_PREFERENCE = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
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
}


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
    if normalized in PROVIDER_ALIASES:
        target = PROVIDER_ALIASES[normalized]
    elif requested in available_set or requested.endswith("ExecutionProvider"):
        target = requested
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

        try:
            options = runtime.SessionOptions()
            # One thread pool sized for latency, mirroring the OpenVINO backend's
            # single-stream LATENCY hint rather than optimizing for throughput.
            options.graph_optimization_level = (
                runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            factory = session_factory or runtime.InferenceSession
            self._session = factory(
                str(self.model_path), sess_options=options, providers=chain
            )
        except Exception as exc:
            raise ModelError(
                f"Could not load ONNX model {self.model_path} on {requested_device}: {exc}"
            ) from exc

        try:
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            self._input_name = inputs[0].name
            self._output_name = outputs[0].name
            declared = list(inputs[0].shape)
        except Exception as exc:
            raise ModelError(f"Could not inspect ONNX model {self.model_path}: {exc}") from exc

        # A static export pins every dimension; a dynamic axis arrives as a
        # string name.  Only a mismatched *static* size is an error, because a
        # dynamic graph legitimately accepts the configured size.
        spatial = [value for value in declared[2:4] if isinstance(value, int)]
        if spatial and any(value != self.inference_size for value in spatial):
            raise ModelError(
                f"ONNX model expects input {declared}, which does not match the "
                f"configured inference size {self.inference_size}."
            )

        active = list(getattr(self._session, "get_providers", lambda: chain)())
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
            "input_name": self._input_name,
            "input_shape": declared,
            "output_name": self._output_name,
            "output_shape": list(outputs[0].shape),
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
