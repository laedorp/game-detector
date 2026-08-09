"""Low-latency OpenVINO runtime wrapper for supported YOLO exports."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import (
    DependencyError,
    Detector,
    DetectorError,
    DeviceNotAvailableError,
    ModelError,
    OutputDecodeError,
)
from .postprocess import FrameTransformLike, decode_yolo_output
from .types import Detection


Decoder = Callable[..., Sequence[Detection]]


def _load_openvino() -> tuple[Any, str]:
    try:
        import openvino as ov
    except ImportError as exc:
        raise DependencyError(
            "OpenVINO is not installed. Install the project's runtime dependencies "
            "before creating OpenVINOYoloDetector."
        ) from exc

    core_type = getattr(ov, "Core", None)
    if core_type is None:
        try:
            from openvino.runtime import Core as core_type
        except ImportError as exc:
            raise DependencyError(
                "The installed OpenVINO package does not expose the Runtime Core API. "
                "Install a supported OpenVINO runtime release."
            ) from exc
    version = str(getattr(ov, "__version__", "unknown"))
    return core_type, version


def _dimension_value(dimension: Any) -> int | None:
    try:
        if dimension.is_dynamic:
            return None
        return int(dimension.get_length())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        try:
            value = int(dimension)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None


def _partial_shape_values(port: Any) -> list[int | None]:
    partial_shape = port.partial_shape
    try:
        rank = partial_shape.rank
        if rank.is_dynamic or rank.get_length() != 4:
            raise ModelError(f"Expected one rank-4 NCHW input, got shape {partial_shape}.")
    except AttributeError as exc:
        raise ModelError(f"Could not inspect model input shape {partial_shape!s}.") from exc
    return [_dimension_value(dimension) for dimension in partial_shape]


def _normalize_inference_size(inference_size: int | Sequence[int]) -> int:
    if isinstance(inference_size, bool):
        raise ValueError("inference_size must be a positive integer.")
    if isinstance(inference_size, int):
        size = inference_size
    else:
        values = tuple(int(value) for value in inference_size)
        if len(values) != 2 or values[0] != values[1]:
            raise ValueError(
                "inference_size must be a positive integer or an equal (height, width) pair."
            )
        size = values[0]
    if size <= 0:
        raise ValueError(f"inference_size must be positive, got {size}.")
    return size


def _load_labels(path: str | Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    labels_path = Path(path).expanduser()
    if not labels_path.is_file():
        raise FileNotFoundError(f"Class label file not found: {labels_path}")

    try:
        if labels_path.suffix.lower() == ".json":
            parsed = json.loads(labels_path.read_text(encoding="utf-8"))
            if isinstance(parsed, list):
                labels = [str(value).strip() for value in parsed]
            elif isinstance(parsed, dict):
                numeric = {int(key): str(value).strip() for key, value in parsed.items()}
                if sorted(numeric) != list(range(len(numeric))):
                    raise ValueError("JSON label keys must be contiguous integers starting at zero.")
                labels = [numeric[index] for index in range(len(numeric))]
            else:
                raise ValueError("JSON labels must be a list or an integer-keyed object.")
        else:
            labels = [
                line.strip()
                for line in labels_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelError(f"Could not load class labels from {labels_path}: {exc}") from exc

    if not labels or any(not label for label in labels):
        raise ModelError(f"Class label file is empty or invalid: {labels_path}")
    return tuple(labels)


def _device_is_available(requested: str, available: Sequence[str]) -> bool:
    # AUTO/MULTI/HETERO/BATCH are OpenVINO virtual devices and therefore do
    # not appear in Core.available_devices.  Let compile_model validate their
    # optional device lists and configuration.
    virtual_device = requested.partition(":")[0].partition("(")[0]
    if virtual_device in {"AUTO", "MULTI", "HETERO", "BATCH"}:
        return True
    if requested in available:
        return True
    # A generic GPU request is valid when OpenVINO enumerates GPU.0, GPU.1, etc.
    return "." not in requested and any(device.startswith(f"{requested}.") for device in available)


def _serializable_property(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serializable_property(item) for item in value]
    return str(value)


class OpenVINOYoloDetector(Detector):
    """Run one static, batch-one YOLO model using a single OpenVINO stream.

    The returned raw output is a view owned by the reusable inference request and
    must be consumed before the next call to :meth:`infer` if callers do not copy it.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path | None = None,
        device: str = "CPU",
        inference_size: int | Sequence[int] = 320,
        confidence: float = 0.25,
        iou: float = 0.45,
        output_format: str = "auto",
        *,
        decoder: Decoder | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OpenVINO model file not found: {self.model_path}")
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
            raise ValueError(
                "output_format must be 'auto', 'end2end', or 'traditional'."
            )
        self.device = str(device).strip().upper()
        if not self.device:
            raise ValueError("device must not be empty.")
        self._decoder = decoder

        core_type, openvino_version = _load_openvino()
        try:
            self._core = core_type()
            self._available_devices = tuple(str(value).upper() for value in self._core.available_devices)
        except Exception as exc:
            raise DependencyError(f"Could not initialize OpenVINO Runtime: {exc}") from exc
        if not _device_is_available(self.device, self._available_devices):
            available_text = ", ".join(self._available_devices) or "none"
            raise DeviceNotAvailableError(
                f"Requested OpenVINO device {self.device!r} is unavailable. "
                f"Detected physical devices: {available_text}."
            )

        try:
            model = self._core.read_model(str(self.model_path))
        except Exception as exc:
            raise ModelError(f"Could not read model {self.model_path}: {exc}") from exc
        if len(model.inputs) != 1:
            raise ModelError(f"Expected exactly one model input, found {len(model.inputs)}.")
        if len(model.outputs) != 1:
            raise ModelError(
                f"Expected one decoded YOLO output, found {len(model.outputs)}. "
                "Export an end-to-end or standard single-output YOLO model, or supply "
                "a custom detector backend."
            )

        original_shape = _partial_shape_values(model.input(0))
        batch, channels, _, last_dimension = original_shape
        if batch not in (None, 1):
            raise ModelError(f"Expected batch size 1, got model input shape {original_shape}.")
        if channels not in (None, 3):
            layout_hint = " The model appears to use NHWC layout." if last_dimension == 3 else ""
            raise ModelError(
                f"Expected NCHW input [1, 3, H, W], got {original_shape}.{layout_hint}"
            )
        if channels is None and last_dimension == 3:
            raise ModelError(
                f"Expected NCHW input [1, 3, H, W], but shape {original_shape} appears NHWC."
            )

        target_shape = [1, 3, self.inference_size, self.inference_size]
        if original_shape != target_shape:
            try:
                model.reshape(target_shape)
            except Exception as exc:
                raise ModelError(
                    f"Could not reshape model input from {original_shape} to {target_shape}: {exc}"
                ) from exc

        compile_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
        try:
            self._compiled_model = self._core.compile_model(
                model,
                self.device,
                compile_config,
            )
            self._input_port = self._compiled_model.input(0)
            self._output_port = self._compiled_model.output(0)
            compiled_shape = tuple(int(value) for value in self._input_port.shape)
            if compiled_shape != tuple(target_shape):
                raise ModelError(
                    f"Compiled model input is {compiled_shape}, expected {tuple(target_shape)}."
                )
            self._infer_request = self._compiled_model.create_infer_request()
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(
                f"Could not compile {self.model_path} for {self.device} with LATENCY hint "
                f"and one stream: {exc}"
            ) from exc

        runtime_properties: dict[str, Any] = {}
        for property_name in ("EXECUTION_DEVICES", "PERFORMANCE_HINT", "NUM_STREAMS"):
            try:
                runtime_properties[property_name.lower()] = _serializable_property(
                    self._compiled_model.get_property(property_name)
                )
            except Exception:
                continue

        self._runtime_summary: dict[str, Any] = {
            "runtime": "OpenVINO",
            "openvino_version": openvino_version,
            "model_path": str(self.model_path),
            "device": self.device,
            "available_devices": list(self._available_devices),
            "input_shape": list(target_shape),
            "input_element_type": str(self._input_port.element_type),
            "output_shape": str(self._output_port.partial_shape),
            "performance_hint": "LATENCY",
            "num_streams_requested": 1,
            "output_format": self.output_format,
            **runtime_properties,
        }

    @property
    def available_devices(self) -> tuple[str, ...]:
        return self._available_devices

    @property
    def runtime_summary(self) -> Mapping[str, Any]:
        return dict(self._runtime_summary)

    def summary(self) -> dict[str, Any]:
        """Return a mutable copy suitable for startup logging or benchmark JSON."""

        return dict(self._runtime_summary)

    def warmup(self, iterations: int = 3) -> None:
        if isinstance(iterations, bool) or iterations < 0:
            raise ValueError("warmup iterations must be a non-negative integer.")
        tensor = np.zeros(
            (1, 3, self.inference_size, self.inference_size),
            dtype=np.float32,
        )
        for _ in range(iterations):
            self.infer(tensor)

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        array = np.asarray(tensor)
        expected_shape = (1, 3, self.inference_size, self.inference_size)
        if array.shape != expected_shape:
            raise ValueError(f"Inference tensor must have shape {expected_shape}, got {array.shape}.")
        if array.dtype != np.float32:
            raise TypeError(f"Inference tensor must use float32, got {array.dtype}.")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        try:
            result = self._infer_request.infer(
                {self._input_port: array},
                share_inputs=True,
                share_outputs=True,
            )
            return np.asarray(result[self._output_port])
        except Exception as exc:
            raise DetectorError(f"OpenVINO inference failed on {self.device}: {exc}") from exc

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
