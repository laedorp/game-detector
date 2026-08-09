"""Detector interface and package-specific errors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .types import Detection


class DetectorError(RuntimeError):
    """Base error for detector setup and execution failures."""


class DependencyError(DetectorError):
    """A required detector runtime dependency is unavailable."""


class DeviceNotAvailableError(DetectorError):
    """The requested inference device is unavailable."""


class ModelError(DetectorError):
    """The model cannot be loaded or does not meet the detector contract."""


class OutputDecodeError(DetectorError):
    """The model output is not a supported YOLO output layout."""


class Detector(ABC):
    """Minimal interface implemented by replaceable detector backends."""

    @property
    @abstractmethod
    def available_devices(self) -> tuple[str, ...]:
        """Physical inference devices reported by the runtime."""

    @property
    @abstractmethod
    def runtime_summary(self) -> Mapping[str, Any]:
        """Serializable runtime and model configuration details."""

    @abstractmethod
    def warmup(self, iterations: int = 3) -> None:
        """Run unmeasured inference iterations to initialize runtime caches."""

    @abstractmethod
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Run inference and return the raw model output."""

    @abstractmethod
    def postprocess(
        self,
        raw: np.ndarray,
        transform: Any | None = None,
        frame_shape: Sequence[int] | None = None,
    ) -> list[Detection]:
        """Decode raw output and map boxes to source-frame coordinates."""
