"""Shared detector result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    """One detection mapped to the original source-frame coordinates."""

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Alias used by renderers that call an ``xyxy`` box a bounding box."""

        return self.xyxy

    @property
    def box(self) -> tuple[float, float, float, float]:
        """Concise bounding-box alias for preview renderers."""

        return self.xyxy

    @property
    def label(self) -> str:
        """Human-readable class-name alias for preview renderers."""

        return self.class_name

    @property
    def x1(self) -> float:
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        return self.xyxy[3]
