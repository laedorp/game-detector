from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class FrameTimings:
    capture_ms: float
    queue_age_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    control_ms: float
    processing_ms: float
    freshness_latency_ms: float
    observed_pipeline_ms: float
    draw_ms: float
    display_ms: float


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    processed_frames: int
    moving_fps: float
    latest: FrameTimings
    average: FrameTimings


class RollingMetrics:
    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")
        self._samples: deque[FrameTimings] = deque(maxlen=window_size)
        self._completion_times: deque[int] = deque(maxlen=window_size)
        self._processed_frames = 0

    def record(self, timings: FrameTimings, completed_ns: int) -> MetricsSnapshot:
        self._samples.append(timings)
        self._completion_times.append(completed_ns)
        self._processed_frames += 1
        return self.snapshot()

    def snapshot(self) -> MetricsSnapshot:
        if not self._samples:
            zero = FrameTimings(
                *(0.0 for _ in range(len(FrameTimings.__dataclass_fields__)))
            )
            return MetricsSnapshot(0, 0.0, zero, zero)

        average = FrameTimings(
            *(
                fmean(getattr(sample, field) for sample in self._samples)
                for field in FrameTimings.__dataclass_fields__
            )
        )
        fps = 0.0
        if len(self._completion_times) >= 2:
            elapsed_seconds = (self._completion_times[-1] - self._completion_times[0]) / 1e9
            if elapsed_seconds > 0:
                fps = (len(self._completion_times) - 1) / elapsed_seconds
        return MetricsSnapshot(
            processed_frames=self._processed_frames,
            moving_fps=fps,
            latest=self._samples[-1],
            average=average,
        )
