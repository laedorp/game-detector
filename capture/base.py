"""Common capture types and thread-safe accounting helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from threading import Condition
from time import monotonic
from types import TracebackType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A captured BGR frame and the timestamps surrounding its read."""

    image: NDArray[np.uint8]
    sequence: int
    read_started_ns: int
    read_completed_ns: int


@dataclass(frozen=True, slots=True)
class CaptureStats:
    """An immutable snapshot of capture counters."""

    frames_read: int = 0
    frames_delivered: int = 0
    frames_overwritten: int = 0
    read_failures: int = 0


class CaptureSource(ABC):
    """Interface shared by file, device, and screen capture sources.

    Threaded sources use the protected latest-frame helpers below. A pending
    packet is replaced when the producer outpaces the consumer, so capture can
    remain fresh without building a latency-inducing FIFO queue.
    """

    def __init__(self, description: str) -> None:
        self._description = description
        self._condition = Condition()
        self._latest_packet: FramePacket | None = None
        self._frames_read = 0
        self._frames_delivered = 0
        self._frames_overwritten = 0
        self._read_failures = 0
        self._started = False
        self._closing = False
        self._closed = False
        self._ended = False
        self._error: str | None = None

    @property
    def description(self) -> str:
        return self._description

    @property
    def ended(self) -> bool:
        with self._condition:
            return self._ended

    @property
    def error(self) -> str | None:
        with self._condition:
            return self._error

    @property
    def stats(self) -> CaptureStats:
        """Return a consistent snapshot; callers cannot mutate the counters."""

        with self._condition:
            return CaptureStats(
                frames_read=self._frames_read,
                frames_delivered=self._frames_delivered,
                frames_overwritten=self._frames_overwritten,
                read_failures=self._read_failures,
            )

    @property
    @abstractmethod
    def actual_settings(self) -> Mapping[str, object]:
        """Properties negotiated by the underlying capture backend."""

    @abstractmethod
    def start(self) -> None:
        """Open the source and begin producing frames."""

    @abstractmethod
    def read(self, timeout: float | None = None) -> FramePacket | None:
        """Return a frame, or ``None`` on timeout/end of stream."""

    @abstractmethod
    def close(self) -> None:
        """Stop capture and release backend resources."""

    def __enter__(self) -> CaptureSource:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _start_once(self) -> bool:
        """Mark a source started and reject attempts to restart a closed source."""

        with self._condition:
            if self._closed:
                raise RuntimeError(f"Cannot restart closed source: {self.description}")
            if self._closing:
                raise RuntimeError(
                    f"Cannot restart source while it is still closing: {self.description}"
                )
            if self._started:
                return False
            self._started = True
            return True

    def _require_started(self) -> None:
        with self._condition:
            if not self._started:
                raise RuntimeError(f"Capture source has not been started: {self.description}")

    def _publish_latest(self, packet: FramePacket) -> None:
        """Publish a packet, replacing an unread packet when necessary."""

        with self._condition:
            if self._closed or self._closing:
                return
            self._frames_read += 1
            if self._latest_packet is not None:
                self._frames_overwritten += 1
            self._latest_packet = packet
            self._condition.notify_all()

    def _read_latest(self, timeout: float | None) -> FramePacket | None:
        self._require_started()
        timeout = _validate_timeout(timeout)
        deadline = None if timeout is None else monotonic() + timeout

        with self._condition:
            while (
                self._latest_packet is None
                and not self._ended
                and not self._closing
            ):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

            if self._latest_packet is None:
                return None

            packet = self._latest_packet
            self._latest_packet = None
            self._frames_delivered += 1
            return packet

    def _record_direct_delivery(self) -> None:
        """Record a successful synchronous file read and delivery."""

        with self._condition:
            self._frames_read += 1
            self._frames_delivered += 1

    def _record_read_failure(self) -> None:
        with self._condition:
            self._read_failures += 1

    def _finish(self, error: str | None = None) -> None:
        with self._condition:
            if error is not None and self._error is None and not self._closed:
                self._error = error
            self._ended = True
            self._condition.notify_all()

    def _mark_closed(self) -> None:
        with self._condition:
            self._closing = False
            self._closed = True
            self._ended = True
            self._latest_packet = None
            self._condition.notify_all()

    def _is_finished(self) -> bool:
        with self._condition:
            return self._ended or self._closing or self._closed

    def _begin_close(self) -> None:
        """Mark shutdown in progress without claiming the worker has exited."""

        with self._condition:
            if self._closed:
                return
            self._closing = True
            self._latest_packet = None
            self._condition.notify_all()

    def _record_close_timeout(self, worker_name: str) -> None:
        """Expose an incomplete close while keeping ``ended`` truthful."""

        with self._condition:
            if self._error is None:
                self._error = (
                    f"Timed out waiting for {worker_name} to stop; "
                    "the capture source is still closing."
                )
            self._condition.notify_all()

    def _complete_close_from_worker(self) -> None:
        """Let a worker finish a close that outlived the caller's join timeout."""

        with self._condition:
            if not self._closing:
                return
            self._closing = False
            self._closed = True
            self._ended = True
            self._latest_packet = None
            self._condition.notify_all()


def _validate_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool):
        raise TypeError("timeout must be a finite non-negative number or None")
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be finite and non-negative or None")
    return timeout
