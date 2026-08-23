from __future__ import annotations

from queue import Queue
from threading import Event
from time import monotonic, perf_counter, sleep
from types import SimpleNamespace
import unittest

import numpy as np

from capture.base import CaptureSource, FramePacket
from capture.desktop_source import DesktopCaptureSource
from capture.dxcam_source import (
    DXCamCaptureSource,
    _camera_global_bounds,
    _output_records,
    _relative_region,
)
from main import _build_capture


_STOP = object()


class _FakeCamera:
    def __init__(
        self,
        output_bounds: tuple[int, int, int, int],
        frames: list[tuple[np.ndarray, float]] | None = None,
    ) -> None:
        self.output_bounds = output_bounds
        self.width = output_bounds[2] - output_bounds[0]
        self.height = output_bounds[3] - output_bounds[1]
        self.frames: Queue[object] = Queue()
        for frame in frames or []:
            self.frames.put(frame)
        self.is_capturing = False
        self.start_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []
        self.release_count = 0
        self.frames_returned = 0
        self.drained = Event()

    def start(self, **kwargs) -> None:
        self.start_calls.append(dict(kwargs))
        self.is_capturing = True

    def get_latest_frame(self, **kwargs):
        self.get_calls.append(dict(kwargs))
        value = self.frames.get(timeout=2.0)
        if value is _STOP:
            return None
        self.frames_returned += 1
        if self.frames.empty():
            self.drained.set()
        return value

    def stop(self) -> None:
        was_capturing = self.is_capturing
        self.is_capturing = False
        if was_capturing:
            self.frames.put(_STOP)

    def release(self) -> None:
        self.release_count += 1
        self.stop()


class _FakeDXCamModule:
    def __init__(self, cameras: dict[tuple[int, int], _FakeCamera]) -> None:
        self.cameras = cameras
        self.create_calls: list[dict[str, object]] = []

    def output_info(self) -> str:
        return "\n".join(
            f"Device[{device}] Output[{output}]: fake"
            for device, output in self.cameras
        )

    def create(self, **kwargs):
        self.create_calls.append(dict(kwargs))
        key = (int(kwargs["device_idx"]), int(kwargs["output_idx"]))
        return self.cameras[key]


class _StubCapture(CaptureSource):
    def __init__(self, backend: str, *, failure: str | None = None, **_kwargs) -> None:
        super().__init__(f"stub {backend}")
        self.backend = backend
        self.failure = failure
        self.closed = False
        self.packet = FramePacket(
            image=np.zeros((2, 3, 3), dtype=np.uint8),
            sequence=0,
            read_started_ns=1,
            read_completed_ns=2,
        )

    @property
    def actual_settings(self):
        return {"backend": self.backend}

    def start(self) -> None:
        self._start_once()
        if self.failure is not None:
            self._finish(self.failure)
            raise RuntimeError(self.failure)

    def read(self, timeout=None):
        del timeout
        self._require_started()
        return self.packet

    def peek_latest(self):
        self._require_started()
        return self.packet

    def close(self) -> None:
        self.closed = True
        self._mark_closed()


class DXCamCaptureTests(unittest.TestCase):
    def test_region_uses_global_coordinates_and_selects_containing_output(self) -> None:
        first = _FakeCamera((0, 0, 1920, 1080))
        timestamp = perf_counter() - 0.002
        expected = np.full((480, 640, 3), 37, dtype=np.uint8)
        second = _FakeCamera(
            (1920, -100, 3840, 980),
            [(expected, timestamp)],
        )
        module = _FakeDXCamModule({(0, 0): first, (1, 0): second})
        source = DXCamCaptureSource(
            region=(2000, 20, 640, 480),
            fps=119.5,
            _dxcam_module=module,
        )

        try:
            source.start()
            packet = source.read(timeout=1.0)
            self.assertIsNotNone(packet)
            assert packet is not None
            self.assertTrue(np.shares_memory(packet.image, expected))
            self.assertEqual(packet.image[0, 0, 0], 37)
            self.assertEqual(packet.read_started_ns, round(timestamp * 1_000_000_000))
            self.assertLessEqual(packet.read_started_ns, packet.read_completed_ns)
            self.assertEqual(source.stats.frames_read, 1)
            self.assertEqual(source.stats.frames_delivered, 1)
            self.assertEqual(source.actual_settings["backend"], "dxcam-dxgi")
            self.assertEqual(source.actual_settings["device_index"], 1)
            self.assertEqual(source.actual_settings["left"], 2000)
            self.assertEqual(source.actual_settings["top"], 20)
            self.assertEqual(
                second.start_calls,
                [
                    {
                        "region": (80, 120, 720, 600),
                        "target_fps": 120,
                        "video_mode": False,
                    }
                ],
            )
            self.assertEqual(
                second.get_calls[0],
                {"copy": True, "with_timestamp": True},
            )
            self.assertEqual(module.create_calls[0]["max_buffer_len"], 2)
            self.assertEqual(module.create_calls[0]["output_color"], "BGR")
            self.assertEqual(module.create_calls[0]["backend"], "dxgi")
            self.assertGreaterEqual(first.release_count, 1)
        finally:
            source.close()

    def test_latest_mailbox_overwrites_older_dxcam_frames(self) -> None:
        now = perf_counter()
        frames = [
            (np.full((4, 5, 3), value, dtype=np.uint8), now + value / 1e6)
            for value in (1, 2, 3)
        ]
        camera = _FakeCamera((0, 0, 5, 4), frames)
        source = DXCamCaptureSource(
            monitor=1,
            fps=240,
            _dxcam_module=_FakeDXCamModule({(0, 0): camera}),
            _monitor_bounds_provider=lambda _monitor: (0, 0, 5, 4),
        )

        try:
            source.start()
            self.assertTrue(camera.drained.wait(1.0))
            deadline = monotonic() + 1.0
            while source.stats.frames_read < 3 and monotonic() < deadline:
                camera.drained.wait(0.01)
            stats_before_peek = source.stats
            peeked = source.peek_latest()
            self.assertIsNotNone(peeked)
            assert peeked is not None
            self.assertEqual(peeked.sequence, 2)
            self.assertEqual(int(peeked.image[0, 0, 0]), 3)
            self.assertEqual(source.stats, stats_before_peek)
            packet = source.read(timeout=1.0)
            self.assertIsNotNone(packet)
            assert packet is not None
            self.assertEqual(packet.sequence, 2)
            self.assertEqual(int(packet.image[0, 0, 0]), 3)
            self.assertEqual(source.stats.frames_read, 3)
            self.assertEqual(source.stats.frames_overwritten, 2)
            self.assertEqual(source.stats.frames_delivered, 1)
        finally:
            source.close()

    def test_presentation_after_wait_start_is_used_as_frame_age_origin(self) -> None:
        # A newly presented desktop frame commonly arrives after the consumer
        # has begun blocking in get_latest_frame(). The presentation timestamp,
        # not that earlier wait start, is the meaningful freshness origin.
        presented = perf_counter() + 0.01

        class DelayedCamera(_FakeCamera):
            def get_latest_frame(self, **kwargs):
                sleep(0.02)
                return super().get_latest_frame(**kwargs)

        camera = DelayedCamera(
            (0, 0, 4, 3),
            [(np.zeros((3, 4, 3), dtype=np.uint8), presented)],
        )
        source = DXCamCaptureSource(
            monitor=1,
            _dxcam_module=_FakeDXCamModule({(0, 0): camera}),
            _monitor_bounds_provider=lambda _monitor: (0, 0, 4, 3),
        )
        try:
            source.start()
            packet = source.read(timeout=1.0)
            assert packet is not None
            self.assertLessEqual(packet.read_started_ns, packet.read_completed_ns)
            self.assertEqual(
                packet.read_started_ns,
                round(presented * 1_000_000_000),
            )
        finally:
            source.close()

    def test_strided_bgr_frame_is_made_contiguous(self) -> None:
        base = np.zeros((3, 8, 3), dtype=np.uint8)
        strided = base[:, ::2, :]
        self.assertFalse(strided.flags.c_contiguous)
        camera = _FakeCamera((0, 0, 4, 3), [(strided, perf_counter())])
        source = DXCamCaptureSource(
            monitor=1,
            _dxcam_module=_FakeDXCamModule({(0, 0): camera}),
            _monitor_bounds_provider=lambda _monitor: (0, 0, 4, 3),
        )
        try:
            source.start()
            packet = source.read(timeout=1.0)
            assert packet is not None
            self.assertTrue(packet.image.flags.c_contiguous)
            self.assertFalse(np.shares_memory(packet.image, strided))
        finally:
            source.close()

    def test_close_timeout_is_bounded_when_native_stop_stalls(self) -> None:
        unblock_stop = Event()

        class StalledStopCamera(_FakeCamera):
            def stop(self) -> None:
                unblock_stop.wait(2.0)
                super().stop()

        camera = StalledStopCamera(
            (0, 0, 4, 3),
            [(np.zeros((3, 4, 3), dtype=np.uint8), perf_counter())],
        )
        source = DXCamCaptureSource(
            monitor=1,
            close_timeout=0.03,
            _dxcam_module=_FakeDXCamModule({(0, 0): camera}),
            _monitor_bounds_provider=lambda _monitor: (0, 0, 4, 3),
        )
        source.start()
        self.assertIsNotNone(source.read(timeout=1.0))

        started = monotonic()
        source.close()
        elapsed = monotonic() - started
        self.assertLess(elapsed, 0.25)
        self.assertFalse(source.ended)
        self.assertIn("still closing", source.error or "")
        # release() must not race stop() and trigger a second native wait.
        self.assertEqual(camera.release_count, 0)

        unblock_stop.set()
        deadline = monotonic() + 1.0
        while not source.ended and monotonic() < deadline:
            sleep(0.01)
        self.assertTrue(source.ended)
        self.assertGreaterEqual(camera.release_count, 1)

    def test_invalid_frame_records_one_read_failure_and_ends(self) -> None:
        camera = _FakeCamera(
            (0, 0, 4, 3),
            [(np.zeros((3, 4), dtype=np.uint8), perf_counter())],
        )
        source = DXCamCaptureSource(
            monitor=1,
            startup_timeout=1.0,
            _dxcam_module=_FakeDXCamModule({(0, 0): camera}),
            _monitor_bounds_provider=lambda _monitor: (0, 0, 4, 3),
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected BGR frame shape"):
            source.start()
        self.assertEqual(source.stats.read_failures, 1)
        self.assertTrue(source.ended)
        source.close()

    def test_rectangle_spanning_outputs_fails_instead_of_capturing_wrong_screen(self) -> None:
        module = _FakeDXCamModule(
            {
                (0, 0): _FakeCamera((0, 0, 100, 100)),
                (0, 1): _FakeCamera((100, 0, 200, 100)),
            }
        )
        source = DXCamCaptureSource(
            region=(50, 0, 100, 100),
            _dxcam_module=module,
        )
        with self.assertRaisesRegex(RuntimeError, "not contained within one DXGI output"):
            source.start()
        source.close()

    def test_geometry_helpers_keep_dxcam_region_right_bottom_semantics(self) -> None:
        self.assertEqual(
            _relative_region((110, -40, 310, 160), (100, -50, 500, 350)),
            (10, 10, 210, 210),
        )
        camera = SimpleNamespace(output_bounds=(-10, -20, 990, 780))
        self.assertEqual(_camera_global_bounds(camera), (-10, -20, 990, 780))
        module = SimpleNamespace(
            output_info=lambda: (
                "Device[0] Output[1]: fake\nDevice[0] Output[1]: duplicate\n"
                "Device[2] Output[0]: fake"
            )
        )
        self.assertEqual(_output_records(module), [(0, 1), (2, 0)])


class PreferredDesktopCaptureTests(unittest.TestCase):
    def test_windows_falls_back_to_mss_when_dxcam_cannot_start(self) -> None:
        created: list[_StubCapture] = []

        def dxcam_factory(**kwargs):
            created.append(_StubCapture("dxcam", failure="DXGI unavailable", **kwargs))
            return created[-1]

        def mss_factory(**kwargs):
            created.append(_StubCapture("mss", **kwargs))
            return created[-1]

        source = DesktopCaptureSource(
            monitor=2,
            fps=144,
            _platform="win32",
            _dxcam_factory=dxcam_factory,
            _mss_factory=mss_factory,
        )
        source.start()
        try:
            self.assertEqual(source.actual_settings["backend"], "mss")
            self.assertEqual(source.actual_settings["preferred_backend"], "dxcam-dxgi")
            self.assertIn("DXGI unavailable", source.actual_settings["fallback_reason"])
            self.assertIs(source.read(timeout=0), created[1].packet)
            self.assertIs(source.peek_latest(), created[1].packet)
            self.assertTrue(created[0].closed)
        finally:
            source.close()

    def test_non_windows_uses_mss_without_constructing_dxcam(self) -> None:
        def forbidden(**_kwargs):
            raise AssertionError("DXcam must not be constructed on Linux")

        source = DesktopCaptureSource(
            _platform="linux",
            _dxcam_factory=forbidden,
            _mss_factory=lambda **kwargs: _StubCapture("mss", **kwargs),
        )
        source.start()
        try:
            self.assertEqual(source.actual_settings, {"backend": "mss"})
        finally:
            source.close()

    def test_if_both_backends_fail_the_error_names_both(self) -> None:
        source = DesktopCaptureSource(
            _platform="win32",
            _dxcam_factory=lambda **kwargs: _StubCapture(
                "dxcam", failure="DX failure", **kwargs
            ),
            _mss_factory=lambda **kwargs: _StubCapture(
                "mss", failure="MSS failure", **kwargs
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "DX failure.*MSS failure"):
            source.start()
        self.assertIn("MSS failure", source.error or "")
        source.close()

    def test_build_capture_uses_platform_preferred_screen_source(self) -> None:
        config = SimpleNamespace(
            source=SimpleNamespace(kind="screen", value=None),
            screen_monitor=1,
            screen_region=None,
            screen_fps=120.0,
        )
        source = _build_capture(config)
        self.assertIsInstance(source, DesktopCaptureSource)


if __name__ == "__main__":
    unittest.main()
