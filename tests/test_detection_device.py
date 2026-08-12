from __future__ import annotations

import unittest

from detection.devices import available_openvino_devices, selectable_openvino_devices
from detection.openvino_yolo import _device_is_available


class OpenVINODeviceValidationTests(unittest.TestCase):
    def test_launcher_discovery_orders_and_normalizes_devices(self) -> None:
        class FakeCore:
            available_devices = ("gpu.1", "CPU", "npu", "GPU.0", "cpu")

        available = available_openvino_devices(FakeCore)

        self.assertEqual(available, ("CPU", "GPU.0", "GPU.1", "NPU"))
        self.assertEqual(
            selectable_openvino_devices(available),
            ("AUTO", "CPU", "GPU", "NPU", "GPU.0", "GPU.1"),
        )

    def test_physical_device_must_be_reported(self) -> None:
        self.assertTrue(_device_is_available("CPU", ("CPU",)))
        self.assertFalse(_device_is_available("GPU", ("CPU",)))

    def test_indexed_physical_device_matches_generic_request(self) -> None:
        self.assertTrue(_device_is_available("GPU", ("CPU", "GPU.0")))

    def test_virtual_devices_are_validated_during_compilation(self) -> None:
        for requested in ("AUTO", "AUTO:CPU", "MULTI:CPU", "HETERO:CPU", "BATCH:CPU"):
            with self.subTest(requested=requested):
                self.assertTrue(_device_is_available(requested, ("CPU",)))


if __name__ == "__main__":
    unittest.main()
