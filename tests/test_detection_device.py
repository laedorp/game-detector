from __future__ import annotations

import unittest

from detection.openvino_yolo import _device_is_available


class OpenVINODeviceValidationTests(unittest.TestCase):
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
