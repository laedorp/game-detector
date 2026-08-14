from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
import sys

from detection.hardware import DirectMLAdapter
from scripts.verify_windows_holdout_adapter import (
    DIRECTX_QUERY,
    HoldoutAdapterError,
    WMI_QUERY,
    verify_adapter_identity,
)
from tests.independent_holdout_environment_fixture import (
    valid_windows_directml_dependency_manifest,
)
from utils.independent_holdout_release_contract import (
    sha256_file,
    validate_holdout_hardware_identity,
)
from utils.release_model_contract import canonical_json_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HoldoutDirectMLAdapterTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, str, Path]:
        receipt = {
            "schema_version": 1,
            "status": "physically_qualified_directml_release_candidate",
            "repository": "owner/repository",
            "tag": "v1.2.3",
            "tag_commit": "a" * 40,
            "gpu_role": "amd_rx_6950_xt",
            "physical_gpu": {
                "product_name": "AMD Radeon RX 6950 XT",
                "directml_adapter_index": 2,
                "vendor_id": "0x1002",
                "device_id": "0x73bf",
                "driver_version": "32.0.21001.9024",
            },
            "candidate": {},
            "qualification_metrics": {},
            "telemetry": {},
            "qualification_run": {"id": 1234, "attempt": 1},
            "privacy": {"redacted": True},
        }
        receipt_path = root / "amd-receipt.json"
        receipt_payload = canonical_json_bytes(receipt)
        receipt_path.write_bytes(receipt_payload)
        manifest_path = root / "dependency-manifest.json"
        manifest = valid_windows_directml_dependency_manifest(PROJECT_ROOT)
        manifest["python"]["executable_sha256"] = sha256_file(
            Path(sys.executable).resolve()
        )
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        return receipt_path, sha256(receipt_payload).hexdigest(), manifest_path

    @staticmethod
    def _powershell(query: str) -> list[dict]:
        if query == WMI_QUERY:
            return [
                {
                    "Name": "AMD Radeon RX 6950 XT",
                    "PNPDeviceID": "PCI\\VEN_1002&DEV_73BF&SUBSYS_00000000",
                    "DriverVersion": "32.0.21001.9024",
                }
            ]
        if query == DIRECTX_QUERY:
            return [
                {
                    "Description": "AMD Radeon RX 6950 XT",
                    "VendorId": "0x1002",
                    "DeviceId": "0x73bf",
                    "DriverVersion": "32.0.21001.9024",
                    "AdapterLuid": "0x00000001_0x00000002",
                }
            ]
        raise AssertionError(query)

    def test_exact_rx6950_identity_is_stable_redacted_and_shared_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt, receipt_sha, manifest = self._fixture(root)
            record = verify_adapter_identity(
                adapter_index=2,
                physical_receipt=receipt,
                physical_receipt_sha256=receipt_sha,
                dependency_manifest=manifest,
                project_root=PROJECT_ROOT,
                adapter_factory=lambda: (
                    DirectMLAdapter(
                        index=2,
                        name="AMD Radeon RX 6950 XT",
                        vendor_id="1002",
                        device_id="73bf",
                        dedicated_vram=16 * 1024**3,
                        adapter_luid="0x00000001_0x00000002",
                    ),
                ),
                powershell_runner=self._powershell,
                available_providers=("DmlExecutionProvider", "CPUExecutionProvider"),
            )

        self.assertEqual(validate_holdout_hardware_identity(record), record)
        serialized = json.dumps(record, sort_keys=True).lower()
        self.assertNotIn("adapterluid", serialized)
        self.assertNotIn("pnpdeviceid", serialized)
        self.assertNotIn("users\\", serialized)

    def test_wrong_receipt_adapter_or_driver_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt, receipt_sha, manifest = self._fixture(root)
            adapter = DirectMLAdapter(
                index=2,
                name="AMD Radeon RX 6950 XT",
                vendor_id="1002",
                device_id="73bf",
                dedicated_vram=16 * 1024**3,
                adapter_luid="0x00000001_0x00000002",
            )
            common = dict(
                physical_receipt=receipt,
                physical_receipt_sha256=receipt_sha,
                dependency_manifest=manifest,
                project_root=PROJECT_ROOT,
                adapter_factory=lambda: (adapter,),
                powershell_runner=self._powershell,
                available_providers=("DmlExecutionProvider",),
            )
            with self.assertRaisesRegex(HoldoutAdapterError, "physical receipt"):
                verify_adapter_identity(adapter_index=1, **common)

            def wrong_driver(query: str) -> list[dict]:
                values = [dict(item) for item in self._powershell(query)]
                values[0]["DriverVersion"] = "wrong"
                return values

            common["powershell_runner"] = wrong_driver
            with self.assertRaisesRegex(HoldoutAdapterError, "driver"):
                verify_adapter_identity(adapter_index=2, **common)


if __name__ == "__main__":
    unittest.main()
