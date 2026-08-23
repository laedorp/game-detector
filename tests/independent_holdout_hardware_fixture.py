"""Canonical redacted RX 6950 XT invariant fixture for holdout tests."""

from __future__ import annotations

from typing import Any

from utils.independent_holdout_release_contract import (
    HOLDOUT_HARDWARE_GPU_ROLE,
    HOLDOUT_HARDWARE_IDENTITY_KIND,
    HOLDOUT_HARDWARE_IDENTITY_STATUS,
    HOLDOUT_HARDWARE_PRODUCT_NAME,
    HOLDOUT_HARDWARE_VENDOR_ID,
)
from utils.release_model_contract import canonical_hash


def valid_rx6950_holdout_hardware_identity(
    *,
    adapter_index: int = 0,
    qualification_run_id: int = 1234,
    public_receipt_sha256: str = "a" * 64,
    device_id: str = "0x73bf",
    driver_version: str = "32.0.21001.9024",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 1,
        "kind": HOLDOUT_HARDWARE_IDENTITY_KIND,
        "status": HOLDOUT_HARDWARE_IDENTITY_STATUS,
        "gpu_role": HOLDOUT_HARDWARE_GPU_ROLE,
        "product_name": HOLDOUT_HARDWARE_PRODUCT_NAME,
        "directml_device": f"DML:{adapter_index}",
        "directml_adapter_index": adapter_index,
        "vendor_id": HOLDOUT_HARDWARE_VENDOR_ID,
        "device_id": device_id,
        "driver_version": driver_version,
        "physical_evidence": {
            "qualification_run_id": qualification_run_id,
            "adapter_index": adapter_index,
            "public_receipt_sha256": public_receipt_sha256,
        },
        "inventory": {
            "dxgi_exact_match_count": 1,
            "wmi_exact_match_count": 1,
            "directx_registry_exact_match_count": 1,
            "dedicated_vram_bytes": 16 * 1024**3,
            "adapter_luid_present_and_correlated": True,
        },
        "cross_checks": {
            "selected_index_matches_physical_qualification": True,
            "product_name_matches_across_sources": True,
            "vendor_device_matches_across_sources": True,
            "driver_version_matches_physical_qualification": True,
            "dml_execution_provider_available": True,
        },
        "privacy": {
            "redacted": True,
            "adapter_luid_disclosed": False,
            "pnp_device_id_disclosed": False,
            "local_paths_disclosed": False,
        },
    }
    body["content_sha256"] = canonical_hash(body)
    return body
