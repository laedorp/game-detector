#!/usr/bin/env python3
"""Fail closed unless the sealed holdout runs on the qualified RX 6950 XT."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from utils.independent_holdout_release_contract import (
    release_environment_record,
    sha256_file,
)
from utils.release_model_contract import canonical_hash, canonical_json_bytes


KIND = "proaim-independent-holdout-directml-adapter-invariant"
STATUS = "verified_before_sealed_member_access"
GPU_ROLE = "amd_rx_6950_xt"
PRODUCT_NAME = "AMD Radeon RX 6950 XT"
VENDOR_ID = "0x1002"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

WMI_QUERY = (
    "Get-CimInstance Win32_VideoController | ForEach-Object { "
    "[ordered]@{Name=[string]$_.Name;PNPDeviceID=[string]$_.PNPDeviceID;"
    "DriverVersion=[string]$_.DriverVersion} } | ConvertTo-Json -Compress"
)
DIRECTX_QUERY = (
    "$Rows=@(); Get-ChildItem -LiteralPath 'HKLM:\\SOFTWARE\\Microsoft\\DirectX' "
    "-ErrorAction Stop | ForEach-Object { $R=Get-ItemProperty -LiteralPath $_.PSPath "
    "-ErrorAction SilentlyContinue; if ($null -ne $R -and $null -ne $R.AdapterLuid) { "
    "$U=[uint64]$R.AdapterLuid; $Rows += [ordered]@{Description=[string]$R.Description;"
    "VendorId=('0x{0:x4}' -f [uint32]$R.VendorId);"
    "DeviceId=('0x{0:x4}' -f [uint32]$R.DeviceId);"
    "DriverVersion=[string]$R.DriverVersion;"
    "AdapterLuid=('0x{0:x8}_0x{1:x8}' -f [uint32](($U -shr 32) -band 0xffffffff),"
    "[uint32]($U -band 0xffffffff))} } }; $Rows | ConvertTo-Json -Compress"
)


class HoldoutAdapterError(RuntimeError):
    """Raised when the pre-access physical GPU identity is not exact."""


def _duplicates_rejected(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HoldoutAdapterError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _json_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicates_rejected,
            parse_constant=lambda value: (_ for _ in ()).throw(
                HoldoutAdapterError(f"non-finite JSON value: {value}")
            ),
        )
    except HoldoutAdapterError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HoldoutAdapterError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise HoldoutAdapterError(f"{description} must be a JSON object")
    return value, payload


def _normalize_product(value: object) -> str:
    return " ".join(str(value or "").strip().split()).upper()


def _hex(value: object) -> str:
    text = str(value or "").strip().lower().removeprefix("0x")
    if re.fullmatch(r"[0-9a-f]{4,8}", text) is None:
        raise HoldoutAdapterError("GPU PCI identifier is invalid")
    return "0x" + text


def _line(value: object, description: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 160 or any(character in text for character in "\r\n\0"):
        raise HoldoutAdapterError(f"{description} is invalid")
    return text


def _powershell_json(query: str) -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HoldoutAdapterError("Windows hardware inventory query failed") from exc
    if completed.returncode != 0:
        raise HoldoutAdapterError("Windows hardware inventory query returned failure")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HoldoutAdapterError("Windows hardware inventory returned invalid JSON") from exc
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(item, dict) for item in values):
        raise HoldoutAdapterError("Windows hardware inventory is empty or invalid")
    return values


def _wmi_identifier(record: Mapping[str, Any]) -> tuple[str, str]:
    pnp = str(record.get("PNPDeviceID") or "")
    vendor = re.search(r"(?:^|[\\&])VEN_([0-9A-Fa-f]{4})(?:[\\&]|$)", pnp)
    device = re.search(r"(?:^|[\\&])DEV_([0-9A-Fa-f]{4})(?:[\\&]|$)", pnp)
    if vendor is None or device is None:
        raise HoldoutAdapterError("WMI GPU omitted its PCI vendor/device identity")
    return "0x" + vendor.group(1).lower(), "0x" + device.group(1).lower()


def verify_adapter_identity(
    *,
    adapter_index: int,
    physical_receipt: Path,
    physical_receipt_sha256: str,
    dependency_manifest: Path,
    project_root: Path,
    adapter_factory: Callable[[], Sequence[Any]] | None = None,
    powershell_runner: Callable[[str], list[dict[str, Any]]] | None = None,
    available_providers: Sequence[str] | None = None,
) -> dict[str, Any]:
    if isinstance(adapter_index, bool) or not isinstance(adapter_index, int) or adapter_index < 0:
        raise HoldoutAdapterError("DirectML adapter index must be a non-negative integer")
    expected_receipt_sha = str(physical_receipt_sha256 or "").strip().lower()
    if SHA256_RE.fullmatch(expected_receipt_sha) is None:
        raise HoldoutAdapterError("physical receipt SHA-256 is invalid")
    receipt, receipt_payload = _json_object(physical_receipt, "physical qualification receipt")
    if sha256(receipt_payload).hexdigest() != expected_receipt_sha:
        raise HoldoutAdapterError("physical qualification receipt SHA-256 differs")
    physical = receipt.get("physical_gpu")
    qualification_run = receipt.get("qualification_run")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "physically_qualified_directml_release_candidate"
        or receipt.get("gpu_role") != GPU_ROLE
        or not isinstance(physical, Mapping)
        or physical.get("directml_adapter_index") != adapter_index
        or _normalize_product(physical.get("product_name"))
        != _normalize_product(PRODUCT_NAME)
        or _hex(physical.get("vendor_id")) != VENDOR_ID
        or not isinstance(qualification_run, Mapping)
        or qualification_run.get("attempt") != 1
        or isinstance(qualification_run.get("id"), bool)
        or not isinstance(qualification_run.get("id"), int)
        or qualification_run.get("id") <= 0
    ):
        raise HoldoutAdapterError(
            "physical receipt does not bind the exact first-attempt RX 6950 XT qualification"
        )
    expected_device = _hex(physical.get("device_id"))
    expected_driver = _line(physical.get("driver_version"), "physical driver version")

    if adapter_factory is None:
        from detection.hardware import scan_windows_directml_adapters

        adapter_factory = scan_windows_directml_adapters
    adapters = list(adapter_factory())
    matches = [item for item in adapters if getattr(item, "index", None) == adapter_index]
    if len(matches) != 1:
        raise HoldoutAdapterError("DXGI inventory does not contain one exact requested index")
    selected = matches[0]
    selected_luid = str(getattr(selected, "adapter_luid", "")).strip().lower()
    if (
        _normalize_product(getattr(selected, "name", ""))
        != _normalize_product(PRODUCT_NAME)
        or _hex(getattr(selected, "vendor_id", "")) != VENDOR_ID
        or _hex(getattr(selected, "device_id", "")) != expected_device
        or int(getattr(selected, "dedicated_vram", 0)) <= 0
        or re.fullmatch(r"0x[0-9a-f]{8}_0x[0-9a-f]{8}", selected_luid) is None
    ):
        raise HoldoutAdapterError("selected DXGI adapter is not the exact qualified RX 6950 XT")

    runner = powershell_runner or _powershell_json
    wmi_matches = []
    for item in runner(WMI_QUERY):
        if _normalize_product(item.get("Name")) != _normalize_product(PRODUCT_NAME):
            continue
        vendor, device = _wmi_identifier(item)
        if vendor == VENDOR_ID and device == expected_device:
            wmi_matches.append(item)
    if len(wmi_matches) != 1 or _line(
        wmi_matches[0].get("DriverVersion"), "WMI driver version"
    ) != expected_driver:
        raise HoldoutAdapterError("WMI identity/driver differs from qualified RX 6950 XT")

    registry_matches = []
    for item in runner(DIRECTX_QUERY):
        if (
            _normalize_product(item.get("Description"))
            == _normalize_product(PRODUCT_NAME)
            and _hex(item.get("VendorId")) == VENDOR_ID
            and _hex(item.get("DeviceId")) == expected_device
            and str(item.get("AdapterLuid") or "").strip().lower() == selected_luid
        ):
            registry_matches.append(item)
    if len(registry_matches) != 1 or _line(
        registry_matches[0].get("DriverVersion"), "DirectX driver version"
    ) != expected_driver:
        raise HoldoutAdapterError(
            "DirectX registry LUID identity/driver differs from DXGI and qualification"
        )

    if available_providers is None:
        try:
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
        except Exception as exc:
            raise HoldoutAdapterError("cannot inspect ONNX Runtime providers") from exc
    if "DmlExecutionProvider" not in tuple(str(item) for item in available_providers):
        raise HoldoutAdapterError("DmlExecutionProvider is unavailable")

    manifest, manifest_payload = _json_object(
        dependency_manifest, "exact dependency manifest"
    )
    environment = release_environment_record(
        manifest,
        dependency_manifest_sha256=sha256(manifest_payload).hexdigest(),
        project_root=project_root,
    )
    if sha256_file(Path(sys.executable).resolve()) != environment["python_executable_sha256"]:
        raise HoldoutAdapterError(
            "adapter checker interpreter differs from the verified dependency manifest"
        )
    body: dict[str, Any] = {
        "schema_version": 1,
        "kind": KIND,
        "status": STATUS,
        "gpu_role": GPU_ROLE,
        "product_name": PRODUCT_NAME,
        "directml_device": f"DML:{adapter_index}",
        "directml_adapter_index": adapter_index,
        "vendor_id": VENDOR_ID,
        "device_id": expected_device,
        "driver_version": expected_driver,
        "physical_evidence": {
            "qualification_run_id": qualification_run["id"],
            "adapter_index": adapter_index,
            "public_receipt_sha256": expected_receipt_sha,
        },
        "inventory": {
            "dxgi_exact_match_count": 1,
            "wmi_exact_match_count": 1,
            "directx_registry_exact_match_count": 1,
            "dedicated_vram_bytes": int(getattr(selected, "dedicated_vram")),
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


def _write_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise HoldoutAdapterError("adapter invariant output must be a new local file")
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise HoldoutAdapterError("adapter invariant temporary output already exists")
    try:
        temporary.write_bytes(canonical_json_bytes(value))
        os.replace(temporary, target)
    except OSError as exc:
        raise HoldoutAdapterError("cannot write adapter invariant output") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-index", required=True, type=int)
    parser.add_argument("--physical-receipt", required=True, type=Path)
    parser.add_argument("--physical-receipt-sha256", required=True)
    parser.add_argument("--dependency-manifest", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_adapter_identity(
            adapter_index=args.adapter_index,
            physical_receipt=args.physical_receipt,
            physical_receipt_sha256=args.physical_receipt_sha256,
            dependency_manifest=args.dependency_manifest,
            project_root=args.project_root,
        )
        _write_no_replace(args.output, result)
    except (HoldoutAdapterError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": STATUS, "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
