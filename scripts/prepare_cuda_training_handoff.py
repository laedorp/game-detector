"""Fail-closed preflight and explicit launcher for fresh v9 CUDA training.

This helper never downloads dependencies or data, never resumes a checkpoint,
and emits a command by default.  ``--execute`` is deliberately protected by an
exact run-name acknowledgement so an inspection command cannot start training.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path, PurePath
import platform
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

try:
    from scripts.fort_dataset_contract import (
        DatasetContractError,
        verify_dataset_contract,
        verify_grouped_dataset_metadata,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from fort_dataset_contract import (
        DatasetContractError,
        verify_dataset_contract,
        verify_grouped_dataset_metadata,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "datasets" / "fort_cuh_player_grouped_v9" / "fort_cuh_grouped.yaml"
DATASET_MANIFEST = DATASET_YAML.parent / "manifest.json"
DATASET_CONTRACT_SCRIPT = PROJECT_ROOT / "scripts" / "fort_dataset_contract.py"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "train_fort_model.py"
TRAINING_PROJECT = PROJECT_ROOT / "runs" / "fort_cuh"

AUDITED_DATASET_MANIFEST_SHA256 = (
    "f09ad355ead4a4dd4504f550f0b390786950faa6702fcd065c6849d765dbdffb"
)
AUDITED_DATASET_CONTENT_SHA256 = (
    "b2979f0ea75e5245944076aab51636ab7173a6e5ef1108d0cfb2e3f0549e0255"
)
AUDITED_DATASET_YAML_SHA256 = (
    "b6868017e5c28f9e10a7c4e8450bc48eb2fcff2b354a9b3018460e3594ffaa62"
)
AUDITED_TRAINING_SCRIPT_SHA256 = (
    "93a57b8b00373dc3be5387002fd37c7226e6966fa26b39431a5c7210e07bdb22"
)
AUDITED_DATASET_CONTRACT_SCRIPT_SHA256 = (
    "804b980e39693c94a53766384d12250a698e9fa2840869d7a9334e228740afdd"
)

EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5060 Laptop GPU"
EXPECTED_DEVICE_CAPABILITY = (12, 0)
EXPECTED_CUDA_RUNTIME = "13.0"
MINIMUM_TOTAL_VRAM_BYTES = int(7.5 * 1024**3)
MINIMUM_FREE_VRAM_BYTES = 6 * 1024**3
EXPECTED_TRAINING_SMOKE: Mapping[str, str] = {
    "fp32_convolution_forward": "passed",
    "fp32_convolution_backward": "passed",
    "optimizer_step": "passed",
    "torchvision_cuda_nms": "passed",
    "selected_yolo_raw_head_fp32_forward": "passed",
    "selected_yolo_raw_head_fp32_backward": "passed",
    "selected_yolo_adamw_step": "passed",
}

# These are the audited training environment's direct numerical dependencies.
# The CPU wheel suffixes are intentionally replaced by matching CUDA 13 wheels.
EXPECTED_PACKAGE_VERSIONS: Mapping[str, str] = {
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
    "ultralytics": "8.4.116",
    "numpy": "2.4.4",
    "opencv-python": "5.0.0.93",
    "PyYAML": "6.0.3",
    "pillow": "12.2.0",
    "matplotlib": "3.11.1",
    "psutil": "7.2.2",
    "scipy": "1.18.0",
    "ultralytics-thop": "2.1.6",
}


@dataclass(frozen=True, slots=True)
class ModelContract:
    filename: str
    sha256: str
    default_run_name: str
    training_batch: int


MODEL_CONTRACTS: Mapping[str, ModelContract] = {
    "n": ModelContract(
        filename="yolo26n.pt",
        sha256="9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        default_run_name="yolo26n_640_player_grouped_v9_rtx5060_fresh",
        training_batch=8,
    ),
    "s": ModelContract(
        filename="yolo26s.pt",
        sha256="646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
        default_run_name="yolo26s_640_player_grouped_v9_rtx5060_fresh",
        training_batch=4,
    ),
}


class CudaTrainingHandoffError(RuntimeError):
    """Raised before any training subprocess is allowed to start."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, description: str) -> Path:
    _verify_directory_chain(path.parent, description)
    if not path.is_file() or _is_linklike(path):
        raise CudaTrainingHandoffError(f"{description} is missing or unsafe: {path}")
    return path


def _require_hash(path: Path, expected: str, description: str) -> str:
    _require_regular_file(path, description)
    actual = _sha256_file(path)
    if actual != expected:
        raise CudaTrainingHandoffError(
            f"{description} SHA-256 mismatch: {actual} != {expected}"
        )
    return actual


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CudaTrainingHandoffError(f"invalid audited dataset manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise CudaTrainingHandoffError("audited dataset manifest root is not an object")
    return value


def verify_audited_dataset() -> dict[str, Any]:
    """Verify metadata plus every image/label against the exact v9 contract."""

    manifest_sha = _require_hash(
        DATASET_MANIFEST,
        AUDITED_DATASET_MANIFEST_SHA256,
        "audited v9 dataset manifest",
    )
    yaml_sha = _require_hash(
        DATASET_YAML,
        AUDITED_DATASET_YAML_SHA256,
        "audited v9 dataset YAML",
    )
    manifest = _load_manifest(DATASET_MANIFEST)
    expected_contract = manifest.get("dataset_contract")
    if not isinstance(expected_contract, dict):
        raise CudaTrainingHandoffError("audited dataset manifest has no exact-file contract")
    if expected_contract.get("content_sha256") != AUDITED_DATASET_CONTENT_SHA256:
        raise CudaTrainingHandoffError("audited dataset manifest has the wrong content hash")
    try:
        verify_grouped_dataset_metadata(DATASET_YAML)
        actual_contract = verify_dataset_contract(DATASET_YAML.parent, expected_contract)
    except DatasetContractError as exc:
        raise CudaTrainingHandoffError(f"audited v9 dataset verification failed: {exc}") from exc
    if actual_contract.get("content_sha256") != AUDITED_DATASET_CONTENT_SHA256:
        raise CudaTrainingHandoffError("audited v9 dataset content SHA-256 mismatch")
    split_summary = {
        name: {
            "images": actual_contract["splits"][name]["images"],
            "boxes": actual_contract["splits"][name]["boxes"],
        }
        for name in ("train", "valid", "test")
    }
    return {
        "yaml": str(DATASET_YAML),
        "yaml_sha256": yaml_sha,
        "manifest": str(DATASET_MANIFEST),
        "manifest_sha256": manifest_sha,
        "content_sha256": actual_contract["content_sha256"],
        "splits": split_summary,
    }


def verify_packages(
    version_getter: Callable[[str], str] = metadata.version,
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package, expected in EXPECTED_PACKAGE_VERSIONS.items():
        try:
            actual = version_getter(package)
        except metadata.PackageNotFoundError as exc:
            raise CudaTrainingHandoffError(
                f"required training package is not installed: {package}=={expected}"
            ) from exc
        if actual != expected:
            raise CudaTrainingHandoffError(
                f"training package mismatch for {package}: {actual} != {expected}"
            )
        versions[package] = actual
    return versions


def _cuda_architecture_supported(
    architectures: Sequence[str], capability: tuple[int, int]
) -> bool:
    suffix = f"{capability[0]}{capability[1]}"
    return f"sm_{suffix}" in architectures or f"compute_{suffix}" in architectures


def verify_cuda_device(torch_module: Any, device_index: int) -> dict[str, Any]:
    if getattr(getattr(torch_module, "version", None), "hip", None) is not None:
        raise CudaTrainingHandoffError("ROCm/HIP PyTorch is not valid for this CUDA handoff")
    torch_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if torch_cuda_version != EXPECTED_CUDA_RUNTIME:
        raise CudaTrainingHandoffError(
            f"PyTorch CUDA runtime mismatch: {torch_cuda_version!r} != {EXPECTED_CUDA_RUNTIME!r}"
        )
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise CudaTrainingHandoffError(
            "CUDA is unavailable; CPU, DirectML, and ROCm devices are rejected"
        )
    device_count = int(cuda.device_count())
    if device_index < 0 or device_index >= device_count:
        raise CudaTrainingHandoffError(
            f"CUDA device index {device_index} is outside 0..{device_count - 1}"
        )
    name = str(cuda.get_device_name(device_index)).strip()
    if name != EXPECTED_DEVICE_NAME:
        raise CudaTrainingHandoffError(
            f"CUDA device {device_index} identity mismatch: {name!r} != {EXPECTED_DEVICE_NAME!r}"
        )
    capability = tuple(int(value) for value in cuda.get_device_capability(device_index))
    if capability != EXPECTED_DEVICE_CAPABILITY:
        raise CudaTrainingHandoffError(
            f"CUDA device capability mismatch: {capability} != {EXPECTED_DEVICE_CAPABILITY}"
        )
    architectures = tuple(str(value) for value in cuda.get_arch_list())
    if not _cuda_architecture_supported(architectures, capability):
        raise CudaTrainingHandoffError(
            f"installed PyTorch wheel has no code for sm_{capability[0]}{capability[1]}: "
            f"{architectures!r}"
        )
    properties = cuda.get_device_properties(device_index)
    total_memory = int(properties.total_memory)
    if total_memory < MINIMUM_TOTAL_VRAM_BYTES:
        raise CudaTrainingHandoffError(
            f"CUDA device has only {total_memory / 1024**3:.2f} GiB total VRAM; "
            f"at least {MINIMUM_TOTAL_VRAM_BYTES / 1024**3:.2f} GiB is required"
        )
    try:
        free_memory, reported_total = cuda.mem_get_info(device_index)
    except (AttributeError, RuntimeError) as exc:
        raise CudaTrainingHandoffError(f"could not query free CUDA VRAM: {exc}") from exc
    free_memory = int(free_memory)
    reported_total = int(reported_total)
    if reported_total <= 0 or free_memory < 0 or free_memory > reported_total:
        raise CudaTrainingHandoffError(
            "CUDA free/total VRAM query returned inconsistent values: "
            f"free={free_memory}, total={reported_total}"
        )
    if free_memory < MINIMUM_FREE_VRAM_BYTES:
        raise CudaTrainingHandoffError(
            f"CUDA device has only {free_memory / 1024**3:.2f} GiB free VRAM; "
            "close GPU applications and provide at least "
            f"{MINIMUM_FREE_VRAM_BYTES / 1024**3:.2f} GiB"
        )
    cudnn = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    if cudnn is None or not cudnn.is_available() or cudnn.version() is None:
        raise CudaTrainingHandoffError("cuDNN is unavailable in the CUDA PyTorch environment")
    try:
        cuda.set_device(device_index)
        torch_module.empty(1, device=f"cuda:{device_index}").add_(1)
        cuda.synchronize(device_index)
    except Exception as exc:
        raise CudaTrainingHandoffError(f"CUDA allocation/kernel preflight failed: {exc}") from exc
    return {
        "index": device_index,
        "name": name,
        "capability": list(capability),
        "compiled_architectures": list(architectures),
        "total_vram_bytes": total_memory,
        "free_vram_bytes": free_memory,
        "reported_total_vram_bytes": reported_total,
        "cuda_runtime": torch_cuda_version,
        "cudnn_version": int(cudnn.version()),
        "torch_import_version": str(getattr(torch_module, "__version__", "")),
    }


CUDA_PROBE_MARKER = "PROAIM_CUDA_PROBE_V1="
ULTRALYTICS_SETTINGS_SCHEMA_VERSION = "0.0.7"
CUDA_PROBE_CODE = r"""
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

from scripts.prepare_cuda_training_handoff import (
    EXPECTED_PACKAGE_VERSIONS,
    EXPECTED_TRAINING_SMOKE,
    verify_cuda_device,
)

try:
    import torch
    import torchvision
    from ultralytics import YOLO, __version__ as ultralytics_version

    device_index = int(sys.argv[1])
    weights_argument = Path(sys.argv[2])
    expected_weights_sha256 = sys.argv[3]
    training_batch = int(sys.argv[4])
    if not weights_argument.is_file() or weights_argument.is_symlink():
        raise RuntimeError("selected checkpoint is missing or unsafe in the CUDA probe")
    weights = weights_argument.resolve(strict=True)
    digest = sha256()
    with weights.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_weights_sha256:
        raise RuntimeError("selected checkpoint changed before the CUDA probe")
    if ultralytics_version != EXPECTED_PACKAGE_VERSIONS["ultralytics"]:
        raise RuntimeError("imported Ultralytics build differs from the audited package")
    device = "cuda:" + str(device_index)
    gpu = verify_cuda_device(torch, device_index)
    if str(getattr(torchvision, "__version__", "")) != EXPECTED_PACKAGE_VERSIONS["torchvision"]:
        raise RuntimeError("imported torchvision build differs from the audited CUDA wheel")
    torch.manual_seed(0)
    layer = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
    inputs = torch.randn((2, 3, 64, 64), device=device)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    loss = layer(inputs).square().mean()
    loss.backward()
    if not math.isfinite(float(loss.detach().cpu())):
        raise RuntimeError("CUDA convolution training smoke produced a non-finite loss")
    if not all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in layer.parameters()
    ):
        raise RuntimeError("CUDA convolution training smoke produced invalid gradients")
    optimizer.step()
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32, device=device)
    kept = torchvision.ops.nms(boxes, scores, 0.5)
    torch.cuda.synchronize(device_index)
    if kept.detach().cpu().tolist() != [0]:
        raise RuntimeError("torchvision CUDA NMS smoke returned the wrong result")

    selected = YOLO(str(weights))
    if selected.task != "detect" or getattr(selected.model, "task", None) != "detect":
        raise RuntimeError("selected checkpoint is not an Ultralytics detection model")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats(device_index)
    network = selected.model.to(device).train()
    selected_optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    selected_optimizer.zero_grad(set_to_none=True)
    selected_inputs = torch.zeros((training_batch, 3, 640, 640), device=device)
    selected_outputs = network(selected_inputs)

    def collect_tensors(value):
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, dict):
            tensors = []
            for nested in value.values():
                tensors.extend(collect_tensors(nested))
            return tensors
        if isinstance(value, (list, tuple)):
            tensors = []
            for nested in value:
                tensors.extend(collect_tensors(nested))
            return tensors
        return []

    output_tensors = collect_tensors(selected_outputs)
    if not output_tensors:
        raise RuntimeError("selected checkpoint CUDA forward returned no tensor outputs")
    if any(tensor.device != torch.device(device) for tensor in output_tensors):
        raise RuntimeError(
            "selected checkpoint CUDA forward returned an output on the wrong device"
        )
    selected_loss = sum(tensor.float().square().mean() for tensor in output_tensors)
    if not bool(torch.isfinite(selected_loss).item()):
        raise RuntimeError("selected checkpoint CUDA forward produced non-finite output")
    selected_loss.backward()
    selected_gradients = [
        parameter.grad for parameter in network.parameters() if parameter.grad is not None
    ]
    if not selected_gradients:
        raise RuntimeError("selected checkpoint CUDA backward produced no gradients")
    gradients_are_finite = torch.stack(
        [torch.isfinite(gradient).all() for gradient in selected_gradients]
    ).all()
    if not bool(gradients_are_finite.item()):
        raise RuntimeError("selected checkpoint CUDA backward produced non-finite gradients")
    selected_optimizer.step()
    parameters_are_finite = torch.stack(
        [torch.isfinite(parameter).all() for parameter in network.parameters()]
    ).all()
    if not bool(parameters_are_finite.item()):
        raise RuntimeError("selected checkpoint AdamW step produced non-finite parameters")
    torch.cuda.synchronize(device_index)
    gpu["torchvision_import_version"] = str(torchvision.__version__)
    gpu["training_smoke"] = dict(EXPECTED_TRAINING_SMOKE)
    gpu["selected_model_smoke"] = {
        "filename": weights.name,
        "sha256": expected_weights_sha256,
        "task": selected.task,
        "training_batch": training_batch,
        "image_size": 640,
        "precision": "fp32",
        "output_tensor_count": len(output_tensors),
        "gradient_tensor_count": len(selected_gradients),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device_index)),
    }
    result = {
        "ok": True,
        "torch_import_version": str(getattr(torch, "__version__", "")),
        "torchvision_import_version": str(getattr(torchvision, "__version__", "")),
        "gpu": gpu,
    }
except Exception as exc:
    result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
print("PROAIM_CUDA_PROBE_V1=" + json.dumps(result, sort_keys=True))
"""


def _write_isolated_ultralytics_settings(config_root: Path) -> Path:
    """Preseed pinned, offline settings so a fresh account emits no unaudited output."""

    settings_directory = config_root / "Ultralytics"
    settings_directory.mkdir()
    settings_path = settings_directory / "settings.json"
    settings = {
        "settings_version": ULTRALYTICS_SETTINGS_SCHEMA_VERSION,
        "datasets_dir": str(PROJECT_ROOT / "datasets"),
        "weights_dir": str(PROJECT_ROOT / "weights"),
        "runs_dir": str(PROJECT_ROOT / "runs"),
        "uuid": "0" * 64,
        "sync": False,
        "api_key": "",
        "openai_api_key": "",
        "clearml": False,
        "comet": False,
        "dvc": False,
        "mlflow": False,
        "neptune": False,
        "raytune": False,
        "tensorboard": False,
        "wandb": False,
        "vscode_msg": False,
        "openvino_msg": False,
    }
    settings_path.write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return settings_path


def probe_cuda_device_isolated(
    device_index: int,
    weights: Path,
    expected_weights_sha256: str,
    training_batch: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Validate CUDA in a short-lived process so its context is freed before training."""

    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise CudaTrainingHandoffError("CUDA device index must be non-negative")
    if (
        isinstance(training_batch, bool)
        or not isinstance(training_batch, int)
        or training_batch <= 0
    ):
        raise CudaTrainingHandoffError("CUDA probe training batch must be positive")
    probe_environment = os.environ.copy()
    probe_environment["PYTHONNOUSERSITE"] = "1"
    # Do not let an inherited developer PYTHONPATH shadow this checkout's
    # ``scripts`` namespace inside the evidence-producing child.
    probe_environment.pop("PYTHONPATH", None)
    try:
        with tempfile.TemporaryDirectory(prefix="proaim-cuda-probe-") as config_directory:
            _write_isolated_ultralytics_settings(Path(config_directory))
            probe_environment["YOLO_CONFIG_DIR"] = config_directory
            completed = runner(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    CUDA_PROBE_CODE,
                    str(device_index),
                    str(weights),
                    expected_weights_sha256,
                    str(training_batch),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
                env=probe_environment,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CudaTrainingHandoffError(f"isolated CUDA probe could not run: {exc}") from exc
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    stderr = completed.stderr if isinstance(completed.stderr, str) else ""
    records = [
        line[len(CUDA_PROBE_MARKER) :]
        for line in stdout.splitlines()
        if line.startswith(CUDA_PROBE_MARKER)
    ]
    nonempty_stdout = [line for line in stdout.splitlines() if line.strip()]
    if (
        completed.returncode != 0
        or stderr.strip()
        or len(records) != 1
        or len(nonempty_stdout) != 1
    ):
        raise CudaTrainingHandoffError(
            "isolated CUDA probe failed or produced unaudited output: "
            f"exit={completed.returncode}, stdout={stdout.strip()!r}, stderr={stderr.strip()!r}"
        )
    try:
        record = json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise CudaTrainingHandoffError("isolated CUDA probe returned invalid JSON") from exc
    if not isinstance(record, dict):
        raise CudaTrainingHandoffError("isolated CUDA probe record is not an object")
    if record.get("ok") is not True:
        error_type = record.get("error_type", "CudaTrainingHandoffError")
        error = record.get("error", "unknown error")
        raise CudaTrainingHandoffError(f"isolated CUDA probe blocked: {error_type}: {error}")
    if record.get("torch_import_version") != EXPECTED_PACKAGE_VERSIONS["torch"]:
        raise CudaTrainingHandoffError("isolated CUDA probe imported the wrong PyTorch build")
    if record.get("torchvision_import_version") != EXPECTED_PACKAGE_VERSIONS["torchvision"]:
        raise CudaTrainingHandoffError("isolated CUDA probe imported the wrong torchvision build")
    gpu = record.get("gpu")
    if not isinstance(gpu, dict):
        raise CudaTrainingHandoffError("isolated CUDA probe has no GPU evidence object")
    required = {
        "index": device_index,
        "name": EXPECTED_DEVICE_NAME,
        "capability": list(EXPECTED_DEVICE_CAPABILITY),
        "cuda_runtime": EXPECTED_CUDA_RUNTIME,
        "torch_import_version": EXPECTED_PACKAGE_VERSIONS["torch"],
    }
    for key, expected in required.items():
        if gpu.get(key) != expected:
            raise CudaTrainingHandoffError(
                f"isolated CUDA probe evidence mismatch for {key}: "
                f"{gpu.get(key)!r} != {expected!r}"
            )
    architectures = gpu.get("compiled_architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not all(isinstance(value, str) for value in architectures)
        or not _cuda_architecture_supported(architectures, EXPECTED_DEVICE_CAPABILITY)
    ):
        raise CudaTrainingHandoffError(
            "isolated CUDA probe has invalid compiled-architecture evidence"
        )
    for key, minimum in (
        ("total_vram_bytes", MINIMUM_TOTAL_VRAM_BYTES),
        ("free_vram_bytes", MINIMUM_FREE_VRAM_BYTES),
        ("reported_total_vram_bytes", 1),
        ("cudnn_version", 1),
    ):
        value = gpu.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise CudaTrainingHandoffError(
                f"isolated CUDA probe has invalid {key} evidence: {value!r}"
            )
    if gpu["free_vram_bytes"] > gpu["reported_total_vram_bytes"]:
        raise CudaTrainingHandoffError(
            "isolated CUDA probe free VRAM exceeds its reported total"
        )
    if gpu.get("torchvision_import_version") != EXPECTED_PACKAGE_VERSIONS["torchvision"]:
        raise CudaTrainingHandoffError(
            "isolated CUDA probe GPU record has the wrong torchvision build"
        )
    if gpu.get("training_smoke") != EXPECTED_TRAINING_SMOKE:
        raise CudaTrainingHandoffError(
            "isolated CUDA probe has invalid training-smoke evidence"
        )
    model_smoke = gpu.get("selected_model_smoke")
    expected_model_evidence = {
        "filename": weights.name,
        "sha256": expected_weights_sha256,
        "task": "detect",
        "training_batch": training_batch,
        "image_size": 640,
        "precision": "fp32",
    }
    if not isinstance(model_smoke, dict) or any(
        model_smoke.get(key) != expected
        for key, expected in expected_model_evidence.items()
    ):
        raise CudaTrainingHandoffError(
            "isolated CUDA probe has invalid selected-model evidence"
        )
    for key in (
        "output_tensor_count",
        "gradient_tensor_count",
        "peak_allocated_vram_bytes",
        "peak_reserved_vram_bytes",
    ):
        value = model_smoke.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CudaTrainingHandoffError(
                f"isolated CUDA probe has invalid selected-model {key}: {value!r}"
            )
    return gpu


def _windows_ac_power() -> bool | None:
    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_uint32),
            ("BatteryFullLifeTime", ctypes.c_uint32),
        ]

    status = SystemPowerStatus()
    try:
        success = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    except (AttributeError, OSError):
        return None
    if not success or status.ACLineStatus == 255:
        return None
    return status.ACLineStatus == 1


def _linux_ac_power() -> bool | None:
    root = Path("/sys/class/power_supply")
    if not root.is_dir():
        return None
    values: list[bool] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        try:
            supply_type = (entry / "type").read_text(encoding="utf-8").strip().casefold()
            online = (entry / "online").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if supply_type in {"mains", "usb", "usb_c", "usb_pd"} and online in {"0", "1"}:
            values.append(online == "1")
    return any(values) if values else None


def discover_ac_power() -> bool | None:
    system = platform.system()
    if system == "Windows":
        return _windows_ac_power()
    if system == "Linux":
        return _linux_ac_power()
    return None


@contextmanager
def training_sleep_inhibitor() -> Iterator[dict[str, Any]]:
    """Prevent Windows idle sleep while an explicitly launched job is running."""

    system = platform.system()
    if system != "Windows":
        yield {"status": "not_requested", "platform": system or "unknown"}
        return
    try:
        set_execution_state = ctypes.windll.kernel32.SetThreadExecutionState
        set_execution_state.argtypes = [ctypes.c_uint32]
        set_execution_state.restype = ctypes.c_uint32
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED. The display may turn off; only
        # idle system sleep is suppressed for the lifetime of this thread.
        request_flags = 0x80000001
        if not set_execution_state(request_flags):
            raise OSError("SetThreadExecutionState returned zero")
    except (AttributeError, OSError) as exc:
        raise CudaTrainingHandoffError(
            f"could not inhibit Windows idle sleep for CUDA training: {exc}"
        ) from exc
    try:
        yield {
            "status": "active_until_training_process_exits",
            "platform": "Windows",
            "mechanism": "SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)",
        }
    finally:
        # ES_CONTINUOUS clears the request. Thread/process exit also clears it,
        # so a rare reset failure must not hide the training subprocess result.
        try:
            set_execution_state(0x80000000)
        except (OSError, ValueError):
            pass


def _require_ac_online(
    power_probe: Callable[[], bool | None], *, after_gpu_probe: bool = False
) -> None:
    ac_power = power_probe()
    if ac_power is False:
        suffix = " remains" if after_gpu_probe else " is"
        raise CudaTrainingHandoffError(
            f"AC power{suffix} offline; plug in the laptop before sustained GPU work"
        )
    if ac_power is not True:
        suffix = " after the CUDA probe" if after_gpu_probe else ""
        raise CudaTrainingHandoffError(
            f"AC power could not be confirmed{suffix}; this laptop training handoff fails closed"
        )


def _safe_run_name(value: str) -> str:
    name = value.strip()
    windows_device_stem = name.split(".", 1)[0].upper()
    windows_reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    if (
        not name
        or name in {".", ".."}
        or name.endswith(".")
        or windows_device_stem in windows_reserved
        or PurePath(name).name != name
        or "/" in name
        or "\\" in name
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name)
    ):
        raise CudaTrainingHandoffError(
            "run name must be a plain 1-80 character name using letters, digits, ., _, or -"
        )
    if "fresh" not in name.casefold():
        raise CudaTrainingHandoffError("fresh CUDA run name must contain 'fresh'")
    return name


def _is_linklike(path: Path) -> bool:
    """Return true for symlinks and Windows directory junctions/reparse redirects."""

    try:
        is_junction = getattr(path, "is_junction", None)
        if path.is_symlink() or (callable(is_junction) and is_junction()):
            return True
        # ``Path.is_junction`` was added after the oldest supported CPython.
        # Reject every Windows reparse point on 3.10/3.11 as a conservative
        # fallback; this includes junctions and other redirecting filesystem
        # objects that must not be followed for an evidence/run path.
        if os.name == "nt" and os.path.lexists(path):
            get_attributes = ctypes.windll.kernel32.GetFileAttributesW
            get_attributes.argtypes = [ctypes.c_wchar_p]
            get_attributes.restype = ctypes.c_uint32
            attributes = int(get_attributes(str(path)))
            if attributes == 0xFFFFFFFF:
                raise OSError(f"GetFileAttributesW failed for {path}")
            return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
        return False
    except OSError as exc:
        raise CudaTrainingHandoffError(f"could not inspect path safety: {path}: {exc}") from exc


def _verify_directory_chain(path: Path, description: str) -> None:
    """Reject existing files/symlinks while allowing not-yet-created directories."""

    for entry in (path, *path.parents):
        if not os.path.lexists(entry):
            continue
        if _is_linklike(entry) or not entry.is_dir():
            raise CudaTrainingHandoffError(
                f"{description} has an unsafe existing path component: {entry}"
            )


def _verify_new_run_directory(run_name: str) -> Path:
    # ``runs/`` is intentionally gitignored, so a clean repository copy may not
    # contain this directory yet. The trainer creates it atomically with the new
    # run directory; every existing path component still has to be a real
    # directory rather than a file or symlink.
    _verify_directory_chain(TRAINING_PROJECT, "training project directory")
    run_dir = TRAINING_PROJECT / _safe_run_name(run_name)
    if os.path.lexists(run_dir):
        raise CudaTrainingHandoffError(
            f"fresh run directory already exists; refusing reuse/resume: {run_dir}"
        )
    return run_dir


def _ensure_evidence_directory() -> Path:
    # Check before mkdir so a pre-existing redirecting ancestor is never
    # followed as a side effect, then check again to catch replacement races.
    _verify_directory_chain(TRAINING_PROJECT, "training project directory")
    try:
        TRAINING_PROJECT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CudaTrainingHandoffError(
            f"could not create the training project directory: {exc}"
        ) from exc
    _verify_directory_chain(TRAINING_PROJECT, "training project directory")

    evidence_directory = TRAINING_PROJECT / ".cuda-training-handoffs"
    if os.path.lexists(evidence_directory):
        if _is_linklike(evidence_directory) or not evidence_directory.is_dir():
            raise CudaTrainingHandoffError(
                f"CUDA handoff evidence directory is unsafe: {evidence_directory}"
            )
        return evidence_directory
    try:
        evidence_directory.mkdir(mode=0o700)
    except FileExistsError:
        if _is_linklike(evidence_directory) or not evidence_directory.is_dir():
            raise CudaTrainingHandoffError(
                f"CUDA handoff evidence directory is unsafe: {evidence_directory}"
            )
    except OSError as exc:
        raise CudaTrainingHandoffError(
            f"could not create CUDA handoff evidence directory: {exc}"
        ) from exc
    return evidence_directory


@contextmanager
def training_execution_lock(device_index: int) -> Iterator[Path]:
    """Hold a non-blocking per-device OS lock for the full training process."""

    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise CudaTrainingHandoffError("CUDA lock device index must be non-negative")
    evidence_directory = _ensure_evidence_directory()
    lock_path = evidence_directory / f"cuda-device-{device_index}.lock"
    if os.path.lexists(lock_path) and (
        _is_linklike(lock_path) or not lock_path.is_file()
    ):
        raise CudaTrainingHandoffError(f"CUDA training lock path is unsafe: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            opened_stat = os.fstat(descriptor)
            path_stat = os.stat(lock_path, follow_symlinks=False)
            unsafe = (
                not stat.S_ISREG(opened_stat.st_mode)
                or opened_stat.st_nlink != 1
                or (opened_stat.st_dev, opened_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
                or _is_linklike(lock_path)
            )
        except Exception:
            os.close(descriptor)
            raise
        if unsafe:
            os.close(descriptor)
            raise CudaTrainingHandoffError(
                f"CUDA training lock path changed or is unsafe: {lock_path}"
            )
        lock_file = os.fdopen(descriptor, "r+b")
    except CudaTrainingHandoffError:
        raise
    except OSError as exc:
        raise CudaTrainingHandoffError(f"could not open CUDA training lock: {exc}") from exc

    system = platform.system()
    locked = False
    try:
        if system == "Windows":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise CudaTrainingHandoffError(
                    f"another CUDA training handoff already holds device {device_index}: "
                    f"{lock_path}"
                ) from exc
        elif system == "Linux":
            import fcntl

            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise CudaTrainingHandoffError(
                    f"another CUDA training handoff already holds device {device_index}: "
                    f"{lock_path}"
                ) from exc
        else:
            raise CudaTrainingHandoffError(
                f"CUDA training execution locking is unsupported on {system or 'this platform'}"
            )
        locked = True
        lock_record = {
            "schema_version": 1,
            "device_index": device_index,
            "holder_pid": os.getpid(),
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write((json.dumps(lock_record, sort_keys=True) + "\n").encode("utf-8"))
        lock_file.flush()
        os.fsync(lock_file.fileno())
        yield lock_path
    finally:
        if locked:
            if system == "Windows":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            elif system == "Linux":
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def persist_execution_authorization(
    report: Mapping[str, Any],
    run_name: str,
    *,
    power_probe: Callable[[], bool | None] = discover_ac_power,
) -> tuple[Path, str]:
    """Atomically persist the exact launch authorization; never used for dry runs."""

    safe_name = _safe_run_name(run_name)
    _verify_new_run_directory(safe_name)
    if power_probe() is not True:
        raise CudaTrainingHandoffError(
            "AC power was not online at launch authorization; training was not started"
        )
    evidence_directory = _ensure_evidence_directory()

    evidence_path = evidence_directory / f"{safe_name}.authorization.json"
    evidence = dict(report)
    evidence["execution_authorization"] = {
        "status": "authorized_to_start_exact_fresh_run",
        "run_name_confirmation": safe_name,
        "ac_reconfirmed": "online",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    serialized = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    try:
        with evidence_path.open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(serialized)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as exc:
        raise CudaTrainingHandoffError(
            "a CUDA launch authorization already exists for this run name; "
            f"choose a new fresh run name: {evidence_path}"
        ) from exc
    except OSError as exc:
        raise CudaTrainingHandoffError(
            f"could not persist CUDA launch authorization: {exc}"
        ) from exc
    _require_regular_file(evidence_path, "CUDA launch authorization")
    return evidence_path, _sha256_file(evidence_path)


def build_fresh_command(
    *,
    python_executable: str,
    weights: Path,
    run_name: str,
    device_index: int,
    batch_size: int,
) -> list[str]:
    return [
        python_executable,
        "-u",
        str(TRAINING_SCRIPT),
        "--data",
        str(DATASET_YAML),
        "--weights",
        str(weights),
        "--project",
        str(TRAINING_PROJECT),
        "--name",
        run_name,
        "--epochs",
        "60",
        "--patience",
        "15",
        "--batch",
        str(batch_size),
        "--imgsz",
        "640",
        "--device",
        str(device_index),
        "--workers",
        "4",
        "--threads",
        "6",
        "--cache",
        "none",
        "--seed",
        "0",
        "--skip-test",
    ]


def _display_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def revalidate_exact_launch_snapshot(
    report: Mapping[str, Any],
    command: Sequence[str],
    *,
    model_size: str,
    run_name: str,
    device_index: int,
    version_getter: Callable[[str], str] = metadata.version,
    power_probe: Callable[[], bool | None] = discover_ac_power,
    dataset_verifier: Callable[[], dict[str, Any]] = verify_audited_dataset,
) -> dict[str, Any]:
    """Rehash launch inputs and bind the exact argv immediately before authorization."""

    if model_size not in MODEL_CONTRACTS:
        raise CudaTrainingHandoffError(f"unsupported model size: {model_size!r}")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise CudaTrainingHandoffError("CUDA device index must be a non-negative integer")
    safe_name = _safe_run_name(run_name)
    model = MODEL_CONTRACTS[model_size]
    weights = PROJECT_ROOT / model.filename
    expected_command = build_fresh_command(
        python_executable=sys.executable,
        weights=weights,
        run_name=safe_name,
        device_index=device_index,
        batch_size=model.training_batch,
    )
    actual_command = list(command)
    if actual_command != expected_command:
        raise CudaTrainingHandoffError(
            "launch command changed after the complete CUDA preflight"
        )
    if report.get("command_argv") != expected_command or report.get(
        "command"
    ) != _display_command(expected_command):
        raise CudaTrainingHandoffError(
            "launch report is not bound to the exact generated command"
        )
    if (
        report.get("status") != "ready_fresh_cuda_training_not_started"
        or report.get("training_started") is not False
        or report.get("python_executable") != sys.executable
        or report.get("fresh_run_directory")
        != str(TRAINING_PROJECT / safe_name)
    ):
        raise CudaTrainingHandoffError("launch report identity/state is inconsistent")

    package_versions = verify_packages(version_getter)
    if report.get("packages") != package_versions:
        raise CudaTrainingHandoffError("training packages changed after CUDA preflight")
    contract_script_sha = _require_hash(
        DATASET_CONTRACT_SCRIPT,
        AUDITED_DATASET_CONTRACT_SCRIPT_SHA256,
        "audited dataset-contract script",
    )
    if report.get("dataset_contract_script") != {
        "path": str(DATASET_CONTRACT_SCRIPT),
        "sha256": contract_script_sha,
    }:
        raise CudaTrainingHandoffError(
            "dataset-contract script evidence changed after CUDA preflight"
        )
    dataset = dataset_verifier()
    if report.get("dataset") != dataset:
        raise CudaTrainingHandoffError("dataset changed after CUDA preflight")
    weights_sha = _require_hash(weights, model.sha256, f"pinned {model.filename} checkpoint")
    if report.get("base_checkpoint") != {
        "model_size": model_size,
        "path": str(weights),
        "sha256": weights_sha,
    }:
        raise CudaTrainingHandoffError("base-checkpoint evidence changed after CUDA preflight")
    training_script_sha = _require_hash(
        TRAINING_SCRIPT,
        AUDITED_TRAINING_SCRIPT_SHA256,
        "audited training script",
    )
    if report.get("training_script") != {
        "path": str(TRAINING_SCRIPT),
        "sha256": training_script_sha,
    }:
        raise CudaTrainingHandoffError("training-script evidence changed after CUDA preflight")
    _verify_new_run_directory(safe_name)
    _require_ac_online(power_probe, after_gpu_probe=True)
    command_sha = sha256(
        json.dumps(
            expected_command,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": "exact_launch_snapshot_revalidated",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_content_sha256": dataset["content_sha256"],
        "dataset_contract_script_sha256": contract_script_sha,
        "base_checkpoint_sha256": weights_sha,
        "training_script_sha256": training_script_sha,
        "command_argv_sha256": command_sha,
        "ac": "online",
    }


def prepare_handoff(
    *,
    model_size: str,
    run_name: str,
    device_index: int,
    torch_module: Any | None = None,
    cuda_probe: Callable[[int, Path, str, int], dict[str, Any]] = probe_cuda_device_isolated,
    version_getter: Callable[[str], str] = metadata.version,
    power_probe: Callable[[], bool | None] = discover_ac_power,
    dataset_verifier: Callable[[], dict[str, Any]] = verify_audited_dataset,
) -> tuple[dict[str, Any], list[str]]:
    if model_size not in MODEL_CONTRACTS:
        raise CudaTrainingHandoffError(f"unsupported model size: {model_size!r}")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise CudaTrainingHandoffError("CUDA device index must be a non-negative integer")
    if sys.implementation.name != "cpython" or not (3, 10) <= sys.version_info[:2] <= (3, 14):
        raise CudaTrainingHandoffError("CPython 3.10 through 3.14 is required")
    _require_ac_online(power_probe)
    package_versions = verify_packages(version_getter)
    dataset_contract_script_sha = _require_hash(
        DATASET_CONTRACT_SCRIPT,
        AUDITED_DATASET_CONTRACT_SCRIPT_SHA256,
        "audited dataset-contract script",
    )
    dataset = dataset_verifier()
    model = MODEL_CONTRACTS[model_size]
    weights = PROJECT_ROOT / model.filename
    weight_sha = _require_hash(weights, model.sha256, f"pinned {model.filename} checkpoint")
    training_script_sha = _require_hash(
        TRAINING_SCRIPT,
        AUDITED_TRAINING_SCRIPT_SHA256,
        "audited training script",
    )
    run_dir = _verify_new_run_directory(run_name)
    command = build_fresh_command(
        python_executable=sys.executable,
        weights=weights,
        run_name=run_name,
        device_index=device_index,
        batch_size=model.training_batch,
    )
    if any(
        argument == "--resume-from" or argument == "--adopt-interrupted-run"
        for argument in command
    ):
        raise CudaTrainingHandoffError("internal error: generated command is not a fresh run")
    # Probe the accelerator last so its free-memory observation is as close as
    # possible to the eventual launch. Production uses a short-lived child;
    # injected modules are retained only for unit tests.
    if torch_module is None:
        gpu = cuda_probe(
            device_index,
            weights,
            weight_sha,
            model.training_batch,
        )
    else:
        if str(getattr(torch_module, "__version__", "")) != EXPECTED_PACKAGE_VERSIONS["torch"]:
            raise CudaTrainingHandoffError(
                "imported PyTorch build differs from installed metadata"
            )
        gpu = verify_cuda_device(torch_module, device_index)
    _require_ac_online(power_probe, after_gpu_probe=True)
    report = {
        "schema_version": 1,
        "status": "ready_fresh_cuda_training_not_started",
        "training_started": False,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": package_versions,
        "gpu": gpu,
        "power": {"ac": "online"},
        "dataset": dataset,
        "dataset_contract_script": {
            "path": str(DATASET_CONTRACT_SCRIPT),
            "sha256": dataset_contract_script_sha,
        },
        "base_checkpoint": {
            "model_size": model_size,
            "path": str(weights),
            "sha256": weight_sha,
        },
        "training_script": {
            "path": str(TRAINING_SCRIPT),
            "sha256": training_script_sha,
        },
        "handoff_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "fresh_run_directory": str(run_dir),
        "supplied_test_split": "skipped_not_independent",
        "command_argv": command,
        "command": _display_command(command),
    }
    return report, command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the exact v9 data/checkpoint/environment and RTX 5060 Laptop, "
            "then emit or explicitly execute a fresh CUDA training command."
        )
    )
    parser.add_argument("--model", choices=tuple(MODEL_CONTRACTS), default="n")
    parser.add_argument(
        "--run-name",
        help="New run name. Defaults to the pinned n/s RTX 5060 fresh-run name.",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start the emitted fresh command after every preflight passes.",
    )
    parser.add_argument(
        "--confirm-run-name",
        help="With --execute, must exactly equal --run-name (or its default).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = MODEL_CONTRACTS[args.model]
    run_name = args.run_name or contract.default_run_name
    try:
        report, command = prepare_handoff(
            model_size=args.model,
            run_name=run_name,
            device_index=args.device_index,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if not args.execute:
            print("Preflight passed. Training was not started.", flush=True)
            return 0
        if args.confirm_run_name != run_name:
            raise CudaTrainingHandoffError(
                "--execute requires --confirm-run-name to exactly match the fresh run name"
            )
        with training_execution_lock(
            args.device_index
        ) as lock_path, training_sleep_inhibitor() as sleep_inhibition:
            # Repeat the entire fail-closed preflight under the device lock. This
            # rehashes every dataset member, base checkpoint, and trainer; reruns
            # the isolated CUDA training smoke; reconfirms packages/AC/free VRAM;
            # and regenerates the exact argv immediately before authorization.
            launch_report, launch_command = prepare_handoff(
                model_size=args.model,
                run_name=run_name,
                device_index=args.device_index,
            )
            launch_report["sleep_inhibition"] = sleep_inhibition
            launch_report["final_launch_revalidation"] = (
                revalidate_exact_launch_snapshot(
                    launch_report,
                    launch_command,
                    model_size=args.model,
                    run_name=run_name,
                    device_index=args.device_index,
                )
            )
            evidence_path, evidence_sha = persist_execution_authorization(
                launch_report, run_name
            )
            print(f"CUDA training lock: {lock_path}", flush=True)
            print(
                f"Launch authorization: {evidence_path} (SHA-256 {evidence_sha})",
                flush=True,
            )
            print(f"Starting exact fresh run: {launch_report['command']}", flush=True)
            try:
                return subprocess.run(
                    launch_command, cwd=PROJECT_ROOT, check=False
                ).returncode
            except OSError as exc:
                raise CudaTrainingHandoffError(
                    f"could not start the training subprocess: {exc}"
                ) from exc
    except CudaTrainingHandoffError as exc:
        print(f"CUDA training handoff blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
