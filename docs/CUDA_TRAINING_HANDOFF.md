# Audited v9 CUDA training handoff

This handoff is for a plugged-in **NVIDIA GeForce RTX 5060 Laptop GPU**. It
verifies the exact audited v9 dataset, the selected base checkpoint, the
training package versions, CUDA device identity/capability, total and free
VRAM, confirmed AC power, and a new output directory before it prints a command.
An unavailable or indeterminate AC reading fails closed. Printing a command does
not start training or create the gitignored output directory. The helper neither
downloads nor packages the private dataset.

CUDA allocation and kernel execution are checked in a short-lived child process.
AC power is confirmed before that GPU workload and again after it.
The probe also reloads and rehashes the selected `n` or `s` checkpoint, then runs
a full-size 640 FP32 raw-head forward/backward and AdamW step at that model's
pinned batch. It requires finite outputs, gradients, and post-AdamW parameters,
and records its measured peak allocated/reserved VRAM. The probe
exits before training begins, so the handoff itself does not retain a CUDA context
or reserve scarce VRAM while it waits for the 8 GB GPU training job.

The RTX 5060 Laptop is an 8 GB Blackwell CUDA GPU. NVIDIA lists the RTX 5060
family at compute capability 12.0, and PyTorch supports CUDA training on
Windows. Sources: [NVIDIA laptop specifications](https://www.nvidia.com/en-us/geforce/laptops/compare/),
[NVIDIA CUDA capability table](https://developer.nvidia.com/cuda/gpus), and
[PyTorch Windows setup](https://pytorch.org/get-started/locally/).

## Create the matching environment on the RTX laptop

Use a clean CPython 3.14 virtual environment from the repository root. These
commands are setup instructions; the preflight itself never installs anything.

```powershell
py -3.14 -m venv .venv-train-cuda130
.\.venv-train-cuda130\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.1.2
python -m pip install --only-binary=:all: torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install --only-binary=:all: ultralytics==8.4.116 numpy==2.4.4 opencv-python==5.0.0.93 PyYAML==6.0.3 pillow==12.2.0 matplotlib==3.11.1 psutil==7.2.2 scipy==1.18.0 ultralytics-thop==2.1.6
python -m pip check
```

The helper requires the CUDA wheel identities `torch==2.13.0+cu130` and
`torchvision==0.28.0+cu130`, not the CPU wheels. PyTorch publishes these CUDA
13 wheels in its [official wheel index](https://download.pytorch.org/whl/cu130/).

Copy the repository and the already-audited local files to the same relative
locations on the training laptop. Do not publish or attach the dataset to a
GitHub release:

- `datasets/fort_cuh_player_grouped_v9/`
- `yolo26n.pt`
- `yolo26s.pt` when testing the larger candidate

## Verify, then start a fresh run

Plug the laptop into AC power, close games and other GPU-heavy applications,
and run the emission-only preflight first:

```powershell
python scripts/prepare_cuda_training_handoff.py --model n
```

A successful result ends with `Training was not started` and prints the exact
fresh command. To deliberately execute that same command, acknowledge its new
run name exactly:

```powershell
python scripts/prepare_cuda_training_handoff.py --model n --execute --confirm-run-name yolo26n_640_player_grouped_v9_rtx5060_fresh
```

Immediately before launch, the helper rechecks that the run name is unused and
atomically creates a one-time JSON authorization record under
`runs/fort_cuh/.cuda-training-handoffs/`. The record contains the exact dataset,
checkpoint, training-script, package, CUDA-device, power, and command evidence
from a complete second preflight under the launch lock, plus one final successful
AC-power check. This immediately rehashes every dataset member, the checkpoint,
the trainer, and the trainer's local dataset-contract dependency against their
audited pinned hashes; repeats the isolated CUDA training smoke and free-VRAM
check; and regenerates the exact command. A final parent-process snapshot again
rehashes those launch inputs and binds the exact argument vector immediately
before authorization. The authorization SHA-256 is printed to the
console. A per-device operating-system lock is held until the training subprocess
exits, so a second `n` or `s` launch from the same checkout cannot race onto the
same GPU. On Windows, the launcher also holds a scoped system-required power
request until the subprocess exits, preventing idle sleep without forcing the
display to remain on; it cannot override an explicit sleep command or closing
the lid. A clean clone
does not need a pre-created `runs/fort_cuh` directory; only `--execute` creates
that ignored output path.

After the `n` run is complete, the larger `s` candidate uses a separate fresh
run and official base checkpoint:

```powershell
python scripts/prepare_cuda_training_handoff.py --model s
python scripts/prepare_cuda_training_handoff.py --model s --execute --confirm-run-name yolo26s_640_player_grouped_v9_rtx5060_fresh
```

Never run the two jobs concurrently on the 8 GB laptop GPU. The commands do not
resume the CPU optimizer state, disable dataset caching, preserve the audited
seed/trainer settings, and skip the already-consumed non-independent test split.
The `n` baseline uses fixed batch 8. The roughly four-times-larger `s` candidate
uses fixed batch 4 to leave headroom for FP32 activations and the Windows display
on an 8 GB device; both retain Ultralytics' nominal-batch gradient accumulation.
The existing trainer also keeps AMP disabled for a controlled model comparison;
that training choice does not make the exported inference model CPU-oriented.

## RX 6950 XT boundary

Do not use this helper for the RX 6950 XT. AMD's current Windows PyTorch/ROCm
support matrix lists selected Radeon 7000/9000 GPUs and does not list the RX
6950 XT. Microsoft's `torch-directml` is also limited to PyTorch 2.3.1, whereas
this audited training environment is PyTorch 2.13. ProAim therefore uses the RX
6950 XT through its DirectML/ONNX **inference** path; it is not an approved
Windows PyTorch training target for this model tournament. Sources:
[AMD Windows support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
and [Microsoft torch-directml support](https://learn.microsoft.com/windows/ai/directml/pytorch-windows).
