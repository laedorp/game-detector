# ProAim

ProAim is a low-latency object detector for local, offline game footage. It can
capture a Moonlight stream, a camera/capture card, or a video file; run one of
the bundled OpenVINO or ONNX models; and show detections and timing in a
user-friendly desktop app.

The custom model recognizes a visual `player` class. It does not know identity,
team, ally, or opponent. The optional third-person self filter is a conservative
screen-position heuristic, so always confirm its orange preview outline.
The bundled generic COCO presets cannot drive clone-player aim output; choose a
trained player preset for that workflow. The large YOLO11l option is benchmark
only.

## Download and launch

Download the matching ZIP from the repository's GitHub **Releases** page,
extract the complete `ProAim` folder, and keep every adjacent file together.

| system | release asset | launch |
| --- | --- | --- |
| Linux, broad CPU/Intel compatibility | `ProAim-Linux-x64.zip` | `ProAim/ProAim` |
| Windows, AMD/Intel/NVIDIA through DirectX 12 | `ProAim-Windows-x64-DirectML.zip` | `ProAim\ProAim.exe` |

The bundles include Python and their selected inference runtime. DirectML is
the supported Windows release path for AMD, Intel, and NVIDIA GPUs.

CUDA builds remain an experimental, manual option. They include CUDA 13/cuDNN
9 user-space libraries and require a compatible NVIDIA display driver, but are
not published by version tags until the frozen archives pass real inference on
a representative NVIDIA machine. Windows developers can run the manual
**Build Windows app** workflow; Linux developers can build from source with
`PROAIM_RUNTIME_VARIANT=cuda` and must validate the result on NVIDIA hardware.

On Linux, `ProAim/install.sh` installs or updates a per-user application-menu
entry named **ProAim**. The executable is copied to
`~/.local/share/proaim/ProAim`, and the previous managed installation is kept at
`~/.local/share/proaim.previous`. No root access is used for the app install.
Run `ProAim/install.sh --uninstall` to remove only marker-verified ProAim
install/backup folders and its menu entry; settings under `~/.config/proaim`
are intentionally retained.

To run the current checkout instead, use:

```bash
./run_gui.sh
```

## One-click workflow

The Qt interface is ProAim's default UI. Its four sections are **Capture
source**, **Detection**, **Hardware**, and **Controller precision**.

1. In **Capture source**, choose **Moonlight / screen capture**, **Capture card
   or camera**, or **Video file**. **Open Moonlight** launches the normal
   Moonlight client.
2. In **Detection**, leave the **Game players** preset marked **Recommended**
   selected for the development-selected custom-player default. Its exact model
   and static H×W come from the hash-bound release pointer. Configure the
   preview, optional centered crop, and third-person self filter here.
3. On a genuinely fresh profile, ProAim scans in the background and applies a
   ready discrete GPU only when it can identify exactly one safe choice. On a
   hybrid Windows laptop, DirectML must also match the GPU to an exact DXGI
   adapter index; this makes the RTX/Radeon win over an integrated GPU without
   guessing. Existing saved choices are never replaced. If the scan is
   ambiguous, open **Hardware**, choose **Scan hardware**, and apply the intended
   entry; **Start detection** otherwise requires explicit confirmation before
   using CPU/OpenVINO. Provider initialization is verified when detection
   starts, and physical GPU activity must still be confirmed in Task Manager.
4. Press **Start detection**. F5 also starts; Escape or **Stop** ends the run.

The preview is a separate window. When capturing a screen rectangle, keep the
preview outside it so it does not feed back into the detector. Live sources use
a one-frame mailbox: when inference falls behind, an unread frame is replaced
instead of building a latency queue. The preview defaults to 15 FPS and is
sampled without sleeping. Native Windows HighGUI uses a dedicated latest-only
worker; Linux/Qt and other thread-affine or unknown GUI backends keep HighGUI on
the process main thread for correctness. The startup log names the active mode.
The runtime's `preview service` moving average is amortized across all processed
frames (non-preview frames contribute zero). In inline
mode it measures frame submission and HighGUI event service; in threaded mode it
measures the owned frame copy and mailbox publication. It is never a display
scanout or glass-to-glass measurement.

On Windows, screen sources use DXcam's Desktop Duplication path with a two-frame
backend ring and the application's one-frame latest mailbox; if DXGI cannot be
initialized, ProAim reports the reason and falls back to MSS. On Linux,
Moonlight screen capture currently requires an X11/Xorg desktop session. On
Wayland, camera and video sources still work, but screen capture does not. Log
out and select an X11/Xorg session before starting ProAim and Moonlight.

## Models and measured tradeoffs

Six model configurations are release-validated. A matching `.xml`/`.bin` pair
is used by OpenVINO; an ONNX graph is used for AMD or NVIDIA GPU execution.

| UI preset | input | purpose | formats |
| --- | ---: | --- | --- |
| Game players — Selected (Recommended) | pointer-declared H×W | development-selected default for the custom `player` class | OpenVINO + ONNX |
| Game players — Responsive 416 INT8 (OpenVINO CPU) | 416 | responsive CPU option | OpenVINO only |
| Game players — Fast 320 | 320 | smallest custom-player fallback | OpenVINO + ONNX |
| People — Balanced 416 (COCO fallback) | 416 | generic COCO people | OpenVINO + ONNX |
| People — Fast 320 | 320 | lower-latency generic COCO people | OpenVINO + ONNX |
| Benchmark only — YOLO11l 640 (slow, generic COCO) | 640 | explicit large-model comparison; not a responsive default | OpenVINO + ONNX |

On the audited Intel Core i7-10850H, the measured model pipeline (preprocess +
inference + postprocess, excluding capture/display) averaged
18.22 ms for the legacy 416 FP32 model, 13.08 ms for 416 INT8, and 12.98 ms for
320 FP32. The INT8 artifact was 67.2% smaller than legacy 416 FP32 and kept more
test recall and mAP50 than 320 FP32. Historical supplied-test quality for that
416 model was 0.7202 mAP50 and 0.3526 mAP50-95. The live default is instead
defined exclusively by `models/RELEASE-DEFAULT.json`; that pointer also binds
the optional detail pass's requested source ROI width (`0` disables it).
Changing it requires the fail-closed sealed-tournament adoption path and does
not qualify release. An adopted asset set includes both runtime formats and
self-contained, path-redacted candidate/training/runtime receipts plus every
sealed public tournament input; the pointer and release manifest hash-bind the
complete inventory. The tournament and adoption records also SHA-bind the exact
shared privacy scanner used to reject workstation paths, and adoption snapshots
that source through atomic publication. A 640 custom export was rejected because it was much
slower without a meaningful accuracy
gain.

Hardware, power mode, drivers, and thermals matter more than a device's product
name. Run `scripts/benchmark_models.py` on the actual computer before locking in
a preset. For a released Windows bundle, run
`ProAimCLI.exe --benchmark-models ...` from the extracted folder so the report
measures the exact packaged runtime and model. The exact method, accuracy
tables, artifact hashes, and reproducible command are in
[docs/MODEL_BENCHMARKS.md](docs/MODEL_BENCHMARKS.md).

## Hardware runtimes

ProAim's responsive live-detection target is a discrete GPU. CPU/OpenVINO is
retained as a portable compatibility and diagnostic path, not as the latency
target for high-detail or detail-pass modes. Final release candidates are
trained fresh with CUDA on the RTX 5060; no CPU training pass is required. The
exported ONNX graph remains portable and can run through a compatible DirectML,
CUDA, MIGraphX/legacy ROCm, or CPU provider selected by that installation.

| hardware | inference path | packaged choice |
| --- | --- | --- |
| Intel or AMD CPU | OpenVINO; ONNX CPU fallback | Linux CPU or Windows DirectML bundle |
| Intel integrated/Arc GPU or Intel NPU | OpenVINO | either bundle for that OS |
| NVIDIA GPU on Linux | ONNX Runtime CUDA | manual source build; experimental until hardware-validated |
| NVIDIA GPU on Windows | ONNX Runtime DirectML; manual CUDA option | Windows DirectML release; CUDA experimental |
| AMD GPU on Windows | ONNX Runtime DirectML | Windows DirectML bundle |
| AMD GPU on Linux | MIGraphX/legacy ROCm only on an AMD-supported GPU/OS combination | experimental source setup; CPU fallback otherwise |

OpenVINO does not provide AMD or NVIDIA GPU plugins. Those devices therefore
use the equivalent ONNX model. GPU runtimes also depend on compatible vendor
drivers installed by the operating system; ProAim does not install drivers or
request administrator access.

## Run from source

Create an isolated environment, install the common application requirements,
then install exactly one ONNX Runtime variant. `requirements.txt` includes
PySide6 for the default Qt interface.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-runtime-cpu.txt
python app.py
```

Replace the last requirements file with exactly one of:

- `requirements-runtime-cuda.txt` for NVIDIA CUDA 13/cuDNN 9 on Linux or Windows;
- `requirements-runtime-directml.txt` for broad Windows GPU compatibility;
- `requirements-runtime-rocm.txt` only for a deliberately qualified legacy
  Linux ROCm setup. ONNX Runtime removed the ROCm provider after 1.22 and now
  directs new AMD integrations to MIGraphX.

The variants all install the same `onnxruntime` Python module and must not be
stacked in one environment. For Linux AMD, first verify the exact GPU, operating
system, ROCm version, and provider against AMD's current support matrix. The RX
6950 XT is not in AMD's current supported Radeon ROCm list, so ProAim recommends
the Windows DirectML bundle for that card rather than presenting an
`HSA_OVERRIDE_GFX_VERSION` workaround as support. If an AMD provider cannot
initialize, use the CPU requirements file and OpenVINO CPU instead.

Tk is not required for the default Qt UI. A source checkout can still use the
legacy `python app.py --tk` launcher when Tk is installed, but release bundles
carry only the supported Qt interface.

Inspect what this environment can actually use:

```bash
python app.py --runtime-info
python scripts/scan_hardware.py
```

`--runtime-info` prints JSON containing Python, OpenVINO devices, ONNX Runtime
version, and available execution providers. A frozen bundle also contains
`DEPENDENCY-MANIFEST.json` with the exact hash-locked package artifacts and
`BUILD-INFO.json` with its source commit, runtime variant, and the manifest's
SHA-256.

## Moonlight and Linux Controller precision

The intended capture path is:

```text
gaming PC: game + Sunshine encode
                 ↓ local network
laptop: Moonlight decode → X11 screen capture → ProAim detection
```

For normal controller play, connect the controller to the laptop and let
Moonlight forward it directly to Sunshine. If the detector preview takes focus,
enable Moonlight's option to process gamepad input in the background. Detection
is not part of this ordinary controller-forwarding path.

**Controller precision** is a separate Linux-only, user-directed proxy for the
tested PXN P5 8K. It temporarily exposes one virtual controller and reshapes
right-stick movement that the user physically makes while LT is held. Use
**Refresh**, complete **Verify LT + right stick…**, choose a curve, and then
press **Start controller precision** before opening Moonlight. It does not read
detections or choose a direction. It requires access to the physical evdev node
and `/dev/uinput`; do not run the whole app as root.

Only after the status says it is active, open Moonlight so Moonlight enumerates
the virtual controller rather than the temporarily grabbed physical one. This
worker does not import video or detector modules, listen on UDP, or consume
target coordinates.

## Optional MAKCU aim

The desktop UI can send bounded relative mouse correction through a supported
MAKCU passthrough board. This path is local and offline. It does not synthesize
clicks, read game memory, inspect network traffic, or interact with anti-cheat
software.

To enable it safely:

1. Connect the MAKCU control/serial side to the detection computer, its output
   side to the other PC, and the physical mouse through the board.
2. On Linux source checkouts, install the narrow device-access rule once with
   `sudo bash scripts/install_makcu_access.sh`. From an extracted/installed
   bundle, run `sudo bash setup/install_makcu_access.sh` instead. Then reconnect
   the board; do not run ProAim as root. The installed rule is
   `packaging/linux/70-game-detector-makcu.rules`.
3. In **Detection → MAKCU aim**, enter an explicit **Target label** such as
   `player`, select the exact board and physical activation button, and complete
   the read-only **Verify … Mouse** check. Verification requires both a press
   and a release and is bound to that board path and button.
4. Start detection, validate the preview and self-filter outline, and use the
   physical button as the activation gate.

Output is neutralized immediately when the gate is released, a detector state
becomes stale, the self filter is unsafe, or the process stops. CLI local output
also requires an explicit physical input-event gate via `--aim-activate-path`.
As a missed-release safeguard, one uninterrupted MAKCU activation is capped at
10 seconds; release and press the gate again to continue.
`--aim-output remote` is deliberately unavailable because this project has no
authenticated, physically gated receiver.

## Command line

The desktop app is recommended. For diagnostics or scripted offline video:

```bash
python app.py --cli --help
python app.py --cli --source test.mp4
python app.py --cli --source 0 --capture-size 1280x720 --capture-fps 60
python app.py --cli --source screen:1 --screen-fps 60
python app.py --cli --source screen --screen-region 0,0,1920,1080 --no-preview
```

Press `q`, Escape, close the preview, or press Ctrl+C to stop. Screen-region
coordinates are global desktop coordinates. `--capture-fps` and `--screen-fps`
limit source sampling; they do not manufacture frames or create a second model
queue.

`--inference-size` accepts the legacy square form (`416`) or an explicit
`HEIGHTxWIDTH` tensor shape such as `384x640`. The order is deliberately H×W,
matching NCHW model dimensions; it is not the W×H order used by
`--capture-size`. Both dimensions must be divisible by 32. A static ONNX model
must have been exported at exactly that shape. Use
`python scripts/export_model.py --imgsz 384x640 ...` to create a matching
rectangular OpenVINO export; square export commands remain unchanged.

Experimental far-detail mode is opt-in with
`--detail-crop-size SOURCE_WIDTH_PIXELS`. It keeps the ordinary full-frame pass,
then runs the same static model and input shape on a centered source ROI whose
height is derived from the model's exact H:W aspect, and
conservatively consolidates overlapping, same-class cross-pass duplicates with
one-to-one matching. New unmatched detail detections are admitted only when
their source-space height is at most 96 pixels on a 1080p reference frame;
matched detail boxes may still refine a primary detection at any size. This
conservative gate reduced duplicate/large-object false positives in the local
validation experiment without reducing its measured far recall, but it is not
yet a release claim. The startup log and metrics JSON derive the actual
linear-detail multiplier from the negotiated frame, applied source crop, and
model tensor aspect. For example, a requested width of 768 pixels produces a
768×768 ROI for a square 416 model and an exact-aspect 765×459 ROI for a
384×640 model. Both
are approximately 2.50× denser than their respective full-frame passes on a
1920×1080 source; the exact reported multiplier still comes from the applied
integer geometry rather than a fixed claim. This approximately doubles
inference work. It is off in the current pointer and in an ordinary custom CLI
unless explicitly requested; a future adopted Recommended pointer may bind a
positive width after tournament selection. Fresh profiles and an explicit
Recommended-preset selection use that bound workload, while existing saved
profiles retain their prior detail setting until the user reselects
Recommended. It remains experimental until it passes the release hardware latency and
far-recall gates. It cannot be combined with legacy `--crop-size`, which
replaces the primary full-frame view.

## Build, test, and reinstall

Ordinary source development can install the build requirements after the
common and one runtime-specific file:

```bash
python -m pip install -r requirements-build.txt
python -m unittest discover -s tests
python scripts/validate_release_assets.py --project-root .
```

Published bundles do not reuse that developer environment. Their workflows
create a fresh CPython 3.13.14 environment and install a target-specific file
from `requirements-locks/` with `--require-hashes`; the build helper refuses to
run if the interpreter, complete distribution set, runtime variant, pip
artifact report, installed RECORD payload bytes, or complete package-root file
set differs. See
[docs/DEPENDENCY_LOCKS.md](docs/DEPENDENCY_LOCKS.md) for the exact profiles,
update procedure, and reproducibility boundary.

The preflight verifies the release manifest, labels, required attribution,
static input/output contracts, all OpenVINO IR pairs, all five ONNX graphs, and
the OpenVINO-only INT8 graph.

Linux CPU build and per-user reinstall:

```bash
PROAIM_RUNTIME_VARIANT=cpu ./scripts/build_linux_app.sh
./dist/ProAim/ProAim --runtime-info
./dist/ProAim/install.sh
```

Use `PROAIM_RUNTIME_VARIANT=cuda` or `rocm` only in a clean environment that has
the matching runtime requirements file installed. CUDA and Linux ROCm are
manual builds because their frozen runtime and driver compatibility must be
verified on matching hardware before distribution.

Windows builds must run on Windows:

```powershell
.\scripts\build_windows_app.ps1 -RuntimeVariant cuda
.\scripts\build_windows_app.ps1 -RuntimeVariant directml
.\dist\ProAim\ProAimCLI.exe --runtime-info
```

CI compiles and tests the source on Linux/CPU and Windows/DirectML, runs the
release-asset preflight, exercises CLI help, and checks runtime information.
Version tags run those gates and stage an immutable Linux/DirectML candidate;
they do not create a public release. Publication requires sealed physical
DirectML evidence for both an AMD Radeon RX 6950 XT and an NVIDIA GeForce RTX
5060 Laptop GPU through the protected dual-GPU release workflow. The separate
manual Windows workflow keeps the CUDA build available for independently
qualified attachment. See [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md)
for the reproducible release procedure.

## Dataset, accuracy limits, and model work

The custom detector is derived from the supplied FORT-Cuh v1 COCO archive. Its
noisy source classes were consolidated into one `player` class; `Yourself` was
too sparsely annotated to train reliable identity recognition. The supplied
splits also contain related video/source groups across train, validation, and
test, so reported accuracy is useful for comparing artifacts but optimistic for
new gameplay.

Prepare or retrain only in the separate export environment described by
`requirements-export.txt` and the scripts in `scripts/`. Training and export
dependencies such as PyTorch and Ultralytics are intentionally absent from the
runtime application. Retain each model's `ATTRIBUTION.md` and regenerate
`models/RELEASE-MANIFEST.sha256` when an approved artifact changes.

The FORT-Cuh source metadata identifies FORT-Cuh v1 by Aviles Joseph from
[Roboflow Universe](https://universe.roboflow.com/aviles-joseph/fort-cuh-mji4f)
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the model
attribution files and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License and limitations

Copyright (C) 2026 ProAim contributors.

ProAim is licensed under
[GNU AGPL-3.0-or-later](LICENSE). Bundled third-party libraries and models keep
their own terms, notices, and attribution; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the attribution beside each
FORT-derived model. Qt/PySide license material is included in release bundles.

Current limitations:

- Linux screen capture is X11/Xorg-only; Windows uses DXGI with MSS fallback;
- `player` means a visual player-like character, not an opponent or ally;
- the self-avatar filter can be wrong during occlusion or close overlap;
- the generic COCO presets are not trained for the game clone;
- MAKCU correction is relative and sensitivity dependent;
- remote output is unavailable;
- controller precision is Linux-only and tested with the PXN P5 8K;
- capture-to-result latency includes capture and queue age but excludes Moonlight
  host/network/decode and display scanout; it is not glass-to-glass latency;
- preview service time is backend-dependent application work and excludes
  physical display scanout;
- ProAim has no game-memory, process-injection, anti-cheat, or online-service
  integration.
