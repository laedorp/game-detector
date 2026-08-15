ProAim for Windows
==================

QUICK START

1. Extract the complete ProAim folder. Do not run the EXE inside the ZIP, and
   do not move ProAim.exe away from its adjacent folders.
2. Open ProAim.exe. ProAimCLI.exe is the bundled diagnostic/worker executable;
   keep it beside the app.
3. In Capture source, choose Moonlight / screen capture, a capture card/camera,
   or a video file. Open Moonlight launches the normal Moonlight client.
4. In Detection, start with the Game players preset marked Recommended. Its
   exact static size and optional detail ROI width are bound to this build.
   Responsive 416 INT8 is
   OpenVINO-only; Fast 320 is the fallback when update rate matters more than
   small/distant-player accuracy.
5. A genuinely fresh profile scans automatically. ProAim applies a discrete
   GPU only when exactly one ready choice has a safe binding. Existing saved
   choices are preserved. If the result is ambiguous, open Hardware, press
   Scan hardware, review the providers, and apply the intended selection;
   starting on CPU otherwise requires explicit confirmation.
6. Press Start detection or F5. Press Escape or Stop to finish.

Existing profiles are not silently opted into a newly selected detail workload.
To adopt the complete current Recommended workload, explicitly reselect that
preset and review the shown detail ROI maximum width before starting.

The app is self-contained and does not require Python. Keep the preview outside
the captured rectangle to prevent feedback. Windows screen capture does not
have the Linux X11 restriction. It prefers DXcam's low-latency Desktop
Duplication backend and automatically falls back to MSS if DXGI cannot start;
the Capture settings log identifies the backend actually in use.

RUNTIME VARIANTS

ProAim-Windows-x64-DirectML.zip is the supported release build for AMD, Intel,
or NVIDIA GPUs through DirectX 12. Intel CPU/GPU/NPU devices may also use
OpenVINO. **Scan hardware** binds an exactly matched DirectML GPU by its DXGI
adapter index; confirm that named GPU is active in Task Manager during the test.

ProAim-Windows-x64-NVIDIA-CUDA.zip is an experimental, manually built variant.
It includes CUDA 13/cuDNN 9 user-space libraries and requires a compatible
NVIDIA display driver. It is not published by version tags until a frozen
archive passes real model inference on representative NVIDIA hardware. Use the
DirectML release unless you are validating that CUDA candidate. BUILD-INFO.json
identifies the exact bundle variant and source commit.

To print the runtimes and execution providers available on this PC, open a
terminal in this folder and run:

    ProAimCLI.exe --runtime-info

For a reproducible performance report of the exact frozen model/runtime, use
the bundled helper with the DXGI index shown by Scan hardware (replace 1
below), then confirm the named GPU is active in Task Manager while it runs:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Qualify-ProAimGpu.ps1 -Provider DirectML -AdapterIndex 1

Keep the resulting evidence directory together when reporting performance.
The helper reads and hash-verifies the release-default model, labels, and input
shape from BUILD-INFO.json. Provider activation alone does not identify the
physical adapter on a hybrid-graphics laptop.

The extracted bundle also includes a fail-closed, one-command evidence helper.
It refuses to overwrite an evidence directory, hashes BUILD-INFO.json, the
frozen executables, labels, and exact model artifacts, and only publishes its
evidence directory after every selected command succeeds. For DirectML:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Qualify-ProAimGpu.ps1 -Provider DirectML -AdapterIndex 1 -RunLive

That DirectML command is the Windows path for AMD cards such as the Radeon RX
6950 XT, as well as Intel and NVIDIA GPUs; Windows does not require ROCm. Use
the adapter index reported by Scan hardware, not a guessed Task Manager number.

For the experimental CUDA bundle:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Qualify-ProAimGpu.ps1 -Provider CUDA -RunLive

The optional live A/B is bounded at 1,000 frames or 60 seconds per mode and
uses the exact release-default model, labels, and input shape recorded in the
bundle's BUILD-INFO.json. The release helper intentionally does not accept an
alternate model: adopt a new launcher default and rebuild the bundle before
qualifying it. Omit -RunLive for model-only evidence. The helper also reads
`detail_crop_size_source_pixels` from that same BUILD-INFO contract and refuses
an override. A positive value is the maximum centered source-ROI width; ROI
height is derived from the model H:W. For a 384x640 model, a request of 768 on
a 1920x1080 source produces the exact-aspect 765x459 ROI (about 2.5x effective
linear magnification), and both live reports bind the applied geometry and its
full-pipeline cost. If the original downloaded ZIP is still
available, -BundleArchivePath also records its SHA-256. The helper validates
provider use with CPU graph-node fallback disabled, but deliberately leaves
physical-GPU confirmation pending. Follow its Task Manager prompt and complete
the template inside the finished evidence directory; software provider
activation alone is not a physical-GPU claim.

The command's execution-policy override applies only to that PowerShell
process; it does not change the machine's saved policy.

LIVE PIPELINE A/B (EXACT FROZEN APP)

The model-only command above does not measure screen capture or preview. With a
Moonlight stream active, add `-RunLive` to the same helper command. It runs one
no-preview pass and one 15 FPS preview pass with the exact BUILD-INFO default;
it does not rely on a model filename or shape copied into this document.

ProAim will not overwrite existing evidence. While each run is active,
confirm in Task Manager that the intended named physical GPU is carrying load,
not an unintended integrated GPU. Keep both JSON files, that confirmation, and
BUILD-INFO.json together. The report includes the model hash, active provider,
DXGI adapter, actual capture backend/fallback, overwrite/failure counts,
preview statistics, and mean/p50/p95/p99 timings. It excludes pairing keys and
auto-detected controller, activation-device, and serial paths.

MODEL SCOPE

The default custom model detects one visual player class. It cannot determine
enemy, ally, identity, or team. The third-person self filter is a conservative
screen-position heuristic; verify its orange preview outline. Generic COCO
models are included only as fallbacks. Model accuracy and benchmark details are
in docs/MODEL_BENCHMARKS.md in the source repository.

OPTIONAL MAKCU OUTPUT

MAKCU output is local, offline, physically gated relative mouse correction. It
does not synthesize clicks, read game memory, or interact with anti-cheat
software.

1. Connect the MAKCU serial/control side to this computer, its output side to
   the other PC, and the physical mouse through the board.
2. With detection stopped, select the exact COM port and activation button.
3. Enter an explicit Target label, such as player.
4. Complete Verify ... Mouse. A valid binding requires a physical press and
   release and is tied to that COM port and button.
5. Start detection and check the preview before using the physical gate.

Output returns to neutral immediately when the gate is released, detections go
stale, the self filter is unsafe, or ProAim stops. Remote output is unavailable
because no authenticated, physically gated receiver is included. One
uninterrupted MAKCU activation is capped at 10 seconds as a missed-release
safeguard; release and press the gate again to continue.

CONTROLLERS

For normal controller play, connect the controller to the Moonlight computer
and let Moonlight forward it directly. Controller precision is Linux-only and
is unavailable in this Windows build.

LICENSE

ProAim is licensed under GNU AGPL-3.0-or-later. LICENSE and
THIRD_PARTY_NOTICES.md are included in this folder. Bundled FORT-derived player
models retain CC BY 4.0 attribution beside their model files, and Qt/PySide
license material is in the licenses folder. This software is provided without
warranty.
