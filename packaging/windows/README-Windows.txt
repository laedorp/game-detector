ProAim for Windows
==================

QUICK START

1. Extract the complete ProAim folder. Do not run the EXE inside the ZIP, and
   do not move ProAim.exe away from its adjacent folders.
2. Open ProAim.exe. ProAimCLI.exe is the bundled diagnostic/worker executable;
   keep it beside the app.
3. In Capture source, choose Moonlight / screen capture, a capture card/camera,
   or a video file. Open Moonlight launches the normal Moonlight client.
4. In Detection, start with Game players - Balanced 416 (Recommended). The
   Responsive 416 INT8 preset is OpenVINO-only; Fast 320 is the fallback when
   update rate matters more than small/distant-player accuracy.
5. In Hardware, press Scan hardware, review which providers are usable, and
   apply the selection.
6. Press Start detection or F5. Press Escape or Stop to finish.

The app is self-contained and does not require Python. Keep the preview outside
the captured rectangle to prevent feedback. Windows screen capture does not
have the Linux X11 restriction.

RUNTIME VARIANTS

ProAim-Windows-x64-DirectML.zip is the supported release build for AMD, Intel,
or NVIDIA GPUs through DirectX 12. Intel CPU/GPU/NPU devices may also use
OpenVINO.

ProAim-Windows-x64-NVIDIA-CUDA.zip is an experimental, manually built variant.
It includes CUDA 13/cuDNN 9 user-space libraries and requires a compatible
NVIDIA display driver. It is not published by version tags until a frozen
archive passes real model inference on representative NVIDIA hardware. Use the
DirectML release unless you are validating that CUDA candidate. BUILD-INFO.json
identifies the exact bundle variant and source commit.

To print the runtimes and execution providers available on this PC, open a
terminal in this folder and run:

    ProAimCLI.exe --runtime-info

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
because no authenticated, physically gated receiver is included.

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
