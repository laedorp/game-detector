# Game Detector

Low-latency object detection for offline/single-player game footage. The laptop captures video, runs a small YOLO detector through OpenVINO, and optionally displays boxes and timing statistics. A separate, Linux-only Controller precision mode can reshape right-stick movement physically produced on a supported controller while LT is held. Detection results are never connected to controller, mouse, keyboard, UDP, or network input.

## One-click desktop app

The recommended interface is the **Game Detector** desktop app. It lets you choose Moonlight/screen capture, a USB camera or capture card, or a video file; adjust detection settings; and start or stop detection without a terminal. It remembers the last settings and shows the detector log in the launcher.

This workstation already has the verified Linux build installed in the application menu. Open the launcher and:

1. Choose **Moonlight / screen**, **Camera / capture card**, or **Video file**.
2. Pick the source options. For Moonlight, the **Open Moonlight** button starts its normal client UI.
3. Leave **Fortnite-style players (Recommended)** selected for the game clone, or choose **General objects (COCO fallback)**. Leave `CPU` and `320` inference size selected for the fastest starting point.
4. In **Detection settings → Third-person view**, leave **Ignore my on-screen character** enabled and choose where your character normally appears.
5. Click **Start detection**. Click **Stop**, press Escape, or close the preview to finish.

The preview is a separate window. When capturing Moonlight, keep it outside the captured monitor/rectangle so it does not appear inside its own input.

To run the source launcher instead of the installed build:

```bash
./run_gui.sh
```

Moonlight screen capture currently requires an X11/Xorg session. The launcher detects Wayland before starting and explains how to switch sessions; camera and video modes still work on Wayland. This laptop now has Plasma's X11 session installed.

Stage 2 supports:

- USB cameras and capture cards through OpenCV
- video files
- a Moonlight window or monitor through X11 screen capture
- YOLO26 end-to-end output and traditional YOLOv8/YOLO11 output
- centered cropping, letterboxing, confidence filtering, and class-aware NMS when needed
- an adjustable third-person self-avatar exclusion heuristic
- a separate Linux-only, LT-held manual controller precision curve for the PXN P5 8K
- latest-frame-only live capture
- preview, FPS, per-stage latency, moving statistics, and clean shutdown

Launcher validation and lifecycle tests are included. Full detector test coverage and static checks remain Stage 3; JSON benchmark mode remains Stage 4.

## Architecture

Live device and screen sources write to a capacity-one mailbox. If inference is slower than capture, an unread frame is replaced instead of queued. Video files are decoded sequentially so test footage is not raced to EOF.

```text
capture → optional center crop → letterbox/normalize → OpenVINO
        → confidence/NMS → source-coordinate boxes → optional self filter
        → metrics/preview
```

Detection uses OpenCV, OpenVINO, NumPy, and MSS. Linux Controller precision additionally uses `evdev` and the kernel's `uinput` interface; that worker has no detector, capture, or network dependency. Ultralytics and PyTorch are isolated to dataset training and model export. The packaged offline runtime excludes OpenVINO's optional analytics client and uses OpenVINO's no-op telemetry fallback.

## Runtime setup

OpenVINO 2026 supports Python 3.10 through 3.14 on its supported Linux distributions.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the runtime and detected devices:

```bash
python -c "from openvino import Core; print(Core().available_devices)"
```

`CPU` should be present. `GPU` appears only when the Intel graphics driver/OpenCL stack is installed and available to the current user.

Running the GUI from source also requires Tk. On CachyOS/Arch, install it with `sudo pacman -S tk`; the packaged desktop build already carries its own Tk runtime.

On Linux, `requirements.txt` also installs `evdev` for Controller precision. Detection remains usable if the operating-system virtual-controller interface is not ready; the Controller precision tab reports that readiness separately.

## Build or reinstall the desktop app

The Linux build is a self-contained PyInstaller one-folder application. It includes Python, Tk, OpenVINO, OpenCV, MSS, Linux `evdev` support, the generic COCO fallback, the custom FORT player model, labels, and the custom model's attribution. Build it with:

```bash
source .venv/bin/activate
python -m pip install -r requirements-build.txt
./scripts/build_linux_app.sh
```

The result is `dist/GameDetector/GameDetector`. Install or update the current user's application-menu entry with:

```bash
./dist/GameDetector/install.sh
```

Keep the complete `GameDetector` directory together when moving it. PyInstaller builds are platform-specific: run `scripts/build_windows_app.ps1` on Windows to create `dist\GameDetector\GameDetector.exe`; a Linux build cannot be renamed or converted into a Windows `.exe`.

Release builds deliberately stop with a list of missing files unless both model pairs, both label files, and `models/fort_player_openvino_model/ATTRIBUTION.md` exist. The Windows build includes both detectors but omits Linux-only `evdev`; Controller precision is disabled there.

## Export the starter model

YOLO26n is downloaded and converted once. Inference is offline after the `.xml` and `.bin` files exist.

The current workspace already contains the exported 320×320 model. Run the following only when provisioning a fresh checkout or replacing the model; the exporter deliberately refuses to overwrite an existing output directory.

Use a separate environment so PyTorch and Ultralytics are not part of the runtime installation:

```bash
python3 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-export.txt
python scripts/export_model.py
deactivate
```

This creates `models/yolo26n_openvino_model/yolo26n.xml` and its weights. The default export is static, batch 1, and 320×320. Use `--imgsz N --output PATH` to create another fixed input size.

YOLO26n is trained on the 80 COCO classes listed in `models/coco80.txt`. Game-specific objects will normally need a custom-trained model. Ultralytics code and models are offered under AGPL-3.0 and Enterprise licensing; review that before redistribution.

## Custom FORT player model

The game-specific model is a one-class `player` detector derived from the supplied FORT-Cuh v1 COCO archive. The preparation script safely reads the archive, clips boxes to image bounds, excludes images without a retained full-player box, and consolidates the inconsistent source vocabulary. Source labels `0`, `Fortnite`, `Player`/`player`, `bots`, `enemy`, `hello`, `people`, and `person` become `player`; `head`, `body`, `ally`, and `Yourself` annotations are excluded.

Prepare the pinned archive and run a small training smoke test before a full CPU training run:

```bash
.venv-export/bin/python scripts/prepare_fort_cuh.py \
  /path/to/FORT-Cuh.v1i.coco.zip \
  --output datasets/fort_cuh_player
.venv-export/bin/python scripts/train_fort_model.py --smoke-test
.venv-export/bin/python scripts/train_fort_model.py
```

The prepared dataset records its archive hash, conversion counts, exclusions, clipping, and conservative cross-split leakage checks in `datasets/fort_cuh_player/manifest.json`. Retain the generated attribution notice when exporting or sharing a derived model.

For the pinned archive, preparation retained 2,459 train images with 3,571 boxes, 593 validation images with 894 boxes, and 321 test images with 523 boxes. It found 26 original-basename groups and 26 explicit video-sequence groups crossing split boundaries, involving 616 retained video frames; it found no cross-split byte-identical image groups. These are conservative filename-based checks, not proof that every remaining frame is independent.

The source dataset states that it is FORT-Cuh v1 by Aviles Joseph from [Roboflow Universe](https://universe.roboflow.com/aviles-joseph/fort-cuh-mji4f), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Repeated original basenames and frames apparently originating from the same source video occur across the supplied train/validation/test assignments. They are reported rather than guessed away, so validation and test metrics are not independent and may be optimistic.

After training, review the reported metrics and use the exact export command printed by `train_fort_model.py`. A release requires `fort_player.xml`, its matching `fort_player.bin`, `models/fort_player.txt`, and the retained attribution at `models/fort_player_openvino_model/ATTRIBUTION.md`; the packaging preflight lists anything still missing.

This model learns a visual `player` class, not identity or team membership. The source labels are noisy, `Yourself` is too sparsely annotated to train a reliable identity class, and the data cannot prove that a detected player is an opponent. Expect misses on unseen skins, poses, effects, distances, or camera views and false positives on similar shapes. Use the visible preview to validate behavior; the separate self-avatar filter remains a conservative screen-position heuristic.

## Command-line run

Activate the runtime environment first:

```bash
source .venv/bin/activate
```

Video file:

```bash
python main.py --source test.mp4
```

USB camera or capture card at device index 0:

```bash
python main.py --source 0 --capture-size 1280x720 --capture-fps 60
```

Common latency options:

```bash
python main.py --source 0 --inference-size 320 --confidence 0.5
python main.py --source 0 --crop-size 640
python main.py --source 0 --no-preview
```

Press `q`, Escape, close the preview window, or press Ctrl+C to stop.

## Third-person self-avatar filter

Enable **Ignore my on-screen character** in the desktop launcher and choose **Left of center**, **Center**, or **Right of center** to match the usual avatar position. This workstation's current Moonlight profile is preconfigured with the option enabled. The preview shows the active orange `SELF ANCHOR ZONE` and reports whether one box was ignored on the current frame.

This is deliberately a conservative screen-position heuristic, not identity or team recognition. It first locks onto one persistent player-like box for three frames, then removes at most that one box when it is at least 28% of frame height, at least 6% of frame width, and its bottom-center reaches the lowest 10% of the configured horizontal area. Material track jumps are left visible for three frames before relocking. If acquisition or ongoing track association is ambiguous, it leaves every box visible. The preview outlines the exact suppressed box in orange as `IGNORED SELF?`. A close overlapping opponent can still inherit the track, so verify that outline during close encounters and disable the option if the camera changes substantially.

The equivalent CLI form is:

```bash
python main.py --source screen --ignore-self \
  --self-zone-left 0.18 --self-zone-width 0.34 --self-zone-height 0.10
```

The CLI filter is off unless `--ignore-self` is supplied. The launcher presets select the normalized geometry for you.

The included COCO model uses the label `person`. A custom model's self-avatar class should use a recognizable label such as `player`, `avatar`, `character`, or `human`. Classes explicitly named `enemy`, `opponent`, `npc`, or `bot` are never eligible for self filtering, nor are unrelated classes such as vehicles.

## Moonlight/Sunshine source

The intended layout is:

```text
gaming PC: game + Sunshine encode
                 ↓ network
laptop: Moonlight hardware decode + screen capture + OpenVINO detection
```

Stage 2 screen capture uses MSS's X11/XCB backend. Log into an Xorg/X11 desktop session; native Wayland capture requires the XDG ScreenCast portal and PipeWire and is not implemented yet.

For best behavior, run Moonlight borderless on a dedicated monitor. A representative Moonlight CLI invocation is:

```bash
moonlight stream MAIN-PC "Desktop" \
  --display-mode borderless \
  --resolution 1920x1080 \
  --fps 60 \
  --video-decoder hardware \
  --no-hdr
```

Confirm the exact executable and options with `moonlight stream --help`; package names can differ by distribution.

Capture the first physical monitor reported by MSS:

```bash
python main.py --source screen:1 --screen-fps 60
```

Or capture only the Moonlight rectangle, using desktop coordinates:

```bash
python main.py --source screen \
  --screen-region 0,0,1920,1080 \
  --screen-fps 60
```

`--screen-region` uses global desktop coordinates and overrides monitor selection.

On a one-monitor laptop, keep the detector preview outside the captured rectangle or add `--no-preview`; otherwise its window can become part of the captured image. On a two-monitor setup, place Moonlight and the detector preview on different monitors.

Moonlight screen capture occurs after host capture, encode, network transport, laptop decode, and composition. The application's reported capture time is therefore screen-grab time, and its pipeline latency is not true gaming-PC-to-display latency. MSS currently requires an X11 `DISPLAY`; on Wayland, use an Xorg session for this version.

### Controller through Moonlight

For ordinary 1:1 play, connect or pair the physical controller with the laptop before starting the Moonlight stream. Moonlight forwards that gamepad directly to Sunshine; detection does not participate in the input path.

Because the detector preview may receive focus, open **Moonlight Settings → Gamepad** and enable **Process gamepad input when Moonlight is in the background**. If a game ignores a controller connected after launch, also try **Force gamepad #1 always connected**. For ordinary controller play, disable Moonlight's Start-button gamepad mouse emulation so holding Start cannot unexpectedly switch modes. Test with USB first; Bluetooth can be added after the basic path works. On a Windows Sunshine host, verify **Sunshine → Input → Controller** and install/reboot for Sunshine's virtual gamepad driver if prompted.

The launcher includes a **Controller help…** button with these checks.

### Linux Controller precision

Controller precision is an optional Linux-only physical-to-virtual controller proxy for the tested **PXN P5 8K** (`36e6:3016`). It temporarily takes exclusive ownership of that physical controller so Moonlight receives only one virtual copy. While you physically hold LT, it applies the selected radial curve to right-stick movement you physically produce; release LT for normal 1:1 movement. It does not choose a direction, track a box, keep a crosshair on an object, or react to detection output.

Before first use:

1. Connect the PXN P5 8K to the laptop, open **Controller precision**, and press **Refresh**.
2. Confirm that `/dev/uinput` exists and is writable by the signed-in desktop user. If it is missing, run `sudo modprobe uinput`, then press **Refresh**. If access is denied, configure a narrow administrator-managed `uaccess` permission for `/dev/uinput`; do not run the whole app as root.
3. Press **Verify LT + right stick…**. When prompted, repeatedly squeeze LT while keeping other controls still, then move only the right stick in full circles. This verification is read-only and binds the observed mapping to that controller.
4. Choose **Gentle**, **Balanced**, or **Strong**, then press **Start controller precision**.
5. Only after the status says it is active, open Moonlight and begin the stream. Starting first lets Moonlight enumerate the virtual controller instead of the temporarily grabbed physical one.

Stop precision from the launcher before unplugging the controller. The worker releases buttons, neutralizes axes, closes the virtual device, and returns the physical controller on normal stop or a handled failure. If controller events are dropped, it stops instead of guessing state.

This path is intentionally isolated from capture and detection: it does not import video or detector modules, listen on UDP, accept target coordinates, or send network commands. The controller remains user-directed at all times. The coordinate-streaming sample in the project discussion is not part of this application and is not connected to Moonlight or `uinput`.

## Timing definitions

- `capture`: duration of OpenCV `read()` or MSS screen grab
- `queue age`: time between delivery by the capture backend and inference processing
- `pre`: crop, resize/letterbox, channel/layout conversion, and normalization
- `infer`: synchronous OpenVINO call
- `post`: confidence filtering, optional NMS, and coordinate restoration
- `pipeline`: capture-call start through completed detections
- `draw`: boxes, labels, and the preview overlay
- `display`: `imshow`, GUI event handling, and window-close polling; not monitor scanout
- `skipped`: frames overwritten in the application's one-frame mailbox

Driver, capture-card, Moonlight, compositor, and display buffering can exist before the timestamps visible to this application. The skipped count also cannot reveal frames lost upstream.

## CLI

```bash
python main.py --help
```

The command-line default remains the generic `models/yolo26n_openvino_model/yolo26n.xml` with `models/coco80.txt`; its inference device is `CPU` and inference size is 320. Fresh desktop-launcher settings instead select **Fortnite-style players (Recommended)** as an atomic model-and-label preset.

The packaged generic fallback is `models/yolo26n_openvino_model/yolo26n.xml` with `models/coco80.txt`. The game-specific pair is `models/fort_player_openvino_model/fort_player.xml` with `models/fort_player.txt`. Keep each `.xml` beside its matching `.bin` and select the label file belonging to that model.

`--output-format auto` treats an `N×6` output as YOLO26 end-to-end detections. A traditional YOLOv8/YOLO11 model trained for exactly two classes can have the same outer shape; run it with `--output-format traditional` so its two class-score columns receive NMS.

## Current limitations

- Moonlight capture is X11-only and coordinate based; it does not find a window by title.
- The custom model recognizes a noisy one-class visual `player` concept; it cannot determine opponent, ally, or identity, and its supplied validation/test splits overlap training sources.
- The generic COCO fallback may not recognize game-specific players.
- The self-avatar filter uses box geometry and cannot prove player identity or team membership.
- Controller precision is Linux-only, currently targets the PXN P5 8K mapping, requires readable controller access plus writable `/dev/uinput`, and only reshapes physical LT/right-stick input.
- No JSON benchmark mode yet.
- No true glass-to-glass latency measurement.
- No detection-driven or coordinate-driven input, automated aiming, game-memory access, or anti-cheat functionality.

Relevant upstream documentation: [OpenVINO device selection](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html), [Ultralytics OpenVINO export](https://docs.ultralytics.com/integrations/openvino/), [YOLO26 end-to-end output](https://docs.ultralytics.com/guides/end2end-detection/), [MSS GNU/Linux capture](https://python-mss.readthedocs.io/stable/usage.html#gnu-linux), and [Moonlight setup](https://github.com/moonlight-stream/moonlight-docs/wiki/Setup-Guide).
