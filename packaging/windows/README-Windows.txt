Game Detector for Windows
=========================

1. Extract the entire GameDetector folder. Do not run the EXE inside the ZIP.
2. Open GameDetector.exe. Keep GameDetectorCLI.exe and all adjacent folders together.
3. Under AI detection + aim, press Refresh devices. Choose AUTO, CPU, GPU, or NPU
   only when the status line reports that device. GPU/NPU availability depends on
   installed OpenVINO-compatible hardware drivers.
4. Choose People - Balanced 416 for range/detail or People - Fast 320 for latency.
5. Connect the MAKCU serial/control USB interface to this PC. Select its COM port,
   choose Right, and run Verify Right Mouse with capture stopped.
6. Start capture + AI preview and confirm boxes and the selected head marker before
   holding Right Mouse.

The app is self-contained and does not require a separate Python installation.
The MAKCU output loop uses only the latest detector state and stops on stale input.
The COCO person model cannot distinguish enemies from allies.