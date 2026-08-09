# Repository guidance

- This application is limited to video capture, offline object detection, preview, and performance measurement for offline/single-player experimentation.
- Do not add input automation, aiming/firing, anti-cheat bypasses, process injection, memory reading, or game modification.
- Keep the deployed runtime small: OpenCV, OpenVINO, NumPy, and MSS. Keep Ultralytics/PyTorch in the export-only environment.
- Preserve latest-frame-only semantics for live sources; never introduce an unbounded or FIFO frame queue.
- Default inference to OpenVINO `CPU`, batch 1, a single synchronous request, and the `LATENCY` performance hint.
- Keep capture, preprocessing, detection, postprocessing, rendering, and metrics modular so sources and models remain replaceable.
- Treat capture-card, Moonlight, compositor, and driver buffering as outside application-visible latency unless a trustworthy upstream timestamp exists.
