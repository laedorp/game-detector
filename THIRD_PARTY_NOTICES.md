# Third-party notices

Copyright (C) 2026 ProAim contributors.

ProAim is distributed under the GNU Affero General Public License,
version 3 or later. See `LICENSE`. The release bundles also contain separate
third-party components under their own licenses. Release bundles retain their
installed package metadata and license payloads where supplied by the wheel;
the Qt LGPL/GPL texts and model attributions are also copied explicitly.

Principal runtime components include:

| component | license | project |
| --- | --- | --- |
| Python runtime | Python Software Foundation License | https://www.python.org/ |
| NumPy | BSD-3-Clause and bundled third-party notices | https://numpy.org/ |
| OpenCV / opencv-python | Apache-2.0 and bundled third-party notices | https://opencv.org/ |
| OpenVINO | Apache-2.0 | https://github.com/openvinotoolkit/openvino |
| ONNX Runtime | MIT and bundled third-party notices | https://onnxruntime.ai/ |
| PySide6 / Qt for Python | LGPL-3.0-only, GPL-2.0-only, or GPL-3.0-only | https://doc.qt.io/qtforpython-6/ |
| MSS | MIT | https://github.com/BoboTiG/python-mss |
| pySerial | BSD-3-Clause | https://github.com/pyserial/pyserial |
| python-evdev (Linux) | BSD-3-Clause | https://python-evdev.readthedocs.io/ |

Because the pySerial 3.5 wheel does not include the full license payload,
`packaging/licenses/pyserial-3.5-BSD-3-Clause.txt` reproduces the upstream
pySerial 3.5 source distribution's `LICENSE.txt` verbatim. It is copied into
release bundles under `licenses/third-party/pyserial/`.

The bundled player models are derived from FORT-Cuh v1, whose supplied
metadata identifies a CC BY 4.0 license. Their required attribution and data
quality limitations are retained next to each model in
`models/fort_player*_openvino_model/ATTRIBUTION.md`.

The exported YOLO model artifacts were produced with Ultralytics tooling and
are distributed in this AGPL-3.0-or-later project. Each generic YOLO IR and
ONNX bundle directory retains the matching export `metadata.yaml`, including
tool version, stated license, task, image size, and class names. Ultralytics
also offers a separate Enterprise license; review that option before
closed-source commercial distribution.

This notice is a practical inventory, not legal advice. If a bundled component
includes a more specific license or notice, that component's own text controls.
Linux and Windows public bundles must be built on the documented release
runner, then reviewed for the final native-library closure; a local development
bundle is not automatically a portable redistribution artifact.
