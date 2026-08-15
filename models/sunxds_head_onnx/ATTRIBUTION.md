# SunXDS 0.8.0 head-localization model

`sunxds_0.8.0.onnx` is a derived ONNX export of the public SunXDS 0.8.0
checkpoint by SunOne. The model detects two classes, `player` and `head`, at a
static 320 x 320 input size. It is used only as a bounded second-stage head
localizer after ProAim's primary player detector has selected and safety-checked
an identity.

- Upstream project: <https://github.com/SunOner/sunone_aimbot>
- Pinned checkpoint commit: `ec17c7d89ea2c940b20081d27da633f4c7491655`
- Pinned checkpoint path: `models/sunxds_0.8.0.pt`
- Upstream checkpoint SHA-256:
  `de8a0cf0d3911751c65193b7b487f830fd3c6cbb53866e23bf8b8be7c33b4baf`
- Derived ONNX SHA-256:
  `93264ec61b86b8459ef64c85a31ab3da294327ee1f95337076e57d8af24bb192`
- Upstream release note:
  <https://www.patreon.com/sunone/posts/sunxds-0-8-0-142572560>

The upstream repository is distributed under the MIT License. The checkpoint
and derived ONNX embed Ultralytics' `AGPL-3.0` model metadata, so ProAim treats
the model artifact as AGPL-3.0. ProAim itself is distributed under
AGPL-3.0-or-later; see the repository `LICENSE` and `THIRD_PARTY_NOTICES.md`.
The upstream author describes this as a free test model with known
shortcomings; this attribution does not imply an accuracy guarantee or an
endorsement by the author.

## Reproducible export

The checked-in ONNX was exported with:

- `ultralytics==8.3.217`
- `torch==2.13.0+cpu`
- `onnx==1.22.0`
- Python API equivalent:

```python
from ultralytics import YOLO

YOLO("sunxds_0.8.0.pt").export(
    format="onnx",
    imgsz=320,
    batch=1,
    half=False,
    dynamic=False,
    simplify=False,
    opset=17,
    nms=False,
    device="cpu",
)
```

Only the statically inspected, derived ONNX is loaded at runtime. The upstream
PyTorch pickle is not bundled or loaded by ProAim.
