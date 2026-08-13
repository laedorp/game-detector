# FORT player model attribution (416 INT8)

This OpenVINO model (`fort_player_416_int8.xml` /
`fort_player_416_int8.bin`) is an INT8 post-training quantization of the
one-class 416-pixel `player` detector fine-tuned from Ultralytics YOLO26n on
the prepared **FORT-Cuh v1** dataset.

## Training data and base model

The training data is derived from **FORT-Cuh v1**, exported from Roboflow
Universe on August 9, 2026 and provided by Roboflow user Aviles Joseph:

https://universe.roboflow.com/aviles-joseph/fort-cuh-mji4f

The source dataset identifies its license as **CC BY 4.0**:

https://creativecommons.org/licenses/by/4.0/

The full preparation record -- label remapping, exclusions, box clipping,
source archive hash, conversion counts, and known split leakage -- is retained
in `datasets/fort_cuh_player/ATTRIBUTION.md` and `manifest.json`.

The detector was fine-tuned from Ultralytics YOLO26n. Ultralytics code and
models are offered under AGPL-3.0 and Enterprise licensing; review those terms
before redistributing this model.

## Training and quantization provenance

- Training run: `runs/fort_cuh/yolo26n_416_player_v1`
- Training: 20 epochs, image size 416, batch 16, seed 0, CPU
- Source OpenVINO graph: `models/fort_player_416_openvino_model/fort_player_416.xml`
- Source XML SHA-256: `6779ead62c3ace7d4de4b5a43dd3cb2cf8c6be84fa1399cd40db1eaf4b8c6450`
- Source BIN SHA-256: `c6d908bce89df473a796c8ad72690872f37e776eb41a70686095baeb77e72895`
- Quantizer: `scripts/quantize_model.py`, SHA-256
  `f3c24023c74f76763915622497caa2a955f987d8fc75fd1fdb52f876b9893a97`
- Quantization: NNCF 3.3.0 post-training quantization, `MIXED` preset,
  OpenVINO 2026.3.0
- Calibration: 300 evenly sampled images from the prepared validation split,
  letterboxed to 416x416 with the runtime's BGR-to-RGB normalization
- Calibration path-list SHA-256:
  `4f512b63522766a1be203a4bd829ab4c81aea315de66409e8f814bb4688a36ba`
- Calibration content-stream SHA-256:
  `7fa00625d7ef446f232df2736c90b8bcd1c3a8c98f3a9b9c4a27800d6ebf086a`
- INT8 XML SHA-256: `e29876a30238511dae38382d358cf36592023f9183c2d35bd2c5a1714b71ee84`
- INT8 BIN SHA-256: `4b6d2893f334c0406092d7ba2011d41f8b5ae6f36fdd3ce6cc9941fcecbd2a20`

On August 12, 2026, running the recorded quantization command again produced
byte-identical XML and BIN artifacts. Exact tool, input, and output hashes are
also embedded in `metadata.yaml`.

## Verified deployment metrics

Ultralytics 8.4.116 evaluated the deployed OpenVINO graph at batch 1 and image
size 416 on an Intel Core i7-10850H CPU. The INT8 graph was compared with the
FP32 OpenVINO graph using identical inputs and settings.

| model | split | images | instances | precision | recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 416 FP32 | validation | 593 | 894 | 0.7343 | 0.6584 | 0.6976 | 0.3336 |
| 416 INT8 | validation | 593 | 894 | 0.7023 | 0.6711 | 0.6681 | 0.3183 |
| 416 FP32 | supplied test | 321 | 523 | 0.7903 | 0.6501 | 0.7202 | 0.3526 |
| 416 INT8 | supplied test | 321 | 523 | 0.7367 | 0.6654 | 0.7065 | 0.3371 |

On the supplied test split, INT8 loses 0.0137 absolute mAP50 and 0.0155
mAP50-95 while gaining 0.0153 recall. In the application-equivalent CPU
benchmark it reduced mean inference latency from 16.56 ms to 11.42 ms and the
combined model artifact size from 9.91 MB to 3.25 MB. Full methodology and
percentiles are in `docs/MODEL_BENCHMARKS.md`.

## Limitations

The supplied validation and test splits are **not independent** of the training
sources: related source frames and augmented images occur across split
boundaries. These metrics are optimistic and are not clean generalization
estimates. Calibration also uses the leaky supplied validation split; the
supplied test split was not used for quantization.

This model learns a visual `player` class only -- not identity and not team
membership. Expect misses on unseen skins, poses, effects, distances, and
camera angles, and false positives on similar shapes. INT8 gains depend on the
processor; benchmark it on the deployment hardware before choosing it.

Retain this notice when sharing this model or anything derived from it. This
notice describes the stated licenses of the inputs; it is not legal advice.
