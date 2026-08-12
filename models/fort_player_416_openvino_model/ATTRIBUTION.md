# FORT player model attribution (416)

This OpenVINO model (`fort_player_416.xml` / `fort_player_416.bin`) is a
one-class `player` detector fine-tuned from Ultralytics YOLO26n on the prepared
**FORT-Cuh v1** dataset.

## Training data

The training data is derived from **FORT-Cuh v1**, exported from Roboflow
Universe on August 9, 2026 and provided by Roboflow user Aviles Joseph:

https://universe.roboflow.com/aviles-joseph/fort-cuh-mji4f

The source dataset identifies its license as **CC BY 4.0**:

https://creativecommons.org/licenses/by/4.0/

The full preparation record — label remapping, exclusions, box clipping, the
source archive hash, and conversion counts — is retained in
`datasets/fort_cuh_player/ATTRIBUTION.md` and `manifest.json`.

## Base model

Fine-tuned from Ultralytics YOLO26n. Ultralytics code and models are offered
under AGPL-3.0 and Enterprise licensing; review those terms before
redistributing this model.

## Training run

- Run: `runs/fort_cuh/yolo26n_416_player_v1`
- 20 epochs, image size 416, batch 16, seed 0, CPU
- Reproducibility record: `runs/fort_cuh/yolo26n_416_player_v1/reproducibility.json`

Reported metrics (single class `player`):

| split | images | instances | precision | recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| validation | 593 | 894 | 0.732 | 0.647 | 0.685 | 0.332 |
| supplied test | 321 | 523 | 0.770 | 0.639 | 0.723 | 0.357 |

Measured OpenVINO CPU latency on an Intel Core i7-10850H: 11.10 ms mean,
12.82 ms p95 (batch 1, `LATENCY` hint, one stream).

## Limitations

The supplied validation and test splits are **not independent** of the training
sources: repeated original basenames and frames from shared source videos occur
across the split boundaries. These metrics may therefore be optimistic and
should not be read as clean generalization estimates.

This model learns a visual `player` class only — not identity, not team
membership. Expect misses on unseen skins, poses, effects, distances, and camera
angles, and false positives on similar shapes.

Retain this notice when sharing this model or anything derived from it. This
notice describes the stated licenses of the inputs; it is not legal advice.
