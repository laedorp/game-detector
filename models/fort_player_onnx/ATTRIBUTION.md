# FORT player model attribution

This OpenVINO model (`fort_player.xml` / `fort_player.bin`) is a one-class
`player` detector fine-tuned from Ultralytics YOLO26n on the prepared
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

- Run: `runs/fort_cuh/yolo26n_320_player_v2`
- 20 epochs, image size 320, batch 16, seed 0, CPU
- Reproducibility record: `runs/fort_cuh/yolo26n_320_player_v2/reproducibility.json`

Reported metrics (single class `player`):

| split | images | instances | precision | recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| validation | 593 | 894 | 0.705 | 0.598 | 0.649 | 0.298 |
| supplied test | 321 | 523 | 0.707 | 0.562 | 0.655 | 0.332 |

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
