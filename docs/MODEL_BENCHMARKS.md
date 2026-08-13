# Player model accuracy and latency

This report records the August 12, 2026 audit of the three completed custom
FORT-Cuh player detector artifacts. It separates detector quality from runtime
latency: accuracy was measured across every image in a named split, while
latency was measured from preloaded images so storage, capture, Moonlight,
display, and video decode cannot distort the model comparison.

## Recommendation

- **416 FP32 is the quality default.** It has the best test precision and
  mAP50-95 of the deployable artifacts.
- **416 INT8 is the responsive CPU option.** On the audited CPU it has nearly
  the 320 FP32 pipeline speed while retaining substantially more recall and
  mAP50. Its artifact provenance was reproduced byte-for-byte during this
  audit.
- **320 FP32 remains the compatibility fallback.** Use it when INT8 is not
  faster on the target processor or when the smallest input tensor matters.
- **Do not ship a 640 custom export from this checkpoint.** Evaluating the
  completed 416 checkpoint at 640 produced no meaningful test mAP50 gain and
  reduced mAP50-95 by 0.0259, while CPU inference was much slower.

Hardware behavior varies. Run the included benchmark on each target rather
than selecting a model from CPU/GPU branding alone.

## Completed checkpoint provenance

Both production checkpoints completed the configured 20 CPU epochs with seed
0, deterministic training enabled, a batch of 16, and the prepared one-class
dataset. They share dataset archive SHA-256
`eb0ea27ecbf7ac14e13485b6c8943a5bcacbb6adabc7d63d571ce3cb9af850b5`,
prepared-manifest SHA-256
`c04e4055b9a1cddb2ec2795c8964863b8dd8e5b440364bd8ee473a0aca8765b1`,
and initial YOLO26n weights SHA-256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

| run | completed epochs | selected epoch | best checkpoint SHA-256 | selected-epoch validation mAP50 / mAP50-95 |
| --- | ---: | ---: | --- | ---: |
| `yolo26n_320_player_v2` | 20 | 19 | `2fba94073f3509a365b4178232a67476e1bb33b4b6c494b9c7e48d4bbf9eb927` | 0.6492 / 0.2985 |
| `yolo26n_416_player_v1` | 20 | 20 | `6f34a85bb1573126f0bb473c451f020cc3abc57ed183781d7d57a446f995a570` | 0.6850 / 0.3314 |

The checked `results.csv`, `args.yaml`, and `reproducibility.json` records agree
on image size, epoch count, dataset, initial weights, and output checkpoint.
The 320 run selected epoch 19 because its combined validation fitness was
slightly higher than epoch 20; the 416 run selected the final epoch.

## Reproducible latency benchmark

Run from the repository root in the runtime environment:

```bash
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python scripts/benchmark_models.py \
  --samples 32 --warmup 30 --iterations 100 --repeats 3 \
  > model-benchmark.json
```

The script is read-only with respect to the repository. It prints one
machine-readable JSON document containing artifact hashes, host/runtime data,
the input selection fingerprint, every repeat, and aggregate mean, median,
p95, and p99 timing. Progress goes to stderr. To benchmark an installed bundle
without the dataset, add `--synthetic`; synthetic results are useful for
runtime comparison but say nothing about accuracy.

Method used for the table below:

- Intel Core i7-10850H, 6 cores / 12 threads, CachyOS Linux 7.1.6
- Python 3.14.6, OpenVINO 2026.3.0
- OpenVINO `LATENCY` performance hint, one stream, synchronous batch 1
- 32 sorted, evenly spaced images from the 321-image supplied test split
- input-selection SHA-256:
  `7352e84f7df5de30bc77656759a120414cf3979367c21737554c6750431a8e13`
- 30 untimed warmup frames, then 3 repeats of 100 frames
- application preprocessing and postprocessing included; frames preloaded
- garbage collection disabled only during timed iterations

| deployed model | artifact bytes | inference mean | inference median | inference p95 | pipeline mean | pipeline p95 | pipeline FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320 FP32 | 9,891,961 | 11.58 ms | 10.93 ms | 16.37 ms | 12.98 ms | 18.94 ms | 77.0 |
| 416 FP32 | 9,909,430 | 16.56 ms | 16.09 ms | 18.05 ms | 18.22 ms | 20.24 ms | 54.9 |
| 416 INT8 PTQ | 3,254,986 | 11.42 ms | 10.58 ms | 13.09 ms | 13.08 ms | 15.59 ms | 76.4 |

Relative to 416 FP32, INT8 reduced mean inference latency by 31.0%, mean
pipeline latency by 28.2%, and artifact bytes by 67.2% on this CPU. Its mean
pipeline time was within 0.8% of 320 FP32. The full JSON from this audit had
SHA-256 `4cef97890af1b08b5d2a6cd66ba19b78e771480e51965de449e962d28886735e`.
Timing is sensitive to power mode, temperature, background work, runtime
version, and device drivers, so compare fresh JSON from the actual machine.

## Reproduced accuracy

Accuracy used Ultralytics 8.4.116, OpenVINO 2026.3.0, batch 1, CPU, the model's
native static image size, and every image in each supplied split. `workers=0`
and plots/JSON export were disabled. The tables below describe the deployed
OpenVINO graphs rather than copying rounded training-console values.

| deployed model | split | images | instances | precision | recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 320 FP32 | validation | 593 | 894 | 0.7000 | 0.5872 | 0.6444 | 0.3001 |
| 416 FP32 | validation | 593 | 894 | 0.7343 | 0.6584 | 0.6976 | 0.3336 |
| 416 INT8 PTQ | validation | 593 | 894 | 0.7023 | 0.6711 | 0.6681 | 0.3183 |
| 320 FP32 | supplied test | 321 | 523 | 0.7684 | 0.5711 | 0.6617 | 0.3283 |
| 416 FP32 | supplied test | 321 | 523 | 0.7903 | 0.6501 | 0.7202 | 0.3526 |
| 416 INT8 PTQ | supplied test | 321 | 523 | 0.7367 | 0.6654 | 0.7065 | 0.3371 |

Compared with 416 FP32 on the supplied test split, INT8 loses 0.0137 absolute
mAP50 and 0.0155 mAP50-95, while recall rises by 0.0153. Compared with 320
FP32, INT8 gains 0.0448 mAP50 and 0.0943 recall at essentially the same mean
pipeline latency on the audited CPU.

### 640 decision

The completed 416 training checkpoint was also evaluated without retraining:

| checkpoint input | split | precision | recall | mAP50 | mAP50-95 | PyTorch CPU inference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 416 | supplied test | 0.7685 | 0.6386 | 0.7228 | 0.3574 | 45.17 ms |
| 640 | supplied test | 0.7675 | 0.6520 | 0.7221 | 0.3315 | 83.19 ms |

At 640, test mAP50 changed by -0.0007 and mAP50-95 by -0.0259. That fails the
meaningful-gain criterion, so no 640 custom model was exported or added to the
release set.

## INT8 provenance verification

The INT8 graph was regenerated in a temporary directory with the local
`scripts/quantize_model.py`, NNCF 3.3.0, OpenVINO 2026.3.0, the FP32 416 graph,
and 300 evenly sampled validation images. Both regenerated files were
byte-identical to the existing artifact:

| artifact | SHA-256 |
| --- | --- |
| source 416 FP32 XML | `6779ead62c3ace7d4de4b5a43dd3cb2cf8c6be84fa1399cd40db1eaf4b8c6450` |
| source 416 FP32 BIN | `c6d908bce89df473a796c8ad72690872f37e776eb41a70686095baeb77e72895` |
| 416 INT8 XML | `e29876a30238511dae38382d358cf36592023f9183c2d35bd2c5a1714b71ee84` |
| 416 INT8 BIN | `4b6d2893f334c0406092d7ba2011d41f8b5ae6f36fdd3ce6cc9941fcecbd2a20` |

The INT8 `metadata.yaml` and attribution now record the tool hash,
calibration-selection hashes, source hashes, output hashes, and verified
accuracy instead of incorrectly identifying the graph as FP32.

## Accuracy caveat

The supplied train, validation, and test directories are not source-independent.
The dataset manifest records 26 repeated original-basename/video-sequence
groups and 41 related source groups crossing split boundaries. Accuracy numbers
are therefore useful for comparing these artifacts under identical inputs, but
they are optimistic estimates of performance on genuinely unseen gameplay.
Future model work should begin with a source-grouped split and additional
third-person gameplay captured from the actual clone.
