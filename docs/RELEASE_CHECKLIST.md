# ProAim release checklist

Use this checklist from a clean checkout of the exact commit intended for the
release. ONNX Runtime variants provide the same Python module, so each build
environment must contain exactly one `requirements-runtime-*.txt` variant.

## 1. Source, license, and artifact gate

- Confirm `git status --short` is empty and record `git rev-parse HEAD`.
- Confirm the version/tag follows the repository's `v*` convention.
- Review `LICENSE` (AGPL-3.0-or-later project license),
  `THIRD_PARTY_NOTICES.md`, `packaging/licenses/LGPL-3.0-only.txt`,
  `packaging/licenses/GPL-3.0-only.txt`,
  `packaging/licenses/pyserial-3.5-BSD-3-Clause.txt`, every FORT-derived
  model's `ATTRIBUTION.md`, and each generic YOLO model's `metadata.yaml`.
- Confirm any intentionally changed model file has a reviewed entry in
  `models/RELEASE-MANIFEST.sha256`. Do not update a hash merely to silence a
  mismatch.

Create the source-test environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
python -m pip install -r requirements-runtime-cpu.txt
```

Run the complete local gate:

```bash
python -m compileall -q app.py config.py main.py aiming capture \
  controller_precision detection launcher scripts tests utils
python -m unittest discover -s tests
python scripts/validate_release_assets.py --project-root .
python app.py --cli --help
python app.py --runtime-info
```

The model preflight must report all six configurations: 416 INT8 player
(OpenVINO-only), 416 FP32 player, 320 FP32 player, COCO 320, COCO 416, and
YOLO11l 640. It verifies the release SHA-256 manifest, labels, required player
attribution/metadata, generic YOLO export provenance, static model input/output
contracts, all OpenVINO IR pairs, and all five ONNX graphs.

## 2. Hardware and model sanity

Run a short benchmark on at least one representative CPU and each GPU runtime
that will be claimed in release notes:

```bash
python scripts/benchmark_models.py --synthetic \
  --samples 32 --warmup 30 --iterations 100 --repeats 3
```

For accuracy comparisons, omit `--synthetic` and use the prepared pinned test
split. Keep the generated JSON and compare it with
`docs/MODEL_BENCHMARKS.md`. Confirm the default remains 416 FP32 unless fresh
accuracy evidence supports a model change; INT8 availability alone is not an
accuracy claim.

In the Qt UI, smoke-test:

- camera/video capture and, on Linux X11, Moonlight screen/region capture;
- **Scan hardware**, device application, and **Show GPU setup instructions**;
- all presets available for the selected backend (INT8 must stay OpenVINO-only);
- preview/draw/crop settings and third-person self filtering;
- F5/**Start detection**, Escape/**Stop**, and closing the app during a run;
- controller precision on its supported Linux/PXN hardware;
- if MAKCU is being tested, explicit target-label validation, exact-board and
  button binding, required physical press-and-release verification, and
  immediate neutral output on release/stale/stop.

Remote output must remain rejected. Neither a UI smoke test nor release notes
should claim enemy/team identity, game-memory access, anti-cheat integration, or
true glass-to-glass latency.

## 3. Local bundle builds

Build each variant from a fresh virtual environment containing its one matching
runtime file. Do not replace one runtime wheel in-place and assume the bundle is
clean.

Linux CPU:

```bash
python -m pip install -r requirements-build.txt
python -m pip install -r requirements-runtime-cpu.txt
PROAIM_RUNTIME_VARIANT=cpu ./scripts/build_linux_app.sh
dist/ProAim/ProAim --cli --help
dist/ProAim/ProAim --runtime-info
```

Linux NVIDIA uses `requirements-runtime-cuda.txt` and
`PROAIM_RUNTIME_VARIANT=cuda`; confirm the resulting bundle contains the CUDA
13 and cuDNN 9 libraries installed by the runtime extra, then test a real model
through the CUDA provider on a machine with a compatible NVIDIA driver. This is
a manual experimental build and is not a tagged release asset. A Linux
AMD/ROCm source build uses `requirements-runtime-rocm.txt` and
`PROAIM_RUNTIME_VARIANT=rocm` on a compatible ROCm host; the automated public
workflow does not publish a ROCm ZIP either.

Windows PowerShell, in separate CUDA and DirectML environments:

```powershell
python -m pip install -r requirements-build.txt
python -m pip install -r requirements-runtime-cuda.txt
.\scripts\build_windows_app.ps1 -RuntimeVariant cuda
.\dist\ProAim\ProAimCLI.exe --cli --help
.\dist\ProAim\ProAimCLI.exe --runtime-info
```

Repeat with `requirements-runtime-directml.txt` and `-RuntimeVariant directml`.
The tagged release ZIPs are:

- `ProAim-Linux-x64.zip`
- `ProAim-Windows-x64-DirectML.zip`

Manual experimental CUDA candidates use these names, but must not be attached
to a tagged release until their extracted frozen archives complete real model
inference on representative NVIDIA hardware:

- `ProAim-Linux-x64-NVIDIA-CUDA.zip`
- `ProAim-Windows-x64-NVIDIA-CUDA.zip`

Open each bundle's `BUILD-INFO.json` and verify its commit, runtime variant,
timestamp, and clean-tree state. Verify `LICENSE`, `THIRD_PARTY_NOTICES.md`,
model attribution, Qt license text, models, labels, and the Windows tester guide
are present.

On Linux, scan every ELF in the completed bundle with `ldd` and reject any
`not found` dependency. Record the highest required `GLIBC_*` symbol and test
the ZIP on the oldest distribution the release claims to support. A local
rolling-release build is suitable for that machine, not a broad Linux artifact.

## 4. CI and publication

`.github/workflows/ci.yml` tests Ubuntu/CPU and Windows/DirectML on pushes and
pull requests to `main`. The release workflow repeats that gate, builds Linux
CPU and Windows DirectML, smoke-tests each frozen CLI, and uploads those two
artifacts. `.github/workflows/build-windows.yml` retains manual CUDA and DirectML
candidate builds; provider enumeration alone does not qualify a CUDA candidate
for publication.

Only after the intended commit passes CI:

```bash
git tag -a vX.Y.Z -m "ProAim vX.Y.Z"
git push origin vX.Y.Z
```

A pushed `v*` tag publishes the Linux CPU and Windows DirectML ZIPs plus
`SHA256SUMS.txt`. A manual `workflow_dispatch` builds those artifacts but does
not publish a GitHub Release.

After publication:

- download each asset from GitHub rather than reusing a local build;
- verify it against `SHA256SUMS.txt`;
- extract it to a new directory and repeat `--runtime-info` and a short
  detection smoke test on matching hardware;
- verify the GitHub release links to the corresponding source/tag, as required
  for AGPL object-code distribution.

## 5. Linux reinstall check

From an extracted or locally built Linux bundle:

```bash
./ProAim/install.sh
# local build equivalent: ./dist/ProAim/install.sh
```

Launch **ProAim** from the application menu and confirm the About dialog shows
AGPL-3.0-or-later and the source link. The per-user installer writes the active
copy to `~/.local/share/proaim` and retains the previous managed version at
`~/.local/share/proaim.previous` for rollback.
