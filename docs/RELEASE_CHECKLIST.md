# ProAim release checklist

Use this checklist from a clean checkout of the exact commit intended for the
release. ONNX Runtime variants provide the same Python module, so each build
environment must contain exactly one runtime variant. Published builds must use
the target hash lock documented in `docs/DEPENDENCY_LOCKS.md`; broad source
requirements are not a release lock.

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
- Inspect `models/RELEASE-DEFAULT.json`. Its self-hash must validate, every
  declared member must match its recorded size and SHA-256, and all five
  qualification fields must remain false. The pointer selects the launcher,
  packaging, and release-validation paths, static HxW, and requested detail ROI
  width (`0` means disabled); do not duplicate or hand-edit those values in
  Python or the PyInstaller spec.

If a newly exported model won development selection, validate its proposed
adoption before changing repository assets:

```bash
python scripts/adopt_fort_release_candidate.py \
  --candidate path/to/staged-candidate \
  --candidate-evaluation path/to/candidate-runtime-eval/metrics.json \
  --tournament-selection path/to/sealed-tournament-selection \
  --validate-only
```

Review the exact tournament winner, runtime backend/pipeline/output head,
static shape, pointer-bound detail ROI width, all seven comparisons, and false
qualification record. Confirm the result contains the sealed-plan,
seven-comparison, derived-winner, and winner-training-results replay receipt;
each comparison must report identical sealed/recomputed hashes at confidence
0.25 and 2,000 bootstrap samples. Repeat without `--validate-only` only for that reviewed
identity. Adoption publishes a new content-addressed directory and
swaps the pointer last; it never overwrites or deletes the previous model and it
does not satisfy independent holdout, reviewed negatives, frozen-build, or
physical-GPU gates. Commit the pointer, additive SHA manifest entries, and the
complete new asset directory together. Never invoke adoption merely to make a
build use an unevaluated checkpoint.

Confirm that the new directory is self-contained: ONNX and OpenVINO model
formats, labels, attribution, exact safe training results, the adoption record,
canonical privacy-redacted candidate/training/winner-runtime receipts, the
sealed selection and seven comparisons, plus its byte-exact plan, four runtime
reports, and n/s training-result inputs. Every item must have a pointer role,
size, SHA-256, and additive release-manifest entry. Search the public JSON and
CSV evidence for absolute POSIX paths, Windows drive paths/backslashes, and
file URIs in both values and field/header names, and local home-directory names;
none are permitted. Confirm the tournament `public_evidence_privacy.sha256` and
adoption `source.public_evidence_sha256` match each other and the exact shared
scanner source used by semantic pointer validation. The selected training-results table must use the pinned
Ultralytics columns, finite numbers, contiguous epochs, and the candidate's
exact row/epoch count. Raw local candidate/training
records remain validation inputs and must not be copied into the bundle.

After adoption, qualify the exact pointer-bound workload's frozen target-GPU
latency first. Do not consume the one-time independent holdout until that
performance-eligibility gate passes; holdout is the final accuracy selection,
not an early benchmark.

The final independent protocol evaluates the tournament's exact adopted
workload: a primary winner must bind width `0`, while a configured
primary-plus-detail winner must bind the exact positive tournament width. The
operator cannot change that workload when planning the holdout.

Fresh profiles and an explicit Recommended-preset selection adopt the pointer's
complete workload. Migrated profiles preserve their saved detail setting; an
existing user opts into a newly selected workload by explicitly reselecting the
Recommended detector and reviewing its displayed requested ROI width.

For a local Linux CPU release reproduction, start with exact CPython 3.13.14
and create a fresh, ignored environment and report directory:

```bash
python3.13 -c 'import platform; assert platform.python_version() == "3.13.14"'
python3.13 -m venv .release-venv
mkdir -p .release-metadata
.release-venv/bin/python -m pip install --require-hashes --no-compile --force-reinstall \
  --report .release-metadata/pip-bootstrap-linux-cpu-py313.json \
  -r requirements-locks/bootstrap-py313.txt
.release-venv/bin/python -m pip install --require-hashes --no-compile --no-build-isolation \
  --force-reinstall \
  --report .release-metadata/pip-dependencies-linux-cpu-py313.json \
  -r requirements-locks/bootstrap-py313.txt \
  -r requirements-locks/linux-cpu-py313.txt
.release-venv/bin/python scripts/write_dependency_manifest.py \
  --profile linux-cpu-py313 \
  --pip-report .release-metadata/pip-bootstrap-linux-cpu-py313.json \
  --pip-report .release-metadata/pip-dependencies-linux-cpu-py313.json \
  --output .release-metadata/linux-cpu-py313-DEPENDENCY-MANIFEST.json
export PATH="$PWD/.release-venv/bin:$PATH"
```

Use the workflow's equivalent `windows-directml-py313` or
`windows-cuda-py313` profile on Windows. Never reuse one profile's environment
for another variant.

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
(OpenVINO-only), the pointer-selected FP32 player at its declared static HxW,
320 FP32 player, COCO 320, COCO 416, and YOLO11l 640. It verifies the
self-hashed default pointer, release SHA-256 manifest, labels, required player
attribution/metadata, generic YOLO export provenance, static model input/output
contracts, every required OpenVINO IR pair, and every required ONNX graph.

Before packaging, inspect `DEPENDENCY-MANIFEST.json` and confirm
`artifact_hash_contract.enforced_before_install` is true, the profile matches
the target, the Python version is 3.13.14, every artifact has a SHA-256, and
`pip check` succeeded. After packaging, confirm the adjacent `BUILD-INFO.json`
schema 2 records the dependency manifest's exact SHA-256 and distribution
count.

## 2. Hardware and model sanity

Run a short benchmark on at least one representative CPU and each GPU runtime
that will be claimed in release notes:

```bash
python scripts/benchmark_models.py --synthetic \
  --samples 32 --warmup 30 --iterations 100 --repeats 3
```

For a Windows ONNX Runtime build, retain a separate JSON file for each claimed
provider and release-default model. Final qualification must use the model,
labels, input shape, and hashes declared by the extracted bundle's
`BUILD-INFO.json`; use its helper so that contract is resolved and verified
automatically:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\Qualify-ProAimGpu.ps1 -Provider DirectML -AdapterIndex 1 -RunLive
```

Use the DXGI adapter index printed by **Scan hardware**; never infer it from
WMI list order. Confirm that `runtime.requested_provider` appears in
`runtime.active_providers`, and retain the runtime-reported provider options.
The `provider_option_overrides` record must contain that `device_id`. Session
creation proves the index was accepted, but final qualification still requires
operating-system GPU activity telemetry for the named physical adapter.

The helper runs the bounded normal CLI twice from the same extracted Windows
candidate, once with `--no-preview` and once with `--preview-fps 15`. It passes
the release-default model, labels, and shape explicitly with `DIRECTML:N`,
`--require-full-provider`, `--max-frames 1000`, `--max-seconds 60`, and a new
`--metrics-json` path for each run. Keep both reports with `BUILD-INFO.json`;
do not qualify a
candidate whose report shows MSS fallback, the wrong adapter, CPU graph-node
fallback, capture failures, or unexplained high tail latency.

For accuracy comparisons, omit `--synthetic` and use the prepared pinned test
split. Keep the generated JSON and compare it with
`docs/MODEL_BENCHMARKS.md`. Confirm every report fingerprints the exact launcher
default recorded by `BUILD-INFO.json`; do not qualify an alternate graph merely
because it was present in the bundle. INT8 availability alone is not an
accuracy claim.

In the Qt UI, smoke-test:

- camera/video capture and, on Linux X11, Moonlight screen/region capture;
- **Scan hardware**, device application, and **Show GPU setup instructions**;
- all presets available for the selected backend (INT8 must stay OpenVINO-only);
- preview/draw/crop settings and third-person self filtering;
- preview close/Escape handling on a static desktop, and no detection-rate
  collapse when the preview is moved, resized, covered, or compositor-limited;
- F5/**Start detection**, Escape/**Stop**, and closing the app during a run;
- controller precision on its supported Linux/PXN hardware;
- if MAKCU is being tested, explicit target-label validation, exact-board and
  button binding, required physical press-and-release verification, and
  immediate neutral output on release/stale/stop.

Remote output must remain rejected. Neither a UI smoke test nor release notes
should claim enemy/team identity, game-memory access, anti-cheat integration, or
true glass-to-glass latency. `preview service` means inline HighGUI
submission/event service on thread-affine backends and owned-copy/latest-mailbox
publication on threaded Windows HighGUI. Its moving average is amortized across
all processed frames (non-preview frames contribute zero), and neither mode
measures physical display scanout.

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
python scripts/smoke_release_default_model.py \
  --bundle dist/ProAim --executable dist/ProAim/ProAim
```

Linux NVIDIA uses `requirements-runtime-cuda.txt` and
`PROAIM_RUNTIME_VARIANT=cuda`; confirm the resulting bundle contains the CUDA
13 and cuDNN 9 libraries installed by the runtime extra, then test a real model
through the CUDA provider on a machine with a compatible NVIDIA driver. This is
a manual experimental build and is not a tagged release asset. A Linux
legacy AMD/ROCm source experiment uses `requirements-runtime-rocm.txt` and
`PROAIM_RUNTIME_VARIANT=rocm` only after checking the exact GPU/OS combination
against AMD's current support matrix. ONNX Runtime removed that provider after
1.22 and directs supported new AMD stacks to MIGraphX. Do not publish this ZIP
or claim RX 6950 XT Linux support; qualify that target through Windows DirectML.

Windows PowerShell, in separate CUDA and DirectML environments:

```powershell
python -m pip install -r requirements-build.txt
python -m pip install -r requirements-runtime-cuda.txt
.\scripts\build_windows_app.ps1 -RuntimeVariant cuda
.\dist\ProAim\ProAimCLI.exe --cli --help
.\dist\ProAim\ProAimCLI.exe --runtime-info
python scripts/smoke_release_default_model.py `
  --bundle dist\ProAim --executable dist\ProAim\ProAimCLI.exe
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
artifacts. The frozen smoke creates a real player-model session and runs one
inference on CPU; GPU release claims still require the full provider benchmark
on matching physical hardware. `.github/workflows/build-windows.yml` retains
manual CUDA and DirectML
candidate builds; provider enumeration alone does not qualify a CUDA candidate
for publication.

Only after the intended commit passes CI:

```bash
git tag -a vX.Y.Z -m "ProAim vX.Y.Z"
git push origin vX.Y.Z
```

A pushed `v*` tag does **not** create a GitHub Release. It stages exactly
`ProAim-Linux-x64.zip`, `ProAim-Windows-x64-DirectML.zip`, and the
content-addressed `RELEASE-CANDIDATE-MANIFEST.json` in the immutable
`ProAim-Release-Candidate` Actions artifact. Record the tag-build run ID, the
candidate-manifest SHA-256, and the DirectML ZIP SHA-256 from the run summary.
The artifact expires after 30 days. A manual `workflow_dispatch` may build
archives for testing, but cannot stage or publish a release candidate.

### Protected dual-GPU DirectML publication

The Windows DirectML ZIP is the primary GPU release, so provider enumeration
or a hosted CPU smoke is not enough to publish it. The exact pointer-selected
`BUILD-INFO.release_default_model` from one staged ZIP must independently pass
on both of these physical products:

- `AMD Radeon RX 6950 XT` (`amd_rx_6950_xt`); and
- `NVIDIA GeForce RTX 5060 Laptop GPU` (`nvidia_rtx_5060_laptop`).

Register each physical Windows machine as an interactive self-hosted runner
with the custom label `proaim-directml-qualification`. Do not run the agent as
a Windows service: both bounded DXcam passes and the post-run Task Manager form
need the logged-in desktop. Keep these runners offline except while qualifying
a reviewed commit, and never expose the label to pull-request workflows.

A repository administrator must create three protected GitHub environments:

- `directml-amd_rx_6950_xt-physical-attestation`;
- `directml-nvidia_rtx_5060_laptop-physical-attestation`; and
- `directml-release-publication`.

For each role-specific attestation environment, require a reviewer other than
the dispatcher/observer, enable prevention of self-review, restrict deployments
to protected `main`, disallow bypass, add the environment-only secret
`DIRECTML_PHYSICAL_ATTESTATION_GUARD`, set
`DIRECTML_PHYSICAL_ATTESTATION_POLICY_VERSION=required-reviewers-v1`, and set
`DIRECTML_INDEPENDENT_REVIEWER_GROUP` to respectively
`amd_rx_6950_xt-independent-reviewers-v1` or
`nvidia_rtx_5060_laptop-independent-reviewers-v1`. Use independent reviewer
assignments for the two products.

For `directml-release-publication`, require another independent reviewer with
self-review prevention, protected-`main` restriction, and no bypass. Add a
different environment-only `DIRECTML_RELEASE_ENVIRONMENT_GUARD`, set
`DIRECTML_RELEASE_ENVIRONMENT_POLICY_VERSION=required-reviewers-v1`, and set
`DIRECTML_RELEASE_REVIEWER_GROUP=directml-publication-independent-reviewers-v1`.
These external controls are not created by committing the workflow; absent or
incorrect guard variables fail closed.

The final accuracy gate also requires two additional protected environments,
which are deliberately not created by the repository:

- `independent-holdout-access`: require reviewers who did not select/train the
  candidate, prevent self-review, restrict to protected `main`, and disallow
  administrator bypass where the GitHub plan supports it. Set the
  environment-only secret `INDEPENDENT_HOLDOUT_ACCESS_GUARD`; variables
  `INDEPENDENT_HOLDOUT_ACCESS_POLICY_VERSION=required-reviewers-v1`,
  `INDEPENDENT_HOLDOUT_ACCESS_REVIEWER_GROUP=independent-holdout-access-reviewers-v1`,
  and `INDEPENDENT_HOLDOUT_RUNNER_POLICY_VERSION=sealed-directml-v1`; and the
  environment-only secrets `INDEPENDENT_HOLDOUT_PACKAGE_PATH` and
  `INDEPENDENT_HOLDOUT_LEDGER_PATH`.
- `independent-holdout-attestation`: assign a separate reviewer group from the
  access reviewers, prevent self-review, restrict to protected `main`, and
  disallow bypass. Set `INDEPENDENT_HOLDOUT_ATTESTATION_GUARD` plus variables
  `INDEPENDENT_HOLDOUT_ATTESTATION_POLICY_VERSION=required-reviewers-v1` and
  `INDEPENDENT_HOLDOUT_ATTESTATION_REVIEWER_GROUP=independent-holdout-attestation-reviewers-v1`.
  Give this environment the same two sealed package/ledger path secrets so it
  can perform the authoritative retired-ledger replay.

Route all three protected holdout jobs only to a dedicated Windows x64 runner
with every label `self-hosted`, `Windows`, `X64`,
`proaim-independent-holdout-directml`, and
`proaim-rx-6950-xt-holdout`. Put it in a runner group restricted to this
repository and the independent-holdout workflow. The machine must expose
exactly the authenticated `AMD Radeon RX 6950 XT` at the supplied numeric DXGI
and DirectML index. Do not schedule pull requests or unrelated workflows on
this runner.

A secret filesystem path is not access control. Run the agent under a dedicated
OS account; grant that account and no unrelated service/process access to the
sealed directory and append-only ledger; deny interactive browsing; and audit
the ACL before each release. Prefer an environment-approval-gated mount or
decryption credential, mount only for the protected step, and unmount/lock it
afterward. Custom labels and policy-marker variables are assertions, not a
sandbox. If the environments, reviewer separation, runner-group restriction,
OS ACLs, exact runner, or sealed mount are absent, do not dispatch the workflow;
the repository side intentionally fails closed but cannot repair missing
external isolation.

After both physical DirectML artifacts are sealed, dispatch **Qualify final
independent holdout** once with the exact tag/source candidate and both
physical-evidence identities. Reruns are forbidden. The workflow authenticates
and stages both physical artifacts, creates/uploads
`ProAim-Independent-Holdout-Frozen-Plan` before mapping any sealed-member path,
then consumes the package once on the RX 6950 XT. Consumption is durably burned
before the first sealed image/COCO read; any later failure permanently blocks a
retry. Success uploads exactly
`ProAim-Verified-Holdout-Prerequisites`, the frozen plan,
`ProAim-Independent-Holdout-Evidence`, and
`ProAim-Independent-Holdout-Attestation`, with no fifth debug/raw artifact.
Record the numeric ID and GitHub Actions digest of all four artifacts and the
successful run ID.

Qualify and publish one exact candidate as follows:

1. From protected `main` at the tagged commit, dispatch **Qualify Windows
   DirectML on physical GPU** once on each required machine. Supply the same
   tag, tag-build run ID, candidate-manifest SHA-256, and DirectML ZIP SHA-256
   to both runs; choose the matching fixed role and the exact DXGI index shown
   by **Scan hardware**. The typed phrases are
   `I ATTEST THAT I OBSERVED AMD Radeon RX 6950 XT RUN DIRECTML FOR vX.Y.Z`
   and
   `I ATTEST THAT I OBSERVED NVIDIA GeForce RTX 5060 Laptop GPU RUN DIRECTML FOR vX.Y.Z`.
2. Each self-hosted run re-resolves the tag and numeric source artifact ID,
   safely extracts the same ZIP, and runs its bundled helper. The benchmark is
   30 warmups and three repeats of 100 timed iterations. Every repeat and the
   aggregate must have inference p95 at or below 35 ms. The no-preview and
   preview-15 screen runs are each bounded at 1,000 frames/60 seconds, require
   at least 120 timing samples, 20 elapsed/update FPS, and p95 observed-pipeline
   and freshness latency at or below 50 ms. Every accelerated run disables CPU
   graph-node and EPFail fallback.
3. The helper accepts only the model, labels, static HxW, hashes, and detail
   scalar in `BUILD-INFO.release_default_model`. When the detail scalar is
   positive it is a maximum source-ROI width, not a square crop. The live
   reports must prove the exact centered model-aspect plan and applied W/H. For
   384x640 with requested width 768 on 1920x1080, the expected exact-aspect ROI
   is 765x459 and approximately 2.5x magnification; the recorded plan is
   clamped by the aspect adjustment even though preprocessing receives the
   already-applied dimensions.
4. A vendor-neutral recorder correlates Windows `GPU Engine` counters to the
   exact extracted `ProAimCLI.exe` path and PID and to the selected DirectX
   adapter LUID. Each benchmark/live interval needs at least two distinct
   captures and positive GPU work, and separate processes may not reuse a PID.
   Both live reports must independently agree on the exact normalized product,
   vendor/device IDs, DXGI index, full provider, model/labels hashes, and DXcam
   capture with no fallback. After automation ends, the named observer must
   complete the local Task Manager form, affirm all three passes, confirm the
   Task Manager engine agrees with LUID telemetry, and retype the exact phrase.
5. The role-specific protected reviewer inspects the immutable raw artifact
   before allowing the sealing job. The sealer recomputes its content manifest
   and every candidate, archive, BUILD-INFO, dependency-manifest, model, labels,
   report, telemetry, and observation hash. Record the evidence run ID and the
   sealed archive, inner qualification manifest, physical attestation, and
   public receipt SHA-256 values from each summary. Each sealed artifact has a
   fixed role-specific name and expires after 30 days.
6. Dispatch **Publish dual-GPU-qualified DirectML release** with the shared
   candidate identity, both distinct physical-evidence run identities and all
   four hashes for each role, plus the independent-holdout run and all four
   holdout artifact ID/digest pairs. The workflow derives the exact manager
   confirmation from the tag. The read-only verifier re-downloads every
   artifact by its numeric ID, rejects any extra holdout artifact, recomputes
   every sealed byte and semantic cross-link, and creates one content-addressed
   publication stage. The protected publication reviewer must inspect that
   exact stage before the write-capable job proceeds.
7. Publication creates a draft, uploads only the Linux and DirectML ZIPs,
   `SHA256SUMS.txt`, privacy-redacted per-GPU/combined receipts, and the
   redacted independent-holdout receipt, then
   downloads and byte-verifies every asset. Immediately before the one public
   transition it re-resolves the tag, source/evidence runs, artifact IDs and
   digests; it repeats tag, release, asset-identity, and byte checks afterward.
   Any failure triggers deletion of the transaction's marker-identified release,
   including recovery from a lost create response. Raw observer identity,
   Task Manager text, PIDs, adapter LUID, paths, telemetry, holdout metrics,
   plan, ledger, attestation, images, and COCO labels never become public; the
   redacted receipts retain the authenticated run/artifact/environment/hardware
   hashes as durable release provenance.

Only after that dual-GPU publication may the separately qualified CUDA bundle
be attached through the CUDA workflow below. CUDA remains a distinct artifact
and is never substituted for either DirectML physical gate.

### Protected Windows CUDA attachment

The automatic `v*` tag workflow deliberately stages only the primary Linux and
DirectML archives. A Windows CUDA ZIP may be added to an **existing published release** only through
`.github/workflows/attach-qualified-cuda.yml`; that workflow never creates a
tag or release.

Before these workflows can be used, register a dedicated Windows self-hosted
runner on the representative NVIDIA computer. Add the custom runner label
`proaim-cuda-qualification` and run the agent interactively in the logged-in
desktop session (not as a Windows service): the bounded DXcam passes require a
real visible desktop. Keep the runner offline except while qualifying a known
commit, and do not use it for pull-request workflows.

A repository administrator must also create two GitHub environments. Configure
`cuda-physical-attestation` with all of these controls:

- require a reviewer other than the person dispatching/observing the run and
  enable prevention of self-review;
- restrict deployment branches to `main` and do not allow protection-rule
  bypass;
- add an environment-only secret named
  `CUDA_PHYSICAL_ATTESTATION_GUARD`; and
- add the environment variable
  `CUDA_PHYSICAL_ATTESTATION_POLICY_VERSION=required-reviewers-v1`.

Configure `cuda-release-publication` with all of these controls:

- require at least one reviewer other than the person dispatching the workflow
  and enable prevention of self-review;
- restrict deployment branches to `main` and do not allow protection-rule
  bypass for this publication path;
- add an environment-only secret named `CUDA_RELEASE_ENVIRONMENT_GUARD` with a
  new random value (do not create a repository or organization secret with the
  same name); and
- add the environment variable
  `CUDA_RELEASE_ENVIRONMENT_POLICY_VERSION=required-reviewers-v1`.

These runner labels, environments, reviewers, secrets, and variables are
external gates; committing the workflows does not configure them. Missing
guards or policy variables fail closed.

Build and qualify one exact candidate as follows:

1. Dispatch `.github/workflows/build-windows.yml` at the commit behind the
   intended existing tag. Wait for that manual run to finish successfully and
   record its numeric run ID. The CUDA artifact expires after 14 days.
2. From `main`, dispatch **Qualify Windows CUDA on physical GPU**. Supply the
   exact tag, build run ID, candidate SHA-256, `nvidia-smi` GPU product name,
   expected observer name, and the exact precommit phrase
   `I ATTEST THAT I OBSERVED <exact GPU> RUN CUDA FOR <exact tag>`.
   The legal-scope acknowledgement must equal
   `PHYSICAL QUALIFICATION ONLY - NVIDIA LEGAL REVIEW REMAINS REQUIRED`; this
   workflow deliberately cannot approve redistribution.
3. The self-hosted job re-resolves the tag and build artifact, extracts the
   candidate safely, runs the bundled helper with CUDA full-provider enforcement
   and both bounded live modes, and records `nvidia-smi` GPU name, UUID, driver,
   compute capability, utilization, memory, and compute-process samples every
   500 ms. The sealer requires at least one correlated benchmark sample and at
   least five correlated samples in each live pass; this is an activity gate,
   not a claim that sparse `nvidia-smi` sampling attributes every frame. After
   all three passes finish, a local Windows desktop form requires the observer
   to enter the Task Manager GPU/engine, affirm each pass separately, and
   retype the exact confirmation. Cancelling, answering no, or a mismatch fails
   the run. The resulting timestamped `LOCAL-PHYSICAL-OBSERVATION.json` is
   included in the immutable raw artifact. The helper-produced
   `qualification-manifest.json` intentionally remains `qualified=false` with
   physical confirmation pending; it cannot attest to a human observation.
4. A reviewer for `cuda-physical-attestation` must inspect the preceding job and
   raw artifact before approving the seal job. Its exact pre-upload
   `RAW-CONTENT-MANIFEST.json` hash is passed directly between jobs; the sealer
   recomputes every recorded path, size, and SHA-256 after download. The seal
   job validates every software report and hash, requires repository-owned
   latency/FPS/sample floors plus temporally correlated non-zero activity and a
   `ProAimCLI.exe` process on the dedicated single-NVIDIA-GPU runner, writes the
   structured completed
   `PHYSICAL-GPU-ATTESTATION.json` and `TASK-MANAGER-CONFIRMATION.txt`, and uploads
   the single accepted artifact
   `ProAim-Windows-CUDA-Qualification-Evidence`. Record its run ID plus the
   inner evidence ZIP, qualification manifest, and physical attestation hashes
   from the summary. Raw and sealed sensitive artifacts expire after 7 days.
5. Separately review the exact NVIDIA wheel metadata and declared
   license/EULA/notice payloads listed by
   `NVIDIA-REDISTRIBUTION-MANIFEST.json`. Confirm the top-level
   `THIRD_PARTY_NOTICES.md` inventories CUDA, cuDNN, and every included NVIDIA
   distribution. The manifest is a payload-integrity gate, not legal advice or
   a conclusion that redistribution is permitted.
6. From the Actions page, dispatch **Attach physically qualified Windows CUDA
   bundle** from `main`. Supply the tag, source build run ID, qualification run
   ID, the fixed evidence artifact name, candidate/evidence/manifest/physical
   attestation hashes, and full GPU name. The confirmation input must equal
   `ATTACH QUALIFIED WINDOWS CUDA TO vX.Y.Z` exactly. The separate legal-review
   input must equal
   `I APPROVE NVIDIA REDISTRIBUTION REVIEW FOR vX.Y.Z` exactly; do not enter it
   until the exact bundled NVIDIA payload and governing terms have been reviewed.
7. The `cuda-release-publication` reviewer must independently inspect the exact
   qualification artifact and its hashes, confirm the candidate identity and
   separate NVIDIA redistribution review, and only then approve publication.

The read-only Windows job resolves lightweight or annotated tags to their
commit; requires both allowlisted successful manual workflows, the same
repository, and exact head SHA; downloads the evidence by qualification run ID;
and rejects missing, expired, duplicate, wrong-name, or multi-file artifacts.
The GitHub Actions outer artifact-digest output is informational; the security
gate is the explicit raw/staged content-manifest hash plus exact inner
archive/file hashes, all recomputed after download. It safely extracts both
ZIPs and verifies the evidence
archive hash, inner manifest and physical-attestation hashes, sealed file set,
BUILD-INFO/CLI/helper/model/labels hashes, full-provider benchmark/live records,
DXcam capture, Task Manager confirmation, GitHub actor/run binding, dedicated
single-GPU/no-preexisting-ProAim runner invariant, strict finite telemetry, and
the repository-owned per-run correlation/sample-density policy. It separately
validates the candidate NVIDIA redistribution
inventory and runs one real release-default-model inference on CPU. That hosted CPU
smoke is an archive-integrity check, not another physical CUDA claim.

After approval, the publish job rechecks the tag, source run, exact staged
content manifest, and staged hashes,
existing release, and every existing ZIP against `SHA256SUMS.txt`. It refuses
to overwrite an existing CUDA asset or proceed with stale/missing checksum
entries. Immediately before mutation and again after final byte verification,
it re-resolves the tag and release identity. It adds the CUDA ZIP plus a
privacy-redacted durable qualification receipt, checksums both, and
intentionally replaces only `SHA256SUMS.txt`, preserving Linux and DirectML
entries; any mutation error or tag movement triggers best-effort rollback to
the original assets. The public receipt omits observer identity, GPU UUID,
process paths, and raw telemetry. Its preview figures measure application work
and preview submission, not physical display scanout. Review the workflow
summary and re-download all three ZIPs plus the receipt for post-publication
verification below.

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
