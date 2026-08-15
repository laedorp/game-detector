# Independent Player-Detection Holdout Protocol

`scripts/prepare_independent_player_holdout.py` prepares a local, auditable
player-detection holdout. It does not download footage, call a network service,
randomly split frames, load pickle files, or create a decoded-image cache. No
third-party footage or annotations are included with ProAim by this workflow.

This protocol has two distinct pools:

- `development`: may be used while selecting models, thresholds, crop layouts,
  preprocessing, and postprocessing.
- `sealed_release_holdout`: may be consumed once by one pre-hashed evaluation
  plan after every candidate and decision rule is frozen, then retired.

The unit of isolation is an entire capture session. Assign each session to one
pool before extracting or annotating frames. Never put frames from the same
recording, burst, replay, or resumed capture in different pools. There is no
frame-level split command in the tool.

## Capture and labeling rules

Use only footage that the operator has the right to use for this evaluation.
Keep the actual license/permission text with the capture. The tool verifies that
the supplied bytes match the declared hashes, but it cannot decide whether a
license assertion is legally valid.

For meaningful independence, the sealed sessions should be newly captured and
must not have contributed frames, clips, pseudo-labels, or manual observations
to training or development decisions. Prefer variation in scene, distance,
lighting, motion, character appearance, capture hardware, and UI state. Record
whole negative sessions as well as scenes containing players.

The class is exactly `player`. A box means a visible player, without claiming
that the player is an enemy, ally, account, or real-world identity. Do not add
team or identity fields. Each box uses source-image pixel coordinates and must
explicitly carry:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [100, 40, 20, 48],
  "area": 960,
  "iscrowd": 0,
  "ignore": 0,
  "occluded": false,
  "truncated": false
}
```

The target height is projected to a fixed 1080-pixel-high reference before
bucket assignment:

`projected height = bbox height × 1080 / source image height`

For example, a 32-pixel-high box in a 720p source is 48 projected pixels and
belongs in `target_33_64`. The pinned inventory targets are:

| Inventory | Minimum | Interpretation |
|---|---:|---|
| `target_33_64` | 400 | primary far-range gate |
| `target_le_32` | 150 | ultra-far descriptive inventory; nonblocking |
| `target_65_96` | 250 | medium/far transition |
| `target_gt_96` | 250 | nearer-target control bucket |
| `reviewed_negatives` | 1,000 | false-positive control inventory |

Every zero-box image must be independently reviewed by two different reviewers.
If they disagree, a third, independent adjudicator must resolve it as negative
before it can enter the negative inventory. Positive images cannot carry a
negative-review record. Raw detected/total counts should always accompany recall
or AP results; a percentage without its denominator is not an adequate gate.

The command-line minimums are configurable for stricter gates or pipeline
fixtures. The package always records both the configured values and the pinned
defaults. Lowering a configured minimum does not make
`meets_pinned_release_gates` true. The `target_le_32` minimum is reported
separately as a descriptive inventory target and does not veto release-candidate
ranking because annotation quality in that bucket has not been established as
a reliable selection gate.

Release eligibility also requires at least 15 distinct capture-session/source
groups overall, at least 15 target-bearing sessions in each 33-64, 65-96, and
>96px gating bucket, and at least 15 sessions contributing reviewed negatives.
Counts concentrated in one recording session are not independent release
evidence.

## Input contract

All paths are POSIX-style paths relative to the input manifest directory. Files
and every existing path component must be regular/non-symlinked. Each digest is
lowercase SHA-256. A capture/environment commit is an exact 40- or 64-hex commit.
Timestamps are explicit UTC values ending in `Z`.

Minimal input-manifest shape:

```json
{
  "schema_version": 1,
  "package_id": "clone-session-holdout-2026-08",
  "pool": "sealed_release_holdout",
  "sessions": [
    {
      "session_id": "capture-session-001",
      "assigned_pool": "sealed_release_holdout",
      "captured_at_utc": "2026-08-13T12:00:00Z",
      "source": {
        "kind": "video",
        "path": "source/session-001.mkv",
        "sha256": "<64 lowercase hex>"
      },
      "license": {
        "path": "source/session-001-license.txt",
        "sha256": "<64 lowercase hex>",
        "identifier": "owned-local-capture",
        "authorization_basis": "captured and supplied by the owner",
        "holdout_use_permitted": true,
        "redistribution_permitted": false
      },
      "capture_environment_commit": "<40 or 64 lowercase hex>",
      "acquisition": {
        "tool_name": "local-capture-tool",
        "tool_path": "tools/capture-tool.exe",
        "tool_sha256": "<64 lowercase hex>",
        "config_path": "source/session-001-capture.json",
        "config_sha256": "<64 lowercase hex>",
        "operator_id": "operator-01"
      }
    }
  ],
  "annotations": {
    "path": "annotations/instances.json",
    "sha256": "<64 lowercase hex>"
  },
  "human_review": {
    "annotation_author_ids": ["annotator-01"],
    "reviewer_ids": ["reviewer-01", "reviewer-02"],
    "adjudicator_ids": ["adjudicator-01"],
    "completed_at_utc": "2026-08-13T18:00:00Z",
    "protocol_path": "annotations/review-protocol.txt",
    "protocol_sha256": "<64 lowercase hex>"
  }
}
```

The COCO file has exactly one category:

```json
{"categories": [{"id": 1, "name": "player"}]}
```

Each COCO image additionally binds its session, original frame, and exact bytes:

```json
{
  "id": 1,
  "file_name": "frames/session-001-000100.png",
  "width": 1920,
  "height": 1080,
  "session_id": "capture-session-001",
  "source_frame_index": 100,
  "sha256": "<64 lowercase hex>"
}
```

A zero-box image also includes:

```json
{
  "negative_review": {
    "reviewer_1": {
      "reviewer_id": "reviewer-01",
      "decision": "negative",
      "reviewed_at_utc": "2026-08-13T16:00:00Z"
    },
    "reviewer_2": {
      "reviewer_id": "reviewer-02",
      "decision": "negative",
      "reviewed_at_utc": "2026-08-13T16:05:00Z"
    },
    "adjudication": {"status": "not_required"}
  }
}
```

For a disagreement, set one reviewer decision to `player_present` and replace
`adjudication` with `status: resolved`, the independent `adjudicator_id`, a
`decision` of `negative`, and `reviewed_at_utc`.

## Prepare and verify

Prepare development first, then supply its generated manifest while preparing
the sealed package. Supply every other pool manifest that could overlap:

```bash
python scripts/prepare_independent_player_holdout.py prepare \
  --input-manifest local-holdout/development/input.json \
  --output local-holdout/prepared-development

python scripts/prepare_independent_player_holdout.py prepare \
  --input-manifest local-holdout/sealed/input.json \
  --output local-holdout/prepared-sealed \
  --reference-manifest local-holdout/prepared-development/HOLDOUT-MANIFEST.json
```

References are opened as manifest metadata only; their image/source members are
not opened. The preparation rejects a reused session id, reused raw-capture
SHA-256, exact image SHA-256 collision, or cross-package dHash64 distance at or
below four by default. The dHash uses fixed integer grayscale conversion and
9-by-8 block means. The tool also compares multiple supplied references with
one another. dHash distance is configurable from zero through sixteen, but
exact duplicate checking is always enabled. The direction is deliberate:
prepare development first and use it as a reference while sealing release data.
A development preparation refuses a sealed-pool reference so sealed metadata
cannot become a tuning input.

Preparation copies the supplied raw source, license evidence, acquisition tool
and config, images, annotation source, and review protocol into a staging
directory. It validates them and publishes the whole directory with a
platform-provided atomic no-replace rename (and fails closed if that primitive
is unavailable). A declared archive is copied and hashed as opaque evidence; it
is never extracted or executed. Every package member has an exact byte count and SHA-256; the
canonical JSON manifest self-binds that inventory. Re-running the same tool
version with identical input bytes and options produces identical package
bytes. An existing destination is never overwritten.

Normal development verification is intentionally denied for a sealed package:

```bash
python scripts/prepare_independent_player_holdout.py verify \
  --package local-holdout/prepared-development \
  --mode development

# Curator-only integrity audit; do not use this mode for model development.
python scripts/prepare_independent_player_holdout.py verify \
  --package local-holdout/prepared-sealed \
  --mode curator
```

## Protected workflow and runner setup

`.github/workflows/qualify-independent-holdout.yml` intentionally fails closed
until an administrator configures the external controls. Create protected
environments `independent-holdout-access` and
`independent-holdout-attestation`, assign separate required reviewer groups,
prevent self-review, restrict deployment branches to protected `main`, and
disable administrator bypass where supported. Configure these exact values:

- access secret `INDEPENDENT_HOLDOUT_ACCESS_GUARD`; access variables
  `INDEPENDENT_HOLDOUT_ACCESS_POLICY_VERSION=required-reviewers-v1`,
  `INDEPENDENT_HOLDOUT_ACCESS_REVIEWER_GROUP=independent-holdout-access-reviewers-v1`,
  and `INDEPENDENT_HOLDOUT_RUNNER_POLICY_VERSION=sealed-directml-v1`;
- attestation secret `INDEPENDENT_HOLDOUT_ATTESTATION_GUARD`; attestation
  variables `INDEPENDENT_HOLDOUT_ATTESTATION_POLICY_VERSION=required-reviewers-v1`
  and
  `INDEPENDENT_HOLDOUT_ATTESTATION_REVIEWER_GROUP=independent-holdout-attestation-reviewers-v1`;
- secrets `INDEPENDENT_HOLDOUT_PACKAGE_PATH` and
  `INDEPENDENT_HOLDOUT_LEDGER_PATH` in both protected environments. The ledger
  must resolve to the package's exact `access-ledger` child.

Use a dedicated Windows x64 RX 6950 XT runner carrying all labels
`self-hosted`, `Windows`, `X64`, `proaim-independent-holdout-directml`, and
`proaim-rx-6950-xt-holdout`. Restrict its runner group to this repository and
workflow; never run pull requests or unrelated jobs under the same OS account.
Path secrecy and custom labels are not access controls. Enforce OS ACLs so only
the dedicated account can read the package/append the ledger, prevent unrelated
processes, and preferably expose the package through an approval-gated
mount/decryption credential that is revoked immediately afterward. Missing
environment protection, reviewer separation, runner isolation, or ACL/mount
controls means the workflow must not be run.

The protected chain produces exactly four Actions artifacts: verified physical
prerequisites, the canonical frozen plan uploaded before any sealed-member
access, the fixed-inventory evidence bundle, and the independent attestation.
Any extra debug/raw artifact makes publication ineligible. A successful run is
consumed by publication only through its numeric run ID and all four numeric
artifact ID/GitHub digest pairs; a manual receipt or self-hash is never
authority.

## Freeze the exact final runtime plan

First finish the development tournament and adopt its one winner with
`scripts/adopt_fort_release_candidate.py`. Adoption is still development-only;
it leaves every qualification flag false. The independent evaluator resolves
that content-addressed adopted candidate dynamically, including its ONNX bytes,
labels, exact rectangular HxW shape and exported output head, exporter hash,
adoption record, path-redacted candidate/training/winner-runtime receipts,
exact training-results bytes, tournament selection manifest, and all seven
paired development comparisons. It also binds byte-for-byte copies of the
sealed tournament plan, all four validation runtime reports, and both n/s
training-results files. It does not accept an arbitrary model path or allow a
compact receipt to replace those underlying records.
`--output-format` is only an optional assertion;
when omitted, the plan uses the adopted tournament winner's bound head, and a
mismatch is rejected.

Before any sealed image or annotation member is opened, freeze the exact GPU,
provider, primary/detail geometry, policy, metric limits, latency limit, and
manual-review note. Release eligibility uses only repository policy
`proaim-independent-holdout-v1`, whose canonical hash is
`9a53c85fb5b0a73842ef890209ef363592c10410ca05a72ff819d24cd135ae7e`:

- confidence 0.25;
- far recall >=0.80 and far false positives <=10;
- medium recall >=0.90 and near recall >=0.95;
- aggregate recall and all-size-FP-adjusted precision >=0.90;
- reviewed-negative false positives =0; and
- exact application-pipeline p95 <=20.0 ms, after exactly three warmup calls.

The threshold CLI options remain available for diagnostic experiments. Any
non-byte-identical rule makes `canonical_release_policy_matched=false` and
`release_evidence_eligible=false`, even if that diagnostic rule passes. Never
derive or adjust either rule from sealed results. Use the repository defaults
for the protected final plan:

```bash
python scripts/evaluate_independent_holdout_runtime.py plan \
  --package local-holdout/prepared-sealed \
  --output local-release-evidence/final-plan.json \
  --dependency-manifest local-release-evidence/windows-directml-py313-DEPENDENCY-MANIFEST.json \
  --hardware-identity local-release-evidence/verified-rx6950-adapter.json \
  --device DML:0 \
  --expected-provider DmlExecutionProvider \
  --detail-crop-size 768
```

The final holdout is not a generic accelerator benchmark: it must run on the
exact authenticated RX 6950 XT through DirectML. The RTX 5060 DirectML and any
CUDA qualification are separate prior physical gates, not alternate devices
for this plan. DirectML requires the RX machine's exact numeric adapter such as
`DML:0`; free-form values such as `DML:evil` are rejected. `AUTO`, generic
`GPU`, CPU, CUDA, square artifacts, a detail setting that
differs from the adopted workload, partial-provider execution, fewer than 2,000
bootstrap samples, or confidence points other than exactly 0.25 and 0.45 are
rejected. Plan creation reads the
self-hashed release pointer and requires `--detail-crop-size` to equal the
adopted tournament winner's production setting: `0` for a primary-only winner,
or the exact positive detail ROI width for a configured winner. The holdout
cannot override the tournament's fail-safe pipeline choice or introduce a new
geometry after selection. The command also reads the self-hashed
`HOLDOUT-MANIFEST.json` metadata so it can bind the package and confirm the
pinned inventory, but it does not call package verification or open the sealed
images/COCO member.

The final plan accepts only `sealed_release_holdout`, requires at least 400
33-64px targets, 250 65-96px targets, 250 >96px targets, and 1,000 independently
reviewed negative images, plus the 15-session diversity minima above. The <=32px
target recall/count remains descriptive and cannot pass or fail a recall gate;
unmatched predictions of every size still enter aggregate precision and
false-positive controls. There is no dataset-YAML argument, split argument, or
grouped-v9 path: labels and capture-session mappings are read directly from the
sealed normalized COCO package.

## Evaluate, consume, and retire in one transaction

Run the frozen plan exactly once. The evaluator, not the workflow caller,
records both UTC transitions from its own clock so that the ledger cannot
claim consumption or retirement before those transitions actually occur:

```bash
python scripts/evaluate_independent_holdout_runtime.py evaluate \
  --package local-holdout/prepared-sealed \
  --plan local-release-evidence/final-plan.json \
  --output local-release-evidence/final-runtime \
  --dependency-manifest local-release-evidence/windows-directml-py313-DEPENDENCY-MANIFEST.json \
  --hardware-identity local-release-evidence/verified-rx6950-adapter.json \
  --event-id release-eval-001 \
  --actor-id release-operator \
  --retirement-event-id release-eval-001-retired \
  --retirement-reason 'exact frozen release evaluation completed'
```

Do not run this one-time evaluation immediately after development selection.
First exercise the provisionally adopted exact workload on the physical target
GPU and frozen build, and require its latency and compatibility gates to pass.
This ordering prevents a configured or larger winner that is too slow on the
target GPU from consuming the only sealed holdout. Those prerequisite results
do not approve the release; they only determine whether opening the holdout is
warranted.

The evaluator holds the exclusive sealed-ledger lock before verifying or
decoding any package member and keeps it for the whole evaluation. It uses the
same batch-one application detector, rectangular full-frame preprocessing,
and source-space postprocessing as the product. A primary-only adopted workload
runs one pass. A configured workload adds the centered model-aspect detail ROI
and one cross-pass merge. Its scalar detail size is the maximum source ROI
width; the production planner derives and records the applied height from the
adopted model's exact H:W. It requires the plan's exact accelerator provider, disables graph
CPU fallback and runtime provider-failure retry, verifies the static artifact
shape, records primary/detail/end-to-end timings, and rehashes the package,
candidate, plan, and loaded source before atomic no-replace publication. The
ONNX and labels are copied once through no-follow file descriptors into a
private, hash/size-verified snapshot; the detector opens only those snapshot
bytes, and the private directory is removed after the evidence/ledger
transaction.

Immediately after acquiring the exclusive ledger lock and confirming there is
no prior event, the transaction durably appends a schema-v1 consumption event
bound to the exact frozen plan. This happens before package verification opens
any sealed image or COCO member. Any later verification, inference, or
publication failure therefore leaves a durable consumed event and forensic
lock; the holdout is burned and cannot be opened by a new dispatch. Do not
delete the lock and rerun.

After `final-runtime/metrics.json` is durably and atomically visible, the
transaction appends a schema-v2 hash-chained retirement event containing the
exact evidence-file SHA-256. It then verifies and flushes the complete chain
before removing the lock. POSIX uses directory `fsync`; Windows uses
write-capable `CreateFileW` handles plus `FlushFileBuffers` and fails closed if
durability is unavailable. The internally recorded consumption and retirement
times must be strictly increasing.

The report retains raw detected/total, misses, predictions, and false positives
for every size bucket at both fixed confidence points. Every detection on the
dual-reviewed negative images is counted as a false positive. It includes both
image-level uncertainty and a capture-session cluster bootstrap in which all
frames from the sampled source session move together. <=32px ground-truth
recall/misses are descriptive, while every unmatched prediction/false positive
from every bucket contributes to release precision.

Verify the immutable evidence, current adopted candidate, frozen rule, exact
plan bytes, evidence self-hash, and two-event retired ledger chain:

```bash
python scripts/evaluate_independent_holdout_runtime.py verify \
  --evidence local-release-evidence/final-runtime \
  --plan local-release-evidence/final-plan.json \
  --package local-holdout/prepared-sealed \
  --dependency-manifest local-release-evidence/windows-directml-py313-DEPENDENCY-MANIFEST.json \
  --hardware-identity local-release-evidence/verified-rx6950-adapter.json \
  --receipt-output local-release-evidence/INDEPENDENT-HOLDOUT-RECEIPT.json

python scripts/evaluate_independent_holdout_runtime.py verify-receipt \
  --receipt local-release-evidence/INDEPENDENT-HOLDOUT-RECEIPT.json \
  --evidence local-release-evidence/final-runtime \
  --plan local-release-evidence/final-plan.json \
  --package local-holdout/prepared-sealed \
  --dependency-manifest local-release-evidence/windows-directml-py313-DEPENDENCY-MANIFEST.json \
  --hardware-identity local-release-evidence/verified-rx6950-adapter.json

python scripts/evaluate_independent_holdout_runtime.py publish-bundle \
  --output local-release-evidence/publication-inputs \
  --receipt local-release-evidence/INDEPENDENT-HOLDOUT-RECEIPT.json \
  --evidence local-release-evidence/final-runtime \
  --plan local-release-evidence/final-plan.json \
  --package local-holdout/prepared-sealed \
  --dependency-manifest local-release-evidence/windows-directml-py313-DEPENDENCY-MANIFEST.json \
  --hardware-identity local-release-evidence/verified-rx6950-adapter.json

python scripts/evaluate_independent_holdout_runtime.py verify-bundle \
  --bundle local-release-evidence/publication-inputs \
  --evidence local-release-evidence/final-runtime \
  --plan local-release-evidence/final-plan.json \
  --package local-holdout/prepared-sealed \
  --dependency-manifest local-release-evidence/windows-directml-py313-DEPENDENCY-MANIFEST.json \
  --hardware-identity local-release-evidence/verified-rx6950-adapter.json
```

The dependency manifest must be generated from the executing fresh,
hash-locked CPython 3.13.14 Windows DirectML environment. The hardware identity
must be the redacted output of `verify_windows_holdout_adapter.py`, bound to
the exact authenticated RX 6950 XT physical receipt and numeric DXGI/DirectML
adapter. A separately protected attestation job first preflights its own exact
environment, then runs the final `verify-bundle` command with
`--authenticated-evaluation-environment`. That flag is only for an
authenticated cross-runner attestation: it validates the evaluation record
against repository policy without falsely requiring path-sensitive installed
RECORD hashes to equal the attester's fresh environment. Never use it during
plan creation, evaluation, or the evaluation job's own verification.

The atomic/no-replace bundle has a fixed inventory:
`ARTIFACT-MANIFEST.json`, `INDEPENDENT-HOLDOUT-RECEIPT.json`, `metrics.json`,
`evaluation-plan.json`, `ledger/consumed.json`, and `ledger/retired.json`.
`verify-bundle` rechecks those exact bytes against the current adopted pointer,
tagged evaluator/application source snapshot, repository policy, confidential
holdout manifest, and retired ledger. The receipt and bundle self-hashes prove
internal integrity, not authenticity: a release workflow must accept them only
from an authenticated protected evaluation run with the plan committed before
sealed access. Never trust a locally fabricated self-hashed receipt.
The repository-owned thresholds, source snapshot, receipt verifier record, and
public bundle contract have one standard-library-only source:
`utils/independent_holdout_release_contract.py`. Hosted publication validation
must import that contract rather than copy threshold values into a workflow.

A successful canonical frozen metric rule produces
`verified_release_eligible_evidence_not_release_approved`. This is a
machine-verifiable input to manual release review, not approval. The evaluator
does not modify `models/RELEASE-DEFAULT.json`; `approved`, model-accuracy,
target-GPU, frozen-build, independent-holdout, hardware, legal-redistribution,
and release-gate flags remain false. Separate frozen-build/physical-GPU and
license review must still be completed by the release workflow.
This v1 receipt proves absolute-threshold evidence only. It does not evaluate a
paired incumbent and cannot support a claim that far detection "improved" by a
specific amount; that requires a separately predeclared paired-incumbent
adapter and image/session bootstrap comparison.

The lower-level `consume` and `retire` subcommands remain available for legacy
or forensic administration, but they do not produce exact runtime evidence and
must not be substituted for the transactional evaluator in a new release.
Ledger events remain canonical JSON, sequentially hash-chained, strictly
increasing in recorded UTC time, and no-replace. A second consumption,
retirement before consumption, event after retirement, or tampering is rejected.

## Limits and release handling

- This is a reproducibility and leakage-control mechanism, not a DRM boundary.
  A user with filesystem access can open files outside the tool. Restrict OS
  permissions and access to the sealed directory.
- A self-hash detects accidental or unanchored changes only when the expected
  hash is retained elsewhere. Put the initial manifest hash and latest ledger
  event hash in write-protected release evidence. A privileged user could
  otherwise rewrite or truncate both a file and its local hash chain.
- dHash is a useful near-duplicate screen, not proof that two scenes are
  semantically independent. It can have false positives and false negatives.
  Review collisions and capture provenance; never relax the threshold merely
  to admit a desired sample.
- The tool verifies source and extracted-frame hashes but cannot prove that an
  image was actually derived from the declared video/frame index. Preserve the
  extraction command/tool and independently spot-check derivation.
- `redistribution_permitted: false` is allowed for a private evaluation
  package. Because raw evidence is copied into the package, do not upload or
  attach such a package to a public release. Publish only aggregate evaluation
  evidence that the license permits.
- Do not use a released holdout again for later model tuning. Capture and seal a
  new session-level holdout for the next release decision.
