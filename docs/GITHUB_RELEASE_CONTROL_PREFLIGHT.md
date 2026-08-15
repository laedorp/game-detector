# GitHub release-control preflight

`scripts/preflight_github_release_controls.py` performs a read-only audit of
GitHub controls that live outside this repository. It never creates or updates
a ruleset, environment, variable, secret, runner, runner group, tag, artifact,
or release. Its HTTP client accepts only repository/owner-scoped paths and
issues only `GET` requests to `https://api.github.com`.

The report is diagnostic evidence, not release authority. Even a passing
report contains `release_authorization_granted: false`; the authenticated
qualification and publication workflows remain authoritative.

## Running it safely

Use a dedicated, short-lived credential rather than `GH_TOKEN`,
`GITHUB_TOKEN`, a release-workflow token, or a long-lived administrator PAT:

```bash
PROAIM_GITHUB_PREFLIGHT_TOKEN='<short-lived read token>' \
  python scripts/preflight_github_release_controls.py \
  --repository OWNER/REPOSITORY > github-release-control-preflight.json
```

The script reads only `PROAIM_GITHUB_PREFLIGHT_TOKEN`; there is no token
command-line option and no fallback to general GitHub credentials. Do not add
this token to a release workflow. Run the audit from a locked administrator
workstation or a separately protected read-only audit job, then destroy the
credential.

For a fine-grained credential, grant only the read permissions needed by the
selected repository and organization:

- repository Metadata read, Actions read, and Environments read; and
- organization Self-hosted runners read.

Do not grant Contents write, Actions write, Environments write,
Administration write, Secrets write, self-hosted-runner write, or release
write. GitHub may require broader *account access* or a different plan to make
some organization runner-group endpoints visible; that is reported as
`unknown`, not treated as a pass.

Exit code `0` means every API field required by this policy was observed and
matched. Exit code `1` means an observed control failed. Exit code `3`
means at least one control was not verifiable through the available
API/credential. Exit code `2` is invalid input, a malformed response, or
another preflight error.

The JSON report contains policy environment names, check result codes, and
aggregate registered-runner counts. It deliberately omits token material,
secret and variable values, reviewer profiles, team membership, runner names,
runner-group names, response bodies, and host details.

## Repository and `main` ruleset policy

The repository must be active and use `main` as its default branch.
Effective rules returned for `main` must include:

- deletion protection and non-fast-forward protection;
- pull requests with at least one approval, stale-review dismissal, approval
  of the last push by another reviewer, and resolved review threads; and
- at least one strict required status check.

Every contributing ruleset must be active, target branches, apply to `main`,
and have no bypass actor. The report is fail-closed if effective rules do not
carry a numeric ruleset provenance ID.

## Exact protected environments

Every environment must use a custom deployment branch inventory containing
exactly `main`, have exactly one required Team reviewer, prevent self-review,
and have administrator bypass disabled. Extra reviewers broaden approval
authority and therefore fail the exact policy.

| Environment | Exact reviewer Team slug |
| --- | --- |
| `directml-amd_rx_6950_xt-physical-attestation` | `amd_rx_6950_xt-independent-reviewers-v1` |
| `directml-nvidia_rtx_5060_laptop-physical-attestation` | `nvidia_rtx_5060_laptop-independent-reviewers-v1` |
| `directml-release-publication` | `directml-publication-independent-reviewers-v1` |
| `independent-holdout-access` | `independent-holdout-access-reviewers-v1` |
| `independent-holdout-attestation` | `independent-holdout-attestation-reviewers-v1` |
| `cuda-physical-attestation` | `cuda-physical-attestation-independent-reviewers-v1` |
| `cuda-release-publication` | `cuda-release-publication-independent-reviewers-v1` |

The environment variable inventory and values must exactly match the workflow
contract:

- both DirectML physical-attestation environments:
  `DIRECTML_PHYSICAL_ATTESTATION_POLICY_VERSION=required-reviewers-v1` and
  the role-specific `DIRECTML_INDEPENDENT_REVIEWER_GROUP` Team slug above;
- `directml-release-publication`:
  `DIRECTML_RELEASE_ENVIRONMENT_POLICY_VERSION=required-reviewers-v1` and
  `DIRECTML_RELEASE_REVIEWER_GROUP=directml-publication-independent-reviewers-v1`;
- `independent-holdout-access`:
  `INDEPENDENT_HOLDOUT_ACCESS_POLICY_VERSION=required-reviewers-v1`,
  `INDEPENDENT_HOLDOUT_ACCESS_REVIEWER_GROUP=independent-holdout-access-reviewers-v1`,
  and `INDEPENDENT_HOLDOUT_RUNNER_POLICY_VERSION=sealed-directml-v1`;
- `independent-holdout-attestation`:
  `INDEPENDENT_HOLDOUT_ATTESTATION_POLICY_VERSION=required-reviewers-v1` and
  `INDEPENDENT_HOLDOUT_ATTESTATION_REVIEWER_GROUP=independent-holdout-attestation-reviewers-v1`;
- `cuda-physical-attestation`:
  `CUDA_PHYSICAL_ATTESTATION_POLICY_VERSION=required-reviewers-v1`; and
- `cuda-release-publication`:
  `CUDA_RELEASE_ENVIRONMENT_POLICY_VERSION=required-reviewers-v1`.

The secret-name inventory must also be exact. The preflight requests only the
GitHub endpoint that lists secret metadata; GitHub never returns secret values.
Required names are:

- `DIRECTML_PHYSICAL_ATTESTATION_GUARD` in each DirectML physical
  environment;
- `DIRECTML_RELEASE_ENVIRONMENT_GUARD` in DirectML publication;
- `INDEPENDENT_HOLDOUT_ACCESS_GUARD`,
  `INDEPENDENT_HOLDOUT_PACKAGE_PATH`, and
  `INDEPENDENT_HOLDOUT_LEDGER_PATH` in holdout access;
- `INDEPENDENT_HOLDOUT_ATTESTATION_GUARD`,
  `INDEPENDENT_HOLDOUT_PACKAGE_PATH`, and
  `INDEPENDENT_HOLDOUT_LEDGER_PATH` in holdout attestation;
- `CUDA_PHYSICAL_ATTESTATION_GUARD` in CUDA physical attestation; and
- `CUDA_RELEASE_ENVIRONMENT_GUARD` in CUDA publication.

## Exact self-hosted runner boundaries

The organization must provide one dedicated runner group per runner class.
Each group must be visible to exactly this repository, contain no unrelated
runner class, and be restricted to exactly one workflow at
`refs/heads/main`:

| Runner class | Minimum registrations | Required labels | Only allowed workflow |
| --- | ---: | --- | --- |
| CUDA physical | 1 | `self-hosted`, `Windows`, `X64`, `proaim-cuda-qualification` | `.github/workflows/qualify-windows-cuda.yml` |
| DirectML physical | 2 | `self-hosted`, `Windows`, `X64`, `proaim-directml-qualification` | `.github/workflows/qualify-windows-directml.yml` |
| Independent holdout | 1 | `self-hosted`, `Windows`, `X64`, `proaim-independent-holdout-directml`, `proaim-rx-6950-xt-holdout` | `.github/workflows/qualify-independent-holdout.yml` |

Offline registered runners still count; availability is operational state,
not an access-control property. A personal-account repository cannot satisfy
the required organization runner-group boundary and fails the preflight.

## API and human-control limits

The preflight deliberately exposes these limits instead of guessing:

- GitHub's environment `GET` schema does not consistently expose the
  “administrators may bypass” setting. An absent field is `unknown`; confirm
  it in repository settings or an authenticated audit log.
- GitHub's deployment-policy list can omit whether a `main` pattern targets a
  branch or a tag. An omitted target type is `unknown`, never assumed to mean
  the protected `main` branch.
- GitHub withholds `ruleset.bypass_actors` unless the caller has write access
  to the ruleset. Do not grant write permission merely to make this diagnostic
  green. Treat an absent field as `unknown` and perform a separate protected
  settings/audit review.
- Organization runner-group endpoints can return `403`/`404` when the plan
  or credential cannot read them. Labels alone are never accepted as proof of
  repository/workflow isolation.
- A Team slug cannot prove that its current members are independent of the
  dispatcher, trainer, observer, or last pusher. Check membership and the
  actual approving identity for each release.
- The API cannot prove the physical GPU, interactive desktop, runner binary,
  OS account, filesystem ACL, sealed mount, append-only ledger, or absence of
  unrelated processes. Retain the physical qualification, hardware identity,
  and holdout evidence gates.
- Secret metadata proves only that an environment-scoped name exists. It
  cannot prove a secret's value, freshness, filesystem target, or that a
  repository/organization secret with the same name is absent elsewhere.
- The REST reads are not an atomic snapshot. Controls can change between GETs;
  preserve GitHub audit evidence and recheck immediately before a release.

These limitations are why this script is not wired into a publication job and
why a passing API report never grants release authorization.

API behavior is pinned to GitHub REST version `2026-03-10`. Primary
references: [repository rulesets](https://docs.github.com/en/rest/repos/rules),
[deployment environments](https://docs.github.com/en/rest/deployments/environments),
[deployment branch policies](https://docs.github.com/en/rest/deployments/branch-policies),
[environment variables](https://docs.github.com/en/rest/actions/variables),
[environment secrets](https://docs.github.com/en/rest/actions/secrets), and
[self-hosted runner groups](https://docs.github.com/en/rest/actions/self-hosted-runner-groups).
