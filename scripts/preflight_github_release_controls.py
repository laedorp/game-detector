#!/usr/bin/env python3
"""Read-only preflight for ProAim's external GitHub release controls.

The repository workflows deliberately do not create their own environments,
reviewers, secrets, rulesets, or self-hosted runner groups.  This module reads
those controls through GitHub's REST API and emits a privacy-reduced report.
Every network request is a GET.  Secret values, variable values, reviewer
profiles, runner names, and runner-group names are never copied to the report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import re
import sys
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


API_ROOT = "https://api.github.com"
API_VERSION = "2026-03-10"
TOKEN_ENVIRONMENT_VARIABLE = "PROAIM_GITHUB_PREFLIGHT_TOKEN"
REPORT_KIND = "proaim-github-release-control-preflight"
REPORT_SCHEMA_VERSION = 1
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
PAGE_SIZE = 30
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"


class PreflightError(RuntimeError):
    """Raised for invalid input or malformed GitHub data."""


class ApiAccessError(PreflightError):
    """A redacted GitHub API failure.

    Response bodies are intentionally discarded: an upstream error can echo
    private organization or runner details and must not enter public logs.
    """

    def __init__(self, status: int | None, path: str, reason: str) -> None:
        super().__init__(f"GitHub API GET {path} failed ({reason})")
        self.status = status
        self.path = path
        self.reason = reason


class GitHubReadApi(Protocol):
    def get_object(self, path: str) -> dict[str, Any]: ...

    def get_list(self, path: str) -> list[dict[str, Any]]: ...

    def get_paginated_object(self, path: str, key: str) -> list[dict[str, Any]]: ...

    def get_paginated_list(self, path: str) -> list[dict[str, Any]]: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        request: Request,
        fp: Any,
        code: int,
        message: str,
        headers: Mapping[str, str],
        new_url: str,
    ) -> None:
        return None


def _strict_json(payload: bytes, context: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise PreflightError(f"{context} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PreflightError(f"{context} contains a non-finite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{context} is not strict UTF-8 JSON") from exc


class ReadOnlyGitHubApi:
    """Minimal GitHub client whose only request method is GET."""

    def __init__(self, token: str, repository: str) -> None:
        normalized_token = token.strip()
        if not normalized_token:
            raise PreflightError(f"{TOKEN_ENVIRONMENT_VARIABLE} is required")
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in normalized_token):
            raise PreflightError(f"{TOKEN_ENVIRONMENT_VARIABLE} contains a control character")
        _split_repository(repository)
        self._token = normalized_token
        self.repository = repository
        self._repo_prefix = f"/repos/{repository}"
        self._org_prefix = f"/orgs/{repository.split('/', 1)[0]}"

    def _validate_path(self, path: str) -> None:
        if not isinstance(path, str) or not path.startswith("/"):
            raise PreflightError("GitHub API path must be root-relative")
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise PreflightError("GitHub API path must not be an absolute URL")
        if any(ord(character) < 0x20 for character in path):
            raise PreflightError("GitHub API path contains a control character")
        path_parts = parsed.path.split("/")
        if ".." in path_parts or "." in path_parts:
            raise PreflightError("GitHub API path contains traversal")
        if not (
            parsed.path == self._repo_prefix
            or parsed.path.startswith(self._repo_prefix + "/")
            or parsed.path == self._org_prefix
            or parsed.path.startswith(self._org_prefix + "/")
        ):
            raise PreflightError("GitHub API path escapes the selected repository/owner")

    def _get(self, path: str) -> Any:
        self._validate_path(path)
        request = Request(API_ROOT + path, method="GET")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("User-Agent", "ProAim-read-only-release-control-preflight")
        request.add_header("X-GitHub-Api-Version", API_VERSION)
        try:
            with build_opener(_NoRedirect()).open(request, timeout=60) as response:
                if int(response.status) != 200:
                    raise ApiAccessError(int(response.status), path, "unexpected-status")
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise PreflightError("GitHub API response exceeded the size limit")
        except HTTPError as exc:
            # Never read or render the response body.
            raise ApiAccessError(int(exc.code), path, f"http-{exc.code}") from exc
        except (OSError, URLError) as exc:
            raise ApiAccessError(None, path, "transport-error") from exc
        return _strict_json(payload, f"GitHub API response for {path}")

    def get_object(self, path: str) -> dict[str, Any]:
        value = self._get(path)
        if not isinstance(value, dict):
            raise PreflightError(f"GitHub API response for {path} is not an object")
        return value

    def get_list(self, path: str) -> list[dict[str, Any]]:
        value = self._get(path)
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise PreflightError(f"GitHub API response for {path} is not an object list")
        return value

    def get_paginated_object(self, path: str, key: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            payload = self.get_object(
                f"{path}{separator}per_page={PAGE_SIZE}&page={page}"
            )
            values = payload.get(key)
            if not isinstance(values, list) or any(
                not isinstance(item, dict) for item in values
            ):
                raise PreflightError(f"GitHub API pagination omitted object list {key!r}")
            result.extend(values)
            if len(values) < PAGE_SIZE:
                return result
        raise PreflightError(f"GitHub API pagination exceeded 100 pages for {path}")

    def get_paginated_list(self, path: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            values = self.get_list(
                f"{path}{separator}per_page={PAGE_SIZE}&page={page}"
            )
            result.extend(values)
            if len(values) < PAGE_SIZE:
                return result
        raise PreflightError(f"GitHub API pagination exceeded 100 pages for {path}")


@dataclass(frozen=True)
class EnvironmentPolicy:
    name: str
    reviewer_team_slug: str
    variables: tuple[tuple[str, str], ...]
    secrets: tuple[str, ...]


@dataclass(frozen=True)
class RunnerPolicy:
    key: str
    custom_label: str
    required_labels: tuple[str, ...]
    workflow_path: str
    minimum_runners: int


ENVIRONMENT_POLICIES = (
    EnvironmentPolicy(
        name="directml-amd_rx_6950_xt-physical-attestation",
        reviewer_team_slug="amd_rx_6950_xt-independent-reviewers-v1",
        variables=(
            ("DIRECTML_INDEPENDENT_REVIEWER_GROUP", "amd_rx_6950_xt-independent-reviewers-v1"),
            ("DIRECTML_PHYSICAL_ATTESTATION_POLICY_VERSION", "required-reviewers-v1"),
        ),
        secrets=("DIRECTML_PHYSICAL_ATTESTATION_GUARD",),
    ),
    EnvironmentPolicy(
        name="directml-nvidia_rtx_5060_laptop-physical-attestation",
        reviewer_team_slug="nvidia_rtx_5060_laptop-independent-reviewers-v1",
        variables=(
            (
                "DIRECTML_INDEPENDENT_REVIEWER_GROUP",
                "nvidia_rtx_5060_laptop-independent-reviewers-v1",
            ),
            ("DIRECTML_PHYSICAL_ATTESTATION_POLICY_VERSION", "required-reviewers-v1"),
        ),
        secrets=("DIRECTML_PHYSICAL_ATTESTATION_GUARD",),
    ),
    EnvironmentPolicy(
        name="directml-release-publication",
        reviewer_team_slug="directml-publication-independent-reviewers-v1",
        variables=(
            (
                "DIRECTML_RELEASE_ENVIRONMENT_POLICY_VERSION",
                "required-reviewers-v1",
            ),
            (
                "DIRECTML_RELEASE_REVIEWER_GROUP",
                "directml-publication-independent-reviewers-v1",
            ),
        ),
        secrets=("DIRECTML_RELEASE_ENVIRONMENT_GUARD",),
    ),
    EnvironmentPolicy(
        name="independent-holdout-access",
        reviewer_team_slug="independent-holdout-access-reviewers-v1",
        variables=(
            ("INDEPENDENT_HOLDOUT_ACCESS_POLICY_VERSION", "required-reviewers-v1"),
            (
                "INDEPENDENT_HOLDOUT_ACCESS_REVIEWER_GROUP",
                "independent-holdout-access-reviewers-v1",
            ),
            ("INDEPENDENT_HOLDOUT_RUNNER_POLICY_VERSION", "sealed-directml-v1"),
        ),
        secrets=(
            "INDEPENDENT_HOLDOUT_ACCESS_GUARD",
            "INDEPENDENT_HOLDOUT_LEDGER_PATH",
            "INDEPENDENT_HOLDOUT_PACKAGE_PATH",
        ),
    ),
    EnvironmentPolicy(
        name="independent-holdout-attestation",
        reviewer_team_slug="independent-holdout-attestation-reviewers-v1",
        variables=(
            (
                "INDEPENDENT_HOLDOUT_ATTESTATION_POLICY_VERSION",
                "required-reviewers-v1",
            ),
            (
                "INDEPENDENT_HOLDOUT_ATTESTATION_REVIEWER_GROUP",
                "independent-holdout-attestation-reviewers-v1",
            ),
        ),
        secrets=(
            "INDEPENDENT_HOLDOUT_ATTESTATION_GUARD",
            "INDEPENDENT_HOLDOUT_LEDGER_PATH",
            "INDEPENDENT_HOLDOUT_PACKAGE_PATH",
        ),
    ),
    EnvironmentPolicy(
        name="cuda-physical-attestation",
        reviewer_team_slug="cuda-physical-attestation-independent-reviewers-v1",
        variables=(("CUDA_PHYSICAL_ATTESTATION_POLICY_VERSION", "required-reviewers-v1"),),
        secrets=("CUDA_PHYSICAL_ATTESTATION_GUARD",),
    ),
    EnvironmentPolicy(
        name="cuda-release-publication",
        reviewer_team_slug="cuda-release-publication-independent-reviewers-v1",
        variables=(("CUDA_RELEASE_ENVIRONMENT_POLICY_VERSION", "required-reviewers-v1"),),
        secrets=("CUDA_RELEASE_ENVIRONMENT_GUARD",),
    ),
)


RUNNER_POLICIES = (
    RunnerPolicy(
        key="cuda-physical",
        custom_label="proaim-cuda-qualification",
        required_labels=("self-hosted", "Windows", "X64", "proaim-cuda-qualification"),
        workflow_path=".github/workflows/qualify-windows-cuda.yml",
        minimum_runners=1,
    ),
    RunnerPolicy(
        key="directml-physical",
        custom_label="proaim-directml-qualification",
        required_labels=(
            "self-hosted",
            "Windows",
            "X64",
            "proaim-directml-qualification",
        ),
        workflow_path=".github/workflows/qualify-windows-directml.yml",
        minimum_runners=2,
    ),
    RunnerPolicy(
        key="independent-holdout",
        custom_label="proaim-independent-holdout-directml",
        required_labels=(
            "self-hosted",
            "Windows",
            "X64",
            "proaim-independent-holdout-directml",
            "proaim-rx-6950-xt-holdout",
        ),
        workflow_path=".github/workflows/qualify-independent-holdout.yml",
        minimum_runners=1,
    ),
)


def _split_repository(repository: str) -> tuple[str, str]:
    if not isinstance(repository, str) or not REPOSITORY_RE.fullmatch(repository):
        raise PreflightError("repository must use exact OWNER/REPOSITORY syntax")
    owner, name = repository.split("/", 1)
    if owner in {".", ".."} or name in {".", ".."}:
        raise PreflightError("repository identity contains traversal")
    return owner, name


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check(status: str, code: str) -> dict[str, str]:
    if status not in {PASS, FAIL, UNKNOWN}:
        raise AssertionError(f"invalid check status {status!r}")
    return {"status": status, "code": code}


def _overall(checks: Mapping[str, Mapping[str, str]]) -> str:
    statuses = [value.get("status") for value in checks.values()]
    if FAIL in statuses:
        return FAIL
    if UNKNOWN in statuses:
        return UNKNOWN
    return PASS


def _safe_api_failure(exc: ApiAccessError) -> dict[str, str]:
    if exc.status == 404:
        return _check(FAIL, "api-resource-not-found")
    if exc.status in {401, 403}:
        return _check(UNKNOWN, "api-read-permission-unavailable")
    return _check(UNKNOWN, "api-read-unavailable")


def _object_list(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        return None
    return value


def _main_ruleset_report(
    api: GitHubReadApi,
    repository: str,
    repository_record: Mapping[str, Any] | None,
    repository_metadata_check: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    prefix = f"/repos/{repository}"
    checks: dict[str, dict[str, str]] = {}
    if repository_record is None:
        checks["repository_metadata"] = dict(
            repository_metadata_check
            or _check(UNKNOWN, "repository-metadata-unavailable")
        )
        return {"branch": "main", "status": _overall(checks), "checks": checks}

    repository_identity = (
        repository_record.get("full_name") == repository
        and _is_int(repository_record.get("id"))
        and repository_record["id"] > 0
        and isinstance(repository_record.get("private"), bool)
    )
    checks["repository_identity"] = _check(
        PASS if repository_identity else FAIL,
        "repository-identity-exact"
        if repository_identity
        else "repository-identity-mismatch-or-malformed",
    )
    checks["default_branch"] = _check(
        PASS if repository_record.get("default_branch") == "main" else FAIL,
        "default-branch-main"
        if repository_record.get("default_branch") == "main"
        else "default-branch-not-main",
    )
    repository_active = (
        repository_record.get("archived") is False
        and repository_record.get("disabled") is False
    )
    checks["repository_active"] = _check(
        PASS if repository_active else FAIL,
        "repository-active" if repository_active else "repository-archived-or-disabled",
    )

    try:
        rules = api.get_paginated_list(prefix + "/rules/branches/main")
    except ApiAccessError as exc:
        checks["effective_rules"] = _safe_api_failure(exc)
        return {"branch": "main", "status": _overall(checks), "checks": checks}

    rules_by_type: dict[str, list[dict[str, Any]]] = {}
    malformed_rule = False
    for rule in rules:
        rule_type = rule.get("type")
        if not isinstance(rule_type, str) or not rule_type:
            malformed_rule = True
            continue
        rules_by_type.setdefault(rule_type, []).append(rule)
    if malformed_rule:
        checks["effective_rule_schema"] = _check(FAIL, "malformed-effective-rule")
    else:
        checks["effective_rule_schema"] = _check(PASS, "effective-rule-schema-valid")

    for required_type in ("deletion", "non_fast_forward"):
        present = bool(rules_by_type.get(required_type))
        checks[required_type] = _check(
            PASS if present else FAIL,
            f"{required_type}-active" if present else f"{required_type}-missing",
        )

    pull_request_ok = False
    for rule in rules_by_type.get("pull_request", []):
        parameters = rule.get("parameters")
        if not isinstance(parameters, dict):
            continue
        review_count = parameters.get("required_approving_review_count")
        if (
            _is_int(review_count)
            and review_count >= 1
            and parameters.get("dismiss_stale_reviews_on_push") is True
            and parameters.get("require_last_push_approval") is True
            and parameters.get("required_review_thread_resolution") is True
        ):
            pull_request_ok = True
            break
    checks["pull_request"] = _check(
        PASS if pull_request_ok else FAIL,
        "pull-request-review-policy-active"
        if pull_request_ok
        else "pull-request-review-policy-insufficient",
    )

    status_checks_ok = False
    for rule in rules_by_type.get("required_status_checks", []):
        parameters = rule.get("parameters")
        if not isinstance(parameters, dict):
            continue
        contexts = parameters.get("required_status_checks")
        if (
            parameters.get("strict_required_status_checks_policy") is True
            and isinstance(contexts, list)
            and len(contexts) >= 1
            and all(
                isinstance(item, dict)
                and isinstance(item.get("context"), str)
                and bool(item["context"].strip())
                for item in contexts
            )
        ):
            status_checks_ok = True
            break
    checks["required_status_checks"] = _check(
        PASS if status_checks_ok else FAIL,
        "strict-status-checks-active"
        if status_checks_ok
        else "strict-status-checks-missing-or-empty",
    )

    relevant_types = {"deletion", "non_fast_forward", "pull_request", "required_status_checks"}
    ruleset_ids: set[int] = set()
    bad_ruleset_id = False
    for rule_type in relevant_types:
        for rule in rules_by_type.get(rule_type, []):
            ruleset_id = rule.get("ruleset_id")
            if _is_int(ruleset_id) and ruleset_id > 0:
                ruleset_ids.add(ruleset_id)
            else:
                bad_ruleset_id = True
    checks["ruleset_provenance"] = _check(
        PASS if ruleset_ids and not bad_ruleset_id else FAIL,
        "effective-rules-have-ruleset-provenance"
        if ruleset_ids and not bad_ruleset_id
        else "effective-rule-ruleset-provenance-missing",
    )

    bypass_unknown = False
    ruleset_detail_valid = bool(ruleset_ids)
    bypass_empty = bool(ruleset_ids)
    for ruleset_id in sorted(ruleset_ids):
        try:
            detail = api.get_object(prefix + f"/rulesets/{ruleset_id}?includes_parents=true")
        except ApiAccessError as exc:
            checks["ruleset_details"] = _safe_api_failure(exc)
            ruleset_detail_valid = False
            bypass_unknown = True
            continue
        if (
            not _is_int(detail.get("id"))
            or detail["id"] != ruleset_id
            or detail.get("enforcement") != "active"
            or detail.get("target") != "branch"
        ):
            ruleset_detail_valid = False
        conditions = detail.get("conditions")
        ref_name = conditions.get("ref_name") if isinstance(conditions, dict) else None
        include = ref_name.get("include") if isinstance(ref_name, dict) else None
        exclude = ref_name.get("exclude") if isinstance(ref_name, dict) else None
        if not (
            isinstance(include, list)
            and all(isinstance(value, str) for value in include)
            and any(value in {"refs/heads/main", "~DEFAULT_BRANCH", "~ALL"} for value in include)
            and isinstance(exclude, list)
            and all(isinstance(value, str) for value in exclude)
            and "refs/heads/main" not in exclude
        ):
            ruleset_detail_valid = False
        if "bypass_actors" not in detail:
            bypass_unknown = True
            bypass_empty = False
        elif detail.get("bypass_actors") != []:
            bypass_empty = False

    if "ruleset_details" not in checks:
        checks["ruleset_details"] = _check(
            PASS if ruleset_detail_valid else FAIL,
            "active-main-ruleset-details-valid"
            if ruleset_detail_valid
            else "active-main-ruleset-details-invalid",
        )
    if bypass_unknown:
        checks["ruleset_bypass"] = _check(
            UNKNOWN,
            "ruleset-bypass-actors-withheld-by-read-api",
        )
    else:
        checks["ruleset_bypass"] = _check(
            PASS if bypass_empty else FAIL,
            "ruleset-bypass-empty" if bypass_empty else "ruleset-bypass-present",
        )
    return {"branch": "main", "status": _overall(checks), "checks": checks}


def _reviewer_team_slug(rule: Mapping[str, Any]) -> str | None:
    if not _is_int(rule.get("id")) or rule["id"] <= 0:
        return None
    reviewers = _object_list(rule.get("reviewers"))
    if reviewers is None or len(reviewers) != 1:
        return None
    reviewer = reviewers[0]
    if reviewer.get("type") != "Team":
        return None
    identity = reviewer.get("reviewer")
    if not isinstance(identity, dict):
        return None
    if not _is_int(identity.get("id")) or identity["id"] <= 0:
        return None
    slug = identity.get("slug")
    return slug if isinstance(slug, str) and slug else None


def _named_inventory(values: Sequence[Mapping[str, Any]]) -> set[str] | None:
    names: set[str] = set()
    for value in values:
        name = value.get("name")
        if not isinstance(name, str) or not name or name in names:
            return None
        names.add(name)
    return names


def _environment_report(
    api: GitHubReadApi,
    repository: str,
    policy: EnvironmentPolicy,
) -> dict[str, Any]:
    encoded_name = quote(policy.name, safe="")
    prefix = f"/repos/{repository}/environments/{encoded_name}"
    checks: dict[str, dict[str, str]] = {}
    try:
        environment = api.get_object(prefix)
    except ApiAccessError as exc:
        checks["environment"] = _safe_api_failure(exc)
        return {"name": policy.name, "status": _overall(checks), "checks": checks}

    name_ok = (
        environment.get("name") == policy.name
        and _is_int(environment.get("id"))
        and environment["id"] > 0
    )
    checks["environment"] = _check(
        PASS if name_ok else FAIL,
        "environment-present" if name_ok else "environment-identity-mismatch",
    )

    protection_rules = _object_list(environment.get("protection_rules"))
    reviewer_rules = (
        [rule for rule in protection_rules if rule.get("type") == "required_reviewers"]
        if protection_rules is not None
        else []
    )
    reviewer_ok = (
        len(reviewer_rules) == 1
        and reviewer_rules[0].get("prevent_self_review") is True
        and _reviewer_team_slug(reviewer_rules[0]) == policy.reviewer_team_slug
    )
    checks["required_reviewers"] = _check(
        PASS if reviewer_ok else FAIL,
        "exact-reviewer-team-and-self-review-prevention"
        if reviewer_ok
        else "reviewer-team-or-self-review-control-mismatch",
    )

    if "can_admins_bypass" not in environment:
        checks["administrator_bypass"] = _check(
            UNKNOWN,
            "environment-admin-bypass-not-exposed-by-read-api",
        )
    else:
        no_admin_bypass = environment.get("can_admins_bypass") is False
        checks["administrator_bypass"] = _check(
            PASS if no_admin_bypass else FAIL,
            "environment-admin-bypass-disabled"
            if no_admin_bypass
            else "environment-admin-bypass-enabled-or-malformed",
        )

    deployment_policy = environment.get("deployment_branch_policy")
    custom_main_mode = (
        isinstance(deployment_policy, dict)
        and deployment_policy.get("protected_branches") is False
        and deployment_policy.get("custom_branch_policies") is True
    )
    checks["deployment_branch_mode"] = _check(
        PASS if custom_main_mode else FAIL,
        "custom-branch-policy-mode"
        if custom_main_mode
        else "deployment-branch-policy-mode-mismatch",
    )
    try:
        branch_policies = api.get_paginated_object(
            prefix + "/deployment-branch-policies", "branch_policies"
        )
        branch_names = _named_inventory(branch_policies)
        exact_main = (
            branch_names == {"main"}
            and len(branch_policies) == 1
            and _is_int(branch_policies[0].get("id"))
            and branch_policies[0]["id"] > 0
        )
        target_type = branch_policies[0].get("type") if exact_main else None
        if exact_main and target_type is None:
            checks["deployment_branches"] = _check(
                UNKNOWN,
                "deployment-policy-target-type-not-exposed-by-read-api",
            )
        else:
            exact_main_branch = exact_main and target_type == "branch"
            checks["deployment_branches"] = _check(
                PASS if exact_main_branch else FAIL,
                "deployment-branch-exact-main"
                if exact_main_branch
                else "deployment-branch-inventory-or-target-mismatch",
            )
    except ApiAccessError as exc:
        checks["deployment_branches"] = _safe_api_failure(exc)

    try:
        variables = api.get_paginated_object(prefix + "/variables", "variables")
        observed: dict[str, str] = {}
        malformed = False
        for variable in variables:
            name = variable.get("name")
            value = variable.get("value")
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or name in observed
            ):
                malformed = True
                continue
            observed[name] = value
        expected = dict(policy.variables)
        variables_ok = not malformed and observed == expected
        checks["variables"] = _check(
            PASS if variables_ok else FAIL,
            "environment-variable-inventory-and-values-exact"
            if variables_ok
            else "environment-variable-inventory-or-value-mismatch",
        )
    except ApiAccessError as exc:
        checks["variables"] = _safe_api_failure(exc)

    try:
        secrets = api.get_paginated_object(prefix + "/secrets", "secrets")
        secret_names = _named_inventory(secrets)
        secrets_ok = secret_names == set(policy.secrets)
        checks["secret_names"] = _check(
            PASS if secrets_ok else FAIL,
            "environment-secret-name-inventory-exact"
            if secrets_ok
            else "environment-secret-name-inventory-mismatch",
        )
    except ApiAccessError as exc:
        checks["secret_names"] = _safe_api_failure(exc)

    return {"name": policy.name, "status": _overall(checks), "checks": checks}


def _label_names(runner: Mapping[str, Any]) -> set[str] | None:
    labels = _object_list(runner.get("labels"))
    if labels is None:
        return None
    names = _named_inventory(labels)
    return names


def _runner_reports(
    api: GitHubReadApi,
    repository: str,
    repository_record: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    owner, repository_name = _split_repository(repository)
    unavailable = [
        {
            "class": policy.key,
            "status": UNKNOWN,
            "registered_runner_count": 0,
            "checks": {
                "runner_group_api": _check(UNKNOWN, "runner-group-api-unavailable")
            },
        }
        for policy in RUNNER_POLICIES
    ]
    if repository_record is None:
        return unavailable
    owner_record = repository_record.get("owner")
    repository_identity = (
        repository_record.get("full_name") == repository
        and _is_int(repository_record.get("id"))
        and repository_record["id"] > 0
        and isinstance(repository_record.get("private"), bool)
        and isinstance(owner_record, dict)
        and owner_record.get("login") == owner
    )
    if not repository_identity:
        for report in unavailable:
            report["status"] = FAIL
            report["checks"]["runner_group_api"] = _check(
                FAIL, "repository-owner-identity-mismatch"
            )
        return unavailable
    if owner_record.get("type") != "Organization":
        for report in unavailable:
            report["status"] = FAIL
            report["checks"]["runner_group_api"] = _check(
                FAIL, "organization-runner-group-required"
            )
        return unavailable

    group_path = (
        f"/orgs/{quote(owner, safe='')}/actions/runner-groups"
        f"?visible_to_repository={quote(repository_name, safe='')}"
    )
    try:
        groups = api.get_paginated_object(group_path, "runner_groups")
    except ApiAccessError as exc:
        check = (
            _check(UNKNOWN, "runner-group-api-unavailable-by-plan-or-permission")
            if exc.status in {401, 403, 404}
            else _check(UNKNOWN, "runner-group-api-read-unavailable")
        )
        for report in unavailable:
            report["checks"]["runner_group_api"] = check
            report["status"] = _overall(report["checks"])
        return unavailable

    group_records: list[dict[str, Any]] = []
    seen_group_ids: set[int] = set()
    for group in groups:
        group_id = group.get("id")
        if (
            not _is_int(group_id)
            or group_id <= 0
            or group_id in seen_group_ids
        ):
            return unavailable
        seen_group_ids.add(group_id)
        base = f"/orgs/{quote(owner, safe='')}/actions/runner-groups/{group_id}"
        try:
            detail = api.get_object(base)
            runners = api.get_paginated_object(base + "/runners", "runners")
        except ApiAccessError:
            # A single inaccessible group makes the runner inventory incomplete.
            return unavailable
        if not any(
            (labels := _label_names(runner)) is not None
            and any(policy.custom_label in labels for policy in RUNNER_POLICIES)
            for runner in runners
        ):
            continue
        repositories: list[dict[str, Any]] | None = None
        if detail.get("visibility") == "selected":
            try:
                repositories = api.get_paginated_object(
                    base + "/repositories", "repositories"
                )
            except ApiAccessError:
                repositories = None
        group_records.append(
            {
                "id": group_id,
                "detail": detail,
                "runners": runners,
                "repositories": repositories,
            }
        )

    repository_public = repository_record.get("private") is False
    reports: list[dict[str, Any]] = []
    matched_groups_by_policy: dict[str, list[dict[str, Any]]] = {
        policy.key: [] for policy in RUNNER_POLICIES
    }
    for group_record in group_records:
        group_runners = group_record["runners"]
        for policy in RUNNER_POLICIES:
            if any(
                (labels := _label_names(runner)) is not None
                and policy.custom_label in labels
                for runner in group_runners
            ):
                matched_groups_by_policy[policy.key].append(group_record)

    for policy in RUNNER_POLICIES:
        checks: dict[str, dict[str, str]] = {}
        matched_groups = matched_groups_by_policy[policy.key]
        one_group = len(matched_groups) == 1
        checks["dedicated_group"] = _check(
            PASS if one_group else FAIL,
            "exactly-one-dedicated-runner-group"
            if one_group
            else "missing-or-multiple-runner-groups",
        )
        matched_runners: list[dict[str, Any]] = []
        if one_group:
            group_record = matched_groups[0]
            detail = group_record["detail"]
            group_runners = group_record["runners"]
            for runner in group_runners:
                labels = _label_names(runner)
                if labels is not None and policy.custom_label in labels:
                    matched_runners.append(runner)

            required_labels = set(policy.required_labels)
            labels_ok = (
                len(matched_runners) >= policy.minimum_runners
                and len(matched_runners) == len(group_runners)
                and all(
                    _is_int(runner.get("id"))
                    and runner["id"] > 0
                    and runner.get("os") == "windows"
                    and (labels := _label_names(runner)) is not None
                    and required_labels.issubset(labels)
                    for runner in matched_runners
                )
            )
            checks["runner_labels"] = _check(
                PASS if labels_ok else FAIL,
                "required-labels-on-dedicated-windows-runners"
                if labels_ok
                else "runner-count-label-or-dedication-mismatch",
            )

            repositories = group_record["repositories"]
            full_names: set[str] | None = None
            if isinstance(repositories, list):
                observed_names: set[str] = set()
                valid_repositories = True
                for value in repositories:
                    full_name = value.get("full_name")
                    repository_id = value.get("id")
                    if (
                        not isinstance(full_name, str)
                        or not full_name
                        or full_name in observed_names
                        or not _is_int(repository_id)
                        or repository_id <= 0
                        or (
                            full_name == repository
                            and repository_id != repository_record["id"]
                        )
                    ):
                        valid_repositories = False
                        break
                    observed_names.add(full_name)
                if valid_repositories:
                    full_names = observed_names
            repository_scope_ok = (
                _is_int(detail.get("id"))
                and detail["id"] == group_record["id"]
                and detail.get("visibility") == "selected"
                and full_names == {repository}
                and detail.get("allows_public_repositories") is repository_public
            )
            checks["repository_scope"] = _check(
                PASS if repository_scope_ok else FAIL,
                "runner-group-exact-repository-scope"
                if repository_scope_ok
                else "runner-group-repository-scope-mismatch",
            )

            expected_workflow = (
                f"{repository}/{policy.workflow_path}@refs/heads/main"
            )
            selected_workflows = detail.get("selected_workflows")
            workflow_scope_ok = (
                detail.get("restricted_to_workflows") is True
                and isinstance(selected_workflows, list)
                and selected_workflows == [expected_workflow]
            )
            checks["workflow_scope"] = _check(
                PASS if workflow_scope_ok else FAIL,
                "runner-group-exact-main-workflow-scope"
                if workflow_scope_ok
                else "runner-group-workflow-scope-mismatch",
            )

            other_labels = {
                other.custom_label for other in RUNNER_POLICIES if other.key != policy.key
            }
            isolated = all(
                (labels := _label_names(runner)) is not None
                and not labels.intersection(other_labels)
                for runner in group_runners
            )
            checks["class_isolation"] = _check(
                PASS if isolated else FAIL,
                "runner-class-isolated" if isolated else "runner-classes-mixed",
            )
        else:
            checks["runner_labels"] = _check(FAIL, "runner-inventory-not-verifiable")
            checks["repository_scope"] = _check(FAIL, "runner-group-scope-not-verifiable")
            checks["workflow_scope"] = _check(FAIL, "runner-workflow-scope-not-verifiable")
            checks["class_isolation"] = _check(FAIL, "runner-class-isolation-not-verifiable")

        reports.append(
            {
                "class": policy.key,
                "status": _overall(checks),
                "registered_runner_count": len(matched_runners),
                "checks": checks,
            }
        )
    return reports


def run_preflight(api: GitHubReadApi, repository: str) -> dict[str, Any]:
    """Run the deterministic preflight without performing any external writes."""

    _split_repository(repository)
    repository_record: dict[str, Any] | None
    repository_metadata_check: dict[str, str] | None = None
    try:
        repository_record = api.get_object(f"/repos/{repository}")
    except ApiAccessError as exc:
        repository_record = None
        repository_metadata_check = _safe_api_failure(exc)

    branch = _main_ruleset_report(
        api,
        repository,
        repository_record,
        repository_metadata_check,
    )
    environments = [
        _environment_report(api, repository, policy)
        for policy in ENVIRONMENT_POLICIES
    ]
    runners = _runner_reports(api, repository, repository_record)

    section_statuses = [branch["status"]]
    section_statuses.extend(environment["status"] for environment in environments)
    section_statuses.extend(runner["status"] for runner in runners)
    failures = sum(status == FAIL for status in section_statuses)
    unknowns = sum(status == UNKNOWN for status in section_statuses)
    api_preflight_passed = failures == 0 and unknowns == 0
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "repository": repository,
        "api_preflight_passed": api_preflight_passed,
        # This read-only diagnostic never grants release authority.  The
        # authenticated qualification/publication chain remains authoritative.
        "release_authorization_granted": False,
        "summary": {
            "failed_sections": failures,
            "unknown_sections": unknowns,
            "verified_sections": sum(status == PASS for status in section_statuses),
        },
        "main_ruleset": branch,
        "environments": environments,
        "runner_groups": runners,
        "manual_boundaries": [
            {
                "id": "environment-administrator-bypass",
                "code": "verify-in-settings-or-authenticated-audit-log-when-rest-field-absent",
            },
            {
                "id": "ruleset-bypass-actors",
                "code": "github-withholds-field-without-ruleset-write-access",
            },
            {
                "id": "deployment-policy-target-type",
                "code": "list-response-may-not-distinguish-main-branch-from-main-tag",
            },
            {
                "id": "reviewer-human-independence",
                "code": "team-slug-does-not-prove-membership-independence-for-a-run",
            },
            {
                "id": "api-point-in-time-consistency",
                "code": "multiple-get-responses-are-not-an-atomic-github-snapshot",
            },
            {
                "id": "runner-host-isolation",
                "code": "rest-labels-do-not-prove-gpu-os-acl-mount-or-process-integrity",
            },
            {
                "id": "secret-values",
                "code": "rest-confirms-names-only-and-never-reveals-secret-values",
            },
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only, privacy-reduced preflight of ProAim GitHub release controls. "
            f"Reads its token only from {TOKEN_ENVIRONMENT_VARIABLE}."
        )
    )
    parser.add_argument(
        "--repository",
        required=True,
        help="Exact GitHub OWNER/REPOSITORY identity",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        token = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE, "")
        api = ReadOnlyGitHubApi(token, args.repository)
        report = run_preflight(api, args.repository)
    except PreflightError as exc:
        print(f"preflight error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    if report["api_preflight_passed"]:
        return 0
    if report["summary"]["failed_sections"]:
        return 1
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
