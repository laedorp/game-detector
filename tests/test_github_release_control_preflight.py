from __future__ import annotations

from copy import deepcopy
from email.message import Message
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from scripts.preflight_github_release_controls import (
    ENVIRONMENT_POLICIES,
    RUNNER_POLICIES,
    ApiAccessError,
    PreflightError,
    ReadOnlyGitHubApi,
    TOKEN_ENVIRONMENT_VARIABLE,
    main,
    run_preflight,
)


REPOSITORY = "owner/proaim"
SENSITIVE_REVIEWER_LOGIN = "private-reviewer-login"
SENSITIVE_RUNNER_NAME = "PRIVATE-WINDOWS-HOST-01"
SENSITIVE_GROUP_NAME = "PRIVATE-RUNNER-GROUP"
SENSITIVE_ERROR_BODY = "PRIVATE-UPSTREAM-ERROR-BODY"


class FakeApi:
    def __init__(self, fixture: dict[str, dict[str, object]]) -> None:
        self.fixture = fixture
        self.calls: list[tuple[str, str, str | None]] = []

    def _value(self, family: str, path: str) -> object:
        try:
            value = self.fixture[family][path]
        except KeyError as exc:
            raise AssertionError(f"missing fake response for {family} {path}") from exc
        if isinstance(value, BaseException):
            raise value
        return deepcopy(value)

    def get_object(self, path: str) -> dict[str, object]:
        self.calls.append(("GET", path, None))
        value = self._value("object", path)
        if not isinstance(value, dict):
            raise AssertionError(f"fake object response is not an object: {path}")
        return value

    def get_list(self, path: str) -> list[dict[str, object]]:
        self.calls.append(("GET", path, None))
        value = self._value("list", path)
        if not isinstance(value, list):
            raise AssertionError(f"fake list response is not a list: {path}")
        return value

    def get_paginated_object(self, path: str, key: str) -> list[dict[str, object]]:
        self.calls.append(("GET", path, key))
        value = self._value("paginated_object", path)
        if not isinstance(value, list):
            raise AssertionError(f"fake paginated response is not a list: {path}")
        return value

    def get_paginated_list(self, path: str) -> list[dict[str, object]]:
        self.calls.append(("GET", path, None))
        value = self._value("paginated_list", path)
        if not isinstance(value, list):
            raise AssertionError(f"fake paginated list is not a list: {path}")
        return value


def _labels(*names: str) -> list[dict[str, object]]:
    return [
        {"id": index + 1, "name": name, "type": "custom"}
        for index, name in enumerate(names)
    ]


def _valid_fixture() -> dict[str, dict[str, object]]:
    fixture: dict[str, dict[str, object]] = {
        "object": {},
        "list": {},
        "paginated_object": {},
        "paginated_list": {},
    }
    repository_record = {
        "id": 1234,
        "full_name": REPOSITORY,
        "default_branch": "main",
        "private": True,
        "archived": False,
        "disabled": False,
        "owner": {"login": "owner", "type": "Organization"},
    }
    fixture["object"][f"/repos/{REPOSITORY}"] = repository_record
    fixture["paginated_list"][f"/repos/{REPOSITORY}/rules/branches/main"] = [
        {"type": "deletion", "ruleset_id": 9},
        {"type": "non_fast_forward", "ruleset_id": 9},
        {
            "type": "pull_request",
            "ruleset_id": 9,
            "parameters": {
                "required_approving_review_count": 1,
                "dismiss_stale_reviews_on_push": True,
                "require_last_push_approval": True,
                "required_review_thread_resolution": True,
                "require_code_owner_review": False,
            },
        },
        {
            "type": "required_status_checks",
            "ruleset_id": 9,
            "parameters": {
                "strict_required_status_checks_policy": True,
                "required_status_checks": [
                    {"context": "Test ProAim / test (ubuntu-24.04)"},
                    {"context": "Test ProAim / test (windows-2022)"},
                ],
            },
        },
    ]
    fixture["object"][f"/repos/{REPOSITORY}/rulesets/9?includes_parents=true"] = {
        "id": 9,
        "target": "branch",
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {
            "ref_name": {"include": ["refs/heads/main"], "exclude": []}
        },
        "rules": [],
    }

    for index, policy in enumerate(ENVIRONMENT_POLICIES, start=1):
        prefix = f"/repos/{REPOSITORY}/environments/{policy.name}"
        fixture["object"][prefix] = {
            "id": 100 + index,
            "name": policy.name,
            "can_admins_bypass": False,
            "protection_rules": [
                {
                    "id": 200 + index,
                    "type": "required_reviewers",
                    "prevent_self_review": True,
                    "reviewers": [
                        {
                            "type": "Team",
                            "reviewer": {
                                "id": 300 + index,
                                "slug": policy.reviewer_team_slug,
                                "name": "Private reviewer display name",
                                "login": SENSITIVE_REVIEWER_LOGIN,
                            },
                        }
                    ],
                },
                {"id": 400 + index, "type": "branch_policy"},
            ],
            "deployment_branch_policy": {
                "protected_branches": False,
                "custom_branch_policies": True,
            },
        }
        fixture["paginated_object"][prefix + "/deployment-branch-policies"] = [
            {"id": 500 + index, "name": "main", "type": "branch"}
        ]
        fixture["paginated_object"][prefix + "/variables"] = [
            {"name": name, "value": value} for name, value in policy.variables
        ]
        fixture["paginated_object"][prefix + "/secrets"] = [
            {"name": name, "updated_at": "2026-08-14T00:00:00Z"}
            for name in policy.secrets
        ]

    group_list_path = (
        "/orgs/owner/actions/runner-groups?visible_to_repository=proaim"
    )
    fixture["paginated_object"][group_list_path] = [
        {"id": index + 1, "name": SENSITIVE_GROUP_NAME + str(index + 1)}
        for index in range(len(RUNNER_POLICIES))
    ]
    for index, policy in enumerate(RUNNER_POLICIES, start=1):
        prefix = f"/orgs/owner/actions/runner-groups/{index}"
        fixture["object"][prefix] = {
            "id": index,
            "name": SENSITIVE_GROUP_NAME + str(index),
            "visibility": "selected",
            "allows_public_repositories": False,
            "restricted_to_workflows": True,
            "selected_workflows": [
                f"{REPOSITORY}/{policy.workflow_path}@refs/heads/main"
            ],
        }
        fixture["paginated_object"][prefix + "/repositories"] = [
            {"id": 1234, "full_name": REPOSITORY}
        ]
        fixture["paginated_object"][prefix + "/runners"] = [
            {
                "id": index * 10 + runner_index,
                "name": f"{SENSITIVE_RUNNER_NAME}-{index}-{runner_index}",
                "os": "windows",
                "status": "offline",
                "labels": _labels(*policy.required_labels),
            }
            for runner_index in range(1, policy.minimum_runners + 1)
        ]
    return fixture


def _environment_prefix(name: str) -> str:
    return f"/repos/{REPOSITORY}/environments/{name}"


def _environment_result(report: dict[str, object], name: str) -> dict[str, object]:
    environments = report["environments"]
    assert isinstance(environments, list)
    return next(value for value in environments if value["name"] == name)


def _runner_result(report: dict[str, object], key: str) -> dict[str, object]:
    runners = report["runner_groups"]
    assert isinstance(runners, list)
    return next(value for value in runners if value["class"] == key)


class GitHubReleaseControlPreflightTests(unittest.TestCase):
    def test_fully_verified_fixture_passes_without_disclosing_private_fields(self) -> None:
        api = FakeApi(_valid_fixture())
        report = run_preflight(api, REPOSITORY)

        self.assertTrue(report["api_preflight_passed"])
        self.assertFalse(report["release_authorization_granted"])
        self.assertEqual(report["summary"], {
            "failed_sections": 0,
            "unknown_sections": 0,
            "verified_sections": 11,
        })
        serialized = json.dumps(report, sort_keys=True)
        for forbidden in (
            SENSITIVE_REVIEWER_LOGIN,
            SENSITIVE_RUNNER_NAME,
            SENSITIVE_GROUP_NAME,
            "Private reviewer display name",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertTrue(api.calls)
        self.assertEqual({method for method, _, _ in api.calls}, {"GET"})

    def test_report_is_deterministic(self) -> None:
        first = run_preflight(FakeApi(_valid_fixture()), REPOSITORY)
        second = run_preflight(FakeApi(_valid_fixture()), REPOSITORY)
        self.assertEqual(first, second)

    def test_ruleset_bypass_is_fail_closed_when_github_withholds_field(self) -> None:
        fixture = _valid_fixture()
        ruleset = fixture["object"][
            f"/repos/{REPOSITORY}/rulesets/9?includes_parents=true"
        ]
        assert isinstance(ruleset, dict)
        ruleset.pop("bypass_actors")

        report = run_preflight(FakeApi(fixture), REPOSITORY)
        main_ruleset = report["main_ruleset"]
        self.assertEqual(main_ruleset["status"], "unknown")
        self.assertEqual(
            main_ruleset["checks"]["ruleset_bypass"],
            {
                "status": "unknown",
                "code": "ruleset-bypass-actors-withheld-by-read-api",
            },
        )
        self.assertFalse(report["api_preflight_passed"])

    def test_environment_admin_bypass_is_fail_closed_when_not_exposed(self) -> None:
        fixture = _valid_fixture()
        policy = ENVIRONMENT_POLICIES[0]
        environment = fixture["object"][_environment_prefix(policy.name)]
        assert isinstance(environment, dict)
        environment.pop("can_admins_bypass")

        report = run_preflight(FakeApi(fixture), REPOSITORY)
        result = _environment_result(report, policy.name)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(
            result["checks"]["administrator_bypass"]["code"],
            "environment-admin-bypass-not-exposed-by-read-api",
        )

    def test_environment_main_policy_target_must_be_an_exposed_branch(self) -> None:
        policy = ENVIRONMENT_POLICIES[0]
        path = _environment_prefix(policy.name) + "/deployment-branch-policies"
        for target, expected_status in ((None, "unknown"), ("tag", "fail")):
            with self.subTest(target=target):
                fixture = _valid_fixture()
                branch_policy = fixture["paginated_object"][path][0]
                if target is None:
                    branch_policy.pop("type")
                else:
                    branch_policy["type"] = target
                report = run_preflight(FakeApi(fixture), REPOSITORY)
                result = _environment_result(report, policy.name)
                self.assertEqual(result["status"], expected_status)

    def test_environment_controls_require_exact_inventory(self) -> None:
        policy = ENVIRONMENT_POLICIES[0]
        prefix = _environment_prefix(policy.name)
        mutations = {
            "wrong reviewer": lambda fixture: fixture["object"][prefix][
                "protection_rules"
            ][0]["reviewers"][0]["reviewer"].__setitem__("slug", "wrong-team"),
            "self review": lambda fixture: fixture["object"][prefix][
                "protection_rules"
            ][0].__setitem__("prevent_self_review", False),
            "extra deployment branch": lambda fixture: fixture["paginated_object"][
                prefix + "/deployment-branch-policies"
            ].append({"id": 999, "name": "release/*"}),
            "wrong variable value": lambda fixture: fixture["paginated_object"][
                prefix + "/variables"
            ][0].__setitem__("value", "PRIVATE-WRONG-VALUE"),
            "extra variable": lambda fixture: fixture["paginated_object"][
                prefix + "/variables"
            ].append({"name": "UNEXPECTED", "value": "PRIVATE-EXTRA-VALUE"}),
            "extra secret": lambda fixture: fixture["paginated_object"][
                prefix + "/secrets"
            ].append({"name": "UNEXPECTED_PRIVATE_SECRET"}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                fixture = _valid_fixture()
                mutate(fixture)
                report = run_preflight(FakeApi(fixture), REPOSITORY)
                result = _environment_result(report, policy.name)
                self.assertEqual(result["status"], "fail")
                serialized = json.dumps(report, sort_keys=True)
                self.assertNotIn("PRIVATE-WRONG-VALUE", serialized)
                self.assertNotIn("PRIVATE-EXTRA-VALUE", serialized)
                self.assertNotIn("UNEXPECTED_PRIVATE_SECRET", serialized)

    def test_ruleset_policy_rejects_weak_reviews_checks_force_push_and_bypass(self) -> None:
        rules_path = f"/repos/{REPOSITORY}/rules/branches/main"
        detail_path = f"/repos/{REPOSITORY}/rulesets/9?includes_parents=true"

        def weak_review(fixture: dict[str, dict[str, object]]) -> None:
            rules = fixture["paginated_list"][rules_path]
            assert isinstance(rules, list)
            pull_request = next(rule for rule in rules if rule["type"] == "pull_request")
            pull_request["parameters"]["require_last_push_approval"] = False

        def no_force_push_rule(fixture: dict[str, dict[str, object]]) -> None:
            rules = fixture["paginated_list"][rules_path]
            assert isinstance(rules, list)
            rules[:] = [rule for rule in rules if rule["type"] != "non_fast_forward"]

        def weak_checks(fixture: dict[str, dict[str, object]]) -> None:
            rules = fixture["paginated_list"][rules_path]
            assert isinstance(rules, list)
            status = next(rule for rule in rules if rule["type"] == "required_status_checks")
            status["parameters"]["strict_required_status_checks_policy"] = False

        def bypass(fixture: dict[str, dict[str, object]]) -> None:
            detail = fixture["object"][detail_path]
            assert isinstance(detail, dict)
            detail["bypass_actors"] = [
                {"actor_type": "OrganizationAdmin", "bypass_mode": "always"}
            ]

        for label, mutate in (
            ("weak review", weak_review),
            ("force push", no_force_push_rule),
            ("weak checks", weak_checks),
            ("bypass", bypass),
        ):
            with self.subTest(label=label):
                fixture = _valid_fixture()
                mutate(fixture)
                report = run_preflight(FakeApi(fixture), REPOSITORY)
                self.assertEqual(report["main_ruleset"]["status"], "fail")
                self.assertFalse(report["api_preflight_passed"])

    def test_runner_group_requires_exact_repo_workflow_labels_and_class_isolation(self) -> None:
        policy = RUNNER_POLICIES[0]
        prefix = "/orgs/owner/actions/runner-groups/1"

        def extra_workflow(fixture: dict[str, dict[str, object]]) -> None:
            detail = fixture["object"][prefix]
            assert isinstance(detail, dict)
            detail["selected_workflows"].append(
                f"{REPOSITORY}/.github/workflows/ci.yml@refs/heads/main"
            )

        def extra_repository(fixture: dict[str, dict[str, object]]) -> None:
            repositories = fixture["paginated_object"][prefix + "/repositories"]
            assert isinstance(repositories, list)
            repositories.append({"id": 9876, "full_name": "owner/other"})

        def missing_label(fixture: dict[str, dict[str, object]]) -> None:
            runners = fixture["paginated_object"][prefix + "/runners"]
            assert isinstance(runners, list)
            runners[0]["labels"] = _labels("self-hosted", "Windows", policy.custom_label)

        def mixed_class(fixture: dict[str, dict[str, object]]) -> None:
            runners = fixture["paginated_object"][prefix + "/runners"]
            assert isinstance(runners, list)
            runners[0]["labels"].append(
                {"id": 999, "name": RUNNER_POLICIES[1].custom_label, "type": "custom"}
            )

        for label, mutate in (
            ("extra workflow", extra_workflow),
            ("extra repository", extra_repository),
            ("missing label", missing_label),
            ("mixed class", mixed_class),
        ):
            with self.subTest(label=label):
                fixture = _valid_fixture()
                mutate(fixture)
                report = run_preflight(FakeApi(fixture), REPOSITORY)
                result = _runner_result(report, policy.key)
                self.assertEqual(result["status"], "fail")
                self.assertNotIn(SENSITIVE_RUNNER_NAME, json.dumps(report))

    def test_runner_group_permission_failure_is_unknown_not_pass(self) -> None:
        fixture = _valid_fixture()
        path = "/orgs/owner/actions/runner-groups?visible_to_repository=proaim"
        fixture["paginated_object"][path] = ApiAccessError(403, path, "http-403")
        report = run_preflight(FakeApi(fixture), REPOSITORY)
        for result in report["runner_groups"]:
            self.assertEqual(result["status"], "unknown")
            self.assertEqual(
                result["checks"]["runner_group_api"]["code"],
                "runner-group-api-unavailable-by-plan-or-permission",
            )
        self.assertFalse(report["api_preflight_passed"])

    def test_missing_repository_is_a_failure_but_denied_metadata_is_unknown(self) -> None:
        path = f"/repos/{REPOSITORY}"
        for status, expected in ((404, "fail"), (403, "unknown")):
            with self.subTest(status=status):
                fixture = _valid_fixture()
                fixture["object"][path] = ApiAccessError(
                    status, path, f"http-{status}"
                )
                report = run_preflight(FakeApi(fixture), REPOSITORY)
                self.assertEqual(report["main_ruleset"]["status"], expected)

    def test_unrelated_default_runner_group_does_not_require_repository_endpoint(self) -> None:
        fixture = _valid_fixture()
        group_list = fixture["paginated_object"][
            "/orgs/owner/actions/runner-groups?visible_to_repository=proaim"
        ]
        group_list.append({"id": 99, "name": "Default"})
        fixture["object"]["/orgs/owner/actions/runner-groups/99"] = {
            "id": 99,
            "name": "Default",
            "visibility": "all",
            "allows_public_repositories": False,
            "restricted_to_workflows": False,
            "selected_workflows": [],
        }
        fixture["paginated_object"][
            "/orgs/owner/actions/runner-groups/99/runners"
        ] = [
            {
                "id": 999,
                "name": "unrelated-runner",
                "os": "linux",
                "labels": _labels("self-hosted", "Linux", "X64"),
            }
        ]
        report = run_preflight(FakeApi(fixture), REPOSITORY)
        self.assertTrue(report["api_preflight_passed"])

    def test_personal_repository_cannot_satisfy_runner_group_boundary(self) -> None:
        fixture = _valid_fixture()
        repository = fixture["object"][f"/repos/{REPOSITORY}"]
        assert isinstance(repository, dict)
        repository["owner"] = {"login": "owner", "type": "User"}
        report = run_preflight(FakeApi(fixture), REPOSITORY)
        for result in report["runner_groups"]:
            self.assertEqual(result["status"], "fail")
            self.assertEqual(
                result["checks"]["runner_group_api"]["code"],
                "organization-runner-group-required",
            )

    def test_policy_identities_are_unique_and_holdout_labels_are_exactly_pinned(self) -> None:
        self.assertEqual(
            len({policy.name for policy in ENVIRONMENT_POLICIES}),
            len(ENVIRONMENT_POLICIES),
        )
        self.assertEqual(
            len({policy.reviewer_team_slug for policy in ENVIRONMENT_POLICIES}),
            len(ENVIRONMENT_POLICIES),
        )
        holdout = next(policy for policy in RUNNER_POLICIES if policy.key == "independent-holdout")
        self.assertEqual(
            holdout.required_labels,
            (
                "self-hosted",
                "Windows",
                "X64",
                "proaim-independent-holdout-directml",
                "proaim-rx-6950-xt-holdout",
            ),
        )

    def test_policy_variable_secret_and_runner_contracts_match_workflow_sources(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workflow_sources = {
            path.name: path.read_text(encoding="utf-8")
            for path in (project_root / ".github" / "workflows").glob("*.yml")
        }
        environment_workflows = {
            "directml-amd_rx_6950_xt-physical-attestation": "qualify-windows-directml.yml",
            "directml-nvidia_rtx_5060_laptop-physical-attestation": "qualify-windows-directml.yml",
            "directml-release-publication": "publish-qualified-directml-release.yml",
            "independent-holdout-access": "qualify-independent-holdout.yml",
            "independent-holdout-attestation": "qualify-independent-holdout.yml",
            "cuda-physical-attestation": "qualify-windows-cuda.yml",
            "cuda-release-publication": "attach-qualified-cuda.yml",
        }
        for policy in ENVIRONMENT_POLICIES:
            source = workflow_sources[environment_workflows[policy.name]]
            with self.subTest(environment=policy.name):
                for variable_name, _ in policy.variables:
                    self.assertIn(variable_name, source)
                for secret_name in policy.secrets:
                    self.assertIn(secret_name, source)
        for policy in RUNNER_POLICIES:
            workflow_name = Path(policy.workflow_path).name
            source = workflow_sources[workflow_name]
            with self.subTest(runner=policy.key):
                for label in policy.required_labels:
                    self.assertIn(label, source)


class _Response:
    status = 200

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.payload[:size] if size >= 0 else self.payload


class _Opener:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[object] = []

    def open(self, request: object, timeout: int) -> object:
        self.requests.append(request)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class ReadOnlyGitHubApiTests(unittest.TestCase):
    def test_http_transport_uses_get_and_fixed_github_origin(self) -> None:
        opener = _Opener(_Response(b'{"default_branch":"main"}'))
        with patch(
            "scripts.preflight_github_release_controls.build_opener",
            return_value=opener,
        ):
            api = ReadOnlyGitHubApi("PRIVATE-TOKEN", REPOSITORY)
            self.assertEqual(api.get_object(f"/repos/{REPOSITORY}"), {"default_branch": "main"})
        request = opener.requests[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, f"https://api.github.com/repos/{REPOSITORY}")
        self.assertEqual(request.get_header("Authorization"), "Bearer PRIVATE-TOKEN")

    def test_http_error_does_not_render_response_body_or_token(self) -> None:
        error = HTTPError(
            url=f"https://api.github.com/repos/{REPOSITORY}",
            code=403,
            msg="Forbidden",
            hdrs=Message(),
            fp=io.BytesIO(SENSITIVE_ERROR_BODY.encode()),
        )
        opener = _Opener(error)
        with patch(
            "scripts.preflight_github_release_controls.build_opener",
            return_value=opener,
        ):
            api = ReadOnlyGitHubApi("PRIVATE-TOKEN", REPOSITORY)
            with self.assertRaises(ApiAccessError) as caught:
                api.get_object(f"/repos/{REPOSITORY}")
        rendered = str(caught.exception)
        self.assertNotIn(SENSITIVE_ERROR_BODY, rendered)
        self.assertNotIn("PRIVATE-TOKEN", rendered)
        self.assertEqual(caught.exception.status, 403)

    def test_transport_rejects_duplicate_keys_and_non_finite_json(self) -> None:
        for payload in (b'{"id":1,"id":2}', b'{"id":NaN}'):
            with self.subTest(payload=payload):
                opener = _Opener(_Response(payload))
                with patch(
                    "scripts.preflight_github_release_controls.build_opener",
                    return_value=opener,
                ):
                    api = ReadOnlyGitHubApi("PRIVATE-TOKEN", REPOSITORY)
                    with self.assertRaises(PreflightError):
                        api.get_object(f"/repos/{REPOSITORY}")

    def test_token_control_characters_are_rejected_before_transport(self) -> None:
        with self.assertRaises(PreflightError):
            ReadOnlyGitHubApi("PRIVATE\nTOKEN", REPOSITORY)

    def test_client_rejects_cross_owner_absolute_and_traversal_paths(self) -> None:
        api = ReadOnlyGitHubApi("PRIVATE-TOKEN", REPOSITORY)
        for path in (
            "https://example.invalid/repos/owner/proaim",
            "/repos/other/repository",
            "/orgs/other/actions/runner-groups",
            f"/repos/{REPOSITORY}/../other",
        ):
            with self.subTest(path=path):
                with self.assertRaises(PreflightError):
                    api.get_object(path)

    def test_cli_does_not_fall_back_to_general_github_credentials(self) -> None:
        stderr = io.StringIO()
        with patch.dict(
            "os.environ",
            {"GH_TOKEN": "DO-NOT-USE", "GITHUB_TOKEN": "DO-NOT-USE"},
            clear=True,
        ), patch("sys.stderr", stderr):
            result = main(["--repository", REPOSITORY])
        self.assertEqual(result, 2)
        self.assertIn(TOKEN_ENVIRONMENT_VARIABLE, stderr.getvalue())
        self.assertNotIn("DO-NOT-USE", stderr.getvalue())

    def test_repository_identity_is_strict(self) -> None:
        for repository in (
            "owner",
            "owner/repo/extra",
            "../repo",
            "owner/..",
            "https://github.com/owner/repo",
        ):
            with self.subTest(repository=repository):
                with self.assertRaises(PreflightError):
                    ReadOnlyGitHubApi("PRIVATE-TOKEN", repository)

    def test_source_contains_no_mutating_http_method(self) -> None:
        source = Path("scripts/preflight_github_release_controls.py").read_text(
            encoding="utf-8"
        )
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            self.assertNotIn(f'method="{method}"', source)
            self.assertNotIn(f"method='{method}'", source)


if __name__ == "__main__":
    unittest.main()
