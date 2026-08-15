"""Explicit, fail-closed activation of a successful MAKCU calibration.

The calibration session evidence is intentionally much larger than the data
needed by the live controller.  This module turns one *strictly revalidated*
successful evidence artifact into a small active profile.  Activation is an
explicit caller action: importing this module, loading evidence, and resolving
a profile never writes a file or changes a controller.

An active profile retains the complete runtime binding, the exact accepted
numeric fit, and both provenance hashes.  Its own hash protects the compact
artifact from accidental or partial modification.  Persistence is private and
atomic.  A missing destination may be created; replacing an existing profile
requires the caller to name the exact hash it expects to replace.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping

from aiming.makcu_calibration import (
    AxisCalibrationFit,
    CalibrationDataError,
    CalibrationQualityError,
    MakcuCalibrationFit,
)
from aiming.makcu_calibration_session import (
    CalibrationEvidenceError,
    CalibrationRuntimeBinding,
    CalibrationSessionEvidence,
    load_session_evidence,
    session_evidence_bytes,
)


ACTIVE_PROFILE_SCHEMA_VERSION = 1
MAX_ACTIVE_PROFILE_BYTES = 128 * 1024
_HASH_RE = re.compile(r"[0-9a-f]{64}")


class CalibrationActivationError(ValueError):
    """An evidence artifact or active profile cannot be safely activated."""


class CalibrationActivationBindingError(CalibrationActivationError):
    """A calibration belongs to a different immutable runtime identity."""


class CalibrationActivationConflictError(CalibrationActivationError):
    """The active-profile destination changed from the caller's expectation."""


@dataclass(frozen=True, slots=True)
class ActiveMakcuCalibrationProfile:
    """Lean, immutable controller input derived from full session evidence."""

    binding: CalibrationRuntimeBinding
    fit: MakcuCalibrationFit
    session_artifact_sha256: str
    core_evidence_sha256: str
    profile_sha256: str
    schema_version: int = ACTIVE_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != ACTIVE_PROFILE_SCHEMA_VERSION
        ):
            raise CalibrationActivationError(
                "unsupported active calibration profile schema"
            )
        if not isinstance(self.binding, CalibrationRuntimeBinding):
            raise CalibrationActivationError("active profile binding is invalid")
        if not isinstance(self.fit, MakcuCalibrationFit):
            raise CalibrationActivationError("active profile fit is invalid")
        for name in (
            "session_artifact_sha256",
            "core_evidence_sha256",
            "profile_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
                raise CalibrationActivationError(f"{name} must be lowercase SHA-256")
        if self.core_evidence_sha256 != self.fit.evidence_sha256:
            raise CalibrationActivationError(
                "active profile core evidence hash does not match its fit"
            )


def _dataclass_dict(value: object) -> dict[str, object]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _binding_dict(binding: CalibrationRuntimeBinding) -> dict[str, object]:
    return _dataclass_dict(binding)


def _axis_fit_dict(fit: AxisCalibrationFit) -> dict[str, object]:
    return _dataclass_dict(fit)


def _fit_dict(fit: MakcuCalibrationFit) -> dict[str, object]:
    return {
        "delay_seconds": fit.delay_seconds,
        "detector_period_seconds": fit.detector_period_seconds,
        "evidence_sha256": fit.evidence_sha256,
        "observation_duty": fit.observation_duty,
        "x": _axis_fit_dict(fit.x),
        "y": _axis_fit_dict(fit.y),
    }


def _profile_document(
    profile: ActiveMakcuCalibrationProfile,
    *,
    include_hash: bool,
) -> dict[str, object]:
    document: dict[str, object] = {
        "binding": _binding_dict(profile.binding),
        "core_evidence_sha256": profile.core_evidence_sha256,
        "fit": _fit_dict(profile.fit),
        "schema_version": profile.schema_version,
        "session_artifact_sha256": profile.session_artifact_sha256,
    }
    if include_hash:
        document["profile_sha256"] = profile.profile_sha256
    return document


def _canonical_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationActivationError(
            "active profile contains noncanonical values"
        ) from exc
    return (text + "\n").encode("utf-8")


def _profile_digest(profile: ActiveMakcuCalibrationProfile) -> str:
    return sha256(
        _canonical_bytes(_profile_document(profile, include_hash=False))
    ).hexdigest()


def active_profile_from_evidence(
    evidence: CalibrationSessionEvidence,
    *,
    expected_binding: CalibrationRuntimeBinding | None = None,
) -> ActiveMakcuCalibrationProfile:
    """Derive a lean profile only after a full strict evidence revalidation.

    Supplying ``expected_binding`` is recommended at an activation boundary;
    equality is deliberately exact rather than compatibility-based.
    """

    try:
        session_evidence_bytes(evidence)
    except (CalibrationEvidenceError, CalibrationDataError, CalibrationQualityError) as exc:
        raise CalibrationActivationError(
            f"calibration session evidence did not pass strict validation: {exc}"
        ) from exc
    if evidence.outcome != "success":
        raise CalibrationActivationError(
            "only successful calibration session evidence can be activated"
        )
    if (
        not evidence.evidence_complete
        or evidence.cleanup_error is not None
        or evidence.fit is None
        or evidence.core_evidence_sha256 is None
    ):
        raise CalibrationActivationError(
            "successful calibration evidence is incomplete or unclean"
        )
    if expected_binding is not None:
        if not isinstance(expected_binding, CalibrationRuntimeBinding):
            raise TypeError("expected_binding must be CalibrationRuntimeBinding")
        if evidence.binding != expected_binding:
            raise CalibrationActivationBindingError(
                "calibration evidence does not exactly match the current runtime binding"
            )

    placeholder = ActiveMakcuCalibrationProfile(
        binding=evidence.binding,
        fit=evidence.fit,
        session_artifact_sha256=evidence.artifact_sha256,
        core_evidence_sha256=evidence.core_evidence_sha256,
        profile_sha256="0" * 64,
    )
    return ActiveMakcuCalibrationProfile(
        binding=placeholder.binding,
        fit=placeholder.fit,
        session_artifact_sha256=placeholder.session_artifact_sha256,
        core_evidence_sha256=placeholder.core_evidence_sha256,
        profile_sha256=_profile_digest(placeholder),
    )


def active_profile_bytes(profile: ActiveMakcuCalibrationProfile) -> bytes:
    """Return strict canonical bytes after verifying the compact artifact hash."""

    if not isinstance(profile, ActiveMakcuCalibrationProfile):
        raise CalibrationActivationError("active profile has the wrong type")
    expected = _profile_digest(profile)
    if profile.profile_sha256 != expected:
        raise CalibrationActivationError(
            "profile_sha256 does not match the active calibration profile"
        )
    return _canonical_bytes(_profile_document(profile, include_hash=True))


def _expect_mapping(
    value: object,
    expected_fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise CalibrationActivationError(f"{name} fields are not canonical")
    return value


def _binding_from_dict(value: object) -> CalibrationRuntimeBinding:
    data = _expect_mapping(
        value,
        {field.name for field in fields(CalibrationRuntimeBinding)},
        "active profile binding",
    )
    try:
        return CalibrationRuntimeBinding(**dict(data))
    except (TypeError, ValueError) as exc:
        raise CalibrationActivationError(f"active profile binding is invalid: {exc}") from exc


def _axis_fit_from_dict(value: object, expected_axis: str) -> AxisCalibrationFit:
    data = _expect_mapping(
        value,
        {field.name for field in fields(AxisCalibrationFit)},
        f"active profile {expected_axis}-axis fit",
    )
    try:
        result = AxisCalibrationFit(**dict(data))
    except (TypeError, ValueError, CalibrationDataError) as exc:
        raise CalibrationActivationError(
            f"active profile {expected_axis}-axis fit is invalid: {exc}"
        ) from exc
    if result.axis != expected_axis:
        raise CalibrationActivationError("active profile fit axes are not canonical")
    return result


def _fit_from_dict(value: object) -> MakcuCalibrationFit:
    expected = {
        "delay_seconds",
        "detector_period_seconds",
        "evidence_sha256",
        "observation_duty",
        "x",
        "y",
    }
    data = _expect_mapping(value, expected, "active profile fit")
    try:
        return MakcuCalibrationFit(
            x=_axis_fit_from_dict(data["x"], "x"),
            y=_axis_fit_from_dict(data["y"], "y"),
            delay_seconds=data["delay_seconds"],
            detector_period_seconds=data["detector_period_seconds"],
            observation_duty=data["observation_duty"],
            evidence_sha256=data["evidence_sha256"],
        )
    except (TypeError, ValueError, CalibrationDataError) as exc:
        raise CalibrationActivationError(f"active profile fit is invalid: {exc}") from exc


def active_profile_from_bytes(payload: bytes) -> ActiveMakcuCalibrationProfile:
    """Parse only exact canonical UTF-8 active-profile JSON."""

    if not isinstance(payload, bytes):
        raise CalibrationActivationError("active profile payload must be bytes")
    if len(payload) > MAX_ACTIVE_PROFILE_BYTES:
        raise CalibrationActivationError("active profile payload is unexpectedly large")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationActivationError(
            "active profile is not valid UTF-8 JSON"
        ) from exc
    data = _expect_mapping(
        document,
        {
            "binding",
            "core_evidence_sha256",
            "fit",
            "profile_sha256",
            "schema_version",
            "session_artifact_sha256",
        },
        "active profile",
    )
    schema_version = data["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise CalibrationActivationError("active profile schema_version must be an integer")
    try:
        profile = ActiveMakcuCalibrationProfile(
            binding=_binding_from_dict(data["binding"]),
            fit=_fit_from_dict(data["fit"]),
            session_artifact_sha256=data["session_artifact_sha256"],
            core_evidence_sha256=data["core_evidence_sha256"],
            profile_sha256=data["profile_sha256"],
            schema_version=schema_version,
        )
    except (TypeError, ValueError, CalibrationDataError) as exc:
        if isinstance(exc, CalibrationActivationError):
            raise
        raise CalibrationActivationError(f"active profile is invalid: {exc}") from exc
    canonical = active_profile_bytes(profile)
    if canonical != payload:
        raise CalibrationActivationError("active profile JSON is not canonical")
    return profile


def _read_private_regular_file(path: Path) -> bytes:
    path_metadata = path.lstat()
    if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISREG(
        path_metadata.st_mode
    ):
        raise CalibrationActivationError("active profile path is not a regular file")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CalibrationActivationError("active profile path is not a regular file")
        if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise PermissionError("active calibration profile must have mode 0600")
        if metadata.st_size > MAX_ACTIVE_PROFILE_BYTES:
            raise CalibrationActivationError(
                "active profile file is unexpectedly large"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            payload = stream.read(MAX_ACTIVE_PROFILE_BYTES + 1)
        if len(payload) > MAX_ACTIVE_PROFILE_BYTES:
            raise CalibrationActivationError(
                "active profile file is unexpectedly large"
            )
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_active_profile(path: str | Path) -> ActiveMakcuCalibrationProfile:
    """Load one canonical, private active profile without resolving identity."""

    return active_profile_from_bytes(_read_private_regular_file(Path(path)))


def active_profile_matches_binding(
    profile: ActiveMakcuCalibrationProfile,
    binding: CalibrationRuntimeBinding,
) -> bool:
    """Return true only for a valid profile with exact binding equality."""

    if not isinstance(binding, CalibrationRuntimeBinding):
        return False
    try:
        active_profile_bytes(profile)
    except (CalibrationActivationError, CalibrationDataError):
        return False
    return profile.binding == binding


def load_active_profile_for_binding(
    path: str | Path,
    binding: CalibrationRuntimeBinding,
) -> ActiveMakcuCalibrationProfile:
    """Strictly load a profile and raise if any binding field is stale."""

    if not isinstance(binding, CalibrationRuntimeBinding):
        raise TypeError("binding must be CalibrationRuntimeBinding")
    profile = load_active_profile(path)
    if profile.binding != binding:
        raise CalibrationActivationBindingError(
            "active calibration profile does not exactly match the runtime binding"
        )
    return profile


def resolve_active_profile(
    path: str | Path,
    binding: CalibrationRuntimeBinding,
) -> ActiveMakcuCalibrationProfile | None:
    """Resolve a valid exact-match profile, or ``None`` if absent/stale.

    Malformed, tampered, or insecure files raise instead of being treated as an
    ordinary cache miss, so callers can surface corruption explicitly.
    """

    if not isinstance(binding, CalibrationRuntimeBinding):
        raise TypeError("binding must be CalibrationRuntimeBinding")
    try:
        profile = load_active_profile(path)
    except FileNotFoundError:
        return None
    return profile if profile.binding == binding else None


def _prepare_private_parent(destination: Path) -> None:
    parent_preexisted = os.path.lexists(destination.parent)
    if not parent_preexisted:
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=False)
    if not parent_preexisted and os.name == "posix":
        os.chmod(destination.parent, 0o700)
    metadata = destination.parent.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(str(destination.parent))
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PermissionError(
            "active calibration profile directory must not be group/world accessible"
        )


def _fsync_directory_best_effort(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # The atomic link/replace is already the commit point.  Directory
        # descriptors are unsupported on some platforms.
        pass


def write_active_profile_atomic(
    path: str | Path,
    profile: ActiveMakcuCalibrationProfile,
    *,
    expected_previous_profile_sha256: str | None = None,
) -> None:
    """Privately publish an active profile without a blind overwrite.

    With no expected hash the destination must not exist.  To replace an
    existing profile, the caller must provide its exact ``profile_sha256``;
    malformed or changed existing state is never overwritten.
    """

    destination = Path(path)
    payload = active_profile_bytes(profile)
    if expected_previous_profile_sha256 is not None and (
        not isinstance(expected_previous_profile_sha256, str)
        or _HASH_RE.fullmatch(expected_previous_profile_sha256) is None
    ):
        raise CalibrationActivationError(
            "expected_previous_profile_sha256 must be lowercase SHA-256"
        )
    _prepare_private_parent(destination)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != payload:
            raise OSError("temporary active-profile verification failed")
        if active_profile_from_bytes(temporary.read_bytes()) != profile:
            raise OSError("temporary active profile did not validate")

        destination_exists = os.path.lexists(destination)
        if expected_previous_profile_sha256 is None:
            if destination_exists:
                raise FileExistsError(str(destination))
            os.link(temporary, destination)
        else:
            if not destination_exists:
                raise CalibrationActivationConflictError(
                    "expected active profile is missing"
                )
            current = load_active_profile(destination)
            if current.profile_sha256 != expected_previous_profile_sha256:
                raise CalibrationActivationConflictError(
                    "active profile changed from the expected previous hash"
                )
            os.replace(temporary, destination)
        _fsync_directory_best_effort(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def activate_session_evidence_file(
    evidence_path: str | Path,
    active_profile_path: str | Path,
    *,
    expected_binding: CalibrationRuntimeBinding,
    expected_previous_profile_sha256: str | None = None,
) -> ActiveMakcuCalibrationProfile:
    """Explicitly validate evidence, derive a profile, and atomically publish it."""

    if not isinstance(expected_binding, CalibrationRuntimeBinding):
        raise TypeError("expected_binding must be CalibrationRuntimeBinding")
    try:
        evidence = load_session_evidence(evidence_path)
    except (CalibrationEvidenceError, CalibrationDataError, CalibrationQualityError) as exc:
        raise CalibrationActivationError(
            f"calibration session evidence did not pass strict loading: {exc}"
        ) from exc
    profile = active_profile_from_evidence(
        evidence,
        expected_binding=expected_binding,
    )
    write_active_profile_atomic(
        active_profile_path,
        profile,
        expected_previous_profile_sha256=expected_previous_profile_sha256,
    )
    return profile
