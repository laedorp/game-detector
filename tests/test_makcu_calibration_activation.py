from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from aiming.makcu_calibration_activation import (
    CalibrationActivationBindingError,
    CalibrationActivationConflictError,
    CalibrationActivationError,
    activate_session_evidence_file,
    active_profile_bytes,
    active_profile_from_bytes,
    active_profile_from_evidence,
    active_profile_matches_binding,
    load_active_profile,
    load_active_profile_for_binding,
    resolve_active_profile,
    write_active_profile_atomic,
)
from aiming.makcu_calibration_session import write_session_evidence_exclusive
from tests.test_makcu_calibration_session import SessionHarness


class MakcuCalibrationActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        successful = SessionHarness()
        successful.arm()
        successful.run()
        assert successful.session.result is not None
        assert successful.session.result.fit is not None
        cls.evidence = successful.session.result.evidence
        cls.binding = cls.evidence.binding
        cls.profile = active_profile_from_evidence(
            cls.evidence,
            expected_binding=cls.binding,
        )

        aborted = SessionHarness()
        aborted.session.abort("operator cancelled", now_ns=aborted.now_ns + 1)
        assert aborted.session.result is not None
        cls.aborted_evidence = aborted.session.result.evidence

    def test_profile_is_lean_canonical_and_retains_exact_provenance(self) -> None:
        profile = self.profile
        self.assertEqual(profile.binding, self.evidence.binding)
        self.assertEqual(profile.fit, self.evidence.fit)
        self.assertEqual(
            profile.session_artifact_sha256,
            self.evidence.artifact_sha256,
        )
        self.assertEqual(
            profile.core_evidence_sha256,
            self.evidence.core_evidence_sha256,
        )
        canonical = active_profile_bytes(profile)
        self.assertEqual(canonical, active_profile_bytes(profile))
        self.assertTrue(canonical.endswith(b"\n"))
        self.assertNotIn(b": ", canonical)
        self.assertEqual(active_profile_from_bytes(canonical), profile)

        document = json.loads(canonical)
        self.assertEqual(
            set(document),
            {
                "binding",
                "core_evidence_sha256",
                "fit",
                "profile_sha256",
                "schema_version",
                "session_artifact_sha256",
            },
        )
        self.assertEqual(
            set(document["binding"]),
            set(self.binding.__dataclass_fields__),
        )

    def test_parser_rejects_malformed_and_noncanonical_profiles(self) -> None:
        canonical = active_profile_bytes(self.profile)
        pretty = json.dumps(json.loads(canonical), indent=2).encode("utf-8")
        with self.assertRaisesRegex(CalibrationActivationError, "canonical"):
            active_profile_from_bytes(pretty)

        malformed_document = json.loads(canonical)
        malformed_document["binding"]["capture_width"] = True
        malformed = (
            json.dumps(malformed_document, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        with self.assertRaisesRegex(CalibrationActivationError, "binding"):
            active_profile_from_bytes(malformed)

        with self.assertRaisesRegex(CalibrationActivationError, "UTF-8 JSON"):
            active_profile_from_bytes(b"not-json\n")

    def test_parser_rejects_tampered_fit_and_hash(self) -> None:
        document = json.loads(active_profile_bytes(self.profile))
        document["fit"]["x"]["gain_pixels_per_count"] += 0.01
        tampered = (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self.assertRaisesRegex(CalibrationActivationError, "profile_sha256"):
            active_profile_from_bytes(tampered)

        document = json.loads(active_profile_bytes(self.profile))
        document["profile_sha256"] = "f" * 64
        tampered_hash = (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self.assertRaisesRegex(CalibrationActivationError, "profile_sha256"):
            active_profile_from_bytes(tampered_hash)

    def test_failed_or_stale_evidence_cannot_be_activated(self) -> None:
        with self.assertRaisesRegex(CalibrationActivationError, "only successful"):
            active_profile_from_evidence(self.aborted_evidence)

        tampered_evidence = replace(
            self.evidence,
            artifact_sha256="f" * 64,
        )
        with self.assertRaisesRegex(CalibrationActivationError, "strict validation"):
            active_profile_from_evidence(tampered_evidence)

        stale_binding = replace(self.binding, active_device="different-gfx1030")
        with self.assertRaisesRegex(
            CalibrationActivationBindingError,
            "exactly match",
        ):
            active_profile_from_evidence(
                self.evidence,
                expected_binding=stale_binding,
            )

    def test_exact_binding_load_and_resolve_fail_closed_when_stale(self) -> None:
        stale_binding = replace(self.binding, capture_fps=120.0)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "active.json"
            write_active_profile_atomic(destination, self.profile)
            self.assertEqual(
                load_active_profile_for_binding(destination, self.binding),
                self.profile,
            )
            self.assertEqual(
                resolve_active_profile(destination, self.binding),
                self.profile,
            )
            self.assertIsNone(resolve_active_profile(destination, stale_binding))
            self.assertIsNone(
                resolve_active_profile(Path(temporary) / "missing.json", self.binding)
            )
            self.assertTrue(
                active_profile_matches_binding(self.profile, self.binding)
            )
            self.assertFalse(
                active_profile_matches_binding(self.profile, stale_binding)
            )
            with self.assertRaises(CalibrationActivationBindingError):
                load_active_profile_for_binding(destination, stale_binding)

    def test_atomic_writer_is_private_and_never_blindly_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "profiles" / "active.json"
            write_active_profile_atomic(destination, self.profile)
            original = destination.read_bytes()
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            self.assertEqual(load_active_profile(destination), self.profile)

            with self.assertRaises(FileExistsError):
                write_active_profile_atomic(destination, self.profile)
            self.assertEqual(destination.read_bytes(), original)

            with self.assertRaises(CalibrationActivationConflictError):
                write_active_profile_atomic(
                    destination,
                    self.profile,
                    expected_previous_profile_sha256="e" * 64,
                )
            self.assertEqual(destination.read_bytes(), original)

            write_active_profile_atomic(
                destination,
                self.profile,
                expected_previous_profile_sha256=self.profile.profile_sha256,
            )
            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(
                [entry.name for entry in destination.parent.iterdir()],
                [destination.name],
            )

    def test_atomic_replace_failure_preserves_last_good_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "active.json"
            write_active_profile_atomic(destination, self.profile)
            previous = destination.read_bytes()
            with mock.patch(
                "aiming.makcu_calibration_activation.os.replace",
                side_effect=OSError("simulated activation rename failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated activation"):
                    write_active_profile_atomic(
                        destination,
                        self.profile,
                        expected_previous_profile_sha256=(
                            self.profile.profile_sha256
                        ),
                    )
            self.assertEqual(destination.read_bytes(), previous)
            self.assertEqual(load_active_profile(destination), self.profile)
            self.assertEqual(os.listdir(temporary), [destination.name])

    def test_corrupt_existing_profile_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "active.json"
            write_active_profile_atomic(destination, self.profile)
            corrupt = destination.read_bytes().replace(b'"schema_version":1', b'"schema_version":2')
            destination.write_bytes(corrupt)
            os.chmod(destination, 0o600)
            with self.assertRaises(CalibrationActivationError):
                write_active_profile_atomic(
                    destination,
                    self.profile,
                    expected_previous_profile_sha256=(
                        self.profile.profile_sha256
                    ),
                )
            self.assertEqual(destination.read_bytes(), corrupt)
            self.assertEqual(os.listdir(temporary), [destination.name])

    def test_explicit_file_activation_uses_strict_session_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path = Path(temporary) / "evidence.json"
            active_path = Path(temporary) / "active.json"
            write_session_evidence_exclusive(evidence_path, self.evidence)
            activated = activate_session_evidence_file(
                evidence_path,
                active_path,
                expected_binding=self.binding,
            )
            self.assertEqual(activated, self.profile)
            self.assertEqual(
                load_active_profile_for_binding(active_path, self.binding),
                self.profile,
            )


if __name__ == "__main__":
    unittest.main()
