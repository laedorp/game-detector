from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.validate_linux_bundle import _version, validate_bundle


class LinuxBundleValidationTests(unittest.TestCase):
    def test_rejects_missing_bundle_and_invalid_version(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_bundle(Path("/definitely/missing"))
        with self.assertRaises(Exception):
            _version("2")

    def test_detects_glibc_floor_and_unresolved_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "app"
            binary.write_bytes(b"\x7fELF data GLIBC_2.35 GLIBC_2.17")
            result = mock.Mock(
                returncode=0,
                stdout="libmissing.so => not found\n",
                stderr="",
            )
            with mock.patch("scripts.validate_linux_bundle.subprocess.run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "libmissing.so"):
                    validate_bundle(root)
                count, highest = validate_bundle(
                    root, allowed_missing=frozenset({"libmissing.so"})
                )
            self.assertEqual(count, 1)
            self.assertEqual(highest, (2, 35))
            with mock.patch("scripts.validate_linux_bundle.subprocess.run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "above allowed"):
                    validate_bundle(
                        root,
                        max_glibc=(2, 34),
                        allowed_missing=frozenset({"libmissing.so"}),
                    )


if __name__ == "__main__":
    unittest.main()
