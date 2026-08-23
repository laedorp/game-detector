from __future__ import annotations

import unittest

from utils.public_evidence import contains_nonportable_path


class PublicEvidencePathTests(unittest.TestCase):
    def test_rejects_absolute_paths_after_arbitrary_delimiters(self) -> None:
        for value in (
            "/tmp/private/model.onnx",
            "loaded_from@/tmp/private/model.onnx",
            "loaded_from§/tmp/private/model.onnx",
            r"loaded_from@C:\Users\private\model.onnx",
            r"loaded_from@\\server\private\model.onnx",
            r"models\private\model.onnx",
            "loaded_from@~/private/model.onnx",
            "loaded_from@file:///tmp/private/model.onnx",
        ):
            with self.subTest(value=value):
                self.assertTrue(contains_nonportable_path(value))

    def test_accepts_relative_names_and_public_urls(self) -> None:
        for value in (
            "models/release-default/model.onnx",
            "metrics/precision(B)",
            "https://example.invalid/releases/model.onnx",
            "https://example.invalid:443/releases/model.onnx#section",
        ):
            with self.subTest(value=value):
                self.assertFalse(contains_nonportable_path(value))

    def test_rejects_local_path_in_public_url_query(self) -> None:
        self.assertTrue(
            contains_nonportable_path(
                "https://example.invalid/report?loaded_from=/tmp/private/model"
            )
        )

    def test_requires_string(self) -> None:
        with self.assertRaises(TypeError):
            contains_nonportable_path(1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
