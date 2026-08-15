from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import parse_args
from detection.base import ModelError
from detection.openvino_yolo import OpenVINOYoloDetector, _normalize_inference_size
from detection.postprocess import decode_yolo_output
from launcher.settings import (
    LauncherSettings,
    MODEL_PRESET_CUSTOM,
    SettingsError,
    save_settings,
)
from scripts.benchmark_models import build_parser as build_benchmark_parser
from scripts.export_model import build_parser as build_export_parser
from tests.test_onnx_backend import make_detector
from utils.inference_size import (
    compact_inference_size,
    format_inference_size,
    normalize_inference_size,
    parse_inference_size,
)
from utils.preprocess import preprocess_frame


try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None

try:
    import openvino as ov
except ImportError:
    ov = None


class InferenceSizeContractTests(unittest.TestCase):
    def test_normalizes_legacy_square_and_explicit_height_width(self) -> None:
        self.assertEqual(normalize_inference_size(416), (416, 416))
        self.assertEqual(normalize_inference_size((384, 640)), (384, 640))
        self.assertEqual(_normalize_inference_size((384, 640)), (384, 640))
        self.assertEqual(compact_inference_size((416, 416)), 416)
        self.assertEqual(compact_inference_size((384, 640)), (384, 640))

    def test_text_contract_is_canonical_and_rejects_ambiguous_forms(self) -> None:
        self.assertEqual(parse_inference_size("416"), (416, 416))
        self.assertEqual(parse_inference_size(" 384 x 640 "), (384, 640))
        self.assertEqual(parse_inference_size("384×640"), (384, 640))
        self.assertEqual(format_inference_size((416, 416)), "416")
        self.assertEqual(format_inference_size((384, 640)), "384x640")
        for value in ("", "384,640", "384x", "0x640", "-1", "384x0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_inference_size(value)

    def test_application_boundaries_reject_non_stride_aligned_dimensions(self) -> None:
        for value in ("383x640", "384x639", "383"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_args(["--inference-size", value])
            with self.subTest(value=value), self.assertRaises(SystemExit):
                build_export_parser().parse_args(["--imgsz", value])
            with self.subTest(value=value), self.assertRaises(SystemExit):
                build_benchmark_parser().parse_args(
                    ["--model", "custom.onnx", "--inference-size", value]
                )

    def test_cli_and_tool_parsers_keep_height_width_order(self) -> None:
        self.assertEqual(
            parse_args(["--inference-size", "384x640"]).inference_size,
            (384, 640),
        )
        self.assertEqual(parse_args(["--inference-size", "416"]).inference_size, (416, 416))
        self.assertEqual(
            build_export_parser().parse_args(["--imgsz", "384x640"]).imgsz,
            (384, 640),
        )
        self.assertEqual(
            build_benchmark_parser()
            .parse_args(["--model", "custom.onnx", "--inference-size", "384x640"])
            .inference_size,
            (384, 640),
        )


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class RectangularPreprocessTests(unittest.TestCase):
    def test_letterbox_uses_rectangular_workspace_and_exact_transform(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        prepared = preprocess_frame(frame, inference_size=(384, 640))

        self.assertEqual(prepared.tensor.shape, (1, 3, 384, 640))
        self.assertAlmostEqual(prepared.transform.scale, 1.0 / 3.0)
        self.assertEqual(prepared.transform.pad_left, 0)
        self.assertEqual(prepared.transform.pad_top, 12)
        self.assertEqual(prepared.transform.model_height, 384)
        self.assertEqual(prepared.transform.model_width, 640)
        center = prepared.transform.to_source_box((320.0, 192.0, 320.0, 192.0))
        self.assertEqual(center, (960.0, 540.0, 960.0, 540.0))

    def test_rectangular_and_square_workspaces_do_not_alias(self) -> None:
        frame = np.full((180, 320, 3), 200, dtype=np.uint8)

        rectangle = preprocess_frame(frame, (192, 320))
        square = preprocess_frame(frame, 320)

        self.assertEqual(rectangle.tensor.shape, (1, 3, 192, 320))
        self.assertEqual(square.tensor.shape, (1, 3, 320, 320))

    def test_center_crop_retains_source_offsets_with_rectangular_model(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        prepared = preprocess_frame(frame, (384, 640), crop_size=640)

        self.assertEqual(prepared.tensor.shape, (1, 3, 384, 640))
        self.assertEqual(prepared.transform.crop_x, 320)
        self.assertEqual(prepared.transform.crop_y, 40)
        self.assertEqual(prepared.transform.pad_left, 128)
        self.assertEqual(prepared.transform.pad_top, 0)
        source_box = prepared.transform.to_source_box((128, 0, 512, 384))
        self.assertEqual(source_box, (320.0, 40.0, 960.0, 680.0))

    def test_postprocess_maps_rectangular_model_coordinates_to_source(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        prepared = preprocess_frame(frame, (384, 640))
        raw = np.asarray([[[160, 102, 480, 282, 0.9, 0]]], dtype=np.float32)

        detections = decode_yolo_output(
            raw,
            transform=prepared.transform,
            labels=("player",),
        )

        self.assertEqual(detections[0].box, (480.0, 270.0, 1440.0, 810.0))


class RectangularOnnxContractTests(unittest.TestCase):
    def test_static_rectangular_model_accepts_only_matching_tensor(self) -> None:
        detector = make_detector(
            inference_size=(384, 640),
            session_kwargs={"input_shape": [1, 3, 384, 640]},
        )

        self.assertEqual(detector.input_size, (384, 640))
        self.assertEqual(detector.inference_size, (384, 640))
        self.assertEqual(
            detector.runtime_summary["configured_input_shape"],
            [1, 3, 384, 640],
        )
        detector.infer(np.zeros((1, 3, 384, 640), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "1, 3, 384, 640"):
            detector.infer(np.zeros((1, 3, 640, 384), dtype=np.float32))

    def test_transposed_static_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ModelError, "configured input shape"):
            make_detector(
                inference_size=(384, 640),
                session_kwargs={"input_shape": [1, 3, 640, 384]},
            )

    def test_dynamic_spatial_axes_accept_rectangle_and_warm_up_exact_shape(self) -> None:
        detector = make_detector(
            inference_size=(384, 640),
            session_kwargs={"input_shape": [1, 3, "height", "width"]},
        )

        detector.warmup(2)

        self.assertEqual(detector._session.runs, 2)


@unittest.skipIf(ov is None, "OpenVINO is not installed")
class RectangularOpenVinoContractTests(unittest.TestCase):
    def test_dynamic_model_is_reshaped_warmed_and_inferred_as_height_width(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "dynamic.xml"
            parameter = ov.opset13.parameter(
                ov.PartialShape([1, 3, -1, -1]),
                np.float32,
                name="images",
            )
            model = ov.Model([ov.opset13.relu(parameter)], [parameter], "dynamic")
            ov.save_model(model, model_path)

            detector = OpenVINOYoloDetector(
                model_path=model_path,
                device="CPU",
                inference_size=(96, 160),
            )
            detector.warmup(1)
            output = detector.infer(
                np.zeros((1, 3, 96, 160), dtype=np.float32)
            )

        self.assertEqual(detector.input_size, (96, 160))
        self.assertEqual(detector.inference_size, (96, 160))
        self.assertEqual(detector.runtime_summary["input_shape"], [1, 3, 96, 160])
        self.assertEqual(output.shape, (1, 3, 96, 160))
        with self.assertRaisesRegex(ValueError, "1, 3, 96, 160"):
            detector.infer(np.zeros((1, 3, 160, 96), dtype=np.float32))

    def test_static_rectangular_model_compiles_without_transposing_axes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "static.xml"
            parameter = ov.opset13.parameter(
                [1, 3, 96, 160],
                np.float32,
                name="images",
            )
            model = ov.Model([ov.opset13.relu(parameter)], [parameter], "static")
            ov.save_model(model, model_path)

            detector = OpenVINOYoloDetector(
                model_path=model_path,
                device="CPU",
                inference_size=(96, 160),
            )
            output = detector.infer(
                np.zeros((1, 3, 96, 160), dtype=np.float32)
            )

        self.assertEqual(detector.runtime_summary["input_shape"], [1, 3, 96, 160])
        self.assertEqual(output.shape, (1, 3, 96, 160))


class RectangularLauncherTests(unittest.TestCase):
    def test_custom_launcher_emits_canonical_height_width_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            labels = root / "labels.txt"
            model.write_bytes(b"onnx")
            labels.write_text("player\n", encoding="utf-8")
            settings = LauncherSettings(
                model_preset=MODEL_PRESET_CUSTOM,
                model_path=str(model),
                labels_path=str(labels),
                backend="onnxruntime",
                inference_size="384 × 640",
            )

            arguments = settings.detector_arguments()

        self.assertEqual(settings.inference_size, "384x640")
        self.assertEqual(
            arguments[arguments.index("--inference-size") + 1],
            "384x640",
        )

    def test_rectangular_setting_round_trips_without_tuple_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "settings.json"
            settings = LauncherSettings(
                model_preset=MODEL_PRESET_CUSTOM,
                inference_size="384x640",
            )

            save_settings(settings, destination)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            loaded = LauncherSettings.from_mapping(payload)

        self.assertEqual(payload["inference_size"], "384x640")
        self.assertEqual(loaded.inference_size, "384x640")

    def test_custom_launcher_rejects_non_stride_aligned_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            labels = root / "labels.txt"
            model.write_bytes(b"onnx")
            labels.write_text("player\n", encoding="utf-8")
            settings = LauncherSettings(
                model_preset=MODEL_PRESET_CUSTOM,
                model_path=str(model),
                labels_path=str(labels),
                backend="onnxruntime",
                inference_size="383x640",
            )

            with self.assertRaisesRegex(SettingsError, "divisible by 32"):
                settings.detector_arguments()


if __name__ == "__main__":
    unittest.main()
