"""Qt desktop launcher for ProAim.

This front end is deliberately a sibling of the Tk launcher rather than a
replacement: both drive the same :class:`~launcher.settings.LauncherSettings`
and the same toolkit-independent process helpers, so the detector command they
produce is identical and neither can drift into being the "real" one.

Layout is a sidebar of sections rather than a tab strip.  Sections grow over
time, and a vertical list stays readable at any count while a tab strip does
not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from aiming.makcu import (
    BUTTON_NAMES,
    MakcuAimConfig,
    MakcuAimingController,
    MakcuError,
    detect_makcu_port,
)

from . import qt_theme
from .process import force_stop, request_stop, start_detector
from .settings import (
    AIM_OUTPUT_MAKCU,
    MODEL_PRESET_CUSTOM,
    MODEL_PRESET_COCO,
    MODEL_PRESET_COCO_BALANCED,
    MODEL_PRESET_COCO_HIGH,
    MODEL_PRESET_FORT_PLAYER,
    MODEL_PRESET_FORT_PLAYER_BALANCED,
    MODEL_PRESETS,
    SETTINGS_VERSION,
    SELF_POSITION_CENTER,
    SELF_POSITION_LEFT,
    SELF_POSITION_RIGHT,
    SOURCE_CAMERA,
    SOURCE_SCREEN,
    SOURCE_VIDEO,
    LauncherSettings,
    SettingsError,
    launcher_command,
    load_settings,
    save_settings,
    settings_path,
)


UNIT = qt_theme.UNIT
PROAIM_BUILD_TAG = "2026-08-10-makcu-monitor-v1"
MAKCU_BUTTON_LABELS = BUTTON_NAMES
AIM_POINT_OPTIONS = (
    ("Upper head", "0.08"),
    ("Head center (Recommended)", "0.12"),
    ("Lower head", "0.16"),
)
AIM_POINT_LABEL_FOR_VALUE = {value: label for label, value in AIM_POINT_OPTIONS}
SELF_POSITION_LABELS = {
    SELF_POSITION_LEFT: "Left of center (common)",
    SELF_POSITION_CENTER: "Center",
    SELF_POSITION_RIGHT: "Right of center",
}
SELF_POSITION_VALUES = {label: value for value, label in SELF_POSITION_LABELS.items()}
STRENGTH_SLIDER_SCALE = 100
SMOOTHING_SLIDER_SCALE = 100
CONFIDENCE_SLIDER_SCALE = 100
MODEL_TIER_LOW = "low"
MODEL_TIER_MID = "mid"
MODEL_TIER_HIGH = "high"
MODEL_TIER_OPTIONS = (
    ("Low-end PC", MODEL_TIER_LOW),
    ("Mid-tier PC", MODEL_TIER_MID),
    ("High-end PC", MODEL_TIER_HIGH),
)
MODEL_TIER_PRESET_KEYS = {
    MODEL_TIER_LOW: (
        MODEL_PRESET_FORT_PLAYER,
        MODEL_PRESET_COCO,
    ),
    MODEL_TIER_MID: (
        MODEL_PRESET_FORT_PLAYER_BALANCED,
        MODEL_PRESET_FORT_PLAYER,
        MODEL_PRESET_COCO_BALANCED,
    ),
    MODEL_TIER_HIGH: (
        MODEL_PRESET_COCO_HIGH,
        MODEL_PRESET_FORT_PLAYER_BALANCED,
        MODEL_PRESET_COCO_BALANCED,
        MODEL_PRESET_FORT_PLAYER,
        MODEL_PRESET_COCO,
    ),
}
MODEL_TIER_DEFAULT_PRESET = {
    MODEL_TIER_LOW: MODEL_PRESET_FORT_PLAYER,
    MODEL_TIER_MID: MODEL_PRESET_FORT_PLAYER_BALANCED,
    MODEL_TIER_HIGH: MODEL_PRESET_COCO_HIGH,
}
HIGH_END_CAPTURE_WIDTH = "1920"
HIGH_END_CAPTURE_HEIGHT = "1080"
HIGH_END_CAPTURE_FPS = "100"


def _label(text: str, role: str = "") -> QLabel:
    widget = QLabel(text)
    if role:
        widget.setProperty("role", role)
    return widget


def _card(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    """Return a titled panel and the layout callers should fill."""

    frame = QFrame()
    frame.setProperty("role", "card")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(UNIT * 2, UNIT * 2, UNIT * 2, UNIT * 2)
    outer.setSpacing(UNIT + 2)

    heading = _label(title.upper(), "sectionHeading")
    outer.addWidget(heading)
    if subtitle:
        note = _label(subtitle, "subtitle")
        note.setWordWrap(True)
        outer.addWidget(note)
    return frame, outer


def _field_row(layout: QGridLayout, row: int, text: str, widget: QWidget) -> None:
    layout.addWidget(_label(text, "fieldLabel"), row, 0, Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(widget, row, 1)


class Section(QWidget):
    """A scrollable page of cards."""

    def __init__(self, title: str, description: str) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(UNIT * 4, UNIT * 3, UNIT * 4, UNIT * 3)
        outer.setSpacing(UNIT * 2)

        outer.addWidget(_label(title, "title"))
        if description:
            note = _label(description, "subtitle")
            note.setWordWrap(True)
            outer.addWidget(note)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        self.body = QVBoxLayout(holder)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(UNIT * 2)
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)

    def add_card(self, card: QWidget) -> None:
        self.body.addWidget(card)

    def add_stretch(self) -> None:
        self.body.addStretch(1)


class LauncherWindow(QMainWindow):
    """Main window: sidebar, section stack, and a persistent run bar."""

    log_line = Signal(str)
    makcu_verification_done = Signal(bool, str, str, str)
    makcu_verification_progress = Signal(str)

    def __init__(self, settings: LauncherSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"ProAim ({PROAIM_BUILD_TAG})")
        self.resize(1080, 760)
        self.setMinimumSize(880, 620)

        self.settings = settings or load_settings(settings_path())
        self.process: subprocess.Popen[str] | None = None
        self._reader: Any | None = None
        self._makcu_verify_thread: threading.Thread | None = None
        self._makcu_verify_cancel = threading.Event()
        self._makcu_monitor_thread: threading.Thread | None = None
        self._makcu_monitor_cancel = threading.Event()
        self._makcu_verified_port = self.settings.aim_makcu_verified_port
        self._makcu_verified_button = self.settings.aim_makcu_verified_button

        self._build_interface()
        self._load_from_settings()

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_process)
        self._poll.start(400)

        self.log_line.connect(self._append_log)
        self.makcu_verification_done.connect(self._apply_makcu_verification_result)
        self.makcu_verification_progress.connect(self._apply_makcu_verification_progress)

    # -- construction ----------------------------------------------------
    def _build_interface(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        columns = QHBoxLayout(root)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(0)

        columns.addWidget(self._build_sidebar())

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_capture_section())
        self.stack.addWidget(self._build_detection_section())
        self.stack.addWidget(self._build_hardware_section())
        right_layout.addWidget(self.stack, 1)
        right_layout.addWidget(self._build_footer())

        columns.addWidget(right, 1)
        self._select_section(0)

    def _build_sidebar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(206)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, UNIT * 3, 0, UNIT * 2)
        layout.setSpacing(0)

        brand = QWidget()
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(UNIT * 2 + 4, 0, UNIT * 2, UNIT * 3)
        brand_layout.setSpacing(2)
        name = _label("PROAIM")
        font = QFont()
        font.setPointSize(12)
        font.setWeight(QFont.Weight.Bold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        name.setFont(font)
        brand_layout.addWidget(name)
        brand_layout.addWidget(_label("Offline object detection", "subtitle"))
        layout.addWidget(brand)

        self.nav_buttons: list[QPushButton] = []
        for index, text in enumerate(("Capture source", "Detection", "Hardware")):
            button = QPushButton(text)
            button.setProperty("role", "nav")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, i=index: self._select_section(i))
            layout.addWidget(button)
            self.nav_buttons.append(button)

        layout.addStretch(1)
        version = _label(f"Settings schema v{SETTINGS_VERSION}", "subtitle")
        version.setContentsMargins(UNIT * 2 + 4, 0, UNIT * 2, 0)
        layout.addWidget(version)
        return bar

    def _build_capture_section(self) -> QWidget:
        section = Section(
            "Capture source",
            "Choose where frames come from. Screen capture reads a Moonlight "
            "window or monitor; a capture card or camera is read directly.",
        )

        card, layout = _card("Source")
        # Exactly one source can be active, so these are radios rather than
        # checkboxes; the widget should state the constraint, not just enforce it.
        self.source_screen = QRadioButton("Moonlight / screen capture")
        self.source_camera = QRadioButton("Capture card or camera")
        self.source_video = QRadioButton("Video file")
        self._source_boxes = {
            SOURCE_SCREEN: self.source_screen,
            SOURCE_CAMERA: self.source_camera,
            SOURCE_VIDEO: self.source_video,
        }
        for mode, box in self._source_boxes.items():
            box.clicked.connect(lambda _=False, m=mode: self._choose_source(m))
            layout.addWidget(box)
        section.add_card(card)

        screen_card, screen_layout = _card(
            "Screen capture", "X11 only. Keep the preview off the captured area."
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(UNIT * 2)
        grid.setVerticalSpacing(UNIT + 2)
        grid.setColumnStretch(1, 1)
        self.screen_monitor = QLineEdit()
        self.screen_fps = QLineEdit()
        _field_row(grid, 0, "Monitor number", self.screen_monitor)
        _field_row(grid, 1, "Capture rate (fps)", self.screen_fps)
        screen_layout.addLayout(grid)
        self.screen_card = screen_card
        section.add_card(screen_card)

        camera_card, camera_layout = _card(
            "Capture card / camera",
            "High-framerate cards usually require the MJPG pixel format; "
            "uncompressed modes are limited by USB bandwidth.",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(UNIT * 2)
        grid.setVerticalSpacing(UNIT + 2)
        grid.setColumnStretch(1, 1)
        self.camera_index = QLineEdit()
        self.capture_width = QLineEdit()
        self.capture_height = QLineEdit()
        self.capture_fps = QLineEdit()
        _field_row(grid, 0, "Device index", self.camera_index)
        _field_row(grid, 1, "Width", self.capture_width)
        _field_row(grid, 2, "Height", self.capture_height)
        _field_row(grid, 3, "Frame rate (fps)", self.capture_fps)
        camera_layout.addLayout(grid)
        self.camera_card = camera_card
        section.add_card(camera_card)

        video_card, video_layout = _card("Video file")
        row = QHBoxLayout()
        row.setSpacing(UNIT)
        self.video_path = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_video)
        row.addWidget(self.video_path, 1)
        row.addWidget(browse)
        video_layout.addLayout(row)
        self.video_card = video_card
        section.add_card(video_card)

        section.add_stretch()
        return section

    def _build_detection_section(self) -> QWidget:
        section = Section(
            "Detection",
            "The model and the thresholds applied to its output.",
        )

        card, layout = _card("Model")
        grid = QGridLayout()
        grid.setHorizontalSpacing(UNIT * 2)
        grid.setVerticalSpacing(UNIT + 2)
        grid.setColumnStretch(1, 1)
        self.model_tier = QComboBox()
        for label, value in MODEL_TIER_OPTIONS:
            self.model_tier.addItem(label, value)
        self.model_tier.currentIndexChanged.connect(self._model_tier_changed)
        self.model_preset = QComboBox()
        for preset in MODEL_PRESETS:
            self.model_preset.addItem(preset.label, preset.key)
        self.model_preset.currentIndexChanged.connect(self._preset_changed)
        _field_row(grid, 0, "Performance tier", self.model_tier)
        _field_row(grid, 1, "Detector", self.model_preset)

        self.custom_model_path = QLineEdit()
        self.custom_labels_path = QLineEdit()
        self.custom_model_browse = QPushButton("Browse model…")
        self.custom_model_browse.clicked.connect(self._browse_custom_model)
        self.custom_labels_browse = QPushButton("Browse labels…")
        self.custom_labels_browse.clicked.connect(self._browse_custom_labels)

        model_row = QWidget()
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.setSpacing(UNIT)
        model_row_layout.addWidget(self.custom_model_path, 1)
        model_row_layout.addWidget(self.custom_model_browse)

        labels_row = QWidget()
        labels_row_layout = QHBoxLayout(labels_row)
        labels_row_layout.setContentsMargins(0, 0, 0, 0)
        labels_row_layout.setSpacing(UNIT)
        labels_row_layout.addWidget(self.custom_labels_path, 1)
        labels_row_layout.addWidget(self.custom_labels_browse)

        _field_row(grid, 2, "Custom model", model_row)
        _field_row(grid, 3, "Custom labels", labels_row)
        layout.addLayout(grid)

        actions = QHBoxLayout()
        actions.setSpacing(UNIT)
        open_train = QPushButton("Open training script")
        open_train.clicked.connect(self._open_training_script)
        open_export = QPushButton("Open export script")
        open_export.clicked.connect(self._open_export_script)
        actions.addWidget(open_train)
        actions.addWidget(open_export)
        actions.addStretch(1)
        layout.addLayout(actions)

        tier_hint = _label(
            "Pick Low, Mid, or High first, then choose a model tuned for that performance class.",
            "subtitle",
        )
        tier_hint.setWordWrap(True)
        layout.addWidget(tier_hint)

        self.preset_note = _label("", "subtitle")
        self.preset_note.setWordWrap(True)
        layout.addWidget(self.preset_note)
        section.add_card(card)

        card, layout = _card("Thresholds")
        grid = QGridLayout()
        grid.setHorizontalSpacing(UNIT * 2)
        grid.setVerticalSpacing(UNIT + 2)
        grid.setColumnStretch(1, 1)
        self.inference_size = QComboBox()
        self.inference_size.addItems(("256", "320", "416", "640"))
        self.inference_size.setEditable(True)
        self.confidence = QSlider(Qt.Orientation.Horizontal)
        self.confidence.setRange(5, 95)
        self.confidence.setSingleStep(1)
        self.confidence_value = _label("0.25", "subtitle")
        self.confidence.valueChanged.connect(self._sync_confidence_label)
        self.iou_threshold = QLineEdit()

        confidence_row = QWidget()
        confidence_layout = QHBoxLayout(confidence_row)
        confidence_layout.setContentsMargins(0, 0, 0, 0)
        confidence_layout.setSpacing(UNIT)
        confidence_layout.addWidget(self.confidence, 1)
        confidence_layout.addWidget(self.confidence_value)

        _field_row(grid, 0, "Inference size", self.inference_size)
        _field_row(grid, 1, "Confidence", confidence_row)
        _field_row(grid, 2, "IoU threshold", self.iou_threshold)
        layout.addLayout(grid)
        section.add_card(card)

        card, layout = _card(
            "Display",
            "The preview is a separate window. When capturing a monitor, keep "
            "it outside the captured area.",
        )
        self.preview = QCheckBox("Show preview window")
        self.draw = QCheckBox("Draw boxes on the preview")
        layout.addWidget(self.preview)
        layout.addWidget(self.draw)
        section.add_card(card)

        card, layout = _card(
            "Third-person self filter",
            "Prevent aiming at your own on-screen avatar by filtering one persistent player-sized box in a bottom anchor zone.",
        )
        self.ignore_self = QCheckBox("Ignore my on-screen character")
        layout.addWidget(self.ignore_self)
        self.self_position = QComboBox()
        for value, label in SELF_POSITION_LABELS.items():
            self.self_position.addItem(label, value)
        grid = QGridLayout()
        grid.setHorizontalSpacing(UNIT * 2)
        grid.setVerticalSpacing(UNIT + 2)
        grid.setColumnStretch(1, 1)
        _field_row(grid, 0, "Character appears", self.self_position)
        layout.addLayout(grid)
        hint = _label(
            "If self-aim still happens, set Character appears to Center or Right to match your camera framing.",
            "subtitle",
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        section.add_card(card)

        card, layout = _card(
            "MAKCU aim",
            "Optional detection-driven mouse correction through a MAKCU passthrough board. "
            "AI movement is sent only while the chosen mouse button is held.",
        )
        self.aim = QCheckBox("Enable MAKCU aim")
        self.aim.toggled.connect(self._update_aim_state)
        layout.addWidget(self.aim)

        grid = QGridLayout()
        grid.setHorizontalSpacing(UNIT * 2)
        grid.setVerticalSpacing(UNIT + 2)
        grid.setColumnStretch(1, 1)

        self.aim_label = QLineEdit()
        self.aim_point = QComboBox()
        for label, value in AIM_POINT_OPTIONS:
            self.aim_point.addItem(label, value)
        self.aim_invert_x = QCheckBox("Invert X")
        self.aim_invert_y = QCheckBox("Invert Y")
        self.aim_makcu_port = QLineEdit()
        self.aim_makcu_port.textChanged.connect(self._makcu_verification_selection_changed)
        self.detect_makcu_button = QPushButton("Detect")
        self.detect_makcu_button.clicked.connect(self._detect_makcu_port)
        self.browse_makcu_button = QPushButton("Browse…")
        self.browse_makcu_button.clicked.connect(self._browse_aim_makcu_port)
        self.aim_makcu_button = QComboBox()
        for index, label in enumerate(MAKCU_BUTTON_LABELS):
            self.aim_makcu_button.addItem(label, index)
        self.aim_makcu_button.currentIndexChanged.connect(self._makcu_verification_selection_changed)
        self.verify_makcu_button = QPushButton("Verify Right Mouse…")
        self.verify_makcu_button.clicked.connect(self.verify_makcu_activation)
        self.monitor_makcu_button = QPushButton("Monitor Mouse Clicks…")
        self.monitor_makcu_button.clicked.connect(self.monitor_makcu_buttons)
        self.aim_makcu_strength = QSlider(Qt.Orientation.Horizontal)
        self.aim_makcu_strength.setRange(5, 250)
        self.aim_makcu_strength.setSingleStep(1)
        self.aim_makcu_strength_value = _label("0.50", "subtitle")
        self.aim_makcu_strength.valueChanged.connect(self._sync_strength_label)
        self.aim_makcu_smoothing = QSlider(Qt.Orientation.Horizontal)
        self.aim_makcu_smoothing.setRange(10, 100)
        self.aim_makcu_smoothing.setSingleStep(1)
        self.aim_makcu_smoothing_value = _label("0.78", "subtitle")
        self.aim_makcu_smoothing.valueChanged.connect(self._sync_smoothing_label)
        self.aim_makcu_max_step = QLineEdit()
        self.aim_makcu_verification_status = _label("", "subtitle")
        self.aim_makcu_verification_status.setWordWrap(True)

        port_row = QWidget()
        port_layout = QHBoxLayout(port_row)
        port_layout.setContentsMargins(0, 0, 0, 0)
        port_layout.setSpacing(UNIT)
        port_layout.addWidget(self.aim_makcu_port, 1)
        port_layout.addWidget(self.detect_makcu_button)
        port_layout.addWidget(self.browse_makcu_button)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(UNIT)
        button_layout.addWidget(self.aim_makcu_button)
        button_layout.addWidget(self.verify_makcu_button)
        button_layout.addWidget(self.monitor_makcu_button)
        button_layout.addStretch(1)

        tuning_row = QWidget()
        tuning_layout = QHBoxLayout(tuning_row)
        tuning_layout.setContentsMargins(0, 0, 0, 0)
        tuning_layout.setSpacing(UNIT)
        tuning_layout.addWidget(_label("Strength", "fieldLabel"))
        tuning_layout.addWidget(self.aim_makcu_strength, 1)
        tuning_layout.addWidget(self.aim_makcu_strength_value)
        tuning_layout.addSpacing(UNIT)
        tuning_layout.addWidget(_label("Smoothing", "fieldLabel"))
        tuning_layout.addWidget(self.aim_makcu_smoothing, 1)
        tuning_layout.addWidget(self.aim_makcu_smoothing_value)
        tuning_layout.addSpacing(UNIT)
        tuning_layout.addWidget(_label("Max step", "fieldLabel"))
        tuning_layout.addWidget(self.aim_makcu_max_step)
        tuning_layout.addStretch(1)

        _field_row(grid, 0, "Target label", self.aim_label)
        _field_row(grid, 1, "Aim point", self.aim_point)
        grid.addWidget(self.aim_invert_x, 1, 2)
        grid.addWidget(self.aim_invert_y, 1, 3)
        _field_row(grid, 2, "MAKCU device", port_row)
        _field_row(grid, 3, "Hold to activate", button_row)
        _field_row(grid, 4, "Tuning", tuning_row)
        layout.addLayout(grid)
        self.aim_makcu_summary = _label("", "subtitle")
        self.aim_makcu_summary.setWordWrap(True)
        layout.addWidget(self.aim_makcu_summary)
        layout.addWidget(self.aim_makcu_verification_status)
        note = _label(
            "Verification is read-only. It watches for a press and release from the selected "
            "MAKCU mouse button and never moves or clicks during the check.",
            "subtitle",
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        build_note = _label(f"MAKCU verifier build: {PROAIM_BUILD_TAG}", "subtitle")
        layout.addWidget(build_note)
        section.add_card(card)

        section.add_stretch()
        return section

    def _build_hardware_section(self) -> QWidget:
        section = Section(
            "Hardware",
            "Detection runs on Intel CPU, Intel graphics and NPU through "
            "OpenVINO, and on AMD or NVIDIA GPUs through ONNX Runtime.",
        )

        card, layout = _card(
            "Automatic selection",
            "Scanning reports every accelerator present and whether this "
            "installation can drive it, then selects the fastest usable one.",
        )
        row = QHBoxLayout()
        row.setSpacing(UNIT)
        scan = QPushButton("Scan hardware")
        scan.setProperty("role", "primary")
        scan.clicked.connect(self._scan_hardware)
        row.addWidget(scan)
        self.install_runtime_button = QPushButton("Download GPU runtime")
        self.install_runtime_button.clicked.connect(self._install_runtime)
        self.install_runtime_button.setEnabled(False)
        self.install_runtime_button.setToolTip("Run a hardware scan first.")
        row.addWidget(self.install_runtime_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.hardware_report = QPlainTextEdit()
        self.hardware_report.setObjectName("LogView")
        self.hardware_report.setReadOnly(True)
        self.hardware_report.setMinimumHeight(190)
        self.hardware_report.setPlainText("No scan run yet.")
        layout.addWidget(self.hardware_report)
        section.add_card(card)

        card, layout = _card("Manual override")
        grid = QGridLayout()
        grid.setHorizontalSpacing(UNIT * 2)
        grid.setVerticalSpacing(UNIT + 2)
        grid.setColumnStretch(1, 1)
        self.backend = QComboBox()
        self.backend.addItems(("openvino", "onnxruntime"))
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(("CPU", "AUTO", "GPU", "NPU"))
        self.detected_accelerator = QComboBox()
        self.detected_accelerator.setToolTip("Run Scan hardware first, then select one detected accelerator profile.")
        self.detected_accelerator.addItem("Run Scan hardware first", None)
        self.apply_detected_accelerator = QPushButton("Use selection")
        self.apply_detected_accelerator.clicked.connect(self._apply_detected_accelerator)
        detected_row = QWidget()
        detected_layout = QHBoxLayout(detected_row)
        detected_layout.setContentsMargins(0, 0, 0, 0)
        detected_layout.setSpacing(UNIT)
        detected_layout.addWidget(self.detected_accelerator, 1)
        detected_layout.addWidget(self.apply_detected_accelerator)
        _field_row(grid, 0, "Detected accelerators", detected_row)
        _field_row(grid, 1, "Backend", self.backend)
        _field_row(grid, 2, "Device", self.device)
        layout.addLayout(grid)
        section.add_card(card)

        section.add_stretch()
        return section

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("Footer")
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(UNIT * 4, UNIT * 2, UNIT * 4, UNIT * 2)
        layout.setSpacing(UNIT + 2)

        layout.addWidget(_label("DETECTOR OUTPUT", "sectionHeading"))
        self.log = QPlainTextEdit()
        self.log.setObjectName("LogView")
        self.log.setReadOnly(True)
        self.log.setFixedHeight(116)
        self.log.setPlaceholderText("Detector output appears here once a run starts.")
        layout.addWidget(self.log)

        row = QHBoxLayout()
        row.setSpacing(UNIT * 2)
        self.status = _label("Idle.", "status")
        row.addWidget(self.status, 1)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setEnabled(False)
        row.addWidget(self.stop_button)

        self.start_button = QPushButton("Start detection")
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self._start)
        row.addWidget(self.start_button)

        layout.addLayout(row)
        return footer

    # -- state -----------------------------------------------------------
    def _select_section(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == index)

    def _choose_source(self, mode: str) -> None:
        for key, box in self._source_boxes.items():
            box.setChecked(key == mode)
        self.screen_card.setVisible(mode == SOURCE_SCREEN)
        self.camera_card.setVisible(mode == SOURCE_CAMERA)
        self.video_card.setVisible(mode == SOURCE_VIDEO)

    def _preset_changed(self) -> None:
        key = self.model_preset.currentData()
        custom = key == MODEL_PRESET_CUSTOM
        self.custom_model_path.setEnabled(custom)
        self.custom_labels_path.setEnabled(custom)
        self.custom_model_browse.setEnabled(custom)
        self.custom_labels_browse.setEnabled(custom)
        for preset in MODEL_PRESETS:
            if preset.key == key:
                self.preset_note.setText(preset.description)
                if preset.inference_size is not None:
                    self.inference_size.setCurrentText(str(preset.inference_size))
                break

    def _model_tier_changed(self) -> None:
        self._refresh_model_preset_options(preferred_key=self.model_preset.currentData())
        if str(self.model_tier.currentData() or MODEL_TIER_MID) == MODEL_TIER_HIGH:
            self.capture_width.setText(HIGH_END_CAPTURE_WIDTH)
            self.capture_height.setText(HIGH_END_CAPTURE_HEIGHT)
            self.capture_fps.setText(HIGH_END_CAPTURE_FPS)
            self.screen_fps.setText(HIGH_END_CAPTURE_FPS)

    def _refresh_model_preset_options(self, *, preferred_key: str | None = None) -> None:
        tier = str(self.model_tier.currentData() or MODEL_TIER_MID)
        allowed = set(MODEL_TIER_PRESET_KEYS.get(tier, MODEL_TIER_PRESET_KEYS[MODEL_TIER_MID]))
        allowed.add(MODEL_PRESET_CUSTOM)
        current = preferred_key or str(self.model_preset.currentData() or "")
        self.model_preset.blockSignals(True)
        self.model_preset.clear()
        for preset in MODEL_PRESETS:
            if preset.key in allowed:
                self.model_preset.addItem(preset.label, preset.key)
        self.model_preset.blockSignals(False)
        index = self.model_preset.findData(current)
        if index < 0:
            index = 0
        self.model_preset.setCurrentIndex(index)
        self._preset_changed()

    def _tier_for_preset(self, preset_key: str) -> str:
        for tier, keys in MODEL_TIER_PRESET_KEYS.items():
            if preset_key in keys:
                return tier
        return MODEL_TIER_MID

    def _load_from_settings(self) -> None:
        s = self.settings
        self.screen_monitor.setText(s.screen_monitor)
        self.screen_fps.setText(s.screen_fps)
        self.camera_index.setText(s.camera_index)
        self.capture_width.setText(s.capture_width)
        self.capture_height.setText(s.capture_height)
        self.capture_fps.setText(s.capture_fps)
        self.video_path.setText(s.video_path)
        self._set_confidence_slider_value(s.confidence)
        self.iou_threshold.setText(s.iou_threshold)
        self.inference_size.setCurrentText(s.inference_size)
        self.preview.setChecked(s.preview)
        self.draw.setChecked(s.draw)
        self.backend.setCurrentText(s.backend)
        self.device.setCurrentText(s.device)
        self.custom_model_path.setText(s.model_path)
        self.custom_labels_path.setText(s.labels_path)
        self.aim.setChecked(s.aim)
        self.aim_label.setText(s.aim_label)
        aim_point = AIM_POINT_LABEL_FOR_VALUE.get(s.aim_head_ratio, AIM_POINT_LABEL_FOR_VALUE["0.12"])
        self.aim_point.setCurrentText(aim_point)
        self.aim_invert_x.setChecked(s.aim_invert_x)
        self.aim_invert_y.setChecked(s.aim_invert_y)
        self.aim_makcu_port.setText(s.aim_makcu_port)
        makcu_button = min(max(self._parse_int(s.aim_makcu_button, default=1), 0), 4)
        self.aim_makcu_button.setCurrentIndex(makcu_button)
        self._set_strength_slider_value(s.aim_makcu_strength)
        self._set_smoothing_slider_value(s.aim_makcu_smoothing_alpha)
        self.aim_makcu_max_step.setText(s.aim_makcu_max_step)
        self.ignore_self.setChecked(s.ignore_self)
        position_index = self.self_position.findData(s.self_position)
        if position_index >= 0:
            self.self_position.setCurrentIndex(position_index)
        tier_value = s.model_tier if s.model_tier in MODEL_TIER_PRESET_KEYS else self._tier_for_preset(s.model_preset)
        tier_index = self.model_tier.findData(tier_value)
        if tier_index >= 0:
            self.model_tier.setCurrentIndex(tier_index)
        preferred_model = s.model_preset
        if preferred_model not in MODEL_TIER_PRESET_KEYS.get(tier_value, ()):
            preferred_model = MODEL_TIER_DEFAULT_PRESET.get(tier_value, s.model_preset)
        self._refresh_model_preset_options(preferred_key=preferred_model)
        self._preset_changed()
        if tier_value == MODEL_TIER_HIGH:
            self.capture_width.setText(HIGH_END_CAPTURE_WIDTH)
            self.capture_height.setText(HIGH_END_CAPTURE_HEIGHT)
            self.capture_fps.setText(HIGH_END_CAPTURE_FPS)
            self.screen_fps.setText(HIGH_END_CAPTURE_FPS)
        self._choose_source(s.source_mode)
        self._makcu_verification_selection_changed()
        self._update_aim_state()

    def collect(self) -> LauncherSettings:
        """Fold the widgets back into a settings object."""

        mode = next(
            (key for key, box in self._source_boxes.items() if box.isChecked()),
            SOURCE_SCREEN,
        )
        s = self.settings
        s.source_mode = mode
        s.screen_monitor = self.screen_monitor.text()
        s.screen_fps = self.screen_fps.text()
        s.camera_index = self.camera_index.text()
        s.capture_width = self.capture_width.text()
        s.capture_height = self.capture_height.text()
        s.capture_fps = self.capture_fps.text()
        s.video_path = self.video_path.text()
        s.confidence = f"{self._confidence_from_slider():.2f}"
        s.iou_threshold = self.iou_threshold.text()
        s.inference_size = self.inference_size.currentText()
        s.preview = self.preview.isChecked()
        s.draw = self.draw.isChecked()
        s.backend = self.backend.currentText()
        s.device = self.device.currentText()
        s.model_tier = str(self.model_tier.currentData() or MODEL_TIER_MID)
        s.model_preset = self.model_preset.currentData()
        s.model_path = self.custom_model_path.text().strip()
        s.labels_path = self.custom_labels_path.text().strip()
        s.aim = self.aim.isChecked()
        s.aim_label = self.aim_label.text().strip()
        s.aim_invert_x = self.aim_invert_x.isChecked()
        s.aim_invert_y = self.aim_invert_y.isChecked()
        s.aim_head_ratio = str(self.aim_point.currentData() or "0.12")
        s.aim_output = AIM_OUTPUT_MAKCU
        s.aim_makcu_port = self.aim_makcu_port.text().strip()
        s.aim_makcu_button = str(self._selected_makcu_button())
        s.aim_makcu_strength = f"{self._strength_from_slider():.2f}"
        s.aim_makcu_smoothing_alpha = f"{self._smoothing_from_slider():.2f}"
        s.aim_makcu_max_step = self.aim_makcu_max_step.text().strip()
        s.aim_makcu_prediction_lead_seconds = self.settings.aim_makcu_prediction_lead_seconds
        s.aim_makcu_derivative_damping_seconds = (
            self.settings.aim_makcu_derivative_damping_seconds
        )
        s.aim_makcu_verified_port = self._makcu_verified_port
        s.aim_makcu_verified_button = self._makcu_verified_button
        s.ignore_self = self.ignore_self.isChecked()
        s.self_position = str(self.self_position.currentData() or SELF_POSITION_LEFT)
        return s

    # -- actions ---------------------------------------------------------
    def _browse_video(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose a video file", "", "Video files (*.mp4 *.mkv *.avi *.mov);;All files (*)"
        )
        if chosen:
            self.video_path.setText(chosen)

    def _browse_aim_makcu_port(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Choose the MAKCU serial device",
            str(Path.home()),
            "All files (*)",
        )
        if chosen:
            self.aim_makcu_port.setText(chosen)

    def _detect_makcu_port(self) -> None:
        if self.process is not None and self.process.poll() is None:
            QMessageBox.information(
                self,
                "Stop detection first",
                "Stop detection before scanning for MAKCU so only one process owns its serial port.",
            )
            return
        try:
            detected = detect_makcu_port()
        except (MakcuError, OSError, ValueError) as exc:
            self._set_status(str(exc), "warn")
            QMessageBox.warning(self, "MAKCU not found", str(exc))
            return
        self.aim_makcu_port.setText(detected)
        self._set_status(f"Detected MAKCU at {detected}.", "ok")

    def _selected_makcu_button(self) -> int:
        return int(self.aim_makcu_button.currentData() or 1)

    def _makcu_verification_matches(self) -> bool:
        return bool(
            self._makcu_verified_port
            and self._makcu_verified_port == self.aim_makcu_port.text().strip()
            and self._makcu_verified_button == str(self._selected_makcu_button())
        )

    def _refresh_makcu_verification_status(self) -> None:
        if self._makcu_verification_matches():
            button = MAKCU_BUTTON_LABELS[self._selected_makcu_button()]
            self.aim_makcu_verification_status.setText(
                f"Verified: {button} press and release were reported by this MAKCU."
            )
        elif self.aim_makcu_port.text().strip():
            self.aim_makcu_verification_status.setText(
                "Not verified. Detect or choose the board, then verify the selected mouse button."
            )
        else:
            self.aim_makcu_verification_status.setText(
                "No MAKCU selected yet. Click Detect, then verify the activation button."
            )
        self._refresh_makcu_summary()

    def _refresh_makcu_summary(self) -> None:
        port = self.aim_makcu_port.text().strip() or "not selected"
        button = MAKCU_BUTTON_LABELS[self._selected_makcu_button()]
        verified = "verified" if self._makcu_verification_matches() else "not verified"
        self.aim_makcu_summary.setText(
            f"Selected board: {port} | activation: {button} Mouse | status: {verified}"
        )

    def _makcu_verification_selection_changed(self) -> None:
        button = MAKCU_BUTTON_LABELS[self._selected_makcu_button()]
        self.verify_makcu_button.setText(f"Verify {button} Mouse…")
        self._refresh_makcu_verification_status()

    def _update_aim_state(self) -> None:
        enabled = self.aim.isChecked()
        for widget in (
            self.aim_label,
            self.aim_point,
            self.aim_invert_x,
            self.aim_invert_y,
            self.aim_makcu_port,
            self.detect_makcu_button,
            self.browse_makcu_button,
            self.aim_makcu_button,
            self.aim_makcu_strength,
            self.aim_makcu_smoothing,
            self.aim_makcu_max_step,
        ):
            widget.setEnabled(enabled)
        verify_busy = (
            self._makcu_verify_thread is not None and self._makcu_verify_thread.is_alive()
        )
        monitor_busy = (
            self._makcu_monitor_thread is not None and self._makcu_monitor_thread.is_alive()
        )
        self.verify_makcu_button.setEnabled(enabled and not verify_busy)
        self.monitor_makcu_button.setEnabled(enabled and not verify_busy and not monitor_busy)

    def monitor_makcu_buttons(self) -> None:
        if self.process is not None and self.process.poll() is None:
            QMessageBox.information(
                self,
                "Stop detection first",
                "Stop detection before monitoring MAKCU so only one process owns its serial port.",
            )
            return
        if self._makcu_verify_thread is not None and self._makcu_verify_thread.is_alive():
            return
        if self._makcu_monitor_thread is not None and self._makcu_monitor_thread.is_alive():
            return

        port = self.aim_makcu_port.text().strip()
        if not port:
            try:
                port = detect_makcu_port()
            except (MakcuError, OSError, ValueError) as exc:
                self._set_status(str(exc), "warn")
                QMessageBox.warning(self, "MAKCU not found", str(exc))
                return
            self.aim_makcu_port.setText(port)

        self._makcu_monitor_cancel.clear()
        self.aim_makcu_verification_status.setText(
            "Monitoring MAKCU button masks for 10s… click your mouse now."
        )
        self._set_status("Monitoring MAKCU button masks…", "warn")
        self._makcu_monitor_thread = threading.Thread(
            target=self._monitor_makcu_buttons_worker,
            args=(port,),
            name="makcu-button-monitor",
            daemon=True,
        )
        self._makcu_monitor_thread.start()
        self._update_aim_state()

    def _monitor_makcu_buttons_worker(self, port: str) -> None:
        controller = MakcuAimingController(MakcuAimConfig(port=port, activation_button=1))
        deadline = time.monotonic() + 10.0
        last_mask = -1
        saw_nonzero = False
        try:
            controller.start(output_loop=False)
            while time.monotonic() < deadline and not self._makcu_monitor_cancel.is_set():
                mask = controller.poll_button_mask()
                if mask != last_mask:
                    last_mask = mask
                    names = [
                        MAKCU_BUTTON_LABELS[index]
                        for index in range(len(MAKCU_BUTTON_LABELS))
                        if mask & (1 << index)
                    ]
                    pressed = ", ".join(names) if names else "none"
                    self.makcu_verification_progress.emit(
                        f"MAKCU mask 0x{mask:02X} | pressed: {pressed}"
                    )
                if mask:
                    saw_nonzero = True
                time.sleep(0.01)
            if not self._makcu_monitor_cancel.is_set():
                if saw_nonzero:
                    self.makcu_verification_progress.emit(
                        "Monitor finished. MAKCU click reports were received."
                    )
                else:
                    self.makcu_verification_progress.emit(
                        "Monitor finished with no click reports. Mouse clicks are not reaching MAKCU input."
                    )
        except (MakcuError, OSError, ValueError) as exc:
            self.makcu_verification_progress.emit(f"Monitor error: {exc}")
        finally:
            controller.stop()
            self._makcu_monitor_thread = None
            self._update_aim_state()

    def verify_makcu_activation(self) -> None:
        if self.process is not None and self.process.poll() is None:
            QMessageBox.information(
                self,
                "Stop detection first",
                "Stop detection before verifying MAKCU so only one process owns its serial port.",
            )
            return
        if self._makcu_verify_thread is not None and self._makcu_verify_thread.is_alive():
            return

        port = self.aim_makcu_port.text().strip()
        if not port:
            try:
                port = detect_makcu_port()
            except (MakcuError, OSError, ValueError) as exc:
                self._set_status(str(exc), "warn")
                QMessageBox.warning(self, "MAKCU not found", str(exc))
                return
            self.aim_makcu_port.setText(port)

        button = self._selected_makcu_button()
        button_name = MAKCU_BUTTON_LABELS[button]
        self._makcu_verify_cancel.clear()
        self.aim_makcu_verification_status.setText(
            f"Waiting for {button_name} press… (no game stream required)"
        )
        self.aim_makcu_summary.setText(
            f"Selected board: {port} | activation: {button_name} Mouse | status: waiting for verification"
        )
        self._set_status("Listening for MAKCU button press…", "warn")
        self.verify_makcu_button.setEnabled(False)
        self._makcu_verify_thread = threading.Thread(
            target=self._verify_makcu_activation_worker,
            args=(port, button),
            name="makcu-button-verifier",
            daemon=True,
        )
        self._makcu_verify_thread.start()

    def _verify_makcu_activation_worker(self, port: str, button: int) -> None:
        controller = MakcuAimingController(
            MakcuAimConfig(port=port, activation_button=button)
        )
        deadline = time.monotonic() + 10.0
        button_name = MAKCU_BUTTON_LABELS[button]
        next_progress = time.monotonic() + 1.0
        saw_any_button_report = False
        last_mask = 0
        try:
            controller.start(output_loop=False)
            self.makcu_verification_progress.emit(
                f"Listening for {button_name}… click the physical mouse connected through MAKCU."
            )
            while time.monotonic() < deadline and not self._makcu_verify_cancel.is_set():
                mask = controller.poll_button_mask()
                if mask:
                    saw_any_button_report = True
                    last_mask = mask
                    detected_button = next(
                        (
                            index
                            for index in range(len(MAKCU_BUTTON_LABELS))
                            if mask & (1 << index)
                        ),
                        None,
                    )
                    if detected_button is not None:
                        detected_name = MAKCU_BUTTON_LABELS[detected_button]
                        selected_name = MAKCU_BUTTON_LABELS[button]
                        self.makcu_verification_progress.emit(
                            f"Detected {detected_name} (mask 0x{mask:02X}). Finishing verification…"
                        )
                        if detected_button == button:
                            detail = f"Detected selected {selected_name} (mask 0x{mask:02X})."
                        else:
                            detail = (
                                f"Selected {selected_name}, but MAKCU reported {detected_name} "
                                f"(mask 0x{mask:02X}). Activation was auto-mapped to {detected_name}."
                            )
                        self.makcu_verification_done.emit(
                            True,
                            port,
                            str(detected_button),
                            detail,
                        )
                        return
                    else:
                        detail = (
                            f"MAKCU reported button mask 0x{mask:02X}, which does not map "
                            "to a supported activation button."
                        )
                    self.makcu_verification_done.emit(False, port, str(button), detail)
                    return
                now = time.monotonic()
                if now >= next_progress:
                    self.makcu_verification_progress.emit(
                        f"Still waiting for {button_name}… ensure your mouse is plugged into MAKCU input, then click again."
                    )
                    next_progress = now + 1.0
                time.sleep(0.01)
            detail = (
                "Verification cancelled."
                if self._makcu_verify_cancel.is_set()
                else (
                    (
                        f"No supported click was detected from this MAKCU. "
                        f"Buttons were reported (last mask 0x{last_mask:02X}) but could not be mapped."
                    )
                    if saw_any_button_report
                    else (
                        "No button reports were received from MAKCU. No game stream is required; "
                        "this usually means the mouse is not routed through MAKCU input."
                    )
                )
            )
            self.makcu_verification_done.emit(False, port, str(button), detail)
        except (MakcuError, OSError, ValueError) as exc:
            self.makcu_verification_done.emit(False, port, str(button), str(exc))
        finally:
            controller.stop()

    def _apply_makcu_verification_progress(self, message: str) -> None:
        self.aim_makcu_verification_status.setText(message)
        self._set_status(message, "ok")

    def _apply_makcu_verification_result(
        self,
        verified: bool,
        port: str,
        button: str,
        detail: str,
    ) -> None:
        self._makcu_verify_thread = None
        if verified:
            selected_before = self._selected_makcu_button()
            verified_button = int(button)
            if verified_button != selected_before:
                self.aim_makcu_button.setCurrentIndex(verified_button)
            self._makcu_verified_port = port
            self._makcu_verified_button = button
            button_name = MAKCU_BUTTON_LABELS[int(button)]
            self._refresh_makcu_verification_status()
            success_text = f"Verified {button_name} on {port}."
            if detail:
                success_text = f"{success_text} {detail}"
            self._set_status(success_text, "ok")
            self.aim_makcu_verification_status.setText(success_text)
            try:
                save_settings(self.collect(), settings_path())
            except OSError:
                pass
        else:
            self.aim_makcu_verification_status.setText(f"Verification failed: {detail}")
            self._set_status(detail, "error")
            QMessageBox.critical(self, "MAKCU verification failed", detail)
        self._update_aim_state()

    def _parse_int(self, value: str, default: int = 0) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    def _strength_from_slider(self) -> float:
        return self.aim_makcu_strength.value() / STRENGTH_SLIDER_SCALE

    def _smoothing_from_slider(self) -> float:
        return self.aim_makcu_smoothing.value() / SMOOTHING_SLIDER_SCALE

    def _confidence_from_slider(self) -> float:
        return self.confidence.value() / CONFIDENCE_SLIDER_SCALE

    def _set_strength_slider_value(self, text: str) -> None:
        try:
            value = float(str(text).strip())
        except (TypeError, ValueError):
            value = 0.5
        scaled = int(round(max(0.05, min(2.5, value)) * STRENGTH_SLIDER_SCALE))
        self.aim_makcu_strength.setValue(scaled)
        self._sync_strength_label(scaled)

    def _set_smoothing_slider_value(self, text: str) -> None:
        try:
            value = float(str(text).strip())
        except (TypeError, ValueError):
            value = 0.78
        scaled = int(round(max(0.10, min(1.0, value)) * SMOOTHING_SLIDER_SCALE))
        self.aim_makcu_smoothing.setValue(scaled)
        self._sync_smoothing_label(scaled)

    def _set_confidence_slider_value(self, text: str) -> None:
        try:
            value = float(str(text).strip())
        except (TypeError, ValueError):
            value = 0.25
        scaled = int(round(max(0.05, min(0.95, value)) * CONFIDENCE_SLIDER_SCALE))
        self.confidence.setValue(scaled)
        self._sync_confidence_label(scaled)

    def _sync_strength_label(self, _value: int) -> None:
        self.aim_makcu_strength_value.setText(f"{self._strength_from_slider():.2f}")

    def _sync_smoothing_label(self, _value: int) -> None:
        self.aim_makcu_smoothing_value.setText(f"{self._smoothing_from_slider():.2f}")

    def _sync_confidence_label(self, _value: int) -> None:
        self.confidence_value.setText(f"{self._confidence_from_slider():.2f}")

    def _browse_custom_model(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Choose model file",
            str(Path.home()),
            "Model files (*.xml *.onnx);;All files (*)",
        )
        if chosen:
            self.custom_model_path.setText(chosen)

    def _browse_custom_labels(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Choose labels file",
            str(Path.home()),
            "Text files (*.txt);;All files (*)",
        )
        if chosen:
            self.custom_labels_path.setText(chosen)

    def _open_training_script(self) -> None:
        script = (Path(__file__).resolve().parent.parent / "scripts" / "train_fort_model.py")
        if not script.is_file():
            QMessageBox.warning(self, "Script missing", f"Training script not found: {script}")
            return
        self.log_line.emit(f"Open terminal and run: python {script}")
        self._set_status("Training script path copied to output log.", "ok")

    def _open_export_script(self) -> None:
        script = (Path(__file__).resolve().parent.parent / "scripts" / "export_model.py")
        if not script.is_file():
            QMessageBox.warning(self, "Script missing", f"Export script not found: {script}")
            return
        self.log_line.emit(f"Open terminal and run: python {script}")
        self._set_status("Export script path copied to output log.", "ok")

    def _apply_detected_accelerator(self) -> None:
        selected = self.detected_accelerator.currentData()
        if not isinstance(selected, dict):
            self._set_status("Run hardware scan first, then select an accelerator.", "warn")
            return
        ready = bool(selected.get("ready", False))
        if not ready:
            hint = str(selected.get("setup_hint", "Install required runtime and scan again."))
            self._set_status(hint, "warn")
            QMessageBox.information(self, "Runtime setup needed", hint)
            return
        self.backend.setCurrentText(str(selected.get("backend", "openvino")))
        self.device.setCurrentText(str(selected.get("device", "CPU")))
        self.inference_size.setCurrentText(str(selected.get("inference_size", "320")))
        suggested_tier = str(selected.get("tier", MODEL_TIER_MID))
        tier_index = self.model_tier.findData(suggested_tier)
        if tier_index >= 0:
            self.model_tier.setCurrentIndex(tier_index)
            self._refresh_model_preset_options(
                preferred_key=MODEL_TIER_DEFAULT_PRESET.get(suggested_tier, self.model_preset.currentData())
            )
        self._set_status("Applied detected accelerator selection.", "ok")

    def _scan_hardware(self) -> None:
        from detection.hardware import describe, scan_and_recommend

        try:
            profile, plans = scan_and_recommend()
        except Exception as exc:
            QMessageBox.critical(self, "Hardware scan failed", str(exc))
            return

        report = describe(profile, plans)
        pending = [plan for plan in plans if not plan.ready and plan.setup_hint]
        if pending:
            report += "\n\nNot yet usable on this machine:\n" + "\n".join(
                f"  - {plan.accelerator.label}: {plan.setup_hint}" for plan in pending
            )
        self.hardware_report.setPlainText(report)

        self._offer_runtime_download(profile, plans)

        self.detected_accelerator.clear()
        for plan in plans:
            state = "ready" if plan.ready else "needs setup"
            text = f"{plan.accelerator.label} -> {plan.backend}/{plan.device} ({state})"
            backend = str(plan.backend).lower()
            device = str(plan.device).upper()
            if backend == "openvino" and device == "CPU":
                tier = MODEL_TIER_LOW
            elif backend == "onnxruntime" and (
                "TENSORRT" in device or "CUDA" in device or "ROCM" in device or "DML" in device
            ):
                tier = MODEL_TIER_HIGH
            else:
                tier = MODEL_TIER_MID
            self.detected_accelerator.addItem(
                text,
                {
                    "backend": plan.backend,
                    "device": plan.device,
                    "inference_size": plan.inference_size,
                    "ready": plan.ready,
                    "setup_hint": plan.setup_hint,
                    "tier": tier,
                },
            )

        ready = [plan for plan in plans if plan.ready]
        if not ready:
            self._set_status("No usable inference device was found. See scan report for setup hints.", "error")
            return
        best = ready[0]
        for index in range(self.detected_accelerator.count()):
            data = self.detected_accelerator.itemData(index)
            if (
                isinstance(data, dict)
                and data.get("backend") == best.backend
                and data.get("device") == best.device
            ):
                self.detected_accelerator.setCurrentIndex(index)
                break
        self._set_status("Hardware scan complete. Choose a detected accelerator and click Use selection.", "ok")

    def _offer_runtime_download(self, profile, plans) -> None:
        """Enable the download button when a GPU needs a runtime we can fetch."""

        from detection.hardware import AcceleratorKind, Vendor
        from detection.runtime_setup import plan_for

        gpus = [
            item
            for item in profile.of_kind(AcceleratorKind.GPU)
            if item.vendor in (Vendor.AMD, Vendor.NVIDIA)
        ]
        blocked = [
            plan
            for plan in plans
            if not plan.ready and plan.backend == "onnxruntime"
        ]
        if not gpus or not blocked:
            self._runtime_plan = None
            self.install_runtime_button.setEnabled(False)
            self.install_runtime_button.setToolTip(
                "No downloadable GPU runtime is needed on this machine."
            )
            return

        self._runtime_plan = plan_for(gpus[0].vendor.value, profile.system)
        self.install_runtime_button.setEnabled(True)
        self.install_runtime_button.setToolTip(
            f"Download {self._runtime_plan.distribution} for {gpus[0].label}."
        )

    def _install_runtime(self) -> None:
        from detection.runtime_setup import (
            RuntimeSetupError,
            describe,
            ensure_runtime,
            installed_distribution,
        )
        from .settings import settings_path

        plan = getattr(self, "_runtime_plan", None)
        if plan is None:
            return

        summary = describe(plan, installed_distribution())
        if plan.needs_driver:
            summary += (
                "\n\nThe download provides the Python runtime only. The vendor "
                "driver must be installed separately; it needs administrator "
                "rights, so this application will not install it for you."
            )
        confirmed = QMessageBox.question(
            self,
            "Download GPU runtime",
            f"{summary}\n\nDownload it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if confirmed is not QMessageBox.StandardButton.Yes:
            return

        self.install_runtime_button.setEnabled(False)
        self._set_status(f"Downloading {plan.distribution}…", "warn")
        QApplication.processEvents()
        try:
            target = ensure_runtime(plan, settings_path().parent)
        except RuntimeSetupError as exc:
            self._set_status(str(exc), "error")
            QMessageBox.critical(self, "Download failed", str(exc))
            self.install_runtime_button.setEnabled(True)
            return

        self._set_status(f"{plan.distribution} installed into {target}.", "ok")
        QMessageBox.information(
            self,
            "Download complete",
            f"{plan.distribution} is installed.\n\nRun the hardware scan again "
            "to select the GPU.",
        )

    def _set_status(self, text: str, state: str = "") -> None:
        self.status.setText(text)
        self.status.setProperty("state", state)
        # A property used by the style sheet only takes effect after the widget
        # is re-polished.
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text.rstrip())

    def _start(self) -> None:
        settings = self.collect()
        try:
            arguments = settings.detector_arguments()
        except SettingsError as exc:
            QMessageBox.warning(self, "Check the settings", str(exc))
            self._set_status(str(exc), "warn")
            return

        try:
            save_settings(settings, settings_path())
        except OSError:
            # A read-only configuration directory must not block a run.
            pass

        command = launcher_command(settings)
        try:
            self.process = start_detector(command)
        except OSError as exc:
            QMessageBox.critical(self, "Could not start detection", str(exc))
            return

        self.log.clear()
        self._set_status("Detection running.", "ok")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._start_reader()

    def _start_reader(self) -> None:
        import threading

        process = self.process
        if process is None or process.stdout is None:
            return

        def pump() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self.log_line.emit(line)

        self._reader = threading.Thread(target=pump, daemon=True)
        self._reader.start()

    def _stop(self) -> None:
        if self.process is None:
            return
        request_stop(self.process)
        self._set_status("Stopping…", "warn")

    def _poll_process(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            return
        code = self.process.returncode
        self.process = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if code == 0:
            self._set_status("Detection finished.", "ok")
        else:
            self._set_status(f"Detection exited with code {code}.", "error")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._makcu_verify_cancel.set()
        self._makcu_monitor_cancel.set()
        if self.process is not None and self.process.poll() is None:
            request_stop(self.process)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                force_stop(self.process)
        event.accept()


def run_gui(argv: Sequence[str] | None = None) -> int:
    """Start the Qt launcher and return its exit status."""

    app = QApplication(list(argv or sys.argv[:1]))
    app.setApplicationName("ProAim")
    app.setStyleSheet(qt_theme.stylesheet())

    icon_path = Path(__file__).resolve().parent.parent / "assets" / "game-detector.svg"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = LauncherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_gui())
