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

from collections.abc import Sequence
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QIcon, QKeySequence, QShortcut
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
from .precision import (
    DEFAULT_PRECISION_PRESET,
    PRECISION_PRESETS,
    ControllerCandidate,
    candidate_identity,
    candidate_label,
    is_controller_ready_line,
    precision_command,
    precision_preset,
    precision_readiness,
    precision_supported,
    pxn_controllers,
    select_saved_candidate,
    verification_calibration,
)
from .process import (
    find_moonlight_executable,
    force_stop,
    kill_process,
    request_stop,
    start_detector,
    start_external_process,
    start_precision_controller,
)
from .settings import (
    AIM_OUTPUT_MAKCU,
    DEFAULT_MODEL_PRESET,
    MODEL_PRESET_CUSTOM,
    MODEL_PRESET_COCO,
    MODEL_PRESET_COCO_BALANCED,
    MODEL_PRESET_COCO_HIGH,
    MODEL_PRESET_FORT_PLAYER,
    MODEL_PRESET_FORT_PLAYER_BALANCED,
    MODEL_PRESET_FORT_PLAYER_BALANCED_INT8,
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
    resource_root,
    save_settings,
    settings_path,
)


UNIT = qt_theme.UNIT
PROAIM_BUILD_TAG = "2026-08-10-makcu-monitor-v1"
SOURCE_REPOSITORY = "https://github.com/laedorp/game-detector"
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
        MODEL_PRESET_FORT_PLAYER_BALANCED_INT8,
        MODEL_PRESET_FORT_PLAYER,
        MODEL_PRESET_COCO,
    ),
    MODEL_TIER_MID: (
        MODEL_PRESET_FORT_PLAYER_BALANCED,
        MODEL_PRESET_FORT_PLAYER_BALANCED_INT8,
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
    # A faster computer should not silently switch from the clone-specific
    # player class to generic COCO semantics. YOLO11l remains an explicit
    # benchmark option within the high tier.
    MODEL_TIER_HIGH: MODEL_PRESET_FORT_PLAYER_BALANCED,
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
    makcu_monitor_done = Signal()
    precision_output_line = Signal(str)
    precision_reader_done = Signal(object)

    def __init__(self, settings: LauncherSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"ProAim ({PROAIM_BUILD_TAG})")
        self.resize(1080, 760)
        self.setMinimumSize(880, 620)

        self.settings = settings or load_settings(settings_path())
        self.process: subprocess.Popen[str] | None = None
        self._reader: Any | None = None
        self._stop_requested = False
        self._makcu_verify_thread: threading.Thread | None = None
        self._makcu_verify_cancel = threading.Event()
        self._makcu_monitor_thread: threading.Thread | None = None
        self._makcu_monitor_cancel = threading.Event()
        self._makcu_verified_port = self.settings.aim_makcu_verified_port
        self._makcu_verified_button = self.settings.aim_makcu_verified_button
        self._closing = False

        self.precision_process: subprocess.Popen[str] | None = None
        self._precision_reader: threading.Thread | None = None
        self._precision_mode: str | None = None
        self._precision_stop_requested = False
        self._precision_ready = False
        self._precision_pending_identity = ""
        self._precision_recent_output: list[str] = []
        self._precision_candidates: dict[str, ControllerCandidate] = {}
        self._precision_selected_path = self.settings.precision_device_path
        self._precision_verified_identity = (
            self.settings.precision_device_identity
            if self.settings.precision_mapping_verified
            else ""
        )
        self._precision_trigger_rest = (
            self.settings.precision_trigger_rest
            if self.settings.precision_mapping_verified
            else ""
        )
        self._precision_trigger_pressed = (
            self.settings.precision_trigger_pressed
            if self.settings.precision_mapping_verified
            else ""
        )
        self._precision_mapping_verified = bool(
            self.settings.precision_mapping_verified
            and self._precision_verified_identity
        )
        self._precision_supported = precision_supported()

        self._build_interface()
        self._load_from_settings()
        self._refresh_precision_devices(silent=True)

        self._start_key = QShortcut(QKeySequence(Qt.Key.Key_F5), self)
        self._start_key.activated.connect(self._start)
        self._stop_key = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._stop_key.activated.connect(self._stop)

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_process)
        self._poll.start(400)

        self.log_line.connect(self._append_log)
        self.makcu_verification_done.connect(self._apply_makcu_verification_result)
        self.makcu_verification_progress.connect(self._apply_makcu_verification_progress)
        self.makcu_monitor_done.connect(self._finish_makcu_monitor)
        self.precision_output_line.connect(self._handle_precision_output_line)
        self.precision_reader_done.connect(self._precision_reader_finished)

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
        self.stack.addWidget(self._build_precision_section())
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
        for index, text in enumerate(
            ("Capture source", "Detection", "Hardware", "Controller precision")
        ):
            button = QPushButton(text)
            button.setProperty("role", "nav")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, i=index: self._select_section(i))
            layout.addWidget(button)
            self.nav_buttons.append(button)

        layout.addStretch(1)
        about = QPushButton("About & license")
        about.setProperty("role", "nav")
        about.setCursor(Qt.CursorShape.PointingHandCursor)
        about.clicked.connect(self._show_about)
        layout.addWidget(about)
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
        self.open_moonlight_button = QPushButton("Open Moonlight")
        self.open_moonlight_button.clicked.connect(self._open_moonlight)
        _field_row(grid, 0, "Monitor number", self.screen_monitor)
        _field_row(grid, 1, "Capture rate (fps)", self.screen_fps)
        grid.addWidget(self.open_moonlight_button, 1, 2)
        screen_layout.addLayout(grid)

        self.use_screen_region = QCheckBox(
            "Capture a specific desktop rectangle instead of the full monitor"
        )
        self.use_screen_region.toggled.connect(self._update_region_state)
        screen_layout.addWidget(self.use_screen_region)
        region_grid = QGridLayout()
        region_grid.setHorizontalSpacing(UNIT * 2)
        region_grid.setVerticalSpacing(UNIT + 2)
        self.screen_x = QLineEdit()
        self.screen_y = QLineEdit()
        self.screen_width = QLineEdit()
        self.screen_height = QLineEdit()
        self._region_widgets = (
            self.screen_x,
            self.screen_y,
            self.screen_width,
            self.screen_height,
        )
        for column, (label, widget) in enumerate(
            (
                ("X", self.screen_x),
                ("Y", self.screen_y),
                ("Width", self.screen_width),
                ("Height", self.screen_height),
            )
        ):
            region_grid.addWidget(_label(label, "fieldLabel"), 0, column)
            region_grid.addWidget(widget, 1, column)
        screen_layout.addLayout(region_grid)
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
        self.crop_size = QLineEdit()
        self.crop_size.setPlaceholderText("Optional centered square crop")
        self.output_format = QComboBox()
        self.output_format.addItems(("auto", "end2end", "traditional"))

        confidence_row = QWidget()
        confidence_layout = QHBoxLayout(confidence_row)
        confidence_layout.setContentsMargins(0, 0, 0, 0)
        confidence_layout.setSpacing(UNIT)
        confidence_layout.addWidget(self.confidence, 1)
        confidence_layout.addWidget(self.confidence_value)

        _field_row(grid, 0, "Inference size", self.inference_size)
        _field_row(grid, 1, "Confidence", confidence_row)
        _field_row(grid, 2, "IoU threshold", self.iou_threshold)
        _field_row(grid, 3, "Centered crop (px)", self.crop_size)
        _field_row(grid, 4, "Model output format", self.output_format)
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
            "Scanning reports every accelerator present and which runtime "
            "provider is installed, then suggests the fastest path. GPU driver "
            "initialization is verified when detection starts.",
        )
        row = QHBoxLayout()
        row.setSpacing(UNIT)
        scan = QPushButton("Scan hardware")
        scan.setProperty("role", "primary")
        scan.clicked.connect(self._scan_hardware)
        row.addWidget(scan)
        self.install_runtime_button = QPushButton("Show GPU setup instructions")
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
        self.backend.currentTextChanged.connect(self._backend_changed)
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

    def _build_precision_section(self) -> QWidget:
        section = Section(
            "Controller precision",
            "Linux-only manual fine control for a PXN P5 8K. This is an "
            "independent controller worker and does not use detections.",
        )

        card, layout = _card(
            "What this does",
            "While LT is held, the selected curve reshapes only the right-stick "
            "movement you physically produce. Release LT for normal 1:1 input. "
            "Start this worker before opening the Moonlight stream.",
        )
        section.add_card(card)

        card, layout = _card("PXN controller and read-only mapping check")
        row = QHBoxLayout()
        row.setSpacing(UNIT)
        self.precision_device = QComboBox()
        self.precision_device.currentIndexChanged.connect(
            self._precision_selection_changed
        )
        self.precision_refresh_button = QPushButton("Refresh")
        self.precision_refresh_button.clicked.connect(self._refresh_precision_devices)
        self.precision_verify_button = QPushButton("Verify LT + right stick…")
        self.precision_verify_button.clicked.connect(self.verify_precision_mapping)
        row.addWidget(self.precision_device, 1)
        row.addWidget(self.precision_refresh_button)
        row.addWidget(self.precision_verify_button)
        layout.addLayout(row)
        self.precision_device_status = _label("", "subtitle")
        self.precision_device_status.setWordWrap(True)
        self.precision_mapping_status = _label("Mapping not verified", "subtitle")
        self.precision_mapping_status.setWordWrap(True)
        layout.addWidget(self.precision_device_status)
        layout.addWidget(self.precision_mapping_status)
        section.add_card(card)

        card, layout = _card("LT precision strength")
        row = QHBoxLayout()
        row.setSpacing(UNIT)
        row.addWidget(_label("Preset", "fieldLabel"))
        self.precision_preset = QComboBox()
        for preset in PRECISION_PRESETS:
            self.precision_preset.addItem(preset.label, preset.key)
        self.precision_preset.currentIndexChanged.connect(
            self._precision_preset_changed
        )
        row.addWidget(self.precision_preset)
        self.precision_preset_description = _label("", "subtitle")
        self.precision_preset_description.setWordWrap(True)
        row.addWidget(self.precision_preset_description, 1)
        layout.addLayout(row)
        section.add_card(card)

        card, layout = _card("Start before opening Moonlight")
        row = QHBoxLayout()
        row.setSpacing(UNIT)
        self.precision_start_button = QPushButton("Start controller precision")
        self.precision_start_button.setProperty("role", "primary")
        self.precision_start_button.clicked.connect(self.start_precision)
        self.precision_stop_button = QPushButton("Stop precision")
        self.precision_stop_button.clicked.connect(self.stop_precision)
        self.precision_moonlight_button = QPushButton("Open Moonlight")
        self.precision_moonlight_button.clicked.connect(self._open_moonlight)
        self.precision_status = _label("Stopped", "status")
        row.addWidget(self.precision_start_button)
        row.addWidget(self.precision_stop_button)
        row.addWidget(self.precision_status, 1)
        row.addWidget(self.precision_moonlight_button)
        layout.addLayout(row)
        section.add_card(card)

        if not self._precision_supported:
            self.precision_device_status.setText(
                "Controller precision is available only on Linux."
            )
            self.precision_mapping_status.setText("Unavailable on this operating system")
            for widget in (
                self.precision_device,
                self.precision_refresh_button,
                self.precision_verify_button,
                self.precision_preset,
                self.precision_start_button,
                self.precision_stop_button,
                self.precision_moonlight_button,
            ):
                widget.setEnabled(False)

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

    def _show_about(self) -> None:
        license_path = resource_root() / "LICENSE"
        license_location = (
            str(license_path)
            if license_path.is_file()
            else f"{SOURCE_REPOSITORY}/blob/main/LICENSE"
        )
        QMessageBox.about(
            self,
            "About ProAim",
            f"ProAim ({PROAIM_BUILD_TAG})\n\n"
            "Copyright (C) 2026 ProAim contributors.\n"
            "Licensed under AGPL-3.0-or-later. This program is distributed "
            "without any warranty; see the license for details.\n\n"
            f"License: {license_location}\n"
            f"Source: {SOURCE_REPOSITORY}",
        )

    def _choose_source(self, mode: str) -> None:
        for key, box in self._source_boxes.items():
            box.setChecked(key == mode)
        self.screen_card.setVisible(mode == SOURCE_SCREEN)
        self.camera_card.setVisible(mode == SOURCE_CAMERA)
        self.video_card.setVisible(mode == SOURCE_VIDEO)

    def _update_region_state(self) -> None:
        enabled = self.use_screen_region.isChecked()
        for widget in self._region_widgets:
            widget.setEnabled(enabled)

    def _preset_changed(self) -> None:
        key = self.model_preset.currentData()
        custom = key == MODEL_PRESET_CUSTOM
        self.custom_model_path.setEnabled(custom)
        self.custom_labels_path.setEnabled(custom)
        self.custom_model_browse.setEnabled(custom)
        self.custom_labels_browse.setEnabled(custom)
        self.output_format.setEnabled(custom)
        for preset in MODEL_PRESETS:
            if preset.key == key:
                self.preset_note.setText(preset.description)
                if preset.inference_size is not None:
                    self.inference_size.setCurrentText(str(preset.inference_size))
                if preset.bundled:
                    self.output_format.setCurrentText("auto")
                    if hasattr(self, "aim_label"):
                        self.aim_label.setText(
                            "player"
                            if preset.key
                            in (
                                MODEL_PRESET_FORT_PLAYER,
                                MODEL_PRESET_FORT_PLAYER_BALANCED,
                                MODEL_PRESET_FORT_PLAYER_BALANCED_INT8,
                            )
                            else "person"
                        )
                break

    def _model_tier_changed(self) -> None:
        tier = str(self.model_tier.currentData() or MODEL_TIER_MID)
        self._refresh_model_preset_options(
            preferred_key=MODEL_TIER_DEFAULT_PRESET.get(tier)
        )
        if tier == MODEL_TIER_HIGH:
            self.capture_width.setText(HIGH_END_CAPTURE_WIDTH)
            self.capture_height.setText(HIGH_END_CAPTURE_HEIGHT)
            self.capture_fps.setText(HIGH_END_CAPTURE_FPS)
            self.screen_fps.setText(HIGH_END_CAPTURE_FPS)

    def _backend_changed(self, _backend: str = "") -> None:
        if not hasattr(self, "model_preset"):
            return
        self._refresh_model_preset_options(
            preferred_key=str(self.model_preset.currentData() or "")
        )

    def _refresh_model_preset_options(self, *, preferred_key: str | None = None) -> None:
        tier = str(self.model_tier.currentData() or MODEL_TIER_MID)
        allowed = set(MODEL_TIER_PRESET_KEYS.get(tier, MODEL_TIER_PRESET_KEYS[MODEL_TIER_MID]))
        allowed.add(MODEL_PRESET_CUSTOM)
        if self.backend.currentText().strip().lower() == "onnxruntime":
            allowed.discard(MODEL_PRESET_FORT_PLAYER_BALANCED_INT8)
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
        self.use_screen_region.setChecked(s.use_screen_region)
        self.screen_x.setText(s.screen_x)
        self.screen_y.setText(s.screen_y)
        self.screen_width.setText(s.screen_width)
        self.screen_height.setText(s.screen_height)
        self._update_region_state()
        self.camera_index.setText(s.camera_index)
        self.capture_width.setText(s.capture_width)
        self.capture_height.setText(s.capture_height)
        self.capture_fps.setText(s.capture_fps)
        self.video_path.setText(s.video_path)
        self._set_confidence_slider_value(s.confidence)
        self.iou_threshold.setText(s.iou_threshold)
        self.inference_size.setCurrentText(s.inference_size)
        self.crop_size.setText(s.crop_size)
        self.output_format.setCurrentText(s.output_format)
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
        self.aim_makcu_port.blockSignals(True)
        self.aim_makcu_button.blockSignals(True)
        self.aim_makcu_port.setText(s.aim_makcu_port)
        makcu_button = min(max(self._parse_int(s.aim_makcu_button, default=1), 0), 4)
        self.aim_makcu_button.setCurrentIndex(makcu_button)
        self.aim_makcu_port.blockSignals(False)
        self.aim_makcu_button.blockSignals(False)
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
        if (
            preferred_model != MODEL_PRESET_CUSTOM
            and preferred_model not in MODEL_TIER_PRESET_KEYS.get(tier_value, ())
        ):
            preferred_model = MODEL_TIER_DEFAULT_PRESET.get(tier_value, s.model_preset)
        self._refresh_model_preset_options(preferred_key=preferred_model)
        self._preset_changed()
        if preferred_model == MODEL_PRESET_CUSTOM:
            # Tier initialization may temporarily select a bundled preset. Put
            # the custom decoder and size back after the final preset is active.
            self.inference_size.setCurrentText(s.inference_size)
            self.output_format.setCurrentText(s.output_format)
        if tier_value == MODEL_TIER_HIGH:
            self.capture_width.setText(HIGH_END_CAPTURE_WIDTH)
            self.capture_height.setText(HIGH_END_CAPTURE_HEIGHT)
            self.capture_fps.setText(HIGH_END_CAPTURE_FPS)
            self.screen_fps.setText(HIGH_END_CAPTURE_FPS)
        self._choose_source(s.source_mode)
        precision_index = self.precision_preset.findData(s.precision_preset)
        if precision_index < 0:
            precision_index = self.precision_preset.findData(DEFAULT_PRECISION_PRESET)
        self.precision_preset.setCurrentIndex(max(precision_index, 0))
        self._precision_preset_changed()
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
        s.use_screen_region = self.use_screen_region.isChecked()
        s.screen_x = self.screen_x.text()
        s.screen_y = self.screen_y.text()
        s.screen_width = self.screen_width.text()
        s.screen_height = self.screen_height.text()
        s.camera_index = self.camera_index.text()
        s.capture_width = self.capture_width.text()
        s.capture_height = self.capture_height.text()
        s.capture_fps = self.capture_fps.text()
        s.video_path = self.video_path.text()
        s.confidence = f"{self._confidence_from_slider():.2f}"
        s.iou_threshold = self.iou_threshold.text()
        s.inference_size = self.inference_size.currentText()
        s.crop_size = self.crop_size.text().strip()
        s.output_format = self.output_format.currentText()
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
        s.aim_makcu_verified_port = (
            self._makcu_verified_port if self._makcu_verification_matches() else ""
        )
        s.aim_makcu_verified_button = (
            self._makcu_verified_button if self._makcu_verification_matches() else ""
        )
        s.ignore_self = self.ignore_self.isChecked()
        s.self_position = str(self.self_position.currentData() or SELF_POSITION_LEFT)
        precision_candidate = self._selected_precision_candidate()
        precision_path = (
            str(precision_candidate.path)
            if precision_candidate is not None
            else self._precision_selected_path
        )
        precision_verified = bool(
            precision_candidate is not None
            and self._precision_mapping_verified
            and self._precision_verified_identity
            and candidate_identity(precision_candidate) == self._precision_verified_identity
        )
        if precision_candidate is None and self._precision_verified_identity:
            # Keep a device-bound record while the same controller is simply
            # unplugged; selecting a different device clears it below.
            precision_verified = self._precision_mapping_verified
        s.precision_device_path = precision_path
        s.precision_device_identity = (
            self._precision_verified_identity if precision_verified else ""
        )
        s.precision_mapping_verified = precision_verified
        s.precision_preset = str(
            self.precision_preset.currentData() or DEFAULT_PRECISION_PRESET
        )
        s.precision_trigger_rest = (
            self._precision_trigger_rest if precision_verified else ""
        )
        s.precision_trigger_pressed = (
            self._precision_trigger_pressed if precision_verified else ""
        )
        return s

    # -- controller precision -------------------------------------------
    def _selected_precision_candidate(self) -> ControllerCandidate | None:
        candidate = self.precision_device.currentData()
        return candidate if isinstance(candidate, ControllerCandidate) else None

    def _refresh_precision_devices(self, _checked: bool = False, *, silent: bool = False) -> None:
        if not self._precision_supported:
            return
        process = self.precision_process
        if process is not None and process.poll() is None:
            if not silent:
                self.precision_status.setText(
                    "Stop controller precision before refreshing devices."
                )
            return
        try:
            candidates = pxn_controllers()
        except OSError as exc:
            candidates = ()
            self.precision_device_status.setText(f"Could not scan controllers: {exc}")

        self._precision_candidates = {
            candidate_label(candidate): candidate for candidate in candidates
        }
        selected = select_saved_candidate(candidates, self._precision_selected_path)
        self.precision_device.blockSignals(True)
        self.precision_device.clear()
        if not candidates:
            self.precision_device.addItem("No PXN P5 8K found", None)
        else:
            for candidate in candidates:
                self.precision_device.addItem(candidate_label(candidate), candidate)
        selected_index = self.precision_device.findData(selected)
        if selected_index >= 0:
            self.precision_device.setCurrentIndex(selected_index)
        elif candidates:
            self.precision_device.setCurrentIndex(-1)
        self.precision_device.blockSignals(False)

        if selected is None:
            if candidates:
                self.precision_device_status.setText(
                    "Choose the PXN P5 8K you want to verify."
                )
            else:
                self.precision_device_status.setText(
                    "PXN P5 8K not found. Connect it by USB, then press Refresh."
                )
            if self._precision_verified_identity:
                self.precision_mapping_status.setText(
                    "A verified controller is saved; reconnect that same device to continue."
                )
            else:
                self.precision_mapping_status.setText("Mapping not verified")
        else:
            self._precision_selected_path = str(selected.path)
            identity_matches = bool(
                self._precision_mapping_verified
                and self._precision_verified_identity
                and candidate_identity(selected) == self._precision_verified_identity
            )
            if not identity_matches:
                self._clear_precision_verification()
            self.precision_device_status.setText(
                f"Found {candidate_label(selected)}"
            )
            self.precision_mapping_status.setText(
                "Mapping verified for this controller — ready to start."
                if identity_matches
                else "Mapping not verified. Run the read-only LT + right-stick check once."
            )
        self._update_precision_controls()

    def _clear_precision_verification(self) -> None:
        self._precision_verified_identity = ""
        self._precision_trigger_rest = ""
        self._precision_trigger_pressed = ""
        self._precision_mapping_verified = False

    def _precision_selection_changed(self, _index: int = -1) -> None:
        candidate = self._selected_precision_candidate()
        if candidate is None:
            self._update_precision_controls()
            return
        new_identity = candidate_identity(candidate)
        self._precision_selected_path = str(candidate.path)
        if new_identity != self._precision_verified_identity:
            self._clear_precision_verification()
            self.precision_mapping_status.setText(
                "Mapping not verified. Run the read-only LT + right-stick check once."
            )
        else:
            self._precision_mapping_verified = True
            self.precision_mapping_status.setText(
                "Mapping verified for this controller — ready to start."
            )
        self.precision_device_status.setText(
            f"Selected {candidate_label(candidate)}"
        )
        self._update_precision_controls()

    def _precision_preset_changed(self, _index: int = -1) -> None:
        key = str(self.precision_preset.currentData() or DEFAULT_PRECISION_PRESET)
        self.precision_preset_description.setText(precision_preset(key).description)

    def _update_precision_controls(self) -> None:
        if not self._precision_supported:
            return
        process = self.precision_process
        busy = process is not None and process.poll() is None
        candidate = self._selected_precision_candidate()
        verified = bool(
            candidate is not None
            and self._precision_mapping_verified
            and self._precision_verified_identity
            and candidate_identity(candidate) == self._precision_verified_identity
        )
        self.precision_device.setEnabled(bool(self._precision_candidates) and not busy)
        self.precision_refresh_button.setEnabled(not busy)
        self.precision_verify_button.setEnabled(
            candidate is not None and candidate.readable and not busy
        )
        self.precision_preset.setEnabled(not busy)
        self.precision_start_button.setEnabled(verified and not busy)
        self.precision_stop_button.setEnabled(busy and not self._precision_stop_requested)
        self.precision_moonlight_button.setEnabled(
            busy
            and self._precision_mode == "run"
            and self._precision_ready
            and not self._precision_stop_requested
        )

    def verify_precision_mapping(self) -> None:
        if not self._precision_supported:
            QMessageBox.critical(
                self, "Linux required", "Controller precision is available only on Linux."
            )
            return
        candidate = self._selected_precision_candidate()
        if candidate is None:
            QMessageBox.warning(
                self,
                "PXN controller not selected",
                "Connect the PXN P5 8K, press Refresh, and select it first.",
            )
            return
        if not candidate.readable:
            QMessageBox.critical(
                self,
                "Controller permission needed",
                "This desktop session cannot read the PXN controller. Reconnect it "
                "while signed in, then press Refresh.",
            )
            return
        try:
            command = precision_command(
                candidate, mode="verify", verification_seconds=4.0
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot verify mapping", str(exc))
            return

        QMessageBox.information(
            self,
            "Read-only mapping check",
            "This check only observes your physical controller. After closing this "
            "message, repeatedly squeeze and release LT, then move only the right "
            "stick in full circles when prompted. Keep all other controls still.",
        )
        self._clear_precision_verification()
        self.precision_mapping_status.setText(
            "Checking LT and right-stick mapping (read-only)…"
        )
        self._start_precision_child(
            command,
            mode="verify",
            pending_identity=candidate_identity(candidate),
        )

    def start_precision(self) -> None:
        if not self._precision_supported:
            return
        if self.precision_process is not None and self.precision_process.poll() is None:
            return
        candidate = self._selected_precision_candidate()
        if candidate is None:
            QMessageBox.warning(
                self,
                "PXN controller not selected",
                "Connect the PXN P5 8K and press Refresh.",
            )
            return
        identity = candidate_identity(candidate)
        if not (
            self._precision_mapping_verified
            and self._precision_verified_identity
            and identity == self._precision_verified_identity
        ):
            QMessageBox.warning(
                self,
                "Verify the controller mapping first",
                "Run the read-only LT + right-stick check for this controller before starting.",
            )
            return
        readiness = precision_readiness(candidate)
        if not readiness.ready:
            detail = f"{readiness.summary}\n\n{readiness.action}".strip()
            self.precision_status.setText(readiness.summary)
            QMessageBox.critical(self, "Controller precision is not ready", detail)
            return
        try:
            trigger_rest = (
                int(self._precision_trigger_rest)
                if self._precision_trigger_rest.strip()
                else None
            )
            trigger_pressed = (
                int(self._precision_trigger_pressed)
                if self._precision_trigger_pressed.strip()
                else None
            )
            command = precision_command(
                candidate,
                mode="run",
                preset_key=str(
                    self.precision_preset.currentData() or DEFAULT_PRECISION_PRESET
                ),
                parent_pid=os.getpid(),
                trigger_rest=trigger_rest,
                trigger_pressed=trigger_pressed,
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Cannot start controller precision", str(exc))
            return
        self._start_precision_child(command, mode="run", pending_identity=identity)

    def _start_precision_child(
        self,
        command: list[str],
        *,
        mode: str,
        pending_identity: str,
    ) -> bool:
        try:
            process = start_precision_controller(command)
        except OSError as exc:
            self.precision_status.setText("Could not start")
            QMessageBox.critical(
                self,
                "Could not start controller helper",
                f"The controller helper could not be started.\n\n{exc}",
            )
            return False
        self.precision_process = process
        self._precision_mode = mode
        self._precision_pending_identity = pending_identity
        self._precision_stop_requested = False
        self._precision_ready = False
        self._precision_recent_output = []
        self.precision_status.setText(
            "Checking mapping…" if mode == "verify" else "Starting…"
        )
        self._append_log(
            "Checking controller mapping…"
            if mode == "verify"
            else "Starting controller precision…"
        )

        def pump() -> None:
            stream = process.stdout
            if stream is not None:
                try:
                    for line in stream:
                        self.precision_output_line.emit(line)
                except (OSError, ValueError) as exc:
                    self.precision_output_line.emit(
                        f"Controller log reader stopped: {exc}\n"
                    )
                finally:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.precision_reader_done.emit(process)

        self._precision_reader = threading.Thread(
            target=pump, name="controller-precision-log-reader", daemon=True
        )
        self._precision_reader.start()
        self._update_precision_controls()
        return True

    def _handle_precision_output_line(self, line: str) -> None:
        if is_controller_ready_line(line):
            process = self.precision_process
            if (
                process is not None
                and process.poll() is None
                and self._precision_mode == "run"
                and not self._precision_stop_requested
            ):
                self._precision_ready = True
                self.precision_status.setText(
                    "Active — hold LT for fine manual control"
                )
                self._append_log("[Controller] Virtual input is ready.")
                self._update_precision_controls()
            return
        self._precision_recent_output.append(line)
        if len(self._precision_recent_output) > 80:
            del self._precision_recent_output[:-80]
        self._append_log(f"[Controller] {line.rstrip()}")

    def _precision_reader_finished(self, process: object) -> None:
        if process is not self.precision_process:
            return
        self._precision_reader = None
        self._poll_precision_process()

    def _poll_precision_process(self) -> None:
        process = self.precision_process
        if process is None:
            return
        return_code = process.poll()
        if return_code is None:
            return
        self._finish_precision_process(process, return_code)

    def _finish_precision_process(
        self, process: subprocess.Popen[str], return_code: int
    ) -> None:
        if self.precision_process is not process:
            return
        mode = self._precision_mode
        pending_identity = self._precision_pending_identity
        stop_requested = self._precision_stop_requested
        became_ready = self._precision_ready
        self.precision_process = None
        self._precision_reader = None
        self._precision_mode = None
        self._precision_pending_identity = ""
        self._precision_stop_requested = False
        self._precision_ready = False

        if mode == "verify":
            candidate = self._selected_precision_candidate()
            calibration = verification_calibration("".join(self._precision_recent_output))
            same_device = bool(
                candidate is not None
                and pending_identity
                and candidate_identity(candidate) == pending_identity
            )
            if return_code == 0 and same_device and calibration is not None and not stop_requested:
                self._precision_verified_identity = pending_identity
                self._precision_trigger_rest = str(calibration[0])
                self._precision_trigger_pressed = str(calibration[1])
                self._precision_mapping_verified = True
                self.precision_mapping_status.setText(
                    "Mapping verified for this controller — ready to start."
                )
                self.precision_status.setText("Mapping verified")
                if not self._closing:
                    QMessageBox.information(
                        self,
                        "Controller mapping verified",
                        "LT and the right-stick axes matched the expected PXN mapping.",
                    )
            else:
                self._clear_precision_verification()
                self.precision_mapping_status.setText(
                    "Mapping not verified. Repeat the check and move only the requested controls."
                )
                self.precision_status.setText(
                    "Mapping check stopped" if stop_requested else "Mapping check failed"
                )
                if not stop_requested and not self._closing:
                    QMessageBox.warning(
                        self,
                        "Mapping not verified",
                        "The expected LT and right-stick axes were not observed.",
                    )
        elif stop_requested and return_code in (0, 130, -2, -15):
            self.precision_status.setText("Stopped")
        elif not became_ready:
            self.precision_status.setText("Could not start")
            if not self._closing:
                QMessageBox.critical(
                    self,
                    "Controller precision could not start",
                    "The controller was not reported active. Check the connection and permissions.",
                )
        elif return_code == 0:
            self.precision_status.setText("Stopped")
        else:
            self.precision_status.setText(f"Stopped with error ({return_code})")

        try:
            save_settings(self.collect(), settings_path())
        except OSError:
            pass
        self._update_precision_controls()

    def stop_precision(self) -> None:
        process = self.precision_process
        if process is None or process.poll() is not None or self._precision_stop_requested:
            return
        self._precision_stop_requested = True
        self.precision_status.setText("Stopping…")
        self._update_precision_controls()
        request_stop(process)
        QTimer.singleShot(
            3000, lambda current=process: self._force_stop_precision_if_current(current)
        )
        QTimer.singleShot(
            6000, lambda current=process: self._kill_precision_if_current(current)
        )

    def _force_stop_precision_if_current(
        self, process: subprocess.Popen[str]
    ) -> None:
        if self.precision_process is process and process.poll() is None:
            self._append_log("Controller helper did not exit; terminating it.")
            force_stop(process)

    def _kill_precision_if_current(self, process: subprocess.Popen[str]) -> None:
        if self.precision_process is process and process.poll() is None:
            self._append_log("Controller helper still did not exit; forcing shutdown.")
            kill_process(process)

    # -- actions ---------------------------------------------------------
    def _browse_video(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose a video file", "", "Video files (*.mp4 *.mkv *.avi *.mov);;All files (*)"
        )
        if chosen:
            self.video_path.setText(chosen)

    def _open_moonlight(self) -> None:
        executable = find_moonlight_executable()
        if not executable:
            QMessageBox.information(
                self,
                "Moonlight was not found",
                "Install Moonlight or open it from your application menu. ProAim "
                "captures the stream after it is running.",
            )
            return
        try:
            start_external_process([executable])
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Could not open Moonlight",
                f"Moonlight was found but could not be opened.\n\n{exc}",
            )
            return
        self._set_status(
            "Moonlight opened — start your stream, then start detection.", "ok"
        )
        self._append_log("Opened Moonlight. Start the stream before detection.")

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
        selected = self.aim_makcu_button.currentData()
        return 1 if selected is None else int(selected)

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
        if not self._makcu_verification_matches():
            # A verification record is bound to one exact board path and
            # button.  Do not retain it after either selection changes.
            self._makcu_verified_port = ""
            self._makcu_verified_button = ""
        self._refresh_makcu_verification_status()

    def _update_aim_state(self) -> None:
        enabled = self.aim.isChecked()
        if enabled:
            # Detection output must never run without the third-person guard.
            # Keep the relationship visible in the UI instead of silently
            # adding a hidden command-line option.
            self.ignore_self.setChecked(True)
        self.ignore_self.setEnabled(not enabled)
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
            self.makcu_monitor_done.emit()

    def _finish_makcu_monitor(self) -> None:
        """Finish monitor UI work on Qt's main thread."""

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
        self._makcu_verified_port = ""
        self._makcu_verified_button = ""
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
        baseline_observed = False
        pressed = False
        expected_mask = 1 << button
        try:
            controller.start(output_loop=False)
            self.makcu_verification_progress.emit(
                f"Listening for {button_name}… click the physical mouse connected through MAKCU."
            )
            while time.monotonic() < deadline and not self._makcu_verify_cancel.is_set():
                mask = controller.poll_button_mask()
                if not baseline_observed:
                    if mask:
                        detail = (
                            "Release every mouse button before verification, then try again. "
                            f"MAKCU initially reported mask 0x{mask:02X}."
                        )
                        self.makcu_verification_done.emit(
                            False, port, str(button), detail
                        )
                        return
                    baseline_observed = True
                    self.makcu_verification_progress.emit(
                        f"Ready for one {button_name} click — press and release it now."
                    )
                elif not pressed and mask:
                    saw_any_button_report = True
                    last_mask = mask
                    if mask != expected_mask:
                        detail = (
                            f"Expected only {button_name} (mask 0x{expected_mask:02X}), "
                            f"but MAKCU reported mask 0x{mask:02X}. No button was verified."
                        )
                        self.makcu_verification_done.emit(False, port, str(button), detail)
                        return
                    self.makcu_verification_progress.emit(
                        f"Detected {button_name} (mask 0x{mask:02X}). Release it to finish verification…"
                    )
                    pressed = True
                elif pressed and mask == 0:
                    detail = (
                        f"Detected a complete {button_name} press and release "
                        f"(mask 0x{last_mask:02X})."
                    )
                    self.makcu_verification_done.emit(
                        True,
                        port,
                        str(button),
                        detail,
                    )
                    return
                elif pressed and mask != expected_mask:
                    detail = (
                        f"The button mask changed from the expected {button_name} "
                        f"mask 0x{expected_mask:02X} to 0x{mask:02X} before release. "
                        "No button was verified."
                    )
                    self.makcu_verification_done.emit(
                        False,
                        port,
                        str(button),
                        detail,
                    )
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
        current_preset = str(self.model_preset.currentData() or "")
        precision = str(selected.get("precision", "")).strip().lower()
        self.backend.setCurrentText(str(selected.get("backend", "openvino")))
        self.device.setCurrentText(str(selected.get("device", "CPU")))
        suggested_tier = str(selected.get("tier", MODEL_TIER_MID))
        tier_index = self.model_tier.findData(suggested_tier)
        if tier_index >= 0:
            # Hardware choice controls the runtime, not the detector's class
            # semantics. Keep a player detector, COCO detector, or custom model
            # within the same family when moving between performance tiers.
            preferred_preset = self._compatible_preset_for_tier(
                current_preset, suggested_tier
            )
            if (
                self.backend.currentText().strip().lower() == "openvino"
                and self.device.currentText().strip().upper() == "CPU"
                and precision == "int8"
                and current_preset
                in (
                    MODEL_PRESET_FORT_PLAYER,
                    MODEL_PRESET_FORT_PLAYER_BALANCED,
                    MODEL_PRESET_FORT_PLAYER_BALANCED_INT8,
                )
            ):
                preferred_preset = MODEL_PRESET_FORT_PLAYER_BALANCED_INT8
            self.model_tier.blockSignals(True)
            self.model_tier.setCurrentIndex(tier_index)
            self.model_tier.blockSignals(False)
            self._refresh_model_preset_options(
                preferred_key=preferred_preset
            )
            if suggested_tier == MODEL_TIER_HIGH:
                self.capture_width.setText(HIGH_END_CAPTURE_WIDTH)
                self.capture_height.setText(HIGH_END_CAPTURE_HEIGHT)
                self.capture_fps.setText(HIGH_END_CAPTURE_FPS)
                self.screen_fps.setText(HIGH_END_CAPTURE_FPS)
        self._set_status("Applied detected accelerator selection.", "ok")

    def _compatible_preset_for_tier(self, current: str, tier: str) -> str:
        allowed = set(
            MODEL_TIER_PRESET_KEYS.get(tier, MODEL_TIER_PRESET_KEYS[MODEL_TIER_MID])
        )
        allowed.add(MODEL_PRESET_CUSTOM)
        if current in allowed:
            return current
        if current in (MODEL_PRESET_FORT_PLAYER, MODEL_PRESET_FORT_PLAYER_BALANCED):
            return (
                MODEL_PRESET_FORT_PLAYER
                if tier == MODEL_TIER_LOW
                else MODEL_PRESET_FORT_PLAYER_BALANCED
            )
        if current in (
            MODEL_PRESET_COCO,
            MODEL_PRESET_COCO_BALANCED,
            MODEL_PRESET_COCO_HIGH,
        ):
            return MODEL_PRESET_COCO if tier == MODEL_TIER_LOW else MODEL_PRESET_COCO_BALANCED
        return MODEL_TIER_DEFAULT_PRESET.get(tier, DEFAULT_MODEL_PRESET)

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
            if plan.ready and plan.backend == "onnxruntime":
                state = "provider found; verify at start"
            else:
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
                    "precision": plan.precision,
                    "ready": plan.ready,
                    "setup_hint": plan.setup_hint,
                    "tier": tier,
                },
            )

        ready = [plan for plan in plans if plan.ready]
        if not ready:
            self._set_status("No inference runtime was found. See scan report for setup hints.", "error")
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
        """Offer setup guidance when a detected GPU needs another runtime."""

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
                "No additional GPU runtime is needed on this machine."
            )
            return

        self._runtime_plan = plan_for(gpus[0].vendor.value, profile.system)
        self.install_runtime_button.setEnabled(True)
        self.install_runtime_button.setToolTip(
            f"Show setup instructions for {self._runtime_plan.distribution}."
        )

    def _install_runtime(self) -> None:
        from detection.runtime_setup import describe, installed_distribution

        plan = getattr(self, "_runtime_plan", None)
        if plan is None:
            return

        summary = describe(plan, installed_distribution())
        if plan.needs_driver:
            summary += (
                "\n\nThe vendor driver must be installed separately; it needs "
                "administrator rights, so this application will not install it for you."
            )
        summary += (
            "\n\nUse the matching prebuilt ProAim release for the easiest setup. "
            "Source developers can install the named package in their virtual "
            "environment, then reopen ProAim and scan again."
        )
        QMessageBox.information(
            self,
            "GPU runtime setup",
            summary,
        )
        self._set_status("GPU runtime setup instructions shown.", "ok")

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
        if self.process is not None and self.process.poll() is None:
            return
        if self._makcu_verify_thread is not None and self._makcu_verify_thread.is_alive():
            QMessageBox.information(
                self,
                "MAKCU verification in progress",
                "Finish the MAKCU button check before starting detection.",
            )
            return
        settings = self.collect()
        if (
            settings.source_mode == SOURCE_SCREEN
            and os.name == "posix"
            and os.environ.get("XDG_SESSION_TYPE", "").strip().lower() == "wayland"
        ):
            message = (
                "Moonlight screen capture needs an X11/Xorg desktop session in this version.\n\n"
                "Log out, choose an Xorg/X11 session, then reopen Moonlight and ProAim. "
                "Cameras and video files still work on Wayland."
            )
            self._set_status("Moonlight capture needs X11/Xorg.", "error")
            QMessageBox.critical(self, "X11/Xorg session required", message)
            return
        if settings.aim:
            if not settings.aim_label.strip():
                QMessageBox.warning(
                    self,
                    "Choose a target label",
                    "MAKCU aim requires an explicit target label such as player or person.",
                )
                self._set_status("Choose a target label before starting.", "warn")
                return
            if not self._makcu_verification_matches():
                QMessageBox.warning(
                    self,
                    "Verify MAKCU activation first",
                    "The selected MAKCU board and activation button must pass a complete "
                    "physical press-and-release check before aim can be enabled.",
                )
                self._set_status("MAKCU activation is not verified.", "warn")
                return
        try:
            settings.detector_arguments()
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
        self._stop_requested = False
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
        if self.process is None or self.process.poll() is not None or self._stop_requested:
            return
        process = self.process
        self._stop_requested = True
        request_stop(process)
        self._set_status("Stopping…", "warn")
        self.stop_button.setEnabled(False)
        QTimer.singleShot(3000, lambda current=process: self._force_stop_if_current(current))
        QTimer.singleShot(6000, lambda current=process: self._kill_if_current(current))

    def _force_stop_if_current(self, process: subprocess.Popen[str]) -> None:
        if self.process is process and process.poll() is None:
            self._append_log("Detector did not exit; terminating it.")
            force_stop(process)

    def _kill_if_current(self, process: subprocess.Popen[str]) -> None:
        if self.process is process and process.poll() is None:
            self._append_log("Detector still did not exit; forcing shutdown.")
            kill_process(process)

    def _poll_process(self) -> None:
        self._poll_precision_process()
        if self.process is None:
            return
        if self.process.poll() is None:
            return
        code = self.process.returncode
        self.process = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self._stop_requested and code in (0, 130, -2, -15):
            self._set_status("Stopped.", "ok")
        elif code == 0:
            self._set_status("Detection finished.", "ok")
        else:
            self._set_status(f"Detection exited with code {code}.", "error")
        self._stop_requested = False

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._closing = True
        self._makcu_verify_cancel.set()
        self._makcu_monitor_cancel.set()
        running = [
            process
            for process in (self.process, self.precision_process)
            if process is not None and process.poll() is None
        ]
        for process in running:
            request_stop(process)
        deadline = time.monotonic() + 3.0
        while any(process.poll() is None for process in running) and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.02)
        remaining = [process for process in running if process.poll() is None]
        for process in remaining:
            force_stop(process)
        deadline = time.monotonic() + 2.0
        while any(process.poll() is None for process in remaining) and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.02)
        for process in remaining:
            if process.poll() is None:
                kill_process(process)
        for thread in (
            self._makcu_verify_thread,
            self._makcu_monitor_thread,
            self._reader,
            self._precision_reader,
        ):
            if (
                isinstance(thread, threading.Thread)
                and thread.is_alive()
                and thread is not threading.current_thread()
            ):
                thread.join(timeout=0.5)
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
