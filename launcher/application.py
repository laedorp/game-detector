"""Tkinter desktop application for configuring and running Game Detector."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import threading

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .process import (
    external_process_environment,
    force_stop,
    kill_process,
    request_stop,
    start_detector,
    start_precision_controller,
)
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
from .settings import (
    DEFAULT_MODEL_PRESET,
    LauncherSettings,
    MODEL_PRESETS,
    MODEL_PRESET_CUSTOM,
    SETTINGS_VERSION,
    SELF_POSITION_CENTER,
    SELF_POSITION_LEFT,
    SELF_POSITION_RIGHT,
    SOURCE_CAMERA,
    SOURCE_SCREEN,
    SOURCE_VIDEO,
    SettingsError,
    launcher_command,
    load_settings,
    model_preset,
    model_preset_paths,
    save_settings,
)


APP_NAME = "Game Detector"
POLL_INTERVAL_MS = 100
LOG_LINE_LIMIT = 5000
SELF_POSITION_LABELS = {
    SELF_POSITION_LEFT: "Left of center (common)",
    SELF_POSITION_CENTER: "Center",
    SELF_POSITION_RIGHT: "Right of center",
}
SELF_POSITION_VALUES = {label: value for value, label in SELF_POSITION_LABELS.items()}
PRECISION_PRESET_LABELS = {preset.key: preset.label for preset in PRECISION_PRESETS}
PRECISION_PRESET_VALUES = {preset.label: preset.key for preset in PRECISION_PRESETS}
MODEL_PRESET_LABELS = {preset.key: preset.label for preset in MODEL_PRESETS}
MODEL_PRESET_VALUES = {preset.label: preset.key for preset in MODEL_PRESETS}


class DetectorLauncher:
    """A responsive Tk front end that runs detection in a child process."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.process: subprocess.Popen[str] | None = None
        self._process_output: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self.precision_process: subprocess.Popen[str] | None = None
        self._precision_process_output: queue.Queue[str | None] = queue.Queue()
        self._precision_reader: threading.Thread | None = None
        self._precision_mode: str | None = None
        self._precision_stop_requested = False
        self._precision_ready = False
        self._precision_guidance_shown = False
        self._precision_pending_identity = ""
        self._precision_recent_output: list[str] = []
        self._closing = False
        self._stop_requested = False
        self._log_lines = 0

        self._loaded = load_settings()
        self._create_variables(self._loaded)
        self._configure_window()
        self._build_interface()
        self._update_source_panel()
        self._update_region_state()
        self._update_draw_state()
        self._update_self_filter_state()
        self._refresh_precision_devices(silent=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<F5>", self._start_shortcut)
        self.root.bind("<Escape>", self._stop_shortcut)
        self.root.after(POLL_INTERVAL_MS, self._poll_process)

    def _create_variables(self, settings: LauncherSettings) -> None:
        self.source_mode = tk.StringVar(value=settings.source_mode)
        self.video_path = tk.StringVar(value=settings.video_path)
        self.camera_index = tk.StringVar(value=settings.camera_index)
        self.capture_width = tk.StringVar(value=settings.capture_width)
        self.capture_height = tk.StringVar(value=settings.capture_height)
        self.capture_fps = tk.StringVar(value=settings.capture_fps)
        self.screen_monitor = tk.StringVar(value=settings.screen_monitor)
        self.use_screen_region = tk.BooleanVar(value=settings.use_screen_region)
        self.screen_x = tk.StringVar(value=settings.screen_x)
        self.screen_y = tk.StringVar(value=settings.screen_y)
        self.screen_width = tk.StringVar(value=settings.screen_width)
        self.screen_height = tk.StringVar(value=settings.screen_height)
        self.screen_fps = tk.StringVar(value=settings.screen_fps)
        selected_model_preset = model_preset(settings.model_preset)
        self.model_preset = tk.StringVar(value=selected_model_preset.label)
        self.model_preset_description = tk.StringVar(
            value=selected_model_preset.description
        )
        self.model_path = tk.StringVar(value=settings.model_path)
        self.labels_path = tk.StringVar(value=settings.labels_path)
        self._active_model_preset = selected_model_preset.key
        self._custom_model_path = (
            settings.model_path
            if selected_model_preset.key == MODEL_PRESET_CUSTOM
            else ""
        )
        self._custom_labels_path = (
            settings.labels_path
            if selected_model_preset.key == MODEL_PRESET_CUSTOM
            else ""
        )
        self._custom_output_format = settings.output_format
        self.device = tk.StringVar(value=settings.device)
        self.inference_size = tk.StringVar(value=settings.inference_size)
        self.crop_size = tk.StringVar(value=settings.crop_size)
        self.confidence = tk.StringVar(value=settings.confidence)
        self.iou_threshold = tk.StringVar(value=settings.iou_threshold)
        self.output_format = tk.StringVar(value=settings.output_format)
        self.ignore_self = tk.BooleanVar(value=settings.ignore_self)
        self.self_position = tk.StringVar(
            value=SELF_POSITION_LABELS.get(
                settings.self_position,
                SELF_POSITION_LABELS[SELF_POSITION_LEFT],
            )
        )
        self.preview = tk.BooleanVar(value=settings.preview)
        self.draw = tk.BooleanVar(value=settings.draw)
        self.status = tk.StringVar(value="Ready")
        preset = precision_preset(settings.precision_preset)
        self.precision_preset = tk.StringVar(value=preset.label)
        self.precision_preset_description = tk.StringVar(value=preset.description)
        self.precision_device_choice = tk.StringVar(value="")
        self.precision_device_status = tk.StringVar(value="Looking for PXN P5 8K…")
        self.precision_mapping_status = tk.StringVar(value="Mapping not verified")
        self.precision_status = tk.StringVar(value="Stopped")
        self.precision_mapping_verified = tk.BooleanVar(
            value=settings.precision_mapping_verified
        )
        self._precision_candidates: dict[str, ControllerCandidate] = {}
        self._precision_selected_path = settings.precision_device_path
        self._precision_verified_identity = (
            settings.precision_device_identity
            if settings.precision_mapping_verified
            else ""
        )
        self._precision_trigger_rest = (
            settings.precision_trigger_rest if settings.precision_mapping_verified else ""
        )
        self._precision_trigger_pressed = (
            settings.precision_trigger_pressed if settings.precision_mapping_verified else ""
        )
        self._precision_supported = precision_supported()

    def _configure_window(self) -> None:
        self.root.title(APP_NAME)
        self.root.geometry("980x790")
        self.root.minsize(820, 680)
        self.root.option_add("*tearOff", False)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("TkDefaultFont", 19, "bold"))
        style.configure("Subtitle.TLabel", foreground="#586273")
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        style.configure("Start.TButton", font=("TkDefaultFont", 11, "bold"), padding=(18, 9))
        style.configure("Stop.TButton", font=("TkDefaultFont", 11), padding=(18, 9))
        style.configure("Status.TLabel", padding=(8, 3))

    def _build_interface(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        header = ttk.Frame(self.root, padding=(18, 14, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Offline object detection for Moonlight, cameras, capture cards, and video files",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(
            header,
            text="LOCAL ONLY",
            foreground="#237a43",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=0, column=1, rowspan=2, sticky="e")

        self.notebook = ttk.Notebook(self.root, padding=(0, 2))
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=18)
        capture_tab = ttk.Frame(self.notebook, padding=14)
        detection_tab = ttk.Frame(self.notebook, padding=14)
        precision_tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(capture_tab, text="  Capture source  ")
        self.notebook.add(detection_tab, text="  Detection settings  ")
        self.notebook.add(precision_tab, text="  Controller precision  ")
        self._build_capture_tab(capture_tab)
        self._build_detection_tab(detection_tab)
        self._build_precision_tab(precision_tab)

        log_frame = ttk.LabelFrame(
            self.root,
            text="Live activity log",
            style="Section.TLabelframe",
            padding=(8, 7),
        )
        log_frame.grid(row=2, column=0, sticky="nsew", padx=18, pady=(10, 8))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame,
            height=10,
            wrap="word",
            state="disabled",
            font=("TkFixedFont", 9),
            background="#14191f",
            foreground="#d7dde7",
            insertbackground="#d7dde7",
            selectbackground="#38506c",
            borderwidth=0,
            padx=8,
            pady=7,
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.tag_configure("warning", foreground="#ffd479")
        self.log.tag_configure("error", foreground="#ff9797")
        self.log.tag_configure("heading", foreground="#8fc7ff")
        self._append_log("Choose a source and press Start detection. F5 also starts; Escape stops.\n", "heading")

        actions = ttk.Frame(self.root, padding=(18, 2, 18, 14))
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(
            actions,
            text="Start detection",
            style="Start.TButton",
            command=self.start,
        )
        self.start_button.grid(row=0, column=0, sticky="w")
        ttk.Label(actions, textvariable=self.status, style="Status.TLabel").grid(
            row=0, column=1, sticky="w", padx=10
        )
        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            style="Stop.TButton",
            command=self.stop,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=2, sticky="e")

    def _build_capture_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        source_choice = ttk.LabelFrame(
            parent,
            text="1. Choose what to watch",
            style="Section.TLabelframe",
            padding=10,
        )
        source_choice.grid(row=0, column=0, sticky="ew")
        choices = (
            ("Moonlight / screen", SOURCE_SCREEN),
            ("Camera / capture card", SOURCE_CAMERA),
            ("Video file", SOURCE_VIDEO),
        )
        for column, (label, value) in enumerate(choices):
            source_choice.columnconfigure(column, weight=1)
            ttk.Radiobutton(
                source_choice,
                text=label,
                value=value,
                variable=self.source_mode,
                command=self._update_source_panel,
            ).grid(row=0, column=column, sticky="w", padx=(2, 16))

        details = ttk.LabelFrame(
            parent,
            text="2. Source options",
            style="Section.TLabelframe",
            padding=10,
        )
        details.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        details.columnconfigure(0, weight=1)
        self._screen_panel = self._build_screen_panel(details)
        self._camera_panel = self._build_camera_panel(details)
        self._video_panel = self._build_video_panel(details)
        for panel in (self._screen_panel, self._camera_panel, self._video_panel):
            panel.grid(row=0, column=0, sticky="nsew")

        display = ttk.LabelFrame(
            parent,
            text="3. Preview",
            style="Section.TLabelframe",
            padding=10,
        )
        display.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(
            display,
            text="Open the live preview window",
            variable=self.preview,
            command=self._update_draw_state,
        ).grid(row=0, column=0, sticky="w")
        self.draw_check = ttk.Checkbutton(
            display,
            text="Draw boxes, labels, and timing",
            variable=self.draw,
        )
        self.draw_check.grid(row=0, column=1, sticky="w", padx=(24, 0))
        ttk.Label(
            display,
            text="Tip: keep the preview outside the captured Moonlight area to avoid a mirror effect.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _build_screen_panel(self, parent: ttk.Frame) -> ttk.Frame:
        panel = ttk.Frame(parent)
        panel.columnconfigure(5, weight=1)
        ttk.Label(
            panel,
            text="Open Moonlight first, then capture its monitor or exact desktop rectangle. X11/Xorg is required on Linux.",
            style="Subtitle.TLabel",
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 9))
        ttk.Label(panel, text="Monitor").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(panel, from_=0, to=16, textvariable=self.screen_monitor, width=7).grid(
            row=1, column=1, sticky="w", padx=(6, 20)
        )
        ttk.Label(panel, text="Max capture FPS").grid(row=1, column=2, sticky="w")
        ttk.Combobox(
            panel,
            textvariable=self.screen_fps,
            values=("30", "60", "90", "120"),
            width=8,
        ).grid(row=1, column=3, sticky="w", padx=(6, 0))
        ttk.Button(panel, text="Open Moonlight", command=self._open_moonlight).grid(
            row=1, column=4, sticky="w", padx=(20, 0)
        )
        ttk.Checkbutton(
            panel,
            text="Capture a specific region instead of the full monitor",
            variable=self.use_screen_region,
            command=self._update_region_state,
        ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(10, 6))

        region_items = (
            ("X", self.screen_x),
            ("Y", self.screen_y),
            ("Width", self.screen_width),
            ("Height", self.screen_height),
        )
        self._region_widgets: list[ttk.Widget] = []
        for column, (label, variable) in enumerate(region_items):
            item = ttk.Frame(panel)
            item.grid(row=3, column=column, sticky="w", padx=(0, 14))
            label_widget = ttk.Label(item, text=label)
            label_widget.grid(row=0, column=0, sticky="w")
            entry = ttk.Entry(item, textvariable=variable, width=9)
            entry.grid(row=1, column=0, sticky="w", pady=(2, 0))
            self._region_widgets.extend((label_widget, entry))

        controller = ttk.LabelFrame(
            panel,
            text="Controller through Moonlight",
            style="Section.TLabelframe",
            padding=(9, 7),
        )
        controller.grid(
            row=4,
            column=0,
            columnspan=6,
            sticky="ew",
            pady=(12, 0),
        )
        controller.columnconfigure(0, weight=1)
        ttk.Label(
            controller,
            text=(
                "Connect your controller to this laptop. Moonlight forwards your "
                "physical controls; detection never controls the gamepad. The optional "
                "precision tab only reshapes right-stick input you produce."
            ),
            style="Subtitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            controller,
            text="Controller help…",
            command=self._show_controller_help,
        ).grid(row=0, column=1, sticky="e", padx=(14, 0))
        return panel

    def _build_camera_panel(self, parent: ttk.Frame) -> ttk.Frame:
        panel = ttk.Frame(parent)
        panel.columnconfigure(6, weight=1)
        ttk.Label(
            panel,
            text="Use a USB camera or HDMI capture card visible to the operating system.",
            style="Subtitle.TLabel",
        ).grid(row=0, column=0, columnspan=7, sticky="w", pady=(0, 9))
        ttk.Label(panel, text="Device index").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            panel,
            textvariable=self.camera_index,
            values=tuple(str(index) for index in range(10)),
            width=7,
        ).grid(row=2, column=0, sticky="w", padx=(0, 18), pady=(2, 0))
        for column, (label, variable, values) in enumerate(
            (
                ("Width", self.capture_width, ("640", "1280", "1920")),
                ("Height", self.capture_height, ("480", "720", "1080")),
                ("Requested FPS", self.capture_fps, ("30", "60", "120")),
            ),
            start=1,
        ):
            ttk.Label(panel, text=label).grid(row=1, column=column, sticky="w", padx=(0, 18))
            ttk.Combobox(panel, textvariable=variable, values=values, width=10).grid(
                row=2, column=column, sticky="w", padx=(0, 18), pady=(2, 0)
            )
        ttk.Label(
            panel,
            text="Leave width, height, and FPS blank to use the device defaults.",
            style="Subtitle.TLabel",
        ).grid(row=3, column=0, columnspan=7, sticky="w", pady=(9, 0))
        return panel

    def _build_video_panel(self, parent: ttk.Frame) -> ttk.Frame:
        panel = ttk.Frame(parent)
        panel.columnconfigure(0, weight=1)
        ttk.Label(
            panel,
            text="Choose a recorded gameplay clip. It is analyzed as fast as the laptop allows and stops at the end.",
            style="Subtitle.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 9))
        ttk.Entry(panel, textvariable=self.video_path).grid(row=1, column=0, sticky="ew")
        ttk.Button(panel, text="Browse…", command=self._browse_video).grid(
            row=1, column=1, sticky="e", padx=(8, 0)
        )
        return panel

    def _build_detection_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        model_frame = ttk.LabelFrame(
            parent,
            text="Detection model",
            style="Section.TLabelframe",
            padding=10,
        )
        model_frame.grid(row=0, column=0, sticky="ew")
        model_frame.columnconfigure(1, weight=1)
        ttk.Label(model_frame, text="Preset").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.model_preset_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_preset,
            values=tuple(preset.label for preset in MODEL_PRESETS),
            state="readonly",
        )
        self.model_preset_combo.grid(row=0, column=1, columnspan=2, sticky="ew")
        self.model_preset_combo.bind(
            "<<ComboboxSelected>>",
            self._model_preset_changed,
        )
        ttk.Label(
            model_frame,
            textvariable=self.model_preset_description,
            style="Subtitle.TLabel",
            wraplength=780,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 2))

        ttk.Label(model_frame, text="OpenVINO model").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        self.model_path_entry = ttk.Entry(model_frame, textvariable=self.model_path)
        self.model_path_entry.grid(row=2, column=1, sticky="ew", pady=(6, 0))
        self.model_browse_button = ttk.Button(
            model_frame,
            text="Browse…",
            command=self._browse_model,
        )
        self.model_browse_button.grid(row=2, column=2, padx=(8, 0), pady=(6, 0))
        ttk.Label(model_frame, text="Class labels").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0)
        )
        self.labels_path_entry = ttk.Entry(
            model_frame,
            textvariable=self.labels_path,
        )
        self.labels_path_entry.grid(row=3, column=1, sticky="ew", pady=(8, 0))
        self.labels_browse_button = ttk.Button(
            model_frame,
            text="Browse…",
            command=self._browse_labels,
        )
        self.labels_browse_button.grid(row=3, column=2, padx=(8, 0), pady=(8, 0))
        self._update_model_preset_controls()

        runtime = ttk.LabelFrame(
            parent,
            text="Speed and accuracy",
            style="Section.TLabelframe",
            padding=10,
        )
        runtime.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for column in range(5):
            runtime.columnconfigure(column, weight=1)
        controls = (
            ("OpenVINO device", self.device, ("CPU", "AUTO", "GPU", "NPU")),
            ("Inference size", self.inference_size, ("256", "320", "416", "640")),
            ("Center crop (optional)", self.crop_size, ("", "512", "640", "720", "1080")),
            ("Confidence", self.confidence, ("0.15", "0.25", "0.40", "0.50", "0.65")),
            ("IoU threshold", self.iou_threshold, ("0.30", "0.45", "0.60")),
        )
        for column, (label, variable, values) in enumerate(controls):
            ttk.Label(runtime, text=label).grid(row=0, column=column, sticky="w", padx=(0, 12))
            ttk.Combobox(runtime, textvariable=variable, values=values, width=15).grid(
                row=1, column=column, sticky="ew", padx=(0, 12), pady=(3, 0)
            )

        third_person = ttk.LabelFrame(
            parent,
            text="Third-person view",
            style="Section.TLabelframe",
            padding=10,
        )
        third_person.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(
            third_person,
            text="Ignore my on-screen character",
            variable=self.ignore_self,
            command=self._update_self_filter_state,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(third_person, text="Character appears:").grid(
            row=0, column=1, sticky="w", padx=(24, 8)
        )
        self.self_position_combo = ttk.Combobox(
            third_person,
            textvariable=self.self_position,
            values=tuple(SELF_POSITION_LABELS.values()),
            state="readonly",
            width=24,
        )
        self.self_position_combo.grid(row=0, column=2, sticky="w")
        ttk.Label(
            third_person,
            text=(
                "After a short lock, removes at most one large, persistent player detection "
                "at the bottom. This uses position—not identity—so check the orange ignored "
                "outline during close encounters."
            ),
            style="Subtitle.TLabel",
            wraplength=820,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(7, 0))

        advanced = ttk.LabelFrame(
            parent,
            text="Model compatibility",
            style="Section.TLabelframe",
            padding=10,
        )
        advanced.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(advanced, text="Output decoder").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            advanced,
            textvariable=self.output_format,
            values=("auto", "end2end", "traditional"),
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky="w", padx=(8, 16))
        ttk.Label(
            advanced,
            text="Keep Auto for the included YOLO26 model. Traditional is for standard YOLOv8/YOLO11 exports.",
            style="Subtitle.TLabel",
        ).grid(row=0, column=2, sticky="w")

        ttk.Label(
            parent,
            text="Lower inference sizes are usually faster. Settings are saved automatically when detection starts or the app closes.",
            style="Subtitle.TLabel",
        ).grid(row=4, column=0, sticky="w", pady=(10, 0))

    def _build_precision_tab(self, parent: ttk.Frame) -> None:
        """Build controls for the independent, user-driven controller worker."""

        parent.columnconfigure(0, weight=1)
        explanation = ttk.LabelFrame(
            parent,
            text="What this does",
            style="Section.TLabelframe",
            padding=10,
        )
        explanation.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            explanation,
            text=(
                "While you hold LT, this reshapes only the right-stick movement you physically produce "
                "so small manual adjustments are easier. It never uses detections, images, target "
                "coordinates, UDP, or network messages to move the controller."
            ),
            wraplength=850,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            explanation,
            text=(
                "Release LT for normal 1:1 controls. Moonlight forwards the resulting virtual "
                "controller to the main PC."
            ),
            style="Subtitle.TLabel",
            wraplength=850,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        device = ttk.LabelFrame(
            parent,
            text="1. PXN controller and read-only mapping check",
            style="Section.TLabelframe",
            padding=10,
        )
        device.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        device.columnconfigure(0, weight=1)
        self.precision_device_combo = ttk.Combobox(
            device,
            textvariable=self.precision_device_choice,
            state="readonly",
        )
        self.precision_device_combo.grid(row=0, column=0, sticky="ew")
        self.precision_device_combo.bind(
            "<<ComboboxSelected>>",
            self._precision_selection_changed,
        )
        self.precision_refresh_button = ttk.Button(
            device,
            text="Refresh",
            command=self._refresh_precision_devices,
        )
        self.precision_refresh_button.grid(row=0, column=1, padx=(8, 0))
        self.precision_verify_button = ttk.Button(
            device,
            text="Verify LT + right stick…",
            command=self.verify_precision_mapping,
            state="disabled",
        )
        self.precision_verify_button.grid(row=0, column=2, padx=(8, 0))
        ttk.Label(
            device,
            textvariable=self.precision_device_status,
            style="Subtitle.TLabel",
            wraplength=820,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(7, 0))
        ttk.Label(
            device,
            textvariable=self.precision_mapping_status,
            foreground="#315f8c",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        curve = ttk.LabelFrame(
            parent,
            text="2. Choose LT precision strength",
            style="Section.TLabelframe",
            padding=10,
        )
        curve.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(curve, text="Preset").grid(row=0, column=0, sticky="w")
        self.precision_preset_combo = ttk.Combobox(
            curve,
            textvariable=self.precision_preset,
            values=tuple(preset.label for preset in PRECISION_PRESETS),
            state="readonly",
            width=18,
        )
        self.precision_preset_combo.grid(row=0, column=1, sticky="w", padx=(8, 16))
        self.precision_preset_combo.bind(
            "<<ComboboxSelected>>",
            self._precision_preset_changed,
        )
        ttk.Label(
            curve,
            textvariable=self.precision_preset_description,
            style="Subtitle.TLabel",
        ).grid(row=0, column=2, sticky="w")

        actions = ttk.LabelFrame(
            parent,
            text="3. Start before opening the Moonlight stream",
            style="Section.TLabelframe",
            padding=10,
        )
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        actions.columnconfigure(2, weight=1)
        self.precision_start_button = ttk.Button(
            actions,
            text="Start controller precision",
            style="Start.TButton",
            command=self.start_precision,
            state="disabled",
        )
        self.precision_start_button.grid(row=0, column=0, sticky="w")
        self.precision_stop_button = ttk.Button(
            actions,
            text="Stop precision",
            style="Stop.TButton",
            command=self.stop_precision,
            state="disabled",
        )
        self.precision_stop_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(actions, textvariable=self.precision_status, style="Status.TLabel").grid(
            row=0, column=2, sticky="w", padx=10
        )
        self.precision_moonlight_button = ttk.Button(
            actions,
            text="Open Moonlight",
            command=self._open_moonlight,
            state="disabled",
        )
        self.precision_moonlight_button.grid(row=0, column=3, sticky="e")
        ttk.Label(
            actions,
            text=(
                "Start precision first, then open Moonlight and begin the stream. Stop precision "
                "here without stopping detection."
            ),
            style="Subtitle.TLabel",
            wraplength=820,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(7, 0))

        if not self._precision_supported:
            self.precision_device_status.set(
                "Controller precision is disabled because this feature is available only on Linux."
            )
            self.precision_mapping_status.set("Unavailable on this operating system")
            for widget in (
                self.precision_device_combo,
                self.precision_refresh_button,
                self.precision_verify_button,
                self.precision_preset_combo,
                self.precision_start_button,
                self.precision_stop_button,
                self.precision_moonlight_button,
            ):
                widget.configure(state="disabled")

    def _update_source_panel(self) -> None:
        mode = self.source_mode.get()
        panels = {
            SOURCE_SCREEN: self._screen_panel,
            SOURCE_CAMERA: self._camera_panel,
            SOURCE_VIDEO: self._video_panel,
        }
        for panel in panels.values():
            panel.grid_remove()
        panels.get(mode, self._screen_panel).grid()

    def _update_region_state(self) -> None:
        state = "normal" if self.use_screen_region.get() else "disabled"
        for widget in getattr(self, "_region_widgets", ()):
            widget.configure(state=state)

    def _update_draw_state(self) -> None:
        if self.preview.get():
            self.draw_check.configure(state="normal")
        else:
            self.draw_check.configure(state="disabled")

    def _update_self_filter_state(self) -> None:
        state = "readonly" if self.ignore_self.get() else "disabled"
        self.self_position_combo.configure(state=state)

    def _model_preset_changed(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        """Switch model+labels together and retain the session's custom pair."""

        if self._active_model_preset == MODEL_PRESET_CUSTOM:
            self._custom_model_path = self.model_path.get()
            self._custom_labels_path = self.labels_path.get()
            self._custom_output_format = self.output_format.get()

        key = MODEL_PRESET_VALUES.get(
            self.model_preset.get(),
            DEFAULT_MODEL_PRESET,
        )
        selected = model_preset(key)
        if selected.bundled:
            model_path, labels_path = model_preset_paths(selected.key)
            # Both included YOLO26 exports use the runtime's automatic decoder.
            # Never carry a custom model's traditional/end2end choice into a
            # bundled preset.
            self.output_format.set("auto")
        else:
            model_path = self._custom_model_path
            labels_path = self._custom_labels_path
            self.output_format.set(self._custom_output_format)

        # Set the pair from the same resolved selection.  The form can never
        # retain one bundled model with another preset's labels.
        self.model_path.set(model_path)
        self.labels_path.set(labels_path)
        self._active_model_preset = selected.key
        self.model_preset_description.set(selected.description)
        self._update_model_preset_controls()

    def _update_model_preset_controls(self) -> None:
        editable = self._active_model_preset == MODEL_PRESET_CUSTOM
        entry_state = "normal" if editable else "readonly"
        button_state = "normal" if editable else "disabled"
        self.model_path_entry.configure(state=entry_state)
        self.labels_path_entry.configure(state=entry_state)
        self.model_browse_button.configure(state=button_state)
        self.labels_browse_button.configure(state=button_state)

    def _selected_precision_candidate(self) -> ControllerCandidate | None:
        return self._precision_candidates.get(self.precision_device_choice.get())

    def _refresh_precision_devices(self, silent: bool = False) -> None:
        if not self._precision_supported:
            return
        process = self.precision_process
        if process is not None and process.poll() is None:
            if not silent:
                self.precision_status.set("Stop precision before refreshing controllers")
            return
        try:
            candidates = pxn_controllers()
        except OSError as exc:
            candidates = ()
            self.precision_device_status.set(f"Could not scan controllers: {exc}")
        labels = [candidate_label(candidate) for candidate in candidates]
        self._precision_candidates = dict(zip(labels, candidates, strict=True))
        self.precision_device_combo.configure(values=tuple(labels))
        selected = select_saved_candidate(candidates, self._precision_selected_path)
        if selected is None:
            self.precision_device_choice.set("")
            if candidates:
                self.precision_device_status.set(
                    "More than one PXN P5 8K was found. Choose the controller to verify."
                )
            else:
                self.precision_device_status.set(
                    "PXN P5 8K not found. Connect it by USB, then press Refresh."
                )
                if self._precision_verified_identity:
                    self.precision_mapping_status.set(
                        "A verified controller is saved; reconnect that same device to continue."
                    )
                else:
                    self.precision_mapping_status.set("Mapping not verified")
        else:
            label = candidate_label(selected)
            self.precision_device_choice.set(label)
            self._precision_selected_path = str(selected.path)
            identity_matches = bool(
                self.precision_mapping_verified.get()
                and self._precision_verified_identity
                and candidate_identity(selected) == self._precision_verified_identity
            )
            if not identity_matches:
                self.precision_mapping_verified.set(False)
                self._precision_verified_identity = ""
                self._precision_trigger_rest = ""
                self._precision_trigger_pressed = ""
            self.precision_device_status.set(f"Found {label}")
            if identity_matches:
                self.precision_mapping_status.set(
                    "Mapping verified for this controller — ready to start."
                )
            else:
                self.precision_mapping_status.set(
                    "Mapping not verified. Run the read-only LT + right-stick check once."
                )
        self._update_precision_controls()

    def _precision_selection_changed(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        candidate = self._selected_precision_candidate()
        if candidate is None:
            self._update_precision_controls()
            return
        new_identity = candidate_identity(candidate)
        self._precision_selected_path = str(candidate.path)
        if new_identity != self._precision_verified_identity:
            self._precision_verified_identity = ""
            self._precision_trigger_rest = ""
            self._precision_trigger_pressed = ""
            self.precision_mapping_verified.set(False)
            self.precision_mapping_status.set(
                "Mapping not verified. Run the read-only LT + right-stick check once."
            )
        else:
            self.precision_mapping_verified.set(True)
            self.precision_mapping_status.set(
                "Mapping verified for this controller — ready to start."
            )
        self.precision_device_status.set(f"Selected {candidate_label(candidate)}")
        self._update_precision_controls()

    def _precision_preset_changed(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        key = PRECISION_PRESET_VALUES.get(
            self.precision_preset.get(),
            DEFAULT_PRECISION_PRESET,
        )
        self.precision_preset_description.set(precision_preset(key).description)

    def _update_precision_controls(self) -> None:
        if not self._precision_supported:
            return
        process = self.precision_process
        busy = process is not None and process.poll() is None
        candidate = self._selected_precision_candidate()
        verified = bool(
            candidate is not None
            and self.precision_mapping_verified.get()
            and self._precision_verified_identity
            and candidate_identity(candidate) == self._precision_verified_identity
        )
        self.precision_device_combo.configure(
            state="readonly" if self._precision_candidates and not busy else "disabled"
        )
        self.precision_refresh_button.configure(state="disabled" if busy else "normal")
        self.precision_verify_button.configure(
            state=(
                "normal"
                if candidate is not None and candidate.readable and not busy
                else "disabled"
            )
        )
        self.precision_preset_combo.configure(state="disabled" if busy else "readonly")
        self.precision_start_button.configure(
            state="normal" if verified and not busy else "disabled"
        )
        self.precision_stop_button.configure(state="normal" if busy else "disabled")
        self.precision_moonlight_button.configure(
            state=(
                "normal"
                if busy
                and self._precision_mode == "run"
                and self._precision_ready
                and not self._precision_stop_requested
                else "disabled"
            )
        )

    def _settings_from_form(self) -> LauncherSettings:
        precision_candidate = self._selected_precision_candidate()
        precision_path = (
            str(precision_candidate.path)
            if precision_candidate is not None
            else self._precision_selected_path
        )
        verified = bool(
            precision_candidate is not None
            and self.precision_mapping_verified.get()
            and self._precision_verified_identity
            and candidate_identity(precision_candidate) == self._precision_verified_identity
        )
        # If a previously verified device is merely unplugged, retain its
        # device-bound record so reconnecting it does not require rechecking.
        if precision_candidate is None and self._precision_verified_identity:
            verified = self.precision_mapping_verified.get()
        return LauncherSettings(
            version=SETTINGS_VERSION,
            source_mode=self.source_mode.get(),
            video_path=self.video_path.get(),
            camera_index=self.camera_index.get(),
            capture_width=self.capture_width.get(),
            capture_height=self.capture_height.get(),
            capture_fps=self.capture_fps.get(),
            screen_monitor=self.screen_monitor.get(),
            use_screen_region=self.use_screen_region.get(),
            screen_x=self.screen_x.get(),
            screen_y=self.screen_y.get(),
            screen_width=self.screen_width.get(),
            screen_height=self.screen_height.get(),
            screen_fps=self.screen_fps.get(),
            model_preset=MODEL_PRESET_VALUES.get(
                self.model_preset.get(),
                DEFAULT_MODEL_PRESET,
            ),
            model_path=self.model_path.get(),
            labels_path=self.labels_path.get(),
            device=self.device.get(),
            inference_size=self.inference_size.get(),
            crop_size=self.crop_size.get(),
            confidence=self.confidence.get(),
            iou_threshold=self.iou_threshold.get(),
            output_format=self.output_format.get(),
            ignore_self=self.ignore_self.get(),
            self_position=SELF_POSITION_VALUES.get(
                self.self_position.get(),
                SELF_POSITION_LEFT,
            ),
            preview=self.preview.get(),
            draw=self.draw.get(),
            precision_device_path=precision_path,
            precision_device_identity=(self._precision_verified_identity if verified else ""),
            precision_mapping_verified=verified,
            precision_preset=PRECISION_PRESET_VALUES.get(
                self.precision_preset.get(),
                DEFAULT_PRECISION_PRESET,
            ),
            precision_trigger_rest=(self._precision_trigger_rest if verified else ""),
            precision_trigger_pressed=(self._precision_trigger_pressed if verified else ""),
        )

    def verify_precision_mapping(self) -> None:
        """Run the controller's two-phase, read-only mapping observation."""

        if not self._precision_supported:
            messagebox.showerror(
                "Linux required",
                "Controller precision is available only on Linux.",
                parent=self.root,
            )
            return
        candidate = self._selected_precision_candidate()
        if candidate is None:
            messagebox.showerror(
                "PXN controller not selected",
                "Connect the PXN P5 8K, press Refresh, and select it first.",
                parent=self.root,
            )
            return
        if not candidate.readable:
            messagebox.showerror(
                "Controller permission needed",
                "This desktop session cannot read the PXN controller. Reconnect it while signed in, "
                "then press Refresh. Do not run the whole app as root.",
                parent=self.root,
            )
            return
        try:
            command = precision_command(candidate, mode="verify", verification_seconds=4.0)
        except ValueError as exc:
            messagebox.showerror("Cannot verify mapping", str(exc), parent=self.root)
            return

        messagebox.showinfo(
            "Read-only mapping check",
            "This check only observes your physical controller. It does not grab it or create "
            "controller input.\n\n"
            "After pressing OK:\n"
            "1. Repeatedly squeeze and release LT for about 4 seconds.\n"
            "2. Then move only the right stick in full circles for about 4 seconds.\n\n"
            "Keep the other controls still. Progress and results appear in the activity log.",
            parent=self.root,
        )
        self._precision_verified_identity = ""
        self._precision_trigger_rest = ""
        self._precision_trigger_pressed = ""
        self.precision_mapping_verified.set(False)
        self.precision_mapping_status.set("Checking LT and right-stick mapping (read-only)…")
        if not self._start_precision_child(
            command,
            mode="verify",
            pending_identity=candidate_identity(candidate),
        ):
            self.precision_mapping_status.set("Mapping not verified")
            return
        try:
            save_settings(self._settings_from_form())
        except OSError as exc:
            self._append_log(f"Warning: controller settings could not be saved: {exc}\n", "warning")

    def start_precision(self) -> None:
        """Start the independent physical-to-virtual precision worker."""

        if not self._precision_supported:
            return
        process = self.precision_process
        if process is not None and process.poll() is None:
            return
        candidate = self._selected_precision_candidate()
        if candidate is None:
            messagebox.showerror(
                "PXN controller not selected",
                "Connect the PXN P5 8K and press Refresh.",
                parent=self.root,
            )
            return
        identity = candidate_identity(candidate)
        if (
            not self.precision_mapping_verified.get()
            or not self._precision_verified_identity
            or identity != self._precision_verified_identity
        ):
            messagebox.showerror(
                "Verify the controller mapping first",
                "Run the read-only LT + right-stick check for this controller before starting.",
                parent=self.root,
            )
            return
        readiness = precision_readiness(candidate)
        if not readiness.ready:
            detail = f"{readiness.summary}\n\n{readiness.action}".strip()
            self.precision_status.set(readiness.summary)
            self._append_log(f"Controller precision not ready: {detail.replace(chr(10), ' ')}\n", "error")
            messagebox.showerror("Controller precision is not ready", detail, parent=self.root)
            return
        preset_key = PRECISION_PRESET_VALUES.get(
            self.precision_preset.get(),
            DEFAULT_PRECISION_PRESET,
        )
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
                preset_key=preset_key,
                parent_pid=os.getpid(),
                trigger_rest=trigger_rest,
                trigger_pressed=trigger_pressed,
            )
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Cannot start controller precision", str(exc), parent=self.root)
            return
        try:
            save_settings(self._settings_from_form())
        except OSError as exc:
            self._append_log(f"Warning: controller settings could not be saved: {exc}\n", "warning")
        self._start_precision_child(command, mode="run", pending_identity=identity)

    def _start_precision_child(
        self,
        command: list[str],
        *,
        mode: str,
        pending_identity: str,
    ) -> bool:
        heading = "Checking controller mapping" if mode == "verify" else "Starting controller precision"
        self._append_log(f"\n{heading}\n", "heading")
        self._append_log(f"$ {_display_command(command)}\n")
        try:
            process = start_precision_controller(command)
        except OSError as exc:
            self.precision_status.set("Could not start")
            self._append_log(f"Error: could not start controller helper: {exc}\n", "error")
            messagebox.showerror(
                "Could not start controller helper",
                f"The controller helper process could not be started.\n\n{exc}",
                parent=self.root,
            )
            self._update_precision_controls()
            return False
        self.precision_process = process
        self._precision_mode = mode
        self._precision_pending_identity = pending_identity
        self._precision_stop_requested = False
        self._precision_ready = False
        self._precision_guidance_shown = False
        self._precision_recent_output = []
        self.precision_status.set("Checking mapping…" if mode == "verify" else "Starting…")
        self._precision_reader = threading.Thread(
            target=self._read_precision_output,
            args=(process,),
            name="controller-precision-log-reader",
            daemon=True,
        )
        self._precision_reader.start()
        self._update_precision_controls()
        return True

    def _precision_moonlight_guidance(
        self,
        process: subprocess.Popen[str] | None,
    ) -> None:
        if (
            process is None
            or self.precision_process is not process
            or process.poll() is not None
            or self._precision_mode != "run"
            or not self._precision_ready
            or self._precision_guidance_shown
            or self._closing
        ):
            return
        self._precision_guidance_shown = True
        self.precision_status.set("Active — hold LT for fine manual control")
        messagebox.showinfo(
            "Controller precision started",
            "The controller worker is running first. Now open Moonlight and begin your stream.\n\n"
            "Hold LT to reshape only the right-stick movement you make; release LT for normal "
            "controls. Detection and target coordinates are never used.",
            parent=self.root,
        )

    def _handle_precision_output_line(self, item: str) -> None:
        """Consume one worker line, including the private ready handshake."""

        if is_controller_ready_line(item):
            process = self.precision_process
            if (
                process is not None
                and process.poll() is None
                and self._precision_mode == "run"
                and not self._precision_stop_requested
                and not self._precision_ready
            ):
                self._precision_ready = True
                self.precision_status.set("Active — hold LT for fine manual control")
                self._append_log(
                    "[Controller] Controller and virtual input are ready.\n",
                    "heading",
                )
                self._update_precision_controls()
                self._precision_moonlight_guidance(process)
            # Never expose the private protocol token in the activity log.
            return

        self._precision_recent_output.append(item)
        if len(self._precision_recent_output) > 80:
            del self._precision_recent_output[:-80]
        lowered = item.lower()
        tag = "error" if "stopped safely:" in lowered or "error:" in lowered else None
        self._append_log(f"[Controller] {item}", tag)

    def stop_precision(self) -> None:
        process = self.precision_process
        if process is None or process.poll() is not None or self._precision_stop_requested:
            return
        self._precision_stop_requested = True
        self.precision_status.set("Stopping…")
        self.precision_stop_button.configure(state="disabled")
        self._append_log("Stopping controller precision…\n", "warning")
        request_stop(process)
        self.root.after(
            3000,
            lambda current=process: self._force_stop_precision_if_current(current),
        )
        self.root.after(
            6000,
            lambda current=process: self._kill_precision_if_current(current),
        )

    def _force_stop_precision_if_current(self, process: subprocess.Popen[str]) -> None:
        if self.precision_process is process and process.poll() is None:
            self._append_log("Controller helper did not exit yet; terminating it.\n", "warning")
            force_stop(process)

    def _kill_precision_if_current(self, process: subprocess.Popen[str]) -> None:
        if self.precision_process is process and process.poll() is None:
            self._append_log("Controller helper still did not exit; forcing shutdown.\n", "error")
            kill_process(process)

    def _read_precision_output(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is not None:
            try:
                for line in stream:
                    self._precision_process_output.put(line)
            except (OSError, ValueError) as exc:
                self._precision_process_output.put(f"Controller log reader stopped: {exc}\n")
            finally:
                try:
                    stream.close()
                except OSError:
                    pass
        self._precision_process_output.put(None)

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        settings = self._settings_from_form()
        if (
            settings.source_mode == SOURCE_SCREEN
            and os.name == "posix"
            and os.environ.get("XDG_SESSION_TYPE", "").strip().lower() == "wayland"
        ):
            message = (
                "Moonlight screen capture needs an X11/Xorg desktop session in this version.\n\n"
                "Log out, choose an Xorg/X11 session on the login screen, then open "
                "Moonlight and Game Detector again. If CachyOS/Arch does not show that "
                "choice, first install it with: sudo pacman -S plasma-x11-session\n\n"
                "Cameras and video files still work on Wayland."
            )
            self.status.set("Moonlight capture needs X11/Xorg")
            self._append_log(f"Cannot start: {message.replace(chr(10), ' ')}\n", "error")
            messagebox.showerror("X11/Xorg session required", message, parent=self.root)
            return
        try:
            command = launcher_command(settings)
        except SettingsError as exc:
            self.status.set("Check the detection settings")
            messagebox.showerror("Cannot start detection", str(exc), parent=self.root)
            return

        try:
            written = save_settings(settings)
        except OSError as exc:
            self._append_log(f"Warning: settings could not be saved: {exc}\n", "warning")
        else:
            self._append_log(f"Settings saved to {written}\n")

        self._append_log("\nStarting detector\n", "heading")
        self._append_log(f"$ {_display_command(command)}\n")
        try:
            process = start_detector(command)
        except OSError as exc:
            self.status.set("Could not start")
            self._append_log(f"Error: could not start detector: {exc}\n", "error")
            messagebox.showerror(
                "Could not start detection",
                f"The detector process could not be started.\n\n{exc}",
                parent=self.root,
            )
            return

        self.process = process
        self._stop_requested = False
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("Starting model…")
        self._reader = threading.Thread(
            target=self._read_output,
            args=(process,),
            name="detector-log-reader",
            daemon=True,
        )
        self._reader.start()

    def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        if self._stop_requested:
            return
        self._stop_requested = True
        self.status.set("Stopping…")
        self.stop_button.configure(state="disabled")
        self._append_log("Stopping detector…\n", "warning")
        request_stop(process)
        self.root.after(3000, lambda current=process: self._force_stop_if_current(current))
        self.root.after(6000, lambda current=process: self._kill_if_current(current))

    def _force_stop_if_current(self, process: subprocess.Popen[str]) -> None:
        if self.process is process and process.poll() is None:
            self._append_log("Detector did not exit yet; terminating it.\n", "warning")
            force_stop(process)

    def _kill_if_current(self, process: subprocess.Popen[str]) -> None:
        if self.process is process and process.poll() is None:
            self._append_log("Detector still did not exit; forcing shutdown.\n", "error")
            kill_process(process)

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is not None:
            try:
                for line in stream:
                    self._process_output.put(line)
            except (OSError, ValueError) as exc:
                self._process_output.put(f"Log reader stopped: {exc}\n")
            finally:
                try:
                    stream.close()
                except OSError:
                    pass
        self._process_output.put(None)

    def _poll_process(self) -> None:
        saw_end = False
        for _ in range(300):
            try:
                item = self._process_output.get_nowait()
            except queue.Empty:
                break
            if item is None:
                saw_end = True
                continue
            lowered = item.lower()
            tag = "error" if "error:" in lowered else "warning" if "warning" in lowered else None
            self._append_log(item, tag)

        process = self.process
        if process is not None:
            return_code = process.poll()
            if return_code is None:
                if not self._stop_requested:
                    self.status.set("Detection running")
            elif saw_end or self._reader is None or not self._reader.is_alive():
                self._finish_process(process, return_code)

        precision_saw_end = False
        for _ in range(300):
            try:
                item = self._precision_process_output.get_nowait()
            except queue.Empty:
                break
            if item is None:
                precision_saw_end = True
                continue
            self._handle_precision_output_line(item)

        precision_process = self.precision_process
        if precision_process is not None:
            return_code = precision_process.poll()
            if return_code is None:
                if not self._precision_stop_requested:
                    if self._precision_mode == "verify":
                        self.precision_status.set("Checking mapping…")
                    elif self._precision_ready:
                        self.precision_status.set("Active — hold LT for fine manual control")
                    else:
                        self.precision_status.set("Starting…")
            elif (
                precision_saw_end
                or self._precision_reader is None
                or not self._precision_reader.is_alive()
            ):
                self._finish_precision_process(precision_process, return_code)

        if not self._closing:
            self.root.after(POLL_INTERVAL_MS, self._poll_process)
        elif self.process is None and self.precision_process is None:
            self.root.destroy()
        else:
            self.root.after(POLL_INTERVAL_MS, self._poll_process)

    def _finish_process(self, process: subprocess.Popen[str], return_code: int) -> None:
        if self.process is not process:
            return
        self.process = None
        self._reader = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self._stop_requested and return_code in (0, 130, -2, -15):
            self.status.set("Stopped")
            self._append_log("Detector stopped.\n", "heading")
        elif return_code == 0:
            self.status.set("Finished")
            self._append_log("Detector finished normally.\n", "heading")
        else:
            self.status.set(f"Stopped with error ({return_code})")
            self._append_log(f"Detector exited with code {return_code}. See the log above.\n", "error")
        self._stop_requested = False

    def _finish_precision_process(
        self,
        process: subprocess.Popen[str],
        return_code: int,
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
        self._precision_guidance_shown = False

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
                self.precision_mapping_verified.set(True)
                self.precision_mapping_status.set(
                    "Mapping verified for this controller — ready to start."
                )
                self.precision_status.set("Mapping verified")
                self._append_log("Controller mapping verified and saved for this PXN.\n", "heading")
                if not self._closing:
                    messagebox.showinfo(
                        "Controller mapping verified",
                        "LT and the right-stick axes matched the PXN P5 8K default mapping. "
                        "You can now start controller precision.",
                        parent=self.root,
                    )
            else:
                self._precision_verified_identity = ""
                self._precision_trigger_rest = ""
                self._precision_trigger_pressed = ""
                self.precision_mapping_verified.set(False)
                self.precision_mapping_status.set(
                    "Mapping not verified. Repeat the check and move only the requested controls."
                )
                if stop_requested and return_code in (0, 130, -2, -15):
                    self.precision_status.set("Mapping check stopped")
                    self._append_log("Controller mapping check stopped.\n", "warning")
                else:
                    self.precision_status.set("Mapping check failed")
                    self._append_log(
                        f"Controller mapping was not verified (code {return_code}).\n",
                        "error",
                    )
                    if not self._closing:
                        messagebox.showwarning(
                            "Mapping not verified",
                            "The expected LT and right-stick axes were not observed. Reconnect the "
                            "PXN, press Refresh, and repeat the check while moving only the requested controls.",
                            parent=self.root,
                        )
        elif stop_requested and return_code in (0, 130, -2, -15):
            self.precision_status.set("Stopped")
            self._append_log("Controller precision stopped.\n", "heading")
        elif not became_ready:
            self.precision_status.set("Could not start")
            self._append_log(
                f"Controller precision exited before input was ready (code {return_code}).\n",
                "error",
            )
            if not self._closing:
                recent = "".join(self._precision_recent_output).strip()
                detail = recent[-1200:] if recent else "See the activity log for details."
                messagebox.showerror(
                    "Controller precision could not start",
                    f"{detail}\n\nThe controller was not reported active. Check the connection and "
                    "permissions, then press Refresh and try again.",
                    parent=self.root,
                )
        elif return_code == 0:
            self.precision_status.set("Stopped")
            self._append_log("Controller precision finished safely.\n", "heading")
        else:
            self.precision_status.set(f"Stopped with error ({return_code})")
            self._append_log(
                f"Controller precision exited with code {return_code}.\n",
                "error",
            )
            if not self._closing:
                recent = "".join(self._precision_recent_output).strip()
                detail = recent[-1200:] if recent else "See the activity log for details."
                messagebox.showerror(
                    "Controller precision stopped",
                    f"{detail}\n\nIf /dev/uinput is missing, run sudo modprobe uinput, then press "
                    "Refresh. Game Detector never makes privileged changes automatically.",
                    parent=self.root,
                )

        try:
            save_settings(self._settings_from_form())
        except OSError as exc:
            self._append_log(f"Warning: controller settings could not be saved: {exc}\n", "warning")
        self._update_precision_controls()

    def _append_log(self, text: str, tag: str | None = None) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text, (tag,) if tag else ())
        self._log_lines += text.count("\n")
        if self._log_lines > LOG_LINE_LIMIT:
            remove = self._log_lines - LOG_LINE_LIMIT + 500
            self.log.delete("1.0", f"{remove + 1}.0")
            self._log_lines -= remove
        self.log.see("end")
        self.log.configure(state="disabled")

    def _browse_video(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a gameplay video",
            initialdir=_initial_directory(self.video_path.get()),
            filetypes=(
                ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"),
                ("All files", "*"),
            ),
        )
        if selected:
            self.video_path.set(selected)

    def _open_moonlight(self) -> None:
        executable = shutil.which("moonlight") or shutil.which("moonlight-qt")
        if executable is None:
            messagebox.showerror(
                "Moonlight was not found",
                "Install Moonlight or open it from your application menu. "
                "Game Detector will capture the Moonlight window after you start your stream.",
                parent=self.root,
            )
            return
        options: dict[str, object] = {}
        if os.name == "nt":
            options["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0)
        else:
            options["start_new_session"] = True
        try:
            subprocess.Popen(
                [executable],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=external_process_environment(),
                **options,
            )
        except OSError as exc:
            messagebox.showerror(
                "Could not open Moonlight",
                f"Moonlight was found but could not be opened.\n\n{exc}",
                parent=self.root,
            )
            return
        self.status.set("Moonlight opened — start your stream, then start detection")
        self._append_log("Opened Moonlight. Choose your PC and stream before starting detection.\n")

    def _show_controller_help(self) -> None:
        messagebox.showinfo(
            "Controller through Moonlight",
            "1. Connect or pair the controller with this laptop before starting the stream. "
            "USB is the simplest first test.\n\n"
            "2. In Moonlight Settings → Gamepad, enable ‘Process gamepad input when "
            "Moonlight is in the background’. This lets the controller keep working if "
            "the detector preview has focus.\n\n"
            "3. Start the Moonlight stream and test a stick or button in the game. If a "
            "game ignores hot-plugging, enable Moonlight’s ‘Force gamepad #1 always "
            "connected’ option and reconnect.\n\n"
            "4. If the main PC still receives nothing, check Sunshine → Input → "
            "Controller. On a Windows host, install Sunshine’s virtual gamepad driver "
            "and reboot if Sunshine requests it.\n\n"
            "Object detection never drives the controller. If you enable Controller "
            "precision, it creates a virtual copy that reshapes only right-stick movement "
            "you physically make while holding LT; it never uses detections or coordinates.",
            parent=self.root,
        )

    def _browse_model(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose an OpenVINO model",
            initialdir=_initial_directory(self.model_path.get()),
            filetypes=(("OpenVINO IR model", "*.xml"), ("All files", "*")),
        )
        if selected:
            self.model_path.set(selected)
            self._custom_model_path = selected

    def _browse_labels(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose class labels",
            initialdir=_initial_directory(self.labels_path.get()),
            filetypes=(("Label files", "*.txt *.json"), ("All files", "*")),
        )
        if selected:
            self.labels_path.set(selected)
            self._custom_labels_path = selected

    def _start_shortcut(self, _event: tk.Event[tk.Misc]) -> str:
        self.start()
        return "break"

    def _stop_shortcut(self, _event: tk.Event[tk.Misc]) -> str:
        self.stop()
        return "break"

    def _on_close(self) -> None:
        if self._closing:
            return
        try:
            save_settings(self._settings_from_form())
        except OSError:
            pass
        detector_running = self.process is not None and self.process.poll() is None
        precision_running = (
            self.precision_process is not None and self.precision_process.poll() is None
        )
        if detector_running or precision_running:
            running = []
            if detector_running:
                running.append("detection")
            if precision_running:
                running.append("controller precision")
            names = " and ".join(running)
            verb = "are" if len(running) > 1 else "is"
            if not messagebox.askyesno(
                "Exit Game Detector?",
                f"{names.capitalize()} {verb} still running. Stop and close the launcher?",
                parent=self.root,
            ):
                return
            self._closing = True
            if detector_running:
                self.stop()
            if precision_running:
                self.stop_precision()
            return

        self._closing = True
        self.root.destroy()


def _display_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _initial_directory(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.parent)
    if candidate.is_dir():
        return str(candidate)
    parent = candidate.parent
    if parent.is_dir():
        return str(parent)
    return str(Path.home())


def run_gui() -> int:
    root = tk.Tk(className="GameDetector")
    DetectorLauncher(root)
    root.mainloop()
    return 0
