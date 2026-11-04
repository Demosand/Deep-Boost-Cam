"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import os
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QProgressBar,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    reset_face_analyser,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode
from modules.video_capture import VideoCapturer
from modules.face_tracker import FaceTracker
from modules.processors.frame import realism, swapper_models
from modules import virtual_camera
from modules.live_pipeline import AsyncDetector, Reprojector, shift_faces
from modules.live_output import LiveOutput
from modules import face_fader, face_library

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

ROOT_HEIGHT = 820
ROOT_WIDTH = 720

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 200


# ─── modern dark stylesheet ───────────────────────────────────────────────

QSS = """
QMainWindow, QDialog { background-color: #1e1e1e; color: #e6e6e6; }
QWidget { color: #e6e6e6; font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif; font-size: 11pt; }

QGroupBox {
    background-color: #262626;
    border: 1px solid #333333;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #9ec5ff;
}

QPushButton {
    background-color: #2d6cdf;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover  { background-color: #3a7af0; }
QPushButton:pressed{ background-color: #1d57c2; }
QPushButton:disabled { background-color: #444; color: #888; }
QPushButton#secondary {
    background-color: #3a3a3a;
}
QPushButton#secondary:hover { background-color: #4a4a4a; }
QPushButton#danger { background-color: #c2412d; }
QPushButton#danger:hover  { background-color: #d8523c; }

QComboBox {
    background-color: #2a2a2a;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 24px;
}
QComboBox:hover { border-color: #2d6cdf; }
QComboBox QAbstractItemView {
    background-color: #2a2a2a;
    selection-background-color: #2d6cdf;
    border: 1px solid #404040;
}

QCheckBox {
    spacing: 8px;
    padding: 4px 0;
}
QCheckBox::indicator {
    width: 36px; height: 18px;
    border-radius: 9px;
    background-color: #3a3a3a;
}
QCheckBox::indicator:checked {
    background-color: #2d6cdf;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #3a3a3a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 1px solid #cccccc;
}
QSlider::sub-page:horizontal {
    background: #2d6cdf;
    border-radius: 3px;
}

QLabel#imageDrop {
    background-color: #2a2a2a;
    border: 2px dashed #444;
    border-radius: 8px;
}
QLabel#statusLabel {
    color: #b9b9b9;
    font-size: 10pt;
    font-style: italic;
}
QLabel#linkLabel {
    color: #6ea8ff;
    text-decoration: underline;
}

QScrollArea { border: none; background: transparent; }

QFrame#card {
    background-color: #262626;
    border-radius: 10px;
}

QTabWidget::pane { border: 1px solid #333; border-radius: 10px; top: -1px; background: #1e1e1e; }
QTabBar::tab {
    background: #262626; color: #b9b9b9;
    padding: 8px 18px; margin-right: 4px;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
}
QTabBar::tab:selected { background: #2d6cdf; color: white; }
QTabBar::tab:hover:!selected { background: #333; }

QSpinBox {
    background-color: #2a2a2a; border: 1px solid #404040;
    border-radius: 6px; padding: 4px 8px; min-height: 22px;
}
QSpinBox:hover { border-color: #2d6cdf; }
QWidget:disabled { color: #6a6a6a; }

QLabel#hint { color: #8f8f8f; font-size: 9pt; }
QWidget#settingsPage { background: #1e1e1e; }
QScrollBar:vertical { background: #1e1e1e; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #444; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #555; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QLabel#valueLabel { color: #9ec5ff; font-weight: 600; min-width: 56px; }
QLabel#banner {
    background-color: #3a2f12; color: #ffd27a;
    border: 1px solid #6b5520; border-radius: 8px; padding: 8px 10px;
}
QLabel#featureSummary { color: #9ec5ff; font-size: 9pt; }
QProgressBar {
    background-color: #2a2a2a; border: 1px solid #404040; border-radius: 6px;
    text-align: center; color: #e6e6e6; min-height: 18px;
}
QProgressBar::chunk { background-color: #2d6cdf; border-radius: 5px; }
QListWidget { background-color: #2a2a2a; border: 1px solid #404040; border-radius: 6px; }
QPushButton#settings { background-color: #3a3a3a; }
QPushButton#settings:hover { background-color: #4a4a4a; }
"""


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
# How long to wait quietly for a live-preview worker to notice the stop flag
# before saying so. Shutdown then keeps waiting: it has to finish.
WORKER_SHUTDOWN_GRACE_MS = 30000
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None

# QFileDialog filter strings, built from the canonical extension sets in
# globals so every dialog stays in sync (no hand-copied lists to drift).
_IMAGE_FILE_FILTER = "Images (" + " ".join(
    f"*{ext}" for ext in modules.globals.IMAGE_EXTENSIONS
) + ")"
_MEDIA_FILE_FILTER = "Media (" + " ".join(
    f"*{ext}" for ext in (*modules.globals.IMAGE_EXTENSIONS, *modules.globals.VIDEO_EXTENSIONS)
) + ")"
_VIDEO_FILE_FILTER = "Videos (" + " ".join(
    f"*{ext}" for ext in modules.globals.VIDEO_EXTENSIONS
) + ")"


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────

# Settings-dialog fields, persisted generically. Defaults are snapshotted at
# import (before load_switch_states runs) for "Reset to defaults".
_EXT_KEYS = (
    "det_interval",
    "tracking_v2", "tracking_optical_flow", "tracking_smoothing",
    "tracking_smoothing_strength", "tracking_grace_frames", "tracking_max_motion",
    "tracking_max_rotation", "tracking_min_confidence", "tracking_auto_redetect",
    "freeze_on_face_lost",
    "adaptive_face_mask", "adaptive_mask_feather",
    "color_matching", "color_matching_strength",
    "temporal_stabilization", "temporal_strength",
    "eye_preservation", "eye_preservation_strength",
    "mouth_preservation", "mouth_preservation_strength",
    "dynamic_enhancement", "dynamic_enhancement_min_face", "dynamic_enhancement_max_motion",
    "use_tensorrt", "trt_fp16", "trt_builder_optimization_level",
    "trt_max_workspace_gb", "trt_timing_cache",
    "swapper_model", "face_parts_mask", "face_parts", "face_parts_feather",
    "occlusion_mask", "use_model_mask", "mask_interval", "capture_fps",
    "virtual_camera", "virtual_camera_resolution",
    "multi_source", "source_paths_extra", "async_detection", "reprojection",
    "grain_matching", "grain_amount", "grain_sharpness_amount",
    "face_fader", "fader_seconds", "profile_fade", "profile_yaw_start", "profile_yaw_end",
    "profile_pitch_start", "profile_pitch_end",
    "upscaler", "upscaler_mode", "upscaler_model", "frame_interpolation",
    "check_updates",
)
_EXT_DEFAULTS = {
    k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
    for k, v in ((k, getattr(modules.globals, k)) for k in _EXT_KEYS)
}


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "capture_resolution": list(modules.globals.capture_resolution),
        "det_size": modules.globals.det_size,
        **{k: getattr(modules.globals, k) for k in _EXT_KEYS},
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        modules.globals.fp_ui = state.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = state.get("show_fps", False)
        # Mouth mask always starts disabled (slider at 0) on launch,
        # regardless of the persisted value — enable it explicitly each session.
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
        # Validate persisted camera settings before trusting them — a hand-
        # edited/corrupt file shouldn't push a bad size into cv2/insightface.
        res = state.get("capture_resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            try:
                w, h = int(res[0]), int(res[1])
                if w > 0 and h > 0:
                    modules.globals.capture_resolution = (w, h)
            except (TypeError, ValueError):
                pass
        if state.get("det_size") in (160, 320, 640):
            modules.globals.det_size = int(state["det_size"])
        for k, default in _EXT_DEFAULTS.items():
            v = state.get(k)
            # Type-checked: a hand-edited file can't push a str into a slider.
            if isinstance(default, bool):
                if isinstance(v, bool):
                    setattr(modules.globals, k, v)
            elif isinstance(default, str):
                if isinstance(v, str):
                    setattr(modules.globals, k, v)
            elif isinstance(default, dict):
                if isinstance(v, dict):
                    merged = dict(default)
                    merged.update({kk: vv for kk, vv in v.items()
                                   if kk in default and isinstance(vv, bool)})
                    setattr(modules.globals, k, merged)
            elif isinstance(default, list):
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    setattr(modules.globals, k, list(v))
            elif isinstance(default, tuple):
                if (isinstance(v, (list, tuple)) and len(v) == len(default)
                        and all(isinstance(x, int) and x >= 0 for x in v)):
                    setattr(modules.globals, k, tuple(v))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                setattr(modules.globals, k, type(default)(v))
        from modules.processors.frame import upscaler as _ups
        if modules.globals.upscaler_model not in _ups.MODELS:
            modules.globals.upscaler_model = "span"
        if modules.globals.upscaler_mode not in ("frame", "face"):
            modules.globals.upscaler_mode = "face"
        if modules.globals.swapper_model not in swapper_models.BY_KEY:
            modules.globals.swapper_model = swapper_models.DEFAULT_KEY
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)
    updateAvailable = Signal(dict)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
        )
        self.setMinimumSize(ROOT_WIDTH, ROOT_HEIGHT)
        self.resize(ROOT_WIDTH, ROOT_HEIGHT)

        # Scrollable root: the window stays usable on small/scaled screens.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll)
        root = QWidget()
        root.setObjectName("settingsPage")  # dark page background
        scroll.setWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Source/Target row
        layout.addLayout(self._build_image_row())

        layout.addLayout(self._build_face_row())

        # Options grid
        layout.addWidget(self._build_options_card())

        # Sliders card
        layout.addWidget(self._build_sliders_card())

        # Action buttons
        layout.addLayout(self._build_action_row())

        # Advanced settings entry point + what is currently on.
        settings_row = QHBoxLayout()
        self.btn_settings = QPushButton("⚙  " + _("Advanced settings"))
        self.btn_settings.setObjectName("settings")
        self.btn_settings.setToolTip(_(
            "Face tracking, detection frequency, realism and TensorRT acceleration"
        ))
        self.btn_settings.clicked.connect(self._on_open_settings)
        settings_row.addWidget(self.btn_settings)
        self.btn_dfm = QPushButton("🧪  " + _("DFM Studio"))
        self.btn_dfm.setObjectName("settings")
        self.btn_dfm.setToolTip(_("Train your own .dfm face model on this GPU"))
        self.btn_dfm.clicked.connect(self._on_open_dfm_studio)
        settings_row.addWidget(self.btn_dfm)
        layout.addLayout(settings_row)
        self._feature_summary = QLabel("")
        self._feature_summary.setObjectName("featureSummary")
        self._feature_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feature_summary.setWordWrap(True)
        layout.addWidget(self._feature_summary)
        self.refresh_feature_summary()

        # Camera selection
        layout.addWidget(self._build_camera_card())

        # Status & footer
        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        footer = QLabel("Deep Live Cam")
        footer.setObjectName("linkLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setCursor(Qt.CursorShape.PointingHandCursor)
        footer.mousePressEvent = lambda _e: webbrowser.open("https://deeplivecam.net")
        layout.addWidget(footer)

    # ── image row ────────────────────────────────────────────────────────

    def _build_image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        # Source column
        src_col = QVBoxLayout()
        self.source_label = _make_image_drop(_("Source face"), (200, 200))
        src_col.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)
        src_row = QHBoxLayout()
        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.setToolTip(
            _("Choose the source face image to swap onto the target")
        )
        self.btn_select_source.clicked.connect(self._on_select_source)
        self.btn_random_face = QPushButton("🔄")
        self.btn_random_face.setObjectName("secondary")
        self.btn_random_face.setFixedWidth(40)
        self.btn_random_face.setToolTip(
            _("Get a random face from thispersondoesnotexist.com")
        )
        self.btn_random_face.clicked.connect(self._on_random_face)
        src_row.addWidget(self.btn_select_source)
        src_row.addWidget(self.btn_random_face)
        src_col.addLayout(src_row)

        # Swap button column
        swap_col = QVBoxLayout()
        swap_col.addStretch(1)
        self.btn_swap = QPushButton("↔")
        self.btn_swap.setObjectName("secondary")
        self.btn_swap.setFixedSize(44, 44)
        self.btn_swap.setToolTip(_("Swap source and target images"))
        self.btn_swap.clicked.connect(self._on_swap_paths)
        swap_col.addWidget(self.btn_swap, alignment=Qt.AlignmentFlag.AlignCenter)
        swap_col.addStretch(1)

        # Target column
        tgt_col = QVBoxLayout()
        self.target_label = _make_image_drop(_("Target"), (200, 200))
        tgt_col.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_select_target = QPushButton(_("Select a target"))
        self.btn_select_target.setToolTip(
            _("Choose the target image or video to apply face swap to")
        )
        self.btn_select_target.clicked.connect(self._on_select_target)
        tgt_col.addWidget(self.btn_select_target)

        row.addLayout(src_col)
        row.addLayout(swap_col)
        row.addLayout(tgt_col)
        return row

    # ── face row: swap on/off + face library ─────────────────────────────

    def _build_face_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_swap_toggle = QPushButton()
        self.btn_swap_toggle.setCheckable(True)
        self.btn_swap_toggle.setChecked(modules.globals.swap_enabled)
        self.btn_swap_toggle.toggled.connect(self._on_swap_toggled)
        self.btn_swap_toggle.setToolTip(_(
            "Turn the face swap on/off without stopping Live. With Face Fader on, "
            "the change is smooth."))
        self._paint_swap_toggle()
        row.addWidget(self.btn_swap_toggle)
        row.addWidget(QLabel(_("Face library:")))
        self.cb_library = QComboBox()
        self.cb_library.setToolTip(_("Saved face sets: choose one to make it the source face"))
        self.cb_library.activated.connect(self._on_library_pick)
        row.addWidget(self.cb_library, 1)
        b_save = QPushButton("💾")
        b_save.setObjectName("secondary")
        b_save.setFixedWidth(40)
        b_save.setToolTip(_("Save the current source photo(s) to the library"))
        b_save.clicked.connect(self._on_library_save)
        row.addWidget(b_save)
        b_del = QPushButton("🗑")
        b_del.setObjectName("secondary")
        b_del.setFixedWidth(40)
        b_del.setToolTip(_("Delete the selected set from the library"))
        b_del.clicked.connect(self._on_library_delete)
        row.addWidget(b_del)
        self.refresh_library()
        return row

    def _paint_swap_toggle(self) -> None:
        on = modules.globals.swap_enabled
        self.btn_swap_toggle.setText(_("Swap: ON") if on else _("Swap: OFF"))
        self.btn_swap_toggle.setObjectName("" if on else "danger")
        self.btn_swap_toggle.style().unpolish(self.btn_swap_toggle)
        self.btn_swap_toggle.style().polish(self.btn_swap_toggle)

    def _on_swap_toggled(self, value: bool) -> None:
        modules.globals.swap_enabled = bool(value)
        self._paint_swap_toggle()

    def refresh_library(self) -> None:
        self.cb_library.clear()
        self.cb_library.addItem(_("(choose a saved face)"), "")
        for name in face_library.list_sets():
            self.cb_library.addItem(f"{name}  ({len(face_library.photos(name))})", name)

    def _on_library_pick(self, idx: int) -> None:
        name = self.cb_library.itemData(idx)
        paths = face_library.photos(name) if name else []
        if not paths:
            return
        from modules import source_identity
        modules.globals.source_path = paths[0]
        modules.globals.source_paths_extra = paths[1:]
        modules.globals.multi_source = len(paths) > 1
        source_identity.reset()
        save_switch_states()
        self.source_label.setPixmap(render_image_preview(paths[0], (200, 200)))
        self.source_label.setText("")
        update_status(_("Face from library:") + f" {name}")

    def _on_library_save(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        paths = ([modules.globals.source_path] if modules.globals.source_path else []) + \
            [p for p in modules.globals.source_paths_extra if p != modules.globals.source_path]
        if not paths:
            update_status(_("Select a source face first"))
            return
        name, ok = QInputDialog.getText(self, _("Face library"), _("Name for this face set:"))
        if not ok or not face_library.safe_name(name):
            return
        try:
            stored = face_library.save_set(name, paths)
        except (OSError, ValueError) as error:
            update_status(_("Could not save:") + f" {error}")
            return
        self.refresh_library()
        self.cb_library.setCurrentIndex(max(0, self.cb_library.findData(stored)))
        update_status(_("Saved to library:") + f" {stored} ({len(paths)})")

    def _on_library_delete(self) -> None:
        name = self.cb_library.currentData()
        if name:
            face_library.delete_set(name)
            self.refresh_library()

    # ── options card ─────────────────────────────────────────────────────

    def _build_options_card(self) -> QGroupBox:
        card = QGroupBox(_("Options"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(6)

        def make(field, label, tip):
            sw = _Switch(_(label), getattr(modules.globals, field), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_keep_fps = make("keep_fps", "Keep fps",
                                "Output video keeps the original frame rate")
        self.sw_keep_audio = make("keep_audio", "Keep audio",
                                  "Copy audio track from the source video to output")
        self.sw_keep_frames = make("keep_frames", "Keep frames",
                                   "Keep extracted frames on disk after processing")
        self.sw_many_faces = make("many_faces", "Many faces",
                                  "Swap every detected face, not just the primary one")
        self.sw_poisson = make("poisson_blend", "Poisson Blend",
                               "Blend face edges smoothly using Poisson blending")
        self.sw_color_fix = make("color_correction", "Fix Blueish Cam",
                                 "Fix blue/green color cast from some webcams")
        self.sw_show_fps = make("show_fps", "Show FPS",
                                "Display frames-per-second counter on the live preview")

        # Map faces is special — closes mapper when toggled off.
        self.sw_map_faces = _Switch(_("Map faces"), modules.globals.map_faces,
                                    _("Manually assign which source face maps to which target face"))
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)

        # Layout: 2 columns of switches
        items = [
            self.sw_keep_fps, self.sw_keep_audio,
            self.sw_keep_frames, self.sw_many_faces,
            self.sw_map_faces, self.sw_show_fps,
            self.sw_poisson, self.sw_color_fix,
        ]
        for i, w in enumerate(items):
            grid.addWidget(w, i // 2, i % 2)

        # Face enhancer dropdown
        enhancer_label = QLabel(_("Face Enhancer:"))
        grid.addWidget(enhancer_label, len(items) // 2, 0)

        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(["None", "GFPGAN", "GPEN-512", "GPEN-256"])
        initial = "None"
        if modules.globals.fp_ui.get("face_enhancer", False):
            initial = "GFPGAN"
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            initial = "GPEN-512"
        elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            initial = "GPEN-256"
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("Select a face enhancement model (None = no enhancement)"))
        grid.addWidget(self.cb_enhancer, len(items) // 2, 1)

        # Face swap model
        grid.addWidget(QLabel(_("Swap model:")), len(items) // 2 + 1, 0)
        self.cb_swapper = QComboBox()
        for spec in swapper_models.SWAPPERS:
            self.cb_swapper.addItem(spec.label, spec.key)
        self.cb_swapper.setCurrentIndex(max(0, self.cb_swapper.findData(modules.globals.swapper_model)))
        self.cb_swapper.currentIndexChanged.connect(self._on_swapper_change)
        self.cb_swapper.setToolTip(_(
            "Inswapper 128: original, fastest to load. ReSwapper 256: inswapper-compatible "
            "256 px retrain. HyperSwap 1A/1B/1C 256: sharper 256 px models with their own "
            "face mask. A new model downloads into models/ on first use (~400-550 MB)."
        ))
        grid.addWidget(self.cb_swapper, len(items) // 2 + 1, 1)

        return card

    # ── sliders card ─────────────────────────────────────────────────────

    def _build_sliders_card(self) -> QGroupBox:
        card = QGroupBox(_("Refinement"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        def slider(min_v, max_v, default, denom, on_change):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(min_v * denom), int(max_v * denom))
            s.setValue(int(default * denom))
            s.valueChanged.connect(lambda iv: on_change(iv / denom))
            return s

        # Transparency
        grid.addWidget(QLabel(_("Transparency")), 0, 0)
        self.s_transparency = slider(0.0, 1.0, 1.0, 100, self._on_transparency_change)
        self.s_transparency.setToolTip(
            _("Blend between original and swapped face (0% = original, 100% = fully swapped)")
        )
        grid.addWidget(self.s_transparency, 0, 1)

        # Sharpness
        grid.addWidget(QLabel(_("Sharpness")), 1, 0)
        self.s_sharpness = slider(0.0, 5.0, 0.0, 10, self._on_sharpness_change)
        self.s_sharpness.setToolTip(_("Sharpen the enhanced face output"))
        grid.addWidget(self.s_sharpness, 1, 1)

        # Mouth mask — always starts at 0 (disabled) on launch
        grid.addWidget(QLabel(_("Mouth Mask")), 2, 0)
        self.s_mouth = slider(0.0, 100.0, 0.0, 1,
                              self._on_mouth_mask_change)
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)
        self.s_mouth.setToolTip(
            _("0 = use swapped mouth, 100 = expose original mouth to chin area")
        )
        grid.addWidget(self.s_mouth, 2, 1)
        return card

    # ── action row ───────────────────────────────────────────────────────

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_start = QPushButton(_("Start"))
        self.btn_start.setToolTip(_("Begin processing the target image/video with selected face"))
        self.btn_start.clicked.connect(self._on_start)

        self.btn_destroy = QPushButton(_("Destroy"))
        self.btn_destroy.setObjectName("danger")
        self.btn_destroy.setToolTip(_("Stop processing and close the application"))
        self.btn_destroy.clicked.connect(lambda: self._destroy_cb())

        self.btn_preview = QPushButton(_("Preview"))
        self.btn_preview.setObjectName("secondary")
        self.btn_preview.setToolTip(_("Show/hide a preview of the processed output"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        row.addWidget(self.btn_start)
        row.addWidget(self.btn_destroy)
        row.addWidget(self.btn_preview)
        return row

    # ── camera card ──────────────────────────────────────────────────────

    def _build_camera_card(self) -> QGroupBox:
        card = QGroupBox(_("Camera"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        # Row 0: camera selector + Live button
        grid.addWidget(QLabel(_("Select Camera:")), 0, 0)
        self._camera_indices, self._camera_names = get_available_cameras()

        self.cb_camera = QComboBox()
        if not self._camera_names or self._camera_names[0] == "No cameras found":
            self.cb_camera.addItem("No cameras found")
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            self.cb_camera.addItems(self._camera_names)
            cam_ok = True
        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        grid.addWidget(self.cb_camera, 0, 1)

        self.btn_live = QPushButton(_("Live"))
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)
        grid.addWidget(self.btn_live, 0, 2)

        # Row 1: capture resolution
        grid.addWidget(QLabel(_("Resolution:")), 1, 0)
        self.cb_resolution = QComboBox()
        # 640x480 is native on virtually every webcam and the safest 30fps
        # mode on USB 2.0. Higher tiers are 16:9 and useful when the camera
        # supports them natively (DSLR + capture card, USB 3.0 webcams).
        self._resolution_options = [
            ("640 x 480", (640, 480)),
            ("960 x 540 (qHD)", (960, 540)),
            ("1280 x 720 (HD)", (1280, 720)),
            ("1920 x 1080 (FHD)", (1920, 1080)),
            ("2560 x 1440 (QHD)", (2560, 1440)),
            ("3840 x 2160 (4K)", (3840, 2160)),
        ]
        self.cb_resolution.currentIndexChanged.connect(self._on_resolution_change)
        self.refresh_resolution_combo()
        self.cb_resolution.setToolTip(_(
            "Requested webcam resolution. Camera may negotiate to its "
            "nearest supported size — actual size printed in console as "
            "'[VideoCapturer] WxH @ FPS'. Applies on next Live start.\n\n"
            "640x480 is native on virtually every webcam and the safest "
            "30fps choice. qHD often works too. HD/FHD typically drop to "
            "~10fps on USB 2.0 webcams due to isochronous bandwidth limits."
        ))
        grid.addWidget(self.cb_resolution, 1, 1, 1, 2)

        # Row 2: face detection size
        grid.addWidget(QLabel(_("Det size:")), 2, 0)
        self.cb_det_size = QComboBox()
        self._det_size_options = [160, 320, 640]
        for v in self._det_size_options:
            self.cb_det_size.addItem(f"{v} x {v}")
        cur_det = int(getattr(modules.globals, 'det_size', modules.globals.DEFAULT_DET_SIZE))
        # Normalize to a valid option first so .index() can never raise: fall back
        # to the default size, then to the last (highest) option if even that is
        # somehow missing.
        if cur_det not in self._det_size_options:
            cur_det = (modules.globals.DEFAULT_DET_SIZE
                       if modules.globals.DEFAULT_DET_SIZE in self._det_size_options
                       else self._det_size_options[-1])
        idx = self._det_size_options.index(cur_det)
        self.cb_det_size.setCurrentIndex(idx)
        self.cb_det_size.currentIndexChanged.connect(self._on_det_size_change)
        self.cb_det_size.setToolTip(_(
            "Face detection input resolution. Lower = faster, less accurate at "
            "distance. Changes take effect on next face analyser init."
        ))
        grid.addWidget(self.cb_det_size, 2, 1, 1, 2)

        # Row 3: capture frame rate
        grid.addWidget(QLabel(_("Camera FPS:")), 3, 0)
        self.cb_fps = QComboBox()
        self._fps_options = [15, 24, 30, 60]
        for v in self._fps_options:
            self.cb_fps.addItem(f"{v} fps")
        fps_now = int(modules.globals.capture_fps)
        self.cb_fps.setCurrentIndex(self._fps_options.index(fps_now) if fps_now in self._fps_options else 3)
        self.cb_fps.currentIndexChanged.connect(self._on_fps_change)
        self.cb_fps.setToolTip(_(
            "Frame rate requested from the camera (applies on next Live start). "
            "The camera may deliver less; the real value is printed in the console."
        ))
        grid.addWidget(self.cb_fps, 3, 1, 1, 2)

        # Row 4: virtual camera
        self.sw_vcam = _Switch(_("Virtual camera (OBS)"), modules.globals.virtual_camera, _(
            "Send the processed picture to the 'OBS Virtual Camera' device, so OBS, "
            "Discord, Zoom or a browser can use it as a webcam. Needs OBS Studio installed; "
            "do not start OBS's own virtual camera at the same time."
        ))
        self.sw_vcam.setEnabled(virtual_camera.available())
        self.sw_vcam.toggled.connect(self._on_vcam_toggled)
        grid.addWidget(self.sw_vcam, 4, 0, 1, 3)

        grid.setColumnStretch(1, 1)
        return card

    def refresh_resolution_combo(self) -> None:
        """Presets plus a 'Custom' entry when the saved size is not a preset."""
        self.cb_resolution.blockSignals(True)
        self.cb_resolution.clear()
        for label, _wh in self._resolution_options:
            self.cb_resolution.addItem(label)
        cur = tuple(modules.globals.capture_resolution)
        idx = next((i for i, (_l, wh) in enumerate(self._resolution_options) if wh == cur), None)
        if idx is None:
            self.cb_resolution.addItem(_("Custom:") + f" {cur[0]} x {cur[1]}")
            idx = len(self._resolution_options)
        self.cb_resolution.setCurrentIndex(idx)
        self.cb_resolution.blockSignals(False)

    def _on_fps_change(self, idx: int) -> None:
        if 0 <= idx < len(self._fps_options):
            modules.globals.capture_fps = self._fps_options[idx]
            save_switch_states()

    def _on_vcam_toggled(self, value: bool) -> None:
        modules.globals.virtual_camera = value
        save_switch_states()
        if not value:
            virtual_camera.stop()
        self.refresh_feature_summary()

    def _on_swapper_change(self, idx: int) -> None:
        key = self.cb_swapper.itemData(idx)
        if not key or key == modules.globals.swapper_model:
            return
        modules.globals.swapper_model = key
        save_switch_states()
        from modules.processors.frame.face_swapper import reset_face_swapper
        reset_face_swapper()
        update_status(
            _("Swap model selected (loads on next use, first time downloads/builds):")
            + f" {swapper_models.spec_for(key).label}"
        )

    def _on_resolution_change(self, idx: int) -> None:
        if 0 <= idx < len(self._resolution_options):
            _label, wh = self._resolution_options[idx]
            modules.globals.capture_resolution = wh
            save_switch_states()
            self.refresh_resolution_combo()  # drops a stale "Custom" entry
            # Pre-translate the static template; append the dynamic size after
            # so the message still participates in localization (update_status
            # only translates whole-string keys).
            update_status(
                _("Capture resolution set (applies on next Live start):")
                + f" {wh[0]}x{wh[1]}"
            )

    def _on_det_size_change(self, idx: int) -> None:
        if 0 <= idx < len(self._det_size_options):
            v = self._det_size_options[idx]
            modules.globals.det_size = v
            reset_face_analyser()
            save_switch_states()
            update_status(
                _("Face detection size set (analyser re-inits on next call):")
                + f" {v}x{v}"
            )

    def refresh_feature_summary(self) -> None:
        g = modules.globals
        on = []
        if g.det_interval:
            on.append(_("Detect every") + f" {g.det_interval}")
        if g.tracking_v2:
            on.append(_("Tracking 2.0") + (" + Optical Flow" if g.tracking_optical_flow else ""))
        if g.freeze_on_face_lost:
            on.append(_("Freeze on loss"))
        for flag, label in (
            ("adaptive_face_mask", "Adaptive mask"), ("color_matching", "Color matching"),
            ("temporal_stabilization", "Temporal stabilization"),
            ("eye_preservation", "Eyes"), ("mouth_preservation", "Mouth"),
            ("grain_matching", "Grain"),
            ("dynamic_enhancement", "Dynamic enhancement"),
        ):
            if getattr(g, flag):
                on.append(_(label))
        if g.use_tensorrt:
            on.append("TensorRT" + (" FP16" if g.trt_fp16 else ""))
        if g.face_parts_mask:
            on.append(_("Face parts"))
        if g.occlusion_mask:
            on.append(_("Occlusion"))
        if g.use_model_mask:
            on.append(_("Model mask"))
        if g.virtual_camera:
            on.append(_("Virtual camera"))
        if g.face_fader:
            on.append(_("Face Fader"))
        if g.profile_fade:
            on.append(_("Profile fade"))
        if g.upscaler:
            on.append(_("Upscaler") + (" " + _("(face)") if g.upscaler_mode == "face" else ""))
        if g.frame_interpolation:
            on.append(_("Frame interpolation"))
        if g.multi_source and g.source_paths_extra:
            on.append(_("Multi-photo source") + f" ({1 + len(g.source_paths_extra)})")
        if g.async_detection:
            on.append(_("Async detection"))
        if g.reprojection:
            on.append(_("Reprojection"))
        self._feature_summary.setText(
            (_("Active:") + " " + " · ".join(on)) if on
            else _("Advanced features: all off (standard mode)")
        )

    def start_update_check(self) -> None:
        if not modules.globals.check_updates or _BRIDGE is None:
            return
        from modules import update_check
        _BRIDGE.updateAvailable.connect(self._on_update_available)
        update_check.check_async(_BRIDGE.updateAvailable.emit)

    def _on_update_available(self, release: dict) -> None:
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle(_("Update available"))
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(_("A new version of Deep-Boost-Cam is available:") + f" {release['tag']}"
                    + "\n" + _("Your version:") + f" {modules.metadata.version}")
        notes = (release.get("notes") or "").strip()
        if notes:
            box.setDetailedText(notes[:4000])
        b_install = box.addButton(_("Install"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(_("Later"), QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is b_install:
            webbrowser.open(release.get("url") or "https://github.com/" + modules.metadata.repo)

    def _on_open_dfm_studio(self) -> None:
        from modules.ui_dfm_studio import DFMStudioDialog
        if getattr(self, "_dfm_dialog", None) is None:
            self._dfm_dialog = DFMStudioDialog(self)
        self._dfm_dialog.show()
        self._dfm_dialog.raise_()

    def refresh_swapper_combo(self) -> None:
        """Re-list swap models (new .dfm files appear in models/dfm)."""
        swapper_models.refresh_dfm()
        self.cb_swapper.blockSignals(True)
        self.cb_swapper.clear()
        for spec in swapper_models.SWAPPERS:
            self.cb_swapper.addItem(spec.label, spec.key)
        self.cb_swapper.setCurrentIndex(max(0, self.cb_swapper.findData(modules.globals.swapper_model)))
        self.cb_swapper.blockSignals(False)

    def _on_open_settings(self) -> None:
        dlg = SettingsDialog(self)
        dlg.exec()
        self.sw_vcam.blockSignals(True)
        self.sw_vcam.setChecked(modules.globals.virtual_camera)
        self.sw_vcam.blockSignals(False)
        self.cb_swapper.blockSignals(True)
        self.cb_swapper.setCurrentIndex(max(0, self.cb_swapper.findData(modules.globals.swapper_model)))
        self.cb_swapper.blockSignals(False)
        self.refresh_resolution_combo()
        self.refresh_feature_summary()

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_select_source(self) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if path and is_image(path):
            modules.globals.source_path = path
            _RECENT_SOURCE_DIR = os.path.dirname(path)
            self.source_label.setPixmap(render_image_preview(path, (200, 200)))
            self.source_label.setText("")
        elif not path:
            return
        else:
            modules.globals.source_path = None
            self.source_label.clear()
            self.source_label.setText(_("Source face"))

    def _on_select_target(self) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            _MEDIA_FILE_FILTER,
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(render_image_preview(path, (200, 200)))
            self.target_label.setText("")
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, (200, 200))
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText(_("Target"))

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/random-person.jpeg",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"expected an image, got {content_type!r}")
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            staging_path = f"{temp_path}.part"
            # Download into a staging file and validate there. Writing straight
            # to temp_path would destroy the image currently in use whenever a
            # later fetch comes back bad, and content-type alone does not prove
            # the bytes decode.
            try:
                with open(staging_path, "wb") as f:
                    f.write(response.content)
                pixmap = render_image_preview(staging_path, (200, 200))
                if imread_unicode(staging_path) is None:
                    raise ValueError("downloaded image could not be decoded")
                os.replace(staging_path, temp_path)
            except Exception:
                try:
                    os.remove(staging_path)
                except OSError:
                    pass
                raise
            modules.globals.source_path = temp_path
            self.source_label.setPixmap(pixmap)
            self.source_label.setText("")
        except Exception as exc:
            print(f"Failed to fetch random face: {exc}")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        self.source_label.setPixmap(render_image_preview(tp, (200, 200)))
        self.target_label.setPixmap(render_image_preview(sp, (200, 200)))
        self.source_label.setText("")
        self.target_label.setText("")

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_enhancer_change(self, choice: str) -> None:
        key_map = {
            "None": None,
            "GFPGAN": "face_enhancer",
            "GPEN-512": "face_enhancer_gpen512",
            "GPEN-256": "face_enhancer_gpen256",
        }
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            _update_tumbler(key, False)
        selected = key_map.get(choice)
        if selected:
            _update_tumbler(selected, True)
        save_switch_states()

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        pct = int(value * 100)
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        update_status(f"Sharpness set to {value:.1f}")

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                _IMAGE_FILE_FILTER,
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                _VIDEO_FILE_FILTER,
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()

    def _on_live(self) -> None:
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        if not modules.globals.map_faces:
            if (modules.globals.source_path is None
                    and swapper_models.needs_source(modules.globals.swapper_model)):
                update_status("Please select a source image first")
                return
            from modules.face_analyser import get_face_analyser
            from modules.processors.frame.face_swapper import get_face_swapper
            if modules.globals.use_tensorrt:
                update_status(
                    "TensorRT: loading engines. The first run builds them "
                    "(several minutes, window may not respond); later runs load from cache."
                )
            get_face_analyser()
            get_face_swapper()
            _open_webcam_preview(camera_index)
        else:
            modules.globals.source_target_map = []
            _open_live_mapper_dialog(camera_index, modules.globals.source_target_map)

    def closeEvent(self, event):
        # Treat OS-level close as Destroy click
        self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── advanced settings dialog ───────────────────────────────────────────


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("hint")
    label.setWordWrap(True)
    return label


def _dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total / (1 << 20)


# Measured on RTX 5060 Ti; filled in from the benchmark (see notes).
UPSCALER_TIMINGS = (
    "Measured in Live on RTX 5060 Ti, TensorRT FP16, 720p camera, in its own thread (does not s"
    "low the swap down): Face only ~20 ms per large face (model itself 4 ms SPAN / Clear Realit"
    "y, 5 ms ESRGAN compact, 24 ms ESRGAN x2); Whole frame ~19 ms SPAN / Clear Reality, ~28 ms "
    "ESRGAN compact, ESRGAN x2 ~255 ms (not for Live)."
)


class _EngineBuilder(QThread):
    progress = Signal(int, str)
    finished_all = Signal(str)

    def __init__(self, jobs):
        super().__init__()
        self._jobs = jobs

    def run(self) -> None:
        errors = []
        for i, (label, fn) in enumerate(self._jobs):
            self.progress.emit(i, _("Building") + f" {i + 1}/{len(self._jobs)}: {label} ...")
            t0 = time.time()
            try:
                fn()
                print(f"[trt-prebuild] {label}: {time.time() - t0:.0f}s", flush=True)
            except Exception as error:
                errors.append(f"{label}: {str(error)[:120]}")
        self.progress.emit(len(self._jobs), "")
        self.finished_all.emit(
            _("Done.") + (" " + _("Errors:") + " " + "; ".join(errors) if errors else
                          " " + _("Engines are cached in trt_cache.")))


class SettingsDialog(QDialog):
    """Tabbed advanced settings. Changes apply live and persist immediately;
    TensorRT options apply on the next start (sessions are built at load)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("Advanced settings"))
        self.resize(920, 800)
        self._controls: dict = {}       # field -> (widget, setter)
        self._deps: list = []           # (master field, [widgets])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        tabs = QTabWidget()
        # "&" is a Qt mnemonic marker in tab titles; escape it.
        tabs.addTab(self._page(self._build_tracking), _("Detection & Tracking").replace("&", "&&"))
        tabs.addTab(self._page(self._build_realism), _("Realism"))
        tabs.addTab(self._page(self._build_masks), _("Model & masks").replace("&", "&&"))
        tabs.addTab(self._page(self._build_faces), _("Faces"))
        tabs.addTab(self._page(self._build_output), _("Camera & output").replace("&", "&&"))
        tabs.addTab(self._page(self._build_accel), _("Acceleration"))
        layout.addWidget(tabs, 1)

        buttons = QHBoxLayout()
        btn_reset = QPushButton(_("Reset to defaults"))
        btn_reset.setObjectName("secondary")
        btn_reset.setToolTip(_("Turn every advanced feature off and restore default values"))
        btn_reset.clicked.connect(self._reset_defaults)
        btn_close = QPushButton(_("Close"))
        btn_close.clicked.connect(self.accept)
        buttons.addWidget(btn_reset)
        buttons.addStretch(1)
        buttons.addWidget(btn_close)
        layout.addLayout(buttons)
        self._sync_deps()

    # ── generic builders ────────────────────────────────────────────────
    def _page(self, build) -> QScrollArea:
        inner = QWidget()
        inner.setObjectName("settingsPage")
        v = QVBoxLayout(inner)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(10)
        build(v)
        v.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        return scroll

    def _set(self, field: str, value) -> None:
        setattr(modules.globals, field, value)
        save_switch_states()
        if field == "virtual_camera" and not value:
            virtual_camera.stop()
        if field.startswith(("color_", "temporal_", "eye_", "mouth_", "adaptive_", "grain_")):
            realism.reset_state()
        if field == "multi_source":
            from modules import source_identity
            source_identity.reset()
        self._sync_deps()

    def _card(self, parent: QVBoxLayout, title: str, hint: str = "") -> QGridLayout:
        card = QGroupBox(title)
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        if hint:
            grid.addWidget(_hint(hint), 0, 0, 1, 3)
        parent.addWidget(card)
        return grid

    def _check(self, grid: QGridLayout, field: str, label: str, tip: str) -> QCheckBox:
        cb = QCheckBox(label)
        cb.setChecked(bool(getattr(modules.globals, field)))
        cb.setToolTip(tip)
        cb.toggled.connect(lambda v, f=field: self._set(f, bool(v)))
        row = grid.rowCount()
        grid.addWidget(cb, row, 0, 1, 3)
        if tip:
            grid.addWidget(_hint(tip), row + 1, 0, 1, 3)
        self._controls[field] = (cb, cb.setChecked)
        return cb

    def _slider(self, grid: QGridLayout, field: str, label: str, lo: float, hi: float,
                denom: int, fmt: Callable[[float], str], tip: str) -> list:
        row = grid.rowCount()
        name = QLabel(label)
        name.setToolTip(tip)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(int(round(lo * denom)), int(round(hi * denom)))
        s.setValue(int(round(float(getattr(modules.globals, field)) * denom)))
        s.setToolTip(tip)
        val = QLabel(fmt(s.value() / denom))
        val.setObjectName("valueLabel")
        is_int = isinstance(_EXT_DEFAULTS[field], int)

        def changed(iv, f=field):
            v = iv / denom
            val.setText(fmt(v))
            self._set(f, int(round(v)) if is_int else float(v))

        s.valueChanged.connect(changed)
        grid.addWidget(name, row, 0)
        grid.addWidget(s, row, 1)
        grid.addWidget(val, row, 2)
        if tip:
            grid.addWidget(_hint(tip), row + 1, 0, 1, 3)
        self._controls[field] = (s, lambda v, s=s: s.setValue(int(round(float(v) * denom))))
        return [name, s, val]

    def _spin(self, grid: QGridLayout, field: str, label: str, lo: int, hi: int,
              suffix: str, tip: str, special: str = "") -> list:
        row = grid.rowCount()
        name = QLabel(label)
        name.setToolTip(tip)
        sp = QSpinBox()
        sp.setRange(lo, hi)
        if suffix:
            sp.setSuffix(suffix)
        if special:
            sp.setSpecialValueText(special)
        sp.setValue(int(getattr(modules.globals, field)))
        sp.setToolTip(tip)
        sp.valueChanged.connect(lambda v, f=field: self._set(f, int(v)))
        grid.addWidget(name, row, 0)
        grid.addWidget(sp, row, 1, 1, 2)
        if tip:
            grid.addWidget(_hint(tip), row + 1, 0, 1, 3)
        self._controls[field] = (sp, lambda v, sp=sp: sp.setValue(int(v)))
        return [name, sp]

    def _depends(self, master: str, widgets: list) -> None:
        self._deps.append((master, widgets))

    def _sync_deps(self) -> None:
        for master, widgets in self._deps:
            on = bool(getattr(modules.globals, master))
            for w in widgets:
                w.setEnabled(on)

    def _reset_defaults(self) -> None:
        old_model = modules.globals.swapper_model
        for field, default in _EXT_DEFAULTS.items():
            value = dict(default) if isinstance(default, dict) else default
            ctrl = self._controls.get(field)
            if ctrl is not None:
                ctrl[1](value)
            setattr(modules.globals, field, value)
        for region, cb in getattr(self, "_part_boxes", {}).items():
            cb.setChecked(modules.globals.face_parts.get(region, True))
        save_switch_states()
        realism.reset_state()
        virtual_camera.stop()
        if modules.globals.swapper_model != old_model:
            from modules.processors.frame.face_swapper import reset_face_swapper
            reset_face_swapper()
        self._sync_deps()
        update_status("Advanced settings reset to defaults")

    # ── tab: detection & tracking ───────────────────────────────────────
    def _build_tracking(self, v: QVBoxLayout) -> None:
        pct = lambda x: f"{x * 100:.0f}%"

        g = self._card(v, _("Face detection"), _(
            "How often the neural face detector runs in Live mode. Between detections "
            "the last position is reused (or tracked, with Face Tracking 2.0)."
        ))
        self._spin(g, "det_interval", _("Detect every"), 0, 30, " " + _("frames"), _(
            "1 = every frame: most accurate, highest GPU load. Higher = faster but the "
            "mask reacts later. Auto = about every 40 ms of camera time "
            "(1 frame at 30 fps, 2 at 60 fps)."
        ), special=_("Auto"))

        g = self._card(v, _("Face Tracking 2.0"), _(
            "Keeps the mask glued to the face between detections and through short "
            "detector misses. Off = standard tracking (hold the last detection)."
        ))
        self._check(g, "tracking_v2", _("Enable Face Tracking 2.0"), "")
        dep = []
        dep.append(self._check(g, "tracking_optical_flow", _("Optical Flow tracking"), _(
            "Follow face feature points with optical flow so the mask moves every frame, "
            "not only on detection frames. Off = hold the last detection."
        )))
        dep.append(self._check(g, "tracking_smoothing", _("Smoothing (One Euro filter)"), _(
            "Removes coordinate jitter while still following fast moves."
        )))
        dep += self._slider(g, "tracking_smoothing_strength", _("Smoothing strength"),
                            0.0, 1.0, 100, pct, _("Higher = calmer mask, slightly more lag."))
        dep += self._spin(g, "tracking_grace_frames", _("Grace period"), 0, 30,
                          " " + _("frames"), _(
            "How many frames the mask is kept when the face is lost. Limited on "
            "purpose: after this the mask is dropped so it cannot freeze on the wrong spot."
        ))
        dep += self._slider(g, "tracking_max_motion", _("Max motion per frame"),
                            0.05, 0.6, 100, pct, _(
            "Tracked steps larger than this (share of face size) are rejected as "
            "errors and trigger re-detection."
        ))
        dep += self._slider(g, "tracking_max_rotation", _("Max head tilt per frame"),
                            5, 45, 1, lambda x: f"{x:.0f}°", _(
            "Pose guard: larger in-plane rotation per frame is treated as a tracking error."
        ))
        dep += self._slider(g, "tracking_min_confidence", _("Face confidence threshold"),
                            0.3, 0.95, 100, lambda x: f"{x:.2f}", _(
            "Detections below this score are treated as a lost face."
        ))
        dep.append(self._check(g, "tracking_auto_redetect", _("Auto re-detection"), _(
            "When tracking fails, run the detector on the very next frame instead of "
            "waiting for the detection interval."
        )))
        self._depends("tracking_v2", dep)

        g = self._card(v, _("Pipeline and latency"), _(
            "Two ways to make Live smoother. They can be combined."
        ))
        self._check(g, "async_detection", _("Parallel detection (pipeline)"), _(
            "Face detection runs in its own thread while the previous frame is being "
            "swapped; the result is carried to the current frame with optical flow. "
            "Measured: +12% FPS on CUDA with a heavy pipeline; little gain with TensorRT "
            "(the detector is already ~2 ms). Applies immediately."
        ))
        self._check(g, "reprojection", _("Low-latency reprojection"), _(
            "The picture is shown at camera rate: the last finished swap is moved onto "
            "the newest camera frame by a tracker, so a slow pipeline delays the "
            "expression, not the position. Measured with HyperSwap+GPEN on CUDA: 15 -> 29 "
            "shown FPS, face position lag 26 -> 8 px. Single face only. Applies on next Live start."
        ))

        g = self._card(v, _("When the face is lost"))
        self._check(g, "freeze_on_face_lost", _("Freeze picture (never show the real face)"), _(
            "If no face is swapped on a frame, the camera output freezes on the last "
            "swapped frame (black before the first one) instead of showing your real face."
        ))

    # ── tab: realism ────────────────────────────────────────────────────
    def _build_realism(self, v: QVBoxLayout) -> None:
        pct = lambda x: f"{x * 100:.0f}%"
        v.addWidget(_hint(_(
            "Algorithm-level improvements of the swap itself. All are off by default; "
            "each costs well under 1 ms per face (they work on the 128x128 aligned face). "
            "Features marked * need face landmarks (one small extra model per detection)."
        )))

        g = self._card(v, _("Adaptive Face Mask") + " *",
                       _("Blend mask follows the face contour: less visible borders."))
        self._check(g, "adaptive_face_mask", _("Enable"), "")
        self._depends("adaptive_face_mask", self._slider(
            g, "adaptive_mask_feather", _("Edge softness"), 0.0, 1.0, 100, pct,
            _("Higher = softer, wider transition into the real skin.")))

        g = self._card(v, _("Color Matching"),
                       _("Matches brightness and skin tone of the swap to your frame."))
        self._check(g, "color_matching", _("Enable"), "")
        self._depends("color_matching", self._slider(
            g, "color_matching_strength", _("Strength"), 0.0, 1.0, 100, pct, ""))

        g = self._card(v, _("Temporal Stabilization"), _(
            "Smart smoothing between frames: less flicker. Backs off automatically "
            "while you talk or move, so there is no ghosting."))
        self._check(g, "temporal_stabilization", _("Enable"), "")
        self._depends("temporal_stabilization", self._slider(
            g, "temporal_strength", _("Strength"), 0.0, 0.9, 100, pct, ""))

        g = self._card(v, _("Eye Preservation") + " *", _(
            "Keeps part of your real eyelids and eyes: natural blinking and gaze. "
            "Too high leaks your own eye shape."))
        self._check(g, "eye_preservation", _("Enable"), "")
        self._depends("eye_preservation", self._slider(
            g, "eye_preservation_strength", _("Strength"), 0.0, 1.0, 100, pct, ""))

        g = self._card(v, _("Mouth Preservation") + " *", _(
            "Feathered mask of your real lips: more natural speech. Independent of the "
            "Mouth Mask slider on the main window."))
        self._check(g, "mouth_preservation", _("Enable"), "")
        self._depends("mouth_preservation", self._slider(
            g, "mouth_preservation_strength", _("Strength"), 0.0, 1.0, 100, pct, ""))

        g = self._card(v, _("Grain & sharpness matching"), _(
            "Swap models draw a clean, smooth face; next to a noisy webcam picture that "
            "smoothness gives the swap away. Measures the camera's noise and detail on your "
            "real face every frame and gives the swapped face the same: sharpens or softens "
            "it and adds the missing sensor grain. ~2 ms for a typical face (8 ms if it fills a 720p frame)."))
        self._check(g, "grain_matching", _("Enable"), "")
        dep = self._slider(g, "grain_amount", _("Grain"), 0.0, 1.5, 100, pct,
                           _("100% = same noise level as the camera; more = grainier."))
        dep += self._slider(g, "grain_sharpness_amount", _("Sharpness matching"), 0.0, 1.0, 100, pct,
                            _("How strongly detail is pulled to the camera's level."))
        self._depends("grain_matching", dep)

        g = self._card(v, _("Dynamic Enhancement"), _(
            "Runs the selected Face Enhancer only when it helps: skipped for small faces "
            "and during fast motion, so less GPU load. Needs a Face Enhancer selected."))
        self._check(g, "dynamic_enhancement", _("Enable"), "")
        dep = self._spin(g, "dynamic_enhancement_min_face", _("Min face height"),
                         40, 600, " px", _("Smaller faces skip enhancement."))
        dep += self._slider(g, "dynamic_enhancement_max_motion", _("Skip above motion"),
                            0.01, 0.3, 100, pct, _(
            "Per-frame motion (share of face size) above which enhancement is skipped. "
            "Motion is measured by Face Tracking 2.0."))
        self._depends("dynamic_enhancement", dep)

    # ── tab: model & masks ──────────────────────────────────────────────
    def _build_masks(self, v: QVBoxLayout) -> None:
        spec = swapper_models.spec_for(modules.globals.swapper_model)
        v.addWidget(_hint(
            _("Swap model is chosen on the main window (Options card). Current:") + f" {spec.label}"
        ))

        g = self._card(v, _("Several source photos"), _(
            "Add more photos of the same person (front, turned, up, down). Their identities "
            "are blended, weighted by your current head pose: the face stays steadier and "
            "turned heads use the matching photo. The main photo is always included. "
            "Not used by DFM models (their identity is built in)."
        ))
        self._check(g, "multi_source", _("Use several source photos"), "")
        self._src_list = QListWidget()
        self._src_list.setMaximumHeight(110)
        self._src_list.addItems(modules.globals.source_paths_extra)
        g.addWidget(self._src_list, g.rowCount(), 0, 1, 3)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        b_add = QPushButton(_("Add photos"))
        b_add.setObjectName("secondary")
        b_add.clicked.connect(self._add_source_photos)
        b_del = QPushButton(_("Remove selected"))
        b_del.setObjectName("secondary")
        b_del.clicked.connect(self._remove_source_photo)
        b_clr = QPushButton(_("Clear"))
        b_clr.setObjectName("danger")
        b_clr.clicked.connect(self._clear_source_photos)
        for b in (b_add, b_del, b_clr):
            row.addWidget(b)
        row.addStretch(1)
        holder = QWidget()
        holder.setLayout(row)
        g.addWidget(holder, g.rowCount(), 0, 1, 3)
        self._depends("multi_source", [self._src_list, holder])

        g = self._card(v, _("Model face mask"), _(
            "HyperSwap models output their own face mask. Using it trims the swap to "
            "the face the model actually generated. No effect on Inswapper/ReSwapper."
        ))
        self._check(g, "use_model_mask", _("Use the model's own mask"), "")

        g = self._card(v, _("Face parts"), _(
            "Choose which parts of the face are swapped; unchecked parts keep your real "
            "face. Uses a face parser (BiSeNet, ~94 MB, downloads once). Hair, ears and "
            "background are never painted, so a fringe over the forehead stays real."
        ))
        self._check(g, "face_parts_mask", _("Enable face part masks"), "")
        names = {
            "forehead": _("Forehead"), "brows": _("Eyebrows"), "eyes": _("Eyes"),
            "nose": _("Nose"), "cheeks": _("Cheeks"), "mouth": _("Mouth / lips"),
            "chin": _("Chin"),
        }
        holder = QWidget()
        grid_parts = QGridLayout(holder)
        grid_parts.setContentsMargins(0, 0, 0, 0)
        self._part_boxes = {}
        for i, region in enumerate(names):
            cb = QCheckBox(names[region])
            cb.setChecked(modules.globals.face_parts.get(region, True))
            cb.toggled.connect(lambda val, r=region: self._set_part(r, val))
            grid_parts.addWidget(cb, i // 2, i % 2)
            self._part_boxes[region] = cb
        g.addWidget(holder, g.rowCount(), 0, 1, 3)
        dep = [holder]
        dep += self._slider(g, "face_parts_feather", _("Edge softness"), 0.0, 1.0, 100,
                            lambda x: f"{x * 100:.0f}%", _("Softer borders between swapped and real parts."))
        self._depends("face_parts_mask", dep)

        g = self._card(v, _("Occlusion"), _(
            "Keeps hands, a microphone, a cup or glasses held in front of the face real "
            "instead of painting the swap over them. Uses XSeg (~70 MB, downloads once)."
        ))
        self._check(g, "occlusion_mask", _("Enable occlusion mask"), "")

        g = self._card(v, _("Mask performance"))
        self._spin(g, "mask_interval", _("Update masks every"), 1, 10, " " + _("frames"), _(
            "1 = every frame (hands are followed instantly). 2-3 roughly halves the cost "
            "of the face-part and occlusion masks; the face stays aligned in the crop, so "
            "a mask a frame old still fits. Single-face mode only."
        ))

    def _store_source_photos(self) -> None:
        from modules import source_identity
        modules.globals.source_paths_extra = [
            self._src_list.item(i).text() for i in range(self._src_list.count())]
        save_switch_states()
        source_identity.reset()

    def _add_source_photos(self) -> None:
        paths, _f = QFileDialog.getOpenFileNames(self, _("Add photos"), _RECENT_SOURCE_DIR or "",
                                                 _IMAGE_FILE_FILTER)
        have = set(modules.globals.source_paths_extra)
        for p in paths:
            if p not in have:
                self._src_list.addItem(p)
        self._store_source_photos()

    def _remove_source_photo(self) -> None:
        for item in self._src_list.selectedItems():
            self._src_list.takeItem(self._src_list.row(item))
        self._store_source_photos()

    def _clear_source_photos(self) -> None:
        self._src_list.clear()
        self._store_source_photos()

    def _set_part(self, region: str, value: bool) -> None:
        parts = dict(modules.globals.face_parts)
        parts[region] = bool(value)
        modules.globals.face_parts = parts
        save_switch_states()

    # ── tab: faces ──────────────────────────────────────────────────────
    def _build_faces(self, v: QVBoxLayout) -> None:
        g = self._card(v, _("Face Fader"), _(
            "Smooth transitions: the 'Swap: ON/OFF' button fades the swap in and out, and "
            "choosing another face (photo or library set) morphs into it instead of jumping. "
            "DFM models fade in/out only (their face is built in)."))
        self._check(g, "face_fader", _("Enable Face Fader"), "")
        self._depends("face_fader", self._slider(
            g, "fader_seconds", _("Transition time"), 0.2, 5.0, 10, lambda x: f"{x:.1f} s", ""))

        g = self._card(v, _("Profile fade"), _(
            "Swap models break down when the head turns far: a smeared half-face appears. "
            "The real head angle (3D landmark model, ~4 ms on CUDA, faster with TensorRT) "
            "smoothly lowers the swap between the start and end angles."))
        self._check(g, "profile_fade", _("Enable profile fade"), "")
        deg = lambda x: f"{x:.0f}°"
        dep = self._slider(g, "profile_yaw_start", _("Turn: start fading"), 10, 80, 1, deg, "")
        dep += self._slider(g, "profile_yaw_end", _("Turn: fully off"), 20, 90, 1, deg, "")
        dep += self._slider(g, "profile_pitch_start", _("Up/down: start fading"), 10, 70, 1, deg, "")
        dep += self._slider(g, "profile_pitch_end", _("Up/down: fully off"), 20, 90, 1, deg, "")
        self._depends("profile_fade", dep)

        g = self._card(v, _("Face library"), _(
            "Saved face sets live in face_library/ inside the program folder. Save and pick "
            "them on the main window (row under the photos); here you can rename or delete."))
        self._lib_list = QListWidget()
        self._lib_list.setMaximumHeight(130)
        self._lib_list.addItems(face_library.list_sets())
        g.addWidget(self._lib_list, g.rowCount(), 0, 1, 3)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        b_ren = QPushButton(_("Rename"))
        b_ren.setObjectName("secondary")
        b_ren.clicked.connect(self._lib_rename)
        b_del = QPushButton(_("Delete"))
        b_del.setObjectName("danger")
        b_del.clicked.connect(self._lib_delete)
        b_open = QPushButton(_("Open folder"))
        b_open.setObjectName("secondary")
        b_open.clicked.connect(lambda: (os.makedirs(face_library.LIB_DIR, exist_ok=True),
                                        os.startfile(face_library.LIB_DIR)))
        for b in (b_ren, b_del, b_open):
            row.addWidget(b)
        row.addStretch(1)
        holder = QWidget()
        holder.setLayout(row)
        g.addWidget(holder, g.rowCount(), 0, 1, 3)

    def _lib_changed(self) -> None:
        self._lib_list.clear()
        self._lib_list.addItems(face_library.list_sets())
        parent = self.parent()
        if hasattr(parent, "refresh_library"):
            parent.refresh_library()

    def _lib_rename(self) -> None:
        item = self._lib_list.currentItem()
        if item is None:
            return
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, _("Rename"), _("New name:"), text=item.text())
        name = face_library.safe_name(name) if ok else ""
        if name and name != item.text() and name not in face_library.list_sets():
            os.replace(os.path.join(face_library.LIB_DIR, item.text()),
                       os.path.join(face_library.LIB_DIR, name))
            self._lib_changed()

    def _lib_delete(self) -> None:
        item = self._lib_list.currentItem()
        if item is not None:
            face_library.delete_set(item.text())
            self._lib_changed()

    # ── tab: camera & output ────────────────────────────────────────────
    def _build_output(self, v: QVBoxLayout) -> None:
        from modules.processors.frame import upscaler
        g = self._card(v, _("Upscaler (x2)"), _(
            "Doubles the output resolution with an AI super-resolution model (TensorRT when "
            "enabled). 'Face only' upscales just the face region (cheap); 'Whole frame' "
            "upscales everything (costly at 720p and above). Applies to the preview and the "
            "virtual camera."))
        self._check(g, "upscaler", _("Enable upscaler"), "")
        dep = []
        for field, label, items in (
            ("upscaler_mode", _("Mode"), [("face", _("Face only")), ("frame", _("Whole frame"))]),
            ("upscaler_model", _("Model"), [(k, _(m.label)) for k, m in upscaler.MODELS.items()]),
        ):
            row = g.rowCount()
            lbl = QLabel(label)
            combo = QComboBox()
            for key, text in items:
                combo.addItem(text, key)
            combo.setCurrentIndex(max(0, combo.findData(getattr(modules.globals, field))))
            combo.currentIndexChanged.connect(
                lambda i, f=field, c=combo: self._set(f, c.itemData(i)))
            g.addWidget(lbl, row, 0)
            g.addWidget(combo, row, 1, 1, 2)
            self._controls[field] = (combo, lambda val, c=combo: c.setCurrentIndex(max(0, c.findData(val))))
            dep += [lbl, combo]
        g.addWidget(_hint(_(UPSCALER_TIMINGS)), g.rowCount(), 0, 1, 3)
        self._depends("upscaler", dep)

        g = self._card(v, _("Frame interpolation"), _(
            "Doubles the Live frame rate (e.g. 30 -> 60 fps) with the RIFE model: an in-between "
            "frame is generated between every two frames. Adds about half a frame of latency. "
            "Measured at 720p with HyperSwap + TensorRT: ~50 fps out of 30. Together with the "
            "upscaler the budget runs out (~30 fps out) - use one of the two. Applies immediately."))
        self._check(g, "frame_interpolation", _("Enable frame interpolation"), "")

        g = self._card(v, _("Custom camera resolution"), _(
            "Any size your camera supports (presets are on the main window). "
            "Applies on next Live start; the camera may pick the nearest supported size."
        ))
        w_now, h_now = modules.globals.capture_resolution
        row = g.rowCount()
        self._res_w, self._res_h = QSpinBox(), QSpinBox()
        for sp, val in ((self._res_w, w_now), (self._res_h, h_now)):
            sp.setRange(160, 7680)
            sp.setSingleStep(16)
            sp.setValue(int(val))
        btn = QPushButton(_("Apply"))
        btn.clicked.connect(self._apply_custom_resolution)
        g.addWidget(self._res_w, row, 0)
        g.addWidget(self._res_h, row, 1)
        g.addWidget(btn, row, 2)

        g = self._card(v, _("Virtual camera (OBS)"), _(
            "Sends the processed picture (without FPS text) to the 'OBS Virtual Camera' "
            "device. In OBS add a Video Capture Device source, or pick the camera in "
            "Discord / Zoom / a browser. Do not start OBS's own virtual camera at the same time."
        ))
        cb = self._check(g, "virtual_camera", _("Enable virtual camera output"), "")
        cb.setEnabled(virtual_camera.available())
        row = g.rowCount()
        lbl = QLabel(_("Output resolution"))
        combo = QComboBox()
        self._vcam_sizes = [(0, 0), (640, 480), (1280, 720), (1920, 1080), (2560, 1440)]
        for wh in self._vcam_sizes:
            combo.addItem(_("Same as camera") if wh == (0, 0) else f"{wh[0]} x {wh[1]}")
        cur = tuple(modules.globals.virtual_camera_resolution)
        combo.setCurrentIndex(self._vcam_sizes.index(cur) if cur in self._vcam_sizes else 0)
        combo.currentIndexChanged.connect(
            lambda i: self._set("virtual_camera_resolution", self._vcam_sizes[i]))
        g.addWidget(lbl, row, 0)
        g.addWidget(combo, row, 1, 1, 2)
        self._controls["virtual_camera_resolution"] = (
            combo, lambda val: combo.setCurrentIndex(
                self._vcam_sizes.index(tuple(val)) if tuple(val) in self._vcam_sizes else 0))

    def _apply_custom_resolution(self) -> None:
        wh = (int(self._res_w.value()), int(self._res_h.value()))
        modules.globals.capture_resolution = wh
        save_switch_states()
        parent = self.parent()
        if hasattr(parent, "refresh_resolution_combo"):
            parent.refresh_resolution_combo()
        update_status(_("Capture resolution set (applies on next Live start):") + f" {wh[0]}x{wh[1]}")

    # ── tab: acceleration ───────────────────────────────────────────────
    def _build_accel(self, v: QVBoxLayout) -> None:
        from importlib import metadata
        import onnxruntime
        from modules.processors.frame._onnx_enhancer import tensorrt_available

        banner = QLabel(_(
            "Experimental. TensorRT options apply after restarting the program. "
            "The first start builds engines for each model (several minutes, the window "
            "may not respond); later starts load them from the cache in seconds."
        ))
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        v.addWidget(banner)

        avail = tensorrt_available()
        try:
            trt_ver = metadata.version("tensorrt-cu12-libs")
        except metadata.PackageNotFoundError:
            trt_ver = "-"
        status = (
            f"ONNX Runtime {onnxruntime.__version__} · TensorRT {trt_ver} · "
            + (_("available") if avail else _("NOT available: install tensorrt-cu12-libs 10.x"))
        )
        g = self._card(v, _("TensorRT"), status)
        cb = self._check(g, "use_tensorrt", _("Use TensorRT (experimental)"), _(
            "Runs detection, swap and enhancer models through TensorRT. Measured on "
            "RTX 5060 Ti: detection 6.2 to 1.95 ms, swap 15.1 to 6.8 ms, GPEN-256 10 to 4.8 ms, "
            "GFPGAN 75 to 50 ms. Requires the CUDA execution provider (--execution-provider cuda)."
        ))
        cb.setEnabled(avail)
        dep = [self._check(g, "trt_fp16", _("FP16 precision"), _(
            "Half precision on Tensor Cores for face detection, recognition and landmarks "
            "(about 3x faster). Swap and enhancer models always run in their own precision: "
            "forcing FP16 there was measured to corrupt the image."
        ))]
        row = g.rowCount()
        lvl_label = QLabel(_("Builder optimization level"))
        lvl = QComboBox()
        lvl.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        lvl.setMinimumContentsLength(12)
        for text in ("0", "1", "2", "3 - " + _("default, recommended"), "4",
                     "5 - " + _("slowest build, max search")):
            lvl.addItem(text)
        lvl.setCurrentIndex(int(modules.globals.trt_builder_optimization_level))
        lvl.currentIndexChanged.connect(
            lambda i: self._set("trt_builder_optimization_level", int(i)))
        lvl.setToolTip(_(
            "Higher levels search more kernels: longer first build, sometimes faster engines."))
        g.addWidget(lvl_label, row, 0)
        g.addWidget(lvl, row, 1, 1, 2)
        self._controls["trt_builder_optimization_level"] = (
            lvl, lambda val: lvl.setCurrentIndex(int(val)))
        dep += [lvl_label, lvl]
        dep += self._spin(g, "trt_max_workspace_gb", _("Builder workspace"), 1, 12, " GB", _(
            "Scratch VRAM TensorRT may use while building. 4 GB fits a 16 GB card "
            "alongside the camera pipeline."
        ))
        dep.append(self._check(g, "trt_timing_cache", _("Timing cache"), _(
            "Reuses kernel timings between models and rebuilds: much faster engine builds."
        )))
        self._depends("use_tensorrt", dep)

        g = self._card(v, _("Build engines now"), _(
            "Builds the TensorRT engines in the background, so Live starts without the "
            "multi-minute pause. Each model takes ~0.5-4 min once; later starts load them "
            "in seconds. Close Live first. Needs 'Use TensorRT' on."
        ))
        from modules import trt_prebuild
        self._build_boxes = []
        holder = QWidget()
        lay = QGridLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        for i, (key, label, fn, default) in enumerate(trt_prebuild.targets()):
            cb = QCheckBox(_(label) if not label.startswith("Swap model:")
                           else _("Swap model:") + label[len("Swap model:"):])
            cb.setChecked(default)
            lay.addWidget(cb, i // 2, i % 2)
            self._build_boxes.append((cb, label, fn))
        g.addWidget(holder, g.rowCount(), 0, 1, 3)
        self._btn_build = QPushButton(_("Build selected engines"))
        self._btn_build.clicked.connect(self._start_build)
        g.addWidget(self._btn_build, g.rowCount(), 0)
        self._build_bar = QProgressBar()
        self._build_bar.setValue(0)
        g.addWidget(self._build_bar, g.rowCount() - 1, 1, 1, 2)
        self._build_log = _hint("")
        g.addWidget(self._build_log, g.rowCount(), 0, 1, 3)
        self._depends("use_tensorrt", [holder, self._btn_build])

        g = self._card(v, _("Updates"), _(
            "At start the program asks GitHub whether a newer Deep-Boost-Cam release exists "
            "and offers to open its page. Nothing is downloaded or installed automatically."))
        self._check(g, "check_updates", _("Check for updates at start"), "")

        g = self._card(v, _("Engine cache"))
        path = _hint(modules.globals.trt_cache_dir)
        # A path has no spaces to wrap at; let it be clipped instead of
        # widening the whole page.
        path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        g.addWidget(path, g.rowCount(), 0, 1, 3)
        self._cache_label = QLabel("")
        g.addWidget(self._cache_label, g.rowCount(), 0, 1, 3)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        btn_open = QPushButton(_("Open folder"))
        btn_open.setObjectName("secondary")
        btn_open.clicked.connect(self._open_cache)
        btn_clear = QPushButton(_("Clear engine cache"))
        btn_clear.setObjectName("danger")
        btn_clear.setToolTip(_("Deletes built engines; they are rebuilt on the next start."))
        btn_clear.clicked.connect(self._clear_cache)
        row.addWidget(btn_open)
        row.addWidget(btn_clear)
        row.addStretch(1)
        holder = QWidget()
        holder.setLayout(row)
        g.addWidget(holder, g.rowCount(), 0, 1, 3)
        self._refresh_cache_label()

    def _start_build(self) -> None:
        if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
            self._build_log.setText(_("Close the Live window first."))
            return
        jobs = [(label, fn) for cb, label, fn in self._build_boxes if cb.isChecked()]
        if not jobs:
            return
        self._btn_build.setEnabled(False)
        self._build_bar.setRange(0, len(jobs))
        self._build_bar.setValue(0)
        self._builder = _EngineBuilder(jobs)
        self._builder.progress.connect(self._on_build_progress)
        self._builder.finished_all.connect(self._on_build_done)
        self._builder.start()

    def _on_build_progress(self, done: int, text: str) -> None:
        self._build_bar.setValue(done)
        self._build_log.setText(text)

    def _on_build_done(self, text: str) -> None:
        self._build_log.setText(text)
        self._btn_build.setEnabled(True)
        self._refresh_cache_label()

    def done(self, result) -> None:  # keep the dialog until a build finishes
        builder = getattr(self, "_builder", None)
        if builder is not None and builder.isRunning():
            self._build_log.setText(_("Engines are still building; wait for it to finish."))
            return
        super().done(result)

    def _refresh_cache_label(self) -> None:
        cache = modules.globals.trt_cache_dir
        size = _dir_size_mb(cache) if os.path.isdir(cache) else 0.0
        self._cache_label.setText(_("Cache size:") + f" {size:.0f} MB")

    def _open_cache(self) -> None:
        os.makedirs(modules.globals.trt_cache_dir, exist_ok=True)
        if platform.system() == "Windows":
            os.startfile(modules.globals.trt_cache_dir)

    def _clear_cache(self) -> None:
        cache = modules.globals.trt_cache_dir
        if not os.path.isdir(cache):
            return
        # Only TensorRT artefacts, never anything else that may live there.
        for name in os.listdir(cache):
            if name.endswith((".engine", ".profile", ".timing")):
                try:
                    os.remove(os.path.join(cache, name))
                except OSError:
                    pass
        self._refresh_cache_label()
        update_status("TensorRT engine cache cleared")


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        source_frame = imread_unicode(modules.globals.source_path)
        if source_frame is None:
            update_status(
                f"Could not read source image: {modules.globals.source_path}"
            )
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(get_one_face(source_frame), temp_frame)
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event,
                 raw_queue: Optional[queue.Queue] = None):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._raw_queue = raw_queue
        self._stop = stop_event

    def run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._stop.set()
                break
            seq += 1
            for q in (self._queue, self._raw_queue):
                if q is not None:
                    _put_latest(q, (seq, frame))


def _put_latest(q: queue.Queue, item) -> None:
    """Non-blocking put that drops the oldest item when the queue is full."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float,
                 reprojector: Optional[Reprojector] = None):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps
        self._reproj = reprojector

    def run(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = None
        last_source_path = None
        reported_source_error = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None
        auto_det_interval = max(1, round(self._fps * 0.04))
        tracker = FaceTracker()
        realism.reset_state()
        face_fader.reset()
        last_output = None
        detector: Optional[AsyncDetector] = None
        output = LiveOutput(self._pq, self._fps, update_status)

        while not self._stop.is_set():
            try:
                item = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue
            seq, frame = item if isinstance(item, tuple) else (0, item)

            # Async detection is toggled live from Settings.
            if modules.globals.async_detection and detector is None:
                detector = AsyncDetector(lambda f: detect_one_face_fast(f),
                                         lambda f: detect_many_faces_fast(f))
            elif not modules.globals.async_detection and detector is not None:
                detector.stop()
                detector = None

            temp_frame = frame
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)
            # Reprojection needs the pre-swap pixels (paste-back is in place).
            raw_frame = temp_frame.copy() if self._reproj is not None else None

            if not modules.globals.map_faces:
                if (
                    modules.globals.source_path
                    and modules.globals.source_path != last_source_path
                ):
                    source_frame = imread_unicode(modules.globals.source_path)
                    if source_frame is None:
                        # Clear the cached path along with the image. It keeps
                        # a source that is only transiently unreadable (still
                        # being written, on a volume that just went away) from
                        # being given up on, and it keeps a bad pick from
                        # sticking: with the path cached, A -> unreadable B -> A
                        # never reloaded A, so swapping stayed off. Retrying
                        # every frame is cheap; the message is reported once per
                        # path so it cannot flood the status line.
                        source_image = None
                        last_source_path = None
                        if reported_source_error != modules.globals.source_path:
                            reported_source_error = modules.globals.source_path
                            update_status(
                                f"Could not read source image: {modules.globals.source_path}"
                            )
                    else:
                        last_source_path = modules.globals.source_path
                        reported_source_error = None
                        source_image = get_one_face(source_frame)

                # Read per frame so Settings changes apply live.
                det_interval = modules.globals.det_interval or auto_det_interval
                need_landmarks = modules.globals.mouth_mask or realism.needs_landmarks()
                use_tracker = modules.globals.tracking_v2 and not modules.globals.many_faces
                det_count += 1
                run_det = det_count % det_interval == 0 or (use_tracker and tracker.need_detect)
                if detector is not None:
                    # Pipeline: hand this frame to the detector thread, use the
                    # newest finished detection carried to this frame by flow.
                    many = bool(modules.globals.many_faces)
                    if run_det:
                        detector.submit(seq, temp_frame, many)
                    res = detector.take()
                    ready = res is not None and res[3] == many
                    found = shift_faces(res[2], res[1], temp_frame) if ready and res[2] else None
                    if use_tracker:
                        cached_target_face = tracker.update(
                            temp_frame, ready, found, want_landmarks=need_landmarks
                        )
                        cached_many_faces = None
                    elif ready:
                        if tracker.face is not None:
                            tracker.reset()
                        cached_many_faces = found if many else None
                        cached_target_face = None if many else found
                elif use_tracker:
                    detected = detect_one_face_fast(temp_frame) if run_det else None
                    cached_target_face = tracker.update(
                        temp_frame, run_det, detected, want_landmarks=need_landmarks
                    )
                    cached_many_faces = None
                else:
                    if tracker.face is not None:
                        tracker.reset()
                    if run_det:
                        if modules.globals.many_faces:
                            cached_target_face = None
                            cached_many_faces = detect_many_faces_fast(temp_frame)
                        else:
                            cached_target_face = detect_one_face_fast(temp_frame)
                            cached_many_faces = None

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                # Fast detection skips the 2d106 landmark model, but the mouth
                # mask needs it. Attach landmarks on demand (computed once per
                # detection cycle — the helper no-ops if already present).
                if need_landmarks and cached_faces:
                    ensure_landmarks(temp_frame, cached_faces)

                # Dynamic enhancement: skip the (expensive) enhancer when the
                # face is too small to show detail or moving fast (motion blur).
                enhance_ok = True
                if modules.globals.dynamic_enhancement:
                    sizes = [float(f.bbox[3] - f.bbox[1]) for f in (cached_faces or [])]
                    enhance_ok = bool(sizes) and max(sizes) >= modules.globals.dynamic_enhancement_min_face
                    if use_tracker and tracker.motion > modules.globals.dynamic_enhancement_max_motion:
                        enhance_ok = False

                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"] and enhance_ok:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if modules.globals.fp_ui.get("face_enhancer_gpen256", False) and enhance_ok:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if modules.globals.fp_ui.get("face_enhancer_gpen512", False) and enhance_ok:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                result = fp.swap_face(source_image, t_face, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None:
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)

                # Freeze: never show the real face. With no face to swap, repeat
                # the last swapped frame (black until the first one exists).
                if modules.globals.freeze_on_face_lost and modules.globals.opacity > 0:
                    has_identity = (source_image is not None
                                    or not swapper_models.needs_source(modules.globals.swapper_model))
                    if cached_faces and has_identity:
                        last_output = temp_frame.copy()
                    else:
                        temp_frame = (last_output.copy() if last_output is not None
                                      else np.zeros_like(temp_frame))
                else:
                    last_output = None
            else:
                modules.globals.target_path = None
                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame_v2(temp_frame)
                    elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                        fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                        if modules.globals.fp_ui.get(fp_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if self._reproj is not None:
                # The presenter thread shows / streams; publish and move on.
                single = (not modules.globals.map_faces and not modules.globals.many_faces
                          and cached_target_face is not None
                          and not (modules.globals.freeze_on_face_lost and last_output is not None
                                   and not cached_faces))
                self._reproj.publish(seq, temp_frame, cached_target_face if single else None, raw_frame)
                continue

            # Overlays go on the preview only, never into the virtual camera.
            overlay = None
            if modules.globals.show_fps:
                overlay = [(f"FPS: {fps:.1f}", 30, 1.0)]
                if modules.globals.tracking_v2 and not modules.globals.map_faces:
                    overlay.append((
                        f"Track: {tracker.state} conf {tracker.confidence:.2f} lost {tracker.lost}",
                        60, 0.6))
            bboxes = [f.bbox for f in (cached_faces or [])] if not modules.globals.map_faces else None
            output.send(temp_frame, bboxes, overlay)

        output.close()
        if detector is not None:
            detector.stop()


class _PresenterWorker(QThread):
    """Reprojection: shows the last processed face moved onto every new camera frame."""

    def __init__(self, raw_queue, display_queue, stop_event, camera_fps: float,
                 reprojector: Reprojector):
        super().__init__()
        self._rq = raw_queue
        self._dq = display_queue
        self._stop = stop_event
        self._fps = camera_fps
        self._reproj = reprojector

    def run(self) -> None:
        stamps = []
        output = LiveOutput(self._dq, self._fps, update_status)
        while not self._stop.is_set():
            try:
                seq, frame = self._rq.get(timeout=0.05)
            except queue.Empty:
                continue
            if modules.globals.live_mirror:
                frame = gpu_flip(frame, 1)
            self._reproj.on_raw(seq, frame)
            out = self._reproj.compose(seq, frame)
            if out is None:
                continue
            now = time.perf_counter()
            stamps = [t for t in stamps if now - t < 1.0] + [now]
            overlay = ([(f"FPS: {len(stamps)} (swap {self._reproj.processing_fps:.0f})", 30, 1.0)]
                       if modules.globals.show_fps else None)
            bbox = self._reproj.current_bbox()
            output.send(out, [bbox] if bbox is not None else None, overlay)
        output.close()


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.setWindowTitle("Live Preview")
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._cap = VideoCapturer(camera_index)
        req_w, req_h = modules.globals.capture_resolution
        if not self._cap.start(req_w, req_h, int(modules.globals.capture_fps)):
            update_status("Failed to start camera")
            QTimer.singleShot(0, self.close)
            return

        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )

        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        # Reprojection is chosen at Live start (it changes the thread layout).
        reproj = Reprojector() if modules.globals.reprojection else None
        raw_queue = queue.Queue(maxsize=2) if reproj is not None else None

        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event, raw_queue
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps, reproj
        )
        self._presenter_worker = (
            _PresenterWorker(raw_queue, self._processed_queue, self._stop_event, camera_fps, reproj)
            if reproj is not None else None
        )
        self._capture_worker.start()
        self._processing_worker.start()
        if self._presenter_worker is not None:
            self._presenter_worker.start()

        # Poll at ~2x camera fps so we never block but also don't burn CPU.
        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        try:
            bgr_frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return
        bgr_frame = fit_image_to_size(bgr_frame, self.width(), self.height())
        self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))

    def closeEvent(self, event) -> None:
        # __init__ can bail out before these exist (e.g. the camera fails to open),
        # and closeEvent still runs.
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        timer = getattr(self, "_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        # Shutdown has to actually complete. Qt aborts the process when a
        # running QThread is destroyed, and holding a straggler in a module
        # global only moves that abort to interpreter exit. Both loops re-check
        # the stop flag between bounded operations (a camera read, a 50ms queue
        # get, one inference), so waiting terminates; a first-time model load
        # is simply slow, which is what the grace period absorbs quietly.
        for name in ("_capture_worker", "_processing_worker", "_presenter_worker"):
            worker = getattr(self, name, None)
            if worker is None:
                continue
            if not worker.wait(WORKER_SHUTDOWN_GRACE_MS):
                print(
                    f"[webcam] still waiting for {name[1:]} to stop...",
                    flush=True,
                )
                worker.wait()
        virtual_camera.stop()
        # Only now: releasing the capture while the capture thread could still
        # be inside cap.read() is not safe.
        cap = getattr(self, "_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        event.accept()


def _open_webcam_preview(camera_index: int) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    _WEBCAM_PREVIEW.show()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


def _close_live_windows() -> None:
    """Close every window that owns worker threads, so the threads stop."""
    for win in (_WEBCAM_PREVIEW, _LIVE_MAPPER):
        if win is None:
            continue
        try:
            win.close()
        except Exception:
            pass


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    def _destroy_with_cleanup(*args, **kwargs):
        # core.destroy() ends in a bare quit(), which raises SystemExit and tears
        # the interpreter down without unwinding Qt. Destroying a running QThread
        # is fatal in Qt, so stop the live-preview workers before that happens.
        _close_live_windows()
        return destroy(*args, **kwargs)

    _MAIN = MainWindow(start, _destroy_with_cleanup)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)
    _MAIN.start_update_check()

    return _Window(_APP, _MAIN)
