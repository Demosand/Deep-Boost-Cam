"""DFM Studio window: build face sets, train, export a .dfm into models/dfm."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import List

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from modules.dfm_studio import WORKSPACES


def _(text: str) -> str:
    from modules import ui
    return ui._(text)


class _ExtractWorker(QThread):
    progress = Signal(str)
    finished_count = Signal(int)

    def __init__(self, inputs: List[str], out_dir: str, resolution: int, every_n: int):
        super().__init__()
        self._args = (inputs, out_dir, resolution, every_n)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        from modules.dfm_studio import extract
        inputs, out_dir, res, every_n = self._args
        try:
            n = extract.extract(inputs, out_dir, res, every_n=every_n,
                                progress=self.progress.emit, should_stop=lambda: self._stop)
        except Exception as error:
            self.progress.emit(f"error: {error}")
            n = 0
        self.finished_count.emit(n)


class DFMStudioDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("DFM Studio"))
        self.resize(860, 820)
        self._inputs = {"src": [], "dst": []}
        self._extractor = None

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        page.setObjectName("settingsPage")
        v = QVBoxLayout(page)
        scroll.setWidget(page)
        outer.addWidget(scroll)

        note = QLabel(_(
            "Trains your own DeepFaceLive-compatible face model (.dfm) on this GPU. "
            "A DFM model learns one identity and keeps it far more stable than one-photo "
            "models. Use only faces you have the right to use (your own, or with consent)."
        ))
        note.setObjectName("banner")
        note.setWordWrap(True)
        v.addWidget(note)

        # ── project ──
        box = QGroupBox(_("1. Project"))
        g = QGridLayout(box)
        g.addWidget(QLabel(_("Model name")), 0, 0)
        self.cb_name = QComboBox()
        self.cb_name.setEditable(True)
        os.makedirs(WORKSPACES, exist_ok=True)
        self.cb_name.addItems(sorted(d for d in os.listdir(WORKSPACES)
                                     if os.path.isdir(os.path.join(WORKSPACES, d))) or ["my_model"])
        self.cb_name.currentTextChanged.connect(lambda _t: self._refresh())
        g.addWidget(self.cb_name, 0, 1, 1, 3)
        g.addWidget(QLabel(_("Resolution")), 1, 0)
        self.cb_res = QComboBox()
        for r in (224, 256, 320):
            self.cb_res.addItem(f"{r} px", r)
        self.cb_res.setCurrentIndex(1)
        g.addWidget(self.cb_res, 1, 1)
        g.addWidget(QLabel(_("Batch")), 1, 2)
        self.sp_batch = QSpinBox()
        self.sp_batch.setRange(2, 32)
        self.sp_batch.setValue(8)
        g.addWidget(self.sp_batch, 1, 3)
        g.addWidget(QLabel(_("Iterations")), 2, 0)
        self.sp_iters = QSpinBox()
        self.sp_iters.setRange(1000, 1000000)
        self.sp_iters.setSingleStep(10000)
        self.sp_iters.setValue(100000)
        g.addWidget(self.sp_iters, 2, 1)
        g.addWidget(QLabel(_("Take every Nth video frame")), 2, 2)
        self.sp_every = QSpinBox()
        self.sp_every.setRange(1, 30)
        self.sp_every.setValue(3)
        g.addWidget(self.sp_every, 2, 3)
        g.addWidget(QLabel(_("Architecture")), 4, 0)
        self.cb_arch = QComboBox()
        self.cb_arch.addItem(_("DF - best likeness"), "df")
        self.cb_arch.addItem(_("LIAE - better lighting and pose transfer"), "liae")
        g.addWidget(self.cb_arch, 4, 1)
        g.addWidget(QLabel(_("Start from model")), 4, 2)
        self.cb_init = QComboBox()
        g.addWidget(self.cb_init, 4, 3)
        g.addWidget(QLabel(_("GAN power")), 5, 0)
        self.sp_gan = QDoubleSpinBox()
        self.sp_gan.setRange(0.0, 1.0)
        self.sp_gan.setSingleStep(0.05)
        self.sp_gan.setDecimals(2)
        g.addWidget(self.sp_gan, 5, 1)
        hint2 = QLabel(_(
            "GAN: 0 = off. Turn on (0.05-0.2) only after the model already looks right "
            "(~50k+): it adds sharp skin detail. 'Start from model' copies the weights of "
            "another trained project with the same resolution and architecture: a new "
            "face then trains several times faster."
        ))
        hint2.setObjectName("hint")
        hint2.setWordWrap(True)
        g.addWidget(hint2, 6, 0, 1, 4)
        hint = QLabel(_(
            "RTX 5060 Ti: ~6 iterations/s at 256 px, batch 8 (100k = ~4.5 h). A usable "
            "model often appears at 30-50k; export any time and keep training."
        ))
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        g.addWidget(hint, 3, 0, 1, 4)
        v.addWidget(box)
        self._fill_init_combo()

        # ── face sets ──
        self._set_labels = {}
        for kind, title, tip in (
            ("src", _("2. Source face (who you become)"),
             _("Videos or photos of the person whose face the model will produce. "
               "1000-3000 varied faces: angles, expressions, lighting.")),
            ("dst", _("3. Your face (the camera face)"),
             _("Videos of yourself as the webcam sees you: same camera, lighting, "
               "turns, talking, blinking. 1000-3000 faces.")),
        ):
            box = QGroupBox(title)
            g = QGridLayout(box)
            h = QLabel(tip)
            h.setObjectName("hint")
            h.setWordWrap(True)
            g.addWidget(h, 0, 0, 1, 4)
            lbl = QLabel("")
            lbl.setWordWrap(True)
            self._set_labels[kind] = lbl
            g.addWidget(lbl, 1, 0, 1, 4)
            b_files = QPushButton(_("Add videos / photos"))
            b_files.setObjectName("secondary")
            b_files.clicked.connect(lambda _c, k=kind: self._add_files(k))
            b_dir = QPushButton(_("Add folder"))
            b_dir.setObjectName("secondary")
            b_dir.clicked.connect(lambda _c, k=kind: self._add_folder(k))
            b_run = QPushButton(_("Extract faces"))
            b_run.clicked.connect(lambda _c, k=kind: self._extract(k))
            b_clear = QPushButton(_("Clear set"))
            b_clear.setObjectName("danger")
            b_clear.clicked.connect(lambda _c, k=kind: self._clear_set(k))
            for i, b in enumerate((b_files, b_dir, b_run, b_clear)):
                g.addWidget(b, 2, i)
            if kind == "dst":
                b_rec = QPushButton("⏺  " + _("Record from camera (guided)"))
                b_rec.clicked.connect(self._record)
                g.addWidget(b_rec, 3, 0, 1, 4)
            v.addWidget(box)
        self.lbl_extract = QLabel("")
        self.lbl_extract.setObjectName("hint")
        v.addWidget(self.lbl_extract)

        # ── training ──
        box = QGroupBox(_("4. Training and export"))
        g = QGridLayout(box)
        self.b_train = QPushButton(_("Start / resume training"))
        self.b_train.clicked.connect(self._start_training)
        self.b_stop = QPushButton(_("Stop"))
        self.b_stop.setObjectName("danger")
        self.b_stop.clicked.connect(self._stop_training)
        self.b_export = QPushButton(_("Export .dfm to models/dfm"))
        self.b_export.setObjectName("secondary")
        self.b_export.clicked.connect(self._export)
        g.addWidget(self.b_train, 0, 0)
        g.addWidget(self.b_stop, 0, 1)
        g.addWidget(self.b_export, 0, 2)
        self.bar = QProgressBar()
        g.addWidget(self.bar, 1, 0, 1, 3)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        g.addWidget(self.lbl_status, 2, 0, 1, 3)
        h = QLabel(_("Preview rows: your face, its reconstruction, the swap. "
                     "Training keeps running if you close this window."))
        h.setObjectName("hint")
        h.setWordWrap(True)
        g.addWidget(h, 3, 0, 1, 3)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        g.addWidget(self.preview, 4, 0, 1, 3)
        v.addWidget(box)
        v.addStretch(1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(2000)
        self._refresh()

    # ── helpers ─────────────────────────────────────────────────────────
    def _name(self) -> str:
        name = "".join(c for c in self.cb_name.currentText().strip() if c.isalnum() or c in "-_")
        return name or "my_model"

    def _ws(self) -> str:
        return os.path.join(WORKSPACES, self._name())

    def _count(self, kind: str) -> int:
        d = os.path.join(self._ws(), kind)
        return len([n for n in os.listdir(d) if n.endswith(".jpg")]) if os.path.isdir(d) else 0

    def _refresh(self) -> None:
        cfg_path = os.path.join(self._ws(), "config.json")
        if os.path.exists(cfg_path):
            try:
                cfg = json.load(open(cfg_path, encoding="utf-8"))
                self.cb_res.setCurrentIndex(max(0, self.cb_res.findData(int(cfg.get("resolution", 256)))))
                self.sp_batch.setValue(int(cfg.get("batch", 8)))
                self.sp_iters.setValue(int(cfg.get("target_iters", 100000)))
                self.cb_arch.setCurrentIndex(max(0, self.cb_arch.findData(cfg.get("arch", "df"))))
                self.sp_gan.setValue(float(cfg.get("gan_power", 0.0)))
                self.cb_init.setCurrentIndex(max(0, self.cb_init.findData(cfg.get("init_from", ""))))
            except (OSError, ValueError):
                pass
        has_model = os.path.exists(os.path.join(self._ws(), "model.pt"))
        # Resolution is fixed once a model exists (weights depend on it).
        self.cb_res.setEnabled(not has_model)
        self.cb_arch.setEnabled(not has_model)
        self.cb_init.setEnabled(not has_model)
        for kind, lbl in self._set_labels.items():
            queued = len(self._inputs[kind])
            lbl.setText(_("Faces in set:") + f" {self._count(kind)}"
                        + (f"  ·  {_('queued files')}: {queued}" if queued else ""))
        self._poll()

    def _status(self):
        path = os.path.join(self._ws(), "status.json")
        if not os.path.exists(path):
            return None, False
        try:
            st = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            return None, False
        alive = st.get("running") and time.time() - os.path.getmtime(path) < 20
        return st, bool(alive)

    def _poll(self) -> None:
        st, alive = self._status()
        self.b_train.setEnabled(not alive)
        self.b_stop.setEnabled(alive)
        self.b_export.setEnabled(os.path.exists(os.path.join(self._ws(), "model.pt")))
        if st:
            self.bar.setRange(0, max(1, int(st.get("target", 1))))
            self.bar.setValue(int(st.get("iter", 0)))
            self.lbl_status.setText(
                (_("Training") if alive else _("Stopped")) +
                f": {st.get('iter', 0)} / {st.get('target', 0)} · loss {st.get('loss', 0)} · "
                f"{st.get('it_s', 0)} it/s · ETA {st.get('eta_min', 0)} min")
        else:
            self.bar.setValue(0)
            self.lbl_status.setText(_("Not trained yet."))
        prev = os.path.join(self._ws(), "preview.jpg")
        if os.path.exists(prev):
            pm = QPixmap(prev)
            if not pm.isNull():
                self.preview.setPixmap(pm.scaledToWidth(min(780, pm.width() * 2),
                                                        Qt.TransformationMode.SmoothTransformation))

    def _fill_init_combo(self) -> None:
        self.cb_init.clear()
        self.cb_init.addItem(_("(from scratch)"), "")
        for d in sorted(os.listdir(WORKSPACES)):
            if os.path.exists(os.path.join(WORKSPACES, d, "model.pt")):
                self.cb_init.addItem(d, d)

    def _record(self) -> None:
        dlg = RecordDialog(os.path.join(self._ws(), "recordings"), self)
        if dlg.exec() and dlg.saved_path:
            self._inputs["dst"].append(dlg.saved_path)
            self._refresh()
            self._extract("dst")

    # ── face sets ───────────────────────────────────────────────────────
    def _add_files(self, kind: str) -> None:
        paths, _f = QFileDialog.getOpenFileNames(
            self, _("Add videos / photos"), "",
            "Media (*.mp4 *.mkv *.avi *.mov *.webm *.png *.jpg *.jpeg *.bmp *.webp)")
        self._inputs[kind] += paths
        self._refresh()

    def _add_folder(self, kind: str) -> None:
        path = QFileDialog.getExistingDirectory(self, _("Add folder"))
        if path:
            self._inputs[kind].append(path)
            self._refresh()

    def _extract(self, kind: str) -> None:
        if self._extractor is not None and self._extractor.isRunning():
            self.lbl_extract.setText(_("Extraction already running..."))
            return
        if not self._inputs[kind]:
            self.lbl_extract.setText(_("Add videos or photos first."))
            return
        self._save_config()
        self._extractor = _ExtractWorker(self._inputs[kind], os.path.join(self._ws(), kind),
                                         int(self.cb_res.currentData()), self.sp_every.value())
        self._extractor.progress.connect(self.lbl_extract.setText)
        self._extractor.finished_count.connect(lambda n, k=kind: self._extract_done(k, n))
        self.lbl_extract.setText(_("Extracting faces (first run loads the models)..."))
        self._extractor.start()

    def _extract_done(self, kind: str, n: int) -> None:
        self._inputs[kind] = []
        self.lbl_extract.setText(_("Faces added:") + f" {n}")
        self._refresh()

    def _clear_set(self, kind: str) -> None:
        shutil.rmtree(os.path.join(self._ws(), kind), ignore_errors=True)
        self._inputs[kind] = []
        self._refresh()

    # ── training ────────────────────────────────────────────────────────
    def _save_config(self) -> None:
        os.makedirs(self._ws(), exist_ok=True)
        with open(os.path.join(self._ws(), "config.json"), "w", encoding="utf-8") as f:
            json.dump({"name": self._name(), "resolution": int(self.cb_res.currentData()),
                       "batch": self.sp_batch.value(), "target_iters": self.sp_iters.value(),
                       "arch": self.cb_arch.currentData(), "gan_power": self.sp_gan.value(),
                       "init_from": self.cb_init.currentData() or ""}, f)

    def _start_training(self) -> None:
        if self._count("src") < 50 or self._count("dst") < 50:
            self.lbl_status.setText(_("Each face set needs at least 50 faces (1000+ recommended)."))
            return
        self._save_config()
        from modules.dfm_studio import PROJECT_DIR
        log = open(os.path.join(self._ws(), "train.log"), "a", encoding="utf-8")
        subprocess.Popen(
            [sys.executable, "-m", "modules.dfm_studio.train", self._ws()],
            cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.lbl_status.setText(_("Starting training (loading faces)..."))
        self.b_train.setEnabled(False)

    def _stop_training(self) -> None:
        open(os.path.join(self._ws(), "STOP"), "w").close()
        self.lbl_status.setText(_("Stopping after the current step (model is saved)..."))

    def _export(self) -> None:
        from modules.dfm_studio import PROJECT_DIR
        self.lbl_status.setText(_("Exporting..."))
        QApplication_process()
        r = subprocess.run([sys.executable, "-m", "modules.dfm_studio.train", self._ws(), "--export"],
                           cwd=PROJECT_DIR, capture_output=True, text=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = (r.stdout or "").strip().splitlines()
        if r.returncode != 0 or not lines:
            self.lbl_status.setText(_("Export failed:") + f" {(r.stderr or '')[-300:]}")
            return
        from modules.processors.frame import swapper_models
        swapper_models.refresh_dfm()
        parent = self.parent()
        if hasattr(parent, "refresh_swapper_combo"):
            parent.refresh_swapper_combo()
        self.lbl_status.setText(_("Exported:") + f" {lines[-1]}  ·  "
                                + _("choose it in 'Swap model' on the main window."))

    def closeEvent(self, event) -> None:
        if self._extractor is not None and self._extractor.isRunning():
            self._extractor.stop()
            self._extractor.wait()
        event.accept()


class RecordDialog(QDialog):
    """Guided capture of the camera face set: prompts for turns and expressions."""

    PROMPTS = [
        "Look straight into the camera", "Slowly turn your head left", "Slowly turn right",
        "Look up", "Look down", "Talk / read something aloud", "Smile", "Blink several times",
        "Raise your eyebrows", "Tilt your head left and right", "Move closer and further",
        "Look straight again, neutral face",
    ]
    STEP_S = 5

    def __init__(self, out_dir: str, parent=None):
        super().__init__(parent)
        from modules import ui
        import modules.globals as G
        self.setWindowTitle(_("Record from camera"))
        self.resize(760, 640)
        self.saved_path = None
        self._out_dir = out_dir
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.cb_cam = QComboBox()
        self._idx, names = ui.get_available_cameras()
        self.cb_cam.addItems(names)
        row.addWidget(QLabel(_("Camera")))
        row.addWidget(self.cb_cam, 1)
        self.b_go = QPushButton(_("Start recording"))
        self.b_go.clicked.connect(self._toggle)
        row.addWidget(self.b_go)
        v.addLayout(row)
        self.prompt = QLabel(_("Press 'Start recording' and follow the prompts (~1 minute)."))
        self.prompt.setObjectName("banner")
        self.prompt.setWordWrap(True)
        v.addWidget(self.prompt)
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumHeight(420)
        v.addWidget(self.view, 1)
        self._res = tuple(G.capture_resolution)
        self._cap = self._writer = None
        self._t0 = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _toggle(self) -> None:
        if self._cap is None:
            self._start()
        else:
            self._finish()

    def _start(self) -> None:
        import cv2
        from modules.video_capture import VideoCapturer
        i = self.cb_cam.currentIndex()
        if i < 0 or i >= len(self._idx):
            return
        self._cap = VideoCapturer(self._idx[i])
        if not self._cap.start(self._res[0], self._res[1], 30):
            self.prompt.setText(_("Failed to start camera"))
            self._cap = None
            return
        os.makedirs(self._out_dir, exist_ok=True)
        self._path = os.path.join(self._out_dir, time.strftime("rec_%Y%m%d_%H%M%S.mp4"))
        size = (self._cap.actual_width, self._cap.actual_height)
        self._writer = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
        self._t0 = time.time()
        self.b_go.setText(_("Stop"))
        self._timer.start(30)

    def _tick(self) -> None:
        import cv2
        from modules.ui import _bgr_to_qpixmap, fit_image_to_size
        ok, frame = self._cap.read()
        if not ok:
            return
        self._writer.write(frame)
        step = int((time.time() - self._t0) // self.STEP_S)
        if step >= len(self.PROMPTS):
            self._finish()
            return
        self.prompt.setText(f"{step + 1}/{len(self.PROMPTS)}:  " + _(self.PROMPTS[step]))
        shown = fit_image_to_size(cv2.flip(frame, 1), self.view.width(), self.view.height())
        self.view.setPixmap(_bgr_to_qpixmap(shown))

    def _finish(self) -> None:
        self._timer.stop()
        if self._writer is not None:
            self._writer.release()
            self.saved_path = self._path
        if self._cap is not None:
            self._cap.release()
        self._cap = self._writer = None
        self.accept() if self.saved_path else self.reject()

    def closeEvent(self, event) -> None:
        if self._cap is not None:
            self._finish()
        event.accept()


def QApplication_process() -> None:
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
