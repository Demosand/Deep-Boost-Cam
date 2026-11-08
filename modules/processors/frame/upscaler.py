"""x2 super-resolution of the Live output (ONNX, runs on TensorRT when enabled).

Modes: "frame" upscales the whole picture; "face" upscales the picture with a
plain resize and runs the model only on the face region (much cheaper, the
face is what people look at). Face crops go through the model at one fixed
size, so TensorRT builds a single engine for them.

x4 models: on a face crop they run at the crop size and the result is area-
downsampled to x2 (sharpest). On the whole frame that costs ~50 ms at 720p
(TRT FP16, RTX 5060 Ti), so there they get a half-size input and their x4
lands exactly on x2 (~4x cheaper).
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, NamedTuple, Optional

import cv2
import numpy as np

import modules.globals as G
from modules.paths import MODELS_DIR

_FF = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"


class SRModel(NamedTuple):
    label: str
    file: str
    scale: int
    url: str
    # TensorRT FP16 vs CUDA, max|diff| on 0..1 output (mean <= 0.0012):
    # SPAN 0.006, Clear Reality 0.010, ESRGAN compact 0.029, ESRGAN x2 0.045.
    fp16_ok: bool


MODELS: Dict[str, SRModel] = {
    "span": SRModel("SPAN (fast)", "span_kendata_x4.onnx", 4, _FF + "span_kendata_x4.onnx", True),
    "clear_reality": SRModel("Clear Reality (fast, natural)", "clear_reality_x4.onnx", 4,
                             _FF + "clear_reality_x4.onnx", True),
    "esrgan_compact": SRModel("Real-ESRGAN compact", "realesr_general_x4v3.onnx", 4,
                              "https://huggingface.co/archgod94/tellcue-models/resolve/main/"
                              "realesr-general-x4v3.onnx", True),
    "esrgan_x2": SRModel("Real-ESRGAN x2 (quality, heavy)", "real_esrgan_x2.onnx", 2,
                         _FF + "real_esrgan_x2.onnx", True),
}
FACE_INPUT = 256   # fixed model input for face mode

_lock = threading.Lock()
_sessions: Dict[str, object] = {}


def _session(key: str):
    spec = MODELS.get(key) or MODELS["span"]
    with _lock:
        if key not in _sessions:
            path = os.path.join(MODELS_DIR, "frame", spec.file)
            if not os.path.isfile(path):
                from modules.model_downloader import fetch_url

                if fetch_url(spec.url, path) is None:
                    raise RuntimeError(f"could not download {spec.file}")
            from modules.processors.frame._onnx_enhancer import create_onnx_session

            _sessions[key] = create_onnx_session(path, allow_fp16=spec.fp16_ok)
        return _sessions[key], spec


def sr_x2(img: np.ndarray, key: Optional[str] = None, fast: bool = False) -> np.ndarray:
    """BGR uint8 -> BGR uint8 at exactly twice the size (fast: x4 models on half input)."""
    session, spec = _session(key or G.upscaler_model)
    h, w = img.shape[:2]
    if fast and spec.scale == 4 and h % 2 == 0 and w % 2 == 0:
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    out = session.run(None, {session.get_inputs()[0].name: to_blob(img)})[0][0]
    out = from_planes(out)
    if out.shape[0] != 2 * h or out.shape[1] != 2 * w:
        out = cv2.resize(out, (2 * w, 2 * h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out)


def to_blob(img: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> RGB float32 NCHW 0..1 (cv2: ~5x faster than numpy)."""
    return cv2.dnn.blobFromImage(img, 1.0 / 255.0, swapRB=True)


def from_planes(chw: np.ndarray) -> np.ndarray:
    """RGB float CHW 0..1 -> BGR uint8 HWC; per-plane saturating convert, then merge
    (numpy clip/transpose/astype took ~20 ms on a 2560x1440 output)."""
    return cv2.merge([cv2.convertScaleAbs(chw[c], alpha=255.0) for c in (2, 1, 0)])


_FEATHER = None


def _feather(w: int, h: int) -> np.ndarray:
    """Soft-edged square alpha (uint8 3-channel) — template resized, not re-blurred."""
    global _FEATHER
    if _FEATHER is None:
        t = np.zeros((96, 96), np.float32)
        t[16:-16, 16:-16] = 1.0
        _FEATHER = (cv2.GaussianBlur(t, (0, 0), 6) * 255).astype(np.uint8)
    a = cv2.resize(_FEATHER, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.merge([a, a, a])


def _face_boxes(bboxes, w: int, h: int) -> List[tuple]:
    boxes = []
    for b in bboxes or []:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        side = 1.5 * max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bx0, by0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        bx1, by1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if bx1 - bx0 >= 16 and by1 - by0 >= 16:
            boxes.append((bx0, by0, bx1, by1))
    return boxes


def upscale(frame: np.ndarray, bboxes=None) -> np.ndarray:
    """x2 output frame; on any model error the plain resize is returned."""
    h, w = frame.shape[:2]
    try:
        if G.upscaler_mode == "frame":
            return sr_x2(frame, fast=True)
        out = cv2.resize(frame, (2 * w, 2 * h), interpolation=cv2.INTER_LINEAR)
        for bx0, by0, bx1, by1 in _face_boxes(bboxes, w, h):
            crop = frame[by0:by1, bx0:bx1]
            cw, ch = bx1 - bx0, by1 - by0
            up = sr_x2(cv2.resize(crop, (FACE_INPUT, FACE_INPUT), interpolation=cv2.INTER_AREA
                                  if max(cw, ch) > FACE_INPUT else cv2.INTER_CUBIC))
            up = cv2.resize(up, (2 * cw, 2 * ch), interpolation=cv2.INTER_AREA)
            # Feathered square: the region edge blends into the plain resize.
            a = _feather(2 * cw, 2 * ch)
            roi = out[2 * by0:2 * by1, 2 * bx0:2 * bx1]
            out[2 * by0:2 * by1, 2 * bx0:2 * bx1] = cv2.add(
                cv2.multiply(up, a, scale=1 / 255.0), cv2.multiply(roi, 255 - a, scale=1 / 255.0))
        return out
    except Exception as error:
        print(f"[upscaler] {error}", flush=True)
        return cv2.resize(frame, (2 * w, 2 * h), interpolation=cv2.INTER_LINEAR)
