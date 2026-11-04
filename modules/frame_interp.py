"""Live frame interpolation (RIFE 4.9, MIT, ONNX).

Each output frame N is preceded by a RIFE midpoint between N-1 and N, released
half a frame apart by live_output.LiveOutput, so 30 fps becomes 60 fps. Cost:
latency grows by about half a frame plus one midpoint.

RIFE runs at <= 640 px width (TRT FP32, RTX 5060 Ti: 9.9 ms at 640x384 vs
36 ms at 1280x768 — the midpoint must be ready within half a frame, 16 ms at
30 fps). The motion comes from the small midpoint; fine detail comes from the
full-size frames (high-pass of their blend), so the in-between frame is not
visibly softer. FP16 is not used: it shifted pixels by up to 0.117.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np

from modules.paths import MODELS_DIR

MODEL_FILE = "rife49.onnx"
MODEL_URL = "https://huggingface.co/edgetools/rife/resolve/main/rife49.onnx"
_PAD = 64
MAX_WIDTH = 640
_lock = threading.Lock()
_session = None


def _get_session():
    global _session
    with _lock:
        if _session is None:
            path = os.path.join(MODELS_DIR, "frame", MODEL_FILE)
            if not os.path.isfile(path):
                import urllib.request

                os.makedirs(os.path.dirname(path), exist_ok=True)
                print("[interp] downloading rife49.onnx ...", flush=True)
                urllib.request.urlretrieve(MODEL_URL, path + ".part")
                os.replace(path + ".part", path)
            from modules.processors.frame._onnx_enhancer import create_onnx_session

            _session = create_onnx_session(path, allow_fp16=False)
        return _session


def _prep(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ph, pw = (-h) % _PAD, (-w) % _PAD
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return cv2.dnn.blobFromImage(img, 1.0 / 255.0, swapRB=True)


def _rife(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = _get_session()
    out = s.run(None, {"img0": _prep(a), "img1": _prep(b),
                       "timestep": np.array([0.5], np.float32)})[0][0]
    h, w = a.shape[:2]
    return cv2.merge([cv2.convertScaleAbs(out[c, :h, :w], alpha=255.0) for c in (2, 1, 0)])


_small_cache = {"src": None, "out": None}


def _small(img: np.ndarray, size) -> np.ndarray:
    """Downscale, reusing the last result (frame N is 'b' now and 'a' next time)."""
    if _small_cache["src"] is img and _small_cache["out"].shape[1::-1] == size:
        return _small_cache["out"]
    out = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    _small_cache.update(src=img, out=out)
    return out


def midpoint(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Frame halfway between a and b (BGR uint8, same size)."""
    try:
        h, w = a.shape[:2]
        if w <= MAX_WIDTH:
            return _rife(a, b)
        size = (MAX_WIDTH, int(round(h * MAX_WIDTH / w)))
        a_s, b_s = _small(a, size), _small(b, size)
        up = cv2.resize(_rife(a_s, b_s), (w, h), interpolation=cv2.INTER_LINEAR)
        blend = cv2.addWeighted(a, 0.5, b, 0.5, 0)
        # Area resize is linear, so down(blend) == blend of the downs: one resize saved.
        low = cv2.resize(cv2.addWeighted(a_s, 0.5, b_s, 0.5, 0), (w, h),
                         interpolation=cv2.INTER_LINEAR)
        # up + (blend - low): motion from RIFE, detail from the full-size frames.
        # Signed detail as two saturating uint8 halves (int16 numpy cost ~7 ms).
        return cv2.subtract(cv2.add(up, cv2.subtract(blend, low)), cv2.subtract(low, blend))
    except Exception as error:
        print(f"[interp] {error}", flush=True)
        return None
