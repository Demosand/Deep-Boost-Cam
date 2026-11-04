"""Face Fader and profile fade: how much of the swap is shown, and of whom.

* Swap on/off ramps over G.fader_seconds instead of cutting (G.swap_enabled
  is the switch, the main window has the button).
* A new source identity (another photo / library set) is reached by slerp of
  the ArcFace embedding over G.fader_seconds, so the face morphs, not jumps.
  Small embedding changes (multi-photo pose blending) are followed directly.
* Profile fade: the head pose from insightface's 1k3d68 model (pitch, yaw,
  roll in degrees) lowers the swap between a start and an end angle, where
  one-shot models break down, instead of showing a smeared half-face.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np

import modules.globals as G

_lock = threading.Lock()
_s = {"strength": 1.0, "t": None, "frm": None, "to": None, "t0": 0.0, "shown": None}
_pose_model = None


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def strength() -> float:
    """0..1 share of the swap shown, ramped when the Face Fader is on."""
    target = 1.0 if G.swap_enabled else 0.0
    with _lock:
        now = time.perf_counter()
        if not G.face_fader or G.fader_seconds <= 0:
            _s["strength"], _s["t"] = target, now
            return target
        dt = 0.0 if _s["t"] is None else now - _s["t"]
        _s["t"] = now
        step = dt / float(G.fader_seconds)
        cur = _s["strength"]
        cur = min(target, cur + step) if target > cur else max(target, cur - step)
        _s["strength"] = cur
        return _smooth(cur)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-4:
        return b
    so = np.sin(omega)
    return (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / so


def blend_identity(source_face):
    """Source face whose embedding moves smoothly to a newly chosen identity."""
    if not G.face_fader or source_face is None or getattr(source_face, "embedding", None) is None:
        return source_face
    emb = np.asarray(source_face.embedding, np.float32)
    scale = float(np.linalg.norm(emb))
    new = emb / max(scale, 1e-6)
    with _lock:
        now = time.perf_counter()
        shown = _s["shown"]
        if shown is None:
            _s.update(shown=new, to=new, frm=new, t0=now)
            return source_face
        if float(np.dot(new, _s["to"])) < 0.9:      # a different person: start a morph
            _s.update(frm=shown, to=new, t0=now)
        else:
            _s["to"] = new                          # same person: follow small changes
        secs = max(float(G.fader_seconds), 1e-3)
        t = _smooth((now - _s["t0"]) / secs)
        cur = _s["to"] if t >= 1.0 else _slerp(_s["frm"], _s["to"], t)
        cur = cur / max(float(np.linalg.norm(cur)), 1e-6)
        _s["shown"] = cur
    if t >= 1.0 and cur is new:
        return source_face
    face = type(source_face)(dict(source_face))
    face.embedding = (cur * scale).astype(np.float32)
    return face


def _get_pose_model():
    global _pose_model
    with _lock:
        if _pose_model is None:
            import insightface
            from modules.processors.frame._onnx_enhancer import build_provider_config

            path = os.path.join(os.path.expanduser("~"), ".insightface", "models",
                                "buffalo_l", "1k3d68.onnx")
            model = insightface.model_zoo.get_model(path, providers=build_provider_config())
            model.prepare(ctx_id=0)
            _pose_model = model
        return _pose_model


def head_pose(frame: np.ndarray, face) -> Optional[np.ndarray]:
    """(pitch, yaw, roll) in degrees, computed once per Face object."""
    pose = getattr(face, "pose", None)
    if pose is None:
        try:
            _get_pose_model().get(frame, face)   # sets face['pose']
            pose = getattr(face, "pose", None)
        except Exception as error:
            print(f"[face_fader] pose failed: {error}", flush=True)
    return pose


def pose_factor(frame: np.ndarray, face) -> float:
    """1 for a frontal face, falling to 0 past the end angles."""
    if not G.profile_fade or face is None:
        return 1.0
    pose = head_pose(frame, face)
    if pose is None:
        return 1.0
    pitch, yaw = abs(float(pose[0])), abs(float(pose[1]))

    def fall(angle, start, end):
        return 1.0 - _smooth((angle - start) / max(end - start, 1e-3))

    return min(fall(yaw, G.profile_yaw_start, G.profile_yaw_end),
               fall(pitch, G.profile_pitch_start, G.profile_pitch_end))


def coverage(frame: np.ndarray, face) -> float:
    """Overall swap coverage for this face on this frame (0..1)."""
    return strength() * pose_factor(frame, face)


def reset() -> None:
    with _lock:
        _s.update(strength=1.0 if G.swap_enabled else 0.0, t=None, shown=None)
