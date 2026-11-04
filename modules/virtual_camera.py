"""Virtual camera output: the processed live frame as a webcam device.

pyvirtualcam drives the "OBS Virtual Camera" DirectShow device installed with
OBS Studio (verified on OBS 32.2.2: frames read back exactly through
DirectShow). Discord, Zoom, browsers and OBS scenes then use it like a webcam.
The device opens lazily with the first frame and closes when Live stops.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

import modules.globals as G

_LOCK = threading.Lock()
_camera = None
_size: Optional[tuple] = None
_error: Optional[str] = None


def available() -> bool:
    try:
        import pyvirtualcam  # noqa: F401
    except ImportError:
        return False
    return True


def last_error() -> Optional[str]:
    return _error


def send(frame: np.ndarray, fps: float) -> None:
    """Push a BGR frame. A device that cannot open turns the option off once."""
    global _camera, _size, _error
    out_w, out_h = G.virtual_camera_resolution
    if out_w <= 0 or out_h <= 0:
        out_h, out_w = frame.shape[:2]
    with _LOCK:
        if _camera is not None and _size != (out_w, out_h):
            _close_locked()  # resolution changed in Settings: reopen
        if _camera is None:
            try:
                import pyvirtualcam

                _camera = pyvirtualcam.Camera(
                    width=out_w, height=out_h, fps=max(1, int(round(fps or 30))),
                    fmt=pyvirtualcam.PixelFormat.BGR, backend="obs", print_fps=False,
                )
                _size = (out_w, out_h)
                _error = None
                print(f"[virtual_camera] streaming {out_w}x{out_h} to {_camera.device}", flush=True)
            except Exception as error:
                _error = (f"Virtual camera could not start: {str(error)[:160]}. "
                          "Install OBS Studio and stop OBS's own virtual camera.")
                print(f"[virtual_camera] {_error}", flush=True)
                G.virtual_camera = False
                return
        if frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        try:
            _camera.send(np.ascontiguousarray(frame))
        except Exception as error:
            _error = f"Virtual camera stopped: {str(error)[:160]}"
            G.virtual_camera = False
            _close_locked()


def _close_locked() -> None:
    global _camera, _size
    if _camera is not None:
        try:
            _camera.close()
        except Exception:
            pass
        print("[virtual_camera] stopped", flush=True)
    _camera = None
    _size = None


def stop() -> None:
    with _LOCK:
        _close_locked()
