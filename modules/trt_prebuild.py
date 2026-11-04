"""Build TensorRT engines ahead of time (Settings -> Acceleration).

Engines are cached per model graph + model path + input shape, so the build
goes through the program's own loaders (same paths, same provider options) —
anything else would cache engines the real run never finds. Each step loads
the model with TensorRT and runs one dummy inference, which builds (or loads)
the engine. Must not run while Live is open: it reloads the swap model.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

import modules.globals as G


def _dummy_feed(session, fixed=None):
    feed = {}
    for i, inp in enumerate(session.get_inputs()):
        shape = (fixed or {}).get(inp.name) or [
            d if isinstance(d, int) and d > 0 else 1 for d in inp.shape
        ]
        dtype = np.float16 if "float16" in inp.type else np.float32
        feed[inp.name] = np.zeros(shape, dtype)
    return feed


def _analysis() -> None:
    from modules.face_analyser import get_face_analyser, reset_face_analyser

    reset_face_analyser()
    fa = get_face_analyser()
    size = int(G.det_size)
    fa.det_model.detect(np.zeros((size, size, 3), np.uint8), max_num=0, metric="default")
    for name in ("recognition", "landmark_2d_106"):
        model = fa.models.get(name)
        if model is not None:
            model.session.run(None, _dummy_feed(model.session))


def _swapper(key: str) -> Callable[[], None]:
    def run() -> None:
        from modules.processors.frame.face_swapper import get_face_swapper, reset_face_swapper

        previous = G.swapper_model
        try:
            G.swapper_model = key
            reset_face_swapper()
            if get_face_swapper() is None:
                raise RuntimeError(f"could not load {key}")
        finally:
            G.swapper_model = previous
            reset_face_swapper()
    return run


def _masks() -> None:
    from modules.processors.frame import face_regions

    for name in ("xseg", "bisenet"):
        session = face_regions._session(name)
        session.run(None, _dummy_feed(session))


def _enhancers() -> None:
    from modules.processors.frame import face_enhancer, face_enhancer_gpen256, face_enhancer_gpen512

    sessions = [face_enhancer.get_face_enhancer()]
    for module in (face_enhancer_gpen256, face_enhancer_gpen512):
        getter = getattr(module, "get_enhancer", None) or getattr(module, "get_face_enhancer", None)
        if getter is not None:
            sessions.append(getter())
    for session in sessions:
        if session is not None:
            session.run(None, _dummy_feed(session))


def _output() -> None:
    """Upscaler (selected model, both modes) + RIFE at the camera resolution, head pose."""
    from modules import face_fader, frame_interp
    from modules.processors.frame import upscaler
    from insightface.app.common import Face

    w, h = (int(v) for v in G.capture_resolution)
    frame = np.zeros((h, w, 3), np.uint8)
    face = Face(bbox=np.array([w * 0.4, h * 0.3, w * 0.6, h * 0.7], np.float32),
                kps=np.array([[w * .45, h * .45], [w * .55, h * .45], [w * .5, h * .52],
                              [w * .46, h * .6], [w * .54, h * .6]], np.float32))
    mode = G.upscaler_mode
    try:
        for G.upscaler_mode in ("face", "frame"):
            upscaler.upscale(frame, [face.bbox])
    finally:
        G.upscaler_mode = mode
    frame_interp.midpoint(frame, frame)
    face_fader._get_pose_model().get(frame, face)


def targets() -> List[Tuple[str, str, Callable[[], None], bool]]:
    """(key, label, build function, checked by default)."""
    from modules.processors.frame import swapper_models

    items = [("analysis", "Face detection / recognition / landmarks", _analysis, True)]
    for spec in swapper_models.SWAPPERS:
        items.append((f"swap:{spec.key}", f"Swap model: {spec.label}", _swapper(spec.key),
                      spec.key == G.swapper_model))
    items.append(("masks", "Masks: occlusion (XSeg) + face parts (BiSeNet)", _masks,
                  bool(G.occlusion_mask or G.face_parts_mask)))
    items.append(("enhancers", "Face enhancers: GFPGAN, GPEN-256, GPEN-512", _enhancers, False))
    items.append(("output", "Upscaler (selected model) + frame interpolation + head pose", _output,
                  bool(G.upscaler or G.frame_interpolation or G.profile_fade)))
    return items
