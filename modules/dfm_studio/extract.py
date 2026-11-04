"""Face set extraction for DFM Studio: videos / photos -> aligned faces + masks.

Faces are aligned with the same DFL whole-face template the DFM loader uses
(swapper_models.DFL_WHOLE_FACE), so training and inference see identical crops.
Masks come from the BiSeNet face parser (skin, brows, eyes, glasses, nose,
mouth; hair and background excluded).
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, List, Optional

import cv2
import numpy as np

from modules import imread_unicode
from modules.globals import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

_FACE_CLASSES = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def expand_inputs(paths: Iterable[str]) -> List[str]:
    """Files and folders -> list of image/video files."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                full = os.path.join(p, name)
                if os.path.isfile(full) and name.lower().endswith(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + (".avi", ".mov", ".webm")):
                    out.append(full)
        elif os.path.isfile(p):
            out.append(p)
    return out


def _frames(path: str, every_n: int):
    if path.lower().endswith(IMAGE_EXTENSIONS):
        img = imread_unicode(path)
        if img is not None:
            yield img
        return
    cap = cv2.VideoCapture(path)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every_n == 0:
                yield frame
            i += 1
    finally:
        cap.release()


def face_mask(crop: np.ndarray) -> np.ndarray:
    """uint8 mask (0/255) of the visible face from the BiSeNet parser."""
    from modules.processors.frame import face_regions

    session = face_regions._session("bisenet")
    x = cv2.resize(crop, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
    x = ((x - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
    labels = session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0].argmax(0)
    mask = np.isin(labels, _FACE_CLASSES).astype(np.uint8) * 255
    return cv2.resize(mask, crop.shape[1::-1], interpolation=cv2.INTER_NEAREST)


def extract(inputs: List[str], out_dir: str, resolution: int, every_n: int = 3,
            max_faces: int = 4000, min_face: int = 64,
            progress: Optional[Callable[[str], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None) -> int:
    """Append aligned faces from inputs to out_dir; returns how many were saved."""
    from modules.face_analyser import detect_many_faces_fast
    from modules.processors.frame.swapper_models import DFL_WHOLE_FACE

    os.makedirs(out_dir, exist_ok=True)
    existing = [n for n in os.listdir(out_dir) if n.endswith(".jpg")]
    index = len(existing)
    saved = 0
    template = DFL_WHOLE_FACE * resolution
    for path in expand_inputs(inputs):
        for frame in _frames(path, every_n):
            if should_stop and should_stop():
                return saved
            if index >= max_faces:
                return saved
            faces = detect_many_faces_fast(frame) or []
            # The largest face only: crowds would mix identities into the set.
            faces = [f for f in faces if (f.bbox[2] - f.bbox[0]) >= min_face]
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            M = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), template, method=cv2.LMEDS)[0]
            if M is None:
                continue
            crop = cv2.warpAffine(frame, M, (resolution, resolution), borderMode=cv2.BORDER_REPLICATE)
            mask = face_mask(crop)
            if mask.mean() < 20:   # parser found almost no face: bad detection
                continue
            name = os.path.join(out_dir, f"{index:05d}")
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(name + ".jpg")
            cv2.imencode(".png", mask)[1].tofile(name + "_mask.png")
            index += 1
            saved += 1
            if progress and saved % 10 == 0:
                progress(f"{os.path.basename(path)}: {index} faces")
    if progress:
        progress(f"done: {index} faces")
    return saved
