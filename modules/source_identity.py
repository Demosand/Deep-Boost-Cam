"""One identity from several photos of the same person.

Swap models see the source only as a 512-d ArcFace embedding. Several photos
make that identity steadier (one photo's lighting or expression stops leaking
into every frame), and weighting them by pose lets a turned target lean on the
photo taken at a similar angle. Off by default (globals.multi_source); then the
single selected photo is used exactly as before.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np

import modules.globals as G

# Width of the pose kernel (pose-proxy units) and the share every photo keeps.
POSE_SCALE = 0.15
BASE_WEIGHT = 0.3

_lock = threading.Lock()
_cache = {"key": None, "identity": None}


def pose_proxies(kps) -> Tuple[float, float]:
    """(yaw, pitch) proxies from the 5 keypoints; ~0 for a frontal face."""
    k = np.asarray(kps, np.float32)
    eyes = (k[0] + k[1]) / 2
    mouth = (k[3] + k[4]) / 2
    eye_dist = max(float(np.linalg.norm(k[1] - k[0])), 1e-3)
    span = max(float(mouth[1] - eyes[1]), 1e-3)
    yaw = float(k[2][0] - eyes[0]) / eye_dist
    pitch = float(k[2][1] - eyes[1]) / span - 0.5
    return yaw, pitch


class SourceIdentity:
    def __init__(self, faces: Sequence, paths: Sequence[str]):
        self.faces = list(faces)
        self.paths = list(paths)
        self._poses = np.array([pose_proxies(f.kps) for f in self.faces], np.float32)
        emb = np.stack([np.asarray(f.embedding, np.float32) for f in self.faces])
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self._normed = emb / np.maximum(norms, 1e-6)
        self._scale = float(norms.mean())
        self._last_key = None
        self._last_face = None

    def __len__(self) -> int:
        return len(self.faces)

    def weights(self, target_kps) -> np.ndarray:
        n = len(self.faces)
        if n == 1:
            return np.ones(1, np.float32)
        pose = np.array(pose_proxies(target_kps), np.float32)
        d2 = ((self._poses - pose) ** 2).sum(axis=1)
        w = BASE_WEIGHT / n + (1.0 - BASE_WEIGHT) * np.exp(-d2 / (2 * POSE_SCALE ** 2))
        return (w / w.sum()).astype(np.float32)

    def face_for(self, target_face):
        if len(self.faces) == 1 or getattr(target_face, "kps", None) is None:
            return self.faces[0]
        yaw, pitch = pose_proxies(target_face.kps)
        # Pose moves slowly: quantise so runs of similar frames reuse the blend.
        key = (round(yaw * 40), round(pitch * 40))
        if key == self._last_key:
            return self._last_face
        w = self.weights(target_face.kps)
        blended = (w[:, None] * self._normed).sum(axis=0)
        blended = blended / max(float(np.linalg.norm(blended)), 1e-6) * self._scale
        face = type(self.faces[0])(dict(self.faces[int(np.argmax(w))]))
        face.embedding = blended.astype(np.float32)   # normed_embedding derives from it
        self._last_key, self._last_face = key, face
        return face


def all_paths() -> List[str]:
    """Main source photo first, then the extra photos (deduplicated)."""
    paths = [G.source_path] if G.source_path else []
    for p in G.source_paths_extra:
        if p and p not in paths:
            paths.append(p)
    return paths


def load(paths: Sequence[str]) -> Tuple[Optional[SourceIdentity], List[str]]:
    """Analyse the photos; returns the identity and the paths that were skipped."""
    from modules import imread_unicode
    from modules.face_analyser import get_one_face

    faces, kept, skipped = [], [], []
    for path in paths:
        image = imread_unicode(path) if os.path.isfile(path) else None
        face = get_one_face(image) if image is not None else None
        if face is None or getattr(face, "embedding", None) is None:
            skipped.append(path)
            continue
        faces.append(face)
        kept.append(path)
    if skipped:
        print(f"[source] skipped (no face / unreadable): {skipped}", flush=True)
    return (SourceIdentity(faces, kept) if faces else None), skipped


def current() -> Optional[SourceIdentity]:
    """Cached identity for the current photo set, or None when the feature is off."""
    if not G.multi_source:
        return None
    paths = tuple(all_paths())
    if len(paths) < 2:
        return None
    with _lock:
        if _cache["key"] != paths:
            _cache["identity"], _ = load(paths)
            _cache["key"] = paths
        return _cache["identity"]


def face_for(source_face, target_face):
    """Pose-blended source face when multi-photo mode is on, else source_face."""
    identity = current()
    if identity is None or len(identity) < 2:
        return source_face
    return identity.face_for(target_face)


def reset() -> None:
    with _lock:
        _cache.update(key=None, identity=None)
