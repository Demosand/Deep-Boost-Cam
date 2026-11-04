"""Face Tracking 2.0 for the live webcam path.

Between neural detections the face position is propagated with pyramidal
Lucas-Kanade optical flow (a similarity transform fitted with RANSAC over
feature points inside the face), then smoothed with a One Euro filter.

A lost face is held for a few frames only (grace period), every tracked
step is bounded in translation / rotation / scale, and tracking failure
requests an immediate re-detection, so the mask cannot drift onto the
wrong part of the head during a fast turn.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np
from insightface.app.common import Face

import modules.globals as G

# Optical flow works on a downscaled grayscale frame: tracking quality is
# the same and LK cost scales with pixel count.
_FLOW_MAX_WIDTH = 640
_MIN_POINTS = 8
_RESEED_BELOW = 20


class _OneEuro:
    """Vectorised One Euro filter (Casiez et al. 2012)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


def _face_size(face) -> float:
    x1, y1, x2, y2 = face.bbox[:4]
    return float(max(x2 - x1, y2 - y1, 1.0))


def _apply_similarity(face, M: np.ndarray, det_score: Optional[float] = None) -> Face:
    """Return a new Face with bbox/kps/landmarks moved by 2x3 similarity M."""
    A, t = M[:, :2], M[:, 2]
    scale = math.sqrt(abs(np.linalg.det(A)))
    x1, y1, x2, y2 = face.bbox[:4]
    c = A @ np.array([(x1 + x2) / 2, (y1 + y2) / 2]) + t
    hw, hh = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    out = Face(
        bbox=np.array([c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh], dtype=np.float32),
        kps=(face.kps @ A.T + t).astype(np.float32),
        det_score=face.det_score if det_score is None else det_score,
    )
    lmk = getattr(face, "landmark_2d_106", None)
    if lmk is not None:
        out.landmark_2d_106 = (lmk @ A.T + t).astype(np.float32)
    return out


class FaceTracker:
    """Single-face tracker. Call update() once per frame."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.face = None          # raw (unsmoothed) tracked face, full-res coords
        self.prev_gray = None
        self.points = None        # Nx1x2 float32 in downscaled coords
        self.scale = 1.0
        self.lost = 0
        self.filter = None
        self.need_detect = False  # ask the caller to detect on the next frame
        self.motion = 0.0         # last per-frame motion, fraction of face size
        self.confidence = 0.0
        self.state = "none"       # none | det | flow | hold

    # ── helpers ────────────────────────────────────────────────────────
    def _gray(self, frame):
        h, w = frame.shape[:2]
        s = min(1.0, _FLOW_MAX_WIDTH / max(w, 1))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if s < 1.0:
            g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return g, s

    def _seed(self, gray, face) -> None:
        s = self.scale
        x1, y1, x2, y2 = (face.bbox[:4] * s).astype(int)
        h, w = gray.shape[:2]
        ix, iy = int((x2 - x1) * 0.12), int((y2 - y1) * 0.12)
        mask = np.zeros_like(gray)
        mask[max(0, y1 + iy):min(h, y2 - iy), max(0, x1 + ix):min(w, x2 - ix)] = 255
        pts = cv2.goodFeaturesToTrack(gray, 80, 0.01, 4, mask=mask)
        kps = (face.kps * s).reshape(-1, 1, 2).astype(np.float32)
        self.points = kps if pts is None else np.vstack([pts.astype(np.float32), kps])

    def _flow(self, gray) -> Optional[np.ndarray]:
        """Similarity transform prev->cur in full-res coords, or None if unreliable."""
        if self.prev_gray is None or self.points is None or len(self.points) < _MIN_POINTS:
            return None
        if self.prev_gray.shape != gray.shape:
            return None
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        p0 = self.points
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **lk)
        if p1 is None:
            return None
        p0r, st_r, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **lk)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st_r.ravel() == 1) & (fb < 1.0)
        if good.sum() < _MIN_POINTS:
            return None
        a, b = p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is None or inl is None:
            return None
        inl = inl.ravel().astype(bool)
        ratio = inl.sum() / len(inl)
        if inl.sum() < _MIN_POINTS or ratio < 0.5:
            return None

        # Motion limits: a step outside them is a tracking error, not a head move.
        s = self.scale
        A = M[:, :2]
        scale = math.sqrt(abs(np.linalg.det(A)))
        rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))
        x1, y1, x2, y2 = self.face.bbox[:4] * s
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        shift = np.linalg.norm(A @ c + M[:, 2] - c) / (_face_size(self.face) * s)
        if (shift > G.tracking_max_motion or abs(rot) > G.tracking_max_rotation
                or not 0.85 < scale < 1.18):
            return None

        self.points = b[inl].reshape(-1, 1, 2).astype(np.float32)
        self.confidence = float(self.confidence * 0.98 * (0.5 + 0.5 * ratio))
        self.motion = float(shift)
        M_full = M.copy()
        M_full[:, 2] /= s
        return M_full

    def _smooth(self, face):
        if not G.tracking_smoothing:
            return face
        size = _face_size(face)
        k = float(np.clip(G.tracking_smoothing_strength, 0.0, 1.0))
        if self.filter is None:
            # Lower min_cutoff = smoother at rest; beta lets fast moves through.
            self.filter = _OneEuro(min_cutoff=3.0 - 2.7 * k, beta=4.0 / size)
        self.filter.beta = 4.0 / size
        raw = face.kps.astype(np.float64)
        sm = self.filter(raw, time.perf_counter())
        M, _ = cv2.estimateAffinePartial2D(raw, sm, method=cv2.LMEDS)
        if M is None:
            return face
        return _apply_similarity(face, M)

    # ── main entry ────────────────────────────────────────────────────
    def update(self, frame, detection_ran: bool, detected, want_landmarks: bool = False):
        """Return the face to swap this frame (or None).

        detection_ran: the detector ran on this frame; detected is its result.
        """
        gray, self.scale = self._gray(frame)
        raw = None

        accepted = (detection_ran and detected is not None
                    and float(detected.det_score) >= G.tracking_min_confidence)
        if accepted:
            if self.face is not None:
                self.motion = float(
                    np.linalg.norm(detected.kps.mean(0) - self.face.kps.mean(0))
                    / _face_size(self.face))
                # A detection far from the track is a different face/position:
                # restart smoothing so the filter doesn't drag the mask across.
                if self.motion > 0.5:
                    self.filter = None
            if want_landmarks:
                from modules.face_analyser import ensure_landmarks
                ensure_landmarks(frame, [detected])
            raw = detected
            self.lost = 0
            self.need_detect = False
            self.confidence = float(detected.det_score)
            self.state = "det"
            self._seed(gray, raw)
        elif self.face is not None:
            if G.tracking_optical_flow:
                M = self._flow(gray)
                if M is not None:
                    raw = _apply_similarity(self.face, M, det_score=self.confidence)
                    self.state = "flow"
                    if len(self.points) < _RESEED_BELOW:
                        self._seed(gray, raw)
            elif not detection_ran:
                raw = self.face  # legacy behaviour: hold the last detection
                self.state = "det"

            if detection_ran or raw is None:
                # Detector missed the face, or tracking failed: bounded hold.
                self.lost += 1
                self.need_detect = G.tracking_auto_redetect
                if self.lost > G.tracking_grace_frames:
                    self.reset()
                    self.prev_gray = gray
                    return None
                if raw is None:
                    raw = self.face
                    self.state = "hold"
                    self.motion = 0.0
                    if G.tracking_optical_flow:
                        self._seed(gray, raw)  # stale points would poison the next flow step

        self.prev_gray = gray
        if raw is None:
            return None
        self.face = raw
        return self._smooth(raw)
