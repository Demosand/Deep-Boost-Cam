"""Live pipeline helpers: async detection and late reprojection.

AsyncDetector — face detection in its own thread. The processing thread
submits the newest frame and picks up the latest finished result; ONNX Runtime
releases the GIL during inference, so detection of frame N+1 overlaps the swap
of frame N. A result is a frame or two old by the time it is used, so it is
carried to the current frame with optical flow (shift_faces).

Reprojector — with a slow pipeline (heavy enhancer, ReSwapper) the swap of a
frame is ready long after the camera took it, so the shown face lags the body.
The presenter tracks the face on every raw camera frame (cheap LK flow) and
moves the last finished swap region onto the newest frame: position is at
most one camera frame old; only the expression keeps the processing latency.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from typing import Callable, Optional

import cv2
import numpy as np

from modules.face_tracker import _apply_similarity

_FLOW_MAX_WIDTH = 640
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def _gray(frame: np.ndarray):
    h, w = frame.shape[:2]
    s = min(1.0, _FLOW_MAX_WIDTH / max(w, 1))
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if s < 1.0:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return g, s


def _seed(gray, bbox, kps, s) -> np.ndarray:
    x1, y1, x2, y2 = (np.asarray(bbox[:4], np.float32) * s).astype(int)
    h, w = gray.shape[:2]
    ix, iy = int((x2 - x1) * 0.12), int((y2 - y1) * 0.12)
    mask = np.zeros_like(gray)
    mask[max(0, y1 + iy):min(h, y2 - iy), max(0, x1 + ix):min(w, x2 - ix)] = 255
    pts = cv2.goodFeaturesToTrack(gray, 60, 0.01, 4, mask=mask)
    base = (np.asarray(kps, np.float32) * s).reshape(-1, 1, 2)
    return base if pts is None else np.vstack([pts.astype(np.float32), base])


def _similarity(g0, g1, pts, s) -> Optional[tuple]:
    """Similarity g0->g1 in full-res coords from tracked points, plus kept points."""
    if pts is None or len(pts) < 6:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, **_LK)
    if p1 is None:
        return None
    p0r, st_r, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **_LK)
    fb = np.linalg.norm((pts - p0r).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st_r.ravel() == 1) & (fb < 1.0)
    if good.sum() < 6:
        return None
    M, inl = cv2.estimateAffinePartial2D(pts[good].reshape(-1, 2), p1[good].reshape(-1, 2),
                                         method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl is None or inl.sum() < 6:
        return None
    M = M.copy()
    M[:, 2] /= s
    return M, p1[good][inl.ravel().astype(bool)].reshape(-1, 1, 2)


def shift_faces(faces, frame_from: np.ndarray, frame_to: np.ndarray):
    """Carry detections made on frame_from onto frame_to (same size)."""
    if faces is None or frame_from is frame_to:
        return faces
    single = not isinstance(faces, (list, tuple))
    items = [faces] if single else list(faces)
    g0, s = _gray(frame_from)
    g1, _ = _gray(frame_to)
    out = []
    for face in items:
        res = _similarity(g0, g1, _seed(g0, face.bbox, face.kps, s), s)
        out.append(face if res is None else _apply_similarity(face, res[0]))
    return out[0] if single else out


class AsyncDetector:
    """Runs detect_fn on the newest submitted frame in a background thread."""

    def __init__(self, detect_one: Callable, detect_many: Callable):
        self._detect_one = detect_one
        self._detect_many = detect_many
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._job = None
        self._result = None
        self._thread = threading.Thread(target=self._run, name="async-detector", daemon=True)
        self._thread.start()

    def submit(self, seq: int, frame: np.ndarray, many: bool) -> None:
        with self._lock:
            self._job = (seq, frame, many)
        self._wake.set()

    def take(self):
        """(seq, frame, faces, many) of the newest finished detection, once."""
        with self._lock:
            result, self._result = self._result, None
        return result

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(0.05):
                continue
            with self._lock:
                job, self._job = self._job, None
                self._wake.clear()
            if job is None:
                continue
            seq, frame, many = job
            try:
                faces = self._detect_many(frame) if many else self._detect_one(frame)
            except Exception as error:  # never kill the live loop
                print(f"[async-detector] {error}", flush=True)
                faces = None
            with self._lock:
                self._result = (seq, frame, faces, many)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2.0)


class Reprojector:
    """Moves the last processed face onto the newest raw frame (single face)."""

    RING = 16      # raw grays kept to anchor a result on the frame it came from
    HISTORY = 90   # tracked keypoints remembered per raw frame

    def __init__(self):
        self._lock = threading.Lock()
        self._ring: "OrderedDict[int, tuple]" = OrderedDict()   # seq -> (gray, scale)
        self._hist: "OrderedDict[int, np.ndarray]" = OrderedDict()  # seq -> kps
        self._kps = None
        self._bbox = None
        self._pts = None
        self._last = None          # (seq, gray, scale)
        self._result = None        # dict(seq, frame, kps, bbox)
        self._stamps = []
        self.processing_fps = 0.0

    # ── raw frames (presenter thread) ─────────────────────────────────
    def on_raw(self, seq: int, frame: np.ndarray) -> None:
        g, s = _gray(frame)
        with self._lock:
            if self._kps is not None and self._last is not None and self._last[1].shape == g.shape:
                res = _similarity(self._last[1], g, self._pts, s)
                if res is not None:
                    M, self._pts = res
                    self._move(M)
                if self._pts is None or len(self._pts) < 20:
                    self._pts = _seed(g, self._bbox, self._kps, s)
                self._hist[seq] = self._kps.copy()
                while len(self._hist) > self.HISTORY:
                    self._hist.popitem(last=False)
            self._ring[seq] = (g, s)
            while len(self._ring) > self.RING:
                self._ring.popitem(last=False)
            self._last = (seq, g, s)

    def _move(self, M) -> None:
        A, t = M[:, :2], M[:, 2]
        self._kps = (self._kps @ A.T + t).astype(np.float32)
        x1, y1, x2, y2 = self._bbox[:4]
        sc = math.sqrt(abs(np.linalg.det(A)))
        c = A @ np.array([(x1 + x2) / 2, (y1 + y2) / 2]) + t
        hw, hh = (x2 - x1) * sc / 2, (y2 - y1) * sc / 2
        self._bbox = np.array([c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh], np.float32)

    # ── processed results (processing thread) ─────────────────────────
    def publish(self, seq: int, frame: np.ndarray, face, raw: Optional[np.ndarray] = None) -> None:
        """raw: the frame before processing; the region the processing changed
        is what gets moved (moving a fixed ellipse would drag old background)."""
        now = time.perf_counter()
        self._stamps = [t for t in self._stamps if now - t < 1.0] + [now]
        self.processing_fps = float(len(self._stamps))
        kps = None if face is None else np.asarray(face.kps, np.float32)
        bbox = None if face is None else np.asarray(face.bbox[:4], np.float32)
        with self._lock:
            if kps is not None:
                if seq in self._hist and self._kps is not None:
                    # Detection/track of the processor corrects our drift.
                    self._kps = self._kps + (kps - self._hist[seq])
                    self._bbox = self._bbox + (bbox - self._bbox) * 0.5
                    self._hist[seq] = kps.copy()
                else:
                    self._anchor(seq, kps, bbox)
            region = _changed_region(frame, raw, bbox) if (bbox is not None and raw is not None) else None
            self._result = dict(seq=seq, frame=frame, kps=kps if region is not None else None,
                                bbox=bbox, region=region)

    def _anchor(self, seq, kps, bbox) -> None:
        """First sighting: carry the result's keypoints from its frame to now."""
        self._kps, self._bbox = kps.copy(), bbox.copy()
        if self._last is None:
            return
        if seq in self._ring and seq != self._last[0]:
            g0, s = self._ring[seq]
            res = _similarity(g0, self._last[1], _seed(g0, bbox, kps, s), s)
            if res is not None:
                self._move(res[0])
        self._hist.clear()
        self._hist[seq] = kps.copy()
        self._hist[self._last[0]] = self._kps.copy()
        self._pts = _seed(self._last[1], self._bbox, self._kps, self._last[2])

    # ── output (presenter thread) ─────────────────────────────────────
    def current_bbox(self):
        with self._lock:
            return None if self._bbox is None else self._bbox.copy()

    def compose(self, seq: int, raw: np.ndarray) -> Optional[np.ndarray]:
        with self._lock:
            r = self._result
            if r is None:
                return None
            if r["kps"] is None or r["seq"] == seq or self._kps is None:
                return r["frame"]
            src = self._hist.get(r["seq"], r["kps"])
            now = self._kps.copy()
        if r["frame"].shape != raw.shape:
            return r["frame"]
        M, _ = cv2.estimateAffinePartial2D(src, now, method=cv2.LMEDS)
        if M is None:
            return r["frame"]
        return _paste_region(raw, r["frame"], r["region"], M)


def _changed_region(processed: np.ndarray, raw: np.ndarray, bbox):
    """(x0, y0, x1, y1, alpha) of the area the swap changed around bbox."""
    h, w = processed.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ax, ay = (x2 - x1) * 0.9, (y2 - y1) * 0.9
    sx0, sy0 = int(max(0, cx - ax)), int(max(0, cy - ay))
    sx1, sy1 = int(min(w, cx + ax)), int(min(h, cy + ay))
    if sx1 <= sx0 or sy1 <= sy0 or raw.shape != processed.shape:
        return None
    diff = cv2.absdiff(processed[sy0:sy1, sx0:sx1], raw[sy0:sy1, sx0:sx1]).max(axis=2)
    mask = (diff > 6).astype(np.uint8)
    size = max(8, int(min(x2 - x1, y2 - y1)))
    k = max(3, size // 8) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    alpha = np.zeros(mask.shape, np.float32)
    cv2.fillConvexPoly(alpha, hull, 1.0)
    # Pull the edge in, then feather: the outer rim is where old background
    # and new frame disagree most.
    e = max(3, size // 25) | 1
    alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
    b = max(3, size // 10) | 1
    return sx0, sy0, sx1, sy1, cv2.GaussianBlur(alpha, (b, b), 0)


def _paste_region(raw, processed, region, M) -> np.ndarray:
    """Warp the processed (changed) face region by M and feather it onto raw."""
    h, w = raw.shape[:2]
    sx0, sy0, sx1, sy1, alpha = region
    # Destination ROI: the source corners moved by M.
    corners = np.array([[sx0, sy0], [sx1, sy0], [sx1, sy1], [sx0, sy1]], np.float32)
    moved = corners @ M[:, :2].T + M[:, 2]
    dx0, dy0 = int(max(0, np.floor(moved[:, 0].min()))), int(max(0, np.floor(moved[:, 1].min())))
    dx1, dy1 = int(min(w, np.ceil(moved[:, 0].max()))), int(min(h, np.ceil(moved[:, 1].max())))
    if dx1 <= dx0 or dy1 <= dy0:
        return raw
    # Map: dst ROI pixel -> src ROI pixel.
    T = M.copy()
    T[:, 2] += T[:, :2] @ np.array([sx0, sy0], np.float32) - np.array([dx0, dy0], np.float32)
    size = (dx1 - dx0, dy1 - dy0)
    patch = cv2.warpAffine(processed[sy0:sy1, sx0:sx1], T, size, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(alpha, T, size, borderValue=0)[..., None]
    out = raw.copy()
    roi = out[dy0:dy1, dx0:dx1]
    out[dy0:dy1, dx0:dx1] = (patch * a + roi * (1.0 - a)).astype(np.uint8)
    return out
