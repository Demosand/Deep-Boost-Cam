"""Swap realism refinements, all computed in inswapper's aligned 128x128 space.

Working in aligned space keeps every step cheap (fixed tiny resolution) and
motion-invariant: the same pixel always means the same part of the face, so
temporal smoothing there does not ghost when the head moves.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np

import modules.globals as G

# insightface 2d106 index groups (same layout face_masking.py uses).
_EYE_R = list(range(33, 43))
_EYE_L = list(range(87, 97))
_LIPS = list(range(52, 72))

_lock = threading.Lock()
_state = {"prev_fake": None, "prev_aimg": None, "color": None, "center": None}


def reset_state() -> None:
    with _lock:
        _state.update(prev_fake=None, prev_aimg=None, color=None, center=None)
    from modules.processors.frame import face_regions
    face_regions.reset_cache()
    _grain_state["stats"] = None


def any_enabled() -> bool:
    return (G.adaptive_face_mask or G.color_matching or G.temporal_stabilization
            or G.eye_preservation or G.mouth_preservation
            or G.face_parts_mask or G.occlusion_mask or G.use_model_mask)


def needs_landmarks() -> bool:
    return G.adaptive_face_mask or G.eye_preservation or G.mouth_preservation


def _aligned_landmarks(face, M) -> Optional[np.ndarray]:
    lmk = getattr(face, "landmark_2d_106", None)
    if lmk is None:
        return None
    return lmk @ M[:, :2].T + M[:, 2]


def _odd(v: float) -> int:
    return max(1, int(v) // 2 * 2 + 1)


def _region_mask(pts: np.ndarray, size: int, dilate: float, blur: float) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.fillConvexPoly(m, hull, 255)
    if dilate > 0:
        k = _odd(dilate)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.GaussianBlur(m, (_odd(blur), _odd(blur)), 0)


def _color_match(fake: np.ndarray, aimg: np.ndarray, size: int) -> np.ndarray:
    """Transfer target skin LAB mean/std onto the swapped face (centre region)."""
    center = _state["center"]
    if center is None or center.shape[0] != size:
        center = np.zeros((size, size), np.uint8)
        cv2.ellipse(center, (size // 2, int(size * 0.55)), (int(size * 0.3), int(size * 0.35)),
                    0, 0, 360, 255, -1)
        _state["center"] = center
    sel = center > 0
    f_lab = cv2.cvtColor(fake, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(aimg, cv2.COLOR_BGR2LAB).astype(np.float32)
    fs, ts = f_lab[sel], t_lab[sel]
    stats = np.concatenate([fs.mean(0), fs.std(0) + 1e-3, ts.mean(0), ts.std(0) + 1e-3])
    # EMA over frames so lighting changes adapt but don't flicker.
    prev = None if G.many_faces else _state["color"]
    stats = stats if prev is None else 0.7 * prev + 0.3 * stats
    _state["color"] = stats
    fm, fsd, tm, tsd = stats[0:3], stats[3:6], stats[6:9], stats[9:12]
    # Limit contrast change: std ratios outside [0.7, 1.4] look painted.
    ratio = np.clip(tsd / fsd, 0.7, 1.4)
    out = (f_lab - fm) * ratio + tm
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    k = float(np.clip(G.color_matching_strength, 0.0, 1.0))
    return cv2.addWeighted(out, k, fake, 1.0 - k, 0)


def _temporal(fake: np.ndarray, aimg: np.ndarray) -> np.ndarray:
    prev_fake, prev_aimg = _state["prev_fake"], _state["prev_aimg"]
    _state["prev_aimg"] = aimg
    if prev_fake is None or prev_fake.shape != fake.shape:
        _state["prev_fake"] = fake
        return fake
    # Expression change in the real face -> trust the new frame (no lag on
    # talking/blinking); a still face -> blend to kill texture shimmer.
    change = float(cv2.absdiff(aimg, prev_aimg).mean())
    w = float(np.clip(G.temporal_strength, 0.0, 0.9)) * float(np.exp(-change / 6.0))
    out = cv2.addWeighted(prev_fake, w, fake, 1.0 - w, 0) if w > 0.01 else fake
    _state["prev_fake"] = out
    return out


def _preserve(fake: np.ndarray, aimg: np.ndarray, lmk: np.ndarray, size: int) -> np.ndarray:
    alpha = np.zeros((size, size), np.float32)
    s = size / 128.0
    if G.eye_preservation:
        for idx in (_EYE_L, _EYE_R):
            alpha = np.maximum(alpha, _region_mask(lmk[idx], size, 5 * s, 9 * s) / 255.0
                               * float(np.clip(G.eye_preservation_strength, 0, 1)))
    if G.mouth_preservation:
        alpha = np.maximum(alpha, _region_mask(lmk[_LIPS], size, 4 * s, 9 * s) / 255.0
                           * float(np.clip(G.mouth_preservation_strength, 0, 1)))
    a = alpha[..., None]
    return (aimg * a + fake * (1.0 - a)).astype(np.uint8)


def adaptive_alpha(lmk: np.ndarray, size: int, ellipse: np.ndarray) -> np.ndarray:
    """Blend mask following the face contour + brows instead of a fixed ellipse."""
    f = float(np.clip(G.adaptive_mask_feather, 0.0, 1.0))
    m = np.zeros((size, size), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(lmk.astype(np.int32)), 255)
    # Pull the edge inside the contour so the feather never shows background.
    k = _odd(size * (0.03 + 0.05 * f))
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    b = _odd(size * (0.06 + 0.18 * f))
    m = cv2.GaussianBlur(m, (b, b), 0)
    return cv2.min(m, ellipse)


def refine(fake: np.ndarray, frame: np.ndarray, face, M: np.ndarray,
           ellipse: np.ndarray, model_mask: Optional[np.ndarray] = None,
           force_model_mask: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (refined bgr_fake, alpha template or None for the default).

    model_mask: the swap model's own face mask (HyperSwap), aligned space.
    """
    size = fake.shape[0]
    aimg = cv2.warpAffine(frame, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
    lmk = _aligned_landmarks(face, M) if needs_landmarks() else None
    with _lock:
        if G.color_matching:
            fake = _color_match(fake, aimg, size)
        if G.temporal_stabilization and not G.many_faces:
            fake = _temporal(fake, aimg)
        else:
            _state["prev_fake"] = None
        if lmk is not None and (G.eye_preservation or G.mouth_preservation):
            fake = _preserve(fake, aimg, lmk, size)
    alpha = adaptive_alpha(lmk, size, ellipse) if (G.adaptive_face_mask and lmk is not None) else None

    # Multiplicative masks: each can only remove swap coverage, never add it.
    factors = []
    if (G.use_model_mask or force_model_mask) and model_mask is not None:
        factors.append(cv2.resize(model_mask, (size, size)) if model_mask.shape[0] != size else model_mask)
    if G.face_parts_mask or G.occlusion_mask:
        from modules.processors.frame import face_regions
        if G.face_parts_mask:
            factors.append(face_regions.parts_mask(aimg))
        if G.occlusion_mask:
            factors.append(face_regions.occlusion_mask(aimg))
    if factors:
        base = (ellipse if alpha is None else alpha).astype(np.float32)
        for f in factors:
            base *= f
        alpha = np.clip(base, 0, 255).astype(np.uint8)
    return fake, alpha


# ─── Grain & sharpness matching (frame space, inside paste-back) ───
# Swap models output a clean, smooth face; next to a noisy webcam image that
# smoothness gives the swap away. Measured on the real frame and the pasted
# face over the same region, at frame pixel scale (doing it in the 128/256 px
# aligned crop would turn grain into blobs after the paste-back upscale).
_grain_state = {"stats": None, "bank": None}
_BANK = 1024


def _noise_bank():
    """Pre-made unit-sigma grain (luma slightly blurred like real sensor noise
    after demosaic/compression, plus weaker chroma). Drawing ~1M normals per
    frame cost ~15 ms; a random window into this bank is free."""
    if _grain_state["bank"] is None:
        rng = np.random.default_rng()
        luma = cv2.GaussianBlur(rng.standard_normal((_BANK, _BANK)).astype(np.float32), (0, 0), 0.6)
        luma /= float(luma.std())
        chroma = rng.standard_normal((_BANK, _BANK, 3)).astype(np.float32) * 0.25
        # One 3-channel bank: a single cv2.scaleAdd per frame (numpy broadcasting
        # of luma + chroma cost ~6 ms on a 700 px face).
        _grain_state["bank"] = luma[..., None] + chroma
    return _grain_state["bank"]


def _bank_window(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if h > _BANK or w > _BANK:
        reps = (-(-h // _BANK), -(-w // _BANK)) + (1,) * (arr.ndim - 2)
        arr = np.tile(arr, reps)
    y = np.random.randint(0, arr.shape[0] - h + 1)
    x = np.random.randint(0, arr.shape[1] - w + 1)
    return arr[y:y + h, x:x + w]


def _hf_stats(gray: np.ndarray, sel: np.ndarray) -> Tuple[float, float]:
    """(noise sigma, detail energy) of a float32 gray image over sel."""
    b1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    noise = np.abs(gray - b1)[sel]
    sigma = 1.4826 * float(np.median(noise)) if noise.size else 0.0   # robust (MAD) sigma
    band = np.abs(b1 - cv2.GaussianBlur(gray, (0, 0), 2.5))[sel]
    return sigma, float(band.mean()) if band.size else 0.0


def match_grain(fake: np.ndarray, target: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Sharpen/soften the pasted face to the frame's detail level and add
    sensor-like grain up to the frame's noise level (uint8 BGR in, out)."""
    sel = alpha > 128
    if sel.sum() < 400:
        return fake
    step = max(1, int(np.sqrt(sel.sum() / 20000)))   # subsample big faces for the stats
    f_gray = cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    s = sel[::step, ::step]
    n_t, e_t = _hf_stats(t_gray[::step, ::step], s)
    n_f, e_f = _hf_stats(f_gray[::step, ::step], s)
    stats = np.array([n_t, e_t, n_f, e_f], np.float32)
    prev = None if G.many_faces else _grain_state["stats"]
    stats = stats if prev is None else 0.8 * prev + 0.2 * stats   # no frame-to-frame flicker
    _grain_state["stats"] = stats
    n_t, e_t, n_f, e_f = (float(x) for x in stats)

    # Detail: unsharp mask when the swap is softer than the camera, blur when
    # sharper (uint8 cv2 ops: ~0.7 ms vs ~4 ms in float numpy at 700 px).
    amount = float(np.clip(e_t / max(e_f, 1e-3) - 1.0, -0.6, 1.5)) * float(G.grain_sharpness_amount)
    out = fake
    if abs(amount) > 0.03:
        blur = cv2.GaussianBlur(fake, (0, 0), 1.2)
        # amount < 0 gives fake*(1-|a|) + blur*|a|: a blend toward the blur.
        out = cv2.addWeighted(fake, 1.0 + amount, blur, -amount, 0)
    # Grain: add only the missing part of the noise (variances add).
    add = np.sqrt(max(n_t ** 2 - n_f ** 2, 0.0)) * float(G.grain_amount)
    if add > 0.2:
        h, w = out.shape[:2]
        grain = np.ascontiguousarray(_bank_window(_noise_bank(), h, w))
        noisy = cv2.scaleAdd(grain, add, out.astype(np.float32))
        out = np.clip(noisy, 0, 255, out=noisy).astype(np.uint8)
    return out
