"""Face-part masks (BiSeNet face parser) and occlusion masks (XSeg).

Both work on the swap model's aligned crop (128 or 256 px), the same space
the paste-back alpha lives in, so the result multiplies straight into it:

* parts_mask: 1 where the swap may paint, chosen per region (forehead, brows,
  eyes, nose, cheeks, mouth, chin). Hair, ears, neck and background are never
  painted, so a fringe over the forehead stays real.
* occlusion_mask: 0 over anything in front of the face (hand, mic, cup).

Forehead / cheeks / chin are all "skin" to BiSeNet; they are split by fixed
rows of the arcface alignment template (eyes at 0.40, mouth at 0.72 of the
crop height), which the alignment keeps stable under head roll.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional

import cv2
import numpy as np

import modules.globals as G
from modules.paths import MODELS_DIR

_FF = "https://github.com/facefusion/facefusion-assets/releases/download/"
_MODELS = {
    "bisenet": ("bisenet_resnet_34.onnx", _FF + "models-3.0.0/bisenet_resnet_34.onnx"),
    "xseg": ("xseg_3.onnx", _FF + "models-3.2.0/xseg_3.onnx"),
}
_SESSIONS: Dict[str, object] = {}
_LOCK = threading.Lock()
# Last mask per kind, reused for G.mask_interval frames. Aligned space keeps
# the face fixed in the crop, so a mask a frame or two old still fits.
_CACHE: Dict[str, dict] = {}

# BiSeNet (CelebAMask-HQ) classes -> our regions. 6 = glasses goes with eyes.
REGION_CLASSES = {
    "brows": (2, 3),
    "eyes": (4, 5, 6),
    "nose": (10,),
    "mouth": (11, 12, 13),
}
SKIN = 1
REGIONS = ("forehead", "brows", "eyes", "nose", "cheeks", "mouth", "chin")
# Split rows of the skin class, as a fraction of the aligned crop height.
FOREHEAD_ROW = 0.31
CHIN_ROW = 0.80

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _session(name: str):
    with _LOCK:
        if name not in _SESSIONS:
            file_name, url = _MODELS[name]
            path = os.path.join(MODELS_DIR, file_name)
            if not os.path.isfile(path):
                from modules.model_downloader import fetch_url

                if fetch_url(url, path) is None:
                    raise RuntimeError(f"could not download {file_name}")
            if name == "xseg":
                path = _xseg_cuda_friendly(path)
            from modules.processors.frame._onnx_enhancer import create_onnx_session

            # TRT FP16 measured: BiSeNet logits max|diff| 0.026 (argmax
            # unchanged) -> allowed; XSeg 0.078 on a 0-1 mask -> FP32.
            _SESSIONS[name] = create_onnx_session(path, allow_fp16=(name == "bisenet"))
        return _SESSIONS[name]


def _xseg_cuda_friendly(model_path: str) -> str:
    """Copy of XSeg the CUDA provider can run entirely on the GPU.

    Stock xseg_3 has a dynamic batch dim and ConvTranspose nodes with
    asymmetric pads [0,0,1,1], which the CUDA provider rejects, so those
    upsampling layers fall back to the CPU with host round-trips. Fixing the
    batch to 1 and replacing the asymmetric pad by symmetric-0 + an explicit
    Slice gives the same output fully on CUDA. (Approach from the
    beekamai/DeepLiveCam fork, AGPL-3.0 like this project.)
    """
    out_path = os.path.join(MODELS_DIR, "xseg_3_cuda.onnx")
    if os.path.isfile(out_path):
        return out_path
    try:
        import onnx
        from onnx import helper, numpy_helper

        model = onnx.load(model_path)
        graph = model.graph
        for tensor in (graph.input[0], graph.output[0]):
            dim = tensor.type.tensor_type.shape.dim[0]
            dim.Clear()
            dim.dim_value = 1
        graph.initializer.extend([
            numpy_helper.from_array(np.array([0, 0], np.int64), "xs_starts"),
            numpy_helper.from_array(np.array([-1, -1], np.int64), "xs_ends"),
            numpy_helper.from_array(np.array([2, 3], np.int64), "xs_axes"),
        ])
        nodes = []
        for node in graph.node:
            pads = next((list(a.ints) for a in node.attribute if a.name == "pads"), None)
            if node.op_type != "ConvTranspose" or pads != [0, 0, 1, 1]:
                nodes.append(node)
                continue
            for attr in node.attribute:
                if attr.name == "pads":
                    del attr.ints[:]
                    attr.ints.extend([0, 0, 0, 0])
            cropped = node.output[0]
            node.output[0] = cropped + "_full"
            nodes.append(node)
            nodes.append(helper.make_node("Slice", [node.output[0], "xs_starts", "xs_ends", "xs_axes"],
                                          [cropped], name=node.name + "_crop"))
        del graph.node[:]
        graph.node.extend(nodes)
        onnx.checker.check_model(model)
        onnx.save(model, out_path)
        return out_path
    except Exception as error:
        print(f"[face_regions] XSeg rewrite failed ({error}); using the stock model")
        return model_path


def _cached(kind: str, size: int):
    """Cached mask if it may be reused this frame, else None (and count)."""
    entry = _CACHE.setdefault(kind, {"mask": None, "n": 0})
    interval = max(1, int(G.mask_interval))
    reuse = (interval > 1 and not G.many_faces and entry["mask"] is not None
             and entry["mask"].shape[0] == size and entry["n"] % interval != 0)
    entry["n"] += 1
    return entry["mask"] if reuse else None


def _store(kind: str, mask: np.ndarray) -> np.ndarray:
    _CACHE[kind]["mask"] = mask
    return mask


def reset_cache() -> None:
    _CACHE.clear()


def parts_enabled() -> bool:
    return bool(G.face_parts_mask)


def occlusion_enabled() -> bool:
    return bool(G.occlusion_mask)


def parts_mask(aimg: np.ndarray) -> Optional[np.ndarray]:
    """Float mask (size x size) of the regions selected for swapping."""
    size = aimg.shape[0]
    hit = _cached("parts", size)
    if hit is not None:
        return hit
    session = _session("bisenet")
    x = cv2.resize(aimg, (512, 512), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    x = ((x.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
    logits = session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0]
    # argmax over 19x512x512 costs ~16 ms on the CPU; the mask is only
    # 128/256 px, so take it over the decimated logits (~4 ms at 256).
    step = max(1, logits.shape[1] // size)
    labels = logits[:, ::step, ::step].argmax(0).astype(np.uint8)
    if labels.shape[0] != size:
        labels = cv2.resize(labels, (size, size), interpolation=cv2.INTER_NEAREST)

    keep = np.zeros((size, size), bool)
    for region, classes in REGION_CLASSES.items():
        if G.face_parts.get(region, True):
            keep |= np.isin(labels, classes)
    skin = labels == SKIN
    rows = np.arange(size)[:, None] / size
    for region, band in (("forehead", rows < FOREHEAD_ROW),
                         ("cheeks", (rows >= FOREHEAD_ROW) & (rows < CHIN_ROW)),
                         ("chin", rows >= CHIN_ROW)):
        if G.face_parts.get(region, True):
            keep |= skin & band

    mask = keep.astype(np.float32)
    # Close parser speckles, then feather so region borders don't show.
    k = max(3, size // 64) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    blur = max(3, int(size * (0.02 + 0.06 * float(G.face_parts_feather)))) | 1
    return _store("parts", cv2.GaussianBlur(mask, (blur, blur), 0))


def occlusion_mask(aimg: np.ndarray) -> Optional[np.ndarray]:
    """Float mask (size x size): 1 = visible face, 0 = occluder in front."""
    size = aimg.shape[0]
    hit = _cached("occlusion", size)
    if hit is not None:
        return hit
    session = _session("xseg")
    x = cv2.resize(aimg, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)[None] / 255.0
    out = session.run(None, {session.get_inputs()[0].name: x})[0].reshape(256, 256)
    mask = cv2.resize(np.clip(out, 0, 1), (size, size), interpolation=cv2.INTER_LINEAR)
    # Soften, then push the transition band inward so occluder edges keep no
    # half-swapped halo (same shaping FaceFusion uses for XSeg).
    sigma = max(1.0, size / 64.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return _store("occlusion", (np.clip(mask, 0.5, 1.0) - 0.5) * 2.0)
