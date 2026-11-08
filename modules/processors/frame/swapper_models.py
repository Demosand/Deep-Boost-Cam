"""Selectable face swap models.

Every model is exposed through the slice of insightface's ``INSwapper`` API the
pipeline uses — ``input_size``, ``session``, ``input_names``/``output_names``
and ``get(img, target_face, source_face, paste_back=False) -> (bgr_fake, M)`` —
so the paste-back, masks and realism code work unchanged for 128 and 256 px.

* inswapper / ReSwapper: insightface's INSwapper class (RGB [0,1] target,
  latent = normed ArcFace embedding @ emap, emap = last graph initializer).
  ReSwapper 256 is the "originalInswapperClassCompatible" export. insightface's
  model router only maps 128 px graphs to INSwapper, so it is built directly.
* HyperSwap 1A/1B/1C: target in [-1,1], plain normed embedding, and a second
  output with the model's own face mask (kept in ``last_mask``).
"""

from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional

import cv2
import numpy as np
import onnxruntime
from insightface.utils import face_align

_FF = "https://github.com/facefusion/facefusion-assets/releases/download/"


class SwapperSpec(NamedTuple):
    key: str
    label: str
    model_file: str
    size: int
    kind: str           # 'inswapper' | 'hyperswap' | 'dfm'
    url: str
    # TensorRT FP16 measured equal to the CUDA output for this model
    # (RTX 5060 Ti, max|diff| vs CUDA): HyperSwap 0.010 -> on; ReSwapper 0.197
    # and inswapper 0.457 -> off (visible artefacts).
    trt_fp16_ok: bool = False
    # CUDA-graph capture; DFM graphs (TF exports) keep nodes on the CPU.
    cuda_graph: bool = True


SWAPPERS: List[SwapperSpec] = [
    SwapperSpec("inswapper_128", "Inswapper 128", "inswapper_128_fp16.onnx", 128, "inswapper",
                "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"),
    SwapperSpec("reswapper_256", "ReSwapper 256", "reswapper_256.onnx", 256, "inswapper",
                "https://huggingface.co/somanchiu/reswapper/resolve/main/"
                "reswapper_256-1567500_originalInswapperClassCompatible.onnx"),
    SwapperSpec("hyperswap_1a_256", "HyperSwap 1A 256", "hyperswap_1a_256.onnx", 256, "hyperswap",
                _FF + "models-3.3.0/hyperswap_1a_256.onnx", trt_fp16_ok=True),
    SwapperSpec("hyperswap_1b_256", "HyperSwap 1B 256", "hyperswap_1b_256.onnx", 256, "hyperswap",
                _FF + "models-3.3.0/hyperswap_1b_256.onnx", trt_fp16_ok=True),
    SwapperSpec("hyperswap_1c_256", "HyperSwap 1C 256", "hyperswap_1c_256.onnx", 256, "hyperswap",
                _FF + "models-3.3.0/hyperswap_1c_256.onnx", trt_fp16_ok=True),
]
DEFAULT_KEY = "inswapper_128"
_BUILTIN = list(SWAPPERS)
DFM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "models", "dfm")


def _dfm_size(file_name: str) -> int:
    """DeepFaceLive names end with the resolution (name_320.dfm)."""
    tail = os.path.splitext(file_name)[0].rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def refresh_dfm() -> None:
    """Re-scan models/dfm for .dfm files (DeepFaceLive / DFM Studio exports)."""
    found = []
    if os.path.isdir(DFM_DIR):
        for name in sorted(os.listdir(DFM_DIR)):
            if name.lower().endswith(".dfm"):
                stem = os.path.splitext(name)[0]
                found.append(SwapperSpec(f"dfm_{stem}", f"DFM: {stem}", f"dfm/{name}",
                                         _dfm_size(name), "dfm", "", cuda_graph=False))
    SWAPPERS[:] = _BUILTIN + found
    BY_KEY.clear()
    BY_KEY.update({s.key: s for s in SWAPPERS})


BY_KEY: Dict[str, SwapperSpec] = {}
refresh_dfm()


def needs_source(key: str) -> bool:
    """DFM models carry their identity; every other model needs a source photo."""
    return spec_for(key).kind != "dfm"


def spec_for(key: str) -> SwapperSpec:
    return BY_KEY.get(key, BY_KEY[DEFAULT_KEY])


def ensure_model_file(spec: SwapperSpec, models_dir: str) -> Optional[str]:
    """Path of the model, downloading it into models/ on first use."""
    path = os.path.join(models_dir, spec.model_file)
    if os.path.isfile(path):
        return path
    if not spec.url:
        return None
    from modules.model_downloader import fetch_url

    print(f"[swapper] downloading {spec.label} ...", flush=True)
    return fetch_url(spec.url, path)


class HyperSwapper:
    """HyperSwap 1A/1B/1C driven through onnxruntime with the INSwapper API."""

    def __init__(self, model_path: str, providers: list):
        self.model_file = model_path
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        shape = next(i.shape for i in self.session.get_inputs() if i.name == "target")
        self.input_size = (shape[3], shape[2])
        self.last_mask: Optional[np.ndarray] = None

    def get(self, img, target_face, source_face, paste_back=False):
        size = self.input_size[0]
        aimg, M = face_align.norm_crop2(img, target_face.kps, size)
        blob = (aimg[:, :, ::-1].astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        emb = np.asarray(source_face.normed_embedding, np.float32).reshape(1, -1)
        out = self.session.run(None, {"source": emb, "target": np.ascontiguousarray(blob)})
        fake = (out[0][0].transpose(1, 2, 0) + 1.0) * 127.5
        bgr_fake = np.clip(fake, 0, 255)[:, :, ::-1].astype(np.uint8)
        self.last_mask = np.clip(out[1][0, 0], 0, 1) if len(out) > 1 else None
        return bgr_fake, M


# DeepFaceLab "whole face" alignment template (unit square), used by DFM models.
DFL_WHOLE_FACE = np.array([
    [0.35342266, 0.39285716], [0.62797622, 0.39285716], [0.48660713, 0.54017860],
    [0.38839287, 0.68750011], [0.59821427, 0.68750011],
], np.float32)


class DFMSwapper:
    """DeepFaceLive .dfm (ONNX): one baked-in identity, no source photo.

    Input in_face:0 [1,S,S,3] BGR 0..1 on a DFL whole-face crop; outputs the
    target face mask, the celebrity face and the celebrity face mask. The
    model's own mask is always used (it is what DeepFaceLive composites with).
    """

    always_mask = True

    def __init__(self, model_path: str, providers: list):
        self.model_file = model_path
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = {i.name: i for i in self.session.get_inputs()}
        self.input_names = list(inputs)
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.in_name = "in_face:0" if "in_face:0" in inputs else self.input_names[0]
        self.morph_name = "morph_value:0" if "morph_value:0" in inputs else None
        size = inputs[self.in_name].shape[1]
        self.input_size = (int(size), int(size)) if isinstance(size, int) and size > 0 else (224, 224)
        names = self.output_names

        def find(*keys):
            return next((i for i, n in enumerate(names) if all(k in n for k in keys)), None)

        celeb_mask = find("celeb_face_mask")
        celeb = next((i for i, n in enumerate(names) if "celeb_face" in n and "mask" not in n), None)
        dst_mask = next((i for i, n in enumerate(names) if "face_mask" in n and "celeb" not in n), None)
        # DeepFaceLive export order when names are unknown.
        self._idx = (dst_mask if dst_mask is not None else 0,
                     celeb if celeb is not None else 1,
                     celeb_mask if celeb_mask is not None else 2)
        self.last_mask: Optional[np.ndarray] = None

    def get(self, img, target_face, source_face=None, paste_back=False):
        size = self.input_size[0]
        M = cv2.estimateAffinePartial2D(np.asarray(target_face.kps, np.float32),
                                        DFL_WHOLE_FACE * size, method=cv2.LMEDS)[0]
        crop = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        # DeepFaceLive sharpens the crop before inference; the models are trained on it.
        crop = cv2.addWeighted(crop, 1.75, cv2.GaussianBlur(crop, (0, 0), 2), -0.75, 0)
        feed = {self.in_name: (crop.astype(np.float32) / 255.0)[None]}
        if self.morph_name is not None:
            feed[self.morph_name] = np.array([1.0], np.float32)
        out = self.session.run(None, feed)
        dst_mask, celeb, celeb_mask = (out[i] for i in self._idx)
        bgr_fake = np.clip(celeb.reshape(size, size, 3) * 255.0, 0, 255).astype(np.uint8)
        mask = np.minimum(dst_mask.reshape(size, size), celeb_mask.reshape(size, size))
        mask = cv2.erode(np.clip(mask, 0, 1).astype(np.float32),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        self.last_mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, size / 36.0))
        return bgr_fake, M


def build(spec: SwapperSpec, model_path: str, providers: list):
    if spec.kind == "dfm":
        return DFMSwapper(model_path, providers)
    if spec.kind == "hyperswap":
        return HyperSwapper(model_path, providers)
    from insightface.model_zoo.inswapper import INSwapper

    session = onnxruntime.InferenceSession(model_path, providers=providers)
    return INSwapper(model_file=model_path, session=session)
