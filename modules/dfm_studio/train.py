"""DFM Studio trainer (run as a subprocess): python -m modules.dfm_studio.train <workspace>

Reads config.json (resolution, batch, target_iters), trains on src/ and dst/
face sets, writes status.json every ~2 s, preview.jpg every ~30 s and
model.pt every 1000 iterations (atomic). Stops cleanly when the file STOP
appears in the workspace, or with --export only exports the current model.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from modules.dfm_studio.model import DFModel, DFMExport, Discriminator


def _load_set(folder: str, resolution: int, limit: int = 3000):
    names = sorted(n for n in os.listdir(folder) if n.endswith(".jpg") and "_mask" not in n)
    if len(names) > limit:   # spread the subset over the whole set
        names = [names[i] for i in np.linspace(0, len(names) - 1, limit).astype(int)]
    imgs, masks = [], []
    for n in names:
        img = cv2.imdecode(np.fromfile(os.path.join(folder, n), np.uint8), cv2.IMREAD_COLOR)
        mask_path = os.path.join(folder, n[:-4] + "_mask.png")
        mask = cv2.imdecode(np.fromfile(mask_path, np.uint8), cv2.IMREAD_GRAYSCALE) \
            if os.path.exists(mask_path) else np.full(img.shape[:2], 255, np.uint8)
        if img.shape[0] != resolution:
            img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        imgs.append(img)
        masks.append(mask)
    if not imgs:
        raise SystemExit(f"no faces in {folder}")
    return (torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(np.stack(masks))[:, None].contiguous())


def _gauss(x, sigma):
    k = int(sigma * 3) * 2 + 1
    return torch.nn.functional.conv2d(
        F.pad(x, (k // 2,) * 4, mode="reflect"),
        _kernel(k, sigma, x.device, x.dtype).expand(x.shape[1], 1, k, k), groups=x.shape[1])


_KERNELS = {}


def _kernel(k, sigma, device, dtype):
    key = (k, sigma, device, dtype)
    if key not in _KERNELS:
        ax = torch.arange(k, device=device, dtype=torch.float32) - k // 2
        g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        g = (g / g.sum())
        _KERNELS[key] = (g[:, None] * g[None, :]).to(dtype)[None, None]
    return _KERNELS[key]


def dssim(a, b, sigma=1.5):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = _gauss(a, sigma), _gauss(b, sigma)
    sa = _gauss(a * a, sigma) - mu_a ** 2
    sb = _gauss(b * b, sigma) - mu_b ** 2
    sab = _gauss(a * b, sigma) - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return ((1 - ssim) / 2).mean()


def _augment(img, mask, warp: bool):
    """Random affine (+flip) on target and mask; warped copy as network input."""
    n, _, h, w = img.shape
    dev = img.device
    ang = (torch.rand(n, device=dev) - 0.5) * math.radians(20)
    sc = 1 + (torch.rand(n, device=dev) - 0.5) * 0.1
    tx, ty = ((torch.rand(2, n, device=dev) - 0.5) * 0.1)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = torch.cos(ang) / sc * flip
    theta[:, 0, 1] = -torch.sin(ang) / sc
    theta[:, 1, 0] = torch.sin(ang) / sc * flip
    theta[:, 1, 1] = torch.cos(ang) / sc
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    target = F.grid_sample(img, grid, padding_mode="border", align_corners=False)
    tmask = F.grid_sample(mask, grid, padding_mode="zeros", align_corners=False)
    inp = target
    if warp:
        # DFL-style random warp: smooth low-res displacement of the input only.
        disp = (torch.rand(n, 2, 5, 5, device=dev) - 0.5) * 0.08
        disp = F.interpolate(disp, size=(h, w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        ident = F.affine_grid(torch.eye(2, 3, device=dev)[None].expand(n, 2, 3), img.shape,
                              align_corners=False)
        inp = F.grid_sample(target, ident + disp, padding_mode="border", align_corners=False)
    # The DFM loader sharpens the live crop (DeepFaceLive does); train on the same.
    inp = (inp * 1.75 - _gauss(inp, 2.0) * 0.75).clamp(0, 1)
    return inp, target, tmask


def _status(ws, **kw):
    tmp = os.path.join(ws, "status.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(kw, f)
    os.replace(tmp, os.path.join(ws, "status.json"))


def export(ws: str, model: DFModel, name: str) -> str:
    from modules.paths import MODELS_DIR

    out_dir = os.path.join(MODELS_DIR, "dfm")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_{model.resolution}.dfm")
    wrapped = DFMExport(model).float().eval().cpu()
    with torch.no_grad():
        torch.onnx.export(
            wrapped, (torch.zeros(1, model.resolution, model.resolution, 3),), path,
            input_names=["in_face:0"],
            output_names=["out_face_mask:0", "out_celeb_face:0", "out_celeb_face_mask:0"],
            opset_version=17, dynamo=False,
        )
    return path


def main() -> None:
    ws = sys.argv[1]
    cfg = json.load(open(os.path.join(ws, "config.json"), encoding="utf-8"))
    res, batch = int(cfg.get("resolution", 256)), int(cfg.get("batch", 8))
    target_iters = int(cfg.get("target_iters", 100000))
    arch = cfg.get("arch", "df")
    gan_power = float(cfg.get("gan_power", 0.0))
    init_from = cfg.get("init_from") or ""
    name = cfg.get("name", os.path.basename(ws))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ckpt_path = os.path.join(ws, "model.pt")
    ck = torch.load(ckpt_path, map_location=dev) if os.path.exists(ckpt_path) else None
    if ck is not None:
        arch = ck.get("arch", "df")   # a trained model keeps its architecture
    model = DFModel(res, arch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.5, 0.999))
    disc = Discriminator().to(dev)
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-5, betas=(0.5, 0.999))
    it = 0
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "disc" in ck:
            disc.load_state_dict(ck["disc"])
            opt_d.load_state_dict(ck["opt_d"])
        it = int(ck["iter"])
    elif init_from:
        # Base model: start from another trained workspace (same arch + resolution).
        base_path = os.path.join(os.path.dirname(ws), init_from, "model.pt")
        base = torch.load(base_path, map_location=dev)["model"]
        own = model.state_dict()
        same = {k: v for k, v in base.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(same, strict=False)
        print(f"[dfm] initialised {len(same)}/{len(own)} tensors from {init_from}", flush=True)

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it, "arch": arch,
                    "res": res, "disc": disc.state_dict(), "opt_d": opt_d.state_dict()},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    if "--export" in sys.argv:
        print(export(ws, model, name), flush=True)
        return

    # Face sets live on the GPU as uint8 (3000 faces at 256 px ~ 0.6 GB each).
    src_img, src_mask = (t.to(dev) for t in _load_set(os.path.join(ws, "src"), res))
    dst_img, dst_mask = (t.to(dev) for t in _load_set(os.path.join(ws, "dst"), res))
    print(f"[dfm] src {len(src_img)} faces, dst {len(dst_img)} faces, res {res}, "
          f"batch {batch}, from iter {it}, device {dev}", flush=True)
    stop_path = os.path.join(ws, "STOP")
    if os.path.exists(stop_path):
        os.remove(stop_path)

    def batch_of(imgs, masks):
        idx = torch.randint(0, len(imgs), (batch,), device=imgs.device)
        x = imgs[idx].float() / 255.0
        m = masks[idx].float() / 255.0
        return x, m

    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
    t_status = t_prev = time.time()
    t_preview, last_it, loss_avg = 0.0, it, None
    while it < target_iters and not os.path.exists(stop_path):
        sx, sm = batch_of(src_img, src_mask)
        dx, dm = batch_of(dst_img, dst_mask)
        s_in, s_tg, s_m = _augment(sx, sm, warp=True)
        d_in, d_tg, d_m = _augment(dx, dm, warp=True)
        with torch.autocast(**amp):
            s_out, s_mo = model.src_recon(s_in)
            d_out, d_mo = model.dst_recon(d_in)
        s_out, s_mo, d_out, d_mo = s_out.float(), s_mo.float(), d_out.float(), d_mo.float()
        s_mb, d_mb = _gauss(s_m, 2.0), _gauss(d_m, 2.0)   # soft mask edges
        loss = (10 * dssim(s_out * s_mb, s_tg * s_mb) + 10 * F.mse_loss(s_out * s_mb, s_tg * s_mb)
                + 10 * dssim(d_out * d_mb, d_tg * d_mb) + 10 * F.mse_loss(d_out * d_mb, d_tg * d_mb)
                + 10 * F.mse_loss(s_mo, s_m) + 10 * F.mse_loss(d_mo, d_m))
        if gan_power > 0:
            # GAN fine-tuning on the source face: adds the high-frequency detail
            # DSSIM/MSE average away. Turn on after the model already looks right.
            fake = s_out * s_mb
            loss = loss + gan_power * F.softplus(-disc(fake)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if gan_power > 0:
            d_loss = (F.softplus(-disc(s_tg * s_mb)).mean()
                      + F.softplus(disc(fake.detach())).mean())
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
        it += 1
        loss_avg = float(loss) if loss_avg is None else loss_avg * 0.98 + float(loss) * 0.02

        now = time.time()
        if now - t_status > 2:
            speed = (it - last_it) / (now - t_prev)
            t_prev, last_it, t_status = now, it, now
            _status(ws, iter=it, target=target_iters, loss=round(loss_avg, 4),
                    it_s=round(speed, 2), eta_min=round((target_iters - it) / max(speed, 1e-3) / 60, 1),
                    running=True)
        if now - t_preview > 30 or it == target_iters:
            t_preview = now
            with torch.no_grad(), torch.autocast(**amp):
                swap, _ = model.swap(d_in[:4])
                rec, _ = model.dst_recon(d_in[:4])
            rows = [torch.cat([a, b, c], dim=2) for a, b, c in zip(d_tg[:4], rec.float(), swap.float())]
            grid = (torch.cat(rows, dim=1).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            cv2.imencode(".jpg", grid)[1].tofile(os.path.join(ws, "preview.jpg"))
        if it % 1000 == 0:
            save()

    save()
    _status(ws, iter=it, target=target_iters, loss=round(loss_avg or 0, 4), it_s=0, eta_min=0,
            running=False)
    print(f"[dfm] stopped at iter {it}", flush=True)


if __name__ == "__main__":
    main()
