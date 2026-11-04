"""DFL-style face autoencoders.

DF:   shared encoder + bottleneck, one decoder per identity (best likeness).
LIAE: shared encoder + decoder, two bottlenecks (AB shared, B camera-only);
      the swap decodes the shared part twice, so lighting/pose transfer better.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act():
    return nn.LeakyReLU(0.1, inplace=True)


class Down(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 5, 2, 2), _act())


class Up(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout * 4, 3, 1, 1), _act(), nn.PixelShuffle(2))


class Res(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        return F.leaky_relu(x + self.c2(F.leaky_relu(self.c1(x), 0.2)), 0.2)


class Encoder(nn.Module):
    """resolution -> resolution/32 spatial, 8*ch channels."""

    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(Down(3, ch), Down(ch, ch * 2), Down(ch * 2, ch * 4),
                                 Down(ch * 4, ch * 8), Down(ch * 8, ch * 8))

    def forward(self, x):
        return self.net(x)


class Inter(nn.Module):
    """Dense bottleneck: forces an identity-free latent, then back to a grid."""

    def __init__(self, resolution, ch=64, ae_dims=256, ae_ch=512):
        super().__init__()
        self.lowres = resolution // 32
        self.ae_ch = ae_ch
        self.fc1 = nn.Linear(ch * 8 * self.lowres ** 2, ae_dims)
        self.fc2 = nn.Linear(ae_dims, ae_ch * self.lowres ** 2)
        self.up = Up(ae_ch, ae_ch)

    def forward(self, x):
        x = self.fc2(self.fc1(x.flatten(1)))
        return self.up(x.view(-1, self.ae_ch, self.lowres, self.lowres))


class Decoder(nn.Module):
    """resolution/16 latent grid -> RGB face + mask (both 0..1)."""

    def __init__(self, ae_ch=512):
        super().__init__()
        self.img = nn.Sequential(Up(ae_ch, 512), Res(512), Up(512, 256), Res(256),
                                 Up(256, 128), Res(128), Up(128, 64))
        self.to_rgb = nn.Conv2d(64, 3, 1)
        self.mask = nn.Sequential(Up(ae_ch, 256), Up(256, 128), Up(128, 64), Up(64, 32))
        self.to_mask = nn.Conv2d(32, 1, 1)

    def forward(self, z):
        return torch.sigmoid(self.to_rgb(self.img(z))), torch.sigmoid(self.to_mask(self.mask(z)))


class DFModel(nn.Module):
    """arch 'df' or 'liae'. All entry points take NCHW BGR 0..1 and return (rgb, mask)."""

    def __init__(self, resolution=256, arch="df"):
        super().__init__()
        assert resolution % 32 == 0, "resolution must be a multiple of 32"
        assert arch in ("df", "liae")
        self.resolution = resolution
        self.arch = arch
        self.encoder = Encoder()
        if arch == "df":
            self.inter = Inter(resolution)
            self.dec_src = Decoder()
            self.dec_dst = Decoder()
        else:
            self.inter_ab = Inter(resolution, ae_ch=256)
            self.inter_b = Inter(resolution, ae_ch=256)
            self.decoder = Decoder(ae_ch=512)

    def src_recon(self, x):
        if self.arch == "df":
            return self.dec_src(self.inter(self.encoder(x)))
        ab = self.inter_ab(self.encoder(x))
        return self.decoder(torch.cat([ab, ab], 1))

    def dst_recon(self, x):
        if self.arch == "df":
            return self.dec_dst(self.inter(self.encoder(x)))
        e = self.encoder(x)
        return self.decoder(torch.cat([self.inter_b(e), self.inter_ab(e)], 1))

    def swap(self, x):
        """Camera face in, source identity out."""
        if self.arch == "df":
            return self.dec_src(self.inter(self.encoder(x)))
        ab = self.inter_ab(self.encoder(x))
        return self.decoder(torch.cat([ab, ab], 1))


class Discriminator(nn.Module):
    """Small PatchGAN for the optional GAN fine-tuning (sharper detail)."""

    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), _act(),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), _act(),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), _act(),
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), _act(),
            nn.Conv2d(ch * 4, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class DFMExport(nn.Module):
    """DeepFaceLive DFM interface: in_face:0 [1,S,S,3] BGR 0..1 (NHWC) ->
    out_face_mask:0, out_celeb_face:0, out_celeb_face_mask:0 (NHWC)."""

    def __init__(self, model: DFModel):
        super().__init__()
        self.m = model

    def forward(self, in_face):
        x = in_face.permute(0, 3, 1, 2)
        celeb, celeb_mask = self.m.swap(x)
        _, dst_mask = self.m.dst_recon(x)
        nhwc = lambda t: t.permute(0, 2, 3, 1)
        return nhwc(dst_mask), nhwc(celeb), nhwc(celeb_mask)
