"""Simple distortions: additive noise, resize (down then back up), crop.

Every stage samples its parameter from ``gen`` on each call, logs it in ``last_params``, and is
differentiable w.r.t. the input. Ranges are explicit constructor arguments; presets scale them
by severity.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from perceptual_media.core.types import ImageBatch
from perceptual_media.distort.base import Distortion, uniform


class GaussianNoise(Distortion):
    """Additive white Gaussian noise, ``sigma ~ U(0, sigma_max)`` (StegaStamp: ``rnd_noise`` 0.02)."""

    name = "noise"

    def __init__(self, sigma_max: float = 0.02) -> None:
        super().__init__()
        self.sigma_max = sigma_max

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        sigma = uniform(0.0, self.sigma_max, gen)
        self.last_params = {"sigma": sigma}
        if sigma == 0:
            return x
        n = torch.randn(x.shape, generator=gen, device=gen.device, dtype=x.dtype).to(x.device)
        return (x + sigma * n).clamp(0, 1)


class Resize(Distortion):
    """Downsample by ``scale ~ U(scale_min, scale_max)`` (antialiased bilinear) then upsample back.

    Models the resolution loss of capture at distance; the output grid equals the input grid.
    """

    name = "resize"

    def __init__(self, scale_min: float = 0.5, scale_max: float = 1.0) -> None:
        super().__init__()
        if not 0 < scale_min <= scale_max <= 1:
            raise ValueError("need 0 < scale_min <= scale_max <= 1")
        self.scale_min, self.scale_max = scale_min, scale_max

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        scale = uniform(self.scale_min, self.scale_max, gen)
        self.last_params = {"scale": scale}
        if scale >= 1.0:
            return x
        h, w = x.shape[-2:]
        small = (max(1, round(h * scale)), max(1, round(w * scale)))
        y = F.interpolate(x, size=small, mode="bilinear", align_corners=False, antialias=True)
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)


class Crop(Distortion):
    """Keep a random axis-aligned rectangle covering ``area ~ U(area_min, area_max)`` of the
    image; everything outside is replaced by ``fill``. Position is random. The rectangle stays
    where it was (no re-centering), so geometry is preserved and only content is lost —
    the partial-capture / occlusion case. Not part of the Week-1/2 channel presets; reserved for
    the Week-5 synchronisation branch (tiled payload / RaptorQ over partial captures).
    """

    name = "crop"

    def __init__(self, area_min: float = 0.5, area_max: float = 1.0, fill: float = 0.5) -> None:
        super().__init__()
        if not 0 < area_min <= area_max <= 1:
            raise ValueError("need 0 < area_min <= area_max <= 1")
        self.area_min, self.area_max, self.fill = area_min, area_max, fill

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        area = uniform(self.area_min, self.area_max, gen)
        h, w = x.shape[-2:]
        side = math.sqrt(area)
        ch, cw = max(1, round(h * side)), max(1, round(w * side))
        y0 = int(uniform(0, h - ch + 1, gen))
        x0 = int(uniform(0, w - cw + 1, gen))
        y0, x0 = min(y0, h - ch), min(x0, w - cw)
        self.last_params = {"area": area, "y0": y0, "x0": x0, "h": ch, "w": cw}
        if ch == h and cw == w:
            return x
        mask = torch.zeros(1, 1, h, w, device=x.device, dtype=x.dtype)
        mask[..., y0 : y0 + ch, x0 : x0 + cw] = 1.0
        return x * mask + self.fill * (1 - mask)
