"""Illumination: the union of PIMoG's light field and StegaStamp's colour jitter.

Applied in order, each independently switchable via its range (0 = off):

1. multiplicative low-frequency gain field (linear ramp at a random orientation, or radial),
   endpoints ``a ~ U(1 − gain, 1)``, ``b ~ U(1, 1 + gain)`` (PIMoG: a∈[0.7,0.9], b∈[1.1,1.3]);
2. contrast ``× U(1 − contrast, 1 + contrast)`` (StegaStamp [0.5, 1.5]);
3. brightness ``+ U(−bri, bri)`` (StegaStamp 0.3) and per-channel cast ``+ U(−cast, cast)``
   (StegaStamp "hue" 0.1);
4. desaturation towards luminance by ``s ~ U(0, sat)`` (StegaStamp 1.0);
5. gamma ``x ** U(1/(1+g), 1+g)``;
then clamp to ``[0, 1]``. Differentiable.
"""

from __future__ import annotations

import math

import torch

from perceptual_media.core.types import ImageBatch
from perceptual_media.distort.base import Distortion
from perceptual_media.distort.basic import _uniform


def _light_field(h: int, w: int, a: float, b: float, kind: str, angle: float, cx: float, cy: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``(1, 1, h, w)`` multiplicative field ramping from ``a`` to ``b``."""
    ys = torch.linspace(0, 1, h, device=device, dtype=dtype)[:, None].expand(h, w)
    xs = torch.linspace(0, 1, w, device=device, dtype=dtype)[None, :].expand(h, w)
    if kind == "linear":
        t = xs * math.cos(angle) + ys * math.sin(angle)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    else:  # radial: bright at (cx, cy), falling off to the far corner
        r = ((xs - cx) ** 2 + (ys - cy) ** 2).sqrt()
        t = 1 - r / (r.max() + 1e-8)
    return (a + (b - a) * t)[None, None]


class Illumination(Distortion):
    name = "illumination"

    def __init__(
        self,
        gain: float = 0.3,
        contrast: float = 0.5,
        brightness: float = 0.3,
        cast: float = 0.1,
        saturation: float = 1.0,
        gamma: float = 0.3,
    ) -> None:
        super().__init__()
        for k, v in dict(gain=gain, contrast=contrast, brightness=brightness, cast=cast, saturation=saturation, gamma=gamma).items():
            if v < 0:
                raise ValueError(f"{k} must be >= 0")
        self.gain, self.contrast, self.brightness, self.cast, self.saturation, self.gamma = gain, contrast, brightness, cast, saturation, gamma

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        h, w = x.shape[-2:]
        p: dict[str, float | str] = {}

        # 1. light field
        if self.gain > 0:
            a, b = _uniform(1 - self.gain, 1.0, gen), _uniform(1.0, 1 + self.gain, gen)
            kind = "linear" if _uniform(0, 1, gen) < 0.5 else "radial"
            angle, cx, cy = _uniform(0, 2 * math.pi, gen), _uniform(0, 1, gen), _uniform(0, 1, gen)
            x = x * _light_field(h, w, a, b, kind, angle, cx, cy, x.device, x.dtype)
            p.update(gain_a=a, gain_b=b, field=kind, field_angle=angle, field_cx=cx, field_cy=cy)
        # 2. contrast
        if self.contrast > 0:
            c = _uniform(1 - self.contrast, 1 + self.contrast, gen)
            x = x * c
            p["contrast"] = c
        # 3. brightness + cast
        if self.brightness > 0:
            bri = _uniform(-self.brightness, self.brightness, gen)
            x = x + bri
            p["brightness"] = bri
        if self.cast > 0:
            cast = [_uniform(-self.cast, self.cast, gen) for _ in range(3)]
            x = x + torch.tensor(cast, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            p.update(cast_r=cast[0], cast_g=cast[1], cast_b=cast[2])
        # 4. desaturation
        if self.saturation > 0:
            s = _uniform(0, self.saturation, gen)
            lum = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)
            x = (1 - s) * x + s * lum
            p["desaturate"] = s
        # 5. gamma (on the clamped signal; gamma of negatives is undefined)
        x = x.clamp(0, 1)
        if self.gamma > 0:
            g = _uniform(1 / (1 + self.gamma), 1 + self.gamma, gen)
            x = x.clamp_min(1e-6) ** g
            p["gamma"] = g
        self.last_params = p
        return x.clamp(0, 1)
