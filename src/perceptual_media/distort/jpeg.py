"""JPEG with 4:2:0 chroma subsampling: a real (PIL) path for evaluation and a differentiable
approximation for training.

The differentiable path follows the real codec step for step — JFIF YCbCr, 2×2 chroma average,
8×8 orthonormal DCT, quantisation by the Annex-K tables scaled with libjpeg's quality rule —
and differs only in the rounding surrogate:

* ``"ste"``: forward is exact ``round``, backward is identity (straight-through). Output equals
  a true quantise/dequantise; recommended when fidelity to the real path matters.
* ``"round_only_at_0"``: StegaStamp's default — ``x³`` for ``|x| < 0.5`` (pushes would-be-zero
  coefficients to zero), identity elsewhere. Non-zero coefficients are *not* quantised.
* ``"diff_round"``: ``round(x) + (x − round(x))³`` (Shin & Song).

Residual differences from PIL come from PIL's fancy chroma upsampling, DCT precision and
the final uint8 rounding; see ``tests/distort/test_jpeg.py`` for the measured PSNR.
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from perceptual_media.core.io import from_uint8, to_uint8
from perceptual_media.core.transforms import block_dct, block_idct, rgb_to_ycbcr, ycbcr_to_rgb
from perceptual_media.core.types import ImageBatch
from perceptual_media.distort.base import Distortion
from perceptual_media.distort.basic import _uniform

Rounding = Literal["ste", "round_only_at_0", "diff_round"]

# ITU-T T.81 Annex K, Tables K.1 (luma) and K.2 (chroma), natural order.
_Q_LUMA = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)
_Q_CHROMA = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def quant_tables(quality: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Luma/chroma tables for ``quality`` in ``[1, 100]``, libjpeg's scaling rule."""
    q = float(min(max(quality, 1.0), 100.0))
    s = 5000.0 / q if q < 50 else 200.0 - 2.0 * q

    def scale(t: torch.Tensor) -> torch.Tensor:
        return torch.floor((t * s + 50.0) / 100.0).clamp(1.0, 255.0)

    return scale(_Q_LUMA), scale(_Q_CHROMA)


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.round(x)

    @staticmethod
    def backward(ctx, g: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return g


def soft_round(x: torch.Tensor, mode: Rounding) -> torch.Tensor:
    if mode == "ste":
        return _RoundSTE.apply(x)
    if mode == "round_only_at_0":
        small = (x.abs() < 0.5).to(x.dtype)
        return small * x**3 + (1 - small) * x
    if mode == "diff_round":
        r = torch.round(x)
        return r + (x - r) ** 3
    raise ValueError(f"unknown rounding {mode!r}")


def jpeg_differentiable(x: ImageBatch, quality: float, rounding: Rounding = "ste", subsample: bool = True) -> ImageBatch:
    """Differentiable JPEG round-trip of ``x`` at ``quality``; output clamped to ``[0, 1]``."""
    b, _, h, w = x.shape
    ph, pw = (-h) % 16, (-w) % 16  # pad to a multiple of 16 so 4:2:0 blocks tile
    xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
    ycc = rgb_to_ycbcr(xp) * 255.0 - 128.0
    ql, qc = (t.to(x.device, x.dtype) for t in quant_tables(quality))

    def code(chan: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        c = block_dct(chan)
        hh, ww = c.shape[-2:]
        qt = q.repeat(hh // 8, ww // 8)
        return block_idct(soft_round(c / qt, rounding) * qt)

    y = code(ycc[:, 0], ql)
    if subsample:
        cb = F.avg_pool2d(ycc[:, 1:2], 2)[:, 0]
        cr = F.avg_pool2d(ycc[:, 2:3], 2)[:, 0]
        cb = F.interpolate(code(cb, qc)[:, None], scale_factor=2, mode="bilinear", align_corners=False)[:, 0]
        cr = F.interpolate(code(cr, qc)[:, None], scale_factor=2, mode="bilinear", align_corners=False)[:, 0]
    else:
        cb, cr = code(ycc[:, 1], qc), code(ycc[:, 2], qc)
    out = ycbcr_to_rgb((torch.stack([y, cb, cr], dim=1) + 128.0) / 255.0)
    return out[..., :h, :w].clamp(0, 1)


def jpeg_pil(x: ImageBatch, quality: int, subsample: bool = True) -> ImageBatch:
    """Real JPEG via PIL (not differentiable): encode/decode each item at integer ``quality``."""
    outs = []
    for arr in to_uint8(x):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality), subsampling="4:2:0" if subsample else "4:4:4")
        buf.seek(0)
        with Image.open(buf) as im:
            outs.append(np.asarray(im.convert("RGB")))
    return from_uint8(np.stack(outs)).to(x.device)


class JPEG(Distortion):
    """JPEG at ``quality ~ U(q_min, q_max)`` with 4:2:0 subsampling.

    ``differentiable=False`` uses PIL (exact, no gradient); ``True`` uses the surrogate.
    StegaStamp trains down to Q = 25. A sampled quality of 100 is a passthrough (no codec),
    so a chain at severity 0 is the exact identity.
    """

    name = "jpeg"

    def __init__(self, q_min: float = 25, q_max: float = 100, differentiable: bool = False, rounding: Rounding = "ste") -> None:
        super().__init__()
        if not 1 <= q_min <= q_max <= 100:
            raise ValueError("need 1 <= q_min <= q_max <= 100")
        self.q_min, self.q_max, self.differentiable, self.rounding = q_min, q_max, differentiable, rounding

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        quality = _uniform(self.q_min, self.q_max, gen)
        self.last_params = {"quality": quality, "differentiable": self.differentiable}
        if quality >= 100:
            return x
        if self.differentiable:
            return jpeg_differentiable(x, quality, self.rounding)
        return jpeg_pil(x, int(round(quality)))
