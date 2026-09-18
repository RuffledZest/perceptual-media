"""Luminance block-DCT view of an image, for the classical marker and the Watson model.

Conventions
-----------
* Luminance is JFIF ``Y`` scaled to ``[0, 255]`` **without** the −128 level shift, so the DC
  coefficient of an 8×8 block is ``8 × mean`` in ``[0, 2040]`` — the scale Watson's luminance
  masking term is defined on (Task 15).
* ``block_dct`` is orthonormal (``core.transforms``), matching ``scipy.fft.dctn(norm="ortho")``.
* Images whose sides are not multiples of 8 are replicate-padded for the transform and cropped
  on the way back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from perceptual_media.core.transforms import (
    block_dct,
    block_idct,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)
from perceptual_media.core.types import ImageBatch, assert_image_batch


def pad_to_8(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Replicate-pad ``(..., H, W)`` so H, W are multiples of 8; returns ``(padded, (ph, pw))``."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        lead = x.shape[:-2]
        x = F.pad(x.reshape(-1, 1, h, w), (0, pw, 0, ph), mode="replicate").reshape(*lead, h + ph, w + pw)
    return x, (ph, pw)


def luma_dct(img: ImageBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    """``(B,3,H,W)`` RGB → ``(Y_coef, Cb, Cr, (ph, pw))``.

    ``Y_coef``: ``(B, H8, W8)`` block-DCT coefficients of luminance in ``[0, 255]`` scale;
    ``Cb``/``Cr``: ``(B, H8, W8)`` chroma planes in ``[0, 1]`` (untouched, passed through);
    ``(ph, pw)``: padding added, for ``from_luma_dct``.
    """
    assert_image_batch(img)
    ycc = rgb_to_ycbcr(img)
    y, (ph, pw) = pad_to_8(ycc[:, 0] * 255.0)
    cb, _ = pad_to_8(ycc[:, 1])
    cr, _ = pad_to_8(ycc[:, 2])
    return block_dct(y), cb, cr, (ph, pw)


def from_luma_dct(y_coef: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor, pad: tuple[int, int]) -> ImageBatch:
    """Inverse of ``luma_dct``; output clamped to ``[0, 1]``."""
    y = block_idct(y_coef) / 255.0
    ph, pw = pad
    h, w = y.shape[-2] - ph, y.shape[-1] - pw
    ycc = torch.stack([y[..., :h, :w], cb[..., :h, :w], cr[..., :h, :w]], dim=1)
    return ycbcr_to_rgb(ycc).clamp(0, 1)


def blocks(coef: torch.Tensor) -> torch.Tensor:
    """``(B, H, W)`` coefficient plane → ``(B, H/8, W/8, 8, 8)`` view (``[..., u, v]`` = row/col frequency)."""
    b, h, w = coef.shape
    return coef.reshape(b, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)


def unblocks(bl: torch.Tensor) -> torch.Tensor:
    """Inverse of ``blocks``."""
    b, nh, nw, _, _ = bl.shape
    return bl.permute(0, 1, 3, 2, 4).reshape(b, nh * 8, nw * 8)


def band_mask(min_sum: int = 3, max_sum: int = 8) -> torch.Tensor:
    """``(8, 8)`` bool: coefficients with ``min_sum <= u + v <= max_sum``.

    Default ``[3, 8]`` is the mid band — above the low frequencies the eye tracks and
    JPEG/blur leave alone, below the high frequencies that compression and defocus destroy
    (brief §4.3). 37 of 64 coefficients.
    """
    u = torch.arange(8)[:, None]
    v = torch.arange(8)[None, :]
    s = u + v
    return (s >= min_sum) & (s <= max_sum)


MID_BAND = band_mask()
