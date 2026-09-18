"""Shared signal transforms: 8×8 block DCT and JFIF YCbCr. Batched, differentiable, any device."""

from __future__ import annotations

import math

import torch

from perceptual_media.core.types import ImageBatch


def _dct_matrix(n: int = 8) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)[:, None]
    i = torch.arange(n, dtype=torch.float64)[None, :]
    d = torch.cos((2 * i + 1) * k * math.pi / (2 * n)) * math.sqrt(2.0 / n)
    d[0] /= math.sqrt(2.0)
    return d.to(torch.float32)


_D = _dct_matrix()


def block_dct(x: torch.Tensor) -> torch.Tensor:
    """``(..., H, W)`` → same shape of 8×8-blockwise orthonormal DCT-II coefficients. H, W % 8 == 0."""
    *lead, h, w = x.shape
    d = _D.to(x.device, x.dtype)
    blocks = x.reshape(*lead, h // 8, 8, w // 8, 8).permute(*range(len(lead)), -4, -2, -3, -1)  # (..., H/8, W/8, 8, 8)
    coef = d @ blocks @ d.T
    return coef.permute(*range(len(lead)), -4, -2, -3, -1).reshape(*lead, h, w)


def block_idct(c: torch.Tensor) -> torch.Tensor:
    """Inverse of ``block_dct``."""
    *lead, h, w = c.shape
    d = _D.to(c.device, c.dtype)
    blocks = c.reshape(*lead, h // 8, 8, w // 8, 8).permute(*range(len(lead)), -4, -2, -3, -1)
    x = d.T @ blocks @ d
    return x.permute(*range(len(lead)), -4, -2, -3, -1).reshape(*lead, h, w)


def rgb_to_ycbcr(x: ImageBatch) -> torch.Tensor:
    """JFIF full-range: ``(B, 3, H, W)`` RGB in ``[0, 1]`` → YCbCr in ``[0, 1]`` (chroma centred at 0.5)."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack([y, cb, cr], dim=1)


def ycbcr_to_rgb(x: torch.Tensor) -> ImageBatch:
    """Inverse of ``rgb_to_ycbcr`` (not clamped)."""
    y, cb, cr = x[:, 0], x[:, 1] - 0.5, x[:, 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=1)
