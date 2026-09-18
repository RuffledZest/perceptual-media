"""Defocus (Gaussian) and motion (line) blur. Differentiable; kernels built per call from ``gen``.

StegaStamp draws one 7×7 kernel per batch from {none, Gaussian σ∈[1,3], line σ∈[0.25,1]} at
400 px. Kernel extent here follows the sampled parameter rather than a fixed 7×7.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from perceptual_media.core.types import ImageBatch
from perceptual_media.distort.base import Distortion
from perceptual_media.distort.basic import _uniform


def _conv_same(x: ImageBatch, kernel: torch.Tensor) -> ImageBatch:
    """Depthwise 2-D convolution with reflect padding; ``kernel`` is ``(kh, kw)``, sums to 1."""
    kh, kw = kernel.shape
    k = kernel.to(x.device, x.dtype)[None, None].expand(x.shape[1], 1, kh, kw)
    xp = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2), mode="reflect")
    return F.conv2d(xp, k, groups=x.shape[1])


def gaussian_kernel(sigma: float, max_radius: int | None = None) -> torch.Tensor:
    """Normalised 2-D Gaussian of radius ``ceil(3σ)`` (or ``max_radius``)."""
    r = max(1, math.ceil(3 * sigma))
    if max_radius is not None:
        r = min(r, max_radius)
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-(t**2) / (2 * sigma**2))
    k = g[:, None] * g[None, :]
    return k / k.sum()


def motion_kernel(length: float, angle_deg: float) -> torch.Tensor:
    """Normalised line kernel of ``length`` px at ``angle_deg`` (0 = horizontal, CCW).

    Pixels within ``length/2`` along the line get weight ``max(0, 1 − perpendicular distance)``,
    so at angle 0 with odd integer length the kernel is exactly a ``1 × length`` box.
    """
    r = max(1, math.ceil(length / 2))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    along = xx * ux + yy * uy
    perp = (-xx * uy + yy * ux).abs()
    w = (1 - perp).clamp(0, 1) * (along.abs() <= length / 2).to(torch.float32)
    if w.sum() == 0:
        w[r, r] = 1.0
    return w / w.sum()


class DefocusBlur(Distortion):
    """Gaussian blur, ``sigma ~ U(sigma_min, sigma_max)`` px; σ < 0.05 is identity."""

    name = "defocus"

    def __init__(self, sigma_min: float = 0.0, sigma_max: float = 3.0, max_radius: int = 15) -> None:
        super().__init__()
        if not 0 <= sigma_min <= sigma_max:
            raise ValueError("need 0 <= sigma_min <= sigma_max")
        self.sigma_min, self.sigma_max, self.max_radius = sigma_min, sigma_max, max_radius

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        sigma = _uniform(self.sigma_min, self.sigma_max, gen)
        self.last_params = {"sigma": sigma}
        if sigma < 0.05:
            return x
        return _conv_same(x, gaussian_kernel(sigma, self.max_radius)).clamp(0, 1)


class MotionBlur(Distortion):
    """Line blur, ``length ~ U(length_min, length_max)`` px, ``angle ~ U(0, 180)``°; length ≤ 1 is identity."""

    name = "motion"

    def __init__(self, length_min: float = 0.0, length_max: float = 9.0) -> None:
        super().__init__()
        if not 0 <= length_min <= length_max:
            raise ValueError("need 0 <= length_min <= length_max")
        self.length_min, self.length_max = length_min, length_max

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        length = _uniform(self.length_min, self.length_max, gen)
        angle = _uniform(0.0, 180.0, gen)
        self.last_params = {"length": length, "angle_deg": angle}
        if length <= 1.0:
            return x
        return _conv_same(x, motion_kernel(length, angle)).clamp(0, 1)
