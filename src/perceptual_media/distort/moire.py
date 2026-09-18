"""Screen → camera moiré as a sampling problem (brief §4.4).

``mode="sampling"`` (default): the image is rendered on a display whose pixels have RGB stripe
subpixels and a black matrix (``up × up`` sub-samples per image pixel: columns R, G, B, black);
the camera lens blurs it with a Gaussian PSF of ``psf_sigma`` sensor pitches; the sensor then
point-samples it on a grid with ``capture_scale`` sensor pixels per display pixel, a random phase
and a small rotation. The sensor image is resized back to the input grid. The beat between the
display grid (1 cycle per image pixel) and the sensor grid (``capture_scale`` cycles per image
pixel) aliases to ``expected_beat_frequency(capture_scale)`` cycles per pixel, measurable by FFT.

``mode="pimog"``: PIMoG's additive pattern ``min(rings, grating)`` at ``amp`` — a look-alike,
kept for reference; its frequency content does not depend on capture geometry.

``strength`` in ``[0, 1]`` blends the moiré result with the input so presets can scale it.
Differentiable (grid_sample / interpolate); the display mask is a fixed multiplicative pattern.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from perceptual_media.core.types import ImageBatch
from perceptual_media.distort.base import Distortion
from perceptual_media.distort.basic import _uniform
from perceptual_media.distort.blur import _conv_same, gaussian_kernel

Mode = Literal["sampling", "pimog"]


def expected_beat_frequency(capture_scale: float) -> float:
    """Aliased frequency (cycles / image pixel) of a 1 cycle/px grid sampled at ``capture_scale`` samples/px."""
    s = capture_scale
    k = round(1.0 / s)
    return abs(1.0 - k * s)


def display_render(x: ImageBatch, up: int = 4) -> torch.Tensor:
    """``(B, 3, H·up, W·up)``: nearest-upsample and apply the RGB-stripe + black-matrix mask,
    normalised so that box-averaging a display pixel returns the original value."""
    b, c, h, w = x.shape
    hi = F.interpolate(x, scale_factor=up, mode="nearest")
    mask = torch.zeros(1, 3, up, up, device=x.device, dtype=x.dtype)
    if up >= 4:
        for ch in range(3):
            mask[0, ch, :-1, ch] = 1.0  # stripe column per channel; last row/cols = black matrix
        mask = mask * (up * up / (up - 1))  # each channel lit in (up-1) of up*up sub-samples
    else:  # too coarse for a matrix: plain pixel grid
        mask[:] = 1.0
    tile = mask.repeat(1, 1, h, w)
    return hi * tile


def sensor_sample(hi: torch.Tensor, up: int, capture_scale: float, phase: tuple[float, float], rotation_deg: float, psf_sigma: float) -> torch.Tensor:
    """Point-sample ``hi`` on a rotated, phase-shifted sensor grid; returns ``(B, 3, nh, nw)``."""
    b, _, hh, ww = hi.shape
    pitch = up / capture_scale  # hi-res px per sensor pixel
    if psf_sigma > 0:
        hi = _conv_same(hi, gaussian_kernel(psf_sigma * pitch, max_radius=12))
    nh, nw = max(2, math.ceil(hh / pitch)), max(2, math.ceil(ww / pitch))
    dev, dt = hi.device, hi.dtype
    ys = (torch.arange(nh, device=dev, dtype=dt) + 0.5 + phase[1]) * pitch
    xs = (torch.arange(nw, device=dev, dtype=dt) + 0.5 + phase[0]) * pitch
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    if rotation_deg:
        a = math.radians(rotation_deg)
        cy, cx = hh / 2, ww / 2
        rx = cx + (gx - cx) * math.cos(a) - (gy - cy) * math.sin(a)
        ry = cy + (gx - cx) * math.sin(a) + (gy - cy) * math.cos(a)
        gx, gy = rx, ry
    grid = torch.stack([gx / ww * 2 - 1, gy / hh * 2 - 1], dim=-1)[None].expand(b, nh, nw, 2)
    return F.grid_sample(hi, grid, mode="bilinear", padding_mode="border", align_corners=False)


def pimog_pattern(h: int, w: int, center: tuple[float, float], theta_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """PIMoG's ``(z + 1) / 2`` with ``z = min(rings, grating)``, shape ``(1, 1, h, w)`` in ``[0.5, 1]``."""
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij")
    r = ((xx + 1 - center[0]) ** 2 + (yy + 1 - center[1]) ** 2).sqrt()
    z1 = 0.5 + 0.5 * torch.cos(2 * math.pi * r)
    th = math.radians(theta_deg)
    z2 = 0.5 + 0.5 * torch.cos(math.cos(th) * (xx + 1) + math.sin(th) * (yy + 1))
    return ((torch.minimum(z1, z2) + 1) / 2)[None, None]


class Moire(Distortion):
    name = "moire"

    def __init__(
        self,
        scale_min: float = 0.6,
        scale_max: float = 1.4,
        psf_sigma: float = 0.3,
        max_rotation_deg: float = 3.0,
        up: int = 4,
        strength: float = 1.0,
        mode: Mode = "sampling",
        amp: float = 0.15,
    ) -> None:
        super().__init__()
        if not 0 < scale_min <= scale_max:
            raise ValueError("need 0 < scale_min <= scale_max")
        if not 0 <= strength <= 1:
            raise ValueError("strength must be in [0, 1]")
        self.scale_min, self.scale_max, self.psf_sigma, self.max_rotation_deg = scale_min, scale_max, psf_sigma, max_rotation_deg
        self.up, self.strength, self.mode, self.amp = up, strength, mode, amp

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        if self.strength == 0:
            self.last_params = {"mode": self.mode, "strength": 0.0}
            return x
        h, w = x.shape[-2:]
        if self.mode == "sampling":
            s = _uniform(self.scale_min, self.scale_max, gen)
            phase = (_uniform(0, 1, gen), _uniform(0, 1, gen))
            rot = _uniform(-self.max_rotation_deg, self.max_rotation_deg, gen)
            self.last_params = {"mode": "sampling", "capture_scale": s, "phase_x": phase[0], "phase_y": phase[1], "rotation_deg": rot, "psf_sigma": self.psf_sigma, "strength": self.strength}
            hi = display_render(x, self.up)
            sensor = sensor_sample(hi, self.up, s, phase, rot, self.psf_sigma)
            y = F.interpolate(sensor, size=(h, w), mode="bilinear", align_corners=False)
        else:
            center = (_uniform(0, w, gen), _uniform(0, h, gen))
            theta = _uniform(0, 180, gen)
            self.last_params = {"mode": "pimog", "center_x": center[0], "center_y": center[1], "theta_deg": theta, "amp": self.amp, "strength": self.strength}
            m = pimog_pattern(h, w, center, theta, x.device, x.dtype)
            y = x * (1 - self.amp) + self.amp * m
        y = y.clamp(0, 1)
        return x + self.strength * (y - x) if self.strength < 1 else y
