"""Synthetic hard-case images (brief §5 W1, §10.3), generated from a seed.

These are the images the literature leaves out and our use case is made of: solid fills,
gradients, text, and procedural textures. Every generator is deterministic given ``gen`` so the
corpus is reproducible byte-for-byte. Output is ``(1, 3, H, W)`` float32 in ``[0, 1]``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from perceptual_media.core.io import from_uint8
from perceptual_media.core.types import ImageBatch


@dataclass
class SynthImage:
    image_id: str
    image_class: str
    source: str  # human-readable recipe
    image: ImageBatch


# ---------------------------------------------------------------------------
# flat
# ---------------------------------------------------------------------------

_FLAT_FILLS: list[tuple[str, tuple[float, float, float]]] = [
    ("near_black", (0.05, 0.05, 0.05)),
    ("dark_grey", (0.25, 0.25, 0.25)),
    ("mid_grey", (0.50, 0.50, 0.50)),
    ("light_grey", (0.75, 0.75, 0.75)),
    ("near_white", (0.95, 0.95, 0.95)),
    ("brand_red", (0.85, 0.10, 0.15)),
    ("brand_blue", (0.00, 0.35, 0.70)),
    ("brand_yellow", (1.00, 0.85, 0.10)),
    ("brand_green", (0.10, 0.60, 0.30)),
]


def flat_images(size: int) -> Iterator[SynthImage]:
    """Solid fills spanning luminance plus brand-like colours (the zero-masking worst case)."""
    for i, (name, rgb) in enumerate(_FLAT_FILLS):
        img = torch.tensor(rgb, dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, size, size).clone()
        yield SynthImage(f"flat_{i:02d}_{name}", "flat", f"flat:{name}", img)


# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------


def gradient_images(size: int) -> Iterator[SynthImage]:
    """Linear (h/v/diagonal) and radial gradients; smooth regions with no texture masking."""
    t = torch.linspace(0, 1, size)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    recipes: list[tuple[str, torch.Tensor]] = [
        ("linear_h_grey", xx.expand(3, size, size)),
        ("linear_v_grey", yy.expand(3, size, size)),
        ("diag_blue_orange", torch.stack([0.9 * (xx + yy) / 2 + 0.05, 0.5 * (xx + yy) / 2 + 0.3, 0.9 - 0.8 * (xx + yy) / 2])),
        ("radial_vignette", (1 - ((xx - 0.5) ** 2 + (yy - 0.5) ** 2).sqrt() * 1.2).clamp(0, 1).expand(3, size, size)),
        ("radial_colour", torch.stack([((xx - 0.3) ** 2 + (yy - 0.4) ** 2).sqrt().clamp(0, 1), 0.4 * torch.ones_like(xx), 1 - ((xx - 0.3) ** 2 + (yy - 0.4) ** 2).sqrt().clamp(0, 1)])),
    ]
    for i, (name, img) in enumerate(recipes):
        yield SynthImage(f"gradient_{i:02d}_{name}", "gradient", f"gradient:{name}", img.clamp(0, 1)[None].contiguous())


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------

_LOREM = (
    "Perceptually lossless machine-readable media. Transform a still image so that a human "
    "cannot distinguish it from the original, while a commodity smartphone camera photographing "
    "it from a print or a screen, at an angle, under uncontrolled lighting, recovers a payload. "
    "This is a communication system, not a provenance system. The signal energy that survives "
    "camera capture at distance is the same energy the eye can detect in smooth regions. "
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if draw.textlength(cand, font=font) <= width:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _render(size: int, bg: tuple[int, int, int], fg: tuple[int, int, int], layout: str) -> ImageBatch:
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    s = size / 512  # scale factor relative to the reference 512 px design
    if layout == "document":
        font = _font(max(6, int(14 * s)))
        margin, y, lh = int(40 * s), int(40 * s), int(20 * s)
        for line in _wrap(d, _LOREM * 3, font, size - 2 * margin):
            if y > size - margin:
                break
            d.text((margin, y), line, fill=fg, font=font)
            y += lh
    elif layout == "two_column":
        font = _font(max(6, int(12 * s)))
        margin, gap, lh = int(32 * s), int(24 * s), int(17 * s)
        colw = (size - 2 * margin - gap) // 2
        for c in range(2):
            x, y = margin + c * (colw + gap), margin
            for line in _wrap(d, _LOREM * 2, font, colw):
                if y > size - margin:
                    break
                d.text((x, y), line, fill=fg, font=font)
                y += lh
    elif layout == "poster":
        big, small = _font(max(10, int(64 * s))), _font(max(6, int(18 * s)))
        d.text((int(40 * s), int(60 * s)), "OPEN", fill=fg, font=big)
        d.text((int(40 * s), int(140 * s)), "STUDIO", fill=fg, font=big)
        d.text((int(40 * s), int(240 * s)), "Sat 21 Sep — 10:00–18:00", fill=fg, font=small)
        y = int(300 * s)
        for line in _wrap(d, _LOREM, small, size - int(80 * s))[:6]:
            d.text((int(40 * s), y), line, fill=fg, font=small)
            y += int(26 * s)
    elif layout == "headline":
        big = _font(max(12, int(96 * s)))
        d.text((int(32 * s), int(160 * s)), "SALE", fill=fg, font=big)
        d.text((int(32 * s), int(280 * s)), "-50%", fill=fg, font=big)
    else:
        raise ValueError(layout)
    return from_uint8(np.asarray(im))


def text_images(size: int) -> Iterator[SynthImage]:
    """Document- and poster-like layouts: sharp edges, large flat backgrounds, brand colours."""
    recipes = [
        ("document_black_on_white", (255, 255, 255), (20, 20, 20), "document"),
        ("two_column_document", (250, 250, 245), (30, 30, 40), "two_column"),
        ("poster_white_on_navy", (10, 35, 90), (245, 245, 245), "poster"),
        ("headline_yellow_on_red", (200, 25, 35), (255, 215, 30), "headline"),
        ("poster_dark_on_yellow", (255, 215, 30), (25, 25, 25), "poster"),
    ]
    for i, (name, bg, fg, layout) in enumerate(recipes):
        yield SynthImage(f"text_{i:02d}_{name}", "text", f"text:{layout}", _render(size, bg, fg, layout))


# ---------------------------------------------------------------------------
# textured (procedural)
# ---------------------------------------------------------------------------


def _value_noise(size: int, gen: torch.Generator, octaves: int = 5, base: int = 4) -> torch.Tensor:
    """Multi-octave smooth value noise (Perlin-like) in ``[0, 1]``, shape ``(size, size)``."""
    out = torch.zeros(size, size)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * (2**o)
        grid = torch.rand(1, 1, n, n, generator=gen)
        up = torch.nn.functional.interpolate(grid, size=(size, size), mode="bicubic", align_corners=False)[0, 0]
        out += amp * up
        total += amp
        amp *= 0.5
    out /= total
    return ((out - out.min()) / (out.max() - out.min() + 1e-8)).clamp(0, 1)


def textured_images(size: int, gen: torch.Generator) -> Iterator[SynthImage]:
    """Procedural textures: the masking-rich easy cases every benchmark is made of."""
    t = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(t, t, indexing="ij")

    def grating(freq_cyc_per_img: float, angle_deg: float) -> torch.Tensor:
        a = math.radians(angle_deg)
        phase = 2 * math.pi * freq_cyc_per_img * (xx * math.cos(a) + yy * math.sin(a)) / size
        return 0.5 + 0.5 * torch.cos(phase)

    vn = _value_noise(size, gen)
    recipes: list[tuple[str, torch.Tensor]] = [
        ("value_noise_grey", vn.expand(3, size, size)),
        ("value_noise_colour", torch.stack([_value_noise(size, gen), _value_noise(size, gen, base=8), _value_noise(size, gen, base=2)])),
        ("stripes_8cyc", grating(8, 0).expand(3, size, size)),
        ("stripes_64cyc_diag", grating(64, 30).expand(3, size, size)),
        ("checker_32", (((xx // (size / 32)) + (yy // (size / 32))) % 2).expand(3, size, size)),
        ("mixed_freq", (0.5 * grating(6, 0) + 0.3 * grating(24, 45) + 0.2 * grating(96, 110)).expand(3, size, size)),
        ("white_noise", torch.rand(3, size, size, generator=gen)),
        ("noise_on_gradient", (0.7 * (xx / size).expand(3, size, size) + 0.3 * torch.rand(3, size, size, generator=gen))),
    ]
    for i, (name, img) in enumerate(recipes):
        yield SynthImage(f"textured_{i:02d}_{name}", "textured", f"textured:{name}", img.clamp(0, 1)[None].contiguous())


def all_synthetic(size: int, seed: int) -> Iterator[SynthImage]:
    """Every synthetic class, in a fixed order, deterministic in ``seed``."""
    gen = torch.Generator().manual_seed(seed)
    yield from flat_images(size)
    yield from gradient_images(size)
    yield from text_images(size)
    yield from textured_images(size, gen)
