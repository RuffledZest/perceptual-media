"""Display sheet for the decode capture round (plan Task 24, screen variant).

One **slide** per (image × marker): a ``resolution`` (1920×1080) PNG with the marked image at
1:1 in the centre, four ArUco fiducials (DICT_4X4_50, ids 0–3 at TL/TR/BR/BL) just outside the
image, and a human-readable slide id. Shown full-screen on the display it is pixel-exact, so
the fiducials give the ingestion a content-independent locator; the corpus-calibration locator
(monitor as fiducial) remains the fallback. **The decoder never sees the fiducials** — they exist
only to measure the channel; the ingestion crops the image before decoding.

``sheet.csv`` records, per slide, exactly what was embedded: marker, strength, the 64-bit message
(hex), the channel bits, the seed, and the marked image's digital fidelity (PSNR/SSIM/LPIPS vs
original) — the same message/inner-code convention as the Week-2/3 sweeps (``trial_seed`` on
``sheet name, image_id, marker, strength``), so the digital and physical results are comparable.
The marked 512² PNG is kept next to the slide (``<id>_marked.png``) for the 2AFC study.

Slides are data (regenerable from this file + seeds) and live under
``paths.captures/sheets/<name>/``, outside the repo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from perceptual_media.core.config import EccConfig, MarkerConfig
from perceptual_media.core.io import from_uint8, to_uint8
from perceptual_media.core.seed import make_generator
from perceptual_media.harness.ecc import build_ecc
from perceptual_media.harness.runner import resolve_device, trial_seed
from perceptual_media.markers.base import get_marker, random_payload
from perceptual_media.metrics.fidelity import fidelity

ARUCO_DICT = cv2.aruco.DICT_4X4_50
BACKGROUND = (40, 40, 44)


@dataclass
class SheetMarker:
    marker: MarkerConfig
    ecc: EccConfig | None = None
    strength: float = 1.0
    images: list[str] | None = None
    """Subset of the sheet's images for this marker (``None`` = all of them)."""


@dataclass
class SheetSpec:
    """Schema for ``configs/capture_sheet.yaml``."""

    name: str
    images: list[str]
    markers: list[SheetMarker]
    seed: int = 0
    resolution: list[int] = field(default_factory=lambda: [1920, 1080])
    image_px: int = 512
    fiducial_px: int = 80
    gap_px: int = 40
    quiet_px: int = 12
    """White border around each fiducial so its black edge is detectable on a dark background."""
    device: str = "auto"


def image_origin(spec: SheetSpec) -> tuple[int, int]:
    return (spec.resolution[0] - spec.image_px) // 2, (spec.resolution[1] - spec.image_px) // 2


def fiducial_boxes(spec: SheetSpec) -> list[tuple[int, int]]:
    """Top-left corner of the four fiducials (ids 0..3 = TL, TR, BR, BL) in slide coordinates."""
    x0, y0 = image_origin(spec)
    f, g, s = spec.fiducial_px, spec.gap_px, spec.image_px
    return [(x0 - g - f, y0 - g - f), (x0 + s + g, y0 - g - f), (x0 + s + g, y0 + s + g), (x0 - g - f, y0 + s + g)]


def render_slide(spec: SheetSpec, marked_u8: np.ndarray, label: str) -> np.ndarray:
    """``(H, W, 3)`` uint8 slide: background, fiducials, the image at 1:1, the label below."""
    w, h = spec.resolution
    slide = np.empty((h, w, 3), np.uint8)
    slide[:] = BACKGROUND
    x0, y0 = image_origin(spec)
    slide[y0 : y0 + spec.image_px, x0 : x0 + spec.image_px] = marked_u8
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    q = spec.quiet_px
    for i, (fx, fy) in enumerate(fiducial_boxes(spec)):
        m = cv2.aruco.generateImageMarker(dictionary, i, spec.fiducial_px)
        slide[fy - q : fy + spec.fiducial_px + q, fx - q : fx + spec.fiducial_px + q] = 255  # quiet zone
        slide[fy : fy + spec.fiducial_px, fx : fx + spec.fiducial_px] = m[..., None]
    im = Image.fromarray(slide)
    draw = ImageDraw.Draw(im)
    draw.text((x0, y0 + spec.image_px + spec.gap_px + spec.fiducial_px + 16), label, fill=(150, 150, 150))
    return np.asarray(im)


def build_sheet(spec: SheetSpec, out_dir: Path, corpus: Path = Path("data/corpus")) -> pd.DataFrame:
    """Embed, render and write every slide plus ``sheet.csv``; returns the manifest."""
    device = resolve_device(spec.device)
    manifest = pd.read_csv(corpus / "manifest.csv").set_index("image_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for sm in spec.markers:
        marker = get_marker(sm.marker.name, sm.marker.n_bits, **sm.marker.params)
        ecc = build_ecc(sm.ecc, marker.n_bits)
        n_message = ecc.k if ecc is not None else marker.n_bits
        for image_id in sm.images if sm.images is not None else spec.images:
            original_u8 = np.asarray(Image.open(corpus / manifest.loc[image_id, "path"]).convert("RGB"))
            x = from_uint8(original_u8).to(device)
            seed = trial_seed(spec.seed, image_id, sm.strength, sm.marker.name, 0.0)
            message = random_payload(1, n_message, make_generator(seed)).to(device)
            payload = ecc.encode(message) if ecc is not None else message
            with torch.no_grad():
                y = marker.embed(x, payload, sm.strength, force=True).clamp(0, 1)
                fid = fidelity(y, x)
            marked_u8 = to_uint8(y)[0]
            slide_id = f"S{idx:02d}"
            label = f"{slide_id}  {image_id}  {sm.marker.name}  s{sm.strength:g}"
            Image.fromarray(render_slide(spec, marked_u8, label)).save(out_dir / f"{slide_id}.png")
            Image.fromarray(marked_u8).save(out_dir / f"{slide_id}_marked.png")
            bits = "".join(str(int(b)) for b in message[0].tolist())
            rows.append(
                {
                    "slide_id": slide_id,
                    "image_id": image_id,
                    "image_class": manifest.loc[image_id, "image_class"],
                    "marker": sm.marker.name,
                    "n_bits": marker.n_bits,
                    "marker_params": json.dumps(sm.marker.params),
                    "ecc": json.dumps(sm.ecc.__dict__) if sm.ecc is not None else "",
                    "strength": sm.strength,
                    "seed": seed,
                    "message_bits": bits,
                    "message_hex": f"{int(bits, 2):0{(n_message + 3) // 4}x}",
                    "psnr": float(fid["psnr"]),
                    "ssim": float(fid["ssim"]),
                    "lpips": float(fid["lpips"]),
                    "original": str(manifest.loc[image_id, "path"]),
                    "file": f"{slide_id}.png",
                }
            )
            idx += 1
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sheet.csv", index=False)
    with open(out_dir / "spec.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "name": spec.name,
                "resolution": spec.resolution,
                "image_px": spec.image_px,
                "fiducial_px": spec.fiducial_px,
                "gap_px": spec.gap_px,
                "quiet_px": spec.quiet_px,
                "image_origin": image_origin(spec),
                "fiducial_boxes": fiducial_boxes(spec),
                "aruco": "DICT_4X4_50",
            },
            fh,
            indent=1,
        )
    return df
