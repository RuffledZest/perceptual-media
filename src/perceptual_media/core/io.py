"""Image I/O: the only place NumPy/PIL representations are converted to and from tensors.

All tensors follow ``core.types``: ``(B, 3, H, W)`` float32 RGB in ``[0, 1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from perceptual_media.core.types import ImageBatch, assert_image_batch


def from_uint8(arr: np.ndarray) -> ImageBatch:
    """``(H, W, 3)`` or ``(B, H, W, 3)`` uint8 RGB → ``(B, 3, H, W)`` float32 in ``[0, 1]``."""
    if arr.dtype != np.uint8:
        raise TypeError(f"expected uint8 array, got {arr.dtype}")
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) or (B, H, W, 3), got {arr.shape}")
    # np.array(..., order="C") always copies: PIL returns read-only views, which torch rejects.
    t = torch.from_numpy(np.array(arr, order="C")).permute(0, 3, 1, 2)
    return t.to(torch.float32).div_(255.0)


def to_uint8(x: ImageBatch) -> np.ndarray:
    """``(B, 3, H, W)`` float32 in ``[0, 1]`` → ``(B, H, W, 3)`` uint8 RGB (round-to-nearest)."""
    assert_image_batch(x)
    q = x.detach().mul(255.0).round_().clamp_(0, 255).to(torch.uint8)
    return q.permute(0, 2, 3, 1).cpu().numpy()


def load_image(path: str | Path) -> ImageBatch:
    """Load one image file as a ``(1, 3, H, W)`` float32 RGB tensor in ``[0, 1]``."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return from_uint8(arr)


def save_image(x: ImageBatch, path: str | Path) -> None:
    """Save a ``(1, 3, H, W)`` or ``(3, H, W)`` tensor to ``path`` (format from extension).

    Values are quantised to 8-bit by round-to-nearest, so a load/save round-trip changes each
    pixel by at most ``1/255``.
    """
    if x.ndim == 3:
        x = x[None]
    if x.shape[0] != 1:
        raise ValueError(f"save_image expects a single image, got batch of {x.shape[0]}")
    arr = to_uint8(x)[0]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
