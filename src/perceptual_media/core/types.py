"""Tensor conventions shared by every module.

Images are ``torch.float32`` tensors of shape ``(B, 3, H, W)``, RGB, values in ``[0, 1]``.
Payloads are ``torch.float32`` tensors of shape ``(B, n_bits)`` with values in ``{0, 1}``.
NumPy / PIL / OpenCV representations exist only at the I/O edges (see ``core.io``).
"""

from __future__ import annotations

import torch

ImageBatch = torch.Tensor
"""``(B, 3, H, W)`` float32 RGB in ``[0, 1]``."""

Payload = torch.Tensor
"""``(B, n_bits)`` float32 with values in ``{0, 1}``."""


class ConventionError(ValueError):
    """Raised when a tensor violates the image/payload convention."""


def assert_image_batch(x: torch.Tensor, *, name: str = "image") -> None:
    """Raise ``ConventionError`` unless ``x`` is a ``(B, 3, H, W)`` float32 tensor in ``[0, 1]``."""
    if not isinstance(x, torch.Tensor):
        raise ConventionError(f"{name}: expected torch.Tensor, got {type(x).__name__}")
    if x.ndim != 4 or x.shape[1] != 3:
        raise ConventionError(f"{name}: expected shape (B, 3, H, W), got {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise ConventionError(f"{name}: expected float32, got {x.dtype}")
    if x.numel() and (x.min() < 0 or x.max() > 1):
        raise ConventionError(
            f"{name}: values must lie in [0, 1], got [{x.min().item():.4g}, {x.max().item():.4g}]"
        )


def assert_payload(p: torch.Tensor, *, name: str = "payload") -> None:
    """Raise ``ConventionError`` unless ``p`` is a ``(B, n_bits)`` float32 tensor of 0/1 values."""
    if not isinstance(p, torch.Tensor):
        raise ConventionError(f"{name}: expected torch.Tensor, got {type(p).__name__}")
    if p.ndim != 2:
        raise ConventionError(f"{name}: expected shape (B, n_bits), got {tuple(p.shape)}")
    if p.dtype != torch.float32:
        raise ConventionError(f"{name}: expected float32, got {p.dtype}")
    if p.numel() and not torch.all((p == 0) | (p == 1)):
        raise ConventionError(f"{name}: values must be exactly 0 or 1")
