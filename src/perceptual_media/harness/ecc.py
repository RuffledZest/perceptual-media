"""Harness-side channel coding: fits a ``k``-bit message into a marker's ``n_bits`` channel.

Markers are ECC-agnostic (they see raw channel bits, as StegaStamp does). ``build_ecc`` returns a
``ChannelCode`` that maps ``message (B, k)`` → ``channel bits (B, marker_bits)`` and back from
``channel LLRs (B, marker_bits)``:

* the inner code (BCH(n, k)) produces ``n`` bits;
* the codeword is repeated ``repeat`` times and zero-padded to ``marker_bits``
  (``n × repeat ≤ marker_bits``); on decode the LLRs of the copies are **summed** before the
  inner decoder — soft repetition combining, ~``10·log10(repeat)`` dB of gain — and the pad bits
  are ignored.

This is what makes a 127-bit classical channel and a 256-bit learned channel comparable on the
same 64-bit message and the same inner code.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from perceptual_media.core.config import EccConfig
from perceptual_media.core.types import Payload


class InnerCode(Protocol):
    n: int
    k: int

    def encode(self, message: Payload) -> Payload: ...
    def decode_soft(self, llrs: torch.Tensor) -> tuple[Payload, torch.Tensor]: ...


class ChannelCode:
    """Inner code + repetition + zero padding to ``marker_bits``."""

    def __init__(self, inner: InnerCode, marker_bits: int, repeat: int = 1) -> None:
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        if inner.n * repeat > marker_bits:
            raise ValueError(f"ecc n={inner.n} x repeat={repeat} = {inner.n * repeat} exceeds marker.n_bits={marker_bits}")
        self.inner, self.marker_bits, self.repeat = inner, marker_bits, repeat
        self.n, self.k = inner.n, inner.k

    def encode(self, message: Payload) -> Payload:
        """``(B, k)`` → ``(B, marker_bits)``."""
        code = self.inner.encode(message)
        b = code.shape[0]
        out = torch.zeros(b, self.marker_bits, dtype=code.dtype, device=code.device)
        out[:, : self.n * self.repeat] = code.repeat(1, self.repeat)
        return out

    def decode_soft(self, llrs: torch.Tensor) -> tuple[Payload, torch.Tensor]:
        """``(B, marker_bits)`` LLRs → ``(message (B, k), ok (B,))``; copies are soft-combined."""
        if llrs.shape[1] != self.marker_bits:
            raise ValueError(f"expected (B, {self.marker_bits}) LLRs, got {tuple(llrs.shape)}")
        used = llrs[:, : self.n * self.repeat].reshape(llrs.shape[0], self.repeat, self.n).sum(dim=1)
        return self.inner.decode_soft(used)


def build_ecc(cfg: EccConfig | None, marker_bits: int) -> ChannelCode | None:
    """Instantiate the configured ECC (or ``None``) sized to the marker's channel."""
    if cfg is None:
        return None
    if cfg.name != "bch":
        raise KeyError(f"unknown ecc {cfg.name!r}; available: ['bch']")
    from perceptual_media.markers.classical.bch import BCHCode  # numba import, deferred

    inner: Any = BCHCode(cfg.n, cfg.k)
    return ChannelCode(inner, marker_bits, cfg.repeat)
