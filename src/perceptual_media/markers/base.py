"""The ``Marker`` protocol: the one interface every embedding scheme implements.

Null, classical spread-spectrum, Video Seal adapter, our own — all of them look the same to
the harness:

* ``embed(img, payload, strength)`` returns a marked image of the same shape.
* ``decode(img)`` returns **soft** per-bit log-likelihood ratios and a scalar detector score.
  Never hard bits: slicing before the ECC throws away several dB (brief §4.5).
* ``capacity(img)`` estimates how many bits the image can carry *invisibly*. ``None`` means the
  scheme has no estimate. This is the refusal hook — it exists from day one even when stubbed,
  because per-image capacity with graceful refusal is the project's core contribution (brief §2).

Tensor conventions are those of ``core.types``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.core.types import (
    ImageBatch,
    Payload,
    assert_image_batch,
    assert_payload,
)


class CapacityError(RuntimeError):
    """Raised by ``embed`` when the image cannot carry the payload invisibly and ``force`` is off."""


@dataclass
class DecodeResult:
    """Soft decoder output.

    ``llrs``: ``(B, n_bits)`` float32; ``llrs > 0`` means bit 1 is more likely, magnitude is
    confidence. ``score``: ``(B,)`` float32 presence-detection statistic (higher = more likely
    marked); its scale is scheme-specific and must be calibrated against an unmarked control set.
    """

    llrs: torch.Tensor
    score: torch.Tensor

    def __post_init__(self) -> None:
        if self.llrs.ndim != 2:
            raise ValueError(f"llrs must be (B, n_bits), got {tuple(self.llrs.shape)}")
        if self.score.shape != (self.llrs.shape[0],):
            raise ValueError(
                f"score must be (B,)={self.llrs.shape[0]}, got {tuple(self.score.shape)}"
            )

    @property
    def n_bits(self) -> int:
        return int(self.llrs.shape[1])

    def hard_bits(self) -> Payload:
        """``(B, n_bits)`` float32 in ``{0, 1}``: 1 where ``llr > 0``. Use only *after* ECC or for BER."""
        return (self.llrs > 0).to(torch.float32)


@runtime_checkable
class Marker(Protocol):
    """See module docstring."""

    n_bits: int

    def embed(self, img: ImageBatch, payload: Payload, strength: float = 1.0, *, force: bool = False) -> ImageBatch:
        """Return a marked copy of ``img`` carrying ``payload`` at ``strength`` (1.0 = nominal).

        Raises ``CapacityError`` if ``capacity(img) < n_bits`` for some item, unless ``force``.
        """
        ...

    def decode(self, img: ImageBatch) -> DecodeResult:
        """Soft-decode a (possibly distorted, possibly unmarked) image."""
        ...

    def capacity(self, img: ImageBatch) -> torch.Tensor | None:
        """``(B,)`` estimated invisible capacity in bits, or ``None`` if the scheme has no estimate."""
        ...


def random_payload(batch: int, n_bits: int, gen: torch.Generator) -> Payload:
    """``(batch, n_bits)`` float32 Bernoulli(0.5) bits drawn from ``gen``."""
    return torch.randint(0, 2, (batch, n_bits), generator=gen, device=gen.device).to(torch.float32)


class NullMarker:
    """Does nothing on embed and guesses on decode.

    Purpose: (1) the unmarked-control path and end-to-end harness testing before any real
    marker exists; (2) a reference line at BER = 0.5 / AUC = 0.5 on every chart.
    ``decode`` is deterministic given ``seed`` and the call count, so a run is reproducible.
    """

    def __init__(self, n_bits: int = 64, seed: int = 0) -> None:
        self.n_bits = n_bits
        self._gen = make_generator(seed)

    def embed(self, img: ImageBatch, payload: Payload, strength: float = 1.0, *, force: bool = False) -> ImageBatch:
        assert_image_batch(img)
        assert_payload(payload)
        if payload.shape != (img.shape[0], self.n_bits):
            raise ValueError(f"payload must be ({img.shape[0]}, {self.n_bits}), got {tuple(payload.shape)}")
        return img.clone()

    def decode(self, img: ImageBatch) -> DecodeResult:
        assert_image_batch(img)
        b = img.shape[0]
        llrs = torch.randn(b, self.n_bits, generator=self._gen).to(img.device)
        score = torch.randn(b, generator=self._gen).to(img.device)
        return DecodeResult(llrs=llrs, score=score)

    def capacity(self, img: ImageBatch) -> torch.Tensor | None:
        return None


_REGISTRY: dict[str, type] = {"null": NullMarker}


def register_marker(name: str, cls: type) -> None:
    """Register a ``Marker`` implementation under ``name`` for ``get_marker``."""
    _REGISTRY[name] = cls


def get_marker(name: str, n_bits: int, **params: Any) -> Marker:
    """Instantiate a registered marker by name (as used in ``MarkerConfig``)."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown marker {name!r}; registered: {sorted(_REGISTRY)}") from None
    return cls(n_bits=n_bits, **params)
