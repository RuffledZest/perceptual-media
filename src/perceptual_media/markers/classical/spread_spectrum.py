"""Classical baseline: additive spread-spectrum in the luminance mid-band DCT (Cox et al. 1997)
under a Watson per-coefficient budget, with a slack-normalised correlation detector.

Embedding
---------
Every mid-band coefficient of every 8×8 luminance block is a *chip*. A keyed permutation assigns
chips round-robin to ``n_bits + 1`` slots — the payload bits plus one **pilot** slot of known
value — and gives each chip a random sign. Chip ``i`` in slot ``s`` is moved by
``strength × slack_i × sign_i × symbol_s``, ``symbol ∈ {−1, +1}``, where ``slack_i`` is the
Watson JND for that coefficient (``markers.classical.watson``). ``strength = 1`` therefore means
one Watson JND per coefficient — which is *not* invisible once ~150k such perturbations pool
(see the Task-15 worklog note); the strength sweep is what locates the visible/invisible line.

Detection (blind)
-----------------
``term_i = sign_i × C_i / slack_i`` (slack recomputed on the received image), so large,
strongly-masked coefficients are down-weighted like in a whitened matched filter. Per slot,
``z_s = mean(term) × √N_s / std(term)`` — a z-score under the host-interference null. LLRs are
``z`` for the payload slots; the presence ``score`` is ``z`` of the pilot slot (a payload-
independent Neyman–Pearson statistic; its null distribution is measured on the control set).

Capacity (provisional, Task 17 AC)
----------------------------------
Host-interference-limited SNR model: with per-chip term variance ``v`` and amplitude
``strength``, a slot of ``N`` chips has ``z ≈ strength √N / √v``; requiring ``z ≥ Φ⁻¹(1 − p)`` for
raw BER ``p`` gives ``N_needed`` and ``capacity = total_chips / N_needed``. It ignores the
channel entirely and treats ``strength`` as the perceptual limit. Branch B replaces it with a
2AFC-calibrated budget. ``embed`` raises ``CapacityError`` when ``capacity < n_bits`` unless
``force=True``.
"""

from __future__ import annotations

import math

import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.core.types import (
    ImageBatch,
    Payload,
    assert_image_batch,
    assert_payload,
)
from perceptual_media.markers.base import CapacityError, DecodeResult, register_marker
from perceptual_media.markers.classical.dct import (
    band_mask,
    blocks,
    from_luma_dct,
    luma_dct,
    unblocks,
)
from perceptual_media.markers.classical.watson import (
    DEFAULT_VIEWING,
    ViewingCondition,
    watson_slack,
)


class _ChipPlan:
    """Keyed chip → (slot, sign) assignment for one image size."""

    def __init__(self, n_chips: int, n_slots: int, key: int, device: torch.device) -> None:
        gen = make_generator(key)
        perm = torch.randperm(n_chips, generator=gen)
        slot = torch.empty(n_chips, dtype=torch.long)
        slot[perm] = torch.arange(n_chips) % n_slots  # chips in permuted order fill slots round-robin
        sign = (torch.randint(0, 2, (n_chips,), generator=gen) * 2 - 1).to(torch.float32)
        self.slot = slot.to(device)
        self.sign = sign.to(device)
        self.n_slots = n_slots
        self.counts = torch.bincount(self.slot, minlength=n_slots).to(torch.float32)


class ClassicalSSMarker:
    name = "classical_ss"

    def __init__(
        self,
        n_bits: int = 127,
        key: int = 0,
        band: tuple[int, int] = (3, 8),
        viewing: ViewingCondition = DEFAULT_VIEWING,
        capacity_strength: float = 1.0,
        capacity_target_ber: float = 1e-2,
    ) -> None:
        self.n_bits = n_bits
        self.key = key
        self.band = band_mask(*band)
        self.viewing = viewing
        self.capacity_strength = capacity_strength
        self.capacity_target_ber = capacity_target_ber
        self._plans: dict[tuple[int, int, str], _ChipPlan] = {}

    # ---- internals ---------------------------------------------------------------------------

    def _plan(self, n_chips: int, device: torch.device) -> _ChipPlan:
        k = (n_chips, self.n_bits + 1, str(device))
        if k not in self._plans:
            self._plans[k] = _ChipPlan(n_chips, self.n_bits + 1, self.key, device)
        return self._plans[k]

    def _chips(self, img: ImageBatch) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """→ ``(coef (B, N), slack (B, N), state)`` for the mid-band chips of ``img``."""
        y, cb, cr, pad = luma_dct(img)
        mask = self.band.to(y.device)
        bl = blocks(y)
        slack = blocks(watson_slack(y, self.viewing))
        return bl[..., mask], slack[..., mask], (bl, cb, cr, pad, mask)

    def _symbols(self, payload: Payload) -> torch.Tensor:
        """``(B, n_bits + 1)`` symbols in {−1, +1}; the last slot is the pilot (+1)."""
        b = payload.shape[0]
        return torch.cat([payload * 2 - 1, torch.ones(b, 1, device=payload.device)], dim=1)

    # ---- Marker protocol ---------------------------------------------------------------------

    def embed(self, img: ImageBatch, payload: Payload, strength: float = 1.0, *, force: bool = False) -> ImageBatch:
        assert_image_batch(img)
        assert_payload(payload)
        if payload.shape != (img.shape[0], self.n_bits):
            raise ValueError(f"payload must be ({img.shape[0]}, {self.n_bits}), got {tuple(payload.shape)}")
        coef, slack, (bl, cb, cr, pad, mask) = self._chips(img)
        b, nh, nw, m = coef.shape
        n = nh * nw * m
        plan = self._plan(n, img.device)
        if not force:
            cap = self._capacity_from_chips(coef.reshape(b, n), slack.reshape(b, n), plan)
            if (cap < self.n_bits).any():
                raise CapacityError(f"estimated capacity {cap.tolist()} bits < {self.n_bits} for some items; pass force=True to embed anyway")
        sym = self._symbols(payload.to(img.device))  # (B, n_slots)
        chip_sym = sym[:, plan.slot]  # (B, N)
        delta = strength * slack.reshape(b, n) * plan.sign * chip_sym
        new = bl.clone()
        new[..., mask] = (coef.reshape(b, n) + delta).reshape(b, nh, nw, m)
        return from_luma_dct(unblocks(new), cb, cr, pad)

    def decode(self, img: ImageBatch) -> DecodeResult:
        assert_image_batch(img)
        coef, slack, _ = self._chips(img)
        b = coef.shape[0]
        n = coef[0].numel()
        plan = self._plan(n, img.device)
        term = plan.sign * (coef.reshape(b, n) / slack.reshape(b, n))
        sums = torch.zeros(b, plan.n_slots, device=img.device).index_add_(1, plan.slot, term)
        mean = sums / plan.counts
        std = term.std(dim=1, keepdim=True).clamp_min(1e-8)
        z = mean * plan.counts.sqrt() / std
        return DecodeResult(llrs=z[:, : self.n_bits], score=z[:, self.n_bits])

    def capacity(self, img: ImageBatch) -> torch.Tensor | None:
        assert_image_batch(img)
        coef, slack, _ = self._chips(img)
        b = coef.shape[0]
        n = coef[0].numel()
        return self._capacity_from_chips(coef.reshape(b, n), slack.reshape(b, n), self._plan(n, img.device))

    def _capacity_from_chips(self, coef: torch.Tensor, slack: torch.Tensor, plan: _ChipPlan) -> torch.Tensor:
        """``(B,)`` estimated bits at ``capacity_strength`` and raw BER ``capacity_target_ber``."""
        term = plan.sign * (coef / slack)
        v = term.var(dim=1).clamp_min(1e-12)
        z_needed = math.sqrt(2.0) * torch.erfinv(torch.tensor(1.0 - 2.0 * self.capacity_target_ber)).item()
        n_needed = (z_needed**2) * v / (self.capacity_strength**2)
        return torch.floor(coef.shape[1] / n_needed.clamp_min(1.0))


register_marker("classical_ss", ClassicalSSMarker)
