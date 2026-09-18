"""Watson's DCT perceptual model → per-coefficient *slack*: how much each 8×8 DCT coefficient
of luminance can change before the change is (predicted to be) visible. This is the classical
budget map (brief §4.1) and the reference the learned budget in Branch B will be measured against.

Three stages, exactly as in A. B. Watson, "DCT quantization matrices visually optimized for
individual images", Proc. SPIE 1913 (1993):

1. **Frequency sensitivity** ``t[u, v]`` — the threshold coefficient amplitude (in 0–255 pixel
   units, orthonormal 8×8 DCT, DC = 8 × mean) at which a lone basis function becomes visible.
   We use the published table (Cox, Miller & Bloom, *Digital Watermarking*, Table 7.2, computed
   by Peterson's method) as ground truth at Watson's reference condition — **32 pixels/degree,
   mean display luminance 65 cd/m²** — and rescale to other viewing conditions with the
   Ahumada–Peterson log-parabola fitted to that table (``_FIT``; max error ±20 %, rms 11 % at the
   reference, applied only as a *ratio* so the reference itself is exact).
2. **Luminance masking** ``t_L = t · (C₀₀ / C̄₀₀)^a_T``, ``a_T = 0.649``, ``C̄₀₀ = 1024`` (8-bit
   mid grey): brighter blocks tolerate more.
3. **Contrast masking** ``s = max(t_L, |C|^w · t_L^(1−w))``, ``w = 0.7``, ``w₀₀ = 0`` (the DC
   term is already handled by luminance masking): busy blocks tolerate more.

**Pinned viewing condition** (plan.md resolved Q4): ``SCREEN_96DPI_60CM`` ≈ 39.6 px/deg — a
laptop screen at arm's length, the condition the Week-4 2AFC study will be run under.
``PRINT_300DPI_40CM`` ≈ 82.5 px/deg is provided for the print channel.

Caveats Watson himself states: the power-law luminance masking is inaccurate below ~10 cd/m²
(very dark blocks) and least valid at DC; the model ignores masking *between* frequencies and
across blocks. It is a threshold model for JPEG-style errors, not a full visibility metric — the
2AFC study is the ground truth, this is the prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from perceptual_media.markers.classical.dct import blocks, unblocks

# --- viewing conditions ---------------------------------------------------------------------


@dataclass(frozen=True)
class ViewingCondition:
    """Display resolution and distance → pixels per degree of visual angle."""

    dpi: float
    distance_cm: float
    name: str = ""

    @property
    def pixels_per_degree(self) -> float:
        cm_per_degree = self.distance_cm * math.tan(math.radians(1.0))
        return cm_per_degree * self.dpi / 2.54


SCREEN_96DPI_60CM = ViewingCondition(96.0, 60.0, "screen_96dpi_60cm")  # ≈ 39.6 px/deg, project default
PRINT_300DPI_40CM = ViewingCondition(300.0, 40.0, "print_300dpi_40cm")  # ≈ 82.5 px/deg
DEFAULT_VIEWING = SCREEN_96DPI_60CM

REFERENCE_PPD = 32.0
"""Pixels/degree the published table was computed for (Watson 1993, §9: "32 pixels/degree")."""

# --- stage 1: frequency sensitivity -----------------------------------------------------------

WATSON_TABLE = torch.tensor(
    [
        [1.40, 1.01, 1.16, 1.66, 2.40, 3.43, 4.79, 6.56],
        [1.01, 1.45, 1.32, 1.52, 2.00, 2.71, 3.67, 4.93],
        [1.16, 1.32, 2.24, 2.59, 2.98, 3.64, 4.60, 5.88],
        [1.66, 1.52, 2.59, 3.77, 4.55, 5.30, 6.28, 7.60],
        [2.40, 2.00, 2.98, 4.55, 6.15, 7.46, 8.71, 10.17],
        [3.43, 2.71, 3.64, 5.30, 7.46, 9.62, 11.58, 13.51],
        [4.79, 3.67, 4.60, 6.28, 8.71, 11.58, 14.50, 17.29],
        [6.56, 4.93, 5.88, 7.60, 10.17, 13.51, 17.29, 21.15],
    ]
)
"""``t[u, v]``: threshold amplitude of DCT basis (u, v), 0–255 pixel units, at ``REFERENCE_PPD``."""

# Ahumada & Peterson (1992) log-parabola, least-squares fitted to WATSON_TABLE at 32 px/deg:
#   log10 T(f, θ) = log10 t_min − log10(r + (1 − r) cos²θ) + K (log10 f − log10 f_min)²
# with f_u = u · ppd / 16 cycles/degree and θ = arcsin(2 f_u f_v / f²).
_FIT = {"log_tmin": math.log10(0.9498), "log_fmin": math.log10(2.753), "K": 1.5194, "r": 0.6175}


def _log_parabola(ppd: float) -> torch.Tensor:
    """``(8, 8)`` log10 threshold from the fitted model at ``ppd``; DC entry is NaN."""
    u = torch.arange(8, dtype=torch.float64)[:, None]
    v = torch.arange(8, dtype=torch.float64)[None, :]
    fu, fv = u * ppd / 16.0, v * ppd / 16.0
    f2 = fu**2 + fv**2
    f = f2.sqrt()
    sin_theta = torch.where(f2 > 0, 2 * fu * fv / f2.clamp_min(1e-12), torch.zeros_like(f2))
    cos2 = 1 - sin_theta**2
    lt = _FIT["log_tmin"] - torch.log10(_FIT["r"] + (1 - _FIT["r"]) * cos2) + _FIT["K"] * (torch.log10(f) - _FIT["log_fmin"]) ** 2
    return lt


def frequency_sensitivity(viewing: ViewingCondition = DEFAULT_VIEWING) -> torch.Tensor:
    """``(8, 8)`` float32 threshold table at ``viewing``: the published table rescaled by the
    fitted parabola's ratio ``model(ppd) / model(32)``. DC is viewing-independent."""
    ppd = viewing.pixels_per_degree
    if abs(ppd - REFERENCE_PPD) < 1e-9:
        return WATSON_TABLE.clone()
    ratio = 10 ** (_log_parabola(ppd) - _log_parabola(REFERENCE_PPD))
    ratio[0, 0] = 1.0
    return (WATSON_TABLE.to(torch.float64) * ratio).to(torch.float32)


# --- stages 2 and 3 ----------------------------------------------------------------------------

A_T = 0.649
"""Luminance-masking exponent (Watson 1993, after Ahumada & Peterson)."""
W_CONTRAST = 0.7
"""Contrast-masking exponent for AC terms; the DC term uses 0."""
DC_REFERENCE = 1024.0
"""DC coefficient of an 8-bit mid-grey block (8 × 128)."""
DC_FLOOR = 8.0
"""Blocks darker than mean level 1 are treated as level 1: the power law is invalid there and a
zero threshold would make the slack degenerate."""


def luminance_masking(t: torch.Tensor, dc: torch.Tensor, a_t: float = A_T) -> torch.Tensor:
    """``t (8, 8)``, ``dc (B, nh, nw)`` → ``(B, nh, nw, 8, 8)`` luminance-masked thresholds."""
    ratio = (dc.clamp_min(DC_FLOOR) / DC_REFERENCE) ** a_t
    return t.to(dc.device, dc.dtype)[None, None, None] * ratio[..., None, None]


def contrast_masking(t_l: torch.Tensor, coef: torch.Tensor, w: float = W_CONTRAST) -> torch.Tensor:
    """``max(t_L, |C|^w · t_L^(1−w))`` elementwise on ``(B, nh, nw, 8, 8)``; DC uses ``w = 0``."""
    w_mat = torch.full((8, 8), w, device=coef.device, dtype=coef.dtype)
    w_mat[0, 0] = 0.0
    masked = coef.abs().clamp_min(1e-12) ** w_mat * t_l ** (1 - w_mat)
    return torch.maximum(t_l, masked)


def watson_slack(
    y_coef: torch.Tensor,
    viewing: ViewingCondition = DEFAULT_VIEWING,
    a_t: float = A_T,
    w: float = W_CONTRAST,
) -> torch.Tensor:
    """Per-coefficient slack, same shape as ``y_coef`` (``(B, H, W)``, luminance block-DCT in
    0–255 units from ``markers.classical.dct.luma_dct``).

    ``slack[b, y, x]`` is the amount coefficient ``(b, y, x)`` may change before the change is
    predicted visible under ``viewing``. Frequency sensitivity → luminance masking (block DC) →
    contrast masking (coefficient magnitude).
    """
    if y_coef.ndim != 3 or y_coef.shape[-1] % 8 or y_coef.shape[-2] % 8:
        raise ValueError(f"expected (B, H, W) with H, W multiples of 8, got {tuple(y_coef.shape)}")
    bl = blocks(y_coef)  # (B, nh, nw, 8, 8)
    t = frequency_sensitivity(viewing)
    t_l = luminance_masking(t, bl[..., 0, 0], a_t)
    s = contrast_masking(t_l, bl, w)
    return unblocks(s)
