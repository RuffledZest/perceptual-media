"""Measure the real screen→camera channel from rectified captures, and match it to the simulator.

For each capture we have the rectified ``image_px²`` crop and the reference it was a photo of.
The *geometric* part of the channel is already characterised (``captures.csv``: measured angle,
distance, scale). This module measures the rest — what the simulator's ``screen_camera`` chain
models as illumination, moiré, blur, resize, JPEG and noise — in a handful of numbers that can be
computed identically on a simulated output:

* ``psnr_raw / ssim_raw / lpips_raw`` — capture vs reference as-is (dominated by exposure);
* colour map — a per-channel **tone curve** (32-bin lookup, ``fit_tone_curve``) fitted from the
  pair, since a phone's tone mapping is nonlinear; the per-channel affine ``gain / offset`` is
  reported alongside as a summary. ``psnr_cc / ssim_cc / lpips_cc`` compare the capture with the
  tone-mapped reference, i.e. everything the channel did *beyond* a global colour change;
* ``blur_sigma`` — the Gaussian σ (image px) that, applied to the colour-mapped reference, best
  predicts the capture (grid search); meaningful on textured content, reported as NaN on flat
  fills where nothing can be measured;
* ``noise_std`` — high-frequency residual std (grey levels) of the capture minus its 5×5 median,
  measured on the reference's flat regions (local reference std < 2) so content is excluded;
* ``moire_amp / moire_freq`` — the strongest periodic component of the capture's residual on the
  same flat regions: amplitude (grey levels) and frequency (cycles / image px);
* ``illum_range`` — peak-to-peak of a smooth (σ = 48 px) fit to the residual, divided by the
  mean level: the low-frequency shading the camera/panel added;
* ``clip_frac`` — fraction of capture pixels at 0 or 255.

``sim_channel_stats`` computes the same numbers for the simulator at a severity, and
``match_severity`` finds the severity whose statistics are closest to a real capture's (weighted
L2 over normalised ``blur_sigma, noise_std, moire_amp, psnr_cc``). That mapping — "condition X
on my rig ≈ severity s" — is the deliverable of the calibration set (plan: Risks, row 1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np
import torch

from perceptual_media.core.io import from_uint8
from perceptual_media.core.seed import make_generator
from perceptual_media.core.types import ImageBatch
from perceptual_media.distort.presets import build_chain
from perceptual_media.metrics.fidelity import fidelity

MIN_FLAT_FRAC = 0.9
"""Residual (noise / moiré / shading) statistics need at least this much flat reference: on a
text poster (71 % flat) the phone's sharpening halos around glyphs leak into the mask and read
as a periodic component, so only the flat and gradient classes carry these statistics."""


@dataclass
class ChannelStats:
    psnr_raw: float
    ssim_raw: float
    lpips_raw: float
    gain_r: float
    gain_g: float
    gain_b: float
    offset_r: float
    offset_g: float
    offset_b: float
    psnr_cc: float
    ssim_cc: float
    lpips_cc: float
    blur_sigma: float
    noise_std: float
    moire_amp: float
    moire_freq: float
    illum_range: float
    clip_frac: float
    flat_frac: float
    """Fraction of the reference used for the flat-region statistics."""

    @classmethod
    def columns(cls) -> list[str]:
        return [f.name for f in fields(cls)]


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def fit_colour_map(ref: np.ndarray, cap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel least-squares ``cap ≈ gain · ref + offset`` on float ``(H, W, 3)`` in [0, 255].
    Fitted on a 4× box-downscale so noise and moiré do not bias it."""
    r = cv2.resize(ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    c = cv2.resize(cap, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    gain, offset = np.zeros(3), np.zeros(3)
    for ch in range(3):
        a = np.stack([r[:, ch], np.ones(len(r))], axis=1)
        if r[:, ch].std() < 1e-3:  # flat reference: gain is unidentifiable, keep 1 and fit offset
            gain[ch], offset[ch] = 1.0, c[:, ch].mean() - r[:, ch].mean()
        else:
            gain[ch], offset[ch] = np.linalg.lstsq(a, c[:, ch], rcond=None)[0]
    return gain, offset


def apply_colour_map(ref: np.ndarray, gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.clip(ref * gain + offset, 0, 255)


def fit_tone_curve(ref: np.ndarray, cap: np.ndarray, bins: int = 32) -> np.ndarray:
    """Per-channel tone curve ``capture = f_c(reference)``: the mean capture value in each of
    ``bins`` reference-level bins, linearly interpolated (a phone's tone mapping is nonlinear,
    which a gain/offset cannot fit). Fitted on a 4× box-downscale. Returns ``ref`` mapped."""
    r4 = cv2.resize(ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    c4 = cv2.resize(cap, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    width = 256 / bins
    centres = (np.arange(bins) + 0.5) * width
    out = np.empty_like(ref)
    for ch in range(3):
        idx = np.clip((r4[:, ch] / width).astype(int), 0, bins - 1)
        sums = np.bincount(idx, weights=c4[:, ch], minlength=bins)
        counts = np.bincount(idx, minlength=bins)
        ok = counts > 0
        out[..., ch] = np.interp(ref[..., ch], centres[ok], sums[ok] / counts[ok])
    return np.clip(out, 0, 255)


def flat_mask(ref: np.ndarray, max_local_std: float = 2.0, win: int = 9) -> np.ndarray:
    """Where the *reference* is locally flat, so residual statistics measure the channel, not content."""
    g = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = cv2.blur(g, (win, win))
    m2 = cv2.blur(g * g, (win, win))
    std = np.sqrt(np.clip(m2 - m * m, 0, None))
    mask = std < max_local_std
    mask[: win // 2, :] = mask[-(win // 2) :, :] = False
    mask[:, : win // 2] = mask[:, -(win // 2) :] = False
    return mask


BLUR_SIGMAS = (0, 0.5, 0.75, 1, 1.5, 2, 3, 4)


def fit_blur_sigma(ref_cc: np.ndarray, cap: np.ndarray, sigmas: tuple[float, ...] = BLUR_SIGMAS) -> float:
    """σ of the Gaussian blur of ``ref_cc`` closest (MSE on grey) to ``cap``; NaN if the reference
    is too flat for blur to be identifiable (its grey std < 4)."""
    rg = cv2.cvtColor(ref_cc.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    cg = cv2.cvtColor(cap.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    if rg.std() < 4:
        return float("nan")
    best, best_err = float("nan"), np.inf
    for s in sigmas:
        b = cv2.GaussianBlur(rg, (0, 0), s) if s > 0 else rg
        err = float(np.mean((b - cg) ** 2))
        if err < best_err:
            best, best_err = float(s), err
    return best


def residual_stats(cap: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    """``noise_std, moire_amp, moire_freq, illum_range`` from the capture's grey channel on ``mask``."""
    g = cv2.cvtColor(cap.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    if mask.mean() < MIN_FLAT_FRAC:  # too little flat reference: residuals would be content
        return float("nan"), float("nan"), float("nan"), float("nan")
    smooth = cv2.GaussianBlur(g, (0, 0), 48)
    illum = float((smooth[mask].max() - smooth[mask].min()) / max(smooth[mask].mean(), 1.0))
    hf = g - cv2.medianBlur(g.astype(np.uint8), 5).astype(np.float32)
    noise = float(hf[mask].std())
    # Periodic content: FFT of the (mask-filled) residual, strongest peak away from DC.
    resid = (g - smooth) * mask
    spec = np.abs(np.fft.fftshift(np.fft.fft2(resid))) / mask.sum()
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    fr = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)  # cycles / px
    spec[fr < 0.01] = 0  # DC and shading
    idx = np.unravel_index(int(np.argmax(spec)), spec.shape)
    moire_amp = float(2 * spec[idx])  # amplitude of that sinusoid, grey levels
    return noise, moire_amp, float(fr[idx]), illum


# ---------------------------------------------------------------------------
# Whole-capture and simulator statistics
# ---------------------------------------------------------------------------


def _fidelity_np(a: np.ndarray, b: np.ndarray, device: torch.device) -> tuple[float, float, float]:
    ta = from_uint8(np.clip(a, 0, 255).astype(np.uint8)).to(device)
    tb = from_uint8(np.clip(b, 0, 255).astype(np.uint8)).to(device)
    f = fidelity(ta, tb)
    return float(f["psnr"]), float(f["ssim"]), float(f["lpips"])


def channel_stats(ref_u8: np.ndarray, cap_u8: np.ndarray, device: torch.device | None = None) -> ChannelStats:
    """All statistics for one ``(H, W, 3)`` uint8 reference / capture pair."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref, cap = ref_u8.astype(np.float32), cap_u8.astype(np.float32)
    psnr_raw, ssim_raw, lpips_raw = _fidelity_np(cap, ref, device)
    gain, offset = fit_colour_map(ref, cap)  # reported as a summary of the colour change
    ref_cc = fit_tone_curve(ref, cap)  # used for the "beyond colour" comparison
    psnr_cc, ssim_cc, lpips_cc = _fidelity_np(cap, ref_cc, device)
    mask = flat_mask(ref)
    noise, moire_amp, moire_freq, illum = residual_stats(cap, mask)
    return ChannelStats(
        psnr_raw=psnr_raw,
        ssim_raw=ssim_raw,
        lpips_raw=lpips_raw,
        gain_r=float(gain[0]),
        gain_g=float(gain[1]),
        gain_b=float(gain[2]),
        offset_r=float(offset[0]),
        offset_g=float(offset[1]),
        offset_b=float(offset[2]),
        psnr_cc=psnr_cc,
        ssim_cc=ssim_cc,
        lpips_cc=lpips_cc,
        blur_sigma=fit_blur_sigma(ref_cc, cap),
        noise_std=noise,
        moire_amp=moire_amp,
        moire_freq=moire_freq,
        illum_range=illum,
        clip_frac=float(np.mean((cap_u8 == 0) | (cap_u8 == 255))),
        flat_frac=float(mask.mean()),
    )


def simulate(ref_u8: np.ndarray, preset: str, severity: float, seed: int, device: torch.device) -> np.ndarray:
    """Run ``preset`` at ``severity`` on one uint8 image; returns uint8 ``(H, W, 3)``."""
    chain = build_chain(preset, severity).to(device)
    x: ImageBatch = from_uint8(ref_u8).to(device)
    gen = make_generator(seed, device)
    with torch.no_grad():
        y = chain(x, gen)
    return (y[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def sim_channel_stats(
    ref_u8: np.ndarray,
    severity: float,
    *,
    preset: str = "screen_channel",
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    device: torch.device | None = None,
) -> dict[str, float]:
    """Mean ``ChannelStats`` of the simulator at ``severity`` over ``seeds`` (NaNs ignored)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [asdict(channel_stats(ref_u8, simulate(ref_u8, preset, severity, s, device), device)) for s in seeds]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


MATCH_KEYS = ("blur_sigma", "noise_std", "moire_amp", "psnr_cc")
"""Keys compared when matching a real capture to a simulator severity."""


def match_severity(real: dict[str, float], sim_grid: list[tuple[float, dict[str, float]]]) -> tuple[float, float]:
    """Severity in ``sim_grid`` (``[(severity, stats), ...]``) closest to ``real``; returns
    ``(severity, distance)``. Each key is normalised by its range over the grid; NaN keys
    (e.g. blur on a flat image) are skipped."""
    keys = [k for k in MATCH_KEYS if np.isfinite(real.get(k, np.nan))]
    scale = {}
    for k in keys:
        vals = [s[k] for _, s in sim_grid]
        scale[k] = max(np.nanmax(vals) - np.nanmin(vals), 1e-6)
    best, best_d = float("nan"), np.inf
    for sev, stats in sim_grid:
        d = float(np.sqrt(np.nanmean([((stats[k] - real[k]) / scale[k]) ** 2 for k in keys])))
        if d < best_d:
            best, best_d = sev, d
    return best, best_d
