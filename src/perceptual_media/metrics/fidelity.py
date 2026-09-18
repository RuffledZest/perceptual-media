"""Perceptual-distance metrics: PSNR, SSIM, LPIPS. Batched, ``(B,)`` per item.

All inputs are ``(B, 3, H, W)`` float32 in ``[0, 1]`` (``core.types``). These are computed on
*marked vs. original* — they measure how visible the embedding is, not what the channel did.

Caveat from the brief (§4.1): PSNR and SSIM correlate poorly with watermark visibility. They are
logged because every paper reports them; the 2AFC study is the ground truth.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from perceptual_media.core.types import ImageBatch, assert_image_batch

PSNR_MAX = 100.0
"""Returned instead of ``inf`` for identical images, so CSV/plots stay finite."""

LPIPS_MIN_SIZE = 32
"""AlexNet's receptive field needs ≥ 32 px; smaller inputs are bilinearly upsampled to this."""


def _check_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    assert_image_batch(a, name="a")
    assert_image_batch(b, name="b")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")


def psnr(a: ImageBatch, b: ImageBatch) -> torch.Tensor:
    """``(B,)`` PSNR in dB over all channels, ``data_range = 1``; capped at ``PSNR_MAX``."""
    _check_pair(a, b)
    mse = ((a - b) ** 2).flatten(1).mean(dim=1)
    out = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-30))
    return out.clamp_max(PSNR_MAX)


def ssim(a: ImageBatch, b: ImageBatch, win_size: int = 7, k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    """``(B,)`` mean SSIM, matching ``skimage.metrics.structural_similarity`` defaults.

    Uniform ``win_size`` window, sample (N−1) covariance, border cropped to the valid region,
    averaged over channels — i.e. ``structural_similarity(a, b, channel_axis=-1, data_range=1)``.
    """
    _check_pair(a, b)
    if win_size % 2 == 0 or win_size < 3:
        raise ValueError("win_size must be odd and >= 3")
    if a.shape[-1] < win_size or a.shape[-2] < win_size:
        raise ValueError(f"image smaller than win_size={win_size}")
    n = win_size * win_size
    cov_norm = n / (n - 1)  # sample covariance, as skimage
    c1, c2 = (k1 * 1.0) ** 2, (k2 * 1.0) ** 2

    def mean(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, win_size, stride=1)

    ux, uy = mean(a), mean(b)
    uxx, uyy, uxy = mean(a * a), mean(b * b), mean(a * b)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return s.flatten(1).mean(dim=1)


_LPIPS_CACHE: dict[str, torch.nn.Module] = {}


def _lpips_model(device: torch.device) -> torch.nn.Module:
    key = str(device)
    if key not in _LPIPS_CACHE:
        with warnings.catch_warnings():
            # lpips 0.1.4 uses torchvision's deprecated `pretrained=` API; not our problem.
            warnings.simplefilter("ignore")
            import lpips as _lpips  # heavy import, deferred

            m = _lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _LPIPS_CACHE[key] = m
    return _LPIPS_CACHE[key]


def lpips(a: ImageBatch, b: ImageBatch) -> torch.Tensor:
    """``(B,)`` LPIPS (AlexNet backbone, the paper default). Model loaded once per device.

    Inputs in ``[0, 1]`` are rescaled to ``[-1, 1]`` internally. Identical inputs give 0.
    Images with a side < ``LPIPS_MIN_SIZE`` are upsampled (both of them, identically) so the
    metric stays defined; the value is then only comparable among images of that size.
    """
    _check_pair(a, b)
    h, w = a.shape[-2:]
    if min(h, w) < LPIPS_MIN_SIZE:
        scale = LPIPS_MIN_SIZE / min(h, w)
        size = (max(LPIPS_MIN_SIZE, round(h * scale)), max(LPIPS_MIN_SIZE, round(w * scale)))
        a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
        b = F.interpolate(b, size=size, mode="bilinear", align_corners=False)
    model = _lpips_model(a.device)
    with torch.no_grad():
        d = model(a, b, normalize=True)
    return d.flatten()


def fidelity(a: ImageBatch, b: ImageBatch) -> dict[str, torch.Tensor]:
    """All three metrics at once: ``{"psnr", "ssim", "lpips"}`` each ``(B,)``."""
    return {"psnr": psnr(a, b), "ssim": ssim(a, b), "lpips": lpips(a, b)}

