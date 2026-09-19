import cv2
import numpy as np
import pytest
import torch

from perceptual_media.capture.channel import (
    channel_stats,
    fit_blur_sigma,
    fit_colour_map,
    fit_tone_curve,
    flat_mask,
    match_severity,
    residual_stats,
)


def _texture(seed=0, size=256):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.integers(0, 256, (size, size, 3)).astype(np.float32), (0, 0), 3)
    return np.clip(cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX), 0, 255)


def test_blur_sigma_recovers_known_blur():
    ref = _texture()
    for true in (0.75, 2.0):
        cap = cv2.GaussianBlur(ref, (0, 0), true)
        assert fit_blur_sigma(ref, cap) == pytest.approx(true)
    assert np.isnan(fit_blur_sigma(np.full((64, 64, 3), 128.0), np.full((64, 64, 3), 200.0)))


def test_colour_maps_recover_gain_offset_and_gamma():
    ref = _texture(1)
    cap = np.clip(ref * np.array([1.2, 1.0, 0.8]) + np.array([10, 20, 30]), 0, 255)
    gain, offset = fit_colour_map(ref, cap)
    np.testing.assert_allclose(gain, [1.2, 1.0, 0.8], atol=0.02)
    np.testing.assert_allclose(offset, [10, 20, 30], atol=2)
    # A gamma curve is not affine; the tone curve should fit it to within a couple of levels.
    cap_g = 255 * (ref / 255) ** 0.6
    mapped = fit_tone_curve(ref, cap_g)
    assert np.abs(mapped - cap_g).mean() < 2.5
    assert np.abs(ref * gain + offset - cap_g).mean() > 5  # the affine fit is clearly worse


def test_residual_stats_find_planted_sinusoid_and_noise():
    rng = np.random.default_rng(0)
    h = w = 256
    yy, xx = np.mgrid[:h, :w]
    freq, amp = 0.15, 6.0
    field = 128 + amp * np.sin(2 * np.pi * freq * xx) + rng.normal(0, 1.0, (h, w))
    cap = np.repeat(np.clip(field, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
    ref = np.full((h, w, 3), 128, np.uint8)
    mask = flat_mask(ref.astype(np.float32))
    assert mask.mean() > 0.9
    noise, moire_amp, moire_freq, illum = residual_stats(cap.astype(np.float32), mask)
    assert moire_freq == pytest.approx(freq, abs=0.01)
    assert moire_amp == pytest.approx(amp, rel=0.25)
    assert illum < 0.05
    # The 5x5-median residual sees the sinusoid too, so noise_std is at least the white noise.
    assert 0.8 < noise < 8


def test_channel_stats_identity_is_clean():
    ref = _texture(2).astype(np.uint8)
    s = channel_stats(ref, ref, torch.device("cpu"))
    assert s.psnr_raw > 60 and s.blur_sigma == 0.0 and s.clip_frac < 1e-3


def test_match_severity_picks_nearest_and_skips_nan():
    grid = [
        (s, {"blur_sigma": 2 * s, "noise_std": 3 * s, "moire_amp": 5 * s, "psnr_cc": 40 - 20 * s})
        for s in (0.0, 0.5, 1.0)
    ]
    sev, dist = match_severity({"blur_sigma": 1.1, "noise_std": 1.4, "moire_amp": 2.6, "psnr_cc": 31}, grid)
    assert sev == 0.5 and dist < 0.1
    sev, _ = match_severity({"blur_sigma": float("nan"), "noise_std": 3.0, "moire_amp": 5.0, "psnr_cc": 20}, grid)
    assert sev == 1.0
