import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity as sk_ssim

from perceptual_media.core.seed import make_generator
from perceptual_media.metrics.fidelity import PSNR_MAX, fidelity, lpips, psnr, ssim


def _pair(seed: int = 0, size: int = 64, sigma: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    g = make_generator(seed)
    x = torch.rand(2, 3, size, size, generator=g)
    y = (x + sigma * torch.randn(x.shape, generator=g)).clamp(0, 1)
    return x, y


def test_psnr_identical_is_capped_finite() -> None:
    x, _ = _pair()
    assert torch.equal(psnr(x, x), torch.full((2,), PSNR_MAX))


def test_psnr_known_value() -> None:
    a = torch.zeros(1, 3, 8, 8)
    b = torch.full((1, 3, 8, 8), 0.1)  # MSE = 0.01 → 20 dB
    assert psnr(a, b).item() == pytest.approx(20.0, abs=1e-3)
    assert psnr(b, a).item() == pytest.approx(20.0, abs=1e-3)


def test_psnr_batched_and_shape_check() -> None:
    x, y = _pair()
    assert psnr(x, y).shape == (2,)
    with pytest.raises(ValueError):
        psnr(x, y[:1])


def test_ssim_matches_skimage() -> None:
    x, y = _pair(seed=1)
    ours = ssim(x, y)
    for i in range(2):
        ref = sk_ssim(x[i].permute(1, 2, 0).numpy(), y[i].permute(1, 2, 0).numpy(), channel_axis=-1, data_range=1.0)
        assert ours[i].item() == pytest.approx(ref, abs=1e-3)


def test_ssim_identical_is_one_and_ordering() -> None:
    x, y_small = _pair(seed=2, sigma=0.02)
    _, y_big = _pair(seed=2, sigma=0.2)
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-6)
    assert (ssim(x, y_small) > ssim(x, y_big)).all()


def test_ssim_rejects_bad_window() -> None:
    x, y = _pair(size=8)
    with pytest.raises(ValueError):
        ssim(x, y, win_size=4)
    with pytest.raises(ValueError):
        ssim(x, y, win_size=9)


def test_lpips_identical_zero_and_ordering() -> None:
    x, y_small = _pair(seed=3, sigma=0.02)
    _, y_big = _pair(seed=3, sigma=0.2)
    assert (lpips(x, x) <= 1e-6).all()
    d_small, d_big = lpips(x, y_small), lpips(x, y_big)
    assert d_small.shape == (2,) and (d_small < d_big).all()


def test_lpips_model_is_cached() -> None:
    from perceptual_media.metrics import fidelity as f

    x, y = _pair(seed=4, size=32)
    lpips(x, y)
    m1 = f._LPIPS_CACHE[str(x.device)]
    lpips(x, y)
    assert f._LPIPS_CACHE[str(x.device)] is m1


def test_fidelity_dict() -> None:
    x, y = _pair(seed=5, size=32)
    d = fidelity(x, y)
    assert set(d) == {"psnr", "ssim", "lpips"} and all(v.shape == (2,) for v in d.values())
    assert np.isfinite(d["psnr"].numpy()).all()


def test_lpips_tiny_images_are_upsampled_not_rejected() -> None:
    x = torch.rand(1, 3, 8, 8)
    assert lpips(x, x).item() <= 1e-6
    y = torch.rand(1, 3, 8, 8)
    assert lpips(x, y).item() > 0
    assert lpips(torch.rand(1, 3, 8, 20), torch.rand(1, 3, 8, 20)).shape == (1,)  # non-square
