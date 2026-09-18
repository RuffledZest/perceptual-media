import numpy as np
import torch
from scipy.fft import dctn, idctn

from perceptual_media.core.seed import make_generator
from perceptual_media.core.transforms import block_dct, block_idct, rgb_to_ycbcr, ycbcr_to_rgb


def test_block_dct_matches_scipy_every_block() -> None:
    x = torch.rand(2, 16, 24, generator=make_generator(0))
    c = block_dct(x)
    for b in range(2):
        for i in range(0, 16, 8):
            for j in range(0, 24, 8):
                ref = dctn(x[b, i : i + 8, j : j + 8].numpy(), norm="ortho")
                assert np.allclose(c[b, i : i + 8, j : j + 8].numpy(), ref, atol=1e-5)


def test_block_idct_matches_scipy_and_roundtrip() -> None:
    c = torch.randn(1, 8, 8, generator=make_generator(1))
    ref = idctn(c[0].numpy(), norm="ortho")
    assert np.allclose(block_idct(c)[0].numpy(), ref, atol=1e-5)
    x = torch.rand(3, 32, 32, generator=make_generator(2))
    assert (block_idct(block_dct(x)) - x).abs().max() < 1e-5


def test_dct_dc_is_8_times_mean() -> None:
    x = torch.full((1, 8, 8), 0.25)
    assert abs(block_dct(x)[0, 0, 0].item() - 8 * 0.25) < 1e-6


def test_ycbcr_roundtrip_and_grey() -> None:
    x = torch.rand(2, 3, 8, 8, generator=make_generator(3))
    assert (ycbcr_to_rgb(rgb_to_ycbcr(x)) - x).abs().max() < 1e-5
    g = torch.full((1, 3, 4, 4), 0.3)
    ycc = rgb_to_ycbcr(g)
    assert torch.allclose(ycc[:, 0], torch.full((1, 4, 4), 0.3), atol=1e-6)
    assert torch.allclose(ycc[:, 1:], torch.full((1, 2, 4, 4), 0.5), atol=1e-6)


def test_gradient_flows() -> None:
    x = torch.rand(1, 16, 16, requires_grad=True)
    block_idct(block_dct(x)).sum().backward()
    assert x.grad is not None and torch.allclose(x.grad, torch.ones_like(x), atol=1e-5)
