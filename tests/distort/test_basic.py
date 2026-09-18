import json

import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.distort.base import DistortionChain
from perceptual_media.distort.basic import Crop, GaussianNoise, Resize
from perceptual_media.harness.results import to_json


def _img(seed: int = 0, size: int = 32) -> torch.Tensor:
    return torch.rand(2, 3, size, size, generator=make_generator(seed))


def _grad_flows(stage: torch.nn.Module) -> bool:
    x = _img().requires_grad_()
    y = stage(x, make_generator(1))
    y.sum().backward()
    return x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


@pytest.mark.parametrize("stage", [GaussianNoise(0.05), Resize(0.5, 0.9), Crop(0.3, 0.8)])
def test_shape_range_determinism_and_gradient(stage: torch.nn.Module) -> None:
    x = _img()
    a = stage(x, make_generator(7)).clone()
    pa = dict(stage.last_params)
    b = stage(x, make_generator(7))
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, b) and stage.last_params == pa
    assert not torch.equal(a, stage(x, make_generator(8)))
    assert _grad_flows(stage)
    json.loads(to_json(stage.last_params))  # params are loggable


def test_noise_sigma_zero_is_identity_and_scale_tracks_sigma() -> None:
    x = _img()
    assert torch.equal(GaussianNoise(0.0)(x, make_generator(0)), x)
    big = GaussianNoise(0.2)
    for seed in range(5):
        y = big(x, make_generator(seed))
        sigma = big.last_params["sigma"]
        assert 0 <= sigma <= 0.2
        # empirical std tracks the sampled sigma (clamping at [0,1] shrinks it slightly)
        assert (y - x).std().item() == pytest.approx(sigma, rel=0.25, abs=2e-3)


def test_resize_scale_one_is_identity_and_downscale_blurs() -> None:
    x = _img(size=64)
    assert torch.equal(Resize(1.0, 1.0)(x, make_generator(0)), x)
    y = Resize(0.25, 0.25)(x, make_generator(0))
    # high-frequency energy drops: neighbouring-pixel difference shrinks
    assert (y[..., 1:] - y[..., :-1]).abs().mean() < 0.5 * (x[..., 1:] - x[..., :-1]).abs().mean()


def test_crop_keeps_rectangle_and_fills_outside() -> None:
    x = _img(size=40)
    c = Crop(0.25, 0.25, fill=0.5)
    y = c(x, make_generator(2))
    p = c.last_params
    assert p["h"] == 20 and p["w"] == 20
    y0, x0 = p["y0"], p["x0"]
    assert torch.equal(y[..., y0 : y0 + 20, x0 : x0 + 20], x[..., y0 : y0 + 20, x0 : x0 + 20])
    outside = y.clone()
    outside[..., y0 : y0 + 20, x0 : x0 + 20] = 0.5
    assert torch.all(outside == 0.5)
    assert torch.equal(Crop(1.0, 1.0)(x, make_generator(0)), x)


def test_bad_ranges() -> None:
    with pytest.raises(ValueError):
        Resize(0.0, 1.0)
    with pytest.raises(ValueError):
        Crop(0.9, 0.5)


def test_chain_of_basics_logs_all_params() -> None:
    chain = DistortionChain([Resize(0.5, 0.9), GaussianNoise(0.02), Crop(0.5, 1.0)], name="basic")
    y = chain(_img(), make_generator(0))
    assert y.shape == (2, 3, 32, 32)
    assert set(chain.last_params) == {"resize", "noise", "crop"}
    assert "scale" in chain.last_params["resize"] and "sigma" in chain.last_params["noise"]
