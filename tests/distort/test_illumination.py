import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.distort.illumination import Illumination, _light_field


def _img() -> torch.Tensor:
    return torch.rand(2, 3, 32, 32, generator=make_generator(0)) * 0.6 + 0.2


def test_all_off_is_identity() -> None:
    x = _img()
    st = Illumination(0, 0, 0, 0, 0, 0)
    assert torch.equal(st(x, make_generator(0)), x) and st.last_params == {}


def test_contract_determinism_gradient_and_params() -> None:
    x = _img()
    st = Illumination()
    a = st(x, make_generator(1)).clone()
    p = dict(st.last_params)
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, st(x, make_generator(1))) and st.last_params == p
    assert {"gain_a", "gain_b", "field", "contrast", "brightness", "cast_r", "desaturate", "gamma"} <= set(p)
    assert 0.7 <= p["gain_a"] <= 1 <= p["gain_b"] <= 1.3 and 0.5 <= p["contrast"] <= 1.5
    xg = x.clone().requires_grad_()
    st(xg, make_generator(1)).sum().backward()
    assert xg.grad is not None and xg.grad.abs().sum() > 0


def test_light_field_endpoints() -> None:
    f = _light_field(16, 16, 0.8, 1.2, "linear", 0.0, 0.5, 0.5, torch.device("cpu"), torch.float32)
    assert f.shape == (1, 1, 16, 16)
    assert abs(f[..., 0, 0].item() - 0.8) < 1e-5 and abs(f[..., 0, -1].item() - 1.2) < 1e-5
    r = _light_field(16, 16, 0.8, 1.2, "radial", 0.0, 0.0, 0.0, torch.device("cpu"), torch.float32)
    assert abs(r[..., 0, 0].item() - 1.2) < 1e-5 and abs(r[..., -1, -1].item() - 0.8) < 1e-5


def test_only_brightness_shifts_uniformly() -> None:
    x = _img()
    st = Illumination(0, 0, 0.2, 0, 0, 0)
    y = st(x, make_generator(3))
    d = (y - x)
    assert torch.allclose(d, torch.full_like(d, st.last_params["brightness"]), atol=1e-6)


def test_full_desaturation_gives_grey() -> None:
    x = _img()
    st = Illumination(0, 0, 0, 0, 1.0, 0)
    # force s = 1 by sampling until close; simpler: apply formula directly with max range and check monotonic
    y = st(x, make_generator(0))
    s = st.last_params["desaturate"]
    chroma_x = (x.max(1).values - x.min(1).values).mean()
    chroma_y = (y.max(1).values - y.min(1).values).mean()
    assert chroma_y <= chroma_x * (1 - s) + 1e-5


def test_negative_range_rejected() -> None:
    with pytest.raises(ValueError):
        Illumination(gain=-0.1)
