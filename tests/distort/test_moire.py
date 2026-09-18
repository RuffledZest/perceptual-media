import pytest
import torch
import torch.nn.functional as F

from perceptual_media.core.seed import make_generator
from perceptual_media.distort.moire import (
    Moire,
    display_render,
    expected_beat_frequency,
    pimog_pattern,
)


def _dominant_freq(y: torch.Tensor) -> float:
    prof = y.mean(1)[0].mean(0)
    prof = prof - prof.mean()
    spec = torch.fft.rfft(prof).abs()
    spec[0] = 0
    return torch.argmax(spec).item() / prof.numel()


def test_display_render_is_energy_preserving() -> None:
    x = torch.rand(1, 3, 8, 8, generator=make_generator(0))
    hi = display_render(x, 4)
    assert hi.shape == (1, 3, 32, 32)
    assert torch.allclose(F.avg_pool2d(hi, 4), x, atol=1e-6)


@pytest.mark.parametrize("scale", [0.7, 0.9, 1.15])
def test_flat_grey_beat_frequency_matches_sampling_theory(scale: float) -> None:
    x = torch.full((1, 3, 256, 256), 0.5)
    m = Moire(scale, scale, psf_sigma=0.0, max_rotation_deg=0.0)
    y = m(x, make_generator(0))
    expected = expected_beat_frequency(scale)
    assert _dominant_freq(y) == pytest.approx(expected, rel=0.10)
    assert (y - 0.5).abs().max() > 0.05  # a visible periodic pattern, not a flat image


def test_expected_beat_frequency() -> None:
    assert expected_beat_frequency(1.0) == 0.0
    assert expected_beat_frequency(0.9) == pytest.approx(0.1)
    assert expected_beat_frequency(0.5) == pytest.approx(0.0)


def test_strength_zero_is_identity_and_blend() -> None:
    x = torch.rand(1, 3, 32, 32, generator=make_generator(1))
    assert torch.equal(Moire(strength=0.0)(x, make_generator(0)), x)
    full = Moire(0.8, 0.8, strength=1.0)(x, make_generator(2))
    half = Moire(0.8, 0.8, strength=0.5)(x, make_generator(2))
    assert torch.allclose(half, x + 0.5 * (full - x), atol=1e-6)


@pytest.mark.parametrize("mode", ["sampling", "pimog"])
def test_contract_determinism_gradient(mode: str) -> None:
    x = torch.rand(1, 3, 32, 32, generator=make_generator(3))
    m = Moire(mode=mode)  # type: ignore[arg-type]
    a = m(x, make_generator(4)).clone()
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, m(x, make_generator(4)))
    assert m.last_params["mode"] == mode
    xg = x.clone().requires_grad_()
    m(xg, make_generator(4)).sum().backward()
    assert xg.grad is not None and xg.grad.abs().sum() > 0


def test_pimog_pattern_range() -> None:
    p = pimog_pattern(16, 16, (5.0, 7.0), 30.0, torch.device("cpu"), torch.float32)
    assert p.shape == (1, 1, 16, 16) and p.min() >= 0.5 and p.max() <= 1.0


def test_bad_args() -> None:
    with pytest.raises(ValueError):
        Moire(0, 1)
    with pytest.raises(ValueError):
        Moire(strength=2)
