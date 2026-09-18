import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.distort.blur import DefocusBlur, MotionBlur, gaussian_kernel, motion_kernel


def test_motion_kernel_angle_zero_is_box() -> None:
    for L in (3, 5, 9):
        k = motion_kernel(L, 0.0)
        r = k.shape[0] // 2
        box = torch.zeros_like(k)
        box[r, r - L // 2 : r + L // 2 + 1] = 1.0 / L
        assert torch.allclose(k, box)


def test_motion_kernel_angle_90_is_vertical_box() -> None:
    k = motion_kernel(5, 90.0)
    assert torch.allclose(k, motion_kernel(5, 0.0).T, atol=1e-6)


def test_gaussian_kernel_normalised_symmetric() -> None:
    k = gaussian_kernel(1.5)
    assert k.shape == (11, 11) and abs(k.sum().item() - 1) < 1e-6
    assert torch.allclose(k, k.T) and torch.allclose(k, k.flip(0))
    assert gaussian_kernel(10.0, max_radius=4).shape == (9, 9)


@pytest.mark.parametrize("stage", [DefocusBlur(0.5, 3.0), MotionBlur(3.0, 9.0)])
def test_blur_stage_contract(stage: torch.nn.Module) -> None:
    x = torch.rand(2, 3, 32, 32, generator=make_generator(0))
    a = stage(x, make_generator(1)).clone()
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, stage(x, make_generator(1)))
    # blur reduces high-frequency energy
    hf = lambda t: (t[..., 1:] - t[..., :-1]).abs().mean()  # noqa: E731
    assert hf(a) < hf(x)
    xg = x.clone().requires_grad_()
    stage(xg, make_generator(1)).sum().backward()
    assert xg.grad is not None and xg.grad.abs().sum() > 0


def test_blur_identity_thresholds() -> None:
    x = torch.rand(1, 3, 16, 16)
    assert torch.equal(DefocusBlur(0.0, 0.0)(x, make_generator(0)), x)
    assert torch.equal(MotionBlur(0.0, 1.0)(x, make_generator(0)), x)


def test_flat_image_is_invariant() -> None:
    x = torch.full((1, 3, 24, 24), 0.3)
    assert torch.allclose(DefocusBlur(2.0, 2.0)(x, make_generator(0)), x, atol=1e-6)
    assert torch.allclose(MotionBlur(7.0, 7.0)(x, make_generator(0)), x, atol=1e-6)
