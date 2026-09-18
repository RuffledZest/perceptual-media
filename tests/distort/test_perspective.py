import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.distort.perspective import (
    Perspective,
    homography_from_corners,
    pose_corners,
    warp,
)
from perceptual_media.metrics.fidelity import psnr


def _img(size: int = 64) -> torch.Tensor:
    g = make_generator(0)
    x = torch.rand(1, 3, size // 4, size // 4, generator=g)
    return torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False).clamp(0, 1)


def test_identity_homography_is_exact() -> None:
    x = _img()
    assert (warp(x, torch.eye(3)) - x).abs().max().item() <= 1e-6


def test_zero_ranges_are_identity_and_last_H_eye() -> None:
    x = _img()
    for st in (Perspective(0, 0, mode="pose"), Perspective(mode="corners", jitter=0)):
        assert torch.equal(st(x, make_generator(0)), x)
        assert torch.equal(st.last_H, torch.eye(3))


def test_homography_from_corners_maps_points() -> None:
    src = torch.tensor([[0, 0], [64, 0], [64, 64], [0, 64]], dtype=torch.float32)
    dst = torch.tensor([[5, 3], [60, 2], [58, 61], [4, 63]], dtype=torch.float32)
    H = homography_from_corners(src, dst).to(torch.float64)
    p = torch.cat([src.to(torch.float64), torch.ones(4, 1, dtype=torch.float64)], 1) @ H.T
    assert torch.allclose(p[:, :2] / p[:, 2:], dst.to(torch.float64), atol=1e-3)


def test_pose_corners_tilt_foreshortens_bottom_edge() -> None:
    c = pose_corners(512, 512, 40, 0, 0, 60)
    top_w, bot_w = c[1, 0] - c[0, 0], c[2, 0] - c[3, 0]
    assert top_w > bot_w  # trapezoid: far edge shorter
    assert abs(top_w.item() - 512) < 1e-3  # re-fit fills the frame
    assert torch.allclose(pose_corners(512, 512, 0, 0, 0, 60), torch.tensor([[0, 0], [512, 0], [512, 512], [0, 512]], dtype=torch.float32), atol=1e-3)


def test_pure_translation_warp_shifts_pixels() -> None:
    x = _img()
    H = torch.tensor([[1, 0, 3], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)  # shift right by 3 px
    y = warp(x, H, fill=0.0)
    assert torch.allclose(y[..., 3:], x[..., :-3], atol=1e-5)
    assert torch.all(y[..., :3] == 0)


@pytest.mark.parametrize("mode", ["pose", "corners"])
def test_stage_contract_and_gradient(mode: str) -> None:
    x = _img()
    st = Perspective(max_angle=30, max_roll=5, mode=mode, jitter=0.1)  # type: ignore[arg-type]
    a = st(x, make_generator(4)).clone()
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, st(x, make_generator(4)))
    assert not torch.equal(a, x) and st.last_params["mode"] == mode
    xg = x.clone().requires_grad_()
    st(xg, make_generator(4)).sum().backward()
    assert xg.grad is not None and xg.grad.abs().sum() > 0


def test_oracle_inverse_recovers_interior() -> None:
    x = _img(128)
    st = Perspective(max_angle=35, max_roll=5, mode="pose")
    y = st(x, make_generator(2))
    r = st.inverse_warp(y)
    inner = slice(24, 104)
    # two bilinear resamplings on a smooth image: interior comes back close
    assert psnr(r[..., inner, inner], x[..., inner, inner]).item() > 30
    assert psnr(y[..., inner, inner], x[..., inner, inner]).item() < psnr(r[..., inner, inner], x[..., inner, inner]).item()


def test_bad_args() -> None:
    with pytest.raises(ValueError):
        Perspective(max_angle=-1)
    with pytest.raises(ValueError):
        Perspective(fov_deg=0)
    with pytest.raises(RuntimeError):
        Perspective().inverse_warp(_img())
