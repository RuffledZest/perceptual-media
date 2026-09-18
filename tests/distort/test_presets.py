import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.distort.presets import available_presets, build_chain
from perceptual_media.metrics.fidelity import psnr


def _img() -> torch.Tensor:
    g = make_generator(0)
    x = torch.rand(1, 3, 16, 16, generator=g)
    return torch.nn.functional.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False).clamp(0, 1)


def test_presets_listed() -> None:
    assert {"identity", "print_camera", "screen_camera", "print_camera_diff", "screen_camera_diff"} <= set(available_presets())


@pytest.mark.parametrize("preset", ["print_camera", "screen_camera"])
def test_severity_zero_is_exact_identity(preset: str) -> None:
    x = _img()
    chain = build_chain(preset, 0.0)
    y = chain(x, make_generator(0))
    assert torch.equal(y, x)


@pytest.mark.parametrize("preset", ["print_camera", "screen_camera"])
def test_severity_scales_damage(preset: str) -> None:
    x = _img()

    def mean_psnr(sev: float) -> float:
        return sum(psnr(build_chain(preset, sev)(x, make_generator(k)), x).item() for k in range(8)) / 8

    p = [mean_psnr(s) for s in (0.1, 0.5, 1.0)]
    assert p[0] > p[1] > p[2]  # more severity → more damage, on average
    assert p[2] < 25  # severity 1 is genuinely harsh


def test_stage_order_and_params_logged() -> None:
    chain = build_chain("screen_camera", 0.5)
    chain(_img(), make_generator(2))
    assert list(chain.last_params) == ["perspective", "illumination", "moire", "defocus", "motion", "resize", "jpeg", "noise"]
    assert "quality" in chain.last_params["jpeg"] and 62.5 <= chain.last_params["jpeg"]["quality"] <= 100
    assert "capture_scale" in chain.last_params["moire"]
    print_chain = build_chain("print_camera", 0.5)
    print_chain(_img(), make_generator(2))
    assert "moire" not in print_chain.last_params


def test_screen_camera_at_severity_one_covers_stegastamp_ranges() -> None:
    chain = build_chain("screen_camera", 1.0)
    stages = {s.name: s for s in chain.stages}
    assert stages["jpeg"].q_min <= 25
    assert stages["noise"].sigma_max >= 0.02
    assert stages["defocus"].sigma_max >= 3.0
    assert stages["illumination"].contrast >= 0.5 and stages["illumination"].brightness >= 0.3 and stages["illumination"].cast >= 0.1
    assert stages["perspective"].max_angle >= 45


def test_diff_variant_has_gradient_end_to_end() -> None:
    x = _img().requires_grad_()
    y = build_chain("print_camera_diff", 0.5)(x, make_generator(3))
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_non_diff_variant_runs_on_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    x = _img().cuda()
    chain = build_chain("screen_camera", 0.5).cuda()
    y = chain(x, make_generator(0, "cuda"))
    assert y.device.type == "cuda" and y.shape == x.shape


def test_fixed_single_axis_presets_are_deterministic_in_severity() -> None:
    x = _img()
    for preset in ("perspective_fixed", "defocus_fixed", "jpeg_fixed", "resize_fixed"):
        a = build_chain(preset, 0.5)(x, make_generator(0))
        b = build_chain(preset, 0.5)(x, make_generator(99))  # different generator, same result
        assert torch.equal(a, b), preset
        assert torch.equal(build_chain(preset, 0.0)(x, make_generator(0)), x), preset
    p = build_chain("perspective_fixed", 0.5)
    p(x, make_generator(0))
    assert p.last_params["perspective"]["tilt_deg"] == 30.0 and p.last_params["perspective"]["yaw_deg"] == 0.0
