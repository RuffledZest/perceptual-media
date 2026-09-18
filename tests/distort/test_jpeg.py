import io

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.fft import dctn

from perceptual_media.core.io import from_uint8, to_uint8
from perceptual_media.core.seed import make_generator
from perceptual_media.distort.jpeg import (
    JPEG,
    block_dct,
    block_idct,
    jpeg_differentiable,
    jpeg_pil,
    quant_tables,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)
from perceptual_media.metrics.fidelity import psnr


def _natural(seed: int = 0, size: int = 64) -> torch.Tensor:
    """Smooth-ish random image (low-pass noise) so JPEG behaves like on photos."""
    g = make_generator(seed)
    x = torch.rand(1, 3, size // 8, size // 8, generator=g)
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return (x + 0.05 * torch.randn(1, 3, size, size, generator=g)).clamp(0, 1)


def test_block_dct_matches_scipy_and_roundtrips() -> None:
    x = torch.rand(1, 16, 16, generator=make_generator(0))
    c = block_dct(x)
    ref = dctn(x[0, :8, :8].numpy(), norm="ortho")
    assert np.allclose(c[0, :8, :8].numpy(), ref, atol=1e-5)
    assert (block_idct(c) - x).abs().max() < 1e-5


def test_ycbcr_roundtrip() -> None:
    x = torch.rand(2, 3, 8, 8, generator=make_generator(1))
    assert (ycbcr_to_rgb(rgb_to_ycbcr(x)) - x).abs().max() < 1e-5


def test_quant_tables_libjpeg_rule() -> None:
    l50, c50 = quant_tables(50)
    assert l50[0, 0] == 16 and c50[0, 0] == 17  # base tables at Q50
    l100, _ = quant_tables(100)
    assert torch.all(l100 == 1)
    l25, _ = quant_tables(25)
    assert l25[0, 0] == 32  # scale 200 → doubled


def test_pil_path_matches_pil_exactly() -> None:
    x = _natural()
    out = jpeg_pil(x, 75)
    buf = io.BytesIO()
    Image.fromarray(to_uint8(x)[0]).save(buf, format="JPEG", quality=75, subsampling="4:2:0")
    buf.seek(0)
    ref = from_uint8(np.asarray(Image.open(buf).convert("RGB")))
    assert torch.equal(out, ref)


@pytest.mark.parametrize("mode", ["ste", "round_only_at_0", "diff_round"])
def test_differentiable_close_to_real_at_q75(mode: str) -> None:
    x = _natural(size=64)
    real = jpeg_pil(x, 75)
    approx = jpeg_differentiable(x, 75, mode)  # type: ignore[arg-type]
    assert psnr(approx, real).item() >= 35.0


def test_differentiable_gradient_and_nonmultiple_of_16() -> None:
    x = _natural(size=40).requires_grad_()  # 40 is not a multiple of 16
    y = jpeg_differentiable(x, 50, "ste")
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_q100_is_near_lossless_except_for_chroma_subsampling() -> None:
    x = _natural()
    # 4:4:4 at Q100: only rounding error remains
    assert psnr(jpeg_differentiable(x, 100, "ste", subsample=False), x).item() > 45
    # 4:2:0 loses per-pixel chroma noise in this synthetic image; PIL loses the same amount
    d, r = jpeg_differentiable(x, 100, "ste"), jpeg_pil(x, 100)
    assert abs(psnr(d, x).item() - psnr(r, x).item()) < 1.0
    assert psnr(d, r).item() > 40


def test_jpeg_stage_logs_and_is_deterministic() -> None:
    x = _natural()
    for diff in (False, True):
        st = JPEG(25, 95, differentiable=diff)
        a = st(x, make_generator(3)).clone()
        q = st.last_params["quality"]
        assert 25 <= q <= 95 and st.last_params["differentiable"] is diff
        assert torch.equal(a, st(x, make_generator(3)))
    with pytest.raises(ValueError):
        JPEG(0, 50)
