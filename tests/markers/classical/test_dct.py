import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.markers.classical.dct import MID_BAND, band_mask, blocks, from_luma_dct, luma_dct, unblocks


def test_luma_dct_roundtrip_multiple_of_8() -> None:
    x = torch.rand(2, 3, 32, 40, generator=make_generator(0))
    y, cb, cr, pad = luma_dct(x)
    assert y.shape == (2, 32, 40) and pad == (0, 0)
    back = from_luma_dct(y, cb, cr, pad)
    assert back.shape == x.shape and (back - x).abs().max() < 1e-4


def test_luma_dct_roundtrip_with_padding() -> None:
    x = torch.rand(1, 3, 21, 13, generator=make_generator(1))
    y, cb, cr, pad = luma_dct(x)
    assert y.shape == (1, 24, 16) and pad == (3, 3)
    back = from_luma_dct(y, cb, cr, pad)
    assert back.shape == x.shape and (back - x).abs().max() < 1e-4


def test_dc_scale_is_0_to_2040() -> None:
    white = torch.ones(1, 3, 8, 8)
    y, *_ = luma_dct(white)
    assert abs(y[0, 0, 0].item() - 2040.0) < 1e-2
    grey = torch.full((1, 3, 8, 8), 0.5)
    y, *_ = luma_dct(grey)
    assert abs(y[0, 0, 0].item() - 1020.0) < 1e-2


def test_blocks_view_roundtrip_and_indexing() -> None:
    c = torch.arange(2 * 16 * 24, dtype=torch.float32).reshape(2, 16, 24)
    b = blocks(c)
    assert b.shape == (2, 2, 3, 8, 8)
    assert b[1, 1, 2, 3, 4] == c[1, 8 + 3, 16 + 4]
    assert torch.equal(unblocks(b), c)


def test_band_mask() -> None:
    m = band_mask(3, 8)
    assert m.shape == (8, 8) and m.dtype == torch.bool
    assert not m[0, 0] and not m[1, 1] and m[1, 2] and m[4, 4] and not m[7, 7]
    assert m.sum() == 37 and torch.equal(m, MID_BAND)
    assert band_mask(0, 14).all()
