import math

import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.markers.classical.dct import blocks, luma_dct
from perceptual_media.markers.classical.watson import (
    A_T,
    DC_REFERENCE,
    PRINT_300DPI_40CM,
    REFERENCE_PPD,
    SCREEN_96DPI_60CM,
    W_CONTRAST,
    WATSON_TABLE,
    ViewingCondition,
    contrast_masking,
    frequency_sensitivity,
    luminance_masking,
    watson_slack,
)


def test_viewing_condition_pixels_per_degree() -> None:
    assert SCREEN_96DPI_60CM.pixels_per_degree == pytest.approx(39.58, abs=0.05)
    assert PRINT_300DPI_40CM.pixels_per_degree == pytest.approx(82.46, abs=0.05)
    ref = ViewingCondition(dpi=32 / (math.tan(math.radians(1)) / 2.54), distance_cm=1.0)
    assert ref.pixels_per_degree == pytest.approx(32.0, abs=1e-6)


def test_table_is_symmetric_and_exact_at_reference() -> None:
    assert torch.equal(WATSON_TABLE, WATSON_TABLE.T)
    ref = ViewingCondition(dpi=REFERENCE_PPD * 2.54 / math.tan(math.radians(1)), distance_cm=1.0)
    assert torch.allclose(frequency_sensitivity(ref), WATSON_TABLE)


def test_finer_pixels_raise_high_frequency_thresholds() -> None:
    t_screen = frequency_sensitivity(SCREEN_96DPI_60CM)
    t_print = frequency_sensitivity(PRINT_300DPI_40CM)
    assert t_screen[0, 0] == WATSON_TABLE[0, 0] and t_print[0, 0] == WATSON_TABLE[0, 0]  # DC fixed
    assert t_screen[7, 7] > WATSON_TABLE[7, 7]
    assert t_print[7, 7] > t_screen[7, 7]
    assert (t_screen > 0).all() and torch.isfinite(t_screen).all()
    assert torch.allclose(t_screen, t_screen.T, atol=1e-5)


def test_hand_computed_block_all_three_stages() -> None:
    # One 8x8 block: DC = 1536 (mean 192, bright), one AC coefficient (1,2) = 40, rest 0.
    coef = torch.zeros(1, 8, 8)
    coef[0, 0, 0] = 1536.0
    coef[0, 1, 2] = 40.0
    ref = ViewingCondition(dpi=REFERENCE_PPD * 2.54 / math.tan(math.radians(1)), distance_cm=1.0)
    s = watson_slack(coef, viewing=ref)

    lum = (1536.0 / DC_REFERENCE) ** A_T  # 1.5^0.649
    # (a) AC coefficient with |C| = 0: slack = t_L = t * lum
    assert s[0, 3, 4].item() == pytest.approx(WATSON_TABLE[3, 4].item() * lum, rel=1e-5)
    # (b) the busy coefficient (1,2): max(t_L, |C|^w t_L^(1-w)) with |C| = 40 > t_L
    t_l = WATSON_TABLE[1, 2].item() * lum
    expected = max(t_l, 40.0**W_CONTRAST * t_l ** (1 - W_CONTRAST))
    assert expected > t_l  # masking active
    assert s[0, 1, 2].item() == pytest.approx(expected, rel=1e-5)
    # (c) DC: contrast-masking exponent 0 → slack = t_L only, regardless of |C00| = 1536
    assert s[0, 0, 0].item() == pytest.approx(WATSON_TABLE[0, 0].item() * lum, rel=1e-5)


def test_bright_blocks_get_more_slack_than_dark() -> None:
    bright = torch.full((1, 3, 16, 16), 0.85)
    dark = torch.full((1, 3, 16, 16), 0.15)
    sb = watson_slack(luma_dct(bright)[0])
    sd = watson_slack(luma_dct(dark)[0])
    assert (sb > sd).all()


def test_textured_blocks_get_more_slack_than_flat() -> None:
    flat = torch.full((1, 3, 16, 16), 0.5)
    tex = (0.5 + 0.3 * torch.randn(1, 3, 16, 16, generator=make_generator(0))).clamp(0, 1)
    sf = watson_slack(luma_dct(flat)[0])
    st = watson_slack(luma_dct(tex)[0])
    # compare AC slack summed per block
    bf, bt = blocks(sf), blocks(st)
    ac = torch.ones(8, 8, dtype=torch.bool)
    ac[0, 0] = False
    assert (bt[..., ac].sum(-1) > bf[..., ac].sum(-1)).all()


def test_dark_floor_keeps_slack_positive() -> None:
    black = torch.zeros(1, 3, 8, 8)
    s = watson_slack(luma_dct(black)[0])
    assert (s > 0).all() and torch.isfinite(s).all()


def test_luminance_and_contrast_helpers_shapes() -> None:
    t = frequency_sensitivity()
    dc = torch.full((2, 3, 4), DC_REFERENCE)
    t_l = luminance_masking(t, dc)
    assert t_l.shape == (2, 3, 4, 8, 8) and torch.allclose(t_l[0, 0, 0], t)
    coef = torch.zeros(2, 3, 4, 8, 8)
    assert torch.allclose(contrast_masking(t_l, coef), t_l)


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        watson_slack(torch.zeros(1, 12, 16))
    with pytest.raises(ValueError):
        watson_slack(torch.zeros(16, 16))
