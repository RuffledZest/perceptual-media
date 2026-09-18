import pytest
import torch

from perceptual_media.core.config import EccConfig
from perceptual_media.core.seed import make_generator
from perceptual_media.harness.ecc import ChannelCode, build_ecc
from perceptual_media.markers.base import random_payload


class _Parity:
    """Toy inner code: n = k + 1, last bit = XOR of message; decode_soft checks parity."""

    n, k = 5, 4

    def encode(self, message: torch.Tensor) -> torch.Tensor:
        par = message.sum(dim=1, keepdim=True) % 2
        return torch.cat([message, par], dim=1)

    def decode_soft(self, llrs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hard = (llrs > 0).float()
        ok = (hard[:, :4].sum(dim=1) % 2) == hard[:, 4]
        return hard[:, :4], ok


def test_channel_code_pads_repeats_and_combines() -> None:
    cc = ChannelCode(_Parity(), marker_bits=12, repeat=2)
    m = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    c = cc.encode(m)
    assert c.shape == (1, 12)
    assert torch.equal(c[:, :5], c[:, 5:10]) and torch.all(c[:, 10:] == 0)
    llrs = c * 2 - 1
    llrs[0, 1] = -3.0  # corrupt bit 1 in copy 1 (should be 0 → still correct sign actually); flip bit 0 instead
    llrs[0, 0] = -0.5  # weak wrong vote in copy 1, strong right vote in copy 2 (+1) → combined +0.5
    msg, ok = cc.decode_soft(llrs)
    assert ok.all() and torch.equal(msg, m)


def test_channel_code_rejects_overflow_and_bad_shapes() -> None:
    with pytest.raises(ValueError):
        ChannelCode(_Parity(), marker_bits=9, repeat=2)
    cc = ChannelCode(_Parity(), marker_bits=8, repeat=1)
    with pytest.raises(ValueError):
        cc.decode_soft(torch.zeros(1, 7))


def test_build_ecc_bch_in_256_channel_with_repeat_gains() -> None:
    cc = build_ecc(EccConfig("bch", 127, 64, repeat=2), marker_bits=256)
    assert cc is not None and cc.k == 64 and cc.marker_bits == 256
    m = random_payload(1, 64, make_generator(0))
    c = cc.encode(m)
    assert c.shape == (1, 256) and torch.equal(c[:, :127], c[:, 127:254]) and torch.all(c[:, 254:] == 0)
    llrs = (c * 2 - 1) * 2.0
    # 16 errors in copy 1 only (> t = 10 for a single copy); copy 2 clean → combined decodes
    llrs[0, :16] *= -1
    msg, ok = cc.decode_soft(llrs)
    assert ok.all() and torch.equal(msg, m)
    single = build_ecc(EccConfig("bch", 127, 64, repeat=1), marker_bits=127)
    msg1, ok1 = single.decode_soft(llrs[:, :127])
    assert not ok1.any()  # same 16 errors defeat the single copy


def test_build_ecc_none_and_overflow() -> None:
    assert build_ecc(None, 64) is None
    with pytest.raises(ValueError, match="exceeds marker.n_bits"):
        build_ecc(EccConfig("bch", 127, 64), marker_bits=64)
