import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.markers.base import random_payload
from perceptual_media.markers.classical.bch import BCHCode
from perceptual_media.metrics.decoding import payload_recovered


@pytest.fixture(scope="module")
def code() -> BCHCode:
    return BCHCode(127, 64)


def _flip(c: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    r = c.clone()
    for b in range(c.shape[0]):
        idx = torch.randperm(c.shape[1], generator=make_generator(seed + b))[:n]
        r[b, idx] = 1 - r[b, idx]
    return r


def test_parameters_and_systematic_encoding(code: BCHCode) -> None:
    assert (code.n, code.k, code.t) == (127, 64, 10)
    m = random_payload(3, 64, make_generator(0))
    c = code.encode(m)
    assert c.shape == (3, 127) and c.dtype == torch.float32 and torch.all((c == 0) | (c == 1))
    assert torch.equal(c[:, :64], m)


def test_corrects_up_to_t_errors(code: BCHCode) -> None:
    m = random_payload(4, 64, make_generator(1))
    c = code.encode(m)
    for n_err in (0, 1, 5, 10):
        msg, ok = code.decode_hard(_flip(c, n_err, 100 + n_err))
        assert ok.all() and torch.equal(msg, m), n_err


def test_fails_beyond_t_errors(code: BCHCode) -> None:
    m = random_payload(4, 64, make_generator(2))
    c = code.encode(m)
    msg, ok = code.decode_hard(_flip(c, 25, 7))
    assert not ok.any()
    assert not (msg == m).all(dim=1).any()


def test_decode_soft_slices_llrs(code: BCHCode) -> None:
    m = random_payload(2, 64, make_generator(3))
    c = code.encode(m)
    llrs = (c * 2 - 1) * torch.rand(2, 127, generator=make_generator(4)) * 5  # random confidences, right signs
    llrs[0, :8] *= -1  # 8 flips in item 0, within t
    msg, ok = code.decode_soft(llrs)
    assert ok.all() and torch.equal(msg, m)


def test_payload_recovered_with_ecc(code: BCHCode) -> None:
    m = random_payload(3, 64, make_generator(5))
    c = code.encode(m)
    llrs = c * 2 - 1
    llrs[1, :10] *= -1  # t errors: recovered
    llrs[2, :30] *= -1  # far beyond t: not recovered
    rec = payload_recovered(llrs, m, ecc=code)
    assert rec.tolist() == [True, True, False]


def test_shape_errors(code: BCHCode) -> None:
    with pytest.raises(ValueError):
        code.encode(torch.zeros(1, 63))
    with pytest.raises(ValueError):
        code.decode_hard(torch.zeros(1, 126))
