import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.core.types import ConventionError
from perceptual_media.markers.base import (
    DecodeResult,
    Marker,
    NullMarker,
    get_marker,
    random_payload,
    register_marker,
)


def test_null_marker_satisfies_protocol() -> None:
    m = NullMarker(n_bits=16)
    assert isinstance(m, Marker)
    assert m.n_bits == 16


def test_embed_is_identity_and_does_not_alias() -> None:
    m = NullMarker(n_bits=8)
    img = torch.rand(2, 3, 8, 8, generator=make_generator(0))
    payload = random_payload(2, 8, make_generator(1))
    out = m.embed(img, payload, strength=3.0)
    assert torch.equal(out, img)
    assert out.data_ptr() != img.data_ptr()


def test_embed_validates_inputs() -> None:
    m = NullMarker(n_bits=8)
    img = torch.rand(1, 3, 4, 4)
    with pytest.raises(ValueError):
        m.embed(img, random_payload(1, 7, make_generator(0)))  # wrong n_bits
    with pytest.raises(ConventionError):
        m.embed(img * 2, random_payload(1, 8, make_generator(0)))  # out of range


def test_decode_shapes_and_hard_bits() -> None:
    m = NullMarker(n_bits=8)
    r = m.decode(torch.rand(3, 3, 4, 4))
    assert r.llrs.shape == (3, 8) and r.score.shape == (3,)
    assert r.n_bits == 8
    hb = r.hard_bits()
    assert hb.dtype == torch.float32 and torch.all((hb == 0) | (hb == 1))
    assert torch.equal(hb, (r.llrs > 0).float())


def test_null_decode_ber_is_about_half() -> None:
    n = 1000
    m = NullMarker(n_bits=n, seed=42)
    payload = random_payload(1, n, make_generator(7))
    ber = (m.decode(torch.rand(1, 3, 4, 4)).hard_bits() != payload).float().mean().item()
    assert 0.45 <= ber <= 0.55


def test_null_decode_is_reproducible_by_seed() -> None:
    img = torch.rand(2, 3, 4, 4)
    a = NullMarker(n_bits=8, seed=5).decode(img)
    b = NullMarker(n_bits=8, seed=5).decode(img)
    c = NullMarker(n_bits=8, seed=6).decode(img)
    assert torch.equal(a.llrs, b.llrs) and torch.equal(a.score, b.score)
    assert not torch.equal(a.llrs, c.llrs)


def test_capacity_is_none() -> None:
    assert NullMarker().capacity(torch.rand(1, 3, 4, 4)) is None


def test_decode_result_validates_shapes() -> None:
    with pytest.raises(ValueError):
        DecodeResult(llrs=torch.zeros(4), score=torch.zeros(1))
    with pytest.raises(ValueError):
        DecodeResult(llrs=torch.zeros(2, 4), score=torch.zeros(3))


def test_random_payload_is_binary_and_seeded() -> None:
    p = random_payload(4, 32, make_generator(0))
    assert p.shape == (4, 32) and p.dtype == torch.float32
    assert torch.all((p == 0) | (p == 1))
    assert torch.equal(p, random_payload(4, 32, make_generator(0)))
    assert 0.3 < p.mean().item() < 0.7


def test_registry() -> None:
    m = get_marker("null", n_bits=12, seed=3)
    assert isinstance(m, NullMarker) and m.n_bits == 12
    with pytest.raises(KeyError, match="unknown marker 'nope'"):
        get_marker("nope", n_bits=1)

    class Custom(NullMarker):
        pass

    register_marker("custom", Custom)
    assert isinstance(get_marker("custom", n_bits=4), Custom)
