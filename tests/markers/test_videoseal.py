import pytest
import torch

pytest.importorskip("videoseal")

from perceptual_media.core.seed import make_generator  # noqa: E402
from perceptual_media.markers.base import (  # noqa: E402
    Marker,
    get_marker,
    random_payload,
)
from perceptual_media.markers.videoseal.adapter import VideoSealMarker  # noqa: E402
from perceptual_media.metrics.decoding import ber  # noqa: E402
from perceptual_media.metrics.fidelity import psnr  # noqa: E402


@pytest.fixture(scope="module")
def marker() -> VideoSealMarker:
    m = VideoSealMarker(256)
    try:
        _ = m.model  # downloads the checkpoint on first use
    except Exception as e:  # network / cache problems → skip, not fail
        pytest.skip(f"videoseal model unavailable: {e}")
    return m


def _photo_like(seed: int = 0, size: int = 256) -> torch.Tensor:
    g = make_generator(seed)
    x = torch.rand(1, 3, size // 16, size // 16, generator=g)
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return (x + 0.05 * torch.randn(1, 3, size, size, generator=g)).clamp(0.05, 0.95)


def test_protocol_and_registry(marker: VideoSealMarker) -> None:
    assert isinstance(marker, Marker) and marker.n_bits == 256
    assert isinstance(get_marker("videoseal", 256), VideoSealMarker)


def test_embed_decode_roundtrip(marker: VideoSealMarker) -> None:
    x = _photo_like(1)
    p = random_payload(1, 256, make_generator(1))
    y = marker.embed(x, p, 1.0)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1 and y.device == x.device
    assert psnr(y, x).item() > 38
    r = marker.decode(y)
    assert r.llrs.shape == (1, 256) and r.score.shape == (1,)
    assert ber(r.llrs, p)[0] < 0.05


def test_strength_scales_residual(marker: VideoSealMarker) -> None:
    x = _photo_like(2)
    p = random_payload(1, 256, make_generator(2))
    assert torch.allclose(marker.embed(x, p, 0.0), x, atol=1e-6)
    y1, y2 = marker.embed(x, p, 1.0), marker.embed(x, p, 2.0)
    assert psnr(y2, x).item() < psnr(y1, x).item()


def test_presence_score_separates(marker: VideoSealMarker) -> None:
    x = _photo_like(3)
    p = random_payload(1, 256, make_generator(3))
    assert marker.decode(marker.embed(x, p, 1.0)).score[0] > marker.decode(x).score[0]


def test_capacity_none_and_nbits_check(marker: VideoSealMarker) -> None:
    assert marker.capacity(_photo_like(4)) is None
    with pytest.raises(ValueError, match="carries 256 bits"):
        _ = VideoSealMarker(64).model
