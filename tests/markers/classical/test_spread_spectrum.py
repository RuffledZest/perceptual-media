import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.distort.basic import GaussianNoise
from perceptual_media.distort.jpeg import JPEG
from perceptual_media.distort.perspective import Perspective
from perceptual_media.harness.runner import builtin_smoke_corpus
from perceptual_media.markers.base import CapacityError, Marker, get_marker, random_payload
from perceptual_media.markers.classical.spread_spectrum import ClassicalSSMarker
from perceptual_media.metrics.decoding import auc, ber
from perceptual_media.metrics.fidelity import psnr


@pytest.fixture(scope="module")
def marker() -> ClassicalSSMarker:
    return ClassicalSSMarker(n_bits=127, key=0)


def _photo_like(seed: int = 0, size: int = 64) -> torch.Tensor:
    g = make_generator(seed)
    x = torch.rand(1, 3, size // 8, size // 8, generator=g)
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return (x + 0.08 * torch.randn(1, 3, size, size, generator=g)).clamp(0.05, 0.95)


def test_protocol_and_registry(marker: ClassicalSSMarker) -> None:
    assert isinstance(marker, Marker) and marker.n_bits == 127
    assert isinstance(get_marker("classical_ss", 127, key=3), ClassicalSSMarker)


def test_no_distortion_zero_ber_on_every_class(marker: ClassicalSSMarker) -> None:
    # Processing gain needs chips: 256 px -> 1024 blocks -> ~300 chips/bit (512 px corpus: ~1200).
    payload = random_payload(1, 127, make_generator(1))
    for item in builtin_smoke_corpus(seed=0, size=256):
        y = marker.embed(item.image, payload, strength=1.0, force=True)
        assert y.shape == item.image.shape and y.min() >= 0 and y.max() <= 1
        assert ber(marker.decode(y).llrs, payload)[0] == 0, item.image_id
    for x in (_photo_like(0, 256), _photo_like(1, 192)):
        y = marker.embed(x, payload, strength=1.0, force=True)
        assert ber(marker.decode(y).llrs, payload)[0] == 0


def test_fidelity_scales_with_strength(marker: ClassicalSSMarker) -> None:
    x = _photo_like(2)
    p = random_payload(1, 127, make_generator(2))
    ps = [psnr(marker.embed(x, p, s, force=True), x).item() for s in (0.25, 0.5, 1.0)]
    assert ps[0] > ps[1] > ps[2]
    assert ps[2] < 45  # strength 1 (one JND per coefficient) is a real perturbation


def test_pilot_score_separates_marked_from_unmarked(marker: ClassicalSSMarker) -> None:
    pos, neg = [], []
    for seed in range(6):
        x = _photo_like(10 + seed, 192)
        p = random_payload(1, 127, make_generator(seed))
        y = marker.embed(x, p, 1.0, force=True)
        pos.append(marker.decode(y).score[0].item())
        neg.append(marker.decode(x).score[0].item())
    assert auc(torch.tensor(pos), torch.tensor(neg)) == 1.0
    assert min(pos) > 3.0 and max(abs(n) for n in neg) < 3.0


def test_wrong_key_decodes_to_noise(marker: ClassicalSSMarker) -> None:
    x = _photo_like(3)
    p = random_payload(1, 127, make_generator(3))
    y = marker.embed(x, p, 1.0, force=True)
    other = ClassicalSSMarker(n_bits=127, key=99)
    assert 0.3 < ber(other.decode(y).llrs, p)[0] < 0.7
    assert abs(other.decode(y).score[0].item()) < 3.0


def test_survives_jpeg_and_noise_but_not_perspective(marker: ClassicalSSMarker) -> None:
    x = _photo_like(4, 96)
    p = random_payload(1, 127, make_generator(4))
    y = marker.embed(x, p, 1.0, force=True)
    z = GaussianNoise(0.02)(JPEG(40, 40)(y, make_generator(0)), make_generator(1))
    assert ber(marker.decode(z).llrs, p)[0] < 0.05
    warped = Perspective(8, 0)(y, make_generator(5))
    assert ber(marker.decode(warped).llrs, p)[0] > 0.3  # block grid desynchronised


def test_capacity_and_refusal(marker: ClassicalSSMarker) -> None:
    flat = torch.full((1, 3, 32, 32), 0.5)
    cap = marker.capacity(flat)
    assert cap.shape == (1,) and cap[0] > 127  # no host interference: plenty of capacity
    strict = ClassicalSSMarker(n_bits=127, key=0, capacity_strength=0.001)  # absurdly weak → refuse
    x = _photo_like(5)
    assert strict.capacity(x)[0] < 127
    with pytest.raises(CapacityError):
        strict.embed(x, random_payload(1, 127, make_generator(0)), 1.0)
    strict.embed(x, random_payload(1, 127, make_generator(0)), 1.0, force=True)  # forced


def test_batched_and_cuda_if_available(marker: ClassicalSSMarker) -> None:
    x = torch.cat([_photo_like(6, 192), _photo_like(7, 192)])
    p = random_payload(2, 127, make_generator(6))
    if torch.cuda.is_available():
        x, p = x.cuda(), p.cuda()
    y = marker.embed(x, p, 1.0, force=True)
    r = marker.decode(y)
    assert r.llrs.shape == (2, 127) and r.score.shape == (2,) and r.llrs.device == x.device
    assert torch.equal(ber(r.llrs, p), torch.zeros(2, device=x.device))


def test_payload_shape_validation(marker: ClassicalSSMarker) -> None:
    with pytest.raises(ValueError):
        marker.embed(_photo_like(8), random_payload(1, 64, make_generator(0)), 1.0, force=True)
