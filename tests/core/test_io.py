from pathlib import Path

import numpy as np
import pytest
import torch

from perceptual_media.core.io import from_uint8, load_image, save_image, to_uint8
from perceptual_media.core.seed import make_generator
from perceptual_media.core.types import ConventionError, assert_image_batch, assert_payload


def test_from_uint8_shape_dtype_range() -> None:
    arr = np.zeros((5, 7, 3), dtype=np.uint8)
    arr[..., 0] = 255
    t = from_uint8(arr)
    assert t.shape == (1, 3, 5, 7)
    assert t.dtype == torch.float32
    assert t[0, 0].min() == 1.0 and t[0, 1].max() == 0.0


def test_uint8_roundtrip_is_exact() -> None:
    arr = np.random.default_rng(0).integers(0, 256, size=(2, 6, 4, 3), dtype=np.uint8)
    assert np.array_equal(to_uint8(from_uint8(arr)), arr)


def test_save_load_roundtrip_within_one_level(tmp_path: Path) -> None:
    x = torch.rand(1, 3, 16, 24, generator=make_generator(0))
    p = tmp_path / "sub" / "img.png"
    save_image(x, p)
    y = load_image(p)
    assert y.shape == x.shape
    assert (y - x).abs().max().item() <= 1 / 255 + 1e-6


def test_save_accepts_chw(tmp_path: Path) -> None:
    x = torch.rand(3, 8, 8, generator=make_generator(1))
    save_image(x, tmp_path / "chw.png")
    assert load_image(tmp_path / "chw.png").shape == (1, 3, 8, 8)


def test_save_rejects_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_image(torch.zeros(2, 3, 4, 4), tmp_path / "b.png")


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(3, 4, 4),  # missing batch dim
        torch.zeros(1, 1, 4, 4),  # wrong channel count
        torch.zeros(1, 3, 4, 4, dtype=torch.float64),  # wrong dtype
        torch.full((1, 3, 4, 4), 1.5),  # out of range
        torch.full((1, 3, 4, 4), -0.1),  # out of range
    ],
)
def test_assert_image_batch_rejects(bad: torch.Tensor) -> None:
    with pytest.raises(ConventionError):
        assert_image_batch(bad)


def test_assert_image_batch_accepts_valid() -> None:
    assert_image_batch(torch.rand(2, 3, 4, 4))


def test_assert_payload() -> None:
    assert_payload(torch.tensor([[0.0, 1.0, 1.0]]))
    with pytest.raises(ConventionError):
        assert_payload(torch.tensor([[0.0, 0.5]]))
    with pytest.raises(ConventionError):
        assert_payload(torch.tensor([0.0, 1.0]))
