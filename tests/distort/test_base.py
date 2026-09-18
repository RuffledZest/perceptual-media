import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.core.types import ConventionError
from perceptual_media.distort.base import Distortion, DistortionChain, Identity
from perceptual_media.distort.presets import (
    available_presets,
    build_chain,
    register_preset,
)


class _Scale(Distortion):
    """Test stage: multiplies by a sampled factor and logs it."""

    name = "scale"

    def _distort(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        f = float(torch.rand(1, generator=gen)) * 0.5
        self.last_params = {"factor": f}
        return x * f


def test_identity_and_shape_check() -> None:
    x = torch.rand(2, 3, 8, 8)
    assert torch.equal(Identity()(x, make_generator(0)), x)
    with pytest.raises(ConventionError):
        Identity()(x * 2, make_generator(0))


def test_chain_composes_and_logs_params() -> None:
    chain = DistortionChain([_Scale(), Identity()], name="test")
    x = torch.ones(1, 3, 4, 4)
    y = chain(x, make_generator(1))
    f = chain.last_params["scale"]["factor"]
    assert 0 <= f < 0.5 and torch.allclose(y, x * f)
    assert chain.last_params == {"scale": {"factor": f}, "identity": {}}


def test_chain_deterministic_given_generator() -> None:
    chain = DistortionChain([_Scale()])
    x = torch.ones(1, 3, 4, 4)
    a = chain(x, make_generator(5)).clone()
    b = chain(x, make_generator(5))
    assert torch.equal(a, b)


def test_presets_registry() -> None:
    assert "identity" in available_presets()
    c = build_chain("identity", 0.0)
    assert c.name == "identity"
    with pytest.raises(KeyError, match="unknown distortion preset"):
        build_chain("nope")
    with pytest.raises(ValueError):
        build_chain("identity", 1.5)
    register_preset("scale_only", lambda s: DistortionChain([_Scale()]))
    assert build_chain("scale_only", 0.3).name == "scale_only"
