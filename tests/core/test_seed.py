import random

import numpy as np
import torch

from perceptual_media.core.seed import make_generator, seed_everything


def _draw() -> tuple[torch.Tensor, np.ndarray, float]:
    return torch.rand(4), np.random.rand(4), random.random()


def test_seed_everything_makes_all_three_rngs_reproducible() -> None:
    seed_everything(123)
    t1, n1, r1 = _draw()
    seed_everything(123)
    t2, n2, r2 = _draw()
    assert torch.equal(t1, t2)
    assert np.array_equal(n1, n2)
    assert r1 == r2


def test_different_seeds_differ() -> None:
    seed_everything(1)
    t1, *_ = _draw()
    seed_everything(2)
    t2, *_ = _draw()
    assert not torch.equal(t1, t2)


def test_make_generator_is_independent_of_global_state() -> None:
    g1 = make_generator(7)
    a = torch.rand(8, generator=g1)
    torch.manual_seed(999)  # perturb global RNG; must not affect an explicit generator
    g2 = make_generator(7)
    b = torch.rand(8, generator=g2)
    assert torch.equal(a, b)
