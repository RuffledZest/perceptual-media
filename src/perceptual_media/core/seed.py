"""Determinism helpers.

Two rules for the whole project:

1. Call ``seed_everything(seed)`` once at the start of an experiment. It seeds Python's
   ``random``, NumPy's global RNG, and torch (CPU + every CUDA device).
2. Anything that draws random numbers *inside* the pipeline (distortions, payload sampling,
   the null marker) takes an explicit ``torch.Generator`` from ``make_generator``, so a single
   trial can be reproduced without replaying the whole run.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU and all CUDA devices) with ``seed``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """Return a fresh ``torch.Generator`` on ``device`` seeded with ``seed``."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen
