"""Distortion contract: ``nn.Module``s that take an explicit generator and log what they sampled.

* ``forward(x, gen)`` maps a ``(B, 3, H, W)`` batch to a batch of the **same shape** (crops pad
  back, warps resample onto the original grid) so downstream code never re-shapes.
* Every random draw comes from ``gen``, never the global RNG, so one trial can be replayed.
* After ``forward``, ``last_params`` holds the sampled parameters for the results log.
* Differentiable wherever the physics allows; non-differentiable stages say so in their docstring.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from perceptual_media.core.types import ImageBatch, assert_image_batch


class Distortion(nn.Module):
    """Base class. Subclasses set ``name`` and implement ``_distort``."""

    name: str = "distortion"

    def __init__(self) -> None:
        super().__init__()
        self.last_params: dict[str, Any] = {}

    def forward(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:  # type: ignore[override]
        assert_image_batch(x)
        out = self._distort(x, gen)
        if out.shape != x.shape:
            raise RuntimeError(f"{self.name}: output shape {tuple(out.shape)} != input {tuple(x.shape)}")
        return out

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        raise NotImplementedError


class Identity(Distortion):
    """No-op. The reference chain and the Task-6 placeholder."""

    name = "identity"

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        self.last_params = {}
        return x


class DistortionChain(Distortion):
    """Apply ``stages`` in order; ``last_params`` is ``{stage.name: stage.last_params}``."""

    def __init__(self, stages: list[Distortion], name: str = "chain") -> None:
        super().__init__()
        self.name = name
        self.stages = nn.ModuleList(stages)

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        params: dict[str, Any] = {}
        for stage in self.stages:
            x = stage(x, gen)
            params[stage.name] = dict(stage.last_params)
        self.last_params = params
        return x
