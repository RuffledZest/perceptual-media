"""Named distortion chains, as referenced by ``DistortionConfig.preset``.

``severity`` in ``[0, 1]`` scales every stage's sampling range; 0 is (close to) identity and 1
covers at least the StegaStamp training ranges (see ``docs/research/stegastamp_pimog_notes.md``).
The ``print_camera`` and ``screen_camera`` presets are added in Task 12.
"""

from __future__ import annotations

from collections.abc import Callable

from perceptual_media.distort.base import DistortionChain, Identity

_PRESETS: dict[str, Callable[[float], DistortionChain]] = {
    "identity": lambda severity: DistortionChain([Identity()], name="identity"),
}


def register_preset(name: str, factory: Callable[[float], DistortionChain]) -> None:
    """Register ``factory(severity) -> DistortionChain`` under ``name``."""
    _PRESETS[name] = factory


def build_chain(preset: str, severity: float = 0.0) -> DistortionChain:
    """Instantiate the named preset at ``severity``."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    try:
        factory = _PRESETS[preset]
    except KeyError:
        raise KeyError(f"unknown distortion preset {preset!r}; available: {sorted(_PRESETS)}") from None
    chain = factory(severity)
    chain.name = preset
    return chain


def available_presets() -> list[str]:
    return sorted(_PRESETS)
