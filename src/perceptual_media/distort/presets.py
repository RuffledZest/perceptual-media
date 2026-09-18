"""Named distortion chains, as referenced by ``DistortionConfig.preset``.

``severity`` in ``[0, 1]`` scales every stage's sampling range. Severity 0 is the identity
(every stage short-circuits); severity 1 covers at least StegaStamp's training ranges
(``docs/research/stegastamp_pimog_notes.md``) so a StegaStamp/Video Seal model evaluated here
sees nothing out of distribution by construction:

======================  ==============================  =======================
stage                   severity 1 range                StegaStamp default
======================  ==============================  =======================
perspective (pose)      tilt/yaw ±60°, roll ±15°        corner jitter ±10 % W
illumination            gain 0.3, contrast ±0.5,        contrast [0.5, 1.5],
                        brightness ±0.3, cast ±0.1,     brightness ±0.3,
                        desaturate ≤1, gamma ±0.3       hue ±0.1, sat ≤1
defocus / motion blur   σ ≤ 3 px / length ≤ 9 px        σ ∈ [1, 3] (7×7)
moiré (screen only)     strength = severity             —
resize                  scale ≥ 0.4                     —
JPEG                    quality ≥ 25                    quality ≥ 25
noise                   σ ≤ 0.02                        σ ≤ 0.02
======================  ==============================  =======================

Order: perspective → illumination → [moiré] → defocus → motion → resize → JPEG → noise.
``print_camera`` is the primary physical preset (brief; literature review), ``screen_camera``
adds moiré.
"""

from __future__ import annotations

from collections.abc import Callable

from perceptual_media.distort.base import Distortion, DistortionChain, Identity
from perceptual_media.distort.basic import GaussianNoise, Resize
from perceptual_media.distort.blur import DefocusBlur, MotionBlur
from perceptual_media.distort.illumination import Illumination
from perceptual_media.distort.jpeg import JPEG
from perceptual_media.distort.moire import Moire
from perceptual_media.distort.perspective import Perspective


def _camera_stages(severity: float, *, moire: bool, differentiable_jpeg: bool = False) -> list[Distortion]:
    s = severity
    stages: list[Distortion] = [
        Perspective(max_angle=60 * s, max_roll=15 * s, mode="pose"),
        Illumination(gain=0.3 * s, contrast=0.5 * s, brightness=0.3 * s, cast=0.1 * s, saturation=1.0 * s, gamma=0.3 * s),
    ]
    if moire:
        stages.append(Moire(scale_min=0.6, scale_max=1.4, psf_sigma=0.3, max_rotation_deg=3 * s, strength=s))
    stages += [
        DefocusBlur(0.0, 3.0 * s),
        MotionBlur(0.0, 9.0 * s),
        Resize(1.0 - 0.6 * s, 1.0),
        JPEG(100 - 75 * s, 100, differentiable=differentiable_jpeg),
        GaussianNoise(0.02 * s),
    ]
    return stages


_PRESETS: dict[str, Callable[[float], DistortionChain]] = {
    "identity": lambda severity: DistortionChain([Identity()], name="identity"),
    "print_camera": lambda severity: DistortionChain(_camera_stages(severity, moire=False), name="print_camera"),
    "screen_camera": lambda severity: DistortionChain(_camera_stages(severity, moire=True), name="screen_camera"),
    # differentiable variants for training (surrogate JPEG)
    "print_camera_diff": lambda severity: DistortionChain(_camera_stages(severity, moire=False, differentiable_jpeg=True), name="print_camera_diff"),
    "screen_camera_diff": lambda severity: DistortionChain(_camera_stages(severity, moire=True, differentiable_jpeg=True), name="screen_camera_diff"),
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
