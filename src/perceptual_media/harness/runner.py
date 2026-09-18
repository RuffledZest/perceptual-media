"""The experiment loop: image × strength × distortion → embed, distort, decode, one row each.

For every trial the runner also runs the **unmarked control**: the original image through the
*identical* distortion (same sampled parameters), decoded with the same marker. Control rows are
what make the false-positive accounting honest (brief §6).

Determinism: a trial seed is derived from ``(cfg.seed, image_id, strength, preset, severity)``,
so a single row can be reproduced without replaying the run.
"""

from __future__ import annotations

import math
import time
import warnings
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from perceptual_media.core.config import CorpusConfig, EccConfig, ExperimentConfig
from perceptual_media.core.seed import make_generator, seed_everything
from perceptual_media.core.types import ImageBatch
from perceptual_media.corpus.manifest import iter_images
from perceptual_media.distort.presets import build_chain
from perceptual_media.harness.results import (
    ResultRow,
    ResultWriter,
    make_run_dir,
    to_json,
    write_summary,
)
from perceptual_media.markers.base import Marker, get_marker, random_payload
from perceptual_media.metrics.decoding import ber, payload_recovered
from perceptual_media.metrics.fidelity import fidelity


@dataclass
class CorpusImage:
    image_id: str
    image_class: str
    image: ImageBatch  # (1, 3, H, W)


def resolve_device(device: str) -> torch.device:
    """``auto`` → CUDA if available else CPU; anything else is passed to ``torch.device``."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def trial_seed(base_seed: int, image_id: str, strength: float, preset: str, severity: float) -> int:
    """Stable 32-bit seed for one trial."""
    key = f"{base_seed}|{image_id}|{strength!r}|{preset}|{severity!r}"
    return zlib.crc32(key.encode("utf-8"))


def builtin_smoke_corpus(seed: int = 0, size: int = 64, n_per_class: int = 2) -> Iterator[CorpusImage]:
    """Tiny in-memory corpus so the harness runs before Task 8's manifest exists.

    Classes mirror the brief's hard cases: ``flat``, ``gradient``, ``textured``.
    """
    gen = make_generator(seed)
    for i in range(n_per_class):
        level = torch.rand(3, 1, 1, generator=gen)
        yield CorpusImage(f"flat{i:02d}", "flat", level.expand(3, size, size)[None].clone())
    for i in range(n_per_class):
        ramp = torch.linspace(0, 1, size)
        img = torch.stack([ramp[None, :].expand(size, size), ramp[:, None].expand(size, size), 0.5 * torch.ones(size, size)])
        yield CorpusImage(f"gradient{i:02d}", "gradient", img[None].clone())
    for i in range(n_per_class):
        yield CorpusImage(f"textured{i:02d}", "textured", torch.rand(1, 3, size, size, generator=gen))


def load_corpus(cfg: CorpusConfig, seed: int = 0) -> Iterator[CorpusImage]:
    """Yield images per ``CorpusConfig`` from the manifest (``pm-corpus build``).

    If the manifest does not exist, falls back to the built-in smoke corpus with a warning.
    """
    if Path(cfg.manifest).exists():
        for row, img in iter_images(cfg.manifest, cfg.classes, cfg.limit, cfg.per_class):
            yield CorpusImage(row.image_id, row.image_class, img)
        return
    warnings.warn(f"manifest {cfg.manifest!r} not found; using built-in smoke corpus", stacklevel=2)
    images: Iterable[CorpusImage] = builtin_smoke_corpus(seed)
    if cfg.classes is not None:
        images = (im for im in images if im.image_class in cfg.classes)
    if cfg.limit is not None:
        images = (im for _, im in zip(range(cfg.limit), images, strict=False))  # truncates on purpose
    yield from images


def build_ecc(cfg: EccConfig | None, marker_bits: int) -> Any:
    """Instantiate the configured ECC (or ``None``) and check it matches the marker's channel width."""
    if cfg is None:
        return None
    if cfg.name != "bch":
        raise KeyError(f"unknown ecc {cfg.name!r}; available: ['bch']")
    from perceptual_media.markers.classical.bch import BCHCode  # numba import, deferred

    ecc = BCHCode(cfg.n, cfg.k)
    if ecc.n != marker_bits:
        raise ValueError(f"ecc n={ecc.n} must equal marker.n_bits={marker_bits}")
    return ecc


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device: torch.device):  # type: ignore[no-untyped-def]
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, (time.perf_counter() - t0) * 1000.0


@dataclass
class _Trial:
    """Everything one embed->distort->decode needs, bound explicitly (no loop-variable closures)."""

    item: CorpusImage
    strength: float
    dcfg: Any
    seed: int
    message: torch.Tensor
    payload: torch.Tensor
    capacity_bits: float


def _run_trial(t: _Trial, *, marked: bool, marker: Marker, ecc: Any, chain: Any, cfg: ExperimentConfig, device: torch.device, writer: ResultWriter, run_id: str, n_message_bits: int) -> None:
    x = t.item.image.to(device)
    enc_ms = math.nan
    fid: dict[str, float] = {}
    if marked:
        original = x
        x, enc_ms = _timed(lambda: marker.embed(original, t.payload, t.strength, force=cfg.force_embed), device)
        # Fidelity is marked-vs-original, before the channel.
        fid = {k: float(v[0]) for k, v in fidelity(x, original).items()}
    # Same distortion seed for marked and control -> identical sampled params.
    x = chain(x, make_generator(t.seed ^ 0x5BD1E995, device.type))
    res, dec_ms = _timed(lambda: marker.decode(x), device)
    writer.write(
        ResultRow(
            run_id=run_id,
            seed=t.seed,
            image_id=t.item.image_id,
            image_class=t.item.image_class,
            marked=marked,
            n_payload_bits=n_message_bits,
            embed_strength=t.strength if marked else 0.0,
            distortion_chain=chain.name,
            distortion_params=to_json({"severity": t.dcfg.severity, **chain.last_params}),
            capture_conditions="",
            ber=float(ber(res.llrs, t.payload)[0]),
            payload_recovered=bool(payload_recovered(res.llrs, t.message, ecc=ecc)[0]),
            detector_score=float(res.score[0]),
            psnr=fid.get("psnr", math.nan),
            ssim=fid.get("ssim", math.nan),
            lpips=fid.get("lpips", math.nan),
            encode_ms=enc_ms,
            decode_ms=dec_ms,
            capacity_bits=t.capacity_bits,
        )
    )


def run_experiment(cfg: ExperimentConfig, corpus: Iterable[CorpusImage] | None = None) -> Path:
    """Run ``cfg`` and return the run directory containing ``results.csv`` and ``summary.json``."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    marker: Marker = get_marker(cfg.marker.name, cfg.marker.n_bits, **cfg.marker.params)
    ecc = build_ecc(cfg.ecc, marker.n_bits)
    n_message_bits = ecc.k if ecc is not None else marker.n_bits
    images = corpus if corpus is not None else load_corpus(cfg.corpus, cfg.seed)
    run_dir = make_run_dir(cfg.output_dir, cfg.name)
    run_id = run_dir.name

    with ResultWriter(run_dir, cfg) as writer:
        for item in images:
            cap = marker.capacity(item.image.to(device))
            capacity_bits = float(cap[0]) if cap is not None else math.nan
            for strength in cfg.strengths:
                for dcfg in cfg.distortions:
                    chain = build_chain(dcfg.preset, dcfg.severity).to(device)
                    seed = trial_seed(cfg.seed, item.image_id, strength, dcfg.preset, dcfg.severity)
                    message = random_payload(1, n_message_bits, make_generator(seed)).to(device)
                    payload = ecc.encode(message) if ecc is not None else message  # channel bits
                    trial = _Trial(item, strength, dcfg, seed, message, payload, capacity_bits)
                    common: dict[str, Any] = dict(marker=marker, ecc=ecc, chain=chain, cfg=cfg, device=device, writer=writer, run_id=run_id, n_message_bits=n_message_bits)
                    _run_trial(trial, marked=True, **common)
                    if cfg.control:
                        _run_trial(trial, marked=False, **common)
    write_summary(run_dir)
    return run_dir
