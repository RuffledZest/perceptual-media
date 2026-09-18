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

import torch

from perceptual_media.core.config import CorpusConfig, ExperimentConfig
from perceptual_media.core.seed import make_generator, seed_everything
from perceptual_media.core.types import ImageBatch
from perceptual_media.distort.presets import build_chain
from perceptual_media.harness.results import ResultRow, ResultWriter, make_run_dir, to_json
from perceptual_media.markers.base import Marker, get_marker, random_payload
from perceptual_media.metrics.decoding import ber, payload_recovered


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
    """Yield images per ``CorpusConfig``. Manifest loading arrives in Task 8; until then the
    built-in smoke corpus is used and a warning says so."""
    if Path(cfg.manifest).exists():
        raise NotImplementedError("manifest corpora arrive in Task 8")
    warnings.warn(f"manifest {cfg.manifest!r} not found; using built-in smoke corpus", stacklevel=2)
    images: Iterable[CorpusImage] = builtin_smoke_corpus(seed)
    if cfg.classes is not None:
        images = (im for im in images if im.image_class in cfg.classes)
    if cfg.limit is not None:
        images = (im for _, im in zip(range(cfg.limit), images))
    yield from images


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device: torch.device):  # type: ignore[no-untyped-def]
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, (time.perf_counter() - t0) * 1000.0


def run_experiment(cfg: ExperimentConfig, corpus: Iterable[CorpusImage] | None = None) -> Path:
    """Run ``cfg`` and return the run directory containing ``results.csv``."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    marker: Marker = get_marker(cfg.marker.name, cfg.marker.n_bits, **cfg.marker.params)
    images = corpus if corpus is not None else load_corpus(cfg.corpus, cfg.seed)
    run_dir = make_run_dir(cfg.output_dir, cfg.name)
    run_id = run_dir.name

    with ResultWriter(run_dir, cfg) as writer:
        for item in images:
            img = item.image.to(device)
            for strength in cfg.strengths:
                for dcfg in cfg.distortions:
                    chain = build_chain(dcfg.preset, dcfg.severity).to(device)
                    seed = trial_seed(cfg.seed, item.image_id, strength, dcfg.preset, dcfg.severity)
                    payload = random_payload(1, marker.n_bits, make_generator(seed)).to(device)

                    def one(x: ImageBatch, marked: bool) -> None:
                        enc_ms = math.nan
                        if marked:
                            x, enc_ms = _timed(lambda: marker.embed(x, payload, strength), device)
                        # Same distortion seed for marked and control → identical sampled params.
                        x = chain(x, make_generator(seed ^ 0x5BD1E995, device.type))
                        res, dec_ms = _timed(lambda: marker.decode(x), device)
                        writer.write(
                            ResultRow(
                                run_id=run_id,
                                seed=seed,
                                image_id=item.image_id,
                                image_class=item.image_class,
                                marked=marked,
                                n_payload_bits=marker.n_bits,
                                embed_strength=strength if marked else 0.0,
                                distortion_chain=chain.name,
                                distortion_params=to_json({"severity": dcfg.severity, **chain.last_params}),
                                capture_conditions="",
                                ber=float(ber(res.llrs, payload)[0]),
                                payload_recovered=bool(payload_recovered(res.llrs, payload)[0]),
                                detector_score=float(res.score[0]),
                                encode_ms=enc_ms,
                                decode_ms=dec_ms,
                            )
                        )

                    one(img, marked=True)
                    if cfg.control:
                        one(img, marked=False)
    return run_dir
