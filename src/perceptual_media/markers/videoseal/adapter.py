"""``VideoSealMarker``: Meta Video Seal (facebookresearch/videoseal, MIT) behind the ``Marker`` protocol.

Learned baseline for Week 3 (brief §3 "borrow", §5 W3). The goal mismatch is known — Video Seal is
a provenance watermark trained for digital transforms — but it is modern PyTorch, actively
maintained, and its extractor is trained with geometric augmentation (perspective, rotation,
crop), which is precisely what killed the classical baseline.

Model: ``videoseal_1.0`` card → ``y_256b_img.pth`` (256 bits, luma-only embedder with a JND
attenuation map, 256 px processing resolution, watermark up/down-sampled to the image size).

Mapping to the protocol
-----------------------
* ``embed``: ``imgs_w = model.embed(img, msgs=payload)["imgs_w"]``; ``strength`` scales the
  residual ``img + strength · (imgs_w − img)`` (``strength = 1`` is the model's own
  ``scaling_w = 0.2``), which keeps the harness's strength axis meaningful across markers.
* ``decode``: ``preds = model.detect(img)["preds"]`` is ``(B, 1 + nbits)``: columns 1: are per-bit
  logits → ``llrs`` (positive ⇒ 1). Column 0 is nominally a detection channel but the image card
  was not trained with one (it sits at ~0.1 for marked and unmarked alike), so the presence
  ``score`` is the **message-confidence energy** ``mean |logit|`` — ≈ 11 on marked vs ≈ 0.5 on
  unmarked images — a blind statistic calibrated on the control set like any other.
* ``capacity``: ``None`` — Video Seal has no per-image estimate (it always emits a mark).

Loading
-------
The PyPI wheel's loader resolves ``videoseal/cards`` and ``configs/attenuation.yaml`` relative
to the *current directory* (it assumes you run from their repo). This adapter reads the
installed card itself, points the attenuation config at a vendored copy, and caches the
checkpoint under ``~/.cache/perceptual_media/videoseal/``.

Install with ``uv sync --extra videoseal``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch

from perceptual_media.core.types import (
    ImageBatch,
    Payload,
    assert_image_batch,
    assert_payload,
)
from perceptual_media.markers.base import DecodeResult, register_marker

_VENDOR = Path(__file__).parent / "vendor"
CACHE_DIR = Path.home() / ".cache" / "perceptual_media" / "videoseal"


def _download(url: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    import requests

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(dst)
    return dst


def load_videoseal(card: str = "videoseal_1.0", device: torch.device | str = "cpu") -> torch.nn.Module:
    """Build the Video Seal model named by ``card`` with cwd-independent paths; eval mode, no grad."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.jit deprecation noise inside videoseal
        import omegaconf
        import videoseal
        from videoseal.utils.cfg import setup_model

        card_path = Path(videoseal.__file__).parent / "cards" / f"{card}.yaml"
        if not card_path.exists():
            raise FileNotFoundError(f"model card {card!r} not found at {card_path}")
        config = omegaconf.OmegaConf.load(card_path)
        config.args.attenuation_config = str(_VENDOR / "attenuation.yaml")
        url = str(config.checkpoint_path)
        ckpt = _download(url, CACHE_DIR / url.rsplit("/", 1)[-1])
        model = setup_model(config, str(ckpt))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class VideoSealMarker:
    name = "videoseal"

    def __init__(self, n_bits: int = 256, card: str = "videoseal_1.0", device: str | None = None) -> None:
        self.card = card
        self._model: torch.nn.Module | None = None
        self._device = device
        self.n_bits = n_bits

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = load_videoseal(self.card, dev)
            nbits = int(self._model.get_random_msg(1).shape[1])  # the model's own message width
            if nbits != self.n_bits:
                raise ValueError(f"card {self.card!r} carries {nbits} bits; configure marker n_bits={nbits}")
        return self._model

    def _to_model(self, img: ImageBatch) -> ImageBatch:
        return img.to(next(self.model.parameters()).device)

    def embed(self, img: ImageBatch, payload: Payload, strength: float = 1.0, *, force: bool = False) -> ImageBatch:
        assert_image_batch(img)
        assert_payload(payload)
        if payload.shape != (img.shape[0], self.n_bits):
            raise ValueError(f"payload must be ({img.shape[0]}, {self.n_bits}), got {tuple(payload.shape)}")
        x = self._to_model(img)
        with torch.no_grad():
            out = self.model.embed(x, msgs=payload.to(x.device), is_video=False)
        delta = out["imgs_w"] - x
        return (x + strength * delta).clamp(0, 1).to(img.device)

    def decode(self, img: ImageBatch) -> DecodeResult:
        assert_image_batch(img)
        x = self._to_model(img)
        with torch.no_grad():
            preds = self.model.detect(x, is_video=False)["preds"]
        llrs = preds.to(img.device).float()[:, 1 : 1 + self.n_bits]
        return DecodeResult(llrs=llrs, score=llrs.abs().mean(dim=1))

    def capacity(self, img: ImageBatch) -> torch.Tensor | None:
        return None


register_marker("videoseal", VideoSealMarker)
