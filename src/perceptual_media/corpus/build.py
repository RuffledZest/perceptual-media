"""``pm-corpus build``: write the evaluation corpus to ``data/corpus/`` with a manifest.

Synthetic classes are generated from ``--seed``; ``--photos`` / ``--faces`` folders contribute a
seeded, fixed-size subset resized (shorter side) and centre-cropped to ``--size``. If
``--photos`` is omitted, ``configs/paths.yaml → datasets.landscape_pictures`` is used when it
exists. Same seed + same source folders ⇒ byte-identical output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from perceptual_media.core.config import load_paths
from perceptual_media.core.io import from_uint8, save_image
from perceptual_media.corpus.manifest import ManifestRow, write_manifest
from perceptual_media.corpus.synth import all_synthetic

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def _resize_crop(path: Path, size: int) -> Image.Image:
    """Resize shorter side to ``size`` (Lanczos) then centre-crop to ``size × size``."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.Resampling.LANCZOS)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        return im.crop((left, top, left + size, top + size))


def build_corpus(
    out: str | Path,
    seed: int = 0,
    size: int = 512,
    photos: str | Path | None = None,
    faces: str | Path | None = None,
    n_photos: int = 40,
    n_faces: int = 20,
) -> Path:
    """Build the corpus; returns the manifest path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[ManifestRow] = []

    for s in all_synthetic(size, seed):
        rel = Path(s.image_class) / f"{s.image_id}.png"
        save_image(s.image, out / rel)
        rows.append(ManifestRow(s.image_id, s.image_class, s.source, rel.as_posix(), size, size))

    for cls, folder, n in (("photo", photos, n_photos), ("face", faces, n_faces)):
        if not folder:
            continue
        folder = Path(folder)
        files = _list_images(folder)
        if not files:
            print(f"warning: no images under {folder}", file=sys.stderr)
            continue
        rng = np.random.default_rng(seed)
        pick = [files[i] for i in sorted(rng.permutation(len(files))[:n].tolist())]
        for i, src in enumerate(pick):
            image_id = f"{cls}_{i:03d}"
            rel = Path(cls) / f"{image_id}.png"
            im = _resize_crop(src, size)
            save_image(from_uint8(np.asarray(im)), out / rel)
            rows.append(ManifestRow(image_id, cls, f"{cls}:{src.name}", rel.as_posix(), size, size))

    manifest = out / "manifest.csv"
    write_manifest(rows, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pm-corpus", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the corpus")
    b.add_argument("--out", default="data/corpus")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--size", type=int, default=512)
    b.add_argument("--photos", help="folder of photographs (default: paths.yaml landscape_pictures)")
    b.add_argument("--faces", help="folder of face images (default: none)")
    b.add_argument("--n-photos", type=int, default=40)
    b.add_argument("--n-faces", type=int, default=20)
    b.add_argument("--no-photos", action="store_true", help="skip the photo class even if paths.yaml has one")
    args = p.parse_args(argv)

    photos = args.photos
    if photos is None and not args.no_photos and Path("configs/paths.yaml").exists():
        cand = load_paths().datasets.get("landscape_pictures", "")
        if cand and Path(cand).is_dir():
            photos = cand
    manifest = build_corpus(args.out, args.seed, args.size, photos, args.faces, args.n_photos, args.n_faces)
    from perceptual_media.corpus.manifest import load_manifest

    rows = load_manifest(manifest)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.image_class] = counts.get(r.image_class, 0) + 1
    print(f"wrote {len(rows)} images to {manifest.parent}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
