import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from perceptual_media.core.config import CorpusConfig
from perceptual_media.corpus.build import build_corpus, main
from perceptual_media.corpus.manifest import CLASSES, ManifestRow, iter_images, load_manifest, write_manifest
from perceptual_media.corpus.synth import all_synthetic, flat_images, gradient_images, text_images, textured_images
from perceptual_media.harness.runner import load_corpus

SIZE = 64


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _photo_folder(tmp_path: Path, n: int = 6) -> Path:
    d = tmp_path / "photos"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        w, h = 90 + 10 * i, 70 + 5 * i  # non-square, varied
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(d / f"p{i}.jpg", quality=90)
    return d


def test_synthetic_class_minimums_and_shapes() -> None:
    gen = torch.Generator().manual_seed(0)
    assert sum(1 for _ in flat_images(SIZE)) >= 6
    assert sum(1 for _ in gradient_images(SIZE)) >= 4
    assert sum(1 for _ in text_images(SIZE)) >= 4
    assert sum(1 for _ in textured_images(SIZE, gen)) >= 6
    for s in all_synthetic(SIZE, 0):
        assert s.image.shape == (1, 3, SIZE, SIZE) and s.image.dtype == torch.float32
        assert 0 <= s.image.min() and s.image.max() <= 1
        assert s.image_class in CLASSES


def test_flat_images_are_truly_flat_and_text_has_edges() -> None:
    for s in flat_images(SIZE):
        assert s.image.flatten(2).std(dim=2).max() == 0
    for s in text_images(SIZE):
        assert s.image.std() > 0.05


def test_build_is_byte_identical_for_same_seed(tmp_path: Path) -> None:
    photos = _photo_folder(tmp_path)
    build_corpus(tmp_path / "a", seed=0, size=SIZE, photos=photos, n_photos=3)
    build_corpus(tmp_path / "b", seed=0, size=SIZE, photos=photos, n_photos=3)
    build_corpus(tmp_path / "c", seed=1, size=SIZE, photos=photos, n_photos=3)
    assert _tree_hash(tmp_path / "a") == _tree_hash(tmp_path / "b")
    assert _tree_hash(tmp_path / "a") != _tree_hash(tmp_path / "c")


def test_manifest_contents_and_photo_subset(tmp_path: Path) -> None:
    photos = _photo_folder(tmp_path)
    manifest = build_corpus(tmp_path / "corpus", seed=0, size=SIZE, photos=photos, n_photos=3)
    rows = load_manifest(manifest)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.image_class] = counts.get(r.image_class, 0) + 1
        assert (manifest.parent / r.path).exists()
        assert r.width == SIZE and r.height == SIZE
    assert counts["photo"] == 3 and "face" not in counts
    assert len({r.image_id for r in rows}) == len(rows)
    # photo files are 64x64 after resize + centre-crop
    for r in rows:
        if r.image_class == "photo":
            with Image.open(manifest.parent / r.path) as im:
                assert im.size == (SIZE, SIZE)


def test_iter_images_filters_and_runner_uses_manifest(tmp_path: Path) -> None:
    manifest = build_corpus(tmp_path / "corpus", seed=0, size=SIZE)
    items = list(iter_images(manifest, classes=["flat", "text"], limit=4))
    assert len(items) == 4 and all(r.image_class == "flat" for r, _ in items)
    assert items[0][1].shape == (1, 3, SIZE, SIZE)
    corpus = list(load_corpus(CorpusConfig(manifest=str(manifest), classes=["gradient"], limit=2)))
    assert [c.image_class for c in corpus] == ["gradient", "gradient"]


def test_manifest_roundtrip(tmp_path: Path) -> None:
    rows = [ManifestRow("a", "flat", "flat:x", "flat/a.png", 8, 8), ManifestRow("b", "photo", "photo:y.jpg", "photo/b.png", 8, 8)]
    write_manifest(rows, tmp_path / "m.csv")
    assert load_manifest(tmp_path / "m.csv") == rows


def test_cli_build_no_photos(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["build", "--out", str(tmp_path / "c"), "--seed", "0", "--size", str(SIZE), "--no-photos"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "flat=" in out and "photo=" not in out
