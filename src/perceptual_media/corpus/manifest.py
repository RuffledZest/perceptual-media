"""``manifest.csv``: the index of a built corpus.

Columns: ``image_id, image_class, source, path, width, height``. ``path`` is relative to the
manifest's directory. Classes: ``flat``, ``gradient``, ``text``, ``textured`` (procedural),
``photo`` (natural photographs), ``face`` (when a folder is provided).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from perceptual_media.core.io import load_image
from perceptual_media.core.types import ImageBatch

CLASSES = ("flat", "gradient", "text", "textured", "photo", "face")


@dataclass
class ManifestRow:
    image_id: str
    image_class: str
    source: str
    path: str
    width: int
    height: int

    @staticmethod
    def columns() -> list[str]:
        return [f.name for f in fields(ManifestRow)]


def write_manifest(rows: list[ManifestRow], path: str | Path) -> None:
    """Write rows as CSV (``\\n`` line endings, UTF-8) — deterministic for a fixed row list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ManifestRow.columns(), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def load_manifest(path: str | Path) -> list[ManifestRow]:
    """Read ``manifest.csv``."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [ManifestRow(r["image_id"], r["image_class"], r["source"], r["path"], int(r["width"]), int(r["height"])) for r in csv.DictReader(fh)]


def select_rows(rows: list[ManifestRow], classes: list[str] | None = None, per_class: int | None = None, limit: int | None = None) -> list[ManifestRow]:
    """Filter by class, cap each class at ``per_class`` (manifest order), then truncate to ``limit``."""
    if classes is not None:
        rows = [r for r in rows if r.image_class in classes]
    if per_class is not None:
        seen: dict[str, int] = {}
        kept = []
        for r in rows:
            if seen.get(r.image_class, 0) < per_class:
                kept.append(r)
                seen[r.image_class] = seen.get(r.image_class, 0) + 1
        rows = kept
    if limit is not None:
        rows = rows[:limit]
    return rows


def iter_images(
    manifest_path: str | Path, classes: list[str] | None = None, limit: int | None = None, per_class: int | None = None
) -> Iterator[tuple[ManifestRow, ImageBatch]]:
    """Yield ``(row, image)`` for manifest rows after ``select_rows``."""
    manifest_path = Path(manifest_path)
    rows = select_rows(load_manifest(manifest_path), classes, per_class, limit)
    for r in rows:
        yield r, load_image(manifest_path.parent / r.path)
