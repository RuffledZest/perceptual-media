"""``pm-capture log`` (``capture.cli``): build ``captures.csv`` for a folder of phone photos (plan Task 23).

One row per capture, combining three sources so nothing has to be typed by hand:

* **nominal** conditions from the filename (``capture.naming``): image, medium, distance, angle,
  lighting, marked;
* **camera** metadata from EXIF: make/model, ISO, exposure, aperture, physical focal length;
* **measured** geometry from the photo itself (``capture.locate``): where the image is, the
  camera-pixels-per-image-pixel scale, and the PnP distance / viewing angle (azimuth, elevation)
  using the rig in ``configs/capture.yaml``.

Side outputs, for eyeballing and for ingestion: ``check/<id>.jpg`` (quarter-size photo with the
screen and image quads drawn) and ``rectified/<id>.png`` (the located image warped back to
``image_px`` square). The CSV is fully derived from the files, so it is regenerated on every run
rather than appended to.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import ExifTags, Image

from perceptual_media.capture.locate import Located, estimate_pose, locate_screen_image, rectify
from perceptual_media.capture.naming import CaptureName, parse_capture_name
from perceptual_media.core.config import CaptureConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class ExifSummary:
    camera: str = ""
    iso: float = float("nan")
    exposure_s: float = float("nan")
    f_number: float = float("nan")
    focal_mm: float = float("nan")
    captured_at: str = ""


def read_exif(path: Path) -> ExifSummary:
    """The few EXIF fields the channel model cares about; blank/NaN when absent."""
    with Image.open(path) as im:
        ex = im.getexif()
        ifd = ex.get_ifd(0x8769)
    tags = {ExifTags.TAGS.get(k, k): v for d in (ex, ifd) for k, v in d.items()}

    def num(key: str) -> float:
        v = tags.get(key)
        try:
            return float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    return ExifSummary(
        camera=" ".join(str(tags[k]) for k in ("Make", "Model") if k in tags),
        iso=num("ISOSpeedRatings"),
        exposure_s=num("ExposureTime"),
        f_number=num("FNumber"),
        focal_mm=num("FocalLength"),
        captured_at=str(tags.get("DateTimeOriginal", "")),
    )


def _corpus_classes(manifest: Path) -> dict[str, str]:
    if not manifest.exists():
        return {}
    df = pd.read_csv(manifest)
    return dict(zip(df["image_id"], df["image_class"], strict=True))


def log_one(path: Path, cfg: CaptureConfig, classes: dict[str, str], out_dir: Path) -> dict[str, object]:
    """Locate, measure and describe one capture; writes its check/rectified images."""
    name: CaptureName = parse_capture_name(path)
    exif = read_exif(path)
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    h, w, _ = rgb.shape
    row: dict[str, object] = {
        "capture_id": path.stem,
        "file": path.name,
        "image_id": name.image_id,
        "image_class": classes.get(name.image_id, ""),
        "medium": name.medium,
        "marked": name.marked,
        "distance_m": name.distance_m,
        "angle_deg": name.angle_deg,
        "lighting": name.lighting,
        "display": cfg.display.name,
        **asdict(exif),
        "width": w,
        "height": h,
    }
    loc: Located | None = None
    if name.medium == "screen":
        loc = locate_screen_image(rgb, resolution=tuple(cfg.display.resolution), image_px=cfg.image_px)
    if loc is None:
        row.update(locate_ok=False)
        return row
    pose = estimate_pose(loc.image_quad, (w, h), cfg.camera.focal_35mm, cfg.image_px * cfg.display.pixel_pitch_mm)
    row.update(
        locate_ok=bool(loc.ok and pose.reprojection_px < 5.0),
        image_quad=json.dumps(np.round(loc.image_quad, 1).tolist()),
        edge_contrast_min=float(loc.contrast.min()),
        px_per_image_px=loc.px_per_image_px,
        measured_distance_m=pose.distance_m,
        measured_angle_deg=pose.angle_deg,
        measured_azimuth_deg=pose.yaw_deg,
        measured_elevation_deg=pose.pitch_deg,
        reprojection_px=pose.reprojection_px,
    )
    (out_dir / "check").mkdir(parents=True, exist_ok=True)
    (out_dir / "rectified").mkdir(parents=True, exist_ok=True)
    vis = cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    if loc.screen_quad is not None:
        cv2.polylines(vis, [(loc.screen_quad / 4).astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 0), 2)
    cv2.polylines(vis, [(loc.image_quad / 4).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    Image.fromarray(vis).save(out_dir / "check" / f"{path.stem}.jpg", quality=70)
    Image.fromarray(rectify(rgb, loc.image_quad, cfg.image_px)).save(out_dir / "rectified" / f"{path.stem}.png")
    return row


def log_folder(folder: Path, cfg: CaptureConfig, manifest: Path = Path("data/corpus/manifest.csv")) -> pd.DataFrame:
    """Build and write ``<folder>/captures.csv``; returns the table."""
    classes = _corpus_classes(manifest)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    rows = []
    for p in files:
        try:
            rows.append(log_one(p, cfg, classes, folder))
        except ValueError as e:  # bad filename: report and keep going
            print(f"skip: {e}", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_csv(folder / "captures.csv", index=False)
    return df


def print_log(df: pd.DataFrame, folder: Path) -> None:
    """Console summary of a captures table."""
    show = (
        "capture_id", "distance_m", "angle_deg", "measured_distance_m", "measured_angle_deg",
        "px_per_image_px", "reprojection_px", "locate_ok",
    )
    cols = [c for c in show if c in df]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    located = int(df["locate_ok"].sum()) if "locate_ok" in df else 0
    print(f"\n{len(df)} captures, {located} located -> {folder / 'captures.csv'}")
