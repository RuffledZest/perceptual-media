import json
import math

import cv2
import numpy as np
import pandas as pd
import pytest

from perceptual_media.capture.ingest import decode_set, locate_by_fiducials, unrectified_crop
from perceptual_media.capture.locate import square_quad, transform_quad
from perceptual_media.capture.sheet import SheetMarker, SheetSpec, build_sheet
from perceptual_media.core.config import CameraConfig, CaptureConfig, DisplayConfig, MarkerConfig


def fake_photo(slide: np.ndarray, tilt_deg: float, size=(2048, 1536), seed=0):
    """Perspective-warp a slide onto a cluttered background, blur and add noise; returns (photo, H)."""
    rng = np.random.default_rng(seed)
    w, h = size
    photo = np.clip(rng.normal(140, 25, (h, w, 3)), 0, 255).astype(np.uint8)
    sh, sw = slide.shape[:2]
    scale = 0.6
    pw, ph = sw * scale, sh * scale
    ox, oy = (w - pw) / 2, (h - ph) / 2
    shrink = 1 - 0.5 * math.sin(math.radians(tilt_deg))
    dst = np.array(
        [[ox, oy], [ox + pw * shrink, oy + 30 * shrink], [ox + pw * shrink, oy + ph - 30 * shrink], [ox, oy + ph]],
        np.float32,
    )
    H = cv2.getPerspectiveTransform(square_quad(0, 0, sw, sh), dst)
    warped = cv2.warpPerspective(slide, H, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), H, (w, h)) > 127
    photo[mask] = warped[mask]
    photo = cv2.GaussianBlur(photo, (0, 0), 1.0)
    photo = np.clip(photo.astype(np.float32) + rng.normal(0, 3, photo.shape), 0, 255).astype(np.uint8)
    return photo, H


@pytest.fixture
def sheet(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "x").mkdir(parents=True)
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 256, (128, 128, 3)).astype(np.float32), (0, 0), 2)
    cv2.imwrite(str(corpus / "x" / "a.png"), cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    pd.DataFrame({"image_id": ["a"], "image_class": ["t"], "path": ["x/a.png"]}).to_csv(
        corpus / "manifest.csv", index=False
    )
    spec = SheetSpec(
        name="t",
        images=["a"],
        markers=[SheetMarker(MarkerConfig(name="null", n_bits=64))],
        resolution=[960, 540],
        image_px=128,
        fiducial_px=48,
        gap_px=24,
        device="cpu",
    )
    out = tmp_path / "sheets" / "t"
    build_sheet(spec, out, corpus)
    return out


def test_locate_by_fiducials_matches_truth(sheet):
    slide = np.asarray(cv2.cvtColor(cv2.imread(str(sheet / "S00.png")), cv2.COLOR_BGR2RGB))
    spec = json.loads((sheet / "spec.json").read_text())
    for tilt in (0.0, 35.0):
        photo, H = fake_photo(slide, tilt)
        x0, y0 = spec["image_origin"]
        truth = transform_quad(square_quad(x0, y0, x0 + 128, y0 + 128), H)
        quad = locate_by_fiducials(photo, spec)
        assert quad is not None
        assert np.linalg.norm(quad - truth, axis=1).max() < 2.0
        assert unrectified_crop(photo, quad, 128).shape == (128, 128, 3)


def test_decode_set_end_to_end(sheet, tmp_path):
    slide = np.asarray(cv2.cvtColor(cv2.imread(str(sheet / "S00.png")), cv2.COLOR_BGR2RGB))
    photo, _ = fake_photo(slide, 25.0)
    captures = tmp_path / "caps"
    captures.mkdir()
    cv2.imwrite(str(captures / "S00__screen__d0.5__a30__room.jpg"), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
    cfg = CaptureConfig(display=DisplayConfig("test", [960, 540], 0.3), camera=CameraConfig("test", 24.0), image_px=128)
    run_dir = decode_set(captures, sheet, cfg, output_dir=tmp_path / "out", device="cpu")
    df = pd.read_csv(run_dir / "results.csv")
    assert sorted(df["distortion_chain"]) == ["real_screen", "real_screen_unrectified"]
    assert df["marked"].all() and (df["n_payload_bits"] == 64).all()
    cond = json.loads(df.loc[0, "capture_conditions"])
    assert cond["located_by"] == "fiducials" and 20 < cond["measured_angle_deg"] < 40
    assert 0.3 < df["ber"].mean() < 0.7  # null marker: random LLRs
    assert (captures / "rectified" / "S00__screen__d0.5__a30__room.png").exists()
    assert (run_dir / "summary.json").exists()
