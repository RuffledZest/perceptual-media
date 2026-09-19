import cv2
import numpy as np
import pandas as pd

from perceptual_media.capture.sheet import (
    SheetMarker,
    SheetSpec,
    build_sheet,
    fiducial_boxes,
    image_origin,
    render_slide,
)
from perceptual_media.core.config import EccConfig, MarkerConfig


def test_render_slide_places_image_and_detectable_fiducials():
    spec = SheetSpec(name="t", images=[], markers=[], resolution=[1280, 720], image_px=256, fiducial_px=48, gap_px=24)
    marked = np.full((256, 256, 3), (200, 30, 30), np.uint8)
    slide = render_slide(spec, marked, "S00 test")
    x0, y0 = image_origin(spec)
    assert slide.shape == (720, 1280, 3)
    np.testing.assert_array_equal(slide[y0 + 10, x0 + 10], (200, 30, 30))
    # Detect on a degraded copy: blurred, noisy, downscaled - closer to a phone photo than the render.
    rng = np.random.default_rng(0)
    photo = cv2.resize(cv2.GaussianBlur(slide, (0, 0), 1.5), None, fx=0.7, fy=0.7)
    photo = np.clip(photo.astype(np.float32) + rng.normal(0, 4, photo.shape), 0, 255).astype(np.uint8)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = det.detectMarkers(photo)
    assert ids is not None and sorted(ids.ravel().tolist()) == [0, 1, 2, 3]
    boxes = dict(zip(ids.ravel().tolist(), corners, strict=True))
    for i, (fx, fy) in enumerate(fiducial_boxes(spec)):
        assert np.allclose(boxes[i].reshape(4, 2)[0], (fx * 0.7, fy * 0.7), atol=2.0)


def test_build_sheet_writes_manifest_and_files(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "x").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for iid in ("a", "b"):
        cv2.imwrite(str(corpus / "x" / f"{iid}.png"), rng.integers(0, 256, (64, 64, 3)).astype(np.uint8))
    pd.DataFrame({"image_id": ["a", "b"], "image_class": ["t", "t"], "path": ["x/a.png", "x/b.png"]}).to_csv(
        corpus / "manifest.csv", index=False
    )
    spec = SheetSpec(
        name="t",
        images=["a", "b"],
        markers=[
            SheetMarker(MarkerConfig(name="null", n_bits=127), EccConfig(n=127, k=64), 1.0, images=["a"]),
            SheetMarker(MarkerConfig(name="null", n_bits=64)),
        ],
        resolution=[640, 360],
        image_px=64,
        fiducial_px=32,
        gap_px=8,
        device="cpu",
    )
    df = build_sheet(spec, tmp_path / "out", corpus)
    assert list(df["slide_id"]) == ["S00", "S01", "S02"]
    assert list(df["image_id"]) == ["a", "a", "b"]
    assert len(df.loc[0, "message_bits"]) == 64 and len(df.loc[1, "message_bits"]) == 64
    assert (tmp_path / "out" / "S02.png").exists() and (tmp_path / "out" / "S02_marked.png").exists()
    assert (tmp_path / "out" / "sheet.csv").exists() and (tmp_path / "out" / "spec.json").exists()
    assert df["psnr"].min() > 60  # null marker: identity embed
