"""``pm-capture decode``: photos of display-sheet slides → located → rectified → decoded → rows
(plan Task 25). Produces an ordinary run directory (``outputs/<sheet>-real-<stamp>/results.csv``
+ ``summary.json``) so ``pm-plot`` / ``pm-compare`` treat the physical channel exactly like a
simulated one. ``distortion_chain`` is:

* ``real_screen`` — the slide's image located by its ArUco fiducials (fallback: the monitor
  locator), perspective-rectified to ``image_px²``, then decoded;
* ``real_screen_unrectified`` — the same capture, but only the axis-aligned bounding box of the
  located quad, resized to ``image_px²`` without perspective correction: what a decoder gets with
  no synchronisation (the "sync gap" the plan asks for);
* ``real_screen_control`` — rectified crops of *unmarked* captures (the calibration set) decoded
  with the same markers: the false-positive accounting.

Capture filenames use the slide id as the image token: ``S08__screen__d1__a30__room.jpg``.
``capture_conditions`` carries the nominal and measured conditions as JSON, plus the EXIF
summary. Fidelity columns are the slide's *digital* marked-vs-original numbers (same meaning as
in the sweeps: the channel is not part of fidelity).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from perceptual_media.capture.locate import Quad, estimate_pose, locate_screen_image, order_quad, rectify, side_lengths
from perceptual_media.capture.log import read_exif
from perceptual_media.capture.naming import parse_capture_name
from perceptual_media.capture.sheet import ARUCO_DICT
from perceptual_media.core.config import CaptureConfig, EccConfig, MarkerConfig
from perceptual_media.core.io import from_uint8
from perceptual_media.harness.ecc import build_ecc
from perceptual_media.harness.results import ResultRow, ResultWriter, make_run_dir, write_summary
from perceptual_media.harness.runner import resolve_device
from perceptual_media.markers.base import Marker, get_marker
from perceptual_media.metrics.decoding import ber, payload_recovered

# ---------------------------------------------------------------------------
# Fiducial locator
# ---------------------------------------------------------------------------


def locate_by_fiducials(rgb: np.ndarray, spec: dict[str, Any], detect_width: int = 2048) -> Quad | None:
    """Image quad from the slide's ArUco markers (needs ≥ 3 of the 4). ``spec`` is ``spec.json``."""
    h, w, _ = rgb.shape
    ds = max(1, round(w / detect_width))
    small = cv2.resize(rgb, (w // ds, h // ds), interpolation=cv2.INTER_AREA) if ds > 1 else rgb
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT))
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return None
    f = spec["fiducial_px"]
    src, dst = [], []
    for c, i in zip(corners, ids.ravel().tolist(), strict=True):
        if i > 3:
            continue
        fx, fy = spec["fiducial_boxes"][i]
        src += [[fx, fy], [fx + f, fy], [fx + f, fy + f], [fx, fy + f]]  # slide coords, ArUco corner order
        dst += (c.reshape(4, 2) * ds).tolist()
    if len(src) < 12:
        return None
    hom, _ = cv2.findHomography(np.array(src, np.float32), np.array(dst, np.float32), cv2.RANSAC, 8.0)
    if hom is None:
        return None
    x0, y0 = spec["image_origin"]
    s = spec["image_px"]
    quad = np.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]], np.float32)
    return cv2.perspectiveTransform(quad.reshape(-1, 1, 2), hom).reshape(4, 2).astype(np.float32)


def unrectified_crop(rgb: np.ndarray, quad: Quad, image_px: int) -> np.ndarray:
    """Axis-aligned bounding box of ``quad``, resized to ``image_px²`` — no perspective correction."""
    x0, y0 = np.floor(quad.min(0)).astype(int)
    x1, y1 = np.ceil(quad.max(0)).astype(int)
    crop = rgb[max(0, y0) : y1, max(0, x0) : x1]
    return cv2.resize(crop, (image_px, image_px), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


@dataclass
class SlideMarker:
    marker: Marker
    ecc: Any
    n_message_bits: int


def _marker_for(row: pd.Series, cache: dict[str, SlideMarker]) -> SlideMarker:
    key = f"{row['marker']}|{row['n_bits']}|{row['marker_params']}|{row['ecc']}"
    if key not in cache:
        mcfg = MarkerConfig(name=row["marker"], n_bits=int(row["n_bits"]), params=json.loads(row["marker_params"]))
        marker = get_marker(mcfg.name, mcfg.n_bits, **mcfg.params)
        ecc = build_ecc(EccConfig(**json.loads(row["ecc"])), marker.n_bits) if row["ecc"] else None
        cache[key] = SlideMarker(marker, ecc, ecc.k if ecc is not None else marker.n_bits)
    return cache[key]


def _message_tensor(bits: str, device: torch.device) -> torch.Tensor:
    return torch.tensor([[float(b) for b in bits]], device=device)


def _decode_row(
    img_u8: np.ndarray, sm: SlideMarker, message: torch.Tensor, payload: torch.Tensor, device: torch.device
) -> tuple[float, bool, float, float]:
    x = from_uint8(img_u8).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        res = sm.marker.decode(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dec_ms = (time.perf_counter() - t0) * 1e3
    return (
        float(ber(res.llrs, payload)[0]),
        bool(payload_recovered(res.llrs, message, ecc=sm.ecc)[0]),
        float(res.score[0]),
        dec_ms,
    )


def decode_set(
    set_dir: Path,
    sheet_dir: Path,
    cfg: CaptureConfig,
    *,
    controls_dir: Path | None = None,
    output_dir: Path = Path("outputs"),
    device: str = "auto",
) -> Path:
    """Decode every slide capture in ``set_dir`` (and control crops in ``controls_dir``); returns the run dir."""
    dev = resolve_device(device)
    # keep_default_na: a marker named "null" and an empty ``ecc`` must survive as strings.
    sheet = pd.read_csv(sheet_dir / "sheet.csv", dtype={"message_bits": str}, keep_default_na=False).set_index(
        "slide_id"
    )
    with open(sheet_dir / "spec.json", encoding="utf-8") as fh:
        spec = json.load(fh)
    image_px = int(spec["image_px"])
    run_dir = make_run_dir(output_dir, f"{sheet_dir.name}-real")
    cache: dict[str, SlideMarker] = {}
    (set_dir / "check").mkdir(exist_ok=True)
    (set_dir / "rectified").mkdir(exist_ok=True)
    log_rows = []

    run_cfg = {
        "name": run_dir.name,
        "marker": {"name": f"real:{sheet_dir.name}"},  # pm-compare labels runs by marker.name
        "sheet": str(sheet_dir),
        "captures": str(set_dir),
        "controls": str(controls_dir) if controls_dir else None,
    }
    with ResultWriter(run_dir, run_cfg) as writer:
        for path in sorted(p for p in set_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            try:
                name = parse_capture_name(path)
            except ValueError as e:
                print(f"skip: {e}")
                continue
            if name.image_id not in sheet.index:
                print(f"skip {path.name}: {name.image_id!r} is not a slide id in {sheet_dir.name}")
                continue
            srow = sheet.loc[name.image_id]
            sm = _marker_for(srow, cache)
            rgb = np.asarray(Image.open(path).convert("RGB"))
            h, w, _ = rgb.shape
            quad = locate_by_fiducials(rgb, spec)
            how = "fiducials"
            if quad is None:
                loc = locate_screen_image(rgb, resolution=tuple(cfg.display.resolution), image_px=image_px)
                if loc is None:
                    print(f"skip {path.name}: could not locate the image")
                    continue
                quad, how = loc.image_quad, "screen"
            quad = order_quad(quad)
            pose = estimate_pose(quad, (w, h), cfg.camera.focal_35mm, image_px * cfg.display.pixel_pitch_mm)
            rect = rectify(rgb, quad, image_px)
            unrect = unrectified_crop(rgb, quad, image_px)
            Image.fromarray(rect).save(set_dir / "rectified" / f"{path.stem}.png")
            vis = cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            cv2.polylines(vis, [(quad / 4).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            Image.fromarray(vis).save(set_dir / "check" / f"{path.stem}.jpg", quality=70)

            exif = read_exif(path)
            conditions = {
                "capture_id": path.stem,
                "slide_id": name.image_id,
                "medium": name.medium,
                "distance_m": name.distance_m,
                "angle_deg": name.angle_deg,
                "lighting": name.lighting,
                "measured_distance_m": round(pose.distance_m, 3),
                "measured_angle_deg": round(pose.angle_deg, 1),
                "measured_azimuth_deg": round(pose.yaw_deg, 1),
                "measured_elevation_deg": round(pose.pitch_deg, 1),
                "px_per_image_px": round(float(side_lengths(quad).mean() / image_px), 2),
                "reprojection_px": round(pose.reprojection_px, 2),
                "located_by": how,
                "display": cfg.display.name,
                "camera": exif.camera,
                "iso": exif.iso,
                "exposure_s": exif.exposure_s,
            }
            message = _message_tensor(srow["message_bits"], dev)
            payload = sm.ecc.encode(message) if sm.ecc is not None else message
            for chain_name, img in (("real_screen", rect), ("real_screen_unrectified", unrect)):
                b, rec, score, dec_ms = _decode_row(img, sm, message, payload, dev)
                writer.write(
                    ResultRow(
                        run_id=run_dir.name,
                        seed=int(srow["seed"]),
                        image_id=str(srow["image_id"]),
                        image_class=str(srow["image_class"]),
                        marked=True,
                        n_payload_bits=sm.n_message_bits,
                        embed_strength=float(srow["strength"]),
                        distortion_chain=chain_name,
                        distortion_params=json.dumps({"marker": srow["marker"]}),
                        capture_conditions=json.dumps(conditions),
                        ber=b,
                        payload_recovered=rec,
                        detector_score=score,
                        psnr=float(srow["psnr"]),
                        ssim=float(srow["ssim"]),
                        lpips=float(srow["lpips"]),
                        decode_ms=dec_ms,
                    )
                )
                if chain_name == "real_screen":
                    log_rows.append(
                        {**conditions, "marker": srow["marker"], "ber": b, "recovered": rec, "score": score}
                    )

        # Controls: unmarked calibration crops through every marker used on the sheet.
        if controls_dir is not None and (controls_dir / "captures.csv").exists():
            ctrl = pd.read_csv(controls_dir / "captures.csv")
            markers = sheet.drop_duplicates(subset=["marker", "n_bits", "marker_params", "ecc"])
            for _, c in ctrl[ctrl["locate_ok"]].iterrows():
                crop = np.asarray(Image.open(controls_dir / "rectified" / f"{c['capture_id']}.png").convert("RGB"))
                for _, mrow in markers.iterrows():
                    sm = _marker_for(mrow, cache)
                    # Any fixed message works for a control row: BER ~ 0.5 and no recovery expected.
                    message = _message_tensor(mrow["message_bits"], dev)
                    payload = sm.ecc.encode(message) if sm.ecc is not None else message
                    b, rec, score, dec_ms = _decode_row(crop, sm, message, payload, dev)
                    writer.write(
                        ResultRow(
                            run_id=run_dir.name,
                            seed=0,
                            image_id=str(c["image_id"]),
                            image_class=str(c["image_class"]),
                            marked=False,
                            n_payload_bits=sm.n_message_bits,
                            embed_strength=0.0,
                            distortion_chain="real_screen_control",
                            distortion_params=json.dumps({"marker": mrow["marker"]}),
                            capture_conditions=json.dumps(
                                {
                                    "capture_id": c["capture_id"],
                                    "distance_m": float(c["distance_m"]),
                                    "angle_deg": float(c["angle_deg"]),
                                    "measured_angle_deg": round(float(c["measured_angle_deg"]), 1),
                                }
                            ),
                            ber=b,
                            payload_recovered=rec,
                            detector_score=score,
                            decode_ms=dec_ms,
                        )
                    )
    write_summary(run_dir)
    if log_rows:
        pd.DataFrame(log_rows).to_csv(set_dir / "decode_log.csv", index=False)
    return run_dir


__all__ = ["decode_set", "locate_by_fiducials", "unrectified_crop"]
