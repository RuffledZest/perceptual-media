"""Find the displayed image inside a phone photo of a screen, rectify it, and estimate the pose.

The **monitor is the fiducial** for screen captures: the panel is a large, uniform, dark
rectangle inside a black bezel. We detect that quad, map it to the display's native pixel grid
(``display.resolution``), and the shown image — rendered at 1:1 and centred by the viewer — is
the centred ``image_px × image_px`` square in that grid. That prediction is then *refined*
locally by snapping each side of the square to the strongest colour edge nearby, so a few
pixels of bezel-corner error or an off-centre window do not matter. The decoder never sees any
of this: ``locate`` exists to measure the channel (plan Task 24/25 note).

Everything here is NumPy/OpenCV on ``(H, W, 3)`` uint8 RGB; detection runs on a ~2k-wide
downscale, refinement on the full-resolution photo.

Pose: with the phone's 35 mm-equivalent focal length (config) and the square's physical size
(``image_px × pixel_pitch_mm``) ``solvePnP`` gives camera distance and the angle
between the panel normal and the line of sight — the *measured* distance / angle logged next
to the nominal ones from the filename.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

Quad = np.ndarray  # (4, 2) float32, order tl, tr, br, bl, in pixel coordinates of the photo


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------


def order_quad(pts: np.ndarray) -> Quad:
    """Any 4+ convex points → the 4 extreme corners ordered ``tl, tr, br, bl``."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = p.sum(1)
    d = p[:, 0] - p[:, 1]
    return np.stack([p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]]).astype(np.float32)


def square_quad(x0: float, y0: float, x1: float, y1: float) -> Quad:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def side_lengths(q: Quad) -> np.ndarray:
    """Lengths of the 4 sides ``top, right, bottom, left``."""
    return np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)


def transform_quad(q: Quad, h: np.ndarray) -> Quad:
    return cv2.perspectiveTransform(q.reshape(-1, 1, 2).astype(np.float32), h).reshape(4, 2)


# ---------------------------------------------------------------------------
# Screen quad
# ---------------------------------------------------------------------------


def _local_std(gray: np.ndarray, k: int) -> np.ndarray:
    g = gray.astype(np.float32)
    m = cv2.blur(g, (k, k))
    m2 = cv2.blur(g * g, (k, k))
    return np.sqrt(np.clip(m2 - m * m, 0, None))


def find_screen_quad(
    rgb: np.ndarray,
    *,
    v_range: tuple[int, int] = (22, 170),
    max_saturation: int = 120,
    max_local_std: float = 8.0,
) -> Quad | None:
    """Detect the display panel as the largest *uniform*, dark, low-saturation region.

    Uniformity (local std over a 15 px window on the downscaled image) is what separates the
    panel from a reflective glass wall or a dark desk that touches the bezel. The shown image
    is a hole in this region (or part of it, for flat fills) and is filled before taking the
    convex hull. Returns ``None`` if no region with 4 usable corners is found.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    v, s = hsv[..., 2], hsv[..., 1]
    mask = (v > v_range[0]) & (v < v_range[1]) & (s < max_saturation) & (_local_std(gray, 15) < max_local_std)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    region = ndimage.binary_fill_holes(labels == (int(np.argmax(sizes)) + 1)).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if len(approx) == 4:
            return order_quad(approx)
    return order_quad(hull)


# ---------------------------------------------------------------------------
# Image quad: predict from the screen, then refine on edges
# ---------------------------------------------------------------------------


def predict_image_quad(screen: Quad, resolution: tuple[int, int], image_px: int) -> Quad:
    """The centred ``image_px`` square of a ``resolution`` display, projected through ``screen``."""
    w, h = resolution
    h_screen = cv2.getPerspectiveTransform(square_quad(0, 0, w, h), screen)
    cx, cy, r = w / 2, h / 2, image_px / 2
    return transform_quad(square_quad(cx - r, cy - r, cx + r, cy + r), h_screen)


def refine_image_quad(
    rgb: np.ndarray,
    quad: Quad,
    image_px: int,
    *,
    search: int,
    margin: int = 96,
    band: int = 10,
    surround_tol: float = 20.0,
) -> tuple[Quad, np.ndarray]:
    """Snap each side of ``quad`` to the strongest colour edge within ``±search`` canonical px
    **whose outer side looks like the surround**.

    Works in a canonical view (the square warped to ``image_px`` plus ``margin``) so the search
    is axis-aligned. The surround colour is the median of a ring beyond the search range, so it
    is surround whenever the true edge lies within ``±search``. A candidate position is admitted
    only if the ``band`` px just outside it are within ``surround_tol`` (L2, grey levels) of that
    colour — this is what stops the snap landing on a strong *interior* edge (a text line, a
    tree line). Among admitted positions the strongest gradient wins; with none admitted the
    strongest gradient overall is used. Returns the refined quad and the gradient at each side
    ``left, right, top, bottom``.
    """
    size = image_px + 2 * margin
    dst = square_quad(margin, margin, margin + image_px, margin + image_px)
    h_canon = cv2.getPerspectiveTransform(quad, dst)
    canon = cv2.warpPerspective(rgb, h_canon, (size, size), flags=cv2.INTER_AREA).astype(np.float32)
    gx = np.abs(cv2.Sobel(canon, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=2) / 4.0
    gy = np.abs(cv2.Sobel(canon, cv2.CV_32F, 0, 1, ksize=3)).sum(axis=2) / 4.0
    inner = slice(margin + int(0.2 * image_px), margin + int(0.8 * image_px))
    col = gx[inner, :].mean(axis=0)
    row = gy[:, inner].mean(axis=1)

    ring = np.ones((size, size), dtype=bool)
    a, b = margin - search - 8, margin + image_px + search + 8
    ring[a:b, a:b] = False
    ring[: max(0, margin - band * 9), :] = False  # never beyond the canonical frame's margin
    surround = np.median(canon[ring].reshape(-1, 3), axis=0)

    def outer_band(p: int, side: str) -> np.ndarray:
        if side == "left":
            return canon[inner, max(0, p - 2 - band) : max(1, p - 2)]
        if side == "right":
            return canon[inner, p + 2 : p + 2 + band]
        if side == "top":
            return canon[max(0, p - 2 - band) : max(1, p - 2), inner]
        return canon[p + 2 : p + 2 + band, inner]

    def snap(profile: np.ndarray, centre: int, side: str) -> int:
        lo, hi = max(1, centre - search), min(len(profile) - 1, centre + search + 1)
        best, best_g = None, -1.0
        for pos in range(lo, hi):
            ob = outer_band(pos, side)
            if ob.size == 0 or np.linalg.norm(ob.reshape(-1, 3).mean(0) - surround) > surround_tol:
                continue
            if profile[pos] > best_g:
                best, best_g = pos, float(profile[pos])
        if best is None:
            best = lo + int(np.argmax(profile[lo:hi]))
        return best

    x0, x1 = snap(col, margin, "left"), snap(col, margin + image_px, "right")
    y0, y1 = snap(row, margin, "top"), snap(row, margin + image_px, "bottom")
    refined = transform_quad(square_quad(x0, y0, x1, y1), np.linalg.inv(h_canon))
    strength = np.array([col[x0], col[x1], row[y0], row[y1]], dtype=np.float32)
    return refined, strength


def rectify(rgb: np.ndarray, quad: Quad, image_px: int) -> np.ndarray:
    """Warp the quad to an axis-aligned ``(image_px, image_px, 3)`` uint8 image (area resampling)."""
    h = cv2.getPerspectiveTransform(quad, square_quad(0, 0, image_px, image_px))
    return cv2.warpPerspective(rgb, h, (image_px, image_px), flags=cv2.INTER_AREA)


def edge_contrast(rgb: np.ndarray, quad: Quad, image_px: int, band: int = 12) -> np.ndarray:
    """Colour difference (L2, grey levels) between a band just inside and just outside each side
    ``left, right, top, bottom`` — the located square should show a contrast on every side."""
    margin = 2 * band
    size = image_px + 2 * margin
    h = cv2.getPerspectiveTransform(quad, square_quad(margin, margin, margin + image_px, margin + image_px))
    canon = cv2.warpPerspective(rgb, h, (size, size), flags=cv2.INTER_AREA).astype(np.float32)
    inner = slice(margin + int(0.2 * image_px), margin + int(0.8 * image_px))
    lo, hi = margin, margin + image_px

    def diff(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a.reshape(-1, 3).mean(0) - b.reshape(-1, 3).mean(0)))

    return np.array(
        [
            diff(canon[inner, lo + 2 : lo + 2 + band], canon[inner, lo - 2 - band : lo - 2]),
            diff(canon[inner, hi - 2 - band : hi - 2], canon[inner, hi + 2 : hi + 2 + band]),
            diff(canon[lo + 2 : lo + 2 + band, inner], canon[lo - 2 - band : lo - 2, inner]),
            diff(canon[hi - 2 - band : hi - 2, inner], canon[hi + 2 : hi + 2 + band, inner]),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    distance_m: float
    """Camera to square centre, metres (scales with the assumed pixel pitch)."""
    angle_deg: float
    """Angle between the panel normal and the line of sight to the camera (0 = head-on)."""
    yaw_deg: float
    """Azimuth of the camera in the panel frame (> 0: camera to the panel's right)."""
    pitch_deg: float
    """Elevation of the camera in the panel frame (> 0: camera above the panel centre)."""
    reprojection_px: float


def camera_matrix(image_wh: tuple[int, int], focal_35mm: float) -> np.ndarray:
    """Pinhole intrinsics from a 35 mm-equivalent focal length (diagonal convention)."""
    w, h = image_wh
    f = focal_35mm / 43.27 * math.hypot(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def estimate_pose(quad: Quad, image_wh: tuple[int, int], focal_35mm: float, square_mm: float) -> Pose:
    """Distance and viewing angle of a physical square of side ``square_mm`` seen as ``quad``."""
    r = square_mm / 2000.0  # metres
    # Object frame: x right, y up, z out of the panel towards the viewer; tl, tr, br, bl.
    obj = np.array([[-r, r, 0], [r, r, 0], [r, -r, 0], [-r, -r, 0]], dtype=np.float64)
    k = camera_matrix(image_wh, focal_35mm)
    # ITERATIVE (planar DLT init + LM) reproduces synthetic poses to < 0.1 px; IPPE_SQUARE on
    # OpenCV 5.0 returned solutions that did not reproject onto their own input.
    ok, rvec, tvec = cv2.solvePnP(obj, quad.astype(np.float64), k, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rot, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    # Camera position in the panel frame (x right, y up, z towards the viewer).
    cam = -rot.T @ t
    x, y, z = (cam / np.linalg.norm(cam)).tolist()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    err = float(np.linalg.norm(proj.reshape(4, 2) - quad, axis=1).mean())
    return Pose(
        distance_m=float(np.linalg.norm(t)),
        angle_deg=math.degrees(math.acos(min(1.0, abs(z)))),
        yaw_deg=math.degrees(math.atan2(x, abs(z))),
        pitch_deg=math.degrees(math.atan2(y, math.hypot(x, z))),
        reprojection_px=err,
    )


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


@dataclass
class Located:
    screen_quad: Quad | None
    image_quad: Quad
    edge_strength: np.ndarray
    """Colour gradient at each snapped side, ``left, right, top, bottom``."""
    contrast: np.ndarray
    """Inside-vs-outside colour contrast per side."""
    px_per_image_px: float
    """Camera pixels per displayed image pixel (mean side length / ``image_px``)."""

    @property
    def ok(self) -> bool:
        """At least 3 sides show a clear inside/outside contrast (the 4th may be completed from
        the square constraint). Whether the capture is *usable* (scale, pose consistency) is
        judged by the logger from ``px_per_image_px`` and the PnP reprojection error."""
        return bool((self.contrast > 6).sum() >= 3)


def locate_screen_image(
    rgb: np.ndarray,
    *,
    resolution: tuple[int, int],
    image_px: int,
    detect_width: int = 2048,
) -> Located | None:
    """Locate the centred 1:1 image on a display in a photo. ``rgb`` is the full-res photo."""
    h, w, _ = rgb.shape
    ds = max(1, round(w / detect_width))
    small = cv2.resize(rgb, (w // ds, h // ds), interpolation=cv2.INTER_AREA) if ds > 1 else rgb
    screen_small = find_screen_quad(small)
    if screen_small is None:
        return None
    screen = screen_small * ds
    quad = predict_image_quad(screen, resolution, image_px)
    # Coarse then fine snapping; the search is in canonical (image) pixels.
    quad, _ = refine_image_quad(rgb, quad, image_px, search=64)
    quad, strength = refine_image_quad(rgb, quad, image_px, search=16)
    contrast = edge_contrast(rgb, quad, image_px)
    quad = complete_weak_side(quad, contrast, screen, resolution)
    return Located(
        screen_quad=screen,
        image_quad=quad,
        edge_strength=strength,
        contrast=edge_contrast(rgb, quad, image_px),
        px_per_image_px=float(side_lengths(quad).mean() / image_px),
    )


def complete_weak_side(
    quad: Quad, contrast: np.ndarray, screen: Quad, resolution: tuple[int, int], weak: float = 25.0
) -> Quad:
    """If exactly one side has no inside/outside contrast (e.g. navy against the panel's oblique
    glow), place it from the other three: in display coordinates the image is a square, so the
    weak side sits one perpendicular span away from its opposite side."""
    order = ("left", "right", "top", "bottom")
    weak_sides = [order[i] for i in range(4) if contrast[i] < weak]
    if len(weak_sides) != 1:
        return quad
    w, h = resolution
    h_screen = cv2.getPerspectiveTransform(square_quad(0, 0, w, h), screen)
    d = transform_quad(quad, np.linalg.inv(h_screen))  # tl, tr, br, bl in display px
    x0, x1 = (d[0, 0] + d[3, 0]) / 2, (d[1, 0] + d[2, 0]) / 2
    y0, y1 = (d[0, 1] + d[1, 1]) / 2, (d[2, 1] + d[3, 1]) / 2
    side = weak_sides[0]
    if side == "left":
        x0 = x1 - (y1 - y0)
    elif side == "right":
        x1 = x0 + (y1 - y0)
    elif side == "top":
        y0 = y1 - (x1 - x0)
    else:
        y1 = y0 + (x1 - x0)
    return transform_quad(square_quad(x0, y0, x1, y1), h_screen)
