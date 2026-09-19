import itertools
import math

import cv2
import numpy as np
import pytest

from perceptual_media.capture.locate import (
    camera_matrix,
    estimate_pose,
    locate_screen_image,
    order_quad,
    rectify,
    square_quad,
    transform_quad,
)
from perceptual_media.capture.naming import CaptureName, parse_capture_name

RES = (1920, 1080)
IMG = 512


def render_scene(
    content: np.ndarray, tilt_deg: float, seed: int = 0, size=(2048, 1536)
) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic phone photo: cluttered background, black bezel, dark uniform panel with the
    ``content`` square centred at 1:1, all warped by a perspective tilt. Returns (photo, truth quad)."""
    rng = np.random.default_rng(seed)
    w, h = size
    # Background: textured (wood grain / cables) with random dark and light rectangles, so
    # neither a plain threshold nor "largest dark region" finds the panel; only "uniform" does.
    photo = np.full((h, w, 3), 150, np.uint8)
    for _ in range(12):
        x0, y0 = rng.integers(0, w - 200), rng.integers(0, h - 200)
        photo[y0 : y0 + rng.integers(50, 400), x0 : x0 + rng.integers(50, 400)] = rng.integers(20, 230, 3)
    grain = rng.normal(0, 14, (h, w, 1))
    photo = np.clip(photo.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    # Display rendered at 0.45 px per display px, then bezel + panel + image in display space.
    scale = 0.45
    dw, dh = int(RES[0] * scale), int(RES[1] * scale)
    panel = np.zeros((dh + 60, dw + 60, 3), np.uint8)  # black bezel 30 px
    panel[30:-30, 30:-30] = (62, 64, 72)
    small = cv2.resize(content, (int(IMG * scale), int(IMG * scale)), interpolation=cv2.INTER_AREA)
    cx, cy = 30 + dw // 2, 30 + dh // 2
    s = small.shape[0]
    panel[cy - s // 2 : cy - s // 2 + s, cx - s // 2 : cx - s // 2 + s] = small
    # Place the panel in the photo with a perspective tilt about the vertical axis.
    ph, pw = panel.shape[:2]
    src = square_quad(0, 0, pw, ph)
    shrink = 1 - 0.5 * math.sin(math.radians(tilt_deg))
    ox, oy = (w - pw) / 2, (h - ph) / 2
    dst = np.array(
        [[ox, oy], [ox + pw * shrink, oy + 40 * shrink], [ox + pw * shrink, oy + ph - 40 * shrink], [ox, oy + ph]],
        np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(panel, H, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), H, (w, h)) > 127
    photo[mask] = warped[mask]
    noise = rng.normal(0, 2, photo.shape)
    photo = np.clip(photo.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    truth = transform_quad(square_quad(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2), H)
    return photo, truth


def _contents():
    rng = np.random.default_rng(1)
    flat = np.full((IMG, IMG, 3), (128, 128, 128), np.uint8)
    noise = np.repeat(rng.integers(0, 256, (IMG, IMG, 1)), 3, axis=2).astype(np.uint8)
    navy = np.full((IMG, IMG, 3), (20, 40, 120), np.uint8)
    cv2.putText(navy, "OPEN", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
    cv2.line(navy, (40, 440), (470, 440), (255, 255, 255), 6)  # strong interior edge near the bottom
    return {"flat": flat, "noise": noise, "navy_text": navy}


@pytest.mark.parametrize("tilt", [0.0, 30.0, 45.0])
@pytest.mark.parametrize("kind", ["flat", "noise", "navy_text"])
def test_locate_recovers_image_quad(kind, tilt):
    photo, truth = render_scene(_contents()[kind], tilt)
    loc = locate_screen_image(photo, resolution=RES, image_px=IMG)
    assert loc is not None and loc.ok
    err = np.linalg.norm(loc.image_quad - truth, axis=1)
    assert err.max() < 3.0, err


def test_rectify_matches_content():
    # Smooth content: the scene down-scales it ~0.45x and rectify resamples again, so white
    # noise would not survive; a blurred field must come back nearly unchanged.
    content = cv2.normalize(cv2.GaussianBlur(_contents()["noise"], (0, 0), 6), None, 0, 255, cv2.NORM_MINMAX)
    photo, truth = render_scene(content, 30.0)
    rect = rectify(photo, truth, IMG)
    a = cv2.resize(rect, (64, 64), interpolation=cv2.INTER_AREA).astype(float)
    b = cv2.resize(content, (64, 64), interpolation=cv2.INTER_AREA).astype(float)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    assert corr > 0.95


def test_order_quad_is_tl_tr_br_bl():
    q = order_quad(np.array([[10, 90], [90, 90], [90, 10], [10, 10]]))
    np.testing.assert_array_equal(q, [[10, 10], [90, 10], [90, 90], [10, 90]])


def test_pose_recovery_on_projected_squares():
    wh = (8192, 6144)
    k = camera_matrix(wh, 24.0)
    r = 0.1405 / 2
    obj = np.array([[-r, r, 0], [r, r, 0], [r, -r, 0], [-r, -r, 0]])
    rng = np.random.default_rng(0)
    for yaw, pitch, dist, ox in itertools.product([0, 30, 45, -30], [0, -20], [0.3, 1.0], [0, 0.1]):
        r_pitch = cv2.Rodrigues(np.array([math.radians(pitch), 0, 0.0]))[0]
        r_yaw = cv2.Rodrigues(np.array([0, math.radians(yaw), 0.0]))[0]
        rot = r_pitch @ r_yaw
        rot_cam = np.diag([1, -1, -1.0]) @ rot
        t = np.array([ox, 0, dist])
        proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(rot_cam)[0], t, k, None)
        quad = proj.reshape(4, 2).astype(np.float32) + rng.normal(0, 1.0, (4, 2)).astype(np.float32)
        cam = -rot_cam.T @ t
        x, y, z = cam / np.linalg.norm(cam)
        pose = estimate_pose(quad, wh, 24.0, 140.5)
        assert abs(pose.distance_m - np.linalg.norm(t)) / np.linalg.norm(t) < 0.01
        assert abs(pose.angle_deg - math.degrees(math.acos(abs(z)))) < 1.5
        assert abs(pose.yaw_deg - math.degrees(math.atan2(x, abs(z)))) < 1.5
        assert abs(pose.pitch_deg - math.degrees(math.atan2(y, math.hypot(x, z)))) < 1.5
        assert pose.reprojection_px < 2.0


def test_parse_capture_name():
    n = parse_capture_name("photo_003__screen__d0.5__a30__room.jpg")
    assert n == CaptureName("photo_003", "screen", 0.5, 30.0, "room", False)
    assert parse_capture_name("flat_02_mid_grey__print__d1__a-15__led__m1.jpg.jpg").marked
    assert n.stem() == "photo_003__screen__d0.5__a30__room"
    with pytest.raises(ValueError):
        parse_capture_name("IMG_1234.jpg")
