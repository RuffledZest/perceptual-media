"""Perspective (homography) warp. Differentiable via ``grid_sample``.

Two sampling modes:

* ``"pose"`` (default, physically motivated): the image is a plane viewed by a pinhole camera
  with field of view ``fov_deg``; sample tilt/yaw ``~ U(−max_angle, max_angle)`` and roll
  ``~ U(−max_roll, max_roll)``, project the plane's corners, and re-fit the projected quad to
  the original frame (so foreshortening is modelled, overall scale is left to ``Resize``).
* ``"corners"`` (StegaStamp): every corner jittered independently by ``U(−d, d)`` in x and y,
  ``d = jitter × width`` (StegaStamp ``rnd_trans = 0.1``).

Coordinates are pixel coordinates with the origin at the top-left corner. ``last_H`` maps
source pixels to destination pixels; ``inverse_warp`` applies ``last_H⁻¹`` (oracle rectification,
for measuring the synchronisation gap).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from perceptual_media.core.types import ImageBatch
from perceptual_media.distort.base import Distortion, uniform

Mode = Literal["pose", "corners"]


def homography_from_corners(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """DLT: ``(4, 2)`` source and destination points → ``(3, 3)`` H with ``dst ~ H @ src``, H[2,2] = 1."""
    rows = []
    for (x, y), (u, v) in zip(src.tolist(), dst.tolist(), strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = torch.tensor(rows, dtype=torch.float64)
    b = dst.to(torch.float64).reshape(-1)
    h = torch.linalg.solve(a, b)
    return torch.cat([h, torch.ones(1, dtype=torch.float64)]).reshape(3, 3).to(torch.float32)


def warp(x: ImageBatch, H: torch.Tensor, fill: float = 0.5) -> ImageBatch:
    """Warp ``x`` so that source pixel ``p`` lands at ``H @ p``; outside the source, ``fill``."""
    b, _, h, w = x.shape
    dev, dt = x.device, x.dtype
    ys = torch.arange(h, device=dev, dtype=dt) + 0.5
    xs = torch.arange(w, device=dev, dtype=dt) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(gx)
    dst = torch.stack([gx, gy, ones], dim=-1).reshape(-1, 3)  # destination pixel centres
    Hinv = torch.linalg.inv(H.to(dev, torch.float64)).to(dt)
    src = dst @ Hinv.T
    src = src[:, :2] / src[:, 2:3]
    # pixel centre coordinates → grid_sample normalised coordinates (align_corners=False)
    gxn = src[:, 0] / w * 2 - 1
    gyn = src[:, 1] / h * 2 - 1
    grid = torch.stack([gxn, gyn], dim=-1).reshape(1, h, w, 2).expand(b, h, w, 2)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = F.grid_sample(torch.ones(1, 1, h, w, device=dev, dtype=dt), grid[:1], mode="bilinear", padding_mode="zeros", align_corners=False)
    return out + fill * (1 - mask)


def pose_corners(w: int, h: int, tilt_deg: float, yaw_deg: float, roll_deg: float, fov_deg: float) -> torch.Tensor:
    """Destination corners (TL, TR, BR, BL) of the image plane seen under the given camera pose,
    re-fitted so the projected quad's bounding box is centred and fills the frame."""
    tx, ty, tz = (math.radians(a) for a in (tilt_deg, yaw_deg, roll_deg))
    rx = torch.tensor([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]], dtype=torch.float64)
    ry = torch.tensor([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]], dtype=torch.float64)
    rz = torch.tensor([[math.cos(tz), -math.sin(tz), 0], [math.sin(tz), math.cos(tz), 0], [0, 0, 1]], dtype=torch.float64)
    R = rz @ ry @ rx
    aspect = w / h
    pts = torch.tensor([[-aspect, -1, 0], [aspect, -1, 0], [aspect, 1, 0], [-aspect, 1, 0]], dtype=torch.float64)
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    cam = pts @ R.T + torch.tensor([0, 0, f * max(aspect, 1.0)], dtype=torch.float64)  # plane centred on the axis
    proj = cam[:, :2] / cam[:, 2:3]
    # re-fit: centre the bounding box and scale so it matches the frame's extent
    lo, hi = proj.min(0).values, proj.max(0).values
    proj = (proj - (lo + hi) / 2) / ((hi - lo) / 2)  # bbox → [-1, 1]²
    return torch.stack([(proj[:, 0] + 1) / 2 * w, (proj[:, 1] + 1) / 2 * h], dim=1).to(torch.float32)


class Perspective(Distortion):
    name = "perspective"

    def __init__(self, max_angle: float = 45.0, max_roll: float = 10.0, fov_deg: float = 60.0, mode: Mode = "pose", jitter: float = 0.1, fill: float = 0.5, fixed: bool = False) -> None:
        """``fixed=True`` (pose mode only): tilt = ``max_angle`` exactly, yaw = roll = 0 — for
        BER-vs-angle curves with no sampling noise."""
        super().__init__()
        self.fixed = fixed
        if max_angle < 0 or max_roll < 0 or jitter < 0:
            raise ValueError("ranges must be >= 0")
        if not 0 < fov_deg < 180:
            raise ValueError("fov_deg must be in (0, 180)")
        self.max_angle, self.max_roll, self.fov_deg, self.mode, self.jitter, self.fill = max_angle, max_roll, fov_deg, mode, jitter, fill
        self.last_H: torch.Tensor | None = None

    def _distort(self, x: ImageBatch, gen: torch.Generator) -> ImageBatch:
        h, w = x.shape[-2:]
        src = torch.tensor([[0, 0], [w, 0], [w, h], [0, h]], dtype=torch.float32)
        if self.mode == "pose":
            if self.fixed:
                tilt, yaw, roll = float(self.max_angle), 0.0, 0.0
            else:
                tilt, yaw, roll = uniform(-self.max_angle, self.max_angle, gen), uniform(-self.max_angle, self.max_angle, gen), uniform(-self.max_roll, self.max_roll, gen)
            dst = pose_corners(w, h, tilt, yaw, roll, self.fov_deg)
            self.last_params = {"mode": "pose", "tilt_deg": tilt, "yaw_deg": yaw, "roll_deg": roll, "fov_deg": self.fov_deg}
            if tilt == 0 and yaw == 0 and roll == 0:
                self.last_H = torch.eye(3)
                return x
        else:
            d = self.jitter * w
            dst = src + torch.tensor([[uniform(-d, d, gen), uniform(-d, d, gen)] for _ in range(4)], dtype=torch.float32)
            self.last_params = {"mode": "corners", "jitter_px": d, "corners": dst.tolist()}
            if d == 0:
                self.last_H = torch.eye(3)
                return x
        self.last_H = homography_from_corners(src, dst)
        return warp(x, self.last_H, self.fill).clamp(0, 1)

    def inverse_warp(self, y: ImageBatch) -> ImageBatch:
        """Oracle rectification: undo the last warp (content lost outside the frame stays lost)."""
        if self.last_H is None:
            raise RuntimeError("no warp has been applied yet")
        return warp(y, torch.linalg.inv(self.last_H.to(torch.float64)).to(torch.float32), self.fill).clamp(0, 1)
