"""Capture filenames carry the *nominal* conditions so a phone's camera roll can be logged
without a separate form:

    <image_id>__<medium>__d<distance_m>__a<angle_deg>__<lighting>[__<marked>].jpg

e.g. ``photo_003__screen__d0.5__a30__room.jpg`` or ``photo_003__print__d1__a0__led__m1.jpg``.
``medium`` ∈ {print, screen}; ``marked`` is ``m1`` (marked) / ``m0`` (control) and defaults to
control when absent (the Week-4 channel-calibration set is unmarked corpus images).
Measured conditions (from geometry / EXIF) are logged separately — see ``capture.log``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MEDIA = ("print", "screen")

_PATTERN = re.compile(
    r"^(?P<image_id>[A-Za-z0-9_]+?)__(?P<medium>print|screen)"
    r"__d(?P<distance>\d+(?:\.\d+)?)__a(?P<angle>-?\d+(?:\.\d+)?)__(?P<lighting>[A-Za-z0-9-]+)"
    r"(?:__m(?P<marked>[01]))?$"
)


@dataclass(frozen=True)
class CaptureName:
    """Nominal conditions parsed from a capture filename."""

    image_id: str
    medium: str
    distance_m: float
    angle_deg: float
    lighting: str
    marked: bool = False

    def stem(self) -> str:
        m = f"__m{int(self.marked)}" if self.marked else ""
        return (
            f"{self.image_id}__{self.medium}__d{_fmt(self.distance_m)}"
            f"__a{_fmt(self.angle_deg)}__{self.lighting}{m}"
        )


def _fmt(x: float) -> str:
    return f"{x:g}"


def parse_capture_name(path: str | Path) -> CaptureName:
    """Parse the nominal conditions from a capture filename; raises ``ValueError`` if malformed."""
    stem = Path(path).name
    # Tolerate a doubled extension (``x.jpg.jpg``) from phone export tools.
    while Path(stem).suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".dng"}:
        stem = Path(stem).stem
    m = _PATTERN.match(stem)
    if m is None:
        raise ValueError(
            f"capture filename {Path(path).name!r} does not match "
            "<image_id>__<print|screen>__d<m>__a<deg>__<lighting>[__m<0|1>]"
        )
    return CaptureName(
        image_id=m["image_id"],
        medium=m["medium"],
        distance_m=float(m["distance"]),
        angle_deg=float(m["angle"]),
        lighting=m["lighting"],
        marked=m["marked"] == "1",
    )
