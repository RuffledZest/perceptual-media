"""Embedding schemes. Importing this package registers the built-in markers with ``get_marker``."""

import contextlib

from perceptual_media.markers import (
    base as base,  # noqa: F401  (NullMarker registers here)
)
from perceptual_media.markers.classical import (
    spread_spectrum as _spread_spectrum,  # noqa: F401
)

with contextlib.suppress(ImportError):  # optional extra: `uv sync --extra videoseal`
    from perceptual_media.markers.videoseal import adapter as _videoseal  # noqa: F401
