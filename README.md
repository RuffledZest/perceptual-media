# perceptual-media

Research code for **perceptually lossless, camera-readable media**: transform a still image so a
human cannot tell it from the original, yet a commodity phone camera photographing it (print or
screen, at an angle, uncontrolled lighting) recovers a 64–256 bit payload. This is a
*communication* system, not a provenance watermark. The core contribution is honest per-image
capacity estimation with graceful refusal. See `claude_markdowns/PROJECT_BRIEF.md` for the full
brief and `tasks/plan.md` for the build plan.

```
uv sync
uv run pytest
```
