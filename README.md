# perceptual-media

Research code for **perceptually lossless, camera-readable media**: transform a still image so a
human cannot tell it from the original, yet a commodity phone camera photographing it (print or
screen, at an angle, uncontrolled lighting) recovers a 64–256 bit payload. This is a
*communication* system, not a provenance watermark. The core contribution is honest per-image
capacity estimation with graceful refusal. See `claude_markdowns/PROJECT_BRIEF.md` for the full
brief and `tasks/plan.md` for the build plan.

```
uv sync                                        # deps (CUDA torch on Windows via cu130 index)
uv run pytest                                  # test suite
uv run pm-corpus build                         # build data/corpus (synthetic + photos from configs/paths.yaml)
uv run pm-run configs/experiments/smoke.yaml   # run an experiment -> outputs/<name>-<stamp>/results.csv
uv run pm-plot outputs/<run>                   # BER-vs-severity, ROC, robustness-imperceptibility frontier
```

Layout: `src/perceptual_media/{core,markers,distort,metrics,harness}`, configs in `configs/`,
external data paths in `configs/paths.yaml`. Progress: `tasks/todo.md`; decisions: `docs/worklog.md`; results: `docs/results/`.

Workflow: `main` is always green (`uv run pytest`, `ruff check`); integrations and hard-to-revert
work happen on a branch and merge when the suite passes there.
