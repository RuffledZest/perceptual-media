# Worklog

Dated journal of decisions, deviations and checkpoints. One entry per working session.
The detailed record of *what changed* is `git log`; this is the record of *why*.
Task numbers refer to `tasks/plan.md`.

## 2026-09-18 — Skeleton, plan, Tasks 0–6, Checkpoint A

**Setup / decisions**
- Skeleton audited against the pre-Claude-Code checklist; added the missing `.gitignore`,
  `tests/`, `data/`, `outputs/`. Package name kept as `perceptual_media` (not `pmc`).
- Plan written (`tasks/plan.md`, `tasks/todo.md`) with 10 architecture decisions. Approved.
- Open questions resolved: corpus photos from Kaggle `arnaud58/landscape-pictures` (external,
  4,319 JPEGs); CUDA switched now (`torch 2.14.0+cu130`, RTX 3050 verified); captures path at
  the external root (**not yet backed up — fix before Week 4**); Watson viewing condition
  pinned to 96 DPI @ 60 cm because the 2AFC study will be run on-screen.
- Reviewed StegaStamp (`models.py`, `train.py`, `utils.py`) and PIMoG (`Noise_Layer.py`) source
  → `docs/research/stegastamp_pimog_notes.md`. Concrete distortion ranges now feed Tasks 10–12.
- Literature review (`claude_markdowns/compass_artifact_*.md`) is consistent with the brief;
  carried one recommendation: print-to-camera is the primary physical preset.
- GitHub remote connected; every task is pushed on completion.

**Deviations from plan (all recorded in commit messages)**
- Task 1: helpers named `to_uint8`/`from_uint8` instead of `to_numpy_uint8`/`from_numpy_uint8`.
- Task 3: added a `get_marker`/`register_marker` registry and `random_payload` (needed by Task 6).
- Task 6: pulled the `Distortion`/`DistortionChain`/`Identity` contract and `build_chain`
  registry forward from Task 9 so the runner has a real chain to call. Task 9 now only adds
  Noise/Resize/Crop. `smoke.yaml` is identity-only until Task 12 adds channel presets.

**Bugs caught**
- Task 6: distortion hook was named `_apply`, shadowing `nn.Module._apply` used by `.to(device)`.
  GPU run failed immediately; renamed to `_distort`. Tests now exercise the device move.
- Task 1: PIL returns read-only arrays; `from_uint8` copies (`np.array(..., order="C")`).

**Checkpoint A** (end of Task 6): `pm-run configs/experiments/smoke.yaml` → 12 rows
(6 marked + 6 control), BER 0.52 / 0.48, AUC 0.58, 3 s on GPU, all §6 columns present.
68 tests, ~3 s, warnings-as-errors clean.

**Tasks 7–13 (same day, continued with blanket approval through Phase 1)**
- Task 7: SSIM implemented to match skimage defaults (1e-7); LPIPS via `lpips` (AlexNet, cached,
  <32 px inputs upsampled). Fidelity is marked-vs-original, before the channel.
- Task 8: corpus = 9 flat + 5 gradient + 5 text + 8 textured (synthetic, seeded) + 40 Kaggle
  photos; byte-identical rebuilds. `face` class deferred (no folder). Class `photo` added
  alongside the brief's four so natural vs procedural texture can be separated in plots.
- Task 10: differentiable JPEG default is a straight-through estimator, not StegaStamp's
  `round_only_at_0` — measured 46–55 dB vs PIL versus 38–52 dB (notes updated). Q100 is a
  passthrough so severity-0 chains are exact identity.
- Task 11: perspective primary mode samples camera pose (tilt/yaw/roll, 60° FOV) and re-fits
  the quad; StegaStamp corner jitter is `mode="corners"`. Oracle `inverse_warp` available.
- Task 12: moiré is a real sampling model (display stripes + matrix → PSF → sensor grid); beat
  frequency verified against `|1 − round(1/s)·s|` within 1 %. Presets at severity 1 cover
  StegaStamp's training ranges by construction.
- Task 13: three §6 figures from CSV only; fixed class→colour mapping (dataviz palette).

**Bugs / fragile tests fixed:** JPEG Q100 test expected losslessness through 4:2:0 (wrong
expectation, PIL loses the same); severity-monotonicity test needed averaging over seeds;
noise-std test needed to compare against the sampled σ.

**Checkpoint W1 (end of Task 13):** `pm-corpus build` → `pm-run null_sweep.yaml` (67 images ×
9 chains × marked+control = 1,206 rows, 33 s on the RTX 3050) → `pm-plot` runs unattended.
NullMarker: BER 0.50, AUC 0.50 on all chains. 147 tests in 14 s. Severity-0.5 samples reviewed
visually: plausible phone photos (foreshortening, softening, colour shift, screen grid texture).

**Next:** Phase 2 / Week 2 — Task 14 (block DCT utils) → Task 15 (Watson slack) → BCH →
classical spread-spectrum marker → Week-2 sweep and failure writeup.
