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

**Next:** Task 7 (PSNR / SSIM / LPIPS).
