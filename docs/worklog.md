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

**Phase 2 started the same day (user: "we should move forward").**
- Task 14: DCT/YCbCr moved to `core/transforms` (shared by JPEG and the classical marker);
  `markers/classical/dct` adds the luma view (Y in 0–255, DC = 8·mean) and the mid band
  (u+v ∈ [3, 8] = 37 coefficients).
- Task 15: Watson model. Read Watson 1993 from source (SPIE 1913): a_T = 0.649, w = 0.7,
  w₀₀ = 0, c̄₀₀ = 1024, reference condition **32 px/deg, L₀ = 65 cd/m²**. The frequency table
  is Cox's Table 7.2 (Peterson's method). I could not reproduce it from memory of the
  Ahumada–Peterson constants, so instead the table is used exactly at the reference and the
  A–P log-parabola is *fitted* to it (t_min 0.95, f_min 2.75 cyc/deg, K 1.52, r 0.62; rms 11 %)
  and applied as a ratio to move to our pinned 96 DPI @ 60 cm (39.6 px/deg): thresholds rise
  ~10 % at low and ~70 % at the highest frequencies. Slack maps eyeballed: match intuition.
- **Design note for Task 17 `capacity()`:** Watson thresholds are per single basis function.
  A flat mid-grey block still has ~194 units of mid-band slack, but spending it across 37
  coefficients pools to a visible error (Watson's β = 4 Minkowski pooling). Capacity must
  budget the *pooled* JND, not the per-coefficient sum — otherwise flat images look markable.

- Task 16: BCH(127,64) via `galois` (t = 10). ECC lives in the **harness**, not the marker
  (`EccConfig`), mirroring StegaStamp; markers see 127 raw channel bits.
- Task 17: `ClassicalSSMarker` (keyed chip plan, pilot slot, slack-normalised correlator,
  provisional capacity + refusal). Test sizes had to grow to 192–256 px: processing gain needs
  chips (64 px = 18 chips/bit, z ≈ 3).
- Task 18: two sweeps (4,824 + 2,412 rows, ~13 min on the 3050), four figures committed under
  `docs/results/figures/`, writeup `docs/results/week2_classical.md`.

**Week-2 result (Checkpoint W2):** classical baseline is perfect on the digital channel and
dead on both camera channels at every severity. Cause: geometric sync — 1° tilt → 17 % BER,
3° → no recovery, oracle rectify → BER 0. Two findings beyond the brief's prediction: (a) at
Watson threshold a mark on a flat image is below the JPEG quantisation step and vanishes at
Q75, while photos survive Q25 — the budget must be channel-aware; (b) one Watson JND per
coefficient pools to LPIPS 0.4–0.6 on flat/gradient images; the invisible operating point is
s ≈ 0.1–0.25 where host interference already costs 3–5 % BER. Control set: 0 false
recoveries in 2,412 trials, pilot z ~ N(0,1) as designed.

**Audit before Week 3 (user request: verify, remove waste, check metrics fit the phase).**
Tooling: `ruff` (E,F,W,I,B,UP,SIM) and `vulture`. Findings and actions:
- `runner.py` inner closure captured loop variables (B023) — correct today, fragile by design;
  refactored to an explicit `_Trial` dataclass + `_run_trial()`.
- `_uniform` was a private helper imported across five modules → public `distort.base.uniform`.
- `smoke.yaml` `limit: 8` selected eight *flat* images (manifest order). Added
  `CorpusConfig.per_class`; smoke now takes 2 of each class.
- Dead code removed: `fidelity.is_finite_psnr`. `Crop` kept (tested; reserved for Week-5
  partial-capture work) and marked as such. Other `vulture` hits are dataclass fields, protocol
  methods and extension APIs — false positives.
- `zip(strict=…)` made explicit in three places; import order normalised (17 files).
- Metrics vs phase: §6 set is complete for Phases 1–3. Two gaps Week 2 exposed, now closed:
  `capacity_bits` column (the flat-image capacity failure was invisible in the CSV) and a
  per-run `summary.json` (per chain × severity: BER, recovery, control BER/recovery, AUC,
  TPR@1 % FPR, fidelity, timings) so Task 22's classical-vs-learned comparison is file-based.
- Known limitation recorded: random draws differ between CPU and CUDA generators, so a run is
  bit-reproducible on the same device type only. Every row still carries its seed and params.

**Branch rule (from here on):** `main` is always green. Integrations of foreign code and
anything hard to revert go on a branch (`week3-videoseal` first) and merge when the suite
passes there. Result writeups only reference runs made from committed `main` hashes.

**Week 3 (branch `week3-videoseal`, Tasks 20–22, Checkpoint W3).**
- Video Seal 1.0.1 from PyPI as an optional extra; `decord` overridden via `[tool.uv]
  override-dependencies`; `requests` added (used, undeclared). Its loader is cwd-relative, so the
  adapter builds the model itself from the installed card + a vendored `attenuation.yaml`, with
  the checkpoint cached under `~/.cache/perceptual_media/videoseal/`.
- The image card's column 0 is not a trained presence channel (≈ 0.1 marked or not); presence
  score = mean |message logit| (≈ 11 vs ≈ 0.4). Recorded in the adapter docstring.
- ECC generalised: `ChannelCode` (inner BCH + repetition + zero-pad) so the 64-bit message and
  BCH(127,64) are identical across a 127-bit and a 256-bit channel (2 soft-combined copies).
- `pm-compare` overlays runs from `summary.json`; `read_results` tolerates columns added after
  a run was written (schema evolution) so Week-2 CSVs still load.
- Result (`docs/results/week3_learned.md`): perspective cliff 1° → ~15° (recovery 96 % at 3°,
  37 % at 15 °, 0 at 30 °); both camera chains dead at severity ≥ 0.5; PSNR 45.4 / LPIPS 0.003 on
  photos reproduces the published 45.75 / 0.003; flat-image JPEG failure recurs (49 % at Q75)
  from a second, independent perceptual attenuation; procedural textures are out of
  distribution (13–24 % BER undistorted). 0 false recoveries in 3,015 control trials.

**Next:** merge `week3-videoseal` → `main`; Phase 4 / Week 4 (physical capture) — Task 23
capture protocol + logger, Task 24 print sheets, Task 25 ingestion, Task 26 2AFC tool. The user
will supply phone captures of six named corpus images once Phase 3 is closed.

## 2026-09-19 — Week 4 (branch `week4-capture`, Tasks 23–26)

**Capture session (user).** BenQ 24" 1080p, Nothing Phone (2a) main camera, office room light,
tap-to-expose on the image, corpus images at 1:1 in a dark full-screen viewer. Calibration set
`screen_calib_20260919`: 6 unmarked images × {0.5 m × 0°/30°/45°, 1 m × 0°} = 24 photos
(8192×6144). Distances/angles by eye; panel brightness deliberately unchanged (vignette centre
clips). Copied from `Downloads/captures` to the external root.

**Task 23** — `capture/{naming,locate,log}.py`, `pm-capture log`. Monitor-as-fiducial locator:
uniform-region panel detection → centred-square prediction → surround-gated edge snap →
square-constraint completion; PnP pose (ITERATIVE; IPPE_SQUARE on OpenCV 5.0 returned
non-reprojecting solutions). Log is derived from filename + EXIF + geometry, nothing typed.
24/24 located, reprojection < 5 px; nominal 0/30/45° measured 2–5 / 29–34 / 45–49°.

**Task 25a** — `capture/channel.py`, `pm-capture channel`, `screen_channel` preset (screen chain
minus perspective). Per-capture tone curve (an affine colour map under-models the phone by
4–8 dB), blur σ, noise / moiré / shading on ≥ 90 %-flat references, sim severity match.
Result `docs/results/week4_channel.md`: head-on 0.5 m moiré (amp 8–9, residual 22–25) is beyond
the simulator at any severity (≤ 4 / 2.4) — Bayer CFA vs subpixels at 3.7 px/px, a regime the
`Moire` stage does not sample; 1 m head-on ≈ severity 0.25; blur grows with angle 0.6 → 2.1 px.
The real channel is per-mechanism, not one severity. Two of six images clip in capture.

**Task 24** — `capture/sheet.py`, `pm-capture sheet`: display sheet (screen variant of the print
sheet), 18 slides (`decode_20260919`: Video Seal on 12 images, classical on the 6 calibration
images), ArUco fiducials with quiet zones, `sheet.csv` with the exact message per slide.
**Task 25b** — `capture/ingest.py`, `pm-capture decode`: fiducial homography → rectify → decode,
plus unrectified crop (sync gap) and control rows; ordinary run dir. Dry run on synthetic photos
of the real slides: Video Seal recovers the photo slide at 17° and 35°, classical fails after
rectification (1–3 px residual registration breaks the block grid) — as the 1° cliff predicts.
**Task 26** — `capture/afc.py`, `pm-capture afc serve|report`: 2AFC page at 1:1 device pixels,
exact binomial intervals. Verified in Chrome.

**Next (needs the user):** shoot the decode round (`docs/capture_protocol.md`, 38–54 shots) and
run the 2AFC with 5 people; then Task 27 report + Checkpoint W4. Candidate simulator fixes if
Branch A: CFA-aware moiré at 2–4 px/px, a sharpening stage, tone-curve illumination.
