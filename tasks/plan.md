# Implementation Plan: Perceptual Media — Phase 1 (Weeks 1–6)

Source of truth for scope: `claude_markdowns/PROJECT_BRIEF.md`. This plan turns §5 (build plan)
and §6 (harness spec) into ordered, verifiable tasks. Weeks 1–2 are fully specified. Weeks 3–4
are medium resolution. Weeks 5–6 are two gated branches chosen by Week-4 results.

Ordering principle from the brief: **build the measurement apparatus before the thing being
measured.** The first end-to-end slice (Checkpoint A) is a *null* marker running through the
full harness, so every later component lands into a working pipeline.

---

## Overview

A Python/PyTorch research harness that (1) embeds a 64–256 bit payload into a still image under
a per-image perceptual budget, (2) simulates the print/screen → phone-camera channel, (3) decodes,
and (4) emits one result row per trial in the exact schema of brief §6, including an unmarked
control set for honest false-positive accounting. Week 2 adds a classical DCT spread-spectrum
baseline with a Watson JND budget map. Week 3 adds a learned baseline (Video Seal). Week 4 is
physical capture. Weeks 5–6 attack whichever constraint bound us.

## Architecture Decisions

These are made now so they don't happen to us later. Change them deliberately, not by drift.

1. **Tensor convention.** All images are `torch.float32`, range `[0, 1]`, shape `(B, 3, H, W)`,
   RGB. NumPy / PIL / OpenCV appear only at the I/O edges (`core/io.py`). Batched everywhere so
   distortions are differentiable and vectorised from day one.
2. **`Marker` protocol is the central API contract.** Every embedding scheme — null, classical,
   Video Seal adapter, our own — implements:
   `embed(img, payload, strength) -> img`, `decode(img) -> DecodeResult(llrs, score)`,
   `capacity(img) -> bits | None`. `decode` returns *soft* per-bit LLRs and a scalar detector
   score, never hard bits (brief §4.5: do not hard-slice early). `capacity` exists from day one
   even when stubbed — it is the refusal hook and the project's core contribution.
3. **Distortions are `nn.Module`s composed by a `DistortionChain`.** Each takes an explicit
   `torch.Generator`, exposes its sampled params for logging, and is differentiable where physics
   permits. Named chains live in `configs/sim/*.yaml` (`print_camera`, `screen_camera`).
4. **One `ResultRow` dataclass mirrors brief §6 exactly.** Every experiment appends rows to
   `outputs/<run>/results.csv` next to the resolved config and git hash. Plots are derived from
   CSV only — never from in-memory state — so any figure is reproducible from a file.
5. **Config = YAML → plain dataclasses.** No Hydra/OmegaConf. Solo research iteration does not
   need a config framework; it needs a file that can be diffed.
6. **Determinism.** `seed_everything(seed)` plus explicit generators. Every row records its seed.
7. **Corpus is generated, not curated.** Hard cases (flat fills, gradients, rendered text,
   procedural textures) are synthesised by code from a seed so `data/` is fully reproducible.
   Photographic / face images come from a user-supplied folder referenced in the manifest.
8. **CPU torch for Weeks 1–2, CUDA build before Week 3.** RTX 3050 6 GB is enough for inference
   and small fine-tunes, not for training a StegaStamp-class model from scratch.
9. **Physical captures live outside the repo** (path set in config, see Open Questions). They
   are the only non-regenerable asset.
10. **Package layout (`src/perceptual_media/`):**
    ```
    core/      types, seed, config, io
    markers/   base (Marker, DecodeResult, NullMarker), classical/, videoseal/
    distort/   base, basic, jpeg, blur, illumination, perspective, moire, presets
    metrics/   fidelity (PSNR/SSIM/LPIPS), decoding (BER/recovery/ROC)
    corpus/    synth, manifest, build (CLI)
    harness/   results, runner, plots, cli
    capture/   (Week 4) logging, print sheets, ingestion, 2AFC
    ```

## Definition of Done (applies to every task)

- `uv run pytest` passes; new code has tests on small synthetic tensors (CPU, < 2 s each).
- Public functions have type hints and a one-line docstring stating tensor shape/range.
- No image files, `data/`, or `outputs/` committed.
- Any decision that deviates from this plan is noted in the task's commit message.

---

## Task List

### Phase 0 — Finish skeleton

#### Task 0: Add harness dependencies and make the first commit
**Description:** Add the Week-1 dependencies the skeleton is missing and land the initial commit
so `.gitignore` is in effect before any generated data exists.

**Acceptance criteria:**
- [x] `uv add lpips scikit-image pyyaml pandas` succeeds on Python 3.13
- [x] `uv run python -c "import lpips, skimage, yaml, pandas"` exits 0
- [x] First commit contains: `.gitignore`, `pyproject.toml`, `uv.lock`, `README.md`, `src/`, `tests/`, `tasks/`, `claude_markdowns/`, `data/.gitkeep`, `outputs/.gitkeep`

**Verification:** `uv run pytest` green; `git log --oneline | wc -l` == 1; `git status` clean.
**Dependencies:** None. **Files:** `pyproject.toml`, `uv.lock`. **Scope:** XS

---

### Phase 1 — Week 1: Harness

#### Task 1: Core conventions — types, seeding, image I/O
**Description:** Establish the tensor convention and the two utilities everything else imports.

**Acceptance criteria:**
- [x] `core/types.py` documents the `(B,3,H,W) float32 [0,1]` convention and exposes `ImageBatch`, `Payload` aliases and an `assert_image_batch()` validator
- [x] `core/seed.py::seed_everything(seed)` makes two runs of `torch.rand(4)` + `np.random.rand(4)` + `random.random()` identical
- [x] `core/io.py::load_image(path) -> (1,3,H,W)` and `save_image(t, path)` round-trip with max abs error ≤ 1/255; `to_uint8` / `from_uint8` helpers

**Verification:** `uv run pytest tests/core -q`
**Dependencies:** Task 0. **Files:** `core/{types,seed,io}.py`, `tests/core/test_{seed,io}.py`. **Scope:** S

#### Task 2: Config loading (YAML → dataclasses)
**Description:** Typed experiment configs with defaults and strict unknown-key errors.

**Acceptance criteria:**
- [x] `core/config.py::load_config(path, schema=ExperimentConfig)` returns a nested dataclass
- [x] Unknown keys raise `ConfigError` naming the key; missing required keys raise naming the key
- [x] `dump_config(cfg, path)` writes YAML that reloads to an equal object

**Verification:** `uv run pytest tests/core/test_config.py -q`
**Dependencies:** Task 1. **Files:** `core/config.py`, `tests/core/test_config.py`, `configs/experiments/smoke.yaml`. **Scope:** S

#### Task 3: `Marker` protocol, `DecodeResult`, `NullMarker`
**Description:** Define the API contract every embedding scheme implements, plus a null
implementation that makes the unmarked-control path and end-to-end testing possible before any
real marker exists.

**Acceptance criteria:**
- [x] `markers/base.py` defines `Marker` (Protocol) with `embed`, `decode`, `capacity`, `n_bits`
- [x] `DecodeResult` holds `llrs: (B, n_bits)` and `score: (B,)`; `.hard_bits()` slices at 0
- [x] `NullMarker.embed` is identity; `.decode` returns seeded random LLRs so BER ≈ 0.5 over 1000 bits (test asserts 0.45–0.55); `.capacity` returns `None`

**Verification:** `uv run pytest tests/markers/test_base.py -q`
**Dependencies:** Task 1. **Files:** `markers/base.py`, `tests/markers/test_base.py`. **Scope:** S

#### Task 4: `ResultRow` schema and `ResultWriter`
**Description:** The one row every experiment emits (brief §6), and a writer that creates a run
directory with results, resolved config, and git hash.

**Acceptance criteria:**
- [x] `ResultRow` fields, exactly: `run_id, seed, image_id, image_class, marked (bool), n_payload_bits, embed_strength, distortion_chain (str), distortion_params (json str), capture_conditions (json str, "" for sim), ber, payload_recovered (bool), detector_score, psnr, ssim, lpips, encode_ms, decode_ms`
- [x] `ResultWriter(run_dir)` appends rows to `results.csv` (flush every row so a crash keeps data), writes `config.yaml` and `git_hash.txt` on open
- [x] `pandas.read_csv` of 100 written rows yields 100 rows with bool/float dtypes preserved

**Verification:** `uv run pytest tests/harness/test_results.py -q`
**Dependencies:** Task 2. **Files:** `harness/results.py`, `tests/harness/test_results.py`. **Scope:** S

#### Task 5: Decoding metrics — BER, payload recovery, ROC / FPR
**Description:** The detection-theory half of the metrics module (brief §4.2). Pure NumPy/torch;
no sklearn.

**Acceptance criteria:**
- [x] `ber(llrs, payload)` = 0 for perfect, 1 for inverted, batched
- [x] `payload_recovered(llrs, payload)` bool per item (all bits correct after hard slice; ECC-aware variant added in Task 16)
- [x] `roc(scores_marked, scores_unmarked)` returns FPR/TPR arrays + AUC; perfectly separable → AUC 1.0; identical distributions → AUC ≈ 0.5
- [x] `fpr_at_tpr(…, tpr=0.95)` and `threshold_for_fpr(scores_unmarked, fpr=1e-3)` with a documented warning that 1e-3/frame is unusable at 30 fps (brief §4.2)

**Verification:** `uv run pytest tests/metrics/test_decoding.py -q`
**Dependencies:** Task 3. **Files:** `metrics/decoding.py`, `tests/metrics/test_decoding.py`. **Scope:** S

#### Task 6: Experiment runner + CLI (vertical slice with `NullMarker`)
**Description:** The loop: for each image × payload × strength × distortion config, embed (or not,
for control), distort, decode, time, emit a row. First version uses in-memory synthetic images and
an `Identity` distortion so it can run before the corpus and distortion modules exist.

**Acceptance criteria:**
- [x] `harness/runner.py::run_experiment(cfg) -> Path` writes a run dir via `ResultWriter`
- [x] Runs *both* `marked=True` and `marked=False` (control) rows for every image when `cfg.control=True` (default)
- [x] `uv run pm-run configs/experiments/smoke.yaml` (entry point replaces the hello-world `main`) completes in < 30 s on CPU and produces ≥ 20 rows with BER in 0.35–0.65 and non-zero `encode_ms`/`decode_ms`
- [x] Fidelity columns are `NaN` until Task 7 lands (documented)

**Verification:** `uv run pytest tests/harness/test_runner.py -q`; manual: inspect `outputs/smoke-*/results.csv`
**Dependencies:** Tasks 2, 3, 4, 5. **Files:** `harness/{runner,cli}.py`, `pyproject.toml` (scripts), `tests/harness/test_runner.py`. **Scope:** M

### Checkpoint A — end-to-end pipeline exists
- [x] `uv run pytest` green
- [x] `pm-run smoke.yaml` produces a CSV with marked + control rows, BER ≈ 0.5
- [x] Review: does `ResultRow` cover everything §6 asks for? Yes — all 13 §6 fields plus run_id/seed/marked; verified by test.

#### Task 7: Fidelity metrics — PSNR, SSIM, LPIPS
**Description:** The perceptual-distance half of the metrics module. Batched, returns per-item.

**Acceptance criteria:**
- [x] `psnr(a, b)`: identical → `inf` (returned as a large finite sentinel, documented); known value on a synthetic pair within 1e-3 dB
- [x] `ssim(a, b)`: matches `skimage.metrics.structural_similarity` on a 64×64 test image within 1e-3
- [x] `lpips(a, b)`: identical → ≤ 1e-6; model loaded once and cached; runs on CPU
- [x] Runner fills the three columns for marked rows (control rows: `NaN`)

**Verification:** `uv run pytest tests/metrics/test_fidelity.py -q`
**Dependencies:** Task 6. **Files:** `metrics/fidelity.py`, `harness/runner.py`, `tests/metrics/test_fidelity.py`. **Scope:** S

#### Task 8: Corpus builder — synthetic hard cases + manifest
**Description:** Generate the deliberately-hard image set (brief §5 W1, §10.3) reproducibly, plus
a manifest that assigns every image a class. Photographic and face images are ingested from a
user-supplied folder, not downloaded.

**Acceptance criteria:**
- [x] `corpus/synth.py` produces at 512×512: `flat` (≥6 solid fills spanning luminance and brand-like colours), `gradient` (linear + radial, ≥4), `text` (≥4 PIL-rendered document/poster layouts), `textured` (≥6 procedural: Perlin-style noise, stripes, checker, mixed-frequency)
- [x] `corpus/build.py` CLI `uv run pm-corpus build --seed 0 --out data/corpus [--photos DIR] [--faces DIR]` writes PNGs + `manifest.csv` (`image_id, image_class, source, path, width, height`)
- [x] Same seed → byte-identical PNGs (test hashes two builds)
- [x] `corpus/manifest.py::load_manifest()` returns rows; runner iterates the manifest instead of in-memory images

**Verification:** `uv run pytest tests/corpus -q`; manual: open `data/corpus/` and eyeball each class
**Dependencies:** Task 6. **Files:** `corpus/{synth,build,manifest}.py`, `harness/runner.py`, `pyproject.toml` (script), `tests/corpus/test_synth.py`. **Scope:** M

#### Task 9: Distortion base + basic distortions
**Description:** The `Distortion` module contract and the simplest members: Identity, GaussianNoise,
Resize (down→up), Crop (random rectangle, with pad-back so shapes stay fixed).

**Acceptance criteria:**
- [x] `distort/base.py`: `Distortion(nn.Module)` with `forward(x, gen) -> x` and `.last_params: dict`; `DistortionChain([...])` composes and merges params under each stage's name
- [x] Deterministic: same generator seed → identical output tensor
- [x] Gradient flows through Noise and Resize (`x.requires_grad_(); out.sum().backward()` gives non-None grad)
- [x] Runner logs `distortion_chain` name and `distortion_params` JSON per row

**Verification:** `uv run pytest tests/distort/test_basic.py -q`
**Dependencies:** Task 8. **Files:** `distort/{base,basic}.py`, `harness/runner.py`, `tests/distort/test_basic.py`. **Scope:** M

#### Task 10: JPEG and blur
**Description:** JPEG with chroma subsampling — a real PIL path for evaluation and a
differentiable approximation for later training — plus defocus (Gaussian) and motion blur.

**Acceptance criteria:**
- [x] `JPEG(quality, differentiable=False)` matches PIL output at that quality (4:2:0) exactly
- [x] `JPEG(differentiable=True)` (DCT + soft-rounding) is within PSNR ≥ 35 dB of the real path at Q=75 and passes the gradient test
- [x] `DefocusBlur(sigma)` and `MotionBlur(length, angle)`; motion blur at angle 0, length L equals a 1×L box filter

**Verification:** `uv run pytest tests/distort/test_jpeg.py tests/distort/test_blur.py -q`
**Dependencies:** Task 9. **Files:** `distort/{jpeg,blur}.py`, tests. **Scope:** M

#### Task 11: Illumination and perspective
**Description:** The two distortions that dominate real capture (brief §1 table, §4.4).

**Acceptance criteria:**
- [x] `Illumination`: spatially-varying multiplicative gain (low-frequency field), additive offset, per-channel colour cast, gamma; all ranges configurable; params logged
- [x] `Perspective(max_corner_jitter)`: samples a homography by jittering the 4 corners, warps with `grid_sample`; jitter 0 → identity (max abs err ≤ 1e-6); exposes `.last_H` and `inverse_warp()` so an oracle-rectified decode is possible
- [x] Gradient test passes for both

**Verification:** `uv run pytest tests/distort/test_illumination.py tests/distort/test_perspective.py -q`
**Dependencies:** Task 9. **Files:** `distort/{illumination,perspective}.py`, tests. **Scope:** M

#### Task 12: Moiré + channel presets
**Description:** Screen→camera moiré as a sampling problem (display subpixel grid × non-integer
resample × sensor grid), using PIMoG's noise layer as the reference implementation. Then the two
named chains the rest of the project evaluates against.

**Acceptance criteria:**
- [x] `Moire(display_ppi, capture_scale)` on a flat grey image produces a periodic pattern whose dominant spatial frequency is measurable via FFT and matches the expected beat frequency within 10 %
- [x] `configs/sim/print_camera.yaml` = Perspective → Illumination → Defocus+Motion → Resize → JPEG → Noise; `configs/sim/screen_camera.yaml` = same with Moiré inserted before Resize; each has a scalar `severity ∈ [0,1]` that scales every stage's range
- [x] `distort/presets.py::load_chain(name, severity)` builds the chain; smoke config runs both

**Verification:** `uv run pytest tests/distort/test_moire.py tests/distort/test_presets.py -q`; manual: save a distorted sample at severity 0.5 and confirm it looks like a phone photo
**Dependencies:** Tasks 10, 11. **Files:** `distort/{moire,presets}.py`, `configs/sim/*.yaml`, tests. **Scope:** M

#### Task 13: Plots — surfaces, not points
**Description:** The three figures brief §6 requires, generated from `results.csv` only.

**Acceptance criteria:**
- [x] `harness/plots.py` produces: (a) BER vs severity per image class, (b) ROC with the control set, (c) robustness–imperceptibility frontier (recovery rate vs LPIPS)
- [x] `uv run pm-plot outputs/<run>` writes PNGs into the run dir; NullMarker run shows a flat 0.5 line and AUC ≈ 0.5
- [x] Each figure's title includes run id and git hash

**Verification:** `uv run pytest tests/harness/test_plots.py -q` (smoke: files exist, non-empty)
**Dependencies:** Task 12. **Files:** `harness/plots.py`, `pyproject.toml`, tests. **Scope:** S

### Checkpoint — Week 1 complete
- [x] `uv run pytest` green, < 60 s total on CPU
- [x] `pm-corpus build` → `pm-run` (NullMarker, both presets, severity grid) → `pm-plot` runs unattended
- [x] Every §6 column populated for marked rows
- [x] Human review of a distorted sample at severity 0.5 — does it look like a phone photo?

---

### Phase 2 — Week 2: Classical baseline (one week, then stop)

#### Task 14: Block DCT utilities
**Acceptance criteria:**
- [x] `markers/classical/dct.py`: batched 8×8 `block_dct` / `block_idct` on luminance; round-trip error < 1e-5; single block matches `scipy.fft.dctn(norm="ortho")`
- [x] `rgb_to_ycbcr` / inverse round-trip within 1e-5

**Dependencies:** Task 1. **Scope:** S

#### Task 15: Watson DCT perceptual model → slack matrix
**Description:** The core of the budget map (brief §4.1). Three components: frequency sensitivity
table, luminance masking, contrast masking. Pin and document the viewing condition (distance, DPI).

**Acceptance criteria:**
- [ ] `watson_slack(y_dct, viewing=ViewingCondition(...)) -> slack` same shape as coefficients
- [ ] Bright block slack > dark block slack; textured block slack > flat block slack; DC term handled
- [ ] Unit test against a hand-computed 8×8 block for all three stages
- [ ] Docstring states the assumed viewing condition and the masking exponent used

**Dependencies:** Task 14. **Scope:** M

#### Task 16: BCH(64 → 127) codec
**Acceptance criteria:**
- [ ] `markers/classical/bch.py` wraps `galois` (or `bchlib`): encode 64 → 127 bits; corrupt ≤ t bits → exact recovery; > t → returns `(None, failed=True)`
- [ ] `metrics/decoding.payload_recovered` gains an `ecc=` path
- [ ] Interface accepts hard bits now but is shaped so a soft-input decoder can replace it

**Dependencies:** Task 5. **Scope:** S

#### Task 17: Spread-spectrum marker with Watson budget + correlation detector
**Description:** Cox-style additive SS in mid-band DCT coefficients, N chips/bit from a keyed PN
sequence, per-coefficient amplitude = `strength × watson_slack`. Detector returns per-bit
correlations (soft LLRs) and a global normalised-correlation detection score.

**Acceptance criteria:**
- [ ] `ClassicalSSMarker` satisfies `Marker`; no-distortion, strength 1.0 → BER 0 on all corpus classes
- [ ] `capacity(img)` returns a first-cut estimate (sum of slack energy in the mid-band / energy needed per bit at target BER) — documented as provisional
- [ ] PSNR/SSIM/LPIPS reported; refusal behaviour: `embed` raises `CapacityError` when `capacity < n_bits` unless `force=True`

**Dependencies:** Tasks 15, 16, 13. **Scope:** M

#### Task 18: Week-2 sweep, plots, and failure writeup
**Acceptance criteria:**
- [ ] `configs/experiments/classical_sweep.yaml`: full corpus × {print, screen} × severity {0, .25, .5, .75, 1} × strength grid × 3 seeds, with control
- [ ] The three §6 plots plus BER-vs-angle (perspective isolated) and BER-vs-blur
- [ ] `docs/results/week2_classical.md`: per-class failure analysis — *where* and *why* it breaks (expected: perspective + blur). This is the reference line for every later chart

**Dependencies:** Task 17. **Scope:** M

### Checkpoint — Week 2 complete
- [ ] Classical numbers on the board; harness validated by a real marker
- [ ] Watson slack maps eyeballed on flat vs textured images — do they match intuition?
- [ ] Stop improving the classical system. Move on.

---

### Phase 3 — Week 3: Learned baseline (medium resolution; refine at Week-2 checkpoint)

- **Task 19:** Switch torch to CUDA build (`[tool.uv.sources]` / index); verify `torch.cuda.is_available()`; harness runs on GPU when available, CPU otherwise.
- **Task 20:** Install Video Seal; reproduce a published number on its own evaluation script. Record the exact commit + numbers in `docs/results/week3_videoseal_repro.md`.
- **Task 21:** `markers/videoseal/adapter.py::VideoSealMarker` implements `Marker` (LLRs from logits, score from detection head). Sync module exposed separately.
- **Task 22:** Run `classical_sweep.yaml` unchanged with `VideoSealMarker`; overlay on Week-2 plots.
- **Fallback:** if Video Seal is unsuitable, StegaStamp in a TF1.x container — budget 2 days max before deciding.

### Checkpoint — Week 3
- [ ] Learned vs classical on identical sweep; the gap is measured, not assumed.

---

### Phase 4 — Week 4: Physical capture rig (medium resolution)

- **Task 23:** Capture protocol doc + `pm-capture log` CLI: assigns capture ids, records `{phone, medium(print/screen), display_or_printer, distance_m, angle_deg, lighting, image_id, marked}` to `captures.csv` at the external captures path.
- **Task 24:** Print-sheet tool: tiles marked + control images onto A4 with a human-readable id and corner fiducials (for *channel measurement* only — the decoder must never depend on them).
- **Task 25:** Ingestion: photo → locate sheet (fiducials) → rectify → crop image → `decode` → `ResultRow` with `capture_conditions` filled. Also decode *without* rectification to measure the sync gap.
- **Task 26:** 2AFC visibility tool: shows original vs marked side by side, forced choice, logs responses; reports accuracy with a binomial CI. Yourself + 5 people.
- **Task 27:** Week-4 report: the robustness surface for *your* hardware + 2AFC result → **Go / no-go per brief §7.**

### Checkpoint — Week 4 (GO / NO-GO)
- [ ] 64 bits at ≥ 90 % at 1 m / 30° / mixed lighting, with 2AFC at chance? → GO, push envelope.
- [ ] If not: which constraint binds — **angle/distance** or **visibility**? Pick the branch below.

---

### Phase 5 — Weeks 5–6: one gap, not three (gated)

**Branch A — synchronisation** (if it breaks on angle/distance): learned rectifier + tiled pilot
+ RaptorQ over tiles so partial capture suffices. Tasks: pilot design, tile codec, rectifier
training on the Week-1 simulator, re-run Week-4 sweep.

**Branch B — budget map + refusal** (if it breaks on visibility; the more likely and more
interesting outcome): replace Watson-only slack with a learned/measured per-image budget
calibrated against the 2AFC data; make `capacity()` honest; publish the refusal rate per class.
Tasks: 2AFC-calibrated visibility model, capacity estimator, refusal policy, re-run sweep.

Either way: a written result in `docs/results/phase1_result.md`. Negative results count.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Differentiable JPEG / moiré sims don't match reality; Week-4 numbers diverge from sim | High | Week 4 exists to measure exactly this; log sim severity alongside real conditions and fit the mapping |
| Video Seal is provenance-shaped and its sync module doesn't cover perspective | Med | Task 20 checks this on day 1 of Week 3; StegaStamp fallback capped at 2 days |
| 6 GB VRAM insufficient for learned-model work | Med | Inference + fine-tune only in Phase 1; rent GPU hours if Branch A needs training |
| Python 3.13 wheel gaps (lpips, galois, Video Seal deps) | Low–Med | Task 0 surfaces it immediately; fallback is pinning `.python-version` to 3.12 before the first commit |
| Corpus too easy / too synthetic | Med | Hard classes are first-class in the manifest; every plot is per-class so flat/text failures can't hide in an average |
| Physical captures lost | High | External backed-up path, decided before Week 4 (Open Question 3) |
| Scope creep into Weeks 5–6 before the Week-4 gate | Med | `todo.md` stops at Task 27; branch tasks are written only after the gate |

## Resolved Questions (2026-09-18)

1. **Photographic corpus source:** Kaggle `arnaud58/landscape-pictures` (4,319 JPEGs, mostly
   800–1600 px), downloaded to `configs/paths.yaml → datasets.landscape_pictures`. Task 8's
   builder samples a fixed, seeded subset (default 40) and resizes/centre-crops to 512×512;
   the manifest records the source filename. **Faces:** not sourced; the `face` class is skipped
   until a folder is provided (`datasets.faces` empty).
2. **CUDA:** done now. `torch==2.14.0+cu130` via `[tool.uv.sources]`, verified on the RTX 3050.
   Task 19 reduces to "harness uses GPU if available".
3. **Captures location:** `configs/paths.yaml → captures` =
   `C:/Users/Asus/Downloads/perceptual-media-data/captures`. **Caveat:** `Downloads` is not
   backed up — before Week 4, either point this at a cloud-synced folder or add the directory to
   a sync client. Everything else under `perceptual-media-data` is re-downloadable.
4. **Watson viewing condition:** decided — default `ViewingCondition(dpi=96, distance_cm=60)`
   (≈ 35 px/degree, a laptop/desktop screen). Reason: the Week-4 2AFC visibility study will be
   run on-screen, and the perceptual model must match the condition the human judgement is made
   under, otherwise the calibration in Branch B is meaningless. A `print_300dpi_40cm` preset
   (≈ 84 px/degree) is provided as a second condition for the print channel; both are logged in
   `ResultRow.capture_conditions` when relevant.

## Reference implementations reviewed

- `claude_markdowns/compass_artifact_*.md` — the pre-brief literature review. Consistent with
  the brief; one recommendation worth carrying: **print-to-camera is the first physical target**
  (medium difficulty, no moiré/refresh). When sweeps get expensive, `print_camera` is the
  primary preset and `screen_camera` the secondary.

- `docs/research/stegastamp_pimog_notes.md` — StegaStamp's `transform_net` (ranges, ramps,
  differentiable JPEG surrogate, blur mixture, perspective sampling) and PIMoG's `Noise_Layer.py`
  (light field, moiré formula). Feeds concrete defaults into Tasks 10–12 and confirms the
  `Marker` LLR interface matches how StegaStamp's decoder is used (BCH outside the network).
