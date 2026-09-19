# TODO — Phase 1

Checklist view of `tasks/plan.md`. Tick tasks only when every acceptance criterion in the plan
is met and `uv run pytest` is green. Weeks 5–6 tasks are written after the Week-4 gate.

## Phase 0 — Skeleton
- [x] Task 0: `uv add lpips scikit-image pyyaml pandas`; first commit (2026-09-18)

## Phase 1 — Week 1: Harness
- [x] Task 1: core — types, `seed_everything`, image I/O (2026-09-18)
- [x] Task 2: config — YAML → dataclasses, strict keys (2026-09-18)
- [x] Task 3: `Marker` protocol, `DecodeResult`, `NullMarker` (2026-09-18)
- [x] Task 4: `ResultRow` (§6 schema) + `ResultWriter` (2026-09-18)
- [x] Task 5: decoding metrics — BER, recovery, ROC/AUC, FPR helpers (2026-09-18)
- [x] Task 6: runner + `pm-run` CLI (vertical slice with NullMarker + Identity) (2026-09-18)
- [x] **Checkpoint A** — end-to-end CSV with marked + control rows, BER ≈ 0.5 (2026-09-18: 12 rows, BER 0.52/0.48, AUC 0.58, 3 s on GPU)
- [x] Task 7: fidelity metrics — PSNR, SSIM, LPIPS (2026-09-18)
- [x] Task 8: corpus builder — flat / gradient / text / textured (+ photos, faces from dir) + manifest (2026-09-18: 67 images, 9/5/5/8 + 40 photos)
- [x] Task 9: distortion base + Identity / Noise / Resize / Crop (2026-09-18)
- [x] Task 10: JPEG (real + differentiable) + defocus / motion blur (2026-09-18)
- [x] Task 11: illumination + perspective (with oracle inverse) (2026-09-18)
- [x] Task 12: moiré + `print_camera` / `screen_camera` presets with severity knob (2026-09-18)
- [x] Task 13: plots — BER vs severity per class, ROC, recovery-vs-LPIPS frontier (2026-09-18)
- [x] **Checkpoint W1** — corpus → run → plot unattended; distorted sample looks like a phone photo (2026-09-18: null_sweep 1,206 rows in 33 s on GPU, 147 tests in 14 s)

## Phase 2 — Week 2: Classical baseline
- [x] Task 14: block DCT + YCbCr utils (2026-09-18)
- [x] Task 15: Watson DCT slack matrix (pinned viewing condition) (2026-09-18)
- [x] Task 16: BCH(64→127) (2026-09-18)
- [x] Task 17: `ClassicalSSMarker` — SS embed under Watson budget, soft correlation detector, provisional `capacity()` + refusal (2026-09-18)
- [x] Task 18: classical sweep + plots + `docs/results/week2_classical.md` (2026-09-18)
- [x] **Checkpoint W2** — reference line exists; stop tuning classical (2026-09-18: sync-limited at 1 deg; flat images fail JPEG alone)

## Phase 3 — Week 3: Learned baseline
- [x] Task 19: ~~CUDA torch build~~ (done in Task 0); GPU-if-available in harness (`device: auto` since Task 6; verified in audit 2026-09-18)
- [x] Task 20: Video Seal installed (optional extra, decord overridden, cwd-independent loader); published SA-V metrics pulled as the reference; recorded (2026-09-18)
- [x] Task 21: `VideoSealMarker` adapter (2026-09-18)
- [x] Task 22: same sweep, overlaid on Week-2 plots (2026-09-18: `pm-compare`, `docs/results/week3_learned.md`)
- [x] **Checkpoint W3** — learned vs classical gap measured (2026-09-18: perspective cliff 1° → ~15°; both dead at ≥30°; flat-image JPEG failure recurs)

## Phase 4 — Week 4: Physical capture
- [x] Task 23: capture protocol + `pm-capture log` (filename-nominal + EXIF + measured PnP pose; 24/24 calibration captures located)
- [x] Task 24: display-sheet tool `pm-capture sheet` (ArUco fiducials + slide ids; screen variant, print deferred)
- [x] Task 25a: channel calibration — `pm-capture channel`, real channel vs `screen_channel` sim, `docs/results/week4_channel.md`
- [x] Task 25b: decode ingestion `pm-capture decode` → run dir with real_screen / real_screen_unrectified / real_screen_control rows (awaiting the decode captures)
- [x] Task 26: 2AFC visibility tool `pm-capture afc serve|report` (study itself pending: self + 5)
- [ ] Task 27: Week-4 report → GO / NO-GO
- [ ] **Checkpoint W4 (gate)** — pick Branch A (sync) or Branch B (budget map + refusal)

## Phase 5 — Weeks 5–6
- [ ] (tasks written after the Week-4 gate)

## Open questions — resolved 2026-09-18 (see plan.md)
- [x] Q1: Kaggle landscape-pictures downloaded to external root; faces deferred
- [x] Q2: CUDA switched now (cu130)
- [x] Q3: captures at external root (needs backup/sync before Week 4)
- [x] Q4: Watson default 96 DPI @ 60 cm; print preset 300 DPI @ 40 cm
