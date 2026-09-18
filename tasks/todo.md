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
- [ ] Task 6: runner + `pm-run` CLI (vertical slice with NullMarker + Identity)
- [ ] **Checkpoint A** — end-to-end CSV with marked + control rows, BER ≈ 0.5
- [ ] Task 7: fidelity metrics — PSNR, SSIM, LPIPS
- [ ] Task 8: corpus builder — flat / gradient / text / textured (+ photos, faces from dir) + manifest
- [ ] Task 9: distortion base + Identity / Noise / Resize / Crop
- [ ] Task 10: JPEG (real + differentiable) + defocus / motion blur
- [ ] Task 11: illumination + perspective (with oracle inverse)
- [ ] Task 12: moiré + `print_camera` / `screen_camera` presets with severity knob
- [ ] Task 13: plots — BER vs severity per class, ROC, recovery-vs-LPIPS frontier
- [ ] **Checkpoint W1** — corpus → run → plot unattended; distorted sample looks like a phone photo

## Phase 2 — Week 2: Classical baseline
- [ ] Task 14: block DCT + YCbCr utils
- [ ] Task 15: Watson DCT slack matrix (pinned viewing condition)
- [ ] Task 16: BCH(64→127)
- [ ] Task 17: `ClassicalSSMarker` — SS embed under Watson budget, soft correlation detector, provisional `capacity()` + refusal
- [ ] Task 18: classical sweep + plots + `docs/results/week2_classical.md`
- [ ] **Checkpoint W2** — reference line exists; stop tuning classical

## Phase 3 — Week 3: Learned baseline
- [ ] Task 19: ~~CUDA torch build~~ (done in Task 0); GPU-if-available in harness
- [ ] Task 20: Video Seal installed; published number reproduced; recorded
- [ ] Task 21: `VideoSealMarker` adapter
- [ ] Task 22: same sweep, overlaid on Week-2 plots
- [ ] **Checkpoint W3** — learned vs classical gap measured

## Phase 4 — Week 4: Physical capture
- [ ] Task 23: capture protocol + `pm-capture log`
- [ ] Task 24: print-sheet tool (ids + fiducials)
- [ ] Task 25: ingestion — locate, rectify, decode, rows with capture conditions; also un-rectified decode
- [ ] Task 26: 2AFC visibility tool + study (self + 5)
- [ ] Task 27: Week-4 report → GO / NO-GO
- [ ] **Checkpoint W4 (gate)** — pick Branch A (sync) or Branch B (budget map + refusal)

## Phase 5 — Weeks 5–6
- [ ] (tasks written after the Week-4 gate)

## Open questions — resolved 2026-09-18 (see plan.md)
- [x] Q1: Kaggle landscape-pictures downloaded to external root; faces deferred
- [x] Q2: CUDA switched now (cu130)
- [x] Q3: captures at external root (needs backup/sync before Week 4)
- [x] Q4: Watson default 96 DPI @ 60 cm; print preset 300 DPI @ 40 cm
