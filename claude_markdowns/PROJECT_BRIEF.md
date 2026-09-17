# Perceptually Lossless Machine-Readable Media — Phase 0/1 Brief

Working document for the repo. Written to be read by both a human and Claude Code.
Status: pre-implementation. Nothing built yet.

---

## 1. What this project is, stated precisely

**Goal.** Transform an arbitrary still image so that (a) a human cannot distinguish it from
the original under normal viewing, and (b) a commodity smartphone camera photographing it —
from a print or a screen, at an angle, under uncontrolled lighting — can recover a 64–256 bit
payload.

**This is a communication system, not a provenance system.** The distinction is not academic.
It determines every design choice:

| | Provenance (SynthID, TrustMark, Stable Signature) | Communication (this project) |
|---|---|---|
| Who decodes | The issuer, server-side, with the key | Any stranger, on-device |
| Channel | Digital distribution, edits, recompression | Physical capture: print/screen → camera |
| Must survive | Diffusion regeneration, inpainting | Perspective, lighting, blur, moiré |
| Failure mode | Missed attribution | Nothing happens when you point your phone |
| Capacity need | 30–256 bits | 64–256 bits |

Do not import provenance-world robustness targets. Surviving a diffusion edit is *not our
problem*. Surviving being photographed at 40° in a lobby *is*.

**Non-goals for Phase 1.** Audio. Video. Anti-removal / adversarial robustness. Multi-frame
accumulation. Anything that requires a specific app ecosystem or hardware.

---

## 2. The central constraint (read this before anything else)

The signal energy that survives camera capture at distance is the *same* energy the eye can
detect in smooth regions. Imperceptibility and physical robustness are not independent axes —
they are two ends of one budget.

Concrete evidence: StegaStamp achieves 98.7% mean bit accuracy on real print+camera captures,
but at **PSNR ≈ 29.7 / SSIM ≈ 0.91**. That is visibly marked. Faint structured texture appears
in flat regions. The paper does not claim invisibility; it claims usable hyperlinks on photographs.

Our stated bar is higher. Therefore:

> **Assumption to test in Phase 1, not assume:** that there exists a per-image perceptual budget
> at true indistinguishability which is still large enough to carry ≥64 bits through a physical
> camera channel — and that this budget is nonzero for a useful fraction of real-world images.

This may be false for some images. A flat, low-texture poster (solid background, large smooth
gradients) has almost no masking capacity. **This is expected and is not a failure of the
approach.** The right engineering response is a system that *measures its own capacity per image
and declines to mark images below threshold*, rather than silently emitting a marked image that
is either visible or undecodable.

**That capability — honest per-image capacity estimation with graceful refusal — does not exist
in any published or deployed system, and is a stronger contribution than another 1% of bit accuracy.**

---

## 3. Build vs. borrow

Do not write from scratch what already exists and is open. Time spent reimplementing DCT
watermarking is time not spent on the open problem.

### Borrow directly
- **StegaStamp** (github.com/tancik/StegaStamp) — learned encoder/decoder, differentiable
  distortion layer, detector+rectifier. This is the physical-channel workhorse. Note it is
  TensorFlow 1.x; budget time for either a PyTorch port or a container with old deps.
- **PIMoG** (github.com/FangHanNUS/PIMoG) — better screen-shooting noise layer specifically
  (perspective + illumination + moiré). Steal the noise layer even if not the whole model.
- **Meta Video Seal / WAM** (github.com/facebookresearch/videoseal) — modern, actively
  maintained PyTorch, includes a *synchronization model* that reverts geometric transforms.
  Wrong goal (provenance) but the sync module and training infrastructure are directly reusable.
- **TrustMark** (Adobe, open source) — for the payload-as-database-key architecture pattern.
- **RaptorQ** (RFC 6330; OpenRQ or a Rust/Python impl) — rateless codes for crop resilience.
- **Camera-Display 1M** (LFM dataset) — 1M real screen→camera captures across 25 camera/display
  pairs. Saves months of data collection for the screen channel.

### Build ourselves
1. **Per-image perceptual budget map with a hard indistinguishability constraint** and the
   capacity estimator / refusal mechanism that follows from it.
2. **Synchronization past the ±45° / 2m cliff** that every published system hits.
3. **Honest presence-detection false-positive accounting** at realistic scanning rates.
4. **The open payload/ECC/sync spec** — nobody has published one for this use case.

### Do NOT build
- Print/retail POS payload delivery. Digimarc owns this with ~800 U.S. patents. Research and
  open-source work is fine; a commercial product in packaging/retail is a legal minefield.
- Anything provenance- or AI-labeling-shaped. That market is saturated by well-funded actors.
- A classical-only system. Classical methods lose to learned ones on physical channels by a wide
  margin. Implement classical *once*, as a baseline and to build intuition — then move on.

---

## 4. Mathematics to actually understand

Grouped by what it buys you. Aim for working understanding, not exam-level rigor. You need to be
able to implement and debug these, not prove theorems about them.

### 4.1 Perceptual modelling — highest priority, this is where our contribution lives
- **Contrast Sensitivity Function (CSF).** Human sensitivity vs. spatial frequency (cycles/degree).
  Peaks ~4 cyc/deg, falls off sharply above ~30. Determines which frequencies can carry energy invisibly.
- **Watson's DCT perceptual model.** The canonical JND model. Three components: frequency
  sensitivity table, luminance masking (brighter blocks tolerate more), contrast masking (busy
  blocks tolerate more). Produces a per-coefficient *slack* matrix = how much you can perturb
  each DCT coefficient invisibly. **Implement this yourself; it's ~100 lines and it is the core
  of the budget map.**
- **Contrast/texture masking and the masking exponent.** Why noise hides in texture and not in sky.
- **Viewing-distance dependence.** Cycles/degree depends on distance and display DPI — the budget
  is not a property of the image alone. Pin an assumed viewing condition and state it.
- **Modern perceptual metrics and their failure modes:** LPIPS, DISTS, CVVDP. Know that PSNR and
  SSIM correlate poorly with watermark visibility — Stable Signature found MSE-optimized marks
  looked *worse* at higher PSNR.

### 4.2 Detection and estimation theory — needed for honest FPR claims
- **Matched filter / correlation detection.** Optimal detection of known signal in noise.
- **Deflection coefficient** d = (μ₁−μ₀)/σ. The single number that predicts detectability.
- **Neyman–Pearson, ROC curves, and why you must report the full curve** not one operating point.
- **The multiple-comparisons problem at scanning rates.** A detector at 30 fps scanning unmarked
  images runs ~2.6M trials/day. A 10⁻³ per-frame FPR is a false positive every 33 seconds. You need
  ~10⁻⁸ or a multi-frame confirmation protocol. SynthID uses conformal p-value calibration for
  exactly this; read that section of their paper.
- **Processing gain from spreading.** Why N chips per bit buys 10·log₁₀(N) dB — the fundamental
  reason a weak distributed signal is recoverable at all.

### 4.3 Transforms and embedding
- **2D DCT**, block-based, JPEG quantization tables. Why mid-frequency coefficients are the
  robustness sweet spot (low = visible, high = destroyed by compression and blur).
- **DWT**, subband structure, why LL carries robustness and HH carries invisibility.
- **Spread spectrum** (Cox et al. 1997): additive, host-interference-limited. Understand *why*
  the host image acts as noise against you.
- **Quantization Index Modulation / dither modulation** (Chen & Wornell 2001): host-interference-
  *rejecting*. Higher capacity at the same distortion. Fragile to amplitude scaling — understand
  why, and what AQIM does about it. This is the single biggest classical capacity lever.
- **Dirty-paper coding / Costa 1983** — know it exists and that it bounds what informed embedding
  can achieve. Do not implement in Phase 1.

### 4.4 Geometry and synchronization — the hard open problem
- **Homography estimation**, DLT, RANSAC. Perspective capture is a homography, not an affine
  transform — this is why RST-invariant classical methods are insufficient.
- **Log-polar mapping and Fourier–Mellin.** Rotation→shift, scale→shift. The classical answer,
  good for RST, *breaks under perspective*. Know it to know its limit.
- **Autocorrelation-based self-synchronization.** Tile the mark periodically; autocorrelation
  peaks form a lattice revealing scale and rotation. Cheap. Also a steganalytic signature.
- **Spatial Transformer Networks** — the learned rectifier StegaStamp uses. This is what actually
  works today.
- **Moiré / aliasing.** Display pixel grid × camera sensor grid. Understand it as a sampling
  problem (Nyquist), because it tells you which spatial frequencies are *destroyed* on the
  screen→camera channel specifically.

### 4.5 Coding
- **BCH** and **Reed–Solomon** — the workhorses for short fixed payloads. StegaStamp: 100 bits =
  56 message + 40 ECC.
- **Interleaving.** Converts contiguous physical damage (glare, fold, crop) into spread erasures.
- **Fountain / RaptorQ codes.** Rateless. Recover from *any* sufficient subset of received tiles.
  This is the correct tool for crop and partial-capture resilience and is underused in this field.
- **Soft-decision decoding / LLRs.** Your decoder outputs confidences, not bits. Throwing that away
  before the ECC costs you several dB. Do not hard-slice early.

### 4.6 Skip for now
Polar codes. LDPC. Dirty-paper trellis. Information-theoretic capacity derivations. Revisit only
if Phase 1 shows the payload is capacity-limited rather than synchronization-limited.

---

## 5. Phase 1 build plan (~6 weeks, aggressive but real)

The ordering principle: **build the measurement apparatus before the thing being measured.**
Most of the risk here is that you can't tell whether a change helped.

### Week 1 — Harness first
- Repo scaffold, Python/PyTorch, deterministic seeds, config-driven experiments.
- **Distortion simulator** (the most reusable asset you will build). Differentiable where possible:
  perspective homography, illumination (multiplicative + additive, colored), blur (defocus + motion),
  JPEG with chroma subsampling, resize, crop, noise, moiré. Port PIMoG's noise layer as reference.
- **Metrics module**: PSNR, SSIM, LPIPS, and BER / payload-recovery-rate / detection ROC.
- **Image corpus**: deliberately include hard cases — flat skies, solid-color posters, gradients,
  text-heavy documents, faces. Do not evaluate only on DIV2K-style textured photos; that hides
  exactly the failure mode we care about.

### Week 2 — Classical baseline (one week, then stop)
- DCT mid-band spread-spectrum embed + correlation detect, with a Watson-model budget map.
- BCH(64→127) payload.
- Purpose: build intuition, produce a reference line on every chart, and validate the harness.
- **Expected result: it fails on physical capture.** That is the point. Measure *how* it fails.

### Week 3 — Reproduce the learned baseline
- Get StegaStamp or Video Seal running end-to-end. Do not modify it yet. Reproduce published numbers
  on simulated distortions.
- If TF1.x is painful, prefer Video Seal (modern PyTorch) and accept the goal mismatch for now.

### Week 4 — Physical capture rig and the reality check
- Print 30–50 marked images. Photograph with 2–3 phones at a grid of conditions:
  distance {0.3, 0.5, 1, 2 m} × angle {0, 15, 30, 45, 60°} × lighting {window, LED, dim}.
  Log everything. This is tedious and it is the single highest-information week in the plan.
- Same grid for screen→camera on 2 displays.
- **Deliverable: your own robustness surface.** Where exactly does it break for *your* hardware?
- Simultaneously: run a 2AFC visibility study on yourself + 5 people. Marked vs. original, forced
  choice. If people score above chance, you are not at indistinguishability, whatever PSNR says.

### Weeks 5–6 — Attack one gap, not three
Pick based on Week 4 results:
- If it breaks on **angle/distance** → synchronization. Learned rectifier + tiled pilot + RaptorQ
  over tiles so partial capture suffices.
- If it breaks on **visibility** → the budget map and refusal mechanism. This is the more
  interesting outcome and the more likely one given our fidelity bar.

Ship a written result either way. Negative results here are publishable and are what tell you
whether the project is viable.

---

## 6. Test harness specification

Every experiment must emit a row with: image id, image class (flat/textured/text/face), payload
bits, embed strength, distortion chain applied, capture conditions, BER, payload recovered (bool),
detector score, PSNR, SSIM, LPIPS, wall-clock encode/decode ms.

Report **surfaces, not points**: BER vs. angle, BER vs. distance, BER vs. payload size, and the
robustness–imperceptibility frontier (recovery rate as a function of LPIPS/perceptual distance).
A single successful scan demo is worthless as evidence.

Always include an **unmarked control set** through the identical pipeline to measure false positives.
This is the discipline most papers skip and the one that will make your results credible.

---

## 7. Go / no-go gates

- **End Week 4:** Can you recover 64 bits at ≥90% success at 1m / 30° / mixed lighting, at a
  fidelity where a 2AFC study is at chance? If yes → the premise holds, push the envelope.
  If no → identify which of the two constraints bound you, and *narrow the claim* (e.g. "textured
  media only", "≤1m", "screen only") rather than abandoning.
- **End Week 6:** Do you have a measured result that is not reproducible from an existing repo's
  README? If not, you are still assembling, not contributing.

---

## 8. Constraints and risks beyond the caveats

1. **Patent exposure.** Digimarc's portfolio (~800 U.S. patents) covers print watermarking,
   synchronization, and detection broadly. Rutgers holds LFM (US 11,790,475). Research and
   open-source publication is low risk; commercialization in print/retail is not. Decide the
   licence posture early and get a real FTO opinion before any commercial step.
2. **Compute.** Training a StegaStamp-class model is days on one modern GPU. Budget for it or
   rent it. Not a blocker, but not free.
3. **Physical data collection is the real time sink,** not the math. Every robustness claim needs
   real captures. Automate logging from day one.
4. **Arbitrary media is genuinely harder than the literature's setting.** Published systems are
   evaluated on curated photographic content. Posters with flat fills, text, and brand colors —
   your actual use case — are the worst case and are underrepresented in every benchmark.
5. **Reproducibility of prior work is uneven.** StegaStamp is TF1.x; several 2022–24 papers have
   no released code or non-reproducible numbers. Budget slippage for this.
6. **Security model is not optional.** A decoded payload can be forged (WMCopier-class attacks
   forge invisible marks on arbitrary images) and collided. Treat every decoded payload as
   untrusted input: it is a lookup key only, never a trust assertion, never auto-navigated,
   length-limited, signature-checked server-side.

---

## 9. Honest assessment of viability

**Pullable:** an open, communication-oriented, camera-decodable payload system for textured
still media with a measured and published robustness envelope, plus a per-image capacity
estimator that refuses images it cannot mark invisibly. Nothing like this is published or deployed.

**Not pullable as stated:** true perceptual indistinguishability + arbitrary media (including
flat/smooth) + casual camera at distance + high reliability, all simultaneously. The energy
budget likely does not exist for the flat-media case. Expect to narrow on the media axis.

**Most likely useful outcome:** a system with a declared operating envelope and honest self-
assessment, rather than a universal invisible QR replacement. That is a smaller claim and a
much more defensible one.

---

## 10. Immediate next actions

1. Create repo, scaffold the harness (Week 1 above). Harness before algorithm.
2. Read, in this order: StegaStamp (arXiv:1904.05343) → PIMoG (ACM MM 2022) → Watson's DCT model →
   Cox et al. 1997 → Chen & Wornell 2001 → SynthID-Image (arXiv:2510.09263), skimming the last
   for its detection/FPR calibration section only.
3. Build the hard-case image corpus *first*. It will shape every later decision.
4. Order/borrow a second phone and a printer you can use repeatedly. Consistency of capture
   hardware matters more than its quality.
