# StegaStamp and PIMoG — implementation notes for the distortion simulator

Read 2026-09-18 from source. Purpose: ground Tasks 9–12 (`distort/`) and Task 21 (learned
baseline) in what actually works, not what the papers summarise. Numbers below are the defaults
in the released code, not necessarily what produced the paper's figures.

Sources
- StegaStamp: github.com/tancik/StegaStamp — `models.py` (`transform_net`), `train.py`
  (argparse defaults), `utils.py` (`get_rand_transform_matrix`, `jpeg_compress_decompress`,
  `random_blur_kernel`). TensorFlow 1.x.
- PIMoG: github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw
  — `Noise_Layer.py`. PyTorch + kornia. (Perspective for *evaluation* is MATLAB in
  `PerspectiveTransformation/`.)

---

## 1. StegaStamp distortion layer (`transform_net`), in order

Every distortion has a `ramp` (steps over which its strength linearly rises from 0 to the
configured max). `ramp_fn(r) = min(global_step / r, 1)`.

| Stage | Formula / sampling | Default max | Ramp (steps) |
|---|---|---|---|
| Perspective | all 4 corners jittered `uniform(-d, d)` in x and y independently, `d = width * rnd_trans * ramp`; homography via `cv2.getPerspectiveTransform`, applied with `tf.contrib.image.transform` (bilinear) | `rnd_trans = 0.1` → ±40 px on 400 px | 10 000 |
| Blur | one 7×7 kernel per batch, chosen from {none, Gaussian σ∈[1, 3], line σ∈[0.25, 1] with width `wmin_line`}; `conv2d`, `SAME` | fixed | none |
| Noise | additive Gaussian, `σ = uniform() * rnd_noise * ramp` | `rnd_noise = 0.02` | 1 000 |
| Brightness + hue | `img + uniform(-rnd_bri, rnd_bri)` (scalar/batch) `+ uniform(-rnd_hue, rnd_hue)` per channel (shape `(B,1,1,3)`) | `rnd_bri = 0.3`, `rnd_hue = 0.1` | 1 000 |
| Contrast | `img * uniform(contrast_low, contrast_high)`, bounds ramp from 1 outward | `[0.5, 1.5]` | 1 000 |
| Saturation | `(1 − s) * img + s * lum`, `s = uniform() * rnd_sat * ramp` | `rnd_sat = 1.0` | 1 000 |
| JPEG | differentiable; `quality = 100 − uniform() * (100 − jpeg_quality) * ramp` | `jpeg_quality = 25` | 1 000 |

Clip to [0, 1] after colour ops.

**Differentiable JPEG (`utils.jpeg_compress_decompress`).** Standard pipeline: RGB→YCbCr,
4:2:0 chroma by 2×2 `avg_pool`, 8×8 DCT, divide by scaled standard luma/chroma tables, *soft
rounding*, dequantise, IDCT. Two rounding surrogates:
- `diff_round(x) = round(x) + (x − round(x))^3` — identity-gradient-ish, used in the Shin & Song
  "JPEG-resistant adversarial images" formulation.
- `round_only_at_0(x) = x^3 if |x| < 0.5 else x` — StegaStamp's default. Only coefficients
  that would quantise to zero are pushed toward zero; everything else passes straight through.
  Cheaper and closer to what JPEG actually destroys (small coefficients).

Quality→table scale: `s = 5000/q if q < 50 else 200 − 2q`, then `table * s / 100` (libjpeg's rule).

**Perspective note.** Corner jitter is *isotropic and independent per corner*. That is not what a
camera at 30–60° does — real capture is a foreshortening along one axis plus mild rotation.
Independent corner jitter covers that set but spends most of its samples on physically
implausible warps. For Task 11 we should sample **camera pose** (tilt, yaw, roll, distance) and
derive the homography, with corner-jitter available as a second mode for comparability.

## 2. StegaStamp training recipe (for Task 21 / any future training)

- Image 400×400, secret 100 bits in the paper (`secret_size` default in code = 20; the released
  model is 100 = 56 message + 44 BCH — note the README's numbers, not the argparse default).
- Encoder: secret → Dense(7500) → 50×50×3 → ×8 upsample → concat with image → U-Net
  [32, 32, 64, 128, 256] → 1×1 conv residual. Decoder: STN (affine, 6 params) → conv stack
  [32,32,64,64,64,128,128] → Dense(512) → Dense(secret).
- Loss: `1.5·L2(YUV, edge-falloff mask) + 1·LPIPS + 1·BCE(secret) + 1·GAN`, image losses ramped
  over 20k steps, secret loss from step 1, edge mask delayed to step 60k. 140k steps, batch 4,
  Adam 1e-4.
- **Secrets are raw Bernoulli(0.5) bits at training time; BCH is applied only at inference.**
  This means the network never sees code structure — the decoder outputs independent per-bit
  logits, which is exactly the soft-LLR interface our `Marker` protocol wants.
- **No refusal, no per-image budget.** Perceptual constraint is a global loss weight. Same
  strength on sky and on grass. This is the gap the brief identifies (§2).

## 3. PIMoG screen-shooting noise layer (`Noise_Layer.py`)

Order and formula (single forward pass):

```
x = perspective(x)                          # kornia warp, corners uniform(-2, 2) px  [tiny!]
x = x * Li * 0.85 + Mo * 0.15 + N(0, 0.001) # Li: light field, Mo: moiré field, both in [0,1]
```

- **Light `Li`** (per-image, NumPy, no gradient): either a linear ramp across the image from `a`
  to `b` with `a = 0.7 + 0.2·U`, `b = 1.1 + 0.2·U`, in one of 4 orientations; or a radial field
  centred at a random point with the same endpoint range. Multiplicative only.
- **Moiré `Mo`** (per-image, NumPy): `z1 = 0.5 + 0.5·cos(2π·r)` where `r` is pixel distance from a
  random centre (concentric rings, period 1 px → aliases into low-frequency rings), and
  `z2 = 0.5 + 0.5·cos(cosθ·j + sinθ·i)` with `θ ~ U[0, 180)` (a grating with angular frequency
  1 rad/px). `z = min(z1, z2)`, `M = (z + 1) / 2`. Added at 15 % amplitude.
- **Noise**: σ ≈ 0.032.
- No blur, no JPEG, no additive colour cast, no chroma subsampling.

**Assessment.** The moiré is a *visual* approximation — it produces something that looks like
moiré, but it's not derived from the display grid × sensor grid sampling model, so its frequency
content doesn't depend on capture scale. The 2-px perspective range is only usable because their
evaluation applies real perspective offline in MATLAB. Worth borrowing: the light-field
parameterisation (ramp / radial, [0.7,0.9]→[1.1,1.3]) and the min-of-two-cosines trick as a
cheap "moiré-like" texture. Not worth borrowing: the perspective range or the fixed 0.85 / 0.15
mix.

## 4. Decisions this feeds into the plan

1. **Task 11 (perspective):** sample camera pose → homography as the primary mode; keep
   StegaStamp-style corner jitter (`rnd_trans` as fraction of width) as `mode="corners"` so we
   can reproduce their numbers.
2. **Task 11 (illumination):** union of both: PIMoG's linear/radial multiplicative field
   ([0.7–0.9] → [1.1–1.3]) *plus* StegaStamp's additive brightness (±0.3), per-channel hue shift
   (±0.1), contrast ([0.5, 1.5]) and desaturation. Each independently switchable with logged params.
3. **Task 10 (JPEG):** *done.* Measured against PIL 4:2:0 on corpus images (PSNR of surrogate
   vs real JPEG): Q25 — STE 46.5 / round_only_at_0 37.9 / diff_round 44.1; Q75 — 48.7 / 43.9 /
   48.5. Default is the straight-through estimator (exact forward, identity gradient);
   StegaStamp's `round_only_at_0` is 5–9 dB further from the real codec and kept as an option.
   Quality→scale rule from libjpeg as above.
4. **Task 10 (blur):** StegaStamp's 7×7 {none, Gaussian σ∈[1,3], line} mixture is a reasonable
   default; make kernel size a parameter (7 px at 400 px is ~1.75 % of width — scale with
   resolution).
5. **Task 12 (moiré):** *done.* Sampling-model moiré (RGB-stripe + black-matrix display render at
   4× → Gaussian lens PSF → point-sampling sensor grid at `capture_scale` px/px with random phase
   and ≤3° rotation → resize back) is the primary; measured beat frequency on flat grey matches
   `|1 − round(1/s)·s|` within 1 % at s = 0.7 / 0.9 / 1.15. PIMoG's min-of-cosines is `mode="pimog"`.
6. **Presets:** `print_camera` and `screen_camera` severity knob should, at severity = 1, cover at
   least the StegaStamp training ranges above so that a StegaStamp/Video Seal model evaluated in
   our harness sees nothing out-of-distribution by construction.
7. **Task 21 (learned baseline):** decoder logits → LLRs directly; StegaStamp's BCH(56→100) sits
   outside the network, same as our Task 16 design. Confirms the `Marker` protocol shape.

## 5. What neither system has (our contribution surface, brief §3 "build ourselves")

- Per-image budget: both apply the same distortion-loss trade-off to every image.
- Refusal: both always emit a marked image.
- Physically-derived perspective sampling and moiré.
- Any false-positive accounting: StegaStamp reports bit accuracy only, no unmarked control set.
