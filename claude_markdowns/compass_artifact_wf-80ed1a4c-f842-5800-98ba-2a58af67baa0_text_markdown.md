# Perceptually Lossless Machine-Readable Media: A Technical Research Review

## TL;DR
- **The specific niche you are targeting — an open-standard, communication-oriented, camera-readable invisible payload (64–256 bits) decodable by any stranger's commodity smartphone from arbitrary media after real-world capture — is only partially occupied and is a genuine open problem.** Print-to-camera and screen-to-camera decoding of ~100-bit payloads is solved *in the lab* (StegaStamp 98.7% mean bit accuracy; PIMoG/cross-attention ~97%; DeepLight frame error rate <0.2 at ~2 m) and *commercially* for print retail (Digimarc Barcode), but every deployed system that "any phone reads" is either proprietary/patented (Digimarc holds ~800 U.S. patents) or optimized for a *different* goal (provenance/AI-labeling, server- or known-key detection), and the open, casual-camera, arbitrary-media combination is not served by any single existing system.
- **Goal alignment is the decisive filter.** The overwhelming majority of well-funded recent work (SynthID, Stable Signature, Video Seal/AudioSeal, TrustMark, C2PA durable credentials) is *provenance/attribution*, tuned for digital transforms and detected server-side with a known key — it explicitly does NOT target casual-camera physical decoding, and its robustness collapses under diffusion regeneration and physical capture. Only three lineages truly share your communication goal: Digimarc (closed), the StegaStamp/LFM/RIHOOP learned print-and-screen lineage (open code, lab-grade), and the screen-camera VLC literature (HiLight, InFrame, ChromaCode, DeepLight — video-oriented, not arbitrary still media).
- **Synchronization under perspective at a distance without visible anchors is the hardest unsolved sub-problem and the place a new project can most contribute** — together with an open, benchmarked, communication-oriented standard spanning still images with honest false-positive accounting and a security model treating decoded payloads as untrusted.

---

## Key Findings

1. **Two literatures, two goals.** "Watermarking" splits into *provenance/attribution* (prove origin; server-side, known-key detection; robust to digital edits) and *communication* (deliver a payload to an arbitrary decoder; must survive physical capture). Your project is squarely in the second, smaller camp.
2. **Payload sizes converge on ~50–256 bits.** StegaStamp: 100-bit message → 56 usable bits after ECC (40 error-correcting bits). Digimarc Barcode: ~98-bit payload tile. SynthID-O: 136 bits in 512×512. Meta Video Seal/PixelSeal: 256 bits, ChunkySeal 1024 bits. Your 64–256 bit target is realistic and consistent with the field.
3. **Channel difficulty ranking (easiest→hardest): digital-only < print-to-camera < screen-to-camera.** Screen-to-camera adds moiré, refresh/rolling-shutter interaction, display gamma and subpixel structure on top of the perspective/illumination problems of print capture.
4. **Learned encoder/decoder systems beat classical methods on physical channels by a wide margin**, primarily because of their *differentiable distortion (noise) layers* — but those same systems are strikingly fragile to diffusion-regeneration removal and to forgery, and the physical-channel insight in their noise layers is directly reusable by a classical baseline.
5. **The chroma-imperceptibility idea is real but double-edged:** chrominance/blue-channel embedding is genuinely less visible (S-cones ≈ 2% of cones), but JPEG/H.264 4:2:0 subsampling and camera ISPs decimate chroma resolution, destroying naïve chroma marks. Deployed screen-camera systems that use color (DeepLight blue-channel, Uber-in-Light RGB-complementary) work only because they combine it with heavy redundancy and ML decoding.
6. **Presence detection false-positive accounting is where marketing and measurement diverge.** SynthID operates at low single-digit FPR and, at internet scale, uses conformal p-value calibration. Any casual-camera system continuously scanning frames needs an explicit, low false-positive presence test — an area most academic papers under-report.

---

## Details

### TASK 1 — Prior Art Mapping (goal alignment is stated explicitly for each)

**Digimarc (Digimarc Barcode / Digimarc Discover).**
- *Goal:* originally provenance/copyright; commercially, **camera-readable payload delivery for retail packaging and print** — the closest deployed analog to your goal, but for POS scanners and print, not casual arbitrary media.
- *Capacity:* Digimarc Barcode uses a **128×128 watermark-element "tile"** carrying roughly a **98-bit binary payload**, redundantly tiled across the whole surface; typically GS1 GTIN + Application Identifiers (serial, price, weight, sell-by). Convolutional coding at rate 1/4 (patent example: 4096-bit encoded signal). Requires ≥203 dpi print; a tile occupies ≈1.28×1.28 inches.
- *Robustness:* reads from any part of the item, at angles, resilient to damage; designed for red-LED POS scanners and Digimarc-enabled apps. Reading requires the Digimarc SDK/enabled hardware — NOT any generic phone camera app.
- *Openness:* **proprietary.** Digimarc's Q1 2025 SEC 10-Q describes "one of the world's most extensive patent portfolios in digital watermarking and related fields, with approximately 800 U.S. patents" (global portfolio, granted plus pending, exceeds 1,100) — a serious freedom-to-operate constraint. Digimarc now co-chairs the C2PA watermarking task force and interoperates with TrustMark for "durable Content Credentials."
- *Where it fails your goal:* closed ecosystem, requires specific decoder software; print-optimized (not arbitrary screenshots/social media); casual "any phone" decoding is not the deployed reality.

**Other commercial/industrial.**
- *Alibaba "Blue Stars"/Visualead dotless visual codes:* visible-but-aesthetic QR variants for anti-counterfeit — provenance, not invisible communication. Requires the Taobao app.
- *"Invisible watermark" retail packaging (Digimarc-driven):* full-surface UPC replacement; POS goal.
- *Snapchat Snapcodes / Amazon SmileCodes:* visible scannable codes, not invisible.
- *Shazam (audio):* audio *fingerprinting* (no embedded payload — passive recognition), a conceptually different, database-match approach worth noting as an alternative to embedding for identifier delivery.
- *Sony/Panasonic and LiFi/VLC display-camera links:* active, hardware-timed, video-oriented (see screen-camera literature below).

**Academic screen-to-camera communication (VLC lineage).** *Goal: unobtrusive communication (payload delivery), matching yours — but for video on displays, not arbitrary still media.*
- **HiLight** (Li, An, Campbell, Zhou; VLCS 2014, Dartmouth): encodes in alpha-channel (translucency) changes over any content; ~1 bit/symbol; measured BER as high as ~40% in later comparisons; up to 500 bps in some settings.
- **InFrame / InFrame++** (Wang, Zhou et al.): dual-layer full-frame VLC; ~12.8 kbps at 120 FPS; BER ~31% (and near 50% at small block sizes) in independent comparison.
- **TextureCode** (Nguyen et al., Rutgers): texture-adaptive; measured BER ~10%; hybrid with HiLight ~22 kbps goodput.
- **Uber-in-Light** (Izz et al., INFOCOM 2016, Rutgers): complementary RGB-channel intensity so net luminance change ≈ 0; green channel left for sync; MFSK modulation limits throughput.
- **ChromaCode** (Zhang, Wu, Yang et al.; MobiCom 2018, Tsinghua): modulates **CIELAB lightness** (perceptually uniform) with complementary flicker-fused frames; **raw >700 kbps, goodput 120 kbps at BER 0.05**, fully imperceptible per 20-user study; Reed-Solomon + convolutional concatenated ECC; explicit moiré/rolling-shutter/perspective handling. (Note: despite the name, it modulates lightness, not chroma.)
- **DeepLight** (Tran, Jayatilaka, Ashok, Misra; IPSN 2021, SMU/CMU; arXiv:2105.05092): **blue-channel-only modulation** + DNN decoder that decodes all spatial bits jointly without per-bit isolation. Verbatim: "a fully functional DeepLight system is able to robustly achieve high decoding accuracy (frame error rate < 0.2) and moderately-high data goodput (≥0.95 Kbps) using a human-held smartphone camera, even over larger screen-camera distances (≈2m)," with screen-extraction "IoU values ≥ 83%" (89% indoors).
- **Revelio** (2025): real-world screen-camera with imperceptible embedding; zero errors up to 2 m at 0°, reliable to ±40°, degrades beyond 2.25 m/40°.
- *Where they fail your goal:* almost all require **video / multi-frame** on an active display; they are not designed for a single static poster/photo, and print is out of scope. They do, however, provide the best empirical physical-channel distortion catalog.

**Print/screen-to-camera robust watermarking (your closest technical lineage). *Goal: communication / invisible hyperlinks — matches yours.***
- **StegaStamp** (Tancik, Mildenhall, Ng; CVPR 2020, Berkeley; arXiv:1904.05343): learned encoder/decoder with differentiable distortion layer (perspective, blur, color, noise, JPEG). **100-bit message; "robust retrieval of 95% of 100 encoded bits in real-world conditions"; "we achieve a mean bit-accuracy of 98.7%" across 18 display/printer×camera combinations (6 displays/printers × 3 cameras); usable hyperlink is 56 message bits + 40 error-correcting bits** (per the GitHub: "100 bit message → 56 bits after ECC"). PSNR ~29.7, SSIM ~0.91 (visible-ish artifacts in flat regions). Open source (github.com/tancik/StegaStamp). This is the canonical "invisible QR replacement."
- **PIMoG** (Fang et al., ACM MM 2022, NUS): differentiable screen-shooting noise layer (perspective+illumination+moiré); **>97% extraction accuracy** across screen-shooting conditions; open source. Degrades sharply beyond ~45° capture angle.
- **RIHOOP** (Jia et al., IEEE T-Cybernetics 2022): 3D-rendering distortion attacks; 100-bit payload, PSNR ~28.6/SSIM ~0.936 at 100 bits, ~100% clean bit accuracy, robust across print+screen in a 1059-photo real test.
- **De-END, ARWGAN, MBRS, TSDL, SepMark, WaveRecovery, cross-attention (2024–25):** successive SOTA on digital + screen-shooting; MBRS reaches ~99.7% at 10% crop but only 30-bit capacity; De-END pushes PSNR ~50 dB with >97% accuracy at ≤40% cropout. Distortion-Agnostic (Luo et al., CVPR 2020) uses adversarial training + channel coding (30 data → 120 coded bits), averaging ~88% bit accuracy on unseen distortions.
- **Light Field Messaging (LFM)** (Wengrowski & Dana; CVPR 2019, Rutgers): models the **camera-display transfer function (CDTF)**; **Camera-Display 1M dataset** (1M captures, 25 camera-display pairs, open). Frontal BER ~5% (vs ~50% for digital steganography); 7–9% on unseen hardware; robust to 45°. Patented (US 11,790,475).
- *Where they fail your goal:* lab-scale, single-model demonstrations; no open interoperable standard; imperceptibility often marginal (StegaStamp visibly alters flat regions); angle/distance envelopes still narrow; localization assumes the mark occupies a large fraction of frame.

**Provenance / AI-labeling watermarks (well-funded, but WRONG goal for you).**
- **Google SynthID-Image** (Gowal et al., Google DeepMind, arXiv:2510.09263, Oct 2025): post-hoc deep watermark. The paper states it "has been used to watermark over ten billion images and video frames across Google's services and its corresponding verification service is available to trusted testers"; by Google I/O May 2026 the cumulative figure was cited as "more than 100 billion images and videos, along with the equivalent of 60,000 years of audio," and adoption now extends to OpenAI, ElevenLabs, Kakao and Nvidia Cosmos. The external variant **SynthID-O encodes a 136-bit payload in 512×512** and reports ~99.6% bit accuracy and ~99.98% TPR on **aggregated random digital transformations** (screenshots, resize, JPEG, color) — not physical camera capture at an angle. Detection is server-side / trusted-tester only; the system deliberately separates detection from payload recovery and operates at a very low FPR using conformal p-values. *Goal: provenance.*
- **Meta Stable Signature** (Fernandez et al. 2023): roots watermark in the latent diffusion decoder. **Video Seal / PixelSeal / AudioSeal / Watermark-Anything (WAM):** open-source suite; **256-bit** flagship, **ChunkySeal 1024-bit**; WAM does localized watermarks surviving inpainting/splicing; includes a synchronization model to revert geometric transforms. *Goal: provenance/attribution; digital robustness.*
- **TrustMark (Adobe)** + **C2PA 2.1 durable Content Credentials:** TrustMark is an **open-source** invisible watermark carrying a "soft-binding" identifier keying a provenance database; approved in C2PA's watermark registry. This is architecturally close to what you want (identifier → database lookup) but *goal is provenance*, and TrustMark's robustness collapses under regeneration (TPR ~9–34% under regen in the VINE benchmark) and it is not built for casual-camera physical decode.
- **Amazon (Titan)**, **Microsoft (Bing)**, **OpenAI** (adopting SynthID): all invisible provenance marks, detector not public (Amazon's is public). *Goal: AI-content labeling.*
- *Where they fail your goal:* they assume digital distribution, known-key/server detection, and do not target casual-camera physical decoding; several are provably removable by diffusion regeneration.

**Classical robust watermarking (the foundation, mostly digital-goal).**
- **Spread-spectrum** (Cox, Kilian, Leighton, Shamoon 1997): additive, host-interference-limited, robust but low capacity.
- **QIM / dither modulation** (Chen & Wornell 2001): host-interference-*rejecting*; embedding capacity ≈1/3 bit/s per Hz per dB; within 1.6 dB of capacity with distortion compensation — substantially better capacity-robustness than spread spectrum. Fragile to amplitude scaling (addressed by AQIM).
- **Dirty-paper / informed coding** (Costa 1983; Miller, Doerr, Cox 2004): side-information-at-encoder achieves high-capacity robust marks.
- **DCT/DWT-domain embedding** with Watson perceptual model: e.g., 2048 bits in 512×512 robust to valumetric distortion.
- **Fourier-Mellin / log-polar RST-invariant** (Ó Ruanaidh & Pun; US 6,282,300): rotation/scale/translation synchronization — the classical answer to geometric registration, but LPM/inverse-LPM degrade image quality and implementation is notoriously hard.
- *Where they fail your goal:* geometric synchronization under *perspective* (not just RST) at a distance, and physical capture noise, are largely outside classical guarantees.

### TASK 2 — Channel Analysis Ranked by Difficulty

**(a) Digital-only (easiest).** Distortions: resize, JPEG/WebP quantization, chroma subsampling (4:2:0), color/brightness/contrast, crop, screenshot resample. Achieved: MBRS ~99.7% at 10% crop; SynthID-O ~99.6% bit accuracy across aggregated random digital transforms; Stable Signature ~0.95 bit accuracy under combined crop+brightness+JPEG. This channel is essentially solved for ~30–256 bits.

**(b) Print-to-camera (medium).** Added physical distortions: halftoning/dithering, ink spread/dot gain, paper reflectance and texture, illumination color temperature, specular highlights, perspective, lens blur, and the printer *and* camera color pipelines. Achieved: StegaStamp 98.7% mean bit accuracy (100-bit message) across print+display; RIHOOP ~100% clean, robust across a 1059-photo real test; Digimarc reads reliably at POS. Distances/angles are modest and payloads ~50–100 usable bits.

**(c) Screen-to-camera (hardest).** All of (b) plus **moiré** (display pixel grid × camera sensor grid aliasing), **display gamma and subpixel (RGB stripe/PenTile) layout**, **rolling-shutter × refresh-rate banding**, **flicker-fusion constraints** (temporal), **auto-exposure/auto-white-balance**, and **backlight non-uniformity**. Achieved (mostly video): ChromaCode BER 0.05 at 120 kbps; DeepLight FER<0.2, ≥0.95 kbps, ~2 m; LFM frontal BER ~5% (7–9% unseen hardware), robust to 45°; Revelio zero errors ≤2 m/0°, reliable ±40°. Screen-shooting *still-image* watermarks (PIMoG/cross-attention) hit >95–97% but degrade sharply past ~45°.

### TASK 3 — Candidate Encoding Matrix

| Approach | Capacity | Imperceptibility | Digital | Print→cam | Screen→cam | Encode/Decode cost | Notes |
|---|---|---|---|---|---|---|---|
| Spatial/pixel perturbation | High raw, low robust | Low unless masked | Med | Low | Low | Very low | Classical LSB unusable physically |
| Relative-luminance region-pair | Low | High | High | Med | Med | Low | Robust, sync-friendly, low capacity |
| Chroma/hue embedding | Med | **High (S-cones ~2%)** | **Low (4:2:0 kills it)** | Med (optimal color dir.) | Med (needs redundancy+ML) | Low | Double-edged; DeepLight blue-only works only w/ ML decoder |
| Texture/statistical | Med | High in texture | Med | Med | Med | Med | TextureCode BER ~10% |
| DCT-domain (Watson) | ~2048 b/512² | High w/ JND | High | Med | Low-Med | Low | Mature, valumetric-robust |
| DWT/wavelet | Med-High | High | High | Med | Med | Low-Med | WaveRecovery screen-shooting |
| Frequency-domain spread spectrum | Low | High | High | Med | Med | Low | Cox et al.; host-interference limited |
| QIM / dither mod | **High** | Med-High | High | Med | Med | Low | Chen-Wornell; ~1.6 dB of capacity; scaling-fragile |
| Informed/dirty-paper coding | **Highest** | High | High | Med | Med | High (encoder) | Best capacity-robustness tradeoff in theory |
| Feature/region-aware perceptual budget | Med | **Highest** | High | Med-High | Med-High | Med | Basis of modern learned encoders |

QIM/dirty-paper fundamentally change the tradeoff: by rejecting host interference they buy capacity at fixed robustness/distortion, but their advantage erodes under the strong geometric/gain distortions of physical capture unless combined with robust synchronization and gain estimation.

**On chroma specifically:** the human visual system is ~4× less sensitive to chrominance than luminance (resolving luminance detail to ~50 cycles/degree vs. chrominance ~10), and only ~2% of retinal cones are blue-sensitive — which is why blue-channel embedding (DeepLight) and RGB-complementary embedding (Uber-in-Light) are less visible. But the same physiology motivates 4:2:0 subsampling in JPEG/H.264/HEVC, which decimates chroma to ¼ resolution and destroys naïve chroma marks; papers targeting compression survivability either embed in luminance or use subsampling-aware schemes. A US patent (Digimarc-style) measured chrominance embedding as *more* robust than luminance *at equal visibility* — but only for print/scan and only when the optimal color direction is chosen. Note ChromaCode modulates CIELAB *lightness*, not chroma, so it is not evidence for chroma-channel robustness.

### TASK 4 — Synchronization and Geometric Registration (the hardest sub-problem)
- **Autocorrelation / periodic tiling:** tile the mark; autocorrelation peaks form a grid whose spacing/orientation reveals scale/rotation. Cheap, self-synchronizing, but weak under perspective (non-affine) and adds a detectable periodic signature.
- **Fourier-Mellin / log-polar (RST-invariant):** maps rotation→shift, scale→shift; theoretically perfect for RST but *not* perspective; LPM resampling harms quality; expensive and fiddly.
- **Template/pilot signals:** embedded reference peaks in DFT enable affine estimation; fast matched-filter recovery; adds signal energy that a steganalyst can find.
- **Feature-point (SIFT/SURF/ORB) anchored embedding:** embed relative to repeatable keypoints; robust to some geometric change but keypoints move under heavy blur/perspective.
- **Learned localization/rectification (StegaStamp detector, DeepLight object-detector, LFM):** a CNN localizes and rectifies the marked region before decoding. **This is what actually works under perspective at a distance today** — StegaStamp's detector + spatial-transformer rectification is the reason it decodes in-the-wild. Cost: a detection network per frame; false-positive presence detection must be calibrated (SynthID's conformal p-values are the state of the art for honest FPR control). Search over rotation/scale/perspective is the dominant decode cost in classical pipelines and is largely amortized into a single forward pass in learned ones.

### TASK 5 — Error Correction and Payload Design
- **Deployed choices:** Digimarc uses **rate-1/4 convolutional coding** (98-bit payload → 4096-bit signal) with tiling redundancy. ChromaCode/DeepLight and most VLC use **concatenated Reed-Solomon + convolutional** codes plus **interleaving** against burst/contiguous damage. StegaStamp uses BCH (56 message + 40 ECC bits). Modern learned systems often add channel coding (Distortion-Agnostic: 30 data → 120 coded bits).
- **BCH/Reed-Solomon** dominate small fixed payloads; **LDPC/polar** appear where longer blocks justify soft-decision decoding.
- **Fountain/Raptor (RaptorQ, RFC 6330)** are the right tool for **crop/occlusion resilience**: rateless, reconstruct from any sufficient subset of tiles — ideal when the mark is redundantly tiled and only a fraction is captured. Open implementations exist (OpenRQ). Korus et al. applied fountain codes to annotation watermarking.
- **Interleaving** across spatially separated tiles converts contiguous physical damage (specular highlight, fold, crop) into spread, correctable erasures.

### TASK 6 — Perceptual Masking and Metrics
- **Budget allocation:** Watson's DCT model (frequency sensitivity + luminance + contrast masking) remains the classical workhorse; DWT pixel-wise masking (PWM) is a wavelet analog (Watson is often found to outperform it). Modern learned encoders implicitly learn a per-region perceptual budget (texture masking) via perceptual losses.
- **Metrics and their inadequacy:** PSNR/SSIM are weakly correlated with watermark visibility; the Stable Signature ablation found MSE-optimized marks looked worse at *higher* PSNR, while Watson-VGG and LPIPS losses gave the most eye-pleasing results. LPIPS/DISTS capture texture/structure but under-weight chroma shifts; recent work (MT-Mark) adds **CVVDP** specifically to model color-difference sensitivity because LPIPS misses subtle color shifts. Consensus: **no single automatic metric is sufficient; human studies remain the gold standard.**
- **Human protocols:** two-alternative forced-choice (2AFC) and forced-choice A/B against the original; SynthID and Video Seal both run human evaluation on corner-case content as the primary fidelity measure.

### TASK 7 — Classical vs Learned (recommendation)
- **Learned beats classical on physical channels by a wide margin**, driven almost entirely by the **differentiable distortion (noise) layer**: LFM frontal BER ~5% vs ~50% for classical digital steganography under camera-display transfer; StegaStamp decodes in-the-wild where classical marks fail. The margin is *wide* on screen/print-to-camera, *narrow-to-none* on pure digital transforms (where DCT/QIM already do well).
- **What the noise layers reveal (reusable by a classical baseline):** the dominant physical distortions worth simulating are (i) perspective/homography, (ii) illumination (multiplicative + additive, colored), (iii) moiré, (iv) blur (defocus + motion), (v) JPEG + chroma subsampling, (vi) color-transfer/white-balance. PIMoG's ablation shows perspective+illumination+moiré+Gaussian captures most of the screen-shooting channel. VINE's insight — that *editing/blurring acts as a low-pass filter, so robust energy must live in low/mid frequencies* — is directly transferable.
- **Recommended hybrid:** classical structure (tiled, RST/perspective-synchronizable carrier with pilot template + fountain-coded payload) with **learned components for (a) the perceptual budget/embedding strength map and (b) the localizer/rectifier and soft decoder.** This gives interpretable synchronization and open ECC while capturing the learned physical-channel gains. Start from a StegaStamp/RIHOOP baseline for the physical channel and a TrustMark-style identifier→database architecture for the payload semantics.

### TASK 8 — Attacks and Security
- **Diffusion regeneration removes essentially all pixel-domain and many latent watermarks:** "Invisible Image Watermarks Are Provably Removable Using Generative AI" (NeurIPS 2024) proves regeneration drives payload mutual information toward zero; "Vanishing Watermarks" (2026) shows near-zero recovery for StegaStamp, TrustMark, VINE after diffusion edits at high visual fidelity. Removal typically costs quality (PSNR ~25, or SSIM drop) — relevant because your *communication* use case may tolerate a legitimately re-encoded image, unlike provenance.
- **Forgery/collusion:** WMCopier forges invisible marks on arbitrary images; black-box forgery attacks defeat semantic (Tree-Ring, Gaussian Shading) marks; content-agnostic marks are vulnerable to steganalysis-based estimation. Collusion (averaging many marked images) strips image-agnostic marks (CoMSMark studies this for multi-screen shooting, showing PIMoG/RIHOOP/StegaStamp forgery accuracy rising to 75–99% under 100-image collusion).
- **Neural decoders are adversarially fragile:** because a learned decoder is a neural net, decoder-gradient-guided perturbations break it like adversarial examples.
- **Security model:** treat every decoded payload as **untrusted input** — it can be forged, replayed, or collided. Do NOT let a decoded ID confer trust; use it only as a lookup key into a server that enforces authentication, rate-limiting, and signature verification (a decoded 256-bit string should carry a cryptographic MAC/signature checked server-side, exactly as C2PA soft-bindings envisage but with anti-forgery hardening).

### TASK 10 — Concrete Pointers
- **Key papers:** Tancik, Mildenhall, Ng, *StegaStamp*, CVPR 2020 (arXiv:1904.05343, code github.com/tancik/StegaStamp). Wengrowski & Dana, *LFM*, CVPR 2019 (Camera-Display 1M dataset; US 11,790,475). Zhu et al., *HiDDeN*, ECCV 2018. Luo et al., *Distortion-Agnostic*, CVPR 2020 (arXiv:2001.04580). Fang et al., *PIMoG*, ACM MM 2022 (code FangHanNUS/PIMoG). Jia et al., *RIHOOP*, IEEE T-Cyb 2022. Zhang et al., *ChromaCode*, MobiCom 2018. Tran et al., *DeepLight*, IPSN 2021 (arXiv:2105.05092). Li/An/Campbell/Zhou, *HiLight*, VLCS 2014. Lu et al., *VINE*, ICLR 2025 (arXiv:2410.18775, code Shilin-LU/VINE, W-Bench dataset). Fernandez et al., *Stable Signature* 2023; *Video Seal* 2024 (code facebookresearch/videoseal). Gowal et al., *SynthID-Image*, DeepMind 2025 (arXiv:2510.09263). Cox et al., *Secure Spread Spectrum*, 1997. Chen & Wornell, *QIM*, 2001. Costa, *Writing on Dirty Paper*, 1983. Ó Ruanaidh & Pun, Fourier-Mellin watermarking.
- **Open source:** StegaStamp, PIMoG, VINE (MIT-style; VINE-B/VINE-R checkpoints on HuggingFace), Meta Video Seal/AudioSeal/WAM/content-seal, Adobe TrustMark, LFM, OpenRQ (RaptorQ).
- **Datasets/benchmarks:** Camera-Display 1M (LFM), W-Bench (VINE), WAVES (removal-attack benchmark), MS-COCO/DIV2K for training.
- **Constraining patents:** Digimarc portfolio (~800 U.S. patents; e.g., US 7,822,969; US 11,386,517; US 12,045,908; halftone/Fourier-Mellin US 7,020,349; RST US 6,282,300). Rutgers LFM US 11,790,475. Freedom-to-operate review is essential before any commercial print/POS application.

---

## TASK 9 — Open Gaps Assessment

**(i) Solved and commercially deployed — do NOT rebuild:**
- Print-to-camera POS/packaging payload delivery: **Digimarc owns this** (98-bit tiles, full-surface, POS scanners, ~800 U.S. patents). Rebuilding for retail print is wasted effort and legally risky.
- Digital-transform-robust provenance watermarking at scale: **SynthID, Stable Signature, TrustMark, Video Seal own this** (30–1024 bits, 99%+ under digital transforms). Do not compete on AI-content labeling / provenance.
- Audio recognition via fingerprint: Shazam-style database matching is mature (though it embeds no payload).

**(ii) Solved in the lab but NOT robustly/openly:**
- Print/screen-to-camera ~100-bit payload decoding from a *photograph* (StegaStamp, RIHOOP, PIMoG, LFM) — works, open code exists, but: single-model demos, narrow angle/distance envelopes (~±45°, ~1–2 m), imperceptibility often marginal, **no interoperable open standard**, and no honest false-positive presence-detection benchmark.
- Fully imperceptible screen-camera *communication* (ChromaCode, DeepLight) — high throughput but requires **video on an active display**, not a static image.

**(iii) Genuinely open / underexplored (where a new project should aim):**
1. **Anchor-free geometric synchronization under perspective at a distance** — the single hardest, least-solved sub-problem. Classical RST-invariants don't cover perspective; learned rectifiers work but aren't open, standardized, or benchmarked for the casual-camera envelope beyond ~45°/2 m.
2. **An open, interoperable, communication-oriented standard** (payload format + ECC + synchronization signal) for *arbitrary still media* — Digimarc is closed; TrustMark is open but provenance-scoped and camera-fragile; nothing fills the open casual-camera slot.
3. **Honest false-positive presence detection** for continuous camera scanning of unmarked imagery — under-reported everywhere except SynthID's conformal calibration.
4. **A security model treating decoded payloads as untrusted**, with anti-forgery/anti-collusion hardening, for the communication (not provenance) use case.

**Verdict on the framing question:** "Open-standard, camera-readable, communication-oriented (not provenance-oriented) invisible payload delivery from arbitrary media" is **an open niche.** The closest occupants each miss one axis: Digimarc (camera-readable + communication, but closed + print-only), StegaStamp/RIHOOP/LFM (camera-readable + communication + open code, but not standardized, narrow envelope, marginal imperceptibility), and TrustMark/SynthID (open/standardized + robust, but provenance goal + not casual-camera). No system occupies all four axes at once.

---

## Recommendations (staged)

**Stage 0 — Scope decision (now).** Commit explicitly to the *communication* goal and to **still images first** (not video), because the video VLC systems (HiLight/InFrame/ChromaCode/DeepLight) do not transfer to a static poster. Decide print-to-camera OR screen-to-camera as the first target — **recommend print-to-camera first** (medium difficulty, StegaStamp/RIHOOP give a working baseline, and it avoids moiré/refresh). Threshold to proceed: reproduce StegaStamp's ~98% bit accuracy on your own print+phone captures.

**Stage 1 — Baseline + honest benchmark (0–3 months).** Stand up StegaStamp and PIMoG as baselines; build a capture harness measuring **BER vs distance, angle, lighting, payload size**, and — critically — **presence-detection false-positive rate** on unmarked images (borrow SynthID's conformal p-value calibration). Adopt 100-bit payload with BCH now; plan RaptorQ tiling for crop resilience. Threshold: >95% decode of a 64-bit payload at ≤1 m, ±30°, mixed lighting.

**Stage 2 — Synchronization research (3–9 months, the differentiator).** Attack the anchor-free perspective-synchronization gap: combine a learned localizer/rectifier with an embedded pilot/template and fountain-coded tiles. Benchmark against the ±45° cliff that PIMoG/cross-attention hit. Threshold to claim contribution: reliable decode at ≥45° and ≥2 m with <1% false positive.

**Stage 3 — Openness + security (9–15 months).** Publish an **open interoperable spec** (payload format, ECC, synchronization signal) — the field's clearest hole given Digimarc is closed and TrustMark is provenance-scoped. Implement the untrusted-payload security model (server-side MAC/signature, anti-forgery, anti-collusion). Add a diffusion-regeneration robustness evaluation (expect and document failure; decide whether re-encoding is acceptable for your use case).

**Benchmarks that would change the plan:** if diffusion-regeneration must be survived, pivot to semantic/latent embedding (accept it cannot be casual-camera decoded — the goals conflict). If moiré/refresh dominate on the target displays, adopt DeepLight-style blue-channel + ML decoding and multi-frame capture. If a freedom-to-operate review finds blocking Digimarc claims for your commercial channel, keep the project research/open-source or design around tiling/synchronization claims.

---

## Caveats
- **Marketing vs measured:** Digimarc's "any imaging system reads it" and "most robust" claims are commercial; the measured reality is POS scanners + enabled apps, not arbitrary phone camera apps. SynthID's headline TPRs are for **digital** transform suites, not angled physical capture; its detector is not publicly available, and cumulative "images watermarked" figures (10 billion in the Oct 2025 paper, 100 billion by May 2026) are company-reported. "Fully imperceptible" claims (ChromaCode, DeepLight) rest on small user studies (~20 users) and specific hardware.
- **Numbers are not directly comparable** across papers: payload sizes, image resolutions (often 128×128 or 512×512), whether ECC is applied, and capture protocols all differ. Treat the tabulated BER/accuracy figures as regime indicators, not head-to-head.
- **VINE, TrustMark, SynthID, Video Seal target provenance/editing robustness, NOT physical camera capture** — do not assume their strong digital numbers transfer to your casual-camera channel; VINE reports no print/screen-camera capture experiments at all (its W-Bench covers diffusion/editing + digital distortions only).
- **Chroma embedding caution:** the "chroma is less perceptible" result is genuine but is partly defeated by 4:2:0 subsampling in the very JPEG/camera pipeline you must survive; do not build on pure-chroma embedding without redundancy.
- **The provenance-vs-communication goal conflict is fundamental to robustness:** the strongest anti-removal marks are entangled with image semantics (latent), which makes them undetectable by a casual camera; the most camera-readable marks (pixel-domain, redundant) are the easiest to remove by regeneration. A project cannot maximize both at once and must choose.
- Some cited items (e.g., 2026-dated arXiv preprints on removal attacks) are very recent and not all peer-reviewed; treat their specific figures as provisional.