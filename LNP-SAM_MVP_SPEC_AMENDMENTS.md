# LNP-SAM MVP Spec — Amendments (v2)

**Status.** Companion changelog to `LNP-SAM_MVP_SPEC.md`. As of v2 the main spec
**incorporates** these amendments (its Revision v2 note, and §1, §4.5, §5.3–5.6, §6.2.3);
this file remains the record of *why* — the defect evidence, root-cause analysis,
validation numbers and references. Where an amendment changes the behaviour of a
numbered section, that section is named. All new thresholds live in `config.yaml` under
`cleanup:` (per spec §4.4).

**Scope of this revision.** Step 1 (Cleanup, §5) only. Steps 2–6 are unchanged.

---

## 1. Why these amendments exist

Running the original §5 cleanup on the 15 golden panels produced SAM-input images
with three defects that would make SAM segment furniture as particles — the exact
failure mode LNP-SAM exists to prevent. Each was verified against the real panels:

| # | Defect (observed) | Panel | Root cause in the original §5 |
|---|---|---|---|
| B1a | Dark **particle rims** marked as scale bars and painted over | `xiong2020…panel02` | §5.3 step 4 excludes *every* detected bar; §5.3's "longest run per row" flags flat dark rim streaks. |
| B1b | A hard-edged **solid black rectangle** (the panel-letter box) left in the SAM input | `xiong2020…panel02` | §5.4 requires an *isolated* filled rectangle (`area/bbox ≥ 0.90`). The black "C" box is 8-connected to the dark spheres below it → one big component (fill ≈ 0.21) → never detected → never excluded → 1074 black px survive. |
| B2 | Painted-over regions are **smooth blobs** that do not blend into the noisy background | `wang2019…panel01` | §5.6 fills with `cv2.inpaint(TELEA)`, which produces smooth gradients. In a high-noise TEM field a smooth patch reads as an object. |
| B3 | The **SEM scale bar** is never removed and fuses with particles | `pylypchuk2021…panel01` | The bar is a burned-in annotation (pure-white, 11 px thick) on a dark data band. §1 puts "SEM data-bar detection" out of MVP scope, and §5.3's `bar_max_thick_frac` (≈9 px here) rejects an 11 px bar anyway. |

**Design principle adopted.** The literature on nanoparticle-metrology pipelines
(§7 References) treats figure furniture as regions to **detect and exclude**, and the
segmenter then **drops any proposal that lands in the exclusion mask** — which the
original spec already specifies in §6.2.3 (`excl_drop_frac = 0.5`). So:

> The **exclusion mask** is the source of truth for "not a particle." Filling is only
> a cosmetic aid; it must not create *new* segmentable artifacts. Therefore the
> exclusion mask must be **complete** (all burned-in furniture) and **precise** (no
> imaged particle content), and the fill must **blend** into the background.

The amendments below make the exclusion complete + precise (A2, A3) and the fill
blend (A1). The ultimate guarantee remains §6.2.3, enforced when Step 2 is built
(see §6 of this document).

---

## 2. Amendment A1 — texture-matched fill (changes §5.6)

**Original §5.6:** fill excluded pixels with `cv2.inpaint(..., INPAINT_TELEA)` per channel.

**Amendment:** keep the TELEA result as a *base* (it gives a locally correct mean and
respects region boundaries), then **re-inject Gaussian noise matched to the local
background's high-frequency amplitude** into the filled pixels, so the region carries
the same noise texture as its surroundings.

Procedure (`cleanup.build_sam_input`, `fill_method: texture`):
1. `base = TELEA inpaint` of the excluded pixels (per channel).
2. Estimate background noise σ robustly from the *un-excluded* crop:
   `hf = gray − GaussianBlur(gray, σ=2)`, `σ_bg = 1.4826 · MAD(hf[valid])`.
3. Add `N(0, (fill_noise_scale·σ_bg·255)²)` (one grayscale field, applied to all
   channels) to the filled pixels only; clip to `[0, 255]`.

`fill_method: telea` restores the exact original §5.6 behaviour.

**Effect.** The black "C" rectangle and the smooth "500 nm"/"b" patches are replaced
by background-toned noise. Measured: wang filled-region high-frequency std
0 → 0.046 (background 0.058); the xiong SAM-input corner is no longer a black block
(mean 0.42, 0 % of pixels below 0.1).

**Rationale.** Inpainting "alters the noise statistical model" of the filled segment
(Ⓡ2); exemplar/noise-consistent fills preserve texture (Ⓡ3). "Inpaint-then-match-
noise" achieves that with the OpenCV primitives already in the stack.

---

## 3. Amendment A2 — OCR-anchored annotation exclusion (augments §5.4)

**Problem.** §5.4 only finds a label box when it is an isolated filled rectangle. When
the box touches particles (common), the connected component is not a clean rectangle
and the box is missed (B1b).

**Amendment (`cleanup.detect_annotation_boxes`).** For each OCR text box:
1. Take the median intensity inside the text bbox. If `≥ annot_ocr_pure_bright` the
   box sits on a **pure-bright** patch; if `≤ annot_ocr_pure_dark`, a **pure-dark**
   patch; otherwise the text is on ordinary background → not a label box, skip.
2. Region-grow that pure patch **inside a bounded window** (the text box padded by
   `annot_grow_pad_frac · max(box side)`), taking the connected component that most
   overlaps the text box. Bounding the window stops the grow from running into
   particles even when the box is 8-connected to them.
3. If the grown area exceeds `annot_grow_area_mult ×` the text-box area (the grow ran
   away), fall back to excluding the padded text box.
4. Exclude the resulting box, dilated by `annot_dilate_px`.

The original §5.4 detector still runs (harmless where it succeeds); A2 is additive.

**Effect.** The "C" box (and analogous "b"/"a" boxes) are excluded in full: xiong
un-excluded black pixels 1074 → 0. Grown boxes stay small (≤ 3.4 % of panel across the
golden set); none swallow a particle.

---

## 4. Amendment A3 — burned-in furniture bars + exclusion scope (augments §5.3)

Two linked changes:

### 4.1 Only *furniture* bars enter the exclusion (changes §5.3 step 4)

**Original §5.3 step 4:** "Every bar → exclusion."

**Amendment (`bar_exclude_scope`).** `detect_bars` still runs for calibration, but only
**furniture bars** are painted into the exclusion mask:
- burned-in bars from §4.2 below, plus
- the single bar the calibration paired with a scale label (`calibrate` now returns it).

Other high-contrast interior streaks (particle rims) are used for nothing and no longer
pollute the mask. `bar_exclude_scope: all` restores the original "every bar" behaviour.

**Effect.** Fixes B1a: xiong rim streaks are no longer painted over (xiong2017 bar-
exclusion 4.1 % → 0 %); the calibrated grayscale bars are still excluded.

### 4.2 Burned-in bar detector (`cleanup.detect_burned_in_bars`)

Detects drawn/burned annotations that §5.3 misses (B3). A candidate horizontal run is a
burned-in bar when **all** hold:
- **near-saturated**: pixels `≥ burn_bar_pure_bright` (white) or `≤ burn_bar_pure_dark` (black);
- **ultra-flat**: per-row std `≤ burn_bar_std_max`;
- **solid block**: thickness in `[burn_bar_min_thick, burn_bar_max_thick_frac·H]`
  (allows the 11 px SEM bar that §5.3's `bar_max_thick_frac` rejected; the min rejects
  1-px streaks);
- **length** `≥ burn_bar_min_len_frac · W`;
- **edge-adjacent**: within `burn_bar_edge_frac` of an image edge;
- **contrast**: the bar mean differs from the rows immediately above/below it by
  `≥ burn_bar_contrast_min`.

The **contrast** test is the key discriminator: a burned-in bar sits on a contrasting
background (white bar on a dark band; black bar on light), whereas a **dark TEM particle
interior** has dark surroundings → low contrast → rejected. Without it, near-black
sphere interiors were wrongly excluded (a regression caught during validation: wang
exclusion 8.5 % → 3.5 %, sphere-interior gouging → 0 px, after adding the contrast +
min-thickness guards).

**Effect.** Fixes B3: the SEM scale bar is fully excluded (576/576 px) and then blended
by A1; no particle interior is touched.

---

## 4b. Amendment A4 — bar length by full extent + larger bar dilation (changes §5.3)

**B4 (calibration accuracy).** A **bracketed** scale bar (`[━━━━]`, with end serifs)
has a body narrower than its full drawn width. §5.3 step 3 sets `L_px` = *median* run
over the bar rows, which measures the body and drops the serifs — on `wang2019` the body
is ~192 px but the serif-to-serif extent is ~208 px, so `L_px = 191.5` gave
`nm_per_px = 2.611`, ≈5.5 % high vs the ~202 px full extent (fails the 5 % exit
criterion).

**Amendment (`bar_length_method: extent`, default).** `L_px` = the bar's full horizontal
extent `x1 − x0` at the detection threshold (serifs and anti-aliased ends included), and
the pairing centre is the extent midpoint. `median` restores the original §5.3 step 3.
Solid bars are unchanged (extent ≈ median): girard 1.852 and pylypchuk2020 1.190 hold;
wang moves 2.611 → **2.404** (L_px 208), now within 5 % of the full-extent length (final
check is against the hand measurement, spec §12.2).

**Also:** `bar_dilate_px` (default 5, was a hard-coded 3) — a slightly larger dilation
around excluded bars removes the faint seam lines left along a bar's exclusion edge.

---

## 4c. Amendment A5 — `touches_exclusion` (changes §6.2, §8.5, §10)

**B5 (fill over a particle).** When furniture overlaps a real particle — the black "C"
box over a hollow sphere's corner (`xiong`), the "30" scale text under a sphere
(`pylypchuk2021`) — removing and filling it leaves a smear/wedge along the boundary.
SAM then traces the particle along the smear: the mask is mostly *outside* the exclusion,
so §6.2.3 keeps it, but with a wrong outline and diameter. A1 reduces the smear but
cannot reconstruct the occluded particle, so the diameter there is intrinsically
unreliable.

**Amendment (one flag, threaded through steps 2 and 6).** Step 2 already computes each
node's exclusion-overlap fraction for the §6.2.3 drop rule. Store it: a surviving node
with `excl_overlap > 0` is flagged `touches_exclusion` (§6.2). In step 6 a particle with
`touches_exclusion` is treated exactly like a `truncated` particle — **counted, but kept
out of the diameter stats** — with the flag propagating to its cavity/core/spots (§8.5),
a `touches_exclusion` column in `particles.csv` and `n_touches_exclusion` in `panel.json`
(§10), and `in_stats = not truncated and not touches_exclusion and not
POSSIBLE_CORE_ONLY`.

**Effect.** The `xiong` sphere whose corner was under the "C" box no longer enters the
size distribution with a smeared diameter; the already-truncated `pylypchuk` sphere was
out of stats regardless. No particle is *deleted* — the count is unchanged, only the
diameter statistics are protected. (Implemented in the spec now; enforced when Steps 2
and 6 are built — they are stubs. Cleanup already exposes the exclusion mask that step 2
must test against.)

---

## 4d. Amendment A6 — OCR-independent geometric label boxes (augments §5.4)

**B6 (label box OCR misses), found in the 40-panel review.** §5.4 and A2 both need an
OCR box to anchor a label box. On `wang2019` panel02 the panel letter "c" was read at
confidence 0.27 (< `ocr_conf_min` 0.30), so nothing anchored its white box and it leaked
into the SAM input as a hard-edged flat white rectangle (area ~10.9k px, 0 % excluded).

**Amendment (`geom_label_boxes`, default on).** Detect the box by shape, no OCR: a
connected component of the bright or dark binary with `area/bbox ≥ annot_box_fill_min`,
area in `[annot_box_area_min_frac, annot_box_area_max_frac] × panel`, whose bounding box
lies within `burn_bar_edge_frac` of an image edge → exclude, dilated `annot_dilate_px`.
Round/irregular particles have bbox fill ≈ π/4 < `annot_box_fill_min`, so they are
rejected; the area cap and edge test keep it off the (large, un-bounded) background.

**Effect.** wang02's "c" box is now excluded and blended. Across the 49 review panels the
detector fires on 19, each small (0.3–1.8 % of panel, none > 8 %); the three auto
calibrations are unchanged (girard 1.852, wang 2.404, pylypchuk2020 1.190).

---

## 5. New `config.yaml` parameters (all under `cleanup:`)

| Param | Default | Amendment | Meaning |
|---|---|---|---|
| `fill_method` | `texture` | A1 | `texture` (inpaint + matched noise) or `telea` (original §5.6) |
| `fill_noise_scale` | `1.0` | A1 | multiplier on estimated background-noise σ |
| `bar_exclude_scope` | `furniture` | A3.1 | `furniture` (burned-in + calibration bar) or `all` (original §5.3 step 4) |
| `annot_ocr_pure_bright` | `0.90` | A2 | OCR-anchored label patch is pure-bright above this |
| `annot_ocr_pure_dark` | `0.15` | A2 | …or pure-dark below this |
| `annot_grow_pad_frac` | `0.7` | A2 | grow window = OCR box padded by this × box size |
| `annot_grow_area_mult` | `8` | A2 | reject grown region above this × OCR-box area (fall back to padded box) |
| `annot_dilate_px` | `2` | A2 | dilation of an accepted annotation box |
| `burn_bar_pure_bright` | `0.95` | A3.2 | burned-in bar near-saturation (bright) |
| `burn_bar_pure_dark` | `0.05` | A3.2 | burned-in bar near-saturation (dark) |
| `burn_bar_std_max` | `0.02` | A3.2 | burned-in bars are ultra-flat |
| `burn_bar_edge_frac` | `0.20` | A3.2 | within this fraction of an image edge |
| `burn_bar_max_thick_frac` | `0.12` | A3.2 | max thickness (fraction of H) |
| `burn_bar_min_len_frac` | `0.03` | A3.2 | min length (fraction of W) |
| `burn_bar_min_thick` | `3` | A3.2 | min thickness (solid block, not a streak) |
| `burn_bar_contrast_min` | `0.20` | A3.2 | min contrast with rows above/below (rejects particle interiors) |
| `bar_length_method` | `extent` | A4 | `extent` (full drawn width incl. serifs) or `median` (original §5.3 step 3) |
| `bar_dilate_px` | `5` | A4 | exclusion dilation around bars (removes seam lines) |
| `geom_label_boxes` | `true` | A6 | detect filled bright/dark label boxes without OCR |
| `annot_box_area_min_frac` | `0.002` | A6 | min geometric label-box area (fraction of panel) |

Nothing in the original §4.5 config was changed except the earlier, separately-approved
`bar_min_thick: 3` (a tuning of §5.3, documented at the point of change).

---

## 6. Interaction with Step 2 (§6.2.3) — required for the guarantee

A1–A3 make the SAM input clean, but the **hard guarantee** that no furniture becomes a
particle is still §6.2.3: *drop a proposal if ≥ `excl_drop_frac` (0.5) of its filled
area is in the exclusion mask.* When Step 2 is implemented it **must** enforce §6.2.3
against the exclusion mask produced here. A1–A3 exist so that (a) SAM is not *tempted*
to segment furniture, and (b) furniture does not *fuse* with a real particle into a mask
that is < 50 % inside the exclusion (which §6.2.3 would then wrongly keep).

---

## 7. Validation (golden panels)

| Check | Before (v1) | After (v2) |
|---|---|---|
| xiong `C` box: un-excluded black px | 1074 | **0** |
| xiong SAM-input corner | solid black block | background-toned noise |
| xiong particle-rim bar-exclusion (xiong2017) | 4.1 % | **0 %** |
| wang fill: high-freq std (vs bg 0.058) | ~0 (smooth) | **0.046** |
| wang sphere-interior gouging | present | **0 px** |
| pylypchuk2021 SEM bar: px excluded | 104 / 576 | **576 / 576** |
| Auto-calibration (girard / wang / pylypchuk2020) | 1.852 / 2.611 / 1.190 | **unchanged** |
| Unit tests | 20 pass | **20 pass** |

---

## 8. Limitations / not addressed here

- **SEM data-band as a whole.** A3 removes the burned-in *bar*; the surrounding
  semi-transparent info band is left to §6.2.3 (its content is not a particle and is
  low-contrast). A dedicated band detector is possible future work.
- **Colored annotations** (e.g. the red "f)") remain out of scope per original §1; they
  are handled by OCR text exclusion when readable and otherwise by §6.2.3.
- **Fill near particle edges.** A1's noise is matched to a global background estimate; a
  fill straddling a particle boundary is plausible but not exemplar-perfect. §6.2.3 is
  the backstop.
- Calibration for the SEM panels still relies on a **manual `nm_per_px`** (original
  §5.5 priority; §12.2 hand measurements) — A3 removes the bar but does not read a
  colored data-bar value.

---

## 8a. Amended test expectations

Added to `tests/test_cleanup.py`:
- `test_bar_min_thick_rejects_thin_streak` — 1-px streak dropped, 3-px bar kept.
- (Planned) burned-in white-bar detection and OCR-anchored dark-box exclusion once toy
  fixtures are added; currently validated on the golden panels via `runs/…`.

`calibrate()` now returns a 6-tuple `(nm_per_px, cal_source, L_px, value_nm,
scale_text, cal_bar)`; the two calibration unit tests were updated accordingly.

---

## 9. References (consulted for this revision)

- Ⓡ1 Automated SEM-based nanoparticle metrology (segmentation + robust scale-bar
  recognition), PMC13441050 — pipelines *localize and exclude* the scale bar before
  segmentation. https://pmc.ncbi.nlm.nih.gov/articles/PMC13441050/
- Ⓡ2 Wavelet-based inpainting detection, arXiv:2408.06429 — object removal alters the
  noise statistics of the filled segment. https://arxiv.org/pdf/2408.06429
- Ⓡ3 Criminisi et al., "Region Filling and Object Removal by Exemplar-Based Image
  Inpainting" — texture is preserved by sampling real background patches.
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2003-84.pdf
- Ⓡ4 `kleinerELM/remove_SEM_scalebar` — practical removal of the SEM info/scale bar
  region. https://github.com/kleinerELM/remove_SEM_scalebar
- Ⓡ5 LLM-assisted scale-bar detection & extraction for SEM, arXiv:2510.11260.
  https://arxiv.org/html/2510.11260
