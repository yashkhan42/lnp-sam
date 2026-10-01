# LNP-SAM MVP Spec

**Goal.** A working prototype that takes lignin nanoparticle (LNP) micrograph panels and returns calibrated particle diameters, fixing the failures NP-SAM showed on our 103-panel baseline. It must beat NP-SAM on 12 golden panels (section 12).

**Core idea.** NP-SAM judges each SAM mask alone and deletes masks with fixed filters. LNP-SAM keeps all of SAM's candidate masks, works out how they relate (duplicate, inside, touching), and only then decides what each mask is.

**Revision v2 (cleanup robustness).** Sections 1, 4.5, 5.3–5.6, 6.2, 8.5 and 10 incorporate the v2 amendments that fix SAM-input defects found on the golden panels: particle rims masked as bars; a hard black label-box rectangle left in; smooth inpaint blobs; an undetected burned-in SEM scale bar; scale-bar length under-measured by dropping end serifs; and fills that smear a particle they overlap. `LNP-SAM_MVP_SPEC_AMENDMENTS.md` holds the defect evidence, root-cause analysis, validation numbers and references; this spec states the resulting behaviour. Design principle: the exclusion mask is the source of truth for "not a particle" and must be complete and precise; filling only blends; a particle that overlaps the exclusion or the border is kept but out of the diameter stats (`touches_exclusion`, 6.2/8.5/10); and the hard guarantee is section 6.2.3 (drop any SAM proposal ≥ `excl_drop_frac` inside the exclusion), which Step 2 must enforce against the mask cleanup produces.

**Revision v3 (resolve robustness, scope).** Sections 1, 4.5, 8.1, 8.3, 8.4, 12.1, 12.2 and 12.4 incorporate the v3 amendments found by running Step 4 on the golden panels: seeds of the wrong polarity inverting ρ (A); rim-dark/rim-bright particles typed as background (B); specks and enclosed gaps distorting the group rules (C); close-up single particles typed as aggregates (D); hollowness defined by a single centred hole of any size (E); and a particle filling the frame typed as background (F). Transparent, low-contrast TEM particles are out of MVP scope, so the three xiong panels showing them leave the golden set (12 gating panels). Each rule change has a config switch restoring the original rule; `LNP-SAM_MVP_SPEC_AMENDMENTS.md` (§10–11) holds the evidence.

## 1. Scope

| In the MVP | Not in the MVP (later) |
|---|---|
| Crop margins; remove text, scale bars and label boxes before SAM; remove burned-in (SEM) scale bars and particle-fused label boxes (v2, §5.3–5.4) | SEM data-**band** value parsing, colored annotations |
| Read nm/px from the scale bar; manual value overrides | Calibration uncertainty |
| SAM ViT-B automatic masks with relaxed thresholds | Tiling, other backends (μSAM, SAM 3) |
| Relation graph: duplicates, containment, adjacency | — |
| Rules: particle, gap, speck, background, orphan; cavity, core, stacked, spot; merged pair split; aggregate flag | Film class, rim-fragment merging, learned classifiers |
| Keep edge-truncated particles (flagged) | Measuring them by arc fit; edge correction |
| Recover the missing parent of an orphan cavity by prompting SAM | Recovering particles from uncovered foreground |
| Hollowness from masks (cavity child) | Radial-profile physics model |
| Aggregates flagged and reported separately | Primary-particle sizing inside aggregates |
| Diameter = equivalent-area diameter; panel stats | Ellipse fits, morphology classes, quality gate |
| Evaluation on 12 annotated golden panels vs NP-SAM | Full benchmark, other baselines |
| Particles with clear solidity in SEM and TEM (v3) | Transparent / low-contrast TEM particles (e.g. thin-walled shells imaged as near-transparent), whose interiors read as cavities |

## 2. Why NP-SAM fails on lignin (what not to copy)

From reading `npsam` 3.2.3 (`npsam.py`, `utils.py`):

| NP-SAM behavior | Lignin failure it causes | LNP-SAM replacement |
|---|---|---|
| SAM box NMS at 0.7 | Thin-shell hollow particles lose either the outer mask or the core | NMS 0.95; relations decide (sections 7–8) |
| Edge filter drops any mask touching the outer 3 px, one mask at a time | Particle at the edge dropped, its pore kept as a "particle" | Truncated particles kept and flagged; children follow their parent (8.5) |
| Shape filter keeps a mask only if `area < 400 px` or `solidity > 0.95` | Small junk (gaps, specks, text glyphs) always survives; aggregates, rings and merged pairs always deleted | No shape filter. Rules use panel-normalized contrast and relations (8.3–8.4) |
| No text or scale-bar handling | Text and bars counted as particles (40 panels) | Removed before SAM (section 5) |
| Discards SAM's `predicted_iou` and `stability_score` | No quality signal downstream | Kept (section 6) |

**Rule for implementers:** no code in LNP-SAM may drop a mask for touching the border or for its area/solidity alone.

## 3. Pipeline

| Step | Input → output | Purpose |
|---|---|---|
| 1. Cleanup | panel → content box, exclusion mask, nm/px, SAM input image | Remove figure furniture, read the scale |
| 2. Proposals | SAM input → candidate masks with scores and features | Permissive: keep everything plausible |
| 3. Graph | candidates → duplicate clusters, parent/child tree, adjacency | Relations between masks |
| 4. Resolve | graph → a role for every mask | The rules |
| 5. Recover | orphan cavities → recovered parent particles; rerun 3–4 once | Fix "pore kept, particle lost" |
| 6. Measure | resolved particles → particle table, panel summary, overlay | Output |

## 4. Setup

### 4.1 Environment

Python 3.11. `torch`, `segment-anything` (`pip install git+https://github.com/facebookresearch/segment-anything.git`), checkpoint `sam_vit_b_01ec64.pth`, `numpy`, `scipy`, `scikit-image`, `opencv-python-headless`, `pandas`, `pyarrow`, `pycocotools`, `easyocr`, `pyyaml`, `pytest`.

NP-SAM baseline runs in a separate environment with `npsam==3.2.3` (its pinned dependencies conflict).

### 4.2 Repository

```
lnpsam/
  config.yaml
  lnpsam/
    io.py          # manifest, image loading, RLE helpers
    cleanup.py     # step 1
    proposals.py   # step 2
    graph.py       # step 3
    resolve.py     # step 4
    recover.py     # step 5
    measure.py     # step 6
    overlay.py
    run.py         # CLI
  baselines/npsam_run.py
  eval/evaluate.py
  tests/
```

Run: `python -m lnpsam.run --manifest data/manifest.csv --config config.yaml --out runs/<name>`

### 4.3 Input manifest (`data/manifest.csv`)

| Column | Required | Meaning |
|---|---|---|
| `panel_id` | yes | Unique, e.g. `girard2024_acssuschemeng_4c00711__p08_x658__panel06` |
| `paper_id` | yes | Part of `panel_id` before the first `__` |
| `image_path` | yes | Path to PNG/JPEG/TIFF |
| `modality` | yes | `TEM` or `SEM` (bright-field TEM assumed for `TEM`) |
| `nm_per_px` | no | Manual calibration; overrides step 1 |

### 4.4 Conventions

| Item | Convention |
|---|---|
| Image | `gray`: float32 in [0, 1] (`0.299R + 0.587G + 0.114B`; 16-bit ÷ 65535; alpha composited on white). `rgb_u8`: uint8 RGB for SAM (gray replicated if needed). |
| Coordinates | Arrays `[row, col]`; points `(x, y)`; boxes `(x0, y0, x1, y1)`, end-exclusive. Everything written to disk is in full-panel coordinates. |
| Masks | COCO RLE (`pycocotools.mask.encode(np.asfortranarray(m.astype(np.uint8)))`), stored as JSON strings. Never store particles as a single label image: masks overlap. |
| Filled mask | `scipy.ndimage.binary_fill_holes(mask)`. All relations and particle outlines use filled masks. |
| Thresholds | All in `config.yaml`. No numeric thresholds hardcoded. |
| Traceability | Every mask gets `role` and `rule_id`. If a case matches no rule, keep the mask, set `role = unresolved`, and log it. Never drop silently. |

### 4.5 `config.yaml` (defaults)

```yaml
cleanup:
  margin_std_max: 0.03        # a border row/col is margin if std below this...
  margin_white: 0.92          # ...and mean above this
  margin_black: 0.08          # ...or below this
  margin_scan_max_frac: 0.25
  ocr_upscale: 2              # 3 if min(H, W) < 300
  ocr_conf_min: 0.3
  text_h_frac: [0.015, 0.15]  # accepted text-box height, fraction of panel height
  text_dilate_frac: 0.2       # dilation = max(2 px, frac * box height)
  bar_bright: 0.85
  bar_dark: 0.15
  bar_min_len_frac: 0.03      # of panel width
  bar_max_thick_frac: 0.04    # of panel height (3 px minimum allowed)
  bar_min_aspect: 6
  bar_flat_std_max: 0.03      # drawn bars are flat along their length
  annot_box_fill_min: 0.90
  annot_box_area_max_frac: 0.10
  nm_per_px_range: [0.05, 200]
  inpaint_radius: 5
  bar_min_thick: 3            # min calibration-bar thickness px (drops thin rim streaks)
  # --- v2 cleanup (see 5.3-5.6 and LNP-SAM_MVP_SPEC_AMENDMENTS.md) ---
  fill_method: texture        # 5.6: 'texture' (inpaint + matched noise) | 'telea'
  fill_noise_scale: 1.0       # 5.6: multiplier on estimated background-noise sigma
  bar_exclude_scope: furniture # 5.3: bars entering the exclusion: 'furniture' | 'all'
  annot_ocr_pure_bright: 0.90 # 5.4: OCR-anchored label patch is pure-bright above this
  annot_ocr_pure_dark: 0.15   # 5.4: ...or pure-dark below this
  annot_grow_pad_frac: 0.7    # 5.4: grow window = OCR box padded by this * box size
  annot_grow_area_mult: 8     # 5.4: reject grown region above this * OCR-box area
  annot_dilate_px: 2          # 5.4: dilation of an accepted annotation box
  burn_bar_pure_bright: 0.95  # 5.3: burned-in bar near-saturation (bright)
  burn_bar_pure_dark: 0.05    # 5.3: burned-in bar near-saturation (dark)
  burn_bar_std_max: 0.02      # 5.3: burned-in bars are ultra-flat
  burn_bar_edge_frac: 0.20    # 5.3: within this fraction of an image edge
  burn_bar_max_thick_frac: 0.12 # 5.3: max thickness (fraction of H)
  burn_bar_min_len_frac: 0.03 # 5.3: min length (fraction of W)
  burn_bar_min_thick: 3       # 5.3: a solid block, not a 1-px streak
  burn_bar_contrast_min: 0.20 # 5.3: min contrast with rows above/below (rejects particle interiors)
  bar_length_method: extent   # 5.3: 'extent' (full drawn width incl. serifs) | 'median' (original step 3)
  bar_dilate_px: 5            # 5.3: exclusion dilation around bars (removes seam lines)
  geom_label_boxes: true      # 5.4: detect filled bright/dark label boxes without OCR
  annot_box_area_min_frac: 0.002 # 5.4: min geometric label-box area (fraction of panel)

sam:
  checkpoint: sam_vit_b_01ec64.pth
  points_per_side: 64
  points_per_batch: 128
  pred_iou_thresh: 0.80       # NP-SAM: 0.88
  stability_score_thresh: 0.85  # NP-SAM: 0.95
  box_nms_thresh: 0.95        # NP-SAM: 0.7
  min_piece_px: 4
  excl_drop_frac: 0.5
  bg_area_frac: 0.25

graph:
  dup_iou: 0.85
  contain_frac: 0.90
  contain_area_ratio: 1.10
  adjacent_px: 2

resolve:
  seed: {solidity: 0.90, circularity: 0.75, quality: 0.85, min_d_px: 6, max_area_frac: 0.10, min_count: 3}
  seed_relaxed: {circularity: 0.60, quality: 0.75}
  polarity_prior: {TEM: -1, SEM: 1}   # -1: particles darker than background
  seed_polarity_check: true   # 8.1 (v3 A): seeds must lie on the polarity_prior side of mu_bg
  bg_rules_use: pl            # 8.3 (v3 B): R-BGPATCH tests 'pl' (rim included) | 'rho_in' (original)
  dense_bg_min_frac: 0.05
  rho_bg_max: 0.35
  rho_particle_min: 0.50
  rim_band_frac: 0.30
  ring_width_frac: 0.25
  gap_contact_min: 0.50
  orphan: {circularity: 0.70, ring_free_min: 0.50, ring_rho_min: 0.60, ring_std_max: 0.25}
  particle: {edge_min: 0.25, quality_min: 0.70, max_area_frac: 0.25}
  large_round: {enabled: true, circularity: 0.75, solidity: 0.90}  # 8.3 (v3 D): large round solid mask -> particle
  frame_particle: true        # 8.1 (v3 F): round solid bg_candidate on the particle side is a particle, not background
  min_particle_nm: 10         # PI to confirm
  speck_rel_d: 0.20           # used when uncalibrated
  split: {max_children: 3, r_min: 0.45, coverage_min: 0.70}
  aggregate: {min_children: 3, coverage_min: 0.40}
  group_members_only: true    # 8.4 (v3 C): C-SPLIT/C-AGG count only particle-like children >= d_min
  cavity: {rule: single_hole, r_min: 0.25, r_max: 0.95, delta_max: 0.25, rho_margin: 0.15}  # 8.4 (v3 E): 'single_hole' | 'size' (original)
  stacked: {r_min: 0.25, delta_min: 0.25, circularity: 0.80, rho_tol: 0.05}
  spot_r_max: 0.25

recover:
  box_scales: [1.6, 2.2]
  contain_min: 0.90
  circularity_min: 0.70
  ring_rho_min: 0.50
  swallow_max: 0.50

measure:
  min_particles_for_stats: 5
```

## 5. Step 1: Cleanup

### 5.1 Content box

For each side, scan inward line by line (at most `margin_scan_max_frac` of the dimension). A line is margin if `std < margin_std_max` and (`mean > margin_white` or `mean < margin_black`). Stop at the first non-margin line. The remaining rectangle is `content_box`.

### 5.2 Text

1. Upscale `gray` by `ocr_upscale` (`cv2.INTER_CUBIC`), convert to uint8.
2. `easyocr.Reader(["en"]).readtext(img, detail=1, paragraph=False)`; create the reader once per process.
3. Convert each box to an axis-aligned bbox in panel coordinates. Keep boxes with confidence ≥ `ocr_conf_min` and height within `text_h_frac × H`.
4. Each kept box → exclusion, dilated by `max(2, text_dilate_frac × box height)`.

### 5.3 Scale bar

Two detectors run. **Calibration bars** feed the scale reading (5.5); **burned-in furniture bars** feed the exclusion mask (5.6). Only *furniture* bars are excluded, so imaged particle rims are never masked (v2; `bar_exclude_scope: all` restores the original "every bar → exclusion" rule).

**Calibration bars.**

1. For each polarity (bright: `gray ≥ bar_bright`; dark: `gray ≤ bar_dark`), find for every row the longest horizontal run.
2. Keep rows whose run ≥ `bar_min_len_frac × W`. Group consecutive kept rows whose runs overlap ≥ 90% into a candidate. A candidate is a bar if its thickness is within `[bar_min_thick, max(3, bar_max_thick_frac × H)]`, length/thickness ≥ `bar_min_aspect`, and the std of `gray` along each of its rows ≤ `bar_flat_std_max`.
3. Bar length `L_px` = median run length over rows whose run ≥ 0.8 × the candidate's longest run (ignores end ticks).

**Burned-in furniture bars (v2)** catch burned-in scale bars (e.g. SEM) that exceed the calibration detector's thickness cap. A candidate run is a burned-in bar when **all** hold: near-saturated (`≥ burn_bar_pure_bright` or `≤ burn_bar_pure_dark`); per-row std ≤ `burn_bar_std_max`; thickness in `[burn_bar_min_thick, burn_bar_max_thick_frac × H]`; run ≥ `burn_bar_min_len_frac × W`; some part within `burn_bar_edge_frac` of an image edge; and the bar mean differs from the rows immediately above/below it by ≥ `burn_bar_contrast_min`. The contrast test rejects dark particle interiors (dark surround → low contrast).

4. **Exclusion.** Each burned-in bar, plus the single bar the calibration paired with a scale label (5.5), → exclusion, dilated 3 px. Other calibration-candidate bars are not excluded.

### 5.4 Label boxes

Filled rectangles behind panel letters (e.g. the white "a" box in every `wang2019` panel, ~11,500 px). Connected components of the bright or dark binary image with `area / bbox area ≥ annot_box_fill_min`, area ≤ `annot_box_area_max_frac × panel area`, containing an OCR box → exclusion, dilated 2 px.

**OCR-anchored annotation boxes (v2).** The connected-component test above misses a label box fused (8-connected) to particles — e.g. a black panel-letter box touching dark spheres forms one non-rectangular component (this left a hard black rectangle in the SAM input). So, additionally: for each OCR text box whose interior median intensity is pure-bright (`≥ annot_ocr_pure_bright`) or pure-dark (`≤ annot_ocr_pure_dark`), region-grow that patch **within a bounded window** (the text box padded by `annot_grow_pad_frac × max(box side)`), take the connected component overlapping the text box, and exclude it, dilated `annot_dilate_px`. If the grown area exceeds `annot_grow_area_mult ×` the text-box area, fall back to the padded text box. Text on ordinary background (neither pure) is not a label box.

**OCR-independent geometric label boxes (v2, `geom_label_boxes`).** Both detectors above need OCR to anchor the box; when the panel letter's OCR confidence falls below `ocr_conf_min` the box leaks as a flat block (e.g. `wang2019` panel02's white "c" box, letter read at conf 0.27). So, additionally, detect the box by shape: a connected component of the bright or dark binary with `area / bbox area ≥ annot_box_fill_min`, area in `[annot_box_area_min_frac, annot_box_area_max_frac] × panel area`, and its bounding box within `burn_bar_edge_frac` of an image edge → exclusion, dilated `annot_dilate_px`. Round/irregular particles have low bbox fill and are rejected; the area and edge limits keep it off the background.

### 5.5 Calibration

1. Normalize OCR text: remove spaces; inside digit runs map `O→0`, `l→1`, `I→1`, `,→.`.
2. Scale label regex (case-insensitive): `^(\d+(?:\.\d+)?)(nm|um|µm|μm|jm|pm)$`. Units to nm: `nm` → 1; `um`, `µm`, `μm`, `jm` → 1000; `pm` → 1000 (OCR usually misreads µm as pm; the plausibility check catches the rest).
   - **Label candidates (v2).** Because EasyOCR groups words differently across platforms (e.g. Linux/CPU returns `200` and `nm` as two boxes), build candidates from single OCR boxes **and** same-line merged pairs — two boxes whose vertical centres are within `0.5 ×` the taller box's height and whose horizontal gap is `≤ 1.5 ×` that height, concatenated left-to-right, with the union as the candidate box. Parse both; prefer a single-box match when both are valid. The candidate (merged union) box anchors bar pairing and is added to the exclusion.
3. Pair each candidate label with the bar whose vertical gap to it ≤ 3 label heights and whose center is within `0.75 × L_px` horizontally; if several bars qualify, take the one with the largest horizontal overlap with the label.
4. `nm_per_px = value_nm / L_px`. Reject if outside `nm_per_px_range`. Record the paired bar; it is the calibration bar excluded in 5.3 step 4 (v2).
5. Priority: manifest `nm_per_px` > auto > missing. Record `cal_source` ∈ {`manual`, `auto`, `missing`}. If missing, everything runs in px and nm columns are empty.

### 5.6 SAM input

Crop `rgb_u8` to `content_box` and fill excluded pixels inside the crop (`fill_method`):

- `telea`: `cv2.inpaint(..., inpaint_radius, cv2.INPAINT_TELEA)` per channel (original).
- `texture` (default, v2): use the TELEA result as a base, then re-inject Gaussian noise matched to the background's high-frequency amplitude — `σ_bg = 1.4826 · MAD(gray − GaussianBlur(gray, 2))` over un-excluded pixels, scaled by `fill_noise_scale` — into the filled pixels only, then clip. This blends filled regions into the noise field instead of leaving the smooth blobs SAM would segment.

`content_mask = content_box & ~exclusion`. Keep the crop offset; convert all later coordinates back to full-panel coordinates before writing.

## 6. Step 2: Proposals

### 6.1 SAM

`SamAutomaticMaskGenerator(sam, points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh, box_nms_thresh=..., crop_n_layers=0, min_mask_region_area=0, output_mode="binary_mask")` on the cropped, inpainted image. Keep `segmentation`, `predicted_iou`, `stability_score`, `point_coords`.

The relaxed thresholds are deliberate: they let the outer mask of thin-shell particles and the outlines of fused particles survive. Expect several times more masks than NP-SAM. Step 4 rejects the junk.

### 6.2 Post-processing (the only removals before step 4)

1. Split each mask into 8-connected pieces; drop pieces < `min_piece_px`. The largest piece keeps the mask's id; other pieces become new nodes with the same scores.
2. `filled = binary_fill_holes(mask)`. Store both `rle` and `filled_rle`, and `n_holes` (hole components ≥ 4 px).
3. Compute `excl_overlap` = fraction of the node's filled area inside the exclusion mask. Drop the node if `excl_overlap ≥ excl_drop_frac` — the hard guarantee that furniture (text, bars, label boxes; section 5) never becomes a particle; 5.6's fill only makes SAM less likely to propose there in the first place, so this check must run against the exclusion mask cleanup produced. A surviving node with `excl_overlap > 0` is flagged `touches_exclusion` (v2): part of its outline lies under removed furniture, which the fill only approximated, so its diameter is unreliable (see 8.5, 10.1). Store `excl_overlap` on the node.
4. Drop exact duplicates (identical `filled_rle`), keeping the higher `quality`.
5. `bg_candidate = True` if filled area ≥ `bg_area_frac × content area` and it touches ≥ 2 content edges.

### 6.3 Node features (on the filled mask)

| Feature | Definition |
|---|---|
| `area`, `eqdiam` | pixels; `eqdiam = sqrt(4·area/π)` |
| `solidity`, `circularity` | `circularity = 4π·area / perimeter²` (`skimage.measure.perimeter`), clipped to 1 |
| `centroid`, `bbox` | |
| `touches_border` | Any pixel within 1 px of the content box edge |
| `mean_int` | Mean `gray` inside |
| `edge_contrast` | `G = \|sobel(gaussian(gray, 1))\|`; mean of `G` on the mask boundary (`find_boundaries(mode="inner")`) ÷ 95th percentile of `G` over `content_mask` |
| `quality` | `0.5·predicted_iou + 0.3·stability_score + 0.2·min(edge_contrast, 1)` |

## 7. Step 3: Graph

1. **Pairs.** Find node pairs whose bboxes intersect or are within `adjacent_px` (sort by `x0` and sweep). For each, compute on the filled masks: `inter`, `iou`, `c_ab = inter/area_a` (fraction of a inside b), `c_ba`.
2. **Duplicates.** Union-find over pairs with `iou ≥ dup_iou`. The member with the highest `quality` is the representative; the others get `role = duplicate`. Only representatives continue.
3. **Parents.** `parent(i)` = the smallest representative `j` with `c_ij ≥ contain_frac` and `area_j ≥ contain_area_ratio × area_i`. `bg_candidate` nodes are never parents. This forms a tree (a forest).
4. **Adjacency.** Representatives with `inter == 0` whose masks come within `adjacent_px` of each other are adjacent.
5. **Children coverage.** For each parent: `coverage = |∪ filled children| / area(parent)`.

## 8. Step 4: Resolve

### 8.1 Panel statistics

Intensity thresholds must work across TEM (dark particles) and SEM (bright particles) and across contrast-adjusted figures, so every intensity is expressed as **ρ**: 0 at the background level, 1 at the typical particle level.

0. **Frame-filling particles (v3 F, `frame_particle`).** Step 2 flags `bg_candidate` by size and edge contact only (6.2.5), so a close-up particle that fills the frame qualifies, is never a parent (7.3), and is typed `background`. Before anything else, a `bg_candidate` representative that is round and solid (`large_round.circularity`, `large_round.solidity`) and whose own (unfilled) pixels lie on the particle side of the content outside its filled mask (`s · (mean_own − mean_rest) > 0`) has its flag cleared, and the Step-3 graph is rebuilt so it can parent its pore. The panel is flagged `FRAME_PARTICLE`. A real background mask fails one test: hole-filled it is a near-rectangle (round and solid) but on the background side, or it is irregular. This runs in Step 4, after the Step-2 cache, so no SAM rerun is needed.
1. **Seeds** (confident isolated particles): representatives with no parent, not `bg_candidate`, not `touches_border`, meeting `resolve.seed`, and (v3 A, `seed_polarity_check`) on the particle side of the background: `s · (mean_int − mu_bg) > 0` with `s = polarity_prior[modality]`. If fewer than `min_count`, retry with `seed_relaxed`. `mu_bg` (step 2) never uses seeds, so it is computed first. Without the polarity test, on clustered or dense panels the only isolated round masks are pores or background holes, which inverts ρ (the control panel lost every particle).
2. **Background.** `U = content_mask & ~dilate(∪ filled representatives except bg_candidates, 2 px)`. If `|U| ≥ dense_bg_min_frac × content area`: `mu_bg = median(gray[U])`. Otherwise (densely packed panel): polarity `s = polarity_prior[modality]`, and `mu_bg` = 10th percentile of content gray if `s = +1`, 90th if `s = −1`.
3. **Particle level.** `mu_p = median(mean_int of seeds)`. If there are no seeds, set `mu_p` = the 90th percentile of content gray if `s = +1`, 10th if `s = −1`, and flag the panel `NO_SEEDS`.
4. **ρ** of any pixel set `R`: `ρ(R) = (mean(gray[R]) − mu_bg) / (mu_p − mu_bg)`.
5. **Reference size.** `d_ref` = median `eqdiam` of seeds (px). `d_min` = `min_particle_nm / nm_per_px` if calibrated, else `speck_rel_d × d_ref`; never below 3 px.

### 8.2 Per-node values

| Value | Definition |
|---|---|
| `rho_in` | ρ of the filled mask |
| `rho_rim` | ρ of filled pixels within `rim_band_frac × (eqdiam/2)` of the boundary |
| `pl` | `max(rho_in, rho_rim)`: particle-likeness. Hollow TEM particles have a dark rim and a light center, so `rho_rim` catches them. |
| `ring` | Pixels 1 to `max(2, ring_width_frac × eqdiam/2)` px outside the mask, inside `content_mask` |
| `ring_free` | `ring` minus all other representatives' filled masks |
| `ring_free_frac` | `\|ring_free\| / \|ring\|` |
| `ring_rho`, `ring_std` | ρ of `ring_free`; std of ρ over 12 equal angular sectors of `ring_free` |
| `contact` | Fraction of boundary pixels within `adjacent_px` of another representative with `pl ≥ rho_particle_min` |

### 8.3 Roots (nodes without a parent)

First matching rule wins:

| Rule | Condition | Role |
|---|---|---|
| R-BG | `bg_candidate` and `rho_in ≤ rho_bg_max` | `background` |
| R-SPECK | `eqdiam < d_min` | `speck` |
| R-ORPHAN | `circularity ≥ orphan.circularity`, `ring_free_frac ≥ orphan.ring_free_min`, `ring_rho ≥ orphan.ring_rho_min`, `ring_std ≤ orphan.ring_std_max` | `orphan` (subtype `cavity` if `rho_in < rho_particle_min`, else `core`) |
| R-GAP | `rho_in ≤ rho_bg_max` and `contact ≥ gap_contact_min` | `gap` |
| R-BGPATCH | `pl ≤ rho_bg_max` (v3 B; `bg_rules_use: rho_in` restores `rho_in`) | `bg_patch` |
| R-PARTICLE-LARGE | (v3 D, `large_round.enabled`) `pl ≥ rho_particle_min`, area above `particle.max_area_frac × content area`, `circularity ≥ large_round.circularity`, `solidity ≥ large_round.solidity` | `particle` |
| R-LARGE | `pl ≥ rho_particle_min` and area above that limit | `aggregate` |
| R-PARTICLE | `pl ≥ rho_particle_min`, `edge_contrast ≥ particle.edge_min`, `quality ≥ particle.quality_min`, `area ≤ particle.max_area_frac × content area` | `particle` |
| R-REJECT | otherwise | `rejected` |

Why orphan comes before gap: both are surrounded by particle material. An orphan (a cavity whose particle SAM missed) is round and its surroundings are covered by **no** mask, so `ring_free_frac` is high and the ring is even. An interstitial gap between touching spheres has circularity around 0.2 and its surroundings are the neighbors' masks.

Why R-BGPATCH tests `pl` but R-GAP keeps `rho_in` (v3 B): `pl` exists so a particle whose interior is at background level but whose rim is particle-like (a hollow TEM particle; an SEM sphere with a bright rim) is not called background. A gap's outline runs along its neighbours' edges, so its rim is particle-like by construction; testing `pl` there would turn gaps into particles. A gap is told apart by `contact`.

Why R-PARTICLE-LARGE (v3 D): a close-up of one particle can fill more than a quarter of the panel; a round, solid mask is one particle, not a cluster.

**Re-rooting.** If a node's parent became `background`, `bg_patch`, `speck`, `gap` or `rejected`, move it to the nearest ancestor that is a `particle` or `aggregate`, or make it a root and type it with this table. Repeat until nothing changes.

### 8.4 Children of a particle

For child `c` of particle `p`:

| Symbol | Definition |
|---|---|
| `r` | `eqdiam(c) / eqdiam(p)` |
| `δ` | centroid distance ÷ `eqdiam(p)/2` |
| `rho_shell` | ρ of `filled(p)` minus all its children (dilated 1 px) |
| particle-like | `pl(c) ≥ rho_particle_min` |

**Group rules on the parent first:**

| Rule | Condition | Result |
|---|---|---|
| C-SPLIT | 2 to `split.max_children` group children, all particle-like with `r ≥ split.r_min`, `coverage ≥ split.coverage_min` | Parent was two or three touching particles merged into one mask (NP-SAM's solidity filter used to delete these). Parent → `merged`; children become roots, typed by 8.3. |
| C-AGG | ≥ `aggregate.min_children` particle-like group children, `coverage ≥ aggregate.coverage_min` | Parent → `aggregate`; particle-like group children → `particle` with `aggregate_id` = parent |

**Group children (v3 C, `group_members_only`).** The group rules count only particle-like children with `eqdiam ≥ d_min`, and `coverage` is computed over those. Other children neither count toward a group rule nor block it: a speck does not turn a single particle into an aggregate, and the gap that a hole-filled merged outline encloses between its particles does not stop C-SPLIT. After C-SPLIT every child, group or not, becomes a root (the enclosed gap is then typed `gap`). With the switch off, all children count (original rule).

**Then each remaining child.** The "dominant" child is the largest one, if every other child has `r ≤ spot_r_max`. The core rule (and, with `cavity.rule: size`, the cavity rule) applies only to the dominant child.

**Hollowness (v3 E, `cavity.rule: single_hole`).** A particle is hollow when it shows exactly one hole, approximately at its centre, of any size. A *hole* is a child that is background-dark (`rho_in < rho_particle_min`) and darker than the shell (`rho_in ≤ rho_shell − cavity.rho_margin`). SAM often outlines one hole with several overlapping masks, so overlapping hole masks count as one hole, represented by its largest mask; hole-like children smaller than `d_min` are noise and do not count. If there is exactly one hole and it has `δ ≤ cavity.delta_max` and `r ≤ cavity.r_max`, it is the cavity; every other hole-like child is a spot. The absolute darkness test keeps SEM edge brightening (a solid sphere's centre darker than its bright rim but still particle-bright) from reading as a hole. `cavity.rule: size` restores the original rule below.

| Rule | Condition | Result |
|---|---|---|
| C-CAVITY | `single_hole`: the one centred hole above. `size` (original): dominant, `cavity.r_min ≤ r ≤ cavity.r_max`, `δ ≤ cavity.delta_max`, `rho_in(c) ≤ rho_shell − cavity.rho_margin` | `cavity` of `p` |
| C-HOLE-SPOT | `single_hole` only: any other hole-like child (a second hole, an off-centre hole, a speck-sized hole) | `spot` (dropped; `n_spots` on parent) |
| C-CORE | dominant, same `r` and `δ` limits, `rho_in(c) ≥ rho_shell + cavity.rho_margin` | `core` of `p` |
| C-STACKED | `r ≥ stacked.r_min`, `δ > stacked.delta_min`, `circularity ≥ stacked.circularity`, `rho_in(c) ≥ rho_shell − stacked.rho_tol` | separate `particle`, also typed as a parent with these rules |
| C-SPOT | `r < spot_r_max` | `spot` (dropped; `n_spots` on parent) |
| C-UNRESOLVED | otherwise | `unresolved` (dropped from counts, kept in `nodes.parquet`) |

Children of a `cavity` or `core` are `spot` (the center-dot case: shell + core + dot → one particle with one cavity). Children of dropped nodes move up to the nearest surviving ancestor and are typed again.

### 8.5 Edges

A `particle` with `touches_border` gets `truncated = True`. A `particle` flagged `touches_exclusion` (6.2; its filled mask overlaps removed furniture) is treated the same way — kept and counted, but out of the diameter stats (10.1) — because part of its outline is under furniture the fill only approximated, so SAM may trace the wrong boundary there. Both flags propagate to the particle's cavity, core and spots. Nothing is deleted for touching the border or the exclusion.

### 8.6 Output of step 4

Every representative has `role` and `rule_id`. Roles: `particle`, `cavity`, `core`, `spot`, `merged`, `aggregate`, `orphan`, `gap`, `bg_patch`, `background`, `speck`, `rejected`, `unresolved` (plus `duplicate` from step 3).

## 9. Step 5: Recover orphans

For each `orphan` with diameter `d_o` and centroid `c_o`:

1. Reuse the SAM embedding from step 2 (`SamPredictor.set_image` on the same cropped image; cache the features, don't recompute).
2. For each scale in `box_scales`: box prompt = square of side `scale × d_o` centered on `c_o`, clipped to the content box. `predictor.predict(box=..., multimask_output=True)` returns 3 masks with scores and low-res logits.
3. A returned mask `m` is valid if: it contains ≥ `contain_min` of the orphan; `circularity(fill(m)) ≥ circularity_min`; ρ of `fill(m)` minus the orphan ≥ `ring_rho_min`; no existing `particle` has more than `swallow_max` of its area inside `m`.
4. Take the valid mask with the highest score. Add it as a new node with `source = recover`, `predicted_iou` = its score, `stability_score` from `segment_anything.utils.amg.calculate_stability_score(logits, 0.0, 1.0)`, features as in 6.3.
5. After all orphans: rerun steps 3 and 4 once on all nodes. No further loops.
6. Orphans still without a parent: `cavity` subtype → `unresolved`, flag `ORPHAN_UNRECOVERED`; `core` subtype → `particle`, flag `POSSIBLE_CORE_ONLY`.

## 10. Step 6: Measure and write

### 10.1 Per particle (`particles.csv`)

`particle_id`, `panel_id`, `paper_id`, `d_px` (eqdiam of filled mask), `d_nm`, `feret_max_nm`, `aspect` (`axis_major_length / axis_minor_length`), `area_nm2`, `cavity_d_nm`, `shell_nm = (d_nm − cavity_d_nm)/2`, `core_d_nm`, `hollow` (has cavity), `truncated`, `touches_exclusion` (v2), `aggregate_id`, `source` (`sam`, `recover`), `rule_id`, `flags`, `in_stats`.

`in_stats = not truncated and not touches_exclusion and not POSSIBLE_CORE_ONLY` (v2). Aggregate members are included and marked, so stats can be recomputed with or without them.

### 10.2 Per panel (`panel.json`)

`nm_per_px`, `cal_source`, `n_particles`, `n_in_stats`, `n_truncated`, `n_touches_exclusion` (v2), `n_hollow`, `n_aggregates`, counts per role, flags, and over `in_stats` particles (only if `n_in_stats ≥ min_particles_for_stats`): mean, SD, median, D10, D90 of `d_nm` (of `d_px` if uncalibrated), computed twice: all particles, and excluding aggregate members.

### 10.3 Other files

`nodes.parquet` (every node: ids, RLEs, features, role, rule_id, parent, flags) and `overlay.png`: particles green, truncated particles green with "T", cavities magenta, gaps blue, specks gray, orphans purple, rejected red dashed, aggregates thick white, exclusion mask red at 30%, calibration text in a corner. Run-level `particles_all.csv` and `panels_all.csv`.

## 11. Tests

Unit tests use toy images drawn with `skimage.draw.disk` (TEM-like: background 0.8, particle 0.3, cavity 0.6) and hand-built masks, so they run without SAM.

| Test | Expected |
|---|---|
| RLE round trip with overlapping masks | Identical masks back |
| Two masks with IoU 0.9 | One representative, one `duplicate` |
| Hollow particle: outer mask, cavity mask, center dot | 1 `particle`, 1 `cavity`, 1 `spot` |
| Three touching disks plus a mask of the gap between them | 3 `particle`, 1 `gap` |
| Disk cut by the border with a pore mask inside | `particle` with `truncated = True`; pore is `cavity` or `spot`, not a particle |
| Hollow particle drawn, only the cavity mask given | `orphan` (subtype `cavity`) |
| Dumbbell mask containing two disk masks | `merged` parent, 2 `particle` |
| Blob containing 5 disk masks | `aggregate` with 5 member particles |
| Image with a drawn 60 px bar and `"200 nm"` (`cv2.putText`) | `nm_per_px` within 2% of 3.33; bar and text inside the exclusion mask |
| Margins of 10 px white | `content_box` within 1 px |

Integration test (needs SAM and GPU): run the full pipeline on the golden panels and check it finishes with the outputs of section 10.

## 12. Evaluation and MVP exit criteria

### 12.1 Golden panels

Chosen from `npsam_panel_labels.csv` to cover the verified failures, plus one control. The first 12 rows are the gating golden panels; rows marked *tracking* are evaluated and reported but do not count toward criterion 6 (12.4).

| Panel | Failure to fix |
|---|---|
| `pylypchuk2021_greenchem_04058d__p05_x46__panel02` | Dark gap masked; right sphere lost (verified) |
| `pylypchuk2021_greenchem_04058d__p05_x46__panel17` | Dark gap masked |
| `pylypchuk2021_greenchem_04058d__p05_x46__panel01` | Edge particle lost, pore kept (verified) |
| `pylypchuk2021_greenchem_04058d__p05_x46__panel04` | Edge particle lost, only pore and scale bar masked |
| `pylypchuk2021_greenchem_04058d__p05_x46__panel19` | Only the pore, outer particle lost |
| `pylypchuk2021_greenchem_04058d__p05_x46__panel11` | Background specks |
| `pylypchuk2021_greenchem_04058d__p06_x50__panel04` | Only masks are scale text; particle at edge |
| `girard2024_acssuschemeng_4c00711__p08_x658__panel06` | Outer + core double count; scale bar as particle |
| `xiong2020_indcrop_112063__p02_x12__panel00` | Half the spheres dropped at the edge |
| `lievonen2016_greenchem_01436k__p04_x48__panel03` | Aggregates split into sub-blobs |
| `wang2019_holzforschung_0062__p08_x77__panel01` | Panel-label box as a particle |
| `pylypchuk2020_acssuschemeng_0c05119__p05_x218__panel02` | Control: NP-SAM's best touching-cluster case; LNP-SAM must not be worse |
| `pylypchuk2021_greenchem_04058d__p05_x46__panel00` | Dense fused aggregate; NP-SAM split the aggregate body into patches. *Tracking* panel for aggregate handling (criterion 7) |

**Deferred (v3, out of MVP scope).** `xiong2020_indcrop_112063__p02_x12__panel02`, `xiong2020_indcrop_112063__p02_x12__panel03` and `xiong2017_acssuschemeng_6b02585__p02_x145__panel04` show transparent, low-contrast TEM particles: the light interiors are particle material, not cavities, so mask-based hollowness reads them wrongly. They return when transparent particles are in scope. `xiong2020…panel00` (opaque spheres) stays.

### 12.2 Ground truth

For each golden panel, annotate two label images (napari; the μSAM annotator speeds this up): `particles.tif` (one label per particle, full outer outline filled, truncated particles as visible) and `cavities.tif` (one label per visible cavity or core). Panels with fused aggregates add an optional third file, `aggregates.tif` (one label per fused aggregate, covering its full outer outline); the individual spheres inside an aggregate still go in `particles.tif` as usual. Only panels with aggregates need this file. Annotate cavities with the same convention as 8.4 (v3 E): a particle's cavity is its single hole near the centre, of any size; when a particle shows two or more holes, or one clearly off-centre, annotate none (they are pores, not hollowness), so that hollow accuracy (12.3) scores what the rule decides. Record the hand-measured `nm_per_px` in `data/gt/calibration.csv`.

### 12.3 Metrics (`eval/evaluate.py`)

| Metric | Definition |
|---|---|
| F1@0.5 | Hungarian matching (`scipy.optimize.linear_sum_assignment` on −IoU) of predicted particles to GT particles; a match needs IoU ≥ 0.5 |
| Count error | `\|n_pred − n_gt\| / n_gt` |
| Median diameter error | Over non-truncated particles on both sides: `\|median(d_pred) − median(d_gt)\| / median(d_gt)` |
| Hollow accuracy | On matched particles: predicted `hollow` vs GT has a cavity |
| Calibration error | `\|nm_per_px_auto − nm_per_px_gt\| / nm_per_px_gt` |
| Aggregate IoU | Only if `aggregates.tif` exists. For each GT aggregate: the highest IoU with any predicted `aggregate` outline (filled mask); 0 if none overlaps |
| Member count error | Only if `aggregates.tif` exists. For each GT aggregate: `\|n_pred − n_gt\| / n_gt`, where `n_pred` = members of the best-matching predicted aggregate (0 if none) and `n_gt` = GT particles with ≥ 50% of their pixels inside the GT aggregate |

NP-SAM baseline (`baselines/npsam_run.py`, npsam environment): `NPSAMImage(path)`, `segment(SAM_model="base", PPS=64)` with all other defaults; every mask in `.seg.data` counts as a predicted particle. Same evaluator.

### 12.4 Exit criteria

The MVP is done when, on the 12 golden panels:

1. No predicted particle lies on text, a scale bar or a label box.
2. No gap between particles is predicted as a particle on the two gap panels.
3. On the three edge-pore panels, the outer particle is present (as `truncated`) and the pore is not a particle.
4. On panels with hollow or cored particles (the edge-pore panels, girard2024), each particle is counted once, with its cavity or core attached.
5. Auto calibration is within 5% of the hand measurement on every golden panel with a readable scale bar.
6. LNP-SAM's F1@0.5 and median diameter error are better than NP-SAM's on at least 9 of the 12 gating panels, and not worse on the control panel. Tracking panels (12.1) are excluded from this count.
7. *(Non-gating.)* On tracking panel `p05_x46__panel00`, the fused cluster is predicted as one aggregate with its members attached. Reported, not required for MVP.

Known MVP limitation: particles SAM never proposes and that leave no orphan cavity (e.g. the lost right sphere in `p05_x46__panel02`) are not recovered. Recovering them from uncovered foreground is the first post-MVP addition.
