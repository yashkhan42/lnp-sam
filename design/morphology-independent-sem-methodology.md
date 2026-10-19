# A morphology-independent methodology for SEM particle analysis

Research proposal, 13 October 2026. Based on inspection of `runs/lab_sem_v1`, source images, current pipeline code, and relevant primary literature. This is a proposed experiment, not an implemented or validated replacement.

## Recommendation

Replace geometry-driven particle identification with **background-aware hierarchical instance segmentation**. Learn whether a region depicts a particle, learn which pixels belong to the same primary particle, and represent visible internal structure and aggregate membership separately. Use morphology to describe the resulting objects, rather than to decide whether those objects exist.

The research hypothesis is:

> Explicit supervision of real SEM background confounders, combined with separate outer-footprint, internal-structure, and instance-relation targets, will reduce background detections at matched particle recall across compact, needle-like, hollow-appearing, and aggregated particles.

This hypothesis is falsifiable. A simpler learned foreground model may perform just as well; if so, use it.

## What the existing run establishes

`lab_sem_v1` is a saved run directory, not a Git branch. The manifest contains 26 source fields. Twenty-five have outputs through measurement; `lab20260602_alkT4__img03` has no saved stage artifacts. The completed fields contain 14,312 raw SAM masks, 15,597 proposal nodes, and 2,367 measured rows, of which 2,105 enter statistics. These are output counts, not accuracy measurements.

Visual inspection of T5 images 07–09 shows bright elongated particles amid faint ridges, wrinkles, and texture. The overlays include elongated detections and broad outlined patches whose appearance is consistent with the user's background-confusion report. A microscopist must adjudicate these regions before they become ground truth.

| Needle-containing field | Raw SAM masks | Strict calibration seeds | Final particle roles | Included in statistics | Median seed circularity |
| --- | ---: | ---: | ---: | ---: | ---: |
| T5 image 07 | 432 | 26 | 81 | 72 | 0.911 |
| T5 image 08 | 493 | 68 | 114 | 110 | 0.862 |
| T5 image 09 | 553 | 51 | 86 | 82 | 0.932 |

All completed fields used strict seeds. **The no-seed fallback did not occur.** The issue is the reliance on round-selected proposals to calibrate intensity and size in fields containing diverse shapes. Some calibration seeds subsequently become gaps, background patches, or rejected masks. In image 09, 15 of 51 seeds finish as gaps or rejected masks. This demonstrates dependence on proposals whose eventual role is uncertain; it does not by itself prove they are visual noise.

T5 image 07 also has an `R-LARGE` aggregate covering 772,138 pixels, about 98.2% of its 1024 × 768 frame, with zero members. The overlay shows a near-frame outline. It is excluded from size statistics, but the aggregate interpretation is poorly supported.

Saved proposal metadata match the current SAM parameter digest and identify SAM ViT-B on CPU. They do not archive resolver parameters or a source commit (`git_commit=null`), so the current resolver explains a plausible mechanism rather than proving exact historical execution for every rule.

The current code has four relevant structural assumptions:

1. `lnpsam/resolve.py:284` selects calibration seeds using solidity and circularity. `:477` accepts ordinary roots using relative brightness, edge contrast, and mask quality. Ordinary particle acceptance already permits elongated shapes.
2. `lnpsam/proposals.py:290` combines predicted IoU, stability, and edge contrast into quality. None is a learned probability that the region is a particle rather than substrate texture.
3. `lnpsam/proposals.py:356` computes filled footprints and `:378` deduplicates by those footprints. Material/appearance holes therefore cannot distinguish otherwise identical candidates. `lnpsam/resolve.py:415` scales rim and surrounding-ring widths by area-equivalent diameter, which is a poor local width for a long needle.
4. `lnpsam/resolve.py:591` uses relative intensity and minimum size to select aggregate members, bypassing ordinary root edge/quality acceptance. The internal-hole logic at `:697` also encodes a restricted single central-hole interpretation.

Tightening circularity would sacrifice needle recall. Adding another brightness threshold would still leave the model without examples of what SEM background looks like.

## Define the target before choosing the model

Particle detection, particle shape, visible internal structure, and aggregation are different questions. A needle can have an internal depression and be part of an aggregate. These should not be four mutually exclusive detection classes.

For each annotated object, retain:

| Target | Meaning | Important restriction |
| --- | --- | --- |
| Outer footprint `E` | The observed projected outer outline, including an enclosed dark interior | Preserve arbitrary concavity; do not infer a disk or convex hull |
| Primary instance ID `Z` | Which primary particle owns each footprint pixel | Mark ambiguous contacts as unknown rather than inventing boundaries |
| Internal appearance `H` | Visible enclosed depression/void-like region inside a footprint | A dark SEM interior alone does not establish physical hollowness or composition |
| Aggregate relation `G` | Whether confidently resolved primary particles belong to one visible group | Distinct from belonging to the same primary instance |
| Uncertainty `U` | Unclear identity, boundary, interior, or contact | Exclude unknown labels from the relevant training loss |

The footprint and internal-appearance mask remain separate throughout analysis. Their difference can describe the visible body/rim, but should not be called a physical material/composition map from intensity alone. No universal hole count, centrality, or circularity is imposed.

For an aggregate with clear constituent boundaries, report the group and its members. For a fused or occluded region whose members cannot be distinguished, annotate the particle-containing foreground footprint while marking primary IDs and contact boundaries unknown. The foreground path can then retain an unresolved particle-containing region with primary count unknown; pairwise group relations alone cannot create a region with no identifiable members. Single-view segmentation cannot recover hidden boundaries reliably without additional evidence.

## Proposed model and inference

Use one shared, high-resolution segmentation model with local detail and a wider contextual field. A U-Net-style decoder is a practical first implementation; a microscopy-adapted SAM encoder can be compared later. Keep native-resolution overlapping tiles so narrow particles are not lost through global resizing. Include native pixel scale and acquisition metadata where available, while retaining an uncalibrated path.

The model learns:

- **Footprint probability:** evidence that a pixel belongs to a particle's outer footprint, learned jointly from particles and verified background.
- **Same-primary-instance affinities:** probabilities that pairs of pixels belong to the same particle. Short-range edges resolve contacts; a limited set of longer-range edges helps maintain continuity along extended objects.
- **Internal-appearance probability:** the optional interior target within each footprint.
- **Object identity confidence:** confidence that a candidate corresponds to a real particle, trained using true instances and false/background candidates. Keep this distinct from predicted mask IoU.
- **Group relations:** a later, object-level head predicting same-aggregate membership between already accepted primary instances.

Learn foreground and affinity losses first. Add internal-appearance and group supervision only where labels are reliable. Balanced background/foreground sampling and explicit hard-negative mining matter more initially than adding a complex topology loss.

For pixel pairs `x,y`, let `a(x,y)` estimate the probability of belonging to the same primary instance. Partition the footprint graph using attractive and repulsive edges, for example with Mutex Watershed. This uses the same reconstruction mechanism for compact, elongated, annular, and irregular outlines. It does not require a center, a star-convex shape, or a fixed number of members. Affinity errors can still cause merges and fragments; the representation does not guarantee success.

Do not constrain all objects to have one connected component or no holes. When a topology auxiliary loss is tested, compare predictions with the annotated topology, and mask uncertain/occluded regions.

After partitioning, apply calibrated object identity confidence to **every primary instance**, including proposed aggregate members. A group relation must never turn a background candidate into a valid particle. Derive shape descriptors and measurements only after acceptance.

## The background mechanism

The central intervention is to make the model learn real negative evidence. Include substrate wrinkles, elongated ridges, scratches, charging patterns, diffuse patches, and noise regions that the current pipeline actually proposes as objects. Treat scientifically ambiguous debris as unknown until the target definition is agreed.

Use expert-verified blank/background crops; pixels outside SAM proposals are not automatically reliable negatives. Mine hard negatives from the training split only, and adjudicate them before adding them to training. Random Gaussian noise alone is insufficient to represent these confounders.

Train with rotations, modest contrast changes, blur and noise compatible with the acquisition, and realistic variation of background context. Object/background recombination is an optional augmentation: preserve the object and immediate boundary neighborhood, avoid unrealistic seams, and compare against ordinary augmentation. It is established prior art and not the primary novelty claim.

Use held-out labeled foreground and background crops to choose the object confidence threshold. Report empirical false particles per annotated background megapixel and precision/recall inside annotated regions. Full-field precision/recall requires additional complete-field annotation. A small validation set cannot establish formal false-discovery guarantees. New imaging conditions need an independent validation check; confidence alone is not evidence of successful transfer.

At inference, inverse-mapped predictions under flips/rotations and overlapping tiles can identify instability. Stable texture can still be false, and weak real needles can be unstable, so consistency should support an uncertainty estimate rather than act as a universal rejection rule. Uncertain objects or contacts enter review.

## Why one representation can cover the morphologies

| Appearance | How it is represented |
| --- | --- |
| Compact particle | Learned footprint with coherent same-instance affinities |
| Needle | The same footprint/affinity model, with native-resolution detail and no circularity acceptance requirement |
| Hollow-appearing particle | One outer-footprint instance plus a separate internal-appearance mask |
| Aggregate | Accepted primary instances plus a separate group relation; unresolved contacts remain unknown |
| Background ridge or noisy patch | Negative semantic/object-identity evidence learned from actual substrate examples |

Length, width, Feret diameter, projected area, equivalent-area diameter, and internal-aperture area can then be reported as descriptors. Equivalent-area diameter is useful as one summary statistic but should not substitute for needle length/width or be interpreted as spherical diameter.

## Fixed initial annotation budget

There is no justified exact minimum before a learning curve. Use a bounded pilot so the annotation cost is an explicit decision:

1. **Twenty particle-containing crops**, approximately 256–512 pixels per side: 12 training, 4 validation, and 4 test. Select crops spanning compact, elongated, hollow-appearing, and aggregate regions. Aim for approximately 200–400 reviewed primary instances, with a hard initial cap of 400 corrected masks. Choose crop extents accordingly; unannotated objects must remain unknown rather than become negative examples. Include variation in size, contrast, blur, and background.
2. **Eight additional background-only crops**, divided 4/2/2 across those splits. These require confirmation of background identity and optional exclusion strokes, rather than individual contour tracing. Include ordinary texture and hard confounders.
3. Correct SAM-assisted outer masks, assign primary IDs, and mark unclear contacts. Mark internal depressions and group membership only where clear; reuse the outer masks for their derived training targets. No skeletons or per-pixel affinities need manual annotation.
4. Time an initial 20 mask corrections before estimating hours. Dense aggregates and weak needles can make a universal time estimate misleading.

Split by **source image**, never by random crops or random particles from the same image. Near-duplicate fields and overlapping fields of view must stay in one split. Aim to cover each appearance in every split; if the 20-crop budget cannot do this, change the crop selection or report the pilot as inconclusive for that appearance. Any expansion of the annotation budget is a separate decision.

Train on nested subsets of 4, 8, and 12 training crops with fixed validation/test fields. Inspect the learning curve and errors before labeling more. If the test set has too few instances of a morphology, report it as inconclusive. For broader claims, acquire additional specimens/sessions and hold out whole acquisition groups. The current 26-field, three-specimen-group dataset cannot establish universal transfer.

## Experiment sequence and decision gates

**First measure proposal recall.** For every annotated real instance, record the best match among all saved SAM candidates before resolver rejection. Examine both overlap and boundary quality, especially for thin objects. Distinguish candidate misses, poor masks, rejection of good masks, and duplicate/merge errors.

- If proposals already cover the relevant objects sufficiently for the intended analysis, start with a small learned particle/background verifier using image context and mask features. This is the cheapest replacement for the rule table and uses the same label set. Its recall remains bounded by the proposer.
- If real needles are missing or consistently fragmented before resolution, use the dense foreground/affinity path. A verifier cannot recover objects that were never proposed.

Compare these baselines under the same image split and annotation budget:

1. Current `lab_sem_v1` outputs and a reproducible current-code rerun with an archived config.
2. Existing SAM proposals plus a learned identity verifier.
3. Ordinary high-resolution foreground/boundary or affinity segmentation trained on the same labels.
4. A microscopy-adapted SAM model such as μSAM, evaluated/adapted on the same SEM task.
5. The proposed footprint/internal-appearance representation and, subsequently, group relations.

Measure object precision/recall and false positives per background area; boundary accuracy; morphology-stratified recall; needle fragmentation; touching-particle merge/split errors; internal-appearance accuracy; group/member accuracy; and length/width/area measurement error. In this crop-budget pilot, score only fully annotated held-out regions, even when inference runs on the full source image. Mark objects crossing ROI boundaries as truncated and evaluate them separately; do not count predictions outside labeled regions as false positives. Complete-field annotation is required for later full-field accuracy claims. Report uncertainty with resampling by independent image/group, not by individual particle. Report review coverage and accuracy of automatically accepted objects together so abstention cannot hide failures.

Tune on validation only. The primary comparison is background false detections **at matched recall**, including needle recall. A preregistered pilot goal could be at least a 50% reduction in background detections without deterioration of morphology-specific recall; this is an experimental target, not a predicted result. Tiny morphology subsets may make that conclusion inconclusive.

Ablate real hard negatives/context, the separate internal target, long-range affinities, and aggregate relations separately. Require the full design to beat the ordinary learned baseline before claiming that the extra representation is necessary. Keep the test set untouched until the final pilot comparison; after using its errors to guide development, collect a new final test set.

## Prior art and candidate originality

The following establish useful components rather than proving this specific SEM solution:

- [Omnipose (2022)](https://www.nature.com/articles/s41592-022-01639-4) studies extended and complex cell morphologies and documents fragmentation associated with center-directed reconstruction in the systems it tested. It motivates avoiding a mandatory particle center; it does not validate this SEM model.
- [Mutex Watershed (ECCV 2018)](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper.php) establishes seed-free segmentation from attractive/repulsive affinities. Affinity grouping is not a new invention here.
- [Segment Anything for Microscopy / μSAM (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11903314/) adapts SAM to microscopy and learns foreground plus center/boundary-distance outputs. It is a serious baseline; microscopy transfer does not automatically establish nanoparticle/background discrimination.
- [SAM + DINO nanoparticle shape classification (2025 preprint)](https://arxiv.org/abs/2508.03235) already combines foundation-model segmentation and embeddings for nanoparticle morphology. That combination alone cannot be claimed as original.
- [EMcopilot (2025)](https://www.nature.com/articles/s41524-025-01800-5) learns morphological and contrast heterogeneity to synthesize EM segmentation training data. Generative EM augmentation is already established.
- [Counterfactual learning and saliency augmentation (2025)](https://doi.org/10.1016/j.imavis.2025.105523) and [Copy-Paste (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html) establish approaches to object/background recombination and background confounding.

A possible original contribution is the **SEM-specific joint target and decision formulation**: observed outer footprint, internal appearance, primary-instance identity, aggregate relations, and background-calibrated acceptance, evaluated across these morphologies. That is currently a candidate contribution. A broader literature review and successful comparisons/ablations are required before calling it novel or unique.

The first actionable step is a small annotation and proposal-recall audit, followed by the learned-verifier baseline. This tests the central diagnosis with limited cost and determines whether a larger dense segmentation model is warranted.
