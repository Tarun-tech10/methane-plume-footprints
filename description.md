# Confirmed Methane Plume Footprints — solution

`solution.py` reads the supplied scenes, trains a segmentation network from scratch
under 5-fold cross-validation, calibrates its two decision knobs on held-out
predictions, and writes `submission.csv`.

```
python3 solution.py <public_dir> <output_csv>
```

Nothing outside the provided dataset is read, loaded or consulted. All weights are
initialised randomly and fitted on the supplied training scenes only. No scene is
matched, geolocated or dated, and every footprint emitted is the thresholded output
of the network — no value is entered by hand.

---

## 1. What the score actually pays for

Re-deriving the metric drives every design choice below, so it is worth being exact.
With `p` the share of scenes that hold a confirmed plume and `D` the mean Dice over
those scenes, accepting a scene that truly holds a plume is worth

```
0.5/N  (decision)  +  0.5·D/(p·N)  (delineation)
```

while correctly rejecting an unconfirmed scene is worth only `0.5/N`. At `p ≈ 0.565`
a missed plume therefore costs about **twice** what a false footprint costs, because
rejecting a real plume forfeits its Dice as well as its decision point. The optimum
is to accept somewhat more scenes than actually hold plumes — the calibration step
finds `0.590`, against a true positive rate of `0.565`.

The same arithmetic reproduces the brief's own reference point: a network with 95 %
recall, a Dice of 0.56 and a false-accept rate of 0.44 on unconfirmed scenes scores
**0.578**, matching the quoted 0.5761. That agreement is the check that the metric
used for all tuning below is the metric being graded.

## 2. What the data turned out to look like

Four measurements shaped the solution.

**Every confirmed footprint covers the scene centre.** Across all 818 positive
training scenes, the greatest distance from the image centre to the nearest mask
pixel is **2.55 px**; every mask intersects a 6 × 6 central box, and 60 % contain the
exact centre pixel. The scenes are crops centred on the flagged enhancement. The
network is given a radius channel so it can use this, and every augmentation used
keeps the centre a fixed point.

**Footprints are small and often broken up.** Median area is 2222 px (0.49 % of the
grid). 64.3 % are a single connected component; the rest run to six, and the pieces
that are *not* joined to the centre carry real mask area — restricting the prediction
to components touching the centre collapses Dice from 0.612 to **0.474**. The
multi-component structure is signal, not noise, and component pruning was dropped.

**Missing retrieval is itself a strong negative cue.** Median missing fraction is
2.2 % on confirmed scenes against 10.0 % on unconfirmed ones (AUC 0.665 on its own).
Reviewers reject enhancements sitting over cloud and water, and the validity mask is
fed to the network as a channel and to the decision stage as a feature.

**The enhancements are faint.** Mean robust z inside a true footprint is 2.70 — a
few-sigma feature on a strongly striped background, not a bright object.

## 3. Resolution: the measurement that bought the compute

The obvious design predicts a mask at the full 672 × 672. Downsampling a *true* mask
to 168 px and resampling it back costs almost nothing:

| mask resolution | Dice ceiling |
|---|---|
| 336 | 0.9925 |
| 224 | 0.9856 |
| **168** | **0.9721** |
| 112 | 0.9477 |

Against an achievable Dice near 0.62, a 0.972 ceiling is not the binding constraint.
So the network emits logits at 168 px and only the two stem convolutions ever touch
full resolution. That is what makes a 5-model fold ensemble fit the budget rather
than a single model.

(One incidental but large find: `channels_last` memory format was **5.4× slower**
than contiguous for this network — 9.8 img/s against 53 img/s. It is not used.)

## 4. Method

**Input — 5 channels, all derived per scene from the one global affine map.**
`ch0` absolute concentration; `ch1` concentration standardised by the scene's own
median and MAD; `ch2` `ch1` minus a coarse valid-weighted local background, i.e. the
enhancement itself; `ch3` where the retrieval succeeded; `ch4` distance from the
scene centre. Missing pixels are excluded from every statistic and zeroed, never
treated as a concentration.

**Network.** A residual U-Net (4.8 M parameters) with two heads: a per-pixel
segmentation head at 168 px and a scene-level head off the pooled bottleneck. The
segmentation head is trained on *all* scenes, unconfirmed ones carrying an all-zero
target — that is what teaches it the difference between an enhancement that survived
review and one that did not. Loss is pos-weighted BCE plus soft Dice over positive
scenes, plus BCE on the scene head.

**Augmentation.** The eight dihedral transforms (plume orientation is arbitrary and
the centre is preserved), contrast jitter on the enhancement channels, and a level
jitter on `ch0` so that no absolute concentration offset can be memorised as a
shortcut. No translation, which would destroy the centring prior.

**Inference.** 5 fold models averaged, each with 4-fold dihedral test-time
augmentation.

**Decision stage.** 21 features are read off each predicted probability map — peak
and top-k means, mass, central-box probability, thresholded areas, component count,
spread, compactness — plus the scene head's own output, the valid fraction and the
central enhancement level. These feed a small logistic regression, expanded with
train-referenced rank and squared-rank terms, fitted on out-of-fold predictions. The
expansion is what lets a linear model match a gradient-boosted one (0.7068 vs 0.7063)
with **no dependency beyond numpy**, which is why no boosting library is used.

**Both knobs are calibrated on held-out data, never assumed:** the mask threshold is
swept over held-out positives, and the accept fraction is swept directly against the
challenge score.

## 5. Why the accept rule is a quantile and not a probability

The decision rule takes the top `q` fraction of test scenes by score rather than
applying a fixed probability cut. The out-of-fold scores come from *single* models
while test scores come from a *5-model ensemble*, whose probabilities are
systematically smoother. A quantile rule is invariant to any monotone shift between
those two distributions — only the ranking has to transfer, and ensembling improves
ranking. A fixed probability threshold would inherit the miscalibration directly.
The brief's statement that both splits hold confirmed and unconfirmed scenes in
similar proportion is what licenses this.

The score is also flat near the optimum (0.704 at q = 0.60, 0.703 at 0.62, 0.698 at
0.58), so the choice is not delicately tuned.

## 6. Results

Cross-validated on the 1447 training scenes (single model per fold, 20 epochs):

| | value |
|---|---|
| scene classification AUC | 0.9752 |
| decision accuracy | 0.9330 |
| recall on confirmed scenes | 0.9670 |
| false accepts on unconfirmed scenes | **0.1113** |
| Dice over confirmed scenes | 0.6177 |
| **challenge score** | **0.7078** |

Reference points from the brief: all-no-plume **0.0000**, from-scratch convolutional
segmentation network **0.5761**.

Most of the margin is the third row. The reference network "still returns a footprint
about half the time" on unconfirmed scenes; this one does so 11 % of the time, at
slightly *higher* recall. Delineation contributes the rest (0.63 Dice on accepted
scenes against 0.56).

That table is the 20-epoch, 4-fold development setting. The shipped configuration
trains 40 epochs over 5 folds; on fold 0 the longer schedule is clearly better:

| | AUC | Dice |
|---|---|---|
| 20 epochs | 0.9708 | 0.6122 |
| 40 epochs + level jitter | **0.9785** | **0.6251** |

and the shipped run itself reports, on its own held-out predictions:

```
mask threshold 0.55  ->  held-out Dice 0.6272
accept fraction 0.590 ->  held-out challenge score 0.7101
```

The threshold sweep is a plateau — 0.6266 / 0.6272 / 0.6272 / 0.6259 at 0.50 / 0.55 /
0.60 / 0.65 — so the selected value sits in the middle of a flat region rather than on
a spike. The submission marks 224 of 380 test scenes (58.9 %) as holding a plume,
against a 56.5 % positive rate in training.

## 7. Things that were tried and did not work

Each was implemented and scored, not argued away.

- **Per-scene adaptive mask threshold.** A per-image oracle threshold would give Dice
  0.693 against 0.612 for a global one, so the headroom is real — but the model's peak
  probability saturates at ~1.0 on essentially every positive scene, and the oracle
  threshold's correlation with anything observable is weak (0.26 with predicted mass,
  0.18 with peak). Every parametric adaptive rule tried landed within 0.0003 of the
  fixed threshold. Dropped.
- **A segmentation specialist trained only on confirmed scenes**, on the theory that
  negatives distract the mask head. Dice 0.6121 against 0.6122 for the multi-task
  model — identical, at double the training cost. Dropped.
- **Component pruning** (centre-connected only 0.474, drop-small 0.6117, centre-or-big
  0.6102, none 0.6116). No variant beat leaving the mask alone.
- **Area matching** the mask to the predicted probability mass: 0.610 at best, below a
  plain threshold.
- **Gradient-boosted decision stage**: 0.7063 against 0.7068 for the expanded linear
  model — not worth a hard dependency on a library that may not be installed.

## 8. Honest limitation

The dataset is split so that no flight day appears on both sides, but flight-day
labels are not provided, and the brief forbids working out which flight a scene
belongs to. Cross-validation here is therefore stratified-random, and scenes from one
flight day can fall on both sides of a fold. Random folds are optimistic relative to
a day-disjoint test split, so **0.7078 should be read as an upper estimate, not a
prediction of the test score.** Two things limit the exposure: the model's
scene-level inputs are standardised per scene, and `ch0` — the only channel carrying
absolute level — is level-jittered during training and has no marginal association
with the label on its own (AUC 0.496). The ranking-based accept rule is likewise
robust to a shift in score scale between the two splits. Even so, the gap between
random-fold and day-disjoint generalisation is real and is not measurable from the
data as supplied.

## 9. Determinism

The execution plan is fixed, not discovered at runtime. Concretely:

- **The device is a constant** (`DEV = 'cuda'`), not the result of a capability
  probe, so there is no second code path to fall back to.
- **The cuDNN autotuner is off.** `benchmark = True` selects convolution kernels by
  *timing candidates on the spot*, which makes the arithmetic depend on machine load
  — two runs can pick different algorithms and produce different numbers.
  `benchmark = False`, `deterministic = True`.
- **No implicit reduced-precision matmul.** `allow_tf32` is set explicitly on both
  backends rather than left at a version-dependent default. Mixed precision is
  unconditional, not switched on a device test.
- **`torch.use_deterministic_algorithms(True)`**, with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before the torch import so cuBLAS
  reductions are reproducible.
- **The one op that could not comply was rewritten.** The decoder's 2x nearest
  upsample accumulates its backward pass with atomics, which do not reproduce bit for
  bit. Every decoder stage is exactly 2x (21 -> 42 -> 84 -> 168), so it is written as
  `expand` + `reshape` instead (`up2`), verified identical to `F.interpolate` in the
  forward direction and deterministic in the backward.
- **No worker pool.** Scenes are read in a plain sequential loop, so no result
  depends on thread scheduling.
- **Every RNG is seeded** — numpy, torch CPU and torch CUDA — and reseeded per fold
  from the fold index, so fold `f` draws the same augmentations regardless of which
  folds ran before it.
- **No wall-clock anywhere.** `time` is not imported. Epochs, folds, batch size and
  every threshold grid are fixed constants; no branch, loop bound or early exit
  depends on elapsed time or on measured throughput.

Verified rather than asserted: the pipeline was run twice end to end from a clean
process on the same configuration, and the two submissions are byte-identical —

```
run A sha256 4cdbc77f21b04d349bff28407719f1feb81d5833507ae527f8c0252c6076fbd0
run B sha256 4cdbc77f21b04d349bff28407719f1feb81d5833507ae527f8c0252c6076fbd0
```

Turning the autotuner off and forcing deterministic kernels costs **4.6 %** of
throughput (158 ms/step against 151 ms/step at batch 8), which is why nothing had to
be traded away to get reproducibility.

## 10. Runtime

Single GPU, offline. The five folds dominate; scene loading and feature extraction
over all 1827 scenes take well under a minute between them, and inference is 5 models
x 4 augmentations over 380 test scenes.

Timings were taken on a laptop RTX 3050 6 GB benchmarked at 9.3 TFLOPS fp16 and
168 GB/s. A full 5-fold, 40-epoch pass over the pipeline ran in 96 min there
(~29 s/epoch), and the determinism settings add 4.6 % on top of that in an isolated
step benchmark. The target device has on the order of 7x that card's tensor
throughput and 3.5x its memory bandwidth, so the same work should land well inside
the budget there, and would still fit at a fraction of the expected speedup.

## 11. Files

- `solution.py` — the whole pipeline, single file, run as above.
- `submission.csv` — output of that script on the supplied test split.
- `description.md` — this document.
