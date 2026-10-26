# Project status

**Last updated:** 2026-08-24
**State:** both phases built, trained and evaluated; leakage audit complete; a
leakage-free split has been rebuilt. **Retraining on the clean split is the next
action and has not been started.**

---

## TL;DR

A working two-phase brain-tumour MRI classifier exists end to end, with
per-class metrics, confusion matrices, Grad-CAM and post-hoc calibration. Then a
perceptual-hash audit found that **33.6% of the supplied test set duplicates
training data**, so the headline scores were inflated. A clean split has been
built and verified; the models have **not** yet been retrained on it.

Treat the leakage-adjusted numbers as the real ones.

| | baseline CNN | transfer (ResNet18) |
|---|---|---|
| test macro-F1 (as reported) | 0.7954 | 0.9367 |
| test macro-F1 (leakage-clean subset) | **0.7561** | **0.8857** |
| test accuracy (leakage-clean subset) | 0.7859 | 0.9114 |

---

## Done

### Phase 1 — baseline CNN
422,308 params, 4× `Conv→BN→ReLU→MaxPool` + 2 dense. AdamW lr=1e-3, batch 32,
20 epochs, best at epoch 16 (val loss 0.3706). Test accuracy 0.8043, macro-F1
0.7954. Checkpoint `outputs/checkpoints/baseline_best.pt`.

Validation loss oscillated hard (spikes to 3.58) — BatchNorm statistics at a
constant lr with no scheduler. Instability, not overfitting. Left as-is
deliberately: section 5's goal is an honest benchmark, not peak accuracy.

### Phase 2 — transfer learning
ResNet18 / ImageNet, `fc` replaced. 4 epochs frozen head (lr=1e-3), then
`layer3`+`layer4` unfrozen (lr=1e-4). Early stopped at epoch 12, best at epoch 7
(val loss 0.0732). Test accuracy 0.9388, macro-F1 0.9367. Checkpoint
`outputs/checkpoints/transfer_best.pt`.

Beats the baseline on every class. Largest gain is meningioma (+0.253 F1, recall
0.598 → 0.967) — the baseline's worst class, as section 6 predicted.

### Phase 3 — Grad-CAM
Both models rendered over the same 15 scans (8 correct, 7 incorrect, all four
classes). **The most informative result in the project:** the transfer model
attends to brain tissue and often the lesion itself, while the baseline
frequently peaks on **background corners and skull rim** while predicting
correctly at p=1.00. Part of the baseline's 80% is earned from framing
artefacts, not anatomy — something no confusion matrix would have revealed.

### Calibration (post-hoc, no retraining)
`T = 1.3241`, glioma logit offset `δ = +2.10`, both fit on the leakage-clean
validation subset only. On the leakage-clean holdout:

- confidently-wrong predictions cut 75% (errors at p>0.99: **20 → 5**)
- NLL −39% (0.588 → 0.361), ECE −41% (0.0609 → 0.0362)
- glioma recall **0.772 → 0.834** for 4.9 points of precision

Params in `outputs/calibration_transfer.json`. Confidence is now usable as a
triage signal; before, it was not.

### Data audits
1. **Exact duplicates** — 7,200 files, 7,013 unique. 187 byte-identical copies
   padding classes to a flat 1400/400. Dropped before splitting.
2. **Near-duplicates (perceptual)** — the finding that changed the project.
   `Training/` and `Testing/` share scans, byte-different but perceptually
   identical, so the leakage is **in the source dataset**, not the split logic:

   | split pair | pairs | at distance 0 | images affected |
   |---|---|---|---|
   | train ↔ val | 526 | 113 | 26.0% of val |
   | train ↔ test | 1378 | 380 | **33.6% of test** |
   | val ↔ test | 221 | 66 | 11.7% of test |

   100% of pairs share a class label (chance would be ~25%). Worst class:
   `notumor`, **86.8%** of its test images duplicated training data — its
   near-perfect recall was largely memorisation.

3. **Label conflicts** — 5 near-duplicate groups carry different diagnoses. Two
   are unambiguous, at Hamming distance 0 on both hashes:
   - `Te-gl_68.jpg` (glioma) ≡ `Te-me_50/61/96` + 3 training meningiomas
   - `Te-gl_160.jpg` (glioma) ≡ `Te-me_159`, `Tr-me_1122` (meningioma)

   Both are in the **test set**, so they impose a small hard ceiling on glioma
   precision. Labels untouched — needs a radiologist, not a heuristic.

### Rebuilt split — `data/processed_clean/`
Strategy: protect the holdout, pay in training data. Any near-duplicate group
touching a `Testing/` image had its training-side members quarantined. Groups
are transitive (union-find) and assigned to train/val as whole units.

966 training images dropped (notumor 674, meningioma 161, pituitary 71,
glioma 60).

| split | glioma | meningioma | notumor | pituitary | total | skew |
|---|---|---|---|---|---|---|
| train | 1139 | 1041 | **516** | 1097 | 3793 | 54.7% |
| val | 201 | 184 | **91** | 194 | 670 | 54.7% |
| test | 386 | 398 | 400 | 400 | 1584 | 3.5% |

**Verified: 0 cross-split near-duplicate pairs** at radius 5 (was 526/1378/221).

---

## Next action — retrain (NOT started)

```bash
MRI_PROCESSED_DIR=data/processed_clean python src/train.py --model baseline --class-weights
MRI_PROCESSED_DIR=data/processed_clean python src/train.py --model transfer --class-weights
```

`--class-weights` is now **required**, not optional: dropping the duplicated
`notumor` images pushed skew from 8.5% to **54.7%**, past the spec's 15–20%
threshold. Weights are in `data/processed_clean/class_weights.json` (notumor
1.84). `dataset.py` reads `MRI_PROCESSED_DIR`, so nothing else changes.

Estimated **3.5–4h** on this CPU-only box (~1.8h baseline, ~1.5–2.1h transfer,
~0.3h eval/Grad-CAM/recalibration). One caveat from experience: a single
fine-tune epoch once took 72 minutes instead of 10 when another job competed for
CPU and memory hit 82%.

Expect scores to **drop** — roughly 30% less training data, and the clean-subset
estimate was already 0.886 macro-F1. The point is that the new numbers will be
true.

Also worth doing when checkpoints are regenerated: save to new filenames so the
current models survive for a side-by-side comparison.

---

## Known limitations

- **No GPU.** CPU-only (i7-10510U, 4 threads). The spec assumed Colab. Runs
  ~20× slower; nothing else changes.
- **Patient-level leakage is unknown and unknowable here.** The dataset ships no
  patient IDs, so slices from the same patient may still straddle splits. The
  perceptual audit narrows this risk; it does not eliminate it.
- **Residual near-duplicate leakage.** Grouping is at radius 5. At radius 8
  there are still ~800 train↔test pairs, but by then the threshold merges
  genuinely distinct scans (1st-percentile distance is 11), so 5 is the
  defensible line.
- **A retracted finding, recorded so it isn't rediscovered.** The dataset holds
  203 explicitly augmented meningioma images (`*-aug-me_*`), and neither hash is
  rotation-invariant, so an initial test suggested **170 extra leaked images via
  mirroring**. A control on the three classes with *no* augmented files killed
  that: mirror-matching fired on 121/400 glioma and 312/400 pituitary, where
  augmentation leakage is impossible — brains are bilaterally symmetric, so
  mirror matches are mostly an artefact. **That 170 figure was wrong.** What
  survived: under rot180 the controls give 0/400 while meningioma gives 9, so
  rot180 grouping is applied **to meningioma only**. Applied pool-wide it
  cascaded through union-find and cut notumor training data to 376 to fix ~19
  real leaks.
- **`notumor` metrics should not be trusted** until the retrain lands.
- **Not a diagnostic device.** Glioma recall is 0.834 post-calibration — roughly
  1 in 6 missed, and some misses land on `notumor`.
- **No single-image inference entry point yet.** Everything runs over a split
  directory via `MRIDataset`. A frontend needs
  `predict_one(image_bytes) -> {probs, label, cam}` wrapping the four
  preprocessing steps plus calibration (~60 lines, not written).

---

## Not in the repo

`.gitignore` excludes the dataset (`archive/` 178 MB, `data/processed*` 232 MB)
and model checkpoints (`outputs/checkpoints/`, 45 MB — a 43 MB binary in git
history is permanent bloat). Regenerate with:

```bash
python src/build_split.py         # data/processed      from archive/
python src/build_split_clean.py   # data/processed_clean from data/processed
python src/train.py --model baseline   # checkpoints
```

Figures, `metrics.csv`, training histories and audit reports **are** committed —
they are the deliverables. If the checkpoints should ship, Git LFS or a GitHub
Release attachment is the right mechanism.

---

## File map

| path | what |
|---|---|
| `src/data_audit.py` | exact-duplicate + class-balance audit |
| `src/build_split.py` | exact-dedup, stratified 85/15, normalise to 224 RGB |
| `src/leakage_check.py` | perceptual near-duplicate audit (dHash + pHash) |
| `src/leakage_impact.py` | scores clean vs contaminated holdout subsets |
| `src/build_split_clean.py` | group-aware leakage-free rebuild |
| `src/dataset.py` | Dataset, transforms, loaders (`MRI_PROCESSED_DIR`) |
| `src/model_baseline.py` | Phase 1 from-scratch CNN |
| `src/model_transfer.py` | Phase 2 ResNet18 + head, staged unfreezing |
| `src/train.py` | training loop, checkpointing, early stopping |
| `src/evaluate.py` | per-class metrics, confusion matrices, curves |
| `src/gradcam.py` | Phase 3 interpretability |
| `src/calibrate.py` | temperature scaling + glioma threshold tuning |
