# Brain Tumor MRI Classification

Four-class brain MRI classifier — `glioma`, `meningioma`, `pituitary`, `notumor` —
built in two phases: a from-scratch baseline CNN, then an ImageNet-pretrained
ResNet18 fine-tuned to beat it. Predictions are explained with Grad-CAM.

Built to [`brain-tumor-mri-classifier-baseline.md`](brain-tumor-mri-classifier-baseline.md).

---

## 1. Setup

```bash
pip install -r requirements.txt

python src/data_audit.py     # audit raw archive: counts, duplicates, image props
python src/build_split.py    # -> data/processed/{train,val,test}

python src/train.py --model baseline      # Phase 1
python src/train.py --model transfer      # Phase 2

python src/evaluate.py --model baseline --split test
python src/evaluate.py --model transfer --split test
python src/evaluate.py --compare --split test

python src/gradcam.py --model transfer --split test --compare   # Phase 3
```

**Environment note:** this project was trained **CPU-only** (Intel i7-10510U,
4 threads, no CUDA device) rather than on the GPU the spec assumes. Everything
runs unchanged on a GPU and will simply be ~20x faster. Measured CPU cost:
6.7 min/epoch for the baseline, 3.9–5.4 min/epoch for the transfer model.

---

## 2. Dataset

Kaggle "Brain Tumor MRI Dataset" (Masoud Nickparvar), supplied locally in
[`archive/`](archive/) rather than pulled through the Kaggle API. Pre-split into
`Training/` and `Testing/`, one subfolder per class.

### The raw archive is not the stock dataset — check your assumptions

Raw counts are perfectly uniform (1400/class train, 400/class test), which the
real Nickparvar dataset is not (it has 7,023 images with uneven class counts).
Hashing every file explains why:

**7,200 files, 7,013 unique — 187 byte-identical duplicates**, used to pad each
class to a flat count. Of the 153 exact-duplicate groups:

| | count | consequence |
|---|---|---|
| spanning `Training`/`Testing` | 0 | *(but see the perceptual audit below)* |
| spanning two classes | **0** | no label conflicts |
| inside one split *and* class | 153 | would leak across a random train/val split |

Splitting the raw 7,200 at random would have put ~170 duplicated training images
on both sides of the train/val boundary and silently inflated every validation
number. [`src/build_split.py`](src/build_split.py) drops the redundant copies
first, so no *byte-identical* image can appear in two splits.

### Exact hashing was not enough — the holdout IS contaminated

Byte-identical matching misses a re-saved JPEG or a marginally different crop.
[`src/leakage_check.py`](src/leakage_check.py) re-audits with two independent
64-bit perceptual hashes (dHash on horizontal gradients, pHash on low-frequency
DCT coefficients), counting only pairs **both** hashes agree are near-identical.
The implementation self-tests first: re-encoding an image at JPEG quality 55
moves it ~0.7 bits, while unrelated images sit 24–29 bits apart, so the signal is
real and not an artefact of a broken hash.

Result at Hamming radius 5:

| split pair | pairs | of which distance 0 | images affected |
|---|---|---|---|
| train ↔ val | 526 | 113 | **26.0% of val** |
| train ↔ test | 1378 | 380 | **33.6% of test** |
| val ↔ test | 221 | 66 | 11.7% of test |

**100% of these pairs share the same class label.** Under coincidence you would
expect ~25% agreement, so these are duplicate images rather than merely similar
anatomy. And because many sit at Hamming distance *zero* on both hashes while
being byte-different, the leakage is in the **source dataset itself** — the
original `Training/` and `Testing/` folders contain the same scans under
different filenames. It is not an artefact of how the split was built.

Contamination is wildly uneven by class:

| class | val w/ near-dup in train | test w/ near-dup in train+val |
|---|---|---|
| glioma | 9.0% | 12.4% |
| meningioma | 10.6% | 33.9% |
| notumor | **68.8%** | **86.8%** |
| pituitary | 19.1% | 12.5% |

347 of 400 test `notumor` images have a near-duplicate in training data. That
class's near-perfect scores are substantially memorisation.

**Correction to an earlier claim in this README:** the exact-hash audit reported
the holdout as uncontaminated. At the byte level that was true; perceptually it
is not. Section 4 reports both the raw and the leakage-adjusted numbers, and the
leakage-adjusted ones are the honest figures.

**Scope limit:** this detects near-duplicate *images*. It cannot detect
same-patient-different-slice leakage, which needs patient IDs the dataset does
not ship. A clean result narrows that risk; it does not eliminate it.

The archive also mixes `L`, `RGB` and `RGBA` modes across **70 distinct image
sizes** — despite the spec describing the data as grayscale. Geometry and
channels are normalised once up front to 3-channel RGB 224×224.

### Split actually used (post-dedup)

85/15 stratified carve-out of `Training/`, seeded (42). `Testing/` is the
untouched holdout, scored once at the end.

| split | glioma | meningioma | notumor | pituitary | total | max–min skew |
|---|---|---|---|---|---|---|
| train | 1190 | 1178 | 1089 | 1158 | **4615** | 8.5% |
| val | 210 | 208 | 192 | 204 | **814** | 8.6% |
| test | 386 | 398 | 400 | 400 | **1584** | 3.5% |

**Imbalance handling:** 8.5% skew is below the spec's 15–20% "significant"
threshold, so training uses **plain `CrossEntropyLoss`**. Inverse-frequency
weights are computed into `data/processed/class_weights.json` and available via
`train.py --class-weights`.

---

## 3. Models

**Phase 1 — baseline CNN** ([`src/model_baseline.py`](src/model_baseline.py)),
422,308 params. Four blocks of `Conv2d → BatchNorm → ReLU → MaxPool`
(3→32→64→128→256, spatial 224→112→56→28→14), global average pool, two dense
layers to 4 logits. AdamW lr=1e-3, batch 32, ≤20 epochs, early stopping on
validation loss (patience 5).

**Phase 2 — transfer** ([`src/model_transfer.py`](src/model_transfer.py)),
ResNet18 / ImageNet, `fc` replaced with a 4-class head. Stage 1 trains the head
alone with the backbone frozen; stage 2 unfreezes `layer3`+`layer4` at lr=1e-4.
Frozen BatchNorm layers are held in eval mode during stage 1 so pretrained
running statistics don't drift while the head is learning against them.

Both phases share one input pipeline ([`src/dataset.py`](src/dataset.py)) for a
fair comparison: ImageNet normalisation, and light augmentation only on train —
horizontal flip, ±10° rotation, ±10% brightness/contrast. No vertical flip or
aggressive cropping, since MRI orientation and anatomy carry diagnostic meaning.

---

## 4. Results

Final numbers on the untouched `Testing/` holdout (n=1584). Full table in
[`outputs/metrics.csv`](outputs/metrics.csv).

> **Read section 2 first.** 36.6% of this holdout near-duplicates the training
> data. The headline table below is the raw holdout score; the
> **leakage-adjusted** table further down (clean subset, n=1004) is the figure
> that actually estimates generalisation. Both are reported rather than only the
> flattering one.

| | baseline CNN | transfer (ResNet18) | Δ |
|---|---|---|---|
| accuracy | 0.8043 | **0.9388** | +0.1345 |
| balanced accuracy | 0.8028 | **0.9375** | +0.1347 |
| macro-F1 | 0.7954 | **0.9367** | +0.1413 |
| params (trainable) | 422,308 | 10,495,492 | |
| best epoch | 16 of 20 | 7 (early stopped at 12) | |

### Per class — the comparison that matters (spec section 7)

| class | base P | tran P | base R | tran R | base F1 | tran F1 | ΔF1 |
|---|---|---|---|---|---|---|---|
| glioma | 0.9206 | 0.9871 | 0.6606 | 0.7927 | 0.7692 | 0.8793 | +0.1101 |
| meningioma | 0.7532 | 0.8770 | 0.5980 | 0.9673 | 0.6667 | **0.9200** | **+0.2533** |
| notumor | 0.8341 | 0.9322 | 0.9675 | 0.9975 | 0.8958 | 0.9638 | +0.0680 |
| pituitary | 0.7476 | 0.9754 | 0.9850 | 0.9925 | 0.8501 | 0.9839 | +0.1338 |

Section 6 predicted improvement "especially on the minority class(es)", and that
is what happened — but the mechanism is worth stating precisely. The classes
aren't imbalanced here (8.5% skew); the *difficulty* is imbalanced. The
baseline's worst class, meningioma, gained the most (**+0.25 F1**), driven almost
entirely by recall: 0.598 → 0.967. The baseline's single largest error mode —
**100 of 398 meningioma scans misread as pituitary** — essentially disappears
(down to 8). Transfer learning helped most exactly where the baseline was
weakest, and the per-class spread narrowed from 0.667–0.896 to 0.879–0.984.

### Both models share one weakness: glioma recall

Glioma is the hardest class for both (baseline 0.66, transfer 0.79 recall). In
the transfer model, **80 of its 97 total errors are missed gliomas** — 51 called
meningioma, 27 called notumor. The 27 glioma→notumor errors are the most serious
in a clinical framing: a missed tumour, not a mislabelled one.

### Test loss is 5.5× val loss — the model is confidently wrong

The transfer model scores val loss 0.0732 but test loss 0.4028, at 93.9%
accuracy. That gap is not noise, and accuracy alone hides it:

| | errors | errors p>0.90 | errors p>0.99 | mean conf. when wrong | % of loss from errors |
|---|---|---|---|---|---|
| transfer / val | 20 | 8 | 2 | 0.796 | 82.1% |
| transfer / test | 97 | 47 | 23 | 0.836 | **96.7%** |
| baseline / test | 310 | 49 | 9 | 0.702 | 84.2% |

Nearly half the transfer model's test errors are made at >90% confidence, and 23
at >99%. Mean confidence on wrong glioma predictions is 0.844. So **its
confidence is not a usable triage signal** — you cannot flag low-confidence cases
for human review, because the failures do not announce themselves. Note the
baseline is actually *better calibrated* when wrong (0.702 vs 0.836); it is just
wrong three times as often.

The val/test gap itself suggests mild distribution shift: the val split is carved
out of `Training/`, while `Testing/` is a separate collection. The same
directional effect appears in the baseline (val 0.371 → test 0.642).

### Leakage-adjusted results — the honest numbers

[`src/leakage_impact.py`](src/leakage_impact.py) scores the holdout split into the
580 images that near-duplicate training data and the 1004 that do not. Full table
in [`outputs/metrics_leakage_adjusted.csv`](outputs/metrics_leakage_adjusted.csv).

| model | subset | n | accuracy | balanced | macro-F1 |
|---|---|---|---|---|---|
| baseline | full holdout | 1584 | 0.8043 | 0.8028 | 0.7954 |
| baseline | contaminated | 580 | 0.8362 | 0.8043 | 0.7626 |
| baseline | **clean** | 1004 | **0.7859** | 0.8077 | **0.7561** |
| transfer | full holdout | 1584 | 0.9388 | 0.9375 | 0.9367 |
| transfer | contaminated | 580 | 0.9862 | 0.9762 | 0.9769 |
| transfer | **clean** | 1004 | **0.9114** | 0.9323 | **0.8857** |

The transfer model scores **0.9862 on images it has effectively already seen** and
0.9114 on ones it hasn't — a 7.5-point gap that is the memorisation signature.
Reported macro-F1 was overstated by **0.051** (0.9367 → 0.8857); the baseline by
0.039.

Per-class recall, clean vs contaminated (transfer): glioma **0.772** vs 0.938,
meningioma 0.966 vs 0.970, notumor 1.000 (n=53) vs 0.997, pituitary 0.991 vs 1.000.
Glioma is where memorisation was doing the most work.

**The headline conclusion survives.** On leakage-clean data the transfer model
still beats the baseline by +0.130 macro-F1 (0.8857 vs 0.7561), so "transfer
learning substantially beats the from-scratch CNN" holds. What does not survive is
the absolute number: this is a ~0.89 macro-F1 model, not a ~0.94 one.

---

## 4b. Post-hoc calibration and glioma threshold tuning

[`src/calibrate.py`](src/calibrate.py) addresses the two measured weaknesses
without retraining. Both parameters are fit on **validation only** — specifically
the leakage-clean val subset (n=602), since fitting on contaminated val would
reward the very overconfidence being corrected — then applied unchanged to test.

- **Temperature scaling:** one scalar, `T = 1.3241`. Logits are divided by `T`, so
  argmax and therefore accuracy are untouched; only confidence is rescaled.
- **Glioma logit offset:** `δ = +2.10` added to the glioma logit, i.e. a lowered
  decision threshold, spending surplus precision (0.989) on missing recall (0.772).

Applied to the leakage-clean holdout (n=1004):

| variant | acc | macro-F1 | NLL | ECE | errors | at p>0.90 | at p>0.99 | glioma P | glioma R |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.9114 | 0.8857 | 0.5880 | 0.0609 | 89 | 43 | 20 | 0.9886 | 0.7722 |
| + temperature | 0.9114 | 0.8857 | 0.4678 | 0.0470 | 89 | 35 | 8 | 0.9886 | 0.7722 |
| + glioma δ | **0.9173** | **0.8980** | **0.3606** | **0.0362** | **83** | **28** | **5** | 0.9400 | **0.8343** |

What this bought:
- **NLL −39%** (0.588 → 0.361) and **ECE −41%** (0.0609 → 0.0362).
- **Confidently-wrong predictions cut 75%**: errors at p>0.99 fall 20 → 5, at
  p>0.90 fall 43 → 28. Confidence becomes a usable triage signal, which was the
  entire point.
- **Glioma recall +6.2 points** (0.772 → 0.834) for 4.9 points of precision
  (0.989 → 0.940) — a favourable trade when the cost of a missed tumour exceeds
  the cost of a second read.
- Accuracy and macro-F1 improved slightly as a side effect, not the goal.

**Methodological caveat, stated plainly:** the *objective* was chosen a priori —
maximise glioma recall subject to precision ≥ 0.90, because raising glioma recall
was the stated goal. A macro-F1 objective on the same val subset picks a milder
`δ = +0.75` (clean-test macro-F1 0.8894, glioma recall 0.7929). The `δ = +2.10`
setting also happens to generalise better, but that was observed after the fact
and is not the reason it was selected. Both are reported so the choice is visible.
Reproduce either with:

```bash
python src/calibrate.py --objective recall_at_precision --precision-floor 0.90   # δ=+2.10
python src/calibrate.py --objective macro_f1                                     # δ=+0.75
```

Parameters land in `outputs/calibration_transfer.json`; the reliability diagrams
and the full δ sweep are in
[`calibration_transfer.png`](outputs/figures/calibration_transfer.png). The sweep
shows macro-F1 is broadly flat across δ ∈ [0, 3] while glioma recall climbs
steadily, which is why the trade is cheap.

**What calibration does not fix:** the underlying leakage, glioma's remaining
0.166 miss rate, or the absence of patient-level splits. It makes the model's
uncertainty honest; it does not make the model better at reading scans.

---

**Figures:** [`confusion_baseline_test.png`](outputs/figures/confusion_baseline_test.png),
[`confusion_transfer_test.png`](outputs/figures/confusion_transfer_test.png),
[`compare_f1_test.png`](outputs/figures/compare_f1_test.png),
[`curves_baseline.png`](outputs/figures/curves_baseline.png),
[`curves_transfer.png`](outputs/figures/curves_transfer.png).

Training curves are worth a look side by side: the transfer model converges
smoothly with a clear step at the unfreeze boundary, while the baseline's
validation loss oscillates violently (spiking to 3.58 and 2.93) even as train loss
falls monotonically. That is BatchNorm statistics being disturbed at a constant
lr=1e-3 with no scheduler — instability, not overfitting (train 0.863 vs val 0.843
at its best epoch). A cosine/step lr schedule is the obvious first improvement to
the baseline; it was left alone deliberately, since section 5's stated goal is a
working pipeline and an honest benchmark rather than peak accuracy.

---

## 5. Grad-CAM — what the models actually look at

[`gradcam_transfer_test.png`](outputs/figures/gradcam_transfer_test.png) and
[`gradcam_baseline_test.png`](outputs/figures/gradcam_baseline_test.png) render
the **same 15 scans** (8 correct, 7 incorrect, spanning all four classes) through
both models, with the CAM computed for the class each model actually predicted.

This turned out to be the most informative deliverable in the project, because
the two models do not differ merely in accuracy — they differ in *what they use*.

**The transfer model attends to brain tissue.** On correct meningioma and glioma
predictions the heat sits on or around the enhancing mass; on pituitary it
concentrates at the sella. Localisation is coarse (ResNet18's `layer4` is 7×7 at
224 input, upsampled ~32×, so the maps are blobby) but anatomically plausible.

**The baseline frequently attends to background and skull.** On several scans it
classifies *correctly at p=1.00* while its CAM peaks in the black corners outside
the head or along the scalp/skull rim — and the actual tumour region reads cold.
Its 80% accuracy is therefore partly earned from framing and acquisition
artefacts that happen to correlate with class, not from the lesion.

That is the concrete payoff of section 8 being mandatory. On the metrics alone
the baseline looks like a respectable weak model; the CAMs show a chunk of its
competence is spurious, and no confusion matrix would have revealed it.

**On the failures**, the overlays separate two distinct error types:
- *Localised but misclassified* — e.g. a true glioma predicted meningioma at
  p=1.00, where the CAM sits squarely on the mass. The model found the lesion and
  misread its type.
- *Mislocalised* — e.g. a true glioma predicted notumor at p=1.00, where heat
  spreads over the ventricles and misses the tumour entirely.

The second kind is the dangerous one, and it lines up with the 27 glioma→notumor
errors above.

**Caveat:** Grad-CAM shows which regions most influenced the logit; it is not a
segmentation and not evidence of clinical correctness. "Roughly corresponds to
the tumour" is the right standard here (as section 8 puts it), and these overlays
were read by eye against no ground-truth tumour masks — the dataset provides
none.

---

## 6. Layout

```
├── archive/                  # raw Kaggle data as supplied (read-only)
├── data/
│   ├── raw/README.md         # points at archive/ + audit summary
│   └── processed/            # 224x224 RGB, deduplicated train/val/test
├── src/
│   ├── data_audit.py         # counts, duplicate + leak detection
│   ├── build_split.py        # dedup, stratified split, normalisation
│   ├── dataset.py            # Dataset, transforms, dataloaders
│   ├── model_baseline.py     # Phase 1 from-scratch CNN
│   ├── model_transfer.py     # Phase 2 ResNet18 + custom head
│   ├── train.py              # training loop, checkpointing, early stopping
│   ├── evaluate.py           # per-class metrics, confusion matrices, curves
│   └── gradcam.py            # Phase 3 interpretability
├── outputs/
│   ├── checkpoints/          # {baseline,transfer}_best.pt
│   ├── figures/              # confusion matrices, curves, Grad-CAM grids
│   └── metrics.csv           # per-class metrics, both phases
└── requirements.txt
```

`data_audit.py` and `build_split.py` are additions to the file list in spec
section 4 — the audit is what surfaced the duplicate problem, and splitting
didn't belong inside `dataset.py`.
