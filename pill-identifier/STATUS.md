# Status

Working log for `pill-identifier`. [README.md](README.md) is the polished write-up;
this file records **what state things are in, what was decided and why, and what went
wrong** — the things a reader (or a future me) needs that don't belong in a results
document.

**Last updated:** 2026-08-18 · **State: complete** for §0–§9 of
[the spec](pill-image-identification-baseline.md), 1 of 4 §10 stretch goals done.

---

## 1. Where things stand

| | |
|---|---|
| Dataset | NIH C3PI Training set, **20 products** (NDC-9), **1,212 real images**, 47–101/class |
| Split | 819 / 209 / 184, grouped by capture session, verified leak-free |
| Phase 1 | CNN from scratch, 459,828 params — **46.01 ± 2.99%** test top-1 |
| Phase 2 | ResNet18 ImageNet — **90.58 ± 2.45%** test top-1 |
| Seeds | 42 / 1 / 2, all headline numbers are mean ± std |
| Interpretability | Grad-CAM + colour-shortcut test, both **negative** (see §4 below) |
| Demo | FastAPI + web UI with **measured** OOD rejection |
| Training cost | 7.9 h CPU across 6 runs (one epoch stalled to 57 min under contention) |

Nothing is running. Everything is on disk and reproducible from
[README §6](README.md#6-reproducing).

## 2. Results

Test set, 184 held-out images, 20 classes, mean ± std over 3 seeds:

| Model | Top-1 (headline) | Top-3 (secondary) | Macro F1 |
|---|---|---|---|
| Stratified-random | 5.40 ± 1.67% | 15.08% | — |
| Majority-class | 8.15% | 22.83% | — |
| Phase 1 — scratch CNN | 46.01 ± 2.99% | 74.64 ± 1.26% | 0.462 |
| **Phase 2 — ResNet18** | **90.58 ± 2.45%** | **98.91 ± 0.94%** | **0.890** |

+44.56pp = **11.5× the combined seed std**. §6 allowed "the baseline held up" as a
finding; it did not hold up.

Secondary result worth keeping in view: accuracy splits sharply by crop quality.
Weak localisation costs the scratch CNN **−24.1pp** but the pretrained model only
**−6.4pp**, so a chunk of the headline gap is my pill detector rather than the models.

## 3. Two retractions

Both were caught by checking rather than by anything failing loudly. Recorded here
because the numbers appeared in earlier conversation and should not resurface.

**97.3% val top-1 — retracted (near-duplicate leakage).** C3PI shoots the same
tablet repeatedly in one session. A per-image split put near-identical twins on both
sides: **21% of test images sat within 0.02 cosine distance of a training image**
(different-class floor is 0.096; closest pair 0.0008). Byte-identical duplicate checks
had found **nothing** — the leak was perceptual, not exact. Fixed by clustering
captures and splitting whole groups (675 groups over 1,212 images; 58% of images
belong to a multi-image group). Post-fix: 0% within 0.05, for **both** test and valid.

**OGYEIv2 results — withdrawn entirely.** Archived at
`outputs/_ogyeiv2_archive/`. Two independent reasons: trained with horizontal/vertical
flips, which §5 forbids because a mirrored imprint exists on no real pill; and built
on the wrong dataset. The interpretability conclusion drawn from them (a strong colour
shortcut, ρ = +0.77) **did not replicate** on C3PI — see §4.

## 4. The interpretability answer, and why it changed

§8 asks whether the model reads the imprint or shortcuts on colour/shape. On C3PI:

| measure | baseline | transfer |
|---|---|---|
| `imprint_corr` | −0.056 ± 0.011 | +0.003 ± 0.003 |
| ρ(colour saturation, per-class F1) | −0.04 | −0.20 |
| F1 pale half / colourful half | 0.490 / 0.434 | 0.875 / 0.905 |

**Neither model reads the imprint, and there is no colour shortcut.** On OGYEIv2 I had
measured a textbook colour shortcut (ρ = +0.77, F1 0.22 on white pills vs 0.89 on
coloured). It vanished on the mandated dataset. Had I not rerun, I would have written
that up as a general property of scratch CNNs — it was a property of that dataset.

The model appears to use gross appearance (outline, size, aspect, broad colour,
texture), which suffices for 20 mostly-distinct products. That predicts the observed
failure mode and it holds: errors inside look-alike groups rise from **8% → 25%** as
accuracy improves, i.e. what survives is exactly the cases where only the imprint
discriminates.

**One number not to trust:** transfer's `on_pill` is 0.409, *lower* than baseline's
0.535, while being twice as accurate. Either it genuinely uses context, or my
estimated mask under-covers the pill. C3PI has no ground-truth masks, so this is
unresolvable here and no conclusion is drawn from it.

## 5. Bugs found in my own work

Kept because each one would have silently corrupted a reported result.

| what | effect | how it surfaced |
|---|---|---|
| Counted index **rows**, not unique images | inflated 9 products exactly 2×; all 9 were in the first 20-class selection | download reconciled to 1,022 files against 1,285 "expected" |
| `train.py` passed stale `masked=` arg | both training runs died instantly | ran it — imports had passed clean because the call is inside `main()` |
| `_signature` returned 2-D, not flattened | matrix product instead of dot; crashed at `float()` | assertion in the patch script, then a real run |
| `CLASS_TO_GROUP.get()` → `None == None` | two *unrelated* solo classes counted as "same look-alike group"; inflated the in-group error stat from 25% → 38% | reading the confused-pairs output and not believing `fludrocortisone → hyzaar` was same-group |
| Figure panel title asserted the OGYEIv2 conclusion | title read "transfer breaks the colour shortcut" while its own legend showed ρ = −0.04 | looked at the rendered figure |
| Localiser v1/v2 | v1 boxed illumination gradients (89–98% of frame); v2 boxed a **US quarter** used as a size reference | visual audit of 12 crops, twice |
| Stale OGYEIv2 numbers in docs/code | `30 classes`, `840 images`, `1,285 images`, dropped classes still in the look-alike table | grep sweep during the final spec audit |

Lesson that generalised: **importing a module proves almost nothing.** Two of these
passed an import check and still failed on first execution. Functions now get
unit-tested against known-answer cases (`spearman` on monotonic/reversed/tied input,
`erode` on a known square, `same_group` on known pairs, the grouped split on synthetic
group sizes).

## 6. Deviations from the spec

All three are deliberate and documented in the README, not silently dropped.

| § | requirement | status |
|---|---|---|
| §2 | ≥100 real images per class | **not met — not achievable.** Only 1 of 2,362 NDC-9 products in C3PI reaches 100 unique consumer photos; 9 reach 60. Dropping classes below the bar leaves far under the 15-class floor. |
| §2 | 70/15/15 split | **67.6 / 17.2 / 15.2.** Capture groups are indivisible (largest holds 51 images). Splitting them is what produced the retracted 97.3%. |
| §3 | "confirm a GPU tensor op works" | **adapted** — `torch.cuda.is_available()` is False on this host, so a CPU forward+backward was verified instead. |

## 7. Demo

`demo/` — upload a photo, get top-3 with confidence or an honest refusal.
See [demo/README.md](demo/README.md).

```bash
python -m uvicorn demo.app:app --port 8000    # http://127.0.0.1:8000
```

The part that took the actual work is the rejection threshold. The model is a closed
set, so **any** photo returns one of the 20 classes. Out-of-set images have median
confidence 0.637; in-distribution *wrong* predictions sit at 0.639 — max-softmax
cannot separate "never seen this pill" from "got this pill wrong".

Measured on 100 photos of 20 products outside the 20 classes:

| threshold | coverage | precision | out-of-set still named |
|---|---|---|---|
| 0.560 — naive "keep 95% of correct" | 97% | 95.6% | **61%** |
| **0.828 — used** | **76.6%** | **99.4%** | **24%** |
| 0.90 | 67.0% | 100% | 15% |

The obvious threshold would have let **61%** of unknown pills through with a confident
drug name. The chosen rule (lowest threshold reaching 99% precision, decided on
validation) cuts that to 24% — still one in four, stated in the UI. The app **refuses
to start** if the measurement file is absent rather than inventing a cutoff.

## 8. What is not done

§10 stretch goals, in the order I'd tackle them:

1. **Better OOD detection** — 24% leakage at 76% coverage is the weakest part of the
   demo. Energy scores, temperature scaling, or a trained rejector.
2. **Better localisation** — highest leverage for Phase 1 (−24pp on fallback crops),
   and it would let the imprint question be asked at a resolution where the answer
   means something.
3. **Retrieval/matching reframe** — how the literature actually treats this, and the
   only version that scales: adding a pill means adding a reference image, not
   retraining. The scaling table in [README §7](README.md) shows why more classes
   won't work (2,349 classes at ~10 photos each).
4. **Multi-attribute heads** (shape / colour / imprint), which would make the imprint
   question directly supervised rather than inferred.

Also unused: **EfficientNet-B0** is implemented and smoke-tested but never trained;
§6 asks for one of the two backbones and ResNet18 is the one with numbers.

## 9. File map

```
src/
├── classes.py            20-class registry (NDC-9 ids, labels, look-alike groups)
├── fetch_c3pi.py         resolve + download the subset from the NLM index
├── localize.py           pill detector + audited centre-crop fallback
├── preprocess.py         crop -> dedup capture groups -> grouped 70/15/15 split
├── check_leakage.py      perceptual train-vs-heldout duplicate audit
├── dataset.py            loaders + conservative augmentation (no flips)
├── model_baseline.py     Phase 1 CNN
├── model_transfer.py     Phase 2 backbones + staged freeze/unfreeze
├── train.py              training loop, top-1/top-3
├── trivial_baseline.py   majority-class + stratified-random
├── evaluate.py           top-1, per-class metrics, confusion + confused pairs
├── gradcam.py            attention measures + figure grid
├── ood_rejection.py      out-of-set rejection measurement
├── aggregate_seeds.py    mean ± std across seeds
└── make_figures.py       4-panel summary figure

demo/app.py + demo/static/index.html

outputs/
├── metrics.csv                  §0 schema, 128 rows, 4 phases
├── summary_seeds.csv            mean ± std per phase
├── ood_rejection.json           threshold sweep + chosen operating point
├── preprocess_stats.json        crop + capture-group stats
├── history_*.csv|json           per-epoch, 6 runs
├── gradcam_analysis_*.json      per-image attention measures, 6 runs
├── confused_pairs_*.csv         most-confused-pairs tables, 6 runs
├── checkpoints/                 6 .pt (gitignored)
├── figures/                     14 png (gitignored)
└── _ogyeiv2_archive/            superseded — do not quote
```

Repo hygiene: `requirements.txt` is `pip freeze` (52 pins; `fastapi`/`uvicorn`/
`python-multipart` are demo-only). Verified that no secret, checkpoint or >5 MB file
is committable.
