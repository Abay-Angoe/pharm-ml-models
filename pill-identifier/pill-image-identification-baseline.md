# Pill Image Identification — Baseline Project Doc

## 0. Conventions (applies to every phase)
- **Fix random seeds** (Python, NumPy, torch) and record them. When comparing two models, a 2-point gap can be seed noise — if time allows, run each phase on 3 seeds and report mean ± std rather than a single number.
- **Include a trivial baseline** before any model: majority-class predictor and stratified-random predictor. Three lines of code, and it's what stops "94% accuracy!" from looking like a result when 94% is the majority class.
- **`metrics.csv` schema** — one row per (phase, seed, class), columns:
  `phase, seed, class, support, precision, recall, f1, accuracy_overall, macro_f1, notes`
  Both phases append to the same file so comparison is a groupby, not a manual merge.
- **Pin dependencies.** Install, verify it works, then `pip freeze > requirements.txt`. Don't hand-write unpinned package names.
- **`.gitignore` from the start:** `.venv/`, `data/`, `outputs/checkpoints/`, `__pycache__/`, `*.pth`. This is a portfolio repo — don't commit a 6.8GB download or model weights.

## 1. Objective
Build an image classifier that identifies a pharmaceutical pill/tablet from a photo, predicting which drug it is based on shape, color, and imprint.

**Scoping constraint:** the full NIH pill libraries cover ~1,000+ distinct pills. That is not a tractable first project. Scope to **15–30 classes** of visually distinct, common medications.

## 2. Dataset — read this section carefully before downloading anything
The NIH pill data is real and still downloadable, but it is **unmaintained** (C3PI was sunset in 2018; Pillbox was retired January 2021), so expect link rot and no support. More importantly, the different NIH releases have wildly different images-per-class, and picking the wrong one will dead-end this project:

| Release | Size | Images per pill | Usable here? |
|---|---|---|---|
| C3PI **Reference** images | ~4,000 images, 6.8GB | ~2 (front/back, controlled lighting) | **No** — far too few per class |
| NLM Pill Image Recognition Challenge set | 2,000 reference + 5,000 consumer, 1,000 pills | ~5 consumer photos | **No** — too few per class for flat classification |
| C3PI **Training** images | ~133,000 consumer-grade images | enough to subset meaningfully | **Yes — use this one** |

- **Use the C3PI Training image set.** It's the only release with enough real photos per pill to support a 15–30 class classification problem.
- **Action before any training:** count actual images per NDC/class in the subset you pick, and report the counts. Target **≥100 real images per class** — real, not post-augmentation. Augmentation doesn't add information; it only regularizes what's already there. Drop any class that can't clear the floor rather than augmenting it to look bigger.
- **Split:** 70/15/15 train/val/test, stratified by class.
- **Context worth knowing:** the published literature on this dataset mostly treats pill ID as an image *retrieval/matching* problem (reference photo ↔ consumer photo) rather than flat classification, precisely because most releases are few-shot per pill. Flat classification on a curated 15–30 class subset is a valid learning project, but it isn't the standard framing — worth a line in the README so the scoping is deliberate rather than accidental.

## 3. Environment
```
python >= 3.10
torch
torchvision
scikit-learn
matplotlib
seaborn
pandas
tqdm
grad-cam
```
Install, confirm a GPU tensor op works, then freeze to pinned versions per section 0.

## 4. Project Structure
This project lives inside a shared parent folder alongside the other two pharma projects (`tox21-classifier/`, `adr-detector/`):
```
pharma-ml-portfolio/
├── pill-identifier/            ← this project
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── src/
│   │   ├── dataset.py
│   │   ├── model_baseline.py
│   │   ├── model_transfer.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── gradcam.py
│   ├── outputs/
│   │   ├── checkpoints/
│   │   ├── figures/
│   │   └── metrics.csv
│   ├── requirements.txt
│   ├── .gitignore
│   ├── .venv/                  ← own virtual environment, kept separate from siblings
│   └── README.md
├── tox21-classifier/           ← sibling project, own baseline doc
├── adr-detector/               ← sibling project, own baseline doc
└── README.md                   ← top-level index, not owned by this doc
```
**Keep a separate virtual environment per project.** This project needs `torch`/`torchvision`; the toxicity project needs `rdkit`/`torch-geometric`; the ADR project needs `transformers`. Sharing one environment across all three risks version conflicts, particularly around `torch`.

## 5. Phase 1 — Baseline CNN
- **Input size:** 224x224, normalized
- **Architecture:** 3–4 conv blocks (Conv2d → BatchNorm → ReLU → MaxPool) → dense layers → N output logits (N = number of classes chosen)
- **Loss:** CrossEntropyLoss, weighted if class counts are uneven
- **Optimizer:** AdamW, lr=1e-3, batch size 32
- **Augmentation — deliberately conservative, and here's why:** section 8 checks whether the model is reading the pill's *imprint*. Augmentation that destroys imprint legibility trains the model to ignore the exact feature we then test for. So:
  - **Rotation: yes, full 0–360°.** Pills are photographed at arbitrary angles; imprints stay legible under rotation.
  - **Blur: very mild only** (small Gaussian sigma). Phone photos aren't always sharp, but heavy blur erases imprint text.
  - **Vertical/horizontal flip: no.** Flipping produces mirror-image imprints, which don't exist on real pills. This teaches the model that reversed text is valid — actively harmful here, even though flips are standard elsewhere.
  - Brightness/contrast jitter: yes, modest.
- **Goal:** working pipeline and a benchmark number, not peak performance

## 6. Phase 2 — Transfer Learning
- **Backbone:** ResNet18 or EfficientNet-B0, pretrained on ImageNet
- **Strategy:** freeze backbone, train head first, then unfreeze last 1–2 blocks and fine-tune at lower lr
- **Hypothesis to test, not a guaranteed result:** pretrained features usually help on fine-grained visual detail like imprints and color, so transfer learning is *expected* to win — but with only 15–30 classes and limited data, a from-scratch CNN can land closer than you'd think. Report whichever wins honestly; "the baseline held up" is a legitimate finding, not a failure.

## 7. Evaluation
- **Top-1 accuracy** as the headline number, plus per-class precision/recall/F1 and macro-F1.
- **Do not report top-5.** With 15–30 classes, top-5 is ~17–33% by chance and conveys nothing. (The literature uses Top-K because it works with ~1,000 classes.) Top-3 is acceptable as a secondary signal if you want a "close enough for a lookup tool" measure — label it as such.
- Confusion matrix — with this many classes, analyze the most-confused *pairs* specifically rather than trying to read the whole grid. Expect confusion between same-color/same-shape pills; that's the interesting result.
- Compare against the trivial baselines from section 0.

## 8. Interpretability
- Grad-CAM on a sample of correct and incorrect predictions per class
- **Key question:** is the model keying on the imprint/text region, or just overall color and shape? If it's the latter, it will fail exactly where a real tool needs to work — distinguishing two white round pills. This is the most informative single check in the project.

## 9. Deliverables
- Checkpoints for both models
- `metrics.csv` per the section 0 schema, covering trivial baselines + both phases
- Confusion matrix figure(s) + most-confused-pairs table
- Grad-CAM figure grid
- README summarizing: which C3PI release and which classes (with counts), why that subset, trivial-baseline vs. both models, and what Grad-CAM showed about imprint vs. color/shape

## 10. Stretch Goals
- Expand class count once the pipeline is validated on the small set
- Reframe as retrieval/matching (reference ↔ consumer image) to match how the literature treats this problem
- Multi-attribute prediction (shape, color, imprint as separate heads)
- Simple demo: upload a photo, get top-3 predictions with confidence
