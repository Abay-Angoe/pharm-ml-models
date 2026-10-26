# Brain Tumor MRI Classification — Baseline Project Doc

## 1. Objective
Build an image classifier that takes a brain MRI scan and predicts one of four classes:
- `glioma`
- `meningioma`
- `pituitary`
- `notumor`

Two phases, built in order:
1. **Baseline CNN** trained from scratch — establishes a working pipeline and a benchmark score.
2. **Transfer learning model** (pretrained backbone, fine-tuned) — should meaningfully beat the baseline.

Success isn't just "high accuracy" — it's a model whose per-class performance is understood and whose predictions can be visually explained (Grad-CAM).

## 2. Dataset
- **Source:** Kaggle — "Brain Tumor MRI Dataset" (Masoud Nickparvar)
- **Size:** ~7,000 images, grayscale MRI scans, 4 classes
- **Structure as downloaded:** pre-split into `Training/` and `Testing/` folders, each with one subfolder per class
- **Action needed:** carve a validation split out of `Training/` (suggest 85/15 train/val), leave `Testing/` untouched as the final holdout
- **Known issue to check:** class counts are not perfectly balanced — verify counts per class before training and handle imbalance (weighted loss or oversampling) if the skew is significant (>15–20% difference between smallest and largest class)

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
grad-cam          # pytorch-grad-cam package, for Phase 4
kaggle            # for dataset download via API
```
GPU not required but strongly preferred (Colab free tier is enough for both phases).

## 4. Project Structure
```
brain-tumor-classifier/
├── data/
│   ├── raw/                  # untouched Kaggle download
│   └── processed/            # train/val/test split actually used
├── src/
│   ├── dataset.py            # Dataset class, transforms, dataloaders
│   ├── model_baseline.py     # from-scratch CNN
│   ├── model_transfer.py     # pretrained backbone + custom head
│   ├── train.py              # training loop, checkpointing
│   ├── evaluate.py           # metrics, confusion matrix
│   └── gradcam.py            # interpretability visualizations
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   └── metrics.csv
├── requirements.txt
└── README.md
```

## 5. Phase 1 — Baseline CNN
- **Input size:** 224x224, resized + center-cropped, normalized
- **Architecture:** 3–4 blocks of `Conv2d → BatchNorm → ReLU → MaxPool`, followed by 1–2 dense layers into 4 output logits
- **Loss:** CrossEntropyLoss (weighted if class imbalance is significant)
- **Optimizer:** AdamW, lr=1e-3
- **Batch size:** 32
- **Epochs:** ~15–20, with early stopping on validation loss
- **Augmentation:** light — horizontal flip, small rotation (±10°), slight brightness/contrast jitter. Avoid aggressive augmentation; MRI orientation and structure carry diagnostic meaning.
- **Goal:** working end-to-end pipeline, not peak accuracy. Record accuracy/F1 as the number to beat.

## 6. Phase 2 — Transfer Learning
- **Backbone:** ResNet18 or EfficientNet-B0, pretrained on ImageNet
- **Strategy:** freeze backbone initially, train only the replaced classification head for a few epochs, then unfreeze the last 1–2 blocks and fine-tune at a lower lr (1e-4 or 1e-5)
- **Same input pipeline** as Phase 1 for a fair comparison
- **Expectation:** noticeable improvement over the baseline, especially on the minority class(es)

## 7. Evaluation (applies to both phases)
- Report **per-class precision, recall, F1** — not just overall accuracy. A model that's 95% accurate but weak on one tumor type is not acceptable in this domain.
- Confusion matrix, saved as a figure
- Track train/val loss curves to check for overfitting
- Final numbers reported on the untouched `Testing/` set only, once, at the end

## 8. Phase 3 — Interpretability
- Add Grad-CAM overlays for a handful of correct and incorrect predictions per class
- Save as figures — visually, the highlighted region should roughly correspond to the tumor location when the prediction is correct
- This is a required deliverable, not optional polish — it's the difference between "trust me" and showing your work

## 9. Deliverables
- Trained checkpoints for both models
- `metrics.csv` with per-class metrics for both phases
- Confusion matrices (baseline vs. transfer) as figures
- Grad-CAM figure grid
- Short README section summarizing: dataset, both models' results side by side, and what the Grad-CAM visualizations show

## 10. Stretch Goals (optional, after baseline works)
- 5-fold cross-validation instead of a single split
- Try a second backbone (e.g., DenseNet121) and compare
- Simple Gradio/Streamlit demo for uploading a scan and getting a prediction + Grad-CAM overlay
