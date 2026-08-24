# Adverse Drug Reaction (ADR) Detection — Baseline Project Doc

## 0. Conventions (applies to every phase)
- **Fix random seeds** (Python, NumPy, torch) and record them. Transformer fine-tunes on small datasets are notoriously seed-sensitive — run 3 seeds and report mean ± std F1. A single run beating the baseline by 2 points is not evidence.
- **Include a trivial baseline** before any model: majority-class predictor and stratified-random predictor. On an imbalanced binary task this is essential — a majority-class predictor can post high accuracy with zero recall on the positive class, which is exactly the failure mode to guard against here.
- **`metrics.csv` schema** — one row per (phase, seed, class), columns:
  `phase, seed, class, support, precision, recall, f1, accuracy_overall, macro_f1, notes`
  Both phases append to the same file so comparison is a groupby.
- **Pin dependencies.** Install, verify a forward pass runs, then `pip freeze > requirements.txt`.
- **`.gitignore` from the start:** `.venv/`, `data/`, `outputs/checkpoints/`, `__pycache__/`, `*.pth`.

## 1. Objective
Build a text classifier that determines whether a sentence describes an adverse drug reaction (ADR/ADE) or not — binary classification. This moves into NLP: tokenization, embeddings, and (in Phase 2) transformer fine-tuning, rather than CNNs.

## 2. Dataset
- **Source:** ADE Corpus V2 via Hugging Face `datasets` (`ade_corpus_v2`), **classification config** — sentences from medical case reports labeled ADE-related or not. Note the dataset ships multiple configs (classification, drug–AE relation, drug–dosage relation); confirm you've loaded the classification one.
- **Measure the class balance, don't assume it.** Load it, count both classes, and report the actual ratio before deciding anything. Only then choose whether to apply class weighting or resampling. Don't take a stated ratio from this doc or from memory as fact — the numbers should come from the data you actually loaded.
- **Split:** this config does **not** ship a standard train/val/test split. Create your own: stratified 80/10/10, seeded and saved to disk so both phases train and evaluate on exactly the same partition. This is non-negotiable — comparing two models across different splits produces a meaningless comparison.
- **One thing to check while you're in there:** the corpus is sentence-level and drawn from case reports, so sentences from the same source document can land in different splits. If document IDs are available, prefer grouping by document to avoid leakage. If they aren't, note the limitation in the README rather than ignoring it.

## 3. Environment
```
python >= 3.10
transformers
datasets
torch
scikit-learn
pandas
matplotlib
```

## 4. Project Structure
This project lives inside a shared parent folder alongside the other two pharma projects (`pill-identifier/`, `tox21-classifier/`):
```
pharma-ml-portfolio/
├── pill-identifier/            ← sibling project, own baseline doc
├── tox21-classifier/           ← sibling project, own baseline doc
├── adr-detector/                ← this project
│   ├── data/
│   │   └── raw/                 # cached via HF datasets
│   ├── src/
│   │   ├── preprocess.py
│   │   ├── model_baseline.py    # TF-IDF + linear classifier
│   │   ├── model_transformer.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── outputs/
│   │   ├── checkpoints/
│   │   ├── figures/
│   │   └── metrics.csv
│   ├── requirements.txt
│   ├── .gitignore
│   ├── .venv/                   ← own virtual environment, kept separate from siblings
│   └── README.md
└── README.md                    ← top-level index, not owned by this doc
```
**Keep a separate virtual environment per project.** This one needs `transformers`/`datasets`, which pull in their own `torch` version pin — isolating it avoids conflicts with the other two projects.

## 5. Phase 1 — Classical NLP Baseline
- **Featurization:** TF-IDF vectorization of sentences
- **Model:** Logistic Regression or linear SVM
- **Why this matters:** this is a legitimate, fast baseline that's genuinely hard to beat by a wide margin on short-text classification — don't skip it to jump straight to a transformer. It gives you a real number to compare against.
- **Tune it, don't strawman it.** Sweep the basics: n-gram range (1,1) vs (1,2), min_df, and the regularization strength C. An untuned baseline makes Phase 2 look better than it is, which defeats the purpose of having a baseline.
- **Goal:** establish precision/recall/F1 benchmark, particularly recall on the ADE class

## 6. Phase 2 — Transformer Fine-Tuning
- **Model:** DistilBERT for a lighter/faster fine-tune, or BioBERT/ClinicalBERT if you want a domain-pretrained option (worth trying since the text is medical case-report language, not general English)
- **Setup:** standard sequence classification fine-tuning — lr ~2e-5, batch size 16, 3–4 epochs, using Hugging Face `Trainer` or a manual PyTorch loop
- **Expectation — likely but not guaranteed.** A transformer should beat TF-IDF here, particularly on recall, since it can pick up phrasing that bag-of-words features miss. But TF-IDF on short single-sentence text with distinctive vocabulary is a strong baseline, and the margin is often smaller than people expect. If the gap is within seed variance (section 0), the honest conclusion is "no meaningful difference," not "the transformer won."

## 7. Evaluation
- **Precision, recall, and F1 for the ADE (positive) class specifically** — this is the metric that matters most here. In this domain, a false negative (missing a real adverse reaction mention) is a worse failure than a false positive, so track recall carefully, not just overall accuracy.
- **Report overall accuracy only alongside the trivial baselines from section 0**, never on its own. On an imbalanced binary task, accuracy is the number most likely to mislead you.
- Confusion matrix for both phases
- Compare Phase 1 vs. Phase 2 side by side on the same test set, with mean ± std across seeds

## 8. Interpretability (stretch, not required for baseline)
- Token-level attribution (e.g. via Captum or attention weight visualization) to show which words drove a classification — useful for sanity-checking that the model is keying on actual symptom/reaction language rather than spurious correlations

## 9. Deliverables
- TF-IDF baseline model + metrics
- Fine-tuned transformer model + metrics
- Side-by-side precision/recall/F1 comparison
- Confusion matrices for both
- README summarizing dataset, class imbalance handling, and results

## 10. Stretch Goals
- Extend from binary to multi-label (which specific reaction is mentioned, not just yes/no)
- Try BioBERT vs. ClinicalBERT vs. DistilBERT head-to-head
- Simple demo: paste a sentence, get ADE probability + highlighted key terms
