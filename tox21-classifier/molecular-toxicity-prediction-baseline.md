# Molecular Toxicity Prediction — Baseline Project Doc

## 0. Conventions (applies to every phase)
- **Fix random seeds** (Python, NumPy, torch) and record them. GNN results on ~8k molecules are seed-sensitive — run each phase on 3 seeds and report mean ± std ROC-AUC. A single run's 2-point gap is not a result.
- **Include a trivial baseline** before any model: for each task, a predictor that always outputs the base rate. Its ROC-AUC is 0.5 by construction, which is exactly the reference point you need given how imbalanced these labels are.
- **`metrics.csv` schema** — one row per (phase, seed, task), columns:
  `phase, seed, task, n_pos, n_neg, n_missing, roc_auc, pr_auc, notes`
  Both phases append to the same file so comparison is a groupby.
- **Pin dependencies.** `rdkit` / `deepchem` / `torch-geometric` are the most install-fragile stack of these three projects. Install, verify an end-to-end featurization runs, then `pip freeze > requirements.txt`. `torch-geometric` must match the installed `torch` build — install torch first, confirm its version, then install PyG against it.
- **`.gitignore` from the start:** `.venv/`, `data/`, `outputs/checkpoints/`, `__pycache__/`, `*.pth`.

## 1. Objective
Given a molecule (as a SMILES string), predict whether it's toxic across the 12 assay endpoints in the Tox21 dataset. This is **multi-label classification** — each molecule can be toxic on some endpoints and not others, and many labels are missing (not every molecule was tested on every assay).

This project is a genuine step up in complexity from image classification — different data representation (molecular graphs, not pixels), different libraries (RDKit, DeepChem/PyTorch Geometric), and a different evaluation approach (ROC-AUC, not accuracy).

## 2. Dataset
- **Source:** Tox21, available via DeepChem (`deepchem.molnet.load_tox21()`) or MoleculeNet directly — ~8,000 compounds, 12 binary toxicity labels per compound, many missing
- **Split:** Use a **scaffold split**, not a random split. Similar molecules (same core structure) cluster together, and a random split leaks near-duplicate molecules between train and test, giving falsely optimistic results. DeepChem provides scaffold splitting out of the box — use it.
- **Missing labels:** mask them out of the loss calculation rather than treating them as negative — a missing test result is not the same as a negative result

## 3. Environment
```
python >= 3.10
rdkit
deepchem
torch
torch-geometric        # for Phase 2 GNN
scikit-learn
pandas
matplotlib
```

## 4. Project Structure
This project lives inside a shared parent folder alongside the other two pharma projects (`pill-identifier/`, `adr-detector/`):
```
pharma-ml-portfolio/
├── pill-identifier/            ← sibling project, own baseline doc
├── tox21-classifier/           ← this project
│   ├── data/
│   │   └── raw/                 # loaded via deepchem, cache here
│   ├── src/
│   │   ├── featurize.py         # SMILES -> fingerprints (Phase 1) and -> graphs (Phase 2)
│   │   ├── model_baseline.py    # fingerprint + classical/MLP classifier
│   │   ├── model_gnn.py         # graph neural network
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
├── adr-detector/                ← sibling project, own baseline doc
└── README.md                    ← top-level index, not owned by this doc
```
**Keep a separate virtual environment per project.** This one needs `rdkit`/`deepchem`/`torch-geometric`, which have their own dependency requirements — isolating it avoids conflicts with the other two projects' `torch` versions.

## 5. Phase 1 — Fingerprint Baseline
- **Featurization:** convert each SMILES string to a Morgan fingerprint (ECFP, radius 2, 2048 bits), a fixed-length bit vector — the standard, fast starting point in cheminformatics, and it needs no deep learning at all
- **Model — pick one and commit: Random Forest, one model per task** (12 independent classifiers). This is deliberately the simplest option:
  - No masking machinery needed. For each task, drop the rows where that task's label is missing and fit on the rest. sklearn handles this naturally; there's no loss function to mask.
  - No GPU, no training loop, trains in minutes.
  - **Do not build the multi-task MLP for Phase 1.** The masked-`BCEWithLogitsLoss` approach is only relevant if you go the neural route, and mixing the two paths here is how this phase turns into a week of plumbing. If you want the MLP, it belongs in stretch goals.
- **Tune it properly.** A lazily-fit RF is a straw man, and the whole point of Phase 2 is to beat something real. Do at least a modest hyperparameter search (n_estimators, max_depth, min_samples_leaf).
- **Goal:** establish the per-task ROC-AUC benchmark. This is a legitimate, commonly-published baseline — not a placeholder step.

## 6. Phase 2 — Graph Neural Network
- **Representation:** molecule as a graph — atoms as nodes (with features: atom type, charge, hybridization, etc.), bonds as edges
- **Architecture:** GCN or GIN (Graph Isomorphism Network) via PyTorch Geometric, 3–4 message-passing layers, then a pooling layer (mean or max) to get a molecule-level embedding, then a classification head for the 12 tasks
- **Loss:** BCEWithLogitsLoss across the 12 tasks, **masked for missing labels** — this is where masking actually applies (Phase 1's RF sidesteps it by dropping rows per task)
- **Same scaffold split** as Phase 1, so the comparison is fair
- **Expectation — read this before you interpret your results.** Do *not* assume the GNN will win. On Tox21 specifically (~8k molecules, scaffold split), a well-tuned Morgan fingerprint with Random Forest or gradient boosting is routinely competitive with — and frequently better than — a GCN/GIN trained from scratch. Graph networks tend to need more data than this to pull ahead. So:
  - The honest framing is: *does a learned graph representation beat a hand-crafted fingerprint on this dataset at this scale?* That's the experiment. Either answer is a real result.
  - If the fingerprint baseline wins, that is not a bug and not a failed project — it's the finding, and it's a well-supported one. Write it up as such rather than tuning the GNN until it wins.

## 7. Evaluation
- **Primary metric: ROC-AUC per task**, then mean ROC-AUC across all 12 — this is the standard Tox21 benchmark metric, not accuracy (labels are heavily imbalanced — most molecules are non-toxic on most assays)
- **Also report PR-AUC per task.** With this much class imbalance, ROC-AUC can look respectable while the model is nearly useless at the positive class. PR-AUC is the sharper instrument here; report both.
- Report Phase 1 vs. Phase 2 side by side, **per task**, not just the average — and with mean ± std across seeds per section 0
- Flag any task where the model performs at or near random (AUC ~0.5) rather than burying it in the average. With 12 tasks, one or two near-random tasks is normal and worth stating plainly.

## 8. Interpretability (stretch, not required for baseline)
- Atom-level attribution (e.g. via GNNExplainer or integrated gradients) to highlight which substructures the model associates with toxicity — genuinely useful but adds real complexity, so don't block the baseline deliverable on this

## 9. Deliverables
- Fingerprint baseline model + per-task ROC-AUC
- GNN model + per-task ROC-AUC
- Side-by-side comparison table/figure
- README explaining the scaffold split choice and what the per-task results suggest

## 10. Stretch Goals
- Try a pretrained molecular representation (e.g. ChemBERTa) instead of training a GNN from scratch
- Substructure attribution/interpretability from section 8
- Extend to a second dataset (e.g. ClinTox) to check generalization
