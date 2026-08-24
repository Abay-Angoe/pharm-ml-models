# Pharma ML Portfolio

Three medical/pharmaceutical ML projects, each a self-contained sub-project with its own environment:

```
pharma-ml-portfolio/
├── pill-identifier/       # image classification — pill ID by shape/color/imprint
├── tox21-classifier/      # cheminformatics — multi-label molecular toxicity prediction
├── adr-detector/          # NLP — adverse drug reaction detection from text
└── README.md              # you are here
```

## Projects

| Project | Task | Data type | Key libraries |
|---|---|---|---|
| `pill-identifier/` | Multi-class image classification | Pill photos (C3PI training set) | torch, torchvision |
| `tox21-classifier/` | Multi-label classification | SMILES / molecular graphs | rdkit, deepchem, torch-geometric |
| `adr-detector/` | Binary text classification | Sentences | transformers, datasets |

Each subfolder has its own baseline spec doc, `requirements.txt`, `.gitignore`, and virtual environment.

## Shared conventions

Each project's baseline doc restates these in its own section 0, but they apply across the repo:

**Separate venv per project — not one shared.** `torch-geometric` must be installed against a specific `torch` build, and `transformers` carries its own `torch` pin. Sharing one environment across all three is how you end up debugging pip resolution instead of models.

**Every project ships a trivial baseline.** Majority-class and stratified-random predictors, computed before any real model. This is the guard against reporting high accuracy that's really just class imbalance.

**Every project ships a tuned classical baseline before its deep model.** RF on fingerprints, TF-IDF + linear, from-scratch CNN. An untuned baseline flatters the fancy model and teaches you nothing.

**Seeds are fixed and results are reported as mean ± std across 3 seeds.** All three projects have small enough datasets that single-run differences of a couple points are noise.

**"The simple model won" is a valid result** in all three projects and gets written up as a finding, not tuned away. This is most likely on `tox21-classifier`, where fingerprint+RF is genuinely competitive with graph networks at this data scale.

**Never commit:** `.venv/`, `data/`, `outputs/checkpoints/`, `*.pth`. The pill dataset alone is multi-GB.

## Status

| Project | Trivial baseline | Phase 1 | Phase 2 | Evaluated | Interpretability |
|---|---|---|---|---|---|
| pill-identifier | ☑ | ☑ | ☑ | ☑ | ☑ |
| tox21-classifier | ☐ | ☐ | ☐ | ☐ | ☐ (stretch) |
| adr-detector | ☐ | ☐ | ☐ | ☐ | ☐ (stretch) |

### pill-identifier — complete

NIH C3PI consumer photographs, 20 products on NDC-9, 1,212 real images (47-101 per
class). Test set 184 images, mean ± std over 3 seeds:

| Model | Top-1 (headline) | Top-3 (secondary) | Macro F1 |
|---|---|---|---|
| majority-class / stratified-random | 8.15% / 5.40 ± 1.67% | 22.83% / 15.08% | — |
| Phase 1 — CNN from scratch | 46.01 ± 2.99% | 74.64 ± 1.26% | 0.462 |
| **Phase 2 — ResNet18 (ImageNet)** | **90.58 ± 2.45%** | **98.91 ± 0.94%** | **0.890** |

+44.6pp = 11.5× the combined seed std, so the gap is real. "The simple model won" did
not happen here. Top-5 is deliberately not reported (25% by chance at 20 classes).

**Two negative interpretability results, both worth having.** Grad-CAM shows
**neither model reads the imprint** (imprint-attention correlation −0.056 baseline,
+0.003 transfer). And the strong colour shortcut measured on a different pill dataset
(ρ = +0.77) **does not replicate here** (ρ = −0.04 / −0.20) — it was dataset-specific,
and reporting it as general would have been wrong. What actually limits the scratch
CNN is crop quality: weak localisation costs it 24pp versus 6pp for the pretrained
model.

**A first set of results was retracted, not patched.** An image-level split gave 97.3%
val top-1; perceptual checking found 21% of test images were near-identical twins of
training images, because C3PI shoots the same tablet repeatedly in one session.
Byte-identical duplicate checks had found nothing. Re-split by capture group (675
groups over 1,212 images) → 0% twins, and every number above comes from that split.

**Scope, stated plainly:** 20 of ~2,362 photographed C3PI products (0.8%), a closed
set that cannot say "unknown", and it does not read imprints — so it cannot separate
two white tablets differing only by marking. See [pill-identifier/README.md](pill-identifier/README.md) §7, and
[pill-identifier/STATUS.md](pill-identifier/STATUS.md) for the working log,
spec deviations and the two retracted result sets.
