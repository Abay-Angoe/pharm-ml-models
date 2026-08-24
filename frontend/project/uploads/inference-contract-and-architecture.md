# Inference Contract & App Architecture — Medical ML Portfolio

## Why this document exists
The frontend will be rewritten or abandoned. The trained models should outlive it. This spec defines the layer that stays stable so that in two years you can still call these models from anything — a notebook, a script, a different UI, someone else's project — without reverse-engineering how they were trained.

## Three layers, and only one of them is durable

```
┌─────────────────────────────────────────────────┐
│  UI — disposable                                │
│  One shell, three renderers, four backends      │
├─────────────────────────────────────────────────┤
│  Service — thin, ~30 lines, also disposable     │
│  FastAPI wrapper. Zero logic.                   │
├─────────────────────────────────────────────────┤
│  Core — DURABLE. This is the real deliverable.  │
│  src/inference.py + model artifact + manifest   │
└─────────────────────────────────────────────────┘
```

**The rule that keeps the core durable:** `src/inference.py` imports only its ML dependencies — torch, sklearn, rdkit, transformers. It never imports FastAPI, never touches HTTP, never reads request objects. It's a function that takes bytes or a string and returns a dict. If the web framework of 2028 is something nobody's heard of yet, you write a new 30-line wrapper and the core is untouched.

## The core function

Every project exposes exactly this, in `src/inference.py`:

```python
def predict_one(
    payload: bytes | str,          # image bytes, or SMILES/sentence string
    variant: str | None = None,    # which trained model; None = shipped default
) -> dict:                         # conforms to the contract schema below
    ...

def health() -> dict:              # callable without running inference
    ...
```

Two functions. No classes required, no framework, no globals beyond a lazily-loaded model cache.

## The contract schema

One JSON shape for all four models. Versioned, because you will change it.

```json
{
  "contract_version": "1.0",

  "model": {
    "id": "brain-mri",
    "name": "Brain MRI Tumor Classifier",
    "variant": "resnet18-transfer",
    "variants_available": ["cnn-baseline", "resnet18-transfer"],
    "status": "trained",
    "trained_at": "2026-08-01T00:00:00Z",
    "artifact_sha256": "…"
  },

  "input": {
    "modality": "image",
    "echo": {
      "kind": "png_b64",
      "data": "…",
      "note": "post-preprocessing 224x224, what the model actually saw"
    }
  },

  "output_mode": "single_label",
  "predicted_label": "glioma",

  "calibrated": true,
  "calibration": {
    "method": "temperature",
    "temperature": 1.32413,
    "logit_offsets": { "glioma": 2.1 }
  },

  "predictions": [
    {
      "label": "glioma",
      "index": 0,
      "prob": 0.83,
      "prob_raw": 0.94,
      "base_rate": null,
      "flags": []
    }
  ],

  "confidence_band": "review_recommended",

  "explanation": {
    "type": "heatmap",
    "payload": { }
  },

  "caveats": [
    {
      "scope": "class",
      "target": "notumor",
      "severity": "warning",
      "text": "Test performance unreliable pending leakage rebuild — 86.8% of this class's test images duplicated training data."
    }
  ],

  "timing_ms": 143
}
```

### Field notes that matter

**`prob_raw` is required, not optional.** It's the uncalibrated softmax. The UI's calibration ghost depends on it, and more importantly it's the honest record of how overconfident the model was before correction. Never omit it, never let it equal `prob` unless calibration genuinely was a no-op.

**`base_rate` is `null` for single-label models and required for multi-label.** A 0.3 on a rare Tox21 assay means something completely different from a 0.3 on a common one; without the base rate the number is uninterpretable.

**`predicted_label` is `null` when `output_mode` is `multi_label`.** Tox21 has no single prediction. Anything that tries to pick a "top" assay is misrepresenting the model.

**`flags` on a prediction row vs. `caveats` with `scope: "class"`.** Both exist because a compromised class needs to be visible *on its own row* and explained *in the evidence panel*. Same information, two placements, both necessary.

**`calibration` is echoed on every response**, not just documented. If the calibration file ever fails to load, the response should say `"calibrated": false` loudly rather than silently returning raw softmax.

### Output modes

| Mode | Used by | `predictions` contains | Sums to 1 |
|---|---|---|---|
| `single_label` | Brain MRI, Pill ID | One row per class | Yes |
| `multi_label` | Tox21 | 12 rows, one per assay | **No** |
| `binary` | ADR | One row (positive class) | N/A |

### Explanation payloads

```json
// type: "heatmap"  — Brain MRI, Pill ID
{ "overlay_png_b64": "…", "raw_png_b64": "…",
  "resolution_note": "7x7 upsampled to 224 — shows influence, not boundaries" }

// type: "token_attribution"  — ADR
{ "tokens": [ {"text": "nausea", "score": 0.42}, {"text": "following", "score": 0.03} ] }

// type: "substructure"  — Tox21 (stretch; ship "none" until built)
{ "atom_scores": [0.1, 0.7, …], "structure_svg_b64": "…" }

// type: "none"
{ "reason": "Substructure attribution not implemented for the RF baseline." }
```

`"none"` is a first-class state, not a failure. The Tox21 fingerprint baseline may ship without attribution, and the UI should say so plainly rather than showing an empty panel.

## The service layer

Each project runs its own process on a fixed port:

| Model | Port |
|---|---|
| brain-mri | 8001 |
| pill-id | 8002 |
| tox21 | 8003 |
| adr | 8004 |

Three endpoints each:

- `GET /health` → model block + `contract_version`, **without loading the model or running inference**. This is what lets the UI's model rack show `not_built` and `baseline_only` states without a failed request and a spinner.
- `POST /predict` → the contract object. Accepts multipart for images, JSON for text.
- `GET /manifest` → the manifest below.

The wrapper is genuinely thin — parse the request, call `predict_one`, return the dict. If a service file grows past about 50 lines, logic has leaked out of the core and should be pushed back down.

## The manifest — this is what makes the models reusable

Ships next to the weights, committed to the repo (it's small):

```json
{
  "model_id": "brain-mri",
  "variant": "resnet18-transfer",
  "artifact": "outputs/checkpoints/resnet18_transfer.pth",
  "artifact_sha256": "…",
  "artifact_size_mb": 44.8,

  "class_order": ["glioma", "meningioma", "notumor", "pituitary"],
  "class_order_note": "Index-sensitive. Do not reorder — checkpoint outputs map positionally.",

  "input_spec": {
    "modality": "image",
    "accepts": ["JPEG", "PNG"],
    "channels_in": ["L", "RGB", "RGBA"],
    "preprocessing": [
      "convert to RGB (3-channel)",
      "resize short side to 224 (bicubic)",
      "center-crop 224x224",
      "normalize mean (0.485,0.456,0.406) std (0.229,0.224,0.225)"
    ],
    "preprocessing_note": "Reproduce exactly. Getting this wrong degrades accuracy silently, with no error raised."
  },

  "calibration_file": "outputs/calibration_transfer.json",

  "metrics": {
    "macro_f1": 0.898,
    "baseline_macro_f1": 0.756,
    "trivial_baseline_macro_f1": null,
    "per_class": { "glioma": { "recall": 0.834 } },
    "seeds": 1,
    "eval_note": "Leakage-clean split."
  },

  "caveats": [
    "Not a diagnostic device.",
    "Glioma recall 0.834 — roughly 1 in 6 missed, some misclassified as notumor.",
    "notumor class unreliable pending leakage rebuild."
  ]
}
```

Write one of these per **variant**, not per project — the from-scratch CNN and the ResNet18 have different metrics and different calibration.

## Golden-output test — the thing that actually saves you

Each project commits one fixture input and its expected output:

```
tests/
├── fixtures/sample_input.png
└── test_golden.py     # predict_one(fixture) matches saved probs to 1e-4
```

This is the highest-value 20 lines in the whole repo. In eighteen months you'll rebuild an environment, silently get a different `torchvision` resize implementation or a different default interpolation, and your model will quietly get worse with no error raised. The golden test catches that on the spot. Without it, you'll never know.

## Frontend structure

```
inspector/
├── src/
│   ├── contract/types.ts        # mirrors the schema above — single source of truth
│   ├── shell/                   # rack, specimen zone, evidence panel, design tokens
│   ├── renderers/
│   │   ├── SingleLabel.tsx      # brain-mri AND pill-id — same output mode
│   │   ├── MultiLabel.tsx       # tox21
│   │   └── Binary.tsx           # adr
│   ├── explanations/
│   │   ├── Heatmap.tsx
│   │   ├── TokenAttribution.tsx
│   │   └── Substructure.tsx
│   └── services/endpoints.ts    # port map, health polling
```

Three renderers, not four — MRI and Pill share one, which is the concrete reason not to build four separate apps. Each renderer folder is self-contained and depends only on `contract/types.ts`, so any one of them can be lifted into a standalone app later without untangling it from the others.

## Build order

1. **`src/inference.py` + manifest + golden test for brain-mri.** It's the only trained model; it defines the contract in practice rather than in theory.
2. **Service wrapper for brain-mri.** Confirm the contract survives a round trip over HTTP.
3. **Frontend shell + `SingleLabel` renderer.** One working end-to-end path.
4. **The other three projects write `inference.py` against this contract from the start** rather than being retrofitted — that's the main reason to settle this before they're built.
5. `MultiLabel` and `Binary` renderers as those models come online.

## Contract changes

Bump `contract_version` on any breaking change. The UI reads it and refuses to render a version it doesn't recognize, rather than silently mis-parsing a field that moved. Additive fields don't need a bump; renamed, removed, or re-typed fields do.
