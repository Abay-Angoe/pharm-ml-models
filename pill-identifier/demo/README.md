# Demo — upload a pill photo, get top-3 or an honest refusal

Implements the §10 stretch goal ("upload a photo, get top-3 predictions with
confidence"). **Not for clinical use.**

```bash
# from the project root, with .venv active
pip install fastapi uvicorn python-multipart
python -m uvicorn demo.app:app --port 8000
# open http://127.0.0.1:8000
```

Inference is ~200 ms on CPU. Serves `outputs/checkpoints/transfer_s1_best.pt`,
selected on **validation** top-1 (95.2%), never on test.

---

## What it does

`POST /api/identify` (multipart image) returns:

| field | why it is there |
|---|---|
| `crop_png_b64` | **the 224px region actually classified** |
| `top3` | label, confidence, look-alike group |
| `rejected` | true when top confidence < the measured threshold |
| `prose` | templated sentences, never generated drug facts |
| `localisation.method` | `detected` or `center_fallback` |

`GET /api/info` returns the class list, look-alike groups, held-out accuracy and
the rejection numbers, so the UI never hardcodes a claim.

## Three design decisions

**1. Preprocessing is imported, not reimplemented.** `demo/app.py` calls
`src/localize.py` and `src/dataset.build_transforms(train=False)` directly. If serving
preprocessing drifts from training preprocessing the model quietly degrades — and a
naively resized full frame puts the pill at ~35 px, where accuracy collapses.

**2. The crop is always shown.** The localiser finds the pill ~57% of the time and
centre-crops the rest. When it crops the wrong thing — during development it once
locked onto a coin used as a size reference — the prediction is meaningless. Returning
the classified crop makes that visible instead of invisible. The UI additionally warns
when the fallback was used.

**3. The rejection threshold is read from a measurement, and the app refuses to start
without it.** `demo/app.py` loads `outputs/ood_rejection.json`; if it is missing it
raises rather than inventing a confidence cutoff.

## The rejection threshold, measured

`src/ood_rejection.py` downloads ~100 C3PI photographs of **20 products outside the 20
classes**, runs them through the identical pipeline, and compares max-softmax against
the in-distribution validation split.

The headline problem: out-of-set images have a median confidence of **0.637**, and
in-distribution *wrong* predictions sit at **0.639**. Max-softmax cannot tell "a pill I
have never seen" from "a pill I got wrong" — both are just confusion.

| threshold | coverage | precision | out-of-set still named |
|---|---|---|---|
| 0.50 | 97.1% | 95.6% | 72.0% |
| 0.70 | 85.6% | 98.3% | 44.0% |
| **0.828** ← used | **76.6%** | **99.4%** | **24.0%** |
| 0.90 | 67.0% | 100.0% | 15.0% |
| 0.99 | 23.4% | 100.0% | 6.0% |

Chosen rule: *lowest threshold reaching 99% in-distribution precision*, decided on
validation. For a pill identifier, naming the wrong drug is far worse than declining,
so precision is the constraint and coverage is what gets spent.

**The residual risk is real and stated in the UI: about 1 in 4 pills from outside the
20 classes still gets confidently named.** A naive "keep 95% of correct answers"
threshold would have been 0.560, where **61%** of out-of-set pills get named — which is
why this was measured rather than guessed.

## Verified behaviour

| case | input | result |
|---|---|---|
| in-set, well localised | GABITRIL 16 mg | 99.0% correct, accepted |
| out-of-set product | unseen C3PI pill | best guess 80.5% → **declined** |
| look-alike ambiguity | LEVOXYL 0.025 mg | 41% vs 34% between the two strengths → declined, and the prose explains it is likely the right product but the wrong strength |
| malformed upload | text file | HTTP 400, `could not decode image` |
| no file | — | HTTP 422 |

## Why the prose is templated

It is built only from labels, confidences and group structure. It cannot state what a
drug does or how to take it, because the model has no such knowledge and a fluent
wrong answer is precisely the failure mode that matters here.

The one case where it says something genuinely useful: when ≥2 of the top-3 are the
same product at different strengths, it says so and tells the user to check the imprint
against the pack — because §4 established that **this model does not read imprints**,
so it cannot be trusted on exactly that distinction.

## Limits, restated

* **20 products** — 0.8% of the C3PI products that have consumer photos.
* **Closed set** — anything else still maps onto one of the 20.
* **Does not read imprints** — cannot reliably separate two white tablets that differ
  only by their marking.
* **Single pill, roughly centred** — the localiser expects one pill, not a handful.
* Held-out accuracy **90.6 ± 2.5% top-1**, and that is on curated C3PI photographs of
  well-photographed products, not arbitrary phone snapshots.
