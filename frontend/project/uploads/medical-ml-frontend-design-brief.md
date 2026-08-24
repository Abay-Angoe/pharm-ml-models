# Design Brief — Medical ML Model Inspector

## The product in one line
A local inspection surface for four medical/pharmaceutical classifiers, where every prediction arrives with its explanation, its calibration, and its known failure modes attached.

**Audience:** the person who trained these models, plus anyone they show them to. Technically literate. Not clinicians, and not patients.

**The page's single job:** let someone submit one input, see what the model predicted, see *why*, and understand how much to trust it — in that order of prominence.

## The through-line

Every design decision should serve one idea: **this tool is honest about what it doesn't know.**

That isn't a disclaimer bolted on at the bottom. It's the product. These models have measured, specific weaknesses — the MRI classifier misses roughly 1 in 6 gliomas, some of them landing on "no tumor"; its `notumor` class is untrustworthy pending a data-leakage rebuild; its raw softmax was so overconfident it produced 20 errors at p>0.99 before temperature calibration cut that to 5. A UI that renders a big green "GLIOMA — 94%" is actively lying about all of that.

So the confidence number is deliberately **not** the hero. The input and the explanation are. The number is small, precise, and always shown next to what it used to be before calibration.

## Signature element: the calibration ghost

Every probability bar renders twice: the **calibrated** value as a solid bar, and the **raw, uncalibrated** value as a faint outline behind it. When the model was overconfident, you see the ghost extending past the solid bar — the correction made visible, on every single prediction.

This is the one thing the app should be remembered by. It's specific to these models (the MRI classifier divides logits by T = 1.32413 and adds +2.1 to the glioma logit), it earns its place by teaching something true, and it can't be lifted onto another product. Spend the boldness here and keep everything else quiet.

## Layout: three zones

```
┌──────────────────────────────────────────────────────────┐
│  MODEL RACK  (persistent, 4 slots, each showing status)  │
├───────────────────────────────┬──────────────────────────┤
│                               │  THE READING             │
│  THE SPECIMEN                 │                          │
│  (dark viewer for images,     │  predicted label         │
│   mono editor for SMILES      │  calibrated probs        │
│   and sentences)              │  + calibration ghosts    │
│                               │  confidence band         │
│  explanation overlay toggle   │                          │
│  / opacity slider             ├──────────────────────────┤
│                               │  THE EVIDENCE            │
│                               │  (model card — peer      │
│                               │   panel, not a tooltip)  │
└───────────────────────────────┴──────────────────────────┘
```

**The specimen** is a dark viewer — this is where images live, and radiology viewers are dark for a reason. Text inputs (SMILES, sentences) sit in the same zone in a mono editor, keeping the spatial logic consistent across modalities.

**The reading** is the prediction. Compact, tabular, quiet.

**The evidence** is the model card, and it is a *peer panel*, not a hidden disclosure. Putting known failure modes at the same visual level as the prediction is the structural expression of the through-line. It carries: performance vs. the trivial baseline, mean ± std across seeds, per-class weak spots, and the caveats that apply to this specific prediction.

## The model rack

Four models, but they are **not a sequence** — don't number them 01/02/03/04. They're instruments on a rack, selected independently. Each slot shows its own status, because three of the four aren't trained yet:

- `trained` — selectable, shows shipped model name
- `baseline only` — Phase 1 exists, Phase 2 doesn't
- `not built` — visible but inert; the rack should show the whole ambition, not hide the gaps

Where two models exist for one slot, let the user switch between them and see both readings. For Tox21 in particular, the shipped model may end up being the *simpler* one — a fingerprint + random forest can genuinely beat a graph network at this data scale — so the UI must present "the simple model won" as a normal, respectable state, not an error or a downgrade.

## Four result renderers, one shell

The shell (specimen / reading / evidence) is constant. What renders inside "the reading" varies by output shape:

**Single-label, few classes — Brain MRI.** Four probability bars with calibration ghosts, ordered by the fixed class index `["glioma", "meningioma", "notumor", "pituitary"]`. Grad-CAM overlay in the specimen zone with an opacity slider against the raw scan. A per-class warning affordance is required, not optional: `notumor` needs a visible flag on its own row, because that class's test performance is compromised by training-data duplication. A model-level caveat can't carry that — it's specific to one row.

**Single-label, many classes — Pill ID.** Same bars, but with 15–30 classes you can't show them all. Show top-3 plus a "most confused with" affordance surfacing the pairs the confusion matrix flagged — same-color, same-shape pills are where this model will fail, and that's the interesting result. The Grad-CAM here answers a specific question: *is the model reading the imprint, or just color and shape?* Consider making that question explicit in the UI rather than leaving the user to interpret the heatmap unaided.

**Multi-label, independent — Tox21.** This is the renderer that breaks if you reuse the softmax component. Twelve assay endpoints, each an independent binary probability. **They do not sum to 1, and the design must not imply they do** — no stacked bar, no pie, no "top prediction." Twelve rows, each showing its own probability against its own base rate, because a 0.3 on a rare assay means something very different from a 0.3 on a common one. Input is a SMILES string in a mono field; if the molecule renders as a 2D structure, that's the specimen.

**Binary — ADR.** One probability, so bars are degenerate. Lead with the sentence itself, with token attribution inline — highlight the words that drove the classification. The specimen *is* the result here. Recall on the positive class is the metric that matters (a missed adverse reaction is worse than a false positive), so the evidence panel should foreground recall rather than accuracy.

## Confidence bands

Post-calibration, low confidence genuinely correlates with being wrong, so "review recommended" is a defensible affordance — it's earned, not decorative.

**Do not use a green-to-red traffic light.** Green reads as "you're fine," and this is explicitly not a diagnostic device; there is no state where the correct message is "you're fine." Use a neutral-to-amber scale instead: confident predictions are *quiet and neutral*, low-confidence ones warm toward amber. The absence of alarm is the positive state. Nothing in this UI should ever be green.

## Visual direction

**Character:** scientific instrument, not consumer health app. The reference points are radiology viewers, lab notebooks, and assay plate readouts — not wellness dashboards. Information-dense, closer to a notebook than a landing page.

**Palette** — cool base, warm paper, one cool accent:

- `--film: #101315` — the specimen viewer surface
- `--slate: #232A2E` — chrome, rack, controls
- `--stock: #E6E7E1` — archival card stock; the evidence panels, deliberately grey-green rather than cream
- `--ink: #171A1C` — text on stock
- `--signal: #2E7D74` — desaturated teal, the single interactive accent
- `--review: #B4802A` — amber, confidence band only

The accent is pushed to the cool end for a functional reason: Grad-CAM renders in a hot colormap (reds and oranges), and a warm accent would compete with the actual data. Keep the interface cool so the heatmap is the only warm thing on screen.

**Typography** — three roles, and the data role is not optional:

- *Display:* a technical grotesque with some width and personality, used sparingly — model names, zone headers. Avoid high-contrast editorial serifs; this is an instrument, not a magazine.
- *Body:* a humanist sans workhorse for prose, caveats, and labels.
- *Data:* a true monospace with **tabular figures**, mandatory. It carries SMILES strings, class indices, model IDs, probability columns, and metric tables. Probabilities that shift horizontally as digits change will make the whole thing feel amateur — lock the figures.

**Motion:** minimal and functional. One place it earns its keep: when a prediction resolves, let the calibration ghost draw first and the calibrated bar settle into it. That single orchestrated moment teaches the core concept better than any label. Everything else static. Respect reduced-motion.

## States to design

- **Not built** — three of four models. An invitation, not an error: what this slot will do once trained.
- **Baseline only** — Phase 1 exists, Phase 2 doesn't. Show which model is speaking.
- **Backend unreachable** — each model runs as its own local service; they will not all be running at once. Say which one is down and how to start it. Errors don't apologize and are never vague.
- **Low confidence / review recommended** — the amber band, with the reason.
- **Compromised class** — the `notumor` case. A row-level flag, distinct from model-level caveats.
- **Empty** — no input submitted yet. An invitation to act.

## Non-negotiables

1. **No diagnostic framing anywhere.** Not in copy, not in iconography, not in color. No checkmarks, no "clear," no stethoscopes or crosses.
2. **Never show a raw softmax number.** Calibration is applied before any confidence value is displayed.
3. **Grad-CAM shows influence, not boundaries.** It's a coarse 7×7 upsample. The UI must not render it in a way that suggests segmentation or tumor margins.
4. **Caveats are structural, not footnotes.** The evidence panel is a peer of the reading, always visible.
5. **The simple model winning is a normal state**, presented without apology.

## Out of scope

Batch processing, training runs, dataset browsing, user accounts, anything persisted server-side, mobile-first layouts (responsive down is fine; this is a desktop instrument).
