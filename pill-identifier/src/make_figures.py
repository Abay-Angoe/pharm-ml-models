"""Build outputs/figures/summary_comparison.png from all seed runs.

Four panels:
  1. validation curves, every seed overlaid
  2. test-set top-1 (headline), top-3 (secondary) and macro-F1, mean +/- std
  3. accuracy by localisation method - how much of the gap is the pill detector
     rather than the model
  4. per-class F1 against pill colour saturation - the colour-shortcut test

Panel titles state what was measured, not a hoped-for conclusion: on this dataset
the colour-shortcut test came back negative, and the title says so.

    python src/make_figures.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gradcam import pill_mask_for_crop, spearman  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
B, T = "#d9480f", "#1864ab"
PHASES = ("baseline", "transfer")


def mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float("nan"), 0.0
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def note_val(notes: str, key: str) -> float | None:
    m = re.search(rf"{key}=([-\d.]+)", notes or "")
    return float(m.group(1)) if m else None


def saturation_per_class(classes: list[str]) -> dict[str, float]:
    """Mean HSV saturation inside the estimated pill mask, per class.

    Measured inside the pill only - averaging the whole crop would mix in
    countertop colour and wash out the signal being tested.
    """
    sat = {}
    for cls in classes:
        vals = []
        for p in sorted((ROOT / "data/processed/test" / cls).glob("*.png"))[:8]:
            rgb = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
            mask = pill_mask_for_crop(rgb)
            if mask is None:
                continue
            hsv = np.asarray(Image.open(p).convert("HSV")).astype(float)
            vals.append(hsv[:, :, 1][mask > 0.5].mean() / 255)
        if vals:
            sat[cls] = float(np.mean(vals))
    return sat


def main() -> int:
    rows = list(csv.DictReader(open(OUTDIR / "metrics.csv", encoding="utf8")))
    macro = [r for r in rows if r["class"] == "__MACRO_AVG__"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax1, ax2, ax3, ax4 = axes.ravel()
    w = 0.36

    # --- panel 1: validation curves ---------------------------------------
    for phase, col in zip(PHASES, (B, T)):
        for i, f in enumerate(sorted(OUTDIR.glob(f"history_{phase}*.csv"))):
            h = list(csv.DictReader(open(f, encoding="utf8")))
            ep = [int(r["epoch"]) for r in h]
            ax1.plot(ep, [float(r["val_top1"]) for r in h], color=col, alpha=0.8,
                     lw=1.2, label=f"{phase} top-1" if i == 0 else None)
            ax1.plot(ep, [float(r["val_top3"]) for r in h], color=col, ls="--",
                     alpha=0.4, lw=1.0,
                     label=f"{phase} top-3" if i == 0 else None)
    ax1.axvline(10.5, color="gray", ls=":", lw=1)
    ax1.annotate("backbone unfrozen", (10.9, 15), fontsize=7, color="gray")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("val accuracy (%)")
    ax1.set_title("Validation accuracy (3 seeds per phase)")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(alpha=0.25)

    # --- panel 2: test results -------------------------------------------
    def agg(phase: str, key: str, scale: float = 1.0) -> tuple[float, float]:
        return mean_std([float(r[key]) * scale for r in macro
                         if r["phase"] == phase and r.get(key)])

    def agg_note(phase: str, key: str) -> tuple[float, float]:
        return mean_std([v for r in macro if r["phase"] == phase
                         for v in [note_val(r["notes"], key)] if v is not None])

    stats = {p: [agg(p, "accuracy_overall"), agg_note(p, "top3"),
                 agg(p, "macro_f1", 100)] for p in PHASES}
    x = np.arange(3)
    for phase, col, off, lab in ((PHASES[0], B, -w / 2, "Phase 1 baseline"),
                                 (PHASES[1], T, w / 2, "Phase 2 transfer")):
        ax2.bar(x + off, [m for m, _ in stats[phase]], w,
                yerr=[s for _, s in stats[phase]], capsize=4, color=col, label=lab)
        for xi, (m, s) in zip(x + off, stats[phase]):
            ax2.text(xi, m + s + 2.5, f"{m:.1f}", ha="center", fontsize=8)
    n_cls = len({r["class"] for r in rows if not r["class"].startswith("__")}) or 20
    triv = [float(r["accuracy_overall"]) for r in macro
            if r["phase"] == "trivial_majority"]
    if triv:
        ax2.axhline(triv[0], color="k", ls=":", lw=1)
        ax2.annotate(f"majority-class trivial baseline ({triv[0]:.1f}%)",
                     (0.55, triv[0] + 3), fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["top-1\n(headline)", "top-3\n(secondary)", "macro F1 x100"])
    ax2.set_ylim(0, 115)
    ax2.set_title(f"Test results, mean +/- std over 3 seeds ({n_cls} classes)")
    ax2.legend(fontsize=8, loc="upper left")

    # --- panel 3: localisation method ------------------------------------
    det = {p: agg_note(p, "det") for p in PHASES}
    fb = {p: agg_note(p, "fb") for p in PHASES}
    xx = np.arange(2)
    for phase, col, off, lab in ((PHASES[0], B, -w / 2, "Phase 1 baseline"),
                                 (PHASES[1], T, w / 2, "Phase 2 transfer")):
        means = [det[phase][0], fb[phase][0]]
        stds = [det[phase][1], fb[phase][1]]
        ax3.bar(xx + off, means, w, yerr=stds, capsize=4, color=col, label=lab)
        for xi, m, s in zip(xx + off, means, stds):
            ax3.text(xi, m + s + 2, f"{m:.1f}", ha="center", fontsize=8)
        ax3.text(0.5, means[1] - 8 if phase == PHASES[0] else means[1] + 6,
                 f"{means[1] - means[0]:+.1f}pp", color=col, fontsize=10,
                 ha="center", fontweight="bold")
    ax3.set_xticks(xx)
    ax3.set_xticklabels(["pill detected\n(n=84)", "centre fallback\n(n=100)"])
    ax3.set_ylabel("test top-1 (%)")
    ax3.set_ylim(0, 110)
    ax3.set_title("Weak localisation costs the scratch CNN ~4x more\n"
                  "than the pretrained model")
    ax3.legend(fontsize=8, loc="lower left")

    # --- panel 4: colour-shortcut test -----------------------------------
    classes = sorted({r["class"] for r in rows if not r["class"].startswith("__")})
    sat = saturation_per_class(classes)
    rhos: dict[str, float] = {}
    for phase, col, marker in zip(PHASES, (B, T), ("o", "^")):
        xs, ys, errs = [], [], []
        for cls in classes:
            f1 = [float(r["f1"]) for r in rows
                  if r["class"] == cls and r["f1"] and r["phase"] == phase]
            if not f1 or cls not in sat:
                continue
            m, s = mean_std(f1)
            xs.append(sat[cls])
            ys.append(m)
            errs.append(s)
        rho = spearman(np.asarray(xs), np.asarray(ys))
        rhos[phase] = rho
        ax4.errorbar(xs, ys, yerr=errs, fmt=marker, color=col, ms=6, alpha=0.8,
                     capsize=2, lw=0.8, ls="none", label=f"{phase}  rho={rho:+.2f}")
    ax4.set_xlabel("pill colour saturation (inside pill)")
    ax4.set_ylabel("per-class F1 (mean of 3 seeds)")
    ax4.set_title("Colour-shortcut test: NEGATIVE on this dataset\n"
                  f"(no positive rho - baseline {rhos.get('baseline', float('nan')):+.2f}, "
                  f"transfer {rhos.get('transfer', float('nan')):+.2f})")
    ax4.legend(fontsize=8, loc="lower right")
    ax4.grid(alpha=0.25)

    fig.tight_layout()
    out = OUTDIR / "figures" / "summary_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
