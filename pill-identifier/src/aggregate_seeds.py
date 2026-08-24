"""Aggregate results across seeds into mean +/- std.

Section 0: datasets here are small enough that a 2-point gap can be seed noise,
so every headline number is reported as mean +/- std over 3 seeds.

Reads outputs/metrics.csv in the section 0 schema
    phase, seed, class, support, precision, recall, f1, accuracy_overall,
    macro_f1, notes
(so this is a groupby, not a manual merge) plus outputs/gradcam_analysis_*.json.
Writes outputs/summary_seeds.csv.

Top-1 is the headline. Top-3 is carried in the `notes` field and shown as a
clearly-labelled secondary signal. Top-5 is not reported anywhere (section 7).

    python src/aggregate_seeds.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "outputs" / "metrics.csv"
OUT = ROOT / "outputs" / "summary_seeds.csv"


def ms(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def note_field(notes: str, key: str) -> float | None:
    """Pull `key=<number>` out of the semicolon-delimited notes column."""
    m = re.search(rf"{key}=([-\d.]+)", notes or "")
    return float(m.group(1)) if m else None


def fmt(m: float, s: float, n: int) -> str:
    return f"{m:8.2f} +/-{s:5.2f}" if n > 1 else f"{m:8.2f}{'':>8}"


def main() -> int:
    if not METRICS.exists():
        print(f"{METRICS} missing - run src/evaluate.py first")
        return 1
    rows = list(csv.DictReader(open(METRICS, encoding="utf8")))
    if "phase" not in (rows[0] if rows else {}):
        print("metrics.csv is in the old schema; delete it and re-run evaluate.py")
        return 1

    # ---- one value per (phase, seed) from the __MACRO_AVG__ rows ----------
    runs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seeds: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["class"] != "__MACRO_AVG__":
            continue
        p = r["phase"]
        runs[p]["top1"].append(float(r["accuracy_overall"]))
        if r["macro_f1"]:
            runs[p]["macro_f1"].append(float(r["macro_f1"]) * 100)
        t3 = note_field(r["notes"], "top3")
        if t3 is not None:
            runs[p]["top3"].append(t3)
        seeds[p].add(r["seed"])

    order = [p for p in ("trivial_majority", "trivial_stratified_random",
                         "baseline", "transfer") if p in runs]
    order += [p for p in runs if p not in order]

    print("=== test-set results, mean +/- std across seeds ===")
    print("    top-1 is the headline; top-3 is a secondary "
          "'close enough for a lookup tool' signal")
    print("    top-5 deliberately not reported (17-33% by chance at this "
          "class count)\n")
    hdr = (f"{'phase':28s} {'n':>2} {'top-1 (headline)':>17} "
           f"{'macro F1 x100':>17} {'top-3 (secondary)':>18}")
    print(hdr)
    print("-" * len(hdr))
    summary = []
    for p in order:
        d = runs[p]
        n = len(d["top1"])
        m1, s1 = ms(d["top1"])
        mf, sf = ms(d["macro_f1"]) if d["macro_f1"] else (float("nan"), 0.0)
        m3, s3 = ms(d["top3"]) if d["top3"] else (float("nan"), 0.0)
        print(f"{p:28s} {n:>2} {fmt(m1, s1, n)} "
              f"{fmt(mf, sf, n) if d['macro_f1'] else f'{chr(45):>16}'} "
              f"{fmt(m3, s3, n) if d['top3'] else f'{chr(45):>17}'}")
        summary.append({
            "phase": p, "n_seeds": n, "seeds": "/".join(sorted(seeds[p])),
            "top1_mean": f"{m1:.2f}", "top1_std": f"{s1:.2f}",
            "macro_f1_mean": f"{mf:.2f}", "macro_f1_std": f"{sf:.2f}",
            "top3_mean": f"{m3:.2f}", "top3_std": f"{s3:.2f}",
        })

    if "baseline" in runs and "transfer" in runs:
        b, bs = ms(runs["baseline"]["top1"])
        t, ts = ms(runs["transfer"]["top1"])
        gap = t - b
        pooled = float(np.hypot(bs, ts)) or 1e-9
        print(f"\ntransfer - baseline (top-1): {gap:+.2f}pp")
        print(f"  seed std: baseline {bs:.2f}, transfer {ts:.2f} "
              f"-> combined {pooled:.2f}")
        if gap > 2 * pooled:
            print(f"  gap = {gap / pooled:.1f}x combined std -> real, not seed noise")
        else:
            print(f"  gap = {gap / pooled:.1f}x combined std -> NOT clearly beyond "
                  f"seed noise; report as inconclusive (section 6 allows "
                  f"'the baseline held up' as a finding)")
        for name in ("trivial_majority", "trivial_stratified_random"):
            if name in runs:
                tv, _ = ms(runs[name]["top1"])
                print(f"  vs {name}: baseline {b - tv:+.1f}pp, "
                      f"transfer {t - tv:+.1f}pp above trivial")

    # ---- per-class F1 across seeds ---------------------------------------
    cls_f1: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["class"].startswith("__") or not r["f1"]:
            continue
        cls_f1[(r["phase"], r["class"])].append(float(r["f1"]))
    if cls_f1 and "baseline" in runs and "transfer" in runs:
        classes = sorted({c for _, c in cls_f1})
        deltas = []
        for c in classes:
            b = cls_f1.get(("baseline", c), [])
            t = cls_f1.get(("transfer", c), [])
            if b and t:
                deltas.append((ms(t)[0] - ms(b)[0], c, ms(b)[0], ms(t)[0]))
        deltas.sort(reverse=True)
        print("\n=== biggest per-class gains from transfer learning ===")
        for d, c, bm, tm in deltas[:6]:
            print(f"  {d:+.2f}  {c:34s} ({bm:.2f} -> {tm:.2f})")
        regress = [x for x in deltas if x[0] < 0]
        print(f"=== classes transfer learning HURT ({len(regress)}) ===")
        for d, c, bm, tm in regress[-6:]:
            print(f"  {d:+.2f}  {c:34s} ({bm:.2f} -> {tm:.2f})")

    # ---- grad-cam attention measures across seeds ------------------------
    cam: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted((ROOT / "outputs").glob("gradcam_analysis_*.json")):
        tag = f.stem.replace("gradcam_analysis_", "")
        m = re.match(r"^(baseline|transfer)", tag)
        if not m:
            continue
        data = json.loads(f.read_text())
        for subset, vals in (data.get("summary") or {}).items():
            for key in ("on_pill", "interior", "imprint_corr"):
                v = vals.get(key)
                if v is not None and np.isfinite(v):
                    cam[m.group(1)][f"{subset}|{key}"].append(v)
    if cam:
        phases = [p for p in ("baseline", "transfer") if p in cam]
        print("\n=== Grad-CAM attention, mean +/- std across seeds ===")
        print(f"{'measure':38s} " + " ".join(f"{p:>18}" for p in phases))
        for k in ("ALL|on_pill", "ALL|interior", "ALL|imprint_corr",
                  "side-lit (engraved)|imprint_corr",
                  "diffuse (washed out)|imprint_corr",
                  "correct predictions|imprint_corr",
                  "incorrect predictions|imprint_corr"):
            cells = []
            for p in phases:
                v = cam[p].get(k, [])
                if not v:
                    cells.append(f"{'-':>18}")
                else:
                    m, s = ms(v)
                    cells.append(f"{m:+9.3f} +/-{s:5.3f}" if len(v) > 1
                                 else f"{m:+9.3f}{'':>9}")
            print(f"{k:38s} " + " ".join(cells))

    if summary:
        with open(OUT, "w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
