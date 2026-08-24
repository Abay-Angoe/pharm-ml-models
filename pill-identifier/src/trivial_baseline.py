"""Trivial predictors, computed before trusting any learned model.

Repo convention: every project ships majority-class and stratified-random
baselines so that a headline accuracy can never be mistaken for a class-imbalance
artefact.

If the classes are balanced both trivial predictors land on chance (1/N top-1).
Confirming that is the point: it proves the learned numbers are real signal, and
documents whether class weighting was needed.

    python src/trivial_baseline.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import PillDataset  # noqa: E402
from evaluate import SCHEMA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METRICS_CSV = ROOT / "outputs" / "metrics.csv"
N_REPEATS = 1000


def topk_from_scores(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    topk = np.argsort(-scores, axis=1)[:, :k]
    return 100.0 * float((topk == targets[:, None]).any(axis=1).mean())


def main() -> int:
    train = PillDataset("train", train_aug=False)
    test = PillDataset("test", train_aug=False, classes=train.classes)
    classes = train.classes
    n_cls = len(classes)

    train_labels = np.array([lbl for _, lbl, _ in train.samples])
    test_labels = np.array([lbl for _, lbl, _ in test.samples])
    counts = np.bincount(train_labels, minlength=n_cls)
    prior = counts / counts.sum()

    print(f"classes {n_cls} | train {len(train_labels)} | test {len(test_labels)}")
    print(f"train per-class counts: {sorted(set(counts.tolist()))}")
    print(f"test  per-class counts: "
          f"{sorted(set(Counter(test_labels.tolist()).values()))}")
    print(f"=> classes are {'EXACTLY balanced' if len(set(counts.tolist())) == 1 else 'UNEVEN'}"
          f"; chance top-1 = {100 / n_cls:.2f}%, top-3 = {300 / n_cls:.2f}%\n")

    results: dict[str, dict[str, float]] = {}

    # 1. majority class - deterministic. With ties, np.argmax takes the first.
    majority = int(np.argmax(counts))
    maj_top1 = 100.0 * float((test_labels == majority).mean())
    # a constant predictor has one real "guess"; score top-3 by ranking the prior
    scores = np.tile(prior, (len(test_labels), 1))
    results["trivial_majority"] = {
        "top1": maj_top1,
        "top3": topk_from_scores(scores, test_labels, 3),
    }
    print(f"majority-class (always '{classes[majority]}'):")
    print(f"  top-1 {maj_top1:5.2f}%   top-3 "
          f"{results['trivial_majority']['top3']:5.2f}% (prior-ranked, secondary)")

    # 2. stratified random - sample from the training prior, averaged over repeats
    rng = np.random.default_rng(0)
    t1, t3 = [], []
    for _ in range(N_REPEATS):
        draw = rng.choice(n_cls, size=len(test_labels), p=prior)
        t1.append(100.0 * float((draw == test_labels).mean()))
        rand_scores = rng.random((len(test_labels), n_cls))
        t3.append(topk_from_scores(rand_scores, test_labels, 3))
    results["trivial_stratified_random"] = {
        "top1": float(np.mean(t1)), "top1_std": float(np.std(t1)),
        "top3": float(np.mean(t3)), "top3_std": float(np.std(t3)),
    }
    print(f"\nstratified-random ({N_REPEATS} repeats):")
    print(f"  top-1 {np.mean(t1):5.2f}% +/- {np.std(t1):.2f}   "
          f"top-3 {np.mean(t3):5.2f}% +/- {np.std(t3):.2f} (secondary)")

    print("\n--- learned models must beat these to mean anything ---")
    print("    (compare against outputs/summary_seeds.csv after training)")

    # append to metrics.csv in the same schema the learned models use
    new = not METRICS_CSV.exists()
    with open(METRICS_CSV, "a", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(SCHEMA)
        for name, r in results.items():
            w.writerow([name, "n/a", "__MACRO_AVG__", len(test_labels),
                        "", "", "", f"{r['top1']:.2f}", "",
                        f"split=test;top3={r['top3']:.2f};trivial_predictor"])
    print(f"\nappended to {METRICS_CSV.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
