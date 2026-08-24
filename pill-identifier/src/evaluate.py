"""Evaluate a checkpoint: top-1 headline, per-class metrics, confusion analysis.

    python src/evaluate.py --checkpoint outputs/checkpoints/baseline_best.pt
    python src/evaluate.py --checkpoint ... --split valid

Reports (section 7):
  * top-1 accuracy as the headline number, plus macro-F1 and per-class P/R/F1
  * top-3 as a clearly-labelled secondary "close enough for a lookup tool"
    signal. Top-5 is NOT reported: at 15-30 classes it is 17-33% by chance.
  * the most-confused pairs, rather than a 30x30 matrix nobody can read; also
    written to outputs/confused_pairs_<tag>_<split>.csv as a deliverable
  * accuracy split by localisation method, and by whether confusions stay
    inside a look-alike group - the latter feeds the section 8 imprint question

Rows are appended to outputs/metrics.csv in the section 0 schema:
    phase, seed, class, support, precision, recall, f1, accuracy_overall,
    macro_f1, notes
so that comparing phases is a groupby rather than a manual merge.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes import CLASS_GROUPS, CLASS_TO_GROUP  # noqa: E402
from dataset import PillDataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
METRICS_CSV = ROOT / "outputs" / "metrics.csv"

# section 0 mandated schema - one row per (phase, seed, class)
SCHEMA = ["phase", "seed", "class", "support", "precision", "recall", "f1",
          "accuracy_overall", "macro_f1", "notes"]

def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    phase = ckpt["phase"]
    if phase == "baseline":
        from model_baseline import build_model

        model = build_model(len(classes))
    else:
        from model_transfer import build_model

        model = build_model(len(classes), backbone=ckpt.get("backbone") or "resnet18")
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, classes, ckpt


@torch.no_grad()
def collect_predictions(model, dataset, device, batch_size: int = 32):
    """Return (probs [N,C], targets [N], methods [N])."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs, all_targets = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu())
        all_targets.append(y)
    return (
        torch.cat(all_probs).numpy(),
        torch.cat(all_targets).numpy(),
        np.array(dataset.methods),
    )


def topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    return 100.0 * float((topk == targets[:, None]).any(axis=1).mean())


def same_group(a: str, b: str) -> bool:
    """Do two classes belong to the SAME look-alike group?

    Classes with no look-alike partner are absent from CLASS_TO_GROUP, so a
    plain `get(a) == get(b)` compares None to None and reports two unrelated
    solo classes as same-group. Guard against that explicitly.
    """
    ga = CLASS_TO_GROUP.get(a)
    return ga is not None and ga == CLASS_TO_GROUP.get(b)


def most_confused(cm: np.ndarray, classes: list[str], top: int = 12):
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], classes[i], classes[j],
                              same_group(classes[i], classes[j])))
    pairs.sort(reverse=True, key=lambda p: p[0])
    return pairs[:top]


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # order classes by group so same-drug confusions cluster near the diagonal
    order = sorted(range(len(classes)),
                   key=lambda i: (CLASS_TO_GROUP.get(classes[i], "zz"), classes[i]))
    cm_o = cm[np.ix_(order, order)]
    labels = [f"{classes[i]}" for i in order]

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm_o, annot=True, fmt="d", cmap="magma_r", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 7}, linewidths=0.4, linecolor="#dddddd")
    # bracket each drug group on the axes
    boundaries, seen = [], None
    for pos, i in enumerate(order):
        g = CLASS_TO_GROUP.get(classes[i], "zz")
        if g != seen:
            boundaries.append(pos)
            seen = g
    for b in boundaries[1:]:
        ax.axhline(b, color="#0d6efd", lw=1.6)
        ax.axvline(b, color="#0d6efd", lw=1.6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\nblue lines separate same-drug groups", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed-note", default="unknown",
                    help="fallback seed label if the checkpoint predates seed recording")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, ckpt = load_model(args.checkpoint, device)
    tag = args.tag or args.checkpoint.stem.replace("_best", "")

    dataset = PillDataset(args.split, train_aug=False, classes=classes)
    probs, targets, methods = collect_predictions(model, dataset, device)
    preds = probs.argmax(axis=1)

    top1 = topk_accuracy(probs, targets, 1)
    top3 = topk_accuracy(probs, targets, 3)
    print(f"=== {tag} on {args.split} ({len(targets)} images, "
          f"{len(classes)} classes) ===")
    print(f"checkpoint epoch {ckpt.get('epoch')}, val top-1 "
          f"{ckpt.get('val_top1', float('nan')):.1f}%")
    print(f"\ntop-1 accuracy (HEADLINE) : {top1:5.2f}%   "
          f"(chance {100 / len(classes):.1f}%)")
    print(f"top-3 accuracy (secondary; 'close enough for a lookup tool') : "
          f"{top3:5.2f}%   (chance {300 / len(classes):.1f}%)")
    print("  [top-5 intentionally not reported: ~17-33% by chance at this "
          "class count]")

    # --- by localisation method -------------------------------------------
    # C3PI has no paired lighting conditions, so slice by how the crop was
    # obtained instead. If fallback crops score much worse, the limiting factor
    # is localisation rather than the model.
    print("\n--- accuracy by localisation method ---")
    by_method: dict[str, float] = {}
    for code, label in (("detected", "pill detected  "),
                        ("center_fallback", "centre fallback")):
        m = methods == code
        if m.sum():
            t1 = topk_accuracy(probs[m], targets[m], 1)
            by_method[code] = t1
            print(f"  {label}: n={m.sum():4d}  top-1 {t1:5.1f}%  "
                  f"top-3 {topk_accuracy(probs[m], targets[m], 3):5.1f}%")

    # --- group structure of the errors ------------------------------------
    wrong = preds != targets
    n_wrong = int(wrong.sum())
    if n_wrong:
        n_same_group = sum(
            same_group(classes[t], classes[p])
            for t, p in zip(targets[wrong], preds[wrong])
        )
        print(f"\n--- error structure ---")
        print(f"  {n_wrong} errors; {n_same_group} "
              f"({100 * n_same_group / n_wrong:.0f}%) "
              f"stay inside the same drug group")
        print("  (high share = model gets the drug family right, "
              "misses the imprint/dose detail)")

    # --- per-class metrics -> metrics.csv ---------------------------------
    report = classification_report(targets, preds, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True,
                                   zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    phase = ckpt.get("phase", "unknown")
    seed = ckpt.get("seed", args.seed_note)
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not METRICS_CSV.exists()
    with open(METRICS_CSV, "a", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(SCHEMA)
        for cls in classes:
            r = report[cls]
            w.writerow([phase, seed, cls, int(r["support"]),
                        f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                        f"{r['f1-score']:.4f}", f"{top1:.2f}",
                        f"{macro_f1:.4f}",
                        f"split={args.split};group={CLASS_TO_GROUP.get(cls, '')};"
                        f"top3={top3:.2f}"])
        w.writerow([phase, seed, "__MACRO_AVG__",
                    int(report["macro avg"]["support"]),
                    f"{report['macro avg']['precision']:.4f}",
                    f"{report['macro avg']['recall']:.4f}",
                    f"{macro_f1:.4f}", f"{top1:.2f}", f"{macro_f1:.4f}",
                    f"split={args.split};top3={top3:.2f};"
                    f"det={by_method.get('detected', float('nan')):.2f};"
                    f"fb={by_method.get('center_fallback', float('nan')):.2f};"
                    f"top5_not_reported_per_spec_s7"])
    print(f"\nmacro F1: {macro_f1:.3f}   "
          f"-> per-class metrics appended to {METRICS_CSV.name}")

    worst = sorted(classes, key=lambda c: report[c]["f1-score"])[:6]
    print("\nweakest classes by F1:")
    for c in worst:
        r = report[c]
        print(f"  {r['f1-score']:.2f}  {c:36s} "
              f"(P {r['precision']:.2f} R {r['recall']:.2f}) "
              f"[{CLASS_TO_GROUP.get(c, '-')}]")

    # --- confusion matrix -------------------------------------------------
    cm = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    pairs = most_confused(cm, classes, top=12)
    if pairs:
        print("\nmost-confused pairs (true -> predicted):")
        for n, a, b, same in pairs:
            mark = "same-group" if same else "CROSS-GROUP"
            print(f"  {n:2d}x  {a:34s} -> {b:34s} [{mark}]")
        # section 9 deliverable: most-confused-pairs table, not just stdout
        pairs_csv = ROOT / "outputs" / f"confused_pairs_{tag}_{args.split}.csv"
        with open(pairs_csv, "w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh)
            w.writerow(["phase", "seed", "count", "true_class", "predicted_class",
                        "same_drug_group"])
            for n, a, b, same in most_confused(cm, classes, top=40):
                w.writerow([phase, seed, n, a, b, int(same)])
        print(f"confused-pairs table -> {pairs_csv.name}")
    out = FIG_DIR / f"confusion_{tag}_{args.split}.png"
    plot_confusion(cm, classes, f"{tag} - {args.split} (top-1 {top1:.1f}%, "
                                f"top-3 {top3:.1f}%)", out)
    print(f"\nconfusion matrix -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
