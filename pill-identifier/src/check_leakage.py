"""Check for near-duplicate leakage between train and test.

Why this matters: C3PI consumer photographs were captured in batches - the same
physical tablet photographed several times in one session, on the same surface
under the same light. Byte-identical duplicates are easy to catch, but *near*
duplicates are not, and if a test image has an almost identical twin in train the
reported accuracy measures memorisation rather than recognition.

Method: reduce every crop to a small normalised grayscale signature, then for each
test image find its nearest train neighbour *of the same class* and report the
distribution of distances. A calibration set of distances between images of
*different* classes gives the scale for "how close is close".

    python src/check_leakage.py
    python src/check_leakage.py --dump 12   # save the most-similar pairs as a sheet
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MANIFEST = PROCESSED / "manifest.csv"
SIG = 32  # signature resolution


def signature(path: Path) -> np.ndarray:
    """Small, contrast-normalised grayscale signature (rotation-sensitive)."""
    with Image.open(path) as im:
        g = im.convert("L").resize((SIG, SIG), Image.Resampling.BOX)
    a = np.asarray(g, dtype=np.float32).ravel()
    a -= a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def rot_signatures(path: Path) -> list[np.ndarray]:
    """Signatures at 0/90/180/270 degrees, matching preprocess.py's grouping.

    The verification must be at least as strict as the de-duplication it checks,
    otherwise a re-oriented twin would pass unnoticed.
    """
    out = []
    with Image.open(path) as im:
        g = im.convert("L")
        for k in range(4):
            r = g.rotate(90 * k, expand=True).resize((SIG, SIG), Image.Resampling.BOX)
            a = np.asarray(r, dtype=np.float32).ravel()
            a -= a.mean()
            n = np.linalg.norm(a)
            out.append(a / n if n > 0 else a)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=0,
                    help="write a contact sheet of the N most similar pairs")
    ap.add_argument("--split", default="test", choices=["test", "valid"],
                    help="which held-out split to check against train")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf8")))
    by_split: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for r in rows:
        p = PROCESSED / r["split"] / r["class"] / f"{r['stem']}.png"
        if p.exists():
            by_split[r["split"]].append((r["class"], p))

    train = by_split["train"]
    test = by_split[args.split]
    print(f"train {len(train)}  {args.split} {len(test)}")

    sig_train = {p: rot_signatures(p) for _, p in train}
    sig_test = {p: signature(p) for _, p in test}

    train_by_cls: dict[str, list[Path]] = defaultdict(list)
    for c, p in train:
        train_by_cls[c].append(p)

    # cosine distance = 1 - dot (signatures are unit norm, zero mean)
    same_best: list[tuple[float, Path, Path]] = []
    for c, p in test:
        v = sig_test[p]
        best, bp = 2.0, None
        for q in train_by_cls[c]:
            d = min(1.0 - float(v @ vq) for vq in sig_train[q])
            if d < best:
                best, bp = d, q
        if bp is not None:
            same_best.append((best, p, bp))

    # calibration: nearest train neighbour from a DIFFERENT class
    rng = np.random.default_rng(0)
    other: list[float] = []
    all_train_paths = [p for _, p in train]
    for c, p in test:
        v = sig_test[p]
        pool = [q for q in rng.choice(all_train_paths, size=min(60, len(all_train_paths)),
                                      replace=False)]
        ds = [min(1.0 - float(v @ vq) for vq in sig_train[q]) for q in pool
              if q.parent.name != c]
        if ds:
            other.append(min(ds))

    d_same = np.array([d for d, _, _ in same_best])
    d_other = np.array(other)
    print("\nnearest-train-neighbour cosine distance for each test image")
    print(f"  same class     : median {np.median(d_same):.3f}  "
          f"p5 {np.percentile(d_same, 5):.3f}  min {d_same.min():.3f}")
    print(f"  different class: median {np.median(d_other):.3f}  "
          f"p5 {np.percentile(d_other, 5):.3f}  min {d_other.min():.3f}")
    print("  (0 = identical; same-class should be lower, that is expected and fine)")

    # A test image is 'suspect' if it is far closer to a train image than the
    # different-class floor would explain.
    floor = float(np.percentile(d_other, 5))
    for thr, label in ((0.02, "near-identical"), (0.05, "very similar"),
                       (0.10, "similar")):
        n = int((d_same < thr).sum())
        print(f"\n  test images within {thr:.2f} of a same-class train image: "
              f"{n}/{len(d_same)} ({100 * n / len(d_same):.1f}%)  [{label}]")
    print(f"\n  different-class 5th percentile (noise floor) = {floor:.3f}")

    worst = sorted(same_best)[: max(args.dump, 10)]
    print("\nmost similar train/test pairs:")
    for d, tp, qp in worst[:10]:
        print(f"  d={d:.4f}  {tp.parent.name:24s} test:{tp.stem[:14]} "
              f"<- train:{qp.stem[:14]}")

    if args.dump:
        import cv2

        TH = 150
        n = min(args.dump, len(worst))
        sheet = np.full((TH * n, TH * 2, 3), 255, np.uint8)
        for i, (d, tp, qp) in enumerate(worst[:n]):
            for j, path in enumerate((tp, qp)):
                img = cv2.resize(cv2.imread(str(path)), (TH, TH))
                sheet[i * TH:(i + 1) * TH, j * TH:(j + 1) * TH] = img
        out = ROOT / "outputs" / "figures" / "_leakage_pairs.png"
        cv2.imwrite(str(out), sheet)
        print(f"\nwrote {out} (left = test, right = nearest train)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
