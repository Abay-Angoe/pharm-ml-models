"""Measure how well the model can say "I don't know".

The classifier is a closed set: softmax over 20 classes always sums to 1, so a
photograph of any of the ~2,342 other C3PI products still returns one of our 20
with some confidence. For a pill identifier that is the worst failure mode
available - a wrong drug name, delivered fluently. A demo therefore needs a
rejection threshold, and the threshold needs to be *measured*, not guessed.

Method (maximum-softmax-probability, the standard baseline):
  1. sample products NOT in our 20 from the C3PI index, download ~100 images
  2. run them through the identical pipeline (localise -> crop -> normalise)
  3. compare the max-softmax distribution against the in-distribution
     validation split
  4. sweep thresholds and report the trade-off honestly: at each threshold, how
     much real accuracy is retained versus how many out-of-set pills are caught

The threshold is chosen on **validation**, never on test.

    python src/ood_rejection.py --checkpoint outputs/checkpoints/transfer_s1_best.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes import EXTENSION, IMAGE_TYPE, NDC9_TO_SLUG  # noqa: E402
from localize import locate, square_crop_box  # noqa: E402

BASE = "https://data.lhncbc.nlm.nih.gov/public/Pills"
UA = "Mozilla/5.0"
ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "raw" / "directory_consumer_grade_images.txt"
OOD_RAW = ROOT / "data" / "raw" / "ood"
OOD_CROPS = ROOT / "data" / "processed_ood"
OUT_JSON = ROOT / "outputs" / "ood_rejection.json"
SIZE = 224


def fetch(url: str, retries: int = 3) -> bytes | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - unmaintained host
            time.sleep(1.5 * (attempt + 1))
    return None


def pick_ood_images(n_products: int, per_product: int, seed: int = 0
                    ) -> list[tuple[str, str]]:
    """Sample (ndc9, remote_path) for products outside our 20 classes."""
    rows = [ln.rstrip("\n").split("|")
            for ln in INDEX.read_text(encoding="utf8", errors="replace").splitlines()]
    rows = [r for r in rows if len(r) >= 5 and r[3] == IMAGE_TYPE
            and r[2].rsplit(".", 1)[-1].upper() == EXTENSION]
    by_prod: dict[str, set[str]] = collections.defaultdict(set)
    for r in rows:
        ndc9 = r[0][:9]
        if ndc9 in NDC9_TO_SLUG:          # skip our own classes
            continue
        by_prod[ndc9].add(r[2])
    # products with enough images to sample from, so we get variety not repeats
    eligible = sorted(p for p, v in by_prod.items() if len(v) >= per_product * 2)
    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(n_products, len(eligible)))
    out: list[tuple[str, str]] = []
    for p in chosen:
        for path in rng.sample(sorted(by_prod[p]), per_product):
            out.append((p, path))
    return out


@torch.no_grad()
def max_softmax(model, crops: list[np.ndarray], device) -> np.ndarray:
    """Max softmax probability per crop, using the eval-time transform."""
    from dataset import build_transforms
    from PIL import Image

    tf = build_transforms(train=False)
    scores = []
    for i in range(0, len(crops), 32):
        batch = torch.stack([
            tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
            for c in crops[i:i + 32]])
        probs = torch.softmax(model(batch.to(device)), dim=1).cpu().numpy()
        scores.append(probs.max(axis=1))
    return np.concatenate(scores) if scores else np.array([])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--n-products", type=int, default=20)
    ap.add_argument("--per-product", type=int, default=5)
    ap.add_argument("--target-retention", type=float, default=0.95,
                    help="fraction of correct in-distribution predictions to keep")
    args = ap.parse_args()

    from dataset import PillDataset
    from evaluate import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, ckpt = load_model(args.checkpoint, device)
    print(f"model: {args.checkpoint.name} (seed {ckpt.get('seed')}, "
          f"val top-1 {ckpt.get('val_top1', float('nan')):.1f}%)")

    # ---- out-of-set images -------------------------------------------------
    wanted = pick_ood_images(args.n_products, args.per_product)
    print(f"\nsampled {len(wanted)} images from {args.n_products} products "
          f"OUTSIDE the 20 classes")
    OOD_RAW.mkdir(parents=True, exist_ok=True)
    OOD_CROPS.mkdir(parents=True, exist_ok=True)
    ood_crops: list[np.ndarray] = []
    n_new = n_cached = n_fail = 0
    for ndc9, remote in wanted:
        dest = OOD_RAW / f"{ndc9}_{Path(remote).name}"
        if not dest.exists():
            data = fetch(f"{BASE}/{urllib.parse.quote(remote)}")
            if data is None or not data.startswith(b"\xff\xd8"):
                n_fail += 1
                continue
            dest.write_bytes(data)
            n_new += 1
        else:
            n_cached += 1
        img = cv2.imread(str(dest))
        if img is None:
            n_fail += 1
            continue
        res = locate(img)
        box = square_crop_box(res, img.shape[1], img.shape[0])
        region = img[box[1]:box[3], box[0]:box[2]]
        if region.size == 0:
            n_fail += 1
            continue
        crop = cv2.resize(region, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OOD_CROPS / dest.name.replace(".JPG", ".png")), crop)
        ood_crops.append(crop)
    print(f"  fetched {n_new}, cached {n_cached}, failed {n_fail} "
          f"-> {len(ood_crops)} usable out-of-set crops")
    if not ood_crops:
        print("no OOD crops - aborting")
        return 1

    # ---- in-distribution: validation split --------------------------------
    val = PillDataset("valid", train_aug=False, classes=classes)
    loader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False)
    id_scores, id_correct = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            id_scores.append(probs.max(axis=1))
            id_correct.append(probs.argmax(axis=1) == y.numpy())
    id_scores = np.concatenate(id_scores)
    id_correct = np.concatenate(id_correct)
    ood_scores = max_softmax(model, ood_crops, device)

    print(f"\nmax-softmax confidence")
    print(f"  in-distribution, correct   (n={id_correct.sum():3d}): "
          f"median {np.median(id_scores[id_correct]):.3f}")
    print(f"  in-distribution, incorrect (n={(~id_correct).sum():3d}): "
          f"median {np.median(id_scores[~id_correct]):.3f}"
          if (~id_correct).any() else "  in-distribution, incorrect: none")
    print(f"  OUT-of-set                 (n={len(ood_scores):3d}): "
          f"median {np.median(ood_scores):.3f}")

    # ---- threshold sweep ---------------------------------------------------
    print(f"\n{'thresh':>7} {'ID correct kept':>16} {'ID wrong kept':>14} "
          f"{'OOD rejected':>13}")
    print("-" * 54)
    rows = []
    for thr in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999):
        keep_ok = float((id_scores[id_correct] >= thr).mean())
        keep_bad = (float((id_scores[~id_correct] >= thr).mean())
                    if (~id_correct).any() else float("nan"))
        rej = float((ood_scores < thr).mean())
        rows.append({"threshold": thr, "id_correct_kept": keep_ok,
                     "id_wrong_kept": keep_bad, "ood_rejected": rej})
        print(f"{thr:7.3f} {keep_ok:15.1%} {keep_bad:13.1%} {rej:12.1%}")

    # ---- recommended operating point --------------------------------------
    # Chosen on validation as: the LOWEST threshold whose in-distribution
    # precision reaches 99%. Rationale for a pill identifier - naming the wrong
    # drug is far worse than declining to answer, so precision is the constraint
    # and coverage is what we spend. Lowest such threshold keeps coverage as high
    # as the precision target allows.
    grid = np.linspace(0.0, 0.999, 200)
    rec = None
    for t in grid:
        shown_ok = float((id_scores[id_correct] >= t).sum())
        shown_bad = float((id_scores[~id_correct] >= t).sum())
        shown = shown_ok + shown_bad
        if shown == 0:
            continue
        precision = shown_ok / shown
        if precision >= 0.99:
            rec = {
                "threshold": float(t),
                "coverage": shown / len(id_scores),
                "precision": precision,
                "ood_rejected": float((ood_scores < t).mean()),
                "ood_still_named": float((ood_scores >= t).mean()),
            }
            break
    if rec:
        print(f"\n--- RECOMMENDED operating point (chosen on validation) ---")
        print(f"  rule: lowest threshold with in-distribution precision >= 99%")
        print(f"  threshold                 : {rec['threshold']:.3f}")
        print(f"  coverage (answers given)  : {rec['coverage']:.1%}")
        print(f"  precision (when it answers): {rec['precision']:.1%}")
        print(f"  out-of-set rejected       : {rec['ood_rejected']:.1%}")
        print(f"  out-of-set STILL NAMED    : {rec['ood_still_named']:.1%}"
              f"  <-- residual risk, must be stated in the UI")

    # also report the naive "retain X% of correct predictions" point for contrast
    thr = float(np.quantile(id_scores[id_correct], 1.0 - args.target_retention))
    rej = float((ood_scores < thr).mean())
    keep_bad = (float((id_scores[~id_correct] >= thr).mean())
                if (~id_correct).any() else float("nan"))
    print(f"\n--- chosen operating point (validation, "
          f"{args.target_retention:.0%} retention) ---")
    print(f"  threshold                      : {thr:.3f}")
    print(f"  out-of-set pills rejected      : {rej:.1%}")
    print(f"  out-of-set pills WRONGLY named : {1 - rej:.1%}  <-- the risk")
    print(f"  in-distribution errors still shown: {keep_bad:.1%}")
    verdict = ("usable as a guard" if rej >= 0.5 else
               "WEAK - max-softmax barely separates out-of-set pills here")
    print(f"  verdict: {verdict}")

    OUT_JSON.write_text(json.dumps({
        "checkpoint": args.checkpoint.name, "seed": ckpt.get("seed"),
        "n_ood_images": len(ood_scores), "n_ood_products": args.n_products,
        "target_retention": args.target_retention,
        "recommended": rec,
        "chosen_threshold": thr, "ood_rejected_at_threshold": rej,
        "id_wrong_kept_at_threshold": keep_bad,
        "sweep": rows,
        "median_conf": {
            "id_correct": float(np.median(id_scores[id_correct])),
            "id_wrong": (float(np.median(id_scores[~id_correct]))
                         if (~id_correct).any() else None),
            "ood": float(np.median(ood_scores)),
        },
        "verdict": verdict,
    }, indent=1))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
