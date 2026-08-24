"""Crop C3PI photos to the pill and build a stratified 70/15/15 split.

C3PI ships no train/val/test split and no segmentation masks, so this script does
both jobs:

  1. localise the pill with src/localize.py and write a square 224x224 crop.
     In the raw photos the pill spans ~10-15% of frame width and sits off-centre,
     so a naive full-frame resize leaves it ~39 px across and erases the imprint -
     the feature section 8 exists to test for.
  2. group near-duplicate captures, then split 70/15/15 stratified by class at
     *group* level with a fixed seed, so near-identical photographs of one
     physical pill never straddle train and test. Splitting happens after
     cropping so the manifest records the group id and split of every crop.

Every crop records the method used ("detected" vs "center_fallback") in the
manifest, because a silent fallback would be indistinguishable from a successful
detection and would quietly become label noise.

Outputs:
    data/processed/<split>/<class>/<stem>.png
    data/processed/manifest.csv       one row per crop: split, class, method, box
    outputs/preprocess_stats.json

    python src/preprocess.py
    python src/preprocess.py --limit 40   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes import CLASSES  # noqa: E402
from localize import locate, square_crop_box  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "c3pi"
OUT = ROOT / "data" / "processed"
MANIFEST = OUT / "manifest.csv"
STATS = ROOT / "outputs" / "preprocess_stats.json"

SIZE = 224
SPLITS = (("train", 0.70), ("valid", 0.15), ("test", 0.15))
SEED = 42

# --- duplicate grouping ---------------------------------------------------
# C3PI photographs were captured in batches: the same physical tablet shot several
# times in one session, same surface, same light. A plain per-image split puts
# near-identical twins on both sides and the reported accuracy then measures
# memorisation. Measured on an image-level split: 21% of test images sat within
# 0.02 cosine distance of a train image, against a different-class noise floor of
# 0.097 - and inspection confirmed they were the same pill in the same pose.
#
# So images are clustered by appearance *within class* and whole clusters are
# assigned to one split. 0.08 is comfortably under the 0.097 floor, so it merges
# real duplicates without collapsing genuinely distinct photos; at this threshold
# every class still has >=9 clusters, enough to populate all three splits.
DUP_THRESHOLD = 0.08
SIG = 32


def _signature(crop_bgr: np.ndarray) -> np.ndarray:
    """Small contrast-normalised grayscale signature of an already-cropped pill."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (SIG, SIG), interpolation=cv2.INTER_AREA)
    # flatten to a 1-D unit vector so the dot product is a scalar cosine
    v = g.astype(np.float32).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else v


def _rot_signatures(crop_bgr: np.ndarray) -> list[np.ndarray]:
    """Signatures at 0/90/180/270 degrees.

    The signature is rotation-sensitive, but a re-shoot of the same pill is often
    rotated. Comparing against all four right-angle rotations catches those
    without the cost of a full rotation-invariant descriptor.
    """
    out = []
    img = crop_bgr
    for _ in range(4):
        out.append(_signature(img))
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return out


def group_duplicates(sigs: dict[Path, np.ndarray],
                     rot_sigs: dict[Path, list[np.ndarray]],
                     threshold: float = DUP_THRESHOLD) -> list[list[Path]]:
    """Union-find clusters of near-duplicate crops (single class)."""
    paths = sorted(sigs)
    parent = {p: p for p in paths}

    def find(a: Path) -> Path:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: Path, b: Path) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(paths):
        va = sigs[a]
        for b in paths[i + 1:]:
            if find(a) == find(b):
                continue
            # min distance over the four right-angle rotations of b
            d = min(1.0 - float(va @ vb) for vb in rot_sigs[b])
            if d < threshold:
                union(a, b)
    clusters: dict[Path, list[Path]] = defaultdict(list)
    for p in paths:
        clusters[find(p)].append(p)
    return sorted(clusters.values(), key=len, reverse=True)


def grouped_split(groups_by_class: dict[str, list[list[Path]]],
                  seed: int = SEED) -> dict[Path, str]:
    """Assign whole duplicate-groups to splits, stratified by class.

    Groups vary in size, so a simple count-based allocation would skew the image
    proportions. Instead walk the groups largest-first and give each to whichever
    split is currently furthest below its image quota - which keeps image-level
    proportions near 70/15/15 while never splitting a group.
    """
    assignment: dict[Path, str] = {}
    rng = random.Random(seed)
    for cls in sorted(groups_by_class):
        groups = list(groups_by_class[cls])
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        n_images = sum(len(g) for g in groups)
        quota = {name: n_images * frac for name, frac in SPLITS}
        placed = {name: 0 for name, _ in SPLITS}

        # seed valid and test with one group each so no class can miss a split
        for name in ("valid", "test"):
            if len(groups) > 2:
                g = min(groups, key=len)      # smallest group, to limit skew
                groups.remove(g)
                for f in g:
                    assignment[f] = name
                placed[name] += len(g)

        for g in groups:
            name = max(placed, key=lambda k: (quota[k] - placed[k]) / max(quota[k], 1e-9))
            for f in g:
                assignment[f] = name
            placed[name] += len(g)
    return assignment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    by_class: dict[str, list[Path]] = defaultdict(list)
    for cls in CLASSES:
        by_class[cls] = sorted((RAW / cls).glob("*.JPG"))
    total = sum(len(v) for v in by_class.values())
    if not total:
        print(f"no images under {RAW} - run src/fetch_c3pi.py first")
        return 1
    empty = [c for c, v in by_class.items() if not v]
    if empty:
        print(f"!! classes with no downloaded images: {empty}")
        return 1

    print(f"{total} raw images across {len(by_class)} classes")
    print("per-class:", {c: len(v) for c, v in sorted(by_class.items())})

    if args.limit:
        keep = set()
        for cls, files in by_class.items():
            keep.update(files[: max(1, args.limit // len(by_class))])
        by_class = {c: [f for f in v if f in keep] for c, v in by_class.items()}

    # ---- pass 1: crop everything, keeping crops in memory for grouping ----
    print("\npass 1/3: localise + crop")
    crops: dict[Path, np.ndarray] = {}
    meta: dict[Path, dict] = {}
    failures: list[str] = []
    methods = Counter()
    processed = 0
    for cls in sorted(by_class):
        for src in by_class[cls]:
            img = cv2.imread(str(src))
            if img is None:
                failures.append(f"{src.name}: unreadable")
                continue
            res = locate(img)
            box = square_crop_box(res, img.shape[1], img.shape[0])
            region = img[box[1]:box[3], box[0]:box[2]]
            if region.size == 0:
                failures.append(f"{src.name}: empty crop {box}")
                continue
            crop = cv2.resize(region, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            crops[src] = crop
            methods[res.method] += 1
            side = max(1, box[2] - box[0])
            meta[src] = {
                "class": cls, "stem": src.stem, "method": res.method,
                "area_frac": f"{res.area_frac:.5f}", "fill": f"{res.fill:.3f}",
                "box": "|".join(str(v) for v in box),
                "orig_w": img.shape[1], "orig_h": img.shape[0],
                "pill_px_at_224": (f"{224 * (res.x1 - res.x0) / side:.1f}"
                                   if res.method == "detected" else ""),
                "pill_px_naive": (f"{224 * (res.x1 - res.x0) / img.shape[1]:.1f}"
                                  if res.method == "detected" else ""),
            }
            processed += 1
            if processed % 300 == 0:
                print(f"  {processed}/{total} ...", flush=True)

    if not crops:
        print("nothing cropped")
        return 1

    # ---- pass 2: group near-duplicates within each class -----------------
    print("\npass 2/3: group near-duplicate captures "
          f"(cosine threshold {DUP_THRESHOLD})")
    groups_by_class: dict[str, list[list[Path]]] = {}
    for cls in sorted(by_class):
        paths = [p for p in by_class[cls] if p in crops]
        if not paths:
            continue
        sigs = {p: _signature(crops[p]) for p in paths}
        rot = {p: _rot_signatures(crops[p]) for p in paths}
        groups_by_class[cls] = group_duplicates(sigs, rot)
    sizes = [len(g) for gs in groups_by_class.values() for g in gs]
    n_groups = len(sizes)
    print(f"  {len(crops)} images -> {n_groups} capture groups "
          f"(median size {np.median(sizes):.1f}, largest {max(sizes)})")
    print(f"  groups per class: min {min(len(g) for g in groups_by_class.values())}, "
          f"max {max(len(g) for g in groups_by_class.values())}")
    multi = sum(1 for s in sizes if s > 1)
    print(f"  {multi}/{n_groups} groups contain >1 image "
          f"({sum(s for s in sizes if s > 1)} images are part of a duplicate set)")

    # ---- pass 3: grouped split + write ----------------------------------
    print("\npass 3/3: grouped stratified split + write")
    assignment = grouped_split(groups_by_class, seed=args.seed)
    group_of: dict[Path, int] = {}
    gid = 0
    for cls in sorted(groups_by_class):
        for g in groups_by_class[cls]:
            for p in g:
                group_of[p] = gid
            gid += 1

    rows: list[dict] = []
    for src, crop in crops.items():
        split = assignment[src]
        cls = meta[src]["class"]
        dest = OUT / split / cls / f"{src.stem}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), crop)
        rows.append({"split": split, "group": group_of[src], **meta[src]})

    if not rows:
        print("nothing processed")
        return 1

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    split_counts = Counter(r["split"] for r in rows)
    per_split_class = defaultdict(Counter)
    for r in rows:
        per_split_class[r["split"]][r["class"]] += 1

    print(f"\nwrote {len(rows)} crops -> {OUT}")
    print(f"localisation: " + ", ".join(
        f"{k} {v} ({100 * v / len(rows):.1f}%)" for k, v in methods.most_common()))
    print("\nsplit sizes (target 70/15/15):")
    for name, frac in SPLITS:
        n = split_counts[name]
        cls_counts = sorted(per_split_class[name].values())
        print(f"  {name:6s} {n:5d} ({100 * n / len(rows):4.1f}%)  "
              f"per-class {cls_counts[0]}-{cls_counts[-1]}, "
              f"{len(per_split_class[name])} classes")
    missing = [(s, c) for s, _ in SPLITS for c in CLASSES
               if per_split_class[s][c] == 0]
    print(f"class/split cells with zero images: {len(missing)}"
          + (f" {missing[:5]}" if missing else " (all classes in all splits)"))

    px = [float(r["pill_px_at_224"]) for r in rows if r["pill_px_at_224"]]
    nv = [float(r["pill_px_naive"]) for r in rows if r["pill_px_naive"]]
    if px:
        print(f"\ndetected subset, pill width at {SIZE}px input:")
        print(f"  after crop   median {np.median(px):5.1f}px")
        print(f"  naive resize median {np.median(nv):5.1f}px")
        print(f"  -> {np.median(px) / max(np.median(nv), 1e-9):.1f}x more pixels "
              f"on the pill")
    if failures:
        print(f"\n{len(failures)} failures:")
        for f in failures[:10]:
            print("   ", f)

    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps({
        "n_crops": len(rows), "methods": dict(methods),
        "dup_threshold": DUP_THRESHOLD,
        "n_groups": n_groups,
        "group_size_median": float(np.median(sizes)),
        "group_size_max": int(max(sizes)),
        "images_in_multi_image_groups": int(sum(s for s in sizes if s > 1)),
        "split_counts": dict(split_counts),
        "per_split_class": {s: dict(c) for s, c in per_split_class.items()},
        "pill_px_median_cropped": float(np.median(px)) if px else None,
        "pill_px_median_naive": float(np.median(nv)) if nv else None,
        "failures": failures, "seed": args.seed,
    }, indent=1))
    print(f"stats -> {STATS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
