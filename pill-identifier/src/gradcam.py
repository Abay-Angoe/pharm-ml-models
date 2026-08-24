"""Grad-CAM, plus a quantitative test of WHAT the model looks at (section 8).

A figure grid alone only supports "the heatmap looks like it's on the pill". The
spec asks something sharper: is the model reading the imprint, or shortcutting on
overall colour and shape? If it is the latter it will fail exactly where a real
tool needs to work - telling two white round pills apart.

Three measures, computed against an estimated pill mask:

  on_pill      share of CAM mass inside the pill silhouette. Low values mean the
               model is using background or lighting, not the pill at all.
  interior     share of the *on-pill* CAM mass in the eroded interior rather than
               the rim annulus. Imprints live in the interior; a pure
               shape/silhouette detector fires on the rim.
  imprint_corr Spearman correlation between the CAM and local edge energy, taken
               *inside the eroded interior*. Engraved text and score lines are
               where interior contrast is high, so a positive correlation is
               evidence of imprint-reading; ~0 means flat-colour shortcutting.

Per section 8 the sample deliberately includes both correct and incorrect
predictions for each class, since the interesting question is whether attention
differs between them.

Two honest caveats specific to C3PI:
  * it ships no segmentation ground truth, so the mask is estimated from the
    image (see pill_mask_for_crop). Numbers are good-faith estimates.
  * it has no paired lighting conditions, so the "engravings visible vs washed
    out" controlled comparison is not available on this dataset. The colour-
    saturation analysis in make_figures.py carries that load instead.

    python src/gradcam.py --checkpoint outputs/checkpoints/transfer_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
SIZE = 224


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared - equivalent to scipy.stats.rankdata."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10:
        return float("nan")
    rx, ry = _rankdata(x.astype(np.float64)), _rankdata(y.astype(np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def pill_mask_for_crop(rgb01: np.ndarray) -> np.ndarray | None:
    """Estimate the pill mask from a 224x224 crop.

    C3PI has no segmentation ground truth, so this is *estimated*: it reuses
    localize.py's illumination-flattened score map, Otsu-thresholds it inside the
    crop and keeps the plausible component nearest the centre (the crop is
    centred on the pill by construction).
    """
    import cv2

    from localize import _score_map, norm01

    bgr = cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8),
                       cv2.COLOR_RGB2BGR)
    try:
        score = _score_map(bgr)
    except cv2.error:
        return None
    u8 = (norm01(score) * 255).astype(np.uint8)
    _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    if n <= 1:
        return None
    h, w = binary.shape
    best, best_d = None, None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (0.03 * h * w <= area <= 0.95 * h * w):
            continue
        cx, cy = centroids[i]
        d = (cx - w / 2) ** 2 + (cy - h / 2) ** 2
        if best_d is None or d < best_d:
            best, best_d = i, d
    if best is None:
        return None
    return (labels == best).astype(np.float32)


def erode(mask: np.ndarray, iterations: int = 6) -> np.ndarray:
    """Binary erosion via min-pooling with a 3x3 structuring element."""
    m = mask.copy()
    for _ in range(iterations):
        p = np.pad(m, 1, mode="constant", constant_values=0)
        m = np.minimum.reduce([
            p[0:-2, 0:-2], p[0:-2, 1:-1], p[0:-2, 2:],
            p[1:-1, 0:-2], p[1:-1, 1:-1], p[1:-1, 2:],
            p[2:, 0:-2], p[2:, 1:-1], p[2:, 2:],
        ])
    return m


def edge_energy(rgb: np.ndarray) -> np.ndarray:
    """Local gradient magnitude - a proxy for engraved text / score lines."""
    grey = rgb.mean(axis=2)
    gx = np.zeros_like(grey)
    gy = np.zeros_like(grey)
    gx[:, 1:-1] = grey[:, 2:] - grey[:, :-2]
    gy[1:-1, :] = grey[2:, :] - grey[:-2, :]
    mag = np.sqrt(gx**2 + gy**2)
    k = np.ones((3, 3)) / 9.0
    pad = np.pad(mag, 1, mode="edge")
    out = np.zeros_like(mag)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * pad[dy:dy + mag.shape[0], dx:dx + mag.shape[1]]
    return out


def analyse(cam: np.ndarray, rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    cam = np.clip(cam, 0, None)
    total = cam.sum()
    if total <= 0 or mask.sum() < 50:
        return {}
    on_pill = float((cam * mask).sum() / total)

    inner = erode(mask, iterations=8)
    rim = np.clip(mask - inner, 0, 1)
    on_mass = (cam * mask).sum()
    interior = float((cam * inner).sum() / on_mass) if on_mass > 0 else float("nan")

    # Correlate only *within the eroded interior*. The pill's silhouette is the
    # strongest edge present, so including the rim would make this measure track
    # "is the CAM near the outline" instead of "is the CAM on the engraving" -
    # and would read negative for any interior-focused model.
    idx = inner > 0.5
    corr = (spearman(cam[idx], edge_energy(rgb)[idx])
            if idx.sum() >= 50 else float("nan"))
    return {
        "on_pill": on_pill,
        "interior": interior,
        "rim": float((cam * rim).sum() / on_mass) if on_mass > 0 else float("nan"),
        "imprint_corr": corr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-class", type=int, default=4,
                    help="max images per class; correct and incorrect both sampled")
    ap.add_argument("--grid-classes", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from classes import CLASS_TO_GROUP
    from dataset import MEAN, STD, PillDataset
    from evaluate import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, ckpt = load_model(args.checkpoint, device)
    tag = args.tag or args.checkpoint.stem.replace("_best", "")
    dataset = PillDataset(args.split, train_aug=False, classes=classes)

    mean, std = np.array(MEAN), np.array(STD)

    # ---- first pass: predictions for everything, so we can deliberately
    # sample both correct and incorrect per class (section 8) ---------------
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    preds: list[int] = []
    with torch.no_grad():
        for x, _ in loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist())

    by_class_correct: dict[int, list[int]] = defaultdict(list)
    by_class_wrong: dict[int, list[int]] = defaultdict(list)
    for i, (_, label, _) in enumerate(dataset.samples):
        (by_class_correct if preds[i] == label else by_class_wrong)[label].append(i)

    half = max(1, args.per_class // 2)
    chosen: list[int] = []
    for label in range(len(classes)):
        wrong = by_class_wrong[label][:half]
        right = by_class_correct[label][: args.per_class - len(wrong)]
        chosen += right + wrong
    n_wrong_sampled = sum(1 for i in chosen if preds[i] != dataset.samples[i][1])
    print(f"sampling {len(chosen)} images over {len(classes)} classes "
          f"({n_wrong_sampled} incorrect, {len(chosen) - n_wrong_sampled} correct)")

    cam_engine = GradCAM(model=model, target_layers=[model.gradcam_target_layer])
    records: list[dict] = []
    cache: dict[int, tuple[np.ndarray, np.ndarray, bool]] = {}
    no_mask = 0
    for n, i in enumerate(chosen, 1):
        path, label, method = dataset.samples[i]
        x, _ = dataset[i]
        xb = x.unsqueeze(0).to(device)
        cam = cam_engine(input_tensor=xb,
                         targets=[ClassifierOutputTarget(label)])[0]
        rgb = np.clip(x.permute(1, 2, 0).numpy() * std + mean, 0, 1)
        ok = preds[i] == label
        cache[i] = (cam, rgb, ok)
        mask = pill_mask_for_crop(rgb)
        if mask is None:
            no_mask += 1
            continue
        stats = analyse(cam, rgb * 255.0, mask)
        if stats:
            stats.update(cls=classes[label],
                         group=CLASS_TO_GROUP.get(classes[label], "solo"),
                         method=method, correct=bool(ok), stem=path.stem)
            records.append(stats)
        if n % 20 == 0:
            print(f"  {n}/{len(chosen)} ...", flush=True)

    if not records:
        print("no records produced - is data/processed present?")
        return 1
    if no_mask:
        print(f"note: mask estimation failed on {no_mask} crops (excluded)")

    def agg(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n=== Grad-CAM attention analysis: {tag} ({args.split}) ===")
    print(f"n = {len(records)} images (mask estimated; C3PI has no ground truth)\n")
    header = (f"{'subset':>26} {'n':>4} {'on_pill':>9} {'interior':>9} "
              f"{'rim':>7} {'imprint_corr':>13}")
    print(header)
    print("-" * len(header))
    subsets: list[tuple[str, list[dict]]] = [
        ("ALL", records),
        ("correct predictions", [r for r in records if r["correct"]]),
        ("incorrect predictions", [r for r in records if not r["correct"]]),
        ("pill detected", [r for r in records if r["method"] == "detected"]),
        ("centre fallback", [r for r in records if r["method"] == "center_fallback"]),
    ]
    for name, rows in subsets:
        if not rows:
            continue
        print(f"{name:>26} {len(rows):4d} {agg(rows, 'on_pill'):9.3f} "
              f"{agg(rows, 'interior'):9.3f} {agg(rows, 'rim'):7.3f} "
              f"{agg(rows, 'imprint_corr'):13.3f}")

    on_pill_all = agg(records, "on_pill")
    interior_all = agg(records, "interior")
    corr_all = agg(records, "imprint_corr")
    corr_ok = agg([r for r in records if r["correct"]], "imprint_corr")
    corr_bad = agg([r for r in records if not r["correct"]], "imprint_corr")

    print("\n--- verdict ---")
    print(f"  CAM mass on the pill    : {on_pill_all:.1%}")
    print("    -> " + ("WEAK: much attention is off-pill (background shortcut)"
                       if on_pill_all < 0.5 else
                       "the model is looking at the pill, not the background"))
    print(f"  interior vs rim         : {interior_all:.1%} interior")
    print("    -> " + ("leaning on silhouette/shape rather than surface detail"
                       if interior_all < 0.35 else
                       "substantial attention on the pill face, where imprints are"))
    print(f"  edge-energy correlation : {corr_all:+.3f} "
          f"(correct {corr_ok:+.3f} vs incorrect {corr_bad:+.3f})")
    if corr_all > 0.05:
        print("    -> IMPRINT-DRIVEN: attention tracks engraved/printed detail")
    elif corr_all > 0.0:
        print("    -> weakly imprint-driven; mostly colour/shape")
    else:
        print("    -> NO imprint evidence: attention does not track engraved "
              "detail; likely a colour/shape shortcut")
    if np.isfinite(corr_ok) and np.isfinite(corr_bad) and corr_ok > corr_bad + 0.02:
        print("    -> and it reads the imprint MORE on the images it gets right, "
              "which links the mechanism to the accuracy")

    out_json = ROOT / "outputs" / f"gradcam_analysis_{tag}.json"
    out_json.write_text(json.dumps(
        {"tag": tag, "split": args.split, "n": len(records),
         "mask_source": "estimated (C3PI has no segmentation ground truth)",
         "masks_failed": no_mask,
         "summary": {name: {k: agg(rows, k) for k in
                            ("on_pill", "interior", "rim", "imprint_corr")}
                     | {"n": len(rows)}
                     for name, rows in subsets if rows},
         "records": records}, indent=1))
    print(f"\nanalysis -> {out_json}")

    # ---- figure grid: correct and incorrect side by side ------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks: list[tuple[str, int, bool]] = []
    for cls in classes[: args.grid_classes]:
        label = classes.index(cls)
        got_ok = [i for i in chosen if dataset.samples[i][1] == label and cache[i][2]]
        got_bad = [i for i in chosen
                   if dataset.samples[i][1] == label and not cache[i][2]]
        for pool, flag in ((got_ok, True), (got_bad, False)):
            if pool:
                picks.append((cls, pool[0], flag))

    ncol = 4
    nrow = int(np.ceil(len(picks) / (ncol // 2)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 2.5))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (cls, i, ok) in enumerate(picks):
        cam, rgb, _ = cache[i]
        r, c = divmod(k, ncol // 2)
        axes[r, c * 2].imshow(rgb)
        axes[r, c * 2].set_title(f"{cls[:20]}", fontsize=7)
        axes[r, c * 2 + 1].imshow(rgb)
        axes[r, c * 2 + 1].imshow(cam, cmap="jet", alpha=0.45)
        axes[r, c * 2 + 1].set_title("CAM  " + ("correct" if ok else "WRONG"),
                                     fontsize=7,
                                     color="green" if ok else "red")
    fig.suptitle(f"Grad-CAM - {tag} ({args.split}): correct and incorrect "
                 f"predictions per class", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = FIG_DIR / f"gradcam_{tag}_{args.split}.png"
    fig.savefig(out_fig, dpi=140)
    plt.close(fig)
    print(f"figure   -> {out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
