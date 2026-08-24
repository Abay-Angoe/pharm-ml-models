"""Locate the pill in a C3PI consumer photograph.

Why this is its own module: in C3PI_Test photos the pill spans roughly 10-15% of
frame width, sits off-centre, and the background is a real countertop or fabric
under uncontrolled lighting. Resizing the whole frame to 224 leaves a ~14 px pill
and destroys the imprint, which is the feature section 8 asks us to test for.

A naive "deviation from the median colour" blob detector was tried first and
failed: on vignetted countertops it locked onto the illumination gradient and
returned boxes covering 89-98% of the frame. The fix that matters is flattening
the illumination *before* thresholding, then scoring candidates on geometry
rather than trusting the largest blob.

Pipeline (all at reduced resolution for speed):
  1. flatten illumination - divide out a heavily blurred copy, which removes
     vignetting and soft shadows while keeping the pill's own edges
  2. build a score map from three complementary cues, because no single one
     covers both a red capsule on a white counter and a white tablet on paper:
       - colour residual after flattening
       - HSV saturation (catches coloured pills)
       - local gradient energy (catches pale pills, which still have a hard edge)
  3. Otsu threshold, then close/open to fuse imprint speckle into one blob
  4. score every connected component on plausibility and pick the best
  5. if nothing plausible survives, fall back to a centre crop and SAY SO -
     silent fallback would look identical to successful detection

`locate()` returns a LocateResult carrying the method used, so preprocess.py can
report exactly how many images were detected vs fell back.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_W = 640          # detection resolution
# Correctly-detected pills measured >=1.3% of frame area; every observed failure
# was a sub-0.5% background speck, so the floor is set above those.
MIN_AREA_FRAC = 0.005
# Correctly-detected pills measured 1.3-4.7% of frame area. Anything much larger
# is a bottle cap, a hand, or a merged pill+coin+shadow region, so cap well below
# the old 0.30 while leaving generous headroom.
MAX_AREA_FRAC = 0.12
# C3PI photographers centred the pill, and the centre-crop fallback is reliable
# because of it. So require a detection's centroid to lie in the central band -
# an off-centre blob is background clutter, and falling back beats trusting it.
MAX_CENTROID_OFFSET = 0.30  # fraction of half-width/height from centre
MAX_ASPECT = 5.0
MIN_FILL = 0.40        # component area / bbox area; pills are roughly convex
# Fallback crop side, as a fraction of the short edge. Audited crops show the
# pill comfortably inside 0.35 because C3PI framing is consistently centred;
# smaller than this starts clipping pills in the wider shots.
CENTER_FALLBACK_FRAC = 0.35
# Some C3PI photos include a coin as a size reference, placed beside the pill.
# A coin is rounder and higher-contrast than many tablets, so it wins on raw
# blob quality - but it sits off to one side while the pill is roughly centred.
# Centrality is therefore weighted hard, and edge-touching blobs are penalised.
CENTRALITY_POWER = 3.0
BORDER_PENALTY = 0.35
BORDER_FRAC = 0.04     # within 4% of an edge counts as touching


@dataclass
class LocateResult:
    x0: int
    y0: int
    x1: int
    y1: int
    method: str          # "detected" | "center_fallback"
    area_frac: float     # pill area as fraction of frame
    fill: float          # component area / bbox area
    score: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


def _flatten_illumination(bgr: np.ndarray) -> np.ndarray:
    """Divide out a large-sigma blur to kill vignetting and soft shadows."""
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=max(bgr.shape[:2]) / 12.0)
    flat = bgr.astype(np.float32) / (blur.astype(np.float32) + 1e-3)
    # centre on 1.0 -> deviation from local background
    return np.abs(flat - 1.0)


def _score_map(bgr: np.ndarray) -> np.ndarray:
    resid = _flatten_illumination(bgr).mean(axis=2)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0

    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), sigmaX=6.0)

    def norm(a: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    return norm(resid) + norm(sat) + norm(edge)


def _candidates(score: np.ndarray) -> list[tuple[float, tuple[int, int, int, int], float, float]]:
    u8 = (norm01(score) * 255).astype(np.uint8)
    _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(3, (score.shape[1] // 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape
    frame_area = float(h * w)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        area_frac = area / frame_area
        if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > MAX_ASPECT:
            continue
        fill = area / max(1, bw * bh)
        if fill < MIN_FILL:
            continue
        # a real pill does not span an entire edge of the frame
        if bw > 0.92 * w or bh > 0.92 * h:
            continue
        # prefer compact, central, decently sized blobs
        cx, cy = centroids[i]
        off_x = abs(cx - w / 2) / (w / 2)
        off_y = abs(cy - h / 2) / (h / 2)
        if max(off_x, off_y) > MAX_CENTROID_OFFSET:
            continue
        centrality = 1.0 - (off_x + off_y) / 2
        mean_score = float(score[y:y + bh, x:x + bw].mean())
        # linear in area (not sqrt) so a real pill outweighs a small speck
        quality = (fill
                   * (centrality ** CENTRALITY_POWER)
                   * mean_score
                   * area_frac)
        # penalise blobs hugging a frame edge - reference coins sit off to a side
        bx, by = BORDER_FRAC * w, BORDER_FRAC * h
        if x < bx or y < by or (x + bw) > w - bx or (y + bh) > h - by:
            quality *= BORDER_PENALTY
        out.append((quality, (int(x), int(y), int(x + bw), int(y + bh)),
                    float(area_frac), float(fill)))
    out.sort(reverse=True, key=lambda t: t[0])
    return out


def norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / max(hi - lo, 1e-6)


def locate(bgr_full: np.ndarray) -> LocateResult:
    """Find the pill. Never raises - falls back to a centre crop."""
    h0, w0 = bgr_full.shape[:2]
    scale = WORK_W / float(w0)
    small = cv2.resize(bgr_full, (WORK_W, max(1, int(round(h0 * scale)))),
                       interpolation=cv2.INTER_AREA)
    try:
        cands = _candidates(_score_map(small))
    except cv2.error:
        cands = []

    if cands:
        quality, (x0, y0, x1, y1), area_frac, fill = cands[0]
        inv = 1.0 / scale
        return LocateResult(int(x0 * inv), int(y0 * inv), int(x1 * inv),
                            int(y1 * inv), "detected", area_frac, fill, quality)

    side = int(min(w0, h0) * CENTER_FALLBACK_FRAC)
    cx, cy = w0 // 2, h0 // 2
    return LocateResult(cx - side // 2, cy - side // 2, cx + side // 2,
                        cy + side // 2, "center_fallback", float("nan"),
                        float("nan"), 0.0)


def square_crop_box(res: LocateResult, w: int, h: int,
                    margin: float = 1.6) -> tuple[int, int, int, int]:
    """Square box centred on the detection, padded, clamped to the frame."""
    bw, bh = res.x1 - res.x0, res.y1 - res.y0
    side = min(max(bw, bh) * margin, min(w, h))
    cx, cy = (res.x0 + res.x1) / 2.0, (res.y0 + res.y1) / 2.0
    cx = min(max(cx, side / 2), w - side / 2)
    cy = min(max(cy, side / 2), h - side / 2)
    return (int(round(cx - side / 2)), int(round(cy - side / 2)),
            int(round(cx + side / 2)), int(round(cy + side / 2)))
