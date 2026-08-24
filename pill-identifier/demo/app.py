"""Demo backend: upload a pill photo, get top-3 candidates or an honest refusal.

Design decisions that matter more than the code:

1. **Preprocessing is imported, never reimplemented.** Inference runs the exact
   training pipeline - src/localize.py to find the pill, the same square crop, the
   same eval-time transform from src/dataset.py. If serving preprocessing drifts
   from training preprocessing the model silently degrades, and a resized full
   frame puts the pill at ~35 px where accuracy collapses.

2. **The crop is returned to the caller.** The localiser detects the pill ~57% of
   the time and centre-crops the rest; when it crops the wrong thing the
   prediction is meaningless. Returning the 224px crop that was actually
   classified makes a bad crop visible instead of invisible.

3. **Rejection threshold is read from the measurement**, not hardcoded here.
   outputs/ood_rejection.json is produced by src/ood_rejection.py against ~100
   C3PI images from products outside the 20 classes.

4. **Prose is templated.** It is built only from labels, confidences and
   look-alike-group structure. It never asserts anything about what a drug does or
   how to take it, because the model has no such knowledge and a fluent wrong
   answer is the failure mode that matters here.

    uvicorn demo.app:app --reload --port 8000    # from the project root
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classes import CLASS_GROUPS, CLASS_TO_GROUP, SLUG_TO_LABEL  # noqa: E402
from dataset import build_transforms  # noqa: E402
from evaluate import load_model  # noqa: E402
from localize import locate, square_crop_box  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
CKPT = ROOT / "outputs" / "checkpoints" / "transfer_s1_best.pt"
OOD_JSON = ROOT / "outputs" / "ood_rejection.json"
METRICS = ROOT / "outputs" / "summary_seeds.csv"
SIZE = 224
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Pill Identifier (20-class demo)")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model, _classes, _ckpt = load_model(CKPT, _device)
_transform = build_transforms(train=False)


def _load_ood_config() -> dict:
    """Rejection threshold from the measured sweep; refuse to guess if absent."""
    if not OOD_JSON.exists():
        raise RuntimeError(
            f"{OOD_JSON} missing - run src/ood_rejection.py first. The demo will "
            f"not invent a confidence threshold.")
    d = json.loads(OOD_JSON.read_text())
    rec = d.get("recommended")
    if not rec:
        raise RuntimeError("ood_rejection.json has no 'recommended' block")
    return {
        "threshold": float(rec["threshold"]),
        "coverage": float(rec["coverage"]),
        "precision": float(rec["precision"]),
        "ood_rejected": float(rec["ood_rejected"]),
        "ood_still_named": float(rec["ood_still_named"]),
        "n_ood_images": int(d.get("n_ood_images", 0)),
        "n_ood_products": int(d.get("n_ood_products", 0)),
    }


OOD = _load_ood_config()


def _test_metrics() -> dict:
    out = {}
    if METRICS.exists():
        import csv

        for r in csv.DictReader(open(METRICS, encoding="utf8")):
            if r["phase"] == "transfer":
                out = {"top1_mean": float(r["top1_mean"]),
                       "top1_std": float(r["top1_std"]),
                       "top3_mean": float(r["top3_mean"]),
                       "n_seeds": int(r["n_seeds"])}
    return out


TEST_METRICS = _test_metrics()


def _same_family(top3: list[dict]) -> bool:
    """Are >=2 of the top-3 the same product family (e.g. one brand, two doses)?"""
    g = top3[0]["group"]
    return bool(g) and sum(1 for t in top3 if t["group"] == g) >= 2


def build_prose(top3: list[dict], rejected: bool) -> str:
    """Deterministic prose. Never states anything about the drug itself."""
    if rejected:
        msg = (
            f"No confident match. The best guess was {top3[0]['label']} at "
            f"{top3[0]['confidence']:.0%}, below the {OOD['threshold']:.0%} "
            f"threshold this demo requires.")
        if _same_family(top3):
            # the informative case: it knows the family but not the strength
            fam = [t for t in top3 if t["group"] == top3[0]["group"]]
            msg += (
                f" Note that its top candidates ({', '.join(t['label'] for t in fam)}) "
                f"are the same product at different strengths — so it likely has the "
                f"right product but cannot tell the strength apart. This model does "
                f"not reliably read imprints, which is what distinguishes them; "
                f"check the marking on the tablet against the pack.")
        else:
            msg += (
                f" That usually means either the pill is not one of the 20 this "
                f"model knows, or the photo was not clear enough.")
        return msg + " Check the crop below — if it is not centred on a single pill, retake the photo."

    best = top3[0]
    parts = [f"Best match: {best['label']} ({best['confidence']:.0%} confidence)."]

    if _same_family(top3):
        others = ", ".join(t["label"] for t in top3[1:])
        parts.append(
            f"The next candidates ({others}) are the same product at different "
            f"strengths. This model does not reliably read imprints, so it cannot "
            f"be trusted to tell those apart - check the marking on the tablet "
            f"against the pack.")
    else:
        others = ", ".join(f"{t['label']} ({t['confidence']:.0%})" for t in top3[1:])
        parts.append(f"Other possibilities: {others}.")
    parts.append(
        f"Measured accuracy on held-out photos: "
        f"{TEST_METRICS.get('top1_mean', float('nan')):.0f}% top-1 over "
        f"{TEST_METRICS.get('n_seeds', 0)} seeds. Not for clinical use.")
    return " ".join(parts)


@app.get("/api/info")
def info() -> JSONResponse:
    """What this model knows, and how well - so the UI never has to guess."""
    return JSONResponse({
        "n_classes": len(_classes),
        "classes": [{"slug": c, "label": SLUG_TO_LABEL.get(c, c),
                     "group": CLASS_TO_GROUP.get(c)} for c in _classes],
        "look_alike_groups": {g: [SLUG_TO_LABEL.get(s, s) for s in ss]
                              for g, ss in CLASS_GROUPS.items()},
        "checkpoint": CKPT.name,
        "val_top1": _ckpt.get("val_top1"),
        "test_metrics": TEST_METRICS,
        "rejection": OOD,
    })


@app.post("/api/identify")
async def identify(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large (max 20 MB)")

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400,
                            detail="could not decode image (expected JPEG/PNG)")

    # identical pipeline to training: localise -> square crop -> 224 -> normalise
    res = locate(img)
    box = square_crop_box(res, img.shape[1], img.shape[0])
    region = img[box[1]:box[3], box[0]:box[2]]
    if region.size == 0:
        raise HTTPException(status_code=422, detail="crop was empty")
    crop = cv2.resize(region, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    tensor = _transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    with torch.no_grad():
        probs = torch.softmax(_model(tensor.unsqueeze(0).to(_device)), dim=1)[0]
    probs = probs.cpu().numpy()

    order = np.argsort(-probs)[:3]
    top3 = [{"slug": _classes[i],
             "label": SLUG_TO_LABEL.get(_classes[i], _classes[i]),
             "group": CLASS_TO_GROUP.get(_classes[i]),
             "confidence": float(probs[i])} for i in order]
    rejected = bool(top3[0]["confidence"] < OOD["threshold"])

    ok, png = cv2.imencode(".png", crop)
    crop_b64 = base64.b64encode(png.tobytes()).decode() if ok else None

    return JSONResponse({
        "rejected": rejected,
        "threshold": OOD["threshold"],
        "top3": top3,
        "prose": build_prose(top3, rejected),
        "localisation": {
            "method": res.method,
            "note": ("pill detected" if res.method == "detected"
                     else "no pill found - fell back to a centre crop, so this "
                          "result is less reliable"),
            "box": list(box),
        },
        "crop_png_b64": crop_b64,
    })


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")
