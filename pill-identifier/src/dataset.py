"""Datasets and loaders for the 20-class C3PI pill subset.

Reads the 224x224 crops written by preprocess.py, using data/processed/manifest.csv
as the index so the split assignment and the localisation method travel with each
sample.

Two things differ from a generic image-folder setup and matter downstream:

  * `methods` exposes whether each crop came from a successful pill detection or
    from the centre-crop fallback. evaluate.py slices accuracy by it, so a drop
    caused by weak localisation cannot be mistaken for a modelling result.
  * augmentation is deliberately conservative - see build_transforms().

Note on what C3PI does not provide: unlike a controlled-capture set, there are no
paired lighting conditions and no segmentation polygons. So the "engravings
visible vs washed out" A/B test is unavailable here, and Grad-CAM masks are
estimated from the image rather than read from ground truth.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MANIFEST = PROCESSED / "manifest.csv"

# ImageNet statistics - required by the pretrained Phase 2 backbones, and used for
# Phase 1 too so both phases see identical inputs.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224


def build_transforms(train: bool) -> transforms.Compose:
    """Augmentation is deliberately conservative (spec section 5).

    Section 8 tests whether the model reads the pill's imprint, so augmentation
    that destroys imprint legibility would train the model to ignore the exact
    feature we then measure - and quietly guarantee a null result.

      rotation  full circle    - pills are photographed at arbitrary angles and
                                 imprints stay legible under rotation
      flips     NONE           - a mirrored imprint does not exist on any real
                                 pill. Flipping teaches the model that reversed
                                 text is valid, which is actively harmful here
                                 even though flips are standard elsewhere.
      blur      very mild only - phone photos are soft, but heavy blur erases
                                 imprint text
      jitter    modest         - lighting varies; colour is a real cue, just not
                                 the only one we want the model to use
    """
    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        # (-180, +180) degrees = the full circle
        transforms.RandomRotation(180, expand=False, fill=(128, 128, 128)),
        # NO RandomHorizontalFlip / RandomVerticalFlip - see docstring
        transforms.RandomResizedCrop(SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.10),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class PillDataset(Dataset):
    def __init__(self, split: str, train_aug: bool | None = None,
                 classes: list[str] | None = None) -> None:
        if not MANIFEST.exists():
            raise FileNotFoundError(
                f"{MANIFEST} missing - run src/preprocess.py first")
        rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf8"))
                if r["split"] == split]
        if not rows:
            raise RuntimeError(f"no rows for split={split!r} in {MANIFEST}")

        self.split = split
        self.classes = classes or sorted({r["class"] for r in rows})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int, str]] = []
        for r in rows:
            if r["class"] not in self.class_to_idx:
                continue
            path = PROCESSED / split / r["class"] / f"{r['stem']}.png"
            if path.exists():
                self.samples.append(
                    (path, self.class_to_idx[r["class"]], r["method"]))
        if not self.samples:
            raise RuntimeError(f"no crop files found for split={split!r}")

        self.transform = build_transforms(
            split == "train" if train_aug is None else train_aug)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label, _ = self.samples[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), label

    @property
    def methods(self) -> list[str]:
        """"detected" or "center_fallback", per sample."""
        return [s[2] for s in self.samples]

    def class_counts(self) -> list[int]:
        counts = [0] * len(self.classes)
        for _, label, _ in self.samples:
            counts[label] += 1
        return counts


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(batch_size: int = 32, num_workers: int = 0, seed: int = 42,
                  ) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    seed_everything(seed)
    train = PillDataset("train")
    val = PillDataset("valid", classes=train.classes)
    test = PillDataset("test", classes=train.classes)

    generator = torch.Generator().manual_seed(seed)
    common = {"num_workers": num_workers, "pin_memory": False}
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   drop_last=False, generator=generator, **common),
        DataLoader(val, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test, batch_size=batch_size, shuffle=False, **common),
        train.classes,
    )


if __name__ == "__main__":
    tr, va, te, classes = build_loaders()
    print(f"classes: {len(classes)}")
    for name, loader in (("train", tr), ("valid", va), ("test", te)):
        ds: PillDataset = loader.dataset  # type: ignore[assignment]
        counts = ds.class_counts()
        m = ds.methods
        print(f"  {name:6s}: {len(ds):5d} imgs | per-class {min(counts)}-{max(counts)}"
              f" | detected {m.count('detected')} "
              f"fallback {m.count('center_fallback')}")
    x, y = next(iter(tr))
    print(f"\nbatch: {tuple(x.shape)} {x.dtype}  labels {tuple(y.shape)}")
    print(f"value range after normalise: [{x.min():.2f}, {x.max():.2f}]")
    print(f"first 5 classes: {classes[:5]}")
