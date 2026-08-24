"""Train Phase 1 (baseline CNN) or Phase 2 (transfer learning).

    python src/train.py --phase baseline --epochs 40
    python src/train.py --phase transfer --epochs 25

Top-1 is the headline metric (section 7). Top-3 is tracked as a secondary
"close enough for a lookup tool" signal and must be labelled as such.

Top-5 is deliberately NOT reported: with 15-30 classes it is 17-33% by chance,
so it conveys almost nothing. (Top-K is standard in the pill literature because
that work uses ~1,000 classes.)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_loaders, seed_everything  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"
HISTORY_DIR = ROOT / "outputs"


def topk_correct(logits: torch.Tensor, targets: torch.Tensor,
                 ks: tuple[int, ...] = (1, 3)) -> dict[int, int]:
    """Count correct predictions at each k, for this batch."""
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    hit = pred.eq(targets.view(-1, 1))
    return {k: int(hit[:, : min(k, maxk)].any(dim=1).sum().item()) for k in ks}


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device) -> dict[str, float]:
    model.eval()
    total = loss_sum = 0
    correct = {1: 0, 3: 0}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        for k, c in topk_correct(logits, y).items():
            correct[k] += c
        total += y.size(0)
    return {
        "loss": loss_sum / total,
        "top1": 100.0 * correct[1] / total,
        "top3": 100.0 * correct[3] / total,
    }


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict[str, float]:
    model.train()
    total = loss_sum = 0
    correct = {1: 0, 3: 0}
    # tqdm only when attached to a terminal - in redirected logs it emits one
    # line per update and drowns the epoch summaries
    bar = tqdm(loader, leave=False, desc="train",
               disable=not sys.stderr.isatty())
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        for k, c in topk_correct(logits, y).items():
            correct[k] += c
        total += y.size(0)
    return {
        "loss": loss_sum / total,
        "top1": 100.0 * correct[1] / total,
        "top3": 100.0 * correct[3] / total,
    }


def build_phase(phase: str, num_classes: int, args) -> tuple[nn.Module, list[dict]]:
    """Return the model plus a per-stage schedule of (epochs, lr, description)."""
    if phase == "baseline":
        from model_baseline import build_model

        model = build_model(num_classes)
        return model, [{"name": "scratch", "epochs": args.epochs, "lr": args.lr}]

    from model_transfer import build_model, set_trainable

    model = build_model(num_classes, backbone=args.backbone)
    # freeze backbone and train the head, then unfreeze the last blocks at a
    # lower lr (section 6)
    head_epochs = max(1, args.epochs // 3)
    return model, [
        {"name": "head-only", "epochs": head_epochs, "lr": args.lr,
         "unfreeze": 0},
        {"name": "finetune-last2", "epochs": args.epochs - head_epochs,
         "lr": args.lr / 10, "unfreeze": 2},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["baseline", "transfer"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train_loader, val_loader, test_loader, classes = build_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed,
    )
    num_classes = len(classes)
    model, schedule = build_phase(args.phase, num_classes, args)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    tag = args.tag or args.phase
    print(f"phase={args.phase} tag={tag} device={device} classes={num_classes}")
    print(f"params={n_params:,}  train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    # C3PI class counts are uneven (33-71 per class in train), so the section 5
    # weighted loss genuinely engages here
    counts = torch.tensor(train_loader.dataset.class_counts(), dtype=torch.float)
    weights = (counts.sum() / (len(counts) * counts)).to(device)
    if torch.allclose(weights, torch.ones_like(weights)):
        print("class counts balanced -> unweighted CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()
    else:
        print(f"class counts uneven -> weighted CrossEntropyLoss "
              f"(min {weights.min():.3f} max {weights.max():.3f})")
        criterion = nn.CrossEntropyLoss(weight=weights)

    history: list[dict] = []
    best = {"top1": -1.0, "top3": -1.0, "epoch": -1}
    ckpt_path = CKPT_DIR / f"{tag}_best.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    epoch = 0
    started = time.time()

    for stage in schedule:
        if args.phase == "transfer":
            from model_transfer import set_trainable

            set_trainable(model, unfreeze_last=stage["unfreeze"])
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train_p = sum(p.numel() for p in trainable)
        optimizer = torch.optim.AdamW(trainable, lr=stage["lr"],
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, stage["epochs"])
        )
        print(f"\n--- stage '{stage['name']}': {stage['epochs']} epochs, "
              f"lr={stage['lr']:.1e}, trainable params={n_train_p:,}")

        for _ in range(stage["epochs"]):
            epoch += 1
            t0 = time.time()
            tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            row = {
                "epoch": epoch, "stage": stage["name"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": tr["loss"], "train_top1": tr["top1"],
                "train_top3": tr["top3"],
                "val_loss": va["loss"], "val_top1": va["top1"],
                "val_top3": va["top3"],
                "secs": time.time() - t0,
            }
            history.append(row)
            flag = ""
            if va["top1"] > best["top1"]:
                best = {"top1": va["top1"], "top3": va["top3"], "epoch": epoch}
                torch.save(
                    {"model_state": model.state_dict(), "classes": classes,
                     "phase": args.phase, "backbone": getattr(args, "backbone", None),
                     "epoch": epoch, "seed": args.seed,
                     "val_top1": va["top1"], "val_top3": va["top3"]},
                    ckpt_path,
                )
                flag = "  <- best"
            print(f"e{epoch:03d} [{stage['name']:15s}] "
                  f"train {tr['loss']:.3f}/{tr['top1']:5.1f}%  "
                  f"val {va['loss']:.3f}/top1 {va['top1']:5.1f}%/top3 "
                  f"{va['top3']:5.1f}%  {row['secs']:.0f}s{flag}")

    elapsed = time.time() - started
    print(f"\nbest val top-1 {best['top1']:.1f}% "
          f"(top-3 {best['top3']:.1f}%, secondary) "
          f"at epoch {best['epoch']}  |  {elapsed / 60:.1f} min total")
    print(f"checkpoint -> {ckpt_path}")

    hist_path = HISTORY_DIR / f"history_{tag}.csv"
    with open(hist_path, "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (HISTORY_DIR / f"history_{tag}.json").write_text(
        json.dumps({"args": vars(args), "best": best,
                    "elapsed_min": elapsed / 60, "history": history}, indent=1)
    )
    print(f"history   -> {hist_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
