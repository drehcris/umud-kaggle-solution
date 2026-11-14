from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from umud.data_split import deduplicate_pairs, grouped_split_pairs
from umud.torch_unet import ThinStructureLoss, UNet


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def find_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    masks_by_stem = {p.stem: p for p in mask_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    pairs = []
    for image_path in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


class MaskDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        size: int = 512,
        augment: bool = False,
        mask_dilate: int = 0,
    ) -> None:
        self.pairs = pairs
        self.size = size
        self.augment = augment
        self.mask_dilate = mask_dilate

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        x = np.asarray(image, dtype=np.float32) / 255.0
        y = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        if self.mask_dilate > 0:
            kernel_size = self.mask_dilate * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            y = cv2.dilate(y, kernel, iterations=1).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                x = np.fliplr(x).copy()
                y = np.fliplr(y).copy()
            if random.random() < 0.65:
                angle = random.uniform(-7.0, 7.0)
                scale = random.uniform(0.94, 1.06)
                tx = random.uniform(-0.03, 0.03) * self.size
                ty = random.uniform(-0.03, 0.03) * self.size
                matrix = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), angle, scale)
                matrix[:, 2] += (tx, ty)
                x = cv2.warpAffine(
                    x, matrix, (self.size, self.size), flags=cv2.INTER_LINEAR, borderValue=0
                )
                y = cv2.warpAffine(
                    y, matrix, (self.size, self.size), flags=cv2.INTER_NEAREST, borderValue=0
                )
            gamma = random.uniform(0.75, 1.35)
            x = np.clip(x**gamma, 0, 1)
            if random.random() < 0.35:
                x = np.clip(x + np.random.normal(0, 0.025, x.shape).astype(np.float32), 0, 1)

        return torch.from_numpy(x[None]), torch.from_numpy(y[None])


def split_pairs(pairs: list[tuple[Path, Path]], val_fraction: float, seed: int) -> tuple[list, list]:
    return grouped_split_pairs(pairs, val_fraction, seed)


@torch.no_grad()
def validate(model: UNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    thresholds = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    scores = {
        threshold: {"dice": [], "precision": [], "recall": []} for threshold in thresholds
    }
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        probs = torch.sigmoid(model(x))
        for threshold in thresholds:
            pred = (probs > threshold).float()
            inter = (pred * y).sum(dim=(1, 2, 3))
            predicted = pred.sum(dim=(1, 2, 3))
            actual = y.sum(dim=(1, 2, 3))
            scores[threshold]["dice"].extend(
                ((2 * inter + 1) / (predicted + actual + 1)).cpu().tolist()
            )
            scores[threshold]["precision"].extend(
                ((inter + 1) / (predicted + 1)).cpu().tolist()
            )
            scores[threshold]["recall"].extend(((inter + 1) / (actual + 1)).cpu().tolist())
    model.train()
    summarized = {
        threshold: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for threshold, metrics in scores.items()
    }
    best_threshold = max(thresholds, key=lambda threshold: summarized[threshold]["dice"])
    return {"threshold": best_threshold, **summarized[best_threshold]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["aponeurosis", "fascicle"], required=True)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=None,
        help="Target dilation radius; defaults to 0 for aponeurosis and 2 for fascicle.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit pairs for smoke tests; omit for normal training.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    pairs = find_pairs(args.image_dir, args.mask_dir)
    if args.max_samples is not None:
        if args.max_samples < 2:
            raise ValueError("--max-samples must be at least 2")
        indexes = np.linspace(0, len(pairs) - 1, min(args.max_samples, len(pairs)), dtype=int)
        pairs = [pairs[index] for index in indexes]
    if len(pairs) < 2:
        raise RuntimeError(f"Found only {len(pairs)} image/mask pairs")
    original_count = len(pairs)
    pairs, duplicate_count = deduplicate_pairs(pairs)
    train_pairs, val_pairs = split_pairs(pairs, args.val_fraction, args.seed)
    mask_dilate = args.mask_dilate
    if mask_dilate is None:
        mask_dilate = 0 if args.task == "aponeurosis" else 2
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(
        f"pairs={original_count} unique={len(pairs)} duplicates_removed={duplicate_count} "
        f"train={len(train_pairs)} val={len(val_pairs)} device={device}"
    )

    train_loader = DataLoader(
        MaskDataset(train_pairs, size=args.size, augment=True, mask_dilate=mask_dilate),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        MaskDataset(val_pairs, size=args.size, mask_dilate=mask_dilate),
        batch_size=args.batch_size,
    )

    model = UNet(base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.03
    )
    pos_weight = 6.0 if args.task == "aponeurosis" else 18.0
    loss_fn = ThinStructureLoss(pos_weight=pos_weight).to(device)
    best_dice = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".csv")
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(
            ["epoch", "loss", "val_dice", "val_precision", "val_recall", "threshold", "lr"]
        )

    for epoch in range(1, args.epochs + 1):
        losses = []
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = validate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch} loss={np.mean(losses):.5f} "
            f"val_dice={metrics['dice']:.5f} val_precision={metrics['precision']:.5f} "
            f"val_recall={metrics['recall']:.5f} threshold={metrics['threshold']:.2f} "
            f"lr={current_lr:.2e}"
        )
        with log_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [
                    epoch,
                    float(np.mean(losses)),
                    metrics["dice"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["threshold"],
                    current_lr,
                ]
            )
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "size": args.size,
                    "base": args.base,
                    "val_dice": best_dice,
                    "val_metrics": metrics,
                    "task": args.task,
                    "mask_dilate": mask_dilate,
                    "threshold": metrics["threshold"],
                    "image_dir": str(args.image_dir),
                    "mask_dir": str(args.mask_dir),
                },
                args.out,
            )
            print(f"saved {args.out}")
        scheduler.step()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
