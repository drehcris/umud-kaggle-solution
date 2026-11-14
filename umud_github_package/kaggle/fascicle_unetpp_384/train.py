from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.15
MASK_DILATION_RADIUS = 2
PATIENCE = 6
NUM_WORKERS = 2
VIEWPORT_PREPROCESSING = True
THRESHOLDS = tuple(float(value) for value in np.arange(0.25, 0.91, 0.05))
ROOT = Path(
    "/kaggle/input/competitions/"
    "umud-challenge-muscle-architecture-in-ultrasound-data"
)
OUTPUT_DIR = Path("/kaggle/working")
CHECKPOINT_PATH = OUTPUT_DIR / "fasc_unetpp_r34_384.pt"
HISTORY_PATH = OUTPUT_DIR / "fasc_unetpp_r34_384_history.csv"
CONFIG_PATH = OUTPUT_DIR / "fasc_unetpp_r34_384_config.json"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "fasc_unetpp_r34_384_split.csv"
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def log(message: str) -> None:
    print(message, flush=True)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def import_segmentation_models_pytorch():
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        log("Installing segmentation-models-pytorch...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "segmentation-models-pytorch>=0.3.4,<0.6",
            ]
        )
        import segmentation_models_pytorch as smp
    return smp


def find_named_directory(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name!r} directory under {root}, found {matches}")
    return matches[0]


def find_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    masks = {
        path.stem: path
        for path in mask_dir.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    pairs = []
    for image_path in image_dir.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = masks.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return sorted(pairs, key=lambda pair: pair[0].stem)


def split_pairs(
    pairs: list[tuple[Path, Path]], val_fraction: float, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    groups: dict[str, list[tuple[Path, Path]]] = {}
    for pair in pairs:
        groups.setdefault(image_fingerprint(pair[0]), []).append(pair)
    group_keys = np.asarray(sorted(groups), dtype=object)
    np.random.default_rng(seed).shuffle(group_keys)
    target = max(1, round(len(pairs) * val_fraction))
    validation: list[tuple[Path, Path]] = []
    training: list[tuple[Path, Path]] = []
    for key in group_keys:
        destination = validation if len(validation) < target else training
        destination.extend(groups[str(key)])
    return training, validation


def image_fingerprint(path: Path) -> str:
    pixels = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype=np.int64).tobytes())
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def deduplicate_pairs(
    pairs: list[tuple[Path, Path]],
) -> tuple[list[tuple[Path, Path]], int]:
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[Path, Path]] = []
    for image_path, mask_path in pairs:
        key = (image_fingerprint(image_path), image_fingerprint(mask_path))
        if key in seen:
            continue
        seen.add(key)
        unique.append((image_path, mask_path))
    return unique, len(pairs) - len(unique)


def _largest_true_run(values: np.ndarray) -> tuple[int, int]:
    padded = np.pad(values.astype(np.int8), 1)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, len(values)
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def training_viewport(image: np.ndarray) -> tuple[int, int, int, int]:
    gray = image.astype(np.float32)
    vertical_texture = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    left, right = _largest_true_run(vertical_texture > 1.0)
    if right - left < image.shape[1] * 0.35:
        left, right = 0, image.shape[1]
    horizontal_texture = np.abs(np.diff(gray[:, left:right], axis=1)).mean(axis=1)
    top, bottom = _largest_true_run(horizontal_texture > 0.5)
    if bottom - top < image.shape[0] * 0.45:
        top, bottom = 0, image.shape[0]
    return left, top, right, bottom


def crop_aligned_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = training_viewport(image)
    image_height, image_width = image.shape
    mask_height, mask_width = mask.shape
    mask_left = round(left / image_width * mask_width)
    mask_right = round(right / image_width * mask_width)
    mask_top = round(top / image_height * mask_height)
    mask_bottom = round(bottom / image_height * mask_height)
    return image[top:bottom, left:right], mask[mask_top:mask_bottom, mask_left:mask_right]


class FascicleDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], augment: bool) -> None:
        self.pairs = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if VIEWPORT_PREPROCESSING:
            image, mask = crop_aligned_pair(image, mask)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.uint8)
        if MASK_DILATION_RADIUS:
            width = 2 * MASK_DILATION_RADIUS + 1
            mask = cv2.dilate(mask, np.ones((width, width), dtype=np.uint8), iterations=1)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = np.ascontiguousarray(image[None], dtype=np.float32)
        mask = np.ascontiguousarray(mask[None], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(mask)

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        if random.random() < 0.70:
            angle = random.uniform(-8.0, 8.0)
            scale = random.uniform(0.92, 1.08)
            center = (IMAGE_SIZE / 2, IMAGE_SIZE / 2)
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            matrix[:, 2] += (
                random.uniform(-0.04, 0.04) * IMAGE_SIZE,
                random.uniform(-0.04, 0.04) * IMAGE_SIZE,
            )
            image = cv2.warpAffine(
                image,
                matrix,
                (IMAGE_SIZE, IMAGE_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                mask,
                matrix,
                (IMAGE_SIZE, IMAGE_SIZE),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        gamma = random.uniform(0.70, 1.45)
        image = np.clip(image**gamma, 0.0, 1.0)
        image = np.clip(image * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08), 0, 1)
        if random.random() < 0.35:
            noise = np.random.normal(0.0, random.uniform(0.01, 0.035), image.shape)
            image = np.clip(image + noise.astype(np.float32), 0.0, 1.0)
        if random.random() < 0.20:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        return image, mask


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive_weight = torch.tensor([12.0], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
    return 0.35 * bce + 0.65 * soft_dice_loss(logits, targets)


@torch.inference_mode()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    totals = {
        threshold: {"dice": 0.0, "precision": 0.0, "recall": 0.0, "count": 0}
        for threshold in THRESHOLDS
    }
    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        probabilities = torch.sigmoid(model(images))
        for threshold in THRESHOLDS:
            predictions = probabilities > threshold
            target_binary = targets > 0.5
            intersection = (predictions & target_binary).sum(dim=(1, 2, 3)).float()
            predicted = predictions.sum(dim=(1, 2, 3)).float()
            actual = target_binary.sum(dim=(1, 2, 3)).float()
            batch_size = images.shape[0]
            totals[threshold]["dice"] += float(
                ((2 * intersection + 1) / (predicted + actual + 1)).sum()
            )
            totals[threshold]["precision"] += float(
                ((intersection + 1) / (predicted + 1)).sum()
            )
            totals[threshold]["recall"] += float(((intersection + 1) / (actual + 1)).sum())
            totals[threshold]["count"] += batch_size

    summarized = {
        threshold: {
            metric: values[metric] / values["count"]
            for metric in ("dice", "precision", "recall")
        }
        for threshold, values in totals.items()
    }
    best_threshold = max(THRESHOLDS, key=lambda value: summarized[value]["dice"])
    return {"threshold": best_threshold, **summarized[best_threshold]}


def build_model(smp, pretrained: bool = True) -> torch.nn.Module:
    weights = "imagenet" if pretrained else None
    try:
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=weights,
            in_channels=1,
            classes=1,
            activation=None,
        )
    except Exception as error:
        if not pretrained:
            raise
        log(f"Could not load ImageNet weights ({error!r}); training with random initialization")
        model = build_model(smp, pretrained=False)
    return model


def main() -> None:
    started = time.time()
    seed_everything(SEED)
    smp = import_segmentation_models_pytorch()

    if not ROOT.exists():
        raise FileNotFoundError(f"Competition data is not mounted at {ROOT}")
    image_dir = find_named_directory(ROOT, "fasc_images_new_model_v1")
    mask_dir = find_named_directory(ROOT, "fasc_masks_new_model_v1")
    pairs = find_pairs(image_dir, mask_dir)
    if len(pairs) < 100:
        raise RuntimeError(f"Only found {len(pairs)} matched fascicle pairs")
    original_count = len(pairs)
    pairs, duplicate_count = deduplicate_pairs(pairs)
    train_pairs, val_pairs = split_pairs(pairs, VAL_FRACTION, SEED)
    with SPLIT_MANIFEST_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "image", "mask"])
        split_lists = (("train", train_pairs), ("validation", val_pairs))
        for split_name, split_pairs_list in split_lists:
            for image_path, mask_path in split_pairs_list:
                writer.writerow([split_name, image_path.name, mask_path.name])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This kernel is configured for GPU training, but CUDA is unavailable")
    log(f"torch={torch.__version__} smp={smp.__version__}")
    log(f"device={torch.cuda.get_device_name(0)} cuda={torch.version.cuda}")
    log(
        f"pairs={original_count} unique={len(pairs)} duplicates_removed={duplicate_count} "
        f"train={len(train_pairs)} val={len(val_pairs)} size={IMAGE_SIZE}"
    )

    train_loader = DataLoader(
        FascicleDataset(train_pairs, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        FascicleDataset(val_pairs, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    base_model = build_model(smp).to(device, memory_format=torch.channels_last)
    model = base_model
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(base_model)
        log(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.03
    )
    scaler = torch.amp.GradScaler("cuda")
    best_dice = -1.0
    stale_epochs = 0

    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(
            ["epoch", "train_loss", "val_dice", "precision", "recall", "threshold", "lr"]
        )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        epoch_started = time.time()
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = combined_loss(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * images.shape[0]
            sample_count += images.shape[0]
            if step % 100 == 0:
                log(
                    f"epoch={epoch:02d}/{EPOCHS} step={step:03d}/{len(train_loader)} "
                    f"loss={running_loss / sample_count:.5f}"
                )

        metrics = validate(model, val_loader, device)
        train_loss = running_loss / sample_count
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_started
        log(
            f"EPOCH {epoch:02d}/{EPOCHS} train_loss={train_loss:.5f} "
            f"val_dice={metrics['dice']:.5f} precision={metrics['precision']:.5f} "
            f"recall={metrics['recall']:.5f} threshold={metrics['threshold']:.2f} "
            f"lr={current_lr:.2e} seconds={elapsed:.1f}"
        )
        with HISTORY_PATH.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [
                    epoch,
                    train_loss,
                    metrics["dice"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["threshold"],
                    current_lr,
                ]
            )

        if metrics["dice"] > best_dice + 1e-5:
            best_dice = metrics["dice"]
            stale_epochs = 0
            checkpoint = {
                "model": base_model.state_dict(),
                "arch": "UnetPlusPlus",
                "encoder": "resnet34",
                "encoder_weights": "imagenet",
                "size": IMAGE_SIZE,
                "task": "fascicle",
                "mask_dilate": MASK_DILATION_RADIUS,
                "threshold": metrics["threshold"],
                "val_dice": best_dice,
                "val_metrics": metrics,
                "seed": SEED,
                "viewport_preprocessing": VIEWPORT_PREPROCESSING,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            log(f"SAVED {CHECKPOINT_PATH.name} val_dice={best_dice:.5f}")
        else:
            stale_epochs += 1
            log(f"no improvement for {stale_epochs}/{PATIENCE} epochs")
        scheduler.step()
        if stale_epochs >= PATIENCE:
            log("Early stopping triggered")
            break

    config = {
        "best_val_dice": best_dice,
        "checkpoint": CHECKPOINT_PATH.name,
        "history": HISTORY_PATH.name,
        "split_manifest": SPLIT_MANIFEST_PATH.name,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "elapsed_seconds": time.time() - started,
        "competition_root": str(ROOT),
        "original_pair_count": original_count,
        "unique_pair_count": len(pairs),
        "duplicates_removed": duplicate_count,
        "viewport_preprocessing": VIEWPORT_PREPROCESSING,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "segmentation_models_pytorch_version": smp.__version__,
        "cuda_version": torch.version.cuda,
    }
    if CHECKPOINT_PATH.exists():
        config["checkpoint_sha256"] = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log(f"DONE best_val_dice={best_dice:.5f} elapsed_minutes={(time.time() - started) / 60:.1f}")


if __name__ == "__main__":
    main()
