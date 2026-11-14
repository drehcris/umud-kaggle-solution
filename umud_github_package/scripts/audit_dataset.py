from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def dimensions(paths: list[Path]) -> Counter[str]:
    sizes: Counter[str] = Counter()
    for path in paths:
        with Image.open(path) as image:
            sizes[f"{image.width}x{image.height}"] += 1
    return sizes


def mask_stats(paths: list[Path]) -> dict[str, float]:
    occupancies = []
    components = []
    for path in paths:
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L")) > 0
        occupancies.append(float(mask.mean()))
        count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        components.append(max(0, count - 1))
    return {
        "occupancy_mean": float(np.mean(occupancies)),
        "occupancy_p05": float(np.percentile(occupancies, 5)),
        "occupancy_p95": float(np.percentile(occupancies, 95)),
        "components_median": float(np.median(components)),
        "components_p95": float(np.percentile(components, 95)),
    }


def paired_paths(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    masks = {path.stem: path for path in image_files(mask_dir)}
    return [(path, masks[path.stem]) for path in image_files(image_dir) if path.stem in masks]


def normalize_gray(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = np.percentile(array, [0.5, 99.5])
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def panel(image_path: Path, mask_path: Path, width: int = 420, height: int = 280) -> Image.Image:
    with Image.open(image_path) as source:
        gray = normalize_gray(source)
    with Image.open(mask_path) as source:
        mask = (np.asarray(source.convert("L")) > 0).astype(np.uint8) * 255

    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay = base.copy()
    overlay[mask > 0] = (255, 55, 55)
    overlay = cv2.addWeighted(base, 0.68, overlay, 0.32, 0)

    result = Image.fromarray(np.concatenate([base, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), overlay], axis=1))
    canvas = Image.new("RGB", (result.width, result.height + 28), "white")
    canvas.paste(result, (0, 28))
    ImageDraw.Draw(canvas).text((8, 7), image_path.name, fill="black")
    return canvas


def contact_sheet(pairs: list[tuple[Path, Path]], out: Path, rows: int = 5) -> None:
    if not pairs:
        return
    indexes = np.linspace(0, len(pairs) - 1, min(rows, len(pairs)), dtype=int)
    panels = [panel(*pairs[index]) for index in indexes]
    sheet = Image.new("RGB", (panels[0].width, sum(item.height for item in panels)), "white")
    y = 0
    for item in panels:
        sheet.paste(item, (0, y))
        y += item.height
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def image_contact_sheet(paths: list[Path], out: Path, rows: int = 10) -> None:
    if not paths:
        return
    indexes = np.linspace(0, len(paths) - 1, min(rows, len(paths)), dtype=int)
    tiles = []
    for index in indexes:
        path = paths[index]
        with Image.open(path) as source:
            gray = normalize_gray(source)
        gray = cv2.resize(gray, (420, 280), interpolation=cv2.INTER_AREA)
        tile = Image.new("RGB", (420, 308), "white")
        tile.paste(Image.fromarray(gray).convert("RGB"), (0, 28))
        ImageDraw.Draw(tile).text((8, 7), path.name, fill="black")
        tiles.append(tile)
    sheet = Image.new("RGB", (840, ((len(tiles) + 1) // 2) * 308), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 420, (index // 2) * 308))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit UMUD image/mask dimensions and label sparsity.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/audit"))
    args = parser.parse_args()

    collections = {
        "aponeurosis": (
            args.root / "apo_imgs_v1" / "apo_images_new_model_v1",
            args.root / "apo_masks_v1" / "apo_masks_new_model_v1",
        ),
        "fascicle": (
            args.root / "fasc_imgs_v1" / "fasc_images_new_model_v1",
            args.root / "fasc_masks_v1" / "fasc_masks_new_model_v1",
        ),
    }
    report: dict[str, object] = {}
    for name, (image_dir, mask_dir) in collections.items():
        images = image_files(image_dir)
        masks = image_files(mask_dir)
        pairs = paired_paths(image_dir, mask_dir)
        report[name] = {
            "images": len(images),
            "masks": len(masks),
            "pairs": len(pairs),
            "unmatched_images": sorted({p.stem for p in images} - {p.stem for p in masks}),
            "image_dimensions": dict(dimensions(images).most_common()),
            "mask_dimensions": dict(dimensions(masks).most_common()),
            "mask_stats": mask_stats(masks),
        }
        contact_sheet(pairs, args.out_dir / f"{name}_pairs.png")

    test_dir = args.root / "test_images_v2" / "test_set_v2"
    tests = image_files(test_dir)
    report["test"] = {"images": len(tests), "dimensions": dict(dimensions(tests).most_common())}
    image_contact_sheet(tests, args.out_dir / "test_images.png")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
