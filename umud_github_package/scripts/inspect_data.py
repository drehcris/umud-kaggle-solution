from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def count_files(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


def first_image_shape(path: Path) -> tuple[int, int] | None:
    for item in sorted(path.rglob("*")):
        if item.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            with Image.open(item) as image:
                return image.size
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    print(f"Root: {root.resolve()}")
    for child in sorted(root.iterdir()):
        if child.is_dir():
            shape = first_image_shape(child)
            print(f"{child.name:24s} files={count_files(child):5d} first_size={shape}")
        else:
            print(f"{child.name:24s} size={child.stat().st_size}")

    sample = root / "sample_submission.csv"
    if sample.exists():
        df = pd.read_csv(sample, sep=None, engine="python", encoding="utf-8-sig")
        df.columns = [name.lstrip("\ufeff") for name in df.columns]
        print("\nSample submission:")
        print(df.head())
        print(df.describe(include="all"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
