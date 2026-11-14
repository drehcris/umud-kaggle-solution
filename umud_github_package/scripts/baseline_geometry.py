from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from umud.geometry import classical_fascicle_orientation, read_gray


DEFAULTS = {
    "pa_deg": 15.0,
    "fl_mm": 75.0,
    "mt_mm": 20.0,
}


def find_test_dir(root: Path) -> Path:
    candidates = [
        root / "test_images_v2" / "test_set_v2",
        root / "test_images",
        root / "test",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = [p for p in root.rglob("*") if p.is_dir() and "test" in p.name.lower()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find test image directory under {root}")


def list_submission_images(root: Path, test_dir: Path) -> list[str]:
    sample = root / "sample_submission.csv"
    actual_images = sorted(
        p.name for p in test_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png"}
    )
    if sample.exists():
        sample_df = pd.read_csv(sample, sep=None, engine="python", encoding="utf-8-sig")
        sample_df.columns = [name.lstrip("\ufeff") for name in sample_df.columns]
        sample_ids = sample_df["image_id"].astype(str).tolist()
        if len(sample_ids) >= len(actual_images):
            return sample_ids
    return actual_images


def heuristic_prediction(image_path: Path) -> dict[str, float]:
    image = read_gray(image_path)
    angle = abs(classical_fascicle_orientation(image))
    pa_deg = float(np.clip(angle, 8.0, 28.0))

    # Until scale extraction/segmentation is trained, use physiologically plausible
    # relationships. This is deliberately conservative for formatting/baseline use.
    mt_mm = float(np.clip(DEFAULTS["mt_mm"] * (image.shape[0] / max(image.shape[1], 1)) / 0.65, 12, 32))
    fl_mm = float(np.clip(mt_mm / max(np.sin(np.radians(pa_deg)), 1e-3), 45, 130))
    return {"pa_deg": pa_deg, "fl_mm": fl_mm, "mt_mm": mt_mm}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/submission_baseline.csv"))
    args = parser.parse_args()

    test_dir = find_test_dir(args.root)
    image_ids = list_submission_images(args.root, test_dir)
    rows = []
    for image_id in tqdm(image_ids):
        image_path = test_dir / image_id
        if image_path.exists():
            pred = heuristic_prediction(image_path)
        else:
            pred = DEFAULTS.copy()
        rows.append({"image_id": image_id, **pred})

    output = pd.DataFrame(rows, columns=["image_id", "pa_deg", "fl_mm", "mt_mm"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False, lineterminator="\r\n")
    print(f"Wrote {args.out} with {len(output)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
