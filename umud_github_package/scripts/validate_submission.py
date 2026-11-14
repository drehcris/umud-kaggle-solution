from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from umud.postprocess import BOUNDS


COLUMNS = ["image_id", "pa_deg", "fl_mm", "mt_mm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a UMUD Kaggle submission CSV.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_header = args.submission.read_bytes().splitlines()[0].decode("utf-8-sig")
    if raw_header != ",".join(COLUMNS):
        raise SystemExit(
            "Invalid submission:\n- Kaggle expects a comma-separated header "
            f"{','.join(COLUMNS)!r}, got {raw_header!r}"
        )
    submission = pd.read_csv(args.submission)
    expected_ids = sorted(
        path.name
        for path in args.test_dir.iterdir()
        if path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    )

    errors: list[str] = []
    if submission.columns.tolist() != COLUMNS:
        errors.append(f"columns must be {COLUMNS}, got {submission.columns.tolist()}")
    if len(submission) != len(expected_ids):
        errors.append(f"row count must be {len(expected_ids)}, got {len(submission)}")
    if submission.image_id.duplicated().any():
        errors.append("image_id contains duplicates")
    if submission.image_id.tolist() != expected_ids:
        errors.append("image_id values or order do not match the sorted test images")

    for target in COLUMNS[1:]:
        values = pd.to_numeric(submission[target], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            errors.append(f"{target} contains missing or non-finite values")
            continue
        lower, upper = BOUNDS[target]
        if ((values < lower) | (values > upper)).any():
            errors.append(f"{target} contains values outside [{lower}, {upper}]")

    if errors:
        raise SystemExit("Invalid submission:\n- " + "\n- ".join(errors))
    print(f"Valid submission: {len(submission)} rows, exact IDs, finite bounded targets")


if __name__ == "__main__":
    main()
