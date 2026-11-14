from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from umud.postprocess import finalize_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn raw UMUD geometry diagnostics into a Kaggle submission."
    )
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--benchmark-calibration",
        action="store_true",
        help="Opt in to experimental calibration fitted on the external expert benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.diagnostics)
    sample = pd.read_csv(args.sample, sep=";", encoding="utf-8-sig")
    final = finalize_predictions(raw, calibrated=args.benchmark_calibration)

    targets = ["pa_deg", "fl_mm", "mt_mm"]
    expected_columns = ["image_id", *targets]
    if sample.columns.tolist() != expected_columns:
        raise SystemExit(f"Unexpected sample columns: {sample.columns.tolist()}")
    submission = final[expected_columns]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False, lineterminator="\r\n")
    print(f"Wrote {len(submission)} predictions to {args.out}")


if __name__ == "__main__":
    main()
