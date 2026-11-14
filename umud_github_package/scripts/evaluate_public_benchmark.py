from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from predict_unet_geometry import load_model, predict_mask
from umud.geometry import architecture_from_masks, read_gray


TOLERANCES = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def expert_mean(frame: pd.DataFrame, suffix: str) -> pd.Series:
    columns = [column for column in frame.columns if column.startswith("R") and column.endswith(suffix)]
    return frame[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def normalized_mae(frame: pd.DataFrame, prefix: str) -> tuple[float, dict[str, float]]:
    errors = {}
    for target, tolerance in TOLERANCES.items():
        error = (frame[f"{prefix}_{target}"] - frame[f"expert_{target}"]).abs().mean()
        errors[target] = float(error)
    score = float(np.mean([errors[target] / tolerance for target, tolerance in TOLERANCES.items()]))
    return score, errors


def leave_one_out_calibration(
    frame: pd.DataFrame, prefix: str
) -> tuple[float, dict[str, float], dict[str, tuple[float, float]]]:
    errors = {}
    coefficients = {}
    for target, tolerance in TOLERANCES.items():
        predicted = frame[f"{prefix}_{target}"].to_numpy(dtype=float)
        expert = frame[f"expert_{target}"].to_numpy(dtype=float)
        calibrated = np.empty_like(predicted)
        for index in range(len(frame)):
            keep = np.arange(len(frame)) != index
            slope, intercept = np.polyfit(predicted[keep], expert[keep], 1)
            calibrated[index] = slope * predicted[index] + intercept
        errors[target] = float(np.mean(np.abs(calibrated - expert)))
        slope, intercept = np.polyfit(predicted, expert, 1)
        coefficients[target] = (float(slope), float(intercept))
    score = float(
        np.mean([errors[target] / tolerance for target, tolerance in TOLERANCES.items()])
    )
    return score, errors, coefficients


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--apo-model", type=Path, required=True)
    parser.add_argument("--fasc-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/public_benchmark_eval.csv"))
    parser.add_argument("--min-pennation", type=float, default=10.0)
    parser.add_argument("--max-pennation", type=float, default=40.0)
    parser.add_argument("--apo-threshold", type=float)
    parser.add_argument("--fasc-threshold", type=float)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    workbook = args.benchmark_dir / "Results_benchmark_architecture_v0.1.0.xlsx"
    labels = pd.read_excel(workbook, sheet_name="Manual_architecture")
    labels["expert_mt_mm"] = expert_mean(labels, "_MT")
    labels["expert_fl_mm"] = expert_mean(labels, "_FL")
    labels["expert_pa_deg"] = expert_mean(labels, "_PA")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    apo_model, apo_size, apo_threshold = load_model(args.apo_model, device)
    fasc_model, fasc_size, fasc_threshold = load_model(args.fasc_model, device)
    apo_threshold = args.apo_threshold if args.apo_threshold is not None else apo_threshold
    fasc_threshold = (
        args.fasc_threshold if args.fasc_threshold is not None else fasc_threshold
    )

    rows = []
    for row in tqdm(labels.itertuples(index=False), total=len(labels)):
        image_path = args.benchmark_dir / f"{row.ImageID}.tif"
        image = read_gray(image_path)
        apo_mask = predict_mask(
            apo_model, image_path, apo_size, apo_threshold, device, args.tta
        )
        fasc_mask = predict_mask(
            fasc_model, image_path, fasc_size, fasc_threshold, device, args.tta
        )
        architecture = architecture_from_masks(
            image,
            apo_mask,
            fasc_mask,
            pixel_spacing_mm=10.0 / float(row.Scale_pixel_per_cm),
            min_pennation_deg=args.min_pennation,
            max_pennation_deg=args.max_pennation,
        )
        prediction = (
            {"pa_deg": np.nan, "fl_mm": np.nan, "mt_mm": np.nan}
            if architecture is None
            else {
                "pa_deg": architecture.pa_deg,
                "fl_mm": architecture.fl_mm,
                "mt_mm": architecture.mt_mm,
            }
        )
        rows.append({"image_id": row.ImageID, **prediction})

    predictions = pd.DataFrame(rows)
    output = labels.merge(predictions, left_on="ImageID", right_on="image_id")
    output = output.rename(
        columns={
            "pa_deg": "model_pa_deg",
            "fl_mm": "model_fl_mm",
            "mt_mm": "model_mt_mm",
            "SMA_PA": "sma_pa_deg",
            "SMA_FL": "sma_fl_mm",
            "SMA_MT": "sma_mt_mm",
            "DLTrack_PA": "dltrack_pa_deg",
            "DLTrack_FL": "dltrack_fl_mm",
            "DLTrack_MT": "dltrack_mt_mm",
        }
    )
    keep = [
        "image_id",
        "expert_pa_deg",
        "expert_fl_mm",
        "expert_mt_mm",
        "model_pa_deg",
        "model_fl_mm",
        "model_mt_mm",
        "sma_pa_deg",
        "sma_fl_mm",
        "sma_mt_mm",
        "dltrack_pa_deg",
        "dltrack_fl_mm",
        "dltrack_mt_mm",
    ]
    output = output[keep]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)

    print(f"Valid model rows: {output.model_pa_deg.notna().sum()}/{len(output)}")
    for prefix in ("model", "sma", "dltrack"):
        valid = output.dropna(subset=[f"{prefix}_{target}" for target in TOLERANCES])
        score, errors = normalized_mae(valid, prefix)
        print(
            f"{prefix:7s} normalized MAE={score:.4f} | "
            f"PA={errors['pa_deg']:.3f} deg, FL={errors['fl_mm']:.3f} mm, "
            f"MT={errors['mt_mm']:.3f} mm (n={len(valid)})"
        )
        if prefix == "model":
            loo_score, loo_errors, coefficients = leave_one_out_calibration(valid, prefix)
            print(
                f"model   LOOCV calibrated={loo_score:.5f} | "
                f"PA={loo_errors['pa_deg']:.3f} deg, FL={loo_errors['fl_mm']:.3f} mm, "
                f"MT={loo_errors['mt_mm']:.3f} mm"
            )
            for target, (slope, intercept) in coefficients.items():
                print(
                    f"  {target} calibration: slope={slope:.10f} "
                    f"intercept={intercept:.10f}"
                )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
