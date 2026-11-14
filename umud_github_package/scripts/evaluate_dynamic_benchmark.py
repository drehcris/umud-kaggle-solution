from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

from predict_unet_geometry import load_model, predict_mask
from umud.geometry import architecture_from_masks, read_gray


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--apo-model", type=Path, required=True)
    parser.add_argument("--fasc-model", type=Path, required=True)
    parser.add_argument("--min-pennation", type=float, default=8.0)
    parser.add_argument("--apo-threshold", type=float)
    parser.add_argument("--fasc-threshold", type=float)
    parser.add_argument("--out", type=Path, default=Path("artifacts/dynamic_benchmark_eval.csv"))
    args = parser.parse_args()

    workbook = args.benchmark_dir / "benchmark_architecture_GM_calf_raise_v0.1.0.xlsx"
    pa = pd.read_excel(workbook, sheet_name="PA")[["frame", "Mean"]].rename(
        columns={"Mean": "expert_pa_deg"}
    )
    fl = pd.read_excel(workbook, sheet_name="FL")[["frame", "Mean"]].rename(
        columns={"Mean": "expert_fl_px"}
    )
    labels = pa.merge(fl, on="frame", how="inner").dropna()
    labels["frame"] = labels.frame.astype(int)

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
    for frame in tqdm(labels.frame.tolist()):
        image_path = args.benchmark_dir / "frames" / f"frame_{frame:05d}.png"
        image = read_gray(image_path)
        apo_mask = predict_mask(apo_model, image_path, apo_size, apo_threshold, device)
        fasc_mask = predict_mask(fasc_model, image_path, fasc_size, fasc_threshold, device)
        architecture = architecture_from_masks(
            image,
            apo_mask,
            fasc_mask,
            pixel_spacing_mm=1.0,
            min_pennation_deg=args.min_pennation,
        )
        rows.append(
            {
                "frame": frame,
                "model_pa_deg": np.nan if architecture is None else architecture.pa_deg,
                "model_fl_px": np.nan if architecture is None else architecture.fl_mm,
            }
        )

    output = labels.merge(pd.DataFrame(rows), on="frame").sort_values("frame")
    for target in ("pa_deg", "fl_px"):
        column = f"model_{target}"
        output[column] = output[column].interpolate(limit_direction="both")
        median = output[column].rolling(5, center=True, min_periods=1).median()
        output[f"filtered_{target}"] = savgol_filter(median, 11, 2, mode="interp")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    valid_count = int(pd.DataFrame(rows).model_pa_deg.notna().sum())
    print(f"Valid model rows: {valid_count}/{len(output)}")
    for target, unit in (("pa_deg", "deg"), ("fl_px", "px")):
        expert = output[f"expert_{target}"]
        for prefix in ("model", "filtered"):
            predicted = output[f"{prefix}_{target}"]
            mae = float((predicted - expert).abs().mean())
            correlation = float(predicted.corr(expert))
            print(f"{prefix:8s} {target}: MAE={mae:.3f} {unit}, r={correlation:.3f}")
        slope, intercept = np.polyfit(output[f"filtered_{target}"], expert, 1)
        print(
            f"  filtered {target} calibration: slope={slope:.10f} "
            f"intercept={intercept:.10f} {unit}"
        )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
