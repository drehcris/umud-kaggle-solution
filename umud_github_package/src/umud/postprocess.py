from __future__ import annotations

import numpy as np
import pandas as pd


STATIC_CALIBRATION = {
    "pa_deg": (0.9139536649, 2.1737963972),
    "fl_mm": (0.6038728923, 16.7575162857),
    "mt_mm": (1.0879958442, -2.7146749201),
}

DYNAMIC_CALIBRATION = {
    "fl_mm": (1.0066077201, -96.3263701095),
}

DEFAULTS = {"pa_deg": 18.3, "fl_mm": 60.8, "mt_mm": 18.1}
BOUNDS = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def apply_benchmark_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    """Calibrate raw geometry with held-out public UMUD expert benchmarks."""

    result = frame.copy()
    dynamic = result["is_dynamic"].astype(bool)
    static = ~dynamic

    for target, (slope, intercept) in STATIC_CALIBRATION.items():
        result.loc[static, target] = slope * result.loc[static, target] + intercept

    # The public dynamic benchmark contains one exercise sequence. Its PA offset
    # does not transfer to the competition's diverse clips, so preserve raw PA.
    fl_slope, fl_intercept_px = DYNAMIC_CALIBRATION["fl_mm"]
    result.loc[dynamic, "fl_mm"] = (
        fl_slope * result.loc[dynamic, "fl_mm"]
        + fl_intercept_px * result.loc[dynamic, "pixel_spacing_mm"]
    )
    mt_slope, mt_intercept = STATIC_CALIBRATION["mt_mm"]
    result.loc[dynamic, "mt_mm"] = mt_slope * result.loc[dynamic, "mt_mm"] + mt_intercept
    return result


def _smooth_dynamic_runs(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dynamic_positions = np.flatnonzero(result["is_dynamic"].to_numpy(dtype=bool))
    if not len(dynamic_positions):
        return result

    run_starts = np.r_[0, np.flatnonzero(np.diff(dynamic_positions) > 1) + 1]
    run_ends = np.r_[run_starts[1:], len(dynamic_positions)]
    for start, end in zip(run_starts, run_ends):
        positions = dynamic_positions[start:end]
        for group_start in range(0, len(positions), 5):
            group = positions[group_start : group_start + 5]
            for target in ("pa_deg", "fl_mm", "mt_mm"):
                values = result.iloc[group][target]
                smoothed = values.rolling(3, center=True, min_periods=1).median()
                result.iloc[group, result.columns.get_loc(target)] = smoothed.to_numpy()
    return result


def finalize_predictions(frame: pd.DataFrame, calibrated: bool = False) -> pd.DataFrame:
    result = apply_benchmark_calibration(frame) if calibrated else frame.copy()
    result = _smooth_dynamic_runs(result)

    for target, default in DEFAULTS.items():
        dynamic_medians = result.loc[result.is_dynamic, target].median()
        static_medians = result.loc[~result.is_dynamic, target].median()
        result.loc[result.is_dynamic, target] = result.loc[result.is_dynamic, target].fillna(
            dynamic_medians if np.isfinite(dynamic_medians) else default
        )
        result.loc[~result.is_dynamic, target] = result.loc[~result.is_dynamic, target].fillna(
            static_medians if np.isfinite(static_medians) else default
        )
        lower, upper = BOUNDS[target]
        result[target] = result[target].clip(lower, upper)
    return result
