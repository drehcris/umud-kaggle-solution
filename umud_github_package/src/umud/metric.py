"""Official UMUD score (kaggle.com/code/paulritsche/umud-score, v9).

score = mean_c(MAE_c / tau_c) + 1e-6 * mean_c(MedAE_c / tau_c) + 1e-9 * mean_c(RMSE_c / tau_c)
with tau = PA 6 deg, FL 12 mm, MT 3 mm, every image weighted equally.

``nan_policy="omit"`` drops NaN pairs per component (useful for benchmarks with missing rater
values); the official metric has no NaNs (``"raise"``).
"""
from __future__ import annotations

import numpy as np

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
COMPONENTS = tuple(TOL)


def _component(pred, true, tau: float, nan_policy: str) -> dict:
    p = np.asarray(pred, float)
    t = np.asarray(true, float)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {t.shape}")
    ok = np.isfinite(p) & np.isfinite(t)
    if not ok.all():
        if nan_policy == "raise":
            raise ValueError("NaN in predictions or targets")
        if nan_policy == "propagate":
            return dict(mae=np.nan, medae=np.nan, rmse=np.nan, bias=np.nan, norm=np.nan, n=int(ok.sum()))
    e = p[ok] - t[ok]
    if e.size == 0:
        return dict(mae=np.nan, medae=np.nan, rmse=np.nan, bias=np.nan, norm=np.nan, n=0)
    mae = float(np.mean(np.abs(e)))
    return dict(mae=mae, medae=float(np.median(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                bias=float(np.mean(e)), norm=mae / tau, n=int(e.size))


def score_arrays(pred: dict, true: dict, nan_policy: str = "raise", report: bool = False):
    """pred/true: dicts with keys pa_deg, fl_mm, mt_mm -> 1-D arrays (aligned by position)."""
    rows = {c: _component(pred[c], true[c], TOL[c], nan_policy) for c in COMPONENTS}
    p = np.mean([rows[c]["mae"] / TOL[c] for c in COMPONENTS])
    s = np.mean([rows[c]["medae"] / TOL[c] for c in COMPONENTS])
    r = np.mean([rows[c]["rmse"] / TOL[c] for c in COMPONENTS])
    sc = float(p + 1e-6 * s + 1e-9 * r)
    return (sc, rows) if report else sc


def umud_score(solution, submission, id_col: str = "image_id", nan_policy: str = "raise",
               report: bool = False):
    """DataFrame interface mirroring the Kaggle metric (merge on id_col)."""
    m = solution.merge(submission, on=id_col, suffixes=("_true", "_pred"))
    if len(m) != len(solution):
        raise ValueError(f"{len(solution) - len(m)} solution rows have no prediction")
    pred = {c: m[f"{c}_pred"].to_numpy(float) for c in COMPONENTS}
    true = {c: m[f"{c}_true"].to_numpy(float) for c in COMPONENTS}
    return score_arrays(pred, true, nan_policy, report)


def per_image_score(pred: dict, true: dict) -> np.ndarray:
    """Per-image contribution (|e_pa|/6 + |e_fl|/12 + |e_mt|/3) / 3; its mean is the main term."""
    return np.mean([np.abs(np.asarray(pred[c], float) - np.asarray(true[c], float)) / TOL[c]
                    for c in COMPONENTS], axis=0)
