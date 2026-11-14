"""Stress-test geometry on narrower fields of view, without neural-network inference.

Crop the SAME GT / out-of-fold probability maps, without stretching coordinates.
Full-view GT-mask geometry is the reference. This isolates sensitivity of geometry
to missing lateral context; it does NOT simulate re-running a network on a crop,
and mask-derived targets are NOT expert measurements or a Kaggle score.
Images are processed sequentially. No torch, calibration, or test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umud import geometry_v2 as g2
from umud.paths import RESEARCH_ROOT


def central_window(width: int, fraction: float) -> slice:
    if width < 1 or not 0 < fraction <= 1:
        raise ValueError("width must be positive and fraction must be in (0, 1]")
    kept = max(1, round(width * fraction))
    left = (width - kept) // 2
    return slice(left, left + kept)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return (np.asarray(im.convert("L")) > 0).astype(np.float32)


def to_shape(a, shape):
    if a is None or a.shape == shape:
        return a
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def raw_measure(a, f, upper=None, lower=None):
    cfg = g2.recommended_config(
        apo_thr=0.5, fasc_thr=0.5, clip=False,
        compute_structure=False, compute_streamline=False,
        edge_source="centroid" if upper is not None and lower is not None else "mask",
    )
    r = g2.measure_apo(a, f, None, None, cfg, apo_upper=upper, apo_lower=lower)
    # The fragment-only branch is intentional: no field streamlines or image
    # intensity, so this is a geometry diagnostic rather than a C18 replacement.
    return {"pa_deg": r.variants.get("pa_global", np.nan),
            "fl_px": r.variants.get("fl_straight", np.nan), "mt_px": r.mt_mm,
            "confidence": r.confidence, "flags": ";".join(r.flags)}


def run(args):
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    pairs_path = RESEARCH_ROOT / "data-labels-geometry/out/apo_fasc_pairs.csv"
    pairs = pd.read_csv(pairs_path).sort_values(["orig", "apo"])
    # Deterministic interleaving makes a limited smoke run cover several layouts.
    pairs["rank"] = pairs.groupby("orig").cumcount()
    pairs = pairs.sort_values(["rank", "orig", "apo"])
    if args.limit:
        pairs = pairs.head(args.limit)
    folds = pd.read_csv(args.fasc_folds)
    reps = dict(zip(folds.stem, folds.representative))
    groups = dict(zip(folds.stem, folds.group))
    ab, fb = g2.MaskBank(args.apo_bank), g2.MaskBank(args.fasc_bank)
    comp = ROOT / "data/umud-challenge-muscle-architecture-in-ultrasound-data"
    rows = []
    start = time.time()
    try:
        for count, p in enumerate(pairs.itertuples(), 1):
            ag = read_mask(comp / "apo_masks_v1/apo_masks_new_model_v1" / f"{p.apo}.tif")
            fg = read_mask(comp / "fasc_masks_v1/fasc_masks_new_model_v1" / f"{p.fasc}.tif")
            if ag.shape != fg.shape:
                raise ValueError(f"unaligned GT masks for {p.apo}")
            ref = raw_measure(ag, fg)
            key = reps.get(p.fasc, p.fasc)
            ap = to_shape(ab.prob("train_oof", f"{args.apo_model}_band", p.apo), ag.shape)
            up = to_shape(ab.prob("train_oof", f"{args.apo_model}_upper", p.apo), ag.shape)
            lo = to_shape(ab.prob("train_oof", f"{args.apo_model}_lower", p.apo), ag.shape)
            fp = to_shape(fb.prob("train_oof", f"{args.fasc_model}_seg", key), ag.shape)
            for source in ("gt", "oof"):
                for frac in args.fractions:
                    x = central_window(ag.shape[1], frac)
                    meta = dict(image_id=p.apo, fasc=p.fasc, layout=p.orig,
                                group=groups.get(p.fasc, p.fasc), source=source,
                                fraction=frac, width=x.stop-x.start)
                    if source == "oof" and (ap is None or fp is None):
                        result = dict(pa_deg=np.nan, fl_px=np.nan, mt_px=np.nan,
                                      confidence=0.0, flags="missing_maps")
                    elif source == "gt":
                        result = raw_measure(ag[:, x], fg[:, x])
                    else:
                        result = raw_measure(ap[:, x], fp[:, x],
                                             None if up is None else up[:, x],
                                             None if lo is None else lo[:, x])
                    row = dict(meta, **result)
                    for c in ("pa_deg", "fl_px", "mt_px"):
                        row[f"ref_{c}"] = ref[c]
                        row[f"error_{c}"] = result[c] - ref[c]
                    rows.append(row)
            if count % 10 == 0:
                pd.DataFrame(rows).to_csv(args.out / "per_image.csv", index=False)
                print(f"{count}/{len(pairs)} images, {time.time()-start:.0f}s", flush=True)
    finally:
        ab.close()
        fb.close()
    per = pd.DataFrame(rows)
    per.to_csv(args.out / "per_image.csv", index=False)
    summary = []
    for (source, fraction, layout), d in per.groupby(["source", "fraction", "layout"]):
        q = dict(source=source, fraction=fraction, layout=layout, n=len(d))
        for c in ("pa_deg", "fl_px", "mt_px"):
            e = d[f"error_{c}"]
            q[f"{c}_n_valid"] = int(e.notna().sum())
            q[f"{c}_mae"] = float(e.abs().mean())
            q[f"{c}_bias"] = float(e.mean())
            if c != "pa_deg":
                q[f"{c}_mean_abs_relative"] = float((e / d[f"ref_{c}"].replace(0, np.nan)).abs().mean())
        summary.append(q)
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = dict(
        protocol="crop existing GT/OOF maps; full-frame mask geometry target; fragment-only branch",
        limitations=["Not a Kaggle score; lengths in native-mask pixels.",
                     "No network is re-run on cropped inputs.",
                     "GT crop disagreement includes real regional anatomy variation.",
                     "The paired subset has no ArtUS, Siemens 1152x864, or Lumify images.",
                     "Geometry definitions were developed on these training labels; this is a diagnostic, not a locked validation set."],
        n_pairs=len(pairs), fractions=args.fractions,
        pair_manifest_sha256=hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        apo_bank=str(args.apo_bank), apo_model=args.apo_model,
        fasc_bank=str(args.fasc_bank), fasc_model=args.fasc_model,
        seconds=time.time()-start,
        peak_rss_mb=rss/(1024**2 if sys.platform == "darwin" else 1024),
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/fov_stress")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.6, 0.35])
    ap.add_argument("--apo-bank", type=Path, default=ROOT / "artifacts/eval_v3/apo2/maskbank_fix")
    ap.add_argument("--apo-model", default="apo_v2fix")
    ap.add_argument("--fasc-bank", type=Path, default=ROOT / "artifacts/kaggle/fasc_orient_v1/maskbank")
    ap.add_argument("--fasc-model", default="fasc_orient_v1")
    ap.add_argument("--fasc-folds", type=Path, default=ROOT / "artifacts/kaggle/fasc_orient_v1/metrics/folds.csv")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if any(not 0 < f <= 1 for f in args.fractions):
        ap.error("fractions must be in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()
