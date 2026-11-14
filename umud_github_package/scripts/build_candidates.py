"""Integrate MASKBANK probability maps + geometry v2 into benchmark tables and test candidates.

Light by design: no neural network, one image at a time, lazy npz keys, one process (~0.3 GB RSS).

Subcommands
-----------
measure  --dataset {static,dynamic,vl,test} --source NAME [--cfg v2|v1|v2h]
         Runs g2.measure on every image of the dataset with the maps of one mask source and caches
         every variant (as_row) in artifacts/candidates/cache/<dataset>__<source>__<cfg>.csv.
         Resumable: images already in the cache are skipped.
table    Scores every cached benchmark run (static / dynamic / vl) for several PA x FL heads and
         writes artifacts/candidates/bench_table.csv (+ shrinkage / coupling studies).
build    Writes the candidate submissions C1..C3 from the cached test runs.

Mask sources (``SOURCES``): single models use their own calibrated thresholds; ensembles average
*calibrated* probabilities (each map is mapped piecewise-linearly so that its operating threshold
becomes 0.5) and threshold the mean at 0.5.

Benchmarks
----------
static  : 35 OSF static images, GT mean of R1..R7 (im_19 R7_MT -> NaN), clean10 subset (not in our
          training data), 2-rater (R7 + Rk) GTs and R7 alone.
dynamic : 167 GM calf-raise frames (Telemed Echo Blaster), PA vs HORIZONTAL (cfg v2h), FL in px.
vl      : UMUD 'Example Annotated Images' (VL, Esaote MyLab70, independent raters). Expert FL / PA
          parsed from the burned-in ImageJ lines (scripts/vl_labels.py ->
          artifacts/candidates/bench/vl_labels.csv). MT is not an expert measurement there; the
          'MT_lines' value is the distance between the drawn aponeurosis lines (loose check only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from umud import geometry_v2 as g2  # noqa: E402
from umud.metric import TOL, score_arrays  # noqa: E402
from umud.paths import RESEARCH_ROOT  # noqa: E402

warnings.filterwarnings("ignore")

SCRATCH = RESEARCH_ROOT  # compatibility name for downstream experiment scripts
OUT = ROOT / "artifacts/candidates"
CACHE = OUT / "cache"
BENCHDIR = OUT / "bench"
BANK_ROOT = ROOT / "artifacts/maskbank"
STATIC = ROOT / "data/external_umud_benchmark/static/benchmark_dataset_architecture_v0.1.0"
DYN = ROOT / "data/external_umud_benchmark/dynamic/benchmark_dataset_architecture_GM_dynamic_v0.1.0"
COMP = ROOT / "data/umud-challenge-muscle-architecture-in-ultrasound-data"
TEST_DIR = COMP / "test_images_v2/test_set_v2"
VL = BANK_ROOT / "vl_data"
CLEAN = [f"im_{k:02d}_arch" for k in (2, 3, 14, 17, 20, 23, 24, 31, 33, 34)]
RATERS = [f"R{k}" for k in range(1, 8)]
C3 = ("pa_deg", "fl_mm", "mt_mm")
BOUNDS = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
DYN_PPC = 77.4

# ----------------------------------------------------------------------------------- sources
# (model, threshold) lists; one entry = single model, two = calibrated-mean ensemble.
OA = [("ours_apo384_noflip", 0.8)]
DA = [("dlt_apo_noflip", 0.5)]
OF = [("ours_fasc384_noflip", 0.95)]
DF = [("dlt_fasc_noflip", 0.2)]
SOURCES = {
    "ours": dict(apo=OA, fasc=OF, fb="dlt_apo_noflip"),
    "dlt": dict(apo=DA, fasc=DF, fb="ours_apo384_noflip"),
    "ens": dict(apo=OA + DA, fasc=OF + DF, fb=None),
    "oA_dF": dict(apo=OA, fasc=DF, fb="dlt_apo_noflip"),
    "dA_oF": dict(apo=DA, fasc=OF, fb="ours_apo384_noflip"),
    "eA_oF": dict(apo=OA + DA, fasc=OF, fb=None),
    "oA_eF": dict(apo=OA, fasc=OF + DF, fb="dlt_apo_noflip"),
    # union variants: thin 1-px fascicle lines of two models rarely overlap pixel-exactly, so a mean
    # behaves like an intersection; the calibrated MAX keeps every confident fragment of either model
    "ensU": dict(apo=OA + DA, fasc=OF + DF, fb="ours_apo384_noflip", fasc_comb="max"),
    "oA_uF": dict(apo=OA, fasc=OF + DF, fb="dlt_apo_noflip", fasc_comb="max"),
}


def calibrate(p: np.ndarray, t: float) -> np.ndarray:
    """Monotone piecewise-linear map with calibrate(t) = 0.5, calibrate(0) = 0, calibrate(1) = 1."""
    return np.where(p < t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / max(1.0 - t, 1e-6)).astype(np.float32)


def recalibrate(p: np.ndarray, t_from: float, t_to: float) -> np.ndarray:
    """Monotone piecewise-linear map sending t_from -> t_to (0 -> 0, 1 -> 1)."""
    return np.where(p < t_from, t_to * p / t_from,
                    t_to + (1.0 - t_to) * (p - t_from) / max(1.0 - t_from, 1e-6)).astype(np.float32)


def load_maps(bank: g2.MaskBank, ds: str, image_id: str, spec: list,
              comb: str = "mean") -> tuple[np.ndarray | None, float]:
    if len(spec) == 1:
        m, t = spec[0]
        return bank.prob(ds, m, image_id), t
    acc = None
    for m, t in spec:
        p = bank.prob(ds, m, image_id)
        if p is None:
            return None, 0.5
        c = calibrate(p, t)
        acc = c if acc is None else (np.maximum(acc, c) if comb == "max" else acc + c)
    return (acc if comb == "max" else acc / len(spec)), 0.5


def make_cfg(name: str, apo_thr: float, fasc_thr: float) -> g2.GeoConfig:
    if name == "v1":
        return g2.GeoConfig.v1().with_(apo_thr=apo_thr, fasc_thr=fasc_thr)
    cfg = g2.recommended_config(apo_thr=apo_thr, fasc_thr=fasc_thr, compute_streamline=False)
    if name.endswith("h"):  # PA against the horizontal (dynamic benchmark convention)
        cfg = cfg.with_(pa_reference="horizontal")
        name = name[:-1]
    # aponeurosis extrapolation beyond the detected band ends (FL needs it when fascicles leave the FOV)
    ext = {"v2": "tangent", "v2p": "poly", "v2g": "global_linear", "v2e": "end_linear"}[name]
    return cfg.with_(edge_extrap=ext)


# ----------------------------------------------------------------------------------- datasets
def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def static_labels() -> pd.DataFrame:
    lab = pd.read_excel(STATIC / "Results_benchmark_architecture_v0.1.0.xlsx", sheet_name="Manual_architecture")
    lab.loc[lab.ImageID == "im_19_arch", "R7_MT"] = np.nan  # 80.13 typo
    return lab.set_index("ImageID")


TEST_LAYOUT = ROOT / "artifacts/final/test_layout.csv"   # umud.layout output (umud-run --stage layout)


def test_layout() -> pd.DataFrame:
    """Scanner family / size / px per cm / group and the video clips of every test image, derived from the images
    by umud.layout (clips from consecutive-frame correlation, not from file names); built on first use."""
    from umud.layout import load_or_build
    return load_or_build(TEST_DIR, TEST_LAYOUT)


def iterate(ds: str):
    """Yield (image_id, gray image, px_per_cm, extra dict) one image at a time."""
    if ds == "static":
        lab = static_labels()
        for iid in lab.index:
            yield iid, gray(STATIC / f"{iid}.tif"), float(lab.loc[iid, "Scale_pixel_per_cm"]), {}
    elif ds == "dynamic":
        for fr in range(168):
            key = f"frame_{fr:05d}"
            p = DYN / "frames" / f"{key}.png"
            if p.exists():
                yield key, gray(p), DYN_PPC, {"frame": fr}
    elif ds == "vl":
        idx = pd.read_csv(VL / "vl_index.csv")
        for r in idx.itertuples():
            yield r.key, gray(VL / "images" / f"{r.key}.png"), float(r.ann_xres), {"set": r.set}
    elif ds == "test":
        lay = test_layout()
        for iid in lay.index:
            yield iid, gray(TEST_DIR / iid), float(lay.loc[iid, "px_per_cm"]), {"group": lay.loc[iid, "group"],
                                                                                   "clip": int(lay.loc[iid, "clip"])}
    else:
        raise ValueError(ds)


def cmd_measure(args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    src = SOURCES[args.source]
    path = CACHE / f"{args.dataset}__{args.source}__{args.cfg}.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(prev.image_id) if len(prev) else set()
    bank = g2.MaskBank(BANK_ROOT)
    t0 = time.time()
    rows, k = [], 0
    for iid, img, ppc, extra in iterate(args.dataset):
        if iid in done:
            continue
        A, ta = load_maps(bank, args.dataset, iid, src["apo"], src.get("apo_comb", "mean"))
        F, tf = load_maps(bank, args.dataset, iid, src["fasc"], src.get("fasc_comb", "mean"))
        cfg = make_cfg(args.cfg, ta, tf)
        if A is None or F is None:
            res = g2._nan_result("missing_maps")
        else:
            fb = None
            if src["fb"] and args.cfg != "v1":
                fb = bank.prob(args.dataset, src["fb"], iid)
                # measure() thresholds the fallback map with the primary threshold: map the fallback
                # model's own operating threshold onto it (monotone, piecewise linear)
                if fb is not None:
                    fb = recalibrate(fb, dict(OA + DA)[src["fb"]], ta)
            try:
                res = g2.measure(A, F, img, ppc, cfg, apo_fallback=fb)
            except Exception as e:  # keep going; record the failure
                res = g2._nan_result(f"error:{type(e).__name__}")
        rows.append(dict(image_id=iid, px_per_cm=ppc, **extra, **res.as_row()))
        k += 1
        if k % 25 == 0:  # rewrite the whole (small) cache: rows have varying diag columns
            pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
            print(f"  {args.dataset}/{args.source}/{args.cfg}: {k} new, {time.time() - t0:.0f}s", flush=True)
    pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    bank.close()
    print(f"done {path.name}: {k} new rows in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------- heads
PA_HEADS = ("fusion", "global", "structure", "central3", "local")
FL_HEADS = ("ensemble", "straight", "central3", "constructed", "med3", "med4", "avg_sc", "triangle", "ens_tri")


def heads(d: pd.DataFrame, pa: str, fl: str, k: float = 1.065) -> pd.DataFrame:
    """Prediction frame (pa_deg, fl_mm, mt_mm) for a PA head and an FL head."""
    p = d[f"pa_{pa}"].astype(float)
    mt = d["mt_mm"].astype(float)
    if fl in ("ensemble", "straight", "central3", "constructed"):
        # note: 'constructed' (and so 'ensemble') is built at the configured (fusion) PA
        f = d[f"fl_{fl}"].astype(float)
    elif fl == "med3":
        f = d[["fl_straight", "fl_central3", "fl_constructed"]].median(axis=1)
    elif fl == "med4":
        tri = k * mt / np.sin(np.radians(p.clip(lower=1.0)))
        f = pd.concat([d[["fl_straight", "fl_central3", "fl_constructed"]], tri.rename("tri")], axis=1).median(axis=1)
    elif fl == "avg_sc":
        f = d[["fl_straight", "fl_constructed"]].mean(axis=1)
    elif fl == "triangle":
        f = k * mt / np.sin(np.radians(p.clip(lower=1.0)))
    elif fl == "ens_tri":
        tri = k * mt / np.sin(np.radians(p.clip(lower=1.0)))
        f = pd.concat([d["fl_ensemble"].astype(float), tri], axis=1).mean(axis=1)
    else:
        raise ValueError(fl)
    return pd.DataFrame({"pa_deg": p, "fl_mm": f, "mt_mm": mt}, index=d.index)


def fill(pred: pd.DataFrame, groups: pd.Series | None = None) -> pd.DataFrame:
    """Fill NaNs by the group median of the predictions (then the global median)."""
    out = pred.copy()
    for c in C3:
        if groups is not None:
            out[c] = out[c].fillna(out.groupby(groups)[c].transform("median"))
        out[c] = out[c].fillna(out[c].median())
    return out


def shrink(pred: pd.DataFrame, conf: pd.Series, groups: pd.Series, lam: float, cols=("pa_deg", "fl_mm"),
           mode: str = "conf") -> pd.DataFrame:
    """Shrink toward the group median of the predictions: w = 1 - lam * (1 - conf) (mode conf) or
    w = 1 - lam (mode const)."""
    out = pred.copy()
    w = (1.0 - lam * (1.0 - conf.fillna(0.0).clip(0, 1))) if mode == "conf" else pd.Series(1.0 - lam, index=pred.index)
    for c in cols:
        med = out.groupby(groups)[c].transform("median")
        out[c] = w * out[c] + (1 - w) * med
    return out


def s_metric(pred: pd.DataFrame, gt: pd.DataFrame, comps=C3) -> dict:
    ids = [i for i in pred.index if i in gt.index]
    r = {}
    norms = []
    for c in comps:
        e = (pred.loc[ids, c].astype(float) - gt.loc[ids, c].astype(float)).dropna()
        n = float(e.abs().mean() / TOL[c]) if len(e) else np.nan
        r[f"{c[:2]}_n"] = n
        r[f"{c[:2]}_bias"] = float(e.mean()) if len(e) else np.nan
        norms.append(n)
    r["S"] = float(np.mean(norms))
    r["n"] = len(ids)
    return r


# ----------------------------------------------------------------------------------- GT sets
def static_gts() -> dict:
    lab = static_labels()

    def mean_of(rs):
        return pd.DataFrame({c: lab[[f"{r}_{k}" for r in rs]].mean(1) for c, k in zip(C3, ("PA", "FL", "MT"))})
    out = {"mean7": mean_of(RATERS), "R7": mean_of(["R7"])}
    for r in RATERS[:-1]:
        out[f"R7+{r}"] = mean_of(["R7", r])
    return out


def dynamic_gt() -> pd.DataFrame:
    xl = DYN / "benchmark_architecture_GM_calf_raise_v0.1.0.xlsx"
    pa = pd.read_excel(xl, sheet_name="PA")[["frame", "Mean"]].rename(columns={"Mean": "pa_deg"})
    fl = pd.read_excel(xl, sheet_name="FL")[["frame", "Mean"]].rename(columns={"Mean": "fl_px"})
    lab = pa.merge(fl, on="frame", how="outer").dropna(subset=["frame"])
    lab["image_id"] = lab.frame.astype(int).map(lambda f: f"frame_{f:05d}")
    lab["fl_mm"] = lab.fl_px * 10 / DYN_PPC
    return lab.set_index("image_id")


def vl_gt() -> pd.DataFrame:
    p = BENCHDIR / "vl_labels.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python scripts/vl_labels.py` (parses the expert lines of the "
                                "OSF 'Example Annotated Images')")
    d = pd.read_csv(p).set_index("key")
    d = d[d.n_fasc >= 2]
    return pd.DataFrame({"pa_deg": d.pa_deg, "fl_mm": d.fl_mm, "mt_mm": d.mt_lines_mm,
                         "pa_chord": d.pa_chord_deg}, index=d.index)


EXPECTED = {"static": 35, "dynamic": 168, "vl": 249, "test": 309}


def load_cache(ds: str, source: str, cfg: str) -> pd.DataFrame | None:
    """Cached per-image results; None unless the run is complete."""
    p = CACHE / f"{ds}__{source}__{cfg}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).drop_duplicates("image_id", keep="last").set_index("image_id")
    return d if len(d) >= EXPECTED[ds] else None


# ----------------------------------------------------------------------------------- table
def bench_rows(source: str, cfg: str = "v2") -> list[dict]:
    rows = []
    st = load_cache("static", source, cfg)
    gts = static_gts()
    # PA vs horizontal does not depend on the aponeurosis extrapolation -> always the v2h run
    dyn = load_cache("dynamic", source, "v2h") if cfg.startswith("v2") else None
    dyn_d = load_cache("dynamic", source, cfg)
    vl = load_cache("vl", source, cfg)
    vgt = vl_gt()
    dgt = dynamic_gt()
    for pa in PA_HEADS:
        for fl in FL_HEADS:
            r = dict(source=source, cfg=cfg, pa=pa, fl=fl)
            if st is not None:
                pr = fill(heads(st, pa, fl))
                for sub, ids in (("s35", list(st.index)), ("c10", CLEAN)):
                    m = s_metric(pr.loc[ids], gts["mean7"])
                    r.update({f"{sub}_{k}": v for k, v in m.items() if k != "n"})
                    r[f"{sub}_2r"] = float(np.mean([s_metric(pr.loc[ids], gts[f"R7+{x}"])["S"] for x in RATERS[:-1]]))
                    r[f"{sub}_R7"] = s_metric(pr.loc[ids], gts["R7"])["S"]
                r["s35_fail"] = int(st[["pa_deg", "fl_mm", "mt_mm"]].isna().any(axis=1).sum())
            if vl is not None:
                pr = fill(heads(vl, pa, fl))
                m = s_metric(pr, vgt, comps=("pa_deg", "fl_mm"))
                r.update({f"vl_{k}": v for k, v in m.items()})
                r["vl_mt_n"] = s_metric(pr, vgt, comps=("mt_mm",))["mt_n"]
                g2c = vgt.rename(columns={"pa_deg": "_pa", "pa_chord": "pa_deg"})
                r["vl_pa_chord_n"] = s_metric(pr, g2c, comps=("pa_deg",))["pa_n"]
                r["vl_fail"] = int(vl[["pa_deg", "fl_mm", "mt_mm"]].isna().any(axis=1).sum())
            if dyn is not None:
                prh = fill(heads(dyn, pa, fl))
                m = s_metric(prh, dgt, comps=("pa_deg",))
                r["dyn_pa_h_n"] = m["pa_n"]
                r["dyn_pa_h_bias"] = m["pa_bias"]
            if dyn_d is not None:
                prd = fill(heads(dyn_d, pa, fl))
                m = s_metric(prd, dgt, comps=("fl_mm",))
                r["dyn_fl_n"] = m["fl_n"]
                r["dyn_fl_bias"] = m["fl_bias"]
            rows.append(r)
    return rows


def static_groups(idx) -> pd.Series:
    """Scanner proxy for the static benchmark: image size."""
    sz = {}
    for iid in idx:
        with Image.open(STATIC / f"{iid}.tif") as im:
            sz[iid] = f"{im.size[0]}x{im.size[1]}"
    return pd.Series(sz)


def shrink_study(source: str, pa: str = "fusion", fl: str = "ensemble", cfg: str = "v2") -> pd.DataFrame:
    rows = []
    st = load_cache("static", source, cfg)
    vl = load_cache("vl", source, cfg)
    if st is None or vl is None:
        return pd.DataFrame()
    gts = static_gts()
    vgt = vl_gt()
    sg = static_groups(st.index)
    vg = pd.Series("vl", index=vl.index)
    for mode in ("conf", "const"):
        for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            for cols in (("pa_deg",), ("fl_mm",), ("pa_deg", "fl_mm"), ("pa_deg", "fl_mm", "mt_mm")):
                r = dict(source=source, cfg=cfg, pa=pa, fl=fl, mode=mode, lam=lam, cols="+".join(c[:2] for c in cols))
                ps = shrink(fill(heads(st, pa, fl)), st.confidence, sg, lam, cols, mode)
                r.update({f"s35_{k}": v for k, v in s_metric(ps, gts["mean7"]).items() if k in ("S", "pa_n", "fl_n", "mt_n")})
                r["c10_S"] = s_metric(ps.loc[CLEAN], gts["mean7"])["S"]
                r["s35_2r"] = float(np.mean([s_metric(ps, gts[f"R7+{x}"])["S"] for x in RATERS[:-1]]))
                pv = shrink(fill(heads(vl, pa, fl)), vl.confidence, vg, lam, cols, mode)
                m = s_metric(pv, vgt, comps=("pa_deg", "fl_mm"))
                r.update({f"vl_{k}": v for k, v in m.items() if k in ("S", "pa_n", "fl_n")})
                r["vl_mt_n"] = s_metric(pv, vgt, comps=("mt_mm",))["mt_n"]
                rows.append(r)
    return pd.DataFrame(rows)


def coupling_study(source: str) -> pd.DataFrame:
    """k = FL sin(PA) / MT on expert labels and on our predictions, and FL from the triangle."""
    rows = []
    gts = static_gts()["mean7"]
    vgt = vl_gt()
    for name, g in (("static_mean7", gts), ("vl_expert(MT_lines)", vgt)):
        k = g.fl_mm * np.sin(np.radians(g.pa_deg)) / g.mt_mm
        rows.append(dict(what=name, k_median=float(k.median()), k_q25=float(k.quantile(.25)), k_q75=float(k.quantile(.75)),
                         n=int(k.notna().sum())))
    for ds in ("static", "vl", "test"):
        d = load_cache(ds, source, "v2")
        if d is None:
            continue
        k = d.fl_ensemble * np.sin(np.radians(d.pa_fusion)) / d.mt_mm
        rows.append(dict(what=f"{ds}_{source}_pred", k_median=float(k.median()), k_q25=float(k.quantile(.25)),
                         k_q75=float(k.quantile(.75)), n=int(k.notna().sum())))
    return pd.DataFrame(rows)


def candidate_bench(source: str = "ours", cfg: str = "v2e") -> pd.DataFrame:
    """The exact C1/C2/C3 recipes (MT<5 -> failure, group fill, shrink) on every benchmark."""
    G, v, dg = static_gts(), vl_gt(), dynamic_gt()
    st, vl = load_cache("static", source, cfg), load_cache("vl", source, cfg)
    dd, dh = load_cache("dynamic", source, cfg), load_cache("dynamic", source, "v2h")
    if any(x is None for x in (st, vl, dd, dh)):
        return pd.DataFrame()
    sg = static_groups(st.index)

    def rec(d, groups, mode):
        p = heads(d, "fusion", "ensemble")
        p.loc[p.mt_mm < 5, list(C3)] = np.nan
        p = fill(p, groups)
        if mode == "C2":
            p = shrink(p, d.confidence, groups, 1.0)
        if mode == "C3":
            p = shrink(p, d.confidence, groups, 1.0, mode="const")
        return p
    rows = []
    for mode in ("C1", "C2", "C3"):
        r = {"recipe": mode}
        p = rec(st, sg, mode)
        m = s_metric(p, G["mean7"])
        r.update(s35=m["S"], s35_pa=m["pa_n"], s35_fl=m["fl_n"], s35_mt=m["mt_n"])
        r["c10"] = s_metric(p.loc[CLEAN], G["mean7"])["S"]
        r["2r"] = float(np.mean([s_metric(p, G[f"R7+{x}"])["S"] for x in RATERS[:-1]]))
        r["R7"] = s_metric(p, G["R7"])["S"]
        p = rec(vl, pd.Series("vl", index=vl.index), mode)
        m = s_metric(p, v, ("pa_deg", "fl_mm"))
        r.update(vl=m["S"], vl_pa=m["pa_n"], vl_fl=m["fl_n"])
        r["vl_mt"] = s_metric(p, v, ("mt_mm",))["mt_n"]
        r["vl_young3"] = s_metric(p.loc[[i for i in p.index if i.startswith("young")]], v)["S"]
        p = rec(dh, pd.Series("d", index=dh.index), mode)
        r["dyn_pa_h"] = s_metric(p, dg, ("pa_deg",))["pa_n"]
        p = rec(dd, pd.Series("d", index=dd.index), mode)
        r["dyn_fl"] = s_metric(p, dg, ("fl_mm",))["fl_n"]
        rows.append(r)
    return pd.DataFrame(rows).round(3)


def cmd_table(args) -> None:
    rows = []
    for s in SOURCES:
        if load_cache("static", s, "v2") is None and load_cache("vl", s, "v2") is None:
            continue
        rows += bench_rows(s, "v2")
    for cfg in ("v2p", "v2g", "v2e"):
        if load_cache("static", "ours", cfg) is not None and load_cache("vl", "ours", cfg) is not None:
            rows += [r for r in bench_rows("ours", cfg) if r["pa"] in ("fusion", "global")]
    if load_cache("static", "ours", "v1") is not None:
        rows += [r for r in bench_rows("ours", "v1") if r["pa"] == "global"]
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "bench_table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    show = [c for c in ["source", "cfg", "pa", "fl", "s35_S", "s35_pa_n", "s35_fl_n", "s35_mt_n", "c10_S", "c10_fl_n",
                        "s35_2r", "vl_S", "vl_pa_n", "vl_fl_n", "vl_mt_n", "vl_pa_chord_n", "dyn_pa_h_n", "dyn_fl_n",
                        "s35_fail", "vl_fail"] if c in T]
    main = T[(T.pa.isin(["fusion", "global", "structure"])) & (T.fl.isin(["ensemble", "straight", "constructed", "med3"]))]
    print(main[show].round(3).to_string(index=False))
    # compact markdown: every source with the default heads + head ablations for the chosen source
    md_cols = [c for c in ["source", "cfg", "pa", "fl", "s35_S", "s35_pa_n", "s35_fl_n", "s35_mt_n", "c10_S", "s35_2r",
                           "s35_R7", "vl_S", "vl_pa_n", "vl_fl_n", "vl_mt_n", "dyn_pa_h_n", "dyn_fl_n", "vl_fail"] if c in T]
    a = T[(T.pa == "fusion") & (T.fl == "ensemble")]
    a = a[a.cfg == "v2"]
    b = T[(T.source == "ours") & (T.cfg == "v2") & ~((T.pa == "fusion") & (T.fl == "ensemble"))
          & T.pa.isin(["fusion", "global", "structure", "central3"])]
    e = T[(T.source == "ours") & T.cfg.isin(["v2", "v2p", "v2g", "v2e"]) & (T.pa == "fusion")
          & T.fl.isin(["ensemble", "straight", "central3", "constructed", "med3"])].sort_values(["fl", "cfg"])
    c = T[(T.cfg == "v1") & T.fl.isin(["straight", "ensemble", "med3"])]
    lines = []
    for title, X in (("Mask sources (PA = fusion, FL = ensemble, tangent apo extrapolation)", a),
                     ("Aponeurosis extrapolation (source ours; v2 = tangent, v2p = quadratic, v2g = global straight line, "
                      "v2e = straight line through the last 30 % of the band)", e),
                     ("Head ablations (source ours, v2)", b),
                     ("v1 geometry (LB 0.467 config; NaN failures filled by the median)", c)):
        if not len(X):
            continue
        X = X[md_cols].round(3)
        lines += [f"#### {title}", "", "| " + " | ".join(md_cols) + " |", "|" + "---|" * len(md_cols)]
        lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in X.itertuples(index=False)]
        lines.append("")
    (OUT / "bench_table.md").write_text("\n".join(lines))
    for s, cfg in (("ours", "v2"), ("ours", "v2g"), ("ours", "v2e"), ("ens", "v2")):
        if load_cache("vl", s, cfg) is None:
            continue
        S = shrink_study(s, cfg=cfg)
        if len(S):
            S.to_csv(OUT / f"bench_shrink_{s}_{cfg}.csv", index=False)
            print(f"\n== shrinkage study ({s} {cfg}, fusion/ensemble)")
            print(S[S["mode"] == "conf"].round(4).to_string(index=False))
    CB = candidate_bench()
    if len(CB):
        CB.to_csv(OUT / "bench_candidates.csv", index=False)
        (OUT / "bench_candidates.md").write_text(
            "\n".join(["| " + " | ".join(CB.columns) + " |", "|" + "---|" * len(CB.columns)]
                      + ["| " + " | ".join(str(x) for x in r) + " |" for r in CB.itertuples(index=False)]) + "\n")
        print("\n== candidate recipes on the benchmarks")
        print(CB.to_string(index=False))
    C = pd.concat([coupling_study(s) for s in ("ours", "ens") if load_cache("static", s, "v2") is not None])
    C.to_csv(OUT / "bench_coupling.csv", index=False)
    print("\n== FL coupling k = FL sin(PA) / MT")
    print(C.round(3).to_string(index=False))


# ----------------------------------------------------------------------------------- build
def postprocess(pred: pd.DataFrame, lay: pd.DataFrame) -> pd.DataFrame:
    """Group-median NaN fill -> median within the 28 verified clips -> clip to the bounds."""
    out = fill(pred, lay.loc[pred.index, "group"])
    clip = lay.loc[out.index, "clip"]
    inside = clip >= 0
    for c in C3:
        med = out.loc[inside, c].groupby(clip[inside]).transform("median")
        out.loc[inside, c] = med
        out[c] = out[c].clip(*BOUNDS[c])
    return out


def write_sub(pred: pd.DataFrame, path: Path) -> None:
    ids = sorted(p.name for p in TEST_DIR.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png"})
    pred = pred.loc[ids]
    lines = ["image_id,pa_deg,fl_mm,mt_mm"] + [f"{i},{r.pa_deg:.4f},{r.fl_mm:.4f},{r.mt_mm:.4f}" for i, r in pred.iterrows()]
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode())


def official_distance(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    ids = a.index.intersection(b.index)
    s, rep = score_arrays({c: a.loc[ids, c].to_numpy(float) for c in C3},
                          {c: b.loc[ids, c].to_numpy(float) for c in C3}, report=True)
    return dict(S=round(s, 4), **{f"{c[:2]}_n": round(rep[c]["norm"], 4) for c in C3},
                **{f"{c[:2]}_bias": round(rep[c]["bias"], 3) for c in C3})


CANDIDATES = {
    # name: (source, pa head, fl head, shrink, description)
    #   shrink: None = per-image values; float lam = confidence-weighted shrinkage of PA and FL to the
    #   scanner-group median with weight w = 1 - lam * (1 - confidence); "group" = w = 0 (PA and FL
    #   replaced by the group median of our own predictions, MT stays per image)
    "C1": ("ours", "v2e", "fusion", "ensemble", None,
           "our U-Net++ maps (+DL_Track apo fallback) + geometry v2 with end-linear aponeurosis extrapolation"),
    "C2": ("ours", "v2e", "fusion", "ensemble", 1.0, "C1 + confidence-weighted shrinkage of PA/FL (w = confidence)"),
    "C3": ("ours", "v2e", "fusion", "ensemble", "group",
           "C1 with PA/FL replaced by the scanner-group medians of C1 (MT per image)"),
}


def cmd_build(args) -> None:
    lay = test_layout()
    cands = dict(CANDIDATES)
    if args.spec:
        cands = json.loads(Path(args.spec).read_text())
    refs = {"v1_LB0.467": pd.read_csv(ROOT / "artifacts/sub_v1_geom_clipmed.csv").set_index("image_id"),
            "eason_0.374": pd.read_csv(SCRATCH / "notebooks-insight/derived/eason_0.37415.csv").set_index("image_id"),
            "vera_0.451": pd.read_csv(SCRATCH / "notebooks-insight/derived/vera_0.45134.csv").set_index("image_id")}
    report = {}
    built = {}
    for name, (source, cfg, pa, fl, lam, desc) in cands.items():
        d = load_cache("test", source, cfg)
        if d is None:
            print(f"{name}: test cache {source}/{cfg} missing or incomplete -> skipped")
            continue
        pred = heads(d, pa, fl)
        # an MT below 5 mm (half the lower clip bound) means a wrong aponeurosis pair: treat the
        # whole row as a failure (group-median fill) instead of clipping garbage (IMG_00284: 0.5 mm)
        bad = pred.mt_mm < 5.0
        pred.loc[bad, list(C3)] = np.nan
        pred = fill(pred, lay.loc[d.index, "group"])
        if lam == "group":
            pred = shrink(pred, d.confidence, lay.loc[d.index, "group"], 1.0, ("pa_deg", "fl_mm"), "const")
        elif lam:
            pred = shrink(pred, d.confidence, lay.loc[d.index, "group"], float(lam), ("pa_deg", "fl_mm"), "conf")
        sub = postprocess(pred, lay)
        tag = "" if not lam else ("_groupPAFL" if lam == "group" else f"_shrink{lam}")
        path = OUT / f"{name}_{source}_{cfg}_{pa}_{fl}{tag}.csv"
        write_sub(sub, path)
        built[name] = sub
        g = sub.groupby(lay.loc[sub.index, "group"])[list(C3)].median().round(2)
        g["n"] = lay.group.value_counts()
        report[name] = dict(file=path.name, desc=desc, source=source, cfg=cfg, pa=pa, fl=fl, shrink=lam,
                            n_nan_raw=int(d[["pa_deg", "fl_mm", "mt_mm"]].isna().any(axis=1).sum()),
                            n_mt_below5=int(bad.sum()),
                            group_medians=g.reset_index().to_dict("records"),
                            dist={k: official_distance(sub, v) for k, v in refs.items()})
        print(f"\n== {name}: {path.name}  ({desc})")
        print(g.to_string())
        for k, v in report[name]["dist"].items():
            print(f"   vs {k:12s} {v}")
    for a in built:
        for b in built:
            if a < b:
                report.setdefault("pairwise", {})[f"{a}-{b}"] = official_distance(built[a], built[b])
    print(json.dumps(report.get("pairwise", {}), indent=1))
    (OUT / "candidates_report.json").write_text(json.dumps(report, indent=1, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--dataset", required=True, choices=["static", "dynamic", "vl", "test"])
    m.add_argument("--source", required=True, choices=list(SOURCES))
    m.add_argument("--cfg", default="v2", choices=["v2", "v1", "v2h", "v2p", "v2g", "v2e", "v2ph", "v2gh", "v2eh"])
    sub.add_parser("table")
    b = sub.add_parser("build")
    b.add_argument("--spec", help="optional JSON {name: [source, cfg, pa, fl, shrink, desc]}")
    args = ap.parse_args()
    {"measure": cmd_measure, "table": cmd_table, "build": cmd_build}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
