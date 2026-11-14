"""Aponeurosis / MT source study: apo_v2 (band + upper/lower edge channels) vs ours_apo384 (apo2 agent).

Light by design: no neural network, one image at a time, lazy npz keys, one process (< 0.5 GB RSS).

Subcommands
-----------
measure --dataset {static,dynamic,vl,test,train_oof} --source NAME [--cfg v2e|v2eh] [--limit N]
        g2.measure_apo() per image with one aponeurosis source (``SOURCES``) and our fascicle maps;
        caches every variant / flag / curve parameter in artifacts/eval_v3/apo2/cache/
        <dataset>__<source>__<cfg>.csv (resumable). train_oof has no fascicle maps (zeros): MT only;
        source "gt" measures the ground-truth masks there (inner-edge MT of the superficial pair).
report  Benchmarks (train_oof MT vs GT-mask MT by layout, static all/clean10/in-train MT vs experts,
        VL drawn-line gap, dynamic FL / PA side effects through the curves), test diagnostics
        (pair failures, Lumify recovery, per-group MT, source disagreement), MT fusions, and the
        per-image test caches for the assembler (artifacts/eval_v3/apo2/test_mt_*.csv).
overlay Disagreement montage: test images with the largest |MT(v2) - MT(ours)|, both inner-edge
        pairs drawn (artifacts/eval_v3/apo2/disagreement_*.png).

Sources
-------
ours  : ours_apo384_noflip @0.8, mask edges, fallback dlt_apo_noflip     (= candidates C1/C2 MT)
v2    : apo_v2 band @0.5 for the pair, inner edges from the upper/lower edge channels (centroid),
        fallbacks ours_apo384 then dlt_apo
v2m   : as v2 but mask (band) edges only            -> value of the edge channels
v2n   : as v2 without fallbacks                     -> raw pair-failure count of apo_v2
oursE : ours pair, inner edges refined by the apo_v2 edge channels (hybrid)
gt    : train_oof only: ground-truth masks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from umud import geometry_v2 as g2  # noqa: E402
from umud.metric import TOL  # noqa: E402
import build_candidates as bc  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path(os.environ.get("APO2_OUT", ROOT / "artifacts/eval_v3/apo2"))   # override: outputs of another map source
CACHE = Path(os.environ.get("APO2_CACHE", OUT / "cache"))
MAIN_BANK = ROOT / "artifacts/maskbank"
# apo_v2 maps re-inferred with repaired BatchNorm statistics; env overrides (e.g. the apo_v3 kernel maskbank with
# APO2_MODEL=apo_v2r | apo_v3) switch the default source of Banks() for every importer (build_candidates_v3, ...)
V2_BANK = Path(os.environ.get("APO2_BANK", ROOT / "artifacts/eval_v3/apo2/maskbank_fix"))
V2_MODEL = os.environ.get("APO2_MODEL", "apo_v2fix")
COMP = ROOT / "data/umud-challenge-muscle-architecture-in-ultrasound-data"
TRAIN_IMG = COMP / "apo_imgs_v1/apo_images_new_model_v1"
TRAIN_MSK = COMP / "apo_masks_v1/apo_masks_new_model_v1"
FOLDS = ROOT / "artifacts/kaggle/apo_v2/apo_v2/folds.csv"
TRAIN_GEO = bc.SCRATCH / "data-labels-geometry/out/train_gt_geometry.csv"
BENCH_MATCH = bc.SCRATCH / "data-labels-geometry/out/bench_matches_fine.csv"
C3 = bc.C3

OURS = ("main", "ours_apo384_noflip", 0.8)
DLT = ("main", "dlt_apo_noflip", 0.5)
V2 = ("v2", "band", 0.5)
SOURCES = {
    "ours": dict(primary=OURS, edges=False, fallbacks=[("dlt", DLT)]),
    "v2": dict(primary=V2, edges=True, fallbacks=[("ours", OURS), ("dlt", DLT)]),
    "v2m": dict(primary=V2, edges=False, fallbacks=[("ours", OURS), ("dlt", DLT)]),
    "v2n": dict(primary=V2, edges=True, fallbacks=[]),
    "oursE": dict(primary=OURS, edges=True, fallbacks=[("dlt", DLT)]),
    "gt": dict(primary=None, edges=False, fallbacks=[]),
}
EXPECTED = {"static": 35, "dynamic": 168, "vl": 249, "test": 309, "train_oof": 1048}


# ----------------------------------------------------------------------------------- banks
class Banks:
    """Lazy access to the main MASKBANK and the apo_v2 bank (with VL key mapping)."""

    def __init__(self, v2_root: Path = V2_BANK, v2_model: str = V2_MODEL):
        self.main = g2.MaskBank(MAIN_BANK)
        self.v2 = g2.MaskBank(v2_root)
        self.v2_model = v2_model
        self.vl_map = {}
        for mf in sorted(Path(self.v2.root).glob("manifest*.csv")):
            m = pd.read_csv(mf)
            m = m[m.dataset == "vl"]
            for s, k in zip(m.source_filename, m.image_id):
                self.vl_map[str(s).replace("vl_", "", 1) if str(s).startswith("vl_") else str(s)] = k
        self.vl_raw = {}
        if (MAIN_BANK / "vl_data/vl_index.csv").exists():
            idx = pd.read_csv(MAIN_BANK / "vl_data/vl_index.csv")
            self.vl_raw = dict(zip(idx.key, idx.raw_source))

    def v2_key(self, ds: str, iid: str) -> str:
        if ds == "vl":
            return self.vl_map.get(self.vl_raw.get(iid, ""), iid)
        return Path(iid).stem

    def get(self, ds: str, iid: str, spec) -> np.ndarray | None:
        bank, model, _ = spec
        if bank == "main":
            return self.main.prob(ds, model, iid)
        return self.v2.prob(ds, f"{self.v2_model}_{model}", self.v2_key(ds, iid))

    def close(self):
        self.main.close()
        self.v2.close()


def make_cfg(name: str, thr: float, edges: bool) -> g2.GeoConfig:
    cfg = g2.recommended_config(apo_thr=thr, fasc_thr=0.95, compute_streamline=False, edge_extrap="end_linear",
                                edge_source="centroid" if edges else "mask")
    if name.endswith("h"):
        cfg = cfg.with_(pa_reference="horizontal")
    return cfg


# ----------------------------------------------------------------------------------- train_oof
def train_meta() -> pd.DataFrame:
    f = pd.read_csv(FOLDS)[["image_id", "layout", "group", "fold"]].set_index("image_id")
    g = pd.read_csv(TRAIN_GEO)
    g = g[g.set == "apo_gt"].set_index("name")[["orig", "img", "ppc", "scale_src"]]
    return f.join(g, how="left")


def iterate_train(meta: pd.DataFrame):
    for iid, r in meta.iterrows():
        img = np.asarray(Image.open(TRAIN_IMG / f"{iid}.tif").convert("L"))
        yield iid, img, r


def measure_one(banks: Banks, ds: str, iid: str, img: np.ndarray, ppc: float, src: dict, cfg_name: str,
                fasc: np.ndarray | None = None, structure: bool = True) -> g2.GeoResult:
    thr = src["primary"][2]
    cfg = make_cfg(cfg_name, thr, src["edges"])
    if not structure:
        cfg = cfg.with_(compute_structure=False)
    A = banks.get(ds, iid, src["primary"])
    if A is None:
        return g2._nan_result("missing_maps")
    F = fasc if fasc is not None else banks.main.prob(ds, "ours_fasc384_noflip", iid)
    if F is None:
        return g2._nan_result("missing_fasc")
    up = lo = None
    if src["edges"]:
        up = banks.get(ds, iid, ("v2", "upper", 0.5))
        lo = banks.get(ds, iid, ("v2", "lower", 0.5))
    fbs = []
    for name, spec in src["fallbacks"]:
        fbs.append((name, banks.get(ds, iid, spec), spec[2]))
    if img is not None and img.shape != A.shape:
        img = cv2.resize(img, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
    return g2.measure_apo(A, F, img, ppc, cfg, apo_upper=up, apo_lower=lo, fallbacks=fbs)


def cmd_measure(args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    src = SOURCES[args.source]
    path = CACHE / f"{args.dataset}__{args.source}__{args.cfg}.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(prev.image_id) if len(prev) else set()
    banks = Banks(Path(args.v2_root), args.v2_model)
    t0 = time.time()
    rows, k = [], 0

    def flush():
        pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)

    if args.dataset == "train_oof":
        meta = train_meta()
        it = iterate_train(meta)
    else:
        it = ((iid, img, ppc, extra) for iid, img, ppc, extra in bc.iterate(args.dataset))
    for item in it:
        if args.limit and k >= args.limit:
            break
        if args.dataset == "train_oof":
            iid, img, r = item
            if iid in done:
                continue
            extra = dict(layout=r.layout, orig=r.orig, fold=int(r.fold), ppc_orig=r.ppc)
            if args.source == "gt":
                m = np.asarray(Image.open(TRAIN_MSK / f"{iid}.tif").convert("L")) > 127
                im = cv2.resize(img, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_AREA) \
                    if img.shape != m.shape else img
                cfg = make_cfg(args.cfg, 0.5, False).with_(compute_structure=False)
                res = g2.measure_apo(m.astype(np.float32), np.zeros(m.shape, np.float32), im, r.ppc, cfg)
                ppc = r.ppc
            else:
                # maps live on the (stretched) image grid: vertical px -> mm through the original height
                oh = float(str(r.orig).split("x")[1]) if isinstance(r.orig, str) else img.shape[0]
                ppc = r.ppc * img.shape[0] / oh if np.isfinite(r.ppc) else np.nan
                res = measure_one(banks, "train_oof", iid, img, ppc, src, args.cfg,
                                  fasc=np.zeros(img.shape, np.float32), structure=False)
        else:
            iid, img, ppc, extra = item
            if iid in done:
                continue
            try:
                res = measure_one(banks, args.dataset, iid, img, ppc, src, args.cfg)
            except Exception as e:  # keep going; record the failure
                res = g2._nan_result(f"error:{type(e).__name__}:{e}")
        row = dict(image_id=iid, px_per_cm=ppc, **extra, **res.as_row())
        rows.append(row)
        k += 1
        if k % 50 == 0:
            flush()
            print(f"  {args.dataset}/{args.source}/{args.cfg}: {k} new, {time.time() - t0:.0f}s", flush=True)
    flush()
    banks.close()
    print(f"done {path.name}: {k} new rows in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------- report helpers
def load(ds: str, source: str, cfg: str = "v2e", complete: bool = True) -> pd.DataFrame | None:
    p = CACHE / f"{ds}__{source}__{cfg}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).drop_duplicates("image_id", keep="last").set_index("image_id")
    if complete and len(d) < EXPECTED[ds]:
        return None
    return d


def mt_stats(pred: pd.Series, gt: pd.Series) -> dict:
    ids = pred.index.intersection(gt.index)
    p, g = pred.loc[ids].astype(float), gt.loc[ids].astype(float)
    ok = p.notna() & g.notna()
    e = (p - g)[ok]
    return dict(n=int(ok.sum()), n_nan=int((~p.notna() & g.notna()).sum()), mae=float(e.abs().mean()),
                medae=float(e.abs().median()), bias=float(e.mean()), mt_n=float(e.abs().mean() / TOL["mt_mm"]),
                p90=float(e.abs().quantile(0.9)) if len(e) else np.nan)


def fill_mt(s: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    s = s.astype(float).copy()
    s[s < 5] = np.nan
    if groups is not None:
        s = s.fillna(s.groupby(groups).transform("median"))
    return s.fillna(s.median())


def fused_mt(a: pd.DataFrame, b: pd.DataFrame, mode: str, col: str = "mt_mm") -> pd.Series:
    """MT fusion of source a (apo_v2 chain) and b (ours chain) per image.

    avg      : mean when both come from their primary map, else the primary one, else a (chain)
    conf     : the value with the larger mt_conf (ties -> a)
    a_first  : a unless a had to fall back and b did not
    b_first  : b unless b had to fall back and a did not
    """
    ids = a.index.intersection(b.index)
    va, vb = a.loc[ids, col].astype(float), b.loc[ids, col].astype(float)
    pa = a.loc[ids, "d_apo_source"].eq("primary") & va.notna() & (va >= 5)
    pb = b.loc[ids, "d_apo_source"].eq("primary") & vb.notna() & (vb >= 5)
    out = va.copy()
    if mode == "avg":
        out = np.where(pa & pb, 0.5 * (va + vb), np.where(pa, va, np.where(pb, vb, va.fillna(vb))))
    elif mode == "conf":
        ca, cb = a.loc[ids, "mt_conf"].fillna(0), b.loc[ids, "mt_conf"].fillna(0)
        ca = ca.where(va.notna() & (va >= 5), -1)
        cb = cb.where(vb.notna() & (vb >= 5), -1)
        out = np.where(cb > ca, vb, va)
    elif mode == "a_first":
        out = np.where(~pa & pb, vb, va.fillna(vb))
    elif mode == "b_first":
        out = np.where(~pb & pa, va, vb.fillna(va))
    else:
        raise ValueError(mode)
    return pd.Series(np.asarray(out, float), index=ids)


def md_table(df: pd.DataFrame, cols=None, nd: int = 3) -> str:
    df = df if cols is None else df[cols]
    df = df.round(nd)
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- report
def report_train_oof(lines: list, summary: dict) -> None:
    gt = load("train_oof", "gt", complete=False)
    if gt is None:
        return
    rows, by = [], []
    for s in ("v2", "v2m", "v2n"):
        d = load("train_oof", s, complete=False)
        if d is None:
            continue
        for var in ("mt_mm", "mt_raw", "mt_quad"):
            if var not in d:
                continue
            st = mt_stats(d[var], gt["mt_mm"])
            ids = d.index.intersection(gt.index)
            rel = ((d.loc[ids, var] - gt.loc[ids, "mt_mm"]).abs() / gt.loc[ids, "mt_mm"]).median()
            ee = (d.loc[ids, var] - gt.loc[ids, "mt_mm"]).dropna()
            core = ee[ee.abs() < 3.0]
            rows.append(dict(source=s, var=var, **st, rel_med=float(rel), n_gross3=int((ee.abs() >= 3).sum()),
                             mae_core=float(core.abs().mean()), bias_core=float(core.mean()),
                             fail=int(d[var].isna().sum()), n_fb=int((d.d_apo_source != "primary").sum())))
        e = (d["mt_mm"] - gt["mt_mm"]).rename("err")
        t = pd.concat([e, gt[["orig", "layout"]]], axis=1).dropna(subset=["err"])
        g = t.groupby(["orig", "layout"]).err.agg(n="size", mae=lambda v: v.abs().mean(),
                                                    medae=lambda v: v.abs().median(), bias="mean",
                                                    n_gross3=lambda v: int((v.abs() >= 3).sum()),
                                                    mae_core=lambda v: v[v.abs() < 3].abs().mean(),
                                                    bias_core=lambda v: v[v.abs() < 3].mean()).reset_index()
        g.insert(0, "source", s)
        by.append(g)
    if not rows:
        return
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "train_oof_mt.csv", index=False)
    lines += ["### train_oof: MT vs GT-mask inner-edge MT (honest out-of-fold apo_v2 maps, mm)", "",
              md_table(T), ""]
    if by:
        B = pd.concat(by)
        B.to_csv(OUT / "train_oof_mt_by_layout.csv", index=False)
        lines += ["#### by original size / training-image layout (MT = mt_mm, mm)", "", md_table(B), ""]
    summary["train_oof"] = T.to_dict("records")


def report_static(lines: list, summary: dict) -> None:
    G = bc.static_gts()
    gt = G["mean7"]
    intrain = [i for i in gt.index if i not in bc.CLEAN]
    rows = []
    srcs = {}
    for s in ("ours", "v2", "v2m", "v2n", "oursE"):
        d = load("static", s)
        if d is None:
            continue
        srcs[s] = d
    if "v2" in srcs and "ours" in srcs:
        for mode in ("avg", "conf", "a_first"):
            f = fused_mt(srcs["v2"], srcs["ours"], mode)
            srcs[f"fuse_{mode}"] = pd.DataFrame({"mt_mm": f}).join(srcs["v2"].drop(columns=["mt_mm"]))
    for s, d in srcs.items():
        for var in ("mt_mm",) if s.startswith("fuse") else ("mt_mm", "mt_raw"):
            if var not in d:
                continue
            mt = fill_mt(d[var])
            r = dict(source=s, var=var)
            for sub, ids in (("s35", list(gt.index)), ("c10", bc.CLEAN), ("in25", intrain)):
                st = mt_stats(mt.loc[ids], gt.loc[ids, "mt_mm"])
                r[f"{sub}_mae"], r[f"{sub}_bias"] = st["mae"], st["bias"]
            r["s35_2r_mae"] = float(np.mean([mt_stats(mt, G[f"R7+{x}"]["mt_mm"])["mae"] for x in bc.RATERS[:-1]]))
            r["s35_R7_mae"] = mt_stats(mt, G["R7"]["mt_mm"])["mae"]
            r["fail"] = int(d[var].isna().sum() + (d[var] < 5).sum())
            if var == "mt_mm" and "pa_fusion" in d:
                pr = bc.fill(bc.heads(d, "fusion", "ensemble"))
                pr["mt_mm"] = mt
                m = bc.s_metric(pr, gt)
                r.update(S35=m["S"], S35_pa=m["pa_n"], S35_fl=m["fl_n"], c10_S=bc.s_metric(pr.loc[bc.CLEAN], gt)["S"])
            rows.append(r)
    if not rows:
        return
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "static_mt.csv", index=False)
    lines += ["### static benchmark: MT vs experts (mm MAE / bias; S = full score with PA fusion / FL ensemble)", "",
              "s35 = all 35 (25 in training of both models), c10 = clean 10 (honest), in25 = the 25 in-train", "",
              md_table(T), ""]
    summary["static"] = T.to_dict("records")


def report_static_oof(lines: list, summary: dict) -> None:
    """Static images that ARE apo training images: compare the out-of-fold apo_v2 prediction on the
    (stretched) training copy with the expert MT -- the honest apo_v2 number for the in-train ones."""
    d = load("train_oof", "v2", complete=False)
    if d is None or not BENCH_MATCH.exists():
        return
    bm = pd.read_csv(BENCH_MATCH)
    bm = bm[(bm.kind == "apo")].sort_values("full_corr", ascending=False).drop_duplicates("bench")
    bm = bm[bm.full_corr > 0.9]
    gt = bc.static_gts()["mean7"]
    st = load("static", "v2")
    so = load("static", "ours")
    rows = []
    for r in bm.itertuples():
        if r.train not in d.index:
            continue
        rows.append(dict(bench=r.bench, train=r.train, corr=r.full_corr, expert=gt.loc[r.bench, "mt_mm"],
                         v2_oof=d.loc[r.train, "mt_mm"], v2_ens=st.loc[r.bench, "mt_mm"] if st is not None else np.nan,
                         ours=so.loc[r.bench, "mt_mm"] if so is not None else np.nan))
    if not rows:
        return
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "static_oof_check.csv", index=False)
    agg = {c: float((T[c] - T.expert).abs().mean()) for c in ("v2_oof", "v2_ens", "ours")}
    lines += [f"### static images present in the apo training set ({len(T)} matched, corr > 0.9): MT MAE vs experts", "",
              "v2_oof = apo_v2 out-of-fold map of the training copy (honest); v2_ens / ours = in-sample", "",
              md_table(T), "", "MAE: " + json.dumps({k: round(v, 3) for k, v in agg.items()}), ""]
    summary["static_oof"] = dict(n=len(T), **agg)


def report_vl(lines: list, summary: dict) -> None:
    v = bc.vl_gt()
    rows, srcs = [], {}
    for s in ("ours", "v2", "v2m", "v2n", "oursE"):
        d = load("vl", s)
        if d is not None:
            srcs[s] = d
    if "v2" in srcs and "ours" in srcs:
        for mode in ("avg", "conf", "a_first"):
            f = fused_mt(srcs["v2"], srcs["ours"], mode)
            srcs[f"fuse_{mode}"] = pd.DataFrame({"mt_mm": f}).join(srcs["v2"].drop(columns=["mt_mm"]))
    for s, d in srcs.items():
        mt = fill_mt(d["mt_mm"])
        st = mt_stats(mt, v["mt_mm"])
        r = dict(source=s, mt_lines_mae=st["mae"], mt_lines_bias=st["bias"], fail=int((d.mt_mm.isna() | (d.mt_mm < 5)).sum()))
        if "mt_raw" in d:
            r["mt_raw_lines_mae"] = mt_stats(fill_mt(d["mt_raw"]), v["mt_mm"])["mae"]
        if "pa_fusion" in d:
            pr = bc.fill(bc.heads(d, "fusion", "ensemble"))
            m = bc.s_metric(pr, v, comps=("pa_deg", "fl_mm"))
            r.update(S_pafl=m["S"], pa_n=m["pa_n"], fl_n=m["fl_n"])
            k = d.fl_ensemble * np.sin(np.radians(d.pa_fusion)) / d.mt_mm
            r["k_FLsinPA_MT"] = float(k.median())
        rows.append(r)
    if not rows:
        return
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "vl_mt.csv", index=False)
    lines += ["### VL (Esaote, 249): MT vs the drawn-line gap (loose), PA/FL side effects vs experts", "", md_table(T), ""]
    summary["vl"] = T.to_dict("records")
    # separate "recovered pairs" from "better curves": images where ours AND v2 use their primary map
    if "ours" in srcs and "v2" in srcs:
        o, a = srcs["ours"], srcs["v2"]
        ids = [i for i in v.index if i in o.index and i in a.index and o.loc[i, "d_apo_source"] == "primary"
               and a.loc[i, "d_apo_source"] == "primary" and np.isfinite(o.loc[i, "mt_mm"]) and np.isfinite(a.loc[i, "mt_mm"])]
        sub = []
        for s in ("ours", "v2", "v2m", "oursE"):
            if s not in srcs:
                continue
            d = srcs[s].loc[ids]
            pr = bc.fill(bc.heads(d, "fusion", "ensemble"))
            r = dict(source=s, n=len(ids), mt_lines_mae=mt_stats(d["mt_mm"], v["mt_mm"])["mae"])
            m = bc.s_metric(pr, v, comps=("pa_deg", "fl_mm"))
            r.update(S_pafl=m["S"], pa_n=m["pa_n"], fl_n=m["fl_n"], fl_bias=m["fl_bias"])
            for grp in ("young", "old"):
                gi = [i for i in ids if i.startswith(grp)]
                r[f"S_pafl_{grp}"] = bc.s_metric(pr.loc[gi], v, comps=("pa_deg", "fl_mm"))["S"]
            sub.append(r)
        S = pd.DataFrame(sub)
        lines += ["#### VL, images where ours and v2 both find a pair with their own map", "", md_table(S), ""]
        summary["vl_common"] = S.to_dict("records")


def report_dynamic(lines: list, summary: dict) -> None:
    dg = bc.dynamic_gt()
    rows = []
    for s in ("ours", "v2", "v2m", "oursE"):
        d, dh = load("dynamic", s, "v2e"), load("dynamic", s, "v2eh")
        if d is None:
            continue
        r = dict(source=s, fail=int(d.mt_mm.isna().sum()))
        for fl in ("ensemble", "straight", "constructed"):
            pr = bc.fill(bc.heads(d, "fusion", fl))
            m = bc.s_metric(pr, dg, comps=("fl_mm",))
            r[f"fl_{fl}_mae"], r[f"fl_{fl}_bias"] = m["fl_n"] * TOL["fl_mm"], m["fl_bias"]
        if dh is not None:
            pr = bc.fill(bc.heads(dh, "fusion", "ensemble"))
            m = bc.s_metric(pr, dg, comps=("pa_deg",))
            r["pa_h_mae"], r["pa_h_bias"] = m["pa_n"] * TOL["pa_deg"], m["pa_bias"]
        r["mt_median"] = float(d.mt_mm.median())
        r["mt_frame_sd"] = float(d.mt_mm.diff().abs().median())
        rows.append(r)
    if not rows:
        return
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "dynamic.csv", index=False)
    lines += ["### dynamic (Telemed GM, 168 frames): FL (mm) / PA vs horizontal side effects; MT frame-to-frame jitter", "",
              md_table(T), ""]
    summary["dynamic"] = T.to_dict("records")


def report_recipes(lines: list, summary: dict) -> None:
    """The exact C1 / C2 recipes (build_candidates.candidate_bench) with each aponeurosis source."""
    G, v, dg = bc.static_gts(), bc.vl_gt(), bc.dynamic_gt()
    rows = []
    for s in ("ours", "v2", "v2m", "oursE"):
        st, vl = load("static", s), load("vl", s)
        dd, dh = load("dynamic", s, "v2e"), load("dynamic", s, "v2eh")
        if any(x is None for x in (st, vl, dd, dh)):
            continue
        sg = bc.static_groups(st.index)

        def rec(d, groups, mode):
            p = bc.heads(d, "fusion", "ensemble")
            p.loc[p.mt_mm < 5, list(C3)] = np.nan
            p = bc.fill(p, groups)
            if mode == "C2":
                p = bc.shrink(p, d.confidence, groups, 1.0)
            return p
        for mode in ("C1", "C2"):
            r = {"source": s, "recipe": mode}
            p = rec(st, sg, mode)
            m = bc.s_metric(p, G["mean7"])
            r.update(s35=m["S"], s35_pa=m["pa_n"], s35_fl=m["fl_n"], s35_mt=m["mt_n"])
            r["c10"] = bc.s_metric(p.loc[bc.CLEAN], G["mean7"])["S"]
            r["2r"] = float(np.mean([bc.s_metric(p, G[f"R7+{x}"])["S"] for x in bc.RATERS[:-1]]))
            r["R7"] = bc.s_metric(p, G["R7"])["S"]
            p = rec(vl, pd.Series("vl", index=vl.index), mode)
            m = bc.s_metric(p, v, ("pa_deg", "fl_mm"))
            r.update(vl=m["S"], vl_pa=m["pa_n"], vl_fl=m["fl_n"])
            r["vl_mt_lines"] = bc.s_metric(p, v, ("mt_mm",))["mt_n"]
            r["vl_young3"] = bc.s_metric(p.loc[[i for i in p.index if i.startswith("young")]], v)["S"]
            r["vl_old_pafl"] = bc.s_metric(p.loc[[i for i in p.index if i.startswith("old")]], v, ("pa_deg", "fl_mm"))["S"]
            p = rec(dh, pd.Series("d", index=dh.index), mode)
            r["dyn_pa_h"] = bc.s_metric(p, dg, ("pa_deg",))["pa_n"]
            p = rec(dd, pd.Series("d", index=dd.index), mode)
            r["dyn_fl"] = bc.s_metric(p, dg, ("fl_mm",))["fl_n"]
            rows.append(r)
    if not rows:
        return
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "bench_recipes.csv", index=False)
    lines += ["### C1 / C2 recipes (PA fusion, FL ensemble; C2 = confidence shrinkage of PA/FL) with each aponeurosis "
              "source -- normalised MAE (official metric), lower is better", "", md_table(T), ""]
    summary["bench_recipes"] = T.to_dict("records")


def report_test(lines: list, summary: dict) -> None:
    lay = bc.test_layout()
    srcs = {s: load("test", s) for s in ("ours", "v2", "v2m", "v2n", "oursE")}
    srcs = {k: v for k, v in srcs.items() if v is not None}
    if not srcs:
        return
    rows = []
    for s, d in srcs.items():
        g = lay.loc[d.index, "group"]
        src = d.d_apo_source.fillna("none")
        r = dict(source=s, n=len(d), primary=int((src == "primary").sum()),
                 **{f"fb_{k}": int((src == k).sum()) for k in ("ours", "dlt")}, none=int((src == "none").sum()),
                 mt_nan=int(d.mt_mm.isna().sum()), mt_lt5=int((d.mt_mm < 5).sum()),
                 lumify_primary=int(((src == "primary") & (g == "philips_lumify")).sum()))
        rows.append(r)
    T = pd.DataFrame(rows)
    lines += ["### test: aponeurosis pair source per image (309)", "", md_table(T), ""]
    summary["test_sources"] = T.to_dict("records")
    # Lumify images where our map has no band at all
    if "ours" in srcs:
        o = srcs["ours"]
        nbp = o["d_n_bands_primary"] if "d_n_bands_primary" in o else o["d_n_bands"]
        empty = o.index[(nbp.fillna(0) == 0)]
        lum = [i for i in empty if lay.loc[i, "group"] == "philips_lumify"]
        rec = []
        for i in lum:
            r = dict(image_id=i, ours_src=o.loc[i, "d_apo_source"], ours_mt=o.loc[i, "mt_mm"])
            for s in ("v2", "v2n"):
                if s in srcs:
                    r[f"{s}_src"], r[f"{s}_mt"] = srcs[s].loc[i, "d_apo_source"], srcs[s].loc[i, "mt_mm"]
                    r[f"{s}_conf"] = srcs[s].loc[i, "mt_conf"]
            rec.append(r)
        if rec:
            R = pd.DataFrame(rec)
            lines += [f"#### images where ours_apo384 has no band ({len(empty)}; Lumify {len(lum)})", "", md_table(R), ""]
            summary["lumify_empty"] = R.to_dict("records")
    # per-group MT distributions
    g_rows = []
    for s, d in srcs.items():
        mt = fill_mt(d.mt_mm, lay.loc[d.index, "group"])
        for grp, v in mt.groupby(lay.loc[d.index, "group"]):
            g_rows.append(dict(source=s, group=grp, n=len(v), median=v.median(), q25=v.quantile(.25),
                               q75=v.quantile(.75), mean=v.mean()))
    Gt = pd.DataFrame(g_rows)
    Gt.to_csv(OUT / "test_group_mt.csv", index=False)
    piv = Gt.pivot(index="group", columns="source", values="median").reset_index()
    lines += ["#### per-group median MT (mm; MT<5 -> group fill)", "", md_table(piv, nd=2), ""]
    # disagreement
    if "v2" in srcs and "ours" in srcs:
        a, b = srcs["v2"], srcs["ours"]
        ids = a.index.intersection(b.index)
        dd = pd.DataFrame({"group": lay.loc[ids, "group"], "clip": lay.loc[ids, "clip"],
                           "mt_v2": a.loc[ids, "mt_mm"], "mt_ours": b.loc[ids, "mt_mm"],
                           "src_v2": a.loc[ids, "d_apo_source"], "src_ours": b.loc[ids, "d_apo_source"],
                           "conf_v2": a.loc[ids, "mt_conf"], "conf_ours": b.loc[ids, "mt_conf"],
                           "pair_v2": a.loc[ids, "d_pair"], "pair_ours": b.loc[ids, "d_pair"],
                           "nb_v2": a.loc[ids, "d_n_bands"], "nb_ours": b.loc[ids, "d_n_bands"]})
        dd["diff"] = dd.mt_v2 - dd.mt_ours
        dd["absdiff"] = dd["diff"].abs()
        dd.sort_values("absdiff", ascending=False).to_csv(OUT / "test_disagreement.csv")
        both = dd[(dd.src_v2 == "primary") & (dd.src_ours == "primary")]
        agg = both.groupby("group")["diff"].agg(n="size", median_diff="median", mad=lambda v: v.abs().median(),
                                               gt1mm=lambda v: int((v.abs() > 1).sum()),
                                               gt3mm=lambda v: int((v.abs() > 3).sum())).reset_index()
        lines += ["#### MT(v2) - MT(ours) where both use their own primary map", "", md_table(agg), "",
                  "largest disagreements (all images):", "",
                  md_table(dd.sort_values("absdiff", ascending=False).head(15).reset_index(),
                           ["image_id", "group", "mt_v2", "mt_ours", "src_v2", "src_ours", "conf_v2", "conf_ours",
                            "nb_v2", "nb_ours"], nd=2), ""]
        summary["test_disagreement"] = agg.to_dict("records")
        dist = {}
        for mode in ("avg", "conf", "a_first", "b_first"):
            f = fill_mt(fused_mt(a, b, mode), lay.loc[ids, "group"])
            dist[mode] = float((f - fill_mt(b.loc[ids, "mt_mm"], lay.loc[ids, "group"])).abs().mean() / 3.0)
        dist["v2"] = float((fill_mt(a.loc[ids, "mt_mm"], lay.loc[ids, "group"])
                            - fill_mt(b.loc[ids, "mt_mm"], lay.loc[ids, "group"])).abs().mean() / 3.0)
        lines += ["MT-component distance to ours (mean |dMT| / 3 mm, raw per image): " +
                  json.dumps({k: round(v, 4) for k, v in dist.items()}), ""]
        summary["test_mt_distance_to_ours"] = dist
    # full C1 / C2 recipes (PA fusion, FL ensemble, MT<5 -> failure, group fill, clip median, bounds)
    # with each aponeurosis source; distances to the submitted C2 (LB 0.38974)
    c2_path = ROOT / "artifacts/candidates/C2_ours_v2e_fusion_ensemble_shrink1.0.csv"
    ref = pd.read_csv(c2_path).set_index("image_id") if c2_path.exists() else None
    built, drows = {}, []
    for s, d in srcs.items():
        for rec in ("C1", "C2"):
            pred = bc.heads(d, "fusion", "ensemble")
            pred.loc[pred.mt_mm < 5.0, list(C3)] = np.nan
            pred = bc.fill(pred, lay.loc[d.index, "group"])
            if rec == "C2":
                pred = bc.shrink(pred, d.confidence, lay.loc[d.index, "group"], 1.0, ("pa_deg", "fl_mm"), "conf")
            sub = bc.postprocess(pred, lay)
            built[(s, rec)] = sub
            r = dict(source=s, recipe=rec)
            if ref is not None:
                r.update({f"toC2_{k}": v for k, v in bc.official_distance(sub, ref).items()})
            if ("ours", rec) in built and s != "ours":
                r.update({f"toOurs_{k}": v for k, v in bc.official_distance(sub, built[("ours", rec)]).items()})
            drows.append(r)
    if drows:
        D = pd.DataFrame(drows)
        D.to_csv(OUT / "test_recipe_distances.csv", index=False)
        lines += ["#### test: full C1/C2 recipes per aponeurosis source -- official-metric distance to the submitted C2 "
                  "and to the same recipe on ours (diagnostic only, no labels)", "", md_table(D), ""]
        summary["test_recipe_distances"] = D.to_dict("records")


CACHE_COLS = ["image_id", "group", "clip", "px_per_cm", "mt_mm", "mt_lin", "mt_raw", "mt_quad", "mt_lin_c", "mt_lin_e",
              "mt_lin_sd", "mt_q", "mt_all", "mt_conf", "confidence", "flags", "d_apo_source", "d_n_bands", "d_pair",
              "d_pair_fallback", "d_apo_fallback_bands", "d_shared_frac", "d_apo_converge", "d_edge_cov_sup",
              "d_edge_cov_deep", "d_sup_thick_px", "d_deep_thick_px", "sup_p1_s", "sup_p1_b", "sup_p2_a", "sup_p2_b",
              "sup_p2_c", "sup_x0", "sup_x1", "sup_res_px", "deep_p1_s", "deep_p1_b", "deep_p2_a", "deep_p2_b",
              "deep_p2_c", "deep_x0", "deep_x1", "deep_res_px", "s_lo", "s_hi", "px_mm", "pa_fusion", "fl_ensemble"]


def write_test_caches(lines: list, summary: dict) -> None:
    lay = bc.test_layout()
    out = {}
    for s in ("ours", "v2", "v2m", "oursE"):
        d = load("test", s)
        if d is None:
            continue
        d = d.copy()
        d["group"] = lay.loc[d.index, "group"]
        d["clip"] = lay.loc[d.index, "clip"]
        cols = [c for c in CACHE_COLS if c in d.reset_index().columns]
        d.reset_index()[cols].to_csv(OUT / f"test_mt_{s}.csv", index=False)
        out[s] = d
    if "v2" in out and "ours" in out:
        a, b = out["v2"], out["ours"]
        F = pd.DataFrame(index=a.index)
        F["group"], F["clip"] = a["group"], a["clip"]
        F["mt_v2"], F["mt_ours"] = a.mt_mm, b.mt_mm
        # recommended per-image MT (apo2 study): apo_v2 chain (band pair + edge-channel inner edges,
        # ours_apo384 -> dlt_apo fallbacks); MT < 5 mm is a failed pair (group-median fill downstream)
        F["mt_recommended"] = a.mt_mm.where(a.mt_mm >= 5)
        F["src_v2"], F["src_ours"] = a.d_apo_source, b.d_apo_source
        F["conf_v2"], F["conf_ours"] = a.mt_conf, b.mt_conf
        for mode in ("avg", "conf", "a_first", "b_first"):
            F[f"mt_fuse_{mode}"] = fused_mt(a, b, mode)
        F.reset_index().to_csv(OUT / "test_mt_fused.csv", index=False)
        lines += [f"test caches: {', '.join(f'test_mt_{s}.csv' for s in out)}, test_mt_fused.csv", ""]


def cmd_report(args) -> None:
    lines = ["# apo2: aponeurosis / MT source study", ""]
    summary = {}
    for f in (report_train_oof, report_static_oof, report_static, report_vl, report_dynamic, report_recipes, report_test,
              write_test_caches):
        try:
            f(lines, summary)
        except Exception as e:  # report what we can
            import traceback
            lines += [f"**{f.__name__} failed: {e!r}**", "", "```", traceback.format_exc()[-1500:], "```", ""]
    (OUT / "apo2_report.md").write_text("\n".join(lines))
    (OUT / "apo2_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\n".join(lines))


# ----------------------------------------------------------------------------------- overlay
def _poly_curve(r, nm, xs):
    y = np.polyval([r[f"{nm}_p2_a"], r[f"{nm}_p2_b"], r[f"{nm}_p2_c"]], np.clip(xs, r[f"{nm}_x0"], r[f"{nm}_x1"]))
    return y


def cmd_overlay(args) -> None:
    a, b = load("test", "v2"), load("test", "ours")
    if a is None or b is None:
        print("need complete test caches for v2 and ours")
        return
    dd = pd.read_csv(OUT / "test_disagreement.csv").set_index("image_id")
    ids = list(dd.index[: args.n])
    tiles = []
    for iid in ids:
        img = np.asarray(Image.open(bc.TEST_DIR / iid).convert("L"))
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for d, col in ((a, (0, 200, 255)), (b, (255, 120, 0))):
            r = d.loc[iid]
            if not np.isfinite(r.get("sup_p2_a", np.nan)):
                continue
            xs = np.arange(int(min(r.sup_x0, r.deep_x0)), int(max(r.sup_x1, r.deep_x1)))
            for nm in ("sup", "deep"):
                ys = _poly_curve(r, nm, xs)
                pts = np.stack([xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], False, col, 2)
        h = 360
        vis = cv2.resize(vis, (int(vis.shape[1] * h / vis.shape[0]), h), interpolation=cv2.INTER_AREA)
        txt = f"{iid} v2 {a.loc[iid, 'mt_mm']:.1f} ({a.loc[iid, 'd_apo_source']}) ours {b.loc[iid, 'mt_mm']:.1f} ({b.loc[iid, 'd_apo_source']})"
        cv2.putText(vis, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(vis)
    if not tiles:
        return
    W = 560
    tiles = [cv2.copyMakeBorder(cv2.resize(t, (W, int(t.shape[0] * W / t.shape[1]))), 2, 2, 2, 2,
                                cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
    rows = [np.concatenate(tiles[i:i + 3] + [np.full_like(tiles[0], 40)] * (3 - len(tiles[i:i + 3])), 1)
            for i in range(0, len(tiles), 3)]
    out = OUT / f"disagreement_top{len(ids)}.png"
    cv2.imwrite(str(out), np.concatenate(rows, 0))
    print("wrote", out, "(orange-yellow = apo_v2, blue = ours)")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--dataset", required=True, choices=list(EXPECTED))
    m.add_argument("--source", required=True, choices=list(SOURCES))
    m.add_argument("--cfg", default="v2e", choices=["v2e", "v2eh"])
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--v2-root", default=str(V2_BANK))
    m.add_argument("--v2-model", default=V2_MODEL)
    sub.add_parser("report")
    o = sub.add_parser("overlay")
    o.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    {"measure": cmd_measure, "report": cmd_report, "overlay": cmd_overlay}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
