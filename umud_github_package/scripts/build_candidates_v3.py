"""Assemble the v3 pipeline: apo_v2 aponeuroses + learned-field PA + empirical-Bayes shrinkage.

Light by design (laptop rules): no neural network, one image at a time, lazy npz keys, one process,
< 0.5 GB RSS.

Subcommands
-----------
measure --dataset {static,dynamic,vl,test,train142} --apo {v2,ours}
    One consistent per-image pass that combines the three agents' pieces:
      aponeuroses : v2   = repaired apo_v2 band @0.5, inner edges from the edge channels, fallbacks
                           ours_apo384 @0.8 then dlt_apo @0.5 (g2.measure_apo; apo2 agent)
                    ours = ours_apo384_noflip @0.8 + dlt_apo fallback (the C1/C2 maps)
      fragments   : ours_fasc384_noflip @0.95 (C1); train142: fasc_orient_v1 seg (out of fold)
      field       : fasc_orient_v1 (seg, cos2, sin2) -> g2.field_architecture (orient agent)
      geometry    : recommended v2 with end-linear extrapolation (v2e)
    Extra columns: the constructed straight fascicle (C1 recipe, fragment sign) on a PA grid
    (``cg_<10*PA>``, 3..45 deg step 0.5) so that ensemble FL can be rebuilt at ANY fused PA later,
    and for dynamic the PA heads against the horizontal (``*_h``, the benchmark convention).
    Cache: artifacts/candidates_v3/cache/<dataset>__<apo>.csv (resumable).
    train142: the 142 training images with both GT masks, both models out of fold (apo_v2fix
    train_oof band+edges, fasc_orient_v1 train_oof seg+field), GT-mask frame, px units.
bench   Benchmark tables for every recipe (apo source x PA head x FL head x shrinkage) on static
        (mean7 / clean10 / 2-rater / R7), VL-244 (lodo3 + lodo4, young / old), dynamic, train142;
        the EB error model is always fitted without the scored dataset. Also test-set distances to
        C2 / C1 / v1 and the public reference CSVs (diagnostic only) -> bench_v3.csv / .md.
build   Candidate CSVs C4..C6 (validated) + per-image tables + candidates_v3_report.json (group medians,
        FL sin(PA)/MT per group, distances, pairwise distances).
"""
from __future__ import annotations

import argparse
import json
import math
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
from umud import shrinkage as sh  # noqa: E402
import build_candidates as bc  # noqa: E402
import eval_apo_v2 as ea  # noqa: E402
import eval_orient as eo  # noqa: E402

warnings.filterwarnings("ignore")

OUT = ROOT / "artifacts/candidates_v3"
CACHE = Path(__import__("os").environ.get("V3_CACHE", OUT / "cache"))  # override for smoke tests
FCFG = g2.FieldConfig()
PA_GRID = np.round(np.arange(3.0, 45.0001, 0.5), 1)
GRID_COLS = [f"cg_{int(round(10 * p)):03d}" for p in PA_GRID]
APO_SRC = {"v2": ea.SOURCES["v2"], "ours": ea.SOURCES["ours"]}
EXPECTED = {"static": 35, "dynamic": 168, "vl": 249, "test": 309, "train142": 142}


# ----------------------------------------------------------------------------------- measure
def frag_sign(res: g2.GeoResult) -> float:
    """Fascicle inclination sign exactly as measure() uses it for fl_constructed."""
    valid = res.fragments or []
    if valid:
        return 1.0 if sum(q["length"] * np.sign(q["rel"]) for q in valid) >= 0 else -1.0
    s = res.diag.get("st_sign", 1.0)
    return 1.0 if (s is None or not np.isfinite(s)) else float(s)


def constructed_grid(res: g2.GeoResult, pa_reference: str = "deep") -> dict:
    cv = res.curves or {}
    if "sup_curve" not in cv:
        return {c: np.nan for c in GRID_COLS}
    sup, deep, s_lo, s_hi, mt_px, mm = (cv[k] for k in ("sup_curve", "deep_curve", "s_lo", "s_hi", "mt_px", "mm"))
    W = res.diag.get("W", 1000)
    xgrid = np.linspace(-3 * W, 4 * W, 6001)
    sgn = frag_sign(res)
    out = {}
    for p, c in zip(PA_GRID, GRID_COLS):
        cons = [g2.constructed_fl(sup, deep, s_lo + f * (s_hi - s_lo), float(p), sgn, mt_px, pa_reference, xgrid)
                for f in (0.4, 0.5, 0.6)]
        cons = [v for v in cons if np.isfinite(v)]
        out[c] = float(np.mean(cons)) * mm if cons else np.nan
    return out


def apo_inputs(banks: ea.Banks, ds: str, iid: str, apo: str):
    src = APO_SRC[apo]
    A = banks.get(ds, iid, src["primary"])
    up = lo = None
    if src["edges"]:
        up = banks.get(ds, iid, ("v2", "upper", 0.5))
        lo = banks.get(ds, iid, ("v2", "lower", 0.5))
    fbs = [(name, banks.get(ds, iid, spec), spec[2]) for name, spec in src["fallbacks"]]
    return A, up, lo, fbs, src["primary"][2], src["edges"]


def measure_one(A, up, lo, fbs, thr, edges, F, fthr, img, ppc, orient, horizontal: bool = False,
                structure: bool = True) -> dict:
    cfg = ea.make_cfg("v2e", thr, edges).with_(fasc_thr=fthr)
    if not structure:
        cfg = cfg.with_(compute_structure=False)
    if img is not None and img.shape != A.shape:
        img = cv2.resize(img, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
    res = g2.measure_apo(A, F, img, ppc, cfg, apo_upper=up, apo_lower=lo, fallbacks=fbs, keep_details=True,
                         orient=orient)
    if orient is not None and res.curves:
        out = g2.field_architecture(res, orient, FCFG, cfg.pa_reference)
        res.variants.update(out["variants"])
        res.diag.update(out["diag"])
    row = eo.row_of(res)          # as_row + constructed FL at the orient agent's fused PAs (fl_c_fF, ...)
    row.update(constructed_grid(res, cfg.pa_reference))
    row["d_frag_sign_c"] = frag_sign(res) if res.curves else np.nan
    if horizontal:   # dynamic benchmark: PA heads against the horizontal
        cfg_h = cfg.with_(pa_reference="horizontal")
        rh = g2.measure_apo(A, F, img, ppc, cfg_h, apo_upper=up, apo_lower=lo, fallbacks=fbs, orient=orient)
        for k in ("pa_global", "pa_structure", "pa_fusion"):
            row[f"{k}_h"] = rh.variants.get(k, np.nan)
        row["mt_mm_h"] = rh.mt_mm
    return row


def cmd_measure(args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.dataset == "train142":
        return measure_train142(args)
    path = CACHE / f"{args.dataset}__{args.apo}.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(prev.image_id) if len(prev) else set()
    banks = ea.Banks()
    obank = g2.MaskBank(eo.OBANK_ROOT)
    kmap = eo.orient_keymap(args.dataset)
    t0, rows, k = time.time(), [], 0

    def flush():
        pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)

    for iid, img, ppc, extra in bc.iterate(args.dataset):
        if iid in done:
            continue
        if args.limit and k >= args.limit:
            break
        okey = kmap.get(iid, Path(iid).stem) if args.dataset == "vl" else Path(iid).stem
        A, up, lo, fbs, thr, edges = apo_inputs(banks, args.dataset, iid, args.apo)
        F = banks.main.prob(args.dataset, "ours_fasc384_noflip", iid)
        orient = eo.load_orient(obank, args.dataset, okey)
        if A is None or F is None or orient is None or orient[1].shape != A.shape:
            r = g2._nan_result("missing_maps").as_row()
        else:
            try:
                r = measure_one(A, up, lo, fbs, thr, edges, F, 0.95, img, ppc, orient,
                                horizontal=args.dataset == "dynamic")
            except Exception as e:  # keep going; record the failure
                r = g2._nan_result(f"error:{type(e).__name__}:{e}").as_row()
        rows.append(dict(image_id=iid, okey=okey, px_per_cm=ppc, **extra, **r))
        del A, up, lo, fbs, F, orient
        k += 1
        if k % 25 == 0:
            flush()
            print(f"  {args.dataset}/{args.apo}: {k} new, {time.time() - t0:.0f}s", flush=True)
    flush()
    banks.close()
    obank.close()
    print(f"done {path.name}: {k} new rows in {time.time() - t0:.0f}s", flush=True)


def measure_train142(args) -> None:
    """Both models out of fold on the 142 dual-GT training images (GT-mask frame, px units)."""
    path = CACHE / "train142__oof.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(prev.image) if len(prev) else set()
    pairs = pd.read_csv(eo.PAIRS)
    reps = eo.rep_map()
    obank = g2.MaskBank(eo.OBANK_ROOT)
    abank = g2.MaskBank(ea.V2_BANK)
    t0, rows = time.time(), []
    rs = lambda a, shp: a if a is None or a.shape == shp else cv2.resize(a, (shp[1], shp[0]),  # noqa: E731
                                                                         interpolation=cv2.INTER_LINEAR)
    for k, r in enumerate(pairs.itertuples()):
        if r.apo in done:
            continue
        Ag = bc.gray(bc.COMP / "apo_masks_v1/apo_masks_new_model_v1" / f"{r.apo}.tif") > 0
        Fg = bc.gray(bc.COMP / "fasc_masks_v1/fasc_masks_new_model_v1" / f"{r.fasc}.tif") > 0
        if Ag.shape != Fg.shape:
            continue
        img = bc.gray(bc.COMP / "apo_imgs_v1/apo_images_new_model_v1" / f"{r.apo}.tif")
        if img.shape != Ag.shape:
            img = np.asarray(Image.fromarray(img).resize((Ag.shape[1], Ag.shape[0]), Image.BILINEAR))
        okey = reps.get(r.fasc, r.fasc)
        meta = dict(image=r.apo, fasc=r.fasc, okey=okey, layout=r.orig, H=Ag.shape[0], W=Ag.shape[1])
        o = eo.load_orient(obank, "train_oof", okey)
        A = abank.prob("train_oof", f"{ea.V2_MODEL}_band", r.apo)
        if o is None or A is None:
            rows.append(dict(meta, **g2._nan_result("missing_maps").as_row()))
            continue
        o = eo.to_mask_frame(o, Ag.shape)
        A = rs(A, Ag.shape)
        up = rs(abank.prob("train_oof", f"{ea.V2_MODEL}_upper", r.apo), Ag.shape)
        lo = rs(abank.prob("train_oof", f"{ea.V2_MODEL}_lower", r.apo), Ag.shape)
        try:
            rr = measure_one(A, up, lo, [], 0.5, True, o[0], 0.5, img, None, o)
        except Exception as e:
            rr = g2._nan_result(f"error:{type(e).__name__}:{e}").as_row()
        rows.append(dict(meta, **rr))
        if k % 10 == 9:
            pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
            print(f"  train142 {k + 1}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    obank.close()
    abank.close()
    print(f"done train142: {len(rows)} new rows in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------- heads
K0 = sh.K0
LAY = None


def layout() -> pd.DataFrame:
    global LAY
    if LAY is None:
        LAY = bc.test_layout()
    return LAY


_MEMO: dict = {}


def load_v3(ds: str, apo: str = "v2", complete: bool = True) -> pd.DataFrame | None:
    key_ = (ds, apo, complete)
    if key_ not in _MEMO:
        _MEMO[key_] = _load_v3(ds, apo, complete)
    return None if _MEMO[key_] is None else _MEMO[key_].copy()


def _load_v3(ds: str, apo: str = "v2", complete: bool = True) -> pd.DataFrame | None:
    p = CACHE / ("train142__oof.csv" if ds == "train142" else f"{ds}__{apo}.csv")
    if not p.exists():
        return None
    key = "image" if ds == "train142" else "image_id"
    d = pd.read_csv(p).drop_duplicates(key, keep="last").set_index(key)
    d = d.drop(columns=[c for c in d.columns if str(c).startswith("x_")])
    if complete and len(d) < EXPECTED[ds]:
        return None
    return v4_merge_extra(d, ds, apo)    # flfix: adds v4_* columns only (no-op without the extra cache)


def _col(d: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(d[c], errors="coerce").astype(float) if c in d else pd.Series(np.nan, index=d.index)


def nanmean_w(cols: list, w) -> pd.Series:
    """Weighted mean over the finite entries of each row (weights renormalised per row)."""
    X = pd.concat(cols, axis=1).to_numpy(float)
    W = np.asarray(w, float)[None, :] * np.isfinite(X)
    s = W.sum(1)
    v = (np.nan_to_num(X) * W).sum(1) / np.where(s > 0, s, np.nan)
    return pd.Series(v, index=cols[0].index)


BASE = {"deep": ("pa_global", "pa_structure", "pa_fd_strip_g", "pa_fusion"),
        "h": ("pa_global_h", "pa_structure_h", "pa_fd_strip_h", "pa_fusion_h")}


def pa_head(d: pd.DataFrame, head: str, ref: str = "deep") -> pd.Series:
    """PA heads (deg).  fusion = C1 (mean of fragment-global and classical PA);  field = learned field
    in the deep strip vs the global deep line (pa_fd_strip_g);  fusF = mean(fusion, field) (orient
    agent's recommendation);  fus3 = mean(global, structure, field);  "w:a,b,c" = weighted mean of
    (global, structure, field) over the finite heads (inverse-variance fusion)."""
    g, s, f, fu = (_col(d, c) for c in BASE[ref])
    if head == "fusion":
        return fu
    if head == "field":
        return f
    if head == "fusF":
        return nanmean_w([fu, f], [1.0, 1.0])
    if head == "fus3":
        return nanmean_w([g, s, f], [1.0, 1.0, 1.0])
    if head == "global":
        return g
    if head == "structure":
        return s
    if head.startswith("w:"):
        return nanmean_w([g, s, f], [float(x) for x in head[2:].split(",")])
    return pa_head_v4(d, head, ref)      # flfix heads (raises ValueError for unknown names)


def cons_at(d: pd.DataFrame, pa: pd.Series) -> pd.Series:
    """Constructed straight-fascicle FL (C1 recipe) at a per-row PA, interpolated on the PA grid."""
    G = d[GRID_COLS].to_numpy(float)
    p = pa.reindex(d.index).to_numpy(float)
    t = (np.clip(np.nan_to_num(p, nan=3.0), PA_GRID[0], PA_GRID[-1]) - PA_GRID[0]) / 0.5
    i0 = np.clip(np.floor(t).astype(int), 0, len(PA_GRID) - 2)
    fr = t - i0
    r = np.arange(len(d))
    a, b = G[r, i0], G[r, i0 + 1]
    v = np.where(np.isfinite(a) & np.isfinite(b), a * (1 - fr) + b * fr, np.where(np.isfinite(a), a, b))
    v[~np.isfinite(p)] = np.nan
    return pd.Series(v, index=d.index)


def fl_head(d: pd.DataFrame, head: str, pa: pd.Series) -> pd.Series:
    """FL heads (mm).  ens = mean(straight, central3, constructed at the head's own PA);  c1 = C1's
    fl_ensemble (constructed at the C1 fusion PA);  ens_tri = mean(ens, k0 MT / sin PA);  med3."""
    st, c3 = _col(d, "fl_straight"), _col(d, "fl_central3")
    if head == "c1":
        return _col(d, "fl_ensemble")
    c = cons_at(d, pa)
    if head == "cons":
        return c
    ens = nanmean_w([st, c3, c], [1.0, 1.0, 1.0])
    if head == "ens":
        return ens
    if head == "med3":
        return pd.concat([st, c3, c], axis=1).median(axis=1)
    if head == "ens_tri":
        tri = K0 * _col(d, "mt_mm") / np.sin(np.radians(pa.clip(lower=1.0)))
        return nanmean_w([ens, tri], [1.0, 1.0])
    return fl_head_v4(d, head, pa)       # flfix heads (raises ValueError for unknown names)


def est_frame(d: pd.DataFrame, ds: str, rec: dict) -> pd.DataFrame:
    """Per-image estimates (pa_deg / fl_mm / mt_mm) of a recipe + every diagnostic of the cache."""
    pa = rec["pa"] if not isinstance(rec["pa"], dict) else rec["pa"].get(ds, rec["pa"]["test"])
    p_deep = pa_head(d, pa, "deep")
    out = d.copy()
    out["pa_deg"] = pa_head(d, pa, "h") if ds == "dynamic" else p_deep
    out["fl_mm"] = fl_head(d, rec["fl"], p_deep)
    out["mt_mm"] = _col(d, "mt_mm")
    out.loc[out["mt_mm"] < 5.0, ["pa_deg", "fl_mm", "mt_mm"]] = np.nan
    if rec.get("xfeat"):
        out["x_fd_dis"] = (_col(d, "pa_fd_strip_g") - _col(d, "pa_fusion")).abs().clip(0, 20)
        out["x_fd_conc"] = _col(d, "d_fd_strip_conc")
    return out


# ----------------------------------------------------------------------------------- recipes
FIT_KW = dict(features=None, ridge=1.0, balance="sqrt")   # the EB agent's chosen fit (grid_lodo4.csv)
SHRINK = {
    "none": None,
    "conf": None,                                             # C2: w = confidence toward the group median
    "eb": dict(method="ml", likelihood="laplace"),           # EB agent's C4 defaults
    "eb_joint": dict(method="ml", likelihood="laplace", joint=True),
    "groupmed": None,
    # sensitivity variants (not defaults): looser tau anchor, Gaussian posterior, conf**2 weights
    "eb_t05": dict(method="ml", likelihood="laplace", prior_kw=dict(prior_sd_logtau=0.5)),
    "eb_t10": dict(method="ml", likelihood="laplace", prior_kw=dict(prior_sd_logtau=1.0)),
    "eb_joint_t05": dict(method="ml", likelihood="laplace", joint=True, prior_kw=dict(prior_sd_logtau=0.5)),
    "eb_gauss": dict(method="ml", likelihood="gauss"),
    "eb_s15": dict(method="ml", likelihood="laplace", sigma_scale=1.5),
    "conf2": None,
    "conf05": None,
    # hybrid: PA shrunk as C2 (w = confidence, toward the group median), FL by EB (Laplace, ML prior)
    "hyb": dict(method="ml", likelihood="laplace", cols=("fl",)),
    "hyb2": dict(method="ml", likelihood="laplace", cols=("fl",)),   # PA w = confidence**2
    # hybrid with the FL prior centre fixed at the plain group median (no precision weighting)
    "hyb_med": dict(method="ml", likelihood="laplace", cols=("fl",), prior_kw=dict(mu_mode="median")),
    "eb_med": dict(method="ml", likelihood="laplace", prior_kw=dict(mu_mode="median")),
}


def R(apo="v2", pa="fusF", fl="ens", shrink="eb", xfeat=False, name=None) -> dict:
    return dict(apo=apo, pa=pa, fl=fl, shrink=shrink, xfeat=xfeat,
                name=name or f"{apo}|{pa if isinstance(pa, str) else 'iv'}|{fl}|{shrink}{'|x' if xfeat else ''}")


def apply_shrink(D: pd.DataFrame, shrink: str, model: sh.ErrorModel | None, ds: str = "") -> pd.DataFrame:
    import eval_shrinkage as es
    if shrink in SHRINK_V4:              # flfix shrinkers (separate code below)
        return apply_shrink_v4(D, shrink, model, ds)
    if shrink == "none":
        P = D.copy()
        for c in sh.COMP:
            P[f"{c}_eb"] = D[sh.COMP[c]]
    elif shrink in ("conf", "conf2", "conf05"):
        if shrink == "conf2":   # stronger: w = confidence**2
            D = D.assign(confidence=_col(D, "confidence").fillna(0).clip(0, 1) ** 2)
        P = sh.confidence_shrink(D, "group", lam=0.5 if shrink == "conf05" else 1.0)
    elif shrink == "groupmed":
        P = sh.group_median(D, "group")
    elif shrink in ("hyb", "hyb2", "hyb_med"):
        Dc = D.assign(confidence=_col(D, "confidence").fillna(0).clip(0, 1) ** 2) if shrink == "hyb2" else D
        Pc = sh.confidence_shrink(Dc, "group", cols=("pa",))
        P = sh.shrink(D, model, group_col="group", clip_col="clip", **SHRINK[shrink])
        P["pa_eb"] = Pc["pa_eb"]
        P["w_pa"] = _col(Dc, "confidence").fillna(0).clip(0, 1)
    else:
        kw = dict(SHRINK[shrink])
        if ds == "dynamic":   # PA vs horizontal there: the FL-PA-MT coupling (deep apo) does not apply
            kw.pop("joint", None)
        P = sh.shrink(D, model, group_col="group", clip_col="clip", **kw)
    return es.fill_group(P)


def bench_frame(rec: dict, split: str = "lodo4") -> pd.DataFrame:
    """Benchmark rows (estimates + *_true + dataset + group), as eval_shrinkage.load_bench."""
    import eval_shrinkage as es
    parts = []
    st = load_v3("static", rec["apo"])
    if st is not None:
        d = est_frame(st, "static", rec)
        g = bc.static_gts()["mean7"].reindex(d.index)
        d["pa_true"], d["fl_true"], d["mt_true"] = g.pa_deg, g.fl_mm, g.mt_mm
        d["dataset"], d["group"] = "static", es.static_group(d.index)
        parts.append(d)
    vl = load_v3("vl", rec["apo"])
    if vl is not None:
        d = est_frame(vl, "vl", rec)
        v = bc.vl_gt().reindex(d.index)
        d["pa_true"], d["fl_true"], d["mt_true"] = v.pa_deg, v.fl_mm, v.mt_mm
        d["dataset"] = np.where(d.index.str.startswith("young"), "vl_young", "vl_old") if split == "lodo4" else "vl"
        d["group"] = d["dataset"]
        parts.append(d)
    dy = load_v3("dynamic", rec["apo"])
    if dy is not None:
        d = est_frame(dy, "dynamic", rec)
        dg = bc.dynamic_gt().reindex(d.index)
        d["pa_true"], d["fl_true"], d["mt_true"] = dg.pa_deg, dg.fl_mm, np.nan
        d["dataset"], d["group"] = "dynamic", "dynamic"
        d = d[d["pa_true"].notna() | d["fl_true"].notna()]
        parts.append(d)
    B = pd.concat(parts)
    B["image_id"] = B.index.astype(str)
    B.index = B["dataset"].str.replace(r"_(young|old)$", "", regex=True) + ":" + B.index.astype(str)
    B["clip"] = -1
    return B


def iv_weights(apo: str, exclude=(), floor: float = 0.0) -> tuple[float, float, float]:
    """Inverse-(co)variance weights of (global, structure, field) PA from the raw second-moment
    matrix of their errors (bias included), averaged over the benchmark datasets with equal weight."""
    Ms = []
    G = bc.static_gts()["mean7"]
    for ds in ("static", "vl", "dynamic"):
        if ds in exclude:
            continue
        d = load_v3(ds, apo)
        if d is None:
            continue
        ref = "h" if ds == "dynamic" else "deep"
        X = pd.concat([pa_head(d, h, ref) for h in ("global", "structure", "field")], axis=1)
        gt = {"static": G.pa_deg, "vl": bc.vl_gt().pa_deg, "dynamic": bc.dynamic_gt().pa_deg}[ds]
        E = X.sub(gt.reindex(X.index), axis=0).dropna()
        Ms.append(E.T.to_numpy() @ E.to_numpy() / len(E))
    M = np.mean(Ms, axis=0)
    w = np.linalg.solve(M, np.ones(3))
    w = np.clip(w / w.sum(), floor, None)
    return tuple(float(x) for x in w / w.sum())


def iv_head(apo: str) -> dict:
    """Per-dataset honest IV heads: each benchmark gets weights fitted without it; test gets all."""
    mk = lambda w: "w:" + ",".join(f"{x:.4f}" for x in w)  # noqa: E731
    return {"static": mk(iv_weights(apo, ("static",))), "vl": mk(iv_weights(apo, ("vl",))),
            "dynamic": mk(iv_weights(apo, ("dynamic",))), "test": mk(iv_weights(apo))}


def run_bench(rec: dict) -> dict:
    """Predictions per (split, dataset): lodo4 for every dataset, lodo3 for VL (VL held out whole)."""
    out = {}
    eb = rec["shrink"].startswith(("eb", "hyb"))
    for split in ("lodo4", "lodo3"):
        B = bench_frame(rec, split)
        for ds in B.dataset.unique():
            if split == "lodo3" and ds != "vl":
                continue
            D = B[B.dataset == ds]
            model = sh.fit_error_model(B[B.dataset != ds], **FIT_KW) if eb else None
            out[(split, ds)] = apply_shrink(D, rec["shrink"], model, ds)
    return out


def _q(P: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({sh.COMP[c]: P[f"{c}_eb"].to_numpy(float) for c in sh.COMP},
                        index=[i.split(":", 1)[1] for i in P.index])


def score_bench(res: dict) -> dict:
    G, vgt, dgt = bc.static_gts(), bc.vl_gt(), bc.dynamic_gt()
    r = {}
    Q = _q(res[("lodo4", "static")])
    m = bc.s_metric(Q, G["mean7"])
    r.update(s35=m["S"], s35_pa=m["pa_n"], s35_fl=m["fl_n"], s35_mt=m["mt_n"])
    r["c10"] = bc.s_metric(Q.loc[bc.CLEAN], G["mean7"])["S"]
    r["in25"] = bc.s_metric(Q.drop(bc.CLEAN), G["mean7"])["S"]
    r["2r"] = float(np.mean([bc.s_metric(Q, G[f"R7+{x}"])["S"] for x in bc.RATERS[:-1]]))
    r["R7"] = bc.s_metric(Q, G["R7"])["S"]
    Qy, Qo = _q(res[("lodo4", "vl_young")]), _q(res[("lodo4", "vl_old")])
    Qv = pd.concat([Qy, Qo])
    m = bc.s_metric(Qv, vgt, ("pa_deg", "fl_mm"))
    r.update(vl4=m["S"], vl4_pa=m["pa_n"], vl4_fl=m["fl_n"], vl_pa_bias=m["pa_bias"], vl_fl_bias=m["fl_bias"])
    r["vl_young"] = bc.s_metric(Qy, vgt, ("pa_deg", "fl_mm"))["S"]
    r["vl_old"] = bc.s_metric(Qo, vgt, ("pa_deg", "fl_mm"))["S"]
    r["vl_young3"] = bc.s_metric(Qy, vgt)["S"]
    r["vl_mtl"] = bc.s_metric(Qv, vgt, ("mt_mm",))["mt_n"]
    Q3 = _q(res[("lodo3", "vl")])
    m = bc.s_metric(Q3, vgt, ("pa_deg", "fl_mm"))
    r.update(vl3=m["S"], vl3_pa=m["pa_n"], vl3_fl=m["fl_n"])
    Qd = _q(res[("lodo4", "dynamic")])
    r["dyn_pa_h"] = bc.s_metric(Qd, dgt, ("pa_deg",))["pa_n"]
    r["dyn_fl"] = bc.s_metric(Qd, dgt, ("fl_mm",))["fl_n"]
    r["dyn"] = 0.5 * (r["dyn_pa_h"] + r["dyn_fl"])
    # one-number summaries (diagnostic): mean over the four independent benchmark families
    r["mean4"] = float(np.mean([r["2r"], r["vl3"], r["vl_young3"], r["dyn"]]))
    return r


def test_frame(rec: dict) -> pd.DataFrame:
    d = load_v3("test", rec["apo"])
    T = est_frame(d, "test", rec)
    lay = layout()
    T["group"] = lay.loc[T.index, "group"]
    T["clip"] = lay.loc[T.index, "clip"].astype(int)
    return T


def test_sub(rec: dict, return_per_image: bool = False):
    """Full test pipeline: estimates -> shrinkage (EB model fitted on ALL benchmarks, lodo4 labels)
    -> group-median NaN fill -> clip median (28 clips) -> bounds."""
    import eval_shrinkage as es
    T = test_frame(rec)
    model = None
    if rec["shrink"].startswith(("eb", "hyb")):
        model = sh.fit_error_model(bench_frame(rec, "lodo4"), **FIT_KW)
    P = apply_shrink(T, rec["shrink"], model, "test")
    sub = es.to_sub(P)
    return (sub, P, model) if return_per_image else sub


def train142_table() -> pd.DataFrame | None:
    """Both models out of fold on the 142 dual-GT training images vs the GT-mask geometry."""
    x = load_v3("train142", complete=False)
    p = eo.CACHE / "train142.csv"
    if x is None or not p.exists():
        return None
    g = pd.read_csv(p)
    gt = g[g.run == "gt"].drop_duplicates("image").set_index("image")
    x = x[x.index.isin(gt.index)]
    rows = []
    for h in ("fusion", "global", "structure", "field", "fusF", "fus3"):
        pa = pa_head(x, h)
        for gname, gcol in (("gt_global", "pa_global"), ("gt_fd_comp_g", "pa_fd_comp_g")):
            e = (pa - gt[gcol].reindex(x.index)).dropna()
            rows.append(dict(comp="PA_deg", head=h, gt=gname, n=len(e), mae=e.abs().mean(), medae=e.abs().median(),
                             bias=e.mean()))
        for fl in ("ens", "c1"):
            if fl == "c1" and h != "fusion":
                continue
            f = fl_head(x, fl, pa)
            for gname, gcol in (("gt_straight", "fl_straight"), ("gt_ensemble", "fl_ensemble")):
                gg = gt[gcol].reindex(x.index)
                e = ((f - gg) / gg).dropna()
                rows.append(dict(comp="FL_rel", head=f"{fl}@{h}", gt=gname, n=len(e), mae=e.abs().mean(),
                                 medae=e.abs().median(), bias=e.mean()))
    gm = gt.mt_mm.reindex(x.index)
    e = ((x.mt_mm - gm) / gm).dropna()
    rows.append(dict(comp="MT_rel", head="v2 apo (OOF)", gt="gt", n=len(e), mae=e.abs().mean(), medae=e.abs().median(),
                     bias=e.mean()))
    # official normalisation in mm (GT-mask frame = original resolution, px/cm from the training scale table)
    ppc = ea.train_meta().ppc.reindex(x.index)
    mmf = 10.0 / ppc
    for h in ("fusion", "field", "fusF", "fus3"):
        pa = pa_head(x, h)
        fl = fl_head(x, "c1" if h == "fusion" else "ens", pa)
        ok = pa.notna() & fl.notna() & x.mt_mm.notna() & ppc.notna() & gt.pa_global.reindex(x.index).notna()
        epa = (pa - gt.pa_global.reindex(x.index))[ok].abs().mean() / 6.0
        efl = ((fl - gt.fl_ensemble.reindex(x.index)) * mmf)[ok].abs().mean() / 12.0
        emt = ((x.mt_mm - gm) * mmf)[ok].abs().mean() / 3.0
        rows.append(dict(comp="S_official", head=f"{h} + ens", gt="gt_global/gt_ensemble/gt_mt", n=int(ok.sum()),
                         mae=float(np.mean([epa, efl, emt])), pa_n=float(epa), fl_n=float(efl), mt_n=float(emt)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- bench
REFS = {
    "C2": (ROOT / "artifacts/candidates/C2_ours_v2e_fusion_ensemble_shrink1.0.csv", 0.38974),
    "C1": (ROOT / "artifacts/candidates/C1_ours_v2e_fusion_ensemble.csv", 0.45436),
    "v1": (ROOT / "artifacts/sub_v1_geom_clipmed.csv", 0.46718),
    "eason": (bc.SCRATCH / "notebooks-insight/derived/eason_0.37415.csv", 0.37415),
    "vera": (bc.SCRATCH / "notebooks-insight/derived/vera_0.45134.csv", 0.45134),
}


def refs() -> dict:
    return {k: pd.read_csv(p).set_index("image_id") for k, (p, _) in REFS.items() if p.exists()}


def recipes() -> list:
    iv_v2 = iv_head("v2")
    return [
        # LB-scored baselines rebuilt from the v3 caches (sanity: must reproduce C1 / C2)
        R("ours", "fusion", "c1", "none", name="C1 (LB 0.454)"),
        R("ours", "fusion", "c1", "conf", name="C2 (LB 0.390)"),
        R("ours", "fusion", "c1", "groupmed", name="C3 group medians"),
        # one agent at a time on top of C2
        R("v2", "fusion", "c1", "conf", name="C2 + apo_v2"),
        R("ours", "fusF", "ens", "conf", name="C2 + field PA (orient F2)"),
        R("ours", "fusion", "c1", "eb", name="C2 with EB (eb C4)"),
        R("ours", "fusion", "c1", "eb_joint", name="C2 with EB joint (eb C5)"),
        R("ours", "fusion", "c1", "hyb_med", name="C2 with hybrid shrink"),
        # all three agents
        R("v2", "fusF", "ens", "none", name="v2+fusF per image"),
        R("v2", "fusF", "ens", "conf", name="v2+fusF+conf [C6]"),
        R("v2", "fusF", "ens", "conf2", name="v2+fusF+conf^2"),
        R("v2", "fusF", "ens", "eb", name="v2+fusF+EB"),
        R("v2", "fusF", "ens", "eb_med", name="v2+fusF+EB median centre"),
        R("v2", "fusF", "ens", "eb_joint", name="v2+fusF+EB joint"),
        R("v2", "fusF", "ens", "hyb", name="v2+fusF+hybrid (weighted centre)"),
        R("v2", "fusF", "ens", "hyb_med", name="v2+fusF+hybrid [C4]"),
        R("v2", "fusion", "c1", "hyb_med", name="v2+C1 PA+hybrid [C5]"),
        R("v2", "fusF", "ens", "hyb_med", xfeat=True, name="C4 + field features in EB"),
        R("v2", "fusF", "ens", "groupmed", name="v2+fusF group medians"),
        # PA head alternatives (apo_v2, hybrid shrink)
        R("v2", "field", "ens", "hyb_med", name="PA field only"),
        R("v2", "fus3", "ens", "hyb_med", name="PA mean(global,structure,field)"),
        R("v2", iv_v2, "ens", "hyb_med", name="PA inverse-variance (LODO w)"),
        # FL head alternatives
        R("v2", "fusF", "ens_tri", "hyb_med", name="FL mean(ens, triangle)"),
        R("v2", "fusF", "med3", "hyb_med", name="FL median of 3"),
        R("v2", "fusF", "c1", "hyb_med", name="FL constructed at C1 PA"),
        # alternative MT / aponeurosis source
        R("ours", "fusF", "ens", "hyb_med", name="C4 with ours apo (C1 MT)"),
    ]


def v1_bench() -> dict:
    """The v1 recipe (LB 0.467 geometry on the maskbank maps; failures filled by the median)."""
    G, vgt = bc.static_gts(), bc.vl_gt()
    r = {}
    st, vl = bc.load_cache("static", "ours", "v1"), bc.load_cache("vl", "ours", "v1")
    if st is not None:
        p = bc.fill(bc.heads(st, "global", "straight"))
        m = bc.s_metric(p, G["mean7"])
        r.update(s35=m["S"], s35_pa=m["pa_n"], s35_fl=m["fl_n"], s35_mt=m["mt_n"])
        r["c10"] = bc.s_metric(p.loc[bc.CLEAN], G["mean7"])["S"]
        r["in25"] = bc.s_metric(p.drop(bc.CLEAN), G["mean7"])["S"]
        r["2r"] = float(np.mean([bc.s_metric(p, G[f"R7+{x}"])["S"] for x in bc.RATERS[:-1]]))
        r["R7"] = bc.s_metric(p, G["R7"])["S"]
    if vl is not None:
        p = bc.fill(bc.heads(vl, "global", "straight"))
        m = bc.s_metric(p, vgt, ("pa_deg", "fl_mm"))
        r.update(vl3=m["S"], vl3_pa=m["pa_n"], vl3_fl=m["fl_n"], vl4=m["S"], vl4_pa=m["pa_n"], vl4_fl=m["fl_n"])
        y = p.loc[[i for i in p.index if i.startswith("young")]]
        o = p.loc[[i for i in p.index if i.startswith("old")]]
        r["vl_young"] = bc.s_metric(y, vgt, ("pa_deg", "fl_mm"))["S"]
        r["vl_old"] = bc.s_metric(o, vgt, ("pa_deg", "fl_mm"))["S"]
        r["vl_young3"] = bc.s_metric(y, vgt)["S"]
        r["vl_mtl"] = bc.s_metric(p, vgt, ("mt_mm",))["mt_n"]
    return r


def test_diag(sub: pd.DataFrame, R_: dict) -> dict:
    lay = layout()
    grp = lay.loc[sub.index, "group"]
    r = {}
    for k, ref in R_.items():
        dd = bc.official_distance(sub, ref)
        r[f"d_{k}"], r[f"d_{k}_pa"], r[f"d_{k}_fl"], r[f"d_{k}_mt"] = dd["S"], dd["pa_n"], dd["fl_n"], dd["mt_n"]
    g = sub.groupby(grp)[list(bc.C3)].median()
    k = (sub.fl_mm * np.sin(np.radians(sub.pa_deg)) / sub.mt_mm).groupby(grp).median()
    for gg in g.index:
        r[f"pa:{gg}"], r[f"fl:{gg}"], r[f"mt:{gg}"], r[f"k:{gg}"] = (g.loc[gg, "pa_deg"], g.loc[gg, "fl_mm"],
                                                                     g.loc[gg, "mt_mm"], k.loc[gg])
    r["k:ALL"] = float((sub.fl_mm * np.sin(np.radians(sub.pa_deg)) / sub.mt_mm).median())
    return r


SHOW = ["name", "s35", "s35_pa", "s35_fl", "s35_mt", "c10", "in25", "2r", "R7", "vl3", "vl3_pa", "vl3_fl", "vl4",
        "vl_young", "vl_old", "vl_young3", "vl_mtl", "dyn_pa_h", "dyn_fl", "mean4", "d_C2", "d_C1", "d_v1"]


def md(df: pd.DataFrame, cols: list, nd: int = 3) -> str:
    cols = [c for c in cols if c in df]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def cmd_bench(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 200)
    R_ = refs()
    rows = []
    r = dict(name="v1 recipe (LB 0.467)", **v1_bench())
    r.update({k: v for k, v in test_diag(R_["v1"], R_).items()})
    rows.append(r)
    for rec in recipes():
        t0 = time.time()
        res = run_bench(rec)
        r = dict(name=rec["name"], apo=rec["apo"], pa=rec["pa"] if isinstance(rec["pa"], str) else json.dumps(rec["pa"]),
                 fl=rec["fl"], shrink=rec["shrink"], xfeat=rec["xfeat"], **score_bench(res))
        sub = test_sub(rec)
        r.update(test_diag(sub, R_))
        rows.append(r)
        print(f"{rec['name']:45s} s35 {r['s35']:.3f} c10 {r['c10']:.3f} 2r {r['2r']:.3f} vl3 {r['vl3']:.3f} "
              f"y3 {r['vl_young3']:.3f} dyn {r['dyn_pa_h']:.3f}/{r['dyn_fl']:.3f} | d_C2 {r['d_C2']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "bench_v3.csv", index=False)
    (OUT / "bench_v3.md").write_text(md(T, SHOW) + "\n")
    tr = train142_table()
    if tr is not None:
        tr.to_csv(OUT / "train142_v3.csv", index=False)
        print(tr.round(3).to_string(index=False))
    W = {"v2": {k: iv_weights("v2", ex) for k, ex in (("all", ()), ("no_static", ("static",)), ("no_vl", ("vl",)),
                                                         ("no_dynamic", ("dynamic",)))},
         "ours": {"all": iv_weights("ours")}}
    (OUT / "iv_weights.json").write_text(json.dumps(W, indent=1))
    print(json.dumps(W, indent=1))
    print(T[[c for c in SHOW if c in T]].round(3).to_string(index=False))


# ----------------------------------------------------------------------------------- build
CANDS = {
    # chosen on the benchmarks (bench_v3.csv; README): never on the LB
    "C4": dict(apo="v2", pa="fusF", fl="ens", shrink="hyb_med",
               desc="best full pipeline: apo_v2 + fusF PA + ens FL; PA conf-shrunk, FL EB (median centre)"),
    "C5": dict(apo="v2", pa="fusion", fl="c1", shrink="hyb_med",
               desc="ablation of C4: C1 fusion PA (no learned field) -> tests the field-PA transfer to test"),
    "C6": dict(apo="v2", pa="fusF", fl="ens", shrink="conf",
               desc="conservative: C4 estimates with C2's LB-validated confidence shrinkage for PA and FL"),
}


def validate(path: Path) -> str:
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "scripts/validate_submission.py"), str(path), "--test-dir",
                        str(bc.TEST_DIR)], capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src")})
    return (r.stdout + r.stderr).strip()


def cmd_build(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = json.loads(Path(args.spec).read_text()) if args.spec else CANDS
    R_ = refs()
    lay = layout()
    report, subs = {}, {}
    for name, kw in cands.items():
        rec = R(**{k: v for k, v in kw.items() if k != "desc"})
        sub, P, model = test_sub(rec, return_per_image=True)
        pa_tag = rec["pa"] if isinstance(rec["pa"], str) else "iv"
        fname = f"{name}_{rec['apo']}_{pa_tag}_{rec['fl']}_{rec['shrink']}{'_x' if rec['xfeat'] else ''}.csv"
        path = OUT / fname
        bc.write_sub(sub, path)
        subs[name] = sub
        keep = ["group", "clip", "eb_unit", "confidence", "flags", "d_apo_source", "pa_deg", "fl_mm", "mt_mm",
                "pa_fusion", "pa_fd_strip_g", "pa_global", "pa_structure", "fl_straight", "fl_central3",
                "fl_constructed", "fl_ensemble", "sigma_pa", "sigma_fl", "mu_pa", "tau_pa", "w_pa", "pa_eb", "mu_fl",
                "tau_fl", "w_fl", "fl_eb", "mt_eb"]
        Q = P[[c for c in keep if c in P]].copy()
        Q["sub_pa_deg"], Q["sub_fl_mm"], Q["sub_mt_mm"] = sub.pa_deg, sub.fl_mm, sub.mt_mm
        Q.index.name = "image_id"
        Q.to_csv(OUT / f"per_image_{name}.csv")
        grp = lay.loc[sub.index, "group"]
        g = sub.groupby(grp)[list(bc.C3)].median().round(2)
        g["k"] = (sub.fl_mm * np.sin(np.radians(sub.pa_deg)) / sub.mt_mm).groupby(grp).median().round(3)
        g["n"] = grp.value_counts()
        if "w_pa" in P:
            g["w_pa"] = P.groupby("group")["w_pa"].mean().round(3)
            g["w_fl"] = P.groupby("group")["w_fl"].mean().round(3)
        elif rec["shrink"] == "conf":
            g["w_pa"] = g["w_fl"] = P.groupby("group")["confidence"].mean().fillna(0).round(3)
        report[name] = dict(file=fname, recipe={k: v for k, v in rec.items()}, desc=kw.get("desc", ""),
                            validate=validate(path), n_mt_below5=int((P.mt_mm.isna()).sum()),
                            group=g.reset_index().to_dict("records"),
                            dist={k: bc.official_distance(sub, v) for k, v in R_.items()},
                            priors=P.attrs.get("priors", {}))
        print(f"\n== {name}: {fname}\n{report[name]['validate']}")
        print(g.to_string())
        for k, v in report[name]["dist"].items():
            print(f"   vs {k:6s} {v}")
    names = list(subs)
    report["pairwise"] = {f"{a}-{b}": bc.official_distance(subs[a], subs[b]) for i, a in enumerate(names)
                          for b in names[i + 1:]}
    print(json.dumps(report["pairwise"], indent=1))
    (OUT / "candidates_v3_report.json").write_text(json.dumps(report, indent=1, default=str))


# =================================================================================================== flfix (v4)
# Critic fixes 2, 3 and 6 (+ fix 5: leak-free held-out constants), kept apart from the v3 code above.
# Existing recipes are untouched: the new PA heads / FL heads / shrink names are only reached through
# one-line fall-through hooks in _load_v3, pa_head, fl_head and apply_shrink, and the extra cache only
# adds v4_* columns (bench_v3 numbers and the C4/C5/C6 files reproduce byte-identically).
#   measure_extra --dataset D   per-image deep-strip fragment PA and insertion-strip field PA vs the
#                               global deep line / the horizontal -> artifacts/eval_v4/flfix/cache/
#   flfix                       benchmark grid (LODO constants), bootstrap CIs vs C4, per-image tables,
#                               test diagnostics, candidate CSVs -> artifacts/eval_v4/flfix/
# New recipe options (strings in the usual R(apo, pa, fl, shrink) fields):
#   pa     "ins"    learned-field PA in the narrow insertion strip (2-12 % of MT above the deep inner
#                   edge, vs the local deep tangent; pa_fd_ins)    "insg" the same strip vs the global line
#          "fusI"   mean(C1 fusion, ins)
#          "frd"    fragment-only deep-strip PA: median PA (vs the global deep line) of the fragments whose
#                   centroid lies 4-30 % of MT above the deep inner edge (>= 2 such fragments, else the
#                   whole-muscle fragment median pa_global)
#          "fusFD"  mean(frd, field strip PA pa_fd_strip_g)
#   fl     "ens_pp" ens with the constructed fascicle rebuilt at the POST-shrinkage PA (fix 2): the PA of
#                   the confidence rule, i.e. exactly the PA that is output (deep-aponeurosis reference)
#   shrink "hyb_jfl" | "hyb_cpl" | "hyb_cplg" | "hyb_floor" (EB FL, sh.shrink_fl_coupled) and
#          "conf_floor" (C2 confidence FL with the coupling floor); PA always from the confidence rule (fix 3)
# ===================================================================================================
import contextlib  # noqa: E402

V4_OUT = ROOT / "artifacts/eval_v4/flfix"
V4_CACHE = Path(__import__("os").environ.get("V4_CACHE", V4_OUT / "cache"))  # override for other map sources
V4_STRIP = FCFG.strip            # (0.04, 0.30): the field deep strip, reused for the fragment deep strip
V4_MIN_FRAG = 2
V4_LONG_CLIPS = (0, 9, 26)       # Telemed clips whose fascicles cross the whole view (critic)


def v4_extra_row(res: g2.GeoResult, orient) -> dict:
    """Deep-strip fragment PA and insertion-strip field PA from a measure(..., keep_details=True) result."""
    cv = res.curves or {}
    out = {"v4_chk_pa_global": res.variants.get("pa_global", np.nan)}
    if "sup_curve" not in cv:
        return out
    deep, s_lo, s_hi, mt_px = cv["deep_curve"], cv["s_lo"], cv["s_hi"], cv["mt_px"]
    frs = res.fragments or []
    hs = [(float(deep(q["cx"])) - q["cy"]) / mt_px for q in frs]
    sel = [q for q, h in zip(frs, hs) if V4_STRIP[0] <= h <= V4_STRIP[1]]
    tot = float(sum(q["length"] for q in frs))
    out["v4_frd_n"] = len(sel)
    out["v4_frd_lenfrac"] = float(sum(q["length"] for q in sel)) / tot if tot > 0 else np.nan
    ok = len(sel) >= V4_MIN_FRAG
    out["v4_pa_frd"] = float(np.median([q["pa"] for q in sel])) if ok else np.nan
    out["v4_pa_frd_loc"] = g2._nanmedian([q["pa_local"] for q in sel]) if ok else np.nan
    out["v4_pa_frd_h"] = float(np.median([abs(float(g2.wrap90(q["angle"]))) for q in sel])) if ok else np.nan
    out["v4_frd_hmed"] = float(np.median(hs)) if hs else np.nan       # where the fragments sit (fraction of MT)
    seg, c2, s2 = g2.orient_maps(orient)
    theta, wmap = g2.field_theta_weight(seg, c2, s2, FCFG.seg_pow)
    lo = s_lo + FCFG.xtrim * (s_hi - s_lo)
    hi = s_hi - FCFG.xtrim * (s_hi - s_lo)
    deep_s, _ = g2._line_fit(deep.xd, deep.yd)
    g_ang = float(np.degrees(np.arctan(deep_s)))
    refs = {"loc": lambda x: np.degrees(np.arctan(deep.slope(x))), "g": lambda x: g_ang, "h": lambda x: 0.0}
    y_from = lambda x: deep(x) - FCFG.strip_ins[1] * mt_px  # noqa: E731
    y_to = lambda x: deep(x) - FCFG.strip_ins[0] * mt_px  # noqa: E731
    for k, ref in refs.items():
        rel, ww = g2._field_region(theta, wmap, y_from, y_to, lo, hi, FCFG.sub, ref)
        st = g2.robust_axial_mean(rel, ww, FCFG.robust_c)
        okp = int((ww > 0.05).sum()) if ww.size else 0
        out[f"v4_pa_fd_ins_{k}"] = abs(st["mean"]) if (np.isfinite(st["mean"]) and okp >= FCFG.min_px) else np.nan
    return out


def cmd_measure_extra(args) -> None:
    """One light pass (no structure PA, no streamlines) that adds the v4_* PA columns; resumable."""
    V4_CACHE.mkdir(parents=True, exist_ok=True)
    path = V4_CACHE / f"{args.dataset}__{args.apo}.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(prev.image_id) if len(prev) else set()
    banks = ea.Banks()
    obank = g2.MaskBank(eo.OBANK_ROOT)
    kmap = eo.orient_keymap(args.dataset)
    t0, rows, k = time.time(), [], 0

    def flush():
        pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)

    for iid, img, ppc, _ in bc.iterate(args.dataset):
        if iid in done:
            continue
        if args.limit and k >= args.limit:
            break
        okey = kmap.get(iid, Path(iid).stem) if args.dataset == "vl" else Path(iid).stem
        A, up, lo, fbs, thr, edges = apo_inputs(banks, args.dataset, iid, args.apo)
        F = banks.main.prob(args.dataset, "ours_fasc384_noflip", iid)
        orient = eo.load_orient(obank, args.dataset, okey)
        r: dict = {}
        if A is not None and F is not None and orient is not None and orient[1].shape == A.shape:
            try:
                cfg = ea.make_cfg("v2e", thr, edges).with_(fasc_thr=0.95, compute_structure=False)
                im = img
                if im is not None and im.shape != A.shape:
                    im = cv2.resize(im, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
                res = g2.measure_apo(A, F, im, ppc, cfg, apo_upper=up, apo_lower=lo, fallbacks=fbs,
                                     keep_details=True, orient=orient)
                r = v4_extra_row(res, orient)
            except Exception as e:  # keep going; record the failure
                r = {"v4_error": f"{type(e).__name__}:{e}"}
        rows.append(dict(image_id=iid, **r))
        del A, up, lo, fbs, F, orient
        k += 1
        if k % 25 == 0:
            flush()
            print(f"  extra {args.dataset}/{args.apo}: {k} new, {time.time() - t0:.0f}s", flush=True)
    flush()
    banks.close()
    obank.close()
    print(f"done extra {path.name}: {k} new rows in {time.time() - t0:.0f}s", flush=True)


def v4_merge_extra(d: pd.DataFrame, ds: str, apo: str) -> pd.DataFrame:
    """Join the v4_* columns of the extra cache (if measured); never touches existing columns."""
    p = V4_CACHE / f"{ds}__{apo}.csv"
    if ds == "train142" or not p.exists():
        return d
    x = pd.read_csv(p).drop_duplicates("image_id", keep="last").set_index("image_id")
    cols = [c for c in x.columns if str(c).startswith("v4_") and c not in d.columns]
    return d.join(x[cols]) if cols else d


V4_PA = {"deep": dict(ins="pa_fd_ins", insg="v4_pa_fd_ins_g", frd="v4_pa_frd", frd_fb="pa_global"),
         "h": dict(ins="v4_pa_fd_ins_h", insg="v4_pa_fd_ins_h", frd="v4_pa_frd_h", frd_fb="pa_global_h")}


def pa_head_v4(d: pd.DataFrame, head: str, ref: str = "deep") -> pd.Series:
    """New PA heads (see the section header); reached from pa_head for names it does not know."""
    if head not in ("ins", "insg", "fusI", "frd", "fusFD"):
        raise ValueError(head)
    m = V4_PA[ref]
    need = [m["ins"]] if head in ("ins", "fusI") else [m["insg"]] if head == "insg" else [m["frd"]]
    miss = [c for c in need if c not in d]
    if miss:
        raise RuntimeError(f"PA head {head!r} needs {miss}: run `measure_extra` first")
    _, _, f, fu = (_col(d, c) for c in BASE[ref])
    ins = _col(d, m["ins"])
    frd = _col(d, m["frd"]).fillna(_col(d, m["frd_fb"]))
    if head == "ins":
        return ins
    if head == "insg":
        return _col(d, m["insg"])
    if head == "fusI":
        return nanmean_w([fu, ins], [1.0, 1.0])
    if head == "frd":
        return frd
    return nanmean_w([frd, f], [1.0, 1.0])     # fusFD


def fl_head_v4(d: pd.DataFrame, head: str, pa: pd.Series) -> pd.Series:
    """New FL heads; "ens_pp" is a placeholder at the raw PA here and is rebuilt by v4_post_group."""
    if head == "ens_pp":
        return fl_head(d, "ens", pa)
    raise ValueError(head)


def v4_pa_conf(p: pd.Series, conf: pd.Series, grp: pd.Series) -> pd.Series:
    """The C2 confidence rule for PA (identical to sh.confidence_shrink on pa_deg)."""
    w = conf.fillna(0.0).clip(0, 1)
    med = p.groupby(grp).transform("median")
    return (w * p + (1 - w) * med).where(p.notna(), med)


def v4_post_group(B: pd.DataFrame, rec: dict) -> pd.DataFrame:
    """After the groups are known: pa_deep_raw (recipe PA, deep reference), pa_deep_c (its confidence-
    shrunk value = the PA that is output, deep reference even on dynamic) and, for fl="ens_pp", FL rebuilt
    with the constructed fascicle at pa_deep_c."""
    B = B.copy()
    if isinstance(rec["pa"], dict):
        raise ValueError("v4 recipes take a single PA head")
    p = pa_head(B, rec["pa"], "deep")
    p[_col(B, "mt_mm").isna()] = np.nan                    # est_frame's MT < 5 mm failure rule
    B["pa_deep_raw"] = p
    B["pa_deep_c"] = v4_pa_conf(p, _col(B, "confidence"), B["group"].astype(str))
    if rec["fl"] == "ens_pp":
        fl = nanmean_w([_col(B, "fl_straight"), _col(B, "fl_central3"), cons_at(B, B["pa_deep_c"])],
                       [1.0, 1.0, 1.0])
        fl[_col(B, "mt_mm").isna()] = np.nan
        B["fl_mm"] = fl
    return B


def bench_frame_v4(rec: dict, split: str = "lodo4") -> pd.DataFrame:
    return v4_post_group(bench_frame(rec, split), rec)


def test_frame_v4(rec: dict) -> pd.DataFrame:
    return v4_post_group(test_frame(rec), rec)


SHRINK_V4 = {
    "hyb_jfl": dict(mode="joint", k_mode="k0"),     # joint (PA, FL, MT) EB posterior, FL marginal only
    "hyb_cpl": dict(mode="cpl", k_mode="k0"),       # FL posterior x coupling at the output PA (expert K0)
    "hyb_cplg": dict(mode="cpl", k_mode="group"),   # same, k = the group's own median (no level shift)
    "hyb_floor": dict(mode="floor", k_mode="k0"),   # EB weight floored when raw FL agrees with the coupling
    "conf_floor": None,                             # C2 confidence weight, same floor
}


def apply_shrink_v4(D: pd.DataFrame, shrink: str, model: sh.ErrorModel | None, ds: str = "") -> pd.DataFrame:
    """Fix-3 FL shrinkers. PA is always the confidence rule (as C4); the coupling is evaluated at the output
    PA with the deep-aponeurosis reference (pa_deep_c; on dynamic the scored PA is vs the horizontal)."""
    import eval_shrinkage as es
    conf = _col(D, "confidence").fillna(0).clip(0, 1)
    pa_raw = D["pa_deep_raw"] if "pa_deep_raw" in D else _col(D, "pa_deg")
    pa_c = D["pa_deep_c"] if "pa_deep_c" in D else v4_pa_conf(pa_raw, conf, D["group"].astype(str))
    if shrink == "conf_floor":
        P = sh.confidence_shrink(D, "group")
        x = _col(D, "fl_mm")
        med = x.groupby(D["group"]).transform("median")
        tri = sh.K0 * _col(D, "mt_mm") / np.sin(np.radians(pa_c.clip(lower=1.0)))
        a = np.exp(-0.5 * (np.log(x / tri) / sh.S_LOGK) ** 2).fillna(0.0)
        w = np.maximum(conf, a)
        P["fl_eb"] = (w * x + (1 - w) * med).where(x.notna(), med)
        P["w_fl"], P["a_floor"], P["fl_tri"] = w, a, tri
    else:
        Pc = sh.confidence_shrink(D, "group", cols=("pa",))
        P = sh.shrink_fl_coupled(D, model, pa_raw, pa_c, group_col="group", clip_col="clip", likelihood="laplace",
                                 prior_kw=dict(mu_mode="median"), **SHRINK_V4[shrink])
        P["pa_eb"] = Pc["pa_eb"]
        P["w_pa"] = conf
    return es.fill_group(P)


def v4_label_tables() -> dict:
    G = bc.static_gts()["mean7"]
    return {"static": G[list(bc.C3)], "vl": bc.vl_gt()[list(bc.C3)], "dynamic": bc.dynamic_gt()[["pa_deg", "fl_mm"]]}


V4_HOLD = {"static": "static", "vl": "vl", "vl_young": "vl", "vl_old": "vl", "dynamic": "dynamic"}
_V4C: dict = {}


def v4_constants(held: str | None) -> dict:
    """POP_SD / K0 / S_LOGK from the expert labels of every benchmark except ``held`` (VL as a whole)."""
    if held not in _V4C:
        _V4C[held] = sh.label_constants({k: v for k, v in v4_label_tables().items() if k != held})
    return _V4C[held]


def run_bench_v4(rec: dict, consts: str = "lodo") -> dict:
    """run_bench with the v4 frames; consts="lodo": the label-derived constants (POP_SD, K0, S_LOGK, and so
    tri_rel) are recomputed without the held-out dataset (VL held out as a whole even in lodo4);
    consts="orig": the module constants (they contain VL labels; = bench_v3)."""
    out = {}
    eb = rec["shrink"].startswith(("eb", "hyb"))
    for split in ("lodo4", "lodo3"):
        B = bench_frame_v4(rec, split)
        for ds in B.dataset.unique():
            if split == "lodo3" and ds != "vl":
                continue
            D = B[B.dataset == ds]
            if consts == "lodo":
                c = v4_constants(V4_HOLD[ds])
                ctx = sh.constants(c["pop_sd"], c["k0"], c["s_logk"])
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                model = sh.fit_error_model(B[B.dataset != ds], **FIT_KW) if eb else None
                out[(split, ds)] = apply_shrink(D, rec["shrink"], model, ds)
    return out


def test_sub_v4(rec: dict, return_per_image: bool = False):
    """test_sub with the v4 frames (error model on every benchmark, module constants, as C4)."""
    import eval_shrinkage as es
    T = test_frame_v4(rec)
    model = sh.fit_error_model(bench_frame_v4(rec, "lodo4"), **FIT_KW) if rec["shrink"].startswith(("eb", "hyb")) \
        else None
    P = apply_shrink(T, rec["shrink"], model, "test")
    sub = es.to_sub(P)
    return (sub, P, model) if return_per_image else sub


TOL3 = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def _pis(pred: pd.DataFrame, gt: pd.DataFrame, comps) -> pd.DataFrame:
    """Per-image normalised absolute errors per component and their mean (S_i)."""
    ids = [i for i in pred.index if i in gt.index]
    E = pd.DataFrame({c: (pred.loc[ids, c].astype(float) - gt.loc[ids, c].astype(float)).abs() / TOL3[c]
                      for c in comps}, index=ids)
    E["S"] = E[list(comps)].mean(axis=1)
    return E


def v4_per_image(res: dict) -> dict:
    """Per-image predictions and normalised errors: static (mean7, 2-rater mean, R7), VL lodo3 / lodo4
    (PA + FL), dynamic (PA vs horizontal, FL)."""
    G, vgt, dgt = bc.static_gts(), bc.vl_gt(), bc.dynamic_gt()
    out = {}
    st = _q(res[("lodo4", "static")])
    m7 = _pis(st, G["mean7"], bc.C3)
    r2 = [_pis(st, G[f"R7+{x}"], bc.C3) for x in bc.RATERS[:-1]]
    two = sum(r2) / len(r2)
    T = st.add_prefix("pred_")
    for c in bc.C3:
        T[f"true7_{c}"] = G["mean7"].reindex(T.index)[c]
    T = T.join(m7.add_prefix("e7_")).join(two.add_prefix("e2r_")).join(_pis(st, G["R7"], bc.C3).add_prefix("eR7_"))
    T["clean"] = T.index.isin(bc.CLEAN)
    out["static"] = T
    for key, part in (("vl3", [("lodo3", "vl")]), ("vl4", [("lodo4", "vl_young"), ("lodo4", "vl_old")])):
        Q = pd.concat([_q(res[k]) for k in part])
        E = _pis(Q, vgt, ("pa_deg", "fl_mm"))
        V = Q.add_prefix("pred_").join(vgt[["pa_deg", "fl_mm"]].add_prefix("true_")).join(E.add_prefix("e_"))
        V["subject"] = V.index.str.split("__").str[1]
        out[key] = V
    Qd = _q(res[("lodo4", "dynamic")])
    E = _pis(Qd, dgt, ("pa_deg", "fl_mm"))
    Dd = Qd.add_prefix("pred_").join(dgt[["pa_deg", "fl_mm"]].add_prefix("true_")).join(E.add_prefix("e_"))
    Dd["frame"] = Dd.index.str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
    out["dyn"] = Dd.sort_values("frame")
    return out


def v4_score(res: dict) -> dict:
    r = score_bench(res)
    pi = v4_per_image(res)
    st = pi["static"]
    r["2r_c10"] = float(st.loc[st.clean, "e2r_S"].mean())
    r["2r_in25"] = float(st.loc[~st.clean, "e2r_S"].mean())
    for c in ("pa_deg", "fl_mm"):
        r[f"2r_{c[:2]}"] = float(st[f"e2r_{c}"].mean())
    return r


def v4_boot(a: pd.Series, b: pd.Series, cluster: pd.Series | None = None, block: int = 0, n_boot: int = 4000,
            seed: int = 0) -> dict:
    """Paired bootstrap of mean(a - b): by image, by cluster (image-weighted: VL subjects) or moving blocks
    of ``block`` consecutive rows (dynamic frames, already in time order)."""
    d = (a - b).dropna()
    rng = np.random.default_rng(seed)
    v = d.to_numpy(float)
    n = len(v)
    if n == 0:
        return dict(diff=np.nan, lo=np.nan, hi=np.nan, p_neg=np.nan, n=0)
    if cluster is not None:
        cl = cluster.reindex(d.index).to_numpy()
        u, inv = np.unique(cl, return_inverse=True)
        s = np.bincount(inv, weights=v)
        c = np.bincount(inv).astype(float)
        idx = rng.integers(0, len(u), (n_boot, len(u)))
        bs = s[idx].sum(1) / c[idx].sum(1)
    elif block:
        nb = int(math.ceil(n / block))
        starts = rng.integers(0, n - block + 1, (n_boot, nb))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
        bs = v[idx].mean(1)
    else:
        bs = v[rng.integers(0, n, (n_boot, n))].mean(1)
    return dict(diff=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                p_neg=float(np.mean(bs < 0)), n=n)


def v4_boot_table(pi_a: dict, pi_b: dict, name_a: str, name_b: str) -> list:
    """Bootstrap CIs of S(a) - S(b) on every benchmark slice (negative = a better)."""
    rows = []
    sa, sb = pi_a["static"], pi_b["static"]
    cl = sa.clean
    for nm, col, mask in (("static35 mean7", "e7_S", None), ("clean10 mean7", "e7_S", cl), ("in25 mean7", "e7_S", ~cl),
                          ("2r all35", "e2r_S", None), ("2r clean10", "e2r_S", cl), ("2r in25", "e2r_S", ~cl),
                          ("2r all35 PA", "e2r_pa_deg", None), ("2r all35 FL", "e2r_fl_mm", None),
                          ("R7 all35", "eR7_S", None)):
        x, y = sa[col], sb[col]
        if mask is not None:
            x, y = x[mask], y[mask]
        rows.append(dict(a=name_a, b=name_b, bench=nm, **v4_boot(x, y)))
    for key, lab in (("vl3", "VL-244 lodo3"), ("vl4", "VL-244 lodo4")):
        va, vb = pi_a[key], pi_b[key]
        for col, sfx in (("e_S", ""), ("e_pa_deg", " PA"), ("e_fl_mm", " FL")):
            rows.append(dict(a=name_a, b=name_b, bench=lab + sfx, **v4_boot(va[col], vb[col], cluster=va["subject"])))
    da, db = pi_a["dyn"], pi_b["dyn"]
    for col, sfx in (("e_S", " (PA_h+FL)"), ("e_pa_deg", " PA_h"), ("e_fl_mm", " FL")):
        rows.append(dict(a=name_a, b=name_b, bench="dynamic" + sfx, **v4_boot(da[col], db[col], block=17)))
    return rows


def v4_test_diag(sub: pd.DataFrame, P: pd.DataFrame, refsubs: dict) -> dict:
    lay = layout()
    grp = lay.loc[sub.index, "group"]
    clip = lay.loc[sub.index, "clip"]
    r = {f"d_{k}": bc.official_distance(sub, v)["S"] for k, v in refsubs.items()}
    k = sub.fl_mm * np.sin(np.radians(sub.pa_deg)) / sub.mt_mm
    for g, gg in sub.groupby(grp):
        r[f"pa:{g}"], r[f"fl:{g}"], r[f"k:{g}"] = gg.pa_deg.median(), gg.fl_mm.median(), k[gg.index].median()
    tri = K0 * sub.mt_mm / np.sin(np.radians(sub.pa_deg))
    for c in V4_LONG_CLIPS:
        ids = clip.index[clip == c]
        r[f"clip{c}_fl"] = float(sub.loc[ids, "fl_mm"].median())
        r[f"clip{c}_pa"] = float(sub.loc[ids, "pa_deg"].median())
        r[f"clip{c}_k"] = float(k[ids].median())
        r[f"clip{c}_tri"] = float(tri[ids].median())
    tel = grp.str.startswith("telemed")
    r["telemed_k_med"] = float(k[tel].median())
    r["telemed_unit_k_lt_0.9"] = int((k[tel].groupby(clip[tel]).median() < 0.9).sum())
    return r


def v4_recipes() -> list:
    """The validation grid: PA head x FL head x shrinker (fix 6 x fix 2 x fix 3), apo_v2 throughout."""
    recs = [R("ours", "fusion", "c1", "conf", name="C2 (LB 0.390)")]
    for pa in ("fusF", "ins", "insg", "fusI", "frd", "fusFD"):
        for fl in ("ens", "ens_pp"):
            for shr in ("hyb_med", "conf", "hyb_jfl", "hyb_cpl", "hyb_cplg", "hyb_floor", "conf_floor"):
                recs.append(R("v2", pa, fl, shr))
    return recs


V4_KEY = {   # name in the tables -> recipe name: the planned contrasts (each vs C4)
    "C4": "v2|fusF|ens|hyb_med", "C6": "v2|fusF|ens|conf", "C2": "C2 (LB 0.390)",
    "fix2 C4pp": "v2|fusF|ens_pp|hyb_med", "fix2 C6pp": "v2|fusF|ens_pp|conf",
    "fix3 jfl": "v2|fusF|ens_pp|hyb_jfl", "fix3 cpl": "v2|fusF|ens_pp|hyb_cpl", "fix3 cplg": "v2|fusF|ens_pp|hyb_cplg",
    "fix3 floor": "v2|fusF|ens_pp|hyb_floor", "fix3 conf_floor": "v2|fusF|ens_pp|conf_floor",
    "fix3 jfl (no pp)": "v2|fusF|ens|hyb_jfl", "fix3 cpl (no pp)": "v2|fusF|ens|hyb_cpl",
    "fix3 cplg (no pp)": "v2|fusF|ens|hyb_cplg", "fix3 floor (no pp)": "v2|fusF|ens|hyb_floor",
    "fix6 ins": "v2|ins|ens_pp|hyb_med", "fix6 insg": "v2|insg|ens_pp|hyb_med", "fix6 fusI": "v2|fusI|ens_pp|hyb_med",
    "fix6 frd": "v2|frd|ens_pp|hyb_med", "fix6 fusFD": "v2|fusFD|ens_pp|hyb_med",
}
V4_CANDS = {   # chosen on the benchmarks (bench_flfix.csv, boot_vs_C4.csv), never on the LB
    "flfix_C4pp": dict(apo="v2", pa="fusF", fl="ens_pp", shrink="hyb_med",
                       desc="ADOPT: C4 + fix 2 (constructed-fascicle FL rebuilt at the post-shrinkage PA)"),
    "flfix_C6pp": dict(apo="v2", pa="fusF", fl="ens_pp", shrink="conf",
                       desc="ADOPT: C6 + fix 2 (conservative partner: C2 confidence shrinkage for FL)"),
    "flfix_C4pp_cpl": dict(apo="v2", pa="fusF", fl="ens_pp", shrink="hyb_cpl",
                           desc="PROBE ONLY (fix 3, benchmark-neutral): C4pp with the FL-only coupling posterior"),
}
V4_SHOW = ["name", "s35", "c10", "in25", "2r", "2r_c10", "R7", "vl3", "vl3_pa", "vl3_fl", "vl4", "vl_young3",
           "dyn_pa_h", "dyn_fl", "dyn", "mean4", "d_C4", "d_C2", "d_C6", "telemed_k_med", "clip0_fl", "clip9_fl",
           "clip26_fl"]


def cmd_flfix(args) -> None:
    V4_OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 300)
    C = {h: v4_constants(h) for h in ("vl", "static", "dynamic")}
    C["orig (module)"] = dict(pop_sd={k: sh.POP_SD[k] for k in sh.COMP}, k0=sh.K0, s_logk=sh.S_LOGK)
    C["all three (rule)"] = sh.label_constants(v4_label_tables())
    (V4_OUT / "constants.json").write_text(json.dumps(C, indent=1))
    c4rec = R("v2", "fusF", "ens", "hyb_med")
    refsubs = {"C4": test_sub(c4rec), "C6": test_sub(R("v2", "fusF", "ens", "conf")),
               "C2": pd.read_csv(REFS["C2"][0]).set_index("image_id")}
    rows, PI, SUBS, PER = [], {}, {}, {}
    keyset = set(V4_KEY.values())
    recs = v4_recipes()
    if args.quick:
        recs = [r for r in recs if r["name"] in keyset]
    for consts in ("orig", "lodo"):
        for rec in recs:
            if consts == "orig" and rec["name"] not in (V4_KEY["C4"], V4_KEY["C6"], V4_KEY["C2"], V4_KEY["fix2 C4pp"]):
                continue
            t0 = time.time()
            res = run_bench_v4(rec, consts)
            r = dict(name=rec["name"], consts=consts, pa=rec["pa"], fl=rec["fl"], shrink=rec["shrink"], **v4_score(res))
            if consts == "lodo":
                sub, P, _ = test_sub_v4(rec, return_per_image=True)
                r.update(v4_test_diag(sub, P, refsubs))
                if rec["name"] in keyset:
                    PI[rec["name"]] = v4_per_image(res)
                    SUBS[rec["name"]] = sub
                    PER[rec["name"]] = P
            rows.append(r)
            print(f"[{consts}] {rec['name']:28s} s35 {r['s35']:.3f} c10 {r['c10']:.3f} 2r {r['2r']:.3f} "
                  f"2rc {r['2r_c10']:.3f} vl3 {r['vl3']:.3f} vl4 {r['vl4']:.3f} dyn {r['dyn_pa_h']:.3f}/{r['dyn_fl']:.3f}"
                  + (f" | dC4 {r['d_C4']:.3f} k {r['telemed_k_med']:.3f} clips {r['clip0_fl']:.0f}/{r['clip9_fl']:.0f}/"
                     f"{r['clip26_fl']:.0f}" if consts == "lodo" else "") + f" ({time.time() - t0:.1f}s)", flush=True)
    T = pd.DataFrame(rows)
    inv = {v: k for k, v in V4_KEY.items()}
    T.insert(1, "key", T.name.map(inv).fillna(""))
    T.to_csv(V4_OUT / "bench_flfix.csv", index=False)
    (V4_OUT / "bench_flfix.md").write_text(md(T[T.consts == "lodo"], ["key"] + V4_SHOW) + "\n\n"
                                           + md(T[T.consts == "orig"], ["key"] + V4_SHOW) + "\n")
    # bootstrap CIs vs C4 (LODO constants)
    B = []
    base = PI[V4_KEY["C4"]]
    for key, nm in V4_KEY.items():
        if nm in PI and key != "C4":
            B += v4_boot_table(PI[nm], base, key, "C4")
    for key in ("fix2 C6pp", "fix3 conf_floor"):   # also against C6 (same FL shrinker family)
        if V4_KEY[key] in PI:
            B += v4_boot_table(PI[V4_KEY[key]], PI[V4_KEY["C6"]], key, "C6")
    for key, nm in V4_KEY.items():                 # isolate fix 3 / fix 6 from fix 2: against C4pp
        if nm in PI and (key.startswith("fix6") or (key.startswith("fix3") and "no pp" not in key)):
            B += v4_boot_table(PI[nm], PI[V4_KEY["fix2 C4pp"]], key, "C4pp")
    Bt = pd.DataFrame(B)
    Bt.to_csv(V4_OUT / "boot_vs_C4.csv", index=False)
    # per-image tables (key recipes)
    for part in ("static", "vl3", "vl4", "dyn"):
        Lp = []
        for key, nm in V4_KEY.items():
            if nm in PI:
                x = PI[nm][part].copy()
                x.insert(0, "recipe", key)
                Lp.append(x)
        pd.concat(Lp).rename_axis("image_id").to_csv(V4_OUT / f"per_image_bench_{part}.csv")
    lay = layout()
    Lt = []
    keep = ["group", "clip", "confidence", "pa_deep_raw", "pa_deep_c", "pa_deg", "fl_mm", "mt_mm", "fl_straight",
            "fl_central3", "sigma_fl", "mu_fl", "tau_fl", "w_fl", "fl_eb", "fl_tri", "s_eff", "a_floor", "pa_eb"]
    for key, nm in V4_KEY.items():
        if nm in PER:
            P = PER[nm]
            x = P[[c for c in keep if c in P]].copy()
            s = SUBS[nm]
            x["sub_pa_deg"], x["sub_fl_mm"], x["sub_mt_mm"] = s.pa_deg, s.fl_mm, s.mt_mm
            x["sub_k"] = s.fl_mm * np.sin(np.radians(s.pa_deg)) / s.mt_mm
            x.insert(0, "recipe", key)
            Lt.append(x)
    pd.concat(Lt).rename_axis("image_id").to_csv(V4_OUT / "per_image_test.csv")
    # candidate CSVs (validated) + a small JSON report
    rep, csubs = {}, {}
    for name, kw in V4_CANDS.items():
        rec = R(**{k: v for k, v in kw.items() if k != "desc"})
        sub, P, _ = test_sub_v4(rec, return_per_image=True)
        path = V4_OUT / f"{name}.csv"
        bc.write_sub(sub, path)
        csubs[name] = sub
        grp = lay.loc[sub.index, "group"]
        g = sub.groupby(grp)[list(bc.C3)].median().round(2)
        g["k"] = (sub.fl_mm * np.sin(np.radians(sub.pa_deg)) / sub.mt_mm).groupby(grp).median().round(3)
        rep[name] = dict(file=path.name, recipe=rec, desc=kw.get("desc", ""), validate=validate(path),
                         dist={k: bc.official_distance(sub, v) for k, v in refsubs.items()},
                         group=g.reset_index().to_dict("records"), diag=v4_test_diag(sub, P, refsubs))
        print(f"== {name}: {kw.get('desc', '')}\n{rep[name]['validate']}\n   vs C4 {rep[name]['dist']['C4']}"
              f"\n   vs C6 {rep[name]['dist']['C6']}\n   vs C2 {rep[name]['dist']['C2']}", flush=True)
    nm_ = list(csubs)
    rep["pairwise"] = {f"{a}-{b}": bc.official_distance(csubs[a], csubs[b]) for i, a in enumerate(nm_) for b in nm_[i + 1:]}
    (V4_OUT / "candidates_flfix_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(T[T.consts == "lodo"][["key"] + [c for c in V4_SHOW if c in T]].round(3).to_string(index=False))
    show = Bt[Bt.bench.isin(["clean10 mean7", "2r all35", "2r clean10", "VL-244 lodo3", "VL-244 lodo4",
                             "dynamic (PA_h+FL)", "dynamic FL"])]
    print(show.round(4).to_string(index=False))


# ----------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--dataset", required=True, choices=["static", "dynamic", "vl", "test", "train142"])
    m.add_argument("--apo", default="v2", choices=list(APO_SRC))
    m.add_argument("--limit", type=int, default=0)
    sub.add_parser("bench")
    b = sub.add_parser("build")
    b.add_argument("--spec", help="optional JSON {name: {apo, pa, fl, shrink, xfeat, desc}}")
    mx = sub.add_parser("measure_extra")    # flfix (v4): extra PA columns -> artifacts/eval_v4/flfix/cache
    mx.add_argument("--dataset", required=True, choices=["static", "dynamic", "vl", "test"])
    mx.add_argument("--apo", default="v2", choices=list(APO_SRC))
    mx.add_argument("--limit", type=int, default=0)
    fx = sub.add_parser("flfix")            # flfix (v4): validation grid -> artifacts/eval_v4/flfix
    fx.add_argument("--quick", action="store_true", help="only the planned contrasts (V4_KEY)")
    args = ap.parse_args()
    {"measure": cmd_measure, "bench": cmd_bench, "build": cmd_build, "measure_extra": cmd_measure_extra,
     "flfix": cmd_flfix}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
