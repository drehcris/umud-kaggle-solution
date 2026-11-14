"""Learned fascicle-orientation field (fasc_orient_v1) in geometry v2: what does it buy?

Light by design (laptop rules): no neural network, one image at a time, lazy npz keys (one key per
access), one process, ~0.3-0.5 GB RSS.

Subcommands
-----------
measure --dataset {static,dynamic,vl,test,train142} [--fsrc ours|seg|union]
    Runs g2.measure_field (= recommended v2e geometry of candidate C1 + field_architecture) and caches
    every variant per image in artifacts/eval_v3/orient/cache/<dataset>__<fsrc>.csv (resumable).
      fsrc = fragment source for the existing variants (pair selection, pa_global, fl_straight, ...):
        ours  : ours_fasc384_noflip @0.95 (the C1 maps)
        seg   : fasc_orient_v1 seg channel @SEG_THR
        union : calibrated max of both @0.5
      The aponeurosis is always C1's (ours_apo384_noflip @0.8 + dlt_apo fallback) on the benchmark /
      test sets, so every difference comes from the fascicle side.
    train142: the 142 training images with both GT masks, in the GT-mask frame, px units:
      gt   : GT apo + GT fascicle masks, field rasterised from the GT fragments ("GT-mask geometry")
      oofF : GT apo + fasc_orient_v1 train_oof maps (fascicle model only)
      oofAF: apo_v2 train_oof band @0.5 + fasc_orient_v1 train_oof maps (both models out of fold)
table
    Scores PA / FL heads on static (mean7 / clean10 / 2-rater / R7), dynamic (PA vs horizontal, FL),
    VL-244 and train142 -> artifacts/eval_v3/orient/bench_orient.csv/.md, error correlations,
    per-group test summaries (Telemed clips) -> test_groups.csv, test_clips.csv.
"""
from __future__ import annotations

import argparse
import json
import math
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
import build_candidates as bc  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path(os.environ.get("ORIENT_OUT", ROOT / "artifacts/eval_v3/orient"))  # override: outputs of another map source
CACHE = Path(os.environ.get("ORIENT_CACHE", OUT / "cache"))  # override only for smoke tests
OBANK_ROOT = Path(os.environ.get("ORIENT_BANK", ROOT / "artifacts/kaggle/fasc_orient_v1/maskbank"))
OMODEL = os.environ.get("ORIENT_MODEL", "fasc_orient_v1")  # override: fasc_orient_v2 / fasc_orient_v2ms (+ ORIENT_BANK)
FOLDS_CSV = Path(os.environ.get("ORIENT_FOLDS", ROOT / "artifacts/kaggle/fasc_orient_v1/metrics/folds.csv"))
APO2_ROOT = ROOT / "artifacts/kaggle/apo_v2/maskbank"
PAIRS = bc.SCRATCH / "data-labels-geometry/out/apo_fasc_pairs.csv"
COMP = bc.COMP
SEG_THR = 0.5
OURS_FT = 0.95
FCFG = g2.FieldConfig()


# ----------------------------------------------------------------------------------- key maps
def _tail(p: str, n: int = 3) -> str:
    parts = Path(str(p).replace("\\", "/")).parts
    return "/".join(parts[-n:]).lower()


def orient_keymap(ds: str) -> dict:
    """main-pipeline image id -> fasc_orient_v1 key (only VL needs a real mapping)."""
    if ds != "vl":
        return {}
    man = pd.read_csv(OBANK_ROOT / "manifest.csv")
    man = man[man.dataset == "vl"]
    by_tail = {}
    for r in man.itertuples():
        by_tail.setdefault(_tail(r.source_filename), r.image_id)
    idx = pd.read_csv(bc.VL / "vl_index.csv")
    out = {}
    for r in idx.itertuples():
        k = by_tail.get(_tail(r.raw_source))
        if k is not None:
            out[r.key] = k
    return out


def load_orient(obank: g2.MaskBank, ds: str, key: str):
    """(seg prob, cos2, sin2) float32, one key at a time."""
    seg = obank.raw(ds, f"{OMODEL}_seg", key)
    c = obank.raw(ds, f"{OMODEL}_cos2", key)
    s = obank.raw(ds, f"{OMODEL}_sin2", key)
    if seg is None or c is None or s is None:
        return None
    return g2.as_prob(seg), g2.decode_orient(c), g2.decode_orient(s)


def fasc_map(fsrc: str, ours: np.ndarray | None, seg: np.ndarray | None):
    if fsrc == "ours":
        return ours, OURS_FT
    if fsrc == "seg":
        return seg, SEG_THR
    if ours is None or seg is None:
        return None, 0.5
    return np.maximum(bc.calibrate(ours, OURS_FT), bc.calibrate(seg, SEG_THR)), 0.5


def extra_constructed(res: g2.GeoResult, pas: dict) -> dict:
    """Constructed straight fascicles (C1 recipe) at extra PA values."""
    cv = res.curves or {}
    if "sup_curve" not in cv:
        return {f"fl_c_{k}": np.nan for k in pas}
    sup, deep, s_lo, s_hi, mt_px, mm = (cv[k] for k in ("sup_curve", "deep_curve", "s_lo", "s_hi", "mt_px", "mm"))
    sgn = res.diag.get("fd_sign", np.nan)
    if not np.isfinite(sgn):
        sgn = 1.0
    W = res.diag.get("W", 1000)
    xgrid = np.linspace(-3 * W, 4 * W, 6001)
    out = {}
    for k, pa in pas.items():
        if not (np.isfinite(pa) and pa > 0.5):
            out[f"fl_c_{k}"] = np.nan
            continue
        cons = [g2.constructed_fl(sup, deep, s_lo + f * (s_hi - s_lo), pa, sgn, mt_px, "deep", xgrid)
                for f in (0.4, 0.5, 0.6)]
        cons = [c for c in cons if np.isfinite(c)]
        out[f"fl_c_{k}"] = float(np.mean(cons)) * mm if cons else np.nan
    return out


def row_of(res: g2.GeoResult) -> dict:
    r = res.as_row()
    v = res.variants
    pg, ps, pf, po = (v.get(k, np.nan) for k in ("pa_global", "pa_structure", "pa_fd_strip_g", "pa_field"))
    fus3 = np.nanmean([pg, ps, pf]) if np.isfinite([pg, ps, pf]).any() else np.nan
    fs = np.nanmean([ps, pf]) if np.isfinite([ps, pf]).any() else np.nan
    pfu = v.get("pa_fusion", np.nan)
    fF = np.nanmean([pfu, pf]) if np.isfinite([pfu, pf]).any() else np.nan     # 0.5 C1 fusion + 0.5 field
    fFo = np.nanmean([pfu, po]) if np.isfinite([pfu, po]).any() else np.nan
    r.update(extra_constructed(res, {"fus3": fus3, "fd": pf, "fs": fs, "fo": po, "fF": fF, "fFo": fFo}))
    return r


# ----------------------------------------------------------------------------------- measure
def cmd_measure(args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.dataset == "train142":
        return measure_train142(args)
    path = CACHE / f"{args.dataset}__{args.fsrc}.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(prev.image_id) if len(prev) else set()
    bank = g2.MaskBank(bc.BANK_ROOT)
    obank = g2.MaskBank(OBANK_ROOT)
    kmap = orient_keymap(args.dataset)
    src = bc.SOURCES["ours"]
    t0, rows, k = time.time(), [], 0
    for iid, img, ppc, extra in bc.iterate(args.dataset):
        if iid in done:
            continue
        okey = kmap.get(iid, Path(iid).stem) if args.dataset == "vl" else Path(iid).stem
        A, ta = bc.load_maps(bank, args.dataset, iid, src["apo"])
        orient = load_orient(obank, args.dataset, okey)
        ours = bank.prob(args.dataset, "ours_fasc384_noflip", iid) if args.fsrc != "seg" else None
        F, tf = fasc_map(args.fsrc, ours, orient[0] if orient is not None else None)
        cfg = bc.make_cfg("v2e", ta, tf)
        if A is None or F is None or orient is None or orient[1].shape != A.shape:
            res = g2._nan_result("missing_maps")
            r = res.as_row()
        else:
            fb = bank.prob(args.dataset, src["fb"], iid)
            if fb is not None:
                fb = bc.recalibrate(fb, dict(bc.OA + bc.DA)[src["fb"]], ta)
            try:
                res = g2.measure_field(A, F, img, ppc, cfg, orient=orient, fcfg=FCFG, apo_fallback=fb)
                r = row_of(res)
            except Exception as e:  # keep going; record the failure
                res = g2._nan_result(f"error:{type(e).__name__}:{e}")
                r = res.as_row()
        rows.append(dict(image_id=iid, okey=okey, px_per_cm=ppc, **extra, **r))
        del A, F, orient, ours
        k += 1
        if k % 25 == 0:
            pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
            print(f"  {args.dataset}/{args.fsrc}: {k} new, {time.time() - t0:.0f}s", flush=True)
    pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    bank.close()
    obank.close()
    print(f"done {path.name}: {k} new rows in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------- train142
def gt_orient_from_mask(F: np.ndarray, dil: int = 1):
    """Dense (seg, cos2, sin2) from GT fragment masks: each component gets its PCA angle."""
    from skimage.measure import label, regionprops
    H, W = F.shape
    seg = np.zeros((H, W), np.float32)
    c2 = np.zeros((H, W), np.float32)
    s2 = np.zeros((H, W), np.float32)
    lab = label(F, connectivity=2)
    k = np.ones((2 * dil + 1, 2 * dil + 1), np.uint8)
    for r in regionprops(lab):
        if r.area < 5:
            continue
        ys, xs = r.coords[:, 0].astype(float), r.coords[:, 1].astype(float)
        ev, evec = np.linalg.eigh(np.cov(np.stack([xs - xs.mean(), ys - ys.mean()])))
        th = math.atan2(evec[1, 1], evec[0, 1])
        y0, x0, y1, x1 = r.bbox
        y0, x0, y1, x1 = max(0, y0 - dil), max(0, x0 - dil), min(H, y1 + dil), min(W, x1 + dil)
        m = cv2.dilate((lab[y0:y1, x0:x1] == r.label).astype(np.uint8), k) > 0
        seg[y0:y1, x0:x1][m] = 1.0
        c2[y0:y1, x0:x1][m] = math.cos(2 * th)
        s2[y0:y1, x0:x1][m] = math.sin(2 * th)
    return seg, c2, s2


def to_mask_frame(orient, shape):
    """Resize (seg, cos2, sin2) from the image frame to the GT-mask frame; angles follow the stretch."""
    seg, c2, s2 = orient
    Hm, Wm = shape
    Hi, Wi = c2.shape
    if (Hi, Wi) == (Hm, Wm):
        return orient
    rs = lambda a: cv2.resize(a, (Wm, Hm), interpolation=cv2.INTER_LINEAR)  # noqa: E731
    seg, c2, s2 = rs(seg), rs(c2), rs(s2)
    mag = np.hypot(c2, s2)
    th = 0.5 * np.arctan2(s2, c2)
    sx, sy = Wm / Wi, Hm / Hi
    thm = np.arctan2(sy * np.sin(th), sx * np.cos(th))
    return seg, (mag * np.cos(2 * thm)).astype(np.float32), (mag * np.sin(2 * thm)).astype(np.float32)


def rep_map() -> dict:
    if not FOLDS_CSV.exists():
        return {}
    f = pd.read_csv(FOLDS_CSV)
    scol = "stem" if "stem" in f else f.columns[0]
    rcol = "representative" if "representative" in f else f.columns[1]
    return dict(zip(f[scol].astype(str), f[rcol].astype(str)))


def measure_train142(args) -> None:
    path = CACHE / "train142.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(zip(prev.image, prev.run)) if len(prev) else set()
    pairs = pd.read_csv(PAIRS)
    reps = rep_map()
    obank = g2.MaskBank(OBANK_ROOT)
    abank = g2.MaskBank(APO2_ROOT)
    base = g2.recommended_config(apo_thr=0.5, fasc_thr=0.5, compute_streamline=False, edge_extrap="end_linear")
    t0, rows = time.time(), []
    for k, r in enumerate(pairs.itertuples()):
        Ag = bc.gray(COMP / "apo_masks_v1/apo_masks_new_model_v1" / f"{r.apo}.tif") > 0
        Fg = bc.gray(COMP / "fasc_masks_v1/fasc_masks_new_model_v1" / f"{r.fasc}.tif") > 0
        if Ag.shape != Fg.shape:
            continue
        img = bc.gray(COMP / "apo_imgs_v1/apo_images_new_model_v1" / f"{r.apo}.tif")
        if img.shape != Ag.shape:
            img = np.asarray(Image.fromarray(img).resize((Ag.shape[1], Ag.shape[0]), Image.BILINEAR))
        okey = reps.get(r.fasc, r.fasc)
        meta = dict(image=r.apo, fasc=r.fasc, okey=okey, layout=r.orig, H=Ag.shape[0], W=Ag.shape[1])
        o = load_orient(obank, "train_oof", okey)
        o = to_mask_frame(o, Ag.shape) if o is not None else None
        ab = abank.prob("train_oof", "apo_v2_band", r.apo)
        if ab is not None and ab.shape != Ag.shape:
            ab = cv2.resize(ab, (Ag.shape[1], Ag.shape[0]), interpolation=cv2.INTER_LINEAR)
        runs = {"gt": (Ag.astype(np.float32), Fg.astype(np.float32), gt_orient_from_mask(Fg),
                       base.with_(compute_structure=False))}
        if o is not None:
            runs["oofF"] = (Ag.astype(np.float32), o[0], o, base)
            if ab is not None:
                runs["oofAF"] = (ab, o[0], o, base)
        for run, (A, F, orient, cfg) in runs.items():
            if (r.apo, run) in done:
                continue
            try:
                res = g2.measure_field(A, F, img, None, cfg, orient=orient, fcfg=FCFG)
                rr = row_of(res)
            except Exception as e:
                rr = g2._nan_result(f"error:{type(e).__name__}:{e}").as_row()
            rows.append(dict(meta, run=run, **rr))
        if k % 10 == 9:
            pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
            print(f"  train142 {k + 1}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    obank.close()
    abank.close()
    print(f"done train142: {len(rows)} new rows in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------- heads
FIELD_MIN_SUPPORT = 0.02   # mean |v|*seg in the deep strip below which the field carries no evidence


def field_ok(d: pd.DataFrame) -> pd.Series:
    w = d["d_fd_strip_wmean"].astype(float) if "d_fd_strip_wmean" in d else pd.Series(np.nan, index=d.index)
    return (w >= FIELD_MIN_SUPPORT) & d.get("pa_fd_strip_g", pd.Series(np.nan, index=d.index)).notna()


def pa_heads(d: pd.DataFrame) -> dict:
    """name -> Series of PA predictions (deep reference)."""
    g = lambda c: d[c].astype(float) if c in d else pd.Series(np.nan, index=d.index)  # noqa: E731
    H = {
        "fusion(C1)": g("pa_fusion"), "global": g("pa_global"), "structure": g("pa_structure"),
        "fd_strip": g("pa_fd_strip"), "fd_strip_g": g("pa_fd_strip_g"), "fd_ins": g("pa_fd_ins"),
        "fd_comp_g": g("pa_fd_comp_g"), "fd_comp": g("pa_fd_comp"), "fdp_strip_g": g("pa_fdp_strip_g"),
        "sl_ins": g("pa_sl_ins"), "field_old": g("pa_field"),
    }
    H["fus3(glob,str,fd)"] = pd.concat([H["global"], H["structure"], H["fd_strip_g"]], axis=1).mean(axis=1)
    H["fus(str,fd)"] = pd.concat([H["structure"], H["fd_strip_g"]], axis=1).mean(axis=1)
    H["fus(glob,fd)"] = pd.concat([H["global"], H["fd_strip_g"]], axis=1).mean(axis=1)
    H["fus(str,fdold)"] = pd.concat([H["structure"], H["field_old"]], axis=1).mean(axis=1)
    H["fusF(C1,fd)"] = pd.concat([H["fusion(C1)"], H["fd_strip_g"]], axis=1).mean(axis=1)
    H["fusF(C1,fdold)"] = pd.concat([H["fusion(C1)"], H["field_old"]], axis=1).mean(axis=1)
    ok = field_ok(d)
    H["fusFg"] = H["fusF(C1,fd)"].where(ok, H["fusion(C1)"])          # gated: no field support -> C1
    H["fus3c(glob,str,fdcomp)"] = pd.concat([H["global"], H["structure"], H["fd_comp_g"]], axis=1).mean(axis=1)
    H["fus4(+sl_ins)"] = pd.concat([H["global"], H["structure"], H["fd_strip_g"], H["sl_ins"]], axis=1).mean(axis=1)
    return H


def fl_heads(d: pd.DataFrame) -> dict:
    g = lambda c: d[c].astype(float) if c in d else pd.Series(np.nan, index=d.index)  # noqa: E731
    H = {"ensemble(C1)": g("fl_ensemble"), "straight": g("fl_straight"), "central3": g("fl_central3"),
         "constructed": g("fl_constructed"), "sl_med": g("fl_sl_med"), "sl_mean": g("fl_sl_mean"),
         "sl_c3": g("fl_sl_c3"), "sl_chord": g("fl_sl_chord"), "c_fd": g("fl_cfd"), "c_fus3": g("fl_c_fus3")}
    H["c_fs"], H["c_fo"] = g("fl_c_fs"), g("fl_c_fo")
    H["ens(c@fd)"] = pd.concat([H["straight"], H["central3"], H["c_fd"]], axis=1).mean(axis=1)
    H["ens(c@fs)"] = pd.concat([H["straight"], H["central3"], H["c_fs"]], axis=1).mean(axis=1)
    H["ens(c@fo)"] = pd.concat([H["straight"], H["central3"], H["c_fo"]], axis=1).mean(axis=1)
    H["c_fF"], H["c_fFo"] = g("fl_c_fF"), g("fl_c_fFo")
    H["ens(c@fusF)"] = pd.concat([H["straight"], H["central3"], H["c_fF"]], axis=1).mean(axis=1)
    H["ens(c@fusFo)"] = pd.concat([H["straight"], H["central3"], H["c_fFo"]], axis=1).mean(axis=1)
    H["ens(c@fusFg)"] = H["ens(c@fusF)"].where(field_ok(d), H["ensemble(C1)"])
    H["ens+sl"] = pd.concat([H["straight"], H["central3"], H["constructed"], H["sl_med"]], axis=1).mean(axis=1)
    H["ens(c@fus3)"] = pd.concat([H["straight"], H["central3"], H["c_fus3"]], axis=1).mean(axis=1)
    H["ens(c@fus3)+sl"] = pd.concat([H["straight"], H["central3"], H["c_fus3"], H["sl_med"]], axis=1).mean(axis=1)
    H["mean(sl,c@fus3)"] = pd.concat([H["sl_med"], H["c_fus3"]], axis=1).mean(axis=1)
    H["mean(ens,sl)"] = pd.concat([H["ensemble(C1)"], H["sl_med"]], axis=1).mean(axis=1)
    return H


def mae_n(pred: pd.Series, gt: pd.Series, tol: float) -> tuple[float, float, int]:
    ids = pred.index.intersection(gt.index)
    p = pred.loc[ids].astype(float)
    p = p.fillna(p.median())
    e = (p - gt.loc[ids].astype(float)).dropna()
    return (float(e.abs().mean() / tol) if len(e) else np.nan, float(e.mean()) if len(e) else np.nan, len(e))


def load(ds: str, fsrc: str) -> pd.DataFrame | None:
    p = CACHE / f"{ds}__{fsrc}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).drop_duplicates("image_id", keep="last").set_index("image_id")


def cmd_table(args) -> None:
    rows, corr_rows = [], []
    G = bc.static_gts()
    vgt = bc.vl_gt()
    dgt = bc.dynamic_gt()
    dyn_h = bc.load_cache("dynamic", "ours", "v2h")
    for fsrc in ("ours", "seg", "union"):
        st, vl, dy = load("static", fsrc), load("vl", fsrc), load("dynamic", fsrc)
        for kind, heads_fn, col, tol in (("PA", pa_heads, "pa_deg", 6.0), ("FL", fl_heads, "fl_mm", 12.0)):
            names = list(heads_fn(st if st is not None else (vl if vl is not None else dy)).keys()) \
                if any(x is not None for x in (st, vl, dy)) else []
            for h in names:
                r = dict(fsrc=fsrc, comp=kind, head=h)
                if st is not None:
                    p = heads_fn(st)[h]
                    r["s35"], r["s35_bias"], _ = mae_n(p, G["mean7"][col], tol)
                    r["c10"], _, _ = mae_n(p.loc[bc.CLEAN], G["mean7"][col], tol)
                    r["2r"] = float(np.mean([mae_n(p, G[f"R7+{x}"][col], tol)[0] for x in bc.RATERS[:-1]]))
                    r["R7"], _, _ = mae_n(p, G["R7"][col], tol)
                if vl is not None:
                    p = heads_fn(vl)[h]
                    r["vl"], r["vl_bias"], r["vl_n"] = mae_n(p, vgt[col], tol)
                if dy is not None:
                    if kind == "FL":
                        p = heads_fn(dy)[h]
                        r["dyn"], r["dyn_bias"], _ = mae_n(p, dgt[col], tol)
                    else:
                        ph = dyn_pa_h(dy, dyn_h).get(h)
                        if ph is not None:
                            r["dyn_h"], r["dyn_h_bias"], _ = mae_n(ph, dgt[col], tol)
                rows.append(r)
        # per-image error correlation field vs fragment PA (complementarity)
        for ds, d, gt in (("static", st, G["mean7"]), ("vl", vl, vgt)):
            if d is None:
                continue
            H = pa_heads(d)
            ids = d.index.intersection(gt.index)
            e = {k: (H[k].loc[ids] - gt.loc[ids, "pa_deg"]) for k in ("global", "structure", "fd_strip_g", "fd_strip",
                                                                        "fd_comp_g", "sl_ins", "fusion(C1)")}
            E = pd.DataFrame(e).dropna()
            for a, b in (("global", "fd_strip_g"), ("structure", "fd_strip_g"), ("fusion(C1)", "fd_strip_g"),
                         ("global", "structure"), ("global", "sl_ins"), ("fd_strip_g", "sl_ins")):
                corr_rows.append(dict(fsrc=fsrc, ds=ds, a=a, b=b, n=len(E), r=float(E[a].corr(E[b])),
                                      mae_a=float(E[a].abs().mean()), mae_b=float(E[b].abs().mean()),
                                      mae_avg=float(((E[a] + E[b]) / 2).abs().mean())))
        if dy is not None and dyn_h is not None:
            P = dyn_pa_h(dy, dyn_h)
            ids = dy.index.intersection(dgt.index)
            E = pd.DataFrame({k: P[k].loc[ids] - dgt.loc[ids, "pa_deg"] for k in ("fusion(C1)", "structure", "fd_strip_g",
                                                                                 "global")}).dropna()
            for a, b in (("global", "fd_strip_g"), ("structure", "fd_strip_g"), ("fusion(C1)", "fd_strip_g")):
                corr_rows.append(dict(fsrc=fsrc, ds="dynamic_h", a=a, b=b, n=len(E), r=float(E[a].corr(E[b])),
                                      mae_a=float(E[a].abs().mean()), mae_b=float(E[b].abs().mean()),
                                      mae_avg=float(((E[a] + E[b]) / 2).abs().mean())))
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "bench_orient.csv", index=False)
    C = pd.DataFrame(corr_rows)
    C.to_csv(OUT / "pa_error_correlation.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(T.round(3).to_string(index=False))
    print(C.round(3).to_string(index=False))
    R = recipe_table()
    print(R.round(3).to_string(index=False))
    tr = train142_table()
    if tr is not None:
        print(tr.round(3).to_string(index=False))


RECIPES = [("fusion(C1)", "ensemble(C1)"), ("fus3(glob,str,fd)", "ensemble(C1)"), ("fus3(glob,str,fd)", "ens(c@fus3)"),
           ("fus3(glob,str,fd)", "ens(c@fus3)+sl"), ("fusion(C1)", "ens+sl"), ("fd_strip_g", "sl_med"),
           ("fus(str,fd)", "ens(c@fus3)+sl"), ("fusion(C1)", "mean(ens,sl)"), ("fusF(C1,fd)", "ensemble(C1)"),
           ("fusF(C1,fd)", "ens(c@fusF)"), ("fusF(C1,fdold)", "ens(c@fusFo)"), ("fd_strip_g", "ens(c@fd)"),
           ("field_old", "ens(c@fo)"), ("fusFg", "ens(c@fusFg)")]


def recipe_frame(d: pd.DataFrame, pa: str, fl: str) -> pd.DataFrame:
    p = pd.DataFrame({"pa_deg": pa_heads(d)[pa], "fl_mm": fl_heads(d)[fl], "mt_mm": d.mt_mm.astype(float)},
                     index=d.index)
    p.loc[p.mt_mm < 5, list(bc.C3)] = np.nan
    return p


def recipe_table() -> pd.DataFrame:
    """C1 (per image) / C2 (confidence shrink) recipes with field heads, on static / VL / dynamic."""
    G, vgt, dgt = bc.static_gts(), bc.vl_gt(), bc.dynamic_gt()
    rows = []
    for fsrc in ("ours", "seg", "union"):
        st, vl, dy = load("static", fsrc), load("vl", fsrc), load("dynamic", fsrc)
        if st is None or vl is None:
            continue
        sg = bc.static_groups(st.index)
        for pa, fl in RECIPES:
            for lam in (0.0, 1.0):
                r = dict(fsrc=fsrc, pa=pa, fl=fl, lam=lam)
                p = bc.fill(recipe_frame(st, pa, fl), sg)
                if lam:
                    p = bc.shrink(p, st.confidence, sg, lam)
                m = bc.s_metric(p, G["mean7"])
                r.update(s35=m["S"], s35_pa=m["pa_n"], s35_fl=m["fl_n"], s35_mt=m["mt_n"])
                r["c10"] = bc.s_metric(p.loc[bc.CLEAN], G["mean7"])["S"]
                r["2r"] = float(np.mean([bc.s_metric(p, G[f"R7+{x}"])["S"] for x in bc.RATERS[:-1]]))
                r["R7"] = bc.s_metric(p, G["R7"])["S"]
                gv = pd.Series("vl", index=vl.index)
                p = bc.fill(recipe_frame(vl, pa, fl), gv)
                if lam:
                    p = bc.shrink(p, vl.confidence, gv, lam)
                m = bc.s_metric(p, vgt, ("pa_deg", "fl_mm"))
                r.update(vl=m["S"], vl_pa=m["pa_n"], vl_fl=m["fl_n"], vl_pa_bias=m["pa_bias"], vl_fl_bias=m["fl_bias"])
                r["vl_young3"] = bc.s_metric(p.loc[[i for i in p.index if i.startswith("young")]], vgt)["S"]
                if dy is not None and lam == 0.0:
                    m = bc.s_metric(bc.fill(recipe_frame(dy, pa, fl)), dgt, ("fl_mm",))
                    r["dyn_fl"] = m["fl_n"]
                rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "bench_recipes.csv", index=False)
    return T


def dyn_pa_h(dy: pd.DataFrame, dyn_h: pd.DataFrame | None) -> dict:
    """PA heads against the HORIZONTAL for the dynamic benchmark: fragment / classical heads from the
    C1 v2h cache, field heads from the horizontal-reference field variants of our run."""
    out = {}
    g = lambda c: dy[c].astype(float) if c in dy else pd.Series(np.nan, index=dy.index)  # noqa: E731
    if dyn_h is not None:
        dh = dyn_h.reindex(dy.index)
        out["fusion(C1)"] = dh.pa_fusion.astype(float)
        out["global"] = dh.pa_global.astype(float)
        out["structure"] = dh.pa_structure.astype(float)
    out["fd_strip_g"] = g("pa_fd_strip_h")
    out["fd_strip"] = g("pa_fd_strip_h")
    out["fd_comp_g"] = g("pa_fd_comp_h")
    out["fdp_strip_g"] = g("pa_fdp_strip_h")
    if dyn_h is not None:
        out["fus3(glob,str,fd)"] = pd.concat([out["global"], out["structure"], out["fd_strip_g"]], axis=1).mean(axis=1)
        out["fus(str,fd)"] = pd.concat([out["structure"], out["fd_strip_g"]], axis=1).mean(axis=1)
        out["fus(glob,fd)"] = pd.concat([out["global"], out["fd_strip_g"]], axis=1).mean(axis=1)
        out["fus3c(glob,str,fdcomp)"] = pd.concat([out["global"], out["structure"], out["fd_comp_g"]], axis=1).mean(axis=1)
        out["fusF(C1,fd)"] = pd.concat([out["fusion(C1)"], out["fd_strip_g"]], axis=1).mean(axis=1)
        out["fusFg"] = out["fusF(C1,fd)"].where(field_ok(dy), out["fusion(C1)"])
    return out


def train142_table() -> pd.DataFrame | None:
    p = CACHE / "train142.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    gt = d[d.run == "gt"].set_index("image")
    rows = []
    gt_pa = {"gt_global": gt.pa_global, "gt_fd_strip_g": gt.pa_fd_strip_g, "gt_fd_comp_g": gt.pa_fd_comp_g}
    gt_fl = {"gt_straight": gt.fl_straight, "gt_sl_med": gt.fl_sl_med, "gt_ensemble": gt.fl_ensemble}
    for run in ("oofF", "oofAF"):
        x = d[d.run == run].set_index("image")
        if not len(x):
            continue
        H = pa_heads(x)
        for h, s in H.items():
            for gname, gs in gt_pa.items():
                ids = s.index.intersection(gs.dropna().index)
                e = (s.loc[ids] - gs.loc[ids]).dropna()
                rows.append(dict(run=run, comp="PA", head=h, gt=gname, n=len(e), mae=float(e.abs().mean()),
                                 medae=float(e.abs().median()), bias=float(e.mean())))
        F = fl_heads(x)
        for h, s in F.items():
            for gname, gs in gt_fl.items():
                ids = s.index.intersection(gs.dropna().index)
                e = ((s.loc[ids] - gs.loc[ids]) / gs.loc[ids]).dropna()
                rows.append(dict(run=run, comp="FL_rel", head=h, gt=gname, n=len(e), mae=float(e.abs().mean()),
                                 medae=float(e.abs().median()), bias=float(e.mean())))
        mt = (x.mt_mm - gt.mt_mm.reindex(x.index)) / gt.mt_mm.reindex(x.index)
        rows.append(dict(run=run, comp="MT_rel", head="mt", gt="gt", n=int(mt.notna().sum()),
                         mae=float(mt.abs().mean()), medae=float(mt.abs().median()), bias=float(mt.mean())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "train142_orient.csv", index=False)
    return T


# ----------------------------------------------------------------------------------- test report
QUAL = ["confidence", "flags", "d_fd_strip_conc", "d_fd_strip_inl", "d_fd_strip_wmean", "d_fd_strip_npx",
        "d_fd_comp_g_conc", "d_fd_seg_frac", "d_sl_n_ok", "d_sl_extrap", "d_sl_spread", "d_sl_curv",
        "d_sl_support_p90", "d_n_frag", "d_pa_iqr", "d_deep_angle", "d_pair", "d_n_bands", "d_extrap_med"]


def compact(d: pd.DataFrame) -> pd.DataFrame:
    P, F = pa_heads(d), fl_heads(d)
    out = pd.DataFrame(index=d.index)
    for c in ("group", "clip", "px_per_cm", "mt_mm"):
        if c in d:
            out[c] = d[c]
    for k, v in P.items():
        out[f"pa:{k}"] = v
    for k, v in F.items():
        out[f"fl:{k}"] = v
    for c in QUAL:
        if c in d:
            out[c] = d[c]
    return out


def cmd_test(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lay = bc.test_layout()
    G, Cl = [], []
    for fsrc in ("ours", "seg", "union"):
        d = load("test", fsrc)
        if d is None:
            continue
        if "group" not in d:
            d = d.join(lay[["group", "clip"]])
        X = compact(d)
        X.to_csv(OUT / f"test_orient_{fsrc}.csv")
        pa_cols = ["pa:fusion(C1)", "pa:global", "pa:structure", "pa:fd_strip_g", "pa:fd_strip", "pa:fd_ins",
                   "pa:fd_comp_g", "pa:sl_ins", "pa:fus3(glob,str,fd)", "pa:field_old", "pa:fusF(C1,fd)", "pa:fusFg"]
        fl_cols = ["fl:ensemble(C1)", "fl:straight", "fl:central3", "fl:constructed", "fl:sl_med", "fl:sl_c3", "fl:c_fd",
                   "fl:c_fus3", "fl:ens(c@fus3)", "fl:ens(c@fus3)+sl", "fl:ens+sl", "fl:ens(c@fusF)", "fl:ens(c@fusFg)"]
        for grp, Z in list(X.groupby("group")) + [("ALL", X)]:
            r = dict(fsrc=fsrc, group=grp, n=len(Z), mt=float(Z.mt_mm.median()))
            for c in pa_cols + fl_cols:
                r[c] = float(Z[c].median())
            for pa, fl in (("pa:fusion(C1)", "fl:ensemble(C1)"), ("pa:fd_strip_g", "fl:sl_med"),
                           ("pa:fus3(glob,str,fd)", "fl:ens(c@fus3)+sl"), ("pa:fd_strip_g", "fl:ensemble(C1)"),
                           ("pa:fus3(glob,str,fd)", "fl:ensemble(C1)"), ("pa:fusFg", "fl:ens(c@fusFg)")):
                k = Z[fl] * np.sin(np.radians(Z[pa])) / Z.mt_mm
                r[f"k[{pa[3:]}|{fl[3:]}]"] = float(k.median())
            r["pa_implied_by_k(ensC1)"] = float(np.degrees(np.arcsin(np.clip(1.067 * Z.mt_mm / Z["fl:ensemble(C1)"], 0, 1))).median())
            r["pa_implied_by_k(sl_med)"] = float(np.degrees(np.arcsin(np.clip(1.067 * Z.mt_mm / Z["fl:sl_med"], 0, 1))).median())
            r["fd_conc_med"] = float(Z.d_fd_strip_conc.median()) if "d_fd_strip_conc" in Z else np.nan
            r["seg_frac_med"] = float(Z.d_fd_seg_frac.median()) if "d_fd_seg_frac" in Z else np.nan
            r["sl_extrap_med"] = float(Z.d_sl_extrap.median()) if "d_sl_extrap" in Z else np.nan
            G.append(r)
        T = X[X["clip"] >= 0]
        for clip, Z in T.groupby("clip"):
            r = dict(fsrc=fsrc, clip=int(clip), group=Z.group.iloc[0], ids=f"{Z.index.min()}..{Z.index.max()}",
                     mt=float(Z.mt_mm.median()))
            for c in pa_cols + fl_cols[:1] + ["fl:sl_med", "fl:ens(c@fusF)", "fl:ens(c@fusFg)"]:
                r[c] = float(Z[c].median())
            r["pa_sd_frames(fusion)"] = float(Z["pa:fusion(C1)"].std())
            r["pa_sd_frames(fd)"] = float(Z["pa:fd_strip_g"].std())
            r["k(C1)"] = float((Z["fl:ensemble(C1)"] * np.sin(np.radians(Z["pa:fusion(C1)"])) / Z.mt_mm).median())
            r["k(fd,sl)"] = float((Z["fl:sl_med"] * np.sin(np.radians(Z["pa:fd_strip_g"])) / Z.mt_mm).median())
            r["k(fusFg)"] = float((Z["fl:ens(c@fusFg)"] * np.sin(np.radians(Z["pa:fusFg"])) / Z.mt_mm).median())
            r["field_ok_frac"] = float(field_ok(d.loc[Z.index]).mean())
            r["conf"] = float(Z.confidence.median())
            r["fd_conc"] = float(Z.d_fd_strip_conc.median())
            Cl.append(r)
    G, Cl = pd.DataFrame(G), pd.DataFrame(Cl)
    G.to_csv(OUT / "test_groups.csv", index=False)
    Cl.to_csv(OUT / "test_clips.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    print(G.round(2).T.to_string())
    print(Cl[Cl.fsrc == "ours"].round(2).to_string(index=False))


# ----------------------------------------------------------------------------------- overlays
def cmd_viz(args) -> None:
    vdir = OUT / "viz"
    vdir.mkdir(parents=True, exist_ok=True)
    bank = g2.MaskBank(bc.BANK_ROOT)
    obank = g2.MaskBank(OBANK_ROOT)
    kmap = orient_keymap(args.dataset)
    want = set(args.ids.split(","))
    src = bc.SOURCES["ours"]
    for iid, img, ppc, extra in bc.iterate(args.dataset):
        if iid not in want and Path(iid).stem not in want:
            continue
        okey = kmap.get(iid, Path(iid).stem) if args.dataset == "vl" else Path(iid).stem
        A, ta = bc.load_maps(bank, args.dataset, iid, src["apo"])
        orient = load_orient(obank, args.dataset, okey)
        ours = bank.prob(args.dataset, "ours_fasc384_noflip", iid)
        F, tf = fasc_map(args.fsrc, ours, orient[0])
        fb = bank.prob(args.dataset, src["fb"], iid)
        if fb is not None:
            fb = bc.recalibrate(fb, dict(bc.OA + bc.DA)[src["fb"]], ta)
        res = g2.measure_field(A, F, img, ppc, bc.make_cfg("v2e", ta, tf), orient=orient, fcfg=FCFG, apo_fallback=fb)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
        seg = orient[0]
        rgb[..., 2] = rgb[..., 2] * (1 - 0.6 * seg) + 255 * 0.6 * seg      # seg channel: red
        om = ours >= OURS_FT
        rgb[om] = 0.4 * rgb[om] + 0.6 * np.array([255, 255, 0], np.float32)   # ours fragments: cyan
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        cv = res.curves or {}
        if "sup_curve" in cv:
            H, W = img.shape
            xs = np.arange(-W, 2 * W, 2.0)
            for crv, col in ((cv["sup_curve"], (0, 255, 0)), (cv["deep_curve"], (255, 128, 0))):
                pts = np.stack([xs, crv(xs)], 1).astype(np.int32)
                cv2.polylines(rgb, [pts], False, col, 1, cv2.LINE_AA)
            for P in cv.get("sl_paths", []):
                cv2.polylines(rgb, [P.astype(np.int32)], False, (0, 255, 255), 1, cv2.LINE_AA)
        v = res.variants
        txt = (f"{iid} PA fus {v.get('pa_fusion', np.nan):.1f} glob {v.get('pa_global', np.nan):.1f} "
               f"str {v.get('pa_structure', np.nan):.1f} | field strip {v.get('pa_fd_strip_g', np.nan):.1f} "
               f"comp {v.get('pa_fd_comp_g', np.nan):.1f} slins {v.get('pa_sl_ins', np.nan):.1f}")
        txt2 = (f"FL ens {v.get('fl_ensemble', np.nan):.1f} sl {v.get('fl_sl_med', np.nan):.1f} "
                f"c@fd {v.get('fl_cfd', np.nan):.1f} MT {res.mt_mm:.1f} conc {res.diag.get('fd_strip_conc', np.nan):.2f}")
        for k, t in enumerate((txt, txt2)):
            cv2.putText(rgb, t, (5, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(rgb, t, (5, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(vdir / f"{args.dataset}_{Path(iid).stem}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", vdir / f"{args.dataset}_{Path(iid).stem}.jpg", flush=True)
        del A, F, orient, ours, rgb
    bank.close()
    obank.close()


# ----------------------------------------------------------------------------------- export
def cmd_export(args) -> None:
    """Per-image recommended columns for the assembler (every dataset, fragment source ours):
    pa_rec = 0.5 * C1 fusion + 0.5 * learned-field deep-strip PA (pa_fd_strip_g; C1 fusion where the
    field is NaN), fl_rec = mean(straight, central3, constructed at pa_rec); MT and confidence are C1's."""
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["group", "clip", "px_per_cm", "confidence", "flags", "pa_global", "pa_structure", "pa_fusion",
            "pa_fd_strip_g", "pa_fd_strip", "pa_field", "pa_fd_comp_g", "pa_fd_strip_h", "pa_fd_comp_h",
            "fl_straight", "fl_central3", "fl_constructed", "fl_ensemble", "fl_c_fF", "fl_cfd", "fl_sl_med",
            "d_fd_strip_wmean", "d_fd_seg_frac", "d_sl_extrap", "d_n_frag", "d_pair", "d_deep_angle"]
    for ds in ("static", "dynamic", "vl", "test"):
        d = load(ds, "ours")
        if d is None:
            continue
        P, F = pa_heads(d), fl_heads(d)
        X = pd.DataFrame(index=d.index)
        X["pa_rec"] = P["fusF(C1,fd)"]
        X["fl_rec"] = F["ens(c@fusF)"]
        X["mt_mm"] = d.mt_mm
        X["pa_c1"] = P["fusion(C1)"]
        X["fl_c1"] = F["ensemble(C1)"]
        X["pa_fieldstrip"] = P["fd_strip_g"]
        X["fl_c_at_field"] = F["ens(c@fd)"]
        for c in keep:
            if c in d and c not in X:
                X[c] = d[c]
        if ds == "dynamic":
            dh = bc.load_cache("dynamic", "ours", "v2h")
            if dh is not None:
                X["pa_rec_h"] = dyn_pa_h(d, dh)["fusF(C1,fd)"]
        X.index.name = "image_id"
        X.to_csv(OUT / f"{ds}_orient_rec.csv")
        print(ds, len(X), "NaN pa_rec", int(X.pa_rec.isna().sum()), "NaN fl_rec", int(X.fl_rec.isna().sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--dataset", required=True, choices=["static", "dynamic", "vl", "test", "train142"])
    m.add_argument("--fsrc", default="ours", choices=["ours", "seg", "union"])
    sub.add_parser("table")
    sub.add_parser("test")
    sub.add_parser("export")
    v = sub.add_parser("viz")
    v.add_argument("--dataset", required=True, choices=["static", "dynamic", "vl", "test"])
    v.add_argument("--ids", required=True, help="comma-separated image ids")
    v.add_argument("--fsrc", default="ours", choices=["ours", "seg", "union"])
    args = ap.parse_args()
    {"measure": cmd_measure, "table": cmd_table, "test": cmd_test, "viz": cmd_viz, "export": cmd_export}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
