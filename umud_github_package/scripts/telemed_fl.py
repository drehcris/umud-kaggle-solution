"""Telemed ArtUS PA / FL study (28 test clips = 140 rows, 45 % of the test set).

Light by design (laptop rules): no neural network, one image at a time, lazy npz keys, one process,
well under 1 GB RSS (peak ~0.5 GB). Nothing here writes outside artifacts/eval_v5/telemed/.

Subcommands (run in this order to reproduce everything)
-----------
extra
    Re-measures static / dynamic / VL / test with the exact v3 aponeurosis + fragment inputs (no
    structure PA, no field) and stores extra straight-fascicle constructions on a PA grid (4..40 deg):
      cl_<pa>  PA vs LOCAL deep tangent, end_linear apo extrapolation  (== the v3 cache cg_* grid)
      cg_<pa>  PA vs the GLOBAL deep line,  end_linear extrapolation
      ll_<pa>  PA vs LOCAL deep tangent, GLOBAL-linear apo extrapolation   <- the proposed FL geometry
      lg_<pa>  PA vs GLOBAL deep line, global-linear extrapolation
    plus fragment straight / central-3 FL under both extrapolations -> cache/extra_<ds>.csv
frames
    Per-clip, per-estimator spread across the 5 near-identical frames -> frame_spread*.csv
trainval [--per-group 4]
    Telemed-layout TRAINING images (fascicle GT fragments; 1088x644 / 1340x810 / 1196x768 ArtUS and
    660x556 EchoBlaster): a classical, GT-guided aponeurosis pair (no network) is shared by a GT run
    (GT fragments + GT-derived orientation) and an OOF run (fasc_orient_v1 train_oof seg + field), so the
    PA / FL estimator error vs GT-fragment geometry is measured on Telemed layouts
    -> trainval_rows.csv, trainval_pa_errors.csv, trainval_pa_profile.csv, trainval_fl.csv, viz/ montages
bench
    (a) FL-given-PA calibration on the labelled benchmarks (expert FL / model at expert PA and at our PA,
    by FL/FOV regime); (b) full v4 pipeline (fusF PA, confidence PA shrink, LODO constants) with the FL
    head x FL-shrinkage grid; (c) paired bootstraps vs C8 -> bench_*.csv
build
    Telemed candidates (PA = C7; FL from the chosen head/shrink) -> telemed_T*.csv (image_id, pa_deg,
    fl_mm; 140 rows), telemed_per_image.csv, telemed_per_clip.csv
aponoise
    Apo-extrapolation part of the Telemed FL noise: ll(pa_c) with apo_v2 vs a classical detector
    -> apo_noise_test_telemed.csv
overlay [--clips 0 1 ...] [--frame 0]
    Exact v3/v4 measurement path on one frame per clip: panel A native crop with an angle fan; panel B
    padded canvas with extrapolated aponeuroses (end_linear dashed, global-linear dotted), fragments,
    central-3 extrapolations, streamlines and candidate fascicles from the deep insertion
    -> overlays/clip_XX.png
"""
from __future__ import annotations

import argparse
import math
import pickle
import resource
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from umud import geometry_v2 as g2  # noqa: E402
import build_candidates_v3 as b3  # noqa: E402

warnings.filterwarnings("ignore")
bc = b3.bc
OUT = Path(__import__("os").environ.get("TELEMED_OUT", ROOT / "artifacts/eval_v5/telemed"))  # override: other map sources
CACHE = Path(__import__("os").environ.get("TELEMED_CACHE", OUT / "cache"))
K0 = 1.067
PAS = np.arange(4.0, 40.01, 1.0)
CANDS = {
    "C2": ROOT / "artifacts/candidates/C2_ours_v2e_fusion_ensemble_shrink1.0.csv",
    "C7": ROOT / "artifacts/candidates_v4/C7_hyb_med_ens_pp_muscle_off.csv",
    "C8": ROOT / "artifacts/candidates_v4/C8_hyb_med_ens_pp_scanner.csv",
    "C10": ROOT / "artifacts/candidates_v4/diagnostic/D_C6pp_muscle_off_conf_ens_pp_muscle_off.csv",
    "C11": ROOT / "artifacts/candidates_v4/C11_fullshrink_cellmedian.csv",
}
C7_PA_OFFSET = 0.3286          # C7 - C9 on every row (rater-style half offset; FL unchanged)


def rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def tel_ids() -> list[str]:
    return [f"IMG_{n:05d}.tif" for n in range(56, 196)]


def clip_of(iid: str) -> int:
    return (int(iid[4:9]) - 56) // 5


# =========================================================================== extra constructions
def glob_curve(c: g2.EdgeCurve) -> g2.EdgeCurve:
    return g2.EdgeCurve(c.xd, c.yd, "poly2", "global_linear")


def _hit(curve, p0, d, Lmax):
    t = np.linspace(0, Lmax, 4000)
    f = (p0[1] + t * d[1]) - curve(p0[0] + t * d[0])
    k = np.flatnonzero(f <= 0)
    if not k.size or k[0] == 0:
        return float("nan")
    i = k[0]
    return float(t[i - 1] - f[i - 1] * (t[i] - t[i - 1]) / (f[i] - f[i - 1]))


def construct(sup, deep, x_mid, pa, sign, ref_slope=None, iters=5, Lmax=8000.0):
    """Straight fascicle at sign*pa to the deep reference (local tangent at the insertion when
    ref_slope is None, else the given global slope), inserted on the deep curve and running to the
    superficial curve, midpoint x at x_mid. Length in px."""
    xi, L = x_mid, float("nan")
    for _ in range(iters):
        yi = float(deep(np.array([xi]))[0])
        s = float(deep.slope(np.array([xi]))[0]) if ref_slope is None else ref_slope
        th = math.radians(math.degrees(math.atan(s)) + sign * pa)
        vx, vy = math.cos(th), math.sin(th)
        if vy > 0:
            vx, vy = -vx, -vy
        L = _hit(sup, (xi, yi - 1e-3), (vx, vy), Lmax)
        if not np.isfinite(L):
            return float("nan")
        xi += x_mid - (xi + 0.5 * L * vx)
    return L


def frag_fl(frs, sup, deep, s_lo, s_hi, W):
    """Straight-fragment FL (median) and central-3 FL with the given curves (as geometry_v2.measure)."""
    xgrid = np.linspace(-3 * W, 4 * W, 6001)
    out = []
    for q in frs:
        pd_ = g2._line_curve_x(q, deep, xgrid, "deep")
        ps_ = g2._line_curve_x(q, sup, xgrid, "sup")
        if pd_ is None or ps_ is None:
            continue
        L = math.hypot(pd_[0] - ps_[0], pd_[1] - ps_[1])
        out.append((L, max(0.0, 1 - q["length"] / max(L, 1e-6)), 0.5 * (pd_[0] + ps_[0])))
    if not out:
        return float("nan"), float("nan")
    A = np.array(out)
    st = float(np.median(A[:, 0]))
    span = s_hi - s_lo
    cen = A[(A[:, 2] >= s_lo + 0.2 * span) & (A[:, 2] <= s_hi - 0.2 * span)]
    if len(cen) < 3:
        cen = A
    cen = cen[np.argsort(cen[:, 1])][:3]
    return st, float(np.mean(cen[:, 0]))


def extra_one(res, A_shape) -> dict:
    cv = res.curves
    sup, deep = cv["sup_curve"], cv["deep_curve"]
    sg, dg = glob_curve(sup), glob_curve(deep)
    s_lo, s_hi, mm = cv["s_lo"], cv["s_hi"], cv["mm"]
    W = res.diag.get("W", A_shape[1])
    sign = b3.frag_sign(res)
    gsl, _ = g2._line_fit(deep.xd, deep.yd)
    r = dict(mt_x=res.mt_mm, sign=sign, deep_gslope=gsl, mm=mm)
    xs = [s_lo + f * (s_hi - s_lo) for f in (0.4, 0.5, 0.6)]
    for pa in PAS:
        tag = f"{int(pa):02d}"
        for nm, (S_, D_, ref) in {"cl": (sup, deep, None), "cg": (sup, deep, gsl),
                                 "ll": (sg, dg, None), "lg": (sg, dg, gsl)}.items():
            v = [construct(S_, D_, x, pa, sign, ref) for x in xs]
            v = [u for u in v if np.isfinite(u)]
            r[f"{nm}_{tag}"] = float(np.mean(v)) * mm if v else np.nan
    frs = res.fragments or []
    a, b = frag_fl(frs, sup, deep, s_lo, s_hi, W)
    c, d = frag_fl(frs, sg, dg, s_lo, s_hi, W)
    r.update(x_st_e=a * mm, x_c3_e=b * mm, x_st_g=c * mm, x_c3_g=d * mm)
    return r


def cmd_extra(args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        path = CACHE / f"extra_{ds}.csv"
        prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
        done = set(prev.image_id) if len(prev) else set()
        banks = b3.ea.Banks()
        rows, t0, k = [], time.time(), 0
        for iid, img, ppc, _ in bc.iterate(ds):
            if iid in done:
                continue
            A, up, lo, fbs, thr, edges = b3.apo_inputs(banks, ds, iid, "v2")
            F = banks.main.prob(ds, "ours_fasc384_noflip", iid)
            r = dict(image_id=iid)
            if A is not None and F is not None:
                cfg = b3.ea.make_cfg("v2e", thr, edges).with_(fasc_thr=0.95, compute_structure=False)
                im2 = img if img.shape == A.shape else cv2.resize(img, (A.shape[1], A.shape[0]),
                                                                  interpolation=cv2.INTER_AREA)
                try:
                    res = g2.measure_apo(A, F, im2, ppc, cfg, apo_upper=up, apo_lower=lo, fallbacks=fbs,
                                         keep_details=True)
                    if res.curves:
                        r.update(extra_one(res, A.shape))
                except Exception as e:  # keep going
                    r["error"] = f"{type(e).__name__}:{e}"
            rows.append(r)
            k += 1
            if k % 50 == 0:
                pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
                print(f"  {ds} {k} {time.time() - t0:.0f}s rss {rss():.2f} GB", flush=True)
        pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
        banks.close()
        print(f"done extra {ds}: {k} rows {time.time() - t0:.0f}s", flush=True)


# =========================================================================== pipeline heads (patch b3)
MY_FL = ("ll_pp", "cg_pp", "lg_pp", "mixll_pp")
_PATCHED = {}


def interp(B: pd.DataFrame, pre: str, pa: pd.Series) -> pd.Series:
    """Extra construction <pre> interpolated at a per-row PA (deg)."""
    cols = [f"x5_{pre}_{int(p):02d}" for p in PAS]
    G = B[cols].to_numpy(float)
    pv = pa.reindex(B.index).to_numpy(float)
    t = np.clip(np.nan_to_num(pv, nan=PAS[0]), PAS[0], PAS[-1]) - PAS[0]
    i0 = np.clip(np.floor(t).astype(int), 0, len(PAS) - 2)
    fr = t - i0
    r = np.arange(len(B))
    a, b = G[r, i0], G[r, i0 + 1]
    v = np.where(np.isfinite(a) & np.isfinite(b), a * (1 - fr) + b * fr, np.where(np.isfinite(a), a, b))
    v[~np.isfinite(pv)] = np.nan
    return pd.Series(v, index=B.index)


def fl_of(B, head, pa_c):
    c = b3._col
    if head in ("ll_pp", "cg_pp", "lg_pp"):
        return interp(B, head[:2], pa_c)
    if head == "mixll_pp":
        ens = b3.nanmean_w([c(B, "fl_straight"), c(B, "fl_central3"), b3.cons_at(B, pa_c)], [1, 1, 1])
        return b3.nanmean_w([ens, interp(B, "ll", pa_c)], [1, 1])
    raise ValueError(head)


def rel_shrink(x, grp, clip=None, srel=0.08):
    """Clip-level normal-normal shrinkage toward the group median, sigma = srel * x,
    tau^2 = max(var(x_u) - mean(sigma_u^2), 0.25 var(x_u)) per group (method of moments)."""
    x = x.astype(float)
    if clip is not None:
        cl = clip.reindex(x.index).fillna(-1).astype(int)
        unit = pd.Series(np.where(cl >= 0, "c" + cl.astype(str), ["i%d" % k for k in range(len(x))]), index=x.index)
    else:
        unit = pd.Series(["i%d" % k for k in range(len(x))], index=x.index)
    g = grp.reindex(x.index).astype(str)
    xu, gu = x.groupby(unit).median(), g.groupby(unit).first()
    out, wout = pd.Series(np.nan, index=x.index), pd.Series(np.nan, index=x.index)
    for G_ in gu.unique():
        v = xu[gu.index[gu == G_]].dropna()
        med = float(np.median(v)) if len(v) else np.nan
        s2 = (srel * v) ** 2
        var = float(np.var(v, ddof=1)) if len(v) > 2 else 0.0
        tau2 = max(var - float(s2.mean()), 0.25 * var) if len(v) > 2 else 0.0
        w = tau2 / (tau2 + s2) if tau2 > 0 else s2 * 0.0
        post = w * v + (1 - w) * med
        for uu in gu.index[gu == G_]:
            m = unit == uu
            out[m] = post.get(uu, med)
            wout[m] = w.get(uu, 0.0)
    return out, wout


def patch_b3() -> None:
    """In-process only: join cache/extra_<ds>.csv (prefix x5_) to the v3 caches, add FL heads rebuilt at
    the post-shrinkage PA (pa_deep_c) and two FL shrinkers (hyb_nofl, hyb_rel<s>)."""
    if _PATCHED:
        return
    o_load, o_post, o_apply, o_flv4 = b3._load_v3, b3.v4_post_group, b3.apply_shrink, b3.fl_head_v4

    def _load(ds, apo="v2", complete=True):
        d = o_load(ds, apo, complete)
        p = CACHE / f"extra_{ds}.csv"
        if d is not None and p.exists() and ds != "train142":
            x = pd.read_csv(p).drop_duplicates("image_id", keep="last").set_index("image_id")
            d = d.join(x.add_prefix("x5_"))
        return d

    def _flv4(d, head, pa):        # raw-PA placeholder; rebuilt at pa_deep_c in _post
        return b3.fl_head(d, "ens", pa) if head in MY_FL else o_flv4(d, head, pa)

    def _post(B, rec):
        if rec["fl"] in MY_FL:
            B = o_post(B, dict(rec, fl="ens_pp"))
            fl = fl_of(B, rec["fl"], B["pa_deep_c"])
            fl[b3._col(B, "mt_mm").isna()] = np.nan
            B["fl_mm"] = fl
            return B
        return o_post(B, rec)

    def _apply(D, shrink, model, ds=""):
        if shrink == "hyb_nofl":       # PA: confidence rule; FL unshrunk (already built at the shrunk PA)
            P = o_apply(D, "hyb_med", model, ds)
            P["fl_eb"], P["w_fl"] = D["fl_mm"], 1.0
            return P
        if shrink.startswith("hyb_rel"):
            P = o_apply(D, "hyb_med", model, ds)
            P["fl_eb"], P["w_fl"] = rel_shrink(D["fl_mm"], D["group"], D["clip"] if "clip" in D else None,
                                               float(shrink[7:]))
            return P
        return o_apply(D, shrink, model, ds)

    b3._load_v3, b3.v4_post_group, b3.apply_shrink, b3.fl_head_v4 = _load, _post, _apply, _flv4
    b3._MEMO.clear()
    _PATCHED["ok"] = True


RECIPES = {   # name: (FL head, FL shrink); PA = fusF + confidence rule everywhere (as C8)
    "C8": ("ens_pp", "hyb_med"), "ens_raw": ("ens_pp", "hyb_nofl"),
    "T1_ll_eb": ("ll_pp", "hyb_med"), "T2_ll_raw": ("ll_pp", "hyb_nofl"), "T3_ll_rel08": ("ll_pp", "hyb_rel0.08"),
    "T4_mix_eb": ("mixll_pp", "hyb_med"), "mix_raw": ("mixll_pp", "hyb_nofl"),
    "cg_raw": ("cg_pp", "hyb_nofl"), "lg_raw": ("lg_pp", "hyb_nofl"),
    "ll_rel04": ("ll_pp", "hyb_rel0.04"), "ll_rel12": ("ll_pp", "hyb_rel0.12"),
}


def rec_of(name):
    fl, shr = RECIPES[name]
    return b3.R("v2", "fusF", fl, shr, name=name)


# =========================================================================== classical apo (trainval)
def _ridge_map(g, sx, sy):
    G = cv2.GaussianBlur(g, (0, 0), sigmaX=sx, sigmaY=sy)
    R = np.clip(-cv2.Sobel(G, cv2.CV_32F, 0, 2, ksize=5), 0, None)
    return G, R / (np.percentile(R, 99.5) + 1e-6)


def detect_apos(img: np.ndarray, guide: np.ndarray, frac: float = 0.95, lam: float = 0.12, track: int = 6) -> dict:
    """GT-guided classical aponeurosis pair (no network): straight-line ridge search on the vertical
    second derivative of a horizontally smoothed image; deep = best ridge below >= frac of the guide
    pixels, sup = best ridge above them, both penalised by exp(-gap / (lam H)) from the guide envelope;
    ridge tracking +-track px, robust poly2, inner edge = ridge -+ median half-max half-thickness.
    Returns a synthetic band map A (0/255) usable by geometry_v2.measure (apo_thr 0.5, mask edges)."""
    g = img.astype(np.float32)
    H, W = g.shape
    vx0, vx1 = g2.estimate_viewport_x(img)
    gy, gx = np.nonzero(guide)
    sel = (gx >= vx0) & (gx < vx1)
    gy, gx = gy[sel], gx[sel]
    if gy.size < 50:
        return dict(ok=False, why="no_guide")
    if gy.size > 3000:
        k = np.random.default_rng(0).choice(gy.size, 3000, replace=False)
        gy, gx = gy[k], gx[k]
    G, R = _ridge_map(g, max(6.0, 0.015 * W), 2.0)
    xs = np.arange(vx0 + 4, vx1 - 4, 4)
    xc = 0.5 * (vx0 + vx1)
    angs = np.arange(-15.0, 15.01, 0.5)
    a_grid = np.arange(0, H, 1.0)
    S = np.zeros((len(angs), len(a_grid)), np.float32)
    for i, ang in enumerate(angs):
        Y = np.round(a_grid[:, None] + math.tan(math.radians(ang)) * (xs[None, :] - xc)).astype(int)
        ok = (Y >= 0) & (Y < H)
        v = np.where(ok, R[np.clip(Y, 0, H - 1), xs[None, :]], 0.0)
        S[i] = v.sum(1) / np.maximum(ok.sum(1), 1) * (ok.mean(1) > 0.6)
    m = 0.01 * H
    best = {}
    for side in ("deep", "sup"):
        bsc, bi, bj = -1.0, -1, -1
        for i, ang in enumerate(angs):
            c = np.sort(gy - math.tan(math.radians(ang)) * (gx - xc))
            fr = (np.searchsorted(c, a_grid - m) / c.size) if side == "deep" else \
                (1.0 - np.searchsorted(c, a_grid + m) / c.size)
            okk = fr >= frac
            if not okk.any():
                continue
            idx = np.flatnonzero(okk)
            a_edge = a_grid[idx[0]] if side == "deep" else a_grid[idx[-1]]
            adj = np.where(okk, S[i] * np.exp(-np.abs(a_grid - a_edge) / (lam * H)), -1.0)
            j = int(np.argmax(adj))
            if adj[j] > bsc:
                bsc, bi, bj = float(adj[j]), i, j
        best[side] = None if bi < 0 else dict(a=float(a_grid[bj]), ang=float(angs[bi]), score=float(S[bi, bj]))
    if best["deep"] is None:
        return dict(ok=False, why="no_deep")
    if best["sup"] is not None and best["sup"]["score"] < 0.30 * best["deep"]["score"]:
        best["sup"] = None
    xb = np.arange(vx0 + 6, vx1 - 6, 6)
    lines = {}
    for side in ("deep", "sup"):
        b = best.get(side)
        if b is None:
            continue
        t = math.tan(math.radians(b["ang"]))
        rx, ry = [], []
        for x in xb:
            y0 = b["a"] + t * (x - xc)
            lo, hi = int(max(0, y0 - track)), int(min(H - 1, y0 + track))
            if hi - lo < 3:
                continue
            col = R[lo:hi + 1, x]
            j = int(np.argmax(col))
            if col[j] > 0.05:
                rx.append(x)
                ry.append(lo + j)
        if len(rx) >= 8:
            rx, ry = np.array(rx, float), np.array(ry, float)
            lines[side] = dict(x=rx, ridge=ry, curve=g2.EdgeCurve(rx, ry, "poly2r", "end_linear"))
    if "deep" not in lines:
        return dict(ok=False, why="deep_refine")
    dc = lines["deep"]["curve"]
    sc_ = lines["sup"]["curve"] if "sup" in lines else None
    ys_top = sc_(xb) if sc_ is not None else np.zeros_like(xb, float)
    ys_bot = dc(xb)
    vals = [G[int(a + 0.2 * (b_ - a)):int(b_ - 0.2 * (b_ - a)), x] for a, b_, x in zip(ys_top, ys_bot, xb) if b_ - a > 10]
    base = float(np.median(np.concatenate(vals))) if vals else float(np.median(G))
    for side, L in lines.items():
        th = []
        for x, yr in zip(L["x"], L["ridge"]):
            x, y = int(x), int(round(yr))
            half, step, n = 0.5 * (G[y, x] + base), (-1 if side == "deep" else 1), 0
            while 0 < y < H - 1 and G[y, x] > half and n < 40:
                y += step
                n += 1
            th.append(n)
        hthk = float(np.clip(np.median(th), 1.0, 15.0))
        L["half_thick"] = hthk
        off = -hthk if side == "deep" else hthk
        L["inner_curve"] = g2.EdgeCurve(L["x"], L["curve"](L["x"]) + off, "poly2", "end_linear")
    A = np.zeros((H, W), np.uint8)
    xx = np.arange(vx0, vx1)
    for side, L in lines.items():
        yi = L["inner_curve"](xx)
        t2 = max(3.0, 2.0 * L["half_thick"])
        for x, y in zip(xx, yi):
            a, b_ = (int(round(y)), int(round(y + t2))) if side == "deep" else (int(round(y - t2)), int(round(y)))
            a, b_ = max(0, a), min(H - 1, b_)
            if b_ > a:
                A[a:b_ + 1, x] = 255
    pseudo = "sup" not in lines
    if pseudo:                    # superficial apo above the FOV: pseudo band at the top (MT/FL invalid)
        A[0:3, vx0:vx1] = 255
    return dict(ok=True, vx0=vx0, vx1=vx1, lines=lines, A=A, pseudo_sup=pseudo)


TV_PPC = {"1088x644": 126.1, "1340x810": 150.2 * 800 / 810, "1196x768": 150.2, "660x556": 77.4}
TV_FRAME = {"1088x644": "mask", "1340x810": "img", "1196x768": "mask", "660x556": "mask"}
# visual QC of the montages (viz/montage_*.png): deep line misplaced (inside the muscle / wrong slope)
TV_BAD = {"image_2429", "image_2430", "image_2431", "image_2434", "image_2435", "image_2436", "image_2437",
          "image_2438"}


def cmd_trainval(args) -> None:
    eo = b3.eo
    dims = pd.read_csv(bc.SCRATCH / "data-labels-geometry/out/dims.csv")
    dims["orig"] = dims.mask_w.astype(str) + "x" + dims.mask_h.astype(str)
    folds = pd.read_csv(ROOT / "artifacts/kaggle/fasc_orient_v1/metrics/folds.csv").set_index("stem")
    sel = dims[(dims.kind == "fasc") & dims.orig.isin(list(TV_PPC))].copy()
    sel["group"] = folds.reindex(sel.name).group.to_numpy()
    sel["rep"] = folds.reindex(sel.name).representative.to_numpy()
    sel = sel.sort_values("name").groupby(["orig", "group"]).head(args.per_group)
    (OUT / "viz").mkdir(parents=True, exist_ok=True)
    obank = g2.MaskBank(eo.OBANK_ROOT)
    path = OUT / "trainval_rows.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(prev.name) if len(prev) else set()
    base = g2.recommended_config(apo_thr=0.5, fasc_thr=0.5, compute_streamline=False, edge_extrap="end_linear",
                                 edge_source="mask")
    rows, t0 = [], time.time()
    print(sel.orig.value_counts().to_dict(), flush=True)
    for k, r in enumerate(sel.itertuples()):
        if r.name in done:
            continue
        img = bc.gray(bc.COMP / "fasc_imgs_v1/fasc_images_new_model_v1" / f"{r.name}.tif")
        Fg = bc.gray(bc.COMP / "fasc_masks_v1/fasc_masks_new_model_v1" / f"{r.name}.tif") > 0
        o = eo.load_orient(obank, "train_oof", r.rep)
        if o is None:
            continue
        if TV_FRAME[r.orig] == "mask":
            Hm, Wm = Fg.shape
            im = cv2.resize(img, (Wm, Hm), interpolation=cv2.INTER_AREA)
            o, F = eo.to_mask_frame(o, (Hm, Wm)), Fg
        else:                                   # GT mask into the 1200x800 test frame
            Hm, Wm = img.shape
            im = img
            F = cv2.resize(Fg.astype(np.uint8) * 255, (Wm, Hm), interpolation=cv2.INTER_AREA) > 60
            if o[1].shape != (Hm, Wm):
                o = eo.to_mask_frame(o, (Hm, Wm))
        ppc = TV_PPC[r.orig]
        det = detect_apos(im, F)
        meta = dict(name=r.name, orig=r.orig, group=r.group, rep=r.rep, ok=det.get("ok"),
                    pseudo_sup=det.get("pseudo_sup"), why=det.get("why", ""), qc_bad=r.name in TV_BAD)
        if not det.get("ok"):
            rows.append(dict(meta, run="oof"))
            continue
        A = det["A"].astype(np.float32) / 255.0
        gto = eo.gt_orient_from_mask(F)
        for run, (FF, orient, cfg) in {"gt": (F.astype(np.float32), gto, base.with_(compute_structure=False)),
                                       "oof": (o[0], o, base)}.items():
            try:
                res = g2.measure_field(A, FF, im, ppc, cfg, orient=orient, fcfg=b3.FCFG)
                rr = eo.row_of(res)
                if run == "oof":
                    rr.update(b3.constructed_grid(res, "deep"))
            except Exception as e:  # noqa
                rr = g2._nan_result(f"error:{type(e).__name__}:{e}").as_row()
            rows.append(dict(meta, run=run, **rr))
        seg, c2, s2 = g2.orient_maps(o)
        th, w = g2.field_theta_weight(seg, c2, s2)
        gs, gc, gsn = gto
        msk = gs > 0.5
        gth = 0.5 * np.degrees(np.arctan2(gsn[msk], gc[msk]))
        d_ = g2.wrap90(th[msk] - gth)
        ww = w[msk]
        rows[-1].update(pix_field_bias=float(np.sum(d_ * ww) / max(ww.sum(), 1e-9)),
                        pix_field_mad=float(np.median(np.abs(d_))), pix_gt_angle=float(np.median(gth)))
        vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        vis[F] = (0, 0, 255)
        for side, col in (("deep", (0, 140, 255)), ("sup", (0, 255, 255))):
            L = det["lines"].get(side)
            if L is not None:
                xx = np.arange(det["vx0"], det["vx1"])
                cv2.polylines(vis, [np.stack([xx, L["inner_curve"](xx)], 1).astype(np.int32).reshape(-1, 1, 2)],
                              False, col, 2)
        tb = cv2.resize(vis, (360, int(360 * vis.shape[0] / vis.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.putText(tb, f"{r.name} {r.orig}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(str(OUT / "viz" / f"tv_{r.name}.png"), tb)
        if k % 20 == 19:
            pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
            print(f"  trainval {k + 1}/{len(sel)} {time.time() - t0:.0f}s rss {rss():.2f} GB", flush=True)
    pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    obank.close()
    montages()
    trainval_report()


def montages(per=25) -> None:
    d = pd.read_csv(OUT / "trainval_rows.csv")
    for lay in TV_PPC:
        names = d[(d.orig == lay) & (d.run == "oof")].name.tolist()
        for s0 in range(0, len(names), per):
            ims = [cv2.imread(str(OUT / "viz" / f"tv_{n}.png")) for n in names[s0:s0 + per]]
            ims = [i for i in ims if i is not None]
            if not ims:
                continue
            h = max(i.shape[0] for i in ims)
            nr = (len(ims) + 4) // 5
            M = np.zeros((nr * h, 5 * 360, 3), np.uint8)
            for k, i in enumerate(ims):
                rr, cc = divmod(k, 5)
                M[rr * h:rr * h + i.shape[0], cc * 360:cc * 360 + i.shape[1]] = i
            cv2.imwrite(str(OUT / "viz" / f"montage_{lay}_{s0}.png"), M)


def trainval_report() -> None:
    d = pd.read_csv(OUT / "trainval_rows.csv")
    d = d[~d.name.isin(TV_BAD)]
    gt, oo = d[d.run == "gt"].set_index("name"), d[d.run == "oof"].set_index("name")
    ids = gt.index.intersection(oo.index)
    gt, oo = gt.loc[ids], oo.loc[ids]
    ok = gt.mt_mm.notna() & oo.mt_mm.notna()
    gt, oo = gt[ok], oo[ok]
    lay = oo.orig
    fusF = oo[["pa_fusion", "pa_fd_strip_g"]].mean(axis=1)
    H = {"frag_global": oo.pa_global, "frag_central3": oo.pa_central3, "structure": oo.pa_structure,
         "fusion(C1)": oo.pa_fusion, "field_strip_g": oo.pa_fd_strip_g, "field_ins": oo.pa_fd_ins,
         "field_comp_g": oo.pa_fd_comp_g, "fusF(C7 raw head)": fusF}
    Gr = {"gt_global": gt.pa_global, "gt_central3": gt.pa_central3, "gt_strip_g": gt.pa_fd_strip_g,
          "gt_ins": gt.pa_fd_ins, "gt_comp_g": gt.pa_fd_comp_g}
    rows = []
    for L in sorted(lay.unique()) + ["ALL_ARTUS"]:
        m = lay.isin(["1088x644", "1340x810", "1196x768"]) if L == "ALL_ARTUS" else (lay == L)
        grp = oo.group[m]
        for h, s in H.items():
            for gname, t in Gr.items():
                e = (s[m] - t[m]).dropna()
                if len(e) < 3:
                    continue
                eg = e.groupby(grp.reindex(e.index)).mean()
                rows.append(dict(layout=L, head=h, ref=gname, n=len(e), n_groups=len(eg), bias=e.mean(),
                                 bias_grp=eg.mean(), mae=e.abs().mean(), medae=e.abs().median()))
    pd.DataFrame(rows).to_csv(OUT / "trainval_pa_errors.csv", index=False)
    prof = pd.DataFrame({"layout": lay, "gt_global": gt.pa_global, "gt_comp": gt.pa_fd_comp_g,
                         "gt_strip": gt.pa_fd_strip_g, "gt_ins": gt.pa_fd_ins,
                         "pix_field_bias": oo.pix_field_bias, "pix_field_mad": oo.pix_field_mad})
    prof["strip_minus_global"] = prof.gt_strip - prof.gt_global
    prof["ins_minus_global"] = prof.gt_ins - prof.gt_global
    prof.to_csv(OUT / "trainval_pa_profile.csv")
    v = ~oo.pseudo_sup.astype(bool) & (lay != "1196x768")       # superficial apo visible -> FL defined
    X = pd.DataFrame({"layout": lay, "gt_straight": gt.fl_straight, "gt_pa": gt.pa_global, "mt_gt": gt.mt_mm,
                      "oof_straight": oo.fl_straight, "oof_c3": oo.fl_central3,
                      "oof_cons_fusF": b3.cons_at(oo, fusF)})[v]
    X["oof_ens_fusF"] = X[["oof_straight", "oof_c3", "oof_cons_fusF"]].mean(axis=1)
    X["k_gt"] = X.gt_straight * np.sin(np.radians(X.gt_pa)) / X.mt_gt
    X.to_csv(OUT / "trainval_fl.csv")
    print(pd.DataFrame(rows).query("ref in ['gt_global','gt_strip_g','gt_ins'] and head=='fusF(C7 raw head)'")
          [["layout", "ref", "n", "n_groups", "bias_grp", "mae"]].round(3).to_string(index=False))
    print(prof.groupby("layout")[["gt_global", "gt_strip", "gt_ins", "strip_minus_global", "ins_minus_global",
                                  "pix_field_bias"]].median().round(2))
    print(X.groupby("layout")[["gt_straight", "oof_straight", "oof_ens_fusF", "k_gt"]].median().round(3))


# =========================================================================== frames
def cmd_frames(args) -> None:
    d = b3.load_v3("test", "v2")
    n = pd.Series(d.index.str[4:9].astype(int), index=d.index)
    t = d[n.between(56, 195)].copy()
    t["clip"] = (n[t.index] - 56) // 5
    t["pa_fusF"] = b3.pa_head(t, "fusF")
    t["fl_ens_fusF"] = b3.fl_head(t, "ens", t.pa_fusF)
    t["fl_cons_fusF"] = b3.cons_at(t, t.pa_fusF)
    t["fl_cpl_fusF"] = K0 * t.mt_mm / np.sin(np.radians(t.pa_fusF))
    est = ["mt_mm", "pa_global", "pa_local", "pa_central3", "pa_structure", "pa_fusion", "pa_fd_strip_g", "pa_fd_ins",
           "pa_fd_comp_g", "pa_sl_ins", "pa_fusF", "fl_straight", "fl_central3", "fl_constructed", "fl_sl_med",
           "fl_ensemble", "fl_ens_fusF", "fl_cons_fusF", "fl_cpl_fusF", "d_deep_angle", "d_sup_angle", "confidence"]
    g = t.groupby("clip")
    sd, rng, med = g[est].std(ddof=1), g[est].agg(lambda x: x.max() - x.min()), g[est].median()
    rel = sd.copy()
    for c in est:
        if c.startswith("fl"):
            rel[c] = sd[c] / med[c]
    pd.concat({"median": med, "sd": sd, "range": rng}, axis=1).to_csv(OUT / "frame_spread.csv")
    S = pd.DataFrame({"median_sd": sd.median(), "mean_sd": sd.mean(), "max_sd": sd.max(),
                      "median_range": rng.median(), "median_rel_sd": rel.median()})
    S.to_csv(OUT / "frame_spread_summary.csv")
    print(S.round(3).to_string())


# =========================================================================== bench
def _boot(a, b, cl=None, block=0, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = (a - b).dropna()
    if cl is not None:
        g = cl.reindex(d.index)
        m, n = d.groupby(g).sum(), d.groupby(g).size()
        u = m.index.to_numpy()
        st = []
        for _ in range(B):
            k = rng.choice(u, len(u))
            st.append(m.loc[k].sum() / n.loc[k].sum())
    elif block:
        v, L = d.to_numpy(), len(d)
        st = []
        for _ in range(B):
            idx = np.concatenate([np.arange(s, s + block) % L for s in rng.integers(0, L, L // block + 1)])[:L]
            st.append(v[idx].mean())
    else:
        v = d.to_numpy()
        st = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def fl_model_table() -> pd.DataFrame:
    """Expert FL / model on static (mean7), VL, dynamic: model at the EXPERT PA (T) and at our fusF PA (F)."""
    G, V, D = bc.static_gts()["mean7"], bc.vl_gt(), bc.dynamic_gt()
    rows = []
    for ds, gt in (("static", G), ("vl", V), ("dynamic", D)):
        c = b3.load_v3(ds, "v2")
        g = gt.reindex(c.index)
        pa_t = g.pa_deg + ((c.pa_fusion - c.pa_fusion_h) if ds == "dynamic" else 0.0)  # dynamic: vs horizontal
        fusF = b3.pa_head(c, "fusF")
        X = pd.DataFrame(index=c.index)
        X["ds"], X["fl_t"], X["mt"] = ds, g.fl_mm, c.mt_mm
        X["fov"] = (c.d_shared_hi - c.d_shared_lo) * c.x5_mm
        for tag, p in (("T", pa_t), ("F", fusF)):
            for pre in ("cl", "cg", "ll", "lg"):
                X[f"{pre}{tag}"] = interp(c, pre, p)
            X[f"cpl{tag}"] = c.mt_mm / np.sin(np.radians(p))
        for k in ("x_st_e", "x_c3_e", "x_st_g", "x_c3_g"):
            X[k] = c[f"x5_{k}"]
        X["ens_e"] = X[["x_st_e", "x_c3_e", "clF"]].mean(axis=1)
        rows.append(X)
    X = pd.concat(rows)
    return X[X.fl_t.notna() & (X.mt > 5)]


def cmd_bench(args) -> None:
    patch_b3()
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    # (a) FL-given-PA calibration
    X = fl_model_table()
    X.to_csv(OUT / "bench_fl_models_per_image.csv")
    cols = ["clT", "cgT", "llT", "lgT", "cplT", "clF", "cgF", "llF", "lgF", "cplF", "x_st_e", "x_c3_e", "ens_e"]
    X["bin"] = np.where(X.ds == "vl", pd.cut(X.fl_t / X.fov, [0, 1.5, 2.0, 2.5, 9]).astype(str), "all")
    rows = []
    for (ds, bn), x in X.groupby(["ds", "bin"]):
        r = dict(ds=ds, fl_fov_bin=bn, n=len(x))
        for c in cols:
            lr = np.log(x.fl_t / x[c]).dropna()
            r[f"{c}_ratio"], r[f"{c}_mabslog"] = float(np.exp(lr.median())), float(np.abs(lr - 0 * lr).mean())
        rows.append(r)
    for ds, x in X.groupby("ds"):
        r = dict(ds=ds, fl_fov_bin="ALL", n=len(x))
        for c in cols:
            lr = np.log(x.fl_t / x[c]).dropna()
            r[f"{c}_ratio"], r[f"{c}_mabslog"] = float(np.exp(lr.median())), float(np.abs(lr).mean())
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "bench_fl_models.csv", index=False)
    show = ["ds", "fl_fov_bin", "n"] + [f"{c}_ratio" for c in ("clT", "cgT", "llT", "cplT", "llF", "ens_e")] + \
        [f"{c}_mabslog" for c in ("llF", "clF", "cgF", "cplF", "x_st_e", "ens_e")]
    print("(a) expert FL / model (median ratio) and mean |log error|; T = at expert PA, F = at our fusF PA")
    print(T[show].round(3).to_string(index=False))
    # (b) full pipeline grid
    rows, PI = [], {}
    for nm in RECIPES:
        res = b3.run_bench_v4(rec_of(nm), "lodo")
        sc = b3.v4_score(res)
        PI[nm] = b3.v4_per_image(res)
        wmean = {}
        for key in (("lodo4", "static"), ("lodo3", "vl"), ("lodo4", "dynamic")):
            wmean[f"w_fl_{key[1]}"] = float(np.nanmean(res[key]["w_fl"]))
        rows.append(dict(name=nm, fl=RECIPES[nm][0], shrink=RECIPES[nm][1], **sc, **wmean))
    T = pd.DataFrame(rows)
    T["S_part"] = (0.35 * T["2r_c10"] + 0.35 * T["vl3"] + 0.15 * T["dyn"]) / 0.85
    T.to_csv(OUT / "bench_pipeline.csv", index=False)
    show = ["name", "s35", "s35_fl", "c10", "2r", "2r_c10", "2r_fl", "R7", "vl3", "vl3_fl", "vl4", "vl4_fl",
            "vl_young3", "dyn_fl", "dyn", "S_part", "w_fl_static", "w_fl_vl", "w_fl_dynamic"]
    print("\n(b) full pipeline (LODO constants; lower is better; S_part = (0.35 c10-2r + 0.35 VL3 + 0.15 dyn)/0.85)")
    print(T[show].round(4).to_string(index=False))
    # (c) paired bootstraps vs C8
    base = PI["C8"]
    out = []
    for nm in RECIPES:
        if nm == "C8":
            continue
        p = PI[nm]
        st, sb = p["static"], base["static"]
        r = dict(name=nm)
        for lab, (a, b, kw) in {
                "c10_2r": (st.loc[st.clean, "e2r_S"], sb.loc[sb.clean, "e2r_S"], {}),
                "all35_2r": (st["e2r_S"], sb["e2r_S"], {}),
                "static_fl_2r": (st["e2r_fl_mm"], sb["e2r_fl_mm"], {}),
                "vl3": (p["vl3"]["e_S"], base["vl3"]["e_S"], dict(cl=p["vl3"]["subject"])),
                "vl3_fl": (p["vl3"]["e_fl_mm"], base["vl3"]["e_fl_mm"], dict(cl=p["vl3"]["subject"])),
                "vl4_fl": (p["vl4"]["e_fl_mm"], base["vl4"]["e_fl_mm"], dict(cl=p["vl4"]["subject"])),
                "dyn_fl": (p["dyn"]["e_fl_mm"], base["dyn"]["e_fl_mm"], dict(block=17))}.items():
            m, lo, hi = _boot(a, b, **kw)
            r[lab], r[lab + "_lo"], r[lab + "_hi"] = m, lo, hi
        out.append(r)
    O = pd.DataFrame(out)
    O.to_csv(OUT / "bench_boot_vs_C8.csv", index=False)
    fmt = O[["name"]].copy()
    for lab in ("c10_2r", "all35_2r", "static_fl_2r", "vl3", "vl3_fl", "vl4_fl", "dyn_fl"):
        fmt[lab] = [f"{a:+.4f} [{b:+.4f},{c:+.4f}]" for a, b, c in zip(O[lab], O[lab + "_lo"], O[lab + "_hi"])]
    print("\n(c) recipe - C8 (negative = better), bootstrap 95% CI (static by image, VL by subject, dyn 17-frame blocks)")
    print(fmt.to_string(index=False))
    with open(CACHE / "bench_per_image.pkl", "wb") as f:
        pickle.dump(PI, f)


# Visual review of overlays/clip_XX.png (frame 0 of each clip; reviewer judgement, +-2 deg at best).
VISUAL = {
    0: "deep rises 2.5 deg, sup flat; striations ~10-13 deg vs deep near insertion; C7 PA plausible; FL 95% extrapolated",
    1: "deep strongly curved (-6 deg); long streak ~14 deg vs deep at right; FL very sensitive to apo extrapolation",
    2: "thin, converging; clear streaks ~15-17 deg; C7 PA ok or ~1 deg low; FL ~74 geometric",
    3: "second muscle below; faint superficial deep apo correctly chosen; PA ~12-14",
    4: "strongly converging (sup -7.4 deg); fascicles ~10-12 deg; straight geometry 74 mm vs C7 89",
    5: "converging; fascicles ~10-14 deg, steeper near deep at right; geometry 87 vs C7 96",
    6: "hypoechoic, few streaks ~14-15 deg near deep; C7 PA maybe ~1 deg low",
    7: "long bright streaks ~14-18 deg; parallel apos; C7 PA ok",
    8: "deep rises 4.8 deg; clear; C7 PA ok",
    9: "deep curved at left; blob texture ~12-14 deg; T1 FL ends 12% short of sup apo, T2 consistent",
    10: "very clear straight parallel fascicles ~11-12 deg; C7 PA good; geometry 125 vs C7 120",
    11: "clear ~13-14 deg; C7 good",
    12: "clear long fascicles ~13-15 deg, second muscle below (pair correct); C11 FL 111 far too long",
    13: "clear ~10-12 deg; C7 good",
    14: "deep rises 3 deg; streaks ~13-15 deg; C7 PA maybe ~1 deg low",
    15: "clear ~13-14 deg; C7 good",
    16: "clear, ~15-16 deg near deep; C7 ok",
    17: "very clear straight parallel fascicles ~14 deg; C7 PA exact; geometry 112 vs C7 117",
    18: "converging; ~11-13 deg; geometry ~80 vs C7 88",
    19: "thin; long flat bright line ~9 deg (septum or fascicle?) vs steeper fibres 15-17 deg near deep: ambiguous",
    20: "thin converging; ~12 deg; geometry 74 vs C7 84",
    21: "thin; long straight lines ~11 deg across FOV; visual FL ~76-80 -> T1 81 consistent, C7 84",
    22: "converging; oblique septum-like line ~20 deg low in the muscle; geometry ~80, C7 87 (k_g 1.02)",
    23: "streaks ~13-16 deg; deep flat; C7 ok",
    24: "deep curved/rising; blobs ~14 deg; geometry 84 vs C7 89",
    25: "blobs ~12-14 deg; geometry 107; T1 98 short (k_g 0.92)",
    26: "partial deep apo (shared 24%), low confidence; fascicles ~7-10 deg to horizontal; PA uncertain",
    27: "deep descends 6.5 deg; long near-horizontal streaks -> ~13-14 deg vs deep; C7 ok",
}

# =========================================================================== build
BUILD = ("T1_ll_eb", "T2_ll_raw", "T3_ll_rel08", "T4_mix_eb")
MAIN = "T1_ll_eb"


def cmd_build(args) -> None:
    patch_b3()
    c7 = pd.read_csv(CANDS["C7"]).set_index("image_id")
    S = {k: pd.read_csv(p).set_index("image_id") for k, p in CANDS.items()}
    tel = tel_ids()
    T = pd.DataFrame(index=pd.Index(tel, name="image_id"))
    T["clip"] = [clip_of(i) for i in tel]
    lay = b3.layout()
    T["group"] = lay.loc[tel, "group"]
    for k, s in S.items():
        T[f"{k}_pa"], T[f"{k}_fl"], T[f"{k}_mt"] = s.loc[tel, "pa_deg"], s.loc[tel, "fl_mm"], s.loc[tel, "mt_mm"]
    P8 = None
    for nm in BUILD:
        sub, P, _ = b3.test_sub_v4(rec_of(nm), return_per_image=True)
        sub = sub.set_index("image_id") if "image_id" in sub else sub
        if not np.allclose(sub.loc[tel, "pa_deg"] + C7_PA_OFFSET, c7.loc[tel, "pa_deg"], atol=2e-3):
            raise RuntimeError(f"{nm}: PA does not match C7 - offset on Telemed rows")
        T[f"{nm}_fl"], T[f"{nm}_w"] = sub.loc[tel, "fl_mm"], P.loc[tel, "w_fl"]
        if P8 is None:
            P8 = P
    # raw per-image ingredients (deep-reference PA after the confidence rule, raw heads)
    d = b3.load_v3("test", "v2").loc[tel]
    pa_c = P8.loc[tel, "pa_deep_c"]
    T["pa_deep_raw"], T["pa_deep_c"], T["confidence"] = P8.loc[tel, "pa_deep_raw"], pa_c, d.confidence
    T["mt_raw"] = d.mt_mm
    for c in ("pa_global", "pa_central3", "pa_structure", "pa_fusion", "pa_fd_strip_g", "pa_fd_ins", "pa_fd_comp_g",
              "pa_sl_ins", "fl_straight", "fl_central3", "fl_constructed", "fl_sl_med", "d_extrap_med",
              "d_deep_angle", "d_sup_angle"):
        T[c] = d[c]
    for pre in ("cl", "ll", "cg", "lg"):
        T[f"geo_{pre}_at_pa_c"] = interp(d, pre, pa_c)
    T["cpl_at_pa_c"] = K0 * d.mt_mm / np.sin(np.radians(pa_c))
    T["fov_mm"] = (d.d_shared_hi - d.d_shared_lo) * d.x5_mm
    T["ens_pp_raw"] = b3.nanmean_w([d.fl_straight, d.fl_central3, b3.cons_at(d, pa_c)], [1, 1, 1])
    for nm in ["C7"] + list(BUILD):
        fl = T[f"{nm}_fl"]
        T[f"k_{nm}"] = fl * np.sin(np.radians(T["C7_pa"])) / T["C7_mt"]
    T.to_csv(OUT / "telemed_per_image.csv")
    # candidate-ready files: PA = C7 (unchanged), FL = recipe
    for nm in BUILD:
        o = pd.DataFrame({"image_id": tel, "pa_deg": T["C7_pa"].to_numpy(), "fl_mm": T[f"{nm}_fl"].to_numpy()})
        o.to_csv(OUT / f"telemed_{nm}.csv", index=False)
    # per-clip table
    num = T.drop(columns=["group"]).groupby("clip").median()
    num.insert(0, "group", T.groupby("clip").group.first().str.replace("telemed_artus_", ""))
    num["frame_sd_fl_ll"] = T.groupby("clip")["geo_ll_at_pa_c"].std()
    num["frame_sd_fl_C7raw"] = T.groupby("clip")["ens_pp_raw"].std()
    num["geo_spread_rel"] = num[[f"geo_{p}_at_pa_c" for p in ("cl", "ll", "cg", "lg")]].std(axis=1) / num.geo_ll_at_pa_c
    num["dFL_T1_minus_C7"] = num[f"{MAIN}_fl"] - num["C7_fl"]
    num["stake_T1_official"] = (num["dFL_T1_minus_C7"].abs() / 12.0) / 3.0      # per row, if fully right/wrong
    fs = OUT / "frame_spread.csv"
    if fs.exists():
        F = pd.read_csv(fs, header=[0, 1], index_col=0)
        for c in ("pa_fusF", "pa_fd_strip_g", "pa_fd_ins", "pa_global", "fl_straight", "fl_central3", "fl_cons_fusF",
                  "fl_cpl_fusF", "mt_mm"):
            num[f"frame_sd_{c}"] = F[("sd", c)].reindex(num.index)
    ov = OUT / "overlay_rows_f0.csv"
    if ov.exists():
        O = pd.read_csv(ov).set_index("clip")
        for nm in ("C2", "C7", "C11", "T1", "T2raw", "straight", "coupling"):
            if f"{nm}_k_globlin" in O:
                num[f"kgeom_{nm}"] = O[f"{nm}_k_globlin"].reindex(num.index)
    num["visual_note"] = pd.Series(VISUAL).reindex(num.index)
    num.to_csv(OUT / "telemed_per_clip.csv")
    # distances (whole test set: C7 with the Telemed FL replaced)
    dist = {}
    for nm in BUILD:
        full = c7.copy()
        full.loc[tel, "fl_mm"] = T[f"{nm}_fl"].to_numpy()
        dist[nm] = bc.official_distance(full, c7)
        for ref in ("C2", "C10"):
            dist[f"{nm}_vs_{ref}"] = bc.official_distance(full, S[ref])["S"]
    pd.DataFrame(dist).T.to_csv(OUT / "telemed_distances.csv")
    pd.set_option("display.width", 320)
    show = ["group", "C7_mt", "C7_pa", "pa_deep_raw", "pa_fusion", "pa_fd_strip_g", "pa_fd_ins", "fl_straight",
            "fl_central3", "ens_pp_raw", "geo_cl_at_pa_c", "geo_ll_at_pa_c", "cpl_at_pa_c", "C2_fl", "C7_fl",
            "C11_fl", "T1_ll_eb_fl", "T1_ll_eb_w", "T2_ll_raw_fl", "T4_mix_eb_fl", "dFL_T1_minus_C7"]
    print(num[show].round(1).to_string())
    print({k: v for k, v in dist.items()})


# =========================================================================== overlay
def measure_full(banks, obank, ds: str, iid: str, img: np.ndarray, ppc: float):
    """Exact build_candidates_v3.measure_one path, returning the live GeoResult + field paths."""
    A, up, lo, fbs, thr, edges = b3.apo_inputs(banks, ds, iid, "v2")
    F = banks.main.prob(ds, "ours_fasc384_noflip", iid)
    orient = b3.eo.load_orient(obank, ds, Path(iid).stem)
    cfg = b3.ea.make_cfg("v2e", thr, edges).with_(fasc_thr=0.95)
    im2 = img if img.shape == A.shape else cv2.resize(img, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
    res = g2.measure_apo(A, F, im2, ppc, cfg, apo_upper=up, apo_lower=lo, fallbacks=fbs, keep_details=True,
                         orient=orient)
    out = g2.field_architecture(res, orient, b3.FCFG, cfg.pa_reference)
    res.variants.update(out["variants"])
    res.diag.update(out["diag"])
    return res, out.get("paths", []), im2, A, orient


def up_dir(deep, x: float, pa: float, sign: float) -> tuple[float, float]:
    ref = math.degrees(math.atan(float(deep.slope(np.array([x]))[0])))
    th = math.radians(ref + sign * pa)
    vx, vy = math.cos(th), math.sin(th)
    if vy > 0:
        vx, vy = -vx, -vy
    return vx, vy


def fasc_from_mid(deep, x_mid: float, pa: float, fl_px: float, sign: float, iters: int = 6):
    xi = x_mid
    for _ in range(iters):
        vx, vy = up_dir(deep, xi, pa, sign)
        xi = x_mid - 0.5 * fl_px * vx
    yi = float(deep(np.array([xi]))[0])
    vx, vy = up_dir(deep, xi, pa, sign)
    return (xi, yi), (xi + fl_px * vx, yi + fl_px * vy)


def sup_hit(sup, p0, d, max_len: float) -> float:
    t = np.linspace(0, max_len, 4000)
    f = p0[1] + t * d[1] - sup(p0[0] + t * d[0])
    k = np.flatnonzero(f <= 0)
    return float(t[k[0]]) if k.size else float("nan")


COLS = {  # BGR
    "straight": (0, 0, 255), "constructed": (255, 0, 255), "streamline": (0, 255, 0), "coupling": (0, 165, 255),
    "C2": (200, 200, 200), "C7": (255, 255, 255), "C11": (120, 120, 255), "T1": (0, 255, 255), "T2raw": (255, 128, 0),
}


def put(vis, txt, x, y, col=(255, 255, 255), s=0.55, th=1):
    cv2.putText(vis, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(vis, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)


def central3_frags(res, n=3):
    cv = res.curves
    s_lo, s_hi = cv["s_lo"], cv["s_hi"]
    fl_frs = [q for q in res.fragments if np.isfinite(q.get("fl_px", np.nan))]
    span = s_hi - s_lo
    cen = [q for q in fl_frs if s_lo + 0.2 * span <= q["mid_x"] <= s_hi - 0.2 * span]
    if len(cen) < n:
        cen = fl_frs
    return sorted(cen, key=lambda q: q["extrap"])[:n]


def render_clip(banks, obank, lay, S, TT, c: int, frame: int = 0) -> dict:
    iid = f"IMG_{56 + 5 * c + frame:05d}.tif"
    img = bc.gray(bc.TEST_DIR / iid)
    ppc = float(lay.loc[iid, "px_per_cm"])
    res, paths, im2, A, orient = measure_full(banks, obank, "test", iid, img, ppc)
    v, cv = res.variants, res.curves
    sup, deep = cv["sup_curve"], cv["deep_curve"]
    sgl, dgl = glob_curve(sup), glob_curve(deep)
    s_lo, s_hi, mt_px, mm = cv["s_lo"], cv["s_hi"], cv["mt_px"], cv["mm"]
    H, W = im2.shape
    ppmm = 1.0 / mm
    sign = b3.frag_sign(res)
    x_c = 0.5 * (s_lo + s_hi)
    mt = res.mt_mm
    pa_fus, pa_fld = v.get("pa_fusion", np.nan), v.get("pa_fd_strip_g", np.nan)
    pa_fusF = np.nanmean([pa_fus, pa_fld])
    cands = [("straight", v.get("pa_global", np.nan), v.get("fl_straight", np.nan)),
             ("constructed", pa_fus, v.get("fl_constructed", np.nan)),
             ("streamline", v.get("pa_sl_ins", np.nan), v.get("fl_sl_med", np.nan)),
             ("coupling", pa_fusF, K0 * mt / math.sin(math.radians(pa_fusF)))]
    for k in ("C2", "C7", "C11"):
        cands.append((k, float(S[k].loc[iid, "pa_deg"]), float(S[k].loc[iid, "fl_mm"])))
    if TT is not None and iid in TT.index:      # FL of T1/T2 is built at pa_deep_c (C7 PA minus offset)
        cands.append(("T1", float(TT.loc[iid, "pa_deep_c"]), float(TT.loc[iid, f"{MAIN}_fl"])))
        cands.append(("T2raw", float(TT.loc[iid, "pa_deep_c"]), float(TT.loc[iid, "T2_ll_raw_fl"])))
    padx, pady = int(1.05 * W), int(0.30 * H)
    CW, CH = W + 2 * padx, H + 2 * pady
    canvas = np.full((CH, CW, 3), 40, np.uint8)
    enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(im2)
    canvas[pady:pady + H, padx:padx + W] = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
    off = np.array([padx, pady], float)

    def P(p):
        return (int(round(p[0] + off[0])), int(round(p[1] + off[1])))

    xs = np.linspace(-padx, W + padx, 600)
    for crv, cg_, col in ((sup, sgl, (0, 255, 255)), (deep, dgl, (0, 140, 255))):
        ys, yg = crv(xs), cg_(xs)
        inside = (xs >= crv.x0) & (xs <= crv.x1)
        pts = np.stack([xs + off[0], ys + off[1]], 1).astype(np.int32)
        ptg = np.stack([xs + off[0], yg + off[1]], 1).astype(np.int32)
        for i in range(len(xs) - 1):
            if inside[i] or (i // 6) % 2 == 0:
                cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), col, 2 if inside[i] else 1)
            if not inside[i] and i % 4 == 0:      # global-linear extrapolation: dotted
                cv2.circle(canvas, tuple(ptg[i]), 1, (255, 255, 255), -1)
    for q in res.fragments or []:
        cv2.line(canvas, P(q["p0"]), P(q["p1"]), (0, 0, 255), 2)
    for q in central3_frags(res):
        if q.get("p_deep") is not None and q.get("p_sup") is not None:
            cv2.line(canvas, P(q["p_deep"]), P(q["p_sup"]), (255, 255, 0), 1, cv2.LINE_AA)
    for pth in paths:
        cv2.polylines(canvas, [(np.asarray(pth, float) + off).astype(np.int32).reshape(-1, 1, 2)], False,
                      (0, 255, 0), 1, cv2.LINE_AA)
    ends = {}
    for j, (nm, pa, fl) in enumerate(cands):
        if not (np.isfinite(pa) and np.isfinite(fl)):
            continue
        dref = dgl if nm in ("T1", "T2raw") else deep       # T1/T2 FL is built in the global-linear geometry
        p0, p1 = fasc_from_mid(dref, x_c + (j - 4) * 0.015 * W, pa, fl * ppmm, sign)
        d = ((p1[0] - p0[0]) / (fl * ppmm), (p1[1] - p0[1]) / (fl * ppmm))
        Le, Lg = sup_hit(sup, p0, d, 3 * fl * ppmm), sup_hit(sgl, p0, d, 3 * fl * ppmm)
        ends[nm] = dict(pa=pa, fl=fl, k_e=fl * ppmm / Le if np.isfinite(Le) else np.nan,
                        k_g=fl * ppmm / Lg if np.isfinite(Lg) else np.nan)
        col = COLS[nm]
        cv2.line(canvas, P(p0), P(p1), col, 3 if nm in ("C7", "T1") else 2, cv2.LINE_AA)
        cv2.circle(canvas, P(p1), 7, col, -1)
        cv2.circle(canvas, P(p0), 4, col, -1)
    cv2.rectangle(canvas, (padx, pady), (padx + W - 1, pady + H - 1), (90, 90, 90), 1)
    B = cv2.resize(canvas, (1800, int(CH * 1800.0 / CW)), interpolation=cv2.INTER_AREA)
    y = 22
    put(B, f"clip {c:02d} {iid} {lay.loc[iid, 'group']} ppc {ppc:.1f}  MT {mt:.1f} mm  conf {res.confidence:.2f}  "
           f"extrap_med {res.diag.get('extrap_med', np.nan):.2f}  deep ang {res.diag.get('deep_angle', np.nan):.1f} "
           f"sup ang {res.diag.get('sup_angle', np.nan):.1f}", 8, y, (255, 255, 255), 0.6)
    y += 24
    put(B, f"raw PA: global {v.get('pa_global', np.nan):.1f} c3 {v.get('pa_central3', np.nan):.1f} struct "
           f"{v.get('pa_structure', np.nan):.1f} fusion {pa_fus:.1f} field-strip {pa_fld:.1f} ins "
           f"{v.get('pa_fd_ins', np.nan):.1f} comp {v.get('pa_fd_comp_g', np.nan):.1f}", 8, y, (255, 255, 255), 0.55)
    for nm, e in ends.items():
        y += 21
        put(B, f"{nm:11s} PA {e['pa']:5.1f}  FL {e['fl']:6.1f}  FL/straight-to-sup end_lin {e['k_e']:.2f} "
               f"glob_lin {e['k_g']:.2f}", 8, y, COLS[nm], 0.52)
    x0 = int(max(0, min(sup.x0, deep.x0) - 10))
    x1 = int(min(W, max(sup.x1, deep.x1) + 10))
    yy = np.linspace(x0, x1, 50)
    y0 = int(max(0, np.nanmin(sup(yy)) - 50))
    y1 = int(min(H, np.nanmax(deep(yy)) + 50))
    Acr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
    fan, clean = Acr.copy(), Acr.copy()
    for crv, col in ((sup, (0, 255, 255)), (deep, (0, 140, 255))):
        xx = np.arange(int(crv.x0), int(crv.x1))
        cv2.polylines(fan, [np.stack([xx, crv(xx)], 1).astype(np.int32).reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
        for xa in np.arange(int(crv.x0), int(crv.x1), 12):
            cv2.line(clean, (int(xa), int(crv(np.array([xa]))[0])), (int(xa + 6), int(crv(np.array([xa + 6]))[0])),
                     col, 1, cv2.LINE_AA)
    span = s_hi - s_lo
    vx_test, _ = up_dir(deep, x_c, 12.0, sign)
    xins = [s_lo + 0.15 * span, s_lo + 0.50 * span] if vx_test > 0 else [s_hi - 0.15 * span, s_hi - 0.50 * span]
    for xi in xins:
        yi = float(deep(np.array([xi]))[0])
        for a, cf in zip((8, 11, 14, 17, 20), [(255, 80, 80), (255, 200, 0), (0, 255, 0), (0, 200, 255), (200, 0, 255)]):
            vx, vy = up_dir(deep, xi, a, sign)
            L = min(0.8 * mt_px / math.sin(math.radians(a)), 0.45 * span)
            p1 = (xi + L * vx, yi + L * vy)
            cv2.line(fan, (int(xi), int(yi)), (int(p1[0]), int(p1[1])), cf, 1, cv2.LINE_AA)
            put(fan, f"{a}", p1[0] + 3, p1[1], cf, 0.45)
    fanc = cv2.addWeighted(fan, 0.85, Acr, 0.15, 0)[y0:y1, x0:x1]
    clean = clean[y0:y1, x0:x1]
    scA = min(1.0, 897 / fanc.shape[1])
    if scA < 1.0:
        sz = (int(fanc.shape[1] * scA), int(fanc.shape[0] * scA))
        fanc, clean = cv2.resize(fanc, sz, interpolation=cv2.INTER_AREA), cv2.resize(clean, sz, interpolation=cv2.INTER_AREA)
    padA = np.full((fanc.shape[0], 1800, 3), 30, np.uint8)
    padA[:, :clean.shape[1]] = clean
    padA[:, 903:903 + fanc.shape[1]] = fanc
    put(padA, f"A: clip {c:02d} native crop (left clean; right rays 8/11/14/17/20 deg vs local deep tangent)", 6, 18,
        (255, 255, 255), 0.5)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "overlays" / f"clip_{c:02d}.png"), np.vstack([padA, np.zeros((6, 1800, 3), np.uint8), B]),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    row = dict(clip=c, image_id=iid, mt=mt, conf=res.confidence, sign=sign)
    for nm, e in ends.items():
        row.update({f"{nm}_pa": e["pa"], f"{nm}_fl": e["fl"], f"{nm}_k_endlin": e["k_e"], f"{nm}_k_globlin": e["k_g"]})
    del A, orient, img, im2, canvas, B, enh, fan, clean
    return row


def cmd_overlay(args) -> None:
    banks = b3.ea.Banks()
    obank = g2.MaskBank(b3.eo.OBANK_ROOT)
    lay = bc.test_layout()
    S = {k: pd.read_csv(p).set_index("image_id") for k, p in CANDS.items()}
    p = OUT / "telemed_per_image.csv"
    TT = pd.read_csv(p).set_index("image_id") if p.exists() else None
    rows = []
    for c in (args.clips if args.clips else range(28)):
        rows.append(render_clip(banks, obank, lay, S, TT, c, args.frame))
        print(f"clip {c:02d} ok  maxrss {rss():.2f} GB", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"overlay_rows_f{args.frame}.csv", index=False)
    banks.close()
    obank.close()


# =========================================================================== apo-model noise on test
def cmd_aponoise(args) -> None:
    """ll(pa_c) with the apo_v2 curves vs with the classical detector (guided by our fragments) on frames
    0/2/4 of every clip; where both find the same superficial pair (|dMT| < 1 mm) the log-ratio SD / sqrt(2)
    estimates the apo-extrapolation part of the Telemed FL noise -> apo_noise_test_telemed.csv"""
    banks = b3.ea.Banks()
    lay = bc.test_layout()
    TT = pd.read_csv(OUT / "telemed_per_image.csv").set_index("image_id")
    rows = []
    for c in range(28):
        for f in (0, 2, 4):
            iid = f"IMG_{56 + 5 * c + f:05d}.tif"
            img = bc.gray(bc.TEST_DIR / iid)
            ppc = float(lay.loc[iid, "px_per_cm"])
            A, up, lo, fbs, thr, edges = b3.apo_inputs(banks, "test", iid, "v2")
            F = banks.main.prob("test", "ours_fasc384_noflip", iid)
            im2 = img if img.shape == A.shape else cv2.resize(img, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
            cfg = b3.ea.make_cfg("v2e", thr, edges).with_(fasc_thr=0.95, compute_structure=False)
            r1 = g2.measure_apo(A, F, im2, ppc, cfg, apo_upper=up, apo_lower=lo, fallbacks=fbs, keep_details=True)
            det = detect_apos(im2, g2.as_prob(F) >= 0.95)
            r = dict(clip=c, frame=f, image_id=iid, mt_v2=r1.mt_mm)
            if det.get("ok") and not det["pseudo_sup"]:
                cfg2 = g2.recommended_config(apo_thr=0.5, fasc_thr=0.95, compute_streamline=False,
                                             compute_structure=False, edge_source="mask")
                r2 = g2.measure(det["A"], F, im2, ppc, cfg2, keep_details=True)
                if r2.curves and r1.curves:
                    pa = float(TT.loc[iid, "pa_deep_c"])
                    e1, e2 = extra_one(r1, A.shape), extra_one(r2, A.shape)
                    k = int(np.clip(np.floor(pa), 4, 39))
                    fr = pa - k
                    r.update(mt_cl=r2.mt_mm, ll_v2=e1[f"ll_{k:02d}"] * (1 - fr) + e1[f"ll_{k + 1:02d}"] * fr,
                             ll_cl=e2[f"ll_{k:02d}"] * (1 - fr) + e2[f"ll_{k + 1:02d}"] * fr,
                             deep_v2=r1.diag.get("deep_angle"), deep_cl=r2.diag.get("deep_angle"),
                             sup_v2=r1.diag.get("sup_angle"), sup_cl=r2.diag.get("sup_angle"))
            rows.append(r)
    banks.close()
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "apo_noise_test_telemed.csv", index=False)
    ok = (R["mt_cl"] - R["mt_v2"]).abs() < 1.0
    d = np.log(R["ll_cl"] / R["ll_v2"])[ok]
    print(f"matching pairs {int(ok.sum())}/{len(R)} frames, {R.loc[ok, 'clip'].nunique()} clips; log-ratio median "
          f"{d.median():+.3f} SD {d.std():.3f} robust {1.4826 * np.median(np.abs(d - d.median())):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extra")
    e.add_argument("--datasets", nargs="*", default=["static", "dynamic", "vl", "test"])
    sub.add_parser("frames")
    t = sub.add_parser("trainval")
    t.add_argument("--per-group", type=int, default=4)
    sub.add_parser("bench")
    sub.add_parser("build")
    sub.add_parser("aponoise")
    o = sub.add_parser("overlay")
    o.add_argument("--clips", type=int, nargs="*", default=None)
    o.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    {"extra": cmd_extra, "frames": cmd_frames, "trainval": cmd_trainval, "bench": cmd_bench, "build": cmd_build,
     "overlay": cmd_overlay, "aponoise": cmd_aponoise}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
