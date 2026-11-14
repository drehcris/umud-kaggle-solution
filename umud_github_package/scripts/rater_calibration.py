"""Rater-style calibration and honest model selection for the v3 pipeline (key: rater).

Light by design (laptop rules): reads the per-image v3 caches only (no images, no masks, no torch),
one process, < 0.5 GB RSS, a few seconds per recipe.

Subcommands
-----------
offsets [--recipe C4] [--B 4000]
    Rater-style offsets that map a recipe's per-image outputs to the expected TEST label style
    = mean(R7, unknown partner), R7 being probably Paul Ritsche and the partner (Gerardo Romeny)
    unknown -> averaged over the 6 possible partners R1..R6 (pooled over (image, partner)).
      * total offset (additive median, and multiplicative = weighted median of T/P) on clean10 / in25 /
        all35 with by-image bootstrap CIs, per-partner range, leave-one-out gains (score units);
      * decomposition total = MEAS (our bias vs the 7-rater consensus; depends on our predictions,
        so in25 is optimistic) + STYLE (2-rater label minus consensus; labels only, no optimism);
      * sign checks of MEAS out of domain: VL-244 (other raters, straight extrapolation; by-subject
        bootstrap) and dynamic (3 raters, PA vs horizontal; block bootstrap), plus each dynamic
        rater's style vs the other two;
      * the pre-stated decision rule (below) -> recommended_offsets_<recipe>.json;
      * expected test gain/loss of applying the offset under a Laplace error model.
    Outputs: artifacts/eval_v4/rater/offsets_<recipe>*.csv, recommended_offsets_<recipe>.json, rater_styles.csv
select --cands C2 C4 C6 [--spec JSON] [--B 2000] [--tag name]
    Pre-registered model-selection score (weights fixed below BEFORE any candidate was scored with
    it) with paired bootstrap CIs of every pairwise difference, plus the public/private LB
    sampling SD of each pairwise difference from the per-image test distance.
    Outputs: artifacts/eval_v4/rater/select_<tag>.{csv,md}
apply --cand C4 [--offsets recommended_offsets_C4.json]
    Writes the candidate with the recommended offsets applied (validated CSV) to
    artifacts/eval_v4/rater/. Never submits.

Decision rule for offsets (from the task, fixed before looking at the numbers)
------------------------------------------------------------------------------
For each component, an offset is recommended only if
  (a) its clean10 bootstrap 95% CI excludes 0 (clean = the 10 static images not in training),
  (b) it is consistent: clean10, in25 and all35 point estimates share the sign, and the
      out-of-domain MEAS checks (VL, dynamic; where the component exists) do not contradict it
      (a contradicting sign = an opposite-sign estimate whose own CI excludes 0),
and the applied value is HALF the clean10 estimate (shrink toward 0) unless strongly supported
(= (a)+(b) AND in25 CI excludes 0 AND VL+dynamic both agree in sign AND clean10 LOO gain at the full
offset is positive with CI excluding 0), in which case the full clean10 estimate may be used.
The STYLE part (label-only) gets the same rule on all35 because it does not depend on our
predictions; it is always capped at half because "R7 = Paul" is unconfirmed.

Pre-registered selection score (fixed 2026-10-21 before scoring any candidate with it)
--------------------------------------------------------------------------------------
  S_sel = 0.35 * clean10-2rater   (3 comps; mean over the 6 (R7+Rk)/2 labels; by-image bootstrap)
        + 0.35 * VL-244 lodo3     (PA+FL; VL held out whole from the EB error model; by-subject bootstrap)
        + 0.15 * dynamic          (PA vs horizontal + FL; lodo4; circular block bootstrap, L = 10 frames)
        + 0.15 * train142         (3 comps vs GT-mask geometry; both models out of fold; the recipe's own
                                   shrinkage with the full-benchmark error model; near-dup-group bootstrap)
Rationale: clean10-2r is the only honest set in the test label style; VL is the only large honest
out-of-domain set and the only one that reproduces the size of the shrinkage gains; dynamic is the
only Telemed-like set (45% of test) but one subject; train142 is in-domain but scored against mask
geometry, not raters. All in official-score units (lower is better). in25 / all35 static columns are
printed for reference only (optimistic: in training) and are NOT in S_sel.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_candidates_v3 as b3  # noqa: E402

bc, sh = b3.bc, b3.sh
OUT = ROOT / "artifacts/eval_v4/rater"
TAU = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
KEY = {"pa_deg": "PA", "fl_mm": "FL", "mt_mm": "MT"}
C3 = tuple(TAU)
PARTNERS = [f"R{k}" for k in range(1, 7)]
CLEAN = list(bc.CLEAN)
WEIGHTS = {"c10_2r": 0.35, "vl3": 0.35, "dyn": 0.15, "t142": 0.15}   # pre-registered (see docstring)
N_TEST, N_PUB = 309, 102                                             # public ~33 % random by image
DYN_BLOCK = 10
# Test-like error scales for the expected-gain calculation (Laplace scale = MAE): PA and FL from
# C4 on VL-244 lodo3 (0.294*6, 0.561*12); MT has no test-like benchmark -> a range.
TEST_S = {"pa_deg": (1.2, 1.8, 2.5), "fl_mm": (4.0, 6.7, 9.0), "mt_mm": (0.2, 0.8, 1.5)}   # MT 0.2 = static-like

BUILTIN = {
    "C1": dict(apo="ours", pa="fusion", fl="c1", shrink="none"),
    "C2": dict(apo="ours", pa="fusion", fl="c1", shrink="conf"),
    **{k: {kk: vv for kk, vv in v.items() if kk != "desc"} for k, v in b3.CANDS.items()},
}


def rec_of(name: str, spec: dict | None = None) -> dict:
    kw = dict((spec or {}).get(name) or BUILTIN[name])
    off = kw.pop("offset", None)
    kw.pop("desc", None)
    r = b3.R(**kw, name=name)
    r["offset"] = off
    return r


def apply_offset(Q: pd.DataFrame, off: dict | None) -> pd.DataFrame:
    """Post-hoc rater offset: pa_add (deg), pa_mult, fl_add (mm), fl_mult, mt_add (mm)."""
    if not off:
        return Q
    Q = Q.copy()
    Q["pa_deg"] = Q["pa_deg"] * off.get("pa_mult", 1.0) + off.get("pa_add", 0.0)
    Q["fl_mm"] = Q["fl_mm"] * off.get("fl_mult", 1.0) + off.get("fl_add", 0.0)
    Q["mt_mm"] = Q["mt_mm"] + off.get("mt_add", 0.0)
    return Q


# ----------------------------------------------------------------------------------- labels
def static_raters() -> dict:
    """{comp: DataFrame images x R1..R7} (R7_MT im_19 typo already NaN)."""
    lab = bc.static_labels()
    return {c: lab[[f"R{j}_{KEY[c]}" for j in range(1, 8)]].set_axis([f"R{j}" for j in range(1, 8)], axis=1)
            for c in C3}


def two_rater(Rc: pd.DataFrame) -> pd.DataFrame:
    """(R7 + Rk)/2 for k = 1..6 (NaN where Rk is missing, e.g. R5 on 18 images)."""
    return pd.DataFrame({k: (Rc["R7"] + Rc[k]) / 2 for k in PARTNERS})


def dyn_raters() -> dict:
    xl = bc.DYN / "benchmark_architecture_GM_calf_raise_v0.1.0.xlsx"
    out = {}
    for c, sheet in (("pa_deg", "PA"), ("fl_mm", "FL")):
        d = pd.read_excel(xl, sheet_name=sheet).dropna(subset=["frame"])
        d.index = d.frame.astype(int).map(lambda f: f"frame_{f:05d}")
        r = d[[f"Rater_{j}_{sheet}_GM" for j in (1, 2, 3)]].set_axis(["D1", "D2", "D3"], axis=1).astype(float)
        out[c] = r * (10 / bc.DYN_PPC) if c == "fl_mm" else r
    return out


# ----------------------------------------------------------------------------------- bootstrap
def wmedian(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if not len(x):
        return np.nan
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def boot_counts(n: int, B: int, rng, clusters: np.ndarray | None = None, block: int = 0) -> np.ndarray:
    """(B, n) multiplicity of each row in each bootstrap replicate: plain by-row, by cluster
    (all rows of a resampled cluster), or circular block bootstrap of length `block` (rows in order)."""
    W = np.zeros((B, n))
    if block:
        nb = int(math.ceil(n / block))
        for b in range(B):
            st = rng.integers(0, n, nb)
            idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
            np.add.at(W[b], idx, 1)
    elif clusters is not None:
        u, inv = np.unique(clusters, return_inverse=True)
        C = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B)   # (B, n_clusters)
        W = C[:, inv].astype(float)
    else:
        W = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
    return W


def ci(x, q=(2.5, 97.5)) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, q[0])), float(np.percentile(x, q[1]))) if len(x) else (np.nan, np.nan)


def fmt(v: float, lo: float, hi: float, nd: int = 3) -> str:
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


# ----------------------------------------------------------------------------------- predictions
def bench_preds(rec: dict) -> dict:
    """Recipe outputs on every benchmark (after shrinkage and the optional offset)."""
    res = b3.run_bench(rec)
    off = rec.get("offset")
    return {"static": apply_offset(b3._q(res[("lodo4", "static")]), off),
            "vl3": apply_offset(b3._q(res[("lodo3", "vl")]), off),
            "dyn": apply_offset(b3._q(res[("lodo4", "dynamic")]), off),
            "t142": apply_offset(train142_preds(rec), off)}


T142_LEN = ("fl_straight", "fl_central3", "fl_ensemble", "fl_constructed", "mt_mm")


def train142_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    x = b3.load_v3("train142", complete=False)
    meta = b3.ea.train_meta().reindex(x.index)
    mmf = 10.0 / meta.ppc
    for c in list(T142_LEN) + b3.GRID_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce") * mmf
    x["px_per_cm"] = meta.ppc
    g = pd.read_csv(b3.eo.CACHE / "train142.csv")
    gt = g[g.run == "gt"].drop_duplicates("image").set_index("image").reindex(x.index)
    G = pd.DataFrame({"pa_deg": gt.pa_global, "fl_mm": gt.fl_ensemble * mmf, "mt_mm": gt.mt_mm * mmf})
    return x, G, meta.group.astype(str)


_T142: dict = {}


def train142_preds(rec: dict) -> pd.DataFrame:
    """train142 (OOF) through the recipe's per-image heads AND its shrinkage (groups = layout,
    < 5 images pooled; EB error model fitted on all four benchmark datasets, none of which is train142)."""
    key = json.dumps({k: rec[k] for k in ("apo", "pa", "fl", "shrink", "xfeat")}, sort_keys=True)
    if key in _T142:
        return _T142[key].copy()
    x, _, _ = train142_frame()
    d = b3.est_frame(x, "train142", rec)
    lay = x["layout"].astype(str).str.replace("1234x687", "1234x688")
    d["group"] = lay.where(lay.map(lay.value_counts()) >= 5, "other")
    d["dataset"], d["clip"] = "train142", -1
    model = sh.fit_error_model(b3.bench_frame(rec, "lodo4"), **b3.FIT_KW) if rec["shrink"].startswith(("eb", "hyb")) else None
    P = b3.apply_shrink(d, rec["shrink"], model, "train142")
    Q = pd.DataFrame({c: P[f"{k}_eb"].astype(float) for k, c in sh.COMP.items()}, index=P.index)
    _T142[key] = Q
    return Q.copy()


# =================================================================================== offsets
def _resid_stats(P: np.ndarray, T: np.ndarray, cons: np.ndarray) -> dict:
    """P (n,), T (n, K) two-rater targets, cons (n,) 7-rater consensus.  Pooled over (image, partner)."""
    R = T - P[:, None]
    ok = np.isfinite(R)
    return dict(add_med=float(np.nanmedian(R)) if ok.any() else np.nan,
                add_mean=float(np.nanmean(R)) if ok.any() else np.nan,
                mult=wmedian((T / P[:, None]).ravel(), np.repeat(P, T.shape[1])) - 1.0,
                meas=float(np.nanmean(cons - P)),
                style=float(np.nanmean(np.nanmean(T, 1) - cons)))


def _loo_gain(P, T, tau, kind: str, frac: float) -> np.ndarray:
    """Per-image score gain (official units, 1/3 weight of one component) when the offset estimated on
    the OTHER images (pooled median / weighted median of T/P) times `frac` is applied to image i."""
    n = len(P)
    g = np.full(n, np.nan)
    for i in range(n):
        o = np.arange(n) != i
        if kind == "add":
            b = np.nanmedian(T[o] - P[o, None])
            new = P[i] + frac * b
        else:
            m = wmedian((T[o] / P[o, None]).ravel(), np.repeat(P[o], T.shape[1])) - 1.0
            new = P[i] * (1 + frac * m)
        e0, e1 = np.abs(T[i] - P[i]), np.abs(T[i] - new)
        if np.isfinite(e0).any():
            g[i] = np.nanmean(e0 - e1) / tau / 3
    return g


def _transfer_gain(Pa, Ta, Pb, Tb, tau, kind, frac) -> float:
    """Offset estimated on set a, applied to set b (score units)."""
    if kind == "add":
        b = np.nanmedian(Ta - Pa[:, None])
        new = Pb + frac * b
    else:
        m = wmedian((Ta / Pa[:, None]).ravel(), np.repeat(Pa, Ta.shape[1])) - 1.0
        new = Pb * (1 + frac * m)
    return float(np.nanmean(np.abs(Tb - Pb[:, None]) - np.abs(Tb - new[:, None])) / tau / 3)


def laplace_gain(b: float, beta: float, s: float) -> float:
    """MAE reduction from shifting predictions by b when the residual (label - pred) ~ Laplace(beta, s)."""
    f = lambda x: abs(x) + s * math.exp(-abs(x) / s)  # noqa: E731
    return f(beta) - f(beta - b)


def cmd_offsets(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rec = rec_of(args.recipe)
    pr = bench_preds(rec)
    st = pr["static"]
    RS = static_raters()
    ids_all = list(st.index)
    sets = {"clean10": CLEAN, "in25": [i for i in ids_all if i not in CLEAN], "all35": ids_all}
    rows, boots = [], {}
    for c in C3:
        Rc = RS[c].reindex(ids_all)
        T2 = two_rater(Rc)
        cons = Rc.mean(1)
        for sname, ids in sets.items():
            P = st.loc[ids, c].to_numpy(float)
            T = T2.loc[ids].to_numpy(float)
            cs = cons.loc[ids].to_numpy(float)
            pt = _resid_stats(P, T, cs)
            W = boot_counts(len(ids), args.B, rng)
            bs = {k: [] for k in pt}
            for b in range(args.B):
                idx = np.repeat(np.arange(len(ids)), W[b].astype(int))
                s = _resid_stats(P[idx], T[idx], cs[idx])
                for k in bs:
                    bs[k].append(s[k])
            boots[(c, sname)] = bs
            r = dict(comp=KEY[c], subset=sname, n=len(ids))
            for k in pt:
                lo, hi = ci(bs[k])
                r[k], r[f"{k}_lo"], r[f"{k}_hi"] = pt[k], lo, hi
                r[f"{k}_p_pos"] = float(np.mean(np.asarray(bs[k]) > 0))
            per = [np.nanmedian(T[:, j] - P) for j in range(T.shape[1])]
            r["partner_min"], r["partner_max"] = float(np.nanmin(per)), float(np.nanmax(per))
            r["partner_med"] = ";".join(f"{PARTNERS[j]}:{v:+.2f}" for j, v in enumerate(per))
            for kind in ("add", "mult"):
                for frac in (0.5, 1.0):
                    g = _loo_gain(P, T, TAU[c], kind, frac)
                    gw = boot_counts(len(g), args.B, rng)
                    gb = (gw * np.nan_to_num(g)).sum(1) / (gw * np.isfinite(g)).sum(1)
                    lo, hi = ci(gb)
                    r[f"loo_{kind}_{frac}"], r[f"loo_{kind}_{frac}_lo"], r[f"loo_{kind}_{frac}_hi"] = (
                        float(np.nanmean(g)), lo, hi)
            r["mae2r_n"] = float(np.nanmean(np.abs(T - P[:, None]))) / TAU[c]
            rows.append(r)
        # transfer in25 -> clean10 (offset fitted on the in-training images, scored on the clean ones)
        Pa, Ta = st.loc[sets["in25"], c].to_numpy(float), T2.loc[sets["in25"]].to_numpy(float)
        Pb, Tb = st.loc[CLEAN, c].to_numpy(float), T2.loc[CLEAN].to_numpy(float)
        rows.append(dict(comp=KEY[c], subset="in25->clean10", n=10,
                         **{f"xfer_{k}_{f}": _transfer_gain(Pa, Ta, Pb, Tb, TAU[c], k, f)
                            for k in ("add", "mult") for f in (0.5, 1.0)}))
    T = pd.DataFrame(rows)

    # ---- per static scanner group (heterogeneity of MEAS / STYLE across scanners)
    grp = b3.bc.static_groups(pd.Index(ids_all))
    grows = []
    for c in C3:
        Rc = RS[c].reindex(ids_all)
        T2, cons = two_rater(Rc), Rc.mean(1)
        for g in grp.unique():
            ids = list(grp[grp == g].index)
            if len(ids) < 3:
                continue
            P = st.loc[ids, c].to_numpy(float)
            s = _resid_stats(P, T2.loc[ids].to_numpy(float), cons.loc[ids].to_numpy(float))
            grows.append(dict(comp=KEY[c], group=g, n=len(ids), n_clean=sum(i in CLEAN for i in ids),
                              meas=s["meas"], style=s["style"], add_med=s["add_med"], mult=s["mult"]))
    Tg = pd.DataFrame(grows)

    # ---- rater styles on static (each rater vs the mean of the other six)
    srows = []
    for c in C3:
        Rc = RS[c]
        for j in Rc.columns:
            oth = Rc.drop(columns=j).mean(1)
            dlt = (Rc[j] - oth).dropna()
            W = boot_counts(len(dlt), args.B, rng)
            bm = (W * dlt.to_numpy()).sum(1) / W.sum(1)
            lo, hi = ci(bm)
            srows.append(dict(comp=KEY[c], rater=j, n=len(dlt), bias=dlt.mean(), lo=lo, hi=hi,
                              ratio=float(np.median(Rc[j] / oth))))
    Ts = pd.DataFrame(srows)

    # ---- out-of-domain MEAS sign checks: VL (by subject) and dynamic (block bootstrap)
    orows = []
    vgt = bc.vl_gt()
    vl = pr["vl3"]
    ids = [i for i in vl.index if i in vgt.index]
    subj = pd.Index(ids).str.split("__").str[1].to_numpy()
    for c in ("pa_deg", "fl_mm"):
        P = vl.loc[ids, c].to_numpy(float)
        Tt = vgt.loc[ids, c].to_numpy(float)
        ok = np.isfinite(P) & np.isfinite(Tt)
        W = boot_counts(ok.sum(), args.B, rng, clusters=subj[ok])
        add = [wmedian((Tt - P)[ok], W[b]) for b in range(args.B)]
        mul = [wmedian((Tt / P)[ok], P[ok] * W[b]) - 1 for b in range(args.B)]
        mean = (W * (Tt - P)[ok]).sum(1) / W.sum(1)
        orows.append(dict(bench="VL-244 lodo3 (other raters)", comp=KEY[c], n=int(ok.sum()),
                          add_med=float(np.median((Tt - P)[ok])), add_lo=ci(add)[0], add_hi=ci(add)[1],
                          add_mean=float(np.mean((Tt - P)[ok])), mean_lo=ci(mean)[0], mean_hi=ci(mean)[1],
                          mult=wmedian((Tt / P)[ok], P[ok]) - 1, mult_lo=ci(mul)[0], mult_hi=ci(mul)[1]))
    dyg = bc.dynamic_gt()
    dy = pr["dyn"]
    DR = dyn_raters()
    for c in ("pa_deg", "fl_mm"):
        ids = [i for i in dy.index if i in dyg.index and np.isfinite(dyg.loc[i, c])]
        P = dy.loc[ids, c].to_numpy(float)
        Tt = DR[c].reindex(ids).mean(1).to_numpy(float)
        W = boot_counts(len(ids), args.B, rng, block=DYN_BLOCK)
        add = [wmedian(Tt - P, W[b]) for b in range(args.B)]
        mul = [wmedian(Tt / P, P * W[b]) - 1 for b in range(args.B)]
        mean = (W * (Tt - P)).sum(1) / W.sum(1)
        orows.append(dict(bench="dynamic (3 raters; PA vs horizontal)", comp=KEY[c], n=len(ids),
                          add_med=float(np.median(Tt - P)), add_lo=ci(add)[0], add_hi=ci(add)[1],
                          add_mean=float(np.mean(Tt - P)), mean_lo=ci(mean)[0], mean_hi=ci(mean)[1],
                          mult=wmedian(Tt / P, P) - 1, mult_lo=ci(mul)[0], mult_hi=ci(mul)[1]))
        for j in DR[c].columns:
            dlt = (DR[c][j] - DR[c].drop(columns=j).mean(1)).reindex(ids).to_numpy(float)
            Wd = boot_counts(len(dlt), args.B, rng, block=DYN_BLOCK)
            bm = (Wd * dlt).sum(1) / Wd.sum(1)
            srows.append(dict(comp=KEY[c], rater=f"dyn_{j}", n=len(dlt), bias=float(np.mean(dlt)),
                              lo=ci(bm)[0], hi=ci(bm)[1], ratio=float(np.median((DR[c][j] / DR[c].drop(columns=j).mean(1)).reindex(ids)))))
    To = pd.DataFrame(orows)
    Ts = pd.DataFrame(srows)

    # ---- decision rule
    rec_off, decisions = {}, []
    for c in C3:
        k = KEY[c]
        g = lambda s: T[(T.comp == k) & (T.subset == s)].iloc[0]  # noqa: E731
        c10, i25, a35 = g("clean10"), g("in25"), g("all35")
        ood = To[To.comp == k]
        for part, est_key, sub_main in (("total_add", "add_med", c10), ("total_mult", "mult", c10),
                                        ("style", "style", a35), ("meas", "meas", c10)):
            v, lo, hi = sub_main[est_key], sub_main[f"{est_key}_lo"], sub_main[f"{est_key}_hi"]
            excl = (lo > 0) or (hi < 0)
            same = np.sign(c10[est_key]) == np.sign(i25[est_key]) == np.sign(a35[est_key])
            if part == "style":       # labels only: the OOD sets cannot speak to R7's style
                contra, agree_all = False, False
            else:
                okey, olo, ohi = ("mult", "mult_lo", "mult_hi") if part == "total_mult" else ("add_med", "add_lo", "add_hi")
                contra = any((np.sign(r[okey]) == -np.sign(v)) and ((r[olo] > 0) or (r[ohi] < 0))
                             for _, r in ood.iterrows())
                agree_all = len(ood) > 0 and all(np.sign(r[okey]) == np.sign(v) for _, r in ood.iterrows())
            loo_key = "loo_mult_1.0" if part == "total_mult" else "loo_add_1.0"
            strong = (excl and same and not contra and part != "style" and agree_all
                      and ((i25[f"{est_key}_lo"] > 0) or (i25[f"{est_key}_hi"] < 0))
                      and c10.get(f"{loo_key}_lo", -1) > 0)
            apply_ = bool(excl and same and not contra)
            val = (v if strong else 0.5 * v) if apply_ else 0.0
            decisions.append(dict(comp=k, part=part, estimate=v, lo=lo, hi=hi, ci_excludes_0=bool(excl),
                                  same_sign_c10_in25_all35=bool(same), ood_contradicts=bool(contra),
                                  ood_all_agree=bool(agree_all), strongly_supported=bool(strong),
                                  recommend=apply_, applied_value=val))
    D = pd.DataFrame(decisions)
    # recommended offset per component: the additive (PA, MT) or multiplicative (FL) total if it passes
    for c, part, name in (("pa_deg", "total_add", "pa_add"), ("fl_mm", "total_mult", "fl_mult"),
                          ("mt_mm", "total_add", "mt_add")):
        r = D[(D.comp == KEY[c]) & (D.part == part)].iloc[0]
        rec_off[name] = (1.0 + r.applied_value) if name.endswith("mult") else r.applied_value

    # ---- expected test effect of the recommended / half / full offsets (Laplace model, test-like scales)
    erows = []
    for c in C3:
        k = KEY[c]
        c10 = T[(T.comp == k) & (T.subset == "clean10")].iloc[0]
        a35 = T[(T.comp == k) & (T.subset == "all35")].iloc[0]
        lev = {"pa_deg": 15.0, "fl_mm": 85.0, "mt_mm": 21.0}[c]   # typical test level (C4 group medians)
        tot = c10["mult"] * lev if c == "fl_mm" else c10["add_med"]
        sty = a35["style"]
        for s in TEST_S[c]:
            for applied_name, b in (("half total", 0.5 * tot), ("full total", tot), ("half style", 0.5 * sty)):
                for truth_name, beta in (("truth = total", tot), ("truth = style only", sty), ("truth = 0", 0.0)):
                    erows.append(dict(comp=k, s=s, applied=applied_name, b=b, truth=truth_name, beta=beta,
                                      gain_score=laplace_gain(b, beta, s) / TAU[c] / 3))
    Te = pd.DataFrame(erows)

    tag = args.recipe
    T.to_csv(OUT / f"offsets_{tag}.csv", index=False)
    Tg.to_csv(OUT / f"offsets_{tag}_by_group.csv", index=False)
    Ts.to_csv(OUT / "rater_styles.csv", index=False)
    To.to_csv(OUT / f"offsets_{tag}_ood.csv", index=False)
    D.to_csv(OUT / f"offsets_{tag}_decision.csv", index=False)
    Te.to_csv(OUT / f"offsets_{tag}_expected_gain.csv", index=False)
    js = dict(recipe=tag, offsets=rec_off,
              note="pa_add deg / fl_mult factor / mt_add mm applied after shrinkage+post-processing; "
                   "half of the clean10 estimate unless strongly supported; 0 = not recommended",
              decision=D.to_dict("records"))
    (OUT / f"recommended_offsets_{tag}.json").write_text(json.dumps(js, indent=1, default=float))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    show = ["comp", "subset", "n", "add_med", "add_med_lo", "add_med_hi", "mult", "mult_lo", "mult_hi", "meas",
            "meas_lo", "meas_hi", "style", "style_lo", "style_hi", "partner_min", "partner_max",
            "loo_add_0.5", "loo_add_1.0", "loo_add_1.0_lo", "loo_add_1.0_hi", "loo_mult_0.5", "loo_mult_1.0",
            "mae2r_n"]
    print(T[[c for c in show if c in T]].round(3).to_string(index=False))
    print(T[T.subset == "in25->clean10"].dropna(axis=1).round(4).to_string(index=False))
    print(Tg.round(3).to_string(index=False))
    print(Ts.round(3).to_string(index=False))
    print(To.round(3).to_string(index=False))
    print(D.round(3).to_string(index=False))
    print(Te.pivot_table(index=["comp", "s", "applied"], columns="truth", values="gain_score").round(4).to_string())
    print(json.dumps(rec_off, indent=1))
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")


# =================================================================================== select
def _abs_err(pred: pd.DataFrame, gt: pd.DataFrame, comps, ids) -> np.ndarray:
    """(n, C) |pred - gt| / tau (NaN where either is missing)."""
    return np.stack([(pred.reindex(ids)[c].to_numpy(float) - gt.reindex(ids)[c].to_numpy(float)) / TAU[c]
                     for c in comps], 1).__abs__()


def _score(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """A (n, C) or (n, C, K partners); W (B, n) multiplicities -> (B,) official score
    (mean over components [and partners] of the bootstrap-weighted MAE over the available images)."""
    if A.ndim == 2:
        A = A[:, :, None]
    v = np.isfinite(A)
    num = np.einsum("bn,nck->bck", W, np.nan_to_num(A))
    den = np.einsum("bn,nck->bck", W, v.astype(float))
    return np.nanmean(num / np.where(den > 0, den, np.nan), axis=(1, 2))


def bench_arrays(preds: dict) -> dict:
    """Per-benchmark error arrays for one candidate (shapes aligned across candidates)."""
    RS = static_raters()
    st = preds["static"]
    out = {}
    for name, ids in (("c10_2r", CLEAN), ("in25_2r", [i for i in st.index if i not in CLEAN]),
                      ("all35_2r", list(st.index))):
        A = np.stack([np.stack([np.abs(st.reindex(ids)[c].to_numpy(float) - two_rater(RS[c]).reindex(ids)[k].to_numpy(float))
                                / TAU[c] for k in PARTNERS], 1) for c in C3], 1)   # (n, C, K)
        out[name] = A
    vgt = bc.vl_gt()
    ids = sorted(i for i in preds["vl3"].index if i in vgt.index)
    out["vl3"] = _abs_err(preds["vl3"], vgt, ("pa_deg", "fl_mm"), ids)
    out["vl3_ids"] = ids
    dg = bc.dynamic_gt()
    ids = sorted(i for i in preds["dyn"].index if i in dg.index)
    out["dyn"] = _abs_err(preds["dyn"], dg, ("pa_deg", "fl_mm"), ids)
    out["dyn_ids"] = ids
    _, G, grp = train142_frame()
    ids = list(G.index)
    out["t142"] = _abs_err(preds["t142"], G, C3, ids)
    out["t142_ids"] = ids
    out["t142_grp"] = grp.reindex(ids).to_numpy()
    return out


def per_image_score(A: np.ndarray) -> np.ndarray:
    """Per-image score (mean over available comps [and partners])."""
    return np.nanmean(A.reshape(A.shape[0], -1), axis=1)


def cmd_select(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    spec = json.loads(Path(args.spec).read_text()) if args.spec else None
    names = args.cands or (list(spec) if spec else ["C2", "C4", "C6"])
    recs = {n: rec_of(n, spec if spec and n in spec else None) for n in names}
    preds = {n: bench_preds(r) for n, r in recs.items()}
    arrs = {n: bench_arrays(p) for n, p in preds.items()}
    a0 = arrs[names[0]]
    # bootstrap multiplicities, shared by all candidates (paired), independent across benchmarks
    Wb = {"c10_2r": boot_counts(len(CLEAN), args.B, rng),
          "in25_2r": boot_counts(25, args.B, rng),
          "all35_2r": boot_counts(35, args.B, rng),
          "vl3": boot_counts(len(a0["vl3_ids"]), args.B, rng,
                             clusters=pd.Index(a0["vl3_ids"]).str.split("__").str[1].to_numpy()),
          "dyn": boot_counts(len(a0["dyn_ids"]), args.B, rng, block=DYN_BLOCK),
          "t142": boot_counts(len(a0["t142_ids"]), args.B, rng, clusters=a0["t142_grp"])}
    bench_names = list(Wb)
    pt, bs = {}, {}
    for n in names:
        pt[n] = {b: float(_score(arrs[n][b], np.ones((1, arrs[n][b].shape[0])))[0]) for b in bench_names}
        bs[n] = {b: _score(arrs[n][b], Wb[b]) for b in bench_names}
        pt[n]["S_sel"] = sum(WEIGHTS[b] * pt[n][b] for b in WEIGHTS)
        bs[n]["S_sel"] = sum(WEIGHTS[b] * bs[n][b] for b in WEIGHTS)
    rows = []
    for n in names:
        r = dict(cand=n, recipe=recs[n]["name"], spec=json.dumps({k: recs[n][k] for k in ("apo", "pa", "fl", "shrink")}),
                 offset=json.dumps(recs[n].get("offset")) if recs[n].get("offset") else "")
        for b in bench_names + ["S_sel"]:
            lo, hi = ci(bs[n][b])
            r[b], r[f"{b}_lo"], r[f"{b}_hi"] = pt[n][b], lo, hi
        rows.append(r)
    Tc = pd.DataFrame(rows)

    # ---- test predictions and LB sampling SD of each pairwise difference
    subs = {}
    for n, r in recs.items():
        s = b3.test_sub(r)
        subs[n] = apply_offset(s, r.get("offset")).clip(lower=pd.Series({c: bc.BOUNDS[c][0] for c in C3}),
                                                      upper=pd.Series({c: bc.BOUNDS[c][1] for c in C3}), axis=1)
    f_pub = N_PUB / N_TEST
    m_priv = N_TEST - N_PUB
    prow = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = dict(pair=f"{a}-{b}")
            for bn in bench_names + ["S_sel"]:
                d = bs[a][bn] - bs[b][bn]
                lo, hi = ci(d)
                r[f"d_{bn}"] = pt[a][bn] - pt[b][bn]
                r[f"d_{bn}_lo"], r[f"d_{bn}_hi"] = lo, hi
                r[f"P({a}<{b})_{bn}" if bn == "S_sel" else f"p_{bn}"] = float(np.mean(d < 0))
            # robustness: on how many of the four S_sel benchmarks does the CI favour a / favour b
            r["wins_a"] = int(sum(r[f"d_{bn}_hi"] < 0 for bn in WEIGHTS))
            r["wins_b"] = int(sum(r[f"d_{bn}_lo"] > 0 for bn in WEIGHTS))
            # rho = sum d_i^2 / sum dist_i^2 on the benchmarks (how much of the prediction distance
            # turns into a per-image score difference); VL is the noisy, test-like one
            for bn in ("vl3", "c10_2r", "all35_2r", "dyn"):
                A, Bb = arrs[a][bn], arrs[b][bn]
                di = per_image_score(A) - per_image_score(Bb)
                if bn.endswith("2r"):
                    ids = CLEAN if bn == "c10_2r" else list(preds[a]["static"].index)
                    pa, pb = preds[a]["static"].reindex(ids), preds[b]["static"].reindex(ids)
                    comps = C3
                elif bn == "vl3":
                    ids = arrs[a]["vl3_ids"]
                    pa, pb = preds[a]["vl3"].reindex(ids), preds[b]["vl3"].reindex(ids)
                    comps = ("pa_deg", "fl_mm")
                else:
                    ids = arrs[a]["dyn_ids"]
                    pa, pb = preds[a]["dyn"].reindex(ids), preds[b]["dyn"].reindex(ids)
                    comps = ("pa_deg", "fl_mm")
                dist = np.nanmean(np.stack([np.abs(pa[c].to_numpy(float) - pb[c].to_numpy(float)) / TAU[c]
                                            for c in comps], 1), 1)
                ok = np.isfinite(di) & np.isfinite(dist)
                r[f"rho_{bn}"] = float((di[ok] ** 2).sum() / max((dist[ok] ** 2).sum(), 1e-12))
            ta, tb = subs[a], subs[b].reindex(subs[a].index)
            dist = np.mean(np.stack([np.abs(ta[c] - tb[c]).to_numpy(float) / TAU[c] for c in C3], 1), 1)
            r["test_dist"] = float(dist.mean())
            ms = float((dist ** 2).mean())
            for tag, rho in (("rhoVL", r["rho_vl3"]), ("rho1", 1.0)):
                S2 = rho * ms
                sd_pub = math.sqrt((1 - f_pub) * S2 / N_PUB)            # public diff around the full-test diff
                sd_priv = math.sqrt((1 - m_priv / N_TEST) * S2 / m_priv)  # private diff around the full-test diff
                r[f"sd_pub_{tag}"] = sd_pub
                r[f"sd_priv_{tag}"] = sd_priv
                r[f"sd_pub_minus_priv_{tag}"] = (N_TEST / m_priv) * sd_pub
            prow.append(r)
    Tp = pd.DataFrame(prow)
    tag = args.tag or "_".join(names)
    Tc.to_csv(OUT / f"select_{tag}.csv", index=False)
    Tp.to_csv(OUT / f"select_{tag}_pairs.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    lines = [f"# Model selection: {', '.join(names)}", "",
             f"S_sel = {' + '.join(f'{w:.2f}*{k}' for k, w in WEIGHTS.items())} (pre-registered; lower is better). "
             f"in25/all35 are in-training (optimistic) and NOT in S_sel. B = {args.B}.", "",
             "| cand | " + " | ".join(bench_names + ["S_sel"]) + " |", "|---|" + "---|" * (len(bench_names) + 1)]
    for _, r in Tc.iterrows():
        lines.append(f"| {r.cand} | " + " | ".join(f"{r[b]:.4f} [{r[f'{b}_lo']:.3f}, {r[f'{b}_hi']:.3f}]"
                                                    for b in bench_names + ["S_sel"]) + " |")
    if any(recs[n].get("offset") for n in names):
        lines += ["", "NOTE: candidates with an `offset` were scored with offsets estimated on the static 2-rater "
                  "labels, so their c10_2r (and in25/all35) values are IN-SAMPLE; the honest leave-one-out "
                  "clean10 gain is in offsets_<recipe>.csv (loo_* columns). VL/dynamic/train142 are not rater-R7 "
                  "labels, so they test only that the offset does not hurt against other raters / mask geometry."]
    lines += ["", "Pairwise differences (a - b; negative = a better), 95% paired bootstrap CI:", "",
              "| pair | " + " | ".join(f"d {b}" for b in WEIGHTS) + " | d S_sel | P(a<b) | CI wins a/b | test dist | rho VL | "
              "SD pub (rhoVL / rho1) | SD pub-priv (rhoVL / rho1) |", "|---|" + "---|" * (len(WEIGHTS) + 7)]
    for _, r in Tp.iterrows():
        a, b = r.pair.split("-")
        lines.append(f"| {r.pair} | " + " | ".join(fmt(r[f'd_{bn}'], r[f'd_{bn}_lo'], r[f'd_{bn}_hi'])
                                                    for bn in WEIGHTS)
                     + f" | {fmt(r['d_S_sel'], r['d_S_sel_lo'], r['d_S_sel_hi'], 4)} | {r[f'P({a}<{b})_S_sel']:.2f}"
                     + f" | {r.wins_a}/{r.wins_b}"
                     + f" | {r.test_dist:.3f} | {r.rho_vl3:.2f} | {r.sd_pub_rhoVL:.4f} / {r.sd_pub_rho1:.4f}"
                     + f" | {r.sd_pub_minus_priv_rhoVL:.4f} / {r.sd_pub_minus_priv_rho1:.4f} |")
    lines += ["", "Optimistic static columns (in training; reference only):", "",
              "| pair | d in25_2r | d all35_2r |", "|---|---|---|"]
    for _, r in Tp.iterrows():
        lines.append(f"| {r.pair} | {fmt(r['d_in25_2r'], r['d_in25_2r_lo'], r['d_in25_2r_hi'])} | "
                     f"{fmt(r['d_all35_2r'], r['d_all35_2r_lo'], r['d_all35_2r_hi'])} |")
    md = "\n".join(lines) + "\n"
    (OUT / f"select_{tag}.md").write_text(md)
    print(md)
    print(Tp[["pair"] + [c for c in Tp.columns if c.startswith("rho_")]].round(3).to_string(index=False))
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")


# =================================================================================== apply
def cmd_apply(args) -> None:
    off = json.loads(Path(args.offsets).read_text())["offsets"]
    rec = rec_of(args.cand)
    sub = b3.test_sub(rec)
    adj = apply_offset(sub, off)
    for c in C3:
        adj[c] = adj[c].clip(*bc.BOUNDS[c])
    tagv = "_".join(f"{k}{v:+.3f}" if not k.endswith("mult") else f"{k}x{v:.4f}" for k, v in off.items())
    path = OUT / f"{args.cand}_rater_{tagv}.csv"
    bc.write_sub(adj, path)
    print(path)
    print(b3.validate(path))
    print("distance to un-offset:", bc.official_distance(adj, sub))


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("offsets")
    o.add_argument("--recipe", default="C4")
    o.add_argument("--B", type=int, default=4000)
    o.add_argument("--seed", type=int, default=0)
    s = sp.add_parser("select")
    s.add_argument("--cands", nargs="*")
    s.add_argument("--spec", help="JSON {name: {apo, pa, fl, shrink, xfeat, offset?}}; built-ins: " + ", ".join(BUILTIN))
    s.add_argument("--B", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tag")
    a = sp.add_parser("apply")
    a.add_argument("--cand", default="C4")
    a.add_argument("--offsets", default=str(OUT / "recommended_offsets_C4.json"))
    args = ap.parse_args()
    {"offsets": cmd_offsets, "select": cmd_select, "apply": cmd_apply}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
