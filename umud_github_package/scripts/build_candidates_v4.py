"""Assemble the v4 candidates: C4pp (critic fix 2) + muscle-aware shrinkage groups (fix 1) + rater offsets.

Light by design (laptop rules): reads the per-image v3/v4 caches only (no images, no masks, no torch),
one process, < 0.5 GB RSS.

Pieces (all validated upstream; see artifacts/candidates_v4/README.md)
-----------------------------------------------------------------------
* flfix agent: FL head ``ens_pp`` (constructed fascicle rebuilt at the POST-shrinkage PA) and the
  leak-free label constants for held-out evaluation (``consts="lodo"``).
* muscle agent: burned-in muscle labels (``umud.muscle_label``; OCR 120/120 correct, IMG_00001-035 all VL by
  the OCR + block rule). Only OCR- and block-labelled images are used (``inferred=False``: the GNB rule
  uses our own PA/FL, so its labels are circular as shrinkage centres).
* rater agent: half rater-style offsets (pa_add / fl_mult / mt_add) by the pre-registered decision rule,
  re-estimated here for the v4 pipeline (``offsets`` subcommand, the rater script's own code).

Muscle-aware shrinkage (critic fix 1), ``groups="muscle"`` (C9, C7)
-------------------------------------------------------------------
cell(i) = '<scanner>|<muscle>' if image i has a trusted muscle label and that scanner x muscle cell has
>= 8 images in the target set AND the cell's median raw FL/FOV <= GATE (2.5, the limit of the benchmark
support; see GATE); otherwise '' (leftover: unknown muscle, small cell, or unsupported regime).
* leftover images keep C8's rule (scanner-group median centre for PA; scanner-group EB prior for FL); they
  move only through the scanner-wide FL prior, because the cell images' FL is rebuilt at their new PA
  (on test <= 0.6 mm, PA unchanged);
* PA (confidence rule): w x + (1 - w) * median(x over the image's cell);
* FL (EB, Laplace likelihood): prior centre = the plain median of the cell's estimates, (tau, c) kept from
  the pooled scanner fit (centre only; ``muscle_refit`` refits them on cell-demeaned residuals instead);
* FL ens_pp is rebuilt at the muscle-aware post-shrinkage PA.
With no trusted labels (every benchmark; Telemed and legacy test groups) the result is bit-identical to
the scanner-group pipeline (checked by ``bench``). Sensitivity variants: see MODES.

Subcommands
-----------
offsets   rater-style offsets for the C8 recipe (rater_calibration.cmd_offsets, honest LODO constants)
bench     benchmark table (LODO constants), identity checks, S_sel with paired bootstrap CIs and LB
          sampling SDs, muscle-grouping simulations (VL young/old split; subsampled test-sized cells;
          semi-synthetic test-calibrated muscle offsets)
build     C7 / C8 / C9 (+ diagnostic variants), validated, + per-group / per-cell diagnostics + report
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_candidates_v3 as b3  # noqa: E402
import rater_calibration as rc  # noqa: E402
from umud import muscle_label as ml  # noqa: E402

warnings.filterwarnings("ignore")
bc, sh = b3.bc, b3.sh
OUT = ROOT / "artifacts/candidates_v4"
MIN_N = 8
C3 = bc.C3
TAU = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
N_TEST, N_PUB = 309, 102
OFFSETS_JSON = OUT / "rater/recommended_offsets_C8.json"
# Support gate (from the benchmarks, never the LB): a scanner x muscle cell gets its own centre only if its
# median raw FL/FOV <= GATE. Binned by the TRUE FL/FOV, our raw FL on VL is unbiased up to 2.5 (median
# raw/true 1.026 / 1.005 / 0.953 / 1.018 in (0,1.5] / (1.5,2] / (2,2.25] / (2.25,2.5]), but no benchmark
# image has a true FL/FOV above 2.5 (VL FOV 47 mm; static <= 2; dynamic <= 1), so a cell centre built from raw
# FL beyond 2.5 is unvalidated (on test: Lumify RF, median 2.86, whose raw coupling k = 1.18 is also above
# the expert 1.05-1.10). Such a cell keeps the scanner prior (= C8). vl_fl_bias_by_fov.csv has the table.
GATE = 2.5
# group modes: gate = regime gate on, fl = FL centres per cell too (else PA only), refit = FL (tau, c)
# refitted on cell-demeaned residuals (else the pooled scanner fit, i.e. centre only), key = cells as plain
# group keys (own (mu, tau, c); leftovers centred on their own median), block = Siemens split by acquisition
# block, inferred = GNB-inferred GM labels too (circular)
MODES = {
    "scanner": None,
    "muscle": dict(gate=True, fl=True, refit=False),                 # C9 / C7
    "muscle_refit": dict(gate=True, fl=True, refit=True),
    "muscle_ungated": dict(gate=False, fl=True, refit=False),
    "muscle_ungated_refit": dict(gate=False, fl=True, refit=True),
    "muscle_pa": dict(gate=False, fl=False, refit=False),
    "muscle_key": dict(gate=False, fl=True, refit=True, key=True),
    "muscle_block": dict(gate=True, fl=True, refit=False, block=True),
    "muscle_inf": dict(gate=True, fl=True, refit=False, inferred=True),
}


def mode(r: dict) -> dict:
    return MODES[r["groups"]] or {}


# ----------------------------------------------------------------------------------- recipes
def V(name: str, groups: str = "scanner", shrink: str = "hyb_med", fl: str = "ens_pp", pa: str = "fusF",
      offset=None, apo: str = "v2", desc: str = "") -> dict:
    """v4 recipe. groups: scanner | muscle | muscle_key | muscle_block. offset: None | dict | "rec"
    (= the re-estimated recommended half offsets, OFFSETS_JSON)."""
    return dict(name=name, apo=apo, pa=pa, fl=fl, shrink=shrink, groups=groups, offset=offset, desc=desc)


def b3rec(r: dict) -> dict:
    return b3.R(r["apo"], r["pa"], r["fl"], r["shrink"], name=r["name"])


def offset_of(r: dict) -> dict | None:
    o = r.get("offset")
    if o == "rec":
        return json.loads(OFFSETS_JSON.read_text())["offsets"]
    return o or None


CANDS = {
    "C7": V("C7", groups="muscle", offset="rec",
            desc="best full pipeline: C8 + muscle-aware shrinkage centres (regime-gated) + half rater-style offsets"),
    "C8": V("C8", desc="conservative: C4 + fix 2 only (FL constructed at the post-shrinkage PA) = flfix_C4pp"),
    "C9": V("C9", groups="muscle", desc="C8 + muscle-aware shrinkage centres only (regime-gated; isolates fix 1)"),
}
DIAG = {   # diagnostic variants (not for submission unless the README says so)
    "D_C4_muscle": V("D_C4_muscle", groups="muscle", fl="ens",
                     desc="literal 'C4 + muscle groups' (no fix 2): isolates fix 1 against C4"),
    "D_C8_offset": V("D_C8_offset", offset="rec", desc="C8 + rater offsets only"),
    "D_C6pp": V("D_C6pp", shrink="conf", desc="C6 + fix 2 (= flfix_C6pp), conservative-shrinkage partner"),
    "D_C6pp_muscle_off": V("D_C6pp_muscle_off", shrink="conf", groups="muscle", offset="rec",
                           desc="C7 with C2's confidence shrinkage for FL (C6pp + muscle + offsets)"),
    "D_C9_ungated": V("D_C9_ungated", groups="muscle_ungated",
                      desc="sensitivity: every n>=8 cell gets its own centre (no FL/FOV regime gate)"),
    "D_C9_ungated_refit": V("D_C9_ungated_refit", groups="muscle_ungated_refit",
                            desc="sensitivity: ungated + FL (tau, c) refitted on cell-demeaned residuals"),
    "D_C9_refit": V("D_C9_refit", groups="muscle_refit", desc="sensitivity: gated + demeaned (tau, c) refit"),
    "D_C9_paonly": V("D_C9_paonly", groups="muscle_pa", desc="sensitivity: ungated, PA centres only"),
    "D_C9_key": V("D_C9_key", groups="muscle_key", desc="sensitivity: ml.muscle_groups keys (own mu, tau, c)"),
    "D_C9_block": V("D_C9_block", groups="muscle_block", desc="sensitivity: gated, Siemens split by block too"),
    "D_C9_inferred": V("D_C9_inferred", groups="muscle_inf", desc="sensitivity: gated + GNB-inferred GM (circular)"),
}


# ----------------------------------------------------------------------------------- muscle cells
def cell_keys(index: pd.Index, scanner: pd.Series, groups: str, ratio: pd.Series | None = None,
              min_n: int = MIN_N) -> pd.Series:
    """'' = leftover (scanner prior, exactly C8), else the cell key. Test images only (benchmarks: all '').
    ratio: per-image raw FL / FOV for the regime gate."""
    empty = pd.Series("", index=index, dtype=object)
    md_ = MODES[groups]
    if not md_:
        return empty
    t = ml.load_test_muscle().reindex(index)
    if t["muscle"].isna().all():
        return empty
    m, _ = ml._trusted(t.fillna(""), 0.9, md_.get("inferred", False))
    base = scanner.reindex(index).astype(str)
    if md_.get("block"):
        base = t["block"].astype(str).where(base == "siemens_juniper", base)
    key = base + "|" + m
    n = key.map(key[m != ""].value_counts()).fillna(0)
    key = key.where((m != "") & (n >= min_n), "").astype(object)
    if md_.get("gate") and ratio is not None:
        med = ratio.reindex(index)[key != ""].groupby(key[key != ""]).median()
        bad = set(med.index[med > GATE])
        key = key.where(~key.isin(bad), "")
    return key


def centres(x: pd.Series, grp: pd.Series, cell: pd.Series) -> pd.Series:
    """Prior centre per row: the median over the row's cell, or over its (scanner) group for leftovers."""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    c = x.groupby(grp.reindex(x.index).astype(str)).transform("median")
    cell = cell.reindex(x.index).fillna("")
    has = cell != ""
    if has.any():
        c[has] = x[has].groupby(cell[has]).transform("median")
    return c


def conf_rule(x: pd.Series, conf: pd.Series, grp: pd.Series, cell: pd.Series) -> pd.Series:
    w = pd.to_numeric(conf, errors="coerce").fillna(0.0).clip(0, 1)
    med = centres(x, grp, cell)
    return (w * x + (1 - w) * med).where(x.notna(), med)


def keyed(D: pd.DataFrame, r: dict, cell: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """groups='muscle_key': the cell key replaces the scanner group (leftovers keep the scanner key, so they
    are centred on the leftovers' own median) and no per-row cell centre is used."""
    if not mode(r).get("key"):
        return D, cell
    cell = cell.reindex(D.index).fillna("")
    return D.assign(group=D["group"].astype(str).where(cell == "", cell)), pd.Series("", index=D.index, dtype=object)


def post_group(B: pd.DataFrame, r: dict, cell: pd.Series) -> pd.DataFrame:
    """b3.v4_post_group with the muscle-aware PA centre: pa_deep_c (the PA that is output, deep reference)
    and, for fl = ens_pp, FL with the constructed fascicle rebuilt at pa_deep_c."""
    B = B.copy()
    p = b3.pa_head(B, r["pa"], "deep")
    p[b3._col(B, "mt_mm").isna()] = np.nan
    B["pa_deep_raw"] = p
    B["pa_deep_c"] = conf_rule(p, b3._col(B, "confidence"), B["group"].astype(str), cell)
    if r["fl"] == "ens_pp":
        fl = b3.nanmean_w([b3._col(B, "fl_straight"), b3._col(B, "fl_central3"), b3.cons_at(B, B["pa_deep_c"])],
                          [1.0, 1.0, 1.0])
        fl[b3._col(B, "mt_mm").isna()] = np.nan
        B["fl_mm"] = fl
    return B


EB_KW = dict(method="ml", likelihood="laplace", cols=("fl",), prior_kw=dict(mu_mode="median"))


def eb_fl(D: pd.DataFrame, model: sh.ErrorModel, cell: pd.Series, sigma: pd.DataFrame | None = None,
          refit: bool = True) -> pd.DataFrame:
    """C4's FL EB (sh.shrink, scanner groups, median centre); then, for the labelled images, the cell-centred
    posterior with (tau, c) refitted per scanner on the cell-demeaned residuals (refit=False: centre-only,
    keeping the scanner-group (tau, c) of the pooled fit)."""
    P = sh.shrink(D, model, group_col="group", clip_col="clip", sigma=sigma, **EB_KW)
    cell = cell.reindex(D.index).fillna("")
    has = cell != ""
    P["cell"] = cell
    P["c_fl"] = P["group"].astype(str).map({k.split("|")[0]: v["c"] for k, v in P.attrs["priors"].items()
                                            if k.endswith("|fl")})
    if not has.any():
        return P
    clip = pd.to_numeric(D.get("clip", pd.Series(-1, index=D.index)), errors="coerce").fillna(-1)
    if (clip[has] >= 0).any():
        raise ValueError("muscle cells must not contain clip units")
    x = b3._col(D, "fl_mm")
    s = P["sigma_fl"].astype(float)
    grp = D["group"].astype(str)
    pri = dict(P.attrs["priors"])
    for g in grp[has].unique():
        idx = has & (grp == g)
        cm = x[idx].groupby(cell[idx]).transform("median")
        r = (x[idx] - cm).to_numpy(float)
        if refit:
            pr = sh.fit_group_prior(r, s[idx].to_numpy(float), "fl", method="ml", mu_mode="median")
            pri[f"{g}|fl|demeaned"] = dict(mu=pr.mu, tau=pr.tau, c=pr.c, n_units=pr.n, method=pr.method)
        else:
            q = pri[f"{g}|fl"]
            pr = sh.Prior(mu=0.0, tau=q["tau"], c=q["c"], n=q["n_units"], method="pooled")
        for i in idx[idx].index:
            prior = sh.Prior(mu=float(cm[i]), tau=pr.tau, c=pr.c)
            v = sh._post_1d(float(x[i]), float(s[i]), prior, "laplace")
            P.at[i, "fl_eb"] = v
            P.at[i, "mu_fl"] = cm[i]
            P.at[i, "tau_fl"] = pr.tau
            P.at[i, "c_fl"] = pr.c
            d = x[i] - cm[i]
            P.at[i, "w_fl"] = float(np.clip((v - cm[i]) / d, -0.5, 1.5)) if np.isfinite(d) and abs(d) > 1e-9 else \
                (1.0 if np.isfinite(x[i]) else 0.0)
    P.attrs["priors"] = pri
    return P


def apply_m(D: pd.DataFrame, r: dict, model: sh.ErrorModel | None, ds: str, cell: pd.Series,
            sigma: pd.DataFrame | None = None) -> pd.DataFrame:
    """Shrinkage with muscle-aware centres ('muscle_key' recipes arrive here already re-keyed by keyed())."""
    import eval_shrinkage as es
    cell = cell.reindex(D.index).fillna("")
    md_ = mode(r)
    fl_cell = cell if md_.get("fl", True) else pd.Series("", index=D.index, dtype=object)
    conf = b3._col(D, "confidence")
    grp = D["group"].astype(str)
    if r["shrink"] == "hyb_med":
        P = eb_fl(D, model, fl_cell, sigma, refit=md_.get("refit", False))
        P["pa_eb"] = conf_rule(b3._col(D, "pa_deg"), conf, grp, cell)
        P["w_pa"] = conf.fillna(0).clip(0, 1)
    elif r["shrink"] == "conf":
        P = D.copy()
        for c, col in sh.COMP.items():
            P[f"{c}_eb"] = b3._col(D, col)
        P["pa_eb"] = conf_rule(b3._col(D, "pa_deg"), conf, grp, cell)
        P["fl_eb"] = conf_rule(b3._col(D, "fl_mm"), conf, grp, fl_cell)
        P["cell"] = cell
    else:
        raise ValueError(r["shrink"])
    return es.fill_group(P)


# ----------------------------------------------------------------------------------- benchmarks
def bench_B(r: dict, split: str, cells: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    B = b3.bench_frame(b3rec(r), split)
    cell = pd.Series("", index=B.index, dtype=object)
    if cells:
        for k, v in cells.items():
            if k in cell.index:
                cell[k] = v
    return post_group(B, r, cell), cell


def run_bench_m(r: dict, consts: str = "lodo", cells: dict | None = None, splits=("lodo4", "lodo3")) -> dict:
    """b3.run_bench_v4 through the muscle-aware code path (cells: {bench row index: cell key})."""
    out = {}
    for split in splits:
        B, cell = bench_B(r, split, cells)
        for ds in B.dataset.unique():
            if split == "lodo3" and ds != "vl":
                continue
            D = B[B.dataset == ds]
            ctx = contextlib.nullcontext()
            if consts == "lodo":
                c = b3.v4_constants(b3.V4_HOLD[ds])
                ctx = sh.constants(c["pop_sd"], c["k0"], c["s_logk"])
            with ctx:
                model = sh.fit_error_model(B[B.dataset != ds], **b3.FIT_KW) if r["shrink"] == "hyb_med" else None
                out[(split, ds)] = apply_m(D, r, model, ds, cell)
    return out


def train142_q(r: dict) -> pd.DataFrame:
    """rc.train142_preds through the v4 path (ens_pp rebuilt at the post-shrinkage PA; no muscle labels)."""
    x, _, _ = rc.train142_frame()
    rr = b3rec(r)
    d = b3.est_frame(x, "train142", rr)
    lay = x["layout"].astype(str).str.replace("1234x687", "1234x688")
    d["group"] = lay.where(lay.map(lay.value_counts()) >= 5, "other")
    d["dataset"], d["clip"] = "train142", -1
    cell = pd.Series("", index=d.index, dtype=object)
    d = post_group(d, r, cell)
    model = sh.fit_error_model(b3.bench_frame_v4(rr, "lodo4"), **b3.FIT_KW) if r["shrink"] == "hyb_med" else None
    P = apply_m(d, r, model, "train142", cell)
    return pd.DataFrame({c: P[f"{k}_eb"].astype(float) for k, c in sh.COMP.items()}, index=P.index)


_BP: dict = {}


def bench_preds(r: dict, consts: str = "lodo") -> dict:
    """{static, vl3, vl4, dyn, t142} predictions (after shrinkage and the optional offset)."""
    key = json.dumps({k: r[k] for k in ("apo", "pa", "fl", "shrink", "groups")}, sort_keys=True) + consts
    if key not in _BP:
        res = run_bench_m(r, consts)
        _BP[key] = dict(res=res, static=b3._q(res[("lodo4", "static")]), vl3=b3._q(res[("lodo3", "vl")]),
                        vl4=pd.concat([b3._q(res[("lodo4", "vl_young")]), b3._q(res[("lodo4", "vl_old")])]),
                        dyn=b3._q(res[("lodo4", "dynamic")]), t142=train142_q(r))
    p = _BP[key]
    off = offset_of(r)
    return {k: (rc.apply_offset(v, off) if k != "res" else v) for k, v in p.items()}


# ----------------------------------------------------------------------------------- test
def test_run(r: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Full test pipeline -> (submission frame, per-image frame, cell keys)."""
    rr = b3rec(r)
    T = b3.test_frame(rr)
    cell0 = cell_keys(T.index, T["group"], r["groups"], ratio=b3._col(T, "fl_mm") / sh._fov_mm(T))
    T, cell = keyed(T, r, cell0)
    T = post_group(T, r, cell)
    model = sh.fit_error_model(b3.bench_frame_v4(rr, "lodo4"), **b3.FIT_KW) if r["shrink"] == "hyb_med" else None
    P = apply_m(T, r, model, "test", cell)
    import eval_shrinkage as es
    sub = es.to_sub(P)
    off = offset_of(r)
    if off:
        sub = rc.apply_offset(sub, off)
        for c in C3:
            sub[c] = sub[c].clip(*bc.BOUNDS[c])
    return sub, P, cell0


def lb_sd(a: pd.DataFrame, b: pd.DataFrame, rho: float) -> dict:
    """Public-LB sampling SD of S(a) - S(b) (102 of 309 images, random by image) from the per-image test
    distance: var(d_i) <= rho E[dist_i^2]; also SD(public - private)."""
    b = b.reindex(a.index)
    dist = np.mean(np.stack([np.abs(a[c] - b[c]).to_numpy(float) / TAU[c] for c in C3], 1), 1)
    S2 = rho * float((dist ** 2).mean())
    f = N_PUB / N_TEST
    sd_pub = math.sqrt((1 - f) * S2 / N_PUB)
    return dict(dist=float(dist.mean()), sd_pub=sd_pub, sd_pub_priv=sd_pub * N_TEST / (N_TEST - N_PUB))


# ----------------------------------------------------------------------------------- offsets
def cmd_offsets(args) -> None:
    """Re-estimate the rater-style offsets for C8 with the rater agent's own code (decision rule fixed in
    rater_calibration's docstring), on the v4 predictions with honest LODO constants."""
    (OUT / "rater").mkdir(parents=True, exist_ok=True)
    r = V("C8")
    rc.OUT = OUT / "rater"
    rc.rec_of = lambda name, spec=None: dict(r, offset=None)  # noqa: E731
    rc.bench_preds = lambda rec: {k: v for k, v in bench_preds(dict(rec, offset=None)).items()
                                  if k in ("static", "vl3", "dyn", "t142")}
    rc.cmd_offsets(argparse.Namespace(recipe="C8", B=args.B, seed=0))


# ----------------------------------------------------------------------------------- selection
def honest_c10_offset(st: pd.DataFrame, off: dict) -> np.ndarray:
    """(10, 3, 6) clean10 2-rater error array for 'st + offset' where each clean image's offset is
    re-estimated from the OTHER 9 clean images (same estimator and the same halving as the rater rule;
    components the rule switched off stay off)."""
    RS = rc.static_raters()
    ids = rc.CLEAN
    A = np.zeros((len(ids), 3, len(rc.PARTNERS)))
    for ci, c in enumerate(C3):
        T2 = rc.two_rater(RS[c]).reindex(ids).to_numpy(float)
        P = st.reindex(ids)[c].to_numpy(float)
        for i in range(len(ids)):
            o = np.arange(len(ids)) != i
            if c == "fl_mm":
                on = off.get("fl_mult", 1.0) != 1.0
                m = rc.wmedian((T2[o] / P[o, None]).ravel(), np.repeat(P[o], T2.shape[1])) - 1.0
                new = P[i] * (1 + 0.5 * m) if on else P[i]
            else:
                on = off.get("pa_add" if c == "pa_deg" else "mt_add", 0.0) != 0.0
                bb = np.nanmedian(T2[o] - P[o, None])
                new = P[i] + 0.5 * bb if on else P[i]
            A[i, ci, :] = np.abs(T2[i] - new) / TAU[c]
    return A


def select(cands: dict, B: int = 2000, seed: int = 0, tag: str = "v4") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rater agent's pre-registered S_sel (0.35 clean10-2r + 0.35 VL lodo3 + 0.15 dynamic + 0.15 train142)
    with honest LODO constants, paired bootstrap CIs, and the LB sampling SD of every pairwise difference.
    For offset candidates the clean10 term is ALSO given leave-one-out (c10_2r_loo, S_sel_loo)."""
    rng = np.random.default_rng(seed)
    names = list(cands)
    preds = {n: bench_preds(cands[n]) for n in names}
    arrs = {n: rc.bench_arrays(p) for n, p in preds.items()}
    for n in names:
        off = offset_of(cands[n])
        base = bench_preds(dict(cands[n], offset=None))["static"]
        arrs[n]["c10_2r_loo"] = honest_c10_offset(base, off) if off else arrs[n]["c10_2r"]
    a0 = arrs[names[0]]
    W = {"c10_2r": rc.boot_counts(len(rc.CLEAN), B, rng), "all35_2r": rc.boot_counts(35, B, rng),
         "vl3": rc.boot_counts(len(a0["vl3_ids"]), B, rng,
                               clusters=pd.Index(a0["vl3_ids"]).str.split("__").str[1].to_numpy()),
         "dyn": rc.boot_counts(len(a0["dyn_ids"]), B, rng, block=rc.DYN_BLOCK),
         "t142": rc.boot_counts(len(a0["t142_ids"]), B, rng, clusters=a0["t142_grp"])}
    W["c10_2r_loo"] = W["c10_2r"]
    bn = list(W)
    pt, bs = {}, {}
    for n in names:
        pt[n] = {b: float(rc._score(arrs[n][b], np.ones((1, arrs[n][b].shape[0])))[0]) for b in bn}
        bs[n] = {b: rc._score(arrs[n][b], W[b]) for b in bn}
        for k, c10 in (("S_sel", "c10_2r"), ("S_sel_loo", "c10_2r_loo")):
            pt[n][k] = 0.35 * pt[n][c10] + 0.35 * pt[n]["vl3"] + 0.15 * pt[n]["dyn"] + 0.15 * pt[n]["t142"]
            bs[n][k] = 0.35 * bs[n][c10] + 0.35 * bs[n]["vl3"] + 0.15 * bs[n]["dyn"] + 0.15 * bs[n]["t142"]
    rows = []
    for n in names:
        rr = dict(cand=n, desc=cands[n].get("desc", ""))
        for b in bn + ["S_sel", "S_sel_loo"]:
            lo, hi = rc.ci(bs[n][b])
            rr[b], rr[f"{b}_lo"], rr[f"{b}_hi"] = pt[n][b], lo, hi
        rows.append(rr)
    Tc = pd.DataFrame(rows)
    subs = {n: test_run(cands[n])[0] for n in names}
    prow = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rr = dict(pair=f"{a}-{b}")
            for k in bn + ["S_sel", "S_sel_loo"]:
                d = bs[a][k] - bs[b][k]
                lo, hi = rc.ci(d)
                rr[f"d_{k}"], rr[f"d_{k}_lo"], rr[f"d_{k}_hi"] = pt[a][k] - pt[b][k], lo, hi
                rr[f"P_{k}"] = float(np.mean(d < 0))
            # rho on VL (share of the prediction distance that becomes a per-image score difference)
            ids = arrs[a]["vl3_ids"]
            di = rc.per_image_score(arrs[a]["vl3"]) - rc.per_image_score(arrs[b]["vl3"])
            pa_, pb_ = preds[a]["vl3"].reindex(ids), preds[b]["vl3"].reindex(ids)
            dist = np.nanmean(np.stack([np.abs(pa_[c].to_numpy(float) - pb_[c].to_numpy(float)) / TAU[c]
                                        for c in ("pa_deg", "fl_mm")], 1), 1)
            ok = np.isfinite(di) & np.isfinite(dist)
            den = float((dist[ok] ** 2).sum())
            rr["rho_vl"] = float((di[ok] ** 2).sum() / den) if den > 1e-12 else np.nan
            for tg, rho in (("rho06", 0.6), ("rho1", 1.0)):
                s = lb_sd(subs[a], subs[b], rho)
                rr["test_dist"] = s["dist"]
                rr[f"sd_pub_{tg}"], rr[f"sd_pub_priv_{tg}"] = s["sd_pub"], s["sd_pub_priv"]
            prow.append(rr)
    Tp = pd.DataFrame(prow)
    Tc.to_csv(OUT / f"select_{tag}.csv", index=False)
    Tp.to_csv(OUT / f"select_{tag}_pairs.csv", index=False)
    return Tc, Tp


# ----------------------------------------------------------------------------------- simulations
def _vl_frames(r: dict, consts: str = "lodo"):
    """lodo3 VL rows after post_group with no cells, and the lodo3 error model (VL held out whole)."""
    B, _ = bench_B(r, "lodo3")
    c = b3.v4_constants("vl")
    with sh.constants(c["pop_sd"], c["k0"], c["s_logk"]):
        model = sh.fit_error_model(B[B.dataset != "vl"], **b3.FIT_KW)
    return B, model, c


def _score_vl(P: pd.DataFrame, gt: pd.DataFrame) -> pd.Series:
    """Per-image official error over PA + FL (lower is better), indexed by image id."""
    Q = b3._q(P)
    g = gt.reindex(Q.index)
    return ((Q["pa_deg"] - g["pa_deg"]).abs() / TAU["pa_deg"] + (Q["fl_mm"] - g["fl_mm"]).abs() / TAU["fl_mm"]) / 2


def sim_vl_split(r: dict, R: int = 300, seed: int = 1) -> dict:
    """Muscle-aware grouping on real data with a real, estimate-independent subgroup label: VL young vs old
    (expert PA 17.0 vs 14.8 deg, FL 76.8 vs 77.0 mm, MT 21.0 vs 18.8 mm). (a) all 244 images, pooled 'vl'
    group vs cells young/old; (b) R random test-sized pseudo-scanners (1 image per subject) with cells of
    n_cell and n_left leftover images (Lumify-like), pooled vs muscle-aware, per-draw difference."""
    gt = bc.vl_gt()
    B, model, c = _vl_frames(r)
    D0 = B[B.dataset == "vl"]
    sub_of = pd.Series(D0.index.str.split(":", n=1).str[1].str.split("__").str[1], index=D0.index)
    age = pd.Series(np.where(D0.index.str.split(":", n=1).str[1].str.startswith("young"), "young", "old"),
                    index=D0.index)
    out = {}
    with sh.constants(c["pop_sd"], c["k0"], c["s_logk"]):
        def run(D, cell, rr):
            D, cell = keyed(D, rr, cell)
            Dp = post_group(D, rr, cell)
            return apply_m(Dp, rr, model, "vl", cell)

        rows = []
        for nm in ("muscle", "muscle_refit", "muscle_pa", "muscle_key"):
            rr = dict(r, groups=nm)
            empty = pd.Series("", index=D0.index, dtype=object)
            cell = ("vl|" + age).astype(object)
            Pp = run(D0, empty, rr)
            Pm = run(D0, cell, rr)
            ep, em = _score_vl(Pp, gt), _score_vl(Pm, gt)
            d = (em - ep)
            bt = b3.v4_boot(em, ep, cluster=pd.Series(d.index.str.split("__").str[1], index=d.index))
            rows.append(dict(variant=nm, setting="all 244 (young 78 / old 166)", pooled=float(ep.mean()),
                             muscle=float(em.mean()), diff=bt["diff"], lo=bt["lo"], hi=bt["hi"], p_neg=bt["p_neg"]))
        out["full"] = rows
        rng = np.random.default_rng(seed)
        first = D0.groupby(sub_of).head(10)          # all rows; per draw pick one image per subject
        draws = []
        for k in range(R):
            subj = rng.permutation(sub_of.unique())
            pick = pd.Series(first.index).groupby(sub_of[first.index].to_numpy()).apply(
                lambda s: s.iloc[rng.integers(len(s))]).reindex(subj).dropna().tolist()
            pick = pd.Index(pick)
            ya = [i for i in pick if age[i] == "young"]
            oa = [i for i in pick if age[i] == "old"]
            ny = no = 16
            nl = 7
            if len(ya) < ny + 4 or len(oa) < no + 4:
                continue
            ids_c = ya[:ny] + oa[:no]
            ids_l = ya[ny:ny + 3] + oa[no:no + nl - 3]
            ids_all = ids_c + ids_l
            D = D0.loc[ids_all]
            empty = pd.Series("", index=D.index, dtype=object)
            cell = pd.Series([("vl|" + age[i]) if i in ids_c else "" for i in D.index], index=D.index, dtype=object)
            rr = dict(r, groups="muscle")
            ep = _score_vl(run(D, empty, rr), gt)
            em = _score_vl(run(D, cell, rr), gt)
            lab = pd.Series(np.where([i in ids_c for i in D.index], "cell", "left"), index=[i.split(":", 1)[1]
                                                                                           for i in D.index])
            dd = em - ep
            draws.append(dict(draw=k, d_all=float(dd.mean()), d_cells=float(dd[lab == "cell"].mean()),
                              d_left=float(dd[lab == "left"].mean()), pooled=float(ep.mean())))
        out["draws"] = pd.DataFrame(draws)
    return out


def sim_synthetic(r: dict, R: int = 300, seed: int = 2) -> pd.DataFrame:
    """Semi-synthetic, test-calibrated muscle mixing on real VL errors (lodo3 model). A pseudo-scanner of
    Lumify size (VL 18, GM 19, RF 14, 7 unknown; 1 image per subject); for the 'GM' and 'RF' cells the
    true AND estimated values are moved by the test cell offsets of our raw estimates on Lumify
    (GM: PA +7.9 deg, FL x0.70; RF: PA -0.6, FL x1.27; unknown: a random half GM-like), error SDs scaled
    with FL, so per-image errors stay real and only the cell centres differ. Recipe C4 (FL head 'ens':
    a PA shift would not be geometric for the constructed-fascicle grid). Pooled vs muscle-aware."""
    gt = bc.vl_gt()
    rr = dict(r, fl="ens")
    B, model, c = _vl_frames(rr)
    D0 = B[B.dataset == "vl"]
    sub_of = pd.Series(D0.index.str.split(":", n=1).str[1].str.split("__").str[1], index=D0.index)
    with sh.constants(c["pop_sd"], c["k0"], c["s_logk"]):
        sig0 = model.predict(D0)
    shifts = {"VL": (0.0, 1.0), "GM": (7.9, 0.70), "RF": (-0.6, 1.27)}
    rng = np.random.default_rng(seed)
    rows = []
    with sh.constants(c["pop_sd"], c["k0"], c["s_logk"]):
        for k in range(R):
            pick = [rng.choice(g.index) for _, g in D0.groupby(sub_of)]
            pick = list(rng.permutation(pick))[:58]
            lab = ["VL"] * 18 + ["GM"] * 19 + ["RF"] * 14 + ["U"] * 7
            mus = {i: m for i, m in zip(pick, lab)}
            D = D0.loc[pick].copy()
            G = gt.reindex([i.split(":", 1)[1] for i in pick]).copy()
            G.index = D.index
            sig = sig0.loc[pick].copy()
            for i in pick:
                m = mus[i]
                if m == "U":
                    m = "GM" if rng.random() < 0.5 else "VL"
                dpa, f = shifts[m]
                for col, gc in (("pa_deg", "pa_deg"),):
                    D.at[i, col] = D.at[i, col] + dpa
                    G.at[i, gc] = G.at[i, gc] + dpa
                D.at[i, "fl_mm"] = D.at[i, "fl_mm"] * f
                G.at[i, "fl_mm"] = G.at[i, "fl_mm"] * f
                sig.at[i, "sigma_fl"] = sig.at[i, "sigma_fl"] * f
            cell = pd.Series(["vl|" + mus[i] if mus[i] != "U" else "" for i in D.index], index=D.index, dtype=object)
            empty = pd.Series("", index=D.index, dtype=object)
            res = {}
            for nm, cl in (("pooled", empty), ("muscle", cell), ("muscle_refit", cell), ("muscle_pa", cell),
                           ("muscle_key", cell)):
                rv = dict(rr, groups=nm if nm != "pooled" else "muscle")
                Dk, ck = keyed(D, rv, cl)
                P = apply_m(Dk, rv, model, "vl", ck, sigma=sig)
                res[nm] = ((P["pa_eb"] - G["pa_deg"]).abs() / 6 + (P["fl_eb"] - G["fl_mm"]).abs() / 12) / 2
                res[nm + ":pa"] = (P["pa_eb"] - G["pa_deg"]).abs() / 6
                res[nm + ":fl"] = (P["fl_eb"] - G["fl_mm"]).abs() / 12
            row = dict(draw=k, pooled=float(res["pooled"].mean()))
            for nm in ("muscle", "muscle_refit", "muscle_pa", "muscle_key"):
                d = res[nm] - res["pooled"]
                row[f"{nm}:d"] = float(d.mean())
                row[f"{nm}:d_PAcomp"] = float((res[nm + ":pa"] - res["pooled:pa"]).mean() / 2)
                row[f"{nm}:d_FLcomp"] = float((res[nm + ":fl"] - res["pooled:fl"]).mean() / 2)
                for m in ("VL", "GM", "RF", "U"):
                    msk = np.array([mus[i] == m for i in D.index])
                    row[f"{nm}:d_{m}"] = float(d[msk].mean())
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- bench
def offset_res(res: dict, off: dict) -> dict:
    out = {}
    for k, v in res.items():
        v = v.copy()
        v["pa_eb"] = v["pa_eb"] * off.get("pa_mult", 1.0) + off.get("pa_add", 0.0)
        v["fl_eb"] = v["fl_eb"] * off.get("fl_mult", 1.0) + off.get("fl_add", 0.0)
        v["mt_eb"] = v["mt_eb"] + off.get("mt_add", 0.0)
        out[k] = v
    return out


def md(df: pd.DataFrame, cols: list, nd: int = 3) -> str:
    return b3.md(df, cols, nd)


def regime_table() -> pd.DataFrame:
    """VL (lodo3 frame, C8 per-image FL): raw/true FL and PA error binned by TRUE and by RAW FL/FOV."""
    B, _ = bench_B(CANDS["C8"], "lodo3")
    D = B[B.dataset == "vl"].copy()
    g = bc.vl_gt().reindex(D.index.str.split(":", n=1).str[1])
    D["fl_true"], D["pa_true"] = g.fl_mm.to_numpy(), g.pa_deg.to_numpy()
    fov = sh._fov_mm(D)
    D["ratio"], D["epa"] = D.fl_mm / D.fl_true, D.pa_deg - D.pa_true
    rows = []
    for by, x in (("true FL/FOV", D.fl_true / fov), ("raw FL/FOV", D.fl_mm / fov)):
        for b, gg in D.groupby(pd.cut(x, [0, 1.5, 2, 2.25, 2.5, 10])):
            rows.append(dict(binned_by=by, bin=str(b), n=len(gg), raw_over_true_median=gg.ratio.median(),
                             cell_median_ratio=gg.fl_mm.median() / gg.fl_true.median() if len(gg) else np.nan,
                             pa_err_median=gg.epa.median()))
    return pd.DataFrame(rows)


def cmd_bench(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 80)
    t0 = time.time()
    rep: dict = {}
    # 1. identity checks: the v4 code path == the flfix / v3 code paths when nothing is muscle-labelled
    chk = {}
    for nm, r in (("C8 vs b3 C4pp", V("C8")), ("C6pp vs b3 C6pp", V("x", shrink="conf")),
                  ("C4 (ens) vs b3 C4", V("x", fl="ens"))):
        mine = run_bench_m(dict(r, groups="muscle"), "lodo")
        ref = b3.run_bench_v4(b3rec(r), "lodo")
        mx = 0.0
        for k in ref:
            a, b = b3._q(mine[k]), b3._q(ref[k])
            mx = max(mx, float(np.nanmax(np.abs(a.to_numpy() - b.reindex(a.index).to_numpy()))))
        chk[nm] = mx
    for nm, r, f in (("test C8 vs flfix_C4pp.csv", V("C8"), ROOT / "artifacts/eval_v4/flfix/flfix_C4pp.csv"),
                     ("test D_C6pp vs flfix_C6pp.csv", V("x", shrink="conf"), ROOT / "artifacts/eval_v4/flfix/flfix_C6pp.csv"),
                     ("test C4 path vs C4_v2_fusF_ens_hyb_med.csv", V("x", fl="ens"),
                      ROOT / "artifacts/candidates_v3/C4_v2_fusF_ens_hyb_med.csv")):
        s = test_run(r)[0]
        chk[nm] = bc.official_distance(s, pd.read_csv(f).set_index("image_id"))["S"]
    rep["identity"] = chk
    print("identity checks (0 = identical):", json.dumps(chk, indent=1), flush=True)
    regime_table().to_csv(OUT / "vl_fl_bias_by_fov.csv", index=False)
    # 2. full benchmark table (LODO constants), all recipes of interest
    recs = {"C2": None, "C4": V("C4", fl="ens"), "C6": V("C6", fl="ens", shrink="conf"),
            "C8": CANDS["C8"], "C6pp": DIAG["D_C6pp"], "C8+offset (= C7 on benchmarks)": DIAG["D_C8_offset"]}
    rows = []
    for nm, r in recs.items():
        if r is None:
            res = b3.run_bench_v4(b3.R("ours", "fusion", "c1", "conf", name="C2"), "lodo")
            off = None
        else:
            res = bench_preds(r)["res"]
            off = offset_of(r)
        if off:   # offsets act after shrinkage: score the offset predictions
            res = offset_res(res, off)
        sc = b3.v4_score(res)
        rows.append(dict(name=nm, **sc))
    TB = pd.DataFrame(rows)
    TB.to_csv(OUT / "bench_v4.csv", index=False)
    cols = ["name", "s35", "c10", "in25", "2r", "2r_c10", "R7", "vl3", "vl3_pa", "vl3_fl", "vl4", "vl_young3",
            "dyn_pa_h", "dyn_fl"]
    print(TB[cols].round(4).to_string(index=False), flush=True)
    # 3. selection score (pre-registered) with LB sampling SDs
    sel = {"C2": dict(V("C2", apo="ours", pa="fusion", fl="c1", shrink="conf"), desc="C2 recipe (LB 0.390)"),
           "C4": dict(V("C4", fl="ens"), desc="v3 C4"), "C6": dict(V("C6", fl="ens", shrink="conf"), desc="v3 C6"),
           "C8": CANDS["C8"], "C9": CANDS["C9"], "C7": CANDS["C7"], "C6pp": DIAG["D_C6pp"],
           "C8off": DIAG["D_C8_offset"]}
    Tc, Tp = select(sel, B=args.B, tag="v4")
    show = ["cand", "c10_2r", "c10_2r_loo", "all35_2r", "vl3", "dyn", "t142", "S_sel", "S_sel_loo"]
    print(Tc[show].round(4).to_string(index=False))
    print(Tp[["pair", "d_c10_2r_loo", "d_vl3", "d_vl3_lo", "d_vl3_hi", "d_dyn", "d_t142", "d_S_sel_loo",
              "d_S_sel_loo_lo", "d_S_sel_loo_hi", "P_S_sel_loo", "test_dist", "rho_vl", "sd_pub_rho06",
              "sd_pub_rho1", "sd_pub_priv_rho1"]].round(4).to_string(index=False), flush=True)
    # 4. simulations of muscle-aware grouping
    sv = sim_vl_split(CANDS["C8"], R=args.R)
    full = pd.DataFrame(sv["full"])
    full.to_csv(OUT / "sim_vl_split_full.csv", index=False)
    dr = sv["draws"]
    dr.to_csv(OUT / "sim_vl_split_draws.csv", index=False)
    print(full.round(4).to_string(index=False))
    q = lambda s: (float(s.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)),  # noqa: E731
                   float((s < 0).mean()))
    rep["sim_vl_split_draws"] = {k: q(dr[k]) for k in ("d_all", "d_cells", "d_left")}
    rep["sim_vl_split_draws"]["n_draws"] = int(len(dr))
    print("VL young/old subsampled (mean, 2.5%, 97.5%, P(muscle better)):", rep["sim_vl_split_draws"], flush=True)
    sy = sim_synthetic(CANDS["C8"], R=args.R)
    sy.to_csv(OUT / "sim_synthetic_draws.csv", index=False)
    rep["sim_synthetic"] = {k: q(sy[k]) for k in sy.columns if ":" in k}
    rep["sim_synthetic"]["pooled_mean"] = float(sy.pooled.mean())
    print("semi-synthetic Lumify-like mixing:", rep["sim_synthetic"], flush=True)
    rep["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    rep["seconds"] = time.time() - t0
    (OUT / "bench_v4_report.json").write_text(json.dumps(rep, indent=1, default=float))
    lines = ["# v4 benchmarks (LODO constants; lower is better)", "", md(TB, cols, 4), "",
             "## Selection score (rater agent's pre-registered S_sel; LODO constants)", "",
             md(Tc, show, 4), "", "## Pairs", "",
             md(Tp, ["pair", "d_c10_2r_loo", "d_vl3", "d_dyn", "d_t142", "d_S_sel_loo", "d_S_sel_loo_lo",
                     "d_S_sel_loo_hi", "P_S_sel_loo", "test_dist", "rho_vl", "sd_pub_rho06", "sd_pub_rho1",
                     "sd_pub_priv_rho1"], 4), "",
             "## Muscle-grouping simulation: VL young/old as the subgroup label", "", md(full, list(full.columns), 4)]
    (OUT / "bench_v4.md").write_text("\n".join(lines) + "\n")
    print(f"peak RSS {rep['peak_rss_gb']:.2f} GB, {rep['seconds']:.0f}s")


# ----------------------------------------------------------------------------------- build
def group_table(sub: pd.DataFrame, P: pd.DataFrame, cell: pd.Series) -> pd.DataFrame:
    lay = b3.layout()
    T = ml.load_test_muscle().reindex(sub.index)
    m, _ = ml._trusted(T.fillna(""), 0.9, False)
    d = pd.DataFrame({"group": lay.loc[sub.index, "group"], "muscle": m.replace("", "unknown"),
                      "cell": cell.reindex(sub.index).replace("", "(scanner)"),
                      "pa": sub.pa_deg, "fl": sub.fl_mm, "mt": sub.mt_mm,
                      "k": sub.fl_mm * np.sin(np.radians(sub.pa_deg)) / sub.mt_mm,
                      "pa_raw": P["pa_deep_raw"].reindex(sub.index), "fl_raw": P["fl_mm"].reindex(sub.index),
                      "w_pa": P.get("w_pa", pd.Series(np.nan, index=P.index)).reindex(sub.index),
                      "w_fl": P.get("w_fl", pd.Series(np.nan, index=P.index)).reindex(sub.index)})
    g1 = d.groupby("group").agg(n=("pa", "size"), pa=("pa", "median"), fl=("fl", "median"), mt=("mt", "median"),
                                k=("k", "median")).assign(muscle="ALL")
    g2 = d.groupby(["group", "muscle"]).agg(n=("pa", "size"), pa=("pa", "median"), fl=("fl", "median"),
                                            mt=("mt", "median"), k=("k", "median"), pa_raw=("pa_raw", "median"),
                                            fl_raw=("fl_raw", "median"), w_pa=("w_pa", "mean"),
                                            w_fl=("w_fl", "mean")).reset_index("muscle")
    return pd.concat([g1, g2]).reset_index().sort_values(["group", "muscle"])


def cmd_build(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostic").mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    refs = {"C2": pd.read_csv(b3.REFS["C2"][0]).set_index("image_id"),
            "C4": pd.read_csv(OUT.parent / "candidates_v3/C4_v2_fusF_ens_hyb_med.csv").set_index("image_id"),
            "C6": pd.read_csv(OUT.parent / "candidates_v3/C6_v2_fusF_ens_conf.csv").set_index("image_id"),
            "C1": pd.read_csv(b3.REFS["C1"][0]).set_index("image_id")}
    allc = {**CANDS, **DIAG}
    subs, rep, tabs = {}, {}, []
    for nm, r in allc.items():
        sub, P, cell = test_run(r)
        path = (OUT if nm in CANDS else OUT / "diagnostic") / f"{nm}_{r['shrink']}_{r['fl']}_{r['groups']}" \
            f"{'_off' if r.get('offset') else ''}.csv"
        bc.write_sub(sub, path)
        subs[nm] = sub
        g = group_table(sub, P, cell)
        g.insert(0, "cand", nm)
        tabs.append(g)
        pri = {k: v for k, v in P.attrs.get("priors", {}).items() if k.endswith("|fl") or "demeaned" in k}
        rep[nm] = dict(file=str(path.relative_to(ROOT)), desc=r["desc"], recipe={k: v for k, v in r.items()
                                                                                  if k != "desc"},
                       offset=offset_of(r), validate=b3.validate(path),
                       n_cell_images=int((cell != "").sum()), cells=cell[cell != ""].value_counts().to_dict(),
                       dist={k: bc.official_distance(sub, v) for k, v in refs.items()}, fl_priors=pri)
        print(f"== {nm}: {rep[nm]['validate']} | cells {rep[nm]['cells']}")
        for k in ("C2", "C4", "C6"):
            print(f"   vs {k}: {rep[nm]['dist'][k]}")
        if nm in CANDS:
            Q = P[[c for c in ("group", "clip", "cell", "confidence", "pa_deep_raw", "pa_deep_c", "pa_deg", "fl_mm",
                               "mt_mm", "sigma_fl", "mu_fl", "tau_fl", "c_fl", "w_fl", "w_pa", "pa_eb", "fl_eb")
                   if c in P]].copy()
            Q["sub_pa_deg"], Q["sub_fl_mm"], Q["sub_mt_mm"] = sub.pa_deg, sub.fl_mm, sub.mt_mm
            Q.index.name = "image_id"
            Q.to_csv(OUT / f"per_image_{nm}.csv")
    G = pd.concat(tabs)
    G.to_csv(OUT / "test_groups_by_muscle.csv", index=False)
    # pairwise distances + LB sampling SDs (rho = 0.6 typical on VL, 1.0 upper bound)
    names = list(CANDS) + ["D_C6pp", "D_C8_offset", "D_C4_muscle"]
    full = {**{k: subs[k] for k in names}, **{k: v for k, v in refs.items()}}
    prow = []
    keys = list(full)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            d = bc.official_distance(full[a], full[b])
            s6, s1 = lb_sd(full[a], full[b], 0.6), lb_sd(full[a], full[b], 1.0)
            prow.append(dict(a=a, b=b, S=d["S"], pa_n=d["pa_n"], fl_n=d["fl_n"], mt_n=d["mt_n"],
                             sd_pub_rho06=s6["sd_pub"], sd_pub_rho1=s1["sd_pub"],
                             sd_pub_priv_rho06=s6["sd_pub_priv"], sd_pub_priv_rho1=s1["sd_pub_priv"]))
    PW = pd.DataFrame(prow)
    PW.to_csv(OUT / "pairwise_distances.csv", index=False)
    print(PW.round(4).to_string(index=False))
    # where the C9 - C8 change sits (per scanner x muscle)
    lay = b3.layout()
    T = ml.load_test_muscle().reindex(subs["C8"].index)
    m, _ = ml._trusted(T.fillna(""), 0.9, False)
    ch = []
    for a, b in (("C9", "C8"), ("C7", "C8"), ("C7", "C9"), ("D_C4_muscle", "C4")):
        A, Bq = subs[a] if a in subs else refs[a], subs[b] if b in subs else refs[b]
        dd = A - Bq.reindex(A.index)
        dd["group"], dd["muscle"] = lay.loc[A.index, "group"], m.replace("", "unknown")
        x = dd.groupby(["group", "muscle"]).agg(n=("pa_deg", "size"), d_pa_mean=("pa_deg", "mean"),
                                                d_fl_mean=("fl_mm", "mean"), d_mt_mean=("mt_mm", "mean"),
                                                d_pa_absmax=("pa_deg", lambda s: s.abs().max()),
                                                d_fl_absmax=("fl_mm", lambda s: s.abs().max())).reset_index()
        x.insert(0, "pair", f"{a}-{b}")
        ch.append(x)
    CH = pd.concat(ch)
    CH.to_csv(OUT / "test_change_by_muscle.csv", index=False)
    print(CH.round(3).to_string(index=False))
    # the stake of each change: expected official-score change if the shift b_i = a_i - b_i is exactly right
    # (+) or entirely unneeded (-), Laplace errors at test-like scales (rater script TEST_S middle values)
    st = []
    S_ = {"pa_deg": 1.8, "fl_mm": 6.7, "mt_mm": 0.8}
    for a, b in (("C9", "C8"), ("D_C9_ungated", "C8"), ("D_C9_paonly", "C8"), ("C7", "C9"), ("C7", "C8"),
                 ("C8", "C4"), ("C8", "C2")):
        A = subs.get(a, refs.get(a))
        Bq = (subs.get(b) if b in subs else refs[b]).reindex(A.index)
        row = dict(a=a, b=b)
        tot = 0.0
        for c in C3:
            sc = S_[c] * (A[c] / 65.0 if c == "fl_mm" else 1.0)     # FL error scale proportional to FL
            bb = (A[c] - Bq[c]).to_numpy(float)
            f = lambda x, s_: np.abs(x) + s_ * np.exp(-np.abs(x) / s_)  # noqa: E731
            g = float(np.mean(f(bb, sc.to_numpy(float) if hasattr(sc, "to_numpy") else sc) - f(0 * bb, sc.to_numpy(float)
                      if hasattr(sc, "to_numpy") else sc))) / TAU[c] / 3
            row[f"stake_{c[:2]}"] = g
            tot += g
        row["stake_total"] = tot
        st.append(row)
    ST = pd.DataFrame(st)
    ST.to_csv(OUT / "stake_by_change.csv", index=False)
    print(ST.round(4).to_string(index=False))
    rep["stake"] = ST.to_dict("records")
    rep["pairwise"] = PW.to_dict("records")
    (OUT / "candidates_v4_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(G[G.cand.isin(list(CANDS))].round(3).to_string(index=False))
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")


# ===================================================================================== v5 (fixes5)
# critic4 fixes 1, 2, 3 and 7 on top of C7. Additive: the v4 subcommands above are unchanged, and the v5
# path with the v4 settings reproduces C7 / C8 / C9 and every v4 benchmark prediction exactly (v5bench
# identity checks). Outputs go to artifacts/candidates_v5/.
#  fix 1  gate="fl": the FL/FOV support gate removes only the FL prior centre of an unsupported cell; its PA
#         centre is kept (the gate's reason, FL extrapolation beyond the view, does not apply to PA).
#         On test: Lumify RF (14 images) gets its own PA centre. gate="both" = v4.
#  fix 2  mtab="v5": the muscle table re-parsed from the cached OCR words by the fixed reader
#         (umud.muscle_label: no 'k' in the GM confusion class; a confusion word on another text line than a
#         clean read does not vote; ambiguous reads are flagged and not trusted) -> candidates_v5/muscle/.
#         On test: IMG_00209 only (overprinted label; GM 0.986 -> 0.881, ambiguous -> leftover).
#  fix 3  flc: MT-aware FL EB prior centre (sh.shrink_fl_centred) with z = log(MT / sin PA_out):
#         "lodoB" log c = log base + beta (z - med z), beta = leak-free predictive slope of log expert FL on z
#                 (sh.fit_centre_slope on the labelled benchmarks minus the held-out dataset and minus the
#                 25 in-training static images; group / cell level kept);
#         "grp1"  the same with beta = 1 (the group's own FL sin PA / MT ratio times MT / sin PA);
#         "k0"    c = K0 MT / sin PA (K0 = the leak-free expert constant on benchmarks; sets the level);
#         "zero"  beta = 0 (identity check: must equal flc=None).
#         Every centre is clipped to base x exp(+-FLC_CLIP). (tau, c) stay those of the group fit.
#  fix 7  em="clean": the EB error model is fitted without the 25 in-training static images.
#
# Decision rules (fixed in this docstring BEFORE any v5 benchmark was scored; decisions from benchmarks,
# never the LB). d = variant - base, S_sel = the pre-registered selection score with the leave-one-out
# clean10 term (S_sel_loo), paired bootstrap 95% CIs.
#  R7 (fix 7, base C7ref): ADOPT unless d S_sel_loo > 0 with its CI excluding 0 (it removes optimistic
#     training residuals, so the burden of proof is on harm).
#  R3 (fix 3, base = C7ref + the R7 decision): a variant is ADOPTED only if d S_sel_loo < 0 with its CI
#     excluding 0 AND none of VL lodo3, VL lodo4, clean10 two-rater (LOO) and dynamic is worse with a CI
#     excluding 0. Several passing -> the lowest S_sel_loo. None -> fix 3 rejected (C12 without it).
#  R1/R2 (fixes 1, 2): not measurable (no benchmark has muscle labels; exact no-ops, checked). ADOPTED as
#     defect corrections (critic4), with their test stakes reported.
OUT5 = ROOT / "artifacts/candidates_v5"
MUSCLE5_DIR = OUT5 / "muscle"
MUSCLE5 = MUSCLE5_DIR / "test_muscle.csv"
OFFSETS5_JSON = OUT5 / "rater/recommended_offsets_C12.json"
DECISIONS5 = OUT5 / "decisions_v5.json"
FLC_CLIP = 0.4
FLC = {"lodoB": dict(level="group"), "grp1": dict(level="group", beta=1.0), "k0": dict(level="k0"),
       "zero": dict(level="group", beta=0.0)}
REF_FILES5 = {"C7": OUT / "C7_hyb_med_ens_pp_muscle_off.csv", "C8": OUT / "C8_hyb_med_ens_pp_scanner.csv",
              "C9": OUT / "C9_hyb_med_ens_pp_muscle.csv", "C11": OUT / "C11_fullshrink_cellmedian.csv",
              "C10": OUT / "diagnostic/D_C6pp_muscle_off_conf_ens_pp_muscle_off.csv"}


def V5(name: str, groups: str = "muscle", offset="rec", mtab: str = "v5", gate: str = "fl", flc: str | None = None,
       em: str = "clean", shrink: str = "hyb_med", fl: str = "ens_pp", desc: str = "") -> dict:
    """v5 recipe = a v4 recipe + mtab (v4|v5), gate (both|fl), flc (None|lodoB|grp1|k0|zero), em (all|clean).
    offset: as V, plus "rec5" = the offsets re-estimated for the C12 base (OFFSETS5_JSON)."""
    r = V(name, groups=groups, shrink=shrink, fl=fl, offset=offset, desc=desc)
    r.update(v5=True, mtab=mtab, gate=gate, flc=flc, em=em)
    return r


def offset_of5(r: dict) -> dict | None:
    if r.get("offset") == "rec5":
        return json.loads(OFFSETS5_JSON.read_text())["offsets"]
    return _offset_of_v4(r)      # the v4 function (offset_of itself is rerouted here inside _v5_scope)


def _v5key(r: dict) -> str:
    return json.dumps({k: r.get(k) for k in ("apo", "pa", "fl", "shrink", "groups", "mtab", "gate", "flc", "em")},
                      sort_keys=True)


# ----------------------------------------------------------------------------------- v5 pieces
def cell_keys5(index: pd.Index, scanner: pd.Series, r: dict, ratio: pd.Series | None = None,
               min_n: int = MIN_N) -> tuple[pd.Series, pd.Series]:
    """(pa_cell, fl_cell) per test image ('' = leftover). cell_keys with the table of r['mtab'] and, for
    gate='fl', the regime gate applied to the FL cell only (fix 1)."""
    empty = pd.Series("", index=index, dtype=object)
    md_ = MODES[r["groups"]]
    if not md_:
        return empty, empty.copy()
    if md_.get("key"):
        raise ValueError("muscle_key recipes are not supported on the v5 path")
    t = ml.load_test_muscle(MUSCLE5 if r.get("mtab") == "v5" else None).reindex(index)
    if t["muscle"].isna().all():
        return empty, empty.copy()
    m, _ = ml._trusted(t.fillna(""), 0.9, md_.get("inferred", False))
    base = scanner.reindex(index).astype(str)
    if md_.get("block"):
        base = t["block"].astype(str).where(base == "siemens_juniper", base)
    key = base + "|" + m
    n = key.map(key[m != ""].value_counts()).fillna(0)
    key = key.where((m != "") & (n >= min_n), "").astype(object)
    gated = key
    if md_.get("gate") and ratio is not None:
        med = ratio.reindex(index)[key != ""].groupby(key[key != ""]).median()
        bad = set(med.index[med > GATE])
        gated = key.where(~key.isin(bad), "").astype(object)
    fl_cell = gated if md_.get("fl", True) else empty.copy()
    pa_cell = key if r.get("gate") == "fl" else gated
    return pa_cell, fl_cell


def zlog(D: pd.DataFrame) -> pd.Series:
    """log(MT / sin PA_out) per row (PA_out = pa_deep_c, the output PA with the deep reference; >= 5 deg)."""
    pa = pd.to_numeric(D["pa_deep_c"], errors="coerce").clip(lower=5.0)
    return np.log(b3._col(D, "mt_mm") / np.sin(np.radians(pa)))


def _not_in25(B: pd.DataFrame) -> pd.Series:
    iid = B["image_id"] if "image_id" in B else pd.Series(B.index.astype(str), index=B.index)
    return (B["dataset"].astype(str) != "static") | iid.isin(bc.CLEAN)


def fit_em5(Btr: pd.DataFrame, r: dict) -> sh.ErrorModel | None:
    """EB error model on the training benchmark rows; em='clean' drops the 25 in-training static images."""
    if r["shrink"] != "hyb_med":
        return None
    if r.get("em") == "clean":
        keep = _not_in25(Btr)
        if (Btr["dataset"] == "static").any() and int((keep & (Btr["dataset"] == "static")).sum()) != len(bc.CLEAN):
            raise RuntimeError("clean static rows not found")
        Btr = Btr[keep]
    return sh.fit_error_model(Btr, **b3.FIT_KW)


def flc_params(r: dict, Btr: pd.DataFrame) -> dict | None:
    """Fix-3 centre settings for one fit (Btr = the labelled training rows; leak-free by construction)."""
    f = r.get("flc")
    if not f:
        return None
    p = dict(FLC[f])
    if f == "lodoB":
        s = sh.fit_centre_slope(Btr, zlog(Btr), keep=_not_in25(Btr))
        p.update(beta=s["beta"], fit=s)
    elif f == "k0":
        p.update(beta=0.0, k0=float(sh.K0))
    return p


def apply_m5(D: pd.DataFrame, r: dict, model, pa_cell: pd.Series, fl_cell: pd.Series, fc: dict | None) -> pd.DataFrame:
    """apply_m with separate PA / FL cells (fix 1) and the optional MT-aware FL centre (fix 3)."""
    import eval_shrinkage as es
    pa_cell = pa_cell.reindex(D.index).fillna("")
    fl_cell = fl_cell.reindex(D.index).fillna("")
    conf = b3._col(D, "confidence")
    grp = D["group"].astype(str)
    if r["shrink"] == "hyb_med":
        if fc:
            P = sh.shrink_fl_centred(D, model, zlog(D), beta=fc.get("beta", 0.0), level=fc["level"], k0=fc.get("k0"),
                                     centre_key=fl_cell, clip_dev=FLC_CLIP)
            P["cell"] = fl_cell
            P["c_fl"] = grp.map({k.split("|")[0]: v["c"] for k, v in P.attrs["priors"].items() if k.endswith("|fl")})
        else:
            P = eb_fl(D, model, fl_cell, refit=False)
        P["pa_eb"] = conf_rule(b3._col(D, "pa_deg"), conf, grp, pa_cell)
        P["w_pa"] = conf.fillna(0).clip(0, 1)
    elif r["shrink"] == "conf":
        if fc:
            raise ValueError("the MT-aware centre is implemented for the EB FL shrinker only")
        P = D.copy()
        for c, col in sh.COMP.items():
            P[f"{c}_eb"] = b3._col(D, col)
        P["pa_eb"] = conf_rule(b3._col(D, "pa_deg"), conf, grp, pa_cell)
        P["fl_eb"] = conf_rule(b3._col(D, "fl_mm"), conf, grp, fl_cell)
        P["cell"] = fl_cell
    else:
        raise ValueError(r["shrink"])
    P["pa_cell"] = pa_cell
    return es.fill_group(P)


_FLC_INFO: dict = {}


def run_bench5(r: dict, consts: str = "lodo") -> dict:
    """run_bench_m through the v5 path (no muscle labels on any benchmark: cells are empty)."""
    out = {}
    for split in ("lodo4", "lodo3"):
        B = b3.bench_frame(b3rec(r), split)
        empty = pd.Series("", index=B.index, dtype=object)
        B = post_group(B, r, empty)
        for ds in B.dataset.unique():
            if split == "lodo3" and ds != "vl":
                continue
            D = B[B.dataset == ds]
            Btr = B[B.dataset != ds]
            ctx = contextlib.nullcontext()
            if consts == "lodo":
                c = b3.v4_constants(b3.V4_HOLD[ds])
                ctx = sh.constants(c["pop_sd"], c["k0"], c["s_logk"])
            with ctx:
                model = fit_em5(Btr, r)
                fc = flc_params(r, Btr)
                out[(split, ds)] = apply_m5(D, r, model, empty, empty, fc)
            _FLC_INFO[(_v5key(r), consts, split, ds)] = fc
    return out


def train142_q5(r: dict) -> pd.DataFrame:
    x, _, _ = rc.train142_frame()
    rr = b3rec(r)
    d = b3.est_frame(x, "train142", rr)
    lay = x["layout"].astype(str).str.replace("1234x687", "1234x688")
    d["group"] = lay.where(lay.map(lay.value_counts()) >= 5, "other")
    d["dataset"], d["clip"] = "train142", -1
    empty = pd.Series("", index=d.index, dtype=object)
    d = post_group(d, r, empty)
    Bf = b3.bench_frame_v4(rr, "lodo4")
    P = apply_m5(d, r, fit_em5(Bf, r), empty, empty, flc_params(r, Bf))
    return pd.DataFrame({c: P[f"{k}_eb"].astype(float) for k, c in sh.COMP.items()}, index=P.index)


_BP5: dict = {}
_bench_preds_v4, _test_run_v4, _offset_of_v4, _OUT_v4 = bench_preds, test_run, offset_of, OUT


def bench_preds5(r: dict, consts: str = "lodo") -> dict:
    """bench_preds for v5 recipes (v4 recipes are passed to the v4 function)."""
    if not r.get("v5"):
        return _bench_preds_v4(r, consts)
    key = _v5key(r) + consts
    if key not in _BP5:
        res = run_bench5(r, consts)
        _BP5[key] = dict(res=res, static=b3._q(res[("lodo4", "static")]), vl3=b3._q(res[("lodo3", "vl")]),
                         vl4=pd.concat([b3._q(res[("lodo4", "vl_young")]), b3._q(res[("lodo4", "vl_old")])]),
                         dyn=b3._q(res[("lodo4", "dynamic")]), t142=train142_q5(r))
    p = _BP5[key]
    off = offset_of5(r)
    return {k: (rc.apply_offset(v, off) if k != "res" else v) for k, v in p.items()}


def test_run5(r: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """test_run through the v5 path -> (submission, per-image frame with pa_cell / cell, pa_cell)."""
    if not r.get("v5"):
        return _test_run_v4(r)
    import eval_shrinkage as es
    rr = b3rec(r)
    T = b3.test_frame(rr)
    pa_cell, fl_cell = cell_keys5(T.index, T["group"], r, ratio=b3._col(T, "fl_mm") / sh._fov_mm(T))
    T = post_group(T, r, pa_cell)
    Bf = b3.bench_frame_v4(rr, "lodo4")
    fc = flc_params(r, Bf)
    P = apply_m5(T, r, fit_em5(Bf, r), pa_cell, fl_cell, fc)
    P.attrs["flc"] = fc
    sub = es.to_sub(P)
    off = offset_of5(r)
    if off:
        sub = rc.apply_offset(sub, off)
        for c in C3:
            sub[c] = sub[c].clip(*bc.BOUNDS[c])
    return sub, P, pa_cell


@contextlib.contextmanager
def _v5_scope():
    """Route the v4 selection code (select, honest_c10_offset) through the v5 functions and write to OUT5."""
    g = globals()
    old = {k: g[k] for k in ("bench_preds", "test_run", "offset_of", "OUT")}
    g.update(bench_preds=bench_preds5, test_run=test_run5, offset_of=offset_of5, OUT=OUT5)
    try:
        yield
    finally:
        g.update(old)


def bench_res5(r: dict) -> dict:
    """Benchmark predictions per (split, dataset) after the recipe's offset (for v4_score / v4_per_image)."""
    res = bench_preds5(r)["res"]
    off = offset_of5(r)
    return offset_res(res, off) if off else res


# ----------------------------------------------------------------------------------- v5 recipes
REF5 = V5("C7ref", mtab="v4", gate="both", flc=None, em="all",
          desc="C7 through the v5 path (v4 settings); must reproduce C7_hyb_med_ens_pp_muscle_off.csv")


def v5_factorial() -> dict:
    """Single fixes on top of C7ref and the fix-3 variants on top of C7ref + fix 7 (the R3 base)."""
    R = {"C7ref": REF5,
         "F1": V5("F1", mtab="v4", gate="fl", em="all", desc="C7 + fix 1 (gate the FL centre only)"),
         "F2": V5("F2", mtab="v5", gate="both", em="all", desc="C7 + fix 2 (OCR hygiene; IMG_00209 ambiguous)"),
         "F7": V5("F7", mtab="v4", gate="both", em="clean", desc="C7 + fix 7 (error model without in25)"),
         "F127": V5("F127", desc="C7 + fixes 1, 2, 7 (= C12 if fix 3 is rejected)"),
         "F12": V5("F12", em="all", desc="C7 + fixes 1, 2 (= C12 if fixes 3 and 7 are rejected)")}
    for f in ("lodoB", "grp1", "k0", "zero"):
        R[f"F7_{f}"] = V5(f"F7_{f}", mtab="v4", gate="both", em="clean", flc=f,
                          desc=f"C7 + fix 7 + fix 3 ({f})")
        R[f"C7_{f}"] = V5(f"C7_{f}", mtab="v4", gate="both", em="all", flc=f, desc=f"C7 + fix 3 ({f})")
    return R


# ----------------------------------------------------------------------------------- v5 muscle (fix 2)
def cmd_v5muscle(args) -> None:
    """Re-parse the cached OCR words (artifacts/eval_v4/muscle/ocr_*words.csv; no new OCR) with the fixed
    reader and re-run the label-free inference rules -> candidates_v5/muscle/{ocr_readings,test_muscle,
    infer_eval}.* + diff_vs_v4.csv. Reads each test image once (layout), one at a time."""
    import read_muscle_labels as rlm
    MUSCLE5_DIR.mkdir(parents=True, exist_ok=True)
    src, orig_words, orig_out = rlm.OUT, rlm._words, rlm.OUT
    try:
        rlm._words = lambda f: orig_words(src / Path(f).name)
        rlm.OUT = MUSCLE5_DIR
        R = rlm.cmd_read(args)
        T = rlm.cmd_infer(args)
    finally:
        rlm._words, rlm.OUT = orig_words, orig_out
    amb = R["ambiguous"].reindex(T.index).astype(str).str.lower().eq("true")
    T["ambiguous"] = amb & T["method"].astype(str).str.startswith("ocr")
    T.to_csv(MUSCLE5)
    T0 = ml.load_test_muscle()
    T1 = ml.load_test_muscle(MUSCLE5)
    m0, _ = ml._trusted(T0, 0.9, False)
    m1, _ = ml._trusted(T1, 0.9, False)
    cols = ["muscle", "confidence", "method", "ocr_raw"]
    D = T0[cols].add_suffix("_v4").join(T1[cols + ["ambiguous"]].add_suffix("_v5"))
    D["trusted_v4"], D["trusted_v5"] = m0, m1
    ch = D[(D.muscle_v4 != D.muscle_v5) | (D.confidence_v4.astype(str) != D.confidence_v5.astype(str))
           | (D.method_v4 != D.method_v5) | (D.trusted_v4 != D.trusted_v5) | T1["ambiguous"].astype(str).eq("True")]
    ch.to_csv(MUSCLE5_DIR / "diff_vs_v4.csv")
    print("changed rows vs the v4 table:\n", ch.to_string())


# ----------------------------------------------------------------------------------- v5 offsets
def c12_base(dec: dict | None = None) -> dict:
    """The C12 recipe without offsets, from the v5bench decisions (fixes 1 and 2 always; 7 and 3 per rule)."""
    dec = dec or json.loads(DECISIONS5.read_text())
    return V5("C12", offset=None, em="clean" if dec["fix7_adopt"] else "all", flc=dec.get("fix3_variant"),
              desc="C7 + adopted critic4 fixes")


def cmd_v5offsets(args) -> None:
    """The rater agent's pre-registered offset rule (rater_calibration.cmd_offsets) on the C12 base recipe."""
    (OUT5 / "rater").mkdir(parents=True, exist_ok=True)
    r = c12_base()
    rc.OUT = OUT5 / "rater"
    rc.rec_of = lambda name, spec=None: dict(r, offset=None)  # noqa: E731
    rc.bench_preds = lambda rec: {k: v for k, v in bench_preds5(dict(rec, offset=None)).items()
                                  if k in ("static", "vl3", "dyn", "t142")}
    rc.cmd_offsets(argparse.Namespace(recipe="C12", B=args.B, seed=0))


# ----------------------------------------------------------------------------------- v5 bench
BENCH_COLS5 = ["name", "s35", "c10", "in25", "2r", "2r_c10", "R7", "vl3", "vl3_pa", "vl3_fl", "vl4", "vl_young3",
               "dyn_pa_h", "dyn_fl"]
SEL_SHOW5 = ["cand", "c10_2r", "c10_2r_loo", "vl3", "dyn", "t142", "S_sel", "S_sel_loo", "S_sel_loo_lo", "S_sel_loo_hi"]


def boot_vs(a: dict, b: dict, name_a: str, name_b: str) -> pd.DataFrame:
    """Paired bootstrap CIs of S(a) - S(b) on every benchmark slice (b3.v4_boot_table; negative = a better)."""
    return pd.DataFrame(b3.v4_boot_table(b3.v4_per_image(bench_res5(a)), b3.v4_per_image(bench_res5(b)),
                                         name_a, name_b))


def _pair(Tp: pd.DataFrame, a: str, b: str) -> pd.Series:
    row = Tp[Tp.pair == f"{a}-{b}"]
    if len(row):
        return row.iloc[0]
    row = Tp[Tp.pair == f"{b}-{a}"].iloc[0].copy()
    for c in row.index:
        if c.startswith("d_") and not c.endswith(("_lo", "_hi")):
            lo, hi = row[f"{c}_lo"], row[f"{c}_hi"]
            row[c], row[f"{c}_lo"], row[f"{c}_hi"] = -row[c], -hi, -lo
    return row


def cmd_v5bench(args) -> None:
    OUT5.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 80)
    t0 = time.time()
    rep: dict = {"rules": "see the v5 section docstring of scripts/build_candidates_v4.py (fixed before scoring)"}
    R = v5_factorial()
    # 0. identity checks
    chk = {}
    for nm, r5, r4 in (("bench C7ref vs v4 C7", REF5, CANDS["C7"]),
                       ("bench C8 path vs v4 C8", V5("x", groups="scanner", mtab="v4", gate="both", em="all"),
                        CANDS["C8"])):
        a, b = run_bench5(r5), run_bench_m(r4)
        chk[nm] = max(float(np.nanmax(np.abs(b3._q(a[k]).to_numpy() - b3._q(b[k]).reindex(b3._q(a[k]).index).to_numpy())))
                      for k in b)
    for nm, r5 in (("bench flc=zero vs flc=None (fix 7 on)", R["F7_zero"]),):
        a, b = run_bench5(r5), run_bench5(R["F7"])
        chk[nm] = max(float(np.nanmax(np.abs(b3._q(a[k]).to_numpy() - b3._q(b[k]).to_numpy()))) for k in b)
    for nm, r5, f in (("test C7ref vs C7 csv", REF5, REF_FILES5["C7"]),
                      ("test C9 path vs C9 csv", V5("x", offset=None, mtab="v4", gate="both", em="all"), REF_FILES5["C9"]),
                      ("test C8 path vs C8 csv", V5("x", groups="scanner", offset=None, mtab="v4", gate="both", em="all"),
                       REF_FILES5["C8"]),
                      ("test flc=zero vs F127 + zero", V5("x", flc="zero"), None)):
        s = test_run5(r5)[0]
        other = test_run5(V5("x"))[0] if f is None else pd.read_csv(f).set_index("image_id")
        chk[nm] = float(np.nanmax(np.abs(s[list(C3)].to_numpy() - other.reindex(s.index)[list(C3)].to_numpy())))
    rep["identity_max_abs"] = chk
    print("identity checks (max |diff|, 0 = identical):", json.dumps(chk, indent=1), flush=True)
    # 1. fix-3 slope diagnostics (leak-free fits per held-out set, and the test fit)
    srows = []
    for split in ("lodo4", "lodo3"):
        B0 = b3.bench_frame(b3rec(REF5), split)
        B = post_group(B0, REF5, pd.Series("", index=B0.index, dtype=object))
        for ds in B.dataset.unique():
            if split == "lodo3" and ds != "vl":
                continue
            Btr = B[B.dataset != ds]
            s = sh.fit_centre_slope(Btr, zlog(Btr), keep=_not_in25(Btr))
            own = sh.fit_centre_slope(B[B.dataset == ds], zlog(B[B.dataset == ds]), keep=_not_in25(B[B.dataset == ds]))
            srows.append(dict(split=split, held_out=ds, beta_fit=s["beta"], beta_raw=s["raw"], r2_fit=s["r2"],
                              own_slope=own["raw"], own_r=list(own["per_dataset"].values())[0]["r"] if own["per_dataset"] else np.nan,
                              per_dataset=json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in s["per_dataset"].items()})))
    Bf = b3.bench_frame_v4(b3rec(REF5), "lodo4")
    s = sh.fit_centre_slope(Bf, zlog(Bf), keep=_not_in25(Bf))
    srows.append(dict(split="test", held_out="(none)", beta_fit=s["beta"], beta_raw=s["raw"], r2_fit=s["r2"],
                      per_dataset=json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in s["per_dataset"].items()})))
    SL = pd.DataFrame(srows)
    SL.to_csv(OUT5 / "fix3_slopes.csv", index=False)
    print(SL.round(3).to_string(index=False), flush=True)
    # 2. benchmark table (LODO constants, offsets applied after shrinkage)
    names = ["C7ref", "F7", "F127", "C7_lodoB", "C7_grp1", "C7_k0", "F7_lodoB", "F7_grp1", "F7_k0"]
    rows = []
    for nm in names:
        rows.append(dict(name=nm, desc=R[nm]["desc"], **b3.v4_score(bench_res5(R[nm]))))
    c2 = b3.run_bench_v4(b3.R("ours", "fusion", "c1", "conf", name="C2"), "lodo")
    rows.append(dict(name="C2 (LB 0.390)", desc="C2 recipe", **b3.v4_score(c2)))
    TB = pd.DataFrame(rows)
    TB.to_csv(OUT5 / "bench_v5.csv", index=False)
    print(TB[BENCH_COLS5].round(4).to_string(index=False), flush=True)
    # 3. per-slice paired bootstrap CIs (static by image, VL by subject, dynamic by 17-frame blocks)
    bt = [boot_vs(R[a], R[b], a, b) for a, b in (("F7", "C7ref"), ("F7_lodoB", "F7"), ("F7_grp1", "F7"), ("F7_k0", "F7"),
                                                 ("C7_lodoB", "C7ref"), ("C7_grp1", "C7ref"), ("C7_k0", "C7ref"))]
    BT = pd.concat(bt)
    BT.to_csv(OUT5 / "boot_v5.csv", index=False)
    print(BT[BT.bench.isin(["2r clean10", "2r all35", "VL-244 lodo3", "VL-244 lodo4", "VL-244 lodo3 FL",
                            "dynamic (PA_h+FL)", "dynamic FL"])].round(4).to_string(index=False), flush=True)
    # 4. pre-registered S_sel with paired bootstrap CIs and LB sampling SDs
    sel = {k: R[k] for k in ("C7ref", "F7", "F127", "F7_lodoB", "F7_grp1", "F7_k0", "C7_lodoB", "C7_grp1", "C7_k0")}
    sel["C2"] = dict(V("C2", apo="ours", pa="fusion", fl="c1", shrink="conf"), desc="C2 recipe (LB 0.390)")
    with _v5_scope():
        Tc, Tp = select(sel, B=args.B, tag="v5")
    print(Tc[SEL_SHOW5].round(4).to_string(index=False))
    print(Tp[["pair", "d_c10_2r_loo", "d_vl3", "d_vl3_lo", "d_vl3_hi", "d_dyn", "d_t142", "d_S_sel_loo",
              "d_S_sel_loo_lo", "d_S_sel_loo_hi", "P_S_sel_loo", "test_dist", "sd_pub_rho06", "sd_pub_rho1"]]
          .round(4).to_string(index=False), flush=True)
    # 5. decisions by the pre-registered rules
    p7 = _pair(Tp, "F7", "C7ref")
    fix7 = not (p7["d_S_sel_loo"] > 0 and p7["d_S_sel_loo_lo"] > 0)
    base = "F7" if fix7 else "C7ref"
    pre = "F7_" if fix7 else "C7_"
    ev3 = {}
    for f in ("lodoB", "grp1", "k0"):
        p = _pair(Tp, pre + f, base)
        bq = BT[(BT.a == pre + f) & (BT.b == base)].set_index("bench")
        worse = {k: bool(v) for k, v in (
            ("VL lodo3", p["d_vl3_lo"] > 0), ("VL lodo4", bq.loc["VL-244 lodo4", "lo"] > 0),
            ("clean10 2r (LOO)", p["d_c10_2r_loo_lo"] > 0), ("dynamic", p["d_dyn_lo"] > 0))}
        ok = bool(p["d_S_sel_loo"] < 0 and p["d_S_sel_loo_hi"] < 0 and not any(worse.values()))
        ev3[f] = dict(d_S_sel_loo=float(p["d_S_sel_loo"]), ci=[float(p["d_S_sel_loo_lo"]), float(p["d_S_sel_loo_hi"])],
                      d_vl3=float(p["d_vl3"]), d_vl4=float(bq.loc["VL-244 lodo4", "diff"]),
                      d_c10_2r_loo=float(p["d_c10_2r_loo"]), d_dyn=float(p["d_dyn"]), worse_ci_excl_0=worse, passes=ok)
    passing = [f for f, v in ev3.items() if v["passes"]]
    best3 = min(passing, key=lambda f: ev3[f]["d_S_sel_loo"]) if passing else None
    dec = dict(fix1_adopt=True, fix2_adopt=True, fix7_adopt=bool(fix7),
               fix7=dict(d_S_sel_loo=float(p7["d_S_sel_loo"]), ci=[float(p7["d_S_sel_loo_lo"]), float(p7["d_S_sel_loo_hi"])]),
               fix3_base=base, fix3=ev3, fix3_variant=best3)
    DECISIONS5.write_text(json.dumps(dec, indent=1, default=float))
    rep["decisions"] = dec
    print("DECISIONS:", json.dumps(dec, indent=1, default=float), flush=True)
    # 6. fix-7 sensitivity on test: EB weights and FL by group (C7ref vs F7)
    s0, P0, _ = test_run5(REF5)
    s7, P7, _ = test_run5(R["F7"])
    lay = b3.layout()
    g = lay.loc[s0.index, "group"]
    S7 = pd.DataFrame({"w_fl_all": P0["w_fl"].reindex(s0.index), "w_fl_clean": P7["w_fl"].reindex(s0.index),
                       "sigma_fl_all": P0["sigma_fl"].reindex(s0.index), "sigma_fl_clean": P7["sigma_fl"].reindex(s0.index),
                       "fl_all": s0.fl_mm, "fl_clean": s7.fl_mm, "group": g})
    G7 = S7.groupby("group").agg(n=("fl_all", "size"), w_fl_all=("w_fl_all", "mean"), w_fl_clean=("w_fl_clean", "mean"),
                                 sigma_all=("sigma_fl_all", "median"), sigma_clean=("sigma_fl_clean", "median"),
                                 fl_all=("fl_all", "median"), fl_clean=("fl_clean", "median"))
    G7["d_fl_mean"] = (S7.fl_clean - S7.fl_all).groupby(g).mean()
    G7["d_fl_absmax"] = (S7.fl_clean - S7.fl_all).abs().groupby(g).max()
    G7.to_csv(OUT5 / "fix7_test_sensitivity.csv")
    rep["fix7_test_dist_vs_C7"] = bc.official_distance(s7, s0)
    print(G7.round(3).to_string(), "\nfix 7 test distance vs C7:", rep["fix7_test_dist_vs_C7"], flush=True)
    # fix-3 on test (Telemed long clips, coupling k) for the record
    k3 = []
    for nm in ("F7", "F7_lodoB", "F7_grp1", "F7_k0"):
        s_, P_, _ = test_run5(R[nm])
        d_ = b3.v4_test_diag(s_, P_, {"C7": s0})
        d_["name"], d_["beta_test"] = nm, (P_.attrs.get("flc") or {}).get("beta")
        k3.append(d_)
    K3 = pd.DataFrame(k3)
    K3.to_csv(OUT5 / "fix3_test_diag.csv", index=False)
    print(K3[["name", "beta_test", "d_C7", "telemed_k_med", "clip0_fl", "clip9_fl", "clip26_fl", "clip0_tri", "clip9_tri"]]
          .round(3).to_string(index=False))
    rep["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    rep["seconds"] = time.time() - t0
    (OUT5 / "bench_v5_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"peak RSS {rep['peak_rss_gb']:.2f} GB, {rep['seconds']:.0f}s")


# ----------------------------------------------------------------------------------- v5 build
def group_table5(sub: pd.DataFrame, P: pd.DataFrame, cell: pd.Series) -> pd.DataFrame:
    """group_table with the v5 muscle table (fix 2)."""
    lay = b3.layout()
    T = ml.load_test_muscle(MUSCLE5).reindex(sub.index)
    m, _ = ml._trusted(T.fillna(""), 0.9, False)
    d = pd.DataFrame({"group": lay.loc[sub.index, "group"], "muscle": m.replace("", "unknown"),
                      "cell": cell.reindex(sub.index).replace("", "(scanner)"),
                      "pa": sub.pa_deg, "fl": sub.fl_mm, "mt": sub.mt_mm,
                      "k": sub.fl_mm * np.sin(np.radians(sub.pa_deg)) / sub.mt_mm,
                      "pa_raw": P["pa_deep_raw"].reindex(sub.index), "fl_raw": P["fl_mm"].reindex(sub.index),
                      "w_pa": P["w_pa"].reindex(sub.index), "w_fl": P["w_fl"].reindex(sub.index)})
    g1 = d.groupby("group").agg(n=("pa", "size"), pa=("pa", "median"), fl=("fl", "median"), mt=("mt", "median"),
                                k=("k", "median")).assign(muscle="ALL")
    g2 = d.groupby(["group", "muscle"]).agg(n=("pa", "size"), pa=("pa", "median"), fl=("fl", "median"),
                                            mt=("mt", "median"), k=("k", "median"), pa_raw=("pa_raw", "median"),
                                            fl_raw=("fl_raw", "median"), w_pa=("w_pa", "mean"),
                                            w_fl=("w_fl", "mean")).reset_index("muscle")
    return pd.concat([g1, g2]).reset_index().sort_values(["group", "muscle"])


def cmd_v5build(args) -> None:
    """C12 (+ diagnostics), validated, with the benchmark table vs C7, S_sel pairs, distances and stakes."""
    (OUT5 / "diagnostic").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    dec = json.loads(DECISIONS5.read_text())
    base = c12_base(dec)
    C12 = dict(base, offset="rec5", desc=f"C7 + critic4 fixes 1, 2{', 7' if dec['fix7_adopt'] else ''}"
                                         f"{', 3 (' + dec['fix3_variant'] + ')' if dec.get('fix3_variant') else ''}"
                                         " + re-estimated half rater offsets")
    R = v5_factorial()
    cands = {"C12": C12}
    diag = {"D_C12_nooff": dict(base, offset=None, desc="C12 without the rater offsets (C9-like partner)"),
            "D_F1": R["F1"], "D_F2": R["F2"], "D_F7": R["F7"], "D_F12": R["F12"]}
    for f in ("lodoB", "grp1", "k0"):
        if f != dec.get("fix3_variant"):
            diag[f"D_C12_fix3_{f}"] = dict(base, offset="rec5", flc=f, desc=f"C12 + fix 3 ({f}) - diagnostic only")
    if dec.get("fix3_variant"):
        diag["D_C12_nofix3"] = dict(base, offset="rec5", flc=None, desc="C12 without fix 3")
    refs = {k: pd.read_csv(p).set_index("image_id") for k, p in REF_FILES5.items() if p.exists()}
    refs["C2"] = pd.read_csv(b3.REFS["C2"][0]).set_index("image_id")
    rep, subs = {"decisions": dec, "offsets": offset_of5(C12)}, {}
    for nm, r in {**cands, **diag}.items():
        sub, P, pa_cell = test_run5(r)
        fl_cell = P["cell"].reindex(sub.index).fillna("") if "cell" in P else pd.Series("", index=sub.index)
        path = (OUT5 if nm in cands else OUT5 / "diagnostic") / f"{nm}.csv"
        if nm == "C12":
            tag = "_".join(["hyb_med_ens_pp_muscle5", "em" + base["em"]] + ([f"flc{base['flc']}"] if base["flc"] else []) + ["off"])
            path = OUT5 / f"C12_{tag}.csv"
        bc.write_sub(sub, path)
        subs[nm] = sub
        rep[nm] = dict(file=str(path.relative_to(ROOT)), desc=r["desc"], recipe={k: v for k, v in r.items() if k != "desc"},
                       validate=b3.validate(path), offset=offset_of5(r),
                       pa_cells=pa_cell[pa_cell != ""].value_counts().to_dict(),
                       fl_cells=fl_cell[fl_cell != ""].value_counts().to_dict(), flc=P.attrs.get("flc"),
                       dist={k: bc.official_distance(sub, v) for k, v in refs.items()})
        print(f"== {nm}: {rep[nm]['validate']}\n   pa cells {rep[nm]['pa_cells']}\n   fl cells {rep[nm]['fl_cells']}")
        for k in ("C7", "C2", "C9"):
            print(f"   vs {k}: {rep[nm]['dist'][k]}")
        if nm == "C12":
            Q = P[[c for c in ("group", "clip", "pa_cell", "cell", "confidence", "pa_deep_raw", "pa_deep_c", "pa_deg",
                               "fl_mm", "mt_mm", "sigma_fl", "mu_fl", "tau_fl", "c_fl", "w_fl", "fl_centre_dev", "w_pa",
                               "pa_eb", "fl_eb") if c in P]].copy()
            Q["sub_pa_deg"], Q["sub_fl_mm"], Q["sub_mt_mm"] = sub.pa_deg, sub.fl_mm, sub.mt_mm
            Q.index.name = "image_id"
            Q.to_csv(OUT5 / "per_image_C12.csv")
            G = group_table5(sub, P, pa_cell)
            G.to_csv(OUT5 / "test_groups_by_muscle_C12.csv", index=False)
    # benchmarks of C12 vs C7 (and C2): table, per-slice bootstrap, S_sel pairs
    rows = [dict(name="C7 (C7ref)", **b3.v4_score(bench_res5(REF5))), dict(name="C12", **b3.v4_score(bench_res5(C12)))]
    rows.append(dict(name="C2 (LB 0.390)", **b3.v4_score(b3.run_bench_v4(b3.R("ours", "fusion", "c1", "conf", name="C2"), "lodo"))))
    TB = pd.DataFrame(rows)
    TB.to_csv(OUT5 / "bench_C12.csv", index=False)
    print(TB[BENCH_COLS5].round(4).to_string(index=False))
    BT = boot_vs(C12, REF5, "C12", "C7")
    BT.to_csv(OUT5 / "boot_C12_vs_C7.csv", index=False)
    print(BT.round(4).to_string(index=False))
    sel = {"C12": C12, "C7": REF5, "C12_nooff": diag["D_C12_nooff"],
           "C2": dict(V("C2", apo="ours", pa="fusion", fl="c1", shrink="conf"), desc="C2 recipe (LB 0.390)")}
    with _v5_scope():
        Tc, Tp = select(sel, B=args.B, tag="C12")
    print(Tc[SEL_SHOW5].round(4).to_string(index=False))
    print(Tp[["pair", "d_c10_2r_loo", "d_vl3", "d_dyn", "d_t142", "d_S_sel_loo", "d_S_sel_loo_lo", "d_S_sel_loo_hi",
              "test_dist", "sd_pub_rho06", "sd_pub_rho1", "sd_pub_priv_rho1"]].round(4).to_string(index=False))
    # distances + LB sampling SDs, where the C12 - C7 change sits, stakes
    full = {"C12": subs["C12"], **{k: v for k, v in subs.items() if k != "C12"}, **refs}
    prow = []
    for b in [k for k in full if k != "C12"]:
        d = bc.official_distance(full["C12"], full[b])
        s6, s1 = lb_sd(full["C12"], full[b], 0.6), lb_sd(full["C12"], full[b], 1.0)
        prow.append(dict(a="C12", b=b, S=d["S"], pa_n=d["pa_n"], fl_n=d["fl_n"], mt_n=d["mt_n"],
                         sd_pub_rho06=s6["sd_pub"], sd_pub_rho1=s1["sd_pub"], sd_pub_priv_rho1=s1["sd_pub_priv"]))
    PW = pd.DataFrame(prow)
    PW.to_csv(OUT5 / "distances_C12.csv", index=False)
    print(PW.round(4).to_string(index=False))
    lay = b3.layout()
    T1 = ml.load_test_muscle(MUSCLE5).reindex(subs["C12"].index)
    m1, _ = ml._trusted(T1.fillna(""), 0.9, False)
    ch = []
    for a, b, A, Bq in (("C12", "C7", subs["C12"], refs["C7"]), ("D_F1", "C7", subs["D_F1"], refs["C7"]),
                        ("D_F2", "C7", subs["D_F2"], refs["C7"]), ("D_F7", "C7", subs["D_F7"], refs["C7"])):
        dd = A - Bq.reindex(A.index)
        dd["group"], dd["muscle"] = lay.loc[A.index, "group"], m1.replace("", "unknown")
        x = dd.groupby(["group", "muscle"]).agg(n=("pa_deg", "size"), d_pa_mean=("pa_deg", "mean"),
                                                d_fl_mean=("fl_mm", "mean"), d_mt_mean=("mt_mm", "mean"),
                                                d_pa_absmax=("pa_deg", lambda s: s.abs().max()),
                                                d_fl_absmax=("fl_mm", lambda s: s.abs().max())).reset_index()
        x.insert(0, "pair", f"{a}-{b}")
        ch.append(x)
    CH = pd.concat(ch)
    CH.to_csv(OUT5 / "test_change_by_muscle_C12.csv", index=False)
    print(CH.round(3).to_string(index=False))
    st = []
    S_ = {"pa_deg": 1.8, "fl_mm": 6.7, "mt_mm": 0.8}
    f = lambda x, s_: np.abs(x) + s_ * np.exp(-np.abs(x) / s_)  # noqa: E731
    for a, b in (("C12", "C7"), ("D_F1", "C7"), ("D_F2", "C7"), ("D_F7", "C7")):
        A, Bq = subs[a], refs[b].reindex(subs[a].index)
        row = dict(a=a, b=b)
        for c in C3:
            sc = (S_[c] * (A[c] / 65.0)).to_numpy(float) if c == "fl_mm" else S_[c]
            bb = (A[c] - Bq[c]).to_numpy(float)
            row[f"stake_{c[:2]}"] = float(np.mean(f(bb, sc) - f(0 * bb, sc))) / TAU[c] / 3
        row["stake_total"] = row["stake_pa"] + row["stake_fl"] + row["stake_mt"]
        st.append(row)
    ST = pd.DataFrame(st)
    ST.to_csv(OUT5 / "stake_C12.csv", index=False)
    print(ST.round(4).to_string(index=False))
    rep["pairwise"] = PW.to_dict("records")
    rep["stake"] = ST.to_dict("records")
    rep["select"] = Tc.to_dict("records")
    rep["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    (OUT5 / "candidates_v5_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"peak RSS {rep['peak_rss_gb']:.2f} GB")


# ----------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("offsets")
    o.add_argument("--B", type=int, default=4000)
    b = sp.add_parser("bench")
    b.add_argument("--B", type=int, default=2000)
    b.add_argument("--R", type=int, default=300)
    sp.add_parser("build")
    # v5 (fixes5): critic4 fixes 1, 2, 3, 7 -> artifacts/candidates_v5 (run in this order)
    sp.add_parser("v5muscle")
    b5 = sp.add_parser("v5bench")
    b5.add_argument("--B", type=int, default=2000)
    o5 = sp.add_parser("v5offsets")
    o5.add_argument("--B", type=int, default=4000)
    c5 = sp.add_parser("v5build")
    c5.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    {"offsets": cmd_offsets, "bench": cmd_bench, "build": cmd_build, "v5muscle": cmd_v5muscle,
     "v5bench": cmd_v5bench, "v5offsets": cmd_v5offsets, "v5build": cmd_v5build}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
