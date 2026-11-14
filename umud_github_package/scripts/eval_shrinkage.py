"""Validate empirical-Bayes shrinkage (src/umud/shrinkage.py) and apply it to the test caches.

Light: reads the per-image caches of scripts/build_candidates.py (artifacts/candidates/cache), no
images, no maps, no torch; one process, < 0.3 GB.

Subcommands
-----------
eval      Validation exactly as the procedure runs on test: the error model is fitted on the OTHER
          benchmark datasets, the group prior is estimated label-free from the target dataset's own
          estimates. Two splits:
            lodo3 = {static, vl, dynamic}: VL held out means the error model never saw a noisy
                    out-of-domain dataset (pessimistic stress test of transfer);
            lodo4 = {static, vl_young, vl_old, dynamic}: VL subjects held out by population (young /
                    old), the error model has seen the other half of VL (the test situation: the final
                    model is trained on all benchmarks incl. VL, then applied to new scanners).
          Compared with C1 (no shrinkage), C2 (w = confidence), C3 (group median). A small documented
          grid over the error-model fit is scored on lodo4. Then fits the final error model on all
          benchmarks -> artifacts/eval_v3/eb/error_model.json, applies it to the test cache and writes
          per-image tables + candidate CSVs (NOT submitted).
selftest  Synthetic unit tests (scratchpad eb/test_shrinkage.py).

Bench rows use the C1 recipe (PA = fusion, FL = ensemble, MT inner edges, MT < 5 mm -> failure) of
cache <dataset>__<source>__<cfg>; dynamic PA comes from the v2h run (PA vs horizontal, the
benchmark's convention). --source/--cfg evaluate other caches; --bench-extra / --test-extra CSVs
(image_id, dataset[, ...] + new estimate or x_* diagnostic columns) override/add columns, so new
per-image estimates from other agents can be re-validated and re-applied without code changes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_candidates as bc  # noqa: E402
from umud import shrinkage as sh  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "artifacts/eval_v3/eb"
CANDS = ROOT / "artifacts/candidates"
SCRATCH_EB = bc.SCRATCH / "eb"
TOLS = {"pa": 6.0, "fl": 12.0, "mt": 3.0}


# ----------------------------------------------------------------------------------- bench data
def c1_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Cache rows -> C1 estimates (pa_deg = fusion, fl_mm = ensemble, mt_mm) + all diagnostics."""
    out = d.copy()
    out["pa_deg"] = d["pa_fusion"].astype(float)
    out["fl_mm"] = d["fl_ensemble"].astype(float)
    out["mt_mm"] = d["mt_mm"].astype(float)
    bad = out["mt_mm"] < 5.0
    out.loc[bad, ["pa_deg", "fl_mm", "mt_mm"]] = np.nan
    return out


def static_group(idx) -> pd.Series:
    """Scanner proxy for the static benchmark: image size (sizes with < 5 images pooled)."""
    g = bc.static_groups(idx)
    vc = g.value_counts()
    return g.where(g.map(vc) >= 5, "other")


def load_bench(source: str = "ours", cfg: str = "v2e", split: str = "lodo4") -> tuple[pd.DataFrame, dict]:
    """All benchmark rows: estimates, truths (*_true), dataset, group (= scanner proxy), diagnostics."""
    parts, extra_gt = [], {}
    st = bc.load_cache("static", source, cfg)
    G = bc.static_gts()
    if st is not None:
        d = c1_frame(st)
        g = G["mean7"].reindex(d.index)
        d["pa_true"], d["fl_true"], d["mt_true"] = g.pa_deg, g.fl_mm, g.mt_mm
        d["dataset"], d["group"] = "static", static_group(d.index)
        parts.append(d)
        extra_gt["static"] = G
    vl = bc.load_cache("vl", source, cfg)
    if vl is not None:
        d = c1_frame(vl)
        v = bc.vl_gt().reindex(d.index)
        d["pa_true"], d["fl_true"] = v.pa_deg, v.fl_mm
        d["mt_true"] = v.mt_mm      # gap between the drawn lines: loose (not an expert MT)
        if split == "lodo4":
            d["dataset"] = np.where(d.index.str.startswith("young"), "vl_young", "vl_old")
        else:
            d["dataset"] = "vl"
        d["group"] = d["dataset"]
        parts.append(d)
    dd, dh = bc.load_cache("dynamic", source, cfg), bc.load_cache("dynamic", source, "v2h")
    if dd is not None and dh is not None:
        d = c1_frame(dd)
        d["pa_deg"] = dh["pa_fusion"].reindex(d.index).astype(float)   # PA vs horizontal
        dg = bc.dynamic_gt().reindex(d.index)
        d["pa_true"], d["fl_true"], d["mt_true"] = dg.pa_deg, dg.fl_mm, np.nan
        d["dataset"], d["group"] = "dynamic", "dynamic"
        d = d[d["pa_true"].notna() | d["fl_true"].notna()]
        parts.append(d)
    B = pd.concat(parts)
    B["image_id"] = B.index.astype(str)
    B.index = B["dataset"].str.replace(r"_(young|old)$", "", regex=True) + ":" + B.index.astype(str)
    B["clip"] = -1
    return B, extra_gt


def merge_extra(df: pd.DataFrame, path: str | None, key_ds: bool) -> pd.DataFrame:
    """Override / add columns from a CSV (image_id[, dataset] + columns)."""
    if not path:
        return df
    X = pd.read_csv(path)
    if key_ds:
        X.index = X["dataset"].str.replace(r"_(young|old)$", "", regex=True) + ":" + X["image_id"].astype(str)
    else:
        X.index = X["image_id"].astype(str)
    out = df.copy()
    for c in X.columns:
        if c not in ("dataset", "image_id"):
            out[c] = X[c].reindex(out.index).where(X[c].reindex(out.index).notna(), out[c] if c in out else np.nan)
    return out


# ----------------------------------------------------------------------------------- scoring
def score(pred: pd.DataFrame, truth: pd.DataFrame, comps=("pa", "fl", "mt")) -> dict:
    r, ns = {}, []
    for c in comps:
        t = truth[f"{c}_true"]
        e = (pred[f"{c}_eb"] - t).dropna()
        if len(e) == 0:
            continue
        n = float(e.abs().mean() / TOLS[c])
        r[f"{c}_n"] = n
        r[f"{c}_bias"] = float(e.mean())
        ns.append(n)
    r["S"] = float(np.mean(ns)) if ns else np.nan
    return r


def fill_group(pred: pd.DataFrame, group_col="group") -> pd.DataFrame:
    for c in sh.COMP:
        k = f"{c}_eb"
        pred[k] = pred[k].fillna(pred.groupby(group_col)[k].transform("median")).fillna(pred[k].median())
    return pred


# ----------------------------------------------------------------------------------- methods
EB = dict(method="ml", likelihood="laplace")
METHODS = {
    # name: kwargs for sh.shrink (None = baseline)
    "C1_none": None,
    "C2_conf": None,
    "C3_groupmed": None,
    "eb_mom_gauss": dict(method="mom", likelihood="gauss", c_fixed=1.0),   # the literal spec: c = 1, MoM tau
    "eb_ml_gauss": dict(method="ml", likelihood="gauss"),                   # linear shrink, (mu, tau, c) by ML
    "eb_ml_laplace": dict(EB),                                              # robust posterior (recommended)
    "eb_ml_laplace_conf": dict(EB, _features=["negconf"]),                  # error model from confidence only
    "eb_ml_laplace_joint": dict(EB, joint=True),                            # + FL sin PA = k0 MT coupling
    "eb_ml_laplace_logfl": dict(EB, fl_space="log"),
    "eb_ml_laplace_pafl_mt": dict(EB, cols=("pa", "fl", "mt")),
}


def run_method(name: str, D: pd.DataFrame, model: sh.ErrorModel | None, **over) -> pd.DataFrame:
    if name == "C1_none":
        p = D.copy()
        for c in sh.COMP:
            p[f"{c}_eb"] = D[sh.COMP[c]]
        return fill_group(p)
    if name == "C2_conf":
        return fill_group(sh.confidence_shrink(D, "group"))
    if name == "C3_groupmed":
        return fill_group(sh.group_median(D, "group"))
    kw = {k: v for k, v in METHODS[name].items() if not k.startswith("_")}
    kw.update(over)
    return fill_group(sh.shrink(D, model, group_col="group", clip_col="clip", **kw))


def static_extras(P: pd.DataFrame, G: dict) -> dict:
    Q = pd.DataFrame({sh.COMP[c]: P[f"{c}_eb"].to_numpy() for c in sh.COMP},
                     index=[i.split(":", 1)[1] for i in P.index])
    return dict(c10=bc.s_metric(Q.loc[bc.CLEAN], G["mean7"])["S"],
                r2=float(np.mean([bc.s_metric(Q, G[f"R7+{x}"])["S"] for x in bc.RATERS[:-1]])),
                R7=bc.s_metric(Q, G["R7"])["S"])


# ----------------------------------------------------------------------------------- eval
def evaluate(B: pd.DataFrame, extra_gt: dict, fit_kw: dict, methods=None, insample: bool = False,
             split: str = "") -> pd.DataFrame:
    res = []
    methods = methods or list(METHODS)
    for ds in B.dataset.unique():
        train = B if insample else B[B.dataset != ds]
        D = B[B.dataset == ds]
        models = {}
        for m in methods:
            if METHODS.get(m) is None:
                model = None
            else:
                feats = METHODS[m].get("_features")
                key = tuple(feats) if feats else None
                if key not in models:
                    models[key] = sh.fit_error_model(train, **dict(fit_kw, **({"features": feats} if feats else {})))
                model = models[key]
            if ds == "dynamic" and METHODS.get(m) and METHODS[m].get("joint"):
                continue  # dynamic PA is vs the horizontal: the FL-PA-MT coupling (deep apo) does not apply
            P = run_method(m, D, model)
            r = dict(split=split, dataset=ds, method=m + ("@insample" if insample else ""))
            pafl = score(P, D, ("pa", "fl"))
            r.update(S_pafl=pafl["S"], pa_n=pafl.get("pa_n"), fl_n=pafl.get("fl_n"),
                     pa_bias=pafl.get("pa_bias"), fl_bias=pafl.get("fl_bias"))
            if ds == "static":
                s3 = score(P, D, ("pa", "fl", "mt"))
                r.update(S3=s3["S"], mt_n=s3.get("mt_n"), **static_extras(P, extra_gt["static"]))
            for c in ("pa", "fl"):
                if f"w_{c}" in P:
                    r[f"mean_w_{c}"] = float(P[f"w_{c}"].mean())
            pri = P.attrs.get("priors", {})
            for c in ("pa", "fl"):
                v = [p for k, p in pri.items() if k.endswith("|" + c)]
                if v:
                    r[f"c_{c}"] = float(np.mean([p["c"] for p in v]))
                    r[f"tau_{c}"] = float(np.mean([p["tau"] for p in v]))
            res.append(r)
    return pd.DataFrame(res)


def calibration(B: pd.DataFrame, fit_kw: dict) -> pd.DataFrame:
    """Held-out error model vs the actual residuals: rank correlation and level (|r| / b)."""
    from scipy.stats import spearmanr
    rows = []
    for ds in B.dataset.unique():
        D = B[B.dataset == ds]
        for fitset in ("held_out", "in_sample"):
            model = sh.fit_error_model(B[B.dataset != ds] if fitset == "held_out" else B, **fit_kw)
            sig = model.predict(D)
            for c in ("pa", "fl", "mt"):
                r = sh._residual(D, c)
                if r.notna().sum() < 10 or c not in model.targets:
                    continue
                rr = (r - r.median()).abs()
                s = sig[f"sigma_{c}_rel" if c == "fl" else f"sigma_{c}"]
                ok = rr.notna() & s.notna()
                q = pd.qcut(s[ok].rank(method="first"), 3, labels=False)
                rows.append(dict(dataset=ds, fit=fitset, comp=c, n=int(ok.sum()),
                                 spearman=float(spearmanr(s[ok], rr[ok])[0]),
                                 spearman_conf=float(spearmanr(-D.confidence[ok], rr[ok])[0])
                                 if D.confidence[ok].nunique() > 1 else np.nan,
                                 level=float(rr[ok].mean() / (s[ok] / np.sqrt(2)).mean()),
                                 mae_low_third=float(rr[ok][q == 0].mean()),
                                 mae_high_third=float(rr[ok][q == 2].mean())))
    return pd.DataFrame(rows)


FEATURE_SETS = {"all": None, "conf": ["negconf"],
                "robust": ["negconf", "log_nfrag", "pa_iqr", "extrap_c3", "fl_iqr", "fl_ens_spread", "tri_rel",
                           "pa_dis", "shared", "flag_apo_fallback", "flag_pair_fallback", "flag_few_fragments",
                           "flag_short_apo_overlap", "flag_pa_implausible"]}


def grid(B4: pd.DataFrame, extra_gt: dict) -> pd.DataFrame:
    """Small grid over the error-model fit; eb_ml_laplace on lodo4 (mean S_pafl over the 4 datasets)."""
    rows = []
    for fs, ridge, bal in itertools.product(FEATURE_SETS, (0.05, 0.3, 1.0), ("sqrt", "equal")):
        kw = dict(features=FEATURE_SETS[fs], ridge=ridge, balance=bal)
        R = evaluate(B4, extra_gt, kw, methods=["eb_ml_laplace"])
        X = R.set_index("dataset")
        rows.append(dict(features=fs, ridge=ridge, balance=bal, **{f"{d}_S": X.loc[d, "S_pafl"] for d in X.index},
                         mean4=float(X.S_pafl.mean())))
    return pd.DataFrame(rows).sort_values("mean4")


# ----------------------------------------------------------------------------------- test
def load_test(source="ours", cfg="v2e") -> pd.DataFrame:
    d = bc.load_cache("test", source, cfg)
    T = c1_frame(d)
    lay = bc.test_layout()
    T["group"] = lay.loc[T.index, "group"]
    T["clip"] = lay.loc[T.index, "clip"].astype(int)
    return T


def to_sub(P: pd.DataFrame) -> pd.DataFrame:
    sub = pd.DataFrame({"pa_deg": P["pa_eb"], "fl_mm": P["fl_eb"], "mt_mm": P["mt_eb"]}, index=P.index)
    return bc.postprocess(sub, bc.test_layout())   # NaN fill, clip median (no-op for EB units), bounds


def md_table(df: pd.DataFrame, cols: list) -> str:
    cols = [c for c in cols if c in df]
    X = df[cols].copy()
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in X.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def cmd_eval(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    B4, extra_gt = load_bench(args.source, args.cfg, "lodo4")
    B3, _ = load_bench(args.source, args.cfg, "lodo3")
    B4 = merge_extra(B4, args.bench_extra, True)
    B3 = merge_extra(B3, args.bench_extra, True)
    print("bench rows:", B4.dataset.value_counts().to_dict(), flush=True)
    # 1) error-model fit settings by lodo4 (small grid; flat optimum, see grid_lodo4.csv)
    H = grid(B4, extra_gt)
    H.to_csv(OUT / "grid_lodo4.csv", index=False)
    print(H.round(4).to_string(index=False), flush=True)
    best = H.iloc[0]
    fit_kw = dict(features=FEATURE_SETS[best.features], ridge=float(best.ridge), balance=best.balance)
    print("chosen error-model fit:", dict(features=best.features, ridge=best.ridge, balance=best.balance), flush=True)
    # 2) all methods, both splits, + in-sample error model (transfer gap)
    R = pd.concat([evaluate(B4, extra_gt, fit_kw, split="lodo4"),
                   evaluate(B3, extra_gt, fit_kw, split="lodo3"),
                   evaluate(B4, extra_gt, fit_kw, methods=["eb_ml_laplace", "eb_ml_laplace_joint"], insample=True,
                            split="lodo4")], ignore_index=True)
    R.to_csv(OUT / "bench_lodo.csv", index=False)
    show = ["split", "dataset", "method", "S_pafl", "pa_n", "fl_n", "S3", "mt_n", "c10", "r2", "R7", "mean_w_pa",
            "mean_w_fl", "c_pa", "c_fl", "tau_pa", "tau_fl"]
    print(R[[c for c in show if c in R]].round(3).to_string(index=False), flush=True)
    W = R.pivot_table(index="method", columns=["split", "dataset"], values="S_pafl")
    print(W.round(3).to_string(), flush=True)
    CAL = calibration(B4, fit_kw)
    CAL.to_csv(OUT / "calibration_lodo4.csv", index=False)
    print(CAL.round(3).to_string(index=False), flush=True)
    # 3) final error model on all benchmarks (lodo4 dataset labels: VL young/old weighted as two sets)
    model = sh.fit_error_model(B4, **fit_kw)
    model.meta.update(source=args.source, cfg=args.cfg, created=time.strftime("%Y-%m-%d %H:%M"),
                      fit=dict(features=best.features, ridge=float(best.ridge), balance=best.balance),
                      shrink_defaults=dict(method="ml", likelihood="laplace", prior_sd_logtau=0.2, prior_sd_logc=2.0,
                                           pop_sd=dict(sh.POP_SD), tau_floor=dict(sh.TAU_FLOOR), k0=sh.K0,
                                           s_logk=sh.S_LOGK, marginal="t", nu=5),
                      note="sigma = sqrt(2) * Laplace scale; FL error modelled on the log ratio; residuals demeaned "
                           "per dataset; features standardised by median/IQR and clipped to +-4")
    model.to_json(OUT / "error_model.json")
    model_conf = sh.fit_error_model(B4, **dict(fit_kw, features=["negconf"]))
    # 4) test
    T = merge_extra(load_test(args.source, args.cfg), args.test_extra, False)
    subs, pri, per = {}, {}, {}
    for m in METHODS:
        mod = model_conf if (METHODS[m] or {}).get("_features") else model
        P = run_method(m, T, mod)
        per[m] = P
        if m.startswith("eb_"):
            pri[m] = P.attrs.get("priors", {})
        subs[m] = to_sub(P)
    (OUT / "test_priors.json").write_text(json.dumps(pri, indent=1, default=float))
    keep = ["group", "clip", "eb_unit", "confidence", "flags", "pa_deg", "fl_mm", "mt_mm", "sigma_pa", "sigma_fl",
            "sigma_mt", "mu_pa", "tau_pa", "w_pa", "pa_eb", "mu_fl", "tau_fl", "w_fl", "fl_eb", "mt_eb"]
    for m in ("eb_ml_laplace", "eb_ml_laplace_joint"):
        P = per[m]
        Q = P[[c for c in keep if c in P]].copy()
        S = subs[m]
        Q["sub_pa_deg"], Q["sub_fl_mm"], Q["sub_mt_mm"] = S.pa_deg, S.fl_mm, S.mt_mm
        Q.to_csv(OUT / f"test_per_image_{m}.csv")
    ref = {}
    for k, f in (("C1", "C1_ours_v2e_fusion_ensemble.csv"), ("C2", "C2_ours_v2e_fusion_ensemble_shrink1.0.csv"),
                 ("C3", "C3_ours_v2e_fusion_ensemble_groupPAFL.csv")):
        p = CANDS / f
        if p.exists():
            ref[k] = pd.read_csv(p).set_index("image_id")
    lay = bc.test_layout()
    rows = []
    for m, sub in subs.items():
        r = dict(method=m)
        for k, v in ref.items():
            dd = bc.official_distance(sub, v)
            r[f"d_{k}"], r[f"d_{k}_pa"], r[f"d_{k}_fl"] = dd["S"], dd["pa_n"], dd["fl_n"]
        g = sub.groupby(lay.loc[sub.index, "group"])[["pa_deg", "fl_mm"]].median()
        r.update({f"med_pa:{gg}": v for gg, v in g.pa_deg.items()})
        r.update({f"med_fl:{gg}": v for gg, v in g.fl_mm.items()})
        k = sub.fl_mm * np.sin(np.radians(sub.pa_deg)) / sub.mt_mm
        r.update({f"k:{gg}": v for gg, v in k.groupby(lay.loc[sub.index, "group"]).median().items()})
        rows.append(r)
    TD = pd.DataFrame(rows)
    TD.to_csv(OUT / "test_distances.csv", index=False)
    print(TD[[c for c in TD if not c.startswith(("med_", "k:"))]].round(3).to_string(index=False))
    print(TD.set_index("method")[[c for c in TD if c.startswith(("med_pa", "med_fl", "k:"))]].T.round(2).to_string())
    # where the recommended EB differs from C2 / C1: official-distance contribution per group
    grp = lay.loc[subs["C1_none"].index, "group"]
    contrib = {}
    for a, b in (("eb_ml_laplace", "C2_conf"), ("eb_ml_laplace", "C1_none"), ("eb_ml_laplace_joint", "C2_conf"),
                 ("C2_conf", "C1_none")):
        d = (subs[a] - subs[b]).abs()
        n = len(d)
        contrib[f"{a}-vs-{b}"] = pd.DataFrame({"pa": d.pa_deg / 6 / 3 / n, "fl": d.fl_mm / 12 / 3 / n}).groupby(grp).sum()
    CG = pd.concat(contrib, axis=1).round(4)
    CG.to_csv(OUT / "test_change_by_group.csv")
    print("\n== official-distance contribution by group (sums to the distance)\n", CG.to_string())
    names = {"eb_ml_laplace": "C4_eb_laplace", "eb_ml_laplace_joint": "C5_eb_laplace_joint"}
    for m, fn in names.items():
        bc.write_sub(subs[m], OUT / f"{fn}.csv")
    wg = {}
    for m in ("eb_ml_laplace", "eb_ml_laplace_joint", "C2_conf"):
        P = per[m]
        if "w_pa" not in P:
            conf = P.confidence.fillna(0).clip(0, 1)
            P = P.assign(w_pa=conf, w_fl=conf)
        x = P.groupby("group")[["w_pa", "w_fl"]].mean()
        wg[m] = x
    WG = pd.concat(wg, axis=1).round(3)
    WG["n_rows"] = T.group.value_counts()
    print("\n== mean weight on the own measurement per test group\n", WG.to_string())
    WG.to_csv(OUT / "test_weights_by_group.csv")
    summary = dict(chosen_fit=dict(features=best.features, ridge=float(best.ridge), balance=best.balance),
                   lodo=[{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                         for r in R.to_dict("records")],
                   test_distances=TD.to_dict("records"),
                   test_priors_recommended=pri.get("eb_ml_laplace", {}))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    # markdown summary
    piv = R[R.split == "lodo4"].pivot_table(index="method", columns="dataset", values="S_pafl")
    dcols = list(piv.columns)
    piv["mean3_no_dyn"] = piv[[c for c in ("static", "vl_old", "vl_young") if c in dcols]].mean(1)
    piv["mean4"] = piv[dcols].mean(1, skipna=False)
    piv3 = R[R.split == "lodo3"].pivot_table(index="method", columns="dataset", values="S_pafl")
    st = R[(R.split == "lodo4") & (R.dataset == "static")].set_index("method")[["S3", "c10", "r2", "R7"]]
    md = ["# Empirical-Bayes shrinkage: benchmark validation (generated by scripts/eval_shrinkage.py)", "",
          f"Error model fit: features={best.features}, ridge={best.ridge}, balance={best.balance}", "",
          "## lodo4, S over PA+FL (official normalisation)", "", md_table(piv.reset_index().sort_values("mean3_no_dyn"),
                                                                           ["method"] + list(piv.columns)),
          "", "## lodo3 (VL held out entirely), S over PA+FL", "",
          md_table(piv3.reset_index(), ["method"] + list(piv3.columns)), "",
          "## static, all three components (lodo4 error model)", "", md_table(st.reset_index(), ["method", "S3", "c10", "r2", "R7"]),
          "", "## test: distance to C1/C2/C3 (official metric)", "",
          md_table(TD, ["method", "d_C1", "d_C2", "d_C3", "d_C2_pa", "d_C2_fl"]), ""]
    (OUT / "README_generated.md").write_text("\n".join(md))
    print("wrote", OUT, flush=True)


# ----------------------------------------------------------------------------------- selftest
def cmd_selftest(args) -> None:
    import unittest
    sys.path.insert(0, str(SCRATCH_EB))
    import test_shrinkage  # noqa: E402
    r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_shrinkage))
    raise SystemExit(0 if r.wasSuccessful() else 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--source", default="ours")
    e.add_argument("--cfg", default="v2e")
    e.add_argument("--bench-extra", default=None, help="CSV image_id,dataset,+columns merged into the bench rows")
    e.add_argument("--test-extra", default=None, help="CSV image_id,+columns merged into the test rows")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"eval": cmd_eval, "selftest": cmd_selftest}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
