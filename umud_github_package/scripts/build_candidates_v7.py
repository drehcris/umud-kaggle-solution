"""shrink7: shrinkage strength per scanner group x component on the CURRENT maps (C12 sources), C18
reproduction, and single-change candidates C20..C24 (artifacts/candidates_v7). Never submits.

Light by design (laptop rules): reads the per-image v3/v4 caches only (no images, no masks, no torch),
one process, sequential, < 0.5 GB RSS, ~15 s per subcommand.

Background (LB facts, lower is better): C1 0.45436 (per-image, OLD maps) -> C2 0.38974 (confidence shrink of
PA + FL toward scanner-group medians); C12 0.37670 (current recipe: apo_v2fix MT, fusF PA with the confidence
rule, ens_pp FL with EB shrinkage, scanner x muscle centres, half rater offset); C18 0.33526 = C12 with the
Telemed FL replaced by the UNSHRUNK per-clip median raw FL. C11 0.51239 (PA/FL = cell medians). C17 0.38091
(C12 without the offset).

Shrinkage strength lambda: prediction = raw + lambda * (current shrunk - raw), per component, per scanner
group; lambda = 1 is the current shrinkage, 0 is none. On test the FL blend is done on the EB unit (the
per-clip median of the per-frame FL for the 28 Telemed clips, the image elsewhere), the PA blend per image
and then the clip median, exactly as the pipeline does. The rater offset is applied after the blend.

Analyses (all on the C12 recipe and maps; LODO constants; paired bootstrap 95% CIs: static by image, VL by
subject, dynamic by circular 10-frame blocks, train142 by near-duplicate group; B = 2000)
* lambda curves {0, .25, .5, .75, 1} + a 0.05-grid argmin, and uniform shrink fractions kappa, per slice;
* the same on the OLD maps with the C2 recipe (ours / fusion / C1 FL / confidence shrink of PA + FL);
* shrinkage gain stratified by the image's weight, by raw FL/FOV (FL) and by why the confidence is < 1 (PA:
  FL-only flags high_extrapolation / fl_spread / fl_triangle_disagree vs PA-relevant flags);
* shrinkage EFFICIENCY per proxy x bin = sum(|e(lam 0)| - |e(lam 1)|) / sum|pred(lam 1) - pred(lam 0)| (share
  of the moved metric mass the shrinkage got right), transferred to any test move by bin; checked against
  the two LB facts (C1 -> C2, C12 -> C18); pre-registered proxies per group (task spec): Telemed -> dynamic;
  others -> clean10 2-rater, VL lodo3, train142; a post-hoc 'regime-matched' variant is reported separately.

Subcommands
-----------
repro     rebuild C12 from the caches and C18 = C12 with Telemed FL at lambda 0; compare with the submitted
          files (byte identity and official distance)
analyse   test shrinkage table, benchmark tables, efficiencies, LB checks, per-group decomposition
build     C20..C24 (single group-level changes from C18) + diagnostics, validated (scripts/validate_submission.py
          + clip-constancy / bound checks), distances, public-LB sampling SDs (b4.lb_sd, rho 0.6 / 1), rater
          offsets re-estimated by the pre-registered rule, expected gains, EVSI, ranking, adoption thresholds
"""
from __future__ import annotations

import argparse
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
import build_candidates_v4 as b4  # noqa: E402

warnings.filterwarnings("ignore")
b3, bc, sh, rc = b4.b3, b4.bc, b4.sh, b4.rc
OUT = ROOT / "artifacts/candidates_v7"
C12_FILE = ROOT / "artifacts/candidates_v5/C12_hyb_med_ens_pp_muscle5_emclean_off.csv"
C18_FILE = ROOT / "artifacts/candidates_v6/C18_C12_telemedFLraw.csv"
PER_IMAGE_C12 = ROOT / "artifacts/candidates_v5/per_image_C12.csv"
C3 = ("pa_deg", "fl_mm", "mt_mm")
TAU = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
CT = {"pa": "pa_deg", "fl": "fl_mm"}
GROUPS = ("siemens_juniper", "telemed_artus_1200x800", "telemed_artus_1088x644", "philips_lumify", "legacy_853",
          "legacy_512")
TELEMED = ("telemed_artus_1200x800", "telemed_artus_1088x644")
NON_TELEMED = tuple(g for g in GROUPS if g not in TELEMED)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
KAPPAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
N_TEST, N_PUB = 309, 102
LB = {"v1": 0.46718, "C1": 0.45436, "C2": 0.38974, "C12": 0.37670, "C11": 0.51239, "C13": 0.41184,
      "C15": 0.41182, "C17": 0.38091, "C18": 0.33526, "C19": 0.41013}
REF_FILES = {"C1": ROOT / "artifacts/candidates/C1_ours_v2e_fusion_ensemble.csv",
             "C2": ROOT / "artifacts/candidates/C2_ours_v2e_fusion_ensemble_shrink1.0.csv",
             "C11": ROOT / "artifacts/candidates_v4/C11_fullshrink_cellmedian.csv",
             "C12": C12_FILE,
             "C13": ROOT / "artifacts/candidates_v5/C13_C12_telemedT1.csv",
             "C17": ROOT / "artifacts/candidates_v6/C17_C12_nooff.csv",
             "C18": C18_FILE,
             "C19": ROOT / "artifacts/candidates_v6/C19_C12_telemedFLcoupling.csv"}
BOOT_B = 2000


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS


def read_sub(p: Path) -> pd.DataFrame:
    return pd.read_csv(p).set_index("image_id")


def lay() -> pd.DataFrame:
    return b3.layout()


# =================================================================================== test state
_TS: dict = {}


def c12_recipe(offset="rec5") -> dict:
    return dict(b4.c12_base(), offset=offset)


def test_state() -> dict:
    """C12 rebuilt from the caches: per-image frame P (raw + shrunk), the C12 submission, the offsets."""
    if not _TS:
        r = c12_recipe()
        sub, P, pa_cell = b4.test_run5(r)
        _TS.update(r=r, sub=sub, P=P, pa_cell=pa_cell, off=b4.offset_of5(r))
    return _TS


def blend(raw: pd.Series, eb: pd.Series, lam) -> pd.Series:
    """raw + lam (eb - raw) where raw is finite (else the filled shrunk value). lam: scalar or Series."""
    raw = pd.to_numeric(raw, errors="coerce").astype(float)
    eb = pd.to_numeric(eb, errors="coerce").astype(float)
    return (raw + lam * (eb - raw)).where(raw.notna(), eb)


def fl_unit_raw(P: pd.DataFrame) -> pd.Series:
    """The raw FL of each row's EB unit (the clip median of the per-image FL for clip units, else the image's)."""
    return pd.to_numeric(P["fl_mm"], errors="coerce").groupby(P["eb_unit"]).transform("median")


def test_sub_lam(lam_pa: dict, lam_fl: dict, off: dict | None = None, P: pd.DataFrame | None = None,
                 pa_add_by_group: dict | None = None, lam_fl_row: pd.Series | None = None,
                 pa_eb_override: pd.Series | None = None) -> pd.DataFrame:
    """Test submission with per-group shrinkage strengths (dict group -> lambda; missing groups = 1).
    off = the rater offsets (default: C12's); pa_add_by_group overrides pa_add per group (optional);
    lam_fl_row = per-row FL lambda (overrides lam_fl where not NaN); pa_eb_override = a per-image shrunk PA
    that replaces the current one where not NaN (then lam_pa applies to it)."""
    import eval_shrinkage as es
    ts = test_state()
    P = ts["P"] if P is None else P
    off = ts["off"] if off is None else off
    g = P["group"].astype(str)
    lp = g.map(lambda x: lam_pa.get(x, 1.0)).astype(float)
    lf = g.map(lambda x: lam_fl.get(x, 1.0)).astype(float)
    if lam_fl_row is not None:
        lf = lam_fl_row.reindex(P.index).astype(float).fillna(lf)
    pa_eb = P["pa_eb"] if pa_eb_override is None else pa_eb_override.reindex(P.index).fillna(P["pa_eb"])
    Q = P.copy()
    Q["pa_eb"] = blend(P["pa_deg"], pa_eb, lp)
    Q["fl_eb"] = blend(fl_unit_raw(P), P["fl_eb"], lf)
    sub = es.to_sub(Q)
    if off:
        sub = rc.apply_offset(sub, off)
        if pa_add_by_group:
            gg = lay().loc[sub.index, "group"].astype(str)
            extra = gg.map(lambda x: pa_add_by_group.get(x, off.get("pa_add", 0.0)) - off.get("pa_add", 0.0))
            sub["pa_deg"] = sub["pa_deg"] + extra.astype(float)
        for c in C3:
            sub[c] = sub[c].clip(*bc.BOUNDS[c])
    return sub


C18_LAM = dict(pa={}, fl={g: 0.0 for g in TELEMED})


def c18_sub() -> pd.DataFrame:
    return test_sub_lam(C18_LAM["pa"], C18_LAM["fl"])


def write_like_c18(sub: pd.DataFrame, path: Path) -> None:
    """The C18 file format: C12's 4-decimal values, Telemed FL at full precision, pandas float repr, CRLF."""
    c12 = test_sub_lam({}, {})
    out = c12[list(C3)].round(4)
    tel = lay().loc[out.index, "group"].isin(TELEMED)
    out.loc[tel, "fl_mm"] = sub.loc[tel[tel].index, "fl_mm"]
    ids = sorted(p.name for p in bc.TEST_DIR.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png"})
    out.loc[ids].reset_index().rename(columns={"index": "image_id"}).to_csv(path, index=False, lineterminator="\r\n")


def official(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    return bc.official_distance(a, b)


def per_image_dist(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    b = b.reindex(a.index)
    return np.mean(np.stack([np.abs(a[c] - b[c]).to_numpy(float) / TAU[c] for c in C3], 1), 1)


def lb_sd(a: pd.DataFrame, b: pd.DataFrame, rho: float) -> dict:
    return b4.lb_sd(a, b, rho)


# =================================================================================== repro
def cmd_repro(args) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostic").mkdir(exist_ok=True)
    ts = test_state()
    c12_file, c18_file = read_sub(C12_FILE), read_sub(C18_FILE)
    c12 = test_sub_lam({}, {})
    c18 = c18_sub()
    rep = {"recipe": {k: v for k, v in ts["r"].items() if k != "desc"}, "offsets": ts["off"]}
    rep["C12_rebuilt_vs_file_maxabs"] = {c: float((c12[c] - c12_file.reindex(c12.index)[c]).abs().max()) for c in C3}
    rep["C12_rebuilt_vs_file_dist"] = official(c12, c12_file)["S"]
    rep["C12_lam_identity_vs_test_run5"] = float(np.nanmax(np.abs(c12[list(C3)].to_numpy() -
                                                                  ts["sub"].reindex(c12.index)[list(C3)].to_numpy())))
    d = (c18 - c18_file.reindex(c18.index)).abs()
    tel = lay().loc[c18.index, "group"].isin(TELEMED)
    rep["C18_rebuilt_vs_file_maxabs"] = {c: float(d[c].max()) for c in C3}
    rep["C18_rebuilt_vs_file_maxabs_telemed_fl"] = float(d.loc[tel, "fl_mm"].max())
    dist = per_image_dist(c18, c18_file)
    rep["C18_rebuilt_vs_file_dist_exact"] = float(dist.mean())
    # per_image_C12.csv route (the main session's construction): Telemed FL = clip median of fl_mm
    pi = pd.read_csv(PER_IMAGE_C12).set_index("image_id")
    tp = pi[pi.group.isin(TELEMED)]
    med = tp.groupby("clip")["fl_mm"].median()
    rep["C18_file_vs_per_image_clipmedian_maxabs"] = float(
        (c18_file.loc[tp.index, "fl_mm"] - tp["clip"].map(med)).abs().max())
    p = OUT / "diagnostic" / "D_C18_rebuilt.csv"
    write_like_c18(c18, p)
    rep["C18_rebuilt_byte_identical"] = p.read_bytes() == C18_FILE.read_bytes()
    rep["C18_rebuilt_file_vs_file_dist_exact"] = float(per_image_dist(read_sub(p), c18_file).mean())
    rep["C18_rebuilt_file_vs_file_maxabs"] = float((read_sub(p) - c18_file).abs().max().max())
    if not rep["C18_rebuilt_byte_identical"]:
        a, b = p.read_bytes().splitlines(), C18_FILE.read_bytes().splitlines()
        rep["C18_rebuilt_lines_differing"] = int(sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)))
        rep["C18_rebuilt_note"] = ("differing lines are Telemed FL values that differ in the last binary digit "
                                   "(17th significant digit) from the main session's per_image_C12.csv route")
    p2 = OUT / "diagnostic" / "D_C18_rebuilt_4dp.csv"
    bc.write_sub(c18, p2)
    rep["C18_4dp_validate"] = b3.validate(p2)
    rep["C18_4dp_vs_file_dist_exact"] = float(per_image_dist(read_sub(p2), c18_file).mean())
    print(json.dumps(rep, indent=1, default=float))
    (OUT / "repro_report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


# =================================================================================== test table
def test_table() -> pd.DataFrame:
    """Per group x component: what the current shrinkage does on test (C12 shrinkage; C18 state noted)."""
    ts = test_state()
    P = ts["P"]
    L = lay()
    rows = []
    c12 = test_sub_lam({}, {})
    for comp in ("pa", "fl"):
        raw_all = test_sub_lam({g: 0.0 for g in GROUPS} if comp == "pa" else {},
                               {g: 0.0 for g in GROUPS} if comp == "fl" else {})
        col = CT[comp]
        for g in GROUPS:
            ids = L.index[L.group == g]
            Pg = P.loc[ids]
            sh_ = (c12.loc[ids, col] - raw_all.loc[ids, col])
            w = Pg["w_pa"] if comp == "pa" else Pg["w_fl"]
            units = Pg["eb_unit"].nunique() if comp == "fl" else (Pg["clip"].nunique() if Pg["clip"].max() >= 0 else len(ids))
            in_c18 = 0.0 if (comp == "fl" and g in TELEMED) else 1.0
            rows.append(dict(group=g, comp=comp.upper(), n=len(ids), units=int(units),
                             w_mean=float(w.mean()), w_median=float(w.median()), frac_w_lt_0p5=float((w < 0.5).mean()),
                             shift_mean_abs=float(sh_.abs().mean()), shift_mean_signed=float(sh_.mean()),
                             shift_p90_abs=float(sh_.abs().quantile(0.9)), shift_max_abs=float(sh_.abs().max()),
                             raw_median=float(raw_all.loc[ids, col].median()), raw_iqr=float(
                                 raw_all.loc[ids, col].quantile(.75) - raw_all.loc[ids, col].quantile(.25)),
                             shrunk_iqr=float(c12.loc[ids, col].quantile(.75) - c12.loc[ids, col].quantile(.25)),
                             dist_contrib=float(sh_.abs().sum() / TAU[col] / 3 / N_TEST),
                             lam_in_C18=in_c18))
    return pd.DataFrame(rows)


def fl_at_raw_pa() -> pd.DataFrame:
    """Diagnostic: how much ens_pp FL would move if the constructed fascicle were rebuilt at the RAW PA
    (instead of the post-shrinkage PA used by C12/C18), per group (per-unit medians)."""
    P = test_state()["P"]
    fl_raw_pa = b3.nanmean_w([b3._col(P, "fl_straight"), b3._col(P, "fl_central3"), b3.cons_at(P, P["pa_deep_raw"])],
                             [1.0, 1.0, 1.0])
    fl_raw_pa[b3._col(P, "mt_mm").isna()] = np.nan
    a = fl_raw_pa.groupby(P["eb_unit"]).transform("median")
    b = fl_unit_raw(P)
    d = (a - b)
    return pd.DataFrame({"n": d.groupby(P.group).size(), "dFL_mean": d.groupby(P.group).mean(),
                         "dFL_mean_abs": d.abs().groupby(P.group).mean(), "dFL_max_abs": d.abs().groupby(P.group).max()})


# =================================================================================== bench frames
def _strip(P: pd.DataFrame) -> pd.DataFrame:
    P = P.copy()
    P.index = [i.split(":", 1)[1] if ":" in str(i) else i for i in P.index]
    return P


def train142_P(r: dict, old: bool = False) -> pd.DataFrame:
    """train142 per-image frame through the recipe (train142_q5 for the v5 path; the b3 path for old maps)."""
    x, _, _ = rc.train142_frame()
    rr = b4.b3rec(r) if not old else r
    d = b3.est_frame(x, "train142", rr)
    lay_ = x["layout"].astype(str).str.replace("1234x687", "1234x688")
    d["group"] = lay_.where(lay_.map(lay_.value_counts()) >= 5, "other")
    d["dataset"], d["clip"] = "train142", -1
    if old:
        return b3.apply_shrink(d, r["shrink"], None, "train142")
    empty = pd.Series("", index=d.index, dtype=object)
    d = b4.post_group(d, r, empty)
    Bf = b3.bench_frame_v4(rr, "lodo4")
    return b4.apply_m5(d, r, b4.fit_em5(Bf, r), empty, empty, b4.flc_params(r, Bf))


def bench_frames(old: bool = False) -> dict:
    """{static, vl3, vl4, dyn, t142}: per-image frames with raw (pa_deg, fl_mm) and shrunk (pa_eb, fl_eb).
    old=False: the C12 recipe on the current maps (LODO constants, as S_sel). old=True: the C2 recipe
    (old 'ours' maps, fusion PA, C1 FL, confidence shrink of PA + FL)."""
    if old:
        rec = b3.R("ours", "fusion", "c1", "conf", name="C2")
        res = b3.run_bench_v4(rec, "lodo")
        t142 = train142_P(rec, old=True)
    else:
        r = c12_recipe(offset=None)
        res = b4.bench_preds5(r)["res"]
        t142 = train142_P(r)
    return {"static": _strip(res[("lodo4", "static")]), "vl3": _strip(res[("lodo3", "vl")]),
            "vl4": _strip(pd.concat([res[("lodo4", "vl_young")], res[("lodo4", "vl_old")]])),
            "dyn": _strip(res[("lodo4", "dynamic")]), "t142": t142}


def truths() -> dict:
    RS = rc.static_raters()
    two = {c: rc.two_rater(RS[c]) for c in ("pa_deg", "fl_mm")}
    G7 = bc.static_gts()["mean7"]
    _, G142, grp142 = rc.train142_frame()
    return dict(two=two, mean7=G7, vl=bc.vl_gt(), dyn=bc.dynamic_gt(), t142=G142, t142_grp=grp142)


# benchmark slices: name -> (frame key, truth, id filter, cluster rule)
SLICES = {
    "clean10_2r": ("static", "two", "clean", "image"),
    "all35_2r*": ("static", "two", "all", "image"),
    "clean10_mean7": ("static", "mean7", "clean", "image"),
    "vl3": ("vl3", "vl", "all", "subject"),
    "vl4": ("vl4", "vl", "all", "subject"),
    "dyn": ("dyn", "dyn", "all", "block"),
    "t142": ("t142", "t142", "all", "t142"),
    "t142_512x512": ("t142", "t142", "512x512", "t142"),
    "t142_800x652": ("t142", "t142", "800x652", "t142"),
    "t142_800x600": ("t142", "t142", "800x600", "t142"),
}
PROXY = {   # test group -> pre-stated proxy slices (task spec)
    "telemed": ["dyn", "clean10_2r"],
    "non_telemed": ["clean10_2r", "vl3", "vl4", "t142"],
}


def slice_ids(F: dict, T: dict, name: str) -> tuple[list, str]:
    key, tr, filt, _ = SLICES[name]
    P = F[key]
    if tr == "two" or tr == "mean7":
        ids = [i for i in P.index if (i in bc.CLEAN) or filt == "all"]
    elif tr == "vl":
        ids = sorted(i for i in P.index if i in T["vl"].index)
    elif tr == "dyn":
        ids = sorted(i for i in P.index if i in T["dyn"].index)
    else:
        ids = [i for i in P.index if i in T["t142"].index]
        if filt != "all":
            ids = [i for i in ids if str(P.loc[i, "group"]) == filt]
    return ids, key


def err_array(pred: pd.Series, name: str, col: str, ids: list, T: dict) -> np.ndarray:
    """(n, K) |pred - truth| / tau (K = 6 two-rater partners on static, else 1)."""
    tr = SLICES[name][1]
    p = pred.reindex(ids).to_numpy(float)
    if tr == "two":
        t = T["two"][col].reindex(ids).to_numpy(float)
        return np.abs(t - p[:, None]) / TAU[col]
    t = {"mean7": T["mean7"], "vl": T["vl"], "dyn": T["dyn"], "t142": T["t142"]}[tr][col].reindex(ids).to_numpy(float)
    return (np.abs(t - p) / TAU[col])[:, None]


def boot_W(name: str, ids: list, T: dict, rng) -> np.ndarray:
    rule = SLICES[name][3]
    n = len(ids)
    if rule == "subject":
        return rc.boot_counts(n, BOOT_B, rng, clusters=pd.Index(ids).str.split("__").str[1].to_numpy())
    if rule == "block":
        return rc.boot_counts(n, BOOT_B, rng, block=rc.DYN_BLOCK)
    if rule == "t142":
        return rc.boot_counts(n, BOOT_B, rng, clusters=T["t142_grp"].reindex(ids).to_numpy())
    return rc.boot_counts(n, BOOT_B, rng)


def wscore(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(B,) weighted mean |e|/tau over images (and partners) with NaN handling."""
    v = np.isfinite(A)
    num = W @ np.nan_to_num(A)          # (B, K)
    den = W @ v.astype(float)
    return np.nanmean(num / np.where(den > 0, den, np.nan), axis=1)


def pt(A: np.ndarray) -> float:
    return float(wscore(A, np.ones((1, A.shape[0])))[0])


def comp_pred(P: pd.DataFrame, comp: str, lam=1.0, kappa: float | None = None, off: dict | None = None) -> pd.Series:
    """Benchmark prediction of one component at shrinkage strength lam (or a uniform shrink fraction kappa
    toward the pipeline's centre: the group median of the raw estimates), + the rater offset for PA."""
    raw = pd.to_numeric(P[CT[comp]], errors="coerce").astype(float)
    eb = pd.to_numeric(P[f"{comp}_eb"], errors="coerce").astype(float)
    if kappa is None:
        v = blend(raw, eb, lam)
    else:
        c = raw.groupby(P["group"].astype(str)).transform("median")
        v = (raw + kappa * (c - raw)).where(raw.notna(), eb)
    if comp == "pa" and off:
        v = v * off.get("pa_mult", 1.0) + off.get("pa_add", 0.0)
    if comp == "fl" and off:
        v = v * off.get("fl_mult", 1.0) + off.get("fl_add", 0.0)
    return v


def wcol(P: pd.DataFrame, comp: str) -> pd.Series:
    """The image's shrink weight on its own measurement (confidence rule: the confidence; EB: w_fl)."""
    if comp == "fl" and "w_fl" in P:
        return pd.to_numeric(P["w_fl"], errors="coerce")
    return pd.to_numeric(P["confidence"], errors="coerce").fillna(0).clip(0, 1)


def lam_curves(F: dict, T: dict, off: dict | None, tag: str, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per slice x component: score at each lambda (and uniform kappa), paired bootstrap CIs of
    score(lambda) - score(1) and of score(0) - score(1); the shrink weights of the slice."""
    rows, krows = [], []
    for name in SLICES:
        ids, key = slice_ids(F, T, name)
        if not ids:
            continue
        P = F[key]
        rng = np.random.default_rng(seed + hash(name) % 1000)
        W = boot_W(name, ids, T, rng)
        for comp in ("pa", "fl"):
            col = CT[comp]
            A1 = err_array(comp_pred(P, comp, 1.0, off=off), name, col, ids, T)
            b1 = wscore(A1, W)
            w = wcol(P, comp).reindex(ids)
            shift = (comp_pred(P, comp, 1.0, off=off) - comp_pred(P, comp, 0.0, off=off)).reindex(ids).abs()
            base = dict(tag=tag, slice=name, comp=comp.upper(), n=len(ids), w_mean=float(w.mean()),
                        shrink_frac_mean=float((1 - w).mean()), shift_mean_abs=float(shift.mean()))
            for lam in LAMBDAS:
                A = err_array(comp_pred(P, comp, lam, off=off), name, col, ids, T)
                d = wscore(A, W) - b1
                lo, hi = rc.ci(d)
                rows.append(dict(base, lam=lam, score=pt(A), d_vs_lam1=pt(A) - pt(A1), lo=lo, hi=hi,
                                 P_better_than_lam1=float(np.mean(d < 0))))
            # fine grid optimum (point + bootstrap distribution of the argmin)
            grid = np.round(np.arange(0, 1.501, 0.05), 2)
            S = np.stack([wscore(err_array(comp_pred(P, comp, g, off=off), name, col, ids, T), W) for g in grid], 1)
            sp = [pt(err_array(comp_pred(P, comp, g, off=off), name, col, ids, T)) for g in grid]
            am = grid[np.nanargmin(S, axis=1)]
            rows.append(dict(base, lam=float(grid[int(np.nanargmin(sp))]), score=float(np.nanmin(sp)),
                             d_vs_lam1=float(np.nanmin(sp) - pt(A1)), lo=float(np.percentile(am, 2.5)),
                             hi=float(np.percentile(am, 97.5)), P_better_than_lam1=np.nan, note="argmin (lo/hi = 95% boot range of argmin)"))
            A0 = err_array(comp_pred(P, comp, 0.0, off=off), name, col, ids, T)
            b0 = wscore(A0, W)
            for k in KAPPAS:
                A = err_array(comp_pred(P, comp, kappa=k, off=off), name, col, ids, T)
                d = wscore(A, W) - b0
                lo, hi = rc.ci(d)
                krows.append(dict(base, kappa=k, score=pt(A), d_vs_raw=pt(A) - pt(A0), lo=lo, hi=hi,
                                  P_better_than_raw=float(np.mean(d < 0))))
    return pd.DataFrame(rows), pd.DataFrame(krows)


WBINS = [0, 0.5, 0.7, 0.85, 0.95, 1.0001]


def wstrat(F: dict, T: dict, off: dict | None, tag: str, seed: int = 1) -> pd.DataFrame:
    """Shrinkage gain (lambda 0 minus lambda 1; > 0 = shrinkage helps) by the image's own shrink weight,
    on the honest slices (clean10 2r, VL lodo3, dynamic, train142), with paired bootstrap CIs."""
    rows = []
    for name in ("clean10_2r", "vl3", "dyn", "t142"):
        ids, key = slice_ids(F, T, name)
        P = F[key]
        for comp in ("pa", "fl"):
            col = CT[comp]
            w = wcol(P, comp).reindex(ids)
            A1 = err_array(comp_pred(P, comp, 1.0, off=off), name, col, ids, T)
            A0 = err_array(comp_pred(P, comp, 0.0, off=off), name, col, ids, T)
            Dg = np.nanmean(A0, 1) - np.nanmean(A1, 1)      # per-image gain of shrinkage (score units of comp)
            b = pd.cut(w, WBINS, right=False)
            for bb, m in pd.Series(np.arange(len(ids)), index=ids).groupby(b.to_numpy()):
                ii = m.to_numpy()
                if not len(ii):
                    continue
                sub_ids = [ids[i] for i in ii]
                rng = np.random.default_rng(seed)
                Wb = boot_W(name, sub_ids, T, rng)
                g = Wb @ np.nan_to_num(Dg[ii]) / np.maximum(Wb @ np.isfinite(Dg[ii]).astype(float), 1e-9)
                lo, hi = rc.ci(g)
                rows.append(dict(tag=tag, slice=name, comp=comp.upper(), w_bin=str(bb), n=len(ii),
                                 w_mean=float(w.iloc[ii].mean()), gain_mean=float(np.nanmean(Dg[ii])), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def test_wbins() -> pd.DataFrame:
    P = test_state()["P"]
    rows = []
    for comp, wc in (("PA", "w_pa"), ("FL", "w_fl")):
        b = pd.cut(P[wc], WBINS, right=False)
        ct = pd.crosstab(P["group"], b)
        for g in ct.index:
            for bb in ct.columns:
                rows.append(dict(comp=comp, group=g, w_bin=str(bb), n=int(ct.loc[g, bb])))
    return pd.DataFrame(rows)


# =================================================================================== rule variants
FL_ONLY_FLAGS = {"high_extrapolation": 0.3, "fl_spread": 0.3, "fl_triangle_disagree": 0.3}   # geometry_v2 penalties
RATIO_BINS = [0, 1.5, 2.0, 2.25, 2.5, 100]


def pa_conf(P: pd.DataFrame) -> pd.Series:
    """PA-specific confidence: the geometry confidence exp(-sum of penalties) without the penalties of the
    FL-only flags (high_extrapolation, fl_spread, fl_triangle_disagree; geometry_v2 lines 1437-1465)."""
    conf = pd.to_numeric(P["confidence"], errors="coerce").clip(1e-6, 1.0)
    fl = P["flags"].fillna("").astype(str).str.split(";")
    fl_pen = fl.apply(lambda xs: sum(FL_ONLY_FLAGS.get(x, 0.0) for x in xs))
    pen = (-np.log(conf) - fl_pen).clip(lower=0.0)
    return np.exp(-pen).where(conf.notna())


def pa_pred_paconf(P: pd.DataFrame, off: dict | None, cell: pd.Series | None = None) -> pd.Series:
    """The confidence rule with the PA-specific confidence (same centres as the pipeline)."""
    raw = pd.to_numeric(P["pa_deg"], errors="coerce").astype(float)
    cell = pd.Series("", index=P.index, dtype=object) if cell is None else cell.reindex(P.index).fillna("")
    v = b4.conf_rule(raw, pa_conf(P), P["group"].astype(str), cell)
    v = v.where(v.notna(), pd.to_numeric(P["pa_eb"], errors="coerce"))
    if off:
        v = v * off.get("pa_mult", 1.0) + off.get("pa_add", 0.0)
    return v


def fl_ratio(P: pd.DataFrame) -> pd.Series:
    return fl_unit_raw(P) / sh._fov_mm(P) if "eb_unit" in P else \
        pd.to_numeric(P["fl_mm"], errors="coerce") / sh._fov_mm(P)


def fl_pred_gated(P: pd.DataFrame, r0: float) -> pd.Series:
    """FL: the current shrinkage only where the raw FL/FOV > r0, raw elsewhere."""
    raw = fl_unit_raw(P) if "eb_unit" in P else pd.to_numeric(P["fl_mm"], errors="coerce")
    lam = (fl_ratio(P) > r0).astype(float)
    return blend(raw, P["fl_eb"], lam)


def strat_tables(F: dict, T: dict, off: dict | None, tag: str, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(1) FL shrinkage gain (lam 0 - lam 1, > 0 = shrinkage helps) by raw FL/FOV bin; (2) PA shrinkage gain
    for images whose confidence is < 1 only through FL-only flags vs through PA-relevant flags."""
    r1, r2 = [], []
    for name in ("clean10_2r", "vl3", "dyn", "t142"):
        ids, key = slice_ids(F, T, name)
        P = F[key]
        A1 = err_array(comp_pred(P, "fl", 1.0, off=off), name, "fl_mm", ids, T)
        A0 = err_array(comp_pred(P, "fl", 0.0, off=off), name, "fl_mm", ids, T)
        g = np.nanmean(A0, 1) - np.nanmean(A1, 1)
        rat = fl_ratio(P).reindex(ids)
        b = pd.cut(rat, RATIO_BINS)
        for bb, m in pd.Series(np.arange(len(ids)), index=ids).groupby(b.to_numpy()):
            ii = m.to_numpy()
            if not len(ii):
                continue
            Wb = boot_W(name, [ids[i] for i in ii], T, np.random.default_rng(seed))
            gb = Wb @ np.nan_to_num(g[ii]) / np.maximum(Wb @ np.isfinite(g[ii]).astype(float), 1e-9)
            lo, hi = rc.ci(gb)
            shift = (comp_pred(P, "fl", 1.0) - comp_pred(P, "fl", 0.0)).reindex(ids).iloc[ii]
            r1.append(dict(tag=tag, slice=name, ratio_bin=str(bb), n=len(ii), w_fl_mean=float(wcol(P, "fl").reindex(ids).iloc[ii].mean()),
                           shift_signed_mean=float(shift.mean()), gain_mean=float(np.nanmean(g[ii])), lo=lo, hi=hi))
        A1 = err_array(comp_pred(P, "pa", 1.0, off=off), name, "pa_deg", ids, T)
        A0 = err_array(comp_pred(P, "pa", 0.0, off=off), name, "pa_deg", ids, T)
        g = np.nanmean(A0, 1) - np.nanmean(A1, 1)
        conf = pd.to_numeric(P["confidence"], errors="coerce").reindex(ids)
        pc = pa_conf(P).reindex(ids)
        kind = np.where(conf >= 0.999, "conf=1", np.where(pc >= 0.999, "FL-only flags", "PA-relevant flags"))
        for kk in ("conf=1", "FL-only flags", "PA-relevant flags"):
            ii = np.where(kind == kk)[0]
            if not len(ii):
                continue
            Wb = boot_W(name, [ids[i] for i in ii], T, np.random.default_rng(seed))
            gb = Wb @ np.nan_to_num(g[ii]) / np.maximum(Wb @ np.isfinite(g[ii]).astype(float), 1e-9)
            lo, hi = rc.ci(gb)
            r2.append(dict(tag=tag, slice=name, kind=kk, n=len(ii), conf_mean=float(conf.iloc[ii].mean()),
                           pa_conf_mean=float(pc.iloc[ii].mean()), gain_mean=float(np.nanmean(g[ii])), lo=lo, hi=hi))
    return pd.DataFrame(r1), pd.DataFrame(r2)


def rule_bench(F: dict, T: dict, off: dict | None, tag: str, seed: int = 4) -> pd.DataFrame:
    """Rule variants vs the current shrinkage (C12/C18 recipe) on every slice: PA with the PA-specific
    confidence; FL shrunk only above a raw FL/FOV threshold; no shrinkage (lam 0). d < 0 = variant better."""
    rows = []
    for name in SLICES:
        ids, key = slice_ids(F, T, name)
        if not ids:
            continue
        P = F[key]
        W = boot_W(name, ids, T, np.random.default_rng(seed))
        variants = {("PA", "pa_conf rule"): pa_pred_paconf(P, off), ("PA", "lam0"): comp_pred(P, "pa", 0.0, off=off)}
        for r0 in (2.0, 2.25, 2.5):
            variants[("FL", f"gate FL/FOV>{r0}")] = fl_pred_gated(P, r0)
        variants[("FL", "lam0")] = comp_pred(P, "fl", 0.0, off=off)
        for (comp, vn), pred in variants.items():
            col = CT[comp.lower()]
            A1 = err_array(comp_pred(P, comp.lower(), 1.0, off=off), name, col, ids, T)
            A = err_array(pred, name, col, ids, T)
            d = wscore(A, W) - wscore(A1, W)
            lo, hi = rc.ci(d)
            mv = (pred - comp_pred(P, comp.lower(), 1.0, off=off)).reindex(ids).abs()
            rows.append(dict(tag=tag, slice=name, comp=comp, variant=vn, n=len(ids), moved_frac=float((mv > 1e-6).mean()),
                             move_mean_abs=float(mv.mean()), score_cur=pt(A1), score_var=pt(A), d=pt(A) - pt(A1), lo=lo, hi=hi,
                             P_better=float(np.mean(d < 0))))
    return pd.DataFrame(rows)


# =================================================================================== transfer model
# Efficiency of the shrinkage on a proxy = sum_i (|e_i(lam 0)| - |e_i(lam 1)|) / sum_i |pred_i(lam 1) - pred_i(lam 0)|
# (both / tau): the share of the moved metric mass that the shrinkage got right (1 = every move fully right,
# -1 = fully wrong). Estimated per bin (FL: raw FL/FOV; PA: why the confidence is < 1), transferred to a test
# move a -> b as  dS = -sum_bins eff_bin * dist_bin  (raw -> shrunk) or  +sum eff_bin * dist_bin (shrunk -> raw).
PROXIES = ("clean10_2r", "vl3", "dyn", "t142")
MIN_BIN = 5


def pa_kind(P: pd.DataFrame) -> pd.Series:
    conf = pd.to_numeric(P["confidence"], errors="coerce")
    pc = pa_conf(P)
    return pd.Series(np.where(conf >= 0.999, "conf=1", np.where(pc >= 0.999, "FL-only flags", "PA-relevant flags")),
                     index=P.index)


def fl_bin(P: pd.DataFrame) -> pd.Series:
    c = pd.cut(fl_ratio(P), RATIO_BINS).astype(object)
    return c.where(c.notna(), "(nan)").astype(str)


def bin_of(P: pd.DataFrame, comp: str) -> pd.Series:
    return pa_kind(P) if comp == "pa" else fl_bin(P)


def efficiencies(F: dict, T: dict, off: dict | None, tag: str, seed: int = 5) -> pd.DataFrame:
    rows = []
    for name in PROXIES:
        ids, key = slice_ids(F, T, name)
        P = F[key]
        W = boot_W(name, ids, T, np.random.default_rng(seed))
        for comp in ("pa", "fl"):
            col = CT[comp]
            p1, p0 = comp_pred(P, comp, 1.0, off=off), comp_pred(P, comp, 0.0, off=off)
            g = np.nanmean(err_array(p0, name, col, ids, T), 1) - np.nanmean(err_array(p1, name, col, ids, T), 1)
            m = ((p1 - p0).abs() / TAU[col]).reindex(ids).to_numpy(float)
            g = np.where(np.isfinite(g), g, 0.0)
            m = np.where(np.isfinite(m), m, 0.0)
            b = bin_of(P, comp).reindex(ids).fillna("(nan)").astype(str).to_numpy()
            for bb in ["ALL"] + sorted(set(b)):
                msk = np.ones(len(ids), bool) if bb == "ALL" else (b == bb)
                Wm = W * msk[None, :]
                num, den = Wm @ g, Wm @ m
                eb = num / np.where(den > 1e-12, den, np.nan)
                lo, hi = rc.ci(eb)
                tot = float(m[msk].sum())
                rows.append(dict(tag=tag, proxy=name, comp=comp.upper(), bin=bb, n=int(msk.sum()),
                                 n_moved=int((m[msk] > 1e-9).sum()), mean_move=float(m[msk].mean()),
                                 eff=float(g[msk].sum() / tot) if tot > 1e-12 else np.nan, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def eff_lookup(E: pd.DataFrame, proxy: str, comp: str, bb: str) -> tuple[float, str]:
    q = E[(E.proxy == proxy) & (E.comp == comp.upper())]
    r = q[q.bin == bb]
    if len(r) and r.iloc[0].n_moved >= MIN_BIN and np.isfinite(r.iloc[0].eff):
        return float(r.iloc[0].eff), "bin"
    a = q[q.bin == "ALL"]
    if len(a) and np.isfinite(a.iloc[0].eff):
        return float(a.iloc[0].eff), "ALL"
    return 0.0, "none"


def move_bins(P: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, comp: str, rows: pd.Index) -> pd.Series:
    """Metric distance of the move a -> b on `rows` for one component, by the transfer bin."""
    col = CT[comp]
    d = (b.loc[rows, col] - a.loc[rows, col]).abs() / TAU[col] / 3 / N_TEST
    return d.groupby(bin_of(P, comp).reindex(rows).fillna("(nan)").astype(str).to_numpy()).sum()


def predict_move(E: pd.DataFrame, P: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, parts: list,
                 direction: str) -> pd.DataFrame:
    """parts: [(comp, groups)]; direction 'to_shrunk' (raw -> shrunk) or 'to_raw'. Predicted dS per proxy."""
    L = lay()
    sgn = -1.0 if direction == "to_shrunk" else 1.0
    out = []
    for proxy in PROXIES:
        tot, used = 0.0, []
        for comp, groups in parts:
            rows = L.index[L.group.isin(groups)]
            for bb, dist in move_bins(P, a, b, comp, rows).items():
                e, how = eff_lookup(E, proxy, comp, str(bb))
                tot += sgn * e * float(dist)
                used.append(f"{comp}:{bb}:{how}")
        out.append(dict(proxy=proxy, dS_pred=tot, fallback=";".join(u for u in used if not u.endswith(":bin"))))
    return pd.DataFrame(out)


NORMAL_FL_BINS = {"(0.0, 1.5]", "(1.5, 2.0]", "(2.0, 2.25]"}
# Post-hoc 'regime-matched' transfer (written after seeing the two LB facts; NOT pre-registered): FL in the
# validated regime (raw FL/FOV <= 2.25) -> the in-domain / Telemed-like proxies (t142, dyn); FL beyond it ->
# VL (the only proxy with such images); PA shrunk only through FL-only flags -> t142 (VL has no such regime
# difference); PA with PA-relevant flags -> mean(VL, t142).
REGIME = {("fl", "normal"): ("t142", "dyn"), ("fl", "long"): ("vl3",), ("pa", "FL-only flags"): ("t142",),
          ("pa", "PA-relevant flags"): ("vl3", "t142")}
PREREG = {"telemed": ("dyn",), "non_telemed": ("clean10_2r", "vl3", "t142")}   # task spec


def regime_eff(E: pd.DataFrame, comp: str, bb: str) -> float:
    key = (comp, ("normal" if bb in NORMAL_FL_BINS else "long")) if comp == "fl" else (comp, bb)
    vals = [eff_lookup(E, p, comp, bb)[0] for p in REGIME.get(key, ())]
    return float(np.mean(vals)) if vals else 0.0


def predict_parts(E: pd.DataFrame, P: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, comps=("pa", "fl"),
                  groups=GROUPS, sign_of_move: str = "auto") -> dict:
    """Predicted dS of the move a -> b, per proxy (pre-registered per group), and by the regime model.
    The move direction per row is read from the data: toward the shrunk value (|b - raw| < |a - raw|) counts
    as 'to_shrunk', away from it as 'to_raw' (so mixed moves, e.g. gates, are handled)."""
    L = lay()
    raw = test_sub_lam({g: 0.0 for g in GROUPS}, {g: 0.0 for g in GROUPS}, P=P) if "eb_unit" in P else None
    out = {k: 0.0 for k in PROXIES}
    out.update(regime=0.0, prereg=0.0)
    for comp in comps:
        col = CT[comp]
        for g in groups:
            rows = L.index[L.group == g]
            d = (b.loc[rows, col] - a.loc[rows, col]).abs() / TAU[col] / 3 / N_TEST
            if raw is not None:
                to_raw = ((b.loc[rows, col] - raw.loc[rows, col]).abs() < (a.loc[rows, col] - raw.loc[rows, col]).abs())
            else:
                to_raw = pd.Series(False, index=rows)
            sg = np.where(to_raw, 1.0, -1.0)
            bins = bin_of(P, comp).reindex(rows).fillna("(nan)").astype(str)
            pre = PREREG["telemed" if g in TELEMED else "non_telemed"]
            for bb in bins.unique():
                m = (bins == bb).to_numpy()
                mass = float((d.to_numpy() * sg)[m].sum())
                for pr in PROXIES:
                    out[pr] += eff_lookup(E, pr, comp, bb)[0] * mass
                out["regime"] += regime_eff(E, comp, bb) * mass
                out["prereg"] += float(np.mean([eff_lookup(E, pr, comp, bb)[0] for pr in pre])) * mass
    return out


def group_decomp(E: pd.DataFrame, Eo: pd.DataFrame, old: dict) -> pd.DataFrame:
    """Per group x component: the C1 -> C2 move (old maps, old proxies) and the C18 -> fully-unshrunk move
    (current maps, current proxies): distance and predicted dS by every proxy and the regime model."""
    ts = test_state()
    c18 = c18_sub()
    c1, Pc1 = old["C1"]
    c2, Pc2 = old["C2"]
    rows = []
    for g in GROUPS:
        for comp in ("pa", "fl"):
            col = CT[comp]
            ids = lay().index[lay().group == g]
            lp = {g: 0.0} if comp == "pa" else {}
            lf = dict(C18_LAM["fl"])
            if comp == "fl":
                lf[g] = 0.0
            raw = test_sub_lam(lp, lf)
            r = dict(group=g, comp=comp.upper(),
                     dist_C1C2=float((c2.loc[ids, col] - c1.loc[ids, col]).abs().sum() / TAU[col] / 3 / N_TEST),
                     dist_C18_unshrink=float((raw.loc[ids, col] - c18.loc[ids, col]).abs().sum() / TAU[col] / 3 / N_TEST))
            po = predict_parts(Eo, Pc2, c1, c2, comps=(comp,), groups=(g,))
            # old maps: the move C1 -> C2 is raw -> shrunk everywhere (predict_parts reads it from P's raw,
            # which for the old frame is not available, so it counts as to_shrunk: sign -1)
            pc = predict_parts(E, ts["P"], c18, raw, comps=(comp,), groups=(g,))
            for k in PROXIES + ("regime",):
                r[f"C1C2_{k}"] = po[k]
            for k in PROXIES + ("regime", "prereg"):
                r[f"unshrink_{k}"] = pc[k]
            rows.append(r)
    return pd.DataFrame(rows)


# =================================================================================== old maps (C1 / C2)
def old_test() -> dict:
    """C1 / C2 rebuilt from the v3 caches (old 'ours' maps) with their per-image frames."""
    out = {}
    for nm, shr in (("C1", "none"), ("C2", "conf")):
        sub, P, _ = b3.test_sub(b3.R("ours", "fusion", "c1", shr, name=nm), return_per_image=True)
        out[nm] = (sub, P)
    return out


def dispersion_table(old: dict) -> pd.DataFrame:
    """Raw per-unit spread per group: old maps (C1 per-image) vs current raw (C12 recipe, lambda 0)."""
    L = lay()
    new_raw = test_sub_lam({g: 0.0 for g in GROUPS}, {g: 0.0 for g in GROUPS})
    c1 = old["C1"][0]
    rows = []
    for g in GROUPS:
        ids = L.index[L.group == g]
        for col in ("pa_deg", "fl_mm"):
            for nm, s in (("old_raw_C1", c1), ("new_raw", new_raw), ("C18", read_sub(C18_FILE))):
                x = s.loc[ids, col]
                med = x.median()
                rows.append(dict(group=g, comp=col, source=nm, median=float(med),
                                 mad_sd=float(1.4826 * (x - med).abs().median()), sd=float(x.std()),
                                 frac_far=float(((x - med).abs() > (8.0 if col == "pa_deg" else 30.0)).mean())))
    return pd.DataFrame(rows)


def move_table(a: pd.DataFrame, b: pd.DataFrame, ref: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Per group x component: distance a -> b and how the move relates to a reference file (C18, our best LB):
    'toward' = |b - ref| < |a - ref| (share of the moved metric mass that went toward ref)."""
    L = lay()
    rows = []
    for g in GROUPS:
        ids = L.index[L.group == g]
        for col in ("pa_deg", "fl_mm", "mt_mm"):
            d = (b.loc[ids, col] - a.loc[ids, col]).abs()
            ea, eb = (a.loc[ids, col] - ref.loc[ids, col]).abs(), (b.loc[ids, col] - ref.loc[ids, col]).abs()
            m = d > 1e-6
            rows.append(dict(pair=tag, group=g, comp=col, n=len(ids), mean_abs_move=float(d.mean()),
                             dist_contrib=float(d.sum() / TAU[col] / 3 / N_TEST),
                             toward_ref_mass=float(((ea - eb).clip(lower=0) * m).sum() / max(d.sum(), 1e-12)),
                             net_toward_ref=float((ea - eb).sum() / max(d.sum(), 1e-12))))
    return pd.DataFrame(rows)


def lb_efficiency() -> pd.DataFrame:
    """LB difference / test distance for the scored pairs (efficiency in [-1, 1]: 1 = every move was fully
    right on the public images, assuming public ~ test)."""
    subs = {k: read_sub(p) for k, p in REF_FILES.items() if p.exists()}
    rows = []
    for a, b in (("C1", "C2"), ("C12", "C18"), ("C12", "C17"), ("C12", "C13"), ("C12", "C19"), ("C12", "C11"),
                 ("C2", "C12")):
        if a not in subs or b not in subs:
            continue
        d = official(subs[b], subs[a])
        s1 = lb_sd(subs[b], subs[a], 1.0)
        dlb = LB[b] - LB[a]
        rows.append(dict(a=a, b=b, lb_a=LB[a], lb_b=LB[b], d_lb=dlb, dist=d["S"], pa_n=d["pa_n"], fl_n=d["fl_n"],
                         mt_n=d["mt_n"], efficiency=-dlb / d["S"] if d["S"] > 0 else np.nan,
                         sd_pub_rho1=s1["sd_pub"]))
    return pd.DataFrame(rows)


# =================================================================================== analyse
_AN: dict = {}


def analysis_inputs() -> dict:
    """Benchmark frames (current and old maps), truths, efficiencies, old-map test frames (memoised)."""
    if not _AN:
        ts = test_state()
        T = truths()
        F, Fo = bench_frames(), bench_frames(old=True)
        _AN.update(T=T, F=F, Fo=Fo, E=efficiencies(F, T, ts["off"], "current"), Eo=efficiencies(Fo, T, None, "old_C2"),
                   old=old_test())
    return _AN


def lb_checks(A: dict) -> pd.DataFrame:
    """The two LB facts against every proxy's transfer prediction (and the post-hoc regime model)."""
    ts = test_state()
    c12, c18 = test_sub_lam({}, {}), c18_sub()
    c1, Pc1 = A["old"]["C1"]
    c2, Pc2 = A["old"]["C2"]
    rows = []
    for nm, E, P, a, b, parts, lb in (("C12->C18 (Telemed FL unshrunk; current maps)", A["E"], ts["P"], c12, c18,
                                       dict(comps=("fl",), groups=TELEMED), LB["C18"] - LB["C12"]),
                                      ("C1->C2 (PA+FL confidence shrink; old maps)", A["Eo"], Pc2, c1, c2,
                                       dict(comps=("pa", "fl"), groups=GROUPS), LB["C2"] - LB["C1"])):
        pr = predict_parts(E, P, a, b, **parts)
        s1 = lb_sd(b, a, 1.0)
        rows.append(dict(move=nm, d_lb=lb, sd_pub_rho1=s1["sd_pub"], dist=official(b, a)["S"],
                         **{f"pred_{k}": v for k, v in pr.items()}))
    return pd.DataFrame(rows)


def cmd_analyse(args) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 500)
    t0 = time.time()
    rep: dict = {}
    ts = test_state()
    TT = test_table()
    TT.to_csv(OUT / "test_shrink_by_group.csv", index=False)
    print("== test: what the current (C12) shrinkage does per group x component\n", TT.round(3).to_string(index=False))
    FR = fl_at_raw_pa()
    FR.to_csv(OUT / "test_fl_at_raw_pa.csv")
    print("== diagnostic: ens_pp FL rebuilt at the RAW PA minus FL at the shrunk PA (per unit)\n", FR.round(3).to_string())
    P = ts["P"]
    TB = pd.DataFrame({"group": P.group, "conf": P.confidence, "pa_conf": pa_conf(P), "pa_kind": pa_kind(P),
                       "w_fl": P.w_fl, "fl_ratio": fl_ratio(P), "fl_bin": fl_bin(P)})
    tb = pd.concat([pd.crosstab(TB.group, TB.pa_kind).add_prefix("PA:"), pd.crosstab(TB.group, TB.fl_bin).add_prefix("FL:"),
                    TB.groupby("group")[["conf", "pa_conf", "w_fl", "fl_ratio"]].mean().add_suffix("_mean")], axis=1)
    tb.to_csv(OUT / "test_bins_by_group.csv")
    print("== test rows per transfer bin\n", tb.round(3).to_string())
    test_wbins().to_csv(OUT / "test_wbins.csv", index=False)
    A = analysis_inputs()
    T, F, Fo = A["T"], A["F"], A["Fo"]
    LC, KC = lam_curves(F, T, ts["off"], "current")
    LCo, KCo = lam_curves(Fo, T, None, "old_C2")
    pd.concat([LC, LCo]).to_csv(OUT / "bench_lambda.csv", index=False)
    pd.concat([KC, KCo]).to_csv(OUT / "bench_kappa.csv", index=False)
    show = ["tag", "slice", "comp", "n", "w_mean", "shift_mean_abs", "lam", "score", "d_vs_lam1", "lo", "hi",
            "P_better_than_lam1"]
    print("== benchmark lambda curves (last row per slice x comp = argmin on a 0.05 grid; lo/hi = its boot range)\n",
          pd.concat([LC, LCo])[show].round(4).to_string(index=False))
    WS = pd.concat([wstrat(F, T, ts["off"], "current"), wstrat(Fo, T, None, "old_C2")])
    WS.to_csv(OUT / "bench_wstrat.csv", index=False)
    S1, S2 = strat_tables(F, T, ts["off"], "current")
    S1o, S2o = strat_tables(Fo, T, None, "old_C2")
    pd.concat([S1, S1o]).to_csv(OUT / "bench_fl_by_ratio.csv", index=False)
    pd.concat([S2, S2o]).to_csv(OUT / "bench_pa_by_flagkind.csv", index=False)
    print("== FL shrink gain by raw FL/FOV (lam0 - lam1 in FL-component units; > 0 = shrink helps)\n",
          pd.concat([S1, S1o]).round(4).to_string(index=False))
    print("== PA shrink gain by why the confidence is < 1\n", pd.concat([S2, S2o]).round(4).to_string(index=False))
    RB = pd.concat([rule_bench(F, T, ts["off"], "current"), rule_bench(Fo, T, None, "old_C2")])
    RB.to_csv(OUT / "bench_rules.csv", index=False)
    print("== rule variants vs the current shrinkage (d < 0 = variant better)\n",
          RB[RB.tag == "current"].round(4).to_string(index=False))
    EF = pd.concat([A["E"], A["Eo"]])
    EF.to_csv(OUT / "bench_efficiency.csv", index=False)
    print("== shrinkage efficiency per proxy x bin (share of the moved metric mass the shrinkage got right)\n",
          EF.round(3).to_string(index=False))
    old = A["old"]
    for nm in ("C1", "C2"):
        rep[f"{nm}_rebuilt_vs_file_dist"] = official(old[nm][0], read_sub(REF_FILES[nm]))["S"]
    print("old-map rebuild check:", {k: v for k, v in rep.items() if k.endswith("file_dist")})
    DT = dispersion_table({"C1": (read_sub(REF_FILES["C1"]), None)})
    DT.to_csv(OUT / "test_dispersion_old_new.csv", index=False)
    print("== raw dispersion per group (old C1 vs current raw vs C18)\n", DT.round(3).to_string(index=False))
    LBc = lb_checks(A)
    LBc.to_csv(OUT / "lb_checks.csv", index=False)
    print("== LB facts vs transfer predictions\n", LBc.round(4).to_string(index=False))
    GD = group_decomp(A["E"], A["Eo"], old)
    GD.to_csv(OUT / "transfer_by_group.csv", index=False)
    print("== per group x comp: C1->C2 predicted dS (old proxies) and C18 -> unshrunk predicted dS (current)\n",
          GD.round(4).to_string(index=False))
    print(GD.sum(numeric_only=True).round(4).to_string())
    EFF = lb_efficiency()
    EFF.to_csv(OUT / "lb_efficiency.csv", index=False)
    print("== LB efficiency of the scored pairs\n", EFF.round(4).to_string(index=False))
    rep["peak_rss_gb"] = rss_gb()
    rep["seconds"] = time.time() - t0
    (OUT / "analyse_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"peak RSS {rep['peak_rss_gb']:.2f} GB, {rep['seconds']:.0f}s")
    return rep


# =================================================================================== offsets (recompute)
def recompute_offsets(B: int = 4000) -> dict:
    """The pre-registered rater rule (rater_calibration.cmd_offsets) re-run on the C12 base recipe; C18 differs
    from C12 only in the Telemed TEST FL, so the benchmark inputs (and the estimates) are those of C12."""
    (OUT / "rater").mkdir(parents=True, exist_ok=True)
    r = b4.c12_base()
    rc.OUT = OUT / "rater"
    rc.rec_of = lambda name, spec=None: dict(r, offset=None)  # noqa: E731
    rc.bench_preds = lambda rec: {k: v for k, v in b4.bench_preds5(dict(rec, offset=None)).items()
                                  if k in ("static", "vl3", "dyn", "t142")}
    rc.cmd_offsets(argparse.Namespace(recipe="C18base", B=B, seed=0))
    dec = pd.read_csv(OUT / "rater" / "offsets_C18base_decision.csv")
    est = {(row.comp, row.part): float(row.estimate) for row in dec.itertuples()}
    rec = {(row.comp, row.part): bool(row.recommend) for row in dec.itertuples()}
    half = json.loads((OUT / "rater" / "recommended_offsets_C18base.json").read_text())["offsets"]
    full = {"pa_add": est[("PA", "total_add")] if rec[("PA", "total_add")] else 0.0, "fl_mult": 1.0,
            "mt_add": est[("MT", "total_add")] if rec[("MT", "total_add")] else 0.0}
    return dict(half=half, full=full, decision=dec)


# =================================================================================== build
SIGMA_FLOOR = 0.5
OFFSET_SIGMA_FLOOR = 0.2    # a uniform offset: the C17 -> C12 LB efficiency was 0.19


def ev_probe(mu: float, sigma: float, tau: float) -> dict:
    """Normal prior on the true (private) gain g ~ N(mu, sigma^2); the public LB measures g with noise tau.
    Adopt iff the posterior mean > 0. EVSI = E[max(m', 0)] - max(mu, 0), m' ~ N(mu, s^2), s^2 = sigma^4/(sigma^2+tau^2)."""
    from scipy.stats import norm
    s = math.sqrt(sigma ** 4 / (sigma ** 2 + tau ** 2)) if sigma > 0 else 0.0
    if s <= 0:
        v = max(mu, 0.0)
        return dict(s=0.0, value_probe_adopt=v, evsi=0.0, p_better=float(mu > 0))
    v = s * norm.pdf(mu / s) + mu * norm.cdf(mu / s)
    return dict(s=s, value_probe_adopt=float(v), evsi=float(v - max(mu, 0.0)), p_better=float(norm.cdf(mu / sigma)))


def candidate_defs(off_full: dict) -> dict:
    """C20..C24: each ONE group-level change from C18 (desc, test builder, the parts the transfer model scores)."""
    ts = test_state()
    P = ts["P"]
    g = P["group"].astype(str)
    nt = g.isin(NON_TELEMED)
    gate = pd.Series(np.where(nt & (fl_ratio(P) <= 2.25), 0.0, np.nan), index=P.index)
    pa_nt = b4.conf_rule(P["pa_deg"], pa_conf(P), g, ts["pa_cell"]).where(nt)
    lam18 = C18_LAM["fl"]
    return {
        "C20_C18_FLraw_all": dict(
            desc="C18 + FL unshrunk on EVERY group (Siemens, Lumify, legacy FL at lambda 0; Telemed already raw)",
            build=lambda: test_sub_lam({}, {gg: 0.0 for gg in GROUPS}), parts=dict(comps=("fl",), groups=NON_TELEMED)),
        "C21_C18_telemedPAraw": dict(
            desc="C18 + PA unshrunk on the two Telemed groups (clip median of the raw per-frame PA + offset)",
            build=lambda: test_sub_lam({gg: 0.0 for gg in TELEMED}, lam18), parts=dict(comps=("pa",), groups=TELEMED)),
        "C22_C18_fulloffset": dict(
            desc=f"C18 with the full-strength rater offsets (PA +{off_full['pa_add']:.3f} deg, MT +{off_full['mt_add']:.4f} mm; "
                 "half before)", build=lambda: test_sub_lam({}, lam18, off=off_full), parts=None),
        "C23_C18_FLgate225": dict(
            desc="C18 + non-Telemed FL unshrunk where the raw FL/FOV <= 2.25 (EB shrinkage kept above 2.25)",
            build=lambda: test_sub_lam({}, lam18, lam_fl_row=gate), parts=dict(comps=("fl",), groups=NON_TELEMED)),
        "C24_C18_PAconf_nonTelemed": dict(
            desc="C18 + PA confidence rule with the PA-specific confidence (FL-only flag penalties removed) on the "
                 "non-Telemed groups", build=lambda: test_sub_lam({}, lam18, pa_eb_override=pa_nt),
            parts=dict(comps=("pa",), groups=NON_TELEMED)),
    }


DIAG_DEFS = {
    "D_C18_PAconf_all": "C18 + PA-specific confidence on ALL groups (= C21-like on Telemed + C24)",
    "D_C18_PAraw_all": "C18 + PA unshrunk on every group",
    "D_C18_FLraw_lumify": "C18 + FL unshrunk on Lumify only",
    "D_C18_FLlam05_nonTelemed": "C18 + non-Telemed FL at lambda 0.5",
    "D_C18_telemedPAraw_FLrebuilt": "C21 + Telemed FL rebuilt at the raw PA (the self-consistent raw Telemed pipeline)",
}


def diag_build(name: str) -> pd.DataFrame:
    ts = test_state()
    P = ts["P"]
    g = P["group"].astype(str)
    lam18 = C18_LAM["fl"]
    if name == "D_C18_PAconf_all":
        return test_sub_lam({}, lam18, pa_eb_override=b4.conf_rule(P["pa_deg"], pa_conf(P), g, ts["pa_cell"]))
    if name == "D_C18_PAraw_all":
        return test_sub_lam({gg: 0.0 for gg in GROUPS}, lam18)
    if name == "D_C18_FLraw_lumify":
        return test_sub_lam({}, dict(lam18, philips_lumify=0.0))
    if name == "D_C18_FLlam05_nonTelemed":
        return test_sub_lam({}, dict(lam18, **{gg: 0.5 for gg in NON_TELEMED}))
    if name == "D_C18_telemedPAraw_FLrebuilt":
        Q = P.copy()
        fl = b3.nanmean_w([b3._col(P, "fl_straight"), b3._col(P, "fl_central3"), b3.cons_at(P, P["pa_deep_raw"])],
                          [1.0, 1.0, 1.0])
        fl[b3._col(P, "mt_mm").isna()] = np.nan
        tel = g.isin(TELEMED)
        Q.loc[tel, "fl_mm"] = fl[tel]
        return test_sub_lam({gg: 0.0 for gg in TELEMED}, lam18, P=Q)
    raise KeyError(name)


def extra_checks(sub: pd.DataFrame) -> dict:
    L = lay().loc[sub.index]
    clip = L["clip"]
    inside = clip >= 0
    const = all(sub.loc[inside].groupby(clip[inside])[c].nunique().max() == 1 for c in C3)
    at_bound = {c: int(((sub[c] <= bc.BOUNDS[c][0] + 1e-9) | (sub[c] >= bc.BOUNDS[c][1] - 1e-9)).sum()) for c in C3}
    return dict(n=len(sub), nan=int(sub[list(C3)].isna().sum().sum()), clips_constant=bool(const), at_bound=at_bound)


def offset_expected(off_half: dict, off_full: dict) -> dict:
    """Expected gain of full vs half offsets from the rater agent's Laplace table (rater/offsets_*_expected_gain)."""
    G = pd.read_csv(OUT / "rater" / "offsets_C18base_expected_gain.csv")
    out = {}
    for truth in ("truth = total", "truth = style only", "truth = 0"):
        v = 0.0
        for comp, s_mid in (("PA", 1.8), ("MT", 0.8)):
            q = G[(G.comp == comp) & (G.s == s_mid) & (G.truth == truth)].set_index("applied")["gain_score"]
            v += float(q["full total"] - q["half total"])
        out[truth] = v
    return out


def cmd_build(args) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostic").mkdir(exist_ok=True)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    t0 = time.time()
    ts = test_state()
    A = analysis_inputs()
    offs = recompute_offsets(args.B)
    rep: dict = {"offsets_half_recomputed": offs["half"], "offsets_full": offs["full"],
                 "offsets_half_C12": ts["off"],
                 "offsets_identical_to_C12": all(abs(offs["half"].get(k, 0) - ts["off"].get(k, 0)) < 1e-9 for k in ts["off"])}
    print("offsets:", json.dumps({k: rep[k] for k in ("offsets_half_recomputed", "offsets_full", "offsets_identical_to_C12")}))
    c18f = read_sub(C18_FILE)
    c18 = c18_sub()
    refs = {k: read_sub(p) for k, p in REF_FILES.items() if p.exists()}
    defs = candidate_defs(offs["full"])
    k18 = c18f.fl_mm * np.sin(np.radians(c18f.pa_deg)) / c18f.mt_mm
    rep["k_median_by_group_C18"] = {g: round(float(k18[lay().loc[c18f.index].group == g].median()), 3) for g in GROUPS}
    print("C18 coupling k = FL sin(PA) / MT by group (expert ~1.067):", rep["k_median_by_group_C18"])
    oexp = offset_expected(ts["off"], offs["full"])
    rows, subs = [], {}
    for nm, dfn in list(defs.items()) + [(k, None) for k in DIAG_DEFS]:
        sub = dfn["build"]() if dfn else diag_build(nm)
        path = (OUT if dfn else OUT / "diagnostic") / f"{nm}.csv"
        bc.write_sub(sub, path)
        subs[nm] = read_sub(path)
        S = subs[nm]
        dist = official(S, c18f)
        s6, s1 = lb_sd(S, c18f, 0.6), lb_sd(S, c18f, 1.0)
        ch = (S - c18f.reindex(S.index))
        L = lay().loc[S.index]
        moved = {g: {c: round(float(ch.loc[L.group == g, c].abs().mean()), 4) for c in C3} for g in GROUPS}
        r = dict(name=nm, kind="candidate" if dfn else "diagnostic", desc=dfn["desc"] if dfn else DIAG_DEFS[nm],
                 validate=b3.validate(path), **extra_checks(S), dist_C18=dist["S"], pa_n=dist["pa_n"], fl_n=dist["fl_n"],
                 mt_n=dist["mt_n"], pa_bias=dist["pa_bias"], fl_bias=dist["fl_bias"], mt_bias=dist["mt_bias"],
                 sd_pub_rho06=s6["sd_pub"], sd_pub_rho1=s1["sd_pub"], sd_pub_priv_rho1=s1["sd_pub_priv"],
                 dist_C12=official(S, refs["C12"])["S"], dist_C2=official(S, refs["C2"])["S"],
                 rows_changed=int((ch.abs() > 1e-6).any(axis=1).sum()), mean_abs_change_by_group=json.dumps(moved))
        parts = dfn["parts"] if dfn else None
        if dfn and parts is None:     # the offset candidate: rater Laplace table
            preds = {f"pred_{k}": np.nan for k in PROXIES}
            mu = -float(np.mean([oexp["truth = total"], oexp["truth = style only"]]))   # dS (negative = better)
            spread = float(np.std([oexp["truth = total"], oexp["truth = style only"], oexp["truth = 0"]]))
            preds.update(pred_regime=np.nan, pred_prereg=mu, offset_gain_by_truth=json.dumps({k: round(v, 5) for k, v in oexp.items()}))
            r.update(preds)
        elif nm == "D_C18_telemedPAraw_FLrebuilt":     # an FL head change: the shrinkage transfer model does not apply
            r.update({f"pred_{k}": np.nan for k in PROXIES + ("regime", "prereg")})
            mu, spread = np.nan, np.nan
        else:
            pr = predict_parts(A["E"], ts["P"], c18, S, **(parts or dict(comps=("pa", "fl"), groups=GROUPS)))
            r.update({f"pred_{k}": v for k, v in pr.items()})
            mu = pr["prereg"]
            tel = set((parts or {"groups": GROUPS})["groups"]) & set(TELEMED)
            vals = [pr[k] for k in PROXIES if (k != "dyn" or tel)] + [pr["regime"]]
            spread = float(np.std(vals))
        # prior SD of the true gain: the proxy spread, floored at 0.5 x distance (on C18 the pre-registered-style
        # proxies missed by ~1 x distance, on C2 by ~0.15 x distance)
        floor = (OFFSET_SIGMA_FLOOR if (dfn and parts is None) else SIGMA_FLOOR) * dist["S"]
        sigma = max(spread, floor) if np.isfinite(spread) else floor
        k = S.fl_mm * np.sin(np.radians(S.pa_deg)) / S.mt_mm
        r["k_median_by_group"] = json.dumps({g: round(float(k[L.group == g].median()), 3) for g in GROUPS})
        if not np.isfinite(mu):
            rows.append(r)
            print(f"== {nm}: {r['validate']}; dist C18 {r['dist_C18']:.4f}", flush=True)
            continue
        r["prior_gain_mu"], r["prior_gain_sigma"] = -mu, sigma
        r["regime_gain"] = -r["pred_regime"] if np.isfinite(r.get("pred_regime", np.nan)) else np.nan
        ev = ev_probe(-mu, sigma, s1["sd_pub_priv"])
        r.update({f"ev_{k}": v for k, v in ev.items()})
        evr = ev_probe(r["regime_gain"], sigma, s1["sd_pub_priv"]) if np.isfinite(r["regime_gain"]) else dict(value_probe_adopt=np.nan, evsi=np.nan)
        r["ev_regime_value_probe_adopt"], r["ev_regime_evsi"] = evr["value_probe_adopt"], evr["evsi"]
        r["rank_score_evsi_x_gain"] = ev["evsi"] * max(-mu, 0.0)
        # model average of the two priors (pre-registered proxies; post-hoc regime model) for the final ranking
        r["ev_value_avg"] = float(np.nanmean([ev["value_probe_adopt"], evr["value_probe_adopt"]]))
        # pre-registered adoption threshold: adopt iff the public gain LB(C18) - LB(cand) > thr (posterior mean > 0)
        tau = s1["sd_pub_priv"]
        r["adopt_if_public_gain_gt"] = float(mu * tau ** 2 / sigma ** 2) if sigma > 0 else np.nan
        rows.append(r)
        print(f"== {nm}: {r['validate']}; dist C18 {r['dist_C18']:.4f} (PA {r['pa_n']:.4f} FL {r['fl_n']:.4f} MT {r['mt_n']:.4f}); "
              f"SD_pub {r['sd_pub_rho06']:.4f}/{r['sd_pub_rho1']:.4f}; prior gain {-mu:+.4f} +- {sigma:.4f}; "
              f"regime {r['regime_gain']:+.4f}; EVSI {ev['evsi']:.4f}", flush=True)
    TB = pd.DataFrame(rows)
    cand = TB[TB.kind == "candidate"].copy()
    cand["rank_value_probe_adopt"] = cand["ev_value_probe_adopt"].rank(ascending=False).astype(int)
    cand["rank_evsi_x_gain"] = cand["rank_score_evsi_x_gain"].rank(ascending=False, method="first").astype(int)
    cand["rank_evsi"] = cand["ev_evsi"].rank(ascending=False, method="first").astype(int)
    cand["rank_final"] = cand["ev_value_avg"].rank(ascending=False, method="first").astype(int)
    TB = TB.merge(cand[["name", "rank_value_probe_adopt", "rank_evsi_x_gain", "rank_evsi", "rank_final"]], on="name", how="left")
    TB.to_csv(OUT / "candidates_v7.csv", index=False)
    cols = ["name", "dist_C18", "pa_n", "fl_n", "mt_n", "sd_pub_rho06", "sd_pub_rho1", "pred_clean10_2r", "pred_vl3",
            "pred_dyn", "pred_t142", "pred_prereg", "pred_regime", "prior_gain_mu", "prior_gain_sigma", "ev_evsi",
            "ev_value_probe_adopt", "ev_regime_value_probe_adopt", "ev_value_avg", "adopt_if_public_gain_gt",
            "rank_value_probe_adopt", "rank_final"]
    print(TB[cols].round(4).to_string(index=False))
    # pairwise distances among the candidates (for interpreting combinations of LB results)
    names = list(defs)
    PW = pd.DataFrame([[official(subs[a], subs[b])["S"] for b in names] for a in names], index=names, columns=names)
    PW.to_csv(OUT / "pairwise_distances_v7.csv")
    print(PW.round(4).to_string())
    rep["candidates"] = TB.to_dict("records")
    rep["oexp_full_minus_half"] = oexp
    rep["peak_rss_gb"] = rss_gb()
    rep["seconds"] = time.time() - t0
    (OUT / "candidates_v7_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"peak RSS {rep['peak_rss_gb']:.2f} GB, {rep['seconds']:.0f}s")
    return rep


# =================================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("repro")
    sp.add_parser("analyse")
    b = sp.add_parser("build")
    b.add_argument("--B", type=int, default=4000, help="bootstrap replicates for the rater-offset rule")
    a = ap.parse_args()
    t0 = time.time()
    {"repro": cmd_repro, "analyse": cmd_analyse, "build": cmd_build}[a.cmd](a)
    print(f"[{a.cmd}] done in {time.time() - t0:.0f}s, peak RSS {rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
