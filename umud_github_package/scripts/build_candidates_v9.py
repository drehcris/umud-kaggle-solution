#!/usr/bin/env python
"""physbase9: single-cell LB probes on the current best B1_J4 after the label-space breakthrough (2026-10-23).

LB facts this builds on (public, lower is better): C18 0.33526; J1 (C18 + Lumify PA/FL post-hoc transformed to the
native 16:9 geometry, c = 0.844) 0.29452; J2 (C18 + Siemens native, c = 1.125) 0.28005; J3 (Telemed-1200 native) loses
(+0.054); B1 = C18 + J1 + J2 = 0.23931 (exact additive prediction); B1_J4 = B1 + Telemed MT -0.5 mm = 0.23693 (best).
The public score is exactly additive over disjoint row x component sets, so every probe here changes exactly ONE
declared cell (scanner group(s) x component) of B1_J4 and its public delta is read exactly.

Never submits, never commits. Laptop-light: one process at a time, < 0.5 GB RSS, CSV caches only (the physical
re-measurement of the 149 Siemens + Lumify images is done by scripts/aniso_phys_v9.py, a copy of FB2's aniso.py).

Spaces
------
IMG  the original per-image caches (artifacts/candidates_v3/cache): image-space measurement, as in C18.
SIE  the cache with the 91 Siemens rows re-measured on maps resampled x -> x / 1.125 (aniso_phys_v9 measure),
     every other row unchanged; the whole C12/C18 recipe (scanner x muscle centres, confidence PA rule, EB FL,
     regime gate, half rater offset) is then re-run on it, so all Siemens shrinkage is in physical space.
LUM  the same for the 58 Lumify rows (x -> x / 0.84375).
Each space is computed in its own child process (V3_CACHE points at that space's cache), exactly as FB2 did.

Probes (all on base B1_J4; one declared cell each)
--------------------------------------------------
E_LUM   LUM.pa+LUM.fl <- pipeline-exact physical rebuild (= FB2 P_LUM values; centres / EB / confidence in phys)
E_SIE   SIE.pa+SIE.fl <- pipeline-exact physical rebuild (= FB2 P_SIE)          (+ PA-only / FL-only splits)
G_LUMf  LUM.fl <- physical raw FL where physical FL/FOV <= 2.25, physical EB FL above (C23 gate in physical mm)
U_LUMp  LUM.pa <- physical raw PA (confidence shrinkage removed) + rater offset
U_SIEp  SIE.pa <- physical raw PA + rater offset
U_SIEf  SIE.fl <- physical raw FL (EB removed; ens_pp FL constructed at the physical post-shrink PA, as C18 Telemed)
C21b    TEL.pa <- C21 (Telemed PA unshrunk; image space -- J3 showed Telemed labels are in image space)
K25b    TEL.pa <- PA + 0.25 (asin(1.067 MT/FL) - (PA - pa_add)), coupling recomputed with B1_J4's (J4) MT
J4b     TEL.mt <- C18 MT - 0.25 (second point of the pre-registered grid {-0.25, -0.5, -0.8})
J5b1    NONTEL.mt <- C18 MT - 0.3 (the pre-registered J5, never scored, moved onto B1_J4)
Diagnostics only (not proposed for slots): U_LUMf (Lumify FL fully unshrunk), Cpa_SIE / Cpa_LUM (PA-specific
confidence, the C24 rule, in physical space).

Predictions
-----------
* shrinkage-type moves: the v7 benchmark-efficiency transfer (bench_efficiency.csv, current maps; benchmarks are
  square-pixel images, so they apply to physical-space measurements) evaluated on the PHYSICAL per-image frame
  (FL/FOV bins and PA flag kinds from the physical measurement), pre-registered and regime-matched variants;
* MT moves: an ABC posterior of (Telemed bias, non-Telemed bias, noise scale) from two LB facts on the same public
  subset: the DLTrack c129/c130 pair (+0.4 mm everywhere -> +0.00887) and B1 -> B1_J4 (-0.5 mm Telemed -> -0.00238);
* coupling move (K25b): the critic7 benchmark blend table scaled by the move size.

Subcommands:  repro | caches | state --space S (child) | predict --space S (child) | mtpost | build | all
Run with PYTHONPATH=<project>/src and the project venv.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# research-era session files (judge / critic / FB2 outputs), archived under artifacts/legacy_research; research
# script only -- the final submissions are rebuilt by src/umud/run.py (configs/final_{A,B}.yaml), not by this file
SCR = Path(os.environ.get("UMUD_RESEARCH_ROOT", ROOT / "artifacts/legacy_research"))
OUT = ROOT / "artifacts/candidates_v9"
WORK = OUT / "work"
PHYS = OUT / "phys"
PY = ROOT / ".venv/bin/python"
TEST_DIR = ROOT / "data/umud-challenge-muscle-architecture-in-ultrasound-data/test_images_v2/test_set_v2"
V3_CACHE_ORIG = ROOT / "artifacts/candidates_v3/cache"
SUBMIT8 = ROOT / "artifacts/candidates_v8/submit_2026-10-23b"
B1J4_FILE = SUBMIT8 / "B1_J4.csv"
B1_FILE = SUBMIT8 / "B1_C18_J1_J2.csv"
C18_FILE = ROOT / "artifacts/candidates_v6/C18_C12_telemedFLraw.csv"
J1_FILE = SCR / "v8/judge/J1_C18_LUMphys084.csv"
J2_FILE = SCR / "v8/judge/J2_C18_SIEphys1125.csv"
J4_FILE = SCR / "v8/judge/J4_C18_TELmt_m05.csv"
J5_FILE = SCR / "v8/judge/J5_C18_NONTELmt_m03.csv"
C21_FILE = ROOT / "artifacts/candidates_v7/C21_C18_telemedPAraw.csv"
K25_FILE = SCR / "critic7/proposals/K25_C18_telemedPAcpl025.csv"
P_FB2 = {"LUM": SCR / "v8/aniso/P_LUM.csv", "SIE": SCR / "v8/aniso/P_SIE.csv"}
EFF_CSV = ROOT / "artifacts/candidates_v7/bench_efficiency.csv"
CPL_BENCH = SCR / "critic7/bench_cpl_blend.csv"
COMBO_PY = SCR / "v8/ops/combo.py"

LB = {"C18": 0.33526, "J1_C18_LUMphys084": 0.29452, "J2_C18_SIEphys1125": 0.28005, "COMBO_J1J2J3": 0.29344,
      "B1_C18_J1_J2": 0.23931, "B1_J4": 0.23693, "J4_C18_TELmt_m05": 0.33526 - 0.00238}
C3 = ("pa_deg", "fl_mm", "mt_mm")
TAU = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
BOUNDS = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
T12, T10, LUM, SIE = "telemed_artus_1200x800", "telemed_artus_1088x644", "philips_lumify", "siemens_juniper"
L853, L512 = "legacy_853", "legacy_512"
TELEMED = (T12, T10)
NONTEL = (SIE, LUM, L853, L512)
SPACE_GROUP = {"IMG": None, "SIE": SIE, "LUM": LUM}
SHORT = {SIE: "SIE", T12: "T12", T10: "T10", LUM: "LUM", L853: "L853", L512: "L512"}
K_EXPERT = 1.067
N_TEST, N_PUB = 309, 102


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def combo():
    spec = importlib.util.spec_from_file_location("ops_combo", COMBO_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_sub(p) -> pd.DataFrame:
    return pd.read_csv(p).set_index("image_id")[list(C3)].astype(float)


def layout() -> pd.DataFrame:
    return combo().layout()


def fin(v: pd.Series, comp: str) -> pd.Series:
    lo, hi = BOUNDS[comp]
    return v.astype(float).clip(lo + 1e-3, hi - 1e-3).round(4)


def dist(a: pd.DataFrame, b: pd.DataFrame, rows=None) -> dict:
    """Official-metric distance contributions (sum over rows / 309 / tau / 3)."""
    b = b.reindex(a.index)
    m = np.ones(len(a), bool) if rows is None else a.index.isin(rows)
    d = {c: float((a[c] - b[c]).abs()[m].sum() / N_TEST / TAU[c] / 3) for c in C3}
    d["S"] = sum(d.values())
    return d


def lb_sd(a: pd.DataFrame, b: pd.DataFrame, rho: float = 0.6, clip_aware: bool = True) -> dict:
    """Public sampling SD of S(a) - S(b) (b4.lb_sd), optionally with Telemed clips as units (frames of a clip
    share their error, so a clip's frames count as one draw; this is the conservative variant)."""
    b = b.reindex(a.index)
    per = np.mean(np.stack([np.abs(a[c] - b[c]).to_numpy(float) / TAU[c] for c in C3], 1), 1)
    if clip_aware:
        lay = layout().loc[a.index]
        key = np.where(lay["clip"] >= 0, "c" + lay["clip"].astype(str), a.index.astype(str))
        s = pd.Series(per, index=a.index).groupby(key).agg(["sum", "size"])
        S2 = rho * float((s["sum"] ** 2).sum() / N_TEST)
    else:
        S2 = rho * float((per ** 2).mean())
    f = N_PUB / N_TEST
    sd_pub = math.sqrt((1 - f) * S2 / N_PUB)
    return dict(sd_pub=sd_pub, sd_pub_priv=sd_pub * N_TEST / (N_TEST - N_PUB))


def validate(path) -> str:
    r = subprocess.run([str(PY), str(ROOT / "scripts/validate_submission.py"), str(path), "--test-dir", str(TEST_DIR)],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    return (r.stdout + r.stderr).strip().splitlines()[-1]


# ====================================================================================== repro
def cmd_repro(args) -> dict:
    """B1_J4 and B1 rebuilt from their components with combo.py (C18 + J1 + J2 [+ J4]); byte and value identity."""
    cb = combo()
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    for name, adds, target in [("B1_C18_J1_J2", [J1_FILE, J2_FILE], B1_FILE),
                               ("B1_J4", [J1_FILE, J2_FILE, J4_FILE], B1J4_FILE)]:
        out = OUT / f"REPRO_{name}.csv"
        r = cb.build(C18_FILE, [(f, C18_FILE) for f in adds], out, scores=LB, name=name, quiet=True)
        a, b = read_sub(out), read_sub(target)
        rep[name] = dict(file=str(out), cells=r["cells"], bytes_identical=out.read_bytes() == target.read_bytes(),
                         max_abs_vs_submitted=float((a - b.loc[a.index]).abs().max().max()),
                         predicted_public=r["prediction"]["pred"], audit=r["audit"], validate=r["validate_msg"],
                         clips_constant=r["clips_constant"])
        print(name, json.dumps(rep[name], default=str))
    (WORK).mkdir(parents=True, exist_ok=True)
    (WORK / "repro.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


# ====================================================================================== caches
def cmd_caches(args) -> None:
    """Physical caches from aniso_phys_v9's rows (re-measure only if the rows are missing)."""
    rows = PHYS / "rows_phys/test__v2_rows.csv"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    if not (PHYS / "factors_test.csv").exists():
        subprocess.run([str(PY), str(ROOT / "scripts/aniso_phys_v9.py"), "factors"], check=True, env=env)
    have = set(pd.read_csv(rows).image_id) if rows.exists() else set()
    lay = layout()
    need = set(lay.index[lay.group.isin([SIE, LUM])])
    if not need <= have:
        subprocess.run([str(PY), str(ROOT / "scripts/aniso_phys_v9.py"), "measure", "--mode", "phys", "--only", "SIE",
                        "LUM"], check=True, env=env)
    sys.path.insert(0, str(ROOT / "scripts"))
    import aniso_phys_v9 as aph
    for v in ("SIE", "LUM"):
        d = aph.assemble_cache(v)
        print("cache", v, d)
    # identity of the re-measured rows with FB2's (scratchpad v8/aniso/rows_phys)
    fb2 = SCR / "v8/aniso/rows_phys/test__v2_rows.csv"
    if fb2.exists():
        a = pd.read_csv(rows).set_index("image_id")
        b = pd.read_csv(fb2).set_index("image_id").loc[a.index]
        num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and not pd.api.types.is_bool_dtype(a[c])
               and c in b and pd.api.types.is_numeric_dtype(b[c])]
        mx = float(np.nanmax((a[num].astype(float) - b[num].astype(float)).abs().to_numpy()))
        oth = [c for c in a.columns if c not in num and c in b]
        neq = {c: int((a[c].where(a[c].notna(), "").astype(str) != b[c].where(b[c].notna(), "").astype(str)).sum())
               for c in oth}
        res = dict(rows=len(a), numeric_cols=len(num), max_abs=mx, nonnumeric_mismatch=sum(neq.values()))
        print("re-measurement vs FB2 rows:", res)
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "remeasure_identity.json").write_text(json.dumps(res, indent=1))


def cache_dir(space: str) -> Path:
    return V3_CACHE_ORIG if space == "IMG" else PHYS / f"v3cache_{space}"


def run_child(cmd: str, space: str, extra: list | None = None) -> None:
    empty = PHYS / "v4cache_empty"
    empty.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, V3_CACHE=str(cache_dir(space)), V4_CACHE=str(empty), PYTHONPATH=str(ROOT / "src"),
               PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    r = subprocess.run([str(PY), __file__, cmd, "--space", space] + (extra or []), env=env, capture_output=True,
                       text=True)
    print(r.stdout.strip()[-1500:])
    if r.returncode:
        print(r.stderr[-4000:])
        raise SystemExit(f"child {cmd} {space} failed")
    print(f"  ({cmd} {space}: {time.time() - t0:.0f}s)")


# ====================================================================================== state (child)
def _b7():
    for p in (str(ROOT / "scripts"), str(ROOT / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import build_candidates_v7 as b7
    assert str(b7.b3.CACHE) == os.environ["V3_CACHE"], (b7.b3.CACHE, os.environ["V3_CACHE"])
    return b7


def cmd_state(args) -> None:
    """Child: the C18 recipe on this space's cache; dump the per-image frame and the shrinkage variants."""
    b7 = _b7()
    b4 = b7.b4
    sh = b7.sh
    ts = b7.test_state()
    P = ts["P"]
    g = P["group"].astype(str)
    lam18 = b7.C18_LAM["fl"]
    G = SPACE_GROUP[args.space]
    groups = [G] if G else list(b7.GROUPS)
    V = {"base": b7.c18_sub()}
    ratio = b7.fl_ratio(P)
    for gg in groups:
        V[f"Upa|{gg}"] = b7.test_sub_lam({gg: 0.0}, lam18)
        V[f"Ufl|{gg}"] = b7.test_sub_lam({}, dict(lam18, **{gg: 0.0}))
        gate = pd.Series(np.where((g == gg) & (ratio <= 2.25), 0.0, np.nan), index=P.index)
        V[f"Gfl225|{gg}"] = b7.test_sub_lam({}, lam18, lam_fl_row=gate)
        pa_g = b4.conf_rule(P["pa_deg"], b7.pa_conf(P), g, ts["pa_cell"]).where(g == gg)
        V[f"Cpa|{gg}"] = b7.test_sub_lam({}, lam18, pa_eb_override=pa_g)
    if G is None:
        V["Upa|TEL"] = b7.test_sub_lam({gg: 0.0 for gg in TELEMED}, lam18)
    raw_all = b7.test_sub_lam({gg: 0.0 for gg in b7.GROUPS}, {gg: 0.0 for gg in b7.GROUPS})
    V["raw_all"] = raw_all
    rows = []
    for k, s in V.items():
        s = s.reindex(P.index)
        rows.append(pd.DataFrame({"variant": k, "image_id": s.index, **{c: s[c].to_numpy(float) for c in C3}}))
    WORK.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(WORK / f"var_{args.space}.csv", index=False)
    fov = sh._fov_mm(P)
    F = pd.DataFrame({
        "group": g, "clip": P["clip"], "pa_raw": pd.to_numeric(P["pa_deg"], errors="coerce"),
        "pa_eb": P["pa_eb"], "pa_deep_c": P["pa_deep_c"], "fl_raw_img": pd.to_numeric(P["fl_mm"], errors="coerce"),
        "fl_raw_unit": b7.fl_unit_raw(P), "fl_eb": P["fl_eb"], "mt": P["mt_mm"] if "mt_mm" in P else np.nan,
        "w_pa": P["w_pa"], "w_fl": P["w_fl"], "mu_fl": P.get("mu_fl"), "tau_fl": P.get("tau_fl"),
        "sigma_fl": P.get("sigma_fl"), "cell": P.get("cell"), "pa_cell": ts["pa_cell"].reindex(P.index),
        "confidence": P["confidence"], "pa_conf": b7.pa_conf(P), "flags": P["flags"], "fov_mm": fov,
        "fl_fov": ratio, "pa_kind": b7.pa_kind(P), "fl_bin": b7.fl_bin(P),
        "pa_centre": b4.centres(P["pa_deg"], g, ts["pa_cell"]),
        "aniso_c": P["aniso_c"] if "aniso_c" in P else 1.0}, index=P.index)
    F.index.name = "image_id"
    F.to_csv(WORK / f"P_{args.space}.csv")
    (WORK / f"state_{args.space}.json").write_text(json.dumps(dict(offsets=ts["off"], cache=os.environ["V3_CACHE"],
                                                                     n=len(P), rss_gb=rss_gb()), indent=1,
                                                                default=float))
    print(f"state {args.space}: {len(V)} variants, rss {rss_gb():.2f} GB")


# ====================================================================================== predict (child)
def cmd_predict(args) -> None:
    """Child: v7 benchmark-efficiency transfer (pre-registered + regime) for the probes of this space, evaluated on
    this space's per-image frame (bins from the physical measurement for SIE / LUM)."""
    b7 = _b7()
    P = b7.test_state()["P"]
    E = pd.read_csv(EFF_CSV)
    E = E[E.tag == "current"]
    specs = json.loads((WORK / "probe_specs.json").read_text())
    base = read_sub(B1J4_FILE).loc[P.index]
    out = {}
    for s in specs:
        if s["space"] != args.space or not s.get("transfer", True):
            continue
        b = read_sub(s["staged"]).loc[P.index]
        r = b7.predict_parts(E, P, base, b, comps=tuple(s["comps"]), groups=tuple(s["groups"]))
        out[s["name"]] = {k: float(v) for k, v in r.items()}
        print(s["name"], {k: round(v, 5) for k, v in out[s["name"]].items()})
    (WORK / f"pred_{args.space}.json").write_text(json.dumps(out, indent=1))


# ====================================================================================== mtpost
def cmd_mtpost(args) -> dict:
    """ABC posterior of the MT level from two LB facts measured on the same public subset.
    truth_MT = C18_MT - d_g + eps; eps ~ Laplace(s) shared by the frames of a Telemed clip (+ N(0, 0.15) per frame),
    Laplace(s_non) per non-Telemed image. Facts: c129 = c130 + 0.4 mm on every row scored +0.00887 vs c130
    (judge mt_check); B1_J4 - B1 (Telemed MT -0.5) = -0.00238. Predictive deltas (vs B1_J4) on the same subsets."""
    sys.path.insert(0, str(SCR / "v8/lb-inference"))
    import sensors as se
    g, P, _ = se.all_sensors(include_ambros=False)
    grp = g["group"].values
    tel = np.array(["telemed" in x for x in grp])
    clip = g["clip"].to_numpy()
    c18 = P["C18"][:, 2]
    c130 = P["c130"][:, 2]
    assert np.allclose(c18, read_sub(C18_FILE).loc[g.index, "mt_mm"].to_numpy())
    ucl, inv = np.unique(np.where(tel, clip, -1 - np.arange(len(grp))), return_inverse=True)
    rng = np.random.default_rng(9)
    obs1, obs2 = 0.00887, -0.00238
    NS = args.nsim
    recs = []
    for s in (0.3, 0.45, 0.6, 0.8):
        for snr in (1.0, 1.5):
            s_non = s * snr
            for dt in np.round(np.arange(-0.1, 0.91, 0.05), 3):
                for dn in np.round(np.arange(-0.4, 1.21, 0.1), 3):
                    ec = rng.laplace(0, 1, (NS, len(ucl)))[:, inv]
                    eps = np.where(tel, ec * s + rng.normal(0, 0.15, (NS, len(grp))), ec * s_non)
                    truth = c18[None, :] - np.where(tel, dt, dn)[None, :] + eps
                    pub = np.argsort(rng.random((NS, len(grp))), 1)[:, :N_PUB]
                    take = lambda a: np.take_along_axis(a, pub, 1)  # noqa: E731
                    t = take(truth)
                    x = np.broadcast_to(c18, truth.shape)
                    xp = take(x)
                    e130 = take(np.broadcast_to(c130, truth.shape)) - t
                    tp = take(np.broadcast_to(tel, truth.shape))
                    m1 = (np.abs(e130 + 0.4) - np.abs(e130)).mean(1) / 9
                    e = xp - t

                    def dlt(new_minus_old_shift_rows, a_shift, b_shift):
                        sel = new_minus_old_shift_rows
                        v = (np.abs(e + b_shift) - np.abs(e + a_shift)) * sel
                        return v.sum(1) / N_PUB / 9

                    j4 = dlt(tp, 0.0, -0.5)
                    rec = dict(s=s, s_non=s_non, d_tel=dt, d_non=dn, m1=m1, j4=j4,
                               J4b=dlt(tp, -0.5, -0.25), J4c=dlt(tp, -0.5, -0.8),
                               J5=dlt(~tp, 0.0, -0.3), J5b=dlt(~tp, 0.0, -0.5), J5c=dlt(~tp, 0.0, -0.15))
                    recs.append(rec)
    # ABC weights (Gaussian kernels on the two facts)
    h1, h2 = args.h1, args.h2
    rows = []
    for r in recs:
        w = np.exp(-0.5 * ((r["m1"] - obs1) / h1) ** 2 - 0.5 * ((r["j4"] - obs2) / h2) ** 2)
        rows.append(dict(s=r["s"], s_non=r["s_non"], d_tel=r["d_tel"], d_non=r["d_non"], w=float(w.sum()),
                         **{k: float((w * r[k]).sum()) for k in ("J4b", "J4c", "J5", "J5b", "J5c")},
                         **{k + "_2": float((w * r[k] ** 2).sum()) for k in ("J4b", "J4c", "J5", "J5b", "J5c")},
                         m1_mean=float(r["m1"].mean()), j4_mean=float(r["j4"].mean())))
    T = pd.DataFrame(rows)
    W = T.w.sum()
    post = {}
    for k in ("J4b", "J4c", "J5", "J5b", "J5c"):
        m = T[k].sum() / W
        sd = math.sqrt(max(T[k + "_2"].sum() / W - m ** 2, 0))
        post[k] = dict(mean=m, sd=sd)
    marg = {}
    for v in ("d_tel", "d_non", "s", "s_non"):
        q = T.groupby(v).w.sum() / W
        marg[v] = {str(k): round(float(x), 4) for k, x in q.items()}
        post[f"{v}_mean"] = float((q * q.index.astype(float)).sum())
    ess = float(W ** 2 / (T.w ** 2).sum()) if W > 0 else 0.0
    # prior-predictive check: the judge's original J4 prediction region
    rep = dict(obs=dict(c129_c130=obs1, J4=obs2), kernel=dict(h1=h1, h2=h2), nsim_per_point=NS, grid_points=len(T),
               ess_points=ess, posterior=post, marginals=marg)
    T.drop(columns=[c for c in T.columns if c.endswith("_2")]).to_csv(WORK / "mtpost_grid.csv", index=False)
    (WORK / "mtpost.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "marginals"}, indent=1, default=float))
    print("marginal d_tel:", marg["d_tel"])
    print("marginal d_non:", marg["d_non"])
    return rep


# ====================================================================================== build
def var_frame(space: str, variant: str) -> pd.DataFrame:
    V = pd.read_csv(WORK / f"var_{space}.csv")
    return V[V.variant == variant].set_index("image_id")[list(C3)]


def kstats(S: pd.DataFrame, rows) -> dict:
    k = (S.loc[rows, "fl_mm"] * np.sin(np.radians(S.loc[rows, "pa_deg"])) / S.loc[rows, "mt_mm"])
    return dict(k_med=float(k.median()), k_mean=float(k.mean()))


def probe_defs(base: pd.DataFrame, lay: pd.DataFrame, off: dict) -> list[dict]:
    """Every probe: name, declared cells, the cell values (full frame), and its pre-registration text."""
    rows = {gg: lay.index[lay.group == gg] for gg in lay.group.unique()}
    tel = lay.index[lay.group.isin(TELEMED)]
    nontel = lay.index[lay.group.isin(NONTEL)]
    C18 = read_sub(C18_FILE).loc[base.index]
    defs = []

    def put(name, cells, values: dict, space, groups, comps, kind, desc, transfer=True, slot_candidate=True):
        F = base.copy()
        for comp, (idx, v) in values.items():
            F.loc[idx, comp] = fin(v.loc[idx], comp)
        defs.append(dict(name=name, cells=cells, frame=F, space=space, groups=groups, comps=comps, kind=kind,
                         desc=desc, transfer=transfer, slot_candidate=slot_candidate))

    for sp, gg in (("LUM", LUM), ("SIE", SIE)):
        s = SHORT[gg]
        Pv = var_frame(sp, "base")
        put(f"E_{s}", [f"{s}.pa", f"{s}.fl"], {"pa_deg": (rows[gg], Pv.pa_deg), "fl_mm": (rows[gg], Pv.fl_mm)}, sp, [gg],
            ["pa", "fl"], "exact",
            f"{s} PA+FL <- pipeline-exact physical rebuild (maps resampled x->x/c before measurement; centres, EB FL, "
            f"confidence PA and rater offset recomputed in physical space) = FB2 P_{s} values; replaces the post-hoc J "
            f"transform of the image-space final values")
        put(f"E_{s}p", [f"{s}.pa"], {"pa_deg": (rows[gg], Pv.pa_deg)}, sp, [gg], ["pa"], "exact",
            f"{s} PA only <- physical rebuild (split of E_{s})", slot_candidate=False)
        put(f"E_{s}f", [f"{s}.fl"], {"fl_mm": (rows[gg], Pv.fl_mm)}, sp, [gg], ["fl"], "exact",
            f"{s} FL only <- physical rebuild (split of E_{s})", slot_candidate=False)
        Up = var_frame(sp, f"Upa|{gg}")
        put(f"U_{s}p", [f"{s}.pa"], {"pa_deg": (rows[gg], Up.pa_deg)}, sp, [gg], ["pa"], "unshrink",
            f"{s} PA <- physical raw PA (fusF deep head on the resampled maps, confidence shrinkage toward the "
            f"physical scanner x muscle centre removed) + half rater offset; FL stays at B1_J4 (J)")
        Uf = var_frame(sp, f"Ufl|{gg}")
        put(f"U_{s}f", [f"{s}.fl"], {"fl_mm": (rows[gg], Uf.fl_mm)}, sp, [gg], ["fl"], "unshrink",
            f"{s} FL <- physical raw FL (ens_pp at the physical post-shrink PA; EB shrinkage removed, as C18 did for "
            f"Telemed FL); PA stays at B1_J4 (J)", slot_candidate=(gg == SIE))
        Gf = var_frame(sp, f"Gfl225|{gg}")
        put(f"G_{s}f", [f"{s}.fl"], {"fl_mm": (rows[gg], Gf.fl_mm)}, sp, [gg], ["fl"], "gate",
            f"{s} FL <- physical raw FL where physical FL/FOV <= 2.25, physical EB FL above (the C23 regime gate "
            f"measured in physical mm)")
        Cp = var_frame(sp, f"Cpa|{gg}")
        put(f"Cpa_{s}", [f"{s}.pa"], {"pa_deg": (rows[gg], Cp.pa_deg)}, sp, [gg], ["pa"], "unshrink",
            f"{s} PA <- physical confidence rule with the PA-specific confidence (C24 rule; FL-only flag penalties "
            f"removed)", slot_candidate=False)
    # Telemed (image space)
    c21 = read_sub(C21_FILE).loc[base.index]
    put("C21b", ["T12.pa", "T10.pa"], {"pa_deg": (tel, c21.pa_deg)}, "IMG", list(TELEMED), ["pa"], "unshrink",
        "Telemed PA <- C21 (per-clip raw PA + offset; confidence shrinkage removed; image space)")
    pa_add = off["pa_add"]
    pre = base.pa_deg - pa_add
    cpl = pd.Series(np.degrees(np.arcsin(np.clip(K_EXPERT * base.mt_mm / base.fl_mm, 0, 1))), index=base.index)
    k25b = base.pa_deg + 0.25 * (cpl - pre)
    put("K25b", ["T12.pa", "T10.pa"], {"pa_deg": (tel, k25b)}, "IMG", list(TELEMED), ["pa"], "coupling",
        "Telemed PA <- PA + 0.25 * (asin(1.067 MT/FL) - (PA - 0.3286)), MT = B1_J4's (C18 - 0.5), FL = C18 raw clip FL",
        transfer=False)
    put("J4b", ["T12.mt", "T10.mt"], {"mt_mm": (tel, C18.mt_mm - 0.25)}, "IMG", list(TELEMED), ["mt"], "mt",
        "Telemed MT <- C18 - 0.25 mm (second grid point; B1_J4 has C18 - 0.5)", transfer=False)
    j5 = read_sub(J5_FILE).loc[base.index]
    NT_CELLS = ["SIE.mt", "LUM.mt", "L853.mt", "L512.mt"]
    put("J5b1", NT_CELLS, {"mt_mm": (nontel, j5.mt_mm)}, "IMG", list(NONTEL), ["mt"],
        "mt", "non-Telemed MT <- C18 - 0.3 mm (pre-registered J5, moved onto B1_J4)", transfer=False)
    # ---- second-stage alternates (pre-built; scheduled only by the pre-registered rules)
    put("J5b2", NT_CELLS, {"mt_mm": (nontel, C18.mt_mm - 0.5)}, "IMG", list(NONTEL), ["mt"], "mt",
        "non-Telemed MT <- C18 - 0.5 mm (grid point 2 of {-0.15, -0.3, -0.5}; only if J5b1 wins)", transfer=False,
        slot_candidate=False)
    put("J5c", NT_CELLS, {"mt_mm": (nontel, C18.mt_mm - 0.15)}, "IMG", list(NONTEL), ["mt"], "mt",
        "non-Telemed MT <- C18 - 0.15 mm (grid point; only if J5b1 wins and J5b2 loses vs J5b1)", transfer=False,
        slot_candidate=False)
    Plum = pd.read_csv(WORK / "P_LUM.csv").set_index("image_id").reindex(base.index)   # physical LUM frame
    Ul = var_frame("LUM", f"Ufl|{LUM}")
    a_rows = rows[LUM][(Plum.loc[rows[LUM], "fl_fov"] <= 2.25).to_numpy()]
    put("GJ_LUMf", ["LUM.fl"], {"fl_mm": (a_rows, Ul.fl_mm)}, "LUM", [LUM], ["fl"], "gate",
        "Lumify FL <- physical raw FL where physical FL/FOV <= 2.25; B1_J4 (J) kept above (the gate without the "
        "physical-EB change on the long rows; fallback if E_LUM loses)", slot_candidate=False)
    put("K26b", ["T12.pa", "T10.pa"], {"pa_deg": (tel, base.pa_deg + 0.5 * (cpl - pre))}, "IMG", list(TELEMED), ["pa"],
        "coupling", "Telemed PA <- 50% toward the coupling PA (J4 MT); only if K25b gains > 0.003", transfer=False,
        slot_candidate=False)
    pre21 = c21.pa_deg - pa_add
    put("K27b", ["T12.pa", "T10.pa"], {"pa_deg": (tel, c21.pa_deg + 0.25 * (cpl - pre21))}, "IMG", list(TELEMED), ["pa"],
        "coupling", "Telemed PA <- C21 raw PA + 25% toward the coupling PA (J4 MT); only if both C21b and K25b win",
        transfer=False, slot_candidate=False)
    return defs


def cmd_build(args) -> dict:
    cb = combo()
    lay = layout()
    base = read_sub(B1J4_FILE).loc[lay.index]
    C18 = read_sub(C18_FILE).loc[lay.index]
    st = {sp: json.loads((WORK / f"state_{sp}.json").read_text()) for sp in ("IMG", "SIE", "LUM")}
    off = st["IMG"]["offsets"]
    assert all(abs(st[sp]["offsets"]["pa_add"] - off["pa_add"]) < 1e-12 for sp in st)
    checks = {}
    # ---- state identities: IMG base = C18; physical bases = FB2's P files on their group
    for sp in ("IMG", "SIE", "LUM"):
        Bv = var_frame(sp, "base").loc[lay.index]
        if sp == "IMG":
            checks["IMG_base_vs_C18_maxabs"] = float((Bv - C18).abs().max().max())
        else:
            gg = SPACE_GROUP[sp]
            idx = lay.index[lay.group == gg]
            Pf = read_sub(P_FB2[sp]).loc[lay.index]
            for comp in ("pa_deg", "fl_mm"):
                checks[f"{sp}_base_vs_FB2_P_{sp}_{comp}_maxabs"] = float(
                    (fin(Bv.loc[idx, comp], comp) - Pf.loc[idx, comp]).abs().max())
            oth = lay.index[lay.group != gg]
            checks[f"{sp}_base_other_groups_vs_C18_maxabs"] = float((Bv.loc[oth] - C18.loc[oth]).abs().max().max())
    c21v = var_frame("IMG", "Upa|TEL").loc[lay.index]
    tel = lay.index[lay.group.isin(TELEMED)]
    checks["C21_regen_vs_file_maxabs"] = float((fin(c21v.loc[tel, "pa_deg"], "pa_deg") - read_sub(C21_FILE).loc[tel, "pa_deg"]).abs().max())
    j5 = read_sub(J5_FILE).loc[lay.index]
    nt = lay.index[lay.group.isin(NONTEL)]
    checks["J5_is_C18_minus_0p3_maxabs"] = float(((C18.loc[nt, "mt_mm"] - 0.3) - j5.loc[nt, "mt_mm"]).abs().max())
    print("identity checks:", json.dumps(checks))
    # ---- probes
    defs = probe_defs(base, lay, off)
    staged = WORK / "staged"
    staged.mkdir(parents=True, exist_ok=True)
    specs, reps = [], {}
    for d in defs:
        sp = staged / f"{d['name']}.csv"
        cb.write_sub(d["frame"], sp)
        specs.append(dict(name=d["name"], staged=str(sp), space=d["space"], groups=d["groups"], comps=d["comps"],
                          transfer=d["transfer"]))
    (WORK / "probe_specs.json").write_text(json.dumps(specs, indent=1))
    for d in defs:
        out = OUT / ("" if d["slot_candidate"] else "alternates") / f"{d['name']}_B1J4.csv"
        r = cb.build(B1J4_FILE, [(staged / f"{d['name']}.csv", B1J4_FILE)], out, scores=LB, name=d["name"], quiet=True)
        X = read_sub(out).loc[lay.index]
        diffmask = (X - base).abs() > 0
        changed = sorted({f"{SHORT[g_]}.{c.split('_')[0]}" for c in C3 for g_ in lay.group[diffmask[c]].unique()})
        exact_iso = set(changed) <= set(d["cells"])
        rows_in = lay.index[lay.group.isin(d["groups"])]
        dd = dist(X, base)
        sd = lb_sd(X, base)
        mean_ch = {f"{SHORT[g_]}.{c.split('_')[0]}": round(float((X.loc[lay.index[lay.group == g_], c] -
                                                                   base.loc[lay.index[lay.group == g_], c]).mean()), 4)
                   for c in C3 for g_ in d["groups"] if f"{SHORT[g_]}.{c.split('_')[0]}" in d["cells"]}
        mean_abs = {f"{SHORT[g_]}.{c.split('_')[0]}": round(float((X.loc[lay.index[lay.group == g_], c] -
                                                                    base.loc[lay.index[lay.group == g_], c]).abs().mean()), 4)
                    for c in C3 for g_ in d["groups"] if f"{SHORT[g_]}.{c.split('_')[0]}" in d["cells"]}
        k_before = {SHORT[g_]: kstats(base, lay.index[lay.group == g_]) for g_ in d["groups"]}
        k_after = {SHORT[g_]: kstats(X, lay.index[lay.group == g_]) for g_ in d["groups"]}
        reps[d["name"]] = dict(file=str(out), cells=d["cells"], kind=d["kind"], desc=d["desc"],
                               slot_candidate=d["slot_candidate"], combo_cells=r["cells"],
                               cells_changed_exact=changed, isolated=bool(exact_iso and not r["outside_declared"]),
                               validate=r["validate_msg"], clips_constant=r["clips_constant"],
                               untouched_max=r["max_abs_untouched_vs_base"], in_bounds=r["in_bounds"],
                               rows_changed=int(diffmask.any(axis=1).sum()), dist=dd, sd_pub=sd["sd_pub"],
                               sd_pub_priv=sd["sd_pub_priv"], mean_change=mean_ch, mean_abs_change=mean_abs,
                               k_before=k_before, k_after=k_after, sha16=r["sha16"], ok=r["ok"])
        print(f"{d['name']:8s} cells {','.join(changed):34s} dist {dd['S']:.5f} sdpp {sd['sd_pub_priv']:.4f} "
              f"iso {reps[d['name']]['isolated']} val {r['validate']} clips {r['clips_constant']} mean {mean_ch}")
    (WORK / "probes_build.json").write_text(json.dumps(dict(checks=checks, probes=reps), indent=1, default=float))
    return dict(checks=checks, probes=reps)


# ====================================================================================== diagnostics
def cmd_diag(args) -> dict:
    """What shrinkage does per cell (image vs physical), centres, FL/FOV, coupling k."""
    lay = layout()
    Pimg = pd.read_csv(WORK / "P_IMG.csv").set_index("image_id")
    Cimg = pd.read_csv(V3_CACHE_ORIG / "test__v2.csv").set_index("image_id")
    base = read_sub(B1J4_FILE).loc[lay.index]
    C18 = read_sub(C18_FILE).loc[lay.index]
    shrink_rows, centre_rows, ratio_rows = [], [], []
    for sp, gg, c in (("SIE", SIE, 1.125), ("LUM", LUM, 0.84375)):
        Pp = pd.read_csv(WORK / f"P_{sp}.csv").set_index("image_id")
        idx = lay.index[lay.group == gg]
        a, b = Pimg.loc[idx], Pp.loc[idx]
        delta = (Cimg.loc[idx, "pa_fusion"] - Cimg.loc[idx, "d_pa_horizontal"]).fillna(0.0)
        phi = np.radians(a.pa_deep_c - delta)
        fac = np.sqrt(np.cos(phi) ** 2 / c ** 2 + np.sin(phi) ** 2)
        for space, Q, f in (("image", a, 1.0), ("image->J (x transform)", a, fac), ("physical", b, 1.0)):
            for reg, m in [("all", np.ones(len(idx), bool)), ("FL/FOV<=2.25", (b.fl_fov <= 2.25).to_numpy()),
                           ("FL/FOV>2.25", (b.fl_fov > 2.25).to_numpy())]:
                if space.startswith("image->"):
                    pa_shift = np.nan
                else:
                    pa_shift = float((Q.pa_eb - Q.pa_raw).abs()[m].mean())
                fl_shift = (Q.fl_eb - Q.fl_raw_unit) * f
                shrink_rows.append(dict(group=SHORT[gg], space=space, regime=reg, n=int(m.sum()),
                                        pa_w_mean=float(Q.w_pa[m].mean()), pa_mean_abs_shift_deg=pa_shift,
                                        pa_mean_signed_shift_deg=float((Q.pa_eb - Q.pa_raw)[m].mean()) if not space.startswith("image->") else np.nan,
                                        fl_w_mean=float(Q.w_fl[m].mean()),
                                        fl_mean_abs_shift_mm=float(fl_shift.abs()[m].mean()),
                                        fl_mean_signed_shift_mm=float(fl_shift[m].mean()),
                                        pa_dist=float((Q.pa_eb - Q.pa_raw).abs()[m].sum() / N_TEST / 6 / 3) if not space.startswith("image->") else np.nan,
                                        fl_dist=float(fl_shift.abs()[m].sum() / N_TEST / 12 / 3)))
        # centres: physical vs the J-implied (image centre transformed with the cell's median tilt)
        for key, sub in b.groupby(b.pa_cell.fillna("").replace("", "(scanner)")):
            ii = sub.index
            ci = float(a.loc[ii, "pa_centre"].median())
            dl = float(delta.loc[ii].median())
            cJ = float(np.degrees(np.arctan(c * np.tan(np.radians(ci - dl))) + np.arctan(c * np.tan(np.radians(dl)))))
            cp = float(sub.pa_centre.median())
            fcell = sub["cell"].fillna("").replace("", "(scanner)").mode().iloc[0]
            muJ = float((a.loc[ii, "mu_fl"] * fac.loc[ii]).median())
            mup = float(sub.mu_fl.median())
            centre_rows.append(dict(group=SHORT[gg], pa_cell=key, fl_cell=fcell, n=len(ii), pa_centre_img=ci,
                                    pa_centre_J_implied=cJ, pa_centre_phys=cp, d_pa=cp - cJ,
                                    fl_centre_img=float(a.loc[ii, "mu_fl"].median()), fl_centre_J_implied=muJ,
                                    fl_centre_phys=mup, d_fl=mup - muJ,
                                    tau_fl_J_implied=float((a.loc[ii, "tau_fl"] * fac.loc[ii]).median()),
                                    tau_fl_phys=float(sub.tau_fl.median())))
        ratio_rows.append(dict(group=SHORT[gg], fov_mm_img=float(a.fov_mm.median()), fov_mm_phys=float(b.fov_mm.median()),
                               flfov_img_med=float(a.fl_fov.median()), flfov_phys_med=float(b.fl_fov.median()),
                               n_gt_225_img=int((a.fl_fov > 2.25).sum()), n_gt_225_phys=int((b.fl_fov > 2.25).sum()),
                               n_gt_25_img=int((a.fl_fov > 2.5).sum()), n_gt_25_phys=int((b.fl_fov > 2.5).sum()),
                               k_raw_phys_med=float((b.fl_raw_unit * np.sin(np.radians(b.pa_raw)) / b.mt).median()),
                               k_shrunk_phys_med=float((b.fl_eb * np.sin(np.radians(b.pa_eb)) / b.mt).median()),
                               k_B1J4_med=kstats(base, idx)["k_med"],
                               k_B1J4_mtm03_med=kstats(base.assign(mt_mm=base.mt_mm - 0.3), idx)["k_med"]))
    # Telemed k on B1_J4 (image space)
    for gg in TELEMED + (L853, L512):
        idx = lay.index[lay.group == gg]
        ratio_rows.append(dict(group=SHORT[gg], k_B1J4_med=kstats(base, idx)["k_med"],
                               k_C18_med=kstats(C18, idx)["k_med"],
                               flfov_img_med=float(Pimg.loc[idx, "fl_fov"].median())))
    S1, S2, S3 = pd.DataFrame(shrink_rows), pd.DataFrame(centre_rows), pd.DataFrame(ratio_rows)
    S1.to_csv(OUT / "diag_shrink_by_cell.csv", index=False)
    S2.to_csv(OUT / "diag_centres.csv", index=False)
    S3.to_csv(OUT / "diag_flfov_coupling.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(S1.round(4).to_string(index=False))
    print(S2.round(3).to_string(index=False))
    print(S3.round(3).to_string(index=False))
    return dict(shrink=S1, centres=S2, ratio=S3)


# ====================================================================================== predictions + ledger
def ev_probe(mu: float, sigma: float, tau: float) -> dict:
    """b7.ev_probe: prior gain ~ N(mu, sigma^2), public reading with noise tau; value of probing and adopting
    when the posterior mean > 0."""
    from scipy.stats import norm
    s = math.sqrt(sigma ** 4 / (sigma ** 2 + tau ** 2)) if sigma > 0 else 0.0
    if s <= 0:
        return dict(value=max(mu, 0.0), evsi=0.0, p_better=float(mu > 0))
    v = s * norm.pdf(mu / s) + mu * norm.cdf(mu / s)
    return dict(value=float(v), evsi=float(v - max(mu, 0.0)), p_better=float(norm.cdf(mu / sigma)))


LB_CAL = 2.0     # LB / transfer-model ratio on the two shrinkage LB facts (C12->C18: 1.9 pre-reg, 3.7 regime;
                 # C1->C2: 2.9 pre-reg, 1.8 regime); applied to the mean of the two transfer predictions
EFF_CAP = 0.6    # |predicted delta| <= 0.6 x distance (the largest LB efficiency seen, C12->C18, was 0.78)
D_NON_CAP = 0.8  # conservative MT prior: non-Telemed bias <= 0.8 mm (the DLTrack-pair fact alone pushes it to the
                 # grid edge once the Telemed bias is pinned at ~0.33 mm by J4)

# Judgement adjustments to the transfer predictions (applied after the calibration; reasons are in the ledger)
ADJUST = {
    "E_LUM": (0.5, 0.5, "coupling check disagrees on the 14 Lumify rows with FL/FOV > 2.5 (mostly RF): k = "
                        "FL sinPA / MT is 1.073 with J and 1.025 with the physical EB (expert 1.067), so the extra "
                        "physical shrinkage there may overshoot; prediction halved, SD >= 0.5 x dist"),
    "E_LUMf": (0.5, 0.5, "as E_LUM"),
    "G_LUMf": (0.5, 0.5, "as E_LUM (G_LUMf carries the same physical-EB change on the long rows)"),
}

RULES = {
    "J5b1": "pre-registered J5 rule: dLB <= -0.001 -> adopt -0.3 and probe J5b2 (-0.5) next; dLB <= -0.006 -> J5b2 at "
            "once; dLB > -0.001 -> stop, keep C18 non-Telemed MT. Medium rule risk (LB-chosen scanner-level constant on "
            "a pre-declared grid); Final A only unless an edge-definition method reproduces it.",
    "E_LUM": "method choice (pipeline path): dLB <= 0 -> adopt the pipeline-exact physical rebuild for Lumify; dLB > 0 "
             "-> keep J (post-hoc transform as the declared pipeline step). If dLB >= +0.003, the physical EB "
             "over-shrinks the FL/FOV > 2.5 rows: use GJ_LUMf (gate on J) instead of G_LUMf next.",
    "E_SIE": "method choice: dLB <= 0 -> adopt the physical rebuild for Siemens; else keep J. (PA/FL split not needed: "
             "PA part dist 0.0012.)",
    "C21b": "dLB < 0 -> adopt Telemed PA unshrunk; if K25b also scored and won, keep the better and probe K27b. "
            "dLB >= 0 -> reject (stop).",
    "U_SIEp": "benchmark-mixed method choice (t142 / regime for, VL against): adopt if dLB < 0; else stop. Cpa_SIE is "
              "the same file within dist 1e-5 (Siemens PA confidence is lowered only by FL-only flags), so a win also "
              "settles the C24 rule for Siemens.",
    "G_SIEf": "regime gate (C23 rule in physical mm): adopt if dLB < 0; else stop. Preferred over U_SIEf (raw on the 2 "
              "rows with FL/FOV > 2.25 that VL says must stay shrunk).",
    "U_SIEf": "raw FL on all Siemens rows: adopt only if dLB < 0 AND dLB < dLB(G_SIEf) - 0.0005 (it overrides the "
              "benchmark-flagged rows); schedule only after G_SIEf.",
    "G_LUMf": "regime gate in physical mm on top of the physical rebuild: adopt if dLB < min(0, dLB(E_LUM)); the LUM.fl "
              "cell then takes the best scored of {J, E_LUMf, G_LUMf}. If E_LUM lost, probe GJ_LUMf instead.",
    "K25b": "dLB < 0 -> adopt; dLB <= -0.003 -> probe K26b next; if C21b also won keep the better and probe K27b; "
            "dLB >= 0 -> reject. Telemed PA uplift cap +0.9 deg respected (mean +0.18 / +0.47 deg).",
    "J4b": "grid point of an LB-chosen constant: dLB <= -0.0005 -> move Telemed MT to -0.25; otherwise keep -0.5. No "
           "-0.8 point (posterior +0.0065). Line search closed after this point.",
    "U_LUMp": "benchmarks against (VL: PA-relevant flags, efficiency 0.49): adopt only if dLB < -0.0005; else stop.",
    "J5b2": "only if J5b1 won: adopt -0.5 if dLB(J5b2) < dLB(J5b1) - 0.0005; else keep -0.3 and close (or J5c if "
            "J5b2 lost by > 0.002).",
    "J5c": "only if J5b1 won and J5b2 lost by > 0.002 vs J5b1: adopt -0.15 if dLB(J5c) < dLB(J5b1) - 0.0005.",
    "GJ_LUMf": "fallback for G_LUMf when E_LUM lost: adopt if dLB < 0.",
    "K26b": "only if K25b gained > 0.003: adopt if dLB(K26b) < dLB(K25b).",
    "K27b": "only if both C21b and K25b won: adopt if dLB(K27b) < min(dLB(C21b), dLB(K25b)).",
}


def mt_post_capped(cap: float) -> dict:
    T = pd.read_csv(WORK / "mtpost_grid.csv")
    S = T[T.d_non <= cap + 1e-9]
    W = S.w.sum()
    return {k: float(S[k].sum() / W) for k in ("J4b", "J4c", "J5", "J5b", "J5c")} | dict(
        d_tel=float((S.w * S.d_tel).sum() / W), d_non=float((S.w * S.d_non).sum() / W))


def cpl_pred(r: dict, lam: float) -> tuple[float, str]:
    """critic7 coupling-blend benchmark (blend on the shrunk PA) scaled to this file's Telemed PA move."""
    cpl = pd.read_csv(CPL_BENCH)
    q = cpl[(cpl.base == "pa_eb") & (cpl.lam == lam)].set_index("slice")
    sc = {s_: float(q.loc[s_, "d"] / max(q.loc[s_, "moved_pa_units"], 1e-9) * r["dist"]["pa_deg"])
          for s_ in ("clean10_2r", "vl3", "t142")}
    return float(np.mean(list(sc.values()))), ", ".join(f"{k} {v:+.4f}" for k, v in sc.items())


def cmd_ledger(args) -> pd.DataFrame:
    B = json.loads((WORK / "probes_build.json").read_text())["probes"]
    pred = {}
    for sp in ("IMG", "SIE", "LUM"):
        p = WORK / f"pred_{sp}.json"
        if p.exists():
            pred.update(json.loads(p.read_text()))
    full = json.loads((WORK / "mtpost.json").read_text())["posterior"]
    capd = mt_post_capped(D_NON_CAP)
    rows = []
    for name, r in B.items():
        d = r["dist"]["S"]
        tau = r["sd_pub_priv"]
        why = ""
        if name in pred:
            p = pred[name]
            mu_d = LB_CAL * 0.5 * (p["prereg"] + p["regime"])
            mu_d = float(np.clip(mu_d, -EFF_CAP * d, EFF_CAP * d))
            sig = max(0.3 * d, LB_CAL * abs(p["regime"] - p["prereg"]) / 2, 0.0005)
            space = "physical" if ("LUM" in name or "SIE" in name) else "image"
            why = (f"benchmark transfer on the {space}-space frame: "
                   f"prereg {p['prereg']:+.4f}, regime {p['regime']:+.4f} (clean10 {p['clean10_2r']:+.4f}, VL "
                   f"{p['vl3']:+.4f}, dyn {p['dyn']:+.4f}, t142 {p['t142']:+.4f}); x{LB_CAL:g} LB calibration, cap "
                   f"{EFF_CAP} x dist")
            if name in ADJUST:
                f_mu, f_sd, txt = ADJUST[name]
                mu_d *= f_mu
                sig = max(sig, f_sd * d)
                why += "; ADJUSTED: " + txt
        elif name in ("J4b", "J5b1", "J5b2", "J5c"):
            key = {"J4b": "J4b", "J5b1": "J5", "J5b2": "J5b", "J5c": "J5c"}[name]
            mu_d = capd[key]
            sig = max(full[key]["sd"], abs(full[key]["mean"] - capd[key]), 0.0005)
            if name in ("J5b2", "J5c"):
                mu_d = capd[key] - capd["J5"]          # relative to J5b1 (the state it would be compared with)
            why = (f"MT ABC posterior from the DLTrack c129/c130 pair + the B1->B1_J4 fact (same public subset), "
                   f"d_non <= {D_NON_CAP}: d_tel {capd['d_tel']:.2f} mm, d_non {capd['d_non']:.2f} mm (uncapped "
                   f"{full['d_tel_mean']:.2f} / {full['d_non_mean']:.2f}; uncapped mean {full[key]['mean']:+.4f})"
                   + ("; delta shown vs J5b1" if name in ("J5b2", "J5c") else ""))
        elif name in ("K25b", "K26b"):
            mu_d, txt = cpl_pred(r, 0.25 if name == "K25b" else 0.5)
            sig = max(0.4 * d, 0.0008)
            why = f"coupling-blend benchmark (critic7, lambda {0.25 if name == 'K25b' else 0.5} on the shrunk PA), scaled to the move: {txt}"
        elif name == "K27b":
            m1 = LB_CAL * 0.5 * (pred["C21b"]["prereg"] + pred["C21b"]["regime"])
            m2, _ = cpl_pred(B["K25b"], 0.25)
            mu_d, sig = m1 + m2, max(0.4 * d, 0.001)
            why = "C21b + K25b predictions added (approximate: the cells overlap)"
        else:
            mu_d, sig = 0.0, max(0.3 * d, 0.0005)
        ev = ev_probe(-mu_d, sig, tau)
        rows.append(dict(name=name, file=r["file"], cells="+".join(r["cells"]), kind=r["kind"],
                         slot_candidate=r["slot_candidate"], change=r["desc"], dist_to_base=round(d, 5),
                         dist_pa=round(r["dist"]["pa_deg"], 5), dist_fl=round(r["dist"]["fl_mm"], 5),
                         dist_mt=round(r["dist"]["mt_mm"], 5), rows_changed=r["rows_changed"],
                         mean_change=json.dumps(r["mean_change"]), mean_abs_change=json.dumps(r["mean_abs_change"]),
                         k_before=json.dumps({k: round(v["k_med"], 3) for k, v in r["k_before"].items()}),
                         k_after=json.dumps({k: round(v["k_med"], 3) for k, v in r["k_after"].items()}),
                         sd_pub=round(r["sd_pub"], 5), sd_pub_priv=round(tau, 5), pred_delta=round(mu_d, 5),
                         pred_sd=round(sig, 5), p_win=round(ev["p_better"], 3), ev_value=round(ev["value"], 5),
                         evsi=round(ev["evsi"], 5), adopt_rule=RULES.get(name, "diagnostic only (not scheduled)"),
                         model=why, isolated=r["isolated"], validate=r["validate"], clips_constant=r["clips_constant"],
                         sha16=r["sha16"]))
    L = pd.DataFrame(rows).sort_values(["slot_candidate", "ev_value"], ascending=[False, False])
    L.insert(0, "rank", np.where(L.slot_candidate, np.arange(1, len(L) + 1), 0))
    L.to_csv(OUT / "ledger_v9.csv", index=False)
    pd.set_option("display.width", 250)
    print(L[["rank", "name", "cells", "dist_to_base", "sd_pub_priv", "pred_delta", "pred_sd", "p_win", "ev_value",
             "evsi", "isolated", "clips_constant"]].to_string(index=False))
    return L


# ====================================================================================== main
def cmd_all(args) -> None:
    cmd_repro(args)
    cmd_caches(args)
    for sp in ("IMG", "SIE", "LUM"):
        run_child("state", sp)
    cmd_build(args)
    for sp in ("IMG", "SIE", "LUM"):
        run_child("predict", sp)
    cmd_mtpost(args)
    cmd_diag(args)
    cmd_ledger(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["repro", "caches", "state", "predict", "build", "mtpost", "diag", "ledger", "all"])
    ap.add_argument("--space", choices=list(SPACE_GROUP), default="IMG")
    ap.add_argument("--nsim", type=int, default=300)
    ap.add_argument("--h1", type=float, default=0.0010)
    ap.add_argument("--h2", type=float, default=0.0004)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"repro": cmd_repro, "caches": cmd_caches, "state": cmd_state, "predict": cmd_predict, "build": cmd_build,
     "mtpost": cmd_mtpost, "diag": cmd_diag, "ledger": cmd_ledger, "all": cmd_all}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
