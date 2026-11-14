"""Evaluate geometry_v2 variants with the official UMUD metric (light, CPU, one image at a time).

Sections (select with --sections):
  static   : 35-image OSF static benchmark. GT = mean of R1..R7 (im_19 R7_MT -> NaN), plus the
             2-rater style GTs (mean of R7 and each other rater, averaged) and R7 alone. Reports
             all 35 and the 10 clean (not in training) images, and leave-one-out selection.
  gtmasks  : GT apo/fasc masks of the static images that are in the training data (subset).
  train142 : 142 training images with BOTH GT masks: classical PA vs GT-fragment PA, FL model
             consistency (k = FL sin(PA) / MT) and robustness to mask degradations.
  dynamic  : 167-frame GM calf-raise benchmark: classical PA vs expert PA (deep vs horizontal
             reference), using the two cached predicted-mask frames (0, 80) for the compartment.

Inputs: --source npz (cached binary U-Net++ masks) or --source maskbank (MASKBANK FORMAT npz files,
lazy keys) with --apo-model/--fasc-model[/--orient-model]. Nothing here runs a neural network.
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umud import geometry_v2 as g2  # noqa: E402
from umud.metric import TOL, score_arrays  # noqa: E402

warnings.filterwarnings("ignore")
from umud.paths import RESEARCH_ROOT

SCRATCH = RESEARCH_ROOT
BENCH = ROOT / "data/external_umud_benchmark/static/benchmark_dataset_architecture_v0.1.0"
DYN = ROOT / "data/external_umud_benchmark/dynamic/benchmark_dataset_architecture_GM_dynamic_v0.1.0"
COMP = ROOT / "data/umud-challenge-muscle-architecture-in-ultrasound-data"
CLEAN = [f"im_{k:02d}_arch" for k in (2, 3, 14, 17, 20, 23, 24, 31, 33, 34)]
RATERS = [f"R{k}" for k in range(1, 8)]
C3 = ("pa_deg", "fl_mm", "mt_mm")


def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


# ------------------------------------------------------------------------------------ labels
def static_labels() -> pd.DataFrame:
    lab = pd.read_excel(BENCH / "Results_benchmark_architecture_v0.1.0.xlsx", sheet_name="Manual_architecture")
    lab.loc[lab.ImageID == "im_19_arch", "R7_MT"] = np.nan  # 80.13 typo
    return lab.set_index("ImageID")


def gt_sets(lab: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """name -> DataFrame[pa_deg, fl_mm, mt_mm] indexed by ImageID."""
    def mean_of(rs):
        return pd.DataFrame({c: lab[[f"{r}_{k}" for r in rs]].mean(1) for c, k in zip(C3, ("PA", "FL", "MT"))})
    out = {"mean7": mean_of(RATERS), "R7": mean_of(["R7"])}
    for r in RATERS[:-1]:
        out[f"R7+{r}"] = mean_of(["R7", r])
    return out


# ------------------------------------------------------------------------------------ sources
class StaticSource:
    def __init__(self, kind: str, npz: Path | None = None, bank_root: Path | None = None,
                 apo_model: str | None = None, fasc_model: str | None = None, orient_model: str | None = None):
        self.kind = kind
        self.orient_model = orient_model
        if kind == "npz":
            self.z = np.load(npz)
        else:
            self.bank = g2.MaskBank(bank_root)
            self.apo_model, self.fasc_model = apo_model, fasc_model
        self.fallback_model = None

    def get(self, image_id: str):
        if self.kind == "npz":
            return self.z[f"{image_id}_apo"], self.z[f"{image_id}_fasc"], None
        A = self.bank.prob("static", self.apo_model, image_id)
        F = self.bank.prob("static", self.fasc_model, image_id)
        O = self.bank.orient("static", self.orient_model, image_id) if self.orient_model else None
        return A, F, O

    def fallback(self, image_id: str):
        if self.kind != "maskbank" or not self.fallback_model:
            return None
        return self.bank.prob("static", self.fallback_model, image_id)


# ------------------------------------------------------------------------------------ scoring
PRIOR = {"pa_deg": 16.0, "fl_mm": 62.6, "mt_mm": 17.8}  # training GT-mask medians (failure fill)


def score_frame(pred: pd.DataFrame, gt: pd.DataFrame, ids) -> dict:
    ids = [i for i in ids if i in pred.index]
    n_fail = int(pred.loc[ids, list(C3)].isna().any(axis=1).sum())
    p = {c: pred.loc[ids, c].fillna(PRIOR[c]).to_numpy(float) for c in C3}
    t = {c: gt.loc[ids, c].to_numpy(float) for c in C3}
    sc, rep = score_arrays(p, t, nan_policy="omit", report=True)
    r = {"score": sc}
    for c in C3:
        r[f"{c[:2]}_n"] = rep[c]["norm"]
        r[f"{c[:2]}_bias"] = rep[c]["bias"]
    r["n"] = len(ids)
    r["n_fail"] = n_fail
    return r


def score_all(pred: pd.DataFrame, gts: dict, subsets: dict) -> dict:
    out = {}
    for sname, ids in subsets.items():
        m = score_frame(pred, gts["mean7"], ids)
        out.update({f"{sname}_{k}": v for k, v in m.items() if k not in ("n",) and not (k == "n_fail" and sname != "all35")})
        out[f"{sname}_2rater"] = float(np.mean([score_frame(pred, gts[f"R7+{r}"], ids)["score"] for r in RATERS[:-1]]))
        out[f"{sname}_R7"] = score_frame(pred, gts["R7"], ids)["score"]
    return out


def per_image_err(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.Series:
    e = [np.abs(pred[c].fillna(PRIOR[c]) - gt.loc[pred.index, c]) / TOL[c] for c in C3]
    return pd.concat(e, axis=1).mean(1, skipna=False)  # NaN if any component is missing


def offline_fl(row: pd.Series, method: str, pa_col: str, k: float = 1.065, tol: float = 0.25,
               w: float = 0.5, primary: str = "fl_straight") -> float:
    """FL headline variants that only need scalar outputs (triangle / gated / averages)."""
    pa = row[pa_col]
    mt = row["mt_mm"]
    tri = k * mt / np.sin(np.radians(pa)) if np.isfinite(pa) and pa > 0.5 else np.nan
    if method == "triangle":
        return tri
    if method == "gated":
        prim = row[primary]
        if not np.isfinite(prim):
            return tri
        if not np.isfinite(tri):
            return prim
        return prim if abs(prim - tri) / tri <= tol else w * prim + (1 - w) * tri
    if method.startswith("avg:") or method.startswith("med:"):
        vals = [row[f"fl_{m}"] for m in method[4:].split("+")]
        return np.nanmedian(vals) if method.startswith("med:") else np.nanmean(vals)
    return row[f"fl_{method}"]


# ------------------------------------------------------------------------------------ static
STRUCTS = {
    "v1": g2.GeoConfig.v1(),
    "v1+tangent": g2.GeoConfig.v1().with_(edge_extrap="tangent"),
    "v1+merge": g2.GeoConfig.v1().with_(apo_merge=True, apo_thick_clip=3.0),
    "v1+superficial": g2.GeoConfig.v1().with_(pair_mode="superficial"),
    "v1+sign": g2.GeoConfig.v1().with_(frag_sign_consistency=True),
    "v1+poly2r": g2.GeoConfig.v1().with_(edge_fit="poly2r"),
    "v1+savgol": g2.GeoConfig.v1().with_(edge_fit="savgol", edge_extrap="tangent"),
    "v1+poly1": g2.GeoConfig.v1().with_(edge_fit="poly1"),
    "v1+mt_poly2": g2.GeoConfig.v1().with_(mt_fit=None),
    "v1+mt_e": g2.GeoConfig.v1().with_(mt_xpos=(0.1, 0.5, 0.9)),
    "v1+blobfilter": g2.GeoConfig.v1().with_(frag_max_minor_ratio=0.35),
}


def override(cfg: g2.GeoConfig, args) -> g2.GeoConfig:
    kw = {}
    if getattr(args, "apo_thr", None) is not None:
        kw["apo_thr"] = args.apo_thr
    if getattr(args, "fasc_thr", None) is not None:
        kw["fasc_thr"] = args.fasc_thr
    return cfg.with_(**kw) if kw else cfg


def v2_default() -> g2.GeoConfig:
    """Recommended default (see the report)."""
    return g2.GeoConfig(pair_mode="superficial", apo_merge=False, apo_thick_clip=3.0, edge_fit="poly2",
                        edge_extrap="tangent", mt_fit="poly1", frag_sign_consistency=False,
                        pa_method="fusion", fl_method="ensemble", st_kind="radon", apo_span_ref="viewport")


def run_static(src: StaticSource, structs: dict, lab: pd.DataFrame, pa_heads: dict | None = None,
               log=print) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for k, image_id in enumerate(lab.index):
        A, F, O = src.get(image_id)
        img = gray(BENCH / f"{image_id}.tif")
        ppc = float(lab.loc[image_id, "Scale_pixel_per_cm"])
        for name, cfg in structs.items():
            heads = (pa_heads or {}).get(name, [cfg.pa_method])
            for pm in heads:
                c = cfg.with_(pa_method=pm, compute_structure=cfg.compute_structure or pm in ("structure", "fusion"))
                fb = src.fallback(image_id) if not name.startswith("v1") else None
                r = g2.measure(A, F, img, ppc, c, orient=O, apo_fallback=fb)
                rows.append(dict(image=image_id, struct=name, pa_head=pm, **r.as_row()))
        if k % 10 == 9:
            log(f"  static {k + 1}/{len(lab)} {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def static_section(args, log) -> dict:
    lab = static_labels()
    gts = gt_sets(lab)
    subsets = {"all35": list(lab.index), "clean10": CLEAN}
    if args.source == "npz":
        src = StaticSource("npz", npz=Path(args.npz))
    else:
        src = StaticSource("maskbank", bank_root=Path(args.maskbank), apo_model=args.apo_model,
                           fasc_model=args.fasc_model, orient_model=args.orient_model)
        src.fallback_model = args.apo_fallback_model
    structs = dict(STRUCTS)
    structs["v2"] = v2_default()
    structs["v2+sign"] = v2_default().with_(frag_sign_consistency=True)
    structs["v2+poly"] = v2_default().with_(edge_extrap="poly")
    structs["v2+end_linear"] = v2_default().with_(edge_extrap="end_linear")
    structs["v2+mt_poly2"] = v2_default().with_(mt_fit=None)
    structs["v2+merge"] = v2_default().with_(apo_merge=True)
    structs["v2+central5"] = v2_default().with_(central_n=5)
    if args.structs:
        structs = {k: v for k, v in structs.items() if k in args.structs.split(",")}
    structs = {k: override(v, args) for k, v in structs.items()}
    pa_heads = {"v2": ["global", "local", "central3", "structure", "fusion"] + (["field"] if args.orient_model else []),
                "v1": ["global", "fusion"]}
    D = run_static(src, structs, lab, pa_heads, log)
    D.to_csv(args.out / f"static_per_image{args.tag}.csv", index=False)
    # ---- variant table
    fl_methods = ["straight", "central3", "streamline", "constructed", "triangle", "gated",
                  "avg:straight+constructed", "avg:central3+constructed", "avg:straight+central3",
                  "avg:straight+central3+constructed", "med:straight+central3+constructed"]
    rows = []
    preds = {}
    for (st, ph), d in D.groupby(["struct", "pa_head"]):
        d = d.set_index("image")
        for flm in fl_methods:
            fl = d.apply(lambda r: offline_fl(r, flm, f"pa_{ph}"), axis=1)
            pred = pd.DataFrame({"pa_deg": d[f"pa_{ph}"], "fl_mm": fl, "mt_mm": d["mt_mm"]})
            key = f"{st}|pa={ph}|fl={flm}"
            preds[key] = pred
            rows.append(dict(struct=st, pa=ph, fl=flm, **score_all(pred, gts, subsets)))
    T = pd.DataFrame(rows).sort_values("all35_score")
    T.to_csv(args.out / f"static_variant_table{args.tag}.csv", index=False)
    # ---- gate tolerance grid (few principled values)
    grows = []
    for st in [s_ for s_ in ["v2"] if s_ in structs]:
        d = D[(D.struct == st) & (D.pa_head == ("fusion"))].set_index("image")
        for prim, tol, w in itertools.product(["fl_straight", "fl_constructed"], [0.1, 0.2, 0.3, 0.5], [0.5, 0.0]):
            fl = d.apply(lambda r: offline_fl(r, "gated", "pa_fusion", tol=tol, w=w, primary=prim), axis=1)
            pred = pd.DataFrame({"pa_deg": d["pa_fusion"], "fl_mm": fl, "mt_mm": d["mt_mm"]})
            preds[f"{st}|pa=fusion|fl=gated({prim[3:]},{tol},{w})"] = pred
            grows.append(dict(struct=st, primary=prim, tol=tol, w=w, **score_all(pred, gts, subsets)))
    G = pd.DataFrame(grows)
    G.to_csv(args.out / f"static_gate_grid{args.tag}.csv", index=False)
    # ---- leave-one-out selection over (a) everything, (b) a few principled candidates
    gt = gts["mean7"]
    E = pd.DataFrame({k: per_image_err(p, gt) for k, p in preds.items()})
    E = E.loc[:, E.notna().all()]

    def loo(cols):
        errs = []
        for i in E.index:
            rest = E.drop(index=i)[cols].mean()
            errs.append(E.loc[i, rest.idxmin()])
        return float(np.mean(errs)), E[cols].mean().idxmin(), float(E[cols].mean().min())

    few = [c for c in E.columns if c.split("|")[0] in ("v1", "v2", "v2+poly") and
           c.split("|")[1] in ("pa=global", "pa=fusion", "pa=structure") and
           c.split("|")[2] in ("fl=straight", "fl=central3", "fl=constructed", "fl=avg:straight+constructed",
                               "fl=avg:central3+constructed", "fl=triangle",
                               "fl=avg:straight+central3+constructed", "fl=med:straight+central3+constructed")]
    loo_all = loo(list(E.columns))
    loo_few = loo(few)
    res = dict(loo_all=dict(loo_score=loo_all[0], insample_best=loo_all[1], insample_score=loo_all[2], n_cands=E.shape[1]),
               loo_few=dict(loo_score=loo_few[0], insample_best=loo_few[1], insample_score=loo_few[2], n_cands=len(few)))
    # clean-subset LOO for the few set
    Ec = E.loc[CLEAN]
    errs = []
    for i in Ec.index:
        rest = E.drop(index=i)[few].mean()
        errs.append(Ec.loc[i, rest.idxmin()])
    res["loo_few_clean10"] = float(np.mean(errs))
    # ---- inter-rater references on the same subsets
    ref = {}
    for sname, ids in subsets.items():
        floor = []
        for a, b in itertools.combinations(RATERS, 2):
            rest = [r for r in RATERS if r not in (a, b)]
            p = pd.DataFrame({c: lab.loc[ids, [f"{r}_{k}" for r in rest]].mean(1) for c, k in zip(C3, ("PA", "FL", "MT"))})
            t = pd.DataFrame({c: lab.loc[ids, [f"{r}_{k}" for r in (a, b)]].mean(1) for c, k in zip(C3, ("PA", "FL", "MT"))})
            floor.append(score_frame(p, t, ids)["score"])
        ref[f"{sname}_consensus5_vs_2rater"] = float(np.mean(floor))
        dl = pd.DataFrame({c: lab.loc[ids, f"DLTrack_{k}"] for c, k in zip(C3, ("PA", "FL", "MT"))})
        ref[f"{sname}_DLTrack_vs_mean7"] = score_frame(dl, gts["mean7"], ids)["score"]
    res["reference"] = ref
    return dict(table=T, gate=G, summary=res)


# ------------------------------------------------------------------------------------ GT masks on static
def gtmask_section(args, log) -> pd.DataFrame:
    lab = static_labels()
    gts = gt_sets(lab)
    mt = pd.read_csv(SCRATCH / "data-labels-geometry/out/bench_matches_central.csv")
    mt = mt[mt.center_corr > 0.95]
    ga = {r.bench: COMP / "apo_masks_v1/apo_masks_new_model_v1" / f"{r.train}.tif" for r in mt[mt.kind == "apo"].itertuples()}
    gf = {r.bench: COMP / "fasc_masks_v1/fasc_masks_new_model_v1" / f"{r.train}.tif" for r in mt[mt.kind == "fasc"].itertuples()}
    z = np.load(args.npz)
    rows = []
    for image_id in lab.index:
        img = gray(BENCH / f"{image_id}.tif")
        ppc = float(lab.loc[image_id, "Scale_pixel_per_cm"])
        for a_src, f_src in [("gt", "gt"), ("gt", "pred"), ("pred", "gt")]:
            if (a_src == "gt" and image_id not in ga) or (f_src == "gt" and image_id not in gf):
                continue
            A = mask(ga[image_id]) if a_src == "gt" else z[f"{image_id}_apo"]
            F = mask(gf[image_id]) if f_src == "gt" else z[f"{image_id}_fasc"]
            if A.shape != img.shape or F.shape != img.shape:
                continue
            for name, cfg in [("v1", g2.GeoConfig.v1()), ("v2", v2_default())]:
                r = g2.measure(A, F, img, ppc, cfg)
                rows.append(dict(image=image_id, apo=a_src, fasc=f_src, cfg=name, **r.as_row()))
    D = pd.DataFrame(rows)
    D.to_csv(args.out / "static_gtmask_per_image.csv", index=False)
    out = []
    for (a, f, c), d in D.groupby(["apo", "fasc", "cfg"]):
        d = d.set_index("image")
        for pa_col, fl_col in [("pa_global", "fl_straight"), ("pa_fusion", "fl_straight"), ("pa_structure", "fl_straight"),
                               ("pa_global", "fl_constructed"), ("pa_fusion", "fl_constructed")]:
            pred = pd.DataFrame({"pa_deg": d[pa_col], "fl_mm": d[fl_col], "mt_mm": d["mt_mm"]})
            out.append(dict(apo=a, fasc=f, cfg=c, pa=pa_col, fl=fl_col, **score_frame(pred, gts["mean7"], list(d.index))))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------ 142 dual GT
def degrade_truncate(A: np.ndarray, keep: float) -> np.ndarray:
    """Keep only the central ``keep`` fraction of the columns where the mask has pixels."""
    cols = np.flatnonzero(A.any(0))
    if cols.size == 0:
        return A
    lo, hi = cols.min(), cols.max()
    c = 0.5 * (lo + hi)
    h = 0.5 * keep * (hi - lo)
    B = A.copy()
    B[:, : int(c - h)] = False
    B[:, int(c + h) + 1:] = False
    return B


def add_spurious_band(A: np.ndarray, F: np.ndarray, rng) -> np.ndarray:
    """Add a parallel band in the middle of the superficial muscle, crossed by fascicles."""
    bands = g2.extract_bands(A, g2.GeoConfig(apo_merge=False))
    if len(bands) < 2:
        return A
    b0, b1 = bands[0], bands[1]
    H, W = A.shape
    s0, i0 = b0.line("bot")
    s1, i1 = b1.line("top")
    B = A.copy()
    xs = np.arange(int(max(b0.x0, b1.x0)), int(min(b0.x1, b1.x1)))
    th = max(4, int(0.5 * b1.thickness))
    for x in xs:
        y = int(0.5 * ((s0 * x + i0) + (s1 * x + i1)))
        B[max(0, y - th // 2): min(H, y + th // 2 + 1), x] = True
    return B


def train142_section(args, log) -> dict:
    pairs = pd.read_csv(SCRATCH / "data-labels-geometry/out/apo_fasc_pairs.csv")
    rng = np.random.default_rng(0)
    base_cfg = v2_default()
    rows, deg = [], []
    t0 = time.time()
    for k, r in enumerate(pairs.itertuples()):
        A = mask(COMP / "apo_masks_v1/apo_masks_new_model_v1" / f"{r.apo}.tif")
        F = mask(COMP / "fasc_masks_v1/fasc_masks_new_model_v1" / f"{r.fasc}.tif")
        if A.shape != F.shape:
            continue
        img = gray(COMP / "apo_imgs_v1/apo_images_new_model_v1" / f"{r.apo}.tif")
        if img.shape != A.shape:
            img = np.asarray(Image.fromarray(img).resize((A.shape[1], A.shape[0]), Image.BILINEAR))
        res = g2.measure(A, F, img, None, base_cfg.with_(compute_streamline=True))
        row = dict(image=r.apo, layout=r.orig, **res.as_row())
        rows.append(row)
        # degradations (geometry only, no classical PA)
        light = base_cfg.with_(compute_structure=False, compute_streamline=False)
        ref_full = g2.measure(A, F, None, None, light.with_(edge_extrap="poly"))
        for kname, cfgd in [("poly2/poly", light.with_(edge_extrap="poly")),
                            ("poly2/tangent", light.with_(edge_extrap="tangent")),
                            ("poly2/end_linear", light.with_(edge_extrap="end_linear")),
                            ("poly2/global_linear", light.with_(edge_extrap="global_linear")),
                            ("poly2r/end_linear", light.with_(edge_fit="poly2r", edge_extrap="end_linear")),
                            ("poly1", light.with_(edge_fit="poly1", edge_extrap="poly"))]:
            for keep in (0.6, 0.4):
                tr = g2.measure(degrade_truncate(A, keep), F, None, None, cfgd)
                deg.append(dict(image=r.apo, test=f"truncate{keep}", cfg=kname,
                                d_mt=tr.mt_mm - ref_full.mt_mm,
                                d_fl=tr.variants.get("fl_straight", np.nan) - ref_full.variants.get("fl_straight", np.nan),
                                d_flc=tr.variants.get("fl_constructed", np.nan) - ref_full.variants.get("fl_constructed", np.nan),
                                d_pa=tr.pa_deg - ref_full.pa_deg, mt=ref_full.mt_mm,
                                fl=ref_full.variants.get("fl_straight", np.nan)))
        # spurious mid band: does the pair survive?
        A2 = add_spurious_band(A, F, rng)
        base = g2.measure(A, F, None, None, light)
        for mode in ("superficial", "coverage", "top2"):
            s = g2.measure(A2, F, None, None, light.with_(pair_mode=mode))
            deg.append(dict(image=r.apo, test="spurious_band", cfg=mode, d_mt=s.mt_mm - base.mt_mm,
                            ok=float(abs(s.mt_mm - base.mt_mm) < 0.05 * base.mt_mm) if np.isfinite(base.mt_mm) else np.nan))
        # second muscle: copy the superficial muscle below the deep aponeurosis (with fascicles)
        bands = g2.extract_bands(A, g2.GeoConfig(apo_merge=False))
        if len(bands) >= 2 and np.isfinite(base.mt_mm):
            H = A.shape[0]
            off = int(bands[1].yc - bands[0].yc)
            if bands[1].yc + off + bands[1].thickness < H:
                A3, F3 = A.copy(), F.copy()
                A3[off:] |= A[:H - off]
                F3[off:] |= F[:H - off]
                for mode in ("superficial", "coverage", "top2"):
                    s = g2.measure(A3, F3, None, None, light.with_(pair_mode=mode))
                    deg.append(dict(image=r.apo, test="second_muscle", cfg=mode, d_mt=s.mt_mm - base.mt_mm,
                                    ok=float(abs(s.mt_mm - base.mt_mm) < 0.05 * base.mt_mm)))
        # fragment dropout: PA/FL stability per method
        frs = g2.extract_fragments(F, light)
        if len(frs) >= 6:
            from skimage.measure import label as sklabel
            lab_ = sklabel(F, connectivity=2)
            ids = np.unique(lab_[lab_ > 0])
            for rep in range(3):
                keep_ids = rng.choice(ids, size=max(2, int(0.3 * ids.size)), replace=False)
                Fd = np.isin(lab_, keep_ids)
                s = g2.measure(A, Fd, None, None, light)
                deg.append(dict(image=r.apo, test="dropout0.3", cfg=f"rep{rep}",
                                d_pa_global=s.variants.get("pa_global", np.nan) - base.variants.get("pa_global", np.nan),
                                d_pa_central3=s.variants.get("pa_central3", np.nan) - base.variants.get("pa_central3", np.nan),
                                d_pa_local=s.variants.get("pa_local", np.nan) - base.variants.get("pa_local", np.nan),
                                d_fl_straight=s.variants.get("fl_straight", np.nan) - base.variants.get("fl_straight", np.nan),
                                d_fl_central3=s.variants.get("fl_central3", np.nan) - base.variants.get("fl_central3", np.nan),
                                fl_ref=base.variants.get("fl_straight", np.nan)))
        if k % 20 == 19:
            log(f"  train142 {k + 1}/{len(pairs)} {time.time() - t0:.0f}s")
    D = pd.DataFrame(rows)
    D.to_csv(args.out / "train142_per_image.csv", index=False)
    G = pd.DataFrame(deg)
    G.to_csv(args.out / "train142_degradations.csv", index=False)
    out = {}
    ok = D[np.isfinite(D.pa_global) & np.isfinite(D.pa_structure)]
    e = ok.pa_structure - ok.pa_global
    out["classical_vs_gtfrag_pa"] = dict(n=len(ok), mae=float(e.abs().mean()), bias=float(e.mean()),
                                         r=float(np.corrcoef(ok.pa_structure, ok.pa_global)[0, 1]),
                                         by_layout=ok.assign(e=e.abs()).groupby("layout").e.mean().round(2).to_dict())
    kk = D.fl_straight * np.sin(np.radians(D.pa_global)) / D.mt_mm
    out["k_fl_sin_pa_over_mt"] = dict(median=float(kk.median()), iqr=[float(kk.quantile(.25)), float(kk.quantile(.75))])
    for a, b in [("fl_constructed", "fl_straight"), ("fl_central3", "fl_straight"), ("fl_streamline", "fl_straight")]:
        ratio = D[a] / D[b]
        out[f"ratio_{a}/{b}"] = dict(median=float(ratio.median()), mad=float((ratio - ratio.median()).abs().median()))
    trn = G[G.test.str.startswith("truncate")]
    trn = trn.assign(rfl=trn.d_fl / trn.fl, rflc=trn.d_flc / trn.fl, rmt=trn.d_mt / trn.mt)
    out["truncation"] = trn.groupby(["test", "cfg"]).agg(
        mt_rel=("rmt", lambda v: float(np.nanmean(np.abs(v)))),
        fl_rel=("rfl", lambda v: float(np.nanmean(np.abs(v)))),
        fl_rel_med=("rfl", lambda v: float(np.nanmedian(np.abs(v)))),
        flc_rel=("rflc", lambda v: float(np.nanmean(np.abs(v)))),
        pa_abs=("d_pa", lambda v: float(np.nanmean(np.abs(v))))).round(3).reset_index().to_dict("records")
    for t in ("spurious_band", "second_muscle"):
        g = G[G.test == t]
        out[t] = g.groupby("cfg").ok.mean().round(3).to_dict()
        out[t + "_n"] = int(g.groupby("cfg").size().max()) if len(g) else 0
    dr = G[G.test == "dropout0.3"]
    if len(dr):
        out["dropout0.3_abs"] = {c: float(np.nanmean(np.abs(dr[c]))) for c in
                                 ["d_pa_global", "d_pa_central3", "d_pa_local", "d_fl_straight", "d_fl_central3"]}
    return out


# ------------------------------------------------------------------------------------ dynamic
def dynamic_section(args, log) -> dict:
    xl = DYN / "benchmark_architecture_GM_calf_raise_v0.1.0.xlsx"
    pa = pd.read_excel(xl, sheet_name="PA")[["frame", "Mean"]].rename(columns={"Mean": "x_pa"})
    fl = pd.read_excel(xl, sheet_name="FL")[["frame", "Mean"]].rename(columns={"Mean": "x_fl"})
    lab = pa.merge(fl, on="frame").dropna()
    lab["frame"] = lab.frame.astype(int)
    bank = g2.MaskBank(Path(args.maskbank)) if args.source == "maskbank" else None
    if bank is not None:
        return dynamic_bank(args, lab, bank, log)
    masks = {}
    for f in (0, 80):
        p = SCRATCH / f"data-labels-geometry/out/dyn_masks_{f}.npz"
        if p.exists():
            z = np.load(p)
            masks[f] = (z["A"], z["F"])
    if not masks:
        return {"error": "no cached dynamic masks"}
    ppc = 77.4
    rows = []
    for fr in lab.frame:
        mf = min(masks, key=lambda m: abs(m - fr))
        A, F = masks[mf]
        img = gray(DYN / "frames" / f"frame_{fr:05d}.png")
        r = dict(frame=fr, mask_frame=mf)
        for ref in ("deep", "horizontal"):
            # fragments come from the mask frame, the classical PA from the current frame
            c = v2_default().with_(pa_reference=ref, compute_streamline=False, st_use_frag_sign=True)
            res = g2.measure(A, F, img, ppc, c)
            r[f"pa_structure_{ref}"] = res.variants["pa_structure"]
            r[f"pa_global_{ref}"] = res.variants["pa_global"]
            if fr == mf:
                r[f"fl_straight_px_{ref}"] = res.variants["fl_straight"] * ppc / 10
                r[f"fl_constructed_px_{ref}"] = res.variants["fl_constructed"] * ppc / 10
        rows.append(r)
    D = lab.merge(pd.DataFrame(rows), on="frame")
    D.to_csv(args.out / "dynamic_per_frame.csv", index=False)
    out = {}
    for ref in ("deep", "horizontal"):
        e = D[f"pa_structure_{ref}"] - D.x_pa
        out[f"classical_pa_vs_{ref}"] = dict(mae=float(e.abs().mean()), bias=float(e.mean()),
                                            r=float(D[f"pa_structure_{ref}"].corr(D.x_pa)), n=int(e.notna().sum()))
    m = D[D.frame.isin(list(masks))]
    out["mask_frames"] = m[["frame", "x_pa", "x_fl", "pa_global_deep", "pa_global_horizontal", "pa_structure_deep",
                            "pa_structure_horizontal", "fl_straight_px_deep", "fl_constructed_px_deep",
                            "fl_constructed_px_horizontal"]].round(2).to_dict("records")
    return out


def dynamic_bank(args, lab: pd.DataFrame, bank: g2.MaskBank, log) -> dict:
    """Full geometry on every dynamic frame from MASKBANK maps (PA convention + FL in px)."""
    ppc = 77.4
    rows = []
    for fr in lab.frame:
        key = f"frame_{fr:05d}"
        A = bank.prob("dynamic", args.apo_model, key)
        F = bank.prob("dynamic", args.fasc_model, key)
        if A is None or F is None:
            continue
        img = gray(DYN / "frames" / f"{key}.png")
        fb = bank.prob("dynamic", args.apo_fallback_model, key) if args.apo_fallback_model else None
        r = dict(frame=fr)
        for ref in ("deep", "horizontal"):
            c = override(v2_default().with_(pa_reference=ref), args)
            res = g2.measure(A, F, img, ppc, c, apo_fallback=fb)
            for k in ("pa_global", "pa_structure", "pa_fusion", "fl_straight", "fl_central3", "fl_constructed",
                      "fl_ensemble", "fl_streamline"):
                v = res.variants.get(k, np.nan)
                r[f"{k}_{ref}"] = v * ppc / 10 if k.startswith("fl_") else v
            r[f"flags_{ref}"] = ";".join(res.flags)
        rows.append(r)
    D = lab.merge(pd.DataFrame(rows), on="frame")
    D.to_csv(args.out / f"dynamic_per_frame{args.tag}.csv", index=False)
    out = {}
    for ref in ("deep", "horizontal"):
        for k in ("pa_global", "pa_structure", "pa_fusion"):
            e = D[f"{k}_{ref}"] - D.x_pa
            out[f"{k}_vs_{ref}"] = dict(mae=float(e.abs().mean()), bias=float(e.mean()), r=float(D[f"{k}_{ref}"].corr(D.x_pa)),
                                       n=int(e.notna().sum()))
        for k in ("fl_straight", "fl_central3", "fl_constructed", "fl_ensemble", "fl_streamline"):
            e = D[f"{k}_{ref}"] - D.x_fl
            out[f"{k}_px_{ref}"] = dict(mae=float(e.abs().mean()), bias=float(e.mean()), r=float(D[f"{k}_{ref}"].corr(D.x_fl)),
                                       ratio=float(D[f"{k}_{ref}"].mean() / D.x_fl.mean()))
    return out


def test_section(args, log) -> dict:
    """Run v1 and v2 on the MASKBANK test maps (no labels): per-image CSV + group summaries."""
    bank = g2.MaskBank(Path(args.maskbank))
    lay = pd.read_csv(SCRATCH / "data-labels-geometry/out/test_layout.csv").set_index("image_id")
    clips = pd.read_csv(SCRATCH / "data-test-structure/test_clips.csv").set_index("name")
    tdir = COMP / "test_images_v2/test_set_v2"
    rows = []
    t0 = time.time()
    for k, image_id in enumerate(lay.index):
        A = bank.prob("test", args.apo_model, image_id)
        F = bank.prob("test", args.fasc_model, image_id)
        img = gray(tdir / image_id)
        ppc = float(lay.loc[image_id, "px_per_cm"])
        grp = lay.loc[image_id, "family"] + ("_" + lay.loc[image_id, "size"] if lay.loc[image_id, "family"] == "telemed_artus" else "")
        r = dict(image_id=image_id, group=grp, clip=int(clips.loc[image_id, "clip"]), px_per_cm=ppc)
        fb = bank.prob("test", args.apo_fallback_model, image_id) if args.apo_fallback_model else None
        for name, cfg in [("v1", override(g2.GeoConfig.v1(), args)), ("v2", override(v2_default(), args))]:
            if A is None or F is None:
                res = g2._nan_result("missing")
            else:
                res = g2.measure(A, F, img, ppc, cfg, apo_fallback=fb if name == "v2" else None)
            r.update({f"{name}_{kk}": vv for kk, vv in res.as_row().items()})
        rows.append(r)
        if k % 50 == 49:
            log(f"  test {k + 1}/{len(lay)} {time.time() - t0:.0f}s")
    D = pd.DataFrame(rows)
    D.to_csv(args.out / f"test_per_image{args.tag}.csv", index=False)
    cols = ["v1_pa_deg", "v2_pa_deg", "v2_pa_global", "v2_pa_structure", "v1_fl_mm", "v2_fl_mm", "v2_fl_straight",
            "v2_fl_central3", "v2_fl_constructed", "v1_mt_mm", "v2_mt_mm", "v2_confidence"]
    G = D.groupby("group")[cols].median().round(2)
    G["n"] = D.groupby("group").size()
    G["pair_changed"] = D.assign(c=D.v1_d_pair != D.v2_d_pair).groupby("group").c.sum()
    G["v2_nan"] = D.groupby("group").v2_mt_mm.apply(lambda v: int(v.isna().sum()))
    flags = D.v2_flags.fillna("").str.split(";").explode()
    return dict(groups=G.reset_index().to_dict("records"), flag_counts=flags[flags != ""].value_counts().to_dict())


# ------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="static,gtmasks,train142,dynamic")
    ap.add_argument("--source", default="npz", choices=["npz", "maskbank"])
    ap.add_argument("--npz", default=str(SCRATCH / "data-labels-geometry/out/bench_pred_masks.npz"))
    ap.add_argument("--maskbank", default=str(ROOT / "artifacts/maskbank"))
    ap.add_argument("--apo-model")
    ap.add_argument("--fasc-model")
    ap.add_argument("--orient-model")
    ap.add_argument("--apo-fallback-model", help="second apo map used when the first yields no pair (v2 only)")
    ap.add_argument("--apo-thr", type=float, help="override apo threshold for every config")
    ap.add_argument("--fasc-thr", type=float, help="override fascicle threshold for every config")
    ap.add_argument("--structs", help="comma list of structural configs to run (default all)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--out", type=Path, default=SCRATCH / "geometry-v2/eval")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    secs = args.sections.split(",")
    summary = {}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    if "static" in secs:
        log("== static benchmark")
        s = static_section(args, log)
        T = s["table"]
        cols = ["struct", "pa", "fl", "all35_score", "all35_pa_n", "all35_fl_n", "all35_mt_n", "all35_2rater",
                "all35_R7", "clean10_score", "clean10_pa_n", "clean10_fl_n", "clean10_mt_n", "clean10_2rater",
                "all35_fl_bias", "all35_pa_bias", "all35_mt_bias"]
        log(T[cols].head(60).round(4).to_string(index=False))
        log(s["gate"].round(4).to_string(index=False))
        log(json.dumps(s["summary"], indent=1))
        summary["static"] = s["summary"]
    if "gtmasks" in secs:
        log("== static benchmark with GT masks (in-train subset)")
        g = gtmask_section(args, log)
        g.to_csv(args.out / "static_gtmask_table.csv", index=False)
        log(g.round(4).to_string(index=False))
    if "train142" in secs:
        log("== 142 dual-GT training images")
        t = train142_section(args, log)
        log(json.dumps(t, indent=1, default=str))
        summary["train142"] = t
    if "dynamic" in secs:
        log("== dynamic benchmark")
        d = dynamic_section(args, log)
        log(json.dumps(d, indent=1, default=str))
        summary["dynamic"] = d
    if "test" in secs:
        log("== test set (maskbank, unlabeled)")
        t = test_section(args, log)
        log(pd.DataFrame(t["groups"]).to_string(index=False))
        log(json.dumps(t["flag_counts"], indent=1))
        summary["test"] = t
    (args.out / f"summary{args.tag}.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
