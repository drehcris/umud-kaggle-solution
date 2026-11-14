"""umud-run: the config-driven UMUD pipeline that rebuilds the final submissions end to end.

    umud-run --config configs/final_A.yaml --stage build        # Final A = KG0 (the best public file, byte-for-byte)
    umud-run --config configs/final_B.yaml --stage build        # Final B = Final A without the LB-only MT constant
    umud-run --config configs/K25b.yaml    --stage build        # the previous best file (Final A before the G0 head)
    umud-run --config configs/C18.yaml     --stage build        # the C18 base file
    umud-run --config configs/final_A.yaml --stage all          # layout -> maps -> measure (verify) -> build
    umud-run --config X.yaml --stage measure --full --datasets test   # re-measure every test image from the maps
    umud-run --config X.yaml --stage build --set rater_style.mode=frozen   # one-off override (YAML value)

Stages
------
layout   Test-set layout from the images: scanner family, size, px/cm, viewport (umud.geometry) and the video
         clips found from image content by consecutive-frame correlation (umud.clips). No file-name rules.
maps     Map inference is GPU work on Kaggle. Writes one ready-to-push kernel folder per map kernel (code from
         kaggle/, kernel-metadata.json, RUN.md with the push / status / output commands and the checkpoint sources)
         and checks the sha256 of the local map files the measure stage reads. Never pushes, never downloads.
measure  Per-image geometry from the maps, one image at a time. Default: verify the per-image caches (sha256) and
         re-measure one image per scanner group, comparing every numeric column with the cache. --full re-measures
         whole datasets (~1.1 s / image).
build    The submission from the per-image caches: PA / FL heads -> shrinkage (scanner x muscle centres, confidence
         PA rule, empirical-Bayes FL) -> Telemed clip medians -> rater-style offsets (re-estimated from the OSF xlsx
         by the pre-registered rule) -> the 4-decimal stage file -> post rules (native-geometry transform with
         factors confirmed by the image detectors, per-scanner MT constant, Telemed PA coupling pull, Telemed G0 PA
         head), each rounded to 4 decimals inside the bounds -> CSV; a config with `derive` (Final B) is then
         derived from another config's file (Final A without its MT constant, coupling term recomputed on the own
         MT). Every choice and constant is read from the config.

Exactness. The scored files were assembled in steps that each wrote a CSV (pandas float repr, CRLF) and re-read it
(pandas default parser). The build stage performs the same steps in memory (``to_text`` / ``from_text``), including
the per-image table round trip that the unshrunk Telemed FL went through, so Final A (KG0), Final B (B'), K25b, B1
and C18 are rebuilt byte-for-byte (tests/test_final_package.py).

Laptop rules: one process, no multiprocessing, no torch, one image / map at a time, < 1.5 GB RSS.
GPL-3.0-only (see LICENSE). External data and models: README.md and THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import inspect
import io
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

SCHEMA = "umud-config/2"
STAGES = ("layout", "maps", "measure", "build", "all")
C3 = ("pa_deg", "fl_mm", "mt_mm")
TAU = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
COMP = {"pa": "pa_deg", "fl": "fl_mm", "mt": "mt_mm"}
SHORT = {v: k for k, v in COMP.items()}
REQUIRED_BLOCKS = ("paths", "inputs", "groups", "layout", "maps", "measure", "geometry_space", "pa_head", "fl_rules",
                   "mt_edge", "shrink", "rater_style", "lb_constants", "output")
DOC_KEYS = {"rationale", "evidence", "note", "notes", "options", "description", "definition", "_resolved", "risk",
            "ledger", "status_note", "prior_note", "how", "declaration", "declarations", "prediction", "use"}
REPO_DEFAULT = Path(__file__).resolve().parents[2]


def rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6       # bytes on macOS, kB on Linux


def log(msg: str) -> None:
    print(f"[umud-run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =================================================================================== config
def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _set_dotted(d: dict, key: str, value) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def get_dotted(d: dict, key: str):
    for p in key.split("."):
        d = d[p]
    return d


def _expand(obj, env: dict):
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, list):
        return [_expand(v, env) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand(v, env) for k, v in obj.items()}
    return obj


REPLACE_KEYS = ("reference", "prediction", "declarations")   # replaced whole by an extending config, never merged


def _raw_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    if raw.get("extends"):
        base = _raw_config((path.parent / raw["extends"]).resolve())
        base.pop("extends", None)
        for k in REPLACE_KEYS:
            if k in raw:
                base.pop(k, None)
        raw = _deep_merge(base, {k: v for k, v in raw.items() if k != "extends"})
    return raw


def load_config(path: str | Path, overrides: dict | None = None, sets: list[str] | None = None) -> dict:
    """YAML config -> dict. `extends: other.yaml` deep-merges this file over `other` (lists and the blocks
    reference / prediction / declarations are replaced).
    ${repo} expands to $UMUD_REPO or to the repository that contains this module; `overrides` (dict) and `sets`
    (["a.b=1", ...], YAML-parsed values) are applied last."""
    path = Path(path).resolve()
    raw = _raw_config(path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    for s in sets or []:
        k, v = s.split("=", 1)
        _set_dotted(raw, k.strip(), yaml.safe_load(v))
    env = {"repo": os.environ.get("UMUD_REPO", str(REPO_DEFAULT)), "name": str(raw.get("name", path.stem))}
    cfg = _expand(raw, env)
    cfg.setdefault("paths", {})["repo"] = env["repo"]
    if cfg.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA!r}, got {cfg.get('schema')!r}")
    miss = [b for b in REQUIRED_BLOCKS if b not in cfg]
    if miss:
        raise ValueError(f"config {path.name} misses blocks {miss}")
    cfg["_resolved"] = dict(file=str(path), sha256=sha256_text(json.dumps(strip_docs(cfg), sort_keys=True,
                                                                            default=str)),
                            overrides=copy.deepcopy(overrides or {}), sets=list(sets or []))
    return cfg


DERIVE_FREE = ("name", "reference", "lb_constants", "derive")   # top-level keys a derived config may change


def config_core(cfg: dict) -> dict:
    """The config without documentation, names, reference, registry, `derive`, output file name, output directory and
    the per-scanner MT constants: what a `derive` config must share with the config it is derived from."""
    c = strip_docs(cfg)
    for k in DERIVE_FREE:
        c.pop(k, None)
    c.get("output", {}).pop("file_name", None)
    c.get("paths", {}).pop("out_dir", None)
    c.get("mt_edge", {}).pop("constant_mm", None)
    return c


def strip_docs(obj):
    """The config without documentation-only keys (rationale, evidence, note, ...): what is hashed."""
    if isinstance(obj, dict):
        return {k: strip_docs(v) for k, v in obj.items() if k not in DOC_KEYS}
    if isinstance(obj, list):
        return [strip_docs(v) for v in obj]
    return obj


def sha256_file(p: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def check_hashes(entries: list[dict], strict: bool = True) -> list[dict]:
    """[{path, sha256}] -> rows with ok flags; raises on a mismatch when strict."""
    rows = []
    for e in entries:
        p = Path(e["path"])
        ok_exist = p.exists()
        got = sha256_file(p) if ok_exist else None
        rows.append(dict(path=str(p), exists=ok_exist, expected=e.get("sha256"), got=got,
                         ok=ok_exist and (e.get("sha256") in (None, got))))
    bad = [r for r in rows if not r["ok"]]
    if bad and strict:
        raise RuntimeError("pinned files missing or changed:\n" + "\n".join(
            f"  {r['path']}: expected {r['expected']}, got {r['got']}" for r in bad))
    return rows


# =================================================================================== repository binding
def bind(cfg: dict) -> SimpleNamespace:
    """Import the repository modules and point every path / module constant they read at the config."""
    P = cfg["paths"]
    repo = Path(P["repo"])
    for sub in ("src", "scripts"):
        s = str(repo / sub)
        if s not in sys.path:
            sys.path.insert(0, s)
    import build_candidates as bc            # noqa: E402
    import build_candidates_v3 as b3         # noqa: E402
    import build_candidates_v4 as b4         # noqa: E402
    import eval_shrinkage as es              # noqa: E402
    import rater_calibration as rc           # noqa: E402
    from umud import muscle_label as ml      # noqa: E402
    from umud import shrinkage as sh         # noqa: E402
    R = SimpleNamespace(bc=bc, b3=b3, b4=b4, es=es, rc=rc, ml=ml, sh=sh, repo=repo)
    bc.STATIC = Path(P["static_dir"])
    bc.DYN = Path(P["dynamic_dir"])
    bc.TEST_DIR = Path(P["test_images"])
    bc.BENCHDIR = Path(P["vl_labels"]).parent
    bc.TEST_LAYOUT = Path(P["test_layout"])
    b3.CACHE = Path(P["v3_cache"])
    b3.V4_CACHE = Path(P["v4_cache"])
    b3._MEMO.clear()
    b3.LAY = None
    C = cfg["shrink"]["constants"]["test"]
    sh.POP_SD = {k: float(v) for k, v in C["pop_sd"].items()}
    sh.TAU_FLOOR = {k: float(v) for k, v in C["tau_floor"].items()}
    sh.K0 = float(C["k0"])
    sh.S_LOGK = float(C["s_logk"])
    b3._V4C.clear()
    check_pinned_defaults(cfg, R)
    return R


def check_pinned_defaults(cfg: dict, R: SimpleNamespace) -> None:
    """Implementation constants that live as function defaults / module constants in the repo code: the config
    records them; any drift between config and code is an error (change the config AND the code together)."""
    pin = cfg["shrink"]["pinned_code_defaults"]
    for fn, key in ((R.sh.fit_group_prior, "fit_group_prior"), (R.sh.fit_error_model, "fit_error_model")):
        sig = inspect.signature(fn)
        for k, v in pin[key].items():
            if sig.parameters[k].default != v:
                raise RuntimeError(f"code drift: shrinkage.{key}({k}={sig.parameters[k].default!r}) != config {v!r}")
    if list(R.sh.DEFAULT_FEATURES) != list(pin["error_model_features"]):
        raise RuntimeError("code drift: shrinkage.DEFAULT_FEATURES differs from config")
    if {k: list(v) for k, v in R.sh.DEFAULT_EXCLUDE.items()} != pin["error_model_exclude"]:
        raise RuntimeError("code drift: shrinkage.DEFAULT_EXCLUDE differs from config")
    if dict(R.sh.DEFAULT_NEFF) != pin["error_model_n_eff"]:
        raise RuntimeError("code drift: shrinkage.DEFAULT_NEFF differs from config")
    g = cfg["fl_rules"]["head"]["constructed"]["grid"]
    grid = np.round(np.arange(g["start"], g["stop"] + 1e-4, g["step"]), 1)
    if not np.array_equal(grid, R.b3.PA_GRID):
        raise RuntimeError("code drift: constructed-FL PA grid of the cache differs from config")
    if tuple(cfg["shrink"]["cells"]["muscles"]) != tuple(R.ml.MUSCLES):
        raise RuntimeError("code drift: muscle_label.MUSCLES differs from config")


# =================================================================================== small helpers
def _col(d: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(d[c], errors="coerce").astype(float) if c in d else pd.Series(np.nan, index=d.index)


def nanmean_w(cols: list, w) -> pd.Series:
    """Weighted mean over the finite entries of each row (weights renormalised per row)."""
    X = pd.concat(cols, axis=1).to_numpy(float)
    W = np.asarray(w, float)[None, :] * np.isfinite(X)
    s = W.sum(1)
    v = (np.nan_to_num(X) * W).sum(1) / np.where(s > 0, s, np.nan)
    return pd.Series(v, index=cols[0].index)


def blend(raw: pd.Series, shrunk: pd.Series, lam: pd.Series) -> pd.Series:
    """raw + lam (shrunk - raw) where raw is finite (else shrunk); lam == 1 returns shrunk exactly."""
    raw = pd.to_numeric(raw, errors="coerce").astype(float)
    shrunk = pd.to_numeric(shrunk, errors="coerce").astype(float)
    lam = pd.Series(lam, index=raw.index).astype(float) if not isinstance(lam, pd.Series) else lam.astype(float)
    v = (raw + lam * (shrunk - raw)).where(raw.notna(), shrunk)
    return v.where(lam != 1.0, shrunk)


def group_map(groups: pd.Series, spec: dict, default_key: str = "default") -> pd.Series:
    return groups.astype(str).map(lambda g: spec.get(g, spec.get(default_key))).astype(float)


def to_text(df: pd.DataFrame, ids: list[str]) -> str:
    """The submission CSV text exactly as every scored file was written (pandas float repr, CRLF)."""
    return df.loc[ids, list(C3)].rename_axis("image_id").reset_index().to_csv(index=False, lineterminator="\r\n")


def from_text(text: str) -> pd.DataFrame:
    """A submission CSV text read back exactly as the scored files were read (pandas default float parser)."""
    return pd.read_csv(io.StringIO(text)).set_index("image_id")[list(C3)].astype(float)


def csv_roundtrip(s: pd.Series) -> pd.Series:
    """A per-image column written to a CSV table and read back (the per-image table of the C12/C18 build)."""
    f = s.rename("v").to_frame()
    f.index.name = "image_id"
    return pd.read_csv(io.StringIO(f.to_csv())).set_index("image_id")["v"].reindex(s.index).rename(s.name)


def read_sub(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("image_id")


def official_distance(a: pd.DataFrame, b: pd.DataFrame) -> float:
    b = b.reindex(a.index)
    return float(np.mean([np.mean(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))) / TAU[c] for c in C3]))


def coupling_pa(mt: pd.Series, fl: pd.Series, k: float, lo: float = 0.0) -> pd.Series:
    """The coupling-implied PA asin(k MT / FL) in degrees (expert coupling FL sin(PA) = k MT)."""
    return pd.Series(np.degrees(np.arcsin(np.clip(k * mt / fl, lo, 1))), index=mt.index)


def g0_head(fused: pd.Series, ins: pd.Series, strip: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The G0 PA head. gate = (ins - fused > 0): the insertion-strip angle is steeper than the fused head (a sign
    test, no threshold; a missing insertion angle leaves the gate off). Where the gate is on, the head is the
    deep-strip field angle against the LOCAL deep tangent (`strip`; missing -> the fused head), else the fused head.
    Returns (head, gate)."""
    gate = (ins - fused) > 0
    return strip.fillna(fused).where(gate, fused), gate


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) \
            and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)
    return a == b


# =================================================================================== assembler
class Assembler:
    """Per-image caches -> submission, every choice read from the config. The shrinkage layer mirrors (and is
    tested against) scripts/build_candidates_v4.py / _v7.py, the reference implementation of the C12/C18 recipe."""

    def __init__(self, cfg: dict, R: SimpleNamespace):
        self.cfg, self.R = cfg, R
        self.groups_all = tuple(cfg["groups"]["all"])
        self.telemed = tuple(cfg["groups"]["telemed"])
        self.non_telemed = tuple(g for g in self.groups_all if g not in self.telemed)
        self._lay = None
        self._model = None
        self.factor_detail: dict = {}

    # ------------------------------------------------------------------ layout / groups
    def expand_groups(self, gs) -> tuple:
        out = []
        for g in gs or []:
            out += list(self.telemed if g == "telemed" else self.non_telemed if g == "non_telemed" else
                        self.groups_all if g == "all" else [g])
        return tuple(out)

    def group_spec(self, spec: dict) -> dict:
        """{default, telemed, non_telemed, <group>} -> {default, <group>: value} (explicit groups win)."""
        out = {"default": spec.get("default")}
        for alias in ("all", "non_telemed", "telemed"):
            if alias in spec:
                for g in self.expand_groups([alias]):
                    out[g] = spec[alias]
        for k, v in spec.items():
            if k in self.groups_all:
                out[k] = v
        return out

    def layout(self) -> pd.DataFrame:
        """Scanner group and clip per test image, from the images (umud.layout; built on first use)."""
        if self._lay is None:
            from umud import layout as lm
            L = self.cfg["layout"]
            path = Path(self.cfg["paths"]["test_layout"])
            if not path.exists():
                lay, _ = lm.build_layout(self.cfg["paths"]["test_images"], threshold=float(L["clips"]["threshold"]),
                                         min_frames=int(L["clips"]["min_frames"]))
                lm.write_layout(lay, path)
            lay = lm.read_layout(path)
            unknown = set(lay["group"]) - set(self.groups_all)
            if unknown:
                raise ValueError(f"layout groups not in config groups.all: {unknown}")
            self._lay = lay
        return self._lay

    def test_ids(self) -> list[str]:
        d = Path(self.cfg["paths"]["test_images"])
        if d.exists():
            return sorted(p.name for p in d.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png"})
        return sorted(self.layout().index)

    # ------------------------------------------------------------------ per-image heads
    def pa_head(self, d: pd.DataFrame, ref: str) -> pd.Series:
        h = self.cfg["pa_head"]
        cols = h["columns"]["horizontal" if ref == "h" else "deep"]
        return nanmean_w([_col(d, c) for c in cols], h["weights"])

    def cons_at(self, d: pd.DataFrame, pa: pd.Series) -> pd.Series:
        """Constructed straight-fascicle FL at a per-row PA, linearly interpolated on the cached PA grid."""
        g = self.cfg["fl_rules"]["head"]["constructed"]["grid"]
        grid = np.round(np.arange(g["start"], g["stop"] + 1e-4, g["step"]), 1)
        cols = [f"{g['prefix']}{int(round(10 * p)):03d}" for p in grid]
        G = d[cols].to_numpy(float)
        p = pa.reindex(d.index).to_numpy(float)
        t = (np.clip(np.nan_to_num(p, nan=grid[0]), grid[0], grid[-1]) - grid[0]) / g["step"]
        i0 = np.clip(np.floor(t).astype(int), 0, len(grid) - 2)
        fr = t - i0
        r = np.arange(len(d))
        a, b = G[r, i0], G[r, i0 + 1]
        v = np.where(np.isfinite(a) & np.isfinite(b), a * (1 - fr) + b * fr, np.where(np.isfinite(a), a, b))
        v[~np.isfinite(p)] = np.nan
        return pd.Series(v, index=d.index)

    def fl_head(self, d: pd.DataFrame, pa: pd.Series) -> pd.Series:
        h = self.cfg["fl_rules"]["head"]
        cols = [_col(d, c) for c in h["columns"]] + [self.cons_at(d, pa)]
        return nanmean_w(cols, list(h["weights"]) + [h["constructed"]["weight"]])

    def est_frame(self, d: pd.DataFrame, ds: str) -> pd.DataFrame:
        """Per-image estimates: PA (deep ref; horizontal on the dynamic benchmark), FL head at the raw PA
        (rebuilt later at the post-shrinkage PA), MT; failure rule MT < threshold -> NaN."""
        p_deep = self.pa_head(d, "deep")
        out = d.copy()
        out["pa_deg"] = self.pa_head(d, "h") if ds == "dynamic" else p_deep
        out["fl_mm"] = self.fl_head(d, p_deep)
        out["mt_mm"] = _col(d, self.cfg["mt_edge"]["source_column"])
        out.loc[out["mt_mm"] < self.cfg["pa_head"]["mt_fail_below_mm"], list(C3)] = np.nan
        return out

    # ------------------------------------------------------------------ shrinkage pieces
    def conf_weight(self, conf: pd.Series) -> pd.Series:
        w = pd.to_numeric(conf, errors="coerce").fillna(0.0).clip(0, 1)
        pw = float(self.cfg["shrink"]["pa"]["weight_power"])
        return w if pw == 1.0 else w ** pw

    def centres(self, x: pd.Series, grp: pd.Series, cell: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce").astype(float)
        c = x.groupby(grp.reindex(x.index).astype(str)).transform("median")
        cell = cell.reindex(x.index).fillna("")
        has = cell != ""
        if has.any():
            c[has] = x[has].groupby(cell[has]).transform("median")
        return c

    def conf_rule(self, x: pd.Series, conf: pd.Series, grp: pd.Series, cell: pd.Series) -> pd.Series:
        w = self.conf_weight(conf)
        med = self.centres(x, grp, cell)
        return (w * x + (1 - w) * med).where(x.notna(), med)

    def post_group(self, B: pd.DataFrame, pa_cell: pd.Series) -> pd.DataFrame:
        """pa_deep_raw / pa_deep_c (confidence-shrunk PA, deep reference) and FL rebuilt at pa_deep_c."""
        B = B.copy()
        p = self.pa_head(B, "deep")
        p[_col(B, "mt_mm").isna()] = np.nan
        B["pa_deep_raw"] = p
        B["pa_deep_c"] = self.conf_rule(p, _col(B, "confidence"), B["group"].astype(str), pa_cell)
        if self.cfg["fl_rules"]["head"]["constructed"]["at"] == "post_shrink_pa":
            fl = self.fl_head(B, B["pa_deep_c"])
            fl[_col(B, "mt_mm").isna()] = np.nan
            B["fl_mm"] = fl
        return B

    def eb_kwargs(self) -> dict:
        f = self.cfg["shrink"]["fl"]
        return dict(method=f["method"], likelihood=f["likelihood"], cols=("fl",), prior_kw=dict(mu_mode=f["prior_mu"]))

    def eb_fl(self, D: pd.DataFrame, model, cell: pd.Series) -> pd.DataFrame:
        """EB FL shrinkage toward the scanner-group prior (unit = clip for Telemed clips, else the image), then
        for images with a scanner x muscle cell the posterior with the cell median as centre."""
        sh = self.R.sh
        f = self.cfg["shrink"]["fl"]
        P = sh.shrink(D, model, group_col="group", clip_col="clip", **self.eb_kwargs())
        cell = cell.reindex(D.index).fillna("")
        has = cell != ""
        P["cell"] = cell
        if not has.any():
            return P
        clip = pd.to_numeric(D.get("clip", pd.Series(-1, index=D.index)), errors="coerce").fillna(-1)
        if (clip[has] >= 0).any():
            raise ValueError("muscle cells must not contain clip units")
        x = _col(D, "fl_mm")
        s = P["sigma_fl"].astype(float)
        grp = D["group"].astype(str)
        pri = dict(P.attrs["priors"])
        for g in grp[has].unique():
            idx = has & (grp == g)
            cm = x[idx].groupby(cell[idx]).transform("median")
            if f["cell_centre"] == "refit":
                pr = sh.fit_group_prior((x[idx] - cm).to_numpy(float), s[idx].to_numpy(float), "fl",
                                        method=f["method"], mu_mode=f["prior_mu"])
            elif f["cell_centre"] == "centre_only":
                q = pri[f"{g}|fl"]
                pr = sh.Prior(mu=0.0, tau=q["tau"], c=q["c"], n=q["n_units"], method="pooled")
            else:
                raise ValueError(f["cell_centre"])
            for i in idx[idx].index:
                v = sh._post_1d(float(x[i]), float(s[i]), sh.Prior(mu=float(cm[i]), tau=pr.tau, c=pr.c),
                                f["likelihood"])
                P.at[i, "fl_eb"] = v
                P.at[i, "mu_fl"] = cm[i]
        return P

    def apply_shrink(self, D: pd.DataFrame, model, pa_cell: pd.Series, fl_cell: pd.Series) -> pd.DataFrame:
        if self.cfg["shrink"]["fl"]["rule"] != "eb" or self.cfg["shrink"]["pa"]["rule"] != "confidence":
            raise NotImplementedError("only pa.rule=confidence with fl.rule=eb (the C12/C18 family) is wired")
        pa_cell = pa_cell.reindex(D.index).fillna("")
        fl_cell = fl_cell.reindex(D.index).fillna("")
        conf = _col(D, "confidence")
        grp = D["group"].astype(str)
        P = self.eb_fl(D, model, fl_cell)
        P["pa_eb"] = self.conf_rule(_col(D, "pa_deg"), conf, grp, pa_cell)
        P["pa_cell"] = pa_cell
        for c in COMP:     # NaN fill: group median, then the global median (eval_shrinkage.fill_group)
            k = f"{c}_eb"
            P[k] = P[k].fillna(P.groupby("group")[k].transform("median")).fillna(P[k].median())
        return P

    def error_model(self, Btr: pd.DataFrame):
        em = self.cfg["shrink"]["error_model"]
        if em["drop_static_in_training"]:
            iid = Btr["image_id"] if "image_id" in Btr else pd.Series(Btr.index.astype(str), index=Btr.index)
            clean = list(self.cfg["benchmarks"]["static_clean10"])
            keep = (Btr["dataset"].astype(str) != "static") | iid.isin(clean)
            if (Btr["dataset"] == "static").any() and int((keep & (Btr["dataset"] == "static")).sum()) != len(clean):
                raise RuntimeError("clean static rows not found")
            Btr = Btr[keep]
        feats = None if em["features"] == "default" else list(em["features"])
        return self.R.sh.fit_error_model(Btr, features=feats, ridge=float(em["ridge"]), balance=em["balance"])

    # ------------------------------------------------------------------ frames
    def load_cache(self, ds: str) -> pd.DataFrame:
        d = self.R.b3.load_v3(ds, self.cfg["measure"]["apo_source"])
        if d is None:
            raise FileNotFoundError(f"per-image cache for {ds} missing or incomplete in {self.cfg['paths']['v3_cache']}")
        return d

    def bench_frame(self, split: str) -> pd.DataFrame:
        """Benchmark rows (estimates + truths + dataset + group), b3.bench_frame + v4_post_group."""
        bc, es = self.R.bc, self.R.es
        parts = []
        d = self.est_frame(self.load_cache("static"), "static")
        g = bc.static_gts()["mean7"].reindex(d.index)
        d["pa_true"], d["fl_true"], d["mt_true"] = g.pa_deg, g.fl_mm, g.mt_mm
        d["dataset"], d["group"] = "static", es.static_group(d.index)
        parts.append(d)
        d = self.est_frame(self.load_cache("vl"), "vl")
        v = bc.vl_gt().reindex(d.index)
        d["pa_true"], d["fl_true"], d["mt_true"] = v.pa_deg, v.fl_mm, v.mt_mm
        d["dataset"] = np.where(d.index.str.startswith("young"), "vl_young", "vl_old") if split == "lodo4" else "vl"
        d["group"] = d["dataset"]
        parts.append(d)
        d = self.est_frame(self.load_cache("dynamic"), "dynamic")
        dg = bc.dynamic_gt().reindex(d.index)
        d["pa_true"], d["fl_true"], d["mt_true"] = dg.pa_deg, dg.fl_mm, np.nan
        d["dataset"], d["group"] = "dynamic", "dynamic"
        d = d[d["pa_true"].notna() | d["fl_true"].notna()]
        parts.append(d)
        B = pd.concat(parts)
        B["image_id"] = B.index.astype(str)
        B.index = B["dataset"].str.replace(r"_(young|old)$", "", regex=True) + ":" + B.index.astype(str)
        B["clip"] = -1
        return self.post_group(B, pd.Series("", index=B.index, dtype=object))

    def test_frame(self) -> pd.DataFrame:
        T = self.est_frame(self.load_cache("test"), "test")
        lay = self.layout()
        T["group"] = lay.loc[T.index, "group"]
        T["clip"] = lay.loc[T.index, "clip"].astype(int)
        return T

    def test_cells(self, T: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """(pa_cell, fl_cell): '<scanner>|<muscle>' for trusted burned-in muscle labels in cells with >= min_n
        images; the FL cell is dropped when the cell's median raw FL/FOV exceeds the regime gate."""
        c = self.cfg["shrink"]["cells"]
        empty = pd.Series("", index=T.index, dtype=object)
        if not c["enabled"]:
            return empty, empty.copy()
        t = self.R.ml.load_test_muscle(self.cfg["paths"]["muscle_table"]).reindex(T.index)
        if t["muscle"].isna().all():
            return empty, empty.copy()
        m, _ = self.R.ml._trusted(t.fillna(""), float(c["min_conf"]), bool(c["use_inferred"]),
                                  drop_ambiguous=bool(c["drop_ambiguous"]))
        key = T["group"].astype(str) + "|" + m
        n = key.map(key[m != ""].value_counts()).fillna(0)
        key = key.where((m != "") & (n >= int(c["min_n"])), "").astype(object)
        ratio = _col(T, "fl_mm") / self.R.sh._fov_mm(T)          # raw-PA FL head / FOV
        med = ratio[key != ""].groupby(key[key != ""]).median()
        bad = set(med.index[med > float(c["fl_fov_gate"])])
        gated = key.where(~key.isin(bad), "").astype(object)
        fl_cell = gated if c["fl_centres"] else empty.copy()
        pa_cell = key if c["gate_applies_to"] == "fl" else gated
        return pa_cell, fl_cell

    def test_state(self) -> dict:
        if not hasattr(self, "_ts"):
            T = self.test_frame()
            pa_cell, fl_cell = self.test_cells(T)
            T = self.post_group(T, pa_cell)
            if self._model is None:
                self._model = self.error_model(self.bench_frame("lodo4"))
            P = self.apply_shrink(T, self._model, pa_cell, fl_cell)
            self._ts = dict(P=P, pa_cell=pa_cell, fl_cell=fl_cell)
        return self._ts

    def pa_specific_conf(self, P: pd.DataFrame) -> pd.Series:
        pen_tab = self.cfg["shrink"]["pa"]["pa_specific_confidence"]["fl_only_flag_penalties"]
        conf = pd.to_numeric(P["confidence"], errors="coerce").clip(1e-6, 1.0)
        fl = P["flags"].fillna("").astype(str).str.split(";")
        fl_pen = fl.apply(lambda xs: sum(pen_tab.get(x, 0.0) for x in xs))
        return np.exp(-(-np.log(conf) - fl_pen).clip(lower=0.0)).where(conf.notna())

    # ------------------------------------------------------------------ shrunk + styled submission (unrounded)
    def shrunk_submission(self) -> pd.DataFrame:
        """Per-group shrinkage strengths, FL gate, NaN fill, clip medians, bounds (before rater style)."""
        ts = self.test_state()
        P = ts["P"]
        g = P["group"].astype(str)
        fr = self.cfg["fl_rules"]
        sp = self.cfg["shrink"]["pa"]
        lam_pa = group_map(g, self.group_spec(sp["lambda"]))
        lam_fl = group_map(g, self.group_spec(fr["shrink_lambda"]))
        fl_raw = pd.to_numeric(P["fl_mm"], errors="coerce")
        if self.cfg["output"]["per_image_table_roundtrip"]:
            fl_raw = csv_roundtrip(fl_raw)       # the unshrunk FL went through the per-image table of the build
        fl_raw_unit = fl_raw.groupby(P["eb_unit"]).transform("median")
        gate = fr["eb_gate_fl_fov"]
        if gate["enabled"]:
            ratio = fl_raw_unit / self.R.sh._fov_mm(P)
            rows = g.isin(self.expand_groups(gate["groups"])) & (ratio <= float(gate["max_ratio"]))
            lam_fl = lam_fl.where(~rows, 0.0)
        pa_eb = P["pa_eb"]
        psc = sp["pa_specific_confidence"]
        if psc["groups"]:
            nt = g.isin(self.expand_groups(psc["groups"]))
            ov = self.conf_rule(P["pa_deg"], self.pa_specific_conf(P), g, ts["pa_cell"]).where(nt)
            pa_eb = ov.reindex(P.index).fillna(P["pa_eb"])
        self._lam_fl = lam_fl
        Q = pd.DataFrame({"pa_deg": blend(P["pa_deg"], pa_eb, lam_pa),
                          "fl_mm": blend(fl_raw_unit, P["fl_eb"], lam_fl),
                          "mt_mm": P["mt_eb"]}, index=P.index)
        return self.postprocess(Q, margin=0.0)

    def postprocess(self, Q: pd.DataFrame, margin: float) -> pd.DataFrame:
        """NaN fill (group median, then global), median within the detected clips, clip to the bounds."""
        lay = self.layout()
        out = Q.copy()
        grp = lay.loc[out.index, "group"]
        for c in C3:
            out[c] = out[c].fillna(out.groupby(grp)[c].transform("median"))
            out[c] = out[c].fillna(out[c].median())
        if self.cfg["shrink"]["clip_median"]:
            out = self.clip_median(out)
        b = self.cfg["output"]["bounds"]
        for c in C3:
            out[c] = out[c].clip(float(b[c][0]) + margin, float(b[c][1]) - margin)
        return out

    def clip_median(self, X: pd.DataFrame) -> pd.DataFrame:
        clip = self.layout().loc[X.index, "clip"]
        inside = clip >= 0
        X = X.copy()
        for c in C3:
            X.loc[inside, c] = X.loc[inside, c].groupby(clip[inside]).transform("median")
        return X

    def offsets(self) -> dict:
        if not hasattr(self, "_off"):
            rs = self.cfg["rater_style"]
            if rs["mode"] == "frozen":
                self._off = dict(pa_add=float(rs["pa_add"]), mt_add=float(rs["mt_add"]), fl_mult=float(rs["fl_mult"]),
                                 fl_mult_nonTel=float(rs["fl_mult_nonTel"]))
            elif rs["mode"] == "reestimate":
                est = self.estimate_rater_style()
                self._off = dict(est["applied"], fl_mult_nonTel=float(rs["fl_mult_nonTel"]))
                self.rater_estimate = est
                for k in ("pa_add", "mt_add", "fl_mult"):
                    if not math.isclose(self._off[k], float(rs[k]), rel_tol=0, abs_tol=float(rs["reestimate_tol"])):
                        raise ValueError(f"rater_style re-estimated {k} = {self._off[k]!r} differs from the declared "
                                         f"{rs[k]!r} by more than {rs['reestimate_tol']}")
            else:
                raise ValueError(rs["mode"])
        return self._off

    def styled_submission(self) -> pd.DataFrame:
        sub = self.shrunk_submission()
        off = self.offsets()
        g = self.layout().loc[sub.index, "group"].astype(str)
        rs = self.cfg["rater_style"]
        pa_add = g.map(lambda x: self.group_spec(rs.get("pa_add_by_group") or {}).get(x, None)).astype(float)
        pa_add = pa_add.fillna(off["pa_add"])
        sub = sub.copy()
        sub["pa_deg"] = sub["pa_deg"] + pa_add
        nt = g.isin(self.non_telemed)
        sub["fl_mm"] = sub["fl_mm"] * off["fl_mult"]
        if off["fl_mult_nonTel"] != 1.0:
            sub.loc[nt, "fl_mm"] = sub.loc[nt, "fl_mm"] * off["fl_mult_nonTel"]
        sub["mt_mm"] = sub["mt_mm"] + off["mt_add"]
        b = self.cfg["output"]["bounds"]
        for c in C3:
            sub[c] = sub[c].clip(float(b[c][0]), float(b[c][1]))
        return sub

    # ------------------------------------------------------------------ the 4-decimal stage file
    def stage_text(self) -> str:
        """The base submission file (C18 format): 4 decimals, except the unshrunk FL cells (fl lambda 0), which
        are written at full precision (the per-clip raw FL was copied unrounded into the scored C18 file)."""
        sub = self.styled_submission()
        S = sub.round(int(self.cfg["output"]["decimals"]))
        if self.cfg["output"]["full_precision_unshrunk_fl"]:
            raw = (self._lam_fl.reindex(sub.index) == 0.0)
            S.loc[raw, "fl_mm"] = sub.loc[raw, "fl_mm"]
        return to_text(S, self.test_ids())

    # ------------------------------------------------------------------ post rules
    def _finish(self, X: pd.DataFrame, clip_median: bool = True, roundtrip: bool = True) -> pd.DataFrame:
        """A post-rule file: clip medians, bounds with the margin, 4 decimals (written and read back)."""
        if clip_median:
            X = self.clip_median(X)
        m = float(self.cfg["output"]["bound_margin"])
        b = self.cfg["output"]["bounds"]
        X = X.copy()
        for c in C3:
            X[c] = X[c].astype(float).clip(float(b[c][0]) + m, float(b[c][1]) - m)
        X = X.round(int(self.cfg["output"]["decimals"]))
        return from_text(to_text(X, self.test_ids())) if roundtrip else X

    def post_rules(self, S: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Rules on the stage file S, each on its own (group x component) cells, in this order:
        1. geometry_space: native-geometry transform of PA / FL for the scanners declared `physical`;
        2. mt_edge.constant_mm: per-scanner MT constant;
        3. pa_head.coupling_pull: Telemed PA pulled toward the coupling-implied PA asin(k MT / FL), computed on the
           result of 1-2 (so with the final MT and FL);
        4. pa_head.telemed_head: the Telemed PA cells re-derived from the G0 head with the PA recipe and the same
           coupling term (Assembler.telemed_head); it supersedes rule 3 on those cells.
        1 and 2 are computed from S; the cells they change are then copied into the state (exactly as the scored
        probe files were composed)."""
        ids = self.test_ids()
        lay = self.layout()
        g = lay.loc[S.index, "group"].astype(str)
        state = S.copy()
        cells: dict = {}
        steps = []

        def put(F: pd.DataFrame, m: pd.Series, comps: list[str], rule: str):
            for k in comps:
                state.loc[m, k] = F.loc[m, k]
            for grp in sorted(set(g[m])):
                for k in comps:
                    cells.setdefault(f"{grp}|{SHORT[k]}", []).append(rule)

        gs = self.cfg["geometry_space"]
        comps = [COMP[c] for c in gs["components"]]
        T = None
        for grp, s in gs["scanners"].items():
            if s["space"] == "image":
                continue
            if s["space"] != "physical" or gs["method"] != "posthoc":
                raise NotImplementedError(f"geometry_space {s['space']} / method {gs['method']}")
            c = self.geometry_factor(grp)
            if T is None:
                T = self.load_cache("test").reindex(S.index)
            delta = (_col(T, gs["delta_columns"][0]) - _col(T, gs["delta_columns"][1])).fillna(0.0)
            from umud.native_geometry import posthoc_transform
            pa_new, fl_new = posthoc_transform(S["pa_deg"], S["fl_mm"], delta, c)
            m = g == grp
            X = S.copy()
            if "pa_deg" in comps:
                X.loc[m, "pa_deg"] = pa_new[m]
            if "fl_mm" in comps:
                X.loc[m, "fl_mm"] = fl_new[m]
            put(self._finish(X), m, comps, f"native_geometry(c={c})")
            steps.append(dict(rule="native_geometry", group=grp, factor=c, components=[SHORT[k] for k in comps]))
        mtc = group_map(g, self.group_spec(self.cfg["mt_edge"]["constant_mm"]))
        for v in sorted(set(mtc[mtc != 0.0])):
            m = mtc == v
            X = S.copy()
            X.loc[m, "mt_mm"] += v
            put(self._finish(X), m, ["mt_mm"], f"mt_constant({v})")
            steps.append(dict(rule="mt_constant", groups=sorted(set(g[m])), value_mm=v))
        cp = self.cfg["pa_head"]["coupling_pull"]
        pre_pull = from_text(to_text(state, ids))      # rules 1-2 as written: the MT / FL the coupling terms see
        if float(cp["weight"]) != 0.0:
            base = pre_pull
            m = g.isin(self.expand_groups(cp["groups"]))
            pre = base["pa_deg"] - self.offsets()["pa_add"]
            cpl = coupling_pa(base["mt_mm"], base["fl_mm"], float(cp["k"]))
            new = base["pa_deg"] + float(cp["weight"]) * (cpl - pre)
            F = base.copy()
            F.loc[m, "pa_deg"] = self._finish(base.assign(pa_deg=new), clip_median=False,
                                                 roundtrip=False).loc[m, "pa_deg"]
            staged = from_text(to_text(F, ids))
            state = base.copy()
            put(staged, m, ["pa_deg"], f"coupling_pull(w={cp['weight']}, k={cp['k']})")
            steps.append(dict(rule="coupling_pull", groups=sorted(set(g[m])), weight=float(cp["weight"]),
                              k=float(cp["k"]), mean_change_deg={grp: round(float((staged.loc[g == grp, "pa_deg"]
                                                                                    - base.loc[g == grp, "pa_deg"]).mean()), 4)
                                                                 for grp in sorted(set(g[m]))}))
        th = self.cfg["pa_head"].get("telemed_head") or {}
        if th.get("enabled", False):
            m, new, st = self.telemed_head(pre_pull, state, th)
            full = state["pa_deg"].copy()
            full[m] = new[m]
            F = state.copy()
            F.loc[m, "pa_deg"] = self._finish(state.assign(pa_deg=full), clip_median=False,
                                              roundtrip=False).loc[m, "pa_deg"]
            staged = from_text(to_text(F, ids))
            before = state.copy()
            for grp in sorted(set(g[m])):          # the head rule carries the coupling term: it replaces rule 3 here
                cells[f"{grp}|pa"] = [r for r in cells.get(f"{grp}|pa", []) if not r.startswith("coupling_pull")]
            put(staged, m, ["pa_deg"], f"telemed_head({st['head']}, coupling={st['coupling']})")

            def mean_d(a: pd.DataFrame, b: pd.DataFrame) -> dict:
                return {grp: round(float((a.loc[g == grp, "pa_deg"] - b.loc[g == grp, "pa_deg"]).mean()), 4)
                        for grp in sorted(set(g[m]))}
            st.update(mean_change_vs_rules12_deg=mean_d(staged, pre_pull), mean_change_vs_rule3_deg=mean_d(staged, before),
                      rows_changed_vs_rule3=int(((staged["pa_deg"] - before["pa_deg"]).abs() > 0)[m].sum()))
            steps.append(st)
        overlap = {k: v for k, v in cells.items() if len(v) > 1}
        if overlap and not self.cfg["output"].get("allow_cell_overlap", False):
            raise ValueError(f"post rules overlap on cells {overlap}")
        return state, dict(cells=cells, steps=steps)

    def telemed_head(self, base: pd.DataFrame, state: pd.DataFrame, th: dict) -> tuple[pd.Series, pd.Series, dict]:
        """Post rule 4 (KG0): the Telemed PA from the G0 head (`g0_head`) on the per-image cache columns, through
        the PA recipe written out for these rows exactly as the scored KG0 file was built:
            w h + (1 - w) median_group(h)     per image (w = clip(confidence, 0, 1): the shrink.pa confidence rule;
                                              Telemed has no muscle cells, so the centre is the scanner-group median)
            -> median over the clip -> + the rater PA offset as written in the 4-decimal files (round(pa_add, 4))
            -> + w_c (asin(k MT / FL) - (PA - offset))   the rule-3 coupling term (weight / k of
                                              pa_head.coupling_pull) on the MT / FL of rules 1-2 (omitted with
                                              coupling: false)
        FL is not rebuilt at the new PA (the scored file kept K25b's FL). The written-out recipe is checked on the
        fused head (fusF) against the pipeline's own Telemed PA after rules 1-2 and after rule 3: both must agree
        within check_tol, else the build stops. Returns (rows, new unrounded PA on those rows, step info)."""
        if th.get("head", "G0") != "G0":
            raise NotImplementedError(f"telemed_head.head {th.get('head')!r} (only G0 is wired)")
        sp = self.cfg["shrink"]["pa"]
        if sp["rule"] != "confidence" or float(sp["weight_power"]) != 1.0:
            raise NotImplementedError("telemed_head needs shrink.pa.rule = confidence with weight_power 1")
        lay = self.layout()
        g = lay.loc[base.index, "group"].astype(str)
        m = g.isin(self.expand_groups(th["groups"]))
        ids = base.index[m]
        grp, clip = g[ids], lay.loc[ids, "clip"].astype(int)
        if (clip < 0).any():
            raise ValueError("telemed_head groups must consist of detected clips")
        T = self.load_cache("test").reindex(ids)
        fused = self.pa_head(T, "deep")                                  # fusF
        head, gate = g0_head(fused, _col(T, th["gate_column"]), _col(T, th["strip_column"]))
        conf = _col(T, "confidence")
        no_cell = pd.Series("", index=ids, dtype=object)
        off = round(float(self.offsets()["pa_add"]), int(self.cfg["output"]["decimals"]))
        cp = self.cfg["pa_head"]["coupling_pull"]
        wc, k = float(cp["weight"]), float(cp["k"])
        cpl = coupling_pa(base.loc[ids, "mt_mm"], base.loc[ids, "fl_mm"], k)
        coupling = bool(th.get("coupling", True))

        def recipe(h: pd.Series) -> pd.Series:
            return self.conf_rule(h, conf, grp, no_cell).groupby(clip).transform("median") + off

        def k25(p: pd.Series) -> pd.Series:
            return p + wc * (cpl - (p - off))

        chk = {"rules12": float((recipe(fused) - base.loc[ids, "pa_deg"]).abs().max())}
        if wc != 0.0:
            chk["rule3"] = float((k25(recipe(fused)) - state.loc[ids, "pa_deg"]).abs().max())
        tol = float(th["check_tol"])
        st = dict(rule="telemed_head", head="G0", groups=sorted(set(grp)), coupling=coupling, coupling_weight=wc, k=k,
                  offset_deg=off, gate=f"{th['gate_column']} - fusF > 0", strip=th["strip_column"], rows=int(m.sum()),
                  gate_on=int(gate.sum()), gate_on_by_group={x: int(gate[grp == x].sum()) for x in sorted(set(grp))},
                  recipe_check_max_abs=chk, check_tol=tol)
        if any(v > tol for v in chk.values()):
            raise RuntimeError(f"telemed_head: the written-out PA recipe does not reproduce the pipeline's Telemed PA "
                               f"on the fused head within {tol}: {chk}")
        new = k25(recipe(head)) if coupling else recipe(head)
        return m, new.reindex(base.index), st

    def derive(self, own_text: str) -> tuple[str, dict]:
        """`derive` (Final B'): this config's file derived from another config's file (`derive.from`, Final A) whose
        core settings are identical (config_core) except the per-scanner MT constants (Final A's J4):
            MT  <- this pipeline's own MT on the `derive.groups` rows;
            PA  <- PA_from + w (asin(k MT_own / FL) - asin(k MT_from / FL)), rounded to the output decimals
                   (the coupling term of rules 3-4 recomputed on the own MT, w / k of pa_head.coupling_pull);
            every other cell is copied from the `from` file.
        This is exactly how the scored Final B' file was made from KG0. It equals this config's own post rules
        (coupling on the own MT computed directly) up to the 4-decimal rounding of the intermediate file: the build
        checks that every other cell is identical and that the PA cells differ by at most derive.max_vs_direct."""
        dv = self.cfg["derive"]
        if dv.get("rule") != "recouple_own_mt":
            raise NotImplementedError(f"derive.rule {dv.get('rule')!r}")
        r = self.cfg["_resolved"]
        src = load_config(Path(r["file"]).parent / dv["from"], overrides=r.get("overrides") or None,
                          sets=r.get("sets") or None)
        a, b = config_core(src), config_core(self.cfg)
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise ValueError(f"derive.from {dv['from']} differs from this config outside the MT constants: {diff}")
        text_from, info_from = Assembler(src, self.R).submission()
        A, own = from_text(text_from), from_text(own_text)
        ids = self.test_ids()
        g = self.layout().loc[A.index, "group"].astype(str)
        m = g.isin(self.expand_groups(dv["groups"]))
        cp = self.cfg["pa_head"]["coupling_pull"]
        w, k = float(cp["weight"]), float(cp["k"])
        D = A.copy()
        D.loc[m, "mt_mm"] = own.loc[m, "mt_mm"]
        new = A["pa_deg"] + w * (coupling_pa(D["mt_mm"], A["fl_mm"], k, -1) - coupling_pa(A["mt_mm"], A["fl_mm"], k, -1))
        full = A["pa_deg"].copy()
        full[m] = new[m]
        D.loc[m, "pa_deg"] = self._finish(A.assign(pa_deg=full), clip_median=False, roundtrip=False).loc[m, "pa_deg"]
        D = from_text(to_text(D, ids))
        dd = (D - own.reindex(D.index)).abs()
        pa_cells = m.to_numpy()[:, None] & (np.array(C3) == "pa_deg")[None, :]
        other = float(np.nanmax(np.where(pa_cells, 0.0, dd.to_numpy())))
        vs_direct = float(dd.loc[m, "pa_deg"].max())
        tol = float(dv["max_vs_direct"])
        info = dict(rule=dv["rule"], source=dict(config=src["_resolved"]["file"], name=src.get("name"),
                                                 sha256=sha256_text(text_from), offsets=info_from.get("offsets")),
                    groups=sorted(set(g[m])), rows=int(m.sum()),
                    mean_change_vs_source={grp: {c: round(float((D.loc[g == grp, c] - A.loc[g == grp, c]).mean()), 4)
                                                 for c in ("pa_deg", "mt_mm")} for grp in sorted(set(g[m]))},
                    cells_changed_vs_source=sorted({f"{grp}|{SHORT[c]}" for c in C3 for grp in
                                                    set(g[(D[c] - A[c]).abs() > 0])}),
                    max_abs_vs_direct_pa=vs_direct, rows_differing_from_direct=int((dd.loc[m, "pa_deg"] > 0).sum()),
                    max_abs_vs_direct_other=other, max_vs_direct=tol)
        if other != 0.0 or vs_direct > tol + 1e-9:
            raise RuntimeError(f"derive: the derived file departs from this config's own post rules: other cells "
                               f"{other}, Telemed PA {vs_direct} (allowed {tol})")
        return to_text(D, ids), info

    def geometry_factor(self, grp: str) -> float:
        """The declared per-scanner factor; with factor_source detect_confirm, the image detector is run on every
        image of the group and must confirm it (|median - factor| <= tolerance), else the build stops."""
        gs = self.cfg["geometry_space"]
        s = gs["scanners"][grp]
        src = s.get("factor_source", "declared")
        c = float(s["factor"])
        if src == "declared":
            return c
        if src != "detect_confirm":
            raise ValueError(src)
        from umud.native_geometry import confirm_factor
        lay = self.layout()
        ids = sorted(lay.index[lay["group"] == grp])
        d = confirm_factor([Path(self.cfg["paths"]["test_images"]) / i for i in ids], s["detector"], c,
                           float(s["tolerance"]))
        self.factor_detail[grp] = d
        return c

    def submission(self) -> tuple[str, dict]:
        """(final CSV text, info)."""
        lbc = self.check_lb_constants()
        text = self.stage_text()
        S = from_text(text)
        rules_active = any(s["space"] != "image" for s in self.cfg["geometry_space"]["scanners"].values()) or \
            any(float(v or 0.0) != 0.0 for v in self.cfg["mt_edge"]["constant_mm"].values()) or \
            float(self.cfg["pa_head"]["coupling_pull"]["weight"]) != 0.0 or \
            bool((self.cfg["pa_head"].get("telemed_head") or {}).get("enabled", False))
        info = dict(lb_constants=lbc, offsets=self.offsets())
        if rules_active:
            state, pr = self.post_rules(S)
            text = to_text(state, self.test_ids())
            info.update(post_rules=pr)
        if (self.cfg.get("derive") or {}).get("enabled", False):
            text, info["derive"] = self.derive(text)
        info["geometry_factors"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_image"}
                                    for k, v in self.factor_detail.items()}
        info["geometry_factors_per_image"] = {k: v.get("per_image") for k, v in self.factor_detail.items()}
        return text, info

    # ------------------------------------------------------------------ lb constants registry
    def check_lb_constants(self) -> list[dict]:
        rows = []
        for e in self.cfg["lb_constants"]:
            targets = e["target"] if isinstance(e["target"], list) else [e["target"]]
            for t in targets:
                val = get_dotted(self.cfg, t)
                on_grid = any(_same(val, x) for x in e["grid"])
                ok = _same(val, e["value"]) and on_grid
                rows.append(dict(name=e["name"], target=t, value=e["value"], config_value=val, on_grid=on_grid,
                                 status=e.get("status"), kind=e.get("kind"), ok=ok))
                if not ok:
                    raise ValueError(f"lb_constants[{e['name']}]: config value {val!r} at {t} must equal the "
                                     f"registered value {e['value']!r} and lie on its grid {e['grid']}")
        return rows

    # ------------------------------------------------------------------ benchmarks + rater style
    def bench_preds(self) -> dict:
        """{static, vl3, vl4, dyn}: benchmark predictions after shrinkage (no offset), each dataset held out of
        its own error model and of the label constants (LODO)."""
        sh, b3 = self.R.sh, self.R.b3
        out = {}
        for split in ("lodo4", "lodo3"):
            B = self.bench_frame(split)
            empty = pd.Series("", index=B.index, dtype=object)
            for ds in B.dataset.unique():
                if split == "lodo3" and ds != "vl":
                    continue
                D, Btr = B[B.dataset == ds], B[B.dataset != ds]
                ctx = contextlib.nullcontext()
                if self.cfg["shrink"]["constants"]["bench"] == "lodo_label_constants":
                    c = b3.v4_constants(b3.V4_HOLD[ds])
                    ctx = sh.constants(c["pop_sd"], c["k0"], c["s_logk"])
                with ctx:
                    out[(split, ds)] = self.apply_shrink(D, self.error_model(Btr), empty, empty)
        q = self.R.b3._q
        return dict(static=q(out[("lodo4", "static")]), vl3=q(out[("lodo3", "vl")]),
                    vl4=pd.concat([q(out[("lodo4", "vl_young")]), q(out[("lodo4", "vl_old")])]),
                    dyn=q(out[("lodo4", "dynamic")]))

    def estimate_rater_style(self, out_dir: Path | None = None) -> dict:
        """Re-estimate the rater-style offsets from the OSF static xlsx (R1..R7) and this pipeline's own benchmark
        predictions with the pre-registered rule (scripts/rater_calibration.py cmd_offsets): label =
        (R7 + partner)/2 averaged over the 6 partners; offset = the clean10 pooled median of (label - ours), applied
        at HALF strength unless strongly supported; a component whose bootstrap CI includes 0 is not applied."""
        rs = self.cfg["rater_style"]["estimate"]
        rc = self.R.rc
        preds = self.bench_preds()
        st = preds["static"]
        RS = rc.static_raters()
        clean = list(self.cfg["benchmarks"]["static_clean10"])
        point = {}
        for c in C3:
            T2 = rc.two_rater(RS[c]).reindex(clean).to_numpy(float)
            Pv = st.reindex(clean)[c].to_numpy(float)
            point[f"{c}_total_add_clean10"] = float(np.nanmedian(T2 - Pv[:, None]))
        tmp = Path(out_dir or (Path(self.cfg["paths"]["out_dir"]) / "rater_style"))
        tmp.mkdir(parents=True, exist_ok=True)
        old = (rc.OUT, rc.rec_of, rc.bench_preds)
        try:
            rc.OUT = tmp
            rc.rec_of = lambda name, spec=None: {}                       # noqa: E731
            rc.bench_preds = lambda rec: {k: preds[k] for k in ("static", "vl3", "dyn")}   # noqa: E731
            with open(tmp / "rule_stdout.txt", "w") as fh, contextlib.redirect_stdout(fh):
                rc.cmd_offsets(argparse.Namespace(recipe=rs["tag"], B=int(rs["B"]), seed=int(rs["seed"])))
        finally:
            rc.OUT, rc.rec_of, rc.bench_preds = old
        js = json.loads((tmp / f"recommended_offsets_{rs['tag']}.json").read_text())
        dec = pd.DataFrame(js["decision"])
        rule = js["offsets"]
        if rs["strength"] == "rule":
            applied = dict(rule)
        elif rs["strength"] == "full":
            e = {(r.comp, r.part): (float(r.estimate), bool(r.recommend)) for r in dec.itertuples()}
            applied = {"pa_add": e[("PA", "total_add")][0] if e[("PA", "total_add")][1] else 0.0, "fl_mult": 1.0,
                       "mt_add": e[("MT", "total_add")][0] if e[("MT", "total_add")][1] else 0.0}
        else:
            raise ValueError(rs["strength"])
        return dict(applied={k: float(v) for k, v in applied.items()}, rule=rule, point=point,
                    decision=dec.to_dict("records"), out_dir=str(tmp))


# =================================================================================== stage: layout
def stage_layout(cfg: dict, out: Path) -> dict:
    from umud import layout as lm
    L = cfg["layout"]["clips"]
    lay, summary = lm.build_layout(cfg["paths"]["test_images"], threshold=float(L["threshold"]),
                                   min_frames=int(L["min_frames"]))
    path = Path(cfg["paths"]["test_layout"])
    lm.write_layout(lay, path)
    rep = dict(file=str(path), sha256=sha256_file(path), **summary,
               groups={k: int(v) for k, v in lay.groupby("group").size().items()}, peak_rss_gb=round(rss_gb(), 3))
    (out / "layout_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"layout: {summary['n_images']} images, {summary['n_clips']} clips; NCC within >= {summary['min_within']:.3f},"
        f" across <= {summary['max_across']:.3f} -> {path}")
    return rep


# =================================================================================== stage: build
def stage_build(cfg: dict, out: Path, R: SimpleNamespace | None = None) -> dict:
    t0 = time.time()
    lay_rep = stage_layout(cfg, out)          # scanner groups and clips from the images, every build
    R = R or bind(cfg)
    strict = cfg["output"].get("strict_pins", True)
    pins = check_hashes(cfg["inputs"], strict=strict) + check_hashes(cfg.get("code", []), strict=strict)
    A = Assembler(cfg, R)
    text, info = A.submission()
    o = cfg["output"]
    path = out / o["file_name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())
    got = from_text(text)
    rep = dict(config=cfg["_resolved"], name=cfg.get("name"), file=str(path), sha256=sha256_file(path),
               rows=len(got), layout={k: lay_rep[k] for k in ("n_clips", "frames_per_clip", "min_within", "max_across")},
               **info, pins_ok=all(r["ok"] for r in pins), environment=env_versions(),
               seconds=round(time.time() - t0, 2), peak_rss_gb=round(rss_gb(), 3))
    if hasattr(A, "rater_estimate"):
        rep["rater_style_reestimated"] = {k: A.rater_estimate[k] for k in ("applied", "rule", "point")}
    if o.get("validate", True):
        rep["validate"] = validate(cfg, path)
    ref = cfg.get("reference") or {}
    if ref.get("sha256"):
        rep["reference_sha256"] = dict(expected=ref["sha256"], ok=rep["sha256"] == ref["sha256"])
    if ref.get("file") and Path(ref["file"]).exists():
        f = read_sub(Path(ref["file"]))
        d = (got - f.reindex(got.index)).abs()
        rep["reference"] = dict(file=ref["file"], public_lb=ref.get("public_lb"),
                                bytes_identical=path.read_bytes() == Path(ref["file"]).read_bytes(),
                                max_abs={c: float(d[c].max()) for c in C3},
                                official_distance=official_distance(got, f))
    g = A.layout().loc[got.index, "group"]
    rep["group_medians"] = {k: {c: round(float(v), 4) for c, v in r.items()}
                            for k, r in got.groupby(g)[list(C3)].median().iterrows()}
    (out / f"{Path(o['file_name']).stem}_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"build {cfg.get('name')}: {path} ({rep['seconds']} s, peak RSS {rep['peak_rss_gb']} GB)")
    if "reference" in rep:
        r = rep["reference"]
        log(f"  vs reference {Path(r['file']).name}: bytes identical {r['bytes_identical']}, max |d| {r['max_abs']}, "
            f"distance {r['official_distance']:.2e}")
    return rep


def validate(cfg: dict, path: Path) -> str:
    repo = Path(cfg["paths"]["repo"])
    r = subprocess.run([sys.executable, str(repo / "scripts/validate_submission.py"), str(path), "--test-dir",
                        cfg["paths"]["test_images"]], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(repo / "src")))
    return (r.stdout + r.stderr).strip()


def env_versions() -> dict:
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                machine=platform.machine(), system=platform.system())


# =================================================================================== stage: measure
def stage_measure(cfg: dict, out: Path, ids: list[str] | None = None, dataset: str = "test", full: bool = False,
                  R: SimpleNamespace | None = None, datasets: list[str] | None = None) -> dict:
    """Verify the per-image caches, then re-measure (one image per group by default, everything with --full) from
    the maps. --full writes <out>/cache (v3 columns) and <out>/cache/v4extra; datasets not re-measured are copied
    from the configured caches so that `--stage build --set paths.v3_cache=<out>/cache
    --set paths.v4_cache=<out>/cache/v4extra` builds from the new measurements."""
    R = R or bind(cfg)
    M = cfg["measure"]
    rep: dict = {"caches": check_hashes([e for e in cfg["inputs"] if e.get("kind") == "cache"], strict=False)}
    check_geometry_config(cfg, R)
    b3 = R.b3
    new_cache = out / "cache"
    new_cache.mkdir(parents=True, exist_ok=True)
    old = (b3.CACHE, b3.V4_CACHE)
    try:
        b3.CACHE = new_cache
        b3.V4_CACHE = new_cache / "v4extra"
        if full:
            todo = list(datasets or M["datasets"])
            (new_cache / "v4extra").mkdir(parents=True, exist_ok=True)
            for ds in todo:
                t1 = time.time()
                b3.cmd_measure(argparse.Namespace(dataset=ds, apo=M["apo_source"], limit=0))
                b3.cmd_measure_extra(argparse.Namespace(dataset=ds, apo=M["apo_source"], limit=0))
                log(f"measured {ds} in {time.time() - t1:.0f} s, peak RSS {rss_gb():.2f} GB")
            copied = []
            for ds in M["datasets"]:
                if ds in todo:
                    continue
                for src_dir, dst_dir in ((Path(old[0]), new_cache), (Path(old[1]), new_cache / "v4extra")):
                    f = src_dir / f"{ds}__{M['apo_source']}.csv"
                    if f.exists() and not (dst_dir / f.name).exists():
                        shutil.copy2(f, dst_dir / f.name)
                        copied.append(str(dst_dir / f.name))
            rep["full"] = dict(measured=todo, copied_from_configured=copied, cache=str(new_cache))
        else:
            ids = ids or smoke_ids(cfg, R)
            rows = measure_ids(cfg, R, dataset, ids)
            new = pd.DataFrame(rows).set_index("image_id")
            new.to_csv(new_cache / f"{dataset}__{M['apo_source']}__smoke.csv")
            frozen = pd.read_csv(Path(old[0]) / f"{dataset}__{M['apo_source']}.csv").drop_duplicates(
                "image_id", keep="last").set_index("image_id").loc[new.index]
            num = [c for c in new.columns if c in frozen.columns and _is_num(new[c]) and _is_num(frozen[c])]
            d = (new[num].astype(float) - frozen[num].astype(float)).abs()
            both_nan = new[num].isna() & frozen[num].isna()
            one_nan = new[num].isna() ^ frozen[num].isna()
            dd = d.where(~both_nan)
            rep["smoke"] = dict(dataset=dataset, ids=ids, n_columns=len(num),
                                max_abs_diff=float(np.nanmax(dd.to_numpy())) if dd.notna().any().any() else 0.0,
                                nan_mismatch=int(one_nan.to_numpy().sum()))
    finally:
        b3.CACHE, b3.V4_CACHE = old
        b3._MEMO.clear()
    rep["peak_rss_gb"] = round(rss_gb(), 3)
    (out / "measure_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"measure: {json.dumps(rep.get('smoke', {'full': True}), default=str)}; peak RSS {rep['peak_rss_gb']} GB")
    return rep


def smoke_ids(cfg: dict, R: SimpleNamespace) -> list[str]:
    """The first image (in file-name order) of every scanner group, plus the configured extra ids."""
    lay = Assembler(cfg, R).layout()
    first = [sorted(lay.index[lay["group"] == g])[0] for g in cfg["groups"]["all"]]
    return first + [i for i in cfg["measure"].get("smoke_extra_ids", []) if i not in first]


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def bind_maps(cfg: dict, R: SimpleNamespace) -> None:
    """Point the map readers (eval_apo_v2.Banks, eval_orient) at the config's banks and check that the
    aponeurosis source spec of build_candidates_v3 equals the config's maps.sources."""
    import eval_apo_v2 as ea
    import eval_orient as eo
    S = cfg["maps"]["sources"]
    v2_root, v2_model = Path(S["apo_primary"]["bank"]), S["apo_primary"]["model"]
    ea.MAIN_BANK = Path(S["fasc_fragments"]["bank"])
    ea.V2_BANK, ea.V2_MODEL = v2_root, v2_model
    ea.Banks.__init__.__defaults__ = (v2_root, v2_model)
    eo.OBANK_ROOT = Path(S["fasc_field"]["bank"])
    eo.OMODEL = S["fasc_field"]["model"]
    src = R.b3.APO_SRC[cfg["measure"]["apo_source"]]
    want = dict(primary=("v2", S["apo_primary"]["channels"][0], float(S["apo_primary"]["thr"])),
                edges=S["apo_primary"]["edges"] == "centroid",
                fallbacks=[(f["model"].split("_")[0], ("main", f["model"], float(f["thr"]))) for f in S["apo_fallbacks"]])
    got = dict(primary=tuple(src["primary"]), edges=bool(src["edges"]),
               fallbacks=[(n, tuple(s)) for n, s in src["fallbacks"]])
    if got != want:
        raise RuntimeError(f"aponeurosis source drift: code {got} vs config {want}")
    if cfg["measure"]["fragments"]["model"] != S["fasc_fragments"]["model"]:
        raise RuntimeError("measure.fragments.model must equal maps.sources.fasc_fragments.model")


def check_geometry_config(cfg: dict, R: SimpleNamespace):
    """The effective geometry / field configs must equal the dumps pinned in the config."""
    import eval_apo_v2 as ea
    from umud import geometry_v2 as g2
    bind_maps(cfg, R)
    M = cfg["measure"]
    src = R.b3.APO_SRC[M["apo_source"]]
    geo = ea.make_cfg(M["geometry"]["name"], src["primary"][2], src["edges"]).with_(fasc_thr=M["fragments"]["thr"])
    got = json.loads(json.dumps(dataclasses.asdict(geo), default=str))
    want = M["geometry"]["resolved"]
    diff = {k: (got.get(k), want.get(k)) for k in set(got) | set(want) if got.get(k) != want.get(k)}
    if diff:
        raise RuntimeError(f"geometry config drift (code vs config): {diff}")
    fgot = json.loads(json.dumps(dataclasses.asdict(g2.FieldConfig()), default=str))
    fdiff = {k: (fgot.get(k), M["field"].get(k)) for k in set(fgot) | set(M["field"]) if fgot.get(k) != M["field"].get(k)}
    if fdiff:
        raise RuntimeError(f"field config drift (code vs config): {fdiff}")
    return geo


def measure_ids(cfg: dict, R: SimpleNamespace, dataset: str, ids: list[str]) -> list[dict]:
    """b3.cmd_measure's per-image body for selected ids (one image and one map key at a time)."""
    import eval_apo_v2 as ea
    import eval_orient as eo
    from umud import geometry_v2 as g2
    M = cfg["measure"]
    if dataset != "test":
        raise NotImplementedError("selected-id re-measurement is wired for the test set (benchmarks: use --full)")
    lay = Assembler(cfg, R).layout()
    bind_maps(cfg, R)
    banks = ea.Banks()
    obank = g2.MaskBank(Path(cfg["maps"]["sources"]["fasc_field"]["bank"]))
    rows = []
    try:
        for iid in ids:
            img = R.bc.gray(Path(cfg["paths"]["test_images"]) / iid)
            ppc = float(lay.loc[iid, "px_per_cm"])
            extra = {"group": lay.loc[iid, "group"], "clip": int(lay.loc[iid, "clip"])}
            A, up, lo, fbs, thr, edges = R.b3.apo_inputs(banks, dataset, iid, M["apo_source"])
            F = banks.main.prob(dataset, M["fragments"]["model"], iid)
            orient = eo.load_orient(obank, dataset, Path(iid).stem)
            if A is None or F is None or orient is None or orient[1].shape != A.shape:
                r = g2._nan_result("missing_maps").as_row()
            else:
                r = R.b3.measure_one(A, up, lo, fbs, thr, edges, F, float(M["fragments"]["thr"]), img, ppc, orient)
            rows.append(dict(image_id=iid, okey=Path(iid).stem, px_per_cm=ppc, **extra, **r))
            del A, up, lo, fbs, F, orient, img
    finally:
        banks.close()
        obank.close()
    return rows


# =================================================================================== stage: maps
def stage_maps(cfg: dict, out: Path) -> dict:
    """Write one ready-to-push Kaggle kernel folder per map kernel and check the local map files."""
    mp = cfg["maps"]
    kdir = out / "kaggle"
    kdir.mkdir(parents=True, exist_ok=True)
    kag = mp["kaggle"]
    rows = []
    for k in mp["kernels"]:
        src = Path(k["code_dir"])
        dst = kdir / k["slug"]
        dst.mkdir(parents=True, exist_ok=True)
        meta = json.loads((src / "kernel-metadata.json").read_text())
        code = src / meta["code_file"]
        shutil.copy2(code, dst / code.name)
        (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        push = f'"{kag["cli"]}" kernels push -p "{dst}"'
        if meta.get("enable_gpu") == "true":
            push += f' --accelerator {kag["accelerator"]}'
        cmd = [push, f'"{kag["cli"]}" kernels status {meta["id"]}',
               f'"{kag["cli"]}" kernels output {meta["id"]} -p "{k["download_to"]}" --force'
               + (f' --file-pattern "{k["file_pattern"]}"' if k.get("file_pattern") else "")]
        srcs = dict(kernel_sources=meta.get("kernel_sources", []), dataset_sources=meta.get("dataset_sources", []),
                    competition_sources=meta.get("competition_sources", []))
        (dst / "RUN.md").write_text(
            f"# {k['slug']}\n\n{k.get('note', '').strip()}\n\nProduces: {', '.join(k.get('produces', []))}\n\n"
            f"Checkpoint / input sources: {json.dumps(srcs)}\n\n"
            "Run on Kaggle (the laptop never runs inference). This stage never pushes; run these by hand:\n\n```\n"
            + "\n".join(cmd) + "\n```\n\nThe frozen outputs used by the configs are at: "
            + str(k.get("frozen_at", "")) + "\n")
        rows.append(dict(slug=k["slug"], id=meta["id"], code=code.name, sources=srcs,
                         in_repo=str(src.resolve()).startswith(str(Path(cfg["paths"]["repo"]).resolve())),
                         commands=cmd))
    files = check_hashes(mp.get("files", []), strict=False)
    rep = dict(kernels=rows, checkpoints=mp.get("checkpoints", []), map_files=files,
               map_files_ok=all(r["ok"] for r in files))
    (out / "maps_report.json").write_text(json.dumps(rep, indent=1))
    log(f"maps: {len(rows)} kernel folders in {kdir}; map files ok={rep['map_files_ok']} ({len(files)} pinned)")
    return rep


# =================================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="umud-run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--out", default=None, help="output directory (default: paths.out_dir of the config)")
    ap.add_argument("--set", action="append", default=[], help="override a config key: a.b.c=value (YAML value)")
    ap.add_argument("--ids", default=None, help="measure: comma-separated test image ids to re-measure")
    ap.add_argument("--dataset", default="test")
    ap.add_argument("--full", action="store_true", help="measure: re-measure every dataset")
    ap.add_argument("--datasets", default=None, help="measure --full: comma-separated subset (e.g. test)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, sets=a.set)
    out = Path(a.out or cfg["paths"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "layout":
        stage_layout(cfg, out)
    R = bind(cfg)
    if a.stage in ("maps", "all"):
        stage_maps(cfg, out)
    if a.stage in ("measure", "all"):
        stage_measure(cfg, out, ids=a.ids.split(",") if a.ids else None, dataset=a.dataset, full=a.full, R=R,
                      datasets=a.datasets.split(",") if a.datasets else None)
    if a.stage in ("build", "all"):
        rep = stage_build(cfg, out, R=R)
        if "Valid submission" not in rep.get("validate", "Valid submission"):
            log(f"VALIDATION FAILED: {rep['validate']}")
            return 2
        exp = (cfg.get("reference") or {}).get("expect")
        if exp == "bytes" and not rep.get("reference", {}).get("bytes_identical"):
            log("REFERENCE MISMATCH: the rebuilt file is not byte-identical to the reference")
            return 3
        if "reference_sha256" in rep:
            log(f"  sha256 {rep['sha256'][:16]}... matches the config: {rep['reference_sha256']['ok']}")
            if not rep["reference_sha256"]["ok"]:
                log("REFERENCE MISMATCH: sha256 of the rebuilt file differs from the config's reference.sha256")
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
