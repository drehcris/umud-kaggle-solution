#!/usr/bin/env python
"""aniso_phys_v9 -- a verbatim copy of FB2's pipeline-exact physical-geometry module
(scratchpad v8/aniso/aniso.py, written 2026-10-22 by the FB2 agent of the UMUD v8 workflow), copied for physbase9
(scripts/build_candidates_v9.py) so the physical caches can be regenerated from the repository.

Changes against the original (nothing else is edited):
  * OUT -> artifacts/candidates_v9/phys (the original wrote to the session scratchpad v8/aniso);
  * this attribution header.
The original module docstring follows unchanged.

FB2 -- pipeline-exact physical geometry for the anisotropic 1200x800 layouts, plus factor detectors that
re-run on new data. Scratch only: nothing in src/ is edited, nothing is submitted or committed.

Convention
----------
c = dx_image / dx_physical when the vertical scale bar (px/cm) is used for both axes, i.e. the horizontal
px/cm divided by the vertical px/cm. A point (x, y) of the provided image is (x / c, y) in the physical
frame (square pixels at the vertical scale). Angles: tan(phi_phys) = c * tan(phi_img). MT is vertical and
does not change.

IMPORTANT PRIOR: the organizer stated (threads 689019 / 748126) that the raters used the vertical scale bar
on the provided images and treated pixels as square, also on the stretched 1200x800 frames. The physical
files built here test the ALTERNATIVE (labels drawn on the native-resolution frames). The images are
anisotropic (the detectors below show it); whether the LABELS are is a separate question that only the
probes (J1/J2/J3 or the P files here) can answer.

Factor detectors  factor(image_path, group=None) -> c   (group=None: the layout is detected from the image)
-----------------------------------------------------------------------------------------------------------
* Telemed ArtUS 1200x800 (left ruler): bottom-axis tick spacing (sub-pixel centroids of the ticks standing
  on the bottom axis line, bottom 8 rows) / vertical px/cm from the sub-pixel centroids of the left-ruler cm
  ticks. Measured ~0.908; native-resolution table: 1340x810 -> 1200x800 = (1200/1340)/(800/810) = 0.90672.
* Philips Lumify PNG: width / height of the filled grey round 'P' orientation disc (connected component in
  the top third / left half -- the disc sits at x 270-420, so a strict left third cuts it on 10 images --
  with the holes (the letter) closed), from a sub-pixel axis-aligned ellipse fit to the 50 % edge crossings
  of 240 rays (bbox and second-moment ratios are reported too). Synthetic discs resampled with bilinear /
  bicubic / Lanczos filters are recovered to <= 0.001; box filtering biases it by up to 0.005. Real frames:
  0.834-0.845 (median 0.840), deterministic per disc pixel phase. Table: 16:9 screen -> 3:2 = 0.84375
  (a 996x556 frame would give 0.8374 -- not separable from 16:9 with this marker).
* Siemens Juniper 1200x800 (right ruler): constant 1.125 = the native 1152x864 -> 1200x800 resample
  (the 474 training frames of this scanner have 1152x864 label masks). FB0 independently found horizontal
  references in the UI triangles (cine-bar end caps 1.120 / 1.125, focus marker 1.123, native 45-degree
  edges assumed); they are not re-implemented here.
* Everything else -> 1.0: Telemed 1088x644 (its tick ratio is measured as a check, 1.000), legacy 853/512,
  static / dynamic / VL benchmark images.
factor() returns the table value when the detector confirms the layout (|c_meas - c_table| <= TOL); the
raw measurement is kept in factor_detail(). mode="measured" returns the measurement instead.

Pipeline-exact physical measurement
-----------------------------------
Every map the C12/C18 per-image pass uses (apo_v2fix band/upper/lower, ours_apo384 + dlt_apo fallbacks,
ours_fasc384_noflip fragments, fasc_orient_v1 seg/cos2/sin2, the grey image) is resampled to the physical
frame, x -> x / c (cv2.warpAffine, exact factor, pixel-centre aligned, bilinear), and the orientation field
angles are mapped theta -> atan2(sin theta, cos theta / c). Then the unchanged per-image pass
(build_candidates_v3.measure_one = measure_apo + field_architecture + constructed-FL grid) runs with the
unchanged vertical px/cm: fragment angles, lengths, quadratic aponeurosis fits, FOV width and every
diagnostic are therefore computed on (x / c, y) points, in px of the vertical scale. mt_mm is pinned to the
image-space value (MT is vertical; the re-measured value is kept as mt_mm_phys for the check). The C12
recipe (test_run5: shrinkage medians per scanner and scanner x muscle, regime gate, EB FL, confidence PA,
rater offsets) and the C18 change (Telemed FL raw per-clip median) are then rebuilt from the physical
cache by the project's own code (V3_CACHE override), so all medians / EB / confidence shrinkage are in
physical space.

Subcommands (one process each, sequential, one image + its maps at a time, peak 0.44 GB RSS)
  factors                      -> factors_test.csv  (309 test images; group from the layout table + auto-detect)
  bench-factors                -> factors_bench.csv (static 35, dynamic 168, VL 249; layout auto-detected)
  measure --mode phys          -> rows_phys/  (239 rows: Lumify, Siemens, T12 at their c; ~4 min)
  measure --mode c1            -> rows_c1/    (all 309 at c = 1: identity check of the re-measurement; ~5 min)
      [--only LUM T12 --measured --tag meas | --c LUM=0.8374 --tag x]   sensitivity variants
  build [C1ID LUM SIE T12 ALL] [--tag x] -> P_<v>.csv, D_C18_rebuilt_c1.csv, build_report.json
  compare                      -> compare_to_J.csv
  fb0-check                    -> factors_vs_fb0.csv (agreement with v8/audit_J and v8/review factor scans)
Run with PYTHONPATH=<project>/src and the project venv.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# research-era session files (judge / critic / FB2 outputs), archived under artifacts/legacy_research; research
# script only -- the final submissions are rebuilt by src/umud/run.py (configs/final_{A,B}.yaml), not by this file
SCR = Path(os.environ.get("UMUD_RESEARCH_ROOT", ROOT / "artifacts/legacy_research"))
OUT = ROOT / "artifacts/candidates_v9/phys"            # FB2 original: SCR / "v8/aniso"
OUT.mkdir(parents=True, exist_ok=True)
JDIR = SCR / "v8/judge"
TEST_DIR = ROOT / "data/umud-challenge-muscle-architecture-in-ultrasound-data/test_images_v2/test_set_v2"
C18_FILE = ROOT / "artifacts/candidates_v6/C18_C12_telemedFLraw.csv"
V3_CACHE_ORIG = ROOT / "artifacts/candidates_v3/cache"
PY = ROOT / ".venv/bin/python"

T12, T10, LUM, SIE = "telemed_artus_1200x800", "telemed_artus_1088x644", "philips_lumify", "siemens_juniper"
TELEMED = (T12, T10)
# native-resolution table (resample to 1200x800, labels kept at native size in training)
NATIVE = {
    SIE: dict(native="1152x864", c=(1200 / 1152) / (800 / 864)),          # 1.125
    T12: dict(native="1340x810", c=(1200 / 1340) / (800 / 810)),          # 0.906716
    LUM: dict(native="16:9 screen", c=(1200 / 800) / (16 / 9)),           # 0.84375
}
TOL = 0.015         # detector must confirm the table value within this. Per-image detector scatter on the real
                    # frames: Lumify ellipse SD 0.003 (range 0.834-0.845; resample-filter bias up to 0.005 on
                    # synthetic box-filtered discs), Telemed ticks SD < 0.001. Table entries of different native
                    # layouts are >= 0.03 apart, except 16:9 (0.8438) vs 996x556 (0.8374): not separable here.
PHYS_GROUPS = {"LUM": LUM, "SIE": SIE, "T12": T12}
J_FILES = {"LUM": "J1_C18_LUMphys084.csv", "SIE": "J2_C18_SIEphys1125.csv", "T12": "J3_C18_T12phys0907.csv"}
TAU = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
C3 = ("pa_deg", "fl_mm", "mt_mm")


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS


def load_gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


# ============================================================================== layout detection
def detect_layout(g: np.ndarray, path=None) -> str:
    """Scanner layout from the image alone (the first, cheap checks of umud.geometry.ultrasound_metadata,
    without its OCR fallback). Returns one of the test group names, 'legacy_512', 'legacy_853' or 'other'.
    Layout rules adapted from AmbrosM, "UMUD Quick and Dirty" (Apache-2.0); see THIRD_PARTY_NOTICES.md."""
    H, W = g.shape
    suffix = Path(path).suffix.lower() if path is not None else ""
    if (W, H) == (1200, 800):
        if suffix == ".png":
            col6, col9 = g[:, 6] > 50, g[:, 9] > 50
            if col6.any() and col6[150:].any() and col9[150:].any():
                return LUM
        if np.all(np.abs(g[87, 1147:1157].astype(int) - 175) <= 3):
            return SIE
        if np.all(g[42, 67:74] > 115):
            return T12
        return "other"
    if (W, H) == (1088, 644):
        return T10
    if H in (512, 513) and W < 600:
        return "legacy_512"
    if H == 853:
        return "legacy_853"
    return "other"


# ============================================================================== Telemed ticks
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of the True runs of a 1-D bool array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _fit_spacing(cen: np.ndarray, guess: float | None = None) -> tuple[float, float, int]:
    """Least-squares spacing of equally spaced tick centroids (integer index by rounding, outliers dropped)."""
    cen = np.sort(np.asarray(cen, float))
    if len(cen) < 3:
        return np.nan, np.nan, len(cen)
    sp0 = guess if guess else float(np.median(np.diff(cen)))
    idx = np.round((cen - cen[0]) / sp0)
    keep = np.abs((cen - cen[0]) - idx * sp0) < 0.1 * sp0
    for _ in range(3):
        A = np.stack([idx[keep], np.ones(keep.sum())], 1)
        (s, b), *_ = np.linalg.lstsq(A, cen[keep], rcond=None)
        res = cen - (idx * s + b)
        keep = np.abs(res) < max(1.0, 4 * np.median(np.abs(res[keep])) + 0.25)
    A = np.stack([idx[keep], np.ones(keep.sum())], 1)
    (s, b), *_ = np.linalg.lstsq(A, cen[keep], rcond=None)
    rms = float(np.sqrt(np.mean((cen[keep] - (idx[keep] * s + b)) ** 2)))
    return float(s), rms, int(keep.sum())


def telemed_ticks(g: np.ndarray) -> dict:
    """Bottom-axis tick spacing and left-ruler px/cm (both sub-pixel), Telemed ArtUS layouts."""
    a = g.astype(np.float32)
    H, W = a.shape
    out = dict(bottom_spacing=np.nan, bottom_n=0, bottom_rms=np.nan, ruler_ppc=np.nan, ruler_n=0,
               ruler_rms=np.nan, ruler_x=-1, axis_y=-1)
    # ---- left ruler: the bright near-constant vertical line in the left 15 % of the frame
    y0, y1 = int(0.1 * H), int(0.9 * H)
    best = None
    for x in range(0, int(0.15 * W)):
        col = a[y0:y1, x]
        if col.mean() > 100 and col.std() < 10 and (best is None or col.mean() > best[1]):
            best = (x, col.mean())
    if best is not None:
        xr = best[0]
        prof = a[:, xr + 1: xr + 4].mean(1)
        bg = float(np.median(prof[y0:y1]))
        hit = prof > bg + 30
        cen = []
        for s, e in _runs(hit):
            if e - s > 5 or s < 2 or e > H - 2:      # focus marker / frame border
                continue
            lo, hi = s - 1, e + 1
            w = np.clip(prof[lo:hi] - bg, 0, None)
            if w.sum() > 0:
                cen.append(float((np.arange(lo, hi) * w).sum() / w.sum()))
        sp, rms, n = _fit_spacing(np.array(cen))
        out.update(ruler_ppc=sp, ruler_n=n, ruler_rms=rms, ruler_x=xr)
    # ---- bottom axis: the bright near-constant row in the bottom 8 rows; ticks stand on it
    yb = None
    for y in range(H - 1, H - 9, -1):
        seg = a[y, W // 8: 7 * W // 8]
        med = float(np.median(seg))
        if med > 120 and (np.abs(seg - med) < 20).mean() > 0.9:
            yb = y
            break
    if yb is not None:
        lv = float(np.median(a[yb, W // 8: 7 * W // 8]))
        blk = a[yb - 4: yb, :]
        hit = blk.min(0) >= lv - 25
        q = blk.mean(0)
        cen = []
        for s, e in _runs(hit):
            if e - s > 4 or s < 6 or e > W - 6:
                continue
            lo, hi = s - 1, e + 1
            bg = min(q[lo - 4: lo - 1].mean(), q[hi + 1: hi + 4].mean())
            w = np.clip(q[lo:hi] - bg, 0, None)
            if w.sum() > 0:
                cen.append(float((np.arange(lo, hi) * w).sum() / w.sum()))
        sp, rms, n = _fit_spacing(np.array(cen))
        out.update(bottom_spacing=sp, bottom_n=n, bottom_rms=rms, axis_y=yb)
    return out


# ============================================================================== Lumify P disc
def _bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    H, W = a.shape
    x = np.clip(x, 0, W - 1.001)
    y = np.clip(y, 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy) + a[y0 + 1, x0] * (1 - fx) * fy
            + a[y0 + 1, x0 + 1] * fx * fy)


def lumify_disc(g: np.ndarray, n_rays: int = 240, levels: tuple = (0.5,)) -> dict:
    """Aspect ratio (width / height) of the filled grey round 'P' marker in the top-left third."""
    from scipy import ndimage as ndi
    a = g.astype(np.float32)
    H, W = a.shape
    sub = a[: H // 3, : W // 2]          # top third, left half (the disc sits at x 270-410)
    lab, k = ndi.label(sub > 25)
    best = None
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (20 <= h <= 70 and 15 <= w <= 70) or not (0.6 <= w / h <= 1.4):
            continue
        if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop >= sub.shape[0] or sl[1].stop >= sub.shape[1]:
            continue
        reg = ndi.binary_fill_holes(lab[sl] == i)
        inner = sub[sl][reg]
        # a filled ellipse covers pi/4 = 0.785 of its bbox; the disc carries the white letter
        if not (0.70 <= reg.mean() <= 0.88) or (inner > 200).mean() < 0.03:
            continue
        if best is None or reg.sum() > best[1].sum():
            best = (sl, reg)
    out = dict(disc_found=False, c_ellipse=np.nan, c_bbox=np.nan, c_moment=np.nan, disc_w=np.nan, disc_h=np.nan,
               ell_rms=np.nan, ell_n=0, disc_x=np.nan, disc_y=np.nan)
    if best is None:
        return out
    sl, reg = best
    ys, xs = np.nonzero(reg)
    ys, xs = ys + sl[0].start, xs + sl[1].start
    cy, cx = ys.mean(), xs.mean()
    out.update(disc_found=True, disc_w=int(sl[1].stop - sl[1].start), disc_h=int(sl[0].stop - sl[0].start),
               c_moment=float(xs.std() / ys.std()), disc_x=float(cx), disc_y=float(cy))
    out["c_bbox"] = out["disc_w"] / out["disc_h"]
    # sub-pixel 50 % crossings along rays from the centroid
    rmax = 0.75 * max(out["disc_w"], out["disc_h"])
    r = np.arange(0.0, rmax, 0.05)
    pts = []
    for t in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        px, py = cx + r * np.cos(t), cy + r * np.sin(t)
        v = _bilinear(a, px, py)
        on = np.flatnonzero(v > 10)
        if not len(on):
            continue
        j_out = on[-1]                          # last sample above the background
        rin = r[j_out] - 2.0                    # local fill level just inside the edge (letter excluded)
        m = (r >= rin - 1.5) & (r <= rin) & (v < 180)
        if m.sum() < 5:
            continue
        fill = float(np.median(v[m]))
        res_l = []
        for q in levels:                        # edge radius = mean of the crossings at these fill fractions
            lvl = q * fill
            jj = np.flatnonzero((v[:-1] >= lvl) & (v[1:] < lvl))
            jj = jj[r[jj] >= rin - 1.5]
            if not len(jj):
                break
            j = jj[-1]
            f = (v[j] - lvl) / (v[j] - v[j + 1])
            res_l.append(r[j] + f * (r[j + 1] - r[j]))
        if len(res_l) != len(levels):
            continue
        re = float(np.mean(res_l))
        pts.append((cx + re * np.cos(t), cy + re * np.sin(t)))
    if len(pts) < 30:
        return out
    P = np.array(pts)
    X, Y = P[:, 0] - cx, P[:, 1] - cy
    keep = np.ones(len(P), bool)
    for _ in range(3):        # axis-aligned ellipse  A x^2 + B y^2 + C x + D y = 1, residual-trimmed
        M = np.stack([X ** 2, Y ** 2, X, Y], 1)
        (A, B, C, D), *_ = np.linalg.lstsq(M[keep], np.ones(keep.sum()), rcond=None)
        x0, y0 = -C / (2 * A), -D / (2 * B)
        F = 1 + A * x0 ** 2 + B * y0 ** 2
        ax, ay = np.sqrt(F / A), np.sqrt(F / B)
        rr = np.sqrt(((X - x0) / ax) ** 2 + ((Y - y0) / ay) ** 2)
        res = (rr - 1) * np.sqrt(ax * ay)                       # ~ radial residual in px
        keep = np.abs(res) < max(0.3, 3 * np.median(np.abs(res[keep])))
    out.update(c_ellipse=float(ax / ay), ell_rms=float(np.sqrt(np.mean(res[keep] ** 2))), ell_n=int(keep.sum()),
               ell_ax=float(2 * ax), ell_ay=float(2 * ay))
    return out


# ============================================================================== factor
def factor_detail(image_path, group: str | None = None, g: np.ndarray | None = None) -> dict:
    """Detector output for one image: layout, c_meas (raw detector), c_table, c (the factor used), method."""
    g = load_gray(image_path) if g is None else g
    lay = detect_layout(g, image_path)
    grp = group or lay
    d = dict(image_id=Path(image_path).name, group=grp, layout_detected=lay, H=g.shape[0], W=g.shape[1],
             c_meas=np.nan, c_table=np.nan, c=1.0, method="default_1.0", note="")
    if grp in TELEMED:
        t = telemed_ticks(g)
        d.update({f"t_{k}": v for k, v in t.items()})
        cm = t["bottom_spacing"] / t["ruler_ppc"] if t["ruler_ppc"] > 0 else np.nan
        d["c_meas"] = cm
        if grp == T12:
            d["c_table"] = NATIVE[T12]["c"]
            d["method"] = "bottom_ticks/left_ruler"
        else:
            d["method"] = "default_1.0 (tick ratio measured as a check)"
            d["c_table"] = 1.0
    elif grp == LUM:
        t = lumify_disc(g)
        d.update({f"p_{k}": v for k, v in t.items()})
        d["c_meas"] = t["c_ellipse"] if np.isfinite(t["c_ellipse"]) else t["c_moment"]
        d["c_table"] = NATIVE[LUM]["c"]
        d["method"] = "P_disc_ellipse_w/h"
    elif grp == SIE:
        d["c_table"] = NATIVE[SIE]["c"]
        d["method"] = "constant_native_1152x864"
    if grp in (T12, LUM, SIE):
        ok = (not np.isfinite(d["c_meas"])) or abs(d["c_meas"] - d["c_table"]) <= TOL
        d["c"] = float(d["c_table"]) if ok else float(d["c_meas"])
        if grp != SIE and not np.isfinite(d["c_meas"]):
            d["note"] = "detector_failed->table"
        elif not ok:
            d["note"] = "off_table->measured"
    if group is not None and lay != group:
        d["note"] = (d["note"] + ";" if d["note"] else "") + f"layout_mismatch:{lay}"
    return d


def factor(image_path, group: str | None = None, mode: str = "table") -> float:
    """c = dx_image / dx_physical for one image (1.0 = square pixels). mode: 'table' | 'measured'."""
    d = factor_detail(image_path, group)
    if mode == "measured" and np.isfinite(d["c_meas"]) and d["group"] in (T12, LUM):
        return float(d["c_meas"])
    return float(d["c"])


# ============================================================================== physical resampling
def warp_x(a: np.ndarray | None, c: float) -> np.ndarray | None:
    """Resample a 2-D map to the physical frame: x_phys = x / c (pixel centres aligned, exact factor)."""
    if a is None or c == 1.0:
        return a
    import cv2
    H, W = a.shape[:2]
    Wn = int(np.ceil(W / c - 1e-9))
    M = np.array([[1.0 / c, 0.0, 0.5 / c - 0.5], [0.0, 1.0, 0.0]], np.float64)
    src = a if a.dtype in (np.uint8, np.float32) else a.astype(np.float32)
    return cv2.warpAffine(src, M, (Wn, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_orient(orient, c: float):
    """(seg, cos2, sin2) floats -> physical frame: resample, then theta -> atan2(sin theta, cos theta / c)."""
    if orient is None or c == 1.0:
        return orient
    seg, c2, s2 = orient
    seg, c2, s2 = warp_x(seg, c), warp_x(c2, c), warp_x(s2, c)
    th = 0.5 * np.arctan2(s2, c2)
    m = np.hypot(c2, s2)
    th2 = np.arctan2(np.sin(th), np.cos(th) / c)
    return seg, (m * np.cos(2 * th2)).astype(np.float32), (m * np.sin(2 * th2)).astype(np.float32)


# ============================================================================== subcommand: factors
def cmd_factors(args) -> None:
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import build_candidates as bc
    lay = bc.test_layout()
    rows = []
    t0 = time.time()
    for iid in lay.index:
        g = load_gray(TEST_DIR / iid)
        d = factor_detail(TEST_DIR / iid, str(lay.loc[iid, "group"]), g)
        d_auto = factor_detail(TEST_DIR / iid, None, g)
        d["layout_table"] = str(lay.loc[iid, "group"])
        d["c_auto"] = d_auto["c"]
        d["px_per_cm_layout"] = float(lay.loc[iid, "px_per_cm"])
        if d["group"] in TELEMED and d.get("t_bottom_spacing") is not None:
            d["c_meas_vs_layout_ppc"] = d["t_bottom_spacing"] / d["px_per_cm_layout"]
        rows.append(d)
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "factors_test.csv", index=False)
    print(f"factors_test.csv: {len(F)} rows, {time.time() - t0:.0f}s, rss {rss_gb():.2f} GB")
    s = F.groupby("group").agg(n=("c", "size"), c_used=("c", "median"), c_meas_med=("c_meas", "median"),
                               c_meas_min=("c_meas", "min"), c_meas_max=("c_meas", "max"),
                               c_meas_sd=("c_meas", "std"), auto_agree=("c_auto", lambda x: 0))
    s["auto_agree"] = F.groupby("group").apply(lambda d: float((np.abs(d.c_auto - d.c) < 1e-12).mean()))
    print(s.to_string())
    print("notes:", F.note.value_counts().to_dict())


def cmd_bench_factors(args) -> None:
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import build_candidates as bc
    rows = []
    t0 = time.time()
    lab = bc.static_labels()
    items = [("static", iid, bc.STATIC / f"{iid}.tif") for iid in lab.index]
    items += [("dynamic", p.stem, p) for p in sorted((bc.DYN / "frames").glob("frame_*.png"))]
    idx = pd.read_csv(bc.VL / "vl_index.csv")
    items += [("vl", r.key, bc.VL / "images" / f"{r.key}.png") for r in idx.itertuples()]
    for ds, key, p in items:
        g = load_gray(p)
        d = factor_detail(p, None, g)          # layout detected from the image alone
        rows.append(dict(dataset=ds, key=key, **{k: d[k] for k in ("image_id", "layout_detected", "H", "W",
                                                                      "c_meas", "c", "method", "note")}))
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "factors_bench.csv", index=False)
    print(f"factors_bench.csv: {len(F)} rows, {time.time() - t0:.0f}s, rss {rss_gb():.2f} GB")
    print(F.groupby("dataset").agg(n=("c", "size"), all_one=("c", lambda x: bool((x == 1.0).all())),
                                   layouts=("layout_detected", lambda x: dict(x.value_counts())),
                                   sizes=("W", lambda x: sorted(set(x)))).to_string())


# ============================================================================== subcommand: measure
def _project_imports():
    for p in (str(ROOT / "scripts"), str(ROOT / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import build_candidates_v3 as b3
    import build_candidates as bc
    import eval_apo_v2 as ea
    import eval_orient as eo
    from umud import geometry_v2 as g2
    return b3, bc, ea, eo, g2


def measure_image(iid: str, c: float, ctx) -> dict:
    """The C12/C18 per-image pass (build_candidates_v3.cmd_measure, test, apo v2) on the physical frame."""
    b3, bc, ea, eo, g2, banks, obank, lay = ctx
    import cv2
    img = bc.gray(TEST_DIR / iid)
    ppc = float(lay.loc[iid, "px_per_cm"])
    extra = {"group": lay.loc[iid, "group"], "clip": int(lay.loc[iid, "clip"])}
    okey = Path(iid).stem
    A, up, lo, fbs, thr, edges = b3.apo_inputs(banks, "test", iid, "v2")
    F = banks.main.prob("test", "ours_fasc384_noflip", iid)
    orient = eo.load_orient(obank, "test", okey)
    if A is None or F is None or orient is None or orient[1].shape != A.shape:
        r = g2._nan_result("missing_maps").as_row()
    else:
        try:
            if c != 1.0:
                if img.shape != A.shape:     # measure_one's own alignment, done before the warp
                    img = cv2.resize(img, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
                A, up, lo, F, img = (warp_x(m, c) for m in (A, up, lo, F, img))
                fbs = [(n, warp_x(m, c), t) for n, m, t in fbs]
                orient = warp_orient(orient, c)
            r = b3.measure_one(A, up, lo, fbs, thr, edges, F, 0.95, img, ppc, orient)
        except Exception as e:  # keep going; record the failure (as cmd_measure does)
            r = g2._nan_result(f"error:{type(e).__name__}:{e}").as_row()
    return dict(image_id=iid, okey=okey, px_per_cm=ppc, **extra, **r)


def cmd_measure(args) -> None:
    b3, bc, ea, eo, g2 = _project_imports()
    lay = bc.test_layout()
    fac = pd.read_csv(OUT / "factors_test.csv").set_index("image_id")
    orig = pd.read_csv(V3_CACHE_ORIG / "test__v2.csv").set_index("image_id")
    only = [PHYS_GROUPS[k] for k in args.only] if args.only else list(PHYS_GROUPS.values())
    cov = {PHYS_GROUPS[k]: float(v) for k, v in (x.split("=") for x in args.c)} if args.c else {}
    if args.mode == "phys":
        ids = [i for i in lay.index if lay.loc[i, "group"] in only]
    else:
        ids = list(lay.index)
    d = OUT / (f"rows_{args.mode}" + (f"_{args.tag}" if args.tag else ""))
    d.mkdir(exist_ok=True)
    path = d / "test__v2_rows.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(prev.image_id) if len(prev) else set()
    banks, obank = ea.Banks(), g2.MaskBank(eo.OBANK_ROOT)
    ctx = (b3, bc, ea, eo, g2, banks, obank, lay)
    rows, t0, k = [], time.time(), 0

    def flush():
        pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)

    for iid in ids:
        if iid in done:
            continue
        if args.limit and k >= args.limit:
            break
        c = (cov.get(lay.loc[iid, "group"], float(fac.loc[iid, "c"]))) if args.mode == "phys" else 1.0
        if args.mode == "phys" and args.measured and lay.loc[iid, "group"] in (T12, LUM) \
                and np.isfinite(fac.loc[iid, "c_meas"]):
            c = float(fac.loc[iid, "c_meas"])       # factor(mode="measured"): the raw per-image detector value
        r = measure_image(iid, c, ctx)
        r["aniso_c"] = c
        if args.mode == "phys":   # MT is vertical: pinned to the image-space value; re-measured value kept
            r["mt_mm_phys"] = r.get("mt_mm", np.nan)
            r["mt_mm"] = float(orig.loc[iid, "mt_mm"])
        rows.append(r)
        k += 1
        if k % 20 == 0:
            flush()
            print(f"  {args.mode}: {k} new, {time.time() - t0:.0f}s, rss {rss_gb():.2f} GB", flush=True)
    flush()
    banks.close()
    obank.close()
    print(f"done {path}: {k} new rows in {time.time() - t0:.0f}s, rss {rss_gb():.2f} GB", flush=True)


# ============================================================================== subcommand: build
VARIANTS = {"C1ID": None, "LUM": (LUM,), "SIE": (SIE,), "T12": (T12,), "ALL": (LUM, SIE, T12)}


def assemble_cache(variant: str, tag: str = "") -> Path:
    """A V3_CACHE dir: the benchmark caches (links to the originals) + a test cache with the variant's rows."""
    d = OUT / (f"v3cache_{variant}" + (f"_{tag}" if tag else ""))
    d.mkdir(exist_ok=True)
    for p in V3_CACHE_ORIG.glob("*.csv"):
        if p.name.startswith("test__"):
            continue
        q = d / p.name
        if not q.exists():
            q.symlink_to(p)
    orig = pd.read_csv(V3_CACHE_ORIG / "test__v2.csv")
    if variant == "C1ID":
        new = pd.read_csv(OUT / "rows_c1/test__v2_rows.csv")
        assert len(new) == len(orig) and set(new.image_id) == set(orig.image_id)
        T = new.drop(columns=["aniso_c"])
        T = T.set_index("image_id").loc[orig.image_id].reset_index()
    else:
        new = pd.read_csv(OUT / ("rows_phys" + (f"_{tag}" if tag else "")) / "test__v2_rows.csv").set_index("image_id")
        T = orig.set_index("image_id").copy()
        ids = [i for i in T.index if T.loc[i, "group"] in VARIANTS[variant]]
        miss = set(ids) - set(new.index)
        assert not miss, f"missing physical rows: {sorted(miss)[:5]}"
        cols = [c for c in T.columns if c in new.columns]
        for c in ("aniso_c", "mt_mm_phys"):
            if c not in T.columns:
                T[c] = np.nan
        for c in cols + ["aniso_c", "mt_mm_phys"]:
            if T[c].dtype != object and new[c].dtype == object:
                T[c] = T[c].astype(object)
            T.loc[ids, c] = new.loc[ids, c].values
        T = T.reset_index()
    T.to_csv(d / "test__v2.csv", index=False)
    return d


def cmd_build_one(args) -> None:
    """Child process (V3_CACHE set by the parent): C12 recipe -> C18 (Telemed FL lambda 0) from that cache."""
    for p in (str(ROOT / "scripts"), str(ROOT / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import build_candidates_v7 as b7
    assert str(b7.b3.CACHE) == os.environ["V3_CACHE"], (b7.b3.CACHE, os.environ["V3_CACHE"])
    sub = b7.c18_sub()
    ts = b7.test_state()
    d = Path(os.environ["V3_CACHE"])
    sub.to_csv(d / "sub_full.csv")
    P = ts["P"]
    keep = [c for c in ("group", "clip", "pa_deg", "fl_mm", "mt_mm", "confidence", "flags", "pa_eb", "fl_eb",
                        "w_pa", "w_fl", "mu_fl", "tau_fl", "c_fl", "sigma_fl", "sigma_pa", "cell", "pa_cell",
                        "eb_unit", "pa_deep_raw", "pa_deep_c", "aniso_c", "mt_mm_phys") if c in P.columns]
    P[keep].to_csv(d / "per_image.csv")
    json.dump(dict(offsets=ts["off"], rss_gb=rss_gb()), open(d / "build_info.json", "w"), indent=1, default=float)
    print(f"built {d.name}: rss {rss_gb():.2f} GB")


def read_sub(p: Path) -> pd.DataFrame:
    return pd.read_csv(p).set_index("image_id")


def write_from_c18(sub: pd.DataFrame, ids, path: Path) -> None:
    """C18 file with the rows `ids` replaced by sub's values (4 dp, kept 1e-3 inside the bounds); every other
    line stays byte-identical to C18."""
    B = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
    ids = set(ids)
    lines = C18_FILE.read_bytes().decode().split("\r\n")
    out = [lines[0]]
    for ln in lines[1:]:
        if not ln:
            continue
        iid = ln.split(",", 1)[0]
        if iid in ids:
            r = sub.loc[iid]
            v = [float(np.clip(r[c], B[c][0] + 1e-3, B[c][1] - 1e-3)) for c in C3]
            ln = f"{iid},{v[0]:.4f},{v[1]:.4f},{v[2]:.4f}"
        out.append(ln)
    path.write_bytes(("\r\n".join(out) + "\r\n").encode())


def validate(path: Path) -> str:
    r = subprocess.run([str(PY), str(ROOT / "scripts/validate_submission.py"), str(path), "--test-dir", str(TEST_DIR)],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    return (r.stdout + r.stderr).strip()


def file_checks(path: Path, groups) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_candidates as bc
    lay = bc.test_layout()
    X, C = read_sub(path), read_sub(C18_FILE)
    g = lay.loc[X.index, "group"]
    tgt = g.isin(groups) if groups else pd.Series(True, index=X.index)
    ch = (X[list(C3)] - C[list(C3)]).abs().max(axis=1) > 0
    tel = lay.loc[X.index, "clip"] >= 0
    clip_const = bool(all(X.loc[tel, c].groupby(lay.loc[X.index[tel], "clip"]).nunique().max() == 1 for c in C3))
    at_bound = int(sum(((X[c] <= lo) | (X[c] >= hi)).sum() for c, (lo, hi) in bc.BOUNDS.items()))
    return dict(validate=validate(path), rows_changed=int(ch.sum()), rows_changed_outside_group=int((ch & ~tgt).sum()),
                rows_in_group=int(tgt.sum()), clips_constant=clip_const, at_bound=at_bound,
                mt_changed=int(((X.mt_mm - C.mt_mm).abs() > 0).sum()))


def dist(A: pd.DataFrame, Bf: pd.DataFrame, m=None) -> dict:
    """Official-metric distance contributions (sum over the masked rows / 309 / tau / 3)."""
    Bf = Bf.reindex(A.index)
    m = np.ones(len(A), bool) if m is None else np.asarray(m, bool)
    d = {k: float((A[k] - Bf[k]).abs()[m].sum() / len(A) / t / 3) for k, t in TAU.items()}
    d["S"] = sum(d.values())
    return d


def cmd_build(args) -> None:
    rep = {}
    tag = args.tag
    for v in args.variants:
        d = assemble_cache(v, tag)
        env = dict(os.environ, V3_CACHE=str(d), V4_CACHE=str(OUT / "v4cache_empty"), PYTHONPATH=str(ROOT / "src"),
                   PYTHONDONTWRITEBYTECODE="1")
        (OUT / "v4cache_empty").mkdir(exist_ok=True)
        t0 = time.time()
        r = subprocess.run([str(PY), __file__, "build-one"], env=env, capture_output=True, text=True)
        print(r.stdout.strip()[-400:], r.stderr.strip()[-2000:], f"({time.time() - t0:.0f}s)")
        if r.returncode:
            raise SystemExit(f"build-one {v} failed")
        sub = pd.read_csv(d / "sub_full.csv", index_col=0)
        C = read_sub(C18_FILE)
        if v == "C1ID":
            diff = (sub[list(C3)] - C.loc[sub.index, list(C3)]).abs()
            rep[v] = dict(maxabs={c: float(diff[c].max()) for c in C3}, dist_vs_C18=dist(sub, C)["S"])
            cache_o = pd.read_csv(V3_CACHE_ORIG / "test__v2.csv").set_index("image_id")
            cache_n = pd.read_csv(d / "test__v2.csv").set_index("image_id").loc[cache_o.index]
            num = [c for c in cache_o.columns if pd.api.types.is_numeric_dtype(cache_o[c]) and c in cache_n
                   and not pd.api.types.is_bool_dtype(cache_o[c])]
            nonnum = [c for c in cache_o.columns if c not in num and c in cache_n]
            dd = (cache_n[num] - cache_o[num]).abs()
            same_nan = (cache_n[num].isna() == cache_o[num].isna()).all().all()
            rep[v]["cache_identity"] = dict(n_cols=len(num), cols_missing=[c for c in cache_o.columns if c not in cache_n],
                                            max_abs=float(np.nanmax(dd.to_numpy())), nan_pattern_equal=bool(same_nan),
                                            nonnum_equal={c: bool((cache_n[c].astype(object).where(cache_n[c].notna(), "").astype(str) == cache_o[c].astype(object).where(cache_o[c].notna(), "").astype(str)).all())
                                                          for c in nonnum})
            p = OUT / "D_C18_rebuilt_c1.csv"
            write_from_c18(sub, sub.index, p)
            rep[v]["file"] = str(p)
            rep[v]["file_vs_C18_maxabs"] = float((read_sub(p)[list(C3)] - C[list(C3)]).abs().max().max())
            rep[v]["file_validate"] = validate(p)
        else:
            groups = VARIANTS[v]
            sys.path.insert(0, str(ROOT / "scripts"))
            import build_candidates as bc
            lay = bc.test_layout()
            ids = [i for i in sub.index if lay.loc[i, "group"] in groups]
            # leak check: the other groups' rows of the rebuilt submission must equal C18 (to CSV rounding)
            oth = [i for i in sub.index if i not in set(ids)]
            leak = float((sub.loc[oth, list(C3)] - C.loc[oth, list(C3)]).abs().max().max())
            p = OUT / (f"P_{v}" + (f"_{tag}" if tag else "") + ".csv")
            write_from_c18(sub, ids, p)
            rep[v] = dict(file=str(p), other_rows_maxabs_vs_C18_before_pinning=leak, **file_checks(p, groups))
            X = read_sub(p)
            m = lay.loc[X.index, "group"].isin(groups).to_numpy()
            rep[v]["dist_vs_C18"] = dist(X, C)
            rep[v]["mean_dPA"] = float((X.pa_deg - C.pa_deg)[m].mean())
            rep[v]["mean_dFL"] = float((X.fl_mm - C.fl_mm)[m].mean())
            rep[v]["mean_relFL"] = float((X.fl_mm / C.fl_mm)[m].mean())
        if tag:
            rep[f"{v}_{tag}"] = rep.pop(v)
        print(v, tag, json.dumps(rep.get(v, rep.get(f"{v}_{tag}")), default=float)[:900])
    prev = json.loads((OUT / "build_report.json").read_text()) if (OUT / "build_report.json").exists() else {}
    prev.update(rep)
    (OUT / "build_report.json").write_text(json.dumps(prev, indent=1, default=float))


# ============================================================================== subcommand: compare
def cmd_compare(args) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_candidates as bc
    lay = bc.test_layout()
    C = read_sub(C18_FILE)
    J = {k: read_sub(JDIR / f) for k, f in J_FILES.items()}
    JALL = C.copy()
    for k, g in PHYS_GROUPS.items():
        m = lay.loc[C.index, "group"] == g
        JALL.loc[m] = J[k].loc[m]
    J["ALL"] = JALL
    rows = []
    for v in ("LUM", "SIE", "T12", "ALL"):
        P = read_sub(OUT / f"P_{v}.csv")
        Jv = J[v]
        grp = lay.loc[P.index, "group"]
        for g in list(grp.unique()) + ["ALL_ROWS"]:
            m = (grp == g).to_numpy() if g != "ALL_ROWS" else np.ones(len(P), bool)
            dpj, dpc, djc = dist(P, Jv, m), dist(P, C, m), dist(Jv, C, m)
            for comp in C3:
                a, b = (P[comp] - C[comp])[m], (Jv[comp] - C[comp])[m]
                rows.append(dict(file=f"P_{v}", vs=J_FILES.get(v, "J1+J2+J3"), group=g, comp=comp, n=int(m.sum()),
                                 dist_P_J=dpj[comp], dist_P_C18=dpc[comp], dist_J_C18=djc[comp],
                                 mean_P_minus_C18=float(a.mean()), mean_J_minus_C18=float(b.mean()),
                                 mean_P_minus_J=float((P[comp] - Jv[comp])[m].mean()),
                                 maxabs_P_minus_J=float((P[comp] - Jv[comp])[m].abs().max()),
                                 corr_dP_dJ=float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan))
            rows.append(dict(file=f"P_{v}", vs=J_FILES.get(v, "J1+J2+J3"), group=g, comp="S", n=int(m.sum()),
                             dist_P_J=dpj["S"], dist_P_C18=dpc["S"], dist_J_C18=djc["S"]))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "compare_to_J.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[T.group.isin(["ALL_ROWS"]) | (T.comp == "S")].round(5).to_string())
    # union check: P_ALL == P_LUM + P_SIE + P_T12 cells
    PA = read_sub(OUT / "P_ALL.csv")
    U = C.copy()
    for v, g in PHYS_GROUPS.items():
        m = lay.loc[C.index, "group"] == g
        U.loc[m] = read_sub(OUT / f"P_{v}.csv").loc[m]
    print("P_ALL vs union of single-group P files, max abs:", float((PA - U).abs().max().max()))


# ============================================================================== subcommand: fb0-check
def cmd_fb0_check(args) -> None:
    """Per-image agreement of this detector with the FB0 audit (v8/audit_J/factors_by_image.csv, every method
    it reports) and with the review8 scan (v8/review/factors_review.csv): share of the group within +-0.005."""
    F = pd.read_csv(OUT / "factors_test.csv").set_index("image_id")
    refs = []
    p0 = SCR / "v8/audit_J/factors_by_image.csv"
    if p0.exists():
        B = pd.read_csv(p0)
        for m, d in B.groupby("method"):
            refs.append(("FB0:" + m.split(":")[0], d.set_index("image_id")["c_measured"]))
    p1 = SCR / "v8/review/factors_review.csv"
    if p1.exists():
        R = pd.read_csv(p1).set_index("image_id")
        refs.append(("review8:" + "ticks|P-disc moments", R["c"]))
    rows = []
    for name, ref in refs:
        j = F.join(ref.rename("c_ref"), how="inner")
        for g, d in j.groupby("group"):
            d = d[d.c_ref.notna()]
            if not len(d):
                continue
            for col in ("c_meas", "c"):
                x = d[col] - d.c_ref
                if x.isna().all():
                    continue
                rows.append(dict(reference=name, group=g, ours=col, n=len(d), n_ref_group=int(len(j[j.group == g])),
                                 mean_diff=float(x.mean()), max_abs_diff=float(x.abs().max()),
                                 share_within_0p005=float((x.abs() <= 0.005).mean()),
                                 ref_median=float(d.c_ref.median()), ours_median=float(d[col].median())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "factors_vs_fb0.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())


# ============================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("factors")
    sp.add_parser("bench-factors")
    m = sp.add_parser("measure")
    m.add_argument("--mode", choices=["phys", "c1"], required=True)
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--only", nargs="*", choices=list(PHYS_GROUPS), default=None)
    m.add_argument("--c", nargs="*", default=None, help="factor overrides, e.g. LUM=0.8374")
    m.add_argument("--tag", default="")
    m.add_argument("--measured", action="store_true", help="per-image detector values instead of the table")
    b = sp.add_parser("build")
    b.add_argument("variants", nargs="*", default=["C1ID", "LUM", "SIE", "T12", "ALL"])
    b.add_argument("--tag", default="")
    sp.add_parser("build-one")
    sp.add_parser("compare")
    sp.add_parser("fb0-check")
    a = ap.parse_args()
    {"factors": cmd_factors, "bench-factors": cmd_bench_factors, "measure": cmd_measure, "build": cmd_build,
     "build-one": cmd_build_one, "compare": cmd_compare, "fb0-check": cmd_fb0_check}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
