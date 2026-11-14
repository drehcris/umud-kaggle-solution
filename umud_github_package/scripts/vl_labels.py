"""Parse the burned-in expert lines of the UMUD 'Example Annotated Images' (VL, Esaote) into
per-image expert FL (mm), PA (deg) and an apo-line MT (mm).

label_masks/<key>.png : 0/255 mask of the pure-red ImageJ lines in ANNOTATED-canvas coordinates.
Lines: superficial aponeurosis (extended), deep aponeurosis (extended), and fascicles drawn as
segmented lines (visible part traced + straight extrapolation to the superficial apo extension).
Scale: ann_xres (px/cm) from vl_index.csv; raw = canvas - (offset_x, offset_y).

Method (one image at a time, light):
 1. apo lines: Hough candidates (near-horizontal, long); superficial = the long line with ~all
    other label pixels below it, deep = the long line with ~all other pixels above it.
 2. fascicle pixels = label pixels > 2.5 px from both apo lines; column-wise run centres are
    tracked left->right with linear prediction (tracks may share a run where lines converge).
 3. each track is smoothed, extended linearly at both ends to the two apo lines, FL = arc length
    between the intersections; PA = angle between the deep-end direction (last 12 % of the track,
    >= 40 px) and the deep apo line. MT_lines = vertical distance of the two lines at 25/50/75 %
    of the raw-image ultrasound viewport.

Writes artifacts/candidates/bench/vl_labels_raw.csv (this parser's output) and artifacts/candidates/bench/vl_labels.csv
(the same table read and re-written by pandas, which is how the file the pipeline reads was first made), and with
--overlays a few overlay PNGs for a visual check. Input: artifacts/maskbank/vl_data (images, annotated images, label
masks and vl_index.csv, extracted from the OSF xbawc "Example Annotated Images", CC BY 4.0).

    python scripts/vl_labels.py [all | KEY ...] [--overlays]

Provenance: written for this project (research step "integrate", 2026-10-21); moved here unchanged except for the
repository-relative paths and the output location. GPL-3.0-only.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umud import geometry_v2 as g2  # noqa: E402

VL = ROOT / "artifacts/maskbank/vl_data"
OUT = ROOT / "artifacts/candidates/bench"


def line_from_pts(xs, ys):
    """Total least squares line through points -> (point, unit direction)."""
    P = np.c_[xs, ys].astype(float)
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    d = vt[0]
    if d[0] < 0:
        d = -d
    return c, d


def dist_to_line(xs, ys, c, d):
    n = np.array([-d[1], d[0]])
    return (np.c_[xs, ys] - c) @ n  # signed (positive = below for d pointing right, y down)


def y_on(c, d, x):
    return c[1] + (np.asarray(x, float) - c[0]) * d[1] / d[0]


def find_apo_lines(m: np.ndarray):
    ys, xs = np.nonzero(m)
    lines = cv2.HoughLines((m > 0).astype(np.uint8), 1, np.pi / 720, 120, min_theta=np.radians(55),
                           max_theta=np.radians(125))
    if lines is None:
        return None
    cands = []
    for rho, th in lines[:80, 0]:
        # normal form x cos th + y sin th = rho
        dist = xs * math.cos(th) + ys * math.sin(th) - rho
        inl = np.abs(dist) < 1.5
        if inl.sum() < 120:
            continue
        c, d = line_from_pts(xs[inl], ys[inl])
        # refine once
        dd = dist_to_line(xs, ys, c, d)
        inl = np.abs(dd) < 1.5
        c, d = line_from_pts(xs[inl], ys[inl])
        dd = dist_to_line(xs, ys, c, d)
        inl = np.abs(dd) < 1.5
        # longest contiguous run of inliers along the line
        t = np.sort((np.c_[xs[inl], ys[inl]] - c) @ d)
        gaps = np.flatnonzero(np.diff(t) > 6)
        segs = np.split(t, gaps + 1)
        seg = max(segs, key=lambda s: s[-1] - s[0])
        length = seg[-1] - seg[0]
        out = ~inl
        below = float(np.mean(dd[out] > 2.0)) if out.any() else 0.0
        above = float(np.mean(dd[out] < -2.0)) if out.any() else 0.0
        cands.append(dict(c=c, d=d, length=length, n=int(inl.sum()), below=below, above=above,
                          ang=math.degrees(math.atan2(d[1], d[0]))))
    if not cands:
        return None
    # dedupe near-identical lines
    uniq = []
    for q in sorted(cands, key=lambda q: -q["n"]):
        if any(abs(q["ang"] - u["ang"]) < 1.0 and abs(y_on(q["c"], q["d"], u["c"][0]) - u["c"][1]) < 3 for u in uniq):
            continue
        uniq.append(q)
    longs = [q for q in uniq if q["length"] > 150]
    # superficial: (nearly) every other label pixel lies below it; deep: above it. Rank by the
    # one-sidedness margin first (a fascicle line always has some pixels on both sides), then length.
    sup = [q for q in longs if q["below"] - q["above"] > 0.5]
    deep = [q for q in longs if q["above"] - q["below"] > 0.5]
    if not sup or not deep:
        return dict(fail="no_apo_lines", cands=uniq)
    ms = max(q["below"] - q["above"] for q in sup)
    md = max(q["above"] - q["below"] for q in deep)
    s = max([q for q in sup if q["below"] - q["above"] >= ms - 0.01], key=lambda q: q["length"])
    d_ = max([q for q in deep if q["above"] - q["below"] >= md - 0.01], key=lambda q: q["length"])
    if y_on(s["c"], s["d"], 0.5 * (s["c"][0] + d_["c"][0])) >= y_on(d_["c"], d_["d"], 0.5 * (s["c"][0] + d_["c"][0])):
        return dict(fail="apo_order", cands=uniq)
    return dict(sup=s, deep=d_, cands=uniq)


def column_tracks(F: np.ndarray, tol: float = 2.5, max_gap: int = 6):
    H, W = F.shape
    xs_any = np.flatnonzero(F.any(0))
    tracks = []  # each: dict(x=[], y=[], last=x)
    active = []
    for x in xs_any:
        col = F[:, x]
        idx = np.flatnonzero(col)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        cs = np.array([r.mean() for r in runs])
        claimed = np.zeros(len(cs), bool)
        still = []
        for t in active:
            if x - t["x"][-1] > max_gap:
                continue
            k = min(10, len(t["x"]))
            if k >= 3:
                s, b = np.polyfit(t["x"][-k:], t["y"][-k:], 1)
                pred = s * x + b
            else:
                pred = t["y"][-1]
            j = int(np.argmin(np.abs(cs - pred)))
            if abs(cs[j] - pred) <= tol + 0.3 * (x - t["x"][-1]):
                t["x"].append(x)
                t["y"].append(cs[j])
                claimed[j] = True
            still.append(t)
        for j in np.flatnonzero(~claimed):
            t = dict(x=[x], y=[cs[j]])
            tracks.append(t)
            still.append(t)
        active = still
    return [t for t in tracks if len(t["x"]) >= 25]


def intersect(c, d, p, q):
    """Intersection of line (c, d) with the line through p with direction q -> point or None."""
    A = np.array([d, -q]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t, s = np.linalg.solve(A, np.asarray(p) - np.asarray(c))
    return np.asarray(c) + t * np.asarray(d)


def parse(key: str, row: pd.Series, raw_img: np.ndarray | None = None) -> dict:
    m = np.asarray(Image.open(VL / "label_masks" / f"{key}.png")) > 0
    res = dict(key=key)
    al = find_apo_lines(m)
    if al is None or "fail" in al:
        res["fail"] = "no_apo_lines"
        return res
    S, D = al["sup"], al["deep"]
    ys, xs = np.nonzero(m)
    ds = dist_to_line(xs, ys, S["c"], S["d"])
    dd = dist_to_line(xs, ys, D["c"], D["d"])
    keep = (np.abs(ds) > 2.5) & (np.abs(dd) > 2.5) & (ds > 0) & (dd < 0)  # strictly between the lines
    F = np.zeros_like(m)
    F[ys[keep], xs[keep]] = True
    tr = column_tracks(F)
    fas = []
    for t in tr:
        x = np.asarray(t["x"], float)
        y = np.asarray(t["y"], float)
        # smooth y with a moving average (window 9) on a dense x grid
        xg = np.arange(x.min(), x.max() + 1)
        yg = np.interp(xg, x, y)
        k = 9
        if len(yg) > k:
            ker = np.ones(k) / k
            ypad = np.r_[np.full(k // 2, yg[0]), yg, np.full(k // 2, yg[-1])]
            ys_ = np.convolve(ypad, ker, mode="valid")
        else:
            ys_ = yg
        n = len(xg)
        e = max(40, int(0.12 * n))
        e = min(e, n)
        cl, dl = line_from_pts(xg[:e], ys_[:e])
        cr, dr = line_from_pts(xg[-e:], ys_[-e:])
        pL, pR = np.array([xg[0], ys_[0]]), np.array([xg[-1], ys_[-1]])
        # which end is near which apo line?
        dLs = abs(dist_to_line([pL[0]], [pL[1]], S["c"], S["d"])[0])
        dLd = abs(dist_to_line([pL[0]], [pL[1]], D["c"], D["d"])[0])
        dRs = abs(dist_to_line([pR[0]], [pR[1]], S["c"], S["d"])[0])
        dRd = abs(dist_to_line([pR[0]], [pR[1]], D["c"], D["d"])[0])
        if dLs + dRd <= dLd + dRs:
            sup_end, sup_dir, deep_end, deep_dir, gap_s, gap_d = pL, dl, pR, dr, dLs, dRd
            deep_side = "right"
        else:
            sup_end, sup_dir, deep_end, deep_dir, gap_s, gap_d = pR, dr, pL, dl, dRs, dLd
            deep_side = "left"
        P_s = intersect(S["c"], S["d"], sup_end, sup_dir)
        P_d = intersect(D["c"], D["d"], deep_end, deep_dir)
        if P_s is None or P_d is None:
            continue
        arc = float(np.sum(np.hypot(np.diff(xg), np.diff(ys_))))
        ext = float(np.hypot(*(P_s - sup_end)) + np.hypot(*(P_d - deep_end)))
        fl_px = arc + ext
        chord = float(np.hypot(*(P_s - P_d)))
        ang_f = math.degrees(math.atan2(deep_dir[1], deep_dir[0]))
        ang_d = math.degrees(math.atan2(D["d"][1], D["d"][0]))
        pa = abs(float(g2.wrap90(ang_f - ang_d)))
        ang_fc = math.degrees(math.atan2(P_s[1] - P_d[1], P_s[0] - P_d[0]))
        pa_chord = abs(float(g2.wrap90(ang_fc - ang_d)))
        fas.append(dict(x0=xg[0], x1=xg[-1], n=n, fl_px=fl_px, chord_px=chord, arc_px=arc, ext_px=ext,
                        gap_s=gap_s, gap_d=gap_d, pa=pa, pa_chord=pa_chord, deep_side=deep_side,
                        ps=P_s, pdp=P_d))
    # keep tracks that span a good part of the sup->deep distance and end near both lines
    good = [f for f in fas if f["gap_s"] < 25 and f["gap_d"] < 25 and f["arc_px"] > 0.35 * f["fl_px"]
            and f["ext_px"] < 0.12 * f["fl_px"]]
    # dedupe tracks hitting the same two intersection points (shared converged runs)
    uniq = []
    for f in sorted(good, key=lambda f: -f["arc_px"]):
        if any(np.hypot(*(f["ps"] - u["ps"])) < 6 and np.hypot(*(f["pdp"] - u["pdp"])) < 6 for u in uniq):
            continue
        uniq.append(f)
    ppc = float(row["ann_xres"])
    res.update(n_tracks=len(tr), n_fasc=len(uniq), sup_ang=S["ang"], deep_ang=D["ang"], sup_len=S["length"],
               deep_len=D["length"], sup_below=S["below"], deep_above=D["above"])
    if uniq:
        res["fl_mm"] = float(np.mean([f["fl_px"] for f in uniq])) * 10 / ppc
        res["fl_chord_mm"] = float(np.mean([f["chord_px"] for f in uniq])) * 10 / ppc
        res["pa_deg"] = float(np.mean([f["pa"] for f in uniq]))
        res["pa_chord_deg"] = float(np.mean([f["pa_chord"] for f in uniq]))
        res["fl_sd_mm"] = float(np.std([f["fl_px"] for f in uniq])) * 10 / ppc
        res["pa_sd"] = float(np.std([f["pa"] for f in uniq]))
        res["ext_frac"] = float(np.mean([f["ext_px"] / f["fl_px"] for f in uniq]))
        res["deep_side"] = uniq[0]["deep_side"]
    # MT from the expert lines at 25/50/75 % of the raw viewport
    ox, oy = float(row["offset_x"]), float(row["offset_y"])
    if raw_img is not None:
        vx0, vx1 = g2.estimate_viewport_x(raw_img)
    else:
        vx0, vx1 = 0, int(row["raw_width"])
    xq = ox + vx0 + (vx1 - vx0) * np.array([0.25, 0.5, 0.75])
    mt_px = float(np.mean(y_on(D["c"], D["d"], xq) - y_on(S["c"], S["d"], xq)))
    res["mt_lines_mm"] = mt_px * 10 / ppc
    res["vx0"], res["vx1"] = vx0, vx1
    res["_fas"] = uniq
    res["_S"], res["_D"] = S, D
    return res


def overlay(key, res, path):
    a = np.asarray(Image.open(VL / "annotated" / f"{key}.png").convert("RGB")).copy()
    H, W = a.shape[:2]
    for L, col in ((res["_S"], (0, 255, 255)), (res["_D"], (255, 0, 255))):
        x0, x1 = 0, W - 1
        cv2.line(a, (x0, int(round(y_on(L["c"], L["d"], x0)))), (x1, int(round(y_on(L["c"], L["d"], x1)))), col, 1)
    for f in res["_fas"]:
        cv2.circle(a, tuple(int(v) for v in f["ps"]), 6, (0, 255, 0), 2)
        cv2.circle(a, tuple(int(v) for v in f["pdp"]), 6, (255, 255, 0), 2)
    cv2.putText(a, f"n={res.get('n_fasc')} FL={res.get('fl_mm', np.nan):.1f} PA={res.get('pa_deg', np.nan):.1f} "
                   f"MTl={res.get('mt_lines_mm', np.nan):.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    Image.fromarray(a).save(path)


def main():
    idx = pd.read_csv(VL / "vl_index.csv")
    args = [a for a in sys.argv[1:] if a != "--overlays"]
    want_overlays = "--overlays" in sys.argv[1:]
    only = args if args and args[0] != "all" else None
    rows = []
    ov = OUT / "vl_overlays"
    if want_overlays:
        ov.mkdir(parents=True, exist_ok=True)
    for k, row in idx.iterrows():
        key = row["key"]
        if only and key not in only and not any(o in key for o in only):
            continue
        raw = np.asarray(Image.open(VL / "images" / f"{key}.png").convert("L"))
        r = parse(key, row, raw)
        if want_overlays and "_fas" in r and (only or k % 25 == 0 or r.get("n_fasc", 0) < 2 or r.get("n_fasc", 0) > 5):
            overlay(key, r, ov / f"{key}.png")
        rows.append({kk: vv for kk, vv in r.items() if not kk.startswith("_")})
    D = pd.DataFrame(rows)
    if not only:
        OUT.mkdir(parents=True, exist_ok=True)
        D.to_csv(OUT / "vl_labels_raw.csv", index=False)
        pd.read_csv(OUT / "vl_labels_raw.csv").to_csv(OUT / "vl_labels.csv", index=False)
    pd.set_option("display.width", 250)
    print(D.drop(columns=[c for c in D.columns if c in ("key",)]).describe().round(2).T.to_string())
    print(D.n_fasc.value_counts().sort_index().to_string())
    if only:
        print(D.round(2).to_string())


if __name__ == "__main__":
    main()
