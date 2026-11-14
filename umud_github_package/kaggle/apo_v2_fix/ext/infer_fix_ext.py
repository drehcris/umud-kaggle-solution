"""UMUD aponeurosis v2: inner-edge-accurate aponeurosis segmentation robust to unseen scanners.

Kaggle GPU kernel (shivateja005/umud-apo-v2).  Single file, self-contained.

* Data: 1048 apo image/mask pairs (exact dedupe; 5-fold GroupKFold on near-duplicate groups,
  64x48 central-crop thumbnail correlation > 0.95).
* Input: generic texture-based viewport crop -> aspect-preserving resize + centred letterbox to
  512x512 (never squashed); the transform is kept so outputs map back to native pixels exactly.
* Targets (3 ch): band mask, upper-edge line, lower-edge line of every band (2 px lines that
  straddle the boundary: upper = rows ymin-1, ymin; lower = rows ymax, ymax+1).
* Model: smp Unet with timm ConvNeXt-S (convnext_small.fb_in22k_ft_in1k) + a full-resolution
  image stem fused before the head (sub-pixel edges; ConvNeXt has no stride-2 features).
* Heavy domain-shift augmentation (narrow-FOV crops, speckle, TGC, JPEG, blur, gamma...).
* Validation every epoch with an inner-edge MT geometry metric at native resolution.
* Outputs: fold checkpoints, config/metrics, MASKBANK npz (test/static/dynamic/vl/train_oof).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)  # fork safety + no CPU oversubscription in DataLoader workers

# --------------------------------------------------------------------------------------
# Run mode: "smoke" (2 folds in parallel, 2 epochs, 40 images, 5 images per inference set)
#           "full"  (5 folds, EPOCHS epochs, all images)
# --------------------------------------------------------------------------------------
RUN_MODE = "full"

SEED = 20261021
SIZE = 512
N_FOLDS = 5
EPOCHS = 40 if RUN_MODE == "full" else 2
BATCH = 8
LR_ENC = 2.0e-4
LR_DEC = 1.0e-3
WEIGHT_DECAY = 1.0e-2
WARMUP_EPOCHS = 1.0
EMA_DECAY = 0.997
NUM_WORKERS = 2
GROUP_CORR = 0.95
ENCODER = "tu-convnext_small.fb_in22k_ft_in1k"
ENCODER_FALLBACKS = ["tu-convnext_small", "tu-tf_efficientnetv2_s.in21k_ft_in1k", "resnet34"]
MODEL_NAME = "apo_v2"
SMOKE_IMAGES = 40
SMOKE_FOLDS = [0, 1]
SMOKE_INFER = 5
# wall-clock guard (seconds from start) after which no new epoch is started
TRAIN_DEADLINE = 3.75 * 3600 if RUN_MODE == "full" else 1800
EXT_URLS = {
    "static": "https://osf.io/download/hb5xj/",
    "dynamic": "https://osf.io/download/vqb7h/",
    "vl_young": "https://osf.io/download/zcuwk/",
    "vl_old": "https://osf.io/download/6ga94/",
}
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
COMP_ROOT = Path("/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data")
WORK = Path("/kaggle/working")
OUT = WORK / MODEL_NAME
BANK = WORK / "maskbank"
EXT_DIR = Path("/tmp/umud_ext")
START = time.time()
TAG = ""


def log(msg: str) -> None:
    line = f"{TAG}[{(time.time() - START) / 60:6.1f}m] {msg}"
    print(line, flush=True)
    try:
        with open(OUT / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Image IO
# --------------------------------------------------------------------------------------
def read_gray(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im, dtype=np.float32)
        a = (a - a.min()) / max(1.0, float(a.max() - a.min())) * 255.0
        return a.astype(np.uint8)
    return np.asarray(im.convert("L"), dtype=np.uint8)


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    h, w = shape
    if m.shape != (h, w):  # masks are at original resolution: plain stretch is the alignment
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return (m > 127).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Viewport detection (generic speckle-texture density; validated on all test layouts)
# --------------------------------------------------------------------------------------
def _runs(flags):
    padded = np.pad(flags.astype(np.int8), 1)
    d = np.diff(padded)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _close_1d(flags, gap):
    flags = flags.copy()
    starts, ends = _runs(~flags)
    for s, e in zip(starts, ends):
        if s > 0 and e < len(flags) and (e - s) <= gap:
            flags[s:e] = True
    return flags


def _largest_run(flags, weights=None):
    starts, ends = _runs(flags)
    if not len(starts):
        return None
    if weights is None:
        score = ends - starts
    else:
        cs = np.concatenate([[0.0], np.cumsum(weights)])
        score = cs[ends] - cs[starts]
    i = int(np.argmax(score))
    return int(starts[i]), int(ends[i])


def detect_viewport(gray: np.ndarray) -> tuple[int, int, int, int]:
    """(x1, y1, x2, y2), x2/y2 exclusive; full frame on failure."""
    h, w = gray.shape
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = np.abs(g[:, 2:] - g[:, :-2])
    gy[1:-1, :] = np.abs(g[2:, :] - g[:-2, :])
    k = max(5, int(round(min(h, w) / 80)) | 1)
    tex = cv2.blur(gx + gy, (k, k))
    active = tex > 2.5
    col = active.mean(axis=0)
    ref = np.percentile(col, 99)
    if ref <= 0.05:
        return 0, 0, w, h
    run = _largest_run(_close_1d(col > 0.3 * ref, max(3, w // 100)), col)
    if run is None:
        return 0, 0, w, h
    x1, x2 = run
    row = active[:, x1:x2].mean(axis=1)
    rref = np.percentile(row, 99)
    run = _largest_run(_close_1d(row > 0.15 * rref, max(5, h // 12)), row)
    y1, y2 = (0, h) if run is None else run
    col2 = active[y1:y2].mean(axis=0)
    ref2 = np.percentile(col2, 99)
    run2 = _largest_run(_close_1d(col2 > 0.3 * ref2, max(3, w // 100)), col2)
    if run2 is not None and (run2[1] - run2[0]) >= 0.5 * (x2 - x1):
        x1, x2 = run2
    weak = tex > 1.0
    wcol = weak[y1:y2].mean(axis=0)
    lim, e = int(0.12 * w), 0
    while x1 > 0 and e < lim and wcol[x1 - 1] > 0.5:
        x1 -= 1
        e += 1
    e = 0
    while x2 < w and e < lim and wcol[x2] > 0.5:
        x2 += 1
        e += 1
    wrow = weak[:, x1:x2].mean(axis=1)
    lim, e = int(0.12 * h), 0
    while y1 > 0 and e < lim and wrow[y1 - 1] > 0.5:
        y1 -= 1
        e += 1
    e = 0
    while y2 < h and e < lim and wrow[y2] > 0.5:
        y2 += 1
        e += 1
    m = k // 2
    my = int(round(0.02 * h))
    x1 = max(0, x1 + m - 2)
    x2 = min(w, x2 - m + 2)
    y1 = max(0, y1 + m - 2 - my)
    y2 = min(h, y2 - m + 2 + my)
    if (x2 - x1) < 0.08 * w or (y2 - y1) < 0.25 * h:
        return 0, 0, w, h
    return int(x1), int(y1), int(x2), int(y2)


def union_box(a, b, shape, margin=4):
    h, w = shape
    return (max(0, min(a[0], b[0] - margin)), max(0, min(a[1], b[1] - margin)),
            min(w, max(a[2], b[2] + margin)), min(h, max(a[3], b[3] + margin)))


# --------------------------------------------------------------------------------------
# Letterbox transform (centre-aligned pixel convention, exactly invertible)
# --------------------------------------------------------------------------------------
def letterbox(gray: np.ndarray, box, size: int = SIZE, mask: np.ndarray | None = None):
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    s = min(size / cw, size / ch)
    nw = int(max(1, min(size, round(cw * s))))
    nh = int(max(1, min(size, round(ch * s))))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas = np.zeros((size, size), np.uint8)
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(np.ascontiguousarray(gray[y1:y2, x1:x2]), (nw, nh),
                                                interpolation=interp)
    tf = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "sx": nw / cw, "sy": nh / ch,
          "ox": int(ox), "oy": int(oy), "H": int(gray.shape[0]), "W": int(gray.shape[1]),
          "nw": nw, "nh": nh}
    mcanvas = None
    if mask is not None:
        mcanvas = np.zeros((size, size), np.float32)
        mcanvas[oy:oy + nh, ox:ox + nw] = cv2.resize(
            np.ascontiguousarray(mask[y1:y2, x1:x2]).astype(np.float32), (nw, nh), interpolation=interp)
    return canvas, tf, mcanvas


def to_native(prob: np.ndarray, tf: dict) -> np.ndarray:
    """prob: (C, S, S) float32 in net space -> (C, H, W) float32 at native resolution."""
    sx, sy = tf["sx"], tf["sy"]
    M = np.array([[sx, 0.0, (0.5 - tf["x1"]) * sx - 0.5 + tf["ox"]],
                  [0.0, sy, (0.5 - tf["y1"]) * sy - 0.5 + tf["oy"]]], np.float64)
    out = np.zeros((prob.shape[0], tf["H"], tf["W"]), np.float32)
    for c in range(prob.shape[0]):
        o = cv2.warpAffine(np.ascontiguousarray(prob[c], dtype=np.float32), M, (tf["W"], tf["H"]),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[c, tf["y1"]:tf["y2"], tf["x1"]:tf["x2"]] = o[tf["y1"]:tf["y2"], tf["x1"]:tf["x2"]]
    return out


# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------
def edge_targets(mask: np.ndarray, min_area: int = 15):
    """Upper / lower edge lines (2 px, straddling the boundary) of every connected band."""
    h, w = mask.shape
    up = np.zeros((h, w), np.uint8)
    lo = np.zeros((h, w), np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, ww, hh, area = stats[k]
        if area < min_area:
            continue
        sub = lab[y:y + hh, x:x + ww] == k
        present = sub.any(axis=0)
        cols = np.flatnonzero(present)
        top = sub.argmax(axis=0)[cols] + y
        bot = (hh - 1 - sub[::-1].argmax(axis=0))[cols] + y
        xs = cols + x
        up[np.clip(top - 1, 0, h - 1), xs] = 1
        up[top, xs] = 1
        lo[bot, xs] = 1
        lo[np.clip(bot + 1, 0, h - 1), xs] = 1
    return up, lo


# --------------------------------------------------------------------------------------
# Geometry (inner-edge MT) used for validation
# --------------------------------------------------------------------------------------
def band_components(mask: np.ndarray, ref_width: float, min_frac: float = 0.15):
    m = mask.astype(np.uint8)
    W = m.shape[1]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for k in range(1, n):
        x, y, ww, hh, area = stats[k]
        if ww < max(8, min_frac * ref_width) or area < 30:
            continue
        sub = lab[y:y + hh, x:x + ww] == k
        present = sub.any(axis=0)
        t = sub.argmax(axis=0).astype(np.float32) + y
        b = (hh - 1 - sub[::-1].argmax(axis=0)).astype(np.float32) + y
        t[~present] = np.nan
        b[~present] = np.nan
        top = np.full(W, np.nan, np.float32)
        bot = np.full(W, np.nan, np.float32)
        top[x:x + ww] = t
        bot[x:x + ww] = b
        comps.append({"top": top, "bot": bot, "yc": float(np.nanmedian((t + b) / 2)),
                      "width": int(present.sum())})
    comps.sort(key=lambda c: c["yc"])
    return comps


def superficial_pair(comps):
    if len(comps) < 2:
        return None
    sup = comps[0]
    sp = ~np.isnan(sup["bot"])
    for c in comps[1:]:
        both = sp & ~np.isnan(c["top"])
        if both.sum() >= 0.2 * min(sup["width"], c["width"]):
            if float(np.median(c["top"][both] - sup["bot"][both])) > 2:
                return sup, c
    return None


def _refine(colprob: np.ndarray, y0: float, r: int) -> float:
    lo = max(0, int(math.floor(y0 - r)))
    hi = min(len(colprob), int(math.ceil(y0 + r)) + 1)
    p = colprob[lo:hi].astype(np.float32)
    p = np.where(p > 0.1, p, 0.0)
    if p.sum() < 0.3:
        return y0
    return float((p * np.arange(lo, hi)).sum() / p.sum())


def inner_mt(pair, low_prob=None, up_prob=None, r: int = 3):
    """Inner-edge vertical MT (px) at 25/50/75% of the shared x-range: (band, edge-refined)."""
    sup, deep = pair
    both = ~np.isnan(sup["bot"]) & ~np.isnan(deep["top"])
    both &= np.nan_to_num(deep["top"], nan=-1) > np.nan_to_num(sup["bot"], nan=1e9)
    xs = np.flatnonzero(both)
    if len(xs) < 5:
        return None, None
    xa, xb = int(xs[0]), int(xs[-1])
    hw = max(2, int(0.02 * (xb - xa)))
    vb, ve = [], []
    for f in (0.25, 0.5, 0.75):
        xc = int(round(xa + f * (xb - xa)))
        cols = [x for x in range(xc - hw, xc + hw + 1) if 0 <= x < len(both) and both[x]]
        if not cols:
            continue
        vb.append(float(np.median([deep["top"][x] - sup["bot"][x] - 1.0 for x in cols])))
        if low_prob is not None and up_prob is not None:
            ge = []
            for x in cols:
                yl = _refine(low_prob[:, x], sup["bot"][x] + 0.5, r)
                yu = _refine(up_prob[:, x], deep["top"][x] - 0.5, r)
                ge.append(yu - yl)
            ve.append(float(np.median(ge)))
    if not vb:
        return None, None
    return float(np.mean(vb)), (float(np.mean(ve)) if ve else None)


def gt_mt(mask: np.ndarray, box) -> float | None:
    comps = band_components(mask, box[2] - box[0])
    pair = superficial_pair(comps)
    if pair is None:
        return None
    return inner_mt(pair)[0]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def find_dir(root: Path, name: str) -> Path:
    matches = sorted(p for p in root.rglob(name) if p.is_dir())
    if not matches:
        raise RuntimeError(f"{name} not found under {root}")
    return matches[0]


def load_training_items():
    img_dir = find_dir(COMP_ROOT, "apo_images_new_model_v1")
    msk_dir = find_dir(COMP_ROOT, "apo_masks_new_model_v1")
    masks = {p.stem: p for p in msk_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    paths = sorted((p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.stem in masks),
                   key=lambda p: p.stem)
    def load_one(p):
        g = read_gray(p)
        m = read_mask(masks[p.stem], g.shape)
        key = hashlib.sha1(g.tobytes() + m.tobytes() + str(g.shape).encode()).hexdigest()
        vp = detect_viewport(g)
        ys, xs = np.nonzero(m)
        mb = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else vp
        vpt = union_box(vp, mb, g.shape)
        return key, {"id": p.stem, "file": p.name, "img": g, "mask": m, "vp": vp, "vp_train": vpt,
                     "layout": f"{g.shape[1]}x{g.shape[0]}"}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:  # threads are joined before any fork
        loaded = list(ex.map(load_one, paths))
    items, seen, dups = [], set(), 0
    for key, it in loaded:
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        items.append(it)
    return items, len(paths), dups


def thumb(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    c = g[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)]
    t = cv2.resize(c, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def make_groups(items) -> np.ndarray:
    T = np.stack([thumb(it["img"]) for it in items])
    C = T @ T.T
    n = len(items)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.nonzero(np.triu(C > GROUP_CORR, 1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def group_kfold(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ug, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(ug))
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_of_group, sizes = {}, np.zeros(n_folds, int)
    for gi in order:
        f = int(np.argmin(sizes + rng.random(n_folds) * 0.1))
        fold_of_group[ug[gi]] = f
        sizes[f] += counts[gi]
    return np.array([fold_of_group[g] for g in groups])


# --------------------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------------------
def augment_sample(it, rng: np.random.Generator):
    img, mask = it["img"], it["mask"]
    x1, y1, x2, y2 = it["vp_train"]
    # random narrow field of view (Lumify-like): keep 30-100% of the viewport width
    if rng.random() < 0.5:
        f = rng.uniform(0.3, 1.0)
        cw = max(32, int(round(f * (x2 - x1))))
        cx1 = x1 + int(rng.integers(0, max(1, (x2 - x1) - cw + 1)))
        x1, x2 = cx1, min(x2, cx1 + cw)
    # small vertical jitter of the crop
    if rng.random() < 0.3:
        hh = y2 - y1
        y1 = min(y2 - 32, y1 + int(rng.uniform(0, 0.06) * hh))
        y2 = max(y1 + 32, y2 - int(rng.uniform(0, 0.06) * hh))
    canvas, tf, mcan = letterbox(img, (x1, y1, x2, y2), SIZE, mask)
    im = canvas.astype(np.float32) / 255.0
    valid = np.zeros((SIZE, SIZE), np.float32)
    valid[tf["oy"]:tf["oy"] + tf["nh"], tf["ox"]:tf["ox"] + tf["nw"]] = 1.0
    # affine: rotation / scale / translation about the canvas centre
    do_rot, do_scale, do_tr = rng.random() < 0.6, rng.random() < 0.7, rng.random() < 0.5
    if do_rot or do_scale or do_tr:
        ang = rng.uniform(-8, 8) if do_rot else 0.0
        sc = math.exp(rng.uniform(math.log(0.8), math.log(1.25))) if do_scale else 1.0
        c = SIZE / 2 - 0.5
        M = cv2.getRotationMatrix2D((c, c), ang, sc)
        if do_tr:
            M[:, 2] += rng.uniform(-0.05, 0.05, 2) * SIZE
        im = cv2.warpAffine(im, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
        mcan = cv2.warpAffine(mcan, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
        valid = cv2.warpAffine(valid, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.5:
        im, mcan, valid = im[:, ::-1], mcan[:, ::-1], valid[:, ::-1]
    band = (mcan > 0.5).astype(np.uint8)
    band = np.ascontiguousarray(band)
    up, lo = edge_targets(band)
    valid = (valid > 0.5).astype(np.float32)
    im = np.ascontiguousarray(im)

    # ---- intensity / scanner-style augmentation (inside the valid region) ----
    ys = np.nonzero(valid.any(axis=1))[0]
    ytop, ybot = (int(ys[0]), int(ys[-1]) + 1) if len(ys) else (0, SIZE)
    if rng.random() < 0.5:  # depth-wise TGC gain curve
        knots = rng.uniform(0.6, 1.4, 5)
        gy = np.interp(np.linspace(0, 4, SIZE), np.arange(5), knots).astype(np.float32)
        gy = np.roll(gy, ytop)
        im = im * gy[:, None]
    if rng.random() < 0.2:  # lateral gain
        knots = rng.uniform(0.85, 1.15, 3)
        gx = np.interp(np.linspace(0, 2, SIZE), np.arange(3), knots).astype(np.float32)
        im = im * gx[None, :]
    if rng.random() < 0.7:
        im = np.clip(im, 0, 1) ** math.exp(rng.uniform(math.log(0.6), math.log(1.6)))
    if rng.random() < 0.7:
        im = im * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)
    if rng.random() < 0.4:  # multiplicative speckle (gamma), spatially correlated
        L = rng.uniform(2.0, 12.0)
        nz = rng.gamma(L, 1.0 / L, (SIZE, SIZE)).astype(np.float32)
        sig = rng.uniform(0.4, 1.5)
        nz = cv2.GaussianBlur(nz, (0, 0), sig)
        nz /= max(1e-3, float(nz.mean()))
        im = im * nz
    if rng.random() < 0.3:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.4, 1.5))
    if rng.random() < 0.2:  # resolution loss
        f = rng.uniform(0.4, 0.85)
        small = cv2.resize(im, (max(8, int(SIZE * f)), max(8, int(SIZE * f))), interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.15:  # edge enhancement (scanner post-processing)
        bl = cv2.GaussianBlur(im, (0, 0), 1.5)
        im = im + rng.uniform(0.3, 1.0) * (im - bl)
    if rng.random() < 0.2:
        im = im + rng.normal(0, rng.uniform(0.01, 0.04), im.shape).astype(np.float32)
    im = np.clip(im, 0, 1) * valid
    if rng.random() < 0.3:
        q = int(rng.integers(25, 91))
        ok, enc = cv2.imencode(".jpg", (im * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            im = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 * valid
    tgt = np.stack([band, up, lo]).astype(np.uint8)
    return im.astype(np.float32), tgt


# --------------------------------------------------------------------------------------
# Torch parts (imported lazily so the parent never initialises CUDA before forking)
# --------------------------------------------------------------------------------------
def torch_parts():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    class TrainDS(Dataset):
        def __init__(self, items):
            self.items = items
            self.rng = None

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            if self.rng is None:  # one generator per (persistent) worker -> fresh augs every epoch
                self.rng = np.random.default_rng([torch.initial_seed() % (2 ** 32), os.getpid()])
            im, tgt = augment_sample(self.items[i], self.rng)
            return torch.from_numpy(im[None]), torch.from_numpy(tgt)

    class EvalDS(Dataset):
        def __init__(self, items, use_vp_key="vp"):
            self.items = items
            self.key = use_vp_key

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            canvas, tf, _ = letterbox(it["img"], it[self.key], SIZE)
            return torch.from_numpy(canvas.astype(np.float32)[None] / 255.0), i

    class ApoNet(nn.Module):
        def __init__(self, smp, encoder, pretrained=True, extra=None):
            super().__init__()
            kw = dict(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                      in_channels=3, classes=3, decoder_channels=(256, 128, 64, 32, 16))
            kw.update(extra or {})
            self.base = smp.Unet(**kw)
            dec_ch = self.base.segmentation_head[0].in_channels
            self.base.segmentation_head = nn.Identity()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
            self.head = nn.Sequential(
                nn.Conv2d(dec_ch + 16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 1))
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            x3 = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
            f = self.base(x3)
            if f.shape[-2:] != x.shape[-2:]:
                f = F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return self.head(torch.cat([f, self.stem(x)], 1))

    return torch, nn, F, DataLoader, TrainDS, EvalDS, ApoNet


def import_smp():
    try:
        import segmentation_models_pytorch as smp  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "segmentation-models-pytorch==0.5.0"])
        import segmentation_models_pytorch as smp  # noqa
    return smp


ENC_EXTRA: dict = {}


def build_model(smp, ApoNet, encoder, pretrained=True):
    return ApoNet(smp, encoder, pretrained, dict(ENC_EXTRA))


RESOLVE_CODE = r'''
import json, sys
import segmentation_models_pytorch as smp
encs = json.loads(sys.argv[1])
for e in encs:
    for extra in ([{"drop_path_rate": 0.1}] if "convnext" in e else []) + [{}]:
        try:
            m = smp.Unet(encoder_name=e, encoder_weights="imagenet", in_channels=3, classes=3,
                         decoder_channels=(256, 128, 64, 32, 16), **extra)
            n = sum(p.numel() for p in m.parameters())
            pc = getattr(getattr(m.encoder, "model", None), "pretrained_cfg", None) or {}
            cfg = {k: str(pc.get(k)) for k in ("hf_hub_id", "license", "tag", "architecture")}
            print("RESOLVED " + json.dumps({"encoder": e, "params": n, "cfg": cfg, "extra": extra}), flush=True)
            sys.exit(0)
        except SystemExit:
            raise
        except Exception as ex:
            print("FAILED", e, extra, repr(ex)[:500], flush=True)
sys.exit(3)
'''


def resolve_encoder():
    """Build candidate encoders in a subprocess: downloads weights into the cache, keeps the parent
    free of torch/OpenMP state before forking the GPU workers."""
    encs = [ENCODER] + ENCODER_FALLBACKS
    r = subprocess.run([sys.executable, "-c", RESOLVE_CODE, json.dumps(encs)], capture_output=True, text=True,
                       timeout=1800)
    for line in (r.stdout + "\n" + r.stderr[-3000:]).splitlines():
        if line.startswith("FAILED") or line.startswith("RESOLVED"):
            log(line[:600])
    for line in r.stdout.splitlines():
        if line.startswith("RESOLVED "):
            return json.loads(line[len("RESOLVED "):])
    raise RuntimeError(f"no encoder could be built; stderr tail: {r.stderr[-2000:]}")


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------
def make_loss(torch, F):
    pw = None

    def dice(p, t):
        inter = (p * t).sum(dim=(2, 3))
        den = p.sum(dim=(2, 3)) + t.sum(dim=(2, 3))
        return 1.0 - (2 * inter + 1.0) / (den + 1.0)

    def loss_fn(logits, tgt):
        nonlocal pw
        logits = logits.float()
        tgt = tgt.float()
        if pw is None:
            pw = torch.tensor([3.0, 10.0, 10.0], device=logits.device).view(1, 3, 1, 1)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw, reduction="none").mean(dim=(2, 3))
        d = dice(torch.sigmoid(logits), tgt)
        per_ch = (0.5 * bce + d).mean(dim=0)  # (3,)
        total = per_ch[0] + 0.75 * (per_ch[1] + per_ch[2])
        return total, per_ch.detach()

    return loss_fn


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def evaluate_native(items, probs_native_iter, gts):
    """probs_native_iter yields (idx, prob(C,H,W) float32 native, tf). Returns summary + rows."""
    rows = []
    dsum, dn = 0.0, 0
    for idx, pn, tf in probs_native_iter:
        it = items[idx]
        x1, y1, x2, y2 = it["vp"]
        band = pn[0] > 0.5
        gtm = it["mask"].astype(bool)
        inter = float((band & gtm).sum())
        dice = (2 * inter + 1) / (float(band.sum() + gtm.sum()) + 1)
        dsum += dice
        dn += 1
        g = gts[idx]
        r = max(3, int(math.ceil(4.0 / max(1e-3, tf["sy"]))))
        comps = band_components(band, x2 - x1)
        pair = superficial_pair(comps)
        mb = me = None
        if pair is not None:
            mb, me = inner_mt(pair, pn[2], pn[1], r)
        rows.append({"id": it["id"], "layout": it["layout"], "dice": dice, "mt_gt": g, "mt_band": mb,
                     "mt_edge": me, "n_pred_bands": len(comps)})
    errs_b, errs_e, rel_b, rel_e, fails = [], [], [], [], 0
    for r_ in rows:
        g = r_["mt_gt"]
        if g is None or g <= 0:
            continue
        if r_["mt_band"] is None:
            fails += 1
            errs_b.append(g)
            errs_e.append(g)
            rel_b.append(1.0)
            rel_e.append(1.0)
            continue
        me = r_["mt_edge"] if r_["mt_edge"] is not None else r_["mt_band"]
        errs_b.append(abs(r_["mt_band"] - g))
        errs_e.append(abs(me - g))
        rel_b.append(min(1.0, abs(r_["mt_band"] - g) / g))
        rel_e.append(min(1.0, abs(me - g) / g))
    n = max(1, len(rel_b))
    summ = {
        "dice": dsum / max(1, dn),
        "mt_mae_px_band": float(np.mean(errs_b)) if errs_b else float("nan"),
        "mt_mae_px_edge": float(np.mean(errs_e)) if errs_e else float("nan"),
        "mt_medae_px_edge": float(np.median(errs_e)) if errs_e else float("nan"),
        "mt_rel_band": float(np.sum(rel_b) / n),
        "mt_rel_edge": float(np.sum(rel_e) / n),
        "mt_fail": fails,
        "n_eval": len(rel_b),
    }
    summ["score"] = 0.5 * (summ["mt_rel_band"] + summ["mt_rel_edge"])
    # signed bias (edge) and pseudo-mm assuming an 18 mm muscle
    sb = [((r_["mt_edge"] if r_["mt_edge"] is not None else r_["mt_band"]) - r_["mt_gt"]) / r_["mt_gt"]
          for r_ in rows if r_["mt_gt"] and r_["mt_band"] is not None]
    summ["mt_rel_bias_edge"] = float(np.mean(sb)) if sb else float("nan")
    summ["pseudo_mm_mae_edge"] = summ["mt_rel_edge"] * 18.0
    return summ, rows


def predict_batch(torch, models, x, flip=True, amp=True):
    """x: (B,1,S,S) cuda. Mean of sigmoid over models (+ hflip)."""
    acc = None
    n = 0
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for m in models:
            p = torch.sigmoid(m(x).float())
            acc = p if acc is None else acc + p
            n += 1
            if flip:
                pf = torch.sigmoid(m(torch.flip(x, dims=[3])).float())
                acc = acc + torch.flip(pf, dims=[3])
                n += 1
    return acc / n


# --------------------------------------------------------------------------------------
# Fold training (runs inside a forked worker that owns one GPU)
# --------------------------------------------------------------------------------------
def train_fold(fold, items, folds, gpu, encoder):
    torch, nn, F, DataLoader, TrainDS, EvalDS, ApoNet = torch_parts()
    smp = import_smp()
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(dev)
    torch.backends.cudnn.benchmark = True
    seed = SEED + 100 * fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tr_items = [it for it, f in zip(items, folds) if f != fold]
    va_idx = [i for i, f in enumerate(folds) if f == fold]
    va_items = [items[i] for i in va_idx]
    gts = [gt_mt(it["mask"], it["vp"]) for it in va_items]
    va_tfs = [letterbox(it["img"], it["vp"], SIZE)[1] for it in va_items]
    log(f"fold {fold}: train={len(tr_items)} val={len(va_items)} gt_mt_ok={sum(g is not None for g in gts)}")

    model = build_model(smp, ApoNet, encoder, pretrained=True).to(dev).to(memory_format=torch.channels_last)
    ema = build_model(smp, ApoNet, encoder, pretrained=False).to(dev).to(memory_format=torch.channels_last)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    esd, msd0 = ema.state_dict(), model.state_dict()
    ema_f = [esd[k] for k in esd if esd[k].dtype.is_floating_point]
    mod_f = [msd0[k] for k in esd if esd[k].dtype.is_floating_point]
    ema_i = [esd[k] for k in esd if not esd[k].dtype.is_floating_point]
    mod_i = [msd0[k] for k in esd if not esd[k].dtype.is_floating_point]
    enc_params = list(model.base.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENC}, {"params": dec_params, "lr": LR_DEC}],
                            weight_decay=WEIGHT_DECAY)
    base_lrs = [LR_ENC, LR_DEC]
    loss_fn = make_loss(torch, F)
    scaler = torch.amp.GradScaler("cuda")

    tl = DataLoader(TrainDS(tr_items), batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                    pin_memory=True, drop_last=True, persistent_workers=NUM_WORKERS > 0)
    vl = DataLoader(EvalDS(va_items, "vp"), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
                    pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    steps_per_epoch = len(tl)
    total_steps = steps_per_epoch * EPOCHS
    warm = int(WARMUP_EPOCHS * steps_per_epoch)
    step = 0
    best, best_epoch, best_summ, best_rows = float("inf"), -1, None, None
    hist_path = OUT / f"fold{fold}_history.csv"
    ck_path = OUT / "ckpt" / f"{MODEL_NAME}_fold{fold}.pt"
    hist_cols = ["epoch", "train_loss", "l_band", "l_up", "l_lo", "dice", "mt_mae_px_band", "mt_mae_px_edge",
                 "mt_medae_px_edge", "mt_rel_band", "mt_rel_edge", "mt_rel_bias_edge", "pseudo_mm_mae_edge",
                 "mt_fail", "n_eval", "score", "lr_enc", "sec"]
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(hist_cols)
    epoch_times = []
    for epoch in range(1, EPOCHS + 1):
        est = np.mean(epoch_times) if epoch_times else 0.0
        if time.time() - START + est > TRAIN_DEADLINE:
            log(f"fold {fold}: deadline reached before epoch {epoch}; stopping")
            break
        t0 = time.time()
        model.train()
        run, nb, chs = 0.0, 0, np.zeros(3)
        for xb, yb in tl:
            # lr schedule: linear warmup then cosine to 1%
            if step < warm:
                fac = 0.05 + 0.95 * step / max(1, warm)
            else:
                pr = (step - warm) / max(1, total_steps - warm)
                fac = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
            for g_, bl in zip(opt.param_groups, base_lrs):
                g_["lr"] = bl * fac
            xb = xb.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
            loss, per = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            d = min(EMA_DECAY, (1 + step) / (10 + step))
            with torch.no_grad():
                torch._foreach_mul_(ema_f, d)
                torch._foreach_add_(ema_f, mod_f, alpha=1 - d)
                for a_, b_ in zip(ema_i, mod_i):
                    a_.copy_(b_)
            run += float(loss.detach())
            chs += per.cpu().numpy()
            nb += 1
        t_train = time.time() - t0
        # ---- validation (EMA weights, native-resolution geometry) ----
        ema.eval()

        def gen():
            with torch.inference_mode():
                for xb, idxs in vl:
                    xb = xb.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
                    pr = predict_batch(torch, [ema], xb, flip=False).cpu().numpy()
                    for b, i in enumerate(idxs.tolist()):
                        yield i, to_native(pr[b], va_tfs[i]), va_tfs[i]

        summ, rows = evaluate_native(va_items, gen(), gts)
        sec = time.time() - t0
        epoch_times.append(sec)
        lr_now = opt.param_groups[0]["lr"]
        rec = {"epoch": epoch, "train_loss": run / max(1, nb), "l_band": chs[0] / max(1, nb),
               "l_up": chs[1] / max(1, nb), "l_lo": chs[2] / max(1, nb), **summ, "lr_enc": lr_now, "sec": sec}
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([rec[c] for c in hist_cols])
        log(f"EPOCH fold={fold} ep={epoch:02d}/{EPOCHS} loss={rec['train_loss']:.4f} "
            f"(band {rec['l_band']:.3f} up {rec['l_up']:.3f} lo {rec['l_lo']:.3f}) dice={summ['dice']:.4f} "
            f"MT_MAE_px band={summ['mt_mae_px_band']:.2f} edge={summ['mt_mae_px_edge']:.2f} "
            f"(med {summ['mt_medae_px_edge']:.2f}) rel band={summ['mt_rel_band']:.4f} edge={summ['mt_rel_edge']:.4f} "
            f"bias={summ['mt_rel_bias_edge']:+.4f} ~mm={summ['pseudo_mm_mae_edge']:.3f} fail={summ['mt_fail']}/"
            f"{summ['n_eval']} score={summ['score']:.4f} lr={lr_now:.2e} train_s={t_train:.0f} total_s={sec:.0f}")
        if epoch in (1, 2, 5) or epoch % 10 == 0:
            eta = np.mean(epoch_times[1:] if len(epoch_times) > 1 else epoch_times) * (EPOCHS - epoch)
            log(f"fold {fold}: epoch {epoch} took {sec:.0f}s; ETA this fold {eta / 60:.1f} min")
        if summ["score"] < best - 1e-6:
            best, best_epoch, best_summ, best_rows = summ["score"], epoch, summ, rows
            sd = {k: (v.half() if v.dtype.is_floating_point else v) for k, v in ema.state_dict().items()}
            torch.save({"model": sd, "encoder": encoder, "size": SIZE, "fold": fold, "epoch": epoch,
                        "val": summ, "model_name": MODEL_NAME, "channels": ["band", "upper", "lower"]}, ck_path)
            log(f"fold {fold}: saved best epoch {epoch} score={best:.4f}")
    if best_rows is not None:
        with open(OUT / f"fold{fold}_val_per_image.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
            w.writeheader()
            w.writerows(best_rows)
        by_layout = {}
        for r_ in best_rows:
            if r_["mt_gt"] and r_["mt_band"] is not None:
                me = r_["mt_edge"] if r_["mt_edge"] is not None else r_["mt_band"]
                by_layout.setdefault(r_["layout"], []).append(abs(me - r_["mt_gt"]) / r_["mt_gt"])
        best_summ = dict(best_summ)
        best_summ["rel_edge_by_layout"] = {k: [len(v), float(np.mean(v))] for k, v in sorted(by_layout.items())}
    with open(OUT / f"fold{fold}_metrics.json", "w") as f:
        json.dump({"fold": fold, "best_epoch": best_epoch, "best_score": best, "best": best_summ,
                   "epochs_run": len(epoch_times), "mean_epoch_sec": float(np.mean(epoch_times)) if epoch_times else None,
                   "n_train": len(tr_items), "n_val": len(va_items)}, f, indent=2)
    del model, ema, opt
    torch.cuda.empty_cache()
    return best


def worker_main(gpu, counter, lock, fold_list, items, folds, encoder):
    global TAG
    TAG = f"[gpu{gpu}] "
    code = 0
    while True:
        with lock:
            j = counter.value
            counter.value += 1
        if j >= len(fold_list):
            break
        fold = fold_list[j]
        t0 = time.time()
        try:
            best = train_fold(fold, items, folds, gpu, encoder)
            log(f"fold {fold} done best_score={best:.4f} minutes={(time.time() - t0) / 60:.1f}")
        except Exception:
            code = 1
            log(f"WORKER ERROR fold {fold}\n" + traceback.format_exc())
    os._exit(code)


# --------------------------------------------------------------------------------------
# External datasets (OSF, CC-BY-4.0) and inference
# --------------------------------------------------------------------------------------
def download(url: str, dest: Path, tries: int = 3) -> bool:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 umud-kaggle"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            return dest.stat().st_size > 1000
        except Exception as e:  # noqa
            log(f"download failed ({k + 1}/{tries}) {url}: {e!r}")
            time.sleep(5)
    return False


def safe_extract(zpath: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            p = (dest / m.filename).resolve()
            if not str(p).startswith(str(dest.resolve())):
                continue
            if m.is_dir():
                p.mkdir(parents=True, exist_ok=True)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(p, "wb") as dst:
                shutil.copyfileobj(src, dst)


def list_images(root: Path):
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._") \
                and "__MACOSX" not in p.parts:
            out.append(p)
    return out


def external_datasets():
    sets = {}
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    for key, url in EXT_URLS.items():
        z = EXT_DIR / f"{key}.zip"
        d = EXT_DIR / key
        try:
            if not d.exists():
                if not download(url, z):
                    continue
                safe_extract(z, d)
                z.unlink(missing_ok=True)
        except Exception as e:  # noqa
            log(f"external {key} failed: {e!r}")
            continue
        imgs = list_images(d)
        if key == "static":
            imgs = [p for p in imgs if p.stem.startswith("im_")]
        if key == "dynamic":
            imgs = [p for p in imgs if p.parent.name == "frames"] or imgs
        log(f"external {key}: {len(imgs)} images")
        name = "vl" if key.startswith("vl") else key
        sets.setdefault(name, []).extend((key, p) for p in imgs)
    return sets


class NpzWriter:
    def __init__(self, path: Path):
        self.zf = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True)
        self.keys = set()

    def add(self, key: str, arr: np.ndarray):
        key = key.replace("/", "_")
        if key in self.keys:
            raise KeyError(key)
        self.keys.add(key)
        with self.zf.open(key + ".npy", "w", force_zip64=True) as f:
            np.lib.format.write_array(f, np.ascontiguousarray(arr), allow_pickle=False)

    def close(self):
        self.zf.close()


def run_inference(encoder, items, folds, trained_folds):
    torch, nn, F, DataLoader, TrainDS, EvalDS, ApoNet = torch_parts()
    smp = import_smp()
    dev = torch.device("cuda:0")
    models = {}
    for k in trained_folds:
        ck = OUT / "ckpt" / f"{MODEL_NAME}_fold{k}.pt"
        if not ck.exists():
            continue
        state = torch.load(ck, map_location="cpu")
        m = build_model(smp, ApoNet, encoder, pretrained=False)
        m.load_state_dict({k_: v.float() if v.dtype.is_floating_point else v for k_, v in state["model"].items()})
        models[k] = m.to(dev).eval().to(memory_format=torch.channels_last)
    if not models:
        log("no checkpoints: skipping inference")
        return
    log(f"inference with folds {sorted(models)}")
    BANK.mkdir(parents=True, exist_ok=True)
    manifest = open(BANK / "manifest.csv", "w", newline="")
    mw = csv.writer(manifest)
    mw.writerow(["dataset", "image_id", "source_filename", "height", "width"])
    vpf = open(BANK / f"{MODEL_NAME}_viewports.csv", "w", newline="")
    vw = csv.writer(vpf)
    vw.writerow(["dataset", "image_id", "x1", "y1", "x2", "y2", "sx", "sy", "ox", "oy", "net_size"])
    chans = ["band", "upper", "lower"]

    def run_set(name, entries, model_list_fn):
        writers = {c: NpzWriter(BANK / f"{name}__{MODEL_NAME}_{c}.npz") for c in chans}
        t0 = time.time()
        used = set()
        n = 0
        for key, gray, src, vp in entries:
            k = key
            j = 1
            while k in used:
                j += 1
                k = f"{key}__{j}"
            used.add(k)
            canvas, tf, _ = letterbox(gray, vp, SIZE)
            x = torch.from_numpy(canvas.astype(np.float32)[None, None] / 255.0).to(dev)
            x = x.to(memory_format=torch.channels_last)
            with torch.inference_mode():
                pr = predict_batch(torch, model_list_fn(key), x, flip=True)[0].cpu().numpy()
            pn = to_native(pr, tf)
            for c, ch in enumerate(chans):
                writers[ch].add(k, np.clip(np.rint(pn[c] * 255.0), 0, 255).astype(np.uint8))
            mw.writerow([name, k, src, gray.shape[0], gray.shape[1]])
            vw.writerow([name, k, *vp, f"{tf['sx']:.6f}", f"{tf['sy']:.6f}", tf["ox"], tf["oy"], SIZE])
            n += 1
        for w in writers.values():
            w.close()
        log(f"maskbank {name}: {n} images in {time.time() - t0:.0f}s")

    all_models = [models[k] for k in sorted(models)]
    limit = SMOKE_INFER if RUN_MODE == "smoke" else None

    # competition test set
    test_dir = find_dir(COMP_ROOT, "test_set_v2")
    tpaths = list_images(test_dir)[:limit] if limit else list_images(test_dir)

    def gen_paths(paths, prefix=None):
        for p in paths:
            g = read_gray(p)
            yield p.stem, g, (f"{prefix}/{p.name}" if prefix else p.name), detect_viewport(g)

    run_set("test", gen_paths(tpaths), lambda key: all_models)
    # OSF benchmark / example datasets
    try:
        ext = external_datasets()
    except Exception as e:  # noqa
        log(f"external datasets failed: {e!r}")
        ext = {}
    for name in ("static", "dynamic", "vl"):
        if name not in ext:
            continue
        lst = ext[name][:limit] if limit else ext[name]

        def gen_ext(lst=lst):
            for sub, p in lst:
                try:
                    g = read_gray(p)
                except Exception as e:  # noqa
                    log(f"skip {p}: {e!r}")
                    continue
                if min(g.shape) < 64:
                    continue
                yield p.stem, g, f"{sub}/{p.relative_to(EXT_DIR / sub)}", detect_viewport(g)

        run_set(name, gen_ext(), lambda key: all_models)
    # out-of-fold predictions for training images (each by the model that did not see it)
    oof = [(i, it) for i, it in enumerate(items) if int(folds[i]) in models]

    def gen_oof():
        for i, it in oof:
            yield it["id"], it["img"], it["file"], it["vp"]

    fold_of = {it["id"]: int(folds[i]) for i, it in enumerate(items)}
    run_set("train_oof", gen_oof(), lambda key: [models[fold_of[key]]])
    manifest.close()
    vpf.close()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ckpt").mkdir(exist_ok=True)
    log(f"RUN_MODE={RUN_MODE} size={SIZE} epochs={EPOCHS} batch={BATCH} encoder={ENCODER}")
    random.seed(SEED)
    np.random.seed(SEED)
    n_gpu = gpu_count()
    log(f"gpus={n_gpu} cpus={os.cpu_count()}")
    if n_gpu == 0:
        raise RuntimeError("no GPU")

    t0 = time.time()
    items, n_pairs, dups = load_training_items()
    log(f"pairs={n_pairs} unique={len(items)} exact_dups_removed={dups} load_s={time.time() - t0:.0f}")
    groups = make_groups(items)
    if RUN_MODE == "smoke":
        sel = np.random.default_rng(SEED).choice(len(items), SMOKE_IMAGES, replace=False)
        sel = np.sort(sel)
        items = [items[i] for i in sel]
        groups = groups[sel]
    folds = group_kfold(groups, N_FOLDS, SEED)
    train_folds = SMOKE_FOLDS if RUN_MODE == "smoke" else list(range(N_FOLDS))
    def _area(b):
        return max(1, (b[2] - b[0]) * (b[3] - b[1]))

    n_union = sum(1 for it in items if _area(it["vp_train"]) > 1.03 * _area(it["vp"]))
    log(f"groups={len(np.unique(groups))} fold_sizes={np.bincount(folds, minlength=N_FOLDS).tolist()} "
        f"vp_union_grew_gt3pct={n_union}")
    with open(OUT / "folds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file", "layout", "group", "fold", "vx1", "vy1", "vx2", "vy2",
                    "tx1", "ty1", "tx2", "ty2"])
        for it, g, fo in zip(items, groups, folds):
            w.writerow([it["id"], it["file"], it["layout"], int(g), int(fo), *it["vp"], *it["vp_train"]])

    # quick target sanity: a few augmented samples (no torch)
    rng = np.random.default_rng(0)
    for i in range(3):
        im, tgt = augment_sample(items[i], rng)
        log(f"aug sample {i}: img mean={im.mean():.3f} band={int(tgt[0].sum())} up={int(tgt[1].sum())} "
            f"lo={int(tgt[2].sum())}")
        if i == 0:
            vis = np.concatenate([im * 255, tgt[0] * 255.0, (tgt[1] | tgt[2]) * 255.0], axis=1)
            cv2.imwrite(str(OUT / "aug_example.png"), vis.astype(np.uint8))

    # make sure smp is installed, then resolve/download the encoder in a subprocess
    smp = import_smp()
    import torch
    res = resolve_encoder()
    global ENC_EXTRA
    encoder, nparams, lic = res["encoder"], res["params"], res["cfg"]
    ENC_EXTRA = res.get("extra") or {}
    log(f"torch={torch.__version__} smp={smp.__version__} encoder={encoder} extra={ENC_EXTRA} "
        f"unet_params={nparams / 1e6:.1f}M pretrained_cfg={lic}")

    ctx = mp.get_context("fork")
    counter = ctx.Value("i", 0)
    lock = ctx.Lock()
    n_workers = max(1, min(n_gpu, len(train_folds)))
    procs = []
    for g in range(n_workers):
        p = ctx.Process(target=worker_main, args=(g, counter, lock, train_folds, items, folds, encoder),
                        daemon=False)
        p.start()
        procs.append(p)
        time.sleep(2)
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    log(f"workers finished exit_codes={codes}")

    fold_metrics = {}
    for k in train_folds:
        mp_ = OUT / f"fold{k}_metrics.json"
        if mp_.exists():
            fold_metrics[k] = json.loads(mp_.read_text())
    scores = [v["best_score"] for v in fold_metrics.values() if v.get("best_score") is not None]
    config = {
        "model_name": MODEL_NAME, "run_mode": RUN_MODE, "encoder": encoder, "params_M": nparams / 1e6,
        "pretrained_cfg": lic, "size": SIZE, "letterbox": "aspect-preserving, centred, zero pad",
        "viewport": "detect_viewport (texture density); training crop = union(viewport, mask bbox)",
        "channels": ["band", "upper", "lower"],
        "edge_convention": "upper = rows ymin-1,ymin; lower = rows ymax,ymax+1 (2 px straddling the boundary); "
                           "probability centroid = boundary position between pixels",
        "maskbank": "uint8 prob*255 at native resolution of the original image file; key = file stem",
        "epochs": EPOCHS, "batch": BATCH, "lr_enc": LR_ENC, "lr_dec": LR_DEC, "wd": WEIGHT_DECAY,
        "ema": EMA_DECAY, "n_folds": N_FOLDS, "group_corr": GROUP_CORR, "trained_folds": train_folds,
        "n_items": len(items), "n_pairs": n_pairs, "exact_dups_removed": dups,
        "fold_metrics": fold_metrics, "cv_mean_score": float(np.mean(scores)) if scores else None,
        "val_metric": "score = mean(0.5*(rel MT error band + rel MT error edge)); MT = inner-edge vertical "
                      "distance at 25/50/75% of shared x-range of the superficial pair, native px",
        "worker_exit_codes": codes, "seed": SEED,
    }
    for key in ("mt_mae_px_band", "mt_mae_px_edge", "mt_rel_edge", "dice"):
        vals = [v["best"][key] for v in fold_metrics.values() if v.get("best")]
        config[f"cv_mean_{key}"] = float(np.mean(vals)) if vals else None
    (OUT / "config.json").write_text(json.dumps(config, indent=2, default=str))
    log(f"CV summary: {json.dumps({k: config[k] for k in config if k.startswith('cv_')})}")

    run_inference(encoder, items, folds, train_folds)
    log(f"ALL DONE total_minutes={(time.time() - START) / 60:.1f}")




# ======================================================================================
# apo2 inference fix (appended by the apo2 agent; everything above is kaggle/apo_v2/train.py)
#
# The maskbank written by umud-apo-v2 v2 does not reproduce the per-epoch validation
# (OOF band Dice ~0 at 0.5 vs 0.83-0.88 logged). Suspected cause: the checkpoints were saved in
# float16; a BatchNorm running_var above 65504 becomes inf in float16 and zeroes that channel.
# This script (CPU or GPU): loads every fold checkpoint, reports non-finite / huge tensors,
# re-estimates the statistics of the broken BatchNorm layers on that fold's TRAINING images (no
# augmentation; the held-out fold is never used for calibration), checks the reproduction on a few
# held-out images with the training-time validation metric, then writes the maskbank for one PART
# as apo_v2fix_{band,upper,lower}.
# ======================================================================================
import copy  # noqa: E402

FIX_NAME = "apo_v2fix"
PART = "ext"           # "oof" | "test" | "ext"
EXT_FOLDS = [0, 1, 2]        # models used for dynamic / VL (CPU budget); static uses all folds
N_CAL = 160                  # calibration images per fold (training split of that fold)
N_CHECK = 16                 # held-out images per fold for the reproduction check


def _find_input(name: str):
    hits = sorted(Path("/kaggle/input").rglob(name))
    return hits[0] if hits else None


def _ac(torch, dev):
    return torch.autocast("cuda", dtype=torch.float16) if dev.type == "cuda" else torch.autocast(
        "cpu", enabled=False)


def _predict(torch, models, x, dev, flip):
    acc, n = None, 0
    with _ac(torch, dev):
        for m in models:
            p = torch.sigmoid(m(x).float())
            acc = p if acc is None else acc + p
            n += 1
            if flip:
                pf = torch.sigmoid(m(torch.flip(x, dims=[3])).float())
                acc = acc + torch.flip(pf, dims=[3])
                n += 1
    return acc / n


def _bn_layers(torch, model):
    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def _recalibrate(torch, model, tr_items, only_bad: bool, dev, n_max: int = N_CAL) -> int:
    bns = _bn_layers(torch, model)
    tgt = [b for b in bns if (not only_bad) or not (bool(torch.isfinite(b.running_var).all())
                                                     and bool(torch.isfinite(b.running_mean).all()))]
    if not tgt:
        return 0
    for b in tgt:
        b.reset_running_stats()
        b.momentum = None  # cumulative average over all calibration batches
    model.eval()
    for b in tgt:
        b.train()
    sel = np.random.default_rng(0).permutation(len(tr_items))[:n_max]
    xs = []
    with torch.inference_mode():
        for j, i in enumerate(sel):
            it = tr_items[int(i)]
            canvas, _, _ = letterbox(it["img"], it["vp"], SIZE)
            xs.append(canvas.astype(np.float32)[None] / 255.0)
            if len(xs) == BATCH or j == len(sel) - 1:
                x = torch.from_numpy(np.stack(xs)).to(dev).to(memory_format=torch.channels_last)
                with _ac(torch, dev):
                    model(x)
                xs = []
    model.eval()
    return len(tgt)


def _val(torch, model, va_items, dev):
    gts = [gt_mt(it["mask"], it["vp"]) for it in va_items]

    def gen():
        with torch.inference_mode():
            for i, it in enumerate(va_items):
                canvas, tf, _ = letterbox(it["img"], it["vp"], SIZE)
                x = torch.from_numpy(canvas.astype(np.float32)[None, None] / 255.0).to(dev)
                x = x.to(memory_format=torch.channels_last)
                pr = _predict(torch, [model], x, dev, False)[0].cpu().numpy()
                yield i, to_native(pr, tf), tf

    summ, _ = evaluate_native(va_items, gen(), gts)
    return summ


def run_inference_fixed(torch, models: dict, items, fold_of: dict, dev):
    BANK.mkdir(parents=True, exist_ok=True)
    manifest = open(BANK / f"manifest_{PART}.csv", "w", newline="")
    mw = csv.writer(manifest)
    mw.writerow(["dataset", "image_id", "source_filename", "height", "width"])
    vpf = open(BANK / f"{FIX_NAME}_viewports_{PART}.csv", "w", newline="")
    vw = csv.writer(vpf)
    vw.writerow(["dataset", "image_id", "x1", "y1", "x2", "y2", "sx", "sy", "ox", "oy", "net_size"])
    chans = ["band", "upper", "lower"]
    flip = dev.type == "cuda"

    def run_set(name, entries, model_list_fn):
        writers = {c: NpzWriter(BANK / f"{name}__{FIX_NAME}_{c}.npz") for c in chans}
        t0 = time.time()
        used, n = set(), 0
        for key, gray, src, vp in entries:
            k, j = key, 1
            while k in used:
                j += 1
                k = f"{key}__{j}"
            used.add(k)
            canvas, tf, _ = letterbox(gray, vp, SIZE)
            x = torch.from_numpy(canvas.astype(np.float32)[None, None] / 255.0).to(dev)
            x = x.to(memory_format=torch.channels_last)
            with torch.inference_mode():
                pr = _predict(torch, model_list_fn(key), x, dev, flip)[0].cpu().numpy()
            pn = to_native(pr, tf)
            for c, ch in enumerate(chans):
                writers[ch].add(k, np.clip(np.rint(pn[c] * 255.0), 0, 255).astype(np.uint8))
            mw.writerow([name, k, src, gray.shape[0], gray.shape[1]])
            vw.writerow([name, k, *vp, f"{tf['sx']:.6f}", f"{tf['sy']:.6f}", tf["ox"], tf["oy"], SIZE])
            n += 1
            if n % 25 == 0:
                manifest.flush()
                log(f"  {name}: {n} images, {time.time() - t0:.0f}s")
        for w in writers.values():
            w.close()
        log(f"maskbank {name}: {n} images in {time.time() - t0:.0f}s (flip={flip})")

    all_models = [models[k] for k in sorted(models)]
    if PART == "test":
        test_dir = find_dir(COMP_ROOT, "test_set_v2")

        def gen_paths(paths):
            for p in paths:
                g = read_gray(p)
                yield p.stem, g, p.name, detect_viewport(g)

        run_set("test", gen_paths(list_images(test_dir)), lambda key: all_models)
    elif PART == "oof":
        def gen_oof():
            for it in items:
                yield it["id"], it["img"], it["file"], it["vp"]

        run_set("train_oof", gen_oof(), lambda key: [models[fold_of[key]]])
    else:
        try:
            ext = external_datasets()
        except Exception as e:  # noqa
            log(f"external datasets failed: {e!r}")
            ext = {}
        ext_models = [models[k] for k in EXT_FOLDS if k in models]
        for name in ("static", "dynamic", "vl"):
            if name not in ext:
                continue
            lst = ext[name]
            if name == "vl":  # raw (unannotated) .tiff images only; the .tif twins carry the drawn lines
                lst = [(s, p) for s, p in lst if p.suffix.lower() == ".tiff"]

            def gen_ext(lst=lst):
                for sub, p in lst:
                    try:
                        g = read_gray(p)
                    except Exception as e:  # noqa
                        log(f"skip {p}: {e!r}")
                        continue
                    if min(g.shape) < 64:
                        continue
                    yield p.stem, g, f"{sub}/{p.relative_to(EXT_DIR / sub)}", detect_viewport(g)

            run_set(name, gen_ext(), (lambda key: all_models) if name == "static" else (lambda key: ext_models))
    manifest.close()
    vpf.close()


def fix_main():
    global ENC_EXTRA
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"apo_v2 inference fix: start PART={PART}")
    ck0 = _find_input("apo_v2_fold0.pt")
    fcsv = _find_input("folds.csv")
    if ck0 is None:
        raise RuntimeError("checkpoints not found under /kaggle/input")
    log(f"ckpt dir {ck0.parent}; folds.csv {fcsv}")
    items, n_pairs, dups = load_training_items()
    log(f"items={len(items)} pairs={n_pairs} dups={dups}")
    fold_of = {}
    with open(fcsv) as f:
        for r in csv.DictReader(f):
            fold_of[r["image_id"]] = int(r["fold"])
    miss = [it["id"] for it in items if it["id"] not in fold_of]
    if miss:
        raise RuntimeError(f"{len(miss)} items missing from folds.csv")
    smp = import_smp()
    torch, nn, F, DataLoader, TrainDS, EvalDS, ApoNet = torch_parts()
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    log(f"device={dev} threads={torch.get_num_threads()} torch={torch.__version__} smp={smp.__version__}")
    ENC_EXTRA = {"drop_path_rate": 0.1}
    need = list(range(N_FOLDS))
    if PART == "ext":
        need = sorted(set(EXT_FOLDS) | set(range(N_FOLDS)))  # static uses all folds
    models, report = {}, {}
    for k in need:
        t0 = time.time()
        ck = ck0.parent / f"{MODEL_NAME}_fold{k}.pt"
        state = torch.load(ck, map_location="cpu", weights_only=False)
        sd = state["model"]
        nonfin = [n for n, v in sd.items() if v.dtype.is_floating_point and not bool(torch.isfinite(v).all())]
        rv = {n: float(v.float().abs().max()) for n, v in sd.items() if n.endswith("running_var")}
        top_rv = sorted(rv.items(), key=lambda t: -t[1] if np.isfinite(t[1]) else -1e30)[:5]
        log(f"fold {k}: epoch {state.get('epoch')} nonfinite={len(nonfin)} {nonfin[:8]} "
            f"top running_var={[(n, round(v, 1)) for n, v in top_rv]}")
        base = build_model(smp, ApoNet, state.get("encoder", ENCODER), pretrained=False)
        base.load_state_dict({n: v.float() if v.dtype.is_floating_point else v for n, v in sd.items()})
        base = base.to(dev).eval().to(memory_format=torch.channels_last)
        tr = [it for it in items if fold_of[it["id"]] != k]
        va = [it for it in items if fold_of[it["id"]] == k][:N_CHECK]
        res, chosen = {}, None
        nb = _recalibrate(torch, base, tr, True, dev) if nonfin else 0
        name = "bad_bn" if nb else "asis"
        res[name] = _val(torch, base, va, dev)
        chosen, model = name, base
        if res[name]["dice"] < 0.7:
            m2 = build_model(smp, ApoNet, state.get("encoder", ENCODER), pretrained=False)
            m2.load_state_dict({n: v.float() if v.dtype.is_floating_point else v for n, v in sd.items()})
            m2 = m2.to(dev).eval().to(memory_format=torch.channels_last)
            _recalibrate(torch, m2, tr, False, dev)
            res["all_bn"] = _val(torch, m2, va, dev)
            if res["all_bn"]["score"] < res[name]["score"]:
                chosen, model = "all_bn", m2
        logged = state.get("val", {})
        for n_, s in res.items():
            log(f"fold {k} variant {n_} (n_bad_bn={nb}): dice={s['dice']:.4f} score={s['score']:.4f} "
                f"mt_rel_edge={s['mt_rel_edge']:.4f} fail={s['mt_fail']} | logged dice={logged.get('dice', float('nan')):.4f} "
                f"score={logged.get('score', float('nan')):.4f}")
        report[k] = dict(nonfinite=nonfin, top_running_var=top_rv, n_bad_bn=nb, check=res, chosen=chosen,
                         logged=logged)
        models[k] = model
        log(f"fold {k}: chosen {chosen} ({time.time() - t0:.0f}s)")
    (OUT / f"fix_report_{PART}.json").write_text(json.dumps(report, indent=1, default=str))
    ok = all(report[k]["check"][report[k]["chosen"]]["dice"] > 0.7 for k in report)
    log(f"reproduction ok={ok}")
    if not ok:
        log("reproduction FAILED: maskbank not written")
        return
    run_inference_fixed(torch, models, items, fold_of, dev)
    log(f"ALL DONE total_minutes={(time.time() - START) / 60:.1f}")


if __name__ == "__main__":
    fix_main()
