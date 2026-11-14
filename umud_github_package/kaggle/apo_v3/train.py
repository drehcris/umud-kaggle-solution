"""UMUD apo_v3: fp32-safe re-run of apo_v2 (5 folds x hflip TTA everywhere) + self-training on unlabeled images.

Kaggle GPU kernel (shivateja005/umud-apo-v3). Single file, self-contained. Attach shivateja005/umud-apo-v2 via
kernel_sources (fold checkpoints, folds.csv, fold metrics).

Phase 1 (teacher = "apo_v2r", always runs):
  * load the 5 apo_v2 fold checkpoints (saved in float16: decoder.blocks.0.conv1.1.running_var overflowed to inf in
    all 5), re-estimate ONLY the non-finite BatchNorm layers on that fold's TRAINING images (no augmentation, the
    held-out fold is never touched), verify on the whole held-out fold against the logged apo_v2 fold metrics, save
    fp32 checkpoints;
  * 5-fold x hflip-TTA maps for test / static / dynamic (all 168) / VL raw (.tiff, 249) and fold-wise OOF maps for
    all 1048 training images (model apo_v2r). This replaces the CPU-repaired apo_v2fix maps (no TTA, 3 folds on
    VL/dynamic).
Phase 2 (student = "apo_v3", SELF_TRAIN):
  * fold k warm-starts from the repaired fold-k teacher (same folds -> honest OOF) and is fine-tuned on its GT folds
    plus pseudo-labelled unlabeled images (test 309, static 35, dynamic every 4th, VL raw 249): band = teacher
    p >= 0.75, background p <= 0.25, ignored in between; edge targets are re-derived from the hard pseudo band
    exactly like the GT edges; images whose teacher band has no superficial pair or > 50 % uncertain band pixels
    are not used. Stronger domain augmentation (Fourier amplitude swap / histogram matching to the unlabeled
    images) on top of the apo_v2 augmentation;
  * float32 EMA checkpoints, reloaded from disk + checked for inf/nan + re-validated before any maskbank is written;
  * 5-fold x hflip maps for the same sets + fold-wise OOF (model apo_v3).
Every epoch also logs an out-of-domain diagnostic on the VL images: inner-edge MT vs the gap between the expert-
drawn aponeurosis lines (median ratio and its spread; the lines are not an expert MT, so only the spread and the
failure count are meaningful).

Outputs (/kaggle/working): maskbank/{test,static,dynamic,vl,train_oof}__{apo_v2r,apo_v3}_{band,upper,lower}.npz
(uint8 prob*255, native resolution, key = file stem, VL repeats get __2; manifest.csv maps keys to source files),
apo_v3/{teacher_ckpt,ckpt}/*.pt (fp32), apo_v3/*fold*_metrics.json, histories, folds.csv, config.json, log.txt.
"""
from __future__ import annotations

import base64
import csv
import hashlib
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
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)

# --------------------------------------------------------------------------------------
RUN_MODE = "full"  # "full" | "smoke" (GPU, tiny) | "cpusmoke" (CPU only, tiny; kernel id *-cpusmoke)
SMOKE = RUN_MODE != "full"
CPU_SMOKE = RUN_MODE == "cpusmoke"
SELF_TRAIN = True  # False = teacher phase only (repaired apo_v2 + TTA maps, ~25 min)
DOMAIN_AUG = True

SEED = 20261021
SIZE = 512
N_FOLDS = 5
EPOCHS = 1 if CPU_SMOKE else (2 if SMOKE else 15)
MAX_STEPS = 3 if CPU_SMOKE else None
BATCH = 2 if CPU_SMOKE else 8
LR_ENC, LR_DEC = 1.0e-4, 4.0e-4  # fine-tuning from the repaired teacher
WEIGHT_DECAY = 1.0e-2
WARMUP_EPOCHS = 0.3
EMA_DECAY = 0.997
NUM_WORKERS = 0 if CPU_SMOKE else 2
GROUP_CORR = 0.95
ENCODER = "tu-convnext_small.fb_in22k_ft_in1k"
TEACHER_NAME = "apo_v2r"
MODEL_NAME = "apo_v3"
SRC_MODEL = "apo_v2"
SMOKE_IMAGES = 40
SMOKE_FOLDS = [0] if CPU_SMOKE else [0, 1]
SMOKE_INFER = 2 if CPU_SMOKE else 5
N_CAL = 8 if CPU_SMOKE else 160
PSEUDO_FRAC = 0.25
W_PS = 0.5
PS_POS, PS_NEG = 0.75, 0.25
PS_MAX_UNCERTAIN = 0.5
DYN_STRIDE = 4
P_FDA, P_HISTM = 0.2, 0.2
VERIFY_TOL_DICE, VERIFY_TOL_SCORE = 0.01, 0.003
TRAIN_DEADLINE = 3.0 * 3600  # no new epoch after this (s from start)
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
PARTS = Path("/tmp/apo3_parts")
EXT_DIR = Path("/tmp/umud_ext")
START = time.time()
TAG = ""
CHANS = ["band", "upper", "lower"]

ITEMS: list[dict] = []
FOLDS: np.ndarray | None = None
UNL: list[dict] = []
DOMAIN_BANK: np.ndarray | None = None
SRC: dict = {}
ENC_EXTRA: dict = {}


def log(msg: str) -> None:
    line = f"{TAG}[{(time.time() - START) / 60:6.1f}m] {msg}"
    print(line, flush=True)
    try:
        with open(OUT / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def tail3(p: str) -> str:
    return "/".join(Path(str(p).replace("\\", "/")).parts[-3:]).lower()


# --------------------------------------------------------------------------------------
# Image IO (identical to apo_v2)
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
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return (m > 127).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Viewport detection (identical to apo_v2)
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
# Letterbox (centre-aligned pixel convention, exactly invertible); mask may be (H,W) or (H,W,C)
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
          "ox": int(ox), "oy": int(oy), "H": int(gray.shape[0]), "W": int(gray.shape[1]), "nw": nw, "nh": nh}
    mcanvas = None
    if mask is not None:
        shp = (size, size) + mask.shape[2:]
        mcanvas = np.zeros(shp, np.float32)
        r = cv2.resize(np.ascontiguousarray(mask[y1:y2, x1:x2]).astype(np.float32), (nw, nh), interpolation=interp)
        mcanvas[oy:oy + nh, ox:ox + nw] = r.reshape((nh, nw) + mask.shape[2:])
    return canvas, tf, mcanvas


def to_native(prob: np.ndarray, tf: dict) -> np.ndarray:
    sx, sy = tf["sx"], tf["sy"]
    M = np.array([[sx, 0.0, (0.5 - tf["x1"]) * sx - 0.5 + tf["ox"]],
                  [0.0, sy, (0.5 - tf["y1"]) * sy - 0.5 + tf["oy"]]], np.float64)
    out = np.zeros((prob.shape[0], tf["H"], tf["W"]), np.float32)
    for c in range(prob.shape[0]):
        o = cv2.warpAffine(np.ascontiguousarray(prob[c], dtype=np.float32), M, (tf["W"], tf["H"]),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[c, tf["y1"]:tf["y2"], tf["x1"]:tf["x2"]] = o[tf["y1"]:tf["y2"], tf["x1"]:tf["x2"]]
    return out


# --------------------------------------------------------------------------------------
# Targets / geometry (identical to apo_v2)
# --------------------------------------------------------------------------------------
def edge_targets(mask: np.ndarray, min_area: int = 15):
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
        comps.append({"top": top, "bot": bot, "yc": float(np.nanmedian((t + b) / 2)), "width": int(present.sum())})
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


def inner_mt(pair, low_prob=None, up_prob=None, r: int = 3, with_x: bool = False):
    sup, deep = pair
    both = ~np.isnan(sup["bot"]) & ~np.isnan(deep["top"])
    both &= np.nan_to_num(deep["top"], nan=-1) > np.nan_to_num(sup["bot"], nan=1e9)
    xs = np.flatnonzero(both)
    if len(xs) < 5:
        return (None, None, []) if with_x else (None, None)
    xa, xb = int(xs[0]), int(xs[-1])
    hw = max(2, int(0.02 * (xb - xa)))
    vb, ve, xq = [], [], []
    for f in (0.25, 0.5, 0.75):
        xc = int(round(xa + f * (xb - xa)))
        cols = [x for x in range(xc - hw, xc + hw + 1) if 0 <= x < len(both) and both[x]]
        if not cols:
            continue
        xq.append(xc)
        vb.append(float(np.median([deep["top"][x] - sup["bot"][x] - 1.0 for x in cols])))
        if low_prob is not None and up_prob is not None:
            ge = []
            for x in cols:
                yl = _refine(low_prob[:, x], sup["bot"][x] + 0.5, r)
                yu = _refine(up_prob[:, x], deep["top"][x] - 0.5, r)
                ge.append(yu - yl)
            ve.append(float(np.median(ge)))
    if not vb:
        return (None, None, []) if with_x else (None, None)
    res = (float(np.mean(vb)), (float(np.mean(ve)) if ve else None))
    return (*res, xq) if with_x else res


def gt_mt(mask: np.ndarray, box) -> float | None:
    pair = superficial_pair(band_components(mask, box[2] - box[0]))
    return None if pair is None else inner_mt(pair)[0]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def find_dir(root: Path, name: str) -> Path:
    for base in (root, Path("/kaggle/input")):
        if base.exists():
            matches = sorted((p for p in base.rglob(name) if p.is_dir()), key=lambda p: len(str(p)))
            if matches:
                return matches[0]
    raise RuntimeError(f"{name} not found")


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

    with ThreadPoolExecutor(max_workers=4) as ex:
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
# Augmentation (apo_v2 + domain augmentation; GT masks or teacher pseudo-labels)
# --------------------------------------------------------------------------------------
def _fda(x: np.ndarray, ref: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    fx = np.fft.rfft2(x)
    amp_x, pha = np.abs(fx), np.angle(fx)
    amp_r = np.abs(np.fft.rfft2(ref))
    b = max(1, int(round(beta * SIZE)))
    rows = np.r_[0:b, SIZE - b:SIZE][:, None]
    cols = np.arange(b)[None, :]
    amp_x[rows, cols] = (1 - alpha) * amp_x[rows, cols] + alpha * amp_r[rows, cols]
    return np.fft.irfft2(amp_x * np.exp(1j * pha), s=x.shape).astype(np.float32)


def _hist_match(x: np.ndarray, valid: np.ndarray, ref: np.ndarray) -> np.ndarray:
    rv = ref[ref > 3]
    xv = x[valid > 0]
    if rv.size < 500 or xv.size < 500:
        return x
    q = np.linspace(0, 1, 65)
    xq = np.maximum.accumulate(np.quantile(xv, q) + np.arange(65) * 1e-6)
    rq = np.quantile(rv.astype(np.float32) / 255.0, q)
    return np.interp(x, xq, rq).astype(np.float32)


def augment_sample(it, rng: np.random.Generator):
    """-> image (S,S) float32, targets (3,S,S) uint8, loss mask (3,S,S) uint8, image weight."""
    img = it["img"]
    pseudo = it.get("pseudo")
    lab = it["mask"] if pseudo is None else pseudo
    x1, y1, x2, y2 = it["vp_train"]
    if rng.random() < 0.5:  # narrow field of view: keep 30-100 % of the viewport width
        f = rng.uniform(0.3, 1.0)
        cw = max(32, int(round(f * (x2 - x1))))
        cx1 = x1 + int(rng.integers(0, max(1, (x2 - x1) - cw + 1)))
        x1, x2 = cx1, min(x2, cx1 + cw)
    if rng.random() < 0.3:
        hh = y2 - y1
        y1 = min(y2 - 32, y1 + int(rng.uniform(0, 0.06) * hh))
        y2 = max(y1 + 32, y2 - int(rng.uniform(0, 0.06) * hh))
    canvas, tf, mcan = letterbox(img, (x1, y1, x2, y2), SIZE, lab)
    im = canvas.astype(np.float32) / 255.0
    valid = np.zeros((SIZE, SIZE), np.float32)
    valid[tf["oy"]:tf["oy"] + tf["nh"], tf["ox"]:tf["ox"] + tf["nw"]] = 1.0
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
    if pseudo is None:
        band = np.ascontiguousarray((mcan > 0.5).astype(np.uint8))
        wmask = np.ones((3, SIZE, SIZE), np.uint8)
        wimg = 1.0
    else:
        p = np.ascontiguousarray(mcan[..., 0]) / 255.0
        band = (p >= PS_POS).astype(np.uint8)
        ign = ((p > PS_NEG) & (p < PS_POS)).astype(np.uint8)
        ign_e = cv2.dilate(ign, np.ones((5, 5), np.uint8))
        wmask = np.stack([1 - ign, 1 - ign_e, 1 - ign_e]).astype(np.uint8)
        wimg = W_PS
    up, lo = edge_targets(band)
    valid = (valid > 0.5).astype(np.float32)
    im = np.ascontiguousarray(im)
    ys = np.nonzero(valid.any(axis=1))[0]
    ytop = int(ys[0]) if len(ys) else 0
    if DOMAIN_AUG and DOMAIN_BANK is not None and len(DOMAIN_BANK):
        r = rng.random()
        if r < P_FDA:
            ref = DOMAIN_BANK[int(rng.integers(len(DOMAIN_BANK)))].astype(np.float32) / 255.0
            im = np.clip(_fda(im, ref, rng.uniform(0.005, 0.03), rng.uniform(0.3, 1.0)), 0, 1) * valid
        elif r < P_FDA + P_HISTM:
            im = _hist_match(im, valid, DOMAIN_BANK[int(rng.integers(len(DOMAIN_BANK)))]) * valid
    if rng.random() < 0.5:
        knots = rng.uniform(0.6, 1.4, 5)
        gy = np.interp(np.linspace(0, 4, SIZE), np.arange(5), knots).astype(np.float32)
        im = im * np.roll(gy, ytop)[:, None]
    if rng.random() < 0.2:
        knots = rng.uniform(0.85, 1.15, 3)
        gx = np.interp(np.linspace(0, 2, SIZE), np.arange(3), knots).astype(np.float32)
        im = im * gx[None, :]
    if rng.random() < 0.7:
        im = np.clip(im, 0, 1) ** math.exp(rng.uniform(math.log(0.6), math.log(1.6)))
    if rng.random() < 0.7:
        im = im * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)
    if rng.random() < 0.4:
        L = rng.uniform(2.0, 12.0)
        nz = rng.gamma(L, 1.0 / L, (SIZE, SIZE)).astype(np.float32)
        nz = cv2.GaussianBlur(nz, (0, 0), rng.uniform(0.4, 1.5))
        im = im * (nz / max(1e-3, float(nz.mean())))
    if rng.random() < 0.3:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.4, 1.5))
    if DOMAIN_AUG and rng.random() < 0.15:  # lateral (beam-width) blur
        im = cv2.GaussianBlur(im, (0, 0), sigmaX=rng.uniform(0.8, 2.5), sigmaY=0.3)
    if rng.random() < 0.2:
        f = rng.uniform(0.4, 0.85)
        small = cv2.resize(im, (max(8, int(SIZE * f)), max(8, int(SIZE * f))), interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.15:
        bl = cv2.GaussianBlur(im, (0, 0), 1.5)
        im = im + rng.uniform(0.3, 1.0) * (im - bl)
    if rng.random() < 0.2:
        im = im + rng.normal(0, rng.uniform(0.01, 0.04), im.shape).astype(np.float32)
    im = np.clip(im, 0, 1) * valid
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", (im * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 91))])
        if ok:
            im = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 * valid
    tgt = np.stack([band, up, lo]).astype(np.uint8)
    return im.astype(np.float32), tgt, wmask, np.float32(wimg)


# --------------------------------------------------------------------------------------
# Torch parts (imported lazily so the parent never initialises CUDA before forking)
# --------------------------------------------------------------------------------------
def torch_parts():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    class TrainDS(Dataset):
        def __init__(self, gt, pl, n_pl):
            self.gt, self.pl, self.n_pl = gt, pl, (n_pl if pl else 0)
            self.calls = 0

        def __len__(self):
            return len(self.gt) + self.n_pl

        def __getitem__(self, i):
            self.calls += 1
            rng = np.random.default_rng([torch.initial_seed() % (2 ** 32), i, self.calls, os.getpid()])
            it = self.gt[i] if i < len(self.gt) else self.pl[int(rng.integers(len(self.pl)))]
            im, tgt, wm, wi = augment_sample(it, rng)
            return torch.from_numpy(im[None]), torch.from_numpy(tgt), torch.from_numpy(wm), torch.tensor(float(wi))

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
    """segmentation-models-pytorch 0.5.0 exactly (as used by apo_v2: the decoder state-dict layout depends on it)."""
    try:
        import segmentation_models_pytorch as smp  # noqa

        if smp.__version__ == "0.5.0":
            return smp
    except ImportError:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "segmentation-models-pytorch==0.5.0"])
    import importlib

    for name in list(sys.modules):
        if name.startswith("segmentation_models_pytorch"):
            del sys.modules[name]
    return importlib.import_module("segmentation_models_pytorch")


def build_model(smp, ApoNet, encoder, pretrained=False):
    return ApoNet(smp, encoder, pretrained, dict(ENC_EXTRA))


def amp(torch, dev):
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def make_loss(torch, F):
    pw = None

    def loss_fn(logits, tgt, wmask, wimg):
        """Masked BCE + Dice per channel and image, image-weighted; band + 0.75 * (upper + lower)."""
        nonlocal pw
        logits, tgt, wm = logits.float(), tgt.float(), wmask.float()
        if pw is None:
            pw = torch.tensor([3.0, 10.0, 10.0], device=logits.device).view(1, 3, 1, 1)
        bce = (F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw, reduction="none") * wm).sum((2, 3)) \
            / wm.sum((2, 3)).clamp_min(1.0)
        p = torch.sigmoid(logits) * wm
        t = tgt * wm
        d = 1.0 - (2 * (p * t).sum((2, 3)) + 1.0) / (p.sum((2, 3)) + t.sum((2, 3)) + 1.0)
        w = wimg.view(-1, 1)
        per_ch = ((0.5 * bce + d) * w).sum(0) / w.sum().clamp_min(1e-6)
        total = per_ch[0] + 0.75 * (per_ch[1] + per_ch[2])
        return total, per_ch.detach()

    return loss_fn


def evaluate_native(items, probs_native_iter, gts):
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
    sb = [((r_["mt_edge"] if r_["mt_edge"] is not None else r_["mt_band"]) - r_["mt_gt"]) / r_["mt_gt"]
          for r_ in rows if r_["mt_gt"] and r_["mt_band"] is not None]
    summ["mt_rel_bias_edge"] = float(np.mean(sb)) if sb else float("nan")
    summ["pseudo_mm_mae_edge"] = summ["mt_rel_edge"] * 18.0
    return summ, rows


def predict_batch(torch, models, x, flip=True):
    acc, n = None, 0
    with amp(torch, x.device):
        for m in models:
            p = torch.sigmoid(m(x).float())
            acc = p if acc is None else acc + p
            n += 1
            if flip:
                pf = torch.sigmoid(m(torch.flip(x, dims=[3])).float())
                acc = acc + torch.flip(pf, dims=[3])
                n += 1
    return acc / n


def val_metrics(torch, DataLoader, EvalDS, model, va_items, va_tfs, gts, dev, flip=False):
    vl = DataLoader(EvalDS(va_items, "vp"), batch_size=BATCH, shuffle=False, num_workers=0)

    def gen():
        with torch.inference_mode():
            for xb, idxs in vl:
                xb = xb.to(dev).to(memory_format=torch.channels_last)
                pr = predict_batch(torch, [model], xb, flip=flip).cpu().numpy()
                for b, i in enumerate(idxs.tolist()):
                    yield i, to_native(pr[b], va_tfs[i]), va_tfs[i]

    return evaluate_native(va_items, gen(), gts)


def vl_lines_metric(torch, model, vl_items, dev) -> dict:
    """VL raw images: inner-edge MT (px) of the predicted superficial pair vs the gap between the expert-drawn
    aponeurosis lines at the same x positions. The lines are not an expert MT -> report the median ratio, the
    spread around it and the failure count."""
    if not vl_items:
        return {}
    ratios, fails = [], 0
    with torch.inference_mode():
        for u in vl_items:
            canvas, tf, _ = letterbox(u["vl_img"], u["vp"], SIZE)
            x = torch.from_numpy(canvas.astype(np.float32)[None, None] / 255.0).to(dev).to(memory_format=torch.channels_last)
            pn = to_native(predict_batch(torch, [model], x, flip=False)[0].cpu().numpy(), tf)
            pair = superficial_pair(band_components(pn[0] > 0.5, tf["x2"] - tf["x1"]))
            if pair is None:
                fails += 1
                continue
            mb, me, xq = inner_mt(pair, pn[2], pn[1], 3, with_x=True)
            if mb is None or not xq:
                fails += 1
                continue
            L = u["lines"]
            gaps = []
            for xc in xq:
                ys = L["sup"][1] + (xc - L["sup"][0]) * L["sup"][3] / L["sup"][2]
                yd = L["deep"][1] + (xc - L["deep"][0]) * L["deep"][3] / L["deep"][2]
                gaps.append(yd - ys)
            g = float(np.mean(gaps))
            if g > 5:
                ratios.append((me if me is not None else mb) / g)
    r = np.asarray(ratios)
    if not len(r):
        return {"vl_fail": fails, "vl_n": 0}
    med = float(np.median(r))
    return {"vl_ratio_med": med, "vl_ratio_spread": float(np.mean(np.abs(r / med - 1))), "vl_fail": fails,
            "vl_n": int(len(r))}


# --------------------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------------------
def nonfinite_keys(torch, sd) -> list[str]:
    return [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all())]


def bn_layers(torch, model, names=None):
    out = []
    for n_, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and (names is None or n_ in names):
            out.append(m)
    return out


def recalibrate(torch, model, items, dev, layer_names=None, n_max: int = N_CAL) -> int:
    """Re-estimate the running stats of the given (default: all) BatchNorm layers, cumulative average over un-
    augmented training crops (viewport letterbox, as at test time)."""
    tgt = bn_layers(torch, model, layer_names)
    if not tgt:
        return 0
    for b in tgt:
        b.reset_running_stats()
        b.momentum = None
    model.eval()
    for b in tgt:
        b.train()
    sel = np.random.default_rng(0).permutation(len(items))[:n_max]
    xs = []
    with torch.inference_mode():
        for j, i in enumerate(sel):
            canvas, _, _ = letterbox(items[int(i)]["img"], items[int(i)]["vp"], SIZE)
            xs.append(canvas.astype(np.float32)[None] / 255.0)
            if len(xs) == BATCH or j == len(sel) - 1:
                x = torch.from_numpy(np.stack(xs)).to(dev).to(memory_format=torch.channels_last)
                with amp(torch, dev):
                    model(x)
                xs = []
    for b in tgt:
        b.momentum = 0.1
    model.eval()
    return len(tgt)


def save_fp32(torch, model, path: Path, meta: dict) -> None:
    sd = {k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "dtype": "float32", **meta}, path)


def load_fp32(torch, model, path: Path) -> list[str]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    bad = nonfinite_keys(torch, st["model"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st["model"].items()})
    return bad


# --------------------------------------------------------------------------------------
# Fold jobs (run inside a forked worker that owns one device)
# --------------------------------------------------------------------------------------
def fold_split(fold):
    tr = [it for it, f in zip(ITEMS, FOLDS) if f != fold]
    va = [it for it, f in zip(ITEMS, FOLDS) if f == fold]
    return tr, va


def repair_teacher(fold, dev):
    """apo_v2 fold checkpoint (fp16) -> repaired fp32 teacher, verified on the whole held-out fold."""
    torch, nn, F, DataLoader, TrainDS, EvalDS, ApoNet = torch_parts()
    smp = import_smp()
    tr, va = fold_split(fold)
    ck = SRC["ckpt"].get(fold)
    if ck is None:
        raise RuntimeError(f"apo_v2 checkpoint for fold {fold} not found")
    state = torch.load(ck, map_location="cpu", weights_only=False)
    sd = state["model"]
    bad = nonfinite_keys(torch, sd)
    dtypes = sorted({str(v.dtype) for v in sd.values() if torch.is_tensor(v)})
    top = sorted(((k, float(v.float().max())) for k, v in sd.items() if k.endswith("running_var")),
                 key=lambda t: -t[1] if math.isfinite(t[1]) else -1e30)[:4]
    log(f"teacher fold {fold}: {ck.name} epoch={state.get('epoch')} dtypes={dtypes} nonfinite={bad} top_running_var={top}")
    bad_w = [k for k in bad if not k.endswith(("running_mean", "running_var"))]
    if bad_w:
        raise RuntimeError(f"fold {fold}: non-finite weights {bad_w[:5]} cannot be repaired")
    model = build_model(smp, ApoNet, state.get("encoder", ENCODER), pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    model = model.to(dev).to(memory_format=torch.channels_last).eval()
    names = sorted({k.rsplit(".", 1)[0] for k in bad})
    n_rec = recalibrate(torch, model, tr, dev, layer_names=set(names)) if names else 0
    gts = [gt_mt(it["mask"], it["vp"]) for it in va]
    va_tfs = [letterbox(it["img"], it["vp"], SIZE)[1] for it in va]
    summ, _ = val_metrics(torch, DataLoader, EvalDS, model, va, va_tfs, gts, dev, flip=False)
    ref = SRC["metrics"].get(fold, {}).get("best", {}) or {}
    ok = bool(ref) and summ["dice"] >= ref.get("dice", 9) - 1.5 * VERIFY_TOL_DICE and \
        summ["score"] <= ref.get("score", -9) + 1.5 * VERIFY_TOL_SCORE  # one-sided: no worse than logged
    if SMOKE:
        ok = all(math.isfinite(float(v)) for v in (summ["dice"], summ["score"]))
    log(f"teacher fold {fold}: recalibrated {n_rec} BN layers {names}; held-out dice={summ['dice']:.4f} "
        f"score={summ['score']:.4f} fail={summ['mt_fail']} | logged dice={ref.get('dice', float('nan')):.4f} "
        f"score={ref.get('score', float('nan')):.4f} -> {'OK' if ok else 'MISMATCH'}")
    path = OUT / "teacher_ckpt" / f"{TEACHER_NAME}_fold{fold}.pt"
    save_fp32(torch, model, path, {"encoder": state.get("encoder", ENCODER), "fold": fold, "src_epoch": state.get("epoch"),
                                   "recalibrated": names, "val": summ, "ok": ok, "model_name": TEACHER_NAME})
    m2 = build_model(smp, ApoNet, state.get("encoder", ENCODER), pretrained=False)
    bad2 = load_fp32(torch, m2, path)
    m2 = m2.to(dev).to(memory_format=torch.channels_last).eval()
    s2, _ = val_metrics(torch, DataLoader, EvalDS, m2, va[:24], va_tfs[:24], gts[:24], dev)
    s1, _ = val_metrics(torch, DataLoader, EvalDS, model, va[:24], va_tfs[:24], gts[:24], dev)
    reload_ok = not bad2 and abs(s2["dice"] - s1["dice"]) < 1e-3
    log(f"teacher fold {fold}: fp32 reload nonfinite={bad2} dice {s2['dice']:.5f} vs {s1['dice']:.5f} -> "
        f"{'OK' if reload_ok else 'FAILED'}")
    (OUT / f"teacher_fold{fold}_metrics.json").write_text(json.dumps(
        {"fold": fold, "ok": bool(ok and reload_ok), "val": summ, "logged": ref, "recalibrated": names,
         "nonfinite_src": bad}, indent=2, default=str))
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def train_fold(fold, dev):
    torch, nn, F, DataLoader, TrainDS, EvalDS, ApoNet = torch_parts()
    smp = import_smp()
    seed = SEED + 100 * fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr, va = fold_split(fold)
    pl = [u for u in UNL if u.get("pseudo") is not None]
    vl_items = [u for u in UNL if u.get("lines") is not None]
    if SMOKE:
        pl = pl[:12]
        vl_items = vl_items[:3]
    n_pl = int(round(len(tr) * PSEUDO_FRAC / (1 - PSEUDO_FRAC))) if pl else 0
    gts = [gt_mt(it["mask"], it["vp"]) for it in va]
    va_tfs = [letterbox(it["img"], it["vp"], SIZE)[1] for it in va]
    tpath = OUT / "teacher_ckpt" / f"{TEACHER_NAME}_fold{fold}.pt"
    model = build_model(smp, ApoNet, ENCODER, pretrained=False)
    bad = load_fp32(torch, model, tpath)
    if bad:
        raise RuntimeError(f"fold {fold}: teacher checkpoint non-finite {bad[:5]}")
    model = model.to(dev).to(memory_format=torch.channels_last)
    ema = build_model(smp, ApoNet, ENCODER, pretrained=False).to(dev).to(memory_format=torch.channels_last)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    # FIX (RunPod run, 2026-10-21): the EMA copy must stay in eval mode. A freshly built module is in train mode, so
    # val_metrics / vl_lines_metric ran it with batch-statistics BatchNorm and their forward passes overwrote its
    # running stats with val / VL batches: best-epoch selection used train-mode scores and every saved student
    # checkpoint failed the eval-mode reload check (then BN-recalibrated, worse than the teacher on its fold).
    ema.eval()
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
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    kw = dict(num_workers=NUM_WORKERS)
    if NUM_WORKERS:
        kw["persistent_workers"] = True
    tl = DataLoader(TrainDS(tr, pl, n_pl), batch_size=BATCH, shuffle=True, pin_memory=dev.type == "cuda",
                    drop_last=True, **kw)
    steps_per_epoch = len(tl) if MAX_STEPS is None else min(len(tl), MAX_STEPS)
    total_steps = steps_per_epoch * EPOCHS
    warm = max(1, int(WARMUP_EPOCHS * steps_per_epoch))
    log(f"fold {fold}: train={len(tr)} pseudo_pool={len(pl)} pseudo/epoch={n_pl} val={len(va)} vl={len(vl_items)} "
        f"steps/epoch={steps_per_epoch}")
    hist_path = OUT / f"fold{fold}_history.csv"
    ck_path = OUT / "ckpt" / f"{MODEL_NAME}_fold{fold}.pt"
    hist_cols = ["epoch", "train_loss", "l_band", "l_up", "l_lo", "dice", "mt_mae_px_band", "mt_mae_px_edge",
                 "mt_medae_px_edge", "mt_rel_band", "mt_rel_edge", "mt_rel_bias_edge", "pseudo_mm_mae_edge", "mt_fail",
                 "n_eval", "score", "vl_ratio_med", "vl_ratio_spread", "vl_fail", "lr_enc", "sec"]
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(hist_cols)

    def record(rec):
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([rec.get(c) for c in hist_cols])

    t0 = time.time()
    summ0, _ = val_metrics(torch, DataLoader, EvalDS, ema, va, va_tfs, gts, dev)
    vm0 = vl_lines_metric(torch, ema, vl_items, dev)
    record({"epoch": 0, **summ0, **vm0, "sec": time.time() - t0})
    log(f"fold {fold}: epoch 0 (teacher) dice={summ0['dice']:.4f} score={summ0['score']:.4f} fail={summ0['mt_fail']} "
        f"VL {json.dumps(vm0)}")
    best, best_epoch, best_summ, best_rows, best_vl = summ0["score"], 0, summ0, None, vm0
    save_fp32(torch, ema, ck_path, {"encoder": ENCODER, "fold": fold, "epoch": 0, "val": summ0, "model_name": MODEL_NAME})
    step = 0
    epoch_times = []
    for epoch in range(1, EPOCHS + 1):
        est = np.mean(epoch_times) if epoch_times else 0.0
        if time.time() - START + est > TRAIN_DEADLINE:
            log(f"fold {fold}: deadline reached before epoch {epoch}; stopping")
            break
        t0 = time.time()
        model.train()
        run, nb, chs = 0.0, 0, np.zeros(3)
        for bi, (xb, yb, wb, wi) in enumerate(tl):
            if MAX_STEPS is not None and bi >= MAX_STEPS:
                break
            if step < warm:
                fac = 0.05 + 0.95 * step / max(1, warm)
            else:
                pr = (step - warm) / max(1, total_steps - warm)
                fac = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
            for g_, bl in zip(opt.param_groups, base_lrs):
                g_["lr"] = bl * fac
            xb = xb.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            yb, wb, wi = yb.to(dev, non_blocking=True), wb.to(dev, non_blocking=True), wi.to(dev)
            opt.zero_grad(set_to_none=True)
            with amp(torch, dev):
                logits = model(xb)
            loss, per = loss_fn(logits, yb, wb, wi)
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
        ema.eval()
        summ, rows = val_metrics(torch, DataLoader, EvalDS, ema, va, va_tfs, gts, dev)
        vm = vl_lines_metric(torch, ema, vl_items, dev)
        sec = time.time() - t0
        epoch_times.append(sec)
        lr_now = opt.param_groups[0]["lr"]
        rec = {"epoch": epoch, "train_loss": run / max(1, nb), "l_band": chs[0] / max(1, nb), "l_up": chs[1] / max(1, nb),
               "l_lo": chs[2] / max(1, nb), **summ, **vm, "lr_enc": lr_now, "sec": sec}
        record(rec)
        log(f"EPOCH fold={fold} ep={epoch:02d}/{EPOCHS} loss={rec['train_loss']:.4f} dice={summ['dice']:.4f} "
            f"MT_MAE_px edge={summ['mt_mae_px_edge']:.2f} (med {summ['mt_medae_px_edge']:.2f}) rel edge="
            f"{summ['mt_rel_edge']:.4f} bias={summ['mt_rel_bias_edge']:+.4f} fail={summ['mt_fail']}/{summ['n_eval']} "
            f"score={summ['score']:.4f} | VL ratio={vm.get('vl_ratio_med', float('nan')):.4f} spread="
            f"{vm.get('vl_ratio_spread', float('nan')):.4f} fail={vm.get('vl_fail')} | lr={lr_now:.2e} "
            f"train_s={t_train:.0f} total_s={sec:.0f}")
        if summ["score"] < best - 1e-6:
            best, best_epoch, best_summ, best_rows, best_vl = summ["score"], epoch, summ, rows, vm
            save_fp32(torch, ema, ck_path, {"encoder": ENCODER, "fold": fold, "epoch": epoch, "val": summ,
                                            "model_name": MODEL_NAME})
            log(f"fold {fold}: saved best epoch {epoch} score={best:.4f}")
    # ---- reload from disk, check finiteness, re-validate before any maskbank is written
    m2 = build_model(smp, ApoNet, ENCODER, pretrained=False)
    bad = load_fp32(torch, m2, ck_path)
    m2 = m2.to(dev).to(memory_format=torch.channels_last).eval()
    s2, rows2 = val_metrics(torch, DataLoader, EvalDS, m2, va, va_tfs, gts, dev)
    ok = not bad and abs(s2["score"] - best_summ["score"]) <= VERIFY_TOL_SCORE and \
        abs(s2["dice"] - best_summ["dice"]) <= VERIFY_TOL_DICE
    recal = False
    if not ok:
        log(f"fold {fold}: reload check FAILED (nonfinite={bad}, {s2['dice']:.4f}/{s2['score']:.4f}) -> BN recalibration")
        recalibrate(torch, m2, tr, dev)
        recal = True
        s2, rows2 = val_metrics(torch, DataLoader, EvalDS, m2, va, va_tfs, gts, dev)
        ok = abs(s2["score"] - best_summ["score"]) <= 2 * VERIFY_TOL_SCORE and \
            abs(s2["dice"] - best_summ["dice"]) <= 2 * VERIFY_TOL_DICE
        if ok:
            save_fp32(torch, m2, ck_path, {"encoder": ENCODER, "fold": fold, "epoch": best_epoch, "val": s2,
                                           "model_name": MODEL_NAME, "bn_recalibrated": True})
    log(f"fold {fold}: reload check {'OK' if ok else 'FAILED'} (recal={recal}) dice={s2['dice']:.4f} "
        f"score={s2['score']:.4f} vs best {best_summ['dice']:.4f}/{best_summ['score']:.4f}")
    rows_out = best_rows if best_rows is not None else rows2
    if rows_out:
        with open(OUT / f"fold{fold}_val_per_image.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    with open(OUT / f"fold{fold}_metrics.json", "w") as f:
        json.dump({"fold": fold, "best_epoch": best_epoch, "best_score": best, "best": best_summ, "best_vl": best_vl,
                   "teacher": summ0, "teacher_vl": vm0, "reload_ok": ok, "reload": s2, "bn_recalibrated": recal,
                   "epochs_run": len(epoch_times), "mean_epoch_sec": float(np.mean(epoch_times)) if epoch_times else None,
                   "n_train": len(tr), "n_val": len(va), "n_pseudo_pool": len(pl), "n_pseudo_epoch": n_pl}, f,
                  indent=2, default=str)
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def fold_worker(gpu, counter, lock, job, fold_list):
    global TAG
    TAG = f"[gpu{gpu}] "
    if not CPU_SMOKE:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    code = 0
    import torch

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4 if CPU_SMOKE else 1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while True:
        with lock:
            j = counter.value
            counter.value += 1
        if j >= len(fold_list):
            break
        fold = fold_list[j]
        t0 = time.time()
        try:
            (repair_teacher if job == "repair" else train_fold)(fold, dev)
            log(f"{job} fold {fold} done in {(time.time() - t0) / 60:.1f} min")
        except Exception:
            code = 1
            log(f"WORKER ERROR {job} fold {fold}\n" + traceback.format_exc())
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
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            and not p.name.startswith("._") and "__MACOSX" not in p.parts]


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
        if key.startswith("vl"):
            imgs = [p for p in imgs if p.suffix.lower() == ".tiff"]  # raw images; .tif copies carry the drawn lines
        log(f"external {key}: {len(imgs)} images")
        name = "vl" if key.startswith("vl") else key
        sets.setdefault(name, []).extend((key, p) for p in imgs)
    return sets


class NpzWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
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


def merge_npz(parts: list[Path], dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    zout = zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True)
    seen, n = set(), 0
    for part in parts:
        with zipfile.ZipFile(part) as zin:
            for name in zin.namelist():
                if name in seen:
                    continue
                seen.add(name)
                with zin.open(name) as s, zout.open(name, "w", force_zip64=True) as d:
                    shutil.copyfileobj(s, d, 1 << 20)
                n += 1
    zout.close()
    return n


def infer_worker(gpu, model_name, ckpt_dir, folds_ok, jobs):
    """jobs: list of (dataset, key, item_index or ('train', i)). Writes part npz files per (dataset, channel)."""
    global TAG
    TAG = f"[inf{gpu}] "
    if not CPU_SMOKE:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    code = 0
    try:
        torch, nn, F, DataLoader, TrainDS, EvalDS, ApoNet = torch_parts()
        smp = import_smp()
        torch.set_num_threads(4 if CPU_SMOKE else 1)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = {}
        for k in folds_ok:
            m = build_model(smp, ApoNet, ENCODER, pretrained=False)
            bad = load_fp32(torch, m, ckpt_dir / f"{model_name}_fold{k}.pt")
            if bad:
                raise RuntimeError(f"{model_name} fold {k}: non-finite tensors at inference {bad[:5]}")
            models[k] = m.to(dev).eval().to(memory_format=torch.channels_last)
        all_models = [models[k] for k in sorted(models)]
        writers = {}
        t0 = time.time()
        for n_done, (ds, key, ref) in enumerate(jobs):
            if ds == "train_oof":
                it = ITEMS[ref]
                f = int(FOLDS[ref])
                if f not in models:
                    continue
                gray, vp, mlist = it["img"], it["vp"], [models[f]]
            else:
                u = UNL[ref]
                gray = read_gray(u["path"])
                vp, mlist = u["vp"], all_models
            canvas, tf, _ = letterbox(gray, vp, SIZE)
            x = torch.from_numpy(canvas.astype(np.float32)[None, None] / 255.0).to(dev).to(memory_format=torch.channels_last)
            with torch.inference_mode():
                pr = predict_batch(torch, mlist, x, flip=True)[0].cpu().numpy()
            pn = to_native(pr, tf)
            for c, ch in enumerate(CHANS):
                if (ds, ch) not in writers:
                    writers[(ds, ch)] = NpzWriter(PARTS / model_name / f"w{gpu}_{ds}__{model_name}_{ch}.npz")
                writers[(ds, ch)].add(key, np.clip(np.rint(pn[c] * 255.0), 0, 255).astype(np.uint8))
            if n_done % 200 == 0:
                log(f"{model_name} inference {n_done}/{len(jobs)} ({time.time() - t0:.0f}s)")
        for w in writers.values():
            w.close()
        log(f"{model_name} inference done: {len(jobs)} images in {time.time() - t0:.0f}s")
    except Exception:
        code = 2
        log("INFERENCE ERROR\n" + traceback.format_exc())
    os._exit(code)


def run_inference(ctx, model_name, ckpt_dir, folds_ok, n_dev):
    jobs = [(u["ds"], u["key"], i) for i, u in enumerate(UNL)]
    jobs += [("train_oof", it["id"], i) for i, it in enumerate(ITEMS) if int(FOLDS[i]) in folds_ok]
    if SMOKE:
        jobs = [j for j in jobs if j[0] != "train_oof"] + [j for j in jobs if j[0] == "train_oof"][:SMOKE_INFER]
    shards = [jobs[g::n_dev] for g in range(n_dev)]
    procs = []
    for g in range(n_dev):
        p = ctx.Process(target=infer_worker, args=(g, model_name, ckpt_dir, folds_ok, shards[g]), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    log(f"{model_name} inference exit codes {[p.exitcode for p in procs]}")
    out = {}
    for ds in sorted({j[0] for j in jobs}):
        for ch in CHANS:
            parts = sorted((PARTS / model_name).glob(f"w*_{ds}__{model_name}_{ch}.npz"))
            out[(ds, ch)] = merge_npz(parts, BANK / f"{ds}__{model_name}_{ch}.npz") if parts else 0
        log(f"maskbank {ds}__{model_name}: {out[(ds, 'band')]} maps")
    shutil.rmtree(PARTS / model_name, ignore_errors=True)
    return out


def prep_pseudo(model_name):
    """Teacher maps (just written) -> pseudo-label crops for the unlabeled training images."""
    files = {}
    stats = []
    for u in UNL:
        if not u["train_pl"]:
            continue
        maps = []
        for ch in CHANS:
            f = files.get((u["ds"], ch))
            if f is None:
                p = BANK / f"{u['ds']}__{model_name}_{ch}.npz"
                if not p.exists():
                    break
                f = files[(u["ds"], ch)] = np.load(p)
            if u["key"] not in f.files:
                break
            maps.append(f[u["key"]])
        if len(maps) != 3:
            continue
        x1, y1, x2, y2 = u["vp"]
        band = maps[0][y1:y2, x1:x2]
        pos = band >= int(PS_POS * 255)
        unc = (band > int(PS_NEG * 255)) & ~pos
        pair = superficial_pair(band_components((band >= 128).astype(np.uint8), x2 - x1))
        frac_unc = float(unc.sum() / max(1, (band >= 128).sum()))
        use = pair is not None and frac_unc <= PS_MAX_UNCERTAIN
        stats.append({"ds": u["ds"], "key": u["key"], "pair": pair is not None, "frac_uncertain": frac_unc, "use": use})
        if use:
            gray = read_gray(u["path"])
            u["img"] = np.ascontiguousarray(gray[y1:y2, x1:x2])
            u["vp_train"] = (0, 0, x2 - x1, y2 - y1)
            u["pseudo"] = np.ascontiguousarray(np.stack([band, maps[1][y1:y2, x1:x2], maps[2][y1:y2, x1:x2]], -1))
    return stats


def decode_vl_lines() -> dict:
    recs = json.loads(zlib.decompress(base64.b64decode("".join(VL_LINES_B64.split()))).decode())
    return {r["tail"]: r for r in recs}


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


def find_source():
    """apo_v2 kernel output attached via kernel_sources: <root>/apo_v2/{ckpt,folds.csv,fold*_metrics.json}."""
    hits = sorted(Path("/kaggle/input").rglob(f"{SRC_MODEL}_fold0.pt"), key=lambda p: len(str(p)))
    if not hits:
        return None
    ck_dir = hits[0].parent
    root = ck_dir.parent
    src = {"root": root, "ckpt": {}, "metrics": {}, "folds": {}}
    for k in range(N_FOLDS):
        p = ck_dir / f"{SRC_MODEL}_fold{k}.pt"
        if p.exists():
            src["ckpt"][k] = p
        mp_ = root / f"fold{k}_metrics.json"
        if mp_.exists():
            src["metrics"][k] = json.loads(mp_.read_text())
    fc = root / "folds.csv"
    if fc.exists():
        with open(fc) as fh:
            for r in csv.DictReader(fh):
                src["folds"][r["image_id"]] = int(r["fold"])
    return src


# --------------------------------------------------------------------------------------
def main():
    global ITEMS, FOLDS, UNL, DOMAIN_BANK, SRC, ENC_EXTRA
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ckpt").mkdir(exist_ok=True)
    log(f"RUN_MODE={RUN_MODE} self_train={SELF_TRAIN} domain_aug={DOMAIN_AUG} epochs={EPOCHS} batch={BATCH}")
    random.seed(SEED)
    np.random.seed(SEED)
    n_gpu = 0 if CPU_SMOKE else gpu_count()
    log(f"gpus={n_gpu} cpus={os.cpu_count()}")
    if n_gpu == 0 and not CPU_SMOKE:
        raise RuntimeError("no GPU")
    SRC = find_source()
    if SRC is None:
        raise RuntimeError(f"{SRC_MODEL} checkpoints not found under /kaggle/input (attach shivateja005/umud-apo-v2)")
    log(f"source {SRC['root']}: ckpts {sorted(SRC['ckpt'])} metrics {sorted(SRC['metrics'])} folds {len(SRC['folds'])}")
    ENC_EXTRA = {"drop_path_rate": 0.1}  # as resolved by apo_v2 (does not change the state-dict keys)

    t0 = time.time()
    items, n_pairs, dups = load_training_items()
    log(f"pairs={n_pairs} unique={len(items)} exact_dups_removed={dups} load_s={time.time() - t0:.0f}")
    groups = make_groups(items)
    folds = group_kfold(groups, N_FOLDS, SEED)
    n_diff = n_miss = 0
    for i, it in enumerate(items):
        f = SRC["folds"].get(it["id"])
        if f is None:
            n_miss += 1
        elif f != folds[i]:
            n_diff += 1
            folds[i] = f
    log(f"folds vs apo_v2 folds.csv: differ={n_diff} missing={n_miss} (apo_v2 folds are authoritative)")
    if SMOKE:
        sel = np.sort(np.random.default_rng(SEED).choice(len(items), SMOKE_IMAGES, replace=False))
        items = [items[i] for i in sel]
        folds = folds[sel]
    ITEMS, FOLDS = items, folds
    train_folds = SMOKE_FOLDS if SMOKE else list(range(N_FOLDS))
    with open(OUT / "folds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file", "layout", "group", "fold", "vx1", "vy1", "vx2", "vy2"])
        for it, g, fo in zip(items, groups if not SMOKE else groups[sel], folds):
            w.writerow([it["id"], it["file"], it["layout"], int(g), int(fo), *it["vp"]])

    # ---- unlabeled sets (keys as in apo_v2: file stem, repeats get __2)
    test_dir = find_dir(COMP_ROOT, "test_set_v2")
    tpaths = list_images(test_dir)
    entries = [("test", p, p.name) for p in tpaths]
    try:
        ext = external_datasets()
    except Exception as e:  # noqa
        log(f"external datasets failed: {e!r}")
        ext = {}
    for name in ("static", "dynamic", "vl"):
        for sub, p in ext.get(name, []):
            entries.append((name, p, f"{sub}/{p.relative_to(EXT_DIR / sub)}"))
    if SMOKE:
        keep = []
        for ds in ("test", "static", "dynamic", "vl"):
            keep += [e for e in entries if e[0] == ds][:SMOKE_INFER + (2 if ds == "vl" else 0)]
        entries = keep
    lines = decode_vl_lines()
    used: dict[str, set] = {}
    UNL = []
    for ds, p, src in entries:
        try:
            g = read_gray(p)
        except Exception as e:  # noqa
            log(f"skip {p}: {e!r}")
            continue
        if min(g.shape) < 64:
            continue
        k, j = p.stem, 1
        while k in used.setdefault(ds, set()):
            j += 1
            k = f"{p.stem}__{j}"
        used[ds].add(k)
        j_ds = sum(1 for u in UNL if u["ds"] == ds)
        u = {"ds": ds, "key": k, "path": p, "src": src, "H": g.shape[0], "W": g.shape[1], "vp": detect_viewport(g),
             "train_pl": SELF_TRAIN and (ds != "dynamic" or j_ds % DYN_STRIDE == 0)}
        if ds == "vl":
            L = lines.get(tail3(src))
            if L is not None and L["H"] == g.shape[0] and L["W"] == g.shape[1]:
                u["lines"] = L
                u["vl_img"] = g  # full raw image for the per-epoch VL diagnostic (lines are in raw coords)
        UNL.append(u)
    counts = {d: sum(u["ds"] == d for u in UNL) for d in ("test", "static", "dynamic", "vl")}
    log(f"unlabeled images: {len(UNL)} {counts}; VL with expert lines: {sum('lines' in u for u in UNL)}")

    smp = import_smp()
    import torch  # noqa: F401  (CPU only in the parent)

    ctx = mp.get_context("fork")
    n_dev = 1 if CPU_SMOKE else max(1, min(n_gpu, 2))

    def run_folds(job, fold_list):
        counter = ctx.Value("i", 0)
        lock = ctx.Lock()
        procs = []
        for g in range(max(1, min(n_dev, len(fold_list)))):
            p = ctx.Process(target=fold_worker, args=(g, counter, lock, job, fold_list), daemon=False)
            p.start()
            procs.append(p)
            time.sleep(2)
        for p in procs:
            p.join()
        log(f"{job} workers exit codes {[p.exitcode for p in procs]}")

    # ---- phase 1: repaired teacher (fp32) + 5-fold x hflip TTA maps
    run_folds("repair", train_folds)
    t_ok = []
    for k in train_folds:
        mp_ = OUT / f"teacher_fold{k}_metrics.json"
        if mp_.exists() and json.loads(mp_.read_text()).get("ok"):
            t_ok.append(k)
    log(f"teacher folds verified: {t_ok}")
    if not t_ok:
        raise RuntimeError("no teacher fold verified; no maskbank written")
    run_inference(ctx, TEACHER_NAME, OUT / "teacher_ckpt", t_ok, n_dev)

    s_ok = []
    if SELF_TRAIN:
        # ---- phase 2: pseudo-labels from the teacher maps, student fine-tuning
        stats = prep_pseudo(TEACHER_NAME)
        with open(OUT / "pseudo_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ds", "key", "pair", "frac_uncertain", "use"])
            w.writeheader()
            w.writerows(stats)
        n_use = sum(s["use"] for s in stats)
        per_ds = {d: sum(bool(s["use"]) and s["ds"] == d for s in stats) for d in ("test", "static", "dynamic", "vl")}
        log(f"pseudo-labels: {n_use}/{len(stats)} images usable {per_ds}")
        bank = [u for u in UNL if u.get("pseudo") is not None or u["ds"] == "test"]
        DOMAIN_BANK = np.stack([letterbox(read_gray(u["path"]), u["vp"], SIZE)[0] for u in bank]) if bank else None
        log(f"domain bank {0 if DOMAIN_BANK is None else len(DOMAIN_BANK)} canvases")
        run_folds("train", [k for k in train_folds if k in t_ok])
        for k in train_folds:
            mp_ = OUT / f"fold{k}_metrics.json"
            if mp_.exists() and json.loads(mp_.read_text()).get("reload_ok"):
                s_ok.append(k)
        log(f"student folds verified: {s_ok}")
        if s_ok:
            run_inference(ctx, MODEL_NAME, OUT / "ckpt", s_ok, n_dev)

    # ---- manifest / config
    with open(BANK / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "image_id", "source_filename", "height", "width"])
        for u in UNL:
            w.writerow([u["ds"], u["key"], u["src"], u["H"], u["W"]])
        for it in ITEMS:
            w.writerow(["train_oof", it["id"], it["file"], *it["img"].shape])
    with open(BANK / f"{MODEL_NAME}_viewports.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "image_id", "x1", "y1", "x2", "y2"])
        for u in UNL:
            w.writerow([u["ds"], u["key"], *u["vp"]])
        for it in ITEMS:
            w.writerow(["train_oof", it["id"], *it["vp"]])
    summ = {}
    for tag, pat in (("teacher", "teacher_fold{}_metrics.json"), ("student", "fold{}_metrics.json")):
        rows = [json.loads((OUT / pat.format(k)).read_text()) for k in train_folds if (OUT / pat.format(k)).exists()]
        for key in ("dice", "score", "mt_rel_edge", "mt_mae_px_edge", "mt_fail"):
            src = "val" if tag == "teacher" else "best"
            vals = [r[src][key] for r in rows if r.get(src) and key in r[src]]
            if vals:
                summ[f"{tag}_cv_{key}"] = float(np.mean(vals))
        if tag == "student":
            for key in ("vl_ratio_spread", "vl_fail"):
                for s2 in ("teacher_vl", "best_vl"):
                    vals = [r[s2][key] for r in rows if r.get(s2) and key in r[s2]]
                    if vals:
                        summ[f"{s2}_{key}"] = float(np.mean(vals))
    config = {"model_name": MODEL_NAME, "teacher_name": TEACHER_NAME, "run_mode": RUN_MODE, "self_train": SELF_TRAIN,
              "encoder": ENCODER, "size": SIZE, "epochs": EPOCHS, "batch": BATCH, "lr_enc": LR_ENC, "lr_dec": LR_DEC,
              "ema": EMA_DECAY, "pseudo": {"frac": PSEUDO_FRAC, "w": W_PS, "pos": PS_POS, "neg": PS_NEG,
                                           "max_uncertain": PS_MAX_UNCERTAIN, "dyn_stride": DYN_STRIDE},
              "domain_aug": DOMAIN_AUG, "teacher_ok_folds": t_ok, "student_ok_folds": s_ok, "summary": summ,
              "maskbank": "uint8 prob*255 at native resolution; key = file stem (repeats __2); 5-fold x hflip "
                          "(train_oof: the fold model that held the image out)",
              "elapsed_min": (time.time() - START) / 60}
    (OUT / "config.json").write_text(json.dumps(config, indent=2, default=str))
    log(f"summary {json.dumps(summ)}")
    log(f"ALL DONE total_minutes={(time.time() - START) / 60:.1f}")


# Expert-drawn aponeurosis lines of the UMUD 'Example Annotated Images' (VL, OSF xbawc, CC-BY-4.0), parsed from the
# burned-in ImageJ lines: per raw .tiff (keyed by the last three path parts) the superficial / deep line as point +
# direction in raw-image pixels. Used only for the per-epoch VL diagnostic. zlib + base64 JSON.
VL_LINES_B64 = """
eNqdXUvPLLlt/S933WlL1IuaXXZepZdBYAwGk2QMTJDMNeCbRRDkv4dHZFWXWlW6qrswPLb7c5+W+OYh9Zf//fLt19//88tPX/74
9X9+8X/69vt//fa3r7//8c3/6dc/vv7xCz2//f7Xv355fPn7f//rf/z2b9/kg//0j//yi5f/5s9ffsqOH1/++ctP7Bw+8bcvP/0l
e/8kH8uDiJ/FVXq4Z+XiH//gnj7//Pjy77/9hs8VV56llkd0/hmTZ3ysBv2YL/Tz/z2+gyzeRUY5CoIswMIzluzbN2bGN7pS0wFa
jE+XSn2ExAKm/YJKpUEjVwZobsAW+AKcu0bn6tMHOZCQ/LMwR3wrDki+lZyLB3iZBF7A0YWnLym3T3LST+a4gC+62/h85CfFGB+B
/JODHh/X2E4lpg4fy/+SvOATQIW94mPFF8fzO7nbK7G7vt1Q8C0lPuSGns63Q6l6uTnUN7pcisgby+UKTBeKikFgFTzHK+jibXTk
5VsFx8OzfzpH8SH/1sAdzi3FZ5ULFAkggZRq+ww+8gmJRkj5AhJdX6gIaXUpPYKvz+ijHkTUCxXxPwCTo02PSOmZKKs2qLS5FPMK
OL4PjsQ6pPCg7J852m3GpNhiOAqbewaoSaj0TM5+RVGlLn5AFwZ06UoXwgSdHF1I8qU5iITXhC/N3I6OYmdKcn76JJ+MPjxTIrWG
WY1h5hV4dB+eXGgWiXvA4uWsJriK8LXTq/UoclV0RfAEBqBSGjzWT0YOA7444rtShnitqnK5VLx/UIIBs29NamAzH5yEnNyTcO9B
zrFGp59UES1MK/DybXiC6FmCo4ev/IwidirxaksiHxUjhqergR7E/AyhqhMjvV7yo/SlEd+VbqRrW8L1ydAJXzNcrV5vVXiJDqaO
RSUyyelxkh+iLqVWr5+ktAAvu/vwxBaFJEInNyQGIwb9Vj0+z0dHEdKziDd+BLEQnlU35A+b8OXRT+QR3pVy5Am88gwIS4jFJDvO
KvJq/2M9mhZqZk/cSWKnHytqgVJJK+DiD4CTGy1B7J5PTzEopb+xo+hlscbwrcFLmJVKu1v2iq/60SyXEd+VapRrfCQhlBwZRNAT
qbF1ai5CPChurvEpDpY0vJNQUKMZiwGcX4HH9+GJ4RNVFHwB7oOThnjt9CRUOZyeWGMRRZZgBmZZQr0WPDR0PozoeEBXrhSDL9HF
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
"""

if __name__ == "__main__":
    main()
