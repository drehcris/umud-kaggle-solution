"""UMUD fasc_orient_v2: fascicle fragment segmentation + doubled-angle orientation field, adapted to the TEST domain.

Same network / encoding as fasc_orient_v1 (smp U-Net, ConvNeXt-S fb_in22k_ft_in1k, 3 channels: fragment logit,
tanh cos2theta, tanh sin2theta; theta in image pixel coords, x right, y down). What is new:

  (a) self-training (noisy student): the unlabeled test (309), static (35), dynamic (every 4th of 168) and VL raw
      (249) images are added to training with pseudo-labels taken from the teacher maskbank (round 1: the v1
      5-fold x hflip ensemble, attached via kernel_sources). Orientation pseudo-targets are kept only where the
      5-fold agreement |v| >= 0.97 and the 25x25 field coherence >= 0.95, near a predicted fragment (gates
      calibrated against the VL expert lines: kept pixels 1.27 deg mean error, rejected 4.6 deg); segmentation
      pseudo-targets are hard (p >= 0.6 / p <= 0.15) with an ignore band. No hand labels of test images.
  (b) multi-scale: zoom-crop augmentation (both dims, up to 2.2x magnification) + inference on the full viewport
      and a 2x2 grid of overlapping zoom tiles (1/0.6 = 1.67x), written as a second maskbank model *_v2ms.
  (c) stronger domain augmentation: Fourier-domain amplitude swap / histogram matching to the unlabeled target
      images, random tone curves, log compression, acoustic-shadow stripes, lateral blur, CLAHE, wider geometry.
  (d) optional (VL_SUP): the 245 expert-drawn VL fascicle lines (UMUD 'Example Annotated Images', OSF xbawc,
      CC-BY-4.0; embedded below as polylines in raw-image coords) as orientation targets inside the muscle,
      with VL subjects split into the same 5 folds so that the VL maps stay out-of-fold by subject.
  Always: the expert VL lines are an out-of-domain validation metric every epoch (track angle error, deep-strip
  error and an image-level deep-strip PA proxy vs the expert PA), logged next to the GT-fold metrics.

Honest evaluation: fold k is warm-started from the WARM_MODEL fold-k checkpoint and uses the WARM_MODEL folds
(folds.csv), so the train OOF maps stay out-of-fold. Checkpoints are saved in float32 and every fold checkpoint
is reloaded from disk, checked for inf/nan and re-validated before any maskbank is written (the apo_v2 fp16
BatchNorm overflow).

Outputs (/kaggle/working): maskbank/{test,static,dynamic,vl,train_oof}__{fasc_orient_v2,fasc_orient_v2ms}_{seg,cos2,
sin2}.npz (uint8, native resolution, v1 encoding), manifest.csv, *_viewports.csv, checkpoints/fold*.pt (fp32),
metrics/ (history, fold json, cv_summary.json, folds.csv, pseudo_stats.csv), logs/, viz/, fasc_orient_v2_config.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")  # never initialise CUDA in the parent before fork
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import base64
import csv
import hashlib
import json
import math
import multiprocessing as mp
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

# OpenCV's worker pool must never start in the parent: a forked child that touches it (cv2.setNumThreads or any
# parallel cv2 op) deadlocks, because the pool threads do not exist after fork (this hung cpusmoke v1/v2).
cv2.setNumThreads(1)

# ----------------------------------------------------------------------------- config
MODE = "full"  # "full" | "smoke" (GPU, tiny subsets) | "cpusmoke" (CPU only, tiny; kernel id *-cpusmoke)
SMOKE = MODE != "full"
CPU_SMOKE = MODE == "cpusmoke"
IN_PROCESS = False  # debugging aid: run training / inference workers in the parent instead of forked processes
T0 = time.time()

MODEL_NAME = "fasc_orient_v2"
MS_NAME = MODEL_NAME + "ms"
WARM_MODEL = "fasc_orient_v1"  # fold k is initialised from this model's fold-k checkpoint (identical folds)
PSEUDO_MODEL = "fasc_orient_v1"  # teacher maskbank (round 1 = v1 ensemble; round 2 = fasc_orient_v2ms)
WARM_START = True
REQUIRE_WARM_START = False  # controlled experiments fail rather than silently changing initialization
REQUIRE_COMPLETE_VL = False  # opt-in pilot guard: every embedded expert record must match a raw image
PSEUDO = True
VL_SUP = False  # expert VL lines as training targets (subject-fold OOF); False = VL is validation only
VL_VALIDATION_BY_FOLD = False  # use the same subject-held-out diagnostics in paired control experiments
STYLE_DATASETS = None  # None keeps v2 behavior; an explicit tuple fixes the style bank across experiment arms
MULTISCALE = True
DOMAIN_AUG = True

SEED = 20261021  # same as v1 (fold reproducibility)
SIZE = 512
N_FOLDS = 5
TRAIN_FOLDS = [0] if CPU_SMOKE else ([0, 1] if SMOKE else [0, 1, 2, 3, 4])
SMOKE_TRAIN, SMOKE_VAL, SMOKE_PL = (8, 4, 6) if CPU_SMOKE else (30, 10, 24)
MAX_EPOCHS = 1 if CPU_SMOKE else (2 if SMOKE else (10 if WARM_START else 30))
MIN_EPOCHS = 1 if SMOKE else 4
MAX_STEPS = 3 if CPU_SMOKE else None  # per epoch (cpusmoke only)
TRAIN_BUDGET_S = 2.7 * 3600  # all training must end by T0 + this
HARD_TRAIN_END_S = 3.0 * 3600
BATCH = 2 if CPU_SMOKE else 8
LR_ENC, LR_DEC = (1e-4, 3e-4) if WARM_START else (2e-4, 6e-4)
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 0.3 if WARM_START else 1.0
LR_FLOOR = 0.02
EMA_DECAY = 0.998
POS_WEIGHT, W_BCE, W_DICE, W_SKEL, W_ORI = 3.0, 1.0, 1.0, 0.5, 1.0
SEG_WIDTH, ORI_RADIUS = 3, 8.0
DUP_CORR, THUMB = 0.95, 48
# self-training
PSEUDO_FRAC = 0.25  # share of pseudo-labelled samples per epoch
W_PS_SEG, W_PS_ORI = 0.5, 0.7
PS_POS, PS_NEG = 0.6, 0.15  # hard seg pseudo-target thresholds (ignored in between)
PS_MAG, PS_COH, PS_COH_K, PS_NEAR = 0.97, 0.95, 25, 0.3
PS_MIN_GATE_FRAC = 0.01  # drop pseudo images with < 1 % confident orientation pixels in the viewport
DYN_STRIDE = 4
# VL expert lines
VL_ORI_RADIUS, W_VL_ORI = 6.0, 1.0
VL_STRIP = (0.04, 0.30)  # deep strip (fraction of the local apo distance above the deep line), as in geometry v2
# augmentation
P_ZOOM, ZOOM_MIN, P_NARROW = 0.25, 0.45, 0.30
ROT_DEG, ASPECT, SCALE_RANGE = 15.0, 0.18, (0.7, 1.3)
P_FDA, P_HISTM = 0.20, 0.20
# multi-scale inference
TILE_FRAC = 0.6
# checks
VERIFY_TOL_ANG, VERIFY_TOL_DICE = 0.05, 0.01
N_CAL = 8 if CPU_SMOKE else 160
ENCODER = "tu-convnext_small.fb_in22k_ft_in1k"
SMP_VERSION = "0.5.0"
TEST_LIMIT = 2 if CPU_SMOKE else (5 if SMOKE else None)
EXTRA_LIMIT = 1 if CPU_SMOKE else (3 if SMOKE else None)
SEG_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

COMP_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
INPUT_ROOT = Path(os.environ.get("UMUD_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("UMUD_OUTPUT_DIR", "/kaggle/working"))
TMP = Path(os.environ.get("UMUD_TEMP_DIR", "/tmp/fo2_tmp"))
MASKBANK = WORK / "maskbank"
CKPT_DIR = WORK / "checkpoints"
LOG_DIR = WORK / "logs"
METRIC_DIR = WORK / "metrics"
VIZ_DIR = WORK / "viz"
PART_DIR = TMP / "parts"
EXTRA_SOURCES = {
    "static": [("https://osf.io/download/hb5xj/", "static.zip")],
    "dynamic": [("https://osf.io/download/vqb7h/", "dynamic.zip")],
    "vl": [
        ("https://osf.io/download/zcuwk/", "vl_young.zip"),
        ("https://osf.io/download/6ga94/", "vl_old.zip"),
    ],
}
CHANNELS = ("seg", "cos2", "sin2")

ITEMS: list[dict] = []  # GT items (filled in main; inherited by forked workers)
UNL: list[dict] = []  # unlabeled items (pseudo-labelled training and/or inference)
BANK_IMGS: np.ndarray | None = None  # letterboxed target-domain canvases for FDA / histogram matching
REF_METRICS: dict = {}  # WARM_MODEL fold metrics (for the warm-start reproduction check)
WARM_CKPTS: dict = {}  # fold -> checkpoint path of WARM_MODEL
_LOG_PREFIX = ""
_LOG_FILE: Path | None = None


# ----------------------------------------------------------------------------- logging
def log(msg: str) -> None:
    line = f"[{(time.time() - T0) / 60:7.2f}m]{_LOG_PREFIX} {msg}\n"
    try:
        os.write(1, line.encode())
    except OSError:
        print(line, end="", flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line)


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:  # noqa: BLE001
        return f"<{exc!r}>"


def tail3(p: str) -> str:
    return "/".join(Path(str(p).replace("\\", "/")).parts[-3:]).lower()


# ----------------------------------------------------------------------------- viewport (identical to v1)
def _largest_run(values: np.ndarray) -> tuple[int, int]:
    padded = np.pad(values.astype(np.int8), 1)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, len(values)
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def _first_true(values: np.ndarray, start: int = 0) -> int | None:
    idx = np.flatnonzero(values[start:])
    return int(start + idx[0]) if len(idx) else None


def _last_true(values: np.ndarray) -> int | None:
    idx = np.flatnonzero(values)
    return int(idx[-1]) if len(idx) else None


def texture_viewport(gray: np.ndarray) -> tuple[int, int, int, int]:
    h, w = gray.shape
    g = gray.astype(np.float32)
    x1, x2 = _largest_run(np.abs(np.diff(g, axis=0)).mean(axis=0) > 1.0)
    if x2 - x1 < w * 0.35:
        bright = gray > max(8, int(np.percentile(gray, 35)))
        x1, x2 = _largest_run(bright.mean(axis=0) > 0.35)
        if x2 - x1 < w * 0.2:
            x1, x2 = 0, w
    y1, y2 = _largest_run(np.abs(np.diff(g[:, x1:x2], axis=1)).mean(axis=1) > 0.5)
    if y2 - y1 < h * 0.45:
        y1, y2 = 0, h
    return x1, y1, x2, y2


def layout_viewport(gray: np.ndarray, suffix: str) -> tuple[tuple[int, int, int, int], str] | None:
    """Scanner-layout rules adapted from AmbrosM, "UMUD Quick and Dirty"
    (https://www.kaggle.com/code/ambrosm/umud-quick-and-dirty), Apache-2.0; see THIRD_PARTY_NOTICES.md."""
    h, w = gray.shape
    if suffix == ".png" and (w, h) == (1200, 800):
        first = _first_true(gray[:, 6] > 50)
        second_minor = _first_true(gray[:, 6] > 50, 150)
        second_major = _first_true(gray[:, 9] > 50, 150)
        last = _last_true(gray[:, 6] > 50)
        if None not in (first, second_minor, second_major, last):
            half = w // 2
            span = int(np.argmin(gray[:, half:].sum(axis=0)))
            if span > 40 and last - first > 200:
                return (half - span, first, half + span, last), "philips_lumify"
    if (w, h) == (1200, 800):
        if np.all(np.abs(gray[87, 1147:1157].astype(int) - 175) <= 3):
            first = _first_true(gray[:, 1150] > 50)
            last = _last_true(gray[:, 1150] > 50)
            if first is not None and last is not None:
                second = _first_true(gray[:, 1150] > 50, first + 20)
                if second is not None and second > first:
                    n = round((last - first) / (second - first))
                    boxes = {7: 142, 8: 163, 9: 211, 10: 249, 11: 282, 12: 308, 13: 331, 14: 349, 15: 142}
                    if n in boxes and last > 300:
                        x1 = boxes[n]
                        return (x1, 91, 1200 - x1, last), "siemens_juniper"
        if np.all(gray[42, 67:74] > 115):
            return (171, 42, 1029, 798), "telemed_artus_1200"
    if (w, h) == (1088, 644):
        return (140, 0, 947, 643), "telemed_artus_1088"
    if h in (512, 513):
        return (0, 0, w, h - 10), "legacy_512"
    if h == 853:
        return (0, 0, w, h), "legacy_853"
    return None


def viewport(gray: np.ndarray, suffix: str) -> tuple[tuple[int, int, int, int], str]:
    h, w = gray.shape
    found = layout_viewport(gray, suffix)
    if found is None:
        box, kind = texture_viewport(gray), "texture"
    else:
        box, kind = found
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 32 or y2 - y1 < 32 or (x2 - x1) * (y2 - y1) < 0.08 * w * h:
        return (0, 0, w, h), kind + "_full"
    return (x1, y1, x2, y2), kind


# ----------------------------------------------------------------------------- fragments (identical to v1)
MIN_PTS = 6


def _fit(points: np.ndarray):
    c = points.mean(0)
    q = points - c
    evals, evecs = np.linalg.eigh(q.T @ q)
    d = evecs[:, 1]
    n = evecs[:, 0]
    return c, d, n, q @ d, q @ n


def decompose_component(points: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    segs: list[np.ndarray] = []
    rem = points
    for _ in range(8):
        if len(rem) < MIN_PTS:
            break
        c, d, n, t, nn = _fit(rem)
        if np.abs(nn).max() <= 2.0:
            a, b = c + d * (t.min() - 0.5), c + d * (t.max() + 0.5)
            segs.append(np.array([a[0], a[1], b[0], b[1]], np.float32))
            break
        best_cnt, best = 0, None
        for _ in range(80):
            i, j = rng.integers(0, len(rem), 2)
            dv = rem[j] - rem[i]
            length = float(np.hypot(*dv))
            if length < 4:
                continue
            normal = np.array([-dv[1], dv[0]]) / length
            inl = np.abs((rem - rem[i]) @ normal) < 1.25
            cnt = int(inl.sum())
            if cnt > best_cnt:
                best_cnt, best = cnt, inl
        if best is None or best_cnt < MIN_PTS:
            break
        c, d, n, _, _ = _fit(rem[best])
        q = rem - c
        t = q @ d
        inl = np.abs(q @ n) < 1.5
        ts = np.sort(t[inl])
        if len(ts) < MIN_PTS:
            break
        cuts = np.flatnonzero(np.diff(ts) > 4.0)
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, len(ts) - 1]
        k = int(np.argmax(ends - starts))
        t0, t1 = ts[starts[k]], ts[ends[k]]
        take = inl & (t >= t0 - 1e-6) & (t <= t1 + 1e-6)
        if take.sum() >= MIN_PTS:
            a, b = c + d * (t0 - 0.5), c + d * (t1 + 0.5)
            segs.append(np.array([a[0], a[1], b[0], b[1]], np.float32))
        if take.sum() == 0:
            break
        rem = rem[~take]
    return segs


def mask_segments(mask: np.ndarray, img_w: int, img_h: int, rng: np.random.Generator):
    mh, mw = mask.shape
    sx, sy = img_w / mw, img_h / mh
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    segs, all_pts, n_split = [], [], 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            continue
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        ys, xs = np.nonzero(lab[y0:y0 + hh, x0:x0 + ww] == i)
        pts = np.stack([(xs + x0 + 0.5) * sx - 0.5, (ys + y0 + 0.5) * sy - 0.5], 1)
        all_pts.append(pts)
        got = decompose_component(pts, rng)
        n_split += len(got) > 1
        segs.extend(got)
    segs_arr = np.stack(segs).astype(np.float32) if segs else np.zeros((0, 4), np.float32)
    pts_arr = np.concatenate(all_pts) if all_pts else np.zeros((0, 2))
    return segs_arr, pts_arr, n - 1, n_split


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def sha1_array(arr: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(arr.shape, np.int64).tobytes())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def thumbnail(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape
    c = crop[int(h * 0.1):max(int(h * 0.1) + 1, int(h * 0.9)), int(w * 0.1):max(int(w * 0.1) + 1, int(w * 0.9))]
    t = cv2.resize(c, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return (t / (np.linalg.norm(t) + 1e-6)).astype(np.float32)


def prep_train(pair: tuple[str, str]) -> dict:
    image_path, mask_path = Path(pair[0]), Path(pair[1])
    rng = np.random.default_rng(int(hashlib.md5(image_path.stem.encode()).hexdigest()[:8], 16))
    img = read_gray(image_path)
    mask = read_gray(mask_path) > 0
    out = {
        "stem": image_path.stem, "name": image_path.name, "mask_name": mask_path.name,
        "ih": sha1_array(img), "mh": sha1_array(np.packbits(mask)), "H": img.shape[0], "W": img.shape[1],
        "mask_shape": mask.shape, "empty": not mask.any(),
    }
    if out["empty"]:
        return out
    segs, pts, ncomp, nsplit = mask_segments(mask, img.shape[1], img.shape[0], rng)
    box, kind = viewport(img, image_path.suffix.lower())
    x1, y1, x2, y2 = box
    inside = ((pts[:, 0] >= x1 - 0.5) & (pts[:, 0] < x2 - 0.5) & (pts[:, 1] >= y1 - 0.5) & (pts[:, 1] < y2 - 0.5))
    coverage = float(inside.mean()) if len(pts) else 1.0
    if coverage < 0.97:
        box, kind = (0, 0, img.shape[1], img.shape[0]), kind + "_lowcov"
        x1, y1, x2, y2 = box
    crop = np.ascontiguousarray(img[y1:y2, x1:x2])
    segs = segs - np.array([x1, y1, x1, y1], np.float32)
    out.update(
        box=box, layout=kind, coverage=coverage, crop=crop, segs=segs.astype(np.float32),
        thumb=thumbnail(crop), ncomp=ncomp, nsplit=nsplit, empty=len(segs) == 0,
    )
    return out


# ----------------------------------------------------------------------------- geometry
def letterbox_matrix(w: int, h: int) -> np.ndarray:
    s0 = SIZE / max(w, h)
    return np.array(
        [[s0, 0.0, (SIZE - s0 * w) / 2 + 0.5 * s0 - 0.5], [0.0, s0, (SIZE - s0 * h) / 2 + 0.5 * s0 - 0.5], [0, 0, 1]],
        np.float64,
    )


def aug_matrix(rng: np.random.Generator) -> np.ndarray:
    c = (SIZE - 1) / 2.0
    lo, hi = SCALE_RANGE
    sc = math.exp(rng.uniform(math.log(lo), math.log(hi))) if rng.random() < 0.7 else 1.0
    ar = math.exp(rng.uniform(-ASPECT, ASPECT)) if rng.random() < 0.35 else 1.0
    phi = math.radians(rng.uniform(-ROT_DEG, ROT_DEG)) if rng.random() < 0.6 else 0.0
    if rng.random() < 0.5:
        tx, ty = rng.uniform(-0.06, 0.06) * SIZE, rng.uniform(-0.06, 0.06) * SIZE
    else:
        tx = ty = 0.0
    flip = -1.0 if rng.random() < 0.5 else 1.0
    tm = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], np.float64)
    fm = np.diag([flip, 1.0, 1.0])
    sm = np.diag([sc * ar, sc / ar, 1.0])
    cs, sn = math.cos(phi), math.sin(phi)
    rm = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], np.float64)
    tp = np.array([[1, 0, c + tx], [0, 1, c + ty], [0, 0, 1]], np.float64)
    return tp @ rm @ sm @ fm @ tm


def warp_image(src: np.ndarray, m: np.ndarray, border=0) -> np.ndarray:
    """Warp src (H,W[,C]) with 3x3 m (src pixel coords -> SIZE canvas), anti-aliased when shrinking."""
    h, w = src.shape[:2]
    lin = math.sqrt(abs(np.linalg.det(m[:2, :2])))
    mw = m
    if lin < 0.75:
        f = lin / 0.9
        nw, nh = max(8, int(round(w * f))), max(8, int(round(h * f)))
        src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        fx, fy = nw / w, nh / h
        pinv = np.array([[1 / fx, 0, 0.5 / fx - 0.5], [0, 1 / fy, 0.5 / fy - 0.5], [0, 0, 1]], np.float64)
        mw = m @ pinv
    return cv2.warpAffine(src, mw[:2].astype(np.float64), (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _poly_mask(m: np.ndarray, w: int, h: int) -> np.ndarray:
    corners = np.array([[-0.5, -0.5, 1], [w - 0.5, -0.5, 1], [w - 0.5, h - 0.5, 1], [-0.5, h - 0.5, 1]]).T
    p = (m @ corners)[:2].T
    out = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillConvexPoly(out, np.round(p * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    return out


_DISK = {}


def disk(r: int) -> np.ndarray:
    if r not in _DISK:
        _DISK[r] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return _DISK[r]


# ----------------------------------------------------------------------------- intensity / domain augmentation
def _fda(x: np.ndarray, ref: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    """Fourier domain adaptation: swap the low-frequency amplitude of x with that of ref (both SIZE x SIZE)."""
    fx = np.fft.rfft2(x)
    fr = np.fft.rfft2(ref)
    amp_x, pha = np.abs(fx), np.angle(fx)
    amp_r = np.abs(fr)
    b = max(1, int(round(beta * SIZE)))
    sl_rows = np.r_[0:b, SIZE - b:SIZE]
    amp_x[sl_rows[:, None], np.arange(b)[None, :]] = (
        (1 - alpha) * amp_x[sl_rows[:, None], np.arange(b)[None, :]] + alpha * amp_r[sl_rows[:, None], np.arange(b)[None, :]])
    out = np.fft.irfft2(amp_x * np.exp(1j * pha), s=x.shape)
    return out.astype(np.float32)


def _hist_match(x: np.ndarray, valid: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match the gray-level distribution of x (valid pixels, [0,1]) to the non-zero pixels of ref (uint8)."""
    rv = ref[ref > 3]
    xv = x[valid > 0]
    if rv.size < 500 or xv.size < 500:
        return x
    q = np.linspace(0, 1, 65)
    xq = np.quantile(xv, q)
    rq = np.quantile(rv.astype(np.float32) / 255.0, q)
    xq = np.maximum.accumulate(xq + np.arange(65) * 1e-6)
    return np.interp(x, xq, rq).astype(np.float32)


def intensity_aug(x8: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x8.astype(np.float32) * (1.0 / 255.0)
    if DOMAIN_AUG and BANK_IMGS is not None and len(BANK_IMGS):
        r = rng.random()
        if r < P_FDA:
            ref = BANK_IMGS[int(rng.integers(len(BANK_IMGS)))].astype(np.float32) / 255.0
            x = np.clip(_fda(x, ref, rng.uniform(0.005, 0.03), rng.uniform(0.3, 1.0)), 0, 1)
        elif r < P_FDA + P_HISTM:
            x = _hist_match(x, valid, BANK_IMGS[int(rng.integers(len(BANK_IMGS)))])
    if DOMAIN_AUG and rng.random() < 0.25:  # random monotone tone curve (scanner LUT)
        knots = np.sort(np.clip(np.linspace(0, 1, 6) + rng.uniform(-0.08, 0.08, 6), 0, 1))
        knots[0], knots[-1] = rng.uniform(0, 0.05), rng.uniform(0.9, 1.0)
        x = np.interp(x, np.linspace(0, 1, 6), np.maximum.accumulate(knots)).astype(np.float32)
    if DOMAIN_AUG and rng.random() < 0.15:  # log / exp compression (dynamic range)
        a = rng.uniform(2.0, 30.0)
        x = (np.log1p(a * x) / math.log1p(a)) if rng.random() < 0.5 else (np.expm1(x * math.log1p(a)) / a)
        x = x.astype(np.float32)
    if rng.random() < 0.8:
        x = np.power(np.clip(x, 0, 1), rng.uniform(0.65, 1.5)).astype(np.float32)
    if rng.random() < 0.8:
        x = x * rng.uniform(0.8, 1.25) + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.4:  # TGC-like vertical gain profile
        yy = np.linspace(-0.5, 0.5, SIZE, dtype=np.float32)
        g = 1.0 + rng.uniform(-0.5, 0.5) * yy + rng.uniform(-0.6, 0.6) * (yy * yy - 1.0 / 12)
        x = x * g[:, None]
    if DOMAIN_AUG and rng.random() < 0.15:  # acoustic shadow stripes (poor probe contact)
        for _ in range(int(rng.integers(1, 3))):
            cx, wd = rng.uniform(0, SIZE), rng.uniform(10, 70)
            xs = np.arange(SIZE, dtype=np.float32)
            prof = np.clip(1 - np.abs(xs - cx) / wd, 0, 1) ** 0.7
            depth0 = rng.uniform(0, 0.6) * SIZE
            vert = np.clip((np.arange(SIZE, dtype=np.float32) - depth0) / 40.0, 0, 1)
            x = x * (1 - rng.uniform(0.3, 0.8) * vert[:, None] * prof[None, :])
    if rng.random() < 0.3:  # correlated multiplicative speckle
        nz = rng.standard_normal((SIZE // 2, SIZE // 2), dtype=np.float32)
        nz = cv2.resize(nz, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = x * (1.0 + rng.uniform(0.05, 0.25) * nz)
    if rng.random() < 0.25:
        x = x + rng.uniform(0.01, 0.04) * rng.standard_normal((SIZE, SIZE), dtype=np.float32)
    if rng.random() < 0.25:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.4))
    if DOMAIN_AUG and rng.random() < 0.15:  # lateral (beam-width) blur
        x = cv2.GaussianBlur(x, (0, 0), sigmaX=rng.uniform(0.8, 2.5), sigmaY=0.3)
    if rng.random() < 0.15:
        f = rng.uniform(0.45, 0.8)
        small = cv2.resize(x, (int(SIZE * f), int(SIZE * f)), interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    x8 = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if DOMAIN_AUG and rng.random() < 0.1:
        clahe = cv2.createCLAHE(clipLimit=float(rng.uniform(1.5, 3.0)), tileGridSize=(8, 8))
        x8 = clahe.apply(x8)
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", x8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 91))])
        if ok:
            x8 = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    x8[valid == 0] = 0
    return x8


# ----------------------------------------------------------------------------- targets
def _nearest_segment_targets(segs_out: np.ndarray, radius: float, width: int = 1):
    """Render segments (K,4 canvas coords): 1-px id map; orientation of the nearest segment within radius."""
    from scipy import ndimage

    idm = np.zeros((SIZE, SIZE), np.uint16)
    k = len(segs_out)
    ctab = np.zeros(k + 1, np.float32)
    stab = np.zeros(k + 1, np.float32)
    okt = np.zeros(k + 1, bool)
    for i, (x1, y1, x2, y2) in enumerate(segs_out):
        if not (np.isfinite([x1, y1, x2, y2]).all()) or max(abs(x1), abs(y1), abs(x2), abs(y2)) > 50 * SIZE:
            continue
        p1 = (int(round(x1 * 16)), int(round(y1 * 16)))
        p2 = (int(round(x2 * 16)), int(round(y2 * 16)))
        cv2.line(idm, p1, p2, i + 1, width, cv2.LINE_8, 4)
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 >= 9.0:
            ctab[i + 1] = (dx * dx - dy * dy) / l2
            stab[i + 1] = 2.0 * dx * dy / l2
            okt[i + 1] = True
    if not (idm > 0).any():
        z = np.zeros((SIZE, SIZE), np.float32)
        return idm, np.zeros((SIZE, SIZE), bool), z, z
    dist, ind = ndimage.distance_transform_edt(idm == 0, return_indices=True)
    nid = idm[ind[0], ind[1]]
    om = (dist <= radius) & okt[nid]
    return idm, om, ctab[nid], stab[nid]


def render_targets(segs_out: np.ndarray, valid: np.ndarray) -> dict:
    """GT fragments (canvas coords) -> seg / skel / orientation targets (v1 definition)."""
    seg = np.zeros((SIZE, SIZE), np.uint8)
    for x1, y1, x2, y2 in segs_out:
        if not (np.isfinite([x1, y1, x2, y2]).all()) or max(abs(x1), abs(y1), abs(x2), abs(y2)) > 50 * SIZE:
            continue
        cv2.line(seg, (int(round(x1 * 16)), int(round(y1 * 16))), (int(round(x2 * 16)), int(round(y2 * 16))), 1,
                 SEG_WIDTH, cv2.LINE_8, 4)
    seg &= valid
    idm, om, c, s = _nearest_segment_targets(segs_out, ORI_RADIUS)
    skel = ((idm > 0) & (valid > 0)).astype(np.uint8)
    om = om & (valid > 0)
    return {"seg": seg, "skel": skel, "ow": om.astype(np.float16), "cos2": np.where(om, c, 0).astype(np.float16),
            "sin2": np.where(om, s, 0).astype(np.float16)}


def rotate_doubled(c: np.ndarray, s: np.ndarray, a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a doubled-angle field through the linear part a2 (2x2) of an affine transform (exact for any A)."""
    th = 0.5 * np.arctan2(s, c)
    dx, dy = np.cos(th), np.sin(th)
    ndx = a2[0, 0] * dx + a2[0, 1] * dy
    ndy = a2[1, 0] * dx + a2[1, 1] * dy
    th2 = np.arctan2(ndy, ndx)
    return np.cos(2 * th2).astype(np.float32), np.sin(2 * th2).astype(np.float32)


def pseudo_targets(pst: np.ndarray, m: np.ndarray, valid: np.ndarray) -> dict:
    """Teacher maps (h,w,4 uint8: seg, cos2, sin2, gate) in sub-crop coords -> canvas pseudo-targets."""
    wm = warp_image(pst.astype(np.float32), m, border=(0.0, 127.5, 127.5, 0.0))  # (S,S,4); border = zero vector
    seg_t = wm[..., 0] / 255.0
    c = wm[..., 1] / 127.5 - 1.0
    s = wm[..., 2] / 127.5 - 1.0
    gate = wm[..., 3] > 127.5
    mag = np.hypot(c, s)
    c2, s2 = rotate_doubled(c, s, m[:2, :2])
    near = cv2.dilate((seg_t >= PS_NEAR).astype(np.uint8), disk(int(ORI_RADIUS))) > 0
    om = near & gate & (mag > 0.5) & (valid > 0)
    pos = seg_t >= PS_POS
    segm = pos | (seg_t <= PS_NEG) | (valid == 0)
    return {"seg": (pos & (valid > 0)).astype(np.uint8), "segm": segm.astype(np.uint8),
            "ow": (om * W_PS_ORI).astype(np.float16), "cos2": np.where(om, c2, 0).astype(np.float16),
            "sin2": np.where(om, s2, 0).astype(np.float16)}


def _line_canvas(line, off, m):
    """apo line (cx, cy, dx, dy) in crop coords minus off -> canvas point + direction."""
    cx, cy, dx, dy = line
    p = m @ np.array([cx - off[0], cy - off[1], 1.0])
    d = m[:2, :2] @ np.array([dx, dy])
    return p[:2], d


def _y_on(p, d, x):
    return p[1] + (x - p[0]) * d[1] / (d[0] if abs(d[0]) > 1e-6 else 1e-6)


def track_targets(tr: dict, off: tuple[float, float], m: np.ndarray, valid: np.ndarray) -> dict:
    """VL expert lines (crop coords) -> canvas targets: orientation near the tracks inside the muscle compartment,
    1-px track skeleton, deep strip mask and the deep-line angle."""
    segs = []
    for pts in tr["fas"]:
        p = np.asarray(pts, np.float64) - np.array(off)
        q = (m[:2, :2] @ p.T + m[:2, 2:3]).T
        for a, b in zip(q[:-1], q[1:]):
            segs.append([a[0], a[1], b[0], b[1]])
    segs = np.asarray(segs, np.float32).reshape(-1, 4)
    idm, om, c, s = _nearest_segment_targets(segs, VL_ORI_RADIUS)
    ps, ds = _line_canvas(tr["sup"], off, m)
    pd_, dd = _line_canvas(tr["deep"], off, m)
    xs = np.arange(SIZE, dtype=np.float32)
    ysup = _y_on(ps, ds, xs)
    ydeep = _y_on(pd_, dd, xs)
    yy = np.arange(SIZE, dtype=np.float32)[:, None]
    thick = np.maximum(1.0, ydeep - ysup)[None, :]
    depth = (ydeep[None, :] - yy) / thick
    comp = (yy > ysup[None, :] + 3) & (yy < ydeep[None, :] - 3) & (valid > 0)
    om = om & comp
    skel = (idm > 0) & comp
    strip = comp & (depth >= VL_STRIP[0]) & (depth <= VL_STRIP[1])
    return {"vow": om, "vcos2": np.where(om, c, 0).astype(np.float16), "vsin2": np.where(om, s, 0).astype(np.float16),
            "vskel": skel.astype(np.uint8), "vstrip": strip.astype(np.uint8),
            "deep_ang": float(math.atan2(dd[1], dd[0]))}


def make_sample(crop: np.ndarray, rng: np.random.Generator | None, segs: np.ndarray | None = None,
                pseudo: np.ndarray | None = None, tracks: dict | None = None, track_train: bool = False) -> dict:
    """One training / validation sample on the SIZE canvas. GT fragments (segs), teacher maps (pseudo) and VL
    expert tracks are transformed with the same affine as the image (angles recomputed exactly)."""
    h, w = crop.shape
    x0, y0, ww, hh = 0, 0, w, h
    if rng is not None:
        r = rng.random()
        if r < P_ZOOM:  # magnified sub-window (both dims): narrow-FOV / high-px-per-cm scanners
            f = rng.uniform(ZOOM_MIN, 1.0)
            ww, hh = max(32, int(round(f * w))), max(32, int(round(f * h)))
        elif r < P_ZOOM + P_NARROW:  # narrow field of view, full depth (e.g. Philips Lumify)
            ww = max(32, int(round(rng.uniform(0.3, 1.0) * w)))
            if rng.random() < 0.3:
                hh = max(32, int(round(rng.uniform(0.65, 1.0) * h)))
        elif rng.random() < 0.15:
            hh = max(32, int(round(rng.uniform(0.65, 1.0) * h)))
        ww, hh = min(ww, w), min(hh, h)
        x0 = int(rng.integers(0, w - ww + 1))
        y0 = int(rng.integers(0, h - hh + 1))
    sub = crop[y0:y0 + hh, x0:x0 + ww]
    m = letterbox_matrix(ww, hh)
    if rng is not None:
        m = aug_matrix(rng) @ m
    img = warp_image(sub, m)
    valid = _poly_mask(m, ww, hh)
    z8 = np.zeros((SIZE, SIZE), np.uint8)
    z16 = np.zeros((SIZE, SIZE), np.float16)
    out = {"seg": z8, "segm": np.ones((SIZE, SIZE), np.uint8), "skel": z8, "ow": z16, "cos2": z16, "sin2": z16}
    is_gt = segs is not None
    if is_gt:
        if len(segs):
            s = segs - np.array([x0, y0, x0, y0], np.float32)
            p1 = (m[:2, :2] @ s[:, :2].T + m[:2, 2:3]).T
            p2 = (m[:2, :2] @ s[:, 2:].T + m[:2, 2:3]).T
            segs_out = np.concatenate([p1, p2], 1)
        else:
            segs_out = np.zeros((0, 4))
        out.update(render_targets(segs_out, valid))
    elif pseudo is not None:
        out.update(pseudo_targets(pseudo[y0:y0 + hh, x0:x0 + ww], m, valid))
    else:
        out["segm"] = z8  # no seg supervision at all
    if tracks is not None:
        tt = track_targets(tracks, (x0, y0), m, valid)
        if track_train:
            vo = tt["vow"]
            out["ow"] = np.where(vo, np.float16(W_VL_ORI), out["ow"]).astype(np.float16)
            out["cos2"] = np.where(vo, tt["vcos2"], out["cos2"]).astype(np.float16)
            out["sin2"] = np.where(vo, tt["vsin2"], out["sin2"]).astype(np.float16)
        else:
            out.update({k: tt[k] for k in ("vcos2", "vsin2", "vskel", "vstrip")})
            out["deep_ang"] = np.float32(tt["deep_ang"])
    if rng is not None:
        img = intensity_aug(img, valid, rng)
    out["img"] = img
    out["valid"] = valid
    out["wseg"] = np.float32(1.0 if is_gt else (W_PS_SEG if pseudo is not None else 0.0))
    out["isgt"] = np.float32(1.0 if is_gt else 0.0)
    return out


def make_input(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = crop.shape
    m = letterbox_matrix(w, h)
    return warp_image(crop, m), m


def unletterbox(maps: np.ndarray, m: np.ndarray, w: int, h: int) -> np.ndarray:
    """maps (SIZE,SIZE,C) float32 in canvas -> (h,w,C) in crop coords."""
    out = cv2.warpAffine(maps, m[:2].astype(np.float64), (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(h, w, -1)


def encode_channels(seg: np.ndarray, cos2: np.ndarray, sin2: np.ndarray) -> dict:
    return {
        "seg": np.clip(np.round(seg * 255.0), 0, 255).astype(np.uint8),
        "cos2": np.clip(np.round((cos2 + 1.0) * 127.5), 0, 255).astype(np.uint8),
        "sin2": np.clip(np.round((sin2 + 1.0) * 127.5), 0, 255).astype(np.uint8),
    }


def view_boxes(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Full viewport + (MULTISCALE) a 2x2 grid of overlapping zoom tiles (TILE_FRAC of each side)."""
    boxes = [(0, 0, w, h)]
    if MULTISCALE:
        tw, th = max(32, int(round(TILE_FRAC * w))), max(32, int(round(TILE_FRAC * h)))
        for yy in (0, h - th):
            for xx in (0, w - tw):
                boxes.append((xx, yy, xx + tw, yy + th))
    return boxes


def tile_weight(tw: int, th: int) -> np.ndarray:
    """Feathered tile weight (h,w,1): 0 at the tile border rising to 1 over 15 % of the shorter side."""
    r = max(2.0, 0.15 * min(tw, th))
    wx = np.clip((np.minimum(np.arange(tw), tw - 1 - np.arange(tw)) + 0.5) / r, 0, 1)
    wy = np.clip((np.minimum(np.arange(th), th - 1 - np.arange(th)) + 0.5) / r, 0, 1)
    return (wy[:, None] * wx[None, :]).astype(np.float32)[..., None]


def stitch_views(maps: np.ndarray, mats: list, boxes: list, w: int, h: int) -> tuple[np.ndarray, np.ndarray | None]:
    """maps (V,S,S,3) canvas outputs per view -> (scale-1 map, multi-scale map) in crop coords (h,w,3)."""
    full = unletterbox(maps[0], mats[0], w, h)
    if len(boxes) == 1:
        return full, None
    num = full.copy()
    den = np.ones((h, w, 1), np.float32)
    for v in range(1, len(boxes)):
        a, b, c, d = boxes[v]
        back = unletterbox(maps[v], mats[v], c - a, d - b)
        wt = tile_weight(c - a, d - b)
        num[b:d, a:c] += back * wt
        den[b:d, a:c] += wt
    return full, num / den


# ----------------------------------------------------------------------------- npz writer
class NpzZip:
    """Incremental np.savez_compressed-compatible writer (np.load reads it lazily)."""

    def __init__(self, path: Path, level: int = 6) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.zf = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level, allowZip64=True)
        self.keys: set[str] = set()

    def add(self, key: str, arr: np.ndarray) -> None:
        if key in self.keys:
            raise KeyError(f"duplicate key {key} in {self.path}")
        self.keys.add(key)
        with self.zf.open(key + ".npy", "w", force_zip64=True) as fh:
            np.lib.format.write_array(fh, np.ascontiguousarray(arr), allow_pickle=False)

    def close(self) -> None:
        self.zf.close()


def merge_npz(parts: list[Path], dest: Path) -> int:
    w = NpzZip(dest)
    n = 0
    for part in parts:
        if not part.exists():
            continue
        with zipfile.ZipFile(part) as zin:
            for name in zin.namelist():
                key = name[:-4]
                if key in w.keys:
                    continue
                w.keys.add(key)
                with zin.open(name) as src, w.zf.open(name, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                n += 1
    w.close()
    return n


# ----------------------------------------------------------------------------- torch parts
def ensure_smp():
    try:
        import segmentation_models_pytorch as smp  # noqa: F401

        if smp.__version__ == SMP_VERSION:
            return smp
    except Exception:  # noqa: BLE001
        pass
    log(f"pip install segmentation-models-pytorch=={SMP_VERSION}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", f"segmentation-models-pytorch=={SMP_VERSION}"])
    import importlib

    for name in list(sys.modules):
        if name.startswith("segmentation_models_pytorch"):
            del sys.modules[name]
    return importlib.import_module("segmentation_models_pytorch")


def build_model(pretrained: bool):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet" if pretrained else None, in_channels=3,
                    classes=3, decoder_channels=(256, 128, 64, 32, 16))


def amp(torch, dev):
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def to_input(img_u8, torch):
    x = img_u8.float().unsqueeze(1) * (1.0 / 255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x.expand(-1, 3, -1, -1) - mean) / std).contiguous(memory_format=torch.channels_last)


def forward_maps(model, x, torch, tta: bool):
    """prob, cos2, sin2 (B,H,W) float32; hflip TTA flips the sin2 sign."""
    b = x.shape[0]
    xin = torch.cat([x, torch.flip(x, dims=[3])], 0) if tta else x
    with amp(torch, x.device):
        out = model(xin).float()
    p = torch.sigmoid(out[:b, 0])
    c = torch.tanh(out[:b, 1])
    s = torch.tanh(out[:b, 2])
    if tta:
        of = torch.flip(out[b:], dims=[3])
        p = 0.5 * (p + torch.sigmoid(of[:, 0]))
        c = 0.5 * (c + torch.tanh(of[:, 1]))
        s = 0.5 * (s - torch.tanh(of[:, 2]))
    return p, c, s


def state_is_finite(sd: dict, torch) -> list[str]:
    return [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all())]


def load_state(model, sd: dict, torch) -> list[str]:
    """Load a (possibly fp16) state dict as fp32. Non-finite BatchNorm running stats are reset (they are
    re-estimated by training or by recalibrate_bn). Returns the list of non-finite tensors found."""
    bad = state_is_finite(sd, torch)
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
    if bad:
        mods = dict(model.named_modules())
        for k in bad:
            mod = mods.get(k.rsplit(".", 1)[0])
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.reset_running_stats()
    return bad


def recalibrate_bn(model, items: list[dict], torch, dev, names: set[str] | None = None, n_max: int = N_CAL) -> int:
    """Re-estimate the running stats (cumulative average over un-augmented training crops) of the BatchNorm layers
    in `names` (module names; default: all BatchNorm layers)."""
    bns = [mm for n_, mm in model.named_modules() if isinstance(mm, torch.nn.modules.batchnorm._BatchNorm)
           and (names is None or n_ in names)]
    if not bns:
        return 0
    for b in bns:
        b.reset_running_stats()
        b.momentum = None
    model.eval()
    for b in bns:
        b.train()
    sel = np.random.default_rng(0).permutation(len(items))[:n_max]
    with torch.no_grad():
        for i in range(0, len(sel), BATCH):
            xs = [make_input(items[int(j)]["crop"])[0] for j in sel[i:i + BATCH]]
            x = to_input(torch.from_numpy(np.stack(xs)).to(dev), torch)
            with amp(torch, dev):
                model(x)
    for b in bns:
        b.momentum = 0.1
    model.eval()
    return len(bns)


class MixSet:
    """GT fragment items + pseudo-labelled (and optionally VL-track supervised) unlabeled items."""

    def __init__(self, gt: list[dict], pl: list[dict], n_pl: int, vl_sup_folds: set[int] | None) -> None:
        self.gt, self.pl, self.n_pl = gt, pl, n_pl if pl else 0
        self.vl_sup_folds = vl_sup_folds or set()
        self.calls = 0  # per worker-process counter -> fresh augmentation every epoch (persistent workers)

    def __len__(self) -> int:
        return len(self.gt) + self.n_pl

    def __getitem__(self, idx: int):
        import torch

        self.calls += 1
        rng = np.random.default_rng([torch.initial_seed() % (2**32), idx, self.calls, os.getpid()])
        if idx < len(self.gt):
            it = self.gt[idx]
            s = make_sample(it["crop"], rng, segs=it["segs"])
        else:
            it = self.pl[int(rng.integers(len(self.pl)))]
            use_tr = it.get("tracks") is not None and it.get("vfold", -1) in self.vl_sup_folds
            s = make_sample(it["crop"], rng, pseudo=it.get("pseudo"), tracks=it["tracks"] if use_tr else None,
                            track_train=True)
        keys = ("img", "valid", "seg", "segm", "skel", "ow", "cos2", "sin2", "wseg", "isgt")
        return {k: torch.from_numpy(np.ascontiguousarray(s[k])) if isinstance(s[k], np.ndarray) and s[k].ndim
                else torch.tensor(float(s[k])) for k in keys}


def _worker_init(_):
    cv2.setNumThreads(1)
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def build_cache(samples: list[dict], keys: tuple[str, ...], torch) -> list[dict]:
    cache = []
    for i in range(0, len(samples), BATCH):
        chunk = samples[i:i + BATCH]
        cache.append({k: torch.from_numpy(np.stack([np.asarray(s[k]) for s in chunk])) for k in keys})
    return cache


def evaluate(model, cache, torch, dev, tta: bool = False) -> dict:
    """GT validation fold (v1 metrics): Dice@thr, mean angle error on GT fragment pixels, median-angle error."""
    model.eval()
    dice_sum = {t: 0.0 for t in SEG_THRESHOLDS}
    n_img, ang_img_sum, ang_img_n, ang_pix_sum, ang_pix_n, med_sum, med_n = 0, 0.0, 0, 0.0, 0, 0.0, 0
    with torch.no_grad():
        for b in cache:
            x = to_input(b["img"].to(dev), torch)
            p, c, s = forward_maps(model, x, torch, tta)
            seg_t = b["seg"].to(dev) > 0
            skel = b["skel"].to(dev) > 0
            om = b["ow"].to(dev) > 0
            valid = b["valid"].to(dev) > 0
            ct = b["cos2"].to(dev).float()
            st = b["sin2"].to(dev).float()
            for t in SEG_THRESHOLDS:
                pred = p > t
                inter = (pred & seg_t).sum((1, 2)).float()
                den = pred.sum((1, 2)).float() + seg_t.sum((1, 2)).float()
                dice_sum[t] += float(torch.where(den > 0, 2 * inter / den.clamp_min(1), torch.ones_like(den)).sum())
            th_p = 0.5 * torch.atan2(s, c)
            th_t = 0.5 * torch.atan2(st, ct)
            d = torch.remainder(th_p - th_t, math.pi)
            d = torch.minimum(d, math.pi - d) * (180.0 / math.pi)
            m = skel & om
            for i in range(x.shape[0]):
                n_img += 1
                mi = m[i]
                cnt = int(mi.sum())
                if cnt < 10:
                    continue
                e = d[i][mi]
                ang_img_sum += float(e.mean())
                ang_img_n += 1
                ang_pix_sum += float(e.sum())
                ang_pix_n += cnt
                gt_med = float(torch.median(th_t[i][mi]))
                pr = (p[i] > 0.5) & valid[i]
                pm = float(torch.median(th_p[i][pr])) if int(pr.sum()) >= 50 else float(torch.median(th_p[i][mi]))
                dm = abs(pm - gt_med) % math.pi
                med_sum += min(dm, math.pi - dm) * 180.0 / math.pi
                med_n += 1
    dices = {t: dice_sum[t] / max(1, n_img) for t in SEG_THRESHOLDS}
    best_t = max(dices, key=dices.get)
    return {
        "dice50": dices[0.5], "dice_best": dices[best_t], "thr_best": best_t,
        "ang_mae": ang_img_sum / max(1, ang_img_n), "ang_mae_pix": ang_pix_sum / max(1, ang_pix_n),
        "med_err": med_sum / max(1, med_n), "n_val": n_img, "n_ang": ang_img_n,
    }


def evaluate_vl(model, cache, torch, dev) -> dict:
    """Out-of-domain check on the VL expert lines: mean field angle error along the drawn tracks (all / deep strip)
    and an image-level deep-strip PA proxy (|v|*seg weighted mean angle vs the drawn deep line) vs the expert PA."""
    if not cache:
        return {}
    model.eval()
    tr, st, pa = [], [], []
    with torch.no_grad():
        for b in cache:
            x = to_input(b["img"].to(dev), torch)
            p, c, s = forward_maps(model, x, torch, False)
            th_p = 0.5 * torch.atan2(s, c)
            th_t = 0.5 * torch.atan2(b["vsin2"].to(dev).float(), b["vcos2"].to(dev).float())
            d = torch.remainder(th_p - th_t, math.pi)
            d = torch.minimum(d, math.pi - d) * (180.0 / math.pi)
            sk = b["vskel"].to(dev) > 0
            strip = b["vstrip"].to(dev) > 0
            w = torch.hypot(c, s) * p
            for i in range(x.shape[0]):
                if int(sk[i].sum()) >= 10:
                    tr.append(float(d[i][sk[i]].mean()))
                ss = sk[i] & strip[i]
                if int(ss.sum()) >= 5:
                    st.append(float(d[i][ss].mean()))
                sm = strip[i]
                if int(sm.sum()) >= 30 and float(w[i][sm].sum()) > 1e-3:
                    wc = float((c[i][sm] * w[i][sm]).sum())
                    ws = float((s[i][sm] * w[i][sm]).sum())
                    thf = 0.5 * math.atan2(ws, wc)
                    rel = math.degrees(thf - float(b["deep_ang"][i]))
                    pa_f = abs((rel + 90.0) % 180.0 - 90.0)
                    pa.append(pa_f - float(b["pa_deg"][i]))
    return {"vl_track_err": float(np.mean(tr)) if tr else float("nan"),
            "vl_strip_err": float(np.mean(st)) if st else float("nan"),
            "vl_pa_mae": float(np.mean(np.abs(pa))) if pa else float("nan"),
            "vl_pa_bias": float(np.mean(pa)) if pa else float("nan"), "vl_n": len(pa)}


def _better(m: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if m["ang_mae"] < best["ang_mae"] - 0.02:
        return True
    return abs(m["ang_mae"] - best["ang_mae"]) <= 0.02 and m["dice50"] > best["dice50"]


def predict_crop(models, crop: np.ndarray, torch, dev) -> tuple[np.ndarray, np.ndarray | None]:
    """All views of one viewport crop in one batch (x hflip TTA), mean over models -> (scale-1, multi-scale)
    float maps (h,w,3) = prob, cos2, sin2 in crop coords."""
    h, w = crop.shape
    boxes = view_boxes(w, h)
    inps, mats = [], []
    for a, b, c, d in boxes:
        inp, m = make_input(np.ascontiguousarray(crop[b:d, a:c]))
        inps.append(inp)
        mats.append(m)
    x = to_input(torch.from_numpy(np.stack(inps)).to(dev), torch)
    acc = None
    with torch.no_grad():
        for model in models:
            p, c, s = forward_maps(model, x, torch, True)
            cur = torch.stack([p, c, s], -1)
            acc = cur if acc is None else acc + cur
    maps = (acc / len(models)).float().cpu().numpy()
    return stitch_views(maps, mats, boxes, w, h)


def to_full(crop_maps: np.ndarray, box, H: int, W: int) -> dict:
    x1, y1, x2, y2 = box
    full = np.zeros((H, W, 3), np.float32)
    full[y1:y2, x1:x2] = crop_maps
    return encode_channels(full[..., 0], full[..., 1], full[..., 2])


def save_viz(path: Path, gray: np.ndarray, enc: dict) -> None:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    seg = enc["seg"].astype(np.float32) / 255.0
    rgb[..., 2] = np.clip(rgb[..., 2] * (1 - seg) + 255 * seg, 0, 255).astype(np.uint8)
    c = enc["cos2"].astype(np.float32) / 127.5 - 1
    s = enc["sin2"].astype(np.float32) / 127.5 - 1
    th = 0.5 * np.arctan2(s, c)
    mag = np.hypot(c, s)
    step = max(16, gray.shape[1] // 40)
    for yy in range(step // 2, gray.shape[0], step):
        for xx in range(step // 2, gray.shape[1], step):
            if mag[yy, xx] < 0.3:
                continue
            dx, dy = math.cos(th[yy, xx]) * step * 0.45, math.sin(th[yy, xx]) * step * 0.45
            cv2.line(rgb, (int(xx - dx), int(yy - dy)), (int(xx + dx), int(yy + dy)), (0, 255, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb)


def full_gray(it: dict) -> np.ndarray:
    g = np.zeros((it["H"], it["W"]), np.uint8)
    x1, y1, x2, y2 = it["box"]
    g[y1:y2, x1:x2] = it["crop"]
    return g


# ----------------------------------------------------------------------------- GPU worker: training
def fold_training_pool(unlabeled: list[dict], fold: int, vl_supervised: bool) -> list[dict]:
    """Exclude held-out VL subjects entirely, including their pseudo-labelled copies.

    Merely masking their track loss in MixSet still lets those images enter the
    training loader (and BatchNorm); with pseudo-labels enabled it supervises them.
    """
    return [u for u in unlabeled if u.get("train_pl") and not (
        vl_supervised and u.get("ds") == "vl" and u.get("vfold", -1) in (-1, fold)
    )]


def train_fold(fold: int, gpu_state: dict, torch, dev) -> None:
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(SEED + fold)
    tr = [it for it in ITEMS if it["fold"] != fold]
    va = [it for it in ITEMS if it["fold"] == fold]
    pl = fold_training_pool(UNL, fold, VL_SUP)
    vl_val = [u for u in UNL if u.get("tracks") is not None
              and (not (VL_SUP or VL_VALIDATION_BY_FOLD) or u.get("vfold") == fold)]
    if VL_SUP:
        train_subjects = {u["tracks"]["subject"] for u in pl if u.get("tracks") is not None}
        val_subjects = {u["tracks"]["subject"] for u in vl_val}
        if train_subjects & val_subjects:
            raise RuntimeError(f"VL subject leakage in fold {fold}: {train_subjects & val_subjects}")
        (METRIC_DIR / f"fold{fold}_vl_split.json").write_text(json.dumps({
            "train_subjects": sorted(train_subjects), "val_subjects": sorted(val_subjects),
            "n_train_items": sum(u.get("ds") == "vl" for u in pl), "n_val_items": len(vl_val),
        }, indent=2))
    if SMOKE:
        tr = [tr[i] for i in rng.choice(len(tr), min(len(tr), SMOKE_TRAIN), replace=False)]
        va = [va[i] for i in rng.choice(len(va), min(len(va), SMOKE_VAL), replace=False)]
        pl = pl[:SMOKE_PL]
        vl_val = vl_val[:max(2, SMOKE_VAL // 2)]
    n_pl = int(round(len(tr) * PSEUDO_FRAC / (1 - PSEUDO_FRAC))) if (PSEUDO or VL_SUP) and pl else 0
    vl_sup_folds = {f for f in range(N_FOLDS) if f != fold} if VL_SUP else set()
    torch.manual_seed(SEED + fold)
    n_cpu = os.cpu_count() or 4
    n_workers = 0 if CPU_SMOKE else max(2, min(4, n_cpu // gpu_state["n_parallel"] + 1))
    ds = MixSet(tr, pl, n_pl, vl_sup_folds)
    kw = dict(num_workers=n_workers, worker_init_fn=_worker_init)
    if n_workers:
        kw.update(persistent_workers=True, prefetch_factor=4)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, pin_memory=dev.type == "cuda", drop_last=True, **kw)
    t_cache = time.time()
    gt_keys = ("img", "seg", "skel", "ow", "valid", "cos2", "sin2")
    cache = build_cache([make_sample(it["crop"], None, segs=it["segs"]) for it in va], gt_keys, torch)
    vl_samples = []
    for u in vl_val:
        s = make_sample(u["crop"], None, tracks=u["tracks"])
        s["pa_deg"] = np.float32(u["tracks"]["pa_deg"])
        vl_samples.append(s)
    vcache = build_cache(vl_samples, ("img", "vcos2", "vsin2", "vskel", "vstrip", "deep_ang", "pa_deg"), torch)
    log(f"fold {fold}: train={len(tr)} pseudo_pool={len(pl)} pseudo/epoch={n_pl} val={len(va)} vl_val={len(vl_val)} "
        f"loader_workers={n_workers} steps/epoch={len(dl)} caches={time.time() - t_cache:.1f}s")

    use_warm = WARM_START and fold in WARM_CKPTS
    st = None
    if use_warm:
        st = torch.load(WARM_CKPTS[fold], map_location="cpu", weights_only=False)
        bad_w = [k for k in state_is_finite(st["state_dict"], torch) if not k.endswith(("running_mean", "running_var"))]
        if bad_w:  # non-finite weights cannot be repaired: start this fold from ImageNet instead
            if REQUIRE_WARM_START:
                raise RuntimeError(f"controlled experiment has non-finite warm weights in fold {fold}")
            log(f"fold {fold}: WARNING warm checkpoint has non-finite WEIGHTS {bad_w[:5]} -> ImageNet init")
            use_warm = False
    model = build_model(pretrained=not use_warm)
    warm_info = {"warm": False}
    if use_warm:
        bad = load_state(model, st["state_dict"], torch)
        warm_info = {"warm": True, "ckpt": str(WARM_CKPTS[fold]), "nonfinite": bad, "src_epoch": st.get("epoch")}
        if bad:
            log(f"fold {fold}: WARNING warm checkpoint has non-finite BN stats {bad[:5]} (reset + recalibrated)")
    del st
    model = model.to(dev).to(memory_format=torch.channels_last)
    if warm_info.get("nonfinite"):  # fp16 overflow of a BN running_var (v1 folds 2 and 4): re-estimate only those
        names = {k.rsplit(".", 1)[0] for k in warm_info["nonfinite"]}
        n_rec = recalibrate_bn(model, tr, torch, dev, names=names)
        log(f"fold {fold}: recalibrated {n_rec} BN layers {sorted(names)} on {min(len(tr), N_CAL)} training crops")
    ema = build_model(pretrained=False).to(dev).to(memory_format=torch.channels_last)
    ema.load_state_dict(model.state_dict())
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    esd, msd = ema.state_dict(), model.state_dict()
    ema_f = [esd[k] for k in esd if esd[k].dtype.is_floating_point]
    mod_f = [msd[k] for k in esd if esd[k].dtype.is_floating_point]
    ema_i = [esd[k] for k in esd if not esd[k].dtype.is_floating_point]
    mod_i = [msd[k] for k in esd if not esd[k].dtype.is_floating_point]

    hist_path = METRIC_DIR / f"history_fold{fold}.csv"
    cols = ["epoch", "loss", "bce", "dice_l", "skel_l", "ori_l", "dice50", "dice_best", "thr_best", "ang_mae",
            "ang_mae_pix", "med_err", "vl_track_err", "vl_strip_err", "vl_pa_mae", "vl_pa_bias", "lr", "sec"]
    with open(hist_path, "w", newline="") as fh:
        csv.writer(fh).writerow(cols)

    def record(epoch, sums, met, vmet, lr_now, sec):
        with open(hist_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, *[round(float(v), 5) for v in sums], met["dice50"], met["dice_best"],
                                     met["thr_best"], met["ang_mae"], met["ang_mae_pix"], met["med_err"],
                                     vmet.get("vl_track_err"), vmet.get("vl_strip_err"), vmet.get("vl_pa_mae"),
                                     vmet.get("vl_pa_bias"), lr_now, sec])

    ckpt_path = CKPT_DIR / f"fold{fold}.pt"

    def save_best(epoch, met, vmet):
        sd = {k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
              for k, v in ema.state_dict().items()}
        torch.save({"state_dict": sd, "epoch": epoch, "metrics": met, "vl_metrics": vmet, "fold": fold,
                    "encoder": ENCODER, "size": SIZE, "classes": CHANNELS, "model": MODEL_NAME, "dtype": "float32",
                    "warm": warm_info}, ckpt_path)

    # epoch 0: the warm-start model (= WARM_MODEL fold k) -> baseline + checkpoint reproduction check
    t_e = time.time()
    met0 = evaluate(ema, cache, torch, dev)
    vmet0 = evaluate_vl(ema, vcache, torch, dev)
    record(0, np.zeros(5), met0, vmet0, 0.0, time.time() - t_e)
    ref = REF_METRICS.get(fold, {})
    repro = None
    if warm_info["warm"] and ref and not SMOKE:
        repro = abs(met0["ang_mae"] - ref["ang_mae"]) <= 0.1 and abs(met0["dice50"] - ref["dice50"]) <= 0.02
    log(f"fold {fold}: epoch 0 ({'warm ' + WARM_MODEL if warm_info['warm'] else 'imagenet'}) dice50={met0['dice50']:.4f} "
        f"ang_mae={met0['ang_mae']:.3f} med_err={met0['med_err']:.3f} | VL {json.dumps(vmet0)} | ref {ref} "
        f"reproduces={repro}")
    if REQUIRE_WARM_START and repro is not True:
        raise RuntimeError(f"controlled experiment failed epoch-0 reference reproduction in fold {fold}")
    best, best_epoch, best_v = (met0, 0, vmet0) if warm_info["warm"] else (None, -1, {})
    if warm_info["warm"]:
        save_best(0, met0, vmet0)

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p_) for p_ in enc_params}
    dec_params = [p_ for p_ in model.parameters() if id(p_) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENC}, {"params": dec_params, "lr": LR_DEC}],
                            weight_decay=WEIGHT_DECAY)
    steps = len(dl) if MAX_STEPS is None else min(len(dl), MAX_STEPS)
    warm = max(1, int(steps * WARMUP_EPOCHS))
    n_epochs = gpu_state.get("n_epochs", MAX_EPOCHS)
    if "ep_time" in gpu_state:
        folds_left = max(1, gpu_state["plan_folds"] - gpu_state["done"])
        n_fit = int((T0 + TRAIN_BUDGET_S - time.time()) / folds_left / (gpu_state["ep_time"] * 1.03))
        n_epochs = int(max(MIN_EPOCHS, min(n_epochs, n_fit)))
        log(f"fold {fold}: n_epochs={n_epochs} (budget fit {n_fit})")
    sched_state = {"total": n_epochs * steps}

    def lr_lambda(step: int) -> float:
        if step < warm:
            return 0.05 + 0.95 * step / warm
        total = max(sched_state["total"], warm + 1)
        prog = min(1.0, (step - warm) / (total - warm))
        return LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    pos_w = torch.tensor([POS_WEIGHT], device=dev)
    OOM_ERRORS = tuple(e for e in (getattr(torch.cuda, "OutOfMemoryError", None), getattr(torch, "OutOfMemoryError", None))
                       if isinstance(e, type))
    n_micro = 1

    def batch_loss(bb):
        """(loss, bce, dice, skel, ori) of one (micro-)batch; GT / pseudo / VL samples weighted per image and pixel."""
        x = to_input(bb["img"], torch)
        seg_t, segm, skel = bb["seg"].float(), bb["segm"].float(), bb["skel"].float()
        ow, ct, st_ = bb["ow"].float(), bb["cos2"].float(), bb["sin2"].float()
        wseg, isgt = bb["wseg"].float(), bb["isgt"].float()
        with amp(torch, dev):
            out = model(x)
        out = out.float()
        logit = out[:, 0]
        bce_map = torch.nn.functional.binary_cross_entropy_with_logits(logit, seg_t, pos_weight=pos_w,
                                                                       reduction="none") * segm
        bce_i = bce_map.sum((1, 2)) / segm.sum((1, 2)).clamp_min(1.0)
        wsum = wseg.sum().clamp_min(1e-6)
        bce = (bce_i * wseg).sum() / wsum
        p = torch.sigmoid(logit)
        pm, tm = p * segm, seg_t * segm
        dice_i = 1 - (2 * (pm * tm).sum((1, 2)) + 1) / (pm.sum((1, 2)) + tm.sum((1, 2)) + 1)
        dice_l = (dice_i * wseg).sum() / wsum
        sk_n = skel.sum((1, 2))
        has = (sk_n > 0) & (isgt > 0)
        skel_l = (1 - (p * skel).sum((1, 2))[has] / sk_n[has]).mean() if bool(has.any()) else logit.sum() * 0
        cp, sp = torch.tanh(out[:, 1]), torch.tanh(out[:, 2])
        ori_l = (((cp - ct).abs() + (sp - st_).abs()) * ow).sum() / (2 * (ow > 0).float().sum().clamp_min(1))
        loss = W_BCE * bce + W_DICE * dice_l + W_SKEL * skel_l + W_ORI * ori_l
        return loss, bce, dice_l, skel_l, ori_l

    gstep = 0
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        model.train()
        t_ep = time.time()
        sums = np.zeros(5)
        nb = 0
        t_wait = 0.0
        t_prev = time.time()
        for step, b in enumerate(dl, 1):
            if MAX_STEPS is not None and step > MAX_STEPS:
                break
            t_wait += time.time() - t_prev
            bt = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            while True:  # CUDA OOM -> retry this batch as 2 (then 4) accumulated micro-batches
                try:
                    opt.zero_grad(set_to_none=True)
                    parts = np.zeros(5)
                    nbat = int(bt["img"].shape[0])
                    cuts = np.linspace(0, nbat, n_micro + 1).astype(int)
                    for i0, i1 in zip(cuts[:-1], cuts[1:]):
                        if i1 <= i0:
                            continue
                        terms = batch_loss({k: v[i0:i1] for k, v in bt.items()})
                        frac = (i1 - i0) / nbat
                        scaler.scale(terms[0] * frac).backward()
                        parts += frac * np.array([float(v.detach()) for v in terms])
                        del terms
                    break
                except OOM_ERRORS:
                    if n_micro >= 4:
                        raise
                    n_micro *= 2
                    opt.zero_grad(set_to_none=True)
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    log(f"fold {fold}: CUDA out of memory at epoch {epoch} step {step} -> {n_micro} micro-batches")
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            gstep += 1
            d = min(EMA_DECAY, (1 + gstep) / (10 + gstep))
            with torch.no_grad():
                torch._foreach_mul_(ema_f, d)
                torch._foreach_add_(ema_f, mod_f, alpha=1 - d)
                for a_, b_ in zip(ema_i, mod_i):
                    a_.copy_(b_)
            sums += parts
            nb += 1
            if step % 50 == 0 or (SMOKE and step % 5 == 0):
                el = time.time() - t_ep
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
                log(f"fold {fold} ep {epoch} step {step}/{steps} loss={sums[0] / nb:.4f} ori={sums[4] / nb:.4f} "
                    f"ms/step={1000 * el / step:.0f} wait_ms/step={1000 * t_wait / step:.0f} mem={mem:.1f}G")
            t_prev = time.time()
        t_train = time.time() - t_ep
        met = evaluate(ema, cache, torch, dev)
        vmet = evaluate_vl(ema, vcache, torch, dev)
        sec = time.time() - t_ep
        lr_now = opt.param_groups[0]["lr"]
        sums /= max(1, nb)
        record(epoch, sums, met, vmet, lr_now, sec)
        log(f"EPOCH fold={fold} ep={epoch}/{n_epochs} loss={sums[0]:.4f} bce={sums[1]:.4f} dice_l={sums[2]:.4f} "
            f"skel_l={sums[3]:.4f} ori_l={sums[4]:.4f} | val(EMA) dice50={met['dice50']:.4f} ang_mae={met['ang_mae']:.3f} "
            f"med_err={met['med_err']:.3f} | VL track={vmet.get('vl_track_err', float('nan')):.3f} "
            f"strip={vmet.get('vl_strip_err', float('nan')):.3f} pa_mae={vmet.get('vl_pa_mae', float('nan')):.3f} "
            f"bias={vmet.get('vl_pa_bias', float('nan')):+.3f} | lr={lr_now:.2e} train={t_train:.0f}s total={sec:.0f}s")
        if _better(met, best):
            best, best_epoch, best_v = met, epoch, vmet
            save_best(epoch, met, vmet)
            log(f"fold {fold}: saved best epoch {epoch} ang_mae={met['ang_mae']:.3f} dice50={met['dice50']:.4f}")
        if epoch == 1 and "n_epochs" not in gpu_state and not SMOKE:
            ep_time = sec
            left = T0 + TRAIN_BUDGET_S - time.time() + ep_time
            per_fold = left / gpu_state["plan_folds"]
            n_epochs = int(max(MIN_EPOCHS, min(MAX_EPOCHS, per_fold // (ep_time * 1.03))))
            gpu_state["n_epochs"] = n_epochs
            gpu_state["ep_time"] = ep_time
            sched_state["total"] = n_epochs * steps
            log(f"fold {fold}: epoch time {ep_time:.0f}s -> n_epochs={n_epochs} for all folds on this GPU")
        if time.time() > T0 + HARD_TRAIN_END_S:
            log(f"fold {fold}: HARD time limit reached after epoch {epoch}; stopping this fold")
            break
    del dl
    log(f"fold {fold}: best epoch {best_epoch} {json.dumps(best)} VL {json.dumps(best_v)}")

    # ---- reload the saved fp32 checkpoint from disk, check finiteness, re-validate (before any maskbank)
    vmodel = build_model(pretrained=False)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bad = load_state(vmodel, st["state_dict"], torch)
    vmodel = vmodel.to(dev).to(memory_format=torch.channels_last).eval()
    met_r = evaluate(vmodel, cache, torch, dev)
    ok = (not bad) and abs(met_r["ang_mae"] - best["ang_mae"]) <= VERIFY_TOL_ANG and \
        abs(met_r["dice50"] - best["dice50"]) <= VERIFY_TOL_DICE
    recal = False
    if not ok:
        log(f"fold {fold}: reload check FAILED (nonfinite={bad}, reloaded {met_r}) -> BN recalibration")
        recalibrate_bn(vmodel, tr, torch, dev)
        recal = True
        met_r = evaluate(vmodel, cache, torch, dev)
        ok = abs(met_r["ang_mae"] - best["ang_mae"]) <= 2 * VERIFY_TOL_ANG and \
            abs(met_r["dice50"] - best["dice50"]) <= 2 * VERIFY_TOL_DICE
        if ok:
            sd = {k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
                  for k, v in vmodel.state_dict().items()}
            st["state_dict"], st["bn_recalibrated"] = sd, True
            torch.save(st, ckpt_path)
    log(f"fold {fold}: reload check {'OK' if ok else 'FAILED'} (recal={recal}) reloaded dice50={met_r['dice50']:.4f} "
        f"ang_mae={met_r['ang_mae']:.3f} vs best {best['dice50']:.4f}/{best['ang_mae']:.3f}")
    res = {"fold": fold, "best_epoch": best_epoch, "n_epochs": n_epochs, "best": best, "best_vl": best_v,
           "epoch0": met0, "epoch0_vl": vmet0, "ref_warm": ref, "warm_reproduces": repro, "warm": warm_info,
           "reload_ok": ok, "reload_metrics": met_r, "bn_recalibrated": recal, "n_train": len(tr), "n_val": len(va),
           "n_pseudo_pool": len(pl), "n_pseudo_epoch": n_pl, "n_vl_val": len(vl_val)}
    if not ok:
        (METRIC_DIR / f"fold{fold}.json").write_text(json.dumps(res, indent=2, default=str))
        log(f"fold {fold}: checkpoint unusable -> no OOF maps, excluded from the ensemble")
        return
    met_tta = evaluate(vmodel, cache, torch, dev, tta=True)
    res["tta"] = met_tta
    log(f"fold {fold}: TTA val dice50={met_tta['dice50']:.4f} ang_mae={met_tta['ang_mae']:.3f} "
        f"med_err={met_tta['med_err']:.3f}")

    # ---- OOF maps (scale-1 and multi-scale) for the held-out fold
    names = [MODEL_NAME] + ([MS_NAME] if MULTISCALE else [])
    writers = {(nm, ch): NpzZip(PART_DIR / f"oof_fold{fold}__{nm}_{ch}.npz", level=3) for nm in names for ch in CHANNELS}
    t_oof = time.time()
    for j, it in enumerate(va):
        s1, ms = predict_crop([vmodel], it["crop"], torch, dev)
        for nm, mp_ in ((MODEL_NAME, s1), (MS_NAME, ms)):
            if mp_ is None or nm not in names:
                continue
            enc = to_full(mp_, it["box"], it["H"], it["W"])
            for ch in CHANNELS:
                writers[(nm, ch)].add(it["stem"], enc[ch])
            if j < 2:
                save_viz(VIZ_DIR / f"oof_fold{fold}_{nm}_{it['stem']}.jpg", full_gray(it), enc)
    for w_ in writers.values():
        w_.close()
    res["oof_seconds"] = time.time() - t_oof
    (METRIC_DIR / f"fold{fold}.json").write_text(json.dumps(res, indent=2, default=str))
    del model, ema, opt, cache, vcache, vmodel
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def gpu_worker(gpu: int, folds: list[int], plan_folds: int, n_parallel: int, forked: bool = True) -> int:
    global _LOG_PREFIX, _LOG_FILE
    if forked and not CPU_SMOKE:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _LOG_PREFIX = f"[gpu{gpu}]"
    _LOG_FILE = LOG_DIR / f"gpu{gpu}.log"
    log(f"worker process pid={os.getpid()} forked={forked}")
    code = 0
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(4 if CPU_SMOKE else 1)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"worker start folds={folds} device={torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}")
        state = {"plan_folds": plan_folds, "n_parallel": n_parallel, "done": 0}
        for k in folds:
            try:
                train_fold(k, state, torch, dev)
            except Exception:  # noqa: BLE001
                code = 1
                log(f"fold {k} FAILED:\n{traceback.format_exc()}")
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            state["done"] += 1
    except BaseException:  # noqa: BLE001
        code = 2
        log(f"worker FAILED:\n{traceback.format_exc()}")
    if forked:
        os._exit(code)
    _LOG_PREFIX, _LOG_FILE = "", LOG_DIR / "main.log"
    return code


# ----------------------------------------------------------------------------- GPU worker: ensemble inference
def infer_worker(gpu: int, shard: list[int], fold_ok: list[int], forked: bool = True) -> int:
    global _LOG_PREFIX, _LOG_FILE
    if forked and not CPU_SMOKE:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _LOG_PREFIX = f"[inf{gpu}]"
    _LOG_FILE = LOG_DIR / f"inf{gpu}.log"
    log(f"inference process pid={os.getpid()} forked={forked}")
    code = 0
    try:
        import torch

        torch.set_num_threads(4 if CPU_SMOKE else 1)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = {}
        for k in fold_ok:
            m = build_model(pretrained=False)
            st = torch.load(CKPT_DIR / f"fold{k}.pt", map_location="cpu", weights_only=False)
            bad = load_state(m, st["state_dict"], torch)
            if bad:
                raise RuntimeError(f"fold {k} checkpoint non-finite at inference: {bad[:5]}")
            models[k] = m.to(dev).eval().to(memory_format=torch.channels_last)
        log(f"inference models {sorted(models)} on {dev}; {len(shard)} images")
        names = [MODEL_NAME] + ([MS_NAME] if MULTISCALE else [])
        writers: dict = {}
        t = time.time()
        for n_done, ui in enumerate(shard):
            u = UNL[ui]
            if VL_SUP and u["ds"] == "vl" and "tracks" in u:
                if u.get("vfold", -1) not in models:
                    log(f"skip {u['key']}: its VL subject fold {u.get('vfold')} has no verified model (OOF only)")
                    continue
                mlist = [models[u["vfold"]]]  # subject-fold OOF: this model never saw the subject's lines
            else:
                mlist = [models[k] for k in sorted(models)]
            try:
                s1, ms = predict_crop(mlist, u["crop"], torch, dev)
            except Exception:  # noqa: BLE001
                log(f"inference failed {u['ds']}/{u['key']}:\n{traceback.format_exc()}")
                continue
            for nm, mp_ in ((MODEL_NAME, s1), (MS_NAME, ms)):
                if mp_ is None or nm not in names:
                    continue
                enc = to_full(mp_, u["box"], u["H"], u["W"])
                for ch in CHANNELS:
                    key = (u["ds"], nm, ch)
                    if key not in writers:
                        writers[key] = NpzZip(PART_DIR / f"inf{gpu}_{u['ds']}__{nm}_{ch}.npz", level=6)
                    writers[key].add(u["key"], enc[ch])
                if u.get("viz"):
                    save_viz(VIZ_DIR / f"{u['ds']}_{nm}_{u['key']}.jpg", full_gray(u), enc)
            if n_done % 100 == 0:
                log(f"inference {n_done}/{len(shard)} ({time.time() - t:.0f}s)")
        for w_ in writers.values():
            w_.close()
        log(f"inference done {len(shard)} images in {time.time() - t:.0f}s")
    except BaseException:  # noqa: BLE001
        code = 2
        log(f"inference worker FAILED:\n{traceback.format_exc()}")
    if forked:
        os._exit(code)
    _LOG_PREFIX, _LOG_FILE = "", LOG_DIR / "main.log"
    return code


# ----------------------------------------------------------------------------- isolated torch helpers
CKPT_CHECK_CODE = r"""
import sys, torch
for i, path in enumerate(sys.argv[1:]):
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["state_dict"]
    bad = [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all())]
    fin = [float(v.float().abs().max()) for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()
           and bool(torch.isfinite(v).all()) and v.numel()]
    dt = sorted({str(v.dtype) for v in sd.values() if torch.is_tensor(v)})
    print(f"{path.rsplit('/', 1)[-1]}: epoch={st.get('epoch')} dtypes={dt} tensors={len(sd)} nonfinite={bad[:5]} "
          f"max|finite|={max(fin):.1f}", flush=True)
"""
PRETRAIN_CODE = r"""
import sys
import segmentation_models_pytorch as smp
m = smp.Unet(encoder_name=sys.argv[1], encoder_weights="imagenet", in_channels=3, classes=3,
             decoder_channels=(256, 128, 64, 32, 16))
print(f"pretrained {sys.argv[1]}: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M params", flush=True)
"""


def run_isolated(code: str, args: list[str]) -> list[str]:
    """Run torch code in a fresh interpreter (never in the parent before fork); returns its stdout lines."""
    r = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        log(f"isolated helper failed (code {r.returncode}): {r.stderr[-2000:]}")
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


# ----------------------------------------------------------------------------- data discovery
def find_dir(name: str) -> Path:
    for base in (INPUT_ROOT / "competitions" / COMP_SLUG, INPUT_ROOT / COMP_SLUG, INPUT_ROOT):
        if base.exists():
            hits = [p for p in base.rglob(name) if p.is_dir()]
            if hits:
                return sorted(hits, key=lambda p: len(str(p)))[0]
    raise FileNotFoundError(name)


def find_model_source(model: str) -> Path | None:
    """Root of an attached kernel output (kernel_sources) that holds <model>_config.json."""
    hits = sorted(INPUT_ROOT.rglob(f"{model}_config.json"), key=lambda p: len(str(p)))
    return hits[0].parent if hits else None


def find_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[str, str]]:
    masks = {p.stem: p for p in mask_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    pairs = [(str(p), str(masks[p.stem])) for p in image_dir.rglob("*")
             if p.suffix.lower() in IMAGE_SUFFIXES and p.stem in masks]
    return sorted(pairs, key=lambda t: Path(t[0]).stem)


def union_find_groups(n: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = root(int(a)), root(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([root(i) for i in range(n)])


def assign_folds(groups: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    order = sorted(range(len(uniq)), key=lambda i: (-counts[i], rng.random()))
    load = np.zeros(N_FOLDS)
    gfold = {}
    for i in order:
        f = int(np.argmin(load + rng.random(N_FOLDS) * 1e-3))
        gfold[uniq[i]] = f
        load[f] += counts[i]
    return np.array([gfold[g] for g in groups])


def prepare_items(pairs: list[tuple[str, str]], warm_folds: dict[str, int]) -> tuple[list[dict], list[list], dict]:
    t = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(min(4, os.cpu_count() or 2)) as pool:
        results = list(pool.imap(prep_train, pairs, chunksize=8))
    log(f"preprocessed {len(results)} pairs in {time.time() - t:.0f}s")
    stats = {"pairs": len(results), "empty_masks": 0, "exact_duplicates": 0, "same_image_other_mask": 0}
    by_img: dict[str, list[dict]] = {}
    for r in results:
        by_img.setdefault(r["ih"], []).append(r)
    items, alias_rows = [], []
    for ih, rs in by_img.items():
        rs.sort(key=lambda r: r["stem"])
        seen_masks, keep = set(), []
        for r in rs:
            if r["mh"] in seen_masks:
                stats["exact_duplicates"] += 1
                continue
            seen_masks.add(r["mh"])
            if r["empty"]:
                stats["empty_masks"] += 1
                continue
            keep.append(r)
        if not keep:
            continue
        rep = keep[0]
        if len(keep) > 1:
            stats["same_image_other_mask"] += len(keep) - 1
            bx, by = rep["box"][0], rep["box"][1]
            allsegs = [rep["segs"]]
            for r in keep[1:]:
                off = np.array([r["box"][0] - bx, r["box"][1] - by] * 2, np.float32)
                allsegs.append(r["segs"] + off)
            segs = np.concatenate(allsegs)
            _, ui = np.unique(np.round(segs, 1), axis=0, return_index=True)
            rep["segs"] = segs[np.sort(ui)]
        rep["aliases"] = [r["stem"] for r in rs]
        items.append(rep)
    items.sort(key=lambda r: r["stem"])
    thumbs = np.stack([it["thumb"] for it in items])
    corr = thumbs @ thumbs.T
    ii, jj = np.nonzero(np.triu(corr > DUP_CORR, 1))
    groups = union_find_groups(len(items), np.stack([ii, jj], 1))
    folds = assign_folds(groups, SEED)
    n_match = n_diff = n_missing = 0
    for i, it in enumerate(items):
        it["group"] = int(groups[i])
        f = int(folds[i])
        if warm_folds:
            wf = warm_folds.get(it["stem"])
            if wf is None:
                n_missing += 1
            elif wf == f:
                n_match += 1
            else:
                n_diff += 1
                f = wf  # the warm-start model's fold is authoritative (honest OOF)
        it["fold"] = f
        del it["thumb"]
        for a in it["aliases"]:
            alias_rows.append([a, it["stem"], it["group"], it["fold"]])
    _, gcounts = np.unique(groups, return_counts=True)
    fl = np.array([it["fold"] for it in items])
    stats.update(
        unique_items=len(items), n_groups=int(len(gcounts)), max_group=int(gcounts.max()),
        fold_sizes=[int((fl == f).sum()) for f in range(N_FOLDS)],
        warm_fold_match=n_match, warm_fold_diff=n_diff, warm_fold_missing=n_missing,
        segments_median=float(np.median([len(it["segs"]) for it in items])),
        layouts={k: int(v) for k, v in zip(*np.unique([it["layout"] for it in items], return_counts=True))},
        crop_mb=float(sum(it["crop"].nbytes for it in items) / 2**20),
    )
    if warm_folds and n_missing:
        if REQUIRE_WARM_START:
            raise RuntimeError(f"controlled experiment has {n_missing} items missing from the warm-start folds")
        log(f"WARNING {n_missing} items have no fold in the warm-start folds.csv (kept the recomputed fold)")
    return items, alias_rows, stats


def download(url: str, dest: Path) -> bool:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (kaggle kernel)"})
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            if zipfile.is_zipfile(dest):
                return True
            log(f"download {url}: not a zip ({dest.stat().st_size} bytes)")
        except Exception as exc:  # noqa: BLE001
            log(f"download {url} attempt {attempt + 1} failed: {exc!r}")
            time.sleep(10)
    return False


def safe_extract(zpath: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.infolist():
            p = (dest / m.filename).resolve()
            if not str(p).startswith(str(root)) or m.is_dir():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(p, "wb") as dst:
                shutil.copyfileobj(src, dst)


def gather_extra(name: str) -> list[tuple[str, Path, str]]:
    """[(image_id, path, source_filename)] for an OSF dataset; keys as in v1 (VL: raw .tiff images only)."""
    root = TMP / "ext" / name
    root.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for url, fname in EXTRA_SOURCES[name]:
        zpath = TMP / "ext" / fname
        if not download(url, zpath):
            continue
        sub = root / Path(fname).stem
        safe_extract(zpath, sub)
        zpath.unlink()
        files += [p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                  and "__MACOSX" not in p.parts and not p.name.startswith("._")]
    if name == "static":
        files = [p for p in files if p.stem.startswith("im_") and p.stem.endswith("_arch")] or files
    elif name == "dynamic":
        files = [p for p in files if "frames" in p.parts or p.stem.startswith("image_")] or files
    elif name == "vl":
        files = [p for p in files if p.suffix.lower() == ".tiff"]  # raw images; .tif copies carry the red lines
    files = sorted(files, key=lambda p: str(p.relative_to(root)))
    stem_counts: dict[str, int] = {}
    for p in files:
        stem_counts[p.stem] = stem_counts.get(p.stem, 0) + 1
    out, used = [], set()
    for p in files:
        key = p.stem
        if stem_counts[p.stem] > 1:
            key = "_".join([*p.parent.parts[-2:], p.stem])
        k2, n = key, 1
        while k2 in used:
            n += 1
            k2 = f"{key}_{n}"
        used.add(k2)
        out.append((k2, p, str(p.relative_to(root))))
    log(f"extra {name}: {len(out)} images")
    return out


class Teacher:
    """Lazy reader of the teacher maskbank (<root>/maskbank/<ds>__<model>_<ch>.npz); VL keys mapped by path tail."""

    def __init__(self, root: Path | None, model: str) -> None:
        self.dir = (root / "maskbank") if root is not None else None
        self.model = model
        self.files: dict = {}
        self.vl_map: dict[str, str] = {}
        if self.dir is not None and (self.dir / "manifest.csv").exists():
            with open(self.dir / "manifest.csv") as fh:
                for r in csv.DictReader(fh):
                    if r["dataset"] == "vl":
                        self.vl_map.setdefault(tail3(r["source_filename"]), r["image_id"])

    def get(self, ds: str, key: str, src: str):
        if self.dir is None:
            return None
        k = self.vl_map.get(tail3(src), key) if ds == "vl" else key
        out = []
        for ch in CHANNELS:
            f = self.files.get((ds, ch))
            if f is None:
                path = self.dir / f"{ds}__{self.model}_{ch}.npz"
                if not path.exists():
                    return None
                f = self.files[(ds, ch)] = np.load(path)
            if k not in f.files:
                return None
            out.append(f[k])
        return out


def decode_vl_tracks() -> dict[str, dict]:
    recs = json.loads(zlib.decompress(base64.b64decode("".join(VL_TRACKS_B64.split()))).decode())
    return {r["tail"]: r for r in recs}


def subject_folds(subjects: list[str]) -> dict[str, int]:
    rng = np.random.default_rng(SEED + 77)
    order = sorted(set(subjects))
    perm = rng.permutation(len(order))
    return {order[j]: int(i % N_FOLDS) for i, j in enumerate(perm)}


def validate_complete_vl(unlabeled: list[dict], tracks: dict[str, dict], folds: list[int],
                         vl_supervised: bool, validation_by_fold: bool = True) -> dict:
    """Fail before GPU training if expert coverage or a requested subject split is incomplete."""
    if not tracks:
        raise RuntimeError("complete VL pilot has no embedded expert records")
    if not (vl_supervised or validation_by_fold):
        raise RuntimeError("complete VL pilot requires subject-fold validation")
    expected = {tail3(key): rec for key, rec in tracks.items()}
    if len(expected) != len(tracks):
        raise RuntimeError("complete VL pilot has duplicate embedded source keys")
    vf = subject_folds([rec["subject"] for rec in tracks.values()])
    matched = {}
    for u in unlabeled:
        if u.get("ds") != "vl" or u.get("tracks") is None:
            continue
        key = tail3(u["src"])
        if key in matched:
            raise RuntimeError(f"complete VL pilot has duplicate matched source key: {key}")
        matched[key] = u
    missing, extra = set(expected) - set(matched), set(matched) - set(expected)
    if missing or extra:
        raise RuntimeError(f"incomplete VL expert coverage: matched={len(matched)}/{len(expected)} "
                           f"missing={sorted(missing)[:5]} unexpected={sorted(extra)[:5]}")
    for key, u in matched.items():
        subject = expected[key]["subject"]
        if u["tracks"].get("subject") != subject or u.get("vfold") != vf[subject]:
            raise RuntimeError(f"VL subject/fold metadata mismatch: {key}")
    report = {"n_expected": len(expected), "n_matched": len(matched), "folds": {}}
    for fold in folds:
        val_keys = {key for key, rec in expected.items() if vf[rec["subject"]] == fold}
        if not val_keys:
            raise RuntimeError(f"complete VL pilot has no expected held-out records in fold {fold}")
        val_subjects = {expected[key]["subject"] for key in val_keys}
        actual_val = {key for key, u in matched.items() if u.get("vfold") == fold}
        if actual_val != val_keys:
            raise RuntimeError(f"incomplete VL held-out coverage in fold {fold}")
        split = {"n_val_items": len(actual_val), "n_val_subjects": len(val_subjects)}
        if vl_supervised:
            train = [u for u in fold_training_pool(unlabeled, fold, True)
                     if u.get("ds") == "vl" and u.get("tracks") is not None]
            train_keys = {tail3(u["src"]) for u in train}
            train_subjects = {u["tracks"]["subject"] for u in train}
            if not train_subjects:
                raise RuntimeError(f"complete VL pilot has no training subjects in fold {fold}")
            if train_subjects & val_subjects:
                raise RuntimeError(f"VL subject leakage in fold {fold}: {train_subjects & val_subjects}")
            if train_keys != set(expected) - val_keys or len(train) != len(train_keys):
                raise RuntimeError(f"incomplete VL training coverage in fold {fold}")
            split.update(n_train_items=len(train), n_train_subjects=len(train_subjects))
        report["folds"][str(fold)] = split
    return report


def prep_unlabeled(datasets: dict, teacher: Teacher, tracks: dict) -> tuple[list[dict], list[dict]]:
    """Unlabeled items: viewport crop (+ teacher pseudo-label stack + gate) (+ VL expert tracks in crop coords)."""
    unl, stats = [], []
    vf = subject_folds([t["subject"] for t in tracks.values()]) if tracks else {}
    for ds, entries in datasets.items():
        for j, (key, path, src) in enumerate(entries):
            try:
                gray = read_gray(path)
            except Exception as exc:  # noqa: BLE001
                log(f"skip {ds}/{src}: {exc!r}")
                continue
            if max(gray.shape) > 4000 or min(gray.shape) < 64:
                continue
            box, kind = viewport(gray, path.suffix.lower())
            x1, y1, x2, y2 = box
            crop = np.ascontiguousarray(gray[y1:y2, x1:x2])
            u = {"ds": ds, "key": key, "src": src, "H": gray.shape[0], "W": gray.shape[1], "box": box, "layout": kind,
                 "crop": crop, "viz": j < (4 if ds == "test" else 2) or (ds == "test" and j % 60 == 0)}
            train_pl = ds != "dynamic" or j % DYN_STRIDE == 0
            st = {"ds": ds, "key": key, "layout": kind, "teacher": False, "gate_frac": 0.0, "near_frac": 0.0}
            if PSEUDO and train_pl:
                tm = teacher.get(ds, key, src)
                if tm is not None and tm[0].shape == gray.shape:
                    seg = tm[0][y1:y2, x1:x2]
                    c = tm[1][y1:y2, x1:x2].astype(np.float32) / 127.5 - 1
                    s = tm[2][y1:y2, x1:x2].astype(np.float32) / 127.5 - 1
                    mag = np.hypot(c, s)
                    mg = mag + 1e-6
                    coh = np.hypot(cv2.blur(c / mg, (PS_COH_K, PS_COH_K)), cv2.blur(s / mg, (PS_COH_K, PS_COH_K)))
                    gate = (mag >= PS_MAG) & (coh >= PS_COH)
                    rad = max(1, int(round(ORI_RADIUS * max(crop.shape) / SIZE)))
                    near = cv2.dilate((seg >= int(PS_NEAR * 255)).astype(np.uint8), disk(rad)) > 0
                    gf = float((gate & near).mean())
                    st.update(teacher=True, gate_frac=gf, near_frac=float(near.mean()),
                              gate_of_near=float((gate & near).sum() / max(1, near.sum())))
                    if gf >= PS_MIN_GATE_FRAC:
                        u["pseudo"] = np.ascontiguousarray(
                            np.stack([seg, tm[1][y1:y2, x1:x2], tm[2][y1:y2, x1:x2], gate.astype(np.uint8) * 255], -1))
                elif tm is not None:
                    log(f"teacher shape mismatch {ds}/{key}: {tm[0].shape} vs {gray.shape}")
            if ds == "vl":
                tr = tracks.get(tail3(src))
                if tr is not None and tr["H"] == gray.shape[0] and tr["W"] == gray.shape[1]:
                    u["tracks"] = {"fas": [[(px - x1, py - y1) for px, py in f["pts"]] for f in tr["fas"]],
                                   "sup": [tr["sup"][0] - x1, tr["sup"][1] - y1, tr["sup"][2], tr["sup"][3]],
                                   "deep": [tr["deep"][0] - x1, tr["deep"][1] - y1, tr["deep"][2], tr["deep"][3]],
                                   "pa_deg": tr["pa_deg"], "subject": tr["subject"]}
                    u["vfold"] = vf.get(tr["subject"], -1)
                    st["tracks"] = len(tr["fas"])
            u["train_pl"] = bool(train_pl and ("pseudo" in u or (VL_SUP and "tracks" in u)))
            st["train_pl"] = u["train_pl"]
            unl.append(u)
            stats.append(st)
    return unl, stats


# ----------------------------------------------------------------------------- main
def main() -> None:
    global ITEMS, UNL, BANK_IMGS, REF_METRICS, WARM_CKPTS, _LOG_FILE
    for d in (MASKBANK, CKPT_DIR, LOG_DIR, METRIC_DIR, VIZ_DIR, PART_DIR, TMP / "ext"):
        d.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = LOG_DIR / "main.log"
    log(f"MODE={MODE} model={MODEL_NAME} warm={WARM_START}:{WARM_MODEL} pseudo={PSEUDO}:{PSEUDO_MODEL} vl_sup={VL_SUP} "
        f"multiscale={MULTISCALE} domain_aug={DOMAIN_AUG} python={sys.version.split()[0]} cpu={os.cpu_count()}")
    log("nvidia-smi: " + sh("nvidia-smi -L").strip().replace("\n", " | "))
    log("mem: " + sh("free -g | head -2").strip().replace("\n", " | "))
    import torch  # import only: no torch ops in the parent (they run in workers / isolated subprocesses)

    smp = ensure_smp()
    import timm

    log(f"torch={torch.__version__} smp={smp.__version__} timm={timm.__version__} cv2={cv2.__version__}")

    # ---- attached sources: warm-start checkpoints + folds, teacher maskbank
    warm_root = find_model_source(WARM_MODEL) if WARM_START else None
    teach_root = find_model_source(PSEUDO_MODEL.removesuffix("ms")) if PSEUDO else None
    log(f"warm source: {warm_root}; teacher source: {teach_root}")
    warm_folds: dict[str, int] = {}
    if warm_root is not None:
        fc = warm_root / "metrics" / "folds.csv"
        if fc.exists():
            with open(fc) as fh:
                for r in csv.DictReader(fh):
                    if r["stem"] == r["representative"]:
                        warm_folds[r["stem"]] = int(r["fold"])
        for k in range(N_FOLDS):
            cp = warm_root / "checkpoints" / f"fold{k}.pt"
            if cp.exists():
                WARM_CKPTS[k] = cp
            fj = warm_root / "metrics" / f"fold{k}.json"
            if fj.exists():
                REF_METRICS[k] = json.loads(fj.read_text()).get("best", {})
        log(f"warm folds.csv: {len(warm_folds)} representatives; checkpoints: {sorted(WARM_CKPTS)}")
        # check every warm checkpoint for inf/nan (fp16 overflow) before using it; in a separate process so that
        # the parent never runs torch ops before forking the workers (OpenMP state + fork can hang the children)
        for line in run_isolated(CKPT_CHECK_CODE, [str(WARM_CKPTS[k]) for k in sorted(WARM_CKPTS)]):
            log("warm ckpt " + line)
    if REQUIRE_WARM_START and (not WARM_START or not warm_folds or set(TRAIN_FOLDS) - set(WARM_CKPTS)):
        raise RuntimeError("controlled experiment requires warm folds.csv and every requested fold checkpoint")
    if WARM_START and len(WARM_CKPTS) < len(TRAIN_FOLDS):
        log("WARNING warm-start checkpoints missing for some folds: those folds start from ImageNet")

    image_dir = find_dir("fasc_images_new_model_v1")
    mask_dir = find_dir("fasc_masks_new_model_v1")
    test_dir = find_dir("test_set_v2")
    pairs = find_pairs(image_dir, mask_dir)
    log(f"found {len(pairs)} fascicle pairs; test dir {test_dir}")
    items, alias_rows, stats = prepare_items(pairs, warm_folds)
    log("data stats: " + json.dumps(stats))
    ITEMS = items
    with open(METRIC_DIR / "folds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "representative", "group", "fold"])
        w.writerows(alias_rows)
    (METRIC_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2))

    # ---- unlabeled / target-domain images (pseudo-labels, FDA bank, VL validation, inference)
    datasets: dict[str, list[tuple[str, Path, str]]] = {}
    test_files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES], key=lambda p: p.name)
    datasets["test"] = [(p.stem, p, p.name) for p in test_files][:TEST_LIMIT]
    for name in ("static", "dynamic", "vl"):
        try:
            ent = gather_extra(name)
            if EXTRA_LIMIT is not None:
                ent = ent[:EXTRA_LIMIT] if name != "vl" else ent[:max(EXTRA_LIMIT, 3)]
            datasets[name] = ent
        except Exception:  # noqa: BLE001
            log(f"extra {name} failed:\n{traceback.format_exc()}")
    tracks = decode_vl_tracks()
    teacher = Teacher(teach_root, PSEUDO_MODEL)
    t = time.time()
    UNL, pstats = prep_unlabeled(datasets, teacher, tracks)
    if REQUIRE_COMPLETE_VL:
        vl_coverage = validate_complete_vl(UNL, tracks, TRAIN_FOLDS, VL_SUP, VL_VALIDATION_BY_FOLD)
        (METRIC_DIR / "vl_coverage.json").write_text(json.dumps(vl_coverage, indent=2))
        log("verified complete VL coverage: " + json.dumps(vl_coverage))
    n_pl = sum(u["train_pl"] for u in UNL)
    n_tr = sum("tracks" in u for u in UNL)
    log(f"unlabeled items: {len(UNL)} ({', '.join(f'{k}={len(v)}' for k, v in datasets.items())}); "
        f"pseudo-labelled for training: {n_pl}; with VL tracks: {n_tr}; {time.time() - t:.0f}s, "
        f"{sum(u['crop'].nbytes + (u['pseudo'].nbytes if 'pseudo' in u else 0) for u in UNL) / 2**20:.0f} MB")
    if pstats:
        with open(METRIC_DIR / "pseudo_stats.csv", "w", newline="") as fh:
            keys = sorted({k for s in pstats for k in s})
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(pstats)
    if PSEUDO and n_pl == 0:
        log("WARNING no pseudo-labelled images (teacher maskbank not attached?) -> training on GT only")
    bank_src = [u for u in UNL if (u["train_pl"] or u["ds"] == "test")
                and (STYLE_DATASETS is None or u["ds"] in STYLE_DATASETS)]
    BANK_IMGS = np.stack([make_input(u["crop"])[0] for u in bank_src]) if bank_src else None
    log(f"domain bank: {0 if BANK_IMGS is None else len(BANK_IMGS)} canvases")

    t = time.time()
    if not WARM_START or len(WARM_CKPTS) < len(TRAIN_FOLDS):  # some fold starts from ImageNet: fill the HF cache
        for line in run_isolated(PRETRAIN_CODE, [ENCODER]):
            log(line)
        log(f"pretrained {ENCODER} cached ({time.time() - t:.0f}s)")

    n_gpu = 0 if CPU_SMOKE else max(1, sh("nvidia-smi -L").count("GPU "))
    if n_gpu >= 2 and len(TRAIN_FOLDS) > 1:
        assign = [TRAIN_FOLDS[0::2], TRAIN_FOLDS[1::2]]
    else:
        assign = [TRAIN_FOLDS]
    plan = max(len(a) for a in assign)
    log(f"fold assignment {assign} plan_folds={plan}")
    ctx = mp.get_context("fork")
    if IN_PROCESS:  # debugging aid only: run the worker in the parent (no fork)
        log(f"gpu worker 0 (in-process) exit code {gpu_worker(0, assign[0], plan, 1, forked=False)}")
    else:
        procs = []
        for g, folds in enumerate(assign):
            p = ctx.Process(target=gpu_worker, args=(g, folds, plan, len(assign)), daemon=False)
            p.start()
            log(f"started gpu worker {g} pid={p.pid} folds={folds}")
            procs.append(p)
            time.sleep(5)
        for g, p in enumerate(procs):
            p.join()
            log(f"gpu worker {g} exit code {p.exitcode}")

    # ---- summaries
    fold_res = []
    for f in TRAIN_FOLDS:
        fp = METRIC_DIR / f"fold{f}.json"
        if fp.exists():
            fold_res.append(json.loads(fp.read_text()))
    ok_folds = [r["fold"] for r in fold_res if r.get("reload_ok")]
    summary: dict = {"folds": fold_res, "ok_folds": ok_folds}
    for key in ("dice50", "ang_mae", "ang_mae_pix", "med_err"):
        for tag, src in (("epoch0", "epoch0"), ("best", "best"), ("tta", "tta")):
            vals = [r[src][key] for r in fold_res if r.get(src) and key in r[src]]
            if vals:
                summary[f"mean_{tag}_{key}"] = float(np.mean(vals))
    for key in ("vl_track_err", "vl_strip_err", "vl_pa_mae", "vl_pa_bias"):
        for tag, src in (("epoch0", "epoch0_vl"), ("best", "best_vl")):
            vals = [r[src][key] for r in fold_res if r.get(src) and key in r[src]]
            if vals:
                summary[f"mean_{tag}_{key}"] = float(np.mean(vals))
    (METRIC_DIR / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("CV summary: " + json.dumps({k: v for k, v in summary.items() if k != "folds"}, default=str))
    if not ok_folds:
        raise RuntimeError("no verified fold checkpoints: no maskbank written")

    # ---- ensemble inference on 2 GPUs (image shards), then merge parts
    names = [MODEL_NAME] + ([MS_NAME] if MULTISCALE else [])
    order = sorted(range(len(UNL)), key=lambda i: (UNL[i]["ds"], UNL[i]["key"]))
    n_inf = max(1, min(2, n_gpu)) if not CPU_SMOKE else 1
    shards = [order[g::n_inf] for g in range(n_inf)]
    t = time.time()
    if IN_PROCESS:
        log(f"inference worker 0 (in-process) exit code {infer_worker(0, shards[0], ok_folds, forked=False)}")
    else:
        procs = []
        for g in range(n_inf):
            p = ctx.Process(target=infer_worker, args=(g, shards[g], ok_folds), daemon=False)
            p.start()
            procs.append(p)
        for g, p in enumerate(procs):
            p.join()
            log(f"inference worker {g} exit code {p.exitcode}")
    log(f"ensemble inference {time.time() - t:.0f}s")
    t = time.time()
    for nm in names:
        for ch in CHANNELS:
            n = merge_npz(sorted(PART_DIR.glob(f"oof_fold*__{nm}_{ch}.npz")), MASKBANK / f"train_oof__{nm}_{ch}.npz")
            log(f"merged {n} OOF maps -> train_oof__{nm}_{ch}.npz")
            for ds in datasets:
                n = merge_npz(sorted(PART_DIR.glob(f"inf*_{ds}__{nm}_{ch}.npz")), MASKBANK / f"{ds}__{nm}_{ch}.npz")
                log(f"merged {n} maps -> {ds}__{nm}_{ch}.npz")
    log(f"merge {time.time() - t:.0f}s")

    written = {}
    for ds in [*datasets, "train_oof"]:
        f = MASKBANK / f"{ds}__{MODEL_NAME}_seg.npz"
        if f.exists():
            with zipfile.ZipFile(f) as z:
                written[ds] = {n[:-4] for n in z.namelist()}
    with open(MASKBANK / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "source_filename", "height", "width"])
        for it in ITEMS:
            if it["stem"] in written.get("train_oof", set()):
                w.writerow(["train_oof", it["stem"], it["name"], it["H"], it["W"]])
        for u in UNL:
            if u["key"] in written.get(u["ds"], set()):
                w.writerow([u["ds"], u["key"], u["src"], u["H"], u["W"]])
    with open(MASKBANK / f"{MODEL_NAME}_viewports.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "layout", "x1", "y1", "x2", "y2", "vl_fold"])
        for u in UNL:
            w.writerow([u["ds"], u["key"], u["layout"], *u["box"], u.get("vfold", "")])
    config = {
        "model": MODEL_NAME, "ms_model": MS_NAME if MULTISCALE else None, "mode": MODE, "encoder": ENCODER,
        "arch": "smp.Unet", "smp": smp.__version__, "torch": torch.__version__, "size": SIZE,
        "warm_start": WARM_START, "warm_model": WARM_MODEL, "pseudo": PSEUDO, "pseudo_model": PSEUDO_MODEL,
        "vl_sup": VL_SUP, "multiscale": MULTISCALE, "tile_frac": TILE_FRAC, "domain_aug": DOMAIN_AUG,
        "require_complete_vl": REQUIRE_COMPLETE_VL,
        "channels": {"0": "seg logit", "1": "tanh cos2theta", "2": "tanh sin2theta"},
        "angle_convention": "theta = 0.5*atan2(sin2, cos2) in image pixel coords of the file (x right, y down); "
                            "rising-to-the-right fascicles have theta < 0",
        "maskbank_encoding": "seg = prob*255; cos2/sin2 = (v+1)/2*255 of the fold x hflip mean (vector NOT "
                             "renormalised: |v| < 1 means low agreement); outside the viewport seg=0, cos2=sin2=128",
        "ms_encoding": "fasc_orient_v2ms = feathered average of the full-viewport view and a 2x2 grid of zoom tiles "
                       f"({TILE_FRAC} of each side), each 5-fold x hflip",
        "vl_maps": "subject-fold OOF (VL_SUP)" if VL_SUP else "5-fold ensemble (VL labels never used for training)",
        "vl_dataset": "raw .tiff images only (the annotated .tif copies carry burned-in lines)",
        "pseudo": {"frac": PSEUDO_FRAC, "w_seg": W_PS_SEG, "w_ori": W_PS_ORI, "pos": PS_POS, "neg": PS_NEG,
                   "mag": PS_MAG, "coh": PS_COH, "coh_k": PS_COH_K, "near": PS_NEAR, "dyn_stride": DYN_STRIDE},
        "train": {"batch": BATCH, "lr_enc": LR_ENC, "lr_dec": LR_DEC, "wd": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS,
                  "ema": EMA_DECAY, "folds": TRAIN_FOLDS, "ok_folds": ok_folds, "seed": SEED},
        "data_stats": stats, "elapsed_min": (time.time() - T0) / 60,
    }
    (WORK / f"{MODEL_NAME}_config.json").write_text(json.dumps(config, indent=2, default=str))
    shutil.rmtree(TMP, ignore_errors=True)
    log(f"DONE elapsed={(time.time() - T0) / 60:.1f} min")


# Expert-drawn fascicle lines of the UMUD 'Example Annotated Images' (VL, Esaote MyLab70; OSF xbawc, CC-BY-4.0),
# parsed from the burned-in ImageJ lines (gpuprep/export_vl_tracks.py): per raw .tiff image (keyed by the last
# three path parts), the superficial / deep aponeurosis lines (point + direction) and each accepted fascicle track
# as a polyline (<= 0.5 px from the smoothed track), raw-image pixel coords. zlib + base64 JSON.
VL_TRACKS_B64 = """
eNqkvcmuLb1yJvYu//jULjIY7DSrmUfS0DAE4UK2rlSy1cF1ayAU/O5mML5gBtfO7mCPdKl/n5VrZUaS0XzN3/6vP/6fP//nH3/1
x3/++//8t3/605/++r/9H3+Kf/rTX/75X//8H//+z//2l/G//9tf/81f/4n++PXHX/7+n/9l/OW//f1//in+1+Mv/uvf/9u//9uf
6Osv//yP/zj+6n/8z//z//7z//WX8Yfzs+T/8+e/2AXG6n/7469KaL/++N//+KsWgvz9f/zxV39bYvyiyPUXUfuqodOv8NVbjb/+
S/iK5e9+/fEPf/6z/F0N9av2+otD/OIcm/xZT/pnsdL4w3/8+/8x/u5//fEff5H/+7ex9K/wK4X0Ff7u19/GFtyKU5VV7LrKSVYU
vsb1/janIqvxF2msKs9P6emr/d24xn/8/R9/RfGr1Dy+/z//w5/Hz/t///mf/vtf/vj/fq0LE8mHU+f54cTBrVLWLxX1wqRfqnx1
WbX532L+qvI16vyKifRr9PnfWnVfY3zd1m6+Rijzwvo7kv5G+f/JpXj+xpj0UvMvE5GuSO9Gmf+u6N3I+qVqbfNr9PGFj6/BMX1+
jfkf//QPf/6n+Qedxu36x3/507/+6x9/VfNX4Czf81X48VP48c/CjwqPMCsj+tIX1xJnWJUmYRVqzy7+eDzBXPu4F21E3AzTTnXG
H4V6En+V5f6n8fkzqhh3vMz7GOddjcd9jOUr5puoivOxUGoI2Y6PY/nwoiGb50reFPnwPh7Z8eG11OsP5/lVWb9qaXPR5/PnqGGo
F+I2Y4PrXBX9bzVsF2o53kTD+IPc4xEN44v1dB4N4SQcUvuIh/AtIMYdOo+I8DYkQv+KaWw18nxqayyPWrae8agpBHYxUWjERJJN
KX3Fmsv8y5b1Lwt/DwqK2E3mZjBfJuwM8lFroeHR81x0dosY5hPo+h7HudWslb6r442fqxK2le5jGj6xk1tR0MvpbkiU/WrGLYeg
q9z8qtS50s2MethWfa5If+kM3rVKNFe4C7n4Vclzhe2yNb/qer2MzTP6VdKr653IQf9Sf3uOegX9tTmzXxX9ZvprSwjbav6GqL+2
cPUr/dZRf23pya1qmN86yq+19yJ90c1eHeeVUsUz1mOh4qnqCWSr+QRSxVNtxa/6fHIVT5XYr9KMm4qnWqJf1Xm9hqfam1sl3Uxa
xJMrfsXzeg1PtSa/ajPCGs5b1ivoc2SNxYazqEW/6gEHnT4djUz9LpWTi7Ca81rZXc5jL7+5y/rClY47GbF76Z3sfqX3oCKWU/Yr
1vuqv5P0d9qq62cilqn7lV69IpZr9qvG7vnPmD5Wvbr7mvWp4r6WoKuGmEzubhXsKHa3gttFqh4f2DeqPv+16mvfsPvKX633u12d
v0ZKcezq41ul8n5X5/C4q48A+tmuHrl9jUfHI7OJXy3pSd86zwOc87art/Ffxikm+xzVFnVXb7qr88lR3+2M1B26uFVK825mvE2a
xmXCc54RkfGO6JPFCs852040I6LoW1HittK/LIgB3lYaEYj4oruFrXS3wF9Wjn6l0YIrIPkcScZxzo/k825Dm/+cO350QNJ4vNJ2
FJQc3AZfcndHQan+mCht+3ed3aqGbWPAdo9VqTjqon35/kV0s01QtDzcbaNYJU15yF5UXeEoiQ1ZmD6k+e8Y335uVonxO4v+t+xv
71zZ7R0hGvnuhQtfObsXbiT5Vy/caVb9WdWd5NVXdd3bzFr20l4r/yK5S3HmRl0T65L68bqV8eU5t67BN3ZZreyaVnahnWTWmgnj
tdGNdNztMuNCwzi5eznubrk7e2dAzsc886SGVZ378qyhxselmVXIpWTHr/MBz0uNd5i3h8jrwuN3jTrh+sIzQkYozqw76YePTXsm
2vPokEvNklQP2oCSNM+NPRdXPPSRdt/Gy6ita1vxMn5WSvwb8cLP8cI/jBeS+zUi41eUNz0Q/5L7+1/mxrm25pG79ZFZzy2099zn
3+iffASJPLjYtKKZr3BsTW9011urtx3ZR8TpOvNNqW+Ona6Nar/fhQ+7K2nRF+3DtWYbm/AsAbN2ArBn6U7COHRTxI7ZXPjUztcX
bnZdkhqhzkXVy858haqucrQ+yIxa7XyMX1hcBZ9jvA+eca99zVZaOw8eOgue8hE89D14ykXw0NvTfSRXPYz6WXo6HFk3EdbTfZTt
LoTGtpR/8djB83jZdFOaBVvIXL6H0TyDUrJkOG/Htx6Luh2wtkYqCvx57M8j3SVR4e7U0bd6bOw008OGlkybnx1wsswGTdeWTNMr
RUJKXtY2pAXakRSHu22I9EfFguNOO2MVGT6hF+R+8KgBaL45ATve/FLZfynsjSPTcF9qfNHovlRL9T6jHBX3Crk0tob4GyHXnkOu
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
nHIG1fZ//8//A7YDGos=
"""

if __name__ == "__main__":
    main()
