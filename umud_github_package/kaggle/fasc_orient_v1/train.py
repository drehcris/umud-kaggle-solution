"""UMUD fasc_orient_v1: fascicle fragment segmentation + doubled-angle orientation field.

Two heads on one smp U-Net (ConvNeXt-S fb_in22k_ft_in1k encoder):
  ch0  fascicle fragment logit (targets: GT fragments rendered 3 px wide at 512)
  ch1  tanh -> cos(2 theta)   } supervised on pixels within 8 px of a GT fragment,
  ch2  tanh -> sin(2 theta)   } theta = that fragment's PCA angle (nearest fragment wins)
theta is measured in image pixel coordinates (x right, y down) of the image file.

Input: viewport crop -> aspect-preserving letterbox to 512 -> 3ch gray + ImageNet norm.
All geometric augmentation is one affine matrix applied to the image AND to the fragment
end points; target angles are recomputed from the transformed end points, so flips,
rotations and anisotropic scaling are handled exactly.

Runs 5 GroupKFold folds (near-duplicate groups) on the 2 T4 GPUs in parallel (forked
workers), writes fold checkpoints, OOF maps and 5-fold + hflip TTA maps for the test set
and the OSF benchmark / VL example images in MASKBANK FORMAT.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")  # never initialise CUDA in the parent before fork
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # peak ~13.7 GB of 15 GB at bs 8

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
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------- config
MODE = "full"  # "smoke" or "full"
SMOKE = MODE == "smoke"
T0 = time.time()

SEED = 20261021
SIZE = 512
N_FOLDS = 5
TRAIN_FOLDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
SMOKE_TRAIN, SMOKE_VAL, SMOKE_REPEAT = 30, 10, 4
MAX_EPOCHS = 2 if SMOKE else 30
MIN_EPOCHS = 2 if SMOKE else 8
TRAIN_BUDGET_S = 3.3 * 3600  # all training (incl. preprocessing) must end by T0 + this
HARD_TRAIN_END_S = 3.5 * 3600  # absolute stop for any epoch loop
BATCH = 8
LR_ENC, LR_DEC, WEIGHT_DECAY = 2e-4, 6e-4, 1e-2
WARMUP_EPOCHS, LR_FLOOR = 1.0, 0.02
POS_WEIGHT, W_BCE, W_DICE, W_SKEL, W_ORI = 3.0, 1.0, 1.0, 0.5, 1.0
SEG_WIDTH, ORI_RADIUS = 3, 8.0
DUP_CORR, THUMB = 0.95, 48
ENCODER = "tu-convnext_small.fb_in22k_ft_in1k"
SMP_VERSION = "0.5.0"
MODEL_NAME = "fasc_orient_v1"
TEST_LIMIT = 5 if SMOKE else None
EXTRA_LIMIT = 2 if SMOKE else 800
SEG_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

COMP_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
WORK = Path("/kaggle/working")
TMP = Path("/tmp/fo_tmp")
MASKBANK = WORK / "maskbank"
CKPT_DIR = WORK / "checkpoints"
LOG_DIR = WORK / "logs"
METRIC_DIR = WORK / "metrics"
VIZ_DIR = WORK / "viz"
OOF_PART_DIR = TMP / "oof_parts"
EXTRA_SOURCES = {
    "static": [("https://osf.io/download/hb5xj/", "static.zip")],
    "dynamic": [("https://osf.io/download/vqb7h/", "dynamic.zip")],
    "vl": [
        ("https://osf.io/download/zcuwk/", "vl_young.zip"),
        ("https://osf.io/download/6ga94/", "vl_old.zip"),
    ],
}
CHANNELS = ("seg", "cos2", "sin2")

ITEMS: list[dict] = []  # filled in main(); inherited by forked GPU workers
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


# ----------------------------------------------------------------------------- viewport
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
    """Scanner-layout rules (same as src/umud/geometry.ultrasound_metadata, no OCR).

    Layout rules adapted from AmbrosM, "UMUD Quick and Dirty"
    (https://www.kaggle.com/code/ambrosm/umud-quick-and-dirty), Apache-2.0; see THIRD_PARTY_NOTICES.md.
    """
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


# ----------------------------------------------------------------------------- fragments
MIN_PTS = 6


def _fit(points: np.ndarray):
    c = points.mean(0)
    q = points - c
    evals, evecs = np.linalg.eigh(q.T @ q)
    d = evecs[:, 1]
    n = evecs[:, 0]
    return c, d, n, q @ d, q @ n


def decompose_component(points: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Split one connected component (N,2 float x,y) into straight segments [x1,y1,x2,y2]."""
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
    """Fragments of a binary mask as segments in IMAGE pixel coords (plain stretch)."""
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


# ----------------------------------------------------------------------------- geometry / sample
def letterbox_matrix(w: int, h: int) -> np.ndarray:
    s0 = SIZE / max(w, h)
    return np.array(
        [[s0, 0.0, (SIZE - s0 * w) / 2 + 0.5 * s0 - 0.5], [0.0, s0, (SIZE - s0 * h) / 2 + 0.5 * s0 - 0.5], [0, 0, 1]],
        np.float64,
    )


def aug_matrix(rng: np.random.Generator) -> np.ndarray:
    c = (SIZE - 1) / 2.0
    sc = math.exp(rng.uniform(math.log(0.8), math.log(1.25))) if rng.random() < 0.7 else 1.0
    ar = math.exp(rng.uniform(-0.12, 0.12)) if rng.random() < 0.3 else 1.0
    phi = math.radians(rng.uniform(-12.0, 12.0)) if rng.random() < 0.6 else 0.0
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


def warp_image(src: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Warp src with 3x3 m (src pixel coords -> SIZE canvas), anti-aliased when shrinking."""
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
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _poly_mask(m: np.ndarray, w: int, h: int) -> np.ndarray:
    corners = np.array([[-0.5, -0.5, 1], [w - 0.5, -0.5, 1], [w - 0.5, h - 0.5, 1], [-0.5, h - 0.5, 1]]).T
    p = (m @ corners)[:2].T
    out = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillConvexPoly(out, np.round(p * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    return out


def intensity_aug(x8: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x8.astype(np.float32) * (1.0 / 255.0)
    if rng.random() < 0.8:
        x = np.power(x, rng.uniform(0.65, 1.5)).astype(np.float32)
    if rng.random() < 0.8:
        x = x * rng.uniform(0.8, 1.25) + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.4:  # time-gain-compensation like vertical gain profile
        yy = np.linspace(-0.5, 0.5, SIZE, dtype=np.float32)
        g = 1.0 + rng.uniform(-0.5, 0.5) * yy + rng.uniform(-0.6, 0.6) * (yy * yy - 1.0 / 12)
        x = x * g[:, None]
    if rng.random() < 0.3:  # correlated multiplicative speckle
        nz = rng.standard_normal((SIZE // 2, SIZE // 2), dtype=np.float32)
        nz = cv2.resize(nz, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = x * (1.0 + rng.uniform(0.05, 0.25) * nz)
    if rng.random() < 0.25:
        x = x + rng.uniform(0.01, 0.04) * rng.standard_normal((SIZE, SIZE), dtype=np.float32)
    if rng.random() < 0.25:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.4))
    if rng.random() < 0.15:
        f = rng.uniform(0.45, 0.8)
        small = cv2.resize(x, (int(SIZE * f), int(SIZE * f)), interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    x8 = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", x8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 91))])
        if ok:
            x8 = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    x8[valid == 0] = 0
    return x8


def render_targets(segs_out: np.ndarray, valid: np.ndarray) -> dict:
    """segs_out (K,4) in canvas coords -> seg/skel/orientation targets."""
    from scipy import ndimage

    seg = np.zeros((SIZE, SIZE), np.uint8)
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
        cv2.line(seg, p1, p2, 1, SEG_WIDTH, cv2.LINE_8, 4)
        cv2.line(idm, p1, p2, i + 1, 1, cv2.LINE_8, 4)
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 >= 9.0:
            ctab[i + 1] = (dx * dx - dy * dy) / l2
            stab[i + 1] = 2.0 * dx * dy / l2
            okt[i + 1] = True
    seg &= valid
    skel = ((idm > 0) & (valid > 0)).astype(np.uint8)
    if (idm > 0).any():
        dist, ind = ndimage.distance_transform_edt(idm == 0, return_indices=True)
        nid = idm[ind[0], ind[1]]
        om = (dist <= ORI_RADIUS) & okt[nid] & (valid > 0)
        cos2 = np.where(om, ctab[nid], 0.0).astype(np.float16)
        sin2 = np.where(om, stab[nid], 0.0).astype(np.float16)
        om = om.astype(np.uint8)
    else:
        om = np.zeros((SIZE, SIZE), np.uint8)
        cos2 = np.zeros((SIZE, SIZE), np.float16)
        sin2 = np.zeros((SIZE, SIZE), np.float16)
    return {"seg": seg, "skel": skel, "om": om, "cos2": cos2, "sin2": sin2}


def make_sample(crop: np.ndarray, segs: np.ndarray, rng: np.random.Generator | None) -> dict:
    h, w = crop.shape
    x0, y0, ww, hh = 0, 0, w, h
    if rng is not None:
        if rng.random() < 0.35:  # narrow field of view (e.g. Philips Lumify)
            ww = max(32, int(round(rng.uniform(0.3, 1.0) * w)))
            x0 = int(rng.integers(0, w - ww + 1))
        if rng.random() < 0.15:
            hh = max(32, int(round(rng.uniform(0.65, 1.0) * h)))
            y0 = int(rng.integers(0, h - hh + 1))
    sub = crop[y0:y0 + hh, x0:x0 + ww]
    m = letterbox_matrix(ww, hh)
    if rng is not None:
        m = aug_matrix(rng) @ m
    img = warp_image(sub, m)
    valid = _poly_mask(m, ww, hh)
    if len(segs):
        s = segs - np.array([x0, y0, x0, y0], np.float32)
        p1 = (m[:2, :2] @ s[:, :2].T + m[:2, 2:3]).T
        p2 = (m[:2, :2] @ s[:, 2:].T + m[:2, 2:3]).T
        segs_out = np.concatenate([p1, p2], 1)
    else:
        segs_out = np.zeros((0, 4))
    tg = render_targets(segs_out, valid)
    if rng is not None:
        img = intensity_aug(img, valid, rng)
    tg["img"] = img
    tg["valid"] = valid
    return tg


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


def _merge_job(args):
    parts, dest = args
    return str(dest), merge_npz([Path(p) for p in parts], Path(dest))


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
    smp = importlib.import_module("segmentation_models_pytorch")
    return smp


def build_model(pretrained: bool):
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=3,
        decoder_channels=(256, 128, 64, 32, 16),
    )


def to_input(img_u8, torch):
    x = img_u8.float().unsqueeze(1) * (1.0 / 255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x.expand(-1, 3, -1, -1) - mean) / std).contiguous(memory_format=torch.channels_last)


def forward_maps(model, x, torch, tta: bool):
    """Returns prob, cos2, sin2 (B,H,W) float32; hflip TTA flips the sin2 sign."""
    b = x.shape[0]
    xin = torch.cat([x, torch.flip(x, dims=[3])], 0) if tta else x
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(xin).float()
    p = torch.sigmoid(out[:b, 0])
    c = torch.tanh(out[:b, 1])
    s = torch.tanh(out[:b, 2])
    if tta:
        of = torch.flip(out[b:], dims=[3])  # (B,C,H,W): un-flip the W axis
        p = 0.5 * (p + torch.sigmoid(of[:, 0]))
        c = 0.5 * (c + torch.tanh(of[:, 1]))
        s = 0.5 * (s - torch.tanh(of[:, 2]))  # mirror: theta -> -theta, so sin(2 theta) changes sign
    return p, c, s


class TrainSet:
    def __init__(self, items: list[dict], augment: bool, repeat: int = 1) -> None:
        self.items = items
        self.augment = augment
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.items) * self.repeat

    def __getitem__(self, idx: int):
        import torch

        it = self.items[idx % len(self.items)]
        seed = (torch.initial_seed() + idx * 7919) % (2**32)
        rng = np.random.default_rng(seed) if self.augment else None
        s = make_sample(it["crop"], it["segs"], rng)
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in s.items()}


def _worker_init(_):
    cv2.setNumThreads(1)
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def build_val_cache(items: list[dict], torch) -> list[dict]:
    samples = [make_sample(it["crop"], it["segs"], None) for it in items]
    cache = []
    for i in range(0, len(samples), BATCH):
        chunk = samples[i:i + BATCH]
        cache.append({k: torch.from_numpy(np.stack([s[k] for s in chunk])) for k in chunk[0]})
    return cache


def evaluate(model, cache, torch, tta: bool = False) -> dict:
    model.eval()
    dice_sum = {t: 0.0 for t in SEG_THRESHOLDS}
    n_img, ang_img_sum, ang_img_n, ang_pix_sum, ang_pix_n, med_sum, med_n = 0, 0.0, 0, 0.0, 0, 0.0, 0
    with torch.no_grad():
        for b in cache:
            x = to_input(b["img"].cuda(non_blocking=True), torch)
            p, c, s = forward_maps(model, x, torch, tta)
            seg_t = b["seg"].cuda() > 0
            skel = b["skel"].cuda() > 0
            om = b["om"].cuda() > 0
            valid = b["valid"].cuda() > 0
            ct = b["cos2"].cuda().float()
            st = b["sin2"].cuda().float()
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


def _better(m: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if m["ang_mae"] < best["ang_mae"] - 0.02:
        return True
    return abs(m["ang_mae"] - best["ang_mae"]) <= 0.02 and m["dice50"] > best["dice50"]


def predict_native(models, gray: np.ndarray, box, torch) -> dict:
    """5-fold + hflip TTA maps at native image resolution (uint8 MASKBANK encoding)."""
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    inp, m = make_input(crop)
    x = to_input(torch.from_numpy(inp[None]).cuda(), torch)
    acc = None
    with torch.no_grad():
        for model in models:
            p, c, s = forward_maps(model, x, torch, True)
            cur = torch.stack([p[0], c[0], s[0]], -1)
            acc = cur if acc is None else acc + cur
    maps = (acc / len(models)).cpu().numpy().astype(np.float32)
    back = unletterbox(maps, m, x2 - x1, y2 - y1)
    full = np.zeros((gray.shape[0], gray.shape[1], 3), np.float32)
    full[y1:y2, x1:x2] = back
    return encode_channels(full[..., 0], full[..., 1], full[..., 2]), maps


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


# ----------------------------------------------------------------------------- GPU worker
def train_fold(fold: int, gpu_state: dict, torch) -> None:
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(SEED + fold)
    tr = [it for it in ITEMS if it["fold"] != fold]
    va = [it for it in ITEMS if it["fold"] == fold]
    if SMOKE:
        tr = [tr[i] for i in rng.choice(len(tr), SMOKE_TRAIN, replace=False)]
        va = [va[i] for i in rng.choice(len(va), SMOKE_VAL, replace=False)]
    torch.manual_seed(SEED + fold)
    n_cpu = os.cpu_count() or 4
    n_workers = max(2, min(4, n_cpu // gpu_state["n_parallel"] + 1))
    ds = TrainSet(tr, augment=True, repeat=SMOKE_REPEAT if SMOKE else 1)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=n_workers, pin_memory=True, drop_last=True,
                    persistent_workers=True, worker_init_fn=_worker_init, prefetch_factor=4)
    t_cache = time.time()
    cache = build_val_cache(va, torch)
    log(f"fold {fold}: train={len(tr)} val={len(va)} loader_workers={n_workers} steps/epoch={len(dl)} "
        f"val_cache={time.time() - t_cache:.1f}s")

    model = build_model(pretrained=True).cuda().to(memory_format=torch.channels_last)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENC}, {"params": dec_params, "lr": LR_DEC}],
                            weight_decay=WEIGHT_DECAY)
    steps = len(dl)
    warm = max(1, int(steps * WARMUP_EPOCHS))
    n_epochs = gpu_state.get("n_epochs", MAX_EPOCHS)
    if "ep_time" in gpu_state:  # later folds on this GPU: re-check the remaining budget
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
    scaler = torch.amp.GradScaler("cuda")
    pos_w = torch.tensor([POS_WEIGHT], device="cuda")
    hist_path = METRIC_DIR / f"history_fold{fold}.csv"
    cols = ["epoch", "loss", "bce", "dice_l", "skel_l", "ori_l", "dice50", "dice_best", "thr_best", "ang_mae",
            "ang_mae_pix", "med_err", "lr", "sec"]
    with open(hist_path, "w", newline="") as fh:
        csv.writer(fh).writerow(cols)
    best, best_epoch = None, 0
    ckpt_path = CKPT_DIR / f"fold{fold}.pt"
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
            t_wait += time.time() - t_prev
            x = to_input(b["img"].cuda(non_blocking=True), torch)
            seg_t = b["seg"].cuda(non_blocking=True).float()
            skel = b["skel"].cuda(non_blocking=True).float()
            om = b["om"].cuda(non_blocking=True).float()
            ct = b["cos2"].cuda(non_blocking=True).float()
            st = b["sin2"].cuda(non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            out = out.float()
            logit = out[:, 0]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, seg_t, pos_weight=pos_w)
            p = torch.sigmoid(logit)
            inter = (p * seg_t).sum((1, 2))
            dice_l = (1 - (2 * inter + 1) / (p.sum((1, 2)) + seg_t.sum((1, 2)) + 1)).mean()
            sk_n = skel.sum((1, 2))
            has = sk_n > 0
            if bool(has.any()):
                skel_l = (1 - (p * skel).sum((1, 2))[has] / sk_n[has]).mean()
            else:
                skel_l = logit.sum() * 0
            cp, sp = torch.tanh(out[:, 1]), torch.tanh(out[:, 2])
            ori_l = (((cp - ct).abs() + (sp - st).abs()) * om).sum() / (2 * om.sum().clamp_min(1))
            loss = W_BCE * bce + W_DICE * dice_l + W_SKEL * skel_l + W_ORI * ori_l
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += [float(loss), float(bce), float(dice_l), float(skel_l), float(ori_l)]
            nb += 1
            if step % 50 == 0 or (SMOKE and step % 5 == 0):
                el = time.time() - t_ep
                log(f"fold {fold} ep {epoch} step {step}/{steps} loss={sums[0] / nb:.4f} ori={sums[4] / nb:.4f} "
                    f"ms/step={1000 * el / step:.0f} wait_ms/step={1000 * t_wait / step:.0f} "
                    f"mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            t_prev = time.time()
        t_train = time.time() - t_ep
        met = evaluate(model, cache, torch)
        sec = time.time() - t_ep
        lr_now = opt.param_groups[0]["lr"]
        sums /= max(1, nb)
        log(f"EPOCH fold={fold} ep={epoch}/{n_epochs} loss={sums[0]:.4f} bce={sums[1]:.4f} dice_l={sums[2]:.4f} "
            f"skel_l={sums[3]:.4f} ori_l={sums[4]:.4f} | val dice50={met['dice50']:.4f} dice_best={met['dice_best']:.4f}"
            f"@{met['thr_best']} ang_mae={met['ang_mae']:.2f}deg ang_pix={met['ang_mae_pix']:.2f} "
            f"med_err={met['med_err']:.2f}deg | lr={lr_now:.2e} train={t_train:.0f}s total={sec:.0f}s")
        with open(hist_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, *[round(float(v), 5) for v in sums], met["dice50"], met["dice_best"],
                                     met["thr_best"], met["ang_mae"], met["ang_mae_pix"], met["med_err"], lr_now, sec])
        if _better(met, best):
            best, best_epoch = met, epoch
            sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
            torch.save({"state_dict": sd, "epoch": epoch, "metrics": met, "fold": fold, "encoder": ENCODER,
                        "size": SIZE, "classes": CHANNELS}, ckpt_path)
            log(f"fold {fold}: saved best epoch {epoch} ang_mae={met['ang_mae']:.3f} dice50={met['dice50']:.4f}")
        if epoch == 1 and "n_epochs" not in gpu_state and not SMOKE:
            ep_time = sec
            left = T0 + TRAIN_BUDGET_S - time.time() + ep_time  # budget incl. this epoch
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
    log(f"fold {fold}: best epoch {best_epoch} {json.dumps(best)}")

    # ---- OOF with best checkpoint (hflip TTA) at native resolution
    state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state["state_dict"].items()})
    model.eval()
    met_tta = evaluate(model, cache, torch, tta=True)
    log(f"fold {fold}: TTA val dice50={met_tta['dice50']:.4f} ang_mae={met_tta['ang_mae']:.2f} "
        f"med_err={met_tta['med_err']:.2f}")
    writers = {ch: NpzZip(OOF_PART_DIR / f"fold{fold}_{ch}.npz", level=6) for ch in CHANNELS}
    t_oof = time.time()
    rows = []
    for j, it in enumerate(va):
        gray = np.zeros((it["H"], it["W"]), np.uint8)
        x1, y1, x2, y2 = it["box"]
        gray[y1:y2, x1:x2] = it["crop"]
        enc, _ = predict_native([model], gray, it["box"], torch)
        for ch in CHANNELS:
            writers[ch].add(it["stem"], enc[ch])
        rows.append(["train_oof", it["stem"], it["name"], it["H"], it["W"]])
        if j < 3:
            save_viz(VIZ_DIR / f"oof_fold{fold}_{it['stem']}.jpg", gray, enc)
    for w in writers.values():
        w.close()
    with open(OOF_PART_DIR / f"fold{fold}_manifest.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    res = {"fold": fold, "best_epoch": best_epoch, "n_epochs": n_epochs, "best": best, "tta": met_tta,
           "n_train": len(tr), "n_val": len(va), "oof_seconds": time.time() - t_oof}
    (METRIC_DIR / f"fold{fold}.json").write_text(json.dumps(res, indent=2))
    del model, opt, cache
    torch.cuda.empty_cache()


def gpu_worker(gpu: int, folds: list[int], plan_folds: int, n_parallel: int) -> None:
    global _LOG_PREFIX, _LOG_FILE
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _LOG_PREFIX = f"[gpu{gpu}]"
    _LOG_FILE = LOG_DIR / f"gpu{gpu}.log"
    code = 0
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        log(f"worker start folds={folds} device={torch.cuda.get_device_name(0)}")
        state = {"plan_folds": plan_folds, "n_parallel": n_parallel, "done": 0}
        for k in folds:
            try:
                train_fold(k, state, torch)
            except Exception:  # noqa: BLE001
                code = 1
                log(f"fold {k} FAILED:\n{traceback.format_exc()}")
                torch.cuda.empty_cache()
            state["done"] += 1
    except BaseException:  # noqa: BLE001
        code = 2
        log(f"worker FAILED:\n{traceback.format_exc()}")
    os._exit(code)


# ----------------------------------------------------------------------------- data discovery
def find_dir(name: str) -> Path:
    for base in (Path("/kaggle/input/competitions") / COMP_SLUG, Path("/kaggle/input") / COMP_SLUG, Path("/kaggle/input")):
        if base.exists():
            hits = [p for p in base.rglob(name) if p.is_dir()]
            if hits:
                return sorted(hits, key=lambda p: len(str(p)))[0]
    raise FileNotFoundError(name)


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


def prepare_items(pairs: list[tuple[str, str]]) -> tuple[list[dict], list[list], dict]:
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
    for i, it in enumerate(items):
        it["group"] = int(groups[i])
        it["fold"] = int(folds[i])
        del it["thumb"]
        for a in it["aliases"]:
            alias_rows.append([a, it["stem"], it["group"], it["fold"]])
    _, gcounts = np.unique(groups, return_counts=True)
    stats.update(
        unique_items=len(items), n_groups=int(len(gcounts)), max_group=int(gcounts.max()),
        fold_sizes=[int((folds == f).sum()) for f in range(N_FOLDS)],
        segments_median=float(np.median([len(it["segs"]) for it in items])),
        split_components=int(sum(it["nsplit"] for it in items)),
        components=int(sum(it["ncomp"] for it in items)),
        lowcov_fullframe=int(sum("_lowcov" in it["layout"] for it in items)),
        layouts={k: int(v) for k, v in zip(*np.unique([it["layout"] for it in items], return_counts=True))},
        crop_mb=float(sum(it["crop"].nbytes for it in items) / 2**20),
    )
    return items, alias_rows, stats


# ----------------------------------------------------------------------------- extra datasets
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


def gather_extra(name: str) -> list[tuple[str, Path, str]]:
    """Returns [(image_id, path, source_filename)] for an OSF dataset (downloaded + unzipped)."""
    root = TMP / "ext" / name
    root.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for url, fname in EXTRA_SOURCES[name]:
        zpath = TMP / "ext" / fname
        if not download(url, zpath):
            continue
        sub = root / Path(fname).stem
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(sub)
        zpath.unlink()
        files += [p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                  and "__MACOSX" not in p.parts and not p.name.startswith("._")]
    if name == "static":
        sel = [p for p in files if p.stem.startswith("im_") and p.stem.endswith("_arch")]
        files = sel or files
    elif name == "dynamic":
        sel = [p for p in files if "frames" in p.parts or p.stem.startswith("image_")]
        files = sel or files
    files = sorted(files, key=lambda p: str(p.relative_to(root)))
    stem_counts: dict[str, int] = {}
    for p in files:
        stem_counts[p.stem] = stem_counts.get(p.stem, 0) + 1
    out, used = [], set()
    for p in files:
        key = p.stem  # non-unique stems (e.g. VL ANON_1 per subject) get their two parent folders prefixed
        if stem_counts[p.stem] > 1:
            key = "_".join([*p.parent.parts[-2:], p.stem])
        k2, n = key, 1
        while k2 in used:
            n += 1
            k2 = f"{key}_{n}"
        used.add(k2)
        out.append((k2, p, str(p.relative_to(root))))
    log(f"extra {name}: {len(out)} images")
    return out[:EXTRA_LIMIT]


# ----------------------------------------------------------------------------- main
def main() -> None:
    global ITEMS, _LOG_FILE
    for d in (MASKBANK, CKPT_DIR, LOG_DIR, METRIC_DIR, VIZ_DIR, OOF_PART_DIR, TMP / "ext"):
        d.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = LOG_DIR / "main.log"
    log(f"MODE={MODE} python={sys.version.split()[0]} cpu={os.cpu_count()}")
    log("nvidia-smi: " + sh("nvidia-smi -L").strip().replace("\n", " | "))
    log("mem: " + sh("free -g | head -2").strip().replace("\n", " | "))
    import torch

    torch.set_num_threads(1)
    smp = ensure_smp()
    import timm

    log(f"torch={torch.__version__} smp={smp.__version__} timm={timm.__version__} cv2={cv2.__version__}")

    image_dir = find_dir("fasc_images_new_model_v1")
    mask_dir = find_dir("fasc_masks_new_model_v1")
    test_dir = find_dir("test_set_v2")
    pairs = find_pairs(image_dir, mask_dir)
    log(f"found {len(pairs)} fascicle pairs in {image_dir} / {mask_dir}; test dir {test_dir}")
    items, alias_rows, stats = prepare_items(pairs)
    log("data stats: " + json.dumps(stats))
    ITEMS = items
    with open(METRIC_DIR / "folds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "representative", "group", "fold"])
        w.writerows(alias_rows)
    (METRIC_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2))

    t = time.time()
    _m = build_model(pretrained=True)  # warm HF cache on CPU before forking (no CUDA in parent)
    n_params = sum(p.numel() for p in _m.parameters())
    del _m
    log(f"pretrained {ENCODER} cached ({n_params / 1e6:.1f}M params, {time.time() - t:.0f}s)")

    n_gpu = max(1, sh("nvidia-smi -L").count("GPU "))
    if n_gpu >= 2 and len(TRAIN_FOLDS) > 1:
        assign = [TRAIN_FOLDS[0::2], TRAIN_FOLDS[1::2]]
    else:
        assign = [TRAIN_FOLDS]
    plan = max(len(a) for a in assign)
    log(f"GPU fold assignment {assign} plan_folds={plan}")
    ctx = mp.get_context("fork")
    procs = []
    for g, folds in enumerate(assign):
        p = ctx.Process(target=gpu_worker, args=(g, folds, plan, len(assign)), daemon=False)
        p.start()
        procs.append(p)
        time.sleep(5)

    # while GPUs train: fetch the OSF benchmark / VL images
    datasets: dict[str, list[tuple[str, Path, str]]] = {}
    test_files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES], key=lambda p: p.name)
    datasets["test"] = [(p.stem, p, p.name) for p in test_files][:TEST_LIMIT]
    log(f"test images: {len(test_files)} (using {len(datasets['test'])})")
    for name in ("static", "dynamic", "vl"):
        try:
            datasets[name] = gather_extra(name)
        except Exception:  # noqa: BLE001
            log(f"extra {name} failed:\n{traceback.format_exc()}")

    while any(p.is_alive() for p in procs):
        time.sleep(30)
    for g, p in enumerate(procs):
        p.join()
        log(f"gpu worker {g} exit code {p.exitcode}")

    # ---- summaries
    fold_res = []
    for f in TRAIN_FOLDS:
        fp = METRIC_DIR / f"fold{f}.json"
        if fp.exists():
            fold_res.append(json.loads(fp.read_text()))
    summary = {"folds": fold_res}
    if fold_res:
        for key in ("dice50", "ang_mae", "ang_mae_pix", "med_err"):
            summary[f"mean_best_{key}"] = float(np.mean([r["best"][key] for r in fold_res]))
            summary[f"mean_tta_{key}"] = float(np.mean([r["tta"][key] for r in fold_res]))
    (METRIC_DIR / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    log("CV summary: " + json.dumps({k: v for k, v in summary.items() if k != "folds"}))

    # ---- merge OOF parts
    manifest_rows: list[list] = []
    jobs = [([str(OOF_PART_DIR / f"fold{f}_{ch}.npz") for f in TRAIN_FOLDS], str(MASKBANK / f"train_oof__{MODEL_NAME}_{ch}.npz"))
            for ch in CHANNELS]
    t = time.time()
    with ctx.Pool(3) as pool:
        for dest, n in pool.imap_unordered(_merge_job, jobs):
            log(f"merged {n} OOF maps -> {Path(dest).name}")
    for f in TRAIN_FOLDS:
        mp_path = OOF_PART_DIR / f"fold{f}_manifest.csv"
        if mp_path.exists():
            with open(mp_path) as fh:
                manifest_rows += list(csv.reader(fh))
    log(f"OOF merge {time.time() - t:.0f}s")

    # ---- 5-fold + hflip TTA inference
    ckpts = sorted(CKPT_DIR.glob("fold*.pt"))
    if not ckpts:
        raise RuntimeError("no checkpoints were produced")
    models = []
    for cp in ckpts:
        m = build_model(pretrained=False)
        st = torch.load(cp, map_location="cpu", weights_only=False)
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st["state_dict"].items()})
        models.append(m.cuda().eval().to(memory_format=torch.channels_last))
    log(f"inference with {len(models)} fold models: {[c.name for c in ckpts]}")
    vp_rows = []
    for name in ("test", "static", "dynamic", "vl"):
        entries = datasets.get(name) or []
        if not entries:
            continue
        t = time.time()
        writers = {ch: NpzZip(MASKBANK / f"{name}__{MODEL_NAME}_{ch}.npz") for ch in CHANNELS}
        n_ok = 0
        for j, (key, path, src) in enumerate(entries):
            try:
                gray = read_gray(path)
                if max(gray.shape) > 4000 or min(gray.shape) < 64:
                    log(f"skip {name}/{src}: shape {gray.shape}")
                    continue
                box, kind = viewport(gray, path.suffix.lower())
                enc, _ = predict_native(models, gray, box, torch)
                for ch in CHANNELS:
                    writers[ch].add(key, enc[ch])
                manifest_rows.append([name, key, src, gray.shape[0], gray.shape[1]])
                vp_rows.append([name, key, kind, *box])
                n_ok += 1
                if j < (4 if name == "test" else 2) or (name == "test" and j % 60 == 0):
                    save_viz(VIZ_DIR / f"{name}_{key}.jpg", gray, enc)
            except Exception:  # noqa: BLE001
                log(f"inference failed {name}/{src}:\n{traceback.format_exc()}")
        for w in writers.values():
            w.close()
        log(f"inference {name}: {n_ok}/{len(entries)} images in {time.time() - t:.0f}s")

    with open(MASKBANK / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "source_filename", "height", "width"])
        w.writerows(manifest_rows)
    with open(MASKBANK / f"{MODEL_NAME}_viewports.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "layout", "x1", "y1", "x2", "y2"])
        w.writerows(vp_rows)
    config = {
        "model": MODEL_NAME, "mode": MODE, "encoder": ENCODER, "arch": "smp.Unet", "smp": smp.__version__,
        "torch": torch.__version__, "size": SIZE, "channels": {"0": "seg logit", "1": "tanh cos2theta", "2": "tanh sin2theta"},
        "angle_convention": "theta = 0.5*atan2(sin2, cos2) in image pixel coords of the file (x right, y down); "
                            "rising-to-the-right fascicles have theta < 0",
        "maskbank_encoding": "seg = prob*255; cos2/sin2 = (v+1)/2*255 of the 5-fold x hflip mean (vector NOT renormalised: "
                             "|v| < 1 means low agreement); outside the viewport seg=0, cos2=sin2=128",
        "input": "viewport crop -> aspect-preserving letterbox 512 (black pad), gray x3, ImageNet norm",
        "targets": {"seg_width_px": SEG_WIDTH, "ori_radius_px": ORI_RADIUS, "pos_weight": POS_WEIGHT,
                    "loss_weights": [W_BCE, W_DICE, W_SKEL, W_ORI]},
        "train": {"batch": BATCH, "lr_enc": LR_ENC, "lr_dec": LR_DEC, "wd": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS,
                  "folds": TRAIN_FOLDS, "dup_corr": DUP_CORR, "seed": SEED},
        "data_stats": stats, "elapsed_min": (time.time() - T0) / 60,
    }
    (WORK / f"{MODEL_NAME}_config.json").write_text(json.dumps(config, indent=2, default=str))
    shutil.rmtree(TMP, ignore_errors=True)
    log(f"DONE elapsed={(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
