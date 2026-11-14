"""Protocol-faithful muscle-architecture measurement from probability maps (geometry v2).

Converts an aponeurosis probability map, a fascicle probability map, optional orientation
channels (doubled-angle cos2/sin2), optional aponeurosis edge maps, the grayscale image and the
vertical px/cm into (pa_deg, fl_mm, mt_mm) plus diagnostics.

Conventions
-----------
* Image coordinates: x to the right (column), y downwards (row). Pixel ``k`` covers ``[k, k+1)``
  so a binary band whose last row is ``b`` has its lower edge at ``y = b + 1``.
* Angles are in degrees in image coordinates (a fascicle rising to the right has a negative angle).
* Orientation channels use the doubled-angle convention ``cos2 = cos(2*theta)``, ``sin2 = sin(2*theta)``.
* MT is the mean of vertical (global image) distances between the INNER edges of the superficial
  and deep aponeuroses at configurable x positions.
* PA is measured against the deep (lower) aponeurosis (``pa_reference="deep"``) or the horizontal.
* One px/cm (vertical scale bar) is applied to both axes; no anisotropy correction (organiser rule).

Every option is switchable through :class:`GeoConfig`; ``GeoConfig.v1()`` reproduces the
geometry that scored 0.467 on the public LB (scratchpad gtgeo.py + 23_test_geometry.py "naive").
All PA/FL variants are computed in one call and returned in ``GeoResult.variants`` so that an
evaluation can compare them without recomputation; the headline values follow the config.

Quick start (integration)
-------------------------
>>> from umud import geometry_v2 as g2
>>> cfg = g2.recommended_config()                  # v2 default (see scripts/eval_geometry_v2.py)
>>> bank = g2.MaskBank("artifacts/maskbank")        # lazy npz access, <dataset>__<model>.npz
>>> res = g2.measure_from_bank(bank, "test", "IMG_00001", "ours_apo384_noflip", "ours_fasc384_noflip",
...                            image=gray_uint8, px_per_cm=135.3, cfg=cfg)
>>> res.pa_deg, res.fl_mm, res.mt_mm, res.confidence, res.flags, res.variants["fl_straight"]
>>> res.as_row()                                    # flat dict for a DataFrame row

Headline methods: PA "fusion" = mean(fragment median vs global deep line, classical directional
line-energy PA from the raw image); FL "ensemble" = mean(straight extrapolation median, 3 central
least-extrapolated fragments, constructed fascicle at the fused PA between the inner-edge curves);
MT = mean vertical inner-edge distance at 25/50/75 % of the shared aponeurosis x-range.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label, regionprops

PA_METHODS = ("global", "local", "central3", "field", "structure", "fusion")
FL_METHODS = ("straight", "central3", "streamline", "triangle", "gated", "constructed", "ensemble")
PA_ALIASES = {"a": "global", "b": "local", "c": "central3", "d": "field", "e": "structure", "f": "fusion"}
FL_ALIASES = {"a": "straight", "b": "central3", "c": "streamline", "d": "triangle", "e": "gated",
              "f": "constructed", "g": "ensemble"}


# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GeoConfig:
    # ---- aponeurosis bands
    apo_thr: float = 0.5               # high threshold on the apo probability
    apo_low: float | None = None       # hysteresis low threshold (None = single threshold)
    apo_edge_level: float | None = None  # probability level of the sub-pixel edge (None = apo_thr)
    apo_subpixel: bool = True          # sub-pixel edge from the probability profile
    apo_close_w: int = 0               # horizontal closing (px) to bridge gaps; 0 = off
    apo_min_area: int = 30
    apo_min_span: float = 0.25         # band x-span >= this fraction of the reference width
    apo_span_ref: str = "image"        # "image" width | "viewport" (detected from the image)
    apo_merge: bool = True             # merge collinear pieces of one band before the span test
    apo_merge_dy: float = 0.025        # max vertical mismatch at the junction (fraction of H)
    apo_merge_gap: float = 0.35        # max x gap between pieces (fraction of W)
    apo_thick_clip: float = 3.0        # drop columns thicker than k*median (bridges/blobs)
    # ---- pair selection
    pair_mode: str = "superficial"     # "superficial" | "coverage" (v1) | "top2"
    pair_min_sep: float = 0.05         # min inner-edge separation (fraction of H)
    pair_max_angle: float = 25.0       # max angle between superficial and deep band lines (deg)
    pair_deep_min_span: float = 0.35   # continuity of the deep band (fraction of W)
    pair_min_cov: float = 0.15         # min fraction of fascicle mass enclosed by the pair
    pair_margin: float = 5.0           # px margin for the enclosure test
    pair_max_cross: float = 0.5        # reject deep candidates crossed by fascicles (0 = off)
    pair_oblique_only: bool = True     # pair tests use only oblique fascicle components
    # ---- inner-edge curves
    edge_fit: str = "poly2"            # "poly1" | "poly2" | "poly2r" (robust) | "savgol"
    edge_extrap: str = "tangent"       # beyond the band ends: "tangent" | "poly" | "end_linear" | "global_linear"
    edge_end_frac: float = 0.3         # end_linear: fraction of band columns used for the end slope
    savgol_frac: float = 0.15          # Savitzky-Golay window as fraction of band columns
    edge_source: str = "mask"          # "mask" | "edgemap" (use apo_upper/apo_lower maps) | "centroid"
                                       # (apo_v2 edge channels: probability-weighted centre per column)
    edge_search: float = 0.0           # centroid: half window (px) around the mask edge; 0 = max(3, H/128)
    edge_min_mass: float = 0.3         # centroid: min edge probability mass per column (else mask edge)
    # ---- MT
    mt_xpos: tuple[float, ...] = (0.25, 0.5, 0.75)
    mt_xref: str = "shared"            # "shared" (data overlap) | "union" | "image"
    mt_fit: str | None = None          # curve fit used for MT (None = edge_fit; v1 = "poly1")
    mt_source: str = "fit"             # "fit" | "raw" (median of raw edge within +-5 columns)
    # ---- fascicle fragments
    fasc_thr: float = 0.5
    frag_min_area: int = 5
    frag_min_len: float = 10.0         # major axis length (px)
    frag_max_minor_ratio: float = 1.0  # minor/major axis; < 1 drops blobs (v1: 1.0 = off)
    frag_inside_margin: float = 5.0    # centre must lie between the curves +- margin (px)
    frag_sign_consistency: bool = False  # drop fragments inclined opposite to the majority
    pa_min: float = 3.0
    pa_max: float = 45.0
    # ---- PA
    pa_method: str = "global"          # one of PA_METHODS (or letter a-e)
    pa_fallback: tuple[str, ...] = ("global", "structure")
    pa_reference: str = "deep"         # "deep" | "horizontal"
    central_frac: float = 0.6          # central part of the shared x-range used by central3
    central_n: int = 3
    field_strip: tuple[float, float] = (0.03, 0.35)  # strip above deep apo, fraction of MT
    # ---- FL
    fl_method: str = "straight"        # one of FL_METHODS (or letter a-f)
    fl_primary: str = "straight"       # primary estimate for the gated FL
    fl_constructed_x: tuple[float, ...] = (0.4, 0.5, 0.6)  # midpoints (fraction of shared range)
    pa_fusion: tuple[str, ...] = ("global", "structure")   # PA methods averaged by "fusion"
    fl_ensemble: tuple[str, ...] = ("straight", "central3", "constructed")  # FL methods combined
    fl_ensemble_agg: str = "mean"      # "mean" | "median"
    fl_k: float = 1.065                # FL = k * MT / sin(PA)
    fl_gate_tol: float = 0.25          # relative disagreement that triggers the gate
    fl_gate_mode: str = "blend"        # "blend" | "fallback"
    fl_blend_w: float = 0.5            # weight of the primary estimate when blending
    # ---- streamlines / orientation fields
    sl_field: str = "auto"             # "learned" | "fragments" | "structure" | "auto"
    sl_seeds: int = 7
    sl_seed_range: tuple[float, float] = (0.3, 0.7)
    sl_step: float = 2.0               # px
    frag_field_sigma: float = 0.35     # Gaussian width of the fragment field (fraction of MT)
    # ---- classical (raw-image) orientation
    st_kind: str = "radon"             # "radon" (directional line energy) | "tensor" | "hessian"
    st_line_mm: float = 10.0           # radon: line-integral length (mm); st_line_frac*MT without ppc
    st_line_frac: float = 0.35
    st_hp_mm: float = 1.0              # radon: high-pass Gaussian sigma (mm)
    st_use_frag_sign: bool = True      # take the fascicle inclination sign from the fragments
    st_mask_outside: bool = True       # radon: zero the high-pass outside the muscle compartment
    st_symmetric: bool = True          # radon: subtract the mirrored (about horizontal) energy
    st_sigma_d: float = 1.5            # derivative scale (px at st_ref_ppc)
    st_sigma_t: float = 6.0            # tensor / vector smoothing (px at st_ref_ppc)
    st_ref_ppc: float = 100.0          # sigmas scale with px_per_cm / st_ref_ppc when ppc given
    st_margin: float = 0.2             # exclude this fraction of MT next to each aponeurosis
    st_region: str = "compartment"     # "compartment" | "deep_strip"
    st_agg: str = "mode"               # "mode" | "median" | "mean"
    st_xtrim: float = 0.1              # trim this fraction of the shared x-range on each side
    st_reject_edge: bool = False       # radon: discard the classical PA when its maximum sits on the
                                       # search boundary (pa_min / pa_max) -> fusion uses the fragments
    flag_pa_range: tuple[float, float] = (6.0, 40.0)  # final PA outside -> flag "pa_implausible"
    # ---- what to compute
    compute_structure: bool = True
    compute_streamline: bool = True
    flag_extrap: float = 0.93          # flag when the median fragment covers < 7 % of its fascicle
    clip: bool = False
    clip_pa: tuple[float, float] = (5.0, 45.0)
    clip_fl: tuple[float, float] = (30.0, 200.0)
    clip_mt: tuple[float, float] = (5.0, 60.0)

    @staticmethod
    def v1() -> "GeoConfig":
        """The geometry that produced public LB 0.467 (gtgeo.py naive)."""
        return GeoConfig(apo_merge=False, apo_thick_clip=0.0, pair_mode="coverage", edge_fit="poly2",
                         edge_extrap="poly", mt_fit="poly1", frag_sign_consistency=False,
                         frag_max_minor_ratio=1.0, pa_method="global", fl_method="straight",
                         compute_structure=False, compute_streamline=False)

    def with_(self, **kw) -> "GeoConfig":
        return replace(self, **kw)


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------
def as_prob(a: np.ndarray | None) -> np.ndarray | None:
    """bool / uint8 (0-255) / float (0-1 or 0-255) -> float32 in [0, 1]."""
    if a is None:
        return None
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == bool:
        return a.astype(np.float32)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    a = a.astype(np.float32)
    if a.size and float(np.nanmax(a)) > 1.0 + 1e-6:
        a = a / 255.0
    return a


def decode_orient(c: np.ndarray) -> np.ndarray:
    """Orientation channel: uint8 (v+1)/2*255 -> float in [-1, 1]; float input returned as is."""
    c = np.asarray(c)
    if c.dtype == np.uint8:
        return c.astype(np.float32) / 255.0 * 2.0 - 1.0
    return c.astype(np.float32)


def as_gray(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return img


def wrap90(d):
    """Wrap an angle difference (deg) into [-90, 90)."""
    return (np.asarray(d, float) + 90.0) % 180.0 - 90.0


def _nanmedian(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _iqr(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.subtract(*np.percentile(v, [75, 25]))) if v.size >= 2 else float("nan")


def _robust_mean(v, floor: float = 3.0) -> float:
    """Mean of the values within max(floor, 2*MAD) of the median."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan")
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    keep = np.abs(v - med) <= max(floor, 2.0 * mad)
    return float(v[keep].mean())


def _line_fit(x, y) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, float(np.mean(y)) if len(y) else 0.0
    s, b = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(s), float(b)


# --------------------------------------------------------------------------------------------
# aponeurosis bands
# --------------------------------------------------------------------------------------------
@dataclass
class Band:
    x: np.ndarray        # column positions (float, integer valued)
    top: np.ndarray      # upper edge y per column
    bot: np.ndarray      # lower edge y per column
    area: int
    prob: float          # mean probability inside the band

    @property
    def x0(self) -> float:
        return float(self.x.min())

    @property
    def x1(self) -> float:
        return float(self.x.max())

    @property
    def span(self) -> float:
        return self.x1 - self.x0 + 1.0

    @property
    def ctr(self) -> np.ndarray:
        return (self.top + self.bot) / 2.0

    @property
    def yc(self) -> float:
        return float(np.median(self.ctr))

    @property
    def thickness(self) -> float:
        return float(np.median(self.bot - self.top))

    def line(self, key: str = "ctr") -> tuple[float, float]:
        return _line_fit(self.x, getattr(self, key))

    def angle(self, key: str = "ctr") -> float:
        return float(np.degrees(np.arctan(self.line(key)[0])))


def _band_from_pixels(ys: np.ndarray, xs: np.ndarray, p: np.ndarray, level: float, subpixel: bool,
                      thick_clip: float) -> Band | None:
    H = p.shape[0]
    x0, x1 = int(xs.min()), int(xs.max())
    n = x1 - x0 + 1
    tmin = np.full(n, H + 1, dtype=np.int64)
    tmax = np.full(n, -1, dtype=np.int64)
    np.minimum.at(tmin, xs - x0, ys)
    np.maximum.at(tmax, xs - x0, ys)
    ok = tmax >= 0
    cols = np.arange(x0, x1 + 1)[ok]
    t, b = tmin[ok], tmax[ok]
    if subpixel:
        pin_t = p[t, cols]
        pout_t = np.where(t - 1 >= 0, p[np.maximum(t - 1, 0), cols], 0.0)
        ft = np.clip((level - pout_t) / np.maximum(pin_t - pout_t, 1e-6), 0.0, 1.0)
        top = t - 0.5 + ft
        pin_b = p[b, cols]
        pout_b = np.where(b + 1 < H, p[np.minimum(b + 1, H - 1), cols], 0.0)
        fb = np.clip((pin_b - level) / np.maximum(pin_b - pout_b, 1e-6), 0.0, 1.0)
        bot = b + 0.5 + fb
    else:
        top, bot = t.astype(float), b + 1.0
    if thick_clip and len(cols) > 10:
        th = bot - top
        keep = th <= thick_clip * np.median(th) + 3.0
        if keep.sum() >= 5:
            cols, top, bot = cols[keep], top[keep], bot[keep]
    return Band(x=cols.astype(float), top=top.astype(float), bot=bot.astype(float), area=int(len(ys)),
                prob=float(p[ys, xs].mean()))


def _merge_bands(bands: list[Band], W: int, H: int, cfg: GeoConfig) -> list[Band]:
    """Greedily merge collinear pieces of one aponeurosis broken by gaps."""
    bands = [b for b in bands]
    changed = True
    while changed and len(bands) > 1:
        changed = False
        bands.sort(key=lambda b: b.x0)
        best = None
        for i, j in itertools.combinations(range(len(bands)), 2):
            a, b = bands[i], bands[j]
            if a.x0 > b.x0:
                a, b = b, a
            overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
            if overlap > 0.2 * min(a.span, b.span):
                continue
            gap = b.x0 - a.x1
            if gap > cfg.apo_merge_gap * W:
                continue
            xj = 0.5 * (a.x1 + b.x0)
            sa, ia = a.line("ctr")
            sb, ib = b.line("ctr")
            # predict each piece at the junction with its own line; pieces must meet
            dy = abs((sa * xj + ia) - (sb * xj + ib))
            dang = abs(np.degrees(np.arctan(sa)) - np.degrees(np.arctan(sb)))
            if dy <= cfg.apo_merge_dy * H and dang <= 12.0:
                cand = (dy, i, j)
                if best is None or cand < best:
                    best = cand
        if best is not None:
            _, i, j = best
            a, b = bands[i], bands[j]
            x = np.concatenate([a.x, b.x])
            o = np.argsort(x, kind="stable")
            x, top, bot = x[o], np.concatenate([a.top, b.top])[o], np.concatenate([a.bot, b.bot])[o]
            # duplicated columns (small overlap): keep the first occurrence
            keep = np.concatenate([[True], np.diff(x) > 0])
            merged = Band(x=x[keep], top=top[keep], bot=bot[keep], area=a.area + b.area,
                          prob=(a.prob * a.area + b.prob * b.area) / max(a.area + b.area, 1))
            bands = [bb for k, bb in enumerate(bands) if k not in (i, j)] + [merged]
            changed = True
    return bands


def estimate_viewport_x(img: np.ndarray) -> tuple[int, int]:
    """Horizontal extent of the ultrasound image area (largest run of textured columns)."""
    g = as_gray(img).astype(np.float32)
    H, W = g.shape
    rows = g[int(0.2 * H):int(0.8 * H)]
    act = (rows.std(axis=0) > 6.0) & (rows.mean(axis=0) > 3.0)
    act = np.convolve(act.astype(float), np.ones(9) / 9, mode="same") > 0.5
    best, start = (0, W), None
    runs = []
    for x, a in enumerate(np.append(act, False)):
        if a and start is None:
            start = x
        elif not a and start is not None:
            runs.append((start, x))
            start = None
    if runs:
        r = max(runs, key=lambda t: t[1] - t[0])
        if r[1] - r[0] >= 0.1 * W:
            best = r
    return int(best[0]), int(best[1])


def extract_bands(apo: np.ndarray, cfg: GeoConfig = GeoConfig(), span_ref: float | None = None) -> list[Band]:
    """Connected aponeurosis bands spanning >= apo_min_span of ``span_ref`` (default: the image
    width; pass the ultrasound viewport width for narrow-FOV layouts), sorted top to bottom."""
    p = as_prob(apo)
    H, W0 = p.shape
    W = float(span_ref) if span_ref else W0
    if cfg.apo_low is not None:
        m = apply_hysteresis_threshold(p, cfg.apo_low, cfg.apo_thr)
        level = cfg.apo_edge_level if cfg.apo_edge_level is not None else cfg.apo_low
    else:
        m = p >= cfg.apo_thr if cfg.apo_thr > 0 else p > 0
        level = cfg.apo_edge_level if cfg.apo_edge_level is not None else cfg.apo_thr
    if cfg.apo_close_w and cfg.apo_close_w > 1:
        k = np.ones((1, int(cfg.apo_close_w)), np.uint8)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    lab = label(m, connectivity=2)
    bands = []
    for r in regionprops(lab):
        if r.area < cfg.apo_min_area:
            continue
        ys, xs = r.coords[:, 0], r.coords[:, 1]
        if cfg.apo_merge:
            if (xs.max() - xs.min() + 1) < 0.04 * W:
                continue
        elif (xs.max() - xs.min() + 1) < cfg.apo_min_span * W:
            continue
        b = _band_from_pixels(ys, xs, p, level, cfg.apo_subpixel, cfg.apo_thick_clip)
        if b is not None and len(b.x) >= 3:
            bands.append(b)
    if cfg.apo_merge:
        bands = _merge_bands(bands, int(W), H, cfg)
    bands = [b for b in bands if b.span >= cfg.apo_min_span * W]
    bands.sort(key=lambda b: b.yc)
    return bands


# --------------------------------------------------------------------------------------------
# inner-edge curves
# --------------------------------------------------------------------------------------------
class EdgeCurve:
    """Smooth y(x) through an inner aponeurosis edge with tangent (or polynomial) extrapolation."""

    def __init__(self, x: np.ndarray, y: np.ndarray, method: str = "poly2", extrap: str = "tangent",
                 savgol_frac: float = 0.15, end_frac: float = 0.3):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        o = np.argsort(x, kind="stable")
        x, y = x[o], y[o]
        self.xd, self.yd = x, y
        self.x0, self.x1 = float(x[0]), float(x[-1])
        self.method, self.extrap = method, extrap
        self._poly = None
        self._lut = None
        if method in ("poly1", "poly2", "poly2r", "poly1r", "poly3"):
            deg = int(method[4])
            deg = max(0, min(deg, len(x) - 1))
            c = np.polyfit(x, y, deg)
            if method.endswith("r"):
                for _ in range(6):
                    r = y - np.polyval(c, x)
                    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 0.25
                    u = r / (4.685 * s)
                    w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
                    if (w > 0).sum() < deg + 3:
                        break
                    c = np.polyfit(x, y, deg, w=np.sqrt(w))
            self._poly = c
            self._dpoly = np.polyder(c) if deg > 0 else np.array([0.0])
            self.y_end = (float(np.polyval(c, self.x0)), float(np.polyval(c, self.x1)))
            self.s_end = (float(np.polyval(self._dpoly, self.x0)), float(np.polyval(self._dpoly, self.x1)))
        elif method == "savgol":
            gx = np.arange(int(np.ceil(self.x0)), int(np.floor(self.x1)) + 1, dtype=float)
            if len(gx) < 7:
                gx = x
            gy = np.interp(gx, x, y)
            n = len(gx)
            win = max(7, int(round(savgol_frac * n)))
            win = win if win % 2 == 1 else win + 1
            win = min(win, n if n % 2 == 1 else n - 1)
            sy = savgol_filter(gy, win, 2, mode="interp") if win >= 5 else gy
            self._lut = (gx, sy)
            k = max(5, int(0.1 * n))
            s0, _ = _line_fit(gx[:k], sy[:k])
            s1, _ = _line_fit(gx[-k:], sy[-k:])
            self.y_end = (float(sy[0]), float(sy[-1]))
            self.s_end = (s0, s1)
        else:
            raise ValueError(f"unknown edge fit {method!r}")
        if extrap in ("end_linear", "global_linear"):
            # straight continuation beyond the band ends with a robust slope instead of the
            # end derivative of the fit (which amplifies curvature at the ends)
            if extrap == "global_linear":
                sg, _ = _line_fit(x, y)
                self.s_end = (sg, sg)
            else:
                n = len(x)
                k = max(5, int(end_frac * n))
                s0, _ = _line_fit(x[:k], y[:k])
                s1, _ = _line_fit(x[-k:], y[-k:])
                self.s_end = (s0, s1)
        elif extrap not in ("tangent", "poly"):
            raise ValueError(f"unknown extrapolation {extrap!r}")

    def _inner(self, x):
        if self._poly is not None:
            return np.polyval(self._poly, x)
        gx, sy = self._lut
        return np.interp(x, gx, sy)

    def _inner_slope(self, x):
        if self._poly is not None:
            return np.polyval(self._dpoly, x)
        gx, sy = self._lut
        d = np.gradient(sy, gx) if len(gx) > 1 else np.zeros_like(sy)
        return np.interp(x, gx, d)

    def __call__(self, x):
        x = np.asarray(x, float)
        if self.extrap == "poly" and self._poly is not None:
            return np.polyval(self._poly, x)
        y = self._inner(np.clip(x, self.x0, self.x1))
        y = np.where(x < self.x0, self.y_end[0] + self.s_end[0] * (x - self.x0), y)
        y = np.where(x > self.x1, self.y_end[1] + self.s_end[1] * (x - self.x1), y)
        return y

    def slope(self, x):
        x = np.asarray(x, float)
        if self.extrap == "poly" and self._poly is not None:
            return np.polyval(self._dpoly, x)
        s = self._inner_slope(np.clip(x, self.x0, self.x1))
        s = np.where(x < self.x0, self.s_end[0], s)
        s = np.where(x > self.x1, self.s_end[1], s)
        return s

    def raw_at(self, x: float, half: float = 5.0) -> float:
        sel = np.abs(self.xd - x) <= half
        return float(np.median(self.yd[sel])) if sel.any() else float(self(x))


def _edge_from_map(band: Band, emap: np.ndarray, which: str, search: float = 4.0) -> np.ndarray:
    """Refine a band edge with an edge-probability map: per column argmax within the band +- search."""
    H = emap.shape[0]
    base = band.top if which == "top" else band.bot
    out = base.copy()
    for k, (x, y0) in enumerate(zip(band.x.astype(int), base)):
        lo = int(max(0, np.floor(y0 - search)))
        hi = int(min(H, np.ceil(y0 + search) + 1))
        col = emap[lo:hi, x]
        if col.size < 3 or col.max() <= 0.05:
            continue
        i = int(np.argmax(col))
        off = 0.0
        if 0 < i < col.size - 1:
            a, b, c = col[i - 1], col[i], col[i + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
        out[k] = lo + i + 0.5 + off
    return out


# --------------------------------------------------------------------------------------------
# pair selection
# --------------------------------------------------------------------------------------------
def _coverage(sup: Band, deep: Band, fy: np.ndarray, fx: np.ndarray, fw: np.ndarray, margin: float,
              key_s: str = "ctr", key_d: str = "ctr") -> float:
    if fy.size == 0:
        return 0.0
    ss, bs = sup.line(key_s)
    sd, bd = deep.line(key_d)
    inside = (fy >= ss * fx + bs - margin) & (fy <= sd * fx + bd + margin)
    return float((fw * inside).sum() / max(fw.sum(), 1e-9))


def _pair_stats(sup: Band, deep: Band, W: int) -> dict:
    lo, hi = max(sup.x0, deep.x0), min(sup.x1, deep.x1)
    if hi <= lo:
        lo, hi = min(sup.x0, deep.x0), max(sup.x1, deep.x1)
    xs = np.linspace(lo, hi, 25)
    ss, bs = sup.line("bot")
    sd, bd = deep.line("top")
    gap = (sd * xs + bd) - (ss * xs + bs)
    return dict(sep=float(np.median(gap)), min_gap=float(gap.min()),
                dang=abs(sup.angle("ctr") - deep.angle("ctr")), overlap=float(hi - lo) / W)


def crossing_ratio(sup: Band, band: Band, fm: np.ndarray, pad: int = 3) -> float:
    """Fascicle density inside ``band`` (+-pad px) relative to the density between ``sup`` and it.

    Fascicles end at a real aponeurosis, so a real deep band has a ratio << 1; a spurious band
    (bright fascicle / septum segmented as aponeurosis) is crossed by fascicles (ratio >~ 1).
    """
    H = fm.shape[0]
    inb = n_inb = comp = n_comp = 0
    ss, bs = sup.line("bot")
    for x, t, b in zip(band.x.astype(int), band.top, band.bot):
        a0, a1 = max(0, int(t) - pad), min(H, int(b) + pad + 1)
        inb += int(fm[a0:a1, x].sum())
        n_inb += a1 - a0
        c0, c1 = max(0, int(ss * x + bs) + pad), max(0, int(t) - pad)
        if c1 > c0:
            comp += int(fm[c0:c1, x].sum())
            n_comp += c1 - c0
    d_in = inb / max(n_inb, 1)
    d_comp = comp / max(n_comp, 1)
    if d_comp <= 0:
        return 0.0 if d_in <= 0 else float("inf")
    return float(d_in / d_comp)


def oblique_fascicle_map(fasc_p: np.ndarray, bands: list[Band], cfg: GeoConfig) -> np.ndarray:
    """Fascicle probability restricted to elongated components inclined >= pa_min to the bands.

    Near-horizontal detections (aponeurosis-like lines that some fascicle models pick up) are
    removed so that they cannot distort the enclosure / crossing tests of the pair selection.
    """
    ref = float(np.median([b.angle("ctr") for b in bands])) if bands else 0.0
    m = fasc_p >= cfg.fasc_thr if cfg.fasc_thr > 0 else fasc_p > 0
    lab = label(m, connectivity=2)
    keep = np.zeros(lab.max() + 1, bool)
    for r in regionprops(lab):
        if r.area < cfg.frag_min_area or r.axis_major_length < cfg.frag_min_len:
            continue
        ys, xs = r.coords[:, 0].astype(float), r.coords[:, 1].astype(float)
        cov = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))
        ev, evec = np.linalg.eigh(cov)
        vx, vy = evec[:, 1]
        ang = math.degrees(math.atan2(vy, vx))
        if abs(float(wrap90(ang - ref))) >= cfg.pa_min:
            keep[r.label] = True
    return np.where(keep[lab], fasc_p, 0.0).astype(np.float32)


def select_pair(bands: list[Band], fasc_p: np.ndarray | None, H: int, W: int,
                cfg: GeoConfig = GeoConfig()) -> tuple[Band, Band, dict] | None:
    """Pick (superficial, deep) aponeurosis bands of the SUPERFICIAL muscle."""
    if len(bands) < 2:
        return None
    info = dict(n_bands=len(bands), pair_mode=cfg.pair_mode, pair_fallback=False)
    if fasc_p is not None:
        fm = fasc_p >= cfg.fasc_thr
        fy, fx = np.nonzero(fm)
        fw = np.ones(fy.size) if cfg.pair_mode == "coverage" else fasc_p[fy, fx].astype(float)
    else:
        fm = None
        fy = fx = fw = np.zeros(0)
    if cfg.pair_mode == "top2" or len(bands) == 2 and cfg.pair_mode == "coverage" or (
            cfg.pair_mode == "coverage" and fy.size == 0):
        info["pair"] = (0, 1)
        return bands[0], bands[1], info

    if cfg.pair_mode == "superficial":
        for i in range(len(bands) - 1):
            cands = []
            for j in range(i + 1, len(bands)):
                st = _pair_stats(bands[i], bands[j], W)
                if st["sep"] < cfg.pair_min_sep * H or st["min_gap"] < 0:
                    continue
                if st["dang"] > cfg.pair_max_angle:
                    continue
                if bands[j].span < cfg.pair_deep_min_span * W:
                    continue
                use_cross = fm is not None and fy.size and cfg.pair_max_cross > 0
                cr = crossing_ratio(bands[i], bands[j], fm) if use_cross else float("nan")
                cands.append((j, st, cr))
            # the crossing test is only informative when some candidate is NOT crossed
            # (some fascicle models also light up the aponeuroses themselves)
            informative = any(np.isfinite(c[2]) and c[2] <= cfg.pair_max_cross for c in cands)
            for j, st, cr in cands:
                if informative and cr > cfg.pair_max_cross:
                    info["n_crossed_rejected"] = info.get("n_crossed_rejected", 0) + 1
                    continue  # crossed by fascicles -> not the deep aponeurosis
                cov = _coverage(bands[i], bands[j], fy, fx, fw, cfg.pair_margin, "bot", "top") if fy.size else 1.0
                if cov >= cfg.pair_min_cov:
                    info.update(pair=(i, j), pair_cov=cov, pair_cross=cr, cross_informative=informative,
                                **{f"pair_{k}": v for k, v in st.items()})
                    return bands[i], bands[j], info
                break  # first valid deep band below i does not enclose fascicles -> next superficial
        info["pair_fallback"] = True
        if fy.size == 0:
            info["pair"] = (0, 1)
            return bands[0], bands[1], info
    # coverage (v1): enclosed fascicle fraction - 0.001 * separation
    best = None
    for i, j in itertools.combinations(range(len(bands)), 2):
        cov = _coverage(bands[i], bands[j], fy, fx, fw, 5.0, "ctr", "ctr")
        ss, bs = bands[i].line("ctr")
        sd, bd = bands[j].line("ctr")
        sep = float(np.median((sd * fx + bd) - (ss * fx + bs)))
        sc = cov - 0.001 * sep
        if best is None or sc > best[0]:
            best = (sc, i, j, cov)
    _, i, j, cov = best
    info.update(pair=(i, j), pair_cov=cov)
    return bands[i], bands[j], info


# --------------------------------------------------------------------------------------------
# fascicle fragments
# --------------------------------------------------------------------------------------------
def extract_fragments(fasc: np.ndarray, cfg: GeoConfig = GeoConfig()) -> list[dict]:
    """Straight fascicle fragments (PCA of each connected component)."""
    p = as_prob(fasc)
    m = p >= cfg.fasc_thr if cfg.fasc_thr > 0 else p > 0
    lab = label(m, connectivity=2)
    out = []
    for r in regionprops(lab):
        if r.area < cfg.frag_min_area:
            continue
        if r.axis_major_length < cfg.frag_min_len:
            continue
        if cfg.frag_max_minor_ratio < 1.0 and r.axis_minor_length > cfg.frag_max_minor_ratio * r.axis_major_length:
            continue
        ys, xs = r.coords[:, 0].astype(float), r.coords[:, 1].astype(float)
        cx, cy = xs.mean(), ys.mean()
        cov = np.cov(np.stack([xs - cx, ys - cy]))
        ev, evec = np.linalg.eigh(cov)
        vx, vy = evec[:, 1]
        if vx < 0:
            vx, vy = -vx, -vy
        proj = (xs - cx) * vx + (ys - cy) * vy
        pmin, pmax = proj.min(), proj.max()
        # pixel centres are at +0.5 in the edge convention used for the aponeurosis
        out.append(dict(cx=cx + 0.5, cy=cy + 0.5, vx=float(vx), vy=float(vy),
                        length=float(pmax - pmin + 1.0), angle=float(np.degrees(np.arctan2(vy, vx))),
                        p0=(cx + 0.5 + pmin * vx, cy + 0.5 + pmin * vy),
                        p1=(cx + 0.5 + pmax * vx, cy + 0.5 + pmax * vy),
                        prob=float(p[r.coords[:, 0], r.coords[:, 1]].mean()),
                        minor=float(r.axis_minor_length), area=int(r.area)))
    return out


def _line_curve_x(fr: dict, curve: EdgeCurve, xgrid: np.ndarray, side: str) -> tuple[float, float] | None:
    """Intersection of the fragment line with a curve; nearest root on the requested side."""
    if abs(fr["vx"]) < 1e-6:
        return None
    m = fr["vy"] / fr["vx"]
    d = curve(xgrid) - (fr["cy"] + (xgrid - fr["cx"]) * m)
    s = np.flatnonzero(np.diff(np.sign(d)) != 0)
    if not len(s):
        return None
    roots = xgrid[s] - d[s] * (xgrid[s + 1] - xgrid[s]) / (d[s + 1] - d[s])
    ly = fr["cy"] + (roots - fr["cx"]) * m
    if side == "deep":
        ok = ly >= fr["cy"] - 1.0
    else:
        ok = ly <= fr["cy"] + 1.0
    if ok.any():
        roots = roots[ok]
    x = roots[np.argmin(np.abs(roots - fr["cx"]))]
    return float(x), float(curve(x))


# --------------------------------------------------------------------------------------------
# orientation fields
# --------------------------------------------------------------------------------------------
class OrientationField:
    """Doubled-angle field (c2, s2) with confidence w on the pixel grid (optionally coarse)."""

    def __init__(self, c2: np.ndarray, s2: np.ndarray, w: np.ndarray, step: float = 1.0, name: str = ""):
        n = np.hypot(c2, s2) + 1e-9
        self.c2 = (c2 / n).astype(np.float32)
        self.s2 = (s2 / n).astype(np.float32)
        self.w = w.astype(np.float32)
        self.step = float(step)
        self.name = name
        self.h, self.wd = c2.shape

    def sample(self, x: float, y: float) -> tuple[float, float, float]:
        # grid node k sits at pixel-centre coordinate (k + 0.5) * step
        gx = x / self.step - 0.5
        gy = y / self.step - 0.5
        if gx < 0 or gy < 0 or gx > self.wd - 1 or gy > self.h - 1:
            return 0.0, 0.0, 0.0
        i0, j0 = int(gy), int(gx)
        i1, j1 = min(i0 + 1, self.h - 1), min(j0 + 1, self.wd - 1)
        fy, fx = gy - i0, gx - j0
        a = (1 - fy) * (1 - fx)
        b = (1 - fy) * fx
        c = fy * (1 - fx)
        d = fy * fx
        c2 = a * self.c2[i0, j0] + b * self.c2[i0, j1] + c * self.c2[i1, j0] + d * self.c2[i1, j1]
        s2 = a * self.s2[i0, j0] + b * self.s2[i0, j1] + c * self.s2[i1, j0] + d * self.s2[i1, j1]
        w = a * self.w[i0, j0] + b * self.w[i0, j1] + c * self.w[i1, j0] + d * self.w[i1, j1]
        return float(c2), float(s2), float(w)


def field_from_orient(cos2: np.ndarray, sin2: np.ndarray, seg: np.ndarray | None = None) -> OrientationField:
    c2, s2 = decode_orient(cos2), decode_orient(sin2)
    mag = np.clip(np.hypot(c2, s2), 0, 1)
    w = mag if seg is None else mag * as_prob(seg)
    return OrientationField(c2, s2, w, 1.0, "learned")


def field_from_fragments(frags: list[dict], H: int, W: int, sigma: float, step: int = 8) -> OrientationField | None:
    if not frags:
        return None
    gy, gx = np.mgrid[0:int(np.ceil(H / step)), 0:int(np.ceil(W / step))]
    px = (gx.ravel() + 0.5) * step
    py = (gy.ravel() + 0.5) * step
    V = np.zeros((2, px.size))
    vg = np.zeros(2)
    for fr in frags:
        a = np.array(fr["p0"])
        b = np.array(fr["p1"])
        ab = b - a
        L2 = float(ab @ ab) + 1e-9
        t = np.clip(((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / L2, 0, 1)
        dx = px - (a[0] + t * ab[0])
        dy = py - (a[1] + t * ab[1])
        wt = fr["length"] * fr["prob"] * np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
        th = math.radians(fr["angle"])
        V[0] += wt * math.cos(2 * th)
        V[1] += wt * math.sin(2 * th)
        vg += fr["length"] * fr["prob"] * np.array([math.cos(2 * th), math.sin(2 * th)])
    vg /= np.linalg.norm(vg) + 1e-9
    tot = np.hypot(V[0], V[1])
    eps = 1e-3 * (tot.max() + 1e-9)
    c2 = (V[0] + eps * vg[0]).reshape(gy.shape)
    s2 = (V[1] + eps * vg[1]).reshape(gy.shape)
    return OrientationField(c2, s2, np.ones_like(c2), float(step), "fragments")


def structure_field(img: np.ndarray, sigma_d: float, sigma_t: float, kind: str = "tensor") -> OrientationField:
    """Classical fascicle orientation of the raw image (scanner agnostic)."""
    g = img.astype(np.float32)
    if kind == "tensor":
        g = cv2.GaussianBlur(g, (0, 0), sigma_d)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_t)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_t)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_t)
        # gradient doubled angle (jxx-jyy, 2jxy); fascicles are perpendicular -> negate
        c2, s2 = -(jxx - jyy), -2.0 * jxy
        tr = jxx + jyy + 1e-6
        coh = np.hypot(c2, s2) / tr
        w = coh * coh
    elif kind == "hessian":
        s = sigma_d
        g = cv2.GaussianBlur(g, (0, 0), s)
        hxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        hyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        hxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        lam_min = 0.5 * (hxx + hyy) - np.sqrt(0.25 * (hxx - hyy) ** 2 + hxy ** 2)
        ridge = np.maximum(0.0, -lam_min)  # bright ridge strength
        # eigenvector of the largest eigenvalue = along-ridge direction
        n = np.hypot(hxx - hyy, 2 * hxy) + 1e-6
        c2 = cv2.GaussianBlur(ridge * (hxx - hyy) / n, (0, 0), sigma_t)
        s2 = cv2.GaussianBlur(ridge * 2 * hxy / n, (0, 0), sigma_t)
        rs = cv2.GaussianBlur(ridge, (0, 0), sigma_t) + 1e-6
        w = np.hypot(c2, s2) / rs
        w = w * w
    else:
        raise ValueError(kind)
    return OrientationField(c2, s2, w, 1.0, f"structure-{kind}")


def _line_kernel(length: float, angle_deg: float, width_sigma: float = 0.8) -> np.ndarray:
    r = int(np.ceil(length / 2 + 2 * width_sigma + 1))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    t = math.radians(angle_deg)
    along = xx * math.cos(t) + yy * math.sin(t)
    across = -xx * math.sin(t) + yy * math.cos(t)
    k = np.exp(-0.5 * (across / width_sigma) ** 2) * np.clip(length / 2 + 0.5 - np.abs(along), 0, 1)
    return (k / k.sum()).astype(np.float32)


def directional_pa(img: np.ndarray, sup: EdgeCurve, deep: EdgeCurve, xlo: float, xhi: float, mt_px: float,
                   line_px: float, margin: float, hp_sigma: float, pa_min: float, pa_max: float,
                   pa_reference: str = "deep", sign: float | None = None, step: float = 1.0,
                   mask_outside: bool = True, symmetric: bool = True) -> dict:
    """Classical PA: the direction whose long line-integral keeps the most image energy.

    The high-passed image is smoothed along each candidate direction with a line kernel of
    ``line_px``; speckle (short correlation length) averages out at every angle while aligned
    fascicles survive, so the energy peaks at the fascicle direction. Angles are measured
    against the global deep-aponeurosis line (or the horizontal).
    """
    H, W = img.shape
    xs = np.arange(max(0, int(xlo)), min(W, int(xhi) + 1))
    if xs.size < 10:
        return dict(pa=float("nan"), sign=float("nan"), prominence=float("nan"))
    top = sup(xs + 0.5) + margin * mt_px
    bot = deep(xs + 0.5) - margin * mt_px
    pad = int(line_px / 2 + 4)
    y0 = int(max(0, np.floor(top.min()) - pad))
    y1 = int(min(H, np.ceil(bot.max()) + pad + 1))
    x0 = int(max(0, xs[0] - pad))
    x1 = int(min(W, xs[-1] + pad + 1))
    if y1 - y0 < 10:
        return dict(pa=float("nan"), sign=float("nan"), prominence=float("nan"))
    g = img[y0:y1, x0:x1].astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), hp_sigma)
    rows = np.arange(y0, y1)[:, None]
    cols = np.arange(x0, x1)[None, :]
    colx = np.clip(cols, xs[0], xs[-1])
    M = ((rows >= sup(colx + 0.5) + margin * mt_px) & (rows <= deep(colx + 0.5) - margin * mt_px)
         & (cols >= xs[0]) & (cols <= xs[-1]))
    if M.sum() < 200:
        return dict(pa=float("nan"), sign=float("nan"), prominence=float("nan"))
    if mask_outside:
        # keep the line integrals inside the muscle: zero the high-pass outside a slightly wider
        # compartment so that the bright aponeurosis bands never enter the kernel support
        Mw = (rows >= sup(colx + 0.5) + 0.5 * margin * mt_px) & (rows <= deep(colx + 0.5) - 0.5 * margin * mt_px)
        hp = np.where(Mw, hp, 0.0).astype(np.float32)
    ds, _ = _line_fit(deep.xd, deep.yd)
    ref = float(np.degrees(np.arctan(ds))) if pa_reference == "deep" else 0.0
    if sign is not None and np.isfinite(sign):
        rels = sign * np.arange(max(0.0, pa_min - step), pa_max + step + 1e-9, step)
        rels = np.sort(rels)
    else:
        rels = np.arange(-pa_max - step, pa_max + step + 1e-9, step)
    def energy(phi: float) -> float:
        R = cv2.filter2D(hp, -1, _line_kernel(line_px, phi), borderType=cv2.BORDER_REFLECT)
        return float(np.mean(R[M] ** 2))

    E = np.array([energy(ref + rl) for rl in rels])
    if symmetric:
        # speckle / horizontal texture is mirror-symmetric about the image horizontal, fascicles
        # are not: subtract the energy of the mirrored direction
        E = E - np.array([energy(-(ref + rl)) for rl in rels])
    a = np.abs(rels)
    inr = (a >= pa_min) & (a <= pa_max)
    if sign is None or not np.isfinite(sign):
        ep = E[inr & (rels > 0)].max()
        en = E[inr & (rels < 0)].max()
        sign = 1.0 if ep >= en else -1.0
    sel = inr & (np.sign(rels) == sign)
    idx = np.flatnonzero(sel)
    i = idx[np.argmax(E[idx])]
    off = 0.0
    if 0 < i < len(E) - 1 and sel[i - 1] and sel[i + 1]:
        den = E[i - 1] - 2 * E[i] + E[i + 1]
        off = 0.5 * (E[i - 1] - E[i + 1]) / den if abs(den) > 1e-12 else 0.0
    pa = abs(rels[i] + off * step)
    prom = float(E[i] / max(np.median(np.abs(E)), 1e-12))
    edge_hit = bool(i == idx[0] or i == idx[-1])
    return dict(pa=float(pa), sign=float(sign), prominence=prom, edge_hit=edge_hit)


def _region_angles(field_: OrientationField, sup: EdgeCurve, deep: EdgeCurve, xlo: float, xhi: float,
                   y_from, y_to, pa_reference: str, sub: int = 2, weight: np.ndarray | None = None):
    """Relative angles (deg, signed, vs deep tangent or horizontal) and weights inside a region."""
    xs = np.arange(int(np.ceil(xlo)), int(np.floor(xhi)) + 1, sub)
    if xs.size == 0 or field_.step != 1.0:
        return np.zeros(0), np.zeros(0)
    H, W = field_.c2.shape
    xs = xs[(xs >= 0) & (xs < W)]
    rel, ws = [], []
    tang = np.degrees(np.arctan(deep.slope(xs + 0.5))) if pa_reference == "deep" else np.zeros(xs.size)
    y0 = y_from(xs + 0.5)
    y1 = y_to(xs + 0.5)
    for k, x in enumerate(xs):
        a, b = int(max(0, np.ceil(y0[k]))), int(min(H, np.floor(y1[k])))
        if b <= a:
            continue
        rows = np.arange(a, b, sub)
        th = 0.5 * np.degrees(np.arctan2(field_.s2[rows, x], field_.c2[rows, x]))
        rel.append(wrap90(th - tang[k]))
        w = field_.w[rows, x]
        if weight is not None:
            w = w * weight[rows, x]
        ws.append(w)
    if not rel:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(rel), np.concatenate(ws).astype(float)


def _angle_stat(rel: np.ndarray, w: np.ndarray, agg: str, pa_min: float, pa_max: float,
                sign: float | None = None) -> tuple[float, float]:
    """PA (deg, positive) from signed relative angles; returns (pa, dominant sign)."""
    ok = np.isfinite(rel) & (w > 0)
    rel, w = rel[ok], w[ok]
    if rel.size < 20 or w.sum() <= 0:
        return float("nan"), float("nan")
    if agg == "mean":
        c = np.sum(w * np.cos(np.radians(2 * rel)))
        s = np.sum(w * np.sin(np.radians(2 * rel)))
        m = 0.5 * np.degrees(np.arctan2(s, c))
        return float(abs(m)), float(np.sign(m))
    a = np.abs(rel)
    band = (a >= pa_min) & (a <= pa_max)
    if sign is None:
        wp = w[band & (rel > 0)].sum()
        wn = w[band & (rel < 0)].sum()
        sign = 1.0 if wp >= wn else -1.0
    sel = band & (np.sign(rel) == sign)
    if sel.sum() < 10:
        return float("nan"), sign
    if agg == "median":
        v, ww = a[sel], w[sel]
        o = np.argsort(v)
        cw = np.cumsum(ww[o])
        return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])]), sign
    # mode of a smoothed weighted histogram (0.5 deg bins, sigma 1.5 deg), refined by parabola
    bins = np.arange(pa_min, pa_max + 0.5, 0.5)
    h, _ = np.histogram(a[sel], bins=bins, weights=w[sel])
    k = np.arange(-6, 7)
    ker = np.exp(-0.5 * (k * 0.5 / 1.5) ** 2)
    hs = np.convolve(h, ker / ker.sum(), mode="same")
    i = int(np.argmax(hs))
    off = 0.0
    if 0 < i < len(hs) - 1:
        den = hs[i - 1] - 2 * hs[i] + hs[i + 1]
        off = 0.5 * (hs[i - 1] - hs[i + 1]) / den if abs(den) > 1e-12 else 0.0
    centers = 0.5 * (bins[:-1] + bins[1:])
    return float(centers[i] + off * 0.5), sign


def _trace(field_: OrientationField, seed: tuple[float, float], d0: np.ndarray, target: EdgeCurve,
           side: str, step: float, max_len: float) -> tuple[float, tuple[float, float]] | None:
    """Integrate a streamline from seed until it crosses the target curve (RK2, sign-continuous)."""
    p = np.array(seed, float)
    prev = d0 / (np.linalg.norm(d0) + 1e-12)
    ema = prev.copy()
    sgn = 1.0 if side == "deep" else -1.0

    def g(q):  # >= 0 once the target is crossed
        return sgn * (q[1] - float(target(q[0])))

    def direc(q, pr):
        c2, s2, w = field_.sample(q[0], q[1])
        if w <= 1e-3 or (c2 == 0.0 and s2 == 0.0):
            return ema
        th = 0.5 * math.atan2(s2, c2)
        d = np.array([math.cos(th), math.sin(th)])
        if d @ pr < 0:
            d = -d
        return d

    g0 = g(p)
    if g0 >= 0:
        return 0.0, (float(p[0]), float(p[1]))
    L = 0.0
    n = int(max_len / step) + 1
    for _ in range(n):
        d1 = direc(p, prev)
        d2 = direc(p + 0.5 * step * d1, d1)
        q = p + step * d2
        gq = g(q)
        if gq >= 0:
            t = g0 / (g0 - gq) if g0 != gq else 1.0
            q = p + t * (q - p)
            return L + t * step, (float(q[0]), float(q[1]))
        L += step
        p, g0, prev = q, gq, d2
        ema = 0.9 * ema + 0.1 * d2
        ema /= np.linalg.norm(ema) + 1e-12
    return None


def constructed_fl(sup: EdgeCurve, deep: EdgeCurve, x_mid: float, pa_deg: float, sign: float, mt_px: float,
                   pa_reference: str, xgrid: np.ndarray, iters: int = 4) -> float:
    """Length of a straight fascicle at angle ``sign*pa_deg`` to the local deep tangent, inserted
    on the deep curve and running up to the superficial curve, with its midpoint at ``x_mid``."""
    rel = sign * pa_deg
    xi = x_mid - 0.5 * mt_px / max(math.tan(math.radians(pa_deg)), 1e-3) * (-1.0 if rel > 0 else 1.0)
    out = float("nan")
    for _ in range(iters):
        yi = float(deep(xi))
        ref = math.degrees(math.atan(float(deep.slope(xi)))) if pa_reference == "deep" else 0.0
        th = math.radians(ref + rel)
        vx, vy = math.cos(th), math.sin(th)
        if vx < 0:
            vx, vy = -vx, -vy
        fr = dict(cx=xi, cy=yi - 1e-3, vx=vx, vy=vy)
        ps = _line_curve_x(fr, sup, xgrid, "sup")
        if ps is None:
            return float("nan")
        out = math.hypot(ps[0] - xi, ps[1] - yi)
        mid = 0.5 * (ps[0] + xi)
        xi += x_mid - mid
    return out


# --------------------------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------------------------
@dataclass
class GeoResult:
    pa_deg: float
    fl_mm: float
    mt_mm: float
    variants: dict = field(default_factory=dict)   # pa_*, fl_* (mm), mt_* (mm)
    diag: dict = field(default_factory=dict)       # scalar diagnostics
    flags: list = field(default_factory=list)
    confidence: float = float("nan")
    curves: dict | None = None                     # sampled inner-edge curves (px) for overlays
    fragments: list | None = None                  # in-muscle fragments with per-fragment values

    def as_row(self, prefix: str = "") -> dict:
        r = {f"{prefix}pa_deg": self.pa_deg, f"{prefix}fl_mm": self.fl_mm, f"{prefix}mt_mm": self.mt_mm,
             f"{prefix}confidence": self.confidence, f"{prefix}flags": ";".join(self.flags)}
        r.update({f"{prefix}{k}": v for k, v in self.variants.items()})
        r.update({f"{prefix}d_{k}": v for k, v in self.diag.items() if np.isscalar(v) or v is None})
        return r


def _nan_result(reason: str, diag: dict | None = None) -> GeoResult:
    d = dict(diag or {})
    d["fail"] = reason
    return GeoResult(float("nan"), float("nan"), float("nan"), {}, d, [reason], 0.0)


def _resolve(name: str, aliases: dict, valid: tuple) -> str:
    name = aliases.get(name, name)
    if name not in valid:
        raise ValueError(f"unknown method {name!r}; valid: {valid}")
    return name


# --------------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------------
def measure(apo_prob: np.ndarray, fasc_prob: np.ndarray, image: np.ndarray | None = None,
            px_per_cm: float | None = None, cfg: GeoConfig = GeoConfig(),
            orient: tuple | None = None, apo_upper: np.ndarray | None = None,
            apo_lower: np.ndarray | None = None, keep_details: bool = False,
            apo_fallback: np.ndarray | None = None) -> GeoResult:
    """Measure PA (deg), FL (mm) and MT (mm) of the superficial muscle.

    apo_prob, fasc_prob : HxW probability maps (bool, uint8 0-255 or float 0-1) at the native
        image resolution.
    image : HxW grayscale (needed for the classical "structure" PA and structure streamlines).
    px_per_cm : vertical scale; None -> lengths are returned in px (diag["units"] = "px").
    orient : optional (cos2, sin2) or (seg, cos2, sin2) maps (uint8 (v+1)/2*255 or float [-1, 1]).
    apo_upper / apo_lower : optional edge maps (prob of an upper / lower aponeurosis edge).
    apo_fallback : optional second aponeurosis map; its bands are added only when the primary map
        does not yield a valid pair (flag "apo_fallback").
    """
    pa_m = _resolve(cfg.pa_method, PA_ALIASES, PA_METHODS)
    fl_m = _resolve(cfg.fl_method, FL_ALIASES, FL_METHODS)
    fl_primary = _resolve(cfg.fl_primary, FL_ALIASES, ("straight", "central3", "streamline"))
    A = as_prob(apo_prob)
    F = as_prob(fasc_prob)
    H, W = A.shape
    if F.shape != (H, W):
        raise ValueError(f"fascicle map {F.shape} != apo map {(H, W)}")
    mm = 10.0 / px_per_cm if px_per_cm else 1.0
    diag: dict = dict(H=H, W=W, units="mm" if px_per_cm else "px")

    # ---------------- aponeuroses
    img0 = as_gray(image)
    Wref = float(W)
    if cfg.apo_span_ref == "viewport" and img0 is not None and img0.shape == (H, W):
        vx0, vx1 = estimate_viewport_x(img0)
        Wref = float(max(vx1 - vx0, 0.1 * W))
        diag["viewport_w"] = Wref
    bands = extract_bands(A, cfg, Wref)
    diag["n_bands"] = len(bands)

    def _pick(bands_):
        Fp = oblique_fascicle_map(F, bands_, cfg) if (cfg.pair_oblique_only and cfg.pair_mode == "superficial"
                                                      and len(bands_) >= 2) else F
        return select_pair(bands_, Fp, H, int(Wref), cfg)

    pr = _pick(bands)
    if pr is None and apo_fallback is not None:
        # add aponeuroses that only the fallback map finds (e.g. a second apo model)
        fb = extract_bands(as_prob(apo_fallback), cfg, Wref)
        add = [b for b in fb if all(abs(b.yc - a.yc) > max(a.thickness, b.thickness, 0.03 * H) for a in bands)]
        if add:
            bands = sorted(bands + add, key=lambda b: b.yc)
            diag["apo_fallback_bands"] = len(add)
            pr = _pick(bands)
    if pr is None:
        return _nan_result("no_aponeurosis_pair", diag)
    sup_b, deep_b, pinfo = pr
    diag.update({k: v for k, v in pinfo.items() if k != "pair"})
    diag["pair"] = "%d,%d" % pinfo["pair"]
    sup_y, deep_y = sup_b.bot, deep_b.top
    if cfg.edge_source == "edgemap":
        if apo_lower is not None:
            sup_y = _edge_from_map(sup_b, as_prob(apo_lower), "bot")
        if apo_upper is not None:
            deep_y = _edge_from_map(deep_b, as_prob(apo_upper), "top")
    elif cfg.edge_source == "centroid":  # apo_v2 edge channels (see edge_centroid)
        if apo_lower is not None:
            sup_y, used = edge_centroid(sup_b, as_prob(apo_lower), "bot", cfg.edge_search, min_mass=cfg.edge_min_mass)
            diag["edge_cov_sup"] = float(used.mean()) if used.size else 0.0
        if apo_upper is not None:
            deep_y, used = edge_centroid(deep_b, as_prob(apo_upper), "top", cfg.edge_search, min_mass=cfg.edge_min_mass)
            diag["edge_cov_deep"] = float(used.mean()) if used.size else 0.0
    sup = EdgeCurve(sup_b.x + 0.5, sup_y, cfg.edge_fit, cfg.edge_extrap, cfg.savgol_frac, cfg.edge_end_frac)
    deep = EdgeCurve(deep_b.x + 0.5, deep_y, cfg.edge_fit, cfg.edge_extrap, cfg.savgol_frac, cfg.edge_end_frac)
    mt_fit = cfg.mt_fit or cfg.edge_fit
    if mt_fit != cfg.edge_fit:
        sup_mt = EdgeCurve(sup_b.x + 0.5, sup_y, mt_fit, cfg.edge_extrap, cfg.savgol_frac, cfg.edge_end_frac)
        deep_mt = EdgeCurve(deep_b.x + 0.5, deep_y, mt_fit, cfg.edge_extrap, cfg.savgol_frac, cfg.edge_end_frac)
    else:
        sup_mt, deep_mt = sup, deep
    s_lo, s_hi = max(sup.x0, deep.x0), min(sup.x1, deep.x1)
    if s_hi <= s_lo:
        s_lo, s_hi = min(sup.x0, deep.x0), max(sup.x1, deep.x1)
        diag["no_overlap"] = True
    diag.update(shared_lo=s_lo, shared_hi=s_hi, shared_frac=(s_hi - s_lo) / W,
                sup_span=sup_b.span / W, deep_span=deep_b.span / W,
                sup_thick_px=sup_b.thickness, deep_thick_px=deep_b.thickness)
    deep_s, deep_i = _line_fit(deep.xd, deep.yd)
    sup_s, _ = _line_fit(sup.xd, sup.yd)
    deep_ang = float(np.degrees(np.arctan(deep_s)))
    diag.update(deep_angle=deep_ang, sup_angle=float(np.degrees(np.arctan(sup_s))))
    diag["apo_converge"] = diag["sup_angle"] - deep_ang

    # ---------------- MT
    def mt_at(fracs, ref=cfg.mt_xref):
        if ref == "image":
            lo, hi = 0.0, float(W)
        elif ref == "union":
            lo, hi = min(sup.x0, deep.x0), max(sup.x1, deep.x1)
        else:
            lo, hi = s_lo, s_hi
        xs = lo + (hi - lo) * np.asarray(fracs, float)
        if cfg.mt_source == "raw":
            d = [deep_mt.raw_at(x) - sup_mt.raw_at(x) for x in xs]
        else:
            d = deep_mt(xs) - sup_mt(xs)
        return float(np.mean(d))

    mt_px = mt_at(cfg.mt_xpos)
    variants = dict(mt_q=mt_at((0.25, 0.5, 0.75)) * mm, mt_e=mt_at((0.1, 0.5, 0.9)) * mm,
                    mt_c=mt_at((0.5,)) * mm, mt_all=mt_at(np.linspace(0, 1, 41)) * mm,
                    mt_img=mt_at((0.25, 0.5, 0.75), "image") * mm)
    if not np.isfinite(mt_px) or mt_px <= 0:
        return _nan_result("bad_mt", diag)
    diag["mt_px"] = mt_px

    # ---------------- fragments
    frags_all = extract_fragments(F, cfg)
    xgrid = np.linspace(-3 * W, 4 * W, 6001)
    ref_ang = deep_ang if cfg.pa_reference == "deep" else 0.0
    frs = []
    for fr in frags_all:
        if not (sup(fr["cx"]) - cfg.frag_inside_margin <= fr["cy"] <= deep(fr["cx"]) + cfg.frag_inside_margin):
            continue
        rel = float(wrap90(fr["angle"] - ref_ang))
        q = dict(fr, rel=rel, pa=abs(rel))
        pd_ = _line_curve_x(fr, deep, xgrid, "deep")
        ps_ = _line_curve_x(fr, sup, xgrid, "sup")
        q["p_deep"], q["p_sup"] = pd_, ps_
        if pd_ is not None:
            tang = float(np.degrees(np.arctan(deep.slope(pd_[0])))) if cfg.pa_reference == "deep" else 0.0
            q["pa_local"] = abs(float(wrap90(fr["angle"] - tang)))
            q["ins_inside"] = deep.x0 <= pd_[0] <= deep.x1
            # gap between the fragment's lower end and its deep insertion
            ends = np.array([fr["p0"], fr["p1"]])
            q["gap_deep"] = float(np.min(np.hypot(ends[:, 0] - pd_[0], ends[:, 1] - pd_[1])))
        else:
            q["pa_local"], q["ins_inside"], q["gap_deep"] = float("nan"), False, float("nan")
        if pd_ is not None and ps_ is not None:
            q["fl_px"] = float(np.hypot(pd_[0] - ps_[0], pd_[1] - ps_[1]))
            q["extrap"] = float(max(0.0, 1.0 - fr["length"] / max(q["fl_px"], 1e-6)))
            q["mid_x"] = 0.5 * (pd_[0] + ps_[0])
        else:
            q["fl_px"], q["extrap"], q["mid_x"] = float("nan"), float("nan"), float("nan")
        frs.append(q)
    diag["n_frag_all"] = len(frags_all)
    diag["n_frag_inside"] = len(frs)
    if cfg.frag_sign_consistency and frs:
        wpos = sum(q["length"] for q in frs if q["rel"] > 0)
        wneg = sum(q["length"] for q in frs if q["rel"] < 0)
        dom = 1.0 if wpos >= wneg else -1.0
        diag["sign_purity"] = max(wpos, wneg) / max(wpos + wneg, 1e-9)
        if diag["sign_purity"] >= 0.6:
            frs = [q for q in frs if np.sign(q["rel"]) == dom or q["pa"] < cfg.pa_min]
    ok = [q for q in frs if cfg.pa_min <= q["pa"] <= cfg.pa_max]
    valid = ok if ok else frs  # v1 behaviour: keep all when none is in range
    diag["n_frag"] = len(valid)
    diag["frag_sign"] = float(np.mean([q["angle"] > 0 for q in frags_all])) if frags_all else float("nan")
    diag["pa_horizontal"] = _nanmedian([abs(q["angle"]) for q in valid]) if valid else float("nan")

    # ---------------- PA variants
    pa = {k: float("nan") for k in PA_METHODS}
    if valid:
        pa["global"] = float(np.median([q["pa"] for q in valid]))
        pa["local"] = _nanmedian([q["pa_local"] for q in valid])
        span = s_hi - s_lo
        c_lo = s_lo + 0.5 * (1 - cfg.central_frac) * span
        c_hi = s_hi - 0.5 * (1 - cfg.central_frac) * span
        cand = [q for q in valid if q["p_deep"] is not None and q["ins_inside"]
                and c_lo <= q["p_deep"][0] <= c_hi and np.isfinite(q["pa_local"])]
        if len(cand) < cfg.central_n:
            cand = [q for q in valid if q["p_deep"] is not None and np.isfinite(q["pa_local"])]
        if cand:
            rel_ = sorted(cand, key=lambda q: -(q["length"] * q["prob"] / (q["length"] + q["gap_deep"])))
            pa["central3"] = _robust_mean([q["pa_local"] for q in rel_[:cfg.central_n]])
        diag["pa_iqr"] = _iqr([q["pa"] for q in valid])
    else:
        diag["pa_iqr"] = float("nan")

    learned = None
    if orient is not None:
        if len(orient) == 3:
            learned = field_from_orient(orient[1], orient[2], orient[0])
        else:
            learned = field_from_orient(orient[0], orient[1])
        lo_, hi_ = s_lo + cfg.st_xtrim * (s_hi - s_lo), s_hi - cfg.st_xtrim * (s_hi - s_lo)
        y_from = lambda x: deep(x) - cfg.field_strip[1] * mt_px
        y_to = lambda x: deep(x) - cfg.field_strip[0] * mt_px
        rel_d, w_d = _region_angles(learned, sup, deep, lo_, hi_, y_from, y_to, cfg.pa_reference)
        pa["field"], _ = _angle_stat(rel_d, w_d, "mean", cfg.pa_min, cfg.pa_max)

    img = as_gray(image)
    sfield = None
    if cfg.compute_structure and img is not None:
        if img.shape != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        sc = (px_per_cm / cfg.st_ref_ppc) if px_per_cm else 1.0
        lo_, hi_ = s_lo + cfg.st_xtrim * (s_hi - s_lo), s_hi - cfg.st_xtrim * (s_hi - s_lo)
        frag_sign = None
        if valid and cfg.st_use_frag_sign:
            sw = sum(q["length"] * np.sign(q["rel"]) for q in valid)
            frag_sign = 1.0 if sw >= 0 else -1.0
        need_field = cfg.st_kind in ("tensor", "hessian") or (
            cfg.compute_streamline and cfg.sl_field in ("structure", "auto"))
        if need_field:
            kind = cfg.st_kind if cfg.st_kind in ("tensor", "hessian") else "tensor"
            sfield = structure_field(img, cfg.st_sigma_d * sc, cfg.st_sigma_t * sc, kind)
        if cfg.st_kind == "radon":
            line_px = (cfg.st_line_mm * px_per_cm / 10.0) if px_per_cm else cfg.st_line_frac * mt_px
            hp_px = (cfg.st_hp_mm * px_per_cm / 10.0) if px_per_cm else 0.08 * mt_px
            dr = directional_pa(img, sup, deep, lo_, hi_, mt_px, max(9.0, line_px), cfg.st_margin,
                                max(2.0, hp_px), cfg.pa_min, cfg.pa_max, cfg.pa_reference, frag_sign,
                                mask_outside=cfg.st_mask_outside, symmetric=cfg.st_symmetric)
            pa["structure"], diag["st_sign"] = dr["pa"], dr["sign"]
            diag["st_prominence"] = dr["prominence"]
            diag["st_edge_hit"] = dr.get("edge_hit", False)
            if cfg.st_reject_edge and diag["st_edge_hit"]:
                pa["structure"] = float("nan")
        else:
            if cfg.st_region == "deep_strip":
                y_from = lambda x: deep(x) - cfg.field_strip[1] * mt_px
                y_to = lambda x: deep(x) - max(cfg.field_strip[0], cfg.st_margin) * mt_px
            else:
                y_from = lambda x: sup(x) + cfg.st_margin * mt_px
                y_to = lambda x: deep(x) - cfg.st_margin * mt_px
            rel_s, w_s = _region_angles(sfield, sup, deep, lo_, hi_, y_from, y_to, cfg.pa_reference)
            pa["structure"], diag["st_sign"] = _angle_stat(rel_s, w_s, cfg.st_agg, cfg.pa_min, cfg.pa_max, frag_sign)
            diag["st_npix"] = int(rel_s.size)
    fus = [pa[PA_ALIASES.get(k, k)] for k in cfg.pa_fusion]
    fus = [v for v in fus if np.isfinite(v)]
    pa["fusion"] = float(np.mean(fus)) if fus else float("nan")
    for k in PA_METHODS:
        variants[f"pa_{k}"] = pa[k]
    pa_final = pa[pa_m]
    pa_used = pa_m
    if not np.isfinite(pa_final):
        for k in cfg.pa_fallback:
            k = PA_ALIASES.get(k, k)
            if np.isfinite(pa.get(k, np.nan)):
                pa_final, pa_used = pa[k], k
                break
    diag["pa_used"] = pa_used

    # ---------------- FL variants (px)
    fl = {k: float("nan") for k in FL_METHODS}
    fl_frs = [q for q in valid if np.isfinite(q["fl_px"])]
    if fl_frs:
        fl["straight"] = float(np.median([q["fl_px"] for q in fl_frs]))
        span = s_hi - s_lo
        c_lo = s_lo + 0.5 * (1 - cfg.central_frac) * span
        c_hi = s_hi - 0.5 * (1 - cfg.central_frac) * span
        cen = [q for q in fl_frs if c_lo <= q["mid_x"] <= c_hi]
        if len(cen) < cfg.central_n:
            cen = fl_frs
        cen = sorted(cen, key=lambda q: q["extrap"])[:cfg.central_n]
        fl["central3"] = float(np.mean([q["fl_px"] for q in cen]))
        diag["extrap_med"] = float(np.median([q["extrap"] for q in fl_frs]))
        diag["extrap_c3"] = float(np.mean([q["extrap"] for q in cen]))
        diag["fl_iqr_rel"] = _iqr([q["fl_px"] for q in fl_frs]) / max(fl["straight"], 1e-6)
        diag["n_fl"] = len(fl_frs)
    else:
        diag.update(extrap_med=float("nan"), extrap_c3=float("nan"), fl_iqr_rel=float("nan"), n_fl=0)

    if cfg.compute_streamline:
        fld = None
        order = {"learned": [learned], "fragments": [None], "structure": [sfield],
                 "auto": [learned, None, sfield]}[cfg.sl_field]
        for cand_f in order:
            if cand_f is None and (cfg.sl_field in ("fragments", "auto")):
                cand_f = field_from_fragments(valid, H, W, max(8.0, cfg.frag_field_sigma * mt_px))
            if cand_f is not None:
                fld = cand_f
                break
        if fld is not None:
            sl, chords, ok_n = [], [], 0
            lo_, hi_ = s_lo + cfg.sl_seed_range[0] * (s_hi - s_lo), s_lo + cfg.sl_seed_range[1] * (s_hi - s_lo)
            for xs0 in np.linspace(lo_, hi_, cfg.sl_seeds):
                ys0 = 0.5 * (float(sup(xs0)) + float(deep(xs0)))
                c2, s2, w = fld.sample(xs0, ys0)
                if w <= 1e-3:
                    continue
                th = 0.5 * math.atan2(s2, c2)
                d = np.array([math.cos(th), math.sin(th)])
                if d[1] < 0:
                    d = -d  # downwards first
                down = _trace(fld, (xs0, ys0), d, deep, "deep", cfg.sl_step, 6.0 * W)
                up = _trace(fld, (xs0, ys0), -d, sup, "sup", cfg.sl_step, 6.0 * W)
                if down is None or up is None:
                    continue
                ok_n += 1
                sl.append(down[0] + up[0])
                chords.append(math.hypot(down[1][0] - up[1][0], down[1][1] - up[1][1]))
            if sl:
                fl["streamline"] = float(np.median(sl))
                diag["sl_curvature"] = float(np.median(np.array(sl) / np.maximum(chords, 1e-6)))
            diag["sl_field"] = fld.name
            diag["sl_n"] = ok_n
    pa_for_tri = pa_final
    if np.isfinite(pa_for_tri) and pa_for_tri > 0.5:
        fl["triangle"] = cfg.fl_k * mt_px / math.sin(math.radians(pa_for_tri))
    # constructed fascicle: straight line at the consensus PA (vs the local deep tangent) whose
    # midpoint sits at the central x positions, measured between the real inner-edge curves
    if np.isfinite(pa_for_tri) and pa_for_tri > 0.5:
        if valid:
            sgn = 1.0 if sum(q["length"] * np.sign(q["rel"]) for q in valid) >= 0 else -1.0
        else:
            sgn = diag.get("st_sign", 1.0)
            sgn = 1.0 if not np.isfinite(sgn) else float(sgn)
        cons = [constructed_fl(sup, deep, s_lo + f * (s_hi - s_lo), pa_for_tri, sgn, mt_px,
                               cfg.pa_reference, xgrid) for f in cfg.fl_constructed_x]
        cons = [c for c in cons if np.isfinite(c)]
        fl["constructed"] = float(np.mean(cons)) if cons else float("nan")
    prim = fl[fl_primary]
    tri = fl["triangle"]
    if np.isfinite(prim) and np.isfinite(tri):
        dis = abs(prim - tri) / tri
        diag["fl_tri_disagree"] = dis
        if dis > cfg.fl_gate_tol:
            fl["gated"] = tri if cfg.fl_gate_mode == "fallback" else cfg.fl_blend_w * prim + (1 - cfg.fl_blend_w) * tri
        else:
            fl["gated"] = prim
    else:
        fl["gated"] = prim if np.isfinite(prim) else tri
        diag["fl_tri_disagree"] = float("nan")
    ens = [fl[FL_ALIASES.get(k, k)] for k in cfg.fl_ensemble]
    ens = [v for v in ens if np.isfinite(v)]
    if ens:
        fl["ensemble"] = float(np.median(ens) if cfg.fl_ensemble_agg == "median" else np.mean(ens))
        diag["fl_ens_spread"] = (max(ens) - min(ens)) / max(np.mean(ens), 1e-6)
    else:
        diag["fl_ens_spread"] = float("nan")
    for k in FL_METHODS:
        variants[f"fl_{k}"] = fl[k] * mm
    fl_final = fl[fl_m]
    if not np.isfinite(fl_final):
        for k in (fl_primary, "straight", "triangle"):
            if np.isfinite(fl[k]):
                fl_final = fl[k]
                diag["fl_used"] = k
                break
    else:
        diag["fl_used"] = fl_m

    # ---------------- flags / confidence
    flags, pen = [], 0.0
    if diag["n_frag"] < 5:
        flags.append("few_fragments"); pen += 0.5
    if np.isfinite(diag["extrap_med"]) and diag["extrap_med"] > cfg.flag_extrap:
        flags.append("high_extrapolation"); pen += 0.3
    if np.isfinite(diag["pa_iqr"]) and diag["pa_iqr"] > 8:
        flags.append("pa_spread"); pen += 0.3
    if np.isfinite(diag["fl_iqr_rel"]) and diag["fl_iqr_rel"] > 0.35:
        flags.append("fl_spread"); pen += 0.3
    if np.isfinite(diag["fl_tri_disagree"]) and diag["fl_tri_disagree"] > cfg.fl_gate_tol:
        flags.append("fl_triangle_disagree"); pen += 0.3
    if np.isfinite(pa["structure"]) and np.isfinite(pa["global"]) and abs(pa["structure"] - pa["global"]) > 5:
        flags.append("pa_classical_disagree"); pen += 0.3
    if len(bands) > 2:
        flags.append("multi_band"); pen += 0.1
    if pinfo.get("pair_fallback"):
        flags.append("pair_fallback"); pen += 0.3
    if diag["shared_frac"] < 0.4:
        flags.append("short_apo_overlap"); pen += 0.3
    if abs(diag["apo_converge"]) > 15:
        flags.append("apo_converge"); pen += 0.2
    if pa_used != pa_m:
        flags.append(f"pa_fallback_{pa_used}"); pen += 0.3
    if diag.get("apo_fallback_bands"):
        flags.append("apo_fallback"); pen += 0.3
    if np.isfinite(pa_final) and not (cfg.flag_pa_range[0] <= pa_final <= cfg.flag_pa_range[1]):
        flags.append("pa_implausible"); pen += 0.5
    conf = float(math.exp(-pen))

    pa_out, fl_out, mt_out = pa_final, fl_final * mm, mt_px * mm
    if cfg.clip:
        pa_out = float(np.clip(pa_out, *cfg.clip_pa)) if np.isfinite(pa_out) else pa_out
        fl_out = float(np.clip(fl_out, *cfg.clip_fl)) if np.isfinite(fl_out) else fl_out
        mt_out = float(np.clip(mt_out, *cfg.clip_mt))
    res = GeoResult(float(pa_out), float(fl_out), float(mt_out), variants, diag, flags, conf)
    if keep_details:
        xs = np.linspace(min(sup.x0, deep.x0), max(sup.x1, deep.x1), 200)
        res.curves = dict(x=xs, sup=sup(xs), deep=deep(xs), sup_x0=sup.x0, sup_x1=sup.x1,
                          deep_x0=deep.x0, deep_x1=deep.x1,
                          mt_x=[s_lo + (s_hi - s_lo) * f for f in cfg.mt_xpos])
        # live curve objects for add-on measurements (field_architecture); not serialised by as_row
        res.curves.update(sup_curve=sup, deep_curve=deep, s_lo=s_lo, s_hi=s_hi, mt_px=mt_px, mm=mm)
        res.fragments = [{k: v for k, v in q.items()} for q in valid]
    return res


# ============================================================================================
# apo_v2 aponeurosis maps (band + upper / lower edge channels)  -- apo2 agent, additive.
# edge_source="centroid" refines the inner edges of the selected pair with the edge channels;
# measure_apo() chains aponeurosis sources (e.g. apo_v2 -> ours_apo384 -> dlt_apo) and adds MT
# variants, an MT confidence and the inner-edge curve parameters (all scalars, so as_row keeps them).
# ============================================================================================
def edge_centroid(band: Band, emap: np.ndarray, which: str, search: float = 0.0, floor: float = 0.1,
                  min_mass: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel band edge from an apo_v2 edge channel (``upper`` for a band's top edge, ``lower`` for
    its bottom edge): probability-weighted row centre per column within +-search px of the mask edge.

    The edge targets are 2-px lines straddling the boundary (upper = rows ymin-1, ymin; lower = rows
    ymax, ymax+1), so the centroid in row-index units is the boundary minus 0.5 and the boundary in
    the geometry convention (pixel k covers [k, k+1)) is centroid + 0.5. Probabilities below ``floor``
    are ignored; columns with less than ``min_mass`` edge probability keep the mask edge.
    Returns (edge y per band column, bool mask of the refined columns).
    """
    H = emap.shape[0]
    base = np.asarray(band.top if which == "top" else band.bot, float)
    if search <= 0:
        search = max(3.0, H / 128.0)
    r = int(math.ceil(search))
    xs = band.x.astype(int)
    c = np.rint(base - 0.5).astype(int)                    # row index of the edge-line centre
    rows = c[:, None] + np.arange(-r, r + 1)[None, :]
    inside = (rows >= 0) & (rows < H)
    rc = np.clip(rows, 0, H - 1)
    p = emap[rc, xs[:, None]].astype(np.float64)
    p = np.where(inside & (p >= floor), p, 0.0)
    mass = p.sum(axis=1)
    used = mass >= min_mass
    cen = (p * rows).sum(axis=1) / np.maximum(mass, 1e-9)
    out = np.where(used, cen + 0.5, base)
    return out, used


def recal_threshold(p: np.ndarray, t_from: float, t_to: float) -> np.ndarray:
    """Monotone piecewise-linear map sending probability t_from -> t_to (0 -> 0, 1 -> 1), so that a
    fallback map thresholded at the primary threshold behaves as at its own operating threshold."""
    p = as_prob(p)
    if abs(t_from - t_to) < 1e-9:
        return p
    return np.where(p < t_from, t_to * p / t_from,
                    t_to + (1.0 - t_to) * (p - t_from) / max(1.0 - t_from, 1e-6)).astype(np.float32)


def apo_v2_config(**overrides) -> GeoConfig:
    """recommended_config() for apo_v2 maps: band threshold 0.5 (sigmoid output of a BCE+Dice model)
    and inner edges from the edge channels (edge_source="centroid")."""
    base = recommended_config(apo_thr=0.5, edge_source="centroid")
    return base.with_(**overrides) if overrides else base


def apo_mt_variants(res: GeoResult, xpos: tuple[float, ...] = (0.25, 0.5, 0.75)) -> dict:
    """MT variants (mm) and inner-edge curve parameters (px) from a keep_details=True result.

    mt_lin  : straight-line fits of the raw inner edges at ``xpos`` of the shared x-range
    mt_quad : quadratic fits at the same positions
    mt_raw  : median of the raw edge gap in +-max(2, 2 % of the shared range) columns (train.py inner_mt)
    mt_lin_c / mt_lin_e : centre only / 10-50-90 %
    plus per-curve poly1 / poly2 coefficients, data ranges, fit residual RMS and the spread of the
    three MT samples (mt_lin_sd)."""
    cv = res.curves or {}
    if "sup_curve" not in cv:
        return {}
    sup, deep = cv["sup_curve"], cv["deep_curve"]
    s_lo, s_hi, mm = float(cv["s_lo"]), float(cv["s_hi"]), float(cv["mm"])
    out = {}
    fits = {}
    for nm, c in (("sup", sup), ("deep", deep)):
        x, y = np.asarray(c.xd, float), np.asarray(c.yd, float)
        p1 = np.polyfit(x, y, 1) if len(x) >= 2 else np.array([0.0, float(np.mean(y))])
        p2 = np.polyfit(x, y, 2) if len(x) >= 3 else np.concatenate([[0.0], p1])
        fits[nm] = (x, y, p1, p2)
        out.update({f"{nm}_p1_s": float(p1[0]), f"{nm}_p1_b": float(p1[1]), f"{nm}_p2_a": float(p2[0]),
                    f"{nm}_p2_b": float(p2[1]), f"{nm}_p2_c": float(p2[2]), f"{nm}_x0": float(x.min()),
                    f"{nm}_x1": float(x.max()), f"{nm}_n": int(len(x)),
                    f"{nm}_res_px": float(np.sqrt(np.mean((y - np.polyval(p1, x)) ** 2)))})
    out.update(s_lo=s_lo, s_hi=s_hi, px_mm=mm)

    def at(fr, deg):
        xs_ = s_lo + (s_hi - s_lo) * np.asarray(fr, float)
        k = 2 if deg == 1 else 3
        return np.polyval(fits["deep"][k], xs_) - np.polyval(fits["sup"][k], xs_)

    d1 = at(xpos, 1)
    out["mt_lin"] = float(np.mean(d1)) * mm
    out["mt_lin_sd"] = float(np.std(d1)) * mm
    out["mt_quad"] = float(np.mean(at(xpos, 2))) * mm
    out["mt_lin_c"] = float(np.mean(at((0.5,), 1))) * mm
    out["mt_lin_e"] = float(np.mean(at((0.1, 0.5, 0.9), 1))) * mm
    hw = max(2.0, 0.02 * (s_hi - s_lo))
    g = []
    for f in xpos:
        xc = s_lo + (s_hi - s_lo) * f
        ss = np.abs(fits["sup"][0] - xc) <= hw
        dd = np.abs(fits["deep"][0] - xc) <= hw
        if ss.any() and dd.any():
            g.append(float(np.median(fits["deep"][1][dd])) - float(np.median(fits["sup"][1][ss])))
    out["mt_raw"] = float(np.mean(g)) * mm if g else float("nan")
    return out


def mt_confidence(res: GeoResult) -> float:
    """Untuned MT-only confidence from the aponeurosis diagnostics (1 = clean two-band pair)."""
    if not np.isfinite(res.mt_mm):
        return 0.0
    d, v, pen = res.diag, res.variants, 0.0
    if d.get("pair_fallback"):
        pen += 0.3
    if d.get("apo_fallback_bands"):
        pen += 0.4
    if (d.get("n_bands") or 0) > 2:
        pen += 0.1
    if (d.get("shared_frac") or 1.0) < 0.4:
        pen += 0.3
    if abs(d.get("apo_converge") or 0.0) > 15:
        pen += 0.2
    mt_px = d.get("mt_px") or np.nan
    for nm in ("sup", "deep"):
        rr = v.get(f"{nm}_res_px")
        if rr is not None and np.isfinite(mt_px) and rr > 0.04 * mt_px:
            pen += 0.2
    sd = v.get("mt_lin_sd")
    if sd is not None and np.isfinite(sd) and res.mt_mm > 0 and sd > 0.15 * res.mt_mm:
        pen += 0.2
    if not (8.0 <= res.mt_mm <= 45.0):
        pen += 0.5
    return float(math.exp(-pen))


def measure_apo(apo_prob: np.ndarray, fasc_prob: np.ndarray, image: np.ndarray | None = None,
                px_per_cm: float | None = None, cfg: GeoConfig | None = None,
                apo_upper: np.ndarray | None = None, apo_lower: np.ndarray | None = None,
                fallbacks: tuple | list = (), keep_details: bool = False, **kw) -> GeoResult:
    """measure() with an aponeurosis-source chain and MT add-ons.

    apo_prob (+ apo_upper / apo_lower edge channels when cfg.edge_source == "centroid") is the
    primary map. ``fallbacks`` = [(name, prob_map, own_threshold), ...]: when the primary map yields
    no valid pair, the bands of the next fallback map are added (measure(apo_fallback=...), the map
    recalibrated so that its own threshold maps onto cfg.apo_thr) until a pair is found.
    Adds diag["apo_source"] (primary | fallback name | none), variants mt_lin / mt_quad / mt_raw /
    curve parameters (apo_mt_variants) and variants["mt_conf"] (mt_confidence).
    """
    cfg = cfg or apo_v2_config()
    res = measure(apo_prob, fasc_prob, image, px_per_cm, cfg, apo_upper=apo_upper, apo_lower=apo_lower,
                  keep_details=True, **kw)
    src = "primary"
    nb_primary = res.diag.get("n_bands")  # bands of the primary map alone (before any fallback)
    if res.diag.get("fail") == "no_aponeurosis_pair":
        src = "none"
        for name, fb, thr in fallbacks:
            if fb is None:
                continue
            r2 = measure(apo_prob, fasc_prob, image, px_per_cm, cfg, apo_upper=apo_upper, apo_lower=apo_lower,
                         keep_details=True, apo_fallback=recal_threshold(fb, thr, cfg.apo_thr), **kw)
            if np.isfinite(r2.mt_mm):
                res, src = r2, name
                break
    res.diag["apo_source"] = src
    res.diag["n_bands_primary"] = nb_primary
    res.variants.update(apo_mt_variants(res))
    res.variants["mt_conf"] = mt_confidence(res)
    if not keep_details:
        res.curves, res.fragments = None, None
    return res


# --------------------------------------------------------------------------------------------
# MASKBANK loader (lazy npz access)
# --------------------------------------------------------------------------------------------
class MaskBank:
    """Reader for <root>/<dataset>__<model>.npz probability maps (uint8 prob*255, native res).

    Orientation heads are stored as separate files with suffixes _seg / _cos2 / _sin2.
    np.load on an npz is lazy: only the requested key is decompressed.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not any(self.root.glob("*__*.npz")) and any((self.root / "maskbank").glob("*__*.npz")):
            self.root = self.root / "maskbank"  # artifacts/maskbank/maskbank layout
        self._open: dict[str, np.lib.npyio.NpzFile] = {}
        self.manifest = None
        mf = self.root / "manifest.csv"
        if mf.exists():
            import pandas as pd
            self.manifest = pd.read_csv(mf)

    def models(self, dataset: str) -> list[str]:
        return sorted(p.name[len(dataset) + 2:-4] for p in self.root.glob(f"{dataset}__*.npz"))

    def _file(self, dataset: str, model: str):
        key = f"{dataset}__{model}"
        if key not in self._open:
            path = self.root / f"{key}.npz"
            if not path.exists():
                return None
            self._open[key] = np.load(path, allow_pickle=False)
        return self._open[key]

    def keys(self, dataset: str, model: str) -> list[str]:
        f = self._file(dataset, model)
        return [] if f is None else list(f.files)

    def raw(self, dataset: str, model: str, image_id: str) -> np.ndarray | None:
        f = self._file(dataset, model)
        if f is None:
            return None
        stem = Path(image_id).stem
        for k in (stem, image_id):
            if k in f.files:
                return f[k]
        return None

    def prob(self, dataset: str, model: str, image_id: str) -> np.ndarray | None:
        a = self.raw(dataset, model, image_id)
        return None if a is None else as_prob(a)

    def orient(self, dataset: str, model: str, image_id: str) -> tuple | None:
        """(seg, cos2, sin2) decoded, or None when the head is not in the bank."""
        c = self.raw(dataset, f"{model}_cos2", image_id)
        s = self.raw(dataset, f"{model}_sin2", image_id)
        if c is None or s is None:
            return None
        g = self.prob(dataset, f"{model}_seg", image_id)
        return (g, decode_orient(c), decode_orient(s)) if g is not None else (decode_orient(c), decode_orient(s))

    def close(self):
        for f in self._open.values():
            f.close()
        self._open.clear()


def measure_from_bank(bank: MaskBank, dataset: str, image_id: str, apo_model: str, fasc_model: str,
                      image: np.ndarray | None = None, px_per_cm: float | None = None,
                      cfg: GeoConfig = GeoConfig(), orient_model: str | None = None,
                      edge_model: str | None = None, **kw) -> GeoResult:
    A = bank.prob(dataset, apo_model, image_id)
    F = bank.prob(dataset, fasc_model, image_id)
    if A is None or F is None:
        return _nan_result("missing_maps")
    orient = bank.orient(dataset, orient_model, image_id) if orient_model else None
    up = lo = None
    if edge_model:
        up = bank.prob(dataset, f"{edge_model}_upper", image_id)
        lo = bank.prob(dataset, f"{edge_model}_lower", image_id)
    return measure(A, F, image, px_per_cm, cfg, orient=orient, apo_upper=up, apo_lower=lo, **kw)


def config_dict(cfg: GeoConfig) -> dict:
    return asdict(cfg)


def recommended_config(**overrides) -> GeoConfig:
    """Recommended v2 default (validated in scripts/eval_geometry_v2.py).

    superficial pair (+crossing / oblique tests), quadratic inner edges extrapolated beyond the band
    ends by a straight line through the last 30 % of the band ("end_linear"; integrate step, our
    maps: static 0.141 vs 0.146 for "tangent", clean10 0.099 vs 0.103, 2-rater 0.213 vs 0.217,
    dynamic FL 2.05 vs 2.11 mm, VL equal -- the end tangent of the quadratic is noisy),
    MT from straight inner-edge fits at 25/50/75 %, PA = fusion(global fragments, classical
    directional PA), FL = ensemble(straight, central3, constructed), band span measured against
    the detected ultrasound viewport (narrow Lumify FOV); a classical PA whose maximum sits on the
    search boundary is discarded (st_reject_edge; never happens on the test set, 11/249 on VL).
    For our U-Net++ maps use apo_thr=0.8
    (the model's calibrated threshold) and fasc_thr=0.95; pass a second apo map (e.g. DL_Track)
    as ``apo_fallback`` to recover images where our apo model misses one aponeurosis.
    """
    base = GeoConfig(pair_mode="superficial", apo_merge=False, apo_thick_clip=3.0, edge_fit="poly2",
                     edge_extrap="end_linear", mt_fit="poly1", frag_sign_consistency=False,
                     pa_method="fusion", fl_method="ensemble", st_kind="radon", apo_thr=0.8, fasc_thr=0.95,
                     apo_span_ref="viewport", st_reject_edge=True)
    return base.with_(**overrides) if overrides else base


# ============================================================================================
# Learned orientation field (fasc_orient_v1: seg + doubled-angle head) -- PA / FL read from the
# field itself.  Additive: measure() results are unchanged; measure_field() = measure(...,
# keep_details=True) + field_architecture(), which adds variants pa_fd_*, pa_sl_*, fl_sl_*,
# fl_cfd_* and diagnostics fd_* / sl_* to the result.
# ============================================================================================
@dataclass(frozen=True)
class FieldConfig:
    seg_pow: float = 1.0                            # pixel weight = |v| * seg ** seg_pow
    strip: tuple[float, float] = (0.04, 0.30)       # deep strip (fraction of MT above the deep inner edge)
    strip_ins: tuple[float, float] = (0.02, 0.12)   # narrow insertion strip
    comp_margin: float = 0.12                       # compartment: drop this fraction of MT next to each apo
    xtrim: float = 0.1                              # trim of the shared x-range on each side
    sub: int = 2                                    # sampling stride (px)
    robust_c: float = 8.0                           # Tukey cut-off (deg) of the robust axial mean
    min_px: int = 30                                # min sampled pixels with weight > 0
    smooth_frac: float = 0.08                       # sigma of the regularised streamline field (fraction of MT)
    min_support: float = 0.15                       # support (fraction of its compartment p90) below which a
                                                    # streamline continues straight (outside FOV / no fascicle)
    n_seeds: int = 7
    seed_range: tuple[float, float] = (0.3, 0.7)
    sl_step: float = 2.0
    ins_frac: float = 0.15                          # insertion angle = chord of the last ins_frac*MT of a streamline
    constructed_x: tuple[float, ...] = (0.4, 0.5, 0.6)


def orient_maps(orient) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """(seg, cos2, sin2) | (cos2, sin2) maps (uint8 MASKBANK encoding or float) -> float32 seg, c2, s2."""
    if len(orient) == 3:
        seg, c2, s2 = orient
        seg = as_prob(seg) if seg is not None else None
    else:
        seg, (c2, s2) = None, orient
    return seg, decode_orient(c2), decode_orient(s2)


def field_theta_weight(seg, c2, s2, seg_pow: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """theta (deg, image coords, (-90, 90]) and pixel weight |v| * seg**p of a doubled-angle field."""
    theta = (0.5 * np.degrees(np.arctan2(s2, c2))).astype(np.float32)
    w = np.clip(np.hypot(c2, s2), 0.0, 1.0).astype(np.float32)
    if seg is not None:
        w = w * (seg if seg_pow == 1.0 else np.power(seg, seg_pow)).astype(np.float32)
    return theta, w


def robust_axial_mean(rel: np.ndarray, w: np.ndarray, c: float = 8.0, iters: int = 6) -> dict:
    """Robust weighted mean of axial angles (deg, mod 180): weighted-histogram mode, then Tukey-
    reweighted doubled-angle mean. Returns mean (signed, [-90, 90)), concentration R of all weight
    about it, inlier weight fraction (|d| < c) and the plain weighted doubled-angle mean."""
    ok = np.isfinite(rel) & (w > 0)
    r, ww = rel[ok].astype(float), w[ok].astype(float)
    out = dict(mean=float("nan"), conc=float("nan"), inl=float("nan"), plain=float("nan"), n=int(r.size),
               wsum=float(ww.sum()))
    if r.size < 5 or ww.sum() <= 0:
        return out
    c2s = np.sum(ww * np.cos(np.radians(2 * r)))
    s2s = np.sum(ww * np.sin(np.radians(2 * r)))
    out["plain"] = float(0.5 * np.degrees(np.arctan2(s2s, c2s)))
    h, e = np.histogram(r, bins=180, range=(-90.0, 90.0), weights=ww)
    k = np.arange(-6, 7)
    ker = np.exp(-0.5 * (k / 2.0) ** 2)
    hs = np.convolve(np.concatenate([h[-6:], h, h[:6]]), ker / ker.sum(), mode="valid")
    m = float(e[int(np.argmax(hs))] + 0.5)
    for _ in range(iters):
        d = wrap90(r - m)
        u = d / c
        tw = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0) * ww
        if tw.sum() <= 0:
            break
        m = float(wrap90(m + 0.5 * np.degrees(np.arctan2(np.sum(tw * np.sin(np.radians(2 * d))),
                                                          np.sum(tw * np.cos(np.radians(2 * d)))))))
    d = wrap90(r - m)
    out["mean"] = m
    out["conc"] = float(np.hypot(np.sum(ww * np.cos(np.radians(2 * d))), np.sum(ww * np.sin(np.radians(2 * d))))
                        / ww.sum())
    out["inl"] = float(ww[np.abs(d) < c].sum() / ww.sum())
    return out


def _field_region(theta: np.ndarray, wmap: np.ndarray, y_from, y_to, xlo: float, xhi: float, sub: int, ref_fn):
    """Signed angles vs ref_fn(x) (deg) and weights of the pixels between y_from(x) and y_to(x)."""
    H, W = theta.shape
    xs = np.arange(max(0, int(np.ceil(xlo))), min(W - 1, int(np.floor(xhi))) + 1, sub)
    empty = (np.zeros(0), np.zeros(0))
    if xs.size < 3:
        return empty
    y0 = np.asarray(y_from(xs + 0.5), float)
    y1 = np.asarray(y_to(xs + 0.5), float)
    ylo = max(0, int(np.floor(np.nanmin(y0))))
    yhi = min(H, int(np.ceil(np.nanmax(y1))) + 1)
    if yhi - ylo < 2:
        return empty
    ys = np.arange(ylo, yhi, sub)
    yy = ys[:, None] + 0.5
    ii, jj = np.nonzero((yy >= y0[None, :]) & (yy <= y1[None, :]))
    if ii.size == 0:
        return empty
    rows, cols = ys[ii], xs[jj]
    ref = np.broadcast_to(np.asarray(ref_fn(cols + 0.5), float), cols.shape)
    return wrap90(theta[rows, cols] - ref), wmap[rows, cols].astype(float)


def _bilinear3(a: np.ndarray, b: np.ndarray, c: np.ndarray, x: float, y: float):
    gx, gy = x - 0.5, y - 0.5
    H, W = a.shape
    if not (0.0 <= gx <= W - 1 and 0.0 <= gy <= H - 1):
        return None
    i0, j0 = int(gy), int(gx)
    i1, j1 = min(i0 + 1, H - 1), min(j0 + 1, W - 1)
    fy, fx = gy - i0, gx - j0
    w00, w01, w10, w11 = (1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx
    return tuple(float(w00 * m[i0, j0] + w01 * m[i0, j1] + w10 * m[i1, j0] + w11 * m[i1, j1]) for m in (a, b, c))


def _trace_regular(vc, vs, sp, thr: float, seed, d0, target: EdgeCurve, side: str, step: float, max_len: float):
    """RK2 streamline through the regularised field until it crosses ``target``; where the support is
    below ``thr`` (outside the FOV / no fascicle evidence) it continues straight in its last direction.
    Returns (length, end point, path points, straight length) or None."""
    p = np.array(seed, float)
    prev = np.asarray(d0, float) / (np.linalg.norm(d0) + 1e-12)
    sgn = 1.0 if side == "deep" else -1.0

    def g(q):
        return sgn * (q[1] - float(target(q[0])))

    def direc(q, pr):
        s = _bilinear3(vc, vs, sp, q[0], q[1])
        if s is None or s[2] < thr or (s[0] == 0.0 and s[1] == 0.0):
            return pr, False
        th = 0.5 * math.atan2(s[1], s[0])
        d = np.array([math.cos(th), math.sin(th)])
        return (-d if d @ pr < 0 else d), True

    g0 = g(p)
    if g0 >= 0:
        return 0.0, (float(p[0]), float(p[1])), [p.copy()], 0.0
    L, Ls, pts = 0.0, 0.0, [p.copy()]
    for _ in range(int(max_len / step) + 1):
        d1, _ = direc(p, prev)
        d2, ok2 = direc(p + 0.5 * step * d1, d1)
        q = p + step * d2
        gq = g(q)
        if gq >= 0:
            t = g0 / (g0 - gq) if g0 != gq else 1.0
            q = p + t * (q - p)
            pts.append(q)
            if not ok2:
                Ls += t * step
            return L + t * step, (float(q[0]), float(q[1])), pts, Ls
        L += step
        if not ok2:
            Ls += step
        pts.append(q)
        p, g0, prev = q, gq, d2
    return None


def _path_tail_angle(pts: list, length: float) -> float:
    """Angle (deg, image coords) of the chord over the last ``length`` px of a path."""
    P = np.asarray(pts, float)
    if len(P) < 2:
        return float("nan")
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    tgt = cum[-1] - length
    k = int(np.searchsorted(cum, tgt)) if tgt > 0 else 0
    k = min(max(k, 0), len(P) - 2)
    d = P[-1] - P[k]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def field_architecture(res: GeoResult, orient, fcfg: FieldConfig = FieldConfig(), pa_reference: str = "deep",
                       pa_for_constructed: dict | None = None) -> dict:
    """PA / FL from a learned orientation field, given a measure(..., keep_details=True) result.

    Variants (lengths in mm when the result has a px/cm, else px):
      pa_fd_strip   robust mean of field angles vs the LOCAL deep tangent in the deep strip
      pa_fd_strip_g same strip vs the GLOBAL deep line (the reference of pa_global / pa_structure)
      pa_fd_ins     narrow insertion strip just above the deep inner edge, vs the local tangent
      pa_fd_comp_g  whole compartment (minus margins) vs the global deep line
      pa_fd_*_h     horizontal reference (dynamic-benchmark convention): pa_fd_strip_h, pa_fd_comp_h
      pa_sl_ins     median insertion angle of the central streamlines at the deep curve (local tangent)
      fl_sl_med / fl_sl_mean / fl_sl_c3 / fl_sl_chord: streamline lengths (median / mean / 3 most central)
      fl_cfd        constructed straight fascicle at pa_fd_strip_g (midpoints at constructed_x)
      fl_c_<k>      constructed fascicle at any PA passed in ``pa_for_constructed`` {k: pa}
    """
    cv = res.curves or {}
    if "sup_curve" not in cv or orient is None:
        return dict(variants={}, diag={})
    sup, deep = cv["sup_curve"], cv["deep_curve"]
    s_lo, s_hi, mt_px, mm = cv["s_lo"], cv["s_hi"], cv["mt_px"], cv["mm"]
    seg, c2, s2 = orient_maps(orient)
    H, W = c2.shape
    theta, wmap = field_theta_weight(seg, c2, s2, fcfg.seg_pow)
    V, D = {}, {}
    lo = s_lo + fcfg.xtrim * (s_hi - s_lo)
    hi = s_hi - fcfg.xtrim * (s_hi - s_lo)
    deep_s, _ = _line_fit(deep.xd, deep.yd)
    g_ang = float(np.degrees(np.arctan(deep_s)))
    ref_local = lambda x: np.degrees(np.arctan(deep.slope(x)))  # noqa: E731
    ref_glob = lambda x: g_ang  # noqa: E731
    ref_hor = lambda x: 0.0  # noqa: E731
    if pa_reference == "horizontal":
        ref_local = ref_glob = ref_hor
    regions = {
        "strip": (lambda x: deep(x) - fcfg.strip[1] * mt_px, lambda x: deep(x) - fcfg.strip[0] * mt_px),
        "ins": (lambda x: deep(x) - fcfg.strip_ins[1] * mt_px, lambda x: deep(x) - fcfg.strip_ins[0] * mt_px),
        "comp": (lambda x: sup(x) + fcfg.comp_margin * mt_px, lambda x: deep(x) - fcfg.comp_margin * mt_px),
    }
    combos = (("strip", "strip", ref_local), ("strip_g", "strip", ref_glob), ("ins", "ins", ref_local),
              ("comp_g", "comp", ref_glob), ("comp", "comp", ref_local), ("strip_h", "strip", ref_hor),
              ("comp_h", "comp", ref_hor))
    sign_votes = []
    for name, reg, ref in combos:
        rel, ww = _field_region(theta, wmap, regions[reg][0], regions[reg][1], lo, hi, fcfg.sub, ref)
        st = robust_axial_mean(rel, ww, fcfg.robust_c)
        okp = int((ww > 0.05).sum()) if ww.size else 0
        pa_v = abs(st["mean"]) if (np.isfinite(st["mean"]) and okp >= fcfg.min_px) else float("nan")
        V[f"pa_fd_{name}"] = pa_v
        if name in ("strip", "strip_g", "comp_g", "comp_h", "strip_h"):
            V[f"pa_fdp_{name}"] = abs(st["plain"]) if np.isfinite(pa_v) else float("nan")
        if name in ("strip", "comp_g", "strip_h"):
            D[f"fd_{name}_conc"] = st["conc"]
            D[f"fd_{name}_inl"] = st["inl"]
            D[f"fd_{name}_wmean"] = float(ww.mean()) if ww.size else 0.0
            D[f"fd_{name}_npx"] = okp
        if name in ("strip_g", "comp_g") and np.isfinite(st["mean"]):
            sign_votes.append(np.sign(st["mean"]))
        if name == "strip_g":
            D["fd_sign"] = float(np.sign(st["mean"])) if np.isfinite(st["mean"]) else float("nan")
    if seg is not None:
        rel, sg = _field_region(theta, seg, regions["comp"][0], regions["comp"][1], lo, hi, fcfg.sub, ref_glob)
        D["fd_seg_frac"] = float((sg > 0.5).mean()) if sg.size else 0.0
    sign = float(np.sign(np.sum(sign_votes))) if sign_votes and np.sum(sign_votes) != 0 else (
        float(sign_votes[0]) if sign_votes else float("nan"))

    # ---- streamlines through the regularised field
    sigma = max(2.0, fcfg.smooth_frac * mt_px)
    mag = np.hypot(c2, s2) + 1e-6
    vc = cv2.GaussianBlur((wmap * c2 / mag).astype(np.float32), (0, 0), sigma)
    vs = cv2.GaussianBlur((wmap * s2 / mag).astype(np.float32), (0, 0), sigma)
    sp = cv2.GaussianBlur(wmap, (0, 0), sigma)
    del mag
    _, spc = _field_region(theta, sp, regions["comp"][0], regions["comp"][1], lo, hi, 4, ref_glob)
    p90 = float(np.percentile(spc, 90)) if spc.size else 0.0
    thr = max(1e-4, fcfg.min_support * p90)
    D["sl_support_p90"] = p90
    lens, chords, ins, extr, xs_ok, paths = [], [], [], [], [], []
    seeds = np.linspace(s_lo + fcfg.seed_range[0] * (s_hi - s_lo), s_lo + fcfg.seed_range[1] * (s_hi - s_lo),
                        fcfg.n_seeds)
    for xs0 in seeds:
        ys0 = 0.5 * (float(sup(xs0)) + float(deep(xs0)))
        s0 = _bilinear3(vc, vs, sp, xs0, ys0)
        if s0 is not None and s0[2] >= thr:
            th = 0.5 * math.atan2(s0[1], s0[0])
        elif np.isfinite(V.get("pa_fd_strip_g", np.nan)) and np.isfinite(sign):
            th = math.radians(g_ang + sign * V["pa_fd_strip_g"])
        else:
            continue
        d = np.array([math.cos(th), math.sin(th)])
        if d[1] < 0:
            d = -d
        down = _trace_regular(vc, vs, sp, thr, (xs0, ys0), d, deep, "deep", fcfg.sl_step, 6.0 * W)
        up = _trace_regular(vc, vs, sp, thr, (xs0, ys0), -d, sup, "sup", fcfg.sl_step, 6.0 * W)
        if down is None or up is None:
            continue
        L = down[0] + up[0]
        if L <= 0:
            continue
        lens.append(L)
        chords.append(math.hypot(down[1][0] - up[1][0], down[1][1] - up[1][1]))
        extr.append((down[3] + up[3]) / L)
        xs_ok.append(xs0)
        paths.append(np.vstack([np.asarray(up[2], float)[::-1], np.asarray(down[2], float)]).astype(np.float32))
        a = _path_tail_angle(down[2], fcfg.ins_frac * mt_px)
        tang = float(np.degrees(np.arctan(deep.slope(down[1][0])))) if pa_reference == "deep" else 0.0
        ins.append(abs(float(wrap90(a - tang))))
    D["sl_n_ok"] = len(lens)
    if lens:
        lens_a = np.array(lens)
        V["fl_sl_med"] = float(np.median(lens_a)) * mm
        V["fl_sl_mean"] = float(np.mean(lens_a)) * mm
        o = np.argsort(np.abs(np.array(xs_ok) - 0.5 * (s_lo + s_hi)))[:3]
        V["fl_sl_c3"] = float(np.mean(lens_a[o])) * mm
        V["fl_sl_chord"] = float(np.median(chords)) * mm
        V["pa_sl_ins"] = float(np.median(ins))
        D["sl_extrap"] = float(np.median(extr))
        D["sl_spread"] = float(_iqr(lens_a) / max(np.median(lens_a), 1e-6)) if len(lens_a) >= 2 else float("nan")
        D["sl_curv"] = float(np.median(lens_a / np.maximum(np.array(chords), 1e-6)))
    else:
        for k in ("fl_sl_med", "fl_sl_mean", "fl_sl_c3", "fl_sl_chord", "pa_sl_ins"):
            V[k] = float("nan")
        D.update(sl_extrap=float("nan"), sl_spread=float("nan"), sl_curv=float("nan"))

    # ---- constructed straight fascicles at field PAs
    xgrid = np.linspace(-3 * W, 4 * W, 6001)
    sgn_c = sign if np.isfinite(sign) else 1.0
    todo = {"cfd": V.get("pa_fd_strip_g", float("nan"))}
    for k, v in (pa_for_constructed or {}).items():
        todo[f"c_{k}"] = v
    for k, pa_v in todo.items():
        if np.isfinite(pa_v) and pa_v > 0.5:
            cons = [constructed_fl(sup, deep, s_lo + f * (s_hi - s_lo), pa_v, sgn_c, mt_px, pa_reference, xgrid)
                    for f in fcfg.constructed_x]
            cons = [c for c in cons if np.isfinite(c)]
            V[f"fl_{k}"] = float(np.mean(cons)) * mm if cons else float("nan")
        else:
            V[f"fl_{k}"] = float("nan")
    return dict(variants=V, diag=D, paths=paths)


def measure_field(apo_prob, fasc_prob, image=None, px_per_cm=None, cfg: GeoConfig = GeoConfig(), orient=None,
                  fcfg: FieldConfig = FieldConfig(), pa_for_constructed=None, **kw) -> GeoResult:
    """measure() plus the learned-field variants of field_architecture() (orient = (seg, cos2, sin2))."""
    kw.pop("keep_details", None)
    res = measure(apo_prob, fasc_prob, image, px_per_cm, cfg, orient=orient, keep_details=True, **kw)
    if orient is not None and res.curves:
        out = field_architecture(res, orient, fcfg, cfg.pa_reference, pa_for_constructed)
        res.variants.update(out["variants"])
        res.diag.update(out["diag"])
        res.curves["sl_paths"] = out.get("paths", [])
    return res
