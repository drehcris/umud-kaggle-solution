"""Empirical-Bayes shrinkage of noisy per-image PA / FL / MT toward scanner-group priors.

Why: on the test scanners our per-image PA/FL are noisy (LB: per-image C1 0.454 -> confidence-shrunk
C2 0.390 with the same MT). This module replaces the ad-hoc ``w = confidence`` by a validated model:

1. **Error model** (``fit_error_model``): the per-image error scale of each component is
   ``log b_i = a + beta . z_i`` where ``z_i`` are standardised observable diagnostics (flags, number
   of fragments, angle IQR, FL spread among variants, extrapolation fraction, FL-vs-triangle and
   fragment-vs-classical PA disagreement, FL/FOV ratio, ...). ``b_i`` is a Laplace scale fitted by
   maximum likelihood (loss ``log b + |r|/b``, convex) with a ridge penalty on ``beta``, on benchmark
   residuals ``r = estimate - truth`` *minus the per-dataset median bias* (a constant bias is not
   removable by shrinkage toward the group's own estimates, so it must not inflate sigma). FL is
   modelled on the relative (log) scale, PA in degrees, MT in mm. The per-image SD used downstream is
   ``sigma = sqrt(2) b`` (Laplace variance).
2. **Group prior** (``fit_group_prior``) per test group g, label-free, from the group's own
   estimates: ``x_u ~ N(mu_g, tau_g^2 + c_g sigma_u^2)``. ``method="mom"`` (the literal spec): c = 1,
   precision-weighted mu, tau^2 = weighted var(x) - sigma^2 (DerSimonian-Laird style, floored, with
   Huber down-weighting of outliers). ``method="ml"`` (default, validated): (mu, tau, c) by a
   Student-t (nu = 5) marginal likelihood with a log-normal hyper-prior on tau centred on the
   between-subject SD of expert labels (PA 3.2 deg, FL 10.5 mm, MT 3.2 mm; consistent across the
   static, VL and dynamic benchmarks; sd(log tau) = 0.2) and a weak one on c (sd(log c) = 2).
   c lets the target data rescale the error model: the spread of a group's own estimates beyond the
   plausible between-subject spread is attributed to noise. This is what makes it transfer to a new
   scanner; c = 1 (MoM) over-shrinks clean sets and under-shrinks VL when held out.
3. **Posterior** per unit: Gaussian likelihood -> ``x' = mu + tau^2/(tau^2 + c sigma^2) (x - mu)``;
   Laplace likelihood (default) -> grid posterior median (keeps moderate deviations when b is small,
   caps believed deviations at ~tau^2/b, robust to gross failures).
   Optional **joint (PA, FL) posterior** that adds the physiological coupling
   ``log(FL sin(PA) / MT) ~ N(log k0, s_k^2)`` (k0 = 1.067, expert labels) as an extra factor, on a
   2-D grid; marginal medians are returned (MAE-optimal).
4. **Clips**: rows sharing a clip id (the 28 Telemed clips of 5 near-identical frames) are one unit:
   the unit value is the frame median and the unit variance is
   ``sigma^2 (rho + (1 - rho)/n)`` with the frame-error correlation rho estimated from the
   within-clip dispersion (label-free).

API
---
>>> model = fit_error_model(bench_df)               # bench_df: estimates + *_true + diagnostics + dataset
>>> model.to_json(path); model = ErrorModel.from_json(path)
>>> out = shrink(test_df, model, group_col="group", clip_col="clip")   # -> pa_eb, fl_eb, w_pa, ...
>>> out.attrs["priors"]                              # per-group mu / tau / c / rho

Input columns: ``pa_deg, fl_mm, mt_mm`` (the per-image estimates to shrink) + whatever diagnostics
exist (geometry_v2 ``as_row()`` names: confidence, flags, d_n_frag, d_pa_iqr, ...). Missing
diagnostics are imputed with the training centre (contribute 0). New diagnostics: pass their column
names as ``extra_features`` to ``fit_error_model`` (any numeric column; also every column whose name
starts with ``x_`` is picked up automatically); the fitted JSON stores everything needed to apply.
Bench truths: ``pa_true, fl_true, mt_true`` (NaN allowed per component) and a ``dataset`` column.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

COMP = {"pa": "pa_deg", "fl": "fl_mm", "mt": "mt_mm"}
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}
# between-subject SD of the expert labels (static mean7 3.5/10.1/3.2, VL 3.0/10.9/3.3,
# dynamic 3.3/8.3): centre of the weak prior on tau. FL in log space: ~0.15.
POP_SD = {"pa": 3.2, "fl": 10.5, "mt": 3.2, "logfl": 0.15}
TAU_FLOOR = {"pa": 0.75, "fl": 3.0, "mt": 0.75, "logfl": 0.04}
K0 = 1.067        # expert FL sin(PA) / MT (static 1.070, VL 1.067)
S_LOGK = 0.11     # per-image SD of log k in expert labels (static 0.10, VL 0.13 with line-MT)
# space in which the error of each component is modelled: abs (units) or rel (log ratio)
ERR_SPACE = {"pa": "abs", "fl": "rel", "mt": "abs"}
SQRT2 = math.sqrt(2.0)

FLAG_NAMES = ("apo_fallback", "pair_fallback", "few_fragments", "multi_band", "short_apo_overlap",
              "pa_implausible", "high_extrapolation", "fl_spread", "fl_triangle_disagree", "pa_spread",
              "pa_classical_disagree")


# ------------------------------------------------------------------------------------ features
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _flag(df: pd.DataFrame, name: str) -> pd.Series:
    if "flags" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    f = df["flags"].fillna("").astype(str)
    return f.str.split(";").apply(lambda xs: float(name in xs))


def _fov_mm(df: pd.DataFrame) -> pd.Series:
    w = _col(df, "d_viewport_w").fillna(_col(df, "d_W"))
    return w / _col(df, "px_per_cm") * 10.0


def _tri_fl(df: pd.DataFrame) -> pd.Series:
    pa = _col(df, "pa_deg").clip(lower=1.0)
    return K0 * _col(df, "mt_mm") / np.sin(np.radians(pa))


FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # summary of the geometry flags (confidence = exp(-sum of penalties))
    "negconf": lambda d: -np.log(_col(d, "confidence").clip(1e-3, 1.0)),
    "log_nfrag": lambda d: np.log1p(_col(d, "d_n_frag").clip(lower=0)),
    "pa_iqr": lambda d: _col(d, "d_pa_iqr").clip(0, 20),
    "extrap": lambda d: _col(d, "d_extrap_med").clip(0, 3),
    "extrap_c3": lambda d: _col(d, "d_extrap_c3").clip(0, 3),
    "fl_iqr": lambda d: _col(d, "d_fl_iqr_rel").clip(0, 2),
    "fl_ens_spread": lambda d: _col(d, "d_fl_ens_spread").clip(0, 2),
    # disagreement of the final estimates with the expert coupling FL = k0 MT / sin(PA)
    "tri_rel": lambda d: np.abs(np.log(_col(d, "fl_mm") / _tri_fl(d))).clip(0, 2),
    # fragment-median PA vs classical directional (image-based) PA
    "pa_dis": lambda d: (_col(d, "pa_global") - _col(d, "pa_structure")).abs().clip(0, 20),
    # learned orientation-field PA vs final PA (only when an orientation model exists)
    "pa_field_dis": lambda d: (_col(d, "pa_field") - _col(d, "pa_deg")).abs().clip(0, 20),
    "shared": lambda d: _col(d, "d_shared_frac").clip(0, 1),
    "apo_conv": lambda d: _col(d, "d_apo_converge").abs().clip(0, 30),
    "log_fl_fov": lambda d: np.log((_col(d, "fl_mm") / _fov_mm(d)).clip(0.05, 20)),
    # d log FL / d PA = cot(PA): low PA makes FL (and the coupling) sensitive to angle errors
    "log_cot_pa": lambda d: np.log(1.0 / np.tan(np.radians(_col(d, "pa_deg").clip(2, 60)))),
    "multi_band": lambda d: (_col(d, "d_n_bands") > 2).astype(float).where(_col(d, "d_n_bands").notna()),
    **{f"flag_{n}": (lambda d, n=n: _flag(d, n)) for n in ("apo_fallback", "pair_fallback", "few_fragments",
                                                         "short_apo_overlap", "pa_implausible")},
}
DEFAULT_FEATURES = ("negconf", "log_nfrag", "pa_iqr", "extrap", "extrap_c3", "fl_iqr", "fl_ens_spread",
                    "tri_rel", "pa_dis", "shared", "apo_conv", "log_fl_fov", "log_cot_pa",
                    "flag_apo_fallback", "flag_pair_fallback", "flag_few_fragments", "flag_short_apo_overlap",
                    "flag_pa_implausible")


def make_features(df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Feature matrix (NaN where a diagnostic is unavailable). Unknown names are read as raw columns."""
    out = {}
    for n in names:
        out[n] = FEATURES[n](df) if n in FEATURES else _col(df, n)
    return pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)


def auto_extra(df: pd.DataFrame) -> list[str]:
    """Columns following the ``x_`` convention for new diagnostics."""
    return [c for c in df.columns if str(c).startswith("x_") and pd.api.types.is_numeric_dtype(df[c])]


# ------------------------------------------------------------------------------------ error model
def _residual(df: pd.DataFrame, comp: str) -> pd.Series:
    est = _col(df, COMP[comp])
    tru = _col(df, f"{comp}_true")
    if ERR_SPACE[comp] == "rel":
        return np.log(est.clip(lower=1e-6) / tru.clip(lower=1e-6))
    return est - tru


@dataclass
class TargetModel:
    intercept: float
    coef: dict
    space: str
    train_bias: dict = field(default_factory=dict)     # per-dataset median residual (removed before fit)
    train_level: dict = field(default_factory=dict)    # per-dataset mean |r| / mean b (calibration)
    n: int = 0


@dataclass
class ErrorModel:
    features: list
    center: dict
    scale: dict
    targets: dict                     # comp -> TargetModel
    zclip: float = 4.0
    meta: dict = field(default_factory=dict)

    # -- prediction
    def z(self, df: pd.DataFrame) -> pd.DataFrame:
        F = make_features(df, self.features)
        Z = (F - pd.Series(self.center)) / pd.Series(self.scale)
        return Z.fillna(0.0).clip(-self.zclip, self.zclip)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-image error SD in the units of the estimate: sigma_pa (deg), sigma_fl (mm), sigma_mt
        (mm), plus sigma_fl_rel (log scale). sigma = sqrt(2) * Laplace scale."""
        Z = self.z(df)
        out = pd.DataFrame(index=df.index)
        for comp, tm in self.targets.items():
            beta = pd.Series(tm.coef).reindex(self.features).fillna(0.0)
            b = np.exp(tm.intercept + Z.to_numpy(dtype=float) @ beta.to_numpy(dtype=float))
            s = SQRT2 * b
            if tm.space == "rel":
                out[f"sigma_{comp}_rel"] = s
                s = s * _col(df, COMP[comp]).to_numpy()
            out[f"sigma_{comp}"] = s
        return out

    # -- (de)serialisation
    def to_dict(self) -> dict:
        return dict(version=1, features=list(self.features), center=self.center, scale=self.scale,
                    zclip=self.zclip, meta=self.meta,
                    targets={k: dict(intercept=v.intercept, coef=v.coef, space=v.space, train_bias=v.train_bias,
                                     train_level=v.train_level, n=v.n) for k, v in self.targets.items()})

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=float))

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorModel":
        return cls(features=list(d["features"]), center=d["center"], scale=d["scale"], zclip=d.get("zclip", 4.0),
                   meta=d.get("meta", {}), targets={k: TargetModel(**v) for k, v in d["targets"].items()})

    @classmethod
    def from_json(cls, path) -> "ErrorModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _fit_laplace_scale(Z: np.ndarray, r: np.ndarray, w: np.ndarray, ridge: float) -> tuple[float, np.ndarray]:
    """min sum_i w_i (eta_i + |r_i| exp(-eta_i)) + ridge/2 |beta|^2, eta = a + Z beta (convex)."""
    a_r = np.abs(r) + 1e-9
    p = Z.shape[1]
    W = w / w.sum()

    def f(th):
        eta = th[0] + Z @ th[1:]
        e = np.exp(-eta)
        loss = np.sum(W * (eta + a_r * e)) + 0.5 * ridge * np.sum(th[1:] ** 2)
        g_eta = W * (1.0 - a_r * e)
        grad = np.concatenate([[g_eta.sum()], Z.T @ g_eta + ridge * th[1:]])
        return loss, grad

    th0 = np.zeros(p + 1)
    th0[0] = math.log(np.sum(W * a_r))
    res = minimize(f, th0, jac=True, method="L-BFGS-B")
    return float(res.x[0]), res.x[1:]


# PA level is not a transferable predictor of PA (or MT) error: on the single-subject dynamic
# sequence the PA error grows with PA (estimator attenuation), an artefact of that one dataset.
DEFAULT_EXCLUDE = {"pa": ("log_cot_pa",), "mt": ("log_cot_pa",)}
# effective number of independent rows per dataset (167 dynamic frames are one subject's video)
DEFAULT_NEFF = {"dynamic": 20}


def fit_error_model(bench: pd.DataFrame, targets: Iterable[str] = ("pa", "fl", "mt"),
                    features: Sequence[str] | None = None, extra_features: Sequence[str] = (),
                    exclude: dict | None = None, dataset_col: str = "dataset", ridge: float = 0.3,
                    balance: str = "sqrt", n_eff: dict | None = None, demean: bool = True,
                    min_rows: int = 10) -> ErrorModel:
    """Fit the per-image error-scale model on benchmark rows.

    bench: estimates ``pa_deg, fl_mm, mt_mm`` + truths ``pa_true, fl_true, mt_true`` (NaN = unknown)
    + diagnostics + ``dataset``. balance: dataset d gets total weight proportional to n_d ("none"),
    sqrt(n_d) ("sqrt") or 1 ("equal"), with n_d replaced by ``n_eff[d]`` when given (default: the
    dynamic video counts as 20 independent rows). exclude: comp -> features not used for that comp.
    """
    base = list(DEFAULT_FEATURES if features is None else features)
    feats = base + [c for c in list(extra_features) + auto_extra(bench) if c not in base]
    excl = DEFAULT_EXCLUDE if exclude is None else exclude
    neff = DEFAULT_NEFF if n_eff is None else n_eff
    F = make_features(bench, feats)
    ds = bench[dataset_col].astype(str) if dataset_col in bench else pd.Series("all", index=bench.index)
    center = F.median().fillna(0.0)
    q = (F.quantile(0.75) - F.quantile(0.25))
    sd = F.std()
    scale = q.where(q > 1e-9, sd).where(lambda s: s > 1e-9, 1.0).fillna(1.0)
    keep = [c for c in feats if F[c].notna().sum() >= min_rows and F[c].nunique() > 1]
    Z = ((F[keep] - center[keep]) / scale[keep]).fillna(0.0).clip(-4, 4)
    tms = {}
    for comp in targets:
        r = _residual(bench, comp)
        ok = r.notna() & np.isfinite(r)
        if ok.sum() < min_rows:
            continue
        fk = [c for c in keep if c not in excl.get(comp, ())]
        bias = r[ok].groupby(ds[ok]).median() if demean else pd.Series(0.0, index=ds[ok].unique())
        rr = r[ok] - ds[ok].map(bias).astype(float)
        cnt = ds[ok].value_counts().astype(float)
        nd = ds[ok].map(cnt).astype(float)
        ne = ds[ok].map({k: min(float(neff.get(k, v)), v) for k, v in cnt.items()}).astype(float)
        # per-row weight w = W_d / n_d with dataset total W_d = n_eff, sqrt(n_eff) or 1
        w = {"none": ne / nd, "sqrt": np.sqrt(ne) / nd, "equal": 1.0 / nd}[balance]
        Zc = Z.loc[ok.index[ok], fk].to_numpy(dtype=float).reshape(int(ok.sum()), len(fk))
        a, beta = _fit_laplace_scale(Zc, rr.to_numpy(), np.asarray(w, float), ridge)
        b = np.exp(a + Zc @ beta)
        lvl = (pd.Series(np.abs(rr.to_numpy()), index=rr.index).groupby(ds[ok]).mean()
               / pd.Series(b, index=rr.index).groupby(ds[ok]).mean())
        tms[comp] = TargetModel(intercept=a, coef={k: float(v) for k, v in zip(fk, beta)}, space=ERR_SPACE[comp],
                                train_bias={k: float(v) for k, v in bias.items()},
                                train_level={k: float(v) for k, v in lvl.items()}, n=int(ok.sum()))
    return ErrorModel(features=keep, center={k: float(center[k]) for k in keep},
                      scale={k: float(scale[k]) for k in keep}, targets=tms,
                      meta=dict(ridge=ridge, balance=balance, demean=demean, n_eff=dict(neff),
                                exclude={k: list(v) for k, v in excl.items()},
                                datasets={k: int(v) for k, v in ds.value_counts().items()}))


# ------------------------------------------------------------------------------------ group prior
@dataclass
class Prior:
    mu: float
    tau: float
    c: float = 1.0
    n: int = 0
    method: str = "mom"


def _huber_w(z: np.ndarray, k: float = 2.5) -> np.ndarray:
    a = np.abs(z)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))


def fit_group_prior(x: np.ndarray, s: np.ndarray, comp: str, method: str = "ml", pop_sd: float | None = None,
                    tau_floor: float | None = None, prior_sd_logtau: float = 0.2, prior_sd_logc: float = 2.0,
                    c_fixed: float | None = None, marginal: str = "t", nu: float = 5.0, iters: int = 50,
                    mu_mode: str = "fit") -> Prior:
    """Label-free prior (mu, tau, c) for one group from its estimates x with error SDs s.

    mu_mode: "fit" (mu estimated jointly, i.e. precision-weighted and robust) or "median" (mu fixed
    at the plain median of the unit estimates: no precision weighting, so no bias when sigma depends
    on the estimate itself, e.g. via FL/FOV)."""
    x = np.asarray(x, float)
    s = np.asarray(s, float)
    ok = np.isfinite(x) & np.isfinite(s) & (s > 0)
    x, s = x[ok], s[ok]
    pop = POP_SD[comp] if pop_sd is None else pop_sd
    floor = TAU_FLOOR[comp] if tau_floor is None else tau_floor
    n = len(x)
    if n == 0:
        return Prior(np.nan, pop, 1.0, 0, method)
    if n == 1:
        return Prior(float(x[0]), pop, 1.0, 1, method)
    # --- MoM (DerSimonian-Laird / Paule-Mandel style), Huber-robust
    c0 = 1.0 if c_fixed is None else c_fixed
    s2 = c0 * s ** 2
    mu = float(np.median(x))
    mad = 1.4826 * np.median(np.abs(x - mu))
    tau2 = max(floor ** 2, mad ** 2 - float(np.median(s2)))
    fix_mu = mu_mode == "median"
    for _ in range(iters):
        v = tau2 + s2
        hw = _huber_w((x - mu) / np.sqrt(v))
        w = hw / v
        mu_new = mu if fix_mu else float(np.sum(w * x) / np.sum(w))
        t_new = float(np.sum(w * ((x - mu_new) ** 2 - s2)) / np.sum(w))
        t_new = max(floor ** 2, t_new)
        if abs(mu_new - mu) < 1e-7 and abs(t_new - tau2) < 1e-9:
            mu, tau2 = mu_new, t_new
            break
        mu, tau2 = mu_new, t_new
    if method == "mom":
        return Prior(mu, math.sqrt(tau2), c0, n, "mom")
    # --- marginal likelihood with weak priors on log tau and log c
    fix_c = c_fixed is not None

    def nll(th):
        m, lt = (mu if fix_mu else th[0]), th[1]
        lc = math.log(c_fixed) if fix_c else th[2]
        v = np.exp(2 * lt) + np.exp(lc) * s ** 2
        z2 = (x - m) ** 2 / v
        if marginal == "t":
            ll = -0.5 * np.log(v) - 0.5 * (nu + 1) * np.log1p(z2 / nu)
        else:
            ll = -0.5 * np.log(v) - 0.5 * z2
        pen = 0.5 * ((lt - math.log(pop)) / prior_sd_logtau) ** 2
        if not fix_c:
            pen += 0.5 * (lc / prior_sd_logc) ** 2
        return -float(np.sum(ll)) + pen

    th0 = [mu, 0.5 * math.log(tau2)] + ([] if fix_c else [0.0])
    sx = float(np.std(x)) + 1e-6
    bounds = [(mu - 10 * sx - 10 * pop, mu + 10 * sx + 10 * pop), (math.log(floor), math.log(20 * pop))]
    if not fix_c:
        bounds.append((math.log(0.05), math.log(20.0)))
    if fix_mu:
        bounds[0] = (mu, mu)
    res = minimize(nll, th0, method="L-BFGS-B", bounds=bounds)
    m, lt = (mu if fix_mu else res.x[0]), res.x[1]
    c = c_fixed if fix_c else float(math.exp(res.x[2]))
    return Prior(float(m), float(math.exp(lt)), float(c), n, "ml")


# ------------------------------------------------------------------------------------ posteriors
def _post_1d(x: float, s: float, pr: Prior, likelihood: str) -> float:
    """Posterior median of theta | x for prior N(mu, tau^2) and error SD sqrt(c) s."""
    if not np.isfinite(x) or not np.isfinite(s):
        return pr.mu
    se = math.sqrt(pr.c) * s
    if likelihood == "gauss":
        B = pr.tau ** 2 / (pr.tau ** 2 + se ** 2)
        return pr.mu + B * (x - pr.mu)
    lo = min(pr.mu - 6 * pr.tau, x - 6 * se)
    hi = max(pr.mu + 6 * pr.tau, x + 6 * se)
    th = np.linspace(lo, hi, 1601)
    lp = -0.5 * ((th - pr.mu) / pr.tau) ** 2
    if likelihood == "laplace":
        lp = lp - np.abs(x - th) / (se / SQRT2)
    elif likelihood.startswith("t"):
        nu = float(likelihood[1:] or 4)
        sc = se * math.sqrt((nu - 2) / nu)
        lp = lp - 0.5 * (nu + 1) * np.log1p(((x - th) / sc) ** 2 / nu)
    else:
        raise ValueError(likelihood)
    p = np.exp(lp - lp.max())
    cdf = np.cumsum(p)
    return float(np.interp(0.5, cdf / cdf[-1], th))


def _lik(th: np.ndarray, x: float, se: float, likelihood: str) -> np.ndarray:
    if likelihood == "gauss":
        return -0.5 * ((x - th) / se) ** 2
    if likelihood == "laplace":
        return -np.abs(x - th) / (se / SQRT2)
    nu = float(likelihood[1:] or 4)
    sc = se * math.sqrt((nu - 2) / nu)
    return -0.5 * (nu + 1) * np.log1p(((x - th) / sc) ** 2 / nu)


def joint_posterior(pa: float, s_pa: float, fl: float, s_fl: float, mt: float, s_mt: float,
                    pr_pa: Prior, pr_fl: Prior, likelihood: str = "gauss", k0: float = K0,
                    s_logk: float = S_LOGK, fl_log: bool = False, n_grid: int = 241) -> tuple[float, float]:
    """Marginal posterior medians of (PA, FL) under group priors, the measurement likelihoods and the
    coupling factor log(FL sin PA / MT) ~ N(log k0, s_logk^2 + (s_mt/mt)^2). NaN measurements are
    simply absent factors (e.g. FL failed -> FL is inferred from PA, MT and the prior)."""
    tp = pr_pa.tau
    pa_lo, pa_hi = max(1.0, pr_pa.mu - 5 * tp), min(60.0, pr_pa.mu + 5 * tp)
    if np.isfinite(pa):
        pa_lo, pa_hi = min(pa_lo, max(1.0, pa - 4 * s_pa)), max(pa_hi, min(60.0, pa + 4 * s_pa))
    g_pa = np.linspace(pa_lo, pa_hi, n_grid)
    if fl_log:   # FL prior is on log FL
        c_lo, c_hi = pr_fl.mu - 5 * pr_fl.tau, pr_fl.mu + 5 * pr_fl.tau
        if np.isfinite(fl):
            c_lo, c_hi = min(c_lo, math.log(max(fl, 1.0)) - 4 * s_fl / max(fl, 1.0)), \
                max(c_hi, math.log(max(fl, 1.0)) + 4 * s_fl / max(fl, 1.0))
        g_fl = np.exp(np.linspace(c_lo, c_hi, n_grid))
    else:
        f_lo, f_hi = max(5.0, pr_fl.mu - 5 * pr_fl.tau), pr_fl.mu + 5 * pr_fl.tau
        if np.isfinite(fl):
            f_lo, f_hi = min(f_lo, max(5.0, fl - 4 * s_fl)), max(f_hi, fl + 4 * s_fl)
        g_fl = np.linspace(f_lo, f_hi, n_grid)
    P, Fm = np.meshgrid(g_pa, g_fl, indexing="ij")
    lp = -0.5 * ((P - pr_pa.mu) / pr_pa.tau) ** 2
    lp = lp + (-0.5 * ((np.log(Fm) - pr_fl.mu) / pr_fl.tau) ** 2 - np.log(Fm) if fl_log
               else -0.5 * ((Fm - pr_fl.mu) / pr_fl.tau) ** 2)
    if np.isfinite(pa) and np.isfinite(s_pa):
        lp = lp + _lik(P, pa, math.sqrt(pr_pa.c) * s_pa, likelihood)
    if np.isfinite(fl) and np.isfinite(s_fl):
        lp = lp + _lik(Fm, fl, math.sqrt(pr_fl.c) * s_fl, likelihood)
    if np.isfinite(mt) and mt > 0 and s_logk > 0:
        sk2 = s_logk ** 2 + (s_mt / mt) ** 2 if np.isfinite(s_mt) else s_logk ** 2
        lk = np.log(Fm * np.sin(np.radians(P)) / (k0 * mt))
        lp = lp - 0.5 * lk ** 2 / sk2
    p = np.exp(lp - lp.max())
    if fl_log:   # non-uniform FL grid: weight by cell width
        p = p * np.gradient(g_fl)[None, :]
    else:
        p = p * 1.0
    m_pa = p.sum(1)
    m_fl = p.sum(0)
    med = lambda g, m: float(np.interp(0.5, np.cumsum(m) / m.sum(), g))  # noqa: E731
    return med(g_pa, m_pa), med(g_fl, m_fl)


# ------------------------------------------------------------------------------------ shrink
def _units(df: pd.DataFrame, group_col: str | None, clip_col: str | None) -> pd.Series:
    g = df[group_col].astype(str) if group_col and group_col in df else pd.Series("all", index=df.index)
    if clip_col and clip_col in df:
        cl = pd.to_numeric(df[clip_col], errors="coerce")
        u = np.where(cl.fillna(-1) >= 0, g + "|clip" + cl.fillna(-1).astype(int).astype(str), "img|" + df.index.astype(str))
        return pd.Series(u, index=df.index)
    return pd.Series("img|" + df.index.astype(str), index=df.index)


def estimate_rho(x: pd.Series, s: pd.Series, unit: pd.Series, default: float = 0.8) -> float:
    """Frame-error correlation from the within-clip dispersion: E var_within = (1 - rho) sigma^2."""
    multi = unit.map(unit.value_counts()) > 1
    if multi.sum() < 4:
        return default
    xx, ss, uu = x[multi], s[multi], unit[multi]
    ok = xx.notna() & ss.notna()
    if ok.sum() < 4:
        return default
    W = xx[ok].groupby(uu[ok]).var(ddof=1).dropna()
    S2 = (ss[ok] ** 2).groupby(uu[ok]).mean().reindex(W.index)
    if len(W) == 0 or S2.sum() <= 0:
        return default
    return float(np.clip(1.0 - np.median(W / S2), 0.0, 1.0))


def shrink(df: pd.DataFrame, model: ErrorModel, group_col: str = "group", clip_col: str | None = "clip",
           cols: Sequence[str] = ("pa", "fl"), method: str = "ml", likelihood: str = "laplace",
           joint: bool = False, fl_space: str = "lin", rho: float | None = None, sigma_scale: float = 1.0,
           c_fixed: float | None = None, marginal: str = "t", prior_kw: dict | None = None,
           joint_kw: dict | None = None, sigma: pd.DataFrame | None = None) -> pd.DataFrame:
    """Shrink per-image estimates toward label-free group priors.

    Returns a copy of df with ``<comp>_eb`` (posterior value, in the units of pa_deg/fl_mm/mt_mm),
    ``w_<comp>`` (effective weight on the own measurement, (x'-mu)/(x-mu)), ``sigma_<comp>``,
    ``mu_<comp>`` and ``tau_<comp>`` per row; components not in ``cols`` are copied unchanged into
    ``<comp>_eb``. ``attrs["priors"]`` holds the per-group fit. ``joint=True`` replaces the PA and FL
    posteriors by the 2-D posterior with the FL sin(PA) = k0 MT coupling (needs pa and fl in cols).
    ``method``: "ml" (mu, tau, c by marginal likelihood), "mom" (c = 1, method of moments),
    "none" (no shrinkage).
    """
    out = df.copy()
    sig = (model.predict(df) if sigma is None else sigma) * sigma_scale
    unit = _units(df, group_col, clip_col)
    grp = df[group_col].astype(str) if group_col in df else pd.Series("all", index=df.index)
    ugrp = grp.groupby(unit).first()
    priors: dict = {}
    post_unit: dict = {}
    pk = dict(prior_kw or {})
    for comp in COMP:
        x = _col(df, COMP[comp])
        out[f"{comp}_eb"] = x
        if f"sigma_{comp}" in sig:
            out[f"sigma_{comp}"] = sig[f"sigma_{comp}"]
    for comp in cols:
        x = _col(df, COMP[comp])
        s = sig[f"sigma_{comp}"].astype(float)
        logsp = comp == "fl" and fl_space == "log"
        if logsp:
            s = sig.get("sigma_fl_rel", s / x).astype(float)
            x = np.log(x.where(x > 0))
        key = "logfl" if logsp else comp
        r_by_g = {}
        for g in ugrp.unique():
            m = grp == g
            r_by_g[g] = (rho if rho is not None else estimate_rho(x[m], s[m], unit[m]))
        xu = x.groupby(unit).median()
        nu_ = x.notna().groupby(unit).sum()
        s2u = (s ** 2).groupby(unit).mean()
        rr = ugrp.map(r_by_g).astype(float)
        su = np.sqrt(s2u * (rr + (1 - rr) / nu_.clip(lower=1)))
        su[nu_ == 0] = np.nan
        xu[nu_ == 0] = np.nan
        pu = pd.Series(np.nan, index=xu.index)
        for g in ugrp.unique():
            ui = ugrp.index[ugrp == g]
            if method == "none":
                pr = Prior(float(np.nanmedian(xu[ui])), np.inf, 1.0, len(ui), "none")
            else:
                pr = fit_group_prior(xu[ui].to_numpy(), su[ui].to_numpy(), key, method=method, c_fixed=c_fixed,
                                     marginal=marginal, **pk)
            priors[(g, comp)] = dict(mu=pr.mu, tau=pr.tau, c=pr.c, n_units=pr.n, rho=r_by_g[g], method=pr.method,
                                     space="log" if logsp else "lin")
            for u in ui:
                if method == "none":
                    pu[u] = xu[u] if np.isfinite(xu[u]) else pr.mu
                else:
                    pu[u] = _post_1d(xu[u], su[u], pr, likelihood)
        post_unit[comp] = (pu, xu, su)
        val = unit.map(pu).astype(float)
        mu_r = grp.map({g: priors[(g, comp)]["mu"] for g in ugrp.unique()}).astype(float)
        tau_r = grp.map({g: priors[(g, comp)]["tau"] for g in ugrp.unique()}).astype(float)
        if logsp:
            val, mu_r, xr = np.exp(val), np.exp(mu_r), np.exp(x)
            tau_r = tau_r * mu_r
        else:
            xr = x
        out[f"{comp}_eb"] = val
        out[f"mu_{comp}"] = mu_r
        out[f"tau_{comp}"] = tau_r
        xu_r = np.exp(unit.map(xu)) if logsp else unit.map(xu)
        d = (xu_r - mu_r)
        out[f"w_{comp}"] = ((val - mu_r) / d.where(d.abs() > 1e-9)).clip(-0.5, 1.5).fillna(1.0)
        out.loc[xr.isna(), f"w_{comp}"] = 0.0
    if joint and "pa" in cols and "fl" in cols:
        pu_pa, xu_pa, su_pa = post_unit["pa"]
        pu_fl, xu_fl, su_fl = post_unit["fl"]
        logsp = fl_space == "log"
        mt_u = _col(df, "mt_mm").groupby(unit).median()
        smt_u = (sig["sigma_mt"] if "sigma_mt" in sig else pd.Series(np.nan, index=df.index)).groupby(unit).mean()
        jk = dict(joint_kw or {})
        jpa, jfl = {}, {}
        for u in xu_pa.index:
            g = ugrp[u]
            ppa = Prior(**{k: priors[(g, "pa")][k] for k in ("mu", "tau", "c")})
            pfl = Prior(**{k: priors[(g, "fl")][k] for k in ("mu", "tau", "c")})
            flv, sflv = xu_fl[u], su_fl[u]
            if logsp and np.isfinite(flv):   # back to mm for the likelihood (delta method)
                flv = math.exp(flv)
                sflv = sflv * flv
            jpa[u], jfl[u] = joint_posterior(xu_pa[u], su_pa[u], flv, sflv, mt_u[u], smt_u[u], ppa, pfl,
                                             likelihood=likelihood, fl_log=logsp, **jk)
        out["pa_eb"] = unit.map(jpa).astype(float)
        out["fl_eb"] = unit.map(jfl).astype(float)
        for comp in ("pa", "fl"):
            d = unit.map(post_unit[comp][1]).astype(float)
            if comp == "fl" and logsp:
                d = np.exp(d)
            d = d - out[f"mu_{comp}"]
            out[f"w_{comp}"] = ((out[f"{comp}_eb"] - out[f"mu_{comp}"]) / d.where(d.abs() > 1e-9)).clip(-0.5, 1.5).fillna(1.0)
            out.loc[_col(df, COMP[comp]).isna(), f"w_{comp}"] = 0.0
    out.attrs["priors"] = {f"{g}|{c}": v for (g, c), v in priors.items()}
    out["eb_unit"] = unit
    return out


# ------------------------------------------------------------------------------------ baselines
def confidence_shrink(df: pd.DataFrame, group_col: str = "group", cols: Sequence[str] = ("pa", "fl"),
                      lam: float = 1.0) -> pd.DataFrame:
    """The C2 recipe: w = 1 - lam (1 - confidence) toward the group median of the estimates."""
    out = df.copy()
    conf = _col(df, "confidence").fillna(0.0).clip(0, 1)
    w = 1.0 - lam * (1.0 - conf)
    for comp in COMP:
        out[f"{comp}_eb"] = _col(df, COMP[comp])
    for comp in cols:
        x = _col(df, COMP[comp])
        med = x.groupby(df[group_col]).transform("median")
        out[f"{comp}_eb"] = (w * x + (1 - w) * med).where(x.notna(), med)
    return out


def group_median(df: pd.DataFrame, group_col: str = "group", cols: Sequence[str] = ("pa", "fl")) -> pd.DataFrame:
    """The C3 recipe: PA/FL replaced by the group median of the estimates."""
    out = df.copy()
    for comp in COMP:
        out[f"{comp}_eb"] = _col(df, COMP[comp])
    for comp in cols:
        out[f"{comp}_eb"] = _col(df, COMP[comp]).groupby(df[group_col]).transform("median")
    return out


# ==================================================================================================
# flfix additions (2026-10-21). Additive: nothing above changes, and with no override the constants
# above are used exactly as before.
#  * label_constants() / constants(): recompute POP_SD, K0 and S_LOGK from a chosen set of expert-label
#    tables, so that a held-out evaluation never uses the held-out labels (the module constants above
#    include VL labels; critic fix 5).
#  * shrink_fl_coupled(): FL-only empirical-Bayes posteriors that use the expert coupling
#    FL sin(PA) / MT ~ k, with the output PA supplied from elsewhere (e.g. the confidence rule;
#    critic fix 3).
# ==================================================================================================
from contextlib import contextmanager  # noqa: E402


def label_constants(tables: dict, min_n: int = 5) -> dict:
    """POP_SD / K0 / S_LOGK from expert-label tables ``{name: DataFrame(pa_deg, fl_mm, mt_mm)}`` (NaN ok).

    Rule (on static mean7 + VL + dynamic it gives PA 3.28 / FL 9.76 / MT 3.25, K0 1.069, S_LOGK 0.117,
    i.e. the hand-set constants above to within rounding): POP_SD[c] = mean over the tables of the
    between-image SD of c (tables with >= min_n values); K0 = mean over the tables of median(k),
    S_LOGK = mean over the tables of SD(log k), k = FL sin(PA) / MT (tables with all three)."""
    sds: dict = {c: [] for c in COMP}
    ks, sk = [], []
    for t in tables.values():
        for c, col in COMP.items():
            if col in t:
                v = pd.to_numeric(t[col], errors="coerce").dropna()
                if len(v) >= min_n:
                    sds[c].append(float(v.std()))
        if all(col in t for col in COMP.values()):
            k = (pd.to_numeric(t["fl_mm"], errors="coerce") * np.sin(np.radians(pd.to_numeric(t["pa_deg"], errors="coerce")))
                 / pd.to_numeric(t["mt_mm"], errors="coerce")).replace([np.inf, -np.inf], np.nan).dropna()
            k = k[k > 0]
            if len(k) >= min_n:
                ks.append(float(k.median()))
                sk.append(float(np.log(k).std()))
    return dict(pop_sd={c: float(np.mean(v)) for c, v in sds.items() if v},
                k0=float(np.mean(ks)) if ks else None, s_logk=float(np.mean(sk)) if sk else None,
                sources=sorted(tables))


@contextmanager
def constants(pop_sd: dict | None = None, k0: float | None = None, s_logk: float | None = None):
    """Temporarily override POP_SD (per component), K0 and S_LOGK.

    They are module globals read at call time by the error-model feature ``tri_rel``, by
    ``fit_group_prior`` (tau anchor) and by ``shrink_fl_coupled``. ``joint_posterior``'s defaults are
    bound at import, so pass k0 / s_logk explicitly (``shrink(joint_kw=...)``) inside the context."""
    global POP_SD, K0, S_LOGK
    old = (POP_SD, K0, S_LOGK)
    try:
        if pop_sd:
            POP_SD = {**POP_SD, **{k: float(v) for k, v in pop_sd.items()}}
        if k0 is not None:
            K0 = float(k0)
        if s_logk is not None:
            S_LOGK = float(s_logk)
        yield
    finally:
        POP_SD, K0, S_LOGK = old


def _unit_values(x: pd.Series, s: pd.Series, unit: pd.Series, grp: pd.Series, ugrp: pd.Series,
                 rho: float | None = None) -> tuple[pd.Series, pd.Series, dict]:
    """Unit (clip) value = frame median and unit SD = sqrt(mean s^2 (rho + (1 - rho) / n)), as in shrink()."""
    r_by_g = {}
    for g in ugrp.unique():
        m = grp == g
        r_by_g[g] = rho if rho is not None else estimate_rho(x[m], s[m], unit[m])
    xu = x.groupby(unit).median()
    nu_ = x.notna().groupby(unit).sum()
    s2u = (s ** 2).groupby(unit).mean()
    rr = ugrp.map(r_by_g).astype(float)
    su = np.sqrt(s2u * (rr + (1 - rr) / nu_.clip(lower=1)))
    su[nu_ == 0] = np.nan
    xu[nu_ == 0] = np.nan
    return xu, su, r_by_g


def _post_fl_coupled(x: float, se: float, pr: Prior, fl_tri: float, s_eff: float, likelihood: str = "laplace",
                     n_grid: int = 1601) -> float:
    """Posterior median of FL for prior N(mu, tau^2) x likelihood(x | FL, sqrt(c) se) x coupling factor
    log FL ~ N(log fl_tri, s_eff^2). A missing x or coupling is simply an absent factor."""
    has_c = bool(np.isfinite(fl_tri) and fl_tri > 0 and np.isfinite(s_eff) and s_eff > 0)
    if not has_c:
        return _post_1d(x, se, pr, likelihood)
    has_x = bool(np.isfinite(x) and np.isfinite(se) and se > 0)
    sc = math.sqrt(pr.c) * se if has_x else float("nan")
    lo = min(pr.mu - 6 * pr.tau, fl_tri * math.exp(-5 * s_eff), x - 6 * sc if has_x else np.inf)
    hi = max(pr.mu + 6 * pr.tau, fl_tri * math.exp(5 * s_eff), x + 6 * sc if has_x else -np.inf)
    th = np.linspace(max(lo, 1.0), max(hi, 2.0), n_grid)
    lp = -0.5 * ((th - pr.mu) / pr.tau) ** 2 - 0.5 * (np.log(th / fl_tri) / s_eff) ** 2
    if has_x:
        lp = lp + _lik(th, x, sc, likelihood)
    p = np.exp(lp - lp.max())
    cdf = np.cumsum(p)
    return float(np.interp(0.5, cdf / cdf[-1], th))


def shrink_fl_coupled(df: pd.DataFrame, model: ErrorModel, pa_raw: pd.Series, pa_c: pd.Series, mode: str = "cpl",
                      k_mode: str = "k0", group_col: str = "group", clip_col: str | None = "clip",
                      likelihood: str = "laplace", prior_kw: dict | None = None, rho: float | None = None,
                      k0: float | None = None, s_logk: float | None = None) -> pd.DataFrame:
    """FL-only posteriors that use the coupling FL sin(PA) / MT ~ k (PA output is NOT changed here).

    pa_raw : per-row raw PA (deg, deep-aponeurosis reference) of the recipe; its EB group prior and the
             error model give the PA uncertainty (and the PA factor of mode "joint").
    pa_c   : per-row PA the coupling is evaluated at: the PA that will be output (e.g. confidence-shrunk).
    mode   : "joint" - FL marginal median of joint_posterior (EB PA prior + PA likelihood on pa_raw, FL
                       prior + likelihood, coupling factor N(log k, s_logk^2 + (s_mt/MT)^2)).
             "cpl"   - 1-D posterior: group FL prior x FL likelihood x coupling factor
                       log FL ~ N(log(k MT / sin pa_c), s_eff^2) with s_eff^2 = s_logk^2 +
                       (cot(pa_c) sd_pa)^2 + (s_mt / MT)^2, sd_pa = Gaussian EB posterior SD of PA.
             "floor" - the plain EB FL posterior, with the weight on the unit's own measurement floored at
                       a = exp(-0.5 (log(x / FL_tri) / s_logk)^2): raw FL that agrees with the coupling at
                       the output PA is kept.
    k_mode : "k0" (expert ratio K0) or "group" (median over the group's units of x sin(pa_c) / MT;
             label-free: redistributes FL within a group by MT / sin(PA) without moving its level).
    Returns shrink(df, model, cols=("fl",), ...) with fl_eb / w_fl replaced, plus fl_tri, s_eff, k_used,
    a_floor (floor weight) and sd_pa_post per row.
    """
    k0 = K0 if k0 is None else float(k0)
    s_logk = S_LOGK if s_logk is None else float(s_logk)
    pk = dict(prior_kw or {})
    out = shrink(df, model, group_col=group_col, clip_col=clip_col, cols=("fl",), likelihood=likelihood,
                 rho=rho, prior_kw=pk)
    sig = model.predict(df)
    unit = _units(df, group_col, clip_col)
    grp = df[group_col].astype(str) if group_col in df else pd.Series("all", index=df.index)
    ugrp = grp.groupby(unit).first()
    x_fl = _col(df, "fl_mm")
    xu_fl, su_fl, _ = _unit_values(x_fl, sig["sigma_fl"].astype(float), unit, grp, ugrp, rho)
    xpa = pd.Series(np.asarray(pa_raw, float), index=df.index)
    xu_pa, su_pa, _ = _unit_values(xpa, sig["sigma_pa"].astype(float), unit, grp, ugrp, rho)
    pcu = pd.Series(np.asarray(pa_c, float), index=df.index).groupby(unit).median()
    mtu = _col(df, "mt_mm").groupby(unit).median()
    smtu = (sig["sigma_mt"] if "sigma_mt" in sig else pd.Series(np.nan, index=df.index)).groupby(unit).mean()
    pri = out.attrs["priors"]
    pr_fl, pr_pa, k_grp = {}, {}, {}
    sinp = np.sin(np.radians(pcu.clip(lower=1.0)))
    for g in ugrp.unique():
        ui = ugrp.index[ugrp == g]
        pr_fl[g] = Prior(**{k: pri[f"{g}|fl"][k] for k in ("mu", "tau", "c")})
        pr_pa[g] = fit_group_prior(xu_pa[ui].to_numpy(), su_pa[ui].to_numpy(), "pa", method="ml", **pk)
        if k_mode == "group":
            kk = (xu_fl[ui] * sinp[ui] / mtu[ui]).replace([np.inf, -np.inf], np.nan).dropna()
            k_grp[g] = float(kk.median()) if len(kk) else k0
        elif k_mode == "k0":
            k_grp[g] = k0
        else:
            raise ValueError(k_mode)
    ku = ugrp.map(k_grp).astype(float)
    fl_tri = ku * mtu / sinp
    tau_pa = ugrp.map({g: pr_pa[g].tau for g in pr_pa}).astype(float)
    c_pa = ugrp.map({g: pr_pa[g].c for g in pr_pa}).astype(float)
    sd_pa = 1.0 / np.sqrt(1.0 / tau_pa ** 2 + 1.0 / (c_pa * su_pa ** 2))
    sd_pa = sd_pa.where(np.isfinite(sd_pa), tau_pa)
    cot = 1.0 / np.tan(np.radians(pcu.clip(lower=1.0)))
    rel_mt = ((smtu / mtu) ** 2).where(lambda v: np.isfinite(v), 0.0)
    s_eff = np.sqrt(s_logk ** 2 + (cot * np.radians(sd_pa)) ** 2 + rel_mt)
    base = out["fl_eb"].groupby(unit).first()
    a_floor = pd.Series(0.0, index=xu_fl.index)
    post = {}
    for u in xu_fl.index:
        g = ugrp[u]
        if mode == "cpl":
            post[u] = _post_fl_coupled(xu_fl[u], su_fl[u], pr_fl[g], fl_tri[u], s_eff[u], likelihood)
        elif mode == "joint":
            post[u] = joint_posterior(xu_pa[u], su_pa[u], xu_fl[u], su_fl[u], mtu[u], smtu[u], pr_pa[g], pr_fl[g],
                                      likelihood=likelihood, k0=float(ku[u]), s_logk=s_logk)[1]
        elif mode == "floor":
            x, mu = xu_fl[u], pr_fl[g].mu
            if not (np.isfinite(x) and np.isfinite(fl_tri[u]) and fl_tri[u] > 0) or abs(x - mu) < 1e-9:
                post[u] = base[u]
                continue
            a = math.exp(-0.5 * (math.log(x / fl_tri[u]) / s_logk) ** 2)
            a_floor[u] = a
            w = max((base[u] - mu) / (x - mu), a)
            post[u] = mu + w * (x - mu)
        else:
            raise ValueError(mode)
    val = unit.map(post).astype(float)
    out["fl_eb"] = val
    xu_r = unit.map(xu_fl).astype(float)
    d = xu_r - out["mu_fl"]
    out["w_fl"] = ((val - out["mu_fl"]) / d.where(d.abs() > 1e-9)).clip(-0.5, 1.5).fillna(1.0)
    out.loc[x_fl.isna(), "w_fl"] = 0.0
    out["fl_tri"] = unit.map(fl_tri).astype(float)
    out["s_eff"] = unit.map(s_eff).astype(float)
    out["k_used"] = unit.map(ku).astype(float)
    out["a_floor"] = unit.map(a_floor).astype(float)
    out["sd_pa_post"] = unit.map(sd_pa).astype(float)
    out.attrs["priors_pa_fl_coupled"] = {f"{g}|pa": dict(mu=p.mu, tau=p.tau, c=p.c) for g, p in pr_pa.items()}
    out.attrs["k_by_group"] = k_grp
    return out


# ==================================================================================================
# fixes5 additions (2026-10-21, critic4 fix 3). Additive: nothing above changes.
#  * fit_centre_slope(): leak-free predictive slope of log expert FL on a per-image log predictor z
#    (z = log(MT / sin PA_out)), within dataset, from labelled benchmark rows only.
#  * shrink_fl_centred(): the FL EB of shrink(cols=("fl",)) with an MT-aware prior centre per unit,
#    log c_u = log base + beta (z_u - median z) (level "group": the group / sub-group median level is kept),
#    or c_u = k0 exp(z_u) (level "k0": the expert coupling sets the level), clipped to base exp(+-clip_dev).
#    (tau, c) are those of the plain group fit, so beta = 0 reproduces shrink() exactly.
# ==================================================================================================
def fit_centre_slope(df: pd.DataFrame, z: pd.Series, truth_col: str = "fl_true", dataset_col: str = "dataset",
                     n_eff: dict | None = None, keep: pd.Series | None = None, lo: float = 0.0,
                     hi: float = 1.2) -> dict:
    """Weighted within-dataset OLS slope (through the dataset medians) of log(truth) on z.

    Rows: finite truth and z (and ``keep`` if given). Each dataset is centred on its own medians (a
    constant offset between datasets is a level, not a slope); dataset d gets total weight sqrt(n_eff_d)
    (n_eff = DEFAULT_NEFF: the dynamic video counts as 20 rows), as the error model. Returns the clipped
    slope ``beta`` plus the raw slope, the weighted R^2 and the per-dataset slopes / correlations / n."""
    y = np.log(_col(df, truth_col).where(lambda v: v > 0))
    zz = pd.to_numeric(pd.Series(np.asarray(z, float), index=df.index), errors="coerce")
    ok = y.notna() & np.isfinite(zz)
    if keep is not None:
        ok &= keep.reindex(df.index).fillna(False).astype(bool)
    ds = df[dataset_col].astype(str) if dataset_col in df else pd.Series("all", index=df.index)
    neff = DEFAULT_NEFF if n_eff is None else n_eff
    yc, zc, w, per = [], [], [], {}
    for d in sorted(ds[ok].unique()):
        m = ok & (ds == d)
        n = int(m.sum())
        if n < 5:
            continue
        a = (y[m] - y[m].median()).to_numpy(float)
        b = (zz[m] - zz[m].median()).to_numpy(float)
        ne = min(float(neff.get(d, n)), float(n))
        yc.append(a)
        zc.append(b)
        w.append(np.full(n, math.sqrt(ne) / n))
        sb = float(np.sum(b * b))
        per[d] = dict(n=n, slope=float(np.sum(a * b) / sb) if sb > 0 else float("nan"),
                      r=float(np.corrcoef(a, b)[0, 1]) if n > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan"))
    if not yc:
        return dict(beta=0.0, raw=float("nan"), r2=float("nan"), per_dataset=per)
    a, b, ww = np.concatenate(yc), np.concatenate(zc), np.concatenate(w)
    sb = float(np.sum(ww * b * b))
    raw = float(np.sum(ww * a * b) / sb) if sb > 0 else 0.0
    res = a - raw * b
    r2 = 1.0 - float(np.sum(ww * res * res)) / max(float(np.sum(ww * a * a)), 1e-12)
    return dict(beta=float(np.clip(raw, lo, hi)), raw=raw, r2=r2, per_dataset=per)


def shrink_fl_centred(df: pd.DataFrame, model: ErrorModel, z: pd.Series, beta: float = 0.0, level: str = "group",
                      k0: float | None = None, centre_key: pd.Series | None = None, clip_dev: float = 0.4,
                      group_col: str = "group", clip_col: str | None = "clip", likelihood: str = "laplace",
                      prior_kw: dict | None = None, sigma: pd.DataFrame | None = None,
                      rho: float | None = None) -> pd.DataFrame:
    """FL-only EB posteriors with an MT-aware prior centre (see the section header).

    z          : per-row log predictor (e.g. log(MT / sin PA_out)); NaN -> the plain base centre.
    centre_key : optional per-row sub-group key ('' = none). Rows sharing a key are centred on the median
                 of their own FL (the muscle cells; single-image units only), the rest on the group's
                 unit median; the z median is taken over the same set. (tau, c) stay those of the group fit.
    level      : "group" (log c = log base + beta (z - median z)) or "k0" (c = k0 exp(z); beta unused).
    Returns shrink(df, model, cols=("fl",), ...) with fl_eb, mu_fl (= the unit centre) and w_fl replaced,
    plus fl_centre_dev (log c - log base) per row; attrs["fl_centre"] records the settings."""
    pk = dict(prior_kw if prior_kw is not None else {"mu_mode": "median"})
    out = shrink(df, model, group_col=group_col, clip_col=clip_col, cols=("fl",), method="ml",
                 likelihood=likelihood, prior_kw=pk, sigma=sigma, rho=rho)
    sig = model.predict(df) if sigma is None else sigma
    unit = out["eb_unit"]
    grp = df[group_col].astype(str) if group_col in df else pd.Series("all", index=df.index)
    ugrp = grp.groupby(unit).first()
    x = _col(df, "fl_mm")
    s = sig["sigma_fl"].astype(float)
    xu, su, _ = _unit_values(x, s, unit, grp, ugrp, rho)
    zu = pd.to_numeric(pd.Series(np.asarray(z, float), index=df.index), errors="coerce").groupby(unit).median()
    key = (pd.Series("", index=df.index, dtype=object) if centre_key is None
           else centre_key.reindex(df.index).fillna("").astype(str))
    ku = key.groupby(unit).first()
    nrow = unit.value_counts()
    if ((ku != "") & (ku.index.map(nrow) > 1)).any():
        raise ValueError("centre_key sub-groups must not contain multi-frame (clip) units")
    kk = K0 if k0 is None else float(k0)
    pri = out.attrs["priors"]
    post, cen, dev = {}, {}, {}
    for g in ugrp.unique():
        ui = ugrp.index[ugrp == g]
        q = pri[f"{g}|fl"]
        fin = (np.isfinite(xu[ui]) & np.isfinite(su[ui]) & (su[ui] > 0)).to_numpy()
        for k in ku[ui].unique():
            sel = (ku[ui] == k).to_numpy()
            uk = ui[sel]
            if k == "":      # the group's own prior centre (median over its finite units, as shrink())
                base, uset = float(q["mu"]), ui[fin]
            else:            # sub-group (cell): median of its own FL estimates
                xk = xu[uk]
                base = float(np.median(xk[np.isfinite(xk)])) if np.isfinite(xk).any() else float(q["mu"])
                uset = ui[sel & fin]
            zs = zu.reindex(uset)
            zs = zs[np.isfinite(zs)]
            zm = float(np.median(zs)) if len(zs) else float("nan")
            for u in uk:
                if level == "k0":
                    d = math.log(kk) + float(zu[u]) - math.log(base) if np.isfinite(zu[u]) and base > 0 else 0.0
                elif level == "group":
                    d = beta * (float(zu[u]) - zm) if np.isfinite(zu[u]) and np.isfinite(zm) else 0.0
                else:
                    raise ValueError(level)
                d = float(np.clip(d, -clip_dev, clip_dev))
                cu = base * math.exp(d)
                post[u] = _post_1d(float(xu[u]), float(su[u]), Prior(mu=cu, tau=float(q["tau"]), c=float(q["c"])),
                                   likelihood)
                cen[u], dev[u] = cu, d
    val = unit.map(post).astype(float)
    mu_r = unit.map(cen).astype(float)
    out["fl_eb"] = val
    out["mu_fl"] = mu_r
    out["fl_centre_dev"] = unit.map(dev).astype(float)
    dd = unit.map(xu).astype(float) - mu_r
    out["w_fl"] = ((val - mu_r) / dd.where(dd.abs() > 1e-9)).clip(-0.5, 1.5).fillna(1.0)
    out.loc[x.isna(), "w_fl"] = 0.0
    out.attrs["fl_centre"] = dict(beta=float(beta), level=level, k0=kk, clip_dev=clip_dev)
    return out
