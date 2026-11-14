"""Read the burned-in muscle label of every UMUD test image and build muscle-aware priors.

Automated metadata reading (like the scale bar): no anatomy is hand-labelled. See
``src/umud/muscle_label.py`` for the OCR reader.

Subcommands (outputs in artifacts/eval_v4/muscle/)
--------------------------------------------------
ocr     tesseract on every test image, one at a time (~5 min): full-frame words of both renderings
        -> ocr_words.csv; targeted line re-reads for weak reads -> ocr_line_words.csv. Resumable.
read    parse the cached words -> ocr_readings.csv (muscle, side, confidence, method, raw text, bbox,
        layout, preset, depth label).
infer   label-free completion of the unlabelled images -> test_muscle.csv
        (image_id, muscle, side, confidence, method, + group / ocr columns) and infer_eval.json
        (leave-one-out accuracy of every inference rule on the OCR-labelled images).
sheets  contact sheets of the label crops with the prediction (and full thumbnails of unknowns)
        -> sheets/*.png, for the visual check.
priors  scanner x muscle medians / spreads of the C4 per-image estimates, mixing of the current
        scanner-only group medians, and the literature reference table -> priors_*.csv / .json.
all     read + infer + sheets + priors (OCR cache must exist; run ``ocr`` first).
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umud import muscle_label as ml  # noqa: E402

TEST = ROOT / "data/umud-challenge-muscle-architecture-in-ultrasound-data/test_images_v2/test_set_v2"
OUT = ROOT / "artifacts/eval_v4/muscle"
PER_IMAGE = ROOT / "artifacts/candidates_v3/per_image_C4.csv"
WORD_COLS = ["image_id", "variant", "text", "conf", "x", "y", "w", "h"]


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # macOS: bytes


def test_ids() -> list[str]:
    return sorted(p.name for p in TEST.iterdir() if p.suffix.lower() in (".tif", ".png"))


def pipeline_groups() -> pd.Series:
    """Scanner group of every test image as used by the v3 pipeline (per_image_C4.csv)."""
    return pd.read_csv(PER_IMAGE, usecols=["image_id", "group", "clip"]).set_index("image_id")


# ------------------------------------------------------------------------------------------- ocr

def cmd_ocr(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fw, fl = OUT / "ocr_words.csv", OUT / "ocr_line_words.csv"
    done = set()
    if fw.exists() and not args.force:
        done = set(pd.read_csv(fw, usecols=["image_id"]).image_id)
    elif fw.exists():
        fw.unlink(); fl.unlink(missing_ok=True)
    ids = [i for i in test_ids() if i not in done]
    t0 = time.time()
    for k, iid in enumerate(ids):
        g = ml.load_gray(TEST / iid)
        H, W = g.shape
        words = ml.ocr_image(g)
        lay = ml.layout_of(iid, W, H, words, g)
        _, line = ml.read_words(g, words, layout=lay)
        for df, f in ((words, fw), (line, fl)):
            df = df.copy()
            df.insert(0, "image_id", iid)
            if not len(df):                           # keep a marker row so the image counts as done
                df = pd.DataFrame([[iid, "none", "", 0.0, 0, 0, 0, 0]], columns=WORD_COLS)
            df[WORD_COLS].to_csv(f, mode="a", header=not f.exists(), index=False)
        if k % 25 == 0:
            print(f"{k}/{len(ids)} {iid} {time.time() - t0:.0f}s rss {rss_gb():.2f} GB", flush=True)
    print(f"ocr done: {len(ids)} images in {time.time() - t0:.0f}s, peak rss {rss_gb():.2f} GB")


def _words(f: Path) -> dict[str, pd.DataFrame]:
    d = pd.read_csv(f, keep_default_na=False, dtype={"text": str})
    d = d[d.variant != "none"]
    for c in ("conf", "x", "y", "w", "h"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return {k: v.drop(columns="image_id").reset_index(drop=True) for k, v in d.groupby("image_id")}


# ------------------------------------------------------------------------------------------ read

def cmd_read(args) -> pd.DataFrame:
    W_all, L_all = _words(OUT / "ocr_words.csv"), _words(OUT / "ocr_line_words.csv")
    empty = pd.DataFrame(columns=["variant", "text", "conf", "x", "y", "w", "h"])
    grp = pipeline_groups()
    rows = []
    for iid in test_ids():
        g = ml.load_gray(TEST / iid)
        Hd, Wd = g.shape
        words = W_all.get(iid, empty)
        lay = ml.layout_of(iid, Wd, Hd, words, g)
        r, _ = ml.read_words(None, words, layout=lay, line_words=L_all.get(iid, empty))
        r.update(layout=lay, **ml.panel_metadata(words, lay))
        r["image_id"] = iid
        rows.append(r)
    R = pd.DataFrame(rows).set_index("image_id")
    R["group"] = grp.reindex(R.index)["group"]
    R["clip"] = grp.reindex(R.index)["clip"]
    R["bbox"] = R["bbox"].map(lambda b: "" if b is None else " ".join(str(z) for z in b))
    R.to_csv(OUT / "ocr_readings.csv")
    print(pd.crosstab(R.group, R.muscle))
    return R


# ----------------------------------------------------------------------------------------- infer
# Label-free completion. Rules, in order (first that fires wins):
#   1. OCR reading with confidence >= OCR_MIN                               -> method "ocr:..."
#   2. same-scanner contiguous ID block whose OCR labels are unanimous (>= 95 %, n >= 10)
#      -> that muscle, confidence = Laplace (k + 1) / (n + 2)                -> "block_unanimous"
#   3. within-scanner Gaussian naive Bayes GM-vs-rest on the pipeline's own raw per-image estimates
#      (C4 pa_deg, log fl_mm, mt_mm), anchors = the OCR-labelled images of the same scanner:
#      p_GM >= P_DECIDE -> GM ("gnb_gm"); p_GM <= 1 - P_DECIDE -> muscle unknown, muscle_group QF
#      (VL or RF: not separable on these features, LOO RF recall ~0 on Siemens)
#      a Siemens 'Tri Surae' preset adds GM evidence (likelihood ratio PRESET_LR) -> "+preset"
#   4. otherwise unknown (all Telemed and legacy images: no text, and the cross-scanner classifier
#      does not transfer: 63-80 % accuracy Siemens <-> Lumify).

OCR_MIN = 0.5
P_DECIDE = 0.9
PRESET_LR = 4.0
GNB_FEATS = ("pa_deg", "log_fl", "mt_mm")
VAR_FLOOR = 0.1


def _num(iid: str) -> int:
    return int(iid.split("_")[1].split(".")[0])


def id_blocks(groups: pd.Series) -> pd.Series:
    """Maximal runs of consecutive image numbers with the same scanner group (label-free)."""
    s = groups.sort_index(key=lambda ix: ix.map(_num))
    b = (s != s.shift()).cumsum()
    return pd.Series([f"{g}#{k}" for g, k in zip(s, b)], index=s.index).reindex(groups.index)


def gnb_proba(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, classes) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z, Zt = (Xtr - mu) / sd, (Xte - mu) / sd
    ll = []
    for c in classes:
        A = Z[ytr == c]
        m, v = A.mean(0), A.var(0) + VAR_FLOOR
        ll.append(-0.5 * (((Zt - m) ** 2) / v + np.log(v)).sum(1) + np.log(len(A) / len(Z)))
    ll = np.array(ll).T
    p = np.exp(ll - ll.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def boot_ci(x: np.ndarray, B: int = 4000, seed: int = 0, stat=np.mean) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    v = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def feature_table() -> pd.DataFrame:
    P = pd.read_csv(PER_IMAGE).set_index("image_id")
    F = P[["pa_deg", "fl_mm", "mt_mm"]].copy()
    F["log_fl"] = np.log(F.fl_mm)
    return F


def cmd_infer(args) -> pd.DataFrame:
    R = pd.read_csv(OUT / "ocr_readings.csv", keep_default_na=False).set_index("image_id")
    for c in ("confidence", "support_VL", "support_GM", "support_RF", "clip"):
        R[c] = pd.to_numeric(R[c], errors="coerce")
    F = feature_table()
    R["block"] = id_blocks(R.group)
    ocr_ok = (R.muscle != "unknown") & (R.confidence >= OCR_MIN)
    out = pd.DataFrame(index=R.index)
    out["muscle"], out["side"], out["confidence"], out["method"] = "unknown", R.side, 0.0, "none"
    for m in ml.MUSCLES:
        out[f"p_{m}"] = np.nan
    sup = R[[f"support_{m}" for m in ml.MUSCLES]].to_numpy(float)
    tot = sup.sum(1, keepdims=True)
    # 1. OCR
    for j, m in enumerate(ml.MUSCLES):
        out.loc[ocr_ok, f"p_{m}"] = np.round((sup[ocr_ok.to_numpy(), j] / tot[ocr_ok.to_numpy(), 0])
                                             * R.confidence[ocr_ok] + (1 - R.confidence[ocr_ok]) / 3, 3)
    out.loc[ocr_ok, "muscle"] = R.muscle[ocr_ok]
    out.loc[ocr_ok, "confidence"] = R.confidence[ocr_ok]
    out.loc[ocr_ok, "method"] = R.method[ocr_ok]
    ev: dict = {"rules": {}, "params": dict(OCR_MIN=OCR_MIN, P_DECIDE=P_DECIDE, PRESET_LR=PRESET_LR,
                                            GNB_FEATS=list(GNB_FEATS), VAR_FLOOR=VAR_FLOOR)}
    # 2. unanimous blocks
    for blk, idx in R.groupby("block").groups.items():
        lab = R.loc[idx][ocr_ok.loc[idx]]
        n = len(lab)
        if n < 10:
            continue
        vc = lab.muscle.value_counts()
        k, top = int(vc.iloc[0]), vc.index[0]
        if k / n < 0.95:
            continue
        conf = (k + 1) / (n + 2)
        todo = [i for i in idx if out.at[i, "method"] == "none"]
        for i in todo:
            out.at[i, "muscle"], out.at[i, "confidence"], out.at[i, "method"] = top, round(conf, 3), "block_unanimous"
            for m in ml.MUSCLES:
                out.at[i, f"p_{m}"] = round(conf if m == top else (1 - conf) / 2, 3)
        ev["rules"][f"block_unanimous:{blk}"] = dict(n_labelled=n, k_top=k, muscle=top, laplace_conf=round(conf, 3),
                                                     assigned=todo)
    # 3. within-scanner GNB, GM vs rest
    for grp in sorted(R.group[ocr_ok].unique()):
        gi = R.index[R.group == grp]
        lab = [i for i in gi if ocr_ok[i]]
        X = F.loc[lab, list(GNB_FEATS)].to_numpy(float)
        y = np.where(R.muscle[lab] == "GM", "GM", "nonGM")
        ok = np.isfinite(X).all(1)
        X, y, lab_ok = X[ok], y[ok], np.array(lab)[ok]
        if (y == "GM").sum() < 5 or (y != "GM").sum() < 5:
            continue
        cls = ["GM", "nonGM"]
        Ploo = np.array([gnb_proba(np.delete(X, i, 0), np.delete(y, i), X[i:i + 1], cls)[0] for i in range(len(X))])
        pg = Ploo[:, 0]
        dec = (pg >= P_DECIDE) | (pg <= 1 - P_DECIDE)
        pred = np.where(pg >= 0.5, "GM", "nonGM")
        corr = pred == y
        e = dict(n=int(len(y)), n_GM=int((y == "GM").sum()), loo_acc=round(float(corr.mean()), 3),
                 loo_acc_ci95=[round(v, 3) for v in boot_ci(corr.astype(float))],
                 decided=int(dec.sum()), decided_acc=round(float(corr[dec].mean()), 3) if dec.any() else None,
                 decided_acc_ci95=[round(v, 3) for v in boot_ci(corr[dec].astype(float))] if dec.any() else None,
                 gm_recall=round(float(corr[y == "GM"].mean()), 3),
                 gm_precision=round(float((y[pred == "GM"] == "GM").mean()), 3) if (pred == "GM").any() else None,
                 errors=[dict(image_id=str(i), ocr=str(R.muscle[i]), p_GM=round(float(p), 3))
                         for i, p, c in zip(lab_ok, pg, corr) if not c])
        # 3-class (reported, not used)
        y3 = R.muscle[lab_ok].to_numpy()
        P3 = np.array([gnb_proba(np.delete(X, i, 0), np.delete(y3, i), X[i:i + 1], list(ml.MUSCLES))[0]
                       for i in range(len(X))])
        pr3 = np.array(ml.MUSCLES)[P3.argmax(1)]
        e["three_class_loo_acc"] = round(float((pr3 == y3).mean()), 3)
        e["three_class_recall"] = {m: round(float((pr3[y3 == m] == m).mean()), 3) for m in ml.MUSCLES if (y3 == m).any()}
        ev["rules"][f"gnb_gm:{grp}"] = e
        # apply to the undecided images of this scanner
        todo = [i for i in gi if out.at[i, "method"] == "none" and np.isfinite(F.loc[i, list(GNB_FEATS)].to_numpy(float)).all()]
        if not todo:
            continue
        Pt = gnb_proba(X, y, F.loc[todo, list(GNB_FEATS)].to_numpy(float), cls)[:, 0]
        for i, p in zip(todo, Pt):
            meth = "gnb_gm"
            if "surae" in str(R.at[i, "preset"]).lower():
                o = p / max(1e-9, 1 - p) * PRESET_LR
                p, meth = o / (1 + o), "gnb_gm+preset"
            blk = R.at[i, "block"]
            bl = R.index[(R.block == blk) & ocr_ok]
            nvl, nrf = (R.muscle[bl] == "VL").sum() + 1, (R.muscle[bl] == "RF").sum() + 1
            out.at[i, "p_GM"] = round(float(p), 3)
            out.at[i, "p_VL"] = round(float((1 - p) * nvl / (nvl + nrf)), 3)
            out.at[i, "p_RF"] = round(float((1 - p) * nrf / (nvl + nrf)), 3)
            if p >= P_DECIDE:
                out.at[i, "muscle"], out.at[i, "confidence"], out.at[i, "method"] = "GM", round(float(p), 3), meth
            elif p <= 1 - P_DECIDE:
                out.at[i, "confidence"], out.at[i, "method"] = round(float(1 - p), 3), meth.replace("gnb_gm", "gnb_qf")
            else:
                out.at[i, "method"] = meth.replace("gnb_gm", "gnb_undecided")
    # same-session adjacency: are consecutive IDs more often the same muscle than chance?
    adj = {}
    rng = np.random.default_rng(0)
    for blk, idx in R.groupby("block").groups.items():
        ids = sorted(idx, key=_num)
        labs = [R.muscle[i] if ocr_ok[i] else None for i in ids]
        pairs = [(a, b) for a, b in zip(labs[:-1], labs[1:]) if a is not None and b is not None]
        if len(pairs) < 10 or len({x for x in labs if x}) < 2:
            continue
        obs = float(np.mean([a == b for a, b in pairs]))
        known = [x for x in labs if x is not None]
        perm = []
        for _ in range(2000):
            q = list(rng.permutation(known))
            it = iter(q)
            lp = [next(it) if x is not None else None for x in labs]
            pp = [(a, b) for a, b in zip(lp[:-1], lp[1:]) if a is not None and b is not None]
            perm.append(np.mean([a == b for a, b in pp]))
        adj[blk] = dict(n_pairs=len(pairs), same_muscle_rate=round(obs, 3),
                        permutation_mean=round(float(np.mean(perm)), 3),
                        p_value_ge=round(float(np.mean(np.array(perm) >= obs)), 4))
    ev["adjacency_test"] = adj
    out["muscle_group"] = np.select([out.muscle == "GM", out.muscle.isin(["VL", "RF"]),
                                     out.method.str.startswith("gnb_qf")], ["GM", "QF", "QF"], "unknown")
    out.loc[out.method == "none", "method"] = np.where(R.layout[out.method == "none"].isin(["telemed", "legacy"]),
                                                       "no_text_layout", "no_label_found")
    out["side"] = out.side.replace("", "unknown")
    out["group"], out["clip"], out["block"] = R["group"], R["clip"], R["block"]
    out["ocr_muscle"], out["ocr_confidence"], out["ocr_raw"], out["preset"] = R.muscle, R.confidence, R.raw, R.preset
    out = out.sort_index(key=lambda ix: ix.map(_num))
    out.index.name = "image_id"
    out.to_csv(OUT / "test_muscle.csv")
    # counts
    cnt = out.groupby(["group", "muscle"]).size().unstack(fill_value=0)
    cnt_m = out.groupby(["group", "method"]).size().unstack(fill_value=0)
    ev["counts_muscle"] = {g: {k: int(v) for k, v in r.items()} for g, r in cnt.iterrows()}
    ev["counts_method"] = {g: {k: int(v) for k, v in r.items() if v} for g, r in cnt_m.iterrows()}
    ev["unknown_per_group"] = {g: int(v) for g, v in (out.muscle == "unknown").groupby(out.group).sum().items()}
    ev["unknown_per_group_after_QF"] = {g: int(v) for g, v in (out.muscle_group == "unknown").groupby(out.group).sum().items()}
    (OUT / "infer_eval.json").write_text(json.dumps(ev, indent=1, default=str))
    print(cnt)
    print(cnt_m)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "errors"} for k, v in ev["rules"].items()}, indent=1, default=str))
    return out


# ---------------------------------------------------------------------------------------- sheets

def cmd_sheets(args) -> None:
    import cv2
    T = pd.read_csv(OUT / "test_muscle.csv", keep_default_na=False).set_index("image_id")
    R = pd.read_csv(OUT / "ocr_readings.csv", keep_default_na=False).set_index("image_id")
    SD = OUT / "sheets"
    SD.mkdir(exist_ok=True)
    for f in SD.glob("*.png"):
        f.unlink()
    col = {"VL": (80, 220, 80), "GM": (60, 160, 255), "RF": (255, 120, 220), "unknown": (160, 160, 160)}

    def label_bar(text, w, c):
        bar = np.full((18, w, 3), 45, np.uint8)
        cv2.putText(bar, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
        return bar

    def grid(tiles, cols, name):
        if not tiles:
            return
        for k in range(0, len(tiles), cols * 8):
            ch = tiles[k:k + cols * 8]
            h = max(t.shape[0] for t in ch)
            w = max(t.shape[1] for t in ch)
            ch = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + 3, 0, w - t.shape[1] + 3, cv2.BORDER_CONSTANT,
                                     value=(25, 25, 25)) for t in ch]
            while len(ch) % cols:
                ch.append(np.full_like(ch[0], 25))
            img = np.concatenate([np.concatenate(ch[i:i + cols], 1) for i in range(0, len(ch), cols)], 0)
            cv2.imwrite(str(SD / f"{name}_{k // (cols * 8)}.png"), img)

    crops, fulls = {}, {}
    for iid, t in T.iterrows():
        lay = R.at[iid, "layout"]
        if lay not in ("siemens", "lumify"):
            continue
        g = ml.load_gray(TEST / iid)
        txt = f"{iid[4:9]} {t.muscle} {t.side} {float(t.confidence):.2f} {t.method}"
        c = col.get(t.muscle, col["unknown"])
        bb = R.at[iid, "bbox"]
        if t.method.startswith("ocr") and bb:
            x, y, w, h = (float(v) for v in bb.split())
            x0, y0 = int(max(0, x - 40)), int(max(0, y - 12))
            x1, y1 = int(min(g.shape[1], x + w + 60)), int(min(g.shape[0], y + h + 12))
            cr = cv2.resize(g[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cr = cv2.cvtColor(cr, cv2.COLOR_GRAY2BGR)
            cr = cv2.copyMakeBorder(cr, 0, 0, 0, max(0, 330 - cr.shape[1]), cv2.BORDER_CONSTANT, value=(0, 0, 0))
            crops.setdefault(lay, []).append(np.concatenate([label_bar(txt + f" '{R.at[iid, 'raw']}'", cr.shape[1], c), cr], 0))
        else:
            th = cv2.resize(g, (480, int(g.shape[0] * 480 / g.shape[1])), interpolation=cv2.INTER_AREA)
            th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            fulls.setdefault(lay, []).append(np.concatenate([label_bar(txt, 480, c), th], 0))
    for lay, tl in crops.items():
        grid(tl, 4, f"ocr_{lay}")
    for lay, tl in fulls.items():
        grid(tl, 3, f"nolabel_{lay}")
    # one frame per Telemed clip + every legacy image: the no-text layouts
    tiles = []
    seen = set()
    for iid, t in T.iterrows():
        if R.at[iid, "layout"] not in ("telemed", "legacy"):
            continue
        key = (t["group"], int(float(t["clip"]))) if float(t["clip"]) >= 0 else iid
        if key in seen:
            continue
        seen.add(key)
        g = ml.load_gray(TEST / iid)
        th = cv2.cvtColor(cv2.resize(g, (300, int(g.shape[0] * 300 / g.shape[1])), interpolation=cv2.INTER_AREA),
                          cv2.COLOR_GRAY2BGR)
        tiles.append(np.concatenate([label_bar(f"{iid[4:9]} {t.group[:12]} {t.method}", 300, col["unknown"]), th], 0))
    grid(tiles, 5, "notext_telemed_legacy")
    print("sheets:", sorted(p.name for p in SD.glob("*.png")))


# ---------------------------------------------------------------------------------------- priors
# Literature reference values (numbers as reported in the cited source; see muscle_priors.json).
LITERATURE = [
    dict(muscle="VL", population="healthy", age="19.9-20.6 y (31 adults)", n=31, mt_mm=22.8, mt_sd=3.2,
         pa_deg=13.2, pa_sd=3.2, fl_mm=None, condition="mid-thigh site, rest",
         source="Blazevich, Gill & Zhou 2006 J Anat 209:289 (Table 2, middle site)",
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2100333/"),
    dict(muscle="RF", population="healthy", age="19.9-20.6 y (31 adults)", n=31, mt_mm=24.0, mt_sd=4.8,
         pa_deg=13.4, pa_sd=3.6, fl_mm=None, condition="mid-thigh site, rest",
         source="Blazevich, Gill & Zhou 2006 J Anat 209:289 (Table 2, middle site)",
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2100333/"),
    dict(muscle="VL", population="healthy (TD, untrained)", age="20.3 y", n=None, mt_mm=24.2, mt_sd=6.3,
         pa_deg=18.3, pa_sd=3.3, fl_mm=78.1, fl_sd=12.7, condition="dominant leg; Siemens ACUSON Juniper (Basel)",
         source="Minghetti, Roth, ..., Ritsche, Keller 2026 (Univ. Basel), Table 4",
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13281589/"),
    dict(muscle="VL", population="healthy (TD, trained)", age="19.8 y", n=None, mt_mm=27.6, mt_sd=4.9,
         pa_deg=22.8, pa_sd=2.6, fl_mm=79.4, fl_sd=12.1, condition="dominant leg; Siemens ACUSON Juniper (Basel)",
         source="Minghetti et al. 2026, Table 4", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13281589/"),
    dict(muscle="VL", population="cerebral palsy (untrained)", age="19.9 y", n=None, mt_mm=24.5, mt_sd=5.5,
         pa_deg=18.3, pa_sd=3.3, fl_mm=80.6, fl_sd=23.7, condition="non-affected leg; Siemens ACUSON Juniper (Basel)",
         source="Minghetti et al. 2026, Table 4", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13281589/"),
    dict(muscle="VL", population="cerebral palsy (untrained)", age="19.9 y", n=None, mt_mm=24.3, mt_sd=4.4,
         pa_deg=15.8, pa_sd=3.4, fl_mm=84.4, fl_sd=22.4, condition="affected leg; Siemens ACUSON Juniper (Basel)",
         source="Minghetti et al. 2026, Table 4", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13281589/"),
    dict(muscle="VL", population="cerebral palsy (trained)", age="19.8 y", n=None, mt_mm=30.3, mt_sd=8.0,
         pa_deg=23.8, pa_sd=3.8, fl_mm=77.5, fl_sd=5.8, condition="non-affected leg; Siemens ACUSON Juniper (Basel)",
         source="Minghetti et al. 2026, Table 4", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13281589/"),
    dict(muscle="VL", population="cerebral palsy", age="12 +- 3 y (children/adolescents)", n=18, mt_mm=12.2, mt_sd=4.4,
         pa_deg=10.25, pa_sd=3.38, fl_mm=71.9, fl_sd=22.4, condition="mid-thigh, resting knee angle, supine",
         source="Moreau, Teefey & Damiano 2009 DMCN (Table II)", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2771733/"),
    dict(muscle="VL", population="healthy (TD)", age="12 y 4 m (7-20 y)", n=12, mt_mm=17.8, mt_sd=6.1,
         pa_deg=13.54, pa_sd=2.94, fl_mm=75.2, fl_sd=12.0, condition="mid-thigh, resting knee angle, supine",
         source="Moreau, Teefey & Damiano 2009 DMCN (Table II)", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2771733/"),
    dict(muscle="RF", population="cerebral palsy", age="12 +- 3 y (children/adolescents)", n=18, mt_mm=13.6, mt_sd=3.8,
         pa_deg=11.58, pa_sd=3.18, fl_mm=70.8, fl_sd=21.4, condition="50% ASIS-patella, resting knee angle",
         source="Moreau, Teefey & Damiano 2009 DMCN (Table II)", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2771733/"),
    dict(muscle="RF", population="healthy (TD)", age="12 y 4 m (7-20 y)", n=12, mt_mm=20.1, mt_sd=3.0,
         pa_deg=12.51, pa_sd=3.36, fl_mm=97.5, fl_sd=22.9, condition="50% ASIS-patella, resting knee angle",
         source="Moreau, Teefey & Damiano 2009 DMCN (Table II)", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2771733/"),
    dict(muscle="RF", population="cerebral palsy", age="19.3 +- 7.8 y (10-37)", n=26, mt_mm=17.5, mt_sd=4.3,
         pa_deg=None, fl_mm=78.4, fl_sd=23.7, condition="GE LOGIQe", source="Pontiff, Oral & Moreau 2026 Clin Biomech (Table 2)",
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13483056/"),
    dict(muscle="VL", population="cerebral palsy", age="19.3 +- 7.8 y (10-37)", n=26, mt_mm=16.3, mt_sd=3.1,
         pa_deg=None, fl_mm=66.5, fl_sd=12.2, condition="GE LOGIQe", source="Pontiff, Oral & Moreau 2026 Clin Biomech (Table 2)",
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13483056/"),
    dict(muscle="GM", population="cerebral palsy", age="19.3 +- 7.8 y (10-37)", n=26, mt_mm=11.2, mt_sd=2.8,
         pa_deg=None, fl_mm=32.7, fl_sd=12.3, condition="GE LOGIQe", source="Pontiff, Oral & Moreau 2026 Clin Biomech (Table 2)",
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13483056/"),
    dict(muscle="GM", population="healthy", age="22.5 +- 2.8 y", n=10, mt_mm=18.15, mt_sd=2.0, pa_deg=16.21, pa_sd=2.8,
         fl_mm=61.9, fl_sd=9.4, condition="MT/PA at the ankle angle of peak isometric moment; FL at 105 deg ankle",
         source="Gallinger et al. 2025 Front Sports Act Living (Tables 1/3, Discussion)",
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12106427/"),
    dict(muscle="GM", population="cerebral palsy (GMFCS I-II)", age="22.8 +- 6.0 y", n=16, mt_mm=13.6, mt_sd=2.2,
         pa_deg=18.8, pa_sd=4.0, fl_mm=None, condition="pre-training CP-T (MT 13.42, PA 20.08) and CP-NT (13.79, 17.59) averaged; recorded at the ankle angle of peak isometric moment",
         source="Gallinger et al. 2025 Front Sports Act Living", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12106427/"),
    dict(muscle="GM", population="healthy", age="young adults", n=None, mt_mm=None, pa_deg=15.8, fl_mm=57.0,
         condition="rest, ankle 90 deg (neutral); at 150 deg (plantarflexed): PA 27.7, FL 34.0",
         source="Narici et al. 1996 J Physiol 496:287 (abstract)", url="https://doi.org/10.1113/jphysiol.1996.sp021685"),
    dict(muscle="GM", population="healthy", age="6 men", n=6, mt_mm=None, pa_deg=None, fl_mm=59.0,
         condition="passive, knee 0 deg / ankle -15 deg (FL); MG angles 22-67 deg over positions",
         source="Kawakami, Ichinose & Fukunaga 1998 J Appl Physiol (abstract)", url="https://doi.org/10.1152/jappl.1998.85.2.398"),
    dict(muscle="GM", population="cerebral palsy (adult men)", age="adults", n=11, mt_mm=None, pa_deg=None, fl_mm=None,
         condition="relative to controls: FL +28 %, PA -41 %, PCSA -47 % (MVIC)",
         source="Hussain, Onambele, Williams & Morse 2017 Muscle Nerve 56:298 (abstract)",
         url="https://e-space.mmu.ac.uk/617788"),
]


def robust(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return dict(n=0)
    q1, q3 = np.percentile(x, [25, 75])
    return dict(n=int(len(x)), median=round(float(x.median()), 2), q1=round(float(q1), 2), q3=round(float(q3), 2),
                rsd=round(float(1.4826 * np.median(np.abs(x - x.median()))), 2), mean=round(float(x.mean()), 2),
                sd=round(float(x.std()), 2) if len(x) > 1 else None)


def cmd_priors(args) -> None:
    T = pd.read_csv(OUT / "test_muscle.csv", keep_default_na=False).set_index("image_id")
    P = pd.read_csv(PER_IMAGE).set_index("image_id")
    D = P.join(T[["muscle", "muscle_group", "method", "confidence"]].rename(columns={"confidence": "m_conf"}))
    D["src"] = np.where(D.method.str.startswith("ocr") | (D.method == "block_unanimous"), "label", "inferred")
    rows = []
    comps = dict(pa=("pa_deg", "sub_pa_deg"), fl=("fl_mm", "sub_fl_mm"), mt=("mt_mm", "sub_mt_mm"))
    for (g, m), d in D.groupby(["group", "muscle"]):
        for subset, dd in (("label", d[d.src == "label"]), ("all", d)):
            if not len(dd) or (subset == "all" and len(dd) == len(d[d.src == "label"])):
                continue
            r = dict(source="ours_C4", scanner=g, muscle=m, subset=subset, n=len(dd))
            for c, (raw, fin) in comps.items():
                for kind, col_ in (("raw", raw), ("final", fin)):
                    st = robust(dd[col_])
                    for k in ("median", "q1", "q3", "rsd"):
                        r[f"{c}_{kind}_{k}"] = st.get(k)
                r[f"{c}_pull_mean"] = round(float((dd[fin] - dd[raw]).mean()), 2)
            rows.append(r)
    for g, d in D.groupby("group"):
        r = dict(source="ours_C4", scanner=g, muscle="ALL", subset="all", n=len(d))
        for c, (raw, fin) in comps.items():
            for kind, col_ in (("raw", raw), ("final", fin)):
                st = robust(d[col_])
                for k in ("median", "q1", "q3", "rsd"):
                    r[f"{c}_{kind}_{k}"] = st.get(k)
            r[f"{c}_pull_mean"] = round(float((d[fin] - d[raw]).mean()), 2)
        rows.append(r)
    S = pd.DataFrame(rows)
    # expert-label benchmarks we hold locally (VL-244 = UMUD ExpDrawnData young/old, Esaote)
    vl = pd.read_csv(ROOT / "artifacts/candidates/bench/vl_labels.csv")
    vl["cohort"] = vl.key.str.split("__").str[0]
    for coh, d in vl.groupby("cohort"):
        r = dict(source="expert_UMUD_ExpDrawnData_VL", scanner="esaote_mylab70", muscle="VL", subset=coh, n=len(d))
        for c, col_ in (("pa", "pa_deg"), ("fl", "fl_mm"), ("mt", "mt_lines_mm")):
            st = robust(d[col_])
            for k in ("median", "q1", "q3", "rsd"):
                r[f"{c}_raw_{k}"] = st.get(k)
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "priors_scanner_muscle.csv", index=False)

    # mixing: what the scanner-only centre does to each muscle, and the muscle-aware counterfactual
    mix = []
    lab = D[D.src == "label"]
    cf = pd.DataFrame(index=D.index, columns=["d_pa", "d_fl"], dtype=float).fillna(0.0)
    for g, d in D.groupby("group"):
        c_pa = float(d.pa_deg.median())                 # C2/C4 PA centre (confidence rule)
        c_fl = float(d.mu_fl.median())                  # C4 EB FL centre (plain median, per group)
        for m, dm in lab[lab.group == g].groupby("muscle"):
            med_pa, med_fl, med_mt = float(dm.pa_deg.median()), float(dm.fl_mm.median()), float(dm.mt_mm.median())
            # bootstrap over the scanner group's images (muscle subset and group median from the same resample)
            rng = np.random.default_rng(1)
            is_m = (d.index.isin(dm.index)).astype(bool)
            pa_a, fl_a = d.pa_deg.to_numpy(float), d.fl_mm.to_numpy(float)
            mu_a = d.mu_fl.to_numpy(float)
            bpa, bfl = [], []
            for _ in range(2000):
                ix = rng.integers(0, len(d), len(d))
                sel = ix[is_m[ix]]
                if len(sel) < 2:
                    continue
                bpa.append(float(np.median(pa_a[sel]) - np.median(pa_a[ix])))
                bfl.append(float(np.median(fl_a[sel]) - np.median(mu_a[ix])))
            k_raw = float((dm.fl_mm * np.sin(np.radians(dm.pa_deg)) / dm.mt_mm).median())
            k_fin = float((dm.sub_fl_mm * np.sin(np.radians(dm.sub_pa_deg)) / dm.sub_mt_mm).median())
            mix.append(dict(scanner=g, muscle=m, n=len(dm), n_group=len(d),
                            group_centre_pa=round(c_pa, 2), muscle_median_pa_raw=round(med_pa, 2),
                            delta_pa=round(med_pa - c_pa, 2),
                            delta_pa_ci95=[round(float(np.percentile(bpa, 2.5)), 2), round(float(np.percentile(bpa, 97.5)), 2)],
                            mean_pull_pa_C4=round(float((dm.sub_pa_deg - dm.pa_deg).mean()), 2),
                            group_centre_fl=round(c_fl, 1), muscle_median_fl_raw=round(med_fl, 1),
                            delta_fl=round(med_fl - c_fl, 1),
                            delta_fl_ci95=[round(float(np.percentile(bfl, 2.5)), 1), round(float(np.percentile(bfl, 97.5)), 1)],
                            mean_pull_fl_C4=round(float((dm.sub_fl_mm - dm.fl_mm).mean()), 1),
                            muscle_median_mt=round(med_mt, 2), group_median_mt=round(float(d.mt_mm.median()), 2),
                            final_median_pa_C4=round(float(dm.sub_pa_deg.median()), 2),
                            final_median_fl_C4=round(float(dm.sub_fl_mm.median()), 1),
                            k_raw=round(k_raw, 3), k_final_C4=round(k_fin, 3)))
            if len(dm) >= 8:
                ids = dm.index
                cf.loc[ids, "d_pa"] = (1 - D.loc[ids, "w_pa"]) * (med_pa - c_pa)
                cf.loc[ids, "d_fl"] = (1 - D.loc[ids, "w_fl"]) * (med_fl - c_fl)
    M = pd.DataFrame(mix)
    M.to_csv(OUT / "priors_mixing.csv", index=False)
    dist = dict(pa=float(cf.d_pa.abs().mean() / 6), fl=float(cf.d_fl.abs().mean() / 12))
    dist["official_distance_C4_vs_muscle_centre_counterfactual"] = round((dist["pa"] + dist["fl"]) / 3, 4)
    dist["pa"], dist["fl"] = round(dist["pa"], 4), round(dist["fl"], 4)
    dist["by_group"] = {g: dict(n_moved=int((d.d_pa != 0).sum()), mean_abs_d_pa=round(float(d.d_pa.abs().mean()), 3),
                                mean_abs_d_fl=round(float(d.d_fl.abs().mean()), 3))
                        for g, d in cf.join(D.group).groupby("group")}
    out = dict(
        note=("Scanner x muscle statistics of the C4 per-image estimates (raw = before shrinkage, final = submitted), "
              "muscle from test_muscle.csv (subset 'label' = OCR or block rule; 'all' adds GNB-inferred). "
              "Counterfactual = first-order change of C4 if the PA (confidence rule) and FL (EB, plain-median centre) "
              "shrinkage centres were the scanner x muscle median for muscles with n >= 8 (w_pa, w_fl kept)."),
        literature=LITERATURE,
        mixing=M.to_dict("records"),
        counterfactual=dist,
        test_population="healthy and cerebral-palsy participants aged 18-28 (competition data page)")
    (OUT / "muscle_priors.json").write_text(json.dumps(out, indent=1, default=str))
    pd.DataFrame(LITERATURE).to_csv(OUT / "priors_literature.csv", index=False)
    pd.set_option("display.width", 250)
    print(M.drop(columns=["delta_pa_ci95", "delta_fl_ci95"]).to_string())
    print(json.dumps(dist, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ocr", "read", "infer", "sheets", "priors", "all"])
    ap.add_argument("--force", action="store_true", help="ocr: redo every image")
    args = ap.parse_args()
    if args.cmd == "ocr":
        cmd_ocr(args)
    if args.cmd in ("read", "all"):
        cmd_read(args)
    if args.cmd in ("infer", "all"):
        cmd_infer(args)
    if args.cmd in ("sheets", "all"):
        cmd_sheets(args)
    if args.cmd in ("priors", "all"):
        cmd_priors(args)
    print(f"peak rss {rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
