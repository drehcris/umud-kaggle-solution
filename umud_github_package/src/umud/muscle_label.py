"""Read the burned-in muscle label (VL / GM / RF) and body side from UMUD test images.

This is automated metadata reading, like decoding the scale bar: the scanner operator typed the
muscle into the UI and the scanner burned that text into the exported frame.

* Siemens Juniper (1200x800 .tif, right ruler): free text such as "Rechts VL", "Links VL 50%",
  "Rechts gm", "gm rechts", "VL Links", usually in the left panel below the parameter list, but the
  operator can place it anywhere (it sometimes overlaps the B-mode area). The preset line of the
  parameter list ("Quadriceps" / "Tri Surae") is read too, as a weak hint.
* Philips Lumify (1200x800 .png): "vl rechts", "gm rechts", "rf rechts", "rechts GM", "gm" ... in the
  lower part of the B-mode sector.
* Telemed ArtUS and the two legacy layouts carry no muscle text (checked on every test image).

Method (no hand labels anywhere): the whole frame is upscaled x2 (bicubic) and OCR'd by tesseract
(``--psm 11`` sparse text, TSV word boxes) in two renderings, ``inv`` (inverted grey, keeps
anti-aliased glyph edges) and ``bin`` (threshold > 150, which deletes most of the darker speckle under
labels placed on the B-mode area). Words are normalised and parsed:

1. side words, fuzzy: rechts/links and OCR truncations (``recht``, ``echts``, ``techts``, ``rec``...);
2. muscle words, with the observed OCR confusions: VL <- ``vl vi vil v1 v|``; GM <- ``gm gjm qm ym gn``
   (``km`` dropped by critic4 fix 2: it was the overprinted IMG_00209 label); RF <- ``rf rrf af``; merged
   words (``virechts``, ``gmrechts``) are split;
3. a muscle word counts when it sits on the same text line as a side word and touches it (either
   order), or, standalone, when it is an exact ``vl``/``gm``/``rf`` with tesseract conf >= 60, a glyph
   height of a UI label and outside the scanner's parameter panels.

When the full-frame reading is weak (unknown or confidence < 0.85) and some word was found, the text
line around each weak read is cropped, upscaled x4, binarised at 4 thresholds and re-read with
``--psm 7`` and ``--psm 11`` (``reread_lines``); this recovers labels typed over the B-mode area
(IMG_00030 "Rechts VL 50%") and partly clipped labels (IMG_00253 "rf rechts" at the sector edge).
These 8 renderings together count as one vote with weight up to ~2.

Every supporting read adds ``(conf_side + conf_muscle) / 200 x quality`` (quality 1.0 exact, 0.8
confusion, x 0.75 if standalone) to its muscle. The label is the arg-max; confidence =
``share x (1 - exp(-1.5 x support))`` (two exact paired reads -> ~0.94, one -> ~0.75, a single
standalone exact read -> ~0.6). Ties / conflicting reads lower the share.

critic4 fix 2 (2026-10-21): when a clean exact read exists, confusion-class words (quality 0.8) no longer
vote for a muscle (``_vote_reads``); a reading is flagged ``ambiguous`` when the reads conflict (winner
share < 0.75) or when the label rests on standalone reads only while a side word carries an unparsed
token in its label slot (``is_ambiguous``; on the test set this flags IMG_00209 only). ``_trusted`` drops
ambiguous rows by default (no-op for tables without the ``ambiguous`` column).

API
---
>>> r = read_image(path)                 # dict(muscle, side, confidence, method, raw, bbox, preset, ...)
>>> words = ocr_image(gray)              # DataFrame of words (variant, text, conf, x, y, w, h) at original scale
>>> r = parse_words(words)               # the parser alone (re-use cached OCR words)
>>> T = load_test_muscle()               # artifacts/eval_v4/muscle/test_muscle.csv (scripts/read_muscle_labels.py)
>>> key = muscle_groups(per_image.group) # '<scanner>|<muscle>' shrinkage keys (n >= 8), else '<scanner>'
>>> c = muscle_centres(per_image.pa_deg, per_image.group)   # per-image scanner x muscle prior centre

Verified: all 120 OCR readings (71 Siemens, 49 Lumify) checked by eye on the contact sheets in
artifacts/eval_v4/muscle/sheets/ -> 120/120 correct muscle (and correct side where read).
"""
from __future__ import annotations

import difflib
import io
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

MUSCLES = ("VL", "GM", "RF")
TESSERACT = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
VARIANTS = ("inv", "bin")
MIN_WORD_CONF = 30.0            # tesseract word confidence below this is speckle
UPSCALE = 2

# --------------------------------------------------------------------------------------------- OCR

def _render(gray: np.ndarray, variant: str) -> np.ndarray:
    import cv2
    up = cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    if variant == "inv":
        return 255 - up
    if variant == "bin":
        return np.where(up > 150, 0, 255).astype(np.uint8)
    raise ValueError(variant)


def _tesseract_tsv(img: np.ndarray, psm: int = 11) -> pd.DataFrame:
    import cv2
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    r = subprocess.run([TESSERACT, "stdin", "stdout", "--psm", str(psm), "-l", "eng", "tsv"],
                       input=buf.tobytes(), capture_output=True, check=True)
    df = pd.read_csv(io.StringIO(r.stdout.decode("utf-8", "replace")), sep="\t", quoting=3,
                     keep_default_na=False, dtype={"text": str})
    df = df[(df.level == 5) & (df.text.astype(str).str.strip() != "")]
    return df


def ocr_image(gray: np.ndarray, variants=VARIANTS) -> pd.DataFrame:
    """OCR words of a grey frame (uint8 HxW) in every rendering, boxes at ORIGINAL scale."""
    out = []
    for v in variants:
        df = _tesseract_tsv(_render(gray, v))
        for w, c, x, y, ww, hh in zip(df.text, df.conf, df.left, df.top, df.width, df.height):
            w = str(w)
            if not re.search(r"[A-Za-z]", w):
                continue
            out.append((v, w, float(c), x / UPSCALE, y / UPSCALE, ww / UPSCALE, hh / UPSCALE))
    return pd.DataFrame(out, columns=["variant", "text", "conf", "x", "y", "w", "h"])


def load_gray(path: str | Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


# ------------------------------------------------------------------------------------------ parser

def _norm(s: str) -> str:
    s = s.lower().replace("|", "l")
    return re.sub(r"[^a-z0-9]", "", s)


_SIDE_WORDS = {"rechts": "R", "links": "L"}
_SIDE_PARTIAL = {"recht": "R", "rechte": "R", "echts": "R", "techts": "R", "rechis": "R", "rec": "R",
                 "rechls": "R", "fechts": "R", "inks": "L", "iinks": "L", "llnks": "L", "linke": "L",
                 "lnks": "L", "link": "L", "rehts": "R", "hts": "R"}


def side_of(s: str) -> tuple[str, float] | None:
    """('R'|'L', quality) for a side word, None otherwise."""
    n = _norm(s)
    if n in _SIDE_WORDS:
        return _SIDE_WORDS[n], 1.0
    if n in _SIDE_PARTIAL:
        return _SIDE_PARTIAL[n], 0.85
    if len(n) >= 5:
        for k, v in _SIDE_WORDS.items():
            if difflib.SequenceMatcher(None, n, k).ratio() >= 0.8:
                return v, 0.8
    return None


_MUSCLE_EXACT = {"vl": "VL", "gm": "GM", "rf": "RF"}
_MUSCLE_RE = (
    ("VL", re.compile(r"^[vy][li1]{1,2}$")),          # vi, vil, v1, vll (v| -> vl by _norm)
    # gjm, qm, ym, gn, grn. 'k' was dropped (critic4 fix 2): the only 'km' on the test set is the
    # overprinted IMG_00209 label ('Rechts' + V/K/R glyphs over 'M'), which must not count as a GM vote.
    ("GM", re.compile(r"^[gqy6][jril]?[mn]$")),
    ("RF", re.compile(r"^r?[ra]f$")),                  # rrf, af
)


def muscle_of(s: str) -> tuple[str, float] | None:
    """(muscle, quality) for a muscle word (1.0 exact, 0.8 known OCR confusion), None otherwise."""
    n = _norm(s)
    if n in _MUSCLE_EXACT:
        return _MUSCLE_EXACT[n], 1.0
    for m, rx in _MUSCLE_RE:
        if rx.match(n):
            return m, 0.8
    return None


_MERGED = re.compile(r"^(?P<a>[a-z0-9]{1,3}?)(?P<b>rechts|links|recht|echts|techts|rehts)$|"
                     r"^(?P<c>rechts|links)(?P<d>[a-z0-9]{2,3})$")


def split_merged(s: str):
    """'virechts' -> ('vi', 'rechts'); 'rechtsgm' -> ('gm', 'rechts'); else None."""
    m = _MERGED.match(_norm(s))
    if not m:
        return None
    if m.group("a") is not None:
        return m.group("a"), m.group("b")
    return m.group("d"), m.group("c")


def _in_ui_panel(x: float, y: float, W: int | None, H: int | None, layout: str | None) -> bool:
    """Boxes of the scanners' fixed parameter panels / depth labels (never the typed label)."""
    if layout == "siemens":
        return (x < 200 and y < 330) or (x > 930 and y > 650) or y > 770
    if layout == "lumify":
        return x > 1080 or (x < 40 and y > 650)
    return False


def collect_reads(words: pd.DataFrame, layout: str | None = None) -> list[dict]:
    """Every muscle / side read in a word table (one list over all renderings)."""
    reads = []
    for v, d in words.groupby("variant", sort=False):
        d = d.reset_index(drop=True)
        S, M = [], []
        for i, r in d.iterrows():
            t = str(r.text)
            if r.conf < MIN_WORD_CONF or r.w > 15 * max(r.h, 6.0):      # speckle 'words'
                continue
            mg = split_merged(t)
            if mg is not None:
                mm, ss = muscle_of(mg[0]), side_of(mg[1])
                if mm and ss:
                    reads.append(dict(muscle=mm[0], side=ss[0], w=r.conf / 100 * mm[1] * ss[1] * 0.95,
                                      variant=v, raw=t, bbox=(r.x, r.y, r.w, r.h), kind="merged", q=mm[1]))
                    continue
            sd = side_of(t)
            if sd:
                S.append((i, sd))
                continue
            mu = muscle_of(t)
            if mu:
                M.append((i, mu))
        used = set()
        for si, (sd, sq) in S:
            a = d.loc[si]
            best = None
            for mi, (mu, mq) in M:
                b = d.loc[mi]
                hh = max(a.h, b.h, 6.0)
                if abs((a.y + a.h / 2) - (b.y + b.h / 2)) > 0.7 * hh:
                    continue
                gap = b.x - (a.x + a.w) if b.x >= a.x else a.x - (b.x + b.w)
                if gap < -0.5 * hh or gap > 2.5 * hh:
                    continue
                w = (a.conf + b.conf) / 200 * mq * sq
                if best is None or w > best[0]:
                    best = (w, mi, mu, b, mq)
            if best is not None:
                w, mi, mu, b, bq = best
                used.add(mi)
                x0, y0 = min(a.x, b.x), min(a.y, b.y)
                x1, y1 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
                raw = f"{a.text} {b.text}" if a.x <= b.x else f"{b.text} {a.text}"
                reads.append(dict(muscle=mu, side=sd, w=w, variant=v, raw=raw,
                                  bbox=(x0, y0, x1 - x0, y1 - y0), kind="paired", q=bq))
            else:                                               # side word, muscle unreadable
                reads.append(dict(muscle=None, side=sd, w=a.conf / 100 * sq * 0.5, variant=v, raw=str(a.text),
                                  bbox=(a.x, a.y, a.w, a.h), kind="side_only", q=None,
                                  adj=_adjacent_token(d, si, set(i for i, _ in S))))
        for mi, (mu, mq) in M:
            if mi in used or mq < 1.0:
                continue
            b = d.loc[mi]
            if b.conf < 60 or not (7 <= b.h <= 22) or (not str(v).startswith("line") and _in_ui_panel(b.x, b.y, None, None, layout)):
                continue
            reads.append(dict(muscle=mu, side=None, w=b.conf / 100 * 0.75, variant=v, raw=str(b.text),
                              bbox=(b.x, b.y, b.w, b.h), kind="standalone", q=1.0))
    return reads


def _adjacent_token(d: pd.DataFrame, si: int, side_idx: set) -> str:
    """The short (1-4 alnum chars) non-side word that touches side word ``si`` on its text line, i.e. a
    label slot whose muscle token did not parse (e.g. the overprinted 'KM' of IMG_00209); '' if none."""
    a = d.loc[si]
    for j, b in d.iterrows():
        if j == si or j in side_idx or b.conf < MIN_WORD_CONF:
            continue
        n = _norm(str(b.text))
        if not (1 <= len(n) <= 4) or not re.search(r"[a-z]", n):
            continue
        hh = max(a.h, b.h, 6.0)
        if abs((a.y + a.h / 2) - (b.y + b.h / 2)) > 0.7 * hh:
            continue
        gap = b.x - (a.x + a.w) if b.x >= a.x else a.x - (b.x + b.w)
        if -0.5 * hh <= gap <= 2.5 * hh:
            return str(b.text)
    return ""


AMBIGUOUS_SHARE = 0.75          # winner's share of the muscle support below this = conflicting reads


def _same_line(a: tuple, b: tuple) -> bool:
    """Two (x, y, w, h) boxes on one text line (vertical centres within 0.7 glyph heights)."""
    hh = max(a[3], b[3], 6.0)
    return abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) <= 0.7 * hh


def _vote_reads(reads: list[dict]) -> list[dict]:
    """critic4 fix 2: a known-OCR-confusion word (quality 0.8) does not vote for a muscle when a clean exact
    read (quality 1.0) of a label exists on a DIFFERENT text line: the clean line is the label and the
    confusion word sits in another (overprinted / unreadable) label slot (IMG_00209: 'Rechts KM' above a
    clean 'GM'). Confusion reads on the clean read's own line are other renderings of the same label and
    keep their vote (dropping them would only deflate verified reads such as IMG_00277 'rrf echts')."""
    exact = [rd["bbox"] for rd in reads if rd.get("q") == 1.0 and rd["muscle"] is not None]
    if not exact:
        return reads
    out = []
    for rd in reads:
        if (rd["muscle"] is not None and rd.get("q") is not None and rd["q"] < 1.0
                and not any(_same_line(rd["bbox"], e) for e in exact)):
            rd = dict(rd, muscle=None, kind=rd["kind"] + "_dropped")
        out.append(rd)
    return out


def is_ambiguous(reads: list[dict], muscle: str, share: float) -> bool:
    """A reading is ambiguous (critic4 fix 2) when (a) the reads conflict (winner share < AMBIGUOUS_SHARE),
    or (b) the label rests on standalone reads only while a side word on another line carries an
    unparsed short token in its label slot (an overprinted / unreadable second label, IMG_00209)."""
    if muscle == "unknown":
        return False
    if share < AMBIGUOUS_SHARE:
        return True
    win = [rd for rd in reads if rd["muscle"] == muscle]
    paired = any(rd["kind"] in ("paired", "merged") for rd in win)
    slot = any((rd["kind"] == "side_only" and rd.get("adj")) or rd["kind"].endswith("_dropped") for rd in reads)
    return (not paired) and slot


def aggregate(reads: list[dict]) -> dict:
    """Vote the reads into one label + confidence (see module docstring)."""
    reads = _vote_reads(reads)
    support = {m: 0.0 for m in MUSCLES}
    sides = {"R": 0.0, "L": 0.0}
    for rd in reads:
        if rd["muscle"] is not None:
            support[rd["muscle"]] += rd["w"]
        if rd["side"] is not None:
            sides[rd["side"]] += rd["w"]
    tot = sum(support.values())
    if tot <= 0:
        muscle, conf, share = "unknown", 0.0, 0.0
    else:
        muscle = max(support, key=support.get)
        top = support[muscle]
        share = top / tot
        conf = share * (1 - math.exp(-1.5 * top))
    side = max(sides, key=sides.get) if sum(sides.values()) > 0 else "unknown"
    ambiguous = is_ambiguous(reads, muscle, share)
    pool = [rd for rd in reads if rd["muscle"] == muscle] if muscle != "unknown" else reads
    best = max(pool, key=lambda rd: rd["w"]) if pool else None
    kinds = sorted({("line_" if str(rd["variant"]).startswith("line") else "") + rd["kind"] for rd in pool}) \
        if muscle != "unknown" else []
    method = ("ocr:" + "+".join(kinds)) if kinds else ("ocr:side_only" if reads else "ocr:none")
    return dict(muscle=muscle, side=side, confidence=round(float(conf), 3), method=method,
                support_VL=round(support["VL"], 3), support_GM=round(support["GM"], 3),
                support_RF=round(support["RF"], 3), n_reads=sum(1 for rd in reads if rd["muscle"] is not None),
                raw=best["raw"] if best else "",
                bbox=tuple(round(float(z), 1) for z in best["bbox"]) if best else None,
                ambiguous=bool(ambiguous))


def parse_words(words: pd.DataFrame, layout: str | None = None, W: int | None = None,
                H: int | None = None) -> dict:
    """Muscle / side reading from OCR words (``ocr_image`` output), without re-reads."""
    return aggregate(collect_reads(words, layout))


LINE_THRESHOLDS = (110, 140, 170, 200)
LINE_PSMS = (7, 11)


def reread_lines(gray: np.ndarray, reads: list[dict]) -> pd.DataFrame:
    """Targeted re-read of the text line around weak reads (side word without muscle, merged or
    standalone words): crop the line strip, upscale x4, binarise at 4 thresholds, tesseract psm 7 and 11.
    Labels typed over the B-mode area (e.g. IMG_00030 'Rechts VL 50%') are recovered this way."""
    import cv2
    H, W = gray.shape
    out, seen = [], []
    for rd in reads:
        if rd["kind"] == "paired" and rd["w"] >= 0.8:
            continue
        x, y, w, h = rd["bbox"]
        hh = max(h, 10.0)
        y0, y1 = int(max(0, y - 1.2 * hh)), int(min(H, y + 2.2 * hh))
        x0, x1 = int(max(0, x - 8 * hh)), int(min(W, x + w + 10 * hh))
        if any(abs(y0 - a) < 6 and abs(x0 - b) < 20 for a, b in seen) or y1 - y0 < 8 or x1 - x0 < 8:
            continue
        seen.append((y0, x0))
        c = cv2.resize(gray[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        for thr in LINE_THRESHOLDS:
            b = cv2.copyMakeBorder(np.where(c > thr, 0, 255).astype(np.uint8), 20, 20, 20, 20,
                                   cv2.BORDER_CONSTANT, value=255)
            for psm in LINE_PSMS:
                df = _tesseract_tsv(b, psm)
                for t, cf, lx, ly, lw, lh in zip(df.text, df.conf, df.left, df.top, df.width, df.height):
                    if re.search(r"[A-Za-z]", str(t)):
                        out.append((f"line_t{thr}_p{psm}", str(t), float(cf), x0 + (lx - 20) / 4,
                                    y0 + (ly - 20) / 4, lw / 4, lh / 4))
    return pd.DataFrame(out, columns=["variant", "text", "conf", "x", "y", "w", "h"])


def read_words(gray: np.ndarray, words: pd.DataFrame, layout: str | None = None,
               line_words: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    """Full-frame reads + (when the reading is weak) targeted line re-reads.

    The line re-reads are one more 'vote' whose weight is the mean over the 8 renderings x 2, so a line
    that reads the same in every rendering counts like two agreeing full-frame renderings."""
    reads = collect_reads(words, layout)
    first = aggregate(reads)
    if line_words is None:
        need = first["muscle"] == "unknown" or first["confidence"] < 0.85 or first.get("ambiguous", False)
        line_words = reread_lines(gray, reads) if need and reads else \
            pd.DataFrame(columns=["variant", "text", "conf", "x", "y", "w", "h"])
    if len(line_words):
        lr = collect_reads(line_words, layout)
        n_sub = len(LINE_THRESHOLDS) * len(LINE_PSMS)
        for rd in lr:
            rd["w"] = rd["w"] * 2.0 / n_sub
        reads = reads + lr
    return aggregate(reads), line_words


# -------------------------------------------------------------------------------- scanner metadata

def layout_of(path: str | Path, W: int, H: int, words: pd.DataFrame | None = None,
              gray: np.ndarray | None = None) -> str:
    """'siemens' | 'lumify' | 'telemed' | 'legacy' from file type, size, panel words and UI colour."""
    p = str(path).lower()
    if p.endswith(".png") and (W, H) == (1200, 800):
        return "lumify"
    if words is not None and len(words):
        low = set(words.text.astype(str).str.lower())
        if {"sendeleistung", "dtce", "quadriceps", "12l3"} & low:
            return "siemens"
        if {"frmavg", "imenh", "frq", "rej", "lndens"} & low:
            return "telemed"
    if (W, H) == (1088, 644):
        return "telemed"
    if (W, H) == (1200, 800):
        if gray is not None:            # Telemed ArtUS UI is mid-grey, Siemens UI is black
            return "telemed" if float(np.median(gray[300:500, 0:25])) > 30 else "siemens"
        return "siemens_or_telemed"
    return "legacy"


def panel_metadata(words: pd.DataFrame, layout: str) -> dict:
    """Scanner preset read from the same OCR words (Siemens 'Quadriceps' / 'Tri Surae'); a weak hint
    only: operators mostly keep the default preset (all OCR-labelled GM images say 'Quadriceps')."""
    out = dict(preset="")
    if not len(words) or layout != "siemens":
        return out
    w = words[words.variant == "inv"]
    row = w[(w.x < 30) & (w.y > 100) & (w.y < 122)]
    out["preset"] = " ".join(row.sort_values("x").text.astype(str)) if len(row) else ""
    return out


def read_image(path: str | Path, words: pd.DataFrame | None = None) -> dict:
    """Full reading for one image (OCR words are computed unless given)."""
    g = load_gray(path)
    H, W = g.shape
    if words is None:
        words = ocr_image(g)
    lay = layout_of(path, W, H, words, g)
    r, _ = read_words(g, words, layout=lay)
    r.update(layout=lay, **panel_metadata(words, lay))
    return r


# ------------------------------------------------------------------------------- pipeline helpers

def load_test_muscle(path: str | Path | None = None) -> pd.DataFrame:
    """test_muscle.csv (written by scripts/read_muscle_labels.py ocr+read+infer), indexed by image_id."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "artifacts/eval_v4/muscle/test_muscle.csv"
    return pd.read_csv(path, keep_default_na=False).set_index("image_id")


def ambiguous_mask(t: pd.DataFrame) -> pd.Series:
    """Rows flagged ambiguous by the reader (column ``ambiguous``; absent in tables written before the
    critic4 fix 2, which then flag nothing)."""
    if "ambiguous" not in t:
        return pd.Series(False, index=t.index)
    a = t["ambiguous"]
    return a.astype(str).str.strip().str.lower().isin(("true", "1", "1.0", "yes"))


def _trusted(t: pd.DataFrame, min_conf: float, use_inferred: bool,
             drop_ambiguous: bool = True) -> tuple[pd.Series, pd.Series]:
    """Trusted muscle per row ('' if none): OCR or block-rule labels (and, if ``use_inferred``, GNB GM with
    confidence >= min_conf). ``drop_ambiguous``: OCR reads flagged ambiguous are not trusted (no-op on
    tables without the ``ambiguous`` column)."""
    meth = t["method"].astype(str)
    conf = pd.to_numeric(t["confidence"], errors="coerce").fillna(0)
    trusted = meth.str.startswith("ocr") | (meth == "block_unanimous")
    if drop_ambiguous:
        trusted &= ~ambiguous_mask(t)
    if use_inferred:
        trusted |= meth.str.startswith("gnb_gm") & (conf >= min_conf)
    m = t["muscle"].where(trusted & t["muscle"].isin(MUSCLES), "")
    qf = t["muscle_group"].astype(str).eq("QF") & (
        trusted | (use_inferred & meth.str.startswith("gnb_qf") & (conf >= min_conf)))
    return m, qf


def muscle_groups(scanner: pd.Series, table: pd.DataFrame | None = None, min_n: int = 8,
                  min_conf: float = 0.9, use_inferred: bool = True) -> pd.Series:
    """Shrinkage-group key per image (critic fix 1): '<scanner>|<muscle>' when the image's muscle is
    known (OCR, block rule, or - if ``use_inferred`` - the GNB GM rule with confidence >= ``min_conf``)
    and that scanner x muscle cell has >= ``min_n`` images; otherwise '<scanner>' (the leftover images
    of that scanner: unknown muscle, GNB-'QF', or small cells). Note that a key-based shrinker then
    centres the leftovers on the leftovers' own median; use ``muscle_centres`` to centre them on the
    pooled VL+RF ('QF') anchors or on the whole-scanner median instead."""
    t = (load_test_muscle() if table is None else table).reindex(scanner.index)
    m, _ = _trusted(t, min_conf, use_inferred)
    key_m = scanner.astype(str) + "|" + m
    n_m = key_m.map(key_m[m != ""].value_counts()).fillna(0)
    return scanner.astype(str).where(~((m != "") & (n_m >= min_n)), key_m)


def muscle_centres(x: pd.Series, scanner: pd.Series, table: pd.DataFrame | None = None, min_n: int = 8,
                   min_conf: float = 0.9, use_inferred: bool = True) -> pd.Series:
    """Per-image prior centre for the estimates ``x`` (index image_id): the median of ``x`` over the
    image's scanner x muscle cell if the muscle is known and the cell has >= ``min_n`` images; for
    VL/RF-or-'QF' images without a usable muscle cell, the median over the scanner's VL+RF(+QF) images
    (if >= ``min_n``); otherwise the whole-scanner median (the current C2/C4 centre)."""
    t = (load_test_muscle() if table is None else table).reindex(x.index)
    m, qf = _trusted(t, min_conf, use_inferred)
    sc = scanner.reindex(x.index).astype(str)
    x = pd.to_numeric(x, errors="coerce")
    c = x.groupby(sc).transform("median")
    q_ok = qf | m.isin(["VL", "RF"])
    for s_ in sc.unique():
        ins = sc == s_
        pool = ins & q_ok
        if pool.sum() >= min_n:
            c = c.where(~(ins & qf), x[pool].median())
        for mu in MUSCLES:
            cell = ins & (m == mu)
            if cell.sum() >= min_n:
                c = c.where(~cell, x[cell].median())
    return c
