"""UMUD maskbank inference pipeline (runs inside a Kaggle GPU kernel).

Stages (each runs in its own subprocess, launched by kernel.py):
  data  - locate competition test images, download the OSF xbawc benchmark zips
          (static + dynamic) and the "Example Annotated Images" (young + old),
          export VL images as PNG + every annotation file to /kaggle/working/vl_data,
          write the manifest.
  torch - our smp U-Net++ R34 384 checkpoints (aponeurosis, fascicle), hflip TTA.
  tf    - DL_Track_US v0.3 VGG16-UNet .h5 models (aponeurosis, fascicle), hflip TTA,
          plus an optional IFSS probe.

Output (MASKBANK FORMAT): /kaggle/working/maskbank/<dataset>__<model>.npz, one key per
image (file stem), uint8 HxW probability*255 at the native resolution of the image file.
Extra: <dataset>__<model>_noflip.npz (original-orientation pass only, no TTA).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")
TMP = Path("/tmp/maskbank")
DATA = TMP / "data"
MODELS = TMP / "models"
MASKBANK = WORK / "maskbank"
VL_OUT = WORK / "vl_data"
INTERNAL_MANIFEST = TMP / "manifest_internal.csv"
INPUT_ROOT = Path("/kaggle/input")
SMOKE = int(os.environ.get("MASKBANK_SMOKE", "0"))  # >0: images per dataset for inference
DATASETS = ("test", "static", "dynamic", "vl")
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
OSF_API = "https://api.osf.io/v2/nodes/{node}/files/osfstorage/"
BENCHMARK_NODE = "xbawc"
VL_URLS = {"young": "https://osf.io/download/zcuwk/", "old": "https://osf.io/download/6ga94/"}
DLT_MODELS = {
    "dlt_apo": ("https://osf.io/download/6jzuy/", "model-apo-VGG16_DiceBCE_512_300.h5"),
    "dlt_fasc": ("https://osf.io/download/6fpkt/", "model-fasc-VGG16_DiceBCE_200_512.h5"),
}
DLT_IFSS = ("https://osf.io/download/n823q/", "model-fasc-IFSS_1Encoder_3d_a08_g2_noaug_nocrop_HL.h5")
TORCH_MODELS = {
    "ours_apo384": "apo_unetpp_r34_384.pt",
    "ours_fasc384": "fasc_unetpp_r34_384.pt",
}
T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


# --------------------------------------------------------------------------- utils
def http_get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "umud-maskbank/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception as error:  # noqa: BLE001
            log(f"GET {url} failed ({error!r}), retry {attempt + 1}/5")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not GET {url}")


def download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        try:
            started = time.time()
            request = urllib.request.Request(url, headers={"User-Agent": "umud-maskbank/1.0"})
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1 << 22)
            partial.rename(destination)
            size_mb = destination.stat().st_size / 1e6
            log(f"downloaded {url} -> {destination.name} {size_mb:.1f} MB in {time.time() - started:.1f}s")
            return destination
        except Exception as error:  # noqa: BLE001
            log(f"download {url} failed ({error!r}), retry {attempt + 1}/5")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}")


def safe_extract(archive: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = member.filename
            if member.is_dir() or "__MACOSX" in name or Path(name).name.startswith("._"):
                continue
            target = (destination / name).resolve()
            if not str(target).startswith(str(root) + os.sep):
                log(f"skipping unsafe zip member {name!r}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def osf_walk(node: str, max_depth: int = 5) -> list[dict]:
    files: list[dict] = []

    def walk(url: str, prefix: str, depth: int) -> None:
        while url:
            payload = http_get_json(url)
            for item in payload["data"]:
                attributes = item["attributes"]
                path = f"{prefix}/{attributes['name']}"
                if attributes["kind"] == "folder":
                    if depth < max_depth:
                        walk(item["relationships"]["files"]["links"]["related"]["href"], path, depth + 1)
                else:
                    files.append(
                        {"path": path, "size": attributes.get("size"), "download": item["links"].get("download")}
                    )
            url = payload["links"].get("next")

    walk(OSF_API.format(node=node), "", 0)
    return files


def find_input_file(name: str) -> Path:
    patterns = [f"notebooks/*/*/{name}", f"*/{name}", f"*/*/{name}"]
    matches = sorted({path for pattern in patterns for path in INPUT_ROOT.glob(pattern) if path.is_file()})
    if not matches:
        matches = sorted(path for path in INPUT_ROOT.rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"{name} not found under {INPUT_ROOT}")
    if len(matches) > 1:
        log(f"multiple matches for {name}: {matches}; using first")
    return matches[0]


def sanitize_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def to_uint8(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(probability * 255.0), 0, 255).astype(np.uint8)


def read_manifest() -> dict[str, list[dict]]:
    by_dataset: dict[str, list[dict]] = {name: [] for name in DATASETS}
    with INTERNAL_MANIFEST.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            row["height"] = int(row["height"])
            row["width"] = int(row["width"])
            by_dataset.setdefault(row["dataset"], []).append(row)
    if SMOKE > 0:
        by_dataset = {name: rows[:SMOKE] for name, rows in by_dataset.items()}
    return by_dataset


def save_bank(dataset: str, model_key: str, arrays: dict[str, np.ndarray]) -> Path:
    MASKBANK.mkdir(parents=True, exist_ok=True)
    path = MASKBANK / f"{dataset}__{model_key}.npz"
    np.savez_compressed(path, **arrays)
    log(f"saved {path.name}: {len(arrays)} keys, {path.stat().st_size / 1e6:.1f} MB")
    return path


def append_summary(row: dict) -> None:
    path = MASKBANK / "summary.csv"
    fields = ["stage", "model", "dataset", "n_images", "n_failed", "seconds", "mean_max_prob",
              "frac_max_below_0.5", "mean_frac_above_0.5", "notes"]
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def prob_stats(probabilities: list[np.ndarray]) -> dict:
    if not probabilities:
        return {"mean_max_prob": "", "frac_max_below_0.5": "", "mean_frac_above_0.5": ""}
    maxima = np.array([float(p.max()) / 255.0 for p in probabilities])
    above = np.array([float((p > 127).mean()) for p in probabilities])
    return {
        "mean_max_prob": round(float(maxima.mean()), 4),
        "frac_max_below_0.5": round(float((maxima < 0.5).mean()), 4),
        "mean_frac_above_0.5": round(float(above.mean()), 5),
    }


# --------------------------------------------------------------------------- data stage
def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.height, image.width


def collect_test() -> list[dict]:
    test_dirs = sorted({p for pattern in ("competitions/*/*/test_set_v2", "competitions/*/test_set_v2",
                                          "*/*/test_set_v2", "*/test_set_v2") for p in INPUT_ROOT.glob(pattern)})
    search_roots = test_dirs or [INPUT_ROOT]
    log(f"test search roots: {search_roots}")
    candidates = [path for root in search_roots for path in root.rglob("IMG_*")
                  if path.suffix.lower() in IMAGE_SUFFIXES]
    candidates = [path for path in candidates if re.fullmatch(r"IMG_\d{5}", path.stem)]
    by_stem: dict[str, Path] = {}
    for path in sorted(candidates):
        by_stem.setdefault(path.stem, path)
    records = []
    for stem, path in sorted(by_stem.items()):
        height, width = image_size(path)
        records.append({"dataset": "test", "image_id": stem, "source_filename": path.name,
                        "height": height, "width": width, "path": str(path)})
    log(f"test: {len(records)} images (dirs: {sorted({str(p.parent) for p in by_stem.values()})})")
    return records


def collect_benchmarks() -> list[dict]:
    files = osf_walk(BENCHMARK_NODE)
    log(f"OSF {BENCHMARK_NODE}: {len(files)} files")
    for item in files:
        log(f"  osf file {item['path']} size={item['size']}")
    (MASKBANK / "osf_xbawc_listing.json").write_text(json.dumps(files, indent=1), encoding="utf-8")
    zips = [item for item in files if item["path"].lower().endswith(".zip")]
    static = sorted(
        (z for z in zips if re.search(r"benchmark.*architecture", z["path"], re.I) and "dynamic" not in z["path"].lower()),
        key=lambda z: z["path"],
    )
    dynamic = sorted(
        (z for z in zips if re.search(r"benchmark.*dynamic", z["path"], re.I)), key=lambda z: z["path"]
    )
    log(f"static zip candidates: {[z['path'] for z in static]}")
    log(f"dynamic zip candidates: {[z['path'] for z in dynamic]}")
    records: list[dict] = []
    if static:
        choice = static[-1]
        archive = download(choice["download"], DATA / "zips" / Path(choice["path"]).name)
        safe_extract(archive, DATA / "static")
        images = sorted(
            p for p in (DATA / "static").rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and re.fullmatch(r"im_\d+_arch", p.stem)
        )
        for path in images:
            height, width = image_size(path)
            records.append({"dataset": "static", "image_id": path.stem, "source_filename": path.name,
                            "height": height, "width": width, "path": str(path)})
        log(f"static: {len(images)} images from {choice['path']}")
    else:
        log("WARNING: no static benchmark zip found")
    if dynamic:
        choice = dynamic[-1]
        archive = download(choice["download"], DATA / "zips" / Path(choice["path"]).name)
        safe_extract(archive, DATA / "dynamic")
        frames = sorted(p for p in (DATA / "dynamic").rglob("frame_*.png"))
        for path in frames:
            height, width = image_size(path)
            records.append({"dataset": "dynamic", "image_id": path.stem, "source_filename": path.name,
                            "height": height, "width": width, "path": str(path)})
        log(f"dynamic: {len(frames)} frames from {choice['path']}")
    else:
        log("WARNING: no dynamic benchmark zip found")
    return records


def tiff_metadata(path: Path) -> dict:
    """ImageJ / resolution metadata of a TIFF (first page)."""
    info: dict = {}
    try:
        from PIL import Image

        with Image.open(path) as image:
            tags = getattr(image, "tag_v2", {})
            description = tags.get(270, "")
            info["imagej_description"] = str(description)[:400].replace("\n", " | ")
            info["xres"] = str(tags.get(282, ""))
            info["yres"] = str(tags.get(283, ""))
            info["resunit"] = str(tags.get(296, ""))
            info["has_ij_metadata"] = int(50839 in tags or 50838 in tags)
            info["n_frames"] = getattr(image, "n_frames", 1)
    except Exception as error:  # noqa: BLE001
        info["error"] = repr(error)[:200]
    return info


def parse_rois(path: Path) -> list[dict]:
    rows = []
    try:
        import roifile
    except ImportError:
        return [{"error": "roifile unavailable"}]
    try:
        rois = roifile.roiread(str(path))
    except Exception as error:  # noqa: BLE001
        return [{"error": repr(error)[:300]}]
    if rois is None:
        return []
    if not isinstance(rois, list):
        rois = [rois]
    for index, roi in enumerate(rois):
        try:
            coordinates = roi.coordinates()
            coords = [] if coordinates is None else np.round(np.asarray(coordinates, dtype=float), 3).tolist()
        except Exception as error:  # noqa: BLE001
            coords = f"error {error!r}"
        rows.append({
            "roi_index": index,
            "roi_name": getattr(roi, "name", ""),
            "roi_type": str(getattr(roi, "roitype", "")),
            "roi_subtype": str(getattr(roi, "subtype", "")),
            "line": json.dumps([float(getattr(roi, a, 0.0)) for a in ("x1", "y1", "x2", "y2")]),
            "position": getattr(roi, "position", ""),
            "n_points": len(coords) if isinstance(coords, list) else "",
            "coords_xy": json.dumps(coords),
        })
    return rows


def preview_table(path: Path, out_lines: list[str]) -> None:
    import pandas as pd

    out_lines.append(f"===== {path}")
    suffix = path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xlsm", ".xls", ".ods"}:
            try:
                sheets = pd.read_excel(path, sheet_name=None, header=None)
            except Exception:  # ImageJ .xls files are tab-separated text
                sheets = {"text": pd.read_csv(path, sep="\t", header=None)}
            for sheet, frame in sheets.items():
                out_lines.append(f"--- sheet {sheet!r} shape={frame.shape}")
                out_lines.append(frame.head(25).to_string(max_cols=30, max_colwidth=40))
        elif suffix in {".csv", ".txt", ".tsv", ".dat", ".json", ".xml", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace").splitlines()
            out_lines.append(f"--- {len(text)} lines")
            out_lines.extend(text[:25])
    except Exception as error:  # noqa: BLE001
        out_lines.append(f"preview error {error!r}")


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        image.seek(0)
        if image.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
            array = np.asarray(image, dtype=np.float32)
            low, high = float(array.min()), float(array.max())
            array = np.clip(np.rint((array - low) / max(high - low, 1e-6) * 255.0), 0, 255).astype(np.uint8)
            return np.repeat(array[:, :, None], 3, axis=2)
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def pure_red(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] > 150) & (rgb[..., 1] < 100) & (rgb[..., 2] < 100)


def match_offset(big: np.ndarray, small: np.ndarray) -> tuple[int, int, float]:
    """Find (dx, dy) such that big[dy:dy+h, dx:dx+w] == small on most pixels."""
    hb, wb = big.shape[:2]
    hs, ws = small.shape[:2]
    if hs > hb or ws > wb:
        return 0, 0, 0.0
    sub = small[::4, ::4]

    def score(dx: int, dy: int) -> float:
        window = big[dy:dy + hs:4, dx:dx + ws:4]
        return float(np.all(window == sub, axis=-1).mean())

    center = ((wb - ws) // 2, (hb - hs) // 2)
    best = (*center, score(*center))
    if best[2] < 0.9:
        offsets = [(dx, dy) for dy in range(hb - hs + 1) for dx in range(wb - ws + 1)]
        if len(offsets) > 3000:
            offsets = offsets[:: max(1, len(offsets) // 3000)]
        for dx, dy in offsets:
            value = score(dx, dy)
            if value > best[2]:
                best = (dx, dy, value)
    return best


def tiff_resolution(path: Path) -> dict:
    out = {"xres": "", "yres": "", "resunit": ""}
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return out
    try:
        from PIL import Image

        with Image.open(path) as image:
            tags = getattr(image, "tag_v2", {})
            out = {"xres": float(tags.get(282, 0) or 0), "yres": float(tags.get(283, 0) or 0),
                   "resunit": int(tags.get(296, 0) or 0)}
    except Exception as error:  # noqa: BLE001
        out["error"] = repr(error)[:200]
    return out


def collect_vl() -> list[dict]:
    """UMUD 'Example Annotated Images' (Esaote VL, young + old).

    Each folder <subject>/<timepoint> holds pairs: an ANNOTATED image (raw frame pasted on a larger
    black canvas with burned-in red lines) and the RAW frame. Pairs are found by exact pixel matching.
    Export: images/<key>.png (raw, model input), annotated/<key>.png (canvas), label_masks/<key>.png
    (255 where the annotated canvas differs from the raw frame / is non-background in the padding,
    canvas coordinates), vl_index.csv (pairing, offsets, scale tags).
    """
    from collections import Counter

    from PIL import Image

    for sub in ("images", "annotated", "label_masks"):
        (VL_OUT / sub).mkdir(parents=True, exist_ok=True)
    try:
        readme = download("https://osf.io/download/tyd74/", DATA / "vl_readme.md")
        text = readme.read_text(encoding="utf-8", errors="replace")
        (VL_OUT / "Readme_Example_Annotated_Images.md").write_text(text, encoding="utf-8")
        log("===== Example Annotated Images Readme.md =====")
        print(text, flush=True)
    except Exception as error:  # noqa: BLE001
        log(f"readme download failed: {error!r}")
    records: list[dict] = []
    tree_lines: list[str] = []
    preview_lines: list[str] = []
    index_rows: list[dict] = []
    used_keys: set[str] = set()
    for set_name, url in VL_URLS.items():
        archive = download(url, DATA / "zips" / f"vl_{set_name}.zip")
        root = DATA / "vl" / set_name
        files = sorted(safe_extract(archive, root))
        ext_counts: dict[str, int] = {}
        for path in files:
            ext_counts[path.suffix.lower()] = ext_counts.get(path.suffix.lower(), 0) + 1
            tree_lines.append(f"{set_name}/{path.relative_to(root)}\t{path.stat().st_size}")
        log(f"vl/{set_name}: {len(files)} files, extensions {ext_counts}")
        images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
        others = [p for p in files if p.suffix.lower() not in IMAGE_SUFFIXES
                  and p.name not in {"Thumbs.db", ".DS_Store", "desktop.ini"}]
        for path in others:  # any non-image file is an annotation candidate: copy verbatim
            rel = path.relative_to(root)
            destination = VL_OUT / "annotations" / set_name / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.stat().st_size <= 50_000_000:
                shutil.copy2(path, destination)
            preview_table(path, preview_lines)
        top_parts = {p.relative_to(root).parts[0] for p in images}
        strip_top = len(top_parts) == 1 and all(len(p.relative_to(root).parts) > 1 for p in images)
        by_folder: dict[Path, list[Path]] = {}
        for path in images:
            by_folder.setdefault(path.parent, []).append(path)
        folders = sorted(by_folder)
        if SMOKE > 0:
            folders = folders[:SMOKE]
        for folder in folders:
            paths = sorted(by_folder[folder])
            arrays = {}
            for path in paths:
                try:
                    arrays[path] = load_rgb(path)
                except Exception as error:  # noqa: BLE001
                    log(f"  cannot read {path}: {error!r}")
            candidates = []
            names = sorted(arrays)
            for i, first in enumerate(names):  # equal-size pairs: annotated = the one red at differing pixels
                for second in names[i + 1:]:
                    a, b = arrays[first], arrays[second]
                    if a.shape != b.shape:
                        continue
                    differ = np.any(a != b, axis=-1)
                    frac = 1.0 - float(differ.mean())
                    if frac < 0.9 or not differ.any():
                        continue
                    red_a = float(pure_red(a)[differ].mean())
                    red_b = float(pure_red(b)[differ].mean())
                    annotated, raw_path = (first, second) if red_a > red_b else (second, first)
                    candidates.append((frac, annotated, raw_path, 0, 0))
            for big_path, big in arrays.items():
                for small_path, small in arrays.items():
                    if big_path == small_path or big.shape == small.shape:
                        continue
                    if big.shape[0] < small.shape[0] or big.shape[1] < small.shape[1]:
                        continue
                    dx, dy, frac = match_offset(big, small)
                    if frac >= 0.9:
                        candidates.append((frac, big_path, small_path, dx, dy))
            annotated_of: dict[Path, tuple] = {}
            used_big: set[Path] = set()
            for frac, big_path, small_path, dx, dy in sorted(candidates, key=lambda c: -c[0]):
                if small_path in annotated_of or big_path in used_big:
                    continue
                annotated_of[small_path] = (big_path, dx, dy, frac)
                used_big.add(big_path)
            for path in paths:
                if path in used_big or path not in arrays:
                    continue  # annotated canvases are exported with their raw partner
                rel = path.relative_to(root)
                parts = list(rel.parts[1:] if strip_top else rel.parts)
                key = sanitize_key("__".join([set_name, *parts[:-1], Path(parts[-1]).stem]))
                base, n = key, 1
                while key in used_keys:
                    n += 1
                    key = f"{base}_{n}"
                used_keys.add(key)
                raw = arrays[path]
                Image.fromarray(raw).save(VL_OUT / "images" / f"{key}.png")
                row = {"key": key, "set": set_name, "folder": str(folder.relative_to(root)),
                       "raw_source": f"{set_name}/{rel}", "raw_height": raw.shape[0], "raw_width": raw.shape[1]}
                raw_res = tiff_resolution(path)
                row.update({f"raw_{k}": v for k, v in raw_res.items()})
                partner = annotated_of.get(path)
                if partner:
                    big_path, dx, dy, frac = partner
                    big = arrays[big_path]
                    hs, ws = raw.shape[:2]
                    mask = np.zeros(big.shape[:2], dtype=bool)
                    mask[dy:dy + hs, dx:dx + ws] = np.any(big[dy:dy + hs, dx:dx + ws] != raw, axis=-1)
                    outside = np.ones(big.shape[:2], dtype=bool)
                    outside[dy:dy + hs, dx:dx + ws] = False
                    if outside.any():
                        pad = big[outside]
                        background = Counter(map(tuple, pad[:: max(1, len(pad) // 20000)].tolist())).most_common(1)[0][0]
                        mask |= outside & np.any(big != np.array(background, dtype=np.uint8), axis=-1)
                    else:
                        background = None
                    Image.fromarray(big).save(VL_OUT / "annotated" / f"{key}.png")
                    Image.fromarray((mask * 255).astype(np.uint8)).save(VL_OUT / "label_masks" / f"{key}.png")
                    colors = Counter(map(tuple, big[mask].tolist())).most_common(4)
                    ann_res = tiff_resolution(big_path)
                    row.update({
                        "annotated_source": f"{set_name}/{big_path.relative_to(root)}",
                        "ann_height": big.shape[0], "ann_width": big.shape[1],
                        "offset_x": dx, "offset_y": dy, "match_frac": round(frac, 4),
                        "label_px": int(mask.sum()), "label_px_outside_raw": int((mask & outside).sum()),
                        "pad_background": json.dumps(background), "label_colors": json.dumps(colors),
                        **{f"ann_{k}": v for k, v in ann_res.items()},
                    })
                else:
                    log(f"  no annotated partner for {rel}")
                index_rows.append(row)
                records.append({"dataset": "vl", "image_id": key, "source_filename": f"{set_name}/{rel}",
                                "height": raw.shape[0], "width": raw.shape[1],
                                "path": str(VL_OUT / "images" / f"{key}.png")})
            del arrays
        log(f"vl/{set_name}: {len(images)} image files, {len(others)} other files, "
            f"{sum(r['set'] == set_name for r in index_rows)} raw exported")
    (VL_OUT / "tree.txt").write_text("\n".join(tree_lines) + "\n", encoding="utf-8")
    if preview_lines:
        (VL_OUT / "labels_preview.txt").write_text("\n".join(preview_lines) + "\n", encoding="utf-8")
    if index_rows:
        fields = []
        for row in index_rows:
            fields.extend(k for k in row if k not in fields)
        with (VL_OUT / "vl_index.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(index_rows)
    paired = sum(1 for r in index_rows if r.get("annotated_source"))
    log(f"vl: {len(records)} raw PNG exported, {paired} with annotated partner")
    log("===== VL INDEX SAMPLE =====")
    for row in index_rows[:6]:
        print(row, flush=True)
    for line in preview_lines[:200]:
        print(line, flush=True)
    return records


def scan_imagej(records: list[dict], dataset: str) -> None:
    """Record ImageJ metadata / overlay ROIs embedded in TIFF files (tags 50838/50839)."""
    meta_rows, roi_rows = [], []
    for record in records:
        path = Path(record["path"])
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        info = tiff_metadata(path)
        if info.get("has_ij_metadata") or info.get("imagej_description"):
            meta_rows.append({"image_id": record["image_id"], **info})
        if info.get("has_ij_metadata"):
            for row in parse_rois(path):
                roi_rows.append({"image_id": record["image_id"], **row})
    log(f"imagej scan {dataset}: {len(meta_rows)} tiffs with ImageJ metadata, {len(roi_rows)} ROI rows")
    for rows, name in ((meta_rows, "imagej_meta"), (roi_rows, "imagej_rois")):
        if rows:
            fields = []
            for row in rows:
                fields.extend(k for k in row if k not in fields)
            with (MASKBANK / f"{name}_{dataset}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    for row in meta_rows[:3]:
        print({k: str(v)[:300] for k, v in row.items()}, flush=True)
    for row in roi_rows[:8]:
        print({k: str(v)[:300] for k, v in row.items()}, flush=True)


def stage_data() -> None:
    MASKBANK.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "roifile"], check=False)
    records = collect_test()
    try:
        records += collect_benchmarks()
    except Exception as error:  # noqa: BLE001
        log(f"ERROR benchmarks: {error!r}")
    try:
        records += collect_vl()
    except Exception as error:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        log(f"ERROR vl: {error!r}")
    for dataset in ("test", "static"):
        try:
            scan_imagej([r for r in records if r["dataset"] == dataset], dataset)
        except Exception as error:  # noqa: BLE001
            log(f"imagej scan {dataset} failed: {error!r}")
    with INTERNAL_MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "image_id", "source_filename", "height", "width", "path"])
        writer.writeheader()
        writer.writerows(records)
    with (MASKBANK / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "image_id", "source_filename", "height", "width"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    counts = {name: sum(r["dataset"] == name for r in records) for name in DATASETS}
    log(f"manifest written: {counts}")


# --------------------------------------------------------------------------- torch stage
def _largest_true_run(values: np.ndarray) -> tuple[int, int]:
    padded = np.pad(values.astype(np.int8), 1)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, len(values)
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def training_viewport(image: np.ndarray) -> tuple[int, int, int, int]:
    """Verbatim copy of kaggle/aponeurosis_unetpp_384/train.py:training_viewport."""
    gray = image.astype(np.float32)
    vertical_texture = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    left, right = _largest_true_run(vertical_texture > 1.0)
    if right - left < image.shape[1] * 0.35:
        left, right = 0, image.shape[1]
    horizontal_texture = np.abs(np.diff(gray[:, left:right], axis=1)).mean(axis=1)
    top, bottom = _largest_true_run(horizontal_texture > 0.5)
    if bottom - top < image.shape[0] * 0.45:
        top, bottom = 0, image.shape[0]
    return left, top, right, bottom


def ensure_smp():
    try:
        import segmentation_models_pytorch as smp

        if smp.__version__ != "0.5.0":
            raise ImportError(f"smp {smp.__version__} != 0.5.0")
    except ImportError as error:
        log(f"installing segmentation-models-pytorch==0.5.0 ({error})")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "segmentation-models-pytorch==0.5.0"])
        import importlib

        import segmentation_models_pytorch as smp

        importlib.reload(smp)
    return smp


def stage_torch() -> None:
    import cv2
    import torch
    from PIL import Image

    smp = ensure_smp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"torch={torch.__version__} smp={smp.__version__} cv2={cv2.__version__} device={device} "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    torch.backends.cudnn.benchmark = False
    manifest = read_manifest()
    viewport_rows = []
    run_info = {}
    for model_key, filename in TORCH_MODELS.items():
        try:
            path = find_input_file(filename)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            meta = {key: value for key, value in checkpoint.items() if key != "model"}
            log(f"{model_key}: {path} keys={sorted(checkpoint.keys())} meta={meta}")
            use_viewport = bool(checkpoint.get("viewport_preprocessing", False))
            size = int(checkpoint.get("size", 384))
            model = smp.UnetPlusPlus(encoder_name=checkpoint.get("encoder", "resnet34"), encoder_weights=None,
                                     in_channels=1, classes=1, activation=None)
            state = {k.removeprefix("module."): v for k, v in checkpoint["model"].items()}
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            run_info[model_key] = {"checkpoint": str(path), "viewport_crop": use_viewport, "size": size,
                                   "meta": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in meta.items()}}
            log(f"{model_key}: viewport_crop={use_viewport} size={size}")
        except Exception as error:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            log(f"ERROR loading {model_key}: {error!r}")
            continue
        for dataset in DATASETS:
            records = manifest.get(dataset, [])
            if not records:
                continue
            started = time.time()
            mean_bank: dict[str, np.ndarray] = {}
            noflip_bank: dict[str, np.ndarray] = {}
            failed = 0
            for index, record in enumerate(records):
                try:
                    with Image.open(record["path"]) as image:
                        gray = np.asarray(image.convert("L"), dtype=np.uint8)
                    height, width = gray.shape
                    assert (height, width) == (record["height"], record["width"]), "size mismatch vs manifest"
                    if use_viewport:
                        left, top, right, bottom = training_viewport(gray)
                    else:
                        left, top, right, bottom = 0, 0, width, height
                    if model_key == "ours_apo384":
                        viewport_rows.append({"dataset": dataset, "image_id": record["image_id"], "left": left,
                                              "top": top, "right": right, "bottom": bottom,
                                              "height": height, "width": width})
                    crop = gray[top:bottom, left:right]
                    x = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                    batch = np.ascontiguousarray(np.stack([x, x[:, ::-1]])[:, None])
                    with torch.inference_mode():
                        logits = model(torch.from_numpy(batch).to(device))
                        probability = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
                    pass0 = probability[0]
                    pass1 = probability[1][:, ::-1]
                    full_mean = np.zeros((height, width), dtype=np.float32)
                    full_noflip = np.zeros((height, width), dtype=np.float32)
                    crop_size = (right - left, bottom - top)
                    full_mean[top:bottom, left:right] = cv2.resize(
                        np.ascontiguousarray(0.5 * (pass0 + pass1)), crop_size, interpolation=cv2.INTER_LINEAR)
                    full_noflip[top:bottom, left:right] = cv2.resize(
                        np.ascontiguousarray(pass0), crop_size, interpolation=cv2.INTER_LINEAR)
                    mean_bank[record["image_id"]] = to_uint8(full_mean)
                    noflip_bank[record["image_id"]] = to_uint8(full_noflip)
                except Exception as error:  # noqa: BLE001
                    failed += 1
                    log(f"  {model_key}/{dataset}/{record['image_id']} FAILED {error!r}")
                if (index + 1) % 50 == 0 or index + 1 == len(records):
                    log(f"  {model_key} {dataset} {index + 1}/{len(records)} ({time.time() - started:.1f}s)")
            save_bank(dataset, model_key, mean_bank)
            save_bank(dataset, f"{model_key}_noflip", noflip_bank)
            append_summary({"stage": "torch", "model": model_key, "dataset": dataset, "n_images": len(mean_bank),
                            "n_failed": failed, "seconds": round(time.time() - started, 1),
                            **prob_stats(list(mean_bank.values())),
                            "notes": f"viewport_crop={use_viewport} size={size} hflipTTA"})
            del mean_bank, noflip_bank
            gc.collect()
        del model
        torch.cuda.empty_cache()
    if viewport_rows:
        with (MASKBANK / "viewports_ours_apo384.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(viewport_rows[0]))
            writer.writeheader()
            writer.writerows(viewport_rows)
    (MASKBANK / "run_info_torch.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- tf stage
def build_vgg16_unet(keras, input_shape=(512, 512, 3)):
    """Architecture of DL_Track_US v0.3.1 (organiser notebook paulritsche/dl-track-0-3-1-aponeurosis-model)."""
    layers = keras.layers

    def conv_block(inputs, filters):
        x = layers.Conv2D(filters, 3, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)

    def decoder_block(inputs, skip, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
        x = layers.Concatenate()([x, skip])
        return conv_block(x, filters)

    inputs = layers.Input(input_shape)
    vgg16 = keras.applications.VGG16(include_top=False, weights=None, input_tensor=inputs)
    s1 = vgg16.get_layer("block1_conv2").output
    s2 = vgg16.get_layer("block2_conv2").output
    s3 = vgg16.get_layer("block3_conv3").output
    s4 = vgg16.get_layer("block4_conv3").output
    b1 = vgg16.get_layer("block5_conv3").output
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    return keras.Model(inputs, outputs, name="VGG16_U-Net")


def keras2_compat_objects(tf) -> dict:
    """Keras-3 shims for Keras-2 h5 configs (Conv2DTranspose carried a 'groups' kwarg in Keras 2)."""
    base = tf.keras.layers.Conv2DTranspose

    class Conv2DTransposeCompat(base):
        def __init__(self, *args, groups=None, **kwargs):  # noqa: ARG002 - Keras 2 always wrote groups=1
            super().__init__(*args, **kwargs)

    return {"Conv2DTranspose": Conv2DTransposeCompat}


def load_keras_model(path: Path):
    import tensorflow as tf

    errors = []
    try:
        model = tf.keras.models.load_model(str(path), compile=False)
        return model, "tf.keras.models.load_model(compile=False)"
    except Exception as error:  # noqa: BLE001
        errors.append(f"load_model: {error!r}"[:1500])
        log(f"load_model failed for {path.name}: {error!r}"[:600])
    try:
        model = tf.keras.models.load_model(str(path), compile=False, custom_objects=keras2_compat_objects(tf))
        return model, "tf.keras.models.load_model(compile=False, custom_objects={Conv2DTranspose: drops Keras-2 'groups'})"
    except Exception as error:  # noqa: BLE001
        errors.append(f"load_model+compat: {error!r}"[:1500])
        log(f"load_model+compat failed for {path.name}: {error!r}"[:1500])
    try:
        model = build_vgg16_unet(tf.keras)
        model.load_weights(str(path))
        return model, "rebuilt VGG16-UNet + load_weights(h5)"
    except Exception as error:  # noqa: BLE001
        errors.append(f"rebuild+load_weights: {error!r}"[:1500])
        log(f"rebuild+load_weights failed for {path.name}: {error!r}"[:1500])
    raise RuntimeError(" || ".join(errors))


def read_bgr(path: str) -> np.ndarray:
    """cv2.imread(path, 1) exactly like DL_Track_US importAndReshapeImage; PIL fallback."""
    import cv2

    image = cv2.imread(path, 1)
    if image is None:
        from PIL import Image

        with Image.open(path) as pil:
            image = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        log(f"  cv2.imread failed for {path}; used PIL fallback")
    return image


def stage_tf() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import cv2
    import tensorflow as tf
    from skimage import __version__ as skimage_version
    from skimage.transform import resize as sk_resize

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as error:  # noqa: BLE001
            log(f"memory growth: {error!r}")
    keras_version = getattr(tf.keras, "__version__", "?")
    log(f"tensorflow={tf.__version__} tf.keras={keras_version} legacy={os.environ.get('TF_USE_LEGACY_KERAS', '0')} "
        f"skimage={skimage_version} cv2={cv2.__version__} gpus={gpus}")
    try:
        img_to_array = tf.keras.utils.img_to_array
    except AttributeError:
        img_to_array = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    manifest = read_manifest()
    run_info = {"tensorflow": tf.__version__, "keras": keras_version, "skimage": skimage_version,
                "legacy_keras": os.environ.get("TF_USE_LEGACY_KERAS", "0"), "models": {}}
    load_failures = 0
    for model_key, (url, filename) in DLT_MODELS.items():
        try:
            path = download(url, MODELS / filename)
            model, how = load_keras_model(path)
            log(f"{model_key}: loaded via {how}; input={model.input_shape} output={model.output_shape} "
                f"params={model.count_params()}")
            run_info["models"][model_key] = {"file": filename, "loader": how, "input": str(model.input_shape),
                                             "output": str(model.output_shape), "params": int(model.count_params())}
        except Exception as error:  # noqa: BLE001
            load_failures += 1
            log(f"ERROR loading {model_key}: {error!r}"[:3000])
            continue
        in_h, in_w = int(model.input_shape[1] or 512), int(model.input_shape[2] or 512)
        if not how.startswith("rebuilt"):
            # cross-check: the organiser-notebook architecture + load_weights must give the same output
            try:
                rebuilt = build_vgg16_unet(tf.keras, (in_h, in_w, 3))
                rebuilt.load_weights(str(path))
                diffs = []
                for record in manifest.get("test", [])[:3]:
                    image = read_bgr(record["path"]).astype(np.float32)[None]
                    x = (sk_resize(image, (1, in_h, in_w, 3), mode="constant", preserve_range=True) / 255.0).astype(np.float32)
                    a = np.asarray(model(x, training=False))
                    b = np.asarray(rebuilt(x, training=False))
                    diffs.append(float(np.abs(a - b).max()))
                log(f"{model_key}: cross-check vs rebuilt architecture max|diff| per image = {diffs}")
                run_info["models"][model_key]["crosscheck_max_abs_diff_vs_rebuilt"] = diffs
                del rebuilt
            except Exception as error:  # noqa: BLE001
                log(f"{model_key}: cross-check failed {error!r}"[:1000])
        for dataset in DATASETS:
            records = manifest.get(dataset, [])
            if not records:
                continue
            started = time.time()
            mean_bank: dict[str, np.ndarray] = {}
            noflip_bank: dict[str, np.ndarray] = {}
            failed = 0
            for index, record in enumerate(records):
                try:
                    image = read_bgr(record["path"])
                    height, width = image.shape[:2]
                    assert (height, width) == (record["height"], record["width"]), "size mismatch vs manifest"
                    array = img_to_array(image)
                    array = np.reshape(array, [-1, height, width, 3])
                    array = sk_resize(array, (1, in_h, in_w, 3), mode="constant", preserve_range=True)
                    array = array / 255.0
                    batch = np.concatenate([array, array[:, :, ::-1, :]], axis=0).astype(np.float32)
                    prediction = model(batch, training=False)
                    prediction = np.asarray(prediction, dtype=np.float32)
                    pass0 = prediction[0, :, :, 0]
                    pass1 = prediction[1, :, ::-1, 0]
                    mean = cv2.resize(np.ascontiguousarray(0.5 * (pass0 + pass1)), (width, height),
                                      interpolation=cv2.INTER_LINEAR)
                    noflip = cv2.resize(np.ascontiguousarray(pass0), (width, height), interpolation=cv2.INTER_LINEAR)
                    mean_bank[record["image_id"]] = to_uint8(mean)
                    noflip_bank[record["image_id"]] = to_uint8(noflip)
                except Exception as error:  # noqa: BLE001
                    failed += 1
                    log(f"  {model_key}/{dataset}/{record['image_id']} FAILED {error!r}")
                if (index + 1) % 50 == 0 or index + 1 == len(records):
                    log(f"  {model_key} {dataset} {index + 1}/{len(records)} ({time.time() - started:.1f}s)")
            save_bank(dataset, model_key, mean_bank)
            save_bank(dataset, f"{model_key}_noflip", noflip_bank)
            append_summary({"stage": "tf", "model": model_key, "dataset": dataset, "n_images": len(mean_bank),
                            "n_failed": failed, "seconds": round(time.time() - started, 1),
                            **prob_stats(list(mean_bank.values())),
                            "notes": "cv2.imread BGR, skimage resize 512 preserve_range, /255, hflipTTA"})
            del mean_bank, noflip_bank
            gc.collect()
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    # optional IFSS probe: only documents the input signature
    try:
        url, filename = DLT_IFSS
        path = download(url, MODELS / filename)
        try:
            model = tf.keras.models.load_model(str(path), compile=False)
            run_info["ifss"] = {"input": str(model.input_shape), "output": str(model.output_shape)}
            log(f"IFSS loaded: input={model.input_shape} output={model.output_shape}")
            del model
        except Exception as error:  # noqa: BLE001
            run_info["ifss"] = {"load_error": repr(error)[:1000],
                                "decision": "skipped: 5-D sequence input and pickled python Lambda layers; "
                                            "unsafe deserialization deliberately NOT enabled"}
            log(f"IFSS load failed (skipped, unsafe Lambda not enabled): {error!r}"[:1500])
            try:
                import h5py

                with h5py.File(path, "r") as handle:
                    config = handle.attrs.get("model_config")
                    text = config.decode() if isinstance(config, bytes) else str(config)
                    run_info["ifss"]["config_head"] = text[:3000]
                    log(f"IFSS model_config head: {text[:1500]}")
            except Exception as inner:  # noqa: BLE001
                log(f"IFSS h5 inspect failed: {inner!r}")
    except Exception as error:  # noqa: BLE001
        log(f"IFSS probe skipped: {error!r}")
    name = "run_info_tf_legacy.json" if os.environ.get("TF_USE_LEGACY_KERAS") == "1" else "run_info_tf.json"
    (MASKBANK / name).write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")
    if load_failures == len(DLT_MODELS):
        sys.exit(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["data", "torch", "tf"], required=True)
    args = parser.parse_args()
    log(f"stage={args.stage} smoke={SMOKE}")
    {"data": stage_data, "torch": stage_torch, "tf": stage_tf}[args.stage]()
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
