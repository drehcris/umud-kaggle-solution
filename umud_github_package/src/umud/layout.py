"""Test-set layout table from the images alone: scanner family, image size, px/cm, viewport and video clips.

    python -m umud.layout <test_image_dir> <out.csv>

Columns: image_id, size ("WxH"), ext, family, px_per_cm, vp_x1, vp_y1, vp_x2, vp_y2, ncc_prev, clip, group.

* family / px_per_cm / viewport: ``umud.geometry.ultrasound_metadata`` (the scanner scale-bar and viewport decoder,
  adapted from AmbrosM "UMUD Quick and Dirty", Apache-2.0; see THIRD_PARTY_NOTICES.md);
* clip: ``umud.clips.detect_clips`` (consecutive-frame correlation of the viewport thumbnails; no file-name ranges);
* group: the scanner group used by the pipeline = family, plus "_<size>" for the Telemed ArtUS layouts.

This replaces the research-era table ``data-labels-geometry/out/test_layout.csv`` (same family / size / px_per_cm
values; its clip column came from the file-name rule (n - 56) // 5, which the content detector reproduces exactly:
tests/test_final_package.py). One image at a time, < 0.2 GB.

GPL-3.0-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from umud import clips as clipmod
from umud.geometry import ultrasound_metadata

IMAGE_SUFFIXES = {".tif", ".tiff", ".png"}
SIZE_SUFFIX_FAMILIES = ("telemed_artus",)


def image_paths(test_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def scanner_group(family: str, size: str) -> str:
    return f"{family}_{size}" if family in SIZE_SUFFIX_FAMILIES else family


def build_layout(test_dir: str | Path, threshold: float = clipmod.DEFAULT_THRESHOLD,
                 min_frames: int = clipmod.DEFAULT_MIN_FRAMES) -> tuple[pd.DataFrame, dict]:
    """(layout frame indexed by image_id, clip-detection summary)."""
    rows = []

    def frames():                      # one image in memory at a time; detect_clips keeps one thumbnail
        for p in image_paths(test_dir):
            with Image.open(p) as im:
                w, h = im.size
                g = np.asarray(im.convert("L"))
            md = ultrasound_metadata(g, p)
            x1, y1, x2, y2 = (int(v) for v in md.viewport)
            rows.append(dict(image_id=p.name, size=f"{w}x{h}", ext=p.suffix, family=md.scanner_family,
                             px_per_cm=md.px_per_cm, vp_x1=x1, vp_y1=y1, vp_x2=x2, vp_y2=y2))
            yield clipmod.FrameInfo(p.name, (w, h), clipmod.thumbnail(g, md.viewport))

    res = clipmod.detect_clips(frames(), threshold=threshold, min_frames=min_frames)
    lay = pd.DataFrame(rows).set_index("image_id")
    lay["ncc_prev"] = pd.Series(res.ncc_prev, index=res.image_ids)
    lay["clip"] = pd.Series(res.clip, index=res.image_ids).astype(int)
    lay["group"] = [scanner_group(f, s) for f, s in zip(lay["family"], lay["size"])]
    sizes = pd.Series(res.clip)[pd.Series(res.clip) >= 0].value_counts()
    summary = dict(n_images=len(lay), n_clips=res.n_clips(), frames_per_clip=sorted(set(sizes.tolist())),
                   threshold=threshold, min_frames=min_frames, **res.margins(),
                   clip_groups=sorted(set(lay.loc[lay["clip"] >= 0, "group"])))
    return lay, summary


def write_layout(lay: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lay.to_csv(path)


def read_layout(path: str | Path) -> pd.DataFrame:
    lay = pd.read_csv(path).set_index("image_id")
    lay["clip"] = lay["clip"].astype(int)
    if "group" not in lay:
        lay["group"] = [scanner_group(f, s) for f, s in zip(lay["family"], lay["size"])]
    return lay


def load_or_build(test_dir: str | Path, path: str | Path) -> pd.DataFrame:
    """Read the layout table at `path`, building it from the images first if it does not exist."""
    path = Path(path)
    if not path.exists():
        lay, _ = build_layout(test_dir)
        write_layout(lay, path)
    return read_layout(path)


def main(argv: list[str] | None = None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 2:
        print(__doc__)
        return 2
    lay, s = build_layout(a[0])
    write_layout(lay, a[1])
    print(f"{len(lay)} images, {s['n_clips']} clips of {s['frames_per_clip']} frames; NCC within >= "
          f"{s['min_within']:.3f}, across <= {s['max_across']:.3f} -> {a[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
