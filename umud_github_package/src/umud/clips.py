"""Video-clip detection from image content (consecutive-frame correlation).

Some test images are consecutive frames of one ultrasound cine clip (the 28 Telemed ArtUS clips of 5 near-identical
frames). The pipeline takes the median of each clip's frames, so the clips must be known. They are found from the
pixels alone, never from file-name ranges:

1. every image is reduced to a grey thumbnail of its ultrasound viewport (``thumbnail``: the viewport crop resized
   to 160 x 120 with area averaging, then z-normalised);
2. images are visited in file-name order and each one is compared with the previous one by normalised
   cross-correlation (NCC = mean of the product of the two z-normalised thumbnails);
3. two neighbours are linked when they have the same image size and NCC >= ``threshold``;
4. every run of >= ``min_frames`` linked images is a clip; clips are numbered 0, 1, ... in file-name order and
   every other image gets clip -1.

On the UMUD test set the within-clip NCC is >= 0.951 and the NCC between any two consecutive images that are not
frames of one clip is <= 0.568 (the earlier ``s07_test_clips`` analysis), so the default threshold 0.8 sits in a
wide gap. Only one thumbnail (77 kB) is held at a time.

GPL-3.0-only. Own code (UMUD challenge solution).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import cv2
import numpy as np

THUMB_SIZE = (160, 120)       # (width, height) of the viewport thumbnail
DEFAULT_THRESHOLD = 0.8
DEFAULT_MIN_FRAMES = 2


def thumbnail(gray: np.ndarray, viewport: tuple[int, int, int, int] | None = None,
              size: tuple[int, int] = THUMB_SIZE) -> np.ndarray:
    """z-normalised float32 thumbnail of the viewport crop (x1, y1, x2, y2) of a grey image."""
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    h, w = g.shape[:2]
    if viewport is not None:
        x1, y1, x2, y2 = (int(v) for v in viewport)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            g = g[y1:y2, x1:x2]
    t = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((t - t.mean()) / (t.std() + 1e-6)).astype(np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two z-normalised thumbnails of the same shape."""
    if a.shape != b.shape:
        return float("nan")
    return float(np.mean(a * b))


@dataclass(frozen=True)
class FrameInfo:
    image_id: str
    size: tuple[int, int]             # (width, height) of the full image
    thumb: np.ndarray


@dataclass(frozen=True)
class ClipResult:
    image_ids: list[str]
    clip: list[int]                   # clip index per image (-1: not part of a clip)
    ncc_prev: list[float]             # NCC with the previous image in file-name order (NaN for the first / size change)

    def as_dict(self) -> dict[str, int]:
        return dict(zip(self.image_ids, self.clip))

    def n_clips(self) -> int:
        return len({c for c in self.clip if c >= 0})

    def margins(self) -> dict[str, float]:
        """Smallest NCC inside a clip and largest NCC between consecutive images of different clips."""
        inside, across = [], []
        for i in range(1, len(self.clip)):
            v = self.ncc_prev[i]
            if not np.isfinite(v):
                continue
            (inside if self.clip[i] >= 0 and self.clip[i] == self.clip[i - 1] else across).append(v)
        return dict(min_within=float(min(inside)) if inside else float("nan"),
                    max_across=float(max(across)) if across else float("nan"))


def detect_clips(frames: Iterable[FrameInfo], threshold: float = DEFAULT_THRESHOLD,
                 min_frames: int = DEFAULT_MIN_FRAMES) -> ClipResult:
    """Clips from a stream of frames in file-name order (only the previous thumbnail is kept)."""
    ids: list[str] = []
    link: list[bool] = []             # link[i]: image i is linked to image i - 1
    nccs: list[float] = []
    prev: FrameInfo | None = None
    for f in frames:
        v = ncc(prev.thumb, f.thumb) if prev is not None and prev.size == f.size else float("nan")
        ids.append(f.image_id)
        nccs.append(v)
        link.append(bool(np.isfinite(v) and v >= threshold))
        prev = f
    clip = [-1] * len(ids)
    k, i = 0, 0
    while i < len(ids):
        j = i + 1
        while j < len(ids) and link[j]:
            j += 1
        if j - i >= min_frames:
            for t in range(i, j):
                clip[t] = k
            k += 1
        i = j
    return ClipResult(ids, clip, nccs)


def frames_from_images(paths: Iterable, viewport_of=None) -> Iterator[FrameInfo]:
    """FrameInfo per image path (sorted by the caller), reading one image at a time.
    viewport_of(gray, path) -> (x1, y1, x2, y2) or None; default: the scanner-layout decoder of umud.geometry."""
    from pathlib import Path

    from PIL import Image

    if viewport_of is None:
        from umud.geometry import ultrasound_metadata

        def viewport_of(g, p):  # noqa: E306
            m = ultrasound_metadata(g, p)
            return m.viewport
    for p in paths:
        p = Path(p)
        with Image.open(p) as im:
            g = np.asarray(im.convert("L"))
        yield FrameInfo(p.name, (g.shape[1], g.shape[0]), thumbnail(g, viewport_of(g, p)))
        del g
