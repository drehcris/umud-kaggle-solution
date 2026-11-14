from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


Pair = tuple[Path, Path]


def image_fingerprint(path: Path) -> str:
    """Hash decoded pixels so container metadata does not hide duplicate images."""

    pixels = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype=np.int64).tobytes())
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def deduplicate_pairs(pairs: list[Pair]) -> tuple[list[Pair], int]:
    seen: set[tuple[str, str]] = set()
    unique: list[Pair] = []
    for image_path, mask_path in pairs:
        key = (image_fingerprint(image_path), image_fingerprint(mask_path))
        if key in seen:
            continue
        seen.add(key)
        unique.append((image_path, mask_path))
    return unique, len(pairs) - len(unique)


def grouped_split_pairs(
    pairs: list[Pair], val_fraction: float, seed: int
) -> tuple[list[Pair], list[Pair]]:
    """Split image groups, keeping identical source images on the same side."""

    groups: dict[str, list[Pair]] = {}
    for pair in pairs:
        groups.setdefault(image_fingerprint(pair[0]), []).append(pair)

    group_keys = np.asarray(sorted(groups), dtype=object)
    np.random.default_rng(seed).shuffle(group_keys)
    target = max(1, round(len(pairs) * val_fraction))
    validation: list[Pair] = []
    training: list[Pair] = []
    for key in group_keys:
        destination = validation if len(validation) < target else training
        destination.extend(groups[str(key)])
    return training, validation
