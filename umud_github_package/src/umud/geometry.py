from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.measure import label, regionprops


@dataclass(frozen=True)
class Line:
    """Line represented as y = slope * x + intercept in pixel coordinates."""

    slope: float
    intercept: float

    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan(self.slope)))

    def y(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.slope * x + self.intercept


@dataclass(frozen=True)
class Architecture:
    pa_deg: float
    fl_mm: float
    mt_mm: float
    apo_angle_deg: float
    fasc_angle_deg: float


@dataclass(frozen=True)
class UltrasoundMetadata:
    """Physical scale and inclusive-exclusive ultrasound viewport."""

    px_per_cm: float
    viewport: tuple[int, int, int, int]
    scanner_family: str


def read_gray(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image)


def binarize_mask(mask: np.ndarray, threshold: int | None = None) -> np.ndarray:
    if threshold is None:
        maximum = float(np.max(mask)) if mask.size else 0.0
        threshold = 0.5 if maximum <= 1.0 else maximum / 2.0
    return mask > threshold


def fit_line_xy(points_yx: np.ndarray) -> Line | None:
    if len(points_yx) < 8:
        return None
    y = points_yx[:, 0].astype(np.float64)
    x = points_yx[:, 1].astype(np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return Line(float(slope), float(intercept))


def aponeurosis_candidates(mask: np.ndarray) -> list[Line]:
    binary = binarize_mask(mask)
    labels = label(binary)
    components = sorted(regionprops(labels), key=lambda r: r.area, reverse=True)
    lines: list[tuple[float, Line]] = []
    for component in components[:8]:
        if component.area < 40:
            continue
        coords = component.coords
        line = fit_line_xy(coords)
        if line is not None:
            lines.append((float(np.median(coords[:, 0])), line))
    lines.sort(key=lambda item: item[0])
    return [line for _, line in lines]


def fit_aponeuroses(
    mask: np.ndarray, fasc_mask: np.ndarray | None = None
) -> tuple[Line | None, Line | None]:
    lines = aponeurosis_candidates(mask)
    if len(lines) < 2:
        return None, None
    if fasc_mask is None or not np.any(fasc_mask > 0):
        return lines[0], lines[-1]

    ys, xs = np.nonzero(fasc_mask > 0)
    if len(xs) > 4000:
        step = max(1, len(xs) // 4000)
        xs, ys = xs[::step], ys[::step]
    best: tuple[float, Line, Line] | None = None
    for upper_index, upper in enumerate(lines[:-1]):
        for lower in lines[upper_index + 1 :]:
            upper_y = upper.y(xs)
            lower_y = lower.y(xs)
            valid_order = lower_y > upper_y
            if valid_order.mean() < 0.9:
                continue
            margin = mask.shape[0] * 0.025
            enclosed = (ys >= upper_y - margin) & (ys <= lower_y + margin)
            coverage = float(enclosed.mean())
            separation = float(np.median(lower_y - upper_y)) / mask.shape[0]
            if not 0.06 <= separation <= 0.75:
                continue
            # Prefer high fascicle coverage, then the tightest anatomical compartment.
            score = coverage - 0.10 * separation
            if best is None or score > best[0]:
                best = (score, upper, lower)
    if best is None:
        return lines[0], lines[-1]
    return best[1], best[2]


def dominant_fascicle_line(mask: np.ndarray) -> Line | None:
    binary = binarize_mask(mask)
    labels = label(binary)
    components = sorted(regionprops(labels), key=lambda r: r.area, reverse=True)
    candidates: list[tuple[float, Line]] = []
    for component in components[:24]:
        if component.area < 15:
            continue
        coords = component.coords
        line = fit_line_xy(coords)
        if line is None:
            continue
        length = max(component.axis_major_length, component.area)
        candidates.append((float(length), line))
    if not candidates:
        return None
    angles = np.array([line.angle_deg() for _, line in candidates], dtype=np.float64)
    weights = np.array([weight for weight, _ in candidates], dtype=np.float64)
    order = np.argsort(angles)
    angles = angles[order]
    weights = weights[order]
    cutoff = weights.sum() / 2
    angle = float(angles[np.searchsorted(np.cumsum(weights), cutoff)])
    slope = float(np.tan(np.radians(angle)))
    intercepts = np.array([line.intercept for _, line in candidates], dtype=np.float64)
    return Line(slope=slope, intercept=float(np.median(intercepts)))


def line_angle_difference_deg(first: Line, second: Line) -> float:
    """Return the acute, orientation-independent angle between two lines."""

    difference = first.angle_deg() - second.angle_deg()
    return float(abs((difference + 90.0) % 180.0 - 90.0))


def line_intersection(first: Line, second: Line) -> tuple[float, float] | None:
    denominator = first.slope - second.slope
    if abs(denominator) < 1e-6:
        return None
    x = (second.intercept - first.intercept) / denominator
    return float(x), float(first.y(x))


def fascicle_architecture_candidates(
    mask: np.ndarray,
    upper: Line,
    lower: Line,
    min_pennation_deg: float = 8.0,
    max_pennation_deg: float = 40.0,
) -> list[tuple[float, float, float, Line]]:
    """Measure individual fascicle fragments after linear extrapolation."""

    labels = label(binarize_mask(mask))
    components = sorted(regionprops(labels), key=lambda region: region.area, reverse=True)
    height, width = mask.shape
    min_length = max(12.0, width * 0.025)
    candidates: list[tuple[float, float, float, Line]] = []

    for component in components[:64]:
        major = float(component.axis_major_length)
        minor = max(float(component.axis_minor_length), 1.0)
        if component.area < 15 or major < min_length or major / minor < 1.5:
            continue

        line = fit_line_xy(component.coords)
        if line is None:
            continue
        centroid_y, centroid_x = component.centroid
        upper_y = float(upper.y(centroid_x))
        lower_y = float(lower.y(centroid_x))
        if not upper_y - 0.03 * height <= centroid_y <= lower_y + 0.03 * height:
            continue

        pennation = line_angle_difference_deg(line, lower)
        if not min_pennation_deg <= pennation <= max_pennation_deg:
            continue

        upper_point = line_intersection(line, upper)
        lower_point = line_intersection(line, lower)
        if upper_point is None or lower_point is None:
            continue
        x_upper, y_upper = upper_point
        x_lower, y_lower = lower_point
        if max(x_upper, x_lower) < -0.75 * width or min(x_upper, x_lower) > 1.75 * width:
            continue
        length_px = float(np.hypot(x_upper - x_lower, y_upper - y_lower))
        if not 0.05 * height <= length_px <= 3.0 * width:
            continue

        weight = major * np.sqrt(major / minor)
        candidates.append((pennation, length_px, float(weight), line))

    return candidates


def line_distance_px(
    upper: Line,
    lower: Line,
    width: int,
    samples: int = 25,
    x_range: tuple[float, float] | None = None,
) -> float:
    left, right = x_range if x_range is not None else (0.0, float(width - 1))
    xs = np.linspace(left, right, samples)
    dy = lower.y(xs) - upper.y(xs)
    denom = np.sqrt(1.0 + ((upper.slope + lower.slope) / 2.0) ** 2)
    return float(np.median(np.abs(dy) / denom))


def vertical_line_distance_px(
    upper: Line,
    lower: Line,
    x_positions: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Average global-vertical separation, matching the challenge annotation protocol."""

    xs = np.asarray(x_positions, dtype=np.float64)
    return float(np.mean(np.abs(lower.y(xs) - upper.y(xs))))


def _largest_true_run(values: np.ndarray) -> tuple[int, int]:
    padded = np.pad(values.astype(np.int8), 1)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, len(values)
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def ultrasound_viewport(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find the largest dense ultrasound rectangle, excluding scanner UI panels."""

    gray = image.astype(np.float32)
    vertical_texture = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    x1, x2 = _largest_true_run(vertical_texture > 1.0)
    if x2 - x1 < image.shape[1] * 0.35:
        bright = image > max(8, int(np.percentile(image, 35)))
        x1, x2 = _largest_true_run(bright.mean(axis=0) > 0.35)
    horizontal_texture = np.abs(np.diff(gray[:, x1:x2], axis=1)).mean(axis=1)
    y1, y2 = _largest_true_run(horizontal_texture > 0.5)
    if y2 - y1 < image.shape[0] * 0.45:
        y1, y2 = 0, image.shape[0]
    return x1, y1, x2, y2


def _first_true(values: np.ndarray, start: int = 0) -> int | None:
    indexes = np.flatnonzero(values[start:])
    return int(start + indexes[0]) if len(indexes) else None


def _last_true(values: np.ndarray) -> int | None:
    indexes = np.flatnonzero(values)
    return int(indexes[-1]) if len(indexes) else None


def ultrasound_metadata(
    image: np.ndarray, image_path: str | Path | None = None
) -> UltrasoundMetadata:
    """Decode scale bars and viewports for the scanner layouts in the challenge.

    The per-layout probe pixels, tick rules and viewport boxes are adapted from AmbrosM,
    "UMUD Quick and Dirty" (https://www.kaggle.com/code/ambrosm/umud-quick-and-dirty),
    Apache-2.0. Modified: rewritten as a standalone function with an OCR fallback for
    unrecognised layouts. See THIRD_PARTY_NOTICES.md.
    """

    height, width = image.shape[:2]
    suffix = Path(image_path).suffix.lower() if image_path is not None else ""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if suffix == ".png" and (width, height) == (1200, 800):
        first_tick = _first_true(gray[:, 6] > 50)
        second_minor = _first_true(gray[:, 6] > 50, 150)
        second_major = _first_true(gray[:, 9] > 50, 150)
        last_tick = _last_true(gray[:, 6] > 50)
        if None not in (first_tick, second_minor, second_major, last_tick):
            assert first_tick is not None and second_minor is not None
            assert second_major is not None and last_tick is not None
            minor_gap = second_minor - first_tick
            major_gap = second_major - first_tick
            px_per_cm = float(major_gap if major_gap < 3 * minor_gap else minor_gap)
            half_width = width // 2
            half_span = int(np.argmin(gray[:, half_width:].sum(axis=0)))
            viewport = (half_width - half_span, first_tick, half_width + half_span, last_tick)
            if px_per_cm > 0:
                return UltrasoundMetadata(px_per_cm, viewport, "philips_lumify")

    if (width, height) == (1200, 800):
        right_ticks = np.all(np.abs(gray[87, 1147:1157].astype(int) - 175) <= 3)
        if right_ticks:
            first_tick = _first_true(gray[:, 1150] > 50)
            last_tick = _last_true(gray[:, 1150] > 50)
            if first_tick is not None and last_tick is not None:
                second_tick = _first_true(gray[:, 1150] > 50, first_tick + 20)
                if second_tick is not None and second_tick > first_tick:
                    tick_count = round((last_tick - first_tick) / (second_tick - first_tick))
                    if 7 <= tick_count <= 14:
                        px_per_cm = (last_tick - first_tick) / tick_count * 2.0
                        boxes = {
                            7: (142, 91, 1058, last_tick),
                            8: (163, 91, 1037, last_tick),
                            9: (211, 91, 989, last_tick),
                            10: (249, 91, 951, last_tick),
                            11: (282, 91, 918, last_tick),
                            12: (308, 91, 892, last_tick),
                            13: (331, 91, 869, last_tick),
                            14: (349, 91, 851, last_tick),
                        }
                        return UltrasoundMetadata(px_per_cm, boxes[tick_count], "siemens_juniper")
                    if tick_count == 15:
                        px_per_cm = (last_tick - first_tick) / 3.0
                        return UltrasoundMetadata(
                            px_per_cm, (142, 91, 1058, last_tick), "siemens_juniper"
                        )

        if np.all(gray[42, 67:74] > 115):
            return UltrasoundMetadata(
                (783.0 - 42.0) / 5.0, (171, 42, 1029, 798), "telemed_artus"
            )

    if (width, height) == (1088, 644):
        return UltrasoundMetadata(630.5 / 5.0, (140, 0, 947, 643), "telemed_artus")

    if height in {512, 513}:
        return UltrasoundMetadata((442.0 - 53.0) / 5.0, (0, 0, width, height - 10), "legacy_512")

    if height == 853:
        px_per_cm = (934.0 - 100.0) / 5.0 if width > 1000 else (879.0 - 44.0) / 5.0
        return UltrasoundMetadata(px_per_cm, (0, 0, width, height), "legacy_853")

    x1, y1, x2, y2 = ultrasound_viewport(gray)
    depth_mm = _ocr_depth_mm(image_path) if image_path is not None else None
    depth_mm = depth_mm or 45.0
    px_per_cm = 10.0 * max(1, y2 - y1) / depth_mm
    return UltrasoundMetadata(px_per_cm, (x1, y1, x2, y2), "fallback")


def _parse_depth_mm(text: str) -> float | None:
    text = text.lower().replace(",", ".")
    text = re.sub(r"(?<=\d)\s*(?:c?e|o)?m\b", "cm", text)
    matches = re.findall(r"(?:tiefe\s*)?-?(\d{1,2}(?:\.\d)?)\s*\.?\s*cm", text)
    values = []
    for match in matches:
        value = float(match)
        if value >= 10 and value in {30.0, 35.0, 40.0, 45.0, 50.0, 60.0}:
            value /= 10.0
        if 2.0 <= value <= 8.0:
            values.append(value * 10.0)
    millimeters = re.findall(r"(?:de(?:pth)?\s*)?(\d{2}(?:\.\d)?)\s*m[mn]\b", text)
    values.extend(float(value) for value in millimeters if 25.0 <= float(value) <= 80.0)
    return values[-1] if values else None


def _ocr_depth_mm(image_path: str | Path) -> float | None:
    path = Path(image_path)
    if not path.exists():
        return None

    source = Image.open(path).convert("L")
    width, height = source.size
    regions = [source]
    if width >= 800:
        regions.extend(
            [
                source.crop(
                    (
                        int(0.72 * width),
                        int(0.58 * height),
                        int(0.93 * width),
                        int(0.92 * height),
                    )
                ),
                source.crop((int(0.78 * width), 0, width, height)),
            ]
        )

    for region in regions:
        prepared = ImageOps.autocontrast(region.resize((region.width * 3, region.height * 3)))
        encoded = BytesIO()
        prepared.save(encoded, format="PNG")
        for page_mode in (11, 6, 7):
            try:
                result = subprocess.run(
                    ["tesseract", "stdin", "stdout", "--psm", str(page_mode)],
                    input=encoded.getvalue(),
                    check=False,
                    capture_output=True,
                    timeout=8,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            value = _parse_depth_mm(result.stdout.decode(errors="ignore"))
            if value is not None:
                return value
    return None


def _scan_window_start(image: np.ndarray) -> int | None:
    """Locate the first sustained ultrasound row in scanner screenshots."""

    _, width = image.shape
    center = image[:, int(0.25 * width) : int(0.75 * width)]
    active = center.mean(axis=1) > 15.0
    run = np.convolve(active.astype(np.int16), np.ones(20, dtype=np.int16), mode="valid")
    starts = np.flatnonzero(run == 20)
    return int(starts[0]) if len(starts) else None


def is_dynamic_ultrasound(image: np.ndarray, image_path: str | Path | None = None) -> bool:
    """Identify the two video-frame layouts present in the challenge test set."""

    height, width = image.shape
    suffix = Path(image_path).suffix.lower() if image_path is not None else ""
    if (width, height) == (1088, 644):
        return True
    if (width, height) != (1200, 800) or suffix not in {".tif", ".tiff"}:
        return False
    start = _scan_window_start(image)
    return start is not None and (start < 75 or start > 120)


def infer_pixel_spacing_mm(image: np.ndarray, image_path: str | Path | None = None) -> float:
    """Infer square-pixel spacing from scanner scale bars."""

    return 10.0 / ultrasound_metadata(image, image_path).px_per_cm


def _select_distinct_fascicles(
    candidates: list[tuple[float, float, float, Line]], width: int, height: int, count: int = 3
) -> list[tuple[float, float, float, Line]]:
    """Prefer long visible fragments while avoiding repeated pieces of one fascicle."""

    selected: list[tuple[float, float, float, Line]] = []
    center_x = width / 2.0
    min_separation = max(4.0, 0.025 * height)
    for candidate in sorted(candidates, key=lambda value: value[2], reverse=True):
        center_y = float(candidate[3].y(center_x))
        if all(abs(center_y - existing[3].y(center_x)) >= min_separation for existing in selected):
            selected.append(candidate)
        if len(selected) == count:
            break
    return selected or candidates[:1]


def architecture_from_masks(
    image: np.ndarray,
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray,
    pixel_spacing_mm: float | None = None,
    image_path: str | Path | None = None,
    min_pennation_deg: float = 8.0,
    max_pennation_deg: float = 40.0,
) -> Architecture | None:
    upper, lower = fit_aponeuroses(apo_mask, fasc_mask)
    if upper is None or lower is None:
        return None

    spacing = (
        pixel_spacing_mm
        if pixel_spacing_mm is not None
        else infer_pixel_spacing_mm(image, image_path=image_path)
    )
    height, width = image.shape[:2]
    metadata = ultrasound_metadata(image, image_path)
    left, _, right, _ = metadata.viewport
    sample_xs = np.linspace(left + 0.25 * (right - left), left + 0.75 * (right - left), 3)
    mt_mm = vertical_line_distance_px(upper, lower, sample_xs) * spacing

    candidates = fascicle_architecture_candidates(
        fasc_mask,
        upper,
        lower,
        min_pennation_deg=min_pennation_deg,
        max_pennation_deg=max_pennation_deg,
    )
    if candidates:
        selected = _select_distinct_fascicles(candidates, width, height)
        pennations = np.asarray([candidate[0] for candidate in selected])
        lengths_px = np.asarray([candidate[1] for candidate in selected])
        pa_deg = float(np.mean(pennations))
        fl_mm = float(np.mean(lengths_px)) * spacing
        representative_index = int(np.argmin(np.abs(pennations - pa_deg)))
        fascicle = selected[representative_index][3]
    else:
        fascicle = dominant_fascicle_line(fasc_mask)
        if fascicle is None:
            return None
        pa_deg = line_angle_difference_deg(fascicle, lower)
        if not min_pennation_deg <= pa_deg <= max_pennation_deg:
            return None
        fl_mm = float(mt_mm / max(np.sin(np.radians(pa_deg)), 1e-3))

    apo_angle = lower.angle_deg()
    fasc_angle = fascicle.angle_deg()
    return Architecture(
        pa_deg=float(pa_deg),
        fl_mm=float(fl_mm),
        mt_mm=float(mt_mm),
        apo_angle_deg=float(apo_angle),
        fasc_angle_deg=float(fasc_angle),
    )


def classical_fascicle_orientation(image: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=8)
    if lines is None:
        return 18.0
    lines = np.asarray(lines).reshape(-1, 4)
    angles: list[float] = []
    lengths: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = [float(v) for v in line]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if 5 <= abs(angle) <= 60:
            angles.append(angle)
            lengths.append(np.hypot(x2 - x1, y2 - y1))
    if not angles:
        return 18.0
    order = np.argsort(angles)
    angles_arr = np.asarray(angles)[order]
    weights_arr = np.asarray(lengths)[order]
    return float(angles_arr[np.searchsorted(np.cumsum(weights_arr), weights_arr.sum() / 2)])
