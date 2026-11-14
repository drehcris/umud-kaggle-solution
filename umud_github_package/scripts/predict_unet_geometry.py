from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from umud.geometry import (
    architecture_from_masks,
    dominant_fascicle_line,
    fit_aponeuroses,
    infer_pixel_spacing_mm,
    is_dynamic_ultrasound,
    read_gray,
    ultrasound_metadata,
)
from umud.postprocess import finalize_predictions
from umud.torch_unet import UNet


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def find_test_dir(root: Path) -> Path:
    for candidate in [
        root / "test_images_v2" / "test_set_v2",
        root / "test_images",
        root / "test",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find test images under {root}")


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, int, float]:
    checkpoint = torch.load(path, map_location=device)
    size = int(checkpoint.get("size", 512))
    architecture = checkpoint.get("arch", "UNet")
    if architecture == "UnetPlusPlus":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise RuntimeError(
                "This checkpoint uses U-Net++; install the training extras with "
                "`uv sync --extra train`."
            ) from error
        model = smp.UnetPlusPlus(
            encoder_name=str(checkpoint.get("encoder", "resnet34")),
            encoder_weights=None,
            in_channels=1,
            classes=1,
            activation=None,
        ).to(device)
    elif architecture == "UNet":
        model = UNet(base=int(checkpoint.get("base", 32))).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture!r}")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.umud_viewport_preprocessing = bool(checkpoint.get("viewport_preprocessing", False))
    return model, size, float(checkpoint.get("threshold", 0.5))


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image_path: Path,
    size: int,
    threshold: float,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    original = Image.open(image_path).convert("L")
    w, h = original.size
    viewport = (0, 0, w, h)
    if getattr(model, "umud_viewport_preprocessing", False):
        source = np.asarray(original)
        viewport = ultrasound_metadata(source, image_path).viewport
        original = original.crop(viewport)
    resized = original.resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(x[None, None]).to(device)
    prob_tensor = torch.sigmoid(model(tensor))
    if tta:
        flipped = torch.flip(tensor, dims=(-1,))
        flipped_prob = torch.flip(torch.sigmoid(model(flipped)), dims=(-1,))
        prob_tensor = 0.5 * (prob_tensor + flipped_prob)
    prob = prob_tensor[0, 0].detach().cpu().numpy()
    left, top, right, bottom = viewport
    prob = cv2.resize(prob, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = (prob > threshold).astype(np.uint8) * 255
    return mask


def submission_ids(root: Path, test_dir: Path) -> list[str]:
    actual_images = sorted(p.name for p in test_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    sample = root / "sample_submission.csv"
    if sample.exists():
        sample_df = pd.read_csv(sample, sep=None, engine="python", encoding="utf-8-sig")
        sample_df.columns = [name.lstrip("\ufeff") for name in sample_df.columns]
        sample_ids = sample_df["image_id"].astype(str).tolist()
        if len(sample_ids) >= len(actual_images):
            return sample_ids
    return actual_images


def save_debug_image(
    image: np.ndarray,
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray,
    out: Path,
) -> None:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[apo_mask > 0] = (70, 70, 255)
    overlay[fasc_mask > 0] = (80, 255, 80)
    overlay = cv2.addWeighted(base, 0.62, overlay, 0.38, 0)

    upper, lower = fit_aponeuroses(apo_mask, fasc_mask)
    fascicle = dominant_fascicle_line(fasc_mask)
    for line, color in ((upper, (0, 160, 255)), (lower, (0, 160, 255)), (fascicle, (255, 255, 0))):
        if line is None:
            continue
        x1, x2 = 0, image.shape[1] - 1
        cv2.line(
            overlay,
            (x1, int(round(line.y(x1)))),
            (x2, int(round(line.y(x2)))),
            color,
            2,
            cv2.LINE_AA,
        )
    masks = np.concatenate(
        [
            cv2.cvtColor(apo_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(fasc_mask, cv2.COLOR_GRAY2BGR),
        ],
        axis=1,
    )
    masks = cv2.resize(masks, (overlay.shape[1], overlay.shape[0] // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.concatenate([overlay, masks], axis=0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apo-model", type=Path, required=True)
    parser.add_argument("--fasc-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/submission_unet_geometry.csv"))
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--debug-every", type=int, default=20)
    parser.add_argument("--diagnostics-out", type=Path)
    parser.add_argument("--min-pennation", type=float, default=8.0)
    parser.add_argument("--apo-threshold", type=float)
    parser.add_argument("--fasc-threshold", type=float)
    parser.add_argument(
        "--benchmark-calibration",
        action="store_true",
        help="Opt in to experimental calibration fitted on the external expert benchmark.",
    )
    parser.add_argument("--tta", action="store_true", help="Average original and horizontal-flip predictions.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    apo_model, apo_size, apo_threshold = load_model(args.apo_model, device)
    fasc_model, fasc_size, fasc_threshold = load_model(args.fasc_model, device)
    apo_threshold = args.apo_threshold if args.apo_threshold is not None else apo_threshold
    fasc_threshold = (
        args.fasc_threshold if args.fasc_threshold is not None else fasc_threshold
    )
    test_dir = find_test_dir(args.root)

    rows = []
    for index, image_id in enumerate(tqdm(submission_ids(args.root, test_dir))):
        image_path = test_dir / image_id
        if not image_path.exists():
            rows.append(
                {
                    "image_id": image_id,
                    "pa_deg": np.nan,
                    "fl_mm": np.nan,
                    "mt_mm": np.nan,
                    "pixel_spacing_mm": np.nan,
                    "is_dynamic": False,
                    "geometry_valid": False,
                }
            )
            continue
        image = read_gray(image_path)
        pixel_spacing_mm = infer_pixel_spacing_mm(image, image_path=image_path)
        is_dynamic = is_dynamic_ultrasound(image, image_path=image_path)
        apo_mask = predict_mask(apo_model, image_path, apo_size, apo_threshold, device, args.tta)
        fasc_mask = predict_mask(
            fasc_model, image_path, fasc_size, fasc_threshold, device, args.tta
        )
        architecture = architecture_from_masks(
            image,
            apo_mask,
            fasc_mask,
            pixel_spacing_mm=pixel_spacing_mm,
            min_pennation_deg=args.min_pennation,
        )
        if args.debug_dir is not None and (
            index % max(1, args.debug_every) == 0 or architecture is None
        ):
            save_debug_image(image, apo_mask, fasc_mask, args.debug_dir / f"{Path(image_id).stem}.jpg")
        rows.append(
            {
                "image_id": image_id,
                "pa_deg": np.nan if architecture is None else architecture.pa_deg,
                "fl_mm": np.nan if architecture is None else architecture.fl_mm,
                "mt_mm": np.nan if architecture is None else architecture.mt_mm,
                "pixel_spacing_mm": pixel_spacing_mm,
                "is_dynamic": is_dynamic,
                "geometry_valid": architecture is not None,
            }
        )

    diagnostics = pd.DataFrame(rows)
    predictions = finalize_predictions(diagnostics, calibrated=args.benchmark_calibration)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions[["image_id", "pa_deg", "fl_mm", "mt_mm"]].to_csv(
        args.out, index=False, lineterminator="\r\n"
    )
    if args.diagnostics_out is not None:
        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(args.diagnostics_out, index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
