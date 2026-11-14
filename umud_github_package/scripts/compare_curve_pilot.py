"""Compare completed paired pilots on identical GT/VL cohorts; never estimate a hidden challenge score."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

GT_FIELDS = {"gt_angle_mae_deg": "ang_mae", "gt_dice50": "dice50", "gt_median_error_deg": "med_err"}
VL_FIELDS = {"vl_track_mae_deg": "vl_track_err", "vl_strip_mae_deg": "vl_strip_err",
             "vl_pa_mae_deg": "vl_pa_mae", "vl_pa_bias_deg": "vl_pa_bias"}


def read_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def finite(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing/non-numeric {label}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def history_rows(path: Path) -> dict[int, dict]:
    with path.open(newline="") as fh:
        records = list(csv.DictReader(fh))
    rows = {}
    for row in records:
        epoch = int(row["epoch"])
        if epoch in rows:
            raise ValueError(f"duplicate epoch {epoch}: {path}")
        rows[epoch] = row
    if 0 not in rows:
        raise ValueError(f"missing epoch-0 baseline: {path}")
    return rows


def heldout_vl(root: Path, fold: int) -> dict[str, tuple]:
    files = list((root / "maskbank").glob("*_viewports.csv"))
    if len(files) != 1:
        raise ValueError(f"need exactly one completed maskbank viewport manifest under {root}; found {len(files)}")
    out = {}
    with files[0].open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] != "vl" or row.get("vl_fold", "") == "" or int(row["vl_fold"]) != fold:
                continue
            key = row["image_id"]
            if key in out:
                raise ValueError(f"duplicate held-out VL image ID: {key}")
            out[key] = tuple(row[k] for k in ("layout", "x1", "y1", "x2", "y2"))
    if not out:
        raise ValueError(f"empty held-out VL cohort: {files[0]}")
    # Only compare images whose three prediction channels were actually written.
    with (root / "maskbank" / "manifest.csv").open(newline="") as fh:
        written = {row["image_id"] for row in csv.DictReader(fh) if row["dataset"] == "vl"}
    if not set(out) <= written:
        raise ValueError(f"held-out VL predictions missing from maskbank manifest under {root}")
    seg_archives = list((root / "maskbank").glob("vl__*_seg.npz"))
    if len(seg_archives) != 1:
        raise ValueError(f"need exactly one VL segmentation archive under {root}")
    prefix = seg_archives[0].name.removesuffix("_seg.npz")
    for channel in ("seg", "cos2", "sin2"):
        archive = root / "maskbank" / f"{prefix}_{channel}.npz"
        with zipfile.ZipFile(archive) as zf:
            keys = {name.removesuffix(".npy") for name in zf.namelist() if name.endswith(".npy")}
        if not set(out) <= keys:
            raise ValueError(f"held-out VL predictions missing from {archive}")
    return out


def heldout_gt(root: Path, fold: int) -> set[str]:
    with (root / "metrics" / "folds.csv").open(newline="") as fh:
        out = {row["stem"] for row in csv.DictReader(fh)
               if row["stem"] == row["representative"] and int(row["fold"]) == fold}
    if not out:
        raise ValueError(f"empty GT validation fold {fold}: {root}")
    return out


def read_arm(name: str, root: Path, fold: int) -> dict:
    metrics = read_json(root / "metrics" / f"fold{fold}.json")
    if metrics.get("fold") != fold:
        raise ValueError(f"{name}: wrong fold in fold metrics")
    if metrics.get("reload_ok") is not True or metrics.get("warm_reproduces") is not True:
        raise ValueError(f"{name}: checkpoint reload or warm-reference reproduction did not pass")
    if metrics.get("warm", {}).get("warm") is not True:
        raise ValueError(f"{name}: missing verified warm initialization")
    if metrics.get("bn_recalibrated"):
        raise ValueError(f"{name}: saved checkpoint was recalibrated; rerun VL evaluation before comparing saved-checkpoint metrics")
    history = history_rows(root / "metrics" / f"history_fold{fold}.csv")
    selected_epoch = int(metrics["best_epoch"])
    if selected_epoch not in history:
        raise ValueError(f"{name}: selected epoch {selected_epoch} is absent from history")
    cohort_vl = heldout_vl(root, fold)
    cohort_gt = heldout_gt(root, fold)
    if int(metrics["n_val"]) != len(cohort_gt) or int(metrics["n_vl_val"]) != len(cohort_vl):
        raise ValueError(f"{name}: fold metric counts disagree with image manifests")
    coverage_file = root / "metrics" / "vl_coverage.json"
    if coverage_file.exists():
        coverage = read_json(coverage_file)
        if coverage["n_expected"] != coverage["n_matched"]:
            raise ValueError(f"{name}: incomplete embedded VL record coverage")
        if coverage["folds"][str(fold)]["n_val_items"] != len(cohort_vl):
            raise ValueError(f"{name}: VL preflight disagrees with held-out image manifest")
    output_rows = []
    for stage, epoch, gt_key, vl_key in (("epoch0", 0, "epoch0", "epoch0_vl"),
                                        ("selected", selected_epoch, "best", "best_vl")):
        gt, vl, recorded = metrics[gt_key], metrics[vl_key], history[epoch]
        if int(gt["n_val"]) != len(cohort_gt) or int(gt["n_ang"]) != len(cohort_gt):
            raise ValueError(f"{name}/{stage}: incomplete GT angle cohort")
        if int(vl["vl_n"]) != len(cohort_vl):
            raise ValueError(f"{name}/{stage}: PA proxy rejected some held-out images; align accepted IDs before comparing")
        row = dict(arm=name, stage=stage, epoch=epoch, n_gt=len(cohort_gt), n_vl=len(cohort_vl),
                   vl_pa_count=int(vl["vl_n"]))
        for fields, source in ((GT_FIELDS, gt), (VL_FIELDS, vl)):
            for column, field in fields.items():
                value = finite(source[field], f"{name}/{stage}/{field}")
                history_value = finite(recorded[field], f"{name}/history/{field}")
                if not math.isclose(value, history_value, rel_tol=1e-8, abs_tol=1e-7):
                    raise ValueError(f"{name}/{stage}: JSON and history disagree for {field}")
                row[column] = value
        output_rows.append(row)
    configs = list(root.glob("*_config.json"))
    config = read_json(configs[0]) if len(configs) == 1 else {}
    log_path = root / "logs" / "gpu0.log"
    logged_memory = re.findall(r"\bmem=([0-9.]+)G", log_path.read_text()) if log_path.exists() else []
    return dict(root=root, metrics=metrics, history=history, vl=cohort_vl, gt=cohort_gt, rows=output_rows,
                elapsed_minutes=config.get("elapsed_min"),
                logged_peak_vram_gib=max(map(float, logged_memory)) if logged_memory else None)


def compare(args) -> None:
    arms = [read_arm("control", args.control, args.fold), read_arm("curves", args.curves, args.fold)]
    control, curves = arms
    if control["gt"] != curves["gt"] or control["vl"] != curves["vl"]:
        raise ValueError("arms do not share exactly the same held-out GT IDs and VL IDs/crop boxes")
    for key in ("n_val", "n_vl_val"):
        if control["metrics"][key] != curves["metrics"][key]:
            raise ValueError(f"arms disagree on {key}")
    baseline_deltas = {}
    for field in (*GT_FIELDS, *VL_FIELDS):
        baseline_deltas[field] = curves["rows"][0][field] - control["rows"][0][field]
        atol = args.baseline_angle_atol if field.endswith("_deg") else 1e-5
        if not math.isclose(control["rows"][0][field], curves["rows"][0][field], rel_tol=1e-6, abs_tol=atol):
            raise ValueError(f"warm baselines differ for {field}; investigate initialization/cohort before comparing")
    split = read_json(args.curves / "metrics" / f"fold{args.fold}_vl_split.json")
    train_subjects, val_subjects = set(split["train_subjects"]), set(split["val_subjects"])
    if not train_subjects or not val_subjects or train_subjects & val_subjects:
        raise ValueError("curves subject split is empty or leaks held-out subjects")
    if int(split["n_val_items"]) != len(curves["vl"]):
        raise ValueError("curves subject split disagrees with held-out image count")
    selected_control, selected_curves = control["rows"][1], curves["rows"][1]
    delta = {field: selected_curves[field] - selected_control[field] for field in (*GT_FIELDS, *VL_FIELDS)}
    cohort_digest = hashlib.sha256("\n".join(sorted(control["vl"])).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path, report_path = args.out / "comparison.csv", args.out / "PILOT_REPORT.md"
    if csv_path.exists() or report_path.exists():
        raise FileExistsError("comparison outputs already exist; use a fresh --out directory")
    rows = [row for arm in arms for row in arm["rows"]]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# Curved-track paired pilot — fold {args.fold}", "",
             f"Verified identical cohorts: {len(control['gt'])} GT images; {len(control['vl'])} VL images "
             f"from {len(val_subjects)} held-out subjects. No train/validation subject overlap.",
             f"VL ID SHA-256: {cohort_digest}.", "",
             f"Epoch-0 angle comparison absolute tolerance: {args.baseline_angle_atol:g}°; "
             "Dice tolerance: 1e-5; relative tolerance: 1e-6.",
             f"Tolerance audit note: {args.baseline_note or 'default strict comparison'}.",
             "Epoch-0 curves minus control: " + ", ".join(f"{k}={v:+.9g}" for k, v in baseline_deltas.items()) + ".", "",
             "| Arm | Stage | Epoch | GT angle MAE (°) | GT Dice | VL track MAE (°) | VL strip MAE (°) | PA proxy MAE (°) | PA bias (°) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['stage']} | {row['epoch']} | {row['gt_angle_mae_deg']:.4f} | "
                     f"{row['gt_dice50']:.4f} | {row['vl_track_mae_deg']:.4f} | {row['vl_strip_mae_deg']:.4f} | "
                     f"{row['vl_pa_mae_deg']:.4f} | {row['vl_pa_bias_deg']:+.4f} |")
    lines.extend(["", "Selected curves minus selected control: "
                  f"GT angle {delta['gt_angle_mae_deg']:+.4f}°, Dice {delta['gt_dice50']:+.4f}; "
                  f"VL track {delta['vl_track_mae_deg']:+.4f}°, strip {delta['vl_strip_mae_deg']:+.4f}°, "
                  f"PA proxy MAE {delta['vl_pa_mae_deg']:+.4f}°.", ""])
    for arm in arms:
        m = arm["metrics"]
        elapsed = f"{float(arm['elapsed_minutes']):.2f}" if arm["elapsed_minutes"] is not None else "unavailable"
        peak = f"{arm['logged_peak_vram_gib']:.1f}" if arm["logged_peak_vram_gib"] is not None else "unavailable"
        lines.append(f"- {arm['rows'][0]['arm']}: selected epoch {m['best_epoch']}; "
                     f"{max(arm['history'])} recorded training epochs; {m['n_train']} GT samples + "
                     f"{m['n_pseudo_epoch']} extra samples/epoch; "
                     f"script wall time {elapsed} min; "
                     f"highest logged allocated VRAM {peak} GiB (rounded training log).")
    lines.extend(["", "Selection uses the existing GT-fold rule; epoch 0 remaining best is a valid negative result. "
                  "Curves has more optimizer steps per epoch. VL is subject-held-out within a supervised scanner domain. "
                  "These diagnostics do not measure the competition score, complete fascicle length, or muscle thickness. "
                  "Confirm any gain on another fold and downstream PA/FL/MT before adoption."])
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"comparison_csv": str(csv_path), "report": str(report_path),
                      "same_gt_images": len(control["gt"]), "same_vl_images": len(control["vl"]),
                      "selected_curves_minus_control": delta,
                      "epoch0_curves_minus_control": baseline_deltas,
                      "baseline_angle_atol_deg": args.baseline_angle_atol,
                      "baseline_tolerance_note": args.baseline_note}, indent=2, allow_nan=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", type=Path, required=True, help="control output root")
    ap.add_argument("--curves", type=Path, required=True, help="curves output root")
    ap.add_argument("--out", type=Path, required=True, help="directory for fresh CSV and Markdown report")
    ap.add_argument("--fold", type=int, default=0, choices=range(5))
    ap.add_argument("--baseline-angle-atol", type=float, default=1e-5,
                    help="absolute epoch-0 angle tolerance; increase only after auditing numerical drift")
    ap.add_argument("--baseline-note", default="",
                    help="required audit explanation if the default baseline angle tolerance is increased")
    args = ap.parse_args()
    if not math.isfinite(args.baseline_angle_atol) or args.baseline_angle_atol <= 0:
        ap.error("--baseline-angle-atol must be positive and finite")
    if args.baseline_angle_atol > 1e-5 and not args.baseline_note.strip():
        ap.error("an increased baseline angle tolerance requires --baseline-note with the numerical audit evidence")
    try:
        compare(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Pilot comparison FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
