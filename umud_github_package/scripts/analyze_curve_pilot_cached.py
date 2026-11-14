"""Audit saved GPU evaluations and summarize paired subjects without loading a model.

The numerical tolerance is an explicit post-run engineering decision, not a
quality gate. Original strict failures are inputs and remain untouched.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from evaluate_curve_pilot_vl import (FIELDS, aggregate_check, paired_bootstrap,
                                   read_json, sha256, write_csv, write_json)


def analyze(args):
    if not args.audit_note.strip():
        raise ValueError("an explicit numerical audit note is required")
    if not 0 < args.tolerance <= 0.001:
        raise ValueError("aggregate tolerance must be positive and at most 0.001 degrees")
    if args.out.exists():
        raise FileExistsError("use a fresh analysis output directory")
    audit = read_json(args.source_audit)
    strict = read_json(args.diagnostics / "VL_EVALUATION_REPORT.json")
    warm = read_json(args.diagnostics / "warm_numerical_diagnostics.json")
    arms = {name: read_json(args.diagnostics / f"{name}_evaluation.json")
            for name in ("control", "curves")}
    repeats = [read_json(p) for p in sorted(args.diagnostics.glob("warm_repeat_*.json"))]
    if len(repeats) < 3 or warm["repetitions"] != len(repeats):
        raise ValueError("need at least three completed fresh-process warm evaluations")
    if strict["status"] == "failed" and not strict.get("error", "").startswith(
            "RuntimeError: selected-checkpoint aggregates exceed strict reproduction tolerance"):
        raise ValueError("strict evaluation has an unresolved failure beyond numerical reproduction")
    if audit["same_gt_images"] != 409 or audit["same_vl_images_and_crop_boxes"] != 50:
        raise ValueError("unexpected original pilot cohort")
    if warm["checkpoint_sha256"] != audit["warm_checkpoint_sha256"]:
        raise ValueError("warm diagnostic checkpoint differs from original source")
    for result in [*arms.values(), *repeats]:
        for key in ("cohort", "core_functions_sha256", "evaluation_parameters_sha256"):
            if result[key] != arms["control"][key]:
                raise ValueError(f"ordered inputs, target hashes or evaluation code differ: {key}")
        for key in ("gpu", "torch", "cuda", "smp", "timm", "cudnn_benchmark", "cudnn_deterministic",
                    "cudnn_allow_tf32", "matmul_allow_tf32", "batch", "size", "tta", "autocast_dtype"):
            if result["runtime"][key] != arms["control"]["runtime"][key]:
                raise ValueError(f"numerical runtime settings differ: {key}")
        if not result["original_evaluator_check"]["passed"]:
            raise ValueError("per-image extraction disagrees with original evaluator")
        if len(result["cohort"]) != 50 or result["metrics"]["vl_n"] != 50:
            raise ValueError("incomplete held-out PA cohort")
        if result["metrics"]["track_n"] != 50 or result["metrics"]["strip_n"] != 50:
            raise ValueError("incomplete held-out track/strip cohort")
    for name, result in arms.items():
        if result["engine_sha256"] != audit["engines"][name]:
            raise ValueError(f"{name} evaluation used different engine bytes")
        if result["checkpoint_sha256"] != sha256(args.source_audit.parent / name / "checkpoints/fold0.pt"):
            raise ValueError(f"{name} evaluation differs from backed-up checkpoint bytes")
    for repeat in repeats:
        if repeat["checkpoint_sha256"] != warm["checkpoint_sha256"]:
            raise ValueError("a warm repeat used different checkpoint bytes")
    selected_checks = {name: aggregate_check(result["metrics"], result["saved_selected_metrics"],
                                              args.tolerance) for name, result in arms.items()}
    if not all(check["passed"] for check in selected_checks.values()):
        raise ValueError(f"selected aggregates exceed the audited tolerance: {selected_checks}")
    if max(row["combined_range"] for row in warm["metrics"].values()) > args.tolerance:
        raise ValueError("observed warm numerical range exceeds the audited tolerance")
    warm_image_ranges = {
        field: max(abs(a["rows"][i][field] - b["rows"][i][field])
                   for a, b in itertools.combinations(repeats, 2) for i in range(50))
        for field in FIELDS
    }
    bootstrap, subjects, leave_out = paired_bootstrap(arms["control"], arms["curves"], 10000, 20261022)
    report = dict(status="audited analysis complete", aggregate_tolerance_deg=args.tolerance,
                  tolerance_decision="post-run numerical engineering audit; not a model-quality criterion",
                  audit_note=args.audit_note, original_strict_status=strict["status"],
                  selected_aggregate_checks=selected_checks, warm_numerical_diagnostics=warm,
                  warm_max_per_image_range_deg=warm_image_ranges, bootstrap=bootstrap,
                  source_audit=audit, input_sha256={p.name: sha256(p) for p in (
                      args.source_audit, args.diagnostics / "VL_EVALUATION_REPORT.json",
                      args.diagnostics / "control_evaluation.json", args.diagnostics / "curves_evaluation.json",
                      args.diagnostics / "warm_numerical_diagnostics.json")})
    args.out.mkdir(parents=True)
    write_json(args.out / "AUDITED_ANALYSIS.json", report)
    write_json(args.out / "paired_subject_bootstrap.json", bootstrap)
    write_csv(args.out / "paired_subjects.csv", subjects)
    write_csv(args.out / "leave_one_subject_out.csv", leave_out)
    lines = ["# Audited exploratory paired subject analysis", "",
             f"50 identical images from 25 held-out subjects. All images have accepted track, strip and PA values. "
             f"The original {strict['selected_tolerance_deg']:g}° strict reproduction check failed and is preserved. "
             f"The audited aggregate tolerance is {args.tolerance:g}°.", "", args.audit_note, "",
             "| Metric | Curves − control (°) | Descriptive subject-cluster interval (°) | Subjects with negative delta |",
             "|---|---:|---:|---:|"]
    for field, row in bootstrap["metrics"].items():
        lo, hi = row["descriptive_cluster_interval_2_5_97_5"]
        lines.append(f"| {field} | {row['curves_minus_control']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | "
                     f"{row['subjects_with_negative_delta']}/25 |")
    lines += ["", "Intervals use 10,000 paired subject resamples, seed 20261022, preserving image weighting. "
              "Equal-subject and leave-one-subject-out estimates are saved separately. This is exploratory "
              "descriptive analysis of one fold; it establishes neither statistical significance nor a Kaggle score. "
              "Signed bias deltas do not by themselves indicate improvement."]
    (args.out / "SUBJECT_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(args.out / "SUBJECT_ANALYSIS.md"),
                      "metrics": bootstrap["metrics"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostics", type=Path, required=True)
    ap.add_argument("--source-audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, required=True)
    ap.add_argument("--audit-note", required=True)
    analyze(ap.parse_args())
