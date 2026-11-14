"""Rebuild an evidence inventory, benchmark-overlap audit and geometry stress summary.

Candidate distance is a distance between predictions, never an estimated score.
The recorded leaderboard values were read from the account on 2026-10-22.
No network requests, model inference, fitting, or submissions are performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umud.metric import COMPONENTS, score_arrays
from umud.paths import RESEARCH_ROOT

KNOWN_SCORES = {
    "submission_unetpp384_t095_calibrated.csv": 0.92533,
    "sub_v1_geom_clipmed.csv": 0.46718,
    "C1_ours_v2e_fusion_ensemble.csv": 0.45436,
    "C2_ours_v2e_fusion_ensemble_shrink1.0.csv": 0.38974,
    "C11_fullshrink_cellmedian.csv": 0.51239,
    "C12_hyb_med_ens_pp_muscle5_emclean_off.csv": 0.37670,
    "C13_C12_telemedT1.csv": 0.41184,
    "C15_C14_telemedT1.csv": 0.41182,
    "C17_C12_nooff.csv": 0.38091,
    "C18_C12_telemedFLraw.csv": 0.33526,
    "C19_C12_telemedFLcoupling.csv": 0.41013,
}


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    base_path = ROOT / "artifacts/candidates_v6/C18_C12_telemedFLraw.csv"
    base = pd.read_csv(base_path).set_index("image_id")
    records = []
    for p in sorted((ROOT / "artifacts").glob("candidates*/**/*.csv")):
        if not (p.name.startswith("C") or p.parent.name == "submit_2026-10-23"):
            continue
        d = pd.read_csv(p)
        if not {"image_id", *COMPONENTS} <= set(d):
            continue
        d = d.set_index("image_id")
        valid = (d.index.is_unique and set(d.index) == set(base.index)
                 and np.isfinite(d[list(COMPONENTS)].to_numpy()).all())
        distance = np.nan
        if valid:
            distance = score_arrays({c: d.loc[base.index, c].to_numpy() for c in COMPONENTS},
                                    {c: base[c].to_numpy() for c in COMPONENTS})
        records.append(dict(path=str(p.relative_to(ROOT)), rows=len(d), ids_and_finiteness_valid=bool(valid),
                            distance_to_C18=distance, known_public_score=KNOWN_SCORES.get(p.name),
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(records).to_csv(out / "candidate_inventory.csv", index=False)
    pd.DataFrame([dict(file=k, public_score=v, source="Kaggle account CLI, verified 2026-10-22")
                  for k, v in KNOWN_SCORES.items()]).to_csv(out / "scored_history.csv", index=False)

    # Conservative near-overlap screen. Correlation is evidence of risk, not proof
    # that a frame is exactly identical. Preserve historical CLEAN for comparison.
    import build_candidates as bc
    import build_candidates_v4 as b4
    evidence = []
    for name, column in (("bench_matches_central.csv", "center_corr"),
                         ("bench_matches_fine.csv", "full_corr")):
        m = pd.read_csv(RESEARCH_ROOT / "data-labels-geometry/out" / name)
        m = m[m[column] >= 0.95]
        evidence.extend(dict(image_id=r.bench, kind=r.kind, train=r.train,
                             correlation=float(getattr(r, column)), evidence=name)
                        for r in m.itertuples())
    pd.DataFrame(evidence).to_csv(out / "overlap_evidence.csv", index=False)
    at_risk = {r["image_id"] for r in evidence}
    strict = [i for i in bc.CLEAN if i not in at_risk]
    pd.DataFrame([dict(image_id=i, historical_clean10=i in bc.CLEAN,
                       overlap_risk=i in at_risk, conservative_subset=i in strict)
                  for i in bc.static_labels().index]).to_csv(out / "static_split_audit.csv", index=False)
    # This is the cached baseline without the fitted rater offset. It is a
    # sensitivity analysis on a repeatedly used benchmark, not a fresh holdout.
    pred = b4.bench_preds5(dict(b4.c12_base(), offset=None))["static"]
    truths = bc.static_gts()
    rows = []
    for name, ids in (("historical_clean10", bc.CLEAN), ("conservative_subset", strict)):
        for truth in ("mean7", *(f"R7+R{k}" for k in range(1, 7))):
            rows.append(dict(subset=name, truth=truth,
                             recipe="C12/C18 static recipe, no rater offset; LODO constants",
                             **bc.s_metric(pred.loc[ids], truths[truth])))
    pd.DataFrame(rows).to_csv(out / "static_sensitivity_nooffset.csv", index=False)

    stress_path = ROOT / "artifacts/fov_stress/per_image.csv"
    if stress_path.exists():
        stress = pd.read_csv(stress_path)
        agg = []
        for (source, fraction), d in stress.groupby(["source", "fraction"]):
            q = dict(source=source, fraction=fraction, n=len(d))
            for c in ("pa_deg", "fl_px", "mt_px"):
                q[f"n_valid_{c}"] = int(d[f"error_{c}"].notna().sum())
                q[f"mae_{c}"] = float(d[f"error_{c}"].abs().mean())
                if c != "pa_deg":
                    q[f"mean_abs_relative_{c}"] = float((d[f"error_{c}"] / d[f"ref_{c}"].replace(0, np.nan)).abs().mean())
            agg.append(q)
        pd.DataFrame(agg).to_csv(out / "fov_summary.csv", index=False)

    manifest = json.loads((RESEARCH_ROOT / "archive_manifest.json").read_text())
    bad = [r["path"] for r in manifest["files"]
           if hashlib.sha256((RESEARCH_ROOT / r["path"]).read_bytes()).hexdigest() != r["sha256"]]
    if bad:
        raise RuntimeError(f"historical archive checksum mismatch: {bad}")
    report = dict(baseline_public_score=0.33526, target=0.20,
                  gap=0.33526-0.20, relative_reduction_required=(0.33526-0.20)/0.33526,
                  baseline_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                  candidate_count=len(records), archive_files_verified=len(manifest["files"]),
                  historical_clean10=bc.CLEAN, conservative_subset=strict,
                  additional_risk_in_clean10=sorted(set(bc.CLEAN) & at_risk),
                  limitations=["No hidden test labels; per-scanner leaderboard error decompositions are unidentifiable without assumptions.",
                               "No fresh validation set is created by excluding a near-overlap from a repeatedly used benchmark.",
                               "Candidate distances are NOT scores or predicted gains.",
                               "No new model was trained; no new file was submitted."])
    (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/progress_audit")
    main(ap.parse_args().out)
