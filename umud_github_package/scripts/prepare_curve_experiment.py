"""Prepare paired GPU pilots for expert curved-track supervision; never launches compute.

Both arms warm-start from the same v1 fold, disable ensemble pseudo-labels, and
use v2's existing domain/crop augmentation. The intervention adds VL track
supervision from other subjects. Generated kernels use the audited v2 engine.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rewrite_constants(source: str, values: dict) -> str:
    for name, value in values.items():
        pattern = re.compile(rf"^{re.escape(name)} = [^#\n]*?(\s*#.*)?$", re.M)
        if len(pattern.findall(source)) != 1:
            raise ValueError(f"expected one top-level assignment to {name}")
        source = pattern.sub(lambda m: f"{name} = {value!r}" + (m.group(1) or ""), source, count=1)
    compile(source, "<generated train.py>", "exec")
    return source


def prepare(out: Path, fold: int = 0, epochs: int = 4):
    template = ROOT / "kaggle/fasc_orient_v2/train.py"
    code = template.read_text()
    meta = json.loads((template.parent / "kernel-metadata.json").read_text())
    common = dict(MODE="full", TRAIN_FOLDS=[fold], MAX_EPOCHS=epochs, MIN_EPOCHS=epochs,
                  PSEUDO=False, WARM_START=True, REQUIRE_WARM_START=True, REQUIRE_COMPLETE_VL=True,
                  VL_VALIDATION_BY_FOLD=True, STYLE_DATASETS=("test",),
                  WARM_MODEL="fasc_orient_v1", MULTISCALE=False, DOMAIN_AUG=True,
                  TRAIN_BUDGET_S=3600.0, HARD_TRAIN_END_S=4500.0)
    records = []
    for arm, supervised in (("control", False), ("curves", True)):
        target = out / arm
        target.mkdir(parents=True, exist_ok=True)
        settings = dict(common, VL_SUP=supervised, MODEL_NAME=f"fasc_curve_v3_{arm}")
        generated = rewrite_constants(code, settings)
        path = target / "train.py"
        if path.exists() and path.read_text() != generated:
            raise FileExistsError(f"use a fresh output directory; refusing to replace {path}")
        path.write_text(generated)
        kernel = dict(meta, id=f"shivateja005/umud-curve-v3-{arm}-pilot",
                      title=f"UMUD Curve V3 {arm.title()} Pilot")
        (target / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2)+"\n")
        tree = ast.parse(generated)
        checks = {}
        for n in tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                if n.targets[0].id in settings:
                    checks[n.targets[0].id] = ast.literal_eval(n.value)
        if checks != settings:
            raise RuntimeError("generated configuration does not match the experiment")
        records.append(dict(arm=arm, path=str(target), settings=settings,
                            sha256=hashlib.sha256(generated.encode()).hexdigest()))
    manifest = dict(
        status="prepared; not GPU-smoke-tested or trained",
        question="Does genuine curved-track supervision improve held-out geometry beyond the same augmentation without it?",
        template_sha256=hashlib.sha256(code.encode()).hexdigest(), arms=records,
        differences=["VL_SUP and MODEL_NAME only; curves adds supervised VL samples, so training steps per epoch increase."],
        validation=["Fold-k GT labels are never training targets for that student.",
                    "Pseudo-labels from the 5-fold teacher ensemble are disabled in both arms.",
                    "VL subjects held out of the curves arm are excluded from its training loader entirely.",
                    "Compare both arms on the SAME held-out VL subjects and the SAME GT fold.",
                    "Both arms use the SAME test-only unlabeled style bank; held-out VL images are not used for augmentation statistics.",
                    "Every embedded VL expert record must match a raw image; requested subject folds require complete train/validation coverage.",
                    "A single fold is a pilot, not a reliable estimate of the final challenge score.",
                    "If promoted, regenerate five-fold runs in a new directory and require all-fold verification."],
        gate_to_expand="GPU smoke passes; curves beats control on held-out VL track and deep-strip angle MAE, with no material GT-fold regression. Confirm on a second fold before all five.",
        remote_requirements=["A CUDA GPU; recommended A100 80GB for headroom (memory/runtime not yet measured for this pilot).",
                             "v1 checkpoint fold{k}.pt, metrics/folds.csv, metrics/fold{k}.json and fasc_orient_v1_config.json in the same source tree.",
                             "Competition data under UMUD_INPUT_ROOT; public OSF downloads require internet.",
                             "Separate UMUD_OUTPUT_DIR and UMUD_TEMP_DIR for each arm.",
                             "Training limits do not stop cloud billing. Use a separate pod lifetime limit and retrieve outputs before teardown."],
        local_verification="Python compilation, config assertions, fold-isolation tests, and numpy target checks only; no local neural inference.",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"prepared": [r["path"] for r in records], "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/gpu_experiments/curve_v3_pilot")
    ap.add_argument("--fold", type=int, default=0, choices=range(5))
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error("--epochs must be positive")
    prepare(args.out, args.fold, args.epochs)
