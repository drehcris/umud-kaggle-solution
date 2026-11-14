"""Remote CUDA-only VL reproduction and exploratory paired subject diagnostics; no training."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

FIELDS = ("vl_track_err", "vl_strip_err", "vl_pa_mae", "vl_pa_bias")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def remote_guard(paths):
    workspace = Path("/workspace")
    if not workspace.is_dir():
        raise RuntimeError("Remote-only tool: /workspace must exist on the GPU pod. Do not run on the laptop.")
    workspace = workspace.resolve()
    for path in paths:
        if not Path(path).resolve().is_relative_to(workspace):
            raise RuntimeError(f"all inputs/outputs must be inside remote /workspace: {path}")


def cuda_runtime():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled for this remote-only evaluation")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True  # exactly the pilot worker's forward setting
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    allocator_probe = torch.empty(1, device=dev)
    del allocator_probe
    torch.cuda.reset_peak_memory_stats(dev)
    return torch, dev


def load_engine(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import prepared engine: {path}")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    if engine.BATCH != 8 or engine.SIZE != 512 or engine.PSEUDO:
        raise RuntimeError("requires prepared BATCH=8, SIZE=512, PSEUDO=False pilot engine")
    if not (engine.VL_SUP or engine.VL_VALIDATION_BY_FOLD):
        raise RuntimeError("engine does not use subject-held-out VL diagnostics")
    import segmentation_models_pytorch as smp
    if smp.__version__ != engine.SMP_VERSION:
        raise RuntimeError(f"requires segmentation-models-pytorch=={engine.SMP_VERSION}; found {smp.__version__}")
    return engine


def restore_entries(engine, archives, work):
    """Reuse gather_extra's exact extraction/order/key rules, with network access replaced by local copies."""
    required = {"vl_young.zip", "vl_old.zip"}
    if {filename for _, filename in engine.EXTRA_SOURCES["vl"]} != required:
        raise RuntimeError("unexpected VL archive names in prepared engine")
    provenance = {}
    for name in sorted(required):
        path = archives / name
        if not path.is_file() or not zipfile.is_zipfile(path):
            raise RuntimeError(f"missing or invalid persistent VL ZIP: {path}")
        provenance[name] = dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)
    engine.TMP = work

    def cached_download(_url, dest):
        if dest.name not in required:
            raise RuntimeError(f"unexpected download request (network disabled): {dest.name}")
        shutil.copyfile(archives / dest.name, dest)
        return True

    engine.download = cached_download
    entries = engine.gather_extra("vl")
    if not entries:
        raise RuntimeError("cached archives yielded no raw VL images")
    return entries, provenance


def prepare_vl(engine, entries, fold, torch):
    import numpy as np

    tracks = engine.decode_vl_tracks()
    unlabeled, _ = engine.prep_unlabeled({"vl": entries}, engine.Teacher(None, engine.PSEUDO_MODEL), tracks)
    coverage = engine.validate_complete_vl(unlabeled, tracks, [fold], engine.VL_SUP,
                                          engine.VL_VALIDATION_BY_FOLD)
    items = [u for u in unlabeled if u.get("tracks") is not None and u.get("vfold") == fold]
    if not items:
        raise RuntimeError(f"empty held-out VL fold {fold}")
    source_paths = {key: path for key, path, _ in entries}
    samples, cohort = [], []
    for u in items:  # preserve prep_unlabeled order and build_cache's BATCH=8 grouping
        sample = engine.make_sample(u["crop"], None, tracks=u["tracks"])
        sample["pa_deg"] = np.float32(u["tracks"]["pa_deg"])
        target_hash = hashlib.sha256()
        for key in ("vcos2", "vsin2", "vskel", "vstrip", "deep_ang", "pa_deg"):
            target_hash.update(np.asarray(sample[key]).tobytes())
        cohort.append(dict(image_id=u["key"], source_tail=engine.tail3(u["src"]),
                           subject=u["tracks"]["subject"], fold=int(u["vfold"]), layout=u["layout"],
                           x1=int(u["box"][0]), y1=int(u["box"][1]),
                           x2=int(u["box"][2]), y2=int(u["box"][3]),
                           raw_sha256=sha256(source_paths[u["key"]]),
                           crop_sha256=hashlib.sha256(u["crop"].tobytes()).hexdigest(),
                           canvas_sha256=hashlib.sha256(sample["img"].tobytes()).hexdigest(),
                           target_sha256=target_hash.hexdigest()))
        samples.append(sample)
    keys = ("img", "vcos2", "vsin2", "vskel", "vstrip", "deep_ang", "pa_deg")
    cache = engine.build_cache(samples, keys, torch)
    return cache, cohort, coverage


def per_image_forward(engine, model, cache, cohort, torch, dev):
    """Same operations and acceptance masks as engine.evaluate_vl; preserve unrounded per-image scalars."""
    import numpy as np

    model.eval()
    rows, offset = [], 0
    with torch.no_grad():
        for b in cache:
            x = engine.to_input(b["img"].to(dev), torch)
            p, c, s = engine.forward_maps(model, x, torch, False)
            th_p = 0.5 * torch.atan2(s, c)
            th_t = 0.5 * torch.atan2(b["vsin2"].to(dev).float(), b["vcos2"].to(dev).float())
            d = torch.remainder(th_p - th_t, math.pi)
            d = torch.minimum(d, math.pi - d) * (180.0 / math.pi)
            sk = b["vskel"].to(dev) > 0
            strip = b["vstrip"].to(dev) > 0
            w = torch.hypot(c, s) * p
            for i in range(x.shape[0]):
                ss, sm = sk[i] & strip[i], strip[i]
                row = dict(cohort[offset], track_pixels=int(sk[i].sum()),
                           strip_track_pixels=int(ss.sum()), strip_pixels=int(sm.sum()),
                           pa_weight_sum=float(w[i][sm].sum()), pa_accepted=False,
                           vl_track_err=None, vl_strip_err=None, vl_pa_mae=None, vl_pa_bias=None,
                           pa_pred_deg=None, pa_gt_deg=float(b["pa_deg"][i]))
                if row["track_pixels"] >= 10:
                    row["vl_track_err"] = float(d[i][sk[i]].mean())
                if row["strip_track_pixels"] >= 5:
                    row["vl_strip_err"] = float(d[i][ss].mean())
                if row["strip_pixels"] >= 30 and row["pa_weight_sum"] > 1e-3:
                    wc = float((c[i][sm] * w[i][sm]).sum())
                    ws = float((s[i][sm] * w[i][sm]).sum())
                    thf = 0.5 * math.atan2(ws, wc)
                    rel = math.degrees(thf - float(b["deep_ang"][i]))
                    pa_f = abs((rel + 90.0) % 180.0 - 90.0)
                    error = pa_f - float(b["pa_deg"][i])
                    row.update(pa_accepted=True, pa_pred_deg=pa_f, vl_pa_mae=abs(error), vl_pa_bias=error)
                for key in (*FIELDS, "pa_weight_sum"):
                    if row[key] is not None and not math.isfinite(row[key]):
                        raise RuntimeError(f"non-finite {key} for {row['image_id']}")
                rows.append(row)
                offset += 1
    if offset != len(cohort):
        raise RuntimeError("per-image evaluation count disagrees with prepared cohort")
    aggregate = {}
    for field in FIELDS:
        values = [row[field] for row in rows if row[field] is not None]
        if not values:
            raise RuntimeError(f"no accepted per-image values for {field}")
        aggregate[field] = float(np.mean(values))
    aggregate["vl_n"] = sum(row["pa_accepted"] for row in rows)
    aggregate["track_n"] = sum(row["vl_track_err"] is not None for row in rows)
    aggregate["strip_n"] = sum(row["vl_strip_err"] is not None for row in rows)
    return rows, aggregate


def aggregate_check(got, reference, tolerance):
    errors = {field: float(got[field]) - float(reference[field]) for field in FIELDS}
    counts_match = int(got["vl_n"]) == int(reference["vl_n"])
    return dict(passed=counts_match and all(math.isfinite(v) and abs(v) <= tolerance for v in errors.values()),
                tolerance_deg=tolerance, differences=errors, pa_count_matches=counts_match)


def evaluate(engine_path, checkpoint, entries, fold, torch, dev, expected_epoch=None):
    import segmentation_models_pytorch as smp
    import timm

    started = time.perf_counter()
    engine = load_engine(engine_path, "vl_eval_" + hashlib.sha256(str(engine_path).encode()).hexdigest()[:12])
    cache, cohort, coverage = prepare_vl(engine, entries, fold, torch)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("fold", fold) != fold:
        raise RuntimeError(f"wrong checkpoint fold: {checkpoint}")
    if expected_epoch is not None and state.get("epoch") != expected_epoch:
        raise RuntimeError(f"checkpoint epoch disagrees with selected fold JSON: {checkpoint}")
    if state.get("bn_recalibrated"):
        raise RuntimeError("checkpoint was recalibrated; its saved VL metrics may be stale")
    model = engine.build_model(pretrained=False)
    bad = engine.load_state(model, state["state_dict"], torch)
    if bad:
        raise RuntimeError(f"non-finite checkpoint tensors; no repairs allowed: {bad[:10]}")
    model = model.to(dev).eval().to(memory_format=torch.channels_last)
    rows, aggregate = per_image_forward(engine, model, cache, cohort, torch, dev)
    original = engine.evaluate_vl(model, cache, torch, dev)
    original_check = aggregate_check(aggregate, original, 1e-5)
    if not original_check["passed"]:
        raise RuntimeError(f"per-image extraction disagrees with engine.evaluate_vl: {original_check}")
    torch.cuda.synchronize(dev)
    result = dict(metrics=aggregate, engine_aggregate=original, original_evaluator_check=original_check,
                  cohort=cohort, cohort_sha256=digest(cohort), coverage=coverage, rows=rows,
                  checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  checkpoint_epoch=state.get("epoch"), engine=str(engine_path), engine_sha256=sha256(engine_path),
                  core_functions_sha256=digest({name: inspect.getsource(getattr(engine, name)) for name in (
                      "build_model", "load_state", "forward_maps", "to_input", "evaluate_vl", "prep_unlabeled",
                      "make_sample", "track_targets", "viewport", "letterbox_matrix", "build_cache", "subject_folds")}),
                  evaluation_parameters_sha256=digest({name: getattr(engine, name) for name in (
                      "BATCH", "SIZE", "ENCODER", "VL_STRIP", "VL_ORI_RADIUS", "IMAGENET_MEAN", "IMAGENET_STD", "CHANNELS")}),
                  elapsed_seconds=time.perf_counter() - started,
                  runtime=dict(gpu=torch.cuda.get_device_name(dev), python=platform.python_version(),
                               torch=torch.__version__, cuda=torch.version.cuda, smp=smp.__version__, timm=timm.__version__,
                               cudnn_benchmark=torch.backends.cudnn.benchmark,
                               cudnn_deterministic=torch.backends.cudnn.deterministic,
                               cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                               matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                               batch=engine.BATCH, size=engine.SIZE, tta=False, autocast_dtype="float16",
                               peak_allocated_vram_gib=torch.cuda.max_memory_allocated(dev) / 2**30))
    del model, cache, state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(control, curves, repetitions, seed):
    import numpy as np

    if control["cohort"] != curves["cohort"]:
        raise RuntimeError("control/curves differ in ordered IDs, subjects, raw/crop/canvas/target hashes")
    subjects = sorted({row["subject"] for row in control["rows"]})
    if len(subjects) < 2:
        raise RuntimeError("subject bootstrap requires at least two subjects")
    by_subject = [{"subject": subject} for subject in subjects]
    leave_out = [{"excluded_subject": subject} for subject in subjects]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), size=repetitions)
    summary = dict(repetitions=repetitions, seed=seed, n_subjects=len(subjects),
                   n_images=len(control["rows"]), interpretation="exploratory descriptive subject-cluster resampling; no significance or hidden-score claim",
                   primary="image-weighted", secondary="equal subject-weighted", metrics={})
    for field in FIELDS:
        c_sums, t_sums, counts = [], [], []
        for i, subject in enumerate(subjects):
            indices = [j for j, row in enumerate(control["rows"]) if row["subject"] == subject]
            c_values, t_values = [], []
            for j in indices:
                c, t = control["rows"][j][field], curves["rows"][j][field]
                if (c is None) != (t is None):
                    raise RuntimeError(f"different accepted image cohorts for {field}: {control['rows'][j]['image_id']}")
                if c is not None:
                    c_values.append(c)
                    t_values.append(t)
            if not c_values:
                raise RuntimeError(f"no accepted {field} images for subject {subject}")
            c_sums.append(sum(c_values))
            t_sums.append(sum(t_values))
            counts.append(len(c_values))
            by_subject[i].update({field + "_control": float(np.mean(c_values)),
                                  field + "_curves": float(np.mean(t_values)),
                                  field + "_delta": float(np.mean(t_values) - np.mean(c_values))})
        c_sums, t_sums, counts = map(lambda x: np.asarray(x, dtype=np.float64), (c_sums, t_sums, counts))
        difference_sums = t_sums - c_sums
        subject_differences = difference_sums / counts
        image_samples = (weights @ difference_sums) / (weights @ counts)
        subject_samples = (weights @ subject_differences) / len(subjects)
        for i in range(len(subjects)):
            leave_out[i].update({field + "_image_weighted_delta": float(
                (difference_sums.sum() - difference_sums[i]) / (counts.sum() - counts[i])),
                                 field + "_subject_weighted_delta": float(
                (subject_differences.sum() - subject_differences[i]) / (len(subjects) - 1))})
        summary["metrics"][field] = dict(
            curves_minus_control=float(difference_sums.sum() / counts.sum()),
            relative_error_reduction=None if field == "vl_pa_bias" else float(1 - t_sums.sum() / c_sums.sum()),
            descriptive_cluster_interval_2_5_97_5=list(map(float, np.percentile(image_samples, [2.5, 97.5]))),
            equal_subject_delta=float(subject_differences.mean()),
            equal_subject_descriptive_interval_2_5_97_5=list(map(float, np.percentile(subject_samples, [2.5, 97.5]))),
            subjects_with_negative_delta=int((subject_differences < 0).sum()),
            subjects_with_positive_delta=int((subject_differences > 0).sum()),
            leave_one_subject_out_image_delta_range=[float(min(r[field + "_image_weighted_delta"] for r in leave_out)),
                                                   float(max(r[field + "_image_weighted_delta"] for r in leave_out))])
    return summary, by_subject, leave_out


def warm_worker(job_path):
    job = read_json(job_path)
    remote_guard([job_path, job["engine"], job["checkpoint"], job["entries"], job["output"]])
    entries = [(key, Path(path), source) for key, path, source in read_json(job["entries"])]
    remote_guard([path for _, path, _ in entries])
    torch, dev = cuda_runtime()
    result = evaluate(Path(job["engine"]), Path(job["checkpoint"]), entries, int(job["fold"]), torch, dev)
    write_json(job["output"], result)


def run(args, report):
    torch, dev = cuda_runtime()
    tmp_parent = Path("/workspace/tmp")
    tmp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="umud_vl_eval_", dir=tmp_parent) as temporary:
        work = Path(temporary)
        restore_engine = load_engine(args.control_engine, "vl_restore_engine")
        entries, archives = restore_entries(restore_engine, args.archives, work)
        report["archives"] = archives
        entries_file = work / "entries.json"
        write_json(entries_file, [(key, str(path), source) for key, path, source in entries])
        results = {}
        for name, engine_path, root in (("control", args.control_engine, args.control),
                                         ("curves", args.curves_engine, args.curves)):
            reference = read_json(root / "metrics" / f"fold{args.fold}.json")
            if reference.get("reload_ok") is not True or reference.get("warm_reproduces") is not True:
                raise RuntimeError(f"{name}: reload/warm reproduction did not pass")
            if reference.get("bn_recalibrated"):
                raise RuntimeError(f"{name}: recalibrated checkpoint has stale saved VL metrics")
            checkpoint = root / "checkpoints" / f"fold{args.fold}.pt"
            result = evaluate(engine_path, checkpoint, entries, args.fold, torch, dev, int(reference["best_epoch"]))
            if len(result["cohort"]) != int(reference["n_vl_val"]):
                raise RuntimeError(f"{name}: cohort count disagrees with selected fold JSON")
            if result["metrics"]["vl_n"] != len(result["cohort"]):
                raise RuntimeError(f"{name}: some held-out images failed PA acceptance")
            result["saved_selected_metrics"] = reference["best_vl"]
            result["saved_epoch0_metrics"] = reference["epoch0_vl"]
            result["selected_reproduction"] = aggregate_check(result["metrics"], reference["best_vl"], args.tolerance)
            write_csv(args.out / f"{name}_per_image.csv", result["rows"])
            write_json(args.out / f"{name}_evaluation.json", result)
            report[name] = {k: v for k, v in result.items() if k not in ("rows", "cohort")}
            results[name] = result
            write_json(args.out / "VL_EVALUATION_REPORT.json", report)
            print(f"{name}: exact aggregate check {result['selected_reproduction']}", flush=True)
        if results["control"]["cohort"] != results["curves"]["cohort"]:
            raise RuntimeError("ordered control/curves cohorts or crop/target hashes differ")
        for key in ("core_functions_sha256", "evaluation_parameters_sha256"):
            if results["control"][key] != results["curves"][key]:
                raise RuntimeError(f"paired evaluation engine cores or parameters differ: {key}")
        if args.warm_checkpoint is not None:
            warm_results = []
            for index in range(args.warm_repeats):
                job_file = work / f"warm_job_{index}.json"
                output = args.out / f"warm_repeat_{index + 1}.json"
                write_json(job_file, dict(engine=str(args.control_engine), checkpoint=str(args.warm_checkpoint),
                                          entries=str(entries_file), output=str(output), fold=args.fold))
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_warm-worker", str(job_file)],
                                         capture_output=True, text=True, timeout=180)
                (args.out / f"warm_repeat_{index + 1}.log").write_text(process.stdout + process.stderr)
                if process.returncode != 0:
                    raise RuntimeError(f"fresh-process warm repeat {index + 1} failed; see its log")
                warm = read_json(output)
                if warm["cohort"] != results["control"]["cohort"]:
                    raise RuntimeError("warm repeat cohort/crop/target hashes differ from selected evaluation")
                if warm["core_functions_sha256"] != results["control"]["core_functions_sha256"]:
                    raise RuntimeError("warm repeat evaluation engine core differs")
                warm_results.append(warm)
            hashes = {warm["checkpoint_sha256"] for warm in warm_results}
            if len(hashes) != 1 or next(iter(hashes)) != sha256(args.warm_checkpoint):
                raise RuntimeError("warm repeats did not use the same checkpoint bytes")
            diagnostics = dict(repetitions=len(warm_results), checkpoint_sha256=next(iter(hashes)),
                               cohort_sha256=results["control"]["cohort_sha256"],
                               note="fresh-process AMP/cuDNN numerical diagnostics; no tolerance automatically relaxed", metrics={})
            for field in FIELDS:
                values = [warm["metrics"][field] for warm in warm_results]
                saved = {name: results[name]["saved_epoch0_metrics"][field] for name in results}
                diagnostics["metrics"][field] = dict(repeats=values, repeated_min=min(values), repeated_max=max(values),
                    repeated_range=max(values) - min(values), saved_baselines=saved,
                    repeat_minus_saved={name: [value - float(baseline) for value in values] for name, baseline in saved.items()},
                    combined_range=max([*values, *saved.values()]) - min([*values, *saved.values()]))
            report["warm_numerical_diagnostics"] = diagnostics
            write_json(args.out / "warm_numerical_diagnostics.json", diagnostics)
        if not all(result["selected_reproduction"]["passed"] for result in results.values()):
            raise RuntimeError("selected-checkpoint aggregates exceed strict reproduction tolerance; per-image and numerical diagnostics preserved")
        bootstrap, subjects, leave_out = paired_bootstrap(results["control"], results["curves"], args.bootstrap, args.seed)
        report["bootstrap"] = bootstrap
        write_csv(args.out / "paired_subjects.csv", subjects)
        write_csv(args.out / "leave_one_subject_out.csv", leave_out)
        write_json(args.out / "paired_subject_bootstrap.json", bootstrap)
        lines = ["# Exploratory paired VL diagnostics", "",
                 f"Exact non-TTA, batch-8 CUDA evaluation: {bootstrap['n_images']} matched images / "
                 f"{bootstrap['n_subjects']} held-out subjects; selected-checkpoint aggregates reproduce saved metrics "
                 f"within {args.tolerance:g}°. Raw/crop/canvas/target hashes and ordered cohorts match.", "",
                 "| Metric | Curves − control (°) | Descriptive cluster resampling interval (°) | Subjects with negative delta |",
                 "|---|---:|---:|---:|"]
        for field in FIELDS:
            value = bootstrap["metrics"][field]
            low, high = value["descriptive_cluster_interval_2_5_97_5"]
            lines.append(f"| {field} | {value['curves_minus_control']:+.6f} | [{low:+.6f}, {high:+.6f}] | "
                         f"{value['subjects_with_negative_delta']}/{bootstrap['n_subjects']} |")
        lines.extend(["", "Primary estimates preserve the pilot's image weighting. Equal-subject estimates and "
                      "leave-one-subject-out sensitivity are separate outputs. Bias is signed; a negative bias delta "
                      "does not necessarily improve its absolute magnitude.", "",
                      f"Subject-cluster resampling uses {args.bootstrap} draws and seed {args.seed}. This follow-up is "
                      "exploratory and descriptive. It provides no independent-fold confirmation, significance claim, "
                      "competition-score estimate, or model-adoption decision. Selected epochs retain the GT selection rule."])
        if args.warm_checkpoint is not None:
            lines.extend(["", "Warm baseline numerical checks used fresh processes; see warm_numerical_diagnostics.json. "
                          "The script did not relax any comparator or reproduction tolerance."])
        (args.out / "VL_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--_warm-worker":
        warm_worker(Path(sys.argv[2]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control-engine", type=Path, required=True)
    ap.add_argument("--curves-engine", type=Path, required=True)
    ap.add_argument("--control", type=Path, required=True, help="control pilot output root")
    ap.add_argument("--curves", type=Path, required=True, help="curves pilot output root")
    ap.add_argument("--archives", type=Path, default=Path("/workspace/external_archives"))
    ap.add_argument("--out", type=Path, required=True, help="fresh dedicated diagnostic output directory")
    ap.add_argument("--fold", type=int, default=0, choices=range(5))
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--tolerance", type=float, default=1e-5, help="selected aggregate tolerance, degrees; maximum 1e-5")
    ap.add_argument("--warm-checkpoint", type=Path, help="optional v1 source for repeated fresh-process epoch-0 diagnostics")
    ap.add_argument("--warm-repeats", type=int, default=3)
    args = ap.parse_args()
    if args.bootstrap < 10000 or args.warm_repeats < 3:
        ap.error("use at least 10000 bootstrap draws and at least 3 warm repeats")
    if not 0 < args.tolerance <= 1e-5:
        ap.error("selected-checkpoint reproduction tolerance must be positive and no greater than 1e-5 degrees")
    paths = [args.control_engine, args.curves_engine, args.control, args.curves, args.archives, args.out]
    if args.warm_checkpoint is not None:
        paths.append(args.warm_checkpoint)
    remote_guard(paths)
    for key in ("control_engine", "curves_engine", "control", "curves", "archives", "out", "warm_checkpoint"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        ap.error("--out must be absent or empty; existing outputs will not be overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(status="running", purpose="exact reproduction plus exploratory paired subject diagnostics",
                  bootstrap_seed=args.seed, selected_tolerance_deg=args.tolerance)
    try:
        run(args, report)
        report["status"] = "passed"
        code = 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(f"VL evaluation FAILED: {report['error']}", file=sys.stderr)
        code = 1
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.out / "VL_EVALUATION_REPORT.json", report)
    print(f"VL evaluation {report['status']}; {args.out / 'VL_EVALUATION_REPORT.json'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
