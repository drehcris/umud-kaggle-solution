"""Remote-only synthetic CUDA smoke; never invokes the pilot's main() or touches its outputs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def smoke(args, report: dict) -> None:
    # Set before torch loads; deterministic CUDA evaluation must survive save/reload.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this smoke only on the GPU pod; CPU fallback is disabled.")
    engine_path, source_path = args.engine.resolve(strict=True), args.checkpoint.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("umud_smoke_engine", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load engine: {engine_path}")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)  # imports definitions; main() stays uncalled
    import segmentation_models_pytorch as smp
    import timm

    if smp.__version__ != engine.SMP_VERSION:
        raise RuntimeError(f"requires segmentation-models-pytorch=={engine.SMP_VERSION}; found {smp.__version__}")
    if engine.SIZE != 512:
        raise RuntimeError(f"this smoke requires the prepared 512-pixel engine; found SIZE={engine.SIZE}")
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    torch.manual_seed(engine.SEED)
    torch.cuda.manual_seed_all(engine.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(device)
    # PyTorch 2.8 needs an initialized allocator before resetting its peak stats.
    init_probe = torch.empty(1, device=device)
    del init_probe
    torch.cuda.reset_peak_memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=30)
    report.update(gpu=props.name, gpu_total_vram_gib=props.total_memory / 2**30,
                  nvidia_smi=driver.stdout.strip(), python=platform.python_version(),
                  torch=torch.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  smp=smp.__version__, timm=timm.__version__, engine=str(engine_path),
                  engine_sha256=sha256(engine_path), warm_checkpoint=str(source_path),
                  warm_checkpoint_sha256=sha256(source_path), batch=2, size=engine.SIZE,
                  synthetic_only=True, seed=engine.SEED, eval_rtol=1e-4, eval_atol=1e-5)
    warm = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(warm, dict) or "state_dict" not in warm:
        raise RuntimeError("warm checkpoint must contain the engine's state_dict")
    if warm.get("fold", 0) != 0:
        raise RuntimeError(f"expected fold-0 source; found fold={warm.get('fold')}")
    model = engine.build_model(pretrained=False)
    bad = engine.load_state(model, warm["state_dict"], torch)
    if bad:
        raise RuntimeError(f"non-finite warm tensors (no repair in smoke): {bad[:10]}")
    report["warm_epoch"] = warm.get("epoch")
    del warm
    model = model.to(device).to(memory_format=torch.channels_last)
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    dec = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc, "lr": engine.LR_ENC},
                            {"params": dec, "lr": engine.LR_DEC}], weight_decay=engine.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    raw = torch.randint(0, 256, (2, engine.SIZE, engine.SIZE), dtype=torch.uint8, device=device)
    x = engine.to_input(raw, torch)
    # Dense synthetic targets exercise all three heads without depending on competition images.
    yy = torch.arange(engine.SIZE, device=device).view(1, -1, 1)
    xx = torch.arange(engine.SIZE, device=device).view(1, 1, -1)
    target = (((yy // 24 + xx // 32) % 2) > 0).float().expand(2, -1, -1)
    skeleton = (target * (xx % 32 == 0)).float()
    model.train()
    probe = next(p for p in model.encoder.parameters() if p.requires_grad)
    before_param = probe.detach().clone()
    torch.cuda.synchronize(device)
    t_step = time.perf_counter()
    report["amp_attempts"] = []
    for attempt in range(1, 13):
        scale_before = float(scaler.get_scale())
        opt.zero_grad(set_to_none=True)
        with engine.amp(torch, device):
            output = model(x)
        output = output.float()
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("non-finite CUDA forward output")
        logits = output[:, 0]
        p = torch.sigmoid(logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=torch.tensor([engine.POS_WEIGHT], device=device))
        dice = (1 - (2 * (p * target).sum((1, 2)) + 1) /
                (p.sum((1, 2)) + target.sum((1, 2)) + 1)).mean()
        skel = (1 - (p * skeleton).sum((1, 2)) / skeleton.sum((1, 2)).clamp_min(1)).mean()
        angle = math.radians(-20)
        ori = 0.5 * ((torch.tanh(output[:, 1]) - math.cos(2 * angle)).abs().mean() +
                     (torch.tanh(output[:, 2]) - math.sin(2 * angle)).abs().mean())
        loss = engine.W_BCE * bce + engine.W_DICE * dice + engine.W_SKEL * skel + engine.W_ORI * ori
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite synthetic loss")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)  # records non-finite gradients so scaler.step can safely skip AdamW
        named_grads = [(name, param.grad) for name, param in model.named_parameters() if param.grad is not None]
        if not named_grads:
            raise RuntimeError("synthetic backward produced no gradients")
        bad_grads = [name for name, grad in named_grads if not bool(torch.isfinite(grad).all())]
        # Accumulate the inspection norm in float64 so finite fp32 gradients cannot overflow
        # merely while measuring their norm. Clipping still uses the engine's strict fp32 path.
        grad_norm = None if bad_grads else torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(grad.detach(), ord=2, dtype=torch.float64) for _, grad in named_grads]), ord=2)
        norm_finite = grad_norm is not None and bool(torch.isfinite(grad_norm))
        info = dict(attempt=attempt, scale_before=scale_before, overflow=bool(bad_grads) or not norm_finite,
                    nonfinite_gradient_tensors=len(bad_grads), nonfinite_gradient_names=bad_grads[:10],
                    gradient_norm=float(grad_norm) if norm_finite else None)
        report["amp_attempts"].append(info)
        if info["overflow"]:
            if not bad_grads:
                # GradScaler can only skip when unscale_ found inf/nan in a gradient tensor.
                # Do not call its optimizer step if only a separately computed norm failed.
                raise RuntimeError("non-finite float64 norm despite finite gradient tensors; refusing optimizer step")
            scaler.step(opt)  # unscale_ found inf/nan: this must skip the optimizer entirely
            scaler.update()  # ordinary GradScaler backoff; recompute the batch at the new scale
            torch.cuda.synchronize(device)
            info.update(scale_after=float(scaler.get_scale()), optimizer_step="skipped_overflow")
            if not torch.equal(before_param, probe.detach()):
                raise RuntimeError("GradScaler overflow changed encoder weights instead of skipping AdamW")
            if info["scale_after"] >= scale_before:
                raise RuntimeError("GradScaler did not reduce its scale after gradient overflow")
            print(f"AMP smoke attempt {attempt}/12 overflow: scale {scale_before:g} -> "
                  f"{info['scale_after']:g}; optimizer skipped", flush=True)
            continue
        if float(grad_norm) <= 0:
            raise RuntimeError("synthetic backward produced no gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize(device)
        info.update(scale_after=float(scaler.get_scale()), optimizer_step="applied")
        if torch.equal(before_param, probe.detach()):
            raise RuntimeError("AdamW did not update the encoder probe parameter")
        report["successful_amp_attempt"] = attempt
        print(f"AMP smoke attempt {attempt}/12 finite: scale {scale_before:g}; "
              f"gradient_norm={float(grad_norm):.6g}; optimizer applied", flush=True)
        break
    else:
        raise RuntimeError("no finite AMP optimizer update after 12 GradScaler backoff attempts")
    report.update(loss=float(loss.detach()), gradient_norm=float(grad_norm),
                  train_step_seconds=time.perf_counter() - t_step, scaler_scale=float(scaler.get_scale()))

    def predict(net):
        net.eval()
        with torch.inference_mode():
            return torch.stack(engine.forward_maps(net, x, torch, tta=False), dim=1).float().cpu()

    before = predict(model)
    repeat = predict(model)
    if not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(repeat).all()):
        raise RuntimeError("non-finite evaluation outputs")
    if not torch.equal(before, repeat):
        raise RuntimeError("evaluation is not deterministic on identical inputs")
    state = {k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
             for k, v in model.state_dict().items()}
    bad = engine.state_is_finite(state, torch)
    if bad:
        raise RuntimeError(f"non-finite trained state: {bad[:10]}")
    checkpoint = args.out / "smoke_only_step.pt"
    torch.save({"state_dict": state, "smoke_only": True, "epoch": -1, "fold": 0,
                "encoder": engine.ENCODER, "size": engine.SIZE, "dtype": "float32"}, checkpoint)
    del state, model, opt, enc, dec, probe, before_param, output, loss, logits, p, bce, dice, skel, ori, named_grads
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if any(v.dtype != torch.float32 for v in reloaded["state_dict"].values() if v.is_floating_point()):
        raise RuntimeError("smoke checkpoint contains non-float32 floating tensors")
    check_model = engine.build_model(pretrained=False)
    bad = engine.load_state(check_model, reloaded["state_dict"], torch)
    if bad:
        raise RuntimeError(f"non-finite reloaded state: {bad[:10]}")
    check_model = check_model.to(device).to(memory_format=torch.channels_last)
    after = predict(check_model)
    if not bool(torch.isfinite(after).all()):
        raise RuntimeError("non-finite reloaded evaluation outputs")
    if not torch.allclose(before, after, rtol=report["eval_rtol"], atol=report["eval_atol"]):
        raise RuntimeError(f"checkpoint reload changed outputs; max_abs_diff={float((before - after).abs().max())}")
    torch.cuda.synchronize(device)
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  max_abs_eval_difference=float((before - after).abs().max()),
                  peak_allocated_vram_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                  peak_reserved_vram_gib=torch.cuda.max_memory_reserved(device) / 2**30,
                  deterministic_eval=True, reload_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="fresh, dedicated smoke output directory")
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        ap.error("--out must be absent or empty; refusing to overwrite existing results")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"status": "running", "purpose": "synthetic CUDA smoke; no quality measurement"}
    try:
        smoke(args, report)
        report["status"] = "passed"
        code = 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(f"CUDA smoke FAILED: {report['error']}", file=sys.stderr)
        code = 1
    report["elapsed_seconds"] = time.perf_counter() - started
    report_path = args.out / "SMOKE_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"CUDA smoke {report['status']}; report: {report_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
