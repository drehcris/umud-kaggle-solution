# apo_v2 BatchNorm fix and re-inference (Kaggle CPU kernels)

These three kernels produced the `apo_v2fix_{band,upper,lower}` aponeurosis maps that both finals use
(`artifacts/eval_v3/apo2/maskbank_fix/`). They are the exact files that ran on Kaggle (2026-10-21).

| folder | kernel | maps written | used by the finals |
|---|---|---|---|
| `test/` | `shivateja005/umud-apo-v2-fix-test` | `test__apo_v2fix_*.npz` (309 test images) | yes (measurement of the test set) |
| `ext/` | `shivateja005/umud-apo-v2-fix-ext` | `{static,dynamic,vl}__apo_v2fix_*.npz` (OSF benchmarks) | yes (error model, rater-style rule) |
| `oof/` | `shivateja005/umud-apo-v2-fix-oof` | `train_oof__apo_v2fix_*.npz` | no (diagnostics only) |

Each script is `kaggle/apo_v2/train.py` (model and preprocessing code) with an appended inference-fix tail; the three
copies differ only in the line `PART = "test" | "ext" | "oof"`. The fix: the `umud-apo-v2` checkpoints were saved in
float16, so BatchNorm running variances above 65504 overflowed to inf. The tail reloads every fold checkpoint,
re-estimates the broken BatchNorm statistics on that fold's own training images (the held-out fold is never used),
checks the reproduction of the logged validation metric on held-out images, and writes the maps (CPU, no TTA).

Checkpoint source: `kernel_sources: shivateja005/umud-apo-v2` (5 fold checkpoints trained by `kaggle/apo_v2/train.py`
on the competition training data only). `umud-run --stage maps` writes ready-to-push copies with the commands.

GPL-3.0-only.
