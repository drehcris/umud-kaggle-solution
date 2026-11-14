# UMUD Challenge — Geometry-First Ultrasound Measurements

Code and prediction files for the Kaggle [UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data).

The task predicts three measurements per test image: `pa_deg` (pennation angle), `fl_mm` (fascicle length), and `mt_mm` (muscle thickness). Lower normalized MAE is better.

## Best public submission

`submissions/best_public_submission_A_plus_v2.csv` is the best file obtained in this workspace, with public leaderboard score **0.21529**. It has 309 rows plus the required header and is ready to upload as a Kaggle submission.

The file was derived from a geometry-first pipeline and includes scanner-specific post-calibration for Philips Lumify, Siemens Juniper, and Telemed Artus images. The competition images and generated model caches are intentionally not included in this repository.

## Reproduction

The full pipeline requires the competition data, model outputs/caches, and external benchmark files. Install the pinned environment with:

```bash
uv sync --locked
PYTHONPATH=src python scripts/umud_run.py --help
```

The geometry/post-processing implementation is in `src/umud/`; inference/training entry points are in `kaggle/`; and pipeline configurations are in `configs/`.

To validate the submission format:

```bash
python scripts/validate_submission.py submissions/best_public_submission_A_plus_v2.csv
```

## Tested variants

`experiments/` contains four one-change calibration variants. `v1_lumify_pa_minus025.csv` scored **0.21583**, worse than the best **0.21529** file. The other variants are retained for research and were not selected.

## Repository contents

- `submissions/`: ready-to-upload prediction CSVs.
- `experiments/`: controlled calibration variants.
- `src/umud/`: geometry, layout, clip detection, metric, shrinkage, and post-processing code.
- `scripts/`: training, inference, evaluation, and validation utilities.
- `kaggle/`: Kaggle kernel entry points.
- `configs/`: pipeline configurations.
- `metadata/test_layout.csv`: small scanner/clip metadata table.

## Attribution and licensing

See `LICENSE`, `LICENSES/`, and `THIRD_PARTY_NOTICES.md`. Do not redistribute competition images, private model caches, or external benchmark files unless their respective terms permit it.
