# Third-party notices and external resources

This project is licensed under GPL-3.0-only (see `LICENSE`). It incorporates or uses the
following third-party material. Everything listed is publicly and freely available.
Everything not listed here (models trained in our own Kaggle kernels, the geometry, shrinkage,
native-geometry, clip-detection, rater-calibration and assembly code) is our own work.

## Code adapted from other sources

| Component | Source | License | Where used | Modification |
|---|---|---|---|---|
| Scanner scale-bar and viewport layout rules (probe pixels, tick rules, per-depth viewport boxes) | AmbrosM, "UMUD Quick and Dirty", https://www.kaggle.com/code/ambrosm/umud-quick-and-dirty | Apache-2.0 (public Kaggle notebook) | `src/umud/geometry.py` (`ultrasound_metadata`, called by `src/umud/layout.py` for the test layout table), `kaggle/fasc_orient_v1/train.py` and `kaggle/fasc_orient_v2/train.py` (`layout_viewport`), `scripts/aniso_phys_v9.py` (`detect_layout`) | Rewritten as standalone functions with new structure and an OCR fallback |
| Competition metric | Paul Ritsche, "umud-score", https://www.kaggle.com/code/paulritsche/umud-score | Apache-2.0 | `src/umud/metric.py` | Independent re-implementation of the published formula |
| DL_Track_US VGG16-UNet inference path | Ritsche et al., DL_Track_US, https://github.com/PaulRitsche/DL_Track_US | Apache-2.0 | `kaggle/infer_maskbank/` (fallback aponeurosis maps; the fallback is never reached by the final submissions: the primary map is used on 761 / 761 test and benchmark images) | Ported preprocessing for batch inference |

The full Apache License 2.0 text is in `LICENSES/Apache-2.0.txt`
(also at https://www.apache.org/licenses/LICENSE-2.0).

## Pretrained models

| Model | Source | Use |
|---|---|---|
| ResNet-34 ImageNet weights | via `segmentation-models-pytorch` 0.5.0 (MIT) / torchvision (BSD) | encoder initialisation (aponeurosis and fascicle U-Net++) |
| `convnext_small.fb_in22k_ft_in1k` | via `timm` 1.0.29 (Apache-2.0; Meta ConvNeXt weights, ImageNet-22k pretraining, ImageNet-1k fine-tuning) | encoder initialisation (apo_v2, fasc_orient_v1) |
| DL_Track_US weights `model-apo-VGG16_DiceBCE_512_300.h5`, `model-fasc-VGG16_DiceBCE_200_512.h5` | OSF node 7mjsc, https://osf.io/7mjsc/ (Apache-2.0) | fallback maps only; not used by the final numbers |

Weights are downloaded from their original sources at run time (inside the Kaggle kernels) and
are not redistributed here. All our models were trained on the competition training data only;
no model was trained or fine-tuned on test images for the final submissions.

## External data

| Dataset | Source | License | Use |
|---|---|---|---|
| UMUD benchmark muscle architecture dataset (static, 35 images, 7 raters) and GM dynamic calf-raise sequence | OSF node xbawc, https://osf.io/xbawc/ | CC BY 4.0 | validation; empirical-Bayes error model; half-strength rater-style offset (as suggested by the organiser, thread 739660) |
| UMUD "Example Annotated Images" (vastus lateralis, expert-drawn fascicles) | OSF node xbawc | CC BY 4.0 | expert FL / PA parsed from the drawn lines (`scripts/vl_labels.py`): validation and the error model |

## Tools and Python packages

| Tool / package | License | Use |
|---|---|---|
| Tesseract OCR 5.x (system program) | Apache-2.0 | reads the burned-in muscle name on Siemens / Lumify images (`scripts/read_muscle_labels.py`) |
| Kaggle CLI | Apache-2.0 | pushing kernels / downloading outputs |
| numpy, pandas, scipy, scikit-image (BSD-3-Clause); opencv-python-headless (Apache-2.0); Pillow (MIT-CMU); openpyxl, PyYAML (MIT); torch, torchvision (BSD-3-Clause); segmentation-models-pytorch (MIT); timm (Apache-2.0) | as listed | runtime dependencies, pinned in `uv.lock` / `requirements-lock.txt` |

## Competition data

The UMUD challenge data is licensed CC BY-NC-SA 4.0 and is used only for this competition.
It is not included in or redistributed with this repository (`data/` and `artifacts/` are git-ignored).
