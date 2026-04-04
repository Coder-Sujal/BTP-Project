# ChandraMarg Implementation Documentation

Last updated: 2026-04-03 (UTC)

## 1. Project Goal

Build a lunar terrain hazard-segmentation pipeline that detects crater regions from DEM-derived features and prepares the foundation for hazard-aware path planning.

Current mission flow:

1. Preprocess lunar geospatial rasters.
2. Generate crater masks from crater catalog coordinates.
3. Create ML-ready tile dataset (`X`, `Y`).
4. Train segmentation model (LTKAU-Net style/attention U-Net baseline) on Kaggle T4 GPUs.
5. Use predicted crater maps for future path planning cost-map generation.

## 2. Repository Structure (Implemented)

```text
project/
├── raw_data/                         # Raw LOLA source files
├── processed/
│   ├── dem/                          # DEM GeoTIFFs
│   ├── hillshade/                    # Hillshade rasters
│   ├── slope/                        # Slope rasters
│   ├── sos/                          # SOS rasters
│   └── masks/                        # Generated crater masks
├── kaggle_upload/
│   ├── X/                            # Input tiles (.npy), shape (256,256,3)
│   ├── Y/                            # Label tiles (.npy), shape (256,256)
│   ├── splits/                       # train/val/test split files + manifest.csv
│   └── integrity_report.json         # Dataset QA report
├── result_images/
│   └── data/                         # Visualization artifacts
├── models/                           # Model artifacts/checkpoints
└── scripts/
    ├── config.py
    ├── compute_sos.py
    ├── generate_masks.py
    ├── create_tiles.py
    ├── mask_verify.py
    ├── visualize_masks.py
    ├── check_dataset_integrity.py
    ├── create_dataset_splits.py
    └── tf_dataset.py
```

## 3. Tech Stack

Language: Python 3.x

Core geospatial/data libs:

- `rasterio`
- `numpy`
- `opencv-python` (`cv2`)
- `pandas`
- `pyproj`
- `matplotlib`
- `scipy`

Training:

- `tensorflow` / `keras` (Kaggle GPU runtime for training)

## 4. Data Pipeline Implemented

## 4.1 DEM and Feature Engineering

1. RAW LOLA data converted to DEM GeoTIFF.
2. Hillshade generated from DEM.
3. Slope generated from DEM.
4. SOS generated from slope via Sobel + smoothing + normalization.

Script: `scripts/compute_sos.py`

What SOS script does:

- Reads slope raster.
- Handles NoData.
- Applies Gaussian smoothing.
- Computes gradient magnitude (`dx`, `dy`).
- Normalizes and enhances SOS.
- Writes `float32` GeoTIFF.

## 4.2 Crater Mask Generation

Script: `scripts/generate_masks.py`

Inputs:

- DEM rasters (`processed/dem/*.tif`)
- Filtered crater catalog (`craters_filtered.csv`)

Core logic:

1. Set `PROJ_IGNORE_CELESTIAL_BODY=YES` to handle lunar CRS transformations.
2. Transform crater coordinates from EPSG:4326 to DEM CRS.
3. Convert projected coordinates to pixel index using `src.index(x, y)`.
4. Convert crater diameter to pixel radius.
5. Draw crater circles on binary mask using OpenCV.
6. Save masks as GeoTIFF in `processed/masks/`.

Mask semantics:

- `1`: crater pixel
- `0`: background

## 4.3 Tile Dataset Creation

Script: `scripts/create_tiles.py`

For each region:

1. Read aligned DEM, hillshade, SOS, and mask rasters.
2. Stack features to `X = [dem, hillshade, sos]` (channel-last).
3. Cut non-overlapping `256x256` patches.
4. Save paired `.npy` files:
   - `kaggle_upload/X/<id>.npy`
   - `kaggle_upload/Y/<id>.npy`

Current dataset volume:

- `X`: 6962 tiles
- `Y`: 6962 tiles

## 5. Quality Assurance Utilities Implemented

## 5.1 Mask Visual Checks

- `scripts/visualize_masks.py`: exports mask PNGs for quick visual inspection.
- `scripts/mask_verify.py`: verifies binary mask values (`0` and `1`).

## 5.2 Dataset Integrity Report

Script: `scripts/check_dataset_integrity.py`

Checks:

1. Paired count consistency between `X` and `Y`.
2. Shape consistency.
3. Non-finite checks (`NaN`, `Inf`) in `X`.
4. Binary-mask validation in `Y`.
5. Channel summary stats.
6. Positive pixel ratio (class balance estimate).

Generated output:

- `kaggle_upload/integrity_report.json`

Observed results:

- Paired samples: 6962
- Shape mismatches: 0
- X non-finite samples: 0
- Mask non-binary samples: 0
- Mask positive ratio: ~0.0796365

## 5.3 Deterministic Split Generation

Script: `scripts/create_dataset_splits.py`

Outputs:

- `kaggle_upload/splits/manifest.csv`
- `kaggle_upload/splits/train.txt`
- `kaggle_upload/splits/val.txt`
- `kaggle_upload/splits/test.txt`

Latest split counts:

- Train: 4873
- Val: 1044
- Test: 1045

## 6. Kaggle Dataset Delivery

Target dataset ID:

- `sujalchodvadiya/lunar-crater-dataset`

Versioning approach used:

- Update existing dataset version with directory mode zip:
  - `kaggle datasets version -p kaggle_upload -m "<message>" --dir-mode zip`

Important packaging note:

- Split manifests were made path-portable for Kaggle usage.

## 7. Training Pipeline Implemented

## 7.1 TensorFlow Input Pipeline

Script: `scripts/tf_dataset.py`

Capabilities:

1. Read split manifest.
2. Resolve data paths.
3. Load `.npy` feature/mask pairs.
4. Enforce binary mask format (`HxW` -> `HxWx1`).
5. Compute train-channel mean/std.
6. Normalize inputs.
7. Apply augmentation (flip + rotation).
8. Build `train_ds`, `val_ds`, `test_ds` with batching + prefetch.

## 7.2 Baseline Training Setup

Notebook training included:

- Attention U-Net style model for crater segmentation.
- Mixed precision in Kaggle runtime.
- Optimizer: Adam.
- Metrics: Dice, IoU, Precision, Recall.
- Callbacks: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping.

## 7.3 Improved Training Setup (Current)

Improvements introduced:

1. Weighted loss:
   - `0.3 * BCE + 0.7 * DiceLoss`
2. Crater-aware oversampling strategy.
3. Threshold sweep strategy for better binary mask conversion after training.

Important runtime fix applied:

- Infinite-cardinality dataset with `.repeat()` required explicit `steps_per_epoch`.

## 8. Model Results So Far

Latest reported training result:

- Early stopping at epoch 26.
- Training snapshot:
  - `dice_coef`: 0.4251
  - `iou_metric`: 0.3570
  - `loss`: 0.4996
  - `precision`: 0.4724
  - `recall`: 0.4583
- Validation snapshot:
  - `val_dice_coef`: 0.3440
  - `val_iou_metric`: 0.3922
  - `val_loss`: 0.5380
  - `val_precision`: 0.4763
  - `val_recall`: 0.3289

Interpretation:

1. Model is learning crater signal and generalizing better than initial baseline.
2. Validation recall is still lower than precision, indicating missed crater pixels remain a key gap.
3. Accuracy improvement is still open and planned for later iteration.

## 9. Definitions Used During Evaluation

Dice coefficient:

- Measures overlap between prediction and ground truth.
- Higher is better, range `[0, 1]`.

IoU (Jaccard):

- Intersection over union of predicted and true mask.
- Higher is better, range `[0, 1]`.

Loss in current run:

- Weighted BCE + Dice loss to address class imbalance and overlap quality simultaneously.

## 10. Current Status Summary

Completed:

1. Geospatial preprocessing pipeline.
2. Crater mask generation with CRS transformation handling.
3. Tile dataset generation (`X`, `Y`).
4. Dataset integrity tooling + deterministic splits.
5. Kaggle dataset publishing and versioning flow.
6. Kaggle training pipeline and first improved model run.
7. First path-planning stack:
   - Costmap generation from crater mask + slope
   - Obstacle inflation and blocked-map export
   - A* path planner with CSV + GeoTIFF route export
   - Route visualization overlay

In progress:

1. Continued model accuracy optimization.
2. Upgrading path-planning input from ground-truth masks to model-predicted crater probability rasters.

## 11. Planned Path-Planning Phase (Next)

Planned steps:

1. Run full-tile inference to get crater probability maps.
2. Reconstruct full-scene probability rasters.
3. Replace GT mask input in costmap with predicted probability raster.
4. Tune inflation radius, threshold, and slope/crater weights against route safety.
5. Add multi-scenario evaluation (multiple start-goal pairs).
6. Integrate post-processing (path simplification/smoothing).

## 12. Reproducibility Commands

Integrity report:

```bash
python3 scripts/check_dataset_integrity.py \
  --x-dir kaggle_upload/X \
  --y-dir kaggle_upload/Y \
  --output-json kaggle_upload/integrity_report.json
```

Split generation:

```bash
python3 scripts/create_dataset_splits.py \
  --x-dir kaggle_upload/X \
  --y-dir kaggle_upload/Y \
  --out-dir kaggle_upload/splits \
  --strategy source \
  --seed 42
```

TensorFlow dataset smoke test:

```bash
python3 scripts/tf_dataset.py \
  --manifest kaggle_upload/splits/manifest.csv \
  --batch-size 16
```

Kaggle dataset version update:

```bash
kaggle datasets version -p kaggle_upload -m "Add splits and integrity report" --dir-mode zip
```

Path-planning costmap generation:

```bash
python3 scripts/build_costmap.py \
  --region 80s \
  --inflation-radius-px 4 \
  --crater-block-threshold 0.5
```

Path planning with A*:

```bash
python3 scripts/plan_path.py \
  --region 80s \
  --start-row 200 --start-col 200 \
  --goal-row 700 --goal-col 700 \
  --allow-nearest-free
```

Route overlay visualization:

```bash
python3 scripts/visualize_route.py --region 80s --downsample 8
```

## 13. Known Caveats

1. Local VM package conflicts can occur between TensorFlow and OpenCV due to NumPy version constraints.
2. Kaggle path resolution requires verifying dataset mount path and manifest existence after each version upload.
3. Segmentation output still needs threshold tuning and error-analysis-driven refinements for stronger recall.

## 14. Recommended Immediate Next Work Items

1. Freeze and archive current best model + threshold + normalization stats.
2. Start inference pipeline to produce full crater probability maps.
3. Run costmap + A* using model predictions (not GT masks).
4. Evaluate path safety metrics across multiple routes and tune planner weights.
