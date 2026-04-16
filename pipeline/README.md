# 4DGS Harmonization Pipeline

Optimizes the spherical harmonic (SH) coefficients of masked Gaussians so that
an inserted object's lighting matches the surrounding scene.

## Overview

Given a trained 4D Gaussian Splatting scene and a per-Gaussian mask (`.pt`)
identifying the object to harmonize, the pipeline:

1. Renders every training view and projects the 3D mask into 2D
2. Runs a pretrained [Harmonizer](https://github.com/ZHKKKe/Harmonizer) CNN to
   predict what the object *should* look like under the scene's lighting
3. Optimizes a per-Gaussian SH residual (`delta_sh`) by backpropagating through
   the differentiable rasterizer to match those targets
4. Bakes the residual into the Gaussians and saves a new `.ply`

## Prerequisites

```bash
# Inside the sa4d conda environment
conda activate sa4d
pip install kornia plyfile lpips   # lpips only needed if --use_lpips
```

The pretrained Harmonizer weights must exist at:
```
~/Harmonizer/pretrained/harmonizer.pth
```

## Quick Start

```bash
cd ~/new_sa4d/sa4d

python -m pipeline.run_harmonize \
    --model_path output/hypernerf/torchocolate \
    --source_path data/hypernerf/torchocolate \
    --mask_path output/hypernerf/torchocolate/segment_results/torchocolate.pt \
    --output_ply output/hypernerf/torchocolate/point_cloud/iteration_14000/harmonized.ply
```

## Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to trained 4DGS output (contains `cfg_args`, `point_cloud/`) |
| `--source_path` | Path to raw training data (contains `dataset.json`, images) |
| `--mask_path` | Path to `.pt` mask table from segmentation pipeline |
| `--output_ply` | Where to save the harmonized `.ply` file |

### Optional

| Argument | Default | Description |
|----------|---------|-------------|
| `--iteration` | `-1` (latest) | Which 4DGS checkpoint iteration to load |
| `--configs` | `None` | Path to `.py` hyperparameter config (e.g. `arguments/hypernerf/default.py`) |
| `--harmonizer_weights` | `~/Harmonizer/pretrained/harmonizer.pth` | Path to Harmonizer `.pth` weights |
| `--num_iterations` | `500` | Number of SH optimization steps |
| `--lr` | `1e-3` | Learning rate for Adam optimizer on `delta_sh` |
| `--reg_weight` | `0.01` | L2 regularization on `delta_sh` (prevents over-correction) |
| `--use_lpips` | `False` | Add perceptual loss (LPIPS) alongside L1 |
| `--lpips_weight` | `0.1` | Weight for LPIPS loss term |
| `--sigma` | `2.0` | Temporal smoothing sigma for Harmonizer filter args |
| `--skip_precompute` | `False` | Load cached targets instead of re-rendering |
| `--precompute_only` | `False` | Only generate targets, skip SH optimization |
| `--targets_dir` | `<model_path>/harmonize_cache` | Directory for cached target images |
| `--log_interval` | `50` | Print loss every N iterations |

## Pipeline Steps in Detail

### Step 1-3: Data Loading (`pipeline/data_loading.py`)

- `load_scene()` reads `cfg_args`, builds a `GaussianModel`, loads the `.ply`
  and `deformation.pth`, and returns all training cameras from the `Scene` object
- `load_mask_table()` loads the `.pt` file which contains:
  - `mask_table`: `[N_frames, N_gaussians]` boolean tensor
  - `time_map`: `[N_frames]` float tensor mapping frames to normalized time
  - `selected_ids` or `removed_ids`: metadata about which object IDs were selected
- `load_harmonizer()` instantiates the Harmonizer CNN and loads pretrained weights

### Step 4: Target Precomputation (`pipeline/precompute_targets.py`)

For each training camera view:

1. **Render** the full scene RGB with `render()` → composite image `[3, H, W]`
2. **Render 2D mask** with `render_mask()` → projects per-Gaussian boolean mask
   through the rasterizer into a pixel-space mask `[1, H, W]`
3. **Predict filter args** — the Harmonizer backbone predicts 6 scalars:
   temperature, brightness, contrast, saturation, highlight, shadow
4. **View-consensus** — average filter args across all views at each timestep
5. **Temporal smoothing** — Gaussian-smooth the args along the time axis
6. **Apply filters** — the Harmonizer's white-box filters generate target images

Targets are cached to `<model_path>/harmonize_cache/harmonize_targets.pt`.
Use `--skip_precompute` on subsequent runs to reload from cache.

### Step 5: SH Optimization (`pipeline/optimize_sh.py`)

Creates zero-initialized residuals:
- `delta_sh_dc`: `[N_object, 1, 3]` — adjusts base color (DC term)
- `delta_sh_rest`: `[N_object, 15, 3]` — adjusts view-dependent appearance

Each iteration:
1. Samples a random (view, frame) pair
2. Builds `base_sh + delta_sh` on the autograd graph
3. Passes through the (frozen) deformation network and differentiable rasterizer
4. Computes masked L1 loss against the precomputed target
5. Backpropagates through the rasterizer to update `delta_sh`

The deformation network is frozen during optimization — its forward pass still
transforms SH coefficients by time, but its weights don't receive gradients.

### Step 6: Save (`pipeline/optimize_sh.py`)

- `apply_delta_sh()` permanently adds the optimized residual to the Gaussian model
- `save_harmonized_ply()` calls `gaussians.save_ply()` to write the result
- A `delta_sh.pt` file is also saved for inspection

## Rendering the Result

The output `.ply` must be rendered with the sa4d renderer (not a mesh viewer):

```bash
cd ~/new_sa4d/sa4d

# Swap in harmonized ply
cd output/hypernerf/torchocolate/point_cloud/iteration_14000
cp scene_point_cloud.ply scene_point_cloud_original.ply
cp harmonized.ply scene_point_cloud.ply
cd ~/new_sa4d/sa4d

# Render video
python render_4dgs.py \
    --model_path output/hypernerf/torchocolate \
    --source_path data/hypernerf/torchocolate \
    --skip_train --skip_test \
    --configs arguments/hypernerf/default.py \
    --mode scene

# Output: output/hypernerf/torchocolate/video/ours_14000/video_rgb.mp4

# Restore original
cd output/hypernerf/torchocolate/point_cloud/iteration_14000
cp scene_point_cloud_original.ply scene_point_cloud.ply
```

Note: check that `CUDA_VISIBLE_DEVICES` at the top of `render_4dgs.py` matches
your GPU (default is `"2"`, you likely want `"0"`).

## Tuning

| Symptom | Fix |
|---------|-----|
| Object color barely changes | Increase `--lr` (e.g. `5e-3`) or `--num_iterations` (e.g. `1000`) |
| Object looks over-saturated / noisy | Increase `--reg_weight` (e.g. `0.05`) or decrease `--lr` |
| Temporal flickering | Increase `--sigma` (e.g. `4.0`) for more smoothing |
| Want sharper detail matching | Add `--use_lpips` |

## File Structure

```
sa4d/pipeline/
├── __init__.py
├── README.md              ← this file
├── run_harmonize.py       ← main entry point
├── data_loading.py        ← load scene, mask, harmonizer
├── precompute_targets.py  ← render views, predict harmonization targets
└── optimize_sh.py         ← backprop delta_sh through rasterizer
```
