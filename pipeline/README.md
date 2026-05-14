# 4DGS Harmonization Pipeline

Optimizes the spherical harmonic (SH) coefficients of masked Gaussians so that
an inserted object's lighting matches the surrounding scene.

## Overview

Given a trained 4D Gaussian Splatting scene and a per-Gaussian mask (`.pt`)
identifying the object to harmonize, the pipeline:

1. Renders every training view and projects the 3D mask into 2D
2. Generates target images — either via a pretrained harmonizer CNN or by
   loading ground-truth Scene B renders (`--harmonizer scene_b`)
3. Optimizes a per-Gaussian SH residual (`delta_sh`) by backpropagating through
   the differentiable rasterizer to match those targets
4. Bakes the residual into the Gaussians and saves a new `.ply`

## Prerequisites

```bash
# Inside the sa4d conda environment
conda activate sa4d
pip install kornia plyfile lpips   # lpips only needed if --use_lpips
```

For the whitebox harmonizer backend, weights must exist at:
```
~/Harmonizer/pretrained/harmonizer.pth
```

## Recommended: DC-Only Harmonization

**This is the recommended approach for relighting.** Optimizing only the DC (zeroth-order)
SH band produces clean, artifact-free color shifts. Optimizing all SH bands causes
rainbow/iridescent artifacts because higher-order bands encode view-dependent color.

### Quick Start (with harmonizer CNN)

```bash
cd ~/new_sa4d/sa4d

python -m pipeline.run_harmonize \
    --model_path output/hypernerf/torchocolate \
    --source_path data/hypernerf/torchocolate \
    --mask_path output/hypernerf/torchocolate/segment_results/torchocolate.pt \
    --output_ply output/hypernerf/torchocolate/point_cloud/iteration_14000/harmonized.ply \
    --configs arguments/hypernerf/default.py \
    --harmonizer whitebox \
    --lr_dc 0.01 \
    --lr_rest 0.0 \
    --num_iterations 500 \
    --reg_weight 0.01
```

### Quick Start (with Scene B ground truth as target)

If you have paired Scene A / Scene B renders (same cameras, different lighting):

```bash
cd ~/new_sa4d/sa4d

python -m pipeline.run_harmonize \
    --model_path output/v4/static_A \
    --source_path data_v4_static/scene_A \
    --mask_path output/v4/static_A/segment_results/cat_mask.pt \
    --output_ply output/v4/static_A/point_cloud/iteration_45000/harmonized.ply \
    --configs arguments/dnerf/joint_static_50k.py \
    --harmonizer scene_b \
    --scene_b_path data_v4_static/scene_B \
    --lr_dc 0.01 \
    --lr_rest 0.0 \
    --num_iterations 500 \
    --reg_weight 0.01
```

## Critical: DC-Only vs All-Band Optimization

| Setting | CLI flags | Result |
|---------|-----------|--------|
| **DC-only (recommended)** | `--lr_dc 0.01 --lr_rest 0.0` | Clean uniform color shift, no artifacts |
| Separate LR | `--lr_dc 0.01 --lr_rest 0.0001` | Mostly clean, very slight shimmer |
| Band regularization | `--lr 0.01 --lambda_sh_band 0.1` | Reduced artifacts but still visible |
| All bands equal (BAD) | `--lr 0.01` | Rainbow/iridescent artifacts, transparent breakup |

**Why:** SH bands l=1,2,3 encode view-dependent color. When the optimizer pushes them
hard to match pixel targets, it creates colors that look correct from the training
viewpoints but produce rainbow artifacts from novel views. DC-only avoids this entirely
by only adjusting the view-independent base color.

## Required File Paths

Your friend needs these files/directories set up:

### 1. Trained 4DGS Model (`--model_path`)

```
output/your_scene/
├── cfg_args                              # saved training config (auto-generated)
├── point_cloud/
│   └── iteration_XXXXX/
│       ├── scene_point_cloud.ply         # the Gaussian model
│       ├── deformation.pth              # deformation network weights
│       ├── deformation_table.pth        # which Gaussians are deformed
│       └── deformation_accum.pth        # deformation accumulator
```

### 2. Source Training Data (`--source_path`)

```
data/your_scene/
├── train/
│   ├── r_0.png
│   ├── r_1.png
│   └── ...
├── test/
│   ├── r_0.png
│   └── ...
├── transforms_train.json                # camera poses + times
└── transforms_test.json
```

The `transforms_*.json` must have `camera_angle_x` and `frames` with
`file_path`, `time`, and `transform_matrix` fields (D-NeRF / Blender format).

### 3. Per-Gaussian Mask (`--mask_path`)

A `.pt` file containing a dict with:

```python
{
    'mask_table': torch.BoolTensor([N_frames, N_gaussians]),  # True = object
    'time_map': torch.FloatTensor([N_frames]),                # normalized time [0, 1]
}
```

`N_gaussians` must match the PLY point count exactly.

**Creating a mask from a bounding box:**
```python
import torch, numpy as np
from plyfile import PlyData

ply = PlyData.read('output/your_scene/point_cloud/iteration_XXXXX/scene_point_cloud.ply')
xyz = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=1)

# Define your object's bounding box in the scene's coordinate space
bbox_min = np.array([x_min, y_min, z_min])
bbox_max = np.array([x_max, y_max, z_max])
in_bbox = ((xyz >= bbox_min) & (xyz <= bbox_max)).all(axis=1)

# For static scenes: 1 frame. For dynamic: N_frames (same mask replicated)
mask_table = torch.tensor(in_bbox, dtype=torch.bool).unsqueeze(0)
time_map = torch.tensor([0.0])  # single frame for static

torch.save({'mask_table': mask_table, 'time_map': time_map},
           'output/your_scene/segment_results/object_mask.pt')
```

### 4. Training Config (`--configs`)

A `.py` file defining `OptimizationParams` and `ModelHiddenParams`. Examples:

- **Static scene (no deformation):** `arguments/dnerf/joint_static_50k.py`
- **Dynamic scene:** `arguments/dnerf/joint_dynamic_50k.py`
- **HyperNeRF:** `arguments/hypernerf/default.py`

Key settings in the config that matter:
- `no_dx=True, no_ds=True, no_dr=True` → disables deformation (static scene)
- `iterations` → must be >= the checkpoint you're loading

### 5. Harmonizer Targets (one of the following)

**Option A: Whitebox Harmonizer CNN** (default)
```
~/Harmonizer/pretrained/harmonizer.pth
~/Harmonizer/src/model/harmonizer.py     # model definition
```
Clone from: https://github.com/ZHKKKe/Harmonizer

**Option B: PCT-Net** (`--harmonizer pctnet`)
```
~/PCT-Net-Image-Harmonization/pretrained_models/PCTNet_CNN.pth
```

**Option C: Scene B ground truth** (`--harmonizer scene_b --scene_b_path <path>`)
```
data/your_scene_B/
├── train/
│   ├── r_0.png      # same camera index as scene_A
│   ├── r_1.png
│   └── ...
└── test/
    └── ...
```
Scene B must have the exact same camera poses and image indices as Scene A.
Only the lighting/content differs.

## Rendering the Result

Use `--ply_path` to render the harmonized PLY without overwriting the original:

```bash
python render_4dgs.py \
    --model_path output/your_scene \
    --skip_train \
    --configs arguments/your_config.py \
    --ply_path output/your_scene/point_cloud/iteration_XXXXX/harmonized.ply
```

Output video: `output/your_scene/video/ours_XXXXX/video_rgb.mp4`

## Full Arguments Reference

### Required

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to trained 4DGS output (contains `cfg_args`, `point_cloud/`) |
| `--source_path` | Path to raw training data (contains `transforms_*.json`, images) |
| `--mask_path` | Path to `.pt` mask table |
| `--output_ply` | Where to save the harmonized `.ply` file |

### Harmonizer Selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--harmonizer` | `whitebox` | Backend: `whitebox`, `pctnet`, or `scene_b` |
| `--harmonizer_weights` | auto | Path to harmonizer weights |
| `--scene_b_path` | — | Path to Scene B directory (required for `scene_b` backend) |

### Optimization (IMPORTANT)

| Argument | Default | Recommended | Description |
|----------|---------|-------------|-------------|
| `--lr_dc` | `--lr` | `0.01` | Learning rate for DC (base color) |
| `--lr_rest` | `--lr` | `0.0` | Learning rate for higher SH bands. **Set to 0 to avoid rainbow artifacts.** |
| `--lr` | `1e-3` | — | Fallback LR if lr_dc/lr_rest not specified |
| `--num_iterations` | `500` | `500` | Optimization steps |
| `--reg_weight` | `0.01` | `0.01` | L2 regularization on delta_sh |
| `--use_lpips` | off | optional | Add perceptual loss |
| `--lpips_weight` | `0.1` | `0.1` | Weight for LPIPS loss |
| `--ssim_weight` | `0.0` | optional | Weight for SSIM loss (try `0.2`) |
| `--lambda_sh_band` | `0.0` | — | Band-weighted L2 on higher SH bands |

### Mask Preprocessing

| Argument | Default | Description |
|----------|---------|-------------|
| `--mask_feather` | `0` | Gaussian blur sigma on 2D mask (pixels) |
| `--mask_core_erode_px` | `0` | Erode mask to define high-weight core |
| `--mask_boundary_weight` | `0.25` | Weight for boundary band pixels |

### Workflow

| Argument | Default | Description |
|----------|---------|-------------|
| `--configs` | `None` | `.py` config file for model hyperparams |
| `--iteration` | `-1` | Checkpoint iteration (`-1` = latest) |
| `--ply_path` | `None` | Override which PLY to load |
| `--skip_precompute` | off | Load cached targets |
| `--precompute_only` | off | Only generate targets, skip optimization |
| `--targets_dir` | auto | Directory for cached targets |
| `--no_diffs` | off | Skip saving diff images |
| `--log_interval` | `50` | Print loss every N iterations |
| `--amplify` | `1.0` | Amplify harmonizer correction |
| `--sigma` | `2.0` | Temporal smoothing (whitebox only) |

## Tuning Guide

| Symptom | Fix |
|---------|-----|
| Rainbow / iridescent artifacts | **Set `--lr_rest 0.0`** (DC-only mode) |
| Object color barely changes | Increase `--lr_dc` (e.g. `0.05`) or `--num_iterations` (e.g. `1000`) |
| Object looks over-saturated | Increase `--reg_weight` (e.g. `0.05`) or decrease `--lr_dc` |
| Halo / glow around object | Try `--mask_core_erode_px 5 --mask_boundary_weight 0.1` |
| Streaks radiating from object | Set `--lr_rest 0.0`. Streaks = large Gaussians with modified higher SH bands |
| Temporal flickering (whitebox) | Increase `--sigma` (e.g. `4.0`) |
| Want sharper detail matching | Add `--use_lpips --lpips_weight 0.1` |

## Pipeline Architecture

```
pipeline/
├── run_harmonize.py              # main entry point
├── data_loading.py               # load scene, mask, harmonizer
├── harmonizer_base.py            # harmonizer backends (whitebox, pctnet, scene_b)
├── precompute_targets.py         # render views, generate target images
├── optimize_sh.py                # backprop delta_sh through rasterizer
├── inspect_targets.py            # visualize precomputed targets
├── export_mask_diagnostics.py    # debug 2D mask projection
├── apply_object_lighting_mismatch.py  # create test mismatch PLYs
├── single_frame_harmonize_test.py     # single-frame test harness
└── transsplat_harmonize_bridge.py     # TranSplat SH prior (experimental)
```

## Step-by-Step Example (Complete)

```bash
# 1. Train your 4DGS scene
python train_4dgs.py \
    -s ./data/your_scene/ \
    --port 6017 \
    --expname "your_scene" \
    --configs arguments/dnerf/joint_static_50k.py

# 2. Create a mask (see "Creating a mask" section above)
#    Save to: output/your_scene/segment_results/object_mask.pt

# 3. Run DC-only harmonization with whitebox harmonizer
python -m pipeline.run_harmonize \
    --model_path output/your_scene \
    --source_path data/your_scene \
    --mask_path output/your_scene/segment_results/object_mask.pt \
    --output_ply output/your_scene/point_cloud/iteration_45000/harmonized.ply \
    --configs arguments/dnerf/joint_static_50k.py \
    --harmonizer whitebox \
    --lr_dc 0.01 \
    --lr_rest 0.0 \
    --num_iterations 500 \
    --reg_weight 0.01

# 4. Render the harmonized result
python render_4dgs.py \
    --model_path output/your_scene \
    --skip_train \
    --configs arguments/dnerf/joint_static_50k.py \
    --ply_path output/your_scene/point_cloud/iteration_45000/harmonized.ply

# 5. Video is at: output/your_scene/video/ours_45000/video_rgb.mp4
```
