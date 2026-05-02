# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Segment-Anything-in-4D (SA4D) — a research project for dynamic 3D scene segmentation combining 4D Gaussian Splatting with SAM (Segment Anything Model). Based on the paper "Segment Any 4D Gaussians". Supports HyperNeRF, DyNeRF, Blender, LLFF, and PanopticSports datasets.

## Environment Setup

```bash
conda create -n sa4d python=3.9
conda activate sa4d
# Requires PyTorch 2.0.1+cu118
conda install -c conda-forge colmap
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/diff-gaussian-rasterization_contrastive_f
pip install -e submodules/simple-knn
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

CUDA submodules (`diff-gaussian-rasterization`, `diff-gaussian-rasterization_contrastive_f`, `simple-knn`) must be installed in editable mode. They contain custom CUDA kernels for differentiable Gaussian rasterization and k-NN.

## Training Pipeline (4 stages)

```bash
# Stage 1: Train 4D Gaussian scene reconstruction
python train_4dgs.py -s ./data/hypernerf/split-cookie/ --port 6017 --expname "hypernerf/split-cookie" --configs arguments/hypernerf/default.py

# Stage 2: Render scene outputs
python render_4dgs.py --model_path "output/hypernerf/split-cookie/" --skip_train --skip_test --configs arguments/hypernerf/default.py

# Stage 3: Train interactive editing (feature extraction + segmentation)
python train_ie.py -s ./data/hypernerf/split-cookie/ -m ./output/hypernerf/split-cookie/ --configs arguments/hypernerf/default.py

# Stage 4: Render segmentation
python render_ie.py --model_path "output/hypernerf/split-cookie/" --skip_train --skip_test --configs arguments/hypernerf/default.py
```

More examples in `command.sh`.

## Data Preprocessing

```bash
# HyperNeRF: COLMAP dense reconstruction + downsample
bash colmap.sh data/hypernerf/broom2 hypernerf
python scripts/downsample_point.py <fused.ply> <output_downsample.ply>

# DyNeRF: extract frames, COLMAP, downsample
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
bash colmap.sh data/dynerf/cut_roasted_beef llff
python scripts/downsample_point.py <fused.ply> <output_downsample.ply>

# Generate pseudo-labels (DEVA object tracking)
bash prepare_pseudo_label.sh ./data/hypernerf/broom2/ 0
```

## Evaluation

```bash
python metrics.py --model_path output/hypernerf/split-cookie/
```

Computes PSNR, SSIM, LPIPS (vgg/alex), MS-SSIM per-view and aggregated.

## Harmonization Pipeline

Lighting harmonization for composited objects (in `pipeline/`):

```bash
python -m pipeline.run_harmonize \
    --model_path output/hypernerf/torchocolate \
    --source_path data/hypernerf/torchocolate \
    --mask_path output/hypernerf/torchocolate/segment_results/torchocolate.pt \
    --output_ply output/hypernerf/torchocolate/point_cloud/iteration_14000/harmonized.ply
```

See `pipeline/README.md` for details.

## Architecture

### Core modules

- **`scene/`** — Scene loading and Gaussian models
  - `feature_gaussian_model.py` — Main model: 3D Gaussians with SH features, deformation network, and SegNet for segmentation (256-class classifier)
  - `gaussian_model.py` — Static-only Gaussian model
  - `deformation.py` — Time-conditioned deformation network using HexPlane 4D representation
  - `dataset_readers.py` — Auto-detects and loads COLMAP, Blender, DyNeRF, HyperNeRF, PanopticSports formats
  - `hyper_loader.py` — HyperNeRF-specific data loading

- **`gaussian_renderer/`** — Differentiable rendering: deformation → projection → CUDA rasterization → alpha blending. Supports RGB, feature, and mask rendering modes.

- **`arguments/`** — Per-dataset, per-scene config files (`arguments/hypernerf/default.py`, `arguments/dynerf/cut_roasted_beef.py`, etc.). Defines model architecture params (HexPlane resolution, MLP depth/width), optimization params (iterations, learning rates, densification schedule).

- **`pipeline/`** — Harmonization: `data_loading.py` → `precompute_targets.py` → `optimize_sh.py`. Optimizes SH residuals to match harmonizer CNN predictions.

- **`utils/`** — Losses (L1, SSIM, LPIPS, classification), SH encoding, camera projection, point cloud operations, segmentation utilities.

### Training loop pattern (train_4dgs.py)

Two-stage: coarse (no deformation) → fine (with deformation network). Each iteration: sample viewpoint → render via differentiable rasterizer → L1+SSIM loss → backprop → densify/prune Gaussians based on gradient accumulation.

### Key data flow

Scene config (`arguments/`) → Dataset reader (`scene/dataset_readers.py`) → Scene class (`scene/__init__.py`) → GaussianModel + Deformation → Renderer (`gaussian_renderer/`) → Loss computation → Optimizer update.

### Parallel code paths

`utils_static/` and `static_scene/` mirror `utils/` and `scene/` for static-only scenes. These are separate copies, not shared code.

## Key Notebooks

- `composite.ipynb` / `composite_cookie_choc.ipynb` — Multi-scene composition
- `delete.ipynb` — Object removal
- `demo_hypernerf.ipynb` / `demo_dynerf.ipynb` — Interactive SAM segmentation
- `generate_mask_tables.ipynb` — Per-Gaussian mask table creation
- `recolor.ipynb` — Appearance adjustment
