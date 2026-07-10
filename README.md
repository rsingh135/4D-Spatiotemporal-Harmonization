# 4D Spatiotemporal Harmonization

**[Spatiotemporal Relighting Presentation](https://docs.google.com/presentation/d/1sYxFy22wpCMfFpZAxKPnyTr5TvHjidlR/edit?usp=sharing&ouid=110820616071956482790&rtpof=true&sd=true)**

This project extends **[SA4D (Segment-Anything-in-4D)](https://github.com/Marine318/sa4d)** — code organized around the paper *Segment Any 4D Gaussians* — with a **relighting harmonization** pipeline. The goal is to composite objects from 4D Gaussian scenes into new environments and **adjust the object’s appearance** (via optimized spherical harmonics) so it matches the host scene’s lighting, using **object masks** derived from the SA4D segmentation workflow.

## What this fork adds

- **`pipeline/`** — Main harmonization code: render training views, run a pretrained image harmonizer (white-box [Harmonizer](https://github.com/ZHKKKe/Harmonizer) or **PCT-Net**), then **optimize SH residuals** on masked Gaussians through the differentiable rasterizer. Entry point: `python -m pipeline.run_harmonize ...`
- **Synthetic data generation** — Blender-based **joint cat / breakdancer** scenes with deliberate lighting mismatch (`scene_A` vs ground-truth `scene_B`) for training and evaluation. Documented in **[`DATA_README.md`](DATA_README.md)** (camera export, multi-pass renders, EC2 notes, dataset layout).
- **Diagnostics and stress tools** — e.g. mask overlays, baked lighting mismatch, optional learned shadow layer. See **[`pipeline/README.md`](pipeline/README.md)**.

For harmonizer arguments, caching, rendering the output PLY, and tuning, read **`pipeline/README.md`**. For Blender assets, paths, and the static vs dynamic dataset recipe, read **`DATA_README.md`**.

## Environment setup

```bash
git submodule update --init --recursive

conda create -n sa4d python=3.9
conda activate sa4d
# PyTorch: e.g. pytorch==2.0.1+cu118 (match your CUDA)
conda install -c conda-forge colmap

pip install -r requirement.txt
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/diff-gaussian-rasterization_contrastive_f
pip install -e submodules/simple-knn

pip install "git+https://github.com/facebookresearch/pytorch3d.git"

cd Tracking-Anything-with-DEVA
pip install -r requirements.txt
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
cd Grounded-Segment-Anything
pip install -e .
cd ../..
```

**Harmonization extras** (from repo root, inside the same env):

```bash
pip install kornia plyfile lpips   # lpips optional unless you pass --use_lpips
```

Pretrained **Harmonizer** weights are expected by default under `~/Harmonizer/pretrained/harmonizer.pth` (overridable via `--harmonizer_weights`). PCT-Net uses `--harmonizer pctnet` with its own default checkpoint path; see `pipeline/run_harmonize.py`.

## Training and rendering (SA4D baseline)

### Data preprocessing

**HyperNeRF-style** (example `virg/broom`): COLMAP dense cloud, then downsample.

```bash
bash colmap.sh data/hypernerf/broom2 hypernerf
python scripts/downsample_point.py \
  data/hypernerf/virg/broom2/colmap/dense/workspace/fused.ply \
  data/hypernerf/virg/broom2/points3D_downsample2.ply
```

**DyNeRF-style**:

```bash
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
bash colmap.sh data/dynerf/cut_roasted_beef llff
python scripts/downsample_point.py \
  data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply \
  data/dynerf/cut_roasted_beef/points3D_downsample2.ply
```

### Pseudo-labels (DEVA)

```bash
bash prepare_pseudo_label.sh ./data/hypernerf/broom2/ 0
bash prepare_pseudo_label.sh ./data/dynerf/cut_roasted_beef 0
```

### Typical 4DGS + interactive editing stack

```bash
# 1) Train 4D Gaussians
python train_4dgs.py -s ./data/hypernerf/split-cookie/ --port 6017 \
  --expname "hypernerf/split-cookie" --configs arguments/hypernerf/default.py

# 2) Render reconstruction
python render_4dgs.py --model_path "output/hypernerf/split-cookie/" \
  --skip_train --skip_test --configs arguments/hypernerf/default.py

# 3) Train interactive editing / segmentation features
python train_ie.py -s ./data/hypernerf/split-cookie/ \
  -m ./output/hypernerf/split-cookie/ --configs arguments/hypernerf/default.py

# 4) Render segmentation
python render_ie.py --model_path "output/hypernerf/split-cookie/" \
  --skip_train --skip_test --configs arguments/hypernerf/default.py
```

More dataset-specific commands are in **`command.sh`**. Metrics: **`python metrics.py --model_path <output_dir>`**.

## Harmonization (after you have a mask)

You need a trained model under `output/...`, matching **`--source_path`** data, and a **per-Gaussian mask table** `.pt` (from the segmentation / notebook workflow, under something like `segment_results/`).

```bash
python -m pipeline.run_harmonize \
  --model_path output/hypernerf/torchocolate \
  --source_path data/hypernerf/torchocolate \
  --mask_path output/hypernerf/torchocolate/segment_results/torchocolate.pt \
  --output_ply output/hypernerf/torchocolate/point_cloud/iteration_14000/harmonized.ply
```

Use **`--configs arguments/hypernerf/default.py`** (or your scene’s config) when hyperparameters must match training. Optional: **`--harmonizer pctnet`**, **`--ply_path`** for a composite PLY, **`--composite`** for foreground-only deformation behavior — see **`pipeline/README.md`**.

## Notebooks and scripts you may run

| Artifact | Role |
| -------- | ---- |
| **`delete.ipynb`** | Remove an object; object IDs are documented in the notebook |
| **`composite.ipynb`**, **`composite_*` notebooks** | Combine multiple Gaussian scenes in shared space |
| **`demo.ipynb`** | Extract / display object IDs |
| **`demo_dynerf.ipynb`**, **`demo_hypernerf.ipynb`** | SAM-based interactive segmentation; refine with losses |
| **`generate_mask_tables.ipynb`** | Build per-Gaussian mask tables for harmonization |
| **`recolor.ipynb`** | Appearance adjustments |
| **`scripts/`** | COLMAP / DyNeRF / Blender export helpers (e.g. `preprocess_dynerf.py`, `downsample_point.py`) |
| **`mock_assets/`** + **`DATA_README.md`** | Blender orbit export and multi-pass rendering for harmonization datasets |

## Repository layout (high level)

- **`train_4dgs.py`**, **`render_4dgs.py`** — 4D Gaussian training and rendering  
- **`train_ie.py`**, **`render_ie.py`** — Interactive editing and segmentation rendering  
- **`scene/`**, **`gaussian_renderer/`**, **`utils/`** — Data loading, model, rasterization, losses  
- **`arguments/`** — Per-dataset Python configs  
- **`pipeline/`** — Harmonization (this fork’s focus); start at **`pipeline/run_harmonize.py`**  
- **`Tracking-Anything-with-DEVA/`** — Tracking / labeling stack (submodule workflow)  

## Citation and upstream

If you use the underlying SA4D method, cite the *Segment Any 4D Gaussians* work and consider citing the original SA4D repository: [github.com/Marine318/sa4d](https://github.com/Marine318/sa4d).

For the image harmonizer backbone, cite the corresponding Harmonizer or PCT-Net paper as appropriate; references are linked from **`pipeline/README.md`**.
