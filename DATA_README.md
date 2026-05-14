# Joint Cat-Breakdancer Scene: Lighting Harmonization Data Generation

## Overview

This pipeline generates training data for **lighting harmonization** using 3D/4D Gaussian Splatting.
We composite a cat (lit by a dark forest HDRI) into an outdoor breakdancer scene (lit by a suburban garden HDRI).
The goal: train on the incorrectly-lit composite (scene_A) and learn to relight the cat to match the outdoor
environment, using scene_B as ground truth supervision.

## Files

All rendering assets live in `mock_assets/`:

| File | Description |
|------|-------------|
| `mock_assets/joint_cat_breakdancer.blend` | Blender scene with breakdancer + cat positioned together |
| `mock_assets/blender_export_dnerf_v4.py` | Generates camera poses (orbit) + renders, outputs transforms JSONs |
| `mock_assets/render_joint_scenes.py` | Renders 3 passes per view: scene_B (GT), scene_A_bg, scene_A_cat, composites scene_A |
| `mock_assets/forest.hdr` | Indoor/forest HDRI (applied to cat in scene_A) |
| `mock_assets/suburban_garden_2k.hdr` | Outdoor HDRI (already in the blend file's world) |

## Blender Setup

Blender 5.1 is at:
- **EC2 (Linux):** `~/new_sa4d/sa4d/blender-5.1.1-linux-x64/blender`
- **macOS (local):** `/Applications/Blender.app/Contents/MacOS/Blender`

Set the `BLENDER` env var for convenience:
```bash
export BLENDER=~/new_sa4d/sa4d/blender-5.1.1-linux-x64/blender
```

Key objects in the blend file:
- `Actual_Cat` (MESH) — the cat
- `Beta_Surface` (MESH) — the breakdancer's visible body
- `Armature` (ARMATURE) — the breakdancer's skeleton (animation, frames 1-75)
- `Camera` (CAMERA)

## Output Structure

```
scene_joint_breakdance_cat/          (or dynamic_scene_joint_breakdance_cat/)
├── transforms_train.json            # Camera poses + time for training views
├── transforms_test.json             # Camera poses + time for test views
├── train/                           # Raw renders from export script
├── test/                            # Raw renders from export script
├── scene_B/                         # Ground truth: outdoor HDRI on everything
│   ├── train/r_0.png ... r_N.png
│   ├── test/r_0.png ... r_N.png
│   └── transforms_{train,test}.json
├── scene_A/                         # Composite: forest-lit cat over outdoor background
│   ├── train/r_0.png ... r_N.png
│   ├── test/r_0.png ... r_N.png
│   └── transforms_{train,test}.json
├── scene_A_bg/                      # Background only (cat hidden), outdoor HDRI
│   ├── train/
│   └── test/
└── scene_A_cat/                     # Cat only (transparent bg), forest HDRI
    ├── train/
    └── test/
```

### What each scene contains

- **scene_B** (ground truth): All objects rendered with outdoor HDRI. This is what the harmonized result should look like.
- **scene_A** (input/composite): The cat is lit by forest.hdr (wrong lighting), composited over the outdoor background. This is what the model trains on.
- **scene_A_bg**: Background with the cat removed. Used as intermediate for compositing.
- **scene_A_cat**: Cat only, rendered with forest HDRI on transparent background. Used as intermediate for compositing.

## Pipeline Commands

### Step 1: Generate camera poses + initial renders

**Static scene (3DGS):**
```bash
$BLENDER mock_assets/joint_cat_breakdancer.blend --background --python mock_assets/blender_export_dnerf_v4.py -- --output_dir ./scene_joint_breakdance_cat --target_objects Actual_Cat Beta_Surface --num_cameras 200 --num_test 40 --resolution 800 --static_frame 1 --radius 2
```

**Dynamic scene (4DGS):**
```bash
$BLENDER mock_assets/joint_cat_breakdancer.blend --background --python mock_assets/blender_export_dnerf_v4.py -- --output_dir ./dynamic_scene_joint_breakdance_cat --target_objects Actual_Cat Beta_Surface --num_cameras 200 --num_test 40 --resolution 800 --radius 2
```

### Step 2: Render the 3 passes (scene_A, scene_A_bg, scene_A_cat, scene_B)

**Static scene:**
```bash
$BLENDER mock_assets/joint_cat_breakdancer.blend --background --python mock_assets/render_joint_scenes.py -- --scene_dir scene_joint_breakdance_cat
```

**Dynamic scene:**
```bash
$BLENDER mock_assets/joint_cat_breakdancer.blend --background --python mock_assets/render_joint_scenes.py -- --scene_dir dynamic_scene_joint_breakdance_cat
```

### Debug: Verify HDRI switching

```bash
$BLENDER mock_assets/joint_cat_breakdancer.blend --background --python mock_assets/render_joint_scenes.py -- --debug-hdri --scene_dir scene_joint_breakdance_cat
```

This renders `debug_hdri/debug_outdoor.png` and `debug_hdri/debug_indoor.png` for visual comparison.

## Export Script Options (blender_export_dnerf_v4.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | `//dnerf_export` | Output directory |
| `--target_object` | None | Single object to orbit around |
| `--target_objects` | None | Multiple objects — orbits around their midpoint |
| `--target_point X Y Z` | `0 0 0` | Explicit orbit center |
| `--num_cameras` | 100 | Number of training views |
| `--num_test` | 20 | Number of test views |
| `--resolution` | 800 | Image resolution (square) |
| `--radius` | -1 (auto) | Orbit radius. Auto-detected from scene camera. |
| `--static_frame` | -1 (animate) | Lock to single frame. Omit for dynamic/4DGS. |
| `--num_rings` | 3 | Number of elevation rings |
| `--elevation_spread` | 15.0 | Degrees above/below camera elevation |
| `--samples` | 64 | Cycles render samples |
| `--seed` | 42 | Random seed for frame assignment |

## Render Script Options (render_joint_scenes.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--scene_dir` | `scene_joint_breakdance_cat` | Scene directory (reads transforms from here, writes outputs here) |
| `--debug-hdri` | off | Render two debug images to verify HDRI swap |

## EC2 Setup (GPU-accelerated rendering + SA4D training)

### Recommended instance
- **g5.xlarge** (NVIDIA A10G, 24GB VRAM) — best price/performance
- **g4dn.xlarge** (NVIDIA T4, 16GB VRAM) — cheaper alternative
- Use **Ubuntu 22.04** AMI with NVIDIA drivers pre-installed (Deep Learning AMI)

### Files to copy to EC2

**For rendering (if you want to re-render on GPU):**
```bash
# From your local machine — copy the mock_assets/ directory:
scp -i conceptgraph.pem -r mock_assets/ ubuntu@<EC2_IP>:~/new_sa4d/sa4d/
```

**For SA4D training (pre-rendered data):**
```bash
# Copy the rendered scene directories
scp -i conceptgraph.pem -r \
    scene_joint_breakdance_cat \
    ubuntu@<EC2_IP>:~/sa4d/data/

# For dynamic scene too:
scp -i conceptgraph.pem -r \
    dynamic_scene_joint_breakdance_cat \
    ubuntu@<EC2_IP>:~/sa4d/data/
```

**If you only want to train (skip re-rendering), you need these subdirectories:**
```
scene_joint_breakdance_cat/
├── scene_A/                    # Training input (wrong-lit composite)
│   ├── transforms_train.json
│   ├── transforms_test.json
│   ├── train/*.png
│   └── test/*.png
└── scene_B/                    # Ground truth (for evaluation)
    ├── transforms_train.json
    ├── transforms_test.json
    ├── train/*.png
    └── test/*.png
```

### Install Blender on EC2 (only if re-rendering)
```bash
# Download Blender 5.1 for Linux
wget https://mirror.clarkson.edu/blender/release/Blender5.1/blender-5.1.1-linux-x64.tar.xz
tar xf blender-5.1.1-linux-x64.tar.xz
export BLENDER=~/blender-5.1.1-linux-x64/blender

# Test GPU detection
$BLENDER --background --python-expr "
import bpy
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.get_devices()
for d in prefs.devices:
    print(f'{d.name}: {d.type}')
"

# Render commands (run from sa4d/ directory)
$BLENDER mock_assets/joint_cat_breakdancer.blend --background --python mock_assets/blender_export_dnerf_v4.py -- --output_dir ./scene_joint_breakdance_cat --target_objects Actual_Cat Beta_Surface --num_cameras 200 --num_test 40 --resolution 800 --static_frame 1 --radius 2
$BLENDER mock_assets/joint_cat_breakdancer.blend --background --python mock_assets/render_joint_scenes.py -- --scene_dir scene_joint_breakdance_cat
```

## SA4D Training (on EC2)

### Static (3DGS):
```bash
# Train on scene_A (wrong lighting)
python train_4dgs.py -s ./data/scene_joint_breakdance_cat/scene_A/ --port 6017 --expname "joint/scene_A" --configs arguments/hypernerf/default.py

# Render
python render_4dgs.py --model_path "output/joint/scene_A/" --skip_train --configs arguments/hypernerf/default.py

# Train segmentation
python train_ie.py -s ./data/scene_joint_breakdance_cat/scene_A/ -m ./output/joint/scene_A/ --configs arguments/hypernerf/default.py

# Render segmentation
python render_ie.py --model_path "output/joint/scene_A/" --skip_train --configs arguments/hypernerf/default.py

# Harmonize cat lighting
python -m pipeline.run_harmonize \
    --model_path output/joint/scene_A \
    --source_path data/scene_joint_breakdance_cat/scene_A \
    --mask_path output/joint/scene_A/segment_results/scene_A.pt \
    --output_ply output/joint/scene_A/point_cloud/iteration_14000/harmonized.ply
```

### Dynamic (4DGS):
Same commands but replace `scene_joint_breakdance_cat` with `dynamic_scene_joint_breakdance_cat`.

## Evaluation

Compare harmonized renders against scene_B (ground truth):
```bash
python metrics.py --model_path output/joint/scene_A/
```

## Fixes Applied (2026-05-02)

Several fixes were needed to make the pipeline work on EC2:

1. **Output path resolution**: Both `blender_export_dnerf_v4.py` and `render_joint_scenes.py` resolved `--output_dir` / `--scene_dir` relative to the blend file (via `bpy.path.abspath`). Fixed to use `os.path.abspath()` so paths resolve relative to the working directory, not the blend file location.

2. **HDRI remapping**: The blend file had a hardcoded HDRI path (`/home/Downloads/suburban_garden_2k.hdr`). Added `remap_missing_images()` to both scripts — it scans all images in the blend and remaps missing paths to look next to the blend file in `mock_assets/`.

3. **PIL dependency removed**: `render_joint_scenes.py` used PIL for compositing (scene_A = scene_A_bg + scene_A_cat). Blender's bundled Python doesn't include PIL. Replaced with numpy-based alpha compositing via `bpy.data.images`.

4. **GPU rendering**: Added OPTIX/CUDA/METAL auto-detection to both scripts so Cycles uses GPU. On EC2 (A10G), OPTIX is preferred.

## Generated Data (2026-05-02)

Both scenes were rendered on EC2 g5.xlarge (NVIDIA A10G) with:
- **200 train + 40 test** views each
- **radius 2** (close orbit around cat + breakdancer midpoint)
- **800x800** resolution, **64 samples** (Cycles)

| Scene | Directory | Size (compressed) |
|-------|-----------|-------------------|
| Static (3DGS) | `scene_joint_breakdance_cat/` | ~729 MB |
| Dynamic (4DGS) | `dynamic_scene_joint_breakdance_cat/` | ~729 MB |

Each contains: `scene_A/`, `scene_B/`, `scene_A_bg/`, `scene_A_cat/` with 200 train + 40 test PNGs and transforms JSONs.

Archives: `scene_joint_breakdance_cat.tar.gz`, `dynamic_scene_joint_breakdance_cat.tar.gz`

## Notes

- The `--` separator between Blender args and script args is **required**
- Static scenes use `--static_frame 1` to lock all views to frame 1 (no breakdancer animation)
- Dynamic scenes omit `--static_frame`, so each camera gets a random animation frame
- The transforms JSONs use D-NeRF format: `camera_angle_x`, `time` (0-1 normalized), `transform_matrix` (4x4 camera-to-world, NeRF/OpenGL convention)
- Radius 2 gives close-up framing of the cat + breakdancer; radius 8.4 was too far
- Recommended camera counts: 200 train / 40 test for both 3DGS and 4DGS
