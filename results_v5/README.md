# Results v5: Breakdancer Relighting (Dynamic 4DGS)

## Scene Setup

- **Scene A**: Breakdancer lit by `forest.hdr` (dark, warm lighting), cat + background lit by `suburban_garden_2k.hdr` (bright, outdoor)
- **Scene B**: Everything lit by `suburban_garden_2k.hdr` (ground truth)
- **Task**: Relight the breakdancer from dark forest lighting to bright outdoor lighting by optimizing SH coefficients
- **Breakdancer mask**: 8,584 / 221,281 Gaussians (3.88%), created via spatial proximity to known breakdancer position

## Training

- Config: `arguments/dnerf/joint_dynamic_50k.py` (50K iterations, deformation enabled)
- Data: `data_v5_dynamic/scene_A/` and `data_v5_dynamic/scene_B/`
- 0.3x animation speed (75 Blender frames), 300 train + 60 test cameras
- Ground holdout fix applied (snow plane occludes object bases correctly)

## Harmonization Experiments

All experiments use `--harmonizer scene_b` (Scene B images as optimization targets) with 500 SH optimization iterations. The loss is computed only inside the 2D breakdancer mask.

### Root cause of rainbow artifacts

The original v5 harmonization (`harmonized_scene_b`) used equal learning rates for all SH bands. Analysis of `delta_sh` showed:

| SH Band | Mean |delta| | Max |delta| |
|---------|---------------|--------------|
| DC (l=0) | 0.0315 | 1.28 |
| l=1 | 0.0904 | 2.05 |
| l=2 | 0.0957 | 2.24 |
| l=3 | 0.1045 | 2.26 |

Higher-order bands had **3x larger deltas than DC**. Since higher SH bands encode view-dependent color, this created iridescent/rainbow artifacts when viewed from different angles. A relighting (dark-to-bright) should primarily be a DC (uniform brightness) shift.

### Configs tested

| Config | CLI flags | Loss | Description |
|--------|-----------|------|-------------|
| `scene_A` | — | — | Original, no harmonization |
| `harmonized_scene_b` | `--lr 0.01` | 0.044 | All SH bands, equal LR (rainbow artifacts) |
| `harmonized_dc_only` | `--lr_dc 0.01 --lr_rest 0.0` | 0.050 | Only optimize DC band, freeze all higher bands |
| `harmonized_sep_lr` | `--lr_dc 0.01 --lr_rest 0.0001` | 0.051 | DC learns 100x faster than higher bands |
| `harmonized_band_reg` | `--lr 0.01 --lambda_sh_band 0.1` | 0.044 | All bands active, but higher bands penalized more |
| `scene_B` | — | — | Ground truth (target) |

### Results

**DC-only is the clear winner.** The breakdancer transitions from dark red/brown to lighter pink without any rainbow/iridescent artifacts. It preserves shape and opacity because only the base color changes, not view-dependent terms.

The remaining gap vs ground truth (loss 0.050 vs 0.0) is because DC-only can only apply a uniform color shift per Gaussian — it cannot capture subtle view-dependent lighting differences. This is an acceptable trade-off vs the rainbow disaster from full SH optimization.

### Metrics (from original v5 run)

| Method | PSNR (dB) | SSIM | LPIPS |
|--------|-----------|------|-------|
| Original A | 18.566 | 0.8840 | 0.3369 |
| Harmonized (all bands) | 19.946 | 0.8632 | 0.3805 |

PSNR improved +1.38 dB but SSIM/LPIPS degraded due to rainbow artifacts. DC-only should improve all three metrics.

## File Structure

```
results_v5/
├── dynamic/
│   ├── scene_A/                      # Original scene A (dark breakdancer)
│   │   ├── scene_point_cloud.ply
│   │   └── video_rgb.mp4
│   ├── scene_B/                      # Ground truth (bright breakdancer)
│   │   ├── scene_point_cloud.ply
│   │   └── video_rgb.mp4
│   ├── scene_A_harmonized/           # Original v5: all SH bands (rainbow)
│   │   ├── harmonized_scene_b.ply
│   │   └── video_rgb.mp4
│   ├── harmonized_dc_only/           # DC-only (best result)
│   │   └── video_rgb.mp4
│   ├── harmonized_sep_lr/            # Separate LR (DC fast, rest slow)
│   │   └── video_rgb.mp4
│   ├── harmonized_band_reg/          # Band regularization
│   │   └── video_rgb.mp4
│   ├── harmonized_scene_b/           # Same as scene_A_harmonized
│   │   └── video_rgb.mp4
│   ├── comparisons/                  # Side-by-side frames (A | harmonized | B)
│   │   └── frame_*.png
│   ├── loss_comparisons/             # Per-config single-view renders
│   │   ├── original.png
│   │   ├── harmonized_v5.png
│   │   ├── dc_only.png
│   │   ├── sep_lr.png
│   │   ├── band_reg.png
│   │   └── scene_b_gt.png
│   └── metrics.txt
└── README.md
```

## Pipeline Changes Made

### 1. SceneBHarmonizer (`pipeline/harmonizer_base.py`)
Added `--harmonizer scene_b` mode that loads Scene B ground-truth images directly as optimization targets instead of running a neural harmonizer (whitebox/PCT-Net). This gives a much stronger supervision signal.

### 2. Binary mask fix (`pipeline/precompute_targets.py`)
The rendered 2D mask values were very low (max ~0.30) for semi-transparent Gaussians, but all code thresholded at 0.5 giving 0% coverage. Fixed by binarizing at threshold 0.01 after rendering.

### 3. SSIM loss (`pipeline/optimize_sh.py`)
Added `--ssim_weight` flag for structural similarity loss alongside L1 and LPIPS.

### 4. Ground holdout fix (`mock_assets/render_joint_scenes.py`)
The cat-only render pass now keeps the snow ground plane visible with a holdout material. This prevents the cat's base/pedestal from showing through the snow in Scene A composites, matching Scene B where the base is naturally occluded.

### 5. Blender export enhancements (`mock_assets/blender_export_dnerf_v4.py`)
- `--time_scale`: Slow down animation (0.3 = use first 30% of frames for denser temporal coverage)
- `--json_only`: Skip rendering, only write transforms JSON (fast iteration on camera poses)

### Key Findings

1. **SH band control is critical for relighting.** Optimizing all SH bands equally causes rainbow/iridescent artifacts. DC-only or heavily penalized higher bands produce clean results.
2. **Loss function matters less than SH band strategy.** L1, LPIPS, SSIM all produce similar artifacts when higher SH bands are unconstrained.
3. **Dynamic 4DGS masks are tricky.** Gaussians near the object in canonical space may deform elsewhere at different timesteps. KNN-based spatial masks work but need careful threshold tuning.
4. **Animation speed directly affects reconstruction quality.** 0.3x speed with the same number of cameras gives ~5 views/timestep instead of ~3, producing sharper Gaussians.
