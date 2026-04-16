"""
Main entry point for the 4DGS harmonization pipeline.

Usage:
    cd sa4d
    python -m pipeline.run_harmonize \
        --model_path output/hypernerf/split-cookie \
        --source_path data/hypernerf/split-cookie \
        --mask_path output/hypernerf/split-cookie/segment_results/split-cookie.pt \
        --output_ply output/hypernerf/split-cookie/point_cloud/iteration_14000/harmonized.ply \
        --num_iterations 500 \
        --lr 1e-3

Pipeline steps:
    1. data_loading.load_scene()            -> loads 4DGS model + cameras
    2. data_loading.load_mask_table()        -> loads per-Gaussian .pt mask
    3. data_loading.load_harmonizer()        -> loads pretrained Harmonizer CNN
    4. precompute_targets.precompute_all_targets() -> renders + harmonizes all views
    5. optimize_sh.optimize()                -> backprops to SH coefficients
    6. optimize_sh.apply_delta_sh()          -> bakes delta into model
    7. optimize_sh.save_harmonized_ply()     -> writes final .ply
"""

import os
import sys
import argparse
import torch

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)


def main():
    parser = argparse.ArgumentParser(description="4DGS Harmonization Pipeline")

    # Required paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained 4DGS model (e.g. output/hypernerf/split-cookie)')
    parser.add_argument('--source_path', type=str, required=True,
                        help='Path to source data (e.g. data/hypernerf/split-cookie)')
    parser.add_argument('--mask_path', type=str, required=True,
                        help='Path to .pt mask table (e.g. segment_results/split-cookie.pt)')
    parser.add_argument('--output_ply', type=str, required=True,
                        help='Output path for harmonized .ply file')

    # Optional config
    parser.add_argument('--configs', type=str, default=None,
                        help='Path to .py config file for hyperparams')
    parser.add_argument('--iteration', type=int, default=-1,
                        help='4DGS checkpoint iteration to load (-1 = latest)')
    parser.add_argument('--ply_path', type=str, default=None,
                        help='Override which .ply file to load (e.g. a test_dark.ply)')
    parser.add_argument('--harmonizer_weights', type=str, default=None,
                        help='Path to harmonizer.pth (default: ~/Harmonizer/pretrained/harmonizer.pth)')

    # Optimization params
    parser.add_argument('--num_iterations', type=int, default=500,
                        help='Number of SH optimization steps')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for delta_sh optimization')
    parser.add_argument('--reg_weight', type=float, default=0.01,
                        help='L2 regularization weight on delta_sh')
    parser.add_argument('--use_lpips', action='store_true',
                        help='Add perceptual (LPIPS) loss')
    parser.add_argument('--lpips_weight', type=float, default=0.1,
                        help='Weight for LPIPS loss')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Temporal smoothing sigma for filter args')

    # Workflow control
    parser.add_argument('--skip_precompute', action='store_true',
                        help='Load cached targets instead of recomputing')
    parser.add_argument('--targets_dir', type=str, default=None,
                        help='Directory for cached targets (default: <model_path>/harmonize_cache)')
    parser.add_argument('--precompute_only', action='store_true',
                        help='Only precompute targets, skip SH optimization')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Print loss every N iterations')

    args = parser.parse_args()

    if args.targets_dir is None:
        args.targets_dir = os.path.join(args.model_path, 'harmonize_cache')

    # ----------------------------------------------------------------
    # Step 1: Load scene (4DGS model + cameras)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading 4DGS scene")
    print("=" * 60)
    from pipeline.data_loading import load_scene, load_mask_table, load_harmonizer

    gaussians, scene, pipe, background = load_scene(
        args.model_path, args.source_path,
        iteration=args.iteration, configs=args.configs)

    if args.ply_path is not None:
        print(f"Overriding PLY with: {args.ply_path}")
        gaussians.load_ply(args.ply_path)
        # If the override PLY changes gaussian count, ensure deformation table matches.
        # (Some render paths read `_deformation_table`; mismatch can cause shape errors.)
        if hasattr(gaussians, "_deformation_table"):
            n_xyz = gaussians._xyz.shape[0]
            if (not torch.is_tensor(gaussians._deformation_table)) or (gaussians._deformation_table.shape[0] != n_xyz):
                gaussians._deformation_table = torch.ones((n_xyz,), device="cuda", dtype=torch.bool)

    # ----------------------------------------------------------------
    # Step 2: Load mask table
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Loading mask table")
    print("=" * 60)
    mask_data = load_mask_table(args.mask_path)

    # Verify dimensions match
    n_gaussians_model = gaussians._xyz.shape[0]
    n_gaussians_mask = mask_data['mask_table'].shape[1]
    assert n_gaussians_model == n_gaussians_mask, (
        f"Gaussian count mismatch: model has {n_gaussians_model}, "
        f"mask has {n_gaussians_mask}")

    # ----------------------------------------------------------------
    # Step 3: Load Harmonizer
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Loading Harmonizer")
    print("=" * 60)
    harmonizer = load_harmonizer(args.harmonizer_weights)

    # ----------------------------------------------------------------
    # Step 4: Precompute targets
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Precomputing harmonization targets")
    print("=" * 60)
    from pipeline.precompute_targets import (
        precompute_all_targets, save_targets, load_targets)

    if args.skip_precompute:
        targets, composites, masks_2d = load_targets(args.targets_dir)
    else:
        targets, composites, masks_2d = precompute_all_targets(
            harmonizer, gaussians, scene, pipe, background,
            mask_data, sigma=args.sigma)
        save_targets(targets, composites, masks_2d, args.targets_dir)

    if args.precompute_only:
        print("\n[run_harmonize] Precompute-only mode. Done.")
        return

    # ----------------------------------------------------------------
    # Step 5: Optimize SH coefficients
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Optimizing SH coefficients")
    print("=" * 60)
    from pipeline.optimize_sh import optimize, apply_delta_sh, save_harmonized_ply

    delta_sh_dc, delta_sh_rest, object_mask = optimize(
        gaussians, scene, pipe, background, mask_data,
        targets, composites, masks_2d,
        num_iterations=args.num_iterations,
        lr=args.lr,
        reg_weight=args.reg_weight,
        use_lpips=args.use_lpips,
        lpips_weight=args.lpips_weight,
        log_interval=args.log_interval)

    # ----------------------------------------------------------------
    # Step 6: Bake delta into model and save
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Saving harmonized model")
    print("=" * 60)
    apply_delta_sh(gaussians, delta_sh_dc, delta_sh_rest, object_mask)
    save_harmonized_ply(gaussians, args.output_ply)

    # Also save the raw delta for inspection
    delta_path = os.path.join(os.path.dirname(args.output_ply), 'delta_sh.pt')
    torch.save({
        'delta_sh_dc': delta_sh_dc.detach().cpu(),
        'delta_sh_rest': delta_sh_rest.detach().cpu(),
        'object_mask': object_mask.cpu(),
    }, delta_path)
    print(f"[run_harmonize] Saved delta_sh to {delta_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Harmonized PLY: {args.output_ply}")
    print(f"  Delta SH:       {delta_path}")
    print(f"  Cached targets: {args.targets_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
