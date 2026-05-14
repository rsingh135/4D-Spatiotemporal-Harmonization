"""
Main entry point for the 4DGS harmonization pipeline.

Usage:
    cd sa4d

    # With original whitebox harmonizer:
    python -m pipeline.run_harmonize \
        --model_path output/hypernerf/split-cookie \
        --source_path data/hypernerf/split-cookie \
        --mask_path output/hypernerf/split-cookie/segment_results/split-cookie.pt \
        --output_ply output/hypernerf/split-cookie/point_cloud/iteration_14000/harmonized.ply

    # With PCT-Net harmonizer:
    python -m pipeline.run_harmonize \
        --harmonizer pctnet \
        --model_path ... --source_path ... --mask_path ... --output_ply ...

Pipeline steps:
    1. data_loading.load_scene()            -> loads 4DGS model + cameras
    2. data_loading.load_mask_table()        -> loads per-Gaussian .pt mask
    3. harmonizer_base.create_harmonizer()   -> loads harmonizer (whitebox or pctnet)
    4. precompute_targets.precompute_all_targets() -> renders + harmonizes all views
    5. optimize_sh.optimize()                -> backprops to SH coefficients
    6. optimize_sh.apply_delta_sh()          -> bakes delta into model
    7. optimize_sh.save_harmonized_ply()     -> writes final .ply

Mask quality (segmentation): Harmonization and optional TranSplat SH priors
operate on Gaussians selected by ``--mask_path``. Incorrect masks leak
background into the object (and vice versa); TranSplat-style losses do not
fix segmentation—improve ``mask_table`` / ``object_mask`` first.

TranSplat bridge (optional): See ``pipeline/transsplat_harmonize_bridge.py``.
Use HDR env maps + Phase-2 reference SH to warm-start or regularize ΔSH
alongside the image harmonizer. Stock ``gaussian_renderer`` does not expose
depth for on-the-fly Phase-1 visibility; pass ``--transsplat_vlm_pt`` or
``--transsplat_skip_visibility``.
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
                        help='Per-Gaussian .pt mask (segment_results/...). Quality is critical: '
                             'TranSplat SH priors do not repair bad segmentation.')
    parser.add_argument('--output_ply', type=str, required=True,
                        help='Output path for harmonized .ply file')

    # Optional config
    parser.add_argument('--configs', type=str, default=None,
                        help='Path to .py config file for hyperparams')
    parser.add_argument('--iteration', type=int, default=-1,
                        help='4DGS checkpoint iteration to load (-1 = latest)')
    parser.add_argument('--ply_path', type=str, default=None,
                        help='Override which .ply file to load (e.g. a composite PLY)')
    parser.add_argument('--composite', action='store_true',
                        help='Composite mode: foreground gaussians (from mask) skip deformation')

    # Harmonizer selection
    parser.add_argument('--harmonizer', type=str, default='whitebox',
                        choices=['whitebox', 'pctnet', 'scene_b'],
                        help='Harmonizer backend: "whitebox", "pctnet", or "scene_b" (use GT images)')
    parser.add_argument('--harmonizer_weights', type=str, default=None,
                        help='Path to harmonizer weights (default: auto per backend)')
    parser.add_argument('--scene_b_path', type=str, default=None,
                        help='Path to scene_B directory (required when --harmonizer scene_b)')

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
    parser.add_argument('--ssim_weight', type=float, default=0.0,
                        help='Weight for SSIM loss (0 = off, try 0.2)')
    parser.add_argument('--lr_dc', type=float, default=None,
                        help='Optional LR for delta_sh_dc (defaults to --lr)')
    parser.add_argument('--lr_rest', type=float, default=None,
                        help='Optional LR for delta_sh_rest (defaults to --lr)')

    # Optional learned shadow plate (experimental)
    parser.add_argument('--shadow_mode', type=str, default='off', choices=['off', 'learned'],
                        help='If learned, optimize extra shadow Gaussians during SH harmonization')
    parser.add_argument('--shadow_n', type=int, default=2048,
                        help='Number of shadow anchor Gaussians')
    parser.add_argument('--shadow_lr', type=float, default=5e-3,
                        help='Adam LR for shadow parameters')
    parser.add_argument('--shadow_reg_weight', type=float, default=0.01,
                        help='L2 regularization on shadow SH')
    parser.add_argument('--shadow_outside_weight', type=float, default=0.05,
                        help='Penalty for darkening pixels outside the object mask')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Temporal smoothing sigma for filter args (whitebox only)')

    # Mask preprocessing (helps boundary halos)
    parser.add_argument('--mask_feather', type=float, default=0.0,
                        help='Feather (Gaussian blur) the rendered 2D mask by this sigma in pixels (0 = off)')
    parser.add_argument('--mask_core_erode_px', type=int, default=0,
                        help='If >0, erode the binary (thr=0.5) mask by this many pixels to define a high-weight core region.')
    parser.add_argument('--mask_boundary_weight', type=float, default=0.25,
                        help='Weight multiplier for the boundary band (outside eroded core but inside mask).')
    parser.add_argument('--mask_weight_power', type=float, default=1.0,
                        help='Optional exponent applied to final per-pixel mask weights (>=1 emphasizes confident interior).')

    # Target amplification
    parser.add_argument('--amplify', type=float, default=1.0,
                        help='Amplify harmonizer correction: target = comp + amplify*(harmonized - comp). '
                             '1.0 = normal, 3.0 = 3x stronger correction, etc.')

    # Workflow control
    parser.add_argument('--skip_precompute', action='store_true',
                        help='Load cached targets instead of recomputing')
    parser.add_argument('--targets_dir', type=str, default=None,
                        help='Directory for cached targets (default: <model_path>/harmonize_cache)')
    parser.add_argument('--diff_dir', type=str, default=None,
                        help='Save per-view diff images here (default: <model_path>/harmonize_diffs)')
    parser.add_argument('--no_diffs', action='store_true',
                        help='Disable saving per-view diff images during precompute (saves disk).')
    parser.add_argument('--precompute_only', action='store_true',
                        help='Only precompute targets, skip SH optimization')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Print loss every N iterations')

    # TranSplat ↔ harmonize bridge (optional SH lighting prior)
    parser.add_argument('--transsplat_hdr_source', type=str, default=None,
                        help='Source HDR/LDR env map path for TranSplat L_lm (use with --transsplat_hdr_target)')
    parser.add_argument('--transsplat_hdr_target', type=str, default=None,
                        help='Target HDR/LDR env map path for TranSplat L_lm')
    parser.add_argument('--transsplat_vlm_pt', type=str, default=None,
                        help='Precomputed V_lm .pt [N_object, K] from TranSplat Phase 1 (recommended)')
    parser.add_argument('--transsplat_skip_visibility', action='store_true',
                        help='Use unobstructed V_lm proxy instead of Phase 1 (approximate; no depth map)')
    parser.add_argument('--transsplat_sh_loss_weight', type=float, default=0.0,
                        help='Weight for MSE between (base+ΔSH) and TranSplat Phase-2 reference SH on object')
    parser.add_argument('--transsplat_warm_start', action='store_true',
                        help='Initialize ΔSH so base+Δ matches TranSplat Phase-2 reference before pixel optimization')
    parser.add_argument('--transsplat_floor_alpha', type=float, default=0.05,
                        help='TranSplat Phase-2 diffuse ratio floor (see radiance_transfer_TranSplat)')
    parser.add_argument('--transsplat_tau_max', type=float, default=3.0,
                        help='TranSplat Phase-2 diffuse ratio clamp tau_max')
    parser.add_argument('--transsplat_num_samples', type=int, default=1200,
                        help='Hemisphere samples for env map → SH projection')
    parser.add_argument('--lambda_sh_band', type=float, default=0.0,
                        help='TranSplat-style band-weighted L2 on delta_sh_rest (higher SH bands penalized more)')

    args = parser.parse_args()

    # Derive a run name from the PLY filename for organizing outputs
    if args.ply_path is not None:
        run_name = os.path.splitext(os.path.basename(args.ply_path))[0]
    else:
        run_name = 'default'

    if args.targets_dir is None:
        args.targets_dir = os.path.join(args.model_path, 'harmonize_cache', run_name)
    if args.diff_dir is None:
        args.diff_dir = os.path.join(args.model_path, 'harmonize_diffs', run_name)
    if args.no_diffs:
        args.diff_dir = None

    # ----------------------------------------------------------------
    # Step 1: Load scene (4DGS model + cameras)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading 4DGS scene")
    print("=" * 60)
    from pipeline.data_loading import load_scene, load_mask_table

    gaussians, scene, pipe, background = load_scene(
        args.model_path, args.source_path,
        iteration=args.iteration, configs=args.configs)

    if args.ply_path is not None:
        print(f"Overriding PLY with: {args.ply_path}")
        gaussians.load_ply(args.ply_path)
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

    n_gaussians_model = gaussians._xyz.shape[0]
    n_gaussians_mask = mask_data['mask_table'].shape[1]
    assert n_gaussians_model == n_gaussians_mask, (
        f"Gaussian count mismatch: model has {n_gaussians_model}, "
        f"mask has {n_gaussians_mask}")

    if args.composite:
        from pipeline.data_loading import get_object_mask
        fg_mask = get_object_mask(mask_data)
        deform_table = ~fg_mask
        gaussians._deformation_table = deform_table
        print(f"[run_harmonize] Composite mode: {deform_table.sum().item()} deformed, "
              f"{(~deform_table).sum().item()} static (foreground)")

    # ----------------------------------------------------------------
    # Step 3: Load Harmonizer
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 3: Loading Harmonizer (backend={args.harmonizer})")
    print("=" * 60)
    from pipeline.harmonizer_base import create_harmonizer
    extra_kwargs = {}
    if args.harmonizer == 'scene_b':
        if not args.scene_b_path:
            raise ValueError("--scene_b_path is required when --harmonizer scene_b")
        extra_kwargs['scene_b_dir'] = args.scene_b_path
    harmonizer = create_harmonizer(args.harmonizer, weights_path=args.harmonizer_weights,
                                   **extra_kwargs)

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
            mask_data, sigma=args.sigma, diff_dir=args.diff_dir,
            mask_feather_sigma=args.mask_feather,
            amplify=args.amplify)
        try:
            save_targets(targets, composites, masks_2d, args.targets_dir)
        except Exception as e:
            # Sweeps can easily fill disks; caching is optional (targets are already in memory).
            print(f"[run_harmonize] WARNING: could not save cached targets (continuing): {e}")

    if args.precompute_only:
        print("\n[run_harmonize] Precompute-only mode. Done.")
        return

    # ----------------------------------------------------------------
    # Step 5: Optimize SH coefficients
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Optimizing SH coefficients")
    print("=" * 60)
    from pipeline.optimize_sh import optimize, apply_delta_sh, bake_shadow_pack_into_gaussians, save_harmonized_ply

    if (args.transsplat_hdr_source or args.transsplat_hdr_target) and not (
        args.transsplat_hdr_source and args.transsplat_hdr_target
    ):
        raise ValueError('Provide both --transsplat_hdr_source and --transsplat_hdr_target, or neither.')

    delta_sh_dc, delta_sh_rest, object_mask, losses, shadow_pack = optimize(
        gaussians, scene, pipe, background, mask_data,
        targets, composites, masks_2d,
        num_iterations=args.num_iterations,
        lr=args.lr,
        lr_dc=args.lr_dc,
        lr_rest=args.lr_rest,
        reg_weight=args.reg_weight,
        use_lpips=args.use_lpips,
        lpips_weight=args.lpips_weight,
        ssim_weight=args.ssim_weight,
        log_interval=args.log_interval,
        shadow_mode=args.shadow_mode,
        shadow_n=args.shadow_n,
        shadow_lr=args.shadow_lr,
        shadow_reg_weight=args.shadow_reg_weight,
        shadow_outside_weight=args.shadow_outside_weight,
        mask_core_erode_px=args.mask_core_erode_px,
        mask_boundary_weight=args.mask_boundary_weight,
        mask_weight_power=args.mask_weight_power,
        transsplat_hdr_source=args.transsplat_hdr_source,
        transsplat_hdr_target=args.transsplat_hdr_target,
        transsplat_vlm_pt=args.transsplat_vlm_pt,
        transsplat_skip_visibility=args.transsplat_skip_visibility,
        transsplat_sh_loss_weight=args.transsplat_sh_loss_weight,
        transsplat_warm_start=args.transsplat_warm_start,
        transsplat_floor_alpha=args.transsplat_floor_alpha,
        transsplat_tau_max=args.transsplat_tau_max,
        transsplat_num_samples=args.transsplat_num_samples,
        lambda_sh_band=args.lambda_sh_band)

    # ----------------------------------------------------------------
    # Step 6: Bake delta into model and save
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Saving harmonized model")
    print("=" * 60)
    apply_delta_sh(gaussians, delta_sh_dc, delta_sh_rest, object_mask)
    bake_shadow_pack_into_gaussians(gaussians, shadow_pack)
    save_harmonized_ply(gaussians, args.output_ply)

    out_dir = os.path.dirname(args.output_ply)

    delta_path = os.path.join(out_dir, 'delta_sh.pt')
    torch.save({
        'delta_sh_dc': delta_sh_dc.detach().cpu(),
        'delta_sh_rest': delta_sh_rest.detach().cpu(),
        'object_mask': object_mask.cpu(),
        'shadow_pack': None if shadow_pack is None else {k: v.detach().cpu() for k, v in shadow_pack.items()},
    }, delta_path)
    print(f"[run_harmonize] Saved delta_sh to {delta_path}")

    # Save loss curve
    loss_path = os.path.join(out_dir, f'{run_name}_losses.pt')
    torch.save({'losses': losses, 'args': vars(args)}, loss_path)
    print(f"[run_harmonize] Saved losses to {loss_path}")

    # Plot loss curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        losses_arr = np.array(losses)
        # Smoothed version (rolling average)
        window = min(50, len(losses_arr) // 5) if len(losses_arr) > 10 else 1
        smoothed = np.convolve(losses_arr, np.ones(window)/window, mode='valid')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses_arr, alpha=0.3, color='blue', label='Per-step loss')
        ax.plot(np.arange(window-1, len(losses_arr)), smoothed,
                color='blue', linewidth=2, label=f'Smoothed (window={window})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'SH Optimization — {run_name}\n'
                     f'lr={args.lr}, reg={args.reg_weight}, amplify={args.amplify}, '
                     f'harmonizer={args.harmonizer}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(out_dir, f'{run_name}_loss_curve.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[run_harmonize] Saved loss curve to {plot_path}")
    except Exception as e:
        print(f"[run_harmonize] Could not save loss plot: {e}")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Harmonized PLY: {args.output_ply}")
    print(f"  Delta SH:       {delta_path}")
    print(f"  Cached targets: {args.targets_dir}")
    print(f"  Diff images:    {args.diff_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
