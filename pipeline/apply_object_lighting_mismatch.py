"""
Bake a lighting mismatch into ONLY the segmented object Gaussians (3D-relevant stress test).

This modifies SH coefficients for Gaussians selected by union(mask_table) and writes a new .ply.
The mask .pt is unchanged (same Gaussian ordering/count), so it remains compatible.

Typical usage:
  cd /home/ubuntu/new_sa4d/sa4d
  python -m pipeline.apply_object_lighting_mismatch \
    --model_path output/hypernerf/split-cookie \
    --source_path data/hypernerf/split-cookie \
    --in_ply output/hypernerf/split-cookie/point_cloud/iteration_14000/clean_chocolate_Bigger.ply \
    --mask_path output/hypernerf/split-cookie/segment_results/composite_inserted_choc_Bigger.pt \
    --out_ply output/hypernerf/split-cookie/point_cloud/iteration_14000/clean_chocolate_Bigger_mismatch.ply \
    --brightness 0.35 --gamma 1.6
"""

import os
import sys
import argparse

import torch

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)


def main():
    p = argparse.ArgumentParser(description="Apply object-only lighting mismatch to a composite PLY")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--source_path", type=str, required=True)
    p.add_argument("--in_ply", type=str, required=True)
    p.add_argument("--mask_path", type=str, required=True)
    p.add_argument("--out_ply", type=str, required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--configs", type=str, default=None)

    p.add_argument("--brightness", type=float, default=0.35, help="Multiply object RGB SH-derived appearance via DC scaling ~(1-brightness)")
    p.add_argument("--gamma", type=float, default=1.6, help="Gamma on object colors in rendered SH domain (approx via clamp+pow on DC contribution)")
    p.add_argument("--cast_r", type=float, default=1.0)
    p.add_argument("--cast_g", type=float, default=1.0)
    p.add_argument("--cast_b", type=float, default=1.0)
    args = p.parse_args()

    from pipeline.data_loading import load_scene, load_mask_table, get_object_mask

    gaussians, scene, pipe, bg = load_scene(args.model_path, args.source_path, iteration=args.iteration, configs=args.configs)
    _ = scene, pipe, bg  # unused but keeps cfg consistent with training

    gaussians.load_ply(args.in_ply)
    if hasattr(gaussians, "_deformation_table"):
        n_xyz = gaussians._xyz.shape[0]
        if (not torch.is_tensor(gaussians._deformation_table)) or (gaussians._deformation_table.shape[0] != n_xyz):
            gaussians._deformation_table = torch.ones((n_xyz,), device="cuda", dtype=torch.bool)

    mask_data = load_mask_table(args.mask_path)
    n_model = int(gaussians._xyz.shape[0])
    n_mask = int(mask_data["mask_table"].shape[1])
    if n_model != n_mask:
        raise ValueError(f"Gaussian count mismatch: model={n_model}, mask={n_mask}")

    obj = get_object_mask(mask_data)
    if obj.sum().item() == 0:
        raise ValueError("Object mask is empty (union across frames).")

    with torch.no_grad():
        # Work in the same SH tensor layout as the model: [N,16,3]
        dc = gaussians._features_dc
        rest = gaussians._features_rest
        sh = torch.cat([dc, rest], dim=1)

        # Approx object-only color manipulation:
        # - scale DC channels (dominant view-independent component)
        # - apply per-channel cast on all SH coeffs (simple)
        sh_obj = sh[obj].clone()
        sh_obj[:, 0:1, :] *= float(max(0.0, 1.0 - float(args.brightness)))
        cast = torch.tensor([args.cast_r, args.cast_g, args.cast_b], device=sh_obj.device, dtype=sh_obj.dtype).view(1, 1, 3)
        sh_obj = sh_obj * cast
        # Gamma-like emphasis: apply pow on magnitude with sign preservation (crude but stable)
        if abs(float(args.gamma) - 1.0) > 1e-6:
            sgn = torch.sign(sh_obj)
            sh_obj = sgn * torch.abs(sh_obj).clamp(min=0.0).pow(float(args.gamma))

        sh[obj] = sh_obj

        new_dc = sh[:, :1, :].contiguous()
        new_rest = sh[:, 1:, :].contiguous()
        gaussians._features_dc = torch.nn.Parameter(new_dc)
        gaussians._features_rest = torch.nn.Parameter(new_rest)

        os.makedirs(os.path.dirname(args.out_ply), exist_ok=True)
        gaussians.save_ply(args.out_ply)

    print(f"[mismatch] Wrote: {args.out_ply}")
    print(f"[mismatch] Modified {int(obj.sum().item())} / {n_model} Gaussians (union object mask)")


if __name__ == "__main__":
    main()
