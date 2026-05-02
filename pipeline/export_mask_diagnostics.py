"""
Multi-view mask + composite diagnostics.

Exports per-view PNG strips:
  [composite | mask | overlay]

This is meant to quickly spot:
  - mask leakage / islands in background
  - misalignment between projected mask and rendered object

Usage (from repo root):
  cd /home/ubuntu/new_sa4d/sa4d
  python -m pipeline.export_mask_diagnostics \
    --model_path output/hypernerf/split-cookie \
    --source_path data/hypernerf/split-cookie \
    --mask_path output/hypernerf/split-cookie/segment_results/composite_inserted_choc_Bigger.pt \
    --ply_path output/hypernerf/split-cookie/point_cloud/iteration_14000/clean_chocolate_Bigger.ply \
    --out_dir output/hypernerf/split-cookie/mask_diag_bigger \
    --mask_feather 0 \
    --max_views 0
"""

import os
import sys
import argparse

import numpy as np
import torch

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)


def _to8b(x: np.ndarray) -> np.ndarray:
    return (255.0 * np.clip(x, 0.0, 1.0)).astype(np.uint8)


def _save_strip(comp_hwc: np.ndarray, mask_hw1: np.ndarray, out_path: str) -> None:
    from PIL import Image

    h, w, _ = comp_hwc.shape
    m = np.repeat(mask_hw1, 3, axis=2)
    ov = np.clip(comp_hwc * 0.65 + m * 0.35, 0.0, 1.0)

    sep = np.full((h, 2, 3), 0.25, dtype=np.float32)
    canvas = np.concatenate([comp_hwc, sep, m, sep, ov], axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(_to8b(canvas)).save(out_path)


def main():
    p = argparse.ArgumentParser(description="Export multi-view mask diagnostics")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--source_path", type=str, required=True)
    p.add_argument("--mask_path", type=str, required=True)
    p.add_argument("--ply_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--configs", type=str, default=None)
    p.add_argument("--use_test_cams", action="store_true")
    p.add_argument("--mask_feather", type=float, default=0.0)
    p.add_argument("--max_views", type=int, default=0, help="0 = all train/test cameras")
    args = p.parse_args()

    from pipeline.data_loading import load_scene, load_mask_table, time_to_frame_idx
    from pipeline.precompute_targets import render_composite_and_mask, feather_mask

    gaussians, scene, pipe, bg = load_scene(args.model_path, args.source_path, iteration=args.iteration, configs=args.configs)
    if args.ply_path is not None:
        gaussians.load_ply(args.ply_path)
        if hasattr(gaussians, "_deformation_table"):
            n_xyz = gaussians._xyz.shape[0]
            if (not torch.is_tensor(gaussians._deformation_table)) or (gaussians._deformation_table.shape[0] != n_xyz):
                gaussians._deformation_table = torch.ones((n_xyz,), device="cuda", dtype=torch.bool)

    mask_data = load_mask_table(args.mask_path)
    n_model = int(gaussians._xyz.shape[0])
    n_mask = int(mask_data["mask_table"].shape[1])
    if n_model != n_mask:
        raise ValueError(f"Gaussian count mismatch: model={n_model}, mask={n_mask}. Fix --ply_path / --mask_path pairing.")

    views = scene.getTestCameras() if args.use_test_cams else scene.getTrainCameras()

    os.makedirs(args.out_dir, exist_ok=True)

    stats = []
    with torch.no_grad():
        n_views = len(views)
        limit = int(args.max_views) if args.max_views and args.max_views > 0 else n_views
        for v_idx in range(min(limit, n_views)):
            view = views[v_idx]
            t = view.time if hasattr(view, "time") else 0.0
            f_idx = time_to_frame_idx(mask_data, t)
            comp, m2d = render_composite_and_mask(view, gaussians, pipe, bg, mask_data, f_idx)
            if args.mask_feather and args.mask_feather > 0:
                m2d = feather_mask(m2d, args.mask_feather)

            comp_hwc = comp.squeeze(0).detach().float().cpu().numpy().transpose(1, 2, 0)
            mask_hw = m2d.squeeze(0).squeeze(0).detach().float().cpu().numpy()[..., None]

            cov = float((m2d > 0.5).float().mean().item())
            stats.append(cov)

            out_path = os.path.join(args.out_dir, f"view{v_idx:04d}_frame{f_idx:04d}.png")
            _save_strip(comp_hwc, mask_hw, out_path)

    if stats:
        arr = np.array(stats, dtype=np.float32)
        print(f"[mask_diag] saved {len(stats)} images to {args.out_dir}/")
        print(f"[mask_diag] coverage thr>0.5: mean={arr.mean():.4f} min={arr.min():.4f} max={arr.max():.4f}")


if __name__ == "__main__":
    main()
