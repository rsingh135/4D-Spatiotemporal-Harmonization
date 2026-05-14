#!/usr/bin/env python3
"""
Quantitative metrics for the harmonization pipeline.

We measure how well a rendered composite matches the *harmonizer target images*
that the optimizer is trained to fit.

Metrics (masked to the projected object region):
  - L1
  - PSNR
  - SSIM

We report BEFORE vs AFTER, where:
  BEFORE: composite PLY (lighting-mismatched input)
  AFTER:  harmonized PLY (ΔSH baked)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_targets(targets_pt: str):
    pack = torch.load(targets_pt, map_location="cpu")
    targets = pack["targets"] if isinstance(pack, dict) and "targets" in pack else pack
    masks_2d = pack.get("masks_2d", None) if isinstance(pack, dict) else None
    return targets, masks_2d


@torch.no_grad()
def render_frame(view, gaussians, pipe, background, cam_type: str) -> torch.Tensor:
    from gaussian_renderer import render
    return render(view, gaussians, pipe, background, cam_type=cam_type)["render"].clamp(0, 1)


def to_numpy_chw(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def masked_metrics(pred_chw: np.ndarray, tgt_chw: np.ndarray, mask_hw: np.ndarray) -> Tuple[float, float, float]:
    """
    pred_chw/tgt_chw: float in [0,1]
    mask_hw: float in {0,1}
    """
    m = mask_hw.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    m = np.clip(m, 0, 1)
    denom = float(np.sum(m)) + 1e-6

    diff = np.abs(pred_chw - tgt_chw)  # [3,H,W]
    l1 = float(np.sum(diff * m[None, :, :]) / (denom * 3.0))

    # For PSNR/SSIM, evaluate on the masked pixels only by cropping to bbox.
    ys, xs = np.where(m > 0.5)
    if ys.size < 10:
        return l1, float("nan"), float("nan")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    p = np.transpose(pred_chw[:, y0:y1, x0:x1], (1, 2, 0))
    t = np.transpose(tgt_chw[:, y0:y1, x0:x1], (1, 2, 0))
    mm = m[y0:y1, x0:x1]

    # Apply mask by blending with target (so unmasked pixels don't affect metric).
    p2 = p * mm[..., None] + t * (1.0 - mm[..., None])

    ps = float(psnr(t, p2, data_range=1.0))
    ss = float(ssim(t, p2, channel_axis=2, data_range=1.0))
    return l1, ps, ss


def compare_two_renders_metrics(a_chw: np.ndarray, b_chw: np.ndarray, mask_hw: np.ndarray) -> Tuple[float, float, float]:
    """
    Compare render A vs render B on pixels where mask==1.
    """
    return masked_metrics(a_chw, b_chw, mask_hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--iteration", type=int, default=14000)
    ap.add_argument("--mask_path", required=True)
    ap.add_argument("--before_ply", required=True)
    ap.add_argument("--after_ply", required=True)
    ap.add_argument("--targets_pt", required=True)
    ap.add_argument("--max_views", type=int, default=0, help="0=all train cams")
    args = ap.parse_args()

    # Make local imports work when invoked as a script.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, ".."))
    if repo_root not in os.sys.path:
        os.sys.path.insert(0, repo_root)

    # load scene
    from pipeline.data_loading import load_scene, load_mask_table, get_object_mask, time_to_frame_idx
    from pipeline.precompute_targets import render_composite_and_mask

    gaussians, scene, pipe, bg = load_scene(args.model_path, args.source_path, iteration=args.iteration, configs=args.configs)
    cam_type = scene.dataset_type

    # mask data (used for (view.time -> frame_idx) mapping)
    mask_data = load_mask_table(args.mask_path)

    # load targets (dict[(v_idx,f_idx)] -> [1,3,H,W])
    targets, _ = load_targets(args.targets_pt)
    if not isinstance(targets, dict) or not targets:
        raise SystemExit(f"Unexpected targets format in {args.targets_pt}")

    views = scene.getTrainCameras()
    limit = int(args.max_views) if args.max_views and args.max_views > 0 else len(views)
    view_ids = list(range(min(limit, len(views))))

    def eval_with_ply(ply_path: str) -> Dict[str, float]:
        # load the ply into the already-loaded GaussianModel
        gaussians.load_ply(os.path.abspath(ply_path))

        # ensure composite-style deformation table: bg deforms, object static
        fg_mask = get_object_mask(mask_data).bool().to("cuda")
        gaussians._deformation_table = (~fg_mask).clone().to("cuda")

        l1s, psnrs, ssims = [], [], []
        for v_idx in view_ids:
            view = views[v_idx]
            f_idx = time_to_frame_idx(mask_data, float(view.time))
            key = repr((v_idx, f_idx))  # harmonize_targets.pt stores stringified tuple keys
            if key not in targets:
                continue

            # render + get projected 2D mask the same way precompute does
            comp, m2d = render_composite_and_mask(view, gaussians, pipe, bg, mask_data, f_idx)
            pred = comp.squeeze(0)  # [3,H,W]
            tgt = targets[key].squeeze(0)  # [3,H,W]
            mask_hw = m2d.squeeze(0).squeeze(0).detach().float().cpu().numpy()  # [H,W]

            l1, p, s = masked_metrics(to_numpy_chw(pred), to_numpy_chw(tgt), mask_hw)
            l1s.append(l1)
            if np.isfinite(p):
                psnrs.append(p)
            if np.isfinite(s):
                ssims.append(s)

        return {
            "n": float(len(l1s)),
            "l1_mean": float(np.mean(l1s)) if l1s else float("nan"),
            "psnr_mean": float(np.mean(psnrs)) if psnrs else float("nan"),
            "ssim_mean": float(np.mean(ssims)) if ssims else float("nan"),
        }

    def background_stability(before_ply: str, after_ply: str) -> Dict[str, float]:
        # Load once per PLY and render in the same loop for consistent camera/time sampling.
        fg_mask = get_object_mask(mask_data).bool().to("cuda")

        def load_ply(p):
            gaussians.load_ply(os.path.abspath(p))
            gaussians._deformation_table = (~fg_mask).clone().to("cuda")

        l1s, psnrs, ssims = [], [], []
        for v_idx in view_ids:
            view = views[v_idx]
            f_idx = time_to_frame_idx(mask_data, float(view.time))

            # We use the same 2D projection mask as the pipeline.
            load_ply(before_ply)
            comp_a, m2d = render_composite_and_mask(view, gaussians, pipe, bg, mask_data, f_idx)
            a = comp_a.squeeze(0)

            load_ply(after_ply)
            comp_b, _ = render_composite_and_mask(view, gaussians, pipe, bg, mask_data, f_idx)
            b = comp_b.squeeze(0)

            mask_hw = m2d.squeeze(0).squeeze(0).detach().float().cpu().numpy()
            bg_mask = (1.0 - mask_hw).astype(np.float32)

            l1, p, s = compare_two_renders_metrics(to_numpy_chw(a), to_numpy_chw(b), bg_mask)
            l1s.append(l1)
            if np.isfinite(p):
                psnrs.append(p)
            if np.isfinite(s):
                ssims.append(s)

        return {
            "n": float(len(l1s)),
            "bg_l1_mean": float(np.mean(l1s)) if l1s else float("nan"),
            "bg_psnr_mean": float(np.mean(psnrs)) if psnrs else float("nan"),
            "bg_ssim_mean": float(np.mean(ssims)) if ssims else float("nan"),
        }

    before = eval_with_ply(args.before_ply)
    after = eval_with_ply(args.after_ply)
    bgstab = background_stability(args.before_ply, args.after_ply)

    print("Masked metrics vs harmonizer targets (train views):")
    print(f"  views evaluated: {int(before['n'])}")
    print("")
    print("  BEFORE (composite/mismatch):")
    print(f"    L1   : {before['l1_mean']:.6f}")
    print(f"    PSNR : {before['psnr_mean']:.3f}")
    print(f"    SSIM : {before['ssim_mean']:.4f}")
    print("")
    print("  AFTER (harmonized):")
    print(f"    L1   : {after['l1_mean']:.6f}")
    print(f"    PSNR : {after['psnr_mean']:.3f}")
    print(f"    SSIM : {after['ssim_mean']:.4f}")
    print("")
    if np.isfinite(before["l1_mean"]) and np.isfinite(after["l1_mean"]):
        rel = (before["l1_mean"] - after["l1_mean"]) / max(before["l1_mean"], 1e-8)
        print(f"  ΔL1 improvement: {rel*100.0:.1f}% (relative)")

    print("")
    print("Background stability (BEFORE vs AFTER renders, unmasked pixels only):")
    print(f"  BG L1   : {bgstab['bg_l1_mean']:.6f}")
    print(f"  BG PSNR : {bgstab['bg_psnr_mean']:.3f}")
    print(f"  BG SSIM : {bgstab['bg_ssim_mean']:.4f}")


if __name__ == "__main__":
    main()

