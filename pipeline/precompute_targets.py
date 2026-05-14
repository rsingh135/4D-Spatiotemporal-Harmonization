"""
Harmonization target precomputation.

This module renders every (view, time) pair, runs a harmonizer to predict
target images, and optionally saves diff visualizations.

Supports two harmonizer backends via the HarmonizerBase abstraction:
  - "whitebox"  : Original Harmonizer (6 global white-box filters)
  - "pctnet"    : PCT-Net (per-pixel color transfer, CVPR 2023)

Key methods:
  precompute_all_targets() -> dict {(view_idx, frame_idx): target_image}
  save_targets() / load_targets()  -> persist to / load from disk
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image


# ── Legacy whitebox-specific helpers (kept for backward compat) ──────────

def predict_filter_args(harmonizer, composite, mask_2d):
    """Whitebox only: predict 6 scalar filter arguments."""
    return harmonizer.predict_arguments(composite, mask_2d)


def apply_filters(harmonizer, composite, mask_2d, arguments):
    """Whitebox only: apply 6 white-box filters."""
    outputs = harmonizer.restore_image(composite, mask_2d, arguments)
    return outputs[-1]


def smooth_filter_args(args_per_frame, sigma=2.0):
    """Temporally smooth whitebox filter arguments across frames."""
    from scipy.ndimage import gaussian_filter1d

    rows = []
    for frame_args in args_per_frame:
        rows.append(torch.stack([a.squeeze() for a in frame_args]))
    stacked = torch.stack(rows)

    arr = stacked.detach().cpu().numpy()
    smoothed = gaussian_filter1d(arr, sigma=sigma, axis=0)
    smoothed = torch.tensor(smoothed, dtype=stacked.dtype, device=stacked.device)

    result = []
    for t_idx in range(smoothed.shape[0]):
        result.append([smoothed[t_idx, f].view(1, 1) for f in range(6)])
    return result


# ── Rendering helper ─────────────────────────────────────────────────────

def render_composite_and_mask(view, gaussians, pipe, background, mask_data, frame_idx):
    """
    Render the full scene composite and a 2D object mask for a given view+frame.
    """
    from gaussian_renderer import render, render_mask

    result = render(view, gaussians, pipe, background, stage='fine')
    composite = result['render'].unsqueeze(0).clamp(0, 1)

    gauss_mask_bool = mask_data['mask_table'][frame_idx]
    gauss_mask_float = gauss_mask_bool.float().unsqueeze(-1)

    mask_result = render_mask(view, gaussians, pipe, background,
                              precomputed_mask=gauss_mask_float)
    mask_2d_raw = mask_result['mask'].unsqueeze(0).clamp(0, 1)
    # Binarize: rendered mask values can be very low (e.g. max ~0.3) when
    # object Gaussians are semi-transparent.  Threshold at 0.01 so any pixel
    # with meaningful object contribution is included.
    mask_2d = (mask_2d_raw > 0.01).float()

    return composite, mask_2d


# ── Mask feathering helper ────────────────────────────────────────────────

def feather_mask(mask_2d, sigma_px: float):
    """
    Feather a [1,1,H,W] mask using a separable Gaussian blur in torch.
    This is mainly to reduce hard/aliased boundaries that create halos.
    """
    if sigma_px is None or sigma_px <= 0:
        return mask_2d

    # Kernel radius ~ 3*sigma (cap to avoid huge kernels)
    radius = int(max(1, min(25, round(3.0 * float(sigma_px)))))
    k = 2 * radius + 1

    device = mask_2d.device
    dtype = mask_2d.dtype

    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(xs * xs) / (2.0 * (sigma_px ** 2)))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable conv: horizontal then vertical
    w_h = kernel_1d.view(1, 1, 1, k)
    w_v = kernel_1d.view(1, 1, k, 1)

    x = F.pad(mask_2d, (radius, radius, 0, 0), mode='reflect')
    x = F.conv2d(x, w_h)
    x = F.pad(x, (0, 0, radius, radius), mode='reflect')
    x = F.conv2d(x, w_v)
    return x.clamp(0, 1)


# ── Diff visualization helper ────────────────────────────────────────────

def _to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def save_diff_image(composite, target, mask_2d, out_path, diff_scale=10.0):
    """
    Save a side-by-side diff image: [composite | target | diff×scale | mask].
    All inputs are [1, C, H, W] tensors on any device.
    """
    comp = composite.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    tgt = target.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mask = mask_2d.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    diff = np.clip(np.abs(tgt - comp) * diff_scale, 0, 1)
    mask_rgb = np.repeat(mask, 3, axis=2)

    # 2px gray separator
    h = comp.shape[0]
    sep = np.full((h, 2, 3), 0.3, dtype=np.float32)
    canvas = np.concatenate([comp, sep, tgt, sep, diff, sep, mask_rgb], axis=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(_to8b(canvas)).save(out_path)


# ── Unified precompute (works with any HarmonizerBase backend) ───────────

def precompute_all_targets(harmonizer, gaussians, scene, pipe, background,
                           mask_data, sigma=2.0, use_train_cams=True,
                           diff_dir=None, mask_feather_sigma: float = 0.0,
                           amplify: float = 1.0):
    """
    Full target precomputation pipeline.

    Works with any harmonizer that implements the HarmonizerBase interface
    (.harmonize(composite, mask_2d) -> target), OR with the legacy whitebox
    Harmonizer that has .predict_arguments / .restore_image.

    Args:
        harmonizer:          HarmonizerBase instance, or legacy Harmonizer nn.Module
        gaussians:           GaussianModel
        scene:               Scene object
        pipe:                PipelineParams
        background:          background tensor
        mask_data:           dict from load_mask_table()
        sigma:               temporal smoothing sigma (whitebox only)
        use_train_cams:      if True, use training cameras; else test cameras
        diff_dir:            if not None, save per-view diff images to this directory
        mask_feather_sigma:  feather the 2D mask by this sigma in pixels (0 = off)
        amplify:             amplify harmonizer correction by this factor.
                             target = comp + amplify * (harmonized - comp).
                             1.0 = normal, 3.0 = 3x stronger, etc.

    Returns:
        targets: dict {(view_idx, frame_idx): tensor [1, 3, H, W]}
        composites: dict {(view_idx, frame_idx): tensor [1, 3, H, W]}
        masks_2d: dict {(view_idx, frame_idx): tensor [1, 1, H, W]}
    """
    from pipeline.data_loading import time_to_frame_idx
    from pipeline.harmonizer_base import HarmonizerBase, SceneBHarmonizer

    views = scene.getTrainCameras() if use_train_cams else scene.getTestCameras()
    use_scene_b = isinstance(harmonizer, SceneBHarmonizer)
    use_abstraction = isinstance(harmonizer, HarmonizerBase) and not use_scene_b

    # ── Step 1: Render composites + masks ──
    print("[precompute] Step 1: Rendering composites and masks...")
    composites = {}
    masks_2d = {}

    with torch.no_grad():
        for v_idx, view in enumerate(tqdm(views, desc="Rendering")):
            view_time = view.time if hasattr(view, 'time') else 0.0
            f_idx = time_to_frame_idx(mask_data, view_time)
            comp, m2d = render_composite_and_mask(
                view, gaussians, pipe, background, mask_data, f_idx)
            if mask_feather_sigma and mask_feather_sigma > 0:
                m2d = feather_mask(m2d, mask_feather_sigma)
            composites[(v_idx, f_idx)] = comp
            masks_2d[(v_idx, f_idx)] = m2d

    # ── Step 2: Generate targets ──
    if amplify != 1.0:
        print(f"[precompute] Amplification factor: {amplify}x")

    if use_scene_b:
        # Load ground-truth Scene B images directly as targets
        split = 'train' if use_train_cams else 'test'
        print(f"[precompute] Step 2: Loading Scene B images as targets ({split})...")
        targets = {}
        for (v_idx, f_idx), comp in tqdm(composites.items(), desc="Loading Scene B"):
            view = views[v_idx]
            view_name = f"{split}/r_{view.image_name}"
            H, W = comp.shape[2], comp.shape[3]
            raw_target = harmonizer.get_target_for_view(view_name, H, W)
            if amplify != 1.0:
                target = (comp + amplify * (raw_target - comp)).clamp(0, 1)
            else:
                target = raw_target
            targets[(v_idx, f_idx)] = target
    elif use_abstraction:
        # New path: direct harmonize() call per view
        print("[precompute] Step 2: Generating targets via harmonizer.harmonize()...")
        targets = {}
        with torch.no_grad():
            for (v_idx, f_idx), comp in tqdm(composites.items(), desc="Harmonizing"):
                m2d = masks_2d[(v_idx, f_idx)]
                raw_target = harmonizer.harmonize(comp, m2d)
                if amplify != 1.0:
                    target = (comp + amplify * (raw_target - comp)).clamp(0, 1)
                else:
                    target = raw_target
                targets[(v_idx, f_idx)] = target
    else:
        # Legacy whitebox path: predict_args → consensus → smooth → apply
        print("[precompute] Step 2a: Predicting filter args (whitebox)...")
        raw_args = {}
        with torch.no_grad():
            for (v_idx, f_idx), comp in tqdm(composites.items(), desc="Filter args"):
                m2d = masks_2d[(v_idx, f_idx)]
                theta = predict_filter_args(harmonizer, comp, m2d)
                raw_args[(v_idx, f_idx)] = theta

        print("[precompute] Step 2b: View-consensus per frame...")
        frame_to_views = {}
        for (v_idx, f_idx) in raw_args:
            frame_to_views.setdefault(f_idx, []).append(v_idx)

        frame_consensus = {}
        for f_idx in sorted(frame_to_views.keys()):
            v_indices = frame_to_views[f_idx]
            stacked = [
                torch.stack([raw_args[(v, f_idx)][f] for v in v_indices])
                for f in range(6)
            ]
            frame_consensus[f_idx] = [s.mean(dim=0) for s in stacked]

        print("[precompute] Step 2c: Temporal smoothing...")
        sorted_frames = sorted(frame_consensus.keys())
        args_seq = [frame_consensus[f] for f in sorted_frames]
        if len(args_seq) > 1:
            smoothed_seq = smooth_filter_args(args_seq, sigma=sigma)
        else:
            smoothed_seq = args_seq
        smoothed_consensus = {}
        for i, f_idx in enumerate(sorted_frames):
            smoothed_consensus[f_idx] = smoothed_seq[i]

        print("[precompute] Step 2d: Generating target images...")
        targets = {}
        with torch.no_grad():
            for (v_idx, f_idx), comp in tqdm(composites.items(), desc="Targets"):
                m2d = masks_2d[(v_idx, f_idx)]
                theta = smoothed_consensus[f_idx]
                raw_target = apply_filters(harmonizer, comp, m2d, theta)
                if amplify != 1.0:
                    target = (comp + amplify * (raw_target - comp)).clamp(0, 1)
                else:
                    target = raw_target
                targets[(v_idx, f_idx)] = target

    # ── Step 3: Save diffs ──
    if diff_dir is not None:
        print(f"[precompute] Saving diff images to {diff_dir}/...")
        all_diffs = []
        all_mask_cov = []
        for (v_idx, f_idx) in sorted(targets.keys()):
            comp = composites[(v_idx, f_idx)]
            tgt = targets[(v_idx, f_idx)]
            m2d = masks_2d[(v_idx, f_idx)]

            out_path = os.path.join(diff_dir, f'view{v_idx:03d}_frame{f_idx:03d}.png')
            save_diff_image(comp, tgt, m2d, out_path)

            # Stats
            m = (m2d > 0.5).float()
            masked_diff = ((tgt - comp) * m).abs()
            mean_diff = masked_diff.sum() / max(1, m.sum())
            mask_cov = (m > 0).float().mean()
            all_diffs.append(mean_diff.item())
            all_mask_cov.append(mask_cov.item())

        print(f"[precompute] Diff stats across {len(all_diffs)} views:")
        print(f"  Mean mask coverage:  {np.mean(all_mask_cov):.4f}")
        print(f"  Mean masked diff:    {np.mean(all_diffs):.6f}")
        print(f"  Max masked diff:     {np.max(all_diffs):.6f}")

    print(f"[precompute] Generated {len(targets)} target images")
    return targets, composites, masks_2d


# ── Save / Load ──────────────────────────────────────────────────────────

def save_targets(targets, composites, masks_2d, out_dir, *, mode: str = "all", dtype: str = "fp16"):
    """
    Save precomputed targets to disk.

    Args:
        mode:
          - "all": save targets + composites + masks_2d (largest)
          - "targets_only": save only targets (smaller)
        dtype:
          - "fp32": store as float32 tensors
          - "fp16": store as float16 tensors (recommended for large sweeps)
    """
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'harmonize_targets.pt')

    if dtype not in ("fp16", "fp32"):
        raise ValueError(f"Unknown dtype={dtype!r}. Use fp16 or fp32.")
    if mode not in ("all", "targets_only"):
        raise ValueError(f"Unknown mode={mode!r}. Use all or targets_only.")

    def _cast(v: torch.Tensor) -> torch.Tensor:
        x = v.detach().cpu()
        if dtype == "fp16" and x.dtype.is_floating_point:
            return x.half()
        if dtype == "fp32" and x.dtype.is_floating_point:
            return x.float()
        return x

    data = {"targets": {str(k): _cast(v) for k, v in targets.items()}, "meta": {"mode": mode, "dtype": dtype}}
    if mode == "all":
        data["composites"] = {str(k): _cast(v) for k, v in composites.items()}
        data["masks_2d"] = {str(k): _cast(v) for k, v in masks_2d.items()}
    torch.save(data, save_path)
    print(f"[precompute] Saved targets to {save_path}")


def load_targets(out_dir):
    """Load precomputed targets from disk."""
    save_path = os.path.join(out_dir, 'harmonize_targets.pt')
    data = torch.load(save_path, map_location='cuda')

    targets = {eval(k): v.cuda() for k, v in data['targets'].items()}
    if "composites" in data:
        composites = {eval(k): v.cuda() for k, v in data['composites'].items()}
    else:
        composites = {}
    if "masks_2d" in data:
        masks_2d = {eval(k): v.cuda() for k, v in data['masks_2d'].items()}
    else:
        masks_2d = {}

    print(f"[precompute] Loaded {len(targets)} targets from {save_path}")
    return targets, composites, masks_2d
