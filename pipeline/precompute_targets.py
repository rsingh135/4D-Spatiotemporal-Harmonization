"""
Harmonization target precomputation.

This module renders every (view, time) pair, runs the Harmonizer to predict
filter arguments, smooths them across views and time, then generates target
images that represent "what correct lighting looks like."

Key methods:
  predict_filter_args()   -> raw 6-scalar filter args for one (composite, mask) pair
  smooth_filter_args()    -> temporally smooth filter args across frames
  precompute_all_targets() -> dict {(view_idx, frame_idx): target_image}
  save_targets() / load_targets()  -> persist to / load from disk
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


def predict_filter_args(harmonizer, composite, mask_2d):
    """
    Run the Harmonizer backbone + regressor to predict 6 filter arguments.

    Args:
        harmonizer: pretrained Harmonizer nn.Module
        composite:  tensor [1, 3, H, W] in [0, 1]
        mask_2d:    tensor [1, 1, H, W] in [0, 1]

    Returns:
        list of 6 tensors, each [1, 1] (the scalar filter arguments)
    """
    return harmonizer.predict_arguments(composite, mask_2d)


def apply_filters(harmonizer, composite, mask_2d, arguments):
    """
    Apply 6 white-box filters with given arguments. Returns harmonized image.

    Args:
        harmonizer:  pretrained Harmonizer nn.Module
        composite:   tensor [1, 3, H, W] in [0, 1]
        mask_2d:     tensor [1, 1, H, W] in [0, 1]
        arguments:   list of 6 tensors [1, 1]

    Returns:
        tensor [1, 3, H, W] — the harmonized image (last filter output)
    """
    outputs = harmonizer.restore_image(composite, mask_2d, arguments)
    return outputs[-1]  # final image after all 6 filters applied


def smooth_filter_args(args_per_frame, sigma=2.0):
    """
    Temporally smooth filter arguments across frames.

    Args:
        args_per_frame: list of T lists, each inner list has 6 tensors [1,1]
        sigma:          Gaussian smoothing sigma (in frames)

    Returns:
        list of T lists, each inner list has 6 tensors [1,1] (smoothed)
    """
    # Stack into [T, 6]
    rows = []
    for frame_args in args_per_frame:
        rows.append(torch.stack([a.squeeze() for a in frame_args]))  # [6]
    stacked = torch.stack(rows)  # [T, 6]

    arr = stacked.detach().cpu().numpy()
    smoothed = gaussian_filter1d(arr, sigma=sigma, axis=0)
    smoothed = torch.tensor(smoothed, dtype=stacked.dtype, device=stacked.device)

    # Rebuild into list-of-lists format
    result = []
    for t_idx in range(smoothed.shape[0]):
        result.append([smoothed[t_idx, f].view(1, 1) for f in range(6)])
    return result


def render_composite_and_mask(view, gaussians, pipe, background, mask_data, frame_idx):
    """
    Render the full scene composite and a 2D object mask for a given view+frame.

    Uses the sa4d render() for the composite and render_mask() for the 2D mask.
    The per-Gaussian mask for this frame is looked up from mask_data.

    Args:
        view:        camera object from scene.getTrainCameras()
        gaussians:   GaussianModel
        pipe:        PipelineParams namespace
        background:  background color tensor
        mask_data:   dict from load_mask_table()
        frame_idx:   int, frame index into mask_table

    Returns:
        composite: tensor [1, 3, H, W] in [0, 1]
        mask_2d:   tensor [1, 1, H, W] in [0, 1]
    """
    from gaussian_renderer import render, render_mask

    # Render full-scene composite image
    result = render(view, gaussians, pipe, background, stage='fine')
    composite = result['render'].unsqueeze(0).clamp(0, 1)  # [1, 3, H, W]

    # Build per-Gaussian float mask [N_gaussians, 1] for this frame
    gauss_mask_bool = mask_data['mask_table'][frame_idx]  # [N_gaussians] bool
    gauss_mask_float = gauss_mask_bool.float().unsqueeze(-1)  # [N_gaussians, 1]

    # Render 2D mask by projecting per-Gaussian mask through rasterizer
    mask_result = render_mask(view, gaussians, pipe, background,
                              precomputed_mask=gauss_mask_float)
    mask_2d = mask_result['mask'].unsqueeze(0).clamp(0, 1)  # [1, 1, H, W]

    return composite, mask_2d


def precompute_all_targets(harmonizer, gaussians, scene, pipe, background,
                           mask_data, sigma=2.0, use_train_cams=True):
    """
    Full target precomputation pipeline:
      1. For each (view, frame), render composite + 2D mask
      2. Predict filter arguments with Harmonizer
      3. Average arguments across views per frame (view-consensus)
      4. Smooth arguments across time
      5. Apply smoothed filters to generate target images

    Args:
        harmonizer:     pretrained Harmonizer nn.Module
        gaussians:      GaussianModel
        scene:          Scene object
        pipe:           PipelineParams
        background:     background tensor
        mask_data:      dict from load_mask_table()
        sigma:          temporal smoothing sigma
        use_train_cams: if True, use training cameras; else test cameras

    Returns:
        targets: dict {(view_idx, frame_idx): tensor [1, 3, H, W]}
        composites: dict {(view_idx, frame_idx): tensor [1, 3, H, W]}
        masks_2d: dict {(view_idx, frame_idx): tensor [1, 1, H, W]}
    """
    from pipeline.data_loading import time_to_frame_idx

    views = scene.getTrainCameras() if use_train_cams else scene.getTestCameras()
    n_frames = mask_data['mask_table'].shape[0]

    # --- Step 1 & 2: Render and predict filter args ---
    print("[precompute] Step 1/4: Rendering composites and predicting filter args...")
    raw_args = {}       # (v_idx, f_idx) -> list of 6 tensors
    composites = {}     # (v_idx, f_idx) -> [1, 3, H, W]
    masks_2d = {}       # (v_idx, f_idx) -> [1, 1, H, W]

    with torch.no_grad():
        for v_idx, view in enumerate(tqdm(views, desc="Views")):
            # Map this camera's time to closest frame
            view_time = view.time if hasattr(view, 'time') else 0.0
            f_idx = time_to_frame_idx(mask_data, view_time)

            comp, m2d = render_composite_and_mask(
                view, gaussians, pipe, background, mask_data, f_idx)
            composites[(v_idx, f_idx)] = comp
            masks_2d[(v_idx, f_idx)] = m2d

            theta = predict_filter_args(harmonizer, comp, m2d)
            raw_args[(v_idx, f_idx)] = theta

    # --- Step 3: Average across views per frame (view-consensus) ---
    print("[precompute] Step 2/4: Computing view-consensus per frame...")
    # Group by frame
    frame_to_views = {}
    for (v_idx, f_idx) in raw_args:
        frame_to_views.setdefault(f_idx, []).append(v_idx)

    frame_consensus = {}  # f_idx -> list of 6 tensors [1,1]
    for f_idx in sorted(frame_to_views.keys()):
        v_indices = frame_to_views[f_idx]
        stacked = [
            torch.stack([raw_args[(v, f_idx)][f] for v in v_indices])
            for f in range(6)
        ]
        frame_consensus[f_idx] = [s.mean(dim=0) for s in stacked]

    # --- Step 4: Smooth across time ---
    print("[precompute] Step 3/4: Temporal smoothing...")
    sorted_frames = sorted(frame_consensus.keys())
    args_seq = [frame_consensus[f] for f in sorted_frames]

    if len(args_seq) > 1:
        smoothed_seq = smooth_filter_args(args_seq, sigma=sigma)
    else:
        smoothed_seq = args_seq

    smoothed_consensus = {}
    for i, f_idx in enumerate(sorted_frames):
        smoothed_consensus[f_idx] = smoothed_seq[i]

    # --- Step 5: Generate target images ---
    print("[precompute] Step 4/4: Generating harmonized target images...")
    targets = {}
    with torch.no_grad():
        for (v_idx, f_idx), comp in tqdm(composites.items(), desc="Targets"):
            m2d = masks_2d[(v_idx, f_idx)]
            # Use the smoothed consensus args for this frame
            theta = smoothed_consensus[f_idx]
            target = apply_filters(harmonizer, comp, m2d, theta)
            targets[(v_idx, f_idx)] = target

    print(f"[precompute] Generated {len(targets)} target images")
    return targets, composites, masks_2d


def save_targets(targets, composites, masks_2d, out_dir):
    """
    Save precomputed targets, composites, and masks to disk.

    Saves as a single .pt file for easy reloading.
    """
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'harmonize_targets.pt')

    # Convert dict keys to lists for serialization
    data = {
        'targets': {str(k): v.cpu() for k, v in targets.items()},
        'composites': {str(k): v.cpu() for k, v in composites.items()},
        'masks_2d': {str(k): v.cpu() for k, v in masks_2d.items()},
    }
    torch.save(data, save_path)
    print(f"[precompute] Saved targets to {save_path}")


def load_targets(out_dir):
    """
    Load precomputed targets from disk.

    Returns:
        targets, composites, masks_2d — same format as precompute_all_targets()
    """
    save_path = os.path.join(out_dir, 'harmonize_targets.pt')
    data = torch.load(save_path, map_location='cuda')

    targets = {eval(k): v.cuda() for k, v in data['targets'].items()}
    composites = {eval(k): v.cuda() for k, v in data['composites'].items()}
    masks_2d = {eval(k): v.cuda() for k, v in data['masks_2d'].items()}

    print(f"[precompute] Loaded {len(targets)} targets from {save_path}")
    return targets, composites, masks_2d
