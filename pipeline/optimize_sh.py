"""
SH coefficient optimization via backpropagation.

This module optimizes a delta_sh residual that gets added to the object
Gaussians' SH coefficients. Gradients flow through the differentiable
rasterizer back to delta_sh.

Key methods:
  create_delta_sh()       -> (delta_sh_dc, delta_sh_rest) learnable params
  render_with_delta_sh()  -> render scene with modified SH on the autograd graph
  train_step()            -> one optimization step
  optimize()              -> full training loop
  apply_delta_sh()        -> bake optimized delta into gaussians permanently
  save_harmonized_ply()   -> save the final .ply file

Gradient path:
  delta_sh → added to SH features → deformation network → rasterizer → pixels → loss
  The CUDA rasterizer backward pass computes grad_sh, which flows back to delta_sh.
"""

import os
import sys
import math
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)


def create_delta_sh(gaussians, object_mask, lr=1e-3):
    """
    Create learnable SH residual tensors for the object Gaussians.

    The residuals are zero-initialized and will be optimized to harmonize
    the object's appearance.

    Args:
        gaussians:    GaussianModel
        object_mask:  bool tensor [N_gaussians] — True for object Gaussians
        lr:           learning rate for Adam

    Returns:
        delta_sh_dc:   tensor [N_object, 1, 3] with requires_grad
        delta_sh_rest: tensor [N_object, 15, 3] with requires_grad  (for sh_degree=3)
        optimizer:     Adam optimizer over the deltas
        object_mask:   the bool mask (passed through for convenience)
    """
    n_object = object_mask.sum().item()
    dc_shape = gaussians._features_dc[object_mask].shape   # [N_obj, 1, 3]
    rest_shape = gaussians._features_rest[object_mask].shape  # [N_obj, 15, 3]

    delta_sh_dc = torch.zeros(dc_shape, device='cuda', requires_grad=True)
    delta_sh_rest = torch.zeros(rest_shape, device='cuda', requires_grad=True)

    optimizer = torch.optim.Adam([delta_sh_dc, delta_sh_rest], lr=lr)

    print(f"[optimize] Created delta_sh: {n_object} object Gaussians, "
          f"dc={list(dc_shape)}, rest={list(rest_shape)}")
    return delta_sh_dc, delta_sh_rest, optimizer, object_mask


def render_with_delta_sh(view, gaussians, pipe, background,
                         delta_sh_dc, delta_sh_rest, object_mask):
    """
    Render the full scene with delta_sh added to the object Gaussians' SH.

    This builds the modified SH tensor on the autograd graph so gradients
    flow from the loss through the rasterizer back to delta_sh_dc/rest.

    The approach:
      1. Read the base SH features (detached from the Gaussian model's params)
      2. Add delta_sh to the object Gaussian entries (ON the compute graph)
      3. Pass the modified SH through deformation → rasterizer

    Args:
        view:          camera object
        gaussians:     GaussianModel
        pipe:          PipelineParams
        background:    background tensor
        delta_sh_dc:   tensor [N_object, 1, 3] (leaf, requires_grad)
        delta_sh_rest: tensor [N_object, 15, 3] (leaf, requires_grad)
        object_mask:   bool tensor [N_gaussians]

    Returns:
        rendered_image: tensor [3, H, W] — differentiable w.r.t. delta_sh
    """
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer)

    means3D = gaussians.get_xyz
    N = means3D.shape[0]

    # -- Build modified SH features on the compute graph --
    # Base features detached: we do NOT want gradients flowing to the
    # original Gaussian parameters, only to our delta_sh leaf tensors.
    base_dc = gaussians._features_dc.detach()      # [N, 1, 3]
    base_rest = gaussians._features_rest.detach()   # [N, 15, 3]

    # Clone so we can scatter-add delta without in-place issues
    mod_dc = base_dc.clone()                        # [N, 1, 3]
    mod_rest = base_rest.clone()                    # [N, 15, 3]

    # Add delta_sh to the object Gaussians — this puts delta on the graph
    mod_dc[object_mask] = base_dc[object_mask] + delta_sh_dc
    mod_rest[object_mask] = base_rest[object_mask] + delta_sh_rest

    shs = torch.cat((mod_dc, mod_rest), dim=1)     # [N, 16, 3]

    # -- Set up rasterizer (mirrors gaussian_renderer.render) --
    screenspace_points = torch.zeros_like(means3D, requires_grad=True, device='cuda') + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(view.image_height),
        image_width=int(view.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=view.world_view_transform.cuda(),
        projmatrix=view.full_proj_transform.cuda(),
        sh_degree=gaussians.active_sh_degree,
        campos=view.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
    )
    time = torch.tensor(view.time).to(means3D.device).repeat(N, 1)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    opacity = gaussians._opacity.detach()
    scales = gaussians._scaling.detach()
    rotations = gaussians._rotation.detach()

    # -- Deformation (time-dependent) --
    # NOTE: Deformation network params must be frozen before calling this function
    # (done once in optimize()) so backward() only computes grads for delta_sh,
    # not for deformation weights. The deformation still runs in forward mode —
    # it transforms SH coefficients based on time — but its weights are fixed.
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = \
        gaussians._deformation(means3D.detach(), scales, rotations, opacity, shs, time)

    scales_final = gaussians.scaling_activation(scales_final)
    rotations_final = gaussians.rotation_activation(rotations_final)
    opacity_final = gaussians.opacity_activation(opacity_final)

    mask = torch.zeros((N, 1), dtype=torch.float, device='cuda')

    rendered_image, _, _, _ = rasterizer(
        means3D=means3D_final,
        means2D=screenspace_points,
        shs=shs_final,
        colors_precomp=None,
        opacities=opacity_final,
        mask=mask,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=None,
    )

    return rendered_image  # [3, H, W]


def train_step(view_idx, frame_idx, view, gaussians, pipe, background,
               delta_sh_dc, delta_sh_rest, object_mask, optimizer,
               targets, masks_2d, reg_weight=0.01, lpips_fn=None, lpips_weight=0.1):
    """
    One optimization step: render, compute loss, backprop.

    Args:
        view_idx, frame_idx: indices into the targets dict
        view:                camera object
        gaussians:           GaussianModel
        pipe, background:    rendering params
        delta_sh_dc/rest:    learnable SH residuals
        object_mask:         bool tensor [N_gaussians]
        optimizer:           Adam optimizer
        targets:             dict {(v_idx, f_idx): [1, 3, H, W]}
        masks_2d:            dict {(v_idx, f_idx): [1, 1, H, W]}
        reg_weight:          L2 regularization weight on delta_sh
        lpips_fn:            optional LPIPS loss function
        lpips_weight:        weight for LPIPS loss

    Returns:
        loss_val: float, the total loss value
    """
    optimizer.zero_grad()

    rendered = render_with_delta_sh(
        view, gaussians, pipe, background,
        delta_sh_dc, delta_sh_rest, object_mask)

    target = targets[(view_idx, frame_idx)].squeeze(0)  # [3, H, W]
    mask_2d = masks_2d[(view_idx, frame_idx)].squeeze(0)  # [1, H, W]

    # Masked L1 loss — only on the object region
    masked_rendered = rendered * mask_2d
    masked_target = target * mask_2d
    loss_l1 = F.l1_loss(masked_rendered, masked_target)

    total_loss = loss_l1

    # Optional perceptual loss
    if lpips_fn is not None:
        loss_lpips = lpips_fn(
            masked_rendered.unsqueeze(0),
            masked_target.unsqueeze(0)
        ).mean()
        total_loss = total_loss + lpips_weight * loss_lpips

    # Regularization: prevent delta_sh from growing too large
    reg_loss = reg_weight * (delta_sh_dc.pow(2).mean() + delta_sh_rest.pow(2).mean())
    total_loss = total_loss + reg_loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def optimize(gaussians, scene, pipe, background, mask_data,
             targets, composites, masks_2d,
             num_iterations=500, lr=1e-3, reg_weight=0.01,
             use_lpips=False, lpips_weight=0.1,
             log_interval=50):
    """
    Full SH optimization loop.

    Args:
        gaussians:      GaussianModel
        scene:          Scene object
        pipe:           PipelineParams
        background:     background tensor
        mask_data:      dict from load_mask_table()
        targets:        dict from precompute_all_targets()
        composites:     dict from precompute_all_targets()
        masks_2d:       dict from precompute_all_targets()
        num_iterations: number of optimization steps
        lr:             learning rate
        reg_weight:     L2 regularization weight
        use_lpips:      whether to use perceptual loss
        lpips_weight:   weight for LPIPS
        log_interval:   print loss every N iterations

    Returns:
        delta_sh_dc:   optimized tensor [N_object, 1, 3]
        delta_sh_rest: optimized tensor [N_object, 15, 3]
        object_mask:   bool tensor [N_gaussians]
    """
    from pipeline.data_loading import get_object_mask

    # Setup
    object_mask = get_object_mask(mask_data)
    delta_sh_dc, delta_sh_rest, optimizer, object_mask = create_delta_sh(
        gaussians, object_mask, lr=lr)

    views = scene.getTrainCameras()

    # Build list of valid (view_idx, frame_idx) pairs that have targets
    valid_keys = list(targets.keys())
    if not valid_keys:
        raise ValueError("No targets to optimize against. Run precompute first.")

    # Build a lookup from valid keys to camera objects
    key_to_view = {}
    for (v_idx, f_idx) in valid_keys:
        if v_idx < len(views):
            key_to_view[(v_idx, f_idx)] = views[v_idx]

    valid_keys = [k for k in valid_keys if k in key_to_view]

    # Optional LPIPS
    lpips_fn = None
    if use_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').cuda().eval()

    # Freeze deformation network: we want gradients to flow through its
    # forward pass (so delta_sh → deformed shs → rasterizer stays differentiable)
    # but NOT accumulate on the deformation weights themselves.
    print("[optimize] Freezing deformation network parameters")
    for param in gaussians._deformation.parameters():
        param.requires_grad_(False)

    print(f"[optimize] Starting SH optimization: {num_iterations} iters, "
          f"lr={lr}, {len(valid_keys)} (view,frame) pairs")

    # Training loop
    losses = []
    try:
        for iteration in tqdm(range(num_iterations), desc="Optimizing SH"):
            # Sample a random (view, frame) pair
            v_idx, f_idx = random.choice(valid_keys)
            view = key_to_view[(v_idx, f_idx)]

            loss_val = train_step(
                v_idx, f_idx, view, gaussians, pipe, background,
                delta_sh_dc, delta_sh_rest, object_mask, optimizer,
                targets, masks_2d, reg_weight=reg_weight,
                lpips_fn=lpips_fn, lpips_weight=lpips_weight)

            losses.append(loss_val)

            if (iteration + 1) % log_interval == 0:
                avg_loss = sum(losses[-log_interval:]) / log_interval
                print(f"  iter {iteration+1}/{num_iterations}  loss={avg_loss:.6f}")
    finally:
        # Always restore deformation network grad state, even on error/interrupt
        for param in gaussians._deformation.parameters():
            param.requires_grad_(True)
        print("[optimize] Restored deformation network parameters")

    final_avg = sum(losses[-min(50, len(losses)):]) / min(50, len(losses))
    print(f"[optimize] Done. Final avg loss: {final_avg:.6f}")

    return delta_sh_dc, delta_sh_rest, object_mask


def apply_delta_sh(gaussians, delta_sh_dc, delta_sh_rest, object_mask):
    """
    Permanently bake the optimized delta_sh into the Gaussian model.

    After this call, gaussians._features_dc and _features_rest contain
    the harmonized SH coefficients.

    Args:
        gaussians:     GaussianModel
        delta_sh_dc:   tensor [N_object, 1, 3]
        delta_sh_rest: tensor [N_object, 15, 3]
        object_mask:   bool tensor [N_gaussians]
    """
    with torch.no_grad():
        gaussians._features_dc.data[object_mask] += delta_sh_dc.data
        gaussians._features_rest.data[object_mask] += delta_sh_rest.data

    print(f"[optimize] Applied delta_sh to {object_mask.sum().item()} Gaussians")


def save_harmonized_ply(gaussians, output_path):
    """
    Save the harmonized Gaussian model as a .ply file.

    Args:
        gaussians:   GaussianModel (with delta_sh already applied)
        output_path: path to save the .ply file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gaussians.save_ply(output_path)
    print(f"[optimize] Saved harmonized PLY to {output_path}")
