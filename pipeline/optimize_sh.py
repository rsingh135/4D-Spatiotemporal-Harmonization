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


def create_delta_sh(gaussians, object_mask, lr=1e-3, lr_dc=None, lr_rest=None):
    """
    Create learnable SH residual tensors for the object Gaussians.

    The residuals are zero-initialized and will be optimized to harmonize
    the object's appearance.

    Args:
        gaussians:    GaussianModel
        object_mask:  bool tensor [N_gaussians] — True for object Gaussians
        lr:           learning rate for Adam (used if lr_dc/lr_rest are None)
        lr_dc:        optional learning rate for DC term (coarse brightness/color)
        lr_rest:      optional learning rate for higher-order SH terms (directional detail)

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

    lr_dc_eff = float(lr if lr_dc is None else lr_dc)
    lr_rest_eff = float(lr if lr_rest is None else lr_rest)
    optimizer = torch.optim.Adam(
        [
            {"params": [delta_sh_dc], "lr": lr_dc_eff, "name": "delta_sh_dc"},
            {"params": [delta_sh_rest], "lr": lr_rest_eff, "name": "delta_sh_rest"},
        ]
    )

    print(f"[optimize] Created delta_sh: {n_object} object Gaussians, "
          f"dc={list(dc_shape)}, rest={list(rest_shape)}, "
          f"lr_dc={lr_dc_eff:g}, lr_rest={lr_rest_eff:g}")
    return delta_sh_dc, delta_sh_rest, optimizer, object_mask


def init_shadow_pack(
    gaussians,
    object_mask: torch.Tensor,
    *,
    n_shadow: int = 2048,
    down_scale: float = 0.25,
    down_offset: float = 0.002,
    base_opacity: float = 0.12,
    lr_shadow: float = 5e-3,
):
    """
    Initialize a small "shadow plate" as extra Gaussians appended during rendering.

    The shadow is represented as learnable full SH coefficients (16x3) for n_shadow points,
    sampled near the support plane under the object (heuristic).

    Returns:
      shadow_pack dict passed to render_with_delta_sh / train_step
      shadow_optimizer
    """
    with torch.no_grad():
        obj_idx = torch.nonzero(object_mask, as_tuple=False).squeeze(1)
        if obj_idx.numel() == 0:
            raise ValueError("object_mask has zero True entries; cannot initialize shadow.")

        # Subsample object points for shadow anchors
        if obj_idx.numel() > n_shadow * 10:
            perm = torch.randperm(obj_idx.numel(), device=obj_idx.device)[: n_shadow * 10]
            pool = obj_idx[perm]
        else:
            pool = obj_idx

        n_pick = int(min(int(n_shadow), int(pool.numel())))
        pick = pool[torch.randperm(pool.numel(), device=pool.device)[:n_pick]]

        xyz_obj = gaussians.get_xyz[pick]  # [Ns,3]
        n = torch.tensor([0.0, 1.0, 0.0], device=xyz_obj.device, dtype=xyz_obj.dtype)  # "up"
        # Push slightly along -Y under object (works well for tabletop scenes); offset is tiny in world units.
        xyz_s = xyz_obj - n.view(1, 3) * float(down_offset)

        # Shadow scales: wide in XZ, thin in Y
        s0 = torch.log(torch.tensor([down_scale, down_scale * 0.15, down_scale], device=xyz_s.device, dtype=xyz_s.dtype)).view(1, 3).repeat(xyz_s.shape[0], 1)
        r0 = torch.zeros((xyz_s.shape[0], 4), device=xyz_s.device, dtype=xyz_s.dtype)
        r0[:, 0] = 1.0

        # Opacity in pre-activation space (logit)
        p = float(base_opacity)
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        o0 = torch.log(torch.tensor(p / (1.0 - p), device=xyz_s.device, dtype=xyz_s.dtype)).view(1, 1).repeat(xyz_s.shape[0], 1)

        # Initialize SH to a dark neutral color (learnable)
        base_dc = gaussians._features_dc.detach()[pick]          # [Ns,1,3]
        base_rest = gaussians._features_rest.detach()[pick]     # [Ns,15,3]
        dark = -0.35
        sh0 = torch.cat((base_dc + dark, base_rest), dim=1).clone()  # [Ns,16,3]

    shadow_sh = sh0.clone().detach().requires_grad_(True)
    shadow_xyz = xyz_s.clone().detach().requires_grad_(True)
    shadow_scale = s0.clone().detach().requires_grad_(True)
    shadow_rot = r0.clone().detach().requires_grad_(True)
    shadow_opacity = o0.clone().detach().requires_grad_(True)

    shadow_optimizer = torch.optim.Adam(
        [
            {"params": [shadow_sh], "lr": float(lr_shadow), "name": "shadow_sh"},
            {"params": [shadow_xyz], "lr": float(lr_shadow) * 0.1, "name": "shadow_xyz"},
            {"params": [shadow_scale], "lr": float(lr_shadow) * 0.1, "name": "shadow_scale"},
            {"params": [shadow_rot], "lr": float(lr_shadow) * 0.05, "name": "shadow_rot"},
            {"params": [shadow_opacity], "lr": float(lr_shadow) * 0.05, "name": "shadow_opacity"},
        ]
    )

    pack = {
        "shadow_sh": shadow_sh,
        "shadow_xyz": shadow_xyz,
        "shadow_scale": shadow_scale,
        "shadow_rot": shadow_rot,
        "shadow_opacity": shadow_opacity,
    }
    print(f"[optimize] Initialized shadow plate: n={pick.numel()}, lr_shadow={lr_shadow:g}")
    return pack, shadow_optimizer


def render_with_delta_sh(view, gaussians, pipe, background,
                         delta_sh_dc, delta_sh_rest, object_mask,
                         shadow_pack=None):
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

    # Optional shadow Gaussians appended for rendering/optimization
    if shadow_pack is not None:
        shs_s = shadow_pack["shadow_sh"]  # [Ns,16,3]
        shs = torch.cat([shs, shs_s], dim=0)
        means3D = torch.cat([means3D, shadow_pack["shadow_xyz"]], dim=0)

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
    time = torch.tensor(view.time).to(means3D.device).repeat(means3D.shape[0], 1)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    opacity = gaussians._opacity.detach()
    scales = gaussians._scaling.detach()
    rotations = gaussians._rotation.detach()

    if shadow_pack is not None:
        opacity = torch.cat([opacity, shadow_pack["shadow_opacity"]], dim=0)
        scales = torch.cat([scales, shadow_pack["shadow_scale"]], dim=0)
        rotations = torch.cat([rotations, shadow_pack["shadow_rot"]], dim=0)

    # -- Deformation (time-dependent) --
    # NOTE: Deformation network params must be frozen before calling this function
    # (done once in optimize()) so backward() only computes grads for delta_sh,
    # not for deformation weights. The deformation still runs in forward mode —
    # it transforms SH coefficients based on time — but its weights are fixed.
    # Foreground gaussians (from a different scene) skip deformation.
    dp = getattr(gaussians, '_deformation_table', None)
    if shadow_pack is None:
        Ntot = N
    else:
        Ntot = means3D.shape[0]

    if dp is not None and not dp.all():
        dp = dp.bool()
        if shadow_pack is not None:
            if dp.shape[0] != N:
                raise ValueError(f"deformation_table length {dp.shape[0]} != base N {N} (unexpected).")
            dp_ext = torch.cat([dp, torch.zeros((Ntot - N,), device=dp.device, dtype=torch.bool)], dim=0)
        else:
            dp_ext = dp

        means3D_final = means3D.detach().clone()
        scales_final = scales.clone()
        rotations_final = rotations.clone()
        opacity_final = opacity.clone()
        shs_final = shs.clone()
        if dp_ext.any():
            m_d, s_d, r_d, o_d, sh_d = gaussians._deformation(
                means3D.detach()[dp_ext], scales[dp_ext], rotations[dp_ext], opacity[dp_ext], shs[dp_ext], time[dp_ext])
            means3D_final[dp_ext] = m_d
            scales_final[dp_ext] = s_d
            rotations_final[dp_ext] = r_d
            opacity_final[dp_ext] = o_d
            shs_final[dp_ext] = sh_d
    else:
        # Original behavior: deform everything — but if we appended shadow points, keep them static.
        if shadow_pack is None:
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = \
                gaussians._deformation(means3D.detach(), scales, rotations, opacity, shs, time)
        else:
            means3D_final = means3D.detach().clone()
            scales_final = scales.clone()
            rotations_final = rotations.clone()
            opacity_final = opacity.clone()
            shs_final = shs.clone()
            base_time = time[:N]
            m_d, s_d, r_d, o_d, sh_d = gaussians._deformation(
                means3D.detach()[:N], scales[:N], rotations[:N], opacity[:N], shs[:N], base_time)
            means3D_final[:N] = m_d
            scales_final[:N] = s_d
            rotations_final[:N] = r_d
            opacity_final[:N] = o_d
            shs_final[:N] = sh_d

    scales_final = gaussians.scaling_activation(scales_final)
    rotations_final = gaussians.rotation_activation(rotations_final)
    opacity_final = gaussians.opacity_activation(opacity_final)

    mask = torch.zeros((means3D_final.shape[0], 1), dtype=torch.float, device='cuda')

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
               targets, masks_2d, reg_weight=0.01, lpips_fn=None, lpips_weight=0.1,
               shadow_pack=None, shadow_optimizer=None,
               shadow_reg_weight=0.01, shadow_outside_weight=0.05,
               mask_core_erode_px: int = 0,
               mask_boundary_weight: float = 0.25,
               mask_weight_power: float = 1.0):
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
    if shadow_optimizer is not None:
        shadow_optimizer.zero_grad(set_to_none=True)

    rendered = render_with_delta_sh(
        view, gaussians, pipe, background,
        delta_sh_dc, delta_sh_rest, object_mask,
        shadow_pack=shadow_pack)

    target = targets[(view_idx, frame_idx)].squeeze(0)  # [3, H, W]
    mask_2d = masks_2d[(view_idx, frame_idx)].squeeze(0)  # [1, H, W]

    # Weighted L1 inside mask. Core/boundary weighting helps reduce edge halos:
    # - define a binary mask via thr=0.5
    # - optionally erode to get a "core" region (full weight)
    # - boundary band gets smaller weight
    mask_bin = (mask_2d > 0.5).float()
    if mask_core_erode_px is not None and int(mask_core_erode_px) > 0:
        r = int(mask_core_erode_px)
        inv = (1.0 - mask_bin).unsqueeze(0)  # [1,1,H,W]
        dil = F.max_pool2d(inv, kernel_size=2 * r + 1, stride=1, padding=r)
        core = (1.0 - dil).squeeze(0)  # [1,H,W]
        core = (core > 0.5).float()
        boundary = (mask_bin - core).clamp(0.0, 1.0)
        w = core + boundary * float(mask_boundary_weight)
    else:
        w = mask_bin

    # Multiply by (possibly soft) mask_2d to keep soft edges if feathering is enabled.
    w = (w * mask_2d.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    if mask_weight_power is not None and abs(float(mask_weight_power) - 1.0) > 1e-6:
        w = w.pow(float(mask_weight_power))

    diff = (rendered - target).abs()
    denom = w.sum().clamp_min(1e-6) * 3.0
    loss_l1 = (diff * w).sum() / denom

    # Keep these around for LPIPS / visualization parity
    masked_rendered = rendered * mask_2d
    masked_target = target * mask_2d

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

    # Shadow regularizers (keep subtle + discourage energy outside object mask)
    if shadow_pack is not None:
        # L2 on shadow SH
        total_loss = total_loss + float(shadow_reg_weight) * shadow_pack["shadow_sh"].pow(2).mean()
        # Outside-mask penalty on rendered image (shadow should mostly live under object)
        outside = (1.0 - mask_2d)
        total_loss = total_loss + float(shadow_outside_weight) * (rendered * outside).abs().mean()

    total_loss.backward()
    optimizer.step()
    if shadow_optimizer is not None:
        shadow_optimizer.step()

    return total_loss.item()


def optimize(gaussians, scene, pipe, background, mask_data,
             targets, composites, masks_2d,
             num_iterations=500, lr=1e-3, lr_dc=None, lr_rest=None, reg_weight=0.01,
             use_lpips=False, lpips_weight=0.1,
             log_interval=50,
             shadow_mode: str = "off",
             shadow_n: int = 2048,
             shadow_lr: float = 5e-3,
             shadow_reg_weight: float = 0.01,
             shadow_outside_weight: float = 0.05,
             mask_core_erode_px: int = 0,
             mask_boundary_weight: float = 0.25,
             mask_weight_power: float = 1.0):
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
        lr_dc:          optional learning rate for DC residual tensor
        lr_rest:        optional learning rate for SH rest residual tensor
        reg_weight:     L2 regularization weight
        use_lpips:      whether to use perceptual loss
        lpips_weight:   weight for LPIPS
        log_interval:   print loss every N iterations

    Returns:
        delta_sh_dc:   optimized tensor [N_object, 1, 3]
        delta_sh_rest: optimized tensor [N_object, 15, 3]
        object_mask:   bool tensor [N_gaussians]
        losses:        list[float]
        shadow_pack:    dict or None (learned shadow parameters if enabled)
    """
    from pipeline.data_loading import get_object_mask

    # Setup
    object_mask = get_object_mask(mask_data)
    delta_sh_dc, delta_sh_rest, optimizer, object_mask = create_delta_sh(
        gaussians, object_mask, lr=lr, lr_dc=lr_dc, lr_rest=lr_rest)

    shadow_pack = None
    shadow_optimizer = None
    if shadow_mode not in ("off", "learned"):
        raise ValueError(f"Unknown shadow_mode={shadow_mode}")
    if shadow_mode == "learned":
        shadow_pack, shadow_optimizer = init_shadow_pack(
            gaussians, object_mask, n_shadow=int(shadow_n), lr_shadow=float(shadow_lr))

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

    lr_dc_eff = float(lr if lr_dc is None else lr_dc)
    lr_rest_eff = float(lr if lr_rest is None else lr_rest)
    print(f"[optimize] Starting SH optimization: {num_iterations} iters, "
          f"lr_dc={lr_dc_eff:g}, lr_rest={lr_rest_eff:g}, {len(valid_keys)} (view,frame) pairs")

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
                lpips_fn=lpips_fn, lpips_weight=lpips_weight,
                shadow_pack=shadow_pack, shadow_optimizer=shadow_optimizer,
                shadow_reg_weight=shadow_reg_weight, shadow_outside_weight=shadow_outside_weight,
                mask_core_erode_px=mask_core_erode_px,
                mask_boundary_weight=mask_boundary_weight,
                mask_weight_power=mask_weight_power)

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

    return delta_sh_dc, delta_sh_rest, object_mask, losses, shadow_pack


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


@torch.no_grad()
def bake_shadow_pack_into_gaussians(gaussians, shadow_pack):
    """
    Append optimized shadow Gaussians into the underlying Gaussian tensors so they persist in .ply.

    This updates:
      _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation
    and extends _deformation_table (if present) with False for appended points.
    """
    if shadow_pack is None:
        return

    sh = shadow_pack["shadow_sh"]  # [Ns,16,3]
    Ns = sh.shape[0]
    dc = sh[:, :1, :].contiguous()
    rest = sh[:, 1:, :].contiguous()

    xyz = shadow_pack["shadow_xyz"].detach()
    op = shadow_pack["shadow_opacity"].detach()
    sc = shadow_pack["shadow_scale"].detach()
    rot = shadow_pack["shadow_rot"].detach()

    gaussians._xyz = torch.nn.Parameter(torch.cat([gaussians._xyz.detach(), xyz], dim=0).requires_grad_(True))
    gaussians._features_dc = torch.nn.Parameter(torch.cat([gaussians._features_dc.detach(), dc], dim=0).requires_grad_(True))
    gaussians._features_rest = torch.nn.Parameter(torch.cat([gaussians._features_rest.detach(), rest], dim=0).requires_grad_(True))
    gaussians._opacity = torch.nn.Parameter(torch.cat([gaussians._opacity.detach(), op], dim=0).requires_grad_(True))
    gaussians._scaling = torch.nn.Parameter(torch.cat([gaussians._scaling.detach(), sc], dim=0).requires_grad_(True))
    gaussians._rotation = torch.nn.Parameter(torch.cat([gaussians._rotation.detach(), rot], dim=0).requires_grad_(True))

    if hasattr(gaussians, "_deformation_table") and torch.is_tensor(gaussians._deformation_table):
        dt = gaussians._deformation_table
        if dt.shape[0] != gaussians._xyz.shape[0] - Ns:
            # If mismatch, rebuild a safe default: deform all old points, keep new static
            gaussians._deformation_table = torch.ones((gaussians._xyz.shape[0] - Ns,), device="cuda", dtype=torch.bool)
            dt = gaussians._deformation_table
        ext = torch.zeros((Ns,), device=dt.device, dtype=torch.bool)
        gaussians._deformation_table = torch.cat([dt, ext], dim=0)

    print(f"[optimize] Baked shadow Gaussians into model: +{Ns} points (total {gaussians._xyz.shape[0]})")


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
