#!/usr/bin/env python3
"""
Difix3D+-style SH distillation onto a composite 4DGS scene.

This is the "static Difix3D" pipeline:
  1. Load the BG cookie 4DGS scene + composite PLY (BG + inserted FG)
  2. For a subset of (camera, frame) pairs, render the composite via the existing
     differentiable rasteriser
  3. Run nvidia/difix on each rendered image to get a cleaned target
  4. Optimise per-Gaussian SH residuals (delta_sh) on the FG (and optionally BG)
     against the cached Difix targets
  5. Bake delta_sh into the Gaussian model and save a new PLY
  6. Re-render the full video

This is the same pattern as ``pipeline/run_harmonize.py`` but with the Harmonizer
CNN replaced by the single-step Difix diffusion model. Compared to the actual
Difix3D+ gsplat trainer this version:
  * skips iterative novel-view fix loops (we use only training poses)
  * does SH-only optimisation (no positions/scales/opacities)
  * preserves 4D dynamics (uses the 4DGS deformation network at render time)

Run::

    python -m pipeline.difix_distill \\
        --model_path output/hypernerf/split-cookie \\
        --source_path data/hypernerf/split-cookie \\
        --composite_ply output/hypernerf/split-cookie/point_cloud/iteration_14000/clean_chocolate_Bigger.ply \\
        --fg_mask output/hypernerf/split-cookie/segment_results/composite_inserted_choc_Bigger.pt \\
        --output_ply output/hypernerf/split-cookie/point_cloud/iteration_14000/difix_distilled_chocBigger.ply \\
        --output_video output/composite_torchocolateBigger/composite_cookie_chocBigger_difix3dplus.mp4 \\
        --difix_env /home/ubuntu/miniconda3/envs/difix3d/bin/python \\
        --num_frames 30 --num_iterations 800 --opt_target fg_only
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)

to8b = lambda x: (255 * np.clip(x.detach().cpu().numpy(), 0, 1)).astype(np.uint8)


# ============================================================================
# Stage 1 — load composite into BG scene
# ============================================================================

def load_composite_scene(model_path, source_path, composite_ply, fg_mask_pt,
                         deformation_mode="foreground_static",
                         configs=None, iteration=-1):
    """Load BG cookie 4DGS, swap in the composite PLY, set deformation_table."""
    from pipeline.data_loading import load_scene, load_mask_table

    print(f"[difix_distill] Loading BG scene at {model_path} ...")
    gaussians, scene, pipe, background = load_scene(
        os.path.abspath(model_path), os.path.abspath(source_path),
        iteration=iteration, configs=configs,
    )
    print(f"[difix_distill] BG-only Gaussians: {gaussians._xyz.shape[0]}")

    print(f"[difix_distill] Loading composite PLY: {composite_ply}")
    gaussians.load_ply(os.path.abspath(composite_ply))
    print(f"[difix_distill] Composite Gaussians: {gaussians._xyz.shape[0]}")

    mask_data = load_mask_table(os.path.abspath(fg_mask_pt))
    fg_mask = mask_data["mask_table"].any(dim=0).to("cuda").bool()
    n_g = gaussians._xyz.shape[0]
    if fg_mask.shape[0] != n_g:
        raise ValueError(
            f"FG mask length {fg_mask.shape[0]} != composite Gaussians {n_g}. "
            "The mask must match the composite PLY."
        )
    print(f"[difix_distill] FG Gaussians: {int(fg_mask.sum().item())}, "
          f"BG Gaussians: {int((~fg_mask).sum().item())}")

    if deformation_mode == "foreground_static":
        # BG deforms, FG stays at canonical (~render_4dgs --composite default)
        gaussians._deformation_table = (~fg_mask).clone()
    elif deformation_mode == "foreground_moves":
        gaussians._deformation_table = fg_mask.clone()
    elif deformation_mode == "all_static":
        gaussians._deformation_table = torch.zeros(n_g, dtype=torch.bool, device="cuda")
    else:
        raise ValueError(f"Unknown deformation_mode={deformation_mode}")
    print(f"[difix_distill] deformation_table: {int(gaussians._deformation_table.sum().item())} "
          f"deform / {int((~gaussians._deformation_table).sum().item())} static "
          f"(mode={deformation_mode})")

    return gaussians, scene, pipe, background, fg_mask


# ============================================================================
# Stage 2 — pick (cam, time) pairs, render composite, run Difix, cache targets
# ============================================================================

def pick_view_frame_pairs(scene, mask_data, num_frames=30, num_views_per_frame=1, seed=0):
    """Return a list of (view_idx, frame_idx) pairs spread evenly across time."""
    train_cams = scene.getTrainCameras()
    n_views = len(train_cams)
    if "time_map" in mask_data:
        n_frames = mask_data["mask_table"].shape[0]
    else:
        n_frames = n_views
    print(f"[difix_distill] Train cameras: {n_views}  |  mask frames: {n_frames}")

    rng = random.Random(seed)
    if num_frames >= n_views:
        view_idxs = list(range(n_views))
    else:
        # Evenly spaced view indices with a small jitter
        step = n_views / num_frames
        view_idxs = [int(min(n_views - 1, round(i * step + rng.uniform(-0.3, 0.3) * step)))
                     for i in range(num_frames)]
        # Deduplicate while preserving order
        seen = set()
        view_idxs = [v for v in view_idxs if not (v in seen or seen.add(v))]

    pairs = []
    for v in view_idxs:
        cam = train_cams[v]
        # Pick the mask frame closest to this camera's time
        if "time_map" in mask_data:
            tm = mask_data["time_map"]
            f = int((tm - float(cam.time)).abs().argmin().item())
        else:
            f = v % n_frames
        pairs.append((v, f))
    return pairs, train_cams


def render_composite_at(view, gaussians, pipe, background, cam_type="hypernerf"):
    """Render composite scene at a single (view, frame) via existing renderer."""
    from gaussian_renderer import render
    with torch.no_grad():
        out = render(view, gaussians, pipe, background, cam_type=cam_type)
    return out["render"].clamp(0, 1)  # [3, H, W]


def render_and_dump_inputs(pairs, train_cams, gaussians, pipe, background,
                           dump_dir, cam_type="hypernerf"):
    os.makedirs(dump_dir, exist_ok=True)
    paths = []
    for i, (v_idx, f_idx) in enumerate(tqdm(pairs, desc="Render composite frames")):
        view = train_cams[v_idx]
        rendered = render_composite_at(view, gaussians, pipe, background, cam_type=cam_type)
        rgb = to8b(rendered).transpose(1, 2, 0)
        path = os.path.join(dump_dir, f"frame_{i:05d}_v{v_idx:04d}_f{f_idx:04d}.png")
        Image.fromarray(rgb).save(path, optimize=True)
        paths.append(path)
    return paths


def run_difix_on_dir(input_dir, output_dir, difix_python, run_difix_script,
                    diffusion_h=576, diffusion_w=1024, rotate_portrait=True):
    """Shell out to the difix3d env and run our existing per-frame runner."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        difix_python, run_difix_script,
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--diffusion-height", str(diffusion_h),
        "--diffusion-width", str(diffusion_w),
    ]
    if rotate_portrait:
        cmd.append("--rotate-portrait")
    print(f"[difix_distill] Running Difix subprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=True)


# ============================================================================
# Stage 3 — set up SH residuals and optimisation
# ============================================================================

def init_delta_sh(gaussians, opt_target, fg_mask):
    """Create learnable SH residuals on either FG-only, BG-only, or all Gaussians."""
    if opt_target == "fg_only":
        opt_mask = fg_mask
    elif opt_target == "bg_only":
        opt_mask = ~fg_mask
    elif opt_target == "all":
        opt_mask = torch.ones_like(fg_mask)
    else:
        raise ValueError(f"Unknown opt_target={opt_target}")

    n_opt = int(opt_mask.sum().item())
    dc_shape = gaussians._features_dc[opt_mask].shape
    rest_shape = gaussians._features_rest[opt_mask].shape
    delta_dc = torch.zeros(dc_shape, device="cuda", requires_grad=True)
    delta_rest = torch.zeros(rest_shape, device="cuda", requires_grad=True)
    print(f"[difix_distill] delta_sh: {n_opt} Gaussians "
          f"(dc={list(dc_shape)}, rest={list(rest_shape)}, target={opt_target})")
    return delta_dc, delta_rest, opt_mask


def render_with_delta_sh(view, gaussians, pipe, background,
                         delta_dc, delta_rest, opt_mask, cam_type="hypernerf"):
    """Re-render the composite with delta_sh added to opt_mask Gaussians,
    keeping the autograd graph alive so backprop reaches delta_dc / delta_rest."""
    import math
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer)

    means3D = gaussians.get_xyz
    N = means3D.shape[0]

    base_dc = gaussians._features_dc.detach()      # [N, 1, 3]
    base_rest = gaussians._features_rest.detach()   # [N, 15, 3]

    mod_dc = base_dc.clone()
    mod_rest = base_rest.clone()
    mod_dc[opt_mask] = base_dc[opt_mask] + delta_dc
    mod_rest[opt_mask] = base_rest[opt_mask] + delta_rest
    shs = torch.cat((mod_dc, mod_rest), dim=1)     # [N, 16, 3]

    screenspace_points = torch.zeros_like(means3D, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(view.image_height),
        image_width=int(view.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=background, scale_modifier=1.0,
        viewmatrix=view.world_view_transform.cuda(),
        projmatrix=view.full_proj_transform.cuda(),
        sh_degree=gaussians.active_sh_degree,
        campos=view.camera_center.cuda(),
        prefiltered=False, debug=getattr(pipe, "debug", False),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    opacity = gaussians._opacity.detach()
    scales = gaussians._scaling.detach()
    rotations = gaussians._rotation.detach()
    time_t = torch.tensor(view.time).to(means3D.device).repeat(means3D.shape[0], 1)

    dp = getattr(gaussians, "_deformation_table", None)
    if dp is not None and not dp.all():
        dp = dp.bool()
        means3D_final = means3D.detach().clone()
        scales_final = scales.clone()
        rotations_final = rotations.clone()
        opacity_final = opacity.clone()
        shs_final = shs.clone()
        if dp.any():
            m_d, s_d, r_d, o_d, sh_d = gaussians._deformation(
                means3D.detach()[dp], scales[dp], rotations[dp],
                opacity[dp], shs[dp], time_t[dp])
            means3D_final[dp] = m_d
            scales_final[dp] = s_d
            rotations_final[dp] = r_d
            opacity_final[dp] = o_d
            shs_final[dp] = sh_d
    else:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = \
            gaussians._deformation(means3D.detach(), scales, rotations, opacity, shs, time_t)

    scales_final = gaussians.scaling_activation(scales_final)
    rotations_final = gaussians.rotation_activation(rotations_final)
    opacity_final = gaussians.opacity_activation(opacity_final)
    mask = torch.zeros((means3D_final.shape[0], 1), dtype=torch.float, device="cuda")

    rendered, _, _, _ = rasterizer(
        means3D=means3D_final, means2D=screenspace_points,
        shs=shs_final, colors_precomp=None,
        opacities=opacity_final, mask=mask,
        scales=scales_final, rotations=rotations_final,
        cov3D_precomp=None,
    )
    return rendered


def optimise(gaussians, scene, pipe, background, pairs, target_paths,
             opt_target, fg_mask,
             num_iterations=800, lr_dc=2e-3, lr_rest=5e-4, reg_weight=1e-3,
             ssim_weight=0.0, log_interval=50, cam_type="hypernerf"):
    """SH residual optimisation against cached Difix targets."""
    delta_dc, delta_rest, opt_mask = init_delta_sh(gaussians, opt_target, fg_mask)
    optimizer = torch.optim.Adam(
        [
            {"params": [delta_dc], "lr": float(lr_dc), "name": "delta_dc"},
            {"params": [delta_rest], "lr": float(lr_rest), "name": "delta_rest"},
        ]
    )

    train_cams = scene.getTrainCameras()
    target_tensors = {}
    for (v_idx, f_idx), p in zip(pairs, target_paths):
        img = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
        target_tensors[(v_idx, f_idx)] = torch.from_numpy(img.transpose(2, 0, 1)).cuda()
    print(f"[difix_distill] Loaded {len(target_tensors)} Difix targets to GPU")

    # Freeze deformation network
    for p in gaussians._deformation.parameters():
        p.requires_grad_(False)

    losses = []
    try:
        for it in tqdm(range(num_iterations), desc="Distil delta_sh"):
            v_idx, f_idx = random.choice(pairs)
            view = train_cams[v_idx]

            optimizer.zero_grad()
            rendered = render_with_delta_sh(
                view, gaussians, pipe, background, delta_dc, delta_rest, opt_mask,
                cam_type=cam_type,
            )
            target = target_tensors[(v_idx, f_idx)]
            if target.shape[1:] != rendered.shape[1:]:
                target = F.interpolate(target.unsqueeze(0), size=rendered.shape[1:],
                                       mode="bilinear", align_corners=False).squeeze(0)
            loss_l1 = (rendered - target).abs().mean()
            reg = reg_weight * (delta_dc.pow(2).mean() + delta_rest.pow(2).mean())
            loss = loss_l1 + reg
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

            if (it + 1) % log_interval == 0:
                avg = sum(losses[-log_interval:]) / log_interval
                print(f"  iter {it+1}/{num_iterations}  loss={avg:.5f}  "
                      f"|delta_dc|={delta_dc.abs().mean().item():.4f}  "
                      f"|delta_rest|={delta_rest.abs().mean().item():.4f}")
    finally:
        for p in gaussians._deformation.parameters():
            p.requires_grad_(True)

    return delta_dc.detach(), delta_rest.detach(), opt_mask, losses


def apply_and_save(gaussians, delta_dc, delta_rest, opt_mask, output_ply, delta_pt):
    """Bake delta_sh into Gaussians and save a new PLY + raw delta tensors."""
    with torch.no_grad():
        gaussians._features_dc.data[opt_mask] += delta_dc
        gaussians._features_rest.data[opt_mask] += delta_rest
    os.makedirs(os.path.dirname(output_ply) or ".", exist_ok=True)
    gaussians.save_ply(output_ply)
    torch.save({
        "delta_sh_dc": delta_dc.cpu(),
        "delta_sh_rest": delta_rest.cpu(),
        "opt_mask": opt_mask.cpu(),
    }, delta_pt)
    print(f"[difix_distill] Saved distilled PLY: {output_ply}")
    print(f"[difix_distill] Saved delta_sh tensors: {delta_pt}")


# ============================================================================
# Stage 4 — re-render mp4 with the distilled PLY
# ============================================================================

def render_video(gaussians, scene, pipe, background, output_video, fps=30,
                 cam_type="hypernerf"):
    import imageio
    from gaussian_renderer import render

    views = scene.getVideoCameras()
    if not views:
        raise RuntimeError("Scene has no video cameras.")
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    frames = []
    for view in tqdm(views, desc="Render distilled video"):
        with torch.no_grad():
            r = render(view, gaussians, pipe, background, cam_type=cam_type)["render"]
        frames.append(to8b(r).transpose(1, 2, 0))
    imageio.mimwrite(output_video, frames, fps=float(fps))
    print(f"[difix_distill] Saved distilled video: {output_video}")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--composite_ply", required=True)
    ap.add_argument("--fg_mask", required=True)
    ap.add_argument("--output_ply", required=True)
    ap.add_argument("--output_video", required=True)
    ap.add_argument("--delta_pt", default=None,
                    help="Where to save the optimised delta_sh tensors. "
                         "Default: alongside output_ply with .pt extension.")
    ap.add_argument("--difix_env", default="/home/ubuntu/miniconda3/envs/difix3d/bin/python")
    ap.add_argument("--run_difix_script", default="/home/ubuntu/Difix3D/run_difix.py")
    ap.add_argument("--workdir", default=None,
                    help="Where to drop intermediate frame PNGs / Difix outputs.")
    ap.add_argument("--num_frames", type=int, default=30)
    ap.add_argument("--num_iterations", type=int, default=800)
    ap.add_argument("--lr_dc", type=float, default=2e-3)
    ap.add_argument("--lr_rest", type=float, default=5e-4)
    ap.add_argument("--reg_weight", type=float, default=1e-3)
    ap.add_argument("--opt_target", choices=["fg_only", "bg_only", "all"],
                    default="fg_only")
    ap.add_argument("--deformation_mode", choices=["foreground_static",
                                                    "foreground_moves", "all_static"],
                    default="foreground_static")
    ap.add_argument("--configs", default="arguments/hypernerf/default.py")
    ap.add_argument("--iteration", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--diffusion_h", type=int, default=576)
    ap.add_argument("--diffusion_w", type=int, default=1024)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.delta_pt is None:
        args.delta_pt = os.path.splitext(args.output_ply)[0] + "_delta_sh.pt"
    if args.workdir is None:
        out_dir_root = os.path.dirname(args.output_video) or "."
        args.workdir = os.path.join(out_dir_root, "difix_distill_workdir")
    raw_dir = os.path.join(args.workdir, "rendered")
    target_dir = os.path.join(args.workdir, "difix_targets")
    os.makedirs(args.workdir, exist_ok=True)

    # ---------------------------------------------------------------- Stage 1
    gaussians, scene, pipe, background, fg_mask = load_composite_scene(
        args.model_path, args.source_path, args.composite_ply, args.fg_mask,
        deformation_mode=args.deformation_mode,
        configs=args.configs, iteration=args.iteration,
    )
    cam_type = scene.dataset_type

    # Reload mask data for time_map
    from pipeline.data_loading import load_mask_table
    mask_data = load_mask_table(os.path.abspath(args.fg_mask))

    # ---------------------------------------------------------------- Stage 2
    pairs, train_cams = pick_view_frame_pairs(scene, mask_data,
                                              num_frames=args.num_frames, seed=args.seed)
    print(f"[difix_distill] Selected {len(pairs)} (view, frame) pairs.")
    raw_paths = render_and_dump_inputs(pairs, train_cams, gaussians, pipe, background,
                                       raw_dir, cam_type=cam_type)
    run_difix_on_dir(raw_dir, target_dir, args.difix_env, args.run_difix_script,
                     diffusion_h=args.diffusion_h, diffusion_w=args.diffusion_w,
                     rotate_portrait=True)
    target_paths = [os.path.join(target_dir, os.path.basename(p)) for p in raw_paths]
    missing = [p for p in target_paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"{len(missing)} Difix target images missing. e.g. {missing[0]}")

    # ---------------------------------------------------------------- Stage 3
    delta_dc, delta_rest, opt_mask, losses = optimise(
        gaussians, scene, pipe, background, pairs, target_paths,
        args.opt_target, fg_mask,
        num_iterations=args.num_iterations,
        lr_dc=args.lr_dc, lr_rest=args.lr_rest, reg_weight=args.reg_weight,
        cam_type=cam_type,
    )
    apply_and_save(gaussians, delta_dc, delta_rest, opt_mask, args.output_ply, args.delta_pt)

    # ---------------------------------------------------------------- Stage 4
    render_video(gaussians, scene, pipe, background, args.output_video,
                 fps=args.fps, cam_type=cam_type)
    print("[difix_distill] Done.")


if __name__ == "__main__":
    main()
