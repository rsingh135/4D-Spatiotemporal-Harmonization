#!/usr/bin/env python3
"""
Run DiFix3D+ distillation on the v5 dynamic Blender scene (Scene A).

Steps:
  1. Load the v5/dynamic_A 4DGS model
  2. Render training views, run DiFix on each frame
  3. Optimise DC-only SH residuals on the breakdancer mask against DiFix targets
  4. Save distilled PLY, render video, build side-by-side comparison

Usage:
    conda activate sa4d
    python scripts/difix_blender_v5.py
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)

to8b = lambda x: (255 * np.clip(x.detach().cpu().numpy(), 0, 1)).astype(np.uint8)

# ---------- defaults ----------
MODEL_PATH = "output/v5/dynamic_A"
SOURCE_PATH = "data_v5_dynamic/scene_A"
MASK_PATH = "output/v5/dynamic_A/segment_results/bd_mask.pt"
CONFIGS = "arguments/dnerf/joint_dynamic_50k.py"
ITERATION = 45000
DIFIX_PYTHON = "/home/ubuntu/miniconda3/envs/difix3d/bin/python"
DIFIX_SCRIPT = "/home/ubuntu/Difix3D/run_difix.py"
OUTPUT_DIR = "results_v5/dynamic/difix_distilled"
NUM_DIFIX_FRAMES = 30
NUM_OPT_ITERS = 500
LR_DC = 0.01
LR_REST = 0.0
REG_WEIGHT = 0.01


def load_scene_for_blender(model_path, source_path, configs, iteration):
    """Load the 4DGS scene using the standard pipeline loader."""
    from pipeline.data_loading import load_scene
    gaussians, scene, pipe, background = load_scene(
        os.path.abspath(model_path),
        os.path.abspath(source_path),
        iteration=iteration,
        configs=configs,
    )
    return gaussians, scene, pipe, background


def render_frame(view, gaussians, pipe, background, cam_type):
    from gaussian_renderer import render
    with torch.no_grad():
        out = render(view, gaussians, pipe, background, cam_type=cam_type)
    return out["render"].clamp(0, 1)


def render_training_views(scene, gaussians, pipe, background, cam_type,
                          num_frames, output_dir, seed=0):
    """Render a subset of training views and save as PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    train_cams = scene.getTrainCameras()
    n = len(train_cams)

    rng = random.Random(seed)
    if num_frames >= n:
        indices = list(range(n))
    else:
        step = n / num_frames
        indices = [int(min(n - 1, round(i * step))) for i in range(num_frames)]
        indices = list(dict.fromkeys(indices))

    paths = []
    for i, idx in enumerate(tqdm(indices, desc="Render train views")):
        view = train_cams[idx]
        rendered = render_frame(view, gaussians, pipe, background, cam_type)
        rgb = to8b(rendered).transpose(1, 2, 0)
        path = os.path.join(output_dir, f"frame_{i:05d}.png")
        Image.fromarray(rgb).save(path, optimize=True)
        paths.append(path)

    return indices, paths


def run_difix(input_dir, output_dir, difix_python, difix_script):
    """Run DiFix on a directory of PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [difix_python, difix_script, "--input-dir", input_dir, "--output-dir", output_dir, "--diffusion-height", "576", "--diffusion-width", "1024", "--rotate-portrait"]
    print(f"Running DiFix: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def render_with_delta_sh(view, gaussians, pipe, background, delta_dc, delta_rest,
                          opt_mask, cam_type):
    """Render with SH residuals applied, keeping autograd graph alive."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    means3D = gaussians.get_xyz
    N = means3D.shape[0]

    base_dc = gaussians._features_dc.detach()
    base_rest = gaussians._features_rest.detach()
    mod_dc = base_dc.clone()
    mod_rest = base_rest.clone()
    mod_dc[opt_mask] = base_dc[opt_mask] + delta_dc
    mod_rest[opt_mask] = base_rest[opt_mask] + delta_rest
    shs = torch.cat((mod_dc, mod_rest), dim=1)

    screenspace_points = torch.zeros_like(means3D, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(view.image_height), image_width=int(view.image_width),
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


def optimise_sh(gaussians, scene, pipe, background, view_indices, target_paths,
                fg_mask, cam_type, num_iterations, lr_dc, lr_rest, reg_weight):
    """Optimise SH residuals against DiFix targets."""
    opt_mask = fg_mask.bool()
    dc_shape = gaussians._features_dc[opt_mask].shape
    rest_shape = gaussians._features_rest[opt_mask].shape
    delta_dc = torch.zeros(dc_shape, device="cuda", requires_grad=True)
    delta_rest = torch.zeros(rest_shape, device="cuda", requires_grad=True)

    params = [{"params": [delta_dc], "lr": lr_dc, "name": "delta_dc"}]
    if lr_rest > 0:
        params.append({"params": [delta_rest], "lr": lr_rest, "name": "delta_rest"})
    optimizer = torch.optim.Adam(params)

    train_cams = scene.getTrainCameras()
    targets = {}
    for idx, path in zip(view_indices, target_paths):
        img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        targets[idx] = torch.from_numpy(img.transpose(2, 0, 1)).cuda()
    print(f"Loaded {len(targets)} DiFix targets")

    for p in gaussians._deformation.parameters():
        p.requires_grad_(False)

    losses = []
    pairs = list(zip(view_indices, target_paths))
    for it in tqdm(range(num_iterations), desc="Optimise SH"):
        idx = random.choice(view_indices)
        view = train_cams[idx]

        optimizer.zero_grad()
        rendered = render_with_delta_sh(view, gaussians, pipe, background,
                                         delta_dc, delta_rest, opt_mask, cam_type)
        target = targets[idx]
        if target.shape[1:] != rendered.shape[1:]:
            target = F.interpolate(target.unsqueeze(0), size=rendered.shape[1:],
                                    mode="bilinear", align_corners=False).squeeze(0)
        loss_l1 = (rendered - target).abs().mean()
        reg = reg_weight * delta_dc.pow(2).mean()
        if lr_rest > 0:
            reg = reg + reg_weight * delta_rest.pow(2).mean()
        loss = loss_l1 + reg
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if (it + 1) % 100 == 0:
            avg = sum(losses[-100:]) / 100
            print(f"  iter {it+1}/{num_iterations}  loss={avg:.5f}  "
                  f"|delta_dc|={delta_dc.abs().mean().item():.4f}")

    for p in gaussians._deformation.parameters():
        p.requires_grad_(True)

    return delta_dc.detach(), delta_rest.detach(), opt_mask, losses


def render_video(gaussians, scene, pipe, background, output_path, cam_type, fps=30):
    """Render video cameras and save as mp4."""
    from gaussian_renderer import render
    views = scene.getVideoCameras()
    if not views:
        print("No video cameras, using test cameras instead")
        views = scene.getTestCameras()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for view in tqdm(views, desc="Render video"):
        with torch.no_grad():
            r = render(view, gaussians, pipe, background, cam_type=cam_type)["render"]
        frames.append(to8b(r).transpose(1, 2, 0))
    imageio.mimwrite(output_path, frames, fps=float(fps), quality=8, macro_block_size=1)
    print(f"Saved video: {output_path} ({len(frames)} frames)")
    return frames


def build_comparison_video(original_dir, distilled_frames, output_path, fps=30):
    """Build side-by-side comparison: original | distilled."""
    import glob
    orig_files = sorted(glob.glob(os.path.join(original_dir, "*.png")))
    n = min(len(orig_files), len(distilled_frames))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison_frames = []
    for i in range(n):
        orig = np.array(Image.open(orig_files[i]))
        dist = distilled_frames[i]
        if orig.shape != dist.shape:
            dist = np.array(Image.fromarray(dist).resize(
                (orig.shape[1], orig.shape[0]), Image.BICUBIC))
        combined = np.concatenate([orig, dist], axis=1)
        comparison_frames.append(combined)
    imageio.mimwrite(output_path, comparison_frames, fps=float(fps),
                     quality=8, macro_block_size=1)
    print(f"Saved comparison video: {output_path} ({n} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=MODEL_PATH)
    ap.add_argument("--source_path", default=SOURCE_PATH)
    ap.add_argument("--mask_path", default=MASK_PATH)
    ap.add_argument("--configs", default=CONFIGS)
    ap.add_argument("--iteration", type=int, default=ITERATION)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    ap.add_argument("--num_difix_frames", type=int, default=NUM_DIFIX_FRAMES)
    ap.add_argument("--num_opt_iters", type=int, default=NUM_OPT_ITERS)
    ap.add_argument("--lr_dc", type=float, default=LR_DC)
    ap.add_argument("--lr_rest", type=float, default=LR_REST)
    ap.add_argument("--reg_weight", type=float, default=REG_WEIGHT)
    ap.add_argument("--opt_target", choices=["fg", "all"], default="fg")
    ap.add_argument("--ply_path", default=None,
                    help="Override PLY to load (e.g. harmonized_dc_only.ply)")
    ap.add_argument("--skip_difix", action="store_true",
                    help="Skip DiFix inference, reuse cached targets")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    render_dir = os.path.join(args.output_dir, "rendered_inputs")
    target_dir = os.path.join(args.output_dir, "difix_targets")
    output_ply = os.path.join(args.output_dir, "difix_distilled.ply")
    output_video = os.path.join(args.output_dir, "video_difix_distilled.mp4")
    compare_video = os.path.join(args.output_dir, "video_comparison.mp4")
    original_video_dir = os.path.join(
        args.model_path, f"video/ours_{args.iteration}/renders")

    # --- Stage 1: Load scene ---
    print("=== Stage 1: Loading scene ===")
    gaussians, scene, pipe, background = load_scene_for_blender(
        args.model_path, args.source_path, args.configs, args.iteration)
    cam_type = scene.dataset_type
    if args.ply_path:
        print(f"Overriding PLY with: {args.ply_path}")
        gaussians.load_ply(os.path.abspath(args.ply_path))
    print(f"Loaded {gaussians._xyz.shape[0]} Gaussians, cam_type={cam_type}")

    # Load mask
    from pipeline.data_loading import load_mask_table
    mask_data = load_mask_table(os.path.abspath(args.mask_path))
    if args.opt_target == "fg":
        fg_mask = mask_data["mask_table"].any(dim=0).to("cuda").bool()
    else:
        fg_mask = torch.ones(gaussians._xyz.shape[0], dtype=torch.bool, device="cuda")
    print(f"Mask: {int(fg_mask.sum().item())} / {fg_mask.shape[0]} Gaussians")

    # --- Stage 2: Render + DiFix ---
    print("=== Stage 2: Render training views + run DiFix ===")
    view_indices, rendered_paths = render_training_views(
        scene, gaussians, pipe, background, cam_type,
        args.num_difix_frames, render_dir)

    if not args.skip_difix:
        run_difix(render_dir, target_dir, DIFIX_PYTHON, DIFIX_SCRIPT)
    else:
        print("Skipping DiFix (using cached targets)")

    target_paths = [os.path.join(target_dir, os.path.basename(p)) for p in rendered_paths]
    missing = [p for p in target_paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"{len(missing)} DiFix targets missing, e.g. {missing[0]}")

    # --- Stage 3: Optimise SH ---
    print("=== Stage 3: Optimise SH residuals ===")
    delta_dc, delta_rest, opt_mask, losses = optimise_sh(
        gaussians, scene, pipe, background, view_indices, target_paths,
        fg_mask, cam_type, args.num_opt_iters, args.lr_dc, args.lr_rest,
        args.reg_weight)

    # Bake and save PLY
    with torch.no_grad():
        gaussians._features_dc.data[opt_mask] += delta_dc
        gaussians._features_rest.data[opt_mask] += delta_rest
    gaussians.save_ply(output_ply)
    torch.save({"delta_dc": delta_dc.cpu(), "delta_rest": delta_rest.cpu(),
                "opt_mask": opt_mask.cpu(), "losses": losses},
               os.path.splitext(output_ply)[0] + "_delta.pt")
    print(f"Saved PLY: {output_ply}")

    # --- Stage 4: Render video + comparison ---
    print("=== Stage 4: Render distilled video ===")
    distilled_frames = render_video(
        gaussians, scene, pipe, background, output_video, cam_type)

    if os.path.isdir(original_video_dir):
        print("=== Building side-by-side comparison ===")
        build_comparison_video(original_video_dir, distilled_frames, compare_video)
    else:
        print(f"Original video dir not found at {original_video_dir}, skipping comparison")

    print("Done!")


if __name__ == "__main__":
    main()
