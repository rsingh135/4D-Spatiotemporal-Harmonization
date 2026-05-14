"""Re-render the composite cookie+chocolate scene with a heuristic drop-shadow.

Reproduces the transforms from `composite_cookie_choc.ipynb` (BG = split-cookie,
FG = torchocolate, scales_bias1=0.7, rotation_bias1=[0,0,-0.45], camera-relative
offset against view 0), then adds a third "static" Gaussian set: a flat oval of dark
semi-opaque Gaussians sitting on the support plane below the chocolate object.

Outputs:
    output/composite_torchocolateBigger/composite_cookie_chocBigger_shadow.mp4
    output/composite_torchocolateBigger/preview_with_shadow.png
    output/composite_torchocolateBigger/preview_no_shadow.png

Run:
    cd /home/ubuntu/new_sa4d/sa4d
    /home/ubuntu/miniconda3/envs/sa4d/bin/python scripts/composite_cookie_choc_with_shadow.py \
        [--down-axis +y|-y|+x|-x|+z|-z] [--no-video] [--n-points 2500] \
        [--half-extent 0.6] [--opacity 0.55] [--drop 0.05]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/ubuntu/new_sa4d/sa4d")
os.chdir("/home/ubuntu/new_sa4d/sa4d")

from argparse import ArgumentParser

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelHiddenParams, ModelParams
from utils.params_utils import merge_hparams
from utils.segment_utils import get_combined_args, to8b
from utils.shadow_utils import build_shadow_under_object
from utils.transform_utils_torch import (
    get_state_at_time,
    init_dynamic_gaussians,
    render,
    transform,
)


# ----------------------------------------------------------------------------
# Paths and transforms (must match composite_cookie_choc.ipynb)
# ----------------------------------------------------------------------------
MODEL0 = "./output/hypernerf/split-cookie"     # background scene
MODEL1 = "./output/hypernerf/torchocolate"     # foreground object scene
CFG_PATH = "./arguments/hypernerf/default.py"

MASK0 = "./output/hypernerf/split-cookie/segment_results/split-cookie.pt"
MASK1 = "./output/hypernerf/torchocolate/segment_results/torchocolateBigger.pt"

USE_COOKIE_MASK = True

OUT_DIR = "./output/composite_torchocolateBigger"
OUT_MP4 = os.path.join(OUT_DIR, "composite_cookie_chocBigger_shadow.mp4")
OUT_PREVIEW_SHADOW = os.path.join(OUT_DIR, "preview_with_shadow.png")
OUT_PREVIEW_NOSHADOW = os.path.join(OUT_DIR, "preview_no_shadow.png")
OUT_PREVIEW_GRID = os.path.join(OUT_DIR, "preview_shadow_compare.png")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def load_scene(model_path: str):
    parser = ArgumentParser()
    mp = ModelParams(parser, sentinel=True)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str, default=CFG_PATH)
    args = get_combined_args(parser, model_path, "scene")
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    args.object_masks = False
    args.need_gt_masks = False
    return init_dynamic_gaussians(mp.extract(args), hp.extract(args), args.iteration)


def camera_axes_world(cam):
    W2C = cam.world_view_transform.cuda()
    C2W = torch.inverse(W2C)
    right = C2W[:3, 0]
    up = C2W[:3, 1]
    forward = -C2W[:3, 2]
    return F.normalize(right, dim=0), F.normalize(up, dim=0), F.normalize(forward, dim=0)


def estimate_world_down(train_cams) -> torch.Tensor:
    """Average camera-+Y world direction. In COLMAP/HyperNeRF convention camera +Y points
    DOWN in image space, so its world-frame value is approximately the gravity vector."""
    accum = torch.zeros(3, device="cuda")
    for cam in train_cams:
        W2C = cam.world_view_transform.cuda()
        C2W = torch.inverse(W2C)
        accum += C2W[:3, 1]
    accum = accum / max(len(train_cams), 1)
    return F.normalize(accum, dim=0)


def estimate_support_plane(
    bg_xyz: torch.Tensor,
    fg_centroid: torch.Tensor,
    cam_position: torch.Tensor,
    *,
    radius: float = 2.0,
    max_points: int = 20000,
):
    """Fit a plane to BG Gaussians within `radius` of `fg_centroid` and return both
    the plane normal (pointing toward the camera, "up out of the surface") and a
    point on the plane (the centroid of the fitted neighborhood).

    Returns:
        (up_normal, plane_point) — both as torch.Tensor of shape (3,) on the same
        device as `bg_xyz`.
    """
    diffs = bg_xyz - fg_centroid.unsqueeze(0)
    sq = (diffs * diffs).sum(dim=1)
    mask = sq < (radius * radius)
    pts = bg_xyz[mask]
    if pts.shape[0] < 50:
        order = torch.argsort(sq)
        pts = bg_xyz[order[: max(50, int(0.02 * bg_xyz.shape[0]))]]
    if pts.shape[0] > max_points:
        idx = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
        pts = pts[idx]

    c = pts.mean(0)
    X = (pts - c).double()
    cov = X.T @ X / max(1, X.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    n = eigvecs[:, 0].float()
    n = F.normalize(n, dim=0)
    to_cam = F.normalize(cam_position - c.float(), dim=0)
    if (n * to_cam).sum() < 0:
        n = -n
    return n, c.float()


def parse_axis(spec: str, train_cams, bg_xyz=None, fg_centroid=None, cam_position=None,
               support_radius: float = 2.0):
    """Returns (down_axis, plane_point_or_None)."""
    spec = spec.strip().lower()
    if spec == "auto":
        return estimate_world_down(train_cams), None
    if spec == "support":
        assert bg_xyz is not None and fg_centroid is not None and cam_position is not None, \
            "support axis requires bg_xyz, fg_centroid and cam_position"
        up, plane_point = estimate_support_plane(
            bg_xyz, fg_centroid, cam_position, radius=support_radius)
        return -up, plane_point  # "down" = into the surface; plane_point used for projection
    sign = 1.0
    if spec.startswith("+"):
        spec = spec[1:]
    elif spec.startswith("-"):
        sign = -1.0
        spec = spec[1:]
    base = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[spec]
    return torch.tensor([sign * b for b in base], device="cuda", dtype=torch.float32), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--down-axis", type=str, default="support",
                    help="World 'down' direction. 'support' (default) PCA-fits BG Gaussians near "
                         "the FG centroid; 'auto' averages camera +Y; or pass +y|-y|+x|-x|+z|-z.")
    ap.add_argument("--support-radius", type=float, default=2.0,
                    help="Radius (world units) of BG Gaussian neighborhood used by 'support' axis.")
    ap.add_argument("--n-points", type=int, default=2500)
    ap.add_argument("--half-extent", type=float, default=0.45,
                    help="Disk radius in world units (ignored when --auto-size is set).")
    ap.add_argument("--auto-size", type=float, default=1.15,
                    help="If >0, auto-size disk radius to (FG footprint 90th-pct) * this factor. "
                         "Set to 0 to use --half-extent.")
    ap.add_argument("--opacity", type=float, default=0.40,
                    help="Peak per-Gaussian opacity at disk center.")
    ap.add_argument("--drop", type=float, default=0.01,
                    help="World units to push the shadow into the support plane (anti-z-fight). "
                         "In plane-projection mode this is a small positive offset; in legacy "
                         "mode it's added to the FG-bottom drop.")
    ap.add_argument("--falloff", type=float, default=2.5,
                    help="Radial alpha falloff exponent (higher = softer edge, more pronounced "
                         "centre).")
    ap.add_argument("--bottom-percentile", type=float, default=0.85,
                    help="(Legacy mode only.) Robust 'bottom of FG' along down axis.")
    ap.add_argument("--gaussian-size", type=float, default=4.0,
                    help="Per-Gaussian size multiplier (larger = softer, more solid shadow).")
    ap.add_argument("--thickness", type=float, default=0.015,
                    help="Out-of-plane thickness of the shadow plate in world units.")
    ap.add_argument("--rgb", type=str, default="0.05,0.04,0.03",
                    help="Comma-separated RGB color in [0,1].")
    ap.add_argument("--no-plane-project", action="store_true",
                    help="Disable orthogonal projection of FG centroid onto the support plane "
                         "(falls back to legacy 'push along down_axis' placement).")
    ap.add_argument("--no-video", action="store_true",
                    help="Only render preview frames, skip the full mp4.")
    ap.add_argument("--video-source", type=str, default="cookie",
                    choices=["cookie", "torchocolate"],
                    help="Which scene's video camera path to use.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mp4-out", type=str, default=OUT_MP4)
    ap.add_argument("--track-fg", action="store_true",
                    help="Recompute the shadow position each frame from the FG's deformed centroid.")
    args = ap.parse_args()

    rgb = tuple(float(x) for x in args.rgb.split(","))
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load both scenes
    # ------------------------------------------------------------------
    print("Loading split-cookie BG ...")
    g0, scene0, background = load_scene(MODEL0)
    print("Loading torchocolate FG ...")
    g1, scene1, _ = load_scene(MODEL1)

    g0.load_mask_table(MASK0)
    g1.load_mask_table(MASK1)

    train_cams0 = scene0.getTrainCameras()

    # ------------------------------------------------------------------
    # Reproduce the FG transform from cell 14 of composite_cookie_choc.ipynb
    # ------------------------------------------------------------------
    scales_bias1 = 0.7
    rotation_bias1 = torch.tensor([0.0, 0.0, -0.45], device="cuda")
    base_offset1 = torch.tensor([2.7, 0.7, 0.2], device="cuda")
    d_right, d_up, d_forward = 0.6, 0.0, 0.9

    test_view = train_cams0[0]
    right, up, forward = camera_axes_world(test_view)
    motion_bias1 = base_offset1 + d_right * right + d_up * up + d_forward * forward

    print(f"FG transform: scale={scales_bias1}, rot={rotation_bias1.cpu().tolist()}, "
          f"motion={motion_bias1.cpu().tolist()}")

    scales_bias0 = 1.0
    rotation_bias0 = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    motion_bias0 = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    # ------------------------------------------------------------------
    # Compute FG world XYZ at t=0 to seed the shadow placement
    # ------------------------------------------------------------------
    with torch.no_grad():
        fg_keep = g1._mask_table.any(dim=0).bool()
        xyz_fg_canon = g1._xyz[fg_keep].detach().clone()
        scaling_fg = g1.scaling_activation(g1._scaling[fg_keep].detach().clone())
        rotation_fg = g1.rotation_activation(g1._rotation[fg_keep].detach().clone())
        opacity_fg = g1.opacity_activation(g1._opacity[fg_keep].detach().clone()).reshape(-1)
        xyz_fg_world, _, _ = transform(
            xyz_fg_canon, rotation_fg, scaling_fg,
            scales_bias1, motion_bias1, rotation_bias1,
        )

    print(f"FG world bbox: min={xyz_fg_world.min(0).values.cpu().tolist()}, "
          f"max={xyz_fg_world.max(0).values.cpu().tolist()}, "
          f"centroid={xyz_fg_world.mean(0).cpu().tolist()}")

    # ------------------------------------------------------------------
    # Pick the down axis & build the shadow
    # ------------------------------------------------------------------
    fg_centroid = xyz_fg_world.mean(0).detach()
    bg_xyz = g0._xyz.detach()
    down_axis, plane_point = parse_axis(
        args.down_axis, train_cams0,
        bg_xyz=bg_xyz, fg_centroid=fg_centroid,
        cam_position=test_view.camera_center.cuda(),
        support_radius=args.support_radius,
    )
    print(f"Using world down axis: {down_axis.cpu().tolist()}")
    if plane_point is not None:
        print(f"Support plane point (BG cluster centroid): {plane_point.cpu().tolist()}")

    use_plane_proj = (plane_point is not None) and (not args.no_plane_project)
    plane_point_arg = plane_point if use_plane_proj else None
    auto_size_arg = float(args.auto_size) if args.auto_size and args.auto_size > 0 else None

    shadow = build_shadow_under_object(
        fg_world_xyz=xyz_fg_world,
        down_axis=down_axis,
        drop_distance=args.drop,
        n_points=args.n_points,
        half_extent=(args.half_extent, args.half_extent),
        thickness=args.thickness,
        rgb_color=rgb,
        opacity=args.opacity,
        falloff=args.falloff,
        fg_weights=opacity_fg,
        bottom_percentile=args.bottom_percentile,
        gaussian_size_mult=args.gaussian_size,
        plane_point=plane_point_arg,
        auto_size_factor=auto_size_arg,
    )
    placement_mode = "plane-projection" if use_plane_proj else "drop-along-axis (legacy)"
    print(f"Shadow built: {shadow._xyz.shape[0]} Gaussians at "
          f"{shadow._xyz.mean(0).cpu().tolist()} (mode={placement_mode})")
    cam_right, cam_up, cam_forward = camera_axes_world(test_view)
    cos_normal_view = float(F.cosine_similarity(down_axis.unsqueeze(0), cam_forward.unsqueeze(0)).item())
    print(f"View-0 forward (world): {cam_forward.cpu().tolist()}")
    print(f"  cos(disk normal, cam fwd) = {cos_normal_view:.3f}  "
          f"(±1 = face-on, 0 = edge-on)")

    # ------------------------------------------------------------------
    # Single-frame previews (no shadow vs with shadow vs side-by-side)
    # ------------------------------------------------------------------
    common_kwargs_no_shadow = dict(
        gaussians=[g0, g1],
        bg_color=background,
        motion_bias=[motion_bias0, motion_bias1],
        rotation_bias=[rotation_bias0, rotation_bias1],
        scales_bias=[scales_bias0, scales_bias1],
        static=[False, False],
        seg=[USE_COOKIE_MASK, True],
        bg=True,
    )
    common_kwargs_shadow = dict(
        gaussians=[g0, g1, shadow],
        bg_color=background,
        motion_bias=[motion_bias0, motion_bias1, torch.zeros(3, device="cuda")],
        rotation_bias=[rotation_bias0, rotation_bias1, torch.zeros(3, device="cuda")],
        scales_bias=[scales_bias0, scales_bias1, 1.0],
        static=[False, False, True],
        seg=[USE_COOKIE_MASK, True, False],
        bg=True,
    )

    with torch.no_grad():
        r_no = render(test_view, test_view.time, **common_kwargs_no_shadow)
        r_yes = render(test_view, test_view.time, **common_kwargs_shadow)

    img_no = to8b(r_no["render"]).transpose(1, 2, 0)
    img_yes = to8b(r_yes["render"]).transpose(1, 2, 0)
    imageio.imwrite(OUT_PREVIEW_NOSHADOW, img_no)
    imageio.imwrite(OUT_PREVIEW_SHADOW, img_yes)
    print(f"Wrote {OUT_PREVIEW_NOSHADOW}\nWrote {OUT_PREVIEW_SHADOW}")

    actual_radius = float(((shadow._xyz - shadow._xyz.mean(0)).norm(dim=1)).max().item())
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(img_no);  axs[0].set_title("Composite — no shadow"); axs[0].axis("off")
    axs[1].imshow(img_yes); axs[1].set_title(
        f"Composite + shadow ({shadow._xyz.shape[0]} gaussians, opa={args.opacity}, "
        f"r≈{actual_radius:.2f}, falloff={args.falloff}, mode={placement_mode})"); axs[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_PREVIEW_GRID, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PREVIEW_GRID}")

    if args.no_video:
        return

    # ------------------------------------------------------------------
    # Full mp4 render
    # ------------------------------------------------------------------
    if args.video_source == "cookie":
        video_cams = scene0.getVideoCameras()
    else:
        video_cams = scene1.getVideoCameras()

    out_path = args.mp4_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=args.fps, quality=8, macro_block_size=1)

    print(f"Rendering {len(video_cams)} frames → {out_path}")
    torch.cuda.empty_cache()
    gc.collect()

    # Anchor offset: relative position of the shadow centre wrt the FG world centroid at t=0.
    # When tracking, we apply only the IN-PLANE component of the FG centroid delta — height
    # changes shouldn't move the shadow laterally on the support plane.
    initial_fg_centre = xyz_fg_world.mean(0).detach().clone()
    initial_shadow_xyz = shadow._xyz.clone()
    if use_plane_proj:
        plane_normal = -down_axis  # points away from surface (toward camera)
    else:
        plane_normal = -down_axis  # same direction is fine: we only need to remove the component along it

    with torch.no_grad():
        for idx, viewpoint in enumerate(tqdm(video_cams)):
            if args.track_fg:
                # Recompute deformed FG centroid for this timestamp, in world space (post transform).
                m_fg, s_fg, r_fg, _, _ = get_state_at_time(
                    g1, timestamp=viewpoint.time, seg=True, static=False,
                )
                if m_fg.shape[0] > 0:
                    m_w, _, _ = transform(m_fg, r_fg, s_fg, scales_bias1, motion_bias1, rotation_bias1)
                    new_centre = m_w.mean(0)
                    delta = new_centre - initial_fg_centre
                    # Project delta onto the support plane (drop the normal component).
                    delta_in_plane = delta - (delta * plane_normal).sum() * plane_normal
                    shadow._xyz = (initial_shadow_xyz + delta_in_plane).contiguous()

            r = render(viewpoint, viewpoint.time, **common_kwargs_shadow)
            frame = to8b(r["render"]).transpose(1, 2, 0)
            writer.append_data(frame)
            del r, frame
            if idx % 50 == 0:
                torch.cuda.empty_cache()

    writer.close()
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
