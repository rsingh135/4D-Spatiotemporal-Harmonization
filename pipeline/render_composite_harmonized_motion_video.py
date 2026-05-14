#!/usr/bin/env python3
"""
Render a **motion video** from a trained 4DGS checkpoint with:

1. **Harmonized appearance** — optional ``delta_sh.pt`` from
   ``pipeline.run_harmonize`` (same tensors as ``optimize_sh.apply_delta_sh``).
2. **Inserted object keeps temporal motion** — per-Gaussian deformation is
   **enabled on the foreground (union object mask)** and **disabled on the
   background**, so the MLP moves the insert while the rest of the scene stays
   in the canonical pose.

This is the **inverse** of ``render_4dgs.py --composite``, which sets
``_deformation_table = ~fg_mask`` (foreground *static*, background deformed).
Here we set ``_deformation_table = fg_mask`` so the **foreground deforms**.

Run from the ``sa4d`` repo root (same as other pipeline scripts)::

    python -m pipeline.render_composite_harmonized_motion_video \\
      --model_path output/hypernerf/split-cookie \\
      --source_path data/hypernerf/split-cookie \\
      --mask_path output/hypernerf/split-cookie/segment_results/split-cookie.pt \\
      --delta_sh_pt path/to/delta_sh.pt \\
      --output_video path/to/out.mp4 \\
      --iteration 14000 --configs arguments/hypernerf/default.py

Omit ``--delta_sh_pt`` to render composite motion **without** baked harmonize
deltas (checkpoint radiance only).

**Mask quality:** Wrong ``mask_table`` mis-classifies Gaussians; fix segmentation
before interpreting the video.

**Note:** Does not modify ``render_4dgs.py`` or other modules.
"""

from __future__ import annotations

import argparse
import os
import sys

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)

import numpy as np
import torch
from tqdm import tqdm

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def _apply_delta_sh_from_checkpoint(gaussians, delta_sh_pt: str, object_mask: torch.Tensor) -> None:
    pack = torch.load(delta_sh_pt, map_location="cuda")
    d_dc = pack["delta_sh_dc"].cuda()
    d_rest = pack["delta_sh_rest"].cuda()
    om = pack.get("object_mask", None)
    if om is not None:
        om = om.cuda().bool()
        if om.shape != object_mask.shape:
            raise ValueError(
                f"delta_sh.pt object_mask shape {tuple(om.shape)} != runtime object_mask {tuple(object_mask.shape)}"
            )
        if not torch.equal(om, object_mask):
            print(
                "[warn] delta_sh.pt object_mask differs from mask_path union; "
                "using union mask from --mask_path for indexing."
            )
    n = int(object_mask.sum().item())
    if d_dc.shape[0] != n or d_rest.shape[0] != n:
        raise ValueError(
            f"delta_sh tensors N={d_dc.shape[0]} but object_mask has {n} True entries"
        )
    with torch.no_grad():
        gaussians._features_dc.data[object_mask] += d_dc
        gaussians._features_rest.data[object_mask] += d_rest
    print(f"[render_composite_harmonized_motion_video] Applied delta_sh from {delta_sh_pt} ({n} Gaussians)")


def _render_video_frames(views, gaussians, pipeline, background, cam_type):
    """Render all video views to uint8 HxWx3 frames (no PNG dump)."""
    from gaussian_renderer import render

    frames = []
    for view in tqdm(views, desc="Video frames"):
        with torch.no_grad():
            rendering = render(view, gaussians, pipeline, background, cam_type=cam_type)["render"]
        frames.append(to8b(rendering).transpose(1, 2, 0))
    return frames


def main():
    p = argparse.ArgumentParser(
        description="Motion video: harmonized ΔSH + 4D deformation on inserted object only (static background)."
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--source_path", type=str, required=True)
    p.add_argument("--mask_path", type=str, required=True, help="segment_results .pt with mask_table")
    p.add_argument(
        "--ply_path",
        type=str,
        default=None,
        help="Optional composite/harmonized .ply override to load after the checkpoint",
    )
    p.add_argument(
        "--delta_sh_pt",
        type=str,
        default=None,
        help="delta_sh.pt from run_harmonize (optional; skip for non-harmonized composite motion)",
    )
    p.add_argument("--output_video", type=str, required=True, help="Output .mp4 path")
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--configs", type=str, default=None)
    p.add_argument("--fps", type=float, default=30.0, help="Video frame rate")
    p.add_argument(
        "--deformation_mode",
        type=str,
        default="foreground_moves",
        choices=["foreground_moves", "legacy_render_4dgs_composite"],
        help="foreground_moves: object deforms in time, background static (default). "
        "legacy_render_4dgs_composite: match render_4dgs --composite (bg moves, fg static).",
    )
    args = p.parse_args()

    from pipeline.data_loading import load_scene, load_mask_table, get_object_mask

    print("=" * 60)
    print("Load 4DGS scene")
    print("=" * 60)
    gaussians, scene, pipe, background = load_scene(
        os.path.abspath(args.model_path),
        os.path.abspath(args.source_path),
        iteration=args.iteration,
        configs=args.configs,
    )
    if args.ply_path:
        print(f"Overriding PLY with: {args.ply_path}")
        gaussians.load_ply(os.path.abspath(args.ply_path))
        if hasattr(gaussians, "_deformation_table"):
            n_xyz = gaussians._xyz.shape[0]
            if (not torch.is_tensor(gaussians._deformation_table)) or (gaussians._deformation_table.shape[0] != n_xyz):
                gaussians._deformation_table = torch.ones((n_xyz,), device="cuda", dtype=torch.bool)

    print("=" * 60)
    print("Load mask + set composite deformation table")
    print("=" * 60)
    mask_data = load_mask_table(os.path.abspath(args.mask_path))
    fg_mask = get_object_mask(mask_data)
    n_g = gaussians._xyz.shape[0]
    if fg_mask.shape[0] != n_g:
        raise ValueError(f"object mask length {fg_mask.shape[0]} != model Gaussians {n_g}")

    if args.deformation_mode == "foreground_moves":
        gaussians._deformation_table = fg_mask.clone().to(device="cuda", dtype=torch.bool)
        print(
            f"[composite] foreground_moves: {int(fg_mask.sum().item())} Gaussians deform in time, "
            f"{int((~fg_mask).sum().item())} background static"
        )
    else:
        gaussians._deformation_table = (~fg_mask).clone().to(device="cuda", dtype=torch.bool)
        print(
            f"[composite] legacy_render_4dgs_composite: {(~fg_mask).sum().item()} deform, "
            f"{int(fg_mask.sum().item())} foreground static"
        )

    if args.delta_sh_pt:
        _apply_delta_sh_from_checkpoint(gaussians, os.path.abspath(args.delta_sh_pt), fg_mask)

    views = scene.getVideoCameras()
    if not views:
        raise RuntimeError("Scene has no video cameras; check dataset / transforms.")
    cam_type = scene.dataset_type

    print("=" * 60)
    print(f"Rendering {len(views)} video frames → {args.output_video}")
    print("=" * 60)
    frames = _render_video_frames(views, gaussians, pipe, background, cam_type)

    out_abs = os.path.abspath(args.output_video)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        import imageio
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Install imageio (e.g. pip install imageio) to write video.") from e
    imageio.mimwrite(args.output_video, frames, fps=float(args.fps))
    print(f"Saved: {args.output_video}")


if __name__ == "__main__":
    main()
