"""
Single-frame (2D) harmonizer + SH-backprop test harness.

This script is meant for controlled debugging on ONE rendered frame:
  1) Render one composite RGB frame + projected 2D object mask.
  2) Apply a synthetic "bad lighting" corruption to the foreground only (inside mask).
  3) Run a harmonizer backend on the corrupted composite.
  4) Optionally run SH-delta optimization on that single frame toward the chosen target.

Run from repo root (IMPORTANT):
  cd /home/ubuntu/new_sa4d/sa4d
  python -m pipeline.single_frame_harmonize_test --help
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)


def _to8b(x: np.ndarray) -> np.ndarray:
    return (255.0 * np.clip(x, 0.0, 1.0)).astype(np.uint8)


def _chw_to_hwc(x_chw: torch.Tensor) -> np.ndarray:
    x = x_chw.detach().float().cpu().numpy()
    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)
    return np.transpose(x, (1, 2, 0))


def _save_strip(images: Dict[str, np.ndarray], out_path: str, sep_px: int = 2) -> None:
    from PIL import Image, ImageDraw

    keys = list(images.keys())
    if not keys:
        raise ValueError("No images to save.")

    h = images[keys[0]].shape[0]
    widths = []
    for k in keys:
        im = images[k]
        if im.ndim != 3 or im.shape[2] != 3 or im.shape[0] != h:
            raise ValueError(f"All images must be HxWx3 with same H. key={k} shape={im.shape}")
        widths.append(im.shape[1])

    total_w = sum(widths) + sep_px * (len(keys) - 1)
    canvas = np.zeros((h, total_w, 3), dtype=np.float32)
    sep = np.full((h, sep_px, 3), 0.25, dtype=np.float32)

    x0 = 0
    for i, (k, w) in enumerate(zip(keys, widths)):
        canvas[:, x0:x0 + w, :] = images[k]
        x0 += w
        if i != len(keys) - 1:
            canvas[:, x0:x0 + sep_px, :] = sep
            x0 += sep_px

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil = Image.fromarray(_to8b(canvas))
    try:
        draw = ImageDraw.Draw(pil)
        x0 = 0
        for k, w in zip(keys, widths):
            draw.rectangle([x0, 0, x0 + w, 18], fill=(0, 0, 0))
            draw.text((x0 + 3, 2), k, fill=(255, 255, 255))
            x0 += w + sep_px
    except Exception:
        pass

    pil.save(out_path)


def feather_mask(mask_2d: torch.Tensor, sigma_px: float) -> torch.Tensor:
    if sigma_px is None or sigma_px <= 0:
        return mask_2d
    radius = int(max(1, min(25, round(3.0 * float(sigma_px)))))
    k = 2 * radius + 1
    device = mask_2d.device
    dtype = mask_2d.dtype
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(xs * xs) / (2.0 * (sigma_px ** 2)))
    kernel_1d = kernel_1d / kernel_1d.sum()
    w_h = kernel_1d.view(1, 1, 1, k)
    w_v = kernel_1d.view(1, 1, k, 1)
    x = F.pad(mask_2d, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, w_h)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, w_v)
    return x.clamp(0.0, 1.0)


@dataclass
class CorruptionParams:
    brightness: float = 0.5
    gamma: float = 1.8
    contrast: float = 1.0
    cast_r: float = 1.0
    cast_g: float = 1.0
    cast_b: float = 1.0


def apply_foreground_corruption(comp: torch.Tensor, mask_2d: torch.Tensor, p: CorruptionParams) -> torch.Tensor:
    if comp.ndim != 4 or comp.shape[1] != 3:
        raise ValueError(f"Expected comp [1,3,H,W], got {tuple(comp.shape)}")
    if mask_2d.ndim != 4 or mask_2d.shape[1] != 1:
        raise ValueError(f"Expected mask [1,1,H,W], got {tuple(mask_2d.shape)}")

    x = comp
    m = mask_2d
    fg = x

    if abs(float(p.contrast) - 1.0) > 1e-6:
        fg = (fg - 0.5) * float(p.contrast) + 0.5
    if abs(float(p.brightness)) > 1e-6:
        fg = fg * float(max(0.0, 1.0 - float(p.brightness)))
    if abs(float(p.gamma) - 1.0) > 1e-6:
        fg = torch.clamp(fg, 0.0, 1.0) ** float(p.gamma)

    cast = torch.tensor([p.cast_r, p.cast_g, p.cast_b], device=fg.device, dtype=fg.dtype).view(1, 3, 1, 1)
    if not torch.allclose(cast, torch.ones_like(cast)):
        fg = fg * cast

    fg = fg.clamp(0.0, 1.0)
    return (x * (1.0 - m) + fg * m).clamp(0.0, 1.0)


def masked_stats(a: torch.Tensor, b: torch.Tensor, mask_2d: torch.Tensor) -> Dict[str, float]:
    if a.ndim == 3:
        a = a.unsqueeze(0)
    if b.ndim == 3:
        b = b.unsqueeze(0)
    if mask_2d.ndim == 3:
        mask_2d = mask_2d.unsqueeze(0)

    m = (mask_2d > 0.5).float()
    cov = float(m.mean().item())
    diff = (a - b).abs() * m
    denom = float(max(1.0, m.sum().item()))
    mean_l1 = float(diff.sum().item() / denom)
    return {"mask_coverage": cov, "mean_masked_l1": mean_l1}


def run_single_frame_sh_opt(
    *,
    gaussians,
    view,
    pipe,
    background,
    mask_data: Dict,
    target: torch.Tensor,   # [1,3,H,W]
    mask_2d: torch.Tensor,  # [1,1,H,W]
    out_dir: str,
    num_iterations: int,
    lr: float,
    lr_dc: Optional[float],
    lr_rest: Optional[float],
    reg_weight: float,
    use_lpips: bool,
    lpips_weight: float,
    log_interval: int,
) -> None:
    from pipeline.data_loading import get_object_mask
    from pipeline.optimize_sh import create_delta_sh, render_with_delta_sh

    object_mask = get_object_mask(mask_data)
    delta_sh_dc, delta_sh_rest, optimizer, object_mask = create_delta_sh(
        gaussians, object_mask, lr=lr, lr_dc=lr_dc, lr_rest=lr_rest
    )

    lpips_fn = None
    if use_lpips:
        import lpips as _lp
        lpips_fn = _lp.LPIPS(net="vgg").cuda().eval()

    for p in gaussians._deformation.parameters():
        p.requires_grad_(False)

    losses, grad_norms = [], []
    try:
        for it in range(num_iterations):
            optimizer.zero_grad(set_to_none=True)
            rendered = render_with_delta_sh(view, gaussians, pipe, background, delta_sh_dc, delta_sh_rest, object_mask)
            rendered_b = rendered.unsqueeze(0)

            # Weighted L1 inside mask (matches optimize_sh.py behavior, but without core/boundary split here)
            w = mask_2d.clamp(0.0, 1.0)
            diff = (rendered_b - target).abs()
            denom = w.sum().clamp_min(1e-6) * 3.0
            loss_l1 = (diff * w).sum() / denom

            masked_r = rendered_b * mask_2d
            masked_t = target * mask_2d
            total = loss_l1

            if lpips_fn is not None:
                total = total + float(lpips_weight) * lpips_fn(masked_r, masked_t).mean()

            total = total + float(reg_weight) * (delta_sh_dc.pow(2).mean() + delta_sh_rest.pow(2).mean())

            total.backward()
            g1 = 0.0 if delta_sh_dc.grad is None else float(delta_sh_dc.grad.detach().norm().item())
            g2 = 0.0 if delta_sh_rest.grad is None else float(delta_sh_rest.grad.detach().norm().item())
            grad_norms.append((g1, g2))
            optimizer.step()

            losses.append(float(total.detach().item()))
            if (it + 1) % log_interval == 0:
                w = min(log_interval, len(losses))
                avg = sum(losses[-w:]) / w
                avg_g1 = sum(g[0] for g in grad_norms[-w:]) / w
                avg_g2 = sum(g[1] for g in grad_norms[-w:]) / w
                print(f"[single-frame sh-opt] iter {it+1}/{num_iterations} loss={avg:.6f} "
                      f"grad(dc)={avg_g1:.3e} grad(rest)={avg_g2:.3e}")
    finally:
        for p in gaussians._deformation.parameters():
            p.requires_grad_(True)

    with torch.no_grad():
        final = render_with_delta_sh(view, gaussians, pipe, background, delta_sh_dc, delta_sh_rest, object_mask)
        final_img = _chw_to_hwc(final)
        tgt_img = _chw_to_hwc(target.squeeze(0))
        mask_img = _chw_to_hwc(mask_2d.squeeze(0))
        diff = np.clip(np.abs(final_img - tgt_img) * 10.0, 0.0, 1.0)
        _save_strip(
            {"final_render": final_img, "target": tgt_img, "|final-target|x10": diff, "mask": mask_img},
            os.path.join(out_dir, "single_frame_shopt_result.png"),
        )

    torch.save(
        {"losses": losses, "grad_norms": grad_norms, "lr": lr, "reg_weight": reg_weight,
         "use_lpips": use_lpips, "lpips_weight": lpips_weight},
        os.path.join(out_dir, "single_frame_shopt_curves.pt"),
    )


def run(args: argparse.Namespace) -> None:
    from pipeline.data_loading import load_scene, load_mask_table, time_to_frame_idx
    from pipeline.precompute_targets import render_composite_and_mask
    from pipeline.harmonizer_base import create_harmonizer

    gaussians, scene, pipe, background = load_scene(
        args.model_path, args.source_path, iteration=args.iteration, configs=args.configs
    )

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
        raise ValueError(
            f"Gaussian count mismatch: model has {n_model}, mask has {n_mask}. "
            "This usually means you must pass the matching --ply_path for that mask (composite PLY), "
            "or pick a mask generated for the current model."
        )

    views = scene.getTestCameras() if args.use_test_cams else scene.getTrainCameras()
    view = views[args.view_idx]
    view_time = view.time if hasattr(view, "time") else 0.0
    frame_idx = time_to_frame_idx(mask_data, view_time)

    with torch.no_grad():
        comp, mask_2d = render_composite_and_mask(view, gaussians, pipe, background, mask_data, frame_idx)
        mask_2d = feather_mask(mask_2d, args.mask_feather)

    comp_bad = apply_foreground_corruption(
        comp,
        mask_2d,
        CorruptionParams(
            brightness=args.brightness,
            gamma=args.gamma,
            contrast=args.contrast,
            cast_r=args.cast_r,
            cast_g=args.cast_g,
            cast_b=args.cast_b,
        ),
    )

    harmonizer = create_harmonizer(args.backend, weights_path=args.harmonizer_weights)
    with torch.no_grad():
        harm = harmonizer.harmonize(comp_bad, mask_2d).clamp(0.0, 1.0)

    target = harm
    if args.amplify != 1.0:
        target = (comp_bad + float(args.amplify) * (harm - comp_bad)).clamp(0.0, 1.0)

    os.makedirs(args.out_dir, exist_ok=True)
    comp_img = _chw_to_hwc(comp.squeeze(0))
    bad_img = _chw_to_hwc(comp_bad.squeeze(0))
    harm_img = _chw_to_hwc(harm.squeeze(0))
    tgt_img = _chw_to_hwc(target.squeeze(0))
    mask_img = _chw_to_hwc(mask_2d.squeeze(0))
    diff_harm = np.clip(np.abs(harm_img - bad_img) * args.diff_scale, 0.0, 1.0)

    _save_strip(
        {"comp": comp_img, "comp_bad": bad_img, "harm": harm_img, "target": tgt_img,
         f"|harm-bad|x{args.diff_scale:g}": diff_harm, "mask": mask_img},
        os.path.join(args.out_dir, "single_frame_strip.png"),
    )

    s_bad = masked_stats(comp, comp_bad, mask_2d)
    s_h = masked_stats(harm, comp_bad, mask_2d)
    s_t = masked_stats(target, comp_bad, mask_2d)

    print("\nSingle-frame harmonizer test")
    print("=" * 70)
    print(f"view_idx={args.view_idx}  frame_idx={frame_idx}  backend={args.backend}")
    print(f"mask_feather={args.mask_feather} px  amplify={args.amplify}  diff_scale={args.diff_scale}")
    print(f"Mask coverage (thr>0.5): {s_bad['mask_coverage']:.4f}")
    print(f"Corruption strength (mean masked L1 comp vs comp_bad): {s_bad['mean_masked_l1']:.6f}")
    print(f"Harmonizer change (mean masked L1 harm vs comp_bad):   {s_h['mean_masked_l1']:.6f}")
    print(f"Target change (mean masked L1 target vs comp_bad):      {s_t['mean_masked_l1']:.6f}")
    print(f"Saved images: {args.out_dir}/single_frame_strip.png")

    if args.run_sh_opt:
        run_single_frame_sh_opt(
            gaussians=gaussians,
            view=view,
            pipe=pipe,
            background=background,
            mask_data=mask_data,
            target=target,
            mask_2d=mask_2d,
            out_dir=args.out_dir,
            num_iterations=args.num_iterations,
            lr=args.lr,
            lr_dc=args.lr_dc,
            lr_rest=args.lr_rest,
            reg_weight=args.reg_weight,
            use_lpips=args.use_lpips,
            lpips_weight=args.lpips_weight,
            log_interval=args.log_interval,
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-frame harmonizer + SH-backprop test harness")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--source_path", type=str, required=True)
    p.add_argument("--mask_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--configs", type=str, default=None)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--ply_path", type=str, default=None)
    p.add_argument("--use_test_cams", action="store_true")
    p.add_argument("--view_idx", type=int, default=0)

    p.add_argument("--backend", type=str, default="whitebox", choices=["whitebox", "pctnet"])
    p.add_argument("--harmonizer_weights", type=str, default=None)
    p.add_argument("--amplify", type=float, default=1.0)

    p.add_argument("--mask_feather", type=float, default=3.0)
    p.add_argument("--brightness", type=float, default=0.6)
    p.add_argument("--gamma", type=float, default=1.8)
    p.add_argument("--contrast", type=float, default=1.0)
    p.add_argument("--cast_r", type=float, default=1.0)
    p.add_argument("--cast_g", type=float, default=1.0)
    p.add_argument("--cast_b", type=float, default=1.0)
    p.add_argument("--diff_scale", type=float, default=10.0)

    p.add_argument("--run_sh_opt", action="store_true")
    p.add_argument("--num_iterations", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_dc", type=float, default=None,
                   help="Optional LR for delta_sh_dc (coarse brightness). Defaults to --lr.")
    p.add_argument("--lr_rest", type=float, default=None,
                   help="Optional LR for delta_sh_rest (directional detail). Defaults to --lr.")
    p.add_argument("--reg_weight", type=float, default=0.01)
    p.add_argument("--use_lpips", action="store_true")
    p.add_argument("--lpips_weight", type=float, default=0.1)
    p.add_argument("--log_interval", type=int, default=50)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

