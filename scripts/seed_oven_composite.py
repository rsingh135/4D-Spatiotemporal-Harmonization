"""Post-IE helper: pick oven-mitt classifier IDs + seed initial composite biases.

After ``train_ie.py`` finishes for ``oven-mitts``, this script:
  1. Mirrors ``classifier.pt`` and ``mlp.pt`` into ``oven-mitts_dark/``.
  2. Renders the classifier ID map for a sample BG frame, dumps PNGs, and prints
     a per-ID summary (pixel count, mean RGB, vertical footprint) so you can pick
     the IDs corresponding to the mitts. It also auto-suggests a set of IDs whose
     mean colour is mitt-like (warm red/pink/orange) and that sit in the lower
     half of the frame.
  3. Computes per-scene Gaussian centroids for the BG and each FG insert mask,
     and prints initial biases (``motion_bias_*``) that put the FGs roughly on
     the BG counter.

Run from the repo root:

    /home/ubuntu/miniconda3/envs/sa4d/bin/python scripts/seed_oven_composite.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import mmcv
from matplotlib import pyplot as plt

# Make the repo importable when run from anywhere.
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from arguments import ModelParams, PipelineParams, ModelHiddenParams
from scene import Scene, GaussianModel
from gaussian_renderer import render, render_contrastive_feature
from utils.segment_utils import get_combined_args, visualize_obj, to8b
from utils.params_utils import merge_hparams
from utils.transform_utils_torch import init_dynamic_gaussians


def load_feature(model_path: str, cfg: str):
    parser = ArgumentParser()
    model_p = ModelParams(parser, sentinel=True)
    pipeline_p = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--mode", default="feature")
    parser.add_argument("--configs", type=str, default=cfg)
    args = get_combined_args(parser, model_path, "feature")
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    args.object_masks = True
    args.need_gt_masks = True
    g = GaussianModel(args.sh_degree, "feature", hp.extract(args), args.feature_dim)
    sc = Scene(model_p.extract(args), g, load_iteration=args.iteration, mode="feature")
    bg_color = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0],
                             dtype=torch.float32, device="cuda")
    bg_feat = torch.zeros(args.feature_dim, dtype=torch.float32, device="cuda")
    return g, sc, pipeline_p.extract(args), bg_color, bg_feat


def load_scene(model_path: str, cfg: str):
    parser = ArgumentParser()
    model_p = ModelParams(parser, sentinel=True)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str, default=cfg)
    args = get_combined_args(parser, model_path, "scene")
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    args.object_masks = False
    args.need_gt_masks = False
    g, sc, _ = init_dynamic_gaussians(model_p.extract(args), hp.extract(args), args.iteration)
    return g, sc


def mirror_ie_weights(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for name in ("classifier.pt", "mlp.pt"):
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            print(f"  ! missing {src}; skip mirror")
            continue
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            print(f"  · {name} already mirrored")
            continue
        shutil.copy2(src, dst)
        print(f"  ✓ {name} -> {dst}")


def analyse_classifier(model_path: str, cfg: str, out_dir: str, ref_idx: int = 0):
    g, sc, pipe, bg_color, bg_feat = load_feature(model_path, cfg)
    train_cams = list(sc.getTrainCameras())
    if not train_cams:
        raise RuntimeError(f"no train cameras for {model_path}")
    view = train_cams[ref_idx]
    with torch.no_grad():
        rgb = render(view, g, pipe, bg_color)["render"]
        ie = render_contrastive_feature(view, g, pipe, bg_feat)["render"]
        logits = g._classifier(ie)
        pred_obj = torch.argmax(logits, dim=0).cpu().numpy()
    rgb_img = to8b(rgb).transpose(1, 2, 0)
    obj_vis = visualize_obj(pred_obj)
    H, W = pred_obj.shape

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(rgb_img); ax[0].set_title(f"oven-mitts train[{ref_idx}] RGB"); ax[0].axis("off")
    ax[1].imshow(obj_vis); ax[1].set_title("classifier argmax IDs"); ax[1].axis("off")
    plt.tight_layout()
    preview_path = os.path.join(out_dir, "oven_mitts_id_preview.png")
    plt.savefig(preview_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved id preview -> {preview_path}")

    rgb_arr = rgb_img.astype(np.float32) / 255.0
    rows, cols = np.indices(pred_obj.shape)

    summary = []
    unique, counts = np.unique(pred_obj, return_counts=True)
    for uid, n in sorted(zip(unique, counts), key=lambda kv: -kv[1]):
        if n < 200:
            continue
        m = pred_obj == uid
        mean_rgb = rgb_arr[m].mean(axis=0)
        red, green, blue = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
        red_dom = red - max(green, blue)
        warmth = (red + green) / 2 - blue
        v_centre = float(rows[m].mean()) / max(H - 1, 1)
        h_centre = float(cols[m].mean()) / max(W - 1, 1)
        summary.append({
            "id": int(uid),
            "pixels": int(n),
            "rgb": [round(red, 3), round(green, 3), round(blue, 3)],
            "red_dom": round(red_dom, 3),
            "warmth": round(warmth, 3),
            "v_centre": round(v_centre, 2),
            "h_centre": round(h_centre, 2),
        })

    summary.sort(key=lambda r: -r["red_dom"])
    print("  IDs ranked by red dominance (top 12):")
    for r in summary[:12]:
        print(
            f"    id={r['id']:3d} px={r['pixels']:6d} rgb={r['rgb']} "
            f"red_dom={r['red_dom']:+.2f} warmth={r['warmth']:+.2f} "
            f"v={r['v_centre']:.2f} h={r['h_centre']:.2f}"
        )

    suggestion = [
        r["id"] for r in summary
        if r["pixels"] >= 600
        and r["red_dom"] >= 0.05
        and r["warmth"] >= 0.05
        and r["v_centre"] >= 0.35
    ]
    if not suggestion:
        suggestion = [r["id"] for r in summary[:3]]
    print(f"  >>> suggested OVEN_MITTS_REMOVE_IDS = {suggestion}")
    return suggestion, summary


def compute_centroids(bg_model: str, fg1_model: str, fg2_model: str,
                       fg1_mask_path: str, fg2_mask_path: str, cfg: str):
    print("Computing scene/object centroids ...")
    g0, _ = load_scene(bg_model, cfg)
    g1, _ = load_scene(fg1_model, cfg)
    g2, _ = load_scene(fg2_model, cfg)
    bg_xyz = g0._xyz.detach().cpu().numpy()
    bg_med = np.median(bg_xyz, axis=0)
    bg_q1 = np.quantile(bg_xyz, 0.25, axis=0)
    bg_q3 = np.quantile(bg_xyz, 0.75, axis=0)
    print(f"  BG xyz median={bg_med.tolist()}  Q1={bg_q1.tolist()}  Q3={bg_q3.tolist()}")

    def centroid_for_mask(g, path, label):
        d = torch.load(path, map_location="cpu")
        m = d["mask_table"].any(dim=0).bool().numpy()
        sel = g._xyz.detach().cpu().numpy()[m]
        if sel.size == 0:
            raise RuntimeError(f"{label} mask is empty")
        c = sel.mean(axis=0)
        print(f"  {label} centroid={c.tolist()} (count={sel.shape[0]})")
        return c

    c1 = centroid_for_mask(g1, fg1_mask_path, "FG1 (americano cup+pour)")
    c2 = centroid_for_mask(g2, fg2_mask_path, "FG2 (cookie)")

    # Default scales already in the notebook.
    s1, s2 = 0.4, 0.4
    # Place targets just above the BG median, slightly forward, separated horizontally.
    spread = max((bg_q3[0] - bg_q1[0]) * 0.15, 0.5)
    fwd = (bg_q3[2] - bg_q1[2]) * 0.10
    target1 = bg_med + np.array([-spread, 0.0, +fwd])
    target2 = bg_med + np.array([+spread, 0.0, +fwd])
    mb1 = target1 - s1 * c1
    mb2 = target2 - s2 * c2
    seed = {
        "scales_bias1": s1,
        "rotation_bias1": [0.0, 0.0, 0.0],
        "motion_bias1": [round(float(x), 3) for x in mb1],
        "scales_bias2": s2,
        "rotation_bias2": [0.0, 0.0, 0.0],
        "motion_bias2": [round(float(x), 3) for x in mb2],
        "_bg_xyz_median": bg_med.tolist(),
        "_centroid_fg1": c1.tolist(),
        "_centroid_fg2": c2.tolist(),
    }
    print("  >>> seed bias defaults:")
    print(json.dumps(seed, indent=2))
    return seed


def main():
    ap = ArgumentParser()
    ap.add_argument("--bg_ie", default="./output/hypernerf/oven-mitts")
    ap.add_argument("--bg_render", default="./output/hypernerf/oven-mitts_dark")
    ap.add_argument("--fg1", default="./output/hypernerf/misc_americano")
    ap.add_argument("--fg2", default="./output/hypernerf/split-cookie")
    ap.add_argument("--cfg", default="./arguments/hypernerf/default.py")
    ap.add_argument("--fg1_mask", default="./output/hypernerf/misc_americano/segment_results/misc_americano_pseudo_ids16-18_q0.95_trimdef_pp.pt")
    ap.add_argument("--fg2_mask", default="./output/hypernerf/split-cookie/segment_results/split-cookie_only_cookie_v01.pt")
    ap.add_argument("--out_dir", default="./output/hypernerf/oven-mitts/seed_artifacts")
    ap.add_argument("--ref_idx", type=int, default=0)
    args = ap.parse_args()

    print("[1] Mirror IE weights into the dark scene")
    mirror_ie_weights(
        os.path.join(args.bg_ie,     "point_cloud", "iteration_14000"),
        os.path.join(args.bg_render, "point_cloud", "iteration_14000"),
    )

    print("\n[2] Analyse oven-mitts classifier IDs")
    suggested_ids, _summary = analyse_classifier(args.bg_ie, args.cfg, args.out_dir, args.ref_idx)

    print("\n[3] Compute centroid-based seed biases")
    if not os.path.exists(args.fg2_mask):
        print(f"  fg2 mask {args.fg2_mask} missing; building from inverted split-cookie.pt ...")
        src = torch.load(os.path.join(args.fg2, "segment_results/split-cookie.pt"), map_location="cpu")
        os.makedirs(os.path.dirname(args.fg2_mask), exist_ok=True)
        torch.save({
            "mask_table": (~src["mask_table"]).bool().contiguous(),
            "time_map": src.get("time_map", torch.linspace(0, 1, src["mask_table"].shape[0])),
            "note": "auto-built by seed_oven_composite.py (invert of split-cookie.pt)",
        }, args.fg2_mask)
    seed = compute_centroids(args.bg_render, args.fg1, args.fg2, args.fg1_mask, args.fg2_mask, args.cfg)

    out_json = os.path.join(args.out_dir, "seed.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"oven_mitts_remove_ids": suggested_ids, **seed}, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
