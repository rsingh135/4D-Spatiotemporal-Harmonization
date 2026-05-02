"""
Inspect precomputed harmonization targets.

Saves side-by-side images of: composite (before) | harmonizer target | difference | 2D mask
for a sample of views, so you can see what the harmonizer is actually predicting.

Usage:
    cd ~/new_sa4d/sa4d
    python -m pipeline.inspect_targets \
        --targets_dir output/hypernerf/split-cookie/harmonize_cache \
        --output_dir output/hypernerf/split-cookie/harmonize_inspect \
        --num_samples 10
"""

import os
import sys
import argparse
import torch
import numpy as np

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Inspect harmonization targets")
    parser.add_argument('--targets_dir', type=str, required=True,
                        help='Directory containing harmonize_targets.pt')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to save inspection images')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of views to inspect (0 = all)')
    parser.add_argument('--diff_scale', type=float, default=10.0,
                        help='Multiplier on difference image for visibility')
    args = parser.parse_args()

    from pipeline.precompute_targets import load_targets

    print(f"Loading targets from {args.targets_dir}...")
    targets, composites, masks_2d = load_targets(args.targets_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    keys = sorted(targets.keys())
    if args.num_samples > 0:
        step = max(1, len(keys) // args.num_samples)
        keys = keys[::step][:args.num_samples]

    print(f"Saving {len(keys)} inspection images to {args.output_dir}/")

    # Stats across all views
    all_diffs = []
    all_mask_coverage = []

    for i, key in enumerate(keys):
        v_idx, f_idx = key
        comp = composites[key].squeeze(0).cpu().numpy()     # [3, H, W]
        tgt = targets[key].squeeze(0).cpu().numpy()          # [3, H, W]
        mask = masks_2d[key].squeeze(0).cpu().numpy()        # [1, H, W]

        # Transpose to [H, W, 3] / [H, W, 1]
        comp_img = comp.transpose(1, 2, 0)
        tgt_img = tgt.transpose(1, 2, 0)
        mask_img = mask.transpose(1, 2, 0)

        # Difference (amplified for visibility)
        diff = np.abs(tgt_img - comp_img) * args.diff_scale
        diff_clipped = np.clip(diff, 0, 1)

        # Masked difference (only in object region)
        masked_diff = np.abs(tgt_img - comp_img) * (mask_img > 0.5)
        mean_diff = masked_diff.sum() / max(1, (mask_img > 0.5).sum())
        mask_coverage = (mask_img > 0.5).mean()

        all_diffs.append(mean_diff)
        all_mask_coverage.append(mask_coverage)

        # Side by side: composite | target | diff | mask
        h, w = comp_img.shape[:2]
        canvas = np.zeros((h, w * 4 + 6, 3), dtype=np.float32)
        canvas[:, 0:w, :] = comp_img
        canvas[:, w+2:2*w+2, :] = tgt_img
        canvas[:, 2*w+4:3*w+4, :] = diff_clipped
        canvas[:, 3*w+6:4*w+6, :] = np.repeat(mask_img, 3, axis=2)

        out_path = os.path.join(args.output_dir, f'view{v_idx:03d}_frame{f_idx:03d}.png')
        from PIL import Image
        Image.fromarray(to8b(canvas)).save(out_path)

        print(f"  [{i+1}/{len(keys)}] view={v_idx}, frame={f_idx}, "
              f"mask_coverage={mask_coverage:.4f}, mean_masked_diff={mean_diff:.6f}")

    print(f"\n{'='*60}")
    print(f"Summary across {len(keys)} sampled views:")
    print(f"  Mean mask coverage:    {np.mean(all_mask_coverage):.4f}")
    print(f"  Mean masked diff:      {np.mean(all_diffs):.6f}")
    print(f"  Max masked diff:       {np.max(all_diffs):.6f}")
    print(f"  Min masked diff:       {np.min(all_diffs):.6f}")
    print(f"{'='*60}")

    if np.mean(all_diffs) < 0.005:
        print("\n⚠️  The harmonizer targets are nearly identical to the composites.")
        print("   This means the Harmonizer CNN doesn't think the object needs correction.")
        print("   The SH optimization will have almost nothing to optimize toward.")
        print("   Consider: the object's lighting may already look correct to the CNN,")
        print("   or the 2D mask coverage may be too small for it to detect a mismatch.")

    if np.mean(all_mask_coverage) < 0.01:
        print("\n⚠️  Very low mask coverage — the inserted object covers <1% of pixels.")
        print("   The harmonizer may not have enough signal to work with.")

    print(f"\nImages saved to: {args.output_dir}/")
    print("Layout: [composite (before)] | [harmonizer target] | [diff × {:.0f}] | [2D mask]".format(args.diff_scale))


if __name__ == '__main__':
    main()
