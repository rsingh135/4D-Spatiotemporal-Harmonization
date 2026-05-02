"""
Create a test scene where the masked object has artificially wrong lighting.
Modifies SH coefficients at different orders to simulate various lighting problems.

SH coefficient structure (per Gaussian, per color channel):
  Order 0 (DC):  1 coeff  — average color from all directions
  Order 1:       3 coeffs — directional lighting gradient (like a single light)
  Order 2:       5 coeffs — soft shadows, ambient variation
  Order 3:       7 coeffs — sharp specular-like highlights
  Total:        16 coeffs × 3 RGB = stored as _features_dc [N,1,3] + _features_rest [N,15,3]

Usage:
    cd ~/new_sa4d/sa4d

    # Darken the object (DC shift)
    python -m pipeline.create_test_scene \
        --model_path output/hypernerf/torchocolate \
        --source_path data/hypernerf/torchocolate \
        --mask_path output/hypernerf/torchocolate/segment_results/torchocolate.pt \
        --output_ply output/hypernerf/torchocolate/point_cloud/iteration_14000/test_dark.ply \
        --effect dark --strength 1.5

    # Simulate light coming from wrong direction (1st order)
    python -m pipeline.create_test_scene ... --effect wrong_light_dir --strength 2.0

    # Flatten all shading (zero out higher orders, keep DC)
    python -m pipeline.create_test_scene ... --effect flat --strength 1.0

    # Apply to foreground only (default) or background only
    python -m pipeline.create_test_scene ... --effect dark --target foreground
    python -m pipeline.create_test_scene ... --effect dark --target background
"""

import os
import sys
import torch
import argparse

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)


# =============================================================================
# Effects registry
# =============================================================================
# Each effect is a function(gaussians, object_mask, strength) that modifies
# the SH coefficients in-place and returns a description string.

def effect_dark(gaussians, mask, s):
    """Uniformly darken — shifts DC down."""
    shift = torch.tensor([-0.7, -0.7, -0.7], device='cuda') * s
    gaussians._features_dc.data[mask] += shift.view(1, 1, 3)
    return f"DC shift {shift.tolist()}"

def effect_bright(gaussians, mask, s):
    """Uniformly brighten — shifts DC up."""
    shift = torch.tensor([0.7, 0.7, 0.7], device='cuda') * s
    gaussians._features_dc.data[mask] += shift.view(1, 1, 3)
    return f"DC shift {shift.tolist()}"

def effect_blue(gaussians, mask, s):
    """Cool/blue cast — reduce R, boost B."""
    shift = torch.tensor([-0.8, -0.5, 0.8], device='cuda') * s
    gaussians._features_dc.data[mask] += shift.view(1, 1, 3)
    return f"DC shift {shift.tolist()}"

def effect_warm(gaussians, mask, s):
    """Warm/orange cast — boost R, reduce B."""
    shift = torch.tensor([0.8, 0.3, -0.6], device='cuda') * s
    gaussians._features_dc.data[mask] += shift.view(1, 1, 3)
    return f"DC shift {shift.tolist()}"

def effect_green(gaussians, mask, s):
    """Green tint."""
    shift = torch.tensor([-0.5, 0.7, -0.5], device='cuda') * s
    gaussians._features_dc.data[mask] += shift.view(1, 1, 3)
    return f"DC shift {shift.tolist()}"

def effect_wrong_light_dir(gaussians, mask, s):
    """Simulate light from the wrong direction by flipping 1st-order SH.

    1st-order SH (indices 0-2 of _features_rest) encode directional lighting.
    Negating them reverses the light direction — things lit from the left
    become lit from the right, etc.
    """
    # _features_rest shape: [N, 15, 3] — indices 0,1,2 are 1st order
    gaussians._features_rest.data[mask, 0:3, :] *= -s
    return f"Flipped 1st-order SH (×{-s})"

def effect_rotate_light(gaussians, mask, s):
    """Rotate the apparent lighting direction by swapping 1st-order SH axes.

    The 3 first-order SH coefficients correspond to Y, Z, X directions.
    Swapping them rotates where the light appears to come from.
    """
    rest = gaussians._features_rest.data[mask]
    # Rotate: shift (Y,Z,X) -> (Z,X,Y)
    original = rest[:, 0:3, :].clone()
    rest[:, 0, :] = original[:, 1, :]  # Y <- Z
    rest[:, 1, :] = original[:, 2, :]  # Z <- X
    rest[:, 2, :] = original[:, 0, :]  # X <- Y
    # Also scale to make it more dramatic
    gaussians._features_rest.data[mask, 0:3, :] *= s
    gaussians._features_rest.data[mask] = rest
    return f"Rotated 1st-order SH axes (strength {s})"

def effect_flat(gaussians, mask, s):
    """Remove all view-dependent shading — zero out higher-order SH.

    Keeps only DC (average color). The object will look completely flat,
    like a solid color blob with no lighting variation. strength controls
    how much to attenuate (1.0 = fully flat, 0.5 = half the shading removed).
    """
    gaussians._features_rest.data[mask] *= (1.0 - s)
    return f"Attenuated higher-order SH by {s*100:.0f}%"

def effect_harsh_shadow(gaussians, mask, s):
    """Simulate harsh directional shadow — darken DC + add strong directional bias.

    Combines darkening with a strong 1st-order directional component,
    like the object is in a harsh shadow with light only from one side.
    """
    # Darken
    dark = torch.tensor([-0.5, -0.5, -0.5], device='cuda') * s
    gaussians._features_dc.data[mask] += dark.view(1, 1, 3)
    # Add strong directional component (light from above-right)
    gaussians._features_rest.data[mask, 0, :] += 0.4 * s   # Y direction
    gaussians._features_rest.data[mask, 2, :] += 0.3 * s   # X direction
    return f"DC {dark.tolist()} + directional 1st-order bias"

def effect_overexposed(gaussians, mask, s):
    """Simulate overexposure — brighten DC + crush higher-order detail.

    Like the object was photographed with too much light: washed out
    with lost shadow detail.
    """
    bright = torch.tensor([0.8, 0.8, 0.8], device='cuda') * s
    gaussians._features_dc.data[mask] += bright.view(1, 1, 3)
    # Reduce contrast in higher orders
    gaussians._features_rest.data[mask] *= max(0.0, 1.0 - 0.6 * s)
    return f"DC {bright.tolist()} + higher-order reduced by {60*s:.0f}%"

def effect_underexposed(gaussians, mask, s):
    """Simulate underexposure — darken DC + boost higher-order noise.

    Like the object was photographed in too little light: dark with
    exaggerated noisy shading.
    """
    dark = torch.tensor([-0.8, -0.8, -0.8], device='cuda') * s
    gaussians._features_dc.data[mask] += dark.view(1, 1, 3)
    # Amplify higher-order (noisy shading in dark scenes)
    gaussians._features_rest.data[mask] *= (1.0 + 0.5 * s)
    return f"DC {dark.tolist()} + higher-order amplified by {50*s:.0f}%"


def effect_indoor_outdoor(gaussians, mask, s):
    """Simulate indoor object placed in outdoor scene.

    Combines: warm color shift (indoor tungsten) + reduced contrast
    (indoor has less dynamic range than outdoor) + wrong light direction.
    This creates the most harmonizer-visible mismatch because it affects
    color statistics, contrast, AND directionality simultaneously.
    """
    # Warm tungsten tint
    warm = torch.tensor([0.4, 0.15, -0.3], device='cuda') * s
    gaussians._features_dc.data[mask] += warm.view(1, 1, 3)
    # Reduce contrast (indoor = flatter lighting)
    gaussians._features_rest.data[mask] *= max(0.1, 1.0 - 0.5 * s)
    # Flip light direction
    gaussians._features_rest.data[mask, 0:3, :] *= -1.0
    return f"Indoor→outdoor: warm DC {warm.tolist()}, contrast reduced {50*s:.0f}%, light flipped"


def effect_outdoor_indoor(gaussians, mask, s):
    """Simulate outdoor object placed in indoor scene.

    Combines: cool color shift (daylight) + boosted contrast + harsh
    directional component. Opposite of indoor_outdoor.
    """
    cool = torch.tensor([-0.2, -0.05, 0.35], device='cuda') * s
    gaussians._features_dc.data[mask] += cool.view(1, 1, 3)
    # Boost contrast (outdoor has harsher shadows)
    gaussians._features_rest.data[mask] *= (1.0 + 0.4 * s)
    # Add strong directional bias
    gaussians._features_rest.data[mask, 0, :] += 0.5 * s
    gaussians._features_rest.data[mask, 2, :] += 0.3 * s
    return f"Outdoor→indoor: cool DC {cool.tolist()}, contrast boosted {40*s:.0f}%, directional bias added"


def effect_full_mismatch(gaussians, mask, s):
    """Maximum lighting mismatch — hits every SH order aggressively.

    DC: strong color shift. Order 1: flipped + amplified. Order 2-3: randomized.
    Designed to produce the most visually obvious before/after for the harmonizer.
    """
    # Strong DC shift
    shift = torch.tensor([0.6, -0.3, -0.4], device='cuda') * s
    gaussians._features_dc.data[mask] += shift.view(1, 1, 3)
    # Flip and amplify 1st order (light direction)
    gaussians._features_rest.data[mask, 0:3, :] *= -1.5 * s
    # Scramble 2nd order (soft shadows)
    gaussians._features_rest.data[mask, 3:8, :] *= -0.8 * s
    # Dampen 3rd order (specular)
    gaussians._features_rest.data[mask, 8:15, :] *= 0.2
    return f"Full mismatch: DC {shift.tolist()}, O1 flipped×{1.5*s:.1f}, O2 scrambled, O3 dampened"


EFFECTS = {
    # DC-only (color/brightness shifts)
    'dark':             effect_dark,
    'bright':           effect_bright,
    'blue':             effect_blue,
    'warm':             effect_warm,
    'green':            effect_green,
    # Directional lighting
    'wrong_light_dir':  effect_wrong_light_dir,
    'rotate_light':     effect_rotate_light,
    # Shading structure
    'flat':             effect_flat,
    # Combined (most realistic)
    'harsh_shadow':     effect_harsh_shadow,
    'overexposed':      effect_overexposed,
    'underexposed':     effect_underexposed,
    # Scene transfer (most harmonizer-visible)
    'indoor_outdoor':   effect_indoor_outdoor,
    'outdoor_indoor':   effect_outdoor_indoor,
    'full_mismatch':    effect_full_mismatch,
}


def main():
    parser = argparse.ArgumentParser(
        description="Create test scene with artificial lighting mismatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Effects and what SH orders they modify:
  DC only (order 0 — overall color):
    dark, bright, blue, warm, green

  Directional (order 1 — light direction):
    wrong_light_dir    flip where light comes from
    rotate_light       rotate the lighting direction

  Shading structure (orders 1-3):
    flat               zero out all view-dependent effects

  Combined (most realistic for testing):
    harsh_shadow       dark + strong directional bias
    overexposed        bright + washed out shading
    underexposed       dark + noisy exaggerated shading
""")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--mask_path', type=str, required=True)
    parser.add_argument('--output_ply', type=str, required=True)
    parser.add_argument('--effect', type=str, default='dark', choices=list(EFFECTS.keys()),
                        help='Type of lighting distortion to apply')
    parser.add_argument('--strength', type=float, default=1.0,
                        help='Multiplier on effect strength (default 1.0)')
    parser.add_argument('--target', type=str, default='foreground',
                        choices=['foreground', 'background'],
                        help='Apply effect to the masked object (foreground) or everything else (background)')
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--configs', type=str, default=None)
    parser.add_argument('--ply_path', type=str, default=None,
                        help='Override which .ply file to load (e.g. a composite PLY)')
    args = parser.parse_args()

    from pipeline.data_loading import load_scene, load_mask_table, get_object_mask

    # Load
    gaussians, scene, pipe, bg = load_scene(
        args.model_path, args.source_path,
        iteration=args.iteration, configs=args.configs)
    if args.ply_path is not None:
        print(f"Overriding PLY with: {args.ply_path}")
        gaussians.load_ply(args.ply_path)
    mask_data = load_mask_table(args.mask_path)
    object_mask = get_object_mask(mask_data)

    # Handle Gaussian count mismatch (e.g. after retraining)
    n_gaussians = gaussians._xyz.shape[0]
    n_mask = object_mask.shape[0]
    if n_gaussians != n_mask:
        print(f"\nWARNING: Gaussian count mismatch — model has {n_gaussians}, "
              f"mask has {n_mask}.")
        print("Mask was generated from an older model. Falling back to "
              "applying effect to ALL Gaussians.")
        object_mask = torch.ones(n_gaussians, dtype=torch.bool, device='cuda')

    # Choose which Gaussians to modify
    if args.target == 'background':
        apply_mask = ~object_mask
        target_label = "background"
    else:
        apply_mask = object_mask
        target_label = "foreground (masked object)"

    n_affected = apply_mask.sum().item()
    n_total = apply_mask.shape[0]

    print(f"\n{'='*60}")
    print(f"Effect:    {args.effect}")
    print(f"Strength:  {args.strength}")
    print(f"Target:    {target_label}")
    print(f"Gaussians: {n_affected} / {n_total} ({n_affected/n_total*100:.1f}%)")
    print(f"{'='*60}")

    # Show before stats
    dc_before = gaussians._features_dc.data[apply_mask].mean(dim=0).squeeze()
    rest_before = gaussians._features_rest.data[apply_mask].abs().mean()
    print(f"\nBefore — DC mean: [{dc_before[0]:.3f}, {dc_before[1]:.3f}, {dc_before[2]:.3f}]"
          f"  higher-order mean_abs: {rest_before:.4f}")

    # Apply effect
    with torch.no_grad():
        description = EFFECTS[args.effect](gaussians, apply_mask, args.strength)

    # Show after stats
    dc_after = gaussians._features_dc.data[apply_mask].mean(dim=0).squeeze()
    rest_after = gaussians._features_rest.data[apply_mask].abs().mean()
    print(f"After  — DC mean: [{dc_after[0]:.3f}, {dc_after[1]:.3f}, {dc_after[2]:.3f}]"
          f"  higher-order mean_abs: {rest_after:.4f}")
    print(f"Applied: {description}")

    # Save
    os.makedirs(os.path.dirname(args.output_ply), exist_ok=True)
    gaussians.save_ply(args.output_ply)
    print(f"\nSaved to {args.output_ply}")


if __name__ == '__main__':
    main()
