#!/usr/bin/env python3
"""
Selective blend of Difix-cleaned frames back into the original frames.

Motivation: harmonization can introduce localized artifacts (halos/shimmer).
Difix helps, but applying it uniformly can also wash out good regions.

This script computes a per-pixel blend weight alpha from the difference between
the original frame and the difix-cleaned frame, then blends:

  out = (1 - alpha) * original + alpha * cleaned

Alpha is derived from a blurred diff magnitude so only regions that changed a
lot get replaced by the cleaned frame.
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import numpy as np
from PIL import Image


def _blur_separable(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur for a 2D float32 array using separable 1D conv."""
    if sigma <= 0:
        return x
    radius = int(max(1, min(25, round(3.0 * float(sigma)))))
    k = 2 * radius + 1
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    w = np.exp(-(xs * xs) / (2.0 * (sigma ** 2))).astype(np.float32)
    w /= np.sum(w)

    # reflect padding
    def conv1d(a: np.ndarray, w1: np.ndarray, axis: int) -> np.ndarray:
        pad = len(w1) // 2
        a_pad = np.pad(a, [(pad, pad) if i == axis else (0, 0) for i in range(a.ndim)], mode="reflect")
        # roll window convolution
        out = np.zeros_like(a, dtype=np.float32)
        for i in range(len(w1)):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + a.shape[axis])
            out += w1[i] * a_pad[tuple(sl)]
        return out

    y = conv1d(x, w, axis=1)  # horizontal (W)
    y = conv1d(y, w, axis=0)  # vertical (H)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True, help="Directory of original PNG frames (frame_*.png)")
    ap.add_argument("--clean_dir", required=True, help="Directory of difix-cleaned PNG frames (frame_*.png)")
    ap.add_argument("--out_dir", required=True, help="Output directory for blended PNG frames")
    ap.add_argument("--sigma", type=float, default=2.0, help="Blur sigma for diff magnitude (pixels)")
    ap.add_argument("--thr_lo", type=float, default=6.0, help="Diff threshold low (0..255)")
    ap.add_argument("--thr_hi", type=float, default=24.0, help="Diff threshold high (0..255)")
    ap.add_argument("--max_alpha", type=float, default=0.85, help="Cap blend alpha")
    ap.add_argument("--min_alpha", type=float, default=0.0, help="Floor blend alpha")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    orig = sorted(glob(os.path.join(args.orig_dir, "frame_*.png")))
    if not orig:
        raise SystemExit(f"No frames found in {args.orig_dir}")

    for op in orig:
        name = os.path.basename(op)
        cp = os.path.join(args.clean_dir, name)
        outp = os.path.join(args.out_dir, name)
        if not os.path.exists(cp):
            raise SystemExit(f"Missing cleaned frame: {cp}")
        if os.path.exists(outp):
            continue

        a = np.asarray(Image.open(op).convert("RGB")).astype(np.float32)
        b = np.asarray(Image.open(cp).convert("RGB")).astype(np.float32)
        if a.shape != b.shape:
            b = np.asarray(Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.BICUBIC)).astype(
                np.float32
            )

        # diff magnitude in RGB
        d = np.sqrt(np.sum((a - b) ** 2, axis=2)).astype(np.float32)  # [H,W]
        d = _blur_separable(d, float(args.sigma))

        lo = float(args.thr_lo)
        hi = float(args.thr_hi)
        if hi <= lo:
            hi = lo + 1e-3
        alpha = (d - lo) / (hi - lo)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha = float(args.min_alpha) + alpha * (float(args.max_alpha) - float(args.min_alpha))
        alpha3 = alpha[..., None]

        out = (1.0 - alpha3) * a + alpha3 * b
        out = np.clip(out, 0, 255).astype(np.uint8)
        Image.fromarray(out).save(outp, optimize=True)


if __name__ == "__main__":
    main()

