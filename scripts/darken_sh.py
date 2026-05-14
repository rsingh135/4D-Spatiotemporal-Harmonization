"""Uniformly darken a 4DGS scene PLY by scaling per-Gaussian SH coefficients.

Preserves all relative lighting variation; multiplies the rendered RGB of every
Gaussian by `--scale` (e.g. 0.5 = half brightness). Math:

    Order-0 (DC):   RGB = SH * C0 + 0.5  ->  f_dc_new = s*f_dc + (s-1)*0.5 / C0
    Order >=1:      additive RGB contribution -> f_rest_new = s * f_rest
"""
import argparse
import os
import shutil
import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814  # SH order-0 basis constant


def darken_ply(in_path: str, out_path: str, scale: float, backup_suffix: str = ".orig") -> None:
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    if in_path == out_path:
        backup = in_path + backup_suffix
        if not os.path.exists(backup):
            shutil.copy2(in_path, backup)
            print(f"[backup] {backup}")

    ply = PlyData.read(in_path)
    el = ply.elements[0]
    data = el.data.copy()  # structured array

    s = float(scale)
    dc_offset = (s - 1.0) * 0.5 / C0

    dc_keys = [n for n in data.dtype.names if n.startswith("f_dc_")]
    rest_keys = [n for n in data.dtype.names if n.startswith("f_rest_")]
    print(f"[ply] {in_path}: {len(data)} Gaussians, {len(dc_keys)} DC + {len(rest_keys)} rest SH coeffs")
    print(f"[scale] s={s}  dc_offset={dc_offset:.6f}")

    for k in dc_keys:
        data[k] = (data[k].astype(np.float32) * s + dc_offset).astype(data[k].dtype)
    for k in rest_keys:
        data[k] = (data[k].astype(np.float32) * s).astype(data[k].dtype)

    new_el = PlyElement.describe(data, el.name)
    PlyData([new_el], text=ply.text).write(out_path)
    print(f"[wrote] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_ply", required=True)
    p.add_argument("--out_ply", required=True)
    p.add_argument("--scale", type=float, default=0.5,
                   help="Brightness multiplier (1.0 = unchanged, 0.5 = half-bright, 0.0 = black).")
    a = p.parse_args()
    darken_ply(a.in_ply, a.out_ply, a.scale)


if __name__ == "__main__":
    main()
