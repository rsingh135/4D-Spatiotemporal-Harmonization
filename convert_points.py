import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HyperNeRF points.npy to an ASCII PLY.")
    parser.add_argument(
        "--in_npy",
        required=True,
        help="Path to points.npy (e.g. data/hypernerf/misc_americano/points.npy)",
    )
    parser.add_argument(
        "--out_ply",
        required=True,
        help="Output PLY path (e.g. data/hypernerf/misc_americano/points3D_downsample.ply)",
    )
    parser.add_argument(
        "--rgb",
        default="128,128,128",
        help="Constant RGB to write for each point, as 'R,G,B'. Default: 128,128,128",
    )
    args = parser.parse_args()

    in_npy = Path(args.in_npy)
    out_ply = Path(args.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)

    rgb_parts = [int(x) for x in args.rgb.split(",")]
    if len(rgb_parts) != 3 or any((c < 0 or c > 255) for c in rgb_parts):
        raise ValueError("--rgb must be three integers in [0,255], like 128,128,128")
    r, g, b = rgb_parts

    points = np.load(str(in_npy))
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points.npy shape [N,>=3], got {points.shape}")
    num_points = int(points.shape[0])

    with open(out_ply, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for p in points:
            f.write(f"{float(p[0])} {float(p[1])} {float(p[2])} 0 0 0 {r} {g} {b}\n")

    print(f"Converted {num_points} points -> {out_ply}")


if __name__ == "__main__":
    main()
