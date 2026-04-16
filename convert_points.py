import numpy as np

# Load the pre-generated points
points = np.load('data/hypernerf/split-cookie/points.npy')
num_points = len(points)

# Define the output path
out_path = 'data/hypernerf/split-cookie/points3D_downsample.ply'

with open(out_path, 'w') as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {num_points}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property float nx\nproperty float ny\nproperty float nz\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")
    
    for p in points:
        # Write: X Y Z | NX NY NZ (0 0 0) | R G B (128 128 128)
        f.write(f"{p[0]} {p[1]} {p[2]} 0 0 0 128 128 128\n")

print(f"✅ Successfully converted {num_points} points with Normals and Colors to {out_path}!")
