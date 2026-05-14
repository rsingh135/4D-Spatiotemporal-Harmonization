"""
Blender D-NeRF export v3 (fixed) — Orbit YOUR camera around a target.

How to use:
    1. Position your camera in Blender so the subject looks good
    2. Press F12 to verify the framing
    3. Save the .blend file
    4. Run this script

    blender scene.blend --background --python blender_export_v3.py -- \
        --output_dir /path/to/output \
        --target_object Cat \
        --num_cameras 100 \
        --num_test 20

Convention: The transform_matrix in the JSON is a 4x4 camera-to-world matrix
in NeRF/OpenGL convention (camera looks along -Z, +Y up), which is the same
as Blender's camera matrix_world. This matches D-NeRF / 4DGaussians expectations.
"""

import bpy
import json
import math
import os
import sys
import random
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="//dnerf_export")
    p.add_argument("--target_object", type=str, default=None,
                   help="Name of object to orbit around (e.g. 'Cat')")
    p.add_argument("--target_objects", type=str, nargs='+', default=None,
                   help="Multiple object names to center between (e.g. 'Actual_Cat' 'Breakdancer')")
    p.add_argument("--target_point", type=float, nargs=3, default=[0, 0, 0],
                   help="Point to orbit around if no target_object(s)")
    p.add_argument("--num_cameras", type=int, default=100)
    p.add_argument("--num_test", type=int, default=20)
    p.add_argument("--resolution", type=int, default=800)
    p.add_argument("--frame_start", type=int, default=-1)
    p.add_argument("--frame_end", type=int, default=-1)
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_rings", type=int, default=3,
                   help="Number of elevation rings")
    p.add_argument("--elevation_spread", type=float, default=15.0,
                   help="Degrees above/below your camera's elevation to sample")
    p.add_argument("--static_frame", type=int, default=-1,
                   help="Lock to a single frame (no animation). -1 = animate as before")
    p.add_argument("--radius", type=float, default=-1,
                   help="Override orbit radius (-1 = auto from scene camera)")
    p.add_argument("--time_scale", type=float, default=1.0,
                   help="Slow down animation: 0.5 = half speed (use first half of frames), "
                        "0.25 = quarter speed, etc. Each camera still gets a unique frame "
                        "but sampled from a narrower time window.")
    return p.parse_args(argv)


def setup_render(resolution, samples):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False

    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        try:
            prefs.preferences.compute_device_type = 'METAL'
            scene.cycles.device = 'GPU'
            for d in prefs.preferences.devices:
                d.use = True
        except:
            try:
                prefs.preferences.compute_device_type = 'CUDA'
                scene.cycles.device = 'GPU'
                for d in prefs.preferences.devices:
                    d.use = True
            except:
                pass


def get_camera_fov_x(cam_obj):
    cam = cam_obj.data
    return 2 * math.atan(cam.sensor_width / (2 * cam.lens))


def matrix_to_list(mat):
    return [[mat[row][col] for col in range(4)] for row in range(4)]


def get_object_center(obj):
    """Get world-space bounding box center of an object.
    For armatures, uses child mesh objects for a more accurate center."""
    # If it's an armature, gather bounding boxes from child meshes
    if obj.type == 'ARMATURE':
        all_corners = []
        for child in obj.children:
            if child.type == 'MESH':
                all_corners.extend(
                    [child.matrix_world @ Vector(c) for c in child.bound_box])
        if all_corners:
            center = sum(all_corners, Vector()) / len(all_corners)
            return center
    # Default: use the object's own bounding box
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox_corners, Vector()) / 8
    return center


def compute_orbit_params(cam_obj, target):
    """
    From the user's camera position and a target point,
    compute the orbit radius, base azimuth, and base elevation.
    """
    cam_pos = cam_obj.matrix_world.translation.copy()
    offset = cam_pos - target

    radius = offset.length
    base_azimuth = math.degrees(math.atan2(offset.y, offset.x))
    horizontal_dist = math.sqrt(offset.x**2 + offset.y**2)
    base_elevation = math.degrees(math.atan2(offset.z, horizontal_dist))

    return radius, base_azimuth, base_elevation


def place_camera_at_orbit(cam_obj, target, radius, azimuth_deg, elevation_deg):
    """
    Place camera at an absolute azimuth and elevation on the orbit sphere,
    then point it at the target using a Track To constraint.

    This recomputes from scratch each time — no accumulation.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(max(-10, min(80, elevation_deg)))  # clamp

    # Position on sphere centered at target
    x = target.x + radius * math.cos(el) * math.cos(az)
    y = target.y + radius * math.cos(el) * math.sin(az)
    z = target.z + radius * math.sin(el)

    cam_obj.location = Vector((x, y, z))

    # Remove old orbit constraint
    for c in list(cam_obj.constraints):
        if c.name == "_orbit_track":
            cam_obj.constraints.remove(c)

    # Create target empty (once)
    empty_name = "_orbit_target"
    if empty_name in bpy.data.objects:
        empty = bpy.data.objects[empty_name]
    else:
        bpy.ops.object.empty_add(location=target)
        empty = bpy.context.object
        empty.name = empty_name

    empty.location = target

    # Add Track To constraint
    constraint = cam_obj.constraints.new('TRACK_TO')
    constraint.name = "_orbit_track"
    constraint.target = empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    # CRITICAL: update so constraint is evaluated before we read matrix_world
    bpy.context.view_layer.update()


def generate_orbit_angles(num_total, base_azimuth, base_elevation, num_rings, elevation_spread):
    """
    Generate (azimuth, elevation) pairs for all cameras.
    Azimuth goes 0-360 evenly. Elevation varies in rings around base_elevation.
    """
    angles = []

    if num_rings <= 1:
        elevations = [base_elevation]
    else:
        elevations = [
            base_elevation - elevation_spread + (2 * elevation_spread * r / (num_rings - 1))
            for r in range(num_rings)
        ]

    # Distribute cameras across rings
    per_ring = num_total // num_rings
    extra = num_total % num_rings

    for ring_idx, elev in enumerate(elevations):
        n = per_ring + (1 if ring_idx < extra else 0)
        for i in range(n):
            # Absolute azimuth (full 360), offset slightly per ring
            az = (360.0 / n) * i + ring_idx * (180.0 / num_rings / n)
            angles.append((az, elev))

    return angles


def export(args):
    rng = random.Random(args.seed)
    scene = bpy.context.scene

    # Output dirs
    output_dir = bpy.path.abspath(args.output_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(bpy.data.filepath), output_dir)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # Camera
    cam_obj = scene.camera
    if not cam_obj:
        print("ERROR: No camera in scene!")
        return

    # Save original camera state to restore later
    orig_location = cam_obj.location.copy()
    orig_rotation = cam_obj.rotation_euler.copy()

    # Set static frame before computing centers (so bounding boxes are correct)
    if args.static_frame >= 0:
        scene.frame_set(args.static_frame)
        bpy.context.view_layer.update()

    # Determine target
    if args.target_objects:
        # Joint mode: center between multiple objects
        centers = []
        for name in args.target_objects:
            obj = bpy.data.objects.get(name)
            if obj is None:
                for o in bpy.data.objects:
                    if o.name.lower() == name.lower():
                        obj = o
                        break
            if obj is None:
                print(f"ERROR: Object '{name}' not found!")
                print(f"Available: {[o.name for o in bpy.data.objects]}")
                return
            c = get_object_center(obj)
            print(f"  '{obj.name}' center: ({c.x:.4f}, {c.y:.4f}, {c.z:.4f})")
            centers.append(c)
        target = sum(centers, Vector()) / len(centers)
        print(f"Joint target (midpoint): ({target.x:.4f}, {target.y:.4f}, {target.z:.4f})")
    elif args.target_object:
        obj = bpy.data.objects.get(args.target_object)
        if obj is None:
            for o in bpy.data.objects:
                if o.name.lower() == args.target_object.lower():
                    obj = o
                    break
        if obj is None:
            print(f"ERROR: Object '{args.target_object}' not found!")
            print(f"Available: {[o.name for o in bpy.data.objects]}")
            return
        target = get_object_center(obj)
        print(f"Target object '{obj.name}' center: ({target.x:.4f}, {target.y:.4f}, {target.z:.4f})")
    else:
        target = Vector(args.target_point)
        print(f"Target point: ({target.x:.4f}, {target.y:.4f}, {target.z:.4f})")

    # Compute orbit parameters from user's camera
    auto_radius, base_azimuth, base_elevation = compute_orbit_params(cam_obj, target)
    radius = args.radius if args.radius > 0 else auto_radius
    print(f"\nOrbit parameters (from your camera):")
    print(f"  Radius:    {radius:.4f} {'(override)' if args.radius > 0 else '(auto)'}")
    print(f"  Azimuth:   {base_azimuth:.1f}°")
    print(f"  Elevation: {base_elevation:.1f}°")
    print(f"  Focal len: {cam_obj.data.lens:.1f}mm")

    # Animation frames
    frame_start = args.frame_start if args.frame_start >= 0 else scene.frame_start
    frame_end = args.frame_end if args.frame_end >= 0 else scene.frame_end
    frames = list(range(frame_start, frame_end + 1, args.frame_step))
    print(f"  Frames:    {frame_start}-{frame_end} ({len(frames)} frames)")
    # min_time / time_range are set after time_scale subsetting (below)

    # Setup render
    setup_render(args.resolution, args.samples)
    camera_angle_x = get_camera_fov_x(cam_obj)
    print(f"  FOV:       {math.degrees(camera_angle_x):.1f}°")

    # Generate all orbit angles
    total = args.num_cameras + args.num_test
    all_angles = generate_orbit_angles(
        total, base_azimuth, base_elevation,
        args.num_rings, args.elevation_spread)

    # Frame assignment per camera
    if args.static_frame >= 0:
        all_frame_indices = [args.static_frame] * total
        print(f"  Static frame mode: locked to frame {args.static_frame}")
    else:
        # Apply time_scale: use only a fraction of the frame range
        if args.time_scale != 1.0:
            n_use = max(1, int(len(frames) * args.time_scale))
            frames_subset = frames[:n_use]
            print(f"  time_scale={args.time_scale}: using frames {frames_subset[0]}-{frames_subset[-1]} "
                  f"({len(frames_subset)}/{len(frames)} frames)")
        else:
            frames_subset = frames
        # Evenly distribute frames across cameras for uniform temporal coverage
        all_frame_indices = [frames_subset[i % len(frames_subset)] for i in range(total)]
        rng.shuffle(all_frame_indices)
        print(f"  Distributed {total} cameras across {len(frames_subset)} frames (~{total // len(frames_subset)} views/frame)")

    # Compute time normalization from the frames actually used
    used_frames = set(all_frame_indices)
    min_time = min(used_frames)
    time_range = max(used_frames) - min_time if len(used_frames) > 1 else 1

    # Render
    def render_split(start_idx, count, split_name):
        frames_json = []
        for i in range(count):
            idx = start_idx + i
            az, elev = all_angles[idx]
            frame_num = all_frame_indices[idx]

            print(f"[{split_name}] {i+1}/{count}: "
                  f"az={az:.1f}° el={elev:.1f}° frame={frame_num}")

            # Place camera at ABSOLUTE position (no accumulation)
            place_camera_at_orbit(cam_obj, target, radius, az, elev)

            # Set animation frame
            scene.frame_set(frame_num)
            bpy.context.view_layer.update()

            # Read the FINAL camera-to-world matrix
            # Blender camera: local -Z = forward, local +Y = up
            # This matches NeRF/OpenGL convention exactly
            c2w = cam_obj.matrix_world.copy()

            # Render
            file_path = f"./{split_name}/r_{i}"
            out_path = os.path.join(output_dir, f"{split_name}/r_{i}.png")
            scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)

            time_val = float(frame_num - min_time) / float(time_range) if time_range > 0 else 0.0
            frames_json.append({
                "file_path": file_path,
                "rotation": 0.0,
                "time": time_val,
                "transform_matrix": matrix_to_list(c2w)
            })
        return frames_json

    print(f"\n{'='*60}")
    print(f"Rendering {args.num_cameras} train + {args.num_test} test")
    print(f"{'='*60}\n")

    train_data = render_split(0, args.num_cameras, "train")
    test_data = render_split(args.num_cameras, args.num_test, "test")

    # Write JSON files
    for name, data in [("train", train_data), ("test", test_data)]:
        path = os.path.join(output_dir, f"transforms_{name}.json")
        with open(path, "w") as f:
            json.dump({"camera_angle_x": camera_angle_x, "frames": data}, f, indent=2)

    # Cleanup: restore camera and remove helper objects
    for c in list(cam_obj.constraints):
        if c.name == "_orbit_track":
            cam_obj.constraints.remove(c)
    cam_obj.location = orig_location
    cam_obj.rotation_euler = orig_rotation

    if "_orbit_target" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["_orbit_target"], do_unlink=True)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Output:    {output_dir}")
    print(f"  Train:     {len(train_data)} views")
    print(f"  Test:      {len(test_data)} views")
    print(f"  Format:    D-NeRF (4DGaussians compatible)")
    print(f"  Convention: NeRF/OpenGL (Blender native)")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    export(args)