
# run as:  blender joint_cat_breakdancer.blend --background --python render_joint_scenes.py
#
# Prerequisites:
#   1. Open breakdance_snow_skyhdri.blend
#   2. File → Append → cat_forest.blend → Object → Cat (+ its materials)
#   3. Position/scale the cat where you want it
#   4. Save as joint_cat_breakdancer.blend
#   5. Then run this script
import bpy
import json
import os
import math
import numpy as np
from mathutils import Matrix

# ── Parse script args ──────────────────────────────────────
import sys
_argv = sys.argv
_script_args = _argv[_argv.index("--") + 1:] if "--" in _argv else []

def _get_arg(name, default):
    if name in _script_args:
        idx = _script_args.index(name)
        if idx + 1 < len(_script_args):
            return _script_args[idx + 1]
    return default

# ── Config ──────────────────────────────────────────────────
BLEND_DIR = os.path.dirname(bpy.data.filepath)

_scene_dir = _get_arg("--scene_dir", "scene_joint_breakdance_cat")
TRANSFORMS_DIR = os.path.abspath(_scene_dir)
OUTPUT_DIR = os.path.abspath(_scene_dir)

INDOOR_HDRI = os.path.join(BLEND_DIR, "forest.hdr")
OUTDOOR_HDRI = None  # will read from existing world

CAT_OBJECT_NAME = "Actual_Cat"
GROUND_OBJECT_NAME = "Mesh_201"
CAMERA_NAME = "Camera"

RES_X = 800
RES_Y = 800
SAMPLES = 64

# Frame range for converting time -> frame number (for dynamic scenes)
FRAME_START = bpy.context.scene.frame_start
FRAME_END = bpy.context.scene.frame_end


# ── Helper: remap missing image paths ────────────────────────
def remap_missing_images():
    """Remap missing image paths to look next to the blend file."""
    blend_dir = os.path.dirname(bpy.data.filepath)
    for img in bpy.data.images:
        if img.filepath and not os.path.exists(bpy.path.abspath(img.filepath)):
            basename = os.path.basename(bpy.path.abspath(img.filepath))
            local_path = os.path.join(blend_dir, basename)
            if os.path.exists(local_path):
                img.filepath = local_path
                print(f"  Remapped image '{img.name}' -> {local_path}")


# ── Helper: set camera from NeRF transform_matrix ──────────
def set_camera_from_nerf(cam_obj, transform_matrix, camera_angle_x):
    """Set camera pose from a NeRF-style 4x4 transform matrix."""
    mat = Matrix([row for row in transform_matrix])
    cam_obj.matrix_world = mat
    cam_obj.data.angle = camera_angle_x


# ── Helper: create holdout material ─────────────────────────
def get_holdout_material():
    name = "_Holdout"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    holdout = nodes.new('ShaderNodeHoldout')
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(holdout.outputs[0], output.inputs['Surface'])
    return mat


# ── Helper: create/get a World with an HDRI ─────────────────
def get_or_create_hdri_world(name, hdri_path):
    if name in bpy.data.worlds:
        return bpy.data.worlds[name]
    world = bpy.data.worlds.new(name)
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new('ShaderNodeBackground')
    env = nodes.new('ShaderNodeTexEnvironment')
    output = nodes.new('ShaderNodeOutputWorld')

    env.image = bpy.data.images.load(hdri_path)
    links.new(env.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], output.inputs['Surface'])

    bg.location = (0, 0)
    env.location = (-300, 0)
    output.location = (300, 0)
    return world


# ── Helper: hide/show objects by name ─────────────────────────
def set_all_visible():
    """Unhide all objects for rendering."""
    for obj in bpy.data.objects:
        obj.hide_render = False

def hide_all_except(keep_names):
    """Hide all renderable objects except those in keep_names."""
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            continue
        obj.hide_render = (obj.name not in keep_names)


# ── Render all views (single scene, swap world/materials/visibility) ──
def render_all_views():
    remap_missing_images()
    scene = bpy.context.scene
    cat_obj = bpy.data.objects[CAT_OBJECT_NAME]
    cam_obj = bpy.data.objects[CAMERA_NAME]
    holdout_mat = get_holdout_material()

    # Store originals
    outdoor_world = scene.world
    orig_mats = [slot.material for slot in cat_obj.material_slots]
    indoor_world = get_or_create_hdri_world("Indoor", INDOOR_HDRI)

    # Common render settings
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.cycles.samples = SAMPLES
    scene.render.engine = 'CYCLES'

    # Enable GPU rendering
    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        for dev_type in ('OPTIX', 'CUDA', 'METAL'):
            try:
                prefs.preferences.compute_device_type = dev_type
                prefs.preferences.get_devices()
                scene.cycles.device = 'GPU'
                for d in prefs.preferences.devices:
                    d.use = True
                print(f"  GPU rendering enabled ({dev_type})")
                break
            except:
                continue

    for split in ["train", "test"]:
        json_path = os.path.join(TRANSFORMS_DIR, f"transforms_{split}.json")
        with open(json_path) as f:
            tfm_data = json.load(f)

        camera_angle_x = tfm_data["camera_angle_x"]
        frames = tfm_data["frames"]

        for i, frame in enumerate(frames):
            print(f"\n{'='*40} {split} view {i}/{len(frames)} {'='*40}")
            set_camera_from_nerf(cam_obj, frame["transform_matrix"], camera_angle_x)

            # Set animation frame from time field
            time_val = frame.get("time", 0.0)
            anim_frame = int(round(FRAME_START + time_val * (FRAME_END - FRAME_START)))
            scene.frame_set(anim_frame)
            print(f"  time={time_val:.3f} -> frame {anim_frame}")

            # ── Render Scene B (Target): outdoor HDRI, all objects ──
            scene.world = outdoor_world
            scene.render.film_transparent = False
            set_all_visible()
            cat_obj.data.materials.clear()
            for mat in orig_mats:
                cat_obj.data.materials.append(mat)
            bpy.context.view_layer.update()

            out_b = os.path.join(OUTPUT_DIR, "scene_B", split, f"r_{i}.png")
            os.makedirs(os.path.dirname(out_b), exist_ok=True)
            scene.render.filepath = out_b
            bpy.ops.render.render(write_still=True)

            # ── Render Scene A pass 1: BG with cat removed, outdoor HDRI ──
            scene.world = outdoor_world
            scene.render.film_transparent = False
            set_all_visible()
            cat_obj.hide_render = True  # just hide the cat entirely
            bpy.context.view_layer.update()

            out_bg = os.path.join(OUTPUT_DIR, "scene_A_bg", split, f"r_{i}.png")
            os.makedirs(os.path.dirname(out_bg), exist_ok=True)
            scene.render.filepath = out_bg
            bpy.ops.render.render(write_still=True)

            # ── Render Scene A pass 2: Cat only, indoor HDRI ──
            # Keep ground plane visible with holdout material so it occludes
            # the cat's base (matching scene_B where snow covers the base)
            scene.world = indoor_world
            scene.render.film_transparent = True
            hide_all_except({CAT_OBJECT_NAME, GROUND_OBJECT_NAME})
            cat_obj.data.materials.clear()
            for mat in orig_mats:
                cat_obj.data.materials.append(mat)
            ground_obj = bpy.data.objects.get(GROUND_OBJECT_NAME)
            ground_orig_mats = []
            if ground_obj:
                ground_orig_mats = [slot.material for slot in ground_obj.material_slots]
                ground_obj.data.materials.clear()
                ground_obj.data.materials.append(holdout_mat)
            bpy.context.view_layer.update()

            out_cat = os.path.join(OUTPUT_DIR, "scene_A_cat", split, f"r_{i}.png")
            os.makedirs(os.path.dirname(out_cat), exist_ok=True)
            scene.render.filepath = out_cat
            bpy.ops.render.render(write_still=True)

            # Restore ground material
            if ground_obj and ground_orig_mats:
                ground_obj.data.materials.clear()
                for mat in ground_orig_mats:
                    ground_obj.data.materials.append(mat)

            # ── Composite Scene A immediately (numpy, no PIL needed) ──
            out_comp = os.path.join(OUTPUT_DIR, "scene_A", split, f"r_{i}.png")
            os.makedirs(os.path.dirname(out_comp), exist_ok=True)
            bg_data = bpy.data.images.load(out_bg)
            cat_data = bpy.data.images.load(out_cat)
            bg_px = np.array(bg_data.pixels[:]).reshape(bg_data.size[1], bg_data.size[0], 4)
            cat_px = np.array(cat_data.pixels[:]).reshape(cat_data.size[1], cat_data.size[0], 4)
            alpha = cat_px[:, :, 3:4]
            comp = bg_px.copy()
            comp[:, :, :3] = cat_px[:, :, :3] * alpha + bg_px[:, :, :3] * (1 - alpha)
            comp[:, :, 3] = 1.0
            result = bpy.data.images.new("_composite", bg_data.size[0], bg_data.size[1], alpha=True)
            result.pixels[:] = comp.ravel().tolist()
            result.file_format = 'PNG'
            result.save_render(out_comp)
            bpy.data.images.remove(bg_data)
            bpy.data.images.remove(cat_data)
            bpy.data.images.remove(result)
            print(f"  Composited: {out_comp}")

            # Restore for next iteration
            set_all_visible()

        # ── Save the transforms JSON for the joint scene ──
        out_json = os.path.join(OUTPUT_DIR, "scene_B", f"transforms_{split}.json")
        out_data = {"camera_angle_x": camera_angle_x, "frames": []}
        for i, frame in enumerate(frames):
            out_data["frames"].append({
                "file_path": f"./{split}/r_{i}",
                "rotation": frame.get("rotation", 0),
                "time": frame.get("time", 0),
                "transform_matrix": frame["transform_matrix"]
            })
        with open(out_json, 'w') as f:
            json.dump(out_data, f, indent=2)

        out_json_a = os.path.join(OUTPUT_DIR, "scene_A", f"transforms_{split}.json")
        os.makedirs(os.path.dirname(out_json_a), exist_ok=True)
        import shutil
        shutil.copy(out_json, out_json_a)

    # Restore outdoor world so the blend file is saved correctly
    scene.world = outdoor_world
    set_all_visible()
    cat_obj.data.materials.clear()
    for mat in orig_mats:
        cat_obj.data.materials.append(mat)


# ── Step 4: Composite Scene A (offline, after rendering) ─────
def composite_scene_a():
    """Composite BG + Cat passes into final Scene A images."""
    from PIL import Image

    for split in ["train", "test"]:
        bg_dir = os.path.join(OUTPUT_DIR, "scene_A_bg", split)
        cat_dir = os.path.join(OUTPUT_DIR, "scene_A_cat", split)
        out_dir = os.path.join(OUTPUT_DIR, "scene_A", split)
        os.makedirs(out_dir, exist_ok=True)

        for fname in sorted(os.listdir(bg_dir)):
            if not fname.endswith('.png'):
                continue
            bg = Image.open(os.path.join(bg_dir, fname)).convert('RGBA')
            cat = Image.open(os.path.join(cat_dir, fname)).convert('RGBA')

            # Alpha-over: paste cat on top of bg
            bg.paste(cat, (0, 0), cat)  # uses cat's alpha as mask
            bg.save(os.path.join(out_dir, fname))

    print(f"Composited Scene A images saved to {os.path.join(OUTPUT_DIR, 'scene_A')}")


# ── Debug: render one view with outdoor vs indoor HDRI ────────
def debug_hdri():
    """Render two test images to verify HDRI switching works."""
    scene = bpy.context.scene
    cam_obj = bpy.data.objects[CAMERA_NAME]
    cat_obj = bpy.data.objects[CAT_OBJECT_NAME]

    outdoor_world = scene.world
    indoor_world = get_or_create_hdri_world("Indoor", INDOOR_HDRI)

    # Print debug info
    print(f"\n{'='*60}")
    print(f"DEBUG HDRI TEST")
    print(f"  Outdoor world: {outdoor_world.name}")
    print(f"  Indoor world:  {indoor_world.name}")
    print(f"  Indoor HDRI path: {INDOOR_HDRI}")
    print(f"  Indoor world nodes:")
    for node in indoor_world.node_tree.nodes:
        print(f"    {node.type}: {node.name}")
        if node.type == 'TEX_ENVIRONMENT':
            print(f"      image: {node.image.name if node.image else 'NONE'}")
            print(f"      filepath: {node.image.filepath if node.image else 'NONE'}")
    print(f"  Cat object: {cat_obj.name}")
    print(f"  Cat materials: {[s.material.name if s.material else 'None' for s in cat_obj.material_slots]}")
    print(f"{'='*60}\n")

    # Render settings
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.cycles.samples = 64
    scene.render.engine = 'CYCLES'

    debug_dir = os.path.join(OUTPUT_DIR, "debug_hdri")
    os.makedirs(debug_dir, exist_ok=True)

    # Render 1: outdoor world, everything visible
    print("Rendering debug_outdoor.png (outdoor HDRI, all objects)...")
    scene.world = outdoor_world
    scene.render.film_transparent = False
    bpy.context.view_layer.update()
    scene.render.filepath = os.path.join(debug_dir, "debug_outdoor.png")
    bpy.ops.render.render(write_still=True)

    # Render 2: indoor world, everything visible
    print("Rendering debug_indoor.png (indoor/forest HDRI, all objects)...")
    scene.world = indoor_world
    scene.render.film_transparent = False
    bpy.context.view_layer.update()
    scene.render.filepath = os.path.join(debug_dir, "debug_indoor.png")
    bpy.ops.render.render(write_still=True)

    # Restore
    scene.world = outdoor_world

    print(f"\nDebug images saved to {debug_dir}/")
    print("  debug_outdoor.png — scene with outdoor HDRI")
    print("  debug_indoor.png  — scene with indoor/forest HDRI")
    print("Compare these two to verify the HDRI swap is working.")


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    argv = sys.argv
    if "--" in argv:
        script_args = argv[argv.index("--") + 1:]
    else:
        script_args = []

    if "--debug-hdri" in script_args:
        debug_hdri()
    else:
        render_all_views()
        bpy.ops.wm.save_as_mainfile(
            filepath=os.path.join(BLEND_DIR, "joint_cat_breakdancer.blend"))
        print("\nDone! Saved .blend with all three scenes intact.")