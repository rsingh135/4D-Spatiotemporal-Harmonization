"""Find positioning that works ACROSS the BG video camera path (not just train views).
Strategy:
  1. Sample N evenly-spaced BG video cameras.
  2. For each, project canonical BG Gaussians and find which world positions
     project to the central 50% of each frame.
  3. The intersection (Gaussians visible from MANY video cameras) is the
     "always-on-screen" region. Take its median as our anchor.
  4. Place FG1 and FG2 there with substantial horizontal separation in the
     direction that corresponds to "screen-right" averaged over cameras.
  5. Render a sweep across the video path (5 frames spread over the camera path
     × 3 timestamps) to confirm both objects are visible throughout.
"""
from __future__ import annotations
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/home/ubuntu/new_sa4d/sa4d')
os.chdir('/home/ubuntu/new_sa4d/sa4d')
import json
import numpy as np
import torch
import mmcv
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from arguments import ModelParams, ModelHiddenParams
from utils.segment_utils import get_combined_args, to8b
from utils.params_utils import merge_hparams
from utils.transform_utils_torch import init_dynamic_gaussians, render

def load(p):
    parser = ArgumentParser(); mp = ModelParams(parser, sentinel=True); hp = ModelHiddenParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--configs', type=str, default='./arguments/hypernerf/default.py')
    args = get_combined_args(parser, p, 'scene')
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    args.object_masks = False; args.need_gt_masks = False
    return init_dynamic_gaussians(mp.extract(args), hp.extract(args), args.iteration)

g0, sc0, bg = load('./output/hypernerf/oven-mitts_dark')
g1, sc1, _ = load('./output/hypernerf/misc_americano')
g2, sc2, _ = load('./output/hypernerf/split-cookie')
g0.load_mask_table('./output/hypernerf/oven-mitts/segment_results/oven-mitts_delete_mitts_v01.pt')
g1.load_mask_table('./output/hypernerf/misc_americano/segment_results/misc_americano_cup_pour_v01.pt')
g2.load_mask_table('./output/hypernerf/split-cookie/segment_results/split-cookie_cookie_hands_v01.pt')

# 1. Sample video cameras evenly.
video_cams = sc0.getVideoCameras()
N_SAMPLES = 8
sample_idx = np.linspace(0, len(video_cams) - 1, N_SAMPLES, dtype=int)
sample_views = [video_cams[i] for i in sample_idx]
print(f'video path: {len(video_cams)} frames, sampling {N_SAMPLES} at {sample_idx.tolist()}')

# 2. For each sampled camera, project canonical BG and tally which Gaussians are visible-and-central.
xyz_canon = g0._xyz.detach()
N = xyz_canon.shape[0]
visit_count = torch.zeros(N, device='cuda', dtype=torch.int32)

cam_centers = []
cam_rights = []
for v in sample_views:
    proj = v.full_proj_transform.cuda()
    h = torch.cat([xyz_canon, torch.ones((N, 1), device=xyz_canon.device)], dim=1)
    p = h @ proj
    w = p[:, 3:4].clamp(min=1e-6)
    ndc = p[:, :3] / w
    central = (ndc[:, 0].abs() <= 0.4) & (ndc[:, 1].abs() <= 0.4) & (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)
    visit_count += central.int()

    # Camera right axis in world frame.
    W2C = v.world_view_transform.cuda()
    C2W = torch.inverse(W2C)
    cam_centers.append(v.camera_center.cpu().numpy())
    cam_rights.append(C2W[:3, 0].cpu().numpy())  # camera-X = right

cam_centers = np.stack(cam_centers, 0)
cam_rights = np.stack(cam_rights, 0)
print(f'cam center range: x=[{cam_centers[:,0].min():.2f},{cam_centers[:,0].max():.2f}], y=[{cam_centers[:,1].min():.2f},{cam_centers[:,1].max():.2f}], z=[{cam_centers[:,2].min():.2f},{cam_centers[:,2].max():.2f}]')

# 3. Always-visible-and-central BG Gaussians.
always_central = (visit_count >= max(1, N_SAMPLES // 2)).cpu().numpy()
n_always = int(always_central.sum())
print(f'BG Gaussians central in >= {N_SAMPLES // 2} of {N_SAMPLES} sampled cameras: {n_always}')

if n_always < 50:
    # Fallback: relax to "visible in at least 1".
    always_central = (visit_count >= 1).cpu().numpy()
    n_always = int(always_central.sum())
    print(f'  fallback to visit>=1: {n_always}')

centre_world = np.median(g0._xyz.detach().cpu().numpy()[always_central], axis=0)
print(f'stable BG floor anchor (across video path): {centre_world}')

# 4. Average screen-right axis (in world frame) across sampled cameras.
right_world = cam_rights.mean(0)
right_world /= np.linalg.norm(right_world)
print(f'avg world-space "right" axis: {right_world}')

# FG canonical medians (already filtered by mask).
fg1_canon = g1._xyz.detach()[g1._mask_table.any(0)]
fg2_canon = g2._xyz.detach()[g2._mask_table.any(0)]
fg1_med = fg1_canon.median(0).values.cpu().numpy()
fg2_med = fg2_canon.median(0).values.cpu().numpy()

# Use the canonical 70%-extent for scale fit, but cap to keep objects compact.
fg1_extent = (fg1_canon.quantile(0.85, dim=0) - fg1_canon.quantile(0.15, dim=0)).cpu().numpy()
fg2_extent = (fg2_canon.quantile(0.85, dim=0) - fg2_canon.quantile(0.15, dim=0)).cpu().numpy()
TARGET_OBJ_SIZE = 1.6  # smaller objects so they don't overlap
sb1 = float(min(0.4, max(0.05, TARGET_OBJ_SIZE / max(fg1_extent[0], fg1_extent[1], 1e-3))))
sb2 = float(min(0.4, max(0.05, TARGET_OBJ_SIZE / max(fg2_extent[0], fg2_extent[1], 1e-3))))
print(f'scales: sb1={sb1:.3f} sb2={sb2:.3f}')

# 5. Pick separation. Visible-central region is some bbox; find its X extent and use ~half.
visible_xyz = g0._xyz.detach().cpu().numpy()[always_central]
extent_world = visible_xyz.max(0) - visible_xyz.min(0)
print(f'visible-central world bbox extent: {extent_world}')
# Separation along right_world: ~3 units (objects of size ~1.6, gap of ~1.4 between).
SEP = 2.5
target1 = centre_world + (-SEP) * right_world
target2 = centre_world + (+SEP) * right_world
mb1 = target1 - sb1 * fg1_med
mb2 = target2 - sb2 * fg2_med
print(f'target1 (cup):    {target1}')
print(f'target2 (cookie): {target2}')
print(f'mb1: {mb1}')
print(f'mb2: {mb2}')

# 6. Render the video path at multiple frames+timestamps. Mp4 drives BOTH camera and time.
sb0 = 1.0; mb0 = torch.zeros(3); rb0 = torch.zeros(3)
mb1_t = torch.tensor(mb1, device='cuda', dtype=torch.float32); rb1 = torch.zeros(3, device='cuda')
mb2_t = torch.tensor(mb2, device='cuda', dtype=torch.float32); rb2 = torch.zeros(3, device='cuda')

# Sample 8 frames evenly across the video path (each has its own time + view).
preview_idx = np.linspace(0, len(video_cams) - 1, 8, dtype=int)
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for i, vix in enumerate(preview_idx):
    v = video_cams[int(vix)]
    with torch.no_grad():
        r = render(v, float(v.time), [g0, g1, g2], bg,
                   motion_bias=[mb0, mb1_t, mb2_t], rotation_bias=[rb0, rb1, rb2], scales_bias=[sb0, sb1, sb2],
                   static=[False] * 3, seg=[True] * 3, bg=True)
    row, col = divmod(i, 4)
    ax[row, col].imshow(to8b(r['render']).transpose(1, 2, 0))
    ax[row, col].set_title(f'video frame {int(vix)}/{len(video_cams) - 1} t={float(v.time):.2f}')
    ax[row, col].axis('off')
plt.tight_layout()
plt.savefig('./output/hypernerf/oven-mitts/seed_artifacts/composite_preview_v11_video_path.png', dpi=110, bbox_inches='tight')
plt.close()

seed = json.load(open('./output/hypernerf/oven-mitts/seed_artifacts/seed.json'))
seed.update({
    'scales_bias1': sb1, 'motion_bias1': [round(float(x), 3) for x in mb1],
    'scales_bias2': sb2, 'motion_bias2': [round(float(x), 3) for x in mb2],
    '_v11_video_path_anchor': centre_world.tolist(),
    '_v11_right_world': right_world.tolist(),
    '_v11_separation': SEP,
})
json.dump(seed, open('./output/hypernerf/oven-mitts/seed_artifacts/seed.json', 'w'), indent=2)
print('saved v11 grid + seed')
