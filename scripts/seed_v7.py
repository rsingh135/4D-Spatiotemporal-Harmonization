"""Robust seed: anchor FG positions to BG world coordinates that actually project
to the centre of camera 0's view. No reliance on deformation-net outputs (which can
produce wild outliers for some Gaussians)."""
from __future__ import annotations
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/home/ubuntu/new_sa4d/sa4d')
import json
from argparse import ArgumentParser

import numpy as np
import torch
import mmcv
from matplotlib import pyplot as plt

from arguments import ModelParams, PipelineParams, ModelHiddenParams
from utils.segment_utils import get_combined_args, to8b
from utils.params_utils import merge_hparams
from utils.transform_utils_torch import init_dynamic_gaussians, render, get_state_at_time


def load(path):
    parser = ArgumentParser()
    mp = ModelParams(parser, sentinel=True); hp = ModelHiddenParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--configs', type=str, default='/home/ubuntu/new_sa4d/sa4d/arguments/hypernerf/default.py')
    args = get_combined_args(parser, path, 'scene')
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    args.object_masks = False; args.need_gt_masks = False
    return init_dynamic_gaussians(mp.extract(args), hp.extract(args), args.iteration)


os.chdir('/home/ubuntu/new_sa4d/sa4d')
print('loading scenes...')
g0, sc0, bg = load('./output/hypernerf/oven-mitts_dark')
g1, sc1, _  = load('./output/hypernerf/misc_americano')
g2, sc2, _  = load('./output/hypernerf/split-cookie')
g0.load_mask_table('./output/hypernerf/oven-mitts/segment_results/oven-mitts_delete_mitts_v01.pt')
g1.load_mask_table('./output/hypernerf/misc_americano/segment_results/misc_americano_pseudo_ids16-18_vote8_ff006_q095.pt')
g2.load_mask_table('./output/hypernerf/split-cookie/segment_results/split-cookie_only_cookie_v01.pt')

view = sc0.getTrainCameras()[0]
proj = view.full_proj_transform.cuda()


def project(world_xyz):
    h = torch.cat([world_xyz, torch.ones((world_xyz.shape[0], 1), device=world_xyz.device)], dim=1)
    p = h @ proj
    w = p[:, 3:4].clamp(min=1e-6)
    return p[:, :3] / w


# Step 1: project ALL canonical BG Gaussians and pick those that project within the central 60% of the frame.
xyz_canon = g0._xyz.detach()
ndc = project(xyz_canon)
in_frustum = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)
in_centre = in_frustum & (ndc[:, 0].abs() <= 0.3) & (ndc[:, 1].abs() <= 0.3)
print(f'BG canonical in frustum: {int(in_frustum.sum())} / {xyz_canon.shape[0]}')
print(f'  central 30%:           {int(in_centre.sum())}')

# Use the median (robust to outliers) of the centrally-projected BG world positions.
centre_world = xyz_canon[in_centre].median(0).values.cpu().numpy()
print(f'BG floor (centre, world): {centre_world}')

# Step 2: same for FG canonical, pick the median of the masked Gaussians (robust).
fg1_canon = g1._xyz.detach()[g1._mask_table.any(0)]
fg2_canon = g2._xyz.detach()[g2._mask_table.any(0)]
fg1_med = fg1_canon.median(0).values.cpu().numpy()
fg2_med = fg2_canon.median(0).values.cpu().numpy()
fg1_extent = (fg1_canon.quantile(0.85, dim=0) - fg1_canon.quantile(0.15, dim=0)).cpu().numpy()
fg2_extent = (fg2_canon.quantile(0.85, dim=0) - fg2_canon.quantile(0.15, dim=0)).cpu().numpy()
print(f'FG1 canonical median: {fg1_med}, 70%-extent={fg1_extent}')
print(f'FG2 canonical median: {fg2_med}, 70%-extent={fg2_extent}')

# Step 3: pick scales such that each FG's 70%-extent fits within ~2 world units (counter-sized).
TARGET_OBJ_SIZE = 2.0
sb1 = float(min(0.5, max(0.05, TARGET_OBJ_SIZE / max(fg1_extent[0], fg1_extent[1], 1e-3))))
sb2 = float(min(0.5, max(0.05, TARGET_OBJ_SIZE / max(fg2_extent[0], fg2_extent[1], 1e-3))))
print(f'scales: sb1={sb1:.3f}, sb2={sb2:.3f}')

# Step 4: place FG1 and FG2 around centre_world, separated horizontally.
# motion_bias = target - sb * canonical_centroid (this lands the canonical centroid on target after scale+translate;
# the deformation field then adds per-frame jitter on top, which is exactly the cup-pour / cookie-break motion we want).
sep = 1.5
target1 = centre_world + np.array([-sep, 0.0, 0.0])
target2 = centre_world + np.array([+sep, 0.0, 0.0])
mb1 = target1 - sb1 * fg1_med
mb2 = target2 - sb2 * fg2_med
print(f'mb1 = {[round(float(x),3) for x in mb1]}')
print(f'mb2 = {[round(float(x),3) for x in mb2]}')

# Step 5: render preview at view 0, t=0.
mb0 = torch.zeros(3); rb0 = torch.zeros(3); sb0 = 1.0
mb1_t = torch.tensor(mb1, device='cuda', dtype=torch.float32); rb1 = torch.zeros(3, device='cuda')
mb2_t = torch.tensor(mb2, device='cuda', dtype=torch.float32); rb2 = torch.zeros(3, device='cuda')
with torch.no_grad():
    res = render(view, 0.0, [g0, g1, g2], bg,
                 motion_bias=[mb0, mb1_t, mb2_t], rotation_bias=[rb0, rb1, rb2], scales_bias=[sb0, sb1, sb2],
                 static=[False] * 3, seg=[True] * 3, bg=True)
plt.figure(figsize=(8, 6)); plt.imshow(to8b(res['render']).transpose(1, 2, 0))
plt.title(f'v7 | view=0 t=0 | sb1={sb1:.2f} sb2={sb2:.2f}'); plt.axis('off')
plt.savefig('./output/hypernerf/oven-mitts/seed_artifacts/composite_preview_v7.png', dpi=120, bbox_inches='tight')
plt.close()

train = sc0.getTrainCameras()
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
for vi, vix in enumerate([0, len(train) // 3, 2 * len(train) // 3]):
    v = train[vix]
    for ti, t in enumerate([0.0, 0.5, 1.0]):
        with torch.no_grad():
            r = render(v, t, [g0, g1, g2], bg,
                       motion_bias=[mb0, mb1_t, mb2_t], rotation_bias=[rb0, rb1, rb2], scales_bias=[sb0, sb1, sb2],
                       static=[False] * 3, seg=[True] * 3, bg=True)
        ax[vi, ti].imshow(to8b(r['render']).transpose(1, 2, 0))
        ax[vi, ti].set_title(f'view={vix} t={t:.2f}'); ax[vi, ti].axis('off')
plt.tight_layout()
plt.savefig('./output/hypernerf/oven-mitts/seed_artifacts/composite_preview_v7_grid.png', dpi=110, bbox_inches='tight')
plt.close()

seed = json.load(open('./output/hypernerf/oven-mitts/seed_artifacts/seed.json'))
seed.update({
    'scales_bias1': sb1, 'motion_bias1': [round(float(x), 3) for x in mb1],
    'scales_bias2': sb2, 'motion_bias2': [round(float(x), 3) for x in mb2],
    '_target_world_floor_centre_canonical_median': centre_world.tolist(),
    '_fg1_canonical_median': fg1_med.tolist(),
    '_fg2_canonical_median': fg2_med.tolist(),
    '_method': 'project canonical BG, take central median; place FG canonical median there',
})
json.dump(seed, open('./output/hypernerf/oven-mitts/seed_artifacts/seed.json', 'w'), indent=2)
print('SAVED v7')
