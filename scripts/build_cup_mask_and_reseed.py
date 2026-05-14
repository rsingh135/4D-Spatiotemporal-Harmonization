"""Build a clean cup+pour mask (inverted DELETE), re-seed FG1 placement using it,
and validate the full multi-view composite."""
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

# 1. Build cup+pour KEEP mask by inverting the DELETE mask.
FG1_DELETE = './output/hypernerf/misc_americano/segment_results/misc_americano_delete_mc0.8_q0.9.pt'
FG1_KEEP = './output/hypernerf/misc_americano/segment_results/misc_americano_cup_pour_v01.pt'
print(f'building {FG1_KEEP} from inverted {FG1_DELETE}...')
d = torch.load(FG1_DELETE, map_location='cpu')
print(f'  source keys: {list(d.keys())}')
d['mask_table'] = (~d['mask_table']).bool()
counts = d['mask_table'].float().sum(dim=1).numpy()
print(f'  inverted per-frame  min={counts.min():.0f} mean={counts.mean():.0f} max={counts.max():.0f}, empty={(counts==0).sum()}')
torch.save(d, FG1_KEEP)
print(f'  saved {FG1_KEEP}')

def load(p):
    parser = ArgumentParser(); mp = ModelParams(parser, sentinel=True); hp = ModelHiddenParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--configs', type=str, default='./arguments/hypernerf/default.py')
    args = get_combined_args(parser, p, 'scene')
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    args.object_masks = False; args.need_gt_masks = False
    return init_dynamic_gaussians(mp.extract(args), hp.extract(args), args.iteration)

# 2. Reload all scenes + masks (with the new cup mask).
g0, sc0, bg = load('./output/hypernerf/oven-mitts_dark')
g1, sc1, _ = load('./output/hypernerf/misc_americano')
g2, sc2, _ = load('./output/hypernerf/split-cookie')
g0.load_mask_table('./output/hypernerf/oven-mitts/segment_results/oven-mitts_delete_mitts_v01.pt')
g1.load_mask_table(FG1_KEEP)
g2.load_mask_table('./output/hypernerf/split-cookie/segment_results/split-cookie_only_cookie_v01.pt')

# 3. Re-seed FG1 placement using new mask. Reuse seed.json's BG floor anchor.
seed = json.load(open('./output/hypernerf/oven-mitts/seed_artifacts/seed.json'))
centre_world = np.array(seed['_target_world_floor_centre_canonical_median'])
fg2_med = np.array(seed['_fg2_canonical_median'])
sb2 = float(seed['scales_bias2'])
mb2 = np.array(seed['motion_bias2'])

# FG1 canonical median (cup+pour mask is dense + clean → median is meaningful).
fg1_canon = g1._xyz.detach()[g1._mask_table.any(0)]
fg1_med = fg1_canon.median(0).values.cpu().numpy()
# 70%-extent for scale fit.
fg1_extent = (fg1_canon.quantile(0.85, dim=0) - fg1_canon.quantile(0.15, dim=0)).cpu().numpy()
print(f'\nnew FG1 canonical median: {fg1_med}, 70%-extent={fg1_extent}')

TARGET_OBJ_SIZE = 2.0
sb1 = float(min(0.5, max(0.05, TARGET_OBJ_SIZE / max(fg1_extent[0], fg1_extent[1], 1e-3))))
sep = 1.5
target1 = centre_world + np.array([-sep, 0.0, 0.0])
mb1 = target1 - sb1 * fg1_med
print(f'new mb1={[round(float(x),3) for x in mb1]}, sb1={sb1:.3f}')

# 4. Render preview at view 0, t=0 with the new mask.
view = sc0.getTrainCameras()[0]
mb1_t = torch.tensor(mb1, device='cuda', dtype=torch.float32); rb1 = torch.zeros(3, device='cuda')
mb2_t = torch.tensor(mb2, device='cuda', dtype=torch.float32); rb2 = torch.zeros(3, device='cuda')
mb0 = torch.zeros(3); rb0 = torch.zeros(3); sb0 = 1.0

with torch.no_grad():
    res = render(view, 0.0, [g0, g1, g2], bg,
                 motion_bias=[mb0, mb1_t, mb2_t], rotation_bias=[rb0, rb1, rb2], scales_bias=[sb0, sb1, sb2],
                 static=[False] * 3, seg=[True] * 3, bg=True)
plt.figure(figsize=(8, 6)); plt.imshow(to8b(res['render']).transpose(1, 2, 0))
plt.title(f'v9 (clean cup mask) | view=0 t=0 | sb1={sb1:.2f} sb2={sb2:.2f}'); plt.axis('off')
plt.savefig('./output/hypernerf/oven-mitts/seed_artifacts/composite_preview_v9.png', dpi=120, bbox_inches='tight')
plt.close()

# Multi-view multi-time grid
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
train = sc0.getTrainCameras()
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
plt.savefig('./output/hypernerf/oven-mitts/seed_artifacts/composite_preview_v9_grid.png', dpi=110, bbox_inches='tight')
plt.close()

# Update seed
seed.update({'scales_bias1': sb1, 'motion_bias1': [round(float(x), 3) for x in mb1],
             '_fg1_canonical_median_v9': fg1_med.tolist(),
             '_fg1_mask_path_v9': FG1_KEEP})
json.dump(seed, open('./output/hypernerf/oven-mitts/seed_artifacts/seed.json', 'w'), indent=2)
print('\nDONE — preview at composite_preview_v9.png + grid at composite_preview_v9_grid.png')
