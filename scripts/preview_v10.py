"""Render final composite preview with both clean masks (cup+pour and cookie+hands no tablecloth)."""
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

# Reuse v9 placement (the cookie centroid shifts only slightly after tablecloth filter).
sb0 = 1.0; mb0 = torch.zeros(3); rb0 = torch.zeros(3)
sb1 = 0.331; mb1 = torch.tensor([-3.847, 1.675, 11.435], device='cuda'); rb1 = torch.zeros(3, device='cuda')
sb2 = 0.34;  mb2 = torch.tensor([-1.764, 0.147, 11.730], device='cuda'); rb2 = torch.zeros(3, device='cuda')

train = sc0.getTrainCameras()
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
for vi, vix in enumerate([0, len(train) // 3, 2 * len(train) // 3]):
    v = train[vix]
    for ti, t in enumerate([0.0, 0.5, 1.0]):
        with torch.no_grad():
            r = render(v, t, [g0, g1, g2], bg,
                       motion_bias=[mb0, mb1, mb2], rotation_bias=[rb0, rb1, rb2], scales_bias=[sb0, sb1, sb2],
                       static=[False] * 3, seg=[True] * 3, bg=True)
        ax[vi, ti].imshow(to8b(r['render']).transpose(1, 2, 0))
        ax[vi, ti].set_title(f'view={vix} t={t:.2f}'); ax[vi, ti].axis('off')
plt.tight_layout()
plt.savefig('./output/hypernerf/oven-mitts/seed_artifacts/composite_preview_v10_no_tablecloth.png', dpi=110, bbox_inches='tight')
plt.close()
print('saved v10 grid')
