"""Validate updated seed: sb1=0.20, sb2=0.34, mb1=[-4.727, 0.662, 12.182], mb2=[-1.764, 0.147, 11.73]."""
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
from arguments import ModelParams, ModelHiddenParams
from utils.segment_utils import get_combined_args, to8b
from utils.params_utils import merge_hparams
from utils.transform_utils_torch import init_dynamic_gaussians, render

os.chdir('/home/ubuntu/new_sa4d/sa4d')

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
g1.load_mask_table('./output/hypernerf/misc_americano/segment_results/misc_americano_pseudo_ids16-18_vote8_ff006_q095.pt')
g2.load_mask_table('./output/hypernerf/split-cookie/segment_results/split-cookie_only_cookie_v01.pt')

sb0 = 1.0; mb0 = torch.zeros(3); rb0 = torch.zeros(3)
sb1 = 0.20; mb1 = torch.tensor([-4.727, 0.662, 12.182], device='cuda'); rb1 = torch.zeros(3, device='cuda')
sb2 = 0.34; mb2 = torch.tensor([-1.764, 0.147, 11.730], device='cuda'); rb2 = torch.zeros(3, device='cuda')

train = sc0.getTrainCameras()
view0 = train[0]
with torch.no_grad():
    res = render(view0, 0.0, [g0, g1, g2], bg,
                 motion_bias=[mb0, mb1, mb2], rotation_bias=[rb0, rb1, rb2], scales_bias=[sb0, sb1, sb2],
                 static=[False] * 3, seg=[True] * 3, bg=True)
plt.figure(figsize=(8, 6)); plt.imshow(to8b(res['render']).transpose(1, 2, 0))
plt.title(f'v8 | view=0 t=0 | sb1={sb1} sb2={sb2}'); plt.axis('off')
plt.savefig('./output/hypernerf/oven-mitts/seed_artifacts/composite_preview_v8.png', dpi=120, bbox_inches='tight')
plt.close()

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
plt.savefig('./output/hypernerf/oven-mitts/seed_artifacts/composite_preview_v8_grid.png', dpi=110, bbox_inches='tight')
plt.close()
print('done v8')
