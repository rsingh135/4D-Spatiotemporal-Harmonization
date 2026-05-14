"""Verify the new mask-sanity cell logic outside the notebook:
1) report per-frame counts;
2) auto-fill empty frames in FG1 mask;
3) render each FG alone (in its own scene) at 3 timestamps;
4) verify no crash on the BG video-camera path with all 3 FGs after the fix.
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

def _report_and_fix_mask(name, g):
    counts = g._mask_table.float().sum(dim=1).cpu().numpy()
    F_total = counts.shape[0]
    n_empty = int((counts == 0).sum())
    print(f'{name}: frames={F_total}, active per-frame  min={counts.min():.0f} mean={counts.mean():.0f} max={counts.max():.0f}, empty_frames={n_empty}')
    if n_empty == 0:
        return 0
    nonempty_idx = np.where(counts > 0)[0]
    if nonempty_idx.size == 0:
        print(f'  WARNING: {name} mask entirely empty')
        return 0
    fixed = 0
    empty_frames_list = list(np.where(counts == 0)[0])
    print(f'  empty frames: {empty_frames_list[:20]}{"..." if len(empty_frames_list) > 20 else ""}')
    for i in np.where(counts == 0)[0]:
        nearest = nonempty_idx[np.argmin(np.abs(nonempty_idx - i))]
        g._mask_table[i] = g._mask_table[nearest].clone()
        fixed += 1
    new_counts = g._mask_table.float().sum(dim=1).cpu().numpy()
    print(f'  filled {fixed} empty frames -> active per-frame  min={new_counts.min():.0f} mean={new_counts.mean():.0f} max={new_counts.max():.0f}')
    return fixed

print('Step 1: per-frame mask counts + auto-fill')
_report_and_fix_mask('FG1 (cup+pour, americano)', g1)
_report_and_fix_mask('FG2 (cookie, split-cookie)', g2)

# Step 2: render FG1 alone in americano at 3 timestamps, FG2 alone in cookie scene at 3 timestamps.
def three(cams):
    n = len(cams)
    return [cams[0], cams[n // 2], cams[n - 1]]

mb_id = torch.zeros(3); rb_id = torch.zeros(3)
fg1_views = three(sc1.getTrainCameras())
fg2_views = three(sc2.getTrainCameras())
print('\nStep 2a: render FG1 alone (americano, white BG)')
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
with torch.no_grad():
    for col, v in enumerate(fg1_views):
        r = render(v, float(v.time), [g1], bg, motion_bias=[mb_id], rotation_bias=[rb_id], scales_bias=[1.0],
                   static=[False], seg=[True], bg=True)
        axes[0, col].imshow(to8b(r['render']).transpose(1, 2, 0))
        axes[0, col].set_title(f'FG1 cup+pour | t={float(v.time):.3f}'); axes[0, col].axis('off')
    print('Step 2b: render FG2 alone (split-cookie, white BG)')
    for col, v in enumerate(fg2_views):
        r = render(v, float(v.time), [g2], bg, motion_bias=[mb_id], rotation_bias=[rb_id], scales_bias=[1.0],
                   static=[False], seg=[True], bg=True)
        axes[1, col].imshow(to8b(r['render']).transpose(1, 2, 0))
        axes[1, col].set_title(f'FG2 cookie | t={float(v.time):.3f}'); axes[1, col].axis('off')
plt.tight_layout()
out_png = './output/hypernerf/oven-mitts/seed_artifacts/fg_mask_sanity.png'
plt.savefig(out_png, dpi=110, bbox_inches='tight')
plt.close()
print(f'  saved {out_png}')

# Step 3: simulate the mp4 render loop on the BG video cams to confirm no crash post-patch.
print('\nStep 3: walk all BG video cameras through render() with all 3 FGs (mp4 simulation)')
sb1 = 0.20; mb1 = torch.tensor([-4.727, 0.662, 12.182], device='cuda'); rb1 = torch.zeros(3, device='cuda')
sb2 = 0.34; mb2 = torch.tensor([-1.764, 0.147, 11.730], device='cuda'); rb2 = torch.zeros(3, device='cuda')
sb0 = 1.0; mb0 = torch.zeros(3); rb0 = torch.zeros(3)
video_cams = sc0.getVideoCameras()
print(f'  {len(video_cams)} video cameras')
errors = 0
empty_t = []
with torch.no_grad():
    for i, v in enumerate(video_cams):
        try:
            r = render(v, float(v.time), [g0, g1, g2], bg,
                       motion_bias=[mb0, mb1, mb2], rotation_bias=[rb0, rb1, rb2], scales_bias=[sb0, sb1, sb2],
                       static=[False] * 3, seg=[True] * 3, bg=True)
        except Exception as e:
            errors += 1
            empty_t.append((i, float(v.time), str(e)[:120]))
        if i % 20 == 0:
            print(f'    frame {i}/{len(video_cams)} t={float(v.time):.3f} ok')
print(f'  done: {errors} errors out of {len(video_cams)} frames')
for i, t, e in empty_t:
    print(f'    frame {i} t={t:.3f}: {e}')
