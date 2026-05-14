"""Filter the cookie mask to drop the checkered tablecloth Gaussians.

Strategy: per-Gaussian colour from the SH degree-0 (DC) coefficients.
- Tablecloth = bright blue squares + bright white squares.
- Cookie = brown/tan (R≈0.55, G≈0.45, B≈0.3).
- Hands = pink skin (R≈0.7, G≈0.55, B≈0.5).

Drop a Gaussian if it satisfies ANY of:
  • blue-dominant (B > R + 0.05  AND  B > G + 0.05)        → catches tablecloth blue
  • near-white   (min(R,G,B) > 0.78)                        → catches tablecloth white
  • blue-grey    (B > 0.55  AND  R < 0.5  AND  G < 0.5)    → catches checker shadow

Anything else (warm tones — skin, cookie, brown crumbs) stays.
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
from utils.sh_utils import SH2RGB

SRC_MASK = './output/hypernerf/split-cookie/segment_results/split-cookie_only_cookie_v01.pt'
DST_MASK = './output/hypernerf/split-cookie/segment_results/split-cookie_cookie_hands_v01.pt'

def load(p):
    parser = ArgumentParser(); mp = ModelParams(parser, sentinel=True); hp = ModelHiddenParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--configs', type=str, default='./arguments/hypernerf/default.py')
    args = get_combined_args(parser, p, 'scene')
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    args.object_masks = False; args.need_gt_masks = False
    return init_dynamic_gaussians(mp.extract(args), hp.extract(args), args.iteration)

print('Loading split-cookie scene...')
g2, sc2, bg = load('./output/hypernerf/split-cookie')

src = torch.load(SRC_MASK, map_location='cpu')
mask_table = src['mask_table'].clone()  # (F, N) bool
F, N = mask_table.shape
print(f'src mask: {F} frames x {N} Gaussians, mean active per frame = {mask_table.float().sum(1).mean():.0f}')

# Per-Gaussian DC colour (degree-0 SH → RGB).
sh_dc = g2._features_dc.detach().cpu().squeeze(1)  # (N, 3)
rgb = SH2RGB(sh_dc).clamp(0, 1).numpy()
R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]

# Tablecloth-mask in canonical Gaussian space (per-Gaussian, applied to ALL frames).
is_blue_dominant = (B > R + 0.05) & (B > G + 0.05)
is_near_white    = (np.minimum(np.minimum(R, G), B) > 0.78)
is_blue_grey     = (B > 0.55) & (R < 0.5) & (G < 0.5)
table_drop = is_blue_dominant | is_near_white | is_blue_grey

print(f'colour-based drop counts (canonical):')
print(f'  blue-dominant: {int(is_blue_dominant.sum())} / {N}')
print(f'  near-white   : {int(is_near_white.sum())} / {N}')
print(f'  blue-grey    : {int(is_blue_grey.sum())} / {N}')
print(f'  union DROP   : {int(table_drop.sum())} / {N}')

# Apply the canonical drop to every frame (AND with NOT table_drop).
keep = torch.from_numpy(~table_drop)  # (N,)
new_mask = mask_table & keep.unsqueeze(0)  # broadcast over F
print(f'\nbefore: per-frame  min={mask_table.float().sum(1).min():.0f} mean={mask_table.float().sum(1).mean():.0f} max={mask_table.float().sum(1).max():.0f}')
print(f'after : per-frame  min={new_mask.float().sum(1).min():.0f} mean={new_mask.float().sum(1).mean():.0f} max={new_mask.float().sum(1).max():.0f}')
empty = int((new_mask.float().sum(1) == 0).sum())
print(f'empty frames after filter: {empty}')

# Save it.
out = {
    'mask_table':  new_mask.bool().contiguous(),
    'time_map':    src.get('time_map', torch.linspace(0, 1, F)),
    'inverted_from': src.get('inverted_from', 'split-cookie.pt'),
    'note': 'cookie+hands only — colour-filtered to drop checkered tablecloth (blue/white squares).',
    'colour_filter': {'blue_dominant': True, 'near_white': True, 'blue_grey': True},
}
torch.save(out, DST_MASK)
print(f'\nsaved {DST_MASK}')

# Render preview (3 timestamps) of the filtered cookie alone.
g2.load_mask_table(DST_MASK)
mb_id = torch.zeros(3); rb_id = torch.zeros(3)
cams = sc2.getTrainCameras()
picks = [cams[0], cams[len(cams) // 2], cams[len(cams) - 1]]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
with torch.no_grad():
    for i, v in enumerate(picks):
        r = render(v, float(v.time), [g2], bg,
                   motion_bias=[mb_id], rotation_bias=[rb_id], scales_bias=[1.0],
                   static=[False], seg=[True], bg=True)
        ax[i].imshow(to8b(r['render']).transpose(1, 2, 0))
        ax[i].set_title(f'cookie+hands (no table) | t={float(v.time):.3f}')
        ax[i].axis('off')
plt.tight_layout()
out_png = './output/hypernerf/oven-mitts/seed_artifacts/fg2_cookie_hands_only.png'
plt.savefig(out_png, dpi=110, bbox_inches='tight')
plt.close()
print(f'preview: {out_png}')
