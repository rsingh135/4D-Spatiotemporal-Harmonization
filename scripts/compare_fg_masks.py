"""For each candidate mask in americano + split-cookie, report stats and render the masked
Gaussians in their native scene at a few timestamps. Helps pick the best cup+pour and cookie masks.

DELETE masks are inverted on load (True = KEEP becomes True = REMOVE, so we flip).
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

print('Loading americano (FG1) and split-cookie (FG2)...')
g1, sc1, bg = load('./output/hypernerf/misc_americano')
g2, sc2, _ = load('./output/hypernerf/split-cookie')

mb_id = torch.zeros(3); rb_id = torch.zeros(3)


def quick_stats(p):
    d = torch.load(p, map_location='cpu')
    mt = d.get('mask_table') if isinstance(d, dict) else d  # some files store dict, some plain tensor
    if mt is None:
        return None, None
    counts = mt.float().sum(dim=1).numpy()
    return counts, mt.shape


def render_fg_at(g, scene, three_t, label, ax_row):
    cams = scene.getTrainCameras()
    n = len(cams)
    picks = [cams[0], cams[n // 2], cams[n - 1]]
    with torch.no_grad():
        for col, v in enumerate(picks):
            r = render(v, float(v.time), [g], bg, motion_bias=[mb_id], rotation_bias=[rb_id], scales_bias=[1.0],
                       static=[False], seg=[True], bg=True)
            ax_row[col].imshow(to8b(r['render']).transpose(1, 2, 0))
            ax_row[col].set_title(f'{label} | t={float(v.time):.2f}'); ax_row[col].axis('off')


# ─── Americano cup+pour candidates ───
# I'll test the existing pseudo masks (which are KEEP-style, True = include the object).
# Skip the DELETE-style files for now (would need inversion logic).
fg1_dir = './output/hypernerf/misc_americano/segment_results'
am_keep_candidates = [
    'misc_americano_ids16-18_q0.95_mc0.pt',
    'misc_americano_pseudo_ids16-18_q0.95.pt',
    'misc_americano_pseudo_ids16-18_q0.95_trimdef.pt',
    'misc_americano_pseudo_ids16-18_q0.95_trimdef_pp.pt',
    'misc_americano_pseudo_ids16-18_q0.95_trimdef_pp_minfrac25.pt',
    'misc_americano_pseudo_ids16-18_vote8_ff006_q095.pt',
    'misc_americano_mc0.5_q0.95_ids18.pt',
]

# I will also try INVERTING the delete masks so True = removed object (cup+pour).
am_delete_candidates = [
    ('misc_americano_delete_mc0.8_q0.9.pt', '_inv'),
    ('misc_americano_delete_rulesumprob_mc0.8_mrp0.35_noq_norad.pt', '_inv'),
    ('misc_americano_delete_rulesumprob_mc0.8_mrp0.35_noq_norad_post_mf0.2_k16_r0.06.pt', '_inv'),
]

print('\n=== AMERICANO CUP+POUR mask candidates ===')
for fname in am_keep_candidates:
    p = os.path.join(fg1_dir, fname)
    counts, shape = quick_stats(p)
    if counts is None:
        print(f'  {fname}: SKIP (not a mask_table dict)')
        continue
    print(f'  {fname}: shape={shape}, per-frame  min={counts.min():.0f} mean={counts.mean():.0f} max={counts.max():.0f}, empty={int((counts==0).sum())}')
for fname, _ in am_delete_candidates:
    p = os.path.join(fg1_dir, fname)
    d = torch.load(p, map_location='cpu')
    if isinstance(d, dict) and 'mask_table' in d:
        mt = d['mask_table']
        # Delete: True = KEEP, False = DELETE. The "object to delete" = ~mask_table.
        inv_counts = (~mt).float().sum(dim=1).numpy()
        print(f'  {fname} (INVERTED): shape={mt.shape}, per-frame  min={inv_counts.min():.0f} mean={inv_counts.mean():.0f} max={inv_counts.max():.0f}, empty={int((inv_counts==0).sum())}')

# Render the top 4 candidates side-by-side. Pick a small interesting subset.
to_render_am = [
    ('misc_americano_pseudo_ids16-18_vote8_ff006_q095.pt', None, 'CURRENT vote8'),
    ('misc_americano_pseudo_ids16-18_q0.95.pt', None, 'pseudo q0.95'),
    ('misc_americano_delete_mc0.8_q0.9.pt', 'invert', 'INV delete mc0.8 (24k)'),
    ('misc_americano_delete_rulesumprob_mc0.8_mrp0.35_noq_norad.pt', 'invert', 'INV delete rulesumprob (54k)'),
]

n = len(to_render_am)
fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
if n == 1: axes = [axes]
for row, (fname, action, label) in enumerate(to_render_am):
    p = os.path.join(fg1_dir, fname)
    if action == 'invert':
        d = torch.load(p, map_location='cpu')
        d['mask_table'] = (~d['mask_table']).bool()
        torch.save(d, '/tmp/_tmp_am_invert.pt')
        g1.load_mask_table('/tmp/_tmp_am_invert.pt')
        os.remove('/tmp/_tmp_am_invert.pt')
    else:
        g1.load_mask_table(p)
    render_fg_at(g1, sc1, None, label, axes[row])
plt.tight_layout()
out_am = './output/hypernerf/oven-mitts/seed_artifacts/fg1_mask_candidates.png'
plt.savefig(out_am, dpi=110, bbox_inches='tight')
plt.close()
print(f'\nsaved {out_am}')

# ─── Split-cookie candidates ───
print('\n=== SPLIT-COOKIE candidates ===')
fg2_dir = './output/hypernerf/split-cookie/segment_results'
ck_files = [
    ('split-cookie.pt', 'invert'),  # original DELETE mask, inverted
    ('split-cookie_only_cookie_v01.pt', None),  # current (we built earlier)
    ('composite_inserted_choc_Bigger_clean.pt', None),  # may already be a KEEP mask
]
for fname, action in ck_files:
    p = os.path.join(fg2_dir, fname)
    d = torch.load(p, map_location='cpu')
    if isinstance(d, dict) and 'mask_table' in d:
        mt = d['mask_table'] if action != 'invert' else (~d['mask_table'])
        counts = mt.float().sum(dim=1).numpy()
        print(f'  {fname}{" (INV)" if action=="invert" else ""}: shape={mt.shape}, per-frame  min={counts.min():.0f} mean={counts.mean():.0f} max={counts.max():.0f}, empty={int((counts==0).sum())}')
    else:
        print(f'  {fname}: not a dict with mask_table (type={type(d)})')

# Skip composite_inserted_choc_Bigger_clean.pt — has wrong Gaussian count (built for a composite scene).
to_render_ck = [
    ('split-cookie.pt', 'invert', 'INV split-cookie.pt'),
    ('split-cookie_only_cookie_v01.pt', None, 'CURRENT only_cookie_v01'),
]
n = len(to_render_ck)
fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
if n == 1: axes = [axes]
for row, (fname, action, label) in enumerate(to_render_ck):
    p = os.path.join(fg2_dir, fname)
    if action == 'invert':
        d = torch.load(p, map_location='cpu')
        d['mask_table'] = (~d['mask_table']).bool()
        tmp = '/tmp/_tmp_ck_invert.pt'; torch.save(d, tmp)
        g2.load_mask_table(tmp); os.remove(tmp)
    else:
        g2.load_mask_table(p)
    render_fg_at(g2, sc2, None, label, axes[row])
plt.tight_layout()
out_ck = './output/hypernerf/oven-mitts/seed_artifacts/fg2_mask_candidates.png'
plt.savefig(out_ck, dpi=110, bbox_inches='tight')
plt.close()
print(f'saved {out_ck}')
