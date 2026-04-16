"""
Data loading utilities for the harmonization pipeline.

This module handles:
  - Loading the 4D Gaussian model (.ply + deformation weights)
  - Loading the per-Gaussian mask table (.pt)
  - Loading camera views from the scene
  - Loading the pretrained Harmonizer model

Key methods:
  load_scene()          -> (gaussians, scene, pipeline_params, background)
  load_mask_table()     -> dict with 'mask_table', 'time_map', etc.
  load_harmonizer()     -> nn.Module (pretrained Harmonizer)
  get_frame_mask()      -> bool tensor [N_gaussians] for a given frame
  get_object_mask()     -> bool tensor [N_gaussians] union across all frames
"""

import os
import sys
import torch
from argparse import ArgumentParser, Namespace

# Add sa4d root to path so imports work
SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)

# Add Harmonizer src to path
HARMONIZER_ROOT = os.path.join(os.path.dirname(SA4D_ROOT), '..', 'Harmonizer')
HARMONIZER_SRC = os.path.join(HARMONIZER_ROOT, 'src')
if os.path.isdir(HARMONIZER_SRC) and HARMONIZER_SRC not in sys.path:
    sys.path.insert(0, HARMONIZER_SRC)


def load_scene(model_path, source_path, iteration=-1, configs=None):
    """
    Load a trained 4DGS scene: Gaussians + deformation field + cameras.

    Args:
        model_path:  e.g. 'output/hypernerf/split-cookie'
        source_path: e.g. 'data/hypernerf/split-cookie'
        iteration:   checkpoint iteration (-1 = latest)
        configs:     path to a .py config file (optional, e.g. 'arguments/hypernerf/default.py')

    Returns:
        gaussians:       GaussianModel with loaded .ply + deformation
        scene:           Scene object (holds train/test/video cameras)
        pipeline_params: PipelineParams namespace
        background:      background color tensor on CUDA
    """
    from arguments import ModelParams, PipelineParams, ModelHiddenParams
    from scene import Scene, GaussianModel

    # Build a minimal args namespace matching cfg_args format
    cfg_path = os.path.join(model_path, 'cfg_args')
    with open(cfg_path) as f:
        args = eval(f.read())

    # Override paths to be absolute
    args.model_path = os.path.abspath(model_path)
    args.source_path = os.path.abspath(source_path)

    # If a .py config is given, merge hyperparams from it
    if configs is not None:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(configs)
        args = merge_hparams(args, config)

    # Extract sub-param groups
    # PipelineParams lives directly on args
    pipe = Namespace(
        convert_SHs_python=getattr(args, 'convert_SHs_python', False),
        compute_cov3D_python=getattr(args, 'compute_cov3D_python', False),
        debug=getattr(args, 'debug', False),
    )

    # ModelHiddenParams — pull all fields that exist on args
    hyperparam = Namespace()
    hp_defaults = {
        'net_width': 64, 'timebase_pe': 4, 'defor_depth': 1,
        'posebase_pe': 10, 'scale_rotation_pe': 2, 'opacity_pe': 2,
        'timenet_width': 64, 'timenet_output': 32, 'bounds': 1.6,
        'plane_tv_weight': 0.0001, 'time_smoothness_weight': 0.01,
        'l1_time_planes': 0.0001,
        'kplanes_config': {'grid_dimensions': 2, 'input_coordinate_dim': 4,
                           'output_coordinate_dim': 32, 'resolution': [64, 64, 64, 25]},
        'multires': [1, 2, 4, 8],
        'no_dx': False, 'no_grid': False, 'no_ds': False, 'no_dr': False,
        'no_do': True, 'no_dshs': True, 'empty_voxel': False,
        'grid_pe': 0, 'static_mlp': False, 'apply_rotation': False,
    }
    for k, v in hp_defaults.items():
        setattr(hyperparam, k, getattr(args, k, v))

    # Build GaussianModel and Scene
    sh_degree = getattr(args, 'sh_degree', 3)
    gaussians = GaussianModel(sh_degree, 'scene', hyperparam)
    scene = Scene(args, gaussians, load_iteration=iteration, mode='scene', shuffle=False)

    bg_color = [1, 1, 1] if getattr(args, 'white_background', True) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    print(f"[data_loading] Loaded {gaussians._xyz.shape[0]} Gaussians "
          f"at iteration {scene.loaded_iter}")

    return gaussians, scene, pipe, background


def load_mask_table(mask_pt_path):
    """
    Load a .pt mask table produced by the segmentation pipeline.

    Args:
        mask_pt_path: path to e.g. 'segment_results/split-cookie.pt'

    Returns:
        dict with keys:
          'mask_table': bool tensor [N_frames, N_gaussians]
          'time_map':   float tensor [N_frames] in [0, 1]
          + any extra keys (prob_thresh, interval, removed_ids)
    """
    data = torch.load(mask_pt_path, map_location='cuda')
    print(f"[data_loading] Mask table: {data['mask_table'].shape[0]} frames, "
          f"{data['mask_table'].shape[1]} gaussians, "
          f"{data['mask_table'].float().mean():.1%} active")
    return data


def get_frame_mask(mask_data, frame_idx):
    """
    Get per-Gaussian boolean mask for a specific frame.

    Args:
        mask_data:  dict from load_mask_table()
        frame_idx:  int, index into mask_table

    Returns:
        bool tensor [N_gaussians] on CUDA
    """
    return mask_data['mask_table'][frame_idx]


def get_object_mask(mask_data):
    """
    Get per-Gaussian boolean mask: union across ALL frames.
    A Gaussian is True if it belongs to the object in any frame.

    Returns:
        bool tensor [N_gaussians] on CUDA
    """
    return mask_data['mask_table'].any(dim=0)


def time_to_frame_idx(mask_data, t):
    """
    Find the closest frame index for a given normalized time t in [0, 1].

    Args:
        mask_data: dict from load_mask_table()
        t:         float, normalized time

    Returns:
        int frame index
    """
    time_map = mask_data['time_map']
    return (time_map - t).abs().argmin().item()


def load_harmonizer(pretrained_path=None):
    """
    Load the pretrained Harmonizer model.

    Args:
        pretrained_path: path to harmonizer.pth weights.
            Defaults to ~/Harmonizer/pretrained/harmonizer.pth

    Returns:
        Harmonizer model in eval mode on CUDA
    """
    if pretrained_path is None:
        pretrained_path = os.path.expanduser('~/Harmonizer/pretrained/harmonizer.pth')

    from model.harmonizer import Harmonizer
    harmonizer = Harmonizer()
    state_dict = torch.load(pretrained_path, map_location='cpu')
    harmonizer.load_state_dict(state_dict)
    harmonizer = harmonizer.cuda().eval()
    print(f"[data_loading] Loaded Harmonizer from {pretrained_path}")
    return harmonizer
