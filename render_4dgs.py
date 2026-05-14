#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from scene import Scene
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
# from gaussian_renderer import GaussianModel
from scene import GaussianModel
from time import time
# import torch.multiprocessing as mp
import threading
import concurrent.futures

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_single_train_view(gaussians, pipeline, background, cam_type, train_dataset, train_idx, out_png):
    """Render exactly one training camera and save a single PNG (for snapshot PLY / comparisons)."""
    n = len(train_dataset)
    if train_idx < 0 or train_idx >= n:
        raise IndexError(f"train_idx={train_idx} out of range for {n} training views")
    view = train_dataset[train_idx]
    makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    with torch.no_grad():
        rendering = render(view, gaussians, pipeline, background, cam_type=cam_type)["render"]
    torchvision.utils.save_image(rendering, out_png)
    print(f"Saved training view {train_idx} to {out_png}")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, video_output=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    # breakpoint()
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        # breakpoint()
        
        rendering = render(view, gaussians, pipeline, background, cam_type=cam_type)["render"]
        
        # rendering = render(view, gaussians, pipeline, background,cam_type=cam_type)#["render"]
        # gaussians.save_ply("./deformed_point_cloud.ply")
        # sys.exit(0)
        
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        render_images.append(to8b(rendering).transpose(1,2,0))
        # print(to8b(rendering).shape)
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_list.append(gt)
        # if idx >= 10:
            # break
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    # print("writing training images.")

    multithread_write(gt_list, gts_path)
    # print("writing rendering images.")

    multithread_write(render_list, render_path)

    
    if video_output is None:
        video_output = os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4')
    makedirs(os.path.dirname(video_output), exist_ok=True)
    try:
        import imageio  # lazy import: only needed when writing videos
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'imageio'. Install it (e.g. `pip install imageio`) or run with --skip_video."
        ) from e
    imageio.mimwrite(video_output, render_images, fps=30)
    print(f"Video saved to {video_output}")
    
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, mode: str, ply_path: str = None, video_output: str = None, mask_path: str = None, composite: bool = False, snapshot_ply: bool = False, train_idx: int = None, train_image_output: str = None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, mode, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, mode=mode, shuffle=False)
        if ply_path is not None:
            print(f"Overriding PLY with: {ply_path}")
            gaussians.load_ply(ply_path)
        if snapshot_ply:
            n = gaussians.get_xyz.shape[0]
            gaussians._deformation_table = torch.zeros((n,), dtype=torch.bool, device="cuda")
            print(f"snapshot_ply: disabled timestep deformation for {n} Gaussians (_deformation_table all False)")
        if composite and mask_path is not None:
            mask_data = torch.load(mask_path, map_location='cuda')
            fg_mask = mask_data['mask_table'].any(dim=0).bool()
            gaussians._deformation_table = ~fg_mask
            n_static = fg_mask.sum().item()
            print(f"Composite mode: {(~fg_mask).sum().item()} deformed, {n_static} static (foreground skip deformation)")
        cam_type = scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if train_idx is not None:
            if train_image_output is None:
                raise ValueError("train_image_output is required when train_idx is set")
            render_single_train_view(gaussians, pipeline, background, cam_type, scene.getTrainCameras(), train_idx, train_image_output)
        elif not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, cam_type, video_output=video_output)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, cam_type, video_output=video_output)
        if not skip_video:
            render_set(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, cam_type, video_output=video_output)
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--mode", type=str, default="scene")
    parser.add_argument("--ply_path", type=str, default=None, help="Override which .ply file to render")
    parser.add_argument("--mask_path", type=str, default=None, help="Mask .pt file — used with --composite to skip deformation on foreground")
    parser.add_argument("--composite", action="store_true", help="Composite mode: foreground gaussians (from mask) skip deformation")
    parser.add_argument("--video_output", type=str, default=None, help="Override output video path")
    parser.add_argument("--snapshot_ply", action="store_true", help="After --ply_path load, disable 4D deformation so the PLY is rendered as a static snapshot")
    parser.add_argument("--train_idx", type=int, default=None, help="If set, render only this training camera index and save --train_image_output (skips full train set / video path)")
    parser.add_argument("--train_image_output", type=str, default=None, help="Output PNG path when --train_idx is set")
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.mode, ply_path=getattr(args, 'ply_path', None), video_output=getattr(args, 'video_output', None), mask_path=getattr(args, 'mask_path', None), composite=getattr(args, 'composite', False), snapshot_ply=getattr(args, "snapshot_ply", False), train_idx=getattr(args, "train_idx", None), train_image_output=getattr(args, "train_image_output", None))