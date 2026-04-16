"""Render only the masked Gaussians to verify which object the mask selects."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchvision
import sys

SA4D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SA4D_ROOT not in sys.path:
    sys.path.insert(0, SA4D_ROOT)

from pipeline.data_loading import load_scene, load_mask_table, get_object_mask
from gaussian_renderer import render_segmentation

gaussians, scene, pipe, bg = load_scene(
    'output/hypernerf/torchocolate', 'data/hypernerf/torchocolate')
mask_data = load_mask_table(
    'output/hypernerf/torchocolate/segment_results/torchocolate.pt')
obj_mask = get_object_mask(mask_data)

view = scene.getTrainCameras()[30]
with torch.no_grad():
    result = render_segmentation(view, gaussians, pipe, bg, mask=obj_mask)
    torchvision.utils.save_image(result['render'], 'masked_object_only.png')
print('Saved masked_object_only.png')
