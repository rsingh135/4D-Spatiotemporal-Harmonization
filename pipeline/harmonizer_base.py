"""
Harmonizer abstraction layer.

Provides a unified interface for different image harmonization models.
Each harmonizer takes a composite image + mask and returns a harmonized image.

Supported backends:
  - "whitebox"  : Original Harmonizer (6 white-box filters, Ke et al.)
  - "pctnet"    : PCT-Net (CVPR 2023, per-pixel color transfer)

Usage:
    from pipeline.harmonizer_base import create_harmonizer
    harmonizer = create_harmonizer("pctnet")
    target = harmonizer.harmonize(composite, mask_2d)  # both [1,3,H,W] / [1,1,H,W]
"""

import os
import sys
import torch
import numpy as np
from abc import ABC, abstractmethod


class HarmonizerBase(ABC):
    """Abstract base for all harmonizer backends."""

    @abstractmethod
    def harmonize(self, composite, mask_2d):
        """
        Harmonize the foreground region of a composite image.

        Args:
            composite: tensor [1, 3, H, W] in [0, 1] on CUDA
            mask_2d:   tensor [1, 1, H, W] in [0, 1] on CUDA

        Returns:
            tensor [1, 3, H, W] in [0, 1] — the harmonized image
        """
        pass


class WhiteboxHarmonizer(HarmonizerBase):
    """Original Harmonizer (Ke et al.) — 6 global white-box filters."""

    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.expanduser('~/Harmonizer/pretrained/harmonizer.pth')

        HARMONIZER_ROOT = os.path.expanduser('~/Harmonizer')
        HARMONIZER_SRC = os.path.join(HARMONIZER_ROOT, 'src')
        if os.path.isdir(HARMONIZER_SRC) and HARMONIZER_SRC not in sys.path:
            sys.path.insert(0, HARMONIZER_SRC)

        from model.harmonizer import Harmonizer
        self.model = Harmonizer()
        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model = self.model.cuda().eval()
        print(f"[harmonizer] Loaded WhiteboxHarmonizer from {weights_path}")

    def harmonize(self, composite, mask_2d):
        with torch.no_grad():
            arguments = self.model.predict_arguments(composite, mask_2d)
            outputs = self.model.restore_image(composite, mask_2d, arguments)
            return outputs[-1]  # final image after all 6 filters

    @property
    def raw_model(self):
        """Access the underlying model for predict_arguments / restore_image."""
        return self.model


class PCTNetHarmonizer(HarmonizerBase):
    """PCT-Net (CVPR 2023) — per-pixel color transfer network."""

    def __init__(self, weights_path=None, model_type='CNN_pct'):
        if weights_path is None:
            weights_path = os.path.expanduser(
                '~/PCT-Net-Image-Harmonization/pretrained_models/PCTNet_CNN.pth')

        PCTNET_ROOT = os.path.expanduser('~/PCT-Net-Image-Harmonization')
        if PCTNET_ROOT not in sys.path:
            sys.path.insert(0, PCTNET_ROOT)

        from iharm.inference.utils import load_model
        from iharm.inference.predictor import Predictor

        self.net = load_model(model_type, weights_path, verbose=False)
        self.device = torch.device('cuda')
        self.net.to(self.device)
        self.predictor = Predictor(self.net, self.device, with_flip=False)
        print(f"[harmonizer] Loaded PCTNetHarmonizer ({model_type}) from {weights_path}")

    def harmonize(self, composite, mask_2d):
        """
        PCT-Net expects numpy uint8 images at two resolutions.
        We convert from torch [1,3,H,W] float in [0,1] and back.
        """
        # Convert to numpy HxWx3 uint8
        comp_np = (composite.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        mask_np = mask_2d.squeeze(0).squeeze(0).cpu().numpy()  # HxW float

        H, W = comp_np.shape[:2]

        # PCT-Net needs a low-res (256x256) version
        import cv2
        comp_lr = cv2.resize(comp_np, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_lr = cv2.resize(mask_np, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Run inference
        _, out_hr = self.predictor.predict(comp_lr, comp_np, mask_lr, mask_np, return_numpy=True)

        # Convert back to torch [1, 3, H, W] float in [0, 1]
        out_tensor = torch.from_numpy(out_hr.astype(np.float32) / 255.0)
        out_tensor = out_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
        return out_tensor.clamp(0, 1)


class SceneBHarmonizer(HarmonizerBase):
    """Use pre-rendered Scene B (ground truth) images directly as targets.

    Instead of running a neural harmonizer, this loads the corresponding
    Scene B image for each view.  The Scene B directory must have the same
    file layout as Scene A (transforms_train.json with file_path entries
    like ``./train/r_0``).
    """

    def __init__(self, scene_b_dir):
        from PIL import Image as _Image
        self._Image = _Image
        self.scene_b_dir = os.path.abspath(scene_b_dir)
        # Build index: filename stem -> full path
        self._images = {}
        for split in ('train', 'test'):
            d = os.path.join(self.scene_b_dir, split)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith('.png'):
                    stem = os.path.splitext(fn)[0]  # e.g. "r_0"
                    self._images[(split, stem)] = os.path.join(d, fn)
        print(f"[harmonizer] SceneBHarmonizer: indexed {len(self._images)} "
              f"images from {self.scene_b_dir}")

    def get_target_for_view(self, view_name, H, W):
        """Load the Scene B image matching a view name like 'train/r_0'."""
        # view_name can be './train/r_0' or 'train/r_0' etc.
        name = view_name.lstrip('./')
        parts = name.split('/')
        if len(parts) == 2:
            split, stem = parts
        else:
            split, stem = 'train', parts[-1]
        stem = os.path.splitext(stem)[0]

        key = (split, stem)
        if key not in self._images:
            raise FileNotFoundError(
                f"Scene B image not found for {key}. "
                f"Available: {list(self._images.keys())[:5]}...")

        img = self._Image.open(self._images[key]).convert('RGB')
        img = img.resize((W, H), self._Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
        return tensor.clamp(0, 1)

    def harmonize(self, composite, mask_2d):
        raise NotImplementedError(
            "SceneBHarmonizer requires view names — use "
            "precompute_all_targets with harmonizer='scene_b' instead.")


def create_harmonizer(backend='whitebox', weights_path=None, **kwargs):
    """
    Factory function to create a harmonizer.

    Args:
        backend:      "whitebox", "pctnet", or "scene_b"
        weights_path: path to pretrained weights (None = default location)
        **kwargs:     extra args passed to the harmonizer constructor
                      For scene_b: scene_b_dir=<path to scene_B/>

    Returns:
        HarmonizerBase instance
    """
    if backend == 'whitebox':
        return WhiteboxHarmonizer(weights_path=weights_path)
    elif backend == 'pctnet':
        return PCTNetHarmonizer(weights_path=weights_path, **kwargs)
    elif backend == 'scene_b':
        scene_b_dir = kwargs.get('scene_b_dir')
        if not scene_b_dir:
            raise ValueError("scene_b backend requires scene_b_dir=<path>")
        return SceneBHarmonizer(scene_b_dir)
    else:
        raise ValueError(f"Unknown harmonizer backend: {backend!r}. "
                         f"Choose from: 'whitebox', 'pctnet', 'scene_b'")
